import Mathlib

namespace max_reached_at_2001_l222_222479

noncomputable def a (n : ℕ) : ℝ := n^2 / 1.001^n

theorem max_reached_at_2001 : ∀ n : ℕ, a 2001 ≥ a n := 
sorry

end max_reached_at_2001_l222_222479


namespace rationalize_denominator_l222_222084

theorem rationalize_denominator 
  (cbrt32_eq_2cbrt4 : (32:ℝ)^(1/3) = 2 * (4:ℝ)^(1/3))
  (cbrt16_eq_2cbrt2 : (16:ℝ)^(1/3) = 2 * (2:ℝ)^(1/3))
  (cbrt64_eq_4 : (64:ℝ)^(1/3) = 4) :
  1 / ((4:ℝ)^(1/3) + (32:ℝ)^(1/3)) = ((2:ℝ)^(1/3)) / 6 :=
  sorry

end rationalize_denominator_l222_222084


namespace unique_sequence_l222_222734

theorem unique_sequence (a : ℕ → ℤ) :
  (∀ n : ℕ, a (n + 1) ^ 2 = 1 + (n + 2021) * a n) →
  (∀ n : ℕ, a n = n + 2019) :=
by
  sorry

end unique_sequence_l222_222734


namespace min_value_of_angle_function_l222_222777

theorem min_value_of_angle_function (α β γ : ℝ) (h1 : α + β + γ = Real.pi) (h2 : 0 < α) (h3 : α < Real.pi) :
  ∃ α, α = (2 * Real.pi / 3) ∧ (4 / α + 1 / (Real.pi - α)) = (9 / Real.pi) := by
  sorry

end min_value_of_angle_function_l222_222777


namespace increase_in_rectangle_area_l222_222703

theorem increase_in_rectangle_area (L B : ℝ) :
  let L' := 1.11 * L
  let B' := 1.22 * B
  let original_area := L * B
  let new_area := L' * B'
  let area_increase := new_area - original_area
  let percentage_increase := (area_increase / original_area) * 100
  percentage_increase = 35.42 :=
by
  sorry

end increase_in_rectangle_area_l222_222703


namespace number_of_nephews_l222_222458

def total_jellybeans : ℕ := 70
def jellybeans_per_child : ℕ := 14
def number_of_nieces : ℕ := 2

theorem number_of_nephews : total_jellybeans / jellybeans_per_child - number_of_nieces = 3 := by
  sorry

end number_of_nephews_l222_222458


namespace pizza_volume_one_piece_l222_222444

theorem pizza_volume_one_piece :
  ∀ (h t: ℝ) (d: ℝ) (n: ℕ), d = 16 → t = 1/2 → n = 8 → h = 8 → 
  ( (π * (d / 2)^2 * t) / n = 4 * π ) :=
by 
  intros h t d n hd ht hn hh
  sorry

end pizza_volume_one_piece_l222_222444


namespace twelve_integers_divisible_by_eleven_l222_222619

theorem twelve_integers_divisible_by_eleven (a : Fin 12 → ℤ) : 
  ∃ (i j : Fin 12), i ≠ j ∧ 11 ∣ (a i - a j) :=
by
  sorry

end twelve_integers_divisible_by_eleven_l222_222619


namespace essay_count_problem_l222_222353

noncomputable def eighth_essays : ℕ := sorry
noncomputable def seventh_essays : ℕ := sorry

theorem essay_count_problem (x : ℕ) (h1 : eighth_essays = x) (h2 : seventh_essays = (1/2 : ℚ) * x - 2) (h3 : eighth_essays + seventh_essays = 118) : 
  seventh_essays = 38 :=
sorry

end essay_count_problem_l222_222353


namespace quadratic_function_a_value_l222_222129

theorem quadratic_function_a_value (a : ℝ) (h₁ : a ≠ 1) :
  (∀ x : ℝ, ∃ c₀ c₁ c₂ : ℝ, (a-1) * x^(a^2 + 1) + 2 * x + 3 = c₂ * x^2 + c₁ * x + c₀) → a = -1 :=
by
  sorry

end quadratic_function_a_value_l222_222129


namespace proof_problem_l222_222298

theorem proof_problem (a b c : ℤ) (h1 : a > 2) (h2 : b < 10) (h3 : c ≥ 0) (h4 : 32 = a + 2 * b + 3 * c) : 
  a = 4 ∧ b = 8 ∧ c = 4 :=
by
  sorry

end proof_problem_l222_222298


namespace balloon_volume_safety_l222_222143

theorem balloon_volume_safety (p V : ℝ) (h_prop : p = 90 / V) (h_burst : p ≤ 150) : 0.6 ≤ V :=
by {
  sorry
}

end balloon_volume_safety_l222_222143


namespace pizza_volume_one_piece_l222_222446

theorem pizza_volume_one_piece
  (thickness : ℝ)
  (diameter : ℝ)
  (pieces : ℝ)
  (h : thickness = 1/2)
  (d : diameter = 16)
  (p : pieces = 8) :
  ∃ (volume_one_piece : ℝ), volume_one_piece = 4 * Real.pi :=
by 
  rcases (pi * (d / 2) ^ 2 * h) / p with v;
  use v;
  sorry

end pizza_volume_one_piece_l222_222446


namespace merchant_marked_price_l222_222967

-- Given conditions: 30% discount on list price, 10% discount on marked price, 25% profit on selling price
variable (L : ℝ) -- List price
variable (C : ℝ) -- Cost price after discount: C = 0.7 * L
variable (M : ℝ) -- Marked price
variable (S : ℝ) -- Selling price after discount on marked price: S = 0.9 * M

noncomputable def proof_problem : Prop :=
  C = 0.7 * L ∧
  C = 0.75 * S ∧
  S = 0.9 * M ∧
  M = 103.7 / 100 * L

theorem merchant_marked_price (L : ℝ) (C : ℝ) (S : ℝ) (M : ℝ) :
  (C = 0.7 * L) → 
  (C = 0.75 * S) → 
  (S = 0.9 * M) → 
  M = 103.7 / 100 * L :=
by
  sorry

end merchant_marked_price_l222_222967


namespace range_of_a_if_solution_non_empty_l222_222738

variable (f : ℝ → ℝ) (a : ℝ)

/-- Given that the solution set of f(x) < | -1 | is non-empty,
    we need to prove that |a| ≥ 4. -/
theorem range_of_a_if_solution_non_empty (h : ∃ x, f x < 1) : |a| ≥ 4 :=
sorry

end range_of_a_if_solution_non_empty_l222_222738


namespace series_sum_l222_222655

variable {c d : ℝ}

theorem series_sum (h : ∑' n : ℕ, c / d ^ ((3 : ℝ) ^ n) = 9) :
  ∑' n : ℕ, c / (c + 2 * d) ^ (n + 1) = 9 / 11 :=
by
  -- The code that follows will include the steps and proof to reach the conclusion
  sorry

end series_sum_l222_222655


namespace smallest_possible_n_l222_222472

theorem smallest_possible_n (n : ℕ) (h : ∃ k : ℕ, 15 * n - 2 = 11 * k) : n % 11 = 6 :=
by
  sorry

end smallest_possible_n_l222_222472


namespace division_value_l222_222638

theorem division_value (a b c : ℝ) 
  (h1 : a / b = 5 / 3) 
  (h2 : b / c = 7 / 2) : 
  c / a = 6 / 35 := 
by
  sorry

end division_value_l222_222638


namespace eighty_first_number_in_set_l222_222137

theorem eighty_first_number_in_set : ∃ n : ℕ, n = 81 ∧ ∀ k : ℕ, (k = 8 * (n - 1) + 5) → k = 645 := by
  sorry

end eighty_first_number_in_set_l222_222137


namespace arithmetic_seq_a12_l222_222643

-- Define an arithmetic sequence
def arithmetic_seq (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

-- Prove that a_12 = 12 given the conditions
theorem arithmetic_seq_a12 :
  ∃ a₁, (arithmetic_seq a₁ 2 2 = -8) → (arithmetic_seq a₁ 2 12 = 12) :=
by
  sorry

end arithmetic_seq_a12_l222_222643


namespace min_value_correct_l222_222042

noncomputable def min_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : ℝ :=
  Real.sqrt ((a^2 + 2 * b^2) * (4 * a^2 + b^2)) / (a * b)

theorem min_value_correct (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  min_value a b ha hb ≥ 3 :=
sorry

end min_value_correct_l222_222042


namespace change_in_max_value_l222_222286

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem change_in_max_value (a b c : ℝ) (h1 : -b^2 / (4 * (a + 1)) + c = -b^2 / (4 * a) + c + 27 / 2)
  (h2 : -b^2 / (4 * (a - 4)) + c = -b^2 / (4 * a) + c - 9) :
  -b^2 / (4 * (a - 2)) + c = -b^2 / (4 * a) + c - 27 / 4 :=
by
  sorry

end change_in_max_value_l222_222286


namespace smallest_integer_ends_in_3_divisible_by_11_correct_l222_222112

def ends_in_3 (n : ℕ) : Prop :=
  n % 10 = 3

def divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def smallest_ends_in_3_divisible_by_11 : ℕ :=
  33

theorem smallest_integer_ends_in_3_divisible_by_11_correct :
  smallest_ends_in_3_divisible_by_11 = 33 ∧ ends_in_3 smallest_ends_in_3_divisible_by_11 ∧ divisible_by_11 smallest_ends_in_3_divisible_by_11 := 
by
  sorry

end smallest_integer_ends_in_3_divisible_by_11_correct_l222_222112


namespace average_student_headcount_l222_222870

theorem average_student_headcount :
  let count_0304 := 10500
  let count_0405 := 10700
  let count_0506 := 11300
  let total_count := count_0304 + count_0405 + count_0506
  let number_of_terms := 3
  let average := total_count / number_of_terms
  average = 10833 :=
by
  sorry

end average_student_headcount_l222_222870


namespace set_intersection_complement_l222_222760

open Set

def I := {n : ℕ | True}
def A := {x ∈ I | 2 ≤ x ∧ x ≤ 10}
def B := {x | Nat.Prime x}

theorem set_intersection_complement :
  A ∩ (I \ B) = {4, 6, 8, 9, 10} := by
  sorry

end set_intersection_complement_l222_222760


namespace distance_covered_at_40_kmph_l222_222300

theorem distance_covered_at_40_kmph (x : ℝ) : 
  (x / 40 + (250 - x) / 60 = 5.4) → (x = 148) :=
by
  intro h
  sorry

end distance_covered_at_40_kmph_l222_222300


namespace value_of_p_h_3_l222_222212

-- Define the functions h and p
def h (x : ℝ) : ℝ := 4 * x + 5
def p (x : ℝ) : ℝ := 6 * x - 11

-- Statement to prove
theorem value_of_p_h_3 : p (h 3) = 91 := sorry

end value_of_p_h_3_l222_222212


namespace single_ticket_cost_l222_222095

/-- Define the conditions: sales total, attendee count, number of couple tickets, and cost of couple tickets. -/
def total_sales : ℤ := 2280
def total_attendees : ℕ := 128
def couple_tickets_sold : ℕ := 16
def cost_of_couple_ticket : ℤ := 35

/-- Define the derived conditions: people covered by couple tickets, single tickets sold, and sales from couple tickets. -/
def people_covered_by_couple_tickets : ℕ := couple_tickets_sold * 2
def single_tickets_sold : ℕ := total_attendees - people_covered_by_couple_tickets
def sales_from_couple_tickets : ℤ := couple_tickets_sold * cost_of_couple_ticket

/-- Define the core equation that ties single ticket sales to the total sales. -/
def core_equation (x : ℤ) : Bool := 
  sales_from_couple_tickets + single_tickets_sold * x = total_sales

-- Finally, the statement that needs to be proved.
theorem single_ticket_cost :
  ∃ x : ℤ, core_equation x ∧ x = 18 := by
  sorry

end single_ticket_cost_l222_222095


namespace simplify_fraction_l222_222953

theorem simplify_fraction : 
    (3 ^ 1011 + 3 ^ 1009) / (3 ^ 1011 - 3 ^ 1009) = 5 / 4 := 
by
  sorry

end simplify_fraction_l222_222953


namespace tangent_addition_l222_222371

theorem tangent_addition (y : ℝ) (h : Real.tan y = -1) : Real.tan (y + Real.pi / 3) = -1 :=
sorry

end tangent_addition_l222_222371


namespace remainder_squared_mod_five_l222_222704

theorem remainder_squared_mod_five (n k : ℤ) (h : n = 5 * k + 3) : ((n - 1) ^ 2) % 5 = 4 :=
by
  sorry

end remainder_squared_mod_five_l222_222704


namespace solve_f_1991_2_1990_l222_222996

-- Define the sum of digits function for an integer k
def sum_of_digits (k : ℕ) : ℕ := k.digits 10 |>.sum

-- Define f1(k) as the square of the sum of digits of k
def f1 (k : ℕ) : ℕ := (sum_of_digits k) ^ 2

-- Define the recursive sequence fn as given in the problem
def fn : ℕ → ℕ → ℕ
| 0, k => k
| n + 1, k => f1 (fn n k)

-- Define the specific problem statement
theorem solve_f_1991_2_1990 : fn 1991 (2 ^ 1990) = 4 := sorry

end solve_f_1991_2_1990_l222_222996


namespace sequence_a4_value_l222_222024

theorem sequence_a4_value :
  ∀ {a : ℕ → ℚ}, (a 1 = 3) → ((∀ n, a (n + 1) = 3 * a n / (a n + 3))) → (a 4 = 3 / 4) :=
by
  intros a h1 hRec
  sorry

end sequence_a4_value_l222_222024


namespace total_fence_length_l222_222877

variable (Darren Doug : ℝ)

-- Definitions based on given conditions
def Darren_paints_more := Darren = 1.20 * Doug
def Darren_paints_360 := Darren = 360

-- The statement to prove
theorem total_fence_length (h1 : Darren_paints_more Darren Doug) (h2 : Darren_paints_360 Darren) : (Darren + Doug) = 660 := 
by
  sorry

end total_fence_length_l222_222877


namespace symmetric_point_min_value_l222_222218

theorem symmetric_point_min_value (a b : ℝ) 
  (h1 : a > 0 ∧ b > 0) 
  (h2 : ∃ (x₀ y₀ : ℝ), x₀ + y₀ - 2 = 0 ∧ 2 * x₀ + y₀ + 3 = 0 ∧ 
        a + b = x₀ + y₀ ∧ ∃ k, k = (y₀ - b) / (x₀ - a) ∧ y₀ = k * x₀ + 2 - k * (a + k * b))
   : ∃ α β, a = β / α ∧  b = 2 * β / α ∧ (1 / a + 8 / b) = 25 / 9 :=
sorry

end symmetric_point_min_value_l222_222218


namespace two_digit_numbers_equal_three_times_product_of_digits_l222_222176

theorem two_digit_numbers_equal_three_times_product_of_digits :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ ∃ a b : ℕ, n = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 3 * a * b} = {15, 24} :=
by
  sorry

end two_digit_numbers_equal_three_times_product_of_digits_l222_222176


namespace sum_of_cosines_dihedral_angles_l222_222270

-- Define the conditions of the problem
def sum_of_plane_angles_trihederal (α β γ : ℝ) : Prop :=
  α + β + γ = 180

-- Define the problem statement
theorem sum_of_cosines_dihedral_angles (α β γ : ℝ) (d1 d2 d3 : ℝ)
  (h : sum_of_plane_angles_trihederal α β γ) : 
  d1 + d2 + d3 = 1 :=
  sorry

end sum_of_cosines_dihedral_angles_l222_222270


namespace Theorem3_l222_222630

theorem Theorem3 {f g : ℝ → ℝ} (T1_eq_1 : ∀ x, f (x + 1) = f x)
  (m : ℕ) (h_g_periodic : ∀ x, g (x + 1 / m) = g x) (hm : m > 1) :
  ∃ k : ℕ, k > 0 ∧ (k = 1 ∨ (k ≠ m ∧ ¬(m % k = 0))) ∧ 
    (∀ x, (f x + g x) = (f (x + 1 / k) + g (x + 1 / k))) := 
sorry

end Theorem3_l222_222630


namespace jellybeans_needed_l222_222032

-- Define the initial conditions as constants
def jellybeans_per_large_glass := 50
def jellybeans_per_small_glass := jellybeans_per_large_glass / 2
def number_of_large_glasses := 5
def number_of_small_glasses := 3

-- Calculate the total number of jellybeans needed
def total_jellybeans : ℕ :=
  (number_of_large_glasses * jellybeans_per_large_glass) + 
  (number_of_small_glasses * jellybeans_per_small_glass)

-- Prove that the total number of jellybeans needed is 325
theorem jellybeans_needed : total_jellybeans = 325 :=
sorry

end jellybeans_needed_l222_222032


namespace ratio_of_side_lengths_l222_222714

theorem ratio_of_side_lengths (w1 w2 : ℝ) (s1 s2 : ℝ)
  (h1 : w1 = 8) (h2 : w2 = 64)
  (v1 : w1 = s1 ^ 3)
  (v2 : w2 = s2 ^ 3) : 
  s2 / s1 = 2 := by
  sorry

end ratio_of_side_lengths_l222_222714


namespace range_of_g_l222_222350

noncomputable def g (x : ℝ) : ℝ :=
  (Real.arccos (x / 3))^2 + 2 * Real.pi * Real.arcsin (x / 3) -
  (Real.arcsin (x / 3))^2 + (Real.pi^2 / 18) * (x^2 + 12 * x + 27)

lemma arccos_arcsin_identity (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) : 
  Real.arccos x + Real.arcsin x = Real.pi / 2 := sorry

theorem range_of_g : ∀ (x : ℝ), -3 ≤ x ∧ x ≤ 3 → ∃ y : ℝ, g x = y ∧ y ∈ Set.Icc (Real.pi^2 / 4) (5 * Real.pi^2 / 2) :=
sorry

end range_of_g_l222_222350


namespace rotation_problem_l222_222462

theorem rotation_problem (y : ℝ) (hy : y < 360) :
  (450 % 360 == 90) ∧ (y == 360 - 90) ∧ (90 + (360 - y) % 360 == 0) → y == 270 :=
by {
  -- Proof steps go here
  sorry
}

end rotation_problem_l222_222462


namespace mean_three_numbers_l222_222541

open BigOperators

theorem mean_three_numbers (a b c : ℝ) (s : Finset ℝ) (h₀ : s.card = 20)
  (h₁ : (∑ x in s, x) / 20 = 45) 
  (h₂ : (∑ x in s ∪ {a, b, c}, x) / 23 = 50) : 
  (a + b + c) / 3 = 250 / 3 :=
by
  sorry

end mean_three_numbers_l222_222541


namespace smallest_non_factor_product_l222_222097

theorem smallest_non_factor_product (a b : ℕ) (h1 : a ≠ b) (h2 : a ∣ 48) (h3 : b ∣ 48) (h4 : ¬ (a * b ∣ 48)) : a * b = 18 :=
by
  -- proof intentionally omitted
  sorry

end smallest_non_factor_product_l222_222097


namespace sixth_graders_more_than_seventh_l222_222585

theorem sixth_graders_more_than_seventh (c_pencil : ℕ) (h_cents : c_pencil > 0)
    (h_cond : ∀ n : ℕ, n * c_pencil = 221 ∨ n * c_pencil = 286)
    (h_sixth_graders : 35 > 0) :
    ∃ n6 n7 : ℕ, n6 > n7 ∧ n6 - n7 = 5 :=
by
  sorry

end sixth_graders_more_than_seventh_l222_222585


namespace elizabeth_initial_bottles_l222_222337

theorem elizabeth_initial_bottles (B : ℕ) (H1 : B - 2 - 1 = (3 * X) → 3 * (B - 3) = 21) : B = 10 :=
by
  sorry

end elizabeth_initial_bottles_l222_222337


namespace find_x_l222_222213

-- Definitions from the conditions
def isPositiveMultipleOf7 (x : ℕ) : Prop := ∃ k : ℕ, x = 7 * k ∧ x > 0
def xSquaredGreaterThan150 (x : ℕ) : Prop := x^2 > 150
def xLessThan40 (x : ℕ) : Prop := x < 40

-- Main problem statement
theorem find_x (x : ℕ) (h1 : isPositiveMultipleOf7 x) (h2 : xSquaredGreaterThan150 x) (h3 : xLessThan40 x) : x = 14 :=
sorry

end find_x_l222_222213


namespace club_members_after_four_years_l222_222373

theorem club_members_after_four_years
  (b : ℕ → ℕ)
  (h_initial : b 0 = 20)
  (h_recursive : ∀ k, b (k + 1) = 3 * (b k) - 10) :
  b 4 = 1220 :=
sorry

end club_members_after_four_years_l222_222373


namespace acuteAngleAt725_l222_222950

noncomputable def hourHandPosition (h : ℝ) (m : ℝ) : ℝ :=
  h * 30 + m / 60 * 30

noncomputable def minuteHandPosition (m : ℝ) : ℝ :=
  m / 60 * 360

noncomputable def angleBetweenHands (h m : ℝ) : ℝ :=
  abs (hourHandPosition h m - minuteHandPosition m)

theorem acuteAngleAt725 : angleBetweenHands 7 25 = 72.5 :=
  sorry

end acuteAngleAt725_l222_222950


namespace min_expression_value_l222_222623

theorem min_expression_value (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + 2 * b = 1) : 
  ∃ x, (x = (a^2 + 1) / a + (2 * b^2 + 1) / b) ∧ x = 4 + 2 * Real.sqrt 2 :=
by
  sorry

end min_expression_value_l222_222623


namespace servings_required_l222_222722

/-- Each serving of cereal is 2.0 cups, and 36 cups are needed. Prove that the number of servings required is 18. -/
theorem servings_required (cups_per_serving : ℝ) (total_cups : ℝ) (h1 : cups_per_serving = 2.0) (h2 : total_cups = 36.0) :
  total_cups / cups_per_serving = 18 :=
by
  sorry

end servings_required_l222_222722


namespace smallest_int_ends_in_3_div_by_11_l222_222125

theorem smallest_int_ends_in_3_div_by_11 :
  ∃ k : ℕ, k > 0 ∧ k % 10 = 3 ∧ k % 11 = 0 ∧ k = 33 :=
by {
  sorry
}

end smallest_int_ends_in_3_div_by_11_l222_222125


namespace find_fourth_number_l222_222060

def nat_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, n ≥ 3 → a n = a (n - 1) + a (n - 2)

variable {a : ℕ → ℕ}

theorem find_fourth_number (h_seq : nat_sequence a) (h7 : a 7 = 42) (h9 : a 9 = 110) : a 4 = 10 :=
by
  -- Placeholder for proof steps
  sorry

end find_fourth_number_l222_222060


namespace minimum_value_condition_l222_222216

-- Define the function y = x^3 - 2ax + a
noncomputable def f (a x : ℝ) : ℝ := x^3 - 2 * a * x + a

-- Define its derivative
noncomputable def f' (a x : ℝ) : ℝ := 3 * x^2 - 2 * a

-- Define the lean theorem statement
theorem minimum_value_condition (a : ℝ) : 
  (∃ x y : ℝ, 0 < x ∧ x < 1 ∧ y = f a x ∧ (∀ z : ℝ, 0 < z ∧ z < 1 → f a z ≥ y)) ∧
  ¬(∃ x y : ℝ, 0 < x ∧ x < 1 ∧ y = f a x ∧ (∀ z : ℝ, 0 < z ∧ z < 1 → f a z < y)) 
  ↔ 0 < a ∧ a < 3 / 2 :=
sorry

end minimum_value_condition_l222_222216


namespace find_m_l222_222220

theorem find_m (m : ℝ) : 
  (∃ α β : ℝ, (α + β = 2 * (m + 1)) ∧ (α * β = m + 4) ∧ ((1 / α) + (1 / β) = 1)) → m = 2 :=
by
  sorry

end find_m_l222_222220


namespace wrapping_paper_area_l222_222437

theorem wrapping_paper_area (l w h : ℝ) (hlw : l > w) (hwh : w > h) (hl : l = 2 * w) : 
    (∃ a : ℝ, a = 5 * w^2 + h^2) :=
by 
  sorry

end wrapping_paper_area_l222_222437


namespace probability_diff_greater_than_one_eq_seven_eighths_l222_222534

noncomputable def prob_greater_than_one : ℝ :=
  let heads_p := 1 / 2
  let tails_p := 1 / 2
  let uniform_dist := @MeasureTheory.Measure.uniform _ _ LinearOrder.IntervalOrdering.NonnegOrderedAddCommGroup.interval ∅ set.Icc 0 2
  let coin_flip (p : ℝ) := (ennreal.of_real p) * uniform_dist

  -- Event: Both flips are tails; both numbers chosen uniformly from [0,2]
  let case1 := (tails_p) ^ 2 * ∫ x in 0..2, ∫ y in 0..2, if (x - y > 1) then 1 else 0

  -- Event: First flip tails for y, second flip heads-tails (prob chosen 2)
  let case2 := (tails_p) * heads_p * (∫ y in 0..2, if (2 - y > 1) then 1 else 0)

  -- Event: First flip tails for x, second flip heads-heads (prob chosen 0)
  let case3 := (heads_p * tails_p) * (∫ x in 0..2, if (x - 0 > 1) then 1 else 0)

  -- Event: First flip heads-tails for both x and y; chose 2 and 0
  let case4 := (heads_p) ^ 2

  -- Sum of all cases for P(x - y > 1)
  let prob_xy_diff_gt_1 := case1 + case2 + case3 + case4
  -- P(|x - y| > 1) is twice that due to symmetry
  2 * prob_xy_diff_gt_1

theorem probability_diff_greater_than_one_eq_seven_eighths :
  prob_greater_than_one = 7 / 8 :=
sorry

end probability_diff_greater_than_one_eq_seven_eighths_l222_222534


namespace intersection_points_circle_l222_222189

-- Defining the two lines based on the parameter u
def line1 (u : ℝ) (x y : ℝ) : Prop := 2 * u * x - 3 * y - 2 * u = 0
def line2 (u : ℝ) (x y : ℝ) : Prop := x - 3 * u * y + 2 = 0

-- Proof statement that shows the intersection points lie on a circle
theorem intersection_points_circle (u x y : ℝ) :
  line1 u x y → line2 u x y → (x - 1)^2 + y^2 = 1 :=
by {
  -- This completes the proof statement, but leaves implementation as exercise
  sorry
}

end intersection_points_circle_l222_222189


namespace roots_situation_depends_on_k_l222_222411

theorem roots_situation_depends_on_k (k : ℝ) : 
  let a := 1
  let b := -3
  let c := 2 - k
  let Δ := b^2 - 4 * a * c
  (Δ > 0) ∨ (Δ = 0) ∨ (Δ < 0) :=
by
  intros
  sorry

end roots_situation_depends_on_k_l222_222411


namespace total_surface_area_correct_l222_222572

-- Definitions for side lengths of the cubes
def side_length_large := 5
def side_length_medium := 2
def side_length_small := 1

-- Surface area calculation for a single cube
def surface_area (side_length : ℕ) : ℕ := 6 * side_length^2

-- Surface areas for each size of the cube
def surface_area_large := surface_area side_length_large
def surface_area_medium := surface_area side_length_medium
def surface_area_small := surface_area side_length_small

-- Total surface areas for medium and small cubes
def surface_area_medium_total := 4 * surface_area_medium
def surface_area_small_total := 4 * surface_area_small

-- Total surface area of the structure
def total_surface_area := surface_area_large + surface_area_medium_total + surface_area_small_total

-- Expected result
def expected_surface_area := 270

-- Proof statement
theorem total_surface_area_correct : total_surface_area = expected_surface_area := by
  sorry

end total_surface_area_correct_l222_222572


namespace smallest_non_factor_l222_222104

-- Definitions of the conditions
def isFactorOf (m n : ℕ) : Prop := n % m = 0
def distinct (a b : ℕ) : Prop := a ≠ b

-- The main statement we need to prove.
theorem smallest_non_factor (a b : ℕ) (h_distinct : distinct a b)
  (h_a_factor : isFactorOf a 48) (h_b_factor : isFactorOf b 48)
  (h_not_factor : ¬ isFactorOf (a * b) 48) :
  a * b = 32 := 
sorry

end smallest_non_factor_l222_222104


namespace sum_of_real_roots_of_even_function_l222_222748

noncomputable def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem sum_of_real_roots_of_even_function (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_intersects : ∃ a b c d, f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ d) :
  a + b + c + d = 0 :=
sorry

end sum_of_real_roots_of_even_function_l222_222748


namespace retail_price_eq_120_l222_222434

noncomputable def retail_price : ℝ :=
  let W := 90
  let P := 0.20 * W
  let SP := W + P
  SP / 0.90

theorem retail_price_eq_120 : retail_price = 120 := by
  sorry

end retail_price_eq_120_l222_222434


namespace price_increase_ratio_l222_222500

theorem price_increase_ratio 
  (c : ℝ)
  (h1 : 351 = c * 1.30) :
  (c + 351) / c = 2.3 :=
sorry

end price_increase_ratio_l222_222500


namespace pq_false_implies_m_range_l222_222594

def p : Prop := ∀ x : ℝ, abs x + x ≥ 0

def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 + m * x + 1 = 0

theorem pq_false_implies_m_range (m : ℝ) :
  (¬ (p ∧ q m)) → -2 < m ∧ m < 2 :=
by
  sorry

end pq_false_implies_m_range_l222_222594


namespace remainder_of_95_times_97_div_12_l222_222557

theorem remainder_of_95_times_97_div_12 : 
  (95 * 97) % 12 = 11 := by
  sorry

end remainder_of_95_times_97_div_12_l222_222557


namespace petri_dish_count_l222_222509

theorem petri_dish_count (total_germs : ℝ) (germs_per_dish : ℝ) (h1 : total_germs = 0.036 * 10^5) (h2 : germs_per_dish = 199.99999999999997) :
  total_germs / germs_per_dish = 18 :=
by
  sorry

end petri_dish_count_l222_222509


namespace smallest_positive_period_tan_l222_222818

noncomputable def max_value (a b x : ℝ) := b + a * Real.sin x = -1
noncomputable def min_value (a b x : ℝ) := b - a * Real.sin x = -5
noncomputable def a_negative (a : ℝ) := a < 0

theorem smallest_positive_period_tan :
  ∃ (a b : ℝ), (max_value a b 0) ∧ (min_value a b 0) ∧ (a_negative a) →
  (1 / |3 * a + b|) * Real.pi = Real.pi / 9 :=
by
  sorry

end smallest_positive_period_tan_l222_222818


namespace variance_remaining_scores_l222_222372

noncomputable def scores : List ℝ := [90, 89, 90, 95, 93, 94, 93]

def remaining_scores (s : List ℝ) : List ℝ :=
s.erase 95 |>.erase 89

def mean (l : List ℝ) : ℝ :=
(l.sum / l.length)

def variance (l : List ℝ) : ℝ :=
let m := mean l in
(sum (l.map (λ x, (x - m) ^ 2)) / l.length)

theorem variance_remaining_scores :
  variance (remaining_scores scores) = 2.8 :=
sorry

end variance_remaining_scores_l222_222372


namespace probability_of_rolling_one_five_times_and_two_once_in_seven_rolls_l222_222214

noncomputable def probability_roll_event : ℚ :=
  let p_one := 1 / 6 in
  let p_two := 1 / 6 in
  let p_other := 2 / 3 in
  let comb := Nat.choose 7 5 * Nat.choose 2 1 in
  comb * p_one^5 * p_two * p_other

theorem probability_of_rolling_one_five_times_and_two_once_in_seven_rolls :
  probability_roll_event = 1 / 417 := 
sorry

end probability_of_rolling_one_five_times_and_two_once_in_seven_rolls_l222_222214


namespace determine_m_l222_222984

variable {x y z : ℝ}

theorem determine_m (h : (5 / (x + y)) = (m / (x + z)) ∧ (m / (x + z)) = (13 / (z - y))) : m = 18 :=
by
  sorry

end determine_m_l222_222984


namespace no_consecutive_days_played_l222_222141

theorem no_consecutive_days_played (john_interval mary_interval : ℕ) :
  john_interval = 16 ∧ mary_interval = 25 → 
  ¬ ∃ (n : ℕ), (n * john_interval + 1 = m * mary_interval ∨ n * john_interval = m * mary_interval + 1) :=
by
  sorry

end no_consecutive_days_played_l222_222141


namespace compound_interest_comparison_l222_222217

theorem compound_interest_comparison :
  (1 + 0.04) < (1 + 0.04 / 12) ^ 12 := sorry

end compound_interest_comparison_l222_222217


namespace scientific_notation_of_number_l222_222525

theorem scientific_notation_of_number :
  1214000 = 1.214 * 10^6 :=
by
  sorry

end scientific_notation_of_number_l222_222525


namespace sum_of_distances_from_circumcenter_to_sides_l222_222207

theorem sum_of_distances_from_circumcenter_to_sides :
  let r1 := 3
  let r2 := 5
  let r3 := 7
  let a := r1 + r2
  let b := r1 + r3
  let c := r2 + r3
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let r_incircle := area / s
  r_incircle = Real.sqrt 7 →
  let sum_distances := (7 / 4) + (7 / (3 * Real.sqrt 6)) + (7 / (Real.sqrt 30))
  sum_distances = (7 / 4) + (7 / (3 * Real.sqrt 6)) + (7 / (Real.sqrt 30))
:= sorry

end sum_of_distances_from_circumcenter_to_sides_l222_222207


namespace calculate_expression_l222_222590

variables (a b : ℝ)

theorem calculate_expression : -a^2 * 2 * a^4 * b = -2 * (a^6) * b :=
by
  sorry

end calculate_expression_l222_222590


namespace simplify_expression_l222_222912

variable (a b : ℝ)
variable (h1 : 0 < a)
variable (h2 : a ≠ 1)
variable (h3 : 0 < b)

theorem simplify_expression : a ^ Real.log (1 / b ^ Real.log a) = 1 / b ^ (Real.log a) ^ 2 :=
by
  sorry

end simplify_expression_l222_222912


namespace two_cos_45_eq_sqrt_two_l222_222415

theorem two_cos_45_eq_sqrt_two
  (h1 : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2) :
  2 * Real.cos (Real.pi / 4) = Real.sqrt 2 :=
sorry

end two_cos_45_eq_sqrt_two_l222_222415


namespace circle_center_l222_222179

theorem circle_center (x y : ℝ) :
  4 * x^2 - 16 * x + 4 * y^2 + 8 * y - 12 = 0 →
  (∃ h k : ℝ, (x - h)^2 + (y - k)^2 = 8 ∧ h = 2 ∧ k = -1) :=
sorry

end circle_center_l222_222179


namespace Emilee_earnings_l222_222384

theorem Emilee_earnings (J R_j T R_t E R_e : ℕ) :
  (R_j * J = 35) → 
  (R_t * T = 30) → 
  (R_j * J + R_t * T + R_e * E = 90) → 
  (R_e * E = 25) :=
by
  intros h1 h2 h3
  sorry

end Emilee_earnings_l222_222384


namespace no_arithmetic_mean_l222_222797

def eight_thirteen : ℚ := 8 / 13
def eleven_seventeen : ℚ := 11 / 17
def five_eight : ℚ := 5 / 8

-- Define the function to calculate the arithmetic mean of two rational numbers
def arithmetic_mean (a b : ℚ) : ℚ :=
(a + b) / 2

-- The theorem statement
theorem no_arithmetic_mean :
  eight_thirteen ≠ arithmetic_mean eleven_seventeen five_eight ∧
  eleven_seventeen ≠ arithmetic_mean eight_thirteen five_eight ∧
  five_eight ≠ arithmetic_mean eight_thirteen eleven_seventeen :=
sorry

end no_arithmetic_mean_l222_222797


namespace analyze_monotonicity_and_find_a_range_l222_222363

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x^2

noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ := Real.exp x - 2 * a * x

theorem analyze_monotonicity_and_find_a_range
  (a : ℝ)
  (h : ∀ x : ℝ, f x a + f_prime x a = 2 - a * x^2) :
  (∀ x : ℝ, a ≤ 0 → f_prime x a > 0) ∧
  (a > 0 → (∀ x : ℝ, (x < Real.log (2 * a) → f_prime x a < 0) ∧ (x > Real.log (2 * a) → f_prime x a > 0))) ∧
  (1 < a ∧ a < Real.exp 1 - 1) :=
sorry

end analyze_monotonicity_and_find_a_range_l222_222363


namespace expected_profit_is_correct_l222_222853

noncomputable def expected_profit : ℝ :=
let profits : List ℝ := [50, 30, -20]
let probabilities : List ℝ := [0.6, 0.3, 0.1]
profits.zip probabilities |>.sum (λ (xp : ℝ × ℝ), xp.1 * xp.2)

theorem expected_profit_is_correct : expected_profit = 37 := by
  sorry

end expected_profit_is_correct_l222_222853


namespace scientific_notation_l222_222222

theorem scientific_notation :
  686530000 = 6.8653 * 10^8 :=
sorry

end scientific_notation_l222_222222


namespace national_flag_length_l222_222812

-- Definitions from the conditions specified in the problem
def width : ℕ := 128
def ratio_length_to_width (L W : ℕ) : Prop := L / W = 3 / 2

-- The main theorem to prove
theorem national_flag_length (L : ℕ) (H : ratio_length_to_width L width) : L = 192 :=
by
  sorry

end national_flag_length_l222_222812


namespace floor_equation_solution_l222_222605

open Int

theorem floor_equation_solution (x : ℝ) :
  (⌊ ⌊ 3 * x ⌋ - 1/2 ⌋ = ⌊ x + 4 ⌋) ↔ (7/3 ≤ x ∧ x < 3) := sorry

end floor_equation_solution_l222_222605


namespace hyperbola_eccentricity_l222_222758

theorem hyperbola_eccentricity (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) 
  (h₂ : 4 * c^2 = 25) (h₃ : a = 1/2) : c/a = 5 :=
by
  sorry

end hyperbola_eccentricity_l222_222758


namespace factorization_l222_222169

theorem factorization (m : ℤ) : m^2 + 3 * m = m * (m + 3) :=
by sorry

end factorization_l222_222169


namespace uniformColorGridPossible_l222_222642

noncomputable def canPaintUniformColor (n : Nat) (G : Matrix (Fin n) (Fin n) (Fin (n - 1))) : Prop :=
  ∀ (row : Fin n), ∃ (c : Fin (n - 1)), ∀ (col : Fin n), G row col = c

theorem uniformColorGridPossible (n : Nat) (G : Matrix (Fin n) (Fin n) (Fin (n - 1))) :
  (∀ r : Fin n, ∃ c₁ c₂ : Fin n, c₁ ≠ c₂ ∧ G r c₁ = G r c₂) ∧
  (∀ c : Fin n, ∃ r₁ r₂ : Fin n, r₁ ≠ r₂ ∧ G r₁ c = G r₂ c) →
  ∃ c : Fin (n - 1), ∀ (row col : Fin n), G row col = c := by
  sorry

end uniformColorGridPossible_l222_222642


namespace two_cos_45_eq_sqrt_2_l222_222414

theorem two_cos_45_eq_sqrt_2 : 2 * Real.cos (pi / 4) = Real.sqrt 2 := by
  sorry

end two_cos_45_eq_sqrt_2_l222_222414


namespace tan_inverse_least_positive_l222_222387

variables (a b x : ℝ)

-- Condition 1: tan(x) = a / (2*b)
def condition1 : Prop := Real.tan x = a / (2 * b)

-- Condition 2: tan(2*x) = 2*b / (a + 2*b)
def condition2 : Prop := Real.tan (2 * x) = (2 * b) / (a + 2 * b)

-- The theorem stating the least positive value of x is arctan(0)
theorem tan_inverse_least_positive (h1 : condition1 a b x) (h2 : condition2 a b x) : ∃ k : ℝ, Real.arctan k = 0 :=
by
  sorry

end tan_inverse_least_positive_l222_222387


namespace frequencies_and_confidence_level_l222_222427

namespace MachineQuality

-- Definitions of the given conditions
def productsA := 200
def firstClassA := 150
def secondClassA := 50

def productsB := 200
def firstClassB := 120
def secondClassB := 80

def totalProducts := productsA + productsB
def totalFirstClass := firstClassA + firstClassB
def totalSecondClass := secondClassA + secondClassB

-- 1. Frequencies of first-class products
def frequencyFirstClassA := firstClassA / productsA
def frequencyFirstClassB := firstClassB / productsB

-- 2. \( K^2 \) calculation
def n := 400
def a := 150
def b := 50
def c := 120
def d := 80

def K_squared := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- The theorem to prove the frequencies and the confidence level
theorem frequencies_and_confidence_level : 
    frequencyFirstClassA = (3 / 4) ∧ frequencyFirstClassB = (3 / 5) ∧ K_squared > 6.635 := 
    by {
        sorry -- Proof steps go here
    }

end MachineQuality

end frequencies_and_confidence_level_l222_222427


namespace fraction_sum_squares_eq_sixteen_l222_222370

variables (x a y b z c : ℝ)

theorem fraction_sum_squares_eq_sixteen
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 0) :
  (x^2 / a^2 + y^2 / b^2 + z^2 / c^2) = 16 := 
sorry

end fraction_sum_squares_eq_sixteen_l222_222370


namespace problem_statement_l222_222768

theorem problem_statement
  (a b c : ℝ)
  (h1 : a + 2 * b + 3 * c = 12)
  (h2 : a^2 + b^2 + c^2 = a * b + a * c + b * c) :
  a + b^2 + c^3 = 14 := 
sorry

end problem_statement_l222_222768


namespace sin_cos_eq_l222_222238

theorem sin_cos_eq (a : ℝ) :
    (∀ x ∈ set.Icc (0:ℝ) (2 * Real.pi), ∀ x1 x2 x3,
    (sin x + Real.sqrt 3 * cos x = a) → (x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3)) ↔ (a = Real.sqrt 3) :=
by
  sorry

end sin_cos_eq_l222_222238


namespace eggs_total_l222_222471

-- Definitions based on conditions
def isPackageSize (n : Nat) : Prop :=
  n = 6 ∨ n = 11

def numLargePacks : Nat := 5

def largePackSize : Nat := 11

-- Mathematical statement to prove
theorem eggs_total : ∃ totalEggs : Nat, totalEggs = numLargePacks * largePackSize :=
  by sorry

end eggs_total_l222_222471


namespace max_value_change_l222_222287

open Real

variables (f : ℝ → ℝ) (x : ℝ)

-- Conditions
def condition1 : Prop := ∀ f, (∃ M1 : ℝ, ∀ x : ℝ, f(x) ≤ M1 ∧ ∃ a, a + x ^ 2 = f ⟹ f(M1 + x ^ 2) - M1 = 27 / 2)
def condition2 : Prop := ∀ f, (∃ M2 : ℝ, ∀ x : ℝ, f(x) ≤ M2 ∧ ∃ b, b - 4 * x ^ 2 = f ⟹ f(M2 - 4 * x ^ 2) - M2 = -9)

-- Statement to prove
theorem max_value_change (f : ℝ → ℝ) 
  (h1 : condition1 f) 
  (h2 : condition2 f) : 
  ∃ C : ℝ, ∀ x : ℝ, C = - 27 / 4 ∧ ∃ c, c - 2 * x ^ 2 = f ⟹ f (C - 2 * x ^ 2) = f C :=
sorry

end max_value_change_l222_222287


namespace sandwiches_difference_l222_222530

-- Define the number of sandwiches Samson ate at lunch on Monday
def sandwichesLunchMonday : ℕ := 3

-- Define the number of sandwiches Samson ate at dinner on Monday (twice as many as lunch)
def sandwichesDinnerMonday : ℕ := 2 * sandwichesLunchMonday

-- Define the total number of sandwiches Samson ate on Monday
def totalSandwichesMonday : ℕ := sandwichesLunchMonday + sandwichesDinnerMonday

-- Define the number of sandwiches Samson ate for breakfast on Tuesday
def sandwichesBreakfastTuesday : ℕ := 1

-- Define the total number of sandwiches Samson ate on Tuesday
def totalSandwichesTuesday : ℕ := sandwichesBreakfastTuesday

-- Define the number of more sandwiches Samson ate on Monday than on Tuesday
theorem sandwiches_difference : totalSandwichesMonday - totalSandwichesTuesday = 8 :=
by
  sorry

end sandwiches_difference_l222_222530


namespace max_correct_answers_l222_222154

theorem max_correct_answers (c w b : ℕ) (h1 : c + w + b = 25) (h2 : 6 * c - 3 * w = 60) : c ≤ 15 :=
by {
  sorry
}

end max_correct_answers_l222_222154


namespace minimum_product_xyz_l222_222236

noncomputable def minimalProduct (x y z : ℝ) : ℝ :=
  3 * x^2 * (1 - 4 * x)

theorem minimum_product_xyz :
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 →
  x + y + z = 1 →
  z = 3 * x →
  x ≤ y ∧ y ≤ z →
  minimalProduct x y z = (9 / 343) :=
by
  intros x y z x_pos y_pos z_pos sum_eq1 z_eq3x inequalities
  sorry

end minimum_product_xyz_l222_222236


namespace part1_part2_l222_222365

noncomputable def f (a x : ℝ) : ℝ := a - 1/x - Real.log x

theorem part1 (a : ℝ) :
  a = 2 → ∃ m b : ℝ, (∀ x : ℝ, f a x = x * m + b) ∧ (∀ y : ℝ, f a 1 = y → b = y ∧ m = 0) :=
by
  sorry

theorem part2 (a : ℝ) :
  (∃! x : ℝ, f a x = 0) → a = 1 :=
by
  sorry

end part1_part2_l222_222365


namespace problem_f1_l222_222091

noncomputable def f : ℝ → ℝ := sorry

theorem problem_f1 (h : ∀ x y : ℝ, f x + f (2 * x + y) + 7 * x * y = f (3 * x - y) + 3 * x^2 + 2) : f 10 = -48 :=
sorry

end problem_f1_l222_222091


namespace root_of_inverse_f_plus_x_eq_k_l222_222484

variable {α : Type*} [Nonempty α] [Field α]
variable (f : α → α)
variable (f_inv : α → α)
variable (k : α)

def root_of_f_plus_x_eq_k (x : α) : Prop :=
  f x + x = k

def inverse_function (f : α → α) (f_inv : α → α) : Prop :=
  ∀ y : α, f (f_inv y) = y ∧ f_inv (f y) = y

theorem root_of_inverse_f_plus_x_eq_k
  (h1 : root_of_f_plus_x_eq_k f 5 k)
  (h2 : inverse_function f f_inv) :
  f_inv (k - 5) + (k - 5) = k :=
by
  sorry

end root_of_inverse_f_plus_x_eq_k_l222_222484


namespace range_of_m_l222_222356

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x ≤ 3 → (x ≤ m → (x < y → y < m))) → m ≥ 3 := 
by
  sorry

end range_of_m_l222_222356


namespace smallest_positive_integer_ends_in_3_divisible_by_11_l222_222115

theorem smallest_positive_integer_ends_in_3_divisible_by_11 :
  ∃ n : ℕ, n > 0 ∧ n % 10 = 3 ∧ n % 11 = 0 ∧ n = 113 :=
by
  -- We claim that 113 is the required number
  use 113
  split
  -- Proof that 113 is positive
  sorry
  split
  -- Proof that 113 ends in 3
  sorry
  split
  -- Proof that 113 is divisible by 11
  sorry
  -- The smallest, smallest in scope will be evident by construction in the final formal proof
  sorry  

end smallest_positive_integer_ends_in_3_divisible_by_11_l222_222115


namespace friends_pay_6_22_l222_222186

noncomputable def cost_per_friend : ℕ :=
  let hamburgers := 5 * 3
  let fries := 4 * 120 / 100
  let soda := 5 * 50 / 100
  let spaghetti := 270 / 100
  let milkshakes := 3 * 250 / 100
  let nuggets := 2 * 350 / 100
  let total_bill := hamburgers + fries + soda + spaghetti + milkshakes + nuggets
  let discount := total_bill * 10 / 100
  let discounted_bill := total_bill - discount
  let birthday_friend := discounted_bill * 30 / 100
  let remaining_amount := discounted_bill - birthday_friend
  remaining_amount / 4

theorem friends_pay_6_22 : cost_per_friend = 622 / 100 :=
by
  sorry

end friends_pay_6_22_l222_222186


namespace problem_C_plus_D_l222_222944

theorem problem_C_plus_D (C D : ℚ)
  (h : ∀ x, (D * x - 17) / (x^2 - 8 * x + 15) = C / (x - 3) + 5 / (x - 5)) :
  C + D = 5.8 :=
sorry

end problem_C_plus_D_l222_222944


namespace container_capacity_l222_222303

-- Define the given conditions
def initially_full (x : ℝ) : Prop := (1 / 4) * x + 300 = (3 / 4) * x

-- Define the proof problem to show that the total capacity is 600 liters
theorem container_capacity : ∃ x : ℝ, initially_full x → x = 600 := sorry

end container_capacity_l222_222303


namespace double_counted_page_number_l222_222407

theorem double_counted_page_number (n x : ℕ) 
  (h1: 1 ≤ x ∧ x ≤ n)
  (h2: (n * (n + 1) / 2) + x = 1997) : 
  x = 44 := 
by
  sorry

end double_counted_page_number_l222_222407


namespace ratio_of_sphere_surface_areas_l222_222687

noncomputable def inscribed_sphere_radius (a : ℝ) : ℝ := a / 2
noncomputable def circumscribed_sphere_radius (a : ℝ) : ℝ := a * (Real.sqrt 3) / 2
noncomputable def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

theorem ratio_of_sphere_surface_areas (a : ℝ) (h : 0 < a) : 
  (sphere_surface_area (circumscribed_sphere_radius a)) / (sphere_surface_area (inscribed_sphere_radius a)) = 3 :=
by
  sorry

end ratio_of_sphere_surface_areas_l222_222687


namespace triangle_integral_y_difference_l222_222382

theorem triangle_integral_y_difference :
  ∀ (y : ℕ), (3 ≤ y ∧ y ≤ 15) → (∃ y_min y_max : ℕ, y_min = 3 ∧ y_max = 15 ∧ (y_max - y_min = 12)) :=
by
  intro y
  intro h
  -- skipped proof
  sorry

end triangle_integral_y_difference_l222_222382


namespace sam_initial_nickels_l222_222927

variable (n_now n_given n_initial : Nat)

theorem sam_initial_nickels (h_now : n_now = 63) (h_given : n_given = 39) (h_relation : n_now = n_initial + n_given) : n_initial = 24 :=
by
  sorry

end sam_initial_nickels_l222_222927


namespace find_temperature_l222_222813

theorem find_temperature 
  (temps : List ℤ)
  (h_len : temps.length = 8)
  (h_mean : (temps.sum / 8 : ℝ) = -0.5)
  (h_temps : temps = [-6, -3, x, -6, 2, 4, 3, 0]) : 
  x = 2 :=
by 
  sorry

end find_temperature_l222_222813


namespace nat_numbers_equal_if_divisible_l222_222249

theorem nat_numbers_equal_if_divisible
  (a b : ℕ)
  (h : ∀ n : ℕ, ∃ m : ℕ, n ≠ m → (a^(n+1) + b^(n+1)) % (a^n + b^n) = 0) :
  a = b :=
sorry

end nat_numbers_equal_if_divisible_l222_222249


namespace find_fourth_number_l222_222073

theorem find_fourth_number (a : ℕ → ℕ) 
  (h1 : ∀ n, n ≥ 2 → a n = a (n - 1) + a (n - 2)) 
  (h2 : a 6 = 42) 
  (h3 : a 8 = 110) : 
  a 3 = 10 := 
sorry

end find_fourth_number_l222_222073


namespace find_primes_l222_222607

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ (m : ℕ), m ∣ n → m = 1 ∨ m = n

def divides (a b : ℕ) : Prop := ∃ k, b = k * a

/- Define the three conditions -/
def condition1 (p q r : ℕ) : Prop := divides p (1 + q ^ r)
def condition2 (p q r : ℕ) : Prop := divides q (1 + r ^ p)
def condition3 (p q r : ℕ) : Prop := divides r (1 + p ^ q)

def satisfies_conditions (p q r : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime r ∧ condition1 p q r ∧ condition2 p q r ∧ condition3 p q r

theorem find_primes (p q r : ℕ) :
  satisfies_conditions p q r ↔ (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 5 ∧ q = 3 ∧ r = 2) ∨ (p = 3 ∧ q = 2 ∧ r = 5) :=
by
  sorry

end find_primes_l222_222607


namespace contrapositive_of_real_roots_l222_222792

theorem contrapositive_of_real_roots (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0 :=
sorry

end contrapositive_of_real_roots_l222_222792


namespace equivalent_annual_rate_approx_l222_222599

noncomputable def annual_rate : ℝ := 0.045
noncomputable def days_in_year : ℝ := 365
noncomputable def daily_rate : ℝ := annual_rate / days_in_year
noncomputable def equivalent_annual_rate : ℝ := (1 + daily_rate) ^ days_in_year - 1

theorem equivalent_annual_rate_approx :
  abs (equivalent_annual_rate - 0.0459) < 0.0001 :=
by sorry

end equivalent_annual_rate_approx_l222_222599


namespace remaining_area_l222_222165

-- Given a regular hexagon and a rhombus composed of two equilateral triangles.
-- Hexagon area is 135 square centimeters.

variable (hexagon_area : ℝ) (rhombus_area : ℝ)
variable (is_regular_hexagon : Prop) (is_composed_of_two_equilateral_triangles : Prop)

-- The conditions
def correct_hexagon_area := hexagon_area = 135
def rhombus_is_composed := is_composed_of_two_equilateral_triangles = true
def hexagon_is_regular := is_regular_hexagon = true

-- Goal: Remaining area after cutting out the rhombus should be 75 square centimeters
theorem remaining_area : 
  correct_hexagon_area hexagon_area →
  hexagon_is_regular is_regular_hexagon →
  rhombus_is_composed is_composed_of_two_equilateral_triangles →
  hexagon_area - rhombus_area = 75 :=
by
  sorry

end remaining_area_l222_222165


namespace train_crosses_pole_in_3_seconds_l222_222860

def train_speed_kmph : ℝ := 60
def train_length_m : ℝ := 50

def speed_conversion (speed_kmph : ℝ) : ℝ := speed_kmph * (1000 / 3600)

def crossing_time (distance_m : ℝ) (speed_mps : ℝ) : ℝ := distance_m / speed_mps

theorem train_crosses_pole_in_3_seconds :
  crossing_time train_length_m (speed_conversion train_speed_kmph) = 3 :=
by
  sorry

end train_crosses_pole_in_3_seconds_l222_222860


namespace inequality_implies_double_l222_222496

-- Define the condition
variables {x y : ℝ}

theorem inequality_implies_double (h : x < y) : 2 * x < 2 * y :=
  sorry

end inequality_implies_double_l222_222496


namespace decimal_representation_l222_222603

theorem decimal_representation :
  (13 : ℝ) / (2 * 5^8) = 0.00001664 := 
  sorry

end decimal_representation_l222_222603


namespace no_real_solution_l222_222990

theorem no_real_solution (x : ℝ) : 
  x ≠ 1 ∧ x ≠ 3 ∧ x ≠ 5 ∧ x ≠ 7 → 
  ¬ (
    (1 / ((x - 1) * (x - 3))) + (1 / ((x - 3) * (x - 5))) + (1 / ((x - 5) * (x - 7))) = 1 / 4
  ) :=
by sorry

end no_real_solution_l222_222990


namespace evaluate_f_at_2_l222_222461

-- Define the polynomial function f(x)
def f (x : ℝ) : ℝ := 3 * x^6 - 2 * x^5 + x^3 + 1

theorem evaluate_f_at_2 : f 2 = 34 :=
by
  -- Insert proof here
  sorry

end evaluate_f_at_2_l222_222461


namespace molecular_weight_correct_l222_222556

noncomputable def molecular_weight_compound : ℝ :=
  (3 * 12.01) + (6 * 1.008) + (1 * 16.00)

theorem molecular_weight_correct :
  molecular_weight_compound = 58.078 := by
  sorry

end molecular_weight_correct_l222_222556


namespace temperature_conversion_l222_222280

theorem temperature_conversion (F : ℝ) (C : ℝ) : 
  F = 95 → 
  C = (F - 32) * 5 / 9 → 
  C = 35 := by
  intro hF hC
  sorry

end temperature_conversion_l222_222280


namespace ratio_proof_l222_222209

theorem ratio_proof (a b c : ℝ) (h1 : b / a = 3) (h2 : c / b = 4) : (a + b) / (b + c) = 4 / 15 := by
  sorry

end ratio_proof_l222_222209


namespace value_of_fraction_zero_l222_222014

theorem value_of_fraction_zero (x : ℝ) (h1 : x^2 - 1 = 0) (h2 : 1 - x ≠ 0) : x = -1 :=
by
  sorry

end value_of_fraction_zero_l222_222014


namespace nguyen_fabric_yards_l222_222531

open Nat

theorem nguyen_fabric_yards :
  let fabric_per_pair := 8.5
  let pairs_needed := 7
  let fabric_still_needed := 49
  let total_fabric_needed := pairs_needed * fabric_per_pair
  let fabric_already_have := total_fabric_needed - fabric_still_needed
  let yards_of_fabric := fabric_already_have / 3
  yards_of_fabric = 3.5 := by
    sorry

end nguyen_fabric_yards_l222_222531


namespace find_x_intercept_l222_222762

variables (a x y : ℝ)
def l1 (a x y : ℝ) : Prop := (a + 2) * x + 3 * y = 5
def l2 (a x y : ℝ) : Prop := (a - 1) * x + 2 * y = 6
def are_parallel (a : ℝ) : Prop := (- (a + 2) / 3) = (- (a - 1) / 2)
def x_intercept_of_l1 (a x : ℝ) : Prop := l1 a x 0

theorem find_x_intercept (h : are_parallel a) : x_intercept_of_l1 7 (5 / 9) := 
sorry

end find_x_intercept_l222_222762


namespace smallest_product_is_298150_l222_222668

def digits : List ℕ := [5, 6, 7, 8, 9, 0]

theorem smallest_product_is_298150 :
  ∃ (a b c : ℕ), 
    a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (a * b * c = 298150) :=
sorry

end smallest_product_is_298150_l222_222668


namespace alan_carla_weight_l222_222863

variable (a b c d : ℝ)

theorem alan_carla_weight (h1 : a + b = 280) (h2 : b + c = 230) (h3 : c + d = 250) (h4 : a + d = 300) :
  a + c = 250 := by
sorry

end alan_carla_weight_l222_222863


namespace average_first_21_multiples_of_8_l222_222562

noncomputable def average_of_multiples (n : ℕ) (a : ℕ) : ℕ :=
  let sum := (n * (a + a * n)) / 2
  sum / n

theorem average_first_21_multiples_of_8 : average_of_multiples 21 8 = 88 :=
by
  sorry

end average_first_21_multiples_of_8_l222_222562


namespace determine_specialty_l222_222275

variables 
  (Peter_is_mathematician Sergey_is_physicist Roman_is_physicist : Prop)
  (Peter_is_chemist Sergey_is_mathematician Roman_is_chemist : Prop)

-- Conditions
axiom cond1 : Peter_is_mathematician → ¬ Sergey_is_physicist
axiom cond2 : ¬ Roman_is_physicist → Peter_is_mathematician
axiom cond3 : ¬ Sergey_is_mathematician → Roman_is_chemist

theorem determine_specialty 
  (h1 : ¬ Roman_is_physicist)
: Peter_is_chemist ∧ Sergey_is_mathematician ∧ Roman_is_physicist := 
by sorry

end determine_specialty_l222_222275


namespace factorization_l222_222170

theorem factorization (m : ℤ) : m^2 + 3 * m = m * (m + 3) :=
by sorry

end factorization_l222_222170


namespace arithmetic_progression_l222_222083

theorem arithmetic_progression (a b c : ℝ) (h : a + c = 2 * b) :
  3 * (a^2 + b^2 + c^2) = 6 * (a - b)^2 + (a + b + c)^2 :=
by
  sorry

end arithmetic_progression_l222_222083


namespace integral_equality_l222_222107

theorem integral_equality :
  ∫ x in (-1 : ℝ)..(1 : ℝ), (Real.tan x) ^ 11 + (Real.cos x) ^ 21
  = 2 * ∫ x in (0 : ℝ)..(1 : ℝ), (Real.cos x) ^ 21 :=
by
  sorry

end integral_equality_l222_222107


namespace period_sine_transformed_l222_222951

theorem period_sine_transformed (x : ℝ) : 
  let y := 3 * Real.sin ((x / 3) + (Real.pi / 4))
  ∃ p : ℝ, (∀ x : ℝ, y = 3 * Real.sin ((x + p) / 3 + (Real.pi / 4)) ↔ y = 3 * Real.sin ((x / 3) + (Real.pi / 4))) ∧ p = 6 * Real.pi :=
sorry

end period_sine_transformed_l222_222951


namespace susan_arrives_before_sam_by_14_minutes_l222_222806

theorem susan_arrives_before_sam_by_14_minutes (d : ℝ) (susan_speed sam_speed : ℝ) (h1 : d = 2) (h2 : susan_speed = 12) (h3 : sam_speed = 5) : 
  let susan_time := d / susan_speed
  let sam_time := d / sam_speed
  let susan_minutes := susan_time * 60
  let sam_minutes := sam_time * 60
  sam_minutes - susan_minutes = 14 := 
by
  sorry

end susan_arrives_before_sam_by_14_minutes_l222_222806


namespace moles_of_H2O_formed_l222_222885

theorem moles_of_H2O_formed (moles_NH4NO3 moles_NaOH : ℕ) (percent_NaOH_reacts : ℝ)
  (h_decomposition : moles_NH4NO3 = 2) (h_NaOH : moles_NaOH = 2) 
  (h_percent : percent_NaOH_reacts = 0.85) : 
  (moles_NaOH * percent_NaOH_reacts = 1.7) :=
by
  sorry

end moles_of_H2O_formed_l222_222885


namespace sum_a_b_l222_222765

theorem sum_a_b (a b : ℚ) (h1 : 3 * a + 5 * b = 47) (h2 : 4 * a + 2 * b = 38) : a + b = 85 / 7 :=
by
  sorry

end sum_a_b_l222_222765


namespace increasing_function_k_l222_222266

open Real

theorem increasing_function_k (k : ℝ) : 
  (∀ x > 0, deriv (λ x, (log x) / x - k * x) x > 0) ↔ k ≤ -1 / (2 * exp 3) :=
by
  sorry

end increasing_function_k_l222_222266


namespace polar_to_rectangular_coordinates_l222_222329

theorem polar_to_rectangular_coordinates 
  (r θ : ℝ) 
  (hr : r = 7) 
  (hθ : θ = 7 * Real.pi / 4) : 
  (r * Real.cos θ, r * Real.sin θ) = (7 * Real.sqrt 2 / 2, -7 * Real.sqrt 2 / 2) := 
by
  sorry

end polar_to_rectangular_coordinates_l222_222329


namespace chrysler_floors_difference_l222_222538

theorem chrysler_floors_difference (C L : ℕ) (h1 : C = 23) (h2 : C + L = 35) : C - L = 11 := by
  sorry

end chrysler_floors_difference_l222_222538


namespace math_problem_l222_222899

def letters := "MATHEMATICS".toList

def vowels := "AAEII".toList
def consonants := "MTHMTCS".toList
def fixed_t := 'T'

def factorial (n : Nat) : Nat := 
  if n = 0 then 1 
  else n * factorial (n - 1)

def arrangements (n : Nat) (reps : List Nat) : Nat := 
  factorial n / reps.foldr (fun r acc => factorial r * acc) 1

noncomputable def vowel_arrangements := arrangements 5 [2, 2]
noncomputable def consonant_arrangements := arrangements 6 [2]

noncomputable def total_arrangements := vowel_arrangements * consonant_arrangements

theorem math_problem : total_arrangements = 10800 := by
  sorry

end math_problem_l222_222899


namespace time_to_cover_length_l222_222978

-- Define the conditions
def speed_escalator : ℝ := 12
def length_escalator : ℝ := 150
def speed_person : ℝ := 3

-- State the theorem to be proved
theorem time_to_cover_length : (length_escalator / (speed_escalator + speed_person)) = 10 := by
  sorry

end time_to_cover_length_l222_222978


namespace smallest_integer_ends_in_3_divisible_by_11_correct_l222_222111

def ends_in_3 (n : ℕ) : Prop :=
  n % 10 = 3

def divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def smallest_ends_in_3_divisible_by_11 : ℕ :=
  33

theorem smallest_integer_ends_in_3_divisible_by_11_correct :
  smallest_ends_in_3_divisible_by_11 = 33 ∧ ends_in_3 smallest_ends_in_3_divisible_by_11 ∧ divisible_by_11 smallest_ends_in_3_divisible_by_11 := 
by
  sorry

end smallest_integer_ends_in_3_divisible_by_11_correct_l222_222111


namespace num_valid_arrangements_l222_222965

-- Define the programs
inductive Program
| A | B | C | D | E
deriving DecidableEq, Fintype

open Program

def isValidArrangement (arrangement : Fin 5 → Program) : Prop :=
  (arrangement 0 = A ∨ arrangement 1 = A) ∧       -- Condition 1: Program A must be in the first two positions
  arrangement 0 ≠ B ∧                            -- Condition 2: Program B cannot be in the first position
  arrangement 4 = C                               -- Condition 3: Program C must be in the last position

def validArrangements : Finset (Fin 5 → Program) :=
  Finset.univ.filter (λ arrangement, isValidArrangement arrangement)

theorem num_valid_arrangements : (validArrangements.card = 10) :=
  by
  sorry

end num_valid_arrangements_l222_222965


namespace marbles_problem_l222_222279

theorem marbles_problem (initial_marble_tyrone : ℕ) (initial_marble_eric : ℕ) (x : ℝ)
  (h1 : initial_marble_tyrone = 125)
  (h2 : initial_marble_eric = 25)
  (h3 : initial_marble_tyrone - x = 3 * (initial_marble_eric + x)) :
  x = 12.5 := 
sorry

end marbles_problem_l222_222279


namespace greatest_drop_is_third_quarter_l222_222946

def priceStart (quarter : ℕ) : ℕ :=
  match quarter with
  | 1 => 10
  | 2 => 7
  | 3 => 9
  | 4 => 5
  | _ => 0 -- default case for invalid quarters

def priceEnd (quarter : ℕ) : ℕ :=
  match quarter with
  | 1 => 7
  | 2 => 9
  | 3 => 5
  | 4 => 6
  | _ => 0 -- default case for invalid quarters

def priceChange (quarter : ℕ) : ℤ :=
  priceStart quarter - priceEnd quarter

def greatestDropInQuarter : ℕ :=
  if priceChange 1 > priceChange 3 then 1
  else if priceChange 2 > priceChange 1 then 2
  else if priceChange 3 > priceChange 4 then 3
  else 4

theorem greatest_drop_is_third_quarter :
  greatestDropInQuarter = 3 :=
by
  -- proof goes here
  sorry

end greatest_drop_is_third_quarter_l222_222946


namespace probability_3_heads_is_40_243_l222_222148

noncomputable def probability_of_heads (n k : ℕ) (r : ℚ) : ℚ :=
(n.choose k) * r^k * (1 - r)^(n - k)

theorem probability_3_heads_is_40_243 (r : ℚ) (hr : r = 1 / 3) :
  let p_3_heads := probability_of_heads 5 3 r in
  ∃ (i j : ℕ), p_3_heads = i / j ∧ i + j = 283 :=
by
  sorry

end probability_3_heads_is_40_243_l222_222148


namespace designed_height_correct_l222_222432
noncomputable def designed_height_of_lower_part (H : ℝ) (L : ℝ) : Prop :=
  H = 2 ∧ (H - L) / L = L / H

theorem designed_height_correct : ∃ L, designed_height_of_lower_part 2 L ∧ L = Real.sqrt 5 - 1 :=
by
  sorry

end designed_height_correct_l222_222432


namespace factor_adjustment_l222_222409

theorem factor_adjustment (a b : ℝ) (h : a * b = 65.08) : a / 100 * (100 * b) = 65.08 :=
by
  sorry

end factor_adjustment_l222_222409


namespace popsicle_sticks_difference_l222_222811

def popsicle_sticks_boys (boys : ℕ) (sticks_per_boy : ℕ) : ℕ :=
  boys * sticks_per_boy

def popsicle_sticks_girls (girls : ℕ) (sticks_per_girl : ℕ) : ℕ :=
  girls * sticks_per_girl

theorem popsicle_sticks_difference : 
    popsicle_sticks_boys 10 15 - popsicle_sticks_girls 12 12 = 6 := by
  sorry

end popsicle_sticks_difference_l222_222811


namespace playerA_winning_conditions_l222_222482

def playerA_has_winning_strategy (n : ℕ) : Prop :=
  (n % 4 = 0) ∨ (n % 4 = 3)

theorem playerA_winning_conditions (n : ℕ) (h : n ≥ 2) : 
  playerA_has_winning_strategy n ↔ (n % 4 = 0 ∨ n % 4 = 3) :=
by sorry

end playerA_winning_conditions_l222_222482


namespace pencils_difference_l222_222532

theorem pencils_difference
  (pencils_in_backpack : ℕ := 2)
  (pencils_at_home : ℕ := 15) :
  pencils_at_home - pencils_in_backpack = 13 := by
  sorry

end pencils_difference_l222_222532


namespace wolves_heads_count_l222_222855

/-- 
A person goes hunting in the jungle and discovers a pack of wolves.
It is known that this person has one head and two legs, 
an ordinary wolf has one head and four legs, and a mutant wolf has two heads and three legs.
The total number of heads of all the people and wolves combined is 21,
and the total number of legs is 57.
-/
theorem wolves_heads_count :
  ∃ (x y : ℕ), (x + 2 * y = 20) ∧ (4 * x + 3 * y = 55) ∧ (x + y > 0) ∧ (x + 2 * y + 1 = 21) ∧ (4 * x + 3 * y + 2 = 57) := 
by {
  sorry
}

end wolves_heads_count_l222_222855


namespace find_larger_number_l222_222822

variables (x y : ℝ)

def sum_cond : Prop := x + y = 17
def diff_cond : Prop := x - y = 7

theorem find_larger_number (h1 : sum_cond x y) (h2 : diff_cond x y) : x = 12 :=
sorry

end find_larger_number_l222_222822


namespace original_length_of_field_l222_222540

theorem original_length_of_field (L W : ℕ) 
  (h1 : L * W = 144) 
  (h2 : (L + 6) * W = 198) : 
  L = 16 := 
by 
  sorry

end original_length_of_field_l222_222540


namespace smallest_integer_ends_in_3_divisible_by_11_correct_l222_222113

def ends_in_3 (n : ℕ) : Prop :=
  n % 10 = 3

def divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def smallest_ends_in_3_divisible_by_11 : ℕ :=
  33

theorem smallest_integer_ends_in_3_divisible_by_11_correct :
  smallest_ends_in_3_divisible_by_11 = 33 ∧ ends_in_3 smallest_ends_in_3_divisible_by_11 ∧ divisible_by_11 smallest_ends_in_3_divisible_by_11 := 
by
  sorry

end smallest_integer_ends_in_3_divisible_by_11_correct_l222_222113


namespace min_value_l222_222519

noncomputable def min_value_expr (a b c d : ℝ) : ℝ :=
  (a + b) / c + (b + c) / a + (c + d) / b

theorem min_value 
  (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  min_value_expr a b c d ≥ 6 
  := sorry

end min_value_l222_222519


namespace probability_inequality_l222_222916

open Probability

theorem probability_inequality :
  let outcomes : List (ℕ × ℕ) := List.product (List.range 9) (List.range 9)
  let event := outcomes.filter (λ (ab : ℕ × ℕ), ab.1 - 2 * ab.2 + 10 > 0)
  (event.length : ℚ) / (outcomes.length : ℚ) = 61 / 81 :=
by
  sorry

end probability_inequality_l222_222916


namespace saffron_milk_caps_and_milk_caps_in_basket_l222_222690

structure MushroomBasket :=
  (total : ℕ)
  (saffronMilkCapCount : ℕ)
  (milkCapCount : ℕ)
  (TotalMushrooms : total = 30)
  (SaffronMilkCapCondition : ∀ (selected : Finset ℕ), selected.card = 12 → ∃ i ∈ selected, i < saffronMilkCapCount)
  (MilkCapCondition : ∀ (selected : Finset ℕ), selected.card = 20 → ∃ i ∈ selected, i < milkCapCount)

theorem saffron_milk_caps_and_milk_caps_in_basket
  (basket : MushroomBasket)
  (TotalMushrooms : basket.total = 30)
  (SaffronMilkCapCondition : ∀ (selected : Finset ℕ), selected.card = 12 → ∃ i ∈ selected, i < basket.saffronMilkCapCount)
  (MilkCapCondition : ∀ (selected : Finset ℕ), selected.card = 20 → ∃ i ∈ selected, i < basket.milkCapCount) :
  basket.saffronMilkCapCount = 19 ∧ basket.milkCapCount = 11 :=
sorry

end saffron_milk_caps_and_milk_caps_in_basket_l222_222690


namespace birds_more_than_storks_l222_222848

theorem birds_more_than_storks :
  let birds := 6
  let initial_storks := 3
  let additional_storks := 2
  let total_storks := initial_storks + additional_storks
  birds - total_storks = 1 := by
  sorry

end birds_more_than_storks_l222_222848


namespace find_a4_l222_222053

open Nat

def sequence (a : Nat → Nat) :=
  ∀ n, n ≥ 3 → a n = a (n - 1) + a (n - 2)

theorem find_a4 (a : ℕ → ℕ)
  (h_seq : sequence a)
  (h_a7 : a 7 = 42)
  (h_a9 : a 9 = 110) :
  a 4 = 10 :=
by
  sorry

end find_a4_l222_222053


namespace find_big_bonsai_cost_l222_222663

-- Given definitions based on conditions
def small_bonsai_cost : ℕ := 30
def num_small_bonsai_sold : ℕ := 3
def num_big_bonsai_sold : ℕ := 5
def total_earnings : ℕ := 190

-- Define the function to calculate total earnings from bonsai sales
def calculate_total_earnings (big_bonsai_cost: ℕ) : ℕ :=
  (num_small_bonsai_sold * small_bonsai_cost) + (num_big_bonsai_sold * big_bonsai_cost)

-- The theorem state
theorem find_big_bonsai_cost (B : ℕ) : calculate_total_earnings B = total_earnings → B = 20 :=
by
  sorry

end find_big_bonsai_cost_l222_222663


namespace find_x_l222_222973

theorem find_x (x : ℝ) (hx_pos : 0 < x) (h: (x / 100) * x = 4) : x = 20 := by
  sorry

end find_x_l222_222973


namespace find_fourth_number_l222_222067

variable (a : ℕ → ℕ)

theorem find_fourth_number (h₁ : a 7 = 42) (h₂ : a 9 = 110)
    (h₃ : ∀ n, n ≥ 3 → a n = a (n-1) + a (n-2)) : a 4 = 10 :=
by
  sorry

end find_fourth_number_l222_222067


namespace num_balls_picked_l222_222709

-- Define the variables involved: the numbers of red, blue, and green balls,
-- and the probability of picking two red balls.

def num_red : ℕ := 3
def num_blue : ℕ := 2
def num_green : ℕ := 3
def total_balls : ℕ := num_red + num_blue + num_green
def prob_both_red : ℝ := 0.10714285714285714

-- The combination function defined in Lean's math library
def comb (n k : ℕ) := Nat.choose n k

-- Statement of the problem
theorem num_balls_picked (n : ℕ) (h : comb total_balls n = 28 ∧ (3 : ℝ) / comb total_balls n = prob_both_red) : n = 2 :=
sorry

end num_balls_picked_l222_222709


namespace ten_years_less_average_age_l222_222507

-- Defining the conditions formally
def lukeAge : ℕ := 20
def mrBernardAgeInEightYears : ℕ := 3 * lukeAge

-- Lean statement to prove the problem
theorem ten_years_less_average_age : 
  mrBernardAgeInEightYears - 8 = 52 → (lukeAge + (mrBernardAgeInEightYears - 8)) / 2 - 10 = 26 := 
by
  intros h
  sorry

end ten_years_less_average_age_l222_222507


namespace difference_of_squares_divisibility_l222_222326

theorem difference_of_squares_divisibility (a b : ℤ) :
  ∃ m : ℤ, (2 * a + 3) ^ 2 - (2 * b + 1) ^ 2 = 8 * m ∧ 
           ¬∃ n : ℤ, (2 * a + 3) ^ 2 - (2 * b + 1) ^ 2 = 16 * n :=
by
  sorry

end difference_of_squares_divisibility_l222_222326


namespace factorize_expression_l222_222171

theorem factorize_expression (m : ℝ) : m^2 + 3 * m = m * (m + 3) :=
by
  sorry

end factorize_expression_l222_222171


namespace total_jellybeans_needed_l222_222029

def large_glass_jellybeans : ℕ := 50
def small_glass_jellybeans : ℕ := large_glass_jellybeans / 2
def num_large_glasses : ℕ := 5
def num_small_glasses : ℕ := 3

theorem total_jellybeans_needed : 
  (num_large_glasses * large_glass_jellybeans) + (num_small_glasses * small_glass_jellybeans) = 325 := 
by
  sorry

end total_jellybeans_needed_l222_222029


namespace cone_radius_l222_222903

theorem cone_radius
    (l : ℝ) (n : ℝ) (r : ℝ)
    (h1 : l = 2 * Real.pi)
    (h2 : n = 120)
    (h3 : l = (n * Real.pi * r) / 180 ) :
    r = 3 :=
sorry

end cone_radius_l222_222903


namespace find_fourth_number_l222_222059

def nat_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, n ≥ 3 → a n = a (n - 1) + a (n - 2)

variable {a : ℕ → ℕ}

theorem find_fourth_number (h_seq : nat_sequence a) (h7 : a 7 = 42) (h9 : a 9 = 110) : a 4 = 10 :=
by
  -- Placeholder for proof steps
  sorry

end find_fourth_number_l222_222059


namespace only_n_divides_2_n_minus_1_l222_222174

theorem only_n_divides_2_n_minus_1 :
  ∀ n : ℕ, n ≥ 1 → (n ∣ (2^n - 1)) → n = 1 :=
by
  sorry

end only_n_divides_2_n_minus_1_l222_222174


namespace sum_of_dimensions_eq_18_sqrt_1_5_l222_222311

theorem sum_of_dimensions_eq_18_sqrt_1_5 (P Q R : ℝ) (h1 : P * Q = 30) (h2 : P * R = 50) (h3 : Q * R = 90) :
  P + Q + R = 18 * Real.sqrt 1.5 :=
sorry

end sum_of_dimensions_eq_18_sqrt_1_5_l222_222311


namespace find_a4_l222_222056

def seq (a : ℕ → ℕ) (n : ℕ) : Prop :=
(∀ n, n ≥ 3 → a n = a (n - 1) + a (n - 2))

theorem find_a4 (a : ℕ → ℕ) (h_seq : seq a) (h_a7 : a 7 = 42) (h_a9 : a 9 = 110) : a 4 = 10 :=
by
  sorry

end find_a4_l222_222056


namespace subsets_with_5_and_6_l222_222763

-- Define the main problem
theorem subsets_with_5_and_6 (s : Finset ℕ) (h : s = {1, 2, 3, 4, 5, 6}) :
  (s.filter (λ x, x = 5 ∨ x = 6)).card = 16 :=
sorry

end subsets_with_5_and_6_l222_222763


namespace probability_A_mc_and_B_tf_probability_at_least_one_mc_l222_222778

-- Define the total number of questions
def total_questions : ℕ := 5

-- Define the number of multiple choice questions and true or false questions
def multiple_choice_questions : ℕ := 3
def true_false_questions : ℕ := 2

-- First proof problem: Probability that A draws a multiple-choice question and B draws a true or false question
theorem probability_A_mc_and_B_tf :
  (multiple_choice_questions * true_false_questions : ℚ) / (total_questions * (total_questions - 1)) = 3 / 10 :=
by
  sorry

-- Second proof problem: Probability that at least one of A and B draws a multiple-choice question
theorem probability_at_least_one_mc :
  1 - (true_false_questions * (true_false_questions - 1) : ℚ) / (total_questions * (total_questions - 1)) = 9 / 10 :=
by
  sorry

end probability_A_mc_and_B_tf_probability_at_least_one_mc_l222_222778


namespace coprime_integers_exist_l222_222358

theorem coprime_integers_exist (a b c : ℚ) (t : ℤ) (h1 : a + b + c = t) (h2 : a^2 + b^2 + c^2 = t) (h3 : t ≥ 0) : 
  ∃ (u v : ℤ), Int.gcd u v = 1 ∧ abc = (u^2 : ℚ) / (v^3 : ℚ) :=
by sorry

end coprime_integers_exist_l222_222358


namespace find_integer_solutions_l222_222737

theorem find_integer_solutions :
  (a b : ℤ) →
  3 * a^2 * b^2 + b^2 = 517 + 30 * a^2 →
  (a = 2 ∧ b = 7) ∨ (a = -2 ∧ b = 7) ∨ (a = 2 ∧ b = -7) ∨ (a = -2 ∧ b = -7) :=
sorry

end find_integer_solutions_l222_222737


namespace mary_income_percentage_more_than_tim_l222_222245

variables (J T M : ℝ)
-- Define the conditions
def condition1 := T = 0.5 * J -- Tim's income is 50% less than Juan's
def condition2 := M = 0.8 * J -- Mary's income is 80% of Juan's

-- Define the theorem stating the question and the correct answer
theorem mary_income_percentage_more_than_tim (J T M : ℝ) 
  (h1 : T = 0.5 * J) 
  (h2 : M = 0.8 * J) : 
  (M - T) / T * 100 = 60 := 
  by sorry

end mary_income_percentage_more_than_tim_l222_222245


namespace taylor_one_basket_probability_l222_222136

-- Definitions based on conditions
def not_make_basket_prob : ℚ := 1 / 3
def make_basket_prob : ℚ := 1 - not_make_basket_prob
def trials : ℕ := 3
def successes : ℕ := 1

def binomial_coefficient (n k : ℕ) : ℕ := n.choose k

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (binomial_coefficient n k) * (p^k) * ((1 - p)^(n - k))

theorem taylor_one_basket_probability : 
  binomial_probability trials successes make_basket_prob = 2 / 9 :=
by
  rw [binomial_probability, binomial_coefficient]
  -- The rest of the proof steps can involve simplifications 
  -- and calculations that were mentioned in the solution.
  sorry

end taylor_one_basket_probability_l222_222136


namespace fencing_required_for_field_l222_222841

noncomputable def fence_length (L W : ℕ) : ℕ := 2 * W + L

theorem fencing_required_for_field :
  ∀ (L W : ℕ), (L = 20) → (440 = L * W) → fence_length L W = 64 :=
by
  intros L W hL hA
  sorry

end fencing_required_for_field_l222_222841


namespace cannot_achieve_141_cents_l222_222094
-- Importing the required library

-- Definitions corresponding to types of coins and their values
def penny := 1
def nickel := 5
def dime := 10
def half_dollar := 50

-- The main statement to prove
theorem cannot_achieve_141_cents :
  ¬∃ (x y z : ℕ), x + y + z = 3 ∧ 
    x * penny + y * nickel + z * dime + (3 - x - y - z) * half_dollar = 141 := 
by
  -- Currently leaving the proof as a sorry
  sorry

end cannot_achieve_141_cents_l222_222094


namespace value_of_2alpha_minus_beta_l222_222198

theorem value_of_2alpha_minus_beta (a β : ℝ) (h1 : 3 * Real.sin a - Real.cos a = 0) 
    (h2 : 7 * Real.sin β + Real.cos β = 0) (h3 : 0 < a ∧ a < Real.pi / 2) 
    (h4 : Real.pi / 2 < β ∧ β < Real.pi) : 
    2 * a - β = -3 * Real.pi / 4 := 
sorry

end value_of_2alpha_minus_beta_l222_222198


namespace seating_arrangements_l222_222274

def total_seats_front := 11
def total_seats_back := 12
def middle_seats_front := 3

def number_of_arrangements := 334

theorem seating_arrangements: 
  (total_seats_front - middle_seats_front) * (total_seats_front - middle_seats_front - 1) / 2 +
  (total_seats_back * (total_seats_back - 1)) / 2 +
  (total_seats_front - middle_seats_front) * total_seats_back +
  total_seats_back * (total_seats_front - middle_seats_front) = number_of_arrangements := 
sorry

end seating_arrangements_l222_222274


namespace min_eccentricity_sum_l222_222761

def circle_O1 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 16
def circle_O2 (x y r : ℝ) : Prop := x^2 + y^2 = r^2 ∧ 0 < r ∧ r < 2

def moving_circle_tangent (e1 e2 : ℝ) (r : ℝ) : Prop :=
  e1 = 2 / (4 - r) ∧ e2 = 2 / (4 + r)

theorem min_eccentricity_sum : ∃ (e1 e2 : ℝ) (r : ℝ), 
  circle_O1 x y ∧ circle_O2 x y r ∧ moving_circle_tangent e1 e2 r ∧
    e1 > e2 ∧ (e1 + 2 * e2) = (3 + 2 * Real.sqrt 2) / 4 :=
sorry

end min_eccentricity_sum_l222_222761


namespace arithmetic_sequence_a5_l222_222201

variable {α : Type*} [LinearOrderedField α]

def is_arithmetic_sequence (a : ℕ → α) :=
  ∃ a_1 d, ∀ n, a (n + 1) = a_1 + n * d

theorem arithmetic_sequence_a5 (a : ℕ → α) (h_seq : is_arithmetic_sequence a) (h_cond : a 1 + a 7 = 12) :
  a 4 = 6 :=
by
  sorry

end arithmetic_sequence_a5_l222_222201


namespace johnny_fishes_l222_222805

theorem johnny_fishes (total_fishes sony_multiple j : ℕ) (h1 : total_fishes = 120) (h2 : sony_multiple = 7) (h3 : total_fishes = j + sony_multiple * j) : j = 15 :=
by sorry

end johnny_fishes_l222_222805


namespace geometric_sequence_ratio_l222_222195

theorem geometric_sequence_ratio
  (a₁ : ℝ) (q : ℝ) (hq : q ≠ 1)
  (S : ℕ → ℝ)
  (hS₃ : S 3 = a₁ * (1 - q^3) / (1 - q))
  (hS₆ : S 6 = a₁ * (1 - q^6) / (1 - q))
  (hS₃_val : S 3 = 2)
  (hS₆_val : S 6 = 18) :
  S 10 / S 5 = 1 + 2^(1/3) + 2^(2/3) :=
sorry

end geometric_sequence_ratio_l222_222195


namespace infinite_subsequence_with_same_gcd_l222_222801

open Nat

theorem infinite_subsequence_with_same_gcd (seq : ℕ → ℕ) (h1 : ∀ n, 0 < seq n) (h2 : ∀ n m, seq n = seq m → seq n = seq m) :
  ∃ (subseq : ℕ → ℕ), (∀ n m, gcd (subseq n) (subseq m) = gcd (subseq 0) (subseq 0)) :=
sorry

end infinite_subsequence_with_same_gcd_l222_222801


namespace total_length_of_ropes_l222_222418

theorem total_length_of_ropes 
  (L : ℕ)
  (first_used second_used : ℕ)
  (h1 : first_used = 42) 
  (h2 : second_used = 12) 
  (h3 : (L - second_used) = 4 * (L - first_used)) :
  2 * L = 104 :=
by
  -- We skip the proof for now
  sorry

end total_length_of_ropes_l222_222418


namespace square_of_rational_l222_222140

theorem square_of_rational (b : ℚ) : b^2 = b * b :=
sorry

end square_of_rational_l222_222140


namespace sum_base9_to_base9_eq_l222_222862

-- Definition of base 9 numbers
def base9_to_base10 (n : ℕ) : ℕ :=
  let digit1 := n % 10
  let digit2 := (n / 10) % 10
  let digit3 := (n / 100) % 10
  digit1 + 9 * digit2 + 81 * digit3

-- Definition of base 10 to base 9 conversion
def base10_to_base9 (n : ℕ) : ℕ :=
  let digit1 := n % 9
  let digit2 := (n / 9) % 9
  let digit3 := (n / 81) % 9
  digit1 + 10 * digit2 + 100 * digit3

-- The theorem to prove
theorem sum_base9_to_base9_eq :
  let x := base9_to_base10 236
  let y := base9_to_base10 327
  let z := base9_to_base10 284
  base10_to_base9 (x + y + z) = 858 :=
by {
  sorry
}

end sum_base9_to_base9_eq_l222_222862


namespace isosceles_triangle_properties_l222_222584

/--
  An isosceles triangle has a base of 6 units and legs of 5 units each.
  Prove:
  1. The area of the triangle is 12 square units.
  2. The radius of the inscribed circle is 1.5 units.
-/
theorem isosceles_triangle_properties (base : ℝ) (legs : ℝ) 
  (h_base : base = 6) (h_legs : legs = 5) : 
  ∃ (area : ℝ) (inradius : ℝ), 
  area = 12 ∧ inradius = 1.5 
  :=
by
  sorry

end isosceles_triangle_properties_l222_222584


namespace value_of_2_68_times_0_74_l222_222957

theorem value_of_2_68_times_0_74 : 
  (268 * 74 = 19732) → (2.68 * 0.74 = 1.9732) :=
by intro h1; sorry

end value_of_2_68_times_0_74_l222_222957


namespace Billy_is_45_l222_222520

variable (B J : ℕ)

-- Condition 1: Billy's age is three times Joe's age
def condition1 : Prop := B = 3 * J

-- Condition 2: The sum of their ages is 60
def condition2 : Prop := B + J = 60

-- The theorem we want to prove: Billy's age is 45
theorem Billy_is_45 (h1 : condition1 B J) (h2 : condition2 B J) : B = 45 := 
sorry

end Billy_is_45_l222_222520


namespace part_a_part_b_l222_222800

noncomputable def probability_Peter_satisfied : ℚ :=
  let total_people := 100
  let men := 50
  let women := 50
  let P_both_men := (men - 1 : ℚ)/ (total_people - 1 : ℚ) * (men - 2 : ℚ)/ (total_people - 2 : ℚ)
  1 - P_both_men

theorem part_a : probability_Peter_satisfied = 25 / 33 := 
  sorry

noncomputable def expected_satisfied_men : ℚ :=
  let men := 50
  probability_Peter_satisfied * men

theorem part_b : expected_satisfied_men = 1250 / 33 := 
  sorry

end part_a_part_b_l222_222800


namespace find_k_l222_222930

-- Definitions based on given conditions
def ellipse_equation (x y : ℝ) (k : ℝ) : Prop :=
  5 * x^2 + k * y^2 = 5

def is_focus (x y : ℝ) : Prop :=
  x = 0 ∧ y = 2

-- Statement of the problem
theorem find_k (k : ℝ) :
  (∀ x y, ellipse_equation x y k) →
  is_focus 0 2 →
  k = 1 :=
sorry

end find_k_l222_222930


namespace propositions_correct_l222_222681

def f (x : Real) (b c : Real) : Real := x * abs x + b * x + c

-- Define proposition P1: When c = 0, y = f(x) is an odd function.
def P1 (b : Real) : Prop :=
  ∀ x : Real, f x b 0 = - f (-x) b 0

-- Define proposition P2: When b = 0 and c > 0, the equation f(x) = 0 has only one real root.
def P2 (c : Real) : Prop :=
  c > 0 → ∃! x : Real, f x 0 c = 0

-- Define proposition P3: The graph of y = f(x) is symmetric about the point (0, c).
def P3 (b c : Real) : Prop :=
  ∀ x : Real, f x b c = 2 * c - f x b c

-- Define the final theorem statement
theorem propositions_correct (b c : Real) : P1 b ∧ P2 c ∧ P3 b c := sorry

end propositions_correct_l222_222681


namespace log_equality_implies_exp_equality_l222_222080

theorem log_equality_implies_exp_equality (x y z a : ℝ) (h : (x * (y + z - x)) / (Real.log x) = (y * (x + z - y)) / (Real.log y) ∧ (y * (x + z - y)) / (Real.log y) = (z * (x + y - z)) / (Real.log z)) :
  x^y * y^x = z^x * x^z ∧ z^x * x^z = y^z * z^y :=
by
  sorry

end log_equality_implies_exp_equality_l222_222080


namespace selling_price_of_mixture_l222_222267

noncomputable def selling_price_per_pound (weight1 weight2 price1 price2 total_weight : ℝ) : ℝ :=
  (weight1 * price1 + weight2 * price2) / total_weight

theorem selling_price_of_mixture :
  selling_price_per_pound 20 10 2.95 3.10 30 = 3.00 :=
by
  -- Skipping the proof part
  sorry

end selling_price_of_mixture_l222_222267


namespace odd_function_evaluation_l222_222241

theorem odd_function_evaluation (f : ℝ → ℝ) (hf : ∀ x, f (-x) = -f x) (h : f (-3) = -2) : f 3 + f 0 = 2 :=
by 
  sorry

end odd_function_evaluation_l222_222241


namespace shaded_area_fraction_l222_222667

theorem shaded_area_fraction (total_grid_squares : ℕ) (number_1_squares : ℕ) (number_9_squares : ℕ) (number_8_squares : ℕ) (partial_squares_1 : ℕ) (partial_squares_2 : ℕ) (partial_squares_3 : ℕ) :
  total_grid_squares = 18 * 8 →
  number_1_squares = 8 →
  number_9_squares = 15 →
  number_8_squares = 16 →
  partial_squares_1 = 6 →
  partial_squares_2 = 6 →
  partial_squares_3 = 8 →
  (2 * (number_1_squares + number_9_squares + number_9_squares + number_8_squares) + (partial_squares_1 + partial_squares_2 + partial_squares_3)) = 2 * (74 : ℕ) →
  (74 / 144 : ℚ) = 37 / 72 :=
by
  intros _ _ _ _ _ _ _ _
  sorry

end shaded_area_fraction_l222_222667


namespace positive_number_property_l222_222971

theorem positive_number_property (x : ℝ) (h_pos : x > 0) (h_property : 0.01 * x * x = 4) : x = 20 :=
sorry

end positive_number_property_l222_222971


namespace charcoal_drawings_count_l222_222826

-- Defining the conditions
def total_drawings : Nat := 25
def colored_pencil_drawings : Nat := 14
def blending_marker_drawings : Nat := 7

-- Defining the target value for charcoal drawings
def charcoal_drawings : Nat := total_drawings - (colored_pencil_drawings + blending_marker_drawings)

-- The theorem we need to prove
theorem charcoal_drawings_count : charcoal_drawings = 4 :=
by
  -- Lean proof goes here, but since we skip the proof, we'll just use 'sorry'
  sorry

end charcoal_drawings_count_l222_222826


namespace smallest_product_of_non_factors_l222_222102

theorem smallest_product_of_non_factors (a b : ℕ) (h_a : a ∣ 48) (h_b : b ∣ 48) (h_distinct : a ≠ b) (h_prod_non_factor : ¬ (a * b ∣ 48)) : a * b = 18 :=
sorry

end smallest_product_of_non_factors_l222_222102


namespace find_primes_l222_222608

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ (m : ℕ), m ∣ n → m = 1 ∨ m = n

def divides (a b : ℕ) : Prop := ∃ k, b = k * a

/- Define the three conditions -/
def condition1 (p q r : ℕ) : Prop := divides p (1 + q ^ r)
def condition2 (p q r : ℕ) : Prop := divides q (1 + r ^ p)
def condition3 (p q r : ℕ) : Prop := divides r (1 + p ^ q)

def satisfies_conditions (p q r : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime r ∧ condition1 p q r ∧ condition2 p q r ∧ condition3 p q r

theorem find_primes (p q r : ℕ) :
  satisfies_conditions p q r ↔ (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 5 ∧ q = 3 ∧ r = 2) ∨ (p = 3 ∧ q = 2 ∧ r = 5) :=
by
  sorry

end find_primes_l222_222608


namespace jellybeans_needed_l222_222034

-- Define the initial conditions as constants
def jellybeans_per_large_glass := 50
def jellybeans_per_small_glass := jellybeans_per_large_glass / 2
def number_of_large_glasses := 5
def number_of_small_glasses := 3

-- Calculate the total number of jellybeans needed
def total_jellybeans : ℕ :=
  (number_of_large_glasses * jellybeans_per_large_glass) + 
  (number_of_small_glasses * jellybeans_per_small_glass)

-- Prove that the total number of jellybeans needed is 325
theorem jellybeans_needed : total_jellybeans = 325 :=
sorry

end jellybeans_needed_l222_222034


namespace equation_has_no_real_solutions_l222_222735

/-- Prove that the graph of the given equation is empty.

Given the equation:
x^2 + 3y^2 - 4x - 6y + 10 = 0,
we need to show that there are no real (x, y) solutions that satisfy this equation.
-/
theorem equation_has_no_real_solutions :
  ∀ (x y : ℝ), x^2 + 3 * y^2 - 4 * x - 6 * y + 10 ≠ 0 :=
by {
  assume x y,
  /- The proof steps would show that the transformed equation cannot be satisfied by any real (x, y) -/
  sorry
}

end equation_has_no_real_solutions_l222_222735


namespace number_of_young_teachers_selected_l222_222451

theorem number_of_young_teachers_selected 
  (total_teachers elderly_teachers middle_aged_teachers young_teachers sample_size : ℕ)
  (h_total: total_teachers = 200)
  (h_elderly: elderly_teachers = 25)
  (h_middle_aged: middle_aged_teachers = 75)
  (h_young: young_teachers = 100)
  (h_sample_size: sample_size = 40)
  : young_teachers * sample_size / total_teachers = 20 := 
sorry

end number_of_young_teachers_selected_l222_222451


namespace average_age_of_team_is_23_l222_222679

noncomputable def average_age_team (A : ℝ) : Prop :=
  let captain_age := 27
  let wicket_keeper_age := 28
  let team_size := 11
  let remaining_players := team_size - 2
  let remaining_average_age := A - 1
  11 * A = 55 + 9 * (A - 1)

theorem average_age_of_team_is_23 : average_age_team 23 := by
  sorry

end average_age_of_team_is_23_l222_222679


namespace percentage_saved_l222_222852

-- Define the actual and saved amount.
def actual_investment : ℕ := 150000
def saved_amount : ℕ := 50000

-- Define the planned investment based on the conditions.
def planned_investment : ℕ := actual_investment + saved_amount

-- Proof goal: The percentage saved is 25%.
theorem percentage_saved : (saved_amount * 100) / planned_investment = 25 := 
by 
  sorry

end percentage_saved_l222_222852


namespace greatest_possible_difference_l222_222012

theorem greatest_possible_difference (x y : ℤ) (hx : 7 < x ∧ x < 9) (hy : 9 < y ∧ y < 15) : 
  ∃ d, d = y - x ∧ d = 6 := 
by
  sorry

end greatest_possible_difference_l222_222012


namespace tulip_area_of_flower_bed_l222_222440

theorem tulip_area_of_flower_bed 
  (CD CF : ℝ) (DE : ℝ := 4) (EF : ℝ := 3) 
  (triangle : ∀ (A B C : ℝ), A = B + C) : 
  CD * CF = 12 :=
by sorry

end tulip_area_of_flower_bed_l222_222440


namespace ice_cream_cone_cost_is_5_l222_222786

noncomputable def cost_of_ice_cream_cone (x : ℝ) : Prop := 
  let total_cost_of_cones := 15 * x
  let total_cost_of_puddings := 5 * 2
  let extra_spent_on_cones := total_cost_of_cones - total_cost_of_puddings
  extra_spent_on_cones = 65

theorem ice_cream_cone_cost_is_5 : ∃ x : ℝ, cost_of_ice_cream_cone x ∧ x = 5 :=
by 
  use 5
  unfold cost_of_ice_cream_cone
  simp
  sorry

end ice_cream_cone_cost_is_5_l222_222786


namespace sum_of_five_consecutive_squares_not_perfect_square_l222_222397

theorem sum_of_five_consecutive_squares_not_perfect_square (n : ℤ) : 
  ¬ ∃ (k : ℤ), k^2 = (n - 2)^2 + (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2 := 
by
  sorry

end sum_of_five_consecutive_squares_not_perfect_square_l222_222397


namespace area_of_rectangle_l222_222802

-- Definitions from problem conditions
variable (AB CD x : ℝ)
variable (h1 : AB = 24)
variable (h2 : CD = 60)
variable (h3 : BC = x)
variable (h4 : BF = 2 * x)
variable (h5 : similar (triangle AEB) (triangle FDC))

-- Goal: Prove the area of rectangle BCFE
theorem area_of_rectangle (h1 : AB = 24) (h2 : CD = 60) (x y : ℝ) 
  (h3 : BC = x) (h4 : BF = 2 * x) (h5 : BC * BF = y) : y = 1440 :=
sorry -- proof will be provided here

end area_of_rectangle_l222_222802


namespace sale_in_fifth_month_l222_222571

def sale_first_month : ℝ := 3435
def sale_second_month : ℝ := 3927
def sale_third_month : ℝ := 3855
def sale_fourth_month : ℝ := 4230
def required_avg_sale : ℝ := 3500
def sale_sixth_month : ℝ := 1991

theorem sale_in_fifth_month :
  (sale_first_month + sale_second_month + sale_third_month + sale_fourth_month + s + sale_sixth_month) / 6 = required_avg_sale ->
  s = 3562 :=
by
  sorry

end sale_in_fifth_month_l222_222571


namespace parabola_distance_focus_l222_222498

theorem parabola_distance_focus (x y : ℝ) (h1 : y^2 = 4 * x) (h2 : (x - 1)^2 + y^2 = 16) : x = 3 := by
  sorry

end parabola_distance_focus_l222_222498


namespace highest_monthly_profit_max_average_profit_l222_222436

noncomputable def profit (x : ℕ) : ℤ :=
if 1 ≤ x ∧ x ≤ 5 then 26 * x - 56
else if 5 < x ∧ x ≤ 12 then 210 - 20 * x
else 0

noncomputable def average_profit (x : ℕ) : ℝ :=
if 1 ≤ x ∧ x ≤ 5 then (13 * ↑x - 43 : ℤ) / ↑x
else if 5 < x ∧ x ≤ 12 then (-10 * ↑x + 200 - 640 / ↑x : ℝ)
else 0

theorem highest_monthly_profit :
  ∃ m p, m = 6 ∧ p = 90 ∧ profit m = p :=
by sorry

theorem max_average_profit (x : ℕ) :
  1 ≤ x ∧ x ≤ 12 →
  average_profit x ≤ 40 ∧ (average_profit 8 = 40 → x = 8) :=
by sorry

end highest_monthly_profit_max_average_profit_l222_222436


namespace max_val_neg_5000_l222_222386

noncomputable def max_val_expression (x y : ℝ) : ℝ :=
  (x^2 + (1 / y^2)) * (x^2 + (1 / y^2) - 100) + (y^2 + (1 / x^2)) * (y^2 + (1 / x^2) - 100)

theorem max_val_neg_5000 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  ∃ x y, x > 0 ∧ y > 0 ∧ max_val_expression x y = -5000 :=
by
  sorry

end max_val_neg_5000_l222_222386


namespace mike_initial_marbles_l222_222915

theorem mike_initial_marbles (n : ℕ) 
  (gave_to_sam : ℕ) (left_with_mike : ℕ)
  (h1 : gave_to_sam = 4)
  (h2 : left_with_mike = 4)
  (h3 : n = gave_to_sam + left_with_mike) : n = 8 := 
by
  sorry

end mike_initial_marbles_l222_222915


namespace original_average_age_l222_222814

-- Definitions based on conditions
def original_strength : ℕ := 12
def new_student_count : ℕ := 12
def new_student_average_age : ℕ := 32
def age_decrease : ℕ := 4
def total_student_count : ℕ := original_strength + new_student_count
def combined_total_age (A : ℕ) : ℕ := original_strength * A + new_student_count * new_student_average_age
def new_average_age (A : ℕ) : ℕ := A - age_decrease

-- Statement of the problem
theorem original_average_age (A : ℕ) (h : combined_total_age A / total_student_count = new_average_age A) : A = 40 := 
by 
  sorry

end original_average_age_l222_222814


namespace range_of_a_l222_222819

variable (a : ℝ)

theorem range_of_a (ha : a ≥ 1/4) : ¬ ∃ x : ℝ, a * x^2 + x + 1 < 0 := sorry

end range_of_a_l222_222819


namespace distance_to_school_l222_222647

variable (v d : ℝ) -- typical speed (v) and distance (d)

theorem distance_to_school :
  (30 / 60 : ℝ) = 1 / 2 ∧ -- 30 minutes is 1/2 hour
  (18 / 60 : ℝ) = 3 / 10 ∧ -- 18 minutes is 3/10 hour
  d = v * (1 / 2) ∧ -- distance for typical day
  d = (v + 12) * (3 / 10) -- distance for quieter day
  → d = 9 := sorry

end distance_to_school_l222_222647


namespace contradiction_method_l222_222669

theorem contradiction_method (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + a = 0 ∧ y^2 - 2*y + a = 0) → a < 1 :=
sorry

end contradiction_method_l222_222669


namespace Teresa_age_at_Michiko_birth_l222_222676

-- Definitions of the conditions
def Teresa_age_now : ℕ := 59
def Morio_age_now : ℕ := 71
def Morio_age_at_Michiko_birth : ℕ := 38

-- Prove that Teresa was 26 years old when she gave birth to Michiko.
theorem Teresa_age_at_Michiko_birth : 38 - (71 - 59) = 26 := by
  -- Provide the proof here
  sorry

end Teresa_age_at_Michiko_birth_l222_222676


namespace books_left_over_l222_222374

theorem books_left_over (n_boxes : ℕ) (books_per_box : ℕ) (new_box_capacity : ℕ) :
  n_boxes = 1575 → books_per_box = 45 → new_box_capacity = 46 →
  (n_boxes * books_per_box) % new_box_capacity = 15 :=
by
  intro h1 h2 h3
  rw [h1, h2, h3]
  -- Actual proof steps would go here
  sorry

end books_left_over_l222_222374


namespace prove_values_l222_222367

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 - 1/x + b

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

theorem prove_values (a b : ℝ) (h1 : a > 0) (h2 : is_integer b) :
  (f a b (Real.log a) = 6 ∧ f a b (Real.log (1 / a)) = 2) ∨
  (f a b (Real.log a) = -2 ∧ f a b (Real.log (1 / a)) = 2) :=
sorry

end prove_values_l222_222367


namespace total_gain_is_19200_l222_222158

noncomputable def total_annual_gain_of_partnership (x : ℝ) (A_share : ℝ) (B_investment_after : ℕ) (C_investment_after : ℕ) : ℝ :=
  let A_investment_time := 12
  let B_investment_time := 12 - B_investment_after
  let C_investment_time := 12 - C_investment_after
  let proportional_sum := x * A_investment_time + 2 * x * B_investment_time + 3 * x * C_investment_time
  let individual_proportion := proportional_sum / A_investment_time
  3 * A_share

theorem total_gain_is_19200 (x A_share : ℝ) (B_investment_after C_investment_after : ℕ) :
  A_share = 6400 →
  B_investment_after = 6 →
  C_investment_after = 8 →
  total_annual_gain_of_partnership x A_share B_investment_after C_investment_after = 19200 :=
by
  intros hA hB hC
  have x_pos : x > 0 := by sorry   -- Additional assumptions if required
  have A_share_pos : A_share > 0 := by sorry -- Additional assumptions if required
  sorry

end total_gain_is_19200_l222_222158


namespace find_fourth_number_l222_222074

theorem find_fourth_number (a : ℕ → ℕ) (h1 : a 7 = 42) (h2 : a 9 = 110)
  (h3 : ∀ n, n ≥ 3 → a n = a (n - 1) + a (n - 2)) : a 4 = 10 := 
sorry

end find_fourth_number_l222_222074


namespace approximate_probability_hit_shot_l222_222851

-- Define the data from the table
def shots : List ℕ := [10, 50, 100, 150, 200, 500, 1000, 2000]
def hits : List ℕ := [9, 40, 70, 108, 143, 361, 721, 1440]
def hit_rates : List ℚ := [0.9, 0.8, 0.7, 0.72, 0.715, 0.722, 0.721, 0.72]

-- State the theorem that the stabilized hit rate is approximately 0.72
theorem approximate_probability_hit_shot : 
  ∃ (p : ℚ), p = 0.72 ∧ 
  ∀ (n : ℕ), n ∈ [150, 200, 500, 1000, 2000] → 
     ∃ (r : ℚ), r = 0.72 ∧ 
     r = (hits.get ⟨shots.indexOf n, sorry⟩ : ℚ) / n := sorry

end approximate_probability_hit_shot_l222_222851


namespace ineq_10_3_minus_9_5_l222_222239

variable {a b c : ℝ}

/-- Given \(a, b, c\) are positive real numbers and \(a + b + c = 1\), prove \(10(a^3 + b^3 + c^3) - 9(a^5 + b^5 + c^5) \geq 1\). -/
theorem ineq_10_3_minus_9_5 (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b + c = 1) :
  10 * (a^3 + b^3 + c^3) - 9 * (a^5 + b^5 + c^5) ≥ 1 := 
sorry

end ineq_10_3_minus_9_5_l222_222239


namespace hash_of_hash_of_hash_of_70_l222_222878

def hash (N : ℝ) : ℝ := 0.4 * N + 2

theorem hash_of_hash_of_hash_of_70 : hash (hash (hash 70)) = 8 := by
  sorry

end hash_of_hash_of_hash_of_70_l222_222878


namespace sin_alpha_sub_beta_cos_beta_l222_222360

variables (α β : ℝ)
variables (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
variables (h1 : Real.sin α = 3 / 5)
variables (h2 : Real.tan (α - β) = -1 / 3)

theorem sin_alpha_sub_beta : Real.sin (α - β) = - Real.sqrt 10 / 10 :=
by
  sorry

theorem cos_beta : Real.cos β = 9 * Real.sqrt 10 / 50 :=
by
  sorry

end sin_alpha_sub_beta_cos_beta_l222_222360


namespace part1_l222_222228

theorem part1 (a b c t m n : ℝ) (h1 : a > 0) (h2 : m = n) (h3 : t = (3 + (t + 1)) / 2) : t = 4 :=
sorry

end part1_l222_222228


namespace find_two_digit_numbers_l222_222177

theorem find_two_digit_numbers :
  ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) → (0 ≤ b ∧ b ≤ 9) → (10 * a + b = 3 * a * b) → (10 * a + b = 15 ∨ 10 * a + b = 24) :=
by
  intros
  sorry

end find_two_digit_numbers_l222_222177


namespace math_problem_l222_222656

open Classical

theorem math_problem (s x y : ℝ) (h₁ : s > 0) (h₂ : x^2 + y^2 ≠ 0) (h₃ : x * s^2 < y * s^2) :
  ¬(-x^2 < -y^2) ∧ ¬(-x^2 < y^2) ∧ ¬(x^2 < -y^2) ∧ ¬(x^2 > y^2) := by
  sorry

end math_problem_l222_222656


namespace equal_split_l222_222653

theorem equal_split (A B C : ℝ) (h1 : A < B) (h2 : B < C) : 
  (B + C - 2 * A) / 3 = (A + B + C) / 3 - A :=
by
  sorry

end equal_split_l222_222653


namespace largest_integer_dividing_sum_of_5_consecutive_integers_l222_222817

theorem largest_integer_dividing_sum_of_5_consecutive_integers :
  ∀ (a : ℤ), ∃ (n : ℤ), n = 5 ∧ 5 ∣ ((a - 2) + (a - 1) + a + (a + 1) + (a + 2)) := by
  sorry

end largest_integer_dividing_sum_of_5_consecutive_integers_l222_222817


namespace company_b_profit_l222_222731

-- Definitions as per problem conditions
def A_profit : ℝ := 90000
def A_share : ℝ := 0.60
def B_share : ℝ := 0.40

-- Theorem statement to be proved
theorem company_b_profit : B_share * (A_profit / A_share) = 60000 :=
by
  sorry

end company_b_profit_l222_222731


namespace find_fourth_number_l222_222065

variable (a : ℕ → ℕ)

theorem find_fourth_number (h₁ : a 7 = 42) (h₂ : a 9 = 110)
    (h₃ : ∀ n, n ≥ 3 → a n = a (n-1) + a (n-2)) : a 4 = 10 :=
by
  sorry

end find_fourth_number_l222_222065


namespace find_m_for_perfect_square_trinomial_l222_222208

theorem find_m_for_perfect_square_trinomial :
  ∃ m : ℤ, (∀ (x y : ℝ), (9 * x^2 + m * x * y + 16 * y^2 = (3 * x + 4 * y)^2) ∨ 
                   (9 * x^2 + m * x * y + 16 * y^2 = (3 * x - 4 * y)^2) ∨ 
                   (9 * x^2 + m * x * y + 16 * y^2 = (-3 * x + 4 * y)^2) ∨ 
                   (9 * x^2 + m * x * y + 16 * y^2 = (-3 * x - 4 * y)^2)) ↔ 
          (m = 24 ∨ m = -24) := 
by
  sorry

end find_m_for_perfect_square_trinomial_l222_222208


namespace avg_annual_growth_rate_equation_l222_222019

variable (x : ℝ)
def foreign_trade_income_2007 : ℝ := 250 -- million yuan
def foreign_trade_income_2009 : ℝ := 360 -- million yuan

theorem avg_annual_growth_rate_equation :
  2.5 * (1 + x) ^ 2 = 3.6 := sorry

end avg_annual_growth_rate_equation_l222_222019


namespace other_root_of_quadratic_l222_222392

theorem other_root_of_quadratic (p x : ℝ) (h : 7 * x^2 + p * x - 9 = 0) (root1 : x = -3) : 
  x = 3 / 7 :=
by
  sorry

end other_root_of_quadratic_l222_222392


namespace lewis_found_20_items_l222_222044

noncomputable def tanya_items : ℕ := 4

noncomputable def samantha_items : ℕ := 4 * tanya_items

noncomputable def lewis_items : ℕ := samantha_items + 4

theorem lewis_found_20_items : lewis_items = 20 := by
  sorry

end lewis_found_20_items_l222_222044


namespace ball_returns_to_Ben_after_three_throws_l222_222847

def circle_throw (n : ℕ) (skip : ℕ) (start : ℕ) : ℕ :=
  (start + skip) % n

theorem ball_returns_to_Ben_after_three_throws :
  ∀ (n : ℕ) (skip : ℕ) (start : ℕ),
  n = 15 → skip = 5 → start = 1 →
  (circle_throw n skip (circle_throw n skip (circle_throw n skip start))) = start :=
by
  intros n skip start hn hskip hstart
  sorry

end ball_returns_to_Ben_after_three_throws_l222_222847


namespace asymptotes_of_hyperbola_l222_222931

theorem asymptotes_of_hyperbola :
  ∀ x y : ℝ, (y^2 / 4 - x^2 / 9 = 1) → (y = (2 / 3) * x ∨ y = -(2 / 3) * x) :=
by
  sorry

end asymptotes_of_hyperbola_l222_222931


namespace peanut_butter_candy_count_l222_222422

theorem peanut_butter_candy_count (B G P : ℕ) 
  (hB : B = 43)
  (hG : G = B + 5)
  (hP : P = 4 * G) :
  P = 192 := by
  sorry

end peanut_butter_candy_count_l222_222422


namespace find_value_of_P_l222_222485

def f (x : ℝ) : ℝ := (x^2 + x - 2)^2002 + 3

theorem find_value_of_P :
  f ( (Real.sqrt 5) / 2 - 1 / 2 ) = 4 := by
  sorry

end find_value_of_P_l222_222485


namespace smallest_crate_side_l222_222438

/-- 
A crate measures some feet by 8 feet by 12 feet on the inside. 
A stone pillar in the shape of a right circular cylinder must fit into the crate for shipping so that 
it rests upright when the crate sits on at least one of its six sides. 
The radius of the pillar is 7 feet. 
Prove that the length of the crate's smallest side is 8 feet.
-/
theorem smallest_crate_side (x : ℕ) (hx : x >= 14) : min (min x 8) 12 = 8 :=
by {
  sorry
}

end smallest_crate_side_l222_222438


namespace smallest_positive_integer_ends_in_3_divisible_by_11_l222_222118

theorem smallest_positive_integer_ends_in_3_divisible_by_11 :
  ∃ n : ℕ, n > 0 ∧ n % 10 = 3 ∧ n % 11 = 0 ∧ n = 113 :=
by
  -- We claim that 113 is the required number
  use 113
  split
  -- Proof that 113 is positive
  sorry
  split
  -- Proof that 113 ends in 3
  sorry
  split
  -- Proof that 113 is divisible by 11
  sorry
  -- The smallest, smallest in scope will be evident by construction in the final formal proof
  sorry  

end smallest_positive_integer_ends_in_3_divisible_by_11_l222_222118


namespace wine_distribution_l222_222193

theorem wine_distribution (m n k s : ℕ) (h : Nat.gcd m (Nat.gcd n k) = 1) (h_s : s < m + n + k) :
  ∃ g : ℕ, g = s := by
  sorry

end wine_distribution_l222_222193


namespace fee_difference_l222_222552

-- Defining the given conditions
def stadium_capacity : ℕ := 2000
def fraction_full : ℚ := 3 / 4
def entry_fee : ℚ := 20

-- Statement to prove
theorem fee_difference :
  let people_at_three_quarters := stadium_capacity * fraction_full
  let total_fees_at_three_quarters := people_at_three_quarters * entry_fee
  let total_fees_full := stadium_capacity * entry_fee
  total_fees_full - total_fees_at_three_quarters = 10000 :=
by
  sorry

end fee_difference_l222_222552


namespace fixed_point_sum_l222_222564

theorem fixed_point_sum (a m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : (m, n) = (1, a * (1-1) + 2)) : m + n = 4 :=
by {
  sorry
}

end fixed_point_sum_l222_222564


namespace always_positive_sum_l222_222682

def f : ℝ → ℝ := sorry  -- assuming f(x) is provided elsewhere

theorem always_positive_sum (f : ℝ → ℝ)
    (h1 : ∀ x, f x = -f (2 - x))
    (h2 : ∀ x, x < 1 → f (x) < f (x + 1))
    (x1 x2 : ℝ)
    (h3 : x1 + x2 > 2)
    (h4 : (x1 - 1) * (x2 - 1) < 0) :
  f x1 + f x2 > 0 :=
by {
  sorry
}

end always_positive_sum_l222_222682


namespace pizza_volume_piece_l222_222448

theorem pizza_volume_piece (h : ℝ) (d : ℝ) (n : ℝ) (V_piece : ℝ) 
  (h_eq : h = 1 / 2) (d_eq : d = 16) (n_eq : n = 8) : 
  V_piece = 4 * Real.pi :=
by
  sorry

end pizza_volume_piece_l222_222448


namespace nearest_whole_number_l222_222672

theorem nearest_whole_number (x : ℝ) (h : x = 7263.4987234) : Int.floor (x + 0.5) = 7263 := by
  sorry

end nearest_whole_number_l222_222672


namespace octagon_area_in_square_l222_222856

def main : IO Unit :=
  IO.println s!"Hello, Lean!"

theorem octagon_area_in_square :
  ∀ (s : ℝ), ∀ (area_square : ℝ), ∀ (area_octagon : ℝ),
  (s * 4 = 160) →
  (s = 40) →
  (area_square = s * s) →
  (area_square = 1600) →
  (∃ (area_triangle : ℝ), area_triangle = 50 ∧ 8 * area_triangle = 400) →
  (area_octagon = area_square - 400) →
  (area_octagon = 1200) :=
by
  intros s area_square area_octagon h1 h2 h3 h4 h5 h6
  sorry

end octagon_area_in_square_l222_222856


namespace range_of_m_l222_222993

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ x^2 + (m - 1) * x + 1 = 0) → m ≤ -1 :=
by
  sorry

end range_of_m_l222_222993


namespace cost_of_lamp_and_flashlight_max_desk_lamps_l222_222671

-- Part 1: Cost of purchasing one desk lamp and one flashlight
theorem cost_of_lamp_and_flashlight (x : ℕ) (desk_lamp_cost flashlight_cost : ℕ) 
        (hx : desk_lamp_cost = x + 20)
        (hdesk : 400 = x / 2 * desk_lamp_cost)
        (hflash : 160 = x * flashlight_cost)
        (hnum : desk_lamp_cost = 2 * flashlight_cost) : 
        desk_lamp_cost = 25 ∧ flashlight_cost = 5 :=
sorry

-- Part 2: Maximum number of desk lamps Rongqing Company can purchase
theorem max_desk_lamps (a : ℕ) (desk_lamp_cost flashlight_cost : ℕ)
        (hc1 : desk_lamp_cost = 25)
        (hc2 : flashlight_cost = 5)
        (free_flashlight : ℕ := a) (required_flashlight : ℕ := 2 * a + 8) 
        (total_cost : ℕ := desk_lamp_cost * a + flashlight_cost * required_flashlight)
        (hcost : total_cost ≤ 670) :
        a ≤ 21 :=
sorry

end cost_of_lamp_and_flashlight_max_desk_lamps_l222_222671


namespace min_value_of_expression_is_6_l222_222791

noncomputable def min_value_of_expression (a b c : ℝ) : ℝ :=
  (a^2 + b^2) / c + (a^2 + c^2) / b + (b^2 + c^2) / a

theorem min_value_of_expression_is_6 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : min_value_of_expression a b c = 6 :=
by
  sorry

end min_value_of_expression_is_6_l222_222791


namespace is_inverse_g1_is_inverse_g2_l222_222431

noncomputable def f (x : ℝ) := 3 + 2*x - x^2

noncomputable def g1 (x : ℝ) := -1 + Real.sqrt (4 - x)
noncomputable def g2 (x : ℝ) := -1 - Real.sqrt (4 - x)

theorem is_inverse_g1 : ∀ x, f (g1 x) = x :=
by
  intro x
  sorry

theorem is_inverse_g2 : ∀ x, f (g2 x) = x :=
by
  intro x
  sorry

end is_inverse_g1_is_inverse_g2_l222_222431


namespace grazing_months_l222_222956

theorem grazing_months
    (total_rent : ℝ)
    (c_rent : ℝ)
    (a_oxen : ℕ)
    (a_months : ℕ)
    (b_oxen : ℕ)
    (c_oxen : ℕ)
    (c_months : ℕ)
    (b_months : ℝ)
    (total_oxen_months : ℝ) :
    total_rent = 140 ∧
    c_rent = 36 ∧
    a_oxen = 10 ∧
    a_months = 7 ∧
    b_oxen = 12 ∧
    c_oxen = 15 ∧
    c_months = 3 ∧
    c_rent / total_rent = (c_oxen * c_months) / total_oxen_months ∧
    total_oxen_months = (a_oxen * a_months) + (b_oxen * b_months) + (c_oxen * c_months)
    → b_months = 5 := by
    sorry

end grazing_months_l222_222956


namespace handshakes_total_count_l222_222424

/-
Statement:
There are 30 gremlins and 20 imps at a Regional Mischief Meet. Only half of the imps are willing to shake hands with each other.
All cooperative imps shake hands with each other. All imps shake hands with each gremlin. Gremlins shake hands with every
other gremlin as well as all the imps. Each pair of creatures shakes hands at most once. Prove that the total number of handshakes is 1080.
-/

theorem handshakes_total_count (gremlins imps cooperative_imps : ℕ)
  (H1 : gremlins = 30)
  (H2 : imps = 20)
  (H3 : cooperative_imps = imps / 2) :
  let handshakes_gremlins := gremlins * (gremlins - 1) / 2
  let handshakes_cooperative_imps := cooperative_imps * (cooperative_imps - 1) / 2
  let handshakes_imps_gremlins := imps * gremlins
  handshakes_gremlins + handshakes_cooperative_imps + handshakes_imps_gremlins = 1080 := 
by {
  sorry
}

end handshakes_total_count_l222_222424


namespace division_remainder_l222_222351

def p (x : ℝ) := x^5 + 2 * x^3 - x + 4
def a : ℝ := 2
def remainder : ℝ := 50

theorem division_remainder :
  p a = remainder :=
sorry

end division_remainder_l222_222351


namespace start_time_is_10_am_l222_222816

-- Definitions related to the problem statements
def distance_AB : ℝ := 600
def speed_A_to_B : ℝ := 70
def speed_B_to_A : ℝ := 80
def meeting_time : ℝ := 14  -- using 24-hour format, 2 pm as 14

-- Prove that the starting time is 10 am given the conditions
theorem start_time_is_10_am (t : ℝ) :
  (speed_A_to_B * t + speed_B_to_A * t = distance_AB) →
  (meeting_time - t = 10) :=
sorry

end start_time_is_10_am_l222_222816


namespace find_divisor_l222_222348

theorem find_divisor (n : ℕ) (d : ℕ) (h1 : n = 105829) (h2 : d = 10) (h3 : ∃ k, n - d = k * d) : d = 3 :=
by
  sorry

end find_divisor_l222_222348


namespace coffee_tea_overlap_l222_222880

theorem coffee_tea_overlap (c t : ℕ) (h_c : c = 80) (h_t : t = 70) : 
  ∃ (b : ℕ), b = 50 := 
by 
  sorry

end coffee_tea_overlap_l222_222880


namespace range_of_a_l222_222895

theorem range_of_a (a : ℝ) :
  (∀ x, (x < -1 ∨ x > 5) ∨ (a < x ∧ x < a + 8)) ↔ (-3 < a ∧ a < -1) :=
by
  sorry

end range_of_a_l222_222895


namespace arithmetic_sequence_problem_l222_222375

variable {a : ℕ → ℝ} {d : ℝ} -- Declare the sequence and common difference

-- Define the arithmetic sequence property
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
def given_conditions (a : ℕ → ℝ) (d : ℝ) : Prop :=
  a 5 + a 10 = 12 ∧ arithmetic_sequence a d

-- Main theorem statement
theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) 
  (h : given_conditions a d) :
  3 * a 7 + a 9 = 24 :=
sorry

end arithmetic_sequence_problem_l222_222375


namespace remainder_4_exp_4_exp_4_exp_4_mod_500_l222_222460

theorem remainder_4_exp_4_exp_4_exp_4_mod_500 :
  (4 ^ 4 ^ 4 ^ 4) % 500 = 36 :=
by
  sorry

end remainder_4_exp_4_exp_4_exp_4_mod_500_l222_222460


namespace ellipse_equation_midpoint_coordinates_l222_222659

noncomputable def ellipse_c := {x : ℝ × ℝ | (x.1^2 / 25) + (x.2^2 / 16) = 1}

theorem ellipse_equation (a b : ℝ) (h1 : a = 5) (h2 : b = 4) :
    ∀ x y : ℝ, x = 0 → y = 4 → (y^2 / b^2 = 1) ∧ (e = 3 / 5) → 
      (a > b ∧ b > 0 ∧ (x^2 / a^2) + (y^2 / b^2) = 1) := 
sorry

theorem midpoint_coordinates (a b : ℝ) (h1 : a = 5) (h2 : b = 4) :
    ∀ x y x1 x2 y1 y2 : ℝ, 
    (y = 4 / 5 * (x - 3)) → 
    (y1 = 4 / 5 * (x1 - 3)) ∧ (y2 = 4 / 5 * (x2 - 3)) ∧ 
    (x1^2 / a^2) + ((y1 - 3)^2 / b^2) = 1 ∧ (x2^2 / a^2) + ((y2 - 3)^2 / b^2) = 1 ∧ 
    (x1 + x2 = 3) → 
    ((x1 + x2) / 2 = 3 / 2) ∧ ((y1 + y2) / 2 = -6 / 5) := 
sorry

end ellipse_equation_midpoint_coordinates_l222_222659


namespace all_meet_standard_l222_222203

variables (A B C : Event) (P : ProbabilitySpace)

axiom P_A : P (A) = 0.8
axiom P_B : P (B) = 0.6
axiom P_C : P (C) = 0.5
axiom indep_ABC : IndepEvents [A, B, C] P

theorem all_meet_standard : P (A ∩ B ∩ C) = 0.24 := 
by {
  sorry
}

end all_meet_standard_l222_222203


namespace correct_calculation_l222_222699

theorem correct_calculation (x : ℝ) :
  (x / 5 + 16 = 58) → (x / 15 + 74 = 88) :=
by
  sorry

end correct_calculation_l222_222699


namespace cubic_polynomial_unique_l222_222180

-- Define the polynomial q(x)
def q (x : ℝ) : ℝ := -x^3 + 4*x^2 - 7*x - 4

-- State the conditions
theorem cubic_polynomial_unique :
  q 1 = -8 ∧
  q 2 = -10 ∧
  q 3 = -16 ∧
  q 4 = -32 :=
by
  -- Expand the function definition for the given inputs.
  -- Add these expansions in the proof part.
  sorry

end cubic_polynomial_unique_l222_222180


namespace MrsBrownCarrotYield_l222_222047

theorem MrsBrownCarrotYield :
  let pacesLength := 25
  let pacesWidth := 30
  let strideLength := 2.5
  let yieldPerSquareFoot := 0.5
  let lengthInFeet := pacesLength * strideLength
  let widthInFeet := pacesWidth * strideLength
  let area := lengthInFeet * widthInFeet
  let yield := area * yieldPerSquareFoot
  yield = 2343.75 :=
by
  sorry

end MrsBrownCarrotYield_l222_222047


namespace geom_series_sum_correct_l222_222729

noncomputable def geometric_series_sum (b1 r : ℚ) (n : ℕ) : ℚ :=
b1 * (1 - r ^ n) / (1 - r)

theorem geom_series_sum_correct :
  geometric_series_sum (3/4) (3/4) 15 = 3177905751 / 1073741824 := by
sorry

end geom_series_sum_correct_l222_222729


namespace sin_16_over_3_pi_l222_222297

theorem sin_16_over_3_pi : Real.sin (16 / 3 * Real.pi) = -Real.sqrt 3 / 2 := 
sorry

end sin_16_over_3_pi_l222_222297


namespace television_final_price_l222_222975

theorem television_final_price :
  let original_price := 1200
  let discount_percent := 0.30
  let tax_percent := 0.08
  let rebate := 50
  let discount := discount_percent * original_price
  let sale_price := original_price - discount
  let tax := tax_percent * sale_price
  let price_including_tax := sale_price + tax
  let final_amount := price_including_tax - rebate
  final_amount = 857.2 :=
by
{
  -- The proof would go here, but it's omitted as per instructions.
  sorry
}

end television_final_price_l222_222975


namespace butterfingers_count_l222_222046

theorem butterfingers_count (total_candy_bars : ℕ) (snickers : ℕ) (mars_bars : ℕ) (h_total : total_candy_bars = 12) (h_snickers : snickers = 3) (h_mars : mars_bars = 2) : 
  ∃ (butterfingers : ℕ), butterfingers = 7 :=
by
  sorry

end butterfingers_count_l222_222046


namespace marble_selection_sum_l222_222633

-- Constants representing the two sets of marbles
def myMarbles : List ℕ := List.range (8 + 1) -- marbles 1 to 8
def mathewsMarbles : List ℕ := List.range (20 + 1) -- marbles 1 to 20

-- Function to calculate the sum of all valid 3-combinations of my marbles
def valid_sums : Finset ℕ :=
  Finset.image List.sum (List.subsetsOfCard 3 myMarbles).toFinset

-- Define counting function for valid combinations
def count_valid_combinations (n : ℕ) : ℕ :=
  ((List.subsetsOfCard 3 myMarbles).filter (λ l, l.sum = n)).length

-- Define the total number of valid combinations
def total_valid_combinations : ℕ := 
  (Finset.range 22).sum count_valid_combinations

-- Theorem stating the question in formal terms
theorem marble_selection_sum :
  total_valid_combinations = ∑ (n in Finset.range 22), count_valid_combinations n := by
sorry

end marble_selection_sum_l222_222633


namespace expected_value_is_one_l222_222823

-- Definitions of the conditions
def num_black : ℕ := 3
def num_red : ℕ := 1

def score (ball : ℕ) : ℕ :=
  if ball = 0 then 2 else 0

def prob_black (num_drawn : ℕ) (remaining : ℕ) : ℚ :=
  (num_black - num_drawn) / remaining.to_rat

def prob_red (num_drawn : ℕ) (remaining : ℕ) : ℚ :=
  (num_red - num_drawn) / remaining.to_rat

def expected_score : ℚ :=
  (prob_black 0 (num_black + num_red) * prob_black 1 (num_black + num_red - 1) * score 1 +
   prob_black 0 (num_black + num_red) * prob_red 1 (num_black + num_red - 1) * score 0 +
   prob_red 0 (num_black + num_red) * prob_black 1 (num_black + num_red - 1) * score 0)

-- The goal
theorem expected_value_is_one : expected_score = 1 := by
  sorry

end expected_value_is_one_l222_222823


namespace negation_of_proposition_l222_222685

theorem negation_of_proposition (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - a * x + 1 < 0) ↔ (∀ x : ℝ, x^2 - a * x + 1 ≥ 0) :=
by sorry

end negation_of_proposition_l222_222685


namespace mod_computation_l222_222592

theorem mod_computation (a b n : ℕ) (h_modulus : n = 7) (h_a : a = 47) (h_b : b = 28) :
  (a^2023 - b^2023) % n = 5 :=
by
  sorry

end mod_computation_l222_222592


namespace like_terms_mn_eq_neg1_l222_222764

variable (m n : ℤ)

theorem like_terms_mn_eq_neg1
  (hx : m + 3 = 4)
  (hy : n + 3 = 1) :
  m + n = -1 :=
sorry

end like_terms_mn_eq_neg1_l222_222764


namespace simplify_eq_l222_222874

theorem simplify_eq {x y z : ℕ} (hx : x > 1) (hy : y > 1) (hz : z > 1) :
  9 * (x : ℝ) - ((10 / (2 * y) / 3 + 7 * z) * Real.pi) =
  9 * (x : ℝ) - (5 * Real.pi / (3 * y) + 7 * z * Real.pi) := by
  sorry

end simplify_eq_l222_222874


namespace new_mean_after_adding_constant_l222_222038

theorem new_mean_after_adding_constant (S : ℝ) (average : ℝ) (n : ℕ) (a : ℝ) :
  n = 15 → average = 40 → a = 15 → S = n * average → (S + n * a) / n = 55 :=
by
  intros hn haverage ha hS
  sorry

end new_mean_after_adding_constant_l222_222038


namespace quadratic_inequality_false_iff_l222_222219

open Real

theorem quadratic_inequality_false_iff (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 ≤ 0) ↔ -1 < a ∧ a < 3 :=
sorry

end quadratic_inequality_false_iff_l222_222219


namespace breakfast_calories_l222_222512

theorem breakfast_calories : ∀ (planned_calories : ℕ) (B : ℕ),
  planned_calories < 1800 →
  B + 900 + 1100 = planned_calories + 600 →
  B = 400 :=
by
  intros
  sorry

end breakfast_calories_l222_222512


namespace stadium_fee_difference_l222_222554

theorem stadium_fee_difference :
  let capacity := 2000
  let entry_fee := 20
  let full_fees := capacity * entry_fee
  let three_quarters_fees := (capacity * 3 / 4) * entry_fee
  full_fees - three_quarters_fees = 10000 :=
by
  sorry

end stadium_fee_difference_l222_222554


namespace total_selling_price_l222_222151

def selling_price_A (purchase_price_A : ℝ) : ℝ :=
  purchase_price_A - (0.15 * purchase_price_A)

def selling_price_B (purchase_price_B : ℝ) : ℝ :=
  purchase_price_B + (0.10 * purchase_price_B)

def selling_price_C (purchase_price_C : ℝ) : ℝ :=
  purchase_price_C - (0.05 * purchase_price_C)

theorem total_selling_price 
  (purchase_price_A : ℝ)
  (purchase_price_B : ℝ)
  (purchase_price_C : ℝ)
  (loss_A : ℝ := 0.15)
  (gain_B : ℝ := 0.10)
  (loss_C : ℝ := 0.05)
  (total_price := selling_price_A purchase_price_A + selling_price_B purchase_price_B + selling_price_C purchase_price_C) :
  purchase_price_A = 1400 → purchase_price_B = 2500 → purchase_price_C = 3200 →
  total_price = 6980 :=
by sorry

end total_selling_price_l222_222151


namespace container_volume_ratio_l222_222163

variable (A B C : ℝ)

theorem container_volume_ratio (h1 : (4 / 5) * A = (3 / 5) * B) (h2 : (3 / 5) * B = (3 / 4) * C) :
  A / C = 15 / 16 :=
sorry

end container_volume_ratio_l222_222163


namespace value_of_f_inv_sum_l222_222662

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def f_inv (y : ℝ) : ℝ := sorry

axiom f_inv_is_inverse : ∀ x : ℝ, f (f_inv x) = x ∧ f_inv (f x) = x
axiom f_condition : ∀ x : ℝ, f x + f (-x) = 2

theorem value_of_f_inv_sum (x : ℝ) : f_inv (2008 - x) + f_inv (x - 2006) = 0 :=
sorry

end value_of_f_inv_sum_l222_222662


namespace police_coverage_l222_222378

-- Define the intersections and streets
inductive Intersection : Type
| A | B | C | D | E | F | G | H | I | J | K

open Intersection

-- Define each street as a set of intersections
def horizontal_streets : List (List Intersection) :=
  [[A, B, C, D], [E, F, G], [H, I, J, K]]

def vertical_streets : List (List Intersection) :=
  [[A, E, H], [B, F, I], [D, G, J]]

def diagonal_streets : List (List Intersection) :=
  [[H, F, C], [C, G, K]]

def all_streets : List (List Intersection) :=
  horizontal_streets ++ vertical_streets ++ diagonal_streets

-- Define the set of police officers' placements
def police_officers : List Intersection := [B, G, H]

-- Check if each street is covered by at least one police officer
def is_covered (street : List Intersection) (officers : List Intersection) : Prop :=
  ∃ i, i ∈ street ∧ i ∈ officers

-- Define the proof problem statement
theorem police_coverage :
  ∀ street ∈ all_streets, is_covered street police_officers :=
by sorry

end police_coverage_l222_222378


namespace simplify_and_evaluate_expression_l222_222256

theorem simplify_and_evaluate_expression (x y : ℝ) (h_x : x = -2) (h_y : y = 1) :
  (((2 * x - (1/2) * y)^2 - ((-y + 2 * x) * (2 * x + y)) + y * (x^2 * y - (5/4) * y)) / x) = -4 :=
by
  sorry

end simplify_and_evaluate_expression_l222_222256


namespace number_of_human_family_members_l222_222504

-- Definitions for the problem
def num_birds := 4
def num_dogs := 3
def num_cats := 18
def bird_feet := 2
def dog_feet := 4
def cat_feet := 4
def human_feet := 2
def human_heads := 1

def animal_feet := (num_birds * bird_feet) + (num_dogs * dog_feet) + (num_cats * cat_feet)
def animal_heads := num_birds + num_dogs + num_cats

def total_feet (H : Nat) := animal_feet + (H * human_feet)
def total_heads (H : Nat) := animal_heads + (H * human_heads)

-- The problem statement translated to Lean
theorem number_of_human_family_members (H : Nat) : (total_feet H) = (total_heads H) + 74 → H = 7 :=
by
  sorry

end number_of_human_family_members_l222_222504


namespace rounding_example_l222_222926

theorem rounding_example (x : ℝ) (h : x = 8899.50241201) : round x = 8900 :=
by
  sorry

end rounding_example_l222_222926


namespace scientific_notation_l222_222221

theorem scientific_notation :
  686530000 = 6.8653 * 10^8 :=
sorry

end scientific_notation_l222_222221


namespace milk_transfer_proof_l222_222455

theorem milk_transfer_proof :
  ∀ (A B C x : ℝ), 
  A = 1232 →
  B = A - 0.625 * A → 
  C = A - B → 
  B + x = C - x → 
  x = 154 :=
by
  intros A B C x hA hB hC hEqual
  sorry

end milk_transfer_proof_l222_222455


namespace find_x_l222_222417

theorem find_x :
  ∃ x : Real, abs (x - 0.052) < 1e-3 ∧
  (0.02^2 + 0.52^2 + 0.035^2) / (0.002^2 + x^2 + 0.0035^2) = 100 :=
by
  sorry

end find_x_l222_222417


namespace Geli_pushups_and_runs_l222_222741

def initial_pushups : ℕ := 10
def increment_pushups : ℕ := 5
def workouts_per_week : ℕ := 3
def weeks_in_a_month : ℕ := 4
def pushups_per_mile_run : ℕ := 30

def workout_days_in_month : ℕ := workouts_per_week * weeks_in_a_month

def pushups_on_day (day : ℕ) : ℕ := initial_pushups + (day - 1) * increment_pushups

def total_pushups : ℕ := (workout_days_in_month / 2) * (initial_pushups + pushups_on_day workout_days_in_month)

def one_mile_runs (total_pushups : ℕ) : ℕ := total_pushups / pushups_per_mile_run

theorem Geli_pushups_and_runs :
  total_pushups = 450 ∧ one_mile_runs total_pushups = 15 :=
by
  -- Here, we should prove total_pushups = 450 and one_mile_runs total_pushups = 15.
  sorry

end Geli_pushups_and_runs_l222_222741


namespace bus_speed_incl_stoppages_l222_222989

theorem bus_speed_incl_stoppages (v_excl : ℝ) (minutes_stopped : ℝ) :
  v_excl = 64 → minutes_stopped = 13.125 →
  v_excl - (v_excl * (minutes_stopped / 60)) = 50 :=
by
  intro v_excl_eq minutes_stopped_eq
  rw [v_excl_eq, minutes_stopped_eq]
  have hours_stopped : ℝ := 13.125 / 60
  have distance_lost : ℝ := 64 * hours_stopped
  have v_incl := 64 - distance_lost
  sorry

end bus_speed_incl_stoppages_l222_222989


namespace value_of_fraction_zero_l222_222015

theorem value_of_fraction_zero (x : ℝ) (h1 : x^2 - 1 = 0) (h2 : 1 - x ≠ 0) : x = -1 :=
by
  sorry

end value_of_fraction_zero_l222_222015


namespace train_cross_bridge_time_l222_222897

/-
  Define the given conditions:
  - Length of the train (lt): 200 m
  - Speed of the train (st_kmh): 72 km/hr
  - Length of the bridge (lb): 132 m
-/

namespace TrainProblem

def length_of_train : ℕ := 200
def speed_of_train_kmh : ℕ := 72
def length_of_bridge : ℕ := 132

/-
  Convert speed from km/hr to m/s
-/
def speed_of_train_ms : ℕ := speed_of_train_kmh * 1000 / 3600

/-
  Calculate total distance to be traveled (train length + bridge length).
-/
def total_distance : ℕ := length_of_train + length_of_bridge

/-
  Use the formula Time = Distance / Speed
-/
def time_to_cross_bridge : ℚ := total_distance / speed_of_train_ms

theorem train_cross_bridge_time : 
  (length_of_train = 200) →
  (speed_of_train_kmh = 72) →
  (length_of_bridge = 132) →
  time_to_cross_bridge = 16.6 :=
by
  intros lt st lb
  sorry

end TrainProblem

end train_cross_bridge_time_l222_222897


namespace Joan_attended_games_l222_222648

def total_games : ℕ := 864
def games_missed_by_Joan : ℕ := 469
def games_attended_by_Joan : ℕ := total_games - games_missed_by_Joan

theorem Joan_attended_games : games_attended_by_Joan = 395 := 
by 
  -- Proof omitted
  sorry

end Joan_attended_games_l222_222648


namespace smallest_non_factor_l222_222103

-- Definitions of the conditions
def isFactorOf (m n : ℕ) : Prop := n % m = 0
def distinct (a b : ℕ) : Prop := a ≠ b

-- The main statement we need to prove.
theorem smallest_non_factor (a b : ℕ) (h_distinct : distinct a b)
  (h_a_factor : isFactorOf a 48) (h_b_factor : isFactorOf b 48)
  (h_not_factor : ¬ isFactorOf (a * b) 48) :
  a * b = 32 := 
sorry

end smallest_non_factor_l222_222103


namespace fraction_condition_l222_222746

theorem fraction_condition (x : ℝ) (h₁ : x > 1) (h₂ : 1 / x < 1) : false :=
sorry

end fraction_condition_l222_222746


namespace students_with_green_eyes_l222_222018

-- Define the variables and given conditions
def total_students : ℕ := 36
def students_with_red_hair (y : ℕ) : ℕ := 3 * y
def students_with_both : ℕ := 12
def students_with_neither : ℕ := 4

-- Define the proof statement
theorem students_with_green_eyes :
  ∃ y : ℕ, 
  (students_with_red_hair y + y - students_with_both + students_with_neither = total_students) ∧
  (students_with_red_hair y ≠ y) → y = 11 :=
by
  sorry

end students_with_green_eyes_l222_222018


namespace total_number_of_feet_l222_222441

theorem total_number_of_feet 
  (H C F : ℕ)
  (h1 : H + C = 44)
  (h2 : H = 24)
  (h3 : F = 2 * H + 4 * C) : 
  F = 128 :=
by
  sorry

end total_number_of_feet_l222_222441


namespace not_necessarily_periodic_l222_222742

-- Define the conditions of the problem
noncomputable def a : ℕ → ℕ := sorry
noncomputable def t : ℕ → ℕ := sorry
axiom h_t : ∀ k : ℕ, ∃ t_k : ℕ, ∀ n : ℕ, a (k + n * t_k) = a k

-- The theorem stating that the sequence is not necessarily periodic
theorem not_necessarily_periodic : ¬ ∃ T : ℕ, ∀ k : ℕ, a (k + T) = a k := sorry

end not_necessarily_periodic_l222_222742


namespace total_money_divided_l222_222301

theorem total_money_divided (A B C T : ℝ) 
    (h1 : A = (2/5) * (B + C)) 
    (h2 : B = (1/5) * (A + C)) 
    (h3 : A = 600) :
    T = A + B + C →
    T = 2100 :=
by 
  sorry

end total_money_divided_l222_222301


namespace positive_number_property_l222_222970

theorem positive_number_property (x : ℝ) (h_pos : x > 0) (h_property : 0.01 * x * x = 4) : x = 20 :=
sorry

end positive_number_property_l222_222970


namespace find_smaller_integer_l222_222278

theorem find_smaller_integer : ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ y = x + 8 ∧ x * y = 80 ∧ x = 2 :=
by
  sorry

end find_smaller_integer_l222_222278


namespace total_jellybeans_needed_l222_222031

def large_glass_jellybeans : ℕ := 50
def small_glass_jellybeans : ℕ := large_glass_jellybeans / 2
def num_large_glasses : ℕ := 5
def num_small_glasses : ℕ := 3

theorem total_jellybeans_needed : 
  (num_large_glasses * large_glass_jellybeans) + (num_small_glasses * small_glass_jellybeans) = 325 := 
by
  sorry

end total_jellybeans_needed_l222_222031


namespace initial_deck_card_count_l222_222304

-- Define the initial conditions
def initial_red_probability (r b : ℕ) : Prop := r * 4 = r + b
def added_black_probability (r b : ℕ) : Prop := r * 5 = 4 * r + 6

theorem initial_deck_card_count (r b : ℕ) (h1 : initial_red_probability r b) (h2 : added_black_probability r b) : r + b = 24 := 
by sorry

end initial_deck_card_count_l222_222304


namespace fly_in_box_maximum_path_length_l222_222715

theorem fly_in_box_maximum_path_length :
  let side1 := 1
  let side2 := Real.sqrt 2
  let side3 := Real.sqrt 3
  let space_diagonal := Real.sqrt (side1^2 + side2^2 + side3^2)
  let face_diagonal1 := Real.sqrt (side1^2 + side2^2)
  let face_diagonal2 := Real.sqrt (side1^2 + side3^2)
  let face_diagonal3 := Real.sqrt (side2^2 + side3^2)
  (4 * space_diagonal + 2 * face_diagonal3) = 4 * Real.sqrt 6 + 2 * Real.sqrt 5 :=
by
  sorry

end fly_in_box_maximum_path_length_l222_222715


namespace externally_tangent_circles_proof_l222_222563

noncomputable def externally_tangent_circles (r r' : ℝ) (φ : ℝ) : Prop :=
  (r + r')^2 * Real.sin φ = 4 * (r - r') * Real.sqrt (r * r')

theorem externally_tangent_circles_proof (r r' φ : ℝ) 
  (h1: r > 0) (h2: r' > 0) (h3: φ ≥ 0 ∧ φ ≤ π) : 
  externally_tangent_circles r r' φ :=
sorry

end externally_tangent_circles_proof_l222_222563


namespace length_of_de_equals_eight_l222_222838

theorem length_of_de_equals_eight
  (a b c d e : ℝ)
  (h1 : a < b)
  (h2 : b < c)
  (h3 : c < d)
  (h4 : d < e)
  (bc : c - b = 3 * (d - c))
  (ab : b - a = 5)
  (ac : c - a = 11)
  (ae : e - a = 21) :
  e - d = 8 := by
  sorry

end length_of_de_equals_eight_l222_222838


namespace correct_relation_is_identity_l222_222583

theorem correct_relation_is_identity : 0 = 0 :=
by {
  -- Skipping proof steps as only statement is required
  sorry
}

end correct_relation_is_identity_l222_222583


namespace marcella_shoes_l222_222294

theorem marcella_shoes :
  ∀ (original_pairs lost_shoes : ℕ), original_pairs = 27 → lost_shoes = 9 → 
  ∃ (remaining_pairs : ℕ), remaining_pairs = 18 ∧ remaining_pairs ≤ original_pairs - lost_shoes / 2 :=
by
  intros original_pairs lost_shoes h1 h2
  use 18
  constructor
  . exact rfl
  . sorry

end marcella_shoes_l222_222294


namespace water_tank_capacity_l222_222560

theorem water_tank_capacity (C : ℝ) :
  0.4 * C - 0.1 * C = 36 → C = 120 :=
by sorry

end water_tank_capacity_l222_222560


namespace smallest_int_ends_in_3_div_by_11_l222_222123

theorem smallest_int_ends_in_3_div_by_11 :
  ∃ k : ℕ, k > 0 ∧ k % 10 = 3 ∧ k % 11 = 0 ∧ k = 33 :=
by {
  sorry
}

end smallest_int_ends_in_3_div_by_11_l222_222123


namespace players_started_first_half_l222_222724

variable (total_players : Nat)
variable (first_half_substitutions : Nat)
variable (second_half_substitutions : Nat)
variable (players_not_playing : Nat)

theorem players_started_first_half :
  total_players = 24 →
  first_half_substitutions = 2 →
  second_half_substitutions = 2 * first_half_substitutions →
  players_not_playing = 7 →
  let total_substitutions := first_half_substitutions + second_half_substitutions 
  let players_played := total_players - players_not_playing
  ∃ S, S + total_substitutions = players_played ∧ S = 11 := 
by
  sorry

end players_started_first_half_l222_222724


namespace sugar_amount_indeterminate_l222_222232

-- Define the variables and conditions
variable (cups_of_flour_needed : ℕ) (cups_of_sugar_needed : ℕ)
variable (cups_of_flour_put_in : ℕ) (cups_of_flour_to_add : ℕ)

-- Conditions
axiom H1 : cups_of_flour_needed = 8
axiom H2 : cups_of_flour_put_in = 4
axiom H3 : cups_of_flour_to_add = 4

-- Problem statement: Prove that the amount of sugar cannot be determined
theorem sugar_amount_indeterminate (h : cups_of_sugar_needed > 0) :
  cups_of_flour_needed = 8 → cups_of_flour_put_in = 4 → cups_of_flour_to_add = 4 → cups_of_sugar_needed > 0 :=
by
  intros
  sorry

end sugar_amount_indeterminate_l222_222232


namespace find_fourth_number_l222_222069

theorem find_fourth_number (a : ℕ → ℕ) 
  (h1 : ∀ n, n ≥ 2 → a n = a (n - 1) + a (n - 2)) 
  (h2 : a 6 = 42) 
  (h3 : a 8 = 110) : 
  a 3 = 10 := 
sorry

end find_fourth_number_l222_222069


namespace exists_positive_b_l222_222234

theorem exists_positive_b (m p : ℕ) (hm : 0 < m) (hp : Prime p)
  (h1 : m^2 ≡ 2 [MOD p])
  (ha : ∃ a : ℕ, 0 < a ∧ a^2 ≡ 2 - m [MOD p]) :
  ∃ b : ℕ, 0 < b ∧ b^2 ≡ m + 2 [MOD p] := 
  sorry

end exists_positive_b_l222_222234


namespace apples_to_pears_l222_222773

theorem apples_to_pears (a o p : ℕ) 
  (h1 : 10 * a = 5 * o) 
  (h2 : 3 * o = 4 * p) : 
  (20 * a) = 40 / 3 * p :=
sorry

end apples_to_pears_l222_222773


namespace decimal_to_binary_thirteen_l222_222467

theorem decimal_to_binary_thirteen : (13 : ℕ) = 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 :=
by
  sorry

end decimal_to_binary_thirteen_l222_222467


namespace largest_number_among_selected_students_l222_222641

def total_students := 80

def smallest_numbers (x y : ℕ) : Prop :=
  x = 6 ∧ y = 14

noncomputable def selected_students (n : ℕ) : ℕ :=
  6 + (n - 1) * 8

theorem largest_number_among_selected_students :
  ∀ (x y : ℕ), smallest_numbers x y → (selected_students 10 = 78) :=
by
  intros x y h
  rw [smallest_numbers] at h
  have h1 : x = 6 := h.1
  have h2 : y = 14 := h.2
  exact rfl

#check largest_number_among_selected_students

end largest_number_among_selected_students_l222_222641


namespace sqrt_of_26244_div_by_100_l222_222635

theorem sqrt_of_26244_div_by_100 (h : Real.sqrt 262.44 = 16.2) : Real.sqrt 2.6244 = 1.62 :=
sorry

end sqrt_of_26244_div_by_100_l222_222635


namespace total_cost_price_l222_222578

theorem total_cost_price (SP1 SP2 SP3 : ℝ) (P1 P2 P3 : ℝ) 
  (h1 : SP1 = 120) (h2 : SP2 = 150) (h3 : SP3 = 200)
  (h4 : P1 = 0.20) (h5 : P2 = 0.25) (h6 : P3 = 0.10) : (SP1 / (1 + P1) + SP2 / (1 + P2) + SP3 / (1 + P3) = 401.82) :=
by
  sorry

end total_cost_price_l222_222578


namespace stadium_fee_difference_l222_222555

theorem stadium_fee_difference :
  let capacity := 2000
  let entry_fee := 20
  let full_fees := capacity * entry_fee
  let three_quarters_fees := (capacity * 3 / 4) * entry_fee
  full_fees - three_quarters_fees = 10000 :=
by
  sorry

end stadium_fee_difference_l222_222555


namespace product_of_w_and_z_l222_222022

variable (EF FG GH HE : ℕ)
variable (w z : ℕ)

-- Conditions from the problem
def parallelogram_conditions : Prop :=
  EF = 42 ∧ FG = 4 * z^3 ∧ GH = 3 * w + 6 ∧ HE = 32 ∧ EF = GH ∧ FG = HE

-- The proof problem proving the requested product given the conditions
theorem product_of_w_and_z (h : parallelogram_conditions EF FG GH HE w z) : (w * z) = 24 :=
by
  sorry

end product_of_w_and_z_l222_222022


namespace option_A_option_D_l222_222235

variable {a : ℕ → ℤ} -- The arithmetic sequence
variable {S : ℕ → ℤ} -- Sum of the first n terms
variable {a1 d : ℤ} -- First term and common difference

-- Conditions for arithmetic sequence
axiom a_n (n : ℕ) : a n = a1 + ↑(n-1) * d
axiom S_n (n : ℕ) : S n = n * a1 + (n * (n - 1) / 2) * d
axiom condition : a 4 + 2 * a 8 = a 6

theorem option_A : a 7 = 0 :=
by
  -- Proof to be done
  sorry

theorem option_D : S 13 = 0 :=
by
  -- Proof to be done
  sorry

end option_A_option_D_l222_222235


namespace percentage_calculation_l222_222712

theorem percentage_calculation :
  ∀ (P : ℝ),
  (0.3 * 0.5 * 4400 = 99) →
  (P * 4400 = 99) →
  P = 0.0225 :=
by
  intros P condition1 condition2
  -- From the given conditions, it follows directly
  sorry

end percentage_calculation_l222_222712


namespace value_of_b_l222_222678

theorem value_of_b (b : ℝ) (h1 : 1/2 * (b / 3) * b = 6) (h2 : b ≥ 0) : b = 6 := sorry

end value_of_b_l222_222678


namespace max_value_is_one_l222_222388

noncomputable def max_value (x y z : ℝ) : ℝ :=
  (x^2 - 2 * x * y + y^2) * (x^2 - 2 * x * z + z^2) * (y^2 - 2 * y * z + z^2)

theorem max_value_is_one :
  ∀ (x y z : ℝ), 0 ≤ x → 0 ≤ y → 0 ≤ z → x + y + z = 3 →
  max_value x y z ≤ 1 :=
by sorry

end max_value_is_one_l222_222388


namespace sum_arithmetic_sequence_l222_222626

def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_arithmetic_sequence {a : ℕ → ℝ} 
  (h_arith : arithmetic_seq a)
  (h1 : a 2^2 + a 7^2 + 2 * a 2 * a 7 = 9)
  (h2 : ∀ n, a n < 0) : 
  S₁₀ = -15 :=
by
  sorry

end sum_arithmetic_sequence_l222_222626


namespace y_intercept_of_line_l222_222362

theorem y_intercept_of_line : 
  (∃ t : ℝ, 4 - 4 * t = 0) → (∃ y : ℝ, y = -2 + 3 * 1) := 
by
  sorry

end y_intercept_of_line_l222_222362


namespace gravel_cost_calculation_l222_222767

def cubicYardToCubicFoot : ℕ := 27
def costPerCubicFoot : ℕ := 8
def volumeInCubicYards : ℕ := 8

theorem gravel_cost_calculation : 
  (volumeInCubicYards * cubicYardToCubicFoot * costPerCubicFoot) = 1728 := 
by
  -- This is just a placeholder to ensure the statement is syntactically correct.
  sorry

end gravel_cost_calculation_l222_222767


namespace smallest_non_factor_product_l222_222098

theorem smallest_non_factor_product (a b : ℕ) (h1 : a ≠ b) (h2 : a ∣ 48) (h3 : b ∣ 48) (h4 : ¬ (a * b ∣ 48)) : a * b = 18 :=
by
  -- proof intentionally omitted
  sorry

end smallest_non_factor_product_l222_222098


namespace point_P_inside_circle_l222_222627

theorem point_P_inside_circle
  (a b c : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : a > b)
  (e : ℝ)
  (h4 : e = 1 / 2)
  (x1 x2 : ℝ)
  (hx1 : a * x1 ^ 2 + b * x1 - c = 0)
  (hx2 : a * x2 ^ 2 + b * x2 - c = 0) :
  x1 ^ 2 + x2 ^ 2 < 2 :=
by
  sorry

end point_P_inside_circle_l222_222627


namespace range_of_omega_l222_222755

noncomputable def function_with_highest_points (ω : ℝ) (x : ℝ) : ℝ :=
  2 * Real.sin (ω * x + Real.pi / 4)

theorem range_of_omega (ω : ℝ) (hω : ω > 0)
  (h : ∀ x ∈ Set.Icc 0 1, 2 * Real.sin (ω * x + Real.pi / 4) = 2) :
  Set.Icc (17 * Real.pi / 4) (25 * Real.pi / 4) :=
by
  sorry

end range_of_omega_l222_222755


namespace solution_to_system_l222_222673

def system_of_equations (x y : ℝ) : Prop := (x^2 - 9 * y^2 = 36) ∧ (3 * x + y = 6)

theorem solution_to_system : 
  {p : ℝ × ℝ | system_of_equations p.1 p.2} = { (12 / 5, -6 / 5), (3, -3) } := 
by sorry

end solution_to_system_l222_222673


namespace find_a4_l222_222050

open Nat

def sequence (a : Nat → Nat) :=
  ∀ n, n ≥ 3 → a n = a (n - 1) + a (n - 2)

theorem find_a4 (a : ℕ → ℕ)
  (h_seq : sequence a)
  (h_a7 : a 7 = 42)
  (h_a9 : a 9 = 110) :
  a 4 = 10 :=
by
  sorry

end find_a4_l222_222050


namespace maximum_automobiles_on_ferry_l222_222839

-- Define the conditions
def ferry_capacity_tons : ℕ := 50
def automobile_min_weight : ℕ := 1600
def automobile_max_weight : ℕ := 3200

-- Define the conversion factor from tons to pounds
def ton_to_pound : ℕ := 2000

-- Define the converted ferry capacity in pounds
def ferry_capacity_pounds := ferry_capacity_tons * ton_to_pound

-- Proof statement
theorem maximum_automobiles_on_ferry : 
  ferry_capacity_pounds / automobile_min_weight = 62 :=
by
  -- Given: ferry capacity is 50 tons and 1 ton = 2000 pounds
  -- Therefore, ferry capacity in pounds is 50 * 2000 = 100000 pounds
  -- The weight of the lightest automobile is 1600 pounds
  -- Maximum number of automobiles = 100000 / 1600 = 62.5
  -- Rounding down to the nearest whole number gives 62
  sorry  -- Proof steps would be filled here

end maximum_automobiles_on_ferry_l222_222839


namespace solution_set_f_gt_5_range_m_f_ge_abs_2m1_l222_222001

noncomputable def f (x : ℝ) : ℝ := abs (2 * x - 1) + abs (x + 3)

theorem solution_set_f_gt_5 :
  {x : ℝ | f x > 5} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 1} :=
by sorry

theorem range_m_f_ge_abs_2m1 :
  (∀ x : ℝ, f x ≥ abs (2 * m + 1)) ↔ -9/4 ≤ m ∧ m ≤ 5/4 :=
by sorry

end solution_set_f_gt_5_range_m_f_ge_abs_2m1_l222_222001


namespace find_a_l222_222857

-- Conditions as definitions:
variable (a : ℝ) (b : ℝ)
variable (A : ℝ × ℝ := (0, 0)) (B : ℝ × ℝ := (a, 0)) (C : ℝ × ℝ := (0, b))
noncomputable def area (a b : ℝ) : ℝ := (1 / 2) * a * b

-- Given conditions:
axiom h1 : b = 4
axiom h2 : area a b = 28
axiom h3 : a > 0

-- The proof goal:
theorem find_a : a = 14 := by
  -- proof omitted
  sorry

end find_a_l222_222857


namespace worker_efficiency_l222_222582

theorem worker_efficiency (Wq : ℝ) (x : ℝ) : 
  (1.4 * (1 / x) = 1 / (1.4 * x)) → 
  (14 * (1 / x + 1 / (1.4 * x)) = 1) → 
  x = 24 :=
by
  sorry

end worker_efficiency_l222_222582


namespace tangent_slope_at_point_552_32_l222_222834

noncomputable def slope_of_tangent_at_point (cx cy px py : ℚ) : ℚ :=
if py - cy = 0 then 
  0 
else 
  (px - cx) / (py - cy)

theorem tangent_slope_at_point_552_32 : slope_of_tangent_at_point 3 2 5 5 = -2 / 3 :=
by
  -- Conditions from problem
  have h1 : slope_of_tangent_at_point 3 2 5 5 = -2 / 3 := 
    sorry
  
  exact h1

end tangent_slope_at_point_552_32_l222_222834


namespace correct_weight_misread_l222_222815

theorem correct_weight_misread (initial_avg correct_avg : ℝ) (num_boys : ℕ) (misread_weight : ℝ)
  (h_initial : initial_avg = 58.4) (h_correct : correct_avg = 58.85) (h_num_boys : num_boys = 20)
  (h_misread_weight : misread_weight = 56) :
  ∃ x : ℝ, x = 65 :=
by
  sorry

end correct_weight_misread_l222_222815


namespace find_x_l222_222972

theorem find_x (x : ℝ) (hx_pos : 0 < x) (h: (x / 100) * x = 4) : x = 20 := by
  sorry

end find_x_l222_222972


namespace airplane_average_speed_l222_222319

-- Define the conditions
def miles_to_kilometers (miles : ℕ) : ℝ :=
  miles * 1.60934

def distance_miles : ℕ := 1584
def time_hours : ℕ := 24

-- Define the problem to prove
theorem airplane_average_speed : 
  (miles_to_kilometers distance_miles) / (time_hours : ℝ) = 106.24 :=
by
  sorry

end airplane_average_speed_l222_222319


namespace larger_number_is_22_l222_222919

theorem larger_number_is_22 (x y : ℕ) (h1 : y = x + 10) (h2 : x + y = 34) : y = 22 :=
by
  sorry

end larger_number_is_22_l222_222919


namespace sum_series_evaluation_l222_222342

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, (if k = 0 then 0 else (2 * k) / (4 : ℝ) ^ k)

theorem sum_series_evaluation : sum_series = 8 / 9 := by
  sorry

end sum_series_evaluation_l222_222342


namespace pizza_volume_one_piece_l222_222443

theorem pizza_volume_one_piece :
  ∀ (h t: ℝ) (d: ℝ) (n: ℕ), d = 16 → t = 1/2 → n = 8 → h = 8 → 
  ( (π * (d / 2)^2 * t) / n = 4 * π ) :=
by 
  intros h t d n hd ht hn hh
  sorry

end pizza_volume_one_piece_l222_222443


namespace two_cos_45_eq_sqrt_2_l222_222413

theorem two_cos_45_eq_sqrt_2 : 2 * Real.cos (pi / 4) = Real.sqrt 2 := by
  sorry

end two_cos_45_eq_sqrt_2_l222_222413


namespace caps_difference_l222_222254

theorem caps_difference (Billie_caps Sammy_caps : ℕ) (Janine_caps := 3 * Billie_caps)
  (Billie_has : Billie_caps = 2) (Sammy_has : Sammy_caps = 8) :
  Sammy_caps - Janine_caps = 2 := by
  -- proof goes here
  sorry

end caps_difference_l222_222254


namespace marble_selection_probability_l222_222567

theorem marble_selection_probability :
  let total_marbles := 9
  let selected_marbles := 4
  let total_ways := Nat.choose total_marbles selected_marbles
  let red_marbles := 3
  let blue_marbles := 3
  let green_marbles := 3
  let ways_one_red := Nat.choose red_marbles 1
  let ways_two_blue := Nat.choose blue_marbles 2
  let ways_one_green := Nat.choose green_marbles 1
  let favorable_outcomes := ways_one_red * ways_two_blue * ways_one_green
  (favorable_outcomes : ℚ) / total_ways = 3 / 14 :=
by
  sorry

end marble_selection_probability_l222_222567


namespace ship_length_l222_222338

theorem ship_length (E S L : ℕ) (h1 : 150 * E = L + 150 * S) (h2 : 90 * E = L - 90 * S) : 
  L = 24 :=
by
  sorry

end ship_length_l222_222338


namespace find_length_AD_l222_222230

-- Given data and conditions
def triangle_ABC (A B C D : Type) : Prop := sorry
def angle_bisector_AD (A B C D : Type) : Prop := sorry
def length_BD : ℝ := 40
def length_BC : ℝ := 45
def length_AC : ℝ := 36

-- Prove that AD = 320 units
theorem find_length_AD (A B C D : Type)
  (h1 : triangle_ABC A B C D)
  (h2 : angle_bisector_AD A B C D)
  (h3 : length_BD = 40)
  (h4 : length_BC = 45)
  (h5 : length_AC = 36) :
  ∃ x : ℝ, x = 320 :=
sorry

end find_length_AD_l222_222230


namespace find_value_of_function_l222_222754

theorem find_value_of_function (f : ℝ → ℝ) (h : ∀ x : ℝ, 2 * f x + f (-x) = 3 * x + 2) : 
  f 2 = 20 / 3 :=
sorry

end find_value_of_function_l222_222754


namespace total_running_duration_l222_222145

-- Conditions
def speed1 := 15 -- speed during the first part in mph
def time1 := 3 -- time during the first part in hours
def speed2 := 19 -- speed during the second part in mph
def distance2 := 190 -- distance during the second part in miles

-- Initialize
def distance1 := speed1 * time1 -- distance covered in the first part in miles

def time2 := distance2 / speed2 -- time to cover the distance in the second part in hours

-- Total duration
def total_duration := time1 + time2

-- Proof statement
theorem total_running_duration : total_duration = 13 :=
by
  sorry

end total_running_duration_l222_222145


namespace number_of_male_students_l222_222262

noncomputable def avg_all : ℝ := 90
noncomputable def avg_male : ℝ := 84
noncomputable def avg_female : ℝ := 92
noncomputable def count_female : ℕ := 24

theorem number_of_male_students (M : ℕ) (T : ℕ) :
  avg_all * (M + count_female) = avg_male * M + avg_female * count_female →
  T = M + count_female →
  M = 8 :=
by
  intro h_avg h_count
  sorry

end number_of_male_students_l222_222262


namespace work_rate_problem_l222_222962

theorem work_rate_problem :
  ∃ (x : ℝ), 
    (0 < x) ∧ 
    (10 * (1 / x + 1 / 40) = 0.5833333333333334) ∧ 
    (x = 30) :=
by
  sorry

end work_rate_problem_l222_222962


namespace red_ball_second_draw_probability_l222_222999

theorem red_ball_second_draw_probability :
  let total_balls := 20
  let red_balls := 10
  let black_balls := 10
  let p_first_red := red_balls / total_balls
  let p_first_black := black_balls / total_balls
  let p_second_red_given_first_red := (red_balls - 1) / (total_balls - 1)
  let p_second_red_given_first_black := red_balls / (total_balls - 1)
  p_first_red * p_second_red_given_first_red + p_first_black * p_second_red_given_first_black = 1 / 2 :=
by
  -- proof steps go here
  sorry

end red_ball_second_draw_probability_l222_222999


namespace find_a4_l222_222049

open Nat

def sequence (a : Nat → Nat) :=
  ∀ n, n ≥ 3 → a n = a (n - 1) + a (n - 2)

theorem find_a4 (a : ℕ → ℕ)
  (h_seq : sequence a)
  (h_a7 : a 7 = 42)
  (h_a9 : a 9 = 110) :
  a 4 = 10 :=
by
  sorry

end find_a4_l222_222049


namespace necessarily_positive_l222_222670

theorem necessarily_positive (x y z : ℝ) (h1 : 0 < x ∧ x < 2) (h2 : -2 < y ∧ y < 0) (h3 : 0 < z ∧ z < 3) : 
  y + 2 * z > 0 := 
sorry

end necessarily_positive_l222_222670


namespace largest_n_divisible_103_l222_222333

theorem largest_n_divisible_103 (n : ℕ) (h1 : n < 103) (h2 : 103 ∣ (n^3 - 1)) : n = 52 :=
sorry

end largest_n_divisible_103_l222_222333


namespace increased_expenses_percent_l222_222152

theorem increased_expenses_percent (S : ℝ) (hS : S = 6250) (initial_save_percent : ℝ) (final_savings : ℝ) 
  (initial_save_percent_def : initial_save_percent = 20) 
  (final_savings_def : final_savings = 250) : 
  (initial_save_percent / 100 * S - final_savings) / (S - initial_save_percent / 100 * S) * 100 = 20 := by
  sorry

end increased_expenses_percent_l222_222152


namespace Dima_claim_false_l222_222167

theorem Dima_claim_false (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : (a*x^2 + b*x + c = 0) → ∃ α β, α < 0 ∧ β < 0 ∧ (α + β = -b/a) ∧ (α*β = c/a)) :
  ¬ ∃ α' β', α' > 0 ∧ β' > 0 ∧ (α' + β' = -c/b) ∧ (α'*β' = a/b) :=
sorry

end Dima_claim_false_l222_222167


namespace arithmetic_sequence_proof_l222_222197

open Nat

def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  a 1 = 2 ∧ (a 2) ^ 2 = (a 1) * (a 5)

def general_formula (a : ℕ → ℤ) (d : ℤ) : Prop :=
  (d = 0 ∧ ∀ n, a n = 2) ∨ (d = 4 ∧ ∀ n, a n = 4 * n - 2)

def sum_seq (a : ℕ → ℤ) (S_n : ℕ → ℤ) (d : ℤ) : Prop :=
  ((∀ n, a n = 2) ∧ (∀ n, S_n n = 2 * n)) ∨ ((∀ n, a n = 4 * n - 2) ∧ (∀ n, S_n n = 4 * n^2 - 2 * n))

theorem arithmetic_sequence_proof :
  ∃ a : ℕ → ℤ, ∃ d : ℤ, arithmetic_seq a d ∧ general_formula a d ∧ ∃ S_n : ℕ → ℤ, sum_seq a S_n d := by
  sorry

end arithmetic_sequence_proof_l222_222197


namespace range_m_l222_222733

noncomputable def even_function (f : ℝ → ℝ) : Prop := 
  ∀ x, f x = f (-x)

noncomputable def decreasing_on_non_neg (f : ℝ → ℝ) : Prop := 
  ∀ ⦃x y⦄, 0 ≤ x → x ≤ y → f y ≤ f x

theorem range_m (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_dec : decreasing_on_non_neg f) :
  ∀ m, f (1 - m) < f m → m < 1 / 2 :=
by
  sorry

end range_m_l222_222733


namespace bones_remaining_l222_222039

namespace Example
variable Juniper_orig Juniper_given Juniper_theft : ℕ

theorem bones_remaining (h1 : Juniper_orig = 4) 
                        (h2 : Juniper_given = 2 * Juniper_orig) 
                        (h3 : Juniper_theft = 2) : 
                        Juniper_orig + Juniper_given - Juniper_theft = 6 :=
by
  sorry
end Example

end bones_remaining_l222_222039


namespace grid_lines_count_l222_222898

   -- Definition of a 4x4 grid
   def grid_points : Finset (ℕ × ℕ) := 
   Finset.product (Finset.range 4) (Finset.range 4)

   -- Definition of what constitutes a line in the grid
   def is_line (p1 p2 : ℕ × ℕ) : Prop := 
   p1 ≠ p2 ∧ (p1.1 = p2.1 ∨ p1.2 = p2.2 ∨ 
              (p1.1 - p2.1 : ℤ) = (p1.2 - p2.2 : ℤ) ∨ 
              (p1.1 - p2.1 : ℤ) = (p2.2 - p1.2 : ℤ))

   -- The theorem substantiating the number of lines
   theorem grid_lines_count : 
     (Finset.card (Finset.filter (λ p : (ℕ × ℕ) × (ℕ × ℕ), is_line p.fst p.snd) 
     (Finset.product grid_points grid_points)) / 2 = 96 :=
   sorry
   
end grid_lines_count_l222_222898


namespace total_tickets_sold_l222_222966

def SeniorPrice : Nat := 10
def RegularPrice : Nat := 15
def TotalSales : Nat := 855
def RegularTicketsSold : Nat := 41

theorem total_tickets_sold : ∃ (S R : Nat), R = RegularTicketsSold ∧ 10 * S + 15 * R = TotalSales ∧ S + R = 65 :=
by
  sorry

end total_tickets_sold_l222_222966


namespace total_boxes_is_4575_l222_222272

-- Define the number of boxes in each warehouse
def num_boxes_in_warehouse_A (x : ℕ) := x
def num_boxes_in_warehouse_B (x : ℕ) := 3 * x
def num_boxes_in_warehouse_C (x : ℕ) := (3 * x) / 2 + 100
def num_boxes_in_warehouse_D (x : ℕ) := 2 * ((3 * x) / 2 + 100) - 50
def num_boxes_in_warehouse_E (x : ℕ) := x + (2 * ((3 * x) / 2 + 100) - 50) - 200

-- Define the condition that warehouse B has 300 more boxes than warehouse E
def condition_B_E (x : ℕ) := 3 * x = num_boxes_in_warehouse_E x + 300

-- Define the total number of boxes calculation
def total_boxes (x : ℕ) := 
    num_boxes_in_warehouse_A x +
    num_boxes_in_warehouse_B x +
    num_boxes_in_warehouse_C x +
    num_boxes_in_warehouse_D x +
    num_boxes_in_warehouse_E x

-- The statement of the problem
theorem total_boxes_is_4575 (x : ℕ) (h : condition_B_E x) : total_boxes x = 4575 :=
by
    sorry

end total_boxes_is_4575_l222_222272


namespace lcm_210_297_l222_222832

theorem lcm_210_297 : Nat.lcm 210 297 = 20790 := 
by sorry

end lcm_210_297_l222_222832


namespace ladder_length_l222_222149

variable (x y : ℝ)

theorem ladder_length :
  (x^2 = 15^2 + y^2) ∧ (x^2 = 24^2 + (y - 13)^2) → x = 25 := by
  sorry

end ladder_length_l222_222149


namespace evaluate_g_at_neg2_l222_222521

def g (x : ℝ) : ℝ := 3 * x^4 - 20 * x^3 + 35 * x^2 - 28 * x - 84

theorem evaluate_g_at_neg2 : g (-2) = 320 := by
  sorry

end evaluate_g_at_neg2_l222_222521


namespace clock_correct_time_fraction_l222_222147

theorem clock_correct_time_fraction :
  let hours := 24
  let incorrect_hours := 6
  let correct_hours_fraction := (hours - incorrect_hours) / hours
  let minutes_per_hour := 60
  let incorrect_minutes_per_hour := 15
  let correct_minutes_fraction := (minutes_per_hour - incorrect_minutes_per_hour) / minutes_per_hour
  correct_hours_fraction * correct_minutes_fraction = (9 / 16) :=
by 
  sorry

end clock_correct_time_fraction_l222_222147


namespace algebraic_expression_evaluation_l222_222624

open Real

noncomputable def x : ℝ := 2 - sqrt 3

theorem algebraic_expression_evaluation :
  (7 + 4 * sqrt 3) * x^2 - (2 + sqrt 3) * x + sqrt 3 = 2 + sqrt 3 :=
by
  sorry

end algebraic_expression_evaluation_l222_222624


namespace number_of_green_pens_l222_222905

theorem number_of_green_pens
  (black_pens : ℕ := 6)
  (red_pens : ℕ := 7)
  (green_pens : ℕ)
  (probability_black : (black_pens : ℚ) / (black_pens + red_pens + green_pens : ℚ) = 1 / 3) :
  green_pens = 5 := 
sorry

end number_of_green_pens_l222_222905


namespace total_questions_solved_l222_222807

-- Define the number of questions Taeyeon solved in a day and the number of days
def Taeyeon_questions_per_day : ℕ := 16
def Taeyeon_days : ℕ := 7

-- Define the number of questions Yura solved in a day and the number of days
def Yura_questions_per_day : ℕ := 25
def Yura_days : ℕ := 6

-- Define the total number of questions Taeyeon and Yura solved
def Total_questions_Taeyeon : ℕ := Taeyeon_questions_per_day * Taeyeon_days
def Total_questions_Yura : ℕ := Yura_questions_per_day * Yura_days
def Total_questions : ℕ := Total_questions_Taeyeon + Total_questions_Yura

-- Prove that the total number of questions solved by Taeyeon and Yura is 262
theorem total_questions_solved : Total_questions = 262 := by
  sorry

end total_questions_solved_l222_222807


namespace top_three_probability_l222_222580

-- Definitions for the real-world problem
def total_ways_to_choose_three_cards : ℕ :=
  52 * 51 * 50

def favorable_ways_to_choose_three_specific_suits : ℕ :=
  13 * 13 * 13 * 6

def probability_top_three_inclusive (total : ℕ) (favorable : ℕ) : ℚ :=
  favorable / total

-- The mathematically equivalent proof problem's Lean statement
theorem top_three_probability:
  probability_top_three_inclusive total_ways_to_choose_three_cards favorable_ways_to_choose_three_specific_suits = 2197 / 22100 :=
by
  sorry

end top_three_probability_l222_222580


namespace avg_of_7_consecutive_integers_l222_222352

theorem avg_of_7_consecutive_integers (a b : ℕ) (h1 : b = (a + (a+1) + (a+2) + (a+3) + (a+4)) / 5) : 
  (b + (b + 1) + (b + 2) + (b + 3) + (b + 4) + (b + 5) + (b + 6)) / 7 = a + 5 := 
  sorry

end avg_of_7_consecutive_integers_l222_222352


namespace two_digit_numbers_equal_three_times_product_of_digits_l222_222175

theorem two_digit_numbers_equal_three_times_product_of_digits :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ ∃ a b : ℕ, n = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 3 * a * b} = {15, 24} :=
by
  sorry

end two_digit_numbers_equal_three_times_product_of_digits_l222_222175


namespace adjusted_distance_buoy_fourth_l222_222725

theorem adjusted_distance_buoy_fourth :
  let a1 := 20  -- distance to the first buoy
  let d := 4    -- common difference (distance between consecutive buoys)
  let ocean_current_effect := 3  -- effect of ocean current
  
  -- distances from the beach to buoys based on their sequence
  let a2 := a1 + d 
  let a3 := a2 + d
  let a4 := a3 + d
  
  -- distance to the fourth buoy without external factors
  let distance_to_fourth_buoy := a1 + 3 * d
  
  -- adjusted distance considering the ocean current
  let adjusted_distance := distance_to_fourth_buoy - ocean_current_effect
  adjusted_distance = 29 := 
by
  let a1 := 20
  let d := 4
  let ocean_current_effect := 3
  let a2 := a1 + d
  let a3 := a2 + d
  let a4 := a3 + d
  let distance_to_fourth_buoy := a1 + 3 * d
  let adjusted_distance := distance_to_fourth_buoy - ocean_current_effect
  sorry

end adjusted_distance_buoy_fourth_l222_222725


namespace exists_subset_no_double_l222_222336

theorem exists_subset_no_double (s : Finset ℕ) (h₁ : s = Finset.range 3000) :
  ∃ t : Finset ℕ, t.card = 2000 ∧ (∀ x ∈ t, ∀ y ∈ t, x ≠ 2 * y ∧ y ≠ 2 * x) :=
by
  sorry

end exists_subset_no_double_l222_222336


namespace truck_initial_gas_ratio_l222_222948

-- Definitions and conditions
def truck_total_capacity : ℕ := 20

def car_total_capacity : ℕ := 12

def car_initial_gas : ℕ := car_total_capacity / 3

def added_gas : ℕ := 18

-- Goal: The ratio of the gas in the truck's tank to its total capacity before she fills it up is 1:2
theorem truck_initial_gas_ratio :
  ∃ T : ℕ, (T + car_initial_gas + added_gas = truck_total_capacity + car_total_capacity) ∧ (T : ℚ) / truck_total_capacity = 1 / 2 :=
by
  sorry

end truck_initial_gas_ratio_l222_222948


namespace girls_from_clay_is_30_l222_222825

-- Definitions for the given conditions
def total_students : ℕ := 150
def total_boys : ℕ := 90
def total_girls : ℕ := 60
def students_jonas : ℕ := 50
def students_clay : ℕ := 70
def students_hart : ℕ := 30
def boys_jonas : ℕ := 25

-- Theorem to prove that the number of girls from Clay Middle School is 30
theorem girls_from_clay_is_30 
  (h1 : total_students = 150)
  (h2 : total_boys = 90)
  (h3 : total_girls = 60)
  (h4 : students_jonas = 50)
  (h5 : students_clay = 70)
  (h6 : students_hart = 30)
  (h7 : boys_jonas = 25) : 
  ∃ girls_clay : ℕ, girls_clay = 30 :=
by 
  sorry

end girls_from_clay_is_30_l222_222825


namespace mary_donated_books_l222_222664

theorem mary_donated_books 
  (s : ℕ) (b_c : ℕ) (b_b : ℕ) (b_y : ℕ) (g_d : ℕ) (g_m : ℕ) (e : ℕ) (s_s : ℕ) 
  (total : ℕ) (out_books : ℕ) (d : ℕ)
  (h1 : s = 72)
  (h2 : b_c = 12)
  (h3 : b_b = 5)
  (h4 : b_y = 2)
  (h5 : g_d = 1)
  (h6 : g_m = 4)
  (h7 : e = 81)
  (h8 : s_s = 3)
  (ht : total = s + b_c + b_b + b_y + g_d + g_m)
  (ho : out_books = total - e)
  (hd : d = out_books - s_s) :
  d = 12 :=
by { sorry }

end mary_donated_books_l222_222664


namespace probability_two_same_number_l222_222872

theorem probability_two_same_number :
  let rolls := 5
  let sides := 8
  let total_outcomes := sides ^ rolls
  let favorable_outcomes := 8 * 7 * 6 * 5 * 4
  let probability_all_different := (favorable_outcomes : ℚ) / total_outcomes
  let probability_at_least_two_same := 1 - probability_all_different
  probability_at_least_two_same = (3256 : ℚ) / 4096 :=
by 
  sorry

end probability_two_same_number_l222_222872


namespace Alice_and_Dave_weight_l222_222864

variable (a b c d : ℕ)

-- Conditions
variable (h1 : a + b = 230)
variable (h2 : b + c = 220)
variable (h3 : c + d = 250)

-- Proof statement
theorem Alice_and_Dave_weight :
  a + d = 260 :=
sorry

end Alice_and_Dave_weight_l222_222864


namespace find_k_l222_222842

theorem find_k (k t : ℤ) (h1 : t = 5 / 9 * (k - 32)) (h2 : t = 75) : k = 167 := 
by 
  sorry

end find_k_l222_222842


namespace fraction_zero_implies_x_is_minus_one_l222_222017

variable (x : ℝ)

theorem fraction_zero_implies_x_is_minus_one (h : (x^2 - 1) / (1 - x) = 0) : x = -1 :=
sorry

end fraction_zero_implies_x_is_minus_one_l222_222017


namespace capacitor_capacitance_l222_222692

theorem capacitor_capacitance 
  (U ε Q : ℝ) 
  (hQ : Q = (U^2 * (ε - 1)^2 * C) /  (2 * ε * (ε + 1)))
  : C = (2 * ε * (ε + 1) * Q) / (U^2 * (ε - 1)^2) :=
by
  sorry

end capacitor_capacitance_l222_222692


namespace pears_for_apples_l222_222771

-- Define the costs of apples, oranges, and pears.
variables {cost_apples cost_oranges cost_pears : ℕ}

-- Condition 1: Ten apples cost the same as five oranges
axiom apples_equiv_oranges : 10 * cost_apples = 5 * cost_oranges

-- Condition 2: Three oranges cost the same as four pears
axiom oranges_equiv_pears : 3 * cost_oranges = 4 * cost_pears

-- Theorem: Tyler can buy 13 pears for the price of 20 apples
theorem pears_for_apples : 20 * cost_apples = 13 * cost_pears :=
sorry

end pears_for_apples_l222_222771


namespace c_alone_finishes_in_60_days_l222_222290

-- Definitions for rates of work
variables (A B C : ℝ)

-- The conditions given in the problem
-- A and B together can finish the job in 15 days
def condition1 : Prop := A + B = 1 / 15
-- A, B, and C together can finish the job in 12 days
def condition2 : Prop := A + B + C = 1 / 12

-- The statement to prove: C alone can finish the job in 60 days
theorem c_alone_finishes_in_60_days 
  (h1 : condition1 A B) 
  (h2 : condition2 A B C) : 
  (1 / C) = 60 :=
by
  sorry

end c_alone_finishes_in_60_days_l222_222290


namespace polar_coordinates_of_point_l222_222982

noncomputable def point_rectangular_to_polar (x y : ℝ) : ℝ × ℝ := 
  let r := Real.sqrt (x^2 + y^2)
  let θ := if y < 0 then 2 * Real.pi + Real.arctan (y / x) else Real.arctan (y / x)
  (r, θ)

theorem polar_coordinates_of_point :
  point_rectangular_to_polar 1 (-1) = (Real.sqrt 2, 7 * Real.pi / 4) :=
by
  unfold point_rectangular_to_polar
  sorry

end polar_coordinates_of_point_l222_222982


namespace possible_values_of_a_l222_222492

theorem possible_values_of_a :
  (∀ x, (x^2 - 3 * x + 2 = 0) → (ax - 2 = 0)) → (a = 0 ∨ a = 1 ∨ a = 2) :=
by
  intro h
  sorry

end possible_values_of_a_l222_222492


namespace smallest_difference_l222_222705

theorem smallest_difference (a b : ℤ) (h1 : a + b < 11) (h2 : a > 6) : a - b = 4 :=
by
  sorry

end smallest_difference_l222_222705


namespace factor_expression_l222_222343

theorem factor_expression (x : ℝ) : 84 * x^7 - 297 * x^13 = 3 * x^7 * (28 - 99 * x^6) :=
by sorry

end factor_expression_l222_222343


namespace find_fourth_number_l222_222071

theorem find_fourth_number (a : ℕ → ℕ) 
  (h1 : ∀ n, n ≥ 2 → a n = a (n - 1) + a (n - 2)) 
  (h2 : a 6 = 42) 
  (h3 : a 8 = 110) : 
  a 3 = 10 := 
sorry

end find_fourth_number_l222_222071


namespace cos_sum_eq_neg_ratio_l222_222210

theorem cos_sum_eq_neg_ratio (γ δ : ℝ) 
  (hγ: Complex.exp (Complex.I * γ) = 4 / 5 + 3 / 5 * Complex.I) 
  (hδ: Complex.exp (Complex.I * δ) = -5 / 13 + 12 / 13 * Complex.I) :
  Real.cos (γ + δ) = -56 / 65 :=
  sorry

end cos_sum_eq_neg_ratio_l222_222210


namespace smallest_positive_integer_ends_in_3_divisible_by_11_l222_222116

theorem smallest_positive_integer_ends_in_3_divisible_by_11 :
  ∃ n : ℕ, n > 0 ∧ n % 10 = 3 ∧ n % 11 = 0 ∧ n = 113 :=
by
  -- We claim that 113 is the required number
  use 113
  split
  -- Proof that 113 is positive
  sorry
  split
  -- Proof that 113 ends in 3
  sorry
  split
  -- Proof that 113 is divisible by 11
  sorry
  -- The smallest, smallest in scope will be evident by construction in the final formal proof
  sorry  

end smallest_positive_integer_ends_in_3_divisible_by_11_l222_222116


namespace sum_of_repeating_decimals_l222_222601

theorem sum_of_repeating_decimals : (0.6666.repeating + 0.4444.repeating : ℝ) = (10 / 9 : ℝ) :=
by
  sorry

end sum_of_repeating_decimals_l222_222601


namespace max_s_value_l222_222939

noncomputable def max_s (m n : ℝ) : ℝ := (m-1)^2 + (n-1)^2 + (m-n)^2

theorem max_s_value (m n : ℝ) (h : m^2 - 4 * n ≥ 0) : 
    ∃ s : ℝ, s = (max_s m n) ∧ s ≤ 9/8 := sorry

end max_s_value_l222_222939


namespace largest_three_digit_geometric_sequence_l222_222283

-- Definitions based on conditions
def is_three_digit_integer (n : ℕ) : Prop := n >= 100 ∧ n < 1000
def digits_distinct (n : ℕ) : Prop := 
  let d₁ := n / 100
  let d₂ := (n / 10) % 10
  let d₃ := n % 10
  d₁ ≠ d₂ ∧ d₂ ≠ d₃ ∧ d₁ ≠ d₃
def geometric_sequence (n : ℕ) : Prop :=
  let d₁ := n / 100
  let d₂ := (n / 10) % 10
  let d₃ := n % 10
  d₁ != 0 ∧ d₂ != 0  ∧ d₃ != 0 ∧ 
  (∃ r: ℚ, d₂ = d₁ * r ∧ d₃ = d₂ * r)

-- Theorem statement
theorem largest_three_digit_geometric_sequence : 
  ∃ n : ℕ, is_three_digit_integer n ∧ digits_distinct n ∧ geometric_sequence n ∧ n = 964 :=
sorry

end largest_three_digit_geometric_sequence_l222_222283


namespace impossible_division_l222_222782

noncomputable def total_matches := 1230

theorem impossible_division :
  ∀ (x y z : ℕ), 
  (x + y + z = total_matches) → 
  (z = (1 / 2) * (x + y + z)) → 
  false :=
by
  sorry

end impossible_division_l222_222782


namespace inequality_proof_l222_222082

theorem inequality_proof
  (x y z : ℝ)
  (h_x : x ≥ 0)
  (h_y : y ≥ 0)
  (h_z : z > 0)
  (h_xy : x ≥ y)
  (h_yz : y ≥ z) :
  (x + y + z) * (x + y - z) * (x - y + z) / (x * y * z) ≥ 3 := by
  sorry

end inequality_proof_l222_222082


namespace find_line_l222_222691

def point_on_line (P : ℝ × ℝ) (m b : ℝ) : Prop :=
  P.2 = m * P.1 + b

def intersection_points_distance (k m b : ℝ) : Prop :=
  |(k^2 - 4*k + 4) - (m*k + b)| = 6

noncomputable def desired_line (m b : ℝ) : Prop :=
  point_on_line (2, 3) m b ∧ ∀ (k : ℝ), intersection_points_distance k m b

theorem find_line : desired_line (-6) 15 := sorry

end find_line_l222_222691


namespace games_bought_l222_222739

/-- 
Given:
1. Geoffrey received €20 from his grandmother.
2. Geoffrey received €25 from his aunt.
3. Geoffrey received €30 from his uncle.
4. Geoffrey now has €125 in his wallet.
5. Geoffrey has €20 left after buying games.
6. Each game costs €35.

Prove that Geoffrey bought 3 games.
-/
theorem games_bought 
  (grandmother_money aunt_money uncle_money total_money left_money game_cost spent_money games_bought : ℤ)
  (h1 : grandmother_money = 20)
  (h2 : aunt_money = 25)
  (h3 : uncle_money = 30)
  (h4 : total_money = 125)
  (h5 : left_money = 20)
  (h6 : game_cost = 35)
  (h7 : spent_money = total_money - left_money)
  (h8 : games_bought = spent_money / game_cost) :
  games_bought = 3 := 
sorry

end games_bought_l222_222739


namespace trisha_dogs_food_expense_l222_222829

theorem trisha_dogs_food_expense :
  ∀ (meat chicken veggies eggs initial remaining final: ℤ),
    meat = 17 → 
    chicken = 22 → 
    veggies = 43 → 
    eggs = 5 → 
    remaining = 35 → 
    initial = 167 →
    final = initial - (meat + chicken + veggies + eggs) - remaining →
    final = 45 := 
by
  intros meat chicken veggies eggs initial remaining final h_meat h_chicken h_veggies h_eggs h_remaining h_initial h_final
  sorry

end trisha_dogs_food_expense_l222_222829


namespace dvd_packs_l222_222986

theorem dvd_packs (cost_per_pack : ℕ) (discount_per_pack : ℕ) (money_available : ℕ) 
  (h_cost : cost_per_pack = 107) 
  (h_discount : discount_per_pack = 106) 
  (h_money : money_available = 93) : 
  (money_available / (cost_per_pack - discount_per_pack)) = 93 := 
by 
  -- Implementation of the proof goes here
  sorry

end dvd_packs_l222_222986


namespace solve_for_x_l222_222087

theorem solve_for_x (x : ℤ) (h : 158 - x = 59) : x = 99 :=
by
  sorry

end solve_for_x_l222_222087


namespace right_angled_isosceles_triangle_third_side_length_l222_222408

theorem right_angled_isosceles_triangle_third_side_length (a b c : ℝ) (h₀ : a = 50) (h₁ : b = 50) (h₂ : a + b + c = 160) : c = 60 :=
by
  -- TODO: Provide proof
  sorry

end right_angled_isosceles_triangle_third_side_length_l222_222408


namespace smallest_product_of_non_factors_l222_222100

theorem smallest_product_of_non_factors (a b : ℕ) (h_a : a ∣ 48) (h_b : b ∣ 48) (h_distinct : a ≠ b) (h_prod_non_factor : ¬ (a * b ∣ 48)) : a * b = 18 :=
sorry

end smallest_product_of_non_factors_l222_222100


namespace frequencies_and_quality_difference_l222_222425

theorem frequencies_and_quality_difference 
  (A_first_class A_second_class B_first_class B_second_class : ℕ)
  (total_A total_B : ℕ)
  (total_first_class total_second_class total : ℕ)
  (critical_value_99 confidence_level : ℕ)
  (freq_A freq_B : ℚ)
  (K_squared : ℚ) :
  A_first_class = 150 →
  A_second_class = 50 →
  B_first_class = 120 →
  B_second_class = 80 →
  total_A = 200 →
  total_B = 200 →
  total_first_class = 270 →
  total_second_class = 130 →
  total = 400 →
  critical_value_99 = 10.828 →
  confidence_level = 99 →
  freq_A = 3 / 4 →
  freq_B = 3 / 5 →
  K_squared = 400 * ((150 * 80 - 50 * 120) ^ 2) / (270 * 130 * 200 * 200) →
  K_squared < critical_value_99 →
  freq_A = 3 / 4 ∧ 
  freq_B = 3 / 5 ∧ 
  confidence_level = 99 := 
by
  intros; 
  sorry

end frequencies_and_quality_difference_l222_222425


namespace riverside_high_badges_l222_222190

/-- Given the conditions on the sums of consecutive prime badge numbers of the debate team members,
prove that Giselle's badge number is 1014, given that the current year is 2025.
-/
theorem riverside_high_badges (p1 p2 p3 p4 : ℕ) (hp1 : Prime p1) (hp2 : Prime p2) (hp3 : Prime p3) (hp4 : Prime p4)
    (hconsec : p1 < p2 ∧ p2 < p3 ∧ p3 < p4 ∧ p4 = p3 + 6)
    (h1 : ∃ x, p1 + p3 = x) (h2 : ∃ y, p1 + p2 = y) (h3 : ∃ z, p2 + p3 = z ∧ z ≤ 31) 
    (h4 : p3 + p4 = 2025) : p4 = 1014 :=
by sorry

end riverside_high_badges_l222_222190


namespace hyperbola_problem_l222_222757

-- Given the conditions of the hyperbola
def hyperbola (x y: ℝ) (b: ℝ) : Prop := (x^2) / 4 - (y^2) / (b^2) = 1 ∧ b > 0

-- Asymptote condition
def asymptote (b: ℝ) : Prop := (b / 2) = (Real.sqrt 6 / 2)

-- Foci, point P condition
def foci_and_point (PF1 PF2: ℝ) : Prop := PF1 / PF2 = 3 / 1 ∧ PF1 - PF2 = 4

-- Math proof problem
theorem hyperbola_problem (b PF1 PF2: ℝ) (P: ℝ × ℝ) :
  hyperbola P.1 P.2 b ∧ asymptote b ∧ foci_and_point PF1 PF2 →
  |PF1 + PF2| = 2 * Real.sqrt 10 :=
by
  sorry

end hyperbola_problem_l222_222757


namespace find_natural_n_l222_222606

theorem find_natural_n (n : ℕ) :
  (992768 ≤ n ∧ n ≤ 993791) ↔ 
  (∀ k : ℕ, k > 0 → k^2 + (n / k^2) = 1991) := sorry

end find_natural_n_l222_222606


namespace energy_calculation_l222_222976

noncomputable def stormy_day_energy_production 
  (energy_per_day : ℝ) (days : ℝ) (number_of_windmills : ℝ) (proportional_increase : ℝ) : ℝ :=
  proportional_increase * (energy_per_day * days * number_of_windmills)

theorem energy_calculation
  (energy_per_day : ℝ) (days : ℝ) (number_of_windmills : ℝ) (wind_speed_proportion : ℝ)
  (stormy_day_energy_per_windmill : ℝ) (s : ℝ)
  (H1 : energy_per_day = 400) 
  (H2 : days = 2) 
  (H3 : number_of_windmills = 3) 
  (H4 : stormy_day_energy_per_windmill = s * energy_per_day)
  : stormy_day_energy_production energy_per_day days number_of_windmills s = s * (400 * 3 * 2) :=
by
  sorry

end energy_calculation_l222_222976


namespace total_pieces_of_clothing_l222_222979

-- Define the conditions:
def boxes : ℕ := 4
def scarves_per_box : ℕ := 2
def mittens_per_box : ℕ := 6

-- Define the target statement:
theorem total_pieces_of_clothing : (boxes * (scarves_per_box + mittens_per_box)) = 32 :=
by
  sorry

end total_pieces_of_clothing_l222_222979


namespace sum_of_factors_of_120_is_37_l222_222686

theorem sum_of_factors_of_120_is_37 :
  ∃ a b c d e : ℤ, (a * b = 120) ∧ (b = a + 1) ∧ (c * d * e = 120) ∧ (d = c + 1) ∧ (e = d + 1) ∧ (a + b + c + d + e = 37) :=
by
  sorry

end sum_of_factors_of_120_is_37_l222_222686


namespace jump_rope_cost_l222_222876

def cost_board_game : ℕ := 12
def cost_playground_ball : ℕ := 4
def saved_money : ℕ := 6
def uncle_money : ℕ := 13
def additional_needed : ℕ := 4

theorem jump_rope_cost :
  let total_money := saved_money + uncle_money
  let total_needed := total_money + additional_needed
  let combined_cost := cost_board_game + cost_playground_ball
  let cost_jump_rope := total_needed - combined_cost
  cost_jump_rope = 7 := by
  sorry

end jump_rope_cost_l222_222876


namespace smallest_non_factor_l222_222105

-- Definitions of the conditions
def isFactorOf (m n : ℕ) : Prop := n % m = 0
def distinct (a b : ℕ) : Prop := a ≠ b

-- The main statement we need to prove.
theorem smallest_non_factor (a b : ℕ) (h_distinct : distinct a b)
  (h_a_factor : isFactorOf a 48) (h_b_factor : isFactorOf b 48)
  (h_not_factor : ¬ isFactorOf (a * b) 48) :
  a * b = 32 := 
sorry

end smallest_non_factor_l222_222105


namespace remaining_leaves_l222_222157

def initial_leaves := 1000
def first_week_shed := (2 / 5 : ℚ) * initial_leaves
def leaves_after_first_week := initial_leaves - first_week_shed
def second_week_shed := (40 / 100 : ℚ) * leaves_after_first_week
def leaves_after_second_week := leaves_after_first_week - second_week_shed
def third_week_shed := (3 / 4 : ℚ) * second_week_shed
def leaves_after_third_week := leaves_after_second_week - third_week_shed

theorem remaining_leaves (initial_leaves first_week_shed leaves_after_first_week second_week_shed leaves_after_second_week third_week_shed leaves_after_third_week: ℚ) : 
  leaves_after_third_week = 180 := by
  sorry

end remaining_leaves_l222_222157


namespace num_integers_condition_l222_222614

theorem num_integers_condition : 
  (∃ (n1 n2 n3 : ℤ), 0 < n1 ∧ n1 < 30 ∧ (∃ k1 : ℤ, (30 - n1) / n1 = k1 ^ 2) ∧
                     0 < n2 ∧ n2 < 30 ∧ (∃ k2 : ℤ, (30 - n2) / n2 = k2 ^ 2) ∧
                     0 < n3 ∧ n3 < 30 ∧ (∃ k3 : ℤ, (30 - n3) / n3 = k3 ^ 2) ∧
                     ∀ n : ℤ, 0 < n ∧ n < 30 ∧ (∃ k : ℤ, (30 - n) / n = k ^ 2) → 
                              (n = n1 ∨ n = n2 ∨ n = n3)) :=
sorry

end num_integers_condition_l222_222614


namespace peanut_butter_candy_count_l222_222420

-- Definitions derived from the conditions
def grape_candy (banana_candy : ℕ) := banana_candy + 5
def peanut_butter_candy (grape_candy : ℕ) := 4 * grape_candy

-- Given condition for the banana jar
def banana_candy := 43

-- The main theorem statement
theorem peanut_butter_candy_count : peanut_butter_candy (grape_candy banana_candy) = 192 :=
by
  sorry

end peanut_butter_candy_count_l222_222420


namespace parabola_directrix_l222_222545

theorem parabola_directrix (x y : ℝ) (h : x^2 = 8 * y) : y = -2 :=
sorry

end parabola_directrix_l222_222545


namespace A_formula_l222_222081

noncomputable def A (i : ℕ) (A₀ θ : ℝ) : ℝ :=
match i with
| 0     => A₀
| (i+1) => (A i A₀ θ * Real.cos θ + Real.sin θ) / (-A i A₀ θ * Real.sin θ + Real.cos θ)

theorem A_formula (A₀ θ : ℝ) (n : ℕ) :
  A n A₀ θ = (A₀ * Real.cos (n * θ) + Real.sin (n * θ)) / (-A₀ * Real.sin (n * θ) + Real.cos (n * θ)) :=
by
  sorry

end A_formula_l222_222081


namespace smallest_positive_integer_ends_in_3_divisible_by_11_l222_222121

theorem smallest_positive_integer_ends_in_3_divisible_by_11 :
  ∃ n : ℕ, n > 0 ∧ n % 10 = 3 ∧ n % 11 = 0 ∧ ∀ m : ℕ, (m > 0 ∧ m % 10 = 3 ∧ m % 11 = 0) → n ≤ m :=
sorry

end smallest_positive_integer_ends_in_3_divisible_by_11_l222_222121


namespace determine_mu_l222_222242

open ProbabilityTheory

noncomputable def random_variable : Type := ℝ
def normal_distribution (μ σ : ℝ) : Measure random_variable := gaussian μ σ

theorem determine_mu (μ σ : ℝ) (ξ : random_variable)
  (hξ : ξ ∼ normal_distribution μ σ)
  (hprob : P(λ x, x > 4, ξ) = 0.5) :
  μ = 4 := sorry

end determine_mu_l222_222242


namespace scientific_notation_of_10760000_l222_222160

theorem scientific_notation_of_10760000 : 
  (10760000 : ℝ) = 1.076 * 10^7 := 
sorry

end scientific_notation_of_10760000_l222_222160


namespace mans_rate_in_still_water_l222_222561

theorem mans_rate_in_still_water
  (V_m V_s : ℝ)
  (h_with_stream : V_m + V_s = 26)
  (h_against_stream : V_m - V_s = 4) :
  V_m = 15 :=
by {
  sorry
}

end mans_rate_in_still_water_l222_222561


namespace singers_in_fifth_verse_l222_222569

theorem singers_in_fifth_verse (choir : ℕ) (absent : ℕ) (participating : ℕ) 
(half_first_verse : ℕ) (third_second_verse : ℕ) (quarter_third_verse : ℕ) 
(fifth_fourth_verse : ℕ) (late_singers : ℕ) :
  choir = 70 → 
  absent = 10 → 
  participating = choir - absent →
  half_first_verse = participating / 2 → 
  third_second_verse = (participating - half_first_verse) / 3 →
  quarter_third_verse = (participating - half_first_verse - third_second_verse) / 4 →
  fifth_fourth_verse = (participating - half_first_verse - third_second_verse - quarter_third_verse) / 5 →
  late_singers = 5 →
  participating = 60 :=
by sorry

end singers_in_fifth_verse_l222_222569


namespace rectangular_plot_width_l222_222139

/-- Theorem: The width of a rectangular plot where the length is thrice its width and the area is 432 sq meters is 12 meters. -/
theorem rectangular_plot_width (w l : ℝ) (h₁ : l = 3 * w) (h₂ : l * w = 432) : w = 12 :=
by
  sorry

end rectangular_plot_width_l222_222139


namespace A_star_B_eq_l222_222041

def A : Set ℝ := {x | ∃ y, y = 2 * x - x^2}
def B : Set ℝ := {y | ∃ x, y = 2^x ∧ x > 0}
def A_star_B : Set ℝ := {x | x ∈ A ∪ B ∧ x ∉ A ∩ B}

theorem A_star_B_eq : A_star_B = {x | x ≤ 1} :=
by {
  sorry
}

end A_star_B_eq_l222_222041


namespace simplify_expression_l222_222398

theorem simplify_expression : 
  (20 * (9 / 14) * (1 / 18) : ℚ) = (5 / 7) := 
by 
  sorry

end simplify_expression_l222_222398


namespace total_pizza_eaten_l222_222596

def don_pizzas : ℝ := 80
def daria_pizzas : ℝ := 2.5 * don_pizzas
def total_pizzas : ℝ := don_pizzas + daria_pizzas

theorem total_pizza_eaten : total_pizzas = 280 := by
  sorry

end total_pizza_eaten_l222_222596


namespace smallest_positive_integer_ends_in_3_divisible_by_11_l222_222120

theorem smallest_positive_integer_ends_in_3_divisible_by_11 :
  ∃ n : ℕ, n > 0 ∧ n % 10 = 3 ∧ n % 11 = 0 ∧ ∀ m : ℕ, (m > 0 ∧ m % 10 = 3 ∧ m % 11 = 0) → n ≤ m :=
sorry

end smallest_positive_integer_ends_in_3_divisible_by_11_l222_222120


namespace squat_percentage_loss_l222_222650

variable (original_squat : ℕ)
variable (original_bench : ℕ)
variable (original_deadlift : ℕ)
variable (lost_deadlift : ℕ)
variable (new_total : ℕ)
variable (unchanged_bench : ℕ)

theorem squat_percentage_loss
  (h1 : original_squat = 700)
  (h2 : original_bench = 400)
  (h3 : original_deadlift = 800)
  (h4 : lost_deadlift = 200)
  (h5 : new_total = 1490)
  (h6 : unchanged_bench = 400) :
  (original_squat - (new_total - (unchanged_bench + (original_deadlift - lost_deadlift)))) * 100 / original_squat = 30 :=
by sorry

end squat_percentage_loss_l222_222650


namespace Correct_Statement_l222_222288

theorem Correct_Statement : 
  (∀ x : ℝ, 7 * x = 4 * x - 3 → 7 * x - 4 * x = -3) ∧
  (∀ x : ℝ, (2 * x - 1) / 3 = 1 + (x - 3) / 2 → 2 * (2 * x - 1) = 6 + 3 * (x - 3)) ∧
  (∀ x : ℝ, 2 * (2 * x - 1) - 3 * (x - 3) = 1 → 4 * x - 2 - 3 * x + 9 = 1) ∧
  (∀ x : ℝ, 2 * (x + 1) = x + 7 → x = 5) :=
by
  sorry

end Correct_Statement_l222_222288


namespace jellybean_total_l222_222028

theorem jellybean_total (large_jellybeans_per_glass : ℕ) 
  (small_jellybeans_per_glass : ℕ) 
  (num_large_glasses : ℕ) 
  (num_small_glasses : ℕ) 
  (h1 : large_jellybeans_per_glass = 50) 
  (h2 : small_jellybeans_per_glass = large_jellybeans_per_glass / 2) 
  (h3 : num_large_glasses = 5) 
  (h4 : num_small_glasses = 3) : 
  (num_large_glasses * large_jellybeans_per_glass + num_small_glasses * small_jellybeans_per_glass) = 325 :=
by
  sorry

end jellybean_total_l222_222028


namespace find_fourth_number_l222_222063

def nat_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, n ≥ 3 → a n = a (n - 1) + a (n - 2)

variable {a : ℕ → ℕ}

theorem find_fourth_number (h_seq : nat_sequence a) (h7 : a 7 = 42) (h9 : a 9 = 110) : a 4 = 10 :=
by
  -- Placeholder for proof steps
  sorry

end find_fourth_number_l222_222063


namespace f_at_2_is_neg_1_l222_222488

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^5 + b * x^3 - x + 2

-- Given condition: f(-2) = 5
axiom h : ∀ (a b : ℝ), f a b (-2) = 5

-- Prove that f(2) = -1 given the above conditions
theorem f_at_2_is_neg_1 (a b : ℝ) (h_ab : f a b (-2) = 5) : f a b 2 = -1 := by
  sorry

end f_at_2_is_neg_1_l222_222488


namespace remainder_abc_div9_l222_222637

theorem remainder_abc_div9 (a b c : ℕ) (ha : a < 9) (hb : b < 9) (hc : c < 9) 
    (h1 : a + 2 * b + 3 * c ≡ 0 [MOD 9]) 
    (h2 : 2 * a + 3 * b + c ≡ 5 [MOD 9]) 
    (h3 : 3 * a + b + 2 * c ≡ 5 [MOD 9]) : 
    (a * b * c) % 9 = 0 := 
sorry

end remainder_abc_div9_l222_222637


namespace greatest_integer_a_l222_222430

theorem greatest_integer_a (a : ℤ) : a * a < 44 → a ≤ 6 :=
by
  intros h
  sorry

end greatest_integer_a_l222_222430


namespace sqrt_three_irrational_l222_222866

-- Define what it means for a number to be rational
def is_rational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Define what it means for a number to be irrational
def is_irrational (x : ℝ) : Prop := ¬ is_rational x

-- State that sqrt(3) is irrational
theorem sqrt_three_irrational : is_irrational (Real.sqrt 3) :=
sorry

end sqrt_three_irrational_l222_222866


namespace cookies_ratio_l222_222869

theorem cookies_ratio (T : ℝ) (h1 : 0 ≤ T) (h_total : 5 + T + 1.4 * T = 29) : T / 5 = 2 :=
by sorry

end cookies_ratio_l222_222869


namespace initial_men_count_l222_222215

theorem initial_men_count (x : ℕ) 
  (h1 : ∀ t : ℕ, t = 25 * x) 
  (h2 : ∀ t : ℕ, t = 12 * 75) : 
  x = 36 := 
by
  sorry

end initial_men_count_l222_222215


namespace mushrooms_collected_l222_222998

theorem mushrooms_collected (x1 x2 x3 x4 : ℕ) 
  (h1 : x1 + x2 = 7) 
  (h2 : x1 + x3 = 9)
  (h3 : x2 + x3 = 10) : x1 = 3 ∧ x2 = 4 ∧ x3 = 6 ∧ x4 = 7 :=
by
  sorry

end mushrooms_collected_l222_222998


namespace jill_arrives_before_jack_l222_222910

theorem jill_arrives_before_jack {distance speed_jill speed_jack : ℝ} (h1 : distance = 1) 
  (h2 : speed_jill = 10) (h3 : speed_jack = 4) :
  (60 * (distance / speed_jack) - 60 * (distance / speed_jill)) = 9 :=
by
  sorry

end jill_arrives_before_jack_l222_222910


namespace min_value_of_x_plus_2y_l222_222752

theorem min_value_of_x_plus_2y {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 8 / x + 1 / y = 1) : x + 2 * y ≥ 18 :=
sorry

end min_value_of_x_plus_2y_l222_222752


namespace recurring_decimals_sum_correct_l222_222602

noncomputable def recurring_decimals_sum : ℚ :=
  let x := (2:ℚ) / 3
  let y := (4:ℚ) / 9
  x + y

theorem recurring_decimals_sum_correct :
  recurring_decimals_sum = 10 / 9 := 
  sorry

end recurring_decimals_sum_correct_l222_222602


namespace no_positive_int_squares_l222_222924

theorem no_positive_int_squares (n : ℕ) (h_pos : 0 < n) :
  ¬ (∃ a b c : ℕ, a ^ 2 = 2 * n ^ 2 + 1 ∧ b ^ 2 = 3 * n ^ 2 + 1 ∧ c ^ 2 = 6 * n ^ 2 + 1) := by
  sorry

end no_positive_int_squares_l222_222924


namespace correct_percentage_fruits_in_good_condition_l222_222576

noncomputable def percentage_fruits_in_good_condition
    (total_oranges : ℕ)
    (total_bananas : ℕ)
    (rotten_percentage_oranges : ℝ)
    (rotten_percentage_bananas : ℝ) : ℝ :=
let rotten_oranges := (rotten_percentage_oranges / 100) * total_oranges
let rotten_bananas := (rotten_percentage_bananas / 100) * total_bananas
let good_condition_oranges := total_oranges - rotten_oranges
let good_condition_bananas := total_bananas - rotten_bananas
let total_fruits_in_good_condition := good_condition_oranges + good_condition_bananas
let total_fruits := total_oranges + total_bananas
(total_fruits_in_good_condition / total_fruits) * 100

theorem correct_percentage_fruits_in_good_condition :
  percentage_fruits_in_good_condition 600 400 15 4 = 89.4 := by
  sorry

end correct_percentage_fruits_in_good_condition_l222_222576


namespace binom_20_10_eq_184756_l222_222631

theorem binom_20_10_eq_184756 
  (h1 : Nat.choose 19 9 = 92378)
  (h2 : Nat.choose 19 10 = Nat.choose 19 9) : 
  Nat.choose 20 10 = 184756 := 
by
  sorry

end binom_20_10_eq_184756_l222_222631


namespace normal_distribution_symmetry_l222_222523

open MeasureTheory

variable (a : ℝ)

theorem normal_distribution_symmetry (X : MeasureTheory.ProbabilityDistributions.Normal 3 2) :
  (∀ a : ℝ, Probability (X < 2 * a + 3) = Probability (X > a - 2)) → a = 5 / 3 :=
by
  intros h
  sorry

end normal_distribution_symmetry_l222_222523


namespace elevator_travel_time_l222_222665

noncomputable def total_time_in_hours (floors : ℕ) (time_first_half : ℕ) (time_next_floors_per_floor : ℕ) (next_floors : ℕ) (time_final_floors_per_floor : ℕ) (final_floors : ℕ) : ℕ :=
  let time_first_part := time_first_half
  let time_next_part := time_next_floors_per_floor * next_floors
  let time_final_part := time_final_floors_per_floor * final_floors
  (time_first_part + time_next_part + time_final_part) / 60

theorem elevator_travel_time :
  total_time_in_hours 20 15 5 5 16 5 = 2 := 
by
  sorry

end elevator_travel_time_l222_222665


namespace range_of_m_l222_222183

theorem range_of_m (m : ℝ) : (-6 < m ∧ m < 2) ↔ ∃ x : ℝ, |x - m| + |x + 2| < 4 :=
by sorry

end range_of_m_l222_222183


namespace sum_squares_mod_divisor_l222_222833

-- Define the sum of the squares from 1 to 10
def sum_squares := (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2 + 8^2 + 9^2 + 10^2)

-- Define the divisor
def divisor := 11

-- Prove that the remainder of sum_squares when divided by divisor is 0
theorem sum_squares_mod_divisor : sum_squares % divisor = 0 :=
by
  sorry

end sum_squares_mod_divisor_l222_222833


namespace find_k_l222_222469

theorem find_k : ∃ b k : ℝ, (∀ x : ℝ, (x + b)^2 = x^2 - 20 * x + k) ∧ k = 100 := by
  sorry

end find_k_l222_222469


namespace max_5_cent_coins_l222_222285

theorem max_5_cent_coins :
  ∃ (x y z : ℕ), 
  x + y + z = 25 ∧ 
  x + 2*y + 5*z = 60 ∧
  (∀ y' z' : ℕ, y' + 4*z' = 35 → z' ≤ 8) ∧
  y + 4*z = 35 ∧ z = 8 := 
sorry

end max_5_cent_coins_l222_222285


namespace income_expenditure_ratio_l222_222933

variable (I S E : ℕ)
variable (hI : I = 16000)
variable (hS : S = 3200)
variable (hExp : S = I - E)

theorem income_expenditure_ratio (I S E : ℕ) (hI : I = 16000) (hS : S = 3200) (hExp : S = I - E) : I / Nat.gcd I E = 5 ∧ E / Nat.gcd I E = 4 := by
  sorry

end income_expenditure_ratio_l222_222933


namespace average_salary_rest_of_workers_l222_222929

theorem average_salary_rest_of_workers
  (avg_salary_all : ℝ)
  (num_all_workers : ℕ)
  (avg_salary_techs : ℝ)
  (num_techs : ℕ)
  (avg_salary_rest : ℝ)
  (num_rest : ℕ) :
  avg_salary_all = 8000 →
  num_all_workers = 21 →
  avg_salary_techs = 12000 →
  num_techs = 7 →
  num_rest = num_all_workers - num_techs →
  avg_salary_rest = (avg_salary_all * num_all_workers - avg_salary_techs * num_techs) / num_rest →
  avg_salary_rest = 6000 :=
by
  intros h_avg_all h_num_all h_avg_techs h_num_techs h_num_rest h_avg_rest
  sorry

end average_salary_rest_of_workers_l222_222929


namespace sqrt_product_simplification_l222_222325

variable (p : ℝ)

theorem sqrt_product_simplification (hp : 0 ≤ p) :
  (Real.sqrt (42 * p) * Real.sqrt (7 * p) * Real.sqrt (14 * p)) = 42 * p * Real.sqrt (7 * p) :=
sorry

end sqrt_product_simplification_l222_222325


namespace find_a4_l222_222054

def seq (a : ℕ → ℕ) (n : ℕ) : Prop :=
(∀ n, n ≥ 3 → a n = a (n - 1) + a (n - 2))

theorem find_a4 (a : ℕ → ℕ) (h_seq : seq a) (h_a7 : a 7 = 42) (h_a9 : a 9 = 110) : a 4 = 10 :=
by
  sorry

end find_a4_l222_222054


namespace megatech_basic_astrophysics_degrees_l222_222955

def budget_allocation (microphotonics home_electronics food_additives gm_microorganisms industrial_lubricants: ℕ) :=
  100 - (microphotonics + home_electronics + food_additives + gm_microorganisms + industrial_lubricants)

noncomputable def degrees_for_astrophysics (percentage: ℕ) :=
  (percentage * 360) / 100

theorem megatech_basic_astrophysics_degrees (microphotonics home_electronics food_additives gm_microorganisms industrial_lubricants: ℕ) :
  microphotonics = 14 →
  home_electronics = 24 →
  food_additives = 10 →
  gm_microorganisms = 29 →
  industrial_lubricants = 8 →
  degrees_for_astrophysics (budget_allocation microphotonics home_electronics food_additives gm_microorganisms industrial_lubricants) = 54 :=
by
  sorry

end megatech_basic_astrophysics_degrees_l222_222955


namespace distance_to_origin_l222_222282

noncomputable def calculate_distance : ℝ :=
  let z := ((1 - complex.i) * (1 + complex.i)) / complex.i in
  complex.abs z

theorem distance_to_origin :
  calculate_distance = 2 :=
by
  sorry

end distance_to_origin_l222_222282


namespace ten_years_less_average_age_l222_222506

-- Defining the conditions formally
def lukeAge : ℕ := 20
def mrBernardAgeInEightYears : ℕ := 3 * lukeAge

-- Lean statement to prove the problem
theorem ten_years_less_average_age : 
  mrBernardAgeInEightYears - 8 = 52 → (lukeAge + (mrBernardAgeInEightYears - 8)) / 2 - 10 = 26 := 
by
  intros h
  sorry

end ten_years_less_average_age_l222_222506


namespace marksmen_consistency_excellent_shots_distribution_expected_value_of_excellent_shots_l222_222708

noncomputable def variance (l : List ℝ) : ℝ := (l.map (λ x, (x - l.average) ^ 2)).sum / (l.length - 1)

def excellent_shots_probability (scores: List ℝ) (threshold: ℝ) : ℝ :=
  (scores.filter (λ x, x >= threshold)).length / scores.length

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ := (Nat.choose n k) * p^k * (1 - p)^(n - k)

theorem marksmen_consistency (A_scores B_scores: List ℝ)
  (hA : A_scores = [7, 8, 7, 9, 5, 4, 9, 10, 7, 4])
  (hB : B_scores = [9, 5, 7, 8, 7, 6, 8, 6, 7, 7]) :
  variance A_scores = 4 ∧ variance B_scores = 1.2 := by sorry

theorem excellent_shots_distribution (A_scores : List ℝ)
  (hA : A_scores = [7, 8, 7, 9, 5, 4, 9, 10, 7, 4])
  (threshold : ℝ)
  (h_threshold : threshold = 8) :
  let p := excellent_shots_probability A_scores threshold in
  binomial_probability 3 0 p = 27/125 ∧
  binomial_probability 3 1 p = 54/125 ∧
  binomial_probability 3 2 p = 36/125 ∧
  binomial_probability 3 3 p = 8/125 := by sorry

theorem expected_value_of_excellent_shots (A_scores : List ℝ)
  (hA : A_scores = [7, 8, 7, 9, 5, 4, 9, 10, 7, 4])
  (threshold : ℝ)
  (h_threshold : threshold = 8) :
  let p := excellent_shots_probability A_scores threshold in
  3 * p = 1.2 := by sorry

end marksmen_consistency_excellent_shots_distribution_expected_value_of_excellent_shots_l222_222708


namespace unattainable_y_l222_222328

theorem unattainable_y (x : ℝ) (h : 4 * x + 5 ≠ 0) : 
  (y = (3 - x) / (4 * x + 5)) → (y ≠ -1/4) :=
sorry

end unattainable_y_l222_222328


namespace quadrilateral_count_correct_triangle_count_correct_intersection_triangle_count_correct_l222_222565

-- 1. Problem: Count of quadrilaterals from 12 points in a semicircle
def semicircle_points : ℕ := 12
def quadrilaterals_from_semicircle_points : ℕ :=
  let points_on_semicircle := 8
  let points_on_diameter := 4
  360 -- This corresponds to the final computed count, skipping calculation details

theorem quadrilateral_count_correct :
  quadrilaterals_from_semicircle_points = 360 := sorry

-- 2. Problem: Count of triangles from 10 points along an angle
def angle_points : ℕ := 10
def triangles_from_angle_points : ℕ :=
  let points_on_one_side := 5
  let points_on_other_side := 4
  90 -- This corresponds to the final computed count, skipping calculation details

theorem triangle_count_correct :
  triangles_from_angle_points = 90 := sorry

-- 3. Problem: Count of triangles from intersection points of parallel lines
def intersection_points : ℕ := 12
def triangles_from_intersections : ℕ :=
  let line_set_1_count := 3
  let line_set_2_count := 4
  200 -- This corresponds to the final computed count, skipping calculation details

theorem intersection_triangle_count_correct :
  triangles_from_intersections = 200 := sorry

end quadrilateral_count_correct_triangle_count_correct_intersection_triangle_count_correct_l222_222565


namespace smallest_n_interesting_meeting_l222_222743

theorem smallest_n_interesting_meeting (m : ℕ) (hm : 2 ≤ m) :
  ∀ (n : ℕ), (n ≤ 3 * m - 1) ∧ (∀ (rep : Finset (Fin (3 * m))), rep.card = n →
  ∃ subrep : Finset (Fin (3 * m)), subrep.card = 3 ∧ ∀ (x y : Fin (3 * m)), x ∈ subrep → y ∈ subrep → x ≠ y → ∃ z : Fin (3 * m), z ∈ subrep ∧ z = x + y) → n = 2 * m + 1 := by
  sorry

end smallest_n_interesting_meeting_l222_222743


namespace length_of_GH_l222_222783

theorem length_of_GH (AB CD GH : ℤ) (h_parallel : AB = 240 ∧ CD = 160 ∧ (AB + CD) = GH*2) : GH = 320 / 3 :=
by sorry

end length_of_GH_l222_222783


namespace problem1_problem2_l222_222616

-- First proof problem
theorem problem1 (a b : ℝ) : a^4 + 6 * a^2 * b^2 + b^4 ≥ 4 * a * b * (a^2 + b^2) :=
by sorry

-- Second proof problem
theorem problem2 (a b : ℝ) : ∃ (x : ℝ), 
  (∀ (x : ℝ), |2 * x - a^4 + (1 - 6 * a^2 * b^2 - b^4)| + 2 * |x - (2 * a^3 * b + 2 * a * b^3 - 1)| ≥ 1) ∧
  ∃ (x : ℝ), |2 * x - a^4 + (1 - 6 * a^2 * b^2 - b^4)| + 2 * |x - (2 * a^3 * b + 2 * a * b^3 - 1)| = 1 :=
by sorry

end problem1_problem2_l222_222616


namespace compute_expression_l222_222327

theorem compute_expression : 19 * 42 + 81 * 19 = 2337 := by
  sorry

end compute_expression_l222_222327


namespace matrix_power_example_l222_222789

open Matrix

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![(Real.cos (π / 4)), 0, -(Real.sin (π / 4)),
    0, 1, 0,
    (Real.sin (π / 4)), 0, (Real.cos (π / 4))]

theorem matrix_power_example : B ^ 2024 = 1 := by
  sorry

end matrix_power_example_l222_222789


namespace smallest_solution_l222_222184

theorem smallest_solution (x : ℝ) : (1 / (x - 3) + 1 / (x - 5) = 5 / (x - 4)) → x = 4 - (Real.sqrt 15) / 3 :=
by
  sorry

end smallest_solution_l222_222184


namespace area_of_triangle_l222_222888

theorem area_of_triangle (a b c : ℝ) (h₁ : a + b = 14) (h₂ : c = 10) (h₃ : a^2 + b^2 = c^2) :
  (1 / 2) * a * b = 24 :=
  sorry

end area_of_triangle_l222_222888


namespace union_sets_l222_222005

def M : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def N : Set ℝ := {x | 2 < x ∧ x ≤ 5}

theorem union_sets :
  M ∪ N = {x | -1 ≤ x ∧ x ≤ 5} := by
  sorry

end union_sets_l222_222005


namespace jellybeans_needed_l222_222033

-- Define the initial conditions as constants
def jellybeans_per_large_glass := 50
def jellybeans_per_small_glass := jellybeans_per_large_glass / 2
def number_of_large_glasses := 5
def number_of_small_glasses := 3

-- Calculate the total number of jellybeans needed
def total_jellybeans : ℕ :=
  (number_of_large_glasses * jellybeans_per_large_glass) + 
  (number_of_small_glasses * jellybeans_per_small_glass)

-- Prove that the total number of jellybeans needed is 325
theorem jellybeans_needed : total_jellybeans = 325 :=
sorry

end jellybeans_needed_l222_222033


namespace fewer_popsicle_sticks_l222_222809

theorem fewer_popsicle_sticks :
  let boys := 10
  let girls := 12
  let sticks_per_boy := 15
  let sticks_per_girl := 12
  let boys_total := boys * sticks_per_boy
  let girls_total := girls * sticks_per_girl
  boys_total - girls_total = 6 := 
by
  let boys := 10
  let girls := 12
  let sticks_per_boy := 15
  let sticks_per_girl := 12
  let boys_total := boys * sticks_per_boy
  let girls_total := girls * sticks_per_girl
  show boys_total - girls_total = 6
  sorry

end fewer_popsicle_sticks_l222_222809


namespace largest_value_of_a_l222_222043

noncomputable def largest_possible_value_of_a (a b c d : ℕ) 
  (h1 : a < 3 * b) (h2 : b < 4 * c) (h3 : c < 5 * d) (h4 : c % 2 = 0) (h5 : d < 150) : Prop :=
  a = 8924

theorem largest_value_of_a (a b c d : ℕ)
  (h1 : a < 3 * b) (h2 : b < 4 * c) (h3 : c < 5 * d) (h4 : c % 2 = 0) (h5 : d < 150)
  (h6 : largest_possible_value_of_a a b c d h1 h2 h3 h4 h5) : a = 8924 := h6

end largest_value_of_a_l222_222043


namespace rate_per_kg_grapes_is_70_l222_222589

-- Let G be the rate per kg for the grapes
def rate_per_kg_grapes (G : ℕ) := G

-- Bruce purchased 8 kg of grapes at rate G per kg
def grapes_cost (G : ℕ) := 8 * G

-- Bruce purchased 11 kg of mangoes at the rate of 55 per kg
def mangoes_cost := 11 * 55

-- Bruce paid a total of 1165 to the shopkeeper
def total_paid := 1165

-- The problem: Prove that the rate per kg for the grapes is 70
theorem rate_per_kg_grapes_is_70 : rate_per_kg_grapes 70 = 70 ∧ grapes_cost 70 + mangoes_cost = total_paid := by
  sorry

end rate_per_kg_grapes_is_70_l222_222589


namespace age_difference_l222_222539

-- Let D denote the daughter's age and M denote the mother's age
variable (D M : ℕ)

-- Conditions given in the problem
axiom h1 : M = 11 * D
axiom h2 : M + 13 = 2 * (D + 13)

-- The main proof statement to show the difference in their current ages
theorem age_difference : M - D = 40 :=
by
  sorry

end age_difference_l222_222539


namespace perpendicular_line_sufficient_condition_l222_222597

theorem perpendicular_line_sufficient_condition (a : ℝ) :
  (-a) * ((a + 2) / 3) = -1 ↔ (a = -3 ∨ a = 1) :=
by {
  sorry
}

#print perpendicular_line_sufficient_condition

end perpendicular_line_sufficient_condition_l222_222597


namespace student_a_score_l222_222452

def total_questions : ℕ := 100
def correct_responses : ℕ := 87
def incorrect_responses : ℕ := total_questions - correct_responses
def score : ℕ := correct_responses - 2 * incorrect_responses

theorem student_a_score : score = 61 := by
  unfold score
  unfold correct_responses
  unfold incorrect_responses
  norm_num
  -- At this point, the theorem is stated, but we insert sorry to satisfy the requirement of not providing the proof.
  sorry

end student_a_score_l222_222452


namespace arc_length_one_radian_l222_222268

-- Given definitions and conditions
def radius : ℝ := 6370
def angle : ℝ := 1

-- Arc length formula
def arc_length (R α : ℝ) : ℝ := R * α

-- Statement to prove
theorem arc_length_one_radian : arc_length radius angle = 6370 := 
by 
  -- Proof goes here
  sorry

end arc_length_one_radian_l222_222268


namespace min_value_of_T_l222_222654

noncomputable def T (x p : ℝ) : ℝ := |x - p| + |x - 15| + |x - (15 + p)|

theorem min_value_of_T (p : ℝ) (hp : 0 < p ∧ p < 15) :
  ∃ x, p ≤ x ∧ x ≤ 15 ∧ T x p = 15 :=
sorry

end min_value_of_T_l222_222654


namespace highest_visitors_at_4pm_yellow_warning_time_at_12_30pm_l222_222706

-- Definitions for cumulative visitors entering and leaving
def y (x : ℕ) : ℕ := 850 * x + 100
def z (x : ℕ) : ℕ := 200 * x - 200

-- Definition for total number of visitors at time x
def w (x : ℕ) : ℕ := y x - z x

-- Proof problem statements
theorem highest_visitors_at_4pm :
  ∀x, x ≤ 9 → w 9 ≥ w x :=
sorry

theorem yellow_warning_time_at_12_30pm :
  ∃x, w x = 2600 :=
sorry

end highest_visitors_at_4pm_yellow_warning_time_at_12_30pm_l222_222706


namespace sector_area_l222_222013

theorem sector_area (α : ℝ) (l : ℝ) (r : ℝ) (S : ℝ) : 
  α = 1 ∧ l = 6 ∧ l = α * r → S = (1/2) * α * r ^ 2 → S = 18 :=
by
  intros h h' 
  sorry

end sector_area_l222_222013


namespace angle_value_is_140_l222_222376

-- Definitions of conditions
def angle_on_straight_line_degrees (x y : ℝ) : Prop := x + y = 180

-- Main statement in Lean
theorem angle_value_is_140 (x : ℝ) (h₁ : angle_on_straight_line_degrees 40 x) : x = 140 :=
by
  -- Proof is omitted (not required as per instructions)
  sorry

end angle_value_is_140_l222_222376


namespace find_blue_balls_l222_222849

theorem find_blue_balls 
  (B : ℕ)
  (red_balls : ℕ := 7)
  (green_balls : ℕ := 4)
  (prob_red_red : ℚ := 7 / 40) -- 0.175 represented as a rational number
  (h : (21 / ((11 + B) * (10 + B) / 2 : ℚ)) = prob_red_red) :
  B = 5 :=
sorry

end find_blue_balls_l222_222849


namespace eq_of_divisibility_condition_l222_222248

theorem eq_of_divisibility_condition (a b : ℕ) (h : ∃ᶠ n in Filter.atTop, (a^n + b^n) ∣ (a^(n+1) + b^(n+1))) : a = b :=
sorry

end eq_of_divisibility_condition_l222_222248


namespace frequency_first_class_machineA_is_3_over_4_frequency_first_class_machineB_is_3_over_5_significant_quality_difference_l222_222426

-- Definitions based on the problem conditions
def machineA_first_class := 150
def machineA_total := 200
def machineB_first_class := 120
def machineB_total := 200
def total_products := machineA_total + machineB_total

-- Frequencies of first-class products
def frequency_machineA : ℚ := machineA_first_class / machineA_total
def frequency_machineB : ℚ := machineB_first_class / machineB_total

-- Values for chi-squared formula
def a := machineA_first_class
def b := machineA_total - machineA_first_class
def c := machineB_first_class
def d := machineB_total - machineB_first_class

-- Given formula for K^2
def K_squared : ℚ := (total_products * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Proof problem statements
theorem frequency_first_class_machineA_is_3_over_4 : frequency_machineA = 3 / 4 := by
  sorry

theorem frequency_first_class_machineB_is_3_over_5 : frequency_machineB = 3 / 5 := by
  sorry

theorem significant_quality_difference : K_squared > 6.635 := by
  sorry

end frequency_first_class_machineA_is_3_over_4_frequency_first_class_machineB_is_3_over_5_significant_quality_difference_l222_222426


namespace coins_problem_l222_222568

theorem coins_problem : ∃ n : ℕ, (n % 8 = 6) ∧ (n % 9 = 7) ∧ (n % 11 = 8) :=
by {
  sorry
}

end coins_problem_l222_222568


namespace team_size_per_team_l222_222587

theorem team_size_per_team (managers employees teams people_per_team : ℕ) 
  (h1 : managers = 23) 
  (h2 : employees = 7) 
  (h3 : teams = 6) 
  (h4 : people_per_team = (managers + employees) / teams) : 
  people_per_team = 5 :=
by 
  sorry

end team_size_per_team_l222_222587


namespace smallest_int_ends_in_3_div_by_11_l222_222126

theorem smallest_int_ends_in_3_div_by_11 :
  ∃ k : ℕ, k > 0 ∧ k % 10 = 3 ∧ k % 11 = 0 ∧ k = 33 :=
by {
  sorry
}

end smallest_int_ends_in_3_div_by_11_l222_222126


namespace pizza_volume_piece_l222_222447

theorem pizza_volume_piece (h : ℝ) (d : ℝ) (n : ℝ) (V_piece : ℝ) 
  (h_eq : h = 1 / 2) (d_eq : d = 16) (n_eq : n = 8) : 
  V_piece = 4 * Real.pi :=
by
  sorry

end pizza_volume_piece_l222_222447


namespace windmere_zoo_two_legged_birds_l222_222798

theorem windmere_zoo_two_legged_birds (b m u : ℕ) (head_count : b + m + u = 300) (leg_count : 2 * b + 4 * m + 3 * u = 710) : b = 230 :=
sorry

end windmere_zoo_two_legged_birds_l222_222798


namespace chess_team_combination_l222_222920

theorem chess_team_combination 
  (players : Finset ℕ) (quadruplets : Finset ℕ) 
  (h_players : players.card = 18) 
  (h_quadruplets : quadruplets.card = 4) 
  (h_team : quadruplets ⊆ players) :
  ∃ (num_ways : ℕ), num_ways = (Nat.choose 14 4) ∧ num_ways = 1001 :=
by
  sorry

end chess_team_combination_l222_222920


namespace ratio_of_ages_l222_222803

theorem ratio_of_ages
  (Sandy_age : ℕ)
  (Molly_age : ℕ)
  (h1 : Sandy_age = 49)
  (h2 : Molly_age = Sandy_age + 14) : (Sandy_age : ℚ) / Molly_age = 7 / 9 :=
by
  -- To complete the proof.
  sorry

end ratio_of_ages_l222_222803


namespace zoe_has_47_nickels_l222_222132

theorem zoe_has_47_nickels (x : ℕ) 
  (h1 : 5 * x + 10 * x + 50 * x = 3050) : 
  x = 47 := 
sorry

end zoe_has_47_nickels_l222_222132


namespace area_of_MNFK_l222_222925

theorem area_of_MNFK (ABNF CMKD MNFK : ℝ) (BN : ℝ) (KD : ℝ) (ABMK : ℝ) (CDFN : ℝ)
  (h1 : BN = 8) (h2 : KD = 9) (h3 : ABMK = 25) (h4 : CDFN = 32) :
  MNFK = 31 :=
by
  have hx : 8 * (MNFK + 25) - 25 = 9 * (MNFK + 32) - 32 := sorry
  exact sorry

end area_of_MNFK_l222_222925


namespace qin_jiushao_algorithm_v2_l222_222694

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 1 + 2 * x + x^2 - 3 * x^3 + 2 * x^4

-- Define the value x to evaluate the polynomial at
def x0 : ℝ := -1

-- Define the intermediate value v2 according to Horner's rule
def v1 : ℝ := 2 * x0^4 - 3 * x0^3 + x0^2
def v2 : ℝ := v1 * x0 + 2

theorem qin_jiushao_algorithm_v2 : v2 = -4 := 
by 
  -- The proof will be here, for now we place sorry.
  sorry

end qin_jiushao_algorithm_v2_l222_222694


namespace g_2187_value_l222_222410

-- Define the function properties and the goal
theorem g_2187_value (g : ℕ → ℝ) (h : ∀ x y m : ℕ, x + y = 3^m → g x + g y = m^3) :
  g 2187 = 343 :=
sorry

end g_2187_value_l222_222410


namespace correct_discount_rate_l222_222310

def purchase_price : ℝ := 200
def marked_price : ℝ := 300
def desired_profit_percentage : ℝ := 0.20

theorem correct_discount_rate :
  ∃ (x : ℝ), 300 * x = 240 ∧ x = 0.80 := 
by
  sorry

end correct_discount_rate_l222_222310


namespace min_value_expression_l222_222481

theorem min_value_expression (x : ℝ) (hx : x > 0) : x + 4/x ≥ 4 :=
sorry

end min_value_expression_l222_222481


namespace sandwiches_difference_l222_222527

-- Conditions definitions
def sandwiches_at_lunch_monday : ℤ := 3
def sandwiches_at_dinner_monday : ℤ := 2 * sandwiches_at_lunch_monday
def total_sandwiches_monday : ℤ := sandwiches_at_lunch_monday + sandwiches_at_dinner_monday
def sandwiches_on_tuesday : ℤ := 1

-- Proof goal
theorem sandwiches_difference :
  total_sandwiches_monday - sandwiches_on_tuesday = 8 :=
  by
  sorry

end sandwiches_difference_l222_222527


namespace find_y_coordinate_of_P_l222_222790

noncomputable def A : ℝ × ℝ := (-4, 0)
noncomputable def B : ℝ × ℝ := (-3, 2)
noncomputable def C : ℝ × ℝ := (3, 2)
noncomputable def D : ℝ × ℝ := (4, 0)
noncomputable def ell1 (P : ℝ × ℝ) : Prop := (P.1 + 4) ^ 2 / 25 + (P.2) ^ 2 / 9 = 1
noncomputable def ell2 (P : ℝ × ℝ) : Prop := (P.1 + 3) ^ 2 / 25 + ((P.2 - 2) ^ 2) / 16 = 1

theorem find_y_coordinate_of_P :
  ∃ y : ℝ,
    ell1 (0, y) ∧ ell2 (0, y) ∧
    y = 6 / 7 ∧
    6 + 7 = 13 :=
by
  sorry

end find_y_coordinate_of_P_l222_222790


namespace john_ate_2_bags_for_dinner_l222_222037

variable (x y : ℕ)
variable (h1 : x + y = 3)
variable (h2 : y ≥ 1)

theorem john_ate_2_bags_for_dinner : x = 2 := 
by sorry

end john_ate_2_bags_for_dinner_l222_222037


namespace center_square_side_length_l222_222402

theorem center_square_side_length (s : ℝ) :
    let total_area := 120 * 120
    let l_shape_area := (5 / 24) * total_area
    let l_shape_total_area := 4 * l_shape_area
    let center_square_area := total_area - l_shape_total_area
    s^2 = center_square_area → s = 49 :=
by
  intro total_area l_shape_area l_shape_total_area center_square_area h
  sorry

end center_square_side_length_l222_222402


namespace shifted_parabola_correct_l222_222640

-- Define original equation of parabola
def original_parabola (x : ℝ) : ℝ := 2 * x^2 - 1

-- Define shifted equation of parabola
def shifted_parabola (x : ℝ) : ℝ := 2 * (x + 1)^2 - 1

-- Proof statement: the expression of the new parabola after shifting 1 unit to the left
theorem shifted_parabola_correct :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x + 1) :=
by
  -- Proof is omitted, sorry
  sorry

end shifted_parabola_correct_l222_222640


namespace min_value_l222_222991

open Real

theorem min_value (x : ℝ) (hx : x > 0) : 6 * x + 1 / x^2 ≥ 7 * (6 ^ (1 / 3)) :=
sorry

end min_value_l222_222991


namespace solution_set_of_inequality_group_l222_222821

theorem solution_set_of_inequality_group (x : ℝ) : (x > -3 ∧ x < 5) ↔ (-3 < x ∧ x < 5) :=
by
  sorry

end solution_set_of_inequality_group_l222_222821


namespace discriminant_of_quadratic_is_321_l222_222429

-- Define the quadratic equation coefficients
def a : ℝ := 4
def b : ℝ := -9
def c : ℝ := -15

-- Define the discriminant formula
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- The proof statement
theorem discriminant_of_quadratic_is_321 : discriminant a b c = 321 := by
  sorry

end discriminant_of_quadratic_is_321_l222_222429


namespace convert_3241_quinary_to_septenary_l222_222330

/-- Convert quinary number 3241_(5) to septenary number, yielding 1205_(7). -/
theorem convert_3241_quinary_to_septenary : 
  let quinary := 3 * 5^3 + 2 * 5^2 + 4 * 5^1 + 1 * 5^0
  let septenary := 1 * 7^3 + 2 * 7^2 + 0 * 7^1 + 5 * 7^0
  quinary = 446 → septenary = 1205 :=
by
  intros
  -- Quinary to Decimal
  have h₁ : 3 * 5^3 + 2 * 5^2 + 4 * 5^1 + 1 * 5^0 = 446 := by norm_num
  -- Decimal to Septenary
  have h₂ : 446 = 1 * 7^3 + 2 * 7^2 + 0 * 7^1 + 5 * 7^0 := by norm_num
  exact sorry

end convert_3241_quinary_to_septenary_l222_222330


namespace tire_price_l222_222548

theorem tire_price (x : ℕ) (h : 4 * x + 5 = 485) : x = 120 :=
by
  sorry

end tire_price_l222_222548


namespace julia_error_approx_97_percent_l222_222645

theorem julia_error_approx_97_percent (x : ℝ) : 
  abs ((6 * x - x / 6) / (6 * x) * 100 - 97) < 1 :=
by 
  sorry

end julia_error_approx_97_percent_l222_222645


namespace find_a4_l222_222057

def seq (a : ℕ → ℕ) (n : ℕ) : Prop :=
(∀ n, n ≥ 3 → a n = a (n - 1) + a (n - 2))

theorem find_a4 (a : ℕ → ℕ) (h_seq : seq a) (h_a7 : a 7 = 42) (h_a9 : a 9 = 110) : a 4 = 10 :=
by
  sorry

end find_a4_l222_222057


namespace trigonometric_identity_proof_l222_222636

theorem trigonometric_identity_proof (α : ℝ) (h : Real.tan α = 3) : (Real.sin (2 * α)) / ((Real.cos α) ^ 2) = 6 :=
by
  sorry

end trigonometric_identity_proof_l222_222636


namespace negation_of_p_l222_222923

open Classical

-- Define proposition p
def p : Prop := ∀ x : ℝ, x^2 + x > 2

-- Define the negation of proposition p
def not_p : Prop := ∃ x : ℝ, x^2 + x ≤ 2

theorem negation_of_p : ¬p ↔ not_p :=
by sorry

end negation_of_p_l222_222923


namespace rectangular_coords_of_neg_theta_l222_222721

theorem rectangular_coords_of_neg_theta 
  (x y z : ℝ) 
  (rho theta phi : ℝ)
  (hx : x = 8)
  (hy : y = 6)
  (hz : z = -3)
  (h_rho : rho = Real.sqrt (x^2 + y^2 + z^2))
  (h_cos_phi : Real.cos phi = z / rho)
  (h_sin_phi : Real.sin phi = Real.sqrt (1 - (Real.cos phi)^2))
  (h_tan_theta : Real.tan theta = y / x) :
  (rho * Real.sin phi * Real.cos (-theta), rho * Real.sin phi * Real.sin (-theta), rho * Real.cos phi) = (8, -6, -3) := 
  sorry

end rectangular_coords_of_neg_theta_l222_222721


namespace extreme_value_sum_l222_222366

noncomputable def f (m n x : ℝ) : ℝ := x^3 + 3 * m * x^2 + n * x + m^2

theorem extreme_value_sum (m n : ℝ) (h1 : f m n (-1) = 0) (h2 : (deriv (f m n)) (-1) = 0) : m + n = 11 := 
sorry

end extreme_value_sum_l222_222366


namespace part1_part2_l222_222750

noncomputable def condition1 : Prop :=
∀ (x y : ℝ), x - 2 * y + 1 = 0

noncomputable def condition2 (p : ℝ) : Prop :=
∀ (x y : ℝ), y^2 = 2 * p * x ∧ p > 0

noncomputable def condition3 (A B : (ℝ × ℝ)) (p : ℝ) : Prop :=
dist A B = 4 * real.sqrt 15 ∧
(∀ (x y : ℝ), (x - 2 * y + 1 = 0) ∧ (y^2 = 2 * p * x))

noncomputable def condition4 (C : ℝ) : Prop :=
F = (C, 0)

noncomputable def condition5 (M N F : (ℝ × ℝ)) : Prop :=
∃ (x1 y1 x2 y2 : ℝ), F = (1, 0) ∧
M = (x1, y1) ∧ N = (x2, y2) ∧ (x1 + x2) / 2 = C ∧ (y1 + y2) / 2 = 0 ∧ (F.1 * x1 + F.2 * y1) * (F.1 * x2 + F.2 * y2) = 0

theorem part1 :
  (∃ (p : ℝ), condition1 ∧ condition2 p ∧ condition3 (1, 0) (2, 0) p) →
  (p = 2) :=
  sorry

theorem part2 :
  (∃ (M N F : (ℝ × ℝ)), condition4 4 ∧ condition5 M N F) →
  (∀ (F : ℝ × ℝ), F = (1,0) →
  F + ℝ∧
  M=(1, C) N=(2, 409)∧
  ((M-product length eqad 0)MFN) =
  (12-8 * (2))) :=
 sorry

end part1_part2_l222_222750


namespace find_c_for_same_solution_l222_222994

theorem find_c_for_same_solution (c : ℝ) (x : ℝ) :
  (3 * x + 5 = 1) ∧ (c * x + 15 = -5) → c = 15 :=
by
  sorry

end find_c_for_same_solution_l222_222994


namespace inspection_probability_l222_222483

noncomputable def defective_items : ℕ := 2
noncomputable def good_items : ℕ := 3
noncomputable def total_items : ℕ := defective_items + good_items

/-- Given 2 defective items and 3 good items mixed together,
the probability that the inspection stops exactly after
four inspections is 3/5 --/
theorem inspection_probability :
  (2 * (total_items - 1) * total_items / (total_items * (total_items - 1) * (total_items - 2) * (total_items - 3))) = (3 / 5) :=
by
  sorry

end inspection_probability_l222_222483


namespace cos_alpha_is_negative_four_fifths_l222_222887

variable (α : ℝ)
variable (H1 : Real.sin α = 3 / 5)
variable (H2 : π / 2 < α ∧ α < π)

theorem cos_alpha_is_negative_four_fifths (H1 : Real.sin α = 3 / 5) (H2 : π / 2 < α ∧ α < π) :
  Real.cos α = -4 / 5 :=
sorry

end cos_alpha_is_negative_four_fifths_l222_222887


namespace exists_difference_divisible_by_11_l222_222620

theorem exists_difference_divisible_by_11 (a : Fin 12 → ℤ) :
  ∃ (i j : Fin 12), i ≠ j ∧ 11 ∣ (a i - a j) :=
  sorry

end exists_difference_divisible_by_11_l222_222620


namespace intersect_at_one_point_l222_222457

-- Define the equations as given in the conditions
def equation1 (b : ℝ) (x : ℝ) : ℝ := b * x ^ 2 + 2 * x + 2
def equation2 (x : ℝ) : ℝ := -2 * x - 2

-- Statement of the theorem
theorem intersect_at_one_point (b : ℝ) :
  (∀ x : ℝ, equation1 b x = equation2 x → x = 1) ↔ b = 1 := sorry

end intersect_at_one_point_l222_222457


namespace approx_values_relationship_l222_222428

theorem approx_values_relationship : 
  (∃ a b : ℝ, 2.35 ≤ a ∧ a ≤ 2.44 ∧ 2.395 ≤ b ∧ b ≤ 2.404 ∧ a = b) ∧
  (∃ a b : ℝ, 2.35 ≤ a ∧ a ≤ 2.44 ∧ 2.395 ≤ b ∧ b ≤ 2.404 ∧ a > b) ∧
  (∃ a b : ℝ, 2.35 ≤ a ∧ a ≤ 2.44 ∧ 2.395 ≤ b ∧ b ≤ 2.404 ∧ a < b) :=
by sorry

end approx_values_relationship_l222_222428


namespace quadratic_function_solution_l222_222349

noncomputable def g (x : ℝ) : ℝ := x^2 + 44 * x + 50

theorem quadratic_function_solution (c d : ℝ)
  (h : ∀ x, (g (g x + x)) / (g x) = x^2 + 44 * x + 50) :
  c = 44 ∧ d = 50 :=
by
  sorry

end quadratic_function_solution_l222_222349


namespace largest_quantity_l222_222988

noncomputable def D := (2007 / 2006) + (2007 / 2008)
noncomputable def E := (2007 / 2008) + (2009 / 2008)
noncomputable def F := (2008 / 2007) + (2008 / 2009)

theorem largest_quantity : D > E ∧ D > F :=
by { sorry }

end largest_quantity_l222_222988


namespace simplify_eval_expr_l222_222257

noncomputable def a : ℝ := (Real.sqrt 2) + 1
noncomputable def b : ℝ := (Real.sqrt 2) - 1

theorem simplify_eval_expr (a b : ℝ) (ha : a = (Real.sqrt 2) + 1) (hb : b = (Real.sqrt 2) - 1) : 
  (a^2 - b^2) / a / (a + (2 * a * b + b^2) / a) = Real.sqrt 2 / 2 :=
by
  sorry

end simplify_eval_expr_l222_222257


namespace twelve_integers_divisible_by_eleven_l222_222618

theorem twelve_integers_divisible_by_eleven (a : Fin 12 → ℤ) : 
  ∃ (i j : Fin 12), i ≠ j ∧ 11 ∣ (a i - a j) :=
by
  sorry

end twelve_integers_divisible_by_eleven_l222_222618


namespace walking_rate_ratio_l222_222850

theorem walking_rate_ratio (R R' : ℝ) (usual_time early_time : ℝ) (H1 : usual_time = 42) (H2 : early_time = 36) 
(H3 : R * usual_time = R' * early_time) : (R' / R = 7 / 6) :=
by
  -- proof to be completed
  sorry

end walking_rate_ratio_l222_222850


namespace polynomial_degree_is_five_l222_222875

noncomputable def expr1 := λ (x : ℚ), x^3
noncomputable def expr2 := λ (x : ℚ), x^2 - 1 / x^2
noncomputable def expr3 := λ (x : ℚ), 1 - 1 / x + 1 / x^3

noncomputable def product := λ (x : ℚ), (expr1(x) * expr2(x)) * expr3(x)
noncomputable def degree := polynomial.degree (polynomial.COEFFS product)

theorem polynomial_degree_is_five : ∀ x : ℚ, polynomial.degree (product x) = 5 := 
by
  sorry

end polynomial_degree_is_five_l222_222875


namespace find_fourth_number_l222_222061

def nat_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, n ≥ 3 → a n = a (n - 1) + a (n - 2)

variable {a : ℕ → ℕ}

theorem find_fourth_number (h_seq : nat_sequence a) (h7 : a 7 = 42) (h9 : a 9 = 110) : a 4 = 10 :=
by
  -- Placeholder for proof steps
  sorry

end find_fourth_number_l222_222061


namespace ellipse_equation_and_lambda_mu_constant_l222_222794

theorem ellipse_equation_and_lambda_mu_constant :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧
  (eccentricity : ℝ) = 2 * real.sqrt 2 / 3 ∧
  (circle_radius : ℝ) = 3 ∧ -- radius obtained from the given circle equation x^2 + y^2 = 9
  (equation : (a : ℝ) (b : ℝ) -> Prop)
    (equation 9 1) ∧ -- (a^2 = 9 and b^2 = 1 which formats as the equation of ellipse)
  (∃ (Q : ℝ × ℝ) (RM MQ NQ : ℝ × ℝ) (λ μ : ℝ), Q = (1,0) ∧ 
     λ + μ = -9 / 4) :=
sorry

end ellipse_equation_and_lambda_mu_constant_l222_222794


namespace select_n_based_on_expected_profit_prob_of_one_failure_is_correct_l222_222963

-- Define the given conditions
def company_conditions := {
  num_production_lines : ℕ := 3,
  prob_failure : ℝ := 1 / 3,
  salary_worker : ℝ := 10000,
  profit_no_failure : ℝ := 120000,
  profit_repaired_failure : ℝ := 80000,
  profit_unrepaired_failure : ℝ := 0,
  max_failures : ℕ := 1
}

-- Define the probabilities and expected profits under the given conditions
def prob_exactly_one_failure (C : company_conditions) : ℝ :=
  nat.choose C.num_production_lines 1 * (C.prob_failure) ^ 1 * (1 - C.prob_failure) ^ (C.num_production_lines - 1)

def expected_profit (C : company_conditions) (n : ℕ) : ℝ :=
  let p0 := (1 - C.prob_failure) ^ C.num_production_lines in -- Probability of no failure
  let p1 := nat.choose C.num_production_lines 1 * (C.prob_failure) ^ 1 * (1 - C.prob_failure) ^ (C.num_production_lines - 1) in -- Probability of exactly 1 failure
  let p2 := nat.choose C.num_production_lines 2 * (C.prob_failure) ^ 2 * (1 - C.prob_failure) ^ (C.num_production_lines - 2) in -- Probability of exactly 2 failures
  let p3 := (C.prob_failure) ^ 3 in -- Probability of exactly 3 failures
  (p0 * (3 * C.profit_no_failure - n * C.salary_worker) +
   p1 * (2 * C.profit_no_failure + C.profit_repaired_failure - n * C.salary_worker) +
   p2 * (C.profit_no_failure + 2 * C.profit_repaired_failure - n * C.salary_worker) +
   p3 * (2 * C.profit_repaired_failure - n * C.salary_worker)) / 1000 -- in thousand dollars

theorem select_n_based_on_expected_profit (C : company_conditions) : 
  expected_profit C 2 > expected_profit C 1 := 
sorry

-- State the theorem for the probability of exactly 1 failure
theorem prob_of_one_failure_is_correct (C : company_conditions) :
  prob_exactly_one_failure C = 4 / 9 := 
sorry

end select_n_based_on_expected_profit_prob_of_one_failure_is_correct_l222_222963


namespace range_of_a_l222_222756

noncomputable def f (x : ℝ) := (Real.log x) / x
noncomputable def g (x a : ℝ) := -Real.exp 1 * x^2 + a * x

theorem range_of_a (a : ℝ) : (∀ x1 : ℝ, ∃ x2 ∈ Set.Icc (1/3) 2, f x1 ≤ g x2 a) → 2 ≤ a :=
sorry

end range_of_a_l222_222756


namespace Teresa_age_at_Michiko_birth_l222_222677

-- Definitions of the conditions
def Teresa_age_now : ℕ := 59
def Morio_age_now : ℕ := 71
def Morio_age_at_Michiko_birth : ℕ := 38

-- Prove that Teresa was 26 years old when she gave birth to Michiko.
theorem Teresa_age_at_Michiko_birth : 38 - (71 - 59) = 26 := by
  -- Provide the proof here
  sorry

end Teresa_age_at_Michiko_birth_l222_222677


namespace centroid_path_is_ellipse_l222_222006

theorem centroid_path_is_ellipse
  (b r : ℝ)
  (C : ℝ → ℝ × ℝ)
  (H1 : ∃ t θ, C t = (r * Real.cos θ, r * Real.sin θ))
  (G : ℝ → ℝ × ℝ)
  (H2 : ∀ t, G t = (1 / 3 * (b + (C t).fst), 1 / 3 * ((C t).snd))) :
  ∃ a c : ℝ, ∀ t, (G t).fst^2 / a^2 + (G t).snd^2 / c^2 = 1 :=
sorry

end centroid_path_is_ellipse_l222_222006


namespace quadratic_roots_real_find_m_value_l222_222002

theorem quadratic_roots_real (m : ℝ) (h_roots : ∃ x1 x2 : ℝ, x1 * x1 + 4 * x1 + (m - 1) = 0 ∧ x2 * x2 + 4 * x2 + (m - 1) = 0) :
  m ≤ 5 :=
by {
  sorry
}

theorem find_m_value (m : ℝ) (x1 x2 : ℝ) (h_eq1 : x1 * x1 + 4 * x1 + (m - 1) = 0) (h_eq2 : x2 * x2 + 4 * x2 + (m - 1) = 0) (h_cond : 2 * (x1 + x2) + x1 * x2 + 10 = 0) :
  m = -1 :=
by {
  sorry
}

end quadratic_roots_real_find_m_value_l222_222002


namespace percent_both_correct_l222_222701

-- Definitions of the given percentages
def A : ℝ := 75
def B : ℝ := 25
def N : ℝ := 20

-- The proof problem statement
theorem percent_both_correct (A B N : ℝ) (hA : A = 75) (hB : B = 25) (hN : N = 20) : A + B - N - 100 = 20 :=
by
  sorry

end percent_both_correct_l222_222701


namespace find_line_equation_through_ellipse_midpoint_l222_222625

theorem find_line_equation_through_ellipse_midpoint {A B : ℝ × ℝ} 
  (hA : (A.fst^2 / 2) + A.snd^2 = 1) 
  (hB : (B.fst^2 / 2) + B.snd^2 = 1) 
  (h_midpoint : (A.fst + B.fst) / 2 = 1 ∧ (A.snd + B.snd) / 2 = 1 / 2) : 
  ∃ k : ℝ, (k = -1) ∧ (∀ x y : ℝ, (y - 1/2 = k * (x - 1)) → 2*x + 2*y - 3 = 0) :=
sorry

end find_line_equation_through_ellipse_midpoint_l222_222625


namespace fewer_popsicle_sticks_l222_222808

theorem fewer_popsicle_sticks :
  let boys := 10
  let girls := 12
  let sticks_per_boy := 15
  let sticks_per_girl := 12
  let boys_total := boys * sticks_per_boy
  let girls_total := girls * sticks_per_girl
  boys_total - girls_total = 6 := 
by
  let boys := 10
  let girls := 12
  let sticks_per_boy := 15
  let sticks_per_girl := 12
  let boys_total := boys * sticks_per_boy
  let girls_total := girls * sticks_per_girl
  show boys_total - girls_total = 6
  sorry

end fewer_popsicle_sticks_l222_222808


namespace sample_size_eq_100_l222_222449

variables (frequency : ℕ) (frequency_rate : ℚ)

theorem sample_size_eq_100 (h1 : frequency = 50) (h2 : frequency_rate = 0.5) :
  frequency / frequency_rate = 100 :=
by
  sorry

end sample_size_eq_100_l222_222449


namespace transform_cos_function_l222_222947

theorem transform_cos_function :
  ∀ x : ℝ, 2 * Real.cos (x + π / 3) =
           2 * Real.cos (2 * (x - π / 12) + π / 6) := 
sorry

end transform_cos_function_l222_222947


namespace math_problem_l222_222010

theorem math_problem (a b : ℝ) (h : |a + 1| + (b - 2)^2 = 0) : (a + b)^9 + a^6 = 2 :=
sorry

end math_problem_l222_222010


namespace second_derivative_at_pi_over_3_l222_222522

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x) * (Real.cos x)

theorem second_derivative_at_pi_over_3 : 
  (deriv (deriv f)) (Real.pi / 3) = -1 :=
  sorry

end second_derivative_at_pi_over_3_l222_222522


namespace opinion_change_difference_l222_222322

variables (initial_enjoy final_enjoy initial_not_enjoy final_not_enjoy : ℕ)
variables (n : ℕ) -- number of students in the class

-- Given conditions
def initial_conditions :=
  initial_enjoy = 40 * n / 100 ∧ initial_not_enjoy = 60 * n / 100

def final_conditions :=
  final_enjoy = 80 * n / 100 ∧ final_not_enjoy = 20 * n / 100

-- The theorem to prove
theorem opinion_change_difference :
  initial_conditions n initial_enjoy initial_not_enjoy →
  final_conditions n final_enjoy final_not_enjoy →
  (40 ≤ initial_enjoy + 20 ∧ 40 ≤ initial_not_enjoy + 20 ∧
  max_change = 60 ∧ min_change = 40 → max_change - min_change = 20) := 
  sorry

end opinion_change_difference_l222_222322


namespace math_problem_l222_222751

noncomputable def parabola (p : ℝ) := {x : ℝ × ℝ // x.2 ^ 2 = 2 * p * x.1}
def line (x y : ℝ) : Prop := x - 2 * y + 1 = 0
def focus (p : ℝ) : ℝ × ℝ := (p, 0)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem math_problem (p : ℝ)
  (h_line_parabola_intersect : ∃ A B : ℝ × ℝ, line A.1 A.2 ∧ line B.1 B.2 ∧ parabola p A ∧ parabola p B ∧ dist A B = 4 * real.sqrt 15)
  (h_focus : ∃ F : ℝ × ℝ, F = focus p)
  (h_points_MN : ∃ M N : ℝ × ℝ, parabola p M ∧ parabola p N ∧ dot_product (M - (p, 0)) (N - (p, 0)) = 0) :
  p = 2 ∧ (∃ M N : ℝ × ℝ, parabola 2 M ∧ parabola 2 N ∧ dot_product (M - (2, 0)) (N - (2, 0)) = 0) ∧
  (∀ M N : ℝ × ℝ, parabola 2 M ∧ parabola 2 N ∧ dot_product (M - (2, 0)) (N - (2, 0)) = 0 → 
  (1/2) * |M.1 * N.2 - M.2 * N.1| = 12 - 8 * real.sqrt 2) := sorry

end math_problem_l222_222751


namespace sin_alpha_beta_gamma_values_l222_222894

open Real

theorem sin_alpha_beta_gamma_values (α β γ : ℝ)
  (h1 : sin α = sin (α + β + γ) + 1)
  (h2 : sin β = 3 * sin (α + β + γ) + 2)
  (h3 : sin γ = 5 * sin (α + β + γ) + 3) :
  sin α * sin β * sin γ = (3/64) ∨ sin α * sin β * sin γ = (1/8) :=
sorry

end sin_alpha_beta_gamma_values_l222_222894


namespace term_of_sequence_l222_222205

def S (n : ℕ) : ℚ := n^2 + 2/3

def a (n : ℕ) : ℚ :=
  if n = 1 then 5/3
  else 2 * n - 1

theorem term_of_sequence (n : ℕ) : a n = 
  if n = 1 then S n 
  else S n - S (n - 1) :=
by
  sorry

end term_of_sequence_l222_222205


namespace log_product_l222_222341

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_product (x y : ℝ) (hx : 0 < x) (hy : 1 < y) :
  log_base (y^3) x * log_base (x^4) (y^3) * log_base (y^5) (x^2) * log_base (x^2) (y^5) * log_base (y^3) (x^4) =
  (1/3) * log_base y x :=
by
  sorry

end log_product_l222_222341


namespace treasure_probability_l222_222969

def prob_treasure := (1 : ℚ) / 4
def prob_traps := (1 : ℚ) / 12
def prob_neither := (2 : ℚ) / 3
def num_islands := 8
def num_treasure_islands := 5

theorem treasure_probability :
  (∃ (n k : ℕ), 
    n = num_islands ∧ 
    k = num_treasure_islands ∧ 
    ( (Nat.choose n k : ℚ) * (prob_treasure ^ k) * (prob_neither ^ (n - k)) = 7 / 432)
  ) :=
sorry

end treasure_probability_l222_222969


namespace machining_defect_probability_l222_222890

theorem machining_defect_probability :
  let defect_rate_process1 := 0.03
  let defect_rate_process2 := 0.05
  let non_defective_rate_process1 := 1 - defect_rate_process1
  let non_defective_rate_process2 := 1 - defect_rate_process2
  let non_defective_rate := non_defective_rate_process1 * non_defective_rate_process2
  let defective_rate := 1 - non_defective_rate
  defective_rate = 0.0785 :=
by
  sorry

end machining_defect_probability_l222_222890


namespace unique_prime_triplets_l222_222610

theorem unique_prime_triplets (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) :
  (p ∣ 1 + q^r) ∧ (q ∣ 1 + r^p) ∧ (r ∣ 1 + p^q) ↔ (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 5 ∧ q = 3 ∧ r = 2) ∨ (p = 3 ∧ q = 2 ∧ r = 5) := 
by
  sorry

end unique_prime_triplets_l222_222610


namespace pizza_volume_one_piece_l222_222445

theorem pizza_volume_one_piece
  (thickness : ℝ)
  (diameter : ℝ)
  (pieces : ℝ)
  (h : thickness = 1/2)
  (d : diameter = 16)
  (p : pieces = 8) :
  ∃ (volume_one_piece : ℝ), volume_one_piece = 4 * Real.pi :=
by 
  rcases (pi * (d / 2) ^ 2 * h) / p with v;
  use v;
  sorry

end pizza_volume_one_piece_l222_222445


namespace count_7_digit_nums_180_reversible_count_7_digit_nums_180_reversible_divis_by_4_sum_of_7_digit_nums_180_reversible_l222_222980

open Nat

def num180Unchanged : Nat := 
  let valid_pairs := [(0, 0), (1, 1), (8, 8), (6, 9), (9, 6)];
  let middle_digits := [0, 1, 8];
  (valid_pairs.length) * ((valid_pairs.length + 1) * (valid_pairs.length + 1) * middle_digits.length)

def num180UnchangedDivBy4 : Nat :=
  let valid_div4_pairs := [(0, 0), (1, 6), (6, 0), (6, 8), (8, 0), (8, 8), (9, 6)];
  let middle_digits := [0, 1, 8];
  valid_div4_pairs.length * (valid_div4_pairs.length / 5) * middle_digits.length

def sum180UnchangedNumbers : Nat :=
   1959460200 -- The sum by the given problem

theorem count_7_digit_nums_180_reversible : num180Unchanged = 300 :=
sorry

theorem count_7_digit_nums_180_reversible_divis_by_4 : num180UnchangedDivBy4 = 75 :=
sorry

theorem sum_of_7_digit_nums_180_reversible : sum180UnchangedNumbers = 1959460200 :=
sorry

end count_7_digit_nums_180_reversible_count_7_digit_nums_180_reversible_divis_by_4_sum_of_7_digit_nums_180_reversible_l222_222980


namespace probability_slope_le_one_l222_222657

noncomputable def point := (ℝ × ℝ)

def Q_in_unit_square (Q : point) : Prop :=
  0 ≤ Q.1 ∧ Q.1 ≤ 1 ∧ 0 ≤ Q.2 ∧ Q.2 ≤ 1

def slope_le_one (Q : point) : Prop :=
  (Q.2 - (1/4)) / (Q.1 - (3/4)) ≤ 1

theorem probability_slope_le_one :
  ∃ p q : ℕ, Q_in_unit_square Q → slope_le_one Q →
  p.gcd q = 1 ∧ (p + q = 11) :=
sorry

end probability_slope_le_one_l222_222657


namespace customers_non_holiday_l222_222144

theorem customers_non_holiday (h : ∀ n, 2 * n = 350) (H : ∃ h : ℕ, h * 8 = 2800) : (2800 / 8 / 2 = 175) :=
by sorry

end customers_non_holiday_l222_222144


namespace local_min_c_value_l222_222489

-- Definition of the function f(x) with its local minimum condition
def f (x c : ℝ) := x * (x - c)^2

-- Theorem stating that for the given function f(x) to have a local minimum at x = 1, the value of c must be 1
theorem local_min_c_value (c : ℝ) (h : ∀ ε > 0, f 1 ε < f c ε) : c = 1 := sorry

end local_min_c_value_l222_222489


namespace most_frequent_data_is_mode_l222_222090

-- Define the options
inductive Options where
  | Mean
  | Mode
  | Median
  | Frequency

-- Define the problem statement
def mostFrequentDataTerm (freqMost : String) : Options :=
  if freqMost == "Mode" then 
    Options.Mode
  else if freqMost == "Mean" then 
    Options.Mean
  else if freqMost == "Median" then 
    Options.Median
  else 
    Options.Frequency

-- Statement of the problem as a theorem
theorem most_frequent_data_is_mode (freqMost : String) :
  mostFrequentDataTerm freqMost = Options.Mode :=
by
  sorry

end most_frequent_data_is_mode_l222_222090


namespace jelly_bean_probabilities_l222_222716

theorem jelly_bean_probabilities :
  let p_red := 0.15
  let p_orange := 0.35
  let p_yellow := 0.2
  let p_green := 0.3
  p_red + p_orange + p_yellow + p_green = 1 :=
by
  sorry

end jelly_bean_probabilities_l222_222716


namespace log_expression_value_l222_222985

noncomputable def log_expression : ℝ :=
  (Real.log (Real.sqrt 27) + Real.log 8 - 3 * Real.log (Real.sqrt 10)) / Real.log 1.2

theorem log_expression_value : log_expression = 3 / 2 :=
  sorry

end log_expression_value_l222_222985


namespace two_cos_45_eq_sqrt_two_l222_222416

theorem two_cos_45_eq_sqrt_two
  (h1 : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2) :
  2 * Real.cos (Real.pi / 4) = Real.sqrt 2 :=
sorry

end two_cos_45_eq_sqrt_two_l222_222416


namespace shadow_length_of_flagpole_is_correct_l222_222570

noncomputable def length_of_shadow_flagpole : ℕ :=
  let h_flagpole : ℕ := 18
  let shadow_building : ℕ := 60
  let h_building : ℕ := 24
  let similar_conditions : Prop := true
  45

theorem shadow_length_of_flagpole_is_correct :
  length_of_shadow_flagpole = 45 := by
  sorry

end shadow_length_of_flagpole_is_correct_l222_222570


namespace simplify_expr_l222_222591

noncomputable def expr : ℝ := Real.sqrt 12 - 3 * Real.sqrt (1 / 3) + Real.sqrt 27 + (Real.pi + 1)^0

theorem simplify_expr : expr = 4 * Real.sqrt 3 + 1 := by
  sorry

end simplify_expr_l222_222591


namespace total_distance_traveled_l222_222840

variable (vm vr t d_up d_down : ℝ)
variable (H_river_speed : vr = 3)
variable (H_row_speed : vm = 6)
variable (H_time : t = 1)

theorem total_distance_traveled (H_upstream : d_up = vm - vr) 
                                (H_downstream : d_down = vm + vr) 
                                (total_time : d_up / (vm - vr) + d_down / (vm + vr) = t) : 
                                2 * (d_up + d_down) = 4.5 := 
                                by
  sorry

end total_distance_traveled_l222_222840


namespace find_a4_l222_222058

def seq (a : ℕ → ℕ) (n : ℕ) : Prop :=
(∀ n, n ≥ 3 → a n = a (n - 1) + a (n - 2))

theorem find_a4 (a : ℕ → ℕ) (h_seq : seq a) (h_a7 : a 7 = 42) (h_a9 : a 9 = 110) : a 4 = 10 :=
by
  sorry

end find_a4_l222_222058


namespace fraction_twins_l222_222302

variables (P₀ I E P_f f : ℕ) (x : ℚ)

def initial_population := P₀ = 300000
def immigrants := I = 50000
def emigrants := E = 30000
def pregnant_fraction := f = 1 / 8
def final_population := P_f = 370000

theorem fraction_twins :
  initial_population P₀ ∧ immigrants I ∧ emigrants E ∧ pregnant_fraction f ∧ final_population P_f →
  x = 1 / 4 :=
by
  sorry

end fraction_twins_l222_222302


namespace triangle_median_equiv_l222_222784

-- Assuming necessary non-computable definitions (e.g., α for angles, R for real numbers) and non-computable nature of some geometric properties.

noncomputable def triangle (A B C : ℝ) := 
A + B + C = Real.pi

noncomputable def length_a (R A : ℝ) : ℝ := 2 * R * Real.sin A
noncomputable def length_b (R B : ℝ) : ℝ := 2 * R * Real.sin B
noncomputable def length_c (R C : ℝ) : ℝ := 2 * R * Real.sin C

noncomputable def median_a (b c A : ℝ) : ℝ := (2 * b * c) / (b + c) * Real.cos (A / 2)

theorem triangle_median_equiv (A B C R : ℝ) (hA : triangle A B C) :
  (1 / (length_a R A) + 1 / (length_b R B) = 1 / (median_a (length_b R B) (length_c R C) A)) ↔ (C = 2 * Real.pi / 3) := 
by sorry

end triangle_median_equiv_l222_222784


namespace set_complement_intersection_l222_222896

open Set

variable (U M N : Set ℕ)

theorem set_complement_intersection :
  U = {1, 2, 3, 4, 5, 6, 7} →
  M = {3, 4, 5} →
  N = {1, 3, 6} →
  {2, 7} = (U \ M) ∩ (U \ N) :=
by
  intros hU hM hN
  rw [hU, hM, hN]
  sorry

end set_complement_intersection_l222_222896


namespace yellow_lights_count_l222_222943

theorem yellow_lights_count (total_lights : ℕ) (red_lights : ℕ) (blue_lights : ℕ) (yellow_lights : ℕ) :
  total_lights = 95 → red_lights = 26 → blue_lights = 32 → yellow_lights = total_lights - (red_lights + blue_lights) → yellow_lights = 37 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end yellow_lights_count_l222_222943


namespace expected_value_unfair_die_l222_222192

theorem expected_value_unfair_die :
  let p8 := 3 / 8
  let p1_7 := (1 - p8) / 7
  let E := p1_7 * (1 + 2 + 3 + 4 + 5 + 6 + 7) + p8 * 8
  E = 5.5 := by
  sorry

end expected_value_unfair_die_l222_222192


namespace count_valid_integers_l222_222613

theorem count_valid_integers :
  {n | (1 ≤ n ∧ n ≤ 100) ∧ (∃ k : ℕ, k * ((n + 1)! ^ (n + 1)) = ((n + 1)^2 - 1)!)}.card = 97 :=
by sorry

end count_valid_integers_l222_222613


namespace gain_percent_l222_222843

variable (C S : ℝ)
variable (h : 65 * C = 50 * S)

theorem gain_percent (h : 65 * C = 50 * S) : (S - C) / C * 100 = 30 :=
by
  sorry

end gain_percent_l222_222843


namespace second_term_of_geometric_series_l222_222867

theorem second_term_of_geometric_series (a r S term2 : ℝ) 
  (h1 : r = 1 / 4)
  (h2 : S = 40)
  (h3 : S = a / (1 - r))
  (h4 : term2 = a * r) : 
  term2 = 7.5 := 
  by
  sorry

end second_term_of_geometric_series_l222_222867


namespace candy_cost_correct_l222_222517

-- Given conditions:
def given_amount : ℝ := 1.00
def change_received : ℝ := 0.46

-- Define candy cost based on given conditions
def candy_cost : ℝ := given_amount - change_received

-- Statement to be proved
theorem candy_cost_correct : candy_cost = 0.54 := 
by
  sorry

end candy_cost_correct_l222_222517


namespace find_m_l222_222150

theorem find_m (m : ℚ) : 
  (∃ m, (∀ x y z : ℚ, ((x, y) = (2, 9) ∨ (x, y) = (15, m) ∨ (x, y) = (35, 4)) ∧ 
  (∀ a b c d e f : ℚ, ((a, b) = (2, 9) ∨ (a, b) = (15, m) ∨ (a, b) = (35, 4)) → 
  ((b - d) / (a - c) = (f - d) / (e - c))) → m = 232 / 33)) :=
sorry

end find_m_l222_222150


namespace sandwiches_difference_l222_222529

-- Define the number of sandwiches Samson ate at lunch on Monday
def sandwichesLunchMonday : ℕ := 3

-- Define the number of sandwiches Samson ate at dinner on Monday (twice as many as lunch)
def sandwichesDinnerMonday : ℕ := 2 * sandwichesLunchMonday

-- Define the total number of sandwiches Samson ate on Monday
def totalSandwichesMonday : ℕ := sandwichesLunchMonday + sandwichesDinnerMonday

-- Define the number of sandwiches Samson ate for breakfast on Tuesday
def sandwichesBreakfastTuesday : ℕ := 1

-- Define the total number of sandwiches Samson ate on Tuesday
def totalSandwichesTuesday : ℕ := sandwichesBreakfastTuesday

-- Define the number of more sandwiches Samson ate on Monday than on Tuesday
theorem sandwiches_difference : totalSandwichesMonday - totalSandwichesTuesday = 8 :=
by
  sorry

end sandwiches_difference_l222_222529


namespace teresa_age_at_michiko_birth_l222_222675

noncomputable def Teresa_age_now : ℕ := 59
noncomputable def Morio_age_now : ℕ := 71
noncomputable def Morio_age_at_Michiko_birth : ℕ := 38

theorem teresa_age_at_michiko_birth :
  (Teresa_age_now - (Morio_age_now - Morio_age_at_Michiko_birth)) = 26 := 
by
  sorry

end teresa_age_at_michiko_birth_l222_222675


namespace no_real_roots_of_quadratic_l222_222688

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b ^ 2 - 4 * a * c

theorem no_real_roots_of_quadratic :
  let a := 2
  let b := -5
  let c := 6
  discriminant a b c < 0 → ¬∃ x : ℝ, 2 * x ^ 2 - 5 * x + 6 = 0 :=
by {
  -- Proof skipped
  sorry
}

end no_real_roots_of_quadratic_l222_222688


namespace circle_diameter_l222_222345
open Real

theorem circle_diameter (A : ℝ) (hA : A = 50.26548245743669) : ∃ d : ℝ, d = 8 :=
by
  sorry

end circle_diameter_l222_222345


namespace range_of_a_l222_222491

theorem range_of_a (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 < 1 ∧ x2 > 1 ∧ x1 * x1 + (a * a - 1) * x1 + a - 2 = 0 ∧ x2 * x2 + (a * a - 1) * x2 + a - 2 = 0) ↔ -2 < a ∧ a < 1 :=
sorry

end range_of_a_l222_222491


namespace find_m_squared_plus_n_squared_l222_222959

theorem find_m_squared_plus_n_squared (m n : ℝ) (h1 : (m - n) ^ 2 = 8) (h2 : (m + n) ^ 2 = 2) : m ^ 2 + n ^ 2 = 5 :=
by
  sorry

end find_m_squared_plus_n_squared_l222_222959


namespace solve_inequality_l222_222537

theorem solve_inequality (x : ℝ) :
  (0 ≤ x^2 - x - 2 ∧ x^2 - x - 2 ≤ 4) ↔
  (-2 ≤ x ∧ x ≤ -1) ∨ (2 ≤ x ∧ x ≤ 3) :=
by sorry

end solve_inequality_l222_222537


namespace black_dogs_count_l222_222271

def number_of_brown_dogs := 20
def number_of_white_dogs := 10
def total_number_of_dogs := 45
def number_of_black_dogs := total_number_of_dogs - (number_of_brown_dogs + number_of_white_dogs)

theorem black_dogs_count : number_of_black_dogs = 15 := by
  sorry

end black_dogs_count_l222_222271


namespace inequality_proof_l222_222913

theorem inequality_proof (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z))) + (y^3 / ((1 + z) * (1 + x))) + (z^3 / ((1 + x) * (1 + y))) ≥ 3 / 4 :=
by {
  sorry
}

end inequality_proof_l222_222913


namespace number_of_articles_l222_222718

-- Define main conditions
variable (N : ℕ) -- Number of articles
variable (CP SP : ℝ) -- Cost price and Selling price per article

-- Condition 1: Cost price of N articles equals the selling price of 15 articles
def condition1 : Prop := N * CP = 15 * SP

-- Condition 2: Selling price includes a 33.33% profit on cost price
def condition2 : Prop := SP = CP * 1.3333

-- Prove that the number of articles N equals 20
theorem number_of_articles (h1 : condition1 N CP SP) (h2 : condition2 CP SP) : N = 20 :=
by sorry

end number_of_articles_l222_222718


namespace simplify_expression_l222_222296

theorem simplify_expression : 4 * Real.sqrt 5 + Real.sqrt 45 - Real.sqrt 8 + 4 * Real.sqrt 2 = 7 * Real.sqrt 5 + 2 * Real.sqrt 2 :=
by sorry

end simplify_expression_l222_222296


namespace pears_for_apples_l222_222770

-- Define the costs of apples, oranges, and pears.
variables {cost_apples cost_oranges cost_pears : ℕ}

-- Condition 1: Ten apples cost the same as five oranges
axiom apples_equiv_oranges : 10 * cost_apples = 5 * cost_oranges

-- Condition 2: Three oranges cost the same as four pears
axiom oranges_equiv_pears : 3 * cost_oranges = 4 * cost_pears

-- Theorem: Tyler can buy 13 pears for the price of 20 apples
theorem pears_for_apples : 20 * cost_apples = 13 * cost_pears :=
sorry

end pears_for_apples_l222_222770


namespace smallest_prime_factor_2379_l222_222559

-- Define the given number
def n : ℕ := 2379

-- Define the condition that 3 is a prime number.
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

-- Define the smallest prime factor
def smallest_prime_factor (n p : ℕ) : Prop :=
  is_prime p ∧ p ∣ n ∧ (∀ q, is_prime q → q ∣ n → p ≤ q)

-- The statement that 3 is the smallest prime factor of 2379
theorem smallest_prime_factor_2379 : smallest_prime_factor n 3 :=
sorry

end smallest_prime_factor_2379_l222_222559


namespace candidate_majority_votes_l222_222505

theorem candidate_majority_votes (total_votes : ℕ) (candidate_percentage other_percentage : ℕ) 
  (h_total_votes : total_votes = 5200)
  (h_candidate_percentage : candidate_percentage = 60)
  (h_other_percentage : other_percentage = 40) :
  (candidate_percentage * total_votes / 100) - (other_percentage * total_votes / 100) = 1040 := 
by
  sorry

end candidate_majority_votes_l222_222505


namespace james_two_point_shots_l222_222513

-- Definitions based on conditions
def field_goals := 13
def field_goal_points := 3
def total_points := 79

-- Statement to be proven
theorem james_two_point_shots :
  ∃ x : ℕ, 79 = (field_goals * field_goal_points) + (2 * x) ∧ x = 20 :=
by
  sorry

end james_two_point_shots_l222_222513


namespace bicycle_cost_price_l222_222156

-- Definitions of conditions
def profit_22_5_percent (x : ℝ) : ℝ := 1.225 * x
def loss_14_3_percent (x : ℝ) : ℝ := 0.857 * x
def profit_32_4_percent (x : ℝ) : ℝ := 1.324 * x
def loss_7_8_percent (x : ℝ) : ℝ := 0.922 * x
def discount_5_percent (x : ℝ) : ℝ := 0.95 * x
def tax_6_percent (x : ℝ) : ℝ := 1.06 * x

theorem bicycle_cost_price (CP_A : ℝ) (TP_E : ℝ) (h : TP_E = 295.88) : 
  CP_A = 295.88 / 1.29058890594 :=
by
  sorry

end bicycle_cost_price_l222_222156


namespace total_work_completed_in_18_days_l222_222135

theorem total_work_completed_in_18_days :
  let amit_work_rate := 1/10
  let ananthu_work_rate := 1/20
  let amit_days := 2
  let amit_work_done := amit_days * amit_work_rate
  let remaining_work := 1 - amit_work_done
  let ananthu_days := remaining_work / ananthu_work_rate
  amit_days + ananthu_days = 18 := 
by
  sorry

end total_work_completed_in_18_days_l222_222135


namespace eating_time_proof_l222_222524

noncomputable def combined_eating_time (time_fat time_thin weight : ℝ) : ℝ :=
  let rate_fat := 1 / time_fat
  let rate_thin := 1 / time_thin
  let combined_rate := rate_fat + rate_thin
  weight / combined_rate

theorem eating_time_proof :
  let time_fat := 12
  let time_thin := 40
  let weight := 5
  combined_eating_time time_fat time_thin weight = (600 / 13) :=
by
  -- placeholder for the proof
  sorry

end eating_time_proof_l222_222524


namespace quadratic_solution1_quadratic_solution2_l222_222400

theorem quadratic_solution1 (x : ℝ) :
  (x^2 + 4 * x - 4 = 0) ↔ (x = -2 + 2 * Real.sqrt 2 ∨ x = -2 - 2 * Real.sqrt 2) :=
by sorry

theorem quadratic_solution2 (x : ℝ) :
  ((x - 1)^2 = 2 * (x - 1)) ↔ (x = 1 ∨ x = 3) :=
by sorry

end quadratic_solution1_quadratic_solution2_l222_222400


namespace min_trials_to_ensure_pass_l222_222314

theorem min_trials_to_ensure_pass (p : ℝ) (n : ℕ) (h₁ : p = 3 / 4) (h₂ : n ≥ 1): 
  (1 - (1 - p) ^ n) > 0.99 → n ≥ 4 :=
by sorry

end min_trials_to_ensure_pass_l222_222314


namespace eulers_formula_l222_222355

structure PlanarGraph :=
(vertices : ℕ)
(edges : ℕ)
(faces : ℕ)
(connected : Prop)

theorem eulers_formula (G: PlanarGraph) (H_conn: G.connected) : G.vertices - G.edges + G.faces = 2 :=
sorry

end eulers_formula_l222_222355


namespace sum_first_8_terms_of_geom_seq_l222_222361

-- Definitions: the sequence a_n, common ratio q, and the fact that specific terms form an arithmetic sequence.
def geom_seq (a : ℕ → ℕ) (a1 : ℕ) (q : ℕ) := ∀ n, a n = a1 * q^(n-1)
def arith_seq (b c d : ℕ) := 2 * b + (c - 2 * b) = d

-- Conditions
variables {a : ℕ → ℕ} {a1 : ℕ} {q : ℕ}
variables (h1 : geom_seq a a1 q) (h2 : q = 2)
variables (h3 : arith_seq (2 * a 4) (a 6) 48)

-- Goal: sum of the first 8 terms of the sequence equals 255
def sum_geometric_sequence (a1 : ℕ) (q : ℕ) (n : ℕ) := a1 * (1 - q^n) / (1 - q)

theorem sum_first_8_terms_of_geom_seq : 
  sum_geometric_sequence a1 q 8 = 255 :=
by
  sorry

end sum_first_8_terms_of_geom_seq_l222_222361


namespace quotient_of_division_l222_222263

theorem quotient_of_division (L S Q : ℕ) (h1 : L - S = 2500) (h2 : L = 2982) (h3 : L = Q * S + 15) : Q = 6 := 
sorry

end quotient_of_division_l222_222263


namespace value_of_a7_l222_222892

-- Let \( \{a_n\} \) be a sequence such that \( S_n \) denotes the sum of the first \( n \) terms.
-- Given \( S_{n+1}, S_{n+2}, S_{n+3} \) form an arithmetic sequence and \( a_2 = -2 \),
-- prove that \( a_7 = 64 \).

theorem value_of_a7 (S : ℕ → ℝ) (a : ℕ → ℝ) :
  (∀ n : ℕ, S (n + 2) + S (n + 1) = 2 * S n) →
  a 2 = -2 →
  (∀ n : ℕ, a (n + 2) = -2 * a (n + 1)) →
  a 7 = 64 :=
by
  -- skip the proof
  sorry

end value_of_a7_l222_222892


namespace union_sets_l222_222769

open Set

variable {α : Type*}

def setA : Set ℝ := { x | -2 < x ∧ x < 0 }
def setB : Set ℝ := { x | -1 < x ∧ x < 1 }
def setC : Set ℝ := { x | -2 < x ∧ x < 1 }

theorem union_sets : setA ∪ setB = setC := 
by {
  sorry
}

end union_sets_l222_222769


namespace price_of_red_car_l222_222795

noncomputable def car_price (total_amount loan_amount interest_rate : ℝ) : ℝ :=
  loan_amount + (total_amount - loan_amount) / (1 + interest_rate)

theorem price_of_red_car :
  car_price 38000 20000 0.15 = 35000 :=
by sorry

end price_of_red_car_l222_222795


namespace isosceles_triangle_base_length_l222_222403

theorem isosceles_triangle_base_length (a b c : ℝ) (h₀ : a = 5) (h₁ : b = 5) (h₂ : a + b + c = 17) : c = 7 :=
by
  -- proof would go here
  sorry

end isosceles_triangle_base_length_l222_222403


namespace find_fourth_number_l222_222070

theorem find_fourth_number (a : ℕ → ℕ) 
  (h1 : ∀ n, n ≥ 2 → a n = a (n - 1) + a (n - 2)) 
  (h2 : a 6 = 42) 
  (h3 : a 8 = 110) : 
  a 3 = 10 := 
sorry

end find_fourth_number_l222_222070


namespace exercise_l222_222211

noncomputable def g (x : ℝ) : ℝ := x^3
noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 1

theorem exercise : f (g 3) = 1457 := by
  sorry

end exercise_l222_222211


namespace arrangement_of_mississippi_no_adjacent_s_l222_222468

-- Conditions: The word "MISSISSIPPI" has 11 letters with specific frequencies: 1 M, 4 I's, 4 S's, 2 P's.
-- No two S's can be adjacent.
def ways_to_arrange_mississippi_no_adjacent_s: Nat :=
  let total_non_s_arrangements := Nat.factorial 7 / (Nat.factorial 4 * Nat.factorial 2)
  let gaps_for_s := Nat.choose 8 4
  total_non_s_arrangements * gaps_for_s

theorem arrangement_of_mississippi_no_adjacent_s : ways_to_arrange_mississippi_no_adjacent_s = 7350 :=
by
  unfold ways_to_arrange_mississippi_no_adjacent_s
  sorry

end arrangement_of_mississippi_no_adjacent_s_l222_222468


namespace range_of_x_squared_plus_y_squared_l222_222546

def increasing (f : ℝ → ℝ) := ∀ x y, x < y → f x < f y
def symmetric_about_origin (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem range_of_x_squared_plus_y_squared 
  (f : ℝ → ℝ) 
  (h_incr : increasing f) 
  (h_symm : symmetric_about_origin f) 
  (h_ineq : ∀ x y, f (x^2 - 6 * x) + f (y^2 - 8 * y + 24) < 0) : 
  ∀ x y, 16 < x^2 + y^2 ∧ x^2 + y^2 < 36 := 
sorry

end range_of_x_squared_plus_y_squared_l222_222546


namespace new_students_l222_222780

theorem new_students (S_i : ℕ) (L : ℕ) (S_f : ℕ) (N : ℕ) 
  (h₁ : S_i = 11) 
  (h₂ : L = 6) 
  (h₃ : S_f = 47) 
  (h₄ : S_f = S_i - L + N) : 
  N = 42 :=
by 
  rw [h₁, h₂, h₃] at h₄
  sorry

end new_students_l222_222780


namespace isabella_haircut_length_l222_222383

-- Define the original length of Isabella's hair.
def original_length : ℕ := 18

-- Define the length of hair cut off.
def cut_off_length : ℕ := 9

-- The length of Isabella's hair after the haircut.
def length_after_haircut : ℕ := original_length - cut_off_length

-- Statement of the theorem we want to prove.
theorem isabella_haircut_length : length_after_haircut = 9 :=
by
  sorry

end isabella_haircut_length_l222_222383


namespace expression_evaluation_l222_222617

-- Define the variables and the given condition
variables (x y : ℝ)

-- Define the equation condition
def equation_condition : Prop := x - 3 * y = 4

-- State the theorem
theorem expression_evaluation (h : equation_condition x y) : 15 * y - 5 * x + 6 = -14 :=
by
  sorry

end expression_evaluation_l222_222617


namespace loan_difference_is_979_l222_222161

noncomputable def compounded_interest (P r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def simple_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r * t)

noncomputable def loan_difference (P : ℝ) : ℝ :=
  let compounded_7_years := compounded_interest P 0.08 12 7
  let half_payment := compounded_7_years / 2
  let remaining_balance := compounded_interest half_payment 0.08 12 8
  let total_compounded := half_payment + remaining_balance
  let total_simple := simple_interest P 0.10 15
  abs (total_compounded - total_simple)

theorem loan_difference_is_979 : loan_difference 15000 = 979 := sorry

end loan_difference_is_979_l222_222161


namespace find_number_l222_222707

theorem find_number : ∃ n : ℝ, 50 + (5 * n) / (180 / 3) = 51 ∧ n = 12 := 
by
  use 12
  sorry

end find_number_l222_222707


namespace Kenny_played_basketball_for_10_hours_l222_222652

theorem Kenny_played_basketball_for_10_hours
  (played_basketball ran practiced_trumpet : ℕ)
  (H1 : practiced_trumpet = 40)
  (H2 : ran = 2 * played_basketball)
  (H3 : practiced_trumpet = 2 * ran) :
  played_basketball = 10 :=
by
  sorry

end Kenny_played_basketball_for_10_hours_l222_222652


namespace original_number_l222_222854

theorem original_number (x : ℝ) (h : 1.2 * x = 1080) : x = 900 := by
  sorry

end original_number_l222_222854


namespace ethanol_total_amount_l222_222977

-- Definitions based on Conditions
def total_tank_capacity : ℕ := 214
def fuel_A_volume : ℕ := 106
def fuel_B_volume : ℕ := total_tank_capacity - fuel_A_volume
def ethanol_in_fuel_A : ℚ := 0.12
def ethanol_in_fuel_B : ℚ := 0.16

-- Theorem Statement
theorem ethanol_total_amount :
  (fuel_A_volume * ethanol_in_fuel_A + fuel_B_volume * ethanol_in_fuel_B) = 30 := 
sorry

end ethanol_total_amount_l222_222977


namespace find_a4_l222_222051

open Nat

def sequence (a : Nat → Nat) :=
  ∀ n, n ≥ 3 → a n = a (n - 1) + a (n - 2)

theorem find_a4 (a : ℕ → ℕ)
  (h_seq : sequence a)
  (h_a7 : a 7 = 42)
  (h_a9 : a 9 = 110) :
  a 4 = 10 :=
by
  sorry

end find_a4_l222_222051


namespace paul_tips_l222_222799

theorem paul_tips (P : ℕ) (h1 : P + 16 = 30) : P = 14 :=
by
  sorry

end paul_tips_l222_222799


namespace sufficient_and_necessary_condition_l222_222316

theorem sufficient_and_necessary_condition (x : ℝ) :
  (x - 2) * (x + 2) > 0 ↔ x > 2 ∨ x < -2 :=
by sorry

end sufficient_and_necessary_condition_l222_222316


namespace composite_proposition_l222_222335

noncomputable def p : Prop := ∃ x : ℝ, x^2 + 2 * x + 5 ≤ 4

noncomputable def q : Prop := ∀ x : ℝ, 0 < x ∧ x < Real.pi / 2 → ¬ (∀ v : ℝ, v = (Real.sin x + 4 / Real.sin x) → v = 4)

theorem composite_proposition : p ∧ ¬q := 
by 
  sorry

end composite_proposition_l222_222335


namespace sum_ratios_l222_222199

variable (a b d : ℕ)

def A_n (a b d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

def arithmetic_sum (a n d : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem sum_ratios (k : ℕ) (h1 : 2 * (a + d) = 7 * k) (h2 : 4 * (a + 3 * d) = 6 * k) :
  arithmetic_sum a 7 d / arithmetic_sum a 3 d = 2 / 1 :=
by
  sorry

end sum_ratios_l222_222199


namespace Joe_first_lift_weight_l222_222779

variable (F S : ℝ)

theorem Joe_first_lift_weight (h1 : F + S = 600) (h2 : 2 * F = S + 300) : F = 300 := 
sorry

end Joe_first_lift_weight_l222_222779


namespace bookshelf_prices_purchasing_plans_l222_222964

/-
We are given the following conditions:
1. 3 * x + 2 * y = 1020
2. 4 * x + 3 * y = 1440

From these conditions, we need to prove that:
1. Price of type A bookshelf (x) is 180 yuan.
2. Price of type B bookshelf (y) is 240 yuan.

Given further conditions:
1. The school plans to purchase a total of 20 bookshelves.
2. Type B bookshelves not less than type A bookshelves.
3. Maximum budget of 4320 yuan.

We need to prove that the following plans are valid:
1. 8 type A bookshelves, 12 type B bookshelves.
2. 9 type A bookshelves, 11 type B bookshelves.
3. 10 type A bookshelves, 10 type B bookshelves.
-/

theorem bookshelf_prices (x y : ℕ) 
  (h1 : 3 * x + 2 * y = 1020) 
  (h2 : 4 * x + 3 * y = 1440) : 
  x = 180 ∧ y = 240 :=
by sorry

theorem purchasing_plans (m : ℕ) 
  (h3 : 8 ≤ m ∧ m ≤ 10) 
  (h4 : 180 * m + 240 * (20 - m) ≤ 4320) 
  (h5 : 20 - m ≥ m) : 
  m = 8 ∨ m = 9 ∨ m = 10 :=
by sorry

end bookshelf_prices_purchasing_plans_l222_222964


namespace unique_prime_triplets_l222_222609

theorem unique_prime_triplets (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) :
  (p ∣ 1 + q^r) ∧ (q ∣ 1 + r^p) ∧ (r ∣ 1 + p^q) ↔ (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 5 ∧ q = 3 ∧ r = 2) ∨ (p = 3 ∧ q = 2 ∧ r = 5) := 
by
  sorry

end unique_prime_triplets_l222_222609


namespace tangent_line_equation_range_of_k_l222_222000

noncomputable def f (x : ℝ) : ℝ := x^2 - x * Real.log x

-- Part (I): Tangent line equation
theorem tangent_line_equation :
  let f (x : ℝ) := x^2 - x * Real.log x
  let p := (1 : ℝ)
  let y := f p
  (∀ x, y = x) :=
sorry

-- Part (II): Range of k
theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, 1 < x → (k / x + x / 2 - f x / x < 0)) → k ≤ 1 / 2 :=
sorry

end tangent_line_equation_range_of_k_l222_222000


namespace solve_for_z_l222_222536

theorem solve_for_z (z : ℂ) (h : 5 - 3 * (I * z) = 3 + 5 * (I * z)) : z = I / 4 :=
sorry

end solve_for_z_l222_222536


namespace isosceles_triangle_perimeter_l222_222508

noncomputable def perimeter_of_isosceles_triangle : ℝ :=
  let BC := 10
  let height := 6
  let half_base := BC / 2
  let side := Real.sqrt (height^2 + half_base^2)
  let perimeter := 2 * side + BC
  perimeter

theorem isosceles_triangle_perimeter :
  let BC := 10
  let height := 6
  perimeter_of_isosceles_triangle = 2 * Real.sqrt (height^2 + (BC / 2)^2) + BC := by
  sorry

end isosceles_triangle_perimeter_l222_222508


namespace find_general_formula_l222_222381

section sequence

variables {R : Type*} [LinearOrderedField R]
variable (c : R)
variable (h_c : c ≠ 0)

def seq (a : Nat → R) : Prop :=
  a 1 = 1 ∧ ∀ n : Nat, n > 0 → a (n + 1) = c * a n + c^(n + 1) * (2 * n + 1)

def general_formula (a : Nat → R) : Prop :=
  ∀ n : Nat, n > 0 → a n = (n^2 - 1) * c^n + c^(n - 1)

theorem find_general_formula :
  ∃ a : Nat → R, seq c a ∧ general_formula c a :=
by
  sorry

end sequence

end find_general_formula_l222_222381


namespace shoe_length_size_15_l222_222293

theorem shoe_length_size_15 : 
  ∀ (length : ℕ → ℝ), 
    (∀ n, 8 ≤ n ∧ n ≤ 17 → length (n + 1) = length n + 1 / 4) → 
    length 17 = (1 + 0.10) * length 8 →
    length 15 = 24.25 :=
by
  intro length h_increase h_largest
  sorry

end shoe_length_size_15_l222_222293


namespace pedestrian_walking_time_in_interval_l222_222155

noncomputable def bus_departure_interval : ℕ := 5  -- Condition 1: Buses depart every 5 minutes
noncomputable def buses_same_direction : ℕ := 11  -- Condition 2: 11 buses passed him going the same direction
noncomputable def buses_opposite_direction : ℕ := 13  -- Condition 3: 13 buses came from opposite direction
noncomputable def bus_speed_factor : ℕ := 8  -- Condition 4: Bus speed is 8 times the pedestrian's speed
noncomputable def min_walking_time : ℚ := 57 + 1 / 7 -- Correct Answer: Minimum walking time
noncomputable def max_walking_time : ℚ := 62 + 2 / 9 -- Correct Answer: Maximum walking time

theorem pedestrian_walking_time_in_interval (t : ℚ)
  (h1 : bus_departure_interval = 5)
  (h2 : buses_same_direction = 11)
  (h3 : buses_opposite_direction = 13)
  (h4 : bus_speed_factor = 8) :
  min_walking_time ≤ t ∧ t ≤ max_walking_time :=
sorry

end pedestrian_walking_time_in_interval_l222_222155


namespace find_a4_l222_222052

open Nat

def sequence (a : Nat → Nat) :=
  ∀ n, n ≥ 3 → a n = a (n - 1) + a (n - 2)

theorem find_a4 (a : ℕ → ℕ)
  (h_seq : sequence a)
  (h_a7 : a 7 = 42)
  (h_a9 : a 9 = 110) :
  a 4 = 10 :=
by
  sorry

end find_a4_l222_222052


namespace decreasing_even_function_condition_l222_222747

theorem decreasing_even_function_condition (f : ℝ → ℝ) 
    (h1 : ∀ x y : ℝ, x < y → y < 0 → f y < f x) 
    (h2 : ∀ x : ℝ, f (-x) = f x) : f 13 < f 9 ∧ f 9 < f 1 := 
by
  sorry

end decreasing_even_function_condition_l222_222747


namespace algebraic_notation_equivalence_l222_222168

-- Define the variables
variables (x y : ℤ)

-- Define "three times x"
def three_times_x (x : ℤ) := 3 * x

-- Define "the cube of y"
def cube_of_y (y : ℤ) := y ^ 3

-- Define the target expression
def target_expression (x y : ℤ) := three_times_x x - cube_of_y y

-- Theorem stating the equivalence of the target expression to 3x - y^3
theorem algebraic_notation_equivalence : target_expression x y = 3 * x - y ^ 3 := by
  sorry

end algebraic_notation_equivalence_l222_222168


namespace price_of_second_tea_l222_222909

theorem price_of_second_tea (P : ℝ) (h1 : 1 * 64 + 1 * P = 2 * 69) : P = 74 := 
by
  sorry

end price_of_second_tea_l222_222909


namespace eq_has_exactly_one_real_root_l222_222159

theorem eq_has_exactly_one_real_root : ∀ x : ℝ, 2007 * x^3 + 2006 * x^2 + 2005 * x = 0 ↔ x = 0 :=
by
sorry

end eq_has_exactly_one_real_root_l222_222159


namespace unique_two_digit_solution_l222_222093

theorem unique_two_digit_solution : ∃! (t : ℕ), 10 ≤ t ∧ t < 100 ∧ 13 * t % 100 = 52 := sorry

end unique_two_digit_solution_l222_222093


namespace preferred_point_condition_l222_222759

theorem preferred_point_condition (x y : ℝ) (h₁ : x^2 + y^2 ≤ 2008)
  (cond : ∀ x' y', (x'^2 + y'^2 ≤ 2008) → (x' ≤ x → y' ≥ y) → (x = x' ∧ y = y')) :
  x^2 + y^2 = 2008 ∧ x ≤ 0 ∧ y ≥ 0 :=
by
  sorry

end preferred_point_condition_l222_222759


namespace intersection_complement_l222_222391

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem intersection_complement :
  A ∩ (U \ B) = {1, 3} :=
by {
  -- To ensure the validity of the theorem, the proof goes here
  sorry
}

end intersection_complement_l222_222391


namespace diagonal_of_square_l222_222958

theorem diagonal_of_square (length_rect width_rect : ℝ) (h1 : length_rect = 45) (h2 : width_rect = 40)
  (area_rect : ℝ) (h3 : area_rect = length_rect * width_rect) (area_square : ℝ) (h4 : area_square = area_rect)
  (side_square : ℝ) (h5 : side_square^2 = area_square) (diagonal_square : ℝ) (h6 : diagonal_square = side_square * Real.sqrt 2) :
  diagonal_square = 60 := by
  sorry

end diagonal_of_square_l222_222958


namespace maximum_sum_of_diagonals_of_rhombus_l222_222575

noncomputable def rhombus_side_length : ℝ := 5
noncomputable def diagonal_bd_max_length : ℝ := 6
noncomputable def diagonal_ac_min_length : ℝ := 6
noncomputable def max_diagonal_sum : ℝ := 14

theorem maximum_sum_of_diagonals_of_rhombus :
  ∀ (s bd ac : ℝ), 
  s = rhombus_side_length → 
  bd ≤ diagonal_bd_max_length → 
  ac ≥ diagonal_ac_min_length → 
  bd + ac ≤ max_diagonal_sum → 
  max_diagonal_sum = 14 :=
by
  sorry

end maximum_sum_of_diagonals_of_rhombus_l222_222575


namespace digit_possibilities_757_l222_222439

theorem digit_possibilities_757
  (N : ℕ)
  (h : N < 10) :
  (∃ d₀ d₁ d₂ : ℕ, (d₀ = 2 ∨ d₀ = 5 ∨ d₀ = 8) ∧
  (d₁ = 2 ∨ d₁ = 5 ∨ d₁ = 8) ∧
  (d₂ = 2 ∨ d₂ = 5 ∨ d₂ = 8) ∧
  (d₀ ≠ d₁) ∧
  (d₀ ≠ d₂) ∧
  (d₁ ≠ d₂)) :=
by
  sorry

end digit_possibilities_757_l222_222439


namespace b_coordinates_bc_equation_l222_222487

section GeometryProof

-- Define point A
def A : ℝ × ℝ := (1, 1)

-- Altitude CD has the equation: 3x + y - 12 = 0
def altitude_CD (x y : ℝ) : Prop := 3 * x + y - 12 = 0

-- Angle bisector BE has the equation: x - 2y + 4 = 0
def angle_bisector_BE (x y : ℝ) : Prop := x - 2 * y + 4 = 0

-- Coordinates of point B
def B : ℝ × ℝ := (-8, -2)

-- Equation of line BC
def line_BC (x y : ℝ) : Prop := 9 * x - 13 * y + 46 = 0

-- Proof statement for the coordinates of point B
theorem b_coordinates : ∃ x y : ℝ, (x, y) = B :=
by sorry

-- Proof statement for the equation of line BC
theorem bc_equation : ∃ (f : ℝ → ℝ → Prop), f = line_BC :=
by sorry

end GeometryProof

end b_coordinates_bc_equation_l222_222487


namespace average_speed_l222_222720

theorem average_speed (x y : ℝ) (h₁ : 0 ≤ x) (h₂ : 0 ≤ y)
  (total_time : x / 4 + y / 3 + y / 6 + x / 4 = 5) :
  (2 * (x + y)) / 5 = 4 :=
by
  sorry

end average_speed_l222_222720


namespace apples_to_pears_l222_222772

theorem apples_to_pears (a o p : ℕ) 
  (h1 : 10 * a = 5 * o) 
  (h2 : 3 * o = 4 * p) : 
  (20 * a) = 40 / 3 * p :=
sorry

end apples_to_pears_l222_222772


namespace cos_double_angle_l222_222194

open Real

theorem cos_double_angle (α : Real) (h : tan α = 3) : cos (2 * α) = -4/5 :=
  sorry

end cos_double_angle_l222_222194


namespace sample_size_is_13_l222_222305

noncomputable def stratified_sample_size : ℕ :=
  let A := 120
  let B := 80
  let C := 60
  let total_units := A + B + C
  let sampled_C_units := 3
  let sampling_fraction := sampled_C_units / C
  let n := sampling_fraction * total_units
  n

theorem sample_size_is_13 :
  stratified_sample_size = 13 := by
  sorry

end sample_size_is_13_l222_222305


namespace solution1_solution2_l222_222749

noncomputable def problem1 (x y : ℝ) (p : ℝ) : Prop :=
  x - 2 * y + 1 = 0 ∧ y^2 = 2 * p * x ∧ 0 < p ∧ (abs (sqrt (1 + 4) * (y - y))) = 4 * sqrt 15

theorem solution1 (p: ℝ) : p = 2 :=
  sorry

noncomputable def problem2 (x y m n : ℝ) : Prop :=
  y^2 = 4 * x ∧ ∃ (F : ℝ × ℝ), F = (1, 0) ∧
  (∀ (M N : ℝ × ℝ), M ∈ y^2 = 4 * x ∧ N ∈ y^2 = 4 * x ∧ (F.1 - M.1) * (F.2 - N.1) + (F.2 - M.2) * (F.2 - N.2) = 0 →
  let area := (1/2) * abs ((N.1 - M.1) * (F.2 - M.2) - (N.2 - M.2) * (F.1 - M.1)) in
  ∃ min_area : ℝ, min_area = 12 - 8 * sqrt 2)

theorem solution2 (x y m n : ℝ) : ∃ min_area : ℝ, min_area = 12 - 8 * sqrt 2 :=
  sorry

end solution1_solution2_l222_222749


namespace rides_with_remaining_tickets_l222_222295

theorem rides_with_remaining_tickets (T_total : ℕ) (T_spent : ℕ) (C_ride : ℕ)
  (h1 : T_total = 40) (h2 : T_spent = 28) (h3 : C_ride = 4) :
  (T_total - T_spent) / C_ride = 3 := by
  sorry

end rides_with_remaining_tickets_l222_222295


namespace john_paid_8000_l222_222514

-- Define the variables according to the conditions
def upfront_fee : ℕ := 1000
def hourly_rate : ℕ := 100
def court_hours : ℕ := 50
def prep_hours : ℕ := 2 * court_hours
def total_hours : ℕ := court_hours + prep_hours
def total_fee : ℕ := upfront_fee + total_hours * hourly_rate
def john_share : ℕ := total_fee / 2

-- Prove that John's share is $8,000
theorem john_paid_8000 : john_share = 8000 :=
by sorry

end john_paid_8000_l222_222514


namespace sequences_with_both_properties_are_constant_l222_222334

-- Definitions according to the problem's conditions
def arithmetic_sequence (seq : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, seq (n + 1) - seq n = seq (n + 2) - seq (n + 1)

def geometric_sequence (seq : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, seq (n + 1) / seq n = seq (n + 2) / seq (n + 1)

-- Definition of the sequence properties combined
def arithmetic_and_geometric_sequence (seq : ℕ → ℝ) : Prop :=
  arithmetic_sequence seq ∧ geometric_sequence seq

-- Problem to prove
theorem sequences_with_both_properties_are_constant (seq : ℕ → ℝ) :
  arithmetic_and_geometric_sequence seq → ∀ n m : ℕ, seq n = seq m :=
sorry

end sequences_with_both_properties_are_constant_l222_222334


namespace find_alpha_minus_beta_find_cos_2alpha_minus_beta_l222_222226

-- Definitions and assumptions
variables (α β : ℝ)
axiom sin_alpha : Real.sin α = (Real.sqrt 5) / 5
axiom sin_beta : Real.sin β = (3 * Real.sqrt 10) / 10
axiom alpha_acute : 0 < α ∧ α < Real.pi / 2
axiom beta_acute : 0 < β ∧ β < Real.pi / 2

-- Statement to prove α - β = -π/4
theorem find_alpha_minus_beta : α - β = -Real.pi / 4 :=
sorry

-- Given α - β = -π/4, statement to prove cos(2α - β) = 3√10 / 10
theorem find_cos_2alpha_minus_beta (h : α - β = -Real.pi / 4) : Real.cos (2 * α - β) = (3 * Real.sqrt 10) / 10 :=
sorry

end find_alpha_minus_beta_find_cos_2alpha_minus_beta_l222_222226


namespace taxi_fare_80_miles_l222_222320

theorem taxi_fare_80_miles (fare_60 : ℝ) (flat_rate : ℝ) (proportional_rate : ℝ) (d : ℝ) (charge_60 : ℝ) 
  (h1 : fare_60 = 150) (h2 : flat_rate = 20) (h3 : proportional_rate * 60 = charge_60) (h4 : charge_60 = (fare_60 - flat_rate)) 
  (h5 : proportional_rate * 80 = d - flat_rate) : d = 193 := 
by
  sorry

end taxi_fare_80_miles_l222_222320


namespace number_of_pens_each_student_gets_l222_222273

theorem number_of_pens_each_student_gets 
    (total_pens : ℕ) (total_pencils : ℕ) (max_students : ℕ)
    (h1 : total_pens = 1001) (h2 : total_pencils = 910) (h3 : max_students = 91) :
  (total_pens / Nat.gcd total_pens total_pencils) = 11 :=
by
  sorry

end number_of_pens_each_student_gets_l222_222273


namespace rate_of_markup_l222_222045

theorem rate_of_markup (S : ℝ) (hS : S = 8)
  (profit_percent : ℝ) (h_profit_percent : profit_percent = 0.20)
  (expense_percent : ℝ) (h_expense_percent : expense_percent = 0.10) :
  (S - (S * (1 - profit_percent - expense_percent))) / (S * (1 - profit_percent - expense_percent)) * 100 = 42.857 :=
by
  sorry

end rate_of_markup_l222_222045


namespace jack_marathon_time_l222_222035

theorem jack_marathon_time :
  ∀ {marathon_distance : ℝ} {jill_time : ℝ} {speed_ratio : ℝ},
    marathon_distance = 40 → 
    jill_time = 4 → 
    speed_ratio = 0.888888888888889 → 
    (marathon_distance / (speed_ratio * (marathon_distance / jill_time))) = 4.5 :=
by
  intros marathon_distance jill_time speed_ratio h1 h2 h3
  rw [h1, h2, h3]
  sorry

end jack_marathon_time_l222_222035


namespace parallel_medians_half_angle_l222_222830

theorem parallel_medians_half_angle {ABC A'B'C' : Triangle}
  (h1 : ABC.is_right_triangle)
  (h2 : A'B'C'.is_right_triangle)
  (h3 : ABC.median_parallel_to_hypotenuse = A'B'C'.median_parallel_to_hypotenuse) :
  ∃ leg1ABC leg2A'B'C' hypotenuse_ABC hypotenuse_A'B'C',
  ∠(leg1ABC, leg2A'B'C') = 1/2 * ∠(hypotenuse_ABC, hypotenuse_A'B'C') := 
sorry

end parallel_medians_half_angle_l222_222830


namespace malachi_selfies_total_l222_222907

theorem malachi_selfies_total (x y : ℕ) 
  (h_ratio : 10 * y = 17 * x)
  (h_diff : y = x + 630) : 
  x + y = 2430 :=
sorry

end malachi_selfies_total_l222_222907


namespace coin_difference_l222_222393

/-- 
  Given that Paul has 5-cent, 20-cent, and 15-cent coins, 
  prove that the difference between the maximum and minimum number of coins
  needed to make exactly 50 cents is 6.
-/
theorem coin_difference :
  ∃ (coins : Nat → Nat),
    (coins 5 + coins 20 + coins 15) = 6 ∧
    (5 * coins 5 + 20 * coins 20 + 15 * coins 15 = 50) :=
sorry

end coin_difference_l222_222393


namespace smallest_x_undefined_l222_222983

theorem smallest_x_undefined :
  (∀ x, 10 * x^2 - 90 * x + 20 = 0 → x = 1 ∨ x = 8) → (∀ x, 10 * x^2 - 90 * x + 20 = 0 → x = 1) :=
by
  sorry

end smallest_x_undefined_l222_222983


namespace negation_proposition_l222_222684

theorem negation_proposition (m : ℤ) :
  ¬(∃ x : ℤ, x^2 + 2*x + m < 0) ↔ ∀ x : ℤ, x^2 + 2*x + m ≥ 0 :=
by
  sorry

end negation_proposition_l222_222684


namespace simplify_expression_l222_222399

variable (y : ℝ)

theorem simplify_expression : 3 * y + 4 * y^2 - 2 - (7 - 3 * y - 4 * y^2) = 8 * y^2 + 6 * y - 9 := 
  by
  sorry

end simplify_expression_l222_222399


namespace mask_donation_equation_l222_222711

theorem mask_donation_equation (x : ℝ) : 
  1 + (1 + x) + (1 + x)^2 = 4.75 :=
sorry

end mask_donation_equation_l222_222711


namespace couch_cost_l222_222732

theorem couch_cost
  (C : ℕ)  -- Cost of the couch
  (table_cost : ℕ := 100)
  (lamp_cost : ℕ := 50)
  (amount_paid : ℕ := 500)
  (amount_owed : ℕ := 400)
  (total_furniture_cost : ℕ := C + table_cost + lamp_cost)
  (remaining_amount_owed : total_furniture_cost - amount_paid = amount_owed) :
   C = 750 := 
sorry

end couch_cost_l222_222732


namespace songs_owned_initially_l222_222473

theorem songs_owned_initially (a b c : ℕ) (hc : c = a + b) (hb : b = 7) (hc_total : c = 13) :
  a = 6 :=
by
  -- Direct usage of the given conditions to conclude the proof goes here.
  sorry

end songs_owned_initially_l222_222473


namespace find_fourth_number_l222_222077

theorem find_fourth_number (a : ℕ → ℕ) (h1 : a 7 = 42) (h2 : a 9 = 110)
  (h3 : ∀ n, n ≥ 3 → a n = a (n - 1) + a (n - 2)) : a 4 = 10 := 
sorry

end find_fourth_number_l222_222077


namespace magnitude_difference_l222_222600

open Complex

noncomputable def c1 : ℂ := 18 - 5 * I
noncomputable def c2 : ℂ := 14 + 6 * I
noncomputable def c3 : ℂ := 3 - 12 * I
noncomputable def c4 : ℂ := 4 + 9 * I

theorem magnitude_difference : 
  Complex.abs ((c1 * c2) - (c3 * c4)) = Real.sqrt 146365 :=
by
  sorry

end magnitude_difference_l222_222600


namespace find_a6_l222_222510

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 2 ∧ a 2 = 5 ∧ ∀ n : ℕ, a (n + 1) = a (n + 2) + a n

theorem find_a6 (a : ℕ → ℤ) (h : seq a) : a 6 = -3 :=
by
  sorry

end find_a6_l222_222510


namespace tom_ate_one_pound_of_carrots_l222_222828

noncomputable def calories_from_carrots (C : ℝ) : ℝ := 51 * C
noncomputable def calories_from_broccoli (C : ℝ) : ℝ := (51 / 3) * (2 * C)
noncomputable def total_calories (C : ℝ) : ℝ :=
  calories_from_carrots C + calories_from_broccoli C

theorem tom_ate_one_pound_of_carrots :
  ∃ C : ℝ, total_calories C = 85 ∧ C = 1 :=
by
  use 1
  simp [total_calories, calories_from_carrots, calories_from_broccoli]
  sorry

end tom_ate_one_pound_of_carrots_l222_222828


namespace probability_all_switches_on_is_correct_l222_222289

-- Mechanical declaration of the problem
structure SwitchState :=
  (state : Fin 2003 → Bool)

noncomputable def probability_all_on (initial : SwitchState) : ℚ :=
  let satisfying_confs := 2
  let total_confs := 2 ^ 2003
  let p := satisfying_confs / total_confs
  p

-- Definition of the term we want to prove
theorem probability_all_switches_on_is_correct :
  ∀ (initial : SwitchState), probability_all_on initial = 1 / 2 ^ 2002 :=
  sorry

end probability_all_switches_on_is_correct_l222_222289


namespace total_students_correct_l222_222727

-- Definitions based on the conditions
def students_germain : Nat := 13
def students_newton : Nat := 10
def students_young : Nat := 12
def overlap_germain_newton : Nat := 2
def overlap_germain_young : Nat := 1

-- Total distinct students (using inclusion-exclusion principle)
def total_distinct_students : Nat :=
  students_germain + students_newton + students_young - overlap_germain_newton - overlap_germain_young

-- The theorem we want to prove
theorem total_students_correct : total_distinct_students = 32 :=
  by
    -- We state the computation directly; proof is omitted
    sorry

end total_students_correct_l222_222727


namespace average_coins_collected_per_day_l222_222396

noncomputable def average_coins (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  (a + (a + (n - 1) * d)) / 2

theorem average_coins_collected_per_day :
  average_coins 10 5 7 = 25 := by
  sorry

end average_coins_collected_per_day_l222_222396


namespace find_the_number_l222_222110

theorem find_the_number :
  ∃ x : ℤ, 65 + (x * 12) / (180 / 3) = 66 ∧ x = 5 :=
by
  existsi (5 : ℤ)
  sorry

end find_the_number_l222_222110


namespace monthly_increase_per_ticket_l222_222526

variable (x : ℝ)

theorem monthly_increase_per_ticket
    (initial_premium : ℝ := 50)
    (percent_increase_per_accident : ℝ := 0.10)
    (tickets : ℕ := 3)
    (final_premium : ℝ := 70) :
    initial_premium * (1 + percent_increase_per_accident) + tickets * x = final_premium → x = 5 :=
by
  intro h
  sorry

end monthly_increase_per_ticket_l222_222526


namespace purely_imaginary_complex_l222_222997

theorem purely_imaginary_complex :
  ∀ (x y : ℤ), (x - 4) ≠ 0 → (y^2 - 3*y - 4) ≠ 0 → (∃ (z : ℂ), z = ⟨0, x^2 + 3*x - 4⟩) → 
    (x = 4 ∧ y ≠ 4 ∧ y ≠ -1) :=
by
  intro x y hx hy hz
  sorry

end purely_imaginary_complex_l222_222997


namespace common_chord_l222_222265

theorem common_chord (circle1 circle2 : ℝ × ℝ → Prop)
  (h1 : ∀ x y, circle1 (x, y) ↔ x^2 + y^2 + 2 * x = 0)
  (h2 : ∀ x y, circle2 (x, y) ↔ x^2 + y^2 - 4 * y = 0) :
  ∀ x y, circle1 (x, y) ∧ circle2 (x, y) ↔ x + 2 * y = 0 := 
by
  sorry

end common_chord_l222_222265


namespace evaluate_imaginary_expression_l222_222987

theorem evaluate_imaginary_expression (i : ℂ) (h_i2 : i^2 = -1) (h_i4 : i^4 = 1) :
  i^14 + i^19 + i^24 + i^29 + 3 * i^34 + 2 * i^39 = -3 - 2 * i :=
by sorry

end evaluate_imaginary_expression_l222_222987


namespace real_root_quadratic_l222_222332

theorem real_root_quadratic (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 9 = 0) ↔ b ≤ -6 ∨ b ≥ 6 := 
sorry

end real_root_quadratic_l222_222332


namespace coconut_grove_nut_yield_l222_222503

theorem coconut_grove_nut_yield (x : ℕ) (Y : ℕ) 
  (h1 : (x + 4) * 60 + x * 120 + (x - 4) * Y = 3 * x * 100)
  (h2 : x = 8) : Y = 180 := 
by
  sorry

end coconut_grove_nut_yield_l222_222503


namespace smallest_whole_number_l222_222127

theorem smallest_whole_number :
  ∃ x : ℕ, x % 3 = 2 ∧ x % 5 = 3 ∧ x % 7 = 4 ∧ x = 23 :=
sorry

end smallest_whole_number_l222_222127


namespace vector_satisfies_condition_l222_222466

def line_l (t : ℝ) : ℝ × ℝ := (2 + 3 * t, 5 + 2 * t)
def line_m (s : ℝ) : ℝ × ℝ := (1 + 2 * s, 3 + 2 * s)

variable (A B P : ℝ × ℝ)

def vector_BA (B A : ℝ × ℝ) : ℝ × ℝ := (A.1 - B.1, A.2 - B.2)
def vector_v : ℝ × ℝ := (1, -1)

theorem vector_satisfies_condition : 
  2 * vector_v.1 - vector_v.2 = 3 := by
  sorry

end vector_satisfies_condition_l222_222466


namespace lcm_12_21_30_l222_222181

theorem lcm_12_21_30 : Nat.lcm (Nat.lcm 12 21) 30 = 420 := by
  sorry

end lcm_12_21_30_l222_222181


namespace gab_score_ratio_l222_222223

theorem gab_score_ratio (S G C O : ℕ) (h1 : S = 20) (h2 : C = 2 * G) (h3 : O = 85) (h4 : S + G + C = O + 55) :
  G / S = 2 := 
by 
  sorry

end gab_score_ratio_l222_222223


namespace total_cost_price_correct_l222_222577

def SP1 : ℝ := 120
def SP2 : ℝ := 150
def SP3 : ℝ := 200
def profit1 : ℝ := 0.20
def profit2 : ℝ := 0.25
def profit3 : ℝ := 0.10

def CP1 := SP1 / (1 + profit1)
def CP2 := SP2 / (1 + profit2)
def CP3 := SP3 / (1 + profit3)

def total_cost_price := CP1 + CP2 + CP3

theorem total_cost_price_correct : total_cost_price = 401.82 :=
by sorry

end total_cost_price_correct_l222_222577


namespace percentage_less_than_m_add_d_l222_222291

def symmetric_about_mean (P : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x, P m - x = P m + x

def within_one_stdev (P : ℝ → ℝ) (m d : ℝ) : Prop :=
  P m - d = 0.68 ∧ P m + d = 0.68

theorem percentage_less_than_m_add_d 
  (P : ℝ → ℝ) (m d : ℝ) 
  (symm : symmetric_about_mean P m)
  (within_stdev : within_one_stdev P m d) : 
  ∃ f, f = 0.84 :=
by
  sorry

end percentage_less_than_m_add_d_l222_222291


namespace right_triangle_hypotenuse_length_l222_222781

theorem right_triangle_hypotenuse_length
  (a b : ℝ)
  (ha : a = 12)
  (hb : b = 16) :
  c = 20 :=
by
  -- Placeholder for the proof
  sorry

end right_triangle_hypotenuse_length_l222_222781


namespace no_more_beverages_needed_l222_222252

namespace HydrationPlan

def daily_water_need := 9
def daily_juice_need := 5
def daily_soda_need := 3
def days := 60

def total_water_needed := daily_water_need * days
def total_juice_needed := daily_juice_need * days
def total_soda_needed := daily_soda_need * days

def water_already_have := 617
def juice_already_have := 350
def soda_already_have := 215

theorem no_more_beverages_needed :
  (water_already_have >= total_water_needed) ∧ 
  (juice_already_have >= total_juice_needed) ∧ 
  (soda_already_have >= total_soda_needed) :=
by 
  -- proof goes here
  sorry

end HydrationPlan

end no_more_beverages_needed_l222_222252


namespace n1_prime_n2_not_prime_l222_222494

def n1 := 1163
def n2 := 16424
def N := 19101112
def N_eq : N = n1 * n2 := by decide

theorem n1_prime : Prime n1 := 
sorry

theorem n2_not_prime : ¬ Prime n2 :=
sorry

end n1_prime_n2_not_prime_l222_222494


namespace otimes_evaluation_l222_222879

def otimes (a b : ℝ) : ℝ := a * b + a - b

theorem otimes_evaluation (a b : ℝ) : 
  otimes a b + otimes (b - a) b = b^2 - b := 
  by
  sorry

end otimes_evaluation_l222_222879


namespace term_in_sequence_l222_222960

   theorem term_in_sequence (n : ℕ) (h1 : 1 ≤ n) (h2 : 6 * n + 1 = 2005) : n = 334 :=
   by
     sorry
   
end term_in_sequence_l222_222960


namespace sum_of_squares_of_coeffs_l222_222952

def poly_coeffs_squared_sum (p : Polynomial ℤ) : ℤ :=
  p.coeff 5 ^ 2 + p.coeff 3 ^ 2 + p.coeff 0 ^ 2

theorem sum_of_squares_of_coeffs (p : Polynomial ℤ) (h : p = 5 * (Polynomial.C 1 * Polynomial.X ^ 5 + Polynomial.C 2 * Polynomial.X ^ 3 + Polynomial.C 3)) :
  poly_coeffs_squared_sum p = 350 :=
by
  sorry

end sum_of_squares_of_coeffs_l222_222952


namespace gift_equation_l222_222837

theorem gift_equation (x : ℝ) : 15 * (x + 40) = 900 := 
by
  sorry

end gift_equation_l222_222837


namespace min_sugar_l222_222459

variable (f s : ℝ)

theorem min_sugar (h1 : f ≥ 10 + 3 * s) (h2 : f ≤ 4 * s) : s ≥ 10 := by
  sorry

end min_sugar_l222_222459


namespace square_divided_into_40_smaller_squares_l222_222700

theorem square_divided_into_40_smaller_squares : ∃ squares : ℕ, squares = 40 :=
by
  sorry

end square_divided_into_40_smaller_squares_l222_222700


namespace function_increasing_l222_222499

variable {α : Type*} [LinearOrderedField α]

def is_increasing (f : α → α) : Prop :=
  ∀ x y : α, x < y → f x < f y

theorem function_increasing (f : α → α) (h : ∀ x1 x2 : α, x1 ≠ x2 → x1 * f x1 + x2 * f x2 > x1 * f x2 + x2 * f x1) :
  is_increasing f :=
by
  sorry

end function_increasing_l222_222499


namespace erik_orange_juice_count_l222_222474

theorem erik_orange_juice_count (initial_money bread_loaves bread_cost orange_juice_cost remaining_money : ℤ)
  (h₁ : initial_money = 86)
  (h₂ : bread_loaves = 3)
  (h₃ : bread_cost = 3)
  (h₄ : orange_juice_cost = 6)
  (h₅ : remaining_money = 59) :
  (initial_money - remaining_money - (bread_loaves * bread_cost)) / orange_juice_cost = 3 :=
by
  sorry

end erik_orange_juice_count_l222_222474


namespace functional_equation_solution_l222_222173

noncomputable def f (x : ℚ) : ℚ := sorry

theorem functional_equation_solution (f : ℚ → ℚ) (f_pos_rat : ∀ x : ℚ, 0 < x → 0 < f x) :
  (∀ x y : ℚ, 0 < x → 0 < y → f x + f y + 2 * x * y * f (x * y) = f (x * y) / f (x + y)) →
  (∀ x : ℚ, 0 < x → f x = 1 / x ^ 2) :=
by
  sorry

end functional_equation_solution_l222_222173


namespace train_crosses_platform_in_34_seconds_l222_222581

theorem train_crosses_platform_in_34_seconds 
    (train_speed_kmph : ℕ) 
    (time_cross_man_sec : ℕ) 
    (platform_length_m : ℕ) 
    (h_speed : train_speed_kmph = 72) 
    (h_time : time_cross_man_sec = 18) 
    (h_platform_length : platform_length_m = 320) 
    : (platform_length_m + (train_speed_kmph * 1000 / 3600) * time_cross_man_sec) / (train_speed_kmph * 1000 / 3600) = 34 :=
by
    sorry

end train_crosses_platform_in_34_seconds_l222_222581


namespace second_pipe_fill_time_l222_222968

theorem second_pipe_fill_time (x : ℝ) :
  let rate1 := 1 / 8
  let rate2 := 1 / x
  let combined_rate := 1 / 4.8
  rate1 + rate2 = combined_rate → x = 12 :=
by
  intros
  sorry

end second_pipe_fill_time_l222_222968


namespace triangle_inequality_inequality_l222_222533

theorem triangle_inequality_inequality {a b c p q r : ℝ}
  (h1 : a + b > c)
  (h2 : b + c > a)
  (h3 : c + a > b)
  (h4 : p + q + r = 0) :
  a^2 * p * q + b^2 * q * r + c^2 * r * p ≤ 0 :=
sorry

end triangle_inequality_inequality_l222_222533


namespace packages_katie_can_make_l222_222651

-- Definition of the given conditions
def number_of_cupcakes_baked := 18
def cupcakes_eaten_by_todd := 8
def cupcakes_per_package := 2

-- The main statement to prove
theorem packages_katie_can_make : 
  (number_of_cupcakes_baked - cupcakes_eaten_by_todd) / cupcakes_per_package = 5 :=
by
  -- Use sorry to skip the proof
  sorry

end packages_katie_can_make_l222_222651


namespace find_x_squared_plus_y_squared_l222_222766

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x * y = 6) (h2 : x^2 - y^2 + x + y = 44) : x^2 + y^2 = 109 :=
sorry

end find_x_squared_plus_y_squared_l222_222766


namespace find_fourth_number_l222_222062

def nat_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, n ≥ 3 → a n = a (n - 1) + a (n - 2)

variable {a : ℕ → ℕ}

theorem find_fourth_number (h_seq : nat_sequence a) (h7 : a 7 = 42) (h9 : a 9 = 110) : a 4 = 10 :=
by
  -- Placeholder for proof steps
  sorry

end find_fourth_number_l222_222062


namespace book_stack_sum_l222_222579

theorem book_stack_sum : 
  let a := 15 -- first term
  let d := -2 -- common difference
  let l := 1 -- last term
  -- n = (l - a) / d + 1
  let n := (l - a) / d + 1
  -- S = n * (a + l) / 2
  let S := n * (a + l) / 2
  S = 64 :=
by
  -- The given conditions
  let a := 15 -- first term
  let d := -2 -- common difference
  let l := 1 -- last term
  -- Calculate the number of terms (n)
  let n := (l - a) / d + 1
  -- Calculate the total sum (S)
  let S := n * (a + l) / 2
  -- Prove the sum is 64
  show S = 64
  sorry

end book_stack_sum_l222_222579


namespace inequality_solution_l222_222740

theorem inequality_solution (x : ℝ) (h : 0 < x) : x^3 - 9*x^2 + 52*x > 0 := 
sorry

end inequality_solution_l222_222740


namespace flower_garden_mystery_value_l222_222380

/-- Prove the value of "花园探秘" given the arithmetic sum conditions and unique digit mapping. -/
theorem flower_garden_mystery_value :
  ∀ (shu_hua_hua_yuan : ℕ) (wo_ai_tan_mi : ℕ),
  shu_hua_hua_yuan + 2011 = wo_ai_tan_mi →
  (∃ (hua yuan tan mi : ℕ),
    0 ≤ hua ∧ hua < 10 ∧
    0 ≤ yuan ∧ yuan < 10 ∧
    0 ≤ tan ∧ tan < 10 ∧
    0 ≤ mi ∧ mi < 10 ∧
    hua ≠ yuan ∧ hua ≠ tan ∧ hua ≠ mi ∧
    yuan ≠ tan ∧ yuan ≠ mi ∧ tan ≠ mi ∧
    shu_hua_hua_yuan = hua * 1000 + yuan * 100 + tan * 10 + mi ∧
    wo_ai_tan_mi = 9713) := sorry

end flower_garden_mystery_value_l222_222380


namespace days_in_april_l222_222007

-- Hannah harvests 5 strawberries daily for the whole month of April.
def harvest_per_day : ℕ := 5
-- She gives away 20 strawberries.
def strawberries_given_away : ℕ := 20
-- 30 strawberries are stolen.
def strawberries_stolen : ℕ := 30
-- She has 100 strawberries by the end of April.
def strawberries_final : ℕ := 100

theorem days_in_april : 
  ∃ (days : ℕ), (days * harvest_per_day = strawberries_final + strawberries_given_away + strawberries_stolen) :=
by
  sorry

end days_in_april_l222_222007


namespace geometric_prod_eight_l222_222359

theorem geometric_prod_eight
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h_arith : ∀ n, a n ≠ 0)
  (h_eq : a 4 + 3 * a 8 = 2 * (a 7)^2)
  (h_geom : ∀ {m n : ℕ}, b m * b (m + n) = b (2 * m + n))
  (h_b_eq_a : b 7 = a 7) :
  b 2 * b 8 * b 11 = 8 :=
sorry

end geometric_prod_eight_l222_222359


namespace number_add_thrice_number_eq_twenty_l222_222307

theorem number_add_thrice_number_eq_twenty (x : ℝ) (h : x + 3 * x = 20) : x = 5 :=
sorry

end number_add_thrice_number_eq_twenty_l222_222307


namespace polygon_sides_l222_222935

theorem polygon_sides (n : ℕ) (c : ℕ) 
  (h₁ : c = n * (n - 3) / 2)
  (h₂ : c = 2 * n) : n = 7 :=
sorry

end polygon_sides_l222_222935


namespace total_distance_travelled_l222_222442

theorem total_distance_travelled (total_time hours_foot hours_bicycle speed_foot speed_bicycle distance_foot : ℕ)
  (h1 : total_time = 7)
  (h2 : speed_foot = 8)
  (h3 : speed_bicycle = 16)
  (h4 : distance_foot = 32)
  (h5 : hours_foot = distance_foot / speed_foot)
  (h6 : hours_bicycle = total_time - hours_foot)
  (distance_bicycle := speed_bicycle * hours_bicycle) :
  distance_foot + distance_bicycle = 80 := 
by
  sorry

end total_distance_travelled_l222_222442


namespace juniper_remaining_bones_l222_222040

-- Conditions
def initial_bones : ℕ := 4
def doubled_bones (b : ℕ) : ℕ := 2 * b
def stolen_bones (b : ℕ) : ℕ := b - 2

-- Theorem Statement
theorem juniper_remaining_bones : stolen_bones (doubled_bones initial_bones) = 6 := by
  -- Proof is omitted, only the statement is required as per instructions
  sorry

end juniper_remaining_bones_l222_222040


namespace find_y_minus_x_l222_222501

theorem find_y_minus_x (x y : ℝ) (h1 : x + y = 8) (h2 : y - 3 * x = 7) : y - x = 7.5 :=
by
  sorry

end find_y_minus_x_l222_222501


namespace expected_heads_of_alice_l222_222162

noncomputable def expected_heads_alice (X Y : ℕ → ℝ) (n : ℕ) :=
  \[ \mathbb{E}[X \mid X \geq Y] = 20 \cdot \frac{2^{38} + \binom{39}{19}}{2^{39} + \binom{39}{19}} \]

theorem expected_heads_of_alice (n : ℕ) (X Y : ℕ → ℝ) :
  ( ∀ i : ℕ, X i = (0:ℝ) ∨ X i = 1) →
  ( ∀ i : ℕ, Y i = (0:ℝ) ∨ Y i = 1) →
  ( ∀ i, X(i) = X(i % n)) →
  ( ∀ i, Y(i) = Y(i % n)) →
  @expected_heads_alice X Y n =
    20 * (2^38 + Mathlib.Combinatorics.Binom.binom(39, 19)) /
    (2^39 + Mathlib.Combinatorics.Binom.binom(39, 19)) := 
sorry

end expected_heads_of_alice_l222_222162


namespace popsicle_sticks_difference_l222_222810

def popsicle_sticks_boys (boys : ℕ) (sticks_per_boy : ℕ) : ℕ :=
  boys * sticks_per_boy

def popsicle_sticks_girls (girls : ℕ) (sticks_per_girl : ℕ) : ℕ :=
  girls * sticks_per_girl

theorem popsicle_sticks_difference : 
    popsicle_sticks_boys 10 15 - popsicle_sticks_girls 12 12 = 6 := by
  sorry

end popsicle_sticks_difference_l222_222810


namespace arithmetic_geometric_sequence_sum_l222_222902

theorem arithmetic_geometric_sequence_sum 
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∃ x y z : ℝ, (x = a ∧ y = -4 ∧ z = b ∨ x = b ∧ y = -4 ∧ z = a) 
                   ∧ (x + z = 2 * y) ∧ (x * z = y^2)) : 
  a + b = 10 :=
by sorry

end arithmetic_geometric_sequence_sum_l222_222902


namespace volume_of_cube_l222_222918

theorem volume_of_cube (a : ℕ) (h : (a^3 - a = a^3 - 5)) : a^3 = 125 :=
by {
  -- The necessary algebraic manipulation follows
  sorry
}

end volume_of_cube_l222_222918


namespace solve_for_x_l222_222259

theorem solve_for_x (x : ℚ) : (2/5 : ℚ) - (1/4 : ℚ) = 1/x → x = 20/3 :=
by
  intro h
  sorry

end solve_for_x_l222_222259


namespace find_m_l222_222551

def circle1 (x y m : ℝ) : Prop := (x + 2)^2 + (y - m)^2 = 9
def circle2 (x y m : ℝ) : Prop := (x - m)^2 + (y + 1)^2 = 4

theorem find_m (m : ℝ) : 
  ∃ x1 y1 x2 y2 : ℝ, 
    circle1 x1 y1 m ∧ 
    circle2 x2 y2 m ∧ 
    (m + 2)^2 + (-1 - m)^2 = 25 → 
    m = 2 :=
by
  sorry

end find_m_l222_222551


namespace jericho_altitude_300_l222_222036

def jericho_altitude (below_sea_level : Int) : Prop :=
  below_sea_level = -300

theorem jericho_altitude_300 (below_sea_level : Int)
  (h1 : below_sea_level = -300) : jericho_altitude below_sea_level :=
by
  sorry

end jericho_altitude_300_l222_222036


namespace max_area_equilateral_triangle_in_rectangle_l222_222820

-- Define the problem parameters
def rect_width : ℝ := 12
def rect_height : ℝ := 15

-- State the theorem to be proved
theorem max_area_equilateral_triangle_in_rectangle 
  (width height : ℝ) (h_width : width = rect_width) (h_height : height = rect_height) :
  ∃ area : ℝ, area = 369 * Real.sqrt 3 - 540 := 
sorry

end max_area_equilateral_triangle_in_rectangle_l222_222820


namespace arc_length_l222_222502

-- Define the radius and central angle
def radius : ℝ := 10
def central_angle : ℝ := 240

-- Theorem to prove the arc length is (40 * π) / 3
theorem arc_length (r : ℝ) (n : ℝ) (h_r : r = radius) (h_n : n = central_angle) : 
  (n * π * r) / 180 = (40 * π) / 3 :=
by
  -- Proof omitted
  sorry

end arc_length_l222_222502


namespace quadratic_negative_roots_pq_value_l222_222465

theorem quadratic_negative_roots_pq_value (r : ℝ) :
  (∃ p q : ℝ, p = -87 ∧ q = -23 ∧ x^2 - (r + 7)*x + r + 87 = 0 ∧ p < r ∧ r < q)
  → ((-87)^2 + (-23)^2 = 8098) :=
by
  sorry

end quadratic_negative_roots_pq_value_l222_222465


namespace number_of_different_duty_schedules_l222_222827

-- Define a structure for students
inductive Student
| A | B | C

-- Define days of the week excluding Sunday as all duties are from Monday to Saturday
inductive Day
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

-- Define the conditions in Lean
def condition_A_does_not_take_Monday (schedules : Day → Student) : Prop :=
  schedules Day.Monday ≠ Student.A

def condition_B_does_not_take_Saturday (schedules : Day → Student) : Prop :=
  schedules Day.Saturday ≠ Student.B

-- Define the function to count valid schedules
noncomputable def count_valid_schedules : ℕ :=
  sorry  -- This would be the computation considering combinatorics

-- Theorem statement to prove the correct answer
theorem number_of_different_duty_schedules 
    (schedules : Day → Student)
    (h1 : condition_A_does_not_take_Monday schedules)
    (h2 : condition_B_does_not_take_Saturday schedules)
    : count_valid_schedules = 42 :=
sorry

end number_of_different_duty_schedules_l222_222827


namespace eustace_age_in_3_years_l222_222339

variable (E M : ℕ)

theorem eustace_age_in_3_years
  (h1 : E = 2 * M)
  (h2 : M + 3 = 21) :
  E + 3 = 39 :=
sorry

end eustace_age_in_3_years_l222_222339


namespace find_integer_pairs_l222_222992

theorem find_integer_pairs :
  ∃ (S : Finset (ℤ × ℤ)), (∀ (m n : ℤ), (m, n) ∈ S ↔ mn ≤ 0 ∧ m^3 + n^3 - 37 * m * n = 343) ∧ S.card = 9 :=
sorry

end find_integer_pairs_l222_222992


namespace sum_composite_l222_222188

theorem sum_composite (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : 34 * a = 43 * b) : ∃ d : ℕ, d > 1 ∧ d < a + b ∧ d ∣ (a + b) :=
by
  sorry

end sum_composite_l222_222188


namespace tangent_line_to_parabola_l222_222470

theorem tangent_line_to_parabola (r : ℝ) :
  (∃ x : ℝ, 2 * x^2 - x - r = 0) ∧
  (∀ x1 x2 : ℝ, (2 * x1^2 - x1 - r = 0) ∧ (2 * x2^2 - x2 - r = 0) → x1 = x2) →
  r = -1 / 8 :=
sorry

end tangent_line_to_parabola_l222_222470


namespace solve_m_problem_l222_222490

theorem solve_m_problem :
  (∃ x : ℝ, -1 < x ∧ x < 1 ∧ x^2 - x - m = 0) →
  m ∈ Set.Ico (-1/4 : ℝ) 2 :=
sorry

end solve_m_problem_l222_222490


namespace smallest_portion_quantity_l222_222401

-- Define the conditions for the problem
def conditions (a1 a2 a3 a4 a5 d : ℚ) : Prop :=
  a2 = a1 + d ∧
  a3 = a1 + 2 * d ∧
  a4 = a1 + 3 * d ∧
  a5 = a1 + 4 * d ∧
  5 * a1 + 10 * d = 100 ∧
  (a3 + a4 + a5) = (1/7) * (a1 + a2)

-- Lean theorem statement
theorem smallest_portion_quantity : 
  ∃ (a1 a2 a3 a4 a5 d : ℚ), conditions a1 a2 a3 a4 a5 d ∧ a1 = 5 / 3 :=
by
  sorry

end smallest_portion_quantity_l222_222401


namespace B_alone_finishes_in_19_point_5_days_l222_222133

-- Define the conditions
def is_half_good(A B : ℝ) : Prop := A = 1 / 2 * B
def together_finish_in_13_days(A B : ℝ) : Prop := (A + B) * 13 = 1

-- Define the statement
theorem B_alone_finishes_in_19_point_5_days (A B : ℝ) (h1 : is_half_good A B) (h2 : together_finish_in_13_days A B) :
  B * 19.5 = 1 :=
by
  sorry

end B_alone_finishes_in_19_point_5_days_l222_222133


namespace total_highlighters_l222_222904

def num_pink_highlighters := 9
def num_yellow_highlighters := 8
def num_blue_highlighters := 5

theorem total_highlighters : 
  num_pink_highlighters + num_yellow_highlighters + num_blue_highlighters = 22 :=
by
  sorry

end total_highlighters_l222_222904


namespace find_principal_amount_l222_222164

-- Define the given conditions
def interest_rate1 : ℝ := 0.08
def interest_rate2 : ℝ := 0.10
def interest_rate3 : ℝ := 0.12
def period1 : ℝ := 4
def period2 : ℝ := 6
def period3 : ℝ := 5
def total_interest_paid : ℝ := 12160

-- Goal is to find the principal amount P
theorem find_principal_amount (P : ℝ) :
  total_interest_paid = P * (interest_rate1 * period1 + interest_rate2 * period2 + interest_rate3 * period3) →
  P = 8000 :=
by
  sorry

end find_principal_amount_l222_222164


namespace simplify_expression_l222_222793

theorem simplify_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h1 : a^4 + b^4 = a + b) (h2 : a^2 + b^2 = 2) :
  (a^2 / b^2 + b^2 / a^2 - 1 / (a^2 * b^2)) = 1 := 
sorry

end simplify_expression_l222_222793


namespace journey_speed_first_half_l222_222574

noncomputable def speed_first_half (total_time : ℝ) (total_distance : ℝ) (second_half_speed : ℝ) : ℝ :=
  let first_half_distance := total_distance / 2
  let second_half_distance := total_distance / 2
  let second_half_time := second_half_distance / second_half_speed
  let first_half_time := total_time - second_half_time
  first_half_distance / first_half_time

theorem journey_speed_first_half
  (total_time : ℝ) (total_distance : ℝ) (second_half_speed : ℝ)
  (h1 : total_time = 10)
  (h2 : total_distance = 224)
  (h3 : second_half_speed = 24) :
  speed_first_half total_time total_distance second_half_speed = 21 := by
  sorry

end journey_speed_first_half_l222_222574


namespace time_to_eat_quarter_l222_222096

noncomputable def total_nuts : ℕ := sorry

def rate_first_crow (N : ℕ) := N / 40
def rate_second_crow (N : ℕ) := N / 36

theorem time_to_eat_quarter (N : ℕ) (T : ℝ) :
  (rate_first_crow N + rate_second_crow N) * T = (1 / 4 : ℝ) * N → 
  T = (90 / 19 : ℝ) :=
by
  intros h
  sorry

end time_to_eat_quarter_l222_222096


namespace probability_two_boys_l222_222871

-- Definitions for the conditions
def total_students : ℕ := 4
def boys : ℕ := 3
def girls : ℕ := 1
def select_students : ℕ := 2

-- Combination function definition
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_two_boys :
  (combination boys select_students) / (combination total_students select_students) = 1 / 2 := by
  sorry

end probability_two_boys_l222_222871


namespace categorize_numbers_l222_222604

def numbers : Set (Rat) := {-16, 0.04, 1/2, -2/3, 25, 0, -3.6, -0.3, 4/3}

def is_integer (x : Rat) : Prop := ∃ z : Int, x = z
def is_fraction (x : Rat) : Prop := ∃ (p q : Int), q ≠ 0 ∧ x = p / q
def is_negative (x : Rat) : Prop := x < 0

def integers (s : Set Rat) : Set Rat := {x | x ∈ s ∧ is_integer x}
def fractions (s : Set Rat) : Set Rat := {x | x ∈ s ∧ is_fraction x}
def negative_rationals (s : Set Rat) : Set Rat := {x | x ∈ s ∧ is_fraction x ∧ is_negative x}

theorem categorize_numbers :
  integers numbers = {-16, 25, 0} ∧
  fractions numbers = {0.04, 1/2, -2/33, -3.6, -0.3, 4/3} ∧
  negative_rationals numbers = {-16, -2/3, -3.6, -0.3} :=
  sorry

end categorize_numbers_l222_222604


namespace p_scale_measurement_l222_222292

theorem p_scale_measurement (a b P S : ℝ) (h1 : 30 = 6 * a + b) (h2 : 60 = 24 * a + b) (h3 : 100 = a * P + b) : P = 48 :=
by
  sorry

end p_scale_measurement_l222_222292


namespace digit_in_452nd_place_l222_222698

def repeating_sequence : List Nat := [3, 6, 8, 4, 2, 1, 0, 5, 2, 6, 3, 1, 5, 7, 8, 9, 4, 7]
def repeat_length : Nat := 18

theorem digit_in_452nd_place :
  (repeating_sequence.get ⟨(452 % repeat_length) - 1, sorry⟩ = 6) :=
sorry

end digit_in_452nd_place_l222_222698


namespace find_fourth_number_l222_222072

theorem find_fourth_number (a : ℕ → ℕ) 
  (h1 : ∀ n, n ≥ 2 → a n = a (n - 1) + a (n - 2)) 
  (h2 : a 6 = 42) 
  (h3 : a 8 = 110) : 
  a 3 = 10 := 
sorry

end find_fourth_number_l222_222072


namespace integer_solution_l222_222281

theorem integer_solution (n : ℤ) (h1 : n + 15 > 16) (h2 : -3 * n > -9) : n = 2 :=
by
  sorry

end integer_solution_l222_222281


namespace pencil_cost_l222_222048

theorem pencil_cost (total_money : ℕ) (num_pencils : ℕ) (h1 : total_money = 50) (h2 : num_pencils = 10) :
    (total_money / num_pencils) = 5 :=
by
  sorry

end pencil_cost_l222_222048


namespace smallest_even_integer_cube_mod_1000_l222_222478

theorem smallest_even_integer_cube_mod_1000 :
  ∃ n : ℕ, (n % 2 = 0) ∧ (n > 0) ∧ (n^3 % 1000 = 392) ∧ (∀ m : ℕ, (m % 2 = 0) ∧ (m > 0) ∧ (m^3 % 1000 = 392) → n ≤ m) ∧ n = 892 := 
sorry

end smallest_even_integer_cube_mod_1000_l222_222478


namespace dave_apps_left_l222_222331

theorem dave_apps_left (A : ℕ) 
  (h1 : 24 = A + 22) : A = 2 :=
by
  sorry

end dave_apps_left_l222_222331


namespace gcd_values_count_l222_222130

theorem gcd_values_count (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 392) : ∃ d, d = 11 := 
sorry

end gcd_values_count_l222_222130


namespace graph_empty_l222_222736

theorem graph_empty {x y : ℝ} : 
  x^2 + 3 * y^2 - 4 * x - 6 * y + 10 = 0 → false := 
by 
  sorry

end graph_empty_l222_222736


namespace find_a_value_l222_222774

-- Define the problem conditions
def line_eq_condition (a : ℝ) := ∃ (k : ℝ), k = 1 ∧ k = -a / (2 * a - 3)

-- Define the proof goal
theorem find_a_value (a : ℝ) (h : line_eq_condition a) : a = 1 :=
by sorry

end find_a_value_l222_222774


namespace nested_expression_value_l222_222464

theorem nested_expression_value : 
  4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4)))))))) = 87380 :=
by 
  sorry

end nested_expression_value_l222_222464


namespace number_of_fridays_l222_222231

theorem number_of_fridays (jan_1_sat : true) (is_non_leap_year : true) : ∃ (n : ℕ), n = 52 :=
by
  -- Conditions: January 1st is Saturday and it is a non-leap year.
  -- We are given that January 1st is a Saturday.
  have jan_1_sat_condition : true := jan_1_sat
  -- We are given that the year is a non-leap year (365 days).
  have non_leap_condition : true := is_non_leap_year
  -- Therefore, there are 52 Fridays in the year.
  use 52
  done

end number_of_fridays_l222_222231


namespace Markus_bags_count_l222_222243

-- Definitions of the conditions
def Mara_bags : ℕ := 12
def Mara_marbles_per_bag : ℕ := 2
def Markus_marbles_per_bag : ℕ := 13
def marbles_difference : ℕ := 2

-- Derived conditions
def Mara_total_marbles : ℕ := Mara_bags * Mara_marbles_per_bag
def Markus_total_marbles : ℕ := Mara_total_marbles + marbles_difference

-- Statement to prove
theorem Markus_bags_count : Markus_total_marbles / Markus_marbles_per_bag = 2 :=
by
  -- Skip the proof, leaving it as a task for the prover
  sorry

end Markus_bags_count_l222_222243


namespace find_z_l222_222475

theorem find_z (z : ℝ) (h : (z^2 - 5 * z + 6) / (z - 2) + (5 * z^2 + 11 * z - 32) / (5 * z - 16) = 1) : z = 1 :=
sorry

end find_z_l222_222475


namespace cyclic_B_E_C_K_l222_222390

variable (A B C D E F G H K M : Point) (ω : Circle A B C)

-- Conditions
variables 
  (hD_on_BC : D ∈ LineSegment B C)
  (hE_on_AD : E ∈ LineSegment A D)
  (hF_on_ADω : F ∈ RayAD ∧ F ∈ ω ∧ F ≠ A)
  (hM_bisects_AF : M ∈ ω ∧ M ∈ Midpoint A F ∧ M ∉ Line A C F)
  (hG_on_MEω : G ∈ RayME ∧ G ∈ ω ∧ G ≠ M)
  (hH_on_GDω : H ∈ RayGD ∧ H ∈ ω ∧ H ≠ G)
  (hK_on_MHAD : K ∈ LineMH ∧ K ∈ LineAD)

-- Goal: B, E, C, K are cyclic.
theorem cyclic_B_E_C_K :
  CyclicQueue B E C K := 
  sorry

end cyclic_B_E_C_K_l222_222390


namespace arithmetic_sequence_ratio_l222_222658

noncomputable def A_n (a d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

noncomputable def B_n (b e : ℤ) (n : ℕ) : ℤ :=
  n * (2 * b + (n - 1) * e) / 2

theorem arithmetic_sequence_ratio (a d b e : ℤ) :
  (∀ n : ℕ, n ≠ 0 → A_n a d n / B_n b e n = (5 * n - 3) / (n + 9)) →
  (a + 5 * d) / (b + 2 * e) = 26 / 7 :=
by
  sorry

end arithmetic_sequence_ratio_l222_222658


namespace arithmetic_prog_sum_l222_222906

theorem arithmetic_prog_sum (a d : ℕ) (h1 : 15 * a + 105 * d = 60) : 2 * a + 14 * d = 8 :=
by
  sorry

end arithmetic_prog_sum_l222_222906


namespace cornbread_pieces_l222_222516

theorem cornbread_pieces (pan_length pan_width piece_length piece_width : ℕ)
  (h₁ : pan_length = 24) (h₂ : pan_width = 20) 
  (h₃ : piece_length = 3) (h₄ : piece_width = 2) :
  (pan_length * pan_width) / (piece_length * piece_width) = 80 := by
  sorry

end cornbread_pieces_l222_222516


namespace ellipse_equation_l222_222753

theorem ellipse_equation (b : Real) (c : Real)
  (h₁ : 0 < b ∧ b < 5) 
  (h₂ : 25 - b^2 = c^2)
  (h₃ : 5 + c = 2 * b) :
  ∃ (b : Real), (b^2 = 16) ∧ (∀ x y : Real, (x^2 / 25 + y^2 / b^2 = 1 ↔ x^2 / 25 + y^2 / 16 = 1)) := 
sorry

end ellipse_equation_l222_222753


namespace line_through_center_of_circle_l222_222204

theorem line_through_center_of_circle 
    (x y : ℝ) 
    (h : x^2 + y^2 - 4*x + 6*y = 0) : 
    3*x + 2*y = 0 :=
sorry

end line_through_center_of_circle_l222_222204


namespace paint_per_color_equal_l222_222515

theorem paint_per_color_equal (total_paint : ℕ) (num_colors : ℕ) (paint_per_color : ℕ) : 
  total_paint = 15 ∧ num_colors = 3 → paint_per_color = 5 := by
  sorry

end paint_per_color_equal_l222_222515


namespace rem_frac_eq_l222_222166

theorem rem_frac_eq :
  (x y : ℚ) (hx : x = -5/6) (hy : y = 3/4) :
  rat.floor (x / y) = -2 → 
  x - y * (rat.floor (x / y)) = 2/3 :=
by
  intros
  sorry

end rem_frac_eq_l222_222166


namespace divisor_proof_l222_222109

def original_number : ℕ := 123456789101112131415161718192021222324252627282930313233343536373839404142434481

def remainder : ℕ := 36

theorem divisor_proof (D : ℕ) (Q : ℕ) (h : original_number = D * Q + remainder) : original_number % D = remainder :=
by 
  sorry

end divisor_proof_l222_222109


namespace union_of_sets_l222_222632

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

theorem union_of_sets : A ∪ B = {-1, 0, 1, 2} := 
by
  sorry

end union_of_sets_l222_222632


namespace squares_in_50th_ring_l222_222981

noncomputable def number_of_squares_in_nth_ring (n : ℕ) : ℕ :=
  8 * n + 6

theorem squares_in_50th_ring : number_of_squares_in_nth_ring 50 = 406 := 
  by
  sorry

end squares_in_50th_ring_l222_222981


namespace An_is_integer_l222_222889

theorem An_is_integer 
  (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_gt : a > b)
  (θ : ℝ) (h_theta : θ > 0 ∧ θ < Real.pi / 2)
  (h_sin : Real.sin θ = 2 * (a * b) / (a^2 + b^2)) :
  ∀ n : ℕ, ∃ k : ℤ, ((a^2 + b^2)^n * Real.sin (n * θ) : ℝ) = k :=
by sorry

end An_is_integer_l222_222889


namespace wesley_breenah_ages_l222_222696

theorem wesley_breenah_ages (w b : ℕ) (h₁ : w = 15) (h₂ : b = 7) (h₃ : w + b = 22) :
  ∃ n : ℕ, 2 * (w + b) = (w + n) + (b + n) := by
  exists 11
  sorry

end wesley_breenah_ages_l222_222696


namespace fraction_difference_l222_222108

def A : ℕ := 3 + 6 + 9
def B : ℕ := 2 + 5 + 8

theorem fraction_difference : (A / B) - (B / A) = 11 / 30 := by
  sorry

end fraction_difference_l222_222108


namespace jellybean_total_l222_222026

theorem jellybean_total (large_jellybeans_per_glass : ℕ) 
  (small_jellybeans_per_glass : ℕ) 
  (num_large_glasses : ℕ) 
  (num_small_glasses : ℕ) 
  (h1 : large_jellybeans_per_glass = 50) 
  (h2 : small_jellybeans_per_glass = large_jellybeans_per_glass / 2) 
  (h3 : num_large_glasses = 5) 
  (h4 : num_small_glasses = 3) : 
  (num_large_glasses * large_jellybeans_per_glass + num_small_glasses * small_jellybeans_per_glass) = 325 :=
by
  sorry

end jellybean_total_l222_222026


namespace proportion_of_white_pieces_l222_222092

theorem proportion_of_white_pieces (x : ℕ) (h1 : 0 < x) :
  let total_pieces := 3 * x
  let white_pieces := x + (1 - (5 / 9)) * x
  (white_pieces / total_pieces) = (13 / 27) :=
by
  sorry

end proportion_of_white_pieces_l222_222092


namespace quad_area_FDBG_l222_222277

open Real

noncomputable def area_quad_FDBG (AB AC area_ABC : ℝ) : ℝ :=
  let AD := AB / 2
  let AE := AC / 2
  let area_ADE := area_ABC / 4
  let x := 2 * area_ABC / (AB * AC)
  let sin_A := x
  let hyp_ratio := sin_A / (area_ABC / AC)
  let factor := hyp_ratio / 2
  let area_AFG := factor * area_ADE
  area_ABC - area_ADE - 2 * area_AFG

theorem quad_area_FDBG (AB AC area_ABC : ℝ) (hAB : AB = 60) (hAC : AC = 15) (harea : area_ABC = 180) :
  area_quad_FDBG AB AC area_ABC = 117 := by
  sorry

end quad_area_FDBG_l222_222277


namespace smallest_positive_integer_ends_in_3_divisible_by_11_l222_222122

theorem smallest_positive_integer_ends_in_3_divisible_by_11 :
  ∃ n : ℕ, n > 0 ∧ n % 10 = 3 ∧ n % 11 = 0 ∧ ∀ m : ℕ, (m > 0 ∧ m % 10 = 3 ∧ m % 11 = 0) → n ≤ m :=
sorry

end smallest_positive_integer_ends_in_3_divisible_by_11_l222_222122


namespace probability_of_yellow_light_l222_222586

def time_red : ℕ := 30
def time_green : ℕ := 25
def time_yellow : ℕ := 5
def total_cycle_time : ℕ := time_red + time_green + time_yellow

theorem probability_of_yellow_light :
  (time_yellow : ℚ) / (total_cycle_time : ℚ) = 1 / 12 :=
by
  sorry

end probability_of_yellow_light_l222_222586


namespace triangle_area_l222_222945

theorem triangle_area (c b : ℝ) (c_eq : c = 15) (b_eq : b = 9) :
  ∃ a : ℝ, a^2 = c^2 - b^2 ∧ (b * a) / 2 = 54 := by
  sorry

end triangle_area_l222_222945


namespace initial_carrots_l222_222824

theorem initial_carrots (n : ℕ) 
    (h1: 3640 = 180 * (n - 4) + 760) 
    (h2: 180 * (n - 4) < 3640) 
    (h3: 4 * 190 = 760) : 
    n = 20 :=
by
  sorry

end initial_carrots_l222_222824


namespace giyoon_chocolates_l222_222493

theorem giyoon_chocolates (C X : ℕ) (h1 : C = 8 * X) (h2 : C = 6 * (X + 1) + 4) : C = 40 :=
by sorry

end giyoon_chocolates_l222_222493


namespace arithmetic_sequence_iff_condition_l222_222547

-- Definitions: A sequence and the condition
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_iff_condition (a : ℕ → ℝ) :
  is_arithmetic_sequence a ↔ (∀ n : ℕ, 2 * a (n + 1) = a n + a (n + 2)) :=
by
  -- Proof is omitted.
  sorry

end arithmetic_sequence_iff_condition_l222_222547


namespace smallest_non_factor_product_l222_222099

theorem smallest_non_factor_product (a b : ℕ) (h1 : a ≠ b) (h2 : a ∣ 48) (h3 : b ∣ 48) (h4 : ¬ (a * b ∣ 48)) : a * b = 18 :=
by
  -- proof intentionally omitted
  sorry

end smallest_non_factor_product_l222_222099


namespace group_elements_eq_one_l222_222023
-- Import the entire math library

-- Define the main theorem
theorem group_elements_eq_one 
  {G : Type*} [Group G] 
  (a b : G) 
  (h1 : a * b^2 = b^3 * a) 
  (h2 : b * a^2 = a^3 * b) : 
  a = 1 ∧ b = 1 := 
  by 
  sorry

end group_elements_eq_one_l222_222023


namespace find_x_l222_222922

-- Define the initial point A with coordinates A(x, -2)
def A (x : ℝ) : ℝ × ℝ := (x, -2)

-- Define the transformation of moving 5 units up and 3 units to the right to obtain point B
def transform (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + 3, p.2 + 5)

-- Define the final point B with coordinates B(1, y)
def B (y : ℝ) : ℝ × ℝ := (1, y)

-- Define the proof problem
theorem find_x (x y : ℝ) (h : transform (A x) = B y) : x = -2 :=
by sorry

end find_x_l222_222922


namespace probability_of_all_heads_or_tails_l222_222995

def num_favorable_outcomes : ℕ := 2

def total_outcomes : ℕ := 2 ^ 5

def probability_all_heads_or_tails : ℚ := num_favorable_outcomes / total_outcomes

theorem probability_of_all_heads_or_tails :
  probability_all_heads_or_tails = 1 / 16 := by
  -- Proof goes here
  sorry

end probability_of_all_heads_or_tails_l222_222995


namespace tangent_line_curve_l222_222932

theorem tangent_line_curve (a b : ℚ) 
  (h1 : 3 * a + b = 1) 
  (h2 : a + b = 2) : 
  b - a = 3 := 
by 
  sorry

end tangent_line_curve_l222_222932


namespace sequence_general_formula_l222_222004

theorem sequence_general_formula (a : ℕ → ℕ)
    (h1 : a 1 = 3) 
    (h2 : a 2 = 4) 
    (h3 : a 3 = 6) 
    (h4 : a 4 = 10) 
    (h5 : a 5 = 18) :
    ∀ n : ℕ, a n = 2^(n-1) + 2 :=
sorry

end sequence_general_formula_l222_222004


namespace find_f_2000_l222_222185

variable (f : ℕ → ℕ)
variable (x : ℕ)

axiom initial_condition : f 0 = 1
axiom recurrence_relation : ∀ x, f (x + 2) = f x + 4 * x + 2

theorem find_f_2000 : f 2000 = 3998001 :=
by
  sorry

end find_f_2000_l222_222185


namespace systematic_sample_contains_18_l222_222454

theorem systematic_sample_contains_18 (employees : Finset ℕ) (sample : Finset ℕ)
    (h1 : employees = Finset.range 52)
    (h2 : sample.card = 4)
    (h3 : ∀ n ∈ sample, n ∈ employees)
    (h4 : 5 ∈ sample)
    (h5 : 31 ∈ sample)
    (h6 : 44 ∈ sample) :
  18 ∈ sample :=
sorry

end systematic_sample_contains_18_l222_222454


namespace volume_of_cube_l222_222917

theorem volume_of_cube (a : ℕ) (h : (a^3 - a = a^3 - 5)) : a^3 = 125 :=
by {
  -- The necessary algebraic manipulation follows
  sorry
}

end volume_of_cube_l222_222917


namespace deepak_present_age_l222_222940

variable (R D : ℕ)

theorem deepak_present_age 
  (h1 : R + 22 = 26) 
  (h2 : R / D = 4 / 3) : 
  D = 3 := 
sorry

end deepak_present_age_l222_222940


namespace total_amount_paid_correct_l222_222308

/--
Given:
1. The marked price of each article is $17.5.
2. A discount of 30% was applied to the total marked price of the pair of articles.

Prove:
The total amount paid for the pair of articles is $24.5.
-/
def total_amount_paid (marked_price_each : ℝ) (discount_rate : ℝ) : ℝ :=
  let marked_price_pair := marked_price_each * 2
  let discount := discount_rate * marked_price_pair
  marked_price_pair - discount

theorem total_amount_paid_correct :
  total_amount_paid 17.5 0.30 = 24.5 :=
by
  sorry

end total_amount_paid_correct_l222_222308


namespace sum_in_Q_l222_222745

open Set

def is_set_P (x : ℤ) : Prop := ∃ k : ℤ, x = 2 * k
def is_set_Q (x : ℤ) : Prop := ∃ k : ℤ, x = 2 * k - 1
def is_set_M (x : ℤ) : Prop := ∃ k : ℤ, x = 4 * k + 1

variables (a b : ℤ)

theorem sum_in_Q (ha : is_set_P a) (hb : is_set_Q b) : is_set_Q (a + b) := 
sorry

end sum_in_Q_l222_222745


namespace second_offset_length_l222_222344

theorem second_offset_length (d h1 area : ℝ) (h_diagonal : d = 28) (h_offset1 : h1 = 8) (h_area : area = 140) :
  ∃ x : ℝ, area = (1/2) * d * (h1 + x) ∧ x = 2 :=
by
  sorry

end second_offset_length_l222_222344


namespace fruit_basket_l222_222550

theorem fruit_basket :
  ∀ (oranges apples bananas peaches : ℕ),
  oranges = 6 →
  apples = oranges - 2 →
  bananas = 3 * apples →
  peaches = bananas / 2 →
  oranges + apples + bananas + peaches = 28 :=
by
  intros oranges apples bananas peaches h_oranges h_apples h_bananas h_peaches
  rw [h_oranges, h_apples, h_bananas, h_peaches]
  sorry

end fruit_basket_l222_222550


namespace train_crossing_time_l222_222900

noncomputable def length_of_train : ℕ := 250
noncomputable def length_of_bridge : ℕ := 350
noncomputable def speed_of_train_kmph : ℕ := 72

noncomputable def speed_of_train_mps : ℕ := (speed_of_train_kmph * 1000) / 3600

noncomputable def total_distance : ℕ := length_of_train + length_of_bridge

theorem train_crossing_time : total_distance / speed_of_train_mps = 30 := by
  sorry

end train_crossing_time_l222_222900


namespace original_price_of_house_l222_222253

theorem original_price_of_house (P : ℝ) 
  (h1 : P * 0.56 = 56000) : P = 100000 :=
sorry

end original_price_of_house_l222_222253


namespace fixed_point_is_5_225_l222_222612

theorem fixed_point_is_5_225 : ∃ a b : ℝ, (∀ k : ℝ, 9 * a^2 + k * a - 5 * k = b) → (a = 5 ∧ b = 225) :=
by
  sorry

end fixed_point_is_5_225_l222_222612


namespace option_A_correct_l222_222954

theorem option_A_correct (x y : ℝ) (hy : y ≠ 0) :
  (-2 * x^2 * y + y) / y = -2 * x^2 + 1 :=
by
  sorry

end option_A_correct_l222_222954


namespace Jill_has_5_peaches_l222_222260

-- Define the variables and their relationships
variables (S Jl Jk : ℕ)

-- Declare the conditions as assumptions
axiom Steven_has_14_peaches : S = 14
axiom Jake_has_6_fewer_peaches_than_Steven : Jk = S - 6
axiom Jake_has_3_more_peaches_than_Jill : Jk = Jl + 3

-- Define the theorem to prove Jill has 5 peaches
theorem Jill_has_5_peaches (S Jk Jl : ℕ) 
  (h1 : S = 14) 
  (h2 : Jk = S - 6)
  (h3 : Jk = Jl + 3) : 
  Jl = 5 := 
by
  sorry

end Jill_has_5_peaches_l222_222260


namespace intersection_of_lines_l222_222883

theorem intersection_of_lines : ∃ (x y : ℝ), 9 * x - 4 * y = 6 ∧ 7 * x + y = 17 ∧ (x, y) = (2, 3) := 
by
  sorry

end intersection_of_lines_l222_222883


namespace absolute_value_condition_l222_222011

theorem absolute_value_condition (a : ℝ) (h : |a| = -a) : a = 0 ∨ a < 0 :=
by
  sorry

end absolute_value_condition_l222_222011


namespace regression_line_intercept_l222_222615

theorem regression_line_intercept
  (x : ℕ → ℝ)
  (y : ℕ → ℝ)
  (h_x_sum : x 1 + x 2 + x 3 + x 4 + x 5 + x 6 = 10)
  (h_y_sum : y 1 + y 2 + y 3 + y 4 + y 5 + y 6 = 4) :
  ∃ a : ℝ, (∀ i, y i = (1 / 4) * x i + a) → a = 1 / 4 :=
by
  sorry

end regression_line_intercept_l222_222615


namespace simplify_and_evaluate_l222_222804

theorem simplify_and_evaluate
  (m : ℝ) (hm : m = 2 + Real.sqrt 2) :
  (1 - (m / (m + 2))) / ((m^2 - 4*m + 4) / (m^2 - 4)) = Real.sqrt 2 :=
by
  sorry

end simplify_and_evaluate_l222_222804


namespace partial_fraction_sum_zero_l222_222324

theorem partial_fraction_sum_zero (A B C D E F : ℚ) :
  (∀ x : ℚ, x ≠ 0 → x ≠ -1 → x ≠ -2 → x ≠ -3 → x ≠ -4 → x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 :=
sorry

end partial_fraction_sum_zero_l222_222324


namespace fraction_sum_squares_eq_sixteen_l222_222369

variables (x a y b z c : ℝ)

theorem fraction_sum_squares_eq_sixteen
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 0) :
  (x^2 / a^2 + y^2 / b^2 + z^2 / c^2) = 16 := 
sorry

end fraction_sum_squares_eq_sixteen_l222_222369


namespace arithmetic_mean_of_fractions_l222_222796

theorem arithmetic_mean_of_fractions :
  let a := 7 / 9
  let b := 5 / 6
  let c := 8 / 9
  2 * b = a + c :=
by
  sorry

end arithmetic_mean_of_fractions_l222_222796


namespace percentage_orange_juice_l222_222306

-- Definitions based on conditions
def total_volume : ℝ := 120
def watermelon_percentage : ℝ := 0.60
def grape_juice_volume : ℝ := 30
def watermelon_juice_volume : ℝ := watermelon_percentage * total_volume
def combined_watermelon_grape_volume : ℝ := watermelon_juice_volume + grape_juice_volume
def orange_juice_volume : ℝ := total_volume - combined_watermelon_grape_volume

-- Lean 4 statement to prove the percentage of orange juice
theorem percentage_orange_juice : (orange_juice_volume / total_volume) * 100 = 15 := by
  -- sorry to skip the proof
  sorry

end percentage_orange_juice_l222_222306


namespace arithmetic_prog_leq_l222_222357

def t3 (s : List ℤ) : ℕ := 
  sorry -- Placeholder for function calculating number of 3-term arithmetic progressions

theorem arithmetic_prog_leq (a : List ℤ) (k : ℕ) (h_sorted : a = List.range k)
  : t3 a ≤ t3 (List.range k) :=
sorry -- Proof here

end arithmetic_prog_leq_l222_222357


namespace sandwiches_difference_l222_222528

-- Conditions definitions
def sandwiches_at_lunch_monday : ℤ := 3
def sandwiches_at_dinner_monday : ℤ := 2 * sandwiches_at_lunch_monday
def total_sandwiches_monday : ℤ := sandwiches_at_lunch_monday + sandwiches_at_dinner_monday
def sandwiches_on_tuesday : ℤ := 1

-- Proof goal
theorem sandwiches_difference :
  total_sandwiches_monday - sandwiches_on_tuesday = 8 :=
  by
  sorry

end sandwiches_difference_l222_222528


namespace smallest_integer_ends_in_3_divisible_by_11_correct_l222_222114

def ends_in_3 (n : ℕ) : Prop :=
  n % 10 = 3

def divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def smallest_ends_in_3_divisible_by_11 : ℕ :=
  33

theorem smallest_integer_ends_in_3_divisible_by_11_correct :
  smallest_ends_in_3_divisible_by_11 = 33 ∧ ends_in_3 smallest_ends_in_3_divisible_by_11 ∧ divisible_by_11 smallest_ends_in_3_divisible_by_11 := 
by
  sorry

end smallest_integer_ends_in_3_divisible_by_11_correct_l222_222114


namespace correct_operation_l222_222131

theorem correct_operation (a : ℝ) : a^5 / a^2 = a^3 := by
  -- Proof steps will be supplied here
  sorry

end correct_operation_l222_222131


namespace first_train_speed_l222_222106

-- Definitions
def train_speeds_opposite (v₁ v₂ t : ℝ) : Prop := v₁ * t + v₂ * t = 910

def train_problem_conditions (v₁ v₂ t : ℝ) : Prop :=
  train_speeds_opposite v₁ v₂ t ∧ v₂ = 80 ∧ t = 6.5

-- Theorem
theorem first_train_speed (v : ℝ) (h : train_problem_conditions v 80 6.5) : v = 60 :=
  sorry

end first_train_speed_l222_222106


namespace raccoon_hid_nuts_l222_222225

theorem raccoon_hid_nuts :
  ∃ (r p : ℕ), r + p = 25 ∧ (p = r - 3) ∧ 5 * r = 6 * p ∧ 5 * r = 70 :=
by
  sorry

end raccoon_hid_nuts_l222_222225


namespace sin_cos_sum_l222_222237

-- Let theta be an angle in the second quadrant
variables (θ : ℝ)
-- Given the condition tan(θ + π / 4) = 1 / 2
variable (h1 : Real.tan (θ + Real.pi / 4) = 1 / 2)
-- Given θ is in the second quadrant
variable (h2 : θ ∈ Set.Ioc (Real.pi / 2) Real.pi)

-- Prove sin θ + cos θ = - sqrt(10) / 5
theorem sin_cos_sum (h1 : Real.tan (θ + Real.pi / 4) = 1 / 2) (h2 : θ ∈ Set.Ioc (Real.pi / 2) Real.pi) :
  Real.sin θ + Real.cos θ = -Real.sqrt 10 / 5 :=
sorry

end sin_cos_sum_l222_222237


namespace max_value_expression_l222_222949

theorem max_value_expression (p : ℝ) (q : ℝ) (h : q = p - 2) :
  ∃ M : ℝ, M = -70 + 96.66666666666667 ∧ (∀ p : ℝ, -3 * p^2 + 24 * p - 50 + 10 * q ≤ M) :=
sorry

end max_value_expression_l222_222949


namespace chord_length_in_circle_l222_222224

theorem chord_length_in_circle 
  (radius : ℝ) 
  (chord_midpoint_perpendicular_radius : ℝ)
  (r_eq_10 : radius = 10)
  (cmp_eq_5 : chord_midpoint_perpendicular_radius = 5) : 
  ∃ (chord_length : ℝ), chord_length = 10 * Real.sqrt 3 := 
by 
  sorry

end chord_length_in_circle_l222_222224


namespace Martha_blocks_end_l222_222395

variable (Ronald_blocks : ℕ) (Martha_start_blocks : ℕ) (Martha_found_blocks : ℕ)
variable (Ronald_has_blocks : Ronald_blocks = 13)
variable (Martha_has_start_blocks : Martha_start_blocks = 4)
variable (Martha_finds_more_blocks : Martha_found_blocks = 80)

theorem Martha_blocks_end : Martha_start_blocks + Martha_found_blocks = 84 :=
by
  have Martha_start_blocks := Martha_has_start_blocks
  have Martha_found_blocks := Martha_finds_more_blocks
  sorry

end Martha_blocks_end_l222_222395


namespace equipment_total_cost_l222_222549

def cost_jersey : ℝ := 25
def cost_shorts : ℝ := 15.20
def cost_socks : ℝ := 6.80
def cost_cleats : ℝ := 40
def cost_water_bottle : ℝ := 12
def cost_one_player := cost_jersey + cost_shorts + cost_socks + cost_cleats + cost_water_bottle
def num_players : ℕ := 25
def total_cost_for_team : ℝ := cost_one_player * num_players

theorem equipment_total_cost :
  total_cost_for_team = 2475 := by
  sorry

end equipment_total_cost_l222_222549


namespace min_value_of_expression_l222_222622

noncomputable def minValue (a : ℝ) : ℝ :=
  1 / (3 - 2 * a) + 2 / (a - 1)

theorem min_value_of_expression : ∀ a : ℝ, 1 < a ∧ a < 3 / 2 → (1 / (3 - 2 * a) + 2 / (a - 1)) ≥ 16 / 9 :=
by
  intro a h
  sorry

end min_value_of_expression_l222_222622


namespace machine_a_produces_50_parts_in_10_minutes_l222_222138

/-- 
Given that machine A produces parts twice as fast as machine B,
and machine B produces 100 parts in 40 minutes at a constant rate,
prove that machine A produces 50 parts in 10 minutes.
-/
theorem machine_a_produces_50_parts_in_10_minutes :
  (machine_b_rate : ℕ → ℕ) → 
  (machine_a_rate : ℕ → ℕ) →
  (htwice_as_fast: ∀ t, machine_a_rate t = (2 * machine_b_rate t)) →
  (hconstant_rate_b: ∀ t1 t2, t1 * machine_b_rate t2 = 100 * t2 / 40)→
  machine_a_rate 10 = 50 :=
by
  sorry

end machine_a_produces_50_parts_in_10_minutes_l222_222138


namespace mark_total_payment_l222_222244

def total_cost (work_hours : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) : ℕ :=
  work_hours * hourly_rate + part_cost

theorem mark_total_payment :
  total_cost 2 75 150 = 300 :=
by
  -- Proof omitted, sorry used to skip the proof
  sorry

end mark_total_payment_l222_222244


namespace number_of_students_l222_222261

theorem number_of_students (n S : ℕ) 
  (h1 : S = 15 * n) 
  (h2 : (S + 36) / (n + 1) = 16) : 
  n = 20 :=
by 
  sorry

end number_of_students_l222_222261


namespace compute_expression_l222_222463

theorem compute_expression : 45 * 28 + 72 * 45 = 4500 :=
by
  sorry

end compute_expression_l222_222463


namespace Carolina_mailed_five_letters_l222_222406

-- Definitions translating the given conditions into Lean
def cost_of_mail (cost_letters cost_packages : ℝ) (num_letters num_packages : ℕ) : ℝ :=
  cost_letters * num_letters + cost_packages * num_packages

-- The main theorem to prove the desired answer
theorem Carolina_mailed_five_letters (P L : ℕ)
  (h1 : L = P + 2)
  (h2 : cost_of_mail 0.37 0.88 L P = 4.49) :
  L = 5 := 
sorry

end Carolina_mailed_five_letters_l222_222406


namespace cost_price_computer_table_l222_222938

theorem cost_price_computer_table (C : ℝ) (S : ℝ) (H1 : S = C + 0.60 * C) (H2 : S = 2000) : C = 1250 :=
by
  -- Proof goes here
  sorry

end cost_price_computer_table_l222_222938


namespace playground_perimeter_is_correct_l222_222683

-- Definition of given conditions
def length_of_playground : ℕ := 110
def width_of_playground : ℕ := length_of_playground - 15

-- Statement of the problem to prove
theorem playground_perimeter_is_correct :
  2 * (length_of_playground + width_of_playground) = 230 := 
by
  sorry

end playground_perimeter_is_correct_l222_222683


namespace second_divisor_correct_l222_222611

noncomputable def smallest_num: Nat := 1012
def known_divisors := [12, 18, 21, 28]
def lcm_divisors: Nat := 252 -- This is the LCM of 12, 18, 21, and 28.
def result: Nat := 14

theorem second_divisor_correct :
  ∃ (d : Nat), d ≠ 12 ∧ d ≠ 18 ∧ d ≠ 21 ∧ d ≠ 28 ∧ d ≠ 252 ∧ (smallest_num - 4) % d = 0 ∧ d = result :=
by
  sorry

end second_divisor_correct_l222_222611


namespace work_speed_ratio_l222_222299

open Real

theorem work_speed_ratio (A B : Type) 
  (A_work_speed B_work_speed : ℝ) 
  (combined_work_time : ℝ) 
  (B_work_time : ℝ)
  (h_combined : combined_work_time = 12)
  (h_B : B_work_time = 36)
  (combined_speed : A_work_speed + B_work_speed = 1 / combined_work_time)
  (B_speed : B_work_speed = 1 / B_work_time) :
  A_work_speed / B_work_speed = 2 :=
by sorry

end work_speed_ratio_l222_222299


namespace divisor_of_1025_l222_222284

theorem divisor_of_1025 : ∃ k : ℕ, 41 * k = 1025 :=
  sorry

end divisor_of_1025_l222_222284


namespace kevin_total_distance_l222_222788

noncomputable def kevin_hop_total_distance_after_seven_leaps : ℚ :=
  let a := (1 / 4 : ℚ)
  let r := (3 / 4 : ℚ)
  let n := 7
  a * (1 - r^n) / (1 - r)

theorem kevin_total_distance (total_distance : ℚ) :
  total_distance = kevin_hop_total_distance_after_seven_leaps → 
  total_distance = 14197 / 16384 := by
  intro h
  sorry

end kevin_total_distance_l222_222788


namespace total_jellybeans_needed_l222_222030

def large_glass_jellybeans : ℕ := 50
def small_glass_jellybeans : ℕ := large_glass_jellybeans / 2
def num_large_glasses : ℕ := 5
def num_small_glasses : ℕ := 3

theorem total_jellybeans_needed : 
  (num_large_glasses * large_glass_jellybeans) + (num_small_glasses * small_glass_jellybeans) = 325 := 
by
  sorry

end total_jellybeans_needed_l222_222030


namespace quadratic_has_distinct_real_roots_l222_222689

def discriminant (a b c : ℝ) : ℝ := b ^ 2 - 4 * a * c

theorem quadratic_has_distinct_real_roots :
  let a := 5
  let b := -2
  let c := -7
  discriminant a b c > 0 :=
by
  sorry

end quadratic_has_distinct_real_roots_l222_222689


namespace buckets_required_l222_222846

theorem buckets_required (C : ℕ) (h : C > 0) : 
  let original_buckets := 25
  let reduced_capacity := 2 / 5
  let total_capacity := original_buckets * C
  let new_buckets := total_capacity / ((2 / 5) * C)
  new_buckets = 63 := 
by
  sorry

end buckets_required_l222_222846


namespace pete_ten_dollar_bills_l222_222394

theorem pete_ten_dollar_bills (owes dollars bills: ℕ) (bill_value_per_bottle : ℕ) (num_bottles : ℕ) (ten_dollar_bills : ℕ):
  owes = 90 →
  dollars = 40 →
  bill_value_per_bottle = 5 →
  num_bottles = 20 →
  dollars + (num_bottles * bill_value_per_bottle) + (ten_dollar_bills * 10) = owes →
  ten_dollar_bills = 4 :=
by
  sorry

end pete_ten_dollar_bills_l222_222394


namespace patrick_age_l222_222921

theorem patrick_age (r_age_future : ℕ) (years_future : ℕ) (half_age : ℕ → ℕ) 
  (h1 : r_age_future = 30) (h2 : years_future = 2) 
  (h3 : ∀ n, half_age n = n / 2) :
  half_age (r_age_future - years_future) = 14 :=
by
  sorry

end patrick_age_l222_222921


namespace linear_equation_with_two_variables_l222_222377

def equation (a x y : ℝ) : ℝ := (a^2 - 4) * x^2 + (2 - 3 * a) * x + (a + 1) * y + 3 * a

theorem linear_equation_with_two_variables (a : ℝ) :
  (equation a x y = 0) ∧ (a^2 - 4 = 0) ∧ (2 - 3 * a ≠ 0) ∧ (a + 1 ≠ 0) →
  (a = 2 ∨ a = -2) :=
by sorry

end linear_equation_with_two_variables_l222_222377


namespace sum_of_80_consecutive_integers_l222_222419

-- Definition of the problem using the given conditions
theorem sum_of_80_consecutive_integers (n : ℤ) (h : (80 * (n + (n + 79))) / 2 = 40) : n = -39 := by
  sorry

end sum_of_80_consecutive_integers_l222_222419


namespace no_rational_solution_5x2_plus_3y2_eq_1_l222_222255

theorem no_rational_solution_5x2_plus_3y2_eq_1 :
  ¬ ∃ (x y : ℚ), 5 * x^2 + 3 * y^2 = 1 := 
sorry

end no_rational_solution_5x2_plus_3y2_eq_1_l222_222255


namespace problem_statement_l222_222775

def complex_number (m : ℂ) : ℂ :=
  (m^2 - 3*m - 4) + (m^2 - 5*m - 6) * Complex.I

theorem problem_statement (m : ℂ) :
  (complex_number m).im = m^2 - 5*m - 6 →
  (complex_number m).re = 0 →
  m ≠ -1 ∧ m ≠ 6 :=
by
  sorry

end problem_statement_l222_222775


namespace mapping_problem_l222_222543

open Set

noncomputable def f₁ (x : ℝ) : ℝ := Real.sqrt x
noncomputable def f₂ (x : ℝ) : ℝ := 1 / x
def f₃ (x : ℝ) : ℝ := x^2 - 2
def f₄ (x : ℝ) : ℝ := x^2

def A₁ : Set ℝ := {1, 4, 9}
def B₁ : Set ℝ := {-3, -2, -1, 1, 2, 3}
def A₂ : Set ℝ := univ
def B₂ : Set ℝ := univ
def A₃ : Set ℝ := univ
def B₃ : Set ℝ := univ
def A₄ : Set ℝ := {-1, 0, 1}
def B₄ : Set ℝ := {-1, 0, 1}

theorem mapping_problem : 
  ¬ (∀ x ∈ A₁, f₁ x ∈ B₁) ∧
  ¬ (∀ x ∈ A₂, x ≠ 0 → f₂ x ∈ B₂) ∧
  (∀ x ∈ A₃, f₃ x ∈ B₃) ∧
  (∀ x ∈ A₄, f₄ x ∈ B₄) :=
by
  sorry

end mapping_problem_l222_222543


namespace mango_rate_l222_222313

theorem mango_rate (x : ℕ) : 
  (sells_rate : ℕ) = 3 → 
  (profit_percent : ℕ) = 50 → 
  (buying_price : ℚ) = 2 := by
  sorry

end mango_rate_l222_222313


namespace purchase_price_is_60_l222_222153

variable (P S D : ℝ)
variable (GP : ℝ := 4)

theorem purchase_price_is_60
  (h1 : S = P + 0.25 * S)
  (h2 : D = 0.80 * S)
  (h3 : GP = D - P) :
  P = 60 :=
by
  sorry

end purchase_price_is_60_l222_222153


namespace hamburgers_left_over_l222_222723

theorem hamburgers_left_over (total_hamburgers served_hamburgers : ℕ) (h1 : total_hamburgers = 9) (h2 : served_hamburgers = 3) :
    total_hamburgers - served_hamburgers = 6 := by
  sorry

end hamburgers_left_over_l222_222723


namespace sum_of_max_marks_l222_222315

theorem sum_of_max_marks :
  ∀ (M S E : ℝ),
  (30 / 100 * M = 180) ∧
  (50 / 100 * S = 200) ∧
  (40 / 100 * E = 120) →
  M + S + E = 1300 :=
by
  intros M S E h
  sorry

end sum_of_max_marks_l222_222315


namespace christine_wander_time_l222_222730

noncomputable def distance : ℝ := 80
noncomputable def speed : ℝ := 20
noncomputable def time : ℝ := distance / speed

theorem christine_wander_time : time = 4 := 
by
  sorry

end christine_wander_time_l222_222730


namespace fruit_platter_has_thirty_fruits_l222_222251

-- Define the conditions
def at_least_five_apples (g_apple r_apple y_apple : ℕ) : Prop :=
  g_apple + r_apple + y_apple ≥ 5

def at_most_five_oranges (r_orange y_orange : ℕ) : Prop :=
  r_orange + y_orange ≤ 5

def kiwi_grape_constraints (g_kiwi p_grape : ℕ) : Prop :=
  g_kiwi + p_grape ≥ 8 ∧ g_kiwi + p_grape ≤ 12 ∧ g_kiwi = p_grape

def at_least_one_each_grape (g_grape p_grape : ℕ) : Prop :=
  g_grape ≥ 1 ∧ p_grape ≥ 1

-- The final statement to prove
theorem fruit_platter_has_thirty_fruits :
  ∃ (g_apple r_apple y_apple r_orange y_orange g_kiwi p_grape g_grape : ℕ),
    at_least_five_apples g_apple r_apple y_apple ∧
    at_most_five_oranges r_orange y_orange ∧
    kiwi_grape_constraints g_kiwi p_grape ∧
    at_least_one_each_grape g_grape p_grape ∧
    g_apple + r_apple + y_apple + r_orange + y_orange + g_kiwi + p_grape + g_grape = 30 :=
sorry

end fruit_platter_has_thirty_fruits_l222_222251


namespace train_crossing_time_l222_222859

/-- Prove the time it takes for a train of length 50 meters running at 60 km/hr to cross a pole is 3 seconds. -/
theorem train_crossing_time
  (speed_kmh : ℝ)
  (length_m : ℝ)
  (conversion_factor : ℝ)
  (time_seconds : ℝ) :
  speed_kmh = 60 →
  length_m = 50 →
  conversion_factor = 1000 / 3600 →
  time_seconds = 3 →
  time_seconds = length_m / (speed_kmh * conversion_factor) := 
by
  intros
  sorry

end train_crossing_time_l222_222859


namespace breadth_of_rectangular_plot_l222_222844

variable (A b l : ℝ)

theorem breadth_of_rectangular_plot :
  (A = 15 * b) ∧ (l = b + 10) ∧ (A = l * b) → b = 5 :=
by
  intro h
  sorry

end breadth_of_rectangular_plot_l222_222844


namespace train_crosses_pole_in_3_seconds_l222_222861

def train_problem (speed_kmh : ℕ) (length_m : ℕ) : ℕ :=
  let speed_ms := (speed_kmh * 1000) / 3600 in
  length_m / speed_ms

theorem train_crosses_pole_in_3_seconds :
  train_problem 60 50 = 3 :=
by
  -- We add a 'sorry' to skip the proof
  sorry

end train_crosses_pole_in_3_seconds_l222_222861


namespace prime_odd_sum_l222_222202

theorem prime_odd_sum (a b : ℕ) (h1 : Prime a) (h2 : Odd b) (h3 : a^2 + b = 2001) : a + b = 1999 :=
sorry

end prime_odd_sum_l222_222202


namespace peanut_butter_candy_count_l222_222423

theorem peanut_butter_candy_count (B G P : ℕ) 
  (hB : B = 43)
  (hG : G = B + 5)
  (hP : P = 4 * G) :
  P = 192 := by
  sorry

end peanut_butter_candy_count_l222_222423


namespace avg_difference_even_avg_difference_odd_l222_222845

noncomputable def avg (seq : List ℕ) : ℚ := (seq.sum : ℚ) / seq.length

def even_ints_20_to_60 := [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60]
def even_ints_10_to_140 := [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140]

def odd_ints_21_to_59 := [21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59]
def odd_ints_11_to_139 := [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 129, 131, 133, 135, 137, 139]

theorem avg_difference_even :
  avg even_ints_20_to_60 - avg even_ints_10_to_140 = -35 := sorry

theorem avg_difference_odd :
  avg odd_ints_21_to_59 - avg odd_ints_11_to_139 = -35 := sorry

end avg_difference_even_avg_difference_odd_l222_222845


namespace quadratic_roots_in_intervals_l222_222025

theorem quadratic_roots_in_intervals (a b c : ℝ) (h₁ : a < b) (h₂ : b < c) :
  ∃ x₁ x₂ : ℝ, (a < x₁ ∧ x₁ < b) ∧ (b < x₂ ∧ x₂ < c) ∧
  3 * x₁^2 - 2 * (a + b + c) * x₁ + (a * b + b * c + c * a) = 0 ∧
  3 * x₂^2 - 2 * (a + b + c) * x₂ + (a * b + b * c + c * a) = 0 :=
by
  sorry

end quadratic_roots_in_intervals_l222_222025


namespace num_subsets_of_P_l222_222368

open Finset

theorem num_subsets_of_P :
  let M := {0, 1, 2, 3, 4}
  let N := {1, 3, 5}
  let P := M ∩ N
  (P.card = 2) → card (powerset P) = 4 :=
by
  intros
  rw [card_powerset, card_eq_two]
  sorry

end num_subsets_of_P_l222_222368


namespace total_students_l222_222317

theorem total_students (a b c d e f : ℕ)  (h : a + b = 15) (h1 : a = 5) (h2 : b = 10) 
(h3 : c = 15) (h4 : d = 10) (h5 : e = 5) (h6 : f = 0) (h_total : a + b + c + d + e + f = 50) : a + b + c + d + e + f = 50 :=
by {exact h_total}

end total_students_l222_222317


namespace term_five_eq_nine_l222_222891

variable (S : ℕ → ℕ) (a : ℕ → ℕ)

-- The sum of the first n terms of the sequence equals n^2.
axiom sum_formula : ∀ n, S n = n^2

-- Definition of the nth term in terms of the sequence sum.
def a_n (n : ℕ) : ℕ := S n - S (n - 1)

-- Goal: Prove that the 5th term, a(5), equals 9.
theorem term_five_eq_nine : a_n S 5 = 9 :=
by
  sorry

end term_five_eq_nine_l222_222891


namespace combinations_with_repetition_l222_222182

theorem combinations_with_repetition (n k: ℕ) : 
  (∑ (x : Fin (n+k-1+1)), function.injective x) = nat.choose (n + k - 1) k := 
begin
  sorry
end

end combinations_with_repetition_l222_222182


namespace f_1984_and_f_1985_l222_222247

namespace Proof

variable {N M : Type} [AddMonoid M] [Zero M] (f : ℕ → M)

-- Conditions
axiom f_10 : f 10 = 0
axiom f_last_digit_3 {n : ℕ} : (n % 10 = 3) → f n = 0
axiom f_mn (m n : ℕ) : f (m * n) = f m + f n

-- Prove f(1984) = 0 and f(1985) = 0
theorem f_1984_and_f_1985 : f 1984 = 0 ∧ f 1985 = 0 :=
by
  sorry

end Proof

end f_1984_and_f_1985_l222_222247


namespace find_two_digit_numbers_l222_222178

theorem find_two_digit_numbers :
  ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) → (0 ≤ b ∧ b ≤ 9) → (10 * a + b = 3 * a * b) → (10 * a + b = 15 ∨ 10 * a + b = 24) :=
by
  intros
  sorry

end find_two_digit_numbers_l222_222178


namespace find_fourth_number_l222_222075

theorem find_fourth_number (a : ℕ → ℕ) (h1 : a 7 = 42) (h2 : a 9 = 110)
  (h3 : ∀ n, n ≥ 3 → a n = a (n - 1) + a (n - 2)) : a 4 = 10 := 
sorry

end find_fourth_number_l222_222075


namespace Alex_donut_holes_covered_l222_222728

noncomputable def Alex_radius : ℝ := 5
noncomputable def Bella_radius : ℝ := 7
noncomputable def Carlos_radius : ℝ := 9

def Alex_surface_area : ℝ := 4 * Real.pi * Alex_radius^2
def Bella_surface_area: ℝ := 4 * Real.pi * Bella_radius^2
def Carlos_surface_area : ℝ := 4 * Real.pi * Carlos_radius^2

theorem Alex_donut_holes_covered :
    ∀ (coating_rate : ℝ), 
    ∀ (start_time: ℝ), Alex_surface_area ≠ 0  → Bella_surface_area ≠ 0 → Carlos_surface_area ≠ 0 →
    ∀ t, t = 63504 * Real.pi →  
    t / Alex_surface_area = 635 :=
by
    intros; sorry

end Alex_donut_holes_covered_l222_222728


namespace history_book_cost_l222_222831

def total_books : ℕ := 90
def cost_math_book : ℕ := 4
def total_price : ℕ := 397
def math_books_bought : ℕ := 53

theorem history_book_cost :
  ∃ (H : ℕ), H = (total_price - (math_books_bought * cost_math_book)) / (total_books - math_books_bought) ∧ H = 5 :=
by
  sorry

end history_book_cost_l222_222831


namespace multiplication_factor_l222_222089

theorem multiplication_factor 
  (avg1 : ℕ → ℕ → ℕ)
  (avg2 : ℕ → ℕ → ℕ)
  (sum1 : ℕ)
  (num1 : ℕ)
  (num2 : ℕ)
  (sum2 : ℕ)
  (factor : ℚ) :
  avg1 sum1 num1 = 7 →
  avg2 sum2 num2 = 84 →
  sum1 = 10 * 7 →
  sum2 = 10 * 84 →
  factor = sum2 / sum1 →
  factor = 12 :=
by
  sorry

end multiplication_factor_l222_222089


namespace ammonia_formation_l222_222476

theorem ammonia_formation (Li3N H2O LiOH NH3 : ℕ) (h₁ : Li3N = 1) (h₂ : H2O = 54) (h₃ : Li3N + 3 * H2O = 3 * LiOH + NH3) :
  NH3 = 1 :=
by
  sorry

end ammonia_formation_l222_222476


namespace sum_of_sequence_l222_222644

noncomputable def sequence (a : ℕ → ℝ) := ∀ n, 2 ≤ n ∧ n ≤ 100 → a n + 2 * a (102 - n) = 3 * 2^n

theorem sum_of_sequence : 
  ∀ (a : ℕ → ℝ), 
  (a 1 = - 2 ^ 101) ∧ (sequence a) →
  (Finset.sum (Finset.range 100) a = -4) :=
by
  sorry

end sum_of_sequence_l222_222644


namespace negation_equivalence_l222_222934

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x ≤ 0) :=
by
  sorry

end negation_equivalence_l222_222934


namespace larger_number_is_30_l222_222942

-- Formalizing the conditions
variables (x y : ℝ)

-- Define the conditions given in the problem
def sum_condition : Prop := x + y = 40
def ratio_condition : Prop := x / y = 3

-- Formalize the problem statement
theorem larger_number_is_30 (h1 : sum_condition x y) (h2 : ratio_condition x y) : x = 30 :=
sorry

end larger_number_is_30_l222_222942


namespace range_of_a_l222_222240

noncomputable def f (x a : ℝ) : ℝ := x * abs (x - a)

theorem range_of_a (a : ℝ) :
  (∀ (x1 x2 : ℝ), 3 ≤ x1 ∧ 3 ≤ x2 ∧ x1 ≠ x2 → (x1 - x2) * (f x1 a - f x2 a) > 0) → a ≤ 3 :=
by sorry

end range_of_a_l222_222240


namespace range_of_m_l222_222480

theorem range_of_m (m : ℝ) (x : ℝ) (hp : (x + 2) * (x - 10) ≤ 0)
  (hq : x^2 - 2 * x + 1 - m^2 ≤ 0) (hm : m > 0) : 0 < m ∧ m ≤ 3 :=
sorry

end range_of_m_l222_222480


namespace retail_price_of_machine_l222_222433

theorem retail_price_of_machine 
  (wholesale_price : ℝ) 
  (discount_rate : ℝ) 
  (profit_rate : ℝ) 
  (selling_price : ℝ) 
  (P : ℝ)
  (h1 : wholesale_price = 90)
  (h2 : discount_rate = 0.10)
  (h3 : profit_rate = 0.20)
  (h4 : selling_price = wholesale_price * (1 + profit_rate))
  (h5 : (P * (1 - discount_rate)) = selling_price) : 
  P = 120 := by
  sorry

end retail_price_of_machine_l222_222433


namespace sum_of_a_b_l222_222009

theorem sum_of_a_b (a b : ℝ) (h1 : a > b) (h2 : |a| = 9) (h3 : b^2 = 4) : a + b = 11 ∨ a + b = 7 := 
sorry

end sum_of_a_b_l222_222009


namespace polynomial_perfect_square_l222_222085

theorem polynomial_perfect_square (x : ℝ) :
  (x + 1) * (x + 2) * (x + 3) * (x + 4) + 1 = (x^2 + 5 * x + 5)^2 :=
by 
  sorry

end polynomial_perfect_square_l222_222085


namespace girls_attending_sports_event_l222_222079

theorem girls_attending_sports_event 
  (total_students attending_sports_event : ℕ) 
  (girls boys : ℕ)
  (h1 : total_students = 1500)
  (h2 : attending_sports_event = 900)
  (h3 : girls + boys = total_students)
  (h4 : (1 / 2) * girls + (3 / 5) * boys = attending_sports_event) :
  (1 / 2) * girls = 500 := 
by
  sorry

end girls_attending_sports_event_l222_222079


namespace total_marbles_l222_222142

theorem total_marbles (marbles_per_row_8 : ℕ) (rows_of_9 : ℕ) (marbles_per_row_1 : ℕ) (rows_of_4 : ℕ) 
  (h1 : marbles_per_row_8 = 9) 
  (h2 : rows_of_9 = 8) 
  (h3 : marbles_per_row_1 = 4) 
  (h4 : rows_of_4 = 1) : 
  (marbles_per_row_8 * rows_of_9 + marbles_per_row_1 * rows_of_4) = 76 :=
by
  sorry

end total_marbles_l222_222142


namespace axis_of_symmetry_range_l222_222542

theorem axis_of_symmetry_range (a : ℝ) : (-(a + 2) / (3 - 4 * a) > 0) ↔ (a < -2 ∨ a > 3 / 4) :=
by
  sorry

end axis_of_symmetry_range_l222_222542


namespace find_fourth_number_l222_222078

theorem find_fourth_number (a : ℕ → ℕ) (h1 : a 7 = 42) (h2 : a 9 = 110)
  (h3 : ∀ n, n ≥ 3 → a n = a (n - 1) + a (n - 2)) : a 4 = 10 := 
sorry

end find_fourth_number_l222_222078


namespace melanies_mother_gave_l222_222246

-- Define initial dimes, dad's contribution, and total dimes now
def initial_dimes : ℕ := 7
def dad_dimes : ℕ := 8
def total_dimes : ℕ := 19

-- Define the number of dimes the mother gave
def mother_dimes := total_dimes - (initial_dimes + dad_dimes)

-- Proof statement
theorem melanies_mother_gave : mother_dimes = 4 := by
  sorry

end melanies_mother_gave_l222_222246


namespace cost_per_person_rounded_is_8_78_l222_222649

noncomputable def total_cost_before_discount : ℝ :=
  let c_cupcakes := 3.5 * 1.50
  let c_pastries := 2.25 * 2.75
  let c_muffins := 5 * 2.10
  c_cupcakes + c_pastries + c_muffins

noncomputable def total_cost_after_discount : ℝ :=
  let discount_amount := 0.20 * total_cost_before_discount
  total_cost_before_discount - discount_amount

noncomputable def cost_per_person : ℝ :=
  total_cost_after_discount / 2

noncomputable def round_to_nearest_cent (x : ℝ) : ℝ :=
  Real.round (x * 100) / 100

theorem cost_per_person_rounded_is_8_78 : round_to_nearest_cent cost_per_person = 8.78 := by
  sorry

end cost_per_person_rounded_is_8_78_l222_222649


namespace ben_eggs_remaining_l222_222323

def initial_eggs : ℕ := 75

def ben_day1_morning : ℝ := 5
def ben_day1_afternoon : ℝ := 4.5
def alice_day1_morning : ℝ := 3.5
def alice_day1_evening : ℝ := 4

def ben_day2_morning : ℝ := 7
def ben_day2_evening : ℝ := 3
def alice_day2_morning : ℝ := 2
def alice_day2_afternoon : ℝ := 4.5
def alice_day2_evening : ℝ := 1.5

def ben_day3_morning : ℝ := 4
def ben_day3_afternoon : ℝ := 3.5
def alice_day3_evening : ℝ := 6.5

def total_eggs_eaten : ℝ :=
  (ben_day1_morning + ben_day1_afternoon + alice_day1_morning + alice_day1_evening) +
  (ben_day2_morning + ben_day2_evening + alice_day2_morning + alice_day2_afternoon + alice_day2_evening) +
  (ben_day3_morning + ben_day3_afternoon + alice_day3_evening)

def remaining_eggs : ℝ :=
  initial_eggs - total_eggs_eaten

theorem ben_eggs_remaining : remaining_eggs = 26 := by
  -- proof goes here
  sorry

end ben_eggs_remaining_l222_222323


namespace square_side_length_equals_4_l222_222936

theorem square_side_length_equals_4 (s : ℝ) (h : s^2 = 4 * s) : s = 4 :=
sorry

end square_side_length_equals_4_l222_222936


namespace smallest_int_neither_prime_nor_square_no_prime_lt_70_l222_222835

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def has_no_prime_factor_less_than (n : ℕ) (k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p < k → ¬ p ∣ n

theorem smallest_int_neither_prime_nor_square_no_prime_lt_70
  (n : ℕ) : 
  n = 5183 ∧ ¬ is_prime n ∧ ¬ is_square n ∧ has_no_prime_factor_less_than n 70 ∧
  (∀ m : ℕ, 0 < m → m < 5183 →
    ¬ (¬ is_prime m ∧ ¬ is_square m ∧ has_no_prime_factor_less_than m 70)) :=
by sorry

end smallest_int_neither_prime_nor_square_no_prime_lt_70_l222_222835


namespace revenue_increase_l222_222702

theorem revenue_increase (P Q : ℝ) :
    let R := P * Q
    let P_new := 1.7 * P
    let Q_new := 0.8 * Q
    let R_new := P_new * Q_new
    R_new = 1.36 * R :=
sorry

end revenue_increase_l222_222702


namespace minimal_benches_l222_222450

theorem minimal_benches (x : ℕ) 
  (standard_adults : ℕ := x * 8) (standard_children : ℕ := x * 12)
  (extended_adults : ℕ := x * 8) (extended_children : ℕ := x * 16) 
  (hx : standard_adults + extended_adults = standard_children + extended_children) :
  x = 1 :=
by
  sorry

end minimal_benches_l222_222450


namespace hyperbola_ratio_l222_222196

theorem hyperbola_ratio (a b c : ℝ)
  (h1 : a > 0) (h2 : b > 0)
  (h_eq : a^2 - b^2 = 1)
  (h_ecc : 2 = c / a)
  (h_focus : c = 1) :
  a / b = Real.sqrt 3 / 3 := by
  have ha : a = 1 / 2 := sorry
  have hc : c = 1 := h_focus
  have hb : b = Real.sqrt 3 / 2 := sorry
  exact sorry

end hyperbola_ratio_l222_222196


namespace maximum_value_of_func_l222_222477

noncomputable def func (x y : ℝ) : ℝ := (x * y) / (x^2 + y^2)

def domain_x (x : ℝ) : Prop := (1/3 : ℝ) ≤ x ∧ x ≤ (2/5 : ℝ)
def domain_y (y : ℝ) : Prop := (1/2 : ℝ) ≤ y ∧ y ≤ (5/8 : ℝ)

theorem maximum_value_of_func :
  ∀ (x y : ℝ), domain_x x → domain_y y → func x y ≤ (20 / 21 : ℝ) ∧ 
  (∃ (x y : ℝ), domain_x x ∧ domain_y y ∧ func x y = (20 / 21 : ℝ)) :=
by sorry

end maximum_value_of_func_l222_222477


namespace find_fourth_number_l222_222066

variable (a : ℕ → ℕ)

theorem find_fourth_number (h₁ : a 7 = 42) (h₂ : a 9 = 110)
    (h₃ : ∀ n, n ≥ 3 → a n = a (n-1) + a (n-2)) : a 4 = 10 :=
by
  sorry

end find_fourth_number_l222_222066


namespace tile_covering_problem_l222_222312

theorem tile_covering_problem :
  let tile_length := 5
  let tile_width := 3
  let region_length := 5 * 12  -- converting feet to inches
  let region_width := 3 * 12   -- converting feet to inches
  let tile_area := tile_length * tile_width
  let region_area := region_length * region_width
  region_area / tile_area = 144 := 
by 
  let tile_length := 5
  let tile_width := 3
  let region_length := 5 * 12
  let region_width := 3 * 12
  let tile_area := tile_length * tile_width
  let region_area := region_length * region_width
  sorry

end tile_covering_problem_l222_222312


namespace initial_buckets_correct_l222_222250

-- Define the conditions as variables
def total_buckets : ℝ := 9.8
def added_buckets : ℝ := 8.8
def initial_buckets : ℝ := total_buckets - added_buckets

-- The theorem to prove the initial amount of water is 1 bucket
theorem initial_buckets_correct : initial_buckets = 1 := 
by
  sorry

end initial_buckets_correct_l222_222250


namespace find_a4_l222_222055

def seq (a : ℕ → ℕ) (n : ℕ) : Prop :=
(∀ n, n ≥ 3 → a n = a (n - 1) + a (n - 2))

theorem find_a4 (a : ℕ → ℕ) (h_seq : seq a) (h_a7 : a 7 = 42) (h_a9 : a 9 = 110) : a 4 = 10 :=
by
  sorry

end find_a4_l222_222055


namespace find_p_q_d_l222_222914

def f (p q d : ℕ) (x : ℤ) : ℤ :=
  if x > 0 then p * x + 4
  else if x = 0 then p * q
  else q * x + d

theorem find_p_q_d :
  ∃ p q d : ℕ, f p q d 3 = 7 ∧ f p q d 0 = 6 ∧ f p q d (-3) = -12 ∧ (p + q + d = 13) :=
by
  sorry

end find_p_q_d_l222_222914


namespace length_of_faster_train_l222_222693

-- Definitions for the given conditions
def speed_faster_train_kmh : ℝ := 50
def speed_slower_train_kmh : ℝ := 32
def time_seconds : ℝ := 15

theorem length_of_faster_train : 
  let speed_relative_kmh := speed_faster_train_kmh - speed_slower_train_kmh
  let speed_relative_mps := speed_relative_kmh * (1000 / 3600)
  let length_faster_train := speed_relative_mps * time_seconds
  length_faster_train = 75 := 
by 
  sorry 

end length_of_faster_train_l222_222693


namespace constants_unique_l222_222882

theorem constants_unique (A B C : ℝ) :
  (∀ x : ℝ, x ≠ 4 ∧ x ≠ 2 → (5 * x) / ((x - 4) * (x - 2) ^ 2) = A / (x - 4) + B / (x - 2) + C / (x - 2) ^ 2) ↔
  A = 5 ∧ B = -5 ∧ C = -5 :=
by
  sorry

end constants_unique_l222_222882


namespace club_membership_l222_222020

theorem club_membership:
  (∃ (committee : ℕ → Prop) (member_assign : (ℕ × ℕ) → ℕ → Prop),
    (∀ i, i < 5 → ∃! m, member_assign (i, m) 2) ∧
    (∀ i j, i < 5 ∧ j < 5 ∧ i ≠ j → ∃! m, m < 10 ∧ member_assign (i, j) m)
  ) → 
  ∃ n, n = 10 :=
by
  sorry

end club_membership_l222_222020


namespace students_per_van_correct_l222_222785

-- Define the conditions.
def num_vans : Nat := 6
def num_minibuses : Nat := 4
def students_per_minibus : Nat := 24
def total_students : Nat := 156

-- Define the number of students on each van is 'V'
def V : Nat := sorry 

-- State the final question/proof.
theorem students_per_van_correct : V = 10 :=
  sorry


end students_per_van_correct_l222_222785


namespace john_spent_fraction_l222_222787

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

end john_spent_fraction_l222_222787


namespace range_of_a_l222_222776

-- Defining the function f : ℝ → ℝ
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - 2 * x + a * Real.log x

-- Main theorem statement
theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0 ∧ (f a x1 = 0 ∧ f a x2 = 0)) → 0 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l222_222776


namespace sum_of_solutions_l222_222128

theorem sum_of_solutions (x1 x2 : ℝ) (h : ∀ (x : ℝ), x^2 - 10 * x + 14 = 0 → x = x1 ∨ x = x2) :
  x1 + x2 = 10 :=
sorry

end sum_of_solutions_l222_222128


namespace intersection_point_divides_chord_l222_222544

theorem intersection_point_divides_chord (R AB PO : ℝ)
    (hR: R = 11) (hAB: AB = 18) (hPO: PO = 7) :
    ∃ (AP PB : ℝ), (AP / PB = 2 ∨ AP / PB = 1 / 2) ∧ (AP + PB = AB) := by
  sorry

end intersection_point_divides_chord_l222_222544


namespace profit_percentage_l222_222309

theorem profit_percentage (CP SP : ℝ) (hCP : CP = 500) (hSP : SP = 725) : 
  100 * (SP - CP) / CP = 45 :=
by
  sorry

end profit_percentage_l222_222309


namespace total_cats_l222_222666

variable (initialCats : ℝ)
variable (boughtCats : ℝ)

theorem total_cats (h1 : initialCats = 11.0) (h2 : boughtCats = 43.0) :
    initialCats + boughtCats = 54.0 :=
by
  sorry

end total_cats_l222_222666


namespace square_area_l222_222858

theorem square_area (x1 x2 : ℝ) (hx1 : x1^2 + 4 * x1 + 3 = 8) (hx2 : x2^2 + 4 * x2 + 3 = 8) (h_eq : y = 8) : 
  (|x1 - x2|) ^ 2 = 36 :=
sorry

end square_area_l222_222858


namespace incorrect_statement_l222_222660

variable (f : ℝ → ℝ)
variable (k : ℝ)
variable (h₁ : f 0 = -1)
variable (h₂ : ∀ x, f' x > k)
variable (h₃ : k > 1)

theorem incorrect_statement :
  ¬ f (1 / (k - 1)) < 1 / (k - 1) :=
sorry

end incorrect_statement_l222_222660


namespace expected_winnings_l222_222719

def probability_heads : ℚ := 1 / 3
def probability_tails : ℚ := 1 / 2
def probability_edge : ℚ := 1 / 6

def winning_heads : ℚ := 2
def winning_tails : ℚ := 2
def losing_edge : ℚ := -4

def expected_value : ℚ := probability_heads * winning_heads + probability_tails * winning_tails + probability_edge * losing_edge

theorem expected_winnings : expected_value = 1 := by
  sorry

end expected_winnings_l222_222719


namespace intersection_M_N_l222_222661

def M : Set ℝ := {x | x^2 + x - 6 < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_M_N : M ∩ N = {x | 1 ≤ x ∧ x < 2} := by
  sorry

end intersection_M_N_l222_222661


namespace xiao_wang_exam_grades_l222_222836

theorem xiao_wang_exam_grades 
  (x y : ℕ) 
  (h1 : (x * y + 98) / (x + 1) = y + 1)
  (h2 : (x * y + 98 + 70) / (x + 2) = y - 1) : 
  x + 2 = 10 ∧ y - 1 = 88 := 
by
  sorry

end xiao_wang_exam_grades_l222_222836


namespace polynomial_simplification_l222_222086

theorem polynomial_simplification (x : ℝ) : 
  (3*x - 2)*(5*x^12 + 3*x^11 + 2*x^10 - x^9) = 15*x^13 - x^12 - 7*x^10 + 2*x^9 :=
by {
  sorry
}

end polynomial_simplification_l222_222086


namespace minimum_value_of_quadratic_function_l222_222003

def quadratic_function (a x : ℝ) : ℝ :=
  4 * x ^ 2 - 4 * a * x + (a ^ 2 - 2 * a + 2)

def min_value_in_interval (f : ℝ → ℝ) (a : ℝ) (interval : Set ℝ) (min_val : ℝ) : Prop :=
  ∀ x ∈ interval, f x ≥ min_val ∧ ∃ y ∈ interval, f y = min_val

theorem minimum_value_of_quadratic_function :
  ∃ a : ℝ, min_value_in_interval (quadratic_function a) a {x | 0 ≤ x ∧ x ≤ 1} 2 ↔ (a = 0 ∨ a = 3 + Real.sqrt 5) :=
by
  sorry

end minimum_value_of_quadratic_function_l222_222003


namespace average_of_four_variables_l222_222634

theorem average_of_four_variables (x y z w : ℝ) (h : (5 / 2) * (x + y + z + w) = 25) :
  (x + y + z + w) / 4 = 2.5 :=
sorry

end average_of_four_variables_l222_222634


namespace jellybean_total_l222_222027

theorem jellybean_total (large_jellybeans_per_glass : ℕ) 
  (small_jellybeans_per_glass : ℕ) 
  (num_large_glasses : ℕ) 
  (num_small_glasses : ℕ) 
  (h1 : large_jellybeans_per_glass = 50) 
  (h2 : small_jellybeans_per_glass = large_jellybeans_per_glass / 2) 
  (h3 : num_large_glasses = 5) 
  (h4 : num_small_glasses = 3) : 
  (num_large_glasses * large_jellybeans_per_glass + num_small_glasses * small_jellybeans_per_glass) = 325 :=
by
  sorry

end jellybean_total_l222_222027


namespace remainder_of_power_is_41_l222_222886

theorem remainder_of_power_is_41 : 
  ∀ (n k : ℕ), n = 2019 → k = 2018 → (n^k) % 100 = 41 :=
  by 
    intros n k hn hk 
    rw [hn, hk] 
    exact sorry

end remainder_of_power_is_41_l222_222886


namespace find_expression_l222_222511

theorem find_expression (x : ℝ) (h : (1 / Real.cos (2022 * x)) + Real.tan (2022 * x) = 1 / 2022) :
  (1 / Real.cos (2022 * x)) - Real.tan (2022 * x) = 2022 :=
by
  sorry

end find_expression_l222_222511


namespace polynomial_has_real_root_l222_222881

theorem polynomial_has_real_root (b : ℝ) : ∃ x : ℝ, x^3 + b * x^2 - 4 * x + b = 0 := 
sorry

end polynomial_has_real_root_l222_222881


namespace find_fourth_number_l222_222064

variable (a : ℕ → ℕ)

theorem find_fourth_number (h₁ : a 7 = 42) (h₂ : a 9 = 110)
    (h₃ : ∀ n, n ≥ 3 → a n = a (n-1) + a (n-2)) : a 4 = 10 :=
by
  sorry

end find_fourth_number_l222_222064


namespace pastries_and_juices_count_l222_222911

theorem pastries_and_juices_count 
  (budget : ℕ) 
  (cost_per_pastry : ℕ) 
  (cost_per_juice : ℕ) 
  (total_money : budget = 50)
  (pastry_cost : cost_per_pastry = 7) 
  (juice_cost : cost_per_juice = 2) : 
  ∃ (p j : ℕ), 7 * p + 2 * j ≤ 50 ∧ p + j = 7 :=
by
  sorry

end pastries_and_juices_count_l222_222911


namespace total_days_on_jury_duty_l222_222385

-- Define the conditions
def jury_selection_days : ℕ := 2
def trial_duration_factor : ℕ := 4
def deliberation_days : ℕ := 6
def deliberation_hours_per_day : ℕ := 16
def hours_per_day : ℕ := 24

-- Calculate the trial duration in days
def trial_days : ℕ := trial_duration_factor * jury_selection_days

-- Calculate the total deliberation time in days
def deliberation_total_hours : ℕ := deliberation_days * deliberation_hours_per_day
def deliberation_days_converted : ℕ := deliberation_total_hours / hours_per_day

-- Statement that John spends a total of 14 days on jury duty
theorem total_days_on_jury_duty : jury_selection_days + trial_days + deliberation_days_converted = 14 :=
sorry

end total_days_on_jury_duty_l222_222385


namespace largest_base4_to_base10_l222_222405

theorem largest_base4_to_base10 : 
  (3 * 4^2 + 3 * 4^1 + 3 * 4^0) = 63 := 
by
  -- sorry to skip the proof steps
  sorry

end largest_base4_to_base10_l222_222405


namespace find_c_l222_222629

-- Define the function f(x)
def f (x c : ℝ) : ℝ := x * (x - c) ^ 2

-- Define the first derivative of f(x)
def f_prime (x c : ℝ) : ℝ := 3 * x ^ 2 - 4 * c * x + c ^ 2

-- Define the condition that f(x) has a local maximum at x = 2
def is_local_max (f' : ℝ → ℝ) (x0 : ℝ) : Prop :=
  f' x0 = 0 ∧ (∀ x, x < x0 → f' x > 0) ∧ (∀ x, x > x0 → f' x < 0)

-- The main theorem stating the equivalent proof problem
theorem find_c (c : ℝ) : is_local_max (f_prime 2) 2 → c = 6 := 
  sorry

end find_c_l222_222629


namespace smallest_positive_integer_ends_in_3_divisible_by_11_l222_222119

theorem smallest_positive_integer_ends_in_3_divisible_by_11 :
  ∃ n : ℕ, n > 0 ∧ n % 10 = 3 ∧ n % 11 = 0 ∧ ∀ m : ℕ, (m > 0 ∧ m % 10 = 3 ∧ m % 11 = 0) → n ≤ m :=
sorry

end smallest_positive_integer_ends_in_3_divisible_by_11_l222_222119


namespace Pyarelal_loss_l222_222456

variables (capital_of_pyarelal capital_of_ashok : ℝ) (total_loss : ℝ)

def is_ninth (a b : ℝ) : Prop := a = b / 9

def applied_loss (loss : ℝ) (ratio : ℝ) : ℝ := ratio * loss

theorem Pyarelal_loss (h1: is_ninth capital_of_ashok capital_of_pyarelal) 
                        (h2: total_loss = 1600) : 
                        applied_loss total_loss (9/10) = 1440 :=
by 
  unfold is_ninth at h1
  sorry

end Pyarelal_loss_l222_222456


namespace fred_money_last_week_l222_222518

-- Definitions for the conditions in the problem
variables {f j : ℕ} (current_fred : ℕ) (current_jason : ℕ) (last_week_jason : ℕ)
variable (earning : ℕ)

-- Conditions
axiom Fred_current_money : current_fred = 115
axiom Jason_current_money : current_jason = 44
axiom Jason_last_week_money : last_week_jason = 40
axiom Earning_amount : earning = 4

-- Theorem statement: prove Fred's money last week
theorem fred_money_last_week (current_fred last_week_jason current_jason earning : ℕ)
  (Fred_current_money : current_fred = 115)
  (Jason_current_money : current_jason = 44)
  (Jason_last_week_money : last_week_jason = 40)
  (Earning_amount : earning = 4)
  : current_fred - earning = 111 :=
sorry

end fred_money_last_week_l222_222518


namespace sum_13_gt_0_l222_222269

noncomputable def a_n : ℕ → ℝ := sorry
noncomputable def S_n : ℕ → ℝ := sorry

axiom a7_gt_0 : 0 < a_n 7
axiom a8_lt_0 : a_n 8 < 0

theorem sum_13_gt_0 : S_n 13 > 0 :=
sorry

end sum_13_gt_0_l222_222269


namespace exists_difference_divisible_by_11_l222_222621

theorem exists_difference_divisible_by_11 (a : Fin 12 → ℤ) :
  ∃ (i j : Fin 12), i ≠ j ∧ 11 ∣ (a i - a j) :=
  sorry

end exists_difference_divisible_by_11_l222_222621


namespace largest_integer_x_l222_222884

theorem largest_integer_x (x : ℤ) : 
  (0.2 : ℝ) < (x : ℝ) / 7 ∧ (x : ℝ) / 7 < (7 : ℝ) / 12 → x = 4 :=
sorry

end largest_integer_x_l222_222884


namespace abigail_collected_43_l222_222318

noncomputable def cans_needed : ℕ := 100
noncomputable def collected_by_alyssa : ℕ := 30
noncomputable def more_to_collect : ℕ := 27
noncomputable def collected_by_abigail : ℕ := cans_needed - (collected_by_alyssa + more_to_collect)

theorem abigail_collected_43 : collected_by_abigail = 43 := by
  sorry

end abigail_collected_43_l222_222318


namespace teresa_age_at_michiko_birth_l222_222674

noncomputable def Teresa_age_now : ℕ := 59
noncomputable def Morio_age_now : ℕ := 71
noncomputable def Morio_age_at_Michiko_birth : ℕ := 38

theorem teresa_age_at_michiko_birth :
  (Teresa_age_now - (Morio_age_now - Morio_age_at_Michiko_birth)) = 26 := 
by
  sorry

end teresa_age_at_michiko_birth_l222_222674


namespace find_de_over_ef_l222_222908

-- Definitions based on problem conditions
variables {A B C D E F : Type} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D] [AddCommGroup E] [AddCommGroup F] 
variables (a b c d e f : A) 
variables (α β γ δ : ℝ)

-- Conditions
-- AD:DB = 2:3
def d_def : A := (3 / 5) • a + (2 / 5) • b
-- BE:EC = 1:4
def e_def : A := (4 / 5) • b + (1 / 5) • c
-- Intersection F of DE and AC
def f_def : A := (5 • d) - (10 • e)

-- Target Proof
theorem find_de_over_ef (h_d: d = d_def a b) (h_e: e = e_def b c) (h_f: f = f_def d e):
  DE / EF = 1 / 5 := 
sorry

end find_de_over_ef_l222_222908


namespace length_of_AB_l222_222573

theorem length_of_AB (x1 y1 x2 y2 : ℝ) 
  (h_parabola_A : y1^2 = 8 * x1) 
  (h_focus_line_A : y1 = 2 * (x1 - 2)) 
  (h_parabola_B : y2^2 = 8 * x2) 
  (h_focus_line_B : y2 = 2 * (x2 - 2)) 
  (h_sum_x : x1 + x2 = 6) : 
  |x1 - x2| = 10 :=
sorry

end length_of_AB_l222_222573


namespace inequality_proof_l222_222535

theorem inequality_proof (a b : ℝ) : a^2 + b^2 + 2 * (a - 1) * (b - 1) ≥ 1 :=
by
  sorry

end inequality_proof_l222_222535


namespace probability_of_sum_15_l222_222961

-- Defining the set of cards and the selection process
def cards : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- A function to check if the sum of chosen cards is 15
def is_sum_15 (s : Finset ℕ) : Prop :=
  s.sum id = 15

-- The set of all possible selections of 4 cards out of 6
def all_selections : Finset (Finset ℕ) := Finset.powersetLen 4 cards

-- The set of all selections where the sum is 15
def favorable_selections : Finset (Finset ℕ) :=
  Finset.filter is_sum_15 all_selections

-- The probability of selecting 4 cards which sum up to 15
def probability_sum_15 : ℚ :=
  (favorable_selections.card : ℚ) / (all_selections.card : ℚ)

theorem probability_of_sum_15 : probability_sum_15 = 2 / 15 := by
  sorry

end probability_of_sum_15_l222_222961


namespace percentage_increase_l222_222937

theorem percentage_increase (P : ℝ) (h : 200 * (1 + P/100) * 0.70 = 182) : 
  P = 30 := 
sorry

end percentage_increase_l222_222937


namespace find_fourth_number_l222_222068

variable (a : ℕ → ℕ)

theorem find_fourth_number (h₁ : a 7 = 42) (h₂ : a 9 = 110)
    (h₃ : ∀ n, n ≥ 3 → a n = a (n-1) + a (n-2)) : a 4 = 10 :=
by
  sorry

end find_fourth_number_l222_222068


namespace solution_set_of_inequality_l222_222941

theorem solution_set_of_inequality (x : ℝ) :
  (3 * x + 5) / (x - 1) > x ↔ x < -1 ∨ (1 < x ∧ x < 5) :=
sorry

end solution_set_of_inequality_l222_222941


namespace sin_y_eq_neg_one_l222_222646

noncomputable def α := Real.arccos (-1 / 5)

theorem sin_y_eq_neg_one (x y z : ℝ) (h1 : x = y - α) (h2 : z = y + α)
  (h3 : (2 + Real.sin x) * (2 + Real.sin z) = (2 + Real.sin y) ^ 2) : Real.sin y = -1 :=
sorry

end sin_y_eq_neg_one_l222_222646


namespace smallest_int_ends_in_3_div_by_11_l222_222124

theorem smallest_int_ends_in_3_div_by_11 :
  ∃ k : ℕ, k > 0 ∧ k % 10 = 3 ∧ k % 11 = 0 ∧ k = 33 :=
by {
  sorry
}

end smallest_int_ends_in_3_div_by_11_l222_222124


namespace butterfly_probability_l222_222593

-- Define the vertices of the cube
inductive Vertex
| A | B | C | D | E | F | G | H

open Vertex

-- Define the edges of the cube
def edges : Vertex → List Vertex
| A => [B, D, E]
| B => [A, C, F]
| C => [B, D, G]
| D => [A, C, H]
| E => [A, F, H]
| F => [B, E, G]
| G => [C, F, H]
| H => [D, E, G]

-- Define a function to simulate the butterfly's movement
noncomputable def move : Vertex → ℕ → List (Vertex × ℕ)
| v, 0 => [(v, 0)]
| v, n + 1 =>
  let nextMoves := edges v
  nextMoves.bind (λ v' => move v' n)

-- Define the probability calculation part
noncomputable def probability_of_visiting_all_vertices (n_moves : ℕ) : ℚ :=
  let total_paths := (3 ^ n_moves : ℕ)
  let valid_paths := 27 -- Based on given final solution step
  valid_paths / total_paths

-- Statement of the problem in Lean 4
theorem butterfly_probability :
  probability_of_visiting_all_vertices 11 = 27 / 177147 :=
by
  sorry

end butterfly_probability_l222_222593


namespace result_of_4_times_3_l222_222495

def operation (a b : ℕ) : ℕ :=
  a^2 + a * Nat.factorial b - b^2

theorem result_of_4_times_3 : operation 4 3 = 31 := by
  sorry

end result_of_4_times_3_l222_222495


namespace find_a_l222_222264

theorem find_a (a : ℤ) (h : ∃ x1 x2 : ℤ, (x - x1) * (x - x2) = (x - a) * (x - 8) - 1) : a = 8 :=
sorry

end find_a_l222_222264


namespace factorize_expression_l222_222172

theorem factorize_expression (m : ℝ) : m^2 + 3 * m = m * (m + 3) :=
by
  sorry

end factorize_expression_l222_222172


namespace slope_of_l_l222_222229

noncomputable def C (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 4 * Real.sin θ)
noncomputable def l (t α : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, 2 + t * Real.sin α)

theorem slope_of_l
  (α θ₁ θ₂ t₁ t₂ : ℝ)
  (h_midpoint : (C θ₁).fst + (C θ₂).fst = 1 + (t₁ + t₂) * Real.cos α ∧ 
                (C θ₁).snd + (C θ₂).snd = 2 + (t₁ + t₂) * Real.sin α) :
  Real.tan α = -2 :=
by
  sorry

end slope_of_l_l222_222229


namespace fourth_divisor_of_9600_l222_222697

theorem fourth_divisor_of_9600 (x : ℕ) (h1 : ∀ (d : ℕ), d = 15 ∨ d = 25 ∨ d = 40 → 9600 % d = 0) 
  (h2 : 9600 / Nat.lcm (Nat.lcm 15 25) 40 = x) : x = 16 := by
  sorry

end fourth_divisor_of_9600_l222_222697


namespace best_shooter_l222_222191

noncomputable def avg_A : ℝ := 9
noncomputable def avg_B : ℝ := 8
noncomputable def avg_C : ℝ := 9
noncomputable def avg_D : ℝ := 9

noncomputable def var_A : ℝ := 1.2
noncomputable def var_B : ℝ := 0.4
noncomputable def var_C : ℝ := 1.8
noncomputable def var_D : ℝ := 0.4

theorem best_shooter :
  (avg_A = 9 ∧ var_A = 1.2) →
  (avg_B = 8 ∧ var_B = 0.4) →
  (avg_C = 9 ∧ var_C = 1.8) →
  (avg_D = 9 ∧ var_D = 0.4) →
  avg_D = 9 ∧ var_D = 0.4 :=
by {
  sorry
}

end best_shooter_l222_222191


namespace vertex_position_l222_222595

-- Definitions based on the conditions of the problem
def quadratic_function (x : ℝ) : ℝ := 3*x^2 + 9*x + 5

-- Theorem that the vertex of the parabola is at x = -1.5
theorem vertex_position : ∃ x : ℝ, x = -1.5 ∧ ∀ y : ℝ, quadratic_function y ≥ quadratic_function x :=
by
  sorry

end vertex_position_l222_222595


namespace fred_earnings_l222_222233
noncomputable def start := 111
noncomputable def now := 115
noncomputable def earnings := now - start

theorem fred_earnings : earnings = 4 :=
by
  sorry

end fred_earnings_l222_222233


namespace total_amount_divided_l222_222146

theorem total_amount_divided (P1 : ℝ) (r1 : ℝ) (r2 : ℝ) (interest : ℝ) (T : ℝ) :
  P1 = 1550 →
  r1 = 0.03 →
  r2 = 0.05 →
  interest = 144 →
  (P1 * r1 + (T - P1) * r2 = interest) → T = 3500 :=
by
  intros hP1 hr1 hr2 hint htotal
  sorry

end total_amount_divided_l222_222146


namespace pencils_calculation_l222_222901

def num_pencil_boxes : ℝ := 4.0
def pencils_per_box : ℝ := 648.0
def total_pencils : ℝ := 2592.0

theorem pencils_calculation : (num_pencil_boxes * pencils_per_box) = total_pencils := 
by
  sorry

end pencils_calculation_l222_222901


namespace arithmetic_operation_equals_l222_222558

theorem arithmetic_operation_equals :
  12.1212 + 17.0005 - 9.1103 = 20.0114 := 
by 
  sorry

end arithmetic_operation_equals_l222_222558


namespace solution_count_l222_222486

noncomputable def count_positive_integer_solutions : Nat :=
  ∑' (x y z : ℕ) in {
      (x, y, z) |
      x > 0 ∧ 
      y > 0 ∧ 
      z > 0 ∧ 
      x + y + z = 15
  }, 1

theorem solution_count : count_positive_integer_solutions = 91 :=
by
  -- Proof to be provided
  sorry

end solution_count_l222_222486


namespace solve_x_l222_222710

noncomputable def diamond (a b : ℝ) : ℝ := a / b

axiom diamond_assoc (a b c : ℝ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (c_nonzero : c ≠ 0) : 
  diamond a (diamond b c) = a / (b / c)

axiom diamond_id (a : ℝ) (a_nonzero : a ≠ 0) : diamond a a = 1

theorem solve_x (x : ℝ) (h₁ : 1008 ≠ 0) (h₂ : 12 ≠ 0) (h₃ : x ≠ 0) : diamond 1008 (diamond 12 x) = 50 → x = 25 / 42 :=
by
  sorry

end solve_x_l222_222710


namespace find_a_l222_222354

theorem find_a (x y a : ℝ) (hx_pos_even : x > 0 ∧ ∃ n : ℕ, x = 2 * n) (hx_le_y : x ≤ y) 
  (h_eq_zero : |3 * y - 18| + |a * x - y| = 0) : 
  a = 3 ∨ a = 3 / 2 ∨ a = 1 :=
sorry

end find_a_l222_222354


namespace monotonicity_of_f_prime_range_of_a_for_real_roots_in_interval_l222_222364

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  Real.exp x - a * x^2

noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ :=
  Real.exp x - 2 * a * x

theorem monotonicity_of_f_prime :
  ∀ (x : ℝ) (a : ℝ), 
  (a ≤ 0 → f_prime x a > 0) ∧ 
  (a > 0 → (x < Real.log(2 * a) → f_prime x a < 0) ∧ (x > Real.log(2 * a) → f_prime x a > 0)) :=
sorry

theorem range_of_a_for_real_roots_in_interval :
  1 < a ∧ a < Real.exp 1 - 1 ↔
  ∃ (x : ℝ), 0 < x ∧ x < 1 ∧ f x a + f_prime x a = 2 - a * x^2 :=
sorry

end monotonicity_of_f_prime_range_of_a_for_real_roots_in_interval_l222_222364


namespace find_b_l222_222412

theorem find_b (a b c : ℤ) (h1 : a + b + c = 120) (h2 : a + 4 = b - 12) (h3 : a + 4 = 3 * c) : b = 60 :=
sorry

end find_b_l222_222412


namespace derivative_at_1_derivative_at_neg_2_derivative_at_x0_l222_222695

noncomputable def f (x : ℝ) : ℝ := 2 / x + x

theorem derivative_at_1 : (deriv f 1) = -1 :=
sorry

theorem derivative_at_neg_2 : (deriv f (-2)) = 1 / 2 :=
sorry

theorem derivative_at_x0 (x0 : ℝ) : (deriv f x0) = -2 / (x0^2) + 1 :=
sorry

end derivative_at_1_derivative_at_neg_2_derivative_at_x0_l222_222695


namespace find_vertex_parabola_l222_222628

-- Define the quadratic equation of the parabola
def parabola_eq (x y : ℝ) : Prop := x^2 - 4 * x + 3 * y + 10 = 0

-- Definition of the vertex of the parabola
def is_vertex (v : ℝ × ℝ) : Prop :=
  ∀ (x y : ℝ), parabola_eq x y → v = (2, -2)

-- The main statement we want to prove
theorem find_vertex_parabola : 
  ∃ v : ℝ × ℝ, is_vertex v :=
by
  use (2, -2)
  intros x y hyp
  sorry

end find_vertex_parabola_l222_222628


namespace evaluate_root_power_l222_222340

theorem evaluate_root_power : (Real.sqrt (Real.sqrt 9))^12 = 729 := 
by sorry

end evaluate_root_power_l222_222340


namespace length_of_row_of_small_cubes_l222_222713

/-!
# Problem: Calculate the length of a row of smaller cubes

A cube with an edge length of 0.5 m is cut into smaller cubes, each with an edge length of 2 mm.
Prove that the length of the row formed by arranging the smaller cubes in a continuous line 
is 31 km and 250 m.
-/

noncomputable def large_cube_edge_length_m : ℝ := 0.5
noncomputable def small_cube_edge_length_mm : ℝ := 2

theorem length_of_row_of_small_cubes :
  let length_mm := 31250000
  (31 : ℝ) * 1000 + (250 : ℝ) = length_mm / 1000 + 250 := 
sorry

end length_of_row_of_small_cubes_l222_222713


namespace find_k_l222_222639

-- Define the function y = kx
def linear_function (k x : ℝ) : ℝ := k * x

-- Define the point P(3,1)
def P : ℝ × ℝ := (3, 1)

theorem find_k (k : ℝ) (h : linear_function k 3 = 1) : k = 1 / 3 :=
by
  sorry

end find_k_l222_222639


namespace length_of_first_platform_l222_222453

noncomputable def speed (distance time : ℕ) :=
  distance / time

theorem length_of_first_platform 
  (L : ℕ) (train_length : ℕ) (time1 time2 : ℕ) (platform2_length : ℕ) (speed : ℕ) 
  (H1 : L + train_length = speed * time1) 
  (H2 : platform2_length + train_length = speed * time2) 
  (train_length_eq : train_length = 30) 
  (time1_eq : time1 = 12) 
  (time2_eq : time2 = 15) 
  (platform2_length_eq : platform2_length = 120) 
  (speed_eq : speed = 10) : L = 90 :=
by
  sorry

end length_of_first_platform_l222_222453


namespace winner_more_votes_than_second_place_l222_222321

theorem winner_more_votes_than_second_place :
  ∃ (W S T F : ℕ), 
    F = 199 ∧
    W = S + (W - S) ∧
    W = T + 79 ∧
    W = F + 105 ∧
    W + S + T + F = 979 ∧
    W - S = 53 :=
by
  sorry

end winner_more_votes_than_second_place_l222_222321


namespace main_l222_222389

def prop_p (x0 : ℝ) : Prop := x0 > -2 ∧ 6 + abs x0 = 5
def p : Prop := ∃ x : ℝ, prop_p x

def q : Prop := ∀ x : ℝ, x < 0 → x^2 + 4 / x^2 ≥ 4

def r : Prop := ∀ x y : ℝ, abs x + abs y ≤ 1 → abs y / (abs x + 2) ≤ 1 / 2
def not_r : Prop := ∃ x y : ℝ, abs x + abs y > 1 ∧ abs y / (abs x + 2) > 1 / 2

theorem main : ¬ p ∧ ¬ p ∨ r ∧ (p ∧ q) := by
  sorry

end main_l222_222389


namespace xyz_value_l222_222497

theorem xyz_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x + 1/y = 5) (h2 : y + 1/z = 3) (h3 : z + 1/x = 2) :
  x * y * z = 10 + 3 * Real.sqrt 11 :=
by
  sorry

end xyz_value_l222_222497


namespace peanut_butter_candy_count_l222_222421

-- Definitions derived from the conditions
def grape_candy (banana_candy : ℕ) := banana_candy + 5
def peanut_butter_candy (grape_candy : ℕ) := 4 * grape_candy

-- Given condition for the banana jar
def banana_candy := 43

-- The main theorem statement
theorem peanut_butter_candy_count : peanut_butter_candy (grape_candy banana_candy) = 192 :=
by
  sorry

end peanut_butter_candy_count_l222_222421


namespace appropriate_sampling_methods_l222_222276

-- Conditions for the first survey
structure Population1 where
  high_income_families : Nat
  middle_income_families : Nat
  low_income_families : Nat
  total : Nat := high_income_families + middle_income_families + low_income_families

def survey1_population : Population1 :=
  { high_income_families := 125,
    middle_income_families := 200,
    low_income_families := 95
  }

-- Condition for the second survey
structure Population2 where
  art_specialized_students : Nat

def survey2_population : Population2 :=
  { art_specialized_students := 5 }

-- The main statement to prove
theorem appropriate_sampling_methods :
  (survey1_population.total >= 100 → stratified_sampling_for_survey1) ∧ 
  (survey2_population.art_specialized_students >= 3 → simple_random_sampling_for_survey2) :=
  sorry

end appropriate_sampling_methods_l222_222276


namespace Clinton_belts_l222_222873

variable {Shoes Belts Hats : ℕ}

theorem Clinton_belts :
  (Shoes = 14) → (Shoes = 2 * Belts) → Belts = 7 :=
by
  sorry

end Clinton_belts_l222_222873


namespace arithmetic_seq_sum_l222_222227

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h₁ : ∀ n k : ℕ, a (n + k) = a n + k * d) 
  (h₂ : a 5 + a 6 + a 7 + a 8 = 20) : a 1 + a 12 = 10 := 
by 
  sorry

end arithmetic_seq_sum_l222_222227


namespace equation_of_latus_rectum_l222_222347

theorem equation_of_latus_rectum (y x : ℝ) : (x = -1/4) ∧ (y^2 = x) ↔ (2 * (1 / 2) = 1) ∧ (l = - (1 / 2) / 2) := sorry

end equation_of_latus_rectum_l222_222347


namespace find_fourth_number_l222_222076

theorem find_fourth_number (a : ℕ → ℕ) (h1 : a 7 = 42) (h2 : a 9 = 110)
  (h3 : ∀ n, n ≥ 3 → a n = a (n - 1) + a (n - 2)) : a 4 = 10 := 
sorry

end find_fourth_number_l222_222076


namespace angle_ACB_is_25_l222_222379

theorem angle_ACB_is_25 (angle_ABD angle_BAC : ℝ) (is_supplementary : angle_ABD + (180 - angle_BAC) = 180) (angle_ABC_eq : angle_BAC = 95) (angle_ABD_eq : angle_ABD = 120) :
  180 - (angle_BAC + (180 - angle_ABD)) = 25 :=
by
  sorry

end angle_ACB_is_25_l222_222379


namespace sugar_snap_peas_l222_222588

theorem sugar_snap_peas (P : ℕ) (h1 : P / 7 = 72 / 9) : P = 56 := 
sorry

end sugar_snap_peas_l222_222588


namespace problem_irrational_number_l222_222865

theorem problem_irrational_number :
  (¬ ∃ (a b : ℤ), b ≠ 0 ∧ (√3 : ℝ) = a / b) ∧ 
  (∃ (a b : ℤ), b ≠ 0 ∧ (0 : ℝ) = a / b) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (-2 : ℝ) = a / b) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (1 / 2 : ℝ) = a / b)
:=
by
  sorry

end problem_irrational_number_l222_222865


namespace nth_smallest_d0_perfect_square_l222_222717

theorem nth_smallest_d0_perfect_square (n : ℕ) : 
  ∃ (d_0 : ℕ), (∃ v : ℕ, ∀ t : ℝ, (2 * t * t + d_0 = v * t) ∧ (∃ k : ℕ, v = k ∧ k * k = v * v)) 
               ∧ d_0 = 4^(n - 1) := 
by sorry

end nth_smallest_d0_perfect_square_l222_222717


namespace scalene_triangle_smallest_angle_sum_l222_222021

theorem scalene_triangle_smallest_angle_sum :
  ∀ (A B C : ℝ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A = 45 ∧ C = 135 → (∃ x y : ℝ, x = y ∧ x = 45 ∧ y = 45 ∧ x + y = 90) :=
by
  intros A B C h
  sorry

end scalene_triangle_smallest_angle_sum_l222_222021


namespace coefficient_of_term_free_of_x_l222_222893

theorem coefficient_of_term_free_of_x 
  (n : ℕ) 
  (h1 : ∀ k : ℕ, k ≤ n → n = 10) 
  (h2 : (n.choose 4 / n.choose 2) = 14 / 3) : 
  ∃ (c : ℚ), c = 5 :=
by
  sorry

end coefficient_of_term_free_of_x_l222_222893


namespace rabbits_and_raccoons_l222_222726

variable (b_r t_r x : ℕ)

theorem rabbits_and_raccoons : 
  2 * b_r = x ∧ 3 * t_r = x ∧ b_r = t_r + 3 → x = 18 := 
by
  sorry

end rabbits_and_raccoons_l222_222726


namespace range_of_a_l222_222744

def prop_p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0
def prop_q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

theorem range_of_a (a : ℝ) : (prop_p a ∨ prop_q a) ∧ ¬(prop_p a ∧ prop_q a) ↔ (a < 0 ∨ (1 / 4 < a ∧ a < 4)) :=
by
  sorry

end range_of_a_l222_222744


namespace parabola_equation_l222_222680

-- Definitions for the given conditions
def parabola_vertex_origin (y x : ℝ) : Prop := y = 0 ↔ x = 0
def axis_of_symmetry_x (y x : ℝ) : Prop := (x = -y) ↔ (x = y)
def focus_on_line (y x : ℝ) : Prop := 3 * x - 4 * y - 12 = 0

-- The statement to be proved
theorem parabola_equation :
  ∀ (y x : ℝ),
  (parabola_vertex_origin y x) ∧ (axis_of_symmetry_x y x) ∧ (focus_on_line y x) →
  y^2 = 16 * x :=
by
  intros y x h
  sorry

end parabola_equation_l222_222680


namespace intersection_A_B_l222_222206

section
  def A : Set ℤ := {-2, 0, 1}
  def B : Set ℤ := {x | x^2 > 1}
  theorem intersection_A_B : A ∩ B = {-2} := 
  by
    sorry
end

end intersection_A_B_l222_222206


namespace total_earnings_l222_222134

theorem total_earnings (x y : ℝ) (h1 : 20 * x * y - 18 * x * y = 120) : 
  18 * x * y + 20 * x * y + 20 * x * y = 3480 := 
by
  sorry

end total_earnings_l222_222134


namespace difference_of_averages_l222_222088

theorem difference_of_averages :
  let avg1 := (20 + 40 + 60) / 3
  let avg2 := (10 + 70 + 16) / 3
  avg1 - avg2 = 8 :=
by
  sorry

end difference_of_averages_l222_222088


namespace fraction_zero_implies_x_is_minus_one_l222_222016

variable (x : ℝ)

theorem fraction_zero_implies_x_is_minus_one (h : (x^2 - 1) / (1 - x) = 0) : x = -1 :=
sorry

end fraction_zero_implies_x_is_minus_one_l222_222016


namespace ratio_of_side_lengths_l222_222868

theorem ratio_of_side_lengths (t s : ℕ) (ht : 2 * t + (20 - 2 * t) = 20) (hs : 4 * s = 20) :
  t / s = 4 / 3 :=
by
  sorry

end ratio_of_side_lengths_l222_222868


namespace interval_contains_root_l222_222404

noncomputable def f (x : ℝ) : ℝ := 3^x - x^2

theorem interval_contains_root : ∃ x ∈ Set.Icc (-1 : ℝ) (0 : ℝ), f x = 0 :=
by
  have f_neg : f (-1) < 0 := by sorry
  have f_zero : f 0 > 0 := by sorry
  sorry

end interval_contains_root_l222_222404


namespace smallest_product_of_non_factors_l222_222101

theorem smallest_product_of_non_factors (a b : ℕ) (h_a : a ∣ 48) (h_b : b ∣ 48) (h_distinct : a ≠ b) (h_prod_non_factor : ¬ (a * b ∣ 48)) : a * b = 18 :=
sorry

end smallest_product_of_non_factors_l222_222101


namespace find_m_value_l222_222200

theorem find_m_value
  (x y : ℤ)
  (h1 : x = 2)
  (h2 : y = m)
  (h3 : 3 * x + 2 * y = 10) : 
  m = 2 :=
by
  sorry

end find_m_value_l222_222200


namespace average_male_grade_l222_222928

theorem average_male_grade (avg_all avg_fem : ℝ) (N_male N_fem : ℕ) 
    (h1 : avg_all = 90) 
    (h2 : avg_fem = 92) 
    (h3 : N_male = 8) 
    (h4 : N_fem = 12) :
    let total_students := N_male + N_fem
    let total_sum_all := avg_all * total_students
    let total_sum_fem := avg_fem * N_fem
    let total_sum_male := total_sum_all - total_sum_fem
    let avg_male := total_sum_male / N_male
    avg_male = 87 :=
by 
  let total_students := N_male + N_fem
  let total_sum_all := avg_all * total_students
  let total_sum_fem := avg_fem * N_fem
  let total_sum_male := total_sum_all - total_sum_fem
  let avg_male := total_sum_male / N_male
  sorry

end average_male_grade_l222_222928


namespace victor_earnings_l222_222435

variable (wage hours_mon hours_tue : ℕ)

def hourly_wage : ℕ := 6
def hours_worked_monday : ℕ := 5
def hours_worked_tuesday : ℕ := 5

theorem victor_earnings :
  (hours_worked_monday + hours_worked_tuesday) * hourly_wage = 60 :=
by
  sorry

end victor_earnings_l222_222435


namespace smallest_positive_integer_ends_in_3_divisible_by_11_l222_222117

theorem smallest_positive_integer_ends_in_3_divisible_by_11 :
  ∃ n : ℕ, n > 0 ∧ n % 10 = 3 ∧ n % 11 = 0 ∧ n = 113 :=
by
  -- We claim that 113 is the required number
  use 113
  split
  -- Proof that 113 is positive
  sorry
  split
  -- Proof that 113 ends in 3
  sorry
  split
  -- Proof that 113 is divisible by 11
  sorry
  -- The smallest, smallest in scope will be evident by construction in the final formal proof
  sorry  

end smallest_positive_integer_ends_in_3_divisible_by_11_l222_222117


namespace rhombus_longer_diagonal_l222_222974

theorem rhombus_longer_diagonal 
  (a b : ℝ) 
  (h₁ : a = 61) 
  (h₂ : b = 44) :
  ∃ d₂ : ℝ, d₂ = 2 * Real.sqrt (a * a - (b / 2) * (b / 2)) :=
sorry

end rhombus_longer_diagonal_l222_222974


namespace simplify_expression1_simplify_expression2_l222_222258

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D F : V)

-- Problem 1:
theorem simplify_expression1 : 
  (D - C) + (C - B) + (B - A) = D - A := 
sorry

-- Problem 2:
theorem simplify_expression2 : 
  (B - A) + (F - D) + (D - C) + (C - B) + (A - F) = 0 := 
sorry

end simplify_expression1_simplify_expression2_l222_222258


namespace domain_f_l222_222346

noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x^2 - 5 * x + 6)

theorem domain_f :
  {x : ℝ | f x ≠ f x} = {x : ℝ | (x < 2) ∨ (2 < x ∧ x < 3) ∨ (3 < x)} :=
by sorry

end domain_f_l222_222346


namespace chessboard_markings_ways_l222_222008

/-- There are exactly 21600 ways to mark 8 squares of an 8x8 chessboard so that no two marked squares are in the same row or column, and none of the four corner squares is marked. -/
theorem chessboard_markings_ways :
  ∃ (squares : Finset (Fin 8 × Fin 8)), 
    squares.card = 8 ∧
    (∀ i j k l, (i, j) ∈ squares → (k, l) ∈ squares → (i ≠ k ∧ j ≠ l) ∧
      (i ≠ 0 ∨ j ≠ 0) ∧
      (i ≠ 0 ∨ j ≠ 7) ∧
      (i ≠ 7 ∨ j ≠ 0) ∧
      (i ≠ 7 ∨ j ≠ 7)) ∧
    squares.card = 21600 := sorry

end chessboard_markings_ways_l222_222008


namespace cost_of_soccer_ball_l222_222566

theorem cost_of_soccer_ball
  (F S : ℝ)
  (h1 : 3 * F + S = 155)
  (h2 : 2 * F + 3 * S = 220) :
  S = 50 :=
sorry

end cost_of_soccer_ball_l222_222566


namespace bowling_ball_weight_l222_222187

theorem bowling_ball_weight (b c : ℝ) (h1 : 5 * b = 3 * c) (h2 : 2 * c = 56) : b = 16.8 := by
  sorry

end bowling_ball_weight_l222_222187


namespace four_digit_non_convertible_to_1992_multiple_l222_222598

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_multiple_of_1992 (n : ℕ) : Prop :=
  n % 1992 = 0

def reachable (n m : ℕ) (k : ℕ) : Prop :=
  ∃ x y z : ℕ, 
    x ≠ m ∧ y ≠ m ∧ z ≠ m ∧
    (n + x * 10^(k-1) + y * 10^(k-2) + z * 10^(k-3)) % 1992 = 0 ∧
    n + x * 10^(k-1) + y * 10^(k-2) + z * 10^(k-3) < 10000

theorem four_digit_non_convertible_to_1992_multiple :
  ∃ n : ℕ, is_four_digit n ∧ (∀ m : ℕ, is_four_digit m ∧ is_multiple_of_1992 m → ¬ reachable n m 3) :=
sorry

end four_digit_non_convertible_to_1992_multiple_l222_222598


namespace fee_difference_l222_222553

-- Defining the given conditions
def stadium_capacity : ℕ := 2000
def fraction_full : ℚ := 3 / 4
def entry_fee : ℚ := 20

-- Statement to prove
theorem fee_difference :
  let people_at_three_quarters := stadium_capacity * fraction_full
  let total_fees_at_three_quarters := people_at_three_quarters * entry_fee
  let total_fees_full := stadium_capacity * entry_fee
  total_fees_full - total_fees_at_three_quarters = 10000 :=
by
  sorry

end fee_difference_l222_222553
