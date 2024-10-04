import Mathlib

namespace find_loss_percentage_l267_267552

theorem find_loss_percentage (CP SP_new : ℝ) (h1 : CP = 875) (h2 : SP_new = CP * 1.04) (h3 : SP_new = SP + 140) : 
  ∃ L : ℝ, SP = CP - (L / 100 * CP) → L = 12 := 
by 
  sorry

end find_loss_percentage_l267_267552


namespace jenny_profit_l267_267601

def cost_per_pan : ℝ := 10.00
def number_of_pans : ℝ := 20
def price_per_pan : ℝ := 25.00

theorem jenny_profit :
  let total_cost := cost_per_pan * number_of_pans in
  let total_revenue := price_per_pan * number_of_pans in
  let profit := total_revenue - total_cost in
  profit = 300 :=
by
  sorry

end jenny_profit_l267_267601


namespace smallest_three_digit_multiple_of_17_l267_267850

theorem smallest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, m >= 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  let n := 102,
  use n,
  split,
  { norm_num, }, -- n >= 100 is obviously true because n = 102
  split,
  { norm_num, }, -- n < 1000 is obviously true because n = 102
  split,
  { exact dvd_refl 17, }, -- 17 divides 102
  { intros m hm,
    cases hm with h100 hm,
    cases hm with h1000 hdiv,
    exact le_trans (nat.le.intro _ (symm hdiv)) (le_of_not_gt (λ h, by norm_num at h)), },
end

end smallest_three_digit_multiple_of_17_l267_267850


namespace line_length_after_erasing_l267_267437

theorem line_length_after_erasing :
  ∀ (initial_length_m : ℕ) (conversion_factor : ℕ) (erased_length_cm : ℕ),
  initial_length_m = 1 → conversion_factor = 100 → erased_length_cm = 33 →
  initial_length_m * conversion_factor - erased_length_cm = 67 :=
by {
  sorry
}

end line_length_after_erasing_l267_267437


namespace sum_of_roots_l267_267061

-- Given condition: the equation to be solved
def equation (x : ℝ) : Prop :=
  x^2 - 7 * x + 2 = 16

-- Define the proof problem: the sum of the values of x that satisfy the equation
theorem sum_of_roots : 
  (∀ x : ℝ, equation x) → x^2 - 7*x - 14 = 0 → (root : ℝ) → (root₀ + root₁) = 7 :=
by
  sorry

end sum_of_roots_l267_267061


namespace smallest_three_digit_multiple_of_17_l267_267829

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := 
begin
  use 102,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    sorry }
end

end smallest_three_digit_multiple_of_17_l267_267829


namespace rational_function_sum_l267_267814

-- Define the problem conditions and the target equality
theorem rational_function_sum (p q : ℝ → ℝ) :
  (∀ x, (p x) / (q x) = (x - 1) / ((x + 1) * (x - 1))) ∧
  (∀ x ≠ -1, q x ≠ 0) ∧
  (q 2 = 3) ∧
  (p 2 = 1) →
  (p x + q x = x^2 + x - 2) := by
  sorry

end rational_function_sum_l267_267814


namespace initial_number_of_orchids_l267_267647

theorem initial_number_of_orchids 
  (initial_orchids : ℕ)
  (cut_orchids : ℕ)
  (final_orchids : ℕ)
  (h_cut : cut_orchids = 19)
  (h_final : final_orchids = 21) :
  initial_orchids + cut_orchids = final_orchids → initial_orchids = 2 :=
by
  sorry

end initial_number_of_orchids_l267_267647


namespace altitude_segments_of_acute_triangle_l267_267523

/-- If two altitudes of an acute triangle divide the sides into segments of lengths 5, 3, 2, and x units,
then x is equal to 10. -/
theorem altitude_segments_of_acute_triangle (a b c d e : ℝ) (h1 : a = 5) (h2 : b = 3) (h3 : c = 2) (h4 : d = x) :
  x = 10 :=
by
  sorry

end altitude_segments_of_acute_triangle_l267_267523


namespace find_a_in_third_quadrant_l267_267243

theorem find_a_in_third_quadrant :
  ∃ a : ℝ, a < 0 ∧ 3 * a^2 + 4 * a^2 = 28 ∧ a = -2 :=
by
  sorry

end find_a_in_third_quadrant_l267_267243


namespace average_homework_time_decrease_l267_267400

theorem average_homework_time_decrease (x : ℝ) :
  (100 * (1 - x)^2 = 70) :=
sorry

end average_homework_time_decrease_l267_267400


namespace total_distance_is_27_l267_267780

-- Condition: Renaldo drove 15 kilometers
def renaldo_distance : ℕ := 15

-- Condition: Ernesto drove 7 kilometers more than one-third of Renaldo's distance
def ernesto_distance := (1 / 3 : ℚ) * renaldo_distance + 7

-- Theorem to prove that total distance driven by both men is 27 kilometers
theorem total_distance_is_27 : renaldo_distance + ernesto_distance = 27 := by
  sorry

end total_distance_is_27_l267_267780


namespace find_value_of_a_3m_2n_l267_267704

variable {a : ℝ} {m n : ℕ}
axiom h1 : a ^ m = 2
axiom h2 : a ^ n = 5

theorem find_value_of_a_3m_2n : a ^ (3 * m - 2 * n) = 8 / 25 := by
  sorry

end find_value_of_a_3m_2n_l267_267704


namespace subtracting_five_equals_thirtyfive_l267_267510

variable (x : ℕ)

theorem subtracting_five_equals_thirtyfive (h : x - 5 = 35) : x / 5 = 8 :=
sorry

end subtracting_five_equals_thirtyfive_l267_267510


namespace smallest_three_digit_multiple_of_17_l267_267940

/--
Prove that the smallest three-digit multiple of 17 is 102.
-/
theorem smallest_three_digit_multiple_of_17 : ∃ k : ℤ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 := 
by
  sorry

end smallest_three_digit_multiple_of_17_l267_267940


namespace range_of_m_l267_267742

theorem range_of_m {x m : ℝ} 
  (h1 : 1 / 3 < x) 
  (h2 : x < 1 / 2) 
  (h3 : |x - m| < 1) : 
  -1 / 2 ≤ m ∧ m ≤ 4 / 3 :=
by
  sorry

end range_of_m_l267_267742


namespace running_time_15mph_l267_267118

theorem running_time_15mph (x y z : ℝ) (h1 : x + y + z = 14) (h2 : 15 * x + 10 * y + 8 * z = 164) :
  x = 3 :=
sorry

end running_time_15mph_l267_267118


namespace cos_minus_sin_l267_267581

theorem cos_minus_sin (α : ℝ) (h1 : Real.sin (2 * α) = 1 / 4) (h2 : Real.pi / 4 < α ∧ α < Real.pi / 2) : 
  Real.cos α - Real.sin α = - (Real.sqrt 3) / 2 :=
sorry

end cos_minus_sin_l267_267581


namespace correct_model_l267_267394

def average_homework_time_decrease (x : ℝ) : Prop :=
  100 * (1 - x) ^ 2 = 70

theorem correct_model (x : ℝ) : average_homework_time_decrease x := 
  sorry

end correct_model_l267_267394


namespace simplify_and_evaluate_expression_l267_267789

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = Real.tan (Real.pi / 3) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2*m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l267_267789


namespace probability_of_earning_1900_equals_6_over_125_l267_267501

-- Representation of a slot on the spinner.
inductive Slot
| Bankrupt 
| Dollar1000
| Dollar500
| Dollar4000
| Dollar400 
deriving DecidableEq

-- Condition: There are 5 slots and each has the same probability.
noncomputable def slots := [Slot.Bankrupt, Slot.Dollar1000, Slot.Dollar500, Slot.Dollar4000, Slot.Dollar400]

-- Probability of earning exactly $1900 in three spins.
def probability_of_1900 : ℚ :=
  let target_combination := [Slot.Dollar500, Slot.Dollar400, Slot.Dollar1000]
  let total_ways := 125
  let successful_ways := 6
  (successful_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_earning_1900_equals_6_over_125 :
  probability_of_1900 = 6 / 125 :=
sorry

end probability_of_earning_1900_equals_6_over_125_l267_267501


namespace apples_in_box_l267_267199

theorem apples_in_box (total_fruit : ℕ) (one_fourth_oranges : ℕ) (half_peaches_oranges : ℕ) (apples_five_peaches : ℕ) :
  total_fruit = 56 →
  one_fourth_oranges = total_fruit / 4 →
  half_peaches_oranges = one_fourth_oranges / 2 →
  apples_five_peaches = 5 * half_peaches_oranges →
  apples_five_peaches = 35 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end apples_in_box_l267_267199


namespace number_of_people_per_cubic_yard_l267_267747

-- Lean 4 statement

variable (P : ℕ) -- Number of people per cubic yard

def city_population_9000 := 9000 * P
def city_population_6400 := 6400 * P

theorem number_of_people_per_cubic_yard :
  city_population_9000 - city_population_6400 = 208000 →
  P = 80 :=
by
  sorry

end number_of_people_per_cubic_yard_l267_267747


namespace smallest_triangle_perimeter_consecutive_even_l267_267220

theorem smallest_triangle_perimeter_consecutive_even :
  ∃ (a b c : ℕ), a = 2 ∧ b = 4 ∧ c = 6 ∧
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧ (a + b + c = 12) :=
by {
  sorry
}

end smallest_triangle_perimeter_consecutive_even_l267_267220


namespace sum_of_altitudes_l267_267732

theorem sum_of_altitudes (a b c : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) (h4 : a^2 + b^2 = c^2) : a + b = 21 :=
by
  -- Using the provided hypotheses, the proof would ensure a + b = 21.
  sorry

end sum_of_altitudes_l267_267732


namespace unique_real_x_satisfies_eq_l267_267534

theorem unique_real_x_satisfies_eq (x : ℝ) (h : x ≠ 0) : (7 * x)^5 = (14 * x)^4 ↔ x = 16 / 7 :=
by sorry

end unique_real_x_satisfies_eq_l267_267534


namespace arithmetic_sequence_eleven_term_l267_267442

theorem arithmetic_sequence_eleven_term (a1 d a11 : ℕ) (h_sum7 : 7 * (2 * a1 + 6 * d) = 154) (h_a1 : a1 = 5) :
  a11 = a1 + 10 * d → a11 = 25 :=
by
  sorry

end arithmetic_sequence_eleven_term_l267_267442


namespace average_inside_time_l267_267164

theorem average_inside_time (j_awake_frac : ℚ) (j_inside_awake_frac : ℚ) (r_awake_frac : ℚ) (r_inside_day_frac : ℚ) :
  j_awake_frac = 2 / 3 →
  j_inside_awake_frac = 1 / 2 →
  r_awake_frac = 3 / 4 →
  r_inside_day_frac = 2 / 3 →
  (24 * j_awake_frac * j_inside_awake_frac + 24 * r_awake_frac * r_inside_day_frac) / 2 = 10 := 
by
    sorry

end average_inside_time_l267_267164


namespace sum_of_first_n_terms_sequence_l267_267256

open Nat

def sequence_term (i : ℕ) : ℚ :=
  if i = 0 then 0 else 1 / (i * (i + 1) / 2 : ℕ)

def sum_of_sequence (n : ℕ) : ℚ :=
  (Finset.range (n+1)).sum fun i => sequence_term i

theorem sum_of_first_n_terms_sequence (n : ℕ) : sum_of_sequence n = 2 * n / (n + 1) := by
  sorry

end sum_of_first_n_terms_sequence_l267_267256


namespace certain_number_eq_neg17_l267_267659

theorem certain_number_eq_neg17 (x : Int) : 47 + x = 30 → x = -17 := by
  intro h
  have : x = 30 - 47 := by
    sorry  -- This is just to demonstrate the proof step. Actual manipulation should prove x = -17
  simp [this]

end certain_number_eq_neg17_l267_267659


namespace triangle_angle_sum_property_l267_267018

theorem triangle_angle_sum_property (A B C : ℝ) (h1: C = 3 * B) (h2: B = 15) : A = 120 :=
by
  -- Proof goes here
  sorry

end triangle_angle_sum_property_l267_267018


namespace jogging_track_circumference_l267_267042

theorem jogging_track_circumference 
  (deepak_speed : ℝ)
  (wife_speed : ℝ)
  (meeting_time : ℝ)
  (circumference : ℝ)
  (H1 : deepak_speed = 4.5)
  (H2 : wife_speed = 3.75)
  (H3 : meeting_time = 4.08) :
  circumference = 33.66 := sorry

end jogging_track_circumference_l267_267042


namespace calculation_l267_267254

theorem calculation : 8 - (7.14 * (1 / 3) - (20 / 9) / (5 / 2)) + 0.1 = 6.62 :=
by
  sorry

end calculation_l267_267254


namespace catch_two_salmon_l267_267357

def totalTroutWeight : ℕ := 8
def numBass : ℕ := 6
def weightPerBass : ℕ := 2
def totalBassWeight : ℕ := numBass * weightPerBass
def campers : ℕ := 22
def weightPerCamper : ℕ := 2
def totalFishWeightRequired : ℕ := campers * weightPerCamper
def totalTroutAndBassWeight : ℕ := totalTroutWeight + totalBassWeight
def additionalFishWeightRequired : ℕ := totalFishWeightRequired - totalTroutAndBassWeight
def weightPerSalmon : ℕ := 12
def numSalmon : ℕ := additionalFishWeightRequired / weightPerSalmon

theorem catch_two_salmon : numSalmon = 2 := by
  sorry

end catch_two_salmon_l267_267357


namespace parametric_area_l267_267559

noncomputable def x (t : ℝ) : ℝ := 3 * (t - Real.sin t)
noncomputable def y (t : ℝ) : ℝ := 3 * (1 - Real.cos t)

theorem parametric_area : 
  ∫ t in (π/2)..(3*π/2), y t * (deriv x t) = 9 * π + 18 :=
by
  -- Proof steps
  sorry

end parametric_area_l267_267559


namespace smallest_three_digit_multiple_of_17_l267_267887

theorem smallest_three_digit_multiple_of_17 : 
    ∃ (n : ℤ), n ≥ 100 ∧ n < 1000 ∧ (∃ (k : ℤ), n = 17 * k) ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l267_267887


namespace ab_square_l267_267033

theorem ab_square (x y : ℝ) (hx : y = 4 * x^2 + 7 * x - 1) (hy : y = -4 * x^2 + 7 * x + 1) :
  (2 * x)^2 + (2 * y)^2 = 50 :=
by
  sorry

end ab_square_l267_267033


namespace part1_part2_l267_267340

-- Definitions and conditions
variables (A B C : ℝ) (a b c : ℝ)
-- Given conditions
variable (h1 : sin C * sin (A - B) = sin B * sin (C - A))

-- Proof for part (1)
theorem part1 : 2 * a^2 = b^2 + c^2 :=
  sorry

-- Given further conditions for part (2)
variables (a_eq : a = 5) (cosA_eq : cos A = 25 / 31)

-- Proof for part (2)
theorem part2 : a + b + c = 14 :=
  sorry

end part1_part2_l267_267340


namespace find_a_l267_267636

theorem find_a 
  (a b c : ℤ) 
  (h_vertex : ∀ x, (a * (x - 2)^2 + 5 = a * x^2 + b * x + c))
  (h_point : ∀ y, y = a * (1 - 2)^2 + 5)
  : a = -1 := by
  sorry

end find_a_l267_267636


namespace fraction_zero_imp_x_eq_two_l267_267011
open Nat Real

theorem fraction_zero_imp_x_eq_two (x : ℝ) (h: (2 - abs x) / (x + 2) = 0) : x = 2 :=
by
  have h1 : 2 - abs x = 0 := sorry
  have h2 : x + 2 ≠ 0 := sorry
  sorry

end fraction_zero_imp_x_eq_two_l267_267011


namespace solve_for_x_l267_267529

theorem solve_for_x (x : ℝ) (h : x ≠ 0) : (7 * x) ^ 5 = (14 * x) ^ 4 → x = 16 / 7 :=
by
  sorry

end solve_for_x_l267_267529


namespace trajectory_of_M_circle_through_fixed_points_l267_267749

variables {M : ℝ × ℝ} {F : ℝ × ℝ}
def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem trajectory_of_M (x y : ℝ) :
  (distance M (1, 0) = distance M (x, 0) + 1) →
  (y^2 = 4 * x ∧ x ≥ 0 ∨ y = 0 ∧ x < 0) :=
sorry

theorem circle_through_fixed_points (x y : ℝ) (A B F : ℝ × ℝ) :
  let C := (set_of (λ p : ℝ × ℝ, p.snd^2 = 4 * p.fst ∧ p.fst ≥ 0)) in
  let line_PQ := (F.1, y) in
  let OP := (0, 0) in
  let OQ := (0, 0) in
  let A := (1, (4 / y)) in
  let B := (1, (4 / (y + 4))) in
  let circle := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 + 2*y)^2 = 4 * (y^2 + 1)} in
  (A ∈ circle ∧ B ∈ circle) →
  ((-1, 0) ∈ circle ∧ (3, 0) ∈ circle) :=
sorry

end trajectory_of_M_circle_through_fixed_points_l267_267749


namespace circumradius_of_triangle_ABC_l267_267275

noncomputable def circumradius (a b c : ℕ) : ℝ :=
  let s := (a + b + c) / 2
  let K := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  (a * b * c) / (4 * K)

theorem circumradius_of_triangle_ABC :
  (circumradius 12 10 7 = 6) :=
by
  sorry

end circumradius_of_triangle_ABC_l267_267275


namespace pizzas_ordered_l267_267412

def number_of_people : ℝ := 8.0
def slices_per_person : ℝ := 2.625
def slices_per_pizza : ℝ := 8.0

theorem pizzas_ordered : ⌈number_of_people * slices_per_person / slices_per_pizza⌉ = 3 := 
by
  sorry

end pizzas_ordered_l267_267412


namespace quotient_of_m_and_n_l267_267137

noncomputable def f (x : ℝ) : ℝ := |Real.log x|

theorem quotient_of_m_and_n (m n : ℝ) (h1 : 0 < m) (h2 : m < n) (h3 : f m = f n) (h4 : ∀ x ∈ Set.Icc (m^2) n, f x ≤ 2) :
  n / m = Real.exp 2 :=
by
  sorry

end quotient_of_m_and_n_l267_267137


namespace simplify_expression_l267_267797

noncomputable def m : ℝ := Real.tan (Real.pi / 3) - 1

theorem simplify_expression (h : m = Real.tan (Real.pi / 3) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2 * m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 := by
  sorry

end simplify_expression_l267_267797


namespace arithmetic_seq_2a9_a10_l267_267752

theorem arithmetic_seq_2a9_a10 (a : ℕ → ℕ) (h1 : a 1 = 1) (h3 : a 3 = 5) 
  (arith_seq : ∀ n : ℕ, ∃ d : ℕ, a n = a 1 + (n - 1) * d) : 2 * a 9 - a 10 = 15 :=
by
  sorry

end arithmetic_seq_2a9_a10_l267_267752


namespace no_solutions_l267_267372

theorem no_solutions (N : ℕ) (d : ℕ) (H : ∀ (i j : ℕ), i ≠ j → d = 6 ∧ d + d = 13) : false :=
by
  sorry

end no_solutions_l267_267372


namespace sarah_homework_problems_l267_267077

theorem sarah_homework_problems (math_pages reading_pages problems_per_page : ℕ) 
  (h1 : math_pages = 4) 
  (h2 : reading_pages = 6) 
  (h3 : problems_per_page = 4) : 
  (math_pages + reading_pages) * problems_per_page = 40 :=
by 
  sorry

end sarah_homework_problems_l267_267077


namespace smallest_three_digit_multiple_of_17_l267_267878

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := sorry

end smallest_three_digit_multiple_of_17_l267_267878


namespace binary_operation_l267_267266

def b11001 := 25  -- binary 11001 is 25 in decimal
def b1101 := 13   -- binary 1101 is 13 in decimal
def b101 := 5     -- binary 101 is 5 in decimal
def b100111010 := 314 -- binary 100111010 is 314 in decimal

theorem binary_operation : (b11001 * b1101 - b101) = b100111010 := by
  -- provide implementation details to prove the theorem
  sorry

end binary_operation_l267_267266


namespace packed_lunch_needs_l267_267551

-- Definitions based on conditions
def students_A : ℕ := 10
def students_B : ℕ := 15
def students_C : ℕ := 20

def total_students : ℕ := students_A + students_B + students_C

def slices_per_sandwich : ℕ := 4
def sandwiches_per_student : ℕ := 2
def bread_slices_per_student : ℕ := sandwiches_per_student * slices_per_sandwich
def total_bread_slices : ℕ := total_students * bread_slices_per_student

def bags_of_chips_per_student : ℕ := 1
def total_bags_of_chips : ℕ := total_students * bags_of_chips_per_student

def apples_per_student : ℕ := 3
def total_apples : ℕ := total_students * apples_per_student

def granola_bars_per_student : ℕ := 1
def total_granola_bars : ℕ := total_students * granola_bars_per_student

-- Proof goals
theorem packed_lunch_needs :
  total_bread_slices = 360 ∧
  total_bags_of_chips = 45 ∧
  total_apples = 135 ∧
  total_granola_bars = 45 :=
by
  sorry

end packed_lunch_needs_l267_267551


namespace large_monkey_doll_cost_l267_267115

theorem large_monkey_doll_cost (S L E : ℝ) 
  (h1 : S = L - 2) 
  (h2 : E = L + 1) 
  (h3 : 300 / S = 300 / L + 25) 
  (h4 : 300 / E = 300 / L - 15) : 
  L = 6 := 
sorry

end large_monkey_doll_cost_l267_267115


namespace spending_together_l267_267679

def sandwich_cost := 2
def hamburger_cost := 2
def hotdog_cost := 1
def juice_cost := 2
def selene_sandwiches := 3
def selene_juices := 1
def tanya_hamburgers := 2
def tanya_juices := 2

def selene_spending : ℕ := (selene_sandwiches * sandwich_cost) + (selene_juices * juice_cost)
def tanya_spending : ℕ := (tanya_hamburgers * hamburger_cost) + (tanya_juices * juice_cost)
def total_spending : ℕ := selene_spending + tanya_spending

theorem spending_together : total_spending = 16 :=
by
  sorry

end spending_together_l267_267679


namespace min_value_a_plus_2b_l267_267443

theorem min_value_a_plus_2b (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + 2 * b + 2 * a * b = 8) :
  a + 2 * b ≥ 4 :=
sorry

end min_value_a_plus_2b_l267_267443


namespace correct_model_l267_267396

def average_homework_time_decrease (x : ℝ) : Prop :=
  100 * (1 - x) ^ 2 = 70

theorem correct_model (x : ℝ) : average_homework_time_decrease x := 
  sorry

end correct_model_l267_267396


namespace smallest_three_digit_multiple_of_17_l267_267944

theorem smallest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m :=
by
  sorry

end smallest_three_digit_multiple_of_17_l267_267944


namespace y_coordinate_of_C_range_l267_267724

noncomputable def A : ℝ × ℝ := (0, 2)

def is_on_parabola (P : ℝ × ℝ) : Prop := (P.2)^2 = P.1 + 4

def is_perpendicular (A B C : ℝ × ℝ) : Prop := 
  let k_AB := (B.2 - A.2) / (B.1 - A.1)
  let k_BC := (C.2 - B.2) / (C.1 - B.1)
  k_AB * k_BC = -1

def range_of_y_C (y_C : ℝ) : Prop := y_C ≤ 0 ∨ y_C ≥ 4

theorem y_coordinate_of_C_range (B C : ℝ × ℝ)
  (hB : is_on_parabola B) (hC : is_on_parabola C) (h_perpendicular : is_perpendicular A B C) : 
  range_of_y_C (C.2) :=
sorry

end y_coordinate_of_C_range_l267_267724


namespace ribbon_each_box_fraction_l267_267315

-- Define the conditions
def total_ribbon_used : ℚ := 5 / 12
def number_of_boxes : ℕ := 5
def ribbon_per_box : ℚ := total_ribbon_used / number_of_boxes

-- Statement to be proved
theorem ribbon_each_box_fraction :
  ribbon_per_box = 1 / 12 :=
  sorry

end ribbon_each_box_fraction_l267_267315


namespace smallest_three_digit_multiple_of_17_l267_267949

theorem smallest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m :=
by
  sorry

end smallest_three_digit_multiple_of_17_l267_267949


namespace trade_ratio_blue_per_red_l267_267176

-- Define the problem conditions
def initial_total_marbles : ℕ := 10
def blue_percentage : ℕ := 40
def kept_red_marbles : ℕ := 1
def final_total_marbles : ℕ := 15

-- Find the number of blue marbles initially
def initial_blue_marbles : ℕ := (blue_percentage * initial_total_marbles) / 100

-- Calculate the number of red marbles initially
def initial_red_marbles : ℕ := initial_total_marbles - initial_blue_marbles

-- Calculate the number of red marbles traded
def traded_red_marbles : ℕ := initial_red_marbles - kept_red_marbles

-- Calculate the number of marbles received from the trade
def traded_marbles : ℕ := final_total_marbles - (initial_blue_marbles + kept_red_marbles)

-- The number of blue marbles received per each red marble traded
def blue_per_red : ℕ := traded_marbles / traded_red_marbles

-- Theorem stating that Pete's friend trades 2 blue marbles for each red marble
theorem trade_ratio_blue_per_red : blue_per_red = 2 := by
  -- Proof steps would go here
  sorry

end trade_ratio_blue_per_red_l267_267176


namespace hyperbola_point_A_l267_267147

def point (x y : ℝ) := (x, y)

section
open_locale real

variables (a b : ℝ) (h_ab : a > 0 ∧ b > 0)
variables (M A : ℝ × ℝ) (P F: ℝ × ℝ)

def hyperbola_eq := x^2 / a^2 - y^2 / b^2 = 1

def M := point (-1) (sqrt 3)

-- Assume M is symmetric with respect to the other asymptote to right focus F
def focus := point 2 0

def moving_point_on_hyperbola (P := point x y) :=
  P satisfies hyperbola_eq 

def point_A := point 3 1

def distance (p1 p2 : ℝ × ℝ) := sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def PA := distance P point_A

def PF := distance P focus

theorem hyperbola_point_A (
  h1 : M = point (-1) (sqrt 3),
  h2 : (P satisfies hyperbola_eq),
  h3 : point_A = point 3 1,
  h4 : focus = point 2 0 ) :
  PA + 1/2 * PF = 5 / 2 :=
begin
  sorry -- Proof to be filled
end

end hyperbola_point_A_l267_267147


namespace total_hexagons_calculation_l267_267598

-- Define the conditions
-- Regular hexagon side length
def hexagon_side_length : ℕ := 3

-- Number of smaller triangles
def small_triangle_count : ℕ := 54

-- Small triangle side length
def small_triangle_side_length : ℕ := 1

-- Define the total number of hexagons calculated
def total_hexagons : ℕ := 36

-- Theorem stating that given the conditions, the total number of hexagons is 36
theorem total_hexagons_calculation :
    (hexagon_side_length = 3) →
    (small_triangle_count = 54) →
    (small_triangle_side_length = 1) →
    total_hexagons = 36 :=
    by
    intros
    sorry

end total_hexagons_calculation_l267_267598


namespace range_of_m_l267_267478

theorem range_of_m {m : ℝ} : 
  (¬ ∃ x : ℝ, (1 / 2 ≤ x ∧ x ≤ 2 ∧ x^2 - 2 * x - m ≤ 0)) → m < -1 :=
by
  sorry

end range_of_m_l267_267478


namespace min_abs_difference_on_hyperbola_l267_267590

theorem min_abs_difference_on_hyperbola : 
  ∀ (x y : ℝ), (x^2 / 8 - y^2 / 4 = 1) → abs (x - y) ≥ 2 := 
by
  intros x y hxy
  sorry

end min_abs_difference_on_hyperbola_l267_267590


namespace xy_value_l267_267297

theorem xy_value (x y : ℝ) (h : (x + y)^2 - (x - y)^2 = 20) : x * y = 5 :=
by
  sorry

end xy_value_l267_267297


namespace five_digit_divisible_by_four_digit_l267_267608

theorem five_digit_divisible_by_four_digit (x y z u v : ℕ) (h1 : 1 ≤ x) (h2 : x < 10) (h3 : y < 10) (h4 : z < 10) (h5 : u < 10) (h6 : v < 10)
  (h7 : (x * 10^4 + y * 10^3 + z * 10^2 + u * 10 + v) % (x * 10^3 + y * 10^2 + u * 10 + v) = 0) : 
  ∃ N, 10 ≤ N ∧ N ≤ 99 ∧ 
  x * 10^4 + y * 10^3 + z * 10^2 + u * 10 + v = N * 10^3 ∧
  10 * (x * 10^3 + y * 10^2 + u * 10 + v) = x * 10^4 + y * 10^3 + z * 10^2 + u * 10 + v :=
sorry

end five_digit_divisible_by_four_digit_l267_267608


namespace ribbon_per_box_l267_267318

theorem ribbon_per_box (ribbon_total ribbon_each : ℚ) (n : ℕ) (hn : n = 5) (h : ribbon_total = 5 / 12) :
  ribbon_each = ribbon_total / n ↔ ribbon_each = 1 / 12 :=
by
  sorry

end ribbon_per_box_l267_267318


namespace cypress_tree_price_l267_267002

def amount_per_cypress_tree (C : ℕ) : Prop :=
  let cabin_price := 129000
  let cash := 150
  let cypress_count := 20
  let pine_count := 600
  let maple_count := 24
  let pine_price := 200
  let maple_price := 300
  let leftover_cash := 350
  let total_amount_raised := cabin_price - cash + leftover_cash
  let total_pine_maple := (pine_count * pine_price) + (maple_count * maple_price)
  let total_cypress := total_amount_raised - total_pine_maple
  let cypress_sale_price := total_cypress / cypress_count
  cypress_sale_price = C

theorem cypress_tree_price : amount_per_cypress_tree 100 :=
by {
  -- Proof skipped
  sorry
}

end cypress_tree_price_l267_267002


namespace print_time_325_pages_l267_267086

theorem print_time_325_pages (pages : ℕ) (rate : ℕ) (delay_pages : ℕ) (delay_time : ℕ)
  (h_pages : pages = 325) (h_rate : rate = 25) (h_delay_pages : delay_pages = 100) (h_delay_time : delay_time = 1) :
  let print_time := pages / rate
  let delays := pages / delay_pages
  let total_time := print_time + delays * delay_time
  total_time = 16 :=
by
  sorry

end print_time_325_pages_l267_267086


namespace jugglers_balls_needed_l267_267377

theorem jugglers_balls_needed (juggler_count balls_per_juggler : ℕ)
  (h_juggler_count : juggler_count = 378)
  (h_balls_per_juggler : balls_per_juggler = 6) :
  juggler_count * balls_per_juggler = 2268 :=
by
  -- This is where the proof would go.
  sorry

end jugglers_balls_needed_l267_267377


namespace new_ratio_books_to_clothes_l267_267640

-- Given initial conditions
def initial_ratio := (7, 4, 3)
def electronics_weight : ℕ := 12
def clothes_removed : ℕ := 8

-- Definitions based on the problem
def part_weight : ℕ := electronics_weight / initial_ratio.2.2
def initial_books_weight : ℕ := initial_ratio.1 * part_weight
def initial_clothes_weight : ℕ := initial_ratio.2.1 * part_weight
def new_clothes_weight : ℕ := initial_clothes_weight - clothes_removed

-- Proof of the new ratio
theorem new_ratio_books_to_clothes : (initial_books_weight, new_clothes_weight) = (7 * part_weight, 2 * part_weight) :=
sorry

end new_ratio_books_to_clothes_l267_267640


namespace smallest_three_digit_multiple_of_17_correct_l267_267917

def smallest_three_digit_multiple_of_17 : ℕ :=
  102

theorem smallest_three_digit_multiple_of_17_correct :
  ∀ n, n >= 100 → n < 1000 → n % 17 = 0 → 102 ≤ n :=
begin
  intros n h1 h2 h3,
  sorry,
end

end smallest_three_digit_multiple_of_17_correct_l267_267917


namespace smallest_three_digit_multiple_of_17_l267_267969

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ ∃ (k : ℕ), n = 17 * k ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l267_267969


namespace fractional_part_shaded_l267_267550

theorem fractional_part_shaded (a : ℝ) (r : ℝ) (sum : ℝ) 
    (h0 : a = 1 / 4) 
    (h1 : r = 1 / 16) 
    (h2 : sum = a / (1 - r)) : 
    sum = 4 / 15 :=
by
  rw [h0, h1] at h2
  exact h2

end fractional_part_shaded_l267_267550


namespace Marta_books_directly_from_bookstore_l267_267029

theorem Marta_books_directly_from_bookstore :
  let total_books_sale := 5
  let price_per_book_sale := 10
  let total_books_online := 2
  let total_cost_online := 40
  let total_spent := 210
  let cost_of_books_directly := 3 * total_cost_online
  let total_cost_sale := total_books_sale * price_per_book_sale
  let cost_per_book_directly := cost_of_books_directly / (total_cost_online / total_books_online)
  total_spent = total_cost_sale + total_cost_online + cost_of_books_directly ∧ (cost_of_books_directly / cost_per_book_directly) = 2 :=
by
  sorry

end Marta_books_directly_from_bookstore_l267_267029


namespace midpoint_one_seventh_one_ninth_l267_267265

theorem midpoint_one_seventh_one_ninth : 
  let a := (1 : ℚ) / 7
  let b := (1 : ℚ) / 9
  (a + b) / 2 = 8 / 63 := 
by
  sorry

end midpoint_one_seventh_one_ninth_l267_267265


namespace find_digit_D_l267_267374

theorem find_digit_D (A B C D : ℕ)
  (h_add : 100 + 10 * A + B + 100 * C + 10 * A + A = 100 * D + 10 * A + B)
  (h_sub : 100 + 10 * A + B - (100 * C + 10 * A + A) = 100 + 10 * A) :
  D = 1 :=
by
  -- Since we're skipping the proof and focusing on the statement only
  sorry

end find_digit_D_l267_267374


namespace ribbon_per_box_l267_267320

theorem ribbon_per_box (total_ribbon : ℚ) (total_boxes : ℕ) (same_ribbon_each_box : ℚ) 
  (h1 : total_ribbon = 5/12) (h2 : total_boxes = 5) : 
  same_ribbon_each_box = 1/12 :=
sorry

end ribbon_per_box_l267_267320


namespace max_eq_zero_max_two_solutions_l267_267526

theorem max_eq_zero_max_two_solutions {a b : Fin 10 → ℝ}
  (h : ∀ i, a i ≠ 0) : 
  ∃ (solution_count : ℕ), solution_count <= 2 ∧
  ∃ (solutions : Fin solution_count → ℝ), 
    ∀ (x : ℝ), (∀ i, max (a i * x + b i) = 0) ↔ ∃ j, x = solutions j := sorry

end max_eq_zero_max_two_solutions_l267_267526


namespace isosceles_triangle_perimeter_l267_267305

noncomputable def is_isosceles_triangle (a b c : ℕ) : Prop :=
  (a = b ∨ a = c ∨ b = c)

def roots_of_quadratic_eq := {x : ℕ | x^2 - 5 * x + 6 = 0}

theorem isosceles_triangle_perimeter
  (a b c : ℕ)
  (h_isosceles : is_isosceles_triangle a b c)
  (h_roots : (a ∈ roots_of_quadratic_eq) ∧ (b ∈ roots_of_quadratic_eq) ∧ (c ∈ roots_of_quadratic_eq)) :
  (a + b + c = 7 ∨ a + b + c = 8) :=
by
  sorry

end isosceles_triangle_perimeter_l267_267305


namespace melted_ice_cream_depth_l267_267549

theorem melted_ice_cream_depth
  (r_sphere : ℝ) (r_cylinder : ℝ) (V_sphere : ℝ) (V_cylinder : ℝ)
  (h : ℝ)
  (hr_sphere : r_sphere = 3)
  (hr_cylinder : r_cylinder = 10)
  (hV_sphere : V_sphere = 4 / 3 * Real.pi * r_sphere^3)
  (hV_cylinder : V_cylinder = Real.pi * r_cylinder^2 * h)
  (volume_conservation : V_sphere = V_cylinder) :
  h = 9 / 25 :=
by
  sorry

end melted_ice_cream_depth_l267_267549


namespace solve_abs_equation_l267_267802

theorem solve_abs_equation (y : ℤ) : (|y - 8| + 3 * y = 12) ↔ (y = 2) :=
by
  sorry  -- skip the proof steps.

end solve_abs_equation_l267_267802


namespace rented_movie_cost_l267_267774

def cost_of_tickets (c_ticket : ℝ) (n_tickets : ℕ) := c_ticket * n_tickets
def total_cost (cost_tickets cost_bought : ℝ) := cost_tickets + cost_bought
def remaining_cost (total_spent cost_so_far : ℝ) := total_spent - cost_so_far

theorem rented_movie_cost
  (c_ticket : ℝ)
  (n_tickets : ℕ)
  (c_bought : ℝ)
  (c_total : ℝ)
  (h1 : c_ticket = 10.62)
  (h2 : n_tickets = 2)
  (h3 : c_bought = 13.95)
  (h4 : c_total = 36.78) :
  remaining_cost c_total (total_cost (cost_of_tickets c_ticket n_tickets) c_bought) = 1.59 :=
by 
  sorry

end rented_movie_cost_l267_267774


namespace decision_has_two_exit_paths_l267_267662

-- Define types representing different flowchart symbols
inductive FlowchartSymbol
| Terminal
| InputOutput
| Process
| Decision

-- Define a function that states the number of exit paths given a flowchart symbol
def exit_paths (s : FlowchartSymbol) : Nat :=
  match s with
  | FlowchartSymbol.Terminal   => 1
  | FlowchartSymbol.InputOutput => 1
  | FlowchartSymbol.Process    => 1
  | FlowchartSymbol.Decision   => 2

-- State the theorem that Decision has two exit paths
theorem decision_has_two_exit_paths : exit_paths FlowchartSymbol.Decision = 2 := by
  sorry

end decision_has_two_exit_paths_l267_267662


namespace determine_n_between_sqrt3_l267_267493

theorem determine_n_between_sqrt3 (n : ℕ) (hpos : 0 < n)
  (hineq : (n + 3) / n < Real.sqrt 3 ∧ Real.sqrt 3 < (n + 4) / (n + 1)) :
  n = 4 :=
sorry

end determine_n_between_sqrt3_l267_267493


namespace ant_impossibility_l267_267557

-- Define the vertices and edges of a cube
structure Cube :=
(vertices : Finset ℕ) -- Representing a finite set of vertices
(edges : Finset (ℕ × ℕ)) -- Representing a finite set of edges between vertices
(valid_edge : ∀ e ∈ edges, ∃ v1 v2, (v1, v2) = e ∨ (v2, v1) = e)
(starting_vertex : ℕ)

-- Ant behavior on the cube
structure AntOnCube (C : Cube) :=
(is_path_valid : List ℕ → Prop) -- A property that checks the path is valid

-- Problem conditions translated: 
-- No retracing and specific visit numbers
noncomputable def ant_problem (C : Cube) (A : AntOnCube C) : Prop :=
  ∀ (path : List ℕ), A.is_path_valid path → ¬ (
    (path.count C.starting_vertex = 25) ∧ 
    (∀ v ∈ C.vertices, v ≠ C.starting_vertex → path.count v = 20)
  )

-- The final theorem statement
theorem ant_impossibility (C : Cube) (A : AntOnCube C) : ant_problem C A :=
by
  -- providing the theorem framework; proof omitted with sorry
  sorry

end ant_impossibility_l267_267557


namespace simplify_expression_l267_267795

noncomputable def m : ℝ := Real.tan (Real.pi / 3) - 1

theorem simplify_expression (h : m = Real.tan (Real.pi / 3) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2 * m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 := by
  sorry

end simplify_expression_l267_267795


namespace remainder_2457634_div_8_l267_267207

theorem remainder_2457634_div_8 : 2457634 % 8 = 2 := by
  sorry

end remainder_2457634_div_8_l267_267207


namespace sugar_price_difference_l267_267205

theorem sugar_price_difference (a b : ℝ) (h : (3 / 5 * a + 2 / 5 * b) - (2 / 5 * a + 3 / 5 * b) = 1.32) :
  a - b = 6.6 :=
by
  sorry

end sugar_price_difference_l267_267205


namespace michael_drove_miles_l267_267355

theorem michael_drove_miles (rental_fee charge_per_mile total_amount_paid : ℝ) (h_rental_fee : rental_fee = 20.99)
  (h_charge_per_mile : charge_per_mile = 0.25) (h_total_amount_paid : total_amount_paid = 95.74) :
  let amount_paid_for_miles := total_amount_paid - rental_fee in
  let number_of_miles := amount_paid_for_miles / charge_per_mile in
  number_of_miles = 299 := 
by
  sorry

end michael_drove_miles_l267_267355


namespace simplify_and_evaluate_expression_l267_267787

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = Real.tan (Real.pi / 3) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2*m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l267_267787


namespace tino_jellybeans_l267_267649

variable (Tino Lee Arnold : ℕ)

-- Conditions
variable (h1 : Tino = Lee + 24)
variable (h2 : Arnold = Lee / 2)
variable (h3 : Arnold = 5)

-- Prove Tino has 34 jellybeans
theorem tino_jellybeans : Tino = 34 :=
by
  sorry

end tino_jellybeans_l267_267649


namespace probability_one_doctor_one_nurse_l267_267540

theorem probability_one_doctor_one_nurse 
    (doctors nurses : ℕ) 
    (total_selected : ℕ) 
    (h_doctors : doctors = 3)
    (h_nurses : nurses = 2)
    (h_total_selected : total_selected = 2) : 
    (Nat.choose doctors 1) * (Nat.choose nurses 1) / (Nat.choose (doctors + nurses) total_selected) = 0.6 :=
by
  have h1 : Nat.choose doctors 1 = 3, by sorry
  have h2 : Nat.choose nurses 1 = 2, by sorry
  have h3 : Nat.choose 5 2 = 10, by sorry
  rw [h1, h2, h3]
  norm_num
  sorry

end probability_one_doctor_one_nurse_l267_267540


namespace sum_of_distinct_digits_base6_l267_267480

theorem sum_of_distinct_digits_base6 (A B C : ℕ) (hA : A < 6) (hB : B < 6) (hC : C < 6) 
  (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h_first_col : C + C % 6 = 4)
  (h_second_col : B + B % 6 = C)
  (h_third_col : A + B % 6 = A) :
  A + B + C = 6 := by
  sorry

end sum_of_distinct_digits_base6_l267_267480


namespace minimum_value_of_x_minus_y_l267_267278

variable (x y : ℝ)
open Real

theorem minimum_value_of_x_minus_y (hx : x > 0) (hy : y < 0) 
  (h : (1 / (x + 2)) + (1 / (1 - y)) = 1 / 6) : 
  x - y = 21 :=
sorry

end minimum_value_of_x_minus_y_l267_267278


namespace Tino_jellybeans_l267_267652

variable (L T A : ℕ)
variable (h1 : T = L + 24)
variable (h2 : A = L / 2)
variable (h3 : A = 5)

theorem Tino_jellybeans : T = 34 :=
by
  sorry

end Tino_jellybeans_l267_267652


namespace part_one_part_two_l267_267333

theorem part_one (A B C a b c : ℝ) (hA : A + B + C = π) 
  (h1 : sin(C) * sin(A - B) = sin(B) * sin(C - A)) : 
  2 * a^2 = b^2 + c^2 := by
  -- Placeholder for the detailed proof
  sorry

theorem part_two (a b c : ℝ) 
  (h1 : 2 * a^2 = b^2 + c^2) (h2 : a = 5) (h3 : cos(acos (5 / 31)) = 25 / 31) : 
  a + b + c = 14 := by
  -- Placeholder for the detailed proof
  sorry

end part_one_part_two_l267_267333


namespace g_g_is_odd_l267_267421

def f (x : ℝ) : ℝ := x^3

def g (x : ℝ) : ℝ := f (f x)

theorem g_g_is_odd : ∀ x : ℝ, g (g (-x)) = -g (g x) :=
by 
-- proof will go here
sorry

end g_g_is_odd_l267_267421


namespace probability_four_dots_collinear_l267_267754

-- Define the 5x5 grid and collinearity
structure Dot := (x : ℕ) (y : ℕ)

def is_collinear (d1 d2 d3 d4 : Dot) : Prop :=
  (d1.x = d2.x ∧ d2.x = d3.x ∧ d3.x = d4.x) ∨
  (d1.y = d2.y ∧ d2.y = d3.y ∧ d3.y = d4.y) ∨
  (d1.x - d1.y = d2.x - d2.y ∧ d2.x - d2.y = d3.x - d3.y ∧ d3.x - d3.y = d4.x - d4.y) ∨
  (d1.x + d1.y = d2.x + d2.y ∧ d2.x + d2.y = d3.x + d3.y ∧ d3.x + d3.y = d4.x + d4.y)

-- Count all combinations
noncomputable def comb (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Given conditions
def total_combinations : ℕ := comb 25 4

def collinear_sets : ℕ := 28

-- Proof statement: The probability that four random dots are collinear
theorem probability_four_dots_collinear :
  (collinear_sets : ℚ) / total_combinations = 14 / 6325 := by
  sorry

end probability_four_dots_collinear_l267_267754


namespace number_of_red_yarns_l267_267767

-- Definitions
def scarves_per_yarn : Nat := 3
def blue_yarns : Nat := 6
def yellow_yarns : Nat := 4
def total_scarves : Nat := 36

-- Theorem
theorem number_of_red_yarns (R : Nat) (H1 : scarves_per_yarn * blue_yarns + scarves_per_yarn * yellow_yarns + scarves_per_yarn * R = total_scarves) :
  R = 2 :=
by
  sorry

end number_of_red_yarns_l267_267767


namespace a_n_bound_l267_267379

theorem a_n_bound (a : ℕ → ℝ) :
  a 1 = 1 ∧ (∀ m n : ℕ, 0 < m ∧ 0 < n → (m + n) * a (m + n) ≤ a m + a n) →
  1 / a 200 > 4 * 10^7 := 
sorry

end a_n_bound_l267_267379


namespace expected_rolls_to_2010_l267_267430

noncomputable def expected_rolls_for_sum (n : ℕ) : ℝ :=
  if n = 2010 then 574.5238095 else sorry

theorem expected_rolls_to_2010 : expected_rolls_for_sum 2010 = 574.5238095 := sorry

end expected_rolls_to_2010_l267_267430


namespace average_homework_time_decrease_l267_267399

theorem average_homework_time_decrease (x : ℝ) :
  (100 * (1 - x)^2 = 70) :=
sorry

end average_homework_time_decrease_l267_267399


namespace smallest_third_term_geometric_l267_267548

theorem smallest_third_term_geometric (d : ℝ) : 
  (∃ d, (7 + d) ^ 2 = 4 * (26 + 2 * d)) → ∃ g3, (g3 = 10 ∨ g3 = 36) ∧ g3 = min (10) (36) :=
by
  sorry

end smallest_third_term_geometric_l267_267548


namespace CanVolume_l267_267232

variable (X Y : Type) [Field X] [Field Y] (V W : X)

theorem CanVolume (mix_ratioX mix_ratioY drawn_volume new_ratioX new_ratioY : ℤ)
  (h1 : mix_ratioX = 5) (h2 : mix_ratioY = 7) (h3 : drawn_volume = 12) 
  (h4 : new_ratioX = 4) (h5 : new_ratioY = 7) :
  V = 72 ∧ W = 72 := 
sorry

end CanVolume_l267_267232


namespace tangent_slope_at_point_x_eq_1_l267_267385

noncomputable def curve (x : ℝ) : ℝ := x^3 - 4 * x
noncomputable def curve_derivative (x : ℝ) : ℝ := 3 * x^2 - 4

theorem tangent_slope_at_point_x_eq_1 : curve_derivative 1 = -1 :=
by {
  -- This is just the theorem statement, no proof is required as per the instructions.
  sorry
}

end tangent_slope_at_point_x_eq_1_l267_267385


namespace olivia_used_pieces_l267_267617

-- Definition of initial pieces of paper and remaining pieces of paper
def initial_pieces : ℕ := 81
def remaining_pieces : ℕ := 25

-- Prove that Olivia used 56 pieces of paper
theorem olivia_used_pieces : (initial_pieces - remaining_pieces) = 56 :=
by
  -- Proof steps can be filled here
  sorry

end olivia_used_pieces_l267_267617


namespace smallest_repeating_block_of_3_over_11_l267_267469

def smallest_repeating_block_length (x y : ℕ) : ℕ :=
  if y ≠ 0 then (10 * x % y).natAbs else 0

theorem smallest_repeating_block_of_3_over_11 :
  smallest_repeating_block_length 3 11 = 2 :=
by sorry

end smallest_repeating_block_of_3_over_11_l267_267469


namespace smallest_three_digit_multiple_of_17_l267_267834

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := 
begin
  use 102,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    sorry }
end

end smallest_three_digit_multiple_of_17_l267_267834


namespace ribbon_per_box_l267_267317

theorem ribbon_per_box (ribbon_total ribbon_each : ℚ) (n : ℕ) (hn : n = 5) (h : ribbon_total = 5 / 12) :
  ribbon_each = ribbon_total / n ↔ ribbon_each = 1 / 12 :=
by
  sorry

end ribbon_per_box_l267_267317


namespace smallest_integer_N_l267_267257

theorem smallest_integer_N : ∃ (N : ℕ), 
  (∀ (a : ℕ → ℕ), ((∀ (i : ℕ), i < 125 -> a i > 0 ∧ a i ≤ N) ∧
  (∀ (i : ℕ), 1 ≤ i ∧ i < 124 → a i > (a (i - 1) + a (i + 1)) / 2) ∧
  (∀ (i j : ℕ), i < 125 ∧ j < 125 ∧ i ≠ j → a i ≠ a j)) → N = 2016) :=
sorry

end smallest_integer_N_l267_267257


namespace relationship_between_x_and_y_l267_267825

theorem relationship_between_x_and_y
  (z : ℤ)
  (x : ℝ)
  (y : ℝ)
  (h1 : x = (z^4 + z^3 + z^2 + z + 1) / (z^2 + 1))
  (h2 : y = (z^3 + z^2 + z + 1) / (z^2 + 1)) :
  (y^2 - 2 * y + 2) * (x + y - y^2) - 1 = 0 := 
by
  sorry

end relationship_between_x_and_y_l267_267825


namespace number_of_square_tiles_l267_267423

/-- A box contains a collection of triangular tiles, square tiles, and pentagonal tiles. 
    There are a total of 30 tiles in the box and a total of 100 edges. 
    We need to show that the number of square tiles is 10. --/
theorem number_of_square_tiles (a b c : ℕ) (h1 : a + b + c = 30) (h2 : 3 * a + 4 * b + 5 * c = 100) : b = 10 := by
  sorry

end number_of_square_tiles_l267_267423


namespace steve_assignments_fraction_l267_267509

theorem steve_assignments_fraction (h_sleep: ℝ) (h_school: ℝ) (h_family: ℝ) (total_hours: ℝ) : 
  h_sleep = 1/3 ∧ 
  h_school = 1/6 ∧ 
  h_family = 10 ∧ 
  total_hours = 24 → 
  (2 / total_hours = 1 / 12) :=
by
  intros h
  sorry

end steve_assignments_fraction_l267_267509


namespace find_k_l267_267637

theorem find_k (k : ℝ) (r s : ℝ) (h1 : r ≠ 0) (h2 : s ≠ 0)
  (h3 : r / s = 3) (h4 : r + s = 4) (h5 : r * s = k) : k = 3 :=
sorry

end find_k_l267_267637


namespace geometric_arithmetic_sequence_ratio_l267_267571

-- Given a positive geometric sequence {a_n} with a_3, a_5, a_6 forming an arithmetic sequence,
-- we need to prove that (a_3 + a_5) / (a_4 + a_6) is among specific values {1, (sqrt 5 - 1) / 2}

theorem geometric_arithmetic_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (h_pos: ∀ n, 0 < a n)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_arith : 2 * a 5 = a 3 + a 6) :
  (a 3 + a 5) / (a 4 + a 6) = 1 ∨ (a 3 + a 5) / (a 4 + a 6) = (Real.sqrt 5 - 1) / 2 :=
by
  -- The proof is omitted
  sorry

end geometric_arithmetic_sequence_ratio_l267_267571


namespace cylinder_volume_eq_pi_over_4_l267_267300

theorem cylinder_volume_eq_pi_over_4
  (r : ℝ)
  (h₀ : r > 0)
  (h₁ : 2 * r = r * 2)
  (h₂ : 4 * π * r^2 = π) : 
  (π * r^2 * (2 * r) = π / 4) :=
by
  sorry

end cylinder_volume_eq_pi_over_4_l267_267300


namespace cleaning_time_together_l267_267026

theorem cleaning_time_together (lisa_time kay_time ben_time sarah_time : ℕ)
  (h_lisa : lisa_time = 8) (h_kay : kay_time = 12) 
  (h_ben : ben_time = 16) (h_sarah : sarah_time = 24) :
  1 / ((1 / (lisa_time:ℚ)) + (1 / (kay_time:ℚ)) + (1 / (ben_time:ℚ)) + (1 / (sarah_time:ℚ))) = (16 / 5 : ℚ) :=
by
  sorry

end cleaning_time_together_l267_267026


namespace complement_union_eq_l267_267386

universe u

-- Definitions based on conditions in a)
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 3}

-- The goal to prove based on c)
theorem complement_union_eq :
  (U \ (M ∪ N)) = {5, 6} := 
by sorry

end complement_union_eq_l267_267386


namespace length_of_plot_l267_267087

theorem length_of_plot (total_poles : ℕ) (distance : ℕ) (one_side : ℕ) (other_side : ℕ) 
  (poles_distance_condition : total_poles = 28) 
  (fencing_condition : distance = 10) 
  (side_condition : one_side = 50) 
  (rectangular_condition : total_poles = (2 * (one_side / distance) + 2 * (other_side / distance))) :
  other_side = 120 :=
by
  sorry

end length_of_plot_l267_267087


namespace wheel_distance_travelled_l267_267553

noncomputable def radius : ℝ := 3
noncomputable def num_revolutions : ℝ := 3
noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r
noncomputable def total_distance (r : ℝ) (n : ℝ) : ℝ := n * circumference r

theorem wheel_distance_travelled :
  total_distance radius num_revolutions = 18 * Real.pi :=
by 
  sorry

end wheel_distance_travelled_l267_267553


namespace power_function_half_value_l267_267294

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x ^ a

theorem power_function_half_value (a : ℝ) (h : (f 4 a) / (f 2 a) = 3) :
  f (1 / 2) a = 1 / 3 :=
by
  sorry  -- Proof goes here

end power_function_half_value_l267_267294


namespace smallest_three_digit_multiple_of_17_l267_267871

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m := by
  use 102
  sorry

end smallest_three_digit_multiple_of_17_l267_267871


namespace gcd_4320_2550_l267_267824

-- Definitions for 4320 and 2550
def a : ℕ := 4320
def b : ℕ := 2550

-- Statement to prove the greatest common factor of a and b is 30
theorem gcd_4320_2550 : Nat.gcd a b = 30 := 
by 
  sorry

end gcd_4320_2550_l267_267824


namespace sum_of_roots_l267_267065

-- Defined the equation x^2 - 7x + 2 - 16 = 0 as x^2 - 7x - 14 = 0
def equation (x : ℝ) := x^2 - 7 * x - 14 = 0 

-- State the theorem leveraging the above condition
theorem sum_of_roots : 
  (∃ x1 x2 : ℝ, equation x1 ∧ equation x2 ∧ x1 ≠ x2) →
  (∃ sum : ℝ, sum = 7) := by
  sorry

end sum_of_roots_l267_267065


namespace throws_to_return_to_elsa_l267_267055

theorem throws_to_return_to_elsa :
  ∃ n, n = 5 ∧ (∀ (k : ℕ), k < n → ((1 + 5 * k) % 13 ≠ 1)) ∧ (1 + 5 * n) % 13 = 1 :=
by
  sorry

end throws_to_return_to_elsa_l267_267055


namespace smallest_three_digit_multiple_of_17_l267_267873

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := sorry

end smallest_three_digit_multiple_of_17_l267_267873


namespace cost_of_drill_bits_l267_267239

theorem cost_of_drill_bits (x : ℝ) (h1 : 5 * x + 0.10 * (5 * x) = 33) : x = 6 :=
sorry

end cost_of_drill_bits_l267_267239


namespace least_range_product_multiple_840_l267_267740

def is_multiple (x y : Nat) : Prop :=
  ∃ k : Nat, y = k * x

theorem least_range_product_multiple_840 : 
  ∃ (a : Nat), a > 0 ∧ ∀ (n : Nat), (n = 3) → is_multiple 840 (List.foldr (· * ·) 1 (List.range' a n)) := 
by {
  sorry
}

end least_range_product_multiple_840_l267_267740


namespace smallest_three_digit_multiple_of_17_l267_267914

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ 17 * n ∧ 17 * n ≤ 999 ∧ 17 * n = 102 :=
begin
  sorry
end

end smallest_three_digit_multiple_of_17_l267_267914


namespace sum_of_x_and_y_l267_267577

theorem sum_of_x_and_y (x y : ℤ) (h : 2 * x * y + x + y = 83) : x + y = 83 ∨ x + y = -85 :=
sorry

end sum_of_x_and_y_l267_267577


namespace smallest_three_digit_multiple_of_17_l267_267916

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ 17 * n ∧ 17 * n ≤ 999 ∧ 17 * n = 102 :=
begin
  sorry
end

end smallest_three_digit_multiple_of_17_l267_267916


namespace smallest_perimeter_even_integer_triangl_l267_267212

theorem smallest_perimeter_even_integer_triangl (n : ℕ) (h : n > 2) :
  let a := 2 * n - 2
  let b := 2 * n
  let c := 2 * n + 2
  2 * n - 2 + 2 * n > 2 * n + 2 ∧
  2 * n - 2 + 2 * n + 2 > 2 * n ∧
  2 * n + 2 * n + 2 > 2 * n - 2 ∧ 
  2 * 3 - 2 + 2 * 3 + 2 * 3 + 2 = 18 :=
by
  { sorry }

end smallest_perimeter_even_integer_triangl_l267_267212


namespace avg_meal_cost_per_individual_is_72_l267_267431

theorem avg_meal_cost_per_individual_is_72
  (total_bill : ℝ)
  (gratuity_percent : ℝ)
  (num_investment_bankers num_clients : ℕ)
  (total_individuals := num_investment_bankers + num_clients)
  (meal_cost_before_gratuity : ℝ := total_bill / (1 + gratuity_percent))
  (average_cost := meal_cost_before_gratuity / total_individuals) :
  total_bill = 1350 ∧ gratuity_percent = 0.25 ∧ num_investment_bankers = 7 ∧ num_clients = 8 →
  average_cost = 72 := by
  sorry

end avg_meal_cost_per_individual_is_72_l267_267431


namespace average_homework_time_decrease_l267_267397

theorem average_homework_time_decrease (x : ℝ) :
  (100 * (1 - x)^2 = 70) :=
sorry

end average_homework_time_decrease_l267_267397


namespace grade_assignment_ways_l267_267244

theorem grade_assignment_ways : (4 ^ 12) = 16777216 :=
by
  -- mathematical proof
  sorry

end grade_assignment_ways_l267_267244


namespace problem_1_problem_2_l267_267593

-- Definitions required for the proof
variables {A B C : ℝ} (a b c : ℝ)
variable (cos_A cos_B cos_C : ℝ)
variables (sin_A sin_C : ℝ)

-- Given conditions
axiom given_condition : (cos_A - 2 * cos_C) / cos_B = (2 * c - a) / b
axiom cos_B_eq : cos_B = 1 / 4
axiom b_eq : b = 2

-- First problem: Proving the value of sin_C / sin_A
theorem problem_1 :
  (cos_A - 2 * cos_C) / cos_B = (2 * c - a) / b → (sin_C / sin_A) = 2 :=
by
  intro h
  sorry

-- Second problem: Proving the area of triangle ABC
theorem problem_2 :
  (cos_B = 1 / 4) → (b = 2) → ((cos_A - 2 * cos_C) / cos_B = (2 * c - a) / b) → (1 / 2 * a * c * sin_A) = (Real.sqrt 15) / 4 :=
by
  intros h1 h2 h3
  sorry

end problem_1_problem_2_l267_267593


namespace chocolate_bars_sold_last_week_l267_267756

-- Definitions based on conditions
def initial_chocolate_bars : Nat := 18
def chocolate_bars_sold_this_week : Nat := 7
def chocolate_bars_needed_to_sell : Nat := 6

-- Define the number of chocolate bars sold so far
def chocolate_bars_sold_so_far : Nat := chocolate_bars_sold_this_week + chocolate_bars_needed_to_sell

-- Target statement to prove
theorem chocolate_bars_sold_last_week :
  initial_chocolate_bars - chocolate_bars_sold_so_far = 5 :=
by
  sorry

end chocolate_bars_sold_last_week_l267_267756


namespace symmetric_circle_eq_l267_267121

/-- The definition of the original circle equation. -/
def original_circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

/-- The definition of the line of symmetry equation. -/
def line_eq (x y : ℝ) : Prop := x - y - 2 = 0

/-- The statement that the equation of the circle that is symmetric to the original circle 
    about the given line is (x - 4)^2 + (y + 1)^2 = 1. -/
theorem symmetric_circle_eq : 
  (∃ x y : ℝ, original_circle_eq x y ∧ line_eq x y) →
  (∀ x y : ℝ, (x - 4)^2 + (y + 1)^2 = 1) :=
by sorry

end symmetric_circle_eq_l267_267121


namespace sum_of_circular_integers_l267_267806

theorem sum_of_circular_integers (a : Fin 10 → ℕ) (h : ∀ i, a i = Nat.gcd (a ((i - 1) % 10)) (a ((i + 1) % 10)) + 1) :
    (Finset.univ.sum (λ i, a i)) = 28 := by
  sorry

end sum_of_circular_integers_l267_267806


namespace expected_rolls_sum_2010_l267_267429

noncomputable def expected_number_of_rolls (n : ℕ) : ℝ :=
  if n = 0 then 0
  else if n ≤ 6 then (n + 5) / 6
  else (1 + (∑ k in finset.range 6, p_k * expected_number_of_rolls (n - k + 1)) / p_n)
  where 
    p_k := (1 : ℝ) / (6 : ℝ)
    p_n := (1 / (6 : ℝ) ^ (n / 6))

theorem expected_rolls_sum_2010 : expected_number_of_rolls 2010 ≈ 574.5238095 := 
  sorry

end expected_rolls_sum_2010_l267_267429


namespace travel_paths_l267_267190

-- Definitions for conditions
def roads_AB : ℕ := 3
def roads_BC : ℕ := 2

-- The theorem statement
theorem travel_paths : roads_AB * roads_BC = 6 := by
  sorry

end travel_paths_l267_267190


namespace chocolate_milk_probability_l267_267359

noncomputable theory

/--
  Robert visits the milk bottling plant for 7 days a week.
  The plant has a 1/2 chance of bottling chocolate milk on weekdays (Monday to Friday).
  The plant has a 3/4 chance of bottling chocolate milk on weekends (Saturday and Sunday).

  Prove that the probability that the plant bottles chocolate milk on exactly 5 of the 7 days Robert visits is 781/1024.
-/
theorem chocolate_milk_probability :
  let weekdays_prob := 1 / 2,
      weekends_prob := 3 / 4,
      total_days := 7,
      target_days := 5 in
  ∃ p : ℚ, p = 781 / 1024 ∧
    let events := (finset.powerset (finset.range 7)) in
    p = ∑ x in events, if x.card = target_days then
        let weekdays_count := (x ∩ (finset.range 5)).card,
            weekends_count := (x ∩ (finset.range 5).compl).card in
        if weekdays_count + weekends_count = target_days then
          (weekdays_prob ^ weekdays_count) * ((1 - weekdays_prob) ^ (5 - weekdays_count)) *
          (weekends_prob ^ weekends_count) * ((1 - weekends_prob) ^ (2 - weekends_count))
        else 0
    else 0 := by sorry

end chocolate_milk_probability_l267_267359


namespace phone_numbers_even_phone_numbers_odd_phone_numbers_ratio_l267_267677

def even_digits : Set ℕ := { 0, 2, 4, 6, 8 }
def odd_digits : Set ℕ := { 1, 3, 5, 7, 9 }

theorem phone_numbers_even : (4 * 5^6) = 62500 := by
  sorry

theorem phone_numbers_odd : 5^7 = 78125 := by
  sorry

theorem phone_numbers_ratio
  (evens : (4 * 5^6) = 62500)
  (odds : 5^7 = 78125) :
  (78125 / 62500 : ℝ) = 1.25 := by
    sorry

end phone_numbers_even_phone_numbers_odd_phone_numbers_ratio_l267_267677


namespace average_homework_time_decrease_l267_267405

theorem average_homework_time_decrease
  (initial_time final_time : ℝ)
  (rate_of_decrease : ℝ)
  (h1 : initial_time = 100)
  (h2 : final_time = 70) :
  initial_time * (1 - rate_of_decrease)^2 = final_time :=
by
  rw [h1, h2]
  sorry

end average_homework_time_decrease_l267_267405


namespace total_cost_eq_l267_267071

noncomputable def total_cost : Real :=
  let os_overhead := 1.07
  let cost_per_millisecond := 0.023
  let tape_mounting_cost := 5.35
  let cost_per_megabyte := 0.15
  let cost_per_kwh := 0.02
  let technician_rate_per_hour := 50.0
  let minutes_to_milliseconds := 60000
  let gb_to_mb := 1024

  -- Define program specifics
  let computer_time_minutes := 45.0
  let memory_gb := 3.5
  let electricity_kwh := 2.0
  let technician_time_minutes := 20.0

  -- Calculate costs
  let computer_time_cost := (computer_time_minutes * minutes_to_milliseconds * cost_per_millisecond)
  let memory_cost := (memory_gb * gb_to_mb * cost_per_megabyte)
  let electricity_cost := (electricity_kwh * cost_per_kwh)
  let technician_time_total_hours := (technician_time_minutes * 2 / 60.0)
  let technician_cost := (technician_time_total_hours * technician_rate_per_hour)

  os_overhead + computer_time_cost + tape_mounting_cost + memory_cost + electricity_cost + technician_cost

theorem total_cost_eq : total_cost = 62677.39 := by
  sorry

end total_cost_eq_l267_267071


namespace solve_x_l267_267181

theorem solve_x (x : ℝ) (h : 0.05 * x + 0.12 * (30 + x) = 15.84) : x = 72 := 
by
  sorry

end solve_x_l267_267181


namespace sum_of_solutions_l267_267063

theorem sum_of_solutions (x : ℝ) : 
  (∃ x : ℝ, x^2 - 7 * x + 2 = 16) → (complex.sum (λ x : ℝ, x^2 - 7 * x - 14)) = 7 := sorry

end sum_of_solutions_l267_267063


namespace integer_solutions_are_zero_l267_267261

-- Definitions for integers and the given equation
def satisfies_equation (a b : ℤ) : Prop :=
  a^2 * b^2 = a^2 + b^2

-- The main statement to prove
theorem integer_solutions_are_zero :
  ∀ (a b : ℤ), satisfies_equation a b → (a = 0 ∧ b = 0) :=
sorry

end integer_solutions_are_zero_l267_267261


namespace smallest_three_digit_multiple_of_17_l267_267883

theorem smallest_three_digit_multiple_of_17 : 
    ∃ (n : ℤ), n ≥ 100 ∧ n < 1000 ∧ (∃ (k : ℤ), n = 17 * k) ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l267_267883


namespace smaller_area_l267_267240

theorem smaller_area (A B : ℝ) (total_area : A + B = 1800) (diff_condition : B - A = (A + B) / 6) :
  A = 750 := 
by
  sorry

end smaller_area_l267_267240


namespace total_amount_received_l267_267665
noncomputable section

variables (B : ℕ) (H1 : (1 / 3 : ℝ) * B = 50)
theorem total_amount_received (H2 : (2 / 3 : ℝ) * B = 100) (H3 : ∀ (x : ℕ), x = 5): 
  100 * 5 = 500 := 
by
  sorry

end total_amount_received_l267_267665


namespace not_necessarily_prime_sum_l267_267248

theorem not_necessarily_prime_sum (nat_ordered_sequence : ℕ → ℕ) :
  (∀ n1 n2 n3 : ℕ, n1 < n2 → n2 < n3 → nat_ordered_sequence n1 + nat_ordered_sequence n2 + nat_ordered_sequence n3 ≠ prime) :=
sorry

end not_necessarily_prime_sum_l267_267248


namespace find_number_l267_267227

theorem find_number (x : ℝ) (h : (1/4) * x = (1/5) * (x + 1) + 1) : x = 24 := 
sorry

end find_number_l267_267227


namespace smallest_three_digit_multiple_of_17_correct_l267_267921

def smallest_three_digit_multiple_of_17 : ℕ :=
  102

theorem smallest_three_digit_multiple_of_17_correct :
  ∀ n, n >= 100 → n < 1000 → n % 17 = 0 → 102 ≤ n :=
begin
  intros n h1 h2 h3,
  sorry,
end

end smallest_three_digit_multiple_of_17_correct_l267_267921


namespace smallest_repeating_block_fraction_3_over_11_l267_267474

theorem smallest_repeating_block_fraction_3_over_11 :
  ∃ n : ℕ, (∃ d : ℕ, (3/11 : ℚ) = d / 10^n) ∧ n = 2 :=
sorry

end smallest_repeating_block_fraction_3_over_11_l267_267474


namespace scientific_notation_of_35000000_l267_267094

theorem scientific_notation_of_35000000 :
  (35_000_000 : ℕ) = 3.5 * 10^7 := by
  sorry

end scientific_notation_of_35000000_l267_267094


namespace sequence_value_G_50_l267_267035

theorem sequence_value_G_50 :
  ∀ G : ℕ → ℚ, (∀ n : ℕ, G (n + 1) = (3 * G n + 1) / 3) ∧ G 1 = 3 → G 50 = 152 / 3 :=
by
  intros
  sorry

end sequence_value_G_50_l267_267035


namespace weight_lifting_requirement_l267_267183

-- Definitions based on conditions
def weight_25 : Int := 25
def weight_10 : Int := 10
def lifts_25 := 16
def total_weight_25 := 2 * weight_25 * lifts_25

def n_lifts_10 (n : Int) := 2 * weight_10 * n

-- Problem statement and theorem to prove
theorem weight_lifting_requirement (n : Int) : n_lifts_10 n = total_weight_25 ↔ n = 40 := by
  sorry

end weight_lifting_requirement_l267_267183


namespace smallest_three_digit_multiple_of_17_l267_267875

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := sorry

end smallest_three_digit_multiple_of_17_l267_267875


namespace distance_between_intersections_l267_267464

open Classical
open Real

noncomputable def curve1 (x y : ℝ) : Prop := y^2 = x
noncomputable def curve2 (x y : ℝ) : Prop := x + 2 * y = 10

theorem distance_between_intersections :
  ∃ (p1 p2 : ℝ × ℝ),
    (curve1 p1.1 p1.2) ∧ (curve2 p1.1 p1.2) ∧
    (curve1 p2.1 p2.2) ∧ (curve2 p2.1 p2.2) ∧
    (dist p1 p2 = 2 * sqrt 55) :=
by
  sorry

end distance_between_intersections_l267_267464


namespace cos_arcsin_of_fraction_l267_267999

theorem cos_arcsin_of_fraction : ∀ x, x = 8 / 17 → x ∈ set.Icc (-1:ℝ) 1 → Real.cos (Real.arcsin x) = 15 / 17 :=
by
  intros x hx h_range
  rw hx
  have h : (x:ℝ)^2 + Real.cos (Real.arcsin x)^2 = 1 := Real.sin_sq_add_cos_sq (Real.arcsin x)
  sorry

end cos_arcsin_of_fraction_l267_267999


namespace smallest_three_digit_multiple_of_17_l267_267869

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m := by
  use 102
  sorry

end smallest_three_digit_multiple_of_17_l267_267869


namespace triangle_area_l267_267699

theorem triangle_area :
  let A := (-3, 0)
  let B := (0, 2)
  let O := (0, 0)
  let area := 1 / 2 * |A.1 * (B.2 - O.2) + B.1 * (O.2 - A.2) + O.1 * (A.2 - B.2)|
  area = 3 := by
  let A := (-3, 0)
  let B := (0, 2)
  let O := (0, 0)
  let area := 1 / 2 * |A.1 * (B.2 - O.2) + B.1 * (O.2 - A.2) + O.1 * (A.2 - B.2)|
  sorry

end triangle_area_l267_267699


namespace tan_diff_sin_double_l267_267141

theorem tan_diff (α : ℝ) (h : Real.tan α = 2) : 
  Real.tan (α - Real.pi / 4) = 1 / 3 := 
by 
  sorry

theorem sin_double (α : ℝ) (h : Real.tan α = 2) : 
  Real.sin (2 * α) = 4 / 5 := 
by 
  sorry

end tan_diff_sin_double_l267_267141


namespace miles_driven_l267_267354

def rental_fee : ℝ := 20.99
def charge_per_mile : ℝ := 0.25
def total_amount_paid : ℝ := 95.74

theorem miles_driven (miles_driven: ℝ) : 
  (total_amount_paid - rental_fee) / charge_per_mile = miles_driven → miles_driven = 299 := by
  intros
  sorry

end miles_driven_l267_267354


namespace tiling_possible_l267_267664

theorem tiling_possible (n x : ℕ) (hx : 7 * x = n^2) : ∃ k : ℕ, n = 7 * k :=
by sorry

end tiling_possible_l267_267664


namespace solution_set_empty_range_l267_267048

theorem solution_set_empty_range (a : ℝ) : 
  (∀ x : ℝ, ax^2 + ax + 3 < 0 → false) ↔ (0 ≤ a ∧ a ≤ 12) := 
sorry

end solution_set_empty_range_l267_267048


namespace quadratic_roots_equation_l267_267717

theorem quadratic_roots_equation (a b c r s : ℝ)
    (h1 : a ≠ 0)
    (h2 : a * r^2 + b * r + c = 0)
    (h3 : a * s^2 + b * s + c = 0) :
    ∃ p q : ℝ, (x^2 - b * x + a * c = 0) ∧ (ar + b, as + b) = (p, q) :=
by
  sorry

end quadratic_roots_equation_l267_267717


namespace ratio_a_to_b_l267_267591

theorem ratio_a_to_b (a b : ℝ) (h : (a - 3 * b) / (2 * a - b) = 0.14285714285714285) : a / b = 4 :=
by 
  -- The proof goes here
  sorry

end ratio_a_to_b_l267_267591


namespace quadratic_equation_factored_form_l267_267096

theorem quadratic_equation_factored_form : 
  ∀ x : ℝ, x^2 - 6 * x - 6 = 0 ↔ (x - 3)^2 = 15 := 
by 
  sorry

end quadratic_equation_factored_form_l267_267096


namespace no_positive_real_solution_l267_267698

open Real

theorem no_positive_real_solution (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) :
  ¬(∀ n : ℕ, 0 < n → (n - 2) / a ≤ ⌊b * n⌋ ∧ ⌊b * n⌋ < (n - 1) / a) :=
by sorry

end no_positive_real_solution_l267_267698


namespace snow_at_least_once_probability_l267_267175

def P_snow_first_two_days : ℚ := 1 / 2
def P_no_snow_first_two_days : ℚ := 1 - P_snow_first_two_days
def P_snow_next_four_days_if_snow_first_two_days : ℚ := 1 / 3
def P_no_snow_next_four_days_if_snow_first_two_days : ℚ := 1 - P_snow_next_four_days_if_snow_first_two_days
def P_snow_next_four_days_if_no_snow_first_two_days : ℚ := 1 / 4
def P_no_snow_next_four_days_if_no_snow_first_two_days : ℚ := 1 - P_snow_next_four_days_if_no_snow_first_two_days

def P_no_snow_first_two_days_total : ℚ := P_no_snow_first_two_days ^ 2
def P_no_snow_next_four_days_given_no_snow_first_two_days : ℚ := P_no_snow_next_four_days_if_no_snow_first_two_days ^ 4
def P_no_snow_next_four_days_given_snow_first_two_days : ℚ := P_no_snow_next_four_days_if_snow_first_two_days ^ 4

def P_no_snow_all_days : ℚ := 
  P_no_snow_first_two_days_total * P_no_snow_next_four_days_given_no_snow_first_two_days +
  (1 - P_no_snow_first_two_days_total) * P_no_snow_next_four_days_given_snow_first_two_days

def P_snow_at_least_once : ℚ := 1 - P_no_snow_all_days

theorem snow_at_least_once_probability : P_snow_at_least_once = 29 / 32 :=
by
  -- sorry to indicate that the proof is skipped
  sorry

end snow_at_least_once_probability_l267_267175


namespace most_lines_of_symmetry_l267_267414

def regular_pentagon_lines_of_symmetry : ℕ := 5
def kite_lines_of_symmetry : ℕ := 1
def regular_hexagon_lines_of_symmetry : ℕ := 6
def isosceles_triangle_lines_of_symmetry : ℕ := 1
def scalene_triangle_lines_of_symmetry : ℕ := 0

theorem most_lines_of_symmetry :
  regular_hexagon_lines_of_symmetry = max
    (max (max (max regular_pentagon_lines_of_symmetry kite_lines_of_symmetry)
              regular_hexagon_lines_of_symmetry)
        isosceles_triangle_lines_of_symmetry)
    scalene_triangle_lines_of_symmetry :=
sorry

end most_lines_of_symmetry_l267_267414


namespace problem_statements_l267_267272

noncomputable def f (x : ℝ) := (Real.exp x - Real.exp (-x)) / 2

noncomputable def g (x : ℝ) := (Real.exp x + Real.exp (-x)) / 2

theorem problem_statements (x : ℝ) :
  (f x < g x) ∧
  ((f x)^2 + (g x)^2 ≥ 1) ∧
  (f (2 * x) = 2 * f x * g x) :=
by
  sorry

end problem_statements_l267_267272


namespace possible_k_values_l267_267075

theorem possible_k_values :
  (∃ k b a c : ℤ, b = 2020 + k ∧ a * (c ^ 2) = (2020 + k) ∧ 
  (k = -404 ∨ k = -1010)) :=
sorry

end possible_k_values_l267_267075


namespace evaluate_expression_l267_267438

theorem evaluate_expression : 1 + 1 / (2 + 1 / (3 + 2)) = 16 / 11 := 
by 
  sorry

end evaluate_expression_l267_267438


namespace geometric_sequence_sum_l267_267303

/-- 
In a geometric sequence of real numbers, the sum of the first 2 terms is 15,
and the sum of the first 6 terms is 195. Prove that the sum of the first 4 terms is 82.
-/
theorem geometric_sequence_sum :
  ∃ (a r : ℝ), (a + a * r = 15) ∧ (a * (1 - r^6) / (1 - r) = 195) ∧ (a * (1 + r + r^2 + r^3) = 82) :=
by
  sorry

end geometric_sequence_sum_l267_267303


namespace system_solution_l267_267803

theorem system_solution (x y : ℝ) 
  (h1 : (x^2 + x * y + y^2) / (x^2 - x * y + y^2) = 3) 
  (h2 : x^3 + y^3 = 2) : x = 1 ∧ y = 1 :=
  sorry

end system_solution_l267_267803


namespace find_legs_of_right_triangle_l267_267132

theorem find_legs_of_right_triangle (x y a Δ : ℝ) 
  (h1 : x^2 + y^2 = a^2) 
  (h2 : 2 * Δ = x * y) : 
  x = (Real.sqrt (a^2 + 4 * Δ) + Real.sqrt (a^2 - 4 * Δ)) / 2 ∧ 
  y = (Real.sqrt (a^2 + 4 * Δ) - Real.sqrt (a^2 - 4 * Δ)) / 2 :=
sorry

end find_legs_of_right_triangle_l267_267132


namespace maximize_S_n_l267_267514

-- Define the general term of the sequence and the sum of the first n terms.
def a_n (n : ℕ) : ℤ := -2 * n + 25

def S_n (n : ℕ) : ℤ := 24 * n - n^2

-- The main statement to prove
theorem maximize_S_n : ∃ (n : ℕ), n = 11 ∧ ∀ m, S_n m ≤ S_n 11 :=
  sorry

end maximize_S_n_l267_267514


namespace find_k_l267_267010

theorem find_k (k : ℝ) : (∃ x y : ℝ, y = -2 * x + 4 ∧ y = k * x ∧ y = x + 2) → k = 4 :=
by
  sorry

end find_k_l267_267010


namespace possible_pairs_copies_each_key_min_drawers_l267_267670

-- Define the number of distinct keys
def num_keys : ℕ := 10

-- Define the function to calculate the number of pairs
def num_pairs (n : ℕ) := n * (n - 1) / 2

-- Theorem for the first question
theorem possible_pairs : num_pairs num_keys = 45 :=
by sorry

-- Define the number of copies needed for each key
def copies_needed (n : ℕ) := n - 1

-- Theorem for the second question
theorem copies_each_key : copies_needed num_keys = 9 :=
by sorry

-- Define the minimum number of drawers Fernando needs to open
def min_drawers_to_open (n : ℕ) := num_pairs n - (n - 1) + 1

-- Theorem for the third question
theorem min_drawers : min_drawers_to_open num_keys = 37 :=
by sorry

end possible_pairs_copies_each_key_min_drawers_l267_267670


namespace john_worked_period_l267_267161

theorem john_worked_period (A : ℝ) (n : ℕ) (h1 : 6 * A = 1 / 2 * (6 * A + n * A)) : n + 1 = 7 :=
by
  sorry

end john_worked_period_l267_267161


namespace certain_event_l267_267973

-- Definitions of the events as propositions
def EventA : Prop := ∃ n : ℕ, n ≥ 1 ∧ (n % 2 = 0)
def EventB : Prop := ∃ t : ℝ, t ≥ 0  -- Simplifying as the event of an advertisement airing
def EventC : Prop := ∃ w : ℕ, w ≥ 1  -- Simplifying as the event of rain in Weinan on a specific future date
def EventD : Prop := true  -- The sun rises from the east in the morning is always true

-- The statement that Event D is the only certain event among the given options
theorem certain_event : EventD ∧ ¬EventA ∧ ¬EventB ∧ ¬EventC :=
by
  sorry

end certain_event_l267_267973


namespace net_pay_rate_l267_267427

def travelTime := 3 -- hours
def speed := 50 -- miles per hour
def fuelEfficiency := 25 -- miles per gallon
def earningsRate := 0.6 -- dollars per mile
def gasolineCost := 3 -- dollars per gallon

theorem net_pay_rate
  (travelTime : ℕ)
  (speed : ℕ)
  (fuelEfficiency : ℕ)
  (earningsRate : ℚ)
  (gasolineCost : ℚ)
  (h_time : travelTime = 3)
  (h_speed : speed = 50)
  (h_fuelEfficiency : fuelEfficiency = 25)
  (h_earningsRate : earningsRate = 0.6)
  (h_gasolineCost : gasolineCost = 3) :
  (earningsRate * speed * travelTime - (speed * travelTime / fuelEfficiency) * gasolineCost) / travelTime = 24 :=
by
  sorry

end net_pay_rate_l267_267427


namespace no_injective_function_satisfying_conditions_l267_267599

open Real

theorem no_injective_function_satisfying_conditions :
  ¬ ∃ (f : ℝ → ℝ), (∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 ≠ f x2)
  ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (∀ x : ℝ, f (x ^ 2) - (f (a * x + b)) ^ 2 ≥ 1 / 4) :=
by
  sorry

end no_injective_function_satisfying_conditions_l267_267599


namespace mr_green_expected_produce_l267_267356

noncomputable def total_produce_yield (steps_length : ℕ) (steps_width : ℕ) (step_length : ℝ)
                                      (yield_carrots : ℝ) (yield_potatoes : ℝ): ℝ :=
  let length_feet := steps_length * step_length
  let width_feet := steps_width * step_length
  let area := length_feet * width_feet
  let yield_carrots_total := area * yield_carrots
  let yield_potatoes_total := area * yield_potatoes
  yield_carrots_total + yield_potatoes_total

theorem mr_green_expected_produce:
  total_produce_yield 18 25 3 0.4 0.5 = 3645 := by
  sorry

end mr_green_expected_produce_l267_267356


namespace scientific_notation_of_3930_billion_l267_267371

theorem scientific_notation_of_3930_billion :
  (3930 * 10^9) = 3.93 * 10^12 :=
sorry

end scientific_notation_of_3930_billion_l267_267371


namespace CarmenBrushLengthIsCorrect_l267_267107

namespace BrushLength

def carlasBrushLengthInInches : ℤ := 12
def conversionRateInCmPerInch : ℝ := 2.5
def lengthMultiplier : ℝ := 1.5

def carmensBrushLengthInCm : ℝ :=
  carlasBrushLengthInInches * lengthMultiplier * conversionRateInCmPerInch

theorem CarmenBrushLengthIsCorrect :
  carmensBrushLengthInCm = 45 := by
  sorry

end BrushLength

end CarmenBrushLengthIsCorrect_l267_267107


namespace age_of_B_l267_267977

/--
A is two years older than B.
B is twice as old as C.
The total of the ages of A, B, and C is 32.
How old is B?
-/
theorem age_of_B (A B C : ℕ) (h1 : A = B + 2) (h2 : B = 2 * C) (h3 : A + B + C = 32) : B = 12 :=
by
  sorry

end age_of_B_l267_267977


namespace hypotenuse_length_l267_267129

-- Define the properties of the right-angled triangle
variables (α β γ : ℝ) (a b c : ℝ)
-- Right-angled triangle condition
axiom right_angled_triangle : α = 30 ∧ β = 60 ∧ γ = 90 → c = 2 * a

-- Given side opposite 30° angle is 6 cm
axiom side_opposite_30_is_6cm : a = 6

-- Proof that hypotenuse is 12 cm
theorem hypotenuse_length : c = 12 :=
by 
  sorry

end hypotenuse_length_l267_267129


namespace part1_part2_l267_267345

variable {A B C a b c : ℝ}

-- Part (1): Prove that 2a^2 = b^2 + c^2 given the condition
theorem part1 (h : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A)) : 2 * a^2 = b^2 + c^2 := 
sorry

-- Part (2): Prove the perimeter of triangle ABC
theorem part2 (a : ℝ) (h_a : a = 5) (cosA : ℝ) (h_cosA : cosA = 25 / 31) : 5 + b + c = 14 := 
sorry

end part1_part2_l267_267345


namespace simplify_and_evaluate_expression_l267_267788

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = Real.tan (Real.pi / 3) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2*m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l267_267788


namespace floor_sqrt_n_sqrt_n_plus1_eq_floor_sqrt_4n_plus_2_l267_267296

theorem floor_sqrt_n_sqrt_n_plus1_eq_floor_sqrt_4n_plus_2 (n : ℕ) (hn : n > 0) :
  Int.floor (Real.sqrt n + Real.sqrt (n + 1)) = Int.floor (Real.sqrt (4 * n + 2)) := 
  sorry

end floor_sqrt_n_sqrt_n_plus1_eq_floor_sqrt_4n_plus_2_l267_267296


namespace triangle_abc_proof_one_triangle_abc_perimeter_l267_267346

-- Defining the basic setup for the triangle and the given condition
variables (a b c A B C : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
variables (hABC : A + B + C = real.pi)
variables (h_sin : real.sin C * real.sin (A - B) = real.sin B * real.sin (C - A))

-- First proof: 2a² = b² + c²
theorem triangle_abc_proof_one : 2 * a^2 = b^2 + c^2 :=
sorry

-- Second proof: the perimeter of the triangle ABC is 14 given a = 5 and cos A = 25/31
variables (ha_val : a = 5) (cosA_val : real.cos A = 25/31)

theorem triangle_abc_perimeter : a + b + c = 14 :=
sorry

end triangle_abc_proof_one_triangle_abc_perimeter_l267_267346


namespace first_alloy_mass_l267_267152

theorem first_alloy_mass (x : ℝ) : 
  (0.12 * x + 2.8) / (x + 35) = 9.454545454545453 / 100 → 
  x = 20 :=
by
  intro h
  sorry

end first_alloy_mass_l267_267152


namespace donor_multiple_l267_267424

def cost_per_box (food_cost : ℕ) (supplies_cost : ℕ) : ℕ := food_cost + supplies_cost

def total_initial_cost (num_boxes : ℕ) (cost_per_box : ℕ) : ℕ := num_boxes * cost_per_box

def additional_boxes (total_boxes : ℕ) (initial_boxes : ℕ) : ℕ := total_boxes - initial_boxes

def donor_contribution (additional_boxes : ℕ) (cost_per_box : ℕ) : ℕ := additional_boxes * cost_per_box

def multiple (donor_contribution : ℕ) (initial_cost : ℕ) : ℕ := donor_contribution / initial_cost

theorem donor_multiple 
    (initial_boxes : ℕ) (box_cost : ℕ) (total_boxes : ℕ) (donor_multi : ℕ)
    (h1 : initial_boxes = 400) 
    (h2 : box_cost = 245) 
    (h3 : total_boxes = 2000)
    : donor_multi = 4 :=
by
    let initial_cost := total_initial_cost initial_boxes box_cost
    let additional_boxes := additional_boxes total_boxes initial_boxes
    let contribution := donor_contribution additional_boxes box_cost
    have h4 : contribution = 392000 := sorry
    have h5 : initial_cost = 98000 := sorry
    have h6 : donor_multi = contribution / initial_cost := sorry
    -- Therefore, the multiple should be 4
    exact sorry

end donor_multiple_l267_267424


namespace stop_signs_per_mile_l267_267173

-- Define the conditions
def miles_traveled := 5 + 2
def stop_signs_encountered := 17 - 3

-- Define the proof statement
theorem stop_signs_per_mile : (stop_signs_encountered / miles_traveled) = 2 := by
  -- Proof goes here
  sorry

end stop_signs_per_mile_l267_267173


namespace smallest_positive_integer_l267_267658

theorem smallest_positive_integer :
  ∃ x : ℤ, 0 < x ∧ (x % 5 = 4) ∧ (x % 7 = 6) ∧ (x % 11 = 10) ∧ x = 384 :=
by
  sorry

end smallest_positive_integer_l267_267658


namespace transformed_function_correct_l267_267457

-- Given function
def f (x : ℝ) : ℝ := 2 * x + 1

-- Main theorem to be proven
theorem transformed_function_correct (x : ℝ) (h : 2 ≤ x ∧ x ≤ 4) : 
  f (x - 1) = 2 * x - 1 :=
by {
  sorry
}

end transformed_function_correct_l267_267457


namespace expected_rolls_to_at_least_3_l267_267174

/-- Expected number of rolls to achieve a sum of at least 3 on a fair six-sided die --/
theorem expected_rolls_to_at_least_3 
  (X : ℕ → Probability measure_on ℝ) 
  (hX : ∀ n, X n = sum (range 1 7) ((λ i, ite i = (1 : ℝ), 1) + (λ i, ite i = (2 : ℝ), 2))) : 
  (∑ i in range 1 7, X i) / 6 = 1.36 := 
sorry

end expected_rolls_to_at_least_3_l267_267174


namespace value_of_x_l267_267144

theorem value_of_x (b x : ℝ) (h₀ : 1 < b) (h₁ : 0 < x) (h₂ : (2 * x) ^ (Real.logb b 2) - (3 * x) ^ (Real.logb b 3) = 0) : x = 1 / 6 :=
by {
  sorry
}

end value_of_x_l267_267144


namespace smallest_three_digit_multiple_of_17_l267_267933

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  use 102,
  split,
  { exact le_of_lt (nat.lt_of_sub_eq_succ rfl), }, -- 102 is greater than or equal to 100
  split,
  { exact nat.lt_of_sub_eq_succ rfl, }, -- 102 is less than 1000
  split,
  { exact nat.dvd_intro_left (6) rfl, }, -- 17 divides 102
  { intros m hm1 hm2 hm3,
    exact nat.le_of_dvd hm1 (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (hm3)))), }, -- 102 is the smallest
end

end smallest_three_digit_multiple_of_17_l267_267933


namespace sum_of_S_values_l267_267489

noncomputable def a : ℕ := 32
noncomputable def b1 : ℕ := 16 -- When M = 73
noncomputable def c : ℕ := 25
noncomputable def b2 : ℕ := 89 -- When M = 146
noncomputable def x1 : ℕ := 14 -- When M = 73
noncomputable def x2 : ℕ := 7 -- When M = 146
noncomputable def y1 : ℕ := 3 -- When M = 73
noncomputable def y2 : ℕ := 54 -- When M = 146
noncomputable def z1 : ℕ := 8 -- When M = 73
noncomputable def z2 : ℕ := 4 -- When M = 146

theorem sum_of_S_values :
  let M1 := a + b1 + c
  let M2 := a + b2 + c
  let S1 := M1 + x1 + y1 + z1
  let S2 := M2 + x2 + y2 + z2
  (S1 = 98) ∧ (S2 = 211) ∧ (S1 + S2 = 309) := by
  sorry

end sum_of_S_values_l267_267489


namespace find_utilities_second_l267_267545

def rent_first : ℝ := 800
def utilities_first : ℝ := 260
def distance_first : ℕ := 31
def rent_second : ℝ := 900
def distance_second : ℕ := 21
def cost_per_mile : ℝ := 0.58
def days_per_month : ℕ := 20
def cost_difference : ℝ := 76

-- Helper definitions
def driving_cost (distance : ℕ) : ℝ :=
  distance * days_per_month * cost_per_mile

def total_cost_first : ℝ :=
  rent_first + utilities_first + driving_cost distance_first

def total_cost_second_no_utilities : ℝ :=
  rent_second + driving_cost distance_second

theorem find_utilities_second :
  ∃ (utilities_second : ℝ),
  total_cost_first - total_cost_second_no_utilities = cost_difference →
  utilities_second = 200 :=
sorry

end find_utilities_second_l267_267545


namespace smallest_three_digit_multiple_of_17_l267_267854

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → n ≤ m) :=
by
  use 102
  have h1 : 100 ≤ 102 := by linarith
  have h2 : 102 ≤ 999 := by linarith
  have h3 : 102 % 17 = 0 := by norm_num
  have h4 : ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → 102 ≤ m := by
    intro m
    intro h
    have h_m1 := h.1
    have h_m2 := h.2.1
    have h_m3 := h.2.2
    have h_min := nat.le_of_dvd (by linarith) h_m3
    exact h_min
  exact ⟨⟨h1, h2⟩, h3, h4⟩

end smallest_three_digit_multiple_of_17_l267_267854


namespace value_of_s_for_g_neg_1_eq_0_l267_267763

def g (x s : ℝ) := 3 * x^5 - 2 * x^3 + x^2 - 4 * x + s

theorem value_of_s_for_g_neg_1_eq_0 (s : ℝ) : g (-1) s = 0 ↔ s = -4 :=
by
  sorry

end value_of_s_for_g_neg_1_eq_0_l267_267763


namespace consecutive_numbers_sum_39_l267_267743

theorem consecutive_numbers_sum_39 (n : ℕ) (hn : n + (n + 1) = 39) : n + 1 = 20 :=
sorry

end consecutive_numbers_sum_39_l267_267743


namespace machine_value_depletion_rate_l267_267544

theorem machine_value_depletion_rate :
  ∃ r : ℝ, 700 * (1 - r)^2 = 567 ∧ r = 0.1 := 
by
  sorry

end machine_value_depletion_rate_l267_267544


namespace both_solve_prob_l267_267299

variable (a b : ℝ) -- Define a and b as real numbers

-- Define the conditions
def not_solve_prob_A := (0 ≤ a) ∧ (a ≤ 1)
def not_solve_prob_B := (0 ≤ b) ∧ (b ≤ 1)
def independent := true -- independence is implicit by the question

-- Define the statement of the proof
theorem both_solve_prob (h1 : not_solve_prob_A a) (h2 : not_solve_prob_B b) :
  (1 - a) * (1 - b) = (1 - a) * (1 - b) :=
by sorry

end both_solve_prob_l267_267299


namespace smallest_perimeter_consecutive_even_triangle_l267_267217

theorem smallest_perimeter_consecutive_even_triangle :
  ∃ (x : ℕ), x % 2 = 0 ∧ 
             x + 2 > 2 ∧ 
             x + 4 > 2 ∧ 
             x > 2 ∧ 
             (let sides := [x, x + 2, x + 4] in 
                (sides.sum) = 18) :=
by
  sorry

end smallest_perimeter_consecutive_even_triangle_l267_267217


namespace differences_repeated_at_least_four_l267_267575

theorem differences_repeated_at_least_four (a : Fin 20 → ℕ)
  (h1 : ∀ i j, i < j → a i < a j)
  (h2 : ∀ i, a i ≤ 70) :
  ∃ d, Finset.card (Finset.filter (λ x, x = d) { d | ∃ i j, i < j ∧ d = a j - a i }) ≥ 4 :=
by
  sorry

end differences_repeated_at_least_four_l267_267575


namespace coefficient_x18_is_zero_coefficient_x17_is_3420_l267_267695

open Polynomial

noncomputable def P : Polynomial ℚ := (1 + X^5 + X^7)^20

theorem coefficient_x18_is_zero : coeff P 18 = 0 :=
sorry

theorem coefficient_x17_is_3420 : coeff P 17 = 3420 :=
sorry

end coefficient_x18_is_zero_coefficient_x17_is_3420_l267_267695


namespace sum_of_roots_l267_267062

-- Given condition: the equation to be solved
def equation (x : ℝ) : Prop :=
  x^2 - 7 * x + 2 = 16

-- Define the proof problem: the sum of the values of x that satisfy the equation
theorem sum_of_roots : 
  (∀ x : ℝ, equation x) → x^2 - 7*x - 14 = 0 → (root : ℝ) → (root₀ + root₁) = 7 :=
by
  sorry

end sum_of_roots_l267_267062


namespace intersection_of_A_and_B_l267_267289

-- Definitions based on conditions
def set_A : Set ℝ := {x | x ≥ 3 ∨ x ≤ -1}
def set_B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Statement of the proof problem
theorem intersection_of_A_and_B : set_A ∩ set_B = {x | -2 ≤ x ∧ x ≤ -1} :=
  sorry

end intersection_of_A_and_B_l267_267289


namespace solution_volume_l267_267521

theorem solution_volume (concentration volume_acid volume_solution : ℝ) 
  (h_concentration : concentration = 0.25) 
  (h_acid : volume_acid = 2.5) 
  (h_formula : concentration = volume_acid / volume_solution) : 
  volume_solution = 10 := 
by
  sorry

end solution_volume_l267_267521


namespace smallest_three_digit_multiple_of_17_l267_267961

theorem smallest_three_digit_multiple_of_17 : ∃ (k : ℕ), 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 :=
by
  use 6
  split
  · exact of_as_true trivial -- 100 <= 17 * k
  split
  · exact of_as_true trivial -- 17 * k <= 999
  · exact of_as_true trivial -- 17 * k = 102
  sorry

end smallest_three_digit_multiple_of_17_l267_267961


namespace coordinates_of_B_l267_267753

theorem coordinates_of_B (A B : ℝ × ℝ) (h1 : A = (-2, 3)) (h2 : (A.1 = B.1 ∨ A.1 + 1 = B.1 ∨ A.1 - 1 = B.1)) (h3 : A.2 = B.2) : 
  B = (-1, 3) ∨ B = (-3, 3) := 
sorry

end coordinates_of_B_l267_267753


namespace sphere_diameter_l267_267089

theorem sphere_diameter 
  (shadow_sphere : ℝ)
  (height_pole : ℝ)
  (shadow_pole : ℝ)
  (parallel_rays : Prop)
  (vertical_objects : Prop)
  (tan_theta : ℝ) :
  shadow_sphere = 12 →
  height_pole = 1.5 →
  shadow_pole = 3 →
  (tan_theta = height_pole / shadow_pole) →
  parallel_rays →
  vertical_objects →
  2 * (shadow_sphere * tan_theta) = 12 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end sphere_diameter_l267_267089


namespace sin_double_angle_l267_267447

open Real

theorem sin_double_angle (a : ℝ) (h1 : cos (5 * π / 2 + a) = 3 / 5) (h2 : -π / 2 < a ∧ a < π / 2) :
  sin (2 * a) = -24 / 25 :=
sorry

end sin_double_angle_l267_267447


namespace krishan_money_l267_267517

theorem krishan_money 
  (x y : ℝ)
  (hx1 : 7 * x * 1.185 = 699.8)
  (hx2 : 10 * x * 0.8 = 800)
  (hy : 17 * x = 8 * y) : 
  16 * y = 3400 := 
by
  -- It's acceptable to leave the proof incomplete due to the focus being on the statement.
  sorry

end krishan_money_l267_267517


namespace smallest_three_digit_multiple_of_17_l267_267966

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ ∃ (k : ℕ), n = 17 * k ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l267_267966


namespace dobarulho_problem_l267_267525

def is_divisible_by (x d : ℕ) : Prop := d ∣ x

def valid_quadruple (A B C D : ℕ) : Prop :=
  (1 ≤ A ∧ A ≤ 9) ∧ (1 ≤ B ∧ B ≤ 9) ∧ (1 ≤ C ∧ C ≤ 9) ∧ (A ≤ 8) ∧ (D > 1) ∧
  is_divisible_by (100 * A + 10 * B + C) D ∧
  is_divisible_by (100 * B + 10 * C + A) D ∧
  is_divisible_by (100 * C + 10 * A + B) D ∧
  is_divisible_by (100 * (A + 1) + 10 * C + B) D ∧
  is_divisible_by (100 * C + 10 * B + (A + 1)) D ∧
  is_divisible_by (100 * B + 10 * (A + 1) + C) D 

theorem dobarulho_problem :
  ∀ (A B C D : ℕ), valid_quadruple A B C D → 
  (A = 3 ∧ B = 7 ∧ C = 0 ∧ D = 37) ∨ 
  (A = 4 ∧ B = 8 ∧ C = 1 ∧ D = 37) ∨
  (A = 5 ∧ B = 9 ∧ C = 2 ∧ D = 37) :=
by sorry

end dobarulho_problem_l267_267525


namespace smallest_three_digit_multiple_of_17_l267_267867

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m := by
  use 102
  sorry

end smallest_three_digit_multiple_of_17_l267_267867


namespace heather_shared_41_blocks_l267_267003

theorem heather_shared_41_blocks :
  ∀ (initial_blocks final_blocks shared_blocks : ℕ),
    initial_blocks = 86 →
    final_blocks = 45 →
    shared_blocks = initial_blocks - final_blocks →
    shared_blocks = 41 :=
by
  intros initial_blocks final_blocks shared_blocks h_initial h_final h_shared
  rw [h_initial, h_final] at h_shared
  exact h_shared.symm

sorry

end heather_shared_41_blocks_l267_267003


namespace caterpillars_and_leaves_l267_267052

def initial_caterpillars : Nat := 14
def caterpillars_after_storm : Nat := initial_caterpillars - 3
def hatched_eggs : Nat := 6
def caterpillars_after_hatching : Nat := caterpillars_after_storm + hatched_eggs
def leaves_eaten_by_babies : Nat := 18
def caterpillars_after_cocooning : Nat := caterpillars_after_hatching - 9
def moth_caterpillars : Nat := caterpillars_after_cocooning / 2
def butterfly_caterpillars : Nat := caterpillars_after_cocooning - moth_caterpillars
def leaves_eaten_per_moth_per_day : Nat := 4
def days_in_week : Nat := 7
def total_leaves_eaten_by_moths : Nat := moth_caterpillars * leaves_eaten_per_moth_per_day * days_in_week
def total_leaves_eaten_by_babies_and_moths : Nat := leaves_eaten_by_babies + total_leaves_eaten_by_moths

theorem caterpillars_and_leaves :
  (caterpillars_after_cocooning = 8) ∧ (total_leaves_eaten_by_babies_and_moths = 130) :=
by
  -- proof to be filled in
  sorry

end caterpillars_and_leaves_l267_267052


namespace double_mean_value_function_range_l267_267691

noncomputable def f (x : ℝ) : ℝ := 2 * x ^ 3 - x ^ 2 + m

theorem double_mean_value_function_range (a : ℝ) (m : ℝ) :
  (1 / 8 : ℝ) < a ∧ a < (1 / 4 : ℝ) ↔
  (∀ x ∈ Icc (0:ℝ) (2*a), 6 * x ^ 2 - 2 * x = 8 * a^2 - 2 * a) ∧
  (f'' (0) = (f(2*a) - f(0)) / (2*a)) :=
begin
  sorry
end

end double_mean_value_function_range_l267_267691


namespace sachin_age_is_49_l267_267072

open Nat

-- Let S be Sachin's age and R be Rahul's age
def Sachin_age : ℕ := 49
def Rahul_age (S : ℕ) := S + 14

theorem sachin_age_is_49 (S R : ℕ) (h1 : R = S + 14) (h2 : S * 9 = R * 7) : S = 49 :=
by sorry

end sachin_age_is_49_l267_267072


namespace solve_inequalities_l267_267365

theorem solve_inequalities {x : ℝ} :
  (3 * x + 1) / 2 > x ∧ (4 * (x - 2) ≤ x - 5) ↔ (-1 < x ∧ x ≤ 1) :=
by sorry

end solve_inequalities_l267_267365


namespace speed_in_still_water_l267_267084

-- Given conditions
def upstream_speed : ℝ := 60
def downstream_speed : ℝ := 90

-- Proof that the speed of the man in still water is 75 kmph
theorem speed_in_still_water :
  (upstream_speed + downstream_speed) / 2 = 75 := 
by
  sorry

end speed_in_still_water_l267_267084


namespace birth_date_16_Jan_1993_l267_267624

noncomputable def year_of_birth (current_date : Nat) (age_years : Nat) :=
  current_date - age_years * 365

noncomputable def month_of_birth (current_date : Nat) (age_years : Nat) (age_months : Nat) :=
  current_date - (age_years * 12 + age_months) * 30

theorem birth_date_16_Jan_1993 :
  let boy_age_years := 10
  let boy_age_months := 1
  let current_date := 16 + 31 * 12 * 2003 -- 16th February 2003 represented in days
  let full_months_lived := boy_age_years * 12 + boy_age_months
  full_months_lived - boy_age_years = 111 → 
  year_of_birth current_date boy_age_years = 1993 ∧ month_of_birth current_date boy_age_years boy_age_months = 31 * 1 * 1993 := 
sorry

end birth_date_16_Jan_1993_l267_267624


namespace difference_longest_shortest_worm_l267_267512

theorem difference_longest_shortest_worm
  (A B C D E : ℝ)
  (hA : A = 0.8)
  (hB : B = 0.1)
  (hC : C = 1.2)
  (hD : D = 0.4)
  (hE : E = 0.7) :
  (max C (max A (max E (max D B))) - min B (min D (min E (min A C)))) = 1.1 :=
by
  sorry

end difference_longest_shortest_worm_l267_267512


namespace simplify_expression_l267_267570

theorem simplify_expression :
  (∃ (x : Real), x = 3 * (Real.sqrt 3 + Real.sqrt 7) / (4 * Real.sqrt (3 + Real.sqrt 5)) ∧ 
    x = Real.sqrt (224 - 22 * Real.sqrt 105) / 8) := sorry

end simplify_expression_l267_267570


namespace rubble_money_left_l267_267785

/-- Rubble has $15 in his pocket. -/
def rubble_initial_amount : ℝ := 15

/-- Each notebook costs $4.00. -/
def notebook_price : ℝ := 4

/-- Each pen costs $1.50. -/
def pen_price : ℝ := 1.5

/-- Rubble needs to buy 2 notebooks. -/
def num_notebooks : ℝ := 2

/-- Rubble needs to buy 2 pens. -/
def num_pens : ℝ := 2

/-- The total cost of the notebooks. -/
def total_notebook_cost : ℝ := num_notebooks * notebook_price

/-- The total cost of the pens. -/
def total_pen_cost : ℝ := num_pens * pen_price

/-- The total amount Rubble spends. -/
def total_spent : ℝ := total_notebook_cost + total_pen_cost

/-- The remaining amount Rubble has after the purchase. -/
def rubble_remaining_amount : ℝ := rubble_initial_amount - total_spent

theorem rubble_money_left :
  rubble_remaining_amount = 4 := 
by
  -- Some necessary steps to complete the proof
  sorry

end rubble_money_left_l267_267785


namespace smallest_three_digit_multiple_of_17_correct_l267_267925

def smallest_three_digit_multiple_of_17 : ℕ :=
  102

theorem smallest_three_digit_multiple_of_17_correct :
  ∀ n, n >= 100 → n < 1000 → n % 17 = 0 → 102 ≤ n :=
begin
  intros n h1 h2 h3,
  sorry,
end

end smallest_three_digit_multiple_of_17_correct_l267_267925


namespace abs_diff_61st_terms_l267_267822

noncomputable def seq_C (n : ℕ) : ℤ := 20 + 15 * (n - 1)
noncomputable def seq_D (n : ℕ) : ℤ := 20 - 15 * (n - 1)

theorem abs_diff_61st_terms :
  |seq_C 61 - seq_D 61| = 1800 := sorry

end abs_diff_61st_terms_l267_267822


namespace cartesian_equation_of_circle_c2_positional_relationship_between_circles_l267_267307
noncomputable def circle_c1 := {p : ℝ × ℝ | (p.1)^2 - 2*p.1 + (p.2)^2 = 0}
noncomputable def circle_c2_polar (theta : ℝ) : ℝ × ℝ := (2 * Real.sin theta * Real.cos theta, 2 * Real.sin theta * Real.sin theta)
noncomputable def circle_c2_cartesian := {p : ℝ × ℝ | (p.1)^2 + (p.2 - 1)^2 = 1}

theorem cartesian_equation_of_circle_c2 :
  ∀ p : ℝ × ℝ, (∃ θ : ℝ, p = circle_c2_polar θ) ↔ p ∈ circle_c2_cartesian :=
by
  sorry

theorem positional_relationship_between_circles :
  ∃ p : ℝ × ℝ, p ∈ circle_c1 ∧ p ∈ circle_c2_cartesian :=
by
  sorry

end cartesian_equation_of_circle_c2_positional_relationship_between_circles_l267_267307


namespace oil_spent_amount_l267_267088

theorem oil_spent_amount :
  ∀ (P R M : ℝ), R = 25 → P = (R / 0.75) → ((M / R) - (M / P) = 5) → M = 500 :=
by
  intros P R M hR hP hOil
  sorry

end oil_spent_amount_l267_267088


namespace smallest_three_digit_multiple_of_17_l267_267891

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m) :=
  sorry

end smallest_three_digit_multiple_of_17_l267_267891


namespace heather_shared_blocks_l267_267004

-- Define the initial number of blocks Heather starts with
def initial_blocks : ℕ := 86

-- Define the final number of blocks Heather ends up with
def final_blocks : ℕ := 45

-- Define the number of blocks Heather shared
def blocks_shared (initial final : ℕ) : ℕ := initial - final

-- Prove that the number of blocks Heather shared is 41
theorem heather_shared_blocks : blocks_shared initial_blocks final_blocks = 41 := by
  -- Proof steps will be added here
  sorry

end heather_shared_blocks_l267_267004


namespace total_numbers_l267_267769

theorem total_numbers (m j c : ℕ) (h1 : m = j + 20) (h2 : j = c - 40) (h3 : c = 80) : m + j + c = 180 := 
by sorry

end total_numbers_l267_267769


namespace max_value_of_linear_combination_l267_267453

theorem max_value_of_linear_combination
  (x y : ℝ)
  (h : x^2 + y^2 = 16 * x + 8 * y + 10) :
  ∃ z, z = 4.58 ∧ (∀ x y, (4 * x + 3 * y) ≤ z ∧ (x^2 + y^2 = 16 * x + 8 * y + 10) → (4 * x + 3 * y) ≤ 4.58) :=
by
  sorry

end max_value_of_linear_combination_l267_267453


namespace smallest_three_digit_multiple_of_17_l267_267932

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  use 102,
  split,
  { exact le_of_lt (nat.lt_of_sub_eq_succ rfl), }, -- 102 is greater than or equal to 100
  split,
  { exact nat.lt_of_sub_eq_succ rfl, }, -- 102 is less than 1000
  split,
  { exact nat.dvd_intro_left (6) rfl, }, -- 17 divides 102
  { intros m hm1 hm2 hm3,
    exact nat.le_of_dvd hm1 (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (hm3)))), }, -- 102 is the smallest
end

end smallest_three_digit_multiple_of_17_l267_267932


namespace probability_same_color_l267_267231

theorem probability_same_color (w b : ℕ) (h_w : w = 8) (h_b : b = 9) :
  (Nat.choose 8 2 + Nat.choose 9 2) / (Nat.choose 17 2) = 8 / 17 :=
by
  sorry

end probability_same_color_l267_267231


namespace nonneg_reals_inequality_l267_267178

theorem nonneg_reals_inequality (a b c d : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d) (h₄ : a^2 + b^2 + c^2 + d^2 = 1) : 
  a + b + c + d - 1 ≥ 16 * a * b * c * d := 
by 
  sorry

end nonneg_reals_inequality_l267_267178


namespace find_x_l267_267737

theorem find_x (x : ℝ) (h : (40 / 100) * x = (25 / 100) * 80) : x = 50 :=
by
  sorry

end find_x_l267_267737


namespace pencils_total_l267_267556

theorem pencils_total (p1 p2 : ℕ) (h1 : p1 = 3) (h2 : p2 = 7) : p1 + p2 = 10 := by
  sorry

end pencils_total_l267_267556


namespace probability_opposite_vertex_l267_267306

theorem probability_opposite_vertex (k : ℕ) (h : k > 0) : 
    P_k = (1 / 6 : ℝ) + (1 / (3 * (-2) ^ k) : ℝ) := 
sorry

end probability_opposite_vertex_l267_267306


namespace part1_part2_l267_267344

variable {A B C a b c : ℝ}

-- Part (1): Prove that 2a^2 = b^2 + c^2 given the condition
theorem part1 (h : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A)) : 2 * a^2 = b^2 + c^2 := 
sorry

-- Part (2): Prove the perimeter of triangle ABC
theorem part2 (a : ℝ) (h_a : a = 5) (cosA : ℝ) (h_cosA : cosA = 25 / 31) : 5 + b + c = 14 := 
sorry

end part1_part2_l267_267344


namespace ratio_of_coeffs_l267_267128

theorem ratio_of_coeffs
  (a b c d e : ℝ) 
  (h_poly : ∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 → x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4) : 
  d / e = 25 / 12 :=
by
  sorry

end ratio_of_coeffs_l267_267128


namespace steve_height_end_second_year_l267_267629

noncomputable def initial_height_ft : ℝ := 5
noncomputable def initial_height_inch : ℝ := 6
noncomputable def inch_to_cm : ℝ := 2.54

noncomputable def initial_height_cm : ℝ :=
  (initial_height_ft * 12 + initial_height_inch) * inch_to_cm

noncomputable def first_growth_spurt : ℝ := 0.15
noncomputable def second_growth_spurt : ℝ := 0.07
noncomputable def height_decrease : ℝ := 0.04

noncomputable def height_after_growths : ℝ :=
  let height_after_first_growth := initial_height_cm * (1 + first_growth_spurt)
  height_after_first_growth * (1 + second_growth_spurt)

noncomputable def final_height_cm : ℝ :=
  height_after_growths * (1 - height_decrease)

theorem steve_height_end_second_year : final_height_cm = 198.03 :=
  sorry

end steve_height_end_second_year_l267_267629


namespace bruce_total_payment_l267_267435

def grapes_quantity : ℕ := 8
def grapes_rate : ℕ := 70
def mangoes_quantity : ℕ := 9
def mangoes_rate : ℕ := 55

def cost_grapes : ℕ := grapes_quantity * grapes_rate
def cost_mangoes : ℕ := mangoes_quantity * mangoes_rate
def total_cost : ℕ := cost_grapes + cost_mangoes

theorem bruce_total_payment : total_cost = 1055 := by
  sorry

end bruce_total_payment_l267_267435


namespace new_difference_greater_l267_267069

theorem new_difference_greater (x y a b : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy : x > y) (ha_pos : 0 < a) (hb_pos : 0 < b) (hab : a ≠ b) :
  (x + a) - (y - b) > x - y :=
by {
  sorry
}

end new_difference_greater_l267_267069


namespace smallest_three_digit_multiple_of_17_l267_267912

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ 17 * n ∧ 17 * n ≤ 999 ∧ 17 * n = 102 :=
begin
  sorry
end

end smallest_three_digit_multiple_of_17_l267_267912


namespace length_of_one_side_of_square_l267_267585

variable (total_ribbon_length : ℕ) (triangle_perimeter : ℕ)

theorem length_of_one_side_of_square (h1 : total_ribbon_length = 78)
                                    (h2 : triangle_perimeter = 46) :
  (total_ribbon_length - triangle_perimeter) / 4 = 8 :=
by
  sorry

end length_of_one_side_of_square_l267_267585


namespace total_distance_driven_l267_267782

def renaldo_distance : ℕ := 15
def ernesto_distance : ℕ := 7 + (renaldo_distance / 3)

theorem total_distance_driven :
  renaldo_distance + ernesto_distance = 27 :=
sorry

end total_distance_driven_l267_267782


namespace polynomial_value_at_minus_1_l267_267660

-- Definitions for the problem conditions
def polynomial_1 (a b : ℤ) (x : ℤ) : ℤ := a * x^3 + b * x + 1
def polynomial_2 (a b : ℤ) (x : ℤ) : ℤ := a * x^3 + b * x - 2

theorem polynomial_value_at_minus_1 :
  ∀ (a b : ℤ), (a + b = 2022) → polynomial_2 a b (-1) = -2024 :=
by
  intro a b h
  sorry

end polynomial_value_at_minus_1_l267_267660


namespace prime_sum_product_l267_267520

theorem prime_sum_product (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h_sum : p + q = 91) : p * q = 178 := 
by
  sorry

end prime_sum_product_l267_267520


namespace unique_real_x_satisfies_eq_l267_267533

theorem unique_real_x_satisfies_eq (x : ℝ) (h : x ≠ 0) : (7 * x)^5 = (14 * x)^4 ↔ x = 16 / 7 :=
by sorry

end unique_real_x_satisfies_eq_l267_267533


namespace smallest_repeating_block_of_3_over_11_l267_267470

def smallest_repeating_block_length (x y : ℕ) : ℕ :=
  if y ≠ 0 then (10 * x % y).natAbs else 0

theorem smallest_repeating_block_of_3_over_11 :
  smallest_repeating_block_length 3 11 = 2 :=
by sorry

end smallest_repeating_block_of_3_over_11_l267_267470


namespace triangle_perimeter_l267_267373

theorem triangle_perimeter (a b : ℕ) (h1 : a = 2) (h2 : b = 3) (x : ℕ) 
  (x_odd : x % 2 = 1) (triangle_ineq : 1 < x ∧ x < 5) : a + b + x = 8 :=
by
  sorry

end triangle_perimeter_l267_267373


namespace smallest_three_digit_multiple_of_17_l267_267868

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m := by
  use 102
  sorry

end smallest_three_digit_multiple_of_17_l267_267868


namespace number_of_fills_l267_267680

-- Definitions based on conditions
def needed_flour : ℚ := 4 + 3 / 4
def cup_capacity : ℚ := 1 / 3

-- The proof statement
theorem number_of_fills : (needed_flour / cup_capacity).ceil = 15 := by
  sorry

end number_of_fills_l267_267680


namespace find_number_eq_fifty_l267_267736

theorem find_number_eq_fifty (x : ℝ) (h : (40 / 100) * x = (25 / 100) * 80) : x = 50 := by 
  sorry

end find_number_eq_fifty_l267_267736


namespace range_of_m_l267_267479

def sufficient_condition (x m : ℝ) : Prop :=
  m - 1 < x ∧ x < m + 1

def inequality (x : ℝ) : Prop :=
  x^2 - 2 * x - 3 > 0

theorem range_of_m (m : ℝ) :
  (∀ x, sufficient_condition x m → inequality x) ↔ (m ≤ -2 ∨ m ≥ 4) :=
by 
  sorry

end range_of_m_l267_267479


namespace final_output_M_l267_267040

-- Definitions of the steps in the conditions
def initial_M : ℕ := 1
def increment_M1 (M : ℕ) : ℕ := M + 1
def increment_M2 (M : ℕ) : ℕ := M + 2

-- Define the final value of M after performing the operations
def final_M : ℕ := increment_M2 (increment_M1 initial_M)

-- The statement to prove
theorem final_output_M : final_M = 4 :=
by
  -- Placeholder for the actual proof
  sorry

end final_output_M_l267_267040


namespace least_value_of_fourth_integer_l267_267073

theorem least_value_of_fourth_integer :
  ∃ (A B C D : ℕ), 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    A + B + C + D = 64 ∧ 
    A = 3 * B ∧ B = C - 2 ∧ 
    D = 52 := sorry

end least_value_of_fourth_integer_l267_267073


namespace fraction_of_females_l267_267776

def local_soccer_league_female_fraction : Prop :=
  ∃ (males_last_year females_last_year : ℕ),
    males_last_year = 30 ∧
    (1.10 * males_last_year : ℝ) = 33 ∧
    (males_last_year + females_last_year : ℝ) * 1.15 = 52 ∧
    (females_last_year : ℝ) * 1.25 = 19 ∧
    (33 + 19 = 52)

theorem fraction_of_females
  : local_soccer_league_female_fraction → 
    ∃ (females fraction : ℝ),
    females = 19 ∧ 
    fraction = 19 / 52 :=
by
  sorry

end fraction_of_females_l267_267776


namespace ribbon_each_box_fraction_l267_267314

-- Define the conditions
def total_ribbon_used : ℚ := 5 / 12
def number_of_boxes : ℕ := 5
def ribbon_per_box : ℚ := total_ribbon_used / number_of_boxes

-- Statement to be proved
theorem ribbon_each_box_fraction :
  ribbon_per_box = 1 / 12 :=
  sorry

end ribbon_each_box_fraction_l267_267314


namespace smallest_three_digit_multiple_of_17_l267_267911

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ 17 * n ∧ 17 * n ≤ 999 ∧ 17 * n = 102 :=
begin
  sorry
end

end smallest_three_digit_multiple_of_17_l267_267911


namespace spirit_concentration_l267_267634

theorem spirit_concentration (vol_a vol_b vol_c : ℕ) (conc_a conc_b conc_c : ℝ)
(h_a : conc_a = 0.45) (h_b : conc_b = 0.30) (h_c : conc_c = 0.10)
(h_vola : vol_a = 4) (h_volb : vol_b = 5) (h_volc : vol_c = 6) : 
  (conc_a * vol_a + conc_b * vol_b + conc_c * vol_c) / (vol_a + vol_b + vol_c) * 100 = 26 := 
by
  sorry

end spirit_concentration_l267_267634


namespace total_selling_price_correct_l267_267987

noncomputable def calculateSellingPrice (price1 price2 price3 loss1 loss2 loss3 taxRate overheadCost : ℝ) : ℝ :=
  let totalPurchasePrice := price1 + price2 + price3
  let tax := taxRate * totalPurchasePrice
  let sellingPrice1 := price1 - (loss1 * price1)
  let sellingPrice2 := price2 - (loss2 * price2)
  let sellingPrice3 := price3 - (loss3 * price3)
  let totalSellingPrice := sellingPrice1 + sellingPrice2 + sellingPrice3
  totalSellingPrice + overheadCost + tax

theorem total_selling_price_correct :
  calculateSellingPrice 750 1200 500 0.10 0.15 0.05 0.05 300 = 2592.5 :=
by 
  -- The proof of this theorem is skipped.
  sorry

end total_selling_price_correct_l267_267987


namespace ratio_of_milk_to_water_l267_267304

namespace MixtureProblem

def initial_milk (total_volume : ℕ) (milk_ratio : ℕ) (water_ratio : ℕ) : ℕ :=
  (milk_ratio * total_volume) / (milk_ratio + water_ratio)

def initial_water (total_volume : ℕ) (milk_ratio : ℕ) (water_ratio : ℕ) : ℕ :=
  (water_ratio * total_volume) / (milk_ratio + water_ratio)

theorem ratio_of_milk_to_water (total_volume : ℕ) (milk_ratio : ℕ) (water_ratio : ℕ) (added_water : ℕ) :
  milk_ratio = 4 → water_ratio = 1 → total_volume = 45 → added_water = 21 → 
  (initial_milk total_volume milk_ratio water_ratio) = 36 →
  (initial_water total_volume milk_ratio water_ratio + added_water) = 30 →
  (36 / 30 : ℚ) = 6 / 5 :=
by
  intros
  sorry

end MixtureProblem

end ratio_of_milk_to_water_l267_267304


namespace shaded_area_l267_267156

-- Defining the conditions
def total_area_of_grid : ℕ := 38
def base_of_triangle : ℕ := 12
def height_of_triangle : ℕ := 4

-- Using the formula for the area of a right triangle
def area_of_unshaded_triangle : ℕ := (base_of_triangle * height_of_triangle) / 2

-- The goal: Prove the area of the shaded region
theorem shaded_area : total_area_of_grid - area_of_unshaded_triangle = 14 :=
by
  sorry

end shaded_area_l267_267156


namespace vegetation_coverage_relationship_l267_267653

noncomputable def conditions :=
  let n := 20
  let sum_x := 60
  let sum_y := 1200
  let sum_xx := 80
  let sum_xy := 640
  (n, sum_x, sum_y, sum_xx, sum_xy)

theorem vegetation_coverage_relationship
  (n sum_x sum_y sum_xx sum_xy : ℕ)
  (h1 : n = 20)
  (h2 : sum_x = 60)
  (h3 : sum_y = 1200)
  (h4 : sum_xx = 80)
  (h5 : sum_xy = 640) :
  let b1 := sum_xy / sum_xx
  let mean_x := sum_x / n
  let mean_y := sum_y / n
  (b1 = 8) ∧ (b1 * (sum_xx / sum_xy) ≤ 1) ∧ ((3, 60) = (mean_x, mean_y)) :=
by
  sorry

end vegetation_coverage_relationship_l267_267653


namespace ribbon_each_box_fraction_l267_267313

-- Define the conditions
def total_ribbon_used : ℚ := 5 / 12
def number_of_boxes : ℕ := 5
def ribbon_per_box : ℚ := total_ribbon_used / number_of_boxes

-- Statement to be proved
theorem ribbon_each_box_fraction :
  ribbon_per_box = 1 / 12 :=
  sorry

end ribbon_each_box_fraction_l267_267313


namespace sum_of_x_and_y_l267_267576

theorem sum_of_x_and_y (x y : ℤ) (h : 2 * x * y + x + y = 83) : x + y = 83 ∨ x + y = -85 :=
sorry

end sum_of_x_and_y_l267_267576


namespace average_inside_time_l267_267165

theorem average_inside_time (j_awake_frac : ℚ) (j_inside_awake_frac : ℚ) (r_awake_frac : ℚ) (r_inside_day_frac : ℚ) :
  j_awake_frac = 2 / 3 →
  j_inside_awake_frac = 1 / 2 →
  r_awake_frac = 3 / 4 →
  r_inside_day_frac = 2 / 3 →
  (24 * j_awake_frac * j_inside_awake_frac + 24 * r_awake_frac * r_inside_day_frac) / 2 = 10 := 
by
    sorry

end average_inside_time_l267_267165


namespace remainder_of_n_plus_2024_l267_267971

-- Define the assumptions
def n : ℤ := sorry  -- n will be some integer
def k : ℤ := sorry  -- k will be some integer

-- Main statement to be proved
theorem remainder_of_n_plus_2024 (h : n % 8 = 3) : (n + 2024) % 8 = 3 := sorry

end remainder_of_n_plus_2024_l267_267971


namespace intersection_M_N_l267_267584

open Set

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | (x + 2) * (x - 1) < 0}

theorem intersection_M_N : M ∩ N = {-1, 0} := by
  sorry

end intersection_M_N_l267_267584


namespace ratio_sum_odd_even_divisors_l267_267329

def M : ℕ := 33 * 38 * 58 * 462

theorem ratio_sum_odd_even_divisors : 
  let sum_odd_divisors := 
    (1 + 3 + 3^2) * (1 + 7) * (1 + 11 + 11^2) * (1 + 19) * (1 + 29)
  let sum_all_divisors := 
    (1 + 2 + 4 + 8) * (1 + 3 + 3^2) * (1 + 7) * (1 + 11 + 11^2) * (1 + 19) * (1 + 29)
  let sum_even_divisors := sum_all_divisors - sum_odd_divisors
  (sum_odd_divisors : ℚ) / sum_even_divisors = 1 / 14 :=
by sorry

end ratio_sum_odd_even_divisors_l267_267329


namespace num_isosceles_triangles_with_perimeter_30_l267_267731

theorem num_isosceles_triangles_with_perimeter_30 : 
  (∃ (s : Finset (ℕ × ℕ)), 
    (∀ (a b : ℕ), (a, b) ∈ s → 2 * a + b = 30 ∧ (a ≥ b) ∧ b ≠ 0 ∧ a + a > b ∧ a + b > a ∧ b + a > a) 
    ∧ s.card = 7) :=
by {
  sorry
}

end num_isosceles_triangles_with_perimeter_30_l267_267731


namespace smallest_three_digit_multiple_of_17_l267_267832

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := 
begin
  use 102,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    sorry }
end

end smallest_three_digit_multiple_of_17_l267_267832


namespace smallest_three_digit_multiple_of_17_l267_267896

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m) :=
  sorry

end smallest_three_digit_multiple_of_17_l267_267896


namespace xiao_peach_days_l267_267975

theorem xiao_peach_days :
  ∀ (xiao_ming_apples xiao_ming_pears xiao_ming_peaches : ℕ)
    (xiao_hong_apples xiao_hong_pears xiao_hong_peaches : ℕ)
    (both_eat_apples both_eat_pears : ℕ)
    (one_eats_apple_other_eats_pear : ℕ),
    xiao_ming_apples = 4 →
    xiao_ming_pears = 6 →
    xiao_ming_peaches = 8 →
    xiao_hong_apples = 5 →
    xiao_hong_pears = 7 →
    xiao_hong_peaches = 6 →
    both_eat_apples = 3 →
    both_eat_pears = 2 →
    one_eats_apple_other_eats_pear = 3 →
    ∃ (both_eat_peaches_days : ℕ),
      both_eat_peaches_days = 4 := 
sorry

end xiao_peach_days_l267_267975


namespace probability_at_8_10_probability_at_8_10_through_5_6_probability_at_8_10_within_circle_l267_267543

-- Definitions based on the conditions laid out in the problem
def fly_paths (n_right n_up : ℕ) : ℕ :=
  (Nat.factorial (n_right + n_up)) / ((Nat.factorial n_right) * (Nat.factorial n_up))

-- Probability for part a
theorem probability_at_8_10 : 
  (fly_paths 8 10) / (2 ^ 18) = (Nat.choose 18 8 : ℚ) / 2 ^ 18 := 
sorry

-- Probability for part b
theorem probability_at_8_10_through_5_6 :
  ((fly_paths 5 6) * (fly_paths 1 0) * (fly_paths 2 4)) / (2 ^ 18) = (6930 : ℚ) / 2 ^ 18 :=
sorry

-- Probability for part c
theorem probability_at_8_10_within_circle :
  (2 * fly_paths 2 7 * fly_paths 6 3 + 2 * fly_paths 3 6 * fly_paths 5 3 + (fly_paths 4 6) ^ 2) / (2 ^ 18) = 
  (2 * Nat.choose 9 2 * Nat.choose 9 6 + 2 * Nat.choose 9 3 * Nat.choose 9 5 + (Nat.choose 9 4) ^ 2 : ℚ) / 2 ^ 18 :=
sorry

end probability_at_8_10_probability_at_8_10_through_5_6_probability_at_8_10_within_circle_l267_267543


namespace smallest_three_digit_multiple_of_17_l267_267845

theorem smallest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, m >= 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  let n := 102,
  use n,
  split,
  { norm_num, }, -- n >= 100 is obviously true because n = 102
  split,
  { norm_num, }, -- n < 1000 is obviously true because n = 102
  split,
  { exact dvd_refl 17, }, -- 17 divides 102
  { intros m hm,
    cases hm with h100 hm,
    cases hm with h1000 hdiv,
    exact le_trans (nat.le.intro _ (symm hdiv)) (le_of_not_gt (λ h, by norm_num at h)), },
end

end smallest_three_digit_multiple_of_17_l267_267845


namespace percentage_puppies_greater_profit_l267_267674

/-- A dog breeder wants to know what percentage of puppies he can sell for a greater profit.
    Puppies with more than 4 spots sell for more money. The last litter had 10 puppies; 
    6 had 5 spots, 3 had 4 spots, and 1 had 2 spots.
    We need to prove that the percentage of puppies that can be sold for more profit is 60%. -/
theorem percentage_puppies_greater_profit
  (total_puppies : ℕ := 10)
  (puppies_with_5_spots : ℕ := 6)
  (puppies_with_4_spots : ℕ := 3)
  (puppies_with_2_spots : ℕ := 1)
  (puppies_with_more_than_4_spots := puppies_with_5_spots) :
  (puppies_with_more_than_4_spots : ℝ) / (total_puppies : ℝ) * 100 = 60 :=
by
  sorry

end percentage_puppies_greater_profit_l267_267674


namespace smallest_three_digit_multiple_of_17_correct_l267_267923

def smallest_three_digit_multiple_of_17 : ℕ :=
  102

theorem smallest_three_digit_multiple_of_17_correct :
  ∀ n, n >= 100 → n < 1000 → n % 17 = 0 → 102 ≤ n :=
begin
  intros n h1 h2 h3,
  sorry,
end

end smallest_three_digit_multiple_of_17_correct_l267_267923


namespace both_games_players_l267_267422

theorem both_games_players (kabadi_players kho_kho_only total_players both_games : ℕ)
  (h_kabadi : kabadi_players = 10)
  (h_kho_kho_only : kho_kho_only = 15)
  (h_total : total_players = 25)
  (h_equation : kabadi_players + kho_kho_only + both_games = total_players) :
  both_games = 0 :=
by
  -- question == answer given conditions
  sorry

end both_games_players_l267_267422


namespace smallest_possible_abc_l267_267761

open Nat

theorem smallest_possible_abc (a b c : ℕ)
  (h₁ : 5 * c ∣ a * b)
  (h₂ : 13 * a ∣ b * c)
  (h₃ : 31 * b ∣ a * c) :
  abc = 4060225 :=
by sorry

end smallest_possible_abc_l267_267761


namespace simple_interest_principal_l267_267268

theorem simple_interest_principal (R T SI : ℝ) (hR : R = 9 / 100) (hT : T = 1) (hSI : SI = 900) : 
  (SI / (R * T) = 10000) :=
by
  sorry

end simple_interest_principal_l267_267268


namespace range_of_a_l267_267135

noncomputable def f (x : ℝ) (a : ℝ) := Real.log (3 * x + a / x - 2)

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, 1 ≤ x → x ≤ y → f x a ≤ f y a) ↔ (-1 < a ∧ a ≤ 3) := 
sorry

end range_of_a_l267_267135


namespace percent_increase_jordan_alex_l267_267368

theorem percent_increase_jordan_alex :
  let pound_to_dollar := 1.5
  let alex_dollars := 600
  let jordan_pounds := 450
  let jordan_dollars := jordan_pounds * pound_to_dollar
  let percent_increase := ((jordan_dollars - alex_dollars) / alex_dollars) * 100
  percent_increase = 12.5 := 
by
  sorry

end percent_increase_jordan_alex_l267_267368


namespace complement_of_union_is_neg3_l267_267139

open Set

variable (U A B : Set Int)

def complement_union (U A B : Set Int) : Set Int :=
  U \ (A ∪ B)

theorem complement_of_union_is_neg3 (U A B : Set Int) (hU : U = {-3, -2, -1, 0, 1, 2, 3, 4, 5, 6})
  (hA : A = {-1, 0, 1, 2, 3}) (hB : B = {-2, 3, 4, 5, 6}) :
  complement_union U A B = {-3} :=
by
  sorry

end complement_of_union_is_neg3_l267_267139


namespace fraction_zero_implies_x_is_two_l267_267014

theorem fraction_zero_implies_x_is_two {x : ℝ} (hfrac : (2 - |x|) / (x + 2) = 0) (hdenom : x ≠ -2) : x = 2 :=
by
  sorry

end fraction_zero_implies_x_is_two_l267_267014


namespace jugglers_balls_needed_l267_267378

theorem jugglers_balls_needed (juggler_count balls_per_juggler : ℕ)
  (h_juggler_count : juggler_count = 378)
  (h_balls_per_juggler : balls_per_juggler = 6) :
  juggler_count * balls_per_juggler = 2268 :=
by
  -- This is where the proof would go.
  sorry

end jugglers_balls_needed_l267_267378


namespace distance_between_skew_edges_l267_267384

-- Definition of the given conditions
def side_length_of_base (a : ℝ) : ℝ := a
def lateral_edge_angle : ℝ := 60

-- Mathematical theorem to prove the distance between skew edges
theorem distance_between_skew_edges (a : ℝ) : Prop :=
  ∀ (A B C P : ℝ) (AP : ℝ), -- Vertices and lateral edge length
  side_length_of_base a = a →
  lateral_edge_angle = 60 →
  let M := (B + C) / 2 in   -- Midpoint of B and C
  let FM := AP * (Real.sin (lateral_edge_angle * Real.pi / 180)) in
  -- Conclusion
  FM = (3 * a) / 4 :=
  sorry

end distance_between_skew_edges_l267_267384


namespace smallest_three_digit_multiple_of_17_l267_267945

theorem smallest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m :=
by
  sorry

end smallest_three_digit_multiple_of_17_l267_267945


namespace total_sum_is_180_l267_267768

/-- Definitions and conditions given in the problem -/
def CoralineNumber : ℕ := 80
def JaydenNumber (C : ℕ) : ℕ := C - 40
def MickeyNumber (J : ℕ) : ℕ := J + 20

/-- Prove that the total sum of their numbers is 180 -/
theorem total_sum_is_180 : 
  let C := CoralineNumber
  let J := JaydenNumber C 
  let M := MickeyNumber J 
  in M + J + C = 180 := sorry

end total_sum_is_180_l267_267768


namespace find_a_l267_267610

noncomputable def f (x a : ℝ) : ℝ := |x - 4| + |x - a|

theorem find_a (a : ℝ) (h : ∃ x, f x a = 3) : a = 1 ∨ a = 7 := 
sorry

end find_a_l267_267610


namespace total_weight_of_balls_l267_267506

theorem total_weight_of_balls :
  let weight_blue := 6
  let weight_brown := 3.12
  let weight_green := 4.5
  weight_blue + weight_brown + weight_green = 13.62 := by
  sorry

end total_weight_of_balls_l267_267506


namespace smallest_three_digit_multiple_of_17_l267_267885

theorem smallest_three_digit_multiple_of_17 : 
    ∃ (n : ℤ), n ≥ 100 ∧ n < 1000 ∧ (∃ (k : ℤ), n = 17 * k) ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l267_267885


namespace parallelogram_sides_l267_267191

theorem parallelogram_sides (perimeter : ℝ) (acute_angle : Real.Angle.o) (obtuse_angle_ratio : ℝ) 
(h_perimeter : perimeter = 90) 
(h_acute_angle : acute_angle = Real.Angle.of_deg 60)
(h_obtuse_angle_ratio : obtuse_angle_ratio = 1 / 3) : 
∃ AB AD : ℝ, AB = 15 ∧ AD = 30 :=
by
  sorry

end parallelogram_sides_l267_267191


namespace popsicle_sticks_left_correct_l267_267572

noncomputable def popsicle_sticks_left (initial : ℝ) (given : ℝ) : ℝ :=
  initial - given

theorem popsicle_sticks_left_correct :
  popsicle_sticks_left 63 50 = 13 :=
by
  sorry

end popsicle_sticks_left_correct_l267_267572


namespace triangle_abc_proof_one_triangle_abc_perimeter_l267_267348

-- Defining the basic setup for the triangle and the given condition
variables (a b c A B C : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
variables (hABC : A + B + C = real.pi)
variables (h_sin : real.sin C * real.sin (A - B) = real.sin B * real.sin (C - A))

-- First proof: 2a² = b² + c²
theorem triangle_abc_proof_one : 2 * a^2 = b^2 + c^2 :=
sorry

-- Second proof: the perimeter of the triangle ABC is 14 given a = 5 and cos A = 25/31
variables (ha_val : a = 5) (cosA_val : real.cos A = 25/31)

theorem triangle_abc_perimeter : a + b + c = 14 :=
sorry

end triangle_abc_proof_one_triangle_abc_perimeter_l267_267348


namespace smallest_three_digit_multiple_of_17_l267_267874

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := sorry

end smallest_three_digit_multiple_of_17_l267_267874


namespace square_side_length_l267_267730

noncomputable def side_length_square_inscribed_in_hexagon : ℝ :=
  50 * Real.sqrt 3

theorem square_side_length (a b: ℝ) (h1 : a = 50) (h2 : b = 50 * (2 - Real.sqrt 3)) 
(s1 s2 s3 s4 s5 s6: ℝ) (ha : s1 = s2) (hb : s2 = s3) (hc : s3 = s4) 
(hd : s4 = s5) (he : s5 = s6) (hf : s6 = s1) : side_length_square_inscribed_in_hexagon = 50 * Real.sqrt 3 :=
by
  sorry

end square_side_length_l267_267730


namespace manager_salary_l267_267074

theorem manager_salary (average_salary_employees : ℕ)
    (employee_count : ℕ) (new_average_salary : ℕ)
    (total_salary_before : ℕ)
    (total_salary_after : ℕ)
    (M : ℕ) :
    average_salary_employees = 1500 →
    employee_count = 20 →
    new_average_salary = 1650 →
    total_salary_before = employee_count * average_salary_employees →
    total_salary_after = (employee_count + 1) * new_average_salary →
    M = total_salary_after - total_salary_before →
    M = 4650 := by
    intros h1 h2 h3 h4 h5 h6
    rw [h6]
    sorry -- The proof is not required, so we use 'sorry' here.

end manager_salary_l267_267074


namespace smallest_three_digit_multiple_of_17_l267_267941

/--
Prove that the smallest three-digit multiple of 17 is 102.
-/
theorem smallest_three_digit_multiple_of_17 : ∃ k : ℤ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 := 
by
  sorry

end smallest_three_digit_multiple_of_17_l267_267941


namespace quadratic_max_value_l267_267273

open Real

variables (a b c x : ℝ)
noncomputable def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_max_value (h₀ : a < 0) (x₀ : ℝ) (h₁ : 2 * a * x₀ + b = 0) : 
  ∀ x : ℝ, f a b c x ≤ f a b c x₀ := sorry

end quadratic_max_value_l267_267273


namespace smallest_three_digit_multiple_of_17_l267_267848

theorem smallest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, m >= 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  let n := 102,
  use n,
  split,
  { norm_num, }, -- n >= 100 is obviously true because n = 102
  split,
  { norm_num, }, -- n < 1000 is obviously true because n = 102
  split,
  { exact dvd_refl 17, }, -- 17 divides 102
  { intros m hm,
    cases hm with h100 hm,
    cases hm with h1000 hdiv,
    exact le_trans (nat.le.intro _ (symm hdiv)) (le_of_not_gt (λ h, by norm_num at h)), },
end

end smallest_three_digit_multiple_of_17_l267_267848


namespace min_value_of_x2_plus_y2_l267_267580

open Real

theorem min_value_of_x2_plus_y2 (x y : ℝ) (h : x^2 + y^2 - 4 * x + 1 = 0) :
  x^2 + y^2 ≥ 7 - 4 * sqrt 3 := sorry

end min_value_of_x2_plus_y2_l267_267580


namespace solution_set_eq_interval_l267_267672

variable (f : ℝ → ℝ)
variable (A : ℝ × ℝ) (B : ℝ × ℝ)
variable (h_increasing : ∀ x1 x2 : ℝ, x1 < x2 → f(x1) < f(x2))
variable (hA : A = (0, -1))
variable (hB : B = (3, 1))

theorem solution_set_eq_interval :
  { x : ℝ | abs (f x) < 1 } = set.Ioo 0 3 :=
by
  sorry

end solution_set_eq_interval_l267_267672


namespace find_slope_of_line_l_l267_267596

-- Define the vectors OA and OB
def OA : ℝ × ℝ := (4, 1)
def OB : ℝ × ℝ := (2, -3)

-- The slope k is such that the lengths of projections of OA and OB on line l are equal
theorem find_slope_of_line_l (k : ℝ) :
  (|4 + k| = |2 - 3 * k|) → (k = 3 ∨ k = -1/2) :=
by
  -- Intentionally leave the proof out
  sorry

end find_slope_of_line_l_l267_267596


namespace ellipse_k_range_l267_267038

theorem ellipse_k_range
  (k : ℝ)
  (h1 : k - 4 > 0)
  (h2 : 10 - k > 0)
  (h3 : k - 4 > 10 - k) :
  7 < k ∧ k < 10 :=
sorry

end ellipse_k_range_l267_267038


namespace sum_of_24_consecutive_integers_is_square_l267_267194

theorem sum_of_24_consecutive_integers_is_square : ∃ n : ℕ, ∃ k : ℕ, (n > 0) ∧ (24 * (2 * n + 23)) = k * k ∧ k * k = 324 :=
by
  sorry

end sum_of_24_consecutive_integers_is_square_l267_267194


namespace question_1_question_2_l267_267134

noncomputable def f (x m : ℝ) : ℝ := abs (x + m) - abs (2 * x - 2 * m)

theorem question_1 (x : ℝ) (m : ℝ) (h : m = 1/2) (h_pos : m > 0) : 
  (f x m ≥ 1/2) ↔ (1/3 ≤ x ∧ x < 1) :=
sorry

theorem question_2 (m : ℝ) (h_pos : m > 0) : 
  (∀ x : ℝ, ∃ t : ℝ, f x m + abs (t - 3) < abs (t + 4)) ↔ (0 < m ∧ m < 7/2) :=
sorry

end question_1_question_2_l267_267134


namespace Michael_selection_l267_267030

theorem Michael_selection :
  (Nat.choose 8 3) * (Nat.choose 5 2) = 560 :=
by
  sorry

end Michael_selection_l267_267030


namespace average_weight_of_three_l267_267037

theorem average_weight_of_three :
  ∀ A B C : ℝ,
  (A + B) / 2 = 40 →
  (B + C) / 2 = 43 →
  B = 31 →
  (A + B + C) / 3 = 45 :=
by
  intros A B C h1 h2 h3
  sorry

end average_weight_of_three_l267_267037


namespace isometric_curve_l267_267778

noncomputable def Q (a b c x y : ℝ) := a * x^2 + 2 * b * x * y + c * y^2

theorem isometric_curve (a b c d e f : ℝ) (h : a * c - b^2 = 0) :
  ∃ (p : ℝ), (Q a b c x y + 2 * d * x + 2 * e * y = f → 
    (y^2 = 2 * p * x) ∨ 
    (∃ c' : ℝ, y^2 = c'^2) ∨ 
    y^2 = 0 ∨ 
    ∀ x y : ℝ, false) :=
sorry

end isometric_curve_l267_267778


namespace polynomial_expansion_l267_267439

theorem polynomial_expansion (z : ℤ) :
  (3 * z^3 + 6 * z^2 - 5 * z - 4) * (4 * z^4 - 3 * z^2 + 7) =
  12 * z^7 + 24 * z^6 - 29 * z^5 - 34 * z^4 + 36 * z^3 + 54 * z^2 + 35 * z - 28 := by
  -- Provide a proof here
  sorry

end polynomial_expansion_l267_267439


namespace ms_hatcher_students_l267_267614

-- Define the number of third-graders
def third_graders : ℕ := 20

-- Condition: The number of fourth-graders is twice the number of third-graders
def fourth_graders : ℕ := 2 * third_graders

-- Condition: The number of fifth-graders is half the number of third-graders
def fifth_graders : ℕ := third_graders / 2

-- The total number of students Ms. Hatcher teaches in a day
def total_students : ℕ := third_graders + fourth_graders + fifth_graders

-- The statement to prove
theorem ms_hatcher_students : total_students = 70 := by
  sorry

end ms_hatcher_students_l267_267614


namespace percent_of_y_l267_267745

theorem percent_of_y (y : ℝ) (hy : y > 0) : (6 * y / 20) + (3 * y / 10) = 0.6 * y :=
by
  sorry

end percent_of_y_l267_267745


namespace smallest_three_digit_multiple_of_17_l267_267877

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := sorry

end smallest_three_digit_multiple_of_17_l267_267877


namespace neq_is_necessary_but_not_sufficient_l267_267481

theorem neq_is_necessary_but_not_sufficient (a b : ℝ) : (a ≠ b) → ¬ (∀ a b : ℝ, (a ≠ b) → (a / b + b / a > 2)) ∧ (∀ a b : ℝ, (a / b + b / a > 2) → (a ≠ b)) :=
by {
    sorry
}

end neq_is_necessary_but_not_sufficient_l267_267481


namespace alejandro_candies_l267_267076

theorem alejandro_candies (n : ℕ) (S_n : ℕ) :
  (S_n = 2^n - 1 ∧ S_n ≥ 2007) → ((2^11 - 1 - 2007 = 40) ∧ (∃ k, k = 11)) :=
  by
    sorry

end alejandro_candies_l267_267076


namespace scientific_notation_35_million_l267_267095

theorem scientific_notation_35_million :
  35000000 = 3.5 * (10 : Float) ^ 7 := 
by
  sorry

end scientific_notation_35_million_l267_267095


namespace area_of_polygon_l267_267157

-- Define the conditions
variables (n : ℕ) (s : ℝ)
-- Given that polygon has 32 sides.
def sides := 32
-- Each side is congruent, and the total perimeter is 64 units.
def perimeter := 64
-- Side length of each side
def side_length := perimeter / sides

-- Area of the polygon we need to prove
def target_area := 96

theorem area_of_polygon : side_length * side_length * sides = target_area := 
by {
  -- Here proof would come in reality, we'll skip it by sorry for now.
  sorry
}

end area_of_polygon_l267_267157


namespace f_is_odd_f_is_monotone_l267_267136

noncomputable def f (k x : ℝ) : ℝ := x + k / x

-- Proving f(x) is odd
theorem f_is_odd (k : ℝ) (hk : k ≠ 0) : ∀ x : ℝ, f k (-x) = -f k x :=
by
  intro x
  sorry

-- Proving f(x) is monotonically increasing on [sqrt(k), +∞) for k > 0
theorem f_is_monotone (k : ℝ) (hk : k > 0) : ∀ x1 x2 : ℝ, 
  x1 ∈ Set.Ici (Real.sqrt k) → x2 ∈ Set.Ici (Real.sqrt k) → x1 < x2 → f k x1 < f k x2 :=
by
  intro x1 x2 hx1 hx2 hlt
  sorry

end f_is_odd_f_is_monotone_l267_267136


namespace width_after_water_rises_l267_267661

noncomputable def parabolic_arch_bridge_width : ℝ × ℝ → Prop :=
  λ ⟨x, y⟩,
  let a := -8 in
  (x^2 = a * y) ∧ (x = 4 ∧ y = -2)

theorem width_after_water_rises :
  ∀ (x y : ℝ), y = -2 + 1/2 →
  x^2 = -8 * y →
  2 * abs x = 4 * real.sqrt 3 :=
by
  intro x y h1 h2
  rw h1 at h2
  simp at h2
  rw abs at h2
  exact ⟨⟨_, _⟩⟩ -- sorry

end width_after_water_rises_l267_267661


namespace athlete_running_minutes_l267_267681

theorem athlete_running_minutes (r w : ℕ) 
  (h1 : r + w = 60)
  (h2 : 10 * r + 4 * w = 450) : 
  r = 35 := 
sorry

end athlete_running_minutes_l267_267681


namespace iris_jackets_l267_267159

theorem iris_jackets (J : ℕ) (h : 10 * J + 12 + 48 = 90) : J = 3 :=
by
  sorry

end iris_jackets_l267_267159


namespace determine_b_l267_267586

def imaginary_unit : Type := {i : ℂ // i^2 = -1}

theorem determine_b (i : imaginary_unit) (b : ℝ) : 
  (2 - i.val) * 4 * i.val = 4 - b * i.val → b = -8 :=
by
  sorry

end determine_b_l267_267586


namespace smallest_triangle_perimeter_consecutive_even_l267_267219

theorem smallest_triangle_perimeter_consecutive_even :
  ∃ (a b c : ℕ), a = 2 ∧ b = 4 ∧ c = 6 ∧
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧ (a + b + c = 12) :=
by {
  sorry
}

end smallest_triangle_perimeter_consecutive_even_l267_267219


namespace find_h_l267_267382

theorem find_h {a b c n k : ℝ} (x : ℝ) (h_val : ℝ) 
  (h_quad : a * x^2 + b * x + c = 3 * (x - 5)^2 + 15) :
  (4 * a) * x^2 + (4 * b) * x + (4 * c) = n * (x - h_val)^2 + k → h_val = 5 :=
sorry

end find_h_l267_267382


namespace courtyard_width_l267_267081

theorem courtyard_width (length : ℕ) (brick_length brick_width : ℕ) (num_bricks : ℕ) (W : ℕ)
  (H1 : length = 25)
  (H2 : brick_length = 20)
  (H3 : brick_width = 10)
  (H4 : num_bricks = 18750)
  (H5 : 2500 * (W * 100) = num_bricks * (brick_length * brick_width)) :
  W = 15 :=
by sorry

end courtyard_width_l267_267081


namespace sum_of_squares_l267_267603

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 0)
  (h2 : a^3 + b^3 + c^3 = a^5 + b^5 + c^5) (h3 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6 / 5 :=
by
  sorry

end sum_of_squares_l267_267603


namespace arithmetic_sequence_sum_l267_267450

variable {a : ℕ → ℕ}

noncomputable def is_arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (h_arith : is_arithmetic_seq a) (h_a5 : a 5 = 2) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 2 * 9 :=
by
  sorry

end arithmetic_sequence_sum_l267_267450


namespace smallest_three_digit_multiple_of_17_l267_267946

theorem smallest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m :=
by
  sorry

end smallest_three_digit_multiple_of_17_l267_267946


namespace quadratic_expression_l267_267719

theorem quadratic_expression (x1 x2 : ℝ) (h1 : x1^2 - 3 * x1 + 1 = 0) (h2 : x2^2 - 3 * x2 + 1 = 0) : 
  x1^2 - 2 * x1 + x2 = 2 :=
sorry

end quadratic_expression_l267_267719


namespace prime_equally_spaced_on_unit_circle_l267_267143

theorem prime_equally_spaced_on_unit_circle :
  (∃! n : ℕ, n ≥ 2 ∧ Nat.Prime n ∧ (∀ z : Fin n → ℂ, (∀ i, complex.abs (z i) = 1) ∧ (∑ i, z i = 0) → (∃ d : ℂ, d ≠ 0 ∧ (∀ i, z i = complex.of_real (cos (2 * π * i / n : ℝ)) + complex.I * complex.of_real (sin (2 * π * i / n : ℝ))))))
  ∧ n = 2 ∨ n = 3 :=
by 
  sorry

end prime_equally_spaced_on_unit_circle_l267_267143


namespace smallest_three_digit_multiple_of_17_l267_267895

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m) :=
  sorry

end smallest_three_digit_multiple_of_17_l267_267895


namespace find_m_l267_267766

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define the set A based on the condition in the problem
def A (m : ℕ) : Set ℕ := {x ∈ U | x^2 - 5 * x + m = 0}

-- Define the complement of A in the universal set U
def complementA (m : ℕ) : Set ℕ := U \ A m

-- Given condition that the complement of A in U is {2, 3}
def complementA_condition : Set ℕ := {2, 3}

-- The proof problem statement: Prove that m = 4 given the conditions
theorem find_m (m : ℕ) (h : complementA m = complementA_condition) : m = 4 :=
sorry

end find_m_l267_267766


namespace repeating_block_length_of_three_elevens_l267_267476

def smallest_repeating_block_length (x : ℚ) : ℕ :=
  sorry -- Definition to compute smallest repeating block length if needed

theorem repeating_block_length_of_three_elevens :
  smallest_repeating_block_length (3 / 11) = 2 :=
sorry

end repeating_block_length_of_three_elevens_l267_267476


namespace complement_of_A_in_U_l267_267744

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≥ 1} ∪ {x | x ≤ 0}
def C_UA : Set ℝ := U \ A

theorem complement_of_A_in_U :
  C_UA = {x | 0 < x ∧ x < 1} :=
sorry

end complement_of_A_in_U_l267_267744


namespace hypotenuse_length_l267_267991

theorem hypotenuse_length (a c : ℝ) (h_perimeter : 2 * a + c = 36) (h_area : (1 / 2) * a^2 = 24) : c = 4 * Real.sqrt 6 :=
by
  sorry

end hypotenuse_length_l267_267991


namespace original_bet_is_40_l267_267235

-- Definition relating payout ratio and payout to original bet
def calculate_original_bet (payout_ratio payout : ℚ) : ℚ :=
  payout / payout_ratio

-- Given conditions
def payout_ratio : ℚ := 3 / 2
def received_payout : ℚ := 60

-- The proof goal
theorem original_bet_is_40 : calculate_original_bet payout_ratio received_payout = 40 :=
by
  sorry

end original_bet_is_40_l267_267235


namespace ribbon_per_box_l267_267316

theorem ribbon_per_box (ribbon_total ribbon_each : ℚ) (n : ℕ) (hn : n = 5) (h : ribbon_total = 5 / 12) :
  ribbon_each = ribbon_total / n ↔ ribbon_each = 1 / 12 :=
by
  sorry

end ribbon_per_box_l267_267316


namespace correct_model_l267_267392

def average_homework_time_decrease (x : ℝ) : Prop :=
  100 * (1 - x) ^ 2 = 70

theorem correct_model (x : ℝ) : average_homework_time_decrease x := 
  sorry

end correct_model_l267_267392


namespace real_root_exists_l267_267692

theorem real_root_exists (p : ℝ) : ∃ x : ℝ, x^4 + 2*p*x^3 + x^3 + 2*p*x + 1 = 0 :=
sorry

end real_root_exists_l267_267692


namespace logical_equivalence_l267_267535

variables (P Q : Prop)

theorem logical_equivalence :
  (¬P → ¬Q) ↔ (Q → P) :=
sorry

end logical_equivalence_l267_267535


namespace expected_rolls_to_sum_2010_l267_267428

/-- The expected number of rolls to achieve a sum of 2010 with a fair six-sided die -/
theorem expected_rolls_to_sum_2010 (die : ℕ → ℕ) (fair_die : ∀ i [1 ≤ i ∧ i ≤ 6], P(die = i) = 1/6) :
  expected_roll_sum die 2010 = 574.5238095 :=
sorry

end expected_rolls_to_sum_2010_l267_267428


namespace simplify_and_evaluate_expression_l267_267786

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = Real.tan (Real.pi / 3) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2*m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l267_267786


namespace price_of_each_sundae_l267_267541

theorem price_of_each_sundae
  (num_ice_cream_bars : ℕ := 125) 
  (num_sundaes : ℕ := 125) 
  (total_price : ℝ := 225)
  (price_per_ice_cream_bar : ℝ := 0.60) :
  ∃ (price_per_sundae : ℝ), price_per_sundae = 1.20 := 
by
  -- Variables for costs of ice-cream bars and sundaes' total cost
  let cost_ice_cream_bars := num_ice_cream_bars * price_per_ice_cream_bar
  let total_cost_sundaes := total_price - cost_ice_cream_bars
  let price_per_sundae := total_cost_sundaes / num_sundaes
  use price_per_sundae
  sorry

end price_of_each_sundae_l267_267541


namespace ms_hatcher_students_l267_267613

-- Define the number of third-graders
def third_graders : ℕ := 20

-- Condition: The number of fourth-graders is twice the number of third-graders
def fourth_graders : ℕ := 2 * third_graders

-- Condition: The number of fifth-graders is half the number of third-graders
def fifth_graders : ℕ := third_graders / 2

-- The total number of students Ms. Hatcher teaches in a day
def total_students : ℕ := third_graders + fourth_graders + fifth_graders

-- The statement to prove
theorem ms_hatcher_students : total_students = 70 := by
  sorry

end ms_hatcher_students_l267_267613


namespace solve_for_b_l267_267262

theorem solve_for_b (b : ℝ) (hb : b + ⌈b⌉ = 17.8) : b = 8.8 := 
by sorry

end solve_for_b_l267_267262


namespace coefficient_of_x_100_l267_267700

-- Define the polynomial P
noncomputable def P : Polynomial ℤ :=
  (Polynomial.C (-1) + Polynomial.X) *
  (Polynomial.C (-2) + Polynomial.X^2) *
  (Polynomial.C (-3) + Polynomial.X^3) *
  (Polynomial.C (-4) + Polynomial.X^4) *
  (Polynomial.C (-5) + Polynomial.X^5) *
  (Polynomial.C (-6) + Polynomial.X^6) *
  (Polynomial.C (-7) + Polynomial.X^7) *
  (Polynomial.C (-8) + Polynomial.X^8) *
  (Polynomial.C (-9) + Polynomial.X^9) *
  (Polynomial.C (-10) + Polynomial.X^10) *
  (Polynomial.C (-11) + Polynomial.X^11) *
  (Polynomial.C (-12) + Polynomial.X^12) *
  (Polynomial.C (-13) + Polynomial.X^13) *
  (Polynomial.C (-14) + Polynomial.X^14) *
  (Polynomial.C (-15) + Polynomial.X^15)

-- State the theorem
theorem coefficient_of_x_100 : P.coeff 100 = 445 :=
  by sorry

end coefficient_of_x_100_l267_267700


namespace solution_l267_267188

-- Definition of the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^2 + 3 * m * x + 1

-- Statement of the problem
theorem solution (m : ℝ) (x : ℝ) (h : quadratic_equation m x = (m - 2) * x^2 + 3 * m * x + 1) : m ≠ 2 :=
by
  sorry

end solution_l267_267188


namespace gcd_lcm_divisible_l267_267643

theorem gcd_lcm_divisible (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : Nat.gcd a b + Nat.lcm a b = a + b) : a % b = 0 ∨ b % a = 0 := 
sorry

end gcd_lcm_divisible_l267_267643


namespace smallest_three_digit_multiple_of_17_l267_267913

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ 17 * n ∧ 17 * n ≤ 999 ∧ 17 * n = 102 :=
begin
  sorry
end

end smallest_three_digit_multiple_of_17_l267_267913


namespace caleb_apples_less_than_kayla_l267_267511

theorem caleb_apples_less_than_kayla :
  ∀ (Kayla Suraya Caleb : ℕ),
  (Kayla = 20) →
  (Suraya = Kayla + 7) →
  (Suraya = Caleb + 12) →
  (Suraya = 27) →
  (Kayla - Caleb = 5) :=
by
  intros Kayla Suraya Caleb hKayla hSuraya1 hSuraya2 hSuraya3
  sorry

end caleb_apples_less_than_kayla_l267_267511


namespace find_DF_l267_267574

-- Conditions
variables {A B C D E F : Type}
variables {BC EF AC DF : ℝ}
variable (h_similar : similar_triangles A B C D E F)
variable (h_BC : BC = 6)
variable (h_EF : EF = 4)
variable (h_AC : AC = 9)

-- Question: Prove DF = 6 given the above conditions
theorem find_DF : DF = 6 :=
by
  sorry

end find_DF_l267_267574


namespace tangent_line_MP_l267_267684

theorem tangent_line_MP
  (O : Type)
  (circle : O → O → Prop)
  (K M N P L : O)
  (is_tangent : O → O → Prop)
  (is_diameter : O → O → O)
  (K_tangent : is_tangent K M)
  (eq_segments : ∀ {P Q R}, circle P Q → circle Q R → circle P R → (P, Q) = (Q, R))
  (diam_opposite : L = is_diameter K L)
  (line_intrsc : ∀ {X Y}, is_tangent X Y → circle X Y → (Y = Y) → P = Y)
  (circ : ∀ {X Y}, circle X Y) :
  is_tangent M P :=
by
  sorry

end tangent_line_MP_l267_267684


namespace ratio_of_smaller_to_bigger_l267_267204

theorem ratio_of_smaller_to_bigger (S B : ℕ) (h_bigger: B = 104) (h_sum: S + B = 143) :
  S / B = 39 / 104 := sorry

end ratio_of_smaller_to_bigger_l267_267204


namespace number_of_truthful_dwarfs_l267_267436

def num_dwarfs : Nat := 10

def likes_vanilla : Nat := num_dwarfs

def likes_chocolate : Nat := num_dwarfs / 2

def likes_fruit : Nat := 1

theorem number_of_truthful_dwarfs : 
  ∃ t l : Nat, 
  t + l = num_dwarfs ∧  -- total number of dwarfs
  t + 2 * l = likes_vanilla + likes_chocolate + likes_fruit ∧  -- total number of hand raises
  t = 4 :=  -- number of truthful dwarfs
  sorry

end number_of_truthful_dwarfs_l267_267436


namespace volume_of_rectangular_prism_l267_267201

theorem volume_of_rectangular_prism :
  ∃ (a b c : ℝ), (a * b = 54) ∧ (b * c = 56) ∧ (a * c = 60) ∧ (a * b * c = 379) :=
by sorry

end volume_of_rectangular_prism_l267_267201


namespace inequality_solution_l267_267034

theorem inequality_solution (x : ℝ) :
  (x + 2 > 3 * (2 - x) ∧ x < (x + 3) / 2) ↔ 1 < x ∧ x < 3 := sorry

end inequality_solution_l267_267034


namespace find_m_range_l267_267454

def vector_a : ℝ × ℝ := (1, 2)
def dot_product (a b : ℝ × ℝ) : ℝ := (a.1 * b.1) + (a.2 * b.2)
def is_acute (a b : ℝ × ℝ) : Prop := dot_product a b > 0

theorem find_m_range (m : ℝ) :
  is_acute vector_a (4, m) → m ∈ Set.Ioo (-2 : ℝ) 8 ∪ Set.Ioi 8 := 
by
  sorry

end find_m_range_l267_267454


namespace thirty_k_divisor_of_929260_l267_267739

theorem thirty_k_divisor_of_929260 (k : ℕ) (h1: 30^k ∣ 929260):
(3^k - k^3 = 2) :=
sorry

end thirty_k_divisor_of_929260_l267_267739


namespace part1_part2_l267_267338

-- Definitions and conditions
variables (A B C : ℝ) (a b c : ℝ)
-- Given conditions
variable (h1 : sin C * sin (A - B) = sin B * sin (C - A))

-- Proof for part (1)
theorem part1 : 2 * a^2 = b^2 + c^2 :=
  sorry

-- Given further conditions for part (2)
variables (a_eq : a = 5) (cosA_eq : cos A = 25 / 31)

-- Proof for part (2)
theorem part2 : a + b + c = 14 :=
  sorry

end part1_part2_l267_267338


namespace cobbler_mends_3_pairs_per_hour_l267_267542

def cobbler_hours_per_day_mon_thu := 8
def cobbler_hours_friday := 11 - 8
def cobbler_total_hours_week := 4 * cobbler_hours_per_day_mon_thu + cobbler_hours_friday
def cobbler_pairs_per_week := 105
def cobbler_pairs_per_hour := cobbler_pairs_per_week / cobbler_total_hours_week

theorem cobbler_mends_3_pairs_per_hour : cobbler_pairs_per_hour = 3 := 
by 
  -- Add the steps if necessary but in this scenario, we are skipping proof details
  sorry

end cobbler_mends_3_pairs_per_hour_l267_267542


namespace find_factor_l267_267091

theorem find_factor (x f : ℕ) (hx : x = 110) (h : x * f - 220 = 110) : f = 3 :=
sorry

end find_factor_l267_267091


namespace range_of_a_l267_267287

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x^2 + a * x + 2)
  (h2 : ∀ y, (∃ x, y = f (f x)) ↔ (∃ x, y = f x)) : a ≥ 4 ∨ a ≤ -2 := 
sorry

end range_of_a_l267_267287


namespace cost_of_two_dogs_l267_267772

theorem cost_of_two_dogs (original_price : ℤ) (profit_margin : ℤ) (num_dogs : ℤ) (final_price : ℤ) :
  original_price = 1000 →
  profit_margin = 30 →
  num_dogs = 2 →
  final_price = original_price + (profit_margin * original_price / 100) →
  num_dogs * final_price = 2600 :=
by
  sorry

end cost_of_two_dogs_l267_267772


namespace CarmenBrushLengthInCentimeters_l267_267109

-- Given conditions
def CarlaBrushLengthInInches : ℝ := 12
def CarmenBrushPercentIncrease : ℝ := 0.5
def InchToCentimeterConversionFactor : ℝ := 2.5

-- Question: What is Carmen's brush length in centimeters?
-- Proof Goal: Prove that Carmen's brush length in centimeters is 45 cm.
theorem CarmenBrushLengthInCentimeters :
  let CarmenBrushLengthInInches := CarlaBrushLengthInInches * (1 + CarmenBrushPercentIncrease)
  CarmenBrushLengthInInches * InchToCentimeterConversionFactor = 45 := by
  -- sorry is used as a placeholder for the completed proof
  sorry

end CarmenBrushLengthInCentimeters_l267_267109


namespace smallest_three_digit_multiple_of_17_l267_267836

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m := by
  sorry

end smallest_three_digit_multiple_of_17_l267_267836


namespace minimum_bench_sections_l267_267246

theorem minimum_bench_sections (N : ℕ) (hN : 8 * N = 12 * N) : N = 3 :=
sorry

end minimum_bench_sections_l267_267246


namespace solution_set_of_abs_inequality_l267_267519

theorem solution_set_of_abs_inequality :
  {x : ℝ // |2 * x - 1| < 3} = {x : ℝ // -1 < x ∧ x < 2} :=
by sorry

end solution_set_of_abs_inequality_l267_267519


namespace lemons_required_for_new_recipe_l267_267433

noncomputable def lemons_needed_to_make_gallons (lemons_original : ℕ) (gallons_original : ℕ) (additional_lemons : ℕ) (additional_gallons : ℕ) (gallons_new : ℕ) : ℝ :=
  let lemons_per_gallon := (lemons_original : ℝ) / (gallons_original : ℝ)
  let additional_lemons_per_gallon := (additional_lemons : ℝ) / (additional_gallons : ℝ)
  let total_lemons_per_gallon := lemons_per_gallon + additional_lemons_per_gallon
  total_lemons_per_gallon * (gallons_new : ℝ)

theorem lemons_required_for_new_recipe : lemons_needed_to_make_gallons 36 48 2 6 18 = 19.5 :=
by
  sorry

end lemons_required_for_new_recipe_l267_267433


namespace arithmetic_sequence_sum_l267_267016

theorem arithmetic_sequence_sum :
  (∀ (a : ℕ → ℤ),  a 1 + a 2 = 4 ∧ a 3 + a 4 = 6 → a 8 + a 9 = 10) :=
sorry

end arithmetic_sequence_sum_l267_267016


namespace terminative_decimal_of_45_div_72_l267_267259

theorem terminative_decimal_of_45_div_72 :
  (45 / 72 : ℚ) = 0.625 :=
sorry

end terminative_decimal_of_45_div_72_l267_267259


namespace length_of_overlapping_part_l267_267976

theorem length_of_overlapping_part
  (l_p : ℕ)
  (n : ℕ)
  (total_length : ℕ)
  (l_o : ℕ) :
  n = 3 →
  l_p = 217 →
  total_length = 627 →
  3 * l_p - 2 * l_o = total_length →
  l_o = 12 := by
  intros n_eq l_p_eq total_length_eq equation
  sorry

end length_of_overlapping_part_l267_267976


namespace volume_of_prism_l267_267202

theorem volume_of_prism (a b c : ℝ) (h1 : a * b = 54) (h2 : b * c = 56) (h3 : a * c = 60) :
    a * b * c = 426 :=
sorry

end volume_of_prism_l267_267202


namespace ratio_problem_l267_267292

theorem ratio_problem
  (a b c d : ℝ)
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 49) :
  d / a = 1 / 122.5 :=
by {
  -- Proof steps would go here
  sorry
}

end ratio_problem_l267_267292


namespace smallest_three_digit_multiple_of_17_l267_267943

/--
Prove that the smallest three-digit multiple of 17 is 102.
-/
theorem smallest_three_digit_multiple_of_17 : ∃ k : ℤ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 := 
by
  sorry

end smallest_three_digit_multiple_of_17_l267_267943


namespace solve_equation_l267_267627

theorem solve_equation (x : ℝ) (h : x ≠ 2 / 3) :
  (6 * x + 2) / (3 * x^2 + 6 * x - 4) = (3 * x) / (3 * x - 2) ↔ (x = (Real.sqrt 6) / 3 ∨ x = -(Real.sqrt 6) / 3) :=
by sorry

end solve_equation_l267_267627


namespace solve_phi_l267_267288

-- Define the problem
noncomputable def f (phi x : ℝ) : ℝ := Real.cos (Real.sqrt 3 * x + phi)
noncomputable def f' (phi x : ℝ) : ℝ := -Real.sqrt 3 * Real.sin (Real.sqrt 3 * x + phi)
noncomputable def g (phi x : ℝ) : ℝ := f phi x + f' phi x

-- Define the main theorem
theorem solve_phi (phi : ℝ) (h : -Real.pi < phi ∧ phi < 0) 
  (even_g : ∀ x, g phi x = g phi (-x)) : phi = -Real.pi / 3 :=
sorry

end solve_phi_l267_267288


namespace find_x_pow_8_l267_267589

theorem find_x_pow_8 (x : ℂ) (h : x + x⁻¹ = real.sqrt 2) : x^8 = 1 := 
sorry

end find_x_pow_8_l267_267589


namespace value_of_x_l267_267146

theorem value_of_x (x y : ℝ) (h1 : y = 2) (h2 : y = 1 / (4 * x + 2)) : x = -3 / 8 := 
by
  sorry

end value_of_x_l267_267146


namespace smallest_three_digit_multiple_of_17_l267_267909

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ 17 * n ∧ 17 * n ≤ 999 ∧ 17 * n = 102 :=
begin
  sorry
end

end smallest_three_digit_multiple_of_17_l267_267909


namespace min_value_reciprocal_sum_l267_267023

open Real

theorem min_value_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = 20) :
  (1 / a + 1 / b) ≥ 1 / 5 :=
by 
  sorry

end min_value_reciprocal_sum_l267_267023


namespace max_value_of_4x_plus_3y_l267_267709

theorem max_value_of_4x_plus_3y (x y : ℝ) (h : x^2 + y^2 = 18 * x + 8 * y + 10) :
  4 * x + 3 * y ≤ 45 :=
sorry

end max_value_of_4x_plus_3y_l267_267709


namespace five_coins_no_105_cents_l267_267270

theorem five_coins_no_105_cents :
  ¬ ∃ (a b c d e : ℕ), a + b + c + d + e = 5 ∧
    (a * 1 + b * 5 + c * 10 + d * 25 + e * 50 = 105) :=
sorry

end five_coins_no_105_cents_l267_267270


namespace average_age_combined_l267_267631

theorem average_age_combined (fifth_graders_count : ℕ) (fifth_graders_avg_age : ℚ)
                             (parents_count : ℕ) (parents_avg_age : ℚ)
                             (grandparents_count : ℕ) (grandparents_avg_age : ℚ) :
  fifth_graders_count = 40 →
  fifth_graders_avg_age = 10 →
  parents_count = 60 →
  parents_avg_age = 35 →
  grandparents_count = 20 →
  grandparents_avg_age = 65 →
  (fifth_graders_count * fifth_graders_avg_age + 
   parents_count * parents_avg_age + 
   grandparents_count * grandparents_avg_age) / 
  (fifth_graders_count + parents_count + grandparents_count) = 95 / 3 := sorry

end average_age_combined_l267_267631


namespace harmonic_inequality_l267_267491

open BigOperators

noncomputable def a (n : ℕ) : ℝ := ∑ i in finset.range (n+1), 1 / (i+1)

theorem harmonic_inequality (n : ℕ) (h : n ≥ 2) : 
  (a n)^2 > 2 * ∑ i in finset.range (n), a (i+1) / (i + 2) :=
begin
  sorry
end

end harmonic_inequality_l267_267491


namespace construction_company_doors_needed_l267_267426

-- Definitions based on conditions
def num_floors_per_building : ℕ := 20
def num_apartments_per_floor : ℕ := 8
def num_buildings : ℕ := 3
def num_doors_per_apartment : ℕ := 10

-- Total number of apartments
def total_apartments : ℕ :=
  num_floors_per_building * num_apartments_per_floor * num_buildings

-- Total number of doors
def total_doors_needed : ℕ :=
  num_doors_per_apartment * total_apartments

-- Theorem statement to prove the number of doors needed
theorem construction_company_doors_needed :
  total_doors_needed = 4800 :=
sorry

end construction_company_doors_needed_l267_267426


namespace one_greater_l267_267282

theorem one_greater (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) 
  (h5 : a + b + c > 1 / a + 1 / b + 1 / c) : 
  (1 < a ∧ b ≤ 1 ∧ c ≤ 1) ∨ (1 < b ∧ a ≤ 1 ∧ c ≤ 1) ∨ (1 < c ∧ a ≤ 1 ∧ b ≤ 1) :=
sorry

end one_greater_l267_267282


namespace num_sets_n_eq_6_num_sets_general_l267_267496

-- Definitions for the sets and conditions.
def S (n : ℕ) := {i | 1 ≤ i ∧ i ≤ n}
def A (a₁ a₂ a₃ : ℕ) (n : ℕ) := {i | i = a₁ ∨ i = a₂ ∨ i = a₃}
def valid_set (a₁ a₂ a₃ : ℕ) (n : ℕ) :=
  a₁ < a₂ ∧ a₂ < a₃ ∧ a₃ - a₂ ≤ 2 ∧ a₁ ∈ S n ∧ a₂ ∈ S n ∧ a₃ ∈ S n

-- Number of sets satisfying the condition for a specific n value
theorem num_sets_n_eq_6 :
  ∀ (n : ℕ), n = 6 → (∃ (count : ℕ), count = 16 ∧ (∀ (a₁ a₂ a₃ : ℕ), valid_set a₁ a₂ a₃ n → true)) :=
by
  sorry

-- Number of sets satisfying the condition for any n >= 5
theorem num_sets_general (n : ℕ) (h : n ≥ 5) :
  ∃ (count : ℕ), count = (n - 2) * (n - 2) ∧ (∀ (a₁ a₂ a₃ : ℕ), valid_set a₁ a₂ a₃ n → true) :=
by
  sorry

end num_sets_n_eq_6_num_sets_general_l267_267496


namespace common_chord_length_of_two_circles_l267_267391

noncomputable def common_chord_length (r : ℝ) : ℝ :=
  if r = 10 then 10 * Real.sqrt 3 else sorry

theorem common_chord_length_of_two_circles (r : ℝ) (h : r = 10) :
  common_chord_length r = 10 * Real.sqrt 3 :=
by
  rw [h]
  sorry

end common_chord_length_of_two_circles_l267_267391


namespace trig_identity_and_perimeter_l267_267335

theorem trig_identity_and_perimeter
  (a b c : ℝ) (A B C : ℝ)
  (h1 : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A))
  (a_val : a = 5)
  (cos_A : Real.cos A = 25 / 31) :
  (2 * a^2 = b^2 + c^2) ∧ (a + b + c = 14) :=
by
  sorry

end trig_identity_and_perimeter_l267_267335


namespace smallest_three_digit_multiple_of_17_l267_267859

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → n ≤ m) :=
by
  use 102
  have h1 : 100 ≤ 102 := by linarith
  have h2 : 102 ≤ 999 := by linarith
  have h3 : 102 % 17 = 0 := by norm_num
  have h4 : ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → 102 ≤ m := by
    intro m
    intro h
    have h_m1 := h.1
    have h_m2 := h.2.1
    have h_m3 := h.2.2
    have h_min := nat.le_of_dvd (by linarith) h_m3
    exact h_min
  exact ⟨⟨h1, h2⟩, h3, h4⟩

end smallest_three_digit_multiple_of_17_l267_267859


namespace smallest_three_digit_multiple_of_17_l267_267929

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  use 102,
  split,
  { exact le_of_lt (nat.lt_of_sub_eq_succ rfl), }, -- 102 is greater than or equal to 100
  split,
  { exact nat.lt_of_sub_eq_succ rfl, }, -- 102 is less than 1000
  split,
  { exact nat.dvd_intro_left (6) rfl, }, -- 17 divides 102
  { intros m hm1 hm2 hm3,
    exact nat.le_of_dvd hm1 (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (hm3)))), }, -- 102 is the smallest
end

end smallest_three_digit_multiple_of_17_l267_267929


namespace smallest_three_digit_multiple_of_17_l267_267844

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m := by
  sorry

end smallest_three_digit_multiple_of_17_l267_267844


namespace garden_least_cost_l267_267028

-- Define the costs per flower type
def cost_sunflower : ℝ := 0.75
def cost_tulip : ℝ := 2
def cost_marigold : ℝ := 1.25
def cost_orchid : ℝ := 4
def cost_violet : ℝ := 3.5

-- Define the areas of each section
def area_top_left : ℝ := 5 * 2
def area_bottom_left : ℝ := 5 * 5
def area_top_right : ℝ := 3 * 5
def area_bottom_right : ℝ := 3 * 4
def area_central_right : ℝ := 5 * 3

-- Calculate the total costs after assigning the most cost-effective layout
def total_cost : ℝ :=
  (area_top_left * cost_orchid) +
  (area_bottom_right * cost_violet) +
  (area_central_right * cost_tulip) +
  (area_bottom_left * cost_marigold) +
  (area_top_right * cost_sunflower)

-- Prove that the total cost is $154.50
theorem garden_least_cost : total_cost = 154.50 :=
by sorry

end garden_least_cost_l267_267028


namespace simplify_and_evaluate_expression_l267_267799

theorem simplify_and_evaluate_expression :
  (1 - 2 / (Real.tan (Real.pi / 3) - 1 + 1)) / 
  ((Real.tan (Real.pi / 3) - 1) ^ 2 - 2 * (Real.tan (Real.pi / 3) - 1) + 1) / 
  ((Real.tan (Real.pi / 3) - 1) ^ 2 - (Real.tan (Real.pi / 3) - 1)) = 
  (3 - Real.sqrt 3) / 3 :=
sorry

end simplify_and_evaluate_expression_l267_267799


namespace late_fisherman_arrival_l267_267618

-- Definitions of conditions
variables (n d : ℕ) -- n is the number of fishermen on Monday, d is the number of days the late fisherman fished
variable (total_fish : ℕ := 370)
variable (fish_per_day_per_fisherman : ℕ := 10)
variable (days_fished : ℕ := 5) -- From Monday to Friday

-- Condition in Lean: total fish caught from Monday to Friday
def total_fish_caught (n d : ℕ) := 50 * n + 10 * d

theorem late_fisherman_arrival (n d : ℕ) (h : total_fish_caught n d = 370) : 
  d = 2 :=
by
  sorry

end late_fisherman_arrival_l267_267618


namespace no_extreme_value_at_5_20_l267_267459

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := 4 * x ^ 2 - k * x - 8

theorem no_extreme_value_at_5_20 (k : ℝ) :
  ¬ (∃ (c : ℝ), (forall (x : ℝ), f k x = f k c + (4 * (x - c) ^ 2 - 8 - 20)) ∧ c = 5) ↔ (k ≤ 40 ∨ k ≥ 160) := sorry

end no_extreme_value_at_5_20_l267_267459


namespace digit_8_appears_300_times_l267_267006

-- Define a function that counts the occurrences of a specific digit in a list of numbers
def count_digit_occurrences (digit : Nat) (range : List Nat) : Nat :=
  range.foldl (λ acc n => acc + (Nat.digits 10 n).count digit) 0

-- Theorem statement: The digit 8 appears 300 times in the list of integers from 1 to 1000
theorem digit_8_appears_300_times : count_digit_occurrences 8 (List.range' 0 1000) = 300 :=
by
  sorry

end digit_8_appears_300_times_l267_267006


namespace smallest_three_digit_multiple_of_17_l267_267960

theorem smallest_three_digit_multiple_of_17 : ∃ (k : ℕ), 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 :=
by
  use 6
  split
  · exact of_as_true trivial -- 100 <= 17 * k
  split
  · exact of_as_true trivial -- 17 * k <= 999
  · exact of_as_true trivial -- 17 * k = 102
  sorry

end smallest_three_digit_multiple_of_17_l267_267960


namespace semicircle_union_shaded_area_l267_267485

noncomputable def semicircle_area (r : ℝ) : ℝ := (1 / 2) * real.pi * r^2

noncomputable def total_semicircle_area : ℝ :=
  semicircle_area 1 + semicircle_area 2 + semicircle_area 1.5

noncomputable def triangle_area_DEF : ℝ := 1.5

noncomputable def shaded_area : ℝ :=
  total_semicircle_area - triangle_area_DEF

theorem semicircle_union_shaded_area:
  shaded_area = 3.625 * real.pi - 1.5 :=
by 
  sorry

end semicircle_union_shaded_area_l267_267485


namespace alice_sold_20_pears_l267_267537

variables (S P C : ℝ)

theorem alice_sold_20_pears (h1 : C = 1.20 * P)
  (h2 : P = 0.50 * S)
  (h3 : S + P + C = 42) : S = 20 :=
by {
  -- mark the proof as incomplete with sorry
  sorry
}

end alice_sold_20_pears_l267_267537


namespace prove_minimality_of_smallest_three_digit_multiple_of_17_l267_267902

def smallest_three_digit_multiple_of_17: ℕ :=
  102

theorem prove_minimality_of_smallest_three_digit_multiple_of_17 (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k ∧ 100 ≤ 17 * k ∧ 17 * k < 1000) 
  → n = 102 :=
    by
      sorry

end prove_minimality_of_smallest_three_digit_multiple_of_17_l267_267902


namespace factorable_polynomial_l267_267113

theorem factorable_polynomial (m : ℤ) :
  (∃ A B C D E F : ℤ, 
    (A * D = 1 ∧ E + B = 4 ∧ C + F = 2 ∧ F + 3 * E + C = m + m^2 - 16)
    ∧ ((A * x + B * y + C) * (D * x + E * y + F) = x^2 + 4 * x * y + 2 * x + m * y + m^2 - 16)) ↔
  (m = 5 ∨ m = -6) :=
by
  sorry

end factorable_polynomial_l267_267113


namespace arithmetic_and_geometric_sequence_l267_267592

-- Definitions based on given conditions
def is_arithmetic_sequence (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def is_geometric_sequence (a b c d e : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c ∧ d / c = e / d

-- Main statement to prove
theorem arithmetic_and_geometric_sequence :
  ∀ (x y a b c : ℝ), 
  is_arithmetic_sequence 1 x y 4 →
  is_geometric_sequence (-2) a b c (-8) →
  (y - x) / b = -1 / 4 :=
by
  sorry

end arithmetic_and_geometric_sequence_l267_267592


namespace smallest_three_digit_multiple_of_17_l267_267861

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → n ≤ m) :=
by
  use 102
  have h1 : 100 ≤ 102 := by linarith
  have h2 : 102 ≤ 999 := by linarith
  have h3 : 102 % 17 = 0 := by norm_num
  have h4 : ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → 102 ≤ m := by
    intro m
    intro h
    have h_m1 := h.1
    have h_m2 := h.2.1
    have h_m3 := h.2.2
    have h_min := nat.le_of_dvd (by linarith) h_m3
    exact h_min
  exact ⟨⟨h1, h2⟩, h3, h4⟩

end smallest_three_digit_multiple_of_17_l267_267861


namespace repeating_block_digits_l267_267468

theorem repeating_block_digits (n d : ℕ) (h1 : n = 3) (h2 : d = 11) : 
  (∃ repeating_block_length, repeating_block_length = 2) := by
  sorry

end repeating_block_digits_l267_267468


namespace inequality_proof_l267_267764

theorem inequality_proof {x y z : ℝ} (hxy : 0 < x) (hyz : 0 < y) (hzx : 0 < z) (h : x * y + y * z + z * x = 1) :
  x * y * z * (x + y) * (y + z) * (x + z) ≥ (1 - x^2) * (1 - y^2) * (1 - z^2) :=
sorry

end inequality_proof_l267_267764


namespace mrs_hilt_baked_pecan_pies_l267_267773

def total_pies (rows : ℕ) (pies_per_row : ℕ) : ℕ :=
  rows * pies_per_row

def pecan_pies (total_pies : ℕ) (apple_pies : ℕ) : ℕ :=
  total_pies - apple_pies

theorem mrs_hilt_baked_pecan_pies :
  let apple_pies := 14
  let rows := 6
  let pies_per_row := 5
  let total := total_pies rows pies_per_row
  pecan_pies total apple_pies = 16 :=
by
  sorry

end mrs_hilt_baked_pecan_pies_l267_267773


namespace relationship_between_a_b_c_l267_267127

variable (a b c : ℝ)
variable (h_a : a = 0.4 ^ 0.2)
variable (h_b : b = 0.4 ^ 0.6)
variable (h_c : c = 2.1 ^ 0.2)

-- Prove the relationship c > a > b
theorem relationship_between_a_b_c : c > a ∧ a > b := by
  sorry

end relationship_between_a_b_c_l267_267127


namespace four_point_questions_l267_267225

theorem four_point_questions (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 4 * y = 100) : y = 10 :=
sorry

end four_point_questions_l267_267225


namespace smallest_three_digit_multiple_of_17_l267_267827

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := 
begin
  use 102,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    sorry }
end

end smallest_three_digit_multiple_of_17_l267_267827


namespace parallel_vectors_m_eq_neg3_l267_267000

-- Define vectors
def vec (x y : ℝ) : ℝ × ℝ := (x, y)
def OA : ℝ × ℝ := vec 3 (-4)
def OB : ℝ × ℝ := vec 6 (-3)
def OC (m : ℝ) : ℝ × ℝ := vec (2 * m) (m + 1)

-- Define the condition that AB is parallel to OC
def is_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

-- Calculate AB
def AB : ℝ × ℝ := (OB.1 - OA.1, OB.2 - OA.2)

-- The theorem to prove
theorem parallel_vectors_m_eq_neg3 (m : ℝ) :
  is_parallel AB (OC m) → m = -3 := by
  sorry

end parallel_vectors_m_eq_neg3_l267_267000


namespace original_bet_l267_267233

-- Define conditions and question
def payout_formula (B P : ℝ) : Prop :=
  P = (3 / 2) * B

def received_payment := 60

-- Define the Lean theorem statement
theorem original_bet (B : ℝ) (h : payout_formula B received_payment) : B = 40 :=
by
  sorry

end original_bet_l267_267233


namespace quadratic_real_roots_l267_267733

theorem quadratic_real_roots (a b c : ℝ) (h : b^2 - 4 * a * c ≥ 0) : ∃ x : ℝ, a * x^2 + b * x + c = 0 :=
sorry

end quadratic_real_roots_l267_267733


namespace problem_1_problem_2_l267_267280

noncomputable def f (a b x : ℝ) := |x + a| + |2 * x - b|

theorem problem_1 (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b)
(h_min : ∀ x, f a b x ≥ 1 ∧ (∃ x₀, f a b x₀ = 1)) :
2 * a + b = 2 :=
sorry

theorem problem_2 (a b t : ℝ) (h_a : 0 < a) (h_b : 0 < b) 
(h_tab : ∀ t > 0, a + 2 * b ≥ t * a * b)
(h_eq : 2 * a + b = 2) :
t ≤ 9 / 2 :=
sorry

end problem_1_problem_2_l267_267280


namespace prove_minimality_of_smallest_three_digit_multiple_of_17_l267_267901

def smallest_three_digit_multiple_of_17: ℕ :=
  102

theorem prove_minimality_of_smallest_three_digit_multiple_of_17 (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k ∧ 100 ≤ 17 * k ∧ 17 * k < 1000) 
  → n = 102 :=
    by
      sorry

end prove_minimality_of_smallest_three_digit_multiple_of_17_l267_267901


namespace baseball_cards_remaining_l267_267771

-- Define the number of baseball cards Mike originally had
def original_cards : ℕ := 87

-- Define the number of baseball cards Sam bought from Mike
def cards_bought : ℕ := 13

-- Prove that the remaining number of baseball cards Mike has is 74
theorem baseball_cards_remaining : original_cards - cards_bought = 74 := by
  sorry

end baseball_cards_remaining_l267_267771


namespace decrease_equation_l267_267408

-- Define the initial and final average daily written homework time
def initial_time : ℝ := 100
def final_time : ℝ := 70

-- Define the rate of decrease factor
variable (x : ℝ)

-- Define the functional relationship between initial, final time and the decrease factor
def average_homework_time (x : ℝ) : ℝ :=
  initial_time * (1 - x)^2

-- The theorem to be proved statement
theorem decrease_equation :
  average_homework_time x = final_time ↔ 100 * (1 - x) ^ 2 = 70 :=
by
  -- Proof skipped
  sorry

end decrease_equation_l267_267408


namespace hexagon_can_be_divided_into_congruent_triangles_l267_267565

section hexagon_division

-- Definitions
variables {H : Type} -- H represents the type for hexagon

-- Conditions
variables (is_hexagon : H → Prop) -- A predicate stating that a shape is a hexagon
variables (lies_on_grid : H → Prop) -- A predicate stating that the hexagon lies on the grid
variables (can_cut_along_grid_lines : H → Prop) -- A predicate stating that cuts can only be made along the grid lines
variables (identical_figures : Type u → Prop) -- A predicate stating that the obtained figures must be identical
variables (congruent_triangles : Type u → Prop) -- A predicate stating that the obtained figures are congruent triangles
variables (area_division : H → Prop) -- A predicate stating that the area of the hexagon is divided equally

-- Theorem statement
theorem hexagon_can_be_divided_into_congruent_triangles (h : H)
  (H_is_hexagon : is_hexagon h)
  (H_on_grid : lies_on_grid h)
  (H_cut : can_cut_along_grid_lines h) :
  ∃ (F : Type u), identical_figures F ∧ congruent_triangles F ∧ area_division h :=
sorry

end hexagon_division

end hexagon_can_be_divided_into_congruent_triangles_l267_267565


namespace earnings_correct_l267_267106

def phonePrice : Nat := 11
def laptopPrice : Nat := 15
def computerPrice : Nat := 18
def tabletPrice : Nat := 12
def smartwatchPrice : Nat := 8

def phoneRepairs : Nat := 9
def laptopRepairs : Nat := 5
def computerRepairs : Nat := 4
def tabletRepairs : Nat := 6
def smartwatchRepairs : Nat := 8

def totalEarnings : Nat := 
  phoneRepairs * phonePrice + 
  laptopRepairs * laptopPrice + 
  computerRepairs * computerPrice + 
  tabletRepairs * tabletPrice + 
  smartwatchRepairs * smartwatchPrice

theorem earnings_correct : totalEarnings = 382 := by
  sorry

end earnings_correct_l267_267106


namespace smallest_three_digit_multiple_of_17_l267_267876

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := sorry

end smallest_three_digit_multiple_of_17_l267_267876


namespace solve_xyz_sum_l267_267804

theorem solve_xyz_sum :
  ∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ (x+y+z)^3 - x^3 - y^3 - z^3 = 378 ∧ x+y+z = 9 :=
by
  sorry

end solve_xyz_sum_l267_267804


namespace unique_solution_exists_l267_267228

theorem unique_solution_exists :
  ∃ (x y : ℝ), x = -13 / 96 ∧ y = 13 / 40 ∧
    (x / Real.sqrt (x^2 + y^2) - 1/x = 7) ∧
    (y / Real.sqrt (x^2 + y^2) + 1/y = 4) :=
by
  sorry

end unique_solution_exists_l267_267228


namespace income_on_first_day_l267_267682

theorem income_on_first_day (income : ℕ → ℚ) (h1 : income 10 = 18)
  (h2 : ∀ n, income (n + 1) = 2 * income n) :
  income 1 = 0.03515625 :=
by
  sorry

end income_on_first_day_l267_267682


namespace min_value_quadratic_l267_267462

noncomputable def quadratic_min (a c : ℝ) : ℝ :=
  (2 / a) + (2 / c)

theorem min_value_quadratic {a c : ℝ} (ha : a > 0) (hc : c > 0) (hac : a * c = 1/4) : 
  quadratic_min a c = 8 :=
sorry

end min_value_quadratic_l267_267462


namespace smallest_three_digit_multiple_of_17_correct_l267_267922

def smallest_three_digit_multiple_of_17 : ℕ :=
  102

theorem smallest_three_digit_multiple_of_17_correct :
  ∀ n, n >= 100 → n < 1000 → n % 17 = 0 → 102 ≤ n :=
begin
  intros n h1 h2 h3,
  sorry,
end

end smallest_three_digit_multiple_of_17_correct_l267_267922


namespace half_angle_in_first_quadrant_l267_267716

theorem half_angle_in_first_quadrant (α : ℝ) (h : 0 < α ∧ α < π / 2) : 0 < α / 2 ∧ α / 2 < π / 4 :=
by
  sorry

end half_angle_in_first_quadrant_l267_267716


namespace price_per_glass_on_first_day_eq_half_l267_267619

structure OrangeadeProblem where
  O : ℝ
  W : ℝ
  P1 : ℝ
  P2 : ℝ
  W_eq_O : W = O
  P2_value : P2 = 0.3333333333333333
  revenue_eq : 2 * O * P1 = 3 * O * P2

theorem price_per_glass_on_first_day_eq_half (prob : OrangeadeProblem) : prob.P1 = 0.50 := 
by
  sorry

end price_per_glass_on_first_day_eq_half_l267_267619


namespace second_rate_of_return_l267_267117

namespace Investment

def total_investment : ℝ := 33000
def interest_total : ℝ := 970
def investment_4_percent : ℝ := 13000
def interest_rate_4_percent : ℝ := 0.04

def amount_second_investment : ℝ := total_investment - investment_4_percent
def interest_from_first_part : ℝ := interest_rate_4_percent * investment_4_percent
def interest_from_second_part (R : ℝ) : ℝ := R * amount_second_investment

theorem second_rate_of_return : (∃ R : ℝ, interest_from_first_part + interest_from_second_part R = interest_total) → 
  R = 0.0225 :=
by
  intro h
  sorry

end Investment

end second_rate_of_return_l267_267117


namespace simplify_and_evaluate_expression_l267_267801

theorem simplify_and_evaluate_expression :
  (1 - 2 / (Real.tan (Real.pi / 3) - 1 + 1)) / 
  ((Real.tan (Real.pi / 3) - 1) ^ 2 - 2 * (Real.tan (Real.pi / 3) - 1) + 1) / 
  ((Real.tan (Real.pi / 3) - 1) ^ 2 - (Real.tan (Real.pi / 3) - 1)) = 
  (3 - Real.sqrt 3) / 3 :=
sorry

end simplify_and_evaluate_expression_l267_267801


namespace calculate_si_l267_267267

section SimpleInterest

def Principal : ℝ := 10000
def Rate : ℝ := 0.04
def Time : ℝ := 1
def SimpleInterest : ℝ := Principal * Rate * Time

theorem calculate_si : SimpleInterest = 400 := by
  -- Proof goes here.
  sorry

end SimpleInterest

end calculate_si_l267_267267


namespace finite_quadruples_n_factorial_l267_267625

theorem finite_quadruples_n_factorial (n a b c : ℕ) (h_pos : 0 < n) (h_cond : n! = a^(n-1) + b^(n-1) + c^(n-1)) : n ≤ 100 :=
by
  sorry

end finite_quadruples_n_factorial_l267_267625


namespace amount_pop_spend_l267_267180

theorem amount_pop_spend
  (total_spent : ℝ)
  (ratio_snap_crackle : ℝ)
  (ratio_crackle_pop : ℝ)
  (spending_eq : total_spent = 150)
  (snap_crackle : ratio_snap_crackle = 2)
  (crackle_pop : ratio_crackle_pop = 3)
  (snap : ℝ)
  (crackle : ℝ)
  (pop : ℝ)
  (snap_eq : snap = ratio_snap_crackle * crackle)
  (crackle_eq : crackle = ratio_crackle_pop * pop)
  (total_eq : snap + crackle + pop = total_spent) :
  pop = 15 := 
by
  sorry

end amount_pop_spend_l267_267180


namespace correct_alarm_clock_time_l267_267775

-- Definitions for the conditions
def alarm_set_time : ℕ := 7 * 60 -- in minutes
def museum_arrival_time : ℕ := 8 * 60 + 50 -- in minutes
def museum_touring_time : ℕ := 1 * 60 + 30 -- in minutes
def alarm_home_time : ℕ := 11 * 60 + 50 -- in minutes

-- The problem: proving the correct time the clock should be set to
theorem correct_alarm_clock_time : 
  (alarm_home_time - (2 * ((museum_arrival_time - alarm_set_time) + museum_touring_time / 2)) = 12 * 60) :=
  by
    sorry

end correct_alarm_clock_time_l267_267775


namespace certain_amount_l267_267734

theorem certain_amount (x : ℝ) (A : ℝ) (h1: x = 900) (h2: 0.25 * x = 0.15 * 1600 - A) : A = 15 :=
by
  sorry

end certain_amount_l267_267734


namespace abs_add_three_eq_two_l267_267008

theorem abs_add_three_eq_two (a : ℝ) (h : a = -1) : |a + 3| = 2 :=
by
  rw [h]
  sorry

end abs_add_three_eq_two_l267_267008


namespace solution_set_of_inequality_l267_267049

theorem solution_set_of_inequality :
  ∀ x : ℝ, (x > -1 ∧ x < 1) ↔ (x^2 - 1 < 0) :=
by
  sorry

end solution_set_of_inequality_l267_267049


namespace smallest_three_digit_multiple_of_17_l267_267955

theorem smallest_three_digit_multiple_of_17 : ∃ (k : ℕ), 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 :=
by
  use 6
  split
  · exact of_as_true trivial -- 100 <= 17 * k
  split
  · exact of_as_true trivial -- 17 * k <= 999
  · exact of_as_true trivial -- 17 * k = 102
  sorry

end smallest_three_digit_multiple_of_17_l267_267955


namespace ms_hatcher_total_students_l267_267616

theorem ms_hatcher_total_students :
  let third_graders := 20
  let fourth_graders := 2 * third_graders
  let fifth_graders := third_graders / 2
  third_graders + fourth_graders + fifth_graders = 70 :=
by 
  let third_graders := 20
  let fourth_graders := 2 * third_graders
  let fifth_graders := third_graders / 2
  show third_graders + fourth_graders + fifth_graders = 70
  sorry

end ms_hatcher_total_students_l267_267616


namespace smallest_triangle_perimeter_l267_267213

theorem smallest_triangle_perimeter : 
  ∀ (a b c : ℕ), 
    (2 ∣ a ∧ 2 ∣ b ∧ 2 ∣ c) ∧ (a = b - 2 ∨ a = b + 2) ∧ (b = c - 2 ∨ b = c + 2) ∧ (a + b > c) ∧ (a + c > b) ∧ (b + c > a) 
    → a + b + c = 12 := 
  sorry

end smallest_triangle_perimeter_l267_267213


namespace number_of_rectangles_l267_267375

open Real Set

-- Given points A, B, C, D on a line L and a length k
variables {A B C D : ℝ} (L : Set ℝ) (k : ℝ)

-- The points are distinct and ordered on the line
axiom h1 : A ≠ B ∧ B ≠ C ∧ C ≠ D
axiom h2 : A < B ∧ B < C ∧ C < D

-- We need to show there are two rectangles with certain properties
theorem number_of_rectangles : 
  (∃ (rect1 rect2 : Set ℝ), 
    rect1 ≠ rect2 ∧ 
    (∃ (a1 b1 c1 d1 : ℝ), rect1 = {a1, b1, c1, d1} ∧ 
      a1 < b1 ∧ b1 < c1 ∧ c1 < d1 ∧ 
      (d1 - c1 = k ∨ c1 - b1 = k)) ∧ 
    (∃ (a2 b2 c2 d2 : ℝ), rect2 = {a2, b2, c2, d2} ∧ 
      a2 < b2 ∧ b2 < c2 ∧ c2 < d2 ∧ 
      (d2 - c2 = k ∨ c2 - b2 = k))
  ) :=
sorry

end number_of_rectangles_l267_267375


namespace average_inside_time_l267_267162

def jonsey_awake_hours := 24 * (2/3)
def jonsey_inside_fraction := 1 - (1/2)
def jonsey_inside_hours := jonsey_awake_hours * jonsey_inside_fraction

def riley_awake_hours := 24 * (3/4)
def riley_inside_fraction := 1 - (1/3)
def riley_inside_hours := riley_awake_hours * riley_inside_fraction

def total_inside_hours := jonsey_inside_hours + riley_inside_hours
def number_of_people := 2
def average_inside_hours := total_inside_hours / number_of_people

theorem average_inside_time (jonsey_awake_hrs : ℝ) (jonsey_inside_frac : ℝ) 
  (jonsey_inside_hrs : ℝ) (riley_awake_hrs : ℝ) (riley_inside_frac : ℝ) 
  (riley_inside_hrs : ℝ) (total_inside_hrs : ℝ) (num_people : ℝ) 
  (avg_inside_hrs : ℝ) :
  jonsey_awake_hrs = 24 * (2 / 3) → 
  jonsey_inside_frac = 1 - (1 / 2) →
  jonsey_inside_hrs = jonsey_awake_hrs * jonsey_inside_frac →
  riley_awake_hrs = 24 * (3 / 4) →
  riley_inside_frac = 1 - (1 / 3) →
  riley_inside_hrs = riley_awake_hrs * riley_inside_frac →
  total_inside_hrs = jonsey_inside_hrs + riley_inside_hrs →
  num_people = 2 →
  avg_inside_hrs = total_inside_hrs / num_people →
  avg_inside_hrs = 10 := 
by
  intros
  sorry

end average_inside_time_l267_267162


namespace max_value_of_4x_plus_3y_l267_267708

theorem max_value_of_4x_plus_3y (x y : ℝ) (h : x^2 + y^2 = 18 * x + 8 * y + 10) :
  4 * x + 3 * y ≤ 45 :=
sorry

end max_value_of_4x_plus_3y_l267_267708


namespace smallest_three_digit_multiple_of_17_l267_267939

/--
Prove that the smallest three-digit multiple of 17 is 102.
-/
theorem smallest_three_digit_multiple_of_17 : ∃ k : ℤ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 := 
by
  sorry

end smallest_three_digit_multiple_of_17_l267_267939


namespace part_1_part_2_l267_267445

-- Definitions based on given conditions
def a : ℕ → ℝ := λ n => 2 * n + 1
noncomputable def b : ℕ → ℝ := λ n => 1 / ((2 * n + 1)^2 - 1)
noncomputable def S : ℕ → ℝ := λ n => n ^ 2 + 2 * n
noncomputable def T : ℕ → ℝ := λ n => n / (4 * (n + 1))

-- Lean statement for proving the problem
theorem part_1 (n : ℕ) :
  ∀ a_3 a_5 a_7 : ℝ, 
  a 3 = a_3 → 
  a_3 = 7 →
  a_5 = a 5 →
  a_7 = a 7 →
  a_5 + a_7 = 26 →
  ∃ a_1 d : ℝ,
    (a 1 = a_1 + 0 * d) ∧
    (a 2 = a_1 + 1 * d) ∧
    (a 3 = a_1 + 2 * d) ∧
    (a 4 = a_1 + 3 * d) ∧
    (a 5 = a_1 + 4 * d) ∧
    (a 7 = a_1 + 6 * d) ∧
    (a n = a_1 + (n - 1) * d) ∧
    (S n = n^2 + 2*n) := sorry

theorem part_2 (n : ℕ) :
  ∀ a_n b_n : ℝ,
  b n = b_n →
  a n = a_n →
  1 / b n = a_n^2 - 1 →
  T n = τ →
  (T n = n / (4 * (n + 1))) := sorry

end part_1_part_2_l267_267445


namespace gain_percentage_l267_267979

theorem gain_percentage (selling_price gain : ℝ) (h1 : selling_price = 225) (h2 : gain = 75) : 
  (gain / (selling_price - gain) * 100) = 50 :=
by
  sorry

end gain_percentage_l267_267979


namespace cube_root_of_sum_l267_267361

def a := 25
def b := 30
def c := 35

theorem cube_root_of_sum :
  Real.cbrt (a^3 + b^3 + c^3) = 5 * Real.cbrt 684 :=
by
  -- identify the common factor
  sorry

end cube_root_of_sum_l267_267361


namespace min_elements_in_AS_l267_267022

theorem min_elements_in_AS (n : ℕ) (h : n ≥ 2) (S : Finset ℝ) (h_card : S.card = n) :
  ∃ (A_S : Finset ℝ), ∀ T : Finset ℝ, (∀ a b : ℝ, a ≠ b → a ∈ S → b ∈ S → (a + b) / 2 ∈ T) → 
  T.card ≥ 2 * n - 3 :=
sorry

end min_elements_in_AS_l267_267022


namespace dog_food_duration_l267_267098

-- Definitions for the given conditions
def number_of_dogs : ℕ := 4
def meals_per_day : ℕ := 2
def grams_per_meal : ℕ := 250
def sacks_of_food : ℕ := 2
def kilograms_per_sack : ℝ := 50
def grams_per_kilogram : ℝ := 1000

-- Lean statement to prove the correct answer
theorem dog_food_duration : 
  ((number_of_dogs * meals_per_day * grams_per_meal / grams_per_kilogram) * sacks_of_food * kilograms_per_sack) / 
  (number_of_dogs * meals_per_day * grams_per_meal / grams_per_kilogram) = 50 :=
by 
  simp only [number_of_dogs, meals_per_day, grams_per_meal, sacks_of_food, kilograms_per_sack, grams_per_kilogram]
  norm_num
  sorry

end dog_food_duration_l267_267098


namespace carlos_marbles_l267_267995

theorem carlos_marbles:
  ∃ M, M > 1 ∧ 
       M % 5 = 1 ∧ 
       M % 7 = 1 ∧ 
       M % 11 = 1 ∧ 
       M % 4 = 2 ∧ 
       M = 386 := by
  sorry

end carlos_marbles_l267_267995


namespace average_of_last_three_l267_267632

theorem average_of_last_three (A B C D : ℝ) 
  (h1 : (A + B + C) / 3 = 6)
  (h2 : A + D = 11)
  (h3 : D = 4) : 
  (B + C + D) / 3 = 5 :=
by
  sorry

end average_of_last_three_l267_267632


namespace probability_of_snow_at_least_once_l267_267620

/-- Probability of snow during the first week of January -/
theorem probability_of_snow_at_least_once :
  let p_no_snow_first_four_days := (3/4 : ℚ)
  let p_no_snow_next_three_days := (2/3 : ℚ)
  let p_no_snow_first_week := p_no_snow_first_four_days^4 * p_no_snow_next_three_days^3
  let p_snow_at_least_once := 1 - p_no_snow_first_week
  p_snow_at_least_once = 68359 / 100000 :=
by
  sorry

end probability_of_snow_at_least_once_l267_267620


namespace half_angle_in_first_or_third_quadrant_l267_267713

noncomputable 
def angle_in_first_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi < α ∧ α < (2 * k + 1) * Real.pi / 2

noncomputable 
def angle_in_first_or_third_quadrant (β : ℝ) : Prop :=
  ∃ k : ℤ, k * Real.pi < β ∧ β < (k + 1/4) * Real.pi ∨
  ∃ i : ℤ, (2 * i + 1) * Real.pi < β ∧ β < (2 * i + 5/4) * Real.pi 

theorem half_angle_in_first_or_third_quadrant (α : ℝ) (h : angle_in_first_quadrant α) :
  angle_in_first_or_third_quadrant (α / 2) :=
  sorry

end half_angle_in_first_or_third_quadrant_l267_267713


namespace smallest_three_digit_multiple_of_17_l267_267860

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → n ≤ m) :=
by
  use 102
  have h1 : 100 ≤ 102 := by linarith
  have h2 : 102 ≤ 999 := by linarith
  have h3 : 102 % 17 = 0 := by norm_num
  have h4 : ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → 102 ≤ m := by
    intro m
    intro h
    have h_m1 := h.1
    have h_m2 := h.2.1
    have h_m3 := h.2.2
    have h_min := nat.le_of_dvd (by linarith) h_m3
    exact h_min
  exact ⟨⟨h1, h2⟩, h3, h4⟩

end smallest_three_digit_multiple_of_17_l267_267860


namespace calculation_result_l267_267655

def initial_number : ℕ := 15
def subtracted_value : ℕ := 2
def added_value : ℕ := 4
def divisor : ℕ := 1
def second_divisor : ℕ := 2
def multiplier : ℕ := 8

theorem calculation_result : 
  (initial_number - subtracted_value + (added_value / divisor : ℕ)) / second_divisor * multiplier = 68 :=
by
  sorry

end calculation_result_l267_267655


namespace mashed_potatoes_count_l267_267363

theorem mashed_potatoes_count :
  ∀ (b s : ℕ), b = 489 → b = s + 10 → s = 479 :=
by
  intros b s h₁ h₂
  sorry

end mashed_potatoes_count_l267_267363


namespace smallest_positive_integer_x_l267_267657

def smallest_x (x : ℕ) : Prop :=
  x > 0 ∧ (450 * x) % 625 = 0

theorem smallest_positive_integer_x :
  ∃ x : ℕ, smallest_x x ∧ ∀ y : ℕ, smallest_x y → x ≤ y ∧ x = 25 :=
by {
  sorry
}

end smallest_positive_integer_x_l267_267657


namespace allie_carl_product_points_l267_267249

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 3
  else 0

def allie_rolls : List ℕ := [2, 6, 3, 2, 5]
def carl_rolls : List ℕ := [1, 4, 3, 6, 6]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.foldr (λ x acc => g x + acc) 0

theorem allie_carl_product_points : (total_points allie_rolls) * (total_points carl_rolls) = 594 :=
  sorry

end allie_carl_product_points_l267_267249


namespace circle_tangent_values_l267_267522

theorem circle_tangent_values (m : ℝ) :
  (∀ x y : ℝ, ((x - m)^2 + (y + 2)^2 = 9) → ((x + 1)^2 + (y - m)^2 = 4)) → 
  m = 2 ∨ m = -5 :=
by
  sorry

end circle_tangent_values_l267_267522


namespace probability_snow_first_week_l267_267623

theorem probability_snow_first_week :
  let p1 := 1/4
  let p2 := 1/3
  let no_snow := (3/4)^4 * (2/3)^3
  let snows_at_least_once := 1 - no_snow
  snows_at_least_once = 29 / 32 := by
  sorry

end probability_snow_first_week_l267_267623


namespace problem1_problem2_l267_267032

-- Problem 1: Remainder of 2011-digit number with each digit 2 when divided by 9 is 8

theorem problem1 : (4022 % 9 = 8) := by
  sorry

-- Problem 2: Remainder of n-digit number with each digit 7 when divided by 9 and n % 9 = 3 is 3

theorem problem2 (n : ℕ) (h : n % 9 = 3) : ((7 * n) % 9 = 3) := by
  sorry

end problem1_problem2_l267_267032


namespace smallest_three_digit_multiple_of_17_l267_267835

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := 
begin
  use 102,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    sorry }
end

end smallest_three_digit_multiple_of_17_l267_267835


namespace arithmetic_mean_value_of_x_l267_267513

theorem arithmetic_mean_value_of_x (x : ℝ) (h : (x + 10 + 20 + 3 * x + 16 + 3 * x + 6) / 5 = 30) : x = 14 := 
by 
  sorry

end arithmetic_mean_value_of_x_l267_267513


namespace determine_radius_of_semicircle_l267_267192

noncomputable def radius_of_semicircle (P : ℝ) : ℝ :=
  P / (Real.pi + 2)

theorem determine_radius_of_semicircle :
  radius_of_semicircle 32.392033717615696 = 6.3 :=
by
  sorry

end determine_radius_of_semicircle_l267_267192


namespace functional_eq_is_linear_l267_267568

theorem functional_eq_is_linear (f : ℚ → ℚ)
  (h : ∀ x y : ℝ, f ((x + y) / 2) = (f x / 2) + (f y / 2)) : ∃ k : ℚ, ∀ x : ℚ, f x = k * x :=
by
  sorry

end functional_eq_is_linear_l267_267568


namespace find_goods_train_speed_l267_267432

-- Definition of given conditions
def speed_of_man_train_kmph : ℝ := 120
def time_goods_train_seconds : ℝ := 9
def length_goods_train_meters : ℝ := 350

-- The proof statement
theorem find_goods_train_speed :
  let relative_speed_mps := (speed_of_man_train_kmph + goods_train_speed_kmph) * (5 / 18)
  ∃ (goods_train_speed_kmph : ℝ), relative_speed_mps = length_goods_train_meters / time_goods_train_seconds ∧ goods_train_speed_kmph = 20 :=
by {
  sorry
}

end find_goods_train_speed_l267_267432


namespace ribbon_per_box_l267_267321

theorem ribbon_per_box (total_ribbon : ℚ) (total_boxes : ℕ) (same_ribbon_each_box : ℚ) 
  (h1 : total_ribbon = 5/12) (h2 : total_boxes = 5) : 
  same_ribbon_each_box = 1/12 :=
sorry

end ribbon_per_box_l267_267321


namespace smallest_number_divisible_l267_267527

theorem smallest_number_divisible (x y : ℕ) (h : x + y = 4728) 
  (h1 : (x + y) % 27 = 0) 
  (h2 : (x + y) % 35 = 0) 
  (h3 : (x + y) % 25 = 0) 
  (h4 : (x + y) % 21 = 0) : 
  x = 4725 := by 
  sorry

end smallest_number_divisible_l267_267527


namespace completing_the_square_l267_267972

theorem completing_the_square (x : ℝ) : (x^2 - 2 * x - 5 = 0) → ((x - 1)^2 = 6) :=
by
  sorry

end completing_the_square_l267_267972


namespace part_one_part_two_l267_267330

theorem part_one (A B C a b c : ℝ) (hA : A + B + C = π) 
  (h1 : sin(C) * sin(A - B) = sin(B) * sin(C - A)) : 
  2 * a^2 = b^2 + c^2 := by
  -- Placeholder for the detailed proof
  sorry

theorem part_two (a b c : ℝ) 
  (h1 : 2 * a^2 = b^2 + c^2) (h2 : a = 5) (h3 : cos(acos (5 / 31)) = 25 / 31) : 
  a + b + c = 14 := by
  -- Placeholder for the detailed proof
  sorry

end part_one_part_two_l267_267330


namespace james_total_points_l267_267311

def f : ℕ := 13
def s : ℕ := 20
def p_f : ℕ := 3
def p_s : ℕ := 2

def total_points : ℕ := (f * p_f) + (s * p_s)

theorem james_total_points : total_points = 79 := 
by
  -- Proof would go here.
  sorry

end james_total_points_l267_267311


namespace ms_hatcher_total_students_l267_267615

theorem ms_hatcher_total_students :
  let third_graders := 20
  let fourth_graders := 2 * third_graders
  let fifth_graders := third_graders / 2
  third_graders + fourth_graders + fifth_graders = 70 :=
by 
  let third_graders := 20
  let fourth_graders := 2 * third_graders
  let fifth_graders := third_graders / 2
  show third_graders + fourth_graders + fifth_graders = 70
  sorry

end ms_hatcher_total_students_l267_267615


namespace smallest_three_digit_multiple_of_17_l267_267830

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := 
begin
  use 102,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    sorry }
end

end smallest_three_digit_multiple_of_17_l267_267830


namespace correct_model_l267_267395

def average_homework_time_decrease (x : ℝ) : Prop :=
  100 * (1 - x) ^ 2 = 70

theorem correct_model (x : ℝ) : average_homework_time_decrease x := 
  sorry

end correct_model_l267_267395


namespace consecutive_page_sum_l267_267045

theorem consecutive_page_sum (n : ℕ) (h : n * (n + 1) * (n + 2) = 479160) : n + (n + 1) + (n + 2) = 234 :=
sorry

end consecutive_page_sum_l267_267045


namespace complement_set_unique_l267_267465

-- Define the universal set U
def U : Set ℕ := {1,2,3,4,5,6,7,8}

-- Define the complement of B with respect to U
def complement_B : Set ℕ := {1,3}

-- The set B that we need to prove
def B : Set ℕ := {2,4,5,6,7,8}

-- State that B is the set with the given complement in U
theorem complement_set_unique (U : Set ℕ) (complement_B : Set ℕ) :
    (U \ complement_B = {2,4,5,6,7,8}) :=
by
    -- We need to prove B is the set {2,4,5,6,7,8}
    sorry

end complement_set_unique_l267_267465


namespace geometric_series_r_l267_267515

theorem geometric_series_r (a r : ℝ) 
    (h1 : a * (1 - r ^ 0) / (1 - r) = 24) 
    (h2 : a * r / (1 - r ^ 2) = 8) : 
    r = 1 / 2 := 
sorry

end geometric_series_r_l267_267515


namespace smallest_three_digit_multiple_of_17_l267_267953

theorem smallest_three_digit_multiple_of_17 : ∃ (k : ℕ), 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 :=
by
  use 6
  split
  · exact of_as_true trivial -- 100 <= 17 * k
  split
  · exact of_as_true trivial -- 17 * k <= 999
  · exact of_as_true trivial -- 17 * k = 102
  sorry

end smallest_three_digit_multiple_of_17_l267_267953


namespace baggies_of_oatmeal_cookies_l267_267500

theorem baggies_of_oatmeal_cookies (total_cookies : ℝ) (chocolate_chip_cookies : ℝ) (cookies_per_baggie : ℝ) 
(h_total : total_cookies = 41)
(h_choc : chocolate_chip_cookies = 13)
(h_baggie : cookies_per_baggie = 9) : 
  ⌊(total_cookies - chocolate_chip_cookies) / cookies_per_baggie⌋ = 3 := 
by 
  sorry

end baggies_of_oatmeal_cookies_l267_267500


namespace negation_of_forall_exp_gt_zero_l267_267043

open Real

theorem negation_of_forall_exp_gt_zero : 
  (¬ (∀ x : ℝ, exp x > 0)) ↔ (∃ x : ℝ, exp x ≤ 0) :=
by
  sorry

end negation_of_forall_exp_gt_zero_l267_267043


namespace smallest_three_digit_multiple_of_17_l267_267882

theorem smallest_three_digit_multiple_of_17 : 
    ∃ (n : ℤ), n ≥ 100 ∧ n < 1000 ∧ (∃ (k : ℤ), n = 17 * k) ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l267_267882


namespace min_chord_length_m_l267_267809

-- Definition of the circle and the line
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 6 * y + 4 = 0
def line_eq (m x y : ℝ) : Prop := m * x - y + 1 = 0

-- Theorem statement: value of m that minimizes the length of the chord
theorem min_chord_length_m (m : ℝ) : m = 1 ↔
  ∃ x y : ℝ, circle_eq x y ∧ line_eq m x y := sorry

end min_chord_length_m_l267_267809


namespace sum_of_x_and_y_l267_267578

theorem sum_of_x_and_y (x y : ℤ) (h : 2 * x * y + x + y = 83) : x + y = 83 ∨ x + y = -85 := 
sorry

end sum_of_x_and_y_l267_267578


namespace smallest_repeating_block_fraction_3_over_11_l267_267472

theorem smallest_repeating_block_fraction_3_over_11 :
  ∃ n : ℕ, (∃ d : ℕ, (3/11 : ℚ) = d / 10^n) ∧ n = 2 :=
sorry

end smallest_repeating_block_fraction_3_over_11_l267_267472


namespace doughnuts_in_shop_l267_267383

def ratio_of_doughnuts_to_muffins : Nat := 5

def number_of_muffins_in_shop : Nat := 10

def number_of_doughnuts (D M : Nat) : Prop :=
  D = ratio_of_doughnuts_to_muffins * M

theorem doughnuts_in_shop :
  number_of_doughnuts D number_of_muffins_in_shop → D = 50 :=
by
  sorry

end doughnuts_in_shop_l267_267383


namespace hypotenuse_45_45_90_l267_267502

theorem hypotenuse_45_45_90 (leg : ℝ) (h_leg : leg = 10) (angle : ℝ) (h_angle : angle = 45) :
  ∃ hypotenuse : ℝ, hypotenuse = leg * Real.sqrt 2 :=
by
  use 10 * Real.sqrt 2
  sorry

end hypotenuse_45_45_90_l267_267502


namespace prove_minimality_of_smallest_three_digit_multiple_of_17_l267_267903

def smallest_three_digit_multiple_of_17: ℕ :=
  102

theorem prove_minimality_of_smallest_three_digit_multiple_of_17 (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k ∧ 100 ≤ 17 * k ∧ 17 * k < 1000) 
  → n = 102 :=
    by
      sorry

end prove_minimality_of_smallest_three_digit_multiple_of_17_l267_267903


namespace paint_cost_contribution_l267_267757

theorem paint_cost_contribution
  (paint_cost_per_gallon : ℕ) 
  (coverage_per_gallon : ℕ) 
  (total_wall_area : ℕ) 
  (two_coats : ℕ) 
  : paint_cost_per_gallon = 45 → coverage_per_gallon = 400 → total_wall_area = 1600 → two_coats = 2 → 
    ((total_wall_area / coverage_per_gallon) * two_coats * paint_cost_per_gallon) / 2 = 180 :=
by
  intros h1 h2 h3 h4
  sorry

end paint_cost_contribution_l267_267757


namespace average_speed_l267_267696

theorem average_speed 
  (total_distance : ℝ) (total_time : ℝ) 
  (h_distance : total_distance = 26) (h_time : total_time = 4) :
  (total_distance / total_time) = 6.5 :=
by
  rw [h_distance, h_time]
  norm_num

end average_speed_l267_267696


namespace integer_part_of_sum_l267_267326

theorem integer_part_of_sum :
  let S := 1 + ∑ n in finset.range 99, 1 / real.sqrt (n.succ.succ) in
  (∃ (H : ∀ n : ℕ, 1 ≤ n → real.sqrt n < 0.5 * (real.sqrt n + real.sqrt (n+1)) < real.sqrt (n + 1)), 
  ∀ S : ℝ, S = 1 + ∑ n in finset.range 99, 1 / real.sqrt (n.succ.succ) → ⌊S⌋ = 18 :=
begin
  sorry
end

end integer_part_of_sum_l267_267326


namespace count_valid_pairs_is_7_l267_267142

def valid_pairs_count : Nat :=
  let pairs := [(1, 1), (1, 2), (2, 1), (2, 2), (3, 2), (3, 3), (4, 4)]
  List.length pairs

theorem count_valid_pairs_is_7 (b c : ℕ) (hb : b > 0) (hc : c > 0) :
  (b^2 - 4 * c ≤ 0) → (c^2 - 4 * b ≤ 0) → valid_pairs_count = 7 :=
by
  sorry

end count_valid_pairs_is_7_l267_267142


namespace repeating_block_digits_l267_267466

theorem repeating_block_digits (n d : ℕ) (h1 : n = 3) (h2 : d = 11) : 
  (∃ repeating_block_length, repeating_block_length = 2) := by
  sorry

end repeating_block_digits_l267_267466


namespace proportion_solution_l267_267148

-- Define the given proportion condition as a hypothesis
variable (x : ℝ)

-- The definition is derived directly from the given problem
def proportion_condition : Prop := x / 5 = 1.2 / 8

-- State the theorem using the given proportion condition to prove x = 0.75
theorem proportion_solution (h : proportion_condition x) : x = 0.75 :=
  by
    sorry

end proportion_solution_l267_267148


namespace smallest_three_digit_multiple_of_17_l267_267958

theorem smallest_three_digit_multiple_of_17 : ∃ (k : ℕ), 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 :=
by
  use 6
  split
  · exact of_as_true trivial -- 100 <= 17 * k
  split
  · exact of_as_true trivial -- 17 * k <= 999
  · exact of_as_true trivial -- 17 * k = 102
  sorry

end smallest_three_digit_multiple_of_17_l267_267958


namespace find_m_value_l267_267729

open Real

-- Define the vectors a and b as specified in the problem
def vec_a (m : ℝ) : ℝ × ℝ := (1, m)
def vec_b : ℝ × ℝ := (3, -2)

-- Define the sum of vectors a and b
def vec_sum (m : ℝ) : ℝ × ℝ := (1 + 3, m - 2)

-- Define the dot product of the vector sum with vector b to be zero as the given condition
def dot_product (m : ℝ) : ℝ := (vec_sum m).1 * vec_b.1 + (vec_sum m).2 * vec_b.2

-- The theorem to prove that given the defined conditions, m equals 8
theorem find_m_value (m : ℝ) (h : dot_product m = 0) : m = 8 := by
  sorry

end find_m_value_l267_267729


namespace integer_a_for_factoring_l267_267122

theorem integer_a_for_factoring (a : ℤ) :
  (∃ c d : ℤ, (x - a) * (x - 10) + 1 = (x + c) * (x + d)) → (a = 8 ∨ a = 12) :=
by
  sorry

end integer_a_for_factoring_l267_267122


namespace total_population_l267_267595

theorem total_population (b g t : ℕ) (h1 : b = 2 * g) (h2 : g = 4 * t) : b + g + t = 13 * t :=
by
  sorry

end total_population_l267_267595


namespace total_vegetables_l267_267054

theorem total_vegetables (b k r : ℕ) (broccoli_weight_kg : ℝ) (broccoli_weight_g : ℝ) 
  (kohlrabi_mult : ℕ) (radish_mult : ℕ) :
  broccoli_weight_kg = 5 ∧ 
  broccoli_weight_g = 0.25 ∧ 
  kohlrabi_mult = 4 ∧ 
  radish_mult = 3 ∧ 
  b = broccoli_weight_kg / broccoli_weight_g ∧ 
  k = kohlrabi_mult * b ∧ 
  r = radish_mult * k →
  b + k + r = 340 := 
by
  sorry

end total_vegetables_l267_267054


namespace prime_b_plus_1_l267_267488

def is_a_good (a b : ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → a * n ≥ b → (Nat.choose (a * n) b - 1) % (a * n + 1) = 0

theorem prime_b_plus_1 (a b : ℕ) (h1 : is_a_good a b) (h2 : ¬ is_a_good a (b + 2)) : Nat.Prime (b + 1) :=
by
  sorry

end prime_b_plus_1_l267_267488


namespace optimal_discount_order_l267_267189

variables (p : ℝ) (d1 : ℝ) (d2 : ℝ)

-- Original price of "Stars Beyond" is 30 dollars
def original_price : ℝ := 30

-- Fixed discount is 5 dollars
def discount_5 : ℝ := 5

-- 25% discount represented as a multiplier
def discount_25 : ℝ := 0.75

-- Applying $5 discount first and then 25% discount
def price_after_5_then_25_discount := discount_25 * (original_price - discount_5)

-- Applying 25% discount first and then $5 discount
def price_after_25_then_5_discount := (discount_25 * original_price) - discount_5

-- The additional savings when applying 25% discount first
def additional_savings := price_after_5_then_25_discount - price_after_25_then_5_discount

theorem optimal_discount_order : 
  additional_savings = 1.25 :=
sorry

end optimal_discount_order_l267_267189


namespace vec_op_l267_267125

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (2, -2)
def two_a : ℝ × ℝ := (2 * 2, 2 * 1)
def result : ℝ × ℝ := (two_a.1 - b.1, two_a.2 - b.2)

theorem vec_op : (2 * a.1 - b.1, 2 * a.2 - b.2) = (2, 4) := by
  sorry

end vec_op_l267_267125


namespace dog_food_duration_l267_267099

-- Definitions for the given conditions
def number_of_dogs : ℕ := 4
def meals_per_day : ℕ := 2
def grams_per_meal : ℕ := 250
def sacks_of_food : ℕ := 2
def kilograms_per_sack : ℝ := 50
def grams_per_kilogram : ℝ := 1000

-- Lean statement to prove the correct answer
theorem dog_food_duration : 
  ((number_of_dogs * meals_per_day * grams_per_meal / grams_per_kilogram) * sacks_of_food * kilograms_per_sack) / 
  (number_of_dogs * meals_per_day * grams_per_meal / grams_per_kilogram) = 50 :=
by 
  simp only [number_of_dogs, meals_per_day, grams_per_meal, sacks_of_food, kilograms_per_sack, grams_per_kilogram]
  norm_num
  sorry

end dog_food_duration_l267_267099


namespace molecular_weight_H2O_correct_l267_267206

-- Define the atomic weights of hydrogen and oxygen, and the molecular weight of H2O
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Define the molecular weight calculation of H2O
def molecular_weight_H2O : ℝ := (2 * atomic_weight_H) + atomic_weight_O

-- Theorem to state the molecular weight of H2O is approximately 18.016 g/mol
theorem molecular_weight_H2O_correct : molecular_weight_H2O = 18.016 :=
by
  -- Putting the value and calculation
  sorry

end molecular_weight_H2O_correct_l267_267206


namespace simplify_and_evaluate_expression_l267_267800

theorem simplify_and_evaluate_expression :
  (1 - 2 / (Real.tan (Real.pi / 3) - 1 + 1)) / 
  ((Real.tan (Real.pi / 3) - 1) ^ 2 - 2 * (Real.tan (Real.pi / 3) - 1) + 1) / 
  ((Real.tan (Real.pi / 3) - 1) ^ 2 - (Real.tan (Real.pi / 3) - 1)) = 
  (3 - Real.sqrt 3) / 3 :=
sorry

end simplify_and_evaluate_expression_l267_267800


namespace smallest_three_digit_multiple_of_17_l267_267880

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := sorry

end smallest_three_digit_multiple_of_17_l267_267880


namespace decrease_equation_l267_267410

-- Define the initial and final average daily written homework time
def initial_time : ℝ := 100
def final_time : ℝ := 70

-- Define the rate of decrease factor
variable (x : ℝ)

-- Define the functional relationship between initial, final time and the decrease factor
def average_homework_time (x : ℝ) : ℝ :=
  initial_time * (1 - x)^2

-- The theorem to be proved statement
theorem decrease_equation :
  average_homework_time x = final_time ↔ 100 * (1 - x) ^ 2 = 70 :=
by
  -- Proof skipped
  sorry

end decrease_equation_l267_267410


namespace point_2000_coordinates_l267_267669

-- Definition to describe the spiral numbering system in the first quadrant
def spiral_number (n : ℕ) : ℕ × ℕ := sorry

-- The task is to prove that the coordinates of the 2000th point are (44, 25).
theorem point_2000_coordinates : spiral_number 2000 = (44, 25) :=
by
  sorry

end point_2000_coordinates_l267_267669


namespace smallest_three_digit_multiple_of_17_l267_267968

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ ∃ (k : ℕ), n = 17 * k ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l267_267968


namespace smallest_three_digit_multiple_of_17_l267_267888

theorem smallest_three_digit_multiple_of_17 : 
    ∃ (n : ℤ), n ≥ 100 ∧ n < 1000 ∧ (∃ (k : ℤ), n = 17 * k) ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l267_267888


namespace part1_part2_l267_267342

variable {A B C a b c : ℝ}

-- Part (1): Prove that 2a^2 = b^2 + c^2 given the condition
theorem part1 (h : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A)) : 2 * a^2 = b^2 + c^2 := 
sorry

-- Part (2): Prove the perimeter of triangle ABC
theorem part2 (a : ℝ) (h_a : a = 5) (cosA : ℝ) (h_cosA : cosA = 25 / 31) : 5 + b + c = 14 := 
sorry

end part1_part2_l267_267342


namespace ratio_problem_l267_267007

theorem ratio_problem
  (A B C : ℚ)
  (h : A / B = 2 / 1)
  (h1 : B / C = 1 / 4) :
  (3 * A + 2 * B) / (4 * C - A) = 4 / 7 := 
sorry

end ratio_problem_l267_267007


namespace total_distance_driven_l267_267783

def renaldo_distance : ℕ := 15
def ernesto_distance : ℕ := 7 + (renaldo_distance / 3)

theorem total_distance_driven :
  renaldo_distance + ernesto_distance = 27 :=
sorry

end total_distance_driven_l267_267783


namespace smallest_three_digit_multiple_of_17_l267_267831

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := 
begin
  use 102,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    sorry }
end

end smallest_three_digit_multiple_of_17_l267_267831


namespace part1_point_A_value_of_m_part1_area_ABC_part2_max_ordinate_P_l267_267444

noncomputable def quadratic_function (m x : ℝ) : ℝ := (m - 2) * x ^ 2 - x - m ^ 2 + 6 * m - 7

theorem part1_point_A_value_of_m (m : ℝ) (h : quadratic_function m (-1) = 2) : m = 5 :=
sorry

theorem part1_area_ABC (area : ℝ) 
  (h₁ : quadratic_function 5 (1 : ℝ) = 0) 
  (h₂ : quadratic_function 5 (-2/3 : ℝ) = 0) : area = 5 / 3 :=
sorry

theorem part2_max_ordinate_P (m : ℝ) (h : - (m - 3) ^ 2 + 2 ≤ 2) : m = 3 :=
sorry

end part1_point_A_value_of_m_part1_area_ABC_part2_max_ordinate_P_l267_267444


namespace isabella_more_than_sam_l267_267310

variable (I S G : ℕ)

def Giselle_money : G = 120 := by sorry
def Isabella_more_than_Giselle : I = G + 15 := by sorry
def total_donation : I + S + G = 345 := by sorry

theorem isabella_more_than_sam : I - S = 45 := by
sorry

end isabella_more_than_sam_l267_267310


namespace average_homework_time_decrease_l267_267403

theorem average_homework_time_decrease
  (initial_time final_time : ℝ)
  (rate_of_decrease : ℝ)
  (h1 : initial_time = 100)
  (h2 : final_time = 70) :
  initial_time * (1 - rate_of_decrease)^2 = final_time :=
by
  rw [h1, h2]
  sorry

end average_homework_time_decrease_l267_267403


namespace implication_equivalence_l267_267974

variable (P Q : Prop)

theorem implication_equivalence :
  (¬Q → ¬P) ∧ (¬P ∨ Q) ↔ (P → Q) :=
by sorry

end implication_equivalence_l267_267974


namespace problem_a51_l267_267158

-- Definitions of given conditions
variable {a : ℕ → ℤ}
variable (h1 : ∀ n : ℕ, a (n + 2) - 2 * a (n + 1) + a n = 16)
variable (h2 : a 63 = 10)
variable (h3 : a 89 = 10)

-- Proof problem statement
theorem problem_a51 :
  a 51 = 3658 :=
by
  sorry

end problem_a51_l267_267158


namespace part1_part2_l267_267339

-- Definitions and conditions
variables (A B C : ℝ) (a b c : ℝ)
-- Given conditions
variable (h1 : sin C * sin (A - B) = sin B * sin (C - A))

-- Proof for part (1)
theorem part1 : 2 * a^2 = b^2 + c^2 :=
  sorry

-- Given further conditions for part (2)
variables (a_eq : a = 5) (cosA_eq : cos A = 25 / 31)

-- Proof for part (2)
theorem part2 : a + b + c = 14 :=
  sorry

end part1_part2_l267_267339


namespace smallest_three_digit_multiple_of_17_l267_267942

/--
Prove that the smallest three-digit multiple of 17 is 102.
-/
theorem smallest_three_digit_multiple_of_17 : ∃ k : ℤ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 := 
by
  sorry

end smallest_three_digit_multiple_of_17_l267_267942


namespace smallest_three_digit_multiple_of_17_l267_267853

theorem smallest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, m >= 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  let n := 102,
  use n,
  split,
  { norm_num, }, -- n >= 100 is obviously true because n = 102
  split,
  { norm_num, }, -- n < 1000 is obviously true because n = 102
  split,
  { exact dvd_refl 17, }, -- 17 divides 102
  { intros m hm,
    cases hm with h100 hm,
    cases hm with h1000 hdiv,
    exact le_trans (nat.le.intro _ (symm hdiv)) (le_of_not_gt (λ h, by norm_num at h)), },
end

end smallest_three_digit_multiple_of_17_l267_267853


namespace simplify_and_evaluate_expression_l267_267798

theorem simplify_and_evaluate_expression :
  (1 - 2 / (Real.tan (Real.pi / 3) - 1 + 1)) / 
  ((Real.tan (Real.pi / 3) - 1) ^ 2 - 2 * (Real.tan (Real.pi / 3) - 1) + 1) / 
  ((Real.tan (Real.pi / 3) - 1) ^ 2 - (Real.tan (Real.pi / 3) - 1)) = 
  (3 - Real.sqrt 3) / 3 :=
sorry

end simplify_and_evaluate_expression_l267_267798


namespace sin_alpha_pi_over_3_plus_sin_alpha_l267_267448

-- Defining the problem with the given conditions
variable (α : ℝ)
variable (hcos : Real.cos (α + (2 / 3) * Real.pi) = 4 / 5)
variable (hα : -Real.pi / 2 < α ∧ α < 0)

-- Statement to prove
theorem sin_alpha_pi_over_3_plus_sin_alpha :
  Real.sin (α + Real.pi / 3) + Real.sin α = -4 * Real.sqrt 3 / 5 :=
sorry

end sin_alpha_pi_over_3_plus_sin_alpha_l267_267448


namespace max_value_f_on_interval_l267_267816

def f (x : ℝ) : ℝ := 2 * x^2 + 4 * x - 1

theorem max_value_f_on_interval : 
  ∀ x ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f x ≤ 15 :=
by
  sorry

end max_value_f_on_interval_l267_267816


namespace base_angle_isosceles_triangle_l267_267130

theorem base_angle_isosceles_triangle (α : ℝ) (hα : α = 108) (isosceles : ∀ (a b c : ℝ), a = b ∨ b = c ∨ c = a) : α = 108 →
  α + β + β = 180 → β = 36 :=
by
  sorry

end base_angle_isosceles_triangle_l267_267130


namespace smallest_three_digit_multiple_of_17_l267_267866

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m := by
  use 102
  sorry

end smallest_three_digit_multiple_of_17_l267_267866


namespace total_items_in_quiz_l267_267027

theorem total_items_in_quiz (score_percent : ℝ) (mistakes : ℕ) (total_items : ℕ) 
  (h1 : score_percent = 80) 
  (h2 : mistakes = 5) :
  total_items = 25 :=
sorry

end total_items_in_quiz_l267_267027


namespace correct_model_l267_267393

def average_homework_time_decrease (x : ℝ) : Prop :=
  100 * (1 - x) ^ 2 = 70

theorem correct_model (x : ℝ) : average_homework_time_decrease x := 
  sorry

end correct_model_l267_267393


namespace Tino_jellybeans_l267_267651

variable (L T A : ℕ)
variable (h1 : T = L + 24)
variable (h2 : A = L / 2)
variable (h3 : A = 5)

theorem Tino_jellybeans : T = 34 :=
by
  sorry

end Tino_jellybeans_l267_267651


namespace volume_of_rectangular_prism_l267_267200

theorem volume_of_rectangular_prism :
  ∃ (a b c : ℝ), (a * b = 54) ∧ (b * c = 56) ∧ (a * c = 60) ∧ (a * b * c = 379) :=
by sorry

end volume_of_rectangular_prism_l267_267200


namespace apples_in_box_l267_267198

theorem apples_in_box :
  (∀ (o p a : ℕ), 
    (o = 1 / 4 * 56) ∧ 
    (p = 1 / 2 * o) ∧ 
    (a = 5 * p) → 
    a = 35) :=
  by sorry

end apples_in_box_l267_267198


namespace math_proof_problem_l267_267791

def expr (m : ℝ) : ℝ := (1 - (2 / (m + 1))) / ((m ^ 2 - 2 * m + 1) / (m ^ 2 - m))

theorem math_proof_problem :
  expr (Real.tan (Real.pi / 3) - 1) = (3 - Real.sqrt 3) / 3 :=
  sorry

end math_proof_problem_l267_267791


namespace parallel_lines_perpendicular_lines_l267_267727

noncomputable def line1 (m : ℝ) := λ x y : ℝ, x + (1 + m) * y = 2 - m
noncomputable def line2 (m : ℝ) := λ x y : ℝ, 2 * m * x + 4 * y = -16

theorem parallel_lines (m : ℝ) :
  (∀ x y : ℝ, line1 m x y → line2 m x y) ↔ m = 1 :=
by
  sorry

theorem perpendicular_lines (m : ℝ) :
  (∀ x y : ℝ, (line1 m x y → (slope1 ≠ 0) ∧ (slope2 ≠ 0) ∧ 
    slope1 * slope2 = -1)) ↔ m = -2 / 3 :=
by
  sorry

end parallel_lines_perpendicular_lines_l267_267727


namespace smallest_three_digit_multiple_of_17_l267_267843

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m := by
  sorry

end smallest_three_digit_multiple_of_17_l267_267843


namespace smallest_three_digit_multiple_of_17_l267_267837

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m := by
  sorry

end smallest_three_digit_multiple_of_17_l267_267837


namespace product_of_integers_abs_val_not_less_than_1_and_less_than_3_l267_267381

theorem product_of_integers_abs_val_not_less_than_1_and_less_than_3 :
  (-2) * (-1) * 1 * 2 = 4 :=
by
  sorry

end product_of_integers_abs_val_not_less_than_1_and_less_than_3_l267_267381


namespace max_value_of_a_l267_267291

theorem max_value_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - 3 > 0 → x < a) ∧
  (∃ x : ℝ, x < a ∧ ¬(x^2 - 2*x - 3 > 0)) →
  a = -1 :=
by
  sorry

end max_value_of_a_l267_267291


namespace find_X_l267_267982

def operation (X Y : Int) : Int := X + 2 * Y 

lemma property_1 (X : Int) : operation X 0 = X := 
by simp [operation]

lemma property_2 (X Y : Int) : operation X (Y - 1) = (operation X Y) - 2 := 
by simp [operation]; linarith

lemma property_3 (X Y : Int) : operation X (Y + 1) = (operation X Y) + 2 := 
by simp [operation]; linarith

theorem find_X (X : Int) : operation X X = -2019 ↔ X = -673 :=
by sorry

end find_X_l267_267982


namespace computer_multiplications_l267_267238

def rate : ℕ := 15000
def time : ℕ := 2 * 3600
def expected_multiplications : ℕ := 108000000

theorem computer_multiplications : rate * time = expected_multiplications := by
  sorry

end computer_multiplications_l267_267238


namespace smallest_three_digit_multiple_of_17_l267_267856

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → n ≤ m) :=
by
  use 102
  have h1 : 100 ≤ 102 := by linarith
  have h2 : 102 ≤ 999 := by linarith
  have h3 : 102 % 17 = 0 := by norm_num
  have h4 : ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → 102 ≤ m := by
    intro m
    intro h
    have h_m1 := h.1
    have h_m2 := h.2.1
    have h_m3 := h.2.2
    have h_min := nat.le_of_dvd (by linarith) h_m3
    exact h_min
  exact ⟨⟨h1, h2⟩, h3, h4⟩

end smallest_three_digit_multiple_of_17_l267_267856


namespace smallest_three_digit_multiple_of_17_l267_267956

theorem smallest_three_digit_multiple_of_17 : ∃ (k : ℕ), 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 :=
by
  use 6
  split
  · exact of_as_true trivial -- 100 <= 17 * k
  split
  · exact of_as_true trivial -- 17 * k <= 999
  · exact of_as_true trivial -- 17 * k = 102
  sorry

end smallest_three_digit_multiple_of_17_l267_267956


namespace repeating_block_length_of_three_elevens_l267_267477

def smallest_repeating_block_length (x : ℚ) : ℕ :=
  sorry -- Definition to compute smallest repeating block length if needed

theorem repeating_block_length_of_three_elevens :
  smallest_repeating_block_length (3 / 11) = 2 :=
sorry

end repeating_block_length_of_three_elevens_l267_267477


namespace prove_minimality_of_smallest_three_digit_multiple_of_17_l267_267899

def smallest_three_digit_multiple_of_17: ℕ :=
  102

theorem prove_minimality_of_smallest_three_digit_multiple_of_17 (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k ∧ 100 ≤ 17 * k ∧ 17 * k < 1000) 
  → n = 102 :=
    by
      sorry

end prove_minimality_of_smallest_three_digit_multiple_of_17_l267_267899


namespace range_AD_dot_BC_l267_267486

noncomputable def vector_dot_product_range (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 1) : ℝ :=
  let ab := 2
  let ac := 1
  let bc := ac - ab
  let ad := x * ac + (1 - x) * ab
  ad * bc

theorem range_AD_dot_BC : 
  ∃ (a b : ℝ), vector_dot_product_range x h1 h2 = a ∧ ∀ (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 1), a ≤ vector_dot_product_range x h1 h2 ∧ vector_dot_product_range x h1 h2 ≤ b :=
sorry

end range_AD_dot_BC_l267_267486


namespace smallest_three_digit_multiple_of_17_l267_267962

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ ∃ (k : ℕ), n = 17 * k ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l267_267962


namespace gh_two_value_l267_267295

def g (x : ℤ) : ℤ := 3 * x ^ 2 + 2
def h (x : ℤ) : ℤ := -5 * x ^ 3 + 2

theorem gh_two_value : g (h 2) = 4334 := by
  sorry

end gh_two_value_l267_267295


namespace linear_function_increasing_l267_267274

theorem linear_function_increasing (x1 x2 y1 y2 : ℝ) (h1 : y1 = 2 * x1 - 1) (h2 : y2 = 2 * x2 - 1) (h3 : x1 > x2) : y1 > y2 :=
by
  sorry

end linear_function_increasing_l267_267274


namespace fraction_zero_implies_x_is_two_l267_267013

theorem fraction_zero_implies_x_is_two {x : ℝ} (hfrac : (2 - |x|) / (x + 2) = 0) (hdenom : x ≠ -2) : x = 2 :=
by
  sorry

end fraction_zero_implies_x_is_two_l267_267013


namespace find_number_l267_267654

theorem find_number (x : ℝ) :
  9 * (((x + 1.4) / 3) - 0.7) = 5.4 ↔ x = 2.5 :=
by sorry

end find_number_l267_267654


namespace smallest_three_digit_multiple_of_17_l267_267970

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ ∃ (k : ℕ), n = 17 * k ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l267_267970


namespace line_circle_no_intersection_l267_267005

theorem line_circle_no_intersection : 
  (∀ x y : ℝ, 3 * x + 4 * y = 12 → (x - 1)^2 + (y + 1)^2 ≠ 1) :=
by
  sorry

end line_circle_no_intersection_l267_267005


namespace relationship_among_abc_l267_267126

noncomputable
def a := 0.2 ^ 1.5

noncomputable
def b := 2 ^ 0.1

noncomputable
def c := 0.2 ^ 1.3

theorem relationship_among_abc : a < c ∧ c < b := by
  sorry

end relationship_among_abc_l267_267126


namespace expression_value_l267_267528

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem expression_value :
  let numerator := factorial 10
  let denominator := (1 + 2) * (3 + 4) * (5 + 6) * (7 + 8) * (9 + 10)
  numerator / denominator = 660 := by
  sorry

end expression_value_l267_267528


namespace Smithtown_left_handed_women_percentage_l267_267150

theorem Smithtown_left_handed_women_percentage :
  ∃ (x y : ℕ), 
    (3 * x + x = 4 * x) ∧
    (3 * y + 2 * y = 5 * y) ∧
    (4 * x = 5 * y) ∧
    (x = y) → 
    let total_population := 4 * x
    let left_handed_women := x
    left_handed_women / total_population = 0.25 :=
sorry

end Smithtown_left_handed_women_percentage_l267_267150


namespace ellipse_symmetry_range_l267_267455

theorem ellipse_symmetry_range :
  ∀ (x₀ y₀ : ℝ), (x₀^2 / 4 + y₀^2 / 2 = 1) →
  ∃ (x₁ y₁ : ℝ), (x₁ = (4 * y₀ - 3 * x₀) / 5) ∧ (y₁ = (3 * y₀ + 4 * x₀) / 5) →
  -10 ≤ 3 * x₁ - 4 * y₁ ∧ 3 * x₁ - 4 * y₁ ≤ 10 :=
by intros x₀ y₀ h_linearity; sorry

end ellipse_symmetry_range_l267_267455


namespace harriet_trip_time_to_B_l267_267668

variables (D : ℝ) (t1 t2 : ℝ)

-- Definitions based on the given problem
def speed_to_b_town := 100
def speed_to_a_ville := 150
def total_time := 5

-- The condition for the total time for the trip
def total_trip_time_eq := t1 / speed_to_b_town + t2 / speed_to_a_ville = total_time

-- Prove that the time Harriet took to drive from A-ville to B-town is 3 hours.
theorem harriet_trip_time_to_B (h : total_trip_time_eq D D) : t1 = 3 :=
sorry

end harriet_trip_time_to_B_l267_267668


namespace jane_bought_two_bagels_l267_267258

variable (b m d k : ℕ)

def problem_conditions : Prop :=
  b + m + d = 6 ∧ 
  (60 * b + 45 * m + 30 * d) = 100 * k

theorem jane_bought_two_bagels (hb : problem_conditions b m d k) : b = 2 :=
  sorry

end jane_bought_two_bagels_l267_267258


namespace remaining_family_member_age_l267_267644

variable (total_age father_age sister_age : ℕ) (remaining_member_age : ℕ)

def mother_age := father_age - 2
def brother_age := father_age / 2
def known_total_age := father_age + mother_age + brother_age + sister_age

theorem remaining_family_member_age : 
  total_age = 200 ∧ 
  father_age = 60 ∧ 
  sister_age = 40 ∧ 
  known_total_age = total_age - remaining_member_age → 
  remaining_member_age = 12 := by
  sorry

end remaining_family_member_age_l267_267644


namespace problem_statement_l267_267039

theorem problem_statement (A B : ℤ) (h1 : A * B = 15) (h2 : -7 * B - 8 * A = -94) : AB + A = 20 := by
  sorry

end problem_statement_l267_267039


namespace problem_1_problem_2_l267_267483

noncomputable def f (x a : ℝ) : ℝ := |2 * x - a| + |2 * x - 1|

-- Problem 1: When a = -1, prove the solution set for f(x) ≤ 2 is [-1/2, 1/2].
theorem problem_1 (x : ℝ) : (f x (-1) ≤ 2) ↔ (-1/2 ≤ x ∧ x ≤ 1/2) := 
sorry

-- Problem 2: If the solution set of f(x) ≤ |2x + 1| contains the interval [1/2, 1], find the range of a.
theorem problem_2 (a : ℝ) : (∀ x, (1/2 ≤ x ∧ x ≤ 1) → f x a ≤ |2 * x + 1|) ↔ (0 ≤ a ∧ a ≤ 3) :=
sorry

end problem_1_problem_2_l267_267483


namespace eliot_account_balance_l267_267626

-- Definitions for the conditions
variables {A E : ℝ}

--- Conditions rephrased into Lean:
-- 1. Al has more money than Eliot.
def al_more_than_eliot (A E : ℝ) : Prop := A > E

-- 2. The difference between their two accounts is 1/12 of the sum of their two accounts.
def difference_condition (A E : ℝ) : Prop := A - E = (1 / 12) * (A + E)

-- 3. If Al's account were to increase by 10% and Eliot's account were to increase by 15%, 
--     then Al would have exactly $22 more than Eliot in his account.
def percentage_increase_condition (A E : ℝ) : Prop := 1.10 * A = 1.15 * E + 22

-- Prove the total statement
theorem eliot_account_balance : 
  ∀ (A E : ℝ), al_more_than_eliot A E → difference_condition A E → percentage_increase_condition A E → E = 146.67 :=
by
  intros A E h1 h2 h3
  sorry

end eliot_account_balance_l267_267626


namespace sum_of_integers_l267_267638

theorem sum_of_integers (n : ℤ) (h : n * (n + 2) = 20400) : n + (n + 2) = 286 ∨ n + (n + 2) = -286 :=
by
  sorry

end sum_of_integers_l267_267638


namespace find_x_l267_267805

variable (x : ℝ)  -- Current distance Teena is behind Loe in miles
variable (t : ℝ) -- Time period in hours
variable (speed_teena : ℝ) -- Speed of Teena in miles per hour
variable (speed_loe : ℝ) -- Speed of Loe in miles per hour
variable (d_ahead : ℝ) -- Distance Teena will be ahead of Loe in 1.5 hours

axiom conditions : speed_teena = 55 ∧ speed_loe = 40 ∧ t = 1.5 ∧ d_ahead = 15

theorem find_x : (speed_teena * t - speed_loe * t = x + d_ahead) → x = 7.5 :=
by
  intro h
  sorry

end find_x_l267_267805


namespace unit_digit_seven_consecutive_l267_267820

theorem unit_digit_seven_consecutive (n : ℕ) : 
  (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6)) % 10 = 0 := 
by
  sorry

end unit_digit_seven_consecutive_l267_267820


namespace ribbon_per_box_l267_267319

theorem ribbon_per_box (total_ribbon : ℚ) (total_boxes : ℕ) (same_ribbon_each_box : ℚ) 
  (h1 : total_ribbon = 5/12) (h2 : total_boxes = 5) : 
  same_ribbon_each_box = 1/12 :=
sorry

end ribbon_per_box_l267_267319


namespace cauchy_solution_l267_267569

theorem cauchy_solution (f : ℚ → ℚ) (h : ∀ x y : ℚ, f ((x + y) / 2) = (f x) / 2 + (f y) / 2) : 
  ∃ c : ℚ, ∀ x : ℚ, f x = c * x := 
sorry

end cauchy_solution_l267_267569


namespace brady_average_hours_per_month_l267_267100

noncomputable def average_hours_per_month (hours_april : ℕ) (hours_june : ℕ) (hours_september : ℕ) : ℕ :=
  (hours_april + hours_june + hours_september) / 3

theorem brady_average_hours_per_month :
  let days := 30 in
  let hours_april := 6 * days in
  let hours_june := 5 * days in
  let hours_september := 8 * days in
  average_hours_per_month hours_april hours_june hours_september = 190 :=
begin
  sorry
end

end brady_average_hours_per_month_l267_267100


namespace smallest_triangle_perimeter_l267_267214

theorem smallest_triangle_perimeter : 
  ∀ (a b c : ℕ), 
    (2 ∣ a ∧ 2 ∣ b ∧ 2 ∣ c) ∧ (a = b - 2 ∨ a = b + 2) ∧ (b = c - 2 ∨ b = c + 2) ∧ (a + b > c) ∧ (a + c > b) ∧ (b + c > a) 
    → a + b + c = 12 := 
  sorry

end smallest_triangle_perimeter_l267_267214


namespace cube_sum_divisible_by_six_l267_267494

theorem cube_sum_divisible_by_six
  (a b c : ℤ)
  (h1 : 6 ∣ (a^2 + b^2 + c^2))
  (h2 : 3 ∣ (a * b + b * c + c * a))
  : 6 ∣ (a^3 + b^3 + c^3) := 
sorry

end cube_sum_divisible_by_six_l267_267494


namespace smallest_three_digit_multiple_of_17_l267_267851

theorem smallest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, m >= 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  let n := 102,
  use n,
  split,
  { norm_num, }, -- n >= 100 is obviously true because n = 102
  split,
  { norm_num, }, -- n < 1000 is obviously true because n = 102
  split,
  { exact dvd_refl 17, }, -- 17 divides 102
  { intros m hm,
    cases hm with h100 hm,
    cases hm with h1000 hdiv,
    exact le_trans (nat.le.intro _ (symm hdiv)) (le_of_not_gt (λ h, by norm_num at h)), },
end

end smallest_three_digit_multiple_of_17_l267_267851


namespace zoey_holidays_l267_267416

def visits_per_year (visits_per_month : ℕ) (months_per_year : ℕ) : ℕ :=
  visits_per_month * months_per_year

def visits_every_two_months (months_per_year : ℕ) : ℕ :=
  months_per_year / 2

def visits_every_four_months (visits_per_period : ℕ) (periods_per_year : ℕ) : ℕ :=
  visits_per_period * periods_per_year

theorem zoey_holidays (visits_per_month_first : ℕ) 
                      (months_per_year : ℕ) 
                      (visits_per_period_third : ℕ) 
                      (periods_per_year : ℕ) : 
  visits_per_year visits_per_month_first months_per_year 
  + visits_every_two_months months_per_year 
  + visits_every_four_months visits_per_period_third periods_per_year = 39 := 
  by 
  sorry

end zoey_holidays_l267_267416


namespace smallest_three_digit_multiple_of_17_correct_l267_267918

def smallest_three_digit_multiple_of_17 : ℕ :=
  102

theorem smallest_three_digit_multiple_of_17_correct :
  ∀ n, n >= 100 → n < 1000 → n % 17 = 0 → 102 ≤ n :=
begin
  intros n h1 h2 h3,
  sorry,
end

end smallest_three_digit_multiple_of_17_correct_l267_267918


namespace simplify_expression_l267_267794

noncomputable def m : ℝ := Real.tan (Real.pi / 3) - 1

theorem simplify_expression (h : m = Real.tan (Real.pi / 3) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2 * m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 := by
  sorry

end simplify_expression_l267_267794


namespace john_receives_more_l267_267312

noncomputable def partnership_difference (investment_john : ℝ) (investment_mike : ℝ) (profit : ℝ) : ℝ :=
  let total_investment := investment_john + investment_mike
  let one_third_profit := profit / 3
  let two_third_profit := 2 * profit / 3
  let john_effort_share := one_third_profit / 2
  let mike_effort_share := one_third_profit / 2
  let ratio_john := investment_john / total_investment
  let ratio_mike := investment_mike / total_investment
  let john_investment_share := ratio_john * two_third_profit
  let mike_investment_share := ratio_mike * two_third_profit
  let john_total := john_effort_share + john_investment_share
  let mike_total := mike_effort_share + mike_investment_share
  john_total - mike_total

theorem john_receives_more (investment_john investment_mike profit : ℝ)
  (h_john : investment_john = 700)
  (h_mike : investment_mike = 300)
  (h_profit : profit = 3000.0000000000005) :
  partnership_difference investment_john investment_mike profit = 800.0000000000001 := 
sorry

end john_receives_more_l267_267312


namespace trader_sold_45_meters_l267_267247

-- Definitions based on conditions
def selling_price_total : ℕ := 4500
def profit_per_meter : ℕ := 12
def cost_price_per_meter : ℕ := 88
def selling_price_per_meter : ℕ := cost_price_per_meter + profit_per_meter

-- The proof goal to show that the trader sold 45 meters of cloth
theorem trader_sold_45_meters : ∃ x : ℕ, selling_price_per_meter * x = selling_price_total ∧ x = 45 := 
by
  sorry

end trader_sold_45_meters_l267_267247


namespace solve_for_x_l267_267530

theorem solve_for_x (x : ℝ) (h : x ≠ 0) : (7 * x) ^ 5 = (14 * x) ^ 4 → x = 16 / 7 :=
by
  sorry

end solve_for_x_l267_267530


namespace steel_parts_count_l267_267566

-- Definitions for conditions
variables (a b : ℕ)

-- Conditions provided in the problem
axiom machines_count : a + b = 21
axiom chrome_parts : 2 * a + 4 * b = 66

-- Statement to prove
theorem steel_parts_count : 3 * a + 2 * b = 51 :=
by
  sorry

end steel_parts_count_l267_267566


namespace solve_linear_equation_l267_267587

theorem solve_linear_equation (a b x : ℝ) (h : a - b = 0) (ha : a ≠ 0) : ax + b = 0 ↔ x = -1 :=
by sorry

end solve_linear_equation_l267_267587


namespace smallest_three_digit_multiple_of_17_l267_267858

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → n ≤ m) :=
by
  use 102
  have h1 : 100 ≤ 102 := by linarith
  have h2 : 102 ≤ 999 := by linarith
  have h3 : 102 % 17 = 0 := by norm_num
  have h4 : ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → 102 ≤ m := by
    intro m
    intro h
    have h_m1 := h.1
    have h_m2 := h.2.1
    have h_m3 := h.2.2
    have h_min := nat.le_of_dvd (by linarith) h_m3
    exact h_min
  exact ⟨⟨h1, h2⟩, h3, h4⟩

end smallest_three_digit_multiple_of_17_l267_267858


namespace proof_integer_probability_division_is_five_sixteenths_l267_267760

def integer_probability_division_is_five_sixteenths (r k : ℤ) (h1 : -4 < r) (h2 : r < 7) (h3 : 0 < k) (h4 : k < 9): Prop :=
  let valid_r := [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6].val
  let valid_k := [1, 2, 3, 4, 5, 6, 7, 8].val
  let pairs := for r in valid_r, k in valid_k do if k ∣ r then some (r, k) else none
  let valid_pairs := pairs.filterMap id
  let probability := valid_pairs.length / 80
  probability = (5/16 : ℚ)

theorem proof_integer_probability_division_is_five_sixteenths (r k : ℤ) (hr1 : -4 < r) (hr2 : r < 7) (hk1 : 0 < k) (hk2 : k < 9) :
  integer_probability_division_is_five_sixteenths r k hr1 hr2 hk1 hk2 :=
by sorry

end proof_integer_probability_division_is_five_sixteenths_l267_267760


namespace smallest_three_digit_multiple_of_17_correct_l267_267919

def smallest_three_digit_multiple_of_17 : ℕ :=
  102

theorem smallest_three_digit_multiple_of_17_correct :
  ∀ n, n >= 100 → n < 1000 → n % 17 = 0 → 102 ≤ n :=
begin
  intros n h1 h2 h3,
  sorry,
end

end smallest_three_digit_multiple_of_17_correct_l267_267919


namespace initial_books_in_library_l267_267646

theorem initial_books_in_library
  (books_out_tuesday : ℕ)
  (books_in_thursday : ℕ)
  (books_out_friday : ℕ)
  (final_books : ℕ)
  (h1 : books_out_tuesday = 227)
  (h2 : books_in_thursday = 56)
  (h3 : books_out_friday = 35)
  (h4 : final_books = 29) : 
  initial_books = 235 :=
by
  sorry

end initial_books_in_library_l267_267646


namespace boat_stream_ratio_l267_267536

theorem boat_stream_ratio (B S : ℝ) (h : 2 * (B - S) = B + S) : B / S = 3 :=
by
  sorry

end boat_stream_ratio_l267_267536


namespace ratio_of_mixture_l267_267611

theorem ratio_of_mixture (x y : ℚ)
  (h1 : 0.6 = (4 * x + 7 * y) / (9 * x + 9 * y))
  (h2 : 50 = 9 * x + 9 * y) : x / y = 8 / 7 := 
sorry

end ratio_of_mixture_l267_267611


namespace subset_implies_all_elements_l267_267182

variable {U : Type}

theorem subset_implies_all_elements (P Q : Set U) (hPQ : P ⊆ Q) (hP_nonempty : P ≠ ∅) (hQ_nonempty : Q ≠ ∅) :
  ∀ x ∈ P, x ∈ Q :=
by 
  sorry

end subset_implies_all_elements_l267_267182


namespace max_additional_pies_l267_267985

theorem max_additional_pies (initial_cherries used_cherries cherries_per_pie : ℕ) 
  (h₀ : initial_cherries = 500) 
  (h₁ : used_cherries = 350) 
  (h₂ : cherries_per_pie = 35) :
  (initial_cherries - used_cherries) / cherries_per_pie = 4 := 
by
  sorry

end max_additional_pies_l267_267985


namespace z_is_1_2_decades_younger_than_x_l267_267516

variable (X Y Z : ℝ)

theorem z_is_1_2_decades_younger_than_x (h : X + Y = Y + Z + 12) : (X - Z) / 10 = 1.2 :=
by
  sorry

end z_is_1_2_decades_younger_than_x_l267_267516


namespace decrease_equation_l267_267407

-- Define the initial and final average daily written homework time
def initial_time : ℝ := 100
def final_time : ℝ := 70

-- Define the rate of decrease factor
variable (x : ℝ)

-- Define the functional relationship between initial, final time and the decrease factor
def average_homework_time (x : ℝ) : ℝ :=
  initial_time * (1 - x)^2

-- The theorem to be proved statement
theorem decrease_equation :
  average_homework_time x = final_time ↔ 100 * (1 - x) ^ 2 = 70 :=
by
  -- Proof skipped
  sorry

end decrease_equation_l267_267407


namespace smallest_three_digit_multiple_of_17_l267_267893

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m) :=
  sorry

end smallest_three_digit_multiple_of_17_l267_267893


namespace correct_statements_about_microbial_counting_l267_267224

def hemocytometer_counts_bacteria_or_yeast : Prop :=
  true -- based on condition 1

def plate_streaking_allows_colony_counting : Prop :=
  false -- count is not done using the plate streaking method, based on the analysis

def dilution_plating_allows_colony_counting : Prop :=
  true -- based on condition 3  
  
def dilution_plating_count_is_accurate : Prop :=
  false -- colony count is often lower than the actual number, based on the analysis

theorem correct_statements_about_microbial_counting :
  (hemocytometer_counts_bacteria_or_yeast ∧ dilution_plating_allows_colony_counting)
= (plate_streaking_allows_colony_counting ∨ dilution_plating_count_is_accurate) :=
by sorry

end correct_statements_about_microbial_counting_l267_267224


namespace collinear_probability_correct_l267_267755

def number_of_dots := 25

def number_of_four_dot_combinations := Nat.choose number_of_dots 4

-- Calculate the different possibilities for collinear sets:
def horizontal_sets := 5 * 5
def vertical_sets := 5 * 5
def diagonal_sets := 2 + 2

def total_collinear_sets := horizontal_sets + vertical_sets + diagonal_sets

noncomputable def probability_collinear : ℚ :=
  total_collinear_sets / number_of_four_dot_combinations

theorem collinear_probability_correct :
  probability_collinear = 6 / 1415 :=
sorry

end collinear_probability_correct_l267_267755


namespace probability_of_snow_at_least_once_l267_267621

/-- Probability of snow during the first week of January -/
theorem probability_of_snow_at_least_once :
  let p_no_snow_first_four_days := (3/4 : ℚ)
  let p_no_snow_next_three_days := (2/3 : ℚ)
  let p_no_snow_first_week := p_no_snow_first_four_days^4 * p_no_snow_next_three_days^3
  let p_snow_at_least_once := 1 - p_no_snow_first_week
  p_snow_at_least_once = 68359 / 100000 :=
by
  sorry

end probability_of_snow_at_least_once_l267_267621


namespace at_most_one_true_l267_267741

theorem at_most_one_true (p q : Prop) (h : ¬(p ∧ q)) : ¬(p ∧ q ∧ ¬(¬p ∧ ¬q)) :=
by
  sorry

end at_most_one_true_l267_267741


namespace find_width_of_room_l267_267020

theorem find_width_of_room
    (length : ℝ) (area : ℝ)
    (h1 : length = 12) (h2 : area = 96) :
    ∃ width : ℝ, width = 8 ∧ area = length * width :=
by
  sorry

end find_width_of_room_l267_267020


namespace nonzero_real_x_satisfies_equation_l267_267531

theorem nonzero_real_x_satisfies_equation :
  ∃ x : ℝ, x ≠ 0 ∧ (7 * x) ^ 5 = (14 * x) ^ 4 ∧ x = 16 / 7 :=
by
  sorry

end nonzero_real_x_satisfies_equation_l267_267531


namespace symmetric_point_yoz_l267_267050

theorem symmetric_point_yoz (x y z : ℝ) (hx : x = 2) (hy : y = 3) (hz : z = 4) :
  (-x, y, z) = (-2, 3, 4) :=
by
  -- The proof is skipped
  sorry

end symmetric_point_yoz_l267_267050


namespace ashu_complete_job_in_20_hours_l267_267369

/--
  Suresh can complete a job in 15 hours.
  Ashutosh alone can complete the same job in some hours.
  Suresh works for 9 hours and then the remaining job is completed by Ashutosh in 8 hours.
  We need to prove that the number of hours it takes for Ashutosh to complete the job alone is 20.
-/
theorem ashu_complete_job_in_20_hours :
  let A : ℝ := 20
  let suresh_work_rate : ℝ := 1 / 15
  let suresh_completed_work_in_9_hours : ℝ := (9 * suresh_work_rate)
  let remaining_work : ℝ := 1 - suresh_completed_work_in_9_hours
  (8 * (1 / A)) = remaining_work → A = 20 :=
by
  sorry

end ashu_complete_job_in_20_hours_l267_267369


namespace sum_of_ten_numbers_in_circle_l267_267808

theorem sum_of_ten_numbers_in_circle : 
  ∀ (a b c d e f g h i j : ℕ), 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f ∧ 0 < g ∧ 0 < h ∧ 0 < i ∧ 0 < j ∧
  a = Nat.gcd b j + 1 ∧ b = Nat.gcd a c + 1 ∧ c = Nat.gcd b d + 1 ∧ d = Nat.gcd c e + 1 ∧ 
  e = Nat.gcd d f + 1 ∧ f = Nat.gcd e g + 1 ∧ g = Nat.gcd f h + 1 ∧ 
  h = Nat.gcd g i + 1 ∧ i = Nat.gcd h j + 1 ∧ j = Nat.gcd i a + 1 → 
  a + b + c + d + e + f + g + h + i + j = 28 :=
by
  intros
  sorry

end sum_of_ten_numbers_in_circle_l267_267808


namespace acme_profit_calculation_l267_267418

theorem acme_profit_calculation :
  let initial_outlay := 12450
  let cost_per_set := 20.75
  let selling_price := 50
  let number_of_sets := 950
  let total_revenue := number_of_sets * selling_price
  let total_manufacturing_costs := initial_outlay + cost_per_set * number_of_sets
  let profit := total_revenue - total_manufacturing_costs 
  profit = 15337.50 := 
by
  sorry

end acme_profit_calculation_l267_267418


namespace circumcircle_tangent_to_omega_l267_267025

variable {α : Type*} [EuclideanGeometry α]

theorem circumcircle_tangent_to_omega
  (ABC : Triangle α) 
  (ω : Circle α)
  (I : Incenter ABC)
  (ℓ : Line α)
  (D E F : α)
  (AD BE CF : Segment α)
  (x y z : Line α)
  (Θ : Triangle α) :
  -- Conditions
  Circumcircle ABC = ω ∧
  IsIncenter I ABC ∧
  ℓ ∩ LineSegment AI = D ∧ ℓ ∩ LineSegment BI = E ∧ ℓ ∩ LineSegment CI = F ∧
  IsPerpendicularBisector x (Segment.from_points AD) ∧
  IsPerpendicularBisector y (Segment.from_points BE) ∧
  IsPerpendicularBisector z (Segment.from_points CF) ∧
  form_Triangle x y z = Θ →
  -- Conclusion
  Tangent (Circumcircle Θ) ω :=
begin
  sorry,
end

end circumcircle_tangent_to_omega_l267_267025


namespace mark_theater_expense_l267_267498

noncomputable def price_per_performance (hours_per_performance : ℕ) (price_per_hour : ℕ) : ℕ :=
  hours_per_performance * price_per_hour

noncomputable def total_cost (num_weeks : ℕ) (num_visits_per_week : ℕ) (price_per_performance : ℕ) : ℕ :=
  num_weeks * num_visits_per_week * price_per_performance

theorem mark_theater_expense :
  ∀(num_weeks num_visits_per_week hours_per_performance price_per_hour : ℕ),
  num_weeks = 6 →
  num_visits_per_week = 1 →
  hours_per_performance = 3 →
  price_per_hour = 5 →
  total_cost num_weeks num_visits_per_week (price_per_performance hours_per_performance price_per_hour) = 90 :=
by
  intros num_weeks num_visits_per_week hours_per_performance price_per_hour
  intro h_num_weeks h_num_visits_per_week h_hours_per_performance h_price_per_hour
  rw [h_num_weeks, h_num_visits_per_week, h_hours_per_performance, h_price_per_hour]
  sorry

end mark_theater_expense_l267_267498


namespace remaining_money_after_purchase_l267_267784

def initial_money : Float := 15.00
def notebook_cost : Float := 4.00
def pen_cost : Float := 1.50
def notebooks_purchased : ℕ := 2
def pens_purchased : ℕ := 2

theorem remaining_money_after_purchase :
  initial_money - (notebook_cost * notebooks_purchased + pen_cost * pens_purchased) = 4.00 := by
  sorry

end remaining_money_after_purchase_l267_267784


namespace frustum_midsection_area_l267_267818

theorem frustum_midsection_area (r1 r2 : ℝ) (h1 : r1 = 2) (h2 : r2 = 3) :
  let r_mid := (r1 + r2) / 2
  let area_mid := Real.pi * r_mid^2
  area_mid = 25 * Real.pi / 4 := by
  sorry

end frustum_midsection_area_l267_267818


namespace decrease_equation_l267_267411

-- Define the initial and final average daily written homework time
def initial_time : ℝ := 100
def final_time : ℝ := 70

-- Define the rate of decrease factor
variable (x : ℝ)

-- Define the functional relationship between initial, final time and the decrease factor
def average_homework_time (x : ℝ) : ℝ :=
  initial_time * (1 - x)^2

-- The theorem to be proved statement
theorem decrease_equation :
  average_homework_time x = final_time ↔ 100 * (1 - x) ^ 2 = 70 :=
by
  -- Proof skipped
  sorry

end decrease_equation_l267_267411


namespace smallest_triangle_perimeter_l267_267215

theorem smallest_triangle_perimeter : 
  ∀ (a b c : ℕ), 
    (2 ∣ a ∧ 2 ∣ b ∧ 2 ∣ c) ∧ (a = b - 2 ∨ a = b + 2) ∧ (b = c - 2 ∨ b = c + 2) ∧ (a + b > c) ∧ (a + c > b) ∧ (b + c > a) 
    → a + b + c = 12 := 
  sorry

end smallest_triangle_perimeter_l267_267215


namespace probability_target_hit_probability_target_hit_by_A_alone_l267_267728

variable (A B : Event)
variable (pa pb : ℝ)
variable (pA : P A = 0.95)
variable (pB : P B = 0.9)
variable (independence : Independent A B)

/- The probability that the target is hit in a single shot is 0.995 -/
theorem probability_target_hit :
  P (A ∪ B) = 0.995 := by
  sorry

/- The probability that the target is hit by shooter A alone is 0.095 -/
theorem probability_target_hit_by_A_alone :
  P (A \cap Bᶜ) = 0.095 := by
  sorry

end probability_target_hit_probability_target_hit_by_A_alone_l267_267728


namespace hexagon_ratio_identity_l267_267167

theorem hexagon_ratio_identity
  (A B C D E F : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
  (AB BC CD DE EF FA : ℝ)
  (angle_B angle_D angle_F : ℝ)
  (h1 : AB / BC * CD / DE * EF / FA = 1)
  (h2 : angle_B + angle_D + angle_F = 360) :
  (BC / AC * AE / EF * FD / DB = 1) := sorry

end hexagon_ratio_identity_l267_267167


namespace corresponding_angle_C1_of_similar_triangles_l267_267703

theorem corresponding_angle_C1_of_similar_triangles
  (α β γ : ℝ)
  (ABC_sim_A1B1C1 : true)
  (angle_A : α = 50)
  (angle_B : β = 95) :
  γ = 35 :=
by
  sorry

end corresponding_angle_C1_of_similar_triangles_l267_267703


namespace original_people_complete_work_in_four_days_l267_267366

noncomputable def original_people_work_days (P D : ℕ) :=
  (2 * P) * 2 = (1 / 2) * (P * D)

theorem original_people_complete_work_in_four_days (P D : ℕ) (h : original_people_work_days P D) : D = 4 :=
by
  sorry

end original_people_complete_work_in_four_days_l267_267366


namespace triangle_abc_proof_one_triangle_abc_perimeter_l267_267347

-- Defining the basic setup for the triangle and the given condition
variables (a b c A B C : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
variables (hABC : A + B + C = real.pi)
variables (h_sin : real.sin C * real.sin (A - B) = real.sin B * real.sin (C - A))

-- First proof: 2a² = b² + c²
theorem triangle_abc_proof_one : 2 * a^2 = b^2 + c^2 :=
sorry

-- Second proof: the perimeter of the triangle ABC is 14 given a = 5 and cos A = 25/31
variables (ha_val : a = 5) (cosA_val : real.cos A = 25/31)

theorem triangle_abc_perimeter : a + b + c = 14 :=
sorry

end triangle_abc_proof_one_triangle_abc_perimeter_l267_267347


namespace regular_price_of_each_shirt_l267_267390

theorem regular_price_of_each_shirt (P : ℝ) :
    let total_shirts := 20
    let sale_price_per_shirt := 0.8 * P
    let tax_rate := 0.10
    let total_paid := 264
    let total_price := total_shirts * sale_price_per_shirt * (1 + tax_rate)
    total_price = total_paid → P = 15 :=
by
  intros
  sorry

end regular_price_of_each_shirt_l267_267390


namespace cost_of_hiring_actors_l267_267986

theorem cost_of_hiring_actors
  (A : ℕ)
  (CostOfFood : ℕ := 150)
  (EquipmentRental : ℕ := 300 + 2 * A)
  (TotalCost : ℕ := 3 * A + 450)
  (SellingPrice : ℕ := 10000)
  (Profit : ℕ := 5950) :
  TotalCost = SellingPrice - Profit → A = 1200 :=
by
  intro h
  sorry

end cost_of_hiring_actors_l267_267986


namespace exists_n_divisible_by_5_l267_267683

theorem exists_n_divisible_by_5 
  (a b c d m : ℤ) 
  (h_div : a * m ^ 3 + b * m ^ 2 + c * m + d ≡ 0 [ZMOD 5]) 
  (h_d_nonzero : d ≠ 0) : 
  ∃ n : ℤ, d * n ^ 3 + c * n ^ 2 + b * n + a ≡ 0 [ZMOD 5] :=
sorry

end exists_n_divisible_by_5_l267_267683


namespace volume_of_prism_l267_267203

theorem volume_of_prism (a b c : ℝ) (h1 : a * b = 54) (h2 : b * c = 56) (h3 : a * c = 60) :
    a * b * c = 426 :=
sorry

end volume_of_prism_l267_267203


namespace find_x2_plus_y2_l267_267697

theorem find_x2_plus_y2 
  (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
  (h1 : x * y + x + y = 83)
  (h2 : x^2 * y + x * y^2 = 1056) :
  x^2 + y^2 = 458 :=
by
  sorry

end find_x2_plus_y2_l267_267697


namespace set_intersection_l267_267285

def A : Set ℝ := {x | x < 5}
def B : Set ℝ := {x | x < 2}
def B_complement : Set ℝ := {x | x ≥ 2}

theorem set_intersection :
  A ∩ B_complement = {x | 2 ≤ x ∧ x < 5} :=
by 
  sorry

end set_intersection_l267_267285


namespace manager_salary_proof_l267_267185

noncomputable def manager_salary 
    (avg_salary_without_manager : ℝ) 
    (num_employees_without_manager : ℕ) 
    (increase_in_avg_salary : ℝ) 
    (new_total_salary : ℝ) : ℝ :=
    new_total_salary - (num_employees_without_manager * avg_salary_without_manager)

theorem manager_salary_proof :
    manager_salary 3500 100 800 (101 * (3500 + 800)) = 84300 :=
by
    sorry

end manager_salary_proof_l267_267185


namespace value_of_k_l267_267712

noncomputable def find_k (x1 x2 : ℝ) (k : ℝ) : Prop :=
  (2 * x1^2 + k * x1 - 2 = 0) ∧ (2 * x2^2 + k * x2 - 2 = 0) ∧ ((x1 - 2) * (x2 - 2) = 10)

theorem value_of_k (x1 x2 : ℝ) (k : ℝ) (h : find_k x1 x2 k) : k = 7 :=
sorry

end value_of_k_l267_267712


namespace find_x_l267_267722

noncomputable def geometric_series_sum (x: ℝ) : ℝ := 
  1 + x + 2 * x^2 + 3 * x^3 + 4 * x^4 + ∑' n: ℕ, (n + 1) * x^(n + 1)

theorem find_x (x: ℝ) (hx : geometric_series_sum x = 16) : x = 15 / 16 := 
by
  sorry

end find_x_l267_267722


namespace smallest_three_digit_multiple_of_17_l267_267957

theorem smallest_three_digit_multiple_of_17 : ∃ (k : ℕ), 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 :=
by
  use 6
  split
  · exact of_as_true trivial -- 100 <= 17 * k
  split
  · exact of_as_true trivial -- 17 * k <= 999
  · exact of_as_true trivial -- 17 * k = 102
  sorry

end smallest_three_digit_multiple_of_17_l267_267957


namespace extremum_and_equal_values_l267_267604

theorem extremum_and_equal_values {f : ℝ → ℝ} {a b x_0 x_1 : ℝ} 
    (hf : ∀ x, f x = (x - 1)^3 - a * x + b)
    (h'x0 : deriv f x_0 = 0)
    (hfx1_eq_fx0 : f x_1 = f x_0)
    (hx1_ne_x0 : x_1 ≠ x_0) :
  x_1 + 2 * x_0 = 3 := sorry

end extremum_and_equal_values_l267_267604


namespace average_homework_time_decrease_l267_267402

theorem average_homework_time_decrease
  (initial_time final_time : ℝ)
  (rate_of_decrease : ℝ)
  (h1 : initial_time = 100)
  (h2 : final_time = 70) :
  initial_time * (1 - rate_of_decrease)^2 = final_time :=
by
  rw [h1, h2]
  sorry

end average_homework_time_decrease_l267_267402


namespace smallest_three_digit_multiple_of_17_l267_267915

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ 17 * n ∧ 17 * n ≤ 999 ∧ 17 * n = 102 :=
begin
  sorry
end

end smallest_three_digit_multiple_of_17_l267_267915


namespace kaylee_more_boxes_to_sell_l267_267324

-- Definitions for the conditions
def total_needed_boxes : ℕ := 33
def sold_to_aunt : ℕ := 12
def sold_to_mother : ℕ := 5
def sold_to_neighbor : ℕ := 4

-- Target proof goal
theorem kaylee_more_boxes_to_sell :
  total_needed_boxes - (sold_to_aunt + sold_to_mother + sold_to_neighbor) = 12 :=
sorry

end kaylee_more_boxes_to_sell_l267_267324


namespace smallest_three_digit_multiple_of_17_correct_l267_267920

def smallest_three_digit_multiple_of_17 : ℕ :=
  102

theorem smallest_three_digit_multiple_of_17_correct :
  ∀ n, n >= 100 → n < 1000 → n % 17 = 0 → 102 ≤ n :=
begin
  intros n h1 h2 h3,
  sorry,
end

end smallest_three_digit_multiple_of_17_correct_l267_267920


namespace monthly_rent_l267_267092

theorem monthly_rent (cost : ℕ) (maintenance_percentage : ℚ) (annual_taxes : ℕ) (desired_return_rate : ℚ) (monthly_rent : ℚ) :
  cost = 20000 ∧
  maintenance_percentage = 0.10 ∧
  annual_taxes = 460 ∧
  desired_return_rate = 0.06 →
  monthly_rent = 153.70 := 
sorry

end monthly_rent_l267_267092


namespace relation_between_u_and_v_l267_267017

def diameter_circle_condition (AB : ℝ) (r : ℝ) : Prop := AB = 2*r
def chord_tangent_condition (AD BC CD : ℝ) (r : ℝ) : Prop := 
  AD + BC = 2*r ∧ CD*CD = (2*r)*(AD + BC)
def point_selection_condition (AD AF CD : ℝ) : Prop := AD = AF + CD

theorem relation_between_u_and_v (AB AD AF BC CD u v r: ℝ)
  (h1: diameter_circle_condition AB r)
  (h2: chord_tangent_condition AD BC CD r)
  (h3: point_selection_condition AD AF CD)
  (h4: u = AF)
  (h5: v^2 = r^2):
  v^2 = u^3 / (2*r - u) := by
  sorry

end relation_between_u_and_v_l267_267017


namespace arithmetic_sequence_a2_a8_l267_267154

theorem arithmetic_sequence_a2_a8 (a : ℕ → ℤ) (d : ℤ)
  (h1 : ∀ n, a n = a 1 + (n - 1) * d)
  (h2 : a 3 + a 4 + a 5 + a 6 + a 7 = 450) :
  a 2 + a 8 = 180 :=
by
  sorry

end arithmetic_sequence_a2_a8_l267_267154


namespace smallest_three_digit_multiple_of_17_l267_267849

theorem smallest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, m >= 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  let n := 102,
  use n,
  split,
  { norm_num, }, -- n >= 100 is obviously true because n = 102
  split,
  { norm_num, }, -- n < 1000 is obviously true because n = 102
  split,
  { exact dvd_refl 17, }, -- 17 divides 102
  { intros m hm,
    cases hm with h100 hm,
    cases hm with h1000 hdiv,
    exact le_trans (nat.le.intro _ (symm hdiv)) (le_of_not_gt (λ h, by norm_num at h)), },
end

end smallest_three_digit_multiple_of_17_l267_267849


namespace ratio_of_sums_l267_267293

theorem ratio_of_sums (a b c : ℚ) (h1 : b / a = 2) (h2 : c / b = 3) : (a + b) / (b + c) = 3 / 8 := 
  sorry

end ratio_of_sums_l267_267293


namespace probability_snow_first_week_l267_267622

theorem probability_snow_first_week :
  let p1 := 1/4
  let p2 := 1/3
  let no_snow := (3/4)^4 * (2/3)^3
  let snows_at_least_once := 1 - no_snow
  snows_at_least_once = 29 / 32 := by
  sorry

end probability_snow_first_week_l267_267622


namespace find_x8_l267_267588

theorem find_x8 (x : ℝ) (h : x + 1/x = Real.sqrt 2) : x^8 = 1 :=
by sorry

end find_x8_l267_267588


namespace john_total_distance_l267_267981

-- Define the given conditions
def initial_speed : ℝ := 45 -- mph
def first_leg_time : ℝ := 2 -- hours
def second_leg_time : ℝ := 3 -- hours
def distance_before_lunch : ℝ := initial_speed * first_leg_time
def distance_after_lunch : ℝ := initial_speed * second_leg_time

-- Define the total distance
def total_distance : ℝ := distance_before_lunch + distance_after_lunch

-- Prove the total distance is 225 miles
theorem john_total_distance : total_distance = 225 := by
  sorry

end john_total_distance_l267_267981


namespace smallest_positive_period_is_pi_axis_of_symmetry_eq_intervals_of_increase_minimum_value_in_interval_l267_267721

noncomputable def f (x : ℝ) : ℝ :=
  cos x * (sqrt 3 * cos x - sin x) - sqrt 3

theorem smallest_positive_period_is_pi :
  ∃ T > 0, (∀ x : ℝ, f (x + T) = f x) ∧ T = π :=
sorry

theorem axis_of_symmetry_eq :
  ∃ k ∈ ℤ, ∀ x : ℝ, (x = 1 / 2 * k * π - π / 12) :=
sorry

theorem intervals_of_increase :
  ∃ k ∈ ℤ, (∀ x : ℝ, k * π + 5 * π / 12 ≤ x ∧ x ≤ k * π + 11 * π / 12 → deriv f x > 0) :=
sorry

theorem minimum_value_in_interval :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ π / 2 ∧ f x = -1 - sqrt 3 / 2 ∧ x = 5 * π / 12 :=
sorry

end smallest_positive_period_is_pi_axis_of_symmetry_eq_intervals_of_increase_minimum_value_in_interval_l267_267721


namespace sum_of_circle_numbers_l267_267807

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem sum_of_circle_numbers (numbers : Fin 10 → ℕ) 
  (h : ∀ i : Fin 10, numbers i = gcd (numbers (i - 1)) (numbers (i + 1)) + 1) : 
  (Finset.univ.sum numbers) = 28 :=
by
  sorry

end sum_of_circle_numbers_l267_267807


namespace smallest_three_digit_multiple_of_17_l267_267841

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m := by
  sorry

end smallest_three_digit_multiple_of_17_l267_267841


namespace simplify_expression_l267_267796

noncomputable def m : ℝ := Real.tan (Real.pi / 3) - 1

theorem simplify_expression (h : m = Real.tan (Real.pi / 3) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2 * m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 := by
  sorry

end simplify_expression_l267_267796


namespace prove_minimality_of_smallest_three_digit_multiple_of_17_l267_267904

def smallest_three_digit_multiple_of_17: ℕ :=
  102

theorem prove_minimality_of_smallest_three_digit_multiple_of_17 (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k ∧ 100 ≤ 17 * k ∧ 17 * k < 1000) 
  → n = 102 :=
    by
      sorry

end prove_minimality_of_smallest_three_digit_multiple_of_17_l267_267904


namespace infinite_solutions_exists_l267_267179

theorem infinite_solutions_exists : 
  ∃ (S : Set (ℕ × ℕ)), (∀ (a b : ℕ), (a, b) ∈ S → 2 * a^2 - 3 * a + 1 = 3 * b^2 + b) 
  ∧ Set.Infinite S :=
sorry

end infinite_solutions_exists_l267_267179


namespace ten_yuan_notes_count_l267_267166

theorem ten_yuan_notes_count (total_notes : ℕ) (total_change : ℕ) (item_cost : ℕ) (change_given : ℕ → ℕ → ℕ) (is_ten_yuan_notes : ℕ → Prop) :
    total_notes = 16 →
    total_change = 95 →
    item_cost = 5 →
    change_given 10 5 = total_change →
    (∃ x y : ℕ, x + y = total_notes ∧ 10 * x + 5 * y = total_change ∧ is_ten_yuan_notes x) → is_ten_yuan_notes 3 :=
by
  sorry

end ten_yuan_notes_count_l267_267166


namespace arithmetic_sequence_a_eq_zero_l267_267707

theorem arithmetic_sequence_a_eq_zero (a : ℝ) :
  (∀ n : ℕ, n > 0 → ∃ S : ℕ → ℝ, S n = (n^2 : ℝ) + 2 * n + a) →
  a = 0 :=
by
  sorry

end arithmetic_sequence_a_eq_zero_l267_267707


namespace smallest_three_digit_multiple_of_17_l267_267864

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m := by
  use 102
  sorry

end smallest_three_digit_multiple_of_17_l267_267864


namespace negation_proposition_l267_267817

theorem negation_proposition : ∀ (a : ℝ), (a > 3) → (a^2 ≥ 9) :=
by
  intros a ha
  sorry

end negation_proposition_l267_267817


namespace Dave_tiles_210_square_feet_l267_267563

theorem Dave_tiles_210_square_feet
  (ratio_charlie_dave : ℕ := 5 / 7)
  (total_area : ℕ := 360)
  : ∀ (work_done_by_dave : ℕ), work_done_by_dave = 210 :=
by
  sorry

end Dave_tiles_210_square_feet_l267_267563


namespace mean_score_40_l267_267186

theorem mean_score_40 (mean : ℝ) (std_dev : ℝ) (h_std_dev : std_dev = 10)
  (h_within_2_std_dev : ∀ (score : ℝ), score ≥ mean - 2 * std_dev)
  (h_lowest_score : ∀ (score : ℝ), score = 20 → score = mean - 20) :
  mean = 40 :=
by
  -- Placeholder for the proof
  sorry

end mean_score_40_l267_267186


namespace smallest_three_digit_multiple_of_17_l267_267852

theorem smallest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, m >= 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  let n := 102,
  use n,
  split,
  { norm_num, }, -- n >= 100 is obviously true because n = 102
  split,
  { norm_num, }, -- n < 1000 is obviously true because n = 102
  split,
  { exact dvd_refl 17, }, -- 17 divides 102
  { intros m hm,
    cases hm with h100 hm,
    cases hm with h1000 hdiv,
    exact le_trans (nat.le.intro _ (symm hdiv)) (le_of_not_gt (λ h, by norm_num at h)), },
end

end smallest_three_digit_multiple_of_17_l267_267852


namespace original_price_of_lens_is_correct_l267_267770

-- Definitions based on conditions
def current_camera_price : ℝ := 4000
def new_camera_price : ℝ := current_camera_price + 0.30 * current_camera_price
def combined_price_paid : ℝ := 5400
def lens_discount : ℝ := 200
def combined_price_before_discount : ℝ := combined_price_paid + lens_discount

-- Calculated original price of the lens
def lens_original_price : ℝ := combined_price_before_discount - new_camera_price

-- The Lean theorem statement to prove the price is correct
theorem original_price_of_lens_is_correct : lens_original_price = 400 := by
  -- You do not need to provide the actual proof steps
  sorry

end original_price_of_lens_is_correct_l267_267770


namespace digits_conditions_l267_267241

noncomputable def original_number : ℕ := 253
noncomputable def reversed_number : ℕ := 352

theorem digits_conditions (a b c : ℕ) : 
  a + b + c = 10 → 
  b = a + c → 
  (original_number = a * 100 + b * 10 + c) → 
  (reversed_number = c * 100 + b * 10 + a) → 
  reversed_number - original_number = 99 :=
by
  intros h1 h2 h3 h4
  sorry

end digits_conditions_l267_267241


namespace smallest_three_digit_multiple_of_17_l267_267886

theorem smallest_three_digit_multiple_of_17 : 
    ∃ (n : ℤ), n ≥ 100 ∧ n < 1000 ∧ (∃ (k : ℤ), n = 17 * k) ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l267_267886


namespace count_positive_integers_in_range_l267_267290

theorem count_positive_integers_in_range :
  ∃ (count : ℕ), count = 11 ∧
    ∀ (n : ℕ), 300 < n^2 ∧ n^2 < 800 → (n ≥ 18 ∧ n ≤ 28) :=
by
  sorry

end count_positive_integers_in_range_l267_267290


namespace min_time_proof_l267_267387

/-
  Problem: 
  Given 5 colored lights that each can shine in one of the colors {red, orange, yellow, green, blue},
  and the colors are all different, and the interval between two consecutive flashes is 5 seconds.
  Define the ordered shining of these 5 lights once as a "flash", where each flash lasts 5 seconds.
  We need to show that the minimum time required to achieve all different flashes (120 flashes) is equal to 1195 seconds.
-/

def min_time_required : Nat :=
  let num_flashes := 5 * 4 * 3 * 2 * 1
  let flash_time := 5 * num_flashes
  let interval_time := 5 * (num_flashes - 1)
  flash_time + interval_time

theorem min_time_proof : min_time_required = 1195 := by
  sorry

end min_time_proof_l267_267387


namespace smallest_three_digit_multiple_of_17_l267_267840

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m := by
  sorry

end smallest_three_digit_multiple_of_17_l267_267840


namespace smallest_three_digit_multiple_of_17_correct_l267_267924

def smallest_three_digit_multiple_of_17 : ℕ :=
  102

theorem smallest_three_digit_multiple_of_17_correct :
  ∀ n, n >= 100 → n < 1000 → n % 17 = 0 → 102 ≤ n :=
begin
  intros n h1 h2 h3,
  sorry,
end

end smallest_three_digit_multiple_of_17_correct_l267_267924


namespace seq_positive_integers_seq_not_divisible_by_2109_l267_267641

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 3 ∧ a 2 = 6 ∧ ∀ n : ℕ, a (n + 2) = (a (n + 1) ^ 2 + 9) / a n

theorem seq_positive_integers (a : ℕ → ℤ) (h : seq a) : ∀ n : ℕ, 0 < a (n + 1) :=
sorry

theorem seq_not_divisible_by_2109 (a : ℕ → ℤ) (h : seq a) : ¬ ∃ m : ℕ, 2109 ∣ a (m + 1) :=
sorry

end seq_positive_integers_seq_not_divisible_by_2109_l267_267641


namespace trig_identity_and_perimeter_l267_267337

theorem trig_identity_and_perimeter
  (a b c : ℝ) (A B C : ℝ)
  (h1 : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A))
  (a_val : a = 5)
  (cos_A : Real.cos A = 25 / 31) :
  (2 * a^2 = b^2 + c^2) ∧ (a + b + c = 14) :=
by
  sorry

end trig_identity_and_perimeter_l267_267337


namespace smallest_three_digit_multiple_of_17_l267_267934

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  use 102,
  split,
  { exact le_of_lt (nat.lt_of_sub_eq_succ rfl), }, -- 102 is greater than or equal to 100
  split,
  { exact nat.lt_of_sub_eq_succ rfl, }, -- 102 is less than 1000
  split,
  { exact nat.dvd_intro_left (6) rfl, }, -- 17 divides 102
  { intros m hm1 hm2 hm3,
    exact nat.le_of_dvd hm1 (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (hm3)))), }, -- 102 is the smallest
end

end smallest_three_digit_multiple_of_17_l267_267934


namespace total_players_l267_267079

theorem total_players (K Kho_only Both : Nat) (hK : K = 10) (hKho_only : Kho_only = 30) (hBoth : Both = 5) : 
  (K - Both) + Kho_only + Both = 40 := by
  sorry

end total_players_l267_267079


namespace smallest_n_for_purple_l267_267996

-- The conditions as definitions
def red := 18
def green := 20
def blue := 22
def purple_cost := 24

-- The mathematical proof problem statement
theorem smallest_n_for_purple : 
  ∃ n : ℕ, purple_cost * n = Nat.lcm (Nat.lcm red green) blue ∧
            ∀ m : ℕ, (purple_cost * m = Nat.lcm (Nat.lcm red green) blue → m ≥ n) ↔ n = 83 := 
by
  sorry

end smallest_n_for_purple_l267_267996


namespace shortest_distance_is_one_l267_267656

-- Define the problem conditions
def circle_eq (x y : ℝ) := x^2 - 6*x + y^2 - 8*y + 9 = 0

-- Define the function to calculate distance from the origin
def distance (x y : ℝ) := Real.sqrt (x^2 + y^2)

noncomputable def shortest_distance_to_circle : ℝ :=
  let center_x := 3
  let center_y := 4
  let radius := 4
  let origin_to_center := distance center_x center_y
  origin_to_center - radius

-- The statement of the proof
theorem shortest_distance_is_one : shortest_distance_to_circle = 1 :=
by 
  sorry

end shortest_distance_is_one_l267_267656


namespace smallest_three_digit_multiple_of_17_l267_267865

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m := by
  use 102
  sorry

end smallest_three_digit_multiple_of_17_l267_267865


namespace decrease_equation_l267_267409

-- Define the initial and final average daily written homework time
def initial_time : ℝ := 100
def final_time : ℝ := 70

-- Define the rate of decrease factor
variable (x : ℝ)

-- Define the functional relationship between initial, final time and the decrease factor
def average_homework_time (x : ℝ) : ℝ :=
  initial_time * (1 - x)^2

-- The theorem to be proved statement
theorem decrease_equation :
  average_homework_time x = final_time ↔ 100 * (1 - x) ^ 2 = 70 :=
by
  -- Proof skipped
  sorry

end decrease_equation_l267_267409


namespace gym_membership_total_cost_l267_267160

-- Definitions for the conditions stated in the problem
def first_gym_monthly_fee : ℕ := 10
def first_gym_signup_fee : ℕ := 50
def first_gym_discount_rate : ℕ := 10
def first_gym_personal_training_cost : ℕ := 25
def first_gym_sessions_per_year : ℕ := 52

def second_gym_multiplier : ℕ := 3
def second_gym_monthly_fee : ℕ := 3 * first_gym_monthly_fee
def second_gym_signup_fee_multiplier : ℕ := 4
def second_gym_discount_rate : ℕ := 10
def second_gym_personal_training_cost : ℕ := 45
def second_gym_sessions_per_year : ℕ := 52

-- Proof of the total amount John paid in the first year
theorem gym_membership_total_cost:
  let first_gym_annual_cost := (first_gym_monthly_fee * 12) +
                                (first_gym_signup_fee * (100 - first_gym_discount_rate) / 100) +
                                (first_gym_personal_training_cost * first_gym_sessions_per_year)
  let second_gym_annual_cost := (second_gym_monthly_fee * 12) +
                                (second_gym_monthly_fee * second_gym_signup_fee_multiplier * (100 - second_gym_discount_rate) / 100) +
                                (second_gym_personal_training_cost * second_gym_sessions_per_year)
  let total_annual_cost := first_gym_annual_cost + second_gym_annual_cost
  total_annual_cost = 4273 := by
  -- Declaration of the variables used in the problem
  let first_gym_annual_cost := 1465
  let second_gym_annual_cost := 2808
  let total_annual_cost := first_gym_annual_cost + second_gym_annual_cost
  -- Simplify and verify the total cost
  sorry

end gym_membership_total_cost_l267_267160


namespace solve_xyz_l267_267694

theorem solve_xyz (x y z : ℕ) (h1 : x > y) (h2 : y > z) (h3 : z > 0) (h4 : x^2 = y * 2^z + 1) :
  (z ≥ 4 ∧ x = 2^(z-1) + 1 ∧ y = 2^(z-2) + 1) ∨
  (z ≥ 5 ∧ x = 2^(z-1) - 1 ∧ y = 2^(z-2) - 1) ∨
  (z ≥ 3 ∧ x = 2^z - 1 ∧ y = 2^z - 2) :=
sorry

end solve_xyz_l267_267694


namespace max_minus_min_eq_32_l267_267133

def f (x : ℝ) : ℝ := x^3 - 12*x + 8

theorem max_minus_min_eq_32 : 
  let M := max (f (-3)) (max (f 3) (max (f (-2)) (f 2)))
  let m := min (f (-3)) (min (f 3) (min (f (-2)) (f 2)))
  M - m = 32 :=
by
  sorry

end max_minus_min_eq_32_l267_267133


namespace find_m_2n_3k_l267_267140

def is_prime (p : ℕ) : Prop := Nat.Prime p

theorem find_m_2n_3k (m n k : ℕ) (h1 : m + n = 2021) (h2 : is_prime (m - 3 * k)) (h3 : is_prime (n + k)) :
  m + 2 * n + 3 * k = 2025 ∨ m + 2 * n + 3 * k = 4040 := by
  sorry

end find_m_2n_3k_l267_267140


namespace chickens_in_farm_l267_267992

theorem chickens_in_farm (c b : ℕ) (h1 : c + b = 9) (h2 : 2 * c + 4 * b = 26) : c = 5 := by sorry

end chickens_in_farm_l267_267992


namespace blocks_per_friend_l267_267325

theorem blocks_per_friend (total_blocks : ℕ) (friends : ℕ) (h1 : total_blocks = 28) (h2 : friends = 4) :
  total_blocks / friends = 7 :=
by
  sorry

end blocks_per_friend_l267_267325


namespace imaginary_part_of_complex_l267_267425

theorem imaginary_part_of_complex (z : ℂ) (h : (1 - I) * z = I) : z.im = 1 / 2 :=
sorry

end imaginary_part_of_complex_l267_267425


namespace estimate_total_height_l267_267779

theorem estimate_total_height :
  let middle_height := 100
  let left_height := 0.80 * middle_height
  let right_height := (left_height + middle_height) - 20
  left_height + middle_height + right_height = 340 := 
by
  sorry

end estimate_total_height_l267_267779


namespace water_required_for_reaction_l267_267263

noncomputable def sodium_hydride_reaction (NaH H₂O NaOH H₂ : Type) : Nat :=
  1

theorem water_required_for_reaction :
  let NaH := 2
  let required_H₂O := 2 -- Derived from balanced chemical equation and given condition
  sodium_hydride_reaction Nat Nat Nat Nat = required_H₂O :=
by
  sorry

end water_required_for_reaction_l267_267263


namespace smallest_three_digit_multiple_of_17_l267_267872

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := sorry

end smallest_three_digit_multiple_of_17_l267_267872


namespace problem1_problem2a_problem2b_l267_267237

-- Problem 1: Deriving y in terms of x
theorem problem1 (x y : ℕ) (h1 : 30 * x + 10 * y = 2000) : y = 200 - 3 * x :=
by sorry

-- Problem 2(a): Minimum ingredient B for at least 220 yuan profit with a=3
theorem problem2a (x y a w : ℕ) (h1 : a = 3) 
  (h2 : 3 * x + 2 * y ≥ 220) (h3 : y = 200 - 3 * x) 
  (h4 : w = 15 * x + 20 * y) : w = 1300 :=
by sorry

-- Problem 2(b): Profit per portion of dessert A for 450 yuan profit with 3100 grams of B
theorem problem2b (x : ℕ) (a : ℕ) (B : ℕ) 
  (h1 : B = 3100) (h2 : 15 * x + 20 * (200 - 3 * x) ≤ B) 
  (h3 : a * x + 2 * (200 - 3 * x) = 450) 
  (h4 : x ≥ 20) : a = 8 :=
by sorry

end problem1_problem2a_problem2b_l267_267237


namespace chloe_fifth_test_score_l267_267389

theorem chloe_fifth_test_score (a1 a2 a3 a4 a5 : ℕ)
  (h1 : a1 = 84) (h2 : a2 = 87) (h3 : a3 = 78) (h4 : a4 = 90)
  (h_avg : (a1 + a2 + a3 + a4 + a5) / 5 ≥ 85) : 
  a5 ≥ 86 :=
by
  sorry

end chloe_fifth_test_score_l267_267389


namespace matrix_projection_ratios_l267_267723

theorem matrix_projection_ratios (x y z : ℚ) (h : 
  (1 / 14 : ℚ) * x - (5 / 14 : ℚ) * y = x ∧
  - (5 / 14 : ℚ) * x + (24 / 14 : ℚ) * y = y ∧
  0 * x + 0 * y + 1 * z = z)
  : y / x = 13 / 5 ∧ z / x = 1 := 
by 
  sorry

end matrix_projection_ratios_l267_267723


namespace smallest_three_digit_multiple_of_17_l267_267951

theorem smallest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m :=
by
  sorry

end smallest_three_digit_multiple_of_17_l267_267951


namespace evaluate_expression_l267_267104

theorem evaluate_expression : (3^2 - 3) + (4^2 - 4) - (5^2 - 5) + (6^2 - 6) = 28 :=
by 
  sorry

end evaluate_expression_l267_267104


namespace proof_subset_l267_267463

def M : Set ℝ := {x | x ≥ 0}
def N : Set ℝ := {0, 1, 2}

theorem proof_subset : N ⊆ M := sorry

end proof_subset_l267_267463


namespace year_2013_is_not_special_l267_267093

def is_special_year (year : ℕ) : Prop :=
  ∃ (month day : ℕ), month * day = year % 100 ∧ month ≥ 1 ∧ month ≤ 12 ∧ day ≥ 1 ∧ day ≤ 31

theorem year_2013_is_not_special : ¬ is_special_year 2013 := by
  sorry

end year_2013_is_not_special_l267_267093


namespace repeating_block_length_of_three_elevens_l267_267475

def smallest_repeating_block_length (x : ℚ) : ℕ :=
  sorry -- Definition to compute smallest repeating block length if needed

theorem repeating_block_length_of_three_elevens :
  smallest_repeating_block_length (3 / 11) = 2 :=
sorry

end repeating_block_length_of_three_elevens_l267_267475


namespace find_number_eq_fifty_l267_267735

theorem find_number_eq_fifty (x : ℝ) (h : (40 / 100) * x = (25 / 100) * 80) : x = 50 := by 
  sorry

end find_number_eq_fifty_l267_267735


namespace line_equation_l267_267083

theorem line_equation :
  ∃ m b, m = 1 ∧ b = 5 ∧ (∀ x y, y = m * x + b ↔ x - y + 5 = 0) :=
by
  sorry

end line_equation_l267_267083


namespace average_growth_rate_of_second_brand_l267_267434

theorem average_growth_rate_of_second_brand 
  (init1 : ℝ) (rate1 : ℝ) (init2 : ℝ) (t : ℝ) (r : ℝ)
  (h1 : init1 = 4.9) (h2 : rate1 = 0.275) (h3 : init2 = 2.5) (h4 : t = 5.647)
  (h_eq : init1 + rate1 * t = init2 + r * t) : 
  r = 0.7 :=
by 
  -- proof steps would go here
  sorry

end average_growth_rate_of_second_brand_l267_267434


namespace smallest_three_digit_multiple_of_17_l267_267926

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  use 102,
  split,
  { exact le_of_lt (nat.lt_of_sub_eq_succ rfl), }, -- 102 is greater than or equal to 100
  split,
  { exact nat.lt_of_sub_eq_succ rfl, }, -- 102 is less than 1000
  split,
  { exact nat.dvd_intro_left (6) rfl, }, -- 17 divides 102
  { intros m hm1 hm2 hm3,
    exact nat.le_of_dvd hm1 (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (hm3)))), }, -- 102 is the smallest
end

end smallest_three_digit_multiple_of_17_l267_267926


namespace cos_arcsin_l267_267998

theorem cos_arcsin (h : 8^2 + 15^2 = 17^2) : Real.cos (Real.arcsin (8 / 17)) = 15 / 17 :=
by
  have opp_hyp_pos : 0 ≤ 8 / 17 := by norm_num
  have hyp_pos : 0 < 17 := by norm_num
  have opp_le_hyp : 8 / 17 ≤ 1 := by norm_num
  sorry

end cos_arcsin_l267_267998


namespace trig_identity_and_perimeter_l267_267334

theorem trig_identity_and_perimeter
  (a b c : ℝ) (A B C : ℝ)
  (h1 : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A))
  (a_val : a = 5)
  (cos_A : Real.cos A = 25 / 31) :
  (2 * a^2 = b^2 + c^2) ∧ (a + b + c = 14) :=
by
  sorry

end trig_identity_and_perimeter_l267_267334


namespace average_homework_time_decrease_l267_267406

theorem average_homework_time_decrease
  (initial_time final_time : ℝ)
  (rate_of_decrease : ℝ)
  (h1 : initial_time = 100)
  (h2 : final_time = 70) :
  initial_time * (1 - rate_of_decrease)^2 = final_time :=
by
  rw [h1, h2]
  sorry

end average_homework_time_decrease_l267_267406


namespace exists_three_distinct_nats_sum_prod_squares_l267_267116

theorem exists_three_distinct_nats_sum_prod_squares :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  (∃ (x : ℕ), a + b + c = x^2) ∧ 
  (∃ (y : ℕ), a * b * c = y^2) :=
sorry

end exists_three_distinct_nats_sum_prod_squares_l267_267116


namespace axis_of_symmetry_l267_267635

noncomputable def f (x : ℝ) := x^2 - 2 * x + Real.cos (x - 1)

theorem axis_of_symmetry :
  ∀ x : ℝ, f (1 + x) = f (1 - x) :=
by 
  sorry

end axis_of_symmetry_l267_267635


namespace initial_house_cats_l267_267242

theorem initial_house_cats (H : ℕ) (H_condition : 13 + H - 10 = 8) : H = 5 :=
by
-- sorry provides a placeholder to skip the actual proof
sorry

end initial_house_cats_l267_267242


namespace common_ratio_of_geometric_sequence_l267_267440

theorem common_ratio_of_geometric_sequence 
  (a : ℝ) (log2_3 log4_3 log8_3: ℝ)
  (h1: log4_3 = log2_3 / 2)
  (h2: log8_3 = log2_3 / 3) 
  (h_geometric: ∀ i j, 
    i = a + log2_3 → 
    j = a + log4_3 →
    j / i = a + log8_3 / j / i / j
  ) :
  (a + log4_3) / (a + log2_3) = 1/3 :=
by
  sorry

end common_ratio_of_geometric_sequence_l267_267440


namespace kylie_gave_21_coins_to_Laura_l267_267021

def coins_from_piggy_bank : ℕ := 15
def coins_from_brother : ℕ := 13
def coins_from_father : ℕ := 8
def coins_left : ℕ := 15

def total_coins_collected : ℕ := coins_from_piggy_bank + coins_from_brother + coins_from_father
def coins_given_to_Laura : ℕ := total_coins_collected - coins_left

theorem kylie_gave_21_coins_to_Laura :
  coins_given_to_Laura = 21 :=
by
  sorry

end kylie_gave_21_coins_to_Laura_l267_267021


namespace quadratic_form_sum_const_l267_267639

theorem quadratic_form_sum_const (a b c x : ℝ) (h : 4 * x^2 - 28 * x - 48 = a * (x + b)^2 + c) : 
  a + b + c = -96.5 :=
by
  sorry

end quadratic_form_sum_const_l267_267639


namespace parallelogram_area_l267_267155

open Real

def line1 (p : ℝ × ℝ) : Prop := p.2 = 2
def line2 (p : ℝ × ℝ) : Prop := p.2 = -2
def line3 (p : ℝ × ℝ) : Prop := 4 * p.1 + 7 * p.2 - 10 = 0
def line4 (p : ℝ × ℝ) : Prop := 4 * p.1 + 7 * p.2 + 20 = 0

theorem parallelogram_area :
  ∃ D : ℝ, D = 30 ∧
  (∀ p : ℝ × ℝ, line1 p ∨ line2 p ∨ line3 p ∨ line4 p) :=
sorry

end parallelogram_area_l267_267155


namespace smallest_three_digit_multiple_of_17_l267_267964

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ ∃ (k : ℕ), n = 17 * k ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l267_267964


namespace apartments_in_each_complex_l267_267648

variable {A : ℕ}

theorem apartments_in_each_complex
    (h1 : ∀ (locks_per_apartment : ℕ), locks_per_apartment = 3)
    (h2 : ∀ (num_complexes : ℕ), num_complexes = 2)
    (h3 : 3 * 2 * A = 72) :
    A = 12 :=
by
  sorry

end apartments_in_each_complex_l267_267648


namespace physics_class_size_l267_267558

theorem physics_class_size (total_students physics_only math_only both : ℕ) 
  (h1 : total_students = 53)
  (h2 : both = 7)
  (h3 : physics_only = 2 * (math_only + both))
  (h4 : total_students = physics_only + math_only + both) :
  physics_only + both = 40 :=
by
  sorry

end physics_class_size_l267_267558


namespace smallest_three_digit_multiple_of_17_l267_267894

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m) :=
  sorry

end smallest_three_digit_multiple_of_17_l267_267894


namespace contradiction_method_l267_267388

theorem contradiction_method (x y : ℝ) (h : x + y ≤ 0) : x ≤ 0 ∨ y ≤ 0 :=
sorry

end contradiction_method_l267_267388


namespace arrangement_condition_l267_267264

theorem arrangement_condition (x y z : ℕ) 
  (hx : x ≥ 1) 
  (hy : y ≥ 1) 
  (hz : z ≥ 1) 
  (H1 : x ≤ y + z) 
  (H2 : y ≤ x + z) 
  (H3 : z ≤ x + y) : 
  ∃ (A : ℕ) (B : ℕ) (C : ℕ), 
    A = x ∧ B = y ∧ C = z ∧
    A ≥ 1 ∧ B ≥ 1 ∧ C ≥ 1 ∧
    (A ≤ B + C) ∧ (B ≤ A + C) ∧ (C ≤ A + B) :=
by
  sorry

end arrangement_condition_l267_267264


namespace range_of_a_l267_267458

-- Define the function f as given in the problem
def f (a x : ℝ) : ℝ := x^3 + a * x^2 + (a + 6) * x + 1

-- The mathematical statement to be proven in Lean
theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, ∃ m M : ℝ, m = (f a x) ∧ M = (f a y) ∧ (∀ z : ℝ, f a z ≥ m) ∧ (∀ z : ℝ, f a z ≤ M)) ↔ 
  (a < -3 ∨ a > 6) :=
sorry

end range_of_a_l267_267458


namespace first_person_days_l267_267628

theorem first_person_days (x : ℝ) (hp : 30 ≥ 0) (ht : 10 ≥ 0) (h_work : 1/x + 1/30 = 1/10) : x = 15 :=
by
  -- Begin by acknowledging the assumptions: hp, ht, and h_work
  sorry

end first_person_days_l267_267628


namespace repeating_block_digits_l267_267467

theorem repeating_block_digits (n d : ℕ) (h1 : n = 3) (h2 : d = 11) : 
  (∃ repeating_block_length, repeating_block_length = 2) := by
  sorry

end repeating_block_digits_l267_267467


namespace stacy_height_last_year_l267_267367

-- Definitions for the conditions
def brother_growth := 1
def stacy_growth := brother_growth + 6
def stacy_current_height := 57
def stacy_last_years_height := stacy_current_height - stacy_growth

-- Proof statement
theorem stacy_height_last_year : stacy_last_years_height = 50 :=
by
  -- proof steps will go here
  sorry

end stacy_height_last_year_l267_267367


namespace coeff_comparison_l267_267607

def a_k (k : ℕ) : ℕ := (2 ^ k) * Nat.choose 100 k

theorem coeff_comparison :
  (Finset.filter (fun r => a_k r < a_k (r + 1)) (Finset.range 100)).card = 67 :=
by
  sorry

end coeff_comparison_l267_267607


namespace prime_factors_identity_l267_267351

theorem prime_factors_identity (w x y z k : ℕ) 
    (h : 2^w * 3^x * 5^y * 7^z * 11^k = 900) : 
      2 * w + 3 * x + 5 * y + 7 * z + 11 * k = 20 :=
by
  sorry

end prime_factors_identity_l267_267351


namespace payment_to_z_l267_267538

-- Definitions of the conditions
def x_work_rate := 1 / 15
def y_work_rate := 1 / 10
def total_payment := 720
def combined_work_rate_xy := x_work_rate + y_work_rate
def combined_work_rate_xyz := 1 / 5
def z_work_rate := combined_work_rate_xyz - combined_work_rate_xy
def z_contribution := z_work_rate * 5
def z_payment := z_contribution * total_payment

-- The statement to be proven
theorem payment_to_z : z_payment = 120 := by
  sorry

end payment_to_z_l267_267538


namespace plains_total_square_miles_l267_267518

theorem plains_total_square_miles (RegionB : ℝ) (h1 : RegionB = 200) (RegionA : ℝ) (h2 : RegionA = RegionB - 50) : 
  RegionA + RegionB = 350 := 
by 
  sorry

end plains_total_square_miles_l267_267518


namespace cost_price_of_book_l267_267980

theorem cost_price_of_book 
  (C : ℝ) 
  (h1 : 1.10 * C = sp10) 
  (h2 : 1.15 * C = sp15)
  (h3 : sp15 - sp10 = 90) : 
  C = 1800 := 
sorry

end cost_price_of_book_l267_267980


namespace sum_of_x_and_y_l267_267579

theorem sum_of_x_and_y (x y : ℤ) (h : 2 * x * y + x + y = 83) : x + y = 83 ∨ x + y = -85 := 
sorry

end sum_of_x_and_y_l267_267579


namespace total_sales_in_december_correct_l267_267746

def ear_muffs_sales_in_december : ℝ :=
  let typeB_sold := 3258
  let typeB_price := 6.9
  let typeC_sold := 3186
  let typeC_price := 7.4
  let total_typeB_sales := typeB_sold * typeB_price
  let total_typeC_sales := typeC_sold * typeC_price
  total_typeB_sales + total_typeC_sales

theorem total_sales_in_december_correct :
  ear_muffs_sales_in_december = 46056.6 :=
by
  sorry

end total_sales_in_december_correct_l267_267746


namespace geometric_sequence_sum_l267_267283

theorem geometric_sequence_sum (a : ℝ) (q : ℝ) (h1 : a * q^2 + a * q^5 = 6)
  (h2 : a * q^4 + a * q^7 = 9) : a * q^6 + a * q^9 = 27 / 2 :=
by
  sorry

end geometric_sequence_sum_l267_267283


namespace average_inside_time_l267_267163

def jonsey_awake_hours := 24 * (2/3)
def jonsey_inside_fraction := 1 - (1/2)
def jonsey_inside_hours := jonsey_awake_hours * jonsey_inside_fraction

def riley_awake_hours := 24 * (3/4)
def riley_inside_fraction := 1 - (1/3)
def riley_inside_hours := riley_awake_hours * riley_inside_fraction

def total_inside_hours := jonsey_inside_hours + riley_inside_hours
def number_of_people := 2
def average_inside_hours := total_inside_hours / number_of_people

theorem average_inside_time (jonsey_awake_hrs : ℝ) (jonsey_inside_frac : ℝ) 
  (jonsey_inside_hrs : ℝ) (riley_awake_hrs : ℝ) (riley_inside_frac : ℝ) 
  (riley_inside_hrs : ℝ) (total_inside_hrs : ℝ) (num_people : ℝ) 
  (avg_inside_hrs : ℝ) :
  jonsey_awake_hrs = 24 * (2 / 3) → 
  jonsey_inside_frac = 1 - (1 / 2) →
  jonsey_inside_hrs = jonsey_awake_hrs * jonsey_inside_frac →
  riley_awake_hrs = 24 * (3 / 4) →
  riley_inside_frac = 1 - (1 / 3) →
  riley_inside_hrs = riley_awake_hrs * riley_inside_frac →
  total_inside_hrs = jonsey_inside_hrs + riley_inside_hrs →
  num_people = 2 →
  avg_inside_hrs = total_inside_hrs / num_people →
  avg_inside_hrs = 10 := 
by
  intros
  sorry

end average_inside_time_l267_267163


namespace discount_percent_l267_267810

theorem discount_percent (MP CP SP : ℝ)
  (h1 : CP = 0.64 * MP)
  (h2 : (SP - CP) / CP * 100 = 34.375) :
  ((MP - SP) / MP * 100) = 14 :=
by
  -- Proof would go here
  sorry

end discount_percent_l267_267810


namespace max_A_plus_B_l267_267328

theorem max_A_plus_B:
  ∃ A B C D : ℕ,
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧
  A + B + C + D = 17 ∧ ∃ k : ℕ, C + D ≠ 0 ∧ A + B = k * (C + D) ∧
  A + B = 16 :=
by sorry

end max_A_plus_B_l267_267328


namespace cube_volume_l267_267645

variables (x s : ℝ)
theorem cube_volume (h : 6 * s^2 = 6 * x^2) : s^3 = x^3 :=
by sorry

end cube_volume_l267_267645


namespace triangle_abc_proof_one_triangle_abc_perimeter_l267_267349

-- Defining the basic setup for the triangle and the given condition
variables (a b c A B C : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
variables (hABC : A + B + C = real.pi)
variables (h_sin : real.sin C * real.sin (A - B) = real.sin B * real.sin (C - A))

-- First proof: 2a² = b² + c²
theorem triangle_abc_proof_one : 2 * a^2 = b^2 + c^2 :=
sorry

-- Second proof: the perimeter of the triangle ABC is 14 given a = 5 and cos A = 25/31
variables (ha_val : a = 5) (cosA_val : real.cos A = 25/31)

theorem triangle_abc_perimeter : a + b + c = 14 :=
sorry

end triangle_abc_proof_one_triangle_abc_perimeter_l267_267349


namespace multiple_of_27_l267_267675

theorem multiple_of_27 (x y z : ℤ) 
  (h1 : (2 * x + 5 * y + 11 * z) = 4 * (x + y + z)) 
  (h2 : (2 * x + 20 * y + 110 * z) = 6 * (2 * x + 5 * y + 11 * z)) :
  ∃ k : ℤ, x + y + z = 27 * k :=
by
  sorry

end multiple_of_27_l267_267675


namespace hoseok_has_least_papers_l267_267322

-- Definitions based on the conditions
def pieces_jungkook : ℕ := 10
def pieces_hoseok : ℕ := 7
def pieces_seokjin : ℕ := pieces_jungkook - 2

-- Theorem stating Hoseok has the least pieces of colored paper
theorem hoseok_has_least_papers : pieces_hoseok < pieces_jungkook ∧ pieces_hoseok < pieces_seokjin := by 
  sorry

end hoseok_has_least_papers_l267_267322


namespace smallest_three_digit_multiple_of_17_l267_267947

theorem smallest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m :=
by
  sorry

end smallest_three_digit_multiple_of_17_l267_267947


namespace closed_under_all_operations_l267_267505

structure sqrt2_num where
  re : ℚ
  im : ℚ

namespace sqrt2_num

def add (x y : sqrt2_num) : sqrt2_num :=
  ⟨x.re + y.re, x.im + y.im⟩

def subtract (x y : sqrt2_num) : sqrt2_num :=
  ⟨x.re - y.re, x.im - y.im⟩

def multiply (x y : sqrt2_num) : sqrt2_num :=
  ⟨x.re * y.re + 2 * x.im * y.im, x.re * y.im + x.im * y.re⟩

def divide (x y : sqrt2_num) : sqrt2_num :=
  let denom := y.re^2 - 2 * y.im^2
  ⟨(x.re * y.re - 2 * x.im * y.im) / denom, (x.im * y.re - x.re * y.im) / denom⟩

theorem closed_under_all_operations (a b c d : ℚ) :
  ∃ (e f : ℚ), 
    add ⟨a, b⟩ ⟨c, d⟩ = ⟨e, f⟩ ∧ 
    ∃ (g h : ℚ), 
    subtract ⟨a, b⟩ ⟨c, d⟩ = ⟨g, h⟩ ∧ 
    ∃ (i j : ℚ), 
    multiply ⟨a, b⟩ ⟨c, d⟩ = ⟨i, j⟩ ∧ 
    ∃ (k l : ℚ), 
    divide ⟨a, b⟩ ⟨c, d⟩ = ⟨k, l⟩ := by
  sorry

end sqrt2_num

end closed_under_all_operations_l267_267505


namespace optimal_play_winner_l267_267823

theorem optimal_play_winner (n : ℕ) (h : n > 1) : (n % 2 = 0) ↔ (first_player_wins: Bool) :=
  sorry

end optimal_play_winner_l267_267823


namespace geometric_sequence_sum_is_120_l267_267718

noncomputable def sum_first_four_geometric_seq (a : ℕ → ℝ) (q : ℝ) : ℝ :=
  a 1 + a 2 + a 3 + a 4

theorem geometric_sequence_sum_is_120 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_pos_geometric : 0 < q ∧ q < 1)
  (h_a3_a5 : a 3 + a 5 = 20)
  (h_a3_a5_product : a 3 * a 5 = 64) 
  (h_geometric : ∀ n, a (n + 1) = a n * q) :
  sum_first_four_geometric_seq a q = 120 :=
sorry

end geometric_sequence_sum_is_120_l267_267718


namespace graph_of_4x2_minus_9y2_is_pair_of_straight_lines_l267_267041

theorem graph_of_4x2_minus_9y2_is_pair_of_straight_lines :
  (∀ x y : ℝ, (4 * x^2 - 9 * y^2 = 0) → (x / y = 3 / 2 ∨ x / y = -3 / 2)) :=
by
  sorry

end graph_of_4x2_minus_9y2_is_pair_of_straight_lines_l267_267041


namespace smallest_three_digit_multiple_of_17_l267_267965

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ ∃ (k : ℕ), n = 17 * k ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l267_267965


namespace smallest_three_digit_multiple_of_17_l267_267910

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ 17 * n ∧ 17 * n ≤ 999 ∧ 17 * n = 102 :=
begin
  sorry
end

end smallest_three_digit_multiple_of_17_l267_267910


namespace total_distance_is_27_l267_267781

-- Condition: Renaldo drove 15 kilometers
def renaldo_distance : ℕ := 15

-- Condition: Ernesto drove 7 kilometers more than one-third of Renaldo's distance
def ernesto_distance := (1 / 3 : ℚ) * renaldo_distance + 7

-- Theorem to prove that total distance driven by both men is 27 kilometers
theorem total_distance_is_27 : renaldo_distance + ernesto_distance = 27 := by
  sorry

end total_distance_is_27_l267_267781


namespace smallest_four_digit_divisible_by_53_ending_in_3_l267_267208

theorem smallest_four_digit_divisible_by_53_ending_in_3 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n ∧ n % 10 = 3 ∧ n = 1113 := 
by
  sorry

end smallest_four_digit_divisible_by_53_ending_in_3_l267_267208


namespace smallest_triangle_perimeter_consecutive_even_l267_267221

theorem smallest_triangle_perimeter_consecutive_even :
  ∃ (a b c : ℕ), a = 2 ∧ b = 4 ∧ c = 6 ∧
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧ (a + b + c = 12) :=
by {
  sorry
}

end smallest_triangle_perimeter_consecutive_even_l267_267221


namespace calculation_correct_l267_267102

theorem calculation_correct : 4 * 6 * 8 - 10 / 2 = 187 := by
  sorry

end calculation_correct_l267_267102


namespace john_total_money_after_3_years_l267_267758

def principal : ℝ := 1000
def rate : ℝ := 0.1
def time : ℝ := 3

/-
  We need to prove that the total money after 3 years is $1300
-/
theorem john_total_money_after_3_years (principal : ℝ) (rate : ℝ) (time : ℝ) :
  principal + (principal * rate * time) = 1300 := by
  sorry

end john_total_money_after_3_years_l267_267758


namespace problem_solution_l267_267446

theorem problem_solution (k m : ℕ) (h1 : 30^k ∣ 929260) (h2 : 20^m ∣ 929260) : (3^k - k^3) + (2^m - m^3) = 2 := 
by sorry

end problem_solution_l267_267446


namespace problem_statement_l267_267350

noncomputable def log_three_four : ℝ := Real.log 4 / Real.log 3
noncomputable def a : ℝ := Real.log (log_three_four) / Real.log (3/4)
noncomputable def b : ℝ := Real.rpow (3/4 : ℝ) 0.5
noncomputable def c : ℝ := Real.rpow (4/3 : ℝ) 0.5

theorem problem_statement : a < b ∧ b < c :=
by
  sorry

end problem_statement_l267_267350


namespace delta_x_not_zero_l267_267597

noncomputable def average_rate_of_change (f : ℝ → ℝ) (x : ℝ) (delta_x : ℝ) : ℝ :=
  (f (x + delta_x) - f x) / delta_x

theorem delta_x_not_zero (f : ℝ → ℝ) (x delta_x : ℝ) (h_neq : delta_x ≠ 0):
  average_rate_of_change f x delta_x ≠ 0 := 
by
  sorry

end delta_x_not_zero_l267_267597


namespace find_multiple_of_sum_l267_267380

-- Define the conditions and the problem statement in Lean
theorem find_multiple_of_sum (a b m : ℤ) 
  (h1 : b = 8) 
  (h2 : b - a = 3) 
  (h3 : a * b = 14 + m * (a + b)) : 
  m = 2 :=
by
  sorry

end find_multiple_of_sum_l267_267380


namespace smallest_three_digit_multiple_of_17_l267_267862

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → n ≤ m) :=
by
  use 102
  have h1 : 100 ≤ 102 := by linarith
  have h2 : 102 ≤ 999 := by linarith
  have h3 : 102 % 17 = 0 := by norm_num
  have h4 : ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → 102 ≤ m := by
    intro m
    intro h
    have h_m1 := h.1
    have h_m2 := h.2.1
    have h_m3 := h.2.2
    have h_min := nat.le_of_dvd (by linarith) h_m3
    exact h_min
  exact ⟨⟨h1, h2⟩, h3, h4⟩

end smallest_three_digit_multiple_of_17_l267_267862


namespace probability_one_side_is_side_of_decagon_l267_267111

theorem probability_one_side_is_side_of_decagon :
  let decagon_vertices := 10
  let total_triangles := Nat.choose decagon_vertices 3
  let favorable_one_side :=
    decagon_vertices * (decagon_vertices - 3) / 2
  let favorable_two_sides := decagon_vertices
  let favorable_outcomes := favorable_one_side + favorable_two_sides
  let probability := favorable_outcomes / total_triangles
  total_triangles = 120 ∧ favorable_outcomes = 60 ∧ probability = 1 / 2 := 
by
  sorry

end probability_one_side_is_side_of_decagon_l267_267111


namespace nonneg_reals_ineq_l267_267499

theorem nonneg_reals_ineq 
  (a b x y : ℝ)
  (ha : 0 ≤ a) (hb : 0 ≤ b)
  (hx : 0 ≤ x) (hy : 0 ≤ y)
  (hab : a^5 + b^5 ≤ 1)
  (hxy : x^5 + y^5 ≤ 1) :
  a^2 * x^3 + b^2 * y^3 ≤ 1 :=
sorry

end nonneg_reals_ineq_l267_267499


namespace smallest_three_digit_multiple_of_17_l267_267833

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := 
begin
  use 102,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    sorry }
end

end smallest_three_digit_multiple_of_17_l267_267833


namespace math_proof_problem_l267_267790

def expr (m : ℝ) : ℝ := (1 - (2 / (m + 1))) / ((m ^ 2 - 2 * m + 1) / (m ^ 2 - m))

theorem math_proof_problem :
  expr (Real.tan (Real.pi / 3) - 1) = (3 - Real.sqrt 3) / 3 :=
  sorry

end math_proof_problem_l267_267790


namespace find_A_in_terms_of_B_and_C_l267_267024

theorem find_A_in_terms_of_B_and_C 
  (A B C : ℝ) (hB : B ≠ 0) 
  (f : ℝ → ℝ) (g : ℝ → ℝ) 
  (hf : ∀ x, f x = A * x - 2 * B^2)
  (hg : ∀ x, g x = B * x + C * x^2)
  (hfg : f (g 1) = 4 * B^2)
  : A = 6 * B * B / (B + C) :=
by
  sorry

end find_A_in_terms_of_B_and_C_l267_267024


namespace weightOfEachPacket_l267_267503

/-- Definition for the number of pounds in one ton --/
def poundsPerTon : ℕ := 2100

/-- Total number of packets filling the 13-ton capacity --/
def numPackets : ℕ := 1680

/-- Capacity of the gunny bag in tons --/
def capacityInTons : ℕ := 13

/-- Total weight of the gunny bag in pounds --/
def totalWeightInPounds : ℕ := capacityInTons * poundsPerTon

/-- Statement that each packet weighs 16.25 pounds --/
theorem weightOfEachPacket : (totalWeightInPounds / numPackets : ℚ) = 16.25 :=
sorry

end weightOfEachPacket_l267_267503


namespace expression_evaluation_l267_267705

variable {x y : ℝ}

theorem expression_evaluation (h : (x-2)^2 + |y-3| = 0) :
  ( (x - 2 * y) * (x + 2 * y) - (x - y) ^ 2 + y * (y + 2 * x) ) / (-2 * y) = 2 :=
by
  sorry

end expression_evaluation_l267_267705


namespace calculation_101_squared_minus_99_squared_l267_267105

theorem calculation_101_squared_minus_99_squared : 101^2 - 99^2 = 400 :=
by
  sorry

end calculation_101_squared_minus_99_squared_l267_267105


namespace smallest_three_digit_multiple_of_17_l267_267881

theorem smallest_three_digit_multiple_of_17 : 
    ∃ (n : ℤ), n ≥ 100 ∧ n < 1000 ∧ (∃ (k : ℤ), n = 17 * k) ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l267_267881


namespace MatthewSharedWithTwoFriends_l267_267353

theorem MatthewSharedWithTwoFriends
  (crackers : ℕ)
  (cakes : ℕ)
  (cakes_per_person : ℕ)
  (persons : ℕ)
  (H1 : crackers = 29)
  (H2 : cakes = 30)
  (H3 : cakes_per_person = 15)
  (H4 : persons * cakes_per_person = cakes) :
  persons = 2 := by
  sorry

end MatthewSharedWithTwoFriends_l267_267353


namespace waiter_earned_total_tips_l267_267250

def tips (c1 c2 c3 c4 c5 : ℝ) := c1 + c2 + c3 + c4 + c5

theorem waiter_earned_total_tips :
  tips 1.50 2.75 3.25 4.00 5.00 = 16.50 := 
by 
  sorry

end waiter_earned_total_tips_l267_267250


namespace max_value_m_l267_267169

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

theorem max_value_m (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : ∀ x : ℝ, quadratic_function a b c (x-4) = quadratic_function a b c (2-x))
  (h3 : ∀ x : ℝ, 0 < x ∧ x < 2 → quadratic_function a b c x ≤ ( (x+1)/2 )^2)
  (h4 : ∀ x : ℝ, quadratic_function a b c x ≥ 0)
  (h_min : ∃ x : ℝ, quadratic_function a b c x = 0) :
  ∃ (m : ℝ), m > 1 ∧ (∃ t : ℝ, ∀ x : ℝ, 1 ≤ x ∧ x ≤ m → quadratic_function a b c (x+t) ≤ x) ∧ m = 9 := 
sorry

end max_value_m_l267_267169


namespace smallest_sum_of_24_consecutive_integers_is_perfect_square_l267_267193

theorem smallest_sum_of_24_consecutive_integers_is_perfect_square :
  ∃ n : ℕ, (n > 0) ∧ (m : ℕ) ∧ (2 * n + 23 = m^2) ∧ (12 * (2 * n + 23) = 300) :=
by
  sorry

end smallest_sum_of_24_consecutive_integers_is_perfect_square_l267_267193


namespace limit_of_subadditive_sequence_l267_267487

theorem limit_of_subadditive_sequence 
  {a : ℕ → ℝ} (h : ∀ n m, a n ≤ a (n + m) ∧ a (n + m) ≤ a n + a m) :
  ∃ L : ℝ, (tendsto (λ n, a n / n) at_top (nhds L)) ∧ 
           L = Inf {x : ℝ | ∃ n, x = a n / n} :=
sorry

end limit_of_subadditive_sequence_l267_267487


namespace prove_minimality_of_smallest_three_digit_multiple_of_17_l267_267907

def smallest_three_digit_multiple_of_17: ℕ :=
  102

theorem prove_minimality_of_smallest_three_digit_multiple_of_17 (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k ∧ 100 ≤ 17 * k ∧ 17 * k < 1000) 
  → n = 102 :=
    by
      sorry

end prove_minimality_of_smallest_three_digit_multiple_of_17_l267_267907


namespace find_x_l267_267222

theorem find_x (a y x : ℤ) (h1 : y = 3) (h2 : a * y + x = 10) (h3 : a = 3) : x = 1 :=
by 
  sorry

end find_x_l267_267222


namespace regression_line_equation_l267_267284

-- Define the conditions in the problem
def slope_of_regression_line : ℝ := 1.23
def center_of_sample_points : ℝ × ℝ := (4, 5)

-- The proof problem to show that the equation of the regression line is y = 1.23x + 0.08
theorem regression_line_equation :
  ∃ b : ℝ, (∀ x y : ℝ, (y = slope_of_regression_line * x + b) 
  → (4, 5) = (x, y)) → b = 0.08 :=
sorry

end regression_line_equation_l267_267284


namespace inverse_h_l267_267168

def f (x : ℝ) : ℝ := 5 * x - 7
def g (x : ℝ) : ℝ := 3 * x + 2
def h (x : ℝ) : ℝ := f (g x)

theorem inverse_h : (∀ x : ℝ, h (15 * x + 3) = x) :=
by
  -- Proof would go here
  sorry

end inverse_h_l267_267168


namespace mitchell_more_than_antonio_l267_267171

-- Definitions based on conditions
def mitchell_pencils : ℕ := 30
def total_pencils : ℕ := 54

-- Definition of the main question
def antonio_pencils : ℕ := total_pencils - mitchell_pencils

-- The theorem to be proved
theorem mitchell_more_than_antonio : mitchell_pencils - antonio_pencils = 6 :=
by
-- Proof is omitted
sorry

end mitchell_more_than_antonio_l267_267171


namespace cos_arcsin_l267_267997

theorem cos_arcsin (h : 8^2 + 15^2 = 17^2) : Real.cos (Real.arcsin (8 / 17)) = 15 / 17 :=
by
  have opp_hyp_pos : 0 ≤ 8 / 17 := by norm_num
  have hyp_pos : 0 < 17 := by norm_num
  have opp_le_hyp : 8 / 17 ≤ 1 := by norm_num
  sorry

end cos_arcsin_l267_267997


namespace complex_number_is_real_implies_m_eq_3_l267_267067

open Complex

theorem complex_number_is_real_implies_m_eq_3 (m : ℝ) :
  (∃ (z : ℂ), z = (1 / (m + 5) : ℝ) + (m^2 + 2 * m - 15) * I ∧ z.im = 0) →
  m = 3 :=
by
  sorry

end complex_number_is_real_implies_m_eq_3_l267_267067


namespace average_homework_time_decrease_l267_267401

theorem average_homework_time_decrease (x : ℝ) :
  (100 * (1 - x)^2 = 70) :=
sorry

end average_homework_time_decrease_l267_267401


namespace jenny_profit_l267_267600

-- Define the constants given in the problem
def cost_per_pan : ℝ := 10.00
def price_per_pan : ℝ := 25.00
def num_pans : ℝ := 20.0

-- Define the total revenue function
def total_revenue (num_pans : ℝ) (price_per_pan : ℝ) : ℝ := num_pans * price_per_pan

-- Define the total cost function
def total_cost (num_pans : ℝ) (cost_per_pan : ℝ) : ℝ := num_pans * cost_per_pan

-- Define the profit function as the total revenue minus the total cost
def total_profit (num_pans : ℝ) (price_per_pan : ℝ) (cost_per_pan : ℝ) : ℝ := 
  total_revenue num_pans price_per_pan - total_cost num_pans cost_per_pan

-- The statement to prove in Lean
theorem jenny_profit : total_profit num_pans price_per_pan cost_per_pan = 300.00 := 
by 
  sorry

end jenny_profit_l267_267600


namespace smallest_three_digit_multiple_of_17_l267_267884

theorem smallest_three_digit_multiple_of_17 : 
    ∃ (n : ℤ), n ≥ 100 ∧ n < 1000 ∧ (∃ (k : ℤ), n = 17 * k) ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l267_267884


namespace sum_of_digits_l267_267197

theorem sum_of_digits :
  ∃ (a b : ℕ), (4 * 100 + a * 10 + 5) + 457 = (9 * 100 + b * 10 + 2) ∧
                (((9 + 2) - b) % 11 = 0) ∧
                (a + b = 4) :=
sorry

end sum_of_digits_l267_267197


namespace smallest_three_digit_multiple_of_17_l267_267928

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  use 102,
  split,
  { exact le_of_lt (nat.lt_of_sub_eq_succ rfl), }, -- 102 is greater than or equal to 100
  split,
  { exact nat.lt_of_sub_eq_succ rfl, }, -- 102 is less than 1000
  split,
  { exact nat.dvd_intro_left (6) rfl, }, -- 17 divides 102
  { intros m hm1 hm2 hm3,
    exact nat.le_of_dvd hm1 (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (hm3)))), }, -- 102 is the smallest
end

end smallest_three_digit_multiple_of_17_l267_267928


namespace smallest_three_digit_multiple_of_17_l267_267857

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → n ≤ m) :=
by
  use 102
  have h1 : 100 ≤ 102 := by linarith
  have h2 : 102 ≤ 999 := by linarith
  have h3 : 102 % 17 = 0 := by norm_num
  have h4 : ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → 102 ≤ m := by
    intro m
    intro h
    have h_m1 := h.1
    have h_m2 := h.2.1
    have h_m3 := h.2.2
    have h_min := nat.le_of_dvd (by linarith) h_m3
    exact h_min
  exact ⟨⟨h1, h2⟩, h3, h4⟩

end smallest_three_digit_multiple_of_17_l267_267857


namespace all_terms_are_integers_l267_267046

-- Define the sequence a_n
def a : ℕ → ℕ
| 0       := 1
| 1       := 1
| 2       := 2
| (n+3)   := (a (n+2) * a (n+1) + Nat.factorial n) / a n

-- Define the theorem that all terms in the sequence are integers
theorem all_terms_are_integers : ∀ n : ℕ, ∃ k : ℕ, a n = k :=
by sorry

end all_terms_are_integers_l267_267046


namespace sum_of_roots_l267_267066

-- Defined the equation x^2 - 7x + 2 - 16 = 0 as x^2 - 7x - 14 = 0
def equation (x : ℝ) := x^2 - 7 * x - 14 = 0 

-- State the theorem leveraging the above condition
theorem sum_of_roots : 
  (∃ x1 x2 : ℝ, equation x1 ∧ equation x2 ∧ x1 ≠ x2) →
  (∃ sum : ℝ, sum = 7) := by
  sorry

end sum_of_roots_l267_267066


namespace neg_five_power_zero_simplify_expression_l267_267671

-- Proof statement for the first question.
theorem neg_five_power_zero : (-5 : ℝ)^0 = 1 := 
by sorry

-- Proof statement for the second question.
theorem simplify_expression (a b : ℝ) : ((-2 * a^2)^2) * (3 * a * b^2) = 12 * a^5 * b^2 := 
by sorry

end neg_five_power_zero_simplify_expression_l267_267671


namespace inequality_range_l267_267301

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |2 * x + 1| - |2 * x - 1| < a) → a > 2 :=
by
  sorry

end inequality_range_l267_267301


namespace find_monthly_salary_l267_267546

variable (S : ℝ)

theorem find_monthly_salary
  (h1 : 0.20 * S - 0.20 * (0.20 * S) = 220) :
  S = 1375 :=
by
  -- Proof goes here
  sorry

end find_monthly_salary_l267_267546


namespace fraction_zero_imp_x_eq_two_l267_267012
open Nat Real

theorem fraction_zero_imp_x_eq_two (x : ℝ) (h: (2 - abs x) / (x + 2) = 0) : x = 2 :=
by
  have h1 : 2 - abs x = 0 := sorry
  have h2 : x + 2 ≠ 0 := sorry
  sorry

end fraction_zero_imp_x_eq_two_l267_267012


namespace channel_bottom_width_l267_267187

theorem channel_bottom_width
  (area : ℝ)
  (top_width : ℝ)
  (depth : ℝ)
  (h_area : area = 880)
  (h_top_width : top_width = 14)
  (h_depth : depth = 80) :
  ∃ (b : ℝ), b = 8 ∧ area = (1/2) * (top_width + b) * depth := 
by
  sorry

end channel_bottom_width_l267_267187


namespace flowers_sold_difference_l267_267689

def number_of_daisies_sold_on_second_day (d2 : ℕ) (d3 : ℕ) (d_sum : ℕ) : Prop :=
  d3 = 2 * d2 - 10 ∧
  d_sum = 45 + d2 + d3 + 120

theorem flowers_sold_difference (d2 : ℕ) (d3 : ℕ) (d_sum : ℕ) 
  (h : number_of_daisies_sold_on_second_day d2 d3 d_sum) :
  45 + d2 + d3 + 120 = 350 → 
  d2 - 45 = 20 := 
by
  sorry

end flowers_sold_difference_l267_267689


namespace parabola_line_intersection_distance_l267_267044

theorem parabola_line_intersection_distance :
  ∀ (x y : ℝ), x^2 = -4 * y ∧ y = x - 1 ∧ x^2 + 4 * x + 4 = 0 →
  abs (y - -1 + (-1 - y)) = 8 :=
by
  sorry

end parabola_line_intersection_distance_l267_267044


namespace f_strictly_decreasing_l267_267701

-- Define the function g(x) = x^2 - 2x - 3
def g (x : ℝ) : ℝ := x^2 - 2 * x - 3

-- Define the function f(x) = log_{1/2}(g(x))
noncomputable def f (x : ℝ) : ℝ := Real.log (g x) / Real.log (1 / 2)

-- The problem statement to prove: f(x) is strictly decreasing on the interval (3, ∞)
theorem f_strictly_decreasing : ∀ x y : ℝ, 3 < x → x < y → f y < f x := by
  sorry

end f_strictly_decreasing_l267_267701


namespace CarmenBrushLengthIsCorrect_l267_267108

namespace BrushLength

def carlasBrushLengthInInches : ℤ := 12
def conversionRateInCmPerInch : ℝ := 2.5
def lengthMultiplier : ℝ := 1.5

def carmensBrushLengthInCm : ℝ :=
  carlasBrushLengthInInches * lengthMultiplier * conversionRateInCmPerInch

theorem CarmenBrushLengthIsCorrect :
  carmensBrushLengthInCm = 45 := by
  sorry

end BrushLength

end CarmenBrushLengthIsCorrect_l267_267108


namespace chasity_candies_l267_267688

theorem chasity_candies :
  let lollipop_cost := 1.5
  let gummy_pack_cost := 2
  let initial_amount := 15
  let lollipops_bought := 4
  let gummies_bought := 2
  let total_spent := lollipops_bought * lollipop_cost + gummies_bought * gummy_pack_cost
  initial_amount - total_spent = 5 :=
by
  -- Let definitions of constants
  let lollipop_cost := 1.5
  let gummy_pack_cost := 2
  let initial_amount := 15
  let lollipops_bought := 4
  let gummies_bought := 2
  -- Total cost calculation
  let total_spent := lollipops_bought * lollipop_cost + gummies_bought * gummy_pack_cost
  -- Proof of the final amount left
  have h : initial_amount - total_spent = 15 - (4 * 1.5 + 2 * 2) := rfl
  simp at h
  exact h

end chasity_candies_l267_267688


namespace evaluate_magnitude_l267_267119

noncomputable def mag1 : ℂ := 3 * Real.sqrt 2 - 3 * Complex.I
noncomputable def mag2 : ℂ := Real.sqrt 5 + 5 * Complex.I
noncomputable def mag3 : ℂ := 2 - 2 * Complex.I

theorem evaluate_magnitude :
  Complex.abs (mag1 * mag2 * mag3) = 18 * Real.sqrt 10 :=
by
  sorry

end evaluate_magnitude_l267_267119


namespace problem1_problem2_l267_267561

-- Problem 1 Proof Statement
theorem problem1 : Real.sin (30 * Real.pi / 180) + abs (-1) - (Real.sqrt 3 - Real.pi) ^ 0 = 1 / 2 := 
  by sorry

-- Problem 2 Proof Statement
theorem problem2 (x: ℝ) (hx : x ≠ 2) : (2 * x - 3) / (x - 2) - (x - 1) / (x - 2) = 1 := 
  by sorry

end problem1_problem2_l267_267561


namespace coord_of_point_M_in_third_quadrant_l267_267583

noncomputable def point_coordinates (x y : ℝ) : Prop :=
  x < 0 ∧ y < 0 ∧ abs y = 1 ∧ abs x = 2

theorem coord_of_point_M_in_third_quadrant : 
  ∃ (x y : ℝ), point_coordinates x y ∧ (x, y) = (-2, -1) := 
by {
  sorry
}

end coord_of_point_M_in_third_quadrant_l267_267583


namespace frac_subtraction_simplified_l267_267103

-- Definitions of the fractions involved.
def frac1 : ℚ := 8 / 19
def frac2 : ℚ := 5 / 57

-- The primary goal is to prove the equality.
theorem frac_subtraction_simplified : frac1 - frac2 = 1 / 3 :=
by {
  -- Proof of the statement.
  sorry
}

end frac_subtraction_simplified_l267_267103


namespace scientific_notation_l267_267080

theorem scientific_notation (h : 0.000000007 = 7 * 10^(-9)) : 0.000000007 = 7 * 10^(-9) :=
by
  sorry

end scientific_notation_l267_267080


namespace correct_calculation_l267_267068

-- Define the base type for exponents
variables (a : ℝ)

theorem correct_calculation :
  (a^3 * a^5 = a^8) ∧ 
  ¬((a^3)^2 = a^5) ∧ 
  ¬(a^5 + a^2 = a^7) ∧ 
  ¬(a^6 / a^2 = a^3) :=
by
  sorry

end correct_calculation_l267_267068


namespace returned_books_percentage_is_correct_l267_267676

-- This function takes initial_books, end_books, and loaned_books and computes the percentage of books returned.
noncomputable def percent_books_returned (initial_books : ℕ) (end_books : ℕ) (loaned_books : ℕ) : ℚ :=
  let books_out_on_loan := initial_books - end_books
  let books_returned := loaned_books - books_out_on_loan
  (books_returned : ℚ) / (loaned_books : ℚ) * 100

-- The main theorem that states the percentage of books returned is 70%
theorem returned_books_percentage_is_correct :
  percent_books_returned 75 57 60 = 70 := by
  sorry

end returned_books_percentage_is_correct_l267_267676


namespace smallest_repeating_block_fraction_3_over_11_l267_267473

theorem smallest_repeating_block_fraction_3_over_11 :
  ∃ n : ℕ, (∃ d : ℕ, (3/11 : ℚ) = d / 10^n) ∧ n = 2 :=
sorry

end smallest_repeating_block_fraction_3_over_11_l267_267473


namespace simple_interest_difference_l267_267058

theorem simple_interest_difference :
  let P : ℝ := 900
  let R1 : ℝ := 4
  let R2 : ℝ := 4.5
  let T : ℝ := 7
  let SI1 := P * R1 * T / 100
  let SI2 := P * R2 * T / 100
  SI2 - SI1 = 31.50 := by
  sorry

end simple_interest_difference_l267_267058


namespace proof_problem_l267_267151

open Real

-- Definitions of curves and transformations
def C1 := { p : ℝ × ℝ | p.1^2 + p.2^2 = 4 }
def C2 := { p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1 }

-- Parametric equation of C2
def parametric_C2 := ∃ α : ℝ, (0 ≤ α ∧ α ≤ 2*π) ∧
  (C2 = { p : ℝ × ℝ | p.1 = 2 * cos α ∧ p.2 = (1/2) * sin α })

-- Equation of line l1 maximizing the perimeter of ABCD
def line_l1 (p : ℝ × ℝ): Prop :=
  p.2 = (1/4) * p.1

theorem proof_problem : parametric_C2 ∧
  ∀ (A B C D : ℝ × ℝ),
    (A ∈ C2 ∧ B ∈ C2 ∧ C ∈ C2 ∧ D ∈ C2) →
    (line_l1 A ∧ line_l1 B) → 
    (line_l1 A ∧ line_l1 B) ∧
    (line_l1 C ∧ line_l1 D) →
    y = (1 / 4) * x :=
sorry

end proof_problem_l267_267151


namespace smallest_three_digit_multiple_of_17_l267_267838

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m := by
  sorry

end smallest_three_digit_multiple_of_17_l267_267838


namespace smallest_three_digit_multiple_of_17_l267_267931

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  use 102,
  split,
  { exact le_of_lt (nat.lt_of_sub_eq_succ rfl), }, -- 102 is greater than or equal to 100
  split,
  { exact nat.lt_of_sub_eq_succ rfl, }, -- 102 is less than 1000
  split,
  { exact nat.dvd_intro_left (6) rfl, }, -- 17 divides 102
  { intros m hm1 hm2 hm3,
    exact nat.le_of_dvd hm1 (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (hm3)))), }, -- 102 is the smallest
end

end smallest_three_digit_multiple_of_17_l267_267931


namespace correct_amendment_statements_l267_267376

/-- The amendment includes the abuse of administrative power by administrative organs 
    to exclude or limit competition. -/
def abuse_of_power_in_amendment : Prop :=
  true

/-- The amendment includes illegal fundraising. -/
def illegal_fundraising_in_amendment : Prop :=
  true

/-- The amendment includes apportionment of expenses. -/
def apportionment_of_expenses_in_amendment : Prop :=
  true

/-- The amendment includes failure to pay minimum living allowances or social insurance benefits according to law. -/
def failure_to_pay_benefits_in_amendment : Prop :=
  true

/-- The amendment further standardizes the exercise of government power. -/
def standardizes_govt_power : Prop :=
  true

/-- The amendment better protects the legitimate rights and interests of citizens. -/
def protects_rights : Prop :=
  true

/-- The amendment expands the channels for citizens' democratic participation. -/
def expands_democratic_participation : Prop :=
  false

/-- The amendment expands the scope of government functions. -/
def expands_govt_functions : Prop :=
  false

/-- The correct answer to which set of statements is true about the amendment is {②, ③}.
    This is encoded as proving (standardizes_govt_power ∧ protects_rights) = true. -/
theorem correct_amendment_statements : (standardizes_govt_power ∧ protects_rights) ∧ 
                                      ¬(expands_democratic_participation ∧ expands_govt_functions) :=
by {
  sorry
}

end correct_amendment_statements_l267_267376


namespace parabola_example_l267_267138

theorem parabola_example (p : ℝ) (hp : p > 0)
    (h_intersect : ∀ x y : ℝ, y = x - p / 2 ∧ y^2 = 2 * p * x → ((x - p / 2)^2 = 2 * p * x))
    (h_AB : ∀ A B : ℝ × ℝ, A.2 = A.1 - p / 2 ∧ B.2 = B.1 - p / 2 ∧ |A.1 - B.1| = 8) :
    p = 2 := 
sorry

end parabola_example_l267_267138


namespace smallest_three_digit_multiple_of_17_l267_267839

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m := by
  sorry

end smallest_three_digit_multiple_of_17_l267_267839


namespace comparison_of_y1_and_y2_l267_267153

variable {k y1 y2 : ℝ}

theorem comparison_of_y1_and_y2 (hk : 0 < k)
    (hy1 : y1 = k)
    (hy2 : y2 = k / 4) :
    y1 > y2 := by
  sorry

end comparison_of_y1_and_y2_l267_267153


namespace masha_can_pay_exactly_with_11_ruble_bills_l267_267229

theorem masha_can_pay_exactly_with_11_ruble_bills (m n k p : ℕ) 
  (h1 : 3 * m + 4 * n + 5 * k = 11 * p) : 
  ∃ q : ℕ, 9 * m + n + 4 * k = 11 * q := 
by {
  sorry
}

end masha_can_pay_exactly_with_11_ruble_bills_l267_267229


namespace problem_statement_l267_267560

-- Definitions corresponding to the given condition
noncomputable def sum_to_n (n : ℕ) : ℤ := (n * (n + 1)) / 2
noncomputable def alternating_sum_to_n (n : ℕ) : ℤ := if n % 2 = 0 then -(n / 2) else (n / 2 + 1)

-- Lean statement for the problem
theorem problem_statement :
  (alternating_sum_to_n 2022) * (sum_to_n 2023 - 1) - (alternating_sum_to_n 2023) * (sum_to_n 2022 - 1) = 2023 :=
sorry

end problem_statement_l267_267560


namespace smallest_three_digit_multiple_of_17_l267_267855

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → n ≤ m) :=
by
  use 102
  have h1 : 100 ≤ 102 := by linarith
  have h2 : 102 ≤ 999 := by linarith
  have h3 : 102 % 17 = 0 := by norm_num
  have h4 : ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → 102 ≤ m := by
    intro m
    intro h
    have h_m1 := h.1
    have h_m2 := h.2.1
    have h_m3 := h.2.2
    have h_min := nat.le_of_dvd (by linarith) h_m3
    exact h_min
  exact ⟨⟨h1, h2⟩, h3, h4⟩

end smallest_three_digit_multiple_of_17_l267_267855


namespace problem1_problem2_l267_267504

namespace MathProofs

theorem problem1 : (0.25 * 4 - ((5 / 6) + (1 / 12)) * (6 / 5)) = (1 / 10) := by
  sorry

theorem problem2 : ((5 / 12) - (5 / 16)) * (4 / 5) + (2 / 3) - (3 / 4) = 0 := by
  sorry

end MathProofs

end problem1_problem2_l267_267504


namespace exchange_positions_l267_267051

theorem exchange_positions : ∀ (people : ℕ), people = 8 → (∃ (ways : ℕ), ways = 336) :=
by sorry

end exchange_positions_l267_267051


namespace general_term_formaula_sum_of_seq_b_l267_267606

noncomputable def seq_a (n : ℕ) := 2 * n + 1

noncomputable def seq_b (n : ℕ) := 1 / ((seq_a n)^2 - 1)

noncomputable def sum_seq_a (n : ℕ) := (Finset.range n).sum seq_a

noncomputable def sum_seq_b (n : ℕ) := (Finset.range n).sum seq_b

theorem general_term_formaula (n : ℕ) (h : n > 0) :
  (∀ n, (seq_a n) > 0) ∧ (∀ n, (seq_a n)^2 + 2 * (seq_a n) = 4 * (sum_seq_a n) + 3) →
  seq_a n = 2 * n + 1 :=
by
  intros
  sorry

theorem sum_of_seq_b (n : ℕ) (h : n > 0) :
  (∀ n, (seq_a n) > 0) ∧ (∀ n, (seq_a n)^2 + 2 * (seq_a n) = 4 * (sum_seq_a n) + 3) →
  sum_seq_b n = n / (4 * (n + 1)) :=
by
  intros
  sorry

end general_term_formaula_sum_of_seq_b_l267_267606


namespace wendy_time_correct_l267_267685

variable (bonnie_time wendy_difference : ℝ)

theorem wendy_time_correct (h1 : bonnie_time = 7.80) (h2 : wendy_difference = 0.25) : 
  (bonnie_time - wendy_difference = 7.55) :=
by
  sorry

end wendy_time_correct_l267_267685


namespace half_angle_in_first_quadrant_l267_267715

theorem half_angle_in_first_quadrant (α : ℝ) (h : 0 < α ∧ α < π / 2) : 0 < α / 2 ∧ α / 2 < π / 4 :=
by
  sorry

end half_angle_in_first_quadrant_l267_267715


namespace rectangle_perimeter_l267_267358

theorem rectangle_perimeter (u v : ℝ) (π : ℝ) (major minor : ℝ) (area_rect area_ellipse : ℝ) 
  (inscribed : area_ellipse = 4032 * π ∧ area_rect = 4032 ∧ major = 2 * (u + v)) :
  2 * (u + v) = 128 := by
  -- Given: the area of the rectangle, the conditions of the inscribed ellipse, and the major axis constraint.
  sorry

end rectangle_perimeter_l267_267358


namespace tino_jellybeans_l267_267650

variable (Tino Lee Arnold : ℕ)

-- Conditions
variable (h1 : Tino = Lee + 24)
variable (h2 : Arnold = Lee / 2)
variable (h3 : Arnold = 5)

-- Prove Tino has 34 jellybeans
theorem tino_jellybeans : Tino = 34 :=
by
  sorry

end tino_jellybeans_l267_267650


namespace infinite_sum_problem_l267_267567

theorem infinite_sum_problem :
  (∑' n : ℕ, if n = 0 then 0 else (3^n) / (1 + 3^n + 3^(n + 1) + 3^(2 * n + 1))) = (1 / 4) := 
by
  sorry

end infinite_sum_problem_l267_267567


namespace nonzero_real_x_satisfies_equation_l267_267532

theorem nonzero_real_x_satisfies_equation :
  ∃ x : ℝ, x ≠ 0 ∧ (7 * x) ^ 5 = (14 * x) ^ 4 ∧ x = 16 / 7 :=
by
  sorry

end nonzero_real_x_satisfies_equation_l267_267532


namespace chastity_leftover_money_l267_267687

theorem chastity_leftover_money (n_lollipops : ℕ) (price_lollipop : ℝ) (n_gummies : ℕ) (price_gummy : ℝ) (initial_money : ℝ) :
  n_lollipops = 4 →
  price_lollipop = 1.50 →
  n_gummies = 2 →
  price_gummy = 2 →
  initial_money = 15 →
  initial_money - ((n_lollipops * price_lollipop) + (n_gummies * price_gummy)) = 5 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end chastity_leftover_money_l267_267687


namespace trajectory_eq_circle_through_fixed_points_l267_267750

-- Definitions based on the conditions 
def dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Proof problem 1
theorem trajectory_eq (M : ℝ × ℝ) (h : dist M (1, 0) = Real.abs M.1 + 1) :
  (M.1 >= 0 -> M.2^2 = 4 * M.1) ∧ (M.1 < 0 -> M.2 = 0) := sorry

noncomputable def C (x : ℝ) := if x >= 0 then (some (λ y : ℝ, y^2 = 4 * x)) else (0)

-- Proof problem 2
theorem circle_through_fixed_points (A B : ℝ × ℝ) (hC : ∀ x, A = (1, (4 / (some (λ y, y = x))))) :
  ∃ (fixed_points : ℝ × ℝ), fixed_points = (-1,0) ∨ fixed_points = (3,0) := sorry

end trajectory_eq_circle_through_fixed_points_l267_267750


namespace move_point_A_l267_267751

theorem move_point_A :
  let A := (-5, 6)
  let A_right := (A.1 + 5, A.2)
  let A_upwards := (A_right.1, A_right.2 + 6)
  A_upwards = (0, 12) := by
  sorry

end move_point_A_l267_267751


namespace average_of_rest_l267_267482

theorem average_of_rest 
  (total_students : ℕ)
  (marks_5_students : ℕ)
  (marks_3_students : ℕ)
  (marks_others : ℕ)
  (average_class : ℚ)
  (remaining_students : ℕ)
  (expected_average : ℚ) 
  (h1 : total_students = 27) 
  (h2 : marks_5_students = 5 * 95) 
  (h3 : marks_3_students = 3 * 0) 
  (h4 : average_class = 49.25925925925926) 
  (h5 : remaining_students = 27 - 5 - 3) 
  (h6 : (marks_5_students + marks_3_students + marks_others) = total_students * average_class)
  : marks_others / remaining_students = expected_average :=
sorry

end average_of_rest_l267_267482


namespace prove_minimality_of_smallest_three_digit_multiple_of_17_l267_267900

def smallest_three_digit_multiple_of_17: ℕ :=
  102

theorem prove_minimality_of_smallest_three_digit_multiple_of_17 (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k ∧ 100 ≤ 17 * k ∧ 17 * k < 1000) 
  → n = 102 :=
    by
      sorry

end prove_minimality_of_smallest_three_digit_multiple_of_17_l267_267900


namespace ceiling_is_multiple_of_3_l267_267811

-- Given conditions:
def polynomial (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1
axiom exists_three_real_roots : ∃ x1 x2 x3 : ℝ, x1 < x2 ∧ x2 < x3 ∧
  polynomial x1 = 0 ∧ polynomial x2 = 0 ∧ polynomial x3 = 0

-- Goal:
theorem ceiling_is_multiple_of_3 (x1 x2 x3 : ℝ) (h1 : x1 < x2) (h2 : x2 < x3)
  (hx1 : polynomial x1 = 0) (hx2 : polynomial x2 = 0) (hx3 : polynomial x3 = 0):
  ∀ n : ℕ, n > 0 → ∃ k : ℤ, k * 3 = ⌈x3^n⌉ := by
  sorry

end ceiling_is_multiple_of_3_l267_267811


namespace root_equation_m_l267_267298

theorem root_equation_m (m : ℝ) : 
  (∃ (x : ℝ), x = -1 ∧ m*x^2 + x - m^2 + 1 = 0) → m = 1 :=
by 
  sorry

end root_equation_m_l267_267298


namespace triangle_area_proof_l267_267594

noncomputable def triangle_area (a b c : ℝ) (B : ℝ) : ℝ :=
  0.5 * a * c * Real.sin B

theorem triangle_area_proof (a b c : ℝ) (B : ℝ) (hB : B = 2 * Real.pi / 3) (hb : b = Real.sqrt 13) (h_sum : a + c = 4) :
  triangle_area a b c B = 3 * Real.sqrt 3 / 4 :=
by
  sorry

end triangle_area_proof_l267_267594


namespace damage_conversion_l267_267082

def usd_to_cad_conversion_rate : ℝ := 1.25
def damage_in_usd : ℝ := 60000000
def damage_in_cad : ℝ := 75000000

theorem damage_conversion :
  damage_in_usd * usd_to_cad_conversion_rate = damage_in_cad :=
sorry

end damage_conversion_l267_267082


namespace fbox_eval_correct_l267_267271

-- Define the function according to the condition
def fbox (a b c : ℕ) : ℕ := a^b - b^c + c^a

-- Propose the theorem 
theorem fbox_eval_correct : fbox 2 0 3 = 10 := 
by
  -- Proof will be provided here
  sorry

end fbox_eval_correct_l267_267271


namespace ratio_SI_CI_l267_267642

-- Defining parameters and conditions.
def P_SI : ℝ := 1750  -- Principal for Simple Interest
def R_SI : ℝ := 8    -- Rate for Simple Interest
def T_SI : ℝ := 3    -- Time for Simple Interest

def P_CI : ℝ := 4000  -- Principal for Compound Interest
def R_CI : ℝ := 10    -- Rate for Compound Interest
def T_CI : ℝ := 2     -- Time for Compound Interest

-- Defining Simple Interest formula calculation.
def SI (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * R * T / 100

-- Defining Compound Interest formula calculation.
def CI (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * ((1 + R / 100) ^ T - 1)

-- Using the given problem definitions in Lean functions.
theorem ratio_SI_CI :
  let simple_interest := SI P_SI R_SI T_SI,
      compound_interest := CI P_CI R_CI T_CI
  in simple_interest / compound_interest = 1 / 2 := sorry

end ratio_SI_CI_l267_267642


namespace range_of_k_l267_267725

theorem range_of_k (x y k : ℝ) 
  (h1 : 2 * x + y = k + 1) 
  (h2 : x + 2 * y = 2) 
  (h3 : x + y < 0) : 
  k < -3 :=
sorry

end range_of_k_l267_267725


namespace average_output_l267_267666

theorem average_output (time1 time2 rate1 rate2 cogs1 cogs2 total_cogs total_time: ℝ) :
  rate1 = 20 → cogs1 = 60 → time1 = cogs1 / rate1 →
  rate2 = 60 → cogs2 = 60 → time2 = cogs2 / rate2 →
  total_cogs = cogs1 + cogs2 → total_time = time1 + time2 →
  (total_cogs / total_time = 30) :=
by
  intros hrate1 hcogs1 htime1 hrate2 hcogs2 htime2 htotalcogs htotaltime
  sorry

end average_output_l267_267666


namespace find_f_half_l267_267281

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_half (x : ℝ) (h₀ : 0 ≤ x ∧ x ≤ Real.pi / 2) (h₁ : f (Real.sin x) = x) : 
  f (1 / 2) = Real.pi / 6 :=
sorry

end find_f_half_l267_267281


namespace sufficient_but_not_necessary_l267_267461

theorem sufficient_but_not_necessary (x : ℝ) (h : x > 4) :
  (x ^ 2 - 5 * x + 4 ≥ 0 ∧ ¬(∀ x, (x ^ 2 - 5 * x + 4 ≥ 0 → x > 4))) :=
by
  sorry

end sufficient_but_not_necessary_l267_267461


namespace annual_interest_rate_is_12_percent_l267_267524

theorem annual_interest_rate_is_12_percent
  (P : ℕ := 750000)
  (I : ℕ := 37500)
  (t : ℕ := 5)
  (months_in_year : ℕ := 12)
  (annual_days : ℕ := 360)
  (days_per_month : ℕ := 30) :
  ∃ r : ℚ, (r * 100 * months_in_year = 12) ∧ I = P * r * t := 
sorry

end annual_interest_rate_is_12_percent_l267_267524


namespace inequality_count_l267_267449

theorem inequality_count {a b : ℝ} (h : 1/a < 1/b ∧ 1/b < 0) :
  (if (|a| > |b|) then 0 else 1) + 
  (if (a + b > ab) then 1 else 0) +
  (if (a / b + b / a > 2) then 1 else 0) + 
  (if (a^2 / b < 2 * a - b) then 1 else 0) = 2 :=
sorry

end inequality_count_l267_267449


namespace smallest_three_digit_multiple_of_17_l267_267927

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  use 102,
  split,
  { exact le_of_lt (nat.lt_of_sub_eq_succ rfl), }, -- 102 is greater than or equal to 100
  split,
  { exact nat.lt_of_sub_eq_succ rfl, }, -- 102 is less than 1000
  split,
  { exact nat.dvd_intro_left (6) rfl, }, -- 17 divides 102
  { intros m hm1 hm2 hm3,
    exact nat.le_of_dvd hm1 (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (hm3)))), }, -- 102 is the smallest
end

end smallest_three_digit_multiple_of_17_l267_267927


namespace math_proof_problem_l267_267792

def expr (m : ℝ) : ℝ := (1 - (2 / (m + 1))) / ((m ^ 2 - 2 * m + 1) / (m ^ 2 - m))

theorem math_proof_problem :
  expr (Real.tan (Real.pi / 3) - 1) = (3 - Real.sqrt 3) / 3 :=
  sorry

end math_proof_problem_l267_267792


namespace cakesServedDuringDinner_today_is_6_l267_267988

def cakesServedDuringDinner (x : ℕ) : Prop :=
  5 + x + 3 = 14

theorem cakesServedDuringDinner_today_is_6 : cakesServedDuringDinner 6 :=
by
  unfold cakesServedDuringDinner
  -- The proof is omitted
  sorry

end cakesServedDuringDinner_today_is_6_l267_267988


namespace equality_proof_l267_267582

variable {a b c : ℝ}

theorem equality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 / (b + c) + b^2 / (c + a) + c^2 / (a + b)) ≥ (1 / 2) * (a + b + c) :=
by
  sorry

end equality_proof_l267_267582


namespace geometric_series_sum_l267_267686

theorem geometric_series_sum :
  let a := (2 : ℚ)
  let r := (1 / 3 : ℚ)
  let n := 6
  (a * (1 - r^n) / (1 - r) = 728 / 243) := 
by
  let a := (2 : ℚ)
  let r := (1 / 3 : ℚ)
  let n := 6
  show a * (1 - r^n) / (1 - r) = 728 / 243
  sorry

end geometric_series_sum_l267_267686


namespace find_m_n_l267_267070

def is_prime (n : Nat) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

theorem find_m_n (p k : ℕ) (hk : 1 < k) (hp : is_prime p) : 
  (∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ (m, n) ≠ (1, 1) ∧ (m^p + n^p) / 2 = (m + n) / 2 ^ k) ↔ k = p :=
sorry

end find_m_n_l267_267070


namespace speed_of_other_train_l267_267059

theorem speed_of_other_train (len1 len2 time : ℝ) (v1 v_other : ℝ) :
  len1 = 200 ∧ len2 = 300 ∧ time = 17.998560115190788 ∧ v1 = 40 →
  v_other = ((len1 + len2) / 1000) / (time / 3600) - v1 :=
by
  intros
  sorry

end speed_of_other_train_l267_267059


namespace set_equality_implies_sum_zero_l267_267495

theorem set_equality_implies_sum_zero
  (x y : ℝ)
  (A : Set ℝ := {x, y, x + y})
  (B : Set ℝ := {0, x^2, x * y}) :
  A = B → x + y = 0 :=
by
  sorry

end set_equality_implies_sum_zero_l267_267495


namespace sum_first_11_terms_l267_267484

variable {a : ℕ → ℕ} -- a is the arithmetic sequence

-- Condition: a_4 + a_8 = 26
axiom condition : a 4 + a 8 = 26

-- Definition of arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

-- Definition of the sum of the first 11 terms
def S_11 (a : ℕ → ℕ) : ℕ := (11 * (a 1 + a 11)) / 2

-- The proof problem statement
theorem sum_first_11_terms (h : is_arithmetic_sequence a) : S_11 a = 143 := 
by 
  sorry

end sum_first_11_terms_l267_267484


namespace average_homework_time_decrease_l267_267398

theorem average_homework_time_decrease (x : ℝ) :
  (100 * (1 - x)^2 = 70) :=
sorry

end average_homework_time_decrease_l267_267398


namespace gcd_A_C_gcd_B_C_l267_267112

def A : ℕ := 177^5 + 30621 * 173^3 - 173^5
def B : ℕ := 173^5 + 30621 * 177^3 - 177^5
def C : ℕ := 173^4 + 30621^2 + 177^4

theorem gcd_A_C : Nat.gcd A C = 30637 := sorry

theorem gcd_B_C : Nat.gcd B C = 30637 := sorry

end gcd_A_C_gcd_B_C_l267_267112


namespace odd_and_periodic_40_l267_267492

noncomputable def f : ℝ → ℝ := sorry

theorem odd_and_periodic_40
  (h₁ : ∀ x : ℝ, f (10 + x) = f (10 - x))
  (h₂ : ∀ x : ℝ, f (20 - x) = -f (20 + x)) :
  (∀ x : ℝ, f (-x) = -f (x)) ∧ (∀ x : ℝ, f (x + 40) = f (x)) :=
by
  sorry

end odd_and_periodic_40_l267_267492


namespace smallest_perimeter_consecutive_even_triangle_l267_267218

theorem smallest_perimeter_consecutive_even_triangle :
  ∃ (x : ℕ), x % 2 = 0 ∧ 
             x + 2 > 2 ∧ 
             x + 4 > 2 ∧ 
             x > 2 ∧ 
             (let sides := [x, x + 2, x + 4] in 
                (sides.sum) = 18) :=
by
  sorry

end smallest_perimeter_consecutive_even_triangle_l267_267218


namespace smallest_three_digit_multiple_of_17_l267_267847

theorem smallest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, m >= 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  let n := 102,
  use n,
  split,
  { norm_num, }, -- n >= 100 is obviously true because n = 102
  split,
  { norm_num, }, -- n < 1000 is obviously true because n = 102
  split,
  { exact dvd_refl 17, }, -- 17 divides 102
  { intros m hm,
    cases hm with h100 hm,
    cases hm with h1000 hdiv,
    exact le_trans (nat.le.intro _ (symm hdiv)) (le_of_not_gt (λ h, by norm_num at h)), },
end

end smallest_three_digit_multiple_of_17_l267_267847


namespace wrongly_noted_mark_is_90_l267_267036

-- Define the given conditions
def avg_marks (n : ℕ) (avg : ℚ) : ℚ := n * avg

def wrong_avg_marks : ℚ := avg_marks 10 100
def correct_avg_marks : ℚ := avg_marks 10 92

-- Equate the difference caused by the wrong mark
theorem wrongly_noted_mark_is_90 (x : ℚ) (h₁ : wrong_avg_marks = 1000) (h₂ : correct_avg_marks = 920) (h : x - 10 = 1000 - 920) : x = 90 := 
by {
  -- Proof goes here
  sorry
}

end wrongly_noted_mark_is_90_l267_267036


namespace tourist_tax_l267_267990

theorem tourist_tax (total_value : ℕ) (non_taxable_amount : ℕ) (tax_rate : ℚ) (tax : ℚ) : 
  total_value = 1720 → 
  non_taxable_amount = 600 → 
  tax_rate = 0.12 → 
  tax = (total_value - non_taxable_amount : ℕ) * tax_rate → 
  tax = 134.40 := 
by 
  intros total_value_eq non_taxable_amount_eq tax_rate_eq tax_eq
  sorry

end tourist_tax_l267_267990


namespace smallest_three_digit_multiple_of_17_l267_267930

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  use 102,
  split,
  { exact le_of_lt (nat.lt_of_sub_eq_succ rfl), }, -- 102 is greater than or equal to 100
  split,
  { exact nat.lt_of_sub_eq_succ rfl, }, -- 102 is less than 1000
  split,
  { exact nat.dvd_intro_left (6) rfl, }, -- 17 divides 102
  { intros m hm1 hm2 hm3,
    exact nat.le_of_dvd hm1 (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (hm3)))), }, -- 102 is the smallest
end

end smallest_three_digit_multiple_of_17_l267_267930


namespace p_at_0_l267_267605

noncomputable def p : Polynomial ℚ := sorry

theorem p_at_0 :
  (∀ n : ℕ, n ≤ 6 → p.eval (2^n) = 1 / (2^n))
  ∧ p.degree = 6 → 
  p.eval 0 = 127 / 64 :=
sorry

end p_at_0_l267_267605


namespace correct_option_is_C_l267_267223

namespace ExponentProof

-- Definitions of conditions
def optionA (a : ℝ) : Prop := a^3 * a^4 = a^12
def optionB (a : ℝ) : Prop := a^3 + a^4 = a^7
def optionC (a : ℝ) : Prop := a^5 / a^3 = a^2
def optionD (a : ℝ) : Prop := (-2 * a)^3 = -6 * a^3

-- Proof problem stating that optionC is the only correct one
theorem correct_option_is_C : ∀ (a : ℝ), ¬ optionA a ∧ ¬ optionB a ∧ optionC a ∧ ¬ optionD a :=
by
  intro a
  sorry

end ExponentProof

end correct_option_is_C_l267_267223


namespace distance_between_P_and_Q_l267_267415

theorem distance_between_P_and_Q : 
  let initial_speed := 40  -- Speed in kmph
  let increment := 20      -- Speed increment in kmph after every 12 minutes
  let segment_duration := 12 / 60 -- Duration of each segment in hours (12 minutes in hours)
  let total_duration := 48 / 60    -- Total duration in hours (48 minutes in hours)
  let total_segments := total_duration / segment_duration -- Number of segments
  (total_segments = 4) ∧ 
  (∀ n : ℕ, n ≥ 0 → n < total_segments → 
    let speed := initial_speed + n * increment
    let distance := speed * segment_duration
    distance = speed * (12 / 60)) 
  → (40 * (12 / 60) + 60 * (12 / 60) + 80 * (12 / 60) + 100 * (12 / 60)) = 56 :=
by
  sorry

end distance_between_P_and_Q_l267_267415


namespace smallest_three_digit_multiple_of_17_l267_267879

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := sorry

end smallest_three_digit_multiple_of_17_l267_267879


namespace mike_earnings_l267_267539

theorem mike_earnings (total_games non_working_games price_per_game : ℕ) 
  (h1 : total_games = 15) (h2 : non_working_games = 9) (h3 : price_per_game = 5) : 
  total_games - non_working_games * price_per_game = 30 :=
by
  rw [h1, h2, h3]
  show 15 - 9 * 5 = 30
  sorry

end mike_earnings_l267_267539


namespace maximum_value_of_w_l267_267711

variables (x y : ℝ)

def condition : Prop := x^2 + y^2 = 18 * x + 8 * y + 10

def w (x y : ℝ) := 4 * x + 3 * y

theorem maximum_value_of_w : ∃ x y, condition x y ∧ w x y = 74 :=
sorry

end maximum_value_of_w_l267_267711


namespace perpendicular_lines_l267_267286

theorem perpendicular_lines :
  ∃ m₁ m₄, (m₁ : ℚ) * (m₄ : ℚ) = -1 ∧
  (∀ x y : ℚ, 4 * y - 3 * x = 16 → y = m₁ * x + 4) ∧
  (∀ x y : ℚ, 3 * y + 4 * x = 15 → y = m₄ * x + 5) :=
by sorry

end perpendicular_lines_l267_267286


namespace probability_X_eq_3_l267_267720

def number_of_ways_to_choose (n k : ℕ) : ℕ :=
  Nat.choose n k

def P_X_eq_3 : ℚ :=
  (number_of_ways_to_choose 5 3) * (number_of_ways_to_choose 3 1) / (number_of_ways_to_choose 8 4)

theorem probability_X_eq_3 : P_X_eq_3 = 3 / 7 := by
  sorry

end probability_X_eq_3_l267_267720


namespace money_made_march_to_august_l267_267507

section
variable (H : ℕ)

-- Given conditions
def hoursMarchToAugust : ℕ := 23
def hoursSeptToFeb : ℕ := 8
def additionalHours : ℕ := 16
def totalCost : ℕ := 600 + 340
def totalHours : ℕ := hoursMarchToAugust + hoursSeptToFeb + additionalHours

-- Total money equation
def totalMoney : ℕ := totalHours * H

-- Theorem to prove the money made from March to August
theorem money_made_march_to_august : totalMoney = totalCost → hoursMarchToAugust * H = 460 :=
by
  intro h
  have hH : H = 20 := by
    sorry
  rw [hH]
  sorry
end

end money_made_march_to_august_l267_267507


namespace smallest_three_digit_multiple_of_17_l267_267959

theorem smallest_three_digit_multiple_of_17 : ∃ (k : ℕ), 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 :=
by
  use 6
  split
  · exact of_as_true trivial -- 100 <= 17 * k
  split
  · exact of_as_true trivial -- 17 * k <= 999
  · exact of_as_true trivial -- 17 * k = 102
  sorry

end smallest_three_digit_multiple_of_17_l267_267959


namespace solve_for_a_l267_267230

theorem solve_for_a (a x : ℝ) (h : (1 / 2) * x + a = -1) (hx : x = 2) : a = -2 :=
by
  sorry

end solve_for_a_l267_267230


namespace students_play_long_tennis_l267_267015

-- Define the parameters for the problem
def total_students : ℕ := 38
def football_players : ℕ := 26
def both_sports_players : ℕ := 17
def neither_sports_players : ℕ := 9

-- Total students playing at least one sport
def at_least_one := total_students - neither_sports_players

-- Define the Lean theorem statement
theorem students_play_long_tennis : at_least_one = football_players + (20 : ℕ) - both_sports_players := 
by 
  -- Translate the given facts into the Lean proof structure
  have h1 : at_least_one = 29 := by rfl -- total_students - neither_sports_players
  have h2 : football_players = 26 := by rfl
  have h3 : both_sports_players = 17 := by rfl
  show 29 = 26 + 20 - 17
  sorry

end students_play_long_tennis_l267_267015


namespace union_intersection_l267_267765

-- Define the sets M, N, and P
def M := ({1} : Set Nat)
def N := ({1, 2} : Set Nat)
def P := ({1, 2, 3} : Set Nat)

-- Prove that (M ∪ N) ∩ P = {1, 2}
theorem union_intersection : (M ∪ N) ∩ P = ({1, 2} : Set Nat) := 
by 
  sorry

end union_intersection_l267_267765


namespace smallest_possible_perimeter_l267_267413

theorem smallest_possible_perimeter (a : ℕ) (h : a > 2) (h_triangle : a < a + (a + 1) ∧ a + (a + 2) > (a + 1) ∧ (a + 1) + (a + 2) > a) :
  3 * a + 3 = 12 :=
by
  sorry

end smallest_possible_perimeter_l267_267413


namespace smallest_three_digit_multiple_of_17_l267_267937

/--
Prove that the smallest three-digit multiple of 17 is 102.
-/
theorem smallest_three_digit_multiple_of_17 : ∃ k : ℤ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 := 
by
  sorry

end smallest_three_digit_multiple_of_17_l267_267937


namespace puppies_per_cage_l267_267547

-- Conditions
variables (total_puppies sold_puppies cages initial_puppies per_cage : ℕ)
variables (h_total : total_puppies = 13)
variables (h_sold : sold_puppies = 7)
variables (h_cages : cages = 3)
variables (h_equal_cages : total_puppies - sold_puppies = cages * per_cage)

-- Question
theorem puppies_per_cage :
  per_cage = 2 :=
by {
  sorry
}

end puppies_per_cage_l267_267547


namespace total_balloons_l267_267554

-- Define the number of balloons Alyssa, Sandy, and Sally have.
def alyssa_balloons : ℕ := 37
def sandy_balloons : ℕ := 28
def sally_balloons : ℕ := 39

-- Theorem stating that the total number of balloons is 104.
theorem total_balloons : alyssa_balloons + sandy_balloons + sally_balloons = 104 :=
by
  -- Proof is omitted for the purpose of this task.
  sorry

end total_balloons_l267_267554


namespace find_f_neg_two_l267_267131

def is_even_function (f : ℝ → ℝ) (h : ℝ → ℝ) := ∀ x, h (-x) = h x

theorem find_f_neg_two (f : ℝ → ℝ) (h : ℝ → ℝ) (hx : ∀ x, h x = f (2*x) + x)
  (h_even : is_even_function f h) 
  (h_f_two : f 2 = 1) : 
  f (-2) = 3 :=
  by
    sorry

end find_f_neg_two_l267_267131


namespace smallest_three_digit_multiple_of_17_l267_267863

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m := by
  use 102
  sorry

end smallest_three_digit_multiple_of_17_l267_267863


namespace sufficient_not_necessary_l267_267352

def M : Set ℤ := {1, 2}
def N (a : ℤ) : Set ℤ := {a^2}

theorem sufficient_not_necessary (a : ℤ) :
  (a = 1 → N a ⊆ M) ∧ (N a ⊆ M → a = 1) = false :=
by 
  sorry

end sufficient_not_necessary_l267_267352


namespace smallest_perimeter_consecutive_even_triangle_l267_267216

theorem smallest_perimeter_consecutive_even_triangle :
  ∃ (x : ℕ), x % 2 = 0 ∧ 
             x + 2 > 2 ∧ 
             x + 4 > 2 ∧ 
             x > 2 ∧ 
             (let sides := [x, x + 2, x + 4] in 
                (sides.sum) = 18) :=
by
  sorry

end smallest_perimeter_consecutive_even_triangle_l267_267216


namespace caleb_ice_cream_vs_frozen_yoghurt_l267_267562

theorem caleb_ice_cream_vs_frozen_yoghurt :
  let cost_chocolate_ice_cream := 6 * 5
  let discount_chocolate := 0.10 * cost_chocolate_ice_cream
  let total_chocolate_ice_cream := cost_chocolate_ice_cream - discount_chocolate

  let cost_vanilla_ice_cream := 4 * 4
  let discount_vanilla := 0.07 * cost_vanilla_ice_cream
  let total_vanilla_ice_cream := cost_vanilla_ice_cream - discount_vanilla

  let total_ice_cream := total_chocolate_ice_cream + total_vanilla_ice_cream

  let cost_strawberry_yoghurt := 3 * 3
  let tax_strawberry := 0.05 * cost_strawberry_yoghurt
  let total_strawberry_yoghurt := cost_strawberry_yoghurt + tax_strawberry

  let cost_mango_yoghurt := 2 * 2
  let tax_mango := 0.03 * cost_mango_yoghurt
  let total_mango_yoghurt := cost_mango_yoghurt + tax_mango

  let total_yoghurt := total_strawberry_yoghurt + total_mango_yoghurt

  (total_ice_cream - total_yoghurt = 28.31) := by
  sorry

end caleb_ice_cream_vs_frozen_yoghurt_l267_267562


namespace christian_sue_need_more_money_l267_267255

-- Definition of initial amounts
def christian_initial := 5
def sue_initial := 7

-- Definition of earnings from activities
def christian_per_yard := 5
def christian_yards := 4
def sue_per_dog := 2
def sue_dogs := 6

-- Definition of perfume cost
def perfume_cost := 50

-- Theorem statement for the math problem
theorem christian_sue_need_more_money :
  let christian_total := christian_initial + (christian_per_yard * christian_yards)
  let sue_total := sue_initial + (sue_per_dog * sue_dogs)
  let total_money := christian_total + sue_total
  total_money < perfume_cost → perfume_cost - total_money = 6 :=
by 
  intros
  let christian_total := christian_initial + (christian_per_yard * christian_yards)
  let sue_total := sue_initial + (sue_per_dog * sue_dogs)
  let total_money := christian_total + sue_total
  sorry

end christian_sue_need_more_money_l267_267255


namespace problem_proof_l267_267762

theorem problem_proof (a b c x y z : ℝ) (h₁ : 17 * x + b * y + c * z = 0) (h₂ : a * x + 29 * y + c * z = 0)
                      (h₃ : a * x + b * y + 53 * z = 0) (ha : a ≠ 17) (hx : x ≠ 0) :
                      (a / (a - 17)) + (b / (b - 29)) + (c / (c - 53)) = 1 :=
by
  -- proof goes here
  sorry

end problem_proof_l267_267762


namespace sum_digits_500_l267_267564

noncomputable def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_digits_500 (k : ℕ) (h : k = 55) :
  sum_digits (63 * 10^k - 64) = 500 :=
by
  sorry

end sum_digits_500_l267_267564


namespace solve_3x_5y_eq_7_l267_267364

theorem solve_3x_5y_eq_7 :
  ∃ (x y k : ℤ), (3 * x + 5 * y = 7) ∧ (x = 4 + 5 * k) ∧ (y = -1 - 3 * k) :=
by 
  sorry

end solve_3x_5y_eq_7_l267_267364


namespace parallel_vectors_imply_x_value_l267_267490

theorem parallel_vectors_imply_x_value (x : ℝ) : 
    let a := (1, 2)
    let b := (-1, x)
    (1 / -1:ℝ) = (2 / x) → x = -2 := 
by
  intro h
  sorry

end parallel_vectors_imply_x_value_l267_267490


namespace smallest_three_digit_multiple_of_17_l267_267948

theorem smallest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m :=
by
  sorry

end smallest_three_digit_multiple_of_17_l267_267948


namespace sum_of_possible_values_l267_267009

theorem sum_of_possible_values
  (x : ℝ)
  (h : (x + 3) * (x - 4) = 22) :
  ∃ s : ℝ, s = 1 :=
sorry

end sum_of_possible_values_l267_267009


namespace fixed_chord_property_l267_267053

theorem fixed_chord_property (d : ℝ) (h₁ : d = 3 / 2) :
  ∀ (x1 x2 m : ℝ) (h₀ : x1 + x2 = m) (h₂ : x1 * x2 = 1 - d),
    ((1 / ((x1 ^ 2) + (m * x1) ^ 2)) + (1 / ((x2 ^ 2) + (m * x2) ^ 2))) = 4 / 9 :=
by
  sorry

end fixed_chord_property_l267_267053


namespace expected_team_a_score_l267_267630

open ProbabilityTheory

noncomputable def team_a_win_prob (match : ℕ) : ℝ :=
match match with
| 0 => 2/3  -- A1 vs B1
| 1 => 2/5  -- A2 vs B2
| _ => 2/5  -- A3 vs B3
end

noncomputable def team_a_expected_score : ℝ :=
  let p_win_0 := 2 / 3
  let p_win_1 := 2 / 5
  let p_win_2 := 2 / 5
  3 * p_win_0 * p_win_1 * p_win_2 +
  2 * (p_win_0 * p_win_1 * (3 / 5) + (1 / 3) * p_win_1 * p_win_2 + p_win_0 * (3 / 5) * p_win_2) +
  1 * (p_win_0 * (3 / 5) * (3 / 5) + (1 / 3) * p_win_1 * (3 / 5) + (1 / 3) * (3 / 5) * p_win_2) +
  0 * ((1 / 3) * (3 / 5) * (3 / 5))

theorem expected_team_a_score : team_a_expected_score = 22 / 15 :=
by
  sorry

end expected_team_a_score_l267_267630


namespace agnes_weekly_hours_l267_267612

-- Given conditions
def mila_hourly_rate : ℝ := 10
def agnes_hourly_rate : ℝ := 15
def mila_hours_per_month : ℝ := 48

-- Derived condition that Mila's earnings in a month equal Agnes's in a month
def mila_monthly_earnings : ℝ := mila_hourly_rate * mila_hours_per_month

-- Prove that Agnes must work 8 hours each week to match Mila's monthly earnings
theorem agnes_weekly_hours (A : ℝ) : 
  agnes_hourly_rate * 4 * A = mila_monthly_earnings → A = 8 := 
by
  intro h
  -- sorry here is a placeholder for the proof
  sorry

end agnes_weekly_hours_l267_267612


namespace mean_of_samantha_scores_l267_267360

noncomputable def arithmetic_mean (l : List ℝ) : ℝ := l.sum / l.length

theorem mean_of_samantha_scores :
  arithmetic_mean [93, 87, 90, 96, 88, 94] = 91.333 :=
by
  sorry

end mean_of_samantha_scores_l267_267360


namespace sum_of_reciprocals_perpendicular_lines_equilateral_triangle_perimeter_polynomial_divisibility_l267_267196

-- Problem 1
theorem sum_of_reciprocals (x y : ℝ) (hx : x + y = 50) (hxy : x * y = 25) : 
  (1 / x) + (1 / y) = 2 := 
by
  sorry

-- Problem 2
theorem perpendicular_lines (a b : ℝ) (ha : a = 2) 
  (eq1 : ∀ x y : ℝ, ax + 2y + 1 = 0) (eq2 : ∀ x y : ℝ, 3x + by + 5 = 0) : 
  b = -3 := 
by
  sorry

-- Problem 3
theorem equilateral_triangle_perimeter (A : ℝ) (hA : A = 100 * real.sqrt 3) : 
  ∃ (p : ℝ), p = 60 :=
by
  sorry

-- Problem 4
theorem polynomial_divisibility (p q : ℝ) (hp : p = 60) 
  (hq : ∀ x : ℝ, (x + 2) ∣ (x^3 - 2*x^2 + p*x + q)) : 
  q = 136 := 
by
  sorry

end sum_of_reciprocals_perpendicular_lines_equilateral_triangle_perimeter_polynomial_divisibility_l267_267196


namespace half_angle_in_first_or_third_quadrant_l267_267714

noncomputable 
def angle_in_first_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi < α ∧ α < (2 * k + 1) * Real.pi / 2

noncomputable 
def angle_in_first_or_third_quadrant (β : ℝ) : Prop :=
  ∃ k : ℤ, k * Real.pi < β ∧ β < (k + 1/4) * Real.pi ∨
  ∃ i : ℤ, (2 * i + 1) * Real.pi < β ∧ β < (2 * i + 5/4) * Real.pi 

theorem half_angle_in_first_or_third_quadrant (α : ℝ) (h : angle_in_first_quadrant α) :
  angle_in_first_or_third_quadrant (α / 2) :=
  sorry

end half_angle_in_first_or_third_quadrant_l267_267714


namespace moles_of_HNO3_l267_267123

theorem moles_of_HNO3 (HNO3 NaHCO3 NaNO3 : ℝ)
  (h1 : NaHCO3 = 1) (h2 : NaNO3 = 1) :
  HNO3 = 1 :=
by sorry

end moles_of_HNO3_l267_267123


namespace find_x_l267_267738

theorem find_x (x : ℝ) (h : (40 / 100) * x = (25 / 100) * 80) : x = 50 :=
by
  sorry

end find_x_l267_267738


namespace set_intersection_l267_267726

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

noncomputable def complement_U_A := U \ A
noncomputable def intersection := B ∩ complement_U_A

theorem set_intersection :
  intersection = ({3, 4} : Set ℕ) := by
  sorry

end set_intersection_l267_267726


namespace smallest_three_digit_multiple_of_17_l267_267846

theorem smallest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, m >= 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  let n := 102,
  use n,
  split,
  { norm_num, }, -- n >= 100 is obviously true because n = 102
  split,
  { norm_num, }, -- n < 1000 is obviously true because n = 102
  split,
  { exact dvd_refl 17, }, -- 17 divides 102
  { intros m hm,
    cases hm with h100 hm,
    cases hm with h1000 hdiv,
    exact le_trans (nat.le.intro _ (symm hdiv)) (le_of_not_gt (λ h, by norm_num at h)), },
end

end smallest_three_digit_multiple_of_17_l267_267846


namespace total_words_in_week_l267_267994

def typing_minutes_MWF : ℤ := 260
def typing_minutes_TTh : ℤ := 150
def typing_minutes_Sat : ℤ := 240
def typing_speed_MWF : ℤ := 50
def typing_speed_TTh : ℤ := 40
def typing_speed_Sat : ℤ := 60

def words_per_day_MWF : ℤ := typing_minutes_MWF * typing_speed_MWF
def words_per_day_TTh : ℤ := typing_minutes_TTh * typing_speed_TTh
def words_Sat : ℤ := typing_minutes_Sat * typing_speed_Sat

def total_words_week : ℤ :=
  (words_per_day_MWF * 3) + (words_per_day_TTh * 2) + words_Sat + 0

theorem total_words_in_week :
  total_words_week = 65400 :=
by
  sorry

end total_words_in_week_l267_267994


namespace original_bet_l267_267234

-- Define conditions and question
def payout_formula (B P : ℝ) : Prop :=
  P = (3 / 2) * B

def received_payment := 60

-- Define the Lean theorem statement
theorem original_bet (B : ℝ) (h : payout_formula B received_payment) : B = 40 :=
by
  sorry

end original_bet_l267_267234


namespace possible_integer_roots_l267_267693

def polynomial (x : ℤ) : ℤ := x^3 + 2 * x^2 - 3 * x - 17

theorem possible_integer_roots :
  ∃ (roots : List ℤ), roots = [1, -1, 17, -17] ∧ ∀ r ∈ roots, polynomial r = 0 := 
sorry

end possible_integer_roots_l267_267693


namespace max_togs_possible_l267_267993

def tag_cost : ℕ := 3
def tig_cost : ℕ := 4
def tog_cost : ℕ := 8
def total_budget : ℕ := 100
def min_tags : ℕ := 1
def min_tigs : ℕ := 1
def min_togs : ℕ := 1

theorem max_togs_possible : 
  ∃ (tags tigs togs : ℕ), tags ≥ min_tags ∧ tigs ≥ min_tigs ∧ togs ≥ min_togs ∧ 
  tag_cost * tags + tig_cost * tigs + tog_cost * togs = total_budget ∧ togs = 11 :=
sorry

end max_togs_possible_l267_267993


namespace sum_of_coefficients_of_poly_l267_267253

-- Define the polynomial
def poly (x y : ℕ) := (2 * x + 3 * y) ^ 12

-- Define the sum of coefficients
def sum_of_coefficients := poly 1 1

-- The theorem stating the result
theorem sum_of_coefficients_of_poly : sum_of_coefficients = 244140625 :=
by
  -- Proof is skipped
  sorry

end sum_of_coefficients_of_poly_l267_267253


namespace percentage_difference_is_20_l267_267170

/-
Given:
Height of sunflowers from Packet A = 192 inches
Height of sunflowers from Packet B = 160 inches

Show:
Percentage difference in height between Packet A and Packet B is 20%.
-/

-- Definitions of heights
def height_packet_A : ℤ := 192
def height_packet_B : ℤ := 160

-- Definition of percentage difference formula
def percentage_difference (hA hB : ℤ) : ℤ := ((hA - hB) * 100) / hB

-- Theorem statement
theorem percentage_difference_is_20 :
  percentage_difference height_packet_A height_packet_B = 20 :=
sorry

end percentage_difference_is_20_l267_267170


namespace inequality_l267_267452

theorem inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (a / (b.sqrt) + b / (a.sqrt)) ≥ (a.sqrt + b.sqrt) :=
by
  sorry

end inequality_l267_267452


namespace smallest_three_digit_multiple_of_17_l267_267936

/--
Prove that the smallest three-digit multiple of 17 is 102.
-/
theorem smallest_three_digit_multiple_of_17 : ∃ k : ℤ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 := 
by
  sorry

end smallest_three_digit_multiple_of_17_l267_267936


namespace smallest_three_digit_multiple_of_17_l267_267952

theorem smallest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m :=
by
  sorry

end smallest_three_digit_multiple_of_17_l267_267952


namespace prove_minimality_of_smallest_three_digit_multiple_of_17_l267_267906

def smallest_three_digit_multiple_of_17: ℕ :=
  102

theorem prove_minimality_of_smallest_three_digit_multiple_of_17 (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k ∧ 100 ≤ 17 * k ∧ 17 * k < 1000) 
  → n = 102 :=
    by
      sorry

end prove_minimality_of_smallest_three_digit_multiple_of_17_l267_267906


namespace smallest_repeating_block_of_3_over_11_l267_267471

def smallest_repeating_block_length (x y : ℕ) : ℕ :=
  if y ≠ 0 then (10 * x % y).natAbs else 0

theorem smallest_repeating_block_of_3_over_11 :
  smallest_repeating_block_length 3 11 = 2 :=
by sorry

end smallest_repeating_block_of_3_over_11_l267_267471


namespace fred_weekend_earnings_l267_267602

noncomputable def fred_initial_dollars : ℕ := 19
noncomputable def fred_final_dollars : ℕ := 40

theorem fred_weekend_earnings :
  fred_final_dollars - fred_initial_dollars = 21 :=
by
  sorry

end fred_weekend_earnings_l267_267602


namespace smallest_three_digit_multiple_of_17_l267_267828

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := 
begin
  use 102,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    sorry }
end

end smallest_three_digit_multiple_of_17_l267_267828


namespace angle_W_in_quadrilateral_l267_267184

theorem angle_W_in_quadrilateral 
  (W X Y Z : ℝ) 
  (h₀ : W + X + Y + Z = 360) 
  (h₁ : W = 3 * X) 
  (h₂ : W = 4 * Y) 
  (h₃ : W = 6 * Z) : 
  W = 206 :=
by
  sorry

end angle_W_in_quadrilateral_l267_267184


namespace smallest_three_digit_multiple_of_17_l267_267889

theorem smallest_three_digit_multiple_of_17 : 
    ∃ (n : ℤ), n ≥ 100 ∧ n < 1000 ∧ (∃ (k : ℤ), n = 17 * k) ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l267_267889


namespace product_scaled_areas_l267_267245

variable (a b c k V : ℝ)

def volume (a b c : ℝ) : ℝ := a * b * c

theorem product_scaled_areas (a b c k : ℝ) (V : ℝ) (hV : V = volume a b c) :
  (k * a * b) * (k * b * c) * (k * c * a) = k^3 * (V^2) := 
by
  -- Proof steps would go here, but we use sorry to skip the proof
  sorry

end product_scaled_areas_l267_267245


namespace inequality_proof_l267_267759

theorem inequality_proof (n : ℕ) (a : Fin n → ℝ) (h1 : 0 < n) (h2 : (Finset.univ.sum a) ≥ 0) :
  (Finset.univ.sum (λ i => Real.sqrt (a i ^ 2 + 1))) ≥
  Real.sqrt (2 * n * (Finset.univ.sum a)) :=
by
  sorry

end inequality_proof_l267_267759


namespace smallest_three_digit_multiple_of_17_l267_267963

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ ∃ (k : ℕ), n = 17 * k ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l267_267963


namespace number_of_hens_l267_267417

-- Let H be the number of hens and C be the number of cows
def hens_and_cows (H C : Nat) : Prop :=
  H + C = 50 ∧ 2 * H + 4 * C = 144

theorem number_of_hens : ∃ H C : Nat, hens_and_cows H C ∧ H = 28 :=
by
  -- The proof is omitted
  sorry

end number_of_hens_l267_267417


namespace total_books_proof_l267_267984

def initial_books : ℝ := 41.0
def added_books_first : ℝ := 33.0
def added_books_next : ℝ := 2.0

theorem total_books_proof : initial_books + added_books_first + added_books_next = 76.0 :=
by
  sorry

end total_books_proof_l267_267984


namespace average_homework_time_decrease_l267_267404

theorem average_homework_time_decrease
  (initial_time final_time : ℝ)
  (rate_of_decrease : ℝ)
  (h1 : initial_time = 100)
  (h2 : final_time = 70) :
  initial_time * (1 - rate_of_decrease)^2 = final_time :=
by
  rw [h1, h2]
  sorry

end average_homework_time_decrease_l267_267404


namespace original_bet_is_40_l267_267236

-- Definition relating payout ratio and payout to original bet
def calculate_original_bet (payout_ratio payout : ℚ) : ℚ :=
  payout / payout_ratio

-- Given conditions
def payout_ratio : ℚ := 3 / 2
def received_payout : ℚ := 60

-- The proof goal
theorem original_bet_is_40 : calculate_original_bet payout_ratio received_payout = 40 :=
by
  sorry

end original_bet_is_40_l267_267236


namespace jellybean_mass_l267_267989

noncomputable def cost_per_gram : ℚ := 7.50 / 250
noncomputable def mass_for_180_cents : ℚ := 1.80 / cost_per_gram

theorem jellybean_mass :
  mass_for_180_cents = 60 := 
  sorry

end jellybean_mass_l267_267989


namespace problem_conditions_l267_267277

open Real

variable {m n : ℝ}

theorem problem_conditions (h1 : 0 < m) (h2 : 0 < n) (h3 : m + n = 2 * m * n) :
  (min (m + n) = 2) ∧ (min (sqrt (m * n)) = 1) ∧
  (min ((n^2) / m + (m^2) / n) = 2) ∧ 
  (max ((sqrt m + sqrt n) / sqrt (m * n)) = 2) :=
by sorry

end problem_conditions_l267_267277


namespace complete_half_job_in_six_days_l267_267085

theorem complete_half_job_in_six_days (x : ℕ) (h1 : 2 * x = x + 6) : x = 6 :=
  by
    sorry

end complete_half_job_in_six_days_l267_267085


namespace sum_of_solutions_l267_267064

theorem sum_of_solutions (x : ℝ) : 
  (∃ x : ℝ, x^2 - 7 * x + 2 = 16) → (complex.sum (λ x : ℝ, x^2 - 7 * x - 14)) = 7 := sorry

end sum_of_solutions_l267_267064


namespace common_difference_ne_3_l267_267706

theorem common_difference_ne_3 
  (d : ℕ) (hd_pos : d > 0) 
  (exists_n : ∃ n : ℕ, 81 = 1 + (n - 1) * d) : 
  d ≠ 3 :=
by sorry

end common_difference_ne_3_l267_267706


namespace standard_eq_of_tangent_circle_l267_267456

-- Define the center and tangent condition of the circle
def center : ℝ × ℝ := (1, 2)
def tangent_to_x_axis (r : ℝ) : Prop := r = center.snd

-- The standard equation of the circle given the center and radius
def standard_eq_circle (h k r : ℝ) : Prop := ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement to prove the standard equation of the circle
theorem standard_eq_of_tangent_circle : 
  ∃ r, tangent_to_x_axis r ∧ standard_eq_circle 1 2 r := 
by 
  sorry

end standard_eq_of_tangent_circle_l267_267456


namespace kaylee_biscuit_sales_l267_267323

theorem kaylee_biscuit_sales:
    ∀ (total_boxes required_boxes : ℕ) (lemon_boxes chocolate_boxes oatmeal_boxes : ℕ),
        required_boxes = 33 ∧ 
        lemon_boxes = 12 ∧ 
        chocolate_boxes = 5 ∧ 
        oatmeal_boxes = 4 →
        total_boxes = lemon_boxes + chocolate_boxes + oatmeal_boxes →
        (required_boxes - total_boxes = 12) :=
begin
  sorry
end

end kaylee_biscuit_sales_l267_267323


namespace Brady_average_hours_l267_267101

-- Definitions based on conditions
def hours_per_day_April : ℕ := 6
def hours_per_day_June : ℕ := 5
def hours_per_day_September : ℕ := 8
def days_in_April : ℕ := 30
def days_in_June : ℕ := 30
def days_in_September : ℕ := 30

-- Definition to prove
def average_hours_per_month : ℕ := 190

-- Theorem statement
theorem Brady_average_hours :
  (hours_per_day_April * days_in_April + hours_per_day_June * days_in_June + hours_per_day_September * days_in_September) / 3 = average_hours_per_month :=
sorry

end Brady_average_hours_l267_267101


namespace mul_99_101_square_98_l267_267252

theorem mul_99_101 : 99 * 101 = 9999 := sorry

theorem square_98 : 98^2 = 9604 := sorry

end mul_99_101_square_98_l267_267252


namespace parallel_vectors_m_eq_neg3_l267_267001

-- Define vectors
def vec (x y : ℝ) : ℝ × ℝ := (x, y)
def OA : ℝ × ℝ := vec 3 (-4)
def OB : ℝ × ℝ := vec 6 (-3)
def OC (m : ℝ) : ℝ × ℝ := vec (2 * m) (m + 1)

-- Define the condition that AB is parallel to OC
def is_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

-- Calculate AB
def AB : ℝ × ℝ := (OB.1 - OA.1, OB.2 - OA.2)

-- The theorem to prove
theorem parallel_vectors_m_eq_neg3 (m : ℝ) :
  is_parallel AB (OC m) → m = -3 := by
  sorry

end parallel_vectors_m_eq_neg3_l267_267001


namespace determine_percentage_of_yellow_in_darker_green_paint_l267_267124

noncomputable def percentage_of_yellow_in_darker_green_paint : Real :=
  let volume_light_green := 5
  let volume_darker_green := 1.66666666667
  let percentage_light_green := 0.20
  let final_percentage := 0.25
  let total_volume := volume_light_green + volume_darker_green
  let total_yellow_required := final_percentage * total_volume
  let yellow_in_light_green := percentage_light_green * volume_light_green
  (total_yellow_required - yellow_in_light_green) / volume_darker_green

theorem determine_percentage_of_yellow_in_darker_green_paint :
  percentage_of_yellow_in_darker_green_paint = 0.4 := by
  sorry

end determine_percentage_of_yellow_in_darker_green_paint_l267_267124


namespace Faye_crayons_l267_267260

theorem Faye_crayons (rows crayons_per_row : ℕ) (h_rows : rows = 7) (h_crayons_per_row : crayons_per_row = 30) : rows * crayons_per_row = 210 :=
by
  sorry

end Faye_crayons_l267_267260


namespace smallest_perimeter_even_integer_triangl_l267_267210

theorem smallest_perimeter_even_integer_triangl (n : ℕ) (h : n > 2) :
  let a := 2 * n - 2
  let b := 2 * n
  let c := 2 * n + 2
  2 * n - 2 + 2 * n > 2 * n + 2 ∧
  2 * n - 2 + 2 * n + 2 > 2 * n ∧
  2 * n + 2 * n + 2 > 2 * n - 2 ∧ 
  2 * 3 - 2 + 2 * 3 + 2 * 3 + 2 = 18 :=
by
  { sorry }

end smallest_perimeter_even_integer_triangl_l267_267210


namespace smallest_three_digit_multiple_of_17_l267_267935

/--
Prove that the smallest three-digit multiple of 17 is 102.
-/
theorem smallest_three_digit_multiple_of_17 : ∃ k : ℤ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 := 
by
  sorry

end smallest_three_digit_multiple_of_17_l267_267935


namespace average_chemistry_mathematics_l267_267667

-- Define the conditions 
variable {P C M : ℝ} -- Marks in Physics, Chemistry, and Mathematics

-- The given condition in the problem
theorem average_chemistry_mathematics (h : P + C + M = P + 130) : (C + M) / 2 = 65 := 
by
  -- This will be the main proof block (we use 'sorry' to omit the actual proof)
  sorry

end average_chemistry_mathematics_l267_267667


namespace sin_cos_quad_ineq_l267_267177

open Real

theorem sin_cos_quad_ineq (x : ℝ) : 
  2 * (sin x) ^ 4 + 3 * (sin x) ^ 2 * (cos x) ^ 2 + 5 * (cos x) ^ 4 ≤ 5 :=
by
  sorry

end sin_cos_quad_ineq_l267_267177


namespace smallest_three_digit_multiple_of_17_l267_267898

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m) :=
  sorry

end smallest_three_digit_multiple_of_17_l267_267898


namespace lines_perpendicular_l267_267269

structure Vec3 :=
(x : ℝ) 
(y : ℝ) 
(z : ℝ)

def line1_dir (x : ℝ) : Vec3 := ⟨x, -1, 2⟩
def line2_dir : Vec3 := ⟨2, 1, 4⟩

def dot_product (v1 v2 : Vec3) : ℝ :=
v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

theorem lines_perpendicular (x : ℝ) :
  dot_product (line1_dir x) line2_dir = 0 ↔ x = -7 / 2 :=
by sorry

end lines_perpendicular_l267_267269


namespace trig_identity_and_perimeter_l267_267336

theorem trig_identity_and_perimeter
  (a b c : ℝ) (A B C : ℝ)
  (h1 : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A))
  (a_val : a = 5)
  (cos_A : Real.cos A = 25 / 31) :
  (2 * a^2 = b^2 + c^2) ∧ (a + b + c = 14) :=
by
  sorry

end trig_identity_and_perimeter_l267_267336


namespace find_b_if_even_function_l267_267813

variable (b c : ℝ)

def f (x : ℝ) : ℝ := x^2 + b * x + c

theorem find_b_if_even_function (h : ∀ x : ℝ, f (-x) = f (x)) : b = 0 := by
  sorry

end find_b_if_even_function_l267_267813


namespace union_of_M_and_N_l267_267609

def M : Set ℝ := {x | x^2 - 6 * x + 5 = 0}
def N : Set ℝ := {x | x^2 - 5 * x = 0}

theorem union_of_M_and_N : M ∪ N = {0, 1, 5} := by
  sorry

end union_of_M_and_N_l267_267609


namespace prove_minimality_of_smallest_three_digit_multiple_of_17_l267_267905

def smallest_three_digit_multiple_of_17: ℕ :=
  102

theorem prove_minimality_of_smallest_three_digit_multiple_of_17 (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k ∧ 100 ≤ 17 * k ∧ 17 * k < 1000) 
  → n = 102 :=
    by
      sorry

end prove_minimality_of_smallest_three_digit_multiple_of_17_l267_267905


namespace inequality_a3_minus_b3_l267_267276

theorem inequality_a3_minus_b3 (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a < b) : a^3 - b^3 < 0 :=
by sorry

end inequality_a3_minus_b3_l267_267276


namespace part_one_part_two_l267_267332

theorem part_one (A B C a b c : ℝ) (hA : A + B + C = π) 
  (h1 : sin(C) * sin(A - B) = sin(B) * sin(C - A)) : 
  2 * a^2 = b^2 + c^2 := by
  -- Placeholder for the detailed proof
  sorry

theorem part_two (a b c : ℝ) 
  (h1 : 2 * a^2 = b^2 + c^2) (h2 : a = 5) (h3 : cos(acos (5 / 31)) = 25 / 31) : 
  a + b + c = 14 := by
  -- Placeholder for the detailed proof
  sorry

end part_one_part_two_l267_267332


namespace sufficient_not_necessary_l267_267419

theorem sufficient_not_necessary (x : ℝ) :
  (x > 1 → x^2 - 2*x + 1 > 0) ∧ (¬(x^2 - 2*x + 1 > 0 → x > 1)) := by
  sorry

end sufficient_not_necessary_l267_267419


namespace smallest_m_for_integral_solutions_l267_267209

theorem smallest_m_for_integral_solutions :
  ∃ m : ℕ, m > 0 ∧ (∃ p q : ℤ, 10 * p * q = 660 ∧ p + q = m/10) ∧ m = 170 :=
by
  sorry

end smallest_m_for_integral_solutions_l267_267209


namespace smallest_three_digit_multiple_of_17_l267_267954

theorem smallest_three_digit_multiple_of_17 : ∃ (k : ℕ), 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 :=
by
  use 6
  split
  · exact of_as_true trivial -- 100 <= 17 * k
  split
  · exact of_as_true trivial -- 17 * k <= 999
  · exact of_as_true trivial -- 17 * k = 102
  sorry

end smallest_three_digit_multiple_of_17_l267_267954


namespace smallest_three_digit_multiple_of_17_l267_267842

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m := by
  sorry

end smallest_three_digit_multiple_of_17_l267_267842


namespace kim_easy_round_correct_answers_l267_267748

variable (E : ℕ)

theorem kim_easy_round_correct_answers 
    (h1 : 2 * E + 3 * 2 + 5 * 4 = 38) : 
    E = 6 := 
sorry

end kim_easy_round_correct_answers_l267_267748


namespace smallest_three_digit_multiple_of_17_l267_267950

theorem smallest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m :=
by
  sorry

end smallest_three_digit_multiple_of_17_l267_267950


namespace necessary_and_sufficient_condition_perpendicular_lines_l267_267078

def are_perpendicular (a : ℝ) : Prop :=
  ∀ x y : ℝ, (x + y = 0) → (x - a * y = 0) → x = 0 ∧ y = 0

theorem necessary_and_sufficient_condition_perpendicular_lines :
  ∀ (a : ℝ), are_perpendicular a → a = 1 :=
sorry

end necessary_and_sufficient_condition_perpendicular_lines_l267_267078


namespace largest_consecutive_odd_integers_sum_255_l267_267195

theorem largest_consecutive_odd_integers_sum_255 : 
  ∃ (n : ℤ), (n + (n + 2) + (n + 4) + (n + 6) + (n + 8) = 255) ∧ (n + 8 = 55) :=
by
  sorry

end largest_consecutive_odd_integers_sum_255_l267_267195


namespace transformed_parabola_equation_l267_267056

-- Conditions
def original_parabola (x : ℝ) : ℝ := 3 * x^2
def translate_downwards (y : ℝ) : ℝ := y - 3

-- Translations
def translate_to_right (x : ℝ) : ℝ := x - 2
def transformed_parabola (x : ℝ) : ℝ := 3 * (x - 2)^2 - 3

-- Assertion
theorem transformed_parabola_equation :
  (∀ x : ℝ, translate_downwards (original_parabola x) = 3 * (translate_to_right x)^2 - 3) := by
  sorry

end transformed_parabola_equation_l267_267056


namespace part1_part2_l267_267341

-- Definitions and conditions
variables (A B C : ℝ) (a b c : ℝ)
-- Given conditions
variable (h1 : sin C * sin (A - B) = sin B * sin (C - A))

-- Proof for part (1)
theorem part1 : 2 * a^2 = b^2 + c^2 :=
  sorry

-- Given further conditions for part (2)
variables (a_eq : a = 5) (cosA_eq : cos A = 25 / 31)

-- Proof for part (2)
theorem part2 : a + b + c = 14 :=
  sorry

end part1_part2_l267_267341


namespace BC_work_time_l267_267673

-- Definitions
def rateA : ℚ := 1 / 4 -- A's rate of work
def rateB : ℚ := 1 / 4 -- B's rate of work
def rateAC : ℚ := 1 / 3 -- A and C's combined rate of work

-- To prove
theorem BC_work_time : 1 / (rateB + (rateAC - rateA)) = 3 := by
  sorry

end BC_work_time_l267_267673


namespace six_digit_start_5_not_possible_l267_267226

theorem six_digit_start_5_not_possible :
  ∀ n : ℕ, (n ≥ 500000 ∧ n < 600000) → (¬ ∃ m : ℕ, (n * 10^6 + m) ^ 2 < 10^12 ∧ (n * 10^6 + m) ^ 2 ≥ 5 * 10^11 ∧ (n * 10^6 + m) ^ 2 < 6 * 10^11) :=
sorry

end six_digit_start_5_not_possible_l267_267226


namespace smallest_perimeter_even_integer_triangl_l267_267211

theorem smallest_perimeter_even_integer_triangl (n : ℕ) (h : n > 2) :
  let a := 2 * n - 2
  let b := 2 * n
  let c := 2 * n + 2
  2 * n - 2 + 2 * n > 2 * n + 2 ∧
  2 * n - 2 + 2 * n + 2 > 2 * n ∧
  2 * n + 2 * n + 2 > 2 * n - 2 ∧ 
  2 * 3 - 2 + 2 * 3 + 2 * 3 + 2 = 18 :=
by
  { sorry }

end smallest_perimeter_even_integer_triangl_l267_267211


namespace trader_profit_percentage_l267_267978

theorem trader_profit_percentage (P : ℝ) (hP : 0 < P) :
  let bought_price := 0.90 * P
  let sold_price := 1.80 * bought_price
  let profit := sold_price - P
  let profit_percentage := (profit / P) * 100
  profit_percentage = 62 := 
by
  let bought_price := 0.90 * P
  let sold_price := 1.80 * bought_price
  let profit := sold_price - P
  let profit_percentage := (profit / P) * 100
  sorry

end trader_profit_percentage_l267_267978


namespace smallest_three_digit_multiple_of_17_l267_267938

/--
Prove that the smallest three-digit multiple of 17 is 102.
-/
theorem smallest_three_digit_multiple_of_17 : ∃ k : ℤ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 := 
by
  sorry

end smallest_three_digit_multiple_of_17_l267_267938


namespace CarmenBrushLengthInCentimeters_l267_267110

-- Given conditions
def CarlaBrushLengthInInches : ℝ := 12
def CarmenBrushPercentIncrease : ℝ := 0.5
def InchToCentimeterConversionFactor : ℝ := 2.5

-- Question: What is Carmen's brush length in centimeters?
-- Proof Goal: Prove that Carmen's brush length in centimeters is 45 cm.
theorem CarmenBrushLengthInCentimeters :
  let CarmenBrushLengthInInches := CarlaBrushLengthInInches * (1 + CarmenBrushPercentIncrease)
  CarmenBrushLengthInInches * InchToCentimeterConversionFactor = 45 := by
  -- sorry is used as a placeholder for the completed proof
  sorry

end CarmenBrushLengthInCentimeters_l267_267110


namespace pioneers_club_attendance_l267_267983

theorem pioneers_club_attendance :
  ∃ (A B : (Fin 11)), A ≠ B ∧
  (∃ (clubs_A clubs_B : Finset (Fin 5)), clubs_A = clubs_B) :=
by
  sorry

end pioneers_club_attendance_l267_267983


namespace box_filling_rate_l267_267508

theorem box_filling_rate (l w h t : ℝ) (hl : l = 7) (hw : w = 6) (hh : h = 2) (ht : t = 21) : 
  (l * w * h) / t = 4 := by
  sorry

end box_filling_rate_l267_267508


namespace smallest_multiple_of_seven_l267_267047

/-- The definition of the six-digit number formed by digits a, b, and c followed by "321". -/
def form_number (a b c : ℕ) : ℕ := 100000 * a + 10000 * b + 1000 * c + 321

/-- The condition that a, b, and c are distinct and greater than 3. -/
def valid_digits (a b c : ℕ) : Prop := a > 3 ∧ b > 3 ∧ c > 3 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem smallest_multiple_of_seven (a b c : ℕ)
  (h_valid : valid_digits a b c)
  (h_mult_seven : form_number a b c % 7 = 0) :
  form_number a b c = 468321 :=
sorry

end smallest_multiple_of_seven_l267_267047


namespace max_fm_n_l267_267460

noncomputable def ln (x : ℝ) : ℝ := Real.log x

def g (m n x : ℝ) : ℝ := (2 * m + 3) * x + n

def condition_f_g (m n : ℝ) : Prop := ∀ x > 0, ln x ≤ g m n x

def f (m : ℝ) : ℝ := 2 * m + 3

theorem max_fm_n (m n : ℝ) (h : condition_f_g m n) : (f m) * n ≤ 1 / Real.exp 2 := sorry

end max_fm_n_l267_267460


namespace students_taking_both_languages_l267_267370

theorem students_taking_both_languages (F S B : ℕ) (hF : F = 21) (hS : S = 21) (h30 : 30 = F - B + (S - B)) : B = 6 :=
by
  rw [hF, hS] at h30
  sorry

end students_taking_both_languages_l267_267370


namespace left_handed_women_percentage_l267_267149

noncomputable section

variables (x y : ℕ) (percentage : ℝ)

-- Conditions
def right_handed_ratio := 3
def left_handed_ratio := 1
def men_ratio := 3
def women_ratio := 2

def total_population_by_hand := right_handed_ratio * x + left_handed_ratio * x -- i.e., 4x
def total_population_by_gender := men_ratio * y + women_ratio * y -- i.e., 5y

-- Main Statement
theorem left_handed_women_percentage (h1 : total_population_by_hand = total_population_by_gender) :
    percentage = 25 :=
by
  sorry

end left_handed_women_percentage_l267_267149


namespace Bert_sandwiches_left_l267_267251

theorem Bert_sandwiches_left : (Bert:Type) → 
  (sandwiches_made : ℕ) → 
  sandwiches_made = 12 → 
  (sandwiches_eaten_day1 : ℕ) → 
  sandwiches_eaten_day1 = sandwiches_made / 2 → 
  (sandwiches_eaten_day2 : ℕ) → 
  sandwiches_eaten_day2 = sandwiches_eaten_day1 - 2 →
  (sandwiches_left : ℕ) → 
  sandwiches_left = sandwiches_made - (sandwiches_eaten_day1 + sandwiches_eaten_day2) → 
  sandwiches_left = 2 := 
  sorry

end Bert_sandwiches_left_l267_267251


namespace part_one_part_two_l267_267331

theorem part_one (A B C a b c : ℝ) (hA : A + B + C = π) 
  (h1 : sin(C) * sin(A - B) = sin(B) * sin(C - A)) : 
  2 * a^2 = b^2 + c^2 := by
  -- Placeholder for the detailed proof
  sorry

theorem part_two (a b c : ℝ) 
  (h1 : 2 * a^2 = b^2 + c^2) (h2 : a = 5) (h3 : cos(acos (5 / 31)) = 25 / 31) : 
  a + b + c = 14 := by
  -- Placeholder for the detailed proof
  sorry

end part_one_part_two_l267_267331


namespace chocolates_initial_l267_267031

variable (x : ℕ)
variable (h1 : 3 * x + 5 + 25 = 5 * x)
variable (h2 : x = 15)

theorem chocolates_initial (x : ℕ) (h1 : 3 * x + 5 + 25 = 5 * x) (h2 : x = 15) : 3 * 15 + 5 = 50 :=
by sorry

end chocolates_initial_l267_267031


namespace highest_page_number_l267_267777

/-- Given conditions: Pat has 19 instances of the digit '7' and an unlimited supply of all 
other digits. Prove that the highest page number Pat can number without exceeding 19 instances 
of the digit '7' is 99. -/
theorem highest_page_number (num_of_sevens : ℕ) (highest_page : ℕ) 
  (h1 : num_of_sevens = 19) : highest_page = 99 :=
sorry

end highest_page_number_l267_267777


namespace find_factor_l267_267090

-- Definitions based on the conditions
def n : ℤ := 155
def result : ℤ := 110
def constant : ℤ := 200

-- Statement to be proved
theorem find_factor (f : ℤ) (h : n * f - constant = result) : f = 2 := by
  sorry

end find_factor_l267_267090


namespace sufficient_but_not_necessary_condition_l267_267145

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x * (x - 1) < 0 → x < 1) ∧ ¬(x < 1 → x * (x - 1) < 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l267_267145


namespace tan_addition_sin_cos_expression_l267_267573

noncomputable def alpha : ℝ := sorry -- this is where alpha would be defined

axiom tan_alpha_eq_two : Real.tan alpha = 2

theorem tan_addition (alpha : ℝ) (h : Real.tan alpha = 2) : (Real.tan (alpha + Real.pi / 4) = -3) :=
by sorry

theorem sin_cos_expression (alpha : ℝ) (h : Real.tan alpha = 2) : 
  (Real.sin (2 * alpha) / (Real.sin (alpha) ^ 2 - Real.cos (2 * alpha) + 1) = 1 / 3) :=
by sorry

end tan_addition_sin_cos_expression_l267_267573


namespace prism_cutout_l267_267097

noncomputable def original_volume : ℕ := 15 * 5 * 4 -- Volume of the original prism
noncomputable def cutout_width : ℕ := 5

variables {x y : ℕ}

theorem prism_cutout:
  -- Given conditions
  (15 > 0) ∧ (5 > 0) ∧ (4 > 0) ∧ (x > 0) ∧ (y > 0) ∧ 
  -- The volume condition
  (original_volume - y * cutout_width * x = 120) →
  -- Prove that x + y = 15
  (x + y = 15) :=
sorry

end prism_cutout_l267_267097


namespace infinite_solutions_iff_c_is_5_over_2_l267_267114

theorem infinite_solutions_iff_c_is_5_over_2 (c : ℝ) :
  (∀ y : ℝ, 3 * (2 + 2 * c * y) = 15 * y + 6) ↔ c = 5 / 2 :=
by 
  sorry

end infinite_solutions_iff_c_is_5_over_2_l267_267114


namespace mrs_lee_grandsons_prob_l267_267172

open ProbabilityTheory

-- Define the probability that number of grandsons is not equal to number of granddaughters
def prob_mrs_lee_grandsons_ne_granddaughters : ℚ :=
  472305 / 531441

theorem mrs_lee_grandsons_prob (n : ℕ) (p : ℚ) (h_n : n = 12) (h_p : p = 2 / 3) :
  P(X ≠ 6) = prob_mrs_lee_grandsons_ne_granddaughters := by
  sorry

end mrs_lee_grandsons_prob_l267_267172


namespace smallest_three_digit_multiple_of_17_l267_267897

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m) :=
  sorry

end smallest_three_digit_multiple_of_17_l267_267897


namespace exactly_one_first_class_probability_at_least_one_second_class_probability_l267_267302

-- Definitions based on the problem statement:
def total_pens : ℕ := 6
def first_class_pens : ℕ := 4
def second_class_pens : ℕ := 2

def total_draws : ℕ := 2

-- Event for drawing exactly one first-class quality pen
def probability_one_first_class := ((first_class_pens.choose 1 * second_class_pens.choose 1) /
                                    (total_pens.choose total_draws) : ℚ)

-- Event for drawing at least one second-class quality pen
def probability_at_least_one_second_class := (1 - (first_class_pens.choose total_draws /
                                                   total_pens.choose total_draws) : ℚ)

-- Statements to prove the probabilities
theorem exactly_one_first_class_probability :
  probability_one_first_class = 8 / 15 :=
sorry

theorem at_least_one_second_class_probability :
  probability_at_least_one_second_class = 3 / 5 :=
sorry

end exactly_one_first_class_probability_at_least_one_second_class_probability_l267_267302


namespace maximum_value_of_w_l267_267710

variables (x y : ℝ)

def condition : Prop := x^2 + y^2 = 18 * x + 8 * y + 10

def w (x y : ℝ) := 4 * x + 3 * y

theorem maximum_value_of_w : ∃ x y, condition x y ∧ w x y = 74 :=
sorry

end maximum_value_of_w_l267_267710


namespace range_of_abs_function_l267_267819

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) + abs (x - 1)

theorem range_of_abs_function : Set.range f = Set.Ici 2 := by
  sorry

end range_of_abs_function_l267_267819


namespace maximum_value_ab_l267_267451

noncomputable def g (x : ℝ) : ℝ := 2 ^ x

theorem maximum_value_ab (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : g a * g b = 2) :
  ab ≤ (1 / 4) := sorry

end maximum_value_ab_l267_267451


namespace smallest_integer_b_l267_267826

theorem smallest_integer_b (b : ℕ) : 27 ^ b > 3 ^ 9 ↔ b = 4 := by
  sorry

end smallest_integer_b_l267_267826


namespace graph_of_3x2_minus_12y2_is_pair_of_straight_lines_l267_267060

theorem graph_of_3x2_minus_12y2_is_pair_of_straight_lines :
  ∀ (x y : ℝ), 3 * x^2 - 12 * y^2 = 0 ↔ (x = 2 * y ∨ x = -2 * y) :=
by
  sorry

end graph_of_3x2_minus_12y2_is_pair_of_straight_lines_l267_267060


namespace selection_of_students_l267_267555

theorem selection_of_students (s : Finset ℕ) (A B : ℕ) (hAB : A ∈ s ∧ B ∈ s) (h_size : s.card = 10) : 
  ∃ t ⊆ s, t.card = 4 ∧ (A ∈ t ∨ B ∈ t) ∧ (s.card * (s.card - 1)) / 2 + (s.card * (s.card - 1) * (s.card - 2)) / 6 * 2 = 140 := sorry

end selection_of_students_l267_267555


namespace range_of_a_l267_267702

theorem range_of_a {x y a : ℝ} (hx : 0 < x) (hy : 0 < y) (h : x + y + 6 = 4 * x * y) : a ≤ 10 / 3 :=
  sorry

end range_of_a_l267_267702


namespace cloaks_always_short_l267_267057

-- Define the problem parameters
variables (Knights Cloaks : Type)
variables [Fintype Knights] [Fintype Cloaks]
variables (h_knights : Fintype.card Knights = 20) (h_cloaks : Fintype.card Cloaks = 20)

-- Assume every knight initially found their cloak too short
variable (too_short : Knights -> Prop)

-- Height order for knights
variable (height_order : LinearOrder Knights)
-- Length order for cloaks
variable (length_order : LinearOrder Cloaks)

-- Sorting function
noncomputable def sorted_cloaks (kn : Knights) : Cloaks := sorry

-- State that after redistribution, every knight's cloak is still too short
theorem cloaks_always_short : 
  ∀ (kn : Knights), too_short kn :=
by sorry

end cloaks_always_short_l267_267057


namespace angle_A_measure_l267_267019

theorem angle_A_measure
  (A B C : Type)
  [triangle A B C]
  (angle_B : ℝ)
  (angle_B_measure : angle_B = 15)
  (angle_C : ℝ)
  (angle_C_measure : angle_C = 3 * angle_B) :
  A = 120 :=
by
  sorry

end angle_A_measure_l267_267019


namespace max_students_can_distribute_equally_l267_267815

-- Define the given numbers of pens and pencils
def pens : ℕ := 1001
def pencils : ℕ := 910

-- State the problem in Lean 4 as a theorem
theorem max_students_can_distribute_equally :
  Nat.gcd pens pencils = 91 :=
sorry

end max_students_can_distribute_equally_l267_267815


namespace find_range_of_m_l267_267420

-- Define properties of ellipses and hyperbolas
def isEllipseY (m : ℝ) : Prop := (8 - m > 2 * m - 1 ∧ 2 * m - 1 > 0)
def isHyperbola (m : ℝ) : Prop := (m + 1) * (m - 2) < 0

-- The range of 'm' such that (p ∨ q) is true and (p ∧ q) is false
def p_or_q_true_p_and_q_false (m : ℝ) : Prop := 
  (isEllipseY m ∨ isHyperbola m) ∧ ¬ (isEllipseY m ∧ isHyperbola m)

-- The range of the real number 'm'
def range_of_m (m : ℝ) : Prop := 
  (-1 < m ∧ m ≤ 1/2) ∨ (2 ≤ m ∧ m < 3)

-- Prove that the above conditions imply the correct range for m
theorem find_range_of_m (m : ℝ) : p_or_q_true_p_and_q_false m → range_of_m m :=
by
  sorry

end find_range_of_m_l267_267420


namespace smallest_three_digit_multiple_of_17_l267_267890

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m) :=
  sorry

end smallest_three_digit_multiple_of_17_l267_267890


namespace ellipse_sum_l267_267812

theorem ellipse_sum (h k a b : ℤ) (h_val : h = 3) (k_val : k = -5) (a_val : a = 7) (b_val : b = 4) : 
  h + k + a + b = 9 :=
by
  rw [h_val, k_val, a_val, b_val]
  norm_num

end ellipse_sum_l267_267812


namespace smallest_three_digit_multiple_of_17_l267_267870

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m := by
  use 102
  sorry

end smallest_three_digit_multiple_of_17_l267_267870


namespace math_proof_problem_l267_267793

def expr (m : ℝ) : ℝ := (1 - (2 / (m + 1))) / ((m ^ 2 - 2 * m + 1) / (m ^ 2 - m))

theorem math_proof_problem :
  expr (Real.tan (Real.pi / 3) - 1) = (3 - Real.sqrt 3) / 3 :=
  sorry

end math_proof_problem_l267_267793


namespace leopards_arrangement_l267_267497

theorem leopards_arrangement :
  let leopards := 10
  let shortest := 3
  let remaining := leopards - shortest
  (shortest! * remaining! = 30240) :=
by
  let leopards := 10
  let shortest := 3
  let remaining := leopards - shortest
  have factorials_eq: shortest! * remaining! = 30240 := sorry
  exact factorials_eq

end leopards_arrangement_l267_267497


namespace smallest_three_digit_multiple_of_17_l267_267892

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m) :=
  sorry

end smallest_three_digit_multiple_of_17_l267_267892


namespace problem1_problem2_l267_267362

-- Problem 1
theorem problem1 (x y : ℤ) (h1 : x = 2) (h2 : y = 2016) :
  (3*x + 2*y)*(3*x - 2*y) - (x + 2*y)*(5*x - 2*y) / (8*x) = -2015 :=
by
  sorry

-- Problem 2
theorem problem2 (x : ℤ) (h1 : x = 2) :
  ((x - 3) / (x^2 - 1)) * ((x^2 + 2*x + 1) / (x - 3)) - (1 / (x - 1) + 1) = 1 :=
by
  sorry

end problem1_problem2_l267_267362


namespace xyz_value_l267_267279

variable {x y z : ℝ}

theorem xyz_value (h1 : (x + y + z) * (x * y + x * z + y * z) = 18) 
                  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 6) : 
                  x * y * z = 4 := 
by
  sorry

end xyz_value_l267_267279


namespace part1_part2_l267_267343

variable {A B C a b c : ℝ}

-- Part (1): Prove that 2a^2 = b^2 + c^2 given the condition
theorem part1 (h : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A)) : 2 * a^2 = b^2 + c^2 := 
sorry

-- Part (2): Prove the perimeter of triangle ABC
theorem part2 (a : ℝ) (h_a : a = 5) (cosA : ℝ) (h_cosA : cosA = 25 / 31) : 5 + b + c = 14 := 
sorry

end part1_part2_l267_267343


namespace polygon_problem_l267_267308

theorem polygon_problem 
  (D : ℕ → ℕ) (m x : ℕ) 
  (H1 : ∀ n, D n = n * (n - 3) / 2)
  (H2 : D m = 3 * D (m - 3))
  (H3 : D (m + x) = 7 * D m) :
  m = 9 ∧ x = 12 ∧ (m + x) - m = 12 :=
by {
  -- the proof would go here, skipped as per the instructions.
  sorry
}

end polygon_problem_l267_267308


namespace proof_problem_l267_267663

open Real

theorem proof_problem :
  (∀ x : ℕ, x * x = 16 → sqrt (x * x) = 4) →
  (∀ x : ℕ, x * x = 16 → sqrt (x * x) = 16/4) →
  (∀ x : ℤ, abs x = 4 → abs (-4) = 4) →
  (∀ x : ℤ, x^2 = 16 → (-4)^2 = 16) →
  (- sqrt 16 = -4) := 
by 
  simp
  sorry

end proof_problem_l267_267663


namespace reduced_bucket_fraction_l267_267821

theorem reduced_bucket_fraction (C : ℝ) (F : ℝ) (h : 25 * F * C = 10 * C) : F = 2 / 5 :=
by sorry

end reduced_bucket_fraction_l267_267821


namespace lamp_height_difference_l267_267309

def old_lamp_height : ℝ := 1
def new_lamp_height : ℝ := 2.3333333333333335
def height_difference : ℝ := new_lamp_height - old_lamp_height

theorem lamp_height_difference :
  height_difference = 1.3333333333333335 := by
  sorry

end lamp_height_difference_l267_267309


namespace smallest_three_digit_multiple_of_17_l267_267967

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ ∃ (k : ℕ), n = 17 * k ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l267_267967


namespace song_book_cost_correct_l267_267690

/-- Define the constants for the problem. -/
def clarinet_cost : ℝ := 130.30
def pocket_money : ℝ := 12.32
def total_spent : ℝ := 141.54

/-- Prove the cost of the song book. -/
theorem song_book_cost_correct :
  (total_spent - clarinet_cost) = 11.24 :=
by
  sorry

end song_book_cost_correct_l267_267690


namespace impossible_sequence_l267_267633

def letters_order : List ℕ := [1, 2, 3, 4, 5]

def is_typing_sequence (order : List ℕ) (seq : List ℕ) : Prop :=
  sorry -- This function will evaluate if a sequence is possible given the order

theorem impossible_sequence : ¬ is_typing_sequence letters_order [4, 5, 2, 3, 1] :=
  sorry

end impossible_sequence_l267_267633


namespace eleventh_term_arithmetic_sequence_l267_267441

theorem eleventh_term_arithmetic_sequence :
  ∀ (S₇ a₁ : ℕ) (a : ℕ → ℕ), 
  (S₇ = 77) → 
  (a₁ = 5) → 
  (S₇ = ∑ i in (finset.range 7), a (i + 1)) → 
  (a 1 = a₁) →
  (∀ n, a n = a₁ + (n - 1) * 2) →  -- The correct common difference d is implicitly assumed to be 2
  a 11 = 25 :=
by 
  intros S₇ a₁ a hS h₁ hSum ha ha_formula
  -- Proof goes here (omitted using sorry for now)
  sorry

end eleventh_term_arithmetic_sequence_l267_267441


namespace BaSO4_molecular_weight_l267_267120

noncomputable def Ba : ℝ := 137.327
noncomputable def S : ℝ := 32.065
noncomputable def O : ℝ := 15.999
noncomputable def BaSO4 : ℝ := Ba + S + 4 * O

theorem BaSO4_molecular_weight : BaSO4 = 233.388 := by
  sorry

end BaSO4_molecular_weight_l267_267120


namespace not_prime_a_l267_267327

theorem not_prime_a 
  (a b : ℕ) 
  (h₁ : 0 < a)
  (h₂ : 0 < b)
  (h₃ : ∃ k : ℤ, (5 * a^4 + a^2) = k * (b^4 + 3 * b^2 + 4))
  : ¬ Nat.Prime a := 
sorry

end not_prime_a_l267_267327


namespace smallest_three_digit_multiple_of_17_l267_267908

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ 17 * n ∧ 17 * n ≤ 999 ∧ 17 * n = 102 :=
begin
  sorry
end

end smallest_three_digit_multiple_of_17_l267_267908


namespace garden_fencing_l267_267678

theorem garden_fencing (length width : ℕ) (h1 : length = 80) (h2 : width = length / 2) : 2 * (length + width) = 240 :=
by
  sorry

end garden_fencing_l267_267678
