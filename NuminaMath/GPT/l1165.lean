import Mathlib

namespace problem_statement_l1165_116518

theorem problem_statement : 8 * 5.4 - 0.6 * 10 / 1.2 = 38.2 :=
by
  sorry

end problem_statement_l1165_116518


namespace total_expense_in_decade_l1165_116551

/-- Definition of yearly expense on car insurance -/
def yearly_expense : ℕ := 2000

/-- Definition of the number of years in a decade -/
def years_in_decade : ℕ := 10

/-- Proof that the total expense in a decade is 20000 dollars -/
theorem total_expense_in_decade : yearly_expense * years_in_decade = 20000 :=
by
  sorry

end total_expense_in_decade_l1165_116551


namespace tim_cantaloupes_l1165_116503

theorem tim_cantaloupes (fred_cantaloupes : ℕ) (total_cantaloupes : ℕ) (h1 : fred_cantaloupes = 38) (h2 : total_cantaloupes = 82) : total_cantaloupes - fred_cantaloupes = 44 :=
by {
  -- proof steps go here
  sorry
}

end tim_cantaloupes_l1165_116503


namespace woman_work_completion_days_l1165_116542

def work_completion_days_man := 6
def work_completion_days_boy := 9
def work_completion_days_combined := 3

theorem woman_work_completion_days : 
  (1 / work_completion_days_man + W + 1 / work_completion_days_boy = 1 / work_completion_days_combined) →
  W = 1 / 18 → 
  1 / W = 18 :=
by
  intros h₁ h₂
  sorry

end woman_work_completion_days_l1165_116542


namespace num_distinct_five_digit_integers_with_product_of_digits_18_l1165_116573

theorem num_distinct_five_digit_integers_with_product_of_digits_18 :
  ∃ (n : ℕ), n = 70 ∧ ∀ (a b c d e : ℕ),
    a * b * c * d * e = 18 ∧ 
    1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 1 ≤ d ∧ d ≤ 9 ∧ 1 ≤ e ∧ e ≤ 9 → 
    (∃ (s : Finset (Fin 100000)), s.card = n) :=
  sorry

end num_distinct_five_digit_integers_with_product_of_digits_18_l1165_116573


namespace find_divisor_l1165_116544

theorem find_divisor (q r D : ℕ) (hq : q = 120) (hr : r = 333) (hD : 55053 = D * q + r) : D = 456 :=
by
  sorry

end find_divisor_l1165_116544


namespace probability_no_3x3_red_square_l1165_116501

def grid_probability (m n : ℕ) : Prop :=
  (gcd m n = 1) ∧ ((m : ℚ) / n = 170 / 171)

theorem probability_no_3x3_red_square (m n : ℕ) (h1 : grid_probability m n) : m + n = 341 :=
by
  sorry

end probability_no_3x3_red_square_l1165_116501


namespace geometric_sequence_smallest_n_l1165_116508

def geom_seq (n : ℕ) (r : ℝ) (b₁ : ℝ) : ℝ := 
  b₁ * r^(n-1)

theorem geometric_sequence_smallest_n 
  (b₁ b₂ b₃ : ℝ) (r : ℝ)
  (h₁ : b₁ = 2)
  (h₂ : b₂ = 6)
  (h₃ : b₃ = 18)
  (h_seq : ∀ n, bₙ = geom_seq n r b₁) :
  ∃ n, n = 5 ∧ geom_seq n r 2 = 324 :=
by
  sorry

end geometric_sequence_smallest_n_l1165_116508


namespace remainder_x_plus_3uy_div_y_l1165_116502

theorem remainder_x_plus_3uy_div_y (x y u v : ℕ) (hx : x = u * y + v) (hv_range : 0 ≤ v ∧ v < y) :
  (x + 3 * u * y) % y = v :=
by
  sorry

end remainder_x_plus_3uy_div_y_l1165_116502


namespace radius_increase_l1165_116593

theorem radius_increase (C1 C2 : ℝ) (h1 : C1 = 30) (h2 : C2 = 40) : 
  let r1 := C1 / (2 * Real.pi)
  let r2 := C2 / (2 * Real.pi)
  let Δr := r2 - r1
  Δr = 5 / Real.pi := by
sorry

end radius_increase_l1165_116593


namespace sqrt_inequality_l1165_116534

theorem sqrt_inequality (x : ℝ) (h : ∀ r : ℝ, r = 2 * x - 1 → r ≥ 0) : x ≥ 1 / 2 :=
sorry

end sqrt_inequality_l1165_116534


namespace percentage_decrease_in_price_l1165_116564

theorem percentage_decrease_in_price (original_price new_price decrease percentage : ℝ) :
  original_price = 1300 → new_price = 988 →
  decrease = original_price - new_price →
  percentage = (decrease / original_price) * 100 →
  percentage = 24 := by
  sorry

end percentage_decrease_in_price_l1165_116564


namespace total_shaded_area_l1165_116541

theorem total_shaded_area 
  (side': ℝ) (d: ℝ) (s: ℝ)
  (h1: 12 / d = 4)
  (h2: d / s = 4) : 
  d = 3 →
  s = 3 / 4 →
  (π * (d / 2) ^ 2 + 8 * s ^ 2) = 9 * π / 4 + 9 / 2 :=
by
  intro h3 h4
  have h5 : d = 3 := h3
  have h6 : s = 3 / 4 := h4
  rw [h5, h6]
  sorry

end total_shaded_area_l1165_116541


namespace Lorin_black_marbles_l1165_116591

variable (B : ℕ)

def Jimmy_yellow_marbles := 22
def Alex_yellow_marbles := Jimmy_yellow_marbles / 2
def Alex_black_marbles := 2 * B
def Alex_total_marbles := Alex_yellow_marbles + Alex_black_marbles

theorem Lorin_black_marbles : Alex_total_marbles = 19 → B = 4 :=
by
  intros h
  unfold Alex_total_marbles at h
  unfold Alex_yellow_marbles at h
  unfold Alex_black_marbles at h
  norm_num at h
  exact sorry

end Lorin_black_marbles_l1165_116591


namespace sue_receives_correct_answer_l1165_116507

theorem sue_receives_correct_answer (x : ℕ) (y : ℕ) (z : ℕ) (h1 : y = 3 * (x + 2)) (h2 : z = 3 * (y - 2)) (hx : x = 6) : z = 66 :=
by
  sorry

end sue_receives_correct_answer_l1165_116507


namespace sum_of_reciprocals_l1165_116587

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 3 * x * y) : 
  1 / x + 1 / y = 3 :=
by
  sorry

end sum_of_reciprocals_l1165_116587


namespace greatest_third_side_l1165_116531

theorem greatest_third_side (a b c : ℝ) (h₀: a = 5) (h₁: b = 11) (h₂ : 6 < c ∧ c < 16) : c ≤ 15 :=
by
  -- assumption applying that c needs to be within 6 and 16
  have h₃ : 6 < c := h₂.1
  have h₄: c < 16 := h₂.2
  -- need to show greatest integer c is 15
  sorry

end greatest_third_side_l1165_116531


namespace tangent_periodic_solution_l1165_116548

theorem tangent_periodic_solution :
  ∃ n : ℤ, -180 < n ∧ n < 180 ∧ (Real.tan (n * Real.pi / 180) = Real.tan (345 * Real.pi / 180)) := by
  sorry

end tangent_periodic_solution_l1165_116548


namespace largest_divisor_of_square_l1165_116585

theorem largest_divisor_of_square (n : ℕ) (h_pos : 0 < n) (h_div : 72 ∣ n ^ 2) : 12 ∣ n := 
sorry

end largest_divisor_of_square_l1165_116585


namespace circle_standard_equation_l1165_116554

noncomputable def circle_equation (a : ℝ) : Prop :=
  ∃ (x y : ℝ), (x - a)^2 + y^2 = 1

theorem circle_standard_equation : circle_equation 2 := by
  sorry

end circle_standard_equation_l1165_116554


namespace arithmetic_sequence_sum_l1165_116567

theorem arithmetic_sequence_sum :
  ∃ a b : ℕ, ∀ d : ℕ,
    d = 5 →
    a = 28 →
    b = 33 →
    a + b = 61 :=
by
  sorry

end arithmetic_sequence_sum_l1165_116567


namespace nap_hours_in_70_days_l1165_116584

-- Define the variables and conditions
variable (n d a b c e : ℕ)  -- assuming they are natural numbers

-- Define the total nap hours function
noncomputable def total_nap_hours (n d a b c e : ℕ) : ℕ :=
  (a + b) * 10

-- The statement to prove
theorem nap_hours_in_70_days (n d a b c e : ℕ) :
  total_nap_hours n d a b c e = (a + b) * 10 :=
by sorry

end nap_hours_in_70_days_l1165_116584


namespace campers_rowing_morning_equals_41_l1165_116506

def campers_went_rowing_morning (hiking_morning : ℕ) (rowing_afternoon : ℕ) (total : ℕ) : ℕ :=
  total - (hiking_morning + rowing_afternoon)

theorem campers_rowing_morning_equals_41 :
  ∀ (hiking_morning rowing_afternoon total : ℕ), hiking_morning = 4 → rowing_afternoon = 26 → total = 71 → campers_went_rowing_morning hiking_morning rowing_afternoon total = 41 := by
  intros hiking_morning rowing_afternoon total hiking_morning_cond rowing_afternoon_cond total_cond
  rw [hiking_morning_cond, rowing_afternoon_cond, total_cond]
  exact rfl

end campers_rowing_morning_equals_41_l1165_116506


namespace exists_strictly_increasing_sequences_l1165_116586

theorem exists_strictly_increasing_sequences :
  ∃ u v : ℕ → ℕ, (∀ n, u n < u (n + 1)) ∧ (∀ n, v n < v (n + 1)) ∧ (∀ n, 5 * u n * (u n + 1) = v n ^ 2 + 1) :=
sorry

end exists_strictly_increasing_sequences_l1165_116586


namespace sum_of_monomials_same_type_l1165_116517

theorem sum_of_monomials_same_type 
  (x y : ℝ) 
  (m n : ℕ) 
  (h1 : m = 1) 
  (h2 : 3 = n + 1) : 
  (2 * x ^ m * y ^ 3) + (-5 * x * y ^ (n + 1)) = -3 * x * y ^ 3 := 
by 
  sorry

end sum_of_monomials_same_type_l1165_116517


namespace lcm_18_24_eq_72_l1165_116580

-- Conditions
def factorization_18 : Nat × Nat := (1, 2) -- 18 = 2^1 * 3^2
def factorization_24 : Nat × Nat := (3, 1) -- 24 = 2^3 * 3^1

-- Definition of LCM using the highest powers from factorizations
def LCM (a b : Nat × Nat) : Nat :=
  let (p1, q1) := a
  let (p2, q2) := b
  (2^max p1 p2) * (3^max q1 q2)

-- Proof statement
theorem lcm_18_24_eq_72 : LCM factorization_18 factorization_24 = 72 :=
by
  sorry

end lcm_18_24_eq_72_l1165_116580


namespace rational_point_partition_exists_l1165_116556

open Set

-- Define rational numbers
noncomputable def Q : Set ℚ :=
  {x | True}

-- Define the set of rational points in the plane
def I : Set (ℚ × ℚ) := 
  {p | p.1 ∈ Q ∧ p.2 ∈ Q}

-- Statement of the theorem
theorem rational_point_partition_exists :
  ∃ (A B : Set (ℚ × ℚ)),
    (∀ (y : ℚ), {p ∈ A | p.1 = y}.Finite) ∧
    (∀ (x : ℚ), {p ∈ B | p.2 = x}.Finite) ∧
    (A ∪ B = I) ∧
    (A ∩ B = ∅) :=
sorry

end rational_point_partition_exists_l1165_116556


namespace sum_consecutive_integers_150_l1165_116515

theorem sum_consecutive_integers_150 (n : ℕ) (a : ℕ) (hn : n ≥ 3) (hdiv : 300 % n = 0) :
  n * (2 * a + n - 1) = 300 ↔ (a > 0) → n = 3 ∨ n = 5 ∨ n = 15 :=
by sorry

end sum_consecutive_integers_150_l1165_116515


namespace remainder_of_3042_div_98_l1165_116597

theorem remainder_of_3042_div_98 : 3042 % 98 = 4 := 
by
  sorry

end remainder_of_3042_div_98_l1165_116597


namespace evaluate_nested_square_root_l1165_116533

-- Define the condition
def pos_real_solution (x : ℝ) : Prop := x = Real.sqrt (18 + x)

-- State the theorem
theorem evaluate_nested_square_root :
  ∃ (x : ℝ), pos_real_solution x ∧ x = (1 + Real.sqrt 73) / 2 :=
sorry

end evaluate_nested_square_root_l1165_116533


namespace tan_30_eq_sqrt3_div_3_l1165_116555

/-- Statement that proves the value of tang of 30 degrees, given the cosine
    and sine values. -/
theorem tan_30_eq_sqrt3_div_3 
  (cos_30 : Real) (sin_30 : Real) 
  (hcos : cos_30 = Real.sqrt 3 / 2) 
  (hsin : sin_30 = 1 / 2) : 
    Real.tan 30 = Real.sqrt 3 / 3 := 
by 
  sorry

end tan_30_eq_sqrt3_div_3_l1165_116555


namespace evaluate_expr_l1165_116500

theorem evaluate_expr : Int.ceil (5 / 4 : ℚ) + Int.floor (-5 / 4 : ℚ) = 0 := by
  sorry

end evaluate_expr_l1165_116500


namespace range_of_m_l1165_116519

noncomputable def f (x : ℝ) : ℝ := sorry -- to be defined as an odd, decreasing function

theorem range_of_m 
  (hf_odd : ∀ x, f (-x) = -f x) -- f is odd
  (hf_decreasing : ∀ x y, x < y → f y < f x) -- f is strictly decreasing
  (h_condition : ∀ m, f (1 - m) + f (1 - m^2) < 0) :
  ∀ m, (0 < m ∧ m < 1) :=
sorry

end range_of_m_l1165_116519


namespace only_option_d_determines_location_l1165_116543

-- Define the problem conditions in Lean
inductive LocationOption where
  | OptionA : LocationOption
  | OptionB : LocationOption
  | OptionC : LocationOption
  | OptionD : LocationOption

-- Define a function that takes a LocationOption and returns whether it can determine a specific location
def determine_location (option : LocationOption) : Prop :=
  match option with
  | LocationOption.OptionD => True
  | LocationOption.OptionA => False
  | LocationOption.OptionB => False
  | LocationOption.OptionC => False

-- Prove that only option D can determine a specific location
theorem only_option_d_determines_location : ∀ (opt : LocationOption), determine_location opt ↔ opt = LocationOption.OptionD := by
  intro opt
  cases opt
  · simp [determine_location, LocationOption.OptionA]
  · simp [determine_location, LocationOption.OptionB]
  · simp [determine_location, LocationOption.OptionC]
  · simp [determine_location, LocationOption.OptionD]

end only_option_d_determines_location_l1165_116543


namespace min_value_of_function_l1165_116566

theorem min_value_of_function (x : ℝ) (hx : x > 0) :
  ∃ y, y = (3 + x + x^2) / (1 + x) ∧ y = -1 + 2 * Real.sqrt 3 :=
sorry

end min_value_of_function_l1165_116566


namespace part1_part2_l1165_116578

open Real

noncomputable def f (x : ℝ) : ℝ :=
  2 * x - (x + 1) * log x

noncomputable def g (x : ℝ) (a : ℝ) : ℝ :=
  x * log x - a * x^2 - 1

/- First part: Prove that for all x \in (1, +\infty), f(x) < 2 -/
theorem part1 (x : ℝ) (hx : 1 < x) : f x < 2 := sorry

/- Second part: Prove that if g(x) = 0 has two roots x₁ and x₂, then 
   (log x₁ + log x₂) / 2 > 1 + 2 / sqrt (x₁ * x₂) -/
theorem part2 (a x₁ x₂ : ℝ) (hx₁ : g x₁ a = 0) (hx₂ : g x₂ a = 0) : 
  (log x₁ + log x₂) / 2 > 1 + 2 / sqrt (x₁ * x₂) := sorry

end part1_part2_l1165_116578


namespace geometric_sequence_b_l1165_116568

theorem geometric_sequence_b (a b c : Real) (h1 : a = 5 + 2 * Real.sqrt 6) (h2 : c = 5 - 2 * Real.sqrt 6) (h3 : ∃ r, b = r * a ∧ c = r * b) :
  b = 1 ∨ b = -1 :=
by
  sorry

end geometric_sequence_b_l1165_116568


namespace abs_ab_eq_2_sqrt_65_l1165_116588

theorem abs_ab_eq_2_sqrt_65
  (a b : ℝ)
  (h1 : b^2 - a^2 = 16)
  (h2 : a^2 + b^2 = 36) :
  |a * b| = 2 * Real.sqrt 65 := 
sorry

end abs_ab_eq_2_sqrt_65_l1165_116588


namespace carlton_outfit_count_l1165_116576

-- Definitions of conditions
def sweater_vests (s : ℕ) : ℕ := 2 * s
def button_up_shirts : ℕ := 3
def outfits (v s : ℕ) : ℕ := v * s

-- Theorem statement
theorem carlton_outfit_count : outfits (sweater_vests button_up_shirts) button_up_shirts = 18 :=
by
  sorry

end carlton_outfit_count_l1165_116576


namespace b3_b7_equals_16_l1165_116523

variable {a b : ℕ → ℝ}
variable {d : ℝ}

-- Conditions: a is an arithmetic sequence with common difference d
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Condition: b is a geometric sequence
def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

-- Given condition on the arithmetic sequence a
def condition_on_a (a : ℕ → ℝ) (d : ℝ) : Prop :=
  2 * a 2 - (a 5) ^ 2 + 2 * a 8 = 0

-- Define the specific arithmetic sequence in terms of d and a5
noncomputable def a_seq (a5 d : ℝ) : ℕ → ℝ
| 0 => a5 - 5 * d
| 1 => a5 - 4 * d
| 2 => a5 - 3 * d
| 3 => a5 - 2 * d
| 4 => a5 - d
| 5 => a5
| 6 => a5 + d
| 7 => a5 + 2 * d
| 8 => a5 + 3 * d
| 9 => a5 + 4 * d
| n => 0 -- extending for unspecified

-- Condition: b_5 = a_5
def b_equals_a (a b : ℕ → ℝ) : Prop :=
  b 5 = a 5

-- Theorem: Given the conditions, prove b_3 * b_7 = 16
theorem b3_b7_equals_16 (a b : ℕ → ℝ) (d : ℝ)
  (ha_seq : is_arithmetic_sequence a d)
  (hb_seq : is_geometric_sequence b)
  (h_cond_a : condition_on_a a d)
  (h_b_equals_a : b_equals_a a b) : b 3 * b 7 = 16 :=
by
  sorry

end b3_b7_equals_16_l1165_116523


namespace relationship_among_a_b_c_l1165_116535

noncomputable def a : ℝ := Real.log 4 / Real.log 5
noncomputable def b : ℝ := (Real.log 3 / Real.log 5)^2
noncomputable def c : ℝ := Real.log 5 / Real.log 4

theorem relationship_among_a_b_c : b < a ∧ a < c := by
  sorry

end relationship_among_a_b_c_l1165_116535


namespace find_n_l1165_116574

noncomputable def factorial : ℕ → ℕ
| 0 => 1
| (n+1) => (n+1) * factorial n

theorem find_n (n : ℕ) (h : n * factorial (n + 1) + factorial (n + 1) = 5040) : n = 5 :=
sorry

end find_n_l1165_116574


namespace boxes_needed_l1165_116559

def num_red_pencils := 45
def num_yellow_pencils := 80
def num_pencils_per_red_box := 15
def num_pencils_per_blue_box := 25
def num_pencils_per_yellow_box := 10
def num_pencils_per_green_box := 30

def num_blue_pencils (x : Nat) := 3 * x + 6
def num_green_pencils (red : Nat) (blue : Nat) := 2 * (red + blue)

def total_boxes_needed : Nat :=
  let red_boxes := num_red_pencils / num_pencils_per_red_box
  let blue_boxes := (num_blue_pencils num_red_pencils) / num_pencils_per_blue_box + 
                    if ((num_blue_pencils num_red_pencils) % num_pencils_per_blue_box) = 0 then 0 else 1
  let yellow_boxes := num_yellow_pencils / num_pencils_per_yellow_box
  let green_boxes := (num_green_pencils num_red_pencils (num_blue_pencils num_red_pencils)) / num_pencils_per_green_box + 
                     if ((num_green_pencils num_red_pencils (num_blue_pencils num_red_pencils)) % num_pencils_per_green_box) = 0 then 0 else 1
  red_boxes + blue_boxes + yellow_boxes + green_boxes

theorem boxes_needed : total_boxes_needed = 30 := sorry

end boxes_needed_l1165_116559


namespace abs_neg_sub_three_eq_zero_l1165_116532

theorem abs_neg_sub_three_eq_zero : |(-3 : ℤ)| - 3 = 0 :=
by sorry

end abs_neg_sub_three_eq_zero_l1165_116532


namespace emily_extra_distance_five_days_l1165_116589

-- Define the distances
def distance_troy : ℕ := 75
def distance_emily : ℕ := 98

-- Emily's extra walking distance in one-way
def extra_one_way : ℕ := distance_emily - distance_troy

-- Emily's extra walking distance in a round trip
def extra_round_trip : ℕ := extra_one_way * 2

-- The extra distance Emily walks in five days
def extra_five_days : ℕ := extra_round_trip * 5

-- Theorem to be proven
theorem emily_extra_distance_five_days : extra_five_days = 230 := by
  -- Proof will go here
  sorry

end emily_extra_distance_five_days_l1165_116589


namespace inequality_implies_strict_inequality_l1165_116513

theorem inequality_implies_strict_inequality (x y z : ℝ) (h : x^2 + x * y + x * z < 0) : y^2 > 4 * x * z :=
sorry

end inequality_implies_strict_inequality_l1165_116513


namespace sampling_methods_correct_l1165_116511

-- Assuming definitions for the populations for both surveys
structure CommunityHouseholds where
  high_income : Nat
  middle_income : Nat
  low_income : Nat

structure ArtisticStudents where
  total_students : Nat

-- Given conditions
def households_population : CommunityHouseholds := { high_income := 125, middle_income := 280, low_income := 95 }
def students_population : ArtisticStudents := { total_students := 15 }

-- Correct answer according to the conditions
def appropriate_sampling_methods (ch: CommunityHouseholds) (as: ArtisticStudents) : String :=
  if ch.high_income > 0 ∧ ch.middle_income > 0 ∧ ch.low_income > 0 ∧ as.total_students ≥ 3 then
    "B" -- ① Stratified sampling, ② Simple random sampling
  else
    "Invalid"

theorem sampling_methods_correct :
  appropriate_sampling_methods households_population students_population = "B" := by
  sorry

end sampling_methods_correct_l1165_116511


namespace find_sum_l1165_116530

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (2 + x) + f (2 - x) = 0

theorem find_sum (f : ℝ → ℝ) (h_odd : odd_function f) (h_func : functional_equation f) (h_val : f 1 = 9) :
  f 2016 + f 2017 + f 2018 = 9 :=
  sorry

end find_sum_l1165_116530


namespace lcm_eq_792_l1165_116581

-- Define the integers
def a : ℕ := 8
def b : ℕ := 9
def c : ℕ := 11

-- Define their prime factorizations (included for clarity, though not directly necessary)
def a_factorization : a = 2^3 := rfl
def b_factorization : b = 3^2 := rfl
def c_factorization : c = 11 := rfl

-- Define the LCM function
def lcm_abc := Nat.lcm (Nat.lcm a b) c

-- Prove that lcm of a, b, c is 792
theorem lcm_eq_792 : lcm_abc = 792 := 
by
  -- Include the necessary properties of LCM and prime factorizations if necessary
  sorry

end lcm_eq_792_l1165_116581


namespace measured_diagonal_length_l1165_116572

theorem measured_diagonal_length (a b c d diag : Real)
  (h1 : a = 1) (h2 : b = 2) (h3 : c = 2.8) (h4 : d = 5) (hd : diag = 7.5) :
  diag = 2.8 :=
sorry

end measured_diagonal_length_l1165_116572


namespace cube_sum_equal_one_l1165_116590

theorem cube_sum_equal_one (x y z : ℝ) (h1 : x + y + z = 3) (h2 : xy + xz + yz = 1) (h3 : xyz = 1) :
  x^3 + y^3 + z^3 = 1 := 
sorry

end cube_sum_equal_one_l1165_116590


namespace remainder_13_pow_2000_mod_1000_l1165_116558

theorem remainder_13_pow_2000_mod_1000 :
  (13^2000) % 1000 = 1 := 
by 
  sorry

end remainder_13_pow_2000_mod_1000_l1165_116558


namespace tracy_first_week_books_collected_l1165_116575

-- Definitions for collection multipliers
def first_week (T : ℕ) := T
def second_week (T : ℕ) := 2 * T + 3 * T
def third_week (T : ℕ) := 3 * T + 4 * T + (T / 2)
def fourth_week (T : ℕ) := 4 * T + 5 * T + T
def fifth_week (T : ℕ) := 5 * T + 6 * T + 2 * T
def sixth_week (T : ℕ) := 6 * T + 7 * T + 3 * T

-- Summing up total books collected
def total_books_collected (T : ℕ) : ℕ :=
  first_week T + second_week T + third_week T + fourth_week T + fifth_week T + sixth_week T

-- Proof statement (unchanged for now)
theorem tracy_first_week_books_collected (T : ℕ) :
  total_books_collected T = 1025 → T = 20 :=
by
  sorry

end tracy_first_week_books_collected_l1165_116575


namespace keith_stored_bales_l1165_116582

theorem keith_stored_bales (initial_bales added_bales final_bales : ℕ) :
  initial_bales = 22 → final_bales = 89 → final_bales = initial_bales + added_bales → added_bales = 67 :=
by
  intros h_initial h_final h_eq
  sorry

end keith_stored_bales_l1165_116582


namespace pastries_total_l1165_116594

-- We start by defining the conditions
def Calvin_pastries (Frank_pastries Grace_pastries : ℕ) : ℕ := Frank_pastries + 8
def Phoebe_pastries (Frank_pastries Grace_pastries : ℕ) : ℕ := Frank_pastries + 8
def Grace_pastries : ℕ := 30
def have_same_pastries (Calvin_pastries Phoebe_pastries Grace_pastries : ℕ) : Prop :=
  Calvin_pastries + 5 = Grace_pastries ∧ Phoebe_pastries + 5 = Grace_pastries

-- Total number of pastries held by Calvin, Phoebe, Frank, and Grace
def total_pastries (Frank_pastries Calvin_pastries Phoebe_pastries Grace_pastries : ℕ) : ℕ :=
  Frank_pastries + Calvin_pastries + Phoebe_pastries + Grace_pastries

-- The statement to prove
theorem pastries_total (Frank_pastries : ℕ) : 
  have_same_pastries (Calvin_pastries Frank_pastries Grace_pastries) (Phoebe_pastries Frank_pastries Grace_pastries) Grace_pastries → 
  Frank_pastries + Calvin_pastries Frank_pastries Grace_pastries + Phoebe_pastries Frank_pastries Grace_pastries + Grace_pastries = 97 :=
by
  sorry

end pastries_total_l1165_116594


namespace D_72_l1165_116547

def D (n : ℕ) : ℕ :=
  -- Definition of D(n) should be provided here
  sorry

theorem D_72 : D 72 = 121 :=
  sorry

end D_72_l1165_116547


namespace number_of_meetings_l1165_116561

-- Define the data for the problem
def pool_length : ℕ := 120
def swimmer_A_speed : ℕ := 4
def swimmer_B_speed : ℕ := 3
def total_time_seconds : ℕ := 15 * 60
def swimmer_A_turn_break_seconds : ℕ := 2
def swimmer_B_turn_break_seconds : ℕ := 0

-- Define the round trip time for each swimmer
def swimmer_A_round_trip_time : ℕ := 2 * (pool_length / swimmer_A_speed) + 2 * swimmer_A_turn_break_seconds
def swimmer_B_round_trip_time : ℕ := 2 * (pool_length / swimmer_B_speed) + 2 * swimmer_B_turn_break_seconds

-- Define the least common multiple of the round trip times
def lcm_round_trip_time : ℕ := Nat.lcm swimmer_A_round_trip_time swimmer_B_round_trip_time

-- Define the statement to prove
theorem number_of_meetings (lcm_round_trip_time : ℕ) : 
  (24 * (total_time_seconds / lcm_round_trip_time) + ((total_time_seconds % lcm_round_trip_time) / (pool_length / (swimmer_A_speed + swimmer_B_speed)))) = 51 := 
sorry

end number_of_meetings_l1165_116561


namespace roses_count_l1165_116552

def total_roses : Nat := 80
def red_roses : Nat := 3 * total_roses / 4
def remaining_roses : Nat := total_roses - red_roses
def yellow_roses : Nat := remaining_roses / 4
def white_roses : Nat := remaining_roses - yellow_roses

theorem roses_count :
  red_roses + white_roses = 75 :=
by
  sorry

end roses_count_l1165_116552


namespace eggplant_weight_l1165_116521

-- Define the conditions
def number_of_cucumbers : ℕ := 25
def weight_per_cucumber_basket : ℕ := 30
def number_of_eggplants : ℕ := 32
def total_weight : ℕ := 1870

-- Define the statement to be proved
theorem eggplant_weight :
  (total_weight - (number_of_cucumbers * weight_per_cucumber_basket)) / number_of_eggplants =
  (1870 - (25 * 30)) / 32 := 
by sorry

end eggplant_weight_l1165_116521


namespace median_of_trapezoid_l1165_116509

theorem median_of_trapezoid (h : ℝ) (x : ℝ) 
  (triangle_area_eq_trapezoid_area : (1 / 2) * 24 * h = ((x + (2 * x)) / 2) * h) : 
  ((x + (2 * x)) / 2) = 12 := by
  sorry

end median_of_trapezoid_l1165_116509


namespace conversion1_conversion2_conversion3_minutes_conversion3_seconds_conversion4_l1165_116592

theorem conversion1 : 4 * 60 + 35 = 275 := by
  sorry

theorem conversion2 : 4 * 1000 + 35 = 4035 := by
  sorry

theorem conversion3_minutes : 678 / 60 = 11 := by
  sorry

theorem conversion3_seconds : 678 % 60 = 18 := by
  sorry

theorem conversion4 : 120000 / 10000 = 12 := by
  sorry

end conversion1_conversion2_conversion3_minutes_conversion3_seconds_conversion4_l1165_116592


namespace interest_rate_l1165_116538

theorem interest_rate (SI P : ℝ) (T : ℕ) (h₁: SI = 70) (h₂ : P = 700) (h₃ : T = 4) : 
  (SI / (P * T)) * 100 = 2.5 :=
by
  sorry

end interest_rate_l1165_116538


namespace truck_initial_gas_ratio_l1165_116546

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

end truck_initial_gas_ratio_l1165_116546


namespace triangle_area_l1165_116596

theorem triangle_area (a b c : ℝ) (h1 : a = 5) (h2 : a + b = 13) (h3 : c = Real.sqrt (a^2 + b^2)) : 
  (1 / 2) * a * b = 20 :=
by
  sorry

end triangle_area_l1165_116596


namespace compare_abc_l1165_116536

noncomputable def a : ℝ := Real.log 5 / Real.log 2
noncomputable def b : ℝ := Real.log 6 / Real.log 2
noncomputable def c : ℝ := 9 ^ (1 / 2 : ℝ)

theorem compare_abc : c > b ∧ b > a := 
by
  sorry

end compare_abc_l1165_116536


namespace range_of_a_l1165_116526

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * x + 1 > 0) ↔ (0 ≤ a ∧ a ≤ 1) := 
sorry

end range_of_a_l1165_116526


namespace identify_first_brother_l1165_116525

-- Definitions for conditions
inductive Brother
| Trulya : Brother
| Falsa : Brother

-- Extracting conditions into Lean 4 statements
def first_brother_says : String := "Both cards are of the purplish suit."
def second_brother_says : String := "This is not true!"

axiom trulya_always_truthful : ∀ (b : Brother) (statement : String), b = Brother.Trulya ↔ (statement = first_brother_says ∨ statement = second_brother_says)
axiom falsa_always_lies : ∀ (b : Brother) (statement : String), b = Brother.Falsa ↔ ¬(statement = first_brother_says ∨ statement = second_brother_says)

-- Proof statement 
theorem identify_first_brother :
  ∃ (b : Brother), b = Brother.Trulya :=
sorry

end identify_first_brother_l1165_116525


namespace none_of_these_l1165_116514

def table : List (ℕ × ℕ) := [(1, 5), (2, 15), (3, 33), (4, 61), (5, 101)]

def formula_A (x : ℕ) : ℕ := 2 * x^3 + 3 * x^2 - x + 1
def formula_B (x : ℕ) : ℕ := 3 * x^3 + x^2 + x + 1
def formula_C (x : ℕ) : ℕ := 2 * x^3 + x^2 + x + 1
def formula_D (x : ℕ) : ℕ := 2 * x^3 + x^2 + x - 1

theorem none_of_these :
  ¬ (∀ (x y : ℕ), (x, y) ∈ table → (y = formula_A x ∨ y = formula_B x ∨ y = formula_C x ∨ y = formula_D x)) :=
by {
  sorry
}

end none_of_these_l1165_116514


namespace total_heads_l1165_116512

theorem total_heads (D P : ℕ) (h1 : D = 9) (h2 : 4 * D + 2 * P = 42) : D + P = 12 :=
by
  sorry

end total_heads_l1165_116512


namespace truncated_pyramid_distance_l1165_116570

noncomputable def distance_from_plane_to_base
  (a b : ℝ) (α : ℝ) : ℝ :=
  (a * (a - b) * Real.tan α) / (3 * a - b)

theorem truncated_pyramid_distance
  (a b : ℝ) (α : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_α : 0 < α) :
  (a * (a - b) * Real.tan α) / (3 * a - b) = distance_from_plane_to_base a b α :=
by
  sorry

end truncated_pyramid_distance_l1165_116570


namespace difference_in_percentage_l1165_116545

noncomputable def principal : ℝ := 600
noncomputable def timePeriod : ℝ := 10
noncomputable def interestDifference : ℝ := 300

theorem difference_in_percentage (R D : ℝ) (h : 60 * (R + D) - 60 * R = 300) : D = 5 := 
by
  -- Proof is not provided, as instructed
  sorry

end difference_in_percentage_l1165_116545


namespace value_of_expression_l1165_116527

theorem value_of_expression (a b : ℤ) (h : 2 * a - b = 10) : 2023 - 2 * a + b = 2013 :=
by
  sorry

end value_of_expression_l1165_116527


namespace simplify_and_evaluate_expr_l1165_116577

theorem simplify_and_evaluate_expr (a b : ℝ) (h1 : a = 1 / 2) (h2 : b = -4) :
  5 * (3 * a ^ 2 * b - a * b ^ 2) - 4 * (-a * b ^ 2 + 3 * a ^ 2 * b) = -11 :=
by
  sorry

end simplify_and_evaluate_expr_l1165_116577


namespace cinematic_academy_member_count_l1165_116550

theorem cinematic_academy_member_count (M : ℝ) 
  (h : (1 / 4) * M = 192.5) : M = 770 := 
by 
  -- proof omitted
  sorry

end cinematic_academy_member_count_l1165_116550


namespace find_m_range_l1165_116579

noncomputable def ellipse_symmetric_points_range (m : ℝ) : Prop :=
  -((2:ℝ) * Real.sqrt (13:ℝ) / 13) < m ∧ m < ((2:ℝ) * Real.sqrt (13:ℝ) / 13)

theorem find_m_range :
  ∃ m : ℝ, ellipse_symmetric_points_range m :=
sorry

end find_m_range_l1165_116579


namespace ratio_female_to_male_l1165_116520

theorem ratio_female_to_male (total_members : ℕ) (female_members : ℕ) (male_members : ℕ) 
  (h1 : total_members = 18) (h2 : female_members = 12) (h3 : male_members = total_members - female_members) : 
  (female_members : ℚ) / (male_members : ℚ) = 2 := 
by 
  sorry

end ratio_female_to_male_l1165_116520


namespace carlos_more_miles_than_dana_after_3_hours_l1165_116571

-- Define the conditions
variable (carlos_total_distance : ℕ)
variable (carlos_advantage : ℕ)
variable (dana_total_distance : ℕ)
variable (time_hours : ℕ)

-- State the condition values that are given in the problem
def conditions : Prop :=
  carlos_total_distance = 50 ∧
  carlos_advantage = 5 ∧
  dana_total_distance = 40 ∧
  time_hours = 3

-- State the proof goal
theorem carlos_more_miles_than_dana_after_3_hours
  (h : conditions carlos_total_distance carlos_advantage dana_total_distance time_hours) :
  carlos_total_distance - dana_total_distance = 10 :=
by
  sorry

end carlos_more_miles_than_dana_after_3_hours_l1165_116571


namespace problem_l1165_116540

theorem problem (a b : ℕ) (h1 : ∃ k : ℕ, a * b = k * k) (h2 : ∃ m : ℕ, (2 * a + 1) * (2 * b + 1) = m * m) :
  ∃ n : ℕ, n % 2 = 0 ∧ n > 2 ∧ ∃ p : ℕ, (a + n) * (b + n) = p * p :=
by
  sorry

end problem_l1165_116540


namespace cookie_cost_per_day_l1165_116583

theorem cookie_cost_per_day
    (days_in_April : ℕ)
    (cookies_per_day : ℕ)
    (total_spent : ℕ)
    (total_cookies : ℕ := days_in_April * cookies_per_day)
    (cost_per_cookie : ℕ := total_spent / total_cookies) :
  days_in_April = 30 ∧ cookies_per_day = 3 ∧ total_spent = 1620 → cost_per_cookie = 18 :=
by
  sorry

end cookie_cost_per_day_l1165_116583


namespace income_to_expenditure_ratio_l1165_116599

theorem income_to_expenditure_ratio (I E S : ℝ) (hI : I = 10000) (hS : S = 4000) (hSavings : S = I - E) : I / E = 5 / 3 := by
  -- To prove: I / E = 5 / 3 given hI, hS, and hSavings
  sorry

end income_to_expenditure_ratio_l1165_116599


namespace vertical_asymptote_values_l1165_116510

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := (x^2 - x + c) / (x^2 + x - 20)

theorem vertical_asymptote_values (c : ℝ) :
  (∃ x : ℝ, (x^2 + x - 20 = 0 ∧ x^2 - x + c = 0) ↔
   (c = -12 ∨ c = -30)) := sorry

end vertical_asymptote_values_l1165_116510


namespace approximation_of_11_28_relative_to_10000_l1165_116522

def place_value_to_approximate (x : Float) (reference : Float) : String :=
  if x < reference / 10 then "tens"
  else if x < reference / 100 then "hundreds"
  else if x < reference / 1000 then "thousands"
  else if x < reference / 10000 then "ten thousands"
  else "greater than ten thousands"

theorem approximation_of_11_28_relative_to_10000:
  place_value_to_approximate 11.28 10000 = "hundreds" :=
by
  -- Insert proof here
  sorry

end approximation_of_11_28_relative_to_10000_l1165_116522


namespace y_eq_fraction_x_l1165_116565

theorem y_eq_fraction_x (p : ℝ) (x y : ℝ) (hx : x = 1 + 2^p) (hy : y = 1 + 2^(-p)) : y = x / (x - 1) :=
sorry

end y_eq_fraction_x_l1165_116565


namespace fraction_irreducible_l1165_116516

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
by {
    sorry
}

end fraction_irreducible_l1165_116516


namespace perfect_square_trinomial_l1165_116569

theorem perfect_square_trinomial (m : ℤ) : 
  (x^2 - (m - 3) * x + 16 = (x - 4)^2) ∨ (x^2 - (m - 3) * x + 16 = (x + 4)^2) ↔ (m = -5 ∨ m = 11) := by
  sorry

end perfect_square_trinomial_l1165_116569


namespace find_savings_l1165_116539

-- Define the problem statement
def income_expenditure_problem (income expenditure : ℝ) (ratio : ℝ) : Prop :=
  (income / ratio = expenditure) ∧ (income = 20000)

-- Define the theorem for savings
theorem find_savings (income expenditure : ℝ) (ratio : ℝ) (h_ratio : ratio = 4 / 5) (h_income : income = 20000) : 
  income_expenditure_problem income expenditure ratio → income - expenditure = 4000 :=
by
  sorry

end find_savings_l1165_116539


namespace smallest_n_for_purple_l1165_116549

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

end smallest_n_for_purple_l1165_116549


namespace surface_area_of_sphere_l1165_116504

noncomputable def sphere_surface_area : ℝ :=
  let AB := 2
  let SA := 2
  let SB := 2
  let SC := 2
  let ABC_is_isosceles_right := true -- denotes the property
  let SABC_on_sphere := true -- denotes the property
  let R := (2 * Real.sqrt 3) / 3
  let surface_area := 4 * Real.pi * R^2
  surface_area

theorem surface_area_of_sphere : sphere_surface_area = (16 * Real.pi) / 3 := 
sorry

end surface_area_of_sphere_l1165_116504


namespace sum_multiple_of_3_probability_l1165_116595

noncomputable def probability_sum_multiple_of_3 (faces : List ℕ) (rolls : ℕ) (multiple : ℕ) : ℚ :=
  if rolls = 3 ∧ multiple = 3 ∧ faces = [1, 2, 3, 4, 5, 6] then 1 / 3 else 0

theorem sum_multiple_of_3_probability :
  probability_sum_multiple_of_3 [1, 2, 3, 4, 5, 6] 3 3 = 1 / 3 :=
by
  sorry

end sum_multiple_of_3_probability_l1165_116595


namespace sum_of_24_terms_l1165_116557

variable (a_1 d : ℝ)

def a (n : ℕ) : ℝ := a_1 + (n - 1) * d

theorem sum_of_24_terms 
  (h : (a 5 + a 10 + a 15 + a 20 = 20)) : 
  (12 * (2 * a_1 + 23 * d) = 120) :=
by
  sorry

end sum_of_24_terms_l1165_116557


namespace isosceles_right_triangle_sums_l1165_116560

theorem isosceles_right_triangle_sums (m n : ℝ)
  (h1: (1 * 2 + m * m + 2 * n) = 0)
  (h2: (1 + m^2 + 4) = (4 + m^2 + n^2)) :
  m + n = -1 :=
by {
  sorry
}

end isosceles_right_triangle_sums_l1165_116560


namespace cuboid_diagonal_length_l1165_116598

theorem cuboid_diagonal_length (x y z : ℝ) 
  (h1 : y * z = Real.sqrt 2) 
  (h2 : z * x = Real.sqrt 3)
  (h3 : x * y = Real.sqrt 6) : 
  Real.sqrt (x^2 + y^2 + z^2) = Real.sqrt 6 :=
sorry

end cuboid_diagonal_length_l1165_116598


namespace smallest_integer_mod_inverse_l1165_116553

theorem smallest_integer_mod_inverse (n : ℕ) (h1 : n > 1) (h2 : gcd n 1001 = 1) : n = 2 :=
sorry

end smallest_integer_mod_inverse_l1165_116553


namespace f_at_neg_one_l1165_116562

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x^2 + 3 * x + 16

noncomputable def f_with_r (x : ℝ) (a r : ℝ) : ℝ := (x^3 + a * x^2 + 3 * x + 16) * (x - r)

theorem f_at_neg_one (a b c r : ℝ) (h1 : ∀ x, g x a = 0 → f_with_r x a r = 0)
  (h2 : a - r = 5) (h3 : 16 - 3 * r = 150) (h4 : -16 * r = c) :
  f_with_r (-1) a r = -1347 :=
by
  sorry

end f_at_neg_one_l1165_116562


namespace shortest_distance_to_line_l1165_116524

open Classical

variables {P A B C : Type} [MetricSpace P] [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables (PA PB PC : ℝ)
variables (l : ℕ) -- l represents the line

-- Given conditions
def PA_dist : ℝ := 4
def PB_dist : ℝ := 5
def PC_dist : ℝ := 2

theorem shortest_distance_to_line (hPA : PA = PA_dist) (hPB : PB = PB_dist) (hPC : PC = PC_dist) :
  ∃ d, d ≤ 2 := 
sorry

end shortest_distance_to_line_l1165_116524


namespace implication_a_lt_b_implies_a_lt_b_plus_1_l1165_116529

theorem implication_a_lt_b_implies_a_lt_b_plus_1 (a b : ℝ) (h : a < b) : a < b + 1 := by
  sorry

end implication_a_lt_b_implies_a_lt_b_plus_1_l1165_116529


namespace determine_days_l1165_116505

-- Define the problem
def team_repair_time (x y : ℕ) : Prop :=
  ((1 / (x:ℝ)) + (1 / (y:ℝ)) = 1 / 18) ∧ 
  ((2 / 3 * x + 1 / 3 * y = 40))

theorem determine_days : ∃ x y : ℕ, team_repair_time x y :=
by
    use 45
    use 30
    have h1: (1/(45:ℝ) + 1/(30:ℝ)) = 1/18 := by
        sorry
    have h2: (2/3*45 + 1/3*30 = 40) := by
        sorry 
    exact ⟨h1, h2⟩

end determine_days_l1165_116505


namespace quadratic_solution_l1165_116537

theorem quadratic_solution (x : ℝ) : x^2 - 2 * x = 0 ↔ (x = 0 ∨ x = 2) := by
  sorry

end quadratic_solution_l1165_116537


namespace arithmetic_sequence_a2_a8_l1165_116528

variable {a : ℕ → ℝ}

-- given condition
axiom h1 : a 4 + a 5 + a 6 = 450

-- problem statement
theorem arithmetic_sequence_a2_a8 : a 2 + a 8 = 300 :=
by
  sorry

end arithmetic_sequence_a2_a8_l1165_116528


namespace domain_of_logarithmic_function_l1165_116563

theorem domain_of_logarithmic_function :
  ∀ x : ℝ, 2 - x > 0 ↔ x < 2 := 
by
  intro x
  sorry

end domain_of_logarithmic_function_l1165_116563
