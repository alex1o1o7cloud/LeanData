import Mathlib

namespace NUMINAMATH_GPT_set_inter_complement_l1527_152776

open Set

variable {α : Type*}
variable (U A B : Set α)

theorem set_inter_complement (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)
  (hU : U = {1, 2, 3, 4, 5})
  (hA : A = {1, 2, 3})
  (hB : B = {1, 4}) :
  ((U \ A) ∩ B) = {4} := 
by
  sorry

end NUMINAMATH_GPT_set_inter_complement_l1527_152776


namespace NUMINAMATH_GPT_mn_necessary_not_sufficient_l1527_152761

variable (m n : ℝ)

def is_ellipse (m n : ℝ) : Prop := 
  (m > 0) ∧ (n > 0) ∧ (m ≠ n)

theorem mn_necessary_not_sufficient : (mn > 0) → (is_ellipse m n) ↔ false := 
by sorry

end NUMINAMATH_GPT_mn_necessary_not_sufficient_l1527_152761


namespace NUMINAMATH_GPT_find_a_l1527_152706

theorem find_a (k x y a : ℝ) (hkx : k ≤ x) (hx3 : x ≤ 3) (hy7 : a ≤ y) (hy7' : y ≤ 7) (hy : y = k * x + 1) :
  a = 5 ∨ a = 1 - 3 * Real.sqrt 6 :=
sorry

end NUMINAMATH_GPT_find_a_l1527_152706


namespace NUMINAMATH_GPT_completion_time_l1527_152740

theorem completion_time (total_work : ℕ) (initial_num_men : ℕ) (initial_efficiency : ℝ)
  (new_num_men : ℕ) (new_efficiency : ℝ) :
  total_work = 12 ∧ initial_num_men = 4 ∧ initial_efficiency = 1.5 ∧
  new_num_men = 6 ∧ new_efficiency = 2.0 →
  total_work / (new_num_men * new_efficiency) = 1 :=
by
  sorry

end NUMINAMATH_GPT_completion_time_l1527_152740


namespace NUMINAMATH_GPT_no_negative_roots_l1527_152718

theorem no_negative_roots (x : ℝ) :
  x^4 - 4 * x^3 - 6 * x^2 - 3 * x + 9 = 0 → 0 ≤ x :=
by
  sorry

end NUMINAMATH_GPT_no_negative_roots_l1527_152718


namespace NUMINAMATH_GPT_probability_of_less_than_5_is_one_half_l1527_152752

noncomputable def probability_of_less_than_5 : ℚ :=
  let total_outcomes := 8
  let successful_outcomes := 4
  successful_outcomes / total_outcomes

theorem probability_of_less_than_5_is_one_half :
  probability_of_less_than_5 = 1 / 2 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_probability_of_less_than_5_is_one_half_l1527_152752


namespace NUMINAMATH_GPT_set_has_one_element_iff_double_root_l1527_152785

theorem set_has_one_element_iff_double_root (k : ℝ) :
  (∃ x, ∀ y, y^2 - k*y + 1 = 0 ↔ y = x) ↔ k = 2 ∨ k = -2 :=
by
  sorry

end NUMINAMATH_GPT_set_has_one_element_iff_double_root_l1527_152785


namespace NUMINAMATH_GPT_nth_term_closed_form_arithmetic_sequence_l1527_152739

open Nat

noncomputable def S (n : ℕ) : ℕ := 3 * n^2 + 4 * n
noncomputable def a (n : ℕ) : ℕ := if h : n > 0 then S n - S (n-1) else S n

theorem nth_term_closed_form (n : ℕ) (h : n > 0) : a n = 6 * n + 1 :=
by
  sorry

theorem arithmetic_sequence (n : ℕ) (h : n > 1) : a n - a (n - 1) = 6 :=
by
  sorry

end NUMINAMATH_GPT_nth_term_closed_form_arithmetic_sequence_l1527_152739


namespace NUMINAMATH_GPT_first_player_wins_if_take_one_initial_l1527_152744

theorem first_player_wins_if_take_one_initial :
  ∃ strategy : ℕ → ℕ, 
    (∀ n, strategy n = if n % 3 = 0 then 1 else 2) ∧ 
    strategy 99 = 1 ∧ 
    strategy 100 = 1 :=
sorry

end NUMINAMATH_GPT_first_player_wins_if_take_one_initial_l1527_152744


namespace NUMINAMATH_GPT_difference_of_squares_l1527_152758

theorem difference_of_squares :
  535^2 - 465^2 = 70000 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_squares_l1527_152758


namespace NUMINAMATH_GPT_books_bought_l1527_152771

theorem books_bought (cost_crayons cost_calculators total_money cost_per_bag bags_bought cost_per_book remaining_money books_bought : ℕ) 
  (h1: cost_crayons = 5 * 5)
  (h2: cost_calculators = 3 * 5)
  (h3: total_money = 200)
  (h4: cost_per_bag = 10)
  (h5: bags_bought = 11)
  (h6: remaining_money = total_money - (cost_crayons + cost_calculators) - (bags_bought * cost_per_bag)) :
  books_bought = remaining_money / cost_per_book → books_bought = 10 :=
by
  sorry

end NUMINAMATH_GPT_books_bought_l1527_152771


namespace NUMINAMATH_GPT_more_pie_eaten_l1527_152789

theorem more_pie_eaten (e f : ℝ) (h1 : e = 0.67) (h2 : f = 0.33) : e - f = 0.34 :=
by sorry

end NUMINAMATH_GPT_more_pie_eaten_l1527_152789


namespace NUMINAMATH_GPT_total_books_l1527_152726

theorem total_books (b1 b2 b3 b4 b5 b6 b7 b8 b9 : ℕ) :
  b1 = 56 →
  b2 = b1 + 2 →
  b3 = b2 + 2 →
  b4 = b3 + 2 →
  b5 = b4 + 2 →
  b6 = b5 + 2 →
  b7 = b6 - 4 →
  b8 = b7 - 4 →
  b9 = b8 - 4 →
  b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9 = 490 :=
by
  sorry

end NUMINAMATH_GPT_total_books_l1527_152726


namespace NUMINAMATH_GPT_inequality_of_cubic_powers_l1527_152703

theorem inequality_of_cubic_powers 
  (a b: ℝ) (h : a ≠ 0 ∧ b ≠ 0) 
  (h_cond : a * |a| > b * |b|) : 
  a^3 > b^3 := by
  sorry

end NUMINAMATH_GPT_inequality_of_cubic_powers_l1527_152703


namespace NUMINAMATH_GPT_smallest_x_exists_l1527_152709

theorem smallest_x_exists {M : ℤ} (h : 2520 = 2^3 * 3^2 * 5 * 7) : 
  ∃ x : ℕ, 2520 * x = M^3 ∧ x = 3675 := 
by {
  sorry
}

end NUMINAMATH_GPT_smallest_x_exists_l1527_152709


namespace NUMINAMATH_GPT_total_pay_is_186_l1527_152701

-- Define the conditions
def regular_rate : ℕ := 3 -- dollars per hour
def regular_hours : ℕ := 40 -- hours
def overtime_rate_multiplier : ℕ := 2
def overtime_hours : ℕ := 11

-- Calculate the regular pay
def regular_pay : ℕ := regular_hours * regular_rate

-- Calculate the overtime pay
def overtime_rate : ℕ := regular_rate * overtime_rate_multiplier
def overtime_pay : ℕ := overtime_hours * overtime_rate

-- Calculate the total pay
def total_pay : ℕ := regular_pay + overtime_pay

-- The statement to be proved
theorem total_pay_is_186 : total_pay = 186 :=
by 
  sorry

end NUMINAMATH_GPT_total_pay_is_186_l1527_152701


namespace NUMINAMATH_GPT_arithmetic_sequence_sums_l1527_152738

variable (a : ℕ → ℕ)

-- Conditions
def condition1 := a 1 + a 4 + a 7 = 39
def condition2 := a 2 + a 5 + a 8 = 33

-- Question and expected answer
def result := a 3 + a 6 + a 9 = 27

theorem arithmetic_sequence_sums (h1 : condition1 a) (h2 : condition2 a) : result a := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sums_l1527_152738


namespace NUMINAMATH_GPT_complement_of_log_set_l1527_152707

-- Define the set A based on the logarithmic inequality condition
def A : Set ℝ := { x : ℝ | Real.log x / Real.log (1 / 2) ≥ 2 }

-- Define the complement of A in the real numbers
noncomputable def complement_A : Set ℝ := { x : ℝ | x ≤ 0 } ∪ { x : ℝ | x > 1 / 4 }

-- The goal is to prove the equivalence
theorem complement_of_log_set :
  complement_A = { x : ℝ | x ≤ 0 } ∪ { x : ℝ | x > 1 / 4 } :=
by
  sorry

end NUMINAMATH_GPT_complement_of_log_set_l1527_152707


namespace NUMINAMATH_GPT_total_donation_l1527_152787

-- Definitions
def cassandra_pennies : ℕ := 5000
def james_deficit : ℕ := 276
def james_pennies : ℕ := cassandra_pennies - james_deficit

-- Theorem to prove the total donation
theorem total_donation : cassandra_pennies + james_pennies = 9724 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_total_donation_l1527_152787


namespace NUMINAMATH_GPT_multiplication_correct_l1527_152792

theorem multiplication_correct (a b c d e f: ℤ) (h₁: a * b = c) (h₂: d * e = f): 
    (63 * 14 = c) → (68 * 14 = f) → c = 882 ∧ f = 952 :=
by sorry

end NUMINAMATH_GPT_multiplication_correct_l1527_152792


namespace NUMINAMATH_GPT_sum_of_three_integers_l1527_152748

theorem sum_of_three_integers :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a * b * c = 125 ∧ a + b + c = 31 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_integers_l1527_152748


namespace NUMINAMATH_GPT_ratio_of_earnings_l1527_152759

theorem ratio_of_earnings (K V S : ℕ) (h1 : K + 30 = V) (h2 : V = 84) (h3 : S = 216) : S / K = 4 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_ratio_of_earnings_l1527_152759


namespace NUMINAMATH_GPT_part_a_part_b_l1527_152783

variable {f : ℝ → ℝ} 

-- Given conditions
axiom condition1 (x y : ℝ) : f (x + y) + 1 = f x + f y
axiom condition2 : f (1/2) = 0
axiom condition3 (x : ℝ) : x > 1/2 → f x < 0

-- Part (a)
theorem part_a (x : ℝ) : f x = 1/2 + 1/2 * f (2 * x) :=
sorry

-- Part (b)
theorem part_b (n : ℕ) (hn : n > 0) (x : ℝ) 
  (hx : 1 / 2^(n + 1) ≤ x ∧ x ≤ 1 / 2^n) : f x ≤ 1 - 1 / 2^n :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l1527_152783


namespace NUMINAMATH_GPT_Karlson_cannot_prevent_Baby_getting_one_fourth_l1527_152794

theorem Karlson_cannot_prevent_Baby_getting_one_fourth 
  (a : ℝ) (h : a > 0) (K : ℝ × ℝ) (hK : 0 < K.1 ∧ K.1 < a ∧ 0 < K.2 ∧ K.2 < a) :
  ∀ (O : ℝ × ℝ) (cut1 cut2 : ℝ), 
    ((O.1 = a/2) ∧ (O.2 = a/2) ∧ (cut1 = K.1 ∧ cut1 = a ∨ cut1 = K.2 ∧ cut1 = a) ∧ 
                             (cut2 = K.1 ∧ cut2 = a ∨ cut2 = K.2 ∧ cut2 = a)) →
  ∃ (piece : ℝ), piece ≥ a^2 / 4 :=
by
  sorry

end NUMINAMATH_GPT_Karlson_cannot_prevent_Baby_getting_one_fourth_l1527_152794


namespace NUMINAMATH_GPT_integer_solutions_of_inequality_system_l1527_152799

theorem integer_solutions_of_inequality_system :
  { x : ℤ | (3 * x - 2) / 3 ≥ 1 ∧ 3 * x + 5 > 4 * x - 2 } = {2, 3, 4, 5, 6} :=
by {
  sorry
}

end NUMINAMATH_GPT_integer_solutions_of_inequality_system_l1527_152799


namespace NUMINAMATH_GPT_sphere_surface_area_l1527_152786

theorem sphere_surface_area (r : ℝ) (h : π * r^2 = 81 * π) : 4 * π * r^2 = 324 * π :=
  sorry

end NUMINAMATH_GPT_sphere_surface_area_l1527_152786


namespace NUMINAMATH_GPT_toby_initial_photos_l1527_152768

-- Defining the problem conditions and proving the initial number of photos Toby had.
theorem toby_initial_photos (X : ℕ) 
  (h1 : ∃ n, X = n - 7) 
  (h2 : ∃ m, m = (n - 7) + 15) 
  (h3 : ∃ k, k = m) 
  (h4 : (k - 3) = 84) 
  : X = 79 :=
sorry

end NUMINAMATH_GPT_toby_initial_photos_l1527_152768


namespace NUMINAMATH_GPT_tan_alpha_eq_4_over_3_expression_value_eq_4_l1527_152782

-- Conditions
variable (α : ℝ) (hα1 : 0 < α) (hα2 : α < (Real.pi / 2)) (h_sin : Real.sin α = 4 / 5)

-- Prove: tan α = 4 / 3
theorem tan_alpha_eq_4_over_3 : Real.tan α = 4 / 3 :=
by
  sorry

-- Prove: the value of the given expression is 4
theorem expression_value_eq_4 : 
  (Real.sin (α + Real.pi) - 2 * Real.cos ((Real.pi / 2) + α)) / 
  (- Real.sin (-α) + Real.cos (Real.pi + α)) = 4 :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_eq_4_over_3_expression_value_eq_4_l1527_152782


namespace NUMINAMATH_GPT_max_roses_purchasable_l1527_152745

theorem max_roses_purchasable 
  (price_individual : ℝ) (price_dozen : ℝ) (price_two_dozen : ℝ) (price_five_dozen : ℝ) 
  (discount_threshold : ℕ) (discount_rate : ℝ) (total_money : ℝ) : 
  (price_individual = 4.50) →
  (price_dozen = 36) →
  (price_two_dozen = 50) →
  (price_five_dozen = 110) →
  (discount_threshold = 36) →
  (discount_rate = 0.10) →
  (total_money = 680) →
  ∃ (roses : ℕ), roses = 364 :=
by
  -- Definitions based on conditions
  intros
  -- The proof steps have been omitted for brevity
  sorry

end NUMINAMATH_GPT_max_roses_purchasable_l1527_152745


namespace NUMINAMATH_GPT_billy_distance_l1527_152708

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem billy_distance :
  distance 0 0 (7 + 4 * Real.sqrt 2) (4 * (Real.sqrt 2 + 1)) = Real.sqrt (129 + 88 * Real.sqrt 2) :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_billy_distance_l1527_152708


namespace NUMINAMATH_GPT_find_int_less_than_neg3_l1527_152762

theorem find_int_less_than_neg3 : 
  ∃ x ∈ ({-4, -2, 0, 3} : Set Int), x < -3 ∧ x = -4 := 
by
  -- formal proof goes here
  sorry

end NUMINAMATH_GPT_find_int_less_than_neg3_l1527_152762


namespace NUMINAMATH_GPT_spaceship_journey_time_l1527_152773

theorem spaceship_journey_time
  (initial_travel_1 : ℕ)
  (first_break : ℕ)
  (initial_travel_2 : ℕ)
  (second_break : ℕ)
  (travel_per_segment : ℕ)
  (break_per_segment : ℕ)
  (total_break_time : ℕ)
  (remaining_break_time : ℕ)
  (num_segments : ℕ)
  (total_travel_time : ℕ)
  (total_time : ℕ) :
  initial_travel_1 = 10 →
  first_break = 3 →
  initial_travel_2 = 10 →
  second_break = 1 →
  travel_per_segment = 11 →
  break_per_segment = 1 →
  total_break_time = 8 →
  remaining_break_time = total_break_time - (first_break + second_break) →
  num_segments = remaining_break_time / break_per_segment →
  total_travel_time = initial_travel_1 + initial_travel_2 + (num_segments * travel_per_segment) →
  total_time = total_travel_time + total_break_time →
  total_time = 72 :=
by
  intros
  sorry

end NUMINAMATH_GPT_spaceship_journey_time_l1527_152773


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_and_mean_l1527_152751

theorem arithmetic_sequence_sum_and_mean :
  let a1 := 1
  let d := 2
  let an := 21
  let n := 11
  let S := (n / 2) * (a1 + an)
  S = 121 ∧ (S / n) = 11 :=
by
  let a1 := 1
  let d := 2
  let an := 21
  let n := 11
  let S := (n / 2) * (a1 + an)
  have h1 : S = 121 := sorry
  have h2 : (S / n) = 11 := by
    rw [h1]
    exact sorry
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_arithmetic_sequence_sum_and_mean_l1527_152751


namespace NUMINAMATH_GPT_friend_balloons_count_l1527_152729

-- Definitions of the conditions
def balloons_you_have : ℕ := 7
def balloons_difference : ℕ := 2

-- Proof problem statement
theorem friend_balloons_count : (balloons_you_have - balloons_difference) = 5 :=
by
  sorry

end NUMINAMATH_GPT_friend_balloons_count_l1527_152729


namespace NUMINAMATH_GPT_factor_expression_l1527_152732

theorem factor_expression (y : ℝ) : 49 - 16*y^2 + 8*y = (7 - 4*y)*(7 + 4*y) := 
sorry

end NUMINAMATH_GPT_factor_expression_l1527_152732


namespace NUMINAMATH_GPT_inequality_proof_l1527_152737

theorem inequality_proof (a b x y : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : x / a < y / b) :
  (1 / 2) * (x / a + y / b) > (x + y) / (a + b) := by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1527_152737


namespace NUMINAMATH_GPT_border_pieces_is_75_l1527_152781

-- Definitions based on conditions
def total_pieces : Nat := 500
def trevor_pieces : Nat := 105
def joe_pieces : Nat := 3 * trevor_pieces
def missing_pieces : Nat := 5

-- Number of border pieces
def border_pieces : Nat := total_pieces - missing_pieces - (trevor_pieces + joe_pieces)

-- Theorem statement
theorem border_pieces_is_75 : border_pieces = 75 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_border_pieces_is_75_l1527_152781


namespace NUMINAMATH_GPT_intersection_closure_M_and_N_l1527_152756

noncomputable def set_M : Set ℝ :=
  { x | 2 / x < 1 }

noncomputable def closure_M : Set ℝ :=
  Set.Icc 0 2

noncomputable def set_N : Set ℝ :=
  { y | ∃ x, y = Real.sqrt (x - 1) }

theorem intersection_closure_M_and_N :
  (closure_M ∩ set_N) = Set.Icc 0 2 :=
by
  sorry

end NUMINAMATH_GPT_intersection_closure_M_and_N_l1527_152756


namespace NUMINAMATH_GPT_flour_more_than_sugar_l1527_152711

/-
  Mary is baking a cake. The recipe calls for 6 cups of sugar and 9 cups of flour. 
  She already put in 2 cups of flour. 
  Prove that the number of additional cups of flour Mary needs is 1 more than the number of additional cups of sugar she needs.
-/

theorem flour_more_than_sugar (s f a : ℕ) (h_s : s = 6) (h_f : f = 9) (h_a : a = 2) :
  (f - a) - s = 1 :=
by
  sorry

end NUMINAMATH_GPT_flour_more_than_sugar_l1527_152711


namespace NUMINAMATH_GPT_john_father_age_difference_l1527_152757

theorem john_father_age_difference (J F X : ℕ) (h1 : J + F = 77) (h2 : J = 15) (h3 : F = 2 * J + X) : X = 32 :=
by
  -- Adding the "sory" to skip the proof
  sorry

end NUMINAMATH_GPT_john_father_age_difference_l1527_152757


namespace NUMINAMATH_GPT_hiker_total_distance_l1527_152788

def hiker_distance (day1_hours day1_speed day2_speed : ℕ) : ℕ :=
  let day2_hours := day1_hours - 1
  let day3_hours := day1_hours
  (day1_hours * day1_speed) + (day2_hours * day2_speed) + (day3_hours * day2_speed)

theorem hiker_total_distance :
  hiker_distance 6 3 4 = 62 := 
by 
  sorry

end NUMINAMATH_GPT_hiker_total_distance_l1527_152788


namespace NUMINAMATH_GPT_angle_sum_property_l1527_152715

theorem angle_sum_property
  (angle1 angle2 angle3 : ℝ) 
  (h1 : angle1 = 58) 
  (h2 : angle2 = 35) 
  (h3 : angle3 = 42) : 
  angle1 + angle2 + angle3 + (180 - (angle1 + angle2 + angle3)) = 180 := 
by 
  sorry

end NUMINAMATH_GPT_angle_sum_property_l1527_152715


namespace NUMINAMATH_GPT_cube_volume_in_pyramid_l1527_152750

-- Definition for the conditions and parameters of the problem
def pyramid_condition (base_length : ℝ) (triangle_side : ℝ) : Prop :=
  base_length = 2 ∧ triangle_side = 2 * Real.sqrt 2

-- Definition for the cube's placement and side length condition inside the pyramid
def cube_side_length (s : ℝ) : Prop :=
  s = (Real.sqrt 6 / 3)

-- The final Lean statement proving the volume of the cube
theorem cube_volume_in_pyramid (base_length triangle_side s : ℝ) 
  (h_base_length : base_length = 2)
  (h_triangle_side : triangle_side = 2 * Real.sqrt 2)
  (h_cube_side_length : s = (Real.sqrt 6 / 3)) :
  (s ^ 3) = (2 * Real.sqrt 6 / 9) := 
by
  -- Using the given conditions to assert the conclusion
  rw [h_cube_side_length]
  have : (Real.sqrt 6 / 3) ^ 3 = 2 * Real.sqrt 6 / 9 := sorry
  exact this

end NUMINAMATH_GPT_cube_volume_in_pyramid_l1527_152750


namespace NUMINAMATH_GPT_perpendicular_lines_a_eq_2_l1527_152775

/-- Given two lines, ax + 2y + 2 = 0 and x - y - 2 = 0, prove that if these lines are perpendicular, then a = 2. -/
theorem perpendicular_lines_a_eq_2 {a : ℝ} :
  (∃ a, (a ≠ 0)) → (∃ x y, ((ax + 2*y + 2 = 0) ∧ (x - y - 2 = 0)) → - (a / 2) * 1 = -1) → a = 2 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_a_eq_2_l1527_152775


namespace NUMINAMATH_GPT_gcd_98_140_245_l1527_152716

theorem gcd_98_140_245 : Nat.gcd (Nat.gcd 98 140) 245 = 7 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_98_140_245_l1527_152716


namespace NUMINAMATH_GPT_inequality_holds_iff_b_lt_a_l1527_152746

theorem inequality_holds_iff_b_lt_a (a b : ℝ) :
  (∀ x : ℝ, (a + 1) * x^2 + a * x + a > b * (x^2 + x + 1)) ↔ b < a :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_iff_b_lt_a_l1527_152746


namespace NUMINAMATH_GPT_cost_price_of_each_watch_l1527_152725

-- Define the given conditions.
def sold_at_loss (C : ℝ) := 0.925 * C
def total_transaction_price (C : ℝ) := 3 * C * 1.053
def sold_for_more (C : ℝ) := 0.925 * C + 265

-- State the theorem to prove the cost price of each watch.
theorem cost_price_of_each_watch (C : ℝ) :
  3 * sold_for_more C = total_transaction_price C → C = 2070.31 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_cost_price_of_each_watch_l1527_152725


namespace NUMINAMATH_GPT_equation_of_perpendicular_line_l1527_152796

theorem equation_of_perpendicular_line :
  ∃ (a b c : ℝ), (5, 3) ∈ {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}
  ∧ (a = 2 ∧ b = 1 ∧ c = -13)
  ∧ (a * 1 + b * (-2) = 0) :=
sorry

end NUMINAMATH_GPT_equation_of_perpendicular_line_l1527_152796


namespace NUMINAMATH_GPT_number_of_birds_is_400_l1527_152722

-- Definitions of the problem
def num_stones : ℕ := 40
def num_trees : ℕ := 3 * num_stones + num_stones
def combined_trees_stones : ℕ := num_trees + num_stones
def num_birds : ℕ := 2 * combined_trees_stones

-- Statement to prove
theorem number_of_birds_is_400 : num_birds = 400 := by
  sorry

end NUMINAMATH_GPT_number_of_birds_is_400_l1527_152722


namespace NUMINAMATH_GPT_function_decomposition_l1527_152728

open Real

noncomputable def f (x : ℝ) : ℝ := log (10^x + 1)
noncomputable def g (x : ℝ) : ℝ := x / 2
noncomputable def h (x : ℝ) : ℝ := log (10^x + 1) - x / 2

theorem function_decomposition :
  ∀ x : ℝ, f x = g x + h x ∧ (∀ x, g (-x) = -g x) ∧ (∀ x, h (-x) = h x) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_function_decomposition_l1527_152728


namespace NUMINAMATH_GPT_smallest_x_value_l1527_152770

theorem smallest_x_value : ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ (3 : ℚ) / 4 = y / (250 + x) ∧ x = 2 := by
  sorry

end NUMINAMATH_GPT_smallest_x_value_l1527_152770


namespace NUMINAMATH_GPT_primes_between_30_and_50_l1527_152724

theorem primes_between_30_and_50 : (Finset.card (Finset.filter Nat.Prime (Finset.Ico 30 51))) = 5 :=
by
  sorry

end NUMINAMATH_GPT_primes_between_30_and_50_l1527_152724


namespace NUMINAMATH_GPT_remainder_product_div_17_l1527_152702

theorem remainder_product_div_17 :
  (2357 ≡ 6 [MOD 17]) → (2369 ≡ 4 [MOD 17]) → (2384 ≡ 0 [MOD 17]) →
  (2391 ≡ 9 [MOD 17]) → (3017 ≡ 9 [MOD 17]) → (3079 ≡ 0 [MOD 17]) →
  (3082 ≡ 3 [MOD 17]) →
  ((2357 * 2369 * 2384 * 2391) * (3017 * 3079 * 3082) ≡ 0 [MOD 17]) :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_remainder_product_div_17_l1527_152702


namespace NUMINAMATH_GPT_prove_a_lt_neg_one_l1527_152791

variable {f : ℝ → ℝ} (a : ℝ)

-- Conditions:
-- 1. f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- 2. f has a period of 3
def has_period_three (f : ℝ → ℝ) : Prop := ∀ x, f (x + 3) = f x

-- 3. f(1) > 1
def f_one_gt_one (f : ℝ → ℝ) : Prop := f 1 > 1

-- 4. f(2) = a
def f_two_eq_a (f : ℝ → ℝ) (a : ℝ) : Prop := f 2 = a

-- Proof statement:
theorem prove_a_lt_neg_one (h1 : is_odd_function f) (h2 : has_period_three f)
  (h3 : f_one_gt_one f) (h4 : f_two_eq_a f a) : a < -1 :=
  sorry

end NUMINAMATH_GPT_prove_a_lt_neg_one_l1527_152791


namespace NUMINAMATH_GPT_xy_value_l1527_152710

theorem xy_value (x y : ℝ) (h : (|x| - 1)^2 + (2 * y + 1)^2 = 0) : xy = 1/2 ∨ xy = -1/2 :=
by {
  sorry
}

end NUMINAMATH_GPT_xy_value_l1527_152710


namespace NUMINAMATH_GPT_force_required_for_bolt_b_20_inch_l1527_152755

noncomputable def force_inversely_proportional (F L : ℝ) : ℝ := F * L

theorem force_required_for_bolt_b_20_inch (F L : ℝ) :
  let handle_length_10 := 10
  let force_length_product_bolt_a := 3000
  let force_length_product_bolt_b := 4000
  let new_handle_length := 20
  (F * handle_length_10 = 400)
  ∧ (F * new_handle_length = 200)
  → force_inversely_proportional 400 10 = 4000
  ∧ force_inversely_proportional 200 20 = 4000
:=
by
  sorry

end NUMINAMATH_GPT_force_required_for_bolt_b_20_inch_l1527_152755


namespace NUMINAMATH_GPT_hyperbola_triangle_area_l1527_152734

/-- The relationship between the hyperbola's asymptotes, tangent, and area proportion -/
theorem hyperbola_triangle_area (a b x0 y0 : ℝ) 
  (h_asymptote1 : ∀ x, y = (b / a) * x)
  (h_asymptote2 : ∀ x, y = -(b / a) * x)
  (h_tangent    : ∀ x y, (x0 * x) / (a ^ 2) - (y0 * y) / (b ^ 2) = 1)
  (h_condition  : (x0 ^ 2) * (a ^ 2) - (y0 ^ 2) * (b ^ 2) = (a ^ 2) * (b ^ 2)) :
  ∃ k : ℝ, k = a ^ 4 :=
sorry

end NUMINAMATH_GPT_hyperbola_triangle_area_l1527_152734


namespace NUMINAMATH_GPT_Alyosha_result_divisible_by_S_l1527_152790

variable (a b S x y : ℤ)
variable (h1 : x + y = S)
variable (h2 : S ∣ a * x + b * y)

theorem Alyosha_result_divisible_by_S :
  S ∣ b * x + a * y :=
sorry

end NUMINAMATH_GPT_Alyosha_result_divisible_by_S_l1527_152790


namespace NUMINAMATH_GPT_percent_of_x_is_y_l1527_152749

theorem percent_of_x_is_y (x y : ℝ) (h : 0.60 * (x - y) = 0.30 * (x + y)) : y = 0.3333 * x :=
by
  sorry

end NUMINAMATH_GPT_percent_of_x_is_y_l1527_152749


namespace NUMINAMATH_GPT_sum_of_numbers_is_216_l1527_152704

-- Define the conditions and what needs to be proved.
theorem sum_of_numbers_is_216 
  (x : ℕ) 
  (h_lcm : Nat.lcm (2 * x) (Nat.lcm (3 * x) (7 * x)) = 126) : 
  2 * x + 3 * x + 7 * x = 216 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_is_216_l1527_152704


namespace NUMINAMATH_GPT_volume_calc_l1527_152714

noncomputable
def volume_of_open_box {l w : ℕ} (sheet_length : l = 48) (sheet_width : w = 38) (cut_length : ℕ) (cut_length_eq : cut_length = 8) : ℕ :=
  let new_length := l - 2 * cut_length
  let new_width := w - 2 * cut_length
  let height := cut_length
  new_length * new_width * height

theorem volume_calc : volume_of_open_box (sheet_length := rfl) (sheet_width := rfl) (cut_length := 8) (cut_length_eq := rfl) = 5632 :=
sorry

end NUMINAMATH_GPT_volume_calc_l1527_152714


namespace NUMINAMATH_GPT_tiled_floor_area_correct_garden_area_correct_seating_area_correct_l1527_152743

noncomputable def length_room : ℝ := 20
noncomputable def width_room : ℝ := 12
noncomputable def width_veranda : ℝ := 2
noncomputable def length_pool : ℝ := 15
noncomputable def width_pool : ℝ := 6

noncomputable def area (length width : ℝ) : ℝ := length * width

noncomputable def area_room : ℝ := area length_room width_room
noncomputable def area_pool : ℝ := area length_pool width_pool
noncomputable def area_tiled_floor : ℝ := area_room - area_pool

noncomputable def total_length : ℝ := length_room + 2 * width_veranda
noncomputable def total_width : ℝ := width_room + 2 * width_veranda
noncomputable def area_total : ℝ := area total_length total_width
noncomputable def area_veranda : ℝ := area_total - area_room
noncomputable def area_garden : ℝ := area_veranda / 2
noncomputable def area_seating : ℝ := area_veranda / 2

theorem tiled_floor_area_correct : area_tiled_floor = 150 := by
  sorry

theorem garden_area_correct : area_garden = 72 := by
  sorry

theorem seating_area_correct : area_seating = 72 := by
  sorry

end NUMINAMATH_GPT_tiled_floor_area_correct_garden_area_correct_seating_area_correct_l1527_152743


namespace NUMINAMATH_GPT_repeating_decimals_sum_is_fraction_l1527_152731

-- Define the repeating decimals as fractions
def x : ℚ := 1 / 3
def y : ℚ := 2 / 99

-- Define the sum of the repeating decimals
def sum := x + y

-- State the theorem
theorem repeating_decimals_sum_is_fraction :
  sum = 35 / 99 := sorry

end NUMINAMATH_GPT_repeating_decimals_sum_is_fraction_l1527_152731


namespace NUMINAMATH_GPT_solve_abs_inequality_l1527_152763

theorem solve_abs_inequality (x : ℝ) :
  abs ((6 - 2 * x + 5) / 4) < 3 ↔ -1 / 2 < x ∧ x < 23 / 2 := 
sorry

end NUMINAMATH_GPT_solve_abs_inequality_l1527_152763


namespace NUMINAMATH_GPT_jewelry_store_total_cost_l1527_152797

-- Definitions for given conditions
def necklace_capacity : Nat := 12
def current_necklaces : Nat := 5
def ring_capacity : Nat := 30
def current_rings : Nat := 18
def bracelet_capacity : Nat := 15
def current_bracelets : Nat := 8

def necklace_cost : Nat := 4
def ring_cost : Nat := 10
def bracelet_cost : Nat := 5

-- Definition for number of items needed to fill displays
def needed_necklaces : Nat := necklace_capacity - current_necklaces
def needed_rings : Nat := ring_capacity - current_rings
def needed_bracelets : Nat := bracelet_capacity - current_bracelets

-- Definition for cost to fill each type of jewelry
def cost_necklaces : Nat := needed_necklaces * necklace_cost
def cost_rings : Nat := needed_rings * ring_cost
def cost_bracelets : Nat := needed_bracelets * bracelet_cost

-- Total cost to fill the displays
def total_cost : Nat := cost_necklaces + cost_rings + cost_bracelets

-- Proof statement
theorem jewelry_store_total_cost : total_cost = 183 := by
  sorry

end NUMINAMATH_GPT_jewelry_store_total_cost_l1527_152797


namespace NUMINAMATH_GPT_sum_of_arithmetic_sequence_l1527_152719

-- Given conditions in the problem
axiom arithmetic_sequence (a : ℕ → ℤ): Prop
axiom are_roots (a b : ℤ): ∃ p q : ℤ, p * q = -5 ∧ p + q = 3 ∧ (a = p ∨ a = q) ∧ (b = p ∨ b = q)

-- The equivalent proof problem statement
theorem sum_of_arithmetic_sequence (a : ℕ → ℤ) 
  (h1 : arithmetic_sequence a)
  (h2 : ∃ p q : ℤ, p * q = -5 ∧ p + q = 3 ∧ (a 2 = p ∨ a 2 = q) ∧ (a 11 = p ∨ a 11 = q)):
  a 5 + a 8 = 3 :=
sorry

end NUMINAMATH_GPT_sum_of_arithmetic_sequence_l1527_152719


namespace NUMINAMATH_GPT_dogs_left_l1527_152721

-- Define the conditions
def total_dogs : ℕ := 50
def dog_houses : ℕ := 17

-- Statement to prove the number of dogs left
theorem dogs_left : (total_dogs % dog_houses) = 16 :=
by sorry

end NUMINAMATH_GPT_dogs_left_l1527_152721


namespace NUMINAMATH_GPT_evaluate_g_at_neg_four_l1527_152765

def g (x : ℤ) : ℤ := 5 * x + 2

theorem evaluate_g_at_neg_four : g (-4) = -18 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_g_at_neg_four_l1527_152765


namespace NUMINAMATH_GPT_final_answer_for_m_l1527_152774

noncomputable def proof_condition_1 (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

noncomputable def proof_condition_2 (x y : ℝ) : Prop :=
  x + 2*y - 3 = 0

noncomputable def proof_condition_perpendicular (x1 y1 x2 y2 : ℝ) : Prop :=
  x1*x2 + y1*y2 = 0

theorem final_answer_for_m :
  (∀ (x y m : ℝ), proof_condition_1 x y m) →
  (∀ (x y : ℝ), proof_condition_2 x y) →
  (∀ (x1 y1 x2 y2 : ℝ), proof_condition_perpendicular x1 y1 x2 y2) →
  m = 12 / 5 :=
sorry

end NUMINAMATH_GPT_final_answer_for_m_l1527_152774


namespace NUMINAMATH_GPT_george_and_hannah_received_A_grades_l1527_152747

-- Define students as propositions
variables (Elena Fred George Hannah : Prop)

-- Define the conditions
def condition1 : Prop := Elena → Fred
def condition2 : Prop := Fred → George
def condition3 : Prop := George → Hannah
def condition4 : Prop := ∃ A1 A2 : Prop, A1 ∧ A2 ∧ (A1 ≠ A2) ∧ (A1 = George ∨ A1 = Hannah) ∧ (A2 = George ∨ A2 = Hannah)

-- The theorem to be proven: George and Hannah received A grades
theorem george_and_hannah_received_A_grades :
  condition1 Elena Fred →
  condition2 Fred George →
  condition3 George Hannah →
  condition4 George Hannah :=
by
  sorry

end NUMINAMATH_GPT_george_and_hannah_received_A_grades_l1527_152747


namespace NUMINAMATH_GPT_sum_of_first_10_terms_l1527_152713

noncomputable def a (n : ℕ) : ℕ := sorry
noncomputable def S (n : ℕ) : ℕ := sorry

variable {n : ℕ}

-- Conditions
axiom h1 : ∀ n, S (n + 1) = S n + a n + 3
axiom h2 : a 5 + a 6 = 29

-- Statement to prove
theorem sum_of_first_10_terms : S 10 = 145 := 
sorry

end NUMINAMATH_GPT_sum_of_first_10_terms_l1527_152713


namespace NUMINAMATH_GPT_cos_double_angle_l1527_152730

variable (α : ℝ)

theorem cos_double_angle (h1 : 0 < α ∧ α < π / 2) 
                         (h2 : Real.cos ( α + π / 4) = 3 / 5) : 
    Real.cos (2 * α) = 24 / 25 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l1527_152730


namespace NUMINAMATH_GPT_correct_statement_l1527_152780

def is_accurate_to (value : ℝ) (place : ℝ) : Prop :=
  ∃ k : ℤ, value = k * place

def statement_A : Prop := is_accurate_to 51000 0.1
def statement_B : Prop := is_accurate_to 0.02 1
def statement_C : Prop := (2.8 = 2.80)
def statement_D : Prop := is_accurate_to (2.3 * 10^4) 1000

theorem correct_statement : statement_D :=
by
  sorry

end NUMINAMATH_GPT_correct_statement_l1527_152780


namespace NUMINAMATH_GPT_student_answered_two_questions_incorrectly_l1527_152769

/-
  Defining the variables and conditions for the problem.
  x: number of questions answered correctly,
  y: number of questions not answered,
  z: number of questions answered incorrectly.
-/

theorem student_answered_two_questions_incorrectly (x y z : ℕ) 
  (h1 : x + y + z = 6) 
  (h2 : 8 * x + 2 * y = 20) : z = 2 :=
by
  /- We know the total number of questions is 6.
     And the total score is 20 with the given scoring rules.
     Thus, we need to prove that z = 2 under these conditions. -/
  sorry

end NUMINAMATH_GPT_student_answered_two_questions_incorrectly_l1527_152769


namespace NUMINAMATH_GPT_total_units_l1527_152772

theorem total_units (A B C: ℕ) (hA: A = 2 + 4 + 6 + 8 + 10 + 12) (hB: B = A) (hC: C = 3 + 5 + 7 + 9) : 
  A + B + C = 108 := 
sorry

end NUMINAMATH_GPT_total_units_l1527_152772


namespace NUMINAMATH_GPT_proof_problem_l1527_152767

open Real

theorem proof_problem 
  (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (h_sum : a + b + c + d = 1)
  : (b * c * d / (1 - a)^2) + (c * d * a / (1 - b)^2) + (d * a * b / (1 - c)^2) + (a * b * c / (1 - d)^2) ≤ 1 / 9 := 
   sorry

end NUMINAMATH_GPT_proof_problem_l1527_152767


namespace NUMINAMATH_GPT_largest_integer_base7_four_digits_l1527_152764

theorem largest_integer_base7_four_digits :
  ∃ M : ℕ, (∀ m : ℕ, 7^3 ≤ m^2 ∧ m^2 < 7^4 → m ≤ M) ∧ M = 48 :=
sorry

end NUMINAMATH_GPT_largest_integer_base7_four_digits_l1527_152764


namespace NUMINAMATH_GPT_percent_exceed_l1527_152766

theorem percent_exceed (x y : ℝ) (h : x = 0.75 * y) : ((y - x) / x) * 100 = 33.33 :=
by
  sorry

end NUMINAMATH_GPT_percent_exceed_l1527_152766


namespace NUMINAMATH_GPT_min_distance_l1527_152795

noncomputable def point_on_curve (x₁ y₁ : ℝ) : Prop :=
  y₁ = x₁^2 - Real.log x₁

noncomputable def point_on_line (x₂ y₂ : ℝ) : Prop :=
  x₂ - y₂ - 2 = 0

theorem min_distance 
  (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : point_on_curve x₁ y₁)
  (h₂ : point_on_line x₂ y₂) 
  : (x₂ - x₁)^2 + (y₂ - y₁)^2 = 2 :=
sorry

end NUMINAMATH_GPT_min_distance_l1527_152795


namespace NUMINAMATH_GPT_monthly_income_of_p_l1527_152742

theorem monthly_income_of_p (P Q R : ℕ) 
    (h1 : (P + Q) / 2 = 5050)
    (h2 : (Q + R) / 2 = 6250)
    (h3 : (P + R) / 2 = 5200) :
    P = 4000 :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_monthly_income_of_p_l1527_152742


namespace NUMINAMATH_GPT_wire_cut_perimeter_equal_l1527_152798

theorem wire_cut_perimeter_equal (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : 4 * (a / 4) = 8 * (b / 8)) :
  a / b = 1 :=
sorry

end NUMINAMATH_GPT_wire_cut_perimeter_equal_l1527_152798


namespace NUMINAMATH_GPT_greatest_m_value_l1527_152753

theorem greatest_m_value (x y z u : ℕ) (hx : x ≥ y) (h1 : x + y = z + u) (h2 : 2 * x * y = z * u) : 
  ∃ m, m = 3 + 2 * Real.sqrt 2 ∧ m ≤ x / y :=
sorry

end NUMINAMATH_GPT_greatest_m_value_l1527_152753


namespace NUMINAMATH_GPT_next_perfect_cube_l1527_152717

theorem next_perfect_cube (x : ℕ) (h : ∃ k : ℕ, x = k^3) : 
  ∃ m : ℕ, m^3 = x + 3 * (x^(1/3))^2 + 3 * x^(1/3) + 1 :=
by
  sorry

end NUMINAMATH_GPT_next_perfect_cube_l1527_152717


namespace NUMINAMATH_GPT_paint_snake_l1527_152723

theorem paint_snake (num_cubes : ℕ) (paint_per_cube : ℕ) (end_paint : ℕ) (total_paint : ℕ) 
  (h_cubes : num_cubes = 2016)
  (h_paint_per_cube : paint_per_cube = 60)
  (h_end_paint : end_paint = 20)
  (h_total_paint : total_paint = 121000) :
  total_paint = (num_cubes * paint_per_cube) + 2 * end_paint :=
by
  rw [h_cubes, h_paint_per_cube, h_end_paint]
  sorry

end NUMINAMATH_GPT_paint_snake_l1527_152723


namespace NUMINAMATH_GPT_equation_solutions_l1527_152778

noncomputable def count_solutions (a : ℝ) : ℕ :=
  if 0 < a ∧ a <= 1 ∨ a = Real.exp (1 / Real.exp 1) then 1
  else if 1 < a ∧ a < Real.exp (1 / Real.exp 1) then 2
  else if a > Real.exp (1 / Real.exp 1) then 0
  else 0

theorem equation_solutions (a : ℝ) (h₀ : 0 < a) :
  (∃! x : ℝ, a^x = x) ↔ count_solutions a = 1 ∨ count_solutions a = 2 ∨ count_solutions a = 0 := sorry

end NUMINAMATH_GPT_equation_solutions_l1527_152778


namespace NUMINAMATH_GPT_find_expression_value_l1527_152733

theorem find_expression_value : 1 + 2 * 3 - 4 + 5 = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_expression_value_l1527_152733


namespace NUMINAMATH_GPT_gcd_f_101_102_l1527_152793

def f (x : ℕ) : ℕ := x^2 + x + 2010

theorem gcd_f_101_102 : Nat.gcd (f 101) (f 102) = 12 := 
by sorry

end NUMINAMATH_GPT_gcd_f_101_102_l1527_152793


namespace NUMINAMATH_GPT_price_of_each_armchair_l1527_152705

theorem price_of_each_armchair
  (sofa_price : ℕ)
  (coffee_table_price : ℕ)
  (total_invoice : ℕ)
  (num_armchairs : ℕ)
  (h_sofa : sofa_price = 1250)
  (h_coffee_table : coffee_table_price = 330)
  (h_invoice : total_invoice = 2430)
  (h_num_armchairs : num_armchairs = 2) :
  (total_invoice - (sofa_price + coffee_table_price)) / num_armchairs = 425 := 
by 
  sorry

end NUMINAMATH_GPT_price_of_each_armchair_l1527_152705


namespace NUMINAMATH_GPT_correct_word_for_blank_l1527_152754

theorem correct_word_for_blank :
  (∀ (word : String), word = "that" ↔ word = "whoever" ∨ word = "someone" ∨ word = "that" ∨ word = "any") :=
by
  sorry

end NUMINAMATH_GPT_correct_word_for_blank_l1527_152754


namespace NUMINAMATH_GPT_rationalize_denominator_l1527_152727

theorem rationalize_denominator : (7 / Real.sqrt 147) = (Real.sqrt 3 / 3) :=
by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_l1527_152727


namespace NUMINAMATH_GPT_James_trout_pounds_l1527_152779

def pounds_trout (T : ℝ) : Prop :=
  let salmon := 1.5 * T
  let tuna := 2 * T
  T + salmon + tuna = 1100

theorem James_trout_pounds :
  ∃ T : ℝ, pounds_trout T ∧ T = 244 :=
sorry

end NUMINAMATH_GPT_James_trout_pounds_l1527_152779


namespace NUMINAMATH_GPT_sixteenth_term_l1527_152700

theorem sixteenth_term :
  (-1)^(16+1) * Real.sqrt (3 * (16 - 1)) = -3 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_GPT_sixteenth_term_l1527_152700


namespace NUMINAMATH_GPT_circular_garden_radius_l1527_152736

theorem circular_garden_radius (r : ℝ) (h1 : 2 * Real.pi * r = (1 / 6) * Real.pi * r^2) : r = 12 :=
by sorry

end NUMINAMATH_GPT_circular_garden_radius_l1527_152736


namespace NUMINAMATH_GPT_average_age_union_l1527_152784

theorem average_age_union
    (A B C : Set Person)
    (a b c : ℕ)
    (sum_A sum_B sum_C : ℝ)
    (h_disjoint_AB : Disjoint A B)
    (h_disjoint_AC : Disjoint A C)
    (h_disjoint_BC : Disjoint B C)
    (h_avg_A : sum_A / a = 40)
    (h_avg_B : sum_B / b = 25)
    (h_avg_C : sum_C / c = 35)
    (h_avg_AB : (sum_A + sum_B) / (a + b) = 33)
    (h_avg_AC : (sum_A + sum_C) / (a + c) = 37.5)
    (h_avg_BC : (sum_B + sum_C) / (b + c) = 30) :
  (sum_A + sum_B + sum_C) / (a + b + c) = 51.6 :=
sorry

end NUMINAMATH_GPT_average_age_union_l1527_152784


namespace NUMINAMATH_GPT_explicit_formula_for_sequence_l1527_152777

theorem explicit_formula_for_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (hSn : ∀ n, S n = 2 * a n + 1) : 
  ∀ n, a n = (-2) ^ (n - 1) := 
by
  sorry

end NUMINAMATH_GPT_explicit_formula_for_sequence_l1527_152777


namespace NUMINAMATH_GPT_factorization_problem_l1527_152760

theorem factorization_problem (a b : ℤ) : 
  (∀ y : ℤ, 2 * y^2 - 5 * y - 12 = (2 * y + a) * (y + b)) → a - b = 1 := 
by
  sorry

end NUMINAMATH_GPT_factorization_problem_l1527_152760


namespace NUMINAMATH_GPT_original_price_of_shoes_l1527_152712

-- Define the conditions.
def discount_rate : ℝ := 0.20
def amount_paid : ℝ := 480

-- Statement of the theorem.
theorem original_price_of_shoes (P : ℝ) (h₀ : P * (1 - discount_rate) = amount_paid) : 
  P = 600 :=
by
  sorry

end NUMINAMATH_GPT_original_price_of_shoes_l1527_152712


namespace NUMINAMATH_GPT_speed_ratio_l1527_152741

variables (H D : ℝ)
variables (duck_leaps hen_leaps : ℕ)
-- hen_leaps and duck_leaps denote the leaps taken by hen and duck respectively

-- conditions given
axiom cond1 : hen_leaps = 6 ∧ duck_leaps = 8
axiom cond2 : 4 * D = 3 * H

-- goal to prove
theorem speed_ratio (H D : ℝ) (hen_leaps duck_leaps : ℕ) (cond1 : hen_leaps = 6 ∧ duck_leaps = 8) (cond2 : 4 * D = 3 * H) : 
  (6 * H) = (8 * D) :=
by
  intros
  sorry

end NUMINAMATH_GPT_speed_ratio_l1527_152741


namespace NUMINAMATH_GPT_quadratic_completing_the_square_l1527_152735

theorem quadratic_completing_the_square :
  ∀ x : ℝ, x^2 - 4 * x - 2 = 0 → (x - 2)^2 = 6 :=
by sorry

end NUMINAMATH_GPT_quadratic_completing_the_square_l1527_152735


namespace NUMINAMATH_GPT_arithmetic_sequence_eightieth_term_l1527_152720

open BigOperators

def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

theorem arithmetic_sequence_eightieth_term :
  ∀ (d : ℝ),
  arithmetic_sequence 3 d 21 = 41 →
  arithmetic_sequence 3 d 80 = 153.1 :=
by
  intros
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_eightieth_term_l1527_152720
