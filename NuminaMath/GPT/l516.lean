import Mathlib

namespace NUMINAMATH_GPT_total_lunch_cost_l516_51644

/-- Janet, a third grade teacher, is picking up the sack lunch order from a local deli for 
the field trip she is taking her class on. There are 35 children in her class, 5 volunteer 
chaperones, and herself. She also ordered three additional sack lunches, just in case 
there was a problem. Each sack lunch costs $7. --/
theorem total_lunch_cost :
  let children := 35
  let chaperones := 5
  let janet := 1
  let additional_lunches := 3
  let price_per_lunch := 7
  let total_lunches := children + chaperones + janet + additional_lunches
  total_lunches * price_per_lunch = 308 :=
by
  sorry

end NUMINAMATH_GPT_total_lunch_cost_l516_51644


namespace NUMINAMATH_GPT_compute_sum_l516_51681

open BigOperators

theorem compute_sum : 
  (1 / 2 ^ 2010 : ℝ) * ∑ n in Finset.range 1006, (-3 : ℝ) ^ n * (Nat.choose 2010 (2 * n)) = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_compute_sum_l516_51681


namespace NUMINAMATH_GPT_probability_two_green_apples_l516_51679

noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_two_green_apples :
  ∀ (total_apples green_apples choose_apples : ℕ),
    total_apples = 7 →
    green_apples = 3 →
    choose_apples = 2 →
    (binom green_apples choose_apples : ℝ) / binom total_apples choose_apples = 1 / 7 :=
by
  intro total_apples green_apples choose_apples
  intro h_total h_green h_choose
  rw [h_total, h_green, h_choose]
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_probability_two_green_apples_l516_51679


namespace NUMINAMATH_GPT_identical_graphs_l516_51677

theorem identical_graphs :
  (∃ (b c : ℝ), (∀ (x y : ℝ), 3 * x + b * y + c = 0 ↔ c * x - 2 * y + 12 = 0) ∧
                 ((b, c) = (1, 6) ∨ (b, c) = (-1, -6))) → ∃ n : ℕ, n = 2 :=
by
  sorry

end NUMINAMATH_GPT_identical_graphs_l516_51677


namespace NUMINAMATH_GPT_sqrt_of_9_l516_51610

theorem sqrt_of_9 (x : ℝ) (h : x^2 = 9) : x = 3 ∨ x = -3 :=
by {
  sorry
}

end NUMINAMATH_GPT_sqrt_of_9_l516_51610


namespace NUMINAMATH_GPT_shorter_side_of_room_l516_51626

theorem shorter_side_of_room
  (P : ℕ) (A : ℕ) (a b : ℕ)
  (perimeter_eq : 2 * a + 2 * b = P)
  (area_eq : a * b = A) (partition_len : ℕ) (partition_cond : partition_len = 5)
  (room_perimeter : P = 60)
  (room_area : A = 200) :
  b = 10 := 
by
  sorry

end NUMINAMATH_GPT_shorter_side_of_room_l516_51626


namespace NUMINAMATH_GPT_roots_product_eq_l516_51671

theorem roots_product_eq
  (a b m p r : ℚ)
  (h₀ : a * b = 3)
  (h₁ : ∀ x, x^2 - m * x + 3 = 0 → (x = a ∨ x = b))
  (h₂ : ∀ x, x^2 - p * x + r = 0 → (x = a + 1 / b ∨ x = b + 1 / a)) : 
  r = 16 / 3 :=
by
  sorry

end NUMINAMATH_GPT_roots_product_eq_l516_51671


namespace NUMINAMATH_GPT_avg_age_10_students_l516_51609

-- Defining the given conditions
def avg_age_15_students : ℕ := 15
def total_students : ℕ := 15
def avg_age_4_students : ℕ := 14
def num_4_students : ℕ := 4
def age_15th_student : ℕ := 9

-- Calculating the total age based on given conditions
def total_age_15_students : ℕ := avg_age_15_students * total_students
def total_age_4_students : ℕ := avg_age_4_students * num_4_students
def total_age_10_students : ℕ := total_age_15_students - total_age_4_students - age_15th_student

-- Problem to be proved
theorem avg_age_10_students : total_age_10_students / 10 = 16 := 
by sorry

end NUMINAMATH_GPT_avg_age_10_students_l516_51609


namespace NUMINAMATH_GPT_linear_inequality_m_eq_zero_l516_51682

theorem linear_inequality_m_eq_zero (m : ℝ) (x : ℝ) : 
  ((m - 2) * x ^ |m - 1| - 3 > 6) → abs (m - 1) = 1 → m ≠ 2 → m = 0 := by
  intros h1 h2 h3
  -- Proof of m = 0 based on given conditions
  sorry

end NUMINAMATH_GPT_linear_inequality_m_eq_zero_l516_51682


namespace NUMINAMATH_GPT_cos_angle_value_l516_51632

noncomputable def cos_angle := Real.cos (19 * Real.pi / 4)

theorem cos_angle_value : cos_angle = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_GPT_cos_angle_value_l516_51632


namespace NUMINAMATH_GPT_solve_y_l516_51618

theorem solve_y : ∃ y : ℚ, 2 * y + 3 * y = 600 - (4 * y + 5 * y + 100) ∧ y = 250 / 7 := by
  sorry

end NUMINAMATH_GPT_solve_y_l516_51618


namespace NUMINAMATH_GPT_part1_part2_l516_51696

variables {A B C : ℝ} {a b c : ℝ} -- Angles and sides of the triangle
variable (h1 : (a - b + c) * (a - b - c) + a * b = 0)
variable (h2 : b * c * Real.sin C = 3 * c * Real.cos A + 3 * a * Real.cos C)

theorem part1 : c = 2 * Real.sqrt 3 :=
by
  sorry

theorem part2 : 6 < a + b ∧ a + b <= 4 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l516_51696


namespace NUMINAMATH_GPT_proof_1_proof_2_l516_51604

-- Definitions of propositions p, q, and r

def p (a : ℝ) : Prop :=
  ∀ x : ℝ, ¬ (x^2 + (a - 1) * x + a^2 ≤ 0)

def q (a : ℝ) : Prop :=
  2 * a^2 - a > 1

def r (a : ℝ) : Prop :=
  (2 * a - 1) / (a - 2) ≤ 1

-- The given proof problem statement 1
theorem proof_1 (a : ℝ) : (p a ∨ q a) ∧ ¬ (p a ∧ q a) → (a ∈ Set.Icc (-1) (-1/2) ∪ Set.Ioo (1/3) 1) :=
sorry

-- The given proof problem statement 2
theorem proof_2 (a : ℝ) : ¬ p a → r a :=
sorry

end NUMINAMATH_GPT_proof_1_proof_2_l516_51604


namespace NUMINAMATH_GPT_smallest_value_am_hm_inequality_l516_51656

theorem smallest_value_am_hm_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (b + d) + 1 / (c + d)) ≥ 8 :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_am_hm_inequality_l516_51656


namespace NUMINAMATH_GPT_refill_cost_calculation_l516_51640

variables (total_spent : ℕ) (refills : ℕ)

def one_refill_cost (total_spent refills : ℕ) : ℕ := total_spent / refills

theorem refill_cost_calculation (h1 : total_spent = 40) (h2 : refills = 4) :
  one_refill_cost total_spent refills = 10 :=
by
  sorry

end NUMINAMATH_GPT_refill_cost_calculation_l516_51640


namespace NUMINAMATH_GPT_sum_first_10_mod_8_is_7_l516_51627

-- Define the sum of the first 10 positive integers
def sum_first_10 : ℕ := 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10

-- Define the divisor
def divisor : ℕ := 8

-- Prove that the remainder of the sum of the first 10 positive integers divided by 8 is 7
theorem sum_first_10_mod_8_is_7 : sum_first_10 % divisor = 7 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_10_mod_8_is_7_l516_51627


namespace NUMINAMATH_GPT_percentage_rotten_bananas_l516_51631

theorem percentage_rotten_bananas :
  let total_oranges := 600
  let total_bananas := 400
  let rotten_oranges_percentage := 0.15
  let good_condition_percentage := 0.878
  let total_fruits := total_oranges + total_bananas 
  let rotten_oranges := rotten_oranges_percentage * total_oranges 
  let good_fruits := good_condition_percentage * total_fruits
  let rotten_fruits := total_fruits - good_fruits
  let rotten_bananas := rotten_fruits - rotten_oranges
  (rotten_bananas / total_bananas) * 100 = 8 := by
  {
    -- Calculations and simplifications go here
    sorry
  }

end NUMINAMATH_GPT_percentage_rotten_bananas_l516_51631


namespace NUMINAMATH_GPT_four_digit_multiples_of_7_l516_51690

theorem four_digit_multiples_of_7 : 
  let smallest_four_digit := 1000
  let largest_four_digit := 9999
  let smallest_multiple_of_7 := (Nat.ceil (smallest_four_digit / 7)) * 7
  let largest_multiple_of_7 := (Nat.floor (largest_four_digit / 7)) * 7
  let count_of_multiples := (Nat.floor (largest_four_digit / 7)) - (Nat.ceil (smallest_four_digit / 7)) + 1
  count_of_multiples = 1286 :=
by
  sorry

end NUMINAMATH_GPT_four_digit_multiples_of_7_l516_51690


namespace NUMINAMATH_GPT_prove_tan_sum_is_neg_sqrt3_l516_51651

open Real

-- Given conditions as definitions
def condition1 (α β : ℝ) : Prop := 0 < α ∧ α < π ∧ 0 < β ∧ β < π
def condition2 (α β : ℝ) : Prop := sin α + sin β = sqrt 3 * (cos α + cos β)

-- The statement of the proof
theorem prove_tan_sum_is_neg_sqrt3 (α β : ℝ) (h1 : condition1 α β) (h2 : condition2 α β) :
  tan (α + β) = -sqrt 3 :=
sorry

end NUMINAMATH_GPT_prove_tan_sum_is_neg_sqrt3_l516_51651


namespace NUMINAMATH_GPT_rhombus_area_l516_51665

theorem rhombus_area 
  (a : ℝ) (d1 d2 : ℝ)
  (h_side : a = Real.sqrt 113)
  (h_diagonal_diff : abs (d1 - d2) = 8)
  (h_geq : d1 ≠ d2) : 
  (a^2 * d1 * d2 / 2 = 194) :=
sorry -- Proof to be completed

end NUMINAMATH_GPT_rhombus_area_l516_51665


namespace NUMINAMATH_GPT_carl_cost_l516_51616

theorem carl_cost (property_damage medical_bills : ℝ) (insurance_coverage : ℝ) (carl_coverage : ℝ) (H1 : property_damage = 40000) (H2 : medical_bills = 70000) (H3 : insurance_coverage = 0.80) (H4 : carl_coverage = 0.20) :
  carl_coverage * (property_damage + medical_bills) = 22000 :=
by
  sorry

end NUMINAMATH_GPT_carl_cost_l516_51616


namespace NUMINAMATH_GPT_perimeter_of_original_square_l516_51646

-- Definitions
variables {x : ℝ}
def rect_width := x
def rect_length := 4 * x
def rect_perimeter := 56
def original_square_perimeter := 32

-- Statement
theorem perimeter_of_original_square (x : ℝ) (h : 28 * x = 56) : 4 * (4 * x) = 32 :=
by
  -- Since the proof is not required, we apply sorry to end the theorem.
  sorry

end NUMINAMATH_GPT_perimeter_of_original_square_l516_51646


namespace NUMINAMATH_GPT_simplify_expression_l516_51676

theorem simplify_expression : (Real.sqrt 12 - |1 - Real.sqrt 3| + (7 + Real.pi)^0) = (Real.sqrt 3 + 2) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l516_51676


namespace NUMINAMATH_GPT_dog_tail_length_l516_51687

theorem dog_tail_length (b h t : ℝ) 
  (h_head : h = b / 6) 
  (h_tail : t = b / 2) 
  (h_total : b + h + t = 30) : 
  t = 9 :=
by
  sorry

end NUMINAMATH_GPT_dog_tail_length_l516_51687


namespace NUMINAMATH_GPT_relationship_of_y_l516_51683

theorem relationship_of_y {k y1 y2 y3 : ℝ} (hk : k > 0) :
  (y1 = k / -1) → (y2 = k / 2) → (y3 = k / 3) → y1 < y3 ∧ y3 < y2 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_relationship_of_y_l516_51683


namespace NUMINAMATH_GPT_tea_bags_count_l516_51692

-- Definitions based on the given problem
def valid_bags (b : ℕ) : Prop :=
  ∃ (a c d : ℕ), a + b - a = b ∧ c + d = b ∧ 3 * c + 2 * d = 41 ∧ 3 * a + 2 * (b - a) = 58

-- Statement of the problem, confirming the proof condition
theorem tea_bags_count (b : ℕ) : valid_bags b ↔ b = 20 :=
by {
  -- The proof is left for completion
  sorry
}

end NUMINAMATH_GPT_tea_bags_count_l516_51692


namespace NUMINAMATH_GPT_middle_aged_selection_l516_51672

def total_teachers := 80 + 160 + 240
def sample_size := 60
def middle_aged_proportion := 160 / total_teachers
def middle_aged_sample := middle_aged_proportion * sample_size

theorem middle_aged_selection : middle_aged_sample = 20 :=
  sorry

end NUMINAMATH_GPT_middle_aged_selection_l516_51672


namespace NUMINAMATH_GPT_intersecting_chords_theorem_l516_51606

theorem intersecting_chords_theorem
  (a b : ℝ) (h1 : a = 12) (h2 : b = 18)
  (c d k : ℝ) (h3 : c = 3 * k) (h4 : d = 8 * k) :
  (a * b = c * d) → (k = 3) → (c + d = 33) :=
by 
  sorry

end NUMINAMATH_GPT_intersecting_chords_theorem_l516_51606


namespace NUMINAMATH_GPT_general_term_sequence_l516_51657

theorem general_term_sequence (a : ℕ → ℝ) (h₁ : a 1 = 1) (hn : ∀ (n : ℕ), a (n + 1) = (10 + 4 * a n) / (1 + a n)) :
  ∀ n : ℕ, a n = 5 - 7 / (1 + (3 / 4) * (-6)^(n - 1)) := 
sorry

end NUMINAMATH_GPT_general_term_sequence_l516_51657


namespace NUMINAMATH_GPT_triangle_right_angle_l516_51652

variable {A B C a b c : ℝ}

theorem triangle_right_angle (h1 : Real.sin (A / 2) ^ 2 = (c - b) / (2 * c)) 
                             (h2 : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A) : 
                             a^2 + b^2 = c^2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_right_angle_l516_51652


namespace NUMINAMATH_GPT_condition_holds_l516_51699

theorem condition_holds 
  (a b c d : ℝ) 
  (h : (a^2 + b^2) / (b^2 + c^2) = (c^2 + d^2) / (d^2 + a^2)) : 
  (a = c ∨ a = -c) ∨ (a^2 - c^2 + d^2 = b^2) :=
by
  sorry

end NUMINAMATH_GPT_condition_holds_l516_51699


namespace NUMINAMATH_GPT_years_passed_l516_51678

-- Let PV be the present value of the machine, FV be the final value of the machine, r be the depletion rate, and t be the time in years.
def PV : ℝ := 900
def FV : ℝ := 729
def r : ℝ := 0.10

-- The formula for exponential decay is FV = PV * (1 - r)^t.
-- Given FV = 729, PV = 900, and r = 0.10, we want to prove that t = 2.

theorem years_passed (t : ℕ) : FV = PV * (1 - r)^t → t = 2 := 
by 
  intro h
  sorry

end NUMINAMATH_GPT_years_passed_l516_51678


namespace NUMINAMATH_GPT_special_divisors_count_of_20_30_l516_51633

def prime_number (n : ℕ) : Prop := n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def number_of_divisors (a : ℕ) (α β : ℕ) : ℕ := (α + 1) * (β + 1)

def count_special_divisors (m n : ℕ) : ℕ :=
  let total_divisors_m := (m + 1) * (n + 1)
  let total_divisors_n := (n + 1) * (n / 2 + 1)
  (total_divisors_m - 1) / 2 - total_divisors_n + 1

theorem special_divisors_count_of_20_30 (d_20_30 d_20_15 : ℕ) :
  let α := 60
  let β := 30
  let γ := 30
  let δ := 15
  prime_number 2 ∧ prime_number 5 ∧
  count_special_divisors α β = 1891 ∧
  count_special_divisors γ δ = 496 →
  d_20_30 = 2 * 1891 / 2 ∧
  d_20_15 = 2 * 496 →
  count_special_divisors 60 30 - count_special_divisors 30 15 + 1 = 450
:= by
  sorry

end NUMINAMATH_GPT_special_divisors_count_of_20_30_l516_51633


namespace NUMINAMATH_GPT_average_scissors_correct_l516_51624

-- Definitions for the initial number of scissors in each drawer
def initial_scissors_first_drawer : ℕ := 39
def initial_scissors_second_drawer : ℕ := 27
def initial_scissors_third_drawer : ℕ := 45

-- Definitions for the new scissors added by Dan
def added_scissors_first_drawer : ℕ := 13
def added_scissors_second_drawer : ℕ := 7
def added_scissors_third_drawer : ℕ := 10

-- Calculate the final number of scissors after Dan's addition
def final_scissors_first_drawer : ℕ := initial_scissors_first_drawer + added_scissors_first_drawer
def final_scissors_second_drawer : ℕ := initial_scissors_second_drawer + added_scissors_second_drawer
def final_scissors_third_drawer : ℕ := initial_scissors_third_drawer + added_scissors_third_drawer

-- Statement to prove the average number of scissors in all three drawers
theorem average_scissors_correct :
  (final_scissors_first_drawer + final_scissors_second_drawer + final_scissors_third_drawer) / 3 = 47 := by
  sorry

end NUMINAMATH_GPT_average_scissors_correct_l516_51624


namespace NUMINAMATH_GPT_ratio_proof_l516_51662

theorem ratio_proof (a b c : ℝ) (h1 : b / a = 3) (h2 : c / b = 4) : (a + b) / (b + c) = 4 / 15 := by
  sorry

end NUMINAMATH_GPT_ratio_proof_l516_51662


namespace NUMINAMATH_GPT_cream_cheese_cost_l516_51693

theorem cream_cheese_cost
  (B C : ℝ)
  (h1 : 2 * B + 3 * C = 12)
  (h2 : 4 * B + 2 * C = 14) :
  C = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_cream_cheese_cost_l516_51693


namespace NUMINAMATH_GPT_possible_denominators_count_l516_51648

theorem possible_denominators_count :
  ∀ a b c : ℕ, 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧ (a ≠ 9 ∨ b ≠ 9 ∨ c ≠ 9) →
  ∃ (D : Finset ℕ), D.card = 7 ∧ 
  ∀ num denom, (num = 100*a + 10*b + c) → (denom = 999) → (gcd num denom > 1) → 
  denom ∈ D := 
sorry

end NUMINAMATH_GPT_possible_denominators_count_l516_51648


namespace NUMINAMATH_GPT_smallest_b_no_inverse_mod75_and_mod90_l516_51602

theorem smallest_b_no_inverse_mod75_and_mod90 :
  ∃ b : ℕ, b > 0 ∧ (∀ n : ℕ, n > 0 → n < b →  ¬ (n.gcd 75 > 1 ∧ n.gcd 90 > 1)) ∧ 
  (b.gcd 75 > 1 ∧ b.gcd 90 > 1) ∧ 
  b = 15 := 
by
  sorry

end NUMINAMATH_GPT_smallest_b_no_inverse_mod75_and_mod90_l516_51602


namespace NUMINAMATH_GPT_evaluate_expression_l516_51630

theorem evaluate_expression : 
  ( (7 : ℝ) ^ (1 / 4) / (7 : ℝ) ^ (1 / 7) ) = 7 ^ (3 / 28) := 
by {
  sorry
}

end NUMINAMATH_GPT_evaluate_expression_l516_51630


namespace NUMINAMATH_GPT_part1_part2_l516_51622

noncomputable def set_A (a : ℝ) : Set ℝ := {x : ℝ | 2 * a ≤ x ∧ x ≤ a + 3}
noncomputable def set_B : Set ℝ := {x : ℝ | x < -1 ∨ x > 1}

theorem part1 (a : ℝ) : (set_A a ∩ set_B = ∅) ↔ (a > 3) :=
by sorry

theorem part2 (a : ℝ) : (set_A a ∪ set_B = Set.univ) ↔ (-2 ≤ a ∧ a ≤ -1 / 2) :=
by sorry

end NUMINAMATH_GPT_part1_part2_l516_51622


namespace NUMINAMATH_GPT_tournament_start_count_l516_51614

theorem tournament_start_count (x : ℝ) (h1 : (0.1 * x = 30)) : x = 300 :=
by
  sorry

end NUMINAMATH_GPT_tournament_start_count_l516_51614


namespace NUMINAMATH_GPT_coordinates_F_l516_51675

-- Definition of point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Reflection over the y-axis
def reflect_y (p : Point) : Point :=
  { x := -p.x, y := p.y }

-- Reflection over the x-axis
def reflect_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

-- Original point F
def F : Point := { x := 3, y := 3 }

-- First reflection over the y-axis
def F' := reflect_y F

-- Second reflection over the x-axis
def F'' := reflect_x F'

-- Goal: Coordinates of F'' after both reflections
theorem coordinates_F'' : F'' = { x := -3, y := -3 } :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_coordinates_F_l516_51675


namespace NUMINAMATH_GPT_hyperbola_center_l516_51654

theorem hyperbola_center (x y : ℝ) :
  ∃ h k : ℝ, (∃ a b : ℝ, a = 9/4 ∧ b = 7/2) ∧ (h, k) = (-2, 3) ∧ 
  (4*x + 8)^2 / 81 - (2*y - 6)^2 / 49 = 1 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_center_l516_51654


namespace NUMINAMATH_GPT_compare_neg_fractions_l516_51653

theorem compare_neg_fractions : (- (3 / 2) < -1) :=
by sorry

end NUMINAMATH_GPT_compare_neg_fractions_l516_51653


namespace NUMINAMATH_GPT_distinct_pairs_l516_51694

-- Definitions of rational numbers and distinctness.
def is_distinct (x y : ℚ) : Prop := x ≠ y

-- Conditions
variables {a b r s : ℚ}

-- Main theorem: prove that there is only 1 distinct pair (a, b)
theorem distinct_pairs (h_ab_distinct : is_distinct a b)
  (h_rs_distinct : is_distinct r s)
  (h_eq : ∀ z : ℚ, (z - r) * (z - s) = (z - a * r) * (z - b * s)) : 
    ∃! (a b : ℚ), ∀ z : ℚ, (z - r) * (z - s) = (z - a * r) * (z - b * s) :=
  sorry

end NUMINAMATH_GPT_distinct_pairs_l516_51694


namespace NUMINAMATH_GPT_tom_remaining_balloons_l516_51601

theorem tom_remaining_balloons (initial_balloons : ℕ) (balloons_given : ℕ) (balloons_remaining : ℕ) 
  (h1 : initial_balloons = 30) (h2 : balloons_given = 16) : balloons_remaining = 14 := 
by
  sorry

end NUMINAMATH_GPT_tom_remaining_balloons_l516_51601


namespace NUMINAMATH_GPT_smallest_positive_period_and_range_sin_2x0_if_zero_of_f_l516_51642

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - (1 / 2) * Real.cos (2 * x)

theorem smallest_positive_period_and_range :
  (∀ x, f (x + Real.pi) = f x) ∧ (Set.range f = Set.Icc (-3 / 2) (5 / 2)) :=
by
  sorry

theorem sin_2x0_if_zero_of_f (x0 : ℝ) (hx0 : 0 ≤ x0 ∧ x0 ≤ Real.pi / 2)
  (hf : f x0 = 0) : Real.sin (2 * x0) = (Real.sqrt 15 - Real.sqrt 3) / 8 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_period_and_range_sin_2x0_if_zero_of_f_l516_51642


namespace NUMINAMATH_GPT_trig_identity_l516_51668

theorem trig_identity (α : ℝ) (h : Real.tan α = 2) : 
  ∃ (res : ℝ), res = 10 / 7 ∧ res = Real.sin α / (Real.sin α ^ 3 - Real.cos α ^ 3) := by
  sorry

end NUMINAMATH_GPT_trig_identity_l516_51668


namespace NUMINAMATH_GPT_set_union_covers_real_line_l516_51643

open Set

def M := {x : ℝ | x < 0 ∨ 2 < x}
def N := {x : ℝ | -Real.sqrt 5 < x ∧ x < Real.sqrt 5}

theorem set_union_covers_real_line : M ∪ N = univ := sorry

end NUMINAMATH_GPT_set_union_covers_real_line_l516_51643


namespace NUMINAMATH_GPT_range_of_m_l516_51617

open Set

def M (m : ℝ) : Set ℝ := {x | x ≤ m}
def N : Set ℝ := {y | ∃ x : ℝ, y = 2^(-x)}

theorem range_of_m (m : ℝ) : (M m ∩ N).Nonempty ↔ m > 0 := sorry

end NUMINAMATH_GPT_range_of_m_l516_51617


namespace NUMINAMATH_GPT_flags_count_l516_51666

-- Define the colors available
inductive Color
| purple | gold | silver

-- Define the number of stripes on the flag
def number_of_stripes : Nat := 3

-- Define a function to calculate the total number of combinations
def total_flags (colors : Nat) (stripes : Nat) : Nat :=
  colors ^ stripes

-- The main theorem we want to prove
theorem flags_count : total_flags 3 number_of_stripes = 27 :=
by
  -- This is the statement only, and the proof is omitted
  sorry

end NUMINAMATH_GPT_flags_count_l516_51666


namespace NUMINAMATH_GPT_union_intersection_l516_51637

-- Define the sets M, N, and P
def M := ({1} : Set Nat)
def N := ({1, 2} : Set Nat)
def P := ({1, 2, 3} : Set Nat)

-- Prove that (M ∪ N) ∩ P = {1, 2}
theorem union_intersection : (M ∪ N) ∩ P = ({1, 2} : Set Nat) := 
by 
  sorry

end NUMINAMATH_GPT_union_intersection_l516_51637


namespace NUMINAMATH_GPT_minimum_value_of_f_l516_51603

noncomputable def f (x : ℝ) : ℝ := x + 1/x + 1/(x + 1/x)

theorem minimum_value_of_f :
  (∀ x : ℝ, x > 0 → f x ≥ 5/2) ∧ (f 1 = 5/2) := by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l516_51603


namespace NUMINAMATH_GPT_small_cubes_for_larger_cube_l516_51636

theorem small_cubes_for_larger_cube (VL VS : ℕ) (h : VL = 125 * VS) : (VL / VS = 125) :=
by {
    sorry
}

end NUMINAMATH_GPT_small_cubes_for_larger_cube_l516_51636


namespace NUMINAMATH_GPT_probability_multiple_of_4_l516_51621

theorem probability_multiple_of_4 :
  let num_cards := 12
  let num_multiple_of_4 := 3
  let prob_start_multiple_of_4 := (num_multiple_of_4 : ℚ) / num_cards
  let prob_RR := (1 / 2 : ℚ) * (1 / 2)
  let prob_L2R := (1 / 4 : ℚ) * (1 / 4)
  let prob_RL := (1 / 2 : ℚ) * (1 / 4)
  let total_prob_stay_multiple_of_4 := prob_RR + prob_L2R + prob_RL
  let prob_end_multiple_of_4 := prob_start_multiple_of_4 * total_prob_stay_multiple_of_4
  prob_end_multiple_of_4 = 7 / 64 :=
by
  let num_cards := 12
  let num_multiple_of_4 := 3
  let prob_start_multiple_of_4 := (num_multiple_of_4 : ℚ) / num_cards
  let prob_RR := (1 / 2 : ℚ) * (1 / 2)
  let prob_L2R := (1 / 4 : ℚ) * (1 / 4)
  let prob_RL := (1 / 2 : ℚ) * (1 / 4)
  let total_prob_stay_multiple_of_4 := prob_RR + prob_L2R + prob_RL
  let prob_end_multiple_of_4 := prob_start_multiple_of_4 * total_prob_stay_multiple_of_4
  have h : prob_end_multiple_of_4 = 7 / 64 := by sorry
  exact h

end NUMINAMATH_GPT_probability_multiple_of_4_l516_51621


namespace NUMINAMATH_GPT_cubes_sum_l516_51607

theorem cubes_sum (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end NUMINAMATH_GPT_cubes_sum_l516_51607


namespace NUMINAMATH_GPT_minimize_surface_area_l516_51611

-- Define the problem conditions
def volume (x y : ℝ) : ℝ := 2 * x^2 * y
def surface_area (x y : ℝ) : ℝ := 2 * (2 * x^2 + 2 * x * y + x * y)

theorem minimize_surface_area :
  ∃ (y : ℝ), 
  (∀ (x : ℝ), volume x y = 72) → 
  1 * 2 * y = 4 :=
by
  sorry

end NUMINAMATH_GPT_minimize_surface_area_l516_51611


namespace NUMINAMATH_GPT_candy_bar_cost_l516_51628

/-- Problem statement:
Todd had 85 cents and spent 53 cents in total on a candy bar and a box of cookies.
The box of cookies cost 39 cents. How much did the candy bar cost? --/
theorem candy_bar_cost (t c s b : ℕ) (ht : t = 85) (hc : c = 39) (hs : s = 53) (h_total : s = b + c) : b = 14 :=
by
  sorry

end NUMINAMATH_GPT_candy_bar_cost_l516_51628


namespace NUMINAMATH_GPT_shortest_chord_line_through_P_longest_chord_line_through_P_l516_51660

theorem shortest_chord_line_through_P (P : ℝ × ℝ) (circle : (ℝ × ℝ) → Prop) (hP : P = (-1, 2))
  (h_circle_eq : ∀ (x y : ℝ), circle (x, y) ↔ x ^ 2 + y ^ 2 = 8) :
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ (x y : ℝ), y = 1/2 * x + 5/2 → a * x + b * y + c = 0)
  ∧ (a = 1) ∧ (b = -2) ∧ (c = 5) := sorry

theorem longest_chord_line_through_P (P : ℝ × ℝ) (circle : (ℝ × ℝ) → Prop) (hP : P = (-1, 2))
  (h_circle_eq : ∀ (x y : ℝ), circle (x, y) ↔ x ^ 2 + y ^ 2 = 8) :
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ (x y : ℝ), y = -2 * x → a * x + b * y + c = 0)
  ∧ (a = 2) ∧ (b = 1) ∧ (c = 0) := sorry

end NUMINAMATH_GPT_shortest_chord_line_through_P_longest_chord_line_through_P_l516_51660


namespace NUMINAMATH_GPT_symmetric_point_yaxis_correct_l516_51684

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def symmetric_yaxis (P : Point3D) : Point3D :=
  { x := -P.x, y := P.y, z := P.z }

theorem symmetric_point_yaxis_correct (P : Point3D) (P' : Point3D) :
  P = {x := 1, y := 2, z := -1} → 
  P' = symmetric_yaxis P → 
  P' = {x := -1, y := 2, z := -1} :=
by
  intros hP hP'
  rw [hP] at hP'
  simp [symmetric_yaxis] at hP'
  exact hP'

end NUMINAMATH_GPT_symmetric_point_yaxis_correct_l516_51684


namespace NUMINAMATH_GPT_feet_more_than_heads_l516_51697

def num_hens := 50
def num_goats := 45
def num_camels := 8
def num_keepers := 15

def feet_per_hen := 2
def feet_per_goat := 4
def feet_per_camel := 4
def feet_per_keeper := 2

def total_heads := num_hens + num_goats + num_camels + num_keepers
def total_feet := (num_hens * feet_per_hen) + (num_goats * feet_per_goat) + (num_camels * feet_per_camel) + (num_keepers * feet_per_keeper)

-- Theorem to prove:
theorem feet_more_than_heads : total_feet - total_heads = 224 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_feet_more_than_heads_l516_51697


namespace NUMINAMATH_GPT_parallel_lines_m_eq_one_l516_51638

theorem parallel_lines_m_eq_one (m : ℝ) :
  (∀ x y : ℝ, x + (1 + m) * y + (m - 2) = 0 ∧ 2 * m * x + 4 * y + 16 = 0 → m = 1) :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_m_eq_one_l516_51638


namespace NUMINAMATH_GPT_find_ending_number_l516_51673

theorem find_ending_number (n : ℕ) 
  (h1 : n ≥ 7) 
  (h2 : ∀ m, 7 ≤ m ∧ m ≤ n → m % 7 = 0)
  (h3 : (7 + n) / 2 = 15) : n = 21 := 
sorry

end NUMINAMATH_GPT_find_ending_number_l516_51673


namespace NUMINAMATH_GPT_increasing_function_on_R_l516_51620

theorem increasing_function_on_R (x1 x2 : ℝ) (h : x1 < x2) : 3 * x1 + 2 < 3 * x2 + 2 := 
by
  sorry

end NUMINAMATH_GPT_increasing_function_on_R_l516_51620


namespace NUMINAMATH_GPT_gloria_pencils_total_l516_51634

-- Define the number of pencils Gloria initially has.
def pencils_gloria_initial : ℕ := 2

-- Define the number of pencils Lisa initially has.
def pencils_lisa_initial : ℕ := 99

-- Define the final number of pencils Gloria will have after receiving all of Lisa's pencils.
def pencils_gloria_final : ℕ := pencils_gloria_initial + pencils_lisa_initial

-- Prove that the final number of pencils Gloria will have is 101.
theorem gloria_pencils_total : pencils_gloria_final = 101 :=
by sorry

end NUMINAMATH_GPT_gloria_pencils_total_l516_51634


namespace NUMINAMATH_GPT_tom_spend_l516_51613

def theater_cost (seat_count : ℕ) (sqft_per_seat : ℕ) (cost_per_sqft : ℕ) (construction_multiplier : ℕ) (partner_percentage : ℝ) : ℝ :=
  let total_sqft := seat_count * sqft_per_seat
  let land_cost := total_sqft * cost_per_sqft
  let construction_cost := construction_multiplier * land_cost
  let total_cost := land_cost + construction_cost
  let partner_contribution := partner_percentage * (total_cost : ℝ)
  total_cost - partner_contribution

theorem tom_spend (partner_percentage : ℝ) :
  theater_cost 500 12 5 2 partner_percentage = 54000 :=
sorry

end NUMINAMATH_GPT_tom_spend_l516_51613


namespace NUMINAMATH_GPT_beth_sheep_l516_51698

-- Definition: number of sheep Beth has (B)
variable (B : ℕ)

-- Condition 1: Aaron has 7 times as many sheep as Beth
def Aaron_sheep (B : ℕ) := 7 * B

-- Condition 2: Together, Aaron and Beth have 608 sheep
axiom together_sheep : B + Aaron_sheep B = 608

-- Theorem: Prove that Beth has 76 sheep
theorem beth_sheep : B = 76 :=
sorry

end NUMINAMATH_GPT_beth_sheep_l516_51698


namespace NUMINAMATH_GPT_total_sand_l516_51688

variable (capacity_per_bag : ℕ) (number_of_bags : ℕ)

theorem total_sand (h1 : capacity_per_bag = 65) (h2 : number_of_bags = 12) : capacity_per_bag * number_of_bags = 780 := by
  sorry

end NUMINAMATH_GPT_total_sand_l516_51688


namespace NUMINAMATH_GPT_range_h_l516_51655

noncomputable def h (x : ℝ) : ℝ := 3 / (3 + 5 * x^2)

theorem range_h (a b : ℝ) (h_range : Set.Ioo a b = Set.Icc 0 1) : a + b = 1 := by
  sorry

end NUMINAMATH_GPT_range_h_l516_51655


namespace NUMINAMATH_GPT_pencils_loss_equates_20_l516_51664

/--
Patrick purchased 70 pencils and sold them at a loss equal to the selling price of some pencils. The cost of 70 pencils is 1.2857142857142856 times the selling price of 70 pencils. Prove that the loss equates to the selling price of 20 pencils.
-/
theorem pencils_loss_equates_20 
  (C S : ℝ) 
  (h1 : C = 1.2857142857142856 * S) :
  (70 * C - 70 * S) = 20 * S :=
by
  sorry

end NUMINAMATH_GPT_pencils_loss_equates_20_l516_51664


namespace NUMINAMATH_GPT_haleigh_needs_46_leggings_l516_51615

-- Define the number of each type of animal
def num_dogs : ℕ := 4
def num_cats : ℕ := 3
def num_spiders : ℕ := 2
def num_parrot : ℕ := 1

-- Define the number of legs each type of animal has
def legs_dog : ℕ := 4
def legs_cat : ℕ := 4
def legs_spider : ℕ := 8
def legs_parrot : ℕ := 2

-- Define the total number of legs function
def total_leggings (d c s p : ℕ) (ld lc ls lp : ℕ) : ℕ :=
  d * ld + c * lc + s * ls + p * lp

-- The statement to be proven
theorem haleigh_needs_46_leggings : total_leggings num_dogs num_cats num_spiders num_parrot legs_dog legs_cat legs_spider legs_parrot = 46 := by
  sorry

end NUMINAMATH_GPT_haleigh_needs_46_leggings_l516_51615


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l516_51667

-- Definitions of conditions
variables {a b c : ℝ}
variables (h : a > 0) (h' : b > 0)
variables (hyp : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1)
variables (parab : ∀ y : ℝ, y^2 = 4 * b * y)
variables (ratio_cond : (b + c) / (c - b) = 5 / 3)

-- Proof statement
theorem hyperbola_eccentricity : ∃ (e : ℝ), e = 4 * Real.sqrt 15 / 15 :=
by
  have hyp_foci_distance : ∃ c : ℝ, c^2 = a^2 + b^2 := sorry
  have e := (4 * Real.sqrt 15) / 15
  use e
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l516_51667


namespace NUMINAMATH_GPT_total_ducks_l516_51669

-- Definitions based on the given conditions
def Muscovy : ℕ := 39
def Cayuga : ℕ := Muscovy - 4
def KhakiCampbell : ℕ := (Cayuga - 3) / 2

-- Proof statement
theorem total_ducks : Muscovy + Cayuga + KhakiCampbell = 90 := by
  sorry

end NUMINAMATH_GPT_total_ducks_l516_51669


namespace NUMINAMATH_GPT_joan_kittens_remaining_l516_51650

def original_kittens : ℕ := 8
def kittens_given_away : ℕ := 2

theorem joan_kittens_remaining : original_kittens - kittens_given_away = 6 := by
  sorry

end NUMINAMATH_GPT_joan_kittens_remaining_l516_51650


namespace NUMINAMATH_GPT_intersect_P_M_l516_51649

def P : Set ℝ := {x | 0 ≤ x ∧ x < 3}
def M : Set ℝ := {x | |x| ≤ 3}

theorem intersect_P_M : (P ∩ M) = {x | 0 ≤ x ∧ x < 3} := by
  sorry

end NUMINAMATH_GPT_intersect_P_M_l516_51649


namespace NUMINAMATH_GPT_unique_function_satisfies_condition_l516_51605

theorem unique_function_satisfies_condition :
  ∃! f : ℝ → ℝ, ∀ x y z : ℝ, f (x * Real.sin y) + f (x * Real.sin z) -
    f x * f (Real.sin y * Real.sin z) + Real.sin (Real.pi * x) ≥ 1 := sorry

end NUMINAMATH_GPT_unique_function_satisfies_condition_l516_51605


namespace NUMINAMATH_GPT_exists_arithmetic_seq_perfect_powers_l516_51629

def is_perfect_power (x : ℕ) : Prop := ∃ (a k : ℕ), k > 1 ∧ x = a^k

theorem exists_arithmetic_seq_perfect_powers (n : ℕ) (hn : n > 1) :
  ∃ (a d : ℕ) (seq : ℕ → ℕ), (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → seq i = a + (i - 1) * d)
  ∧ ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → is_perfect_power (seq i)
  ∧ d ≠ 0 :=
sorry

end NUMINAMATH_GPT_exists_arithmetic_seq_perfect_powers_l516_51629


namespace NUMINAMATH_GPT_largest_four_digit_negative_integer_congruent_to_2_mod_17_l516_51608

theorem largest_four_digit_negative_integer_congruent_to_2_mod_17 :
  ∃ (n : ℤ), (n % 17 = 2 ∧ n > -10000 ∧ n < -999) ∧ ∀ m : ℤ, (m % 17 = 2 ∧ m > -10000 ∧ m < -999) → m ≤ n :=
sorry

end NUMINAMATH_GPT_largest_four_digit_negative_integer_congruent_to_2_mod_17_l516_51608


namespace NUMINAMATH_GPT_g_periodic_6_l516_51647

def g (a b c : ℝ) : ℝ × ℝ × ℝ :=
  (a + b, b + c, a + c)

def g_iter (n : Nat) (triple : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  match n with
  | 0 => triple
  | n + 1 => g (g_iter n triple).1 (g_iter n triple).2.1 (g_iter n triple).2.2

theorem g_periodic_6 {a b c : ℝ} (h : ∃ n : Nat, n > 0 ∧ g_iter n (a, b, c) = (a, b, c))
  (h' : (a, b, c) ≠ (0, 0, 0)) : g_iter 6 (a, b, c) = (a, b, c) :=
by
  sorry

end NUMINAMATH_GPT_g_periodic_6_l516_51647


namespace NUMINAMATH_GPT_value_of_x_l516_51680

theorem value_of_x (b x : ℝ) (h₀ : 1 < b) (h₁ : 0 < x) (h₂ : (2 * x) ^ (Real.logb b 2) - (3 * x) ^ (Real.logb b 3) = 0) : x = 1 / 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_value_of_x_l516_51680


namespace NUMINAMATH_GPT_length_segment_ZZ_l516_51685

variable (Z : ℝ × ℝ) (Z' : ℝ × ℝ)

theorem length_segment_ZZ' 
  (h_Z : Z = (-5, 3)) (h_Z' : Z' = (5, 3)) : 
  dist Z Z' = 10 := by
  sorry

end NUMINAMATH_GPT_length_segment_ZZ_l516_51685


namespace NUMINAMATH_GPT_starting_number_of_three_squares_less_than_2300_l516_51635

theorem starting_number_of_three_squares_less_than_2300 : 
  ∃ n1 n2 n3 : ℕ, n1 < n2 ∧ n2 < n3 ∧ n3^2 < 2300 ∧ n2^2 < 2300 ∧ n1^2 < 2300 ∧ n3^2 ≥ 2209 ∧ n2^2 ≥ 2116 ∧ n1^2 = 2025 :=
by {
  sorry
}

end NUMINAMATH_GPT_starting_number_of_three_squares_less_than_2300_l516_51635


namespace NUMINAMATH_GPT_problem_geometric_sequence_l516_51689

variable {α : Type*} [LinearOrderedField α]

noncomputable def geom_sequence_5_8 (a : α) (h : a + 8 * a = 2) : α :=
  (a * 2^4 + a * 2^7)

theorem problem_geometric_sequence : ∃ (a : α), (a + 8 * a = 2) ∧ geom_sequence_5_8 a (sorry) = 32 := 
by sorry

end NUMINAMATH_GPT_problem_geometric_sequence_l516_51689


namespace NUMINAMATH_GPT_stamps_per_light_envelope_l516_51691

theorem stamps_per_light_envelope 
  (stamps_heavy : ℕ) (stamps_light : ℕ → ℕ) (total_light : ℕ) (total_stamps_light : ℕ)
  (total_envelopes : ℕ) :
  (∀ n, n > 5 → stamps_heavy = 5) →
  (∀ n, n <= 5 → stamps_light n = total_stamps_light / total_light) →
  total_light = 6 →
  total_stamps_light = 52 →
  total_envelopes = 14 →
  stamps_light 5 = 9 :=
by
  sorry

end NUMINAMATH_GPT_stamps_per_light_envelope_l516_51691


namespace NUMINAMATH_GPT_negation_universal_proposition_l516_51623

theorem negation_universal_proposition :
  (¬∀ x : ℝ, 0 ≤ x → x^3 + x ≥ 0) ↔ (∃ x : ℝ, 0 ≤ x ∧ x^3 + x < 0) :=
by sorry

end NUMINAMATH_GPT_negation_universal_proposition_l516_51623


namespace NUMINAMATH_GPT_cost_one_dozen_pens_l516_51663

variable (cost_of_pen cost_of_pencil : ℝ)
variable (ratio : ℝ)
variable (dozen_pens_cost : ℝ)

axiom cost_equation : 3 * cost_of_pen + 5 * cost_of_pencil = 200
axiom ratio_pen_pencil : cost_of_pen = 5 * cost_of_pencil

theorem cost_one_dozen_pens : dozen_pens_cost = 12 * cost_of_pen := 
  by
    sorry

end NUMINAMATH_GPT_cost_one_dozen_pens_l516_51663


namespace NUMINAMATH_GPT_derivative_at_one_l516_51625

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 - 1

theorem derivative_at_one : deriv f 1 = 2 :=
by sorry

end NUMINAMATH_GPT_derivative_at_one_l516_51625


namespace NUMINAMATH_GPT_sequence_difference_l516_51639

theorem sequence_difference
  (a : ℕ → ℤ)
  (h : ∀ n : ℕ, a (n + 1) - a n - n = 0) :
  a 2017 - a 2016 = 2016 :=
by
  sorry

end NUMINAMATH_GPT_sequence_difference_l516_51639


namespace NUMINAMATH_GPT_italian_dressing_mixture_l516_51670

/-- A chef is using a mixture of two brands of Italian dressing. 
  The first brand contains 8% vinegar, and the second brand contains 13% vinegar.
  The chef wants to make 320 milliliters of a dressing that is 11% vinegar.
  This statement proves the amounts required for each brand of dressing. -/

theorem italian_dressing_mixture
  (x y : ℝ)
  (hx : x + y = 320)
  (hv : 0.08 * x + 0.13 * y = 0.11 * 320) :
  x = 128 ∧ y = 192 :=
sorry

end NUMINAMATH_GPT_italian_dressing_mixture_l516_51670


namespace NUMINAMATH_GPT_range_of_a_l516_51658

theorem range_of_a (a : ℝ) (in_fourth_quadrant : (a+2 > 0) ∧ (a-3 < 0)) : -2 < a ∧ a < 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l516_51658


namespace NUMINAMATH_GPT_apple_juice_cost_l516_51612

noncomputable def cost_of_apple_juice (cost_per_orange_juice : ℝ) (total_bottles : ℕ) (total_cost : ℝ) (orange_juice_bottles : ℕ) : ℝ :=
  (total_cost - cost_per_orange_juice * orange_juice_bottles) / (total_bottles - orange_juice_bottles)

theorem apple_juice_cost :
  let cost_per_orange_juice := 0.7
  let total_bottles := 70
  let total_cost := 46.2
  let orange_juice_bottles := 42
  cost_of_apple_juice cost_per_orange_juice total_bottles total_cost orange_juice_bottles = 0.6 := by
    sorry

end NUMINAMATH_GPT_apple_juice_cost_l516_51612


namespace NUMINAMATH_GPT_radius_larger_circle_l516_51641

theorem radius_larger_circle (r : ℝ) (AC BC : ℝ) (h1 : 5 * r = AC / 2) (h2 : 15 = BC) : 
  5 * r = 18.75 :=
by
  sorry

end NUMINAMATH_GPT_radius_larger_circle_l516_51641


namespace NUMINAMATH_GPT_area_of_rectangle_l516_51600

theorem area_of_rectangle (width length : ℝ) (h_width : width = 5.4) (h_length : length = 2.5) : width * length = 13.5 :=
by
  -- We are given that the width is 5.4 and the length is 2.5
  -- We need to show that the area (width * length) is 13.5
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l516_51600


namespace NUMINAMATH_GPT_evaluate_expression_l516_51619

theorem evaluate_expression (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 3) :
  (3 * x ^ 2 - 2 * x + 1) / ((x + 2) * (x - 3)) - (x ^ 2 - 5 * x + 6) / ((x + 2) * (x - 3)) =
  (2 * x ^ 2 + 3 * x - 5) / ((x + 2) * (x - 3)) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l516_51619


namespace NUMINAMATH_GPT_num_ordered_pairs_no_real_solution_l516_51695

theorem num_ordered_pairs_no_real_solution : 
  {n : ℕ // ∃ (b c : ℕ), b > 0 ∧ c > 0 ∧ (b^2 - 4*c < 0 ∨ c^2 - 4*b < 0) ∧ n = 6 } := by
sorry

end NUMINAMATH_GPT_num_ordered_pairs_no_real_solution_l516_51695


namespace NUMINAMATH_GPT_find_xy_l516_51661

theorem find_xy (x y : ℝ) (h1 : x^5 + y^5 = 33) (h2 : x + y = 3) :
  (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) :=
by
  sorry

end NUMINAMATH_GPT_find_xy_l516_51661


namespace NUMINAMATH_GPT_ratio_of_mixture_l516_51659

theorem ratio_of_mixture (x y : ℚ)
  (h1 : 0.6 = (4 * x + 7 * y) / (9 * x + 9 * y))
  (h2 : 50 = 9 * x + 9 * y) : x / y = 8 / 7 := 
sorry

end NUMINAMATH_GPT_ratio_of_mixture_l516_51659


namespace NUMINAMATH_GPT_count_values_of_b_l516_51674

theorem count_values_of_b : 
  ∃! n : ℕ, (n = 4) ∧ (∀ b : ℕ, (b > 0) → (b ≤ 100) → (∃ k : ℤ, 5 * b^2 + 12 * b + 4 = k^2) → 
    (b = 4 ∨ b = 20 ∨ b = 44 ∨ b = 76)) :=
by
  sorry

end NUMINAMATH_GPT_count_values_of_b_l516_51674


namespace NUMINAMATH_GPT_sequence_bound_l516_51645

theorem sequence_bound (a : ℕ → ℝ) (h1 : ∀ n, a n > 0) (h2 : ∀ n, (a n)^2 ≤ a n - a (n+1)) :
  ∀ n, a n < 1 / n :=
by
  sorry

end NUMINAMATH_GPT_sequence_bound_l516_51645


namespace NUMINAMATH_GPT_students_count_l516_51686

theorem students_count (n : ℕ) (avg_age_n_students : ℕ) (sum_age_7_students1 : ℕ) (sum_age_7_students2 : ℕ) (last_student_age : ℕ) :
  avg_age_n_students = 15 →
  sum_age_7_students1 = 7 * 14 →
  sum_age_7_students2 = 7 * 16 →
  last_student_age = 15 →
  (sum_age_7_students1 + sum_age_7_students2 + last_student_age = avg_age_n_students * n) →
  n = 15 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_students_count_l516_51686
