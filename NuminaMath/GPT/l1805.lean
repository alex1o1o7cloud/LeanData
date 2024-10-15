import Mathlib

namespace NUMINAMATH_GPT_John_new_weekly_earnings_l1805_180582

theorem John_new_weekly_earnings (original_earnings : ℕ) (raise_percentage : ℚ) 
  (raise_in_dollars : ℚ) (new_weekly_earnings : ℚ)
  (h1 : original_earnings = 30) 
  (h2 : raise_percentage = 33.33) 
  (h3 : raise_in_dollars = (raise_percentage / 100) * original_earnings) 
  (h4 : new_weekly_earnings = original_earnings + raise_in_dollars) :
  new_weekly_earnings = 40 := sorry

end NUMINAMATH_GPT_John_new_weekly_earnings_l1805_180582


namespace NUMINAMATH_GPT_triangle_perimeter_l1805_180595

-- Conditions as definitions
def is_isosceles_triangle (a b c : ℕ) : Prop :=
  a = b ∨ b = c ∨ c = a

def has_sides (a b : ℕ) : Prop :=
  a = 4 ∨ b = 4 ∨ a = 9 ∨ b = 9

def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the isosceles triangle with specified sides
structure IsoTriangle :=
  (a b c : ℕ)
  (iso : is_isosceles_triangle a b c)
  (valid_sides : has_sides a b ∧ has_sides a c ∧ has_sides b c)
  (triangle : triangle_inequality a b c)

-- The statement to prove perimeter
def perimeter (T : IsoTriangle) : ℕ :=
  T.a + T.b + T.c

-- The theorem we aim to prove
theorem triangle_perimeter (T : IsoTriangle) (h: T.a = 9 ∧ T.b = 9 ∧ T.c = 4) : perimeter T = 22 :=
sorry

end NUMINAMATH_GPT_triangle_perimeter_l1805_180595


namespace NUMINAMATH_GPT_units_digit_17_pow_31_l1805_180532

theorem units_digit_17_pow_31 : (17 ^ 31) % 10 = 3 := by
  sorry

end NUMINAMATH_GPT_units_digit_17_pow_31_l1805_180532


namespace NUMINAMATH_GPT_hair_cut_amount_l1805_180567

theorem hair_cut_amount (initial_length final_length cut_length : ℕ) (h1 : initial_length = 11) (h2 : final_length = 7) : cut_length = 4 :=
by 
  sorry

end NUMINAMATH_GPT_hair_cut_amount_l1805_180567


namespace NUMINAMATH_GPT_product_xyz_l1805_180504

/-- Prove that if x + 1/y = 2 and y + 1/z = 3, then xyz = 1/11. -/
theorem product_xyz {x y z : ℝ} (h1 : x + 1 / y = 2) (h2 : y + 1 / z = 3) : x * y * z = 1 / 11 :=
sorry

end NUMINAMATH_GPT_product_xyz_l1805_180504


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1805_180590

theorem simplify_and_evaluate :
  ∀ (a b : ℚ), a = 2 → b = -1/2 → (a - 2 * (a - b^2) + 3 * (-a + b^2) = -27/4) :=
by
  intros a b ha hb
  rw [ha, hb]
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1805_180590


namespace NUMINAMATH_GPT_second_shift_production_l1805_180538

-- Question: Prove that the number of cars produced by the second shift is 1,100 given the conditions
-- Conditions:
-- 1. P_day = 4 * P_second
-- 2. P_day + P_second = 5,500

theorem second_shift_production (P_day P_second : ℕ) (h1 : P_day = 4 * P_second) (h2 : P_day + P_second = 5500) :
  P_second = 1100 := by
  sorry

end NUMINAMATH_GPT_second_shift_production_l1805_180538


namespace NUMINAMATH_GPT_walking_distance_l1805_180570

theorem walking_distance (west east : ℤ) (h_west : west = 5) (h_east : east = -5) : west + east = 10 := 
by 
  rw [h_west, h_east] 
  sorry

end NUMINAMATH_GPT_walking_distance_l1805_180570


namespace NUMINAMATH_GPT_interest_rate_difference_l1805_180581

def simple_interest (P R T : ℕ) : ℕ := (P * R * T) / 100

def si1 (R1 : ℕ) : ℕ := simple_interest 800 R1 10
def si2 (R2 : ℕ) : ℕ := simple_interest 800 R2 10

theorem interest_rate_difference (R1 R2 : ℕ) (h : si2 R2 = si1 R1 + 400) : R2 - R1 = 5 := 
by sorry

end NUMINAMATH_GPT_interest_rate_difference_l1805_180581


namespace NUMINAMATH_GPT_total_cost_750_candies_l1805_180518

def candy_cost (candies : ℕ) (cost_per_box : ℕ) (candies_per_box : ℕ) (discount_threshold : ℕ) (discount_rate : ℝ) : ℝ :=
  let boxes := candies / candies_per_box
  let total_cost := boxes * cost_per_box
  if candies > discount_threshold then
    (1 - discount_rate) * total_cost
  else
    total_cost

theorem total_cost_750_candies :
  candy_cost 750 8 30 500 0.1 = 180 :=
by sorry

end NUMINAMATH_GPT_total_cost_750_candies_l1805_180518


namespace NUMINAMATH_GPT_original_mixture_percentage_l1805_180558

variables (a w : ℝ)

-- Conditions given
def condition1 : Prop := a / (a + w + 2) = 0.3
def condition2 : Prop := (a + 2) / (a + w + 4) = 0.4

theorem original_mixture_percentage (h1 : condition1 a w) (h2 : condition2 a w) : (a / (a + w)) * 100 = 36 :=
by
sorry

end NUMINAMATH_GPT_original_mixture_percentage_l1805_180558


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l1805_180589

-- Proof for part 1
theorem problem1 (α : ℝ) (h : Real.tan α = 3) :
  (3 * Real.sin α + Real.cos α) / (Real.sin α - 2 * Real.cos α) = 10 :=
sorry

-- Proof for part 2
theorem problem2 (α : ℝ) :
  (-Real.sin (Real.pi + α) + Real.sin (-α) - Real.tan (2 * Real.pi + α)) / 
  (Real.tan (α + Real.pi) + Real.cos (-α) + Real.cos (Real.pi - α)) = -1 :=
sorry

-- Proof for part 3
theorem problem3 (α : ℝ) (h : Real.sin α + Real.cos α = 1 / 2) (hα : 0 < α ∧ α < Real.pi) :
  Real.sin α * Real.cos α = -3 / 8 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l1805_180589


namespace NUMINAMATH_GPT_cards_ratio_l1805_180524

theorem cards_ratio (b_c : ℕ) (m_c : ℕ) (m_l : ℕ) (m_g : ℕ) 
  (h1 : b_c = 20) 
  (h2 : m_c = b_c + 8) 
  (h3 : m_l = 14) 
  (h4 : m_g = m_c - m_l) : 
  m_g / m_c = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cards_ratio_l1805_180524


namespace NUMINAMATH_GPT_fraction_value_l1805_180579

theorem fraction_value :
  (1^4 + 2009^4 + 2010^4) / (1^2 + 2009^2 + 2010^2) = 4038091 := by
  sorry

end NUMINAMATH_GPT_fraction_value_l1805_180579


namespace NUMINAMATH_GPT_distance_to_directrix_l1805_180599

theorem distance_to_directrix (x y d : ℝ) (a b c : ℝ) (F1 F2 M : ℝ × ℝ)
  (h_ellipse : x^2 / 25 + y^2 / 9 = 1)
  (h_a : a = 5)
  (h_b : b = 3)
  (h_c : c = 4)
  (h_M_on_ellipse : M.snd^2 / (a^2) + M.fst^2 / (b^2) = 1)
  (h_dist_F1M : dist M F1 = 8) :
  d = 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_distance_to_directrix_l1805_180599


namespace NUMINAMATH_GPT_find_divisor_l1805_180535

def remainder : Nat := 1
def quotient : Nat := 54
def dividend : Nat := 217

theorem find_divisor : ∃ divisor : Nat, (dividend = divisor * quotient + remainder) ∧ divisor = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l1805_180535


namespace NUMINAMATH_GPT_martha_cards_l1805_180510

theorem martha_cards (start_cards : ℕ) : start_cards + 76 = 79 → start_cards = 3 :=
by
  sorry

end NUMINAMATH_GPT_martha_cards_l1805_180510


namespace NUMINAMATH_GPT_max_divisor_of_five_consecutive_integers_l1805_180542

theorem max_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intros n
  sorry

end NUMINAMATH_GPT_max_divisor_of_five_consecutive_integers_l1805_180542


namespace NUMINAMATH_GPT_set_equivalence_l1805_180585

open Set

def set_A : Set ℝ := { x | x^2 - 2 * x > 0 }
def set_B : Set ℝ := { y | ∃ x : ℝ, y = 2^x }

theorem set_equivalence : (univ \ set_B) ∪ set_A = (Iic 1) ∪ Ioi 2 :=
sorry

end NUMINAMATH_GPT_set_equivalence_l1805_180585


namespace NUMINAMATH_GPT_red_cars_in_lot_l1805_180544

theorem red_cars_in_lot (B : ℕ) (hB : B = 90) (ratio_condition : 3 * B = 8 * R) : R = 33 :=
by
  -- Given
  have h1 : B = 90 := hB
  have h2 : 3 * B = 8 * R := ratio_condition

  -- To solve
  sorry

end NUMINAMATH_GPT_red_cars_in_lot_l1805_180544


namespace NUMINAMATH_GPT_number_of_larger_planes_l1805_180572

variable (S L : ℕ)
variable (h1 : S + L = 4)
variable (h2 : 130 * S + 145 * L = 550)

theorem number_of_larger_planes : L = 2 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_number_of_larger_planes_l1805_180572


namespace NUMINAMATH_GPT_base6_arithmetic_l1805_180529

theorem base6_arithmetic :
  let a := 4512
  let b := 2324
  let c := 1432
  let base := 6
  let a_b10 := 4 * base^3 + 5 * base^2 + 1 * base + 2
  let b_b10 := 2 * base^3 + 3 * base^2 + 2 * base + 4
  let c_b10 := 1 * base^3 + 4 * base^2 + 3 * base + 2
  let result_b10 := a_b10 - b_b10 + c_b10
  let result_base6 := 4020
  (result_b10 / base^3) % base = 4 ∧
  (result_b10 / base^2) % base = 0 ∧
  (result_b10 / base) % base = 2 ∧
  result_b10 % base = 0 →
  result_base6 = 4020 := by
  sorry

end NUMINAMATH_GPT_base6_arithmetic_l1805_180529


namespace NUMINAMATH_GPT_focus_of_parabola_tangent_to_circle_directrix_l1805_180540

theorem focus_of_parabola_tangent_to_circle_directrix :
  ∃ p : ℝ, p > 0 ∧
  (∃ (x y : ℝ), x ^ 2 + y ^ 2 - 6 * x - 7 = 0 ∧
  ∀ x y : ℝ, y ^ 2 = 2 * p * x → x = -p) →
  (1, 0) = (p, 0) :=
by
  sorry

end NUMINAMATH_GPT_focus_of_parabola_tangent_to_circle_directrix_l1805_180540


namespace NUMINAMATH_GPT_alice_wins_chomp_l1805_180576

def symmetrical_strategy (n : ℕ) : Prop :=
  ∃ strategy : (ℕ × ℕ) → (ℕ × ℕ), 
  (∀ turn : ℕ × ℕ, 
    strategy turn = 
      if turn = (1,1) then (1,1)
      else if turn.fst = 2 ∧ turn.snd = 2 then (2,2)
      else if turn.fst = 1 then (turn.snd, 1)
      else (1, turn.fst)) 

theorem alice_wins_chomp (n : ℕ) (h : 1 ≤ n) : 
  symmetrical_strategy n := 
sorry

end NUMINAMATH_GPT_alice_wins_chomp_l1805_180576


namespace NUMINAMATH_GPT_range_of_m_l1805_180534

noncomputable def f (x m a : ℝ) : ℝ := Real.exp (x + 1) - m * a
noncomputable def g (x a : ℝ) : ℝ := a * Real.exp x - x

theorem range_of_m (h : ∃ a : ℝ, ∀ x : ℝ, f x m a ≤ g x a) : m ≥ -1 / Real.exp 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1805_180534


namespace NUMINAMATH_GPT_find_k_l1805_180573

theorem find_k (x y k : ℝ) 
  (line1 : y = 3 * x + 2) 
  (line2 : y = -4 * x - 14) 
  (line3 : y = 2 * x + k) :
  k = -2 / 7 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_k_l1805_180573


namespace NUMINAMATH_GPT_problem_statement_l1805_180506

-- Definition of sum of digits function
def S (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

-- Definition of the function f₁
def f₁ (k : ℕ) : ℕ :=
  (S k) ^ 2

-- Definition of the function fₙ₊₁
def f : ℕ → ℕ → ℕ
| 0, k => k
| (n+1), k => f₁ (f n k)

-- Theorem stating the proof problem
theorem problem_statement : f 2005 (2 ^ 2006) = 169 :=
  sorry

end NUMINAMATH_GPT_problem_statement_l1805_180506


namespace NUMINAMATH_GPT_car_speed_is_48_l1805_180527

theorem car_speed_is_48 {v : ℝ} : (3600 / v = 75) → v = 48 := 
by {
  sorry
}

end NUMINAMATH_GPT_car_speed_is_48_l1805_180527


namespace NUMINAMATH_GPT_problem1_problem2_l1805_180523

-- Definitions of the sets A, B, C
def A (a : ℝ) : Set ℝ := { x | x^2 - a*x + a^2 - 12 = 0 }
def B : Set ℝ := { x | x^2 - 2*x - 8 = 0 }
def C (m : ℝ) : Set ℝ := { x | m*x + 1 = 0 }

-- Problem 1: If A = B, then a = 2
theorem problem1 (a : ℝ) (h : A a = B) : a = 2 := sorry

-- Problem 2: If B ∪ C m = B, then m ∈ {-1/4, 0, 1/2}
theorem problem2 (m : ℝ) (h : B ∪ C m = B) : m = -1/4 ∨ m = 0 ∨ m = 1/2 := sorry

end NUMINAMATH_GPT_problem1_problem2_l1805_180523


namespace NUMINAMATH_GPT_waiter_tables_l1805_180565

theorem waiter_tables (total_customers : ℕ) (customers_left : ℕ) (people_per_table : ℕ) (remaining_customers : ℕ) (number_of_tables : ℕ) 
  (h1 : total_customers = 22)
  (h2 : customers_left = 14)
  (h3 : people_per_table = 4)
  (h4 : remaining_customers = total_customers - customers_left)
  (h5 : number_of_tables = remaining_customers / people_per_table) :
  number_of_tables = 2 :=
by
  sorry

end NUMINAMATH_GPT_waiter_tables_l1805_180565


namespace NUMINAMATH_GPT_quadratic_function_min_value_in_interval_l1805_180554

noncomputable def quadratic_function (x : ℝ) : ℝ :=
  x^2 - 6 * x + 10

theorem quadratic_function_min_value_in_interval :
  ∀ (x : ℝ), 2 ≤ x ∧ x < 5 → (∃ min_val : ℝ, min_val = 1) ∧ (∀ upper_bound : ℝ, ∃ x0 : ℝ, x0 < 5 ∧ quadratic_function x0 > upper_bound) := 
by
  sorry

end NUMINAMATH_GPT_quadratic_function_min_value_in_interval_l1805_180554


namespace NUMINAMATH_GPT_inequality_proof_l1805_180519

variable (a b c : ℝ)

theorem inequality_proof
  (h1 : a > b) :
  a * c^2 ≥ b * c^2 := 
sorry

end NUMINAMATH_GPT_inequality_proof_l1805_180519


namespace NUMINAMATH_GPT_ratio_of_speeds_l1805_180512

theorem ratio_of_speeds (v_A v_B : ℝ) (h1 : 500 / v_A = 400 / v_B) : v_A / v_B = 5 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_speeds_l1805_180512


namespace NUMINAMATH_GPT_associate_professor_pencils_l1805_180550

theorem associate_professor_pencils
  (A B P : ℕ)
  (h1 : A + B = 7)
  (h2 : P * A + B = 10)
  (h3 : A + 2 * B = 11) :
  P = 2 :=
by {
  -- Variables declarations and assumptions
  -- Combine and manipulate equations to prove P = 2
  sorry
}

end NUMINAMATH_GPT_associate_professor_pencils_l1805_180550


namespace NUMINAMATH_GPT_number_of_ways_to_divide_l1805_180501

def shape_17_cells : Type := sorry -- We would define the structure of the shape here
def checkerboard_pattern : shape_17_cells → Prop := sorry -- The checkerboard pattern condition
def num_black_cells (s : shape_17_cells) : ℕ := 9 -- Number of black cells
def num_gray_cells (s : shape_17_cells) : ℕ := 8 -- Number of gray cells
def divides_into (s : shape_17_cells) (rectangles : ℕ) (squares : ℕ) : Prop := sorry -- Division condition

theorem number_of_ways_to_divide (s : shape_17_cells) (h1 : checkerboard_pattern s) (h2 : divides_into s 8 1) :
  num_black_cells s = 9 ∧ num_gray_cells s = 8 → 
  (∃ ways : ℕ, ways = 10) := 
sorry

end NUMINAMATH_GPT_number_of_ways_to_divide_l1805_180501


namespace NUMINAMATH_GPT_G_is_odd_l1805_180549

noncomputable def G (F : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ :=
  F x * (1 / (a^x - 1) + 1 / 2)

theorem G_is_odd (F : ℝ → ℝ) (a : ℝ) (h : a > 0) (h₁ : a ≠ 1) (h₂ : ∀ x : ℝ, F (-x) = - F x) :
  ∀ x : ℝ, G F a (-x) = - G F a x :=
by 
  sorry

end NUMINAMATH_GPT_G_is_odd_l1805_180549


namespace NUMINAMATH_GPT_distribute_balls_into_boxes_l1805_180588

theorem distribute_balls_into_boxes : 
  let n := 5
  let k := 4
  (n.choose (k - 1) + k - 1).choose (k - 1) = 56 :=
by
  sorry

end NUMINAMATH_GPT_distribute_balls_into_boxes_l1805_180588


namespace NUMINAMATH_GPT_total_memory_space_l1805_180514

def morning_songs : Nat := 10
def afternoon_songs : Nat := 15
def night_songs : Nat := 3
def song_size : Nat := 5

theorem total_memory_space : (morning_songs + afternoon_songs + night_songs) * song_size = 140 := by
  sorry

end NUMINAMATH_GPT_total_memory_space_l1805_180514


namespace NUMINAMATH_GPT_sin_monotonically_decreasing_l1805_180560

open Real

theorem sin_monotonically_decreasing (f : ℝ → ℝ) (x : ℝ) :
  (∀ x, f x = sin (2 * x + π / 3)) →
  (0 ≤ x ∧ x ≤ π) →
  (∀ x, (π / 12) ≤ x ∧ x ≤ (7 * π / 12)) →
  ∀ x y, (x < y → f y ≤ f x) := by
  sorry

end NUMINAMATH_GPT_sin_monotonically_decreasing_l1805_180560


namespace NUMINAMATH_GPT_segment_length_reflection_l1805_180586

theorem segment_length_reflection (Z : ℝ×ℝ) (Z' : ℝ×ℝ) (hx : Z = (5, 2)) (hx' : Z' = (5, -2)) :
  dist Z Z' = 4 := by
  sorry

end NUMINAMATH_GPT_segment_length_reflection_l1805_180586


namespace NUMINAMATH_GPT_math_lovers_l1805_180568

/-- The proof problem: 
Given 1256 students in total and the difference of 408 between students who like math and others,
prove that the number of students who like math is 424, given that students who like math are fewer than 500.
--/
theorem math_lovers (M O : ℕ) (h1 : M + O = 1256) (h2: O - M = 408) (h3 : M < 500) : M = 424 :=
by
  sorry

end NUMINAMATH_GPT_math_lovers_l1805_180568


namespace NUMINAMATH_GPT_log_inequality_l1805_180517

noncomputable def a := Real.log 6 / Real.log 3
noncomputable def b := Real.log 10 / Real.log 5
noncomputable def c := Real.log 14 / Real.log 7

theorem log_inequality :
  a > b ∧ b > c :=
by
  sorry

end NUMINAMATH_GPT_log_inequality_l1805_180517


namespace NUMINAMATH_GPT_obtuse_angle_only_dihedral_planar_l1805_180531

/-- Given the range of three types of angles, prove that only the dihedral angle's planar angle can be obtuse. -/
theorem obtuse_angle_only_dihedral_planar 
  (α : ℝ) (β : ℝ) (γ : ℝ) 
  (hα : 0 < α ∧ α ≤ 90)
  (hβ : 0 ≤ β ∧ β ≤ 90)
  (hγ : 0 ≤ γ ∧ γ < 180) : 
  (90 < γ ∧ (¬(90 < α)) ∧ (¬(90 < β))) :=
by 
  sorry

end NUMINAMATH_GPT_obtuse_angle_only_dihedral_planar_l1805_180531


namespace NUMINAMATH_GPT_school_survey_l1805_180596

theorem school_survey (n k smallest largest : ℕ) (h1 : n = 24) (h2 : k = 4) (h3 : smallest = 3) (h4 : 1 ≤ smallest ∧ smallest ≤ n) (h5 : largest - smallest = (k - 1) * (n / k)) : 
  largest = 21 :=
by {
  sorry
}

end NUMINAMATH_GPT_school_survey_l1805_180596


namespace NUMINAMATH_GPT_negation_of_P_l1805_180580

def P (x : ℝ) : Prop := x^2 + x - 1 < 0

theorem negation_of_P : (¬ ∀ x, P x) ↔ ∃ x : ℝ, x^2 + x - 1 ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_P_l1805_180580


namespace NUMINAMATH_GPT_largest_of_A_B_C_l1805_180553

noncomputable def A : ℝ := (3003 / 3002) + (3003 / 3004)
noncomputable def B : ℝ := (3003 / 3004) + (3005 / 3004)
noncomputable def C : ℝ := (3004 / 3003) + (3004 / 3005)

theorem largest_of_A_B_C : A > B ∧ A ≥ C := by
  sorry

end NUMINAMATH_GPT_largest_of_A_B_C_l1805_180553


namespace NUMINAMATH_GPT_ratio_shorter_to_longer_l1805_180543

-- Define the total length and the length of the shorter piece
def total_length : ℕ := 90
def shorter_length : ℕ := 20

-- Define the length of the longer piece
def longer_length : ℕ := total_length - shorter_length

-- Define the ratio of shorter piece to longer piece
def ratio := shorter_length / longer_length

-- The target statement to prove
theorem ratio_shorter_to_longer : ratio = 2 / 7 := by
  sorry

end NUMINAMATH_GPT_ratio_shorter_to_longer_l1805_180543


namespace NUMINAMATH_GPT_greater_quadratic_solution_l1805_180597

theorem greater_quadratic_solution : ∀ (x : ℝ), x^2 + 15 * x - 54 = 0 → x = -18 ∨ x = 3 →
  max (-18) 3 = 3 := by
  sorry

end NUMINAMATH_GPT_greater_quadratic_solution_l1805_180597


namespace NUMINAMATH_GPT_angle_BAD_measure_l1805_180577

theorem angle_BAD_measure (D_A_C : ℝ) (AB_AC : AB = AC) (AD_BD : AD = BD) (h : D_A_C = 39) :
  B_A_D = 70.5 :=
by sorry

end NUMINAMATH_GPT_angle_BAD_measure_l1805_180577


namespace NUMINAMATH_GPT_triangle_inequality_l1805_180545

theorem triangle_inequality (a b c : ℝ) (habc : a + b > c ∧ a + c > b ∧ b + c > a) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l1805_180545


namespace NUMINAMATH_GPT_arithmetic_sequence_first_term_and_common_difference_l1805_180525

def a_n (n : ℕ) : ℕ := 2 * n + 5

theorem arithmetic_sequence_first_term_and_common_difference :
  a_n 1 = 7 ∧ ∀ n : ℕ, a_n (n + 1) - a_n n = 2 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_first_term_and_common_difference_l1805_180525


namespace NUMINAMATH_GPT_distinguishable_arrangements_l1805_180552

theorem distinguishable_arrangements :
  let brown := 1
  let purple := 1
  let green := 3
  let yellow := 3
  let blue := 2
  let total := brown + purple + green + yellow + blue
  (Nat.factorial total) / (Nat.factorial brown * Nat.factorial purple * Nat.factorial green * Nat.factorial yellow * Nat.factorial blue) = 50400 := 
by
  let brown := 1
  let purple := 1
  let green := 3
  let yellow := 3
  let blue := 2
  let total := brown + purple + green + yellow + blue
  sorry

end NUMINAMATH_GPT_distinguishable_arrangements_l1805_180552


namespace NUMINAMATH_GPT_lila_will_have_21_tulips_l1805_180592

def tulip_orchid_ratio := 3 / 4

def initial_orchids := 16

def added_orchids := 12

def total_orchids : ℕ := initial_orchids + added_orchids

def groups_of_orchids : ℕ := total_orchids / 4

def total_tulips : ℕ := 3 * groups_of_orchids

theorem lila_will_have_21_tulips :
  total_tulips = 21 := by
  sorry

end NUMINAMATH_GPT_lila_will_have_21_tulips_l1805_180592


namespace NUMINAMATH_GPT_rectangle_area_error_percentage_l1805_180526

theorem rectangle_area_error_percentage 
  (L W : ℝ)
  (measured_length : ℝ := L * 1.16)
  (measured_width : ℝ := W * 0.95)
  (actual_area : ℝ := L * W)
  (measured_area : ℝ := measured_length * measured_width) :
  ((measured_area - actual_area) / actual_area) * 100 = 10.2 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_area_error_percentage_l1805_180526


namespace NUMINAMATH_GPT_union_of_A_and_B_l1805_180516

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 1 < x ∧ x < 6}
def union_AB := {x : ℝ | 1 < x ∧ x ≤ 8}

theorem union_of_A_and_B : A ∪ B = union_AB :=
sorry

end NUMINAMATH_GPT_union_of_A_and_B_l1805_180516


namespace NUMINAMATH_GPT_general_equation_of_line_l1805_180583

-- Define the points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (0, 1)

-- Define what it means for a line to pass through two points
def line_through_points (l : ℝ → ℝ → Prop) (A B : ℝ × ℝ) : Prop :=
  l A.1 A.2 ∧ l B.1 B.2

-- Define the equation of line l
def line_l (x y : ℝ) : Prop := x - 2 * y + 2 = 0

-- The theorem that needs to be proven
theorem general_equation_of_line : line_through_points line_l A B := 
by
  sorry

end NUMINAMATH_GPT_general_equation_of_line_l1805_180583


namespace NUMINAMATH_GPT_inequality_proof_l1805_180515

noncomputable def a := Real.log 1 / Real.log 3
noncomputable def b := Real.log 1 / Real.log (1 / 2)
noncomputable def c := (1/2)^(1/3)

theorem inequality_proof : b > c ∧ c > a := 
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l1805_180515


namespace NUMINAMATH_GPT_hcl_reaction_l1805_180557

theorem hcl_reaction
  (stoichiometry : ∀ (HCl NaHCO3 H2O CO2 NaCl : ℕ), HCl = NaHCO3 ∧ H2O = NaHCO3 ∧ CO2 = NaHCO3 ∧ NaCl = NaHCO3)
  (naHCO3_moles : ℕ)
  (reaction_moles : naHCO3_moles = 3) :
  ∃ (HCl_moles : ℕ), HCl_moles = naHCO3_moles :=
by
  sorry

end NUMINAMATH_GPT_hcl_reaction_l1805_180557


namespace NUMINAMATH_GPT_values_of_a_and_b_range_of_c_isosceles_perimeter_l1805_180521

def a : ℝ := 3
def b : ℝ := 4

axiom triangle_ABC (c : ℝ) : 0 < c

noncomputable def equation_condition (a b : ℝ) : Prop :=
  |a-3| + (b-4)^2 = 0

noncomputable def is_valid_c (c : ℝ) : Prop :=
  1 < c ∧ c < 7

theorem values_of_a_and_b (h : equation_condition a b) : a = 3 ∧ b = 4 := sorry

theorem range_of_c (h : equation_condition a b) : is_valid_c c := sorry

noncomputable def isosceles_triangle (c : ℝ) : Prop :=
  c = 4 ∨ c = 3

theorem isosceles_perimeter (h : equation_condition a b) (hc : isosceles_triangle c) : (3 + 3 + 4 = 10) ∨ (4 + 4 + 3 = 11) := sorry

end NUMINAMATH_GPT_values_of_a_and_b_range_of_c_isosceles_perimeter_l1805_180521


namespace NUMINAMATH_GPT_find_circle_eq_find_range_of_dot_product_l1805_180509

open Real
open Set

-- Define the problem conditions
def line_eq (x y : ℝ) : Prop := x - sqrt 3 * y = 4
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the point P inside the circle and condition that |PA|, |PO|, |PB| form a geometric sequence
def geometric_sequence_condition (x y : ℝ) : Prop :=
  sqrt ((x + 2)^2 + y^2) * sqrt ((x - 2)^2 + y^2) = x^2 + y^2

-- Prove the equation of the circle
theorem find_circle_eq :
  (∃ (r : ℝ), ∀ (x y : ℝ), line_eq x y → r = 2) → circle_eq x y :=
by
  -- skipping the proof
  sorry

-- Prove the range of values for the dot product
theorem find_range_of_dot_product :
  (∀ (x y : ℝ), circle_eq x y ∧ geometric_sequence_condition x y) →
  -2 < (x^2 - 1 * y^2 - 1) → (x^2 - 4 + y^2) < 0 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_find_circle_eq_find_range_of_dot_product_l1805_180509


namespace NUMINAMATH_GPT_oz_lost_words_count_l1805_180561
-- We import the necessary library.

-- Define the context.
def total_letters := 69
def forbidden_letter := 7

-- Define function to calculate lost words when a specific letter is forbidden.
def lost_words (total_letters : ℕ) (forbidden_letter : ℕ) : ℕ :=
  let one_letter_lost := 1
  let two_letter_lost := 2 * (total_letters - 1)
  one_letter_lost + two_letter_lost

-- State the theorem.
theorem oz_lost_words_count :
  lost_words total_letters forbidden_letter = 139 :=
by
  sorry

end NUMINAMATH_GPT_oz_lost_words_count_l1805_180561


namespace NUMINAMATH_GPT_zero_point_six_one_eight_method_l1805_180539

theorem zero_point_six_one_eight_method (a b : ℝ) (h : a = 2 ∧ b = 4) : 
  ∃ x₁ x₂, x₁ = a + 0.618 * (b - a) ∧ x₂ = a + b - x₁ ∧ (x₁ = 3.236 ∨ x₂ = 2.764) := by
  sorry

end NUMINAMATH_GPT_zero_point_six_one_eight_method_l1805_180539


namespace NUMINAMATH_GPT_min_value_fraction_108_l1805_180594

noncomputable def min_value_fraction (x y z w : ℝ) : ℝ :=
(x + y) / (x * y * z * w)

theorem min_value_fraction_108 (x y z w : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : w > 0) (h_sum : x + y + z + w = 1) :
  min_value_fraction x y z w = 108 :=
sorry

end NUMINAMATH_GPT_min_value_fraction_108_l1805_180594


namespace NUMINAMATH_GPT_three_digit_number_count_correct_l1805_180520

noncomputable
def count_three_digit_numbers (digits : List ℕ) : ℕ :=
  if h : digits.length = 5 then
    (5 * 4 * 3 : ℕ)
  else
    0

theorem three_digit_number_count_correct :
  count_three_digit_numbers [1, 3, 5, 7, 9] = 60 :=
by
  unfold count_three_digit_numbers
  simp only [List.length, if_pos]
  rfl

end NUMINAMATH_GPT_three_digit_number_count_correct_l1805_180520


namespace NUMINAMATH_GPT_kendall_nickels_l1805_180537

def value_of_quarters (q : ℕ) : ℝ := q * 0.25
def value_of_dimes (d : ℕ) : ℝ := d * 0.10
def value_of_nickels (n : ℕ) : ℝ := n * 0.05

theorem kendall_nickels (q d : ℕ) (total : ℝ) (hq : q = 10) (hd : d = 12) (htotal : total = 4) : 
  ∃ n : ℕ, value_of_nickels n = total - (value_of_quarters q + value_of_dimes d) ∧ n = 6 :=
by
  sorry

end NUMINAMATH_GPT_kendall_nickels_l1805_180537


namespace NUMINAMATH_GPT_parallel_lines_solution_l1805_180593

theorem parallel_lines_solution (a : ℝ) :
  (∀ x y : ℝ, a = 0 → (x + 2 * a * y - 1 = 0 ∧ (2 * a - 1) * x - a * y - 1 = 0) → (x = y)) ∨ 
  (∀ x y : ℝ, a = 1/4 → (x + 2 * a * y - 1 = 0 ∧ (2 * a - 1) * x - a * y - 1 = 0) → (x = y)) :=
sorry

end NUMINAMATH_GPT_parallel_lines_solution_l1805_180593


namespace NUMINAMATH_GPT_simon_number_of_legos_l1805_180530

variable (Kent_legos : ℕ) (Bruce_legos : ℕ) (Simon_legos : ℕ)

def Kent_condition : Prop := Kent_legos = 40
def Bruce_condition : Prop := Bruce_legos = Kent_legos + 20 
def Simon_condition : Prop := Simon_legos = Bruce_legos + (Bruce_legos * 20 / 100)

theorem simon_number_of_legos : Kent_condition Kent_legos ∧ Bruce_condition Kent_legos Bruce_legos ∧ Simon_condition Bruce_legos Simon_legos → Simon_legos = 72 := by
  intros h
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_simon_number_of_legos_l1805_180530


namespace NUMINAMATH_GPT_min_value_of_x_plus_2y_l1805_180598

theorem min_value_of_x_plus_2y {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 8 / x + 1 / y = 1) : x + 2 * y ≥ 18 :=
sorry

end NUMINAMATH_GPT_min_value_of_x_plus_2y_l1805_180598


namespace NUMINAMATH_GPT_a_is_perfect_square_l1805_180502

theorem a_is_perfect_square {a : ℕ} (h : ∀ n : ℕ, ∃ d : ℕ, d ≠ 1 ∧ d % n = 1 ∧ d ∣ n ^ 2 * a - 1) : ∃ k : ℕ, a = k ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_a_is_perfect_square_l1805_180502


namespace NUMINAMATH_GPT_weight_around_59_3_l1805_180513

noncomputable def weight_at_height (height: ℝ) : ℝ := 0.75 * height - 68.2

theorem weight_around_59_3 (x : ℝ) (h : x = 170) : abs (weight_at_height x - 59.3) < 1 :=
by
  sorry

end NUMINAMATH_GPT_weight_around_59_3_l1805_180513


namespace NUMINAMATH_GPT_simplify_correct_l1805_180587

def simplify_expression (a b : ℤ) : ℤ :=
  (30 * a + 70 * b) + (15 * a + 45 * b) - (12 * a + 60 * b)

theorem simplify_correct (a b : ℤ) : simplify_expression a b = 33 * a + 55 * b :=
by 
  sorry -- Proof to be filled in later

end NUMINAMATH_GPT_simplify_correct_l1805_180587


namespace NUMINAMATH_GPT_largest_value_of_c_l1805_180578

theorem largest_value_of_c : ∀ c : ℝ, (3 * c + 6) * (c - 2) = 9 * c → c ≤ 4 :=
by
  intros c hc
  have : (3 * c + 6) * (c - 2) = 9 * c := hc
  sorry

end NUMINAMATH_GPT_largest_value_of_c_l1805_180578


namespace NUMINAMATH_GPT_number_of_triangles_with_perimeter_nine_l1805_180562

theorem number_of_triangles_with_perimeter_nine : 
  ∃ (a b c : ℕ), a + b + c = 9 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a :=
by
  sorry  -- Proof steps are omitted.

end NUMINAMATH_GPT_number_of_triangles_with_perimeter_nine_l1805_180562


namespace NUMINAMATH_GPT_perimeter_of_rectangle_WXYZ_l1805_180574

theorem perimeter_of_rectangle_WXYZ 
  (WE XF EG FH : ℝ)
  (h1 : WE = 10)
  (h2 : XF = 25)
  (h3 : EG = 20)
  (h4 : FH = 50) :
  let p := 53 -- By solving the equivalent problem, where perimeter is simplified to 53/1 which gives p = 53 and q = 1
  let q := 29
  p + q = 102 := 
by
  sorry

end NUMINAMATH_GPT_perimeter_of_rectangle_WXYZ_l1805_180574


namespace NUMINAMATH_GPT_pair_product_not_72_l1805_180591

theorem pair_product_not_72 : (2 * (-36) ≠ 72) :=
by
  sorry

end NUMINAMATH_GPT_pair_product_not_72_l1805_180591


namespace NUMINAMATH_GPT_democrats_ratio_l1805_180556

noncomputable def F : ℕ := 240
noncomputable def M : ℕ := 480
noncomputable def D_F : ℕ := 120
noncomputable def D_M : ℕ := 120

theorem democrats_ratio (total_participants : ℕ := 720)
  (h1 : F + M = total_participants)
  (h2 : D_F = 120)
  (h3 : D_F = 1/2 * F)
  (h4 : D_M = 1/4 * M)
  (h5 : D_F + D_M = 240)
  (h6 : F + M = 720) : (D_F + D_M) / total_participants = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_democrats_ratio_l1805_180556


namespace NUMINAMATH_GPT_arithmetic_prog_sum_l1805_180575

theorem arithmetic_prog_sum (a d : ℕ) (h1 : 15 * a + 105 * d = 60) : 2 * a + 14 * d = 8 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_prog_sum_l1805_180575


namespace NUMINAMATH_GPT_find_p_q_l1805_180541

variable (R : Set ℝ)

def A (p : ℝ) : Set ℝ := {x | x^2 + p * x + 12 = 0}
def B (q : ℝ) : Set ℝ := {x | x^2 - 5 * x + q = 0}

theorem find_p_q 
  (h : (R \ (A p)) ∩ (B q) = {2}) : p + q = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_p_q_l1805_180541


namespace NUMINAMATH_GPT_enlarged_banner_height_l1805_180503

-- Definitions and theorem statement
theorem enlarged_banner_height 
  (original_width : ℝ) 
  (original_height : ℝ) 
  (new_width : ℝ) 
  (scaling_factor : ℝ := new_width / original_width ) 
  (new_height : ℝ := original_height * scaling_factor) 
  (h1 : original_width = 3) 
  (h2 : original_height = 2) 
  (h3 : new_width = 15): 
  new_height = 10 := 
by 
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_enlarged_banner_height_l1805_180503


namespace NUMINAMATH_GPT_cost_price_computer_table_l1805_180511

theorem cost_price_computer_table :
  ∃ CP : ℝ, CP * 1.25 = 5600 ∧ CP = 4480 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_computer_table_l1805_180511


namespace NUMINAMATH_GPT_find_skirts_l1805_180507

variable (blouses : ℕ) (skirts : ℕ) (slacks : ℕ)
variable (blouses_in_hamper : ℕ) (slacks_in_hamper : ℕ) (skirts_in_hamper : ℕ)
variable (clothes_in_hamper : ℕ)

-- Given conditions
axiom h1 : blouses = 12
axiom h2 : slacks = 8
axiom h3 : blouses_in_hamper = (75 * blouses) / 100
axiom h4 : slacks_in_hamper = (25 * slacks) / 100
axiom h5 : skirts_in_hamper = 3
axiom h6 : clothes_in_hamper = blouses_in_hamper + slacks_in_hamper + skirts_in_hamper
axiom h7 : clothes_in_hamper = 11

-- Proof goal: proving the total number of skirts
theorem find_skirts : skirts_in_hamper = (50 * skirts) / 100 → skirts = 6 :=
by sorry

end NUMINAMATH_GPT_find_skirts_l1805_180507


namespace NUMINAMATH_GPT_category_D_cost_after_discount_is_correct_l1805_180564

noncomputable def total_cost : ℝ := 2500
noncomputable def percentage_A : ℝ := 0.30
noncomputable def percentage_B : ℝ := 0.25
noncomputable def percentage_C : ℝ := 0.20
noncomputable def percentage_D : ℝ := 0.25
noncomputable def discount_A : ℝ := 0.03
noncomputable def discount_B : ℝ := 0.05
noncomputable def discount_C : ℝ := 0.07
noncomputable def discount_D : ℝ := 0.10

noncomputable def cost_before_discount_D : ℝ := total_cost * percentage_D
noncomputable def discount_amount_D : ℝ := cost_before_discount_D * discount_D
noncomputable def cost_after_discount_D : ℝ := cost_before_discount_D - discount_amount_D

theorem category_D_cost_after_discount_is_correct : cost_after_discount_D = 562.5 := 
by 
  sorry

end NUMINAMATH_GPT_category_D_cost_after_discount_is_correct_l1805_180564


namespace NUMINAMATH_GPT_total_money_shared_l1805_180571

theorem total_money_shared 
  (A B C D total : ℕ) 
  (h1 : A = 3 * 15)
  (h2 : B = 5 * 15)
  (h3 : C = 6 * 15)
  (h4 : D = 8 * 15)
  (h5 : A = 45) :
  total = A + B + C + D → total = 330 :=
by
  sorry

end NUMINAMATH_GPT_total_money_shared_l1805_180571


namespace NUMINAMATH_GPT_rice_mixture_ratio_l1805_180563

-- Definitions for the given conditions
def cost_per_kg_rice1 : ℝ := 5
def cost_per_kg_rice2 : ℝ := 8.75
def cost_per_kg_mixture : ℝ := 7.50

-- The problem: ratio of two quantities
theorem rice_mixture_ratio (x y : ℝ) (h : cost_per_kg_rice1 * x + cost_per_kg_rice2 * y = 
                                     cost_per_kg_mixture * (x + y)) :
  y / x = 2 := 
sorry

end NUMINAMATH_GPT_rice_mixture_ratio_l1805_180563


namespace NUMINAMATH_GPT_max_saved_houses_l1805_180548

theorem max_saved_houses (n c : ℕ) (h₁ : 1 ≤ c ∧ c ≤ n / 2) : 
  ∃ k, k = n^2 + c^2 - n * c - c :=
by
  sorry

end NUMINAMATH_GPT_max_saved_houses_l1805_180548


namespace NUMINAMATH_GPT_percentage_spent_l1805_180546

theorem percentage_spent (initial_amount remaining_amount : ℝ) 
  (h_initial : initial_amount = 1200) 
  (h_remaining : remaining_amount = 840) : 
  (initial_amount - remaining_amount) / initial_amount * 100 = 30 :=
by
  sorry

end NUMINAMATH_GPT_percentage_spent_l1805_180546


namespace NUMINAMATH_GPT_percent_of_70_is_56_l1805_180505

theorem percent_of_70_is_56 : (70 / 125) * 100 = 56 := by
  sorry

end NUMINAMATH_GPT_percent_of_70_is_56_l1805_180505


namespace NUMINAMATH_GPT_negation_of_forall_l1805_180536

theorem negation_of_forall (h : ¬ ∀ x > 0, Real.exp x > x + 1) : ∃ x > 0, Real.exp x < x + 1 :=
sorry

end NUMINAMATH_GPT_negation_of_forall_l1805_180536


namespace NUMINAMATH_GPT_yoongi_has_fewest_apples_l1805_180559

noncomputable def yoongi_apples : ℕ := 4
noncomputable def yuna_apples : ℕ := 5
noncomputable def jungkook_apples : ℕ := 6 * 3

theorem yoongi_has_fewest_apples : yoongi_apples < yuna_apples ∧ yoongi_apples < jungkook_apples := by
  sorry

end NUMINAMATH_GPT_yoongi_has_fewest_apples_l1805_180559


namespace NUMINAMATH_GPT_keaton_apple_earnings_l1805_180566

theorem keaton_apple_earnings
  (orange_harvest_interval : ℕ)
  (orange_income_per_harvest : ℕ)
  (total_yearly_income : ℕ)
  (orange_harvests_per_year : ℕ)
  (orange_yearly_income : ℕ)
  (apple_yearly_income : ℕ) :
  orange_harvest_interval = 2 →
  orange_income_per_harvest = 50 →
  total_yearly_income = 420 →
  orange_harvests_per_year = 12 / orange_harvest_interval →
  orange_yearly_income = orange_harvests_per_year * orange_income_per_harvest →
  apple_yearly_income = total_yearly_income - orange_yearly_income →
  apple_yearly_income = 120 :=
by
  sorry

end NUMINAMATH_GPT_keaton_apple_earnings_l1805_180566


namespace NUMINAMATH_GPT_find_x_when_y_64_l1805_180555

theorem find_x_when_y_64 (x y k : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y)
  (h_inv_prop : x^3 * y = k) (h_given : x = 2 ∧ y = 8 ∧ k = 64) :
  y = 64 → x = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_x_when_y_64_l1805_180555


namespace NUMINAMATH_GPT_selection_competition_l1805_180547

variables (p q r : Prop)

theorem selection_competition 
  (h1 : p ∨ q) 
  (h2 : ¬ (p ∧ q)) 
  (h3 : ¬ q ∧ r) : p ∧ ¬ q ∧ r :=
by
  sorry

end NUMINAMATH_GPT_selection_competition_l1805_180547


namespace NUMINAMATH_GPT_eval_expr_at_3_l1805_180533

theorem eval_expr_at_3 : (3^2 - 5 * 3 + 6) / (3 - 2) = 0 := by
  sorry

end NUMINAMATH_GPT_eval_expr_at_3_l1805_180533


namespace NUMINAMATH_GPT_sum_is_45_l1805_180551

noncomputable def sum_of_numbers (a b c : ℝ) : ℝ :=
  a + b + c

theorem sum_is_45 {a b c : ℝ} (h1 : ∃ a b c, (a ≤ b ∧ b ≤ c) ∧ b = 10)
  (h2 : (a + b + c) / 3 = a + 20)
  (h3 : (a + b + c) / 3 = c - 25) :
  sum_of_numbers a b c = 45 := 
sorry

end NUMINAMATH_GPT_sum_is_45_l1805_180551


namespace NUMINAMATH_GPT_value_of_expression_l1805_180500

def x : ℝ := 12
def y : ℝ := 7

theorem value_of_expression : (x - y) * (x + y) = 95 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1805_180500


namespace NUMINAMATH_GPT_periodic_functions_exist_l1805_180522

theorem periodic_functions_exist (p1 p2 : ℝ) (h1 : p1 > 0) (h2 : p2 > 0) :
    ∃ (f1 f2 : ℝ → ℝ), (∀ x, f1 (x + p1) = f1 x) ∧ (∀ x, f2 (x + p2) = f2 x) ∧ ∃ T > 0, ∀ x, (f1 - f2) (x + T) = (f1 - f2) x :=
sorry

end NUMINAMATH_GPT_periodic_functions_exist_l1805_180522


namespace NUMINAMATH_GPT_Vanya_bullets_l1805_180584

theorem Vanya_bullets (initial_bullets : ℕ) (hits : ℕ) (shots_made : ℕ) (hits_reward : ℕ) :
  initial_bullets = 10 →
  shots_made = 14 →
  hits = shots_made / 2 →
  hits_reward = 3 →
  (initial_bullets + hits * hits_reward) - shots_made = 17 :=
by
  intros
  sorry

end NUMINAMATH_GPT_Vanya_bullets_l1805_180584


namespace NUMINAMATH_GPT_prove_sets_l1805_180569

noncomputable def A := { y : ℝ | ∃ x : ℝ, y = 3^x }
def B := { x : ℝ | x^2 - 4 ≤ 0 }

theorem prove_sets :
  A ∪ B = { x : ℝ | x ≥ -2 } ∧ A ∩ B = { x : ℝ | 0 < x ∧ x ≤ 2 } :=
by {
  sorry
}

end NUMINAMATH_GPT_prove_sets_l1805_180569


namespace NUMINAMATH_GPT_sum_two_smallest_prime_factors_l1805_180528

theorem sum_two_smallest_prime_factors (n : ℕ) (h : n = 462) : 
  (2 + 3) = 5 := 
by {
  sorry
}

end NUMINAMATH_GPT_sum_two_smallest_prime_factors_l1805_180528


namespace NUMINAMATH_GPT_expand_and_simplify_l1805_180508

theorem expand_and_simplify (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = a^5 + 19 * a^4 + 137 * a^3 + 461 * a^2 + 702 * a + 360 :=
  sorry

end NUMINAMATH_GPT_expand_and_simplify_l1805_180508
