import Mathlib

namespace NUMINAMATH_GPT_determine_properties_range_of_m_l1054_105489

noncomputable def f (a x : ℝ) : ℝ := (a / (a - 1)) * (2^x - 2^(-x))

theorem determine_properties (a : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) :
  (∀ x : ℝ, f a (-x) = -f a x) ∧
  ((0 < a ∧ a < 1) → ∀ x1 x2 : ℝ, x1 < x2 → f a x1 > f a x2) ∧
  (a > 1 → ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2) := 
sorry

theorem range_of_m (a m : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) (h_m_in_I : -1 < m ∧ m < 1) :
  f a (m - 1) + f a m < 0 ↔ 
  ((0 < a ∧ a < 1 → (1 / 2) < m ∧ m < 1) ∧
  (a > 1 → 0 < m ∧ m < (1 / 2))) := 
sorry

end NUMINAMATH_GPT_determine_properties_range_of_m_l1054_105489


namespace NUMINAMATH_GPT_jacob_find_more_l1054_105418

theorem jacob_find_more :
  let initial_shells := 2
  let ed_limpet_shells := 7
  let ed_oyster_shells := 2
  let ed_conch_shells := 4
  let total_shells := 30
  let ed_shells := ed_limpet_shells + ed_oyster_shells + ed_conch_shells + initial_shells
  let jacob_shells := total_shells - ed_shells
  (jacob_shells - ed_limpet_shells - ed_oyster_shells - ed_conch_shells = 2) := 
by 
  sorry

end NUMINAMATH_GPT_jacob_find_more_l1054_105418


namespace NUMINAMATH_GPT_problem1_problem2_l1054_105402

-- Sub-problem 1
theorem problem1 (x y : ℝ) (h1 : 9 * x + 10 * y = 1810) (h2 : 11 * x + 8 * y = 1790) : 
  x - y = -10 := 
sorry

-- Sub-problem 2
theorem problem2 (x y : ℝ) (h1 : 2 * x + 2.5 * y = 1200) (h2 : 1000 * x + 900 * y = 530000) :
  x = 350 ∧ y = 200 := 
sorry

end NUMINAMATH_GPT_problem1_problem2_l1054_105402


namespace NUMINAMATH_GPT_scrap_rate_independence_l1054_105459

theorem scrap_rate_independence (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  (1 - (1 - a) * (1 - b)) = 1 - (1 - a) * (1 - b) :=
by
  sorry

end NUMINAMATH_GPT_scrap_rate_independence_l1054_105459


namespace NUMINAMATH_GPT_pamTotalApples_l1054_105486

-- Define the given conditions
def applesPerGeraldBag : Nat := 40
def applesPerPamBag := 3 * applesPerGeraldBag
def pamBags : Nat := 10

-- Statement to prove
theorem pamTotalApples : pamBags * applesPerPamBag = 1200 :=
by
  sorry

end NUMINAMATH_GPT_pamTotalApples_l1054_105486


namespace NUMINAMATH_GPT_largest_angle_in_triangle_PQR_is_75_degrees_l1054_105438

noncomputable def largest_angle (p q r : ℝ) : ℝ :=
  if p + q + 2 * r = p^2 ∧ p + q - 2 * r = -1 then 
    Real.arccos ((p^2 + q^2 - (p^2 + p*q + (1/2)*q^2)/2) / (2 * p * q)) * (180/Real.pi)
  else 
    0

theorem largest_angle_in_triangle_PQR_is_75_degrees (p q r : ℝ) (h1 : p + q + 2 * r = p^2) (h2 : p + q - 2 * r = -1) :
  largest_angle p q r = 75 :=
by sorry

end NUMINAMATH_GPT_largest_angle_in_triangle_PQR_is_75_degrees_l1054_105438


namespace NUMINAMATH_GPT_weight_of_new_boy_l1054_105449

theorem weight_of_new_boy (W : ℕ) (original_weight : ℕ) (total_new_weight : ℕ)
  (h_original_avg : original_weight = 5 * 35)
  (h_new_avg : total_new_weight = 6 * 36)
  (h_new_weight : total_new_weight = original_weight + W) :
  W = 41 := by
  sorry

end NUMINAMATH_GPT_weight_of_new_boy_l1054_105449


namespace NUMINAMATH_GPT_memory_efficiency_problem_l1054_105469

theorem memory_efficiency_problem (x : ℝ) (hx : x ≠ 0) :
  (100 / x - 100 / (1.2 * x) = 5 / 12) ↔ (100 / x - 100 / ((1 + 0.20) * x) = 5 / 12) :=
by sorry

end NUMINAMATH_GPT_memory_efficiency_problem_l1054_105469


namespace NUMINAMATH_GPT_figure_at_1000th_position_position_of_1000th_diamond_l1054_105497

-- Define the repeating sequence
def repeating_sequence : List String := ["△", "Λ", "◇", "Λ", "⊙", "□"]

-- Lean 4 statement for (a)
theorem figure_at_1000th_position :
  repeating_sequence[(1000 % repeating_sequence.length) - 1] = "Λ" :=
by sorry

-- Define the arithmetic sequence for diamond positions
def diamond_position (n : Nat) : Nat :=
  3 + (n - 1) * 6

-- Lean 4 statement for (b)
theorem position_of_1000th_diamond :
  diamond_position 1000 = 5997 :=
by sorry

end NUMINAMATH_GPT_figure_at_1000th_position_position_of_1000th_diamond_l1054_105497


namespace NUMINAMATH_GPT_simplify_fraction_when_b_equals_4_l1054_105488

theorem simplify_fraction_when_b_equals_4 (b : ℕ) (h : b = 4) : (18 * b^4) / (27 * b^3) = 8 / 3 :=
by {
  -- we use the provided condition to state our theorem goals.
  sorry
}

end NUMINAMATH_GPT_simplify_fraction_when_b_equals_4_l1054_105488


namespace NUMINAMATH_GPT_calculate_f3_minus_f4_l1054_105413

-- Defining the function f and the given conditions
variables (f : ℝ → ℝ)
variable (odd_f : ∀ x, f (-x) = -f x)
variable (periodic_f : ∀ x, f (x + 2) = -f x)
variable (f1 : f 1 = 1)

-- Proving the required equality
theorem calculate_f3_minus_f4 : f 3 - f 4 = -1 :=
by
  sorry

end NUMINAMATH_GPT_calculate_f3_minus_f4_l1054_105413


namespace NUMINAMATH_GPT_volume_is_correct_l1054_105417

def volume_of_box (x : ℝ) : ℝ :=
  (14 - 2 * x) * (10 - 2 * x) * x

theorem volume_is_correct (x : ℝ) :
  volume_of_box x = 140 * x - 48 * x^2 + 4 * x^3 :=
by
  sorry

end NUMINAMATH_GPT_volume_is_correct_l1054_105417


namespace NUMINAMATH_GPT_S7_is_28_l1054_105437

variables {a_n : ℕ → ℤ} -- Sequence definition
variables {S_n : ℕ → ℤ} -- Sum of the first n terms

-- Define an arithmetic sequence condition
def is_arithmetic_sequence (a_n : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a_n (n + 1) = a_n n + d

-- Given conditions
axiom sum_condition : a_n 2 + a_n 4 + a_n 6 = 12
axiom sum_formula (n : ℕ) : S_n n = n * (a_n 1 + a_n n) / 2
axiom arith_seq : is_arithmetic_sequence a_n

-- The statement to be proven
theorem S7_is_28 : S_n 7 = 28 :=
sorry

end NUMINAMATH_GPT_S7_is_28_l1054_105437


namespace NUMINAMATH_GPT_number_of_ordered_triples_l1054_105465

theorem number_of_ordered_triples :
  ∃ n, (∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ Nat.lcm a b = 12 ∧ Nat.gcd b c = 6 ∧ Nat.lcm c a = 24) ∧ n = 4 :=
sorry

end NUMINAMATH_GPT_number_of_ordered_triples_l1054_105465


namespace NUMINAMATH_GPT_union_A_B_l1054_105408

-- Define them as sets
def A : Set ℝ := {x | -3 < x ∧ x ≤ 2}
def B : Set ℝ := {x | -2 < x ∧ x ≤ 3}

-- Statement of the theorem
theorem union_A_B : A ∪ B = {x | -3 < x ∧ x ≤ 3} :=
by
  sorry

end NUMINAMATH_GPT_union_A_B_l1054_105408


namespace NUMINAMATH_GPT_sum_of_squares_l1054_105436

theorem sum_of_squares (a b c : ℝ) :
  a + b + c = 4 → ab + ac + bc = 4 → a^2 + b^2 + c^2 = 8 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l1054_105436


namespace NUMINAMATH_GPT_constants_solution_l1054_105494

theorem constants_solution (A B C : ℝ) :
  (∀ x : ℝ, x ≠ 4 ∧ x ≠ 2 → 
    (5 * x^2 / ((x - 4) * (x - 2)^2) = A / (x - 4) + B / (x - 2) + C / (x - 2)^2)) ↔ 
    (A = 20 ∧ B = -15 ∧ C = -10) :=
by
  sorry

end NUMINAMATH_GPT_constants_solution_l1054_105494


namespace NUMINAMATH_GPT_quadratic_roots_sum_squares_l1054_105498

theorem quadratic_roots_sum_squares {a b : ℝ} 
  (h₁ : a + b = -1) 
  (h₂ : a * b = -5) : 
  2 * a^2 + a + b^2 = 16 :=
by sorry

end NUMINAMATH_GPT_quadratic_roots_sum_squares_l1054_105498


namespace NUMINAMATH_GPT_work_fraction_left_l1054_105451

theorem work_fraction_left (A_days B_days : ℕ) (work_days : ℕ)
  (hA : A_days = 15) (hB : B_days = 20) (h_work : work_days = 3) :
  1 - (work_days * ((1 / A_days) + (1 / B_days))) = 13 / 20 :=
by
  rw [hA, hB, h_work]
  simp
  sorry

end NUMINAMATH_GPT_work_fraction_left_l1054_105451


namespace NUMINAMATH_GPT_numberOfBooks_correct_l1054_105480

variable (totalWeight : ℕ) (weightPerBook : ℕ)

def numberOfBooks (totalWeight weightPerBook : ℕ) : ℕ :=
  totalWeight / weightPerBook

theorem numberOfBooks_correct (h1 : totalWeight = 42) (h2 : weightPerBook = 3) :
  numberOfBooks totalWeight weightPerBook = 14 := by
  sorry

end NUMINAMATH_GPT_numberOfBooks_correct_l1054_105480


namespace NUMINAMATH_GPT_degree_to_radian_l1054_105419

theorem degree_to_radian (deg : ℝ) (h : deg = 50) : deg * (Real.pi / 180) = (5 / 18) * Real.pi :=
by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_degree_to_radian_l1054_105419


namespace NUMINAMATH_GPT_remainder_when_divided_by_6_l1054_105407

theorem remainder_when_divided_by_6 (a : ℕ) (h1 : a % 2 = 1) (h2 : a % 3 = 2) : a % 6 = 5 :=
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_6_l1054_105407


namespace NUMINAMATH_GPT_h_at_7_over_5_eq_0_l1054_105466

def h (x : ℝ) : ℝ := 5 * x - 7

theorem h_at_7_over_5_eq_0 : h (7 / 5) = 0 := 
by 
  sorry

end NUMINAMATH_GPT_h_at_7_over_5_eq_0_l1054_105466


namespace NUMINAMATH_GPT_find_x_in_sequence_l1054_105493

theorem find_x_in_sequence :
  ∃ x : ℕ, x = 32 ∧
    2 + 3 = 5 ∧
    5 + 6 = 11 ∧
    11 + 9 = 20 ∧
    20 + (9 + 3) = x ∧
    x + (9 + 3 + 3) = 47 :=
by
  sorry

end NUMINAMATH_GPT_find_x_in_sequence_l1054_105493


namespace NUMINAMATH_GPT_smallest_number_divisible_by_11_and_conditional_modulus_l1054_105476

theorem smallest_number_divisible_by_11_and_conditional_modulus :
  ∃ n : ℕ, (n % 11 = 0) ∧ (n % 3 = 2) ∧ (n % 4 = 2) ∧ (n % 5 = 2) ∧ (n % 6 = 2) ∧ (n % 7 = 2) ∧ n = 2102 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_divisible_by_11_and_conditional_modulus_l1054_105476


namespace NUMINAMATH_GPT_stones_required_correct_l1054_105442

/- 
Given:
- The hall measures 36 meters long and 15 meters broad.
- Each stone measures 6 decimeters by 5 decimeters.

We need to prove that the number of stones required to pave the hall is 1800.
-/
noncomputable def stones_required 
  (hall_length_m : ℕ) 
  (hall_breadth_m : ℕ) 
  (stone_length_dm : ℕ) 
  (stone_breadth_dm : ℕ) : ℕ :=
  (hall_length_m * 10) * (hall_breadth_m * 10) / (stone_length_dm * stone_breadth_dm)

theorem stones_required_correct : 
  stones_required 36 15 6 5 = 1800 :=
by 
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_stones_required_correct_l1054_105442


namespace NUMINAMATH_GPT_intersect_single_point_l1054_105456

theorem intersect_single_point (k : ℝ) :
  (∃ x : ℝ, (x^2 + k * x + 1 = 0) ∧
   ∀ x y : ℝ, (x^2 + k * x + 1 = 0 → y^2 + k * y + 1 = 0 → x = y))
  ↔ (k = 2 ∨ k = -2) :=
by
  sorry

end NUMINAMATH_GPT_intersect_single_point_l1054_105456


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_of_quadratic_l1054_105473

noncomputable def sum_of_squares_of_roots (p q : ℝ) (a b : ℝ) : Prop :=
  a^2 + b^2 = 4 * p^2 - 6 * q

theorem sum_of_squares_of_roots_of_quadratic
  (p q a b : ℝ)
  (h1 : a + b = 2 * p / 3)
  (h2 : a * b = q / 3)
  (h3 : a * a + b * b = 4 * p^2 - 6 * q) :
  sum_of_squares_of_roots p q a b :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_of_quadratic_l1054_105473


namespace NUMINAMATH_GPT_correct_option_l1054_105405

theorem correct_option :
  (3 * a^2 - a^2 = 2 * a^2) ∧
  (¬ (a^2 * a^3 = a^6)) ∧
  (¬ ((3 * a)^2 = 6 * a^2)) ∧
  (¬ (a^6 / a^3 = a^2)) :=
by
  -- We only need to state the theorem; the proof details are omitted per the instructions.
  sorry

end NUMINAMATH_GPT_correct_option_l1054_105405


namespace NUMINAMATH_GPT_find_multiplier_l1054_105440

theorem find_multiplier (x y : ℝ) (hx : x = 0.42857142857142855) (hx_nonzero : x ≠ 0) (h_eq : (x * y) / 7 = x^2) : y = 3 :=
sorry

end NUMINAMATH_GPT_find_multiplier_l1054_105440


namespace NUMINAMATH_GPT_josiah_hans_age_ratio_l1054_105499

theorem josiah_hans_age_ratio (H : ℕ) (J : ℕ) (hH : H = 15) (hSum : (J + 3) + (H + 3) = 66) : J / H = 3 :=
by
  sorry

end NUMINAMATH_GPT_josiah_hans_age_ratio_l1054_105499


namespace NUMINAMATH_GPT_range_of_m_if_not_p_and_q_l1054_105431

def p (m : ℝ) : Prop := 2 < m

def q (m : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 - 4 * m * x + 4 * m - 3 ≥ 0

theorem range_of_m_if_not_p_and_q (m : ℝ) : ¬ p m ∧ q m → 1 ≤ m ∧ m ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_if_not_p_and_q_l1054_105431


namespace NUMINAMATH_GPT_evaluate_fraction_l1054_105458

open Complex

theorem evaluate_fraction (a b : ℂ) (h : a ≠ 0 ∧ b ≠ 0) (h_eq : a^2 - a*b + b^2 = 0) : 
  (a^6 + b^6) / (a + b)^6 = 1 / 18 := by
  sorry

end NUMINAMATH_GPT_evaluate_fraction_l1054_105458


namespace NUMINAMATH_GPT_find_min_value_l1054_105404

noncomputable def expression (x : ℝ) : ℝ :=
  (Real.sin x ^ 8 + Real.cos x ^ 8 + 2) / (Real.sin x ^ 6 + Real.cos x ^ 6 + 2)

theorem find_min_value : ∃ x : ℝ, expression x = 5 / 4 :=
sorry

end NUMINAMATH_GPT_find_min_value_l1054_105404


namespace NUMINAMATH_GPT_zero_function_is_uniq_l1054_105475

theorem zero_function_is_uniq (f : ℝ → ℝ) :
  (∀ (x : ℝ) (hx : x ≠ 0) (y : ℝ), f (x^2 + y) ≥ (1/x + 1) * f y) → 
  (∀ x, f x = 0) :=
by
  sorry

end NUMINAMATH_GPT_zero_function_is_uniq_l1054_105475


namespace NUMINAMATH_GPT_fraction_halfway_between_3_4_and_5_6_is_19_24_l1054_105483

noncomputable def fraction_halfway (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between_3_4_and_5_6_is_19_24 :
  fraction_halfway (3 / 4) (5 / 6) = 19 / 24 :=
by
  sorry

end NUMINAMATH_GPT_fraction_halfway_between_3_4_and_5_6_is_19_24_l1054_105483


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1054_105423

theorem solution_set_of_inequality : 
  { x : ℝ | (3 - 2 * x) * (x + 1) ≤ 0 } = { x : ℝ | -1 ≤ x ∧ x ≤ 3 / 2 } :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1054_105423


namespace NUMINAMATH_GPT_total_distance_travelled_l1054_105426

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

end NUMINAMATH_GPT_total_distance_travelled_l1054_105426


namespace NUMINAMATH_GPT_happy_boys_count_l1054_105429

def total_children := 60
def happy_children := 30
def sad_children := 10
def neither_happy_nor_sad_children := total_children - happy_children - sad_children

def total_boys := 19
def total_girls := 41
def sad_girls := 4
def neither_happy_nor_sad_boys := 7

def sad_boys := sad_children - sad_girls

theorem happy_boys_count :
  total_boys - sad_boys - neither_happy_nor_sad_boys = 6 :=
by
  sorry

end NUMINAMATH_GPT_happy_boys_count_l1054_105429


namespace NUMINAMATH_GPT_prime_remainder_l1054_105490

theorem prime_remainder (p : ℕ) (k : ℕ) (h1 : Prime p) (h2 : p > 3) :
  (∃ k, p = 6 * k + 1 ∧ (p^3 + 17) % 24 = 18) ∨
  (∃ k, p = 6 * k - 1 ∧ (p^3 + 17) % 24 = 16) :=
by
  sorry

end NUMINAMATH_GPT_prime_remainder_l1054_105490


namespace NUMINAMATH_GPT_number_exceeds_its_part_l1054_105455

theorem number_exceeds_its_part (x : ℝ) (h : x = 3/8 * x + 25) : x = 40 :=
by sorry

end NUMINAMATH_GPT_number_exceeds_its_part_l1054_105455


namespace NUMINAMATH_GPT_cost_flying_X_to_Y_l1054_105444

def distance_XY : ℝ := 4500 -- Distance from X to Y in km
def cost_per_km_flying : ℝ := 0.12 -- Cost per km for flying in dollars
def booking_fee_flying : ℝ := 120 -- Booking fee for flying in dollars

theorem cost_flying_X_to_Y : 
    distance_XY * cost_per_km_flying + booking_fee_flying = 660 := by
  sorry

end NUMINAMATH_GPT_cost_flying_X_to_Y_l1054_105444


namespace NUMINAMATH_GPT_teammates_of_oliver_l1054_105422

-- Define the player characteristics
structure Player :=
  (name   : String)
  (eyes   : String)
  (hair   : String)

-- Define the list of players with their given characteristics
def players : List Player := [
  {name := "Daniel", eyes := "Green", hair := "Red"},
  {name := "Oliver", eyes := "Gray", hair := "Brown"},
  {name := "Mia", eyes := "Gray", hair := "Red"},
  {name := "Ella", eyes := "Green", hair := "Brown"},
  {name := "Leo", eyes := "Green", hair := "Red"},
  {name := "Zoe", eyes := "Green", hair := "Brown"}
]

-- Define the condition for being on the same team
def same_team (p1 p2 : Player) : Bool :=
  (p1.eyes = p2.eyes && p1.hair ≠ p2.hair) || (p1.eyes ≠ p2.eyes && p1.hair = p2.hair)

-- Define the criterion to check if two players are on the same team as Oliver
def is_teammate_of_oliver (p : Player) : Bool :=
  let oliver := players[1] -- Oliver is the second player in the list
  same_team oliver p

-- Formal proof statement
theorem teammates_of_oliver : 
  is_teammate_of_oliver players[2] = true ∧ is_teammate_of_oliver players[3] = true :=
by
  -- Provide the intended proof here
  sorry

end NUMINAMATH_GPT_teammates_of_oliver_l1054_105422


namespace NUMINAMATH_GPT_quadratic_no_real_roots_l1054_105420

theorem quadratic_no_real_roots (c : ℝ) : 
  (∀ x : ℝ, ¬(x^2 + x - c = 0)) ↔ c < -1/4 := 
sorry

end NUMINAMATH_GPT_quadratic_no_real_roots_l1054_105420


namespace NUMINAMATH_GPT_y_is_75_percent_of_x_l1054_105446

variable (x y z : ℝ)

-- Conditions
def condition1 : Prop := 0.45 * z = 0.72 * y
def condition2 : Prop := z = 1.20 * x

-- Theorem to prove y = 0.75 * x
theorem y_is_75_percent_of_x (h1 : condition1 z y) (h2 : condition2 x z) : y = 0.75 * x :=
by sorry

end NUMINAMATH_GPT_y_is_75_percent_of_x_l1054_105446


namespace NUMINAMATH_GPT_focus_of_parabola_l1054_105400

theorem focus_of_parabola (x y : ℝ) : (y^2 + 4 * x = 0) → (x = -1 ∧ y = 0) :=
by sorry

end NUMINAMATH_GPT_focus_of_parabola_l1054_105400


namespace NUMINAMATH_GPT_xiaoming_money_l1054_105474

open Real

noncomputable def verify_money_left (M P_L : ℝ) : Prop := M = 12 * P_L

noncomputable def verify_money_right (M P_R : ℝ) : Prop := M = 14 * P_R

noncomputable def price_relationship (P_L P_R : ℝ) : Prop := P_R = P_L - 1

theorem xiaoming_money (M P_L P_R : ℝ) 
  (h1 : verify_money_left M P_L) 
  (h2 : verify_money_right M P_R) 
  (h3 : price_relationship P_L P_R) : 
  M = 84 := 
  by
  sorry

end NUMINAMATH_GPT_xiaoming_money_l1054_105474


namespace NUMINAMATH_GPT_simplify_expression_l1054_105411

theorem simplify_expression (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1054_105411


namespace NUMINAMATH_GPT_banana_price_l1054_105472

theorem banana_price (b : ℝ) : 
    (∃ x : ℕ, 0.70 * x + b * (9 - x) = 5.60 ∧ x + (9 - x) = 9) → b = 0.60 :=
by
  intro h
  obtain ⟨x, hx1, hx2⟩ := h
  -- equations to work with:
  -- 0.70 * x + b * (9 - x) = 5.60
  -- x + (9 - x) = 9
  sorry

end NUMINAMATH_GPT_banana_price_l1054_105472


namespace NUMINAMATH_GPT_perfect_square_polynomial_l1054_105463

theorem perfect_square_polynomial (m : ℝ) :
  (∃ f : ℝ → ℝ, ∀ x, x^2 - (m + 1) * x + 1 = (f x) * (f x)) → (m = 1 ∨ m = -3) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_polynomial_l1054_105463


namespace NUMINAMATH_GPT_boiling_point_fahrenheit_l1054_105495

-- Define the conditions as hypotheses
def boils_celsius : ℝ := 100
def melts_celsius : ℝ := 0
def melts_fahrenheit : ℝ := 32
def pot_temp_celsius : ℝ := 55
def pot_temp_fahrenheit : ℝ := 131

-- Theorem to prove the boiling point in Fahrenheit
theorem boiling_point_fahrenheit : ∀ (boils_celsius : ℝ) (melts_celsius : ℝ) (melts_fahrenheit : ℝ) 
                                    (pot_temp_celsius : ℝ) (pot_temp_fahrenheit : ℝ),
  boils_celsius = 100 →
  melts_celsius = 0 →
  melts_fahrenheit = 32 →
  pot_temp_celsius = 55 →
  pot_temp_fahrenheit = 131 →
  ∃ boils_fahrenheit : ℝ, boils_fahrenheit = 212 :=
by
  intros
  existsi 212
  sorry

end NUMINAMATH_GPT_boiling_point_fahrenheit_l1054_105495


namespace NUMINAMATH_GPT_find_natural_number_l1054_105441

theorem find_natural_number (n : ℕ) (h : ∃ k : ℕ, n^2 - 19 * n + 95 = k^2) : n = 5 ∨ n = 14 := by
  sorry

end NUMINAMATH_GPT_find_natural_number_l1054_105441


namespace NUMINAMATH_GPT_a8_value_l1054_105435

def sequence_sum (n : ℕ) : ℕ := 2^n - 1

def nth_term (S : ℕ → ℕ) (n : ℕ) : ℕ :=
  S n - S (n - 1)

theorem a8_value : nth_term sequence_sum 8 = 128 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_a8_value_l1054_105435


namespace NUMINAMATH_GPT_range_of_a_l1054_105481

def A : Set ℝ := { x | x^2 - 3 * x + 2 ≤ 0 }
def B (a : ℝ) : Set ℝ := { x | 1 / (x - 3) < a }

theorem range_of_a (a : ℝ) : A ⊆ B a ↔ a > -1/2 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l1054_105481


namespace NUMINAMATH_GPT_option_d_is_right_triangle_l1054_105415

noncomputable def right_triangle (a b c : ℝ) : Prop :=
  a^2 + c^2 = b^2

theorem option_d_is_right_triangle (a b c : ℝ) (h : a^2 = b^2 - c^2) :
  right_triangle a b c :=
by
  sorry

end NUMINAMATH_GPT_option_d_is_right_triangle_l1054_105415


namespace NUMINAMATH_GPT_percent_problem_l1054_105430

theorem percent_problem (x : ℝ) (hx : 0.60 * 600 = 0.50 * x) : x = 720 :=
by
  sorry

end NUMINAMATH_GPT_percent_problem_l1054_105430


namespace NUMINAMATH_GPT_union_of_A_and_B_l1054_105448

-- Define the sets A and B
def A : Set ℤ := {-1, 0}
def B : Set ℤ := {0, 1}

-- Prove that the union of A and B is {-1, 0, 1}
theorem union_of_A_and_B : A ∪ B = {-1, 0, 1} :=
  by sorry

end NUMINAMATH_GPT_union_of_A_and_B_l1054_105448


namespace NUMINAMATH_GPT_sale_in_fifth_month_l1054_105447

-- Define the sales in the first, second, third, fourth, and sixth months
def a1 : ℕ := 7435
def a2 : ℕ := 7927
def a3 : ℕ := 7855
def a4 : ℕ := 8230
def a6 : ℕ := 5991

-- Define the average sale
def avg_sale : ℕ := 7500

-- Define the number of months
def months : ℕ := 6

-- The total sales required for the average sale to be 7500 over 6 months.
def total_sales : ℕ := avg_sale * months

-- Calculate the sales in the first four months
def sales_first_four_months : ℕ := a1 + a2 + a3 + a4

-- Calculate the total sales for the first four months plus the sixth month.
def sales_first_four_and_sixth : ℕ := sales_first_four_months + a6

-- Prove the sale in the fifth month
theorem sale_in_fifth_month : ∃ a5 : ℕ, total_sales = sales_first_four_and_sixth + a5 ∧ a5 = 7562 :=
by
  sorry


end NUMINAMATH_GPT_sale_in_fifth_month_l1054_105447


namespace NUMINAMATH_GPT_age_of_new_person_l1054_105428

theorem age_of_new_person (T A : ℤ) (h : (T / 10 - 3) = (T - 40 + A) / 10) : A = 10 :=
sorry

end NUMINAMATH_GPT_age_of_new_person_l1054_105428


namespace NUMINAMATH_GPT_garden_perimeter_l1054_105425

-- Definitions for length and breadth
def length := 150
def breadth := 100

-- Theorem that states the perimeter of the rectangular garden
theorem garden_perimeter : (2 * (length + breadth)) = 500 :=
by sorry

end NUMINAMATH_GPT_garden_perimeter_l1054_105425


namespace NUMINAMATH_GPT_sum_of_arithmetic_sequence_is_54_l1054_105460

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_arithmetic_sequence_is_54 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (h_condition : 2 * a 8 = 6 + a 11) : 
  S 9 = 54 :=
sorry

end NUMINAMATH_GPT_sum_of_arithmetic_sequence_is_54_l1054_105460


namespace NUMINAMATH_GPT_exponential_rule_l1054_105454

theorem exponential_rule (a : ℝ) : (a ^ 3) ^ 2 = a ^ 6 :=  
  sorry

end NUMINAMATH_GPT_exponential_rule_l1054_105454


namespace NUMINAMATH_GPT_eggs_needed_per_month_l1054_105471

def saly_needs : ℕ := 10
def ben_needs : ℕ := 14
def ked_needs : ℕ := ben_needs / 2
def weeks_in_month : ℕ := 4

def total_weekly_need : ℕ := saly_needs + ben_needs + ked_needs
def total_monthly_need : ℕ := total_weekly_need * weeks_in_month

theorem eggs_needed_per_month : total_monthly_need = 124 := by
  sorry

end NUMINAMATH_GPT_eggs_needed_per_month_l1054_105471


namespace NUMINAMATH_GPT_Kyle_is_25_l1054_105491

variable (Tyson_age : ℕ := 20)
variable (Frederick_age : ℕ := 2 * Tyson_age)
variable (Julian_age : ℕ := Frederick_age - 20)
variable (Kyle_age : ℕ := Julian_age + 5)

theorem Kyle_is_25 : Kyle_age = 25 := by
  sorry

end NUMINAMATH_GPT_Kyle_is_25_l1054_105491


namespace NUMINAMATH_GPT_ratio_of_areas_l1054_105452

theorem ratio_of_areas (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
    let S₁ := (1 - p * q * r) * (1 - p * q * r)
    let S₂ := (1 + p + p * q) * (1 + q + q * r) * (1 + r + r * p)
    S₁ / S₂ = (S₁ / S₂) := sorry

end NUMINAMATH_GPT_ratio_of_areas_l1054_105452


namespace NUMINAMATH_GPT_natural_numbers_solution_l1054_105496

theorem natural_numbers_solution :
  ∃ (a b c d : ℕ), 
    ab = c + d ∧ a + b = cd ∧
    ((a, b, c, d) = (2, 2, 2, 2) ∨ (a, b, c, d) = (2, 3, 5, 1) ∨ 
     (a, b, c, d) = (3, 2, 5, 1) ∨ (a, b, c, d) = (2, 2, 1, 5) ∨ 
     (a, b, c, d) = (3, 2, 1, 5) ∨ (a, b, c, d) = (2, 3, 1, 5)) :=
by
  sorry

end NUMINAMATH_GPT_natural_numbers_solution_l1054_105496


namespace NUMINAMATH_GPT_scheduling_arrangements_l1054_105468

-- We want to express this as a problem to prove the number of scheduling arrangements.

theorem scheduling_arrangements (n : ℕ) (h : n = 6) :
  (Nat.choose 6 1) * (Nat.choose 5 1) * (Nat.choose 4 2) = 180 := by
  sorry

end NUMINAMATH_GPT_scheduling_arrangements_l1054_105468


namespace NUMINAMATH_GPT_proposition_correctness_l1054_105409

theorem proposition_correctness :
  (∀ x : ℝ, (|x-1| < 2) → (x < 3)) ∧
  (∀ (P Q : Prop), (Q → ¬ P) → (P → ¬ Q)) :=
by 
sorry

end NUMINAMATH_GPT_proposition_correctness_l1054_105409


namespace NUMINAMATH_GPT_february_sales_increase_l1054_105453

theorem february_sales_increase (Slast : ℝ) (r : ℝ) (Sthis : ℝ) 
  (h_last_year_sales : Slast = 320) 
  (h_percent_increase : r = 0.25) : 
  Sthis = 400 :=
by
  have h1 : Sthis = Slast * (1 + r) := sorry
  sorry

end NUMINAMATH_GPT_february_sales_increase_l1054_105453


namespace NUMINAMATH_GPT_charity_tickets_solution_l1054_105445

theorem charity_tickets_solution (f h d p : ℕ) (ticket_count : f + h + d = 200)
  (revenue : f * p + h * (p / 2) + d * (2 * p) = 3600) : f = 80 := by
  sorry

end NUMINAMATH_GPT_charity_tickets_solution_l1054_105445


namespace NUMINAMATH_GPT_surface_area_original_cube_l1054_105467

theorem surface_area_original_cube
  (n : ℕ)
  (edge_length_smaller : ℕ)
  (smaller_cubes : ℕ)
  (original_surface_area : ℕ)
  (h1 : n = 3)
  (h2 : edge_length_smaller = 4)
  (h3 : smaller_cubes = 27)
  (h4 : 6 * (n * edge_length_smaller) ^ 2 = original_surface_area) :
  original_surface_area = 864 := by
  sorry

end NUMINAMATH_GPT_surface_area_original_cube_l1054_105467


namespace NUMINAMATH_GPT_composite_sum_l1054_105478

theorem composite_sum (x y n : ℕ) (hx : x > 1) (hy : y > 1) (h : x^2 + x * y - y = n^2) :
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = x + y + 1 :=
sorry

end NUMINAMATH_GPT_composite_sum_l1054_105478


namespace NUMINAMATH_GPT_find_g_at_6_l1054_105401

def g (x : ℝ) : ℝ := 3 * x ^ 4 - 20 * x ^ 3 + 37 * x ^ 2 - 18 * x - 80

theorem find_g_at_6 : g 6 = 712 := by
  -- We apply the remainder theorem to determine the value of g(6).
  sorry

end NUMINAMATH_GPT_find_g_at_6_l1054_105401


namespace NUMINAMATH_GPT_taxi_fare_l1054_105479

-- Define the necessary values and functions based on the problem conditions
def starting_price : ℝ := 6
def additional_charge_per_km : ℝ := 1.5
def distance (P : ℝ) : Prop := P > 6

-- Lean proposition to state the problem
theorem taxi_fare (P : ℝ) (hP : distance P) : 
  (starting_price + additional_charge_per_km * (P - 6)) = 1.5 * P - 3 := 
by 
  sorry

end NUMINAMATH_GPT_taxi_fare_l1054_105479


namespace NUMINAMATH_GPT_system_solution_exists_l1054_105462

theorem system_solution_exists (x y: ℝ) :
    (y^2 = (x + 8) * (x^2 + 2) ∧ y^2 - (8 + 4 * x) * y + (16 + 16 * x - 5 * x^2) = 0) → 
    ((x = 0 ∧ (y = 4 ∨ y = -4)) ∨ (x = -2 ∧ (y = 6 ∨ y = -6)) ∨ (x = 19 ∧ (y = 99 ∨ y = -99))) :=
    sorry

end NUMINAMATH_GPT_system_solution_exists_l1054_105462


namespace NUMINAMATH_GPT_inequality_of_weighted_squares_equality_conditions_of_weighted_squares_l1054_105464

theorem inequality_of_weighted_squares
  (x y a b : ℝ)
  (h_sum : a + b = 1)
  (h_nonneg_a : a ≥ 0)
  (h_nonneg_b : b ≥ 0) :
  (a * x + b * y)^2 ≤ a * x^2 + b * y^2 :=
by sorry

theorem equality_conditions_of_weighted_squares
  (x y a b : ℝ)
  (h_sum : a + b = 1)
  (h_nonneg_a : a ≥ 0)
  (h_nonneg_b : b ≥ 0) :
  (a * x + b * y)^2 = a * x^2 + b * y^2
  ↔ (a = 0 ∨ b = 0 ∨ x = y) :=
by sorry

end NUMINAMATH_GPT_inequality_of_weighted_squares_equality_conditions_of_weighted_squares_l1054_105464


namespace NUMINAMATH_GPT_FerrisWheelCostIsSix_l1054_105492

structure AmusementPark where
  roller_coaster_cost : ℕ
  log_ride_cost : ℕ
  initial_tickets : ℕ
  additional_tickets_needed : ℕ

def ferris_wheel_cost (a : AmusementPark) : ℕ :=
  let total_needed := a.initial_tickets + a.additional_tickets_needed
  let total_ride_cost := a.roller_coaster_cost + a.log_ride_cost
  total_needed - total_ride_cost

theorem FerrisWheelCostIsSix (a : AmusementPark) 
  (h₁ : a.roller_coaster_cost = 5)
  (h₂ : a.log_ride_cost = 7)
  (h₃ : a.initial_tickets = 2)
  (h₄ : a.additional_tickets_needed = 16) :
  ferris_wheel_cost a = 6 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_FerrisWheelCostIsSix_l1054_105492


namespace NUMINAMATH_GPT_cookie_radius_l1054_105434

theorem cookie_radius (x y : ℝ) : x^2 + y^2 + 28 = 6*x + 20*y → ∃ r, r = 9 :=
by
  sorry

end NUMINAMATH_GPT_cookie_radius_l1054_105434


namespace NUMINAMATH_GPT_circle_equation_exists_l1054_105414

def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (2, -2)
def l (p : ℝ × ℝ) : Prop := p.1 - p.2 + 1 = 0
def is_on_circle (C : ℝ × ℝ) (p : ℝ × ℝ) (r : ℝ) : Prop :=
  (p.1 - C.1)^2 + (p.2 - C.2)^2 = r^2

theorem circle_equation_exists :
  ∃ C : ℝ × ℝ, C.1 - C.2 + 1 = 0 ∧
  (is_on_circle C A 5) ∧
  (is_on_circle C B 5) ∧
  is_on_circle C (-3, -2) 5 :=
sorry

end NUMINAMATH_GPT_circle_equation_exists_l1054_105414


namespace NUMINAMATH_GPT_Rebecca_tips_calculation_l1054_105421

def price_haircut : ℤ := 30
def price_perm : ℤ := 40
def price_dye_job : ℤ := 60
def cost_hair_dye_box : ℤ := 10
def num_haircuts : ℕ := 4
def num_perms : ℕ := 1
def num_dye_jobs : ℕ := 2
def total_end_day : ℤ := 310

noncomputable def total_service_earnings : ℤ := 
  num_haircuts * price_haircut + num_perms * price_perm + num_dye_jobs * price_dye_job

noncomputable def total_hair_dye_cost : ℤ := 
  num_dye_jobs * cost_hair_dye_box

noncomputable def earnings_after_cost : ℤ := 
  total_service_earnings - total_hair_dye_cost

noncomputable def tips : ℤ := 
  total_end_day - earnings_after_cost

theorem Rebecca_tips_calculation : tips = 50 := by
  sorry

end NUMINAMATH_GPT_Rebecca_tips_calculation_l1054_105421


namespace NUMINAMATH_GPT_algebraic_expression_value_l1054_105416

variable (x y A B : ℤ)
variable (x_val : x = -1)
variable (y_val : y = 2)
variable (A_def : A = 2*x + y)
variable (B_def : B = 2*x - y)

theorem algebraic_expression_value : 
  (A^2 - B^2) * (x - 2*y) = 80 := 
by
  rw [x_val, y_val, A_def, B_def]
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1054_105416


namespace NUMINAMATH_GPT_crocodiles_count_l1054_105433

-- Definitions of constants
def alligators : Nat := 23
def vipers : Nat := 5
def total_dangerous_animals : Nat := 50

-- Theorem statement
theorem crocodiles_count :
  total_dangerous_animals - alligators - vipers = 22 :=
by
  sorry

end NUMINAMATH_GPT_crocodiles_count_l1054_105433


namespace NUMINAMATH_GPT_binomial_product_l1054_105470

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_product : binomial 10 3 * binomial 8 3 = 6720 := 
by 
  sorry

end NUMINAMATH_GPT_binomial_product_l1054_105470


namespace NUMINAMATH_GPT_relationship_bx_x2_a2_l1054_105443

theorem relationship_bx_x2_a2 {a b x : ℝ} (h1 : b < x) (h2 : x < a) (h3 : 0 < a) (h4 : 0 < b) : 
  b * x < x^2 ∧ x^2 < a^2 :=
by sorry

end NUMINAMATH_GPT_relationship_bx_x2_a2_l1054_105443


namespace NUMINAMATH_GPT_find_value_of_fraction_l1054_105484

variable {x y : ℝ}

theorem find_value_of_fraction (h1 : x > 0) (h2 : y > x) (h3 : y > 0) (h4 : x / y + y / x = 3) : 
  (x + y) / (y - x) = Real.sqrt 5 := 
by sorry

end NUMINAMATH_GPT_find_value_of_fraction_l1054_105484


namespace NUMINAMATH_GPT_overall_percent_change_l1054_105461

theorem overall_percent_change (x : ℝ) : 
  (0.85 * x * 1.25 * 0.9 / x - 1) * 100 = -4.375 := 
by 
  sorry

end NUMINAMATH_GPT_overall_percent_change_l1054_105461


namespace NUMINAMATH_GPT_smallest_two_digit_multiple_of_17_l1054_105432

theorem smallest_two_digit_multiple_of_17 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ n % 17 = 0 ∧ ∀ m, (10 ≤ m ∧ m < n ∧ m % 17 = 0) → false := sorry

end NUMINAMATH_GPT_smallest_two_digit_multiple_of_17_l1054_105432


namespace NUMINAMATH_GPT_Steve_pencils_left_l1054_105424

-- Define the initial number of boxes and pencils per box
def boxes := 2
def pencils_per_box := 12
def initial_pencils := boxes * pencils_per_box

-- Define the number of pencils given to Lauren and the additional pencils given to Matt
def pencils_to_Lauren := 6
def diff_Lauren_Matt := 3
def pencils_to_Matt := pencils_to_Lauren + diff_Lauren_Matt

-- Calculate the total pencils given away
def pencils_given_away := pencils_to_Lauren + pencils_to_Matt

-- Number of pencils left with Steve
def pencils_left := initial_pencils - pencils_given_away

-- The statement to prove
theorem Steve_pencils_left : pencils_left = 9 := by
  sorry

end NUMINAMATH_GPT_Steve_pencils_left_l1054_105424


namespace NUMINAMATH_GPT_max_possible_value_l1054_105406

theorem max_possible_value (P Q : ℤ) (hP : P * P ≤ 729 ∧ 729 ≤ -P * P * P)
  (hQ : Q * Q ≤ 729 ∧ 729 ≤ -Q * Q * Q) :
  10 * (P - Q) = 180 :=
by
  sorry

end NUMINAMATH_GPT_max_possible_value_l1054_105406


namespace NUMINAMATH_GPT_point_A_equidistant_l1054_105412

/-
This statement defines the problem of finding the coordinates of point A that is equidistant from points B and C.
-/
theorem point_A_equidistant (x : ℝ) :
  (dist (x, 0, 0) (3, 5, 6)) = (dist (x, 0, 0) (1, 2, 3)) ↔ x = 14 :=
by {
  sorry
}

end NUMINAMATH_GPT_point_A_equidistant_l1054_105412


namespace NUMINAMATH_GPT_directrix_of_parabola_l1054_105427

-- Define the given conditions
def parabola_focus_on_line (p : ℝ) := ∃ (x y : ℝ), y^2 = 2 * p * x ∧ 2 * x + 3 * y - 8 = 0

-- Define the statement to be proven
theorem directrix_of_parabola (p : ℝ) (h: parabola_focus_on_line p) : 
   ∃ (d : ℝ), d = -4 := 
sorry

end NUMINAMATH_GPT_directrix_of_parabola_l1054_105427


namespace NUMINAMATH_GPT_problem_l1054_105457

variable (m : ℝ)

def p (m : ℝ) : Prop := m ≤ 2
def q (m : ℝ) : Prop := 0 < m ∧ m < 1

theorem problem (hpq : ¬ (p m ∧ q m)) (hlpq : p m ∨ q m) : m ≤ 0 ∨ (1 ≤ m ∧ m ≤ 2) := 
sorry

end NUMINAMATH_GPT_problem_l1054_105457


namespace NUMINAMATH_GPT_school_trip_seat_count_l1054_105482

theorem school_trip_seat_count :
  ∀ (classrooms students_per_classroom seats_per_bus : ℕ),
  classrooms = 87 →
  students_per_classroom = 58 →
  seats_per_bus = 29 →
  ∀ (total_students total_buses_needed : ℕ),
  total_students = classrooms * students_per_classroom →
  total_buses_needed = (total_students + seats_per_bus - 1) / seats_per_bus →
  seats_per_bus = 29 := by
  intros classrooms students_per_classroom seats_per_bus
  intros h1 h2 h3
  intros total_students total_buses_needed
  intros h4 h5
  sorry

end NUMINAMATH_GPT_school_trip_seat_count_l1054_105482


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l1054_105403

theorem necessary_and_sufficient_condition (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 > 2 ∧ x2 < 2 ∧ x1^2 - m * x1 - 1 = 0 ∧ x2^2 - m * x2 - 1 = 0) ↔ m > 1.5 :=
by
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l1054_105403


namespace NUMINAMATH_GPT_problem_statement_l1054_105477

-- Define the function g and specify its properties
def g : ℕ → ℕ := sorry

axiom g_property (a b : ℕ) : g (a^2 + b^2) + g (a + b) = (g a)^2 + (g b)^2

-- Define the values of m and t that arise from the constraints on g(49)
def m : ℕ := 2
def t : ℕ := 106

-- Prove that the product m * t is 212
theorem problem_statement : m * t = 212 :=
by {
  -- Since g_property is an axiom, we use it to derive that
  -- g(49) can only take possible values 0 and 106,
  -- thus m = 2 and t = 106.
  sorry
}

end NUMINAMATH_GPT_problem_statement_l1054_105477


namespace NUMINAMATH_GPT_sum_reciprocal_factors_of_12_l1054_105487

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end NUMINAMATH_GPT_sum_reciprocal_factors_of_12_l1054_105487


namespace NUMINAMATH_GPT_James_wait_weeks_l1054_105439

def JamesExercising (daysPainSubside : ℕ) (healingMultiplier : ℕ) (delayAfterHealing : ℕ) (totalDaysUntilHeavyLift : ℕ) : ℕ :=
  let healingTime := daysPainSubside * healingMultiplier
  let startWorkingOut := healingTime + delayAfterHealing
  let waitingPeriodDays := totalDaysUntilHeavyLift - startWorkingOut
  waitingPeriodDays / 7

theorem James_wait_weeks : 
  JamesExercising 3 5 3 39 = 3 :=
by
  sorry

end NUMINAMATH_GPT_James_wait_weeks_l1054_105439


namespace NUMINAMATH_GPT_stream_speed_l1054_105450

-- Define the conditions
def still_water_speed : ℝ := 15
def upstream_time_factor : ℕ := 2

-- Define the theorem
theorem stream_speed (t v : ℝ) (h : (still_water_speed + v) * t = (still_water_speed - v) * (upstream_time_factor * t)) : v = 5 :=
by
  sorry

end NUMINAMATH_GPT_stream_speed_l1054_105450


namespace NUMINAMATH_GPT_brownies_pieces_count_l1054_105485

-- Definitions of the conditions
def pan_length : ℕ := 24
def pan_width : ℕ := 15
def pan_area : ℕ := pan_length * pan_width -- pan_area = 360

def piece_length : ℕ := 3
def piece_width : ℕ := 2
def piece_area : ℕ := piece_length * piece_width -- piece_area = 6

-- Definition of the question and proving the expected answer
theorem brownies_pieces_count : (pan_area / piece_area) = 60 := by
  sorry

end NUMINAMATH_GPT_brownies_pieces_count_l1054_105485


namespace NUMINAMATH_GPT_cucumber_kinds_l1054_105410

theorem cucumber_kinds (x : ℕ) :
  (3 * 5) + (4 * x) + 30 + 85 = 150 → x = 5 :=
by
  intros h
  -- h : 15 + 4 * x + 30 + 85 = 150 

  -- Proof would go here
  sorry

end NUMINAMATH_GPT_cucumber_kinds_l1054_105410
