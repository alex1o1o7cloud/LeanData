import Mathlib

namespace NUMINAMATH_GPT_symmetry_about_origin_l159_15901

noncomputable def f (x : ℝ) : ℝ := x * Real.log (-x)
noncomputable def g (x : ℝ) : ℝ := x * Real.log x

theorem symmetry_about_origin :
  ∀ x : ℝ, f (-x) = -g (-x) :=
by
  sorry

end NUMINAMATH_GPT_symmetry_about_origin_l159_15901


namespace NUMINAMATH_GPT_units_digit_of_516n_divisible_by_12_l159_15979

theorem units_digit_of_516n_divisible_by_12 (n : ℕ) (h₀ : n ≤ 9) :
  (516 * 10 + n) % 12 = 0 ↔ n = 0 ∨ n = 4 :=
by 
  sorry

end NUMINAMATH_GPT_units_digit_of_516n_divisible_by_12_l159_15979


namespace NUMINAMATH_GPT_original_rectangle_perimeter_difference_is_multiple_of_7_relationship_for_seamless_combination_l159_15934

-- Condition declarations
variables (x y : ℕ)
variables (hx : x > 0) (hy : y > 0)
variables (h_sum : x + y = 25)
variables (h_diff : (x + 5) * (y + 5) - (x - 2) * (y - 2) = 196)

-- Lean 4 statements for the proof problem
theorem original_rectangle_perimeter (x y : ℕ) (hx : x > 0) (hy : y > 0) (h_sum : x + y = 25) : 2 * (x + y) = 50 := by
  sorry

theorem difference_is_multiple_of_7 (x y : ℕ) (hx : x > 0) (hy : y > 0) 
  (h_diff : (x + 5) * (y + 5) - (x - 2) * (y - 2) = 196) : 7 ∣ ((x + 5) * (y + 5) - (x - 2) * (y - 2)) := by
  sorry

theorem relationship_for_seamless_combination (x y : ℕ) (h_sum : x + y = 25) (h_seamless : (x+5)*(y+5) = (x*(y+5))) : x = y + 5 := by
  sorry

end NUMINAMATH_GPT_original_rectangle_perimeter_difference_is_multiple_of_7_relationship_for_seamless_combination_l159_15934


namespace NUMINAMATH_GPT_average_percent_score_is_77_l159_15939

def numberOfStudents : ℕ := 100

def percentage_counts : List (ℕ × ℕ) :=
[(100, 7), (90, 18), (80, 35), (70, 25), (60, 10), (50, 3), (40, 2)]

noncomputable def average_score (counts : List (ℕ × ℕ)) : ℚ :=
  (counts.foldl (λ acc p => acc + (p.1 * p.2)) 0 : ℚ) / numberOfStudents

theorem average_percent_score_is_77 : average_score percentage_counts = 77 := by
  sorry

end NUMINAMATH_GPT_average_percent_score_is_77_l159_15939


namespace NUMINAMATH_GPT_smallest_lcm_not_multiple_of_25_l159_15951

theorem smallest_lcm_not_multiple_of_25 (n : ℕ) (h1 : n % 36 = 0) (h2 : n % 45 = 0) (h3 : n % 25 ≠ 0) : n = 180 := 
by 
  sorry

end NUMINAMATH_GPT_smallest_lcm_not_multiple_of_25_l159_15951


namespace NUMINAMATH_GPT_solve_base_6_addition_l159_15965

variables (X Y k : ℕ)

theorem solve_base_6_addition (h1 : Y + 3 = X) (h2 : ∃ k, X + 5 = 2 + 6 * k) : X + Y = 3 :=
sorry

end NUMINAMATH_GPT_solve_base_6_addition_l159_15965


namespace NUMINAMATH_GPT_factory_output_increase_l159_15919

theorem factory_output_increase (x : ℝ) (h : (1 + x / 100) ^ 4 = 4) : x = 41.4 :=
by
  -- Given (1 + x / 100) ^ 4 = 4
  sorry

end NUMINAMATH_GPT_factory_output_increase_l159_15919


namespace NUMINAMATH_GPT_stratified_sampling_workshops_l159_15974

theorem stratified_sampling_workshops (units_A units_B units_C sample_B n : ℕ) 
(hA : units_A = 96) 
(hB : units_B = 84) 
(hC : units_C = 60) 
(hSample_B : sample_B = 7) 
(hn : (sample_B : ℚ) / n = (units_B : ℚ) / (units_A + units_B + units_C)) : 
  n = 70 :=
  by
  sorry

end NUMINAMATH_GPT_stratified_sampling_workshops_l159_15974


namespace NUMINAMATH_GPT_total_wet_surface_area_l159_15963

def cistern_length (L : ℝ) := L = 5
def cistern_width (W : ℝ) := W = 4
def water_depth (D : ℝ) := D = 1.25

theorem total_wet_surface_area (L W D A : ℝ) 
  (hL : cistern_length L) 
  (hW : cistern_width W) 
  (hD : water_depth D) :
  A = 42.5 :=
by
  subst hL
  subst hW
  subst hD
  sorry

end NUMINAMATH_GPT_total_wet_surface_area_l159_15963


namespace NUMINAMATH_GPT_gcd_three_numbers_4557_1953_5115_l159_15994

theorem gcd_three_numbers_4557_1953_5115 : Nat.gcd (Nat.gcd 4557 1953) 5115 = 93 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_three_numbers_4557_1953_5115_l159_15994


namespace NUMINAMATH_GPT_tom_mowing_lawn_l159_15924

theorem tom_mowing_lawn (hours_to_mow : ℕ) (time_worked : ℕ) (fraction_mowed_per_hour : ℚ) : 
  (hours_to_mow = 6) → 
  (time_worked = 3) → 
  (fraction_mowed_per_hour = (1 : ℚ) / hours_to_mow) → 
  (1 - (time_worked * fraction_mowed_per_hour) = (1 : ℚ) / 2) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_tom_mowing_lawn_l159_15924


namespace NUMINAMATH_GPT_solution_l159_15943

noncomputable def problem : Prop :=
  let num_apprentices := 200
  let num_junior := 20
  let num_intermediate := 60
  let num_senior := 60
  let num_technician := 40
  let num_senior_technician := 20
  let total_technician := num_technician + num_senior_technician
  let sampling_ratio := 10 / num_apprentices
  
  -- Number of technicians (including both technician and senior technicians) in the exchange group
  let num_technicians_selected := total_technician * sampling_ratio

  -- Probability Distribution of X
  let P_X_0 := 7 / 24
  let P_X_1 := 21 / 40
  let P_X_2 := 7 / 40
  let P_X_3 := 1 / 120

  -- Expected value of X
  let E_X := (0 * P_X_0) + (1 * P_X_1) + (2 * P_X_2) + (3 * P_X_3)
  E_X = 9 / 10

theorem solution : problem :=
  sorry

end NUMINAMATH_GPT_solution_l159_15943


namespace NUMINAMATH_GPT_sum_of_sides_is_seven_l159_15971

def triangle_sides : ℕ := 3
def quadrilateral_sides : ℕ := 4
def sum_of_sides : ℕ := triangle_sides + quadrilateral_sides

theorem sum_of_sides_is_seven : sum_of_sides = 7 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_sides_is_seven_l159_15971


namespace NUMINAMATH_GPT_gcd_lcm_sum_l159_15909

-- Define the given numbers
def a1 := 54
def b1 := 24
def a2 := 48
def b2 := 18

-- Define the GCD and LCM functions in Lean
def gcd_ab := Nat.gcd a1 b1
def lcm_cd := Nat.lcm a2 b2

-- Define the final sum
def final_sum := gcd_ab + lcm_cd

-- State the equality that represents the problem
theorem gcd_lcm_sum : final_sum = 150 := by
  sorry

end NUMINAMATH_GPT_gcd_lcm_sum_l159_15909


namespace NUMINAMATH_GPT_even_iff_a_zero_max_value_f_l159_15925

noncomputable def f (x a : ℝ) : ℝ := -x^2 + |x - a| + a + 1

theorem even_iff_a_zero (a : ℝ) : (∀ x, f x a = f (-x) a) ↔ a = 0 :=
by {
  -- Proof is omitted
  sorry
}

theorem max_value_f (a : ℝ) : 
  ∃ max_val : ℝ, 
    ( 
      (-1/2 < a ∧ a ≤ 0 ∧ max_val = 5/4) ∨ 
      (0 < a ∧ a < 1/2 ∧ max_val = 5/4 + 2*a) ∨ 
      ((a ≤ -1/2 ∨ a ≥ 1/2) ∧ max_val = -a^2 + a + 1)
    ) :=
by {
  -- Proof is omitted
  sorry
}

end NUMINAMATH_GPT_even_iff_a_zero_max_value_f_l159_15925


namespace NUMINAMATH_GPT_lesser_number_is_32_l159_15967

variable (x y : ℕ)

theorem lesser_number_is_32 (h1 : y = 2 * x) (h2 : x + y = 96) : x = 32 := 
sorry

end NUMINAMATH_GPT_lesser_number_is_32_l159_15967


namespace NUMINAMATH_GPT_ratio_P_K_is_2_l159_15935

theorem ratio_P_K_is_2 (P K M : ℝ) (r : ℝ)
  (h1: P + K + M = 153)
  (h2: P = r * K)
  (h3: P = (1/3) * M)
  (h4: M = K + 85) : r = 2 :=
  sorry

end NUMINAMATH_GPT_ratio_P_K_is_2_l159_15935


namespace NUMINAMATH_GPT_calculate_seven_a_sq_minus_four_a_sq_l159_15923

variable (a : ℝ)

theorem calculate_seven_a_sq_minus_four_a_sq : 7 * a^2 - 4 * a^2 = 3 * a^2 := 
by
  sorry

end NUMINAMATH_GPT_calculate_seven_a_sq_minus_four_a_sq_l159_15923


namespace NUMINAMATH_GPT_cylindrical_coordinates_of_point_l159_15953

noncomputable def cylindrical_coordinates (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x = -r then Real.pi else 0 -- From the step if cos θ = -1
  (r, θ, z)

theorem cylindrical_coordinates_of_point :
  cylindrical_coordinates (-5) 0 (-8) = (5, Real.pi, -8) :=
by
  -- placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_cylindrical_coordinates_of_point_l159_15953


namespace NUMINAMATH_GPT_num_repeating_decimals_between_1_and_20_l159_15933

def is_repeating_decimal (a b : ℕ) : Prop :=
  ∀ p q : ℕ, ¬ b = 2^p * 5^q

theorem num_repeating_decimals_between_1_and_20 :
  ∃ (cnt : ℕ), cnt = 20 ∧
  ∀ n : ℕ, (1 ≤ n ∧ n ≤ 20) → is_repeating_decimal n 18 := 
by
  sorry

end NUMINAMATH_GPT_num_repeating_decimals_between_1_and_20_l159_15933


namespace NUMINAMATH_GPT_infinite_seq_condition_l159_15985

theorem infinite_seq_condition (x : ℕ → ℕ) (n m : ℕ) : 
  (∀ i, x i = 0 → x (i + m) = 1) → 
  (∀ i, x i = 1 → x (i + n) = 0) → 
  ∃ d p q : ℕ, n = 2^d * p ∧ m = 2^d * q ∧ p % 2 = 1 ∧ q % 2 = 1  :=
by 
  intros h1 h2 
  sorry

end NUMINAMATH_GPT_infinite_seq_condition_l159_15985


namespace NUMINAMATH_GPT_find_domain_l159_15931

noncomputable def domain (x : ℝ) : Prop :=
  (2 * x + 1 ≥ 0) ∧ (3 - 4 * x ≥ 0)

theorem find_domain :
  {x : ℝ | domain x} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 3/4} :=
by
  sorry

end NUMINAMATH_GPT_find_domain_l159_15931


namespace NUMINAMATH_GPT_problem_false_proposition_l159_15937

def p : Prop := ∀ x : ℝ, |x| = x ↔ x > 0

def q : Prop := (¬ ∃ x : ℝ, x^2 - x > 0) ↔ ∀ x : ℝ, x^2 - x ≤ 0

theorem problem_false_proposition : ¬ (p ∧ q) :=
by
  sorry

end NUMINAMATH_GPT_problem_false_proposition_l159_15937


namespace NUMINAMATH_GPT_first_lock_stall_time_eq_21_l159_15986

-- Definitions of time taken by locks
def firstLockTime : ℕ := 21 -- This will be proven at the end

variables {x : ℕ} -- time for the first lock
variables (secondLockTime : ℕ) (bothLocksTime : ℕ)

-- Conditions given in the problem
axiom lock_relation : secondLockTime = 3 * x - 3
axiom second_lock_time : secondLockTime = 60
axiom combined_locks_time : bothLocksTime = 300

-- Question: Prove that the first lock time is 21 minutes
theorem first_lock_stall_time_eq_21 :
  (bothLocksTime = 5 * secondLockTime) ∧ (secondLockTime = 60) ∧ (bothLocksTime = 300) → x = 21 :=
sorry

end NUMINAMATH_GPT_first_lock_stall_time_eq_21_l159_15986


namespace NUMINAMATH_GPT_general_term_l159_15954

noncomputable def F (n : ℕ) : ℝ :=
  1 / (Real.sqrt 5) * (((1 + Real.sqrt 5) / 2)^(n-2) - ((1 - Real.sqrt 5) / 2)^(n-2))

noncomputable def a : ℕ → ℝ
| 0 => 1
| 1 => 5
| n+2 => a (n+1) * a n / Real.sqrt ((a (n+1))^2 + (a n)^2 + 1)

theorem general_term (n : ℕ) :
  a n = (2^(F (n+2)) * 13^(F (n+1)) * 5^(-2 * F (n+1)) - 1)^(1/2) := sorry

end NUMINAMATH_GPT_general_term_l159_15954


namespace NUMINAMATH_GPT_john_saves_1200_yearly_l159_15984

noncomputable def former_rent_per_month (sq_ft_cost : ℝ) (sq_ft : ℝ) : ℝ :=
  sq_ft_cost * sq_ft

noncomputable def new_rent_per_month (total_cost : ℝ) (roommates : ℝ) : ℝ :=
  total_cost / roommates

noncomputable def monthly_savings (former_rent : ℝ) (new_rent : ℝ) : ℝ :=
  former_rent - new_rent

noncomputable def annual_savings (monthly_savings : ℝ) : ℝ :=
  monthly_savings * 12

theorem john_saves_1200_yearly :
  let former_rent := former_rent_per_month 2 750
  let new_rent := new_rent_per_month 2800 2
  let monthly_savings := monthly_savings former_rent new_rent
  annual_savings monthly_savings = 1200 := 
by 
  sorry

end NUMINAMATH_GPT_john_saves_1200_yearly_l159_15984


namespace NUMINAMATH_GPT_ladder_rungs_count_l159_15991

theorem ladder_rungs_count :
  ∃ (n : ℕ), ∀ (start mid : ℕ),
    start = n / 2 →
    mid = ((start + 5 - 7) + 8 + 7) →
    mid = n →
    n = 27 :=
by
  sorry

end NUMINAMATH_GPT_ladder_rungs_count_l159_15991


namespace NUMINAMATH_GPT_percentage_carnations_l159_15905

variable (F : ℕ)
variable (H1 : F ≠ 0) -- Non-zero flowers
variable (H2 : ∀ (y : ℕ), 5 * y = F → 2 * y ≠ 0) -- Two fifths of the pink flowers are roses.
variable (H3 : ∀ (z : ℕ), 7 * z = 3 * (F - F / 2 - F / 5) → 6 * z ≠ 0) -- Six sevenths of the red flowers are carnations.
variable (H4 : ∀ (w : ℕ), 5 * w = F → w ≠ 0) -- One fifth of the flowers are yellow tulips.
variable (H5 : 2 * F / 2 = F) -- Half of the flowers are pink.
variable (H6 : ∀ (c : ℕ), 10 * c = F → c ≠ 0) -- Total flowers in multiple of 10

theorem percentage_carnations :
  (exists (pc rc : ℕ), 70 * (pc + rc) = 55 * F) :=
sorry

end NUMINAMATH_GPT_percentage_carnations_l159_15905


namespace NUMINAMATH_GPT_derivative_at_3_l159_15947

def f (x : ℝ) : ℝ := x^2

theorem derivative_at_3 : deriv f 3 = 6 := by
  sorry

end NUMINAMATH_GPT_derivative_at_3_l159_15947


namespace NUMINAMATH_GPT_initial_money_l159_15996

theorem initial_money (M : ℝ)
  (clothes : M * (1 / 3) = M - M * (2 / 3))
  (food : (M - M * (1 / 3)) * (1 / 5) = (M - M * (1 / 3)) - ((M - M * (1 / 3)) * (4 / 5)))
  (travel : ((M - M * (1 / 3)) - ((M - M * (1 / 3)) * (1 / 5))) * (1 / 4) = ((M - M * (1 / 3)) - ((M - M * (1 / 3)) * (1 / 5))) - (((M - M * (1 / 3)) - ((M - M * (1 / 3)) * (1 / 5)))) * (3 / 4))
  (left : ((M - M * (1 / 3)) - (((M - M * (1 / 3)) - ((M - M * (1 / 3)) * (1 / 5))) * (1 / 4))) = 400)
  : M = 1000 := 
sorry

end NUMINAMATH_GPT_initial_money_l159_15996


namespace NUMINAMATH_GPT_determine_k_l159_15908

noncomputable def p (x y : ℝ) : ℝ := x^2 - y^2
noncomputable def q (x y : ℝ) : ℝ := Real.log (x - y)

def m (k : ℝ) : ℝ := 2 * k
def w (n : ℝ) : ℝ := n + 1

theorem determine_k (k : ℝ) (c : ℝ → ℝ → ℝ) (v : ℝ → ℝ → ℝ) (n : ℝ) :
  p 32 6 = k * c 32 6 ∧
  p 45 10 = m k * c 45 10 ∧
  q 15 5 = n * v 15 5 ∧
  q 28 7 = w n * v 28 7 →
  k = 1925 / 1976 :=
by
  sorry

end NUMINAMATH_GPT_determine_k_l159_15908


namespace NUMINAMATH_GPT_part1_part2_l159_15978

noncomputable def f (x a : ℝ) : ℝ := |x - 1| + |2 * x + a|

theorem part1 (x : ℝ) : f x 1 + |x - 1| ≥ 3 :=
  sorry

theorem part2 (a : ℝ) (h : ∃ x : ℝ, f x a = 2) : a = 2 ∨ a = -6 :=
  sorry

end NUMINAMATH_GPT_part1_part2_l159_15978


namespace NUMINAMATH_GPT_bugs_meet_on_diagonal_l159_15948

noncomputable def isosceles_trapezoid (A B C D : Type) : Prop :=
  ∃ (AB CD : ℝ), (AB > CD) ∧ (AB = AB) ∧ (CD = CD)

noncomputable def same_speeds (speed1 speed2 : ℝ) : Prop :=
  speed1 = speed2

noncomputable def opposite_directions (path1 path2 : ℝ → ℝ) (diagonal_length : ℝ) : Prop :=
  ∀ t, path1 t = diagonal_length - path2 t

noncomputable def bugs_meet (A B C D : Type) (path1 path2 : ℝ → ℝ) (T : ℝ) : Prop :=
  ∃ t ≤ T, path1 t = path2 t

theorem bugs_meet_on_diagonal :
  ∀ (A B C D : Type) (speed : ℝ) (path1 path2 : ℝ → ℝ) (diagonal_length cycle_period : ℝ),
  isosceles_trapezoid A B C D →
  same_speeds speed speed →
  (∀ t, 0 ≤ t → t ≤ cycle_period) →
  opposite_directions path1 path2 diagonal_length →
  bugs_meet A B C D path1 path2 cycle_period :=
by
  intros
  sorry

end NUMINAMATH_GPT_bugs_meet_on_diagonal_l159_15948


namespace NUMINAMATH_GPT_average_price_per_bottle_l159_15995

/-
  Given:
  * Number of large bottles: 1300
  * Price per large bottle: 1.89
  * Number of small bottles: 750
  * Price per small bottle: 1.38
  
  Prove:
  The approximate average price per bottle is 1.70
-/
theorem average_price_per_bottle : 
  let num_large_bottles := 1300
  let price_per_large_bottle := 1.89
  let num_small_bottles := 750
  let price_per_small_bottle := 1.38
  let total_cost_large_bottles := num_large_bottles * price_per_large_bottle
  let total_cost_small_bottles := num_small_bottles * price_per_small_bottle
  let total_number_bottles := num_large_bottles + num_small_bottles
  let overall_total_cost := total_cost_large_bottles + total_cost_small_bottles
  let average_price := overall_total_cost / total_number_bottles
  average_price = 1.70 :=
by
  sorry

end NUMINAMATH_GPT_average_price_per_bottle_l159_15995


namespace NUMINAMATH_GPT_find_brick_length_l159_15966

-- Definitions of dimensions
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6
def wall_length : ℝ := 750
def wall_height : ℝ := 600
def wall_thickness : ℝ := 22.5
def num_bricks : ℝ := 6000

-- Volume calculations
def volume_wall : ℝ := wall_length * wall_height * wall_thickness
def volume_brick (x : ℝ) : ℝ := x * brick_width * brick_height

-- Statement of the problem
theorem find_brick_length (length_of_brick : ℝ) :
  volume_wall = num_bricks * volume_brick length_of_brick → length_of_brick = 25 :=
by
  simp [volume_wall, volume_brick, num_bricks, brick_width, brick_height, wall_length, wall_height, wall_thickness]
  intro h 
  sorry

end NUMINAMATH_GPT_find_brick_length_l159_15966


namespace NUMINAMATH_GPT_cody_marbles_l159_15972

theorem cody_marbles (M : ℕ) (h1 : M / 3 + 5 + 7 = M) : M = 18 :=
by
  have h2 : 3 * M / 3 + 3 * 5 + 3 * 7 = 3 * M := by sorry
  have h3 : 3 * M / 3 = M := by sorry
  have h4 : 3 * 7 = 21 := by sorry
  have h5 : M + 15 + 21 = 3 * M := by sorry
  have h6 : M = 18 := by sorry
  exact h6

end NUMINAMATH_GPT_cody_marbles_l159_15972


namespace NUMINAMATH_GPT_carol_can_invite_friends_l159_15956

-- Definitions based on the problem's conditions
def invitations_per_pack := 9
def packs_bought := 5

-- Required proof statement
theorem carol_can_invite_friends :
  invitations_per_pack * packs_bought = 45 :=
by
  sorry

end NUMINAMATH_GPT_carol_can_invite_friends_l159_15956


namespace NUMINAMATH_GPT_students_chocolate_milk_l159_15926

-- Definitions based on the problem conditions
def students_strawberry_milk : ℕ := 15
def students_regular_milk : ℕ := 3
def total_milks_taken : ℕ := 20

-- The proof goal
theorem students_chocolate_milk : total_milks_taken - (students_strawberry_milk + students_regular_milk) = 2 := by
  -- The proof steps will go here (not required as per instructions)
  sorry

end NUMINAMATH_GPT_students_chocolate_milk_l159_15926


namespace NUMINAMATH_GPT_circus_tickets_l159_15975

variable (L U : ℕ)

theorem circus_tickets (h1 : L + U = 80) (h2 : 30 * L + 20 * U = 2100) : L = 50 :=
by
  sorry

end NUMINAMATH_GPT_circus_tickets_l159_15975


namespace NUMINAMATH_GPT_two_equal_sum_partition_three_equal_sum_partition_l159_15969

-- Definition 1: Sum of the set X_n
def sum_X_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Definition 2: Equivalences for partitioning X_n into two equal sum parts
def partition_two_equal_sum (n : ℕ) : Prop :=
  (n % 4 = 0 ∨ n % 4 = 3) ↔ ∃ (A B : Finset ℕ), A ∪ B = Finset.range n ∧ A ∩ B = ∅ ∧ A.sum id = B.sum id

-- Definition 3: Equivalences for partitioning X_n into three equal sum parts
def partition_three_equal_sum (n : ℕ) : Prop :=
  (n % 3 ≠ 1) ↔ ∃ (A B C : Finset ℕ), A ∪ B ∪ C = Finset.range n ∧ (A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅) ∧ A.sum id = B.sum id ∧ B.sum id = C.sum id

-- Main theorem statements
theorem two_equal_sum_partition (n : ℕ) : partition_two_equal_sum n :=
  sorry

theorem three_equal_sum_partition (n : ℕ) : partition_three_equal_sum n :=
  sorry

end NUMINAMATH_GPT_two_equal_sum_partition_three_equal_sum_partition_l159_15969


namespace NUMINAMATH_GPT_avg_growth_rate_equation_l159_15993

/-- This theorem formalizes the problem of finding the equation for the average growth rate of working hours.
    Given that the average working hours in the first week are 40 hours and in the third week are 48.4 hours,
    we need to show that the equation for the growth rate \(x\) satisfies \( 40(1 + x)^2 = 48.4 \). -/
theorem avg_growth_rate_equation (x : ℝ) (first_week_hours third_week_hours : ℝ) 
  (h1: first_week_hours = 40) (h2: third_week_hours = 48.4) :
  40 * (1 + x) ^ 2 = 48.4 :=
sorry

end NUMINAMATH_GPT_avg_growth_rate_equation_l159_15993


namespace NUMINAMATH_GPT_deleted_files_l159_15992

variable {initial_files : ℕ}
variable {files_per_folder : ℕ}
variable {folders : ℕ}

noncomputable def files_deleted (initial_files files_in_folders : ℕ) : ℕ :=
  initial_files - files_in_folders

theorem deleted_files (h1 : initial_files = 27) (h2 : files_per_folder = 6) (h3 : folders = 3) :
  files_deleted initial_files (files_per_folder * folders) = 9 :=
by
  sorry

end NUMINAMATH_GPT_deleted_files_l159_15992


namespace NUMINAMATH_GPT_find_a_and_an_l159_15941

-- Given Sequences
def S (n : ℕ) (a : ℝ) : ℝ := 3^n - a

def is_geometric_sequence (a_n : ℕ → ℝ) : Prop := ∃ a1 q, q ≠ 1 ∧ ∀ n, a_n n = a1 * q^n

-- The main statement
theorem find_a_and_an (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (a : ℝ) :
  (∀ n, S_n n = 3^n - a) ∧ is_geometric_sequence a_n →
  ∃ a, a = 1 ∧ ∀ n, a_n n = 2 * 3^(n-1) :=
by
  sorry

end NUMINAMATH_GPT_find_a_and_an_l159_15941


namespace NUMINAMATH_GPT_g_symmetry_solutions_l159_15950

noncomputable def g : ℝ → ℝ := sorry

theorem g_symmetry_solutions (g_def: ∀ (x : ℝ), x ≠ 0 → g x + 3 * g (1 / x) = 6 * x^2) :
  ∀ (x : ℝ), g x = g (-x) → x = 1 ∨ x = -1 :=
by
  sorry

end NUMINAMATH_GPT_g_symmetry_solutions_l159_15950


namespace NUMINAMATH_GPT_total_cookies_after_three_days_l159_15980

-- Define the initial conditions
def cookies_baked_monday : ℕ := 32
def cookies_baked_tuesday : ℕ := cookies_baked_monday / 2
def cookies_baked_wednesday_before : ℕ := cookies_baked_tuesday * 3
def brother_ate : ℕ := 4

-- Define the total cookies before brother ate any
def total_cookies_before : ℕ := cookies_baked_monday + cookies_baked_tuesday + cookies_baked_wednesday_before

-- Define the total cookies after brother ate some
def total_cookies_after : ℕ := total_cookies_before - brother_ate

-- The proof statement
theorem total_cookies_after_three_days : total_cookies_after = 92 := by
  -- Here, we would provide the proof, but we add sorry for now to compile successfully.
  sorry

end NUMINAMATH_GPT_total_cookies_after_three_days_l159_15980


namespace NUMINAMATH_GPT_find_a6_geometric_sequence_l159_15982

noncomputable def geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

theorem find_a6_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (h1 : geom_seq a q) (h2 : a 4 = 7) (h3 : a 8 = 63) : 
  a 6 = 21 :=
sorry

end NUMINAMATH_GPT_find_a6_geometric_sequence_l159_15982


namespace NUMINAMATH_GPT_exactly_one_passes_l159_15981

theorem exactly_one_passes (P_A P_B : ℚ) (hA : P_A = 3 / 5) (hB : P_B = 1 / 3) : 
  (1 - P_A) * P_B + P_A * (1 - P_B) = 8 / 15 :=
by
  -- skipping the proof as per requirement
  sorry

end NUMINAMATH_GPT_exactly_one_passes_l159_15981


namespace NUMINAMATH_GPT_circle_sector_radius_l159_15958

theorem circle_sector_radius (r : ℝ) :
  (2 * r + (r * (Real.pi / 3)) = 144) → r = 432 / (6 + Real.pi) := by
  sorry

end NUMINAMATH_GPT_circle_sector_radius_l159_15958


namespace NUMINAMATH_GPT_calculate_expression_l159_15989

theorem calculate_expression : (3.15 * 2.5) - 1.75 = 6.125 := 
by
  -- The proof is omitted, indicated by sorry
  sorry

end NUMINAMATH_GPT_calculate_expression_l159_15989


namespace NUMINAMATH_GPT_number_of_small_branches_l159_15961

-- Define the number of small branches grown by each branch as a variable
variable (x : ℕ)

-- Define the total number of main stems, branches, and small branches
def total := 1 + x + x * x

theorem number_of_small_branches (h : total x = 91) : x = 9 :=
by
  -- Proof is not required as per instructions
  sorry

end NUMINAMATH_GPT_number_of_small_branches_l159_15961


namespace NUMINAMATH_GPT_compute_expression_l159_15957

noncomputable def quadratic_roots (p q : ℝ) (α β γ δ : ℝ) : Prop :=
  (α * β = -2) ∧ (α + β = -p) ∧ (γ * δ = -2) ∧ (γ + δ = -q)

theorem compute_expression (p q α β γ δ : ℝ) 
  (h₁ : quadratic_roots p q α β γ δ) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = -(p^2 - q^2) :=
by
  -- We will provide the proof here
  sorry

end NUMINAMATH_GPT_compute_expression_l159_15957


namespace NUMINAMATH_GPT_triangle_sin_ratio_cos_side_l159_15945

noncomputable section

variables (A B C a b c : ℝ)
variables (h1 : a + b + c = 5)
variables (h2 : Real.cos B = 1 / 4)
variables (h3 : Real.cos A - 2 * Real.cos C = (2 * c - a) / b * Real.cos B)

theorem triangle_sin_ratio_cos_side :
  (Real.sin C / Real.sin A = 2) ∧ (b = 2) :=
  sorry

end NUMINAMATH_GPT_triangle_sin_ratio_cos_side_l159_15945


namespace NUMINAMATH_GPT_inverse_proportion_quadrants_l159_15998

theorem inverse_proportion_quadrants (k : ℝ) :
  (∀ (x : ℝ), x ≠ 0 → ((x > 0 → (k - 3) / x > 0) ∧ (x < 0 → (k - 3) / x < 0))) → k > 3 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_inverse_proportion_quadrants_l159_15998


namespace NUMINAMATH_GPT_minimum_value_of_quadratic_l159_15988

theorem minimum_value_of_quadratic (x : ℝ) : ∃ (y : ℝ), (∀ x : ℝ, y ≤ x^2 + 2) ∧ (y = 2) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_quadratic_l159_15988


namespace NUMINAMATH_GPT_average_birth_rate_l159_15913

theorem average_birth_rate (B : ℕ) (death_rate : ℕ) (net_increase : ℕ) (seconds_per_day : ℕ) 
  (two_sec_intervals : ℕ) (H1 : death_rate = 2) (H2 : net_increase = 86400) (H3 : seconds_per_day = 86400) 
  (H4 : two_sec_intervals = seconds_per_day / 2) 
  (H5 : net_increase = (B - death_rate) * two_sec_intervals) : B = 4 := 
by 
  sorry

end NUMINAMATH_GPT_average_birth_rate_l159_15913


namespace NUMINAMATH_GPT_fewer_students_played_thursday_l159_15920

variable (w t : ℕ)

theorem fewer_students_played_thursday (h1 : w = 37) (h2 : w + t = 65) : w - t = 9 :=
by
  sorry

end NUMINAMATH_GPT_fewer_students_played_thursday_l159_15920


namespace NUMINAMATH_GPT_fewer_miles_per_gallon_city_l159_15938

-- Define the given conditions.
def miles_per_tankful_highway : ℕ := 420
def miles_per_tankful_city : ℕ := 336
def miles_per_gallon_city : ℕ := 24

-- Define the question as a theorem that proves how many fewer miles per gallon in the city compared to the highway.
theorem fewer_miles_per_gallon_city (G : ℕ) (hG : G = miles_per_tankful_city / miles_per_gallon_city) :
  miles_per_tankful_highway / G - miles_per_gallon_city = 6 :=
by
  -- The proof will be provided here.
  sorry

end NUMINAMATH_GPT_fewer_miles_per_gallon_city_l159_15938


namespace NUMINAMATH_GPT_sum_of_excluded_numbers_l159_15946

theorem sum_of_excluded_numbers (S : ℕ) (X : ℕ) (n m : ℕ) (averageN : ℕ) (averageM : ℕ)
  (h1 : S = 34 * 8) 
  (h2 : n = 8) 
  (h3 : m = 6) 
  (h4 : averageN = 34) 
  (h5 : averageM = 29) 
  (hS : S = n * averageN) 
  (hX : S - X = m * averageM) : 
  X = 98 := by
  sorry

end NUMINAMATH_GPT_sum_of_excluded_numbers_l159_15946


namespace NUMINAMATH_GPT_bill_picked_apples_l159_15922

-- Definitions from conditions
def children := 2
def apples_per_child_per_teacher := 3
def favorite_teachers := 2
def apples_per_pie := 10
def pies_baked := 2
def apples_left := 24

-- Number of apples given to teachers
def apples_for_teachers := children * apples_per_child_per_teacher * favorite_teachers

-- Number of apples used for pies
def apples_for_pies := pies_baked * apples_per_pie

-- The final theorem to be stated
theorem bill_picked_apples :
  apples_for_teachers + apples_for_pies + apples_left = 56 := 
sorry

end NUMINAMATH_GPT_bill_picked_apples_l159_15922


namespace NUMINAMATH_GPT_find_fx_l159_15952

theorem find_fx (f : ℝ → ℝ) 
  (h1 : ∀ x > 0, f (-x) = -(2 * x - 3)) 
  (h2 : ∀ x < 0, -f x = f (-x)) :
  ∀ x < 0, f x = 2 * x + 3 :=
by
  sorry

end NUMINAMATH_GPT_find_fx_l159_15952


namespace NUMINAMATH_GPT_tiffany_bags_difference_l159_15932

theorem tiffany_bags_difference : 
  ∀ (monday_bags next_day_bags : ℕ), monday_bags = 7 → next_day_bags = 12 → next_day_bags - monday_bags = 5 := 
by
  intros monday_bags next_day_bags h1 h2
  sorry

end NUMINAMATH_GPT_tiffany_bags_difference_l159_15932


namespace NUMINAMATH_GPT_negative_integer_solution_l159_15902

theorem negative_integer_solution (M : ℤ) (h1 : 2 * M^2 + M = 12) (h2 : M < 0) : M = -4 :=
sorry

end NUMINAMATH_GPT_negative_integer_solution_l159_15902


namespace NUMINAMATH_GPT_math_problem_l159_15921

noncomputable def problem_statement : Prop :=
  ∃ b c : ℝ, 
  (∀ x : ℝ, (x^2 - b * x + c < 0) ↔ (-3 < x ∧ x < 2)) ∧ 
  (b + c = -7)

theorem math_problem : problem_statement := 
by
  sorry

end NUMINAMATH_GPT_math_problem_l159_15921


namespace NUMINAMATH_GPT_find_b_plus_m_l159_15999

def line1 (m : ℝ) (x : ℝ) : ℝ := m * x + 7
def line2 (b : ℝ) (x : ℝ) : ℝ := 4 * x + b

theorem find_b_plus_m :
  ∃ (m b : ℝ), line1 m 8 = 11 ∧ line2 b 8 = 11 ∧ b + m = -20.5 :=
sorry

end NUMINAMATH_GPT_find_b_plus_m_l159_15999


namespace NUMINAMATH_GPT_seven_divides_n_l159_15987

theorem seven_divides_n (n : ℕ) (h1 : n ≥ 2) (h2 : n ∣ 3^n + 4^n) : 7 ∣ n :=
sorry

end NUMINAMATH_GPT_seven_divides_n_l159_15987


namespace NUMINAMATH_GPT_value_of_y_l159_15928

theorem value_of_y : (∃ y : ℝ, (1 / 3 - 1 / 4 = 4 / y) ∧ y = 48) :=
by
  sorry

end NUMINAMATH_GPT_value_of_y_l159_15928


namespace NUMINAMATH_GPT_expected_interval_is_correct_l159_15907

-- Define the travel times via northern and southern routes
def travel_time_north : ℝ := 17
def travel_time_south : ℝ := 11

-- Define the average time difference between train arrivals
noncomputable def avg_time_diff : ℝ := 1.25

-- The average time difference for traveling from home to work versus work to home
noncomputable def time_diff_home_to_work : ℝ := 1

-- Define the expected interval between trains
noncomputable def expected_interval_between_trains := 3

-- Proof problem statement
theorem expected_interval_is_correct :
  ∃ (T : ℝ), (T = expected_interval_between_trains)
  → (travel_time_north - travel_time_south + 2 * avg_time_diff = time_diff_home_to_work)
  → (T = 3) := 
by
  use 3 
  intro h1 h2
  sorry

end NUMINAMATH_GPT_expected_interval_is_correct_l159_15907


namespace NUMINAMATH_GPT_who_finished_in_7th_place_l159_15973

theorem who_finished_in_7th_place:
  ∀ (Alex Ben Charlie David Ethan : ℕ),
  (Ethan + 4 = Alex) →
  (David + 1 = Ben) →
  (Charlie = Ben + 3) →
  (Alex = Ben + 2) →
  (Ethan + 2 = David) →
  (Ben = 5) →
  Alex = 7 :=
by
  intros Alex Ben Charlie David Ethan h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_who_finished_in_7th_place_l159_15973


namespace NUMINAMATH_GPT_translated_parabola_eq_l159_15983

-- Define the original parabola
def orig_parabola (x : ℝ) : ℝ := -2 * x^2

-- Define the translation function
def translate_upwards (f : ℝ → ℝ) (dy : ℝ) : (ℝ → ℝ) :=
  fun x => f x + dy

-- Define the translated parabola
def translated_parabola := translate_upwards orig_parabola 3

-- State the theorem
theorem translated_parabola_eq:
  translated_parabola = (fun x : ℝ => -2 * x^2 + 3) :=
by
  sorry

end NUMINAMATH_GPT_translated_parabola_eq_l159_15983


namespace NUMINAMATH_GPT_Skylar_chickens_less_than_triple_Colten_l159_15942

def chickens_count (S Q C : ℕ) : Prop := 
  Q + S + C = 383 ∧ 
  Q = 2 * S + 25 ∧ 
  C = 37

theorem Skylar_chickens_less_than_triple_Colten (S Q C : ℕ) 
  (h : chickens_count S Q C) : (3 * C - S = 4) := 
sorry

end NUMINAMATH_GPT_Skylar_chickens_less_than_triple_Colten_l159_15942


namespace NUMINAMATH_GPT_slices_per_sandwich_l159_15955

theorem slices_per_sandwich (total_sandwiches : ℕ) (total_slices : ℕ) (h1 : total_sandwiches = 5) (h2 : total_slices = 15) :
  total_slices / total_sandwiches = 3 :=
by sorry

end NUMINAMATH_GPT_slices_per_sandwich_l159_15955


namespace NUMINAMATH_GPT_proof_l159_15968

open Set

variable (U M P : Set ℕ)

noncomputable def prob_statement : Prop :=
  let C_U (A : Set ℕ) : Set ℕ := {x ∈ U | x ∉ A}
  U = {1,2,3,4,5,6,7,8} ∧ M = {2,3,4} ∧ P = {1,3,6} ∧ C_U (M ∪ P) = {5,7,8}

theorem proof : prob_statement {1,2,3,4,5,6,7,8} {2,3,4} {1,3,6} :=
by
  sorry

end NUMINAMATH_GPT_proof_l159_15968


namespace NUMINAMATH_GPT_percent_fair_hair_l159_15960

theorem percent_fair_hair (total_employees : ℕ) (total_women_fair_hair : ℕ)
  (percent_fair_haired_women : ℕ) (percent_women_fair_hair : ℕ)
  (h1 : total_women_fair_hair = (total_employees * percent_women_fair_hair) / 100)
  (h2 : percent_fair_haired_women * total_women_fair_hair = total_employees * 10) :
  (25 * total_employees = 100 * total_women_fair_hair) :=
by {
  sorry
}

end NUMINAMATH_GPT_percent_fair_hair_l159_15960


namespace NUMINAMATH_GPT_recycling_drive_target_l159_15997

-- Define the collection totals for each section
def section_collections_first_week : List ℝ := [260, 290, 250, 270, 300, 310, 280, 265]

-- Compute total collection for the first week
def total_first_week (collections: List ℝ) : ℝ := collections.sum

-- Compute collection for the second week with a 10% increase
def second_week_increase (collection: ℝ) : ℝ := collection * 1.10
def total_second_week (collections: List ℝ) : ℝ := (collections.map second_week_increase).sum

-- Compute collection for the third week with a 30% increase from the second week
def third_week_increase (collection: ℝ) : ℝ := collection * 1.30
def total_third_week (collections: List ℝ) : ℝ := (collections.map (second_week_increase)).sum * 1.30

-- Total target collection is the sum of collections for three weeks
def target (collections: List ℝ) : ℝ := total_first_week collections + total_second_week collections + total_third_week collections

-- Main theorem to prove
theorem recycling_drive_target : target section_collections_first_week = 7854.25 :=
by
  sorry -- skipping the proof

end NUMINAMATH_GPT_recycling_drive_target_l159_15997


namespace NUMINAMATH_GPT_initial_birds_l159_15911

-- Given conditions
def number_birds_initial (x : ℕ) : Prop :=
  ∃ (y : ℕ), y = 4 ∧ (x + y = 6)

-- Proof statement
theorem initial_birds : ∃ x : ℕ, number_birds_initial x ↔ x = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_initial_birds_l159_15911


namespace NUMINAMATH_GPT_no_integer_roots_l159_15918

theorem no_integer_roots (a b : ℤ) : ¬∃ u : ℤ, u^2 + 3 * a * u + 3 * (2 - b^2) = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_integer_roots_l159_15918


namespace NUMINAMATH_GPT_sequence_contains_infinite_squares_l159_15930

theorem sequence_contains_infinite_squares :
  ∃ f : ℕ → ℕ, ∀ m : ℕ, ∃ n : ℕ, f (n + m) * f (n + m) = 1 + 17 * (n + m) ^ 2 :=
sorry

end NUMINAMATH_GPT_sequence_contains_infinite_squares_l159_15930


namespace NUMINAMATH_GPT_math_problem_l159_15970

variable {x y : ℝ}
variable (hx : x ≠ 0) (hy : y ≠ 0) (h := y^2 - 1 / x^2 ≠ 0) (h₁ := x^2 * y^2 ≠ 1)

theorem math_problem :
  (x^2 - 1 / y^2) / (y^2 - 1 / x^2) = x^2 / y^2 :=
sorry

end NUMINAMATH_GPT_math_problem_l159_15970


namespace NUMINAMATH_GPT_evaluate_expression_l159_15903

theorem evaluate_expression (x : ℕ) (h : x = 5) : 2 * x ^ 2 + 5 = 55 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l159_15903


namespace NUMINAMATH_GPT_problem_1_problem_2_l159_15916

variable {a : ℕ → ℝ}
variable (n : ℕ)

-- Conditions of the problem
def seq_positive : ∀ (k : ℕ), a k > 0 := sorry
def a1 : a 1 = 1 := sorry
def recurrence (n : ℕ) : a (n + 1) = (a n + 1) / (12 * a n) := sorry

-- Proofs to be provided
theorem problem_1 : ∀ n : ℕ, a (2 * n + 1) < a (2 * n - 1) := 
by 
  apply sorry 

theorem problem_2 : ∀ n : ℕ, 1 / 6 ≤ a n ∧ a n ≤ 1 := 
by 
  apply sorry 

end NUMINAMATH_GPT_problem_1_problem_2_l159_15916


namespace NUMINAMATH_GPT_sum_of_b_is_negative_twelve_l159_15936

-- Conditions: the quadratic equation and its property having exactly one solution
def quadratic_equation (b : ℝ) : Prop :=
  ∀ x : ℝ, 3 * x^2 + b * x + 6 * x + 10 = 0

-- Statement to prove: sum of the values of b is -12, 
-- given the condition that the equation has exactly one solution
theorem sum_of_b_is_negative_twelve :
  ∀ b1 b2 : ℝ, (quadratic_equation b1 ∧ quadratic_equation b2) ∧
  (∀ x : ℝ, 3 * x^2 + (b1 + 6) * x + 10 = 0 ∧ 3 * x^2 + (b2 + 6) * x + 10 = 0) ∧
  (∀ b : ℝ, b = b1 ∨ b = b2) →
  b1 + b2 = -12 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_b_is_negative_twelve_l159_15936


namespace NUMINAMATH_GPT_distinct_real_roots_l159_15977

noncomputable def g (x d : ℝ) : ℝ := x^2 + 4*x + d

theorem distinct_real_roots (d : ℝ) :
  (∃! x : ℝ, g (g x d) d = 0) ↔ d = 0 :=
sorry

end NUMINAMATH_GPT_distinct_real_roots_l159_15977


namespace NUMINAMATH_GPT_circle_chord_length_equal_l159_15906

def equation_of_circle (D E F : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0

def distances_equal (D E F : ℝ) : Prop :=
  (D^2 ≠ E^2 ∧ E^2 > 4 * F) → 
  (∀ x y : ℝ, (x^2 + y^2 + D * x + E * y + F = 0) → (x = -D/2) ∧ (y = -E/2) → (abs x = abs y))

theorem circle_chord_length_equal (D E F : ℝ) (h : D^2 ≠ E^2 ∧ E^2 > 4 * F) :
  distances_equal D E F :=
by
  sorry

end NUMINAMATH_GPT_circle_chord_length_equal_l159_15906


namespace NUMINAMATH_GPT_price_difference_is_correct_l159_15940

noncomputable def total_cost : ℝ := 70.93
noncomputable def cost_of_pants : ℝ := 34.0
noncomputable def cost_of_belt : ℝ := total_cost - cost_of_pants
noncomputable def price_difference : ℝ := cost_of_belt - cost_of_pants

theorem price_difference_is_correct :
  price_difference = 2.93 := by
  sorry

end NUMINAMATH_GPT_price_difference_is_correct_l159_15940


namespace NUMINAMATH_GPT_addition_comm_subtraction_compare_multiplication_comm_subtraction_greater_l159_15944

theorem addition_comm (a b : ℕ) : a + b = b + a :=
by sorry

theorem subtraction_compare {a b c : ℕ} (h1 : a < b) (h2 : c = 28) : 56 - c < 65 - c :=
by sorry

theorem multiplication_comm (a b : ℕ) : a * b = b * a :=
by sorry

theorem subtraction_greater {a b c : ℕ} (h1 : a - b = 18) (h2 : a - c = 27) (h3 : 32 = b) (h4 : 23 = c) : a - b > a - c :=
by sorry

end NUMINAMATH_GPT_addition_comm_subtraction_compare_multiplication_comm_subtraction_greater_l159_15944


namespace NUMINAMATH_GPT_raghu_investment_l159_15910

-- Define the conditions as Lean definitions
def invest_raghu : Real := sorry
def invest_trishul := 0.90 * invest_raghu
def invest_vishal := 1.10 * invest_trishul
def invest_chandni := 1.15 * invest_vishal
def total_investment := invest_raghu + invest_trishul + invest_vishal + invest_chandni

-- State the proof problem
theorem raghu_investment (h : total_investment = 10700) : invest_raghu = 2656.25 :=
by
  sorry

end NUMINAMATH_GPT_raghu_investment_l159_15910


namespace NUMINAMATH_GPT_increase_in_average_weight_l159_15964

variable (A : ℝ)

theorem increase_in_average_weight (h1 : ∀ (A : ℝ), 4 * A - 65 + 71 = 4 * (A + 1.5)) :
  (71 - 65) / 4 = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_increase_in_average_weight_l159_15964


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l159_15914

variable (a b : ℝ)

theorem sufficient_but_not_necessary_condition (h : |b| + a < 0) : b^2 < a^2 :=
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l159_15914


namespace NUMINAMATH_GPT_girls_more_than_boys_by_155_l159_15915

def number_of_girls : Real := 542.0
def number_of_boys : Real := 387.0
def difference : Real := number_of_girls - number_of_boys

theorem girls_more_than_boys_by_155 :
  difference = 155.0 := 
by
  sorry

end NUMINAMATH_GPT_girls_more_than_boys_by_155_l159_15915


namespace NUMINAMATH_GPT_minimum_value_expression_l159_15900

noncomputable def expr (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2 - 2*x - 2*y + 2) + 
  Real.sqrt (x^2 + y^2 - 2*x + 4*y + 2*Real.sqrt 3*y + 8 + 4*Real.sqrt 3) +
  Real.sqrt (x^2 + y^2 + 8*x + 4*Real.sqrt 3*x - 4*y + 32 + 16*Real.sqrt 3)

theorem minimum_value_expression : (∃ x y : ℝ, expr x y = 3*Real.sqrt 6 + 4*Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_minimum_value_expression_l159_15900


namespace NUMINAMATH_GPT_total_votes_l159_15959

-- Define the conditions
variables (V : ℝ) (votes_second_candidate : ℝ) (percent_second_candidate : ℝ)
variables (h1 : votes_second_candidate = 240)
variables (h2 : percent_second_candidate = 0.30)

-- Statement: The total number of votes is 800 given the conditions.
theorem total_votes (h : percent_second_candidate * V = votes_second_candidate) : V = 800 :=
sorry

end NUMINAMATH_GPT_total_votes_l159_15959


namespace NUMINAMATH_GPT_number_of_cakes_sold_l159_15976

namespace Bakery

variables (cakes pastries sold_cakes sold_pastries : ℕ)

-- Defining the conditions
def pastries_sold := 154
def more_pastries_than_cakes := 76

-- Defining the problem statement
theorem number_of_cakes_sold (h1 : sold_pastries = pastries_sold) 
                             (h2 : sold_pastries = sold_cakes + more_pastries_than_cakes) : 
                             sold_cakes = 78 :=
by {
  sorry
}

end Bakery

end NUMINAMATH_GPT_number_of_cakes_sold_l159_15976


namespace NUMINAMATH_GPT_merchant_discount_l159_15917

-- Definitions based on conditions
def original_price : ℝ := 1
def increased_price : ℝ := original_price * 1.2
def final_price : ℝ := increased_price * 0.8
def actual_discount : ℝ := original_price - final_price

-- The theorem to be proved
theorem merchant_discount : actual_discount = 0.04 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_merchant_discount_l159_15917


namespace NUMINAMATH_GPT_periodicity_iff_condition_l159_15904

-- Define the given conditions
variable (f : ℝ → ℝ)
variable (h_even : ∀ x, f (-x) = f x)

-- State the problem
theorem periodicity_iff_condition :
  (∀ x, f (1 - x) = f (1 + x)) ↔ (∀ x, f (x + 2) = f x) :=
sorry

end NUMINAMATH_GPT_periodicity_iff_condition_l159_15904


namespace NUMINAMATH_GPT_factorize_expression_l159_15949

theorem factorize_expression (x : ℝ) : 2 * x^3 - 8 * x^2 + 8 * x = 2 * x * (x - 2) ^ 2 := 
sorry

end NUMINAMATH_GPT_factorize_expression_l159_15949


namespace NUMINAMATH_GPT_production_time_l159_15962

variable (a m : ℝ) -- Define a and m as real numbers

-- State the problem as a theorem in Lean
theorem production_time : (a / m) * 200 = 200 * (a / m) := by
  sorry

end NUMINAMATH_GPT_production_time_l159_15962


namespace NUMINAMATH_GPT_max_cookies_Andy_eats_l159_15927

theorem max_cookies_Andy_eats (cookies_total : ℕ) (h_cookies_total : cookies_total = 30) 
  (exists_pos_a : ∃ a : ℕ, a > 0 ∧ 3 * a = 30 - a ∧ (∃ k : ℕ, 3 * a = k ∧ ∃ m : ℕ, a = m)) 
  : ∃ max_a : ℕ, max_a ≤ 7 ∧ 3 * max_a < cookies_total ∧ 3 * max_a ∣ cookies_total ∧ max_a = 6 :=
by
  sorry

end NUMINAMATH_GPT_max_cookies_Andy_eats_l159_15927


namespace NUMINAMATH_GPT_cost_price_of_cloth_l159_15929

-- Definitions for conditions
def sellingPrice (totalMeters : ℕ) : ℕ := 8500
def profitPerMeter : ℕ := 15
def totalMeters : ℕ := 85

-- Proof statement with conditions and expected proof
theorem cost_price_of_cloth : 
  (sellingPrice totalMeters) = 8500 -> 
  profitPerMeter = 15 -> 
  totalMeters = 85 -> 
  (8500 - (profitPerMeter * totalMeters)) / totalMeters = 85 := 
by 
  sorry

end NUMINAMATH_GPT_cost_price_of_cloth_l159_15929


namespace NUMINAMATH_GPT_open_box_volume_l159_15990

theorem open_box_volume (l w s : ℕ) (h1 : l = 50)
  (h2 : w = 36) (h3 : s = 8) : (l - 2 * s) * (w - 2 * s) * s = 5440 :=
by {
  sorry
}

end NUMINAMATH_GPT_open_box_volume_l159_15990


namespace NUMINAMATH_GPT_determine_fake_coin_l159_15912

theorem determine_fake_coin (N : ℕ) : 
  (∃ (n : ℕ), N = 2 * n + 2) ↔ (∃ (n : ℕ), N = 2 * n + 2) := by 
  sorry

end NUMINAMATH_GPT_determine_fake_coin_l159_15912
