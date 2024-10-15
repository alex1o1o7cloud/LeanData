import Mathlib

namespace NUMINAMATH_GPT_factorize_expr_l2037_203797

-- Define the expression
def expr (a : ℝ) := -3 * a + 12 * a^2 - 12 * a^3

-- State the theorem
theorem factorize_expr (a : ℝ) : expr a = -3 * a * (1 - 2 * a)^2 :=
by
  sorry

end NUMINAMATH_GPT_factorize_expr_l2037_203797


namespace NUMINAMATH_GPT_total_budget_is_correct_l2037_203776

-- Define the costs of TV, fridge, and computer based on the given conditions
def cost_tv : ℕ := 600
def cost_computer : ℕ := 250
def cost_fridge : ℕ := cost_computer + 500

-- Statement to prove the total budget
theorem total_budget_is_correct : cost_tv + cost_computer + cost_fridge = 1600 :=
by
  sorry

end NUMINAMATH_GPT_total_budget_is_correct_l2037_203776


namespace NUMINAMATH_GPT_average_after_modifications_l2037_203768

theorem average_after_modifications (S : ℕ) (sum_initial : S = 1080)
  (sum_after_removals : S - 80 - 85 = 915)
  (sum_after_additions : 915 + 75 + 75 = 1065) :
  (1065 / 12 : ℚ) = 88.75 :=
by sorry

end NUMINAMATH_GPT_average_after_modifications_l2037_203768


namespace NUMINAMATH_GPT_equal_share_payment_l2037_203745

theorem equal_share_payment (A B C : ℝ) (h : A < B) (h2 : B < C) :
  (B + C + (A + C - 2 * B) / 3) + (A + C - 2 * B / 3) = 2 * C - A - B / 3 :=
sorry

end NUMINAMATH_GPT_equal_share_payment_l2037_203745


namespace NUMINAMATH_GPT_cooler_capacity_l2037_203796

theorem cooler_capacity (linemen: ℕ) (linemen_drink: ℕ) 
                        (skill_position: ℕ) (skill_position_drink: ℕ) 
                        (linemen_count: ℕ) (skill_position_count: ℕ) 
                        (skill_wait: ℕ) 
                        (h1: linemen_count = 12) 
                        (h2: linemen_drink = 8) 
                        (h3: skill_position_count = 10) 
                        (h4: skill_position_drink = 6) 
                        (h5: skill_wait = 5):
 linemen_count * linemen_drink + skill_wait * skill_position_drink = 126 :=
by
  sorry

end NUMINAMATH_GPT_cooler_capacity_l2037_203796


namespace NUMINAMATH_GPT_circumscribed_sphere_surface_area_l2037_203752

theorem circumscribed_sphere_surface_area
  (x y z : ℝ)
  (h1 : x * y = Real.sqrt 6)
  (h2 : y * z = Real.sqrt 2)
  (h3 : z * x = Real.sqrt 3) :
  let l := Real.sqrt (x^2 + y^2 + z^2)
  let R := l / 2
  4 * Real.pi * R^2 = 6 * Real.pi :=
by sorry

end NUMINAMATH_GPT_circumscribed_sphere_surface_area_l2037_203752


namespace NUMINAMATH_GPT_profit_percentage_l2037_203763

theorem profit_percentage (SP : ℝ) (CP : ℝ) (hSP : SP = 100) (hCP : CP = 83.33) :
    (SP - CP) / CP * 100 = 20 :=
by
  rw [hSP, hCP]
  norm_num
  sorry

end NUMINAMATH_GPT_profit_percentage_l2037_203763


namespace NUMINAMATH_GPT_length_of_escalator_l2037_203713

-- Given conditions
def escalator_speed : ℝ := 12 -- ft/sec
def person_speed : ℝ := 8 -- ft/sec
def time : ℝ := 8 -- seconds

-- Length of the escalator
def length : ℝ := 160 -- feet

-- Theorem stating the length of the escalator given the conditions
theorem length_of_escalator
  (h1 : escalator_speed = 12)
  (h2 : person_speed = 8)
  (h3 : time = 8)
  (combined_speed := escalator_speed + person_speed) :
  combined_speed * time = length :=
by
  -- Here the proof would go, but it's omitted as per instructions
  sorry

end NUMINAMATH_GPT_length_of_escalator_l2037_203713


namespace NUMINAMATH_GPT_find_n_l2037_203764

variable (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (n : ℕ)

def isArithmeticSeq (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sumTo (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n, S n = (n + 1) * a 0 + (n * (n + 1) / 2) * (a 1 - a 0)

theorem find_n 
  (h_arith : isArithmeticSeq a)
  (h_a2 : a 2 = 2) 
  (h_S_diff : ∀ n, n > 3 → S n - S (n - 3) = 54)
  (h_Sn : S n = 100)
  : n = 10 := 
by
  sorry

end NUMINAMATH_GPT_find_n_l2037_203764


namespace NUMINAMATH_GPT_cubic_sum_of_reciprocals_roots_l2037_203731

theorem cubic_sum_of_reciprocals_roots :
  ∀ (a b c : ℝ),
  a ≠ b → b ≠ c → c ≠ a →
  0 < a ∧ a < 1 → 0 < b ∧ b < 1 → 0 < c ∧ c < 1 →
  (24 * a^3 - 38 * a^2 + 18 * a - 1 = 0) ∧
  (24 * b^3 - 38 * b^2 + 18 * b - 1 = 0) ∧
  (24 * c^3 - 38 * c^2 + 18 * c - 1 = 0) →
  ((1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) = 2 / 3) :=
by intros a b c neq_ab neq_bc neq_ca a_range b_range c_range roots_eqns
   sorry

end NUMINAMATH_GPT_cubic_sum_of_reciprocals_roots_l2037_203731


namespace NUMINAMATH_GPT_correct_sum_rounded_l2037_203718

-- Define the conditions: sum and rounding
def sum_58_46 : ℕ := 58 + 46
def round_to_nearest_hundred (n : ℕ) : ℕ :=
  if n % 100 >= 50 then ((n / 100) + 1) * 100 else (n / 100) * 100

-- state the theorem
theorem correct_sum_rounded :
  round_to_nearest_hundred sum_58_46 = 100 :=
by
  sorry

end NUMINAMATH_GPT_correct_sum_rounded_l2037_203718


namespace NUMINAMATH_GPT_min_polyline_distance_l2037_203711

-- Define the polyline distance between two points P(x1, y1) and Q(x2, y2).
noncomputable def polyline_distance (P Q : ℝ × ℝ) : ℝ :=
  |P.1 - Q.1| + |P.2 - Q.2|

-- Define the circle x^2 + y^2 = 1.
def on_circle (P : ℝ × ℝ) : Prop :=
  P.1 ^ 2 + P.2 ^ 2 = 1

-- Define the line 2x + y = 2√5.
def on_line (P : ℝ × ℝ) : Prop :=
  2 * P.1 + P.2 = 2 * Real.sqrt 5

-- Statement of the minimum distance problem.
theorem min_polyline_distance : 
  ∀ P Q : ℝ × ℝ, on_circle P → on_line Q → 
  polyline_distance P Q ≥ Real.sqrt 5 / 2 :=
sorry

end NUMINAMATH_GPT_min_polyline_distance_l2037_203711


namespace NUMINAMATH_GPT_given_conditions_implies_a1d1_a2d2_a3d3_eq_zero_l2037_203710

theorem given_conditions_implies_a1d1_a2d2_a3d3_eq_zero
  (a1 a2 a3 d1 d2 d3 : ℝ)
  (h : ∀ x : ℝ, 
    x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 =
    (x^2 + a1 * x + d1) * (x^2 + a2 * x + d2) * (x^2 + a3 * x + d3) * (x^2 - x + 1)) :
  a1 * d1 + a2 * d2 + a3 * d3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_given_conditions_implies_a1d1_a2d2_a3d3_eq_zero_l2037_203710


namespace NUMINAMATH_GPT_christen_potatoes_l2037_203784

theorem christen_potatoes :
  let total_potatoes := 60
  let homer_rate := 4
  let christen_rate := 6
  let alex_potatoes := 2
  let homer_minutes := 6
  homer_minutes * homer_rate + christen_rate * ((total_potatoes + alex_potatoes - homer_minutes * homer_rate) / (homer_rate + christen_rate)) = 24 := 
sorry

end NUMINAMATH_GPT_christen_potatoes_l2037_203784


namespace NUMINAMATH_GPT_speed_excluding_stoppages_l2037_203741

-- Conditions
def speed_with_stoppages := 33 -- kmph
def stoppage_time_per_hour := 16 -- minutes

-- Conversion of conditions to statements
def running_time_per_hour := 60 - stoppage_time_per_hour -- minutes
def running_time_in_hours := running_time_per_hour / 60 -- hours

-- Proof Statement
theorem speed_excluding_stoppages : 
  (speed_with_stoppages = 33) → (stoppage_time_per_hour = 16) → (75 = 33 / (44 / 60)) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_speed_excluding_stoppages_l2037_203741


namespace NUMINAMATH_GPT_calculate_sequences_l2037_203765

-- Definitions of sequences and constants
def a (n : ℕ) := 2 * n + 1
def b (n : ℕ) := 3 ^ n
def S (n : ℕ) := n * (n + 2)
def T (n : ℕ) := (3 / 4 : ℚ) - (2 * n + 3) / (2 * (n + 1) * (n + 2))

-- Hypotheses and proofs
theorem calculate_sequences (d : ℕ) (a1 : ℕ) (h_d : d = 2) (h_a1 : a1 = 3) :
  ∀ n, (a n = 2 * n + 1) ∧ (b 1 = a 1) ∧ (b 2 = a 4) ∧ (b 3 = a 13) ∧ (b n = 3 ^ n) ∧
  (S n = n * (n + 2)) ∧ (T n = (3 / 4 : ℚ) - (2 * n + 3) / (2 * (n + 1) * (n + 2))) :=
by
  intros
  -- Skipping proof steps with sorry
  sorry

end NUMINAMATH_GPT_calculate_sequences_l2037_203765


namespace NUMINAMATH_GPT_total_students_l2037_203733

theorem total_students (x : ℝ) :
  (x - (1/2)*x - (1/4)*x - (1/8)*x = 3) → x = 24 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_total_students_l2037_203733


namespace NUMINAMATH_GPT_smallest_value_y_l2037_203750

theorem smallest_value_y : ∃ y : ℝ, 3 * y ^ 2 + 33 * y - 90 = y * (y + 18) ∧ (∀ z : ℝ, 3 * z ^ 2 + 33 * z - 90 = z * (z + 18) → y ≤ z) ∧ y = -18 := 
sorry

end NUMINAMATH_GPT_smallest_value_y_l2037_203750


namespace NUMINAMATH_GPT_proportional_function_decreases_l2037_203740

theorem proportional_function_decreases
  (k : ℝ) (h : k ≠ 0) (h_point : ∃ k, (-4 : ℝ) = k * 2) :
  ∀ x1 x2 : ℝ, x1 < x2 → (k * x1) > (k * x2) :=
by
  sorry

end NUMINAMATH_GPT_proportional_function_decreases_l2037_203740


namespace NUMINAMATH_GPT_problem_statement_l2037_203753

-- Define the conditions as Lean predicates
def is_odd (n : ℕ) : Prop := n % 2 = 1
def between_400_and_600 (n : ℕ) : Prop := 400 < n ∧ n < 600
def divisible_by_55 (n : ℕ) : Prop := n % 55 = 0

-- Define a function to calculate the sum of the digits
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + (n % 100 / 10) + (n % 10)

-- Main theorem to prove
theorem problem_statement (N : ℕ)
  (h_odd : is_odd N)
  (h_range : between_400_and_600 N)
  (h_divisible : divisible_by_55 N) :
  sum_of_digits N = 18 :=
sorry

end NUMINAMATH_GPT_problem_statement_l2037_203753


namespace NUMINAMATH_GPT_find_x_given_distance_l2037_203754

theorem find_x_given_distance (x : ℝ) : abs (x - 4) = 1 → (x = 5 ∨ x = 3) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_x_given_distance_l2037_203754


namespace NUMINAMATH_GPT_price_per_pot_l2037_203793

-- Definitions based on conditions
def total_pots : ℕ := 80
def proportion_not_cracked : ℚ := 3 / 5
def total_revenue : ℚ := 1920

-- The Lean statement to prove she sold each clay pot for $40
theorem price_per_pot : (total_revenue / (total_pots * proportion_not_cracked)) = 40 := 
by sorry

end NUMINAMATH_GPT_price_per_pot_l2037_203793


namespace NUMINAMATH_GPT_problem1_problem2_l2037_203794

-- Definitions for the sets and conditions
def setA : Set ℝ := {x | -1 < x ∧ x < 2}
def setB (a : ℝ) : Set ℝ := if a > 0 then {x | x ≤ -2 ∨ x ≥ (1 / a)} else ∅

-- Problem 1: Prove the intersection for a == 1
theorem problem1 : (setB 1) ∩ setA = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

-- Problem 2: Prove the range of a
theorem problem2 (a : ℝ) (h : setB a ⊆ setAᶜ) : 0 < a ∧ a ≤ 1/2 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2037_203794


namespace NUMINAMATH_GPT_quadratic_root_zero_l2037_203722

theorem quadratic_root_zero (k : ℝ) :
    (∃ x : ℝ, x = 0 ∧ (k - 1) * x ^ 2 + 6 * x + k ^ 2 - k = 0) → k = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_zero_l2037_203722


namespace NUMINAMATH_GPT_pyramid_base_length_l2037_203716

theorem pyramid_base_length (A : ℝ) (h : ℝ) (s : ℝ) 
  (hA : A = 120)
  (hh : h = 40)
  (hs : A = 20 * s) : 
  s = 6 :=
by
  sorry

end NUMINAMATH_GPT_pyramid_base_length_l2037_203716


namespace NUMINAMATH_GPT_volume_of_red_tetrahedron_in_colored_cube_l2037_203759

noncomputable def red_tetrahedron_volume (side_length : ℝ) : ℝ :=
  let cube_volume := side_length ^ 3
  let clear_tetrahedron_volume := (1/3) * (1/2 * side_length * side_length) * side_length
  let red_tetrahedron_volume := (cube_volume - 4 * clear_tetrahedron_volume)
  red_tetrahedron_volume

theorem volume_of_red_tetrahedron_in_colored_cube 
: red_tetrahedron_volume 8 = 512 / 3 := by
  sorry

end NUMINAMATH_GPT_volume_of_red_tetrahedron_in_colored_cube_l2037_203759


namespace NUMINAMATH_GPT_problem1_problem2_l2037_203758

-- Definitions for the inequalities
def f (x a : ℝ) : ℝ := abs (x - a) - 1

-- Problem 1: Given a = 2, solve the inequality f(x) + |2x - 3| > 0
theorem problem1 (x : ℝ) (h1 : abs (x - 2) + abs (2 * x - 3) > 1) : (x ≥ 2 ∨ x ≤ 4 / 3) := sorry

-- Problem 2: If the inequality f(x) > |x - 3| has solutions, find the range of a
theorem problem2 (a : ℝ) (h2 : ∃ x : ℝ, abs (x - a) - abs (x - 3) > 1) : a < 2 ∨ a > 4 := sorry

end NUMINAMATH_GPT_problem1_problem2_l2037_203758


namespace NUMINAMATH_GPT_intersection_when_a_minus2_range_of_a_if_A_subset_B_l2037_203792

namespace ProofProblem

open Set

-- Definitions
def A (a : ℝ) : Set ℝ := { x | 2 * a - 1 ≤ x ∧ x ≤ a + 3 }
def B : Set ℝ := { x | x < -1 ∨ x > 5 }

-- Theorem (1)
theorem intersection_when_a_minus2 : 
  A (-2) ∩ B = { x : ℝ | -5 ≤ x ∧ x < -1 } :=
by
  sorry

-- Theorem (2)
theorem range_of_a_if_A_subset_B : 
  A a ⊆ B → (a ∈ Iic (-4) ∨ a ∈ Ici 3) :=
by
  sorry

end ProofProblem

end NUMINAMATH_GPT_intersection_when_a_minus2_range_of_a_if_A_subset_B_l2037_203792


namespace NUMINAMATH_GPT_alyssa_money_after_movies_and_carwash_l2037_203702

theorem alyssa_money_after_movies_and_carwash : 
  ∀ (allowance spent earned : ℕ), 
  allowance = 8 → 
  spent = allowance / 2 → 
  earned = 8 → 
  (allowance - spent + earned = 12) := 
by 
  intros allowance spent earned h_allowance h_spent h_earned 
  rw [h_allowance, h_spent, h_earned] 
  simp 
  sorry

end NUMINAMATH_GPT_alyssa_money_after_movies_and_carwash_l2037_203702


namespace NUMINAMATH_GPT_ice_cream_flavors_l2037_203725

-- Definition of the problem setup
def number_of_flavors : ℕ :=
  let scoops := 5
  let dividers := 2
  let total_objects := scoops + dividers
  Nat.choose total_objects dividers

-- Statement of the theorem
theorem ice_cream_flavors : number_of_flavors = 21 := by
  -- The proof of the theorem will use combinatorics to show the result.
  sorry

end NUMINAMATH_GPT_ice_cream_flavors_l2037_203725


namespace NUMINAMATH_GPT_chairs_stools_legs_l2037_203717

theorem chairs_stools_legs (x : ℕ) (h1 : 4 * x + 3 * (16 - x) = 60) : 4 * x + 3 * (16 - x) = 60 :=
by
  exact h1

end NUMINAMATH_GPT_chairs_stools_legs_l2037_203717


namespace NUMINAMATH_GPT_OBrien_current_hats_l2037_203743

-- Definition of the number of hats that Fire chief Simpson has
def Simpson_hats : ℕ := 15

-- Definition of the number of hats that Policeman O'Brien had before losing one
def OBrien_initial_hats (Simpson_hats : ℕ) : ℕ := 2 * Simpson_hats + 5

-- Final proof statement that Policeman O'Brien now has 34 hats
theorem OBrien_current_hats : OBrien_initial_hats Simpson_hats - 1 = 34 := by
  -- Proof will go here, but is skipped for now
  sorry

end NUMINAMATH_GPT_OBrien_current_hats_l2037_203743


namespace NUMINAMATH_GPT_factorize_expression_l2037_203747

theorem factorize_expression (a x y : ℤ) : a * x - a * y = a * (x - y) :=
  sorry

end NUMINAMATH_GPT_factorize_expression_l2037_203747


namespace NUMINAMATH_GPT_number_added_is_8_l2037_203779

theorem number_added_is_8
  (x y : ℕ)
  (h1 : x = 265)
  (h2 : x / 5 + y = 61) :
  y = 8 :=
by
  sorry

end NUMINAMATH_GPT_number_added_is_8_l2037_203779


namespace NUMINAMATH_GPT_part1_part2_l2037_203746

def f (x : ℝ) := |x + 4| - |x - 1|
def g (x : ℝ) := |2 * x - 1| + 3

theorem part1 (x : ℝ) : (f x > 3) → x > 0 :=
by sorry

theorem part2 (a : ℝ) : (∃ x, f x + 1 < 4^a - 5 * 2^a) ↔ (a < 0 ∨ a > 2) :=
by sorry

end NUMINAMATH_GPT_part1_part2_l2037_203746


namespace NUMINAMATH_GPT_fraction_ratio_l2037_203707

theorem fraction_ratio :
  ∃ (x y : ℕ), y ≠ 0 ∧ (x:ℝ) / (y:ℝ) = 240 / 1547 ∧ ((x:ℝ) / (y:ℝ)) / (2 / 13) = (5 / 34) / (7 / 48) :=
sorry

end NUMINAMATH_GPT_fraction_ratio_l2037_203707


namespace NUMINAMATH_GPT_determine_velocities_l2037_203709

theorem determine_velocities (V1 V2 : ℝ) (h1 : 60 / V2 = 60 / V1 + 5) (h2 : |V1 - V2| = 1)
  (h3 : 0 < V1) (h4 : 0 < V2) : V1 = 4 ∧ V2 = 3 :=
by
  sorry

end NUMINAMATH_GPT_determine_velocities_l2037_203709


namespace NUMINAMATH_GPT_watch_correction_l2037_203788

def watch_loss_per_day : ℚ := 13 / 4

def hours_from_march_15_noon_to_march_22_9am : ℚ := 7 * 24 + 21

def per_hour_loss : ℚ := watch_loss_per_day / 24

def total_loss_in_minutes : ℚ := hours_from_march_15_noon_to_march_22_9am * per_hour_loss

theorem watch_correction :
  total_loss_in_minutes = 2457 / 96 :=
by
  sorry

end NUMINAMATH_GPT_watch_correction_l2037_203788


namespace NUMINAMATH_GPT_equal_volume_cubes_l2037_203799

noncomputable def volume_box : ℝ := 1 -- volume of the cubical box in cubic meters

noncomputable def edge_length_small_cube : ℝ := 0.04 -- edge length of small cubes in meters

noncomputable def number_of_cubes : ℝ := 15624.999999999998 -- number of small cubes

noncomputable def volume_small_cube : ℝ := edge_length_small_cube^3 -- volume of one small cube

theorem equal_volume_cubes : volume_box = volume_small_cube * number_of_cubes :=
  by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_equal_volume_cubes_l2037_203799


namespace NUMINAMATH_GPT_encode_mathematics_l2037_203744

def robotCipherMapping : String → String := sorry

theorem encode_mathematics :
  robotCipherMapping "MATHEMATICS" = "2232331122323323132" := sorry

end NUMINAMATH_GPT_encode_mathematics_l2037_203744


namespace NUMINAMATH_GPT_time_to_be_apart_l2037_203783

noncomputable def speed_A : ℝ := 17.5
noncomputable def speed_B : ℝ := 15
noncomputable def initial_distance : ℝ := 65
noncomputable def final_distance : ℝ := 32.5

theorem time_to_be_apart (x : ℝ) :
  x = 1 ∨ x = 3 ↔ 
  (x * (speed_A + speed_B) = initial_distance - final_distance ∨ 
   x * (speed_A + speed_B) = initial_distance + final_distance) :=
sorry

end NUMINAMATH_GPT_time_to_be_apart_l2037_203783


namespace NUMINAMATH_GPT_first_part_is_13_l2037_203728

-- Definitions for the conditions
variables (x y : ℕ)

-- Conditions given in the problem
def condition1 : Prop := x + y = 24
def condition2 : Prop := 7 * x + 5 * y = 146

-- The theorem we need to prove
theorem first_part_is_13 (h1 : condition1 x y) (h2 : condition2 x y) : x = 13 :=
sorry

end NUMINAMATH_GPT_first_part_is_13_l2037_203728


namespace NUMINAMATH_GPT_min_value_of_sum_squares_l2037_203773

theorem min_value_of_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) :
    x^2 + y^2 + z^2 ≥ 10 :=
sorry

end NUMINAMATH_GPT_min_value_of_sum_squares_l2037_203773


namespace NUMINAMATH_GPT_correct_conclusions_l2037_203736

open Real

noncomputable def parabola (a b c : ℝ) : ℝ → ℝ :=
  λ x => a*x^2 + b*x + c

theorem correct_conclusions (a b c m n : ℝ)
  (h1 : c < 0)
  (h2 : parabola a b c 1 = 1)
  (h3 : parabola a b c m = 0)
  (h4 : parabola a b c n = 0)
  (h5 : n ≥ 3) :
  (4*a*c - b^2 < 4*a) ∧
  (n = 3 → ∃ t : ℝ, parabola a b c 2 = t ∧ t > 1) ∧
  (∀ x : ℝ, parabola a b (c - 1) x = 0 → (0 < m ∧ m ≤ 1/3)) :=
sorry

end NUMINAMATH_GPT_correct_conclusions_l2037_203736


namespace NUMINAMATH_GPT_geometric_sequence_x_l2037_203723

theorem geometric_sequence_x (x : ℝ) (h : 1 * x = x ∧ x * x = 9) : x = 3 ∨ x = -3 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_x_l2037_203723


namespace NUMINAMATH_GPT_club_members_addition_l2037_203727

theorem club_members_addition
  (current_members : ℕ := 10)
  (desired_members : ℕ := 2 * current_members + 5)
  (additional_members : ℕ := desired_members - current_members) :
  additional_members = 15 :=
by
  -- proof placeholder
  sorry

end NUMINAMATH_GPT_club_members_addition_l2037_203727


namespace NUMINAMATH_GPT_largest_expression_value_l2037_203755

-- Definitions of the expressions
def expr_A : ℕ := 3 + 0 + 1 + 8
def expr_B : ℕ := 3 * 0 + 1 + 8
def expr_C : ℕ := 3 + 0 * 1 + 8
def expr_D : ℕ := 3 + 0 + 1^2 + 8
def expr_E : ℕ := 3 * 0 * 1^2 * 8

-- Statement of the theorem
theorem largest_expression_value :
  max expr_A (max expr_B (max expr_C (max expr_D expr_E))) = 12 :=
by
  sorry

end NUMINAMATH_GPT_largest_expression_value_l2037_203755


namespace NUMINAMATH_GPT_ratio_in_two_years_l2037_203791

def son_age : ℕ := 22
def man_age : ℕ := son_age + 24

theorem ratio_in_two_years :
  (man_age + 2) / (son_age + 2) = 2 := 
sorry

end NUMINAMATH_GPT_ratio_in_two_years_l2037_203791


namespace NUMINAMATH_GPT_find_value_of_expression_l2037_203760

theorem find_value_of_expression (x y : ℝ) (h1 : |x| = 2) (h2 : |y| = 3) (h3 : x / y < 0) :
  (2 * x - y = 7) ∨ (2 * x - y = -7) :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_expression_l2037_203760


namespace NUMINAMATH_GPT_mn_not_equal_l2037_203721

-- Define conditions for the problem
def isValidN (N : ℕ) (n : ℕ) : Prop :=
  0 ≤ N ∧ N < 10^n ∧ N % 4 = 0 ∧ ((N.digits 10).sum % 4 = 0)

-- Define the number M_n of integers N satisfying the conditions
noncomputable def countMn (n : ℕ) : ℕ :=
  Nat.card { N : ℕ | isValidN N n }

-- Define the theorem stating the problem's conclusion
theorem mn_not_equal (n : ℕ) (hn : n > 0) : 
  countMn n ≠ 10^n / 16 :=
sorry

end NUMINAMATH_GPT_mn_not_equal_l2037_203721


namespace NUMINAMATH_GPT_rectangle_area_l2037_203751

theorem rectangle_area (L W P A : ℕ) (h1 : P = 52) (h2 : L = 11) (h3 : 2 * L + 2 * W = P) : 
  A = L * W → A = 165 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l2037_203751


namespace NUMINAMATH_GPT_ratio_of_areas_l2037_203724

-- Define the conditions
def angle_Q_smaller_circle : ℝ := 60
def angle_Q_larger_circle : ℝ := 30
def arc_length_equal (C1 C2 : ℝ) : Prop := 
  (angle_Q_smaller_circle / 360) * C1 = (angle_Q_larger_circle / 360) * C2

-- The required Lean statement that proves the ratio of the areas
theorem ratio_of_areas (C1 C2 r1 r2 : ℝ) 
  (arc_eq : arc_length_equal C1 C2) : 
  (π * r1^2) / (π * r2^2) = 1 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l2037_203724


namespace NUMINAMATH_GPT_intersection_of_PQ_RS_correct_l2037_203772

noncomputable def intersection_point (P Q R S : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let t := 1/9
  let s := 2/3
  (3 + 10 * t, -4 - 10 * t, 4 + 5 * t)

theorem intersection_of_PQ_RS_correct :
  let P := (3, -4, 4)
  let Q := (13, -14, 9)
  let R := (-3, 6, -9)
  let S := (1, -2, 7)
  intersection_point P Q R S = (40/9, -76/9, 49/9) :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_of_PQ_RS_correct_l2037_203772


namespace NUMINAMATH_GPT_smallest_five_digit_multiple_of_9_starting_with_7_l2037_203782

theorem smallest_five_digit_multiple_of_9_starting_with_7 :
  ∃ (n : ℕ), (70000 ≤ n ∧ n < 80000) ∧ (n % 9 = 0) ∧ n = 70002 :=
sorry

end NUMINAMATH_GPT_smallest_five_digit_multiple_of_9_starting_with_7_l2037_203782


namespace NUMINAMATH_GPT_train_length_l2037_203732

theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (conversion_factor : ℝ) (speed_ms : ℝ) (distance_m : ℝ) 
  (h1 : speed_kmh = 36) 
  (h2 : time_s = 28)
  (h3 : conversion_factor = 1000 / 3600) -- convert km/hr to m/s
  (h4 : speed_ms = speed_kmh * conversion_factor)
  (h5 : distance_m = speed_ms * time_s) :
  distance_m = 280 := 
by
  sorry

end NUMINAMATH_GPT_train_length_l2037_203732


namespace NUMINAMATH_GPT_largest_possible_A_l2037_203706

-- Define natural numbers
variables (A B C : ℕ)

-- Given conditions
def division_algorithm (A B C : ℕ) : Prop := A = 8 * B + C
def B_equals_C (B C : ℕ) : Prop := B = C

-- The proof statement
theorem largest_possible_A (h1 : division_algorithm A B C) (h2 : B_equals_C B C) : A = 63 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_largest_possible_A_l2037_203706


namespace NUMINAMATH_GPT_find_x_l2037_203789

theorem find_x (x : ℝ) :
  (1 / 3) * ((3 * x + 4) + (7 * x - 5) + (4 * x + 9)) = (5 * x - 3) → x = 17 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2037_203789


namespace NUMINAMATH_GPT_chang_total_apples_l2037_203712

def sweet_apple_price : ℝ := 0.5
def sour_apple_price : ℝ := 0.1
def sweet_apple_percentage : ℝ := 0.75
def sour_apple_percentage : ℝ := 1 - sweet_apple_percentage
def total_earnings : ℝ := 40

theorem chang_total_apples : 
  (total_earnings / (sweet_apple_percentage * sweet_apple_price + sour_apple_percentage * sour_apple_price)) = 100 :=
by
  sorry

end NUMINAMATH_GPT_chang_total_apples_l2037_203712


namespace NUMINAMATH_GPT_sculpture_exposed_surface_area_l2037_203774

theorem sculpture_exposed_surface_area :
  let l₁ := 9
  let l₂ := 6
  let l₃ := 4
  let l₄ := 1

  let exposed_bottom_layer := 9 + 16
  let exposed_second_layer := 6 + 10
  let exposed_third_layer := 4 + 8
  let exposed_top_layer := 5

  l₁ + l₂ + l₃ + l₄ = 20 →
  exposed_bottom_layer + exposed_second_layer + exposed_third_layer + exposed_top_layer = 58 :=
by {
  sorry
}

end NUMINAMATH_GPT_sculpture_exposed_surface_area_l2037_203774


namespace NUMINAMATH_GPT_preimage_of_43_is_21_l2037_203761

def f (x y : ℝ) : ℝ × ℝ := (x + 2 * y, 2 * x - y)

theorem preimage_of_43_is_21 : f 2 1 = (4, 3) :=
by {
  -- Proof omitted
  sorry
}

end NUMINAMATH_GPT_preimage_of_43_is_21_l2037_203761


namespace NUMINAMATH_GPT_min_sum_of_factors_l2037_203708

theorem min_sum_of_factors (a b : ℤ) (h1 : a * b = 72) : a + b ≥ -73 :=
sorry

end NUMINAMATH_GPT_min_sum_of_factors_l2037_203708


namespace NUMINAMATH_GPT_field_trip_cost_l2037_203780

def candy_bar_price : ℝ := 1.25
def candy_bars_sold : ℤ := 188
def money_from_grandma : ℝ := 250

theorem field_trip_cost : (candy_bars_sold * candy_bar_price + money_from_grandma) = 485 := 
by
  sorry

end NUMINAMATH_GPT_field_trip_cost_l2037_203780


namespace NUMINAMATH_GPT_fare_for_90_miles_l2037_203705

noncomputable def fare_cost (miles : ℕ) (base_fare cost_per_mile : ℝ) : ℝ :=
  base_fare + cost_per_mile * miles

theorem fare_for_90_miles (base_fare : ℝ) (cost_per_mile : ℝ)
  (h1 : base_fare = 30)
  (h2 : fare_cost 60 base_fare cost_per_mile = 150)
  (h3 : cost_per_mile = (150 - base_fare) / 60) :
  fare_cost 90 base_fare cost_per_mile = 210 :=
  sorry

end NUMINAMATH_GPT_fare_for_90_miles_l2037_203705


namespace NUMINAMATH_GPT_f_neg_eq_f_l2037_203714

noncomputable def f : ℝ → ℝ := sorry

axiom f_not_identically_zero :
  ∃ x, f x ≠ 0

axiom functional_equation :
  ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a + 2 * f b

theorem f_neg_eq_f (x : ℝ) : f (-x) = f x := 
sorry

end NUMINAMATH_GPT_f_neg_eq_f_l2037_203714


namespace NUMINAMATH_GPT_find_a_l2037_203703

noncomputable def center_radius_circle1 (x y : ℝ) := x^2 + y^2 = 16
noncomputable def center_radius_circle2 (x y a : ℝ) := (x - a)^2 + y^2 = 1
def centers_tangent (a : ℝ) : Prop := |a| = 5 ∨ |a| = 3

theorem find_a (a : ℝ) (h1 : center_radius_circle1 x y) (h2 : center_radius_circle2 x y a) : centers_tangent a :=
sorry

end NUMINAMATH_GPT_find_a_l2037_203703


namespace NUMINAMATH_GPT_taylor_one_basket_probability_l2037_203769

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

end NUMINAMATH_GPT_taylor_one_basket_probability_l2037_203769


namespace NUMINAMATH_GPT_break_even_number_of_books_l2037_203787

-- Definitions from conditions.
def fixed_cost : ℝ := 50000
def variable_cost_per_book : ℝ := 4
def selling_price_per_book : ℝ := 9

-- Main statement proving the break-even point.
theorem break_even_number_of_books 
  (x : ℕ) : (selling_price_per_book * x = fixed_cost + variable_cost_per_book * x) → (x = 10000) :=
by
  sorry

end NUMINAMATH_GPT_break_even_number_of_books_l2037_203787


namespace NUMINAMATH_GPT_solution_set_l2037_203798

noncomputable def satisfies_inequality (x : ℝ) : Prop :=
  (x - 2) / (x - 4) ≥ 3

theorem solution_set :
  {x : ℝ | satisfies_inequality x} = {x : ℝ | 4 < x ∧ x ≤ 5} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l2037_203798


namespace NUMINAMATH_GPT_neg_p_equivalence_l2037_203735

theorem neg_p_equivalence:
  (∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x : ℝ, Real.sin x > 1) :=
sorry

end NUMINAMATH_GPT_neg_p_equivalence_l2037_203735


namespace NUMINAMATH_GPT_part_a_part_a_rev_l2037_203770

variable (x y : ℝ)

theorem part_a (hx : x > 0) (hy : y > 0) : x + y > |x - y| :=
sorry

theorem part_a_rev (h : x + y > |x - y|) : x > 0 ∧ y > 0 :=
sorry

end NUMINAMATH_GPT_part_a_part_a_rev_l2037_203770


namespace NUMINAMATH_GPT_cube_face_sum_l2037_203729

theorem cube_face_sum
  (a d b e c f : ℕ)
  (pos_a : 0 < a) (pos_d : 0 < d) (pos_b : 0 < b) (pos_e : 0 < e) (pos_c : 0 < c) (pos_f : 0 < f)
  (hd : (a + d) * (b + e) * (c + f) = 2107) :
  a + d + b + e + c + f = 57 :=
sorry

end NUMINAMATH_GPT_cube_face_sum_l2037_203729


namespace NUMINAMATH_GPT_four_digit_numbers_permutations_l2037_203775

theorem four_digit_numbers_permutations (a b : ℕ) (h1 : a = 3) (h2 : b = 0) : 
  (if a = 3 ∧ b = 0 then 3 else 0) = 3 :=
by
  sorry

end NUMINAMATH_GPT_four_digit_numbers_permutations_l2037_203775


namespace NUMINAMATH_GPT_matrix_eigenvalue_neg7_l2037_203720

theorem matrix_eigenvalue_neg7 (M : Matrix (Fin 2) (Fin 2) ℝ) :
  (∀ (v : Fin 2 → ℝ), M.mulVec v = -7 • v) →
  M = !![-7, 0; 0, -7] :=
by
  intro h
  -- proof goes here
  sorry

end NUMINAMATH_GPT_matrix_eigenvalue_neg7_l2037_203720


namespace NUMINAMATH_GPT_water_evaporation_l2037_203777

theorem water_evaporation (m : ℝ) 
  (evaporation_day1 : m' = m * (0.1)) 
  (evaporation_day2 : m'' = (m * 0.9) * 0.1) 
  (total_evaporation : total = m' + m'')
  (water_added : 15 = total) 
  : m = 1500 / 19 := by
  sorry

end NUMINAMATH_GPT_water_evaporation_l2037_203777


namespace NUMINAMATH_GPT_number_solution_l2037_203748

theorem number_solution : ∃ x : ℝ, x + 9 = x^2 ∧ x = (1 + Real.sqrt 37) / 2 :=
by
  use (1 + Real.sqrt 37) / 2
  simp
  sorry

end NUMINAMATH_GPT_number_solution_l2037_203748


namespace NUMINAMATH_GPT_point_C_lies_within_region_l2037_203766

def lies_within_region (x y : ℝ) : Prop :=
  (x + y - 1 < 0) ∧ (x - y + 1 > 0)

theorem point_C_lies_within_region : lies_within_region 0 (-2) :=
by {
  -- Proof is omitted as per the instructions
  sorry
}

end NUMINAMATH_GPT_point_C_lies_within_region_l2037_203766


namespace NUMINAMATH_GPT_solve_inequality_system_l2037_203778

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_system_l2037_203778


namespace NUMINAMATH_GPT_sequence_formula_l2037_203781

-- Define the problem when n >= 2
theorem sequence_formula (n : ℕ) (h : n ≥ 2) : 
  1 / (n^2 - 1) = (1 / 2) * (1 / (n - 1) - 1 / (n + 1)) := 
by {
  sorry
}

end NUMINAMATH_GPT_sequence_formula_l2037_203781


namespace NUMINAMATH_GPT_total_fuel_two_weeks_l2037_203704

def fuel_used_this_week : ℝ := 15
def percentage_less_last_week : ℝ := 0.2
def fuel_used_last_week : ℝ := fuel_used_this_week * (1 - percentage_less_last_week)
def total_fuel_used : ℝ := fuel_used_this_week + fuel_used_last_week

theorem total_fuel_two_weeks : total_fuel_used = 27 := 
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_total_fuel_two_weeks_l2037_203704


namespace NUMINAMATH_GPT_population_doubles_l2037_203737

theorem population_doubles (initial_population: ℕ) (initial_year: ℕ) (doubling_period: ℕ) (target_population : ℕ) (target_year : ℕ) : 
  initial_population = 500 → 
  initial_year = 2023 → 
  doubling_period = 20 → 
  target_population = 8000 → 
  target_year = 2103 :=
by 
  sorry

end NUMINAMATH_GPT_population_doubles_l2037_203737


namespace NUMINAMATH_GPT_Claire_photos_is_5_l2037_203738

variable (Claire_photos : ℕ)
variable (Lisa_photos : ℕ := 3 * Claire_photos)
variable (Robert_photos : ℕ := Claire_photos + 10)

theorem Claire_photos_is_5
  (h1 : Lisa_photos = Robert_photos) :
  Claire_photos = 5 :=
by
  sorry

end NUMINAMATH_GPT_Claire_photos_is_5_l2037_203738


namespace NUMINAMATH_GPT_length_AB_proof_l2037_203785

noncomputable def length_AB (AB BC CA : ℝ) (DEF DE EF DF : ℝ) (angle_BAC angle_DEF : ℝ) : ℝ :=
  if h : (angle_BAC = 120 ∧ angle_DEF = 120 ∧ AB = 5 ∧ BC = 17 ∧ CA = 12 ∧ DE = 9 ∧ EF = 15 ∧ DF = 12) then
    (5 * 15) / 17
  else
    0

theorem length_AB_proof : length_AB 5 17 12 9 15 12 120 120 = 75 / 17 := by
  sorry

end NUMINAMATH_GPT_length_AB_proof_l2037_203785


namespace NUMINAMATH_GPT_correct_relation_l2037_203742

open Set

def U : Set ℝ := univ

def A : Set ℝ := { x | x^2 < 4 }

def B : Set ℝ := { x | x > 2 }

def comp_of_B : Set ℝ := U \ B

theorem correct_relation : A ∩ comp_of_B = A := by
  sorry

end NUMINAMATH_GPT_correct_relation_l2037_203742


namespace NUMINAMATH_GPT_sentence_structure_diff_l2037_203757

-- Definitions based on sentence structures.
def sentence_A := "得不焚，殆有神护者" -- passive
def sentence_B := "重为乡党所笑" -- passive
def sentence_C := "而文采不表于后也" -- post-positioned prepositional
def sentence_D := "是以见放" -- passive

-- Definition to check if the given sentence is passive
def is_passive (s : String) : Prop :=
  s = sentence_A ∨ s = sentence_B ∨ s = sentence_D

-- Definition to check if the given sentence is post-positioned prepositional
def is_post_positioned_prepositional (s : String) : Prop :=
  s = sentence_C

-- Theorem to prove
theorem sentence_structure_diff :
  (is_post_positioned_prepositional sentence_C) ∧ ¬(is_passive sentence_C) :=
by
  sorry

end NUMINAMATH_GPT_sentence_structure_diff_l2037_203757


namespace NUMINAMATH_GPT_exponent_multiplication_l2037_203795

variable (a x y : ℝ)

theorem exponent_multiplication :
  a^x = 2 →
  a^y = 3 →
  a^(x + y) = 6 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_exponent_multiplication_l2037_203795


namespace NUMINAMATH_GPT_true_propositions_for_quadratic_equations_l2037_203749

theorem true_propositions_for_quadratic_equations :
  (∀ (a b c : ℤ), a ≠ 0 → (∃ Δ : ℤ, Δ^2 = b^2 - 4 * a * c → ∃ x : ℚ, x^2 + (b / a) * x + (c / a) = 0)) ∧
  (∀ (a b c : ℤ), a ≠ 0 → (∃ x : ℚ, x^2 + (b / a) * x + (c / a) = 0 → ∃ Δ : ℤ, Δ^2 = b^2 - 4 * a * c)) ∧
  (¬ ∀ (a b c : ℝ), a ≠ 0 → (∃ x : ℚ, x^2 + (b / a) * x + (c / a) = 0)) ∧
  (∀ (a b c : ℤ), a ≠ 0 ∧ a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 → ¬∃ x : ℚ, x^2 + (b / a) * x + (c / a) = 0) :=
by sorry

end NUMINAMATH_GPT_true_propositions_for_quadratic_equations_l2037_203749


namespace NUMINAMATH_GPT_max_projection_area_of_tetrahedron_l2037_203790

/-- 
Two adjacent faces of a tetrahedron are isosceles right triangles with a hypotenuse of 2,
and they form a dihedral angle of 60 degrees. The tetrahedron rotates around the common edge
of these faces. The maximum area of the projection of the rotating tetrahedron onto 
the plane containing the given edge is 1.
-/
theorem max_projection_area_of_tetrahedron (S hypotenuse dihedral max_proj_area : ℝ)
  (is_isosceles_right_triangle : ∀ (a b : ℝ), a^2 + b^2 = hypotenuse^2)
  (hypotenuse_len : hypotenuse = 2)
  (dihedral_angle : dihedral = 60) :
  max_proj_area = 1 :=
  sorry

end NUMINAMATH_GPT_max_projection_area_of_tetrahedron_l2037_203790


namespace NUMINAMATH_GPT_cost_of_door_tickets_l2037_203762

theorem cost_of_door_tickets (x : ℕ) 
  (advanced_purchase_cost : ℕ)
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (advanced_tickets_sold : ℕ)
  (total_revenue_advanced : ℕ := advanced_tickets_sold * advanced_purchase_cost)
  (door_tickets_sold : ℕ := total_tickets - advanced_tickets_sold) : 
  advanced_purchase_cost = 8 ∧
  total_tickets = 140 ∧
  total_revenue = 1720 ∧
  advanced_tickets_sold = 100 →
  door_tickets_sold * x + total_revenue_advanced = total_revenue →
  x = 23 := 
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_cost_of_door_tickets_l2037_203762


namespace NUMINAMATH_GPT_vector_subtraction_l2037_203726

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 4)
def b : ℝ × ℝ := (-1, 1)

-- Define the expression to be proven
def expression : ℝ × ℝ := ((2 * a.1 - b.1), (2 * a.2 - b.2))

-- The theorem statement
theorem vector_subtraction : expression = (5, 7) :=
by
  -- The proof will be provided here
  sorry

end NUMINAMATH_GPT_vector_subtraction_l2037_203726


namespace NUMINAMATH_GPT_last_two_digits_2007_pow_20077_l2037_203739

theorem last_two_digits_2007_pow_20077 : (2007 ^ 20077) % 100 = 7 := 
by sorry

end NUMINAMATH_GPT_last_two_digits_2007_pow_20077_l2037_203739


namespace NUMINAMATH_GPT_Johnson_Smith_tied_end_May_l2037_203771

def home_runs_Johnson : List ℕ := [2, 12, 15, 8, 14, 11, 9, 16]
def home_runs_Smith : List ℕ := [5, 9, 10, 12, 15, 12, 10, 17]

def total_without_June (runs: List ℕ) : Nat := List.sum (runs.take 5 ++ runs.drop 5)
def estimated_June (total: Nat) : Nat := total / 8

theorem Johnson_Smith_tied_end_May :
  let total_Johnson := total_without_June home_runs_Johnson;
  let total_Smith := total_without_June home_runs_Smith;
  let estimated_June_Johnson := estimated_June total_Johnson;
  let estimated_June_Smith := estimated_June total_Smith;
  let total_with_June_Johnson := total_Johnson + estimated_June_Johnson;
  let total_with_June_Smith := total_Smith + estimated_June_Smith;
  (List.sum (home_runs_Johnson.take 5) = List.sum (home_runs_Smith.take 5)) :=
by
  sorry

end NUMINAMATH_GPT_Johnson_Smith_tied_end_May_l2037_203771


namespace NUMINAMATH_GPT_ratio_B_to_C_l2037_203730

theorem ratio_B_to_C (A_share B_share C_share : ℝ) 
  (total : A_share + B_share + C_share = 510) 
  (A_share_val : A_share = 360) 
  (B_share_val : B_share = 90)
  (C_share_val : C_share = 60)
  (A_cond : A_share = (2 / 3) * B_share) 
  : B_share / C_share = 3 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_B_to_C_l2037_203730


namespace NUMINAMATH_GPT_intersection_M_N_l2037_203786

noncomputable def M : Set ℝ := { x | x^2 = x }
noncomputable def N : Set ℝ := { x | Real.log x ≤ 0 }

theorem intersection_M_N :
  M ∩ N = {1} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l2037_203786


namespace NUMINAMATH_GPT_b_greater_than_neg3_l2037_203756

def a_n (n : ℕ) (b : ℝ) : ℝ := n^2 + b * n

theorem b_greater_than_neg3 (b : ℝ) :
  (∀ (n : ℕ), 0 < n → a_n (n + 1) b > a_n n b) → b > -3 :=
by
  sorry

end NUMINAMATH_GPT_b_greater_than_neg3_l2037_203756


namespace NUMINAMATH_GPT_plane_second_trace_line_solutions_l2037_203719

noncomputable def num_solutions_second_trace_line
  (first_trace_line : Line)
  (angle_with_projection_plane : ℝ)
  (intersection_outside_paper : Prop) : ℕ :=
2

theorem plane_second_trace_line_solutions
  (first_trace_line : Line)
  (angle_with_projection_plane : ℝ)
  (intersection_outside_paper : Prop) :
  num_solutions_second_trace_line first_trace_line angle_with_projection_plane intersection_outside_paper = 2 := by
sorry

end NUMINAMATH_GPT_plane_second_trace_line_solutions_l2037_203719


namespace NUMINAMATH_GPT_number_of_solutions_l2037_203701

-- Define the main theorem with the correct conditions
theorem number_of_solutions : 
  (∃ (x₁ x₂ x₃ x₄ x₅ : ℕ), 
     x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧ x₅ > 0 ∧ x₁ + x₂ + x₃ + x₄ + x₅ = 10) 
  → 
  (∃ t : ℕ, t = 70) :=
by 
  sorry

end NUMINAMATH_GPT_number_of_solutions_l2037_203701


namespace NUMINAMATH_GPT_merge_coins_n_ge_3_merge_coins_n_eq_2_l2037_203700

-- For Part 1
theorem merge_coins_n_ge_3 (n : ℕ) (hn : n ≥ 3) :
  ∃ (m : ℕ), m = 1 ∨ m = 2 :=
sorry

-- For Part 2
theorem merge_coins_n_eq_2 (r s : ℕ) :
  ∃ (k : ℕ), r + s = 2^k * Nat.gcd r s :=
sorry

end NUMINAMATH_GPT_merge_coins_n_ge_3_merge_coins_n_eq_2_l2037_203700


namespace NUMINAMATH_GPT_distance_AD_35_l2037_203734

-- Definitions based on conditions
variables (A B C D : Point)
variable (distance : Point → Point → ℝ)
variable (angle : Point → Point → Point → ℝ)
variable (dueEast : Point → Point → Prop)
variable (northOf : Point → Point → Prop)

-- Conditions
def conditions : Prop :=
  dueEast A B ∧
  angle A B C = 90 ∧
  distance A C = 15 * Real.sqrt 3 ∧
  angle B A C = 30 ∧
  northOf D C ∧
  distance C D = 10

-- The question: Proving the distance between points A and D
theorem distance_AD_35 (h : conditions A B C D distance angle dueEast northOf) :
  distance A D = 35 :=
sorry

end NUMINAMATH_GPT_distance_AD_35_l2037_203734


namespace NUMINAMATH_GPT_distinct_prime_sum_product_l2037_203715

open Nat

-- Definitions
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- The problem statement
theorem distinct_prime_sum_product (a b c : ℕ) (h1 : is_prime a) (h2 : is_prime b) 
    (h3 : is_prime c) (h4 : a ≠ 1) (h5 : b ≠ 1) (h6 : c ≠ 1) 
    (h7 : a ≠ b) (h8 : b ≠ c) (h9 : a ≠ c) : 

    1994 + a + b + c = a * b * c :=
sorry

end NUMINAMATH_GPT_distinct_prime_sum_product_l2037_203715


namespace NUMINAMATH_GPT_find_nabla_l2037_203767

theorem find_nabla (nabla : ℤ) (h : 4 * (-3) = nabla + 3) : nabla = -15 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_nabla_l2037_203767
