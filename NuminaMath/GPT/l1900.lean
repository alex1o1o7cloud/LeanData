import Mathlib

namespace NUMINAMATH_GPT_square_difference_l1900_190009

theorem square_difference (x : ℤ) (h : x^2 = 9801) : (x + 1) * (x - 1) = 9800 :=
by {
  sorry
}

end NUMINAMATH_GPT_square_difference_l1900_190009


namespace NUMINAMATH_GPT_correct_inequality_relation_l1900_190057

theorem correct_inequality_relation :
  ¬(∀ (a b c : ℝ), a > b ↔ a * (c^2) > b * (c^2)) ∧
  ¬(∀ (a b : ℝ), a > b → (1/a) < (1/b)) ∧
  ¬(∀ (a b c d : ℝ), a > b ∧ b > 0 ∧ c > d → a/d > b/c) ∧
  (∀ (a b c : ℝ), a > b ∧ b > 1 ∧ c < 0 → a^c < b^c) := sorry

end NUMINAMATH_GPT_correct_inequality_relation_l1900_190057


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1900_190051

theorem arithmetic_sequence_common_difference 
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : S 3 = a 1 + a 2 + a 3)
  (h2 : S 2 = a 1 + a 2)
  (h3 : a 2 = a 1 + d)
  (h4 : a 3 = a 1 + 2 * d)
  (h5 : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1900_190051


namespace NUMINAMATH_GPT_ratio_shorter_to_longer_l1900_190066

theorem ratio_shorter_to_longer (x y : ℝ) (h1 : x < y) (h2 : x + y - Real.sqrt (x^2 + y^2) = y / 3) : x / y = 5 / 12 :=
sorry

end NUMINAMATH_GPT_ratio_shorter_to_longer_l1900_190066


namespace NUMINAMATH_GPT_hexagon_coloring_unique_l1900_190001

-- Define the coloring of the hexagon using enumeration
inductive Color
  | green
  | blue
  | orange

-- Assume we have a function that represents the coloring of the hexagons
-- with the constraints given in the problem
def is_valid_coloring (coloring : ℕ → ℕ → Color) : Prop :=
  ∀ x y : ℕ, -- For all hexagons
  (coloring x y = Color.green ∧ x = 0 ∧ y = 0) ∨ -- The labeled hexagon G is green
  (coloring x y ≠ coloring (x + 1) y ∧ -- No two hexagons with a common side have the same color
   coloring x y ≠ coloring (x - 1) y ∧ 
   coloring x y ≠ coloring x (y + 1) ∧
   coloring x y ≠ coloring x (y - 1))

-- The problem is to prove there are exactly 2 valid colorings of the hexagon grid
theorem hexagon_coloring_unique :
  ∃ (count : ℕ), count = 2 ∧
  ∀ coloring : (ℕ → ℕ → Color), is_valid_coloring coloring → count = 2 :=
by
  sorry

end NUMINAMATH_GPT_hexagon_coloring_unique_l1900_190001


namespace NUMINAMATH_GPT_unique_function_eq_id_l1900_190027

theorem unique_function_eq_id (f : ℝ → ℝ) :
  (∀ x : ℝ, x ≠ 0 → f x = x^2 * f (1 / x)) →
  (∀ x y : ℝ, f (x + y) = f x + f y) →
  (f 1 = 1) →
  (∀ x : ℝ, f x = x) :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_unique_function_eq_id_l1900_190027


namespace NUMINAMATH_GPT_cakes_served_for_lunch_l1900_190024

theorem cakes_served_for_lunch (total_cakes: ℕ) (dinner_cakes: ℕ) (lunch_cakes: ℕ) 
  (h1: total_cakes = 15) 
  (h2: dinner_cakes = 9) 
  (h3: total_cakes = lunch_cakes + dinner_cakes) : 
  lunch_cakes = 6 := 
by 
  sorry

end NUMINAMATH_GPT_cakes_served_for_lunch_l1900_190024


namespace NUMINAMATH_GPT_therapists_next_meeting_day_l1900_190047

theorem therapists_next_meeting_day : Nat.lcm (Nat.lcm 5 2) (Nat.lcm 9 3) = 90 := by
  -- Given that Alex works every 5 days,
  -- Brice works every 2 days,
  -- Emma works every 9 days,
  -- and Fiona works every 3 days, we need to show that the LCM of these numbers is 90.
  sorry

end NUMINAMATH_GPT_therapists_next_meeting_day_l1900_190047


namespace NUMINAMATH_GPT_mul_72519_9999_eq_725117481_l1900_190035

theorem mul_72519_9999_eq_725117481 : 72519 * 9999 = 725117481 := by
  sorry

end NUMINAMATH_GPT_mul_72519_9999_eq_725117481_l1900_190035


namespace NUMINAMATH_GPT_mod_equiv_1043_36_mod_equiv_1_10_l1900_190087

open Int

-- Define the integers involved
def a : ℤ := -1043
def m1 : ℕ := 36
def m2 : ℕ := 10

-- Theorems to prove modulo equivalence
theorem mod_equiv_1043_36 : a % m1 = 1 := by
  sorry

theorem mod_equiv_1_10 : 1 % m2 = 1 := by
  sorry

end NUMINAMATH_GPT_mod_equiv_1043_36_mod_equiv_1_10_l1900_190087


namespace NUMINAMATH_GPT_coefficient_x4_in_expansion_sum_l1900_190039

theorem coefficient_x4_in_expansion_sum :
  (Nat.choose 5 4 + Nat.choose 6 4 + Nat.choose 7 4 = 55) :=
by
  sorry

end NUMINAMATH_GPT_coefficient_x4_in_expansion_sum_l1900_190039


namespace NUMINAMATH_GPT_topological_sort_possible_l1900_190034
-- Import the necessary library

-- Definition of simple, directed, and acyclic graph (DAG)
structure SimpleDirectedAcyclicGraph (V : Type*) :=
  (E : V → V → Prop)
  (acyclic : ∀ v : V, ¬(E v v)) -- no loops
  (simple : ∀ (u v : V), (E u v) → ¬(E v u)) -- no bidirectional edges
  (directional : ∀ (u v w : V), E u v → E v w → E u w) -- directional transitivity

-- Existence of topological sort definition
def topological_sort_exists {V : Type*} (G : SimpleDirectedAcyclicGraph V) : Prop :=
  ∃ (numbering : V → ℕ), ∀ (u v : V), (G.E u v) → (numbering u > numbering v)

-- Theorem statement
theorem topological_sort_possible (V : Type*) (G : SimpleDirectedAcyclicGraph V) : topological_sort_exists G :=
  sorry

end NUMINAMATH_GPT_topological_sort_possible_l1900_190034


namespace NUMINAMATH_GPT_rebecca_income_l1900_190013

variable (R : ℝ) -- Rebecca's current yearly income (denoted as R)
variable (increase : ℝ := 7000) -- The increase in Rebecca's income
variable (jimmy_income : ℝ := 18000) -- Jimmy's yearly income
variable (combined_income : ℝ := (R + increase) + jimmy_income) -- Combined income after increase
variable (new_income_ratio : ℝ := 0.55) -- Proportion of total income that is Rebecca's new income

theorem rebecca_income : (R + increase) = new_income_ratio * combined_income → R = 15000 :=
by
  sorry

end NUMINAMATH_GPT_rebecca_income_l1900_190013


namespace NUMINAMATH_GPT_minimum_value_is_1_l1900_190043

def minimum_value_expression (x y : ℝ) : ℝ :=
  x^2 + y^2 - 8*x + 6*y + 26

theorem minimum_value_is_1 (x y : ℝ) (h : x ≥ 4) : 
  minimum_value_expression x y ≥ 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_minimum_value_is_1_l1900_190043


namespace NUMINAMATH_GPT_eq_condition_implies_inequality_l1900_190041

theorem eq_condition_implies_inequality (a : ℝ) (h_neg_root : 2 * a - 4 < 0) : (a - 3) * (a - 4) > 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_eq_condition_implies_inequality_l1900_190041


namespace NUMINAMATH_GPT_puppies_adopted_per_day_l1900_190063

theorem puppies_adopted_per_day 
    (initial_puppies : ℕ) 
    (additional_puppies : ℕ) 
    (total_days : ℕ) 
    (total_puppies : ℕ)
    (H1 : initial_puppies = 5) 
    (H2 : additional_puppies = 35) 
    (H3 : total_days = 5) 
    (H4 : total_puppies = initial_puppies + additional_puppies) : 
    total_puppies / total_days = 8 := by
  sorry

end NUMINAMATH_GPT_puppies_adopted_per_day_l1900_190063


namespace NUMINAMATH_GPT_temperature_on_tuesday_l1900_190078

variable (T W Th F : ℕ)

-- Conditions
def cond1 : Prop := (T + W + Th) / 3 = 32
def cond2 : Prop := (W + Th + F) / 3 = 34
def cond3 : Prop := F = 44

-- Theorem statement
theorem temperature_on_tuesday : cond1 T W Th → cond2 W Th F → cond3 F → T = 38 :=
by
  sorry

end NUMINAMATH_GPT_temperature_on_tuesday_l1900_190078


namespace NUMINAMATH_GPT_evaluate_expression_l1900_190089

theorem evaluate_expression (x y : ℕ) (hx : x = 4) (hy : y = 5) :
  (1 / (y : ℚ) / (1 / (x : ℚ)) + 2) = 14 / 5 :=
by
  rw [hx, hy]
  simp
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1900_190089


namespace NUMINAMATH_GPT_part_one_part_two_l1900_190084

def universal_set : Set ℝ := Set.univ
def A : Set ℝ := { x | 1 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }
def C (a : ℝ) : Set ℝ := { x | x < a }

noncomputable def C_R_A : Set ℝ := { x | x < 1 ∨ x ≥ 7 }
noncomputable def C_R_A_union_B : Set ℝ := C_R_A ∪ B

theorem part_one : C_R_A_union_B = { x | x < 1 ∨ x > 2 } :=
sorry

theorem part_two (a : ℝ) (h : A ⊆ C a) : a ≥ 7 :=
sorry

end NUMINAMATH_GPT_part_one_part_two_l1900_190084


namespace NUMINAMATH_GPT_blue_pill_cost_l1900_190052

theorem blue_pill_cost :
  ∃ (y : ℝ), (∀ (d : ℝ), d = 45) ∧
  (∀ (b : ℝ) (r : ℝ), b = y ∧ r = y - 2) ∧
  ((21 : ℝ) * 45 = 945) ∧
  (b + r = 45) ∧
  y = 23.5 := 
by
  sorry

end NUMINAMATH_GPT_blue_pill_cost_l1900_190052


namespace NUMINAMATH_GPT_abs_diff_ps_pds_eq_31_100_l1900_190079

-- Defining the conditions
def num_red : ℕ := 500
def num_black : ℕ := 700
def num_blue : ℕ := 800
def total_marbles : ℕ := num_red + num_black + num_blue
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Calculating P_s and P_d
def ways_same_color : ℕ := choose num_red 2 + choose num_black 2 + choose num_blue 2
def total_ways : ℕ := choose total_marbles 2
def P_s : ℚ := ways_same_color / total_ways

def ways_different_color : ℕ := num_red * num_black + num_red * num_blue + num_black * num_blue
def P_d : ℚ := ways_different_color / total_ways

-- Proving the statement
theorem abs_diff_ps_pds_eq_31_100 : |P_s - P_d| = (31 : ℚ) / 100 := by
  sorry

end NUMINAMATH_GPT_abs_diff_ps_pds_eq_31_100_l1900_190079


namespace NUMINAMATH_GPT_hcf_lcm_fraction_l1900_190059

theorem hcf_lcm_fraction (m n : ℕ) (HCF : Nat.gcd m n = 6) (LCM : Nat.lcm m n = 210) (sum_mn : m + n = 72) : 
  (1 / m : ℚ) + (1 / n : ℚ) = 2 / 35 :=
by
  sorry

end NUMINAMATH_GPT_hcf_lcm_fraction_l1900_190059


namespace NUMINAMATH_GPT_volume_and_area_of_pyramid_l1900_190011

-- Define the base of the pyramid.
def rect (EF FG : ℕ) : Prop := EF = 10 ∧ FG = 6

-- Define the perpendicular relationships and height of the pyramid.
def pyramid (EF FG PE : ℕ) : Prop := 
  rect EF FG ∧
  PE = 10 ∧ 
  (PE > 0) -- Given conditions include perpendicular properties, implying height is positive.

-- Problem translation: Prove the volume and area calculations.
theorem volume_and_area_of_pyramid (EF FG PE : ℕ) 
  (h1 : rect EF FG) 
  (h2 : PE = 10) : 
  (1 / 3 * EF * FG * PE = 200 ∧ 1 / 2 * EF * FG = 30) := 
by
  sorry

end NUMINAMATH_GPT_volume_and_area_of_pyramid_l1900_190011


namespace NUMINAMATH_GPT_gcd_3_1200_1_3_1210_1_l1900_190042

theorem gcd_3_1200_1_3_1210_1 : 
  Int.gcd (3^1200 - 1) (3^1210 - 1) = 59048 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_3_1200_1_3_1210_1_l1900_190042


namespace NUMINAMATH_GPT_parallelogram_not_symmetrical_l1900_190074

def is_symmetrical (shape : String) : Prop :=
  shape = "Circle" ∨ shape = "Rectangle" ∨ shape = "Isosceles Trapezoid"

theorem parallelogram_not_symmetrical : ¬ is_symmetrical "Parallelogram" :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_not_symmetrical_l1900_190074


namespace NUMINAMATH_GPT_stickers_initial_count_l1900_190002

variable (initial : ℕ) (lost : ℕ)

theorem stickers_initial_count (lost_stickers : lost = 6) (remaining_stickers : initial - lost = 87) : initial = 93 :=
by {
  sorry
}

end NUMINAMATH_GPT_stickers_initial_count_l1900_190002


namespace NUMINAMATH_GPT_sum_lent_is_1500_l1900_190049

/--
A person lent a certain sum of money at 4% per annum at simple interest.
In 4 years, the interest amounted to Rs. 1260 less than the sum lent.
Prove that the sum lent was Rs. 1500.
-/
theorem sum_lent_is_1500
  (P : ℝ) (r : ℝ) (t : ℝ) (I : ℝ)
  (h1 : r = 4) (h2 : t = 4)
  (h3 : I = P - 1260)
  (h4 : I = P * r * t / 100):
  P = 1500 :=
by
  sorry

end NUMINAMATH_GPT_sum_lent_is_1500_l1900_190049


namespace NUMINAMATH_GPT_halfway_fraction_l1900_190007

theorem halfway_fraction (a b : ℚ) (h₁ : a = 3/4) (h₂ : b = 5/6) : (a + b) / 2 = 19 / 24 :=
by
  sorry

end NUMINAMATH_GPT_halfway_fraction_l1900_190007


namespace NUMINAMATH_GPT_tan_sum_simplification_l1900_190094

open Real

theorem tan_sum_simplification :
  tan 70 + tan 50 - sqrt 3 * tan 70 * tan 50 = -sqrt 3 := by
  sorry

end NUMINAMATH_GPT_tan_sum_simplification_l1900_190094


namespace NUMINAMATH_GPT_third_number_is_32_l1900_190081

theorem third_number_is_32 (A B C : ℕ) 
  (hA : A = 24) (hB : B = 36) 
  (hHCF : Nat.gcd (Nat.gcd A B) C = 32) 
  (hLCM : Nat.lcm (Nat.lcm A B) C = 1248) : 
  C = 32 := 
sorry

end NUMINAMATH_GPT_third_number_is_32_l1900_190081


namespace NUMINAMATH_GPT_reformulate_and_find_product_l1900_190003

theorem reformulate_and_find_product (a b x y : ℝ)
  (h : a^8 * x * y - a^7 * y - a^6 * x = a^5 * (b^5 - 2)) :
  ∃ m' n' p' : ℤ, (a^m' * x - a^n') * (a^p' * y - a^3) = a^5 * b^5 ∧ m' * n' * p' = 48 :=
by
  sorry

end NUMINAMATH_GPT_reformulate_and_find_product_l1900_190003


namespace NUMINAMATH_GPT_unique_solution_l1900_190055

theorem unique_solution (x y z : ℝ) 
  (h : x^2 + 2*x + y^2 + 4*y + z^2 + 6*z = -14) : 
  x = -1 ∧ y = -2 ∧ z = -3 :=
by
  -- entering main proof section
  sorry

end NUMINAMATH_GPT_unique_solution_l1900_190055


namespace NUMINAMATH_GPT_quadratic_inequality_ab_l1900_190050

/-- Given a quadratic inequality ax^2 + bx + 1 > 0 with solution set -1 < x < 1/3,
    prove that ab = 6. -/
theorem quadratic_inequality_ab (a b : ℝ) (h1 : ∀ x, -1 < x ∧ x < 1 / 3 → a * x ^ 2 + b * x + 1 > 0):
  a * b = 6 := 
sorry

end NUMINAMATH_GPT_quadratic_inequality_ab_l1900_190050


namespace NUMINAMATH_GPT_number_of_teams_l1900_190030

theorem number_of_teams (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 :=
sorry

end NUMINAMATH_GPT_number_of_teams_l1900_190030


namespace NUMINAMATH_GPT_intersecting_lines_at_point_find_b_plus_m_l1900_190044

theorem intersecting_lines_at_point_find_b_plus_m :
  ∀ (m b : ℝ),
  (12 = m * 4 + 2) →
  (12 = -2 * 4 + b) →
  (b + m = 22.5) :=
by
  intros m b h1 h2
  sorry

end NUMINAMATH_GPT_intersecting_lines_at_point_find_b_plus_m_l1900_190044


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l1900_190096

theorem geometric_sequence_common_ratio (a₁ : ℚ) (q : ℚ) 
  (S : ℕ → ℚ) (hS : ∀ n, S n = a₁ * (1 - q^n) / (1 - q)) 
  (h : 8 * S 6 = 7 * S 3) : 
  q = -1/2 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l1900_190096


namespace NUMINAMATH_GPT_sin_add_pi_over_2_l1900_190028

theorem sin_add_pi_over_2 (θ : ℝ) (h : Real.cos θ = -3 / 5) : Real.sin (θ + π / 2) = -3 / 5 :=
sorry

end NUMINAMATH_GPT_sin_add_pi_over_2_l1900_190028


namespace NUMINAMATH_GPT_total_fish_bought_l1900_190060

theorem total_fish_bought (gold_fish blue_fish : Nat) (h1 : gold_fish = 15) (h2 : blue_fish = 7) : gold_fish + blue_fish = 22 := by
  sorry

end NUMINAMATH_GPT_total_fish_bought_l1900_190060


namespace NUMINAMATH_GPT_gcd_bn_bn1_l1900_190021

def b (n : ℕ) : ℤ := (7^n - 1) / 6
def e (n : ℕ) : ℤ := Int.gcd (b n) (b (n + 1))

theorem gcd_bn_bn1 (n : ℕ) : e n = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_bn_bn1_l1900_190021


namespace NUMINAMATH_GPT_percent_voters_for_A_l1900_190085

-- Definitions from conditions
def total_voters : ℕ := 100
def percent_democrats : ℝ := 0.70
def percent_republicans : ℝ := 0.30
def percent_dems_for_A : ℝ := 0.80
def percent_reps_for_A : ℝ := 0.30

-- Calculations based on definitions
def num_democrats := total_voters * percent_democrats
def num_republicans := total_voters * percent_republicans
def dems_for_A := num_democrats * percent_dems_for_A
def reps_for_A := num_republicans * percent_reps_for_A
def total_for_A := dems_for_A + reps_for_A

-- Proof problem statement
theorem percent_voters_for_A : (total_for_A / total_voters) * 100 = 65 :=
by
  sorry

end NUMINAMATH_GPT_percent_voters_for_A_l1900_190085


namespace NUMINAMATH_GPT_average_age_is_27_l1900_190076

variables (a b c : ℕ)

def average_age_of_a_and_c (a c : ℕ) := (a + c) / 2

def age_of_b := 23

def average_age_of_a_b_and_c (a b c : ℕ) := (a + b + c) / 3

theorem average_age_is_27 (h1 : average_age_of_a_and_c a c = 29) (h2 : b = age_of_b) :
  average_age_of_a_b_and_c a b c = 27 := by
  sorry

end NUMINAMATH_GPT_average_age_is_27_l1900_190076


namespace NUMINAMATH_GPT_leo_peeled_potatoes_l1900_190088

noncomputable def lucy_rate : ℝ := 4
noncomputable def leo_rate : ℝ := 6
noncomputable def total_potatoes : ℝ := 60
noncomputable def lucy_time_alone : ℝ := 6
noncomputable def total_potatoes_left : ℝ := total_potatoes - lucy_rate * lucy_time_alone
noncomputable def combined_rate : ℝ := lucy_rate + leo_rate
noncomputable def combined_time : ℝ := total_potatoes_left / combined_rate
noncomputable def leo_potatoes : ℝ := combined_time * leo_rate

theorem leo_peeled_potatoes :
  leo_potatoes = 22 :=
by
  sorry

end NUMINAMATH_GPT_leo_peeled_potatoes_l1900_190088


namespace NUMINAMATH_GPT_charlie_steps_proof_l1900_190032

-- Define the conditions
def Steps_Charlie_3km : ℕ := 5350
def Laps : ℚ := 2.5

-- Define the total steps Charlie can make in 2.5 laps
def Steps_Charlie_total : ℕ := 13375

-- The statement to prove
theorem charlie_steps_proof : Laps * Steps_Charlie_3km = Steps_Charlie_total :=
by
  sorry

end NUMINAMATH_GPT_charlie_steps_proof_l1900_190032


namespace NUMINAMATH_GPT_child_ticket_cost_l1900_190020

/-- Defining the conditions and proving the cost of a child's ticket --/
theorem child_ticket_cost:
  (∀ c: ℕ, 
      -- Revenue from Monday
      (7 * c + 5 * 4) + 
      -- Revenue from Tuesday
      (4 * c + 2 * 4) = 
      -- Total revenue for both days
      61 
    ) → 
    -- Proving c
    (c = 3) :=
by
  sorry

end NUMINAMATH_GPT_child_ticket_cost_l1900_190020


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1900_190090

theorem solution_set_of_inequality (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1) :
  {x : ℝ | (a - x) * (x - 1 / a) > 0} = {x : ℝ | a < x ∧ x < 1 / a} :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1900_190090


namespace NUMINAMATH_GPT_time_difference_correct_l1900_190099

-- Definitions based on conditions
def malcolm_speed : ℝ := 5 -- Malcolm's speed in minutes per mile
def joshua_speed : ℝ := 7 -- Joshua's speed in minutes per mile
def race_length : ℝ := 12 -- Length of the race in miles

-- Calculate times based on speeds and race length
def malcolm_time : ℝ := malcolm_speed * race_length
def joshua_time : ℝ := joshua_speed * race_length

-- The statement that the difference in finish times is 24 minutes
theorem time_difference_correct : joshua_time - malcolm_time = 24 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_time_difference_correct_l1900_190099


namespace NUMINAMATH_GPT_right_triangle_area_and_hypotenuse_l1900_190097

-- Definitions based on given conditions
def a : ℕ := 24
def b : ℕ := 2 * a + 10

-- Statements based on the questions and correct answers
theorem right_triangle_area_and_hypotenuse :
  (1 / 2 : ℝ) * (a : ℝ) * (b : ℝ) = 696 ∧ (Real.sqrt ((a : ℝ)^2 + (b : ℝ)^2) = Real.sqrt 3940) := by
  sorry

end NUMINAMATH_GPT_right_triangle_area_and_hypotenuse_l1900_190097


namespace NUMINAMATH_GPT_negation_of_universal_l1900_190031

theorem negation_of_universal :
  ¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 > 0 :=
by
  sorry    -- Proof is not required, just the statement.

end NUMINAMATH_GPT_negation_of_universal_l1900_190031


namespace NUMINAMATH_GPT_ice_cream_stall_difference_l1900_190019

theorem ice_cream_stall_difference (d : ℕ) 
  (h1 : ∃ d, 10 + (10 + d) + (10 + 2*d) + (10 + 3*d) + (10 + 4*d) = 90) : 
  d = 4 :=
by
  sorry

end NUMINAMATH_GPT_ice_cream_stall_difference_l1900_190019


namespace NUMINAMATH_GPT_reeya_fifth_score_l1900_190062

theorem reeya_fifth_score
  (s1 s2 s3 s4 avg: ℝ)
  (h1: s1 = 65)
  (h2: s2 = 67)
  (h3: s3 = 76)
  (h4: s4 = 82)
  (h_avg: avg = 75) :
  ∃ s5, s1 + s2 + s3 + s4 + s5 = 5 * avg ∧ s5 = 85 :=
by
  use 85
  sorry

end NUMINAMATH_GPT_reeya_fifth_score_l1900_190062


namespace NUMINAMATH_GPT_solve_for_y_l1900_190061

theorem solve_for_y (x y : ℝ) (h1 : x * y = 25) (h2 : x / y = 36) (h3 : x > 0) (h4 : y > 0) : y = 5 / 6 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l1900_190061


namespace NUMINAMATH_GPT_platform_length_l1900_190017

variable (L : ℝ) -- The length of the platform
variable (train_length : ℝ := 300) -- The length of the train
variable (time_pole : ℝ := 26) -- Time to cross the signal pole
variable (time_platform : ℝ := 39) -- Time to cross the platform

theorem platform_length :
  (train_length / time_pole) = (train_length + L) / time_platform → L = 150 := sorry

end NUMINAMATH_GPT_platform_length_l1900_190017


namespace NUMINAMATH_GPT_simplify_and_evaluate_expr_find_ab_l1900_190038

theorem simplify_and_evaluate_expr (x y : ℝ) (hx : x = 0.5) (hy : y = -1) :
  (x - 5 * y) * (-x - 5 * y) - (-x + 5 * y)^2 = -5.5 :=
by
  rw [hx, hy]
  sorry

theorem find_ab (a b : ℝ) (h : a^2 - 2 * a + b^2 + 4 * b + 5 = 0) :
  (a + b) ^ 2013 = -1 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expr_find_ab_l1900_190038


namespace NUMINAMATH_GPT_vertical_asymptotes_count_l1900_190008

theorem vertical_asymptotes_count : 
  let f (x : ℝ) := (x - 2) / (x^2 + 4*x - 5) 
  ∃! c : ℕ, c = 2 :=
by
  sorry

end NUMINAMATH_GPT_vertical_asymptotes_count_l1900_190008


namespace NUMINAMATH_GPT_foci_distance_of_hyperbola_l1900_190067

theorem foci_distance_of_hyperbola :
  ∀ (x y : ℝ), (x^2 / 32 - y^2 / 8 = 1) → 2 * (Real.sqrt (32 + 8)) = 4 * Real.sqrt 10 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_foci_distance_of_hyperbola_l1900_190067


namespace NUMINAMATH_GPT_percentage_conveyance_l1900_190058

def percentage_on_food := 40 / 100
def percentage_on_rent := 20 / 100
def percentage_on_entertainment := 10 / 100
def salary := 12500
def savings := 2500

def total_percentage_spent := percentage_on_food + percentage_on_rent + percentage_on_entertainment
def total_spent := salary - savings
def amount_spent_on_conveyance := total_spent - (salary * total_percentage_spent)
def percentage_spent_on_conveyance := (amount_spent_on_conveyance / salary) * 100

theorem percentage_conveyance : percentage_spent_on_conveyance = 10 :=
by sorry

end NUMINAMATH_GPT_percentage_conveyance_l1900_190058


namespace NUMINAMATH_GPT_interior_angle_regular_octagon_l1900_190086

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) :
  let sum_of_interior_angles := 180 * (n - 2)
  let each_angle := sum_of_interior_angles / n
  each_angle = 135 :=
by
  sorry

end NUMINAMATH_GPT_interior_angle_regular_octagon_l1900_190086


namespace NUMINAMATH_GPT_intersection_of_sets_l1900_190005

def A : Set ℝ := {0, 1, 2, 3}
def B : Set ℝ := {1, 3, 4}
def C : Set ℝ := {x | x > 2 ∨ x < 1}

theorem intersection_of_sets :
  (A ∪ B) ∩ C = {0, 3, 4} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l1900_190005


namespace NUMINAMATH_GPT_find_pq_cube_l1900_190082

theorem find_pq_cube (p q : ℝ) (h1 : p + q = 5) (h2 : p * q = 3) : (p + q) ^ 3 = 125 := 
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_find_pq_cube_l1900_190082


namespace NUMINAMATH_GPT_S9_is_45_l1900_190070

-- Define the required sequence and conditions
variable {a : ℕ → ℝ} -- a function that gives us the arithmetic sequence
variable {S : ℕ → ℝ} -- a function that gives us the sum of the first n terms of the sequence

-- Define the condition that a_2 + a_8 = 10
axiom a2_a8_condition : a 2 + a 8 = 10

-- Define the arithmetic property of the sequence
axiom arithmetic_property (n m : ℕ) : a (n + m) = a n + a m

-- Define the sum formula for the first n terms of an arithmetic sequence
axiom sum_formula (n : ℕ) : S n = (n / 2) * (a 1 + a n)

-- The main theorem to prove
theorem S9_is_45 : S 9 = 45 :=
by
  -- Here would go the proof, but it is omitted
  sorry

end NUMINAMATH_GPT_S9_is_45_l1900_190070


namespace NUMINAMATH_GPT_peanuts_total_correct_l1900_190029

def initial_peanuts : ℕ := 4
def added_peanuts : ℕ := 6
def total_peanuts : ℕ := initial_peanuts + added_peanuts

theorem peanuts_total_correct : total_peanuts = 10 := by
  sorry

end NUMINAMATH_GPT_peanuts_total_correct_l1900_190029


namespace NUMINAMATH_GPT_heavy_rain_duration_l1900_190026

-- Define the conditions as variables and constants
def initial_volume := 100 -- Initial volume in liters
def final_volume := 280   -- Final volume in liters
def flow_rate := 2        -- Flow rate in liters per minute

-- Define the duration query as a theorem to be proved
theorem heavy_rain_duration : 
  (final_volume - initial_volume) / flow_rate = 90 := 
by
  sorry

end NUMINAMATH_GPT_heavy_rain_duration_l1900_190026


namespace NUMINAMATH_GPT_Liliane_more_soda_than_Alice_l1900_190065

variable (J : ℝ) -- Represents the amount of soda Jacqueline has

-- Conditions: Representing the amounts for Benjamin, Liliane, and Alice
def B := 1.75 * J
def L := 1.60 * J
def A := 1.30 * J

-- Question: Proving the relationship in percentage terms between the amounts Liliane and Alice have
theorem Liliane_more_soda_than_Alice :
  (L - A) / A * 100 = 23 := 
by sorry

end NUMINAMATH_GPT_Liliane_more_soda_than_Alice_l1900_190065


namespace NUMINAMATH_GPT_notebook_pen_cost_correct_l1900_190077

noncomputable def notebook_pen_cost : Prop :=
  ∃ (x y : ℝ), 
  3 * x + 2 * y = 7.40 ∧ 
  2 * x + 5 * y = 9.75 ∧ 
  (x + 3 * y) = 5.53

theorem notebook_pen_cost_correct : notebook_pen_cost :=
sorry

end NUMINAMATH_GPT_notebook_pen_cost_correct_l1900_190077


namespace NUMINAMATH_GPT_expenses_each_month_l1900_190095
noncomputable def total_expenses (worked_hours1 worked_hours2 worked_hours3 : ℕ) (rate1 rate2 rate3 : ℕ) (total_left : ℕ) : ℕ :=
  (worked_hours1 * rate1) + (worked_hours2 * rate2) + (worked_hours3 * rate3) - total_left

theorem expenses_each_month (hours1 : ℕ)
  (hours2 : ℕ)
  (hours3 : ℕ)
  (rate1 : ℕ)
  (rate2 : ℕ)
  (rate3 : ℕ)
  (left_over : ℕ) :
  hours1 = 20 → 
  rate1 = 10 →
  hours2 = 30 →
  rate2 = 20 →
  hours3 = 5 →
  rate3 = 40 →
  left_over = 500 → 
  total_expenses hours1 hours2 hours3 rate1 rate2 rate3 left_over = 500 := by
  intros h1 r1 h2 r2 h3 r3 l
  sorry

end NUMINAMATH_GPT_expenses_each_month_l1900_190095


namespace NUMINAMATH_GPT_probability_of_all_red_is_correct_l1900_190083

noncomputable def probability_of_all_red_drawn : ℚ :=
  let total_ways := (Nat.choose 10 5)   -- Total ways to choose 5 balls from 10
  let red_ways := (Nat.choose 5 5)      -- Ways to choose all 5 red balls
  red_ways / total_ways

theorem probability_of_all_red_is_correct :
  probability_of_all_red_drawn = 1 / 252 := by
  sorry

end NUMINAMATH_GPT_probability_of_all_red_is_correct_l1900_190083


namespace NUMINAMATH_GPT_evaluate_f_at_2_l1900_190033

def f (x : ℝ) : ℝ := x^2 - x

theorem evaluate_f_at_2 : f 2 = 2 := by
  sorry

end NUMINAMATH_GPT_evaluate_f_at_2_l1900_190033


namespace NUMINAMATH_GPT_quadratic_coefficients_l1900_190098

theorem quadratic_coefficients (b c : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + b * x + c = 0 ↔ (x = -1 ∨ x = 3)) → 
  b = -4 ∧ c = -6 :=
by
  intro h
  -- The proof would go here, but we'll skip it.
  sorry

end NUMINAMATH_GPT_quadratic_coefficients_l1900_190098


namespace NUMINAMATH_GPT_problem_solution_l1900_190048

theorem problem_solution (x : ℝ) : 
  (x < -2 ∨ (-2 < x ∧ x ≤ 0) ∨ (0 < x ∧ x < 2) ∨ (2 ≤ x ∧ x < (15 - Real.sqrt 257) / 8) ∨ ((15 + Real.sqrt 257) / 8 < x)) ↔ 
  (x^2 - 1) / (x + 2) ≥ 3 / (x - 2) + 7 / 4 := sorry

end NUMINAMATH_GPT_problem_solution_l1900_190048


namespace NUMINAMATH_GPT_minimum_toothpicks_removal_l1900_190010

theorem minimum_toothpicks_removal
    (num_toothpicks : ℕ) 
    (num_triangles : ℕ) 
    (h1 : num_toothpicks = 40) 
    (h2 : num_triangles > 35) :
    ∃ (min_removal : ℕ), min_removal = 15 
    := 
    sorry

end NUMINAMATH_GPT_minimum_toothpicks_removal_l1900_190010


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1900_190071

-- Definitions from conditions
def A : Set ℤ := {x | x - 1 ≥ 0}
def B : Set ℤ := {0, 1, 2}

-- Proof statement
theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1900_190071


namespace NUMINAMATH_GPT_base10_to_base4_156_eq_2130_l1900_190016

def base10ToBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec loop (n : ℕ) (acc : List ℕ) : List ℕ :=
      if n = 0 then acc
      else loop (n / 4) ((n % 4) :: acc)
    loop n []

theorem base10_to_base4_156_eq_2130 :
  base10ToBase4 156 = [2, 1, 3, 0] := sorry

end NUMINAMATH_GPT_base10_to_base4_156_eq_2130_l1900_190016


namespace NUMINAMATH_GPT_min_value_inequality_l1900_190004

theorem min_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 9) :
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ≥ 9 :=
sorry

end NUMINAMATH_GPT_min_value_inequality_l1900_190004


namespace NUMINAMATH_GPT_third_box_weight_l1900_190073

def box1_height := 1 -- inches
def box1_width := 2 -- inches
def box1_length := 4 -- inches
def box1_weight := 30 -- grams

def box2_height := 3 * box1_height
def box2_width := 2 * box1_width
def box2_length := box1_length

def box3_height := box2_height
def box3_width := box2_width / 2
def box3_length := box2_length

def volume (height : ℕ) (width : ℕ) (length : ℕ) : ℕ := height * width * length

def weight (box1_weight : ℕ) (box1_volume : ℕ) (box3_volume : ℕ) : ℕ := 
  box3_volume / box1_volume * box1_weight

theorem third_box_weight :
  weight box1_weight (volume box1_height box1_width box1_length) 
  (volume box3_height box3_width box3_length) = 90 :=
by
  sorry

end NUMINAMATH_GPT_third_box_weight_l1900_190073


namespace NUMINAMATH_GPT_number_of_buses_l1900_190040

theorem number_of_buses (total_people : ℕ) (bus_capacity : ℕ) (h1 : total_people = 1230) (h2 : bus_capacity = 48) : 
  Nat.ceil (total_people / bus_capacity : ℝ) = 26 := 
by 
  unfold Nat.ceil 
  sorry

end NUMINAMATH_GPT_number_of_buses_l1900_190040


namespace NUMINAMATH_GPT_distinct_complex_numbers_count_l1900_190025

theorem distinct_complex_numbers_count :
  let real_choices := 10
  let imag_choices := 9
  let distinct_complex_numbers := real_choices * imag_choices
  distinct_complex_numbers = 90 :=
by
  sorry

end NUMINAMATH_GPT_distinct_complex_numbers_count_l1900_190025


namespace NUMINAMATH_GPT_specific_natural_numbers_expr_l1900_190023

theorem specific_natural_numbers_expr (a b c : ℕ) 
  (h1 : Nat.gcd a b = 1) (h2 : Nat.gcd b c = 1) (h3 : Nat.gcd c a = 1) : 
  ∃ n : ℕ, (n = 7 ∨ n = 8) ∧ (n = (a + b) / c + (b + c) / a + (c + a) / b) :=
by sorry

end NUMINAMATH_GPT_specific_natural_numbers_expr_l1900_190023


namespace NUMINAMATH_GPT_hyperbola_m_range_l1900_190075

-- Given conditions
def is_hyperbola_equation (m : ℝ) : Prop :=
  ∃ x y : ℝ, (4 - m) ≠ 0 ∧ (2 + m) ≠ 0 ∧ x^2 / (4 - m) - y^2 / (2 + m) = 1

-- Prove the range of m is -2 < m < 4
theorem hyperbola_m_range (m : ℝ) : is_hyperbola_equation m → (-2 < m ∧ m < 4) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_m_range_l1900_190075


namespace NUMINAMATH_GPT_combined_rate_of_three_cars_l1900_190064

theorem combined_rate_of_three_cars
  (m : ℕ)
  (ray_avg : ℕ)
  (tom_avg : ℕ)
  (alice_avg : ℕ)
  (h1 : ray_avg = 30)
  (h2 : tom_avg = 15)
  (h3 : alice_avg = 20) :
  let total_distance := 3 * m
  let total_gasoline := m / ray_avg + m / tom_avg + m / alice_avg
  (total_distance / total_gasoline) = 20 := 
by
  sorry

end NUMINAMATH_GPT_combined_rate_of_three_cars_l1900_190064


namespace NUMINAMATH_GPT_find_m_l1900_190068

noncomputable def ellipse := {p : ℝ × ℝ | (p.1 ^ 2 / 25) + (p.2 ^ 2 / 16) = 1}
noncomputable def hyperbola (m : ℝ) := {p : ℝ × ℝ | (p.1 ^ 2 / m) - (p.2 ^ 2 / 5) = 1}

theorem find_m (m : ℝ) (h1 : ∃ f : ℝ × ℝ, f ∈ ellipse ∧ f ∈ hyperbola m) : m = 4 := by
  sorry

end NUMINAMATH_GPT_find_m_l1900_190068


namespace NUMINAMATH_GPT_find_e_of_x_l1900_190093

noncomputable def x_plus_inv_x_eq_five (x : ℝ) : Prop :=
  x + (1 / x) = 5

theorem find_e_of_x (x : ℝ) (h : x_plus_inv_x_eq_five x) : 
  x^2 + (1 / x)^2 = 23 := sorry

end NUMINAMATH_GPT_find_e_of_x_l1900_190093


namespace NUMINAMATH_GPT_distance_between_points_on_line_l1900_190037

theorem distance_between_points_on_line 
  (p q r s : ℝ)
  (line_eq : q = 2 * p + 3) 
  (s_eq : s = 2 * r + 6) :
  Real.sqrt ((r - p)^2 + (s - q)^2) = Real.sqrt (5 * (r - p)^2 + 12 * (r - p) + 9) :=
sorry

end NUMINAMATH_GPT_distance_between_points_on_line_l1900_190037


namespace NUMINAMATH_GPT_number_of_connections_l1900_190072

theorem number_of_connections (n m : ℕ) (h1 : n = 30) (h2 : m = 4) :
    (n * m) / 2 = 60 := by
  -- Since each switch is connected to 4 others,
  -- and each connection is counted twice, 
  -- the number of unique connections is 60.
  sorry

end NUMINAMATH_GPT_number_of_connections_l1900_190072


namespace NUMINAMATH_GPT_initial_bags_l1900_190000

variable (b : ℕ)

theorem initial_bags (h : 5 * (b - 2) = 45) : b = 11 := 
by 
  sorry

end NUMINAMATH_GPT_initial_bags_l1900_190000


namespace NUMINAMATH_GPT_triangle_formation_l1900_190056

theorem triangle_formation (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
  (h₄ : c^2 = a^2 + b^2 + a * b) : 
  a + b > c ∧ a + c > b ∧ c + (a + b) > a :=
by
  sorry

end NUMINAMATH_GPT_triangle_formation_l1900_190056


namespace NUMINAMATH_GPT_joan_remaining_balloons_l1900_190036

def initial_balloons : ℕ := 9
def lost_balloons : ℕ := 2
def remaining_balloons : ℕ := initial_balloons - lost_balloons

theorem joan_remaining_balloons : remaining_balloons = 7 := by
  sorry

end NUMINAMATH_GPT_joan_remaining_balloons_l1900_190036


namespace NUMINAMATH_GPT_problem_equiv_conditions_l1900_190012

theorem problem_equiv_conditions (n : ℕ) :
  (∀ a : ℕ, n ∣ a^n - a) ↔ (∀ p : ℕ, p ∣ n → Prime p → ¬ p^2 ∣ n ∧ (p - 1) ∣ (n - 1)) :=
sorry

end NUMINAMATH_GPT_problem_equiv_conditions_l1900_190012


namespace NUMINAMATH_GPT_cleaning_time_is_correct_l1900_190006

-- Define the given conditions
def vacuuming_minutes_per_day : ℕ := 30
def vacuuming_days_per_week : ℕ := 3
def dusting_minutes_per_day : ℕ := 20
def dusting_days_per_week : ℕ := 2

-- Define the total cleaning time per week
def total_cleaning_time_per_week : ℕ :=
  (vacuuming_minutes_per_day * vacuuming_days_per_week) + (dusting_minutes_per_day * dusting_days_per_week)

-- State the theorem we want to prove
theorem cleaning_time_is_correct : total_cleaning_time_per_week = 130 := by
  sorry

end NUMINAMATH_GPT_cleaning_time_is_correct_l1900_190006


namespace NUMINAMATH_GPT_inequality_solution_set_l1900_190018

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_monotonically_decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x

theorem inequality_solution_set
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_mono_dec : is_monotonically_decreasing_on_nonneg f) :
  { x : ℝ | f 1 - f (1 / x) < 0 } = { x : ℝ | x < -1 ∨ x > 1 } :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l1900_190018


namespace NUMINAMATH_GPT_intersection_M_N_l1900_190046

open Set

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | (x + 2) * (x - 1) < 0}

theorem intersection_M_N : M ∩ N = {-1, 0} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1900_190046


namespace NUMINAMATH_GPT_solve_system_l1900_190053

theorem solve_system :
  ∃ x y : ℚ, (4 * x - 7 * y = -20) ∧ (9 * x + 3 * y = -21) ∧ (x = -69 / 25) ∧ (y = 32 / 25) := by
  sorry

end NUMINAMATH_GPT_solve_system_l1900_190053


namespace NUMINAMATH_GPT_power_six_tens_digit_l1900_190091

def tens_digit (x : ℕ) : ℕ := (x / 10) % 10

theorem power_six_tens_digit (n : ℕ) (hn : tens_digit (6^n) = 1) : n = 3 :=
sorry

end NUMINAMATH_GPT_power_six_tens_digit_l1900_190091


namespace NUMINAMATH_GPT_football_sampling_l1900_190045

theorem football_sampling :
  ∀ (total_members football_members basketball_members volleyball_members total_sample : ℕ),
  total_members = 120 →
  football_members = 40 →
  basketball_members = 60 →
  volleyball_members = 20 →
  total_sample = 24 →
  (total_sample * football_members / (football_members + basketball_members + volleyball_members) = 8) :=
by 
  intros total_members football_members basketball_members volleyball_members total_sample h_total_members h_football_members h_basketball_members h_volleyball_members h_total_sample
  sorry

end NUMINAMATH_GPT_football_sampling_l1900_190045


namespace NUMINAMATH_GPT_root_of_equation_in_interval_l1900_190080

theorem root_of_equation_in_interval :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ 2^x = 2 - x := 
sorry

end NUMINAMATH_GPT_root_of_equation_in_interval_l1900_190080


namespace NUMINAMATH_GPT_opposite_numbers_l1900_190014

theorem opposite_numbers (a b : ℤ) (h1 : -5^2 = a) (h2 : (-5)^2 = b) : a = -b :=
by sorry

end NUMINAMATH_GPT_opposite_numbers_l1900_190014


namespace NUMINAMATH_GPT_sum_of_four_circles_l1900_190022

open Real

theorem sum_of_four_circles:
  ∀ (s c : ℝ), 
  (2 * s + 3 * c = 26) → 
  (3 * s + 2 * c = 23) → 
  (4 * c = 128 / 5) :=
by
  intros s c h1 h2
  sorry

end NUMINAMATH_GPT_sum_of_four_circles_l1900_190022


namespace NUMINAMATH_GPT_combined_cost_of_items_is_221_l1900_190054

def wallet_cost : ℕ := 22
def purse_cost : ℕ := 4 * wallet_cost - 3
def shoes_cost : ℕ := wallet_cost + purse_cost + 7
def combined_cost : ℕ := wallet_cost + purse_cost + shoes_cost

theorem combined_cost_of_items_is_221 : combined_cost = 221 := by
  sorry

end NUMINAMATH_GPT_combined_cost_of_items_is_221_l1900_190054


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1900_190069

theorem sufficient_but_not_necessary (x y : ℝ) (h₁ : x = 2) (h₂ : y = -1) :
    (x + y - 1 = 0) ∧ ¬ ∀ x y, (x + y - 1 = 0) → (x = 2 ∧ y = -1) :=
  by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1900_190069


namespace NUMINAMATH_GPT_smallest_four_digit_number_l1900_190015

theorem smallest_four_digit_number :
  ∃ m : ℕ, (1000 ≤ m) ∧ (m < 10000) ∧ (∃ n : ℕ, 21 * m = n^2) ∧ m = 1029 :=
by sorry

end NUMINAMATH_GPT_smallest_four_digit_number_l1900_190015


namespace NUMINAMATH_GPT_hypotenuse_length_l1900_190092

theorem hypotenuse_length {a b c : ℝ} (h1 : a = 3) (h2 : b = 4) (h3 : c ^ 2 = a ^ 2 + b ^ 2) : c = 5 :=
by
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l1900_190092
