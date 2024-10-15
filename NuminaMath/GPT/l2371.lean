import Mathlib

namespace NUMINAMATH_GPT_ellipse_circle_inequality_l2371_237110

theorem ellipse_circle_inequality
  (a b : ℝ) (x y : ℝ)
  (x1 y1 x2 y2 : ℝ)
  (h_ellipse1 : (x1^2) / (a^2) + (y1^2) / (b^2) = 1)
  (h_ellipse2 : (x2^2) / (a^2) + (y2^2) / (b^2) = 1)
  (h_ab : a > b ∧ b > 0)
  (h_circle : (x - x1) * (x - x2) + (y - y1) * (y - y2) = 0) :
  x^2 + y^2 ≤ (3/2) * a^2 + (1/2) * b^2 :=
sorry

end NUMINAMATH_GPT_ellipse_circle_inequality_l2371_237110


namespace NUMINAMATH_GPT_balance_expenses_l2371_237164

-- Define the basic amounts paid by Alice, Bob, and Carol
def alicePaid : ℕ := 120
def bobPaid : ℕ := 150
def carolPaid : ℕ := 210

-- The total expenditure
def totalPaid : ℕ := alicePaid + bobPaid + carolPaid

-- Each person's share of the total expenses
def eachShare : ℕ := totalPaid / 3

-- Amount Alice should give to balance the expenses
def a : ℕ := eachShare - alicePaid

-- Amount Bob should give to balance the expenses
def b : ℕ := eachShare - bobPaid

-- The statement to be proven
theorem balance_expenses : a - b = 30 :=
by
  sorry

end NUMINAMATH_GPT_balance_expenses_l2371_237164


namespace NUMINAMATH_GPT_candy_peanut_butter_is_192_l2371_237102

/-
   Define the conditions and the statement to be proved.
   The definitions follow directly from the problem's conditions.
-/
def candy_problem : Prop :=
  ∃ (peanut_butter_jar grape_jar banana_jar coconut_jar : ℕ),
    banana_jar = 43 ∧
    grape_jar = banana_jar + 5 ∧
    peanut_butter_jar = 4 * grape_jar ∧
    coconut_jar = 2 * banana_jar - 10 ∧
    peanut_butter_jar = 192
  -- The tuple (question, conditions, correct answer) is translated into this lemma

theorem candy_peanut_butter_is_192 : candy_problem :=
  by
    -- Skipping the actual proof as requested
    sorry

end NUMINAMATH_GPT_candy_peanut_butter_is_192_l2371_237102


namespace NUMINAMATH_GPT_find_length_AB_l2371_237158

variables {A B C D E : Type} -- Define variables A, B, C, D, E as types, representing points

-- Define lengths of the segments AD and CD
def length_AD : ℝ := 2
def length_CD : ℝ := 2

-- Define the angles at vertices B, C, and D
def angle_B : ℝ := 30
def angle_C : ℝ := 90
def angle_D : ℝ := 120

-- The goal is to prove the length of segment AB
theorem find_length_AB : 
  (∃ (A B C D : Type) 
    (angle_B angle_C angle_D length_AD length_CD : ℝ), 
      angle_B = 30 ∧ 
      angle_C = 90 ∧ 
      angle_D = 120 ∧ 
      length_AD = 2 ∧ 
      length_CD = 2) → 
  (length_AB = 6) := by sorry

end NUMINAMATH_GPT_find_length_AB_l2371_237158


namespace NUMINAMATH_GPT_area_of_triangle_AEB_l2371_237191

structure Rectangle :=
  (A B C D : Type)
  (AB : ℝ)
  (BC : ℝ)
  (F G E : Type)
  (DF : ℝ)
  (GC : ℝ)
  (AF_BG_intersect_at_E : Prop)

def rectangle_example : Rectangle := {
  A := Unit,
  B := Unit,
  C := Unit,
  D := Unit,
  AB := 8,
  BC := 4,
  F := Unit,
  G := Unit,
  E := Unit,
  DF := 2,
  GC := 3,
  AF_BG_intersect_at_E := true
}

theorem area_of_triangle_AEB (r : Rectangle) (h : r = rectangle_example) :
  ∃ area : ℝ, area = 128 / 3 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_AEB_l2371_237191


namespace NUMINAMATH_GPT_parallel_lines_necessary_and_sufficient_l2371_237133

-- Define the lines l1 and l2
def line1 (a : ℝ) (x y : ℝ) : Prop := 2 * x - a * y + 1 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := (a - 1) * x - y + a = 0

-- State the theorem
theorem parallel_lines_necessary_and_sufficient (a : ℝ) :
  (∀ x y : ℝ, line1 a x y ↔ line2 a x y) ↔ a = 2 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_parallel_lines_necessary_and_sufficient_l2371_237133


namespace NUMINAMATH_GPT_division_sequence_l2371_237186

theorem division_sequence : (120 / 5) / 2 / 3 = 4 := by
  sorry

end NUMINAMATH_GPT_division_sequence_l2371_237186


namespace NUMINAMATH_GPT_ajay_distance_l2371_237142

/- Definitions -/
def speed : ℝ := 50 -- Ajay's speed in km/hour
def time : ℝ := 30 -- Time taken in hours

/- Theorem statement -/
theorem ajay_distance : (speed * time = 1500) :=
by
  sorry

end NUMINAMATH_GPT_ajay_distance_l2371_237142


namespace NUMINAMATH_GPT_pace_ratio_l2371_237160

variable (P P' D : ℝ)

-- Usual time to reach the office in minutes
def T_usual := 120

-- Time to reach the office on the late day in minutes
def T_late := 140

-- Distance to the office is the same
def office_distance_usual := P * T_usual
def office_distance_late := P' * T_late

theorem pace_ratio (h : office_distance_usual = office_distance_late) : P' / P = 6 / 7 :=
by
  sorry

end NUMINAMATH_GPT_pace_ratio_l2371_237160


namespace NUMINAMATH_GPT_four_integers_sum_product_odd_impossible_l2371_237124

theorem four_integers_sum_product_odd_impossible (a b c d : ℤ) :
  ¬ (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ d % 2 = 1 ∧ 
     (a + b + c + d) % 2 = 1) :=
by
  sorry

end NUMINAMATH_GPT_four_integers_sum_product_odd_impossible_l2371_237124


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l2371_237183

-- Define the conditions and question in Lean 4
variable (a : ℝ) 

-- State the theorem based on the conditions and the correct answer
theorem necessary_and_sufficient_condition :
  (a > 0) ↔ (
    let z := (⟨-a, -5⟩ : ℂ)
    ∃ (x y : ℝ), (z = x + y * I) ∧ x < 0 ∧ y < 0
  ) := by
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l2371_237183


namespace NUMINAMATH_GPT_determine_x1_l2371_237140

theorem determine_x1
  (x1 x2 x3 x4 : ℝ)
  (h1 : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1/3) :
  x1 = 4/5 :=
by
  sorry

end NUMINAMATH_GPT_determine_x1_l2371_237140


namespace NUMINAMATH_GPT_time_to_cross_is_correct_l2371_237174

noncomputable def train_cross_bridge_time : ℝ :=
  let length_train := 130
  let speed_train_kmh := 45
  let length_bridge := 245.03
  let speed_train_ms := (speed_train_kmh * 1000) / 3600
  let total_distance := length_train + length_bridge
  let time := total_distance / speed_train_ms
  time

theorem time_to_cross_is_correct : train_cross_bridge_time = 30.0024 :=
by
  sorry

end NUMINAMATH_GPT_time_to_cross_is_correct_l2371_237174


namespace NUMINAMATH_GPT_largest_composite_not_written_l2371_237118

theorem largest_composite_not_written (n : ℕ) (hn : n = 2022) : ¬ ∃ d > 1, 2033 = n + d := 
by
  sorry

end NUMINAMATH_GPT_largest_composite_not_written_l2371_237118


namespace NUMINAMATH_GPT_length_of_leg_of_isosceles_right_triangle_l2371_237120

def is_isosceles_right_triangle (a b h : ℝ) : Prop :=
  a = b ∧ a^2 + b^2 = h^2

def median_to_hypotenuse (m h : ℝ) : Prop :=
  m = h / 2

theorem length_of_leg_of_isosceles_right_triangle (m : ℝ) (h a : ℝ)
  (h1 : median_to_hypotenuse m h)
  (h2 : h = 2 * m)
  (h3 : is_isosceles_right_triangle a a h) :
  a = 15 * Real.sqrt 2 :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_length_of_leg_of_isosceles_right_triangle_l2371_237120


namespace NUMINAMATH_GPT_find_m_l2371_237197

open Set

def U : Set ℕ := {0, 1, 2, 3}
def A (m : ℤ) : Set ℕ := {x ∈ U | x^2 + m * x = 0}
def complement_A (m : ℤ) : Set ℕ := {1, 2}

theorem find_m (m : ℤ) (hA : complement_A m = U \ A m) : m = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l2371_237197


namespace NUMINAMATH_GPT_nine_skiers_four_overtakes_impossible_l2371_237189

theorem nine_skiers_four_overtakes_impossible :
  ∀ (skiers : Fin 9 → ℝ),  -- skiers are represented by their speeds
  (∀ i j, i < j → skiers i ≤ skiers j) →  -- skiers start sequentially and maintain constant speeds
  ¬(∀ i, (∃ a b : Fin 9, (a ≠ i ∧ b ≠ i ∧ (skiers a < skiers i ∧ skiers i < skiers b ∨ skiers b < skiers i ∧ skiers i < skiers a)))) →
    false := 
by
  sorry

end NUMINAMATH_GPT_nine_skiers_four_overtakes_impossible_l2371_237189


namespace NUMINAMATH_GPT_positive_integer_representation_l2371_237176

theorem positive_integer_representation (a b c n : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) 
  (h₄ : n = (abc + a * b + a) / (abc + c * b + c)) : n = 1 ∨ n = 2 := 
by
  sorry

end NUMINAMATH_GPT_positive_integer_representation_l2371_237176


namespace NUMINAMATH_GPT_balloon_arrangements_l2371_237198

theorem balloon_arrangements :
  let n := 7
  let k1 := 2
  let k2 := 2
  (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 :=
by
  -- Use the conditions directly from the problem
  let n := 7
  let k1 := 2
  let k2 := 2
  -- Mathematically, the number of arrangements of the multiset
  have h : (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 := sorry
  exact h

end NUMINAMATH_GPT_balloon_arrangements_l2371_237198


namespace NUMINAMATH_GPT_David_is_8_years_older_than_Scott_l2371_237130

noncomputable def DavidAge : ℕ := 14 -- Since David was 8 years old, 6 years ago
noncomputable def RichardAge : ℕ := DavidAge + 6
noncomputable def ScottAge : ℕ := (RichardAge + 8) / 2 - 8
noncomputable def AgeDifference : ℕ := DavidAge - ScottAge

theorem David_is_8_years_older_than_Scott :
  AgeDifference = 8 :=
by
  sorry

end NUMINAMATH_GPT_David_is_8_years_older_than_Scott_l2371_237130


namespace NUMINAMATH_GPT_rhombus_perimeter_l2371_237100

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 16) : 
  4 * (Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) = 40 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_perimeter_l2371_237100


namespace NUMINAMATH_GPT_waiter_total_customers_l2371_237199

theorem waiter_total_customers (tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) (tables_eq : tables = 6) (women_eq : women_per_table = 3) (men_eq : men_per_table = 5) :
  tables * (women_per_table + men_per_table) = 48 :=
by
  sorry

end NUMINAMATH_GPT_waiter_total_customers_l2371_237199


namespace NUMINAMATH_GPT_satellite_modular_units_24_l2371_237122

-- Define basic parameters
variables (U N S : ℕ)
def fraction_upgraded : ℝ := 0.2

-- Define the conditions as Lean premises
axiom non_upgraded_per_unit_eq_sixth_total_upgraded : N = S / 6
axiom fraction_sensors_upgraded : (S : ℝ) = fraction_upgraded * (S + U * N)

-- The main statement to be proved
theorem satellite_modular_units_24 (h1 : N = S / 6) (h2 : (S : ℝ) = fraction_upgraded * (S + U * N)) : U = 24 :=
by
  -- The actual proof steps will be written here.
  sorry

end NUMINAMATH_GPT_satellite_modular_units_24_l2371_237122


namespace NUMINAMATH_GPT_parity_of_exponentiated_sum_l2371_237155

theorem parity_of_exponentiated_sum
  : (1 ^ 1994 + 9 ^ 1994 + 8 ^ 1994 + 6 ^ 1994) % 2 = 0 := 
by
  sorry

end NUMINAMATH_GPT_parity_of_exponentiated_sum_l2371_237155


namespace NUMINAMATH_GPT_inequality_holds_l2371_237123

theorem inequality_holds (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x * y = 4) :
  (1 / (x + 3) + 1 / (y + 3) ≤ 2 / 5) ∧ (1 / (x + 3) + 1 / (y + 3) = 2 / 5 ↔ x = 2 ∧ y = 2) :=
sorry

end NUMINAMATH_GPT_inequality_holds_l2371_237123


namespace NUMINAMATH_GPT_cube_volume_surface_area_x_l2371_237173

theorem cube_volume_surface_area_x (x s : ℝ) (h1 : s^3 = 8 * x) (h2 : 6 * s^2 = 2 * x) : x = 1728 :=
by
  sorry

end NUMINAMATH_GPT_cube_volume_surface_area_x_l2371_237173


namespace NUMINAMATH_GPT_find_angle_l2371_237143

theorem find_angle (A : ℝ) (h : 0 < A ∧ A < π) 
  (c : 4 * π * Real.sin A - 3 * Real.arccos (-1/2) = 0) :
  A = π / 6 ∨ A = 5 * π / 6 :=
sorry

end NUMINAMATH_GPT_find_angle_l2371_237143


namespace NUMINAMATH_GPT_benjamin_decade_expense_l2371_237168

-- Define the constants
def yearly_expense : ℕ := 3000
def years : ℕ := 10

-- Formalize the statement
theorem benjamin_decade_expense : yearly_expense * years = 30000 := 
by
  sorry

end NUMINAMATH_GPT_benjamin_decade_expense_l2371_237168


namespace NUMINAMATH_GPT_tangent_line_through_point_l2371_237180

theorem tangent_line_through_point (a : ℝ) : 
  ∃ l : ℝ → ℝ, 
    (∀ x y : ℝ, (x - 1)^2 + y^2 = 4 → y = a) ∧ 
    (∀ x y : ℝ, y = l x → (x - 1)^2 + y^2 = 4) → 
    a = 0 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_through_point_l2371_237180


namespace NUMINAMATH_GPT_bricks_in_chimney_proof_l2371_237179

noncomputable def bricks_in_chimney (h : ℕ) : Prop :=
  let brenda_rate := h / 8
  let brandon_rate := h / 12
  let combined_rate_with_decrease := (brenda_rate + brandon_rate) - 12
  (6 * combined_rate_with_decrease = h) 

theorem bricks_in_chimney_proof : ∃ h : ℕ, bricks_in_chimney h ∧ h = 288 :=
sorry

end NUMINAMATH_GPT_bricks_in_chimney_proof_l2371_237179


namespace NUMINAMATH_GPT_unique_diff_subset_l2371_237117

noncomputable def exists_unique_diff_subset : Prop :=
  ∃ S : Set ℕ, 
    (∀ n : ℕ, n > 0 → ∃! (a b : ℕ), a ∈ S ∧ b ∈ S ∧ n = a - b)

theorem unique_diff_subset : exists_unique_diff_subset :=
  sorry

end NUMINAMATH_GPT_unique_diff_subset_l2371_237117


namespace NUMINAMATH_GPT_problem_translation_l2371_237105

variables {a : ℕ → ℤ} (S : ℕ → ℤ)

-- Definition of the arithmetic sequence and its sum function
def is_arithmetic_sequence (a : ℕ → ℤ) :=
  ∃ (d : ℤ), ∀ (n m : ℕ), a (n + 1) = a n + d

-- Sum of the first n terms defined recursively
def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  if n = 0 then 0 else a n + sum_first_n_terms a (n - 1)

-- Conditions
axiom h1 : is_arithmetic_sequence a
axiom h2 : S 5 > S 6

-- To be proved: Option D does not necessarily hold
theorem problem_translation : ¬(a 3 + a 6 + a 12 < 2 * a 7) := sorry

end NUMINAMATH_GPT_problem_translation_l2371_237105


namespace NUMINAMATH_GPT_cupSaucersCombination_cupSaucerSpoonCombination_twoDifferentItemsCombination_l2371_237171

-- Part (a)
theorem cupSaucersCombination :
  (5 : ℕ) * (3 : ℕ) = 15 :=
by
  -- Proof goes here
  sorry

-- Part (b)
theorem cupSaucerSpoonCombination :
  (5 : ℕ) * (3 : ℕ) * (4 : ℕ) = 60 :=
by
  -- Proof goes here
  sorry

-- Part (c)
theorem twoDifferentItemsCombination :
  (5 * 3 + 5 * 4 + 3 * 4 : ℕ) = 47 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_cupSaucersCombination_cupSaucerSpoonCombination_twoDifferentItemsCombination_l2371_237171


namespace NUMINAMATH_GPT_cos_3theta_l2371_237178

theorem cos_3theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : Real.cos (3 * θ) = -11 / 16 := by
  sorry

end NUMINAMATH_GPT_cos_3theta_l2371_237178


namespace NUMINAMATH_GPT_Frank_work_hours_l2371_237136

def hoursWorked (h_monday h_tuesday h_wednesday h_thursday h_friday h_saturday : Nat) : Nat :=
  h_monday + h_tuesday + h_wednesday + h_thursday + h_friday + h_saturday

theorem Frank_work_hours
  (h_monday : Nat := 8)
  (h_tuesday : Nat := 10)
  (h_wednesday : Nat := 7)
  (h_thursday : Nat := 9)
  (h_friday : Nat := 6)
  (h_saturday : Nat := 4) :
  hoursWorked h_monday h_tuesday h_wednesday h_thursday h_friday h_saturday = 44 :=
by
  unfold hoursWorked
  sorry

end NUMINAMATH_GPT_Frank_work_hours_l2371_237136


namespace NUMINAMATH_GPT_min_time_calculation_l2371_237169

noncomputable def min_time_to_receive_keys (diameter cyclist_speed_road cyclist_speed_alley pedestrian_speed : ℝ) : ℝ :=
  let radius := diameter / 2
  let circumference := 2 * Real.pi * radius
  let distance_pedestrian := pedestrian_speed * 1
  let min_time := (2 * Real.pi * radius - 2 * distance_pedestrian) / (cyclist_speed_road + cyclist_speed_alley)
  min_time

theorem min_time_calculation :
  min_time_to_receive_keys 4 15 20 6 = (2 * Real.pi - 2) / 21 :=
by
  sorry

end NUMINAMATH_GPT_min_time_calculation_l2371_237169


namespace NUMINAMATH_GPT_three_digit_numbers_proof_l2371_237147

-- Definitions and conditions
def are_digits_distinct (A B C : ℕ) := (A ≠ B) ∧ (B ≠ C) ∧ (A ≠ C)

def is_arithmetic_mean (A B C : ℕ) := 2 * B = A + C

def geometric_mean_property (A B C : ℕ) := 
  (100 * A + 10 * B + C) * (100 * C + 10 * A + B) = (100 * B + 10 * C + A)^2

-- statement of the proof problem
theorem three_digit_numbers_proof :
  ∃ A B C : ℕ, (10 ≤ A) ∧ (A ≤ 99) ∧ (10 ≤ B) ∧ (B ≤ 99) ∧ (10 ≤ C) ∧ (C ≤ 99) ∧
  (A * 100 + B * 10 + C = 432 ∨ A * 100 + B * 10 + C = 864) ∧
  are_digits_distinct A B C ∧
  is_arithmetic_mean A B C ∧
  geometric_mean_property A B C :=
by {
  -- The Lean proof goes here
  sorry
}

end NUMINAMATH_GPT_three_digit_numbers_proof_l2371_237147


namespace NUMINAMATH_GPT_andrew_paid_in_dollars_l2371_237192

def local_currency_to_dollars (units : ℝ) : ℝ := units * 0.25

def cost_of_fruits : ℝ :=
  let cost_grapes := 7 * 68
  let cost_mangoes := 9 * 48
  let cost_apples := 5 * 55
  let cost_oranges := 4 * 38
  let total_cost_grapes_mangoes := cost_grapes + cost_mangoes
  let total_cost_apples_oranges := cost_apples + cost_oranges
  let discount_grapes_mangoes := 0.10 * total_cost_grapes_mangoes
  let discounted_grapes_mangoes := total_cost_grapes_mangoes - discount_grapes_mangoes
  let discounted_apples_oranges := total_cost_apples_oranges - 25
  let total_discounted_cost := discounted_grapes_mangoes + discounted_apples_oranges
  let sales_tax := 0.05 * total_discounted_cost
  let total_tax := sales_tax + 15
  let total_amount_with_taxes := total_discounted_cost + total_tax
  total_amount_with_taxes

theorem andrew_paid_in_dollars : local_currency_to_dollars cost_of_fruits = 323.79 :=
  by
  sorry

end NUMINAMATH_GPT_andrew_paid_in_dollars_l2371_237192


namespace NUMINAMATH_GPT_number_of_trees_in_park_l2371_237141

def number_of_trees (length width area_per_tree : ℕ) : ℕ :=
  (length * width) / area_per_tree

theorem number_of_trees_in_park :
  number_of_trees 1000 2000 20 = 100000 :=
by
  sorry

end NUMINAMATH_GPT_number_of_trees_in_park_l2371_237141


namespace NUMINAMATH_GPT_complete_the_square_l2371_237177

theorem complete_the_square (m n : ℕ) :
  (∀ x : ℝ, x^2 - 6 * x = 1 → (x - m)^2 = n) → m + n = 13 :=
by
  sorry

end NUMINAMATH_GPT_complete_the_square_l2371_237177


namespace NUMINAMATH_GPT_men_wages_l2371_237182

theorem men_wages (W : ℕ) (wage : ℕ) :
  (5 + W + 8) * wage = 75 ∧ 5 * wage = W * wage ∧ W * wage = 8 * wage → 
  wage = 5 := 
by
  sorry

end NUMINAMATH_GPT_men_wages_l2371_237182


namespace NUMINAMATH_GPT_max_blocks_fit_l2371_237125

-- Defining the dimensions of the box and blocks
def box_length : ℝ := 4
def box_width : ℝ := 3
def box_height : ℝ := 2

def block_length : ℝ := 3
def block_width : ℝ := 1
def block_height : ℝ := 1

-- Theorem stating the maximum number of blocks that fit
theorem max_blocks_fit : (24 / 3 = 8) ∧ (1 * 3 * 2 = 6) → 6 = 6 := 
by
  sorry

end NUMINAMATH_GPT_max_blocks_fit_l2371_237125


namespace NUMINAMATH_GPT_problem_statement_l2371_237107

noncomputable def f_B (x : ℝ) : ℝ := -x^2
noncomputable def f_D (x : ℝ) : ℝ := Real.cos x

theorem problem_statement :
  (∀ x : ℝ, f_B (-x) = f_B x) ∧ (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → x2 < 1 → f_B x1 > f_B x2) ∧
  (∀ x : ℝ, f_D (-x) = f_D x) ∧ (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → x2 < 1 → f_D x1 > f_D x2) :=
  sorry

end NUMINAMATH_GPT_problem_statement_l2371_237107


namespace NUMINAMATH_GPT_equal_area_centroid_S_l2371_237111

noncomputable def P : ℝ × ℝ := (-4, 3)
noncomputable def Q : ℝ × ℝ := (7, -5)
noncomputable def R : ℝ × ℝ := (0, 6)
noncomputable def centroid (A B C : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

theorem equal_area_centroid_S (x y : ℝ) (h : (x, y) = centroid P Q R) :
  10 * x + y = 34 / 3 := by
  sorry

end NUMINAMATH_GPT_equal_area_centroid_S_l2371_237111


namespace NUMINAMATH_GPT_tan_theta_minus_pi_over_4_l2371_237113

theorem tan_theta_minus_pi_over_4 (θ : Real) (h1 : θ ∈ Set.Ioc (-(π / 2)) 0)
  (h2 : Real.sin (θ + π / 4) = 3 / 5) : Real.tan (θ - π / 4) = - (4 / 3) :=
by
  /- Proof goes here -/
  sorry

end NUMINAMATH_GPT_tan_theta_minus_pi_over_4_l2371_237113


namespace NUMINAMATH_GPT_scientific_notation_GDP_l2371_237187

theorem scientific_notation_GDP (h : 1 = 10^9) : 32.07 * 10^9 = 3.207 * 10^10 := by
  sorry

end NUMINAMATH_GPT_scientific_notation_GDP_l2371_237187


namespace NUMINAMATH_GPT_odd_square_diff_div_by_eight_l2371_237139

theorem odd_square_diff_div_by_eight (n p : ℤ) : 
  (2 * n + 1)^2 - (2 * p + 1)^2 % 8 = 0 := 
by 
-- Here we declare the start of the proof.
  sorry

end NUMINAMATH_GPT_odd_square_diff_div_by_eight_l2371_237139


namespace NUMINAMATH_GPT_band_members_minimum_n_l2371_237108

theorem band_members_minimum_n 
  (n : ℕ) 
  (h1 : n % 6 = 3) 
  (h2 : n % 8 = 5) 
  (h3 : n % 9 = 7) : 
  n ≥ 165 := 
sorry

end NUMINAMATH_GPT_band_members_minimum_n_l2371_237108


namespace NUMINAMATH_GPT_find_x_collinear_l2371_237170

def vec := ℝ × ℝ

def collinear (u v: vec): Prop :=
  ∃ k: ℝ, u = (k * v.1, k * v.2)

theorem find_x_collinear:
  ∀ (x: ℝ), (let a : vec := (1, 2)
              let b : vec := (x, 1)
              collinear a (a.1 - b.1, a.2 - b.2)) → x = 1 / 2 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_find_x_collinear_l2371_237170


namespace NUMINAMATH_GPT_sum_sublist_eq_100_l2371_237162

theorem sum_sublist_eq_100 {l : List ℕ}
  (h_len : l.length = 2 * 31100)
  (h_max : ∀ x ∈ l, x ≤ 100)
  (h_sum : l.sum = 200) :
  ∃ (s : List ℕ), s ⊆ l ∧ s.sum = 100 := 
sorry

end NUMINAMATH_GPT_sum_sublist_eq_100_l2371_237162


namespace NUMINAMATH_GPT_green_peppers_weight_l2371_237188

theorem green_peppers_weight (total_weight : ℝ) (w : ℝ) (h1 : total_weight = 5.666666667)
  (h2 : 2 * w = total_weight) : w = 2.8333333335 :=
by
  sorry

end NUMINAMATH_GPT_green_peppers_weight_l2371_237188


namespace NUMINAMATH_GPT_sum_first_10_terms_l2371_237181

variable (a : ℕ → ℕ)

def condition (p q : ℕ) : Prop :=
  p + q = 11 ∧ p < q

axiom condition_a_p_a_q : ∀ (p q : ℕ), (condition p q) → (a p + a q = 2^p)

theorem sum_first_10_terms (a : ℕ → ℕ) (h : ∀ (p q : ℕ), condition p q → a p + a q = 2^p) : 
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = 62) :=
by 
  sorry

end NUMINAMATH_GPT_sum_first_10_terms_l2371_237181


namespace NUMINAMATH_GPT_four_times_sum_of_cubes_gt_cube_sum_l2371_237196

theorem four_times_sum_of_cubes_gt_cube_sum
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  4 * (a^3 + b^3) > (a + b)^3 :=
by
  sorry

end NUMINAMATH_GPT_four_times_sum_of_cubes_gt_cube_sum_l2371_237196


namespace NUMINAMATH_GPT_find_x_l2371_237134

theorem find_x : ∃ x : ℝ, (0.40 * x - 30 = 50) ∧ x = 200 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2371_237134


namespace NUMINAMATH_GPT_triangle_number_arrangement_l2371_237156

noncomputable def numbers := [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

theorem triangle_number_arrangement : 
  ∃ (f : Fin 9 → Fin 9), 
    (numbers[f 0] + numbers[f 1] + numbers[f 2] = 
     numbers[f 3] + numbers[f 4] + numbers[f 5] ∧ 
     numbers[f 3] + numbers[f 4] + numbers[f 5] = 
     numbers[f 6] + numbers[f 7] + numbers[f 8]) :=
sorry

end NUMINAMATH_GPT_triangle_number_arrangement_l2371_237156


namespace NUMINAMATH_GPT_minimum_handshakes_l2371_237119

noncomputable def min_handshakes (n : ℕ) (k : ℕ) : ℕ :=
  (n * k) / 2

theorem minimum_handshakes (n k : ℕ) (h1 : n = 30) (h2 : k = 3) :
  min_handshakes n k = 45 :=
by
  -- We provide the conditions directly
  -- n = 30, k = 3
  rw [h1, h2]
  -- then show that min_handshakes 30 3 = 45
  show min_handshakes 30 3 = 45
  sorry 

end NUMINAMATH_GPT_minimum_handshakes_l2371_237119


namespace NUMINAMATH_GPT_unknown_number_is_7_l2371_237157

theorem unknown_number_is_7 (x : ℤ) (hx : x > 0)
  (h : (1 / 4 : ℚ) * (10 * x + 7 - x ^ 2) - x = 0) : x = 7 :=
  sorry

end NUMINAMATH_GPT_unknown_number_is_7_l2371_237157


namespace NUMINAMATH_GPT_c_linear_combination_of_a_b_l2371_237114

-- Definitions of vectors a, b, and c as given in the problem
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, -1)
def c : ℝ × ℝ := (-1, 2)

-- Theorem stating the relationship between vectors a, b, and c
theorem c_linear_combination_of_a_b :
  c = (1 / 2 : ℝ) • a + (-3 / 2 : ℝ) • b :=
  sorry

end NUMINAMATH_GPT_c_linear_combination_of_a_b_l2371_237114


namespace NUMINAMATH_GPT_min_cans_needed_l2371_237132

theorem min_cans_needed (C : ℕ → ℕ) (H : C 1 = 15) : ∃ n, C n * n >= 64 ∧ ∀ m, m < n → C 1 * m < 64 :=
by
  sorry

end NUMINAMATH_GPT_min_cans_needed_l2371_237132


namespace NUMINAMATH_GPT_integer_values_b_l2371_237152

theorem integer_values_b (b : ℤ) : 
  (∃ (x1 x2 : ℤ), x1 + x2 = -b ∧ x1 * x2 = 7 * b) ↔ b = 0 ∨ b = 36 ∨ b = -28 ∨ b = -64 :=
by
  sorry

end NUMINAMATH_GPT_integer_values_b_l2371_237152


namespace NUMINAMATH_GPT_Dongdong_test_score_l2371_237103

theorem Dongdong_test_score (a b c : ℕ) (h1 : a + b + c = 280) : a ≥ 94 ∨ b ≥ 94 ∨ c ≥ 94 :=
by
  sorry

end NUMINAMATH_GPT_Dongdong_test_score_l2371_237103


namespace NUMINAMATH_GPT_salary_calculation_l2371_237121

variable {A B : ℝ}

theorem salary_calculation (h1 : A + B = 6000) (h2 : 0.05 * A = 0.15 * B) : A = 4500 :=
by
  sorry

end NUMINAMATH_GPT_salary_calculation_l2371_237121


namespace NUMINAMATH_GPT_solve_for_a_l2371_237116

-- Definitions: Real number a, Imaginary unit i, complex number.
def is_purely_imaginary (z : ℂ) : Prop :=
  z.re = 0

theorem solve_for_a :
  ∀ (a : ℝ) (i : ℂ),
    i = Complex.I →
    is_purely_imaginary ( (3 * i / (1 + 2 * i)) * (1 - (a / 3) * i) ) →
    a = -6 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l2371_237116


namespace NUMINAMATH_GPT_increasing_on_interval_l2371_237190

noncomputable def f1 (x : ℝ) : ℝ := Real.sin x
noncomputable def f2 (x : ℝ) : ℝ := x * Real.exp 2
noncomputable def f3 (x : ℝ) : ℝ := x^3 - x
noncomputable def f4 (x : ℝ) : ℝ := Real.log x - x

theorem increasing_on_interval (x : ℝ) (h : 0 < x) : 
  f2 (x) = x * Real.exp 2 ∧ 
  (∀(x y : ℝ), 0 < x → 0 < y → x < y →  f1 x < f1 y) ∧ 
  (∀(x y : ℝ), 0 < x → 0 < y → x < y →  f3 x < f3 y) ∧ 
  (∀(x y : ℝ), 0 < x → 0 < y → x < y →  f4 x < f4 y) :=
by sorry

end NUMINAMATH_GPT_increasing_on_interval_l2371_237190


namespace NUMINAMATH_GPT_nearest_integer_to_3_plus_sqrt2_pow_four_l2371_237166

open Real

theorem nearest_integer_to_3_plus_sqrt2_pow_four : 
  (∃ n : ℤ, abs (n - (3 + (sqrt 2))^4) < 0.5) ∧ 
  (abs (382 - (3 + (sqrt 2))^4) < 0.5) := 
by 
  sorry

end NUMINAMATH_GPT_nearest_integer_to_3_plus_sqrt2_pow_four_l2371_237166


namespace NUMINAMATH_GPT_total_pens_bought_l2371_237161

theorem total_pens_bought (r : ℕ) (r_gt_10 : r > 10) (r_divides_357 : 357 % r = 0) (r_divides_441 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
by sorry

end NUMINAMATH_GPT_total_pens_bought_l2371_237161


namespace NUMINAMATH_GPT_max_odd_numbers_in_pyramid_l2371_237145

-- Define the properties of the pyramid
def is_sum_of_immediate_below (p : Nat → Nat → Nat) : Prop :=
  ∀ r c : Nat, r > 0 → p r c = p (r - 1) c + p (r - 1) (c + 1)

-- Define what it means for a number to be odd
def is_odd (n : Nat) : Prop := n % 2 = 1

-- Define the pyramid structure and number of rows
def pyramid (n : Nat) := { p : Nat → Nat → Nat // is_sum_of_immediate_below p ∧ n = 6 }

-- Theorem statement
theorem max_odd_numbers_in_pyramid (p : Nat → Nat → Nat) (h : is_sum_of_immediate_below p ∧ 6 = 6) : ∃ k : Nat, (∀ i j, is_odd (p i j) → k ≤ 14) := 
sorry

end NUMINAMATH_GPT_max_odd_numbers_in_pyramid_l2371_237145


namespace NUMINAMATH_GPT_even_function_value_l2371_237131

theorem even_function_value (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f (-x) = f x)
  (h_def : ∀ x : ℝ, 0 < x → f x = 2^x + 1) :
  f (-2) = 5 :=
  sorry

end NUMINAMATH_GPT_even_function_value_l2371_237131


namespace NUMINAMATH_GPT_percent_absent_students_l2371_237149

def total_students : ℕ := 180
def num_boys : ℕ := 100
def num_girls : ℕ := 80
def fraction_boys_absent : ℚ := 1 / 5
def fraction_girls_absent : ℚ := 1 / 4

theorem percent_absent_students : 
  (fraction_boys_absent * num_boys + fraction_girls_absent * num_girls) / total_students = 22.22 / 100 := 
  sorry

end NUMINAMATH_GPT_percent_absent_students_l2371_237149


namespace NUMINAMATH_GPT_num_people_in_group_l2371_237165

-- Given conditions as definitions
def cost_per_adult_meal : ℤ := 3
def num_kids : ℤ := 7
def total_cost : ℤ := 15

-- Statement to prove
theorem num_people_in_group : 
  ∃ (num_adults : ℤ), 
    total_cost = num_adults * cost_per_adult_meal ∧ 
    (num_adults + num_kids) = 12 :=
by
  sorry

end NUMINAMATH_GPT_num_people_in_group_l2371_237165


namespace NUMINAMATH_GPT_trains_meet_in_2067_seconds_l2371_237144

def length_of_train1 : ℝ := 100  -- Length of Train 1 in meters
def length_of_train2 : ℝ := 200  -- Length of Train 2 in meters
def initial_distance : ℝ := 630  -- Initial distance between trains in meters
def speed_of_train1_kmh : ℝ := 90  -- Speed of Train 1 in km/h
def speed_of_train2_kmh : ℝ := 72  -- Speed of Train 2 in km/h

noncomputable def speed_of_train1_ms : ℝ := speed_of_train1_kmh * (1000 / 3600)
noncomputable def speed_of_train2_ms : ℝ := speed_of_train2_kmh * (1000 / 3600)
noncomputable def relative_speed : ℝ := speed_of_train1_ms + speed_of_train2_ms
noncomputable def total_distance : ℝ := initial_distance + length_of_train1 + length_of_train2
noncomputable def time_to_meet : ℝ := total_distance / relative_speed

theorem trains_meet_in_2067_seconds : time_to_meet = 20.67 := 
by
  sorry

end NUMINAMATH_GPT_trains_meet_in_2067_seconds_l2371_237144


namespace NUMINAMATH_GPT_cube_root_sum_lt_sqrt_sum_l2371_237153

theorem cube_root_sum_lt_sqrt_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (a^3 + b^3)^(1/3) < (a^2 + b^2)^(1/2) := by
    sorry

end NUMINAMATH_GPT_cube_root_sum_lt_sqrt_sum_l2371_237153


namespace NUMINAMATH_GPT_avg_decrease_by_one_l2371_237129

noncomputable def average_decrease (obs : Fin 7 → ℕ) : ℕ :=
  let sum6 := 90
  let seventh := 8
  let new_sum := sum6 + seventh
  let new_avg := new_sum / 7
  let old_avg := 15
  old_avg - new_avg

theorem avg_decrease_by_one :
  (average_decrease (fun _ => 0)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_avg_decrease_by_one_l2371_237129


namespace NUMINAMATH_GPT_intersection_points_l2371_237193

noncomputable def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 9 * x - 15
noncomputable def parabola2 (x : ℝ) : ℝ := x^2 - 6 * x + 10

noncomputable def x1 : ℝ := (3 + Real.sqrt 209) / 4
noncomputable def x2 : ℝ := (3 - Real.sqrt 209) / 4

noncomputable def y1 : ℝ := parabola1 x1
noncomputable def y2 : ℝ := parabola1 x2

theorem intersection_points :
  (parabola1 x1 = parabola2 x1) ∧ (parabola1 x2 = parabola2 x2) :=
by
  sorry

end NUMINAMATH_GPT_intersection_points_l2371_237193


namespace NUMINAMATH_GPT_correct_calculation_l2371_237150

theorem correct_calculation :
  (∀ (x : ℝ), (x^3 * 2 * x^4 = 2 * x^7) ∧
  (x^6 / x^3 = x^2) ∧
  ((x^3)^4 = x^7) ∧
  (x^2 + x = x^3)) → 
  (∀ (x : ℝ), x^3 * 2 * x^4 = 2 * x^7) :=
by
  intros h x
  have A := h x
  exact A.1

end NUMINAMATH_GPT_correct_calculation_l2371_237150


namespace NUMINAMATH_GPT_eccentricity_of_given_ellipse_l2371_237137

noncomputable def eccentricity_of_ellipse : ℝ :=
  let a : ℝ := 1
  let b : ℝ := 1 / 2
  let c : ℝ := Real.sqrt (a ^ 2 - b ^ 2)
  c / a

theorem eccentricity_of_given_ellipse :
  eccentricity_of_ellipse = Real.sqrt (3) / 2 :=
by
  -- Proof is omitted.
  sorry

end NUMINAMATH_GPT_eccentricity_of_given_ellipse_l2371_237137


namespace NUMINAMATH_GPT_train_length_is_900_l2371_237106

def train_length_crossing_pole (L V : ℕ) : Prop :=
  L = V * 18

def train_length_crossing_platform (L V : ℕ) : Prop :=
  L + 1050 = V * 39

theorem train_length_is_900 (L V : ℕ) (h1 : train_length_crossing_pole L V) (h2 : train_length_crossing_platform L V) : L = 900 := 
by
  sorry

end NUMINAMATH_GPT_train_length_is_900_l2371_237106


namespace NUMINAMATH_GPT_smallest_yellow_candies_l2371_237167
open Nat

theorem smallest_yellow_candies 
  (h_red : ∃ c : ℕ, 16 * c = 720)
  (h_green : ∃ c : ℕ, 18 * c = 720)
  (h_blue : ∃ c : ℕ, 20 * c = 720)
  : ∃ n : ℕ, 30 * n = 720 ∧ n = 24 := 
by
  -- Provide the proof here
  sorry

end NUMINAMATH_GPT_smallest_yellow_candies_l2371_237167


namespace NUMINAMATH_GPT_parabola_distance_l2371_237109

theorem parabola_distance (a : ℝ) :
  (abs (1 + (1 / (4 * a))) = 2 → a = 1 / 4) ∨ 
  (abs (1 - (1 / (4 * a))) = 2 → a = -1 / 12) := by 
  sorry

end NUMINAMATH_GPT_parabola_distance_l2371_237109


namespace NUMINAMATH_GPT_point_coordinates_l2371_237126

/-- Given the vector from point A to point B, if point A is the origin, then point B will have coordinates determined by the vector. -/
theorem point_coordinates (A B: ℝ × ℝ) (v: ℝ × ℝ) 
  (h: A = (0, 0)) (h_v: v = (-2, 4)) (h_ab: B = (A.1 + v.1, A.2 + v.2)): 
  B = (-2, 4) :=
by
  sorry

end NUMINAMATH_GPT_point_coordinates_l2371_237126


namespace NUMINAMATH_GPT_simplify_fraction_l2371_237159

-- Define the numbers involved and state their GCD
def num1 := 90
def num2 := 8100

-- State the GCD condition using a Lean 4 statement
def gcd_condition (a b : ℕ) := Nat.gcd a b = 90

-- Define the original fraction and the simplified fraction
def original_fraction := num1 / num2
def simplified_fraction := 1 / 90

-- State the proof problem that the original fraction simplifies to the simplified fraction
theorem simplify_fraction : gcd_condition num1 num2 → original_fraction = simplified_fraction := 
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2371_237159


namespace NUMINAMATH_GPT_marcia_savings_l2371_237112

def hat_price := 60
def regular_price (n : ℕ) := n * hat_price
def discount_price (discount_percentage: ℕ) (price: ℕ) := price - (price * discount_percentage) / 100
def promotional_price := hat_price + discount_price 25 hat_price + discount_price 35 hat_price

theorem marcia_savings : (regular_price 3 - promotional_price) * 100 / regular_price 3 = 20 :=
by
  -- The proof steps would follow here.
  sorry

end NUMINAMATH_GPT_marcia_savings_l2371_237112


namespace NUMINAMATH_GPT_range_of_b_l2371_237101

noncomputable def f (b x : ℝ) : ℝ := -x^3 + b * x

theorem range_of_b (b : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 1 → -3 * x^2 + b ≥ 0) ↔ b ≥ 3 := sorry

end NUMINAMATH_GPT_range_of_b_l2371_237101


namespace NUMINAMATH_GPT_total_distance_eq_l2371_237194

def distance_traveled_by_bus : ℝ := 2.6
def distance_traveled_by_subway : ℝ := 5.98
def total_distance_traveled : ℝ := distance_traveled_by_bus + distance_traveled_by_subway

theorem total_distance_eq : total_distance_traveled = 8.58 := by
  sorry

end NUMINAMATH_GPT_total_distance_eq_l2371_237194


namespace NUMINAMATH_GPT_trapezium_area_correct_l2371_237195

def a : ℚ := 20  -- Length of the first parallel side
def b : ℚ := 18  -- Length of the second parallel side
def h : ℚ := 20  -- Distance (height) between the parallel sides

def trapezium_area (a b h : ℚ) : ℚ :=
  (1/2) * (a + b) * h

theorem trapezium_area_correct : trapezium_area a b h = 380 := 
  by
    sorry  -- Proof goes here

end NUMINAMATH_GPT_trapezium_area_correct_l2371_237195


namespace NUMINAMATH_GPT_range_of_a_l2371_237135

theorem range_of_a (a b : ℝ) :
  (∀ x : ℝ, (a * x^2 + b * x + 1 < 2)) ∧ (a - b + 1 = 1) → (-4 < a ∧ a ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2371_237135


namespace NUMINAMATH_GPT_fraction_zero_l2371_237175

theorem fraction_zero (x : ℝ) (h₁ : x - 3 = 0) (h₂ : x ≠ 0) : (x - 3) / (4 * x) = 0 :=
by
  sorry

end NUMINAMATH_GPT_fraction_zero_l2371_237175


namespace NUMINAMATH_GPT_george_earnings_after_deductions_l2371_237151

noncomputable def george_total_earnings : ℕ := 35 + 12 + 20 + 21

noncomputable def tax_deduction (total_earnings : ℕ) : ℚ := total_earnings * 0.10

noncomputable def uniform_fee : ℚ := 15

noncomputable def final_earnings (total_earnings : ℕ) (tax_deduction : ℚ) (uniform_fee : ℚ) : ℚ :=
  total_earnings - tax_deduction - uniform_fee

theorem george_earnings_after_deductions : 
  final_earnings george_total_earnings (tax_deduction george_total_earnings) uniform_fee = 64.2 := 
  by
  sorry

end NUMINAMATH_GPT_george_earnings_after_deductions_l2371_237151


namespace NUMINAMATH_GPT_mean_of_four_integers_l2371_237146

theorem mean_of_four_integers (x : ℝ) (h : (78 + 83 + 82 + x) / 4 = 80) : x = 77 ∧ x = 80 - 3 :=
by
  have h1 : 78 + 83 + 82 + x = 4 * 80 := by sorry
  have h2 : 78 + 83 + 82 = 243 := by sorry
  have h3 : 243 + x = 320 := by sorry
  have h4 : x = 320 - 243 := by sorry
  have h5 : x = 77 := by sorry
  have h6 : x = 80 - 3 := by sorry
  exact ⟨h5, h6⟩

end NUMINAMATH_GPT_mean_of_four_integers_l2371_237146


namespace NUMINAMATH_GPT_miae_closer_than_hyori_l2371_237138

def bowl_volume : ℝ := 1000
def miae_estimate : ℝ := 1100
def hyori_estimate : ℝ := 850

def miae_difference : ℝ := abs (miae_estimate - bowl_volume)
def hyori_difference : ℝ := abs (bowl_volume - hyori_estimate)

theorem miae_closer_than_hyori : miae_difference < hyori_difference :=
by
  sorry

end NUMINAMATH_GPT_miae_closer_than_hyori_l2371_237138


namespace NUMINAMATH_GPT_sum_terms_sequence_l2371_237104

noncomputable def geometric_sequence := ℕ → ℝ

variables (a : geometric_sequence)
variables (r : ℝ) (h_pos : ∀ n, a n > 0)

-- Geometric sequence condition
axiom geom_seq (n : ℕ) : a (n + 1) = a n * r

-- Given condition
axiom h_condition : a 3 * a 5 + a 2 * a 10 + 2 * a 4 * a 6 = 100

-- The goal is to prove that a_4 + a_6 = 10
theorem sum_terms_sequence : a 4 + a 6 = 10 :=
by
  sorry

end NUMINAMATH_GPT_sum_terms_sequence_l2371_237104


namespace NUMINAMATH_GPT_primes_equal_l2371_237148

def is_prime (n : ℕ) : Prop := n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_equal (p q r n : ℕ) (h_prime_p : is_prime p) (h_prime_q : is_prime q)
(h_prime_r : is_prime r) (h_pos_n : 0 < n)
(h1 : (p + n) % (q * r) = 0)
(h2 : (q + n) % (r * p) = 0)
(h3 : (r + n) % (p * q) = 0) : p = q ∧ q = r := by
  sorry

end NUMINAMATH_GPT_primes_equal_l2371_237148


namespace NUMINAMATH_GPT_find_c_d_of_cubic_common_roots_l2371_237154

theorem find_c_d_of_cubic_common_roots 
  (c d : ℝ)
  (h1 : ∃ r s : ℝ, r ≠ s ∧ (r ^ 3 + c * r ^ 2 + 12 * r + 7 = 0) ∧ (s ^ 3 + c * s ^ 2 + 12 * s + 7 = 0))
  (h2 : ∃ r s : ℝ, r ≠ s ∧ (r ^ 3 + d * r ^ 2 + 15 * r + 9 = 0) ∧ (s ^ 3 + d * s ^ 2 + 15 * s + 9 = 0)) :
  c = 5 ∧ d = 4 :=
sorry

end NUMINAMATH_GPT_find_c_d_of_cubic_common_roots_l2371_237154


namespace NUMINAMATH_GPT_correct_calculation_l2371_237127

theorem correct_calculation (a : ℝ) : (-a)^10 / (-a)^3 = -a^7 :=
by sorry

end NUMINAMATH_GPT_correct_calculation_l2371_237127


namespace NUMINAMATH_GPT_bakery_baguettes_l2371_237172

theorem bakery_baguettes : 
  ∃ B : ℕ, 
  (∃ B : ℕ, 3 * B - 138 = 6) ∧ 
  B = 48 :=
by
  sorry

end NUMINAMATH_GPT_bakery_baguettes_l2371_237172


namespace NUMINAMATH_GPT_find_y_l2371_237185

theorem find_y
  (x y : ℕ) 
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y)
  (hr : x % y = 8)
  (hq : x / y = 96) 
  (hr_decimal : (x:ℚ) / (y:ℚ) = 96.16) :
  y = 50 := 
sorry

end NUMINAMATH_GPT_find_y_l2371_237185


namespace NUMINAMATH_GPT_find_max_sum_pair_l2371_237128

theorem find_max_sum_pair :
  ∃ a b : ℕ, 2 * a * b + 3 * b = b^2 + 6 * a + 6 ∧ (∀ a' b' : ℕ, 2 * a' * b' + 3 * b' = b'^2 + 6 * a' + 6 → a + b ≥ a' + b') ∧ a = 5 ∧ b = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_max_sum_pair_l2371_237128


namespace NUMINAMATH_GPT_range_of_k_l2371_237163

theorem range_of_k (k : ℝ) : (∃ x : ℝ, k * x^2 - 2 * x + 1 = 0) ↔ k ≤ 1 :=
by sorry

end NUMINAMATH_GPT_range_of_k_l2371_237163


namespace NUMINAMATH_GPT_father_twice_marika_age_in_2036_l2371_237184

-- Definitions of the initial conditions
def marika_age_2006 : ℕ := 10
def father_age_2006 : ℕ := 5 * marika_age_2006

-- Definition of the statement to be proven
theorem father_twice_marika_age_in_2036 : 
  ∃ x : ℕ, (2006 + x = 2036) ∧ (father_age_2006 + x = 2 * (marika_age_2006 + x)) :=
by {
  sorry 
}

end NUMINAMATH_GPT_father_twice_marika_age_in_2036_l2371_237184


namespace NUMINAMATH_GPT_tangent_line_intersect_x_l2371_237115

noncomputable def tangent_intercept_x : ℚ := 9/2

theorem tangent_line_intersect_x (x : ℚ)
  (h₁ : x > 0)
  (h₂ : ∃ r₁ r₂ d : ℚ, r₁ = 3 ∧ r₂ = 5 ∧ d = 12 ∧ x = (r₂ * d) / (r₁ + r₂)) :
  x = tangent_intercept_x :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_intersect_x_l2371_237115
