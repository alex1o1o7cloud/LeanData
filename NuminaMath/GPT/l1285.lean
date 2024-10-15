import Mathlib

namespace NUMINAMATH_GPT_island_knight_majority_villages_l1285_128505

def NumVillages := 1000
def NumInhabitants := 99
def TotalKnights := 54054
def AnswersPerVillage : ℕ := 66 -- Number of villagers who answered "more knights"
def RemainingAnswersPerVillage : ℕ := 33 -- Number of villagers who answered "more liars"

theorem island_knight_majority_villages : 
  ∃ n : ℕ, n = 638 ∧ (66 * n + 33 * (NumVillages - n) = TotalKnights) :=
by -- Begin the proof
  sorry -- Proof to be filled in later

end NUMINAMATH_GPT_island_knight_majority_villages_l1285_128505


namespace NUMINAMATH_GPT_sum_C_D_eq_one_fifth_l1285_128545

theorem sum_C_D_eq_one_fifth (D C : ℚ) :
  (∀ x : ℚ, (Dx - 13) / (x^2 - 9 * x + 20) = C / (x - 4) + 5 / (x - 5)) →
  (C + D) = 1/5 :=
by
  sorry

end NUMINAMATH_GPT_sum_C_D_eq_one_fifth_l1285_128545


namespace NUMINAMATH_GPT_sin_is_odd_l1285_128595

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

theorem sin_is_odd : is_odd_function sin :=
by
  sorry

end NUMINAMATH_GPT_sin_is_odd_l1285_128595


namespace NUMINAMATH_GPT_symmetric_difference_card_l1285_128568

variable (x y : Finset ℤ)
variable (h1 : x.card = 16)
variable (h2 : y.card = 18)
variable (h3 : (x ∩ y).card = 6)

theorem symmetric_difference_card :
  (x \ y ∪ y \ x).card = 22 := by sorry

end NUMINAMATH_GPT_symmetric_difference_card_l1285_128568


namespace NUMINAMATH_GPT_cube_fit_count_cube_volume_percentage_l1285_128541

-- Definitions based on the conditions in the problem.
def box_length : ℕ := 8
def box_width : ℕ := 4
def box_height : ℕ := 12
def cube_side : ℕ := 4

-- Definitions for the calculated values.
def num_cubes_length : ℕ := box_length / cube_side
def num_cubes_width : ℕ := box_width / cube_side
def num_cubes_height : ℕ := box_height / cube_side

def total_cubes : ℕ := num_cubes_length * num_cubes_width * num_cubes_height

def volume_cube : ℕ := cube_side^3
def volume_cubes_total : ℕ := total_cubes * volume_cube
def volume_box : ℕ := box_length * box_width * box_height

def percentage_volume : ℕ := (volume_cubes_total * 100) / volume_box

-- The proof statements.
theorem cube_fit_count : total_cubes = 6 := by
  sorry

theorem cube_volume_percentage : percentage_volume = 100 := by
  sorry

end NUMINAMATH_GPT_cube_fit_count_cube_volume_percentage_l1285_128541


namespace NUMINAMATH_GPT_exponentiation_calculation_l1285_128563

theorem exponentiation_calculation : 3000 * (3000 ^ 3000) ^ 2 = 3000 ^ 6001 := by
  sorry

end NUMINAMATH_GPT_exponentiation_calculation_l1285_128563


namespace NUMINAMATH_GPT_scheduled_conference_games_total_l1285_128507

def number_of_teams_in_A := 7
def number_of_teams_in_B := 5
def games_within_division (n : Nat) : Nat := n * (n - 1)
def interdivision_games := 7 * 5
def rivalry_games := 7

theorem scheduled_conference_games_total : 
  let games_A := games_within_division number_of_teams_in_A
  let games_B := games_within_division number_of_teams_in_B
  let total_games := games_A + games_B + interdivision_games + rivalry_games
  total_games = 104 :=
by
  sorry

end NUMINAMATH_GPT_scheduled_conference_games_total_l1285_128507


namespace NUMINAMATH_GPT_gcd_a2_14a_49_a_7_l1285_128580

theorem gcd_a2_14a_49_a_7 (a : ℤ) (k : ℤ) (h : a = 2100 * k) :
  Int.gcd (a^2 + 14*a + 49) (a + 7) = 7 := 
by
  sorry

end NUMINAMATH_GPT_gcd_a2_14a_49_a_7_l1285_128580


namespace NUMINAMATH_GPT_sum_of_acute_angles_l1285_128536

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π / 2)
variable (hβ : 0 < β ∧ β < π / 2)
variable (hcosα : Real.cos α = 1 / Real.sqrt 10)
variable (hcosβ : Real.cos β = 1 / Real.sqrt 5)

theorem sum_of_acute_angles :
  α + β = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_GPT_sum_of_acute_angles_l1285_128536


namespace NUMINAMATH_GPT_log_relationship_l1285_128591

theorem log_relationship (a b : ℝ) (x : ℝ) (h₁ : 6 * (Real.log (x) / Real.log (a)) ^ 2 + 5 * (Real.log (x) / Real.log (b)) ^ 2 = 12 * (Real.log (x) ^ 2) / (Real.log (a) * Real.log (b))) :
  a = b^(5/3) ∨ a = b^(3/5) := by
  sorry

end NUMINAMATH_GPT_log_relationship_l1285_128591


namespace NUMINAMATH_GPT_strap_pieces_l1285_128555

/-
  Given the conditions:
  1. The sum of the lengths of the two straps is 64 cm.
  2. The longer strap is 48 cm longer than the shorter strap.
  
  Prove that the number of pieces of strap that equal the length of the shorter strap 
  that can be cut from the longer strap is 7.
-/

theorem strap_pieces (S L : ℕ) (h1 : S + L = 64) (h2 : L = S + 48) :
  L / S = 7 :=
by
  sorry

end NUMINAMATH_GPT_strap_pieces_l1285_128555


namespace NUMINAMATH_GPT_hyperbola_asymptote_eq_l1285_128575

-- Define the given hyperbola equation and its asymptote
def hyperbola_eq (a : ℝ) : Prop :=
  ∃ x y : ℝ, (x^2 / a^2) - (y^2 / 4) = 1

def asymptote_eq (a : ℝ) : Prop :=
  ∀ x : ℝ, ∃ y : ℝ, y = (1/2) * x

-- State the main theorem
theorem hyperbola_asymptote_eq :
  (∃ a : ℝ, hyperbola_eq a ∧ asymptote_eq a) →
  (∃ x y : ℝ, (x^2 / 16) - (y^2 / 4) = 1) := 
by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptote_eq_l1285_128575


namespace NUMINAMATH_GPT_eggs_in_basket_l1285_128567

theorem eggs_in_basket (x : ℕ) (h₁ : 600 / x + 1 = 600 / (x - 20)) : x = 120 :=
sorry

end NUMINAMATH_GPT_eggs_in_basket_l1285_128567


namespace NUMINAMATH_GPT_total_amount_received_is_1465_l1285_128553

-- defining the conditions
def principal_1 : ℝ := 4000
def principal_2 : ℝ := 8200
def rate_1 : ℝ := 0.11
def rate_2 : ℝ := rate_1 + 0.015

-- defining the interest from each account
def interest_1 := principal_1 * rate_1
def interest_2 := principal_2 * rate_2

-- stating the total amount received
def total_received := interest_1 + interest_2

-- proving the total amount received
theorem total_amount_received_is_1465 : total_received = 1465 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_total_amount_received_is_1465_l1285_128553


namespace NUMINAMATH_GPT_apples_handed_out_l1285_128572

theorem apples_handed_out 
  (initial_apples : ℕ)
  (pies_made : ℕ)
  (apples_per_pie : ℕ)
  (H : initial_apples = 50)
  (H1 : pies_made = 9)
  (H2 : apples_per_pie = 5) :
  initial_apples - (pies_made * apples_per_pie) = 5 := 
by
  sorry

end NUMINAMATH_GPT_apples_handed_out_l1285_128572


namespace NUMINAMATH_GPT_solve_for_x_l1285_128565

theorem solve_for_x :
  ∃ x : ℤ, (225 - 4209520 / ((1000795 + (250 + x) * 50) / 27)) = 113 ∧ x = 40 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1285_128565


namespace NUMINAMATH_GPT_derivative_f_l1285_128562

noncomputable def f (x : ℝ) := x * Real.cos x - Real.sin x

theorem derivative_f :
  ∀ x : ℝ, deriv f x = -x * Real.sin x :=
by
  sorry

end NUMINAMATH_GPT_derivative_f_l1285_128562


namespace NUMINAMATH_GPT_product_divisible_by_60_l1285_128531

open Nat

theorem product_divisible_by_60 (S : Finset ℕ) (h_card : S.card = 10) (h_sum : S.sum id = 62) :
  60 ∣ S.prod id :=
  sorry

end NUMINAMATH_GPT_product_divisible_by_60_l1285_128531


namespace NUMINAMATH_GPT_simplify_expression_l1285_128542

theorem simplify_expression (x : ℝ) : 
  (3*x - 4)*(2*x + 9) - (x + 6)*(3*x + 2) = 3*x^2 - x - 48 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1285_128542


namespace NUMINAMATH_GPT_characterization_of_M_l1285_128518

noncomputable def M : Set ℂ := {z : ℂ | (z - 1) ^ 2 = Complex.abs (z - 1) ^ 2}

theorem characterization_of_M : M = {z : ℂ | ∃ r : ℝ, z = r} :=
by
  sorry

end NUMINAMATH_GPT_characterization_of_M_l1285_128518


namespace NUMINAMATH_GPT_water_added_l1285_128513

theorem water_added (capacity : ℝ) (percentage_initial : ℝ) (percentage_final : ℝ) :
  capacity = 120 →
  percentage_initial = 0.30 →
  percentage_final = 0.75 →
  ((percentage_final * capacity) - (percentage_initial * capacity)) = 54 :=
by intros
   sorry

end NUMINAMATH_GPT_water_added_l1285_128513


namespace NUMINAMATH_GPT_totalOwlsOnFence_l1285_128514

-- Define the conditions given in the problem
def initialOwls : Nat := 3
def joinedOwls : Nat := 2

-- Define the total number of owls
def totalOwls : Nat := initialOwls + joinedOwls

-- State the theorem we want to prove
theorem totalOwlsOnFence : totalOwls = 5 := by
  sorry

end NUMINAMATH_GPT_totalOwlsOnFence_l1285_128514


namespace NUMINAMATH_GPT_units_digit_of_7_pow_3_l1285_128582

theorem units_digit_of_7_pow_3 : (7 ^ 3) % 10 = 3 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_of_7_pow_3_l1285_128582


namespace NUMINAMATH_GPT_side_length_of_S2_l1285_128547

variable (r s : ℝ)

theorem side_length_of_S2 (h1 : 2 * r + s = 2100) (h2 : 2 * r + 3 * s = 3400) : s = 650 := by
  sorry

end NUMINAMATH_GPT_side_length_of_S2_l1285_128547


namespace NUMINAMATH_GPT_evaluate_at_2_l1285_128529

-- Define the polynomial function using Lean
def f (x : ℝ) : ℝ := 3 * x^3 + 2 * x^2 + x + 1

-- State the theorem that f(2) evaluates to 35 using Horner's method
theorem evaluate_at_2 : f 2 = 35 := by
  sorry

end NUMINAMATH_GPT_evaluate_at_2_l1285_128529


namespace NUMINAMATH_GPT_percent_unionized_men_is_70_l1285_128584

open Real

def total_employees : ℝ := 100
def percent_men : ℝ := 0.5
def percent_unionized : ℝ := 0.6
def percent_women_nonunion : ℝ := 0.8
def percent_men_nonunion : ℝ := 0.2

def num_men := total_employees * percent_men
def num_unionized := total_employees * percent_unionized
def num_nonunion := total_employees - num_unionized
def num_men_nonunion := num_nonunion * percent_men_nonunion
def num_men_unionized := num_men - num_men_nonunion

theorem percent_unionized_men_is_70 :
  (num_men_unionized / num_unionized) * 100 = 70 := by
  sorry

end NUMINAMATH_GPT_percent_unionized_men_is_70_l1285_128584


namespace NUMINAMATH_GPT_CE_squared_plus_DE_squared_proof_l1285_128510

noncomputable def CE_squared_plus_DE_squared (radius : ℝ) (diameter : ℝ) (BE : ℝ) (angle_AEC : ℝ) : ℝ :=
  if radius = 10 ∧ diameter = 20 ∧ BE = 4 ∧ angle_AEC = 30 then 200 else sorry

theorem CE_squared_plus_DE_squared_proof : CE_squared_plus_DE_squared 10 20 4 30 = 200 := by
  sorry

end NUMINAMATH_GPT_CE_squared_plus_DE_squared_proof_l1285_128510


namespace NUMINAMATH_GPT_length_PQ_l1285_128522

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

noncomputable def distance (P Q : Point3D) : ℝ :=
  Real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2 + (P.z - Q.z) ^ 2)

def P : Point3D := { x := 3, y := 4, z := 5 }

def Q : Point3D := { x := 3, y := 4, z := 0 }

theorem length_PQ : distance P Q = 5 :=
by
  sorry

end NUMINAMATH_GPT_length_PQ_l1285_128522


namespace NUMINAMATH_GPT_polygon_sides_eq_14_l1285_128586

def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem polygon_sides_eq_14 (n : ℕ) (h : n + num_diagonals n = 77) : n = 14 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_eq_14_l1285_128586


namespace NUMINAMATH_GPT_sugar_fill_count_l1285_128546

noncomputable def sugar_needed_for_one_batch : ℚ := 3 + 1/2
noncomputable def total_batches : ℕ := 2
noncomputable def cup_capacity : ℚ := 1/3
noncomputable def total_sugar_needed : ℚ := total_batches * sugar_needed_for_one_batch

theorem sugar_fill_count : (total_sugar_needed / cup_capacity) = 21 :=
by
  -- Assuming necessary preliminary steps already defined, we just check the equality directly
  sorry

end NUMINAMATH_GPT_sugar_fill_count_l1285_128546


namespace NUMINAMATH_GPT_at_most_one_negative_l1285_128585

theorem at_most_one_negative (a b c : ℝ) (h1 : a + b + c ≥ 0) (h2 : abc ≤ 0) : 
  (a < 0 ∧ b >= 0 ∧ c >= 0) ∨ (a >= 0 ∧ b < 0 ∧ c >= 0) ∨ (a >= 0 ∧ b >= 0 ∧ c < 0) ∨ 
  (a >= 0 ∧ b >= 0 ∧ c >= 0) :=
sorry

end NUMINAMATH_GPT_at_most_one_negative_l1285_128585


namespace NUMINAMATH_GPT_polygon_interior_exterior_relation_l1285_128538

theorem polygon_interior_exterior_relation :
  ∃ (n : ℕ), (n > 2) ∧ ((n - 2) * 180 = 4 * 360) ∧ n = 10 :=
by
  sorry

end NUMINAMATH_GPT_polygon_interior_exterior_relation_l1285_128538


namespace NUMINAMATH_GPT_sum_of_three_numbers_is_98_l1285_128500

variable (A B C : ℕ) (h_ratio1 : A = 2 * (B / 3)) (h_ratio2 : B = 30) (h_ratio3 : B = 5 * (C / 8))

theorem sum_of_three_numbers_is_98 : A + B + C = 98 := by
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_is_98_l1285_128500


namespace NUMINAMATH_GPT_critical_temperature_of_water_l1285_128570

/--
Given the following conditions:
1. The temperature at which solid, liquid, and gaseous water coexist is the triple point.
2. The temperature at which water vapor condenses is the condensation point.
3. The maximum temperature at which liquid water can exist.
4. The minimum temperature at which water vapor can exist.

Prove that the critical temperature of water is the maximum temperature at which liquid water can exist.
-/
theorem critical_temperature_of_water :
    ∀ (triple_point condensation_point maximum_liquid_temp minimum_vapor_temp critical_temp : ℝ), 
    (critical_temp = maximum_liquid_temp) ↔
    ((critical_temp ≠ triple_point) ∧ (critical_temp ≠ condensation_point) ∧ (critical_temp ≠ minimum_vapor_temp)) := 
  sorry

end NUMINAMATH_GPT_critical_temperature_of_water_l1285_128570


namespace NUMINAMATH_GPT_regular_seminar_fee_l1285_128587

-- Define the main problem statement
theorem regular_seminar_fee 
  (F : ℝ) 
  (discount_per_teacher : ℝ) 
  (number_of_teachers : ℕ)
  (food_allowance_per_teacher : ℝ)
  (total_spent : ℝ) :
  discount_per_teacher = 0.95 * F →
  number_of_teachers = 10 →
  food_allowance_per_teacher = 10 →
  total_spent = 1525 →
  (number_of_teachers * discount_per_teacher + number_of_teachers * food_allowance_per_teacher = total_spent) →
  F = 150 := 
  by sorry

end NUMINAMATH_GPT_regular_seminar_fee_l1285_128587


namespace NUMINAMATH_GPT_even_decreasing_function_l1285_128581

noncomputable def f : ℝ → ℝ := sorry

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x

theorem even_decreasing_function :
  is_even f →
  is_decreasing_on_nonneg f →
  f 1 > f (-2) ∧ f (-2) > f 3 :=
by
  sorry

end NUMINAMATH_GPT_even_decreasing_function_l1285_128581


namespace NUMINAMATH_GPT_depth_of_channel_l1285_128508

theorem depth_of_channel (h : ℝ) 
  (top_width : ℝ := 12) (bottom_width : ℝ := 6) (area : ℝ := 630) :
  1 / 2 * (top_width + bottom_width) * h = area → h = 70 :=
sorry

end NUMINAMATH_GPT_depth_of_channel_l1285_128508


namespace NUMINAMATH_GPT_larger_number_is_37_point_435_l1285_128592

theorem larger_number_is_37_point_435 (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 96) (h3 : x > y) : x = 37.435 :=
by
  sorry

end NUMINAMATH_GPT_larger_number_is_37_point_435_l1285_128592


namespace NUMINAMATH_GPT_multiple_of_a_age_l1285_128564

theorem multiple_of_a_age (A B M : ℝ) (h1 : A = B + 5) (h2 : A + B = 13) (h3 : M * (A + 7) = 4 * (B + 7)) : M = 2.75 :=
sorry

end NUMINAMATH_GPT_multiple_of_a_age_l1285_128564


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_eq_21_l1285_128577

theorem sum_of_squares_of_roots_eq_21 (a : ℝ) :
  (∃ x1 x2 : ℝ, x1^2 + x2^2 = 21 ∧ x1 + x2 = -a ∧ x1 * x2 = 2*a) ↔ a = -3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_eq_21_l1285_128577


namespace NUMINAMATH_GPT_minimum_reciprocal_sum_l1285_128548

theorem minimum_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 1) : 
  (∃ z : ℝ, (∀ x y : ℝ, 0 < x → 0 < y → x + 2 * y = 1 → z ≤ (1 / x + 2 / y)) ∧ z = 35 / 6) :=
  sorry

end NUMINAMATH_GPT_minimum_reciprocal_sum_l1285_128548


namespace NUMINAMATH_GPT_number_of_single_windows_upstairs_l1285_128504

theorem number_of_single_windows_upstairs :
  ∀ (num_double_windows_downstairs : ℕ)
    (glass_panels_per_double_window : ℕ)
    (num_single_windows_upstairs : ℕ)
    (glass_panels_per_single_window : ℕ)
    (total_glass_panels : ℕ),
  num_double_windows_downstairs = 6 →
  glass_panels_per_double_window = 4 →
  glass_panels_per_single_window = 4 →
  total_glass_panels = 80 →
  num_single_windows_upstairs = (total_glass_panels - (num_double_windows_downstairs * glass_panels_per_double_window)) / glass_panels_per_single_window →
  num_single_windows_upstairs = 14 :=
by
  intros
  sorry

end NUMINAMATH_GPT_number_of_single_windows_upstairs_l1285_128504


namespace NUMINAMATH_GPT_larry_stickers_l1285_128578

theorem larry_stickers (initial_stickers : ℕ) (lost_stickers : ℕ) (final_stickers : ℕ) 
  (initial_eq_93 : initial_stickers = 93) 
  (lost_eq_6 : lost_stickers = 6) 
  (final_eq : final_stickers = initial_stickers - lost_stickers) : 
  final_stickers = 87 := 
  by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_larry_stickers_l1285_128578


namespace NUMINAMATH_GPT_ship_length_in_steps_l1285_128566

theorem ship_length_in_steps (E S L : ℝ) (H1 : L + 300 * S = 300 * E) (H2 : L - 60 * S = 60 * E) :
  L = 100 * E :=
by sorry

end NUMINAMATH_GPT_ship_length_in_steps_l1285_128566


namespace NUMINAMATH_GPT_inequality_proof_l1285_128517

theorem inequality_proof (a b : ℝ) (h1 : b < 0) (h2 : 0 < a) : a - b > a + b :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1285_128517


namespace NUMINAMATH_GPT_eq_satisfied_in_entire_space_l1285_128520

theorem eq_satisfied_in_entire_space (x y z : ℝ) : 
  (x + y + z)^2 = x^2 + y^2 + z^2 ↔ xy + xz + yz = 0 :=
by
  sorry

end NUMINAMATH_GPT_eq_satisfied_in_entire_space_l1285_128520


namespace NUMINAMATH_GPT_geometric_sequence_first_term_l1285_128509

variable (a y z : ℕ)
variable (r : ℕ)
variable (h₁ : 16 = a * r^2)
variable (h₂ : 128 = a * r^4)

theorem geometric_sequence_first_term 
  (h₃ : r = 2) : a = 4 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_first_term_l1285_128509


namespace NUMINAMATH_GPT_find_boxes_l1285_128551

variable (John Jules Joseph Stan : ℕ)

-- Conditions
axiom h1 : John = 30
axiom h2 : John = 6 * Jules / 5 -- Equivalent to John having 20% more boxes than Jules
axiom h3 : Jules = Joseph + 5
axiom h4 : Joseph = Stan / 5 -- Equivalent to Joseph having 80% fewer boxes than Stan

-- Theorem to prove
theorem find_boxes (h1 : John = 30) (h2 : John = 6 * Jules / 5) (h3 : Jules = Joseph + 5) (h4 : Joseph = Stan / 5) : Stan = 100 :=
sorry

end NUMINAMATH_GPT_find_boxes_l1285_128551


namespace NUMINAMATH_GPT_chlorine_discount_l1285_128574

theorem chlorine_discount
  (cost_chlorine : ℕ)
  (cost_soap : ℕ)
  (num_chlorine : ℕ)
  (num_soap : ℕ)
  (discount_soap : ℤ)
  (total_savings : ℤ)
  (price_chlorine : ℤ)
  (price_soap_after_discount : ℤ)
  (total_price_before_discount : ℤ)
  (total_price_after_discount : ℤ)
  (goal_discount : ℤ) :
  cost_chlorine = 10 →
  cost_soap = 16 →
  num_chlorine = 3 →
  num_soap = 5 →
  discount_soap = 25 →
  total_savings = 26 →
  price_soap_after_discount = (1 - (discount_soap / 100)) * 16 →
  total_price_before_discount = (num_chlorine * cost_chlorine) + (num_soap * cost_soap) →
  total_price_after_discount = (num_chlorine * ((100 - goal_discount) / 100) * cost_chlorine) + (num_soap * 12) →
  total_price_before_discount - total_price_after_discount = total_savings →
  goal_discount = 20 :=
by
  intros
  sorry

end NUMINAMATH_GPT_chlorine_discount_l1285_128574


namespace NUMINAMATH_GPT_day_of_100th_day_of_2005_l1285_128599

-- Define the days of the week
inductive Weekday
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday
deriving DecidableEq, Repr

open Weekday

-- Define a function to add days to a given weekday
def add_days (d: Weekday) (n: ℕ) : Weekday :=
  match d with
  | Sunday => [Sunday, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday].get? (n % 7) |>.getD Sunday
  | Monday => [Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday].get? (n % 7) |>.getD Monday
  | Tuesday => [Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday, Monday].get? (n % 7) |>.getD Tuesday
  | Wednesday => [Wednesday, Thursday, Friday, Saturday, Sunday, Monday, Tuesday].get? (n % 7) |>.getD Wednesday
  | Thursday => [Thursday, Friday, Saturday, Sunday, Monday, Tuesday, Wednesday].get? (n % 7) |>.getD Thursday
  | Friday => [Friday, Saturday, Sunday, Monday, Tuesday, Wednesday, Thursday].get? (n % 7) |>.getD Friday
  | Saturday => [Saturday, Sunday, Monday, Tuesday, Wednesday, Thursday, Friday].get? (n % 7) |>.getD Saturday

-- State the theorem
theorem day_of_100th_day_of_2005 :
  add_days Tuesday 55 = Monday :=
by sorry

end NUMINAMATH_GPT_day_of_100th_day_of_2005_l1285_128599


namespace NUMINAMATH_GPT_neither_A_B_C_prob_correct_l1285_128537

noncomputable def P (A B C : Prop) : Prop :=
  let P_A := 0.25
  let P_B := 0.35
  let P_C := 0.40
  let P_A_and_B := 0.10
  let P_A_and_C := 0.15
  let P_B_and_C := 0.20
  let P_A_and_B_and_C := 0.05
  
  let P_A_or_B_or_C := 
    P_A + P_B + P_C - P_A_and_B - P_A_and_C - P_B_and_C + P_A_and_B_and_C
  
  let P_neither_A_nor_B_nor_C := 1 - P_A_or_B_or_C
    
  P_neither_A_nor_B_nor_C = 0.45

theorem neither_A_B_C_prob_correct :
  P A B C := by
  sorry

end NUMINAMATH_GPT_neither_A_B_C_prob_correct_l1285_128537


namespace NUMINAMATH_GPT_find_starting_number_l1285_128552

-- Define that there are 15 even integers between a starting number and 40
def even_integers_range (n : ℕ) : Prop :=
  ∃ k : ℕ, (1 ≤ k) ∧ (k = 15) ∧ (n + 2*(k-1) = 40)

-- Proof statement
theorem find_starting_number : ∃ n : ℕ, even_integers_range n ∧ n = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_starting_number_l1285_128552


namespace NUMINAMATH_GPT_draw_13_cards_no_straight_flush_l1285_128594

theorem draw_13_cards_no_straight_flush :
  let deck_size := 52
  let suit_count := 4
  let rank_count := 13
  let non_straight_flush_draws (n : ℕ) := 3^n - 3
  n = rank_count →
  ∀ (draw : ℕ), draw = non_straight_flush_draws n :=
by
-- Proof would be here
sorry

end NUMINAMATH_GPT_draw_13_cards_no_straight_flush_l1285_128594


namespace NUMINAMATH_GPT_three_friends_at_least_50_mushrooms_l1285_128597

theorem three_friends_at_least_50_mushrooms (a : Fin 7 → ℕ) (h_sum : (Finset.univ.sum a) = 100) (h_different : Function.Injective a) :
  ∃ i j k : Fin 7, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ (a i + a j + a k) ≥ 50 :=
by
  sorry

end NUMINAMATH_GPT_three_friends_at_least_50_mushrooms_l1285_128597


namespace NUMINAMATH_GPT_Megan_pictures_left_l1285_128506

theorem Megan_pictures_left (zoo_pictures museum_pictures deleted_pictures : ℕ) 
  (h1 : zoo_pictures = 15) 
  (h2 : museum_pictures = 18) 
  (h3 : deleted_pictures = 31) : 
  zoo_pictures + museum_pictures - deleted_pictures = 2 := 
by
  sorry

end NUMINAMATH_GPT_Megan_pictures_left_l1285_128506


namespace NUMINAMATH_GPT_total_rattlesnakes_l1285_128583

-- Definitions based on the problem's conditions
def total_snakes : ℕ := 200
def boa_constrictors : ℕ := 40
def pythons : ℕ := 3 * boa_constrictors
def other_snakes : ℕ := total_snakes - (pythons + boa_constrictors)

-- Statement to be proved
theorem total_rattlesnakes : other_snakes = 40 := 
by 
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_total_rattlesnakes_l1285_128583


namespace NUMINAMATH_GPT_factory_toys_production_each_day_l1285_128554

theorem factory_toys_production_each_day 
  (weekly_production : ℕ)
  (days_worked_per_week : ℕ)
  (h1 : weekly_production = 4560)
  (h2 : days_worked_per_week = 4) : 
  (weekly_production / days_worked_per_week) = 1140 :=
  sorry

end NUMINAMATH_GPT_factory_toys_production_each_day_l1285_128554


namespace NUMINAMATH_GPT_mr_green_garden_yield_l1285_128573

noncomputable def garden_yield (steps_length steps_width step_length yield_per_sqft : ℝ) : ℝ :=
  let length_ft := steps_length * step_length
  let width_ft := steps_width * step_length
  let area := length_ft * width_ft
  area * yield_per_sqft

theorem mr_green_garden_yield :
  garden_yield 18 25 2.5 0.5 = 1406.25 :=
by
  sorry

end NUMINAMATH_GPT_mr_green_garden_yield_l1285_128573


namespace NUMINAMATH_GPT_shorter_piece_length_l1285_128596

theorem shorter_piece_length :
  ∃ (x : ℝ), x + 2 * x = 69 ∧ x = 23 :=
by
  sorry

end NUMINAMATH_GPT_shorter_piece_length_l1285_128596


namespace NUMINAMATH_GPT_ratio_long_side_brush_width_l1285_128557

theorem ratio_long_side_brush_width 
  (l : ℝ) (w : ℝ) (d : ℝ) (total_area : ℝ) (painted_area : ℝ) (b : ℝ) 
  (h1 : l = 9)
  (h2 : w = 4)
  (h3 : total_area = l * w)
  (h4 : total_area / 3 = painted_area)
  (h5 : d = Real.sqrt (l^2 + w^2))
  (h6 : d * b = painted_area) :
  l / b = (3 * Real.sqrt 97) / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_long_side_brush_width_l1285_128557


namespace NUMINAMATH_GPT_median_production_l1285_128527

def production_data : List ℕ := [5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10]

def median (l : List ℕ) : ℕ :=
  if l.length % 2 = 1 then
    l.nthLe (l.length / 2) sorry
  else
    let m := l.length / 2
    (l.nthLe (m - 1) sorry + l.nthLe m sorry) / 2

theorem median_production :
  median (production_data) = 8 :=
by
  sorry

end NUMINAMATH_GPT_median_production_l1285_128527


namespace NUMINAMATH_GPT_smallest_two_digit_prime_with_conditions_l1285_128501

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

noncomputable def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ is_prime n

theorem smallest_two_digit_prime_with_conditions :
  ∃ p : ℕ, is_prime p ∧ 10 ≤ p ∧ p < 100 ∧ (p / 10 = 3) ∧ is_composite (((p % 10) * 10) + (p / 10) + 5) ∧ p = 31 :=
by
  sorry

end NUMINAMATH_GPT_smallest_two_digit_prime_with_conditions_l1285_128501


namespace NUMINAMATH_GPT_order_of_numbers_l1285_128558

def base16_to_dec (s : String) : ℕ := sorry
def base6_to_dec (s : String) : ℕ := sorry
def base4_to_dec (s : String) : ℕ := sorry
def base2_to_dec (s : String) : ℕ := sorry

theorem order_of_numbers:
  let a := base16_to_dec "3E"
  let b := base6_to_dec "210"
  let c := base4_to_dec "1000"
  let d := base2_to_dec "111011"
  a = 62 ∧ b = 78 ∧ c = 64 ∧ d = 59 →
  b > c ∧ c > a ∧ a > d :=
by
  intros
  sorry

end NUMINAMATH_GPT_order_of_numbers_l1285_128558


namespace NUMINAMATH_GPT_oil_spending_l1285_128511

-- Define the original price per kg of oil
def original_price (P : ℝ) := P

-- Define the reduced price per kg of oil
def reduced_price (P : ℝ) := 0.75 * P

-- Define the reduced price as Rs. 60
def reduced_price_fixed := 60

-- State the condition that reduced price enables 5 kgs more oil
def extra_kg := 5

-- The amount of money spent by housewife at reduced price which is to be proven as Rs. 1200
def amount_spent (M : ℝ) := M

-- Define the problem to prove in Lean 4
theorem oil_spending (P X : ℝ) (h1 : reduced_price P = reduced_price_fixed) (h2 : X * original_price P = (X + extra_kg) * reduced_price_fixed) : amount_spent ((X + extra_kg) * reduced_price_fixed) = 1200 :=
  sorry

end NUMINAMATH_GPT_oil_spending_l1285_128511


namespace NUMINAMATH_GPT_operation_ab_equals_nine_l1285_128535

variable (a b : ℝ)

def operation (x y : ℝ) : ℝ := a * x + b * y - 1

theorem operation_ab_equals_nine
  (h1 : operation a b 1 2 = 4)
  (h2 : operation a b (-2) 3 = 10)
  : a * b = 9 :=
by
  sorry

end NUMINAMATH_GPT_operation_ab_equals_nine_l1285_128535


namespace NUMINAMATH_GPT_consecutive_integers_l1285_128590

theorem consecutive_integers (a b c : ℝ)
  (h1 : ∃ k : ℤ, a + b = k ∧ b + c = k + 1 ∧ c + a = k + 2)
  (h2 : ∃ k : ℤ, b + c = 2 * k + 1) :
  ∃ n : ℤ, a = n + 2 ∧ b = n + 1 ∧ c = n := 
sorry

end NUMINAMATH_GPT_consecutive_integers_l1285_128590


namespace NUMINAMATH_GPT_trigonometric_identity_l1285_128534

-- Definition for the given condition
def tan_alpha (α : ℝ) : Prop := Real.tan α = 2

-- The proof goal
theorem trigonometric_identity (α : ℝ) (h : tan_alpha α) : 
  Real.cos (π + α) * Real.cos (π / 2 + α) = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1285_128534


namespace NUMINAMATH_GPT_profit_at_15_is_correct_l1285_128589

noncomputable def profit (x : ℝ) : ℝ := (2 * x - 20) * (40 - x)

theorem profit_at_15_is_correct :
  profit 15 = 1250 := by
  sorry

end NUMINAMATH_GPT_profit_at_15_is_correct_l1285_128589


namespace NUMINAMATH_GPT_find_f_of_2_l1285_128540

theorem find_f_of_2 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (2 * x) = 4 * x - 1) : f 2 = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_f_of_2_l1285_128540


namespace NUMINAMATH_GPT_fruits_left_l1285_128524

theorem fruits_left (plums guavas apples given : ℕ) (h1 : plums = 16) (h2 : guavas = 18) (h3 : apples = 21) (h4 : given = 40) : 
  (plums + guavas + apples - given = 15) :=
by
  sorry

end NUMINAMATH_GPT_fruits_left_l1285_128524


namespace NUMINAMATH_GPT_cosine_product_l1285_128576

-- Definitions for the conditions of the problem
variable (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
variables (circle : Set A) (inscribed_pentagon : Set A)
variables (AB BC CD DE AE : ℝ) (cosB cosACE : ℝ)

-- Conditions
axiom pentagon_inscribed_in_circle : inscribed_pentagon ⊆ circle
axiom AB_eq_3 : AB = 3
axiom BC_eq_3 : BC = 3
axiom CD_eq_3 : CD = 3
axiom DE_eq_3 : DE = 3
axiom AE_eq_2 : AE = 2

-- Theorem statement
theorem cosine_product :
  (1 - cosB) * (1 - cosACE) = (1 / 9) := 
sorry

end NUMINAMATH_GPT_cosine_product_l1285_128576


namespace NUMINAMATH_GPT_maximize_profit_l1285_128502

/-- A car sales company purchased a total of 130 vehicles of models A and B, 
with x vehicles of model A purchased. The profit y is defined by selling 
prices and factory prices of both models. -/
def total_profit (x : ℕ) : ℝ := -2 * x + 520

theorem maximize_profit :
  ∃ x : ℕ, (130 - x ≤ 2 * x) ∧ (total_profit x = 432) ∧ (∀ y : ℕ, (130 - y ≤ 2 * y) → (total_profit y ≤ 432)) :=
by {
  sorry
}

end NUMINAMATH_GPT_maximize_profit_l1285_128502


namespace NUMINAMATH_GPT_time_to_cross_first_platform_l1285_128525

variable (length_first_platform : ℝ)
variable (length_second_platform : ℝ)
variable (time_to_cross_second_platform : ℝ)
variable (length_of_train : ℝ)

theorem time_to_cross_first_platform :
  length_first_platform = 160 →
  length_second_platform = 250 →
  time_to_cross_second_platform = 20 →
  length_of_train = 110 →
  (270 / (360 / 20) = 15) := 
by
  intro h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_time_to_cross_first_platform_l1285_128525


namespace NUMINAMATH_GPT_nancy_threw_out_2_carrots_l1285_128559

theorem nancy_threw_out_2_carrots :
  ∀ (x : ℕ), 12 - x + 21 = 31 → x = 2 :=
by
  sorry

end NUMINAMATH_GPT_nancy_threw_out_2_carrots_l1285_128559


namespace NUMINAMATH_GPT_real_part_of_z_l1285_128598

theorem real_part_of_z (z : ℂ) (h : ∃ (r : ℝ), z^2 + z = r) : z.re = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_real_part_of_z_l1285_128598


namespace NUMINAMATH_GPT_running_time_around_pentagon_l1285_128519

theorem running_time_around_pentagon :
  let l₁ := 40
  let l₂ := 50
  let l₃ := 60
  let l₄ := 45
  let l₅ := 55
  let v₁ := 9 * 1000 / 60
  let v₂ := 8 * 1000 / 60
  let v₃ := 7 * 1000 / 60
  let v₄ := 6 * 1000 / 60
  let v₅ := 5 * 1000 / 60
  let t₁ := l₁ / v₁
  let t₂ := l₂ / v₂
  let t₃ := l₃ / v₃
  let t₄ := l₄ / v₄
  let t₅ := l₅ / v₅
  t₁ + t₂ + t₃ + t₄ + t₅ = 2.266 := by
    sorry

end NUMINAMATH_GPT_running_time_around_pentagon_l1285_128519


namespace NUMINAMATH_GPT_cannot_be_sum_of_consecutive_nat_iff_power_of_two_l1285_128526

theorem cannot_be_sum_of_consecutive_nat_iff_power_of_two (n : ℕ) : 
  (∀ a b : ℕ, n ≠ (b - a + 1) * (a + b) / 2) ↔ (∃ k : ℕ, n = 2 ^ k) := by
  sorry

end NUMINAMATH_GPT_cannot_be_sum_of_consecutive_nat_iff_power_of_two_l1285_128526


namespace NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l1285_128528

theorem common_difference_of_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) 
  (h1 : ∀ n, a n = a 0 + n * d) (h2 : a 5 = 10) (h3 : a 10 = -5) : d = -3 := 
by 
  sorry

end NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l1285_128528


namespace NUMINAMATH_GPT_sum_of_reciprocals_l1285_128579

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) : 
  1/x + 1/y = 3/8 := by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_l1285_128579


namespace NUMINAMATH_GPT_tanya_bought_11_pears_l1285_128571

variable (P : ℕ)

-- Define the given conditions about the number of different fruits Tanya bought
def apples : ℕ := 4
def pineapples : ℕ := 2
def basket_of_plums : ℕ := 1

-- Define the total number of fruits initially and the remaining fruits
def initial_fruit_total : ℕ := 18
def remaining_fruit_total : ℕ := 9
def half_fell_out_of_bag : ℕ := remaining_fruit_total * 2

-- The main theorem to prove
theorem tanya_bought_11_pears (h : P + apples + pineapples + basket_of_plums = initial_fruit_total) : P = 11 := by
  -- providing a placeholder for the proof
  sorry

end NUMINAMATH_GPT_tanya_bought_11_pears_l1285_128571


namespace NUMINAMATH_GPT_point_bisector_second_quadrant_l1285_128532

theorem point_bisector_second_quadrant (a : ℝ) : 
  (a < 0 ∧ 2 > 0) ∧ (2 = -a) → a = -2 :=
by sorry

end NUMINAMATH_GPT_point_bisector_second_quadrant_l1285_128532


namespace NUMINAMATH_GPT_num_k_values_lcm_l1285_128560

-- Define prime factorizations of given numbers
def nine_pow_nine := 3^18
def twelve_pow_twelve := 2^24 * 3^12
def eighteen_pow_eighteen := 2^18 * 3^36

-- Number of values of k making eighteen_pow_eighteen the LCM of nine_pow_nine, twelve_pow_twelve, and k
def number_of_k_values : ℕ := 
  19 -- Based on calculations from the proof

theorem num_k_values_lcm :
  ∀ (k : ℕ), eighteen_pow_eighteen = Nat.lcm (Nat.lcm nine_pow_nine twelve_pow_twelve) k → ∃ n, n = number_of_k_values :=
  sorry -- Add the proof later

end NUMINAMATH_GPT_num_k_values_lcm_l1285_128560


namespace NUMINAMATH_GPT_min_value_of_squares_l1285_128539

theorem min_value_of_squares (a b t : ℝ) (h : a + b = t) : (a^2 + b^2) ≥ t^2 / 2 := 
by
  sorry

end NUMINAMATH_GPT_min_value_of_squares_l1285_128539


namespace NUMINAMATH_GPT_isosceles_triangle_angle_B_l1285_128543

theorem isosceles_triangle_angle_B (A B C : ℝ)
  (h_triangle : (A + B + C = 180))
  (h_exterior_A : 180 - A = 110)
  (h_sum_angles : A + B + C = 180) :
  B = 70 ∨ B = 55 ∨ B = 40 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_angle_B_l1285_128543


namespace NUMINAMATH_GPT_false_statement_D_l1285_128530

theorem false_statement_D :
  ¬ (∀ {α β : ℝ}, α = β → (true → true → true → α = β ↔ α = β)) :=
by
  sorry

end NUMINAMATH_GPT_false_statement_D_l1285_128530


namespace NUMINAMATH_GPT_children_more_than_adults_l1285_128521

-- Conditions
def total_members : ℕ := 120
def adult_percentage : ℝ := 0.40
def child_percentage : ℝ := 1 - adult_percentage

-- Proof problem statement
theorem children_more_than_adults : 
  let number_of_adults := adult_percentage * total_members
  let number_of_children := child_percentage * total_members
  let difference := number_of_children - number_of_adults
  difference = 24 :=
by
  sorry

end NUMINAMATH_GPT_children_more_than_adults_l1285_128521


namespace NUMINAMATH_GPT_find_expression_l1285_128550

theorem find_expression (a b : ℝ) (h₁ : a - b = 5) (h₂ : a * b = 2) :
  a^2 - a * b + b^2 = 27 := 
by
  sorry

end NUMINAMATH_GPT_find_expression_l1285_128550


namespace NUMINAMATH_GPT_longest_segment_CD_l1285_128533

variables (A B C D : Type)
variables (angle_ABD angle_ADB angle_BDC angle_CBD : ℝ)

axiom angle_ABD_eq : angle_ABD = 30
axiom angle_ADB_eq : angle_ADB = 65
axiom angle_BDC_eq : angle_BDC = 60
axiom angle_CBD_eq : angle_CBD = 80

theorem longest_segment_CD
  (h_ABD : angle_ABD = 30)
  (h_ADB : angle_ADB = 65)
  (h_BDC : angle_BDC = 60)
  (h_CBD : angle_CBD = 80) : false :=
sorry

end NUMINAMATH_GPT_longest_segment_CD_l1285_128533


namespace NUMINAMATH_GPT_remainder_when_divided_by_multiple_of_10_l1285_128561

theorem remainder_when_divided_by_multiple_of_10 (N : ℕ) (hN : ∃ k : ℕ, N = 10 * k) (hrem : (19 ^ 19 + 19) % N = 18) : N = 10 := by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_multiple_of_10_l1285_128561


namespace NUMINAMATH_GPT_proof_b_greater_a_greater_c_l1285_128544

def a : ℤ := -2 * 3^2
def b : ℤ := (-2 * 3)^2
def c : ℤ := - (2 * 3)^2

theorem proof_b_greater_a_greater_c (ha : a = -18) (hb : b = 36) (hc : c = -36) : b > a ∧ a > c := 
by
  rw [ha, hb, hc]
  exact And.intro (by norm_num) (by norm_num)

end NUMINAMATH_GPT_proof_b_greater_a_greater_c_l1285_128544


namespace NUMINAMATH_GPT_jack_more_emails_morning_than_afternoon_l1285_128516

def emails_afternoon := 3
def emails_morning := 5

theorem jack_more_emails_morning_than_afternoon :
  emails_morning - emails_afternoon = 2 :=
by
  sorry

end NUMINAMATH_GPT_jack_more_emails_morning_than_afternoon_l1285_128516


namespace NUMINAMATH_GPT_veenapaniville_private_independent_district_A_l1285_128523

theorem veenapaniville_private_independent_district_A :
  let total_schools := 50
  let public_schools := 25
  let parochial_schools := 16
  let private_schools := 9
  let district_A_schools := 18
  let district_B_schools := 17
  let district_B_private := 2
  let remaining_schools := total_schools - district_A_schools - district_B_schools
  let each_kind_in_C := remaining_schools / 3
  let district_C_private := each_kind_in_C
  let district_A_private := private_schools - district_B_private - district_C_private
  district_A_private = 2 := by
  sorry

end NUMINAMATH_GPT_veenapaniville_private_independent_district_A_l1285_128523


namespace NUMINAMATH_GPT_sum_of_Ns_l1285_128515

theorem sum_of_Ns (N R : ℝ) (hN_nonzero : N ≠ 0) (h_eq : N - 3 * N^2 = R) : 
  ∃ N1 N2 : ℝ, N1 ≠ 0 ∧ N2 ≠ 0 ∧ 3 * N1^2 - N1 + R = 0 ∧ 3 * N2^2 - N2 + R = 0 ∧ (N1 + N2) = 1 / 3 :=
sorry

end NUMINAMATH_GPT_sum_of_Ns_l1285_128515


namespace NUMINAMATH_GPT_line_intersects_circle_l1285_128588

theorem line_intersects_circle (k : ℝ) :
  ∃ x y : ℝ, (kx - y - k +1 = 0) ∧ (x^2 + y^2 = 4) :=
sorry

end NUMINAMATH_GPT_line_intersects_circle_l1285_128588


namespace NUMINAMATH_GPT_prime_divides_product_of_divisors_l1285_128556

theorem prime_divides_product_of_divisors (p : ℕ) (n : ℕ) (a : Fin n → ℕ) 
(Hp : Nat.Prime p) (Hdiv : p ∣ (Finset.univ.prod a)) : 
∃ i : Fin n, p ∣ a i :=
sorry

end NUMINAMATH_GPT_prime_divides_product_of_divisors_l1285_128556


namespace NUMINAMATH_GPT_systems_on_second_street_l1285_128503

-- Definitions based on the conditions
def commission_per_system : ℕ := 25
def total_commission : ℕ := 175
def systems_on_first_street (S : ℕ) := S / 2
def systems_on_third_street : ℕ := 0
def systems_on_fourth_street : ℕ := 1

-- Question: How many security systems did Rodney sell on the second street?
theorem systems_on_second_street (S : ℕ) :
  S / 2 + S + 0 + 1 = total_commission / commission_per_system → S = 4 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_systems_on_second_street_l1285_128503


namespace NUMINAMATH_GPT_income_expenditure_ratio_l1285_128549

variable (I S E : ℕ)
variable (hI : I = 16000)
variable (hS : S = 3200)
variable (hExp : S = I - E)

theorem income_expenditure_ratio (I S E : ℕ) (hI : I = 16000) (hS : S = 3200) (hExp : S = I - E) : I / Nat.gcd I E = 5 ∧ E / Nat.gcd I E = 4 := by
  sorry

end NUMINAMATH_GPT_income_expenditure_ratio_l1285_128549


namespace NUMINAMATH_GPT_difference_of_squares_l1285_128569

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 8) : x^2 - y^2 = 160 :=
sorry

end NUMINAMATH_GPT_difference_of_squares_l1285_128569


namespace NUMINAMATH_GPT_avg_annual_growth_rate_optimal_selling_price_l1285_128512

theorem avg_annual_growth_rate (v2022 v2024 : ℕ) (x : ℝ) 
  (h1 : v2022 = 200000) 
  (h2 : v2024 = 288000)
  (h3: v2024 = v2022 * (1 + x)^2) :
  x = 0.2 :=
by
  sorry

theorem optimal_selling_price (cost : ℝ) (initial_price : ℝ) (initial_cups : ℕ) 
  (price_drop_effect : ℝ) (initial_profit : ℝ) (daily_profit : ℕ) (y : ℝ)
  (h1 : cost = 6)
  (h2 : initial_price = 25) 
  (h3 : initial_cups = 300)
  (h4 : price_drop_effect = 1)
  (h5 : initial_profit = 6300)
  (h6 : (y - cost) * (initial_cups + 30 * (initial_price - y)) = daily_profit) :
  y = 20 :=
by
  sorry

end NUMINAMATH_GPT_avg_annual_growth_rate_optimal_selling_price_l1285_128512


namespace NUMINAMATH_GPT_identity_true_for_any_abc_l1285_128593

theorem identity_true_for_any_abc : 
  ∀ (a b c : ℝ), (a + b) * (b + c) * (c + a) = (a + b + c) * (a * b + b * c + c * a) - a * b * c :=
by
  sorry

end NUMINAMATH_GPT_identity_true_for_any_abc_l1285_128593
