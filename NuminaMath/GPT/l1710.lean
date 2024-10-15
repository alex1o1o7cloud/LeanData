import Mathlib

namespace NUMINAMATH_GPT_sandy_final_position_and_distance_l1710_171030

-- Define the conditions as statements
def walked_south (distance : ℕ) := distance = 20
def turned_left_facing_east := true
def walked_east (distance : ℕ) := distance = 20
def turned_left_facing_north := true
def walked_north (distance : ℕ) := distance = 20
def turned_right_facing_east := true
def walked_east_again (distance : ℕ) := distance = 20

-- Final position computation as a proof statement
theorem sandy_final_position_and_distance :
  ∃ (d : ℕ) (dir : String), walked_south 20 → turned_left_facing_east → walked_east 20 →
  turned_left_facing_north → walked_north 20 →
  turned_right_facing_east → walked_east_again 20 ∧ d = 40 ∧ dir = "east" :=
by
  sorry

end NUMINAMATH_GPT_sandy_final_position_and_distance_l1710_171030


namespace NUMINAMATH_GPT_ab_value_l1710_171038

theorem ab_value (a b : ℚ) (h1 : 3 * a - 8 = 0) (h2 : b = 3) : a * b = 8 :=
by
  sorry

end NUMINAMATH_GPT_ab_value_l1710_171038


namespace NUMINAMATH_GPT_minimum_rectangles_needed_l1710_171099

def type1_corners := 12
def type2_corners := 12
def group_size := 3

theorem minimum_rectangles_needed (cover_type1: ℕ) (cover_type2: ℕ)
  (type1_corners coverable_by_one: ℕ) (type2_groups_num: ℕ) :
  type1_corners = 12 → type2_corners = 12 → type2_groups_num = 4 →
  group_size = 3 → cover_type1 + cover_type2 = 12 :=
by
  intros h1 h2 h3 h4 
  sorry

end NUMINAMATH_GPT_minimum_rectangles_needed_l1710_171099


namespace NUMINAMATH_GPT_fraction_non_throwers_left_handed_l1710_171063

theorem fraction_non_throwers_left_handed (total_players : ℕ) (num_throwers : ℕ) (total_right_handed : ℕ) (all_throwers_right_handed : ∀ x, x < num_throwers → true) (num_right_handed := total_right_handed - num_throwers) (non_throwers := total_players - num_throwers) (num_left_handed := non_throwers - num_right_handed) : 
    total_players = 70 → 
    num_throwers = 40 → 
    total_right_handed = 60 → 
    (∃ f: ℚ, f = num_left_handed / non_throwers ∧ f = 1/3) := 
by {
  sorry
}

end NUMINAMATH_GPT_fraction_non_throwers_left_handed_l1710_171063


namespace NUMINAMATH_GPT_count_valid_integers_1_to_999_l1710_171098

-- Define a function to count the valid integers
def count_valid_integers : Nat :=
  let digits := [1, 2, 6, 7, 9]
  let one_digit_count := 5
  let two_digit_count := 5 * 5
  let three_digit_count := 5 * 5 * 5
  one_digit_count + two_digit_count + three_digit_count

-- The theorem we want to prove
theorem count_valid_integers_1_to_999 : count_valid_integers = 155 := by
  sorry

end NUMINAMATH_GPT_count_valid_integers_1_to_999_l1710_171098


namespace NUMINAMATH_GPT_tire_miles_used_l1710_171092

theorem tire_miles_used (total_miles : ℕ) (number_of_tires : ℕ) (tires_in_use : ℕ)
  (h_total_miles : total_miles = 40000) (h_number_of_tires : number_of_tires = 6)
  (h_tires_in_use : tires_in_use = 4) : 
  (total_miles * tires_in_use) / number_of_tires = 26667 := 
by 
  sorry

end NUMINAMATH_GPT_tire_miles_used_l1710_171092


namespace NUMINAMATH_GPT_boys_and_girls_in_class_l1710_171051

theorem boys_and_girls_in_class (b g : ℕ) (h1 : b + g = 21) (h2 : 5 * b + 2 * g = 69) 
: b = 9 ∧ g = 12 := by
  sorry

end NUMINAMATH_GPT_boys_and_girls_in_class_l1710_171051


namespace NUMINAMATH_GPT_soccer_camp_afternoon_kids_l1710_171095

def num_kids_in_camp : ℕ := 2000
def fraction_going_to_soccer_camp : ℚ := 1 / 2
def fraction_going_to_soccer_camp_in_morning : ℚ := 1 / 4

noncomputable def num_kids_going_to_soccer_camp := num_kids_in_camp * fraction_going_to_soccer_camp
noncomputable def num_kids_going_to_soccer_camp_in_morning := num_kids_going_to_soccer_camp * fraction_going_to_soccer_camp_in_morning
noncomputable def num_kids_going_to_soccer_camp_in_afternoon := num_kids_going_to_soccer_camp - num_kids_going_to_soccer_camp_in_morning

theorem soccer_camp_afternoon_kids : num_kids_going_to_soccer_camp_in_afternoon = 750 :=
by
  sorry

end NUMINAMATH_GPT_soccer_camp_afternoon_kids_l1710_171095


namespace NUMINAMATH_GPT_price_per_glass_second_day_l1710_171048

theorem price_per_glass_second_day 
  (O W : ℕ)  -- O is the amount of orange juice used on each day, W is the amount of water used on the first day
  (V : ℕ)   -- V is the volume of one glass
  (P₁ : ℚ)  -- P₁ is the price per glass on the first day
  (P₂ : ℚ)  -- P₂ is the price per glass on the second day
  (h1 : W = O)  -- First day, water is equal to orange juice
  (h2 : V > 0)  -- Volume of one glass > 0
  (h3 : P₁ = 0.48)  -- Price per glass on the first day
  (h4 : (2 * O / V) * P₁ = (3 * O / V) * P₂)  -- Revenue's are the same
  : P₂ = 0.32 :=  -- Prove that price per glass on the second day is 0.32
by
  sorry

end NUMINAMATH_GPT_price_per_glass_second_day_l1710_171048


namespace NUMINAMATH_GPT_jenn_has_five_jars_l1710_171079

/-- Each jar can hold 160 quarters, the bike costs 180 dollars, 
    Jenn will have 20 dollars left over, 
    and a quarter is worth 0.25 dollars.
    Prove that Jenn has 5 jars full of quarters. -/
theorem jenn_has_five_jars :
  let quarters_per_jar := 160
  let bike_cost := 180
  let money_left := 20
  let total_money_needed := bike_cost + money_left
  let quarter_value := 0.25
  let total_quarters_needed := total_money_needed / quarter_value
  let jars := total_quarters_needed / quarters_per_jar
  
  jars = 5 :=
by
  sorry

end NUMINAMATH_GPT_jenn_has_five_jars_l1710_171079


namespace NUMINAMATH_GPT_possible_values_f2001_l1710_171072

noncomputable def f : ℕ → ℝ := sorry

lemma functional_equation (a b d : ℕ) (h₁ : 1 < a) (h₂ : 1 < b) (h₃ : d = Nat.gcd a b) :
  f (a * b) = f d * (f (a / d) + f (b / d)) :=
sorry

theorem possible_values_f2001 :
  f 2001 = 0 ∨ f 2001 = 1 / 2 :=
sorry

end NUMINAMATH_GPT_possible_values_f2001_l1710_171072


namespace NUMINAMATH_GPT_total_boys_across_grades_is_692_l1710_171032

theorem total_boys_across_grades_is_692 (ga_girls gb_girls gc_girls : ℕ) (ga_boys : ℕ) :
  ga_girls = 256 →
  ga_girls = ga_boys + 52 →
  gb_girls = 360 →
  gb_boys = gb_girls - 40 →
  gc_girls = 168 →
  gc_girls = gc_boys →
  ga_boys + gb_boys + gc_boys = 692 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_total_boys_across_grades_is_692_l1710_171032


namespace NUMINAMATH_GPT_factor_expression_l1710_171037

theorem factor_expression (m n x y : ℝ) :
  m * (x - y) + n * (y - x) = (x - y) * (m - n) := by
  sorry

end NUMINAMATH_GPT_factor_expression_l1710_171037


namespace NUMINAMATH_GPT_intersection_P_complement_Q_l1710_171033

-- Defining the sets P and Q
def R := Set ℝ
def P : Set ℝ := {x | x^2 + 2 * x - 3 = 0}
def Q : Set ℝ := {x | Real.log x < 1}
def complement_R_Q : Set ℝ := {x | x ≤ 0 ∨ x ≥ Real.exp 1}
def intersection := {x | x ∈ P ∧ x ∈ complement_R_Q}

-- Statement of the theorem
theorem intersection_P_complement_Q : 
  intersection = {-3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_P_complement_Q_l1710_171033


namespace NUMINAMATH_GPT_solve_for_x_l1710_171081

theorem solve_for_x (x : ℚ) (h : x > 0) (hx : 3 * x^2 + 7 * x - 20 = 0) : x = 5 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l1710_171081


namespace NUMINAMATH_GPT_xy2_plus_2y_divides_2x2y_plus_xy2_plus_8x_l1710_171075

theorem xy2_plus_2y_divides_2x2y_plus_xy2_plus_8x (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) :
  (x * y^2 + 2 * y) ∣ (2 * x^2 * y + x * y^2 + 8 * x) ↔ 
  (∃ a : ℕ, 0 < a ∧ x = a ∧ y = 2 * a) ∨ (x = 3 ∧ y = 1) ∨ (x = 8 ∧ y = 1) :=
by
  sorry

end NUMINAMATH_GPT_xy2_plus_2y_divides_2x2y_plus_xy2_plus_8x_l1710_171075


namespace NUMINAMATH_GPT_factor_expr_l1710_171023

variable (x : ℝ)

def expr : ℝ := (20 * x ^ 3 + 100 * x - 10) - (-5 * x ^ 3 + 5 * x - 10)

theorem factor_expr :
  expr x = 5 * x * (5 * x ^ 2 + 19) :=
by
  sorry

end NUMINAMATH_GPT_factor_expr_l1710_171023


namespace NUMINAMATH_GPT_length_of_BC_l1710_171029

theorem length_of_BC (x : ℝ) (h1 : (20 * x^2) / 3 - (400 * x) / 3 = 140) :
  ∃ (BC : ℝ), BC = 29 := 
by
  sorry

end NUMINAMATH_GPT_length_of_BC_l1710_171029


namespace NUMINAMATH_GPT_monochromatic_triangle_l1710_171090

def R₃ (n : ℕ) : ℕ := sorry

theorem monochromatic_triangle {n : ℕ} (h1 : R₃ 2 = 6)
  (h2 : ∀ n, R₃ (n + 1) ≤ (n + 1) * R₃ n - n + 1) :
  R₃ n ≤ 3 * Nat.factorial n :=
by
  induction n with
  | zero => sorry -- base case proof
  | succ n ih => sorry -- inductive step proof

end NUMINAMATH_GPT_monochromatic_triangle_l1710_171090


namespace NUMINAMATH_GPT_allie_carl_product_points_l1710_171016

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 3
  else 0

def allie_rolls : List ℕ := [2, 6, 3, 2, 5]
def carl_rolls : List ℕ := [1, 4, 3, 6, 6]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.foldr (λ x acc => g x + acc) 0

theorem allie_carl_product_points : (total_points allie_rolls) * (total_points carl_rolls) = 594 :=
  sorry

end NUMINAMATH_GPT_allie_carl_product_points_l1710_171016


namespace NUMINAMATH_GPT_base8_subtraction_l1710_171024

theorem base8_subtraction : (325 : Nat) - (237 : Nat) = 66 :=
by 
  sorry

end NUMINAMATH_GPT_base8_subtraction_l1710_171024


namespace NUMINAMATH_GPT_x_coordinate_of_P_l1710_171054

noncomputable section

open Real

-- Define the standard properties of the parabola and point P
def parabola (p : ℝ) (x y : ℝ) := (y ^ 2 = 4 * x)

def distance (P F : ℝ × ℝ) : ℝ := 
  sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2)

-- Position of the focus for the given parabola y^2 = 4x; Focus F(1, 0)
def focus : ℝ × ℝ := (1, 0)

-- The given conditions translated into Lean form
def on_parabola (x y : ℝ) := parabola 2 x y ∧ distance (x, y) focus = 5

-- The theorem we need to prove: If point P satisfies these conditions, then its x-coordinate is 4
theorem x_coordinate_of_P (P : ℝ × ℝ) (h : on_parabola P.1 P.2) : P.1 = 4 :=
by
  sorry

end NUMINAMATH_GPT_x_coordinate_of_P_l1710_171054


namespace NUMINAMATH_GPT_only_book_A_l1710_171021

theorem only_book_A (purchasedBoth : ℕ) (purchasedOnlyB : ℕ) (purchasedA : ℕ) (purchasedB : ℕ) 
  (h1 : purchasedBoth = 500)
  (h2 : 2 * purchasedOnlyB = purchasedBoth)
  (h3 : purchasedA = 2 * purchasedB)
  (h4 : purchasedB = purchasedOnlyB + purchasedBoth) :
  purchasedA - purchasedBoth = 1000 :=
by
  sorry

end NUMINAMATH_GPT_only_book_A_l1710_171021


namespace NUMINAMATH_GPT_right_triangle_area_and_perimeter_l1710_171074

theorem right_triangle_area_and_perimeter (a c : ℕ) (h₁ : c = 13) (h₂ : a = 5) :
  ∃ (b : ℕ), b^2 = c^2 - a^2 ∧
             (1/2 : ℝ) * (a : ℝ) * (b : ℝ) = 30 ∧
             (a + b + c : ℕ) = 30 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_area_and_perimeter_l1710_171074


namespace NUMINAMATH_GPT_length_of_row_of_small_cubes_l1710_171046

/-!
# Problem: Calculate the length of a row of smaller cubes

A cube with an edge length of 0.5 m is cut into smaller cubes, each with an edge length of 2 mm.
Prove that the length of the row formed by arranging the smaller cubes in a continuous line 
is 31 km and 250 m.
-/

noncomputable def large_cube_edge_length_m : ℝ := 0.5
noncomputable def small_cube_edge_length_mm : ℝ := 2

theorem length_of_row_of_small_cubes :
  let length_mm := 31250000
  (31 : ℝ) * 1000 + (250 : ℝ) = length_mm / 1000 + 250 := 
sorry

end NUMINAMATH_GPT_length_of_row_of_small_cubes_l1710_171046


namespace NUMINAMATH_GPT_range_of_a_l1710_171000

theorem range_of_a (a : ℝ) (a_seq : ℕ → ℝ)
  (h_cond : ∀ (n : ℕ), n > 0 → (a_seq n = if n ≤ 4 then 2^n - 1 else -n^2 + (a - 1) * n))
  (h_max_a5 : ∀ (n : ℕ), n > 0 → a_seq n ≤ a_seq 5) :
  9 ≤ a ∧ a ≤ 12 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1710_171000


namespace NUMINAMATH_GPT_min_value_xy_expression_l1710_171071

theorem min_value_xy_expression (x y : ℝ) : ∃ c : ℝ, (∀ x y : ℝ, (xy - 1)^2 + (x + y)^2 ≥ c) ∧ c = 1 :=
by {
  -- Placeholder for proof
  sorry
}

end NUMINAMATH_GPT_min_value_xy_expression_l1710_171071


namespace NUMINAMATH_GPT_isabella_canadian_dollars_sum_l1710_171052

def sum_of_digits (n : Nat) : Nat :=
  (n % 10) + ((n / 10) % 10)

theorem isabella_canadian_dollars_sum (d : Nat) (H: 10 * d = 7 * d + 280) : sum_of_digits d = 12 :=
by
  sorry

end NUMINAMATH_GPT_isabella_canadian_dollars_sum_l1710_171052


namespace NUMINAMATH_GPT_sqrt_of_25_l1710_171067

theorem sqrt_of_25 : ∃ x : ℝ, x^2 = 25 ∧ (x = 5 ∨ x = -5) :=
by {
  sorry
}

end NUMINAMATH_GPT_sqrt_of_25_l1710_171067


namespace NUMINAMATH_GPT_exists_close_ratios_l1710_171001

theorem exists_close_ratios (S : Finset ℝ) (h : S.card = 2000) :
  ∃ (a b c d : ℝ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a > b ∧ c > d ∧ (a ≠ c ∨ b ≠ d) ∧
  abs ((a - b) / (c - d) - 1) < 1 / 100000 :=
sorry

end NUMINAMATH_GPT_exists_close_ratios_l1710_171001


namespace NUMINAMATH_GPT_definite_integral_abs_poly_l1710_171084

theorem definite_integral_abs_poly :
  ∫ x in (-2 : ℝ)..(2 : ℝ), |x^2 - 2*x| = 8 :=
by
  sorry

end NUMINAMATH_GPT_definite_integral_abs_poly_l1710_171084


namespace NUMINAMATH_GPT_area_ratio_proof_l1710_171035

open Real

noncomputable def area_ratio (FE AF DE CD ABCE : ℝ) :=
  (AF = 3 * FE) ∧ (CD = 3 * DE) ∧ (ABCE = 16 * FE^2) →
  (10 * FE^2 / ABCE = (5 / 8))

theorem area_ratio_proof (FE AF DE CD ABCE : ℝ) :
  AF = 3 * FE → CD = 3 * DE → ABCE = 16 * FE^2 →
  10 * FE^2 / ABCE = 5 / 8 :=
by
  intro hAF hCD hABCE
  sorry

end NUMINAMATH_GPT_area_ratio_proof_l1710_171035


namespace NUMINAMATH_GPT_empty_pencil_cases_l1710_171078

theorem empty_pencil_cases (total_cases pencil_cases pen_cases both_cases : ℕ) 
  (h1 : total_cases = 10)
  (h2 : pencil_cases = 5)
  (h3 : pen_cases = 4)
  (h4 : both_cases = 2) : total_cases - (pencil_cases + pen_cases - both_cases) = 3 := by
  sorry

end NUMINAMATH_GPT_empty_pencil_cases_l1710_171078


namespace NUMINAMATH_GPT_proportion_problem_l1710_171065

theorem proportion_problem 
  (x : ℝ) 
  (third_number : ℝ) 
  (h1 : 0.75 / x = third_number / 8) 
  (h2 : x = 0.6) 
  : third_number = 10 := 
by 
  sorry

end NUMINAMATH_GPT_proportion_problem_l1710_171065


namespace NUMINAMATH_GPT_sum_series_eq_4_div_9_l1710_171017

theorem sum_series_eq_4_div_9 : ∑' k : ℕ, (k + 1) / 4^(k + 1) = 4 / 9 := 
sorry

end NUMINAMATH_GPT_sum_series_eq_4_div_9_l1710_171017


namespace NUMINAMATH_GPT_area_of_rectangular_garden_l1710_171096

-- Definition of conditions
def width : ℕ := 14
def length : ℕ := 3 * width

-- Statement for proof of the area of the rectangular garden
theorem area_of_rectangular_garden :
  length * width = 588 := 
by
  sorry

end NUMINAMATH_GPT_area_of_rectangular_garden_l1710_171096


namespace NUMINAMATH_GPT_solution_sum_l1710_171039

theorem solution_sum (m n : ℝ) (h₀ : m ≠ 0) (h₁ : m^2 + m * n - m = 0) : m + n = 1 := 
by 
  sorry

end NUMINAMATH_GPT_solution_sum_l1710_171039


namespace NUMINAMATH_GPT_part1_part2_l1710_171036

open Real

noncomputable def part1_statement (m : ℝ) : Prop := ∀ x : ℝ, m * x^2 - 2 * m * x - 1 < 0

noncomputable def part2_statement (x : ℝ) : Prop := 
  ∀ (m : ℝ), |m| ≤ 1 → (m * x^2 - 2 * m * x - 1 < 0)

theorem part1 : part1_statement m ↔ (-1 < m ∧ m ≤ 0) :=
sorry

theorem part2 : part2_statement x ↔ ((1 - sqrt 2 < x ∧ x < 1) ∨ (1 < x ∧ x < 1 + sqrt 2)) :=
sorry

end NUMINAMATH_GPT_part1_part2_l1710_171036


namespace NUMINAMATH_GPT_mean_weight_is_70_357_l1710_171077

def weights_50 : List ℕ := [57]
def weights_60 : List ℕ := [60, 64, 64, 66, 69]
def weights_70 : List ℕ := [71, 73, 73, 75, 77, 78, 79, 79]

def weights := weights_50 ++ weights_60 ++ weights_70

def total_weight : ℕ := List.sum weights
def total_players : ℕ := List.length weights
def mean_weight : ℚ := (total_weight : ℚ) / total_players

theorem mean_weight_is_70_357 :
  mean_weight = 70.357 := 
sorry

end NUMINAMATH_GPT_mean_weight_is_70_357_l1710_171077


namespace NUMINAMATH_GPT_find_angle_A_find_range_expression_l1710_171006

-- Define the variables and conditions in a way consistent with Lean's syntax
variables {α β γ : Type}
variables (a b c : ℝ) (A B C : ℝ)

-- The mathematical conditions translated to Lean
def triangle_condition (a b c A B C : ℝ) : Prop := (b + c) / a = Real.cos B + Real.cos C

-- Statement for Proof 1: Prove that A = π/2 given the conditions
theorem find_angle_A (h : triangle_condition a b c A B C) : A = Real.pi / 2 :=
sorry

-- Statement for Proof 2: Prove the range of the given expression under the given conditions
theorem find_range_expression (h : triangle_condition a b c A B C) (hA : A = Real.pi / 2) :
  ∃ (l u : ℝ), l = Real.sqrt 3 + 2 ∧ u = Real.sqrt 3 + 3 ∧ (2 * Real.cos (B / 2) ^ 2 + 2 * Real.sqrt 3 * Real.cos (C / 2) ^ 2) ∈ Set.Ioc l u :=
sorry

end NUMINAMATH_GPT_find_angle_A_find_range_expression_l1710_171006


namespace NUMINAMATH_GPT_solve_inequality_l1710_171049

theorem solve_inequality (a x : ℝ) : 
  (ax^2 + (a - 1) * x - 1 < 0) ↔ (
  (a = 0 ∧ x > -1) ∨ 
  (a > 0 ∧ -1 < x ∧ x < 1/a) ∨
  (-1 < a ∧ a < 0 ∧ (x < 1/a ∨ x > -1)) ∨ 
  (a = -1 ∧ x ≠ -1) ∨ 
  (a < -1 ∧ (x < -1 ∨ x > 1/a))
) := sorry

end NUMINAMATH_GPT_solve_inequality_l1710_171049


namespace NUMINAMATH_GPT_garden_perimeter_l1710_171050

-- We are given:
variables (a b : ℝ)
variables (h1 : b = 3 * a)
variables (h2 : a^2 + b^2 = 34^2)
variables (h3 : a * b = 240)

-- We must prove:
theorem garden_perimeter (h4 : a^2 + 9 * a^2 = 1156) (h5 : 10 * a^2 = 1156) (h6 : a^2 = 115.6) 
  (h7 : 3 * a^2 = 240) (h8 : a^2 = 80) :
  2 * (a + b) = 72 := 
by
  sorry

end NUMINAMATH_GPT_garden_perimeter_l1710_171050


namespace NUMINAMATH_GPT_product_b2_b7_l1710_171025

def is_increasing_arithmetic_sequence (bs : ℕ → ℤ) :=
  ∀ n m : ℕ, n < m → bs n < bs m

def arithmetic_sequence (bs : ℕ → ℤ) (d : ℤ) :=
  ∀ n : ℕ, bs (n + 1) - bs n = d

theorem product_b2_b7 (bs : ℕ → ℤ) (d : ℤ) (h_incr : is_increasing_arithmetic_sequence bs)
    (h_arith : arithmetic_sequence bs d)
    (h_prod : bs 4 * bs 5 = 10) :
    bs 2 * bs 7 = -224 ∨ bs 2 * bs 7 = -44 :=
by
  sorry

end NUMINAMATH_GPT_product_b2_b7_l1710_171025


namespace NUMINAMATH_GPT_sum_of_two_numbers_l1710_171002

theorem sum_of_two_numbers (x y : ℝ) (h1 : 0.5 * x + 0.3333 * y = 11)
(h2 : max x y = y) (h3 : y = 15) : x + y = 27 :=
by
  -- Skip the proof and add sorry
  sorry

end NUMINAMATH_GPT_sum_of_two_numbers_l1710_171002


namespace NUMINAMATH_GPT_fish_remain_approximately_correct_l1710_171076

noncomputable def remaining_fish : ℝ :=
  let west_initial := 1800
  let east_initial := 3200
  let north_initial := 500
  let south_initial := 2300
  let a := 3
  let b := 4
  let c := 2
  let d := 5
  let e := 1
  let f := 3
  let west_caught := (a / b) * west_initial
  let east_caught := (c / d) * east_initial
  let south_caught := (e / f) * south_initial
  let west_left := west_initial - west_caught
  let east_left := east_initial - east_caught
  let south_left := south_initial - south_caught
  let north_left := north_initial
  west_left + east_left + south_left + north_left

theorem fish_remain_approximately_correct :
  abs (remaining_fish - 4403) < 1 := 
  sorry

end NUMINAMATH_GPT_fish_remain_approximately_correct_l1710_171076


namespace NUMINAMATH_GPT_positive_difference_of_sums_l1710_171068

def sum_first_n (n : Nat) : Nat := n * (n + 1) / 2

def sum_first_n_even (n : Nat) : Nat := 2 * sum_first_n n

def sum_first_n_odd (n : Nat) : Nat := n * n

theorem positive_difference_of_sums :
  let S1 := sum_first_n_even 25
  let S2 := sum_first_n_odd 20
  S1 - S2 = 250 := by
  sorry

end NUMINAMATH_GPT_positive_difference_of_sums_l1710_171068


namespace NUMINAMATH_GPT_seventh_grader_count_l1710_171066

variables {x n : ℝ}

noncomputable def number_of_seventh_graders (x n : ℝ) :=
  10 * x = 10 * x ∧  -- Condition 1
  4.5 * n = 4.5 * n ∧  -- Condition 2
  11 * x = 11 * x ∧  -- Condition 3
  5.5 * n = 5.5 * n ∧  -- Condition 4
  5.5 * n = (11 * x * (11 * x - 1)) / 2 ∧  -- Condition 5
  n = x * (11 * x - 1)  -- Condition 6

theorem seventh_grader_count (x n : ℝ) (h : number_of_seventh_graders x n) : x = 1 :=
  sorry

end NUMINAMATH_GPT_seventh_grader_count_l1710_171066


namespace NUMINAMATH_GPT_determine_contents_l1710_171086

inductive Color
| White
| Black

open Color

-- Definitions of the mislabeled boxes
def mislabeled (box : Nat → List Color) : Prop :=
  ¬ (box 1 = [Black, Black] ∧ box 2 = [Black, White]
     ∧ box 3 = [White, White])

-- Draw a ball from a box revealing its content
def draw_ball (box : Nat → List Color) (i : Nat) (c : Color) : Prop :=
  c ∈ box i

-- theorem statement
theorem determine_contents (box : Nat → List Color) (c : Color) (h : draw_ball box 3 c) (hl : mislabeled box) :
  (c = White → box 3 = [White, White] ∧ box 2 = [Black, White] ∧ box 1 = [Black, Black]) ∧
  (c = Black → box 3 = [Black, Black] ∧ box 2 = [Black, White] ∧ box 1 = [White, White]) :=
by
  sorry

end NUMINAMATH_GPT_determine_contents_l1710_171086


namespace NUMINAMATH_GPT_divisor_five_l1710_171043

theorem divisor_five {D : ℝ} (h : 95 / D + 23 = 42) : D = 5 := by
  sorry

end NUMINAMATH_GPT_divisor_five_l1710_171043


namespace NUMINAMATH_GPT_total_bus_capacity_l1710_171027

def left_seats : ℕ := 15
def right_seats : ℕ := left_seats - 3
def people_per_seat : ℕ := 3
def back_seat_capacity : ℕ := 8

theorem total_bus_capacity :
  (left_seats + right_seats) * people_per_seat + back_seat_capacity = 89 := by
  sorry

end NUMINAMATH_GPT_total_bus_capacity_l1710_171027


namespace NUMINAMATH_GPT_typist_current_salary_l1710_171013

def original_salary : ℝ := 4000.0000000000005
def increased_salary (os : ℝ) : ℝ := os + (os * 0.1)
def decreased_salary (is : ℝ) : ℝ := is - (is * 0.05)

theorem typist_current_salary : decreased_salary (increased_salary original_salary) = 4180 :=
by
  sorry

end NUMINAMATH_GPT_typist_current_salary_l1710_171013


namespace NUMINAMATH_GPT_isosceles_base_angle_l1710_171062

theorem isosceles_base_angle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A = B ∨ A = C) (h3 : A = 80 ∨ B = 80 ∨ C = 80) : (A = 80 ∧ B = 80) ∨ (A = 80 ∧ C = 80) ∨ (B = 80 ∧ C = 50) ∨ (C = 80 ∧ B = 50) :=
sorry

end NUMINAMATH_GPT_isosceles_base_angle_l1710_171062


namespace NUMINAMATH_GPT_employee_pay_per_week_l1710_171009

theorem employee_pay_per_week (total_pay : ℝ) (ratio : ℝ) (pay_b : ℝ)
  (h1 : total_pay = 570)
  (h2 : ratio = 1.5)
  (h3 : total_pay = pay_b * (ratio + 1)) :
  pay_b = 228 :=
sorry

end NUMINAMATH_GPT_employee_pay_per_week_l1710_171009


namespace NUMINAMATH_GPT_paint_left_l1710_171042

-- Define the conditions
def total_paint_needed : ℕ := 333
def paint_needed_to_buy : ℕ := 176

-- State the theorem
theorem paint_left : total_paint_needed - paint_needed_to_buy = 157 := 
by 
  sorry

end NUMINAMATH_GPT_paint_left_l1710_171042


namespace NUMINAMATH_GPT_correct_operation_l1710_171053

theorem correct_operation (a b : ℝ) : 
  (3 * Real.sqrt 7 + 7 * Real.sqrt 3 ≠ 10 * Real.sqrt 10) ∧ 
  (Real.sqrt (2 * a) * Real.sqrt (3) * a = Real.sqrt (6) * a) ∧ 
  (Real.sqrt a - Real.sqrt b ≠ Real.sqrt (a - b)) ∧ 
  (Real.sqrt (20 / 45) ≠ 4 / 9) :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l1710_171053


namespace NUMINAMATH_GPT_find_width_of_plot_l1710_171057

def length : ℕ := 90
def poles : ℕ := 52
def distance_between_poles : ℕ := 5
def perimeter : ℕ := poles * distance_between_poles

theorem find_width_of_plot (perimeter_eq : perimeter = 2 * (length + width)) : width = 40 := by
  sorry

end NUMINAMATH_GPT_find_width_of_plot_l1710_171057


namespace NUMINAMATH_GPT_product_gcd_lcm_150_90_l1710_171073

theorem product_gcd_lcm_150_90 (a b : ℕ) (h1 : a = 150) (h2 : b = 90): Nat.gcd a b * Nat.lcm a b = a * b := by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_product_gcd_lcm_150_90_l1710_171073


namespace NUMINAMATH_GPT_sales_tax_difference_l1710_171060

theorem sales_tax_difference
  (item_price : ℝ)
  (rate1 rate2 : ℝ)
  (h_rate1 : rate1 = 0.0725)
  (h_rate2 : rate2 = 0.0675)
  (h_item_price : item_price = 40) :
  item_price * rate1 - item_price * rate2 = 0.20 :=
by
  -- Since we are required to skip the proof, we put sorry here.
  sorry

end NUMINAMATH_GPT_sales_tax_difference_l1710_171060


namespace NUMINAMATH_GPT_sum_of_ages_is_42_l1710_171003

-- Define the variables for present ages of the son (S) and the father (F)
variables (S F : ℕ)

-- Define the conditions:
-- 1. 6 years ago, the father's age was 4 times the son's age.
-- 2. After 6 years, the son's age will be 18 years.

def son_age_condition := S + 6 = 18
def father_age_6_years_ago_condition := F - 6 = 4 * (S - 6)

-- Theorem statement to prove:
theorem sum_of_ages_is_42 (S F : ℕ)
  (h1 : son_age_condition S)
  (h2 : father_age_6_years_ago_condition F S) :
  S + F = 42 :=
sorry

end NUMINAMATH_GPT_sum_of_ages_is_42_l1710_171003


namespace NUMINAMATH_GPT_number_of_labelings_l1710_171007

-- Define the concept of a truncated chessboard with 8 squares
structure TruncatedChessboard :=
(square_labels : Fin 8 → ℕ)
(condition : ∀ i j, i ≠ j → square_labels i ≠ square_labels j)

-- Assuming a wider adjacency matrix for "connected" (has at least one common vertex)
def connected (i j : Fin 8) : Prop := sorry

-- Define the non-consecutiveness condition
def non_consecutive (board : TruncatedChessboard) :=
  ∀ i j, connected i j → (board.square_labels i ≠ board.square_labels j + 1 ∧
                          board.square_labels i ≠ board.square_labels j - 1)

-- Theorem statement
theorem number_of_labelings : ∃ c : Fin 8 → ℕ, ∀ b : TruncatedChessboard, non_consecutive b → 
  (b.square_labels = c) := sorry

end NUMINAMATH_GPT_number_of_labelings_l1710_171007


namespace NUMINAMATH_GPT_part_1_part_2_part_3_l1710_171085

/-- Defining a structure to hold the values of x and y as given in the problem --/
structure PhoneFeeData (α : Type) :=
  (x : α) (y : α)

def problem_data : List (PhoneFeeData ℝ) :=
  [
    ⟨1, 18.4⟩, ⟨2, 18.8⟩, ⟨3, 19.2⟩, ⟨4, 19.6⟩, ⟨5, 20⟩, ⟨6, 20.4⟩
  ]

noncomputable def phone_fee_equation (x : ℝ) : ℝ := 0.4 * x + 18

theorem part_1 :
  ∀ data ∈ problem_data, phone_fee_equation data.x = data.y :=
by
  sorry

theorem part_2 : phone_fee_equation 10 = 22 :=
by
  sorry

theorem part_3 : ∀ x : ℝ, phone_fee_equation x = 26 → x = 20 :=
by
  sorry

end NUMINAMATH_GPT_part_1_part_2_part_3_l1710_171085


namespace NUMINAMATH_GPT_value_of_x_l1710_171082

theorem value_of_x (x : ℝ) (h : x = 80 + 0.2 * 80) : x = 96 :=
sorry

end NUMINAMATH_GPT_value_of_x_l1710_171082


namespace NUMINAMATH_GPT_children_marbles_problem_l1710_171041

theorem children_marbles_problem (n x N : ℕ) 
  (h1 : N = n * x)
  (h2 : 1 + (N - 1) / 10 = x) :
  n = 9 ∧ x = 9 :=
by
  sorry

end NUMINAMATH_GPT_children_marbles_problem_l1710_171041


namespace NUMINAMATH_GPT_avg_of_x_y_is_41_l1710_171089

theorem avg_of_x_y_is_41 
  (x y : ℝ) 
  (h : (4 + 6 + 8 + x + y) / 5 = 20) 
  : (x + y) / 2 = 41 := 
by 
  sorry

end NUMINAMATH_GPT_avg_of_x_y_is_41_l1710_171089


namespace NUMINAMATH_GPT_milkman_total_profit_l1710_171055

-- Declare the conditions
def initialMilk : ℕ := 50
def initialWater : ℕ := 15
def firstMixtureMilk : ℕ := 30
def firstMixtureWater : ℕ := 8
def remainingMilk : ℕ := initialMilk - firstMixtureMilk
def secondMixtureMilk : ℕ := remainingMilk
def secondMixtureWater : ℕ := 7
def costOfMilkPerLiter : ℕ := 20
def sellingPriceFirstMixturePerLiter : ℕ := 17
def sellingPriceSecondMixturePerLiter : ℕ := 15
def totalCostOfMilk := (firstMixtureMilk + secondMixtureMilk) * costOfMilkPerLiter
def totalRevenueFirstMixture := (firstMixtureMilk + firstMixtureWater) * sellingPriceFirstMixturePerLiter
def totalRevenueSecondMixture := (secondMixtureMilk + secondMixtureWater) * sellingPriceSecondMixturePerLiter
def totalRevenue := totalRevenueFirstMixture + totalRevenueSecondMixture
def totalProfit := totalRevenue - totalCostOfMilk

-- Proof statement
theorem milkman_total_profit : totalProfit = 51 := by
  sorry

end NUMINAMATH_GPT_milkman_total_profit_l1710_171055


namespace NUMINAMATH_GPT_calculate_value_l1710_171070

def a : ℕ := 2500
def b : ℕ := 2109
def d : ℕ := 64

theorem calculate_value : (a - b) ^ 2 / d = 2389 := by
  sorry

end NUMINAMATH_GPT_calculate_value_l1710_171070


namespace NUMINAMATH_GPT_average_age_of_class_l1710_171028

theorem average_age_of_class 
  (avg_age_8 : ℕ → ℕ)
  (avg_age_6 : ℕ → ℕ)
  (age_15th : ℕ)
  (A : ℕ)
  (h1 : avg_age_8 8 = 112)
  (h2 : avg_age_6 6 = 96)
  (h3 : age_15th = 17)
  (h4 : 15 * A = (avg_age_8 8) + (avg_age_6 6) + age_15th)
  : A = 15 :=
by
  sorry

end NUMINAMATH_GPT_average_age_of_class_l1710_171028


namespace NUMINAMATH_GPT_balance_five_diamonds_bullets_l1710_171069

variables (a b c : ℝ)

-- Conditions
def condition1 : Prop := 4 * a + 2 * b = 12 * c
def condition2 : Prop := 2 * a = b + 4 * c

-- Theorem statement
theorem balance_five_diamonds_bullets (h1 : condition1 a b c) (h2 : condition2 a b c) : 5 * b = 5 * c :=
by
  sorry

end NUMINAMATH_GPT_balance_five_diamonds_bullets_l1710_171069


namespace NUMINAMATH_GPT_manager_salary_is_correct_l1710_171012

noncomputable def manager_salary (avg_salary_50_employees : ℝ) (increase_in_avg : ℝ) : ℝ :=
  let total_salary_50_employees := 50 * avg_salary_50_employees
  let new_avg_salary := avg_salary_50_employees + increase_in_avg
  let total_salary_51_people := 51 * new_avg_salary
  let manager_salary := total_salary_51_people - total_salary_50_employees
  manager_salary

theorem manager_salary_is_correct :
  manager_salary 2500 1500 = 79000 :=
by
  sorry

end NUMINAMATH_GPT_manager_salary_is_correct_l1710_171012


namespace NUMINAMATH_GPT_area_of_fourth_rectangle_l1710_171094

-- The conditions provided in the problem
variables (x y z w : ℝ)
variables (h1 : x * y = 24) (h2 : x * w = 12) (h3 : z * w = 8)

-- The problem statement with the conclusion
theorem area_of_fourth_rectangle :
  (∃ (x y z w : ℝ), ((x * y = 24 ∧ x * w = 12 ∧ z * w = 8) ∧ y * z = 16)) :=
sorry

end NUMINAMATH_GPT_area_of_fourth_rectangle_l1710_171094


namespace NUMINAMATH_GPT_maximize_profit_l1710_171022

-- Define the price of the book
variables (p : ℝ) (p_max : ℝ)
-- Define the revenue function
def R (p : ℝ) : ℝ := p * (150 - 4 * p)
-- Define the profit function accounting for fixed costs of $200
def P (p : ℝ) := R p - 200
-- Set the maximum feasible price
def max_price_condition := p_max = 30
-- Define the price that maximizes the profit
def optimal_price := 18.75

-- The theorem to be proved
theorem maximize_profit : p_max = 30 → p = 18.75 → P p = 2612.5 :=
by {
  sorry
}

end NUMINAMATH_GPT_maximize_profit_l1710_171022


namespace NUMINAMATH_GPT_mary_money_after_purchase_l1710_171047

def mary_initial_money : ℕ := 58
def pie_cost : ℕ := 6
def mary_friend_money : ℕ := 43  -- This is an extraneous condition, included for completeness.

theorem mary_money_after_purchase : mary_initial_money - pie_cost = 52 := by
  sorry

end NUMINAMATH_GPT_mary_money_after_purchase_l1710_171047


namespace NUMINAMATH_GPT_work_days_for_A_l1710_171058

theorem work_days_for_A (x : ℕ) : 
  (∀ a b, 
    (a = 1 / (x : ℚ)) ∧ 
    (b = 1 / 20) ∧ 
    (8 * (a + b) = 14 / 15) → 
    x = 15) :=
by
  intros a b h
  have ha : a = 1 / (x : ℚ) := h.1
  have hb : b = 1 / 20 := h.2.1
  have hab : 8 * (a + b) = 14 / 15 := h.2.2
  sorry

end NUMINAMATH_GPT_work_days_for_A_l1710_171058


namespace NUMINAMATH_GPT_remainder_polynomial_l1710_171010

theorem remainder_polynomial (x : ℤ) : (1 + x) ^ 2010 % (1 + x + x^2) = 1 := 
  sorry

end NUMINAMATH_GPT_remainder_polynomial_l1710_171010


namespace NUMINAMATH_GPT_part1_solution_set_m1_part2_find_m_l1710_171004

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (m+1) * x^2 - m * x + m - 1

theorem part1_solution_set_m1 :
  { x : ℝ | f x 1 > 0 } = { x : ℝ | x < 0 } ∪ { x : ℝ | x > 0.5 } :=
by
  sorry

theorem part2_find_m :
  (∀ x : ℝ, f x m + 1 > 0 ↔ x > 1.5 ∧ x < 3) → m = -9/7 :=
by
  sorry

end NUMINAMATH_GPT_part1_solution_set_m1_part2_find_m_l1710_171004


namespace NUMINAMATH_GPT_molecular_weight_of_7_moles_KBrO3_l1710_171018

def potassium_atomic_weight : ℝ := 39.10
def bromine_atomic_weight : ℝ := 79.90
def oxygen_atomic_weight : ℝ := 16.00
def oxygen_atoms_in_KBrO3 : ℝ := 3

def KBrO3_molecular_weight : ℝ := 
  potassium_atomic_weight + bromine_atomic_weight + (oxygen_atomic_weight * oxygen_atoms_in_KBrO3)

def moles := 7

theorem molecular_weight_of_7_moles_KBrO3 : KBrO3_molecular_weight * moles = 1169.00 := 
by {
  -- The proof would be here, but it is omitted as instructed.
  sorry
}

end NUMINAMATH_GPT_molecular_weight_of_7_moles_KBrO3_l1710_171018


namespace NUMINAMATH_GPT_suit_price_the_day_after_sale_l1710_171034

def originalPrice : ℕ := 300
def increaseRate : ℚ := 0.20
def couponDiscount : ℚ := 0.30
def additionalReduction : ℚ := 0.10

def increasedPrice := originalPrice * (1 + increaseRate)
def priceAfterCoupon := increasedPrice * (1 - couponDiscount)
def finalPrice := increasedPrice * (1 - additionalReduction)

theorem suit_price_the_day_after_sale 
  (op : ℕ := originalPrice) 
  (ir : ℚ := increaseRate) 
  (cd : ℚ := couponDiscount) 
  (ar : ℚ := additionalReduction) :
  finalPrice = 324 := 
sorry

end NUMINAMATH_GPT_suit_price_the_day_after_sale_l1710_171034


namespace NUMINAMATH_GPT_function_domain_l1710_171097

theorem function_domain (x : ℝ) :
  (x + 5 ≥ 0) ∧ (x + 2 ≠ 0) ↔ (x ≥ -5) ∧ (x ≠ -2) :=
by
  sorry

end NUMINAMATH_GPT_function_domain_l1710_171097


namespace NUMINAMATH_GPT_total_eggs_examined_l1710_171080

def trays := 7
def eggs_per_tray := 10

theorem total_eggs_examined : trays * eggs_per_tray = 70 :=
by 
  sorry

end NUMINAMATH_GPT_total_eggs_examined_l1710_171080


namespace NUMINAMATH_GPT_minimum_value_exists_l1710_171064

theorem minimum_value_exists (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : 2 * m + n = 1) : 
  ∃ (min_val : ℝ), min_val = (3 + 2 * Real.sqrt 2) ∧ (1 / m + 1 / n ≥ min_val) :=
by {
  -- Proof will be provided here.
  sorry
}

end NUMINAMATH_GPT_minimum_value_exists_l1710_171064


namespace NUMINAMATH_GPT_simplify_expression_l1710_171026

theorem simplify_expression (x y : ℤ) (h1 : x = -2) (h2 : y = 3) :
  (x + 2 * y)^2 - (x + y) * (2 * x - y) = 23 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1710_171026


namespace NUMINAMATH_GPT_no_sqrt_negative_number_l1710_171093

theorem no_sqrt_negative_number (a b c d : ℝ) (hA : a = (-3)^2) (hB : b = 0) (hC : c = 1/8) (hD : d = -6^3) : 
  ¬ (∃ x : ℝ, x^2 = d) :=
by
  sorry

end NUMINAMATH_GPT_no_sqrt_negative_number_l1710_171093


namespace NUMINAMATH_GPT_compare_negatives_l1710_171045

theorem compare_negatives : -4 < -2.1 := 
sorry

end NUMINAMATH_GPT_compare_negatives_l1710_171045


namespace NUMINAMATH_GPT_quadrilateral_area_l1710_171044

def diagonal : ℝ := 15
def offset1 : ℝ := 6
def offset2 : ℝ := 4

theorem quadrilateral_area :
  (1/2) * diagonal * (offset1 + offset2) = 75 :=
by 
  sorry

end NUMINAMATH_GPT_quadrilateral_area_l1710_171044


namespace NUMINAMATH_GPT_counts_duel_with_marquises_l1710_171008

theorem counts_duel_with_marquises (x y z k : ℕ) (h1 : 3 * x = 2 * y) (h2 : 6 * y = 3 * z)
    (h3 : ∀ c : ℕ, c = x → ∃ m : ℕ, m = k) : k = 6 :=
by
  sorry

end NUMINAMATH_GPT_counts_duel_with_marquises_l1710_171008


namespace NUMINAMATH_GPT_range_of_a_l1710_171015

open Set

def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2 * a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 1}

theorem range_of_a (a : ℝ) (h : A a ∩ B = ∅) :
  a ∈ Iic (-1 / 2) ∪ Ici 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1710_171015


namespace NUMINAMATH_GPT_total_eggs_collected_l1710_171087

-- Define the variables given in the conditions
def Benjamin_eggs := 6
def Carla_eggs := 3 * Benjamin_eggs
def Trisha_eggs := Benjamin_eggs - 4

-- State the theorem using the conditions and correct answer in the equivalent proof problem
theorem total_eggs_collected :
  Benjamin_eggs + Carla_eggs + Trisha_eggs = 26 := by
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_total_eggs_collected_l1710_171087


namespace NUMINAMATH_GPT_find_a_l1710_171061

theorem find_a (a : ℝ) (x : ℝ) : (a - 1) * x^|a| + 4 = 0 → |a| = 1 → a ≠ 1 → a = -1 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_a_l1710_171061


namespace NUMINAMATH_GPT_minimum_value_of_f_l1710_171040

noncomputable def f (x a : ℝ) := (1/3) * x^3 + (a-1) * x^2 - 4 * a * x + a

theorem minimum_value_of_f (a : ℝ) (h : a < -1) :
  (if -3/2 < a then ∀ (x : ℝ), 2 ≤ x ∧ x ≤ 3 → f x a ≥ f (-2*a) a
   else ∀ (x : ℝ), 2 ≤ x ∧ x ≤ 3 → f x a ≥ f 3 a) :=
sorry

end NUMINAMATH_GPT_minimum_value_of_f_l1710_171040


namespace NUMINAMATH_GPT_sum_of_money_is_6000_l1710_171019

noncomputable def original_interest (P R : ℝ) := (P * R * 3) / 100
noncomputable def new_interest (P R : ℝ) := (P * (R + 2) * 3) / 100

theorem sum_of_money_is_6000 (P R : ℝ) (h : new_interest P R - original_interest P R = 360) : P = 6000 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_money_is_6000_l1710_171019


namespace NUMINAMATH_GPT_probability_of_two_points_is_three_sevenths_l1710_171014

/-- Define the problem's conditions and statement. -/
def num_choices (n : ℕ) : ℕ :=
  match n with
  | 1 => 4  -- choose 1 option from 4
  | 2 => 6  -- choose 2 options from 4 (binomial coefficient)
  | 3 => 4  -- choose 3 options from 4 (binomial coefficient)
  | _ => 0

def total_ways : ℕ := 14  -- Total combinations of choosing 1 to 3 options from 4

def two_points_ways : ℕ := 6  -- 3 ways for 1 correct, 3 ways for 2 correct (B, C, D combinations)

def probability_two_points : ℚ :=
  (two_points_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_two_points_is_three_sevenths :
  probability_two_points = (3 / 7 : ℚ) :=
sorry

end NUMINAMATH_GPT_probability_of_two_points_is_three_sevenths_l1710_171014


namespace NUMINAMATH_GPT_range_of_a_l1710_171020

theorem range_of_a (a : ℝ) 
  (p : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → a ≥ Real.exp x) 
  (q : ∃ x : ℝ, x^2 - 4 * x + a ≤ 0) : 
  e ≤ a ∧ a ≤ 4 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1710_171020


namespace NUMINAMATH_GPT_express_in_scientific_notation_l1710_171059

theorem express_in_scientific_notation : (250000 : ℝ) = 2.5 * 10^5 := 
by {
  -- proof
  sorry
}

end NUMINAMATH_GPT_express_in_scientific_notation_l1710_171059


namespace NUMINAMATH_GPT_difference_of_profit_share_l1710_171011

theorem difference_of_profit_share (a b c : ℕ) (pa pb pc : ℕ) (profit_b : ℕ) 
  (a_capital : a = 8000) (b_capital : b = 10000) (c_capital : c = 12000) 
  (b_profit_share : profit_b = 1600)
  (investment_ratio : pa / 4 = pb / 5 ∧ pb / 5 = pc / 6) :
  pa - pc = 640 := 
sorry

end NUMINAMATH_GPT_difference_of_profit_share_l1710_171011


namespace NUMINAMATH_GPT_percentage_wearing_blue_shirts_l1710_171005

theorem percentage_wearing_blue_shirts (total_students : ℕ) (red_percentage green_percentage : ℕ) 
  (other_students : ℕ) (H1 : total_students = 900) (H2 : red_percentage = 28) 
  (H3 : green_percentage = 10) (H4 : other_students = 162) : 
  (44 : ℕ) = 100 - (red_percentage + green_percentage + (other_students * 100 / total_students)) :=
by
  sorry

end NUMINAMATH_GPT_percentage_wearing_blue_shirts_l1710_171005


namespace NUMINAMATH_GPT_line_through_center_eq_line_chord_len_eq_l1710_171031

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

noncomputable def point_P : ℝ × ℝ := (2, 2)

def line_through_center (x y : ℝ) : Prop := 2 * x - y - 2 = 0

def line_chord_len (x y : ℝ) : Prop := 3 * x - 4 * y + 2 = 0 ∨ x = 2

theorem line_through_center_eq (x y : ℝ) (hC : circle_eq x y) :
  line_through_center x y :=
sorry

theorem line_chord_len_eq (x y : ℝ) (hC : circle_eq x y) (hP : x = 2 ∧ y = 2 ∧ (line_through_center x y)) :
  line_chord_len x y :=
sorry

end NUMINAMATH_GPT_line_through_center_eq_line_chord_len_eq_l1710_171031


namespace NUMINAMATH_GPT_cos_double_angle_l1710_171091

theorem cos_double_angle (α : ℝ) (h : Real.sin (α + Real.pi / 2) = 1 / 2) : Real.cos (2 * α) = -1 / 2 := by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l1710_171091


namespace NUMINAMATH_GPT_find_2n_plus_m_l1710_171083

theorem find_2n_plus_m (n m : ℤ) (h1 : 3 * n - m < 5) (h2 : n + m > 26) (h3 : 3 * m - 2 * n < 46) : 2 * n + m = 36 :=
sorry

end NUMINAMATH_GPT_find_2n_plus_m_l1710_171083


namespace NUMINAMATH_GPT_count_arithmetic_sequence_l1710_171088

theorem count_arithmetic_sequence :
  ∃ n, 195 - (n - 1) * 3 = 12 ∧ n = 62 :=
by {
  sorry
}

end NUMINAMATH_GPT_count_arithmetic_sequence_l1710_171088


namespace NUMINAMATH_GPT_student_losses_one_mark_l1710_171056

def number_of_marks_lost_per_wrong_answer (correct_ans marks_attempted total_questions total_marks correct_questions : ℤ) : ℤ :=
  (correct_ans * correct_questions - total_marks) / (total_questions - correct_questions)

theorem student_losses_one_mark
  (correct_ans : ℤ)
  (marks_attempted : ℤ)
  (total_questions : ℤ)
  (total_marks : ℤ)
  (correct_questions : ℤ)
  (total_wrong : ℤ):
  correct_ans = 4 →
  total_questions = 80 →
  total_marks = 120 →
  correct_questions = 40 →
  total_wrong = total_questions - correct_questions →
  number_of_marks_lost_per_wrong_answer correct_ans marks_attempted total_questions total_marks correct_questions = 1 :=
by
  sorry

end NUMINAMATH_GPT_student_losses_one_mark_l1710_171056
