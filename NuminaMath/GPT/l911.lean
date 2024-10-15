import Mathlib

namespace NUMINAMATH_GPT_tank_fill_time_l911_91111

theorem tank_fill_time (R1 R2 t_required : ℝ) (hR1: R1 = 1 / 8) (hR2: R2 = 1 / 12) (hT : t_required = 4.8) :
  t_required = 1 / (R1 + R2) :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_tank_fill_time_l911_91111


namespace NUMINAMATH_GPT_calories_left_for_dinner_l911_91102

def breakfast_calories : ℝ := 353
def lunch_calories : ℝ := 885
def snack_calories : ℝ := 130
def daily_limit : ℝ := 2200

def total_consumed : ℝ :=
  breakfast_calories + lunch_calories + snack_calories

theorem calories_left_for_dinner : daily_limit - total_consumed = 832 :=
by
  sorry

end NUMINAMATH_GPT_calories_left_for_dinner_l911_91102


namespace NUMINAMATH_GPT_circles_intersect_iff_l911_91170

-- Definitions of the two circles and their parameters
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 9

def circle2 (x y r : ℝ) : Prop := x^2 + y^2 + 8 * x - 6 * y + 25 - r^2 = 0

-- Lean statement to prove the range of r
theorem circles_intersect_iff (r : ℝ) (hr : 0 < r) :
  (∃ x y : ℝ, circle1 x y ∧ circle2 x y r) ↔ (2 < r ∧ r < 8) :=
by
  sorry

end NUMINAMATH_GPT_circles_intersect_iff_l911_91170


namespace NUMINAMATH_GPT_crackers_eaten_l911_91145

-- Define the number of packs and their respective number of crackers
def num_packs_8 : ℕ := 5
def num_packs_10 : ℕ := 10
def num_packs_12 : ℕ := 7
def num_packs_15 : ℕ := 3

def crackers_per_pack_8 : ℕ := 8
def crackers_per_pack_10 : ℕ := 10
def crackers_per_pack_12 : ℕ := 12
def crackers_per_pack_15 : ℕ := 15

-- Calculate the total number of animal crackers
def total_crackers : ℕ :=
  (num_packs_8 * crackers_per_pack_8) +
  (num_packs_10 * crackers_per_pack_10) +
  (num_packs_12 * crackers_per_pack_12) +
  (num_packs_15 * crackers_per_pack_15)

-- Define the number of students who didn't eat their crackers and the respective number of crackers per pack
def num_students_not_eaten : ℕ := 4
def different_crackers_not_eaten : List ℕ := [8, 10, 12, 15]

-- Calculate the total number of crackers not eaten by adding those packs.
def total_crackers_not_eaten : ℕ := different_crackers_not_eaten.sum

-- Theorem to prove the total number of crackers eaten.
theorem crackers_eaten : total_crackers - total_crackers_not_eaten = 224 :=
by
  -- Total crackers: 269
  -- Subtract crackers not eaten: 8 + 10 + 12 + 15 = 45
  -- Therefore: 269 - 45 = 224
  sorry

end NUMINAMATH_GPT_crackers_eaten_l911_91145


namespace NUMINAMATH_GPT_min_value_f_l911_91130

open Real

noncomputable def f (x : ℝ) : ℝ :=
  sqrt (15 - 12 * cos x) + 
  sqrt (4 - 2 * sqrt 3 * sin x) +
  sqrt (7 - 4 * sqrt 3 * sin x) +
  sqrt (10 - 4 * sqrt 3 * sin x - 6 * cos x)

theorem min_value_f : ∃ x : ℝ, f x = 6 := 
sorry

end NUMINAMATH_GPT_min_value_f_l911_91130


namespace NUMINAMATH_GPT_houses_in_block_l911_91128

theorem houses_in_block (junk_mail_per_house : ℕ) (total_junk_mail : ℕ) (h1 : junk_mail_per_house = 2) (h2 : total_junk_mail = 14) :
  total_junk_mail / junk_mail_per_house = 7 := by
  sorry

end NUMINAMATH_GPT_houses_in_block_l911_91128


namespace NUMINAMATH_GPT_compound_cost_correct_l911_91105

noncomputable def compound_cost_per_pound (limestone_cost shale_mix_cost : ℝ) (total_weight limestone_weight : ℝ) : ℝ :=
  let shale_mix_weight := total_weight - limestone_weight
  let total_cost := (limestone_weight * limestone_cost) + (shale_mix_weight * shale_mix_cost)
  total_cost / total_weight

theorem compound_cost_correct :
  compound_cost_per_pound 3 5 100 37.5 = 4.25 := by
  sorry

end NUMINAMATH_GPT_compound_cost_correct_l911_91105


namespace NUMINAMATH_GPT_probability_each_mailbox_has_at_least_one_letter_l911_91142

noncomputable def probability_mailbox (total_letters : ℕ) (mailboxes : ℕ) : ℚ := 
  let total_ways := mailboxes ^ total_letters
  let favorable_ways := Nat.choose total_letters (mailboxes - 1) * (mailboxes - 1).factorial
  favorable_ways / total_ways

theorem probability_each_mailbox_has_at_least_one_letter :
  probability_mailbox 3 2 = 3 / 4 := by
  sorry

end NUMINAMATH_GPT_probability_each_mailbox_has_at_least_one_letter_l911_91142


namespace NUMINAMATH_GPT_gymnast_score_difference_l911_91167

theorem gymnast_score_difference 
  (x1 x2 x3 x4 x5 : ℝ)
  (h1 : x2 + x3 + x4 + x5 = 36)
  (h2 : x1 + x2 + x3 + x4 = 36.8) :
  x1 - x5 = 0.8 :=
by sorry

end NUMINAMATH_GPT_gymnast_score_difference_l911_91167


namespace NUMINAMATH_GPT_greatest_value_a2_b2_c2_d2_l911_91193

theorem greatest_value_a2_b2_c2_d2 :
  ∃ (a b c d : ℝ), a + b = 12 ∧ ab + c + d = 54 ∧ ad + bc = 105 ∧ cd = 50 ∧ a^2 + b^2 + c^2 + d^2 = 124 := by
  sorry

end NUMINAMATH_GPT_greatest_value_a2_b2_c2_d2_l911_91193


namespace NUMINAMATH_GPT_no_solution_ineq_l911_91166

theorem no_solution_ineq (m : ℝ) : 
  (∀ x : ℝ, x - m ≥ 0 → ¬(0.5 * x + 0.5 < 2)) → m ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_ineq_l911_91166


namespace NUMINAMATH_GPT_problem_system_of_equations_l911_91123

-- Define the problem as a theorem in Lean 4
theorem problem_system_of_equations (x y c d : ℝ) (h1 : 4 * x + 2 * y = c) (h2 : 6 * y - 12 * x = d) (h3 : d ≠ 0) :
  c / d = -1 / 3 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_problem_system_of_equations_l911_91123


namespace NUMINAMATH_GPT_average_page_count_l911_91127

theorem average_page_count 
  (n1 n2 n3 n4 : ℕ)
  (p1 p2 p3 p4 total_students : ℕ)
  (h1 : n1 = 8)
  (h2 : p1 = 3)
  (h3 : n2 = 10)
  (h4 : p2 = 5)
  (h5 : n3 = 7)
  (h6 : p3 = 2)
  (h7 : n4 = 5)
  (h8 : p4 = 4)
  (h9 : total_students = 30) :
  (n1 * p1 + n2 * p2 + n3 * p3 + n4 * p4) / total_students = 36 / 10 := 
sorry

end NUMINAMATH_GPT_average_page_count_l911_91127


namespace NUMINAMATH_GPT_unique_nat_pair_l911_91168

theorem unique_nat_pair (p : ℕ) (hp_prime : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ (n m : ℕ), n ≠ m ∧ (2 / p : ℚ) = (1 / n + 1 / m : ℚ) ∧ ∀ (n' m' : ℕ), 
  n' ≠ m' ∧ (2 / p : ℚ) = (1 / n' + 1 / m' : ℚ) → (n', m') = (n, m) ∨ (n', m') = (m, n) :=
by
  sorry

end NUMINAMATH_GPT_unique_nat_pair_l911_91168


namespace NUMINAMATH_GPT_william_tickets_l911_91107

theorem william_tickets (initial_tickets final_tickets : ℕ) (h1 : initial_tickets = 15) (h2 : final_tickets = 18) : 
  final_tickets - initial_tickets = 3 := 
by
  sorry

end NUMINAMATH_GPT_william_tickets_l911_91107


namespace NUMINAMATH_GPT_pedro_more_squares_l911_91156

theorem pedro_more_squares
  (jesus_squares : ℕ)
  (linden_squares : ℕ)
  (pedro_squares : ℕ)
  (jesus_linden_combined : jesus_squares + linden_squares = 135)
  (pedro_total : pedro_squares = 200) :
  pedro_squares - (jesus_squares + linden_squares) = 65 :=
by
  sorry

end NUMINAMATH_GPT_pedro_more_squares_l911_91156


namespace NUMINAMATH_GPT_cube_root_of_64_eq_two_pow_m_l911_91192

theorem cube_root_of_64_eq_two_pow_m (m : ℕ) (h : (64 : ℝ) ^ (1 / 3) = (2 : ℝ) ^ m) : m = 2 := 
sorry

end NUMINAMATH_GPT_cube_root_of_64_eq_two_pow_m_l911_91192


namespace NUMINAMATH_GPT_no_real_solutions_eqn_l911_91160

theorem no_real_solutions_eqn : ∀ x : ℝ, (2 * x - 4 * x + 7)^2 + 1 ≠ -|x^2 - 1| :=
by
  intro x
  sorry

end NUMINAMATH_GPT_no_real_solutions_eqn_l911_91160


namespace NUMINAMATH_GPT_find_f_one_seventh_l911_91104

-- Define the function f
variable (f : ℝ → ℝ)

-- Given conditions
variable (monotonic_f : MonotonicOn f (Set.Ioi 0))
variable (h : ∀ x ∈ Set.Ioi (0 : ℝ), f (f x - 1 / x) = 2)

-- Define the domain
variable (x : ℝ)
variable (hx : x ∈ Set.Ioi (0 : ℝ))

-- The theorem to prove
theorem find_f_one_seventh : f (1 / 7) = 8 := by
  -- proof starts here
  sorry

end NUMINAMATH_GPT_find_f_one_seventh_l911_91104


namespace NUMINAMATH_GPT_cube_problem_l911_91112

theorem cube_problem (n : ℕ) (H1 : 6 * n^2 = 1 / 3 * 6 * n^3) : n = 3 :=
sorry

end NUMINAMATH_GPT_cube_problem_l911_91112


namespace NUMINAMATH_GPT_kerosene_cost_l911_91110

theorem kerosene_cost (R E K : ℕ) (h1 : E = R) (h2 : K = 6 * E) (h3 : R = 24) : 2 * K = 288 :=
by
  sorry

end NUMINAMATH_GPT_kerosene_cost_l911_91110


namespace NUMINAMATH_GPT_triangle_ABC_perimeter_l911_91150

noncomputable def triangle_perimeter (A B C D : Type) (AD BC AC AB : ℝ) : ℝ :=
  AD + BC + AC + AB

theorem triangle_ABC_perimeter (A B C D : Type) (AD BC : ℝ) (cos_BDC : ℝ) (angle_sum : ℝ) (AC : ℝ) (AB : ℝ) :
  AD = 3 → BC = 2 → cos_BDC = 13 / 20 → angle_sum = 180 → 
  (triangle_perimeter A B C D AD BC AC AB = 11) :=
by
  sorry

end NUMINAMATH_GPT_triangle_ABC_perimeter_l911_91150


namespace NUMINAMATH_GPT_ellipse_iff_constant_sum_l911_91154

-- Let F_1 and F_2 be two fixed points in the plane.
variables (F1 F2 : Point)
-- Let d be a constant.
variable (d : ℝ)

-- A point M in a plane
variable (M : Point)

-- Define the distance function between two points.
def dist (P Q : Point) : ℝ := sorry

-- Definition: M is on an ellipse with foci F1 and F2
def on_ellipse (M F1 F2 : Point) (d : ℝ) : Prop :=
  dist M F1 + dist M F2 = d

-- Proof that shows the two parts of the statement
theorem ellipse_iff_constant_sum :
  (∀ M, on_ellipse M F1 F2 d) ↔ (∀ M, dist M F1 + dist M F2 = d) ∧ d > dist F1 F2 :=
sorry

end NUMINAMATH_GPT_ellipse_iff_constant_sum_l911_91154


namespace NUMINAMATH_GPT_oranges_in_second_group_l911_91103

namespace oranges_problem

-- Definitions coming from conditions
def cost_of_apple : ℝ := 0.21
def total_cost_1 : ℝ := 1.77
def total_cost_2 : ℝ := 1.27
def num_apples_group1 : ℕ := 6
def num_oranges_group1 : ℕ := 3
def num_apples_group2 : ℕ := 2
def cost_of_orange : ℝ := 0.17
def num_oranges_group2 : ℕ := 5 -- derived from the solution involving $0.85/$0.17.

-- Price calculation functions and conditions
def price_group1 (cost_of_orange : ℝ) : ℝ :=
  num_apples_group1 * cost_of_apple + num_oranges_group1 * cost_of_orange

def price_group2 (num_oranges_group2 cost_of_orange : ℝ) : ℝ :=
  num_apples_group2 * cost_of_apple + num_oranges_group2 * cost_of_orange

theorem oranges_in_second_group :
  (price_group1 cost_of_orange = total_cost_1) →
  (price_group2 num_oranges_group2 cost_of_orange = total_cost_2) →
  num_oranges_group2 = 5 :=
by
  intros h1 h2
  sorry

end oranges_problem

end NUMINAMATH_GPT_oranges_in_second_group_l911_91103


namespace NUMINAMATH_GPT_age_of_oldest_sibling_l911_91191

theorem age_of_oldest_sibling (Kay_siblings : ℕ) (Kay_age : ℕ) (youngest_sibling_age : ℕ) (oldest_sibling_age : ℕ) 
  (h1 : Kay_siblings = 14) (h2 : Kay_age = 32) (h3 : youngest_sibling_age = Kay_age / 2 - 5) 
  (h4 : oldest_sibling_age = 4 * youngest_sibling_age) : oldest_sibling_age = 44 := 
sorry

end NUMINAMATH_GPT_age_of_oldest_sibling_l911_91191


namespace NUMINAMATH_GPT_outer_boundary_diameter_l911_91178

def width_jogging_path : ℝ := 10
def width_vegetable_garden : ℝ := 12
def diameter_pond : ℝ := 20

theorem outer_boundary_diameter :
  2 * (diameter_pond / 2 + width_vegetable_garden + width_jogging_path) = 64 := by
  sorry

end NUMINAMATH_GPT_outer_boundary_diameter_l911_91178


namespace NUMINAMATH_GPT_radius_of_tangent_intersection_l911_91139

variable (x y : ℝ)

def circle_eq : Prop := x^2 + y^2 = 25

def tangent_condition : Prop := y = 5 ∧ x = 0

theorem radius_of_tangent_intersection (h1 : circle_eq x y) (h2 : tangent_condition x y) : ∃r : ℝ, r = 5 :=
by sorry

end NUMINAMATH_GPT_radius_of_tangent_intersection_l911_91139


namespace NUMINAMATH_GPT_max_a_b_c_d_l911_91185

theorem max_a_b_c_d (a c d b : ℤ) (hb : b > 0) (h1 : a + b = c) (h2 : b + c = d) (h3 : c + d = a) 
: a + b + c + d = -5 :=
by
  sorry

end NUMINAMATH_GPT_max_a_b_c_d_l911_91185


namespace NUMINAMATH_GPT_integer_triples_soln_l911_91121

theorem integer_triples_soln (x y z : ℤ) :
  (x^3 + y^3 + z^3 - 3*x*y*z = 2003) ↔ ( (x = 668 ∧ y = 668 ∧ z = 667) ∨ (x = 668 ∧ y = 667 ∧ z = 668) ∨ (x = 667 ∧ y = 668 ∧ z = 668) ) := 
by
  sorry

end NUMINAMATH_GPT_integer_triples_soln_l911_91121


namespace NUMINAMATH_GPT_chocolate_syrup_per_glass_l911_91174

-- Definitions from the conditions
def each_glass_volume : ℝ := 8
def milk_per_glass : ℝ := 6.5
def total_milk : ℝ := 130
def total_chocolate_syrup : ℝ := 60
def total_chocolate_milk : ℝ := 160

-- Proposition and statement to prove
theorem chocolate_syrup_per_glass : 
  (total_chocolate_milk / each_glass_volume) * milk_per_glass = total_milk → 
  (each_glass_volume - milk_per_glass = 1.5) := 
by 
  sorry

end NUMINAMATH_GPT_chocolate_syrup_per_glass_l911_91174


namespace NUMINAMATH_GPT_average_condition_l911_91140

theorem average_condition (x : ℝ) :
  (1275 + x) / 51 = 80 * x → x = 1275 / 4079 :=
by
  sorry

end NUMINAMATH_GPT_average_condition_l911_91140


namespace NUMINAMATH_GPT_salary_increase_difference_l911_91143

structure Person where
  name : String
  salary : ℕ
  raise_percent : ℕ
  investment_return : ℕ

def hansel := Person.mk "Hansel" 30000 10 5
def gretel := Person.mk "Gretel" 30000 15 4
def rapunzel := Person.mk "Rapunzel" 40000 8 6
def rumpelstiltskin := Person.mk "Rumpelstiltskin" 35000 12 7
def cinderella := Person.mk "Cinderella" 45000 7 8
def jack := Person.mk "Jack" 50000 6 10

def salary_increase (p : Person) : ℕ := p.salary * p.raise_percent / 100
def investment_return (p : Person) : ℕ := salary_increase p * p.investment_return / 100
def total_increase  (p : Person) : ℕ := salary_increase p + investment_return p

def problem_statement : Prop :=
  let hansel_increase := total_increase hansel
  let gretel_increase := total_increase gretel
  let rapunzel_increase := total_increase rapunzel
  let rumpelstiltskin_increase := total_increase rumpelstiltskin
  let cinderella_increase := total_increase cinderella
  let jack_increase := total_increase jack

  let highest_increase := max gretel_increase (max rumpelstiltskin_increase (max cinderella_increase (max rapunzel_increase (max jack_increase hansel_increase))))
  let lowest_increase := min gretel_increase (min rumpelstiltskin_increase (min cinderella_increase (min rapunzel_increase (min jack_increase hansel_increase))))

  highest_increase - lowest_increase = 1530

theorem salary_increase_difference : problem_statement := by
  sorry

end NUMINAMATH_GPT_salary_increase_difference_l911_91143


namespace NUMINAMATH_GPT_benny_has_24_books_l911_91125

def books_sandy : ℕ := 10
def books_tim : ℕ := 33
def total_books : ℕ := 67

def books_benny : ℕ := total_books - (books_sandy + books_tim)

theorem benny_has_24_books : books_benny = 24 := by
  unfold books_benny
  unfold total_books
  unfold books_sandy
  unfold books_tim
  sorry

end NUMINAMATH_GPT_benny_has_24_books_l911_91125


namespace NUMINAMATH_GPT_difference_between_numbers_l911_91133

theorem difference_between_numbers (a b : ℕ) (h1 : a + b = 24365) (h2 : a % 5 = 0) (h3 : (a / 10) = 2 * b) : a - b = 19931 :=
by sorry

end NUMINAMATH_GPT_difference_between_numbers_l911_91133


namespace NUMINAMATH_GPT_sin_cos_sum_l911_91153

theorem sin_cos_sum (α : ℝ) (h : ∃ (c : ℝ), Real.sin α = -1 / c ∧ Real.cos α = 2 / c ∧ c = Real.sqrt 5) :
  Real.sin α + Real.cos α = Real.sqrt 5 / 5 :=
by sorry

end NUMINAMATH_GPT_sin_cos_sum_l911_91153


namespace NUMINAMATH_GPT_fraction_transformation_correct_l911_91136

theorem fraction_transformation_correct
  {a b : ℝ} (hb : b ≠ 0) : 
  (2 * a) / (2 * b) = a / b := by
  sorry

end NUMINAMATH_GPT_fraction_transformation_correct_l911_91136


namespace NUMINAMATH_GPT_min_value_inverse_sum_l911_91182

variable {x y : ℝ}

theorem min_value_inverse_sum (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 4) : (1/x + 1/y) ≥ 1 :=
  sorry

end NUMINAMATH_GPT_min_value_inverse_sum_l911_91182


namespace NUMINAMATH_GPT_total_cost_eq_l911_91196

noncomputable def total_cost : Real :=
  let os_overhead := 1.07
  let cost_per_millisecond := 0.023
  let tape_mounting_cost := 5.35
  let cost_per_megabyte := 0.15
  let cost_per_kwh := 0.02
  let technician_rate_per_hour := 50.0
  let minutes_to_milliseconds := 60000
  let gb_to_mb := 1024

  -- Define program specifics
  let computer_time_minutes := 45.0
  let memory_gb := 3.5
  let electricity_kwh := 2.0
  let technician_time_minutes := 20.0

  -- Calculate costs
  let computer_time_cost := (computer_time_minutes * minutes_to_milliseconds * cost_per_millisecond)
  let memory_cost := (memory_gb * gb_to_mb * cost_per_megabyte)
  let electricity_cost := (electricity_kwh * cost_per_kwh)
  let technician_time_total_hours := (technician_time_minutes * 2 / 60.0)
  let technician_cost := (technician_time_total_hours * technician_rate_per_hour)

  os_overhead + computer_time_cost + tape_mounting_cost + memory_cost + electricity_cost + technician_cost

theorem total_cost_eq : total_cost = 62677.39 := by
  sorry

end NUMINAMATH_GPT_total_cost_eq_l911_91196


namespace NUMINAMATH_GPT_ab_bc_cd_da_leq_1_over_4_l911_91135

theorem ab_bc_cd_da_leq_1_over_4 (a b c d : ℝ) (h : a + b + c + d = 1) : 
  a * b + b * c + c * d + d * a ≤ 1 / 4 := 
sorry

end NUMINAMATH_GPT_ab_bc_cd_da_leq_1_over_4_l911_91135


namespace NUMINAMATH_GPT_find_original_number_l911_91137

theorem find_original_number (x : ℕ) (h1 : 10 * x + 9 + 2 * x = 633) : x = 52 :=
by
  sorry

end NUMINAMATH_GPT_find_original_number_l911_91137


namespace NUMINAMATH_GPT_probability_three_dice_less_than_seven_l911_91124

open Nat

def probability_of_exactly_three_less_than_seven (dice_count : ℕ) (sides : ℕ) (target_faces : ℕ) : ℚ :=
  let p : ℚ := target_faces / sides
  let q : ℚ := 1 - p
  (Nat.choose dice_count (dice_count / 2)) * (p^(dice_count / 2)) * (q^(dice_count / 2))

theorem probability_three_dice_less_than_seven :
  probability_of_exactly_three_less_than_seven 6 12 6 = 5 / 16 := by
  sorry

end NUMINAMATH_GPT_probability_three_dice_less_than_seven_l911_91124


namespace NUMINAMATH_GPT_chain_of_tangent_circles_exists_iff_integer_angle_multiple_l911_91138

noncomputable def angle_between_tangent_circles (R₁ R₂ : Circle) (line : Line) : ℝ :=
-- the definition should specify how we get the angle between the tangent circles
sorry

def n_tangent_circles_exist (R₁ R₂ : Circle) (n : ℕ) : Prop :=
-- the definition should specify the existence of a chain of n tangent circles
sorry

theorem chain_of_tangent_circles_exists_iff_integer_angle_multiple 
  (R₁ R₂ : Circle) (n : ℕ) (line : Line) : 
  n_tangent_circles_exist R₁ R₂ n ↔ ∃ k : ℤ, angle_between_tangent_circles R₁ R₂ line = k * (360 / n) :=
sorry

end NUMINAMATH_GPT_chain_of_tangent_circles_exists_iff_integer_angle_multiple_l911_91138


namespace NUMINAMATH_GPT_julia_shortfall_l911_91116

-- Definitions based on the problem conditions
def rock_and_roll_price : ℕ := 5
def pop_price : ℕ := 10
def dance_price : ℕ := 3
def country_price : ℕ := 7
def quantity : ℕ := 4
def julia_money : ℕ := 75

-- Proof problem: Prove that Julia is short $25
theorem julia_shortfall : (quantity * rock_and_roll_price + quantity * pop_price + quantity * dance_price + quantity * country_price) - julia_money = 25 := by
  sorry

end NUMINAMATH_GPT_julia_shortfall_l911_91116


namespace NUMINAMATH_GPT_polynomial_distinct_positive_roots_l911_91117

theorem polynomial_distinct_positive_roots (a b : ℝ) (P : ℝ → ℝ) (hP : ∀ x, P x = x^3 + a * x^2 + b * x - 1) 
(hroots : ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ x1 > 0 ∧ x2 > 0 ∧ x3 > 0 ∧ P x1 = 0 ∧ P x2 = 0 ∧ P x3 = 0) : 
  P (-1) < -8 := 
by
  sorry

end NUMINAMATH_GPT_polynomial_distinct_positive_roots_l911_91117


namespace NUMINAMATH_GPT_angle_in_second_quadrant_l911_91197

theorem angle_in_second_quadrant (α : ℝ) (h₁ : -2 * Real.pi < α) (h₂ : α < -Real.pi) : 
  α = -4 → (α > -3 * Real.pi / 2 ∧ α < -Real.pi / 2) :=
by
  intros hα
  sorry

end NUMINAMATH_GPT_angle_in_second_quadrant_l911_91197


namespace NUMINAMATH_GPT_cost_of_toast_l911_91164

theorem cost_of_toast (egg_cost : ℕ) (toast_cost : ℕ)
  (dale_toasts : ℕ) (dale_eggs : ℕ)
  (andrew_toasts : ℕ) (andrew_eggs : ℕ)
  (total_cost : ℕ)
  (h1 : egg_cost = 3)
  (h2 : dale_toasts = 2)
  (h3 : dale_eggs = 2)
  (h4 : andrew_toasts = 1)
  (h5 : andrew_eggs = 2)
  (h6 : 2 * toast_cost + dale_eggs * egg_cost 
        + andrew_toasts * toast_cost + andrew_eggs * egg_cost = total_cost) :
  total_cost = 15 → toast_cost = 1 :=
by
  -- Proof not needed
  sorry

end NUMINAMATH_GPT_cost_of_toast_l911_91164


namespace NUMINAMATH_GPT_convert_deg_to_min_compare_negatives_l911_91155

theorem convert_deg_to_min : (0.3 : ℝ) * 60 = 18 :=
by sorry

theorem compare_negatives : -2 > -3 :=
by sorry

end NUMINAMATH_GPT_convert_deg_to_min_compare_negatives_l911_91155


namespace NUMINAMATH_GPT_sequence_infinite_divisibility_l911_91159

theorem sequence_infinite_divisibility :
  ∃ (u : ℕ → ℤ), (∀ n, u (n + 2) = u (n + 1) ^ 2 - u n) ∧ u 1 = 39 ∧ u 2 = 45 ∧ (∀ N, ∃ k ≥ N, 1986 ∣ u k) := 
by
  sorry

end NUMINAMATH_GPT_sequence_infinite_divisibility_l911_91159


namespace NUMINAMATH_GPT_solve_inequality_l911_91169

def otimes (x y : ℝ) : ℝ := x * (1 - y)

theorem solve_inequality (x : ℝ) : (otimes (x-2) (x+2) < 2) ↔ x ∈ Set.Iio 0 ∪ Set.Ioi 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l911_91169


namespace NUMINAMATH_GPT_two_numbers_solution_l911_91186

noncomputable def a := 8 + Real.sqrt 58
noncomputable def b := 8 - Real.sqrt 58

theorem two_numbers_solution : 
  (Real.sqrt (a * b) = Real.sqrt 6) ∧ ((2 * a * b) / (a + b) = 3 / 4) → 
  (a = 8 + Real.sqrt 58 ∧ b = 8 - Real.sqrt 58) ∨ (a = 8 - Real.sqrt 58 ∧ b = 8 + Real.sqrt 58) := 
by
  sorry

end NUMINAMATH_GPT_two_numbers_solution_l911_91186


namespace NUMINAMATH_GPT_total_amount_spent_by_jim_is_50_l911_91113

-- Definitions for conditions
def cost_per_gallon_nc : ℝ := 2.00  -- Cost per gallon in North Carolina
def gallons_nc : ℕ := 10  -- Gallons bought in North Carolina
def additional_cost_per_gallon_va : ℝ := 1.00  -- Additional cost per gallon in Virginia
def gallons_va : ℕ := 10  -- Gallons bought in Virginia

-- Definition for total cost in North Carolina
def total_cost_nc : ℝ := gallons_nc * cost_per_gallon_nc

-- Definition for cost per gallon in Virginia
def cost_per_gallon_va : ℝ := cost_per_gallon_nc + additional_cost_per_gallon_va

-- Definition for total cost in Virginia
def total_cost_va : ℝ := gallons_va * cost_per_gallon_va

-- Definition for total amount spent
def total_spent : ℝ := total_cost_nc + total_cost_va

-- Theorem to prove
theorem total_amount_spent_by_jim_is_50 : total_spent = 50.00 :=
by
  -- Place proof here
  sorry

end NUMINAMATH_GPT_total_amount_spent_by_jim_is_50_l911_91113


namespace NUMINAMATH_GPT_solve_arithmetic_sequence_sum_l911_91184

noncomputable def arithmetic_sequence_sum : ℕ :=
  let a : ℕ := 3
  let b : ℕ := 10
  let c : ℕ := 17
  let e : ℕ := 32
  let d := b - a
  let c_term := c + d
  let d_term := c_term + d
  c_term + d_term

theorem solve_arithmetic_sequence_sum : arithmetic_sequence_sum = 55 :=
by
  sorry

end NUMINAMATH_GPT_solve_arithmetic_sequence_sum_l911_91184


namespace NUMINAMATH_GPT_leadership_selection_ways_l911_91101

theorem leadership_selection_ways (M : ℕ) (chiefs : ℕ) (supporting_chiefs : ℕ) (officers_per_supporting_chief : ℕ) 
  (M_eq : M = 15) (chiefs_eq : chiefs = 1) (supporting_chiefs_eq : supporting_chiefs = 2) 
  (officers_eq : officers_per_supporting_chief = 3) : 
  (M * (M - 1) * (M - 2) * (Nat.choose (M - 3) officers_per_supporting_chief) * (Nat.choose (M - 6) officers_per_supporting_chief)) = 3243240 := by
  simp [M_eq, chiefs_eq, supporting_chiefs_eq, officers_eq]
  norm_num
  sorry

end NUMINAMATH_GPT_leadership_selection_ways_l911_91101


namespace NUMINAMATH_GPT_monotonically_increasing_interval_l911_91177

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.sqrt 3 * Real.cos x

theorem monotonically_increasing_interval : 
  ∀ x ∈ Set.Icc (-Real.pi) 0, 
  x ∈ Set.Icc (-Real.pi/6) 0 ↔ deriv f x = 0 := sorry

end NUMINAMATH_GPT_monotonically_increasing_interval_l911_91177


namespace NUMINAMATH_GPT_unique_function_l911_91131

theorem unique_function (f : ℝ → ℝ) 
  (H : ∀ (x y : ℝ), f (f x + 9 * y) = f y + 9 * x + 24 * y) : 
  ∀ x : ℝ, f x = 3 * x :=
by 
  sorry

end NUMINAMATH_GPT_unique_function_l911_91131


namespace NUMINAMATH_GPT_percentage_is_26_53_l911_91118

noncomputable def percentage_employees_with_six_years_or_more (y: ℝ) : ℝ :=
  let total_employees := 10*y + 4*y + 6*y + 5*y + 8*y + 3*y + 5*y + 4*y + 2*y + 2*y
  let employees_with_six_years_or_more := 5*y + 4*y + 2*y + 2*y
  (employees_with_six_years_or_more / total_employees) * 100

theorem percentage_is_26_53 (y: ℝ) (hy: y ≠ 0): percentage_employees_with_six_years_or_more y = 26.53 :=
by
  sorry

end NUMINAMATH_GPT_percentage_is_26_53_l911_91118


namespace NUMINAMATH_GPT_starting_number_range_l911_91109

theorem starting_number_range (n : ℕ) (h₁: ∀ m : ℕ, (m > n) → (m ≤ 50) → (m = 55) → True) : n = 54 :=
sorry

end NUMINAMATH_GPT_starting_number_range_l911_91109


namespace NUMINAMATH_GPT_tate_total_years_l911_91183

-- Define the conditions
def high_school_years : Nat := 3
def gap_years : Nat := 2
def bachelor_years : Nat := 2 * high_school_years
def certification_years : Nat := 1
def work_experience_years : Nat := 1
def master_years : Nat := bachelor_years / 2
def phd_years : Nat := 3 * (high_school_years + bachelor_years + master_years)

-- Define the total years Tate spent
def total_years : Nat :=
  high_school_years + gap_years +
  bachelor_years + certification_years +
  work_experience_years + master_years + phd_years

-- State the theorem
theorem tate_total_years : total_years = 52 := by
  sorry

end NUMINAMATH_GPT_tate_total_years_l911_91183


namespace NUMINAMATH_GPT_eldest_age_l911_91149

theorem eldest_age (A B C : ℕ) (x : ℕ) 
  (h1 : A = 5 * x)
  (h2 : B = 7 * x)
  (h3 : C = 8 * x)
  (h4 : (5 * x - 7) + (7 * x - 7) + (8 * x - 7) = 59) :
  C = 32 := 
by 
  sorry

end NUMINAMATH_GPT_eldest_age_l911_91149


namespace NUMINAMATH_GPT_jogging_distance_apart_l911_91158

theorem jogging_distance_apart
  (alice_speed : ℝ)
  (bob_speed : ℝ)
  (time_in_minutes : ℝ)
  (distance_apart : ℝ)
  (h1 : alice_speed = 1 / 12)
  (h2 : bob_speed = 3 / 40)
  (h3 : time_in_minutes = 120)
  (h4 : distance_apart = alice_speed * time_in_minutes + bob_speed * time_in_minutes) :
  distance_apart = 19 := by
  sorry

end NUMINAMATH_GPT_jogging_distance_apart_l911_91158


namespace NUMINAMATH_GPT_math_problem_l911_91126

open Real

variables {a b c d e f : ℝ}

theorem math_problem 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hf : f > 0)
  (hcond : abs (sqrt (a * b) - sqrt (c * d)) ≤ 2) :
  (e / a + b / e) * (e / c + d / e) ≥ (f / a - b) * (d - f / c) :=
sorry

end NUMINAMATH_GPT_math_problem_l911_91126


namespace NUMINAMATH_GPT_sequence_formula_l911_91179

def seq (n : ℕ) : ℕ := 
  match n with
  | 0     => 1
  | (n+1) => 2 * seq n + 3

theorem sequence_formula (n : ℕ) (h1 : n ≥ 1) : 
  seq n = 2^n + 1 - 3 :=
sorry

end NUMINAMATH_GPT_sequence_formula_l911_91179


namespace NUMINAMATH_GPT_simplify_and_evaluate_at_x_eq_4_l911_91120

noncomputable def simplify_and_evaluate (x : ℚ) : ℚ :=
  (x - 1 - (3 / (x + 1))) / ((x^2 - 2*x) / (x + 1))

theorem simplify_and_evaluate_at_x_eq_4 : simplify_and_evaluate 4 = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_at_x_eq_4_l911_91120


namespace NUMINAMATH_GPT_opposite_of_2021_l911_91165

theorem opposite_of_2021 : -(2021) = -2021 := 
sorry

end NUMINAMATH_GPT_opposite_of_2021_l911_91165


namespace NUMINAMATH_GPT_complement_A_in_U_l911_91141

def U : Set ℕ := {1, 3, 5, 7, 9}
def A : Set ℕ := {1, 5, 7}

theorem complement_A_in_U : (U \ A) = {3, 9} := 
by sorry

end NUMINAMATH_GPT_complement_A_in_U_l911_91141


namespace NUMINAMATH_GPT_probability_same_class_l911_91198

-- Define the problem conditions
def num_classes : ℕ := 3
def total_scenarios : ℕ := num_classes * num_classes
def same_class_scenarios : ℕ := num_classes

-- Formulate the proof problem
theorem probability_same_class :
  (same_class_scenarios : ℚ) / total_scenarios = 1 / 3 :=
sorry

end NUMINAMATH_GPT_probability_same_class_l911_91198


namespace NUMINAMATH_GPT_largest_rectangle_area_l911_91173

theorem largest_rectangle_area (x y : ℝ) (h : 2 * x + 2 * y = 60) : x * y ≤ 225 :=
by
  sorry

end NUMINAMATH_GPT_largest_rectangle_area_l911_91173


namespace NUMINAMATH_GPT_committee_count_l911_91146

theorem committee_count :
  let total_owners := 30
  let not_willing := 3
  let eligible_owners := total_owners - not_willing
  let committee_size := 5
  eligible_owners.choose committee_size = 65780 := by
  let total_owners := 30
  let not_willing := 3
  let eligible_owners := total_owners - not_willing
  let committee_size := 5
  have lean_theorem : eligible_owners.choose committee_size = 65780 := sorry
  exact lean_theorem

end NUMINAMATH_GPT_committee_count_l911_91146


namespace NUMINAMATH_GPT_lemon_ratio_l911_91129

variable (Levi Jayden Eli Ian : ℕ)

theorem lemon_ratio (h1: Levi = 5)
    (h2: Jayden = Levi + 6)
    (h3: Jayden = Eli / 3)
    (h4: Levi + Jayden + Eli + Ian = 115) :
    Eli = Ian / 2 :=
by
  sorry

end NUMINAMATH_GPT_lemon_ratio_l911_91129


namespace NUMINAMATH_GPT_new_students_count_l911_91152

theorem new_students_count (O N : ℕ) (avg_class_age avg_new_students_age avg_decrease original_strength : ℕ)
  (h1 : avg_class_age = 40)
  (h2 : avg_new_students_age = 32)
  (h3 : avg_decrease = 4)
  (h4 : original_strength = 8)
  (total_age_class : ℕ := avg_class_age * original_strength)
  (new_avg_age : ℕ := avg_class_age - avg_decrease)
  (total_age_new_students : ℕ := avg_new_students_age * N)
  (total_students : ℕ := original_strength + N)
  (new_total_age : ℕ := total_age_class + total_age_new_students)
  (new_avg_class_age : ℕ := new_total_age / total_students)
  (h5 : new_avg_class_age = new_avg_age) : N = 8 :=
by
  sorry

end NUMINAMATH_GPT_new_students_count_l911_91152


namespace NUMINAMATH_GPT_problem_1_problem_2_l911_91172

-- Statements for our proof problems
theorem problem_1 (a b : ℝ) : a^2 + b^2 ≥ 2 * (2 * a - b) - 5 :=
sorry

theorem problem_2 (a b : ℝ) :
  a^a * b^b ≥ (a * b)^((a + b) / 2) ∧ (a = b ↔ a^a * b^b = (a * b)^((a + b) / 2)) :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l911_91172


namespace NUMINAMATH_GPT_jennifer_fruits_left_l911_91100

-- Definitions based on the conditions
def pears : ℕ := 15
def oranges : ℕ := 30
def apples : ℕ := 2 * pears
def cherries : ℕ := oranges / 2
def grapes : ℕ := 3 * apples
def pineapples : ℕ := pears + oranges + apples + cherries + grapes

-- Definitions for the number of fruits given to the sister
def pears_given : ℕ := 3
def oranges_given : ℕ := 5
def apples_given : ℕ := 5
def cherries_given : ℕ := 7
def grapes_given : ℕ := 3

-- Calculations based on the conditions for what's left after giving fruits
def pears_left : ℕ := pears - pears_given
def oranges_left : ℕ := oranges - oranges_given
def apples_left : ℕ := apples - apples_given
def cherries_left : ℕ := cherries - cherries_given
def grapes_left : ℕ := grapes - grapes_given

def remaining_pineapples : ℕ := pineapples - (pineapples / 2)

-- Total number of fruits left
def total_fruits_left : ℕ := pears_left + oranges_left + apples_left + cherries_left + grapes_left + remaining_pineapples

-- Theorem statement
theorem jennifer_fruits_left : total_fruits_left = 247 :=
by
  -- The detailed proof would go here
  sorry

end NUMINAMATH_GPT_jennifer_fruits_left_l911_91100


namespace NUMINAMATH_GPT_rectangle_perimeter_l911_91151

variable (a b : ℕ)

theorem rectangle_perimeter (h1 : a ≠ b) (h2 : ab = 8 * (a + b)) : 
  2 * (a + b) = 66 := 
sorry

end NUMINAMATH_GPT_rectangle_perimeter_l911_91151


namespace NUMINAMATH_GPT_abs_sum_condition_l911_91144

theorem abs_sum_condition (a b : ℝ) (h₁ : |a| = 2) (h₂ : b = -1) : |a + b| = 1 ∨ |a + b| = 3 :=
by
  sorry

end NUMINAMATH_GPT_abs_sum_condition_l911_91144


namespace NUMINAMATH_GPT_ring_width_l911_91171

noncomputable def innerCircumference : ℝ := 352 / 7
noncomputable def outerCircumference : ℝ := 528 / 7

noncomputable def radius (C : ℝ) : ℝ := C / (2 * Real.pi)

theorem ring_width :
  let r_inner := radius innerCircumference
  let r_outer := radius outerCircumference
  r_outer - r_inner = 4 :=
by
  -- Definitions for inner and outer radius
  let r_inner := radius innerCircumference
  let r_outer := radius outerCircumference
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_ring_width_l911_91171


namespace NUMINAMATH_GPT_jade_handled_84_transactions_l911_91108

def Mabel_transactions : ℕ := 90

def Anthony_transactions (mabel : ℕ) : ℕ := mabel + mabel / 10

def Cal_transactions (anthony : ℕ) : ℕ := (2 * anthony) / 3

def Jade_transactions (cal : ℕ) : ℕ := cal + 18

theorem jade_handled_84_transactions :
  Jade_transactions (Cal_transactions (Anthony_transactions Mabel_transactions)) = 84 := 
sorry

end NUMINAMATH_GPT_jade_handled_84_transactions_l911_91108


namespace NUMINAMATH_GPT_correct_calculated_value_l911_91148

theorem correct_calculated_value (x : ℤ) (h : x - 749 = 280) : x + 479 = 1508 :=
by 
  sorry

end NUMINAMATH_GPT_correct_calculated_value_l911_91148


namespace NUMINAMATH_GPT_area_triangle_QDA_l911_91187

-- Define the points
def Q : ℝ × ℝ := (0, 15)
def A (q : ℝ) : ℝ × ℝ := (q, 15)
def D (p : ℝ) : ℝ × ℝ := (0, p)

-- Define the conditions
variable (q : ℝ) (p : ℝ)
variable (hq : q > 0) (hp : p < 15)

-- Theorem stating the area of the triangle QDA in terms of q and p
theorem area_triangle_QDA : 
  1 / 2 * q * (15 - p) = 1 / 2 * q * (15 - p) :=
by sorry

end NUMINAMATH_GPT_area_triangle_QDA_l911_91187


namespace NUMINAMATH_GPT_diameter_in_scientific_notation_l911_91190

def diameter : ℝ := 0.00000011
def scientific_notation (d : ℝ) : Prop := d = 1.1e-7

theorem diameter_in_scientific_notation : scientific_notation diameter :=
by
  sorry

end NUMINAMATH_GPT_diameter_in_scientific_notation_l911_91190


namespace NUMINAMATH_GPT_multiplier_for_second_part_l911_91163

theorem multiplier_for_second_part {x y k : ℝ} (h1 : x + y = 52) (h2 : 10 * x + k * y = 780) (hy : y = 30.333333333333332) (hx : x = 21.666666666666668) :
  k = 18.571428571428573 :=
by
  sorry

end NUMINAMATH_GPT_multiplier_for_second_part_l911_91163


namespace NUMINAMATH_GPT_smallest_angle_terminal_side_l911_91119

theorem smallest_angle_terminal_side (θ : ℝ) (H : θ = 2011) :
  ∃ φ : ℝ, 0 ≤ φ ∧ φ < 360 ∧ (∃ k : ℤ, φ = θ - 360 * k) ∧ φ = 211 :=
by
  sorry

end NUMINAMATH_GPT_smallest_angle_terminal_side_l911_91119


namespace NUMINAMATH_GPT_f_2015_equals_2_l911_91195

noncomputable def f : ℝ → ℝ :=
sorry

theorem f_2015_equals_2 (f_even : ∀ x : ℝ, f (-x) = f x)
    (f_shift : ∀ x : ℝ, f (-x) = f (2 + x))
    (f_log : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → f x = Real.log (3 * x + 1) / Real.log 2) :
    f 2015 = 2 :=
sorry

end NUMINAMATH_GPT_f_2015_equals_2_l911_91195


namespace NUMINAMATH_GPT_polynomial_divisibility_l911_91106

def poly1 (x : ℝ) (k : ℝ) : ℝ := 3*x^3 - 9*x^2 + k*x - 12

theorem polynomial_divisibility (k : ℝ) :
  (∀ (x : ℝ), poly1 x k = (x - 3) * (3*x^2 + 4)) → (poly1 3 k = 0) := sorry

end NUMINAMATH_GPT_polynomial_divisibility_l911_91106


namespace NUMINAMATH_GPT_trigonometric_identity_l911_91175

theorem trigonometric_identity (α : ℝ) (h : Real.tan (Real.pi + α) = 2) :
  (Real.sin (α - Real.pi) + Real.cos (Real.pi - α)) / (Real.sin (Real.pi + α) - Real.cos (Real.pi - α)) = 3 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l911_91175


namespace NUMINAMATH_GPT_solve_inequality_l911_91189

theorem solve_inequality (x : ℝ) : 
  (0 < (x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6))) ↔ 
  (x < 2) ∨ (4 < x ∧ x < 5) ∨ (6 < x) :=
by 
  sorry

end NUMINAMATH_GPT_solve_inequality_l911_91189


namespace NUMINAMATH_GPT_set_union_intersection_l911_91194

-- Definitions
def A : Set ℤ := {-1, 0}
def B : Set ℤ := {0, 1}
def C : Set ℤ := {1, 2}

-- Theorem statement
theorem set_union_intersection : (A ∩ B ∪ C) = {0, 1, 2} :=
by
  sorry

end NUMINAMATH_GPT_set_union_intersection_l911_91194


namespace NUMINAMATH_GPT_matrix_inverse_l911_91199

variable (N : Matrix (Fin 2) (Fin 2) ℚ) 
variable (I : Matrix (Fin 2) (Fin 2) ℚ)
variable (c d : ℚ)

def M1 : Matrix (Fin 2) (Fin 2) ℚ := ![![3, 0], ![2, -4]]

def M2 : Matrix (Fin 2) (Fin 2) ℚ := ![![1, 0], ![0, 1]]

theorem matrix_inverse (hN : N = M1) 
                       (hI : I = M2) 
                       (hc : c = 1/12) 
                       (hd : d = 1/12) :
                       N⁻¹ = c • N + d • I := by
  sorry

end NUMINAMATH_GPT_matrix_inverse_l911_91199


namespace NUMINAMATH_GPT_probability_A_B_C_adjacent_l911_91115

theorem probability_A_B_C_adjacent (students : Fin 5 → Prop) (A B C : Fin 5) :
  (students A ∧ students B ∧ students C) →
  (∃ n m : ℕ, n = 48 ∧ m = 12 ∧ m / n = (1 : ℚ) / 4) :=
by
  sorry

end NUMINAMATH_GPT_probability_A_B_C_adjacent_l911_91115


namespace NUMINAMATH_GPT_y_percent_of_x_l911_91147

theorem y_percent_of_x (x y : ℝ) (h : 0.60 * (x - y) = 0.20 * (x + y)) : y / x = 0.5 :=
sorry

end NUMINAMATH_GPT_y_percent_of_x_l911_91147


namespace NUMINAMATH_GPT_squirrels_acorns_l911_91134

theorem squirrels_acorns (x : ℕ) : 
    (5 * (x - 15) = 575) → 
    x = 130 := 
by 
  intros h
  sorry

end NUMINAMATH_GPT_squirrels_acorns_l911_91134


namespace NUMINAMATH_GPT_B_subset_A_l911_91180

variable {α : Type*}
variable (A B : Set α)

def A_def : Set ℝ := { x | x ≥ 1 }
def B_def : Set ℝ := { x | x > 2 }

theorem B_subset_A : B_def ⊆ A_def :=
sorry

end NUMINAMATH_GPT_B_subset_A_l911_91180


namespace NUMINAMATH_GPT_inequality_does_not_hold_l911_91157

theorem inequality_does_not_hold (x y : ℝ) (h : x > y) : ¬ (-3 * x > -3 * y) :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_does_not_hold_l911_91157


namespace NUMINAMATH_GPT_sectors_containing_all_numbers_l911_91162

theorem sectors_containing_all_numbers (n : ℕ) (h : 0 < n) :
  ∃ (s : Finset (Fin (2 * n))), (s.card = n) ∧ (∀ i : Fin n, ∃ j : Fin (2 * n), j ∈ s ∧ (j.val % n) + 1 = i.val) :=
  sorry

end NUMINAMATH_GPT_sectors_containing_all_numbers_l911_91162


namespace NUMINAMATH_GPT_vertical_asymptote_l911_91188

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := (x^2 - x + c) / (x^2 - 6*x + 8)

theorem vertical_asymptote (c : ℝ) :
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 4 → ((x^2 - x + c) ≠ 0)) ∨
  (∀ x : ℝ, ((x^2 - x + c) = 0) ↔ (x = 2) ∨ (x = 4)) →
  c = -2 ∨ c = -12 :=
sorry

end NUMINAMATH_GPT_vertical_asymptote_l911_91188


namespace NUMINAMATH_GPT_pair_exists_l911_91132

def exists_pair (a b : ℕ → ℕ) : Prop :=
  ∃ p q : ℕ, p < q ∧ a p ≤ a q ∧ b p ≤ b q

theorem pair_exists (a b : ℕ → ℕ) : exists_pair a b :=
sorry

end NUMINAMATH_GPT_pair_exists_l911_91132


namespace NUMINAMATH_GPT_num_digits_abc_l911_91181

theorem num_digits_abc (a b c : ℕ) (n : ℕ) (h_a : 10^(n-1) ≤ a ∧ a < 10^n) (h_b : 10^(n-1) ≤ b ∧ b < 10^n) (h_c : 10^(n-1) ≤ c ∧ c < 10^n) :
  ¬ ((Int.natAbs ((10^(n-1) : ℕ) * (10^(n-1) : ℕ) * (10^(n-1) : ℕ)) + 1 = 3*n) ∧
     (Int.natAbs ((10^(n-1) : ℕ) * (10^(n-1) : ℕ) * (10^(n-1) : ℕ)) + 1 = 3*n - 1) ∧
     (Int.natAbs ((10^(n-1) : ℕ) * (10^(n-1) : ℕ) * (10^(n-1) : ℕ)) + 1 = 3*n - 2)) :=
sorry

end NUMINAMATH_GPT_num_digits_abc_l911_91181


namespace NUMINAMATH_GPT_smallest_possible_value_abs_sum_l911_91122

theorem smallest_possible_value_abs_sum :
  ∃ x : ℝ, (∀ y : ℝ, abs (y + 3) + abs (y + 5) + abs (y + 7) ≥ abs (x + 3) + abs (x + 5) + abs (x + 7))
  ∧ (abs (x + 3) + abs (x + 5) + abs (x + 7) = 4) := by
  sorry

end NUMINAMATH_GPT_smallest_possible_value_abs_sum_l911_91122


namespace NUMINAMATH_GPT_arithmetic_sequence_properties_l911_91176

noncomputable def general_term_formula (a₁ : ℕ) (S₃ : ℕ) (n : ℕ) (d : ℕ) : ℕ :=
  a₁ + (n - 1) * d

noncomputable def sum_of_double_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  (2 * (a₁ + (n - 1) * d)) * n / 2

theorem arithmetic_sequence_properties
  (a₁ : ℕ) (S₃ : ℕ)
  (h₁ : a₁ = 2)
  (h₂ : S₃ = 9) :
  general_term_formula a₁ S₃ n (a₁ + 2 * ((S₃ - 3 * a₁) / 3)) = n + 1 ∧
  sum_of_double_sequence a₁ (a₁ + 2 * ((S₃ - 3 * a₁) / 3)) n = 2^(n+2) - 4 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_properties_l911_91176


namespace NUMINAMATH_GPT_rectangle_area_inscribed_circle_l911_91161

theorem rectangle_area_inscribed_circle (r : ℝ) (h : r = 7) (ratio : ℝ) (hratio : ratio = 3) : 
  (2 * r) * (ratio * (2 * r)) = 588 :=
by
  rw [h, hratio]
  sorry

end NUMINAMATH_GPT_rectangle_area_inscribed_circle_l911_91161


namespace NUMINAMATH_GPT_atomic_weight_S_is_correct_l911_91114

-- Conditions
def molecular_weight_BaSO4 : Real := 233
def atomic_weight_Ba : Real := 137.33
def atomic_weight_O : Real := 16
def num_O_in_BaSO4 : Nat := 4

-- Definition of total weight of Ba and O
def total_weight_Ba_O := atomic_weight_Ba + num_O_in_BaSO4 * atomic_weight_O

-- Expected atomic weight of S
def atomic_weight_S : Real := molecular_weight_BaSO4 - total_weight_Ba_O

-- Theorem to prove that the atomic weight of S is 31.67
theorem atomic_weight_S_is_correct : atomic_weight_S = 31.67 := by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_atomic_weight_S_is_correct_l911_91114
