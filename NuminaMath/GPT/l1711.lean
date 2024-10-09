import Mathlib

namespace smallest_x_value_l1711_171125

theorem smallest_x_value : ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ (3 : ℚ) / 4 = y / (250 + x) ∧ x = 2 := by
  sorry

end smallest_x_value_l1711_171125


namespace rectangular_to_polar_coordinates_l1711_171185

theorem rectangular_to_polar_coordinates :
  ∃ r θ, (r > 0) ∧ (0 ≤ θ ∧ θ < 2 * Real.pi) ∧ (r, θ) = (5, 7 * Real.pi / 4) :=
by
  sorry

end rectangular_to_polar_coordinates_l1711_171185


namespace triangle_inequality_range_l1711_171151

theorem triangle_inequality_range {a b c : ℝ} (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  1 ≤ (a^2 + b^2 + c^2) / (a * b + b * c + c * a) ∧ (a^2 + b^2 + c^2) / (a * b + b * c + c * a) < 2 := 
by 
  sorry

end triangle_inequality_range_l1711_171151


namespace largest_integer_base7_four_digits_l1711_171136

theorem largest_integer_base7_four_digits :
  ∃ M : ℕ, (∀ m : ℕ, 7^3 ≤ m^2 ∧ m^2 < 7^4 → m ≤ M) ∧ M = 48 :=
sorry

end largest_integer_base7_four_digits_l1711_171136


namespace equation_solutions_l1711_171103

noncomputable def count_solutions (a : ℝ) : ℕ :=
  if 0 < a ∧ a <= 1 ∨ a = Real.exp (1 / Real.exp 1) then 1
  else if 1 < a ∧ a < Real.exp (1 / Real.exp 1) then 2
  else if a > Real.exp (1 / Real.exp 1) then 0
  else 0

theorem equation_solutions (a : ℝ) (h₀ : 0 < a) :
  (∃! x : ℝ, a^x = x) ↔ count_solutions a = 1 ∨ count_solutions a = 2 ∨ count_solutions a = 0 := sorry

end equation_solutions_l1711_171103


namespace final_answer_for_m_l1711_171100

noncomputable def proof_condition_1 (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

noncomputable def proof_condition_2 (x y : ℝ) : Prop :=
  x + 2*y - 3 = 0

noncomputable def proof_condition_perpendicular (x1 y1 x2 y2 : ℝ) : Prop :=
  x1*x2 + y1*y2 = 0

theorem final_answer_for_m :
  (∀ (x y m : ℝ), proof_condition_1 x y m) →
  (∀ (x y : ℝ), proof_condition_2 x y) →
  (∀ (x1 y1 x2 y2 : ℝ), proof_condition_perpendicular x1 y1 x2 y2) →
  m = 12 / 5 :=
sorry

end final_answer_for_m_l1711_171100


namespace prove_a_lt_neg_one_l1711_171117

variable {f : ℝ → ℝ} (a : ℝ)

-- Conditions:
-- 1. f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- 2. f has a period of 3
def has_period_three (f : ℝ → ℝ) : Prop := ∀ x, f (x + 3) = f x

-- 3. f(1) > 1
def f_one_gt_one (f : ℝ → ℝ) : Prop := f 1 > 1

-- 4. f(2) = a
def f_two_eq_a (f : ℝ → ℝ) (a : ℝ) : Prop := f 2 = a

-- Proof statement:
theorem prove_a_lt_neg_one (h1 : is_odd_function f) (h2 : has_period_three f)
  (h3 : f_one_gt_one f) (h4 : f_two_eq_a f a) : a < -1 :=
  sorry

end prove_a_lt_neg_one_l1711_171117


namespace garden_perimeter_l1711_171171

theorem garden_perimeter
  (a b : ℝ)
  (h1: a^2 + b^2 = 225)
  (h2: a * b = 54) :
  2 * (a + b) = 2 * Real.sqrt 333 :=
by
  sorry

end garden_perimeter_l1711_171171


namespace lines_parallel_if_perpendicular_to_same_plane_l1711_171186

variables (m n : Line) (α : Plane)

-- Define conditions using Lean's logical constructs
def perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry -- This would define the condition
def parallel_lines (l1 l2 : Line) : Prop := sorry -- This would define the condition

-- The statement to prove
theorem lines_parallel_if_perpendicular_to_same_plane 
  (h1 : perpendicular_to_plane m α) 
  (h2 : perpendicular_to_plane n α) : 
  parallel_lines m n :=
sorry

end lines_parallel_if_perpendicular_to_same_plane_l1711_171186


namespace categorize_numbers_l1711_171137

def numbers : Set (Rat) := {-16, 0.04, 1/2, -2/3, 25, 0, -3.6, -0.3, 4/3}

def is_integer (x : Rat) : Prop := ∃ z : Int, x = z
def is_fraction (x : Rat) : Prop := ∃ (p q : Int), q ≠ 0 ∧ x = p / q
def is_negative (x : Rat) : Prop := x < 0

def integers (s : Set Rat) : Set Rat := {x | x ∈ s ∧ is_integer x}
def fractions (s : Set Rat) : Set Rat := {x | x ∈ s ∧ is_fraction x}
def negative_rationals (s : Set Rat) : Set Rat := {x | x ∈ s ∧ is_fraction x ∧ is_negative x}

theorem categorize_numbers :
  integers numbers = {-16, 25, 0} ∧
  fractions numbers = {0.04, 1/2, -2/33, -3.6, -0.3, 4/3} ∧
  negative_rationals numbers = {-16, -2/3, -3.6, -0.3} :=
  sorry

end categorize_numbers_l1711_171137


namespace percent_exceed_l1711_171140

theorem percent_exceed (x y : ℝ) (h : x = 0.75 * y) : ((y - x) / x) * 100 = 33.33 :=
by
  sorry

end percent_exceed_l1711_171140


namespace set_inter_complement_l1711_171104

open Set

variable {α : Type*}
variable (U A B : Set α)

theorem set_inter_complement (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)
  (hU : U = {1, 2, 3, 4, 5})
  (hA : A = {1, 2, 3})
  (hB : B = {1, 4}) :
  ((U \ A) ∩ B) = {4} := 
by
  sorry

end set_inter_complement_l1711_171104


namespace ratio_of_earnings_l1711_171122

theorem ratio_of_earnings (K V S : ℕ) (h1 : K + 30 = V) (h2 : V = 84) (h3 : S = 216) : S / K = 4 :=
by
  -- proof goes here
  sorry

end ratio_of_earnings_l1711_171122


namespace fractional_part_sum_leq_l1711_171167

noncomputable def fractional_part (z : ℝ) : ℝ :=
  z - (⌊z⌋ : ℝ)

theorem fractional_part_sum_leq (x y : ℝ) :
  fractional_part (x + y) ≤ fractional_part x + fractional_part y :=
by
  sorry

end fractional_part_sum_leq_l1711_171167


namespace find_f_neg3_l1711_171177

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom periodic_function : ∀ x : ℝ, f (x + 4) = f x
axiom sum_equation : f 1 + f 2 + f 3 + f 4 + f 5 = 6

theorem find_f_neg3 : f (-3) = 6 := by
  sorry

end find_f_neg3_l1711_171177


namespace jewelry_store_total_cost_l1711_171120

-- Definitions for given conditions
def necklace_capacity : Nat := 12
def current_necklaces : Nat := 5
def ring_capacity : Nat := 30
def current_rings : Nat := 18
def bracelet_capacity : Nat := 15
def current_bracelets : Nat := 8

def necklace_cost : Nat := 4
def ring_cost : Nat := 10
def bracelet_cost : Nat := 5

-- Definition for number of items needed to fill displays
def needed_necklaces : Nat := necklace_capacity - current_necklaces
def needed_rings : Nat := ring_capacity - current_rings
def needed_bracelets : Nat := bracelet_capacity - current_bracelets

-- Definition for cost to fill each type of jewelry
def cost_necklaces : Nat := needed_necklaces * necklace_cost
def cost_rings : Nat := needed_rings * ring_cost
def cost_bracelets : Nat := needed_bracelets * bracelet_cost

-- Total cost to fill the displays
def total_cost : Nat := cost_necklaces + cost_rings + cost_bracelets

-- Proof statement
theorem jewelry_store_total_cost : total_cost = 183 := by
  sorry

end jewelry_store_total_cost_l1711_171120


namespace find_income_of_deceased_l1711_171188
noncomputable def income_of_deceased_member 
  (members_before : ℕ) (avg_income_before : ℕ) 
  (members_after : ℕ) (avg_income_after : ℕ) : ℕ :=
  (members_before * avg_income_before) - (members_after * avg_income_after)

theorem find_income_of_deceased 
  (members_before avg_income_before members_after avg_income_after : ℕ) :
  income_of_deceased_member 4 840 3 650 = 1410 :=
by
  -- Problem claims income_of_deceased_member = Income before - Income after
  sorry

end find_income_of_deceased_l1711_171188


namespace cost_per_container_is_21_l1711_171145

-- Define the given problem conditions as Lean statements.

--  Let w be the number of weeks represented by 210 days.
def number_of_weeks (days: ℕ) : ℕ := days / 7
def weeks : ℕ := number_of_weeks 210

-- Let p be the total pounds of litter used over the number of weeks.
def pounds_per_week : ℕ := 15
def total_litter_pounds (weeks: ℕ) : ℕ := weeks * pounds_per_week
def total_pounds : ℕ := total_litter_pounds weeks

-- Let c be the number of 45-pound containers needed for the total pounds of litter.
def pounds_per_container : ℕ := 45
def number_of_containers (total_pounds pounds_per_container: ℕ) : ℕ := total_pounds / pounds_per_container
def containers : ℕ := number_of_containers total_pounds pounds_per_container

-- Given the total cost, find the cost per container.
def total_cost : ℕ := 210
def cost_per_container (total_cost containers: ℕ) : ℕ := total_cost / containers
def cost : ℕ := cost_per_container total_cost containers

-- Prove that the cost per container is 21.
theorem cost_per_container_is_21 : cost = 21 := by
  sorry

end cost_per_container_is_21_l1711_171145


namespace range_of_a_l1711_171102

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ (-3/5 < a ∧ a ≤ 1) := 
sorry

end range_of_a_l1711_171102


namespace symmetric_about_y_l1711_171113

theorem symmetric_about_y (m n : ℤ) (h1 : 2 * n - m = -14) (h2 : m = 4) : (m + n) ^ 2023 = -1 := by
  sorry

end symmetric_about_y_l1711_171113


namespace Karlson_cannot_prevent_Baby_getting_one_fourth_l1711_171133

theorem Karlson_cannot_prevent_Baby_getting_one_fourth 
  (a : ℝ) (h : a > 0) (K : ℝ × ℝ) (hK : 0 < K.1 ∧ K.1 < a ∧ 0 < K.2 ∧ K.2 < a) :
  ∀ (O : ℝ × ℝ) (cut1 cut2 : ℝ), 
    ((O.1 = a/2) ∧ (O.2 = a/2) ∧ (cut1 = K.1 ∧ cut1 = a ∨ cut1 = K.2 ∧ cut1 = a) ∧ 
                             (cut2 = K.1 ∧ cut2 = a ∨ cut2 = K.2 ∧ cut2 = a)) →
  ∃ (piece : ℝ), piece ≥ a^2 / 4 :=
by
  sorry

end Karlson_cannot_prevent_Baby_getting_one_fourth_l1711_171133


namespace sale_record_is_negative_five_l1711_171116

-- Given that a purchase of 10 items is recorded as +10
def purchase_record (items : Int) : Int := items

-- Prove that the sale of 5 items should be recorded as -5
theorem sale_record_is_negative_five : purchase_record 10 = 10 → purchase_record (-5) = -5 :=
by
  intro h
  sorry

end sale_record_is_negative_five_l1711_171116


namespace pieces_per_package_l1711_171169

-- Definitions from conditions
def total_pieces_of_gum : ℕ := 486
def number_of_packages : ℕ := 27

-- Mathematical statement to prove
theorem pieces_per_package : total_pieces_of_gum / number_of_packages = 18 := sorry

end pieces_per_package_l1711_171169


namespace rectangle_area_in_triangle_l1711_171164

theorem rectangle_area_in_triangle (c k y : ℝ) (h1 : c > 0) (h2 : k > 0) (h3 : 0 < y) (h4 : y < k) : 
  ∃ A : ℝ, A = y * ((c * (k - y)) / k) := 
by
  sorry

end rectangle_area_in_triangle_l1711_171164


namespace restaurant_bill_l1711_171156

theorem restaurant_bill
    (t : ℝ)
    (h1 : ∀ k : ℝ, k = 9 * (t / 10 + 3)) :
    t = 270 :=
by
    sorry

end restaurant_bill_l1711_171156


namespace monotone_increasing_intervals_exists_x0_implies_p_l1711_171135

noncomputable def f (x : ℝ) := 6 * Real.log x + x ^ 2 - 8 * x
noncomputable def g (x : ℝ) (p : ℝ) := p / x + x ^ 2

theorem monotone_increasing_intervals :
  (∀ x, (0 < x ∧ x ≤ 1) → ∃ ε > 0, ∀ y, x < y → f y > f x) ∧
  (∀ x, (3 ≤ x) → ∃ ε > 0, ∀ y, x < y → f y > f x) := by
  sorry

theorem exists_x0_implies_p :
  (∃ x0, 1 ≤ x0 ∧ x0 ≤ Real.exp 1 ∧ f x0 > g x0 p) → p < -8 := by
  sorry

end monotone_increasing_intervals_exists_x0_implies_p_l1711_171135


namespace complement_union_A_B_l1711_171190

open Set

variable {U : Type*} [Preorder U] [BoundedOrder U]

def A : Set ℝ := {x | x < 1}
def B : Set ℝ := {x | x ≥ 2}

theorem complement_union_A_B :
  compl (A ∪ B) = {x : ℝ | 1 ≤ x ∧ x < 2} :=
by
  sorry

end complement_union_A_B_l1711_171190


namespace correct_statement_l1711_171197

-- Definitions as per conditions
def P1 : Prop := ∃ x : ℝ, x^2 = 64 ∧ abs x ^ 3 = 2
def P2 : Prop := ∀ x : ℝ, x = 0 → (¬∃ y, y * x = 1 ∧ -x = y)
def P3 : Prop := ∀ x y : ℝ, x + y = 0 → abs x / abs y = -1
def P4 : Prop := ∀ x a : ℝ, abs x + x = a → a > 0

-- The proof problem
theorem correct_statement : P1 ∧ ¬P2 ∧ ¬P3 ∧ ¬P4 := by
  sorry

end correct_statement_l1711_171197


namespace p_necessary_not_sufficient_for_p_and_q_l1711_171166

-- Define statements p and q as propositions
variables (p q : Prop)

-- Prove that "p is true" is a necessary but not sufficient condition for "p ∧ q is true"
theorem p_necessary_not_sufficient_for_p_and_q : (p ∧ q → p) ∧ (p → ¬ (p ∧ q)) :=
by sorry

end p_necessary_not_sufficient_for_p_and_q_l1711_171166


namespace intersection_of_A_and_B_l1711_171184

def A : Set ℕ := {0, 2}
def B : Set ℕ := {1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {2} :=
by
  sorry

end intersection_of_A_and_B_l1711_171184


namespace estimated_white_balls_l1711_171147

noncomputable def estimate_white_balls (total_balls draws white_draws : ℕ) : ℕ :=
  total_balls * white_draws / draws

theorem estimated_white_balls (total_balls draws white_draws : ℕ) (h1 : total_balls = 20)
  (h2 : draws = 100) (h3 : white_draws = 40) :
  estimate_white_balls total_balls draws white_draws = 8 := by
  sorry

end estimated_white_balls_l1711_171147


namespace equation_of_perpendicular_line_l1711_171111

theorem equation_of_perpendicular_line :
  ∃ (a b c : ℝ), (5, 3) ∈ {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}
  ∧ (a = 2 ∧ b = 1 ∧ c = -13)
  ∧ (a * 1 + b * (-2) = 0) :=
sorry

end equation_of_perpendicular_line_l1711_171111


namespace expected_number_of_hits_l1711_171195

variable (W : ℝ) (n : ℕ)
def expected_hits (W : ℝ) (n : ℕ) : ℝ := W * n

theorem expected_number_of_hits :
  W = 0.75 → n = 40 → expected_hits W n = 30 :=
by
  intros hW hn
  rw [hW, hn]
  norm_num
  sorry

end expected_number_of_hits_l1711_171195


namespace set_has_one_element_iff_double_root_l1711_171118

theorem set_has_one_element_iff_double_root (k : ℝ) :
  (∃ x, ∀ y, y^2 - k*y + 1 = 0 ↔ y = x) ↔ k = 2 ∨ k = -2 :=
by
  sorry

end set_has_one_element_iff_double_root_l1711_171118


namespace explicit_formula_for_sequence_l1711_171105

theorem explicit_formula_for_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (hSn : ∀ n, S n = 2 * a n + 1) : 
  ∀ n, a n = (-2) ^ (n - 1) := 
by
  sorry

end explicit_formula_for_sequence_l1711_171105


namespace simple_interest_sum_l1711_171146

theorem simple_interest_sum :
  let P := 1750
  let CI := 4000 * ((1 + (10 / 100))^2) - 4000
  let SI := (1 / 2) * CI
  SI = (P * 8 * 3) / 100 
  :=
by
  -- Definitions
  let P := 1750
  let CI := 4000 * ((1 + 10 / 100)^2) - 4000
  let SI := (1 / 2) * CI
  
  -- Claim
  have : SI = (P * 8 * 3) / 100 := sorry

  exact this

end simple_interest_sum_l1711_171146


namespace geometric_sequence_third_sixth_term_l1711_171175

theorem geometric_sequence_third_sixth_term (a r : ℝ) 
  (h3 : a * r^2 = 18) 
  (h6 : a * r^5 = 162) : 
  a = 2 ∧ r = 3 := 
sorry

end geometric_sequence_third_sixth_term_l1711_171175


namespace sphere_surface_area_l1711_171119

theorem sphere_surface_area (r : ℝ) (h : π * r^2 = 81 * π) : 4 * π * r^2 = 324 * π :=
  sorry

end sphere_surface_area_l1711_171119


namespace laura_weekly_mileage_l1711_171149

-- Define the core conditions

-- Distance to school per round trip (house <-> school)
def school_trip_distance : ℕ := 20

-- Number of trips to school per week
def school_trips_per_week : ℕ := 7

-- Distance to supermarket: 10 miles farther than school
def extra_distance_to_supermarket : ℕ := 10
def supermarket_trip_distance : ℕ := school_trip_distance + 2 * extra_distance_to_supermarket

-- Number of trips to supermarket per week
def supermarket_trips_per_week : ℕ := 2

-- Calculate the total weekly distance
def total_distance_per_week : ℕ := 
  (school_trips_per_week * school_trip_distance) +
  (supermarket_trips_per_week * supermarket_trip_distance)

-- Theorem to prove the total distance Laura drives per week
theorem laura_weekly_mileage :
  total_distance_per_week = 220 := by
  sorry

end laura_weekly_mileage_l1711_171149


namespace factor_expression_l1711_171178

theorem factor_expression (x : ℝ) : 84 * x^7 - 297 * x^13 = 3 * x^7 * (28 - 99 * x^6) :=
by sorry

end factor_expression_l1711_171178


namespace valid_N_count_l1711_171192

theorem valid_N_count : 
  (∃ n : ℕ, 0 < n ∧ (49 % (n + 3) = 0) ∧ (49 / (n + 3)) % 2 = 1) → 
  (∃ count : ℕ, count = 2) :=
sorry

end valid_N_count_l1711_171192


namespace part_a_part_b_l1711_171106

variable {f : ℝ → ℝ} 

-- Given conditions
axiom condition1 (x y : ℝ) : f (x + y) + 1 = f x + f y
axiom condition2 : f (1/2) = 0
axiom condition3 (x : ℝ) : x > 1/2 → f x < 0

-- Part (a)
theorem part_a (x : ℝ) : f x = 1/2 + 1/2 * f (2 * x) :=
sorry

-- Part (b)
theorem part_b (n : ℕ) (hn : n > 0) (x : ℝ) 
  (hx : 1 / 2^(n + 1) ≤ x ∧ x ≤ 1 / 2^n) : f x ≤ 1 - 1 / 2^n :=
sorry

end part_a_part_b_l1711_171106


namespace cannot_fit_rectangle_l1711_171194

theorem cannot_fit_rectangle 
  (w1 h1 : ℕ) (w2 h2 : ℕ) 
  (h1_pos : 0 < h1) (w1_pos : 0 < w1)
  (h2_pos : 0 < h2) (w2_pos : 0 < w2) :
  w1 = 5 → h1 = 6 → w2 = 3 → h2 = 8 →
  ¬(w2 ≤ w1 ∧ h2 ≤ h1) :=
by
  intros H1 W1 H2 W2
  sorry

end cannot_fit_rectangle_l1711_171194


namespace perpendicular_lines_a_eq_2_l1711_171108

/-- Given two lines, ax + 2y + 2 = 0 and x - y - 2 = 0, prove that if these lines are perpendicular, then a = 2. -/
theorem perpendicular_lines_a_eq_2 {a : ℝ} :
  (∃ a, (a ≠ 0)) → (∃ x y, ((ax + 2*y + 2 = 0) ∧ (x - y - 2 = 0)) → - (a / 2) * 1 = -1) → a = 2 :=
by
  sorry

end perpendicular_lines_a_eq_2_l1711_171108


namespace circular_park_diameter_factor_l1711_171162

theorem circular_park_diameter_factor (r : ℝ) :
  (π * (3 * r)^2) / (π * r^2) = 9 ∧ (2 * π * (3 * r)) / (2 * π * r) = 3 :=
by
  sorry

end circular_park_diameter_factor_l1711_171162


namespace average_age_union_l1711_171107

theorem average_age_union
    (A B C : Set Person)
    (a b c : ℕ)
    (sum_A sum_B sum_C : ℝ)
    (h_disjoint_AB : Disjoint A B)
    (h_disjoint_AC : Disjoint A C)
    (h_disjoint_BC : Disjoint B C)
    (h_avg_A : sum_A / a = 40)
    (h_avg_B : sum_B / b = 25)
    (h_avg_C : sum_C / c = 35)
    (h_avg_AB : (sum_A + sum_B) / (a + b) = 33)
    (h_avg_AC : (sum_A + sum_C) / (a + c) = 37.5)
    (h_avg_BC : (sum_B + sum_C) / (b + c) = 30) :
  (sum_A + sum_B + sum_C) / (a + b + c) = 51.6 :=
sorry

end average_age_union_l1711_171107


namespace more_pie_eaten_l1711_171128

theorem more_pie_eaten (e f : ℝ) (h1 : e = 0.67) (h2 : f = 0.33) : e - f = 0.34 :=
by sorry

end more_pie_eaten_l1711_171128


namespace solve_abs_inequality_l1711_171124

theorem solve_abs_inequality (x : ℝ) :
  abs ((6 - 2 * x + 5) / 4) < 3 ↔ -1 / 2 < x ∧ x < 23 / 2 := 
sorry

end solve_abs_inequality_l1711_171124


namespace multiplication_correct_l1711_171131

theorem multiplication_correct (a b c d e f: ℤ) (h₁: a * b = c) (h₂: d * e = f): 
    (63 * 14 = c) → (68 * 14 = f) → c = 882 ∧ f = 952 :=
by sorry

end multiplication_correct_l1711_171131


namespace mileage_per_gallon_l1711_171189

noncomputable def car_mileage (distance: ℝ) (gasoline: ℝ) : ℝ :=
  distance / gasoline

theorem mileage_per_gallon :
  car_mileage 190 4.75 = 40 :=
by
  -- proof omitted
  sorry

end mileage_per_gallon_l1711_171189


namespace probability_intersection_of_diagonals_hendecagon_l1711_171191

-- Definition statements expressing the given conditions and required probability

def total_diagonals (n : ℕ) : ℕ := (Nat.choose n 2) - n

def ways_to_choose_2_diagonals (n : ℕ) : ℕ := Nat.choose (total_diagonals n) 2

def ways_sets_of_intersecting_diagonals (n : ℕ) : ℕ := Nat.choose n 4

def probability_intersection_lies_inside (n : ℕ) : ℚ :=
  ways_sets_of_intersecting_diagonals n / ways_to_choose_2_diagonals n

theorem probability_intersection_of_diagonals_hendecagon :
  probability_intersection_lies_inside 11 = 165 / 473 := 
by
  sorry

end probability_intersection_of_diagonals_hendecagon_l1711_171191


namespace reciprocal_of_neg3_l1711_171127

theorem reciprocal_of_neg3 : (1 : ℚ) / (-3 : ℚ) = -1 / 3 := 
by
  sorry

end reciprocal_of_neg3_l1711_171127


namespace modulus_of_z_l1711_171160

noncomputable def z : ℂ := (Complex.I / (1 + 2 * Complex.I))

theorem modulus_of_z : Complex.abs z = (Real.sqrt 5) / 5 := by
  sorry

end modulus_of_z_l1711_171160


namespace remainder_sum_l1711_171144

theorem remainder_sum (a b c d : ℕ) 
  (h_a : a % 30 = 15) 
  (h_b : b % 30 = 7) 
  (h_c : c % 30 = 22) 
  (h_d : d % 30 = 6) : 
  (a + b + c + d) % 30 = 20 := 
by
  sorry

end remainder_sum_l1711_171144


namespace Alyosha_result_divisible_by_S_l1711_171129

variable (a b S x y : ℤ)
variable (h1 : x + y = S)
variable (h2 : S ∣ a * x + b * y)

theorem Alyosha_result_divisible_by_S :
  S ∣ b * x + a * y :=
sorry

end Alyosha_result_divisible_by_S_l1711_171129


namespace sufficient_conditions_for_equation_l1711_171182

theorem sufficient_conditions_for_equation 
  (a b c : ℤ) :
  (a = b ∧ b = c + 1) ∨ (a = c ∧ b - 1 = c) →
  a * (a - b) + b * (b - c) + c * (c - a) = 2 :=
by
  sorry

end sufficient_conditions_for_equation_l1711_171182


namespace daily_production_l1711_171193

theorem daily_production (x : ℕ) (hx1 : 216 / x > 4)
  (hx2 : 3 * x + (x + 8) * ((216 / x) - 4) = 232) : 
  x = 24 := by
sorry

end daily_production_l1711_171193


namespace books_bought_l1711_171126

theorem books_bought (cost_crayons cost_calculators total_money cost_per_bag bags_bought cost_per_book remaining_money books_bought : ℕ) 
  (h1: cost_crayons = 5 * 5)
  (h2: cost_calculators = 3 * 5)
  (h3: total_money = 200)
  (h4: cost_per_bag = 10)
  (h5: bags_bought = 11)
  (h6: remaining_money = total_money - (cost_crayons + cost_calculators) - (bags_bought * cost_per_bag)) :
  books_bought = remaining_money / cost_per_book → books_bought = 10 :=
by
  sorry

end books_bought_l1711_171126


namespace outfit_combination_count_l1711_171109

theorem outfit_combination_count (c : ℕ) (s p h sh : ℕ) (c_eq_6 : c = 6) (s_eq_c : s = c) (p_eq_c : p = c) (h_eq_c : h = c) (sh_eq_c : sh = c) :
  (c^4) - c = 1290 :=
by
  sorry

end outfit_combination_count_l1711_171109


namespace haley_seeds_total_l1711_171183

-- Conditions
def seeds_in_big_garden : ℕ := 35
def small_gardens : ℕ := 7
def seeds_per_small_garden : ℕ := 3

-- Question rephrased as a problem with the correct answer
theorem haley_seeds_total : seeds_in_big_garden + small_gardens * seeds_per_small_garden = 56 := by
  sorry

end haley_seeds_total_l1711_171183


namespace proportion_correct_l1711_171161

theorem proportion_correct (x y : ℝ) (h : 5 * y = 4 * x) : x / y = 5 / 4 :=
sorry

end proportion_correct_l1711_171161


namespace parabola_c_value_l1711_171123

theorem parabola_c_value (b c : ℝ) 
  (h1 : 2 * b + c = 6) 
  (h2 : -2 * b + c = 2)
  (vertex_cond : ∃ x y : ℝ, y = x^2 + b * x + c ∧ y = -x + 4) : 
  c = 4 :=
sorry

end parabola_c_value_l1711_171123


namespace total_allocation_is_1800_l1711_171181

-- Definitions from conditions.
def part_value (amount_food : ℕ) (ratio_food : ℕ) : ℕ :=
  amount_food / ratio_food

def total_parts (ratio_household : ℕ) (ratio_food : ℕ) (ratio_misc : ℕ) : ℕ :=
  ratio_household + ratio_food + ratio_misc

def total_amount (part_value : ℕ) (total_parts : ℕ) : ℕ :=
  part_value * total_parts

-- Given conditions
def ratio_household := 5
def ratio_food := 4
def ratio_misc := 1
def amount_food := 720

-- Prove the total allocation
theorem total_allocation_is_1800 
  (amount_food : ℕ := 720) 
  (ratio_household : ℕ := 5) 
  (ratio_food : ℕ := 4) 
  (ratio_misc : ℕ := 1) : 
  total_amount (part_value amount_food ratio_food) (total_parts ratio_household ratio_food ratio_misc) = 1800 :=
by
  sorry

end total_allocation_is_1800_l1711_171181


namespace difference_of_squares_l1711_171121

theorem difference_of_squares :
  535^2 - 465^2 = 70000 :=
by
  sorry

end difference_of_squares_l1711_171121


namespace calculate_c_l1711_171165

-- Define the given equation as a hypothesis
theorem calculate_c (a b k c : ℝ) (h : (1 / (k * a) - 1 / (k * b) = 1 / c)) :
  c = k * a * b / (b - a) :=
by
  sorry

end calculate_c_l1711_171165


namespace triangle_inequalities_l1711_171141

-- Definitions of the variables
variables {ABC : Triangle} {r : ℝ} {R : ℝ} {ρ_a ρ_b ρ_c : ℝ} {P_a P_b P_c : ℝ}

-- Problem statement based on given conditions and proof requirement
theorem triangle_inequalities (ABC : Triangle) (r : ℝ) (R : ℝ) (ρ_a ρ_b ρ_c : ℝ) (P_a P_b P_c : ℝ) :
  (3/2) * r ≤ ρ_a + ρ_b + ρ_c ∧ ρ_a + ρ_b + ρ_c ≤ (3/4) * R ∧ 4 * r ≤ P_a + P_b + P_c ∧ P_a + P_b + P_c ≤ 2 * R :=
  sorry

end triangle_inequalities_l1711_171141


namespace spencer_walk_distance_l1711_171196

theorem spencer_walk_distance :
  let distance_house_library := 0.3
  let distance_library_post_office := 0.1
  let total_distance := 0.8
  (total_distance - (distance_house_library + distance_library_post_office)) = 0.4 :=
by
  sorry

end spencer_walk_distance_l1711_171196


namespace difference_is_divisible_by_p_l1711_171176

-- Lean 4 statement equivalent to the math proof problem
theorem difference_is_divisible_by_p
  (a : ℕ → ℕ) (p : ℕ) (d : ℕ)
  (h_prime : Nat.Prime p)
  (h_prog : ∀ i j: ℕ, 1 ≤ i ∧ i ≤ p ∧ 1 ≤ j ∧ j ≤ p ∧ i < j → a j = a (i + 1) + (j - 1) * d)
  (h_a_gt_p : a 1 > p)
  (h_arith_prog_primes : ∀ i: ℕ, 1 ≤ i ∧ i ≤ p → Nat.Prime (a i)) :
  d % p = 0 := sorry

end difference_is_divisible_by_p_l1711_171176


namespace n_minus_two_is_square_of_natural_number_l1711_171170

theorem n_minus_two_is_square_of_natural_number (n : ℕ) (h_n : n ≥ 3) (h_odd_m : Odd (1 / 2 * n * (n - 1))) :
  ∃ k : ℕ, n - 2 = k^2 := 
  by
  sorry

end n_minus_two_is_square_of_natural_number_l1711_171170


namespace cost_of_500_cookies_in_dollars_l1711_171159

def cost_in_cents (cookies : Nat) (cost_per_cookie : Nat) : Nat :=
  cookies * cost_per_cookie

def cents_to_dollars (cents : Nat) : Nat :=
  cents / 100

theorem cost_of_500_cookies_in_dollars :
  cents_to_dollars (cost_in_cents 500 2) = 10
:= by
  sorry

end cost_of_500_cookies_in_dollars_l1711_171159


namespace radius_correct_l1711_171179

open Real

noncomputable def radius_of_circle
  (tangent_length : ℝ) 
  (secant_internal_segment : ℝ) 
  (tangent_secant_perpendicular : Prop) : ℝ := sorry

theorem radius_correct
  (tangent_length : ℝ) 
  (secant_internal_segment : ℝ) 
  (tangent_secant_perpendicular : Prop)
  (h1 : tangent_length = 12) 
  (h2 : secant_internal_segment = 10) 
  (h3 : tangent_secant_perpendicular) : radius_of_circle tangent_length secant_internal_segment tangent_secant_perpendicular = 13 := 
sorry

end radius_correct_l1711_171179


namespace work_completed_by_a_l1711_171199

theorem work_completed_by_a (a b : ℕ) (work_in_30_days : a + b = 4 * 30) (a_eq_3b : a = 3 * b) : (120 / a) = 40 :=
by
  -- Given a + b = 120 and a = 3 * b, prove that 120 / a = 40
  sorry

end work_completed_by_a_l1711_171199


namespace car_speed_is_80_l1711_171158

theorem car_speed_is_80 : ∃ v : ℝ, (1 / v * 3600 = 45) ∧ (v = 80) :=
by
  sorry

end car_speed_is_80_l1711_171158


namespace border_pieces_is_75_l1711_171134

-- Definitions based on conditions
def total_pieces : Nat := 500
def trevor_pieces : Nat := 105
def joe_pieces : Nat := 3 * trevor_pieces
def missing_pieces : Nat := 5

-- Number of border pieces
def border_pieces : Nat := total_pieces - missing_pieces - (trevor_pieces + joe_pieces)

-- Theorem statement
theorem border_pieces_is_75 : border_pieces = 75 :=
by
  -- Proof goes here
  sorry

end border_pieces_is_75_l1711_171134


namespace solve_system_l1711_171150

theorem solve_system (a b c : ℝ)
  (h1 : b + c = 10 - 4 * a)
  (h2 : a + c = -16 - 4 * b)
  (h3 : a + b = 9 - 4 * c) :
  2 * a + 2 * b + 2 * c = 1 :=
by
  sorry

end solve_system_l1711_171150


namespace correct_statement_l1711_171114

def is_accurate_to (value : ℝ) (place : ℝ) : Prop :=
  ∃ k : ℤ, value = k * place

def statement_A : Prop := is_accurate_to 51000 0.1
def statement_B : Prop := is_accurate_to 0.02 1
def statement_C : Prop := (2.8 = 2.80)
def statement_D : Prop := is_accurate_to (2.3 * 10^4) 1000

theorem correct_statement : statement_D :=
by
  sorry

end correct_statement_l1711_171114


namespace gcd_f_101_102_l1711_171132

def f (x : ℕ) : ℕ := x^2 + x + 2010

theorem gcd_f_101_102 : Nat.gcd (f 101) (f 102) = 12 := 
by sorry

end gcd_f_101_102_l1711_171132


namespace three_minus_pi_to_zero_l1711_171174

theorem three_minus_pi_to_zero : (3 - Real.pi) ^ 0 = 1 := by
  -- proof goes here
  sorry

end three_minus_pi_to_zero_l1711_171174


namespace area_of_region_l1711_171154

def region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (abs p.1 - p.1)^2 + (abs p.2 - p.2)^2 ≤ 16 ∧ 2 * p.2 + p.1 ≤ 0}

noncomputable def area : ℝ := sorry

theorem area_of_region : area = 5 + Real.pi := by
  sorry

end area_of_region_l1711_171154


namespace evaluate_g_at_neg_four_l1711_171139

def g (x : ℤ) : ℤ := 5 * x + 2

theorem evaluate_g_at_neg_four : g (-4) = -18 := 
by 
  sorry

end evaluate_g_at_neg_four_l1711_171139


namespace percentage_increase_in_area_is_96_l1711_171148

theorem percentage_increase_in_area_is_96 :
  let r₁ := 5
  let r₃ := 7
  let A (r : ℝ) := Real.pi * r^2
  ((A r₃ - A r₁) / A r₁) * 100 = 96 := by
  sorry

end percentage_increase_in_area_is_96_l1711_171148


namespace even_natural_number_factors_count_l1711_171173

def is_valid_factor (a b c : ℕ) : Prop := 
  1 ≤ a ∧ a ≤ 3 ∧ 
  0 ≤ b ∧ b ≤ 2 ∧ 
  0 ≤ c ∧ c ≤ 2 ∧ 
  a + b + c ≤ 4

noncomputable def count_valid_factors : ℕ :=
  Nat.card { x : ℕ × ℕ × ℕ // is_valid_factor x.1 x.2.1 x.2.2 }

theorem even_natural_number_factors_count : count_valid_factors = 15 := 
  sorry

end even_natural_number_factors_count_l1711_171173


namespace true_statements_l1711_171112

theorem true_statements :
  (5 ∣ 25) ∧ (19 ∣ 209 ∧ ¬ (19 ∣ 63)) ∧ (30 ∣ 90) ∧ (14 ∣ 28 ∧ 14 ∣ 56) ∧ (9 ∣ 180) :=
by
  have A : 5 ∣ 25 := sorry
  have B1 : 19 ∣ 209 := sorry
  have B2 : ¬ (19 ∣ 63) := sorry
  have C : 30 ∣ 90 := sorry
  have D1 : 14 ∣ 28 := sorry
  have D2 : 14 ∣ 56 := sorry
  have E : 9 ∣ 180 := sorry
  exact ⟨A, ⟨B1, B2⟩, C, ⟨D1, D2⟩, E⟩

end true_statements_l1711_171112


namespace corina_problem_l1711_171152

variable (P Q : ℝ)

theorem corina_problem (h1 : P + Q = 16) (h2 : P - Q = 4) : P = 10 :=
sorry

end corina_problem_l1711_171152


namespace zoo_children_count_l1711_171143

theorem zoo_children_count:
  ∀ (C : ℕ), 
  (10 * C + 16 * 10 = 220) → 
  C = 6 :=
by
  intro C
  intro h
  sorry

end zoo_children_count_l1711_171143


namespace frequency_distribution_necessary_l1711_171142

/-- Definition of the necessity to use Frequency Distribution to understand 
the proportion of first-year high school students in the city whose height 
falls within a certain range -/
def necessary_for_proportion (A B C D : Prop) : Prop := D

theorem frequency_distribution_necessary (A B C D : Prop) :
  necessary_for_proportion A B C D ↔ D :=
by
  sorry

end frequency_distribution_necessary_l1711_171142


namespace total_fruits_is_78_l1711_171155

def oranges_louis : Nat := 5
def apples_louis : Nat := 3

def oranges_samantha : Nat := 8
def apples_samantha : Nat := 7

def oranges_marley : Nat := 2 * oranges_louis
def apples_marley : Nat := 3 * apples_samantha

def oranges_edward : Nat := 3 * oranges_louis
def apples_edward : Nat := 3 * apples_louis

def total_fruits_louis : Nat := oranges_louis + apples_louis
def total_fruits_samantha : Nat := oranges_samantha + apples_samantha
def total_fruits_marley : Nat := oranges_marley + apples_marley
def total_fruits_edward : Nat := oranges_edward + apples_edward

def total_fruits_all : Nat :=
  total_fruits_louis + total_fruits_samantha + total_fruits_marley + total_fruits_edward

theorem total_fruits_is_78 : total_fruits_all = 78 := by
  sorry

end total_fruits_is_78_l1711_171155


namespace find_some_number_l1711_171153

theorem find_some_number (a : ℕ) (some_number : ℕ)
  (h1 : a = 105)
  (h2 : a ^ 3 = 21 * 35 * some_number * 35) :
  some_number = 21 :=
by
  sorry

end find_some_number_l1711_171153


namespace don_walking_speed_l1711_171101

theorem don_walking_speed 
  (distance_between_homes : ℝ)
  (cara_walking_speed : ℝ)
  (cara_distance_before_meeting : ℝ)
  (time_don_starts_after_cara : ℝ)
  (total_distance : distance_between_homes = 45)
  (cara_speed : cara_walking_speed = 6)
  (cara_distance : cara_distance_before_meeting = 30)
  (time_after_cara : time_don_starts_after_cara = 2) :
  ∃ (v : ℝ), v = 5 := by
    sorry

end don_walking_speed_l1711_171101


namespace min_distance_l1711_171110

noncomputable def point_on_curve (x₁ y₁ : ℝ) : Prop :=
  y₁ = x₁^2 - Real.log x₁

noncomputable def point_on_line (x₂ y₂ : ℝ) : Prop :=
  x₂ - y₂ - 2 = 0

theorem min_distance 
  (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : point_on_curve x₁ y₁)
  (h₂ : point_on_line x₂ y₂) 
  : (x₂ - x₁)^2 + (y₂ - y₁)^2 = 2 :=
sorry

end min_distance_l1711_171110


namespace toby_initial_photos_l1711_171138

-- Defining the problem conditions and proving the initial number of photos Toby had.
theorem toby_initial_photos (X : ℕ) 
  (h1 : ∃ n, X = n - 7) 
  (h2 : ∃ m, m = (n - 7) + 15) 
  (h3 : ∃ k, k = m) 
  (h4 : (k - 3) = 84) 
  : X = 79 :=
sorry

end toby_initial_photos_l1711_171138


namespace total_donation_l1711_171115

-- Definitions
def cassandra_pennies : ℕ := 5000
def james_deficit : ℕ := 276
def james_pennies : ℕ := cassandra_pennies - james_deficit

-- Theorem to prove the total donation
theorem total_donation : cassandra_pennies + james_pennies = 9724 :=
by
  -- Proof is omitted
  sorry

end total_donation_l1711_171115


namespace simplify_fraction_l1711_171163

open Complex

theorem simplify_fraction :
  (3 + 3 * I) / (-1 + 3 * I) = -1.2 - 1.2 * I :=
by
  sorry

end simplify_fraction_l1711_171163


namespace student_answered_two_questions_incorrectly_l1711_171130

/-
  Defining the variables and conditions for the problem.
  x: number of questions answered correctly,
  y: number of questions not answered,
  z: number of questions answered incorrectly.
-/

theorem student_answered_two_questions_incorrectly (x y z : ℕ) 
  (h1 : x + y + z = 6) 
  (h2 : 8 * x + 2 * y = 20) : z = 2 :=
by
  /- We know the total number of questions is 6.
     And the total score is 20 with the given scoring rules.
     Thus, we need to prove that z = 2 under these conditions. -/
  sorry

end student_answered_two_questions_incorrectly_l1711_171130


namespace min_geometric_ratio_l1711_171168

theorem min_geometric_ratio (q : ℝ) (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) = a n * q)
(h2 : 1 < q) (h3 : q < 2) : q = 6 / 5 := by
  sorry

end min_geometric_ratio_l1711_171168


namespace find_d_l1711_171172

open Real

theorem find_d (a b c d : ℝ) 
  (h : a^3 + b^3 + c^3 + a^2 + b^2 + 1 = d^2 + d + sqrt (a + b + c - 2 * d)) : 
  d = 1 ∨ d = -(4 / 3) :=
sorry

end find_d_l1711_171172


namespace pies_can_be_made_l1711_171198

def total_apples : Nat := 51
def apples_handout : Nat := 41
def apples_per_pie : Nat := 5

theorem pies_can_be_made :
  ((total_apples - apples_handout) / apples_per_pie) = 2 := by
  sorry

end pies_can_be_made_l1711_171198


namespace nicky_cristina_race_l1711_171157

theorem nicky_cristina_race :
  ∀ (head_start t : ℕ), ∀ (cristina_speed nicky_speed time_nicky_run : ℝ),
  head_start = 12 →
  cristina_speed = 5 →
  nicky_speed = 3 →
  ((cristina_speed * t) = (nicky_speed * t + nicky_speed * head_start)) →
  time_nicky_run = head_start + t →
  time_nicky_run = 30 :=
by
  intros
  sorry

end nicky_cristina_race_l1711_171157


namespace symmetric_line_proof_l1711_171187

-- Define the given lines
def line_l (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0
def axis_of_symmetry (x y : ℝ) : Prop := x + y = 0

-- Define the final symmetric line to be proved
def symmetric_line (x y : ℝ) : Prop := 4 * x - 3 * y - 5 = 0

-- State the theorem
theorem symmetric_line_proof (x y : ℝ) : 
  (line_l (-y) (-x)) → 
  axis_of_symmetry x y → 
  symmetric_line x y := 
sorry

end symmetric_line_proof_l1711_171187


namespace cats_left_l1711_171180

theorem cats_left (siamese_cats : ℕ) (house_cats : ℕ) (cats_sold : ℕ) (total_initial_cats : ℕ) (remaining_cats : ℕ) :
  siamese_cats = 15 → house_cats = 49 → cats_sold = 19 → total_initial_cats = siamese_cats + house_cats → remaining_cats = total_initial_cats - cats_sold → remaining_cats = 45 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2] at h4
  rw [h4, h3] at h5
  exact h5

end cats_left_l1711_171180
