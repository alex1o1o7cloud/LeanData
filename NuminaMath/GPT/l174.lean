import Mathlib

namespace cube_volume_of_surface_area_l174_174016

-- Define the condition: the surface area S is 864 square units
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- The proof problem: Given that the surface area of a cube is 864 square units,
-- prove that the volume of the cube is 1728 cubic units
theorem cube_volume_of_surface_area (S : ℝ) (hS : S = 864) : 
  ∃ V : ℝ, V = 1728 ∧ ∃ s : ℝ, surface_area s = S ∧ V = s^3 :=
by 
  sorry

end cube_volume_of_surface_area_l174_174016


namespace inequality_B_inequality_C_l174_174871

variable (a b : ℝ)

theorem inequality_B (h : 0 < a ∧ 0 < b ∧ (1 / Real.sqrt a > 1 / Real.sqrt b)) :
  (b / (a + b) + a / (2 * b) ≥ (2 * Real.sqrt 2 - 1) / 2) :=
by sorry

theorem inequality_C (h : 0 < a ∧ 0 < b ∧ (1 / Real.sqrt a > 1 / Real.sqrt b)) :
  ((b + 1) / (a + 1) < b / a) :=
by sorry

end inequality_B_inequality_C_l174_174871


namespace sugar_percentage_of_second_solution_l174_174119

theorem sugar_percentage_of_second_solution :
  ∀ (W : ℝ) (P : ℝ),
  (0.10 * W * (3 / 4) + P / 100 * (1 / 4) * W = 0.18 * W) → 
  (P = 42) :=
by
  intros W P h
  sorry

end sugar_percentage_of_second_solution_l174_174119


namespace stones_required_to_pave_hall_l174_174164

theorem stones_required_to_pave_hall :
  ∀ (hall_length_m hall_breadth_m stone_length_dm stone_breadth_dm: ℕ),
  hall_length_m = 72 →
  hall_breadth_m = 30 →
  stone_length_dm = 6 →
  stone_breadth_dm = 8 →
  (hall_length_m * 10 * hall_breadth_m * 10) / (stone_length_dm * stone_breadth_dm) = 4500 := by
  intros _ _ _ _ h_length h_breadth h_slength h_sbreadth
  sorry

end stones_required_to_pave_hall_l174_174164


namespace units_digit_pow_prod_l174_174662

theorem units_digit_pow_prod : 
  ((2 ^ 2023) * (5 ^ 2024) * (11 ^ 2025)) % 10 = 0 :=
by
  sorry

end units_digit_pow_prod_l174_174662


namespace selected_people_take_B_l174_174465

def arithmetic_sequence (a d n : Nat) : Nat := a + (n - 1) * d

theorem selected_people_take_B (a d total sampleCount start n_upper n_lower : Nat) :
  a = 9 →
  d = 30 →
  total = 960 →
  sampleCount = 32 →
  start = 451 →
  n_upper = 25 →
  n_lower = 16 →
  (960 / 32) = d → 
  (10 = n_upper - n_lower + 1) ∧ 
  ∀ n, (n_lower ≤ n ∧ n ≤ n_upper) → (start ≤ arithmetic_sequence a d n ∧ arithmetic_sequence a d n ≤ 750) :=
by sorry

end selected_people_take_B_l174_174465


namespace binomial_10_3_l174_174264

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end binomial_10_3_l174_174264


namespace sum_of_fifth_powers_cannot_conclude_sum_of_fourth_powers_l174_174964

-- Definition of conditions
variables {a b c d : ℝ} 

-- First proof statement
theorem sum_of_fifth_powers 
  (h1 : a + b = c + d) 
  (h2 : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 := 
sorry

-- Second proof statement
theorem cannot_conclude_sum_of_fourth_powers 
  (h1 : a + b = c + d) 
  (h2 : a^3 + b^3 = c^3 + d^3) : 
  ¬(a^4 + b^4 = c^4 + d^4) := 
sorry

end sum_of_fifth_powers_cannot_conclude_sum_of_fourth_powers_l174_174964


namespace binom_10_3_eq_120_l174_174183

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l174_174183


namespace cube_volume_l174_174028

theorem cube_volume (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
sorry

end cube_volume_l174_174028


namespace triangle_PQR_not_right_l174_174896

-- Definitions based on conditions
def isIsosceles (a b c : ℝ) (angle1 angle2 : ℝ) : Prop := (angle1 = angle2) ∧ (a = c)

def perimeter (a b c : ℝ) : ℝ := a + b + c

def isRightTriangle (a b c : ℝ) : Prop := a * a = b * b + c * c

-- Given conditions
def PQR : ℝ := 10
def PRQ : ℝ := 10
def QR : ℝ := 6
def angle_PQR : ℝ := 1
def angle_PRQ : ℝ := 1

-- Lean statement for the proof problem
theorem triangle_PQR_not_right 
  (h1 : isIsosceles PQR QR PRQ angle_PQR angle_PRQ)
  (h2 : QR = 6)
  (h3 : PRQ = 10):
  ¬ isRightTriangle PQR QR PRQ ∧ perimeter PQR QR PRQ = 26 :=
by {
    sorry
}

end triangle_PQR_not_right_l174_174896


namespace parallel_line_plane_l174_174695

noncomputable def line : Type := sorry
noncomputable def plane : Type := sorry

-- Predicate for parallel lines
noncomputable def is_parallel_line (a b : line) : Prop := sorry

-- Predicate for parallel line and plane
noncomputable def is_parallel_plane (a : line) (α : plane) : Prop := sorry

-- Predicate for line contained within the plane
noncomputable def contained_in_plane (b : line) (α : plane) : Prop := sorry

theorem parallel_line_plane
  (a b : line) (α : plane)
  (h1 : is_parallel_line a b)
  (h2 : ¬ contained_in_plane a α)
  (h3 : contained_in_plane b α) :
  is_parallel_plane a α :=
sorry

end parallel_line_plane_l174_174695


namespace ewan_sequence_has_113_l174_174665

def sequence_term (n : ℕ) : ℤ := 11 * n - 8

theorem ewan_sequence_has_113 : ∃ n : ℕ, sequence_term n = 113 := by
  sorry

end ewan_sequence_has_113_l174_174665


namespace binom_10_3_eq_120_l174_174192

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l174_174192


namespace hyperbola_asymptotes_l174_174744

theorem hyperbola_asymptotes (x y : ℝ) :
  (x^2 / 4 - y^2 = 1) → (y = x / 2 ∨ y = -x / 2) :=
sorry

end hyperbola_asymptotes_l174_174744


namespace incorrect_method_D_l174_174775

-- Conditions definitions
def conditionA (locus : Set α) (cond : α → Prop) :=
  ∀ p, (p ∈ locus ↔ cond p)

def conditionB (locus : Set α) (cond : α → Prop) :=
  ∀ p, (cond p ↔ p ∈ locus)

def conditionC (locus : Set α) (cond : α → Prop) :=
  ∀ p, (¬ (p ∈ locus) ↔ ¬ (cond p))

def conditionD (locus : Set α) (cond : α → Prop) :=
  ∀ p, (p ∈ locus → cond p) ∧ (∃ p, cond p ∧ ¬ (p ∈ locus))

def conditionE (locus : Set α) (cond : α → Prop) :=
  ∀ p, (cond p ↔ p ∈ locus)

-- Main theorem
theorem incorrect_method_D {α : Type} (locus : Set α) (cond : α → Prop) :
  conditionD locus cond →
  ¬ (conditionA locus cond) ∧
  ¬ (conditionB locus cond) ∧
  ¬ (conditionC locus cond) ∧
  ¬ (conditionE locus cond) :=
  sorry

end incorrect_method_D_l174_174775


namespace total_musicians_count_l174_174461

-- Define the given conditions
def orchestra_males := 11
def orchestra_females := 12
def choir_males := 12
def choir_females := 17

-- Total number of musicians in the orchestra
def orchestra_musicians := orchestra_males + orchestra_females

-- Total number of musicians in the band
def band_musicians := 2 * orchestra_musicians

-- Total number of musicians in the choir
def choir_musicians := choir_males + choir_females

-- Total number of musicians in the orchestra, band, and choir
def total_musicians := orchestra_musicians + band_musicians + choir_musicians

-- The theorem to prove
theorem total_musicians_count : total_musicians = 98 :=
by
  -- Lean proof part goes here.
  sorry

end total_musicians_count_l174_174461


namespace combination_10_3_eq_120_l174_174246

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end combination_10_3_eq_120_l174_174246


namespace binomial_10_3_eq_120_l174_174345

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l174_174345


namespace combination_10_3_eq_120_l174_174354

open Nat

theorem combination_10_3_eq_120 : (10.choose 3) = 120 := 
by
  sorry

end combination_10_3_eq_120_l174_174354


namespace fifth_term_arithmetic_sequence_l174_174938

variable (x y : ℝ)

def a1 := x + 2 * y^2
def a2 := x - 2 * y^2
def a3 := x + 3 * y
def a4 := x - 4 * y
def d := a2 - a1

theorem fifth_term_arithmetic_sequence : y = -1/2 → 
  x - 10 * y^2 - 4 * y^2 = x - 7/2 := by
  sorry

end fifth_term_arithmetic_sequence_l174_174938


namespace sum_of_arithmetic_sequence_15_terms_l174_174509

/-- An arithmetic sequence starts at 3 and has a common difference of 4.
    Prove that the sum of the first 15 terms of this sequence is 465. --/
theorem sum_of_arithmetic_sequence_15_terms :
  let a := 3
  let d := 4
  let n := 15
  let aₙ := a + (n - 1) * d
  (n / 2) * (a + aₙ) = 465 :=
by
  sorry

end sum_of_arithmetic_sequence_15_terms_l174_174509


namespace germs_per_dish_calc_l174_174153

theorem germs_per_dish_calc :
    let total_germs := 0.036 * 10^5
    let total_dishes := 36000 * 10^(-3)
    (total_germs / total_dishes) = 100 := by
    sorry

end germs_per_dish_calc_l174_174153


namespace problem_l174_174993

-- Define \(\alpha\)
def alpha : ℝ := 49 * Real.pi / 48

-- Define the expression
def expr : ℝ := 4 * (Real.sin(alpha) ^ 3 * Real.cos(49 * Real.pi / 16) + 
                     Real.cos(alpha) ^ 3 * Real.sin(49 * Real.pi / 16)) * 
                     Real.cos(49 * Real.pi / 12)

-- The main theorem
theorem problem : expr = 0.75 :=
  sorry

end problem_l174_174993


namespace inequality_proof_l174_174524

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧
  (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 :=
sorry

end inequality_proof_l174_174524


namespace students_remaining_after_four_stops_l174_174702

theorem students_remaining_after_four_stops :
  let initial_students := 60 
  let fraction_remaining := (2 / 3 : ℚ)
  let stop1_students := initial_students * fraction_remaining
  let stop2_students := stop1_students * fraction_remaining
  let stop3_students := stop2_students * fraction_remaining
  let stop4_students := stop3_students * fraction_remaining
  stop4_students = (320 / 27 : ℚ) :=
by
  sorry

end students_remaining_after_four_stops_l174_174702


namespace tangent_line_and_area_l174_174058

noncomputable def tangent_line_equation (t : ℝ) : String := 
  "x + e^t * y - t - 1 = 0"

noncomputable def area_triangle_MON (t : ℝ) : ℝ :=
  (t + 1)^2 / (2 * Real.exp t)

theorem tangent_line_and_area (t : ℝ) (ht : t > 0) :
  tangent_line_equation t = "x + e^t * y - t - 1 = 0" ∧
  area_triangle_MON t = (t + 1)^2 / (2 * Real.exp t) := by
  sorry

end tangent_line_and_area_l174_174058


namespace find_k_l174_174077

/- Definitions for vectors -/
def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (-3, 2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

/- Prove that if ka + b is perpendicular to a, then k = -1/5 -/
theorem find_k (k : ℝ) : 
  dot_product (k • (1, 2) + (-3, 2)) (1, 2) = 0 → 
  k = -1 / 5 := 
  sorry

end find_k_l174_174077


namespace binomial_coefficient_10_3_l174_174280

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l174_174280


namespace middle_aged_employees_participating_l174_174977

-- Define the total number of employees and the ratio
def total_employees : ℕ := 1200
def ratio_elderly : ℕ := 1
def ratio_middle_aged : ℕ := 5
def ratio_young : ℕ := 6

-- Define the number of employees chosen for the performance
def chosen_employees : ℕ := 36

-- Calculate the number of middle-aged employees participating in the performance
theorem middle_aged_employees_participating : (36 * ratio_middle_aged / (ratio_elderly + ratio_middle_aged + ratio_young)) = 15 :=
by
  sorry

end middle_aged_employees_participating_l174_174977


namespace smallest_nineteen_multiple_l174_174771

theorem smallest_nineteen_multiple (n : ℕ) 
  (h₁ : 19 * n ≡ 5678 [MOD 11]) : n = 8 :=
by sorry

end smallest_nineteen_multiple_l174_174771


namespace carter_average_goals_l174_174056

theorem carter_average_goals (C : ℝ)
  (h1 : C + (1 / 2) * C + (C - 3) = 7) : C = 4 :=
by
  sorry

end carter_average_goals_l174_174056


namespace inequality_abc_l174_174924

variable (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
variable (cond : a + b + c = (1/a) + (1/b) + (1/c))

theorem inequality_abc : a + b + c ≥ 3 / (a * b * c) :=
by
  sorry

end inequality_abc_l174_174924


namespace bus_interval_duration_l174_174070

-- Definition of the conditions
def total_minutes : ℕ := 60
def total_buses : ℕ := 11
def intervals : ℕ := total_buses - 1

-- Theorem stating the interval between each bus departure
theorem bus_interval_duration : total_minutes / intervals = 6 := 
by
  -- The proof is omitted. 
  sorry

end bus_interval_duration_l174_174070


namespace binom_10_3_eq_120_l174_174316

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l174_174316


namespace loggers_count_l174_174830

theorem loggers_count 
  (cut_rate : ℕ) 
  (forest_width : ℕ) 
  (forest_height : ℕ) 
  (tree_density : ℕ) 
  (days_per_month : ℕ) 
  (months : ℕ) 
  (total_loggers : ℕ)
  (total_trees : ℕ := forest_width * forest_height * tree_density) 
  (total_days : ℕ := days_per_month * months)
  (trees_cut_down_per_logger : ℕ := cut_rate * total_days) 
  (expected_loggers : ℕ := total_trees / trees_cut_down_per_logger) 
  (h1: cut_rate = 6)
  (h2: forest_width = 4)
  (h3: forest_height = 6)
  (h4: tree_density = 600)
  (h5: days_per_month = 30)
  (h6: months = 10)
  (h7: total_loggers = expected_loggers)
: total_loggers = 8 := 
by {
    sorry
}

end loggers_count_l174_174830


namespace roots_of_polynomial_l174_174516

noncomputable def polynomial : Polynomial ℝ := Polynomial.mk [6, -11, 6, -1]

theorem roots_of_polynomial :
  (∃ r1 r2 r3 : ℝ, polynomial = (X - C r1) * (X - C r2) * (X - C r3) ∧ {r1, r2, r3} = {1, 2, 3}) :=
sorry

end roots_of_polynomial_l174_174516


namespace inequality_geq_l174_174920

variable {a b c : ℝ}

theorem inequality_geq (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 1 / a + 1 / b + 1 / c) : 
  a + b + c ≥ 3 / (a * b * c) := 
sorry

end inequality_geq_l174_174920


namespace binom_10_3_eq_120_l174_174215

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end binom_10_3_eq_120_l174_174215


namespace minValue_expression_l174_174082

noncomputable def minValue (x y : ℝ) : ℝ :=
  4 / x^2 + 4 / (x * y) + 1 / y^2

theorem minValue_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : (x - 2 * y)^2 = (x * y)^3) :
  minValue x y = 4 * Real.sqrt 2 :=
sorry

end minValue_expression_l174_174082


namespace combined_mass_of_individuals_l174_174477

-- Define constants and assumptions
def boat_length : ℝ := 4 -- in meters
def boat_breadth : ℝ := 3 -- in meters
def sink_depth_first_person : ℝ := 0.01 -- in meters (1 cm)
def sink_depth_second_person : ℝ := 0.02 -- in meters (2 cm)
def density_water : ℝ := 1000 -- in kg/m³ (density of freshwater)

-- Define volumes displaced
def volume_displaced_first : ℝ := boat_length * boat_breadth * sink_depth_first_person
def volume_displaced_both : ℝ := boat_length * boat_breadth * (sink_depth_first_person + sink_depth_second_person)

-- Define weights (which are equal to the masses under the assumption of constant gravity)
def weight_first_person : ℝ := volume_displaced_first * density_water
def weight_both_persons : ℝ := volume_displaced_both * density_water

-- Statement to prove the combined weight
theorem combined_mass_of_individuals : weight_both_persons = 360 :=
by
  -- Skip the proof
  sorry

end combined_mass_of_individuals_l174_174477


namespace remaining_dresses_pockets_count_l174_174569

-- Definitions translating each condition in the problem.
def total_dresses : Nat := 24
def dresses_with_pockets : Nat := total_dresses / 2
def dresses_with_two_pockets : Nat := dresses_with_pockets / 3
def total_pockets : Nat := 32

-- Question translated into a proof problem using Lean's logic.
theorem remaining_dresses_pockets_count :
  (total_pockets - (dresses_with_two_pockets * 2)) / (dresses_with_pockets - dresses_with_two_pockets) = 3 := by
  sorry

end remaining_dresses_pockets_count_l174_174569


namespace false_statement_l174_174030

-- Define the geometrical conditions based on the problem statements
variable {A B C D: Type}

-- A rhombus with equal diagonals is a square
def rhombus_with_equal_diagonals_is_square (R : A) : Prop := 
  ∀ (a b : A), a = b → true

-- A rectangle with perpendicular diagonals is a square
def rectangle_with_perpendicular_diagonals_is_square (Rec : B) : Prop :=
  ∀ (a b : B), a = b → true

-- A parallelogram with perpendicular and equal diagonals is a square
def parallelogram_with_perpendicular_and_equal_diagonals_is_square (P : C) : Prop :=
  ∀ (a b : C), a = b → true

-- A quadrilateral with perpendicular and bisecting diagonals is a square
def quadrilateral_with_perpendicular_and_bisecting_diagonals_is_square (Q : D) : Prop :=
  ∀ (a b : D), (a = b) → true 

-- The main theorem: Statement D is false
theorem false_statement (Q : D) : ¬ (quadrilateral_with_perpendicular_and_bisecting_diagonals_is_square Q) := 
  sorry

end false_statement_l174_174030


namespace meaningful_fraction_l174_174885

theorem meaningful_fraction (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 :=
by sorry

end meaningful_fraction_l174_174885


namespace binom_10_3_eq_120_l174_174214

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end binom_10_3_eq_120_l174_174214


namespace has_three_real_zeros_l174_174888

noncomputable def f (x m : ℝ) : ℝ := 2 * x^3 - 6 * x + m

theorem has_three_real_zeros (m : ℝ) : 
    (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ m = 0 ∧ f x₂ m = 0 ∧ f x₃ m = 0) ↔ (-4 < m ∧ m < 4) :=
sorry

end has_three_real_zeros_l174_174888


namespace arithmetic_sequence_is_a_l174_174453

theorem arithmetic_sequence_is_a
  (a : ℚ) (d : ℚ)
  (h1 : 140 + d = a)
  (h2 : a + d = 45 / 28)
  (h3 : a > 0) :
  a = 3965 / 56 :=
by
  sorry

end arithmetic_sequence_is_a_l174_174453


namespace find_s_t_l174_174568

theorem find_s_t 
  (FG GH EH : ℝ)
  (angleE angleF : ℝ)
  (h1 : FG = 10)
  (h2 : GH = 15)
  (h3 : EH = 12)
  (h4 : angleE = 45)
  (h5 : angleF = 45)
  (s t : ℕ)
  (h6 : 12 + 7.5 * Real.sqrt 2 = s + Real.sqrt t) :
  s + t = 5637 :=
sorry

end find_s_t_l174_174568


namespace constant_COG_of_mercury_column_l174_174978

theorem constant_COG_of_mercury_column (L : ℝ) (A : ℝ) (beta_g : ℝ) (beta_m : ℝ) (alpha_g : ℝ) (x : ℝ) :
  L = 1 ∧ A = 1e-4 ∧ beta_g = 1 / 38700 ∧ beta_m = 1 / 5550 ∧ alpha_g = beta_g / 3 ∧
  x = (2 / (3 * 38700)) / ((1 / 5550) - (2 / 116100)) →
  x = 0.106 :=
by
  sorry

end constant_COG_of_mercury_column_l174_174978


namespace longest_segment_is_CD_l174_174712

-- Define points A, B, C, D
def A := (-3, 0)
def B := (0, 2)
def C := (3, 0)
def D := (0, -1)

-- Angles in triangle ABD
def angle_ABD := 35
def angle_BAD := 95
def angle_ADB := 50

-- Angles in triangle BCD
def angle_BCD := 55
def angle_BDC := 60
def angle_CBD := 65

-- Length comparison conclusion from triangle ABD
axiom compare_lengths_ABD : ∀ (AD AB BD : ℝ), AD < AB ∧ AB < BD

-- Length comparison conclusion from triangle BCD
axiom compare_lengths_BCD : ∀ (BC BD CD : ℝ), BC < BD ∧ BD < CD

-- Combine results
theorem longest_segment_is_CD : ∀ (AD AB BD BC CD : ℝ), AD < AB → AB < BD → BC < BD → BD < CD → CD ≥ AD ∧ CD ≥ AB ∧ CD ≥ BD ∧ CD ≥ BC :=
by
  intros AD AB BD BC CD h1 h2 h3 h4
  sorry

end longest_segment_is_CD_l174_174712


namespace inequality_proof_l174_174799

theorem inequality_proof (x y z : ℝ) : 
  ( (x^3) / (x^3 + 2 * (y^2) * z) + 
    (y^3) / (y^3 + 2 * (z^2) * x) + 
    (z^3) / (z^3 + 2 * (x^2) * y) ) ≥ 1 := 
by 
  sorry

end inequality_proof_l174_174799


namespace mod_congruence_l174_174375

theorem mod_congruence (N : ℕ) (hN : N > 1) (h1 : 69 % N = 90 % N) (h2 : 90 % N = 125 % N) : 81 % N = 4 := 
by {
    sorry
}

end mod_congruence_l174_174375


namespace rick_books_division_l174_174586

theorem rick_books_division (books_per_group initial_books final_groups : ℕ) 
  (h_initial : initial_books = 400) 
  (h_books_per_group : books_per_group = 25) 
  (h_final_groups : final_groups = 16) : 
  ∃ divisions : ℕ, (divisions = 4) ∧ 
    ∃ f : ℕ → ℕ, 
    (f 0 = initial_books) ∧ 
    (f divisions = books_per_group * final_groups) ∧ 
    (∀ n, 1 ≤ n → n ≤ divisions → f n = f (n - 1) / 2) := 
by 
  sorry

end rick_books_division_l174_174586


namespace find_a5_l174_174710

variable {a_n : ℕ → ℤ} -- Type of the arithmetic sequence
variable (d : ℤ)       -- Common difference of the sequence

-- Assuming the sequence is defined as an arithmetic progression
axiom arithmetic_seq (a d : ℤ) : ∀ n : ℕ, a_n n = a + n * d

theorem find_a5
  (h : a_n 3 + a_n 4 + a_n 5 + a_n 6 + a_n 7 = 45):
  a_n 5 = 9 :=
by 
  sorry

end find_a5_l174_174710


namespace inequality_inequality_holds_l174_174781

theorem inequality_inequality_holds (x y z : ℝ) : 
  (x^3 / (x^3 + 2 * y^2 * z)) + (y^3 / (y^3 + 2 * z^2 * x)) + (z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_inequality_holds_l174_174781


namespace probability_student_less_than_25_l174_174630

-- Defining the problem conditions
def total_students : ℕ := 100
def percent_male : ℕ := 40
def percent_female : ℕ := 100 - percent_male
def percent_male_25_or_older : ℕ := 40
def percent_female_25_or_older : ℕ := 30

-- Calculation based on the conditions
def num_male_students := (percent_male * total_students) / 100
def num_female_students := (percent_female * total_students) / 100
def num_male_25_or_older := (percent_male_25_or_older * num_male_students) / 100
def num_female_25_or_older := (percent_female_25_or_older * num_female_students) / 100

def num_25_or_older := num_male_25_or_older + num_female_25_or_older
def num_less_than_25 := total_students - num_25_or_older
def probability_less_than_25 := (num_less_than_25: ℚ) / total_students

-- Define the theorem
theorem probability_student_less_than_25 :
  probability_less_than_25 = 0.66 := by
  sorry

end probability_student_less_than_25_l174_174630


namespace sum_of_fifth_powers_l174_174968

variable (a b c d : ℝ)

theorem sum_of_fifth_powers (h₁ : a + b = c + d) (h₂ : a^3 + b^3 = c^3 + d^3) :
  a^5 + b^5 = c^5 + d^5 := 
sorry

end sum_of_fifth_powers_l174_174968


namespace solve_system_l174_174403

variable (x y z : ℝ)

theorem solve_system :
  (y + z = 20 - 4 * x) →
  (x + z = -18 - 4 * y) →
  (x + y = 10 - 4 * z) →
  (2 * x + 2 * y + 2 * z = 4) :=
by
  intros h1 h2 h3
  sorry

end solve_system_l174_174403


namespace rationalize_denominator_ABC_l174_174433

theorem rationalize_denominator_ABC :
  let expr := (2 + Real.sqrt 5) / (3 - 2 * Real.sqrt 5)
  ∃ A B C : ℤ, expr = A + B * Real.sqrt C ∧ A * B * (C:ℤ) = -560 :=
by
  sorry

end rationalize_denominator_ABC_l174_174433


namespace inequality_holds_l174_174812

theorem inequality_holds (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by
  sorry

end inequality_holds_l174_174812


namespace probability_same_color_is_correct_l174_174637

-- Define the total number of each color marbles
def red_marbles : ℕ := 5
def white_marbles : ℕ := 6
def blue_marbles : ℕ := 7
def green_marbles : ℕ := 4

-- Define the total number of marbles
def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles + green_marbles

-- Define the probability calculation function
def probability_all_same_color : ℚ :=
  (red_marbles * (red_marbles - 1) * (red_marbles - 2) * (red_marbles - 3) / (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3))) +
  (white_marbles * (white_marbles - 1) * (white_marbles - 2) * (white_marbles - 3) / (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3))) +
  (blue_marbles * (blue_marbles - 1) * (blue_marbles - 2) * (blue_marbles - 3) / (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3))) +
  (green_marbles * (green_marbles - 1) * (green_marbles - 2) * (green_marbles - 3) / (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3)))

-- Define the theorem to prove the computed probability
theorem probability_same_color_is_correct :
  probability_all_same_color = 106 / 109725 := sorry

end probability_same_color_is_correct_l174_174637


namespace binomial_10_3_eq_120_l174_174208

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l174_174208


namespace mixed_number_calculation_l174_174178

theorem mixed_number_calculation :
  (481 + 1/6) + (265 + 1/12) + (904 + 1/20) - (184 + 29/30) - (160 + 41/42) - (703 + 55/56) = 603 + 3/8 := 
sorry

end mixed_number_calculation_l174_174178


namespace smaller_cube_volume_l174_174483

theorem smaller_cube_volume
  (V_L : ℝ) (N : ℝ) (SA_diff : ℝ) 
  (h1 : V_L = 8)
  (h2 : N = 8)
  (h3 : SA_diff = 24) :
  (∀ V_S : ℝ, V_L = N * V_S → V_S = 1) :=
by
  sorry

end smaller_cube_volume_l174_174483


namespace binomial_10_3_l174_174262

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end binomial_10_3_l174_174262


namespace infinite_gcd_one_l174_174584

theorem infinite_gcd_one : ∃ᶠ n in at_top, Int.gcd n ⌊Real.sqrt 2 * n⌋ = 1 := sorry

end infinite_gcd_one_l174_174584


namespace inverse_passes_through_point_l174_174084

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log 2 ((x - a) / (x + 1))

theorem inverse_passes_through_point 
  (h : ∀ x, (f a x = (-2 : ℝ)) ↔ x = 3) : a = 2 :=
by 
  sorry

end inverse_passes_through_point_l174_174084


namespace find_equation_of_line_l174_174394

-- Define the given conditions
def center_of_circle : ℝ × ℝ := (0, 3)
def perpendicular_line_slope : ℝ := -1
def perpendicular_line_equation (x y : ℝ) : Prop := x + y + 1 = 0

-- Define the proof problem
theorem find_equation_of_line (x y : ℝ) (l_passes_center : (x, y) = center_of_circle)
 (l_is_perpendicular : ∀ x y, perpendicular_line_equation x y ↔ (x-y+3=0)) : x - y + 3 = 0 :=
sorry

end find_equation_of_line_l174_174394


namespace binomial_10_3_l174_174298

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end binomial_10_3_l174_174298


namespace combination_10_3_l174_174291

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end combination_10_3_l174_174291


namespace necessary_but_not_sufficient_condition_l174_174903

theorem necessary_but_not_sufficient_condition (x : ℝ) (h : x > e) : x > 1 :=
sorry

end necessary_but_not_sufficient_condition_l174_174903


namespace hyperbola_eccentricity_l174_174937

theorem hyperbola_eccentricity (m : ℤ) (h1 : -2 < m) (h2 : m < 2) : 
  let a := m
  let b := (4 - m^2).sqrt 
  let c := (a^2 + b^2).sqrt
  let e := c / a
  e = 2 := by
sorry

end hyperbola_eccentricity_l174_174937


namespace cube_volume_of_surface_area_l174_174019

-- Define the condition: the surface area S is 864 square units
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- The proof problem: Given that the surface area of a cube is 864 square units,
-- prove that the volume of the cube is 1728 cubic units
theorem cube_volume_of_surface_area (S : ℝ) (hS : S = 864) : 
  ∃ V : ℝ, V = 1728 ∧ ∃ s : ℝ, surface_area s = S ∧ V = s^3 :=
by 
  sorry

end cube_volume_of_surface_area_l174_174019


namespace correct_factorization_l174_174472

theorem correct_factorization (a : ℝ) : 
  (a ^ 2 + 4 * a ≠ a ^ 2 * (a + 4)) ∧ 
  (a ^ 2 - 9 ≠ (a + 9) * (a - 9)) ∧ 
  (a ^ 2 + 4 * a + 2 ≠ (a + 2) ^ 2) → 
  (a ^ 2 - 2 * a + 1 = (a - 1) ^ 2) :=
by sorry

end correct_factorization_l174_174472


namespace transportation_tax_correct_l174_174504

def engine_power : ℕ := 250
def tax_rate : ℕ := 75
def months_owned : ℕ := 2
def total_months_in_year : ℕ := 12

def annual_tax : ℕ := engine_power * tax_rate
def adjusted_tax : ℕ := (annual_tax * months_owned) / total_months_in_year

theorem transportation_tax_correct :
  adjusted_tax = 3125 := by
  sorry

end transportation_tax_correct_l174_174504


namespace find_x_l174_174063

theorem find_x (x : ℝ) (h : ⌊x⌋ + x = 15/4) : x = 7/4 :=
sorry

end find_x_l174_174063


namespace inequality_1_inequality_2_inequality_3_inequality_4_l174_174411

-- Definitions of distances
def d_a : ℝ := sorry
def d_b : ℝ := sorry
def d_c : ℝ := sorry
def R_a : ℝ := sorry
def R_b : ℝ := sorry
def R_c : ℝ := sorry
def a : ℝ := sorry
def b : ℝ := sorry
def c : ℝ := sorry

def R : ℝ := sorry -- Circumradius
def r : ℝ := sorry -- Inradius

-- Inequality 1
theorem inequality_1 : a * R_a ≥ c * d_c + b * d_b := 
  sorry

-- Inequality 2
theorem inequality_2 : d_a * R_a + d_b * R_b + d_c * R_c ≥ 2 * (d_a * d_b + d_b * d_c + d_c * d_a) :=
  sorry

-- Inequality 3
theorem inequality_3 : R_a + R_b + R_c ≥ 2 * (d_a + d_b + d_c) :=
  sorry

-- Inequality 4
theorem inequality_4 : R_a * R_b * R_c ≥ (R / (2 * r)) * (d_a + d_b) * (d_b + d_c) * (d_c + d_a) :=
  sorry

end inequality_1_inequality_2_inequality_3_inequality_4_l174_174411


namespace combination_10_3_eq_120_l174_174353

open Nat

theorem combination_10_3_eq_120 : (10.choose 3) = 120 := 
by
  sorry

end combination_10_3_eq_120_l174_174353


namespace binomial_coefficient_10_3_l174_174270

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l174_174270


namespace dogwood_tree_count_l174_174947

theorem dogwood_tree_count (n d1 d2 d3 d4 d5: ℕ) 
  (h1: n = 39)
  (h2: d1 = 24)
  (h3: d2 = d1 / 2)
  (h4: d3 = 4 * d2)
  (h5: d4 = 5)
  (h6: d5 = 15):
  n + d1 + d2 + d3 + d4 + d5 = 143 :=
by
  sorry

end dogwood_tree_count_l174_174947


namespace maximum_p_value_l174_174391

noncomputable def max_p_value (a b c : ℝ) : ℝ :=
  2 / (a^2 + 1) - 2 / (b^2 + 1) + 3 / (c^2 + 1)

theorem maximum_p_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c + a + c = b) :
  ∃ p_max, p_max = 10 / 3 ∧ ∀ p, p = max_p_value a b c → p ≤ p_max :=
sorry

end maximum_p_value_l174_174391


namespace power_function_half_l174_174881

theorem power_function_half (a : ℝ) (ha : (4 : ℝ)^a / (2 : ℝ)^a = 3) : (1 / 2 : ℝ) ^ a = 1 / 3 := 
by
  sorry

end power_function_half_l174_174881


namespace divide_subtract_result_l174_174126

theorem divide_subtract_result (x : ℕ) (h : (x - 26) / 2 = 37) : 48 - (x / 4) = 23 := 
by
  sorry

end divide_subtract_result_l174_174126


namespace initial_incorrect_average_l174_174129

theorem initial_incorrect_average :
  let avg_correct := 24
  let incorrect_insertion := 26
  let correct_insertion := 76
  let n := 10  
  let correct_sum := avg_correct * n
  let incorrect_sum := correct_sum - correct_insertion + incorrect_insertion   
  avg_correct * n - correct_insertion + incorrect_insertion = incorrect_sum →
  incorrect_sum / n = 19 :=
by 
  sorry

end initial_incorrect_average_l174_174129


namespace inequality_proof_l174_174793

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 := 
by 
  sorry

end inequality_proof_l174_174793


namespace star_value_l174_174942

variable (a b : ℤ)
noncomputable def star (a b : ℤ) : ℚ := (1 : ℚ) / a + (1 : ℚ) / b

theorem star_value
  (h1 : a + b = 11)
  (h2 : a * b = 24)
  (h3 : a ≠ 0)
  (h4 : b ≠ 0) :
  star a b = 11 / 24 := by
  sorry

end star_value_l174_174942


namespace length_of_AB_l174_174474

theorem length_of_AB :
  ∃ (a b c d e : ℝ), (a < b) ∧ (b < c) ∧ (c < d) ∧ (d < e) ∧
  (b - a = 5) ∧ -- AB = 5
  ((c - b) = 2 * (d - c)) ∧ -- bc = 2 * cd
  (d - e) = 4 ∧ -- de = 4
  (c - a) = 11 ∧ -- ac = 11
  (e - a) = 18 := -- ae = 18
by 
  sorry

end length_of_AB_l174_174474


namespace binom_10_3_eq_120_l174_174216

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end binom_10_3_eq_120_l174_174216


namespace smallest_d_for_inverse_domain_l174_174573

noncomputable def g (x : ℝ) : ℝ := 2 * (x + 1)^2 - 7

theorem smallest_d_for_inverse_domain : ∃ d : ℝ, (∀ x1 x2 : ℝ, x1 ≥ d → x2 ≥ d → g x1 = g x2 → x1 = x2) ∧ d = -1 :=
by
  use -1
  constructor
  · sorry
  · rfl

end smallest_d_for_inverse_domain_l174_174573


namespace three_term_inequality_l174_174819

theorem three_term_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_abc : a * b * c = 1) :
  (a^3 / (a^3 + 2 * b^2)) + (b^3 / (b^3 + 2 * c^2)) + (c^3 / (c^3 + 2 * a^2)) ≥ 1 :=
by
  sorry

end three_term_inequality_l174_174819


namespace find_side_a_in_triangle_l174_174898

noncomputable def triangle_side_a (cosA : ℝ) (b : ℝ) (S : ℝ) (a : ℝ) : Prop :=
  cosA = 4/5 ∧ b = 2 ∧ S = 3 → a = Real.sqrt 13

-- Theorem statement with explicit conditions and proof goal
theorem find_side_a_in_triangle
  (cosA : ℝ) (b : ℝ) (S : ℝ) (a : ℝ) :
  cosA = 4 / 5 → b = 2 → S = 3 → a = Real.sqrt 13 :=
by 
  intros 
  sorry

end find_side_a_in_triangle_l174_174898


namespace tangent_line_at_2_l174_174068

open Real

noncomputable def curve (x : ℝ) : ℝ := 1 / x

def tangent_line_equation (p : ℝ × ℝ) : ℝ → ℝ := λ x, -(1 / 4) * (x - p.1) + p.2

theorem tangent_line_at_2 :
  ∀ x y : ℝ, curve 2 = 1 / 2 ∧ p = (2, 1 / 2) →
  y = tangent_line_equation p x →
  x + 4 * y - 4 = 0 :=
by
  intro x y h1 h2
  sorry

end tangent_line_at_2_l174_174068


namespace inequality_ge_one_l174_174806

theorem inequality_ge_one {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_ge_one_l174_174806


namespace a719_divisible_by_11_l174_174859

theorem a719_divisible_by_11 (a : ℕ) (h : a < 10) : (∃ k : ℤ, a - 15 = 11 * k) ↔ a = 4 :=
by
  sorry

end a719_divisible_by_11_l174_174859


namespace inequality_proof_l174_174792

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 := 
by 
  sorry

end inequality_proof_l174_174792


namespace rowing_distance_l174_174486

theorem rowing_distance
  (rowing_speed_in_still_water : ℝ)
  (velocity_of_current : ℝ)
  (total_time : ℝ)
  (H1 : rowing_speed_in_still_water = 5)
  (H2 : velocity_of_current = 1)
  (H3 : total_time = 1) :
  ∃ (D : ℝ), D = 2.4 := 
sorry

end rowing_distance_l174_174486


namespace rectangle_perimeter_l174_174407

theorem rectangle_perimeter 
  (w : ℝ) (l : ℝ) (hw : w = Real.sqrt 3) (hl : l = Real.sqrt 6) : 
  2 * (w + l) = 2 * Real.sqrt 3 + 2 * Real.sqrt 6 := 
by 
  sorry

end rectangle_perimeter_l174_174407


namespace evaluate_f_x_plus_3_l174_174739

def f (x : ℝ) : ℝ := x^2

theorem evaluate_f_x_plus_3 (x : ℝ) : f (x + 3) = x^2 + 6 * x + 9 := by
  sorry

end evaluate_f_x_plus_3_l174_174739


namespace sum_of_b_for_unique_solution_l174_174139

theorem sum_of_b_for_unique_solution :
  (∃ b1 b2, (3 * (0:ℝ)^2 + (b1 + 6) * 0 + 7 = 0 ∧ 3 * (0:ℝ)^2 + (b2 + 6) * 0 + 7 = 0) ∧ 
   ((b1 + 6)^2 - 4 * 3 * 7 = 0) ∧ ((b2 + 6)^2 - 4 * 3 * 7 = 0) ∧ 
   b1 + b2 = -12)  :=
by
  sorry

end sum_of_b_for_unique_solution_l174_174139


namespace base9_sum_correct_l174_174651

def base9_addition (a b c : ℕ) : ℕ :=
  a + b + c

theorem base9_sum_correct :
  base9_addition (263) (452) (247) = 1073 :=
by sorry

end base9_sum_correct_l174_174651


namespace ratio_b4_b3_a2_a1_l174_174730

variables {x y d d' : ℝ}
variables {a1 a2 a3 b1 b2 b3 b4 : ℝ}
-- Conditions
variables (h1 : x ≠ y)
variables (h2 : a1 = x + d)
variables (h3 : a2 = x + 2 * d)
variables (h4 : a3 = x + 3 * d)
variables (h5 : y = x + 4 * d)
variables (h6 : b2 = x + d')
variables (h7 : b3 = x + 2 * d')
variables (h8 : y = x + 3 * d')
variables (h9 : b4 = x + 4 * d')

theorem ratio_b4_b3_a2_a1 :
  (b4 - b3) / (a2 - a1) = 8 / 3 :=
by sorry

end ratio_b4_b3_a2_a1_l174_174730


namespace binom_10_3_l174_174335

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end binom_10_3_l174_174335


namespace points_connected_l174_174680

theorem points_connected (m l : ℕ) (h1 : l < m) (h2 : Even (l * m)) :
  ∃ points : Finset (ℕ × ℕ), ∀ p ∈ points, (∃ q, q ∈ points ∧ (p ≠ q → p.snd = q.snd → p.fst = q.fst)) :=
sorry

end points_connected_l174_174680


namespace prime_factor_of_sum_of_four_consecutive_integers_l174_174061

theorem prime_factor_of_sum_of_four_consecutive_integers (n : ℤ) : 
  2 ∣ ((n - 1) + n + (n + 1) + (n + 2)) :=
by 
  sorry

end prime_factor_of_sum_of_four_consecutive_integers_l174_174061


namespace find_minimal_product_l174_174501

theorem find_minimal_product : ∃ x y : ℤ, (20 * x + 19 * y = 2019) ∧ (x * y = 2623) ∧ (∀ z w : ℤ, (20 * z + 19 * w = 2019) → |x - y| ≤ |z - w|) :=
by
  -- definitions and theorems to prove the problem would be placed here
  sorry

end find_minimal_product_l174_174501


namespace inequality_abc_l174_174928

variable (a b c : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hc : c > 0)
variable (cond : a + b + c = (1 / a) + (1 / b) + (1 / c))

theorem inequality_abc : a + b + c ≥ 3 / (a * b * c) :=
sorry

end inequality_abc_l174_174928


namespace combination_10_3_l174_174286

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end combination_10_3_l174_174286


namespace binom_10_3_l174_174334

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end binom_10_3_l174_174334


namespace binomial_10_3_eq_120_l174_174344

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l174_174344


namespace cube_volume_of_surface_area_l174_174017

-- Define the condition: the surface area S is 864 square units
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- The proof problem: Given that the surface area of a cube is 864 square units,
-- prove that the volume of the cube is 1728 cubic units
theorem cube_volume_of_surface_area (S : ℝ) (hS : S = 864) : 
  ∃ V : ℝ, V = 1728 ∧ ∃ s : ℝ, surface_area s = S ∧ V = s^3 :=
by 
  sorry

end cube_volume_of_surface_area_l174_174017


namespace inequality_inequality_holds_l174_174783

theorem inequality_inequality_holds (x y z : ℝ) : 
  (x^3 / (x^3 + 2 * y^2 * z)) + (y^3 / (y^3 + 2 * z^2 * x)) + (z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_inequality_holds_l174_174783


namespace total_musicians_is_98_l174_174458

-- Define the number of males and females in the orchestra
def males_in_orchestra : ℕ := 11
def females_in_orchestra : ℕ := 12

-- Define the total number of musicians in the orchestra
def total_in_orchestra : ℕ := males_in_orchestra + females_in_orchestra

-- Define the number of musicians in the band as twice the number in the orchestra
def total_in_band : ℕ := 2 * total_in_orchestra

-- Define the number of males and females in the choir
def males_in_choir : ℕ := 12
def females_in_choir : ℕ := 17

-- Define the total number of musicians in the choir
def total_in_choir : ℕ := males_in_choir + females_in_choir

-- Prove that the total number of musicians in the orchestra, band, and choir is 98
theorem total_musicians_is_98 : total_in_orchestra + total_in_band + total_in_choir = 98 :=
by {
  -- Adding placeholders for the proof steps
  sorry
}

end total_musicians_is_98_l174_174458


namespace cos_alpha_solution_l174_174074

theorem cos_alpha_solution (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.tan α = 1 / 2) : 
  Real.cos α = 2 * Real.sqrt 5 / 5 :=
by
  sorry

end cos_alpha_solution_l174_174074


namespace sum_of_fifth_powers_l174_174966

variable (a b c d : ℝ)

theorem sum_of_fifth_powers (h₁ : a + b = c + d) (h₂ : a^3 + b^3 = c^3 + d^3) :
  a^5 + b^5 = c^5 + d^5 := 
sorry

end sum_of_fifth_powers_l174_174966


namespace mackenzie_new_disks_l174_174654

noncomputable def price_new (U N : ℝ) : Prop := 6 * N + 2 * U = 127.92

noncomputable def disks_mackenzie_buys (U N x : ℝ) : Prop := x * N + 8 * U = 133.89

theorem mackenzie_new_disks (U N x : ℝ) (h1 : U = 9.99) (h2 : price_new U N) (h3 : disks_mackenzie_buys U N x) :
  x = 3 :=
by
  sorry

end mackenzie_new_disks_l174_174654


namespace simplest_quadratic_radicals_same_type_l174_174406

theorem simplest_quadratic_radicals_same_type (m n : ℕ)
  (h : ∀ {a : ℕ}, (a = m - 1 → a = 2) ∧ (a = 4 * n - 1 → a = 7)) :
  m + n = 5 :=
sorry

end simplest_quadratic_radicals_same_type_l174_174406


namespace combination_10_3_eq_120_l174_174244

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end combination_10_3_eq_120_l174_174244


namespace juniper_initial_bones_l174_174716

theorem juniper_initial_bones (B : ℕ) (h : 2 * B - 2 = 6) : B = 4 := 
by
  sorry

end juniper_initial_bones_l174_174716


namespace geometric_seq_arith_condition_half_l174_174676

variables {a : ℕ → ℝ} {q : ℝ}

-- Conditions from the problem
def geometric_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q
def positive_terms (a : ℕ → ℝ) := ∀ n, a n > 0
def arithmetic_condition (a : ℕ → ℝ) (q : ℝ) := 
  a 1 = q * a 0 ∧ (1/2 : ℝ) * a 2 = a 1 + 2 * a 0

-- The statement to be proven
theorem geometric_seq_arith_condition_half (a : ℕ → ℝ) (q : ℝ) :
  geometric_seq a q →
  positive_terms a →
  arithmetic_condition a q →
  q = 2 →
  (a 2 + a 3) / (a 3 + a 4) = 1 / 2 :=
by
  intros h1 h2 h3 hq
  sorry

end geometric_seq_arith_condition_half_l174_174676


namespace min_value_of_sum_eq_l174_174389

theorem min_value_of_sum_eq : ∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * a + b = a * b - 1 → a + 2 * b = 5 + 2 * Real.sqrt 6 :=
by
  intros a b h
  sorry

end min_value_of_sum_eq_l174_174389


namespace binom_10_3_eq_120_l174_174189

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l174_174189


namespace inequality_ge_one_l174_174803

theorem inequality_ge_one {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_ge_one_l174_174803


namespace tan_of_acute_angle_l174_174541

theorem tan_of_acute_angle (α : ℝ) (h1 : α > 0 ∧ α < π / 2) (h2 : Real.cos (π / 2 + α) = -3/5) : Real.tan α = 3 / 4 :=
by
  sorry

end tan_of_acute_angle_l174_174541


namespace Pam_read_more_than_Harrison_l174_174751

theorem Pam_read_more_than_Harrison :
  ∀ (assigned : ℕ) (Harrison : ℕ) (Pam : ℕ) (Sam : ℕ),
    assigned = 25 →
    Harrison = assigned + 10 →
    Sam = 2 * Pam →
    Sam = 100 →
    Pam - Harrison = 15 :=
by
  intros assigned Harrison Pam Sam h1 h2 h3 h4
  sorry

end Pam_read_more_than_Harrison_l174_174751


namespace player_B_wins_l174_174431

variable {R : Type*} [Ring R]

noncomputable def polynomial_game (n : ℕ) (f : Polynomial R) : Prop :=
  (f.degree = 2 * n) ∧ (∃ (a b : R) (x y : R), f.eval x = 0 ∨ f.eval y = 0)

theorem player_B_wins (n : ℕ) (f : Polynomial ℝ)
  (h1 : n ≥ 2)
  (h2 : f.degree = 2 * n) :
  polynomial_game n f :=
by
  sorry

end player_B_wins_l174_174431


namespace unique_positive_integer_satisfies_condition_l174_174132

def is_positive_integer (n : ℕ) : Prop := n > 0

def condition (n : ℕ) : Prop := 20 - 5 * n ≥ 15

theorem unique_positive_integer_satisfies_condition :
  ∃! n : ℕ, is_positive_integer n ∧ condition n :=
by
  sorry

end unique_positive_integer_satisfies_condition_l174_174132


namespace angle_Z_90_l174_174706

-- Definitions and conditions from step a)
def Triangle (X Y Z : ℝ) : Prop :=
  X + Y + Z = 180

def in_triangle_XYZ (X Y Z : ℝ) : Prop :=
  Triangle X Y Z ∧ (X + Y = 90)

-- Proof problem from step c)
theorem angle_Z_90 (X Y Z : ℝ) (h : in_triangle_XYZ X Y Z) : Z = 90 :=
  by
  sorry

end angle_Z_90_l174_174706


namespace total_weekly_sleep_correct_l174_174479

-- Definition of the weekly sleep time for cougar, zebra, and lion
def cougar_sleep_even_days : Nat := 4
def cougar_sleep_odd_days : Nat := 6
def zebra_sleep_even_days := (cougar_sleep_even_days + 2)
def zebra_sleep_odd_days := (cougar_sleep_odd_days + 2)
def lion_sleep_even_days := (zebra_sleep_even_days - 3)
def lion_sleep_odd_days := (cougar_sleep_odd_days + 1)

def total_weekly_sleep_time : Nat :=
  (4 * cougar_sleep_odd_days + 3 * cougar_sleep_even_days) + -- Cougar's total sleep in a week
  (4 * zebra_sleep_odd_days + 3 * zebra_sleep_even_days) + -- Zebra's total sleep in a week
  (4 * lion_sleep_odd_days + 3 * lion_sleep_even_days) -- Lion's total sleep in a week

theorem total_weekly_sleep_correct : total_weekly_sleep_time = 123 := 
by
  -- Total for the week according to given conditions
  sorry -- Proof is omitted, only the statement is required

end total_weekly_sleep_correct_l174_174479


namespace dice_probability_sum_18_l174_174891

theorem dice_probability_sum_18 : 
  (∃ d1 d2 d3 : ℕ, 1 ≤ d1 ∧ d1 ≤ 8 ∧ 1 ≤ d2 ∧ d2 ≤ 8 ∧ 1 ≤ d3 ∧ d3 ≤ 8 ∧ d1 + d2 + d3 = 18) →
  (1/8 : ℚ) * (1/8) * (1/8) * 9 = 9 / 512 :=
by 
  sorry

end dice_probability_sum_18_l174_174891


namespace combination_10_3_eq_120_l174_174248

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end combination_10_3_eq_120_l174_174248


namespace part_a_part_b_l174_174970

-- Part (a)
theorem part_a {a b c d : ℝ} (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) :
  a^5 + b^5 = c^5 + d^5 :=
sorry

-- Part (b)
theorem part_b {a b c d : ℝ} (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) :
  ¬(a^4 + b^4 = c^4 + d^4) :=
counter_example

end part_a_part_b_l174_174970


namespace max_property_l174_174729

noncomputable def f : ℚ → ℚ := sorry

axiom f_zero : f 0 = 0
axiom f_pos_of_nonzero : ∀ α : ℚ, α ≠ 0 → f α > 0
axiom f_mul : ∀ α β : ℚ, f (α * β) = f α * f β
axiom f_add : ∀ α β : ℚ, f (α + β) ≤ f α + f β
axiom f_bounded_by_1989 : ∀ m : ℤ, f m ≤ 1989

theorem max_property (α β : ℚ) (h : f α ≠ f β) : f (α + β) = max (f α) (f β) := sorry

end max_property_l174_174729


namespace sandy_shopping_l174_174123

variable (X : ℝ)

theorem sandy_shopping (h : 0.70 * X = 210) : X = 300 := by
  sorry

end sandy_shopping_l174_174123


namespace volume_region_cone_sphere_l174_174645

noncomputable def volume_of_region_between_cone_and_sphere (R α : ℝ) : ℝ :=
  (4 / 3) * Real.pi * R^3 * (Real.sin (Real.pi / 4 - α / 2))^4 / (Real.sin α)

theorem volume_region_cone_sphere (R α : ℝ) :
  volume_of_region_between_cone_and_sphere R α = 
    (4 / 3) * Real.pi * R^3 * (Real.sin (Real.pi / 4 - α / 2))^4 / (Real.sin α) :=
by
  -- To be proved
  sorry

end volume_region_cone_sphere_l174_174645


namespace problem_1_l174_174576

open Set

variable (R : Set ℝ)
variable (A : Set ℝ := { x | 2 * x^2 - 7 * x + 3 ≤ 0 })
variable (B : Set ℝ := { x | x^2 + a < 0 })

theorem problem_1 (a : ℝ) : (a = -4 → (A ∩ B = { x : ℝ | 1 / 2 ≤ x ∧ x < 2 } ∧ A ∪ B = { x : ℝ | -2 < x ∧ x ≤ 3 })) ∧
  ((compl A ∩ B = B) → a ≥ -2) := by
  sorry

end problem_1_l174_174576


namespace marge_final_plants_l174_174426

-- Definitions corresponding to the conditions
def seeds_planted := 23
def seeds_never_grew := 5
def plants_grew := seeds_planted - seeds_never_grew
def plants_eaten := plants_grew / 3
def uneaten_plants := plants_grew - plants_eaten
def plants_strangled := uneaten_plants / 3
def survived_plants := uneaten_plants - plants_strangled
def effective_addition := 1

-- The main statement we need to prove
theorem marge_final_plants : 
  (plants_grew - plants_eaten - plants_strangled + effective_addition) = 9 := 
by
  sorry

end marge_final_plants_l174_174426


namespace maximum_fraction_l174_174045

theorem maximum_fraction (a b h : ℝ) (d : ℝ) (h_d_def : d = Real.sqrt (a^2 + b^2 + h^2)) :
  (a + b + h) / d ≤ Real.sqrt 3 :=
sorry

end maximum_fraction_l174_174045


namespace magnitude_of_power_l174_174852

noncomputable def z : ℂ := 4 + 2 * Real.sqrt 2 * Complex.I

theorem magnitude_of_power :
  Complex.abs (z ^ 4) = 576 := by
  sorry

end magnitude_of_power_l174_174852


namespace compute_product_l174_174463

-- Define the conditions
variables {x y : ℝ} (h1 : x - y = 5) (h2 : x^3 - y^3 = 35)

-- Define the theorem to be proved
theorem compute_product (h1 : x - y = 5) (h2 : x^3 - y^3 = 35) : x * y = 190 / 9 := 
sorry

end compute_product_l174_174463


namespace area_of_trapezium_l174_174668

-- Definitions for the problem conditions
def base1 : ℝ := 20
def base2 : ℝ := 18
def height : ℝ := 5

-- The theorem to prove
theorem area_of_trapezium : 
  1 / 2 * (base1 + base2) * height = 95 :=
by
  sorry

end area_of_trapezium_l174_174668


namespace three_term_inequality_l174_174825

theorem three_term_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_abc : a * b * c = 1) :
  (a^3 / (a^3 + 2 * b^2)) + (b^3 / (b^3 + 2 * c^2)) + (c^3 / (c^3 + 2 * a^2)) ≥ 1 :=
by
  sorry

end three_term_inequality_l174_174825


namespace tan_alpha_minus_beta_alpha_plus_beta_l174_174385

variable (α β : ℝ)

-- Conditions as hypotheses
axiom tan_alpha : Real.tan α = 2
axiom tan_beta : Real.tan β = -1 / 3
axiom alpha_range : 0 < α ∧ α < Real.pi / 2
axiom beta_range : Real.pi / 2 < β ∧ β < Real.pi

-- Proof statements
theorem tan_alpha_minus_beta : Real.tan (α - β) = 7 := by
  sorry

theorem alpha_plus_beta : α + β = 5 * Real.pi / 4 := by
  sorry

end tan_alpha_minus_beta_alpha_plus_beta_l174_174385


namespace binom_10_3_l174_174333

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end binom_10_3_l174_174333


namespace midpoint_of_five_points_on_grid_l174_174580

theorem midpoint_of_five_points_on_grid 
    (points : Fin 5 → ℤ × ℤ) :
    ∃ i j : Fin 5, i ≠ j ∧ ((points i).fst + (points j).fst) % 2 = 0 
    ∧ ((points i).snd + (points j).snd) % 2 = 0 :=
by sorry

end midpoint_of_five_points_on_grid_l174_174580


namespace probability_X_interval_l174_174612

noncomputable def fx (x c : ℝ) : ℝ :=
  if -c ≤ x ∧ x ≤ c then (1 / c) * (1 - (|x| / c))
  else 0

theorem probability_X_interval (c : ℝ) (hc : 0 < c) :
  (∫ x in (c / 2)..c, fx x c) = 1 / 8 :=
sorry

end probability_X_interval_l174_174612


namespace probability_function_meaningful_l174_174121

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

def is_meaningful (x : ℝ) : Prop := 1 - x^2 > 0

def measure_interval (a b : ℝ) : ℝ := b - a

theorem probability_function_meaningful:
  let interval_a := -2
  let interval_b := 1
  let meaningful_a := -1
  let meaningful_b := 1
  let total_interval := measure_interval interval_a interval_b
  let meaningful_interval := measure_interval meaningful_a meaningful_b
  let P := meaningful_interval / total_interval
  (P = (2/3)) :=
by
  sorry

end probability_function_meaningful_l174_174121


namespace common_divisors_count_l174_174551

def prime_exponents (n : Nat) : List (Nat × Nat) :=
  if n = 9240 then [(2, 3), (3, 1), (5, 1), (7, 1), (11, 1)]
  else if n = 10800 then [(2, 4), (3, 3), (5, 2)]
  else []

def gcd_prime_exponents (exps1 exps2 : List (Nat × Nat)) : List (Nat × Nat) :=
  exps1.filterMap (fun (p1, e1) =>
    match exps2.find? (fun (p2, _) => p1 = p2) with
    | some (p2, e2) => if e1 ≤ e2 then some (p1, e1) else some (p1, e2)
    | none => none
  )

def count_divisors (exps : List (Nat × Nat)) : Nat :=
  exps.foldl (fun acc (_, e) => acc * (e + 1)) 1

theorem common_divisors_count :
  count_divisors (gcd_prime_exponents (prime_exponents 9240) (prime_exponents 10800)) = 16 :=
by
  sorry

end common_divisors_count_l174_174551


namespace range_of_k_l174_174564

noncomputable section

open Classical

variables {A B C k : ℝ}

def is_acute_triangle (A B C : ℝ) := A < 90 ∧ B < 90 ∧ C < 90

theorem range_of_k (hA : A = 60) (hBC : BC = 6) (h_acute : is_acute_triangle A B C) : 
  2 * Real.sqrt 3 < k ∧ k < 4 * Real.sqrt 3 :=
sorry

end range_of_k_l174_174564


namespace friend_wants_to_take_5_marbles_l174_174430

theorem friend_wants_to_take_5_marbles
  (total_marbles : ℝ)
  (clear_marbles : ℝ)
  (black_marbles : ℝ)
  (other_marbles : ℝ)
  (friend_marbles : ℝ)
  (h1 : clear_marbles = 0.4 * total_marbles)
  (h2 : black_marbles = 0.2 * total_marbles)
  (h3 : other_marbles = total_marbles - clear_marbles - black_marbles)
  (h4 : friend_marbles = 2)
  (friend_total_marbles : ℝ)
  (h5 : friend_marbles = 0.4 * friend_total_marbles) :
  friend_total_marbles = 5 := by
  sorry

end friend_wants_to_take_5_marbles_l174_174430


namespace binomial_10_3_eq_120_l174_174341

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l174_174341


namespace task_assignment_l174_174882

theorem task_assignment (volunteers : ℕ) (tasks : ℕ) (selected : ℕ) (h_volunteers : volunteers = 6) (h_tasks : tasks = 4) (h_selected : selected = 4) :
  ((Nat.factorial volunteers) / (Nat.factorial (volunteers - selected))) = 360 :=
by
  rw [h_volunteers, h_selected]
  norm_num
  sorry

end task_assignment_l174_174882


namespace cube_volume_from_surface_area_l174_174008

theorem cube_volume_from_surface_area (A : ℕ) (h1 : A = 864) : 
  ∃ V : ℕ, V = 1728 :=
by
  sorry

end cube_volume_from_surface_area_l174_174008


namespace common_roots_exist_and_unique_l174_174861

noncomputable def polynomial1 (a : ℝ) : Polynomial ℝ :=
  Polynomial.X ^ 3 + Polynomial.C a * Polynomial.X ^ 2 + Polynomial.C 20 * Polynomial.X + Polynomial.C 10

noncomputable def polynomial2 (b : ℝ) : Polynomial ℝ :=
  Polynomial.X ^ 3 + Polynomial.C b * Polynomial.X ^ 2 + Polynomial.C 17 * Polynomial.X + Polynomial.C 12

theorem common_roots_exist_and_unique (a b : ℝ) :
  (∃ r s : ℝ, r ≠ s ∧ Polynomial.is_root (polynomial1 a) r ∧ Polynomial.is_root (polynomial1 a) s ∧
     Polynomial.is_root (polynomial2 b) r ∧ Polynomial.is_root (polynomial2 b) s) ↔ (a = 1 ∧ b = 0) :=
begin
  sorry
end

end common_roots_exist_and_unique_l174_174861


namespace geometric_sequence_value_sum_l174_174895

variable {a : ℕ → ℝ}

noncomputable def is_geometric_sequence (a : ℕ → ℝ) :=
  ∀ n m, a (n + m) * a 0 = a n * a m

theorem geometric_sequence_value_sum {a : ℕ → ℝ}
  (hpos : ∀ n, a n > 0)
  (geom : is_geometric_sequence a)
  (given : a 0 * a 2 + 2 * a 1 * a 3 + a 2 * a 4 = 16) 
  : a 1 + a 3 = 4 :=
sorry

end geometric_sequence_value_sum_l174_174895


namespace cube_volume_l174_174027

theorem cube_volume (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
sorry

end cube_volume_l174_174027


namespace monotonicity_of_f_f_greater_than_2_ln_a_plus_3_div_2_l174_174691

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem monotonicity_of_f (a : ℝ) :
  (a ≤ 0 → ∀ x y, x < y → f x a > f y a) ∧
  (a > 0 →
    (∀ x, x < Real.log (1 / a) → f x a > f (Real.log (1 / a)) a) ∧
    (∀ x, x > Real.log (1 / a) → f x a > f (Real.log (1 / a)) a)) :=
sorry

theorem f_greater_than_2_ln_a_plus_3_div_2 (a : ℝ) (h : a > 0) (x : ℝ) :
  f x a > 2 * Real.log a + 3 / 2 :=
sorry

end monotonicity_of_f_f_greater_than_2_ln_a_plus_3_div_2_l174_174691


namespace function_positive_on_interval_l174_174746

theorem function_positive_on_interval (a : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → (2 - a^2) * x + a > 0) ↔ 0 < a ∧ a < 2 :=
by
  sorry

end function_positive_on_interval_l174_174746


namespace sum_q_p_values_l174_174853

def p (x : ℤ) : ℤ := x^2 - 4
def q (x : ℤ) : ℤ := -x

def q_p_composed (x : ℤ) : ℤ := q (p x)

theorem sum_q_p_values :
  q_p_composed (-3) + q_p_composed (-2) + q_p_composed (-1) + q_p_composed 0 + 
  q_p_composed 1 + q_p_composed 2 + q_p_composed 3 = 0 := by
  sorry

end sum_q_p_values_l174_174853


namespace tangent_line_at_one_l174_174397

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + Real.log x

theorem tangent_line_at_one (a : ℝ)
  (h : ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - 1| ∧ |x - 1| < δ → |(f a x - f a 1) / (x - 1) - 3| < ε) :
  ∃ m b, m = 3 ∧ b = -2 ∧ (∀ x y, y = f a x → m * x = y + b) := sorry

end tangent_line_at_one_l174_174397


namespace sum_g_eq_half_l174_174107

noncomputable def g (n : ℕ) : ℝ := ∑' k, if h : k ≥ 3 then (1 / (k : ℝ) ^ n) else 0

theorem sum_g_eq_half : (∑' n, if h : n ≥ 3 then g n else 0) = 1 / 2 := by
  sorry

end sum_g_eq_half_l174_174107


namespace Bernardo_wins_with_smallest_M_l174_174176

-- Define the operations
def Bernardo_op (n : ℕ) : ℕ := 3 * n
def Lucas_op (n : ℕ) : ℕ := n + 75

-- Define the game behavior
def game_sequence (M : ℕ) : List ℕ :=
  [M, Bernardo_op M, Lucas_op (Bernardo_op M), Bernardo_op (Lucas_op (Bernardo_op M)),
   Lucas_op (Bernardo_op (Lucas_op (Bernardo_op M))),
   Bernardo_op (Lucas_op (Bernardo_op (Lucas_op (Bernardo_op M)))),
   Lucas_op (Bernardo_op (Lucas_op (Bernardo_op (Lucas_op (Bernardo_op M))))),
   Bernardo_op (Lucas_op (Bernardo_op (Lucas_op (Bernardo_op (Lucas_op (Bernardo_op M))))))]

-- Define winning condition
def Bernardo_wins (M : ℕ) : Prop :=
  let seq := game_sequence M
  seq.get! 5 < 1200 ∧ seq.get! 6 >= 1200

-- Sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

-- The final theorem statement
theorem Bernardo_wins_with_smallest_M :
  Bernardo_wins 9 ∧ (∀ M < 9, ¬Bernardo_wins M) ∧ sum_of_digits 9 = 9 :=
by
  sorry

end Bernardo_wins_with_smallest_M_l174_174176


namespace inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8_l174_174532

theorem inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
    1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 := 
sorry

end inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8_l174_174532


namespace matrix_multiplication_problem_l174_174104

variable {A B : Matrix (Fin 2) (Fin 2) ℝ}

theorem matrix_multiplication_problem 
  (h1 : A + B = A * B)
  (h2 : A * B = ![![5, 2], ![-2, 4]]) :
  B * A = ![![5, 2], ![-2, 4]] :=
sorry

end matrix_multiplication_problem_l174_174104


namespace cos_B_value_l174_174060

-- Define the sides of the triangle
def AB : ℝ := 8
def AC : ℝ := 10
def right_angle_at_A : Prop := true

-- Define the cosine function within the context of the given triangle
noncomputable def cos_B : ℝ := AB / AC

-- The proof statement asserting the condition
theorem cos_B_value : cos_B = 4 / 5 :=
by
  -- Given conditions
  have h1 : AB = 8 := rfl
  have h2 : AC = 10 := rfl
  -- Direct computation
  sorry

end cos_B_value_l174_174060


namespace binomial_10_3_l174_174258

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end binomial_10_3_l174_174258


namespace binom_10_3_l174_174232

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end binom_10_3_l174_174232


namespace cost_of_sunglasses_l174_174838

noncomputable def cost_per_pair 
  (price_per_pair : ℕ) 
  (pairs_sold : ℕ) 
  (sign_cost : ℕ) 
  (profits_half : ℕ) 
  (profit : ℕ) : ℕ :=
  let total_revenue := price_per_pair * pairs_sold in
  let total_cost := total_revenue - (profits_half * 2) in
  total_cost / pairs_sold

theorem cost_of_sunglasses :
  ∀ (price_per_pair pairs_sold sign_cost profits_half profit : ℕ),
    price_per_pair = 30 → 
    pairs_sold = 10 → 
    sign_cost = 20 → 
    (profits_half = sign_cost → profit = profits_half * 2) →
    cost_per_pair price_per_pair pairs_sold sign_cost profits_half profit = 26 :=
begin
  intros,
  sorry
end

end cost_of_sunglasses_l174_174838


namespace find_stamps_l174_174776

def stamps_problem (x y : ℕ) : Prop :=
  (x + y = 70) ∧ (y = 4 * x + 5)

theorem find_stamps (x y : ℕ) (h : stamps_problem x y) : 
  x = 13 ∧ y = 57 :=
sorry

end find_stamps_l174_174776


namespace combination_10_3_l174_174284

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end combination_10_3_l174_174284


namespace slope_of_parallel_line_l174_174622

-- Given condition: the equation of the line
def line_equation (x y : ℝ) : Prop := 2 * x - 4 * y = 9

-- Goal: the slope of any line parallel to the given line is 1/2
theorem slope_of_parallel_line (x y : ℝ) (m : ℝ) :
  (∀ x y, line_equation x y) → m = 1 / 2 := by
  sorry

end slope_of_parallel_line_l174_174622


namespace combination_10_3_eq_120_l174_174355

open Nat

theorem combination_10_3_eq_120 : (10.choose 3) = 120 := 
by
  sorry

end combination_10_3_eq_120_l174_174355


namespace inequality_solution_l174_174380

theorem inequality_solution (x : ℝ) : x^3 - 12 * x^2 > -36 * x ↔ x ∈ Set.Ioo 0 6 ∪ Set.Ioi 6 := by
  sorry

end inequality_solution_l174_174380


namespace rectangle_inscribed_area_l174_174660

variables (b h x : ℝ) 

theorem rectangle_inscribed_area (hb : 0 < b) (hh : 0 < h) (hx : 0 < x) (hx_lt_h : x < h) :
  ∃ A, A = (b * x * (h - x)) / h :=
sorry

end rectangle_inscribed_area_l174_174660


namespace binomial_10_3_eq_120_l174_174206

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l174_174206


namespace binomial_coefficient_10_3_l174_174279

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l174_174279


namespace cube_volume_l174_174023

theorem cube_volume (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
sorry

end cube_volume_l174_174023


namespace binomial_10_3_eq_120_l174_174347

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l174_174347


namespace prime_factorization_of_expression_l174_174111

theorem prime_factorization_of_expression :
  2 * 3 * 5 * 7 - 1 = 11 * 19 :=
sorry

end prime_factorization_of_expression_l174_174111


namespace sum_of_fifth_powers_l174_174967

variable (a b c d : ℝ)

theorem sum_of_fifth_powers (h₁ : a + b = c + d) (h₂ : a^3 + b^3 = c^3 + d^3) :
  a^5 + b^5 = c^5 + d^5 := 
sorry

end sum_of_fifth_powers_l174_174967


namespace slope_of_line_intersecting_hyperbola_l174_174684

theorem slope_of_line_intersecting_hyperbola 
  (A B : ℝ × ℝ)
  (hA : A.1^2 - A.2^2 = 1)
  (hB : B.1^2 - B.2^2 = 1)
  (midpoint_condition : (A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = 1) :
  (B.2 - A.2) / (B.1 - A.1) = 2 :=
by
  sorry

end slope_of_line_intersecting_hyperbola_l174_174684


namespace area_of_triangle_BCD_l174_174711

-- Define the points A, B, C, D
variables {A B C D : Type} 

-- Define the lengths of segments AC and CD
variables (AC CD : ℝ)
-- Define the area of triangle ABC
variables (area_ABC : ℝ)

-- Define height h
variables (h : ℝ)

-- Initial conditions
axiom length_AC : AC = 9
axiom length_CD : CD = 39
axiom area_ABC_is_36 : area_ABC = 36
axiom height_is_8 : h = (2 * area_ABC) / AC

-- Define the area of triangle BCD
def area_BCD (CD h : ℝ) : ℝ := 0.5 * CD * h

-- The theorem that we want to prove
theorem area_of_triangle_BCD : area_BCD 39 8 = 156 :=
by
  sorry

end area_of_triangle_BCD_l174_174711


namespace x_sq_y_sq_value_l174_174086

theorem x_sq_y_sq_value (x y : ℝ) 
  (h1 : x + y = 25) 
  (h2 : x^2 + y^2 = 169) 
  (h3 : x^3 * y^3 + y^3 * x^3 = 243) :
  x^2 * y^2 = 51984 := 
by 
  -- Proof to be added
  sorry

end x_sq_y_sq_value_l174_174086


namespace charlie_pennies_l174_174701

variable (a c : ℕ)

theorem charlie_pennies (h1 : c + 1 = 4 * (a - 1)) (h2 : c - 1 = 3 * (a + 1)) : c = 31 := 
by
  sorry

end charlie_pennies_l174_174701


namespace student_answers_all_correctly_l174_174745

/-- 
The exam tickets have 2 theoretical questions and 1 problem each. There are 28 tickets. 
A student is prepared for 50 theoretical questions out of 56 and 22 problems out of 28.
The probability that by drawing a ticket at random, and the student answers all questions 
correctly is 0.625.
-/
theorem student_answers_all_correctly :
  let total_theoretical := 56
  let total_problems := 28
  let prepared_theoretical := 50
  let prepared_problems := 22
  let p_correct_theoretical := (prepared_theoretical * (prepared_theoretical - 1)) / (total_theoretical * (total_theoretical - 1))
  let p_correct_problem := prepared_problems / total_problems
  let combined_probability := p_correct_theoretical * p_correct_problem
  combined_probability = 0.625 :=
  sorry

end student_answers_all_correctly_l174_174745


namespace both_shots_unsuccessful_both_shots_successful_exactly_one_shot_successful_at_least_one_shot_successful_at_most_one_shot_successful_l174_174855

variable (p q : Prop)

-- 1. Both shots were unsuccessful
theorem both_shots_unsuccessful : ¬p ∧ ¬q := sorry

-- 2. Both shots were successful
theorem both_shots_successful : p ∧ q := sorry

-- 3. Exactly one shot was successful
theorem exactly_one_shot_successful : (¬p ∧ q) ∨ (p ∧ ¬q) := sorry

-- 4. At least one shot was successful
theorem at_least_one_shot_successful : p ∨ q := sorry

-- 5. At most one shot was successful
theorem at_most_one_shot_successful : ¬(p ∧ q) := sorry

end both_shots_unsuccessful_both_shots_successful_exactly_one_shot_successful_at_least_one_shot_successful_at_most_one_shot_successful_l174_174855


namespace no_square_pair_l174_174735

/-- 
Given integers a, b, and c, where c > 0, if a(a + 4) = c^2 and (a + 2 + c)(a + 2 - c) = 4, 
then the numbers a(a + 4) and b(b + 4) cannot both be squares.
-/
theorem no_square_pair (a b c : ℤ) (hc_pos : c > 0) (ha_eq : a * (a + 4) = c^2) 
  (hfac_eq : (a + 2 + c) * (a + 2 - c) = 4) : ¬(∃ d e : ℤ, d^2 = a * (a + 4) ∧ e^2 = b * (b + 4)) :=
by sorry

end no_square_pair_l174_174735


namespace range_a_real_numbers_l174_174511

theorem range_a_real_numbers (x a : ℝ) : 
  (∀ x : ℝ, (x - a) * (1 - (x + a)) < 1) → (a ∈ Set.univ) :=
by
  sorry

end range_a_real_numbers_l174_174511


namespace binomial_coefficient_10_3_l174_174275

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l174_174275


namespace combined_girls_avg_l174_174650

variables (A a B b : ℕ) -- Number of boys and girls at Adams and Baker respectively.
variables (avgBoysAdams avgGirlsAdams avgAdams avgBoysBaker avgGirlsBaker avgBaker : ℚ)

-- Conditions
def avgAdamsBoys := 72
def avgAdamsGirls := 78
def avgAdamsCombined := 75
def avgBakerBoys := 84
def avgBakerGirls := 91
def avgBakerCombined := 85
def combinedAvgBoys := 80

-- Equations derived from the problem statement
def equations : Prop :=
  (72 * A + 78 * a) / (A + a) = 75 ∧
  (84 * B + 91 * b) / (B + b) = 85 ∧
  (72 * A + 84 * B) / (A + B) = 80

-- The goal is to show the combined average score of girls
def combinedAvgGirls := 85

theorem combined_girls_avg (h : equations A a B b):
  (78 * (6 * b / 7) + 91 * b) / ((6 * b / 7) + b) = 85 := by
  sorry

end combined_girls_avg_l174_174650


namespace multiply_exponents_l174_174992

theorem multiply_exponents (a : ℝ) : 2 * a^3 * 3 * a^2 = 6 * a^5 := by
  sorry

end multiply_exponents_l174_174992


namespace cube_weight_l174_174042

theorem cube_weight (l1 l2 V1 V2 k : ℝ) (h1: l2 = 2 * l1) (h2: V1 = l1^3) (h3: V2 = (2 * l1)^3) (h4: w2 = 48) (h5: V2 * k = w2) (h6: V1 * k = w1):
  w1 = 6 :=
by
  sorry

end cube_weight_l174_174042


namespace smallest_positive_n_l174_174770

theorem smallest_positive_n : ∃ (n : ℕ), n = 626 ∧ ∀ m : ℕ, m < 626 → ¬ (sqrt m - sqrt (m - 1) < 0.02) := by
  sorry

end smallest_positive_n_l174_174770


namespace goods_train_speed_l174_174834

def speed_of_goods_train (length_in_meters : ℕ) (time_in_seconds : ℕ) (speed_of_man_train_kmph : ℕ) : ℕ :=
  let length_in_km := length_in_meters / 1000
  let time_in_hours := time_in_seconds / 3600
  let relative_speed_kmph := (length_in_km * 3600) / time_in_hours
  relative_speed_kmph - speed_of_man_train_kmph

theorem goods_train_speed :
  speed_of_goods_train 280 9 50 = 62 := by
  sorry

end goods_train_speed_l174_174834


namespace binom_10_3_eq_120_l174_174313

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l174_174313


namespace marie_eggs_total_l174_174915

variable (x : ℕ) -- Number of eggs in each box

-- Conditions as definitions
def egg_weight := 10 -- weight of each egg in ounces
def total_boxes := 4 -- total number of boxes
def remaining_boxes := 3 -- boxes left after one is discarded
def remaining_weight := 90 -- total weight of remaining eggs in ounces

-- Proof statement
theorem marie_eggs_total : remaining_boxes * egg_weight * x = remaining_weight → total_boxes * x = 12 :=
by
  intros h
  sorry

end marie_eggs_total_l174_174915


namespace binom_10_3_eq_120_l174_174191

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l174_174191


namespace james_car_new_speed_l174_174094

-- Define the conditions and the statement to prove
variable (original_speed supercharge_increase weight_reduction : ℝ)
variable (original_speed_gt_zero : 0 < original_speed)

theorem james_car_new_speed :
  original_speed = 150 →
  supercharge_increase = 0.3 →
  weight_reduction = 10 →
  original_speed * (1 + supercharge_increase) + weight_reduction = 205 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  -- Calculate the speed after supercharging
  have supercharged_speed : ℝ := 150 * (1 + 0.3)
  calc
    150 * (1 + 0.3) + 10 = 195 + 10 : by norm_num
                       ... = 205 : by norm_num
  sorry

end james_car_new_speed_l174_174094


namespace sum_of_fifth_powers_cannot_conclude_sum_of_fourth_powers_l174_174965

-- Definition of conditions
variables {a b c d : ℝ} 

-- First proof statement
theorem sum_of_fifth_powers 
  (h1 : a + b = c + d) 
  (h2 : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 := 
sorry

-- Second proof statement
theorem cannot_conclude_sum_of_fourth_powers 
  (h1 : a + b = c + d) 
  (h2 : a^3 + b^3 = c^3 + d^3) : 
  ¬(a^4 + b^4 = c^4 + d^4) := 
sorry

end sum_of_fifth_powers_cannot_conclude_sum_of_fourth_powers_l174_174965


namespace zookeeper_feeding_ways_l174_174495

/-- We define the total number of ways the zookeeper can feed all the animals following the rules. -/
def feed_animal_ways : ℕ :=
  6 * 5^2 * 4^2 * 3^2 * 2^2 * 1^2

/-- Theorem statement: The number of ways to feed all the animals is 86400. -/
theorem zookeeper_feeding_ways : feed_animal_ways = 86400 :=
by
  sorry

end zookeeper_feeding_ways_l174_174495


namespace solve_inequality_range_of_m_l174_174038

noncomputable def f (x : ℝ) : ℝ := abs (x - 2)
noncomputable def g (x m : ℝ) : ℝ := - abs (x + 3) + m

theorem solve_inequality (x a : ℝ) :
  (f x + a - 1 > 0) ↔
  (a = 1 → x ≠ 2) ∧
  (a > 1 → true) ∧
  (a < 1 → x < a + 1 ∨ x > 3 - a) := by sorry

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, f x ≥ g x m) ↔ m < 5 := by sorry

end solve_inequality_range_of_m_l174_174038


namespace mul_powers_same_base_l174_174849

theorem mul_powers_same_base (a : ℝ) : a^3 * a^4 = a^7 := 
by 
  sorry

end mul_powers_same_base_l174_174849


namespace sale_second_month_l174_174638

def sale_first_month : ℝ := 5700
def sale_third_month : ℝ := 6855
def sale_fourth_month : ℝ := 3850
def sale_fifth_month : ℝ := 14045
def average_sale : ℝ := 7800

theorem sale_second_month : 
  ∃ x : ℝ, -- there exists a sale in the second month such that...
    (sale_first_month + x + sale_third_month + sale_fourth_month + sale_fifth_month) / 5 = average_sale
    ∧ x = 7550 := 
by
  sorry

end sale_second_month_l174_174638


namespace calculation_equals_106_25_l174_174054

noncomputable def calculation : ℝ := 2.5 * 8.5 * (5.2 - 0.2)

theorem calculation_equals_106_25 : calculation = 106.25 := 
by
  sorry

end calculation_equals_106_25_l174_174054


namespace remainder_of_8_pow_6_plus_1_mod_7_l174_174148

theorem remainder_of_8_pow_6_plus_1_mod_7 :
  (8^6 + 1) % 7 = 2 := by
  sorry

end remainder_of_8_pow_6_plus_1_mod_7_l174_174148


namespace participants_l174_174916

variable {A B C D : Prop}

theorem participants (h1 : A → B) (h2 : ¬C → ¬B) (h3 : C → ¬D) :
  (¬A ∧ C ∧ B ∧ ¬D) ∨ ¬B :=
by
  -- The proof is not provided
  sorry

end participants_l174_174916


namespace value_y1_y2_l174_174734

variable {x1 x2 y1 y2 : ℝ}

-- Points on the inverse proportion function
def on_graph (x y : ℝ) : Prop := y = -3 / x

-- Given conditions
theorem value_y1_y2 (hx1 : on_graph x1 y1) (hx2 : on_graph x2 y2) (hxy : x1 * x2 = 2) : y1 * y2 = 9 / 2 :=
by
  sorry

end value_y1_y2_l174_174734


namespace inequality_ge_one_l174_174808

theorem inequality_ge_one {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_ge_one_l174_174808


namespace smallest_n_l174_174769

theorem smallest_n (n : ℕ) (h : ↑n > 0 ∧ (Real.sqrt (↑n) - Real.sqrt (↑n - 1)) < 0.02) : n = 626 := 
by
  sorry

end smallest_n_l174_174769


namespace peanuts_in_box_l174_174703

variable (original_peanuts : Nat)
variable (additional_peanuts : Nat)

theorem peanuts_in_box (h1 : original_peanuts = 4) (h2 : additional_peanuts = 4) :
  original_peanuts + additional_peanuts = 8 := 
by
  sorry

end peanuts_in_box_l174_174703


namespace solve_system_of_equations_l174_174767

theorem solve_system_of_equations (x y : ℚ)
  (h1 : 15 * x + 24 * y = 18)
  (h2 : 24 * x + 15 * y = 63) :
  x = 46 / 13 ∧ y = -19 / 13 := 
sorry

end solve_system_of_equations_l174_174767


namespace arabella_dance_steps_l174_174846

theorem arabella_dance_steps :
  exists T1 T2 T3 : ℕ,
    T1 = 30 ∧
    T3 = T1 + T2 ∧
    T1 + T2 + T3 = 90 ∧
    (T2 / T1 : ℚ) = 1 / 2 :=
by
  sorry

end arabella_dance_steps_l174_174846


namespace larger_cookie_sugar_l174_174160

theorem larger_cookie_sugar :
  let initial_cookies := 40
  let initial_sugar_per_cookie := 1 / 8
  let total_sugar := initial_cookies * initial_sugar_per_cookie
  let larger_cookies := 25
  let sugar_per_larger_cookie := total_sugar / larger_cookies
  sugar_per_larger_cookie = 1 / 5 := by
sorry

end larger_cookie_sugar_l174_174160


namespace speed_of_stream_l174_174944

variable (v : ℝ)

theorem speed_of_stream (h : (64 / (24 + v)) = (32 / (24 - v))) : v = 8 := 
by
  sorry

end speed_of_stream_l174_174944


namespace polygon_sides_l174_174951

theorem polygon_sides (n : ℕ) (h : n - 1 = 2022) : n = 2023 :=
by
  sorry

end polygon_sides_l174_174951


namespace proposition_statementC_l174_174624

-- Definitions of each statement
def statementA := "Draw a parallel line to line AB"
def statementB := "Take a point C on segment AB"
def statementC := "The complement of equal angles are equal"
def statementD := "Is the perpendicular segment the shortest?"

-- Proving that among the statements A, B, C, and D, statement C is the proposition
theorem proposition_statementC : 
  (statementC = "The complement of equal angles are equal") :=
by
  -- We assume it directly from the equivalence given in the problem statement
  sorry

end proposition_statementC_l174_174624


namespace combination_10_3_eq_120_l174_174243

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end combination_10_3_eq_120_l174_174243


namespace find_m_l174_174693

-- Define the set A
def A (m : ℝ) : Set ℝ := {0, m, m^2 - 3 * m + 2}

-- Main theorem statement
theorem find_m (m : ℝ) (h : 2 ∈ A m) : m = 3 := by
  sorry

end find_m_l174_174693


namespace students_only_in_math_l174_174171

-- Define the sets and their cardinalities according to the problem conditions
def total_students : ℕ := 120
def math_students : ℕ := 85
def foreign_language_students : ℕ := 65
def sport_students : ℕ := 50
def all_three_classes : ℕ := 10

-- Define the Lean theorem to prove the number of students taking only a math class
theorem students_only_in_math (total : ℕ) (M F S : ℕ) (MFS : ℕ)
  (H_total : total = 120)
  (H_M : M = 85)
  (H_F : F = 65)
  (H_S : S = 50)
  (H_MFS : MFS = 10) :
  (M - (MFS + MFS - MFS) = 35) :=
sorry

end students_only_in_math_l174_174171


namespace inequality_abc_l174_174926

variable (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
variable (cond : a + b + c = (1/a) + (1/b) + (1/c))

theorem inequality_abc : a + b + c ≥ 3 / (a * b * c) :=
by
  sorry

end inequality_abc_l174_174926


namespace flower_beds_and_circular_path_fraction_l174_174836

noncomputable def occupied_fraction 
  (yard_length : ℕ)
  (yard_width : ℕ)
  (side1 : ℕ)
  (side2 : ℕ)
  (triangle_leg : ℕ)
  (circle_radius : ℕ) : ℝ :=
  let flower_bed_area := 2 * (1 / 2 : ℝ) * triangle_leg^2
  let circular_path_area := Real.pi * circle_radius ^ 2
  let occupied_area := flower_bed_area + circular_path_area
  occupied_area / (yard_length * yard_width)

theorem flower_beds_and_circular_path_fraction
  (yard_length : ℕ)
  (yard_width : ℕ)
  (side1 : ℕ)
  (side2 : ℕ)
  (triangle_leg : ℕ)
  (circle_radius : ℕ)
  (h1 : side1 = 20)
  (h2 : side2 = 30)
  (h3 : triangle_leg = (side2 - side1) / 2)
  (h4 : yard_length = 30)
  (h5 : yard_width = 5)
  (h6 : circle_radius = 2) :
  occupied_fraction yard_length yard_width side1 side2 triangle_leg circle_radius = (25 + 4 * Real.pi) / 150 :=
by sorry

end flower_beds_and_circular_path_fraction_l174_174836


namespace perpendicular_lines_slope_eq_l174_174709

theorem perpendicular_lines_slope_eq (m : ℝ) :
  (∀ x y : ℝ, x - 2 * y + 5 = 0 → 
               2 * x + m * y - 6 = 0 → 
               (1 / 2) * (-2 / m) = -1) →
  m = 1 := 
by sorry

end perpendicular_lines_slope_eq_l174_174709


namespace jack_total_cost_l174_174413

def cost_of_tires (n : ℕ) (price_per_tire : ℕ) : ℕ := n * price_per_tire
def cost_of_window (price_per_window : ℕ) : ℕ := price_per_window

theorem jack_total_cost :
  cost_of_tires 3 250 + cost_of_window 700 = 1450 :=
by
  sorry

end jack_total_cost_l174_174413


namespace diane_owes_money_l174_174062

theorem diane_owes_money (initial_amount winnings total_losses : ℤ) (h_initial : initial_amount = 100) (h_winnings : winnings = 65) (h_losses : total_losses = 215) : 
  initial_amount + winnings - total_losses = -50 := by
  sorry

end diane_owes_money_l174_174062


namespace polygon_diagonals_regions_l174_174713

theorem polygon_diagonals_regions (n : ℕ) (hn : n ≥ 3) :
  let D := n * (n - 3) / 2
  let P := n * (n - 1) * (n - 2) * (n - 3) / 24
  let R := D + P + 1
  R = n * (n - 1) * (n - 2) * (n - 3) / 24 + n * (n - 3) / 2 + 1 :=
by
  sorry

end polygon_diagonals_regions_l174_174713


namespace remainder_division_l174_174640

theorem remainder_division (k : ℤ) (N : ℤ) (h : N = 133 * k + 16) : N % 50 = 49 := by
  sorry

end remainder_division_l174_174640


namespace math_problem_l174_174870

-- Definitions of conditions
def cond1 (x a y b z c : ℝ) : Prop := x / a + y / b + z / c = 1
def cond2 (x a y b z c : ℝ) : Prop := a / x + b / y + c / z = 0

-- Theorem statement
theorem math_problem (x a y b z c : ℝ)
  (h1 : cond1 x a y b z c) (h2 : cond2 x a y b z c) :
  (x^2 / a^2) + (y^2 / b^2) + (z^2 / c^2) = 1 :=
by
  sorry

end math_problem_l174_174870


namespace inequality_proof_l174_174789

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 := 
by 
  sorry

end inequality_proof_l174_174789


namespace combined_jail_time_in_weeks_l174_174758

-- Definitions based on conditions
def days_of_protest : ℕ := 30
def number_of_cities : ℕ := 21
def daily_arrests_per_city : ℕ := 10
def days_in_jail_pre_trial : ℕ := 4
def sentence_weeks : ℕ := 2
def jail_fraction_of_sentence : ℕ := 1 / 2

-- Calculate the combined weeks of jail time
theorem combined_jail_time_in_weeks : 
  (days_of_protest * daily_arrests_per_city * number_of_cities) * 
  (days_in_jail_pre_trial + (sentence_weeks * 7 * jail_fraction_of_sentence)) / 
  7 = 9900 := 
by sorry

end combined_jail_time_in_weeks_l174_174758


namespace ratio_john_amount_l174_174493

theorem ratio_john_amount (total_amount : ℕ) (john_received : ℕ) (h_total : total_amount = 4800) (h_john : john_received = 1600) :
  (john_received : ℚ) / total_amount = 1 / 3 :=
by
  rw [h_total, h_john]
  norm_num
  exact sorry

end ratio_john_amount_l174_174493


namespace number_of_books_in_shipment_l174_174157

theorem number_of_books_in_shipment
  (T : ℕ)                   -- The total number of books
  (displayed_ratio : ℚ)     -- Fraction of books displayed
  (remaining_books : ℕ)     -- Number of books in the storeroom
  (h1 : displayed_ratio = 0.3)
  (h2 : remaining_books = 210)
  (h3 : (1 - displayed_ratio) * T = remaining_books) :
  T = 300 := 
by
  -- Add your proof here
  sorry

end number_of_books_in_shipment_l174_174157


namespace combination_10_3_eq_120_l174_174356

open Nat

theorem combination_10_3_eq_120 : (10.choose 3) = 120 := 
by
  sorry

end combination_10_3_eq_120_l174_174356


namespace inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8_l174_174533

theorem inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
    1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 := 
sorry

end inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8_l174_174533


namespace interest_rate_difference_l174_174647

theorem interest_rate_difference (P T : ℝ) (R1 R2 : ℝ) (I_diff : ℝ) (hP : P = 2100) 
  (hT : T = 3) (hI : I_diff = 63) :
  R2 - R1 = 0.01 :=
by
  sorry

end interest_rate_difference_l174_174647


namespace binomial_10_3_eq_120_l174_174200

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l174_174200


namespace gas_station_total_boxes_l174_174163

theorem gas_station_total_boxes
  (chocolate_boxes : ℕ)
  (sugar_boxes : ℕ)
  (gum_boxes : ℕ)
  (licorice_boxes : ℕ)
  (sour_boxes : ℕ)
  (h_chocolate : chocolate_boxes = 3)
  (h_sugar : sugar_boxes = 5)
  (h_gum : gum_boxes = 2)
  (h_licorice : licorice_boxes = 4)
  (h_sour : sour_boxes = 7) :
  chocolate_boxes + sugar_boxes + gum_boxes + licorice_boxes + sour_boxes = 21 := by
  sorry

end gas_station_total_boxes_l174_174163


namespace rectangular_to_polar_l174_174996

variable (x y : ℝ)

def is_polar_coordinate (x y r theta : ℝ) : Prop :=
  r = Real.sqrt (x * x + y * y) ∧ tan theta = y / x

theorem rectangular_to_polar :
  is_polar_coordinate 6 (2 * Real.sqrt 3) (4 * Real.sqrt 3) (Real.pi / 6) :=
by
  dsimp [is_polar_coordinate]
  sorry

end rectangular_to_polar_l174_174996


namespace part_a_part_b_l174_174969

-- Part (a)
theorem part_a {a b c d : ℝ} (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) :
  a^5 + b^5 = c^5 + d^5 :=
sorry

-- Part (b)
theorem part_b {a b c d : ℝ} (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) :
  ¬(a^4 + b^4 = c^4 + d^4) :=
counter_example

end part_a_part_b_l174_174969


namespace paint_cost_decrease_l174_174502

variables (C P : ℝ)
variable (cost_decrease_canvas : ℝ := 0.40)
variable (total_cost_decrease : ℝ := 0.56)
variable (paint_to_canvas_ratio : ℝ := 4)

theorem paint_cost_decrease (x : ℝ) : 
  P = 4 * C ∧ 
  P * (1 - x) + C * (1 - cost_decrease_canvas) = (1 - total_cost_decrease) * (P + C) → 
  x = 0.60 :=
by
  intro h
  sorry

end paint_cost_decrease_l174_174502


namespace corey_candies_l174_174600

-- Definitions based on conditions
variable (T C : ℕ)
variable (totalCandies : T + C = 66)
variable (tapangaExtra : T = C + 8)

-- Theorem to prove Corey has 29 candies
theorem corey_candies : C = 29 :=
by
  sorry

end corey_candies_l174_174600


namespace find_minimal_product_l174_174500

theorem find_minimal_product : ∃ x y : ℤ, (20 * x + 19 * y = 2019) ∧ (x * y = 2623) ∧ (∀ z w : ℤ, (20 * z + 19 * w = 2019) → |x - y| ≤ |z - w|) :=
by
  -- definitions and theorems to prove the problem would be placed here
  sorry

end find_minimal_product_l174_174500


namespace stacked_cubes_surface_area_is_945_l174_174370

def volumes : List ℕ := [512, 343, 216, 125, 64, 27, 8, 1]

def side_length (v : ℕ) : ℕ := v^(1/3)

def num_visible_faces (i : ℕ) : ℕ :=
  if i == 0 then 5 else 3 -- Bottom cube has 5 faces visible, others have 3 due to rotation

def surface_area (s : ℕ) (faces : ℕ) : ℕ :=
  faces * s^2

def total_surface_area (volumes : List ℕ) : ℕ :=
  (volumes.zipWith surface_area (volumes.enum.map (λ (i, v) => num_visible_faces i))).sum

theorem stacked_cubes_surface_area_is_945 :
  total_surface_area volumes = 945 := 
by 
  sorry

end stacked_cubes_surface_area_is_945_l174_174370


namespace no_valid_road_network_l174_174653

theorem no_valid_road_network
  (k_A k_B k_C : ℕ)
  (h_kA : k_A ≥ 2)
  (h_kB : k_B ≥ 2)
  (h_kC : k_C ≥ 2) :
  ¬ ∃ (t : ℕ) (d : ℕ → ℕ), t ≥ 7 ∧ 
    (∀ i j, i ≠ j → d i ≠ d j) ∧
    (∀ i, i < 4 * (k_A + k_B + k_C) + 4 → d i = i + 1) :=
sorry

end no_valid_road_network_l174_174653


namespace cards_left_l174_174174

theorem cards_left (bask_boxes : ℕ) (bask_cards_per_box : ℕ) (base_boxes : ℕ) (base_cards_per_box : ℕ) (cards_given : ℕ) :
  bask_boxes = 4 → bask_cards_per_box = 10 → base_boxes = 5 → base_cards_per_box = 8 → cards_given = 58 →
  (bask_boxes * bask_cards_per_box + base_boxes * base_cards_per_box - cards_given) = 22 :=
begin
  sorry, -- proof is skipped as per the instructions
end

end cards_left_l174_174174


namespace kate_needs_more_money_l174_174099

theorem kate_needs_more_money
  (pen_price : ℝ)
  (notebook_price : ℝ)
  (artset_price : ℝ)
  (kate_pen_money_fraction : ℝ)
  (notebook_discount : ℝ)
  (artset_discount : ℝ)
  (kate_artset_money : ℝ) :
  pen_price = 30 →
  notebook_price = 20 →
  artset_price = 50 →
  kate_pen_money_fraction = 1/3 →
  notebook_discount = 0.15 →
  artset_discount = 0.4 →
  kate_artset_money = 10 →
  (pen_price - kate_pen_money_fraction * pen_price) +
  (notebook_price * (1 - notebook_discount)) +
  (artset_price * (1 - artset_discount) - kate_artset_money) = 57 :=
by
  sorry

end kate_needs_more_money_l174_174099


namespace Freddy_is_18_l174_174536

-- Definitions based on the conditions
def Job_age : Nat := 5
def Stephanie_age : Nat := 4 * Job_age
def Freddy_age : Nat := Stephanie_age - 2

-- Statement to prove
theorem Freddy_is_18 : Freddy_age = 18 := by
  sorry

end Freddy_is_18_l174_174536


namespace total_problems_l174_174048

theorem total_problems (C W : ℕ) (h1 : 3 * C + 5 * W = 110) (h2 : C = 20) : C + W = 30 :=
by {
  sorry
}

end total_problems_l174_174048


namespace max_gcd_of_sequence_l174_174678

/-- Define the sequence as a function. -/
def a (n : ℕ) : ℕ := 100 + n^2

/-- Define the greatest common divisor of the sequence terms. -/
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

/-- State the theorem of the maximum value of d. -/
theorem max_gcd_of_sequence : ∃ n : ℕ, d n = 401 := sorry

end max_gcd_of_sequence_l174_174678


namespace system_solution_fraction_l174_174863

theorem system_solution_fraction (x y z : ℝ) (h1 : x + (-95/9) * y + 4 * z = 0)
  (h2 : 4 * x + (-95/9) * y - 3 * z = 0) (h3 : 3 * x + 5 * y - 4 * z = 0) (hx_ne_zero : x ≠ 0) 
  (hy_ne_zero : y ≠ 0) (hz_ne_zero : z ≠ 0) : 
  (x * z) / (y ^ 2) = 20 :=
sorry

end system_solution_fraction_l174_174863


namespace pond_length_l174_174090

theorem pond_length (V W D L : ℝ) (hV : V = 1600) (hW : W = 10) (hD : D = 8) :
  L = 20 ↔ V = L * W * D :=
by
  sorry

end pond_length_l174_174090


namespace binom_10_3_eq_120_l174_174218

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end binom_10_3_eq_120_l174_174218


namespace boxes_in_carton_of_pencils_l174_174644

def cost_per_box_pencil : ℕ := 2
def cost_per_box_marker : ℕ := 4
def boxes_per_carton_marker : ℕ := 5
def cartons_of_pencils : ℕ := 20
def cartons_of_markers : ℕ := 10
def total_spent : ℕ := 600

theorem boxes_in_carton_of_pencils : ∃ x : ℕ, 20 * (2 * x) + 10 * (5 * 4) = 600 :=
by
  sorry

end boxes_in_carton_of_pencils_l174_174644


namespace rick_division_steps_l174_174591

theorem rick_division_steps (initial_books : ℕ) (final_books : ℕ) 
  (h_initial : initial_books = 400) (h_final : final_books = 25) : 
  (∀ n : ℕ, (initial_books / (2^n) = final_books) → n = 4) :=
by
  sorry

end rick_division_steps_l174_174591


namespace vijay_work_alone_in_24_days_l174_174167

theorem vijay_work_alone_in_24_days (ajay_rate vijay_rate combined_rate : ℝ) 
  (h1 : ajay_rate = 1 / 8) 
  (h2 : combined_rate = 1 / 6) 
  (h3 : ajay_rate + vijay_rate = combined_rate) : 
  vijay_rate = 1 / 24 := 
sorry

end vijay_work_alone_in_24_days_l174_174167


namespace combination_10_3_eq_120_l174_174358

open Nat

theorem combination_10_3_eq_120 : (10.choose 3) = 120 := 
by
  sorry

end combination_10_3_eq_120_l174_174358


namespace number_of_ways_2020_l174_174719

-- We are defining b_i explicitly restricted by the conditions in the problem.
def b (i : ℕ) : ℕ :=
  sorry

-- Given conditions
axiom h_bounds : ∀ i, 0 ≤ b i ∧ b i ≤ 99
axiom h_indices : ∀ (i : ℕ), i < 4

-- Main theorem statement
theorem number_of_ways_2020 (M : ℕ) 
  (h : 2020 = b 3 * 1000 + b 2 * 100 + b 1 * 10 + b 0) 
  (htotal : M = 203) : 
  M = 203 :=
  by 
    sorry

end number_of_ways_2020_l174_174719


namespace binom_10_3_eq_120_l174_174310

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l174_174310


namespace total_interest_is_68_l174_174487

-- Definitions of the initial conditions
def amount_2_percent : ℝ := 600
def amount_4_percent : ℝ := amount_2_percent + 800
def interest_rate_2_percent : ℝ := 0.02
def interest_rate_4_percent : ℝ := 0.04
def invested_total_1 : ℝ := amount_2_percent
def invested_total_2 : ℝ := amount_4_percent

-- The total interest calculation
def interest_2_percent : ℝ := invested_total_1 * interest_rate_2_percent
def interest_4_percent : ℝ := invested_total_2 * interest_rate_4_percent

-- Claim: The total interest earned is $68
theorem total_interest_is_68 : interest_2_percent + interest_4_percent = 68 := by
  sorry

end total_interest_is_68_l174_174487


namespace inequality_proof_l174_174800

theorem inequality_proof (x y z : ℝ) : 
  ( (x^3) / (x^3 + 2 * (y^2) * z) + 
    (y^3) / (y^3 + 2 * (z^2) * x) + 
    (z^3) / (z^3 + 2 * (x^2) * y) ) ≥ 1 := 
by 
  sorry

end inequality_proof_l174_174800


namespace binom_10_3_l174_174228

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end binom_10_3_l174_174228


namespace cube_volume_l174_174022

theorem cube_volume (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
sorry

end cube_volume_l174_174022


namespace sum_of_fifth_powers_l174_174959

theorem sum_of_fifth_powers (a b c d : ℝ) (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 := sorry

end sum_of_fifth_powers_l174_174959


namespace proof_problem_l174_174902

def h (x : ℝ) : ℝ := 2 * x + 4
def k (x : ℝ) : ℝ := 4 * x + 6

theorem proof_problem : h (k 3) - k (h 3) = -6 :=
by
  sorry

end proof_problem_l174_174902


namespace tray_contains_correct_number_of_pieces_l174_174975

-- Define the dimensions of the tray
def tray_width : ℕ := 24
def tray_length : ℕ := 20
def tray_area : ℕ := tray_width * tray_length

-- Define the dimensions of each brownie piece
def piece_width : ℕ := 3
def piece_length : ℕ := 4
def piece_area : ℕ := piece_width * piece_length

-- Define the goal: the number of pieces of brownies that the tray contains
def num_pieces : ℕ := tray_area / piece_area

-- The statement to prove
theorem tray_contains_correct_number_of_pieces :
  num_pieces = 40 :=
by
  sorry

end tray_contains_correct_number_of_pieces_l174_174975


namespace find_room_width_l174_174608

theorem find_room_width
  (length : ℝ)
  (cost_per_sqm : ℝ)
  (total_cost : ℝ)
  (h_length : length = 10)
  (h_cost_per_sqm : cost_per_sqm = 900)
  (h_total_cost : total_cost = 42750) :
  ∃ width : ℝ, width = 4.75 :=
by
  sorry

end find_room_width_l174_174608


namespace perimeter_of_triangle_XYZ_l174_174449

/-- 
  Given the inscribed circle of triangle XYZ is tangent to XY at P,
  its radius is 15, XP = 30, and PY = 36, then the perimeter of 
  triangle XYZ is 83.4.
-/
theorem perimeter_of_triangle_XYZ :
  ∀ (XYZ : Type) (P : XYZ) (radius : ℝ) (XP PY perimeter : ℝ),
    radius = 15 → 
    XP = 30 → 
    PY = 36 →
    perimeter = 83.4 :=
by 
  intros XYZ P radius XP PY perimeter h_radius h_XP h_PY
  sorry

end perimeter_of_triangle_XYZ_l174_174449


namespace transport_tax_correct_l174_174506

def engine_power : ℕ := 250
def tax_rate : ℕ := 75
def months_owned : ℕ := 2
def months_in_year : ℕ := 12
def annual_tax : ℕ := engine_power * tax_rate
def adjusted_tax : ℕ := (annual_tax * months_owned) / months_in_year

theorem transport_tax_correct :
  adjusted_tax = 3125 :=
by
  sorry

end transport_tax_correct_l174_174506


namespace part_a_part_b_l174_174962

open Real

theorem part_a (a b c d : ℝ) (h₁ : a + b = c + d) (h₂ : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 :=
sorry

theorem part_b (a b c d : ℝ) (h₁ : a + b = c + d) (h₂ : a^3 + b^3 = c^3 + d^3) : ¬ (a^4 + b^4 = c^4 + d^4) :=
begin
  intro h,
  have : ¬ (1 + 1 = 16 + 16),
  { norm_num, },
  exact this h,
end

end part_a_part_b_l174_174962


namespace part_a_part_b_l174_174961

open Real

theorem part_a (a b c d : ℝ) (h₁ : a + b = c + d) (h₂ : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 :=
sorry

theorem part_b (a b c d : ℝ) (h₁ : a + b = c + d) (h₂ : a^3 + b^3 = c^3 + d^3) : ¬ (a^4 + b^4 = c^4 + d^4) :=
begin
  intro h,
  have : ¬ (1 + 1 = 16 + 16),
  { norm_num, },
  exact this h,
end

end part_a_part_b_l174_174961


namespace parallel_lines_count_l174_174558

theorem parallel_lines_count (n : ℕ) (h : 7 * (n - 1) = 588) : n = 85 :=
sorry

end parallel_lines_count_l174_174558


namespace binom_10_3_l174_174230

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end binom_10_3_l174_174230


namespace car_new_speed_l174_174093

theorem car_new_speed (original_speed : ℝ) (supercharge_percent : ℝ) (weight_cut_speed_increase : ℝ) :
  original_speed = 150 → supercharge_percent = 0.30 → weight_cut_speed_increase = 10 → 
  original_speed * (1 + supercharge_percent) + weight_cut_speed_increase = 205 :=
by
  intros h_orig h_supercharge h_weight
  rw [h_orig, h_supercharge]
  sorry

end car_new_speed_l174_174093


namespace binom_10_3_l174_174329

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end binom_10_3_l174_174329


namespace number_of_valid_sequences_l174_174554

-- Definitions for conditions
def digit := Fin 10 -- Digit can be any number from 0 to 9
def is_odd (n : digit) : Prop := n.val % 2 = 1
def is_even (n : digit) : Prop := n.val % 2 = 0

def valid_sequence (s : Fin 8 → digit) : Prop :=
  ∀ i : Fin 7, (is_odd (s i) ↔ is_even (s (i+1)))

-- Theorem statement
theorem number_of_valid_sequences : 
  ∃ n, n = 781250 ∧ 
    ∃ s : (Fin 8 → digit), valid_sequence s :=
sorry -- Proof is not required

end number_of_valid_sequences_l174_174554


namespace ratio_five_to_one_l174_174632

theorem ratio_five_to_one (x : ℕ) (h : 5 / 1 = x / 9) : x = 45 :=
  sorry

end ratio_five_to_one_l174_174632


namespace arc_length_given_curve_l174_174177

noncomputable def arc_length_polar (ρ : ℝ → ℝ) (φ₀ φ₁ : ℝ) : ℝ :=
  ∫ φ in φ₀..φ₁, sqrt ((ρ φ)^2 + (deriv ρ φ)^2)

theorem arc_length_given_curve :
  arc_length_polar (λ φ, 5 * (1 - cos φ)) (-π / 3) 0 = 20 * (1 - sqrt 3 / 2) := sorry

end arc_length_given_curve_l174_174177


namespace women_population_percentage_l174_174088

theorem women_population_percentage (W M : ℕ) (h : M = 2 * W) : (W : ℚ) / (M : ℚ) = (50 : ℚ) / 100 :=
by
  -- Proof omitted
  sorry

end women_population_percentage_l174_174088


namespace binomial_10_3_eq_120_l174_174205

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l174_174205


namespace difference_of_cats_l174_174732

-- Definitions based on given conditions
def number_of_cats_sheridan : ℕ := 11
def number_of_cats_garrett : ℕ := 24

-- Theorem statement (proof problem) based on the question and correct answer
theorem difference_of_cats : (number_of_cats_garrett - number_of_cats_sheridan) = 13 := by
  sorry

end difference_of_cats_l174_174732


namespace roof_shingles_area_l174_174747

-- Definitions based on given conditions
def base_main_roof : ℝ := 20.5
def height_main_roof : ℝ := 25
def upper_base_porch : ℝ := 2.5
def lower_base_porch : ℝ := 4.5
def height_porch : ℝ := 3
def num_gables_main_roof : ℕ := 2
def num_trapezoids_porch : ℕ := 4

-- Proof problem statement
theorem roof_shingles_area : 
  2 * (1 / 2 * base_main_roof * height_main_roof) +
  4 * (1 / 2 * (upper_base_porch + lower_base_porch) * height_porch) = 554.5 :=
by sorry

end roof_shingles_area_l174_174747


namespace binom_10_3_eq_120_l174_174219

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end binom_10_3_eq_120_l174_174219


namespace total_musicians_count_l174_174460

-- Define the given conditions
def orchestra_males := 11
def orchestra_females := 12
def choir_males := 12
def choir_females := 17

-- Total number of musicians in the orchestra
def orchestra_musicians := orchestra_males + orchestra_females

-- Total number of musicians in the band
def band_musicians := 2 * orchestra_musicians

-- Total number of musicians in the choir
def choir_musicians := choir_males + choir_females

-- Total number of musicians in the orchestra, band, and choir
def total_musicians := orchestra_musicians + band_musicians + choir_musicians

-- The theorem to prove
theorem total_musicians_count : total_musicians = 98 :=
by
  -- Lean proof part goes here.
  sorry

end total_musicians_count_l174_174460


namespace total_calories_box_l174_174829

-- Definitions from the conditions
def bags := 6
def cookies_per_bag := 25
def calories_per_cookie := 18

-- Given the conditions, prove the total calories equals 2700
theorem total_calories_box : bags * cookies_per_bag * calories_per_cookie = 2700 := by
  sorry

end total_calories_box_l174_174829


namespace original_number_count_l174_174594

theorem original_number_count (k S : ℕ) (M : ℚ)
  (hk : k > 0)
  (hM : M = S / k)
  (h_add15 : (S + 15) / (k + 1) = M + 2)
  (h_add1 : (S + 16) / (k + 2) = M + 1) :
  k = 6 :=
by
  -- Proof will go here
  sorry

end original_number_count_l174_174594


namespace inequality_proof_l174_174794

theorem inequality_proof (x y z : ℝ) : 
  ( (x^3) / (x^3 + 2 * (y^2) * z) + 
    (y^3) / (y^3 + 2 * (z^2) * x) + 
    (z^3) / (z^3 + 2 * (x^2) * y) ) ≥ 1 := 
by 
  sorry

end inequality_proof_l174_174794


namespace min_value_of_expression_l174_174673

theorem min_value_of_expression (x y z : ℝ) : ∃ a : ℝ, (∀ x y z : ℝ, x^2 + x * y + y^2 + y * z + z^2 ≥ a) ∧ (a = 0) :=
sorry

end min_value_of_expression_l174_174673


namespace value_of_b_l174_174762

theorem value_of_b (a b : ℕ) (r : ℝ) (h₁ : a = 2020) (h₂ : r = a / b) (h₃ : r = 0.5) : b = 4040 := 
by
  -- Hint: The proof takes steps to transform the conditions using basic algebraic manipulations.
  sorry

end value_of_b_l174_174762


namespace tom_books_l174_174142

theorem tom_books (books_may books_june books_july : ℕ) (h_may : books_may = 2) (h_june : books_june = 6) (h_july : books_july = 10) : 
books_may + books_june + books_july = 18 := by
sorry

end tom_books_l174_174142


namespace part1_part2_l174_174689

def f (x a : ℝ) : ℝ := |x + a - 1| + |x - 2 * a|

-- Define the first part of the problem
theorem part1 (a : ℝ) (h : f 1 a < 3) : -2/3 < a ∧ a < 4/3 :=
sorry

-- Define the second part of the problem
theorem part2 (a x : ℝ) (h1 : a ≥ 1) : f x a ≥ 2 :=
sorry

end part1_part2_l174_174689


namespace systematic_sampling_draw_l174_174145

theorem systematic_sampling_draw
  (x : ℕ) (h1 : 1 ≤ x ∧ x ≤ 8)
  (h2 : 160 ≥ 8 * 20)
  (h3 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 20 → 
    160 ≥ ((k - 1) * 8 + 1 + 7))
  (h4 : ∀ y : ℕ, y = 1 + (15 * 8) → y = 126)
: x = 6 := 
sorry

end systematic_sampling_draw_l174_174145


namespace part_a_part_b_l174_174960

open Real

theorem part_a (a b c d : ℝ) (h₁ : a + b = c + d) (h₂ : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 :=
sorry

theorem part_b (a b c d : ℝ) (h₁ : a + b = c + d) (h₂ : a^3 + b^3 = c^3 + d^3) : ¬ (a^4 + b^4 = c^4 + d^4) :=
begin
  intro h,
  have : ¬ (1 + 1 = 16 + 16),
  { norm_num, },
  exact this h,
end

end part_a_part_b_l174_174960


namespace div_eq_implies_eq_l174_174031

theorem div_eq_implies_eq (a b : ℕ) (h : (4 * a * b - 1) ∣ (4 * a^2 - 1)^2) : a = b :=
sorry

end div_eq_implies_eq_l174_174031


namespace isabella_hair_length_l174_174901

-- Define conditions: original length and doubled length
variable (original_length : ℕ)
variable (doubled_length : ℕ := 36)

-- Theorem: Prove that if the original length doubled equals 36, then the original length is 18.
theorem isabella_hair_length (h : 2 * original_length = doubled_length) : original_length = 18 := by
  sorry

end isabella_hair_length_l174_174901


namespace solve_quadratic_l174_174124

theorem solve_quadratic (x : ℚ) (h_pos : x > 0) (h_eq : 3 * x^2 + 8 * x - 35 = 0) : 
    x = 7/3 :=
by
    sorry

end solve_quadratic_l174_174124


namespace arithmetic_sequence_sufficient_not_necessary_l174_174634

variables {a b c d : ℤ}

-- Proving sufficiency: If a, b, c, d form an arithmetic sequence, then a + d = b + c.
def arithmetic_sequence (a b c d : ℤ) : Prop := 
  a + d = 2*b ∧ b + c = 2*a

theorem arithmetic_sequence_sufficient_not_necessary (h : arithmetic_sequence a b c d) : a + d = b + c ∧ ∃ (x y z w : ℤ), x + w = y + z ∧ ¬ arithmetic_sequence x y z w :=
by {
  sorry
}

end arithmetic_sequence_sufficient_not_necessary_l174_174634


namespace ratio_swordfish_to_pufferfish_l174_174089

theorem ratio_swordfish_to_pufferfish (P S : ℕ) (n : ℕ) 
  (hP : P = 15)
  (hTotal : S + P = 90)
  (hRelation : S = n * P) : 
  (S : ℚ) / (P : ℚ) = 5 := 
by 
  sorry

end ratio_swordfish_to_pufferfish_l174_174089


namespace combined_jail_time_in_weeks_l174_174759

-- Definitions based on conditions
def days_of_protest : ℕ := 30
def number_of_cities : ℕ := 21
def daily_arrests_per_city : ℕ := 10
def days_in_jail_pre_trial : ℕ := 4
def sentence_weeks : ℕ := 2
def jail_fraction_of_sentence : ℕ := 1 / 2

-- Calculate the combined weeks of jail time
theorem combined_jail_time_in_weeks : 
  (days_of_protest * daily_arrests_per_city * number_of_cities) * 
  (days_in_jail_pre_trial + (sentence_weeks * 7 * jail_fraction_of_sentence)) / 
  7 = 9900 := 
by sorry

end combined_jail_time_in_weeks_l174_174759


namespace roots_of_polynomial_l174_174517

theorem roots_of_polynomial :
  roots (λ x : ℝ, x^3 - 6 * x^2 + 11 * x - 6) = {1, 2, 3} := 
sorry

end roots_of_polynomial_l174_174517


namespace factor_quadratic_expression_l174_174515

theorem factor_quadratic_expression (x y : ℝ) :
  5 * x^2 + 6 * x * y - 8 * y^2 = (x + 2 * y) * (5 * x - 4 * y) :=
by
  sorry

end factor_quadratic_expression_l174_174515


namespace gcd_f_x_l174_174075

def f (x : ℤ) : ℤ := (5 * x + 3) * (11 * x + 2) * (14 * x + 7) * (3 * x + 8)

theorem gcd_f_x (x : ℤ) (hx : x % 3456 = 0) : Int.gcd (f x) x = 48 := by
  sorry

end gcd_f_x_l174_174075


namespace probability_even_sum_5_balls_drawn_l174_174478

theorem probability_even_sum_5_balls_drawn :
  let total_ways := (Nat.choose 12 5)
  let favorable_ways := (Nat.choose 6 0) * (Nat.choose 6 5) + 
                        (Nat.choose 6 2) * (Nat.choose 6 3) + 
                        (Nat.choose 6 4) * (Nat.choose 6 1)
  favorable_ways / total_ways = 1 / 2 :=
by sorry

end probability_even_sum_5_balls_drawn_l174_174478


namespace add_decimals_l174_174496

theorem add_decimals :
  5.467 + 3.92 = 9.387 :=
by
  sorry

end add_decimals_l174_174496


namespace range_of_a_l174_174400

variable (a : ℝ)

theorem range_of_a
  (h : ∃ x : ℝ, x^2 + 2 * a * x + 1 < 0) :
  a < -1 ∨ a > 1 :=
by {
  sorry
}

end range_of_a_l174_174400


namespace test_question_count_l174_174626

def total_test_questions 
  (total_points : ℕ) 
  (points_per_2pt : ℕ) 
  (points_per_4pt : ℕ) 
  (num_2pt_questions : ℕ) 
  (num_4pt_questions : ℕ) : Prop :=
  total_points = points_per_2pt * num_2pt_questions + points_per_4pt * num_4pt_questions 

theorem test_question_count 
  (total_points : ℕ) 
  (points_per_2pt : ℕ) 
  (points_per_4pt : ℕ) 
  (num_2pt_questions : ℕ) 
  (correct_total_questions : ℕ) :
  total_test_questions total_points points_per_2pt points_per_4pt num_2pt_questions (correct_total_questions - num_2pt_questions) → correct_total_questions = 40 :=
by
  intros h
  sorry

end test_question_count_l174_174626


namespace smallest_nonprime_in_range_l174_174726

def smallest_nonprime_with_no_prime_factors_less_than_20 (m : ℕ) : Prop :=
  ¬(Nat.Prime m) ∧ m > 10 ∧ ∀ p : ℕ, Nat.Prime p → p < 20 → ¬(p ∣ m)

theorem smallest_nonprime_in_range :
  smallest_nonprime_with_no_prime_factors_less_than_20 529 ∧ 520 < 529 ∧ 529 ≤ 540 := 
by 
  sorry

end smallest_nonprime_in_range_l174_174726


namespace binom_10_3_eq_120_l174_174187

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l174_174187


namespace mixed_number_calculation_l174_174179

theorem mixed_number_calculation :
  (481 + 1/6) + (265 + 1/12) + (904 + 1/20) - (184 + 29/30) - (160 + 41/42) - (703 + 55/56) = 603 + 3/8 := 
sorry

end mixed_number_calculation_l174_174179


namespace percentage_of_number_is_40_l174_174582

theorem percentage_of_number_is_40 (N : ℝ) (P : ℝ) 
  (h1 : (1/4) * (1/3) * (2/5) * N = 35) 
  (h2 : (P/100) * N = 420) : 
  P = 40 := 
by
  sorry

end percentage_of_number_is_40_l174_174582


namespace gcd_13642_19236_34176_l174_174670

theorem gcd_13642_19236_34176 : Int.gcd (Int.gcd 13642 19236) 34176 = 2 := 
sorry

end gcd_13642_19236_34176_l174_174670


namespace smallest_number_l174_174510

/-
  Let's declare each number in its base form as variables,
  convert them to their decimal equivalents, and assert that the decimal
  value of $(31)_4$ is the smallest among the given numbers.

  Note: We're not providing the proof steps, just the statement.
-/

noncomputable def A_base7_to_dec : ℕ := 2 * 7^1 + 0 * 7^0
noncomputable def B_base5_to_dec : ℕ := 3 * 5^1 + 0 * 5^0
noncomputable def C_base6_to_dec : ℕ := 2 * 6^1 + 3 * 6^0
noncomputable def D_base4_to_dec : ℕ := 3 * 4^1 + 1 * 4^0

theorem smallest_number : D_base4_to_dec < A_base7_to_dec ∧ D_base4_to_dec < B_base5_to_dec ∧ D_base4_to_dec < C_base6_to_dec := by
  sorry

end smallest_number_l174_174510


namespace jogging_walking_ratio_l174_174998

theorem jogging_walking_ratio (total_time walk_time jog_time: ℕ) (h1 : total_time = 21) (h2 : walk_time = 9) (h3 : jog_time = total_time - walk_time) :
  (jog_time : ℚ) / walk_time = 4 / 3 :=
by
  sorry

end jogging_walking_ratio_l174_174998


namespace amount_spent_on_petrol_l174_174497

theorem amount_spent_on_petrol
    (rent milk groceries education miscellaneous savings salary petrol : ℝ)
    (h1 : rent = 5000)
    (h2 : milk = 1500)
    (h3 : groceries = 4500)
    (h4 : education = 2500)
    (h5 : miscellaneous = 2500)
    (h6 : savings = 0.10 * salary)
    (h7 : savings = 2000)
    (total_salary : salary = 20000) : petrol = 2000 := by
  sorry

end amount_spent_on_petrol_l174_174497


namespace solution_inequality_l174_174700

theorem solution_inequality (p p' q q' : ℝ) (hp : p ≠ 0) (hp' : p' ≠ 0)
    (h : -q / p > -q' / p') : q / p < q' / p' :=
by
  sorry

end solution_inequality_l174_174700


namespace probability_club_then_queen_l174_174755

theorem probability_club_then_queen : 
  let total_cards := 52
  let total_clubs := 13
  let total_queens := 4
  let queen_of_clubs := 1
  let non_queen_clubs := total_clubs - queen_of_clubs
  
  let prob_queen_of_clubs_then_other_queen := (queen_of_clubs / total_cards) * ((total_queens - 1) / (total_cards - 1))
  let prob_non_queen_clubs_then_queen := (non_queen_clubs / total_cards) * (total_queens / (total_cards - 1))
  let total_probability := prob_queen_of_clubs_then_other_queen + prob_non_queen_clubs_then_queen
  
  total_probability = 1 / 52 := by
  let total_cards := 52
  let total_clubs := 13
  let total_queens := 4
  let queen_of_clubs := 1
  let non_queen_clubs := total_clubs - queen_of_clubs
  
  let prob_queen_of_clubs_then_other_queen := (queen_of_clubs / total_cards) * ((total_queens - 1) / (total_cards - 1))
  let prob_non_queen_clubs_then_queen := (non_queen_clubs / total_cards) * (total_queens / (total_cards - 1))
  let total_probability := prob_queen_of_clubs_then_other_queen + prob_non_queen_clubs_then_queen
  
  sorry

end probability_club_then_queen_l174_174755


namespace binom_10_3_l174_174323

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end binom_10_3_l174_174323


namespace adam_teaches_650_students_in_10_years_l174_174984

noncomputable def students_in_n_years (n : ℕ) : ℕ :=
  if n = 1 then 40
  else if n = 2 then 60
  else if n = 3 then 70
  else if n <= 10 then 70
  else 0 -- beyond the scope of this problem

theorem adam_teaches_650_students_in_10_years :
  (students_in_n_years 1 + students_in_n_years 2 + students_in_n_years 3 +
   students_in_n_years 4 + students_in_n_years 5 + students_in_n_years 6 +
   students_in_n_years 7 + students_in_n_years 8 + students_in_n_years 9 +
   students_in_n_years 10) = 650 :=
by
  sorry

end adam_teaches_650_students_in_10_years_l174_174984


namespace original_class_strength_l174_174605

theorem original_class_strength 
  (x : ℕ) 
  (h1 : ∀ a_avg n, a_avg = 40 → n = x)
  (h2 : ∀ b_avg m, b_avg = 32 → m = 12)
  (h3 : ∀ new_avg, new_avg = 36 → ((x * 40 + 12 * 32) = ((x + 12) * 36))) : 
  x = 12 :=
by 
  sorry

end original_class_strength_l174_174605


namespace perpendicular_line_equation_l174_174067

theorem perpendicular_line_equation 
  (p : ℝ × ℝ)
  (L1 : ℝ → ℝ → Prop)
  (L2 : ℝ → ℝ → ℝ → Prop) 
  (hx : p = (1, -1)) 
  (hL1 : ∀ x y, L1 x y ↔ 3 * x - 2 * y = 0) 
  (hL2 : ∀ x y m, L2 x y m ↔ 2 * x + 3 * y + m = 0) :
  ∃ m : ℝ, L2 (p.1) (p.2) m ∧ 2 * p.1 + 3 * p.2 + m = 0 :=
by
  sorry

end perpendicular_line_equation_l174_174067


namespace binomial_10_3_eq_120_l174_174199

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l174_174199


namespace terminating_decimal_count_l174_174677

theorem terminating_decimal_count : ∃ n, n = 23 ∧ (∀ k, 1 ≤ k ∧ k ≤ 499 → (∃ m, k = 21 * m)) :=
by
  sorry

end terminating_decimal_count_l174_174677


namespace susan_more_cats_than_bob_l174_174598

-- Given problem: Initial and transaction conditions
def susan_initial_cats : ℕ := 21
def bob_initial_cats : ℕ := 3
def susan_additional_cats : ℕ := 5
def bob_additional_cats : ℕ := 7
def susan_gives_bob_cats : ℕ := 4

-- Declaration to find the difference between Susan's and Bob's cats
def final_susan_cats (initial : ℕ) (additional : ℕ) (given : ℕ) : ℕ := initial + additional - given
def final_bob_cats (initial : ℕ) (additional : ℕ) (received : ℕ) : ℕ := initial + additional + received

-- The proof statement which we need to show
theorem susan_more_cats_than_bob : 
  final_susan_cats susan_initial_cats susan_additional_cats susan_gives_bob_cats - 
  final_bob_cats bob_initial_cats bob_additional_cats susan_gives_bob_cats = 8 := by
  sorry

end susan_more_cats_than_bob_l174_174598


namespace triangle_inequality_l174_174727

variables {A B C P D E F : Type} -- Variables representing points in the plane.
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P]
variables (PD PE PF PA PB PC : ℝ) -- Distances corresponding to the points.

-- Condition stating P lies inside or on the boundary of triangle ABC
axiom P_in_triangle_ABC : ∀ (A B C P : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P], 
  (PD > 0 ∧ PE > 0 ∧ PF > 0 ∧ PA > 0 ∧ PB > 0 ∧ PC > 0)

-- Objective statement to prove
theorem triangle_inequality (PD PE PF PA PB PC : ℝ) 
  (h1 : PA ≥ 0) 
  (h2 : PB ≥ 0) 
  (h3 : PC ≥ 0) 
  (h4 : PD ≥ 0) 
  (h5 : PE ≥ 0) 
  (h6 : PF ≥ 0) :
  PA + PB + PC ≥ 2 * (PD + PE + PF) := 
sorry -- Proof to be provided later.

end triangle_inequality_l174_174727


namespace cube_volume_from_surface_area_l174_174002

theorem cube_volume_from_surface_area (S : ℝ) (h : S = 864) : ∃ V : ℝ, V = 1728 :=
by
  -- Assume surface area formula S = 6s^2, solve steps skipped and go directly to conclusion
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  use V
  sorry

end cube_volume_from_surface_area_l174_174002


namespace binom_10_3_l174_174327

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end binom_10_3_l174_174327


namespace find_k_l174_174956

-- Defining the conditions used in the problem context
def line_condition (k a b : ℝ) : Prop :=
  (b = 4 * k + 1) ∧ (5 = k * a + 1) ∧ (b + 1 = k * a + 1)

-- The statement of the theorem
theorem find_k (a b k : ℝ) (h : line_condition k a b) : k = 3 / 4 :=
by sorry

end find_k_l174_174956


namespace students_not_playing_games_l174_174753

theorem students_not_playing_games 
  (total_students : ℕ)
  (basketball_players : ℕ)
  (volleyball_players : ℕ)
  (both_players : ℕ)
  (h1 : total_students = 20)
  (h2 : basketball_players = (1 / 2) * total_students)
  (h3 : volleyball_players = (2 / 5) * total_students)
  (h4 : both_players = (1 / 10) * total_students) :
  total_students - ((basketball_players + volleyball_players) - both_players) = 4 :=
by
  sorry

end students_not_playing_games_l174_174753


namespace binom_10_3_l174_174227

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end binom_10_3_l174_174227


namespace xiao_ying_should_pay_l174_174113

variable (x y z : ℝ)

def equation1 := 3 * x + 7 * y + z = 14
def equation2 := 4 * x + 10 * y + z = 16
def equation3 := 2 * (x + y + z) = 20

theorem xiao_ying_should_pay :
  equation1 x y z →
  equation2 x y z →
  equation3 x y z :=
by
  intros h1 h2
  sorry

end xiao_ying_should_pay_l174_174113


namespace three_term_inequality_l174_174822

theorem three_term_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_abc : a * b * c = 1) :
  (a^3 / (a^3 + 2 * b^2)) + (b^3 / (b^3 + 2 * c^2)) + (c^3 / (c^3 + 2 * a^2)) ≥ 1 :=
by
  sorry

end three_term_inequality_l174_174822


namespace number_of_sequences_alternating_parity_l174_174553

/-- 
The number of sequences of 8 digits x_1, x_2, ..., x_8 where no two adjacent x_i have the same parity is 781,250.
-/
theorem number_of_sequences_alternating_parity : 
  let num_sequences := 10 * 5^7 
  ∑ x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8 (digits : Fin 8 → Fin 10), 
    (∀ i : Fin 7, digits i % 2 ≠ digits (i + 1) % 2) → 1 = 781250 :=
by sorry

end number_of_sequences_alternating_parity_l174_174553


namespace cube_volume_from_surface_area_l174_174001

theorem cube_volume_from_surface_area (S : ℝ) (h : S = 864) : ∃ V : ℝ, V = 1728 :=
by
  -- Assume surface area formula S = 6s^2, solve steps skipped and go directly to conclusion
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  use V
  sorry

end cube_volume_from_surface_area_l174_174001


namespace binomial_10_3_eq_120_l174_174204

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l174_174204


namespace binom_10_3_l174_174336

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end binom_10_3_l174_174336


namespace common_divisors_count_l174_174552

-- Definitions of the numbers involved
def n1 : ℕ := 9240
def n2 : ℕ := 10800

-- Prime factorizations based on the conditions
def factor_n1 : Prop := n1 = 2^3 * 3^1 * 5^1 * 7 * 11
def factor_n2 : Prop := n2 = 2^3 * 3^3 * 5^2

-- GCD as defined in the conditions
def gcd_value : ℕ := Nat.gcd n1 n2

-- Proof problem: prove the number of positive divisors of the gcd of n1 and n2 is 16
theorem common_divisors_count (h1 : factor_n1) (h2 : factor_n2) : Nat.divisors (Nat.gcd n1 n2).card = 16 := by
  sorry

end common_divisors_count_l174_174552


namespace compute_div_mul_l174_174057

theorem compute_div_mul (x y z : Int) (h : y ≠ 0) (hx : x = -100) (hy : y = -25) (hz : z = -6) :
  (((-x) / (-y)) * -z) = -24 := by
  sorry

end compute_div_mul_l174_174057


namespace binom_10_3_eq_120_l174_174223

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end binom_10_3_eq_120_l174_174223


namespace binom_10_3_eq_120_l174_174213

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end binom_10_3_eq_120_l174_174213


namespace y_intercept_of_line_l174_174646

theorem y_intercept_of_line (m x y b : ℝ) (h_slope : m = 4) (h_point : (x, y) = (199, 800)) (h_line : y = m * x + b) :
    b = 4 :=
by
  sorry

end y_intercept_of_line_l174_174646


namespace inequality_holds_l174_174810

theorem inequality_holds (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by
  sorry

end inequality_holds_l174_174810


namespace gcd_possible_values_l174_174773

theorem gcd_possible_values (a b : ℕ) (hab : a * b = 288) : 
  ∃ S : Finset ℕ, (∀ g : ℕ, g ∈ S ↔ ∃ p q r s : ℕ, p + r = 5 ∧ q + s = 2 ∧ g = 2^min p r * 3^min q s) 
  ∧ S.card = 14 := 
sorry

end gcd_possible_values_l174_174773


namespace problem1_problem2_l174_174599

variable {Ω : Type}

-- Probabilities for the first firing
variables (A1 A2 A3 : Ω → Prop)
variables (P1 : ProbabilisticMeasure Ω) [P1.IsProbabilityMeasure]
variables (PA1 PA2 PA3 : ℝ)

axiom pa1_def : PA1 = 1 / 2
axiom pa2_def : PA2 = 4 / 5
axiom pa3_def : PA3 = 3 / 5

-- Equivalence of the first problem
theorem problem1 :
  P1[A1] = PA1 → P1[A2] = PA2 → P1[A3] = PA3 →
  P1[A1 ∧ ¬A2 ∧ ¬A3] + P1[¬A1 ∧ A2 ∧ ¬A3] + P1[¬A1 ∧ ¬A2 ∧ A3] = 13 / 50 :=
by
  intros h1 h2 h3
  have PA1_not_children := P1[¬A2] * P1[¬A3]
  have P1_children := focused_prob h1 h2 h3
  calc
    P1[A1 ∧ ¬A2 ∧ ¬A3] + P1[¬A1 ∧ A2 ∧ ¬A3] + P1[¬A1 ∧ ¬A2 ∧ A3]
        = 13 / 50 := sorry

-- Probabilities after both firings
variables (A1' A2' A3' : Ω → Prop)
variables (PA1' PA2' PA3' : ℝ)

axiom pa1p_def : PA1' = 4 / 5
axiom pa2p_def : PA2' = 1 / 2
axiom pa3p_def : PA3' = 2 / 3

-- Equivalence of the second problem
theorem problem2 :
  P1[A1 → A1'] = PA1' → P1[A2 → A2'] = PA2' → P1[A3 → A3'] = PA3' →
  let p := 2 / 5 in 
  let n := 3 in
  (n * p = 1.2) :=
by
  intros h1' h2' h3'
  let p := 2 / 5
  let n := 3
  have calc := expected_val lean_exp 
  show EqProp (n * p = 1.2) := sorry

end problem1_problem2_l174_174599


namespace cos_8_identity_l174_174697

theorem cos_8_identity (m : ℝ) (h : Real.sin 74 = m) : 
  Real.cos 8 = Real.sqrt ((1 + m) / 2) :=
sorry

end cos_8_identity_l174_174697


namespace inequality_proof_l174_174526

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧
  (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 :=
sorry

end inequality_proof_l174_174526


namespace range_of_a_l174_174085

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (2 * a - x > 1 → x < 2 * a - 1)) ∧
  (∀ x : ℝ, (2 * x + 5 > 3 * a → x > (3 * a - 5) / 2)) ∧
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 6 →
    (x < 2 * a - 1 ∧ x > (3 * a - 5) / 2))) →
  7 / 3 ≤ a ∧ a ≤ 7 / 2 :=
by
  sorry

end range_of_a_l174_174085


namespace binomial_10_3_l174_174300

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end binomial_10_3_l174_174300


namespace soccer_team_starters_l174_174583

open Nat

-- Definitions representing the conditions
def total_players : ℕ := 18
def twins_included : ℕ := 2
def remaining_players : ℕ := total_players - twins_included
def starters_to_choose : ℕ := 7 - twins_included

-- Theorem statement to assert the solution
theorem soccer_team_starters :
  Nat.choose remaining_players starters_to_choose = 4368 :=
by
  -- Placeholder for proof
  sorry

end soccer_team_starters_l174_174583


namespace arithmetic_sequence_geometric_term_ratio_l174_174872

theorem arithmetic_sequence_geometric_term_ratio (a : ℕ → ℤ) (d : ℤ) (h₀ : d ≠ 0)
  (h₁ : a 1 = a 1)
  (h₂ : a 3 = a 1 + 2 * d)
  (h₃ : a 4 = a 1 + 3 * d)
  (h_geom : (a 1 + 2 * d)^2 = a 1 * (a 1 + 3 * d)) :
  (a 1 + a 5 + a 17) / (a 2 + a 6 + a 18) = 8 / 11 :=
by
  sorry

end arithmetic_sequence_geometric_term_ratio_l174_174872


namespace binomial_coefficient_10_3_l174_174278

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l174_174278


namespace find_angle_l174_174936

-- Given the complement condition
def complement_condition (x : ℝ) : Prop :=
  x + 2 * (4 * x + 10) = 90

-- Proving the degree measure of the angle
theorem find_angle (x : ℝ) : complement_condition x → x = 70 / 9 := by
  intro hc
  sorry

end find_angle_l174_174936


namespace combination_10_3_eq_120_l174_174252

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end combination_10_3_eq_120_l174_174252


namespace negation_of_proposition_l174_174450

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, x ≤ -1 ∨ x ≥ 2) ↔ ∀ x : ℝ, -1 < x ∧ x < 2 :=
by
  sorry

end negation_of_proposition_l174_174450


namespace find_difference_l174_174051

-- Define the initial amounts each person paid.
def Alex_paid : ℕ := 95
def Tom_paid : ℕ := 140
def Dorothy_paid : ℕ := 110
def Sammy_paid : ℕ := 155

-- Define the total spent and the share per person.
def total_spent : ℕ := Alex_paid + Tom_paid + Dorothy_paid + Sammy_paid
def share : ℕ := total_spent / 4

-- Define how much each person needs to pay or should receive.
def Alex_balance : ℤ := share - Alex_paid
def Tom_balance : ℤ := Tom_paid - share
def Dorothy_balance : ℤ := share - Dorothy_paid
def Sammy_balance : ℤ := Sammy_paid - share

-- Define the values of t and d.
def t : ℤ := 0
def d : ℤ := 15

-- The proof goal
theorem find_difference : t - d = -15 := by
  sorry

end find_difference_l174_174051


namespace select_students_l174_174131

-- Definitions for the conditions
variables (A B C D E : Prop)

-- Conditions
def condition1 : Prop := A → B ∧ ¬E
def condition2 : Prop := (B ∨ E) → ¬D
def condition3 : Prop := C ∨ D

-- The main theorem
theorem select_students (hA : A) (h1 : condition1 A B E) (h2 : condition2 B E D) (h3 : condition3 C D) : B ∧ C :=
by 
  sorry

end select_students_l174_174131


namespace work_days_l174_174473

theorem work_days (A B C : ℝ) (h₁ : A + B = 1 / 15) (h₂ : C = 1 / 7.5) : 1 / (A + B + C) = 5 :=
by
  sorry

end work_days_l174_174473


namespace area_of_circle_l174_174619

def circle_area (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x - 4 * y = 1

theorem area_of_circle : ∃ (area : ℝ), area = 6 * Real.pi :=
by sorry

end area_of_circle_l174_174619


namespace dagger_example_l174_174999

def dagger (m n p q : ℚ) : ℚ := 2 * m * p * (q / n)

theorem dagger_example : dagger 5 8 3 4 = 15 := by
  sorry

end dagger_example_l174_174999


namespace solve_equation_1_solve_equation_2_l174_174440

theorem solve_equation_1 (x : ℝ) : 2 * (x + 1)^2 - 49 = 1 ↔ x = 4 ∨ x = -6 := sorry

theorem solve_equation_2 (x : ℝ) : (1 / 2) * (x - 1)^3 = -4 ↔ x = -1 := sorry

end solve_equation_1_solve_equation_2_l174_174440


namespace fixed_point_of_family_of_lines_l174_174610

theorem fixed_point_of_family_of_lines :
  ∀ (m : ℝ), ∃ (x y : ℝ), (2 * x - m * y + 1 - 3 * m = 0) ∧ (x = -1 / 2) ∧ (y = -3) :=
by
  intro m
  use -1 / 2, -3
  constructor
  · sorry
  constructor
  · rfl
  · rfl

end fixed_point_of_family_of_lines_l174_174610


namespace sum_of_midpoint_coordinates_l174_174133

theorem sum_of_midpoint_coordinates :
  let x1 := 3
  let y1 := -1
  let x2 := 11
  let y2 := 21
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  midpoint_x + midpoint_y = 17 := by
  sorry

end sum_of_midpoint_coordinates_l174_174133


namespace sum_of_fifth_powers_l174_174957

theorem sum_of_fifth_powers (a b c d : ℝ) (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 := sorry

end sum_of_fifth_powers_l174_174957


namespace find_three_digit_number_l174_174066

-- Define digits a, b, c where a is non-zero for the three-digit number
variables (a b c : ℕ)
-- Conditions for digits
variables (ha : a ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) (hb : b ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) (hc : c ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})

-- Define the three-digit number
def number := 100 * a + 10 * b + c

-- Define the sum of the digits
def digit_sum := a + b + c

-- Theorem to prove the characterization of the number
theorem find_three_digit_number (h : number a b c = 12 * digit_sum a b c) :
  number a b c = 108 :=
by {
  sorry
}

end find_three_digit_number_l174_174066


namespace total_population_increase_l174_174893
-- Import the required library

-- Define the conditions for Region A and Region B
def regionA_births_0_14 (time: ℕ) := time / 20
def regionA_births_15_64 (time: ℕ) := time / 30
def regionB_births_0_14 (time: ℕ) := time / 25
def regionB_births_15_64 (time: ℕ) := time / 35

-- Define the total number of people in each age group for both regions
def regionA_population_0_14 := 2000
def regionA_population_15_64 := 6000
def regionB_population_0_14 := 1500
def regionB_population_15_64 := 5000

-- Define the total time in seconds
def total_time := 25 * 60

-- Proof statement
theorem total_population_increase : 
  regionA_population_0_14 * regionA_births_0_14 total_time +
  regionA_population_15_64 * regionA_births_15_64 total_time +
  regionB_population_0_14 * regionB_births_0_14 total_time +
  regionB_population_15_64 * regionB_births_15_64 total_time = 227 := 
by sorry

end total_population_increase_l174_174893


namespace fruit_seller_loss_percentage_l174_174043

theorem fruit_seller_loss_percentage :
  ∃ (C : ℝ), 
    (5 : ℝ) = C - (6.25 - C * (1 + 0.05)) → 
    (C = 6.25) → 
    (C - 5 = 1.25) → 
    (1.25 / 6.25 * 100 = 20) :=
by 
  sorry

end fruit_seller_loss_percentage_l174_174043


namespace a_n_is_perfect_square_l174_174575

def seqs (a b : ℕ → ℤ) : Prop :=
  a 0 = 1 ∧ b 0 = 0 ∧ ∀ n, a (n + 1) = 7 * a n + 6 * b n - 3 ∧ b (n + 1) = 8 * a n + 7 * b n - 4

theorem a_n_is_perfect_square (a b : ℕ → ℤ) (h : seqs a b) :
  ∀ n, ∃ k : ℤ, a n = k^2 :=
by
  sorry

end a_n_is_perfect_square_l174_174575


namespace line_through_A_with_equal_intercepts_l174_174373

theorem line_through_A_with_equal_intercepts (x y : ℝ) (A : ℝ × ℝ) (hx : A = (2, 1)) :
  (∃ k : ℝ, x + y = k ∧ x + y - 3 = 0) ∨ (x - 2 * y = 0) :=
sorry

end line_through_A_with_equal_intercepts_l174_174373


namespace number_of_daisies_is_two_l174_174122

theorem number_of_daisies_is_two :
  ∀ (total_flowers daisies tulips sunflowers remaining_flowers : ℕ), 
    total_flowers = 12 →
    sunflowers = 4 →
    (3 / 5) * remaining_flowers = tulips →
    (2 / 5) * remaining_flowers = sunflowers →
    remaining_flowers = total_flowers - daisies - sunflowers →
    daisies = 2 :=
by
  intros total_flowers daisies tulips sunflowers remaining_flowers 
  sorry

end number_of_daisies_is_two_l174_174122


namespace Joan_seashells_l174_174097

theorem Joan_seashells (J_J : ℕ) (J : ℕ) (h : J + J_J = 14) (hJJ : J_J = 8) : J = 6 :=
by
  sorry

end Joan_seashells_l174_174097


namespace combination_10_3_l174_174293

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end combination_10_3_l174_174293


namespace necessary_not_sufficient_condition_l174_174044

theorem necessary_not_sufficient_condition (x : ℝ) :
  ((-6 ≤ x ∧ x ≤ 3) → (-5 ≤ x ∧ x ≤ 3)) ∧
  (¬ ((-5 ≤ x ∧ x ≤ 3) → (-6 ≤ x ∧ x ≤ 3))) :=
by
  -- Need proof steps here
  sorry

end necessary_not_sufficient_condition_l174_174044


namespace part1_part2a_part2b_part2c_l174_174875

def f (x a : ℝ) := |2 * x - 1| + |x - a|

theorem part1 (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2) : f x 3 ≤ 4 := sorry

theorem part2a (a x : ℝ) (h0 : a < 1 / 2) (h1 : a ≤ x ∧ x ≤ 1 / 2) : f x a = |x - 1 + a| := sorry

theorem part2b (a x : ℝ) (h0 : a = 1 / 2) (h1 : x = 1 / 2) : f x a = |x - 1 + a| := sorry

theorem part2c (a x : ℝ) (h0 : a > 1 / 2) (h1 : 1 / 2 ≤ x ∧ x ≤ a) : f x a = |x - 1 + a| := sorry

end part1_part2a_part2b_part2c_l174_174875


namespace teal_more_blue_l174_174039

theorem teal_more_blue (total : ℕ) (green : ℕ) (both_green_blue : ℕ) (neither_green_blue : ℕ)
  (h1 : total = 150) (h2 : green = 90) (h3 : both_green_blue = 40) (h4 : neither_green_blue = 25) :
  ∃ (blue : ℕ), blue = 75 :=
by
  sorry

end teal_more_blue_l174_174039


namespace exercise_l174_174933

theorem exercise (a b : ℕ) (h1 : 656 = 3 * 7^2 + a * 7 + b) (h2 : 656 = 3 * 10^2 + a * 10 + b) : 
  (a * b) / 15 = 1 :=
by
  sorry

end exercise_l174_174933


namespace roots_opposite_k_eq_2_l174_174563

theorem roots_opposite_k_eq_2 (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 + x2 = 0 ∧ x1 * x2 = -1 ∧ x1 ≠ x2 ∧ x1*x1 + (k-2)*x1 - 1 = 0 ∧ x2*x2 + (k-2)*x2 - 1 = 0) → k = 2 :=
by
  sorry

end roots_opposite_k_eq_2_l174_174563


namespace second_set_parallel_lines_l174_174567

theorem second_set_parallel_lines (n : ℕ) :
  (5 * (n - 1)) = 280 → n = 71 :=
by
  intros h
  sorry

end second_set_parallel_lines_l174_174567


namespace binomial_10_3_eq_120_l174_174350

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l174_174350


namespace power_multiplication_l174_174865

variable (x y m n : ℝ)

-- Establishing our initial conditions
axiom h1 : 10^x = m
axiom h2 : 10^y = n

theorem power_multiplication : 10^(2*x + 3*y) = m^2 * n^3 :=
by
  sorry

end power_multiplication_l174_174865


namespace number_of_wheels_on_each_bicycle_l174_174946

theorem number_of_wheels_on_each_bicycle 
  (num_bicycles : ℕ)
  (num_tricycles : ℕ)
  (wheels_per_tricycle : ℕ)
  (total_wheels : ℕ)
  (h_bicycles : num_bicycles = 24)
  (h_tricycles : num_tricycles = 14)
  (h_wheels_tricycle : wheels_per_tricycle = 3)
  (h_total_wheels : total_wheels = 90) :
  2 * num_bicycles + 3 * num_tricycles = 90 → 
  num_bicycles = 24 → 
  num_tricycles = 14 → 
  wheels_per_tricycle = 3 → 
  total_wheels = 90 → 
  ∃ b : ℕ, b = 2 :=
by
  sorry

end number_of_wheels_on_each_bicycle_l174_174946


namespace jail_time_calculation_l174_174761

def total_arrests (arrests_per_day : ℕ) (cities : ℕ) (days : ℕ) : ℕ := 
  arrests_per_day * cities * days

def jail_time_before_trial (arrests : ℕ) (days_before_trial : ℕ) : ℕ := 
  days_before_trial * arrests

def jail_time_after_trial (arrests : ℕ) (weeks_after_trial : ℕ) : ℕ := 
  weeks_after_trial * arrests

def combined_jail_time (weeks_before_trial : ℕ) (weeks_after_trial : ℕ) : ℕ := 
  weeks_before_trial + weeks_after_trial

noncomputable def total_jail_time_in_weeks : ℕ := 
  let arrests := total_arrests 10 21 30
  let weeks_before_trial := jail_time_before_trial arrests 4 / 7
  let weeks_after_trial := jail_time_after_trial arrests 1
  combined_jail_time weeks_before_trial weeks_after_trial

theorem jail_time_calculation : 
  total_jail_time_in_weeks = 9900 :=
sorry

end jail_time_calculation_l174_174761


namespace combination_10_3_eq_120_l174_174352

open Nat

theorem combination_10_3_eq_120 : (10.choose 3) = 120 := 
by
  sorry

end combination_10_3_eq_120_l174_174352


namespace problem_l174_174867

variable {a b c : ℝ}

theorem problem (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  a + 4 / b ≤ -4 ∨ b + 4 / c ≤ -4 ∨ c + 4 / a ≤ -4 := 
sorry

end problem_l174_174867


namespace total_number_of_outfits_l174_174444

noncomputable def number_of_outfits (shirts pants ties jackets : ℕ) :=
  shirts * pants * ties * jackets

theorem total_number_of_outfits :
  number_of_outfits 8 5 5 3 = 600 :=
by
  sorry

end total_number_of_outfits_l174_174444


namespace part_a_part_b_l174_174117

-- Part (a)
theorem part_a (students : Fin 67) (answers : Fin 6 → Bool) :
  ∃ (s1 s2 : Fin 67), s1 ≠ s2 ∧ answers s1 = answers s2 := by
  sorry

-- Part (b)
theorem part_b (students : Fin 67) (points : Fin 6 → ℤ)
  (h_points : ∀ k, points k = k ∨ points k = -k) :
  ∃ (scores : Fin 67 → ℤ), ∃ (s1 s2 s3 s4 : Fin 67),
  s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ s2 ≠ s3 ∧ s2 ≠ s4 ∧ s3 ≠ s4 ∧
  scores s1 = scores s2 ∧ scores s1 = scores s3 ∧ scores s1 = scores s4 := by
  sorry

end part_a_part_b_l174_174117


namespace inequality_solution_l174_174381

theorem inequality_solution (x : ℝ) : x^3 - 12 * x^2 > -36 * x ↔ x ∈ Set.Ioo 0 6 ∪ Set.Ioi 6 := by
  sorry

end inequality_solution_l174_174381


namespace rick_books_division_l174_174587

theorem rick_books_division (books_per_group initial_books final_groups : ℕ) 
  (h_initial : initial_books = 400) 
  (h_books_per_group : books_per_group = 25) 
  (h_final_groups : final_groups = 16) : 
  ∃ divisions : ℕ, (divisions = 4) ∧ 
    ∃ f : ℕ → ℕ, 
    (f 0 = initial_books) ∧ 
    (f divisions = books_per_group * final_groups) ∧ 
    (∀ n, 1 ≤ n → n ≤ divisions → f n = f (n - 1) / 2) := 
by 
  sorry

end rick_books_division_l174_174587


namespace binom_10_3_eq_120_l174_174217

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end binom_10_3_eq_120_l174_174217


namespace CoreyCandies_l174_174602

theorem CoreyCandies (T C : ℕ) (h1 : T + C = 66) (h2 : T = C + 8) : C = 29 :=
by
  sorry

end CoreyCandies_l174_174602


namespace inequality_proof_l174_174520

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  1 ≤ ((x + y) * (x^3 + y^3)) / (x^2 + y^2)^2 ∧ 
  ((x + y) * (x^3 + y^3)) / (x^2 + y^2)^2 ≤ 9 / 8 :=
sorry

end inequality_proof_l174_174520


namespace part1_part2_l174_174876

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) - abs (x - 2)

theorem part1 : 
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 1} :=
sorry 

noncomputable def g (x : ℝ) : ℝ := f x - x^2 + x

theorem part2 (m : ℝ) : 
  (∃ x : ℝ, f x ≥ x^2 - x + m) → m ≤ 5/4 :=
sorry 

end part1_part2_l174_174876


namespace avg_of_first_21_multiples_l174_174620

theorem avg_of_first_21_multiples (n : ℕ) (h : (21 * 11 * n / 21) = 88) : n = 8 :=
by
  sorry

end avg_of_first_21_multiples_l174_174620


namespace binomial_10_3_l174_174304

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end binomial_10_3_l174_174304


namespace don_walking_speed_l174_174055

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

end don_walking_speed_l174_174055


namespace binom_10_3_l174_174330

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end binom_10_3_l174_174330


namespace annual_income_of_A_l174_174631

theorem annual_income_of_A 
  (ratio_AB : ℕ → ℕ → Prop)
  (income_C : ℕ)
  (income_B_more_C : ℕ → ℕ → Prop)
  (income_B_from_ratio : ℕ → ℕ → Prop)
  (income_C_value : income_C = 16000)
  (income_B_condition : ∀ c, income_B_more_C 17920 c)
  (income_A_condition : ∀ b, ratio_AB 5 (b/2))
  : ∃ a, a = 537600 :=
by
  sorry

end annual_income_of_A_l174_174631


namespace find_a_for_tangency_l174_174548

-- Definitions of line and parabola
def line (x y : ℝ) : Prop := x - y - 1 = 0
def parabola (x y : ℝ) (a : ℝ) : Prop := y = a * x^2

-- The tangency condition for quadratic equations
def tangency_condition (a : ℝ) : Prop := 1 - 4 * a = 0

theorem find_a_for_tangency (a : ℝ) :
  (∀ x y, line x y → parabola x y a → tangency_condition a) → a = 1/4 :=
by
  -- Proof omitted
  sorry

end find_a_for_tangency_l174_174548


namespace cows_dogs_ratio_l174_174948

theorem cows_dogs_ratio (C D : ℕ) (hC : C = 184) (hC_remain : 3 / 4 * C = 138)
  (hD_remain : 1 / 4 * D + 138 = 161) : C / D = 2 :=
sorry

end cows_dogs_ratio_l174_174948


namespace factory_output_l174_174101

variable (a : ℝ)
variable (n : ℕ)
variable (r : ℝ)

-- Initial condition: the output value increases by 10% each year for 5 years
def annual_growth (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^n

-- Theorem statement
theorem factory_output (a : ℝ) : annual_growth a 1.1 5 = 1.1^5 * a :=
by
  sorry

end factory_output_l174_174101


namespace binomial_10_3_l174_174253

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end binomial_10_3_l174_174253


namespace binom_10_3_l174_174326

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end binom_10_3_l174_174326


namespace min_a_l174_174443

-- Definitions and conditions used in the problem
def eqn (a x : ℝ) : Prop :=
  2 * sin (π - (π * x^2) / 12) * cos ((π / 6) * sqrt (9 - x^2)) + 1 =
  a + 2 * sin ((π / 6) * sqrt (9 - x^2)) * cos ((π * x^2) / 12)

-- Statement to prove the minimum value of a
theorem min_a : ∃ x : ℝ, eqn 2 x := sorry

end min_a_l174_174443


namespace power_six_sum_l174_174404

theorem power_six_sum (x : ℝ) (h : x + 1 / x = 3) : x^6 + 1 / x^6 = 322 := 
by 
  sorry

end power_six_sum_l174_174404


namespace problem1_l174_174656

theorem problem1 : 13 + (-24) - (-40) = 29 := by
  sorry

end problem1_l174_174656


namespace polar_to_cartesian_l174_174877

theorem polar_to_cartesian (θ ρ x y : ℝ) (h1 : ρ = 2 * Real.sin θ) (h2 : x = ρ * Real.cos θ) (h3 : y = ρ * Real.sin θ) :
  x^2 + (y - 1)^2 = 1 :=
sorry

end polar_to_cartesian_l174_174877


namespace binomial_10_3_l174_174308

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end binomial_10_3_l174_174308


namespace james_drinks_per_day_l174_174416

-- condition: James buys 5 packs of sodas, each contains 12 sodas
def num_packs : Nat := 5
def sodas_per_pack : Nat := 12
def sodas_bought : Nat := num_packs * sodas_per_pack

-- condition: James already had 10 sodas
def sodas_already_had : Nat := 10

-- condition: James finishes all the sodas in 1 week (7 days)
def days_in_week : Nat := 7

-- total sodas
def total_sodas : Nat := sodas_bought + sodas_already_had

-- number of sodas james drinks per day
def sodas_per_day : Nat := 10

-- proof problem
theorem james_drinks_per_day : (total_sodas / days_in_week) = sodas_per_day :=
  sorry

end james_drinks_per_day_l174_174416


namespace fraction_of_income_from_tips_l174_174152

theorem fraction_of_income_from_tips (S T I : ℚ) (h1 : T = (5/3) * S) (h2 : I = S + T) :
  T / I = 5 / 8 :=
by
  -- We're only required to state the theorem, not prove it.
  sorry

end fraction_of_income_from_tips_l174_174152


namespace intersect_lines_at_single_point_l174_174539

open Triangle

theorem intersect_lines_at_single_point
  {A B C A1 B1 C1 K L M N P : Point}
  (h_reg_pentagon_KLNP : regular_pentagon MKLNP)
  (h_KL_on_BC : K ∈ line.segment B C)
  (hKL_mid_A1 : midpoint K L A1)
  (h_points_on_sides : M ∈ line.segment A B ∧ N ∈ line.segment A C)
  (h_C1_def : midpoint C1 P ∧ C1 ∈ line.segment A B)
  (h_B1_def : midpoint B1 P ∧ B1 ∈ line.segment A C)
  (h_ratios : (|B - A1| / |A1 - C|) * (|C - B1| / |B1 - A|) * (|A - C1| / |C1 - B|) = 1) :
  concurrent (line.mk A A1) (line.mk B B1) (line.mk C C1) :=
by
  sorry

end intersect_lines_at_single_point_l174_174539


namespace inequality_holds_l174_174815

theorem inequality_holds (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by
  sorry

end inequality_holds_l174_174815


namespace sam_distinct_meals_count_l174_174053

-- Definitions based on conditions
def main_dishes := ["Burger", "Pasta", "Salad"]
def beverages := ["Soda", "Juice"]
def snacks := ["Chips", "Cookie", "Apple"]

-- Definition to exclude invalid combinations
def is_valid_combination (main : String) (beverage : String) : Bool :=
  if main = "Burger" && beverage = "Soda" then false else true

-- Number of valid combinations
def count_valid_meals : Nat :=
  main_dishes.length * beverages.length * snacks.length - snacks.length

theorem sam_distinct_meals_count : count_valid_meals = 15 := 
  sorry

end sam_distinct_meals_count_l174_174053


namespace distribute_problems_l174_174985

theorem distribute_problems :
  let n_problems := 7
  let n_friends := 12
  (n_friends ^ n_problems) = 35831808 :=
by 
  sorry

end distribute_problems_l174_174985


namespace cube_volume_l174_174021

theorem cube_volume (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
sorry

end cube_volume_l174_174021


namespace amy_seeds_l174_174652

-- Define the conditions
def bigGardenSeeds : Nat := 47
def smallGardens : Nat := 9
def seedsPerSmallGarden : Nat := 6

-- Define the total seeds calculation
def totalSeeds := bigGardenSeeds + smallGardens * seedsPerSmallGarden

-- The theorem to be proved
theorem amy_seeds : totalSeeds = 101 := by
  sorry

end amy_seeds_l174_174652


namespace union_complement_eq_l174_174078

open Set

variable {U : Set ℝ} {A B : Set ℝ}

-- Define the universal set U as ℝ
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {x : ℝ | (x + 2) * (x - 1) > 0}

-- Define set B
def B : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 0}

-- Define the complement of B in U
def C_U_B : Set ℝ := compl B

theorem union_complement_eq :
  A ∪ C_U_B = {x : ℝ | x < -1 ∨ x ≥ 0} := by
    sorry

end union_complement_eq_l174_174078


namespace binomial_10_3_eq_120_l174_174202

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l174_174202


namespace ball_bounce_height_l174_174040

noncomputable def height_after_bounces (h₀ : ℝ) (r : ℝ) (b : ℕ) : ℝ :=
  h₀ * (r ^ b)

theorem ball_bounce_height
  (h₀ : ℝ) (r : ℝ) (hb : ℕ) (h₀_pos : h₀ > 0) (r_pos : 0 < r ∧ r < 1) (h₀_val : h₀ = 320) (r_val : r = 3 / 4) (height_limit : ℝ) (height_limit_val : height_limit = 40):
  (hb ≥ 6) ∧ height_after_bounces h₀ r hb < height_limit :=
by
  sorry

end ball_bounce_height_l174_174040


namespace det_scaled_matrices_l174_174544

variable (a b c d : ℝ)

-- Given condition: determinant of the original matrix
def det_A : ℝ := Matrix.det ![![a, b], ![c, d]]

-- Problem statement: determinants of the scaled matrices
theorem det_scaled_matrices
    (h: det_A a b c d = 3) :
  Matrix.det ![![3 * a, 3 * b], ![3 * c, 3 * d]] = 27 ∧
  Matrix.det ![![4 * a, 2 * b], ![4 * c, 2 * d]] = 24 :=
by
  sorry

end det_scaled_matrices_l174_174544


namespace sum_divisible_by_5_and_7_remainder_12_l174_174169

theorem sum_divisible_by_5_and_7_remainder_12 :
  let a := 105
  let d := 35
  let n := 2013
  let S := (n * (2 * a + (n - 1) * d)) / 2
  S % 12 = 3 :=
by
  sorry

end sum_divisible_by_5_and_7_remainder_12_l174_174169


namespace binomial_10_3_l174_174295

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end binomial_10_3_l174_174295


namespace solution_set_inequality_l174_174674

theorem solution_set_inequality (x : ℝ) (h : x ≠ 0) : 
  (x - 1) / x > 1 → x < 0 := 
by 
  sorry

end solution_set_inequality_l174_174674


namespace rick_division_steps_l174_174593

theorem rick_division_steps (initial_books : ℕ) (final_books : ℕ) 
  (h_initial : initial_books = 400) (h_final : final_books = 25) : 
  (∀ n : ℕ, (initial_books / (2^n) = final_books) → n = 4) :=
by
  sorry

end rick_division_steps_l174_174593


namespace binom_10_3_eq_120_l174_174317

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l174_174317


namespace inequality_proof_l174_174525

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧
  (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 :=
sorry

end inequality_proof_l174_174525


namespace graph_not_pass_second_quadrant_l174_174105

theorem graph_not_pass_second_quadrant (a b : ℝ) (h1 : a > 1) (h2 : b < -1) :
  ¬ ∃ (x : ℝ), y = a^x + b ∧ x < 0 ∧ y > 0 :=
by
  sorry

end graph_not_pass_second_quadrant_l174_174105


namespace find_a_tangent_slope_at_point_l174_174543

theorem find_a_tangent_slope_at_point :
  ∃ (a : ℝ), (∃ (y : ℝ), y = (fun (x : ℝ) => x^4 + a * x^2 + 1) (-1) ∧ (∃ (y' : ℝ), y' = (fun (x : ℝ) => 4 * x^3 + 2 * a * x) (-1) ∧ y' = 8)) ∧ a = -6 :=
by
  -- Used to skip the proof
  sorry

end find_a_tangent_slope_at_point_l174_174543


namespace find_f3_l174_174940

theorem find_f3 (f : ℝ → ℝ) (h1 : ∀ x y : ℝ, x * f y = y * f x) (h2 : f 15 = 20) : f 3 = 4 := 
  sorry

end find_f3_l174_174940


namespace number_of_ways_to_break_targets_l174_174408

theorem number_of_ways_to_break_targets :
  let target_seq : Multiset.Char := multiset.of_list ['X', 'X', 'X', 'Y', 'Y', 'Y', 'Z', 'Z'] in
  (target_seq.powerset_len 8).card / (multiset.of_list ['X', 'X', 'X']).card! / (multiset.of_list ['Y', 'Y', 'Y']).card! / (multiset.of_list ['Z', 'Z']).card! = 560 := 
by
  let target_seq := ['X', 'X', 'X', 'Y', 'Y', 'Y', 'Z', 'Z'];
  have h_mult_seq : multiset.of_list target_seq = multiset.of_list ['X', 'X', 'X'] + multiset.of_list ['Y', 'Y', 'Y'] + multiset.of_list ['Z', 'Z'] :=
    by rw [multiset.of_list, multiset.of_list, multiset.of_list, multiset.of_list, multiset.of_list, multiset.of_list]; refl;
  rw [h_mult_seq];
  sorry

end number_of_ways_to_break_targets_l174_174408


namespace binomial_10_3_l174_174256

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end binomial_10_3_l174_174256


namespace round_trip_in_first_trip_l174_174116

def percentage_rt_trip_first_trip := 0.3 -- 30%
def percentage_2t_trip_second_trip := 0.6 -- 60%
def percentage_ow_trip_third_trip := 0.45 -- 45%

theorem round_trip_in_first_trip (P1 P2 P3: ℝ) (C1 C2 C3: ℝ) 
  (h1 : P1 = 0.3) 
  (h2 : 0 < P1 ∧ P1 < 1) 
  (h3 : P2 = 0.6) 
  (h4 : 0 < P2 ∧ P2 < 1) 
  (h5 : P3 = 0.45) 
  (h6 : 0 < P3 ∧ P3 < 1) 
  (h7 : C1 + C2 + C3 = 1) 
  (h8 : (C1 = (1 - P1) * 0.15)) 
  (h9 : C2 = 0.2 * P2) 
  (h10 : C3 = 0.1 * P3) :
  P1 = 0.3 := by
  sorry

end round_trip_in_first_trip_l174_174116


namespace find_rate_of_current_l174_174945

noncomputable def rate_of_current : ℝ := 
  let speed_still_water := 42
  let distance_downstream := 33.733333333333334
  let time_hours := 44 / 60
  (distance_downstream / time_hours) - speed_still_water

theorem find_rate_of_current : rate_of_current = 4 :=
by sorry

end find_rate_of_current_l174_174945


namespace m_greater_than_p_l174_174423

theorem m_greater_than_p (p m n : ℕ) (prime_p : Prime p) (pos_m : 0 < m) (pos_n : 0 < n)
    (eq : p^2 + m^2 = n^2) : m > p := 
by 
  sorry

end m_greater_than_p_l174_174423


namespace binom_10_3_eq_120_l174_174319

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l174_174319


namespace binomial_10_3_l174_174255

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end binomial_10_3_l174_174255


namespace park_shape_l174_174641

def cost_of_fencing (side_count : ℕ) (side_cost : ℕ) := side_count * side_cost

theorem park_shape (total_cost : ℕ) (side_cost : ℕ) (h_total : total_cost = 224) (h_side : side_cost = 56) : 
  (∃ sides : ℕ, sides = total_cost / side_cost ∧ sides = 4) ∧ (∀ (sides : ℕ),  cost_of_fencing sides side_cost = total_cost → sides = 4 → sides = 4 ∧ (∀ (x y z w : ℕ), x = y → y = z → z = w → w = x)) :=
by
  sorry

end park_shape_l174_174641


namespace binom_10_3_l174_174233

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end binom_10_3_l174_174233


namespace math_problem_l174_174752

noncomputable def a (n : ℕ) : ℚ :=
nat.rec_on n (1/2) (λ n a_n, 1 / (1 - a_n))

theorem math_problem :
  a 2 = 2 ∧ a 3 = -1 ∧ a 4 = 1/2 ∧ a 2010 = -1 ∧ a 2011 = 1/2 ∧ a 2012 = 2 :=
by
  sorry

end math_problem_l174_174752


namespace combination_10_3_eq_120_l174_174240

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end combination_10_3_eq_120_l174_174240


namespace a_minus_2_values_l174_174581

theorem a_minus_2_values (a : ℝ) (h : |a| = 3) : a - 2 = 1 ∨ a - 2 = -5 :=
by {
  -- the theorem states that given the absolute value condition, a - 2 can be 1 or -5
  sorry
}

end a_minus_2_values_l174_174581


namespace factorize_l174_174666

variables (a b x y : ℝ)

theorem factorize : (a * x - b * y)^2 + (a * y + b * x)^2 = (x^2 + y^2) * (a^2 + b^2) :=
by
  sorry

end factorize_l174_174666


namespace baseball_card_decrease_l174_174151

theorem baseball_card_decrease (V₀ : ℝ) (V₁ V₂ : ℝ)
  (h₁: V₁ = V₀ * (1 - 0.20))
  (h₂: V₂ = V₁ * (1 - 0.20)) :
  ((V₀ - V₂) / V₀) * 100 = 36 :=
by
  sorry

end baseball_card_decrease_l174_174151


namespace combination_10_3_l174_174288

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end combination_10_3_l174_174288


namespace inequality_ge_one_l174_174802

theorem inequality_ge_one {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_ge_one_l174_174802


namespace inequality_holds_l174_174817

theorem inequality_holds (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by
  sorry

end inequality_holds_l174_174817


namespace three_digit_number_108_l174_174065

theorem three_digit_number_108 (a b c : ℕ) (ha : a ≠ 0) (h₀ : a < 10) (h₁ : b < 10) (h₂ : c < 10) (h₃: 100*a + 10*b + c = 12*(a + b + c)) : 
  100*a + 10*b + c = 108 := 
by 
  sorry

end three_digit_number_108_l174_174065


namespace yellow_balls_count_l174_174642

theorem yellow_balls_count {R B Y G : ℕ} 
  (h1 : R + B + Y + G = 531)
  (h2 : R + B = Y + G + 31)
  (h3 : Y = G + 22) : 
  Y = 136 :=
by
  -- The proof is skipped, as requested.
  sorry

end yellow_balls_count_l174_174642


namespace total_musicians_is_98_l174_174459

-- Define the number of males and females in the orchestra
def males_in_orchestra : ℕ := 11
def females_in_orchestra : ℕ := 12

-- Define the total number of musicians in the orchestra
def total_in_orchestra : ℕ := males_in_orchestra + females_in_orchestra

-- Define the number of musicians in the band as twice the number in the orchestra
def total_in_band : ℕ := 2 * total_in_orchestra

-- Define the number of males and females in the choir
def males_in_choir : ℕ := 12
def females_in_choir : ℕ := 17

-- Define the total number of musicians in the choir
def total_in_choir : ℕ := males_in_choir + females_in_choir

-- Prove that the total number of musicians in the orchestra, band, and choir is 98
theorem total_musicians_is_98 : total_in_orchestra + total_in_band + total_in_choir = 98 :=
by {
  -- Adding placeholders for the proof steps
  sorry
}

end total_musicians_is_98_l174_174459


namespace total_children_on_playground_l174_174475

theorem total_children_on_playground
  (boys : ℕ) (girls : ℕ)
  (h_boys : boys = 44) (h_girls : girls = 53) :
  boys + girls = 97 :=
by 
  -- Proof omitted
  sorry

end total_children_on_playground_l174_174475


namespace solve_inequality_l174_174613

theorem solve_inequality : { x : ℝ | 3 * x^2 - 1 > 13 - 5 * x } = { x : ℝ | x < -7 ∨ x > 2 } :=
by
  sorry

end solve_inequality_l174_174613


namespace binomial_10_3_l174_174302

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end binomial_10_3_l174_174302


namespace sin_cos_cos_sin_unique_pair_exists_uniq_l174_174439

noncomputable def theta (x : ℝ) : ℝ := Real.sin (Real.cos x) - x

theorem sin_cos_cos_sin_unique_pair_exists_uniq (h : 0 < c ∧ c < (1/2) * Real.pi ∧ 0 < d ∧ d < (1/2) * Real.pi) :
  (∃! (c d : ℝ), Real.sin (Real.cos c) = c ∧ Real.cos (Real.sin d) = d ∧ c < d) :=
sorry

end sin_cos_cos_sin_unique_pair_exists_uniq_l174_174439


namespace complement_of_A_relative_to_U_l174_174911

-- Define the universal set U and set A
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 4, 5}

-- Define the proof statement for the complement of A with respect to U
theorem complement_of_A_relative_to_U : (U \ A) = {2} := by
  sorry

end complement_of_A_relative_to_U_l174_174911


namespace range_of_h_l174_174069

noncomputable def h : ℝ → ℝ
| x => if x = -7 then 0 else 2 * (x - 3)

theorem range_of_h :
  (Set.range h) = Set.univ \ {-20} :=
sorry

end range_of_h_l174_174069


namespace goose_eggs_at_pond_l174_174733

noncomputable def total_goose_eggs (E : ℝ) : Prop :=
  (5 / 12) * (5 / 16) * (5 / 9) * (3 / 7) * E = 84

theorem goose_eggs_at_pond : 
  ∃ E : ℝ, total_goose_eggs E ∧ E = 678 :=
by
  use 678
  dsimp [total_goose_eggs]
  sorry

end goose_eggs_at_pond_l174_174733


namespace common_measure_of_segments_l174_174372

theorem common_measure_of_segments (a b : ℚ) (h₁ : a = 4 / 15) (h₂ : b = 8 / 21) : 
  (∃ (c : ℚ), c = 1 / 105 ∧ ∃ (n₁ n₂ : ℕ), a = n₁ * c ∧ b = n₂ * c) := 
by {
  sorry
}

end common_measure_of_segments_l174_174372


namespace sqrt_abc_abc_sum_eq_231_l174_174106

-- Defining the conditions
variables (a b c : ℝ) -- a, b, c are real numbers
variable h1 : b + c = 16
variable h2 : c + a = 18
variable h3 : a + b = 20

-- The statement to prove
theorem sqrt_abc_abc_sum_eq_231
(h1 : b + c = 16)
(h2 : c + a = 18)
(h3 : a + b = 20)
: sqrt (a * b * c * (a + b + c)) = 231 :=
sorry -- Proof to be filled in

end sqrt_abc_abc_sum_eq_231_l174_174106


namespace binomial_10_3_eq_120_l174_174349

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l174_174349


namespace transportation_tax_correct_l174_174503

def engine_power : ℕ := 250
def tax_rate : ℕ := 75
def months_owned : ℕ := 2
def total_months_in_year : ℕ := 12

def annual_tax : ℕ := engine_power * tax_rate
def adjusted_tax : ℕ := (annual_tax * months_owned) / total_months_in_year

theorem transportation_tax_correct :
  adjusted_tax = 3125 := by
  sorry

end transportation_tax_correct_l174_174503


namespace min_sum_of_segments_is_305_l174_174750

noncomputable def min_sum_of_segments : ℕ := 
  let a : ℕ := 3
  let b : ℕ := 5
  100 * a + b

theorem min_sum_of_segments_is_305 : min_sum_of_segments = 305 := by
  sorry

end min_sum_of_segments_is_305_l174_174750


namespace students_liking_both_l174_174707

theorem students_liking_both (total_students sports_enthusiasts music_enthusiasts neither : ℕ)
  (h1 : total_students = 55)
  (h2: sports_enthusiasts = 43)
  (h3: music_enthusiasts = 34)
  (h4: neither = 4) : 
  ∃ x, ((sports_enthusiasts - x) + x + (music_enthusiasts - x) = total_students - neither) ∧ (x = 22) :=
by
  sorry -- Proof omitted

end students_liking_both_l174_174707


namespace sunglasses_cost_l174_174839

open Real

def cost_per_pair (selling_price_per_pair : ℝ) (num_pairs_sold : ℝ) (sign_cost : ℝ) : ℝ := 
  (num_pairs_sold * selling_price_per_pair - 2 * sign_cost) / num_pairs_sold

theorem sunglasses_cost (sp : ℝ) (n : ℝ) (sc : ℝ) (H1 : sp = 30) (H2 : n = 10) (H3 : sc = 20) :
  cost_per_pair sp n sc = 26 :=
by
  rw [H1, H2, H3]
  simp [cost_per_pair]
  norm_num
  sorry

end sunglasses_cost_l174_174839


namespace nails_for_smaller_planks_l174_174679

def total_large_planks := 13
def nails_per_plank := 17
def total_nails := 229

def nails_for_large_planks : ℕ :=
  total_large_planks * nails_per_plank

theorem nails_for_smaller_planks :
  total_nails - nails_for_large_planks = 8 :=
by
  -- Proof goes here
  sorry

end nails_for_smaller_planks_l174_174679


namespace combination_10_3_eq_120_l174_174247

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end combination_10_3_eq_120_l174_174247


namespace minimum_value_of_a_plus_2b_l174_174386

theorem minimum_value_of_a_plus_2b 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h : 2 * a + b = a * b - 1) 
  : a + 2 * b = 5 + 2 * Real.sqrt 6 :=
sorry

end minimum_value_of_a_plus_2b_l174_174386


namespace solve_for_y_l174_174596

theorem solve_for_y (y : ℕ) : 8^4 = 2^y → y = 12 :=
by
  sorry

end solve_for_y_l174_174596


namespace girl_walked_distance_l174_174833

-- Define the conditions
def speed : ℝ := 5 -- speed in kmph
def time : ℝ := 6 -- time in hours

-- Define the distance calculation
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

-- The proof statement that we need to show
theorem girl_walked_distance :
  distance speed time = 30 := by
  sorry

end girl_walked_distance_l174_174833


namespace binomial_10_3_l174_174297

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end binomial_10_3_l174_174297


namespace big_eighteen_basketball_games_count_l174_174446

def num_teams_in_division := 6
def num_teams := 18
def games_within_division := 3
def games_between_divisions := 1
def divisions := 3

theorem big_eighteen_basketball_games_count :
  (num_teams * ((num_teams_in_division - 1) * games_within_division + (num_teams - num_teams_in_division) * games_between_divisions)) / 2 = 243 :=
by
  have teams_in_other_divisions : num_teams - num_teams_in_division = 12 := rfl
  have games_per_team_within_division : (num_teams_in_division - 1) * games_within_division = 15 := rfl
  have games_per_team_between_division : 12 * games_between_divisions = 12 := rfl
  sorry

end big_eighteen_basketball_games_count_l174_174446


namespace square_tiles_count_l174_174158

theorem square_tiles_count (p s : ℕ) (h1 : p + s = 30) (h2 : 5 * p + 4 * s = 122) : s = 28 :=
sorry

end square_tiles_count_l174_174158


namespace binom_10_3_l174_174324

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end binom_10_3_l174_174324


namespace percent_decrease_correct_l174_174052

def original_price_per_pack : ℚ := 7 / 3
def promotional_price_per_pack : ℚ := 8 / 4
def percent_decrease_in_price (old_price new_price : ℚ) : ℚ := 
  ((old_price - new_price) / old_price) * 100

theorem percent_decrease_correct :
  percent_decrease_in_price original_price_per_pack promotional_price_per_pack = 14 := by
  sorry

end percent_decrease_correct_l174_174052


namespace nested_composition_l174_174724

def g (x : ℝ) : ℝ := x^2 - 3 * x + 2

theorem nested_composition : g (g (g (g (g (g 2))))) = 2 := by
  sorry

end nested_composition_l174_174724


namespace cube_volume_l174_174026

theorem cube_volume (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
sorry

end cube_volume_l174_174026


namespace inequality_inequality_holds_l174_174785

theorem inequality_inequality_holds (x y z : ℝ) : 
  (x^3 / (x^3 + 2 * y^2 * z)) + (y^3 / (y^3 + 2 * z^2 * x)) + (z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_inequality_holds_l174_174785


namespace find_trajectory_l174_174488

noncomputable def trajectory_of_center (r : ℝ) (r_pos : r > 0) : Prop :=
  let O1 := (-3, 0) in
  let O2 := (3, 0) in
  ∀ (M : ℝ × ℝ),
    dist M O1 = 1 + r →
    dist M O2 = 9 - r →
    (dist M O1 + dist M O2 = 10) →
    ((M.fst^2) / 25 + (M.snd^2) / 16 = 1)

theorem find_trajectory :
  trajectory_of_center r r_pos := sorry

end find_trajectory_l174_174488


namespace revenue_fell_by_percentage_l174_174447

theorem revenue_fell_by_percentage :
  let old_revenue : ℝ := 69.0
  let new_revenue : ℝ := 52.0
  let percentage_decrease : ℝ := ((old_revenue - new_revenue) / old_revenue) * 100
  abs (percentage_decrease - 24.64) < 1e-2 :=
by
  sorry

end revenue_fell_by_percentage_l174_174447


namespace binomial_10_3_l174_174305

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end binomial_10_3_l174_174305


namespace no_roots_less_than_x0_l174_174607

theorem no_roots_less_than_x0
  (x₀ a b c d : ℝ)
  (h₁ : ∀ x ≥ x₀, x^2 + a * x + b > 0)
  (h₂ : ∀ x ≥ x₀, x^2 + c * x + d > 0) :
  ∀ x ≥ x₀, x^2 + ((a + c) / 2) * x + ((b + d) / 2) > 0 := 
by
  sorry

end no_roots_less_than_x0_l174_174607


namespace unknown_number_lcm_hcf_l174_174941

theorem unknown_number_lcm_hcf (a b : ℕ) 
  (lcm_ab : Nat.lcm a b = 192) 
  (hcf_ab : Nat.gcd a b = 16) 
  (known_number : a = 64) :
  b = 48 :=
by
  sorry -- Proof is omitted as per instruction

end unknown_number_lcm_hcf_l174_174941


namespace problem1_problem2_problem3_problem4_l174_174912

def R : Set ℝ := Set.univ
def A : Set ℝ := {x | 1 < x ∧ x < 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 6}

theorem problem1 : A ∩ B = {x | 3 ≤ x ∧ x < 5} := sorry

theorem problem2 : A ∪ B = {x | 1 < x ∧ x ≤ 6} := sorry

theorem problem3 : (Set.compl A) ∩ B = {x | 5 ≤ x ∧ x ≤ 6} :=
sorry

theorem problem4 : Set.compl (A ∩ B) = {x | x < 3 ∨ x ≥ 5} := sorry

end problem1_problem2_problem3_problem4_l174_174912


namespace binom_10_3_l174_174231

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end binom_10_3_l174_174231


namespace domain_of_function_l174_174448

-- Definitions of the conditions
def condition1 (x : ℝ) : Prop := x - 5 ≠ 0
def condition2 (x : ℝ) : Prop := x - 2 > 0

-- The theorem stating the domain of the function
theorem domain_of_function (x : ℝ) : condition1 x ∧ condition2 x ↔ 2 < x ∧ x ≠ 5 :=
by
  sorry

end domain_of_function_l174_174448


namespace find_x_squared_plus_inverse_squared_l174_174390

theorem find_x_squared_plus_inverse_squared (x : ℝ) (h : x^2 - 3 * x + 1 = 0) : x^2 + (1 / x)^2 = 7 :=
by
  sorry

end find_x_squared_plus_inverse_squared_l174_174390


namespace binomial_10_3_eq_120_l174_174209

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l174_174209


namespace value_of_each_baseball_card_l174_174579

theorem value_of_each_baseball_card (x : ℝ) (h : 2 * x + 3 = 15) : x = 6 := by
  sorry

end value_of_each_baseball_card_l174_174579


namespace meaningful_fraction_l174_174884

theorem meaningful_fraction (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 :=
by sorry

end meaningful_fraction_l174_174884


namespace range_of_f_l174_174862

noncomputable def g (x : ℝ) := 15 - 2 * Real.cos (2 * x) - 4 * Real.sin x

noncomputable def f (x : ℝ) := Real.sqrt (g x ^ 2 - 245)

theorem range_of_f : (Set.range f) = Set.Icc 0 14 := sorry

end range_of_f_l174_174862


namespace roots_equivalence_l174_174722

open Polynomial

variables {p q α β γ δ : ℝ}

-- Conditions:
-- (1) α and β are roots of x^2 + px - 2 = 0
-- (2) γ and δ are roots of x^2 + qx - 2 = 0
theorem roots_equivalence 
  (h1 : (X^2 + C p * X - C 2).is_roots α β) 
  (h2 : (X^2 + C q * X - C 2).is_roots γ δ) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = -2 * (q^2 - p^2) :=
sorry

end roots_equivalence_l174_174722


namespace binom_10_3_l174_174225

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end binom_10_3_l174_174225


namespace volunteer_arrangements_l174_174943

theorem volunteer_arrangements (students : Fin 5 → String) (events : Fin 3 → String)
  (A : String) (high_jump : String)
  (h : ∀ (arrange : Fin 3 → Fin 5), ¬(students (arrange 0) = A ∧ events 0 = high_jump)) :
  ∃! valid_arrangements, valid_arrangements = 48 :=
by
  sorry

end volunteer_arrangements_l174_174943


namespace temperature_on_friday_l174_174130

def temperatures (M T W Th F : ℝ) : Prop :=
  (M + T + W + Th) / 4 = 48 ∧
  (T + W + Th + F) / 4 = 40 ∧
  M = 42

theorem temperature_on_friday (M T W Th F : ℝ) (h : temperatures M T W Th F) : 
  F = 10 :=
  by
    -- problem statement
    sorry

end temperature_on_friday_l174_174130


namespace binom_10_3_eq_120_l174_174309

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l174_174309


namespace inequality_abc_l174_174923

variable (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
variable (cond : a + b + c = (1/a) + (1/b) + (1/c))

theorem inequality_abc : a + b + c ≥ 3 / (a * b * c) :=
by
  sorry

end inequality_abc_l174_174923


namespace binomial_10_3_l174_174265

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end binomial_10_3_l174_174265


namespace minimum_15_equal_differences_l174_174723

-- Definition of distinct integers a_i
def distinct_sequence (a : Fin 100 → ℕ) : Prop :=
  ∀ i j : Fin 100, i < j → a i < a j

-- Definition of the differences d_i
def differences (a : Fin 100 → ℕ) (d : Fin 99 → ℕ) : Prop :=
  ∀ i : Fin 99, d i = a ⟨i + 1, Nat.lt_of_lt_of_le (Nat.succ_lt_succ i.2) (by norm_num)⟩ - a i

-- Main theorem statement
theorem minimum_15_equal_differences (a : Fin 100 → ℕ) (d : Fin 99 → ℕ) :
  (∀ i : Fin 100, 1 ≤ a i ∧ a i ≤ 400) →
  distinct_sequence a →
  differences a d →
  ∃ t : Finset ℕ, t.card ≥ 15 ∧ ∀ x : ℕ, x ∈ t → (∃ i j : Fin 99, i ≠ j ∧ d i = x ∧ d j = x) :=
sorry

end minimum_15_equal_differences_l174_174723


namespace inequality_proof_l174_174791

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 := 
by 
  sorry

end inequality_proof_l174_174791


namespace angleC_is_36_l174_174913

theorem angleC_is_36 
  (p q r : ℝ)  -- fictitious types for lines, as Lean needs a type here
  (A B C : ℝ)  -- Angles as Real numbers
  (hpq : p = q)  -- Line p is parallel to line q (represented equivalently for Lean)
  (h : A = 1/4 * B)
  (hr : B + C = 180)
  (vert_opposite : C = A) :
  C = 36 := 
by
  sorry

end angleC_is_36_l174_174913


namespace jelly_beans_problem_l174_174118

/-- Mrs. Wonderful's jelly beans problem -/
theorem jelly_beans_problem : ∃ n_girls n_boys : ℕ, 
  (n_boys = n_girls + 2) ∧
  ((n_girls ^ 2) + ((n_girls + 2) ^ 2) = 394) ∧
  (n_girls + n_boys = 28) :=
by
  sorry

end jelly_beans_problem_l174_174118


namespace probability_snow_at_least_once_l174_174377

theorem probability_snow_at_least_once :
  let first_five_days_no_snow := (1 / 2 * (4 / 5) + 1 / 2 * (7 / 10)) ^ 5,
      next_five_days_no_snow := (1 / 2 * (2 / 3) + 1 / 2 * (5 / 6)) ^ 5
  in 1 - (first_five_days_no_snow * next_five_days_no_snow) = 58806 / 59049 :=
by
  sorry

end probability_snow_at_least_once_l174_174377


namespace quadratic_unbounded_above_l174_174518

theorem quadratic_unbounded_above : ∀ (x y : ℝ), ∃ M : ℝ, ∀ z : ℝ, M < (2 * x^2 + 4 * x * y + 5 * y^2 + 8 * x - 6 * y + z) :=
by
  intro x y
  use 1000 -- Example to denote that for any point greater than 1000
  intro z
  have h1 : 2 * x^2 + 4 * x * y + 5 * y^2 + 8 * x - 6 * y + z ≥ 2 * 0^2 + 4 * 0 * y + 5 * y^2 + 8 * 0 - 6 * y + z := by sorry
  sorry

end quadratic_unbounded_above_l174_174518


namespace boy_travel_speed_l174_174976

theorem boy_travel_speed 
  (v : ℝ)
  (travel_distance : ℝ := 10) 
  (return_speed : ℝ := 2) 
  (total_time : ℝ := 5.8)
  (distance : ℝ := 9.999999999999998) :
  (v = 12.5) → (travel_distance = distance) →
  (total_time = (travel_distance / v) + (travel_distance / return_speed)) :=
by
  sorry

end boy_travel_speed_l174_174976


namespace combination_10_3_l174_174289

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end combination_10_3_l174_174289


namespace hakimi_age_is_40_l174_174455

variable (H : ℕ)
variable (Jared_age : ℕ) (Molly_age : ℕ := 30)
variable (total_age : ℕ := 120)

theorem hakimi_age_is_40 (h1 : Jared_age = H + 10) (h2 : H + Jared_age + Molly_age = total_age) : H = 40 :=
by
  sorry

end hakimi_age_is_40_l174_174455


namespace minimize_shelves_books_l174_174484

theorem minimize_shelves_books : 
  ∀ (n : ℕ),
    (n > 0 ∧ 130 % n = 0 ∧ 195 % n = 0) → 
    (n ≤ 65) := sorry

end minimize_shelves_books_l174_174484


namespace binomial_10_3_l174_174266

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end binomial_10_3_l174_174266


namespace longest_side_triangle_l174_174609

theorem longest_side_triangle (x : ℝ) 
  (h1 : 7 + (x + 4) + (2 * x + 1) = 36) : 
  max 7 (max (x + 4) (2 * x + 1)) = 17 :=
by sorry

end longest_side_triangle_l174_174609


namespace weighted_average_correct_l174_174059

-- Define the marks and credits for each subject
def marks_english := 90
def marks_mathematics := 92
def marks_physics := 85
def marks_chemistry := 87
def marks_biology := 85

def credits_english := 3
def credits_mathematics := 4
def credits_physics := 4
def credits_chemistry := 3
def credits_biology := 2

-- Define the weighted sum and total credits
def weighted_sum := marks_english * credits_english + marks_mathematics * credits_mathematics + marks_physics * credits_physics + marks_chemistry * credits_chemistry + marks_biology * credits_biology
def total_credits := credits_english + credits_mathematics + credits_physics + credits_chemistry + credits_biology

-- Prove that the weighted average is 88.0625
theorem weighted_average_correct : (weighted_sum.toFloat / total_credits.toFloat) = 88.0625 :=
by 
  sorry

end weighted_average_correct_l174_174059


namespace rover_can_explore_planet_l174_174485

noncomputable def equatorial_length : ℝ := 400
noncomputable def total_path_length : ℝ := 600
noncomputable def max_distance_from_path : ℝ := 50

def can_fully_explore_planet (equatorial_length : ℝ) (max_distance_from_path : ℝ) (total_path_length : ℝ) : Prop :=
  ∀ (p : ℝ × ℝ × ℝ), ∃ (q : ℝ × ℝ × ℝ), 
    (dist p q < max_distance_from_path) ∧ 
    (total_distance_traveled ≤ total_path_length)

-- Now we state the theorem to be proven:
theorem rover_can_explore_planet :
  can_fully_explore_planet equatorial_length max_distance_from_path total_path_length :=
sorry

end rover_can_explore_planet_l174_174485


namespace combination_10_3_eq_120_l174_174357

open Nat

theorem combination_10_3_eq_120 : (10.choose 3) = 120 := 
by
  sorry

end combination_10_3_eq_120_l174_174357


namespace find_number_l174_174835

theorem find_number (x : ℤ) (h : 3 * (2 * x + 15) = 75) : x = 5 :=
by
  sorry

end find_number_l174_174835


namespace rectangle_ratio_l174_174535

theorem rectangle_ratio 
  (s : ℝ) -- side length of the inner square
  (x y : ℝ) -- longer side and shorter side of the rectangle
  (h_inner_area : s^2 = (inner_square_area : ℝ))
  (h_outer_area : 9 * inner_square_area = outer_square_area)
  (h_outer_side_eq : (s + 2 * y)^2 = outer_square_area)
  (h_longer_side_eq : x + y = 3 * s) :
  x / y = 2 :=
by sorry

end rectangle_ratio_l174_174535


namespace rick_group_division_l174_174589

theorem rick_group_division :
  ∀ (total_books : ℕ), total_books = 400 → 
  (∃ n : ℕ, (∀ (books_per_category : ℕ) (divisions : ℕ), books_per_category = total_books / (2 ^ divisions) → books_per_category = 25 → divisions = n) ∧ n = 4) :=
by
  sorry

end rick_group_division_l174_174589


namespace cube_volume_from_surface_area_l174_174010

theorem cube_volume_from_surface_area (A : ℕ) (h1 : A = 864) : 
  ∃ V : ℕ, V = 1728 :=
by
  sorry

end cube_volume_from_surface_area_l174_174010


namespace alloy_ratio_proof_l174_174156

def ratio_lead_to_tin_in_alloy_a (x y : ℝ) (ha : 0 < x) (hb : 0 < y) : Prop :=
  let weight_tin_in_a := (y / (x + y)) * 170
  let weight_tin_in_b := (3 / 8) * 250
  let total_tin := weight_tin_in_a + weight_tin_in_b
  total_tin = 221.25

theorem alloy_ratio_proof (x y : ℝ) (ha : 0 < x) (hb : 0 < y) (hc : ratio_lead_to_tin_in_alloy_a x y ha hb) : y / x = 3 :=
by
  -- Proof is omitted
  sorry

end alloy_ratio_proof_l174_174156


namespace initial_investment_calculation_l174_174165

theorem initial_investment_calculation
  (x : ℝ)  -- initial investment at 5% per annum
  (h₁ : x * 0.05 + 4000 * 0.08 = (x + 4000) * 0.06) :
  x = 8000 :=
by
  -- skip the proof
  sorry

end initial_investment_calculation_l174_174165


namespace inequality_holds_l174_174813

theorem inequality_holds (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by
  sorry

end inequality_holds_l174_174813


namespace quadrilateral_false_statement_l174_174029

-- Definitions for quadrilateral properties
def is_rhombus (q : ℝ × ℝ × ℝ × ℝ) : Prop := q.1 = q.2 ∧ q.3 = q.4
def equal_diagonals (d1 d2 : ℝ) : Prop := d1 = d2
def is_rectangle (q : ℝ × ℝ × ℝ × ℝ) : Prop := q.1 = q.2 ∧ q.3 = q.4 ∧ q.1 = 90 ∧ q.3 = 90
def perpendicular (a b : ℝ) : Prop := a * b = 0
def is_parallelogram (q : ℝ × ℝ × ℝ × ℝ) : Prop := q.1 = q.3 ∧ q.2 = q.4
def bisects (d1 d2 : ℝ) : Prop := d1 = d2 / 2

-- The problem statement
theorem quadrilateral_false_statement :
  ¬ (∀ (q : ℝ × ℝ × ℝ × ℝ) (d1 d2 : ℝ),
    (is_rhombus q ∧ equal_diagonals d1 d2 → q.1 = 90 ∧ q.2 = 90) ∧
    (is_rectangle q ∧ perpendicular d1 d2 → q.1 = q.2) ∧
    (is_parallelogram q ∧ perpendicular d1 d2 ∧ equal_diagonals d1 d2 → q.1 = 90 ∧ q.2 = 90) ∧
    (perpendicular d1 d2 ∧ bisects d1 d2 → q.1 = 90 ∧ q.2 = 90)) :=
sorry

end quadrilateral_false_statement_l174_174029


namespace part1_part2_l174_174690

open Real

noncomputable def f (x : ℝ) : ℝ := log x

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := log x + a * x^2 - 3 * x

theorem part1 (a : ℝ) (h_tangent : deriv (λ x, g x a) 1 = 0) : a = 1 :=
by 
  have h_deriv : ∀ x, deriv (λ x, log x + a * x^2 - 3 * x) x = (1 / x) + 2 * a * x - 3,
  { intro x, exact deriv_add (deriv_add (deriv_log x) (deriv_const_mul x (a * x) 2)) (deriv_const_mul x (-3) 1) },
  have h1 := h_deriv 1, rw [h_tangent, add_eq_zero_iff] at h1, linarith

theorem part2 (a : ℝ) (h_a : a = 1) :
  (∀ x > 0, deriv (λ x, g x 1) x = (1 / x) + 2 * x - 3) ∧
  (fderiv ℝ (λ x, g x 1) 1).toLinearMap 1 = -2 ∧
  (fderiv ℝ (λ x, g x 1) (1 / 2)).toLinearMap (1 / 2) = -log 2 - 5 / 4 :=
by sorry

end part1_part2_l174_174690


namespace cube_volume_of_surface_area_l174_174015

-- Define the condition: the surface area S is 864 square units
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- The proof problem: Given that the surface area of a cube is 864 square units,
-- prove that the volume of the cube is 1728 cubic units
theorem cube_volume_of_surface_area (S : ℝ) (hS : S = 864) : 
  ∃ V : ℝ, V = 1728 ∧ ∃ s : ℝ, surface_area s = S ∧ V = s^3 :=
by 
  sorry

end cube_volume_of_surface_area_l174_174015


namespace evaluate_F_2_f_3_l174_174555

def f (a : ℕ) : ℕ := a^2 - 2*a
def F (a b : ℕ) : ℕ := b^2 + a*b

theorem evaluate_F_2_f_3 : F 2 (f 3) = 15 := by
  sorry

end evaluate_F_2_f_3_l174_174555


namespace train_length_180_l174_174983

noncomputable def train_length (time_seconds : ℕ) (speed_kmh : ℕ) : ℕ :=
  (speed_kmh * 1000 / 3600) * time_seconds

theorem train_length_180 :
  train_length 6 108 = 180 :=
sorry

end train_length_180_l174_174983


namespace ocean_depth_350_l174_174041

noncomputable def depth_of_ocean (total_height : ℝ) (volume_ratio_above_water : ℝ) : ℝ :=
  let volume_ratio_below_water := 1 - volume_ratio_above_water
  let height_below_water := (volume_ratio_below_water^(1 / 3)) * total_height
  total_height - height_below_water

theorem ocean_depth_350 :
  depth_of_ocean 10000 (1 / 10) = 350 :=
by
  sorry

end ocean_depth_350_l174_174041


namespace rick_group_division_l174_174590

theorem rick_group_division :
  ∀ (total_books : ℕ), total_books = 400 → 
  (∃ n : ℕ, (∀ (books_per_category : ℕ) (divisions : ℕ), books_per_category = total_books / (2 ^ divisions) → books_per_category = 25 → divisions = n) ∧ n = 4) :=
by
  sorry

end rick_group_division_l174_174590


namespace tangency_condition_and_point_l174_174765

variable (a b p q : ℝ)

/-- Condition for the line y = px + q to be tangent to the ellipse b^2 x^2 + a^2 y^2 = a^2 b^2. -/
theorem tangency_condition_and_point
  (h_cond : a^2 * p^2 + b^2 - q^2 = 0)
  : 
  ∃ (x₀ y₀ : ℝ), 
  x₀ = - (a^2 * p) / q ∧
  y₀ = b^2 / q ∧ 
  (b^2 * x₀^2 + a^2 * y₀^2 = a^2 * b^2 ∧ y₀ = p * x₀ + q) :=
sorry

end tangency_condition_and_point_l174_174765


namespace forty_percent_of_thirty_percent_l174_174557

theorem forty_percent_of_thirty_percent (x : ℝ) 
  (h : 0.3 * 0.4 * x = 48) : 0.4 * 0.3 * x = 48 :=
by
  sorry

end forty_percent_of_thirty_percent_l174_174557


namespace combination_10_3_eq_120_l174_174250

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end combination_10_3_eq_120_l174_174250


namespace inequality_proof_l174_174795

theorem inequality_proof (x y z : ℝ) : 
  ( (x^3) / (x^3 + 2 * (y^2) * z) + 
    (y^3) / (y^3 + 2 * (z^2) * x) + 
    (z^3) / (z^3 + 2 * (x^2) * y) ) ≥ 1 := 
by 
  sorry

end inequality_proof_l174_174795


namespace sum_of_fifth_powers_cannot_conclude_sum_of_fourth_powers_l174_174963

-- Definition of conditions
variables {a b c d : ℝ} 

-- First proof statement
theorem sum_of_fifth_powers 
  (h1 : a + b = c + d) 
  (h2 : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 := 
sorry

-- Second proof statement
theorem cannot_conclude_sum_of_fourth_powers 
  (h1 : a + b = c + d) 
  (h2 : a^3 + b^3 = c^3 + d^3) : 
  ¬(a^4 + b^4 = c^4 + d^4) := 
sorry

end sum_of_fifth_powers_cannot_conclude_sum_of_fourth_powers_l174_174963


namespace factorize_polynomial_l174_174856

theorem factorize_polynomial (m x : ℝ) : m * x^2 - 6 * m * x + 9 * m = m * (x - 3) ^ 2 :=
by sorry

end factorize_polynomial_l174_174856


namespace store_discount_l174_174627

theorem store_discount (P : ℝ) :
  let P1 := 0.9 * P
  let P2 := 0.86 * P1
  P2 = 0.774 * P :=
by
  let P1 := 0.9 * P
  let P2 := 0.86 * P1
  sorry

end store_discount_l174_174627


namespace spend_on_rent_and_utilities_l174_174114

variable (P : ℝ) -- The percentage of her income she used to spend on rent and utilities
variable (I : ℝ) -- Her previous monthly income
variable (increase : ℝ) -- Her salary increase
variable (new_percentage : ℝ) -- The new percentage her rent and utilities amount to

noncomputable def initial_conditions : Prop :=
I = 1000 ∧ increase = 600 ∧ new_percentage = 0.25

theorem spend_on_rent_and_utilities (h : initial_conditions I increase new_percentage) :
    (P / 100) * I = 0.25 * (I + increase) → 
    P = 40 :=
by
  sorry

end spend_on_rent_and_utilities_l174_174114


namespace Ben_Cards_Left_l174_174175

theorem Ben_Cards_Left :
  (4 * 10 + 5 * 8 - 58) = 22 :=
by
  sorry

end Ben_Cards_Left_l174_174175


namespace hakimi_age_l174_174456

theorem hakimi_age
  (avg_age : ℕ)
  (num_friends : ℕ)
  (molly_age : ℕ)
  (age_diff : ℕ)
  (total_age := avg_age * num_friends)
  (combined_age := total_age - molly_age)
  (jared_age := age_diff)
  (hakimi_age := combined_age - jared_age)
  (avg_age = 40)
  (num_friends = 3)
  (molly_age = 30)
  (age_diff = 10)
  (combined_age = 90)
  (hakimi_age = 40) : 
  ∃ age : ℕ, age = hakimi_age :=
by
  sorry

end hakimi_age_l174_174456


namespace monotonic_intervals_intersection_points_l174_174910

open Real

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := (1 / 2) * x^2 - a * log x

-- Define the derivative of f
def f_derivative (x : ℝ) (a : ℝ) : ℝ := (x^2 - a) / x

-- Define the function g
def g (x : ℝ) (a : ℝ) : ℝ := x^2 - (a + 1) * x

-- Define the function F
def F (x : ℝ) (a : ℝ) : ℝ := f x a - g x a

-- Prove monotonic intervals of f
theorem monotonic_intervals (a : ℝ) :
  (f_derivative ⟹ 0) ↔ (a ≤ 0 ↔ ∀ x > 0, f_derivative x a > 0) ∧
  (a > 0 ↔ ∀ x ∈ Ioi 0, (x < sqrt a ↔ f_derivative x a < 0) ∧ (x > sqrt a ↔ f_derivative x a > 0)) := sorry

-- Prove number of intersection points between f(x) and g(x)
theorem intersection_points (a : ℝ) (ha : a ≥ 0) :
  ∃! x > 0, F x a = 0 := sorry

end monotonic_intervals_intersection_points_l174_174910


namespace three_term_inequality_l174_174818

theorem three_term_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_abc : a * b * c = 1) :
  (a^3 / (a^3 + 2 * b^2)) + (b^3 / (b^3 + 2 * c^2)) + (c^3 / (c^3 + 2 * a^2)) ≥ 1 :=
by
  sorry

end three_term_inequality_l174_174818


namespace factorial_power_of_two_divisibility_l174_174438

def highestPowerOfTwoDividingFactorial (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n + 1), n / (2^k)

def binaryOnesCount (n : ℕ) : ℕ :=
  n.foldl (λ acc b, acc + if b then 1 else 0) 0

theorem factorial_power_of_two_divisibility (n : ℕ) :
  (n! % 2^(n - 1) = 0) ↔ (∃ k : ℕ, n = 2^k) :=
begin
  sorry
end

end factorial_power_of_two_divisibility_l174_174438


namespace probability_odd_product_not_even_l174_174617

def isOdd (n : ℕ) : Prop := n % 2 = 1

def diceFaces : Finset ℕ := {1, 2, 3, 4, 5, 6}

def possibleOutcomes : Finset (ℕ × ℕ) := Finset.product diceFaces diceFaces

def oddProductOutcomes : Finset (ℕ × ℕ) := possibleOutcomes.filter (λ p, isOdd (p.1) ∧ isOdd (p.2))

theorem probability_odd_product_not_even : 
  (oddProductOutcomes.card : ℚ) / (possibleOutcomes.card : ℚ) = 1 / 4 := 
by
  sorry

end probability_odd_product_not_even_l174_174617


namespace roots_of_cubic_l174_174574

/-- Let p, q, and r be the roots of the polynomial x^3 - 15x^2 + 10x + 24 = 0. 
   The value of (1 + p)(1 + q)(1 + r) is equal to 2. -/
theorem roots_of_cubic (p q r : ℝ)
  (h1 : p + q + r = 15)
  (h2 : p * q + q * r + r * p = 10)
  (h3 : p * q * r = -24) :
  (1 + p) * (1 + q) * (1 + r) = 2 := 
by 
  sorry

end roots_of_cubic_l174_174574


namespace geometric_progression_common_ratio_l174_174894

theorem geometric_progression_common_ratio (r : ℝ) (a : ℝ) (h_pos : 0 < a)
    (h_geom_prog : ∀ (n : ℕ), a * r^(n-1) = a * r^n + a * r^(n+1) + a * r^(n+2)) :
    r^3 + r^2 + r - 1 = 0 :=
by
  sorry

end geometric_progression_common_ratio_l174_174894


namespace common_ratio_of_geometric_seq_l174_174451

variable {a : ℕ → ℚ} -- The sequence
variable {d : ℚ} -- Common difference

-- Assuming the arithmetic and geometric sequence properties
def is_arithmetic_seq (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def is_geometric_seq (a1 a4 a5 : ℚ) (q : ℚ) : Prop :=
  a4 = a1 * q ∧ a5 = a4 * q

theorem common_ratio_of_geometric_seq (h_arith: is_arithmetic_seq a d) (h_nonzero_d : d ≠ 0)
  (h_geometric: is_geometric_seq (a 1) (a 4) (a 5) (1 / 3)) : (a 4 / a 1) = 1 / 3 :=
by
  sorry

end common_ratio_of_geometric_seq_l174_174451


namespace cube_volume_from_surface_area_l174_174011

theorem cube_volume_from_surface_area (A : ℕ) (h1 : A = 864) : 
  ∃ V : ℕ, V = 1728 :=
by
  sorry

end cube_volume_from_surface_area_l174_174011


namespace percentage_of_hundred_l174_174764

theorem percentage_of_hundred : (30 / 100) * 100 = 30 := 
by
  sorry

end percentage_of_hundred_l174_174764


namespace bruce_can_buy_11_bags_l174_174991

-- Defining the total initial amount
def initial_amount : ℕ := 200

-- Defining the quantities and prices of items
def packs_crayons   : ℕ := 5
def price_crayons   : ℕ := 5
def total_crayons   : ℕ := packs_crayons * price_crayons

def books          : ℕ := 10
def price_books    : ℕ := 5
def total_books    : ℕ := books * price_books

def calculators    : ℕ := 3
def price_calc     : ℕ := 5
def total_calc     : ℕ := calculators * price_calc

-- Total cost of all items
def total_cost : ℕ := total_crayons + total_books + total_calc

-- Calculating the change Bruce will have after buying the items
def change : ℕ := initial_amount - total_cost

-- Cost of each bag
def price_bags : ℕ := 10

-- Number of bags Bruce can buy with the change
def num_bags : ℕ := change / price_bags

-- Proposition stating the main problem
theorem bruce_can_buy_11_bags : num_bags = 11 := by
  sorry

end bruce_can_buy_11_bags_l174_174991


namespace combination_10_3_l174_174287

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end combination_10_3_l174_174287


namespace train_speed_l174_174034

theorem train_speed 
  (length : ℝ)
  (time : ℝ)
  (relative_speed : ℝ)
  (conversion_factor : ℝ)
  (h_length : length = 120)
  (h_time : time = 4)
  (h_relative_speed : relative_speed = 60)
  (h_conversion_factor : conversion_factor = 3.6) :
  (relative_speed / 2) * conversion_factor = 108 :=
by
  sorry

end train_speed_l174_174034


namespace inequality_proof_l174_174801

theorem inequality_proof (x y z : ℝ) : 
  ( (x^3) / (x^3 + 2 * (y^2) * z) + 
    (y^3) / (y^3 + 2 * (z^2) * x) + 
    (z^3) / (z^3 + 2 * (x^2) * y) ) ≥ 1 := 
by 
  sorry

end inequality_proof_l174_174801


namespace minimize_sum_of_squares_l174_174108

theorem minimize_sum_of_squares (x1 x2 x3 : ℝ) (hpos1 : 0 < x1) (hpos2 : 0 < x2) (hpos3 : 0 < x3)
  (h_eq : x1 + 3 * x2 + 5 * x3 = 100) : x1^2 + x2^2 + x3^2 = 2000 / 7 := 
sorry

end minimize_sum_of_squares_l174_174108


namespace equal_striped_areas_l174_174566

theorem equal_striped_areas (A B C D : ℝ) (h_AD_DB : D = A + B) (h_CD2 : C^2 = A * B) :
  (π * C^2 / 4 = π * B^2 / 8 - π * A^2 / 8 - π * D^2 / 8) := 
sorry

end equal_striped_areas_l174_174566


namespace solve_for_x_l174_174125

theorem solve_for_x (x : ℝ) (h : 4 * x + 45 ≠ 0) :
  (8 * x^2 + 80 * x + 4) / (4 * x + 45) = 2 * x + 3 → x = -131 / 22 := 
by 
  sorry

end solve_for_x_l174_174125


namespace binomial_10_3_l174_174303

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end binomial_10_3_l174_174303


namespace petya_wrong_l174_174918

theorem petya_wrong : ∃ (a b : ℕ), b^2 ∣ a^5 ∧ ¬ (b ∣ a^2) :=
by
  use 4
  use 32
  sorry

end petya_wrong_l174_174918


namespace calculate_expression_l174_174850

theorem calculate_expression : sqrt 4 - abs (sqrt 3 - 2) + (-1)^2023 = sqrt 3 - 1 := by
  sorry

end calculate_expression_l174_174850


namespace example_problem_l174_174611

def operation (a b : ℕ) : ℕ := (a + b) * (a - b)

theorem example_problem : 50 - operation 8 5 = 11 := by
  sorry

end example_problem_l174_174611


namespace binom_10_3_eq_120_l174_174193

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l174_174193


namespace corey_candies_l174_174601

-- Definitions based on conditions
variable (T C : ℕ)
variable (totalCandies : T + C = 66)
variable (tapangaExtra : T = C + 8)

-- Theorem to prove Corey has 29 candies
theorem corey_candies : C = 29 :=
by
  sorry

end corey_candies_l174_174601


namespace combination_10_3_eq_120_l174_174359

open Nat

theorem combination_10_3_eq_120 : (10.choose 3) = 120 := 
by
  sorry

end combination_10_3_eq_120_l174_174359


namespace binomial_10_3_l174_174261

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end binomial_10_3_l174_174261


namespace min_value_of_sum_eq_l174_174388

theorem min_value_of_sum_eq : ∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * a + b = a * b - 1 → a + 2 * b = 5 + 2 * Real.sqrt 6 :=
by
  intros a b h
  sorry

end min_value_of_sum_eq_l174_174388


namespace binomial_10_3_l174_174259

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end binomial_10_3_l174_174259


namespace simplify_and_evaluate_l174_174595

-- Define the constants
def a : ℤ := -1
def b : ℤ := 2

-- Declare the expression
def expr : ℤ := 7 * a ^ 2 * b + (-4 * a ^ 2 * b + 5 * a * b ^ 2) - (2 * a ^ 2 * b - 3 * a * b ^ 2)

-- Declare the final evaluated result
def result : ℤ := 2 * ((-1 : ℤ) ^ 2) + 8 * (-1) * (2 : ℤ) ^ 2 

-- The theorem we want to prove
theorem simplify_and_evaluate : expr = result :=
by
  sorry

end simplify_and_evaluate_l174_174595


namespace bruce_can_buy_11_bags_l174_174990

-- Defining the total initial amount
def initial_amount : ℕ := 200

-- Defining the quantities and prices of items
def packs_crayons   : ℕ := 5
def price_crayons   : ℕ := 5
def total_crayons   : ℕ := packs_crayons * price_crayons

def books          : ℕ := 10
def price_books    : ℕ := 5
def total_books    : ℕ := books * price_books

def calculators    : ℕ := 3
def price_calc     : ℕ := 5
def total_calc     : ℕ := calculators * price_calc

-- Total cost of all items
def total_cost : ℕ := total_crayons + total_books + total_calc

-- Calculating the change Bruce will have after buying the items
def change : ℕ := initial_amount - total_cost

-- Cost of each bag
def price_bags : ℕ := 10

-- Number of bags Bruce can buy with the change
def num_bags : ℕ := change / price_bags

-- Proposition stating the main problem
theorem bruce_can_buy_11_bags : num_bags = 11 := by
  sorry

end bruce_can_buy_11_bags_l174_174990


namespace value_of_diamond_l174_174699

def diamond (a b : ℕ) : ℕ := 4 * a + 2 * b

theorem value_of_diamond : diamond 6 3 = 30 :=
by {
  sorry
}

end value_of_diamond_l174_174699


namespace infinite_series_sum_l174_174658

theorem infinite_series_sum :
  (∑' n : ℕ, if n = 0 then 0 else (3^n / (1 + 3^n + 3^(n+1) + 3^(2*n+1)))) = 1 / 4 :=
by
  sorry

end infinite_series_sum_l174_174658


namespace binom_10_3_eq_120_l174_174185

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l174_174185


namespace wall_clock_ring_interval_l174_174883

theorem wall_clock_ring_interval 
  (n : ℕ)                -- Number of rings in a day
  (total_minutes : ℕ)    -- Total minutes in a day
  (intervals : ℕ) :       -- Number of intervals
  n = 6 ∧ total_minutes = 1440 ∧ intervals = n - 1 ∧ intervals = 5
    → (1440 / intervals = 288 ∧ 288 / 60 = 4∧ 288 % 60 = 48) := sorry

end wall_clock_ring_interval_l174_174883


namespace correct_average_l174_174033

theorem correct_average (avg_incorrect : ℕ) (old_num new_num : ℕ) (n : ℕ)
  (h_avg : avg_incorrect = 15)
  (h_old_num : old_num = 26)
  (h_new_num : new_num = 36)
  (h_n : n = 10) :
  (avg_incorrect * n + (new_num - old_num)) / n = 16 := by
  sorry

end correct_average_l174_174033


namespace seating_arrangement_ways_l174_174708

-- Define the problem conditions in Lean 4
def number_of_ways_to_seat (total_chairs : ℕ) (total_people : ℕ) := 
  Nat.factorial total_chairs / Nat.factorial (total_chairs - total_people)

-- Define the specific theorem to be proved
theorem seating_arrangement_ways : number_of_ways_to_seat 8 5 = 6720 :=
by
  sorry

end seating_arrangement_ways_l174_174708


namespace find_fraction_2012th_l174_174939

def sequence_term (p : ℕ) : ℚ :=
  let candidates := {f : ℚ // f.denom ≤ floor (f.num / 2) ∧ f.num < f.denom ∧ 0 < f}
  (candidates.sort (λ x y, (x.denom, x.num) < (y.denom, y.num))).nth (p - 1) 

theorem find_fraction_2012th (m n: ℕ) (hmn : nat.coprime m n) (hp : m < n ∧ 1 ≤ m ∧ 1 < n) (h : sequence_term 2012 = m / n) :
  m + n = 61 :=
  sorry

end find_fraction_2012th_l174_174939


namespace inequality_geq_l174_174922

variable {a b c : ℝ}

theorem inequality_geq (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 1 / a + 1 / b + 1 / c) : 
  a + b + c ≥ 3 / (a * b * c) := 
sorry

end inequality_geq_l174_174922


namespace train_cross_time_l174_174494

def train_length := 100
def bridge_length := 275
def train_speed_kmph := 45

noncomputable def train_speed_mps : ℝ :=
  (train_speed_kmph * 1000.0) / 3600.0

theorem train_cross_time :
  let total_distance := train_length + bridge_length
  let speed := train_speed_mps
  let time := total_distance / speed
  time = 30 :=
by 
  -- Introduce definitions to make sure they align with the initial conditions
  let total_distance := train_length + bridge_length
  let speed := train_speed_mps
  let time := total_distance / speed
  -- Prove time = 30
  sorry

end train_cross_time_l174_174494


namespace binomial_coefficient_10_3_l174_174268

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l174_174268


namespace combination_10_3_eq_120_l174_174239

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end combination_10_3_eq_120_l174_174239


namespace find_a_b_l174_174866

theorem find_a_b (a b x y : ℝ) (h1 : x = 2) (h2 : y = 4) (h3 : a * x + b * y = 16) (h4 : b * x - a * y = -12) : a = 4 ∧ b = 2 := by
  sorry

end find_a_b_l174_174866


namespace vertical_asymptote_l174_174675

theorem vertical_asymptote (x : ℝ) : 4 * x - 9 = 0 → x = 9 / 4 := by
  sorry

end vertical_asymptote_l174_174675


namespace min_value_abs_b_minus_c_l174_174073

-- Define the problem conditions
def condition1 (a b c : ℝ) : Prop :=
  (a - 2 * b - 1)^2 + (a - c - Real.log c)^2 = 0

-- Define the theorem to be proved
theorem min_value_abs_b_minus_c {a b c : ℝ} (h : condition1 a b c) : |b - c| = 1 :=
sorry

end min_value_abs_b_minus_c_l174_174073


namespace books_problem_l174_174577

variable (L W : ℕ) -- L for Li Ming's initial books, W for Wang Hong's initial books

theorem books_problem (h1 : L = W + 26) (h2 : L - 14 = W + 14 - 2) : 14 = 14 :=
by
  sorry

end books_problem_l174_174577


namespace inequality_geq_l174_174919

variable {a b c : ℝ}

theorem inequality_geq (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 1 / a + 1 / b + 1 / c) : 
  a + b + c ≥ 3 / (a * b * c) := 
sorry

end inequality_geq_l174_174919


namespace binom_10_3_l174_174235

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end binom_10_3_l174_174235


namespace buffaloes_added_l174_174974

-- Let B be the daily fodder consumption of one buffalo in units
noncomputable def daily_fodder_buffalo (B : ℝ) := B
noncomputable def daily_fodder_cow (B : ℝ) := (3 / 4) * B
noncomputable def daily_fodder_ox (B : ℝ) := (3 / 2) * B

-- Initial conditions
def initial_buffaloes := 15
def initial_cows := 24
def initial_oxen := 8
def initial_days := 24
noncomputable def total_initial_fodder (B : ℝ) := (initial_buffaloes * daily_fodder_buffalo B) + (initial_oxen * daily_fodder_ox B) + (initial_cows * daily_fodder_cow B)
noncomputable def total_fodder (B : ℝ) := total_initial_fodder B * initial_days

-- New conditions after adding cows and buffaloes
def additional_cows := 60
def new_days := 9
noncomputable def total_new_daily_fodder (B : ℝ) (x : ℝ) := ((initial_buffaloes + x) * daily_fodder_buffalo B) + (initial_oxen * daily_fodder_ox B) + ((initial_cows + additional_cows) * daily_fodder_cow B)

-- Proof statement: Prove that given the conditions, the number of additional buffaloes, x, is 30.
theorem buffaloes_added (B : ℝ) : 
  (total_fodder B = total_new_daily_fodder B 30 * new_days) :=
by sorry

end buffaloes_added_l174_174974


namespace sum_of_all_two_digit_numbers_l174_174182

theorem sum_of_all_two_digit_numbers : 
  let digits := [0, 1, 2, 3, 4, 5]
  let tens_digits := [1, 2, 3, 4, 5]
  let num_ones_digits := digits.length
  let num_tens_digits := tens_digits.length
  let sum_tens_place := 10 * (tens_digits.sum) * num_ones_digits
  let sum_ones_place := (digits.sum) * num_tens_digits
  sum_tens_place + sum_ones_place = 975 :=
by 
  let digits := [0, 1, 2, 3, 4, 5]
  let tens_digits := [1, 2, 3, 4, 5]
  let num_ones_digits := digits.length
  let num_tens_digits := tens_digits.length
  let sum_tens_place := 10 * (tens_digits.sum) * num_ones_digits
  let sum_ones_place := (digits.sum) * num_tens_digits
  show sum_tens_place + sum_ones_place = 975
  sorry

end sum_of_all_two_digit_numbers_l174_174182


namespace egg_hunt_ratio_l174_174565

theorem egg_hunt_ratio :
  ∃ T : ℕ, (3 * T + 30 = 400 ∧ T = 123) ∧ (60 : ℚ) / (T - 20 : ℚ) = 60 / 103 :=
by
  sorry

end egg_hunt_ratio_l174_174565


namespace inequality_proof_l174_174522

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  1 ≤ ((x + y) * (x^3 + y^3)) / (x^2 + y^2)^2 ∧ 
  ((x + y) * (x^3 + y^3)) / (x^2 + y^2)^2 ≤ 9 / 8 :=
sorry

end inequality_proof_l174_174522


namespace inequality_holds_l174_174811

theorem inequality_holds (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by
  sorry

end inequality_holds_l174_174811


namespace range_of_a_l174_174718

theorem range_of_a (a : ℝ) :
  let A := {x | x^2 + 4 * x = 0}
  let B := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0}
  A ∩ B = B → (a = 1 ∨ a ≤ -1) := 
by
  sorry

end range_of_a_l174_174718


namespace area_of_square_A_l174_174741

noncomputable def square_areas (a b : ℕ) : Prop :=
  (b ^ 2 = 81) ∧ (a = b + 4)

theorem area_of_square_A : ∃ a b : ℕ, square_areas a b → a ^ 2 = 169 :=
by
  sorry

end area_of_square_A_l174_174741


namespace pyramid_volume_correct_l174_174508

noncomputable def pyramid_volume (AB BC PO : ℝ) (angle_APB : ℝ) : ℝ :=
  let base_area := AB * BC in
  let height := PO in
  (1 / 3) * base_area * height

theorem pyramid_volume_correct :
  let A := (2 : ℝ)
  let B := (1 : ℝ)
  let angle_APB := (real.pi / 2)  -- 90 degrees in radians
  let height_PO := (real.sqrt 5 / 2)
  pyramid_volume A B height_PO angle_APB = real.sqrt 5 / 3 :=
by
  sorry

end pyramid_volume_correct_l174_174508


namespace points_satisfy_l174_174079

theorem points_satisfy (x y : ℝ) : 
  (y^2 - y = x^2 - x) ↔ (y = x ∨ y = 1 - x) :=
by sorry

end points_satisfy_l174_174079


namespace find_b_perpendicular_l174_174889

theorem find_b_perpendicular (a b : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 1 = 0 ∧ 3 * x + b * y + 5 = 0 → 
  - (a / 2) * - (3 / b) = -1) → b = -3 := 
sorry

end find_b_perpendicular_l174_174889


namespace john_reaching_floor_pushups_l174_174570

-- Definitions based on conditions
def john_train_days_per_week : ℕ := 5
def reps_to_progress : ℕ := 20
def variations : ℕ := 3  -- wall, incline, knee

-- Mathematical statement
theorem john_reaching_floor_pushups : 
  (reps_to_progress * variations) / john_train_days_per_week = 12 := 
by
  sorry

end john_reaching_floor_pushups_l174_174570


namespace julie_initial_savings_l174_174827

theorem julie_initial_savings (S r : ℝ) 
  (h1 : (S / 2) * r * 2 = 120) 
  (h2 : (S / 2) * ((1 + r)^2 - 1) = 124) : 
  S = 1800 := 
sorry

end julie_initial_savings_l174_174827


namespace sonika_initial_deposit_l174_174441

variable (P R : ℝ)

theorem sonika_initial_deposit :
  (P + (P * R * 3) / 100 = 9200) → (P + (P * (R + 2.5) * 3) / 100 = 9800) → P = 8000 :=
by
  intros h1 h2
  sorry

end sonika_initial_deposit_l174_174441


namespace ratio_sum_l174_174127

variable (x y z : ℝ)

-- Conditions
axiom geometric_sequence : 16 * y^2 = 15 * x * z
axiom arithmetic_sequence : 2 / y = 1 / x + 1 / z

-- Theorem to prove
theorem ratio_sum : x ≠ 0 → y ≠ 0 → z ≠ 0 → 
  (16 * y^2 = 15 * x * z) → (2 / y = 1 / x + 1 / z) → (x / z + z / x = 34 / 15) :=
by
  -- proof goes here
  sorry

end ratio_sum_l174_174127


namespace combination_10_3_l174_174285

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end combination_10_3_l174_174285


namespace problem_statement_l174_174109

noncomputable def h (y : ℂ) : ℂ := y^5 - y^3 + 1
noncomputable def p (y : ℂ) : ℂ := y^2 - 3

theorem problem_statement (y_1 y_2 y_3 y_4 y_5 : ℂ) (hroots : ∀ y, h y = 0 ↔ y = y_1 ∨ y = y_2 ∨ y = y_3 ∨ y = y_4 ∨ y = y_5) :
  (p y_1) * (p y_2) * (p y_3) * (p y_4) * (p y_5) = 22 :=
by
  sorry

end problem_statement_l174_174109


namespace first_pipe_fills_cistern_in_10_hours_l174_174952

noncomputable def time_to_fill (x : ℝ) : Prop :=
  let first_pipe_rate := 1 / x
  let second_pipe_rate := 1 / 12
  let third_pipe_rate := 1 / 15
  let combined_rate := first_pipe_rate + second_pipe_rate - third_pipe_rate
  combined_rate = 7 / 60

theorem first_pipe_fills_cistern_in_10_hours : time_to_fill 10 :=
by
  sorry

end first_pipe_fills_cistern_in_10_hours_l174_174952


namespace slope_range_midpoint_coordinates_product_AM_AN_l174_174392

noncomputable def circle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 4)^2 = 4

noncomputable def line_l1 (x y k : ℝ) : Prop :=
  y = k * (x - 1)

noncomputable def line_l2 (x y : ℝ) : Prop :=
  x + 2 * y + 2 = 0

noncomputable def AM (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem slope_range (k : ℝ) : 
  (∃ x y, line_l1 x y k ∧ circle x y) ↔ k > 3/4 := sorry

theorem midpoint_coordinates (k : ℝ) :
  (∃ x y, line_l1 x y k ∧ circle x y) →
  (∃ (xM yM : ℝ), xM = (k^2 + 4*k + 3) / (k^2 + 1) ∧ yM = (4*k^2 + 2*k) / (k^2 + 1)) := sorry

theorem product_AM_AN (k : ℝ) (xM yM xN yN : ℝ) : 
  line_l1 xM yM k ∧ circle xM yM ∧ 
  line_l1 xN yN k ∧ line_l2 xN yN → 
  AM 1 0 xM yM * AM 1 0 xN yN = 6 := sorry 

end slope_range_midpoint_coordinates_product_AM_AN_l174_174392


namespace inequality_proof_l174_174798

theorem inequality_proof (x y z : ℝ) : 
  ( (x^3) / (x^3 + 2 * (y^2) * z) + 
    (y^3) / (y^3 + 2 * (z^2) * x) + 
    (z^3) / (z^3 + 2 * (x^2) * y) ) ≥ 1 := 
by 
  sorry

end inequality_proof_l174_174798


namespace greatest_divisor_of_arithmetic_sequence_sum_l174_174621

theorem greatest_divisor_of_arithmetic_sequence_sum (x c : ℕ) (hx : x > 0) (hc : c > 0) :
  ∃ k, (∀ (S : ℕ), S = 6 * (2 * x + 11 * c) → k ∣ S) ∧ k = 6 :=
by
  sorry

end greatest_divisor_of_arithmetic_sequence_sum_l174_174621


namespace total_cost_of_refueling_l174_174643

theorem total_cost_of_refueling 
  (smaller_tank_capacity : ℤ)
  (larger_tank_capacity : ℤ)
  (num_smaller_planes : ℤ)
  (num_larger_planes : ℤ)
  (fuel_cost_per_liter : ℤ)
  (service_charge_per_plane : ℤ)
  (total_cost : ℤ) :
  smaller_tank_capacity = 60 →
  larger_tank_capacity = 90 →
  num_smaller_planes = 2 →
  num_larger_planes = 2 →
  fuel_cost_per_liter = 50 →
  service_charge_per_plane = 100 →
  total_cost = (num_smaller_planes * smaller_tank_capacity + num_larger_planes * larger_tank_capacity) * (fuel_cost_per_liter / 100) + (num_smaller_planes + num_larger_planes) * service_charge_per_plane →
  total_cost = 550 :=
by
  intros
  sorry

end total_cost_of_refueling_l174_174643


namespace part_a_least_moves_part_b_least_moves_l174_174146

def initial_position : Nat := 0
def total_combinations : Nat := 10^6
def excluded_combinations : List Nat := [0, 10^5, 2 * 10^5, 3 * 10^5, 4 * 10^5, 5 * 10^5, 6 * 10^5, 7 * 10^5, 8 * 10^5, 9 * 10^5]

theorem part_a_least_moves : total_combinations - 1 = 10^6 - 1 := by
  simp [total_combinations, Nat.pow]

theorem part_b_least_moves : total_combinations - excluded_combinations.length = 10^6 - 10 := by
  simp [total_combinations, excluded_combinations, Nat.pow, List.length]

end part_a_least_moves_part_b_least_moves_l174_174146


namespace Brian_watch_animal_videos_l174_174847

theorem Brian_watch_animal_videos :
  let cat_video := 4
  let dog_video := 2 * cat_video
  let gorilla_video := 2 * (cat_video + dog_video)
  let elephant_video := cat_video + dog_video + gorilla_video
  let dolphin_video := cat_video + dog_video + gorilla_video + elephant_video
  let total_time := cat_video + dog_video + gorilla_video + elephant_video + dolphin_video
  total_time = 144 := by
{
  let cat_video := 4
  let dog_video := 2 * cat_video
  let gorilla_video := 2 * (cat_video + dog_video)
  let elephant_video := cat_video + dog_video + gorilla_video
  let dolphin_video := cat_video + dog_video + gorilla_video + elephant_video
  let total_time := cat_video + dog_video + gorilla_video + elephant_video + dolphin_video
  have h1 : total_time = (4 + 8 + 24 + 36 + 72) := sorry
  exact h1
}

end Brian_watch_animal_videos_l174_174847


namespace romance_movie_tickets_l174_174748

-- Define the given conditions.
def horror_movie_tickets := 93
def relationship (R : ℕ) := 3 * R + 18 = horror_movie_tickets

-- The theorem we need to prove
theorem romance_movie_tickets (R : ℕ) (h : relationship R) : R = 25 :=
by sorry

end romance_movie_tickets_l174_174748


namespace minimum_value_of_a_plus_2b_l174_174387

theorem minimum_value_of_a_plus_2b 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h : 2 * a + b = a * b - 1) 
  : a + 2 * b = 5 + 2 * Real.sqrt 6 :=
sorry

end minimum_value_of_a_plus_2b_l174_174387


namespace sector_angle_maximized_l174_174492

noncomputable def central_angle_maximized (r : ℝ) := (20 - 2 * r) / r

noncomputable def sector_area (r : ℝ) (θ : ℝ) := (1 / 2) * r^2 * θ

theorem sector_angle_maximized :
  ∀ r θ : ℝ, 2 * r + r * θ = 20 → 
             θ = central_angle_maximized r → 
             ∃ (area_deriv : ℝ), 
               area_deriv = deriv (λ r, sector_area r (central_angle_maximized r)) r ∧ 
               area_deriv = 0 → 
               θ = 2 :=
by
  intros r θ h1 h2
  sorry

end sector_angle_maximized_l174_174492


namespace temperature_difference_l174_174150

-- Definitions based on the conditions
def refrigeration_compartment_temperature : ℤ := 5
def freezer_compartment_temperature : ℤ := -2

-- Mathematically equivalent proof problem statement
theorem temperature_difference : refrigeration_compartment_temperature - freezer_compartment_temperature = 7 := by
  sorry

end temperature_difference_l174_174150


namespace reconstruct_point_A_l174_174481

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A E' F' G' H' : V)

theorem reconstruct_point_A (E F G H : V) (p q r s : ℝ)
  (hE' : E' = 2 • F - E)
  (hF' : F' = 2 • G - F)
  (hG' : G' = 2 • H - G)
  (hH' : H' = 2 • E - H)
  : p = 1/4 ∧ q = 1/4  ∧ r = 1/4  ∧ s = 1/4  :=
by
  sorry

end reconstruct_point_A_l174_174481


namespace stream_speed_zero_l174_174468

theorem stream_speed_zero (v_c v_s : ℝ)
  (h1 : v_c - v_s - 2 = 9)
  (h2 : v_c + v_s + 1 = 12) :
  v_s = 0 := 
sorry

end stream_speed_zero_l174_174468


namespace child_ticket_cost_l174_174050

theorem child_ticket_cost 
    (total_people : ℕ) 
    (total_money_collected : ℤ) 
    (adult_ticket_price : ℤ) 
    (children_attended : ℕ) 
    (adults_count : ℕ) 
    (total_adult_cost : ℤ) 
    (total_child_cost : ℤ) 
    (c : ℤ)
    (total_people_eq : total_people = 22)
    (total_money_collected_eq : total_money_collected = 50)
    (adult_ticket_price_eq : adult_ticket_price = 8)
    (children_attended_eq : children_attended = 18)
    (adults_count_eq : adults_count = total_people - children_attended)
    (total_adult_cost_eq : total_adult_cost = adults_count * adult_ticket_price)
    (total_child_cost_eq : total_child_cost = children_attended * c)
    (money_collected_eq : total_money_collected = total_adult_cost + total_child_cost) 
  : c = 1 := 
  by
    sorry

end child_ticket_cost_l174_174050


namespace range_of_a_l174_174399

noncomputable def f (x : ℝ) := Real.exp x
noncomputable def g (a x : ℝ) := a * Real.sqrt x
noncomputable def f' (x₀ : ℝ) := Real.exp x₀
noncomputable def g' (a t : ℝ) := a / (2 * Real.sqrt t)

theorem range_of_a (a : ℝ) (x₀ t : ℝ) (hx₀ : x₀ = 1 - t) (ht_pos : t > 0)
  (h1 : f x₀ = Real.exp x₀)
  (h2 : g a t = a * Real.sqrt t)
  (h3 : f x₀ = g' a t)
  (h4 : (Real.exp x₀ - a * Real.sqrt t) / (x₀ - t) = Real.exp x₀) :
    0 < a ∧ a ≤ Real.sqrt (2 * Real.exp 1) :=
sorry

end range_of_a_l174_174399


namespace binomial_10_3_l174_174260

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end binomial_10_3_l174_174260


namespace combination_10_3_eq_120_l174_174361

open Nat

theorem combination_10_3_eq_120 : (10.choose 3) = 120 := 
by
  sorry

end combination_10_3_eq_120_l174_174361


namespace base_number_is_five_l174_174705

variable (a x y : Real)

theorem base_number_is_five (h1 : xy = 1) (h2 : (a ^ (x + y) ^ 2) / (a ^ (x - y) ^ 2) = 625) : a = 5 := 
sorry

end base_number_is_five_l174_174705


namespace find_a_for_max_y_l174_174401

theorem find_a_for_max_y (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ a → 2 * (x - 1)^2 - 3 ≤ 15) →
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ a ∧ 2 * (x - 1)^2 - 3 = 15) →
  a = 4 :=
by sorry

end find_a_for_max_y_l174_174401


namespace find_r_in_geometric_series_l174_174365

theorem find_r_in_geometric_series
  (a r : ℝ)
  (h1 : a / (1 - r) = 15)
  (h2 : a / (1 - r^2) = 6) :
  r = 2 / 3 :=
sorry

end find_r_in_geometric_series_l174_174365


namespace combination_10_3_l174_174290

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end combination_10_3_l174_174290


namespace three_term_inequality_l174_174821

theorem three_term_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_abc : a * b * c = 1) :
  (a^3 / (a^3 + 2 * b^2)) + (b^3 / (b^3 + 2 * c^2)) + (c^3 / (c^3 + 2 * a^2)) ≥ 1 :=
by
  sorry

end three_term_inequality_l174_174821


namespace simplify_expression_l174_174147

theorem simplify_expression (x : ℝ) : 
  3 - 5 * x - 7 * x^2 + 9 + 11 * x - 13 * x^2 - 15 + 17 * x + 19 * x^2 + 2 * x^3 - 4 * x^3 + 6 * x^3 = 
  4 * x^3 - x^2 + 23 * x - 3 :=
by -- proof steps are omitted
  sorry

end simplify_expression_l174_174147


namespace sum_a2_a9_l174_174545

variable {a : ℕ → ℝ} -- Define the sequence a_n
variable {S : ℕ → ℝ} -- Define the sum sequence S_n

-- The conditions
def arithmetic_sum (S : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ) : Prop :=
  S n = (n * (a 1 + a n)) / 2

axiom S_10 : arithmetic_sum S a 10
axiom S_10_value : S 10 = 100

-- The goal
theorem sum_a2_a9 (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : S 10 = 100) (h2 : arithmetic_sum S a 10) :
  a 2 + a 9 = 20 := 
sorry

end sum_a2_a9_l174_174545


namespace log_base_30_of_8_l174_174064

theorem log_base_30_of_8 (a b : ℝ) (h1 : Real.log 5 = a) (h2 : Real.log 3 = b) : 
  Real.logb 30 8 = (3 * (1 - a)) / (1 + b) :=
by
  sorry

end log_base_30_of_8_l174_174064


namespace part_a_part_b_l174_174628

theorem part_a {a b c : ℝ} : ∃ x : ℝ, (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a) = 0 :=
sorry

theorem part_b {a b c : ℝ} : (a + b + c) ^ 2 ≥ 3 * (a * b + b * c + c * a) :=
sorry

end part_a_part_b_l174_174628


namespace binom_10_3_eq_120_l174_174195

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l174_174195


namespace total_spears_is_78_l174_174578

-- Define the spear production rates for each type of wood
def spears_from_sapling := 3
def spears_from_log := 9
def spears_from_bundle := 7
def spears_from_trunk := 15

-- Define the quantity of each type of wood
def saplings := 6
def logs := 1
def bundles := 3
def trunks := 2

-- Prove that the total number of spears is 78
theorem total_spears_is_78 : (saplings * spears_from_sapling) + (logs * spears_from_log) + (bundles * spears_from_bundle) + (trunks * spears_from_trunk) = 78 :=
by 
  -- Calculation can be filled here
  sorry

end total_spears_is_78_l174_174578


namespace combination_10_3_l174_174282

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end combination_10_3_l174_174282


namespace logarithmic_AMGM_inequality_l174_174931

theorem logarithmic_AMGM_inequality (a b c : ℝ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) :
  2 * ((Real.log b / (a * Real.log a)) / (a + b) + 
       (Real.log c / (b * Real.log b)) / (b + c) + 
       (Real.log a / (c * Real.log c)) / (c + a)) 
  ≥ 9 / (a + b + c) := 
sorry

end logarithmic_AMGM_inequality_l174_174931


namespace transport_tax_correct_l174_174505

def engine_power : ℕ := 250
def tax_rate : ℕ := 75
def months_owned : ℕ := 2
def months_in_year : ℕ := 12
def annual_tax : ℕ := engine_power * tax_rate
def adjusted_tax : ℕ := (annual_tax * months_owned) / months_in_year

theorem transport_tax_correct :
  adjusted_tax = 3125 :=
by
  sorry

end transport_tax_correct_l174_174505


namespace range_of_m_l174_174890

theorem range_of_m (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 - m * x - m < 0) ↔ -4 ≤ m ∧ m ≤ 0 :=
by
  sorry

end range_of_m_l174_174890


namespace problem_I_problem_II_l174_174037

-- Problem I statement
theorem problem_I (a b c : ℝ) (h : a + b + c = 1) : (a + 1)^2 + (b + 1)^2 + (c + 1)^2 ≥ 16 / 3 := 
by
  sorry

-- Problem II statement
theorem problem_II (a : ℝ) : 
  (∀ x : ℝ, abs (x - a) + abs (2 * x - 1) ≥ 2) →
  a ∈ Set.Iic (-3/2) ∪ Set.Ici (5/2) :=
by 
  sorry

end problem_I_problem_II_l174_174037


namespace inequality_proof_l174_174787

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 := 
by 
  sorry

end inequality_proof_l174_174787


namespace marge_final_plants_l174_174425

-- Definitions corresponding to the conditions
def seeds_planted := 23
def seeds_never_grew := 5
def plants_grew := seeds_planted - seeds_never_grew
def plants_eaten := plants_grew / 3
def uneaten_plants := plants_grew - plants_eaten
def plants_strangled := uneaten_plants / 3
def survived_plants := uneaten_plants - plants_strangled
def effective_addition := 1

-- The main statement we need to prove
theorem marge_final_plants : 
  (plants_grew - plants_eaten - plants_strangled + effective_addition) = 9 := 
by
  sorry

end marge_final_plants_l174_174425


namespace inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8_l174_174531

theorem inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
    1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 := 
sorry

end inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8_l174_174531


namespace cube_volume_from_surface_area_l174_174004

theorem cube_volume_from_surface_area (S : ℝ) (h : S = 864) : ∃ V : ℝ, V = 1728 :=
by
  -- Assume surface area formula S = 6s^2, solve steps skipped and go directly to conclusion
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  use V
  sorry

end cube_volume_from_surface_area_l174_174004


namespace sector_area_l174_174687

theorem sector_area (r α : ℝ) (h_r : r = 3) (h_α : α = 2) : (1/2 * r^2 * α) = 9 := by
  sorry

end sector_area_l174_174687


namespace largest_stickers_per_page_l174_174843

theorem largest_stickers_per_page :
  Nat.gcd (Nat.gcd 1050 1260) 945 = 105 := 
sorry

end largest_stickers_per_page_l174_174843


namespace day_of_week_306_2003_l174_174405

-- Note: Definitions to support the conditions and the proof
def day_of_week (n : ℕ) : ℕ := n % 7

-- Theorem statement: Given conditions lead to the conclusion that the 306th day of the year 2003 falls on a Sunday
theorem day_of_week_306_2003 :
  (day_of_week (15) = 2) → (day_of_week (306) = 0) :=
by sorry

end day_of_week_306_2003_l174_174405


namespace inequality_proof_l174_174530

theorem inequality_proof (x y : ℝ) (hx: 0 < x) (hy: 0 < y) : 
  1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧ 
  (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 := 
by 
  sorry

end inequality_proof_l174_174530


namespace not_product_of_consecutive_integers_l174_174120

theorem not_product_of_consecutive_integers (n k : ℕ) (hn : n > 0) (hk : k > 0) :
  ∀ (m : ℕ), 2 * (n ^ k) ^ 3 + 4 * (n ^ k) + 10 ≠ m * (m + 1) := by
sorry

end not_product_of_consecutive_integers_l174_174120


namespace find_c_l174_174892

open Real

noncomputable def triangle_side_c (a b c : ℝ) (A B C : ℝ) :=
  A = (π / 4) ∧
  2 * b * sin B - c * sin C = 2 * a * sin A ∧
  (1/2) * b * c * (sqrt 2)/2 = 3 →
  c = 2 * sqrt 2
  
theorem find_c {a b c A B C : ℝ} (h : triangle_side_c a b c A B C) : c = 2 * sqrt 2 :=
sorry

end find_c_l174_174892


namespace salary_percentage_change_l174_174932

theorem salary_percentage_change (S : ℝ) (x : ℝ) :
  (S * (1 - (x / 100)) * (1 + (x / 100)) = S * 0.84) ↔ (x = 40) :=
by
  sorry

end salary_percentage_change_l174_174932


namespace rick_group_division_l174_174588

theorem rick_group_division :
  ∀ (total_books : ℕ), total_books = 400 → 
  (∃ n : ℕ, (∀ (books_per_category : ℕ) (divisions : ℕ), books_per_category = total_books / (2 ^ divisions) → books_per_category = 25 → divisions = n) ∧ n = 4) :=
by
  sorry

end rick_group_division_l174_174588


namespace inequality_abc_l174_174929

variable (a b c : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hc : c > 0)
variable (cond : a + b + c = (1 / a) + (1 / b) + (1 / c))

theorem inequality_abc : a + b + c ≥ 3 / (a * b * c) :=
sorry

end inequality_abc_l174_174929


namespace cube_volume_from_surface_area_l174_174006

theorem cube_volume_from_surface_area (A : ℕ) (h1 : A = 864) : 
  ∃ V : ℕ, V = 1728 :=
by
  sorry

end cube_volume_from_surface_area_l174_174006


namespace binom_10_3_eq_120_l174_174188

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l174_174188


namespace visits_365_days_l174_174661

theorem visits_365_days : 
  let alice_visits := 3
  let beatrix_visits := 4
  let claire_visits := 5
  let total_days := 365
  ∃ days_with_exactly_two_visits, days_with_exactly_two_visits = 54 :=
by
  sorry

end visits_365_days_l174_174661


namespace repeated_application_of_g_on_2_l174_174725

def g (x : ℝ) : ℝ := x^2 - 3 * x + 2

theorem repeated_application_of_g_on_2 :
  g(g(g(g(g(g(2)))))) = 2 :=
by
  sorry

end repeated_application_of_g_on_2_l174_174725


namespace trigonometric_identity_proof_l174_174868

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) (α : ℝ) (β : ℝ) : ℝ :=
  a * Real.sin (π * x + α) + b * Real.cos (π * x + β) + 4

theorem trigonometric_identity_proof
  (a b α β : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0)
  (h : f 2014 a b α β = 5) :
  f 2015 a b α β = 3 :=
by
  sorry

end trigonometric_identity_proof_l174_174868


namespace union_complement_with_B_l174_174424

namespace SetTheory

open Set

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Definition of the complement of A relative to U in Lean
def C_U (A U : Set ℕ) : Set ℕ := U \ A

-- Theorem statement
theorem union_complement_with_B (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 4}) (hA : A = {0, 1, 2, 3}) (hB : B = {2, 3, 4}) : 
  (C_U A U) ∪ B = {2, 3, 4} :=
by
  -- Proof goes here
  sorry

end SetTheory

end union_complement_with_B_l174_174424


namespace intersection_complement_eq_l174_174402

def M : Set ℝ := {-1, 1, 2, 4}
def N : Set ℝ := {x : ℝ | x^2 - 2 * x ≥ 3 }
def N_complement : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

theorem intersection_complement_eq :
  M ∩ N_complement = {1, 2} :=
by
  sorry

end intersection_complement_eq_l174_174402


namespace binom_10_3_l174_174236

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end binom_10_3_l174_174236


namespace binomial_coefficient_10_3_l174_174273

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l174_174273


namespace three_term_inequality_l174_174820

theorem three_term_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_abc : a * b * c = 1) :
  (a^3 / (a^3 + 2 * b^2)) + (b^3 / (b^3 + 2 * c^2)) + (c^3 / (c^3 + 2 * a^2)) ≥ 1 :=
by
  sorry

end three_term_inequality_l174_174820


namespace jack_total_cost_l174_174412

def cost_of_tires (n : ℕ) (price_per_tire : ℕ) : ℕ := n * price_per_tire
def cost_of_window (price_per_window : ℕ) : ℕ := price_per_window

theorem jack_total_cost :
  cost_of_tires 3 250 + cost_of_window 700 = 1450 :=
by
  sorry

end jack_total_cost_l174_174412


namespace inequality_ge_one_l174_174807

theorem inequality_ge_one {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_ge_one_l174_174807


namespace faster_train_passes_slower_in_54_seconds_l174_174618

-- Definitions of the conditions.
def length_of_train := 75 -- Length of each train in meters.
def speed_faster_train := 46 * 1000 / 3600 -- Speed of the faster train in m/s.
def speed_slower_train := 36 * 1000 / 3600 -- Speed of the slower train in m/s.
def relative_speed := speed_faster_train - speed_slower_train -- Relative speed in m/s.
def total_distance := 2 * length_of_train -- Total distance to cover to pass the slower train.

-- The proof statement.
theorem faster_train_passes_slower_in_54_seconds : total_distance / relative_speed = 54 := by
  sorry

end faster_train_passes_slower_in_54_seconds_l174_174618


namespace points_satisfying_inequality_l174_174471

theorem points_satisfying_inequality (x y : ℝ) :
  ( ( (x * y + 1) / (x + y) )^2 < 1) ↔ 
  ( (-1 < x ∧ x < 1) ∧ (y < -1 ∨ y > 1) ) ∨ 
  ( (x < -1 ∨ x > 1) ∧ (-1 < y ∧ y < 1) ) := 
sorry

end points_satisfying_inequality_l174_174471


namespace binomial_10_3_eq_120_l174_174198

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l174_174198


namespace sqrt_eq_pm_4_l174_174614

theorem sqrt_eq_pm_4 : {x : ℝ | x * x = 16} = {4, -4} :=
by sorry

end sqrt_eq_pm_4_l174_174614


namespace smallest_n_ineq_l174_174768

theorem smallest_n_ineq : ∃ n : ℕ, 3 * Real.sqrt n - 2 * Real.sqrt (n - 1) < 0.03 ∧ 
  (∀ m : ℕ, (3 * Real.sqrt m - 2 * Real.sqrt (m - 1) < 0.03) → n ≤ m) ∧ n = 433715589 :=
by
  sorry

end smallest_n_ineq_l174_174768


namespace number_of_pairs_of_shoes_size_40_to_42_200_pairs_l174_174950

theorem number_of_pairs_of_shoes_size_40_to_42_200_pairs 
  (total_pairs_sample : ℕ)
  (freq_3rd_group : ℝ)
  (freq_1st_group : ℕ)
  (freq_2nd_group : ℕ)
  (freq_4th_group : ℕ)
  (total_pairs_200 : ℕ)
  (scaled_pairs_size_40_42 : ℕ)
: total_pairs_sample = 40 ∧ freq_3rd_group = 0.25 ∧ freq_1st_group = 6 ∧ freq_2nd_group = 7 ∧ freq_4th_group = 9 ∧ total_pairs_200 = 200 ∧ scaled_pairs_size_40_42 = 40 :=
sorry

end number_of_pairs_of_shoes_size_40_to_42_200_pairs_l174_174950


namespace binom_10_3_eq_120_l174_174321

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l174_174321


namespace probability_of_drawing_orange_marble_second_l174_174173

noncomputable def probability_second_marble_is_orange (total_A white_A black_A : ℕ) (total_B orange_B green_B blue_B : ℕ) (total_C orange_C green_C blue_C : ℕ) : ℚ := 
  let p_white := (white_A : ℚ) / total_A
  let p_black := (black_A : ℚ) / total_A
  let p_orange_B := (orange_B : ℚ) / total_B
  let p_orange_C := (orange_C : ℚ) / total_C
  (p_white * p_orange_B) + (p_black * p_orange_C)

theorem probability_of_drawing_orange_marble_second :
  probability_second_marble_is_orange 9 4 5 15 7 5 3 10 4 4 2 = 58 / 135 :=
by
  sorry

end probability_of_drawing_orange_marble_second_l174_174173


namespace geometry_problem_l174_174546

noncomputable def vertices_on_hyperbola (A B C : ℝ × ℝ) : Prop :=
  (∃ x1 y1, A = (x1, y1) ∧ 2 * x1^2 - y1^2 = 4) ∧
  (∃ x2 y2, B = (x2, y2) ∧ 2 * x2^2 - y2^2 = 4) ∧
  (∃ x3 y3, C = (x3, y3) ∧ 2 * x3^2 - y3^2 = 4)

noncomputable def midpoints (A B C M N P : ℝ × ℝ) : Prop :=
  (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) ∧
  (N = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)) ∧
  (P = ((C.1 + A.1) / 2, (C.2 + A.2) / 2))

noncomputable def slopes (A B C M N P : ℝ × ℝ) (k1 k2 k3 : ℝ) : Prop :=
  k1 ≠ 0 ∧ k2 ≠ 0 ∧ k3 ≠ 0 ∧
  k1 = M.2 / M.1 ∧ k2 = N.2 / N.1 ∧ k3 = P.2 / P.1

noncomputable def sum_of_slopes (A B C : ℝ × ℝ) (k1 k2 k3 : ℝ) : Prop :=
  ((A.2 - B.2) / (A.1 - B.1) +
   (B.2 - C.2) / (B.1 - C.1) +
   (C.2 - A.2) / (C.1 - A.1)) = -1

theorem geometry_problem 
  (A B C M N P : ℝ × ℝ) (k1 k2 k3 : ℝ) 
  (h1 : vertices_on_hyperbola A B C)
  (h2 : midpoints A B C M N P) 
  (h3 : slopes A B C M N P k1 k2 k3) 
  (h4 : sum_of_slopes A B C k1 k2 k3) :
  1/k1 + 1/k2 + 1/k3 = -1 / 2 :=
sorry

end geometry_problem_l174_174546


namespace part2_l174_174547

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1|

theorem part2 (x y : ℝ) (h₁ : |x - y - 1| ≤ 1 / 3) (h₂ : |2 * y + 1| ≤ 1 / 6) :
  f x < 1 := 
by
  sorry

end part2_l174_174547


namespace range_of_a_l174_174560

theorem range_of_a (a : ℝ) : (∃ x : ℝ, 4^x - (a + 3) * 2^x + 1 = 0) → a ≥ -1 := sorry

end range_of_a_l174_174560


namespace simple_interest_rate_l174_174629

theorem simple_interest_rate (P A T : ℕ) (P_val : P = 750) (A_val : A = 900) (T_val : T = 8) : 
  ∃ (R : ℚ), R = 2.5 :=
by {
  sorry
}

end simple_interest_rate_l174_174629


namespace michael_pets_kangaroos_l174_174112

theorem michael_pets_kangaroos :
  let total_pets := 24
  let fraction_dogs := 1 / 8
  let fraction_not_cows := 3 / 4
  let fraction_not_cats := 2 / 3
  let num_dogs := fraction_dogs * total_pets
  let num_cows := (1 - fraction_not_cows) * total_pets
  let num_cats := (1 - fraction_not_cats) * total_pets
  let num_kangaroos := total_pets - num_dogs - num_cows - num_cats
  num_kangaroos = 7 :=
by
  sorry

end michael_pets_kangaroos_l174_174112


namespace tea_sales_revenue_l174_174476

theorem tea_sales_revenue (x : ℝ) (price_last_year price_this_year : ℝ) (yield_last_year yield_this_year : ℝ) (revenue_last_year revenue_this_year : ℝ) :
  price_this_year = 10 * price_last_year →
  yield_this_year = 198.6 →
  yield_last_year = 198.6 + 87.4 →
  revenue_this_year = 198.6 * price_this_year →
  revenue_last_year = yield_last_year * price_last_year →
  revenue_this_year = revenue_last_year + 8500 →
  revenue_this_year = 9930 := 
by
  sorry

end tea_sales_revenue_l174_174476


namespace cube_volume_from_surface_area_l174_174007

theorem cube_volume_from_surface_area (A : ℕ) (h1 : A = 864) : 
  ∃ V : ℕ, V = 1728 :=
by
  sorry

end cube_volume_from_surface_area_l174_174007


namespace salary_calculation_l174_174137

variable {A B : ℝ}

theorem salary_calculation (h1 : A + B = 6000) (h2 : 0.05 * A = 0.15 * B) : A = 4500 :=
by
  sorry

end salary_calculation_l174_174137


namespace power_identity_l174_174636

theorem power_identity (x : ℝ) : (x ^ 10 = 25 ^ 5) → x = 5 := by
  sorry

end power_identity_l174_174636


namespace probability_condition_l174_174159

def balls : Nat := 12

def drawing_event (draws : List Nat) (target : Nat) : Prop :=
  ∃ ball, draws.filter (λ x, x = ball).length ≥ target

def all_balls_once (draws : List Nat) : Prop :=
  ∀ ball, draws.filter (λ x, x = ball).length > 0

def simulation_result :=
  0.02236412255

theorem probability_condition (h : ∀ draws : List Nat, 
  (drawing_event draws 12 ∨ all_balls_once draws) → drawing_event draws 12) :
  Prob(drawing_event [rand_ball (n+1) | n in range 144] 12) = simulation_result := sorry

end probability_condition_l174_174159


namespace combination_10_3_eq_120_l174_174249

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end combination_10_3_eq_120_l174_174249


namespace part_a_part_b_l174_174971

-- Part (a)
theorem part_a {a b c d : ℝ} (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) :
  a^5 + b^5 = c^5 + d^5 :=
sorry

-- Part (b)
theorem part_b {a b c d : ℝ} (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) :
  ¬(a^4 + b^4 = c^4 + d^4) :=
counter_example

end part_a_part_b_l174_174971


namespace binom_10_3_eq_120_l174_174196

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l174_174196


namespace binom_10_3_l174_174331

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end binom_10_3_l174_174331


namespace graph_function_quadrant_l174_174134

theorem graph_function_quadrant (x y : ℝ): 
  (∀ x : ℝ, y = -x + 2 → (x < 0 → y ≠ -3 + - x)) := 
sorry

end graph_function_quadrant_l174_174134


namespace perpendicular_graphs_solve_a_l174_174135

theorem perpendicular_graphs_solve_a (a : ℝ) : 
  (∀ x y : ℝ, 2 * y + x + 3 = 0 → 3 * y + a * x + 2 = 0 → 
  ∀ m1 m2 : ℝ, (y = m1 * x + b1 → m1 = -1 / 2) →
  (y = m2 * x + b2 → m2 = -a / 3) →
  m1 * m2 = -1) → a = -6 :=
by
  sorry

end perpendicular_graphs_solve_a_l174_174135


namespace combination_10_3_eq_120_l174_174360

open Nat

theorem combination_10_3_eq_120 : (10.choose 3) = 120 := 
by
  sorry

end combination_10_3_eq_120_l174_174360


namespace total_frames_l174_174419

def frames_per_page : ℝ := 143.0

def pages : ℝ := 11.0

theorem total_frames : frames_per_page * pages = 1573.0 :=
by
  sorry

end total_frames_l174_174419


namespace inequality_preservation_l174_174698

theorem inequality_preservation (a b x : ℝ) (h : a > b) : a * 2^x > b * 2^x :=
sorry

end inequality_preservation_l174_174698


namespace sheets_in_height_l174_174980

theorem sheets_in_height (sheets_per_ream : ℕ) (thickness_per_ream : ℝ) (target_thickness : ℝ) 
  (h₀ : sheets_per_ream = 500) (h₁ : thickness_per_ream = 5.0) (h₂ : target_thickness = 7.5) :
  target_thickness / (thickness_per_ream / sheets_per_ream) = 750 :=
by sorry

end sheets_in_height_l174_174980


namespace reduced_population_l174_174828

theorem reduced_population (initial_population : ℕ)
  (percentage_died : ℝ)
  (percentage_left : ℝ)
  (h_initial : initial_population = 8515)
  (h_died : percentage_died = 0.10)
  (h_left : percentage_left = 0.15) :
  ((initial_population - (⌊percentage_died * initial_population⌋₊ : ℕ)) - 
   (⌊percentage_left * (initial_population - (⌊percentage_died * initial_population⌋₊ : ℕ))⌋₊ : ℕ)) = 6515 :=
by
  sorry

end reduced_population_l174_174828


namespace budget_allocations_and_percentage_changes_l174_174161

theorem budget_allocations_and_percentage_changes (X : ℝ) :
  (14 * X / 100, 24 * X / 100, 15 * X / 100, 19 * X / 100, 8 * X / 100, 20 * X / 100) = 
  (0.14 * X, 0.24 * X, 0.15 * X, 0.19 * X, 0.08 * X, 0.20 * X) ∧
  ((14 - 12) / 12 * 100 = 16.67 ∧
   (24 - 22) / 22 * 100 = 9.09 ∧
   (15 - 13) / 13 * 100 = 15.38 ∧
   (19 - 18) / 18 * 100 = 5.56 ∧
   (8 - 7) / 7 * 100 = 14.29 ∧
   ((20 - (100 - (12 + 22 + 13 + 18 + 7))) / (100 - (12 + 22 + 13 + 18 + 7)) * 100) = -28.57) := by
  sorry

end budget_allocations_and_percentage_changes_l174_174161


namespace minimum_ab_value_is_two_l174_174686

noncomputable def minimum_value_ab (a b : ℝ) (h1 : a^2 ≠ 0) (h2 : b ≠ 0)
  (h3 : a^2 * b = a^2 + 1) : ℝ :=
|a * b|

theorem minimum_ab_value_is_two (a b : ℝ) (h1 : a^2 ≠ 0) (h2 : b ≠ 0)
  (h3 : a^2 * b = a^2 + 1) : minimum_value_ab a b h1 h2 h3 = 2 := by
  sorry

end minimum_ab_value_is_two_l174_174686


namespace unique_rational_solution_l174_174036

theorem unique_rational_solution (x y z : ℚ) (h : x^3 + 3*y^3 + 9*z^3 - 9*x*y*z = 0) : x = 0 ∧ y = 0 ∧ z = 0 := 
by {
  sorry
}

end unique_rational_solution_l174_174036


namespace xy_gt_1_necessary_but_not_sufficient_l174_174623

-- To define the conditions and prove the necessary and sufficient conditions.

variable (x y : ℝ)

-- The main statement to prove once conditions are defined.
theorem xy_gt_1_necessary_but_not_sufficient : 
  (x > 1 ∧ y > 1 → x * y > 1) ∧ ¬ (x * y > 1 → x > 1 ∧ y > 1) := 
by 
  sorry

end xy_gt_1_necessary_but_not_sufficient_l174_174623


namespace isabel_weekly_run_distance_l174_174900

theorem isabel_weekly_run_distance
  (circuit_length : ℕ)
  (morning_laps : ℕ)
  (afternoon_laps : ℕ)
  (days_in_week : ℕ)
  : circuit_length = 365 → morning_laps = 7 → afternoon_laps = 3 → days_in_week = 7 →
    (morning_laps * circuit_length + afternoon_laps * circuit_length) * days_in_week = 25550 :=
by
  intros h_circuit h_morning h_afternoon h_days
  rw [h_circuit, h_morning, h_afternoon, h_days]
  have morning_distance : 7 * 365 = 2555 := rfl
  have afternoon_distance : 3 * 365 = 1095 := rfl
  have total_day_distance : 2555 + 1095 = 3650 := rfl
  have week_distance : 3650 * 7 = 25550 := rfl
  exact week_distance

end isabel_weekly_run_distance_l174_174900


namespace binom_10_3_eq_120_l174_174314

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l174_174314


namespace binomial_10_3_eq_120_l174_174340

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l174_174340


namespace matrix_determinant_eq_16_l174_174513

theorem matrix_determinant_eq_16 (x : ℝ) :
  (3 * x) * (4 * x) - (2 * x) = 16 ↔ x = 4 / 3 ∨ x = -1 :=
by sorry

end matrix_determinant_eq_16_l174_174513


namespace inequality_proof_l174_174788

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 := 
by 
  sorry

end inequality_proof_l174_174788


namespace inequality_inequality_holds_l174_174784

theorem inequality_inequality_holds (x y z : ℝ) : 
  (x^3 / (x^3 + 2 * y^2 * z)) + (y^3 / (y^3 + 2 * z^2 * x)) + (z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_inequality_holds_l174_174784


namespace find_expression_for_an_l174_174482

-- Definitions for the problem conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a 1 * q ^ n

def problem_conditions (a : ℕ → ℝ) (q : ℝ) :=
  geometric_sequence a q ∧
  a 1 + a 3 = 10 ∧
  a 2 + a 4 = 5

-- Statement of the problem
theorem find_expression_for_an (a : ℕ → ℝ) (q : ℝ) :
  problem_conditions a q → ∀ n : ℕ, a n = 2 ^ (4 - n) :=
sorry

end find_expression_for_an_l174_174482


namespace calculation_l174_174659

theorem calculation :
  12 - 10 + 8 / 2 * 5 + 4 - 6 * 3 + 1 = 9 :=
by
  sorry

end calculation_l174_174659


namespace combined_resistance_parallel_l174_174826

theorem combined_resistance_parallel (x y r : ℝ) (hx : x = 4) (hy : y = 5)
  (h_combined : 1 / r = 1 / x + 1 / y) : r = 20 / 9 := by
  sorry

end combined_resistance_parallel_l174_174826


namespace binomial_coefficient_10_3_l174_174274

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l174_174274


namespace hiker_distance_l174_174639

noncomputable def distance_from_start (north south east west : ℕ) : ℝ :=
  let north_south := north - south
  let east_west := east - west
  Real.sqrt (north_south ^ 2 + east_west ^ 2)

theorem hiker_distance :
  distance_from_start 24 8 15 9 = 2 * Real.sqrt 73 := by
  sorry

end hiker_distance_l174_174639


namespace arithmetic_sequence_sum_l174_174663

theorem arithmetic_sequence_sum :
  ∃ x y z d : ℝ, 
  d = (31 - 4) / 5 ∧ 
  x = 4 + d ∧ 
  y = x + d ∧ 
  z = 16 + d ∧ 
  (x + y + z) = 45.6 :=
by
  sorry

end arithmetic_sequence_sum_l174_174663


namespace line_parabola_intersections_l174_174692

theorem line_parabola_intersections (k : ℝ) :
  ((∃ x y, y = k * (x - 2) + 1 ∧ y^2 = 4 * x) ↔ k = 0) ∧
  (¬∃ x₁ x₂, x₁ ≠ x₂ ∧ (k * (x₁ - 2) + 1)^2 = 4 * x₁ ∧ (k * (x₂ - 2) + 1)^2 = 4 * x₂) ∧
  (¬∃ x y, y = k * (x - 2) + 1 ∧ y^2 = 4 * x) :=
by sorry

end line_parabola_intersections_l174_174692


namespace arithmetic_geometric_sequence_l174_174072

theorem arithmetic_geometric_sequence (d : ℤ) (a_1 a_2 a_5 : ℤ)
  (h1 : d ≠ 0)
  (h2 : a_2 = a_1 + d)
  (h3 : a_5 = a_1 + 4 * d)
  (h4 : a_2 ^ 2 = a_1 * a_5) :
  a_5 = 9 * a_1 := 
sorry

end arithmetic_geometric_sequence_l174_174072


namespace binomial_coefficient_10_3_l174_174269

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l174_174269


namespace hexadecagon_area_l174_174491

theorem hexadecagon_area (r : ℝ) : 
  (∃ A : ℝ, A = 4 * r^2 * Real.sqrt (2 - Real.sqrt 2)) :=
sorry

end hexadecagon_area_l174_174491


namespace circle_center_radius_l174_174935

theorem circle_center_radius
    (x y : ℝ)
    (eq_circle : (x - 2)^2 + y^2 = 4) :
    (2, 0) = (2, 0) ∧ 2 = 2 :=
by
  sorry

end circle_center_radius_l174_174935


namespace minimize_abs_difference_and_product_l174_174498

theorem minimize_abs_difference_and_product (x y : ℤ) (n : ℤ) 
(h1 : 20 * x + 19 * y = 2019)
(h2 : |x - y| = 18) 
: x * y = 2623 :=
sorry

end minimize_abs_difference_and_product_l174_174498


namespace solve_abs_inequality_l174_174512

theorem solve_abs_inequality (x : ℝ) :
  3 ≤ abs ((x - 3)^2 - 4) ∧ abs ((x - 3)^2 - 4) ≤ 7 ↔ 3 - Real.sqrt 11 ≤ x ∧ x ≤ 3 + Real.sqrt 11 :=
sorry

end solve_abs_inequality_l174_174512


namespace value_of_a_l174_174869

theorem value_of_a {a x : ℝ} (h1 : x > 0) (h2 : 2 * x + 1 > a * x) : a ≤ 2 :=
sorry

end value_of_a_l174_174869


namespace last_digit_322_power_111569_l174_174367

theorem last_digit_322_power_111569 : (322 ^ 111569) % 10 = 2 := 
by {
  sorry
}

end last_digit_322_power_111569_l174_174367


namespace number_of_hens_l174_174032

theorem number_of_hens (H C : ℕ) (h1 : H + C = 44) (h2 : 2 * H + 4 * C = 128) : H = 24 :=
by
  sorry

end number_of_hens_l174_174032


namespace train_length_l174_174648

theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (length_train : ℝ) 
  (h_speed : speed_kmph = 50)
  (h_time : time_sec = 18) 
  (h_length : length_train = 250) : 
  (speed_kmph * 1000 / 3600) * time_sec = length_train :=
by 
  rw [h_speed, h_time, h_length]
  sorry

end train_length_l174_174648


namespace ratio_of_houses_second_to_first_day_l174_174715

theorem ratio_of_houses_second_to_first_day 
    (houses_day1 : ℕ)
    (houses_day2 : ℕ)
    (sales_per_house : ℕ)
    (sold_pct_day2 : ℝ) 
    (total_sales_day1 : ℕ)
    (total_sales_day2 : ℝ) :
    houses_day1 = 20 →
    sales_per_house = 2 →
    sold_pct_day2 = 0.8 →
    total_sales_day1 = houses_day1 * sales_per_house →
    total_sales_day2 = sold_pct_day2 * houses_day2 * sales_per_house →
    total_sales_day1 = total_sales_day2 →
    (houses_day2 : ℝ) / houses_day1 = 5 / 4 :=
by
    intro h1 h2 h3 h4 h5 h6
    sorry

end ratio_of_houses_second_to_first_day_l174_174715


namespace inequality_solution_l174_174379

theorem inequality_solution (x : ℝ) : x^3 - 12 * x^2 > -36 * x ↔ x ∈ (Set.Ioo 0 6) ∪ (Set.Ioi 6) :=
by
  sorry

end inequality_solution_l174_174379


namespace find_speed_of_A_l174_174141

noncomputable def speed_of_A_is_7_5 (a : ℝ) : Prop :=
  -- Conditions
  ∃ (b : ℝ), b = a + 5 ∧ 
  (60 / a = 100 / b) → 
  -- Conclusion
  a = 7.5

-- Statement in Lean 4
theorem find_speed_of_A (a : ℝ) (h : speed_of_A_is_7_5 a) : a = 7.5 :=
  sorry

end find_speed_of_A_l174_174141


namespace binomial_10_3_eq_120_l174_174203

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l174_174203


namespace matrix_pow_2018_l174_174995

open Matrix

-- Define the specific matrix
def A : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 0], ![1, 1]]

-- Formalize the statement
theorem matrix_pow_2018 : A ^ 2018 = ![![1, 0], ![2018, 1]] :=
  sorry

end matrix_pow_2018_l174_174995


namespace inequality_holds_l174_174814

theorem inequality_holds (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by
  sorry

end inequality_holds_l174_174814


namespace binomial_10_3_eq_120_l174_174210

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l174_174210


namespace gcd_840_1764_gcd_98_63_l174_174635

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := 
by sorry

theorem gcd_98_63 : Nat.gcd 98 63 = 7 :=
by sorry

end gcd_840_1764_gcd_98_63_l174_174635


namespace mixed_number_sum_l174_174180

theorem mixed_number_sum :
  481 + 1/6  + 265 + 1/12 + 904 + 1/20 -
  (184 + 29/30) - (160 + 41/42) - (703 + 55/56) =
  603 + 3/8 :=
by
  sorry

end mixed_number_sum_l174_174180


namespace power_sum_eq_l174_174848

theorem power_sum_eq : (-2)^2011 + (-2)^2012 = 2^2011 := by
  sorry

end power_sum_eq_l174_174848


namespace binomial_coefficient_10_3_l174_174267

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l174_174267


namespace combination_10_3_eq_120_l174_174245

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end combination_10_3_eq_120_l174_174245


namespace total_chickens_and_ducks_l174_174115

-- Definitions based on conditions
def num_chickens : Nat := 45
def more_chickens_than_ducks : Nat := 8
def num_ducks : Nat := num_chickens - more_chickens_than_ducks

-- The proof statement
theorem total_chickens_and_ducks : num_chickens + num_ducks = 82 := by
  -- The actual proof is omitted, only the statement is required
  sorry

end total_chickens_and_ducks_l174_174115


namespace johns_pants_cost_50_l174_174098

variable (P : ℝ)

theorem johns_pants_cost_50 (h1 : P + 1.60 * P = 130) : P = 50 := 
by
  sorry

end johns_pants_cost_50_l174_174098


namespace jail_time_weeks_l174_174757

theorem jail_time_weeks (days_protest : ℕ) (cities : ℕ) (arrests_per_day : ℕ)
  (days_pre_trial : ℕ) (half_week_sentence_days : ℕ) :
  days_protest = 30 →
  cities = 21 →
  arrests_per_day = 10 →
  days_pre_trial = 4 →
  half_week_sentence_days = 7 →
  (21 * 30 * 10 * (4 + 7)) / 7 = 9900 :=
by
  intros h_days_protest h_cities h_arrests_per_day h_days_pre_trial h_half_week_sentence_days
  rw [h_days_protest, h_cities, h_arrests_per_day, h_days_pre_trial, h_half_week_sentence_days]
  exact sorry

end jail_time_weeks_l174_174757


namespace six_letter_words_no_substring_amc_l174_174878

theorem six_letter_words_no_substring_amc : 
  let alphabet := ['A', 'M', 'C']
  let totalNumberOfWords := 3^6
  let numberOfWordsContainingAMC := 4 * 3^3 - 1
  let numberOfWordsNotContainingAMC := totalNumberOfWords - numberOfWordsContainingAMC
  numberOfWordsNotContainingAMC = 622 :=
by
  sorry

end six_letter_words_no_substring_amc_l174_174878


namespace smallest_a_value_l174_174442

theorem smallest_a_value :
  ∃ (a : ℝ), (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 →
    2 * (Real.sin (Real.pi - (Real.pi * x^2 / 12))) * (Real.cos (Real.pi / 6 * Real.sqrt (9 - x^2))) + 1 = a + 2 * (Real.sin (Real.pi / 6 * Real.sqrt (9 - x^2))) * (Real.cos (Real.pi * x^2 / 12))) ∧
    ∀ a' : ℝ, (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 →
      2 * (Real.sin (Real.pi - (Real.pi * x^2 / 12))) * (Real.cos (Real.pi / 6 * Real.sqrt (9 - x^2))) + 1 = a' + 2 * (Real.sin (Real.pi / 6 * Real.sqrt (9 - x^2))) * (Real.cos (Real.pi * x^2 / 12))) →
      a ≤ a'
  := sorry

end smallest_a_value_l174_174442


namespace triangle_inequality_l174_174972

variable {α : Type*} [LinearOrderedField α]

/-- Given a triangle ABC with sides a, b, c, circumradius R, 
exradii r_a, r_b, r_c, and given 2R ≤ r_a, we need to show that a > b, a > c, 2R > r_b, and 2R > r_c. -/
theorem triangle_inequality (a b c R r_a r_b r_c : α) (h₁ : 2 * R ≤ r_a) :
  a > b ∧ a > c ∧ 2 * R > r_b ∧ 2 * R > r_c := by
  sorry

end triangle_inequality_l174_174972


namespace traffic_light_probability_l174_174986

open ProbabilityTheory MeasureTheory Set Real

noncomputable def period : ℝ := (60 + 45) / 60 -- Total period, in minutes

noncomputable def greenLightInterval : Set ℝ := Ioc 0 1 -- Interval representing green light

noncomputable def uniformDist : MeasureSpace ℝ := 
  volume.restrict (Ioc 0 period) / (volume (Ioc 0 period))

theorem traffic_light_probability :
  (volume.restrict greenLightInterval uniformDist).val = 4 / 7 :=
by 
  sorry

end traffic_light_probability_l174_174986


namespace combination_10_3_l174_174283

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end combination_10_3_l174_174283


namespace inequality_proof_l174_174528

theorem inequality_proof (x y : ℝ) (hx: 0 < x) (hy: 0 < y) : 
  1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧ 
  (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 := 
by 
  sorry

end inequality_proof_l174_174528


namespace binom_10_3_eq_120_l174_174190

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l174_174190


namespace rick_books_division_l174_174585

theorem rick_books_division (books_per_group initial_books final_groups : ℕ) 
  (h_initial : initial_books = 400) 
  (h_books_per_group : books_per_group = 25) 
  (h_final_groups : final_groups = 16) : 
  ∃ divisions : ℕ, (divisions = 4) ∧ 
    ∃ f : ℕ → ℕ, 
    (f 0 = initial_books) ∧ 
    (f divisions = books_per_group * final_groups) ∧ 
    (∀ n, 1 ≤ n → n ≤ divisions → f n = f (n - 1) / 2) := 
by 
  sorry

end rick_books_division_l174_174585


namespace eq_x2_inv_x2_and_x8_inv_x8_l174_174696

theorem eq_x2_inv_x2_and_x8_inv_x8 (x : ℝ) 
  (h : 47 = x^4 + 1 / x^4) : 
  (x^2 + 1 / x^2 = 7) ∧ (x^8 + 1 / x^8 = -433) :=
by
  sorry

end eq_x2_inv_x2_and_x8_inv_x8_l174_174696


namespace garden_watering_system_pumps_l174_174480

-- Define conditions
def rate := 500 -- gallons per hour
def time := 30 / 60 -- hours, i.e., converting 30 minutes to hours

-- Theorem statement
theorem garden_watering_system_pumps :
  rate * time = 250 := by
  sorry

end garden_watering_system_pumps_l174_174480


namespace certain_number_is_36_75_l174_174615

theorem certain_number_is_36_75 (A B C X : ℝ) (h_ratio_A : A = 5 * (C / 8)) (h_ratio_B : B = 6 * (C / 8)) (h_C : C = 42) (h_relation : A + C = B + X) :
  X = 36.75 :=
by
  sorry

end certain_number_is_36_75_l174_174615


namespace find_product_of_constants_l174_174905

theorem find_product_of_constants
  (M1 M2 : ℝ)
  (h : ∀ x : ℝ, (x - 1) * (x - 2) ≠ 0 → (45 * x - 31) / (x * x - 3 * x + 2) = M1 / (x - 1) + M2 / (x - 2)) :
  M1 * M2 = -826 :=
sorry

end find_product_of_constants_l174_174905


namespace inequality_abc_l174_174927

variable (a b c : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hc : c > 0)
variable (cond : a + b + c = (1 / a) + (1 / b) + (1 / c))

theorem inequality_abc : a + b + c ≥ 3 / (a * b * c) :=
sorry

end inequality_abc_l174_174927


namespace proposition_judgement_l174_174774

theorem proposition_judgement (p q : Prop) (a b c x : ℝ) :
  (¬ (p ∨ q) → (¬ p ∧ ¬ q)) ∧
  (¬ (a > b → a * c^2 > b * c^2)) ∧
  (¬ (∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0)) ∧
  ((x^2 - 3*x + 2 = 0) → (x = 2)) =
  false := sorry

end proposition_judgement_l174_174774


namespace yuna_has_biggest_number_l174_174777

-- Define the collections
def yoongi_collected : ℕ := 4
def jungkook_collected : ℕ := 6 - 3
def yuna_collected : ℕ := 5

-- State the theorem
theorem yuna_has_biggest_number :
  yuna_collected > yoongi_collected ∧ yuna_collected > jungkook_collected :=
by
  sorry

end yuna_has_biggest_number_l174_174777


namespace alpha_minus_beta_eq_pi_div_4_l174_174540

open Real

theorem alpha_minus_beta_eq_pi_div_4 (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 4) 
(h : tan α = (cos β + sin β) / (cos β - sin β)) : α - β = π / 4 :=
sorry

end alpha_minus_beta_eq_pi_div_4_l174_174540


namespace binom_10_3_l174_174238

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end binom_10_3_l174_174238


namespace value_of_h_l174_174081

theorem value_of_h (h : ℝ) : (∃ x : ℝ, x^3 + h * x - 14 = 0 ∧ x = 3) → h = -13/3 :=
by
  sorry

end value_of_h_l174_174081


namespace binomial_coefficient_10_3_l174_174271

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l174_174271


namespace hawkeye_fewer_mainecoons_than_gordon_l174_174417

-- Definitions based on conditions
def JamiePersians : ℕ := 4
def JamieMaineCoons : ℕ := 2
def GordonPersians : ℕ := JamiePersians / 2
def GordonMaineCoons : ℕ := JamieMaineCoons + 1
def TotalCats : ℕ := 13
def JamieTotalCats : ℕ := JamiePersians + JamieMaineCoons
def GordonTotalCats : ℕ := GordonPersians + GordonMaineCoons
def JamieAndGordonTotalCats : ℕ := JamieTotalCats + GordonTotalCats
def HawkeyeTotalCats : ℕ := TotalCats - JamieAndGordonTotalCats
def HawkeyePersians : ℕ := 0
def HawkeyeMaineCoons : ℕ := HawkeyeTotalCats - HawkeyePersians

-- Theorem statement to prove: Hawkeye owns 1 fewer Maine Coon than Gordon
theorem hawkeye_fewer_mainecoons_than_gordon : HawkeyeMaineCoons + 1 = GordonMaineCoons :=
by
  sorry

end hawkeye_fewer_mainecoons_than_gordon_l174_174417


namespace quadratic_inequality_solution_l174_174393

theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x : ℝ, ax^2 + 2 * x + 1 > 0) ↔ (a > 1) :=
by
  sorry

end quadratic_inequality_solution_l174_174393


namespace inequality_solution_l174_174738

theorem inequality_solution :
  { x : ℝ // x < 2 ∨ (3 < x ∧ x < 6) ∨ (7 < x ∧ x < 8) } →
  ((x - 3) * (x - 5) * (x - 7)) / ((x - 2) * (x - 6) * (x - 8)) > 0 :=
by
  sorry

end inequality_solution_l174_174738


namespace inequality_abc_l174_174930

variable (a b c : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hc : c > 0)
variable (cond : a + b + c = (1 / a) + (1 / b) + (1 / c))

theorem inequality_abc : a + b + c ≥ 3 / (a * b * c) :=
sorry

end inequality_abc_l174_174930


namespace parts_per_hour_l174_174454

theorem parts_per_hour (x y : ℝ) (h₁ : 90 / x = 120 / y) (h₂ : x + y = 35) : x = 15 ∧ y = 20 :=
by
  sorry

end parts_per_hour_l174_174454


namespace cuboid_edge_length_l174_174743

-- This is the main statement we want to prove
theorem cuboid_edge_length (L : ℝ) (w : ℝ) (h : ℝ) (V : ℝ) (w_eq : w = 5) (h_eq : h = 3) (V_eq : V = 30) :
  V = L * w * h → L = 2 :=
by
  -- Adding the sorry allows us to compile and acknowledge the current placeholder for the proof.
  sorry

end cuboid_edge_length_l174_174743


namespace sum_of_fractions_l174_174657

theorem sum_of_fractions :
  (1 / 10) + (2 / 10) + (3 / 10) + (4 / 10) + (5 / 10) + (6 / 10) + (7 / 10) + (8 / 10) + (10 / 10) + (60 / 10) = 10.6 := by
  sorry

end sum_of_fractions_l174_174657


namespace two_b_leq_a_plus_c_l174_174681

variable (t a b c : ℝ)

theorem two_b_leq_a_plus_c (ht : t > 1)
  (h : 2 / Real.log t / Real.log b = 1 / Real.log t / Real.log a + 1 / Real.log t / Real.log c) :
  2 * b ≤ a + c := by sorry

end two_b_leq_a_plus_c_l174_174681


namespace chocolates_sold_in_second_week_l174_174953

theorem chocolates_sold_in_second_week
  (c₁ c₂ c₃ c₄ c₅ : ℕ)
  (h₁ : c₁ = 75)
  (h₃ : c₃ = 75)
  (h₄ : c₄ = 70)
  (h₅ : c₅ = 68)
  (h_mean : (c₁ + c₂ + c₃ + c₄ + c₅) / 5 = 71) :
  c₂ = 67 := 
sorry

end chocolates_sold_in_second_week_l174_174953


namespace ball_returns_to_bella_after_13_throws_l174_174140

def girl_after_throws (start : ℕ) (throws : ℕ) : ℕ :=
  (start + throws * 5) % 13

theorem ball_returns_to_bella_after_13_throws :
  girl_after_throws 1 13 = 1 :=
sorry

end ball_returns_to_bella_after_13_throws_l174_174140


namespace binom_10_3_eq_120_l174_174320

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l174_174320


namespace binomial_10_3_eq_120_l174_174197

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l174_174197


namespace sin_eq_cos_510_l174_174672

theorem sin_eq_cos_510 (n : ℤ) (h1 : -180 ≤ n ∧ n ≤ 180) (h2 : Real.sin (n * Real.pi / 180) = Real.cos (510 * Real.pi / 180)) :
  n = -60 :=
sorry

end sin_eq_cos_510_l174_174672


namespace binomial_10_3_l174_174257

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end binomial_10_3_l174_174257


namespace inequality_inequality_holds_l174_174780

theorem inequality_inequality_holds (x y z : ℝ) : 
  (x^3 / (x^3 + 2 * y^2 * z)) + (y^3 / (y^3 + 2 * z^2 * x)) + (z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_inequality_holds_l174_174780


namespace rick_division_steps_l174_174592

theorem rick_division_steps (initial_books : ℕ) (final_books : ℕ) 
  (h_initial : initial_books = 400) (h_final : final_books = 25) : 
  (∀ n : ℕ, (initial_books / (2^n) = final_books) → n = 4) :=
by
  sorry

end rick_division_steps_l174_174592


namespace three_term_inequality_l174_174824

theorem three_term_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_abc : a * b * c = 1) :
  (a^3 / (a^3 + 2 * b^2)) + (b^3 / (b^3 + 2 * c^2)) + (c^3 / (c^3 + 2 * a^2)) ≥ 1 :=
by
  sorry

end three_term_inequality_l174_174824


namespace area_N1N2N3_relative_l174_174897

-- Definitions
variable (A B C D E F N1 N2 N3 : Type)
-- Assuming D, E, F are points on sides BC, CA, AB respectively such that CD, AE, BF are one-fourth of their respective sides.
variable (area_ABC : ℝ)  -- Total area of triangle ABC
variable (area_N1N2N3 : ℝ)  -- Area of triangle N1N2N3

-- Given conditions
variable (H1 : CD = 1 / 4 * BC)
variable (H2 : AE = 1 / 4 * CA)
variable (H3 : BF = 1 / 4 * AB)

-- The expected result
theorem area_N1N2N3_relative :
  area_N1N2N3 = 7 / 15 * area_ABC :=
sorry

end area_N1N2N3_relative_l174_174897


namespace binom_10_3_eq_120_l174_174312

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l174_174312


namespace original_class_strength_l174_174128

variable (x : ℕ)

/-- The average age of an adult class is 40 years.
  18 new students with an average age of 32 years join the class, 
  therefore decreasing the average by 4 years.
  Find the original strength of the class.
-/
theorem original_class_strength (h1 : 40 * x + 18 * 32 = (x + 18) * 36) : x = 18 := 
by sorry

end original_class_strength_l174_174128


namespace combination_10_3_eq_120_l174_174241

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end combination_10_3_eq_120_l174_174241


namespace binom_10_3_l174_174226

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end binom_10_3_l174_174226


namespace intersection_lines_l174_174136

theorem intersection_lines (c d : ℝ) :
    (∃ x y, x = (1/3) * y + c ∧ y = (1/3) * x + d ∧ x = 3 ∧ y = -1) →
    c + d = 4 / 3 :=
by
  sorry

end intersection_lines_l174_174136


namespace binomial_10_3_eq_120_l174_174346

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l174_174346


namespace binomial_10_3_l174_174263

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end binomial_10_3_l174_174263


namespace area_transformed_region_l174_174103

-- Define the transformation matrix
def matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 1], ![4, 3]]

-- Define the area of region T
def area_T := 6

-- The statement we want to prove: the area of T' is 30.
theorem area_transformed_region :
  let det := matrix.det
  area_T * det = 30 :=
by
  sorry

end area_transformed_region_l174_174103


namespace interest_calculation_years_l174_174742

theorem interest_calculation_years (P r : ℝ) (diff : ℝ) (n : ℕ) 
  (hP : P = 3600) (hr : r = 0.10) (hdiff : diff = 36) 
  (h_eq : P * (1 + r)^n - P - (P * r * n) = diff) : n = 2 :=
sorry

end interest_calculation_years_l174_174742


namespace binomial_10_3_l174_174254

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end binomial_10_3_l174_174254


namespace binomial_coefficient_10_3_l174_174272

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l174_174272


namespace distance_between_riya_and_priya_l174_174435

theorem distance_between_riya_and_priya (speed_riya speed_priya : ℝ) (time_hours : ℝ)
  (h1 : speed_riya = 21) (h2 : speed_priya = 22) (h3 : time_hours = 1) :
  speed_riya * time_hours + speed_priya * time_hours = 43 := by
  sorry

end distance_between_riya_and_priya_l174_174435


namespace find_k_l174_174556

variable (m n k : ℝ)

-- Conditions from the problem
def quadratic_roots : Prop := (m + n = -2) ∧ (m * n = k) ∧ (1/m + 1/n = 6)

-- Theorem statement
theorem find_k (h : quadratic_roots m n k) : k = -1/3 :=
sorry

end find_k_l174_174556


namespace inequality_inequality_holds_l174_174782

theorem inequality_inequality_holds (x y z : ℝ) : 
  (x^3 / (x^3 + 2 * y^2 * z)) + (y^3 / (y^3 + 2 * z^2 * x)) + (z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_inequality_holds_l174_174782


namespace expected_value_is_one_dollar_l174_174490

def star_prob := 1 / 4
def moon_prob := 1 / 2
def sun_prob := 1 / 4

def star_prize := 2
def moon_prize := 4
def sun_penalty := -6

def expected_winnings := star_prob * star_prize + moon_prob * moon_prize + sun_prob * sun_penalty

theorem expected_value_is_one_dollar : expected_winnings = 1 := by
  sorry

end expected_value_is_one_dollar_l174_174490


namespace inequality_proof_l174_174523

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧
  (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 :=
sorry

end inequality_proof_l174_174523


namespace jack_total_damage_costs_l174_174415

def cost_per_tire := 250
def number_of_tires := 3
def cost_of_window := 700

def total_cost_of_tires := cost_per_tire * number_of_tires
def total_cost_of_damages := total_cost_of_tires + cost_of_window

theorem jack_total_damage_costs : total_cost_of_damages = 1450 := 
by
  -- Using the definitions provided
  -- total_cost_of_tires = 250 * 3 = 750
  -- total_cost_of_damages = 750 + 700 = 1450
  sorry

end jack_total_damage_costs_l174_174415


namespace Ivan_ball_ways_l174_174429

theorem Ivan_ball_ways (N : ℕ) :
  let guests := (Fin (2 * N)) in
  let hats := vector bool (2 * N) in
  let black_count := hats.toList.count true = N in
  let white_count := hats.toList.count false = N in
  let no_adj_same_color := ∀ i, hats.nth i ≠ hats.nth ((i + 1) % (2 * N)) in
  black_count → white_count → no_adj_same_color → (2 * N)! = (2 * N)! :=
by
  intros
  sorry

end Ivan_ball_ways_l174_174429


namespace cube_split_with_333_l174_174374

theorem cube_split_with_333 (m : ℕ) (h1 : m > 1)
  (h2 : ∃ k : ℕ, (333 = 2 * k + 1) ∧ (333 + 2 * (k - k) + 2) * k = m^3 ) :
  m = 18 := sorry

end cube_split_with_333_l174_174374


namespace isosceles_triangles_perimeter_l174_174110

theorem isosceles_triangles_perimeter (c d : ℕ) 
  (h1 : ¬(7 = c ∧ 10 = d) ∧ ¬(7 = d ∧ 10 = c))
  (h2 : 2 * c + d = 24) :
  d = 2 :=
sorry

end isosceles_triangles_perimeter_l174_174110


namespace difference_between_shares_l174_174445

def investment_months (amount : ℕ) (months : ℕ) : ℕ :=
  amount * months

def ratio (investment_months : ℕ) (total_investment_months : ℕ) : ℚ :=
  investment_months / total_investment_months

def profit_share (ratio : ℚ) (total_profit : ℝ) : ℝ :=
  ratio * total_profit

theorem difference_between_shares :
  let suresh_investment := 18000
  let rohan_investment := 12000
  let sudhir_investment := 9000
  let suresh_months := 12
  let rohan_months := 9
  let sudhir_months := 8
  let total_profit := 3795
  let suresh_investment_months := investment_months suresh_investment suresh_months
  let rohan_investment_months := investment_months rohan_investment rohan_months
  let sudhir_investment_months := investment_months sudhir_investment sudhir_months
  let total_investment_months := suresh_investment_months + rohan_investment_months + sudhir_investment_months
  let suresh_ratio := ratio suresh_investment_months total_investment_months
  let rohan_ratio := ratio rohan_investment_months total_investment_months
  let sudhir_ratio := ratio sudhir_investment_months total_investment_months
  let rohan_share := profit_share rohan_ratio total_profit
  let sudhir_share := profit_share sudhir_ratio total_profit
  rohan_share - sudhir_share = 345 :=
by
  sorry

end difference_between_shares_l174_174445


namespace binom_10_3_l174_174234

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end binom_10_3_l174_174234


namespace paperback_copies_sold_l174_174754

theorem paperback_copies_sold
  (H P : ℕ)
  (h1 : H = 36000)
  (h2 : H + P = 440000) :
  P = 404000 :=
by
  rw [h1] at h2
  sorry

end paperback_copies_sold_l174_174754


namespace find_triplets_l174_174858

theorem find_triplets (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a ^ b ∣ b ^ c - 1) ∧ (a ^ c ∣ c ^ b - 1)) ↔ (a = 1 ∨ (b = 1 ∧ c = 1)) :=
by sorry

end find_triplets_l174_174858


namespace binomial_10_3_l174_174307

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end binomial_10_3_l174_174307


namespace mixed_number_sum_l174_174181

theorem mixed_number_sum :
  481 + 1/6  + 265 + 1/12 + 904 + 1/20 -
  (184 + 29/30) - (160 + 41/42) - (703 + 55/56) =
  603 + 3/8 :=
by
  sorry

end mixed_number_sum_l174_174181


namespace combination_10_3_eq_120_l174_174242

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end combination_10_3_eq_120_l174_174242


namespace inequality_inequality_holds_l174_174779

theorem inequality_inequality_holds (x y z : ℝ) : 
  (x^3 / (x^3 + 2 * y^2 * z)) + (y^3 / (y^3 + 2 * z^2 * x)) + (z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_inequality_holds_l174_174779


namespace inequality_proof_l174_174527

theorem inequality_proof (x y : ℝ) (hx: 0 < x) (hy: 0 < y) : 
  1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧ 
  (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 := 
by 
  sorry

end inequality_proof_l174_174527


namespace evaluate_expression_zero_l174_174664

-- Define the variables and conditions
def x : ℕ := 4
def z : ℕ := 0

-- State the property to be proved
theorem evaluate_expression_zero : z * (2 * z - 5 * x) = 0 := by
  sorry

end evaluate_expression_zero_l174_174664


namespace binom_10_3_eq_120_l174_174220

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end binom_10_3_eq_120_l174_174220


namespace divide_segment_mean_proportional_l174_174854

theorem divide_segment_mean_proportional (a : ℝ) (x : ℝ) : 
  ∃ H : ℝ, H > 0 ∧ H < a ∧ H = (a * (Real.sqrt 5 - 1) / 2) :=
sorry

end divide_segment_mean_proportional_l174_174854


namespace original_people_in_room_l174_174410

theorem original_people_in_room (x : ℕ) 
  (h1 : 3 * x / 4 - 3 * x / 20 = 16) : x = 27 :=
sorry

end original_people_in_room_l174_174410


namespace binomial_10_3_eq_120_l174_174342

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l174_174342


namespace cube_volume_from_surface_area_l174_174003

theorem cube_volume_from_surface_area (S : ℝ) (h : S = 864) : ∃ V : ℝ, V = 1728 :=
by
  -- Assume surface area formula S = 6s^2, solve steps skipped and go directly to conclusion
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  use V
  sorry

end cube_volume_from_surface_area_l174_174003


namespace christopher_strolling_time_l174_174851

theorem christopher_strolling_time
  (initial_distance : ℝ) (initial_speed : ℝ) (break_time : ℝ)
  (continuation_distance : ℝ) (continuation_speed : ℝ)
  (H1 : initial_distance = 2) (H2 : initial_speed = 4)
  (H3 : break_time = 0.25) (H4 : continuation_distance = 3)
  (H5 : continuation_speed = 6) :
  (initial_distance / initial_speed + break_time + continuation_distance / continuation_speed) = 1.25 := 
  sorry

end christopher_strolling_time_l174_174851


namespace remainder_3_45_plus_4_mod_5_l174_174466

theorem remainder_3_45_plus_4_mod_5 :
  (3 ^ 45 + 4) % 5 = 2 := 
by {
  sorry
}

end remainder_3_45_plus_4_mod_5_l174_174466


namespace total_cost_magic_decks_l174_174633

theorem total_cost_magic_decks (price_per_deck : ℕ) (frank_decks : ℕ) (friend_decks : ℕ) :
  price_per_deck = 7 ∧ frank_decks = 3 ∧ friend_decks = 2 → 
  (price_per_deck * frank_decks + price_per_deck * friend_decks) = 35 :=
by
  sorry

end total_cost_magic_decks_l174_174633


namespace prove_inequality_l174_174102

noncomputable def problem_statement (p q r : ℝ) (n : ℕ) (h_pqr : p * q * r = 1) : Prop :=
  (1 / (p^n + q^n + 1)) + (1 / (q^n + r^n + 1)) + (1 / (r^n + p^n + 1)) ≤ 1

theorem prove_inequality (p q r : ℝ) (n : ℕ) (h_pqr : p * q * r = 1) (h_pos_p : 0 < p) (h_pos_q : 0 < q) (h_pos_r : 0 < r) : 
  problem_statement p q r n h_pqr :=
by
  sorry

end prove_inequality_l174_174102


namespace solve_inequality_l174_174737

theorem solve_inequality 
  (x : ℝ) :
  (\{x \mid \frac{(x - 3) * (x - 5) * (x - 7)}{(x - 2) * (x - 6) * (x - 8)} > 0\} =
  {x | x < 2} ∪ {x | 3 < x ∧ x < 5} ∪ {x | 6 < x ∧ x < 7}  ∪ {x | 8 < x}) :=
sorry

end solve_inequality_l174_174737


namespace set_intersection_complement_l174_174694

open Set

universe u

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

/-- Given the universal set U={0,1,2,3,4,5}, sets A={0,2,4}, and B={0,5}, prove that
    the intersection of A and the complement of B in U is {2,4}. -/
theorem set_intersection_complement:
  U = {0, 1, 2, 3, 4, 5} →
  A = {0, 2, 4} →
  B = {0, 5} →
  A ∩ (U \ B) = {2, 4} := 
by
  intros hU hA hB
  sorry

end set_intersection_complement_l174_174694


namespace side_length_of_square_l174_174879

theorem side_length_of_square : 
  ∀ (L : ℝ), L = 28 → (L / 4) = 7 :=
by
  intro L h
  rw [h]
  norm_num

end side_length_of_square_l174_174879


namespace inequality_ge_one_l174_174805

theorem inequality_ge_one {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_ge_one_l174_174805


namespace sales_fraction_l174_174100

theorem sales_fraction (A D : ℝ) (h : D = 2 * A) : D / (11 * A + D) = 2 / 13 :=
by
  sorry

end sales_fraction_l174_174100


namespace population_increase_difference_l174_174409

noncomputable def births_per_day : ℝ := 24 / 6
noncomputable def deaths_per_day : ℝ := 24 / 16
noncomputable def net_increase_per_day : ℝ := births_per_day - deaths_per_day
noncomputable def annual_increase_regular_year : ℝ := net_increase_per_day * 365
noncomputable def annual_increase_leap_year : ℝ := net_increase_per_day * 366

theorem population_increase_difference :
  annual_increase_leap_year - annual_increase_regular_year = 2.5 :=
by {
  sorry
}

end population_increase_difference_l174_174409


namespace combination_10_3_eq_120_l174_174363

open Nat

theorem combination_10_3_eq_120 : (10.choose 3) = 120 := 
by
  sorry

end combination_10_3_eq_120_l174_174363


namespace count_remainders_gte_l174_174376

def remainder (a N : ℕ) : ℕ := a % N

theorem count_remainders_gte (N : ℕ) : 
  (∀ a, a > 0 → remainder a 1000 > remainder a 1001 → N ≤ 1000000) →
  N = 499500 :=
by
  sorry

end count_remainders_gte_l174_174376


namespace not_late_prob_expected_encounters_l174_174955

open MeasureTheory ProbabilityTheory

noncomputable def redLightProbability : ProbabilityTheory.ProbabilitySpace ℝ := sorry

-- Define the probability of encountering red lights at each post
def prob_A : ProbabilityTheory.ProbMeasure ℝ := sorry
def prob_B : ProbabilityTheory.ProbMeasure ℝ := sorry
def prob_C : ProbabilityTheory.ProbMeasure ℝ := sorry
def prob_D : ProbabilityTheory.ProbMeasure ℝ := sorry

-- Define the indicator random variables for encountering red lights at each post
def indicator_A : ProbabilityTheory.RandomVariable ℝ := sorry
def indicator_B : ProbabilityTheory.RandomVariable ℝ := sorry
def indicator_C : ProbabilityTheory.RandomVariable ℝ := sorry
def indicator_D : ProbabilityTheory.RandomVariable ℝ := sorry

-- Define X as the total number of red lights encountered
def X := indicator_A + indicator_B + indicator_C + indicator_D

-- Define the condition for being late
def is_late : ℝ → Prop := λ x, x ≥ 3

-- Define the probability of not being late
def prob_not_late : ℝ := 
  ProbabilityTheory.P (λ x, ¬is_late x)
  
-- Define the expected value of X
def expected_X : ℝ := 
  ProbabilityTheory.ExpectedValue X

theorem not_late_prob : prob_not_late redLightProbability = 29/36 := sorry
theorem expected_encounters : expected_X = 5/3 := sorry

end not_late_prob_expected_encounters_l174_174955


namespace minimize_abs_difference_and_product_l174_174499

theorem minimize_abs_difference_and_product (x y : ℤ) (n : ℤ) 
(h1 : 20 * x + 19 * y = 2019)
(h2 : |x - y| = 18) 
: x * y = 2623 :=
sorry

end minimize_abs_difference_and_product_l174_174499


namespace binom_10_3_eq_120_l174_174186

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l174_174186


namespace arithmetic_sequence_common_difference_l174_174682

theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) (d : ℝ) 
  (h_seq : ∀ n, a (n + 1) = a n + d)
  (h_variance : (1/5) * ((a 1 - (a 3)) ^ 2 + (a 2 - (a 3)) ^ 2 + (a 3 - (a 3)) ^ 2 + (a 4 - (a 3)) ^ 2 + (a 5 - (a 3)) ^ 2) = 8) :
  d = 2 ∨ d = -2 := 
sorry

end arithmetic_sequence_common_difference_l174_174682


namespace squirrel_divides_acorns_l174_174166

theorem squirrel_divides_acorns (total_acorns parts_per_month remaining_acorns month_acorns winter_months spring_acorns : ℕ)
  (h1 : total_acorns = 210)
  (h2 : parts_per_month = 3)
  (h3 : winter_months = 3)
  (h4 : remaining_acorns = 60)
  (h5 : month_acorns = total_acorns / winter_months)
  (h6 : spring_acorns = 30)
  (h7 : month_acorns - remaining_acorns = spring_acorns / parts_per_month) :
  parts_per_month = 3 :=
by
  sorry

end squirrel_divides_acorns_l174_174166


namespace power_function_passing_through_point_l174_174973

theorem power_function_passing_through_point :
  ∃ (α : ℝ), (2:ℝ)^α = 4 := by
  sorry

end power_function_passing_through_point_l174_174973


namespace cube_volume_l174_174024

theorem cube_volume (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
sorry

end cube_volume_l174_174024


namespace binom_10_3_l174_174332

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end binom_10_3_l174_174332


namespace cube_volume_from_surface_area_l174_174012

theorem cube_volume_from_surface_area (A : ℕ) (h1 : A = 864) : 
  ∃ V : ℕ, V = 1728 :=
by
  sorry

end cube_volume_from_surface_area_l174_174012


namespace solve_system_l174_174597

def system_of_equations : Prop :=
  ∃ (x y : ℝ), 2 * x - y = 6 ∧ x + 2 * y = -2 ∧ x = 2 ∧ y = -2

theorem solve_system : system_of_equations := by
  sorry

end solve_system_l174_174597


namespace analytical_expression_l174_174538

theorem analytical_expression (k : ℝ) (h : k ≠ 0) (x y : ℝ) (hx : x = 4) (hy : y = 6) 
  (eqn : y = k * x) : y = (3 / 2) * x :=
by {
  sorry
}

end analytical_expression_l174_174538


namespace binom_10_3_eq_120_l174_174315

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l174_174315


namespace integer_values_count_l174_174366

theorem integer_values_count (x : ℤ) :
  ∃ k, (∀ n : ℤ, (3 ≤ Real.sqrt (3 * n + 1) ∧ Real.sqrt (3 * n + 1) < 5) ↔ ((n = 3) ∨ (n = 4) ∨ (n = 5) ∨ (n = 6) ∨ (n = 7)) ∧ k = 5) :=
by
  sorry

end integer_values_count_l174_174366


namespace three_term_inequality_l174_174823

theorem three_term_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_abc : a * b * c = 1) :
  (a^3 / (a^3 + 2 * b^2)) + (b^3 / (b^3 + 2 * c^2)) + (c^3 / (c^3 + 2 * a^2)) ≥ 1 :=
by
  sorry

end three_term_inequality_l174_174823


namespace combination_10_3_eq_120_l174_174351

open Nat

theorem combination_10_3_eq_120 : (10.choose 3) = 120 := 
by
  sorry

end combination_10_3_eq_120_l174_174351


namespace total_hours_correct_l174_174917

/-- Definitions for the times each person has left to finish their homework. -/
noncomputable def Jacob_time : ℕ := 18
noncomputable def Greg_time : ℕ := Jacob_time - 6
noncomputable def Patrick_time : ℕ := 2 * Greg_time - 4

/-- Proving the total time left for Patrick, Greg, and Jacob to finish their homework. -/

theorem total_hours_correct : Jacob_time + Greg_time + Patrick_time = 50 := by
  sorry

end total_hours_correct_l174_174917


namespace inequality_inequality_holds_l174_174778

theorem inequality_inequality_holds (x y z : ℝ) : 
  (x^3 / (x^3 + 2 * y^2 * z)) + (y^3 / (y^3 + 2 * z^2 * x)) + (z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_inequality_holds_l174_174778


namespace find_m_l174_174549

theorem find_m (m : ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ) (h : a = (2, -4) ∧ b = (-3, m) ∧ (‖a‖ * ‖b‖ + (a.1 * b.1 + a.2 * b.2)) = 0) : m = 6 := 
by 
  sorry

end find_m_l174_174549


namespace max_C_usage_l174_174997

-- Definition of variables (concentration percentages and weights)
def A_conc := 3 / 100
def B_conc := 8 / 100
def C_conc := 11 / 100

def target_conc := 7 / 100
def total_weight := 100

def max_A := 50
def max_B := 70
def max_C := 60

-- Equation to satisfy
def conc_equation (x y : ℝ) : Prop :=
  C_conc * x + B_conc * y + A_conc * (total_weight - x - y) = target_conc * total_weight

-- Definition with given constraints
def within_constraints (x y : ℝ) : Prop :=
  x ≤ max_C ∧ y ≤ max_B ∧ (total_weight - x - y) ≤ max_A

-- The theorem that needs to be proved
theorem max_C_usage (x y : ℝ) : within_constraints x y ∧ conc_equation x y → x ≤ 50 :=
by
  sorry

end max_C_usage_l174_174997


namespace combination_10_3_l174_174294

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end combination_10_3_l174_174294


namespace inequality_proof_l174_174521

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  1 ≤ ((x + y) * (x^3 + y^3)) / (x^2 + y^2)^2 ∧ 
  ((x + y) * (x^3 + y^3)) / (x^2 + y^2)^2 ≤ 9 / 8 :=
sorry

end inequality_proof_l174_174521


namespace binom_10_3_eq_120_l174_174211

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end binom_10_3_eq_120_l174_174211


namespace quadratic_equation_with_means_l174_174559

theorem quadratic_equation_with_means (α β : ℝ) 
  (h_am : (α + β) / 2 = 8) 
  (h_gm : Real.sqrt (α * β) = 15) : 
  (Polynomial.X^2 - Polynomial.C (α + β) * Polynomial.X + Polynomial.C (α * β) = 0) := 
by
  have h1 : α + β = 16 := by linarith
  have h2 : α * β = 225 := by sorry
  rw [h1, h2]
  sorry

end quadratic_equation_with_means_l174_174559


namespace micheal_work_separately_40_days_l174_174428

-- Definitions based on the problem conditions
def work_complete_together (M A : ℕ) : Prop := (1/(M:ℝ) + 1/(A:ℝ) = 1/20)
def remaining_work_completed_by_adam (A : ℕ) : Prop := (1/(A:ℝ) = 1/40)

-- The theorem we want to prove
theorem micheal_work_separately_40_days (M A : ℕ) 
  (h1 : work_complete_together M A) 
  (h2 : remaining_work_completed_by_adam A) : 
  M = 40 := 
by 
  sorry  -- Placeholder for proof

end micheal_work_separately_40_days_l174_174428


namespace length_of_bridge_is_correct_l174_174049

noncomputable def length_of_bridge (length_of_train : ℕ) (time_in_seconds : ℕ) (speed_in_kmph : ℝ) : ℝ :=
  let speed_in_mps := speed_in_kmph * (1000 / 3600)
  time_in_seconds * speed_in_mps - length_of_train

theorem length_of_bridge_is_correct :
  length_of_bridge 150 40 42.3 = 320 := by
  sorry

end length_of_bridge_is_correct_l174_174049


namespace cube_volume_from_surface_area_l174_174000

theorem cube_volume_from_surface_area (S : ℝ) (h : S = 864) : ∃ V : ℝ, V = 1728 :=
by
  -- Assume surface area formula S = 6s^2, solve steps skipped and go directly to conclusion
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  use V
  sorry

end cube_volume_from_surface_area_l174_174000


namespace ratio_third_to_second_year_l174_174168

-- Define the yearly production of the apple tree
def first_year_production : Nat := 40
def second_year_production : Nat := 2 * first_year_production + 8
def total_production_three_years : Nat := 194
def third_year_production : Nat := total_production_three_years - (first_year_production + second_year_production)

-- Define the ratio calculation
def ratio (a b : Nat) : (Nat × Nat) := 
  let gcd_ab := Nat.gcd a b 
  (a / gcd_ab, b / gcd_ab)

-- Prove the ratio of the third year's production to the second year's production
theorem ratio_third_to_second_year : 
  ratio third_year_production second_year_production = (3, 4) :=
  sorry

end ratio_third_to_second_year_l174_174168


namespace probability_multiple_of_3_when_die_rolled_twice_l174_174162

theorem probability_multiple_of_3_when_die_rolled_twice :
  let total_outcomes := 36
  let favorable_outcomes := 12
  (12 / 36 : ℚ) = 1 / 3 :=
by
  sorry

end probability_multiple_of_3_when_die_rolled_twice_l174_174162


namespace binom_10_3_eq_120_l174_174224

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end binom_10_3_eq_120_l174_174224


namespace orthocenter_parallelogram_bisector_circumcircle_l174_174384

/-- Given H is the orthocenter of the acute triangle ABC,
G is such that the quadrilateral ABGH is a parallelogram,
and I is a point on the line GH such that AC bisects the segment HI.
If the line AC intersects the circumcircle of triangle GCI at points C and J,
then IJ = AH. -/
theorem orthocenter_parallelogram_bisector_circumcircle
  (A B C G H I J : Point)
  (h_orthocenter : is_orthocenter H (triangle.mk A B C))
  (h_parallelogram : is_parallelogram (quadrilateral.mk A B G H))
  (h_on_line : is_on_line I G H)
  (h_bisects : is_bisector (A, C) (H, I))
  (h_circumcircle : is_circumcircle (triangle.mk G C I) J (point_of) AC)
  : distance I J = distance A H := sorry

end orthocenter_parallelogram_bisector_circumcircle_l174_174384


namespace sum_of_solutions_l174_174772

theorem sum_of_solutions (s : Finset ℝ) :
  (∀ x ∈ s, |x^2 - 16 * x + 60| = 4) →
  s.sum id = 24 := 
by
  sorry

end sum_of_solutions_l174_174772


namespace binomial_10_3_eq_120_l174_174338

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l174_174338


namespace combination_10_3_eq_120_l174_174364

open Nat

theorem combination_10_3_eq_120 : (10.choose 3) = 120 := 
by
  sorry

end combination_10_3_eq_120_l174_174364


namespace gcd_condition_l174_174572

theorem gcd_condition (a b c : ℕ) (h1 : Nat.gcd a b = 255) (h2 : Nat.gcd a c = 855) :
  Nat.gcd b c = 15 :=
sorry

end gcd_condition_l174_174572


namespace James_age_after_x_years_l174_174717

variable (x : ℕ)
variable (Justin Jessica James : ℕ)

-- Define the conditions
theorem James_age_after_x_years 
  (H1 : Justin = 26) 
  (H2 : Jessica = Justin + 6) 
  (H3 : James = Jessica + 7)
  (H4 : James + 5 = 44) : 
  James + x = 39 + x := 
by 
  -- proof steps go here 
  sorry

end James_age_after_x_years_l174_174717


namespace complement_correct_l174_174721

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 5}
def complement_U (M: Set ℕ) (U: Set ℕ) := {x ∈ U | x ∉ M}

theorem complement_correct : complement_U M U = {3, 4, 6} :=
by 
  sorry

end complement_correct_l174_174721


namespace inequality_problem_l174_174071

theorem inequality_problem (x y a b : ℝ) (h1 : x > y) (h2 : y > 1) (h3 : 0 < a) (h4 : a < b) (h5 : b < 1) : (a ^ x < b ^ y) :=
by 
  sorry

end inequality_problem_l174_174071


namespace inequality_ge_one_l174_174804

theorem inequality_ge_one {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_ge_one_l174_174804


namespace water_park_children_l174_174604

theorem water_park_children (cost_adult cost_child total_cost : ℝ) (c : ℕ) 
  (h1 : cost_adult = 1)
  (h2 : cost_child = 0.75)
  (h3 : total_cost = 3.25) :
  c = 3 :=
by
  sorry

end water_park_children_l174_174604


namespace jail_time_weeks_l174_174756

theorem jail_time_weeks (days_protest : ℕ) (cities : ℕ) (arrests_per_day : ℕ)
  (days_pre_trial : ℕ) (half_week_sentence_days : ℕ) :
  days_protest = 30 →
  cities = 21 →
  arrests_per_day = 10 →
  days_pre_trial = 4 →
  half_week_sentence_days = 7 →
  (21 * 30 * 10 * (4 + 7)) / 7 = 9900 :=
by
  intros h_days_protest h_cities h_arrests_per_day h_days_pre_trial h_half_week_sentence_days
  rw [h_days_protest, h_cities, h_arrests_per_day, h_days_pre_trial, h_half_week_sentence_days]
  exact sorry

end jail_time_weeks_l174_174756


namespace factorial_divisible_by_power_of_two_iff_l174_174437

theorem factorial_divisible_by_power_of_two_iff (n : ℕ) :
  (nat.factorial n) % (2^(n-1)) = 0 ↔ ∃ k : ℕ, n = 2^k := 
by
  sorry

end factorial_divisible_by_power_of_two_iff_l174_174437


namespace binom_10_3_eq_120_l174_174322

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l174_174322


namespace volume_removed_percentage_l174_174046

noncomputable def volume_of_box (length width height : ℝ) : ℝ := 
  length * width * height

noncomputable def volume_of_cube (side : ℝ) : ℝ := 
  side ^ 3

noncomputable def volume_removed (length width height side : ℝ) : ℝ :=
  8 * (volume_of_cube side)

noncomputable def percentage_removed (length width height side : ℝ) : ℝ :=
  (volume_removed length width height side) / (volume_of_box length width height) * 100

theorem volume_removed_percentage :
  percentage_removed 20 15 12 4 = 14.22 := 
by
  sorry

end volume_removed_percentage_l174_174046


namespace collinear_points_m_equals_4_l174_174704

theorem collinear_points_m_equals_4 (m : ℝ)
  (h1 : (3 - 12) / (1 - -2) = (-6 - 12) / (m - -2)) : m = 4 :=
by
  sorry

end collinear_points_m_equals_4_l174_174704


namespace product_of_x_and_y_l174_174170

theorem product_of_x_and_y (x y a b : ℝ)
  (h1 : x = b^(3/2))
  (h2 : y = a)
  (h3 : a + a = b^2)
  (h4 : y = b)
  (h5 : a + a = b^(3/2))
  (h6 : b = 3) :
  x * y = 9 * Real.sqrt 3 := 
  sorry

end product_of_x_and_y_l174_174170


namespace solve_cubic_fraction_l174_174736

noncomputable def problem_statement (x : ℝ) :=
  (x = (-(3:ℝ) + Real.sqrt 13) / 4) ∨ (x = (-(3:ℝ) - Real.sqrt 13) / 4)

theorem solve_cubic_fraction (x : ℝ) (h : (x^3 + 2*x^2 + 3*x + 5) / (x + 2) = x + 4) : 
  problem_statement x :=
by
  sorry

end solve_cubic_fraction_l174_174736


namespace chord_line_equation_l174_174860

theorem chord_line_equation (x y : ℝ) :
  (∃ (x1 y1 x2 y2 : ℝ), y1^2 = -8 * x1 ∧ y2^2 = -8 * x2 ∧ (x1 + x2) / 2 = -1 ∧ (y1 + y2) / 2 = 1 ∧ y - 1 = -4 * (x + 1)) →
  4 * x + y + 3 = 0 :=
by
  sorry

end chord_line_equation_l174_174860


namespace least_number_of_trees_l174_174979

theorem least_number_of_trees :
  ∃ n : ℕ, (n % 5 = 0) ∧ (n % 6 = 0) ∧ (n % 7 = 0) ∧ n = 210 :=
by
  sorry

end least_number_of_trees_l174_174979


namespace jenny_sold_192_packs_l174_174095

-- Define the conditions
def boxes_sold : ℝ := 24.0
def packs_per_box : ℝ := 8.0

-- The total number of packs sold
def total_packs_sold : ℝ := boxes_sold * packs_per_box

-- Proof statement that total packs sold equals 192.0
theorem jenny_sold_192_packs : total_packs_sold = 192.0 :=
by
  sorry

end jenny_sold_192_packs_l174_174095


namespace binomial_10_3_eq_120_l174_174337

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l174_174337


namespace binomial_coefficient_10_3_l174_174277

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l174_174277


namespace binom_10_3_l174_174325

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end binom_10_3_l174_174325


namespace find_f_neg1_l174_174421

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 2^x + 2*x - 1 else -2^(-x) + 2*x + 1

theorem find_f_neg1 : f (-1) = -3 :=
by
  -- The proof is omitted.
  sorry

end find_f_neg1_l174_174421


namespace solve_abs_eq_l174_174149

theorem solve_abs_eq (x : ℝ) (h : |x + 2| = |x - 3|) : x = 1 / 2 :=
sorry

end solve_abs_eq_l174_174149


namespace maximum_value_of_k_l174_174864

noncomputable def max_k (m : ℝ) : ℝ := 
  if 0 < m ∧ m < 1 / 2 then 
    1 / m + 2 / (1 - 2 * m) 
  else 
    0

theorem maximum_value_of_k : ∀ m : ℝ, (0 < m ∧ m < 1 / 2) → (∀ k : ℝ, (1 / m + 2 / (1 - 2 * m) ≥ k) → k ≤ 8) :=
  sorry

end maximum_value_of_k_l174_174864


namespace f_one_value_l174_174685

noncomputable def f (x : ℝ) : ℝ := sorry

axiom h_f_defined : ∀ x, x > 0 → ∃ y, f x = y
axiom h_f_strict_increasing : ∀ x y, 0 < x → 0 < y → x < y → f x < f y
axiom h_f_eq : ∀ x, x > 0 → f x * f (f x + 1/x) = 1

theorem f_one_value : f 1 = (1 + Real.sqrt 5) / 2 := 
by
  sorry

end f_one_value_l174_174685


namespace binomial_10_3_l174_174306

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end binomial_10_3_l174_174306


namespace Alyssa_puppies_l174_174844

theorem Alyssa_puppies (initial_puppies give_away_puppies : ℕ) (h_initial : initial_puppies = 12) (h_give_away : give_away_puppies = 7) :
  initial_puppies - give_away_puppies = 5 :=
by
  sorry

end Alyssa_puppies_l174_174844


namespace factorial_power_of_two_iff_power_of_two_l174_174436

-- Assuming n is a positive integer
variable {n : ℕ} (h : n > 0)

theorem factorial_power_of_two_iff_power_of_two :
  (∃ k : ℕ, n = 2^k ) ↔ ∃ m : ℕ, 2^(n-1) ∣ n! :=
by {
  sorry
}

end factorial_power_of_two_iff_power_of_two_l174_174436


namespace dale_slices_of_toast_l174_174988

theorem dale_slices_of_toast
  (slice_cost : ℤ) (egg_cost : ℤ)
  (dale_eggs : ℤ) (andrew_slices : ℤ) (andrew_eggs : ℤ)
  (total_cost : ℤ)
  (cost_eq : slice_cost = 1)
  (egg_cost_eq : egg_cost = 3)
  (dale_eggs_eq : dale_eggs = 2)
  (andrew_slices_eq : andrew_slices = 1)
  (andrew_eggs_eq : andrew_eggs = 2)
  (total_cost_eq : total_cost = 15)
  :
  ∃ T : ℤ, (slice_cost * T + egg_cost * dale_eggs) + (slice_cost * andrew_slices + egg_cost * andrew_eggs) = total_cost ∧ T = 2 :=
by
  sorry

end dale_slices_of_toast_l174_174988


namespace opposite_of_neg3_squared_l174_174749

theorem opposite_of_neg3_squared : -(-3^2) = 9 :=
by
  sorry

end opposite_of_neg3_squared_l174_174749


namespace stating_martha_painting_time_l174_174427

/-- 
  Theorem stating the time it takes for Martha to paint the kitchen is 42 hours.
-/
theorem martha_painting_time :
  let width1 := 12
  let width2 := 16
  let height := 10
  let area_pair1 := 2 * width1 * height
  let area_pair2 := 2 * width2 * height
  let total_area := area_pair1 + area_pair2
  let coats := 3
  let total_paint_area := total_area * coats
  let painting_speed := 40
  let time_required := total_paint_area / painting_speed
  time_required = 42 := by
    -- Since we are asked not to provide the proof steps, we use sorry to skip the proof.
    sorry

end stating_martha_painting_time_l174_174427


namespace probability_no_consecutive_tails_probability_no_consecutive_tails_in_five_tosses_l174_174914

def countWays (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else countWays (n - 1) + countWays (n - 2)

theorem probability_no_consecutive_tails : countWays 5 = 13 :=
by
  sorry

theorem probability_no_consecutive_tails_in_five_tosses : 
  (countWays 5) / (2^5 : ℕ) = 13 / 32 :=
by
  sorry

end probability_no_consecutive_tails_probability_no_consecutive_tails_in_five_tosses_l174_174914


namespace circle_diameter_l174_174934

theorem circle_diameter (A : ℝ) (h : A = 64 * Real.pi) : ∃ (d : ℝ), d = 16 :=
by
  sorry

end circle_diameter_l174_174934


namespace action_figures_more_than_books_proof_l174_174096

-- Definitions for the conditions
def books := 3
def action_figures_initial := 4
def action_figures_added := 2

-- Definition for the total action figures
def action_figures_total := action_figures_initial + action_figures_added

-- Definition for the number difference
def action_figures_more_than_books := action_figures_total - books

-- Proof statement
theorem action_figures_more_than_books_proof : action_figures_more_than_books = 3 :=
by
  sorry

end action_figures_more_than_books_proof_l174_174096


namespace inequality_solution_l174_174378

theorem inequality_solution (x : ℝ) : x^3 - 12 * x^2 > -36 * x ↔ x ∈ (Set.Ioo 0 6) ∪ (Set.Ioi 6) :=
by
  sorry

end inequality_solution_l174_174378


namespace probability_xavier_yvonne_not_zelda_wendell_l174_174731

theorem probability_xavier_yvonne_not_zelda_wendell
  (P_Xavier_solves : ℚ)
  (P_Yvonne_solves : ℚ)
  (P_Zelda_solves : ℚ)
  (P_Wendell_solves : ℚ) :
  P_Xavier_solves = 1/4 →
  P_Yvonne_solves = 1/3 →
  P_Zelda_solves = 5/8 →
  P_Wendell_solves = 1/2 →
  (P_Xavier_solves * P_Yvonne_solves * (1 - P_Zelda_solves) * (1 - P_Wendell_solves)) = 1/64 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp
  sorry

end probability_xavier_yvonne_not_zelda_wendell_l174_174731


namespace picnic_attendance_l174_174138

theorem picnic_attendance (L x : ℕ) (h1 : L + x = 2015) (h2 : L - (x - 1) = 4) : x = 1006 := 
by
  sorry

end picnic_attendance_l174_174138


namespace notebook_cost_l174_174489

-- Define the cost of notebook (n) and cost of cover (c)
variables (n c : ℝ)

-- Given conditions as definitions
def condition1 := n + c = 3.50
def condition2 := n = c + 2

-- Prove that the cost of the notebook (n) is 2.75
theorem notebook_cost (h1 : condition1 n c) (h2 : condition2 n c) : n = 2.75 := 
by
  sorry

end notebook_cost_l174_174489


namespace sum_possible_values_l174_174904

theorem sum_possible_values (x y : ℝ) (h : x * y - x / y^3 - y / x^3 = 4) :
  (x - 2) * (y - 2) = 4 ∨ (x - 2) * (y - 2) = 0 → (4 + 0 = 4) :=
by
  sorry

end sum_possible_values_l174_174904


namespace john_spent_on_candy_l174_174989

theorem john_spent_on_candy (M : ℝ) 
  (h1 : M = 29.999999999999996)
  (h2 : 1/5 + 1/3 + 1/10 = 19/30) :
  (11 / 30) * M = 11 :=
by {
  sorry
}

end john_spent_on_candy_l174_174989


namespace cube_volume_l174_174025

theorem cube_volume (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
sorry

end cube_volume_l174_174025


namespace parallel_line_slope_y_intercept_l174_174467

theorem parallel_line_slope_y_intercept (x y : ℝ) (h : 3 * x - 6 * y = 12) :
  ∃ (m b : ℝ), m = 1 / 2 ∧ b = -2 := 
by { sorry }

end parallel_line_slope_y_intercept_l174_174467


namespace coefficients_sum_even_odd_split_sum_binomial_coeff_sum_l174_174383

noncomputable def problem_statement (a : ℕ → ℤ) (x : ℤ) :=
  Σ i in finset.range 8, a i * x ^ i = (1 - 2 * x) ^ 7

theorem coefficients_sum (a : ℕ → ℤ) : (Σ x in finset.range 8, a x) = 0 :=
sorry

theorem even_odd_split_sum (a : ℕ → ℤ) :
  (Σ i in finset.range 4, a (2 * i)) = 129 ∧ (Σ i in finset.range 4, a (2 * i + 1)) = -128 :=
sorry

theorem binomial_coeff_sum : (Σ i in finset.range 8, Nat.choose 7 i) = 128 :=
sorry

end coefficients_sum_even_odd_split_sum_binomial_coeff_sum_l174_174383


namespace trigonometric_expression_simplification_l174_174994

theorem trigonometric_expression_simplification
  (α : ℝ) 
  (hα : α = 49 * Real.pi / 48) :
  4 * (Real.sin α ^ 3 * Real.cos (3 * α) + 
       Real.cos α ^ 3 * Real.sin (3 * α)) * 
  Real.cos (4 * α) = 0.75 := 
by 
  sorry

end trigonometric_expression_simplification_l174_174994


namespace binomial_10_3_l174_174299

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end binomial_10_3_l174_174299


namespace find_min_sum_of_squares_l174_174906

open Real

theorem find_min_sum_of_squares
  (x1 x2 x3 : ℝ)
  (h1 : 0 < x1)
  (h2 : 0 < x2)
  (h3 : 0 < x3)
  (h4 : 2 * x1 + 4 * x2 + 6 * x3 = 120) :
  x1^2 + x2^2 + x3^2 >= 350 :=
sorry

end find_min_sum_of_squares_l174_174906


namespace meaningful_if_and_only_if_l174_174887

theorem meaningful_if_and_only_if (x : ℝ) : (∃ y : ℝ, y = (1 / (x - 1))) ↔ x ≠ 1 :=
by 
  sorry

end meaningful_if_and_only_if_l174_174887


namespace cube_volume_of_surface_area_l174_174018

-- Define the condition: the surface area S is 864 square units
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- The proof problem: Given that the surface area of a cube is 864 square units,
-- prove that the volume of the cube is 1728 cubic units
theorem cube_volume_of_surface_area (S : ℝ) (hS : S = 864) : 
  ∃ V : ℝ, V = 1728 ∧ ∃ s : ℝ, surface_area s = S ∧ V = s^3 :=
by 
  sorry

end cube_volume_of_surface_area_l174_174018


namespace warriors_can_defeat_dragon_l174_174616

theorem warriors_can_defeat_dragon (n : ℕ) (h : n = 20^20) :
  (∀ n, n % 2 = 0 ∨ n % 3 = 0) → (∃ m, m = 0) := 
sorry

end warriors_can_defeat_dragon_l174_174616


namespace total_ages_l174_174507

variable (Bill_age Caroline_age : ℕ)
variable (h1 : Bill_age = 2 * Caroline_age - 1) (h2 : Bill_age = 17)

theorem total_ages : Bill_age + Caroline_age = 26 :=
by
  sorry

end total_ages_l174_174507


namespace binom_10_3_eq_120_l174_174318

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l174_174318


namespace max_value_abcd_l174_174076

-- Define the digits and constraints on them
def distinct_digits (a b c d e : ℕ) : Prop := 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

-- Encode the given problem as a Lean theorem
theorem max_value_abcd (a b c d e : ℕ) 
  (h₀ : distinct_digits a b c d e)
  (h₁ : 0 ≤ a ∧ a ≤ 9) 
  (h₂ : 0 ≤ b ∧ b ≤ 9) 
  (h₃ : 0 ≤ c ∧ c ≤ 9) 
  (h₄ : 0 ≤ d ∧ d ≤ 9)
  (h₅ : 0 ≤ e ∧ e ≤ 9)
  (h₆ : e ≠ 0)
  (h₇ : a * 1000 + b * 100 + c * 10 + d = (a * 100 + a * 10 + d) * e) :
  a * 1000 + b * 100 + c * 10 + d = 3015 :=
by {
  sorry
}

end max_value_abcd_l174_174076


namespace smallest_positive_y_l174_174368

theorem smallest_positive_y (y : ℕ) (h : 42 * y + 8 ≡ 4 [MOD 24]) : y = 2 :=
sorry

end smallest_positive_y_l174_174368


namespace inequality_proof_l174_174786

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 := 
by 
  sorry

end inequality_proof_l174_174786


namespace sin_A_plus_B_lt_sin_A_add_sin_B_l174_174092

variable {A B : ℝ}
variable (A_pos : 0 < A)
variable (B_pos : 0 < B)
variable (AB_sum_pi : A + B < π)

theorem sin_A_plus_B_lt_sin_A_add_sin_B (a b : ℝ) (h1 : a = Real.sin (A + B)) (h2 : b = Real.sin A + Real.sin B) : 
  a < b := by
  sorry

end sin_A_plus_B_lt_sin_A_add_sin_B_l174_174092


namespace binomial_10_3_eq_120_l174_174207

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l174_174207


namespace binomial_10_3_l174_174301

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end binomial_10_3_l174_174301


namespace sum_of_digits_next_perfect_square_222_l174_174469

-- Define the condition for the perfect square that begins with "222"
def starts_with_222 (n: ℕ) : Prop :=
  n / 10^3 = 222

-- Define the sum of the digits function
def sum_of_digits (n: ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Statement for the Lean 4 statement: 
-- Prove that the sum of the digits of the next perfect square that starts with "222" is 18
theorem sum_of_digits_next_perfect_square_222 : sum_of_digits (492 ^ 2) = 18 :=
by
  sorry -- Proof omitted

end sum_of_digits_next_perfect_square_222_l174_174469


namespace inequality_abc_l174_174925

variable (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
variable (cond : a + b + c = (1/a) + (1/b) + (1/c))

theorem inequality_abc : a + b + c ≥ 3 / (a * b * c) :=
by
  sorry

end inequality_abc_l174_174925


namespace inequality_holds_l174_174816

theorem inequality_holds (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by
  sorry

end inequality_holds_l174_174816


namespace fewest_seats_occupied_l174_174457

def min_seats_occupied (N : ℕ) : ℕ :=
  if h : N % 4 = 0 then (N / 2) else (N / 2) + 1

theorem fewest_seats_occupied (N : ℕ) (h : N = 150) : min_seats_occupied N = 74 := by
  sorry

end fewest_seats_occupied_l174_174457


namespace cube_volume_from_surface_area_l174_174009

theorem cube_volume_from_surface_area (A : ℕ) (h1 : A = 864) : 
  ∃ V : ℕ, V = 1728 :=
by
  sorry

end cube_volume_from_surface_area_l174_174009


namespace inequality_ge_one_l174_174809

theorem inequality_ge_one {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_ge_one_l174_174809


namespace cube_volume_of_surface_area_l174_174013

-- Define the condition: the surface area S is 864 square units
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- The proof problem: Given that the surface area of a cube is 864 square units,
-- prove that the volume of the cube is 1728 cubic units
theorem cube_volume_of_surface_area (S : ℝ) (hS : S = 864) : 
  ∃ V : ℝ, V = 1728 ∧ ∃ s : ℝ, surface_area s = S ∧ V = s^3 :=
by 
  sorry

end cube_volume_of_surface_area_l174_174013


namespace find_n_l174_174671

theorem find_n :
  ∃ (n : ℤ), -180 ≤ n ∧ n ≤ 180 ∧ sin (n * Real.pi / 180) = cos (510 * Real.pi / 180) → n = -60 :=
by
  sorry

end find_n_l174_174671


namespace polynomial_evaluation_l174_174371

theorem polynomial_evaluation (x : ℝ) (h1 : x^2 - 3 * x - 10 = 0) (h2 : 0 < x) : 
  x^3 - 3 * x^2 - 10 * x + 5 = 5 :=
sorry

end polynomial_evaluation_l174_174371


namespace calc_j_inverse_l174_174422

noncomputable def i : ℂ := Complex.I  -- Equivalent to i^2 = -1 definition of complex imaginary unit
noncomputable def j : ℂ := i + 1      -- Definition of j

theorem calc_j_inverse :
  (j - j⁻¹)⁻¹ = (-3 * i + 1) / 5 :=
by 
  -- The statement here only needs to declare the equivalence, 
  -- without needing the proof
  sorry

end calc_j_inverse_l174_174422


namespace binom_10_3_l174_174237

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end binom_10_3_l174_174237


namespace ratio_of_boys_to_girls_l174_174949

-- Variables for the number of boys, girls, and teachers
variables (B G T : ℕ)

-- Conditions from the problem
def number_of_girls := G = 60
def number_of_teachers := T = (20 * B) / 100
def total_people := B + G + T = 114

-- Proving the ratio of boys to girls is 3:4 given the conditions
theorem ratio_of_boys_to_girls 
  (hG : number_of_girls G)
  (hT : number_of_teachers B T)
  (hTotal : total_people B G T) :
  B / 15 = 3 ∧ G / 15 = 4 :=
by {
  sorry
}

end ratio_of_boys_to_girls_l174_174949


namespace product_pqr_l174_174083

/-- Mathematical problem statement -/
theorem product_pqr (p q r : ℤ) (hp: p ≠ 0) (hq: q ≠ 0) (hr: r ≠ 0)
  (h1 : p + q + r = 36)
  (h2 : (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / r + 540 / (p * q * r) = 1) :
  p * q * r = 864 :=
sorry

end product_pqr_l174_174083


namespace diameter_of_circle_l174_174766

theorem diameter_of_circle (A : ℝ) (h : A = 100 * Real.pi) : ∃ d : ℝ, d = 20 :=
by
  sorry

end diameter_of_circle_l174_174766


namespace range_of_a_range_of_m_l174_174683

-- Definition of proposition p: Equation has real roots
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - a * x + a + 3 = 0

-- Definition of proposition q: m - 1 <= a <= m + 1
def q (m a : ℝ) : Prop := m - 1 ≤ a ∧ a ≤ m + 1

-- Part (I): Range of a when ¬p is true
theorem range_of_a (a : ℝ) (hp : ¬ p a) : -2 < a ∧ a < 6 :=
sorry

-- Part (II): Range of m when p is a necessary but not sufficient condition for q
theorem range_of_m (m : ℝ) (hnp : ∀ a, q m a → p a) (hns : ∃ a, q m a ∧ ¬p a) : m ≤ -3 ∨ m ≥ 7 :=
sorry

end range_of_a_range_of_m_l174_174683


namespace relationship_among_abcd_l174_174880

theorem relationship_among_abcd (a b c d : ℝ) 
  (h1 : a < b) 
  (h2 : d < c) 
  (h3 : (c - a) * (c - b) < 0) 
  (h4 : (d - a) * (d - b) > 0) : 
  d < a ∧ a < c ∧ c < b := 
by
  sorry

end relationship_among_abcd_l174_174880


namespace wendy_distance_difference_l174_174763

-- Defining the distances ran and walked by Wendy
def distance_ran : ℝ := 19.83
def distance_walked : ℝ := 9.17

-- The theorem to prove the difference in distance
theorem wendy_distance_difference : distance_ran - distance_walked = 10.66 := by
  -- Proof goes here
  sorry

end wendy_distance_difference_l174_174763


namespace inequality_proof_l174_174790

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 := 
by 
  sorry

end inequality_proof_l174_174790


namespace binom_10_3_eq_120_l174_174212

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end binom_10_3_eq_120_l174_174212


namespace binomial_10_3_l174_174296

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end binomial_10_3_l174_174296


namespace perpendicular_lines_slope_l174_174143

theorem perpendicular_lines_slope (a : ℝ) (h1 :  a * (a + 2) = -1) : a = -1 :=
by 
-- Perpendicularity condition given
sorry

end perpendicular_lines_slope_l174_174143


namespace percentage_volume_removed_l174_174047

/-- A solid box has dimensions 20 cm by 15 cm by 12 cm. 
A new solid is formed by removing a cube of 4 cm on a side from each of the eight corners. 
We need to prove that the percentage of the original volume removed is approximately 14.22%. -/
theorem percentage_volume_removed :
  let volume_original_box := 20 * 15 * 12
  let volume_one_cube := 4^3
  let total_volume_removed := 8 * volume_one_cube
  let percentage_removed := (total_volume_removed : ℚ) / volume_original_box * 100
  percentage_removed ≈ 14.22 := sorry

end percentage_volume_removed_l174_174047


namespace overall_percentage_increase_correct_l174_174418

def initial_salary : ℕ := 60
def first_raise_salary : ℕ := 90
def second_raise_salary : ℕ := 120
def gym_deduction : ℕ := 10

def final_salary : ℕ := second_raise_salary - gym_deduction
def salary_difference : ℕ := final_salary - initial_salary
def percentage_increase : ℚ := (salary_difference : ℚ) / initial_salary * 100

theorem overall_percentage_increase_correct :
  percentage_increase = 83.33 := by
  sorry

end overall_percentage_increase_correct_l174_174418


namespace hyperbola_eccentricity_l174_174874

theorem hyperbola_eccentricity : 
  let a := Real.sqrt 2
  let b := 1
  let c := Real.sqrt (a^2 + b^2)
  (c / a) = Real.sqrt 6 / 2 := 
by
  sorry

end hyperbola_eccentricity_l174_174874


namespace total_cost_l174_174714

def daily_rental_cost : ℝ := 25
def cost_per_mile : ℝ := 0.20
def duration_days : ℕ := 4
def distance_miles : ℕ := 400

theorem total_cost 
: (daily_rental_cost * duration_days + cost_per_mile * distance_miles) = 180 := 
by
  sorry

end total_cost_l174_174714


namespace max_n_for_Sn_neg_l174_174720

noncomputable def Sn (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  (n * (a 1 + a n)) / 2

theorem max_n_for_Sn_neg (a : ℕ → ℝ) (h1 : ∀ n : ℕ, (n + 1) * Sn n a < n * Sn (n + 1) a)
  (h2 : a 8 / a 7 < -1) :
  ∀ n : ℕ, S_13 < 0 ∧ S_14 > 0 →
  ∀ m : ℕ, m > 13 → Sn m a ≥ 0 :=
sorry

end max_n_for_Sn_neg_l174_174720


namespace binom_10_3_eq_120_l174_174222

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end binom_10_3_eq_120_l174_174222


namespace problem1_problem2_l174_174398

noncomputable def f (a x : ℝ) : ℝ := x^2 + 2 * a * x - 3

theorem problem1 (a : ℝ) (h : f a (a + 1) - f a a = 9) : a = 2 :=
by sorry

theorem problem2 (a : ℝ) (h : ∃ x, f a x = -4 ∧ ∀ y, f a y ≥ -4) : a = 1 ∨ a = -1 :=
by sorry

end problem1_problem2_l174_174398


namespace cube_volume_from_surface_area_l174_174005

theorem cube_volume_from_surface_area (A : ℕ) (h1 : A = 864) : 
  ∃ V : ℕ, V = 1728 :=
by
  sorry

end cube_volume_from_surface_area_l174_174005


namespace Kelly_baking_powder_difference_l174_174625

theorem Kelly_baking_powder_difference : 0.4 - 0.3 = 0.1 :=
by 
  -- sorry is a placeholder for a proof
  sorry

end Kelly_baking_powder_difference_l174_174625


namespace binomial_10_3_eq_120_l174_174343

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l174_174343


namespace profit_difference_l174_174649

-- Define the initial investments
def investment_A : ℚ := 8000
def investment_B : ℚ := 10000
def investment_C : ℚ := 12000

-- Define B's profit share
def profit_B : ℚ := 1700

-- Prove that the difference between A and C's profit shares is Rs. 680
theorem profit_difference (investment_A investment_B investment_C profit_B: ℚ) (hA : investment_A = 8000) (hB : investment_B = 10000) (hC : investment_C = 12000) (pB : profit_B = 1700) :
    let ratio_A : ℚ := 4
    let ratio_B : ℚ := 5
    let ratio_C : ℚ := 6
    let part_value : ℚ := profit_B / ratio_B
    let profit_A : ℚ := ratio_A * part_value
    let profit_C : ℚ := ratio_C * part_value
    profit_C - profit_A = 680 := 
by
  sorry

end profit_difference_l174_174649


namespace three_digit_numbers_satisfy_condition_l174_174857

theorem three_digit_numbers_satisfy_condition : 
  ∃ (x y z : ℕ), 
    1 ≤ x ∧ x ≤ 9 ∧ 
    0 ≤ y ∧ y ≤ 9 ∧ 
    0 ≤ z ∧ z ≤ 9 ∧ 
    x + y + z = (10 * x + y) - (10 * y + z) ∧ 
    (100 * x + 10 * y + z = 209 ∨ 
     100 * x + 10 * y + z = 428 ∨ 
     100 * x + 10 * y + z = 647 ∨ 
     100 * x + 10 * y + z = 866 ∨ 
     100 * x + 10 * y + z = 214 ∨ 
     100 * x + 10 * y + z = 433 ∨ 
     100 * x + 10 * y + z = 652 ∨ 
     100 * x + 10 * y + z = 871) := sorry

end three_digit_numbers_satisfy_condition_l174_174857


namespace find_p_l174_174154

theorem find_p (m n p : ℝ) 
  (h1 : m = (n / 2) - (2 / 5)) 
  (h2 : m + p = ((n + 4) / 2) - (2 / 5)) :
  p = 2 :=
sorry

end find_p_l174_174154


namespace sqrt_factorial_multiplication_squared_l174_174470

theorem sqrt_factorial_multiplication_squared :
  (Real.sqrt (Nat.factorial 4 * Nat.factorial 3))^2 = 144 := 
by
  sorry

end sqrt_factorial_multiplication_squared_l174_174470


namespace sum_of_fifth_powers_l174_174958

theorem sum_of_fifth_powers (a b c d : ℝ) (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 := sorry

end sum_of_fifth_powers_l174_174958


namespace find_cos_squared_y_l174_174035

noncomputable def α : ℝ := Real.arccos (-3 / 7)

def arithmetic_progression (a b c : ℝ) : Prop :=
  2 * b = a + c

def transformed_arithmetic_progression (a b c : ℝ) : Prop :=
  14 / Real.cos b = 1 / Real.cos a + 1 / Real.cos c

theorem find_cos_squared_y (x y z : ℝ)
  (h1 : arithmetic_progression x y z)
  (h2 : transformed_arithmetic_progression x y z)
  (hα : 2 * α = z - x) : Real.cos y ^ 2 = 10 / 13 :=
by
  sorry

end find_cos_squared_y_l174_174035


namespace fixed_point_exists_trajectory_M_trajectory_equation_l174_174395

variable (m : ℝ)
def line_l (x y : ℝ) : Prop := 2 * x + (1 + m) * y + 2 * m = 0
def point_P (x y : ℝ) : Prop := x = -1 ∧ y = 0

theorem fixed_point_exists :
  ∃ x y : ℝ, (line_l m x y ∧ x = 1 ∧ y = -2) :=
by
  sorry

theorem trajectory_M :
  ∃ (M: ℝ × ℝ), (line_l m M.1 M.2 ∧ M = (0, -1)) :=
by
  sorry

theorem trajectory_equation (x y : ℝ) :
  ∃ (x y : ℝ), (x + 1) ^ 2  + y ^ 2 = 2 :=
by
  sorry

end fixed_point_exists_trajectory_M_trajectory_equation_l174_174395


namespace combination_10_3_eq_120_l174_174251

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end combination_10_3_eq_120_l174_174251


namespace arthur_num_hamburgers_on_first_day_l174_174987

theorem arthur_num_hamburgers_on_first_day (H D : ℕ) (hamburgers_1 hamburgers_2 : ℕ) (hotdogs_1 hotdogs_2 : ℕ)
  (h1 : hamburgers_1 * H + hotdogs_1 * D = 10)
  (h2 : hamburgers_2 * H + hotdogs_2 * D = 7)
  (hprice : D = 1)
  (h1_hotdogs : hotdogs_1 = 4)
  (h2_hotdogs : hotdogs_2 = 3) : 
  hamburgers_1 = 1 := 
by
  sorry

end arthur_num_hamburgers_on_first_day_l174_174987


namespace filling_tank_with_pipes_l174_174837

theorem filling_tank_with_pipes :
  let Ra := 1 / 70
  let Rb := 2 * Ra
  let Rc := 2 * Rb
  let Rtotal := Ra + Rb + Rc
  Rtotal = 1 / 10 →  -- Given the combined rate fills the tank in 10 hours
  3 = 3 :=  -- Number of pipes used to fill the tank
by
  intros Ra Rb Rc Rtotal h
  simp [Ra, Rb, Rc] at h
  sorry

end filling_tank_with_pipes_l174_174837


namespace combination_10_3_l174_174292

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end combination_10_3_l174_174292


namespace jack_total_damage_costs_l174_174414

def cost_per_tire := 250
def number_of_tires := 3
def cost_of_window := 700

def total_cost_of_tires := cost_per_tire * number_of_tires
def total_cost_of_damages := total_cost_of_tires + cost_of_window

theorem jack_total_damage_costs : total_cost_of_damages = 1450 := 
by
  -- Using the definitions provided
  -- total_cost_of_tires = 250 * 3 = 750
  -- total_cost_of_damages = 750 + 700 = 1450
  sorry

end jack_total_damage_costs_l174_174414


namespace D_is_largest_l174_174954

def D := (2008 / 2007) + (2008 / 2009)
def E := (2008 / 2009) + (2010 / 2009)
def F := (2009 / 2008) + (2009 / 2010) - (1 / 2009)

theorem D_is_largest : D > E ∧ D > F := by
  sorry

end D_is_largest_l174_174954


namespace find_width_of_floor_l174_174981

variable (w : ℝ) -- width of the floor

theorem find_width_of_floor (h1 : w - 4 > 0) (h2 : 10 - 4 > 0) 
                            (area_rug : (10 - 4) * (w - 4) = 24) : w = 8 :=
by
  sorry

end find_width_of_floor_l174_174981


namespace vector_magnitude_l174_174907

noncomputable def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
(v1.1 + v2.1, v1.2 + v2.2)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem vector_magnitude :
  ∀ (x y : ℝ), let a := (x, 2)
               let b := (1, y)
               let c := (2, -6)
               (a.1 * c.1 + a.2 * c.2 = 0) →
               (b.1 * (-c.2) - b.2 * c.1 = 0) →
               magnitude (vec_add a b) = 5 * Real.sqrt 2 :=
by
  intros x y a b c h₁ h₂
  let a := (x, 2)
  let b := (1, y)
  let c := (2, -6)
  sorry

end vector_magnitude_l174_174907


namespace binom_10_3_eq_120_l174_174221

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end binom_10_3_eq_120_l174_174221


namespace find_digit_D_l174_174462

def is_digit (n : ℕ) : Prop := n < 10

theorem find_digit_D (A B C D : ℕ) (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D) (h4 : B ≠ C)
  (h5 : B ≠ D) (h6 : C ≠ D) (h7 : is_digit A) (h8 : is_digit B) (h9 : is_digit C) (h10 : is_digit D) :
  (1000 * A + 100 * B + 10 * C + D) * 2 = 5472 → D = 6 := 
by
  sorry

end find_digit_D_l174_174462


namespace dimitri_weekly_calories_l174_174514

-- Define the calories for each type of burger
def calories_burger_a : ℕ := 350
def calories_burger_b : ℕ := 450
def calories_burger_c : ℕ := 550

-- Define the daily consumption of each type of burger
def daily_consumption_a : ℕ := 2
def daily_consumption_b : ℕ := 1
def daily_consumption_c : ℕ := 3

-- Define the duration in days
def duration_in_days : ℕ := 7

-- Define the total number of calories Dimitri consumes in a week
noncomputable def total_weekly_calories : ℕ :=
  (daily_consumption_a * calories_burger_a +
   daily_consumption_b * calories_burger_b +
   daily_consumption_c * calories_burger_c) * duration_in_days

theorem dimitri_weekly_calories : total_weekly_calories = 19600 := 
by 
  sorry

end dimitri_weekly_calories_l174_174514


namespace quadratic_has_two_distinct_real_roots_l174_174369

theorem quadratic_has_two_distinct_real_roots (m : ℝ) :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (∀ p : ℝ, (p = x1 ∨ p = x2) → (p ^ 2 + (4 * m + 1) * p + m = 0)) :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l174_174369


namespace weaving_additional_yards_l174_174842

theorem weaving_additional_yards {d : ℝ} :
  (∃ d : ℝ, (30 * 5 + (30 * 29) / 2 * d = 390) → d = 16 / 29) :=
sorry

end weaving_additional_yards_l174_174842


namespace inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8_l174_174534

theorem inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
    1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 := 
sorry

end inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8_l174_174534


namespace binom_10_3_eq_120_l174_174194

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l174_174194


namespace inequality_geq_l174_174921

variable {a b c : ℝ}

theorem inequality_geq (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 1 / a + 1 / b + 1 / c) : 
  a + b + c ≥ 3 / (a * b * c) := 
sorry

end inequality_geq_l174_174921


namespace cube_volume_of_surface_area_l174_174014

-- Define the condition: the surface area S is 864 square units
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- The proof problem: Given that the surface area of a cube is 864 square units,
-- prove that the volume of the cube is 1728 cubic units
theorem cube_volume_of_surface_area (S : ℝ) (hS : S = 864) : 
  ∃ V : ℝ, V = 1728 ∧ ∃ s : ℝ, surface_area s = S ∧ V = s^3 :=
by 
  sorry

end cube_volume_of_surface_area_l174_174014


namespace bianca_total_bags_l174_174655

theorem bianca_total_bags (bags_recycled_points : ℕ) (bags_not_recycled : ℕ) (total_points : ℕ) (total_bags : ℕ) 
  (h1 : bags_recycled_points = 5) 
  (h2 : bags_not_recycled = 8) 
  (h3 : total_points = 45) 
  (recycled_bags := total_points / bags_recycled_points) :
  total_bags = recycled_bags + bags_not_recycled := 
by 
  sorry

end bianca_total_bags_l174_174655


namespace inequality_condition_sufficient_l174_174667

theorem inequality_condition_sufficient (A B C : ℝ) (x y z : ℝ) 
  (hA : 0 ≤ A) 
  (hB : 0 ≤ B) 
  (hC : 0 ≤ C) 
  (hABC : A^2 + B^2 + C^2 ≤ 2 * (A * B + A * C + B * C)) :
  A * (x - y) * (x - z) + B * (y - z) * (y - x) + C * (z - x) * (z - y) ≥ 0 :=
sorry

end inequality_condition_sufficient_l174_174667


namespace cube_volume_of_surface_area_l174_174020

-- Define the condition: the surface area S is 864 square units
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- The proof problem: Given that the surface area of a cube is 864 square units,
-- prove that the volume of the cube is 1728 cubic units
theorem cube_volume_of_surface_area (S : ℝ) (hS : S = 864) : 
  ∃ V : ℝ, V = 1728 ∧ ∃ s : ℝ, surface_area s = S ∧ V = s^3 :=
by 
  sorry

end cube_volume_of_surface_area_l174_174020


namespace meaningful_if_and_only_if_l174_174886

theorem meaningful_if_and_only_if (x : ℝ) : (∃ y : ℝ, y = (1 / (x - 1))) ↔ x ≠ 1 :=
by 
  sorry

end meaningful_if_and_only_if_l174_174886


namespace cube_faces_consecutive_sum_l174_174831

noncomputable def cube_face_sum (n : ℕ) : ℕ :=
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5)

theorem cube_faces_consecutive_sum (n : ℕ) (h1 : ∀ i, i ∈ [0, 5] -> (2 * n + 5 + n + 5 - 6) = 6) (h2 : n = 12) :
  cube_face_sum n = 87 :=
  sorry

end cube_faces_consecutive_sum_l174_174831


namespace find_y_l174_174452

variable (x y z : ℝ)

theorem find_y
    (h₀ : x + y + z = 150)
    (h₁ : x + 10 = y - 10)
    (h₂ : y - 10 = 3 * z) :
    y = 74.29 :=
by
    sorry

end find_y_l174_174452


namespace multiples_of_231_l174_174080

theorem multiples_of_231 (h : ∀ i j : ℕ, 1 ≤ i → i < j → j ≤ 99 → i % 2 = 1 → 231 ∣ 10^j - 10^i) :
  ∃ n, n = 416 :=
by sorry

end multiples_of_231_l174_174080


namespace combination_10_3_l174_174281

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end combination_10_3_l174_174281


namespace intersection_eq_l174_174909

def S : Set ℝ := { x | x > -2 }
def T : Set ℝ := { x | -4 ≤ x ∧ x ≤ 1 }

theorem intersection_eq : S ∩ T = { x | -2 < x ∧ x ≤ 1 } :=
by
  simp [S, T]
  sorry

end intersection_eq_l174_174909


namespace binomial_10_3_eq_120_l174_174348

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l174_174348


namespace binomial_coefficient_10_3_l174_174276

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l174_174276


namespace inequality_proof_l174_174796

theorem inequality_proof (x y z : ℝ) : 
  ( (x^3) / (x^3 + 2 * (y^2) * z) + 
    (y^3) / (y^3 + 2 * (z^2) * x) + 
    (z^3) / (z^3 + 2 * (x^2) * y) ) ≥ 1 := 
by 
  sorry

end inequality_proof_l174_174796


namespace seq_general_formula_l174_174091

open Nat

def seq (a : ℕ+ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ+, a (n + 1) = 2 * a n / (2 + a n)

theorem seq_general_formula (a : ℕ+ → ℝ) (h : seq a) :
  ∀ n : ℕ+, a n = 2 / (n + 1) :=
by
  sorry

end seq_general_formula_l174_174091


namespace find_sum_of_integers_l174_174740

theorem find_sum_of_integers (w x y z : ℤ)
  (h1 : w - x + y = 7)
  (h2 : x - y + z = 8)
  (h3 : y - z + w = 4)
  (h4 : z - w + x = 3) : w + x + y + z = 11 :=
by
  sorry

end find_sum_of_integers_l174_174740


namespace jail_time_calculation_l174_174760

def total_arrests (arrests_per_day : ℕ) (cities : ℕ) (days : ℕ) : ℕ := 
  arrests_per_day * cities * days

def jail_time_before_trial (arrests : ℕ) (days_before_trial : ℕ) : ℕ := 
  days_before_trial * arrests

def jail_time_after_trial (arrests : ℕ) (weeks_after_trial : ℕ) : ℕ := 
  weeks_after_trial * arrests

def combined_jail_time (weeks_before_trial : ℕ) (weeks_after_trial : ℕ) : ℕ := 
  weeks_before_trial + weeks_after_trial

noncomputable def total_jail_time_in_weeks : ℕ := 
  let arrests := total_arrests 10 21 30
  let weeks_before_trial := jail_time_before_trial arrests 4 / 7
  let weeks_after_trial := jail_time_after_trial arrests 1
  combined_jail_time weeks_before_trial weeks_after_trial

theorem jail_time_calculation : 
  total_jail_time_in_weeks = 9900 :=
sorry

end jail_time_calculation_l174_174760


namespace sin_cos_fraction_eq_two_l174_174537

theorem sin_cos_fraction_eq_two (α : ℝ) (h : Real.tan α = 3) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2 :=
sorry

end sin_cos_fraction_eq_two_l174_174537


namespace B_visible_from_A_l174_174688

noncomputable def visibility_condition (a : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x → x < 3 → 4 * x - 2 > 2 * x^2

theorem B_visible_from_A (a : ℝ) : visibility_condition a ↔ a < 10 :=
by
  -- sorry statement is used to skip the proof part.
  sorry

end B_visible_from_A_l174_174688


namespace divisible_by_eight_l174_174144

def expr (n : ℕ) : ℕ := 3^(4*n + 1) + 5^(2*n + 1)

theorem divisible_by_eight (n : ℕ) : expr n % 8 = 0 :=
  sorry

end divisible_by_eight_l174_174144


namespace value_of_a_plus_b_l174_174396

noncomputable def f (a b x : ℝ) := x / (a * x + b)

theorem value_of_a_plus_b (a b : ℝ) (h₁: a ≠ 0) (h₂: f a b (-4) = 4)
    (h₃: ∀ x, f a b (f a b x) = x) : a + b = 3 / 2 :=
sorry

end value_of_a_plus_b_l174_174396


namespace isabel_weekly_distance_l174_174899

def circuit_length : ℕ := 365
def morning_runs : ℕ := 7
def afternoon_runs : ℕ := 3
def days_per_week : ℕ := 7

def morning_distance := morning_runs * circuit_length
def afternoon_distance := afternoon_runs * circuit_length
def daily_distance := morning_distance + afternoon_distance
def weekly_distance := daily_distance * days_per_week

theorem isabel_weekly_distance : weekly_distance = 25550 := by
  sorry

end isabel_weekly_distance_l174_174899


namespace combined_cost_price_is_250_l174_174982

axiom store_selling_conditions :
  ∃ (CP_A CP_B CP_C : ℝ),
    (CP_A = (110 + 70) / 2) ∧
    (CP_B = (90 + 30) / 2) ∧
    (CP_C = (150 + 50) / 2) ∧
    (CP_A + CP_B + CP_C = 250)

theorem combined_cost_price_is_250 : ∃ (CP_A CP_B CP_C : ℝ), CP_A + CP_B + CP_C = 250 :=
by sorry

end combined_cost_price_is_250_l174_174982


namespace find_p_l174_174873

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 5
def parabola_eq (p x y : ℝ) : Prop := y^2 = 2 * p * x
def quadrilateral_is_rectangle (A B C D : ℝ × ℝ) : Prop := 
  A.1 = C.1 ∧ B.1 = D.1 ∧ A.2 = D.2 ∧ B.2 = C.2

theorem find_p (A B C D : ℝ × ℝ) (p : ℝ) (h1 : ∃ x y, circle_eq x y ∧ parabola_eq p x y) 
  (h2 : ∃ x y, circle_eq x y ∧ x = 0) 
  (h3 : quadrilateral_is_rectangle A B C D) 
  (h4 : 0 < p) : 
  p = 2 := 
sorry

end find_p_l174_174873


namespace dogs_remaining_end_month_l174_174172

theorem dogs_remaining_end_month :
  let initial_dogs := 200
  let dogs_arrive_w1 := 30
  let dogs_adopt_w1 := 40
  let dogs_arrive_w2 := 40
  let dogs_adopt_w2 := 50
  let dogs_arrive_w3 := 30
  let dogs_adopt_w3 := 30
  let dogs_adopt_w4 := 70
  let dogs_return_w4 := 20
  initial_dogs + (dogs_arrive_w1 - dogs_adopt_w1) + 
  (dogs_arrive_w2 - dogs_adopt_w2) +
  (dogs_arrive_w3 - dogs_adopt_w3) + 
  (-dogs_adopt_w4 - dogs_return_w4) = 90 := by
  sorry

end dogs_remaining_end_month_l174_174172


namespace shaded_area_isosceles_right_triangle_l174_174606

theorem shaded_area_isosceles_right_triangle (y : ℝ) :
  (∃ (x : ℝ), 2 * x^2 = y^2) ∧
  (∃ (A : ℝ), A = (1 / 2) * (y^2 / 2)) ∧
  (∃ (shaded_area : ℝ), shaded_area = (1 / 2) * (y^2 / 4)) →
  (shaded_area = y^2 / 8) :=
sorry

end shaded_area_isosceles_right_triangle_l174_174606


namespace find_p_power_l174_174542

theorem find_p_power (p : ℕ) (h1 : p % 2 = 0) (h2 : (p + 1) % 10 = 7) : 
  (p % 10)^3 % 10 = (p % 10)^1 % 10 :=
by
  sorry

end find_p_power_l174_174542


namespace exists_n_for_A_of_non_perfect_square_l174_174432

theorem exists_n_for_A_of_non_perfect_square (A : ℕ) (h : ∀ k : ℕ, k^2 ≠ A) :
  ∃ n : ℕ, A = ⌊ n + Real.sqrt n + 1/2 ⌋ :=
sorry

end exists_n_for_A_of_non_perfect_square_l174_174432


namespace problem_statement_l174_174562

noncomputable def bernoulli : ℝ → MeasureTheory.Measure ↥ℝ := sorry
noncomputable def binomial : ℕ → ℝ → MeasureTheory.Measure ↥ℝ := sorry

variables (X : MeasureTheory.Measure ↥ℝ) (Y : MeasureTheory.Measure ↥ℝ)
          (p_X : ℝ := 0.7) (n_Y : ℕ := 10) (p_Y : ℝ := 0.8)

-- X follows Bernoulli distribution with success probability of 0.7
hypothesis hX : X = bernoulli p_X

-- Y follows Binomial distribution with n = 10, p = 0.8
hypothesis hY : Y = binomial n_Y p_Y

-- Expected value and variance for X (Bernoulli)
def EX := p_X
def DX := p_X * (1 - p_X)

-- Expected value and variance for Y (Binomial)
def EY := n_Y * p_Y
def DY := n_Y * p_Y * (1 - p_Y)

theorem problem_statement : EX = 0.7 ∧ DX = 0.21 ∧ EY = 8 ∧ DY = 1.6 :=
by {
  sorry
}

end problem_statement_l174_174562


namespace blocks_to_get_home_l174_174434

-- Definitions based on conditions provided
def blocks_to_park := 4
def blocks_to_school := 7
def trips_per_day := 3
def total_daily_blocks := 66

-- The proof statement for the number of blocks Ray walks to get back home
theorem blocks_to_get_home 
  (h1: blocks_to_park = 4)
  (h2: blocks_to_school = 7)
  (h3: trips_per_day = 3)
  (h4: total_daily_blocks = 66) : 
  (total_daily_blocks / trips_per_day - (blocks_to_park + blocks_to_school) = 11) :=
by
  sorry

end blocks_to_get_home_l174_174434


namespace whisker_relationship_l174_174382

theorem whisker_relationship :
  let P_whiskers := 14
  let C_whiskers := 22
  (C_whiskers - P_whiskers = 8) ∧ (C_whiskers / P_whiskers = 11 / 7) :=
by
  let P_whiskers := 14
  let C_whiskers := 22
  have h1 : C_whiskers - P_whiskers = 8 := by sorry
  have h2 : C_whiskers / P_whiskers = 11 / 7 := by sorry
  exact And.intro h1 h2

end whisker_relationship_l174_174382


namespace binom_10_3_eq_120_l174_174184

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l174_174184


namespace binom_10_3_eq_120_l174_174311

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l174_174311


namespace time_to_clear_l174_174464

def length_train1 := 121 -- in meters
def length_train2 := 153 -- in meters
def speed_train1 := 80 * 1000 / 3600 -- converting km/h to meters/s
def speed_train2 := 65 * 1000 / 3600 -- converting km/h to meters/s

def total_distance := length_train1 + length_train2
def relative_speed := speed_train1 + speed_train2

theorem time_to_clear : 
  (total_distance / relative_speed : ℝ) = 6.80 :=
by
  sorry

end time_to_clear_l174_174464


namespace binomial_10_3_eq_120_l174_174339

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l174_174339


namespace prob_same_color_l174_174087

-- Define the given conditions
def total_pieces : ℕ := 15
def black_pieces : ℕ := 6
def white_pieces : ℕ := 9
def prob_two_black : ℚ := 1/7
def prob_two_white : ℚ := 12/35

-- Define the statement to be proved
theorem prob_same_color : prob_two_black + prob_two_white = 17 / 35 := by
  sorry

end prob_same_color_l174_174087


namespace binomial_10_3_eq_120_l174_174201

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l174_174201


namespace max_obtuse_in_convex_quadrilateral_l174_174550

-- Definition and problem statement
def is_obtuse (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

def convex_quadrilateral (a b c d : ℝ) : Prop :=
  a + b + c + d = 360 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0

theorem max_obtuse_in_convex_quadrilateral (a b c d : ℝ) :
  convex_quadrilateral a b c d →
  (is_obtuse a → (is_obtuse b → ¬ (is_obtuse c ∧ is_obtuse d))) →
  (is_obtuse b → (is_obtuse a → ¬ (is_obtuse c ∧ is_obtuse d))) →
  (is_obtuse c → (is_obtuse a → ¬ (is_obtuse b ∧ is_obtuse d))) →
  (is_obtuse d → (is_obtuse a → ¬ (is_obtuse b ∧ is_obtuse c))) :=
by
  intros h_convex h1 h2 h3 h4
  sorry

end max_obtuse_in_convex_quadrilateral_l174_174550


namespace number_of_friends_dividing_bill_l174_174841

theorem number_of_friends_dividing_bill :
  ∃ n : ℕ, 45 * n = 135 ∧ n = 3 :=
begin
  use 3,
  split,
  { -- 45 * 3 = 135
    norm_num,
  },
  { -- n = 3
    refl,
  }
end

end number_of_friends_dividing_bill_l174_174841


namespace larger_triangle_perimeter_l174_174845

-- Given conditions
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

def is_isosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.a = t.c ∨ t.b = t.c

def similar (t1 t2 : Triangle) (k : ℝ) : Prop :=
  t1.a / t2.a = k ∧ t1.b / t2.b = k ∧ t1.c / t2.c = k

def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

-- Define specific triangles based on the problem
def smaller_triangle : Triangle := {a := 12, b := 12, c := 15}
def larger_triangle_ratio : ℝ := 2
def larger_triangle : Triangle := {a := 12 * larger_triangle_ratio, b := 12 * larger_triangle_ratio, c := 15 * larger_triangle_ratio}

-- Main theorem statement
theorem larger_triangle_perimeter : perimeter larger_triangle = 78 :=
by 
  sorry

end larger_triangle_perimeter_l174_174845


namespace inequality_proof_l174_174519

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  1 ≤ ((x + y) * (x^3 + y^3)) / (x^2 + y^2)^2 ∧ 
  ((x + y) * (x^3 + y^3)) / (x^2 + y^2)^2 ≤ 9 / 8 :=
sorry

end inequality_proof_l174_174519


namespace combinatorial_identity_inequality_solution_l174_174155

-- Part 1
theorem combinatorial_identity (n : ℕ) (h1 : n = 10) :
    Nat.choose (3 * n) (38 - n) + Nat.choose (n + 21) (3 * n) = 496 :=
by
  sorry

-- Part 2
theorem inequality_solution (x : ℕ) (h1 : 2 ≤ x) (h2 : x ≤ 9) :
    Nat.factorial 9 / Nat.factorial (9 - x) > 
    6 * Nat.factorial 9 / Nat.factorial (11 - x) → 
    x ∈ {2, 3, 4, 5, 6, 7} :=
by
  sorry

end combinatorial_identity_inequality_solution_l174_174155


namespace inequality_proof_l174_174797

theorem inequality_proof (x y z : ℝ) : 
  ( (x^3) / (x^3 + 2 * (y^2) * z) + 
    (y^3) / (y^3 + 2 * (z^2) * x) + 
    (z^3) / (z^3 + 2 * (x^2) * y) ) ≥ 1 := 
by 
  sorry

end inequality_proof_l174_174797


namespace q_value_l174_174728

noncomputable def prove_q (a b m p q : Real) :=
  (a * b = 5) → 
  (b + 1/a) * (a + 1/b) = q →
  q = 36/5

theorem q_value (a b : ℝ) (h_roots : a * b = 5) : (b + 1/a) * (a + 1/b) = 36 / 5 :=
by 
  sorry

end q_value_l174_174728


namespace grape_juice_amount_l174_174832

-- Definitions for the conditions
def total_weight : ℝ := 150
def orange_percentage : ℝ := 0.35
def watermelon_percentage : ℝ := 0.35

-- Theorem statement to prove the amount of grape juice
theorem grape_juice_amount : 
  (total_weight * (1 - orange_percentage - watermelon_percentage)) = 45 :=
by
  sorry

end grape_juice_amount_l174_174832


namespace divide_bill_evenly_l174_174840

variable (totalBill amountPaid : ℕ)
variable (numberOfFriends : ℕ)

theorem divide_bill_evenly (h1 : totalBill = 135) (h2 : amountPaid = 45) (h3 : numberOfFriends * amountPaid = totalBill) :
  numberOfFriends = 3 := by
  sorry

end divide_bill_evenly_l174_174840


namespace smallest_n_condition_l174_174571

noncomputable def distance_origin_to_point (n : ℕ) : ℝ := Real.sqrt (n)

noncomputable def radius_Bn (n : ℕ) : ℝ := distance_origin_to_point n - 1

def condition_Bn_contains_point_with_coordinate_greater_than_2 (n : ℕ) : Prop :=
  radius_Bn n > 2

theorem smallest_n_condition : ∃ n : ℕ, n ≥ 10 ∧ condition_Bn_contains_point_with_coordinate_greater_than_2 n :=
  sorry

end smallest_n_condition_l174_174571


namespace binom_10_3_l174_174229

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end binom_10_3_l174_174229


namespace area_of_trapezium_l174_174669

-- Definitions based on conditions
def length_parallel_side1 : ℝ := 20 -- length of the first parallel side
def length_parallel_side2 : ℝ := 18 -- length of the second parallel side
def distance_between_sides : ℝ := 5 -- distance between the parallel sides

-- Statement to prove
theorem area_of_trapezium (a b h : ℝ) :
  a = length_parallel_side1 → b = length_parallel_side2 → h = distance_between_sides →
  (a + b) * h / 2 = 95 :=
by
  intros ha hb hh
  rw [ha, hb, hh]
  sorry

end area_of_trapezium_l174_174669


namespace max_value_of_g_l174_174420

noncomputable def g (x : ℝ) : ℝ :=
  Real.sqrt (x * (80 - x)) + Real.sqrt (x * (10 - x))

theorem max_value_of_g :
  ∃ y_0 N, (∀ x, 0 ≤ x ∧ x ≤ 10 → g x ≤ N) ∧ g y_0 = N ∧ y_0 = 33.75 ∧ N = 22.5 := 
by
  -- Proof goes here.
  sorry

end max_value_of_g_l174_174420


namespace rectangle_in_triangle_area_l174_174908

theorem rectangle_in_triangle_area (b h : ℕ) (hb : b = 12) (hh : h = 8)
  (x : ℕ) (hx : x = h / 2) : (b * x / 2) = 48 := 
by
  sorry

end rectangle_in_triangle_area_l174_174908


namespace CoreyCandies_l174_174603

theorem CoreyCandies (T C : ℕ) (h1 : T + C = 66) (h2 : T = C + 8) : C = 29 :=
by
  sorry

end CoreyCandies_l174_174603


namespace combination_10_3_eq_120_l174_174362

open Nat

theorem combination_10_3_eq_120 : (10.choose 3) = 120 := 
by
  sorry

end combination_10_3_eq_120_l174_174362


namespace inequality_proof_l174_174529

theorem inequality_proof (x y : ℝ) (hx: 0 < x) (hy: 0 < y) : 
  1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧ 
  (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 := 
by 
  sorry

end inequality_proof_l174_174529


namespace value_of_X_when_S_reaches_15000_l174_174561

def X : Nat → Nat
| 0       => 5
| (n + 1) => X n + 3

def S : Nat → Nat
| 0       => 0
| (n + 1) => S n + X (n + 1)

theorem value_of_X_when_S_reaches_15000 :
  ∃ n, S n ≥ 15000 ∧ X n = 299 := by
  sorry

end value_of_X_when_S_reaches_15000_l174_174561


namespace binom_10_3_l174_174328

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end binom_10_3_l174_174328
