import Mathlib

namespace NUMINAMATH_CALUDE_candidate_X_loses_by_6_percent_l268_26854

/-- Represents the political parties --/
inductive Party
  | Republican
  | Democrat
  | Independent

/-- Represents the candidates --/
inductive Candidate
  | X
  | Y

/-- The ratio of registered voters for each party --/
def partyRatio : Party → Nat
  | Party.Republican => 3
  | Party.Democrat => 2
  | Party.Independent => 5

/-- The percentage of voters from each party expected to vote for candidate X --/
def votePercentageForX : Party → Rat
  | Party.Republican => 70/100
  | Party.Democrat => 30/100
  | Party.Independent => 40/100

/-- The percentage of registered voters who will not vote --/
def nonVoterPercentage : Rat := 10/100

/-- Theorem stating that candidate X is expected to lose by approximately 6% --/
theorem candidate_X_loses_by_6_percent :
  ∃ (total_voters : Nat),
    total_voters > 0 →
    let votes_for_X := (partyRatio Party.Republican * (votePercentageForX Party.Republican : Rat) +
                        partyRatio Party.Democrat * (votePercentageForX Party.Democrat : Rat) +
                        partyRatio Party.Independent * (votePercentageForX Party.Independent : Rat)) *
                       (1 - nonVoterPercentage) * total_voters
    let votes_for_Y := (partyRatio Party.Republican * (1 - votePercentageForX Party.Republican : Rat) +
                        partyRatio Party.Democrat * (1 - votePercentageForX Party.Democrat : Rat) +
                        partyRatio Party.Independent * (1 - votePercentageForX Party.Independent : Rat)) *
                       (1 - nonVoterPercentage) * total_voters
    let total_votes := votes_for_X + votes_for_Y
    let percentage_difference := (votes_for_Y - votes_for_X) / total_votes * 100
    abs (percentage_difference - 6) < 1 := by
  sorry

end NUMINAMATH_CALUDE_candidate_X_loses_by_6_percent_l268_26854


namespace NUMINAMATH_CALUDE_pyramid_solution_l268_26840

/-- Represents the structure of the number pyramid --/
structure NumberPyramid where
  row2_1 : ℕ
  row2_2 : ℕ → ℕ
  row2_3 : ℕ → ℕ
  row3_1 : ℕ → ℕ
  row3_2 : ℕ → ℕ
  row4   : ℕ → ℕ

/-- The specific number pyramid instance from the problem --/
def problemPyramid : NumberPyramid := {
  row2_1 := 11
  row2_2 := λ x => 6 + x
  row2_3 := λ x => x + 7
  row3_1 := λ x => 11 + (6 + x)
  row3_2 := λ x => (6 + x) + (x + 7)
  row4   := λ x => (11 + (6 + x)) + ((6 + x) + (x + 7))
}

/-- The theorem stating that x = 10 in this specific number pyramid --/
theorem pyramid_solution :
  ∃ x : ℕ, problemPyramid.row4 x = 60 ∧ x = 10 := by
  sorry


end NUMINAMATH_CALUDE_pyramid_solution_l268_26840


namespace NUMINAMATH_CALUDE_bezout_identity_solutions_l268_26887

theorem bezout_identity_solutions (a b d u v : ℤ) 
  (h_gcd : d = Int.gcd a b) 
  (h_bezout : a * u + b * v = d) : 
  (∀ x y : ℤ, a * x + b * y = d ↔ ∃ k : ℤ, x = u + k * b ∧ y = v - k * a) ∧
  {p : ℤ × ℤ | a * p.1 + b * p.2 = d} = {p : ℤ × ℤ | ∃ k : ℤ, p = (u + k * b, v - k * a)} :=
by sorry

end NUMINAMATH_CALUDE_bezout_identity_solutions_l268_26887


namespace NUMINAMATH_CALUDE_max_product_sum_200_l268_26880

theorem max_product_sum_200 : 
  ∀ x y : ℤ, x + y = 200 → x * y ≤ 10000 := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_200_l268_26880


namespace NUMINAMATH_CALUDE_exists_valid_pet_counts_unique_valid_pet_counts_total_pets_is_nineteen_l268_26806

/-- Represents the number of pets Frankie has of each type -/
structure PetCounts where
  dogs : Nat
  cats : Nat
  snakes : Nat
  parrots : Nat

/-- Defines the conditions given in the problem -/
def validPetCounts (p : PetCounts) : Prop :=
  p.dogs = 2 ∧
  p.snakes > p.cats ∧
  p.parrots = p.cats - 1 ∧
  p.dogs + p.cats = 6 ∧
  p.dogs + p.cats + p.snakes + p.parrots = 19

/-- Theorem stating that there exists a valid pet count configuration -/
theorem exists_valid_pet_counts : ∃ p : PetCounts, validPetCounts p :=
  sorry

/-- Theorem proving the uniqueness of the valid pet count configuration -/
theorem unique_valid_pet_counts (p q : PetCounts) 
  (hp : validPetCounts p) (hq : validPetCounts q) : p = q :=
  sorry

/-- Main theorem proving that the total number of pets is 19 -/
theorem total_pets_is_nineteen (p : PetCounts) (h : validPetCounts p) :
  p.dogs + p.cats + p.snakes + p.parrots = 19 :=
  sorry

end NUMINAMATH_CALUDE_exists_valid_pet_counts_unique_valid_pet_counts_total_pets_is_nineteen_l268_26806


namespace NUMINAMATH_CALUDE_first_group_has_four_weavers_l268_26857

/-- The number of mat-weavers in the first group -/
def first_group_weavers : ℕ := 4

/-- The number of mats woven by the first group -/
def first_group_mats : ℕ := 4

/-- The number of days taken by the first group -/
def first_group_days : ℕ := 4

/-- The number of mat-weavers in the second group -/
def second_group_weavers : ℕ := 8

/-- The number of mats woven by the second group -/
def second_group_mats : ℕ := 16

/-- The number of days taken by the second group -/
def second_group_days : ℕ := 8

/-- The rate of weaving is the same for both groups -/
axiom same_rate : (first_group_mats : ℚ) / first_group_weavers / first_group_days = 
                  (second_group_mats : ℚ) / second_group_weavers / second_group_days

theorem first_group_has_four_weavers : first_group_weavers = 4 := by
  sorry

end NUMINAMATH_CALUDE_first_group_has_four_weavers_l268_26857


namespace NUMINAMATH_CALUDE_vertex_C_coordinates_l268_26848

-- Define the coordinate type
def Coordinate := ℝ × ℝ

-- Define the line equation type
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the triangle
structure Triangle where
  A : Coordinate
  B : Coordinate
  C : Coordinate

-- Define the problem conditions
def problem_conditions (t : Triangle) : Prop :=
  t.A = (5, 1) ∧
  ∃ (eq_CM : LineEquation), eq_CM = ⟨2, -1, -5⟩ ∧
  ∃ (eq_BH : LineEquation), eq_BH = ⟨1, -2, -5⟩

-- State the theorem
theorem vertex_C_coordinates (t : Triangle) :
  problem_conditions t → t.C = (4, 3) := by
  sorry

end NUMINAMATH_CALUDE_vertex_C_coordinates_l268_26848


namespace NUMINAMATH_CALUDE_complex_equation_solution_l268_26883

theorem complex_equation_solution (z : ℂ) (h : (1 + Complex.I) / z = 1 - Complex.I) : z = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l268_26883


namespace NUMINAMATH_CALUDE_total_profit_is_6300_l268_26881

/-- Represents the profit sharing scenario between Tom and Jose -/
structure ProfitSharing where
  tom_investment : ℕ
  tom_months : ℕ
  jose_investment : ℕ
  jose_months : ℕ
  jose_profit : ℕ

/-- Calculates the total profit based on the given profit sharing scenario -/
def calculate_total_profit (ps : ProfitSharing) : ℕ :=
  let tom_investment_months := ps.tom_investment * ps.tom_months
  let jose_investment_months := ps.jose_investment * ps.jose_months
  let ratio_denominator := tom_investment_months + jose_investment_months
  let tom_profit := (tom_investment_months * ps.jose_profit) / jose_investment_months
  tom_profit + ps.jose_profit

/-- Theorem stating that the total profit for the given scenario is 6300 -/
theorem total_profit_is_6300 (ps : ProfitSharing) 
  (h1 : ps.tom_investment = 3000) 
  (h2 : ps.tom_months = 12) 
  (h3 : ps.jose_investment = 4500) 
  (h4 : ps.jose_months = 10) 
  (h5 : ps.jose_profit = 3500) : 
  calculate_total_profit ps = 6300 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_6300_l268_26881


namespace NUMINAMATH_CALUDE_storm_deposit_calculation_l268_26831

-- Define the reservoir capacity and initial content
def reservoir_capacity : ℝ := 400000000000
def initial_content : ℝ := 220000000000

-- Define the initial and final fill percentages
def initial_fill_percentage : ℝ := 0.5500000000000001
def final_fill_percentage : ℝ := 0.85

-- Define the amount of water deposited by the storm
def storm_deposit : ℝ := 120000000000

-- Theorem statement
theorem storm_deposit_calculation :
  initial_content = initial_fill_percentage * reservoir_capacity ∧
  storm_deposit = final_fill_percentage * reservoir_capacity - initial_content :=
by sorry

end NUMINAMATH_CALUDE_storm_deposit_calculation_l268_26831


namespace NUMINAMATH_CALUDE_algae_cells_after_ten_days_l268_26862

def algae_growth (initial_cells : ℕ) (split_factor : ℕ) (days : ℕ) : ℕ :=
  initial_cells * split_factor ^ days

theorem algae_cells_after_ten_days :
  algae_growth 1 3 10 = 59049 := by
  sorry

end NUMINAMATH_CALUDE_algae_cells_after_ten_days_l268_26862


namespace NUMINAMATH_CALUDE_counterexample_exists_l268_26852

theorem counterexample_exists : ∃ (a b : ℝ), a < b ∧ -3 * a ≥ -3 * b := by sorry

end NUMINAMATH_CALUDE_counterexample_exists_l268_26852


namespace NUMINAMATH_CALUDE_a_grade_implies_conditions_l268_26858

-- Define the conditions for receiving an A
def receivesA (score : ℝ) (submittedAll : Bool) : Prop :=
  score ≥ 90 ∧ submittedAll

-- Define the theorem
theorem a_grade_implies_conditions 
  (score : ℝ) (submittedAll : Bool) :
  receivesA score submittedAll → 
  (score ≥ 90 ∧ submittedAll) :=
by
  sorry

-- The proof is omitted as per instructions

end NUMINAMATH_CALUDE_a_grade_implies_conditions_l268_26858


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l268_26893

theorem inscribed_circle_radius (a b c : ℝ) (h1 : a = 8) (h2 : b = 15) (h3 : c = 17) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area / s = 3 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l268_26893


namespace NUMINAMATH_CALUDE_cookie_bags_l268_26814

theorem cookie_bags (total_cookies : ℕ) (cookies_per_bag : ℕ) (h1 : total_cookies = 33) (h2 : cookies_per_bag = 11) :
  total_cookies / cookies_per_bag = 3 := by
  sorry

end NUMINAMATH_CALUDE_cookie_bags_l268_26814


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l268_26865

/-- The value of d for which the line y = 3x + d is tangent to the parabola y^2 = 12x -/
theorem tangent_line_to_parabola : ∃ d : ℝ, 
  (∀ x y : ℝ, y = 3*x + d → y^2 = 12*x → 
    ∀ ε > 0, ∃ δ > 0, ∀ x' y' : ℝ, 
      (x' - x)^2 + (y' - y)^2 < δ^2 → 
      (y' - (3*x' + d))^2 > ε^2 * ((y')^2 - 12*x')) ∧
  d = 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l268_26865


namespace NUMINAMATH_CALUDE_certain_fraction_proof_l268_26899

theorem certain_fraction_proof (x y : ℚ) :
  (x / y) / (3 / 7) = 0.46666666666666673 / (1 / 2) →
  x / y = 0.4 := by
sorry

end NUMINAMATH_CALUDE_certain_fraction_proof_l268_26899


namespace NUMINAMATH_CALUDE_sam_filled_four_bags_saturday_l268_26867

/-- The number of bags Sam filled on Saturday -/
def saturday_bags : ℕ := sorry

/-- The number of bags Sam filled on Sunday -/
def sunday_bags : ℕ := 3

/-- The number of cans in each bag -/
def cans_per_bag : ℕ := 6

/-- The total number of cans collected -/
def total_cans : ℕ := 42

/-- Theorem stating that Sam filled 4 bags on Saturday -/
theorem sam_filled_four_bags_saturday : saturday_bags = 4 := by
  sorry

end NUMINAMATH_CALUDE_sam_filled_four_bags_saturday_l268_26867


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_l268_26890

/-- Two vectors in R² are perpendicular if and only if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- The theorem states that if vectors (m, 4) and (m+4, 1) are perpendicular, then m = -2 -/
theorem perpendicular_vectors_m (m : ℝ) :
  perpendicular (m, 4) (m + 4, 1) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_l268_26890


namespace NUMINAMATH_CALUDE_line_through_point_l268_26839

/-- 
Given a line with equation 3ax + (2a+1)y = 3a+3 that passes through the point (3, -9),
prove that a = -1.
-/
theorem line_through_point (a : ℝ) : 
  (3 * a * 3 + (2 * a + 1) * (-9) = 3 * a + 3) → a = -1 := by
sorry

end NUMINAMATH_CALUDE_line_through_point_l268_26839


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l268_26889

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 2 = 1

-- Define the asymptote equations
def asymptote1 (x y : ℝ) : Prop := y = (Real.sqrt 2 / 2) * x
def asymptote2 (x y : ℝ) : Prop := y = -(Real.sqrt 2 / 2) * x

-- Theorem statement
theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), hyperbola x y →
  (asymptote1 x y ∨ asymptote2 x y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l268_26889


namespace NUMINAMATH_CALUDE_football_draws_l268_26815

/-- Represents the possible outcomes of a football match -/
inductive MatchResult
  | Win
  | Draw
  | Loss

/-- Calculate points for a given match result -/
def pointsForResult (result : MatchResult) : Nat :=
  match result with
  | MatchResult.Win => 3
  | MatchResult.Draw => 1
  | MatchResult.Loss => 0

/-- Represents the results of a series of matches -/
structure MatchResults :=
  (wins : Nat)
  (draws : Nat)
  (losses : Nat)

/-- Calculate total points for a series of matches -/
def totalPoints (results : MatchResults) : Nat :=
  3 * results.wins + results.draws

/-- The main theorem to prove -/
theorem football_draws (results : MatchResults) :
  results.wins + results.draws + results.losses = 5 →
  totalPoints results = 7 →
  results.draws = 1 ∨ results.draws = 4 := by
  sorry


end NUMINAMATH_CALUDE_football_draws_l268_26815


namespace NUMINAMATH_CALUDE_divisibility_count_l268_26841

theorem divisibility_count : ∃ (S : Finset Nat), 
  (∀ n ∈ S, n ≤ 30 ∧ n > 0 ∧ (n! % (n * (n + 2) / 3) = 0)) ∧ 
  Finset.card S = 30 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_count_l268_26841


namespace NUMINAMATH_CALUDE_candidate_selection_probability_l268_26896

/-- Represents the probability distribution of Excel skills among job candidates -/
structure ExcelSkills where
  beginner : ℝ
  intermediate : ℝ
  advanced : ℝ
  none : ℝ
  sum_to_one : beginner + intermediate + advanced + none = 1

/-- Represents the probability distribution of shift preferences among job candidates -/
structure ShiftPreference where
  day : ℝ
  night : ℝ
  sum_to_one : day + night = 1

/-- Represents the probability distribution of weekend work preferences among job candidates -/
structure WeekendPreference where
  willing : ℝ
  not_willing : ℝ
  sum_to_one : willing + not_willing = 1

/-- Theorem stating the probability of selecting a candidate with specific characteristics -/
theorem candidate_selection_probability 
  (excel : ExcelSkills)
  (shift : ShiftPreference)
  (weekend : WeekendPreference)
  (h1 : excel.beginner = 0.35)
  (h2 : excel.intermediate = 0.25)
  (h3 : excel.advanced = 0.2)
  (h4 : excel.none = 0.2)
  (h5 : shift.day = 0.7)
  (h6 : shift.night = 0.3)
  (h7 : weekend.willing = 0.4)
  (h8 : weekend.not_willing = 0.6) :
  (excel.intermediate + excel.advanced) * shift.night * weekend.not_willing = 0.081 := by
  sorry

end NUMINAMATH_CALUDE_candidate_selection_probability_l268_26896


namespace NUMINAMATH_CALUDE_inequalities_solution_range_l268_26800

theorem inequalities_solution_range (m : ℝ) : 
  (∃ (x₁ x₂ x₃ x₄ : ℤ), 
    (∀ x : ℤ, (3 * ↑x - m < 0 ∧ 7 - 2 * ↑x < 5) ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)) ∧
    x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄) →
  (15 < m ∧ m ≤ 18) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_solution_range_l268_26800


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l268_26843

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  (1 / x + 3 / y) ≥ 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l268_26843


namespace NUMINAMATH_CALUDE_solutions_of_x_squared_equals_x_l268_26823

theorem solutions_of_x_squared_equals_x : 
  {x : ℝ | x^2 = x} = {0, 1} := by sorry

end NUMINAMATH_CALUDE_solutions_of_x_squared_equals_x_l268_26823


namespace NUMINAMATH_CALUDE_count_distinct_z_values_l268_26871

def is_three_digit (n : ℤ) : Prop := 100 ≤ n ∧ n ≤ 999

def reverse_digits (n : ℤ) : ℤ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  100 * ones + 10 * tens + hundreds

def z_value (x : ℤ) : ℤ := |x - reverse_digits x|

def satisfies_conditions (x : ℤ) : Prop :=
  is_three_digit x ∧ 
  is_three_digit (reverse_digits x) ∧
  (z_value x) % 33 = 0

theorem count_distinct_z_values :
  ∃ (S : Finset ℤ), 
    (∀ x, satisfies_conditions x → z_value x ∈ S) ∧ 
    (∀ z ∈ S, ∃ x, satisfies_conditions x ∧ z_value x = z) ∧
    Finset.card S = 10 :=
sorry

end NUMINAMATH_CALUDE_count_distinct_z_values_l268_26871


namespace NUMINAMATH_CALUDE_plane_centroid_sum_l268_26833

-- Define the plane and points
def Plane := {plane : ℝ → ℝ → ℝ → Prop | ∃ (a b c : ℝ), ∀ x y z, plane x y z ↔ (x / a + y / b + z / c = 1)}

def distance_from_origin (plane : Plane) : ℝ := sorry

def intersect_x_axis (plane : Plane) : ℝ := sorry
def intersect_y_axis (plane : Plane) : ℝ := sorry
def intersect_z_axis (plane : Plane) : ℝ := sorry

def centroid (a b c : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

-- Theorem statement
theorem plane_centroid_sum (plane : Plane) :
  let a := (intersect_x_axis plane, 0, 0)
  let b := (0, intersect_y_axis plane, 0)
  let c := (0, 0, intersect_z_axis plane)
  let (p, q, r) := centroid a b c
  distance_from_origin plane = Real.sqrt 2 →
  a ≠ (0, 0, 0) ∧ b ≠ (0, 0, 0) ∧ c ≠ (0, 0, 0) →
  1 / p^2 + 1 / q^2 + 1 / r^2 = 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_plane_centroid_sum_l268_26833


namespace NUMINAMATH_CALUDE_inverse_of_A_l268_26801

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, -1; 2, 5]

theorem inverse_of_A :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![5/22, 1/22; -1/11, 2/11]
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_l268_26801


namespace NUMINAMATH_CALUDE_trigonometric_identities_l268_26849

theorem trigonometric_identities :
  (Real.cos (15 * π / 180))^2 - (Real.sin (15 * π / 180))^2 = Real.sqrt 3 / 2 ∧
  Real.sin (π / 8) * Real.cos (π / 8) = Real.sqrt 2 / 4 ∧
  Real.tan (15 * π / 180) = 2 - Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l268_26849


namespace NUMINAMATH_CALUDE_problem_solution_l268_26820

theorem problem_solution (x y : ℚ) : 
  x = 152 → 
  x^3*y - 3*x^2*y + 3*x*y = 912000 → 
  y = 3947/15200 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l268_26820


namespace NUMINAMATH_CALUDE_midpoint_sum_equals_vertex_sum_l268_26891

theorem midpoint_sum_equals_vertex_sum (d e f : ℝ) :
  let vertex_sum := d + e + f
  let midpoint_sum := (d + e) / 2 + (d + f) / 2 + (e + f) / 2
  vertex_sum = midpoint_sum := by sorry

end NUMINAMATH_CALUDE_midpoint_sum_equals_vertex_sum_l268_26891


namespace NUMINAMATH_CALUDE_infinite_rational_square_sum_169_l268_26853

theorem infinite_rational_square_sum_169 : 
  ∀ n : ℕ, ∃ x y : ℚ, x^2 + y^2 = 169 ∧ 
  (∀ m : ℕ, m < n → ∃ x' y' : ℚ, x'^2 + y'^2 = 169 ∧ (x' ≠ x ∨ y' ≠ y)) :=
by sorry

end NUMINAMATH_CALUDE_infinite_rational_square_sum_169_l268_26853


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l268_26817

/-- Given a quadratic equation (k-1)x^2 + 6x + 9 = 0 with two equal real roots, prove that k = 2 -/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x^2 + 6 * x + 9 = 0 ∧ 
   ∀ y : ℝ, (k - 1) * y^2 + 6 * y + 9 = 0 → y = x) → 
  k = 2 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l268_26817


namespace NUMINAMATH_CALUDE_hassan_apple_trees_count_l268_26897

/-- The number of apple trees Hassan has -/
def hassan_apple_trees : ℕ := 1

/-- The number of orange trees Ahmed has -/
def ahmed_orange_trees : ℕ := 8

/-- The number of orange trees Hassan has -/
def hassan_orange_trees : ℕ := 2

/-- The number of apple trees Ahmed has -/
def ahmed_apple_trees : ℕ := 4 * hassan_apple_trees

/-- The total number of trees in Ahmed's orchard -/
def ahmed_total_trees : ℕ := ahmed_orange_trees + ahmed_apple_trees

/-- The total number of trees in Hassan's orchard -/
def hassan_total_trees : ℕ := hassan_orange_trees + hassan_apple_trees

theorem hassan_apple_trees_count :
  hassan_apple_trees = 1 ∧
  ahmed_orange_trees = 8 ∧
  hassan_orange_trees = 2 ∧
  ahmed_apple_trees = 4 * hassan_apple_trees ∧
  ahmed_total_trees = hassan_total_trees + 9 := by
  sorry

end NUMINAMATH_CALUDE_hassan_apple_trees_count_l268_26897


namespace NUMINAMATH_CALUDE_impossible_perpendicular_l268_26809

-- Define the types for planes and lines
variable {Point : Type*}
variable {Line : Type*}
variable {Plane : Type*}

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (intersect : Line → Line → Point → Prop)

-- Define the theorem
theorem impossible_perpendicular 
  (α : Plane) (a b : Line) (P : Point)
  (h1 : perpendicular a α)
  (h2 : intersect a b P) :
  ¬ (perpendicular b α) := by
  sorry

end NUMINAMATH_CALUDE_impossible_perpendicular_l268_26809


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_98_l268_26812

theorem largest_four_digit_divisible_by_98 :
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 98 = 0 → n ≤ 9998 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_98_l268_26812


namespace NUMINAMATH_CALUDE_call_center_team_ratio_l268_26886

theorem call_center_team_ratio (a b : ℚ) : 
  (∀ (c : ℚ), a * (3/5 * c) / (b * c) = 3/11) →
  a / b = 5/11 := by
sorry

end NUMINAMATH_CALUDE_call_center_team_ratio_l268_26886


namespace NUMINAMATH_CALUDE_friends_total_amount_l268_26866

/-- The total amount of money received by three friends from selling video games -/
def total_amount (zachary_games : ℕ) (price_per_game : ℕ) (jason_percent : ℕ) (ryan_extra : ℕ) : ℕ :=
  let zachary_amount := zachary_games * price_per_game
  let jason_amount := zachary_amount + (jason_percent * zachary_amount) / 100
  let ryan_amount := jason_amount + ryan_extra
  zachary_amount + jason_amount + ryan_amount

/-- Theorem stating that the total amount received by the three friends is $770 -/
theorem friends_total_amount :
  total_amount 40 5 30 50 = 770 := by
  sorry

end NUMINAMATH_CALUDE_friends_total_amount_l268_26866


namespace NUMINAMATH_CALUDE_fifth_term_value_l268_26850

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  sum_formula : ∀ n, S n = n / 2 * (a 1 + a n)

/-- Theorem: If S₆ = 12 and a₂ = 5 in an arithmetic sequence, then a₅ = -1 -/
theorem fifth_term_value (seq : ArithmeticSequence) 
  (sum_6 : seq.S 6 = 12) 
  (second_term : seq.a 2 = 5) : 
  seq.a 5 = -1 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_value_l268_26850


namespace NUMINAMATH_CALUDE_roberts_cash_amount_l268_26803

theorem roberts_cash_amount (raw_materials_cost machinery_cost : ℝ) 
  (h1 : raw_materials_cost = 100)
  (h2 : machinery_cost = 125)
  (total_amount : ℝ) :
  raw_materials_cost + machinery_cost + 0.1 * total_amount = total_amount →
  total_amount = 250 := by
sorry

end NUMINAMATH_CALUDE_roberts_cash_amount_l268_26803


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l268_26835

theorem consecutive_integers_sum_of_squares (n : ℕ) : 
  (n - 1) ^ 2 + n ^ 2 + (n + 1) ^ 2 = 770 → n + 1 = 17 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l268_26835


namespace NUMINAMATH_CALUDE_sum_and_ratio_problem_l268_26838

theorem sum_and_ratio_problem (x y : ℚ) 
  (sum_eq : x + y = 520)
  (ratio_eq : x / y = 3 / 4) :
  y - x = 520 / 7 := by
sorry

end NUMINAMATH_CALUDE_sum_and_ratio_problem_l268_26838


namespace NUMINAMATH_CALUDE_parabola_directrix_l268_26845

/-- The directrix of a parabola with equation y = (1/4)x^2 -/
def directrix_of_parabola (x y : ℝ) : Prop :=
  y = (1/4) * x^2 → y = -1

theorem parabola_directrix : 
  ∀ x y : ℝ, directrix_of_parabola x y :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l268_26845


namespace NUMINAMATH_CALUDE_new_person_weight_l268_26855

/-- Given a group of 12 people where one person weighing 62 kg is replaced by a new person,
    causing the average weight to increase by 4.8 kg, prove that the new person weighs 119.6 kg. -/
theorem new_person_weight (initial_count : Nat) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 12 →
  weight_increase = 4.8 →
  replaced_weight = 62 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 119.6 :=
by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l268_26855


namespace NUMINAMATH_CALUDE_gcd_special_numbers_l268_26860

theorem gcd_special_numbers : 
  let m : ℕ := 555555555
  let n : ℕ := 1111111111
  Nat.gcd m n = 1 := by sorry

end NUMINAMATH_CALUDE_gcd_special_numbers_l268_26860


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l268_26864

theorem arithmetic_sequence_sum (a : ℕ → ℕ) : 
  (∀ n, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence condition
  a 0 = 3 →                            -- first term is 3
  a 1 = 9 →                            -- second term is 9
  a 6 = 33 →                           -- last (seventh) term is 33
  a 4 + a 5 = 60 :=                    -- sum of fifth and sixth terms is 60
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l268_26864


namespace NUMINAMATH_CALUDE_smaller_number_problem_l268_26828

theorem smaller_number_problem (x y : ℝ) : 
  y = 3 * x ∧ x + y = 124 → x = 31 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l268_26828


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l268_26804

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + a 5 = 8)
  (h_fourth : a 4 = 7) : 
  a 5 = 10 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l268_26804


namespace NUMINAMATH_CALUDE_cylinder_sphere_cone_volume_ratio_l268_26811

/-- Given a cylinder with volume 128π cm³, the ratio of the volume of a sphere 
(with radius equal to the base radius of the cylinder) to the volume of a cone 
(with the same radius and height as the cylinder) is 2. -/
theorem cylinder_sphere_cone_volume_ratio : 
  ∀ (r h : ℝ), 
  r > 0 → h > 0 →
  π * r^2 * h = 128 * π →
  (4/3 * π * r^3) / (1/3 * π * r^2 * h) = 2 := by
sorry

end NUMINAMATH_CALUDE_cylinder_sphere_cone_volume_ratio_l268_26811


namespace NUMINAMATH_CALUDE_isosceles_triangle_rectangle_equal_area_l268_26802

/-- Given an isosceles triangle with base b and height h, and a rectangle with base b and height 2b,
    if their areas are equal, then the height of the triangle is 4 times the base. -/
theorem isosceles_triangle_rectangle_equal_area (b h : ℝ) (b_pos : 0 < b) :
  (1 / 2 : ℝ) * b * h = b * (2 * b) → h = 4 * b := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_rectangle_equal_area_l268_26802


namespace NUMINAMATH_CALUDE_correct_matching_probability_l268_26885

-- Define the number of students and pictures
def num_students : ℕ := 4

-- Define the function to calculate the factorial
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define the total number of possible arrangements
def total_arrangements : ℕ := factorial num_students

-- Define the number of correct arrangements
def correct_arrangements : ℕ := 1

-- State the theorem
theorem correct_matching_probability :
  (correct_arrangements : ℚ) / total_arrangements = 1 / 24 := by
  sorry

end NUMINAMATH_CALUDE_correct_matching_probability_l268_26885


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l268_26827

theorem quadratic_equation_roots (x : ℝ) : 
  (∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ r₁^2 + 4*r₁ = 0 ∧ r₂^2 + 4*r₂ = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_equation_roots_l268_26827


namespace NUMINAMATH_CALUDE_recipe_multiplier_is_six_l268_26888

/-- Represents the ratio of butter to flour in a recipe -/
structure RecipeRatio where
  butter : ℚ
  flour : ℚ

/-- The original recipe ratio -/
def originalRatio : RecipeRatio := { butter := 2, flour := 5 }

/-- The amount of butter used in the new recipe -/
def newButterAmount : ℚ := 12

/-- Calculates how many times the original recipe is being made -/
def recipeMultiplier (original : RecipeRatio) (newButter : ℚ) : ℚ :=
  newButter / original.butter

theorem recipe_multiplier_is_six :
  recipeMultiplier originalRatio newButterAmount = 6 := by
  sorry

#eval recipeMultiplier originalRatio newButterAmount

end NUMINAMATH_CALUDE_recipe_multiplier_is_six_l268_26888


namespace NUMINAMATH_CALUDE_arithmetic_seq_common_diff_l268_26863

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ       -- Common difference
  S : ℕ → ℝ  -- Sum function
  seq_def : ∀ n, a (n + 1) = a n + d
  sum_def : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2

/-- If 2S₃ = 3S₂ + 6 for an arithmetic sequence, then the common difference is 2 -/
theorem arithmetic_seq_common_diff (seq : ArithmeticSequence) 
    (h : 2 * seq.S 3 = 3 * seq.S 2 + 6) : seq.d = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_common_diff_l268_26863


namespace NUMINAMATH_CALUDE_initial_patio_rows_l268_26874

/-- Represents a rectangular patio -/
structure Patio where
  rows : ℕ
  cols : ℕ

/-- Checks if a patio is valid according to the given conditions -/
def isValidPatio (p : Patio) : Prop :=
  p.rows * p.cols = 60 ∧
  2 * p.cols = (3 * p.rows) / 2 ∧
  (p.rows + 5) * (p.cols - 3) = 60

theorem initial_patio_rows : 
  ∃ (p : Patio), isValidPatio p ∧ p.rows = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_patio_rows_l268_26874


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l268_26869

theorem simplify_and_rationalize (x : ℝ) :
  1 / (1 + 1 / (Real.sqrt 5 + 2)) = (Real.sqrt 5 + 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l268_26869


namespace NUMINAMATH_CALUDE_probability_different_tens_proof_l268_26842

/-- The number of integers in the range 10 to 79, inclusive. -/
def total_numbers : ℕ := 70

/-- The number of different tens digits available in the range 10 to 79. -/
def available_tens_digits : ℕ := 7

/-- The number of integers for each tens digit. -/
def numbers_per_tens : ℕ := 10

/-- The number of integers to be chosen. -/
def chosen_count : ℕ := 7

/-- The probability of selecting 7 different integers from the range 10 to 79 (inclusive)
    such that each has a different tens digit. -/
def probability_different_tens : ℚ := 10000000 / 93947434

theorem probability_different_tens_proof :
  (numbers_per_tens ^ chosen_count : ℚ) / (total_numbers.choose chosen_count) = probability_different_tens :=
sorry

end NUMINAMATH_CALUDE_probability_different_tens_proof_l268_26842


namespace NUMINAMATH_CALUDE_root_transformation_l268_26851

theorem root_transformation (α β : ℝ) : 
  (2 * α^2 - 5 * α + 3 = 0) → 
  (2 * β^2 - 5 * β + 3 = 0) → 
  ((2 * α - 7)^2 + 9 * (2 * α - 7) + 20 = 0) ∧
  ((2 * β - 7)^2 + 9 * (2 * β - 7) + 20 = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_transformation_l268_26851


namespace NUMINAMATH_CALUDE_worker_arrival_delay_l268_26892

theorem worker_arrival_delay (usual_speed : ℝ) (usual_time : ℝ) : 
  usual_time = 60 → 
  let new_speed := (4/5) * usual_speed
  let new_time := usual_time * (usual_speed / new_speed)
  new_time - usual_time = 15 := by sorry

end NUMINAMATH_CALUDE_worker_arrival_delay_l268_26892


namespace NUMINAMATH_CALUDE_committee_selections_of_seven_l268_26879

/-- The number of ways to select a chairperson and a deputy chairperson from a committee. -/
def committee_selections (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: The number of ways to select a chairperson and a deputy chairperson 
    from a committee of 7 members is 42. -/
theorem committee_selections_of_seven : committee_selections 7 = 42 := by
  sorry

end NUMINAMATH_CALUDE_committee_selections_of_seven_l268_26879


namespace NUMINAMATH_CALUDE_all_zero_function_l268_26807

-- Define the type of our function
def IntFunction := Nat → Nat

-- Define the conditions
def satisfiesConditions (f : IntFunction) : Prop :=
  (∀ m n : Nat, m > 0 ∧ n > 0 → f (m * n) = f m + f n) ∧
  (f 2008 = 0) ∧
  (∀ n : Nat, n > 0 ∧ n % 2008 = 39 → f n = 0)

-- State the theorem
theorem all_zero_function (f : IntFunction) :
  satisfiesConditions f → ∀ n : Nat, n > 0 → f n = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_all_zero_function_l268_26807


namespace NUMINAMATH_CALUDE_equation_solution_l268_26875

theorem equation_solution : ∃ n : ℕ, 3^n * 9^n = 81^(n-12) ∧ n = 48 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l268_26875


namespace NUMINAMATH_CALUDE_circle_area_equal_perimeter_l268_26825

theorem circle_area_equal_perimeter (s : ℝ) (r : ℝ) : 
  s > 0 → 
  r > 0 → 
  s^2 = 16 → 
  4 * s = 2 * Real.pi * r → 
  Real.pi * r^2 = 64 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circle_area_equal_perimeter_l268_26825


namespace NUMINAMATH_CALUDE_solution_set_and_range_of_a_l268_26895

def f (a x : ℝ) : ℝ := |x - a| + x

theorem solution_set_and_range_of_a :
  (∀ x : ℝ, f 3 x ≥ x + 4 ↔ (x ≤ -1 ∨ x ≥ 7)) ∧
  (∀ a : ℝ, a > 0 →
    (∀ x : ℝ, x ∈ Set.Icc 1 3 → f a x ≥ x + 2 * a^2) ↔
    (-1 ≤ a ∧ a ≤ 1/2)) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_and_range_of_a_l268_26895


namespace NUMINAMATH_CALUDE_find_x_l268_26877

theorem find_x : ∃ x : ℝ, (3 * x + 5) / 5 = 13 ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l268_26877


namespace NUMINAMATH_CALUDE_fundraiser_percentage_increase_l268_26819

def fundraiser (initial_rate : ℝ) (total_hours : ℕ) (initial_hours : ℕ) (total_amount : ℝ) : Prop :=
  let remaining_hours := total_hours - initial_hours
  let initial_amount := initial_rate * initial_hours
  let remaining_amount := total_amount - initial_amount
  let new_rate := remaining_amount / remaining_hours
  let percentage_increase := (new_rate - initial_rate) / initial_rate * 100
  percentage_increase = 20

theorem fundraiser_percentage_increase :
  fundraiser 5000 26 12 144000 := by
  sorry

end NUMINAMATH_CALUDE_fundraiser_percentage_increase_l268_26819


namespace NUMINAMATH_CALUDE_cos_fifteen_squared_formula_l268_26829

theorem cos_fifteen_squared_formula : 2 * (Real.cos (15 * π / 180))^2 - 1 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_fifteen_squared_formula_l268_26829


namespace NUMINAMATH_CALUDE_locus_of_P_is_ellipse_l268_26821

-- Define the circle F₁
def circle_F₁ (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 36

-- Define the fixed point F₂
def F₂ : ℝ × ℝ := (2, 0)

-- Define a point on the circle F₁
def point_on_F₁ (A : ℝ × ℝ) : Prop := circle_F₁ A.1 A.2

-- Define the center of F₁
def F₁ : ℝ × ℝ := (-2, 0)

-- Define the perpendicular bisector of F₂A
def perp_bisector (A : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  (P.1 - F₂.1) * (A.1 - F₂.1) + (P.2 - F₂.2) * (A.2 - F₂.2) = 0 ∧
  (P.1 - F₂.1)^2 + (P.2 - F₂.2)^2 = (P.1 - A.1)^2 + (P.2 - A.2)^2

-- Define P as the intersection of perpendicular bisector and radius F₁A
def point_P (A : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  perp_bisector A P ∧
  ∃ (t : ℝ), P = (F₁.1 + t * (A.1 - F₁.1), F₁.2 + t * (A.2 - F₁.2))

-- Theorem: The locus of P is an ellipse with equation x²/9 + y²/5 = 1
theorem locus_of_P_is_ellipse :
  ∀ (P : ℝ × ℝ), (∃ (A : ℝ × ℝ), point_on_F₁ A ∧ point_P A P) ↔ 
  P.1^2 / 9 + P.2^2 / 5 = 1 :=
sorry

end NUMINAMATH_CALUDE_locus_of_P_is_ellipse_l268_26821


namespace NUMINAMATH_CALUDE_fraction_equality_l268_26816

theorem fraction_equality : (24 + 12) / ((5 - 3) * 2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l268_26816


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l268_26808

-- Define an isosceles triangle
structure IsoscelesTriangle where
  baseAngle : ℝ
  vertexAngle : ℝ
  isIsosceles : baseAngle ≥ 0 ∧ vertexAngle ≥ 0
  angleSum : baseAngle + baseAngle + vertexAngle = 180

-- Theorem statement
theorem isosceles_triangle_vertex_angle 
  (triangle : IsoscelesTriangle) 
  (h : triangle.baseAngle = 80) : 
  triangle.vertexAngle = 20 :=
by
  sorry

#check isosceles_triangle_vertex_angle

end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l268_26808


namespace NUMINAMATH_CALUDE_jackies_lotion_order_l268_26884

/-- The number of lotion bottles Jackie ordered -/
def lotion_bottles : ℕ := 3

/-- The free shipping threshold in cents -/
def free_shipping_threshold : ℕ := 5000

/-- The total cost of shampoo and conditioner in cents -/
def shampoo_conditioner_cost : ℕ := 2000

/-- The cost of one bottle of lotion in cents -/
def lotion_cost : ℕ := 600

/-- The additional amount Jackie needs to spend to reach the free shipping threshold in cents -/
def additional_spend : ℕ := 1200

theorem jackies_lotion_order :
  lotion_bottles * lotion_cost = free_shipping_threshold - shampoo_conditioner_cost - additional_spend :=
by sorry

end NUMINAMATH_CALUDE_jackies_lotion_order_l268_26884


namespace NUMINAMATH_CALUDE_translator_selection_count_l268_26876

/-- Represents the number of translators for each category -/
structure TranslatorCounts where
  total : Nat
  english : Nat
  japanese : Nat
  both : Nat

/-- Represents the required number of translators for each language -/
structure RequiredTranslators where
  english : Nat
  japanese : Nat

/-- Calculates the number of ways to select translators given the constraints -/
def countTranslatorSelections (counts : TranslatorCounts) (required : RequiredTranslators) : Nat :=
  sorry

/-- Theorem stating that there are 29 different ways to select the translators -/
theorem translator_selection_count :
  let counts : TranslatorCounts := ⟨8, 3, 3, 2⟩
  let required : RequiredTranslators := ⟨3, 2⟩
  countTranslatorSelections counts required = 29 :=
by sorry

end NUMINAMATH_CALUDE_translator_selection_count_l268_26876


namespace NUMINAMATH_CALUDE_opposite_corners_not_tileable_different_color_cells_tileable_l268_26834

/-- Represents a chessboard -/
structure Chessboard :=
  (size : Nat)
  (removed : List (Nat × Nat))

/-- Represents a domino -/
inductive Domino
  | horizontal : Nat → Nat → Domino
  | vertical : Nat → Nat → Domino

/-- A tiling of a chessboard with dominoes -/
def Tiling := List Domino

/-- Returns true if the given coordinates represent a black square on the chessboard -/
def isBlack (x y : Nat) : Bool :=
  (x + y) % 2 = 0

/-- Returns true if the two given cells have different colors -/
def differentColors (x1 y1 x2 y2 : Nat) : Bool :=
  isBlack x1 y1 ≠ isBlack x2 y2

/-- Returns true if the given tiling is valid for the given chessboard -/
def isValidTiling (board : Chessboard) (tiling : Tiling) : Bool :=
  sorry

theorem opposite_corners_not_tileable :
  ∀ (board : Chessboard),
    board.size = 8 →
    board.removed = [(0, 0), (7, 7)] →
    ¬∃ (tiling : Tiling), isValidTiling board tiling :=
  sorry

theorem different_color_cells_tileable :
  ∀ (board : Chessboard) (x1 y1 x2 y2 : Nat),
    board.size = 8 →
    x1 < 8 ∧ y1 < 8 ∧ x2 < 8 ∧ y2 < 8 →
    differentColors x1 y1 x2 y2 →
    board.removed = [(x1, y1), (x2, y2)] →
    ∃ (tiling : Tiling), isValidTiling board tiling :=
  sorry

end NUMINAMATH_CALUDE_opposite_corners_not_tileable_different_color_cells_tileable_l268_26834


namespace NUMINAMATH_CALUDE_words_with_consonant_l268_26826

def letter_set : Finset Char := {'A', 'B', 'C', 'D', 'E'}
def vowel_set : Finset Char := {'A', 'E'}
def word_length : Nat := 5

def total_words : Nat := letter_set.card ^ word_length
def all_vowel_words : Nat := vowel_set.card ^ word_length

theorem words_with_consonant :
  total_words - all_vowel_words = 3093 :=
sorry

end NUMINAMATH_CALUDE_words_with_consonant_l268_26826


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_is_three_root_six_over_four_l268_26856

/-- A pyramid with a square base and equilateral triangle lateral faces -/
structure Pyramid :=
  (base_side : ℝ)
  (base_is_square : base_side > 0)
  (lateral_faces_are_equilateral : True)

/-- A cube inscribed in a pyramid -/
structure InscribedCube (p : Pyramid) :=
  (side_length : ℝ)
  (base_on_pyramid_base : True)
  (top_vertices_touch_midpoints : True)

/-- The volume of the inscribed cube -/
def inscribed_cube_volume (p : Pyramid) (c : InscribedCube p) : ℝ :=
  c.side_length ^ 3

/-- Main theorem: The volume of the inscribed cube is 3√6/4 -/
theorem inscribed_cube_volume_is_three_root_six_over_four 
  (p : Pyramid) 
  (h_base : p.base_side = 2)
  (c : InscribedCube p) : 
  inscribed_cube_volume p c = 3 * Real.sqrt 6 / 4 :=
sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_is_three_root_six_over_four_l268_26856


namespace NUMINAMATH_CALUDE_warehouse_theorem_l268_26878

def warehouse_problem (second_floor_space : ℝ) (boxes_space : ℝ) : Prop :=
  let first_floor_space := 2 * second_floor_space
  let total_space := first_floor_space + second_floor_space
  let available_space := total_space - boxes_space
  (boxes_space = 5000) ∧
  (boxes_space = second_floor_space / 4) ∧
  (available_space = 55000)

theorem warehouse_theorem :
  ∃ (second_floor_space : ℝ), warehouse_problem second_floor_space 5000 :=
sorry

end NUMINAMATH_CALUDE_warehouse_theorem_l268_26878


namespace NUMINAMATH_CALUDE_shelby_total_stars_l268_26824

/-- The number of gold stars Shelby earned yesterday -/
def stars_yesterday : ℕ := 4

/-- The number of gold stars Shelby earned today -/
def stars_today : ℕ := 3

/-- The total number of gold stars Shelby earned -/
def total_stars : ℕ := stars_yesterday + stars_today

/-- Theorem stating that the total number of gold stars Shelby earned is 7 -/
theorem shelby_total_stars : total_stars = 7 := by sorry

end NUMINAMATH_CALUDE_shelby_total_stars_l268_26824


namespace NUMINAMATH_CALUDE_infinite_series_sum_l268_26898

open Real
open BigOperators

theorem infinite_series_sum : 
  (∑' n : ℕ, (3 * n + 2) / (n * (n + 1) * (n + 3))) = 10/3 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l268_26898


namespace NUMINAMATH_CALUDE_compute_expression_l268_26861

theorem compute_expression : 8 * (1 / 3)^3 - 1 = -19 / 27 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l268_26861


namespace NUMINAMATH_CALUDE_cubic_root_equation_solution_l268_26872

theorem cubic_root_equation_solution :
  ∀ x : ℝ, (((30 * x + (30 * x + 18) ^ (1/3)) ^ (1/3)) = 18) → x = 2907/15 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_equation_solution_l268_26872


namespace NUMINAMATH_CALUDE_de_bruijn_semi_integer_l268_26873

/-- A semi-integer rectangle is a rectangle where at least one vertex has integer coordinates -/
def SemiIntegerRectangle (d : ℕ) := (Fin d → ℝ) → Prop

/-- A box with dimensions B₁, ..., Bₗ -/
def Box (l : ℕ) (B : Fin l → ℝ) := Set (Fin l → ℝ)

/-- A block with dimensions b₁, ..., bₖ -/
def Block (k : ℕ) (b : Fin k → ℝ) := Set (Fin k → ℝ)

/-- A tiling of a box by blocks -/
def Tiling (l k : ℕ) (B : Fin l → ℝ) (b : Fin k → ℝ) := 
  Box l B → Set (Block k b)

theorem de_bruijn_semi_integer 
  (l k : ℕ) (B : Fin l → ℝ) (b : Fin k → ℝ) 
  (tiling : Tiling l k B b) 
  (semi_int : SemiIntegerRectangle k) :
  ∀ i, ∃ j, ∃ (n : ℕ), B j = n * b i :=
sorry

end NUMINAMATH_CALUDE_de_bruijn_semi_integer_l268_26873


namespace NUMINAMATH_CALUDE_total_peaches_l268_26830

/-- The number of peaches initially in the basket -/
def initial_peaches : ℕ := 20

/-- The number of peaches added to the basket -/
def added_peaches : ℕ := 25

/-- Theorem stating the total number of peaches after addition -/
theorem total_peaches : initial_peaches + added_peaches = 45 := by
  sorry

end NUMINAMATH_CALUDE_total_peaches_l268_26830


namespace NUMINAMATH_CALUDE_polynomial_symmetry_l268_26882

/-- Given a polynomial function f(x) = ax^5 + bx^3 + cx + 7 where a, b, c are real constants,
    if f(-2011) = -17, then f(2011) = 31 -/
theorem polynomial_symmetry (a b c : ℝ) :
  let f := λ x : ℝ => a * x^5 + b * x^3 + c * x + 7
  (f (-2011) = -17) → (f 2011 = 31) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_l268_26882


namespace NUMINAMATH_CALUDE_greatest_x_value_l268_26894

theorem greatest_x_value : ∃ (x_max : ℝ),
  (∀ x : ℝ, (x^2 - 3*x - 70) / (x - 10) = 5 / (x + 7) → x ≤ x_max) ∧
  ((x_max^2 - 3*x_max - 70) / (x_max - 10) = 5 / (x_max + 7)) ∧
  x_max = -2 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l268_26894


namespace NUMINAMATH_CALUDE_zero_of_f_l268_26836

/-- The function f(x) = (x+1)^2 -/
def f (x : ℝ) : ℝ := (x + 1)^2

/-- Theorem: -1 is the zero of the function f(x) = (x+1)^2 -/
theorem zero_of_f : f (-1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_of_f_l268_26836


namespace NUMINAMATH_CALUDE_tan_C_value_triangle_area_l268_26818

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
def satisfies_condition_1 (t : Triangle) : Prop :=
  (Real.sin t.A / t.a) + (Real.sin t.B / t.b) = (Real.cos t.C / t.c)

def satisfies_condition_2 (t : Triangle) : Prop :=
  t.a^2 + t.b^2 - t.c^2 = 8

-- Theorem 1
theorem tan_C_value (t : Triangle) (h : satisfies_condition_1 t) :
  Real.tan t.C = 1/2 := by sorry

-- Theorem 2
theorem triangle_area (t : Triangle) (h1 : satisfies_condition_1 t) (h2 : satisfies_condition_2 t) :
  (1/2) * t.a * t.b * Real.sin t.C = 1 := by sorry

end NUMINAMATH_CALUDE_tan_C_value_triangle_area_l268_26818


namespace NUMINAMATH_CALUDE_simplify_expression_l268_26868

theorem simplify_expression (x : ℝ) : 3*x^2 + 4 - 5*x^3 - x^3 + 3 - 3*x^2 = -6*x^3 + 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l268_26868


namespace NUMINAMATH_CALUDE_circle_with_common_chord_as_diameter_l268_26832

/-- C₁ is the first given circle -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 4*x + y + 1 = 0

/-- C₂ is the second given circle -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y + 1 = 0

/-- C is the circle we need to prove -/
def C (x y : ℝ) : Prop := 5*x^2 + 5*y^2 + 6*x + 12*y + 5 = 0

/-- The common chord of C₁ and C₂ -/
def common_chord (x y : ℝ) : Prop := y = 2*x

theorem circle_with_common_chord_as_diameter :
  ∀ x y : ℝ, C x y ↔ 
    (∃ a b : ℝ, C₁ a b ∧ C₂ a b ∧ common_chord a b ∧
      (x - a)^2 + (y - b)^2 = ((x - a) - (b - y))^2 / 4) :=
sorry

end NUMINAMATH_CALUDE_circle_with_common_chord_as_diameter_l268_26832


namespace NUMINAMATH_CALUDE_green_rotten_no_smell_count_l268_26805

/-- Represents the types of fruits in the orchard -/
inductive Fruit
| Apple
| Orange
| Pear

/-- Represents the colors of fruits -/
inductive Color
| Red
| Green
| Orange
| Yellow
| Brown

structure OrchardData where
  total_fruits : Fruit → ℕ
  color_distribution : Fruit → Color → ℚ
  rotten_percentage : Fruit → ℚ
  strong_smell_percentage : Fruit → ℚ

def orchard_data : OrchardData := {
  total_fruits := λ f => match f with
    | Fruit.Apple => 200
    | Fruit.Orange => 150
    | Fruit.Pear => 100,
  color_distribution := λ f c => match f, c with
    | Fruit.Apple, Color.Red => 1/2
    | Fruit.Apple, Color.Green => 1/2
    | Fruit.Orange, Color.Orange => 2/5
    | Fruit.Orange, Color.Yellow => 3/5
    | Fruit.Pear, Color.Green => 3/10
    | Fruit.Pear, Color.Brown => 7/10
    | _, _ => 0,
  rotten_percentage := λ f => match f with
    | Fruit.Apple => 2/5
    | Fruit.Orange => 1/4
    | Fruit.Pear => 7/20,
  strong_smell_percentage := λ f => match f with
    | Fruit.Apple => 7/10
    | Fruit.Orange => 1/2
    | Fruit.Pear => 4/5
}

/-- Calculates the number of green rotten fruits without a strong smell in the orchard -/
def green_rotten_no_smell (data : OrchardData) : ℕ :=
  sorry

theorem green_rotten_no_smell_count :
  green_rotten_no_smell orchard_data = 14 :=
sorry

end NUMINAMATH_CALUDE_green_rotten_no_smell_count_l268_26805


namespace NUMINAMATH_CALUDE_additional_songs_count_l268_26813

def original_songs : ℕ := 25
def song_duration : ℕ := 3
def total_duration : ℕ := 105

theorem additional_songs_count :
  (total_duration - original_songs * song_duration) / song_duration = 10 := by
  sorry

end NUMINAMATH_CALUDE_additional_songs_count_l268_26813


namespace NUMINAMATH_CALUDE_evaluate_expression_l268_26810

theorem evaluate_expression (b : ℚ) (h : b = 4/3) :
  (6*b^2 - 17*b + 8) * (3*b - 4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l268_26810


namespace NUMINAMATH_CALUDE_decreasing_function_inequality_l268_26844

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem decreasing_function_inequality
  (f : ℝ → ℝ) (h_dec : is_decreasing f) (m n : ℝ)
  (h_ineq : f m - f n > f (-m) - f (-n)) :
  m - n < 0 :=
sorry

end NUMINAMATH_CALUDE_decreasing_function_inequality_l268_26844


namespace NUMINAMATH_CALUDE_cost_price_approximation_l268_26837

/-- The cost price of a single toy given the selling conditions -/
def cost_price_of_toy (num_toys : ℕ) (total_selling_price : ℚ) (gain_in_toys : ℕ) : ℚ :=
  let selling_price_per_toy := total_selling_price / num_toys
  let x := selling_price_per_toy / (1 + gain_in_toys / num_toys)
  x

/-- Theorem stating the cost price of a toy given the problem conditions -/
theorem cost_price_approximation :
  let result := cost_price_of_toy 18 23100 3
  (result > 1099.99) ∧ (result < 1100.01) := by
  sorry

#eval cost_price_of_toy 18 23100 3

end NUMINAMATH_CALUDE_cost_price_approximation_l268_26837


namespace NUMINAMATH_CALUDE_function_properties_a_value_l268_26846

noncomputable section

-- Define the natural exponential function
def exp (x : ℝ) := Real.exp x

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (a * exp x - a - x) * exp x

theorem function_properties (h : ∀ x : ℝ, f 1 x ≥ 0) :
  (∃! x₀ : ℝ, ∀ x : ℝ, f 1 x ≤ f 1 x₀) ∧
  (∃ x₀ : ℝ, ∀ x : ℝ, f 1 x ≤ f 1 x₀ ∧ 0 < f 1 x₀ ∧ f 1 x₀ < 1/4) :=
sorry

theorem a_value (h : ∀ a : ℝ, a ≥ 0 → ∀ x : ℝ, f a x ≥ 0) :
  ∃! a : ℝ, a = 1 ∧ ∀ x : ℝ, f a x ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_function_properties_a_value_l268_26846


namespace NUMINAMATH_CALUDE_f_plus_one_nonnegative_min_a_value_l268_26870

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * (Real.log x - 1)

-- Theorem 1: f(x) + 1 ≥ 0 for all x > 0
theorem f_plus_one_nonnegative : ∀ x > 0, f x + 1 ≥ 0 := by sorry

-- Theorem 2: The minimum value of a such that 4f'(x) ≤ a(x+1) - 8 for all x > 0 is 4
theorem min_a_value : 
  (∃ a : ℝ, ∀ x > 0, 4 * (Real.log x) ≤ a * (x + 1) - 8) ∧ 
  (∀ a < 4, ∃ x > 0, 4 * (Real.log x) > a * (x + 1) - 8) := by sorry

end NUMINAMATH_CALUDE_f_plus_one_nonnegative_min_a_value_l268_26870


namespace NUMINAMATH_CALUDE_dima_puts_more_berries_l268_26822

/-- Represents the berry-picking process of Dima and Sergey -/
structure BerryPicking where
  total_berries : ℕ
  dima_basket_rate : ℚ
  sergey_basket_rate : ℚ
  dima_speed : ℚ
  sergey_speed : ℚ

/-- Calculates the difference in berries put in the basket by Dima and Sergey -/
def berry_difference (bp : BerryPicking) : ℕ :=
  sorry

/-- Theorem stating the difference in berries put in the basket -/
theorem dima_puts_more_berries (bp : BerryPicking) 
  (h1 : bp.total_berries = 900)
  (h2 : bp.dima_basket_rate = 1/2)
  (h3 : bp.sergey_basket_rate = 2/3)
  (h4 : bp.dima_speed = 2 * bp.sergey_speed) :
  berry_difference bp = 100 :=
sorry

end NUMINAMATH_CALUDE_dima_puts_more_berries_l268_26822


namespace NUMINAMATH_CALUDE_product_of_two_numbers_l268_26859

theorem product_of_two_numbers (x y : ℝ) 
  (h1 : x * y = 15 * (x - y)) 
  (h2 : x + y = 8 * (x - y)) : 
  x * y = 100 / 7 := by
sorry

end NUMINAMATH_CALUDE_product_of_two_numbers_l268_26859


namespace NUMINAMATH_CALUDE_roys_pen_ratio_l268_26847

/-- Proves that the ratio of black pens to blue pens is 2:1 given the conditions of Roy's pen collection --/
theorem roys_pen_ratio :
  ∀ (blue black red : ℕ),
    blue = 2 →
    red = 2 * black - 2 →
    blue + black + red = 12 →
    black / blue = 2 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_roys_pen_ratio_l268_26847
