import Mathlib

namespace quadratic_equation_solution_l1327_132721

theorem quadratic_equation_solution (c : ℝ) : 
  (∃ x : ℝ, x^2 - x + c = 0 ∧ x = 1) → 
  (∃ x : ℝ, x^2 - x + c = 0 ∧ x = 0) :=
by sorry

end quadratic_equation_solution_l1327_132721


namespace danny_share_l1327_132732

/-- Represents the share of money each person receives -/
structure Share :=
  (amount : ℝ)
  (removed : ℝ)

/-- The problem setup -/
def problem_setup :=
  (total : ℝ) →
  (alice : Share) →
  (bond : Share) →
  (charlie : Share) →
  (danny : Share) →
  Prop

/-- The conditions of the problem -/
def conditions (total : ℝ) (alice bond charlie danny : Share) : Prop :=
  total = 2210 ∧
  alice.removed = 30 ∧
  bond.removed = 50 ∧
  charlie.removed = 40 ∧
  danny.removed = 2 * charlie.removed ∧
  (alice.amount - alice.removed) / (bond.amount - bond.removed) = 11 / 18 ∧
  (alice.amount - alice.removed) / (charlie.amount - charlie.removed) = 11 / 24 ∧
  (alice.amount - alice.removed) / (danny.amount - danny.removed) = 11 / 32 ∧
  alice.amount + bond.amount + charlie.amount + danny.amount = total

/-- The theorem to prove -/
theorem danny_share (total : ℝ) (alice bond charlie danny : Share) :
  conditions total alice bond charlie danny →
  danny.amount = 916.80 :=
sorry

end danny_share_l1327_132732


namespace opposite_sum_zero_sum_zero_opposite_exists_opposite_not_negative_one_negative_one_ratio_opposite_l1327_132759

-- Define opposite numbers
def opposite (a b : ℝ) : Prop := a = -b

-- Statement 1
theorem opposite_sum_zero (a b : ℝ) : opposite a b → a + b = 0 := by sorry

-- Statement 2
theorem sum_zero_opposite (a b : ℝ) : a + b = 0 → opposite a b := by sorry

-- Statement 3
theorem exists_opposite_not_negative_one : ∃ a b : ℝ, opposite a b ∧ a / b ≠ -1 := by sorry

-- Statement 4
theorem negative_one_ratio_opposite (a b : ℝ) (h : b ≠ 0) : a / b = -1 → opposite a b := by sorry

end opposite_sum_zero_sum_zero_opposite_exists_opposite_not_negative_one_negative_one_ratio_opposite_l1327_132759


namespace combined_molecular_weight_l1327_132738

/-- The atomic weight of Hydrogen in atomic mass units (amu) -/
def H_weight : ℝ := 1.008

/-- The atomic weight of Oxygen in atomic mass units (amu) -/
def O_weight : ℝ := 15.999

/-- The atomic weight of Carbon in atomic mass units (amu) -/
def C_weight : ℝ := 12.011

/-- The molecular weight of H2O in atomic mass units (amu) -/
def H2O_weight : ℝ := 2 * H_weight + O_weight

/-- The molecular weight of CO2 in atomic mass units (amu) -/
def CO2_weight : ℝ := C_weight + 2 * O_weight

/-- The molecular weight of CH4 in atomic mass units (amu) -/
def CH4_weight : ℝ := C_weight + 4 * H_weight

/-- The combined molecular weight of H2O, CO2, and CH4 in atomic mass units (amu) -/
def combined_weight : ℝ := H2O_weight + CO2_weight + CH4_weight

theorem combined_molecular_weight :
  combined_weight = 78.067 := by sorry

end combined_molecular_weight_l1327_132738


namespace power_equation_solution_l1327_132740

theorem power_equation_solution : ∃ K : ℕ, (81 ^ 2) * (27 ^ 3) = 3 ^ K ∧ K = 17 := by
  sorry

end power_equation_solution_l1327_132740


namespace root_implies_q_value_l1327_132714

theorem root_implies_q_value (p q : ℝ) :
  (Complex.I : ℂ) ^ 2 = -1 →
  (3 : ℂ) * (5 + Complex.I) ^ 2 + p * (5 + Complex.I) + q = 0 →
  q = 78 := by
  sorry

end root_implies_q_value_l1327_132714


namespace cube_face_projections_l1327_132764

/-- Given three faces of a unit cube sharing a common vertex, if their projections onto a fixed plane
have areas in the ratio 6:10:15, then the sum of these areas is 31/19. -/
theorem cube_face_projections (x y z : ℝ) : 
  x > 0 ∧ y > 0 ∧ z > 0 →  -- Ensure positive areas
  x^2 + y^2 + z^2 = 1 →  -- Sum of squares of projection areas equals 1
  x / 6 = y / 10 ∧ y / 10 = z / 15 →  -- Ratio condition
  x + y + z = 31 / 19 := by
sorry

end cube_face_projections_l1327_132764


namespace tower_surface_area_theorem_l1327_132781

def cube_volume (v : ℝ) : ℝ := v

def cube_side_length (v : ℝ) : ℝ := v ^ (1/3)

def cube_surface_area (s : ℝ) : ℝ := 6 * s^2

def tower_surface_area (v1 v2 v3 v4 : ℝ) : ℝ :=
  let s1 := cube_side_length v1
  let s2 := cube_side_length v2
  let s3 := cube_side_length v3
  let s4 := cube_side_length v4
  cube_surface_area s1 + 
  (cube_surface_area s2 - s1^2) + 
  (cube_surface_area s3 - s2^2) + 
  cube_surface_area s4

theorem tower_surface_area_theorem :
  tower_surface_area 1 27 125 343 = 494 := by sorry

end tower_surface_area_theorem_l1327_132781


namespace tangent_line_equation_l1327_132795

-- Define the curve
def f (x : ℝ) : ℝ := x^3

-- Define the point of tangency
def P : ℝ × ℝ := (1, 1)

-- Define the slope of the tangent line at P
def m : ℝ := 3

-- Statement of the theorem
theorem tangent_line_equation :
  ∀ x y : ℝ, (x - P.1) * m = y - P.2 ↔ 3*x - y - 2 = 0 :=
sorry

end tangent_line_equation_l1327_132795


namespace scientific_notation_equivalence_l1327_132734

theorem scientific_notation_equivalence : 
  274000000 = 2.74 * (10 ^ 8) := by sorry

end scientific_notation_equivalence_l1327_132734


namespace call_processing_ratio_l1327_132713

/-- Represents the ratio of Team A members to Team B members -/
def team_ratio : ℚ := 5 / 8

/-- Represents the fraction of total calls processed by Team B -/
def team_b_calls : ℚ := 8 / 9

/-- Proves that the ratio of calls processed by each member of Team A to each member of Team B is 1:5 -/
theorem call_processing_ratio :
  let team_a_calls := 1 - team_b_calls
  let team_a_members := team_ratio * team_b_members
  (team_a_calls / team_a_members) / (team_b_calls / team_b_members) = 1 / 5 :=
by sorry

end call_processing_ratio_l1327_132713


namespace square_partition_theorem_l1327_132709

/-- A rectangle with side lengths a and b -/
structure Rectangle where
  a : ℝ
  b : ℝ
  a_pos : 0 < a
  b_pos : 0 < b

/-- Predicate indicating if one rectangle can be placed inside another (possibly with rotation) -/
def can_fit_inside (r1 r2 : Rectangle) : Prop :=
  (r1.a ≤ r2.a ∧ r1.b ≤ r2.b) ∨ (r1.a ≤ r2.b ∧ r1.b ≤ r2.a)

theorem square_partition_theorem (n : ℕ) (hn : n^2 ≥ 4) :
  ∃ (rectangles : Fin (n^2) → Rectangle),
    (∀ i j, i ≠ j → rectangles i ≠ rectangles j) →
    (∃ (chosen : Fin (2*n) → Fin (n^2)),
      ∀ i j, i < j → can_fit_inside (rectangles (chosen i)) (rectangles (chosen j))) :=
  sorry

end square_partition_theorem_l1327_132709


namespace evan_future_books_l1327_132799

/-- Calculates the number of books Evan will have in 5 years -/
def books_in_five_years (books_two_years_ago : ℕ) : ℕ :=
  let current_books := books_two_years_ago - 40
  5 * current_books + 60

/-- Proves that Evan will have 860 books in 5 years -/
theorem evan_future_books :
  books_in_five_years 200 = 860 := by
  sorry

#eval books_in_five_years 200

end evan_future_books_l1327_132799


namespace sqrt_problem_1_sqrt_problem_2_l1327_132719

-- Problem 1
theorem sqrt_problem_1 : Real.sqrt 6 * Real.sqrt 3 - 6 * Real.sqrt (1/2) = 0 := by sorry

-- Problem 2
theorem sqrt_problem_2 : (Real.sqrt 20 + Real.sqrt 5) / Real.sqrt 5 = 3 := by sorry

end sqrt_problem_1_sqrt_problem_2_l1327_132719


namespace min_cubes_surface_area_52_l1327_132728

/-- The surface area of a rectangular prism -/
def surface_area (l w h : ℕ) : ℕ := 2 * (l * w + l * h + w * h)

/-- The volume of a rectangular prism -/
def volume (l w h : ℕ) : ℕ := l * w * h

/-- The minimum number of unit cubes needed to form a rectangular prism with surface area 52 -/
theorem min_cubes_surface_area_52 :
  (∃ l w h : ℕ, surface_area l w h = 52 ∧ 
    volume l w h = 16 ∧
    ∀ l' w' h' : ℕ, surface_area l' w' h' = 52 → volume l' w' h' ≥ 16) :=
by sorry

end min_cubes_surface_area_52_l1327_132728


namespace expression_evaluation_l1327_132787

theorem expression_evaluation :
  let x : ℤ := -1
  let y : ℤ := 2
  (2 * x^2 - 2 * y^2) - 3 * (x^2 * y^2 + x^2) + 3 * (x^2 * y^2 + y^2) = 3 := by
  sorry

end expression_evaluation_l1327_132787


namespace function_equality_l1327_132708

theorem function_equality (f : ℝ → ℝ) :
  (∀ x : ℝ, f (x + 1) = 2 * x - 1) →
  (∀ x : ℝ, f x = 2 * x - 3) :=
by
  sorry

end function_equality_l1327_132708


namespace solution_check_l1327_132784

-- Define the equation
def equation (x y : ℚ) : Prop := x - 2 * y = 1

-- Define the sets of values
def setA : ℚ × ℚ := (0, -1/2)
def setB : ℚ × ℚ := (1, 1)
def setC : ℚ × ℚ := (1, 0)
def setD : ℚ × ℚ := (-1, -1)

-- Theorem stating that setB is not a solution while others are
theorem solution_check :
  ¬(equation setB.1 setB.2) ∧
  (equation setA.1 setA.2) ∧
  (equation setC.1 setC.2) ∧
  (equation setD.1 setD.2) :=
sorry

end solution_check_l1327_132784


namespace signals_coincide_l1327_132700

def town_hall_period : ℕ := 18
def library_period : ℕ := 24
def fire_station_period : ℕ := 36

def coincidence_time : ℕ := 72

theorem signals_coincide :
  coincidence_time = Nat.lcm town_hall_period (Nat.lcm library_period fire_station_period) :=
by sorry

end signals_coincide_l1327_132700


namespace f_is_quasi_even_l1327_132786

/-- A function is quasi-even if f(-x) = f(x) only for a finite number of non-zero arguments x. -/
def QuasiEven (f : ℝ → ℝ) : Prop :=
  ∃ (S : Finset ℝ), ∀ x ≠ 0, f (-x) = f x ↔ x ∈ S

/-- The function f(x) = x³ - 2x -/
def f (x : ℝ) : ℝ := x^3 - 2*x

/-- Theorem: f(x) = x³ - 2x is a quasi-even function -/
theorem f_is_quasi_even : QuasiEven f := by
  sorry

end f_is_quasi_even_l1327_132786


namespace adjacent_roll_probability_adjacent_roll_probability_proof_l1327_132704

/-- The probability that no two adjacent people roll the same number on an eight-sided die
    when six people sit around a circular table. -/
theorem adjacent_roll_probability : ℚ :=
  117649 / 262144

/-- The number of people sitting around the circular table. -/
def num_people : ℕ := 6

/-- The number of sides on the die. -/
def die_sides : ℕ := 8

/-- The probability of rolling a different number than the previous person. -/
def diff_roll_prob : ℚ := 7 / 8

theorem adjacent_roll_probability_proof :
  adjacent_roll_probability = diff_roll_prob ^ num_people :=
sorry

end adjacent_roll_probability_adjacent_roll_probability_proof_l1327_132704


namespace hclo4_moles_required_l1327_132758

theorem hclo4_moles_required (naoh_moles : ℝ) (naclo4_moles : ℝ) (h2o_moles : ℝ) 
  (hclo4_participation_rate : ℝ) :
  naoh_moles = 3 →
  naclo4_moles = 3 →
  h2o_moles = 3 →
  hclo4_participation_rate = 0.8 →
  ∃ (hclo4_moles : ℝ), 
    hclo4_moles = naoh_moles / hclo4_participation_rate ∧ 
    hclo4_moles = 3.75 :=
by sorry

end hclo4_moles_required_l1327_132758


namespace tims_doctor_visit_cost_l1327_132753

theorem tims_doctor_visit_cost (tim_total_payment : ℝ) (cat_visit_cost : ℝ) (cat_insurance_coverage : ℝ) (tim_insurance_coverage_percent : ℝ) : 
  tim_total_payment = 135 →
  cat_visit_cost = 120 →
  cat_insurance_coverage = 60 →
  tim_insurance_coverage_percent = 75 →
  ∃ (doctor_visit_cost : ℝ),
    doctor_visit_cost = 300 ∧
    tim_total_payment = (1 - tim_insurance_coverage_percent / 100) * doctor_visit_cost + (cat_visit_cost - cat_insurance_coverage) :=
by sorry

end tims_doctor_visit_cost_l1327_132753


namespace similar_triangle_perimeter_l1327_132788

theorem similar_triangle_perimeter (a b c : ℝ) (h1 : a = 12) (h2 : b = 12) (h3 : c = 15) 
  (h4 : a = b) (h5 : c ≥ a) (h6 : c ≥ b) (long_side : ℝ) (h7 : long_side = 45) : 
  (long_side / c) * (a + b + c) = 117 :=
by sorry

end similar_triangle_perimeter_l1327_132788


namespace sqrt_eighteen_div_sqrt_two_equals_three_l1327_132729

theorem sqrt_eighteen_div_sqrt_two_equals_three : 
  Real.sqrt 18 / Real.sqrt 2 = 3 := by
  sorry

end sqrt_eighteen_div_sqrt_two_equals_three_l1327_132729


namespace corner_sum_is_sixteen_l1327_132766

/-- Represents a 3x3 grid with integer entries -/
def Grid := Matrix (Fin 3) (Fin 3) ℤ

/-- The sum of elements in a given row -/
def row_sum (g : Grid) (i : Fin 3) : ℤ :=
  g i 0 + g i 1 + g i 2

/-- The sum of elements in a given column -/
def col_sum (g : Grid) (j : Fin 3) : ℤ :=
  g 0 j + g 1 j + g 2 j

/-- The sum of elements in the main diagonal -/
def main_diag_sum (g : Grid) : ℤ :=
  g 0 0 + g 1 1 + g 2 2

/-- The sum of elements in the anti-diagonal -/
def anti_diag_sum (g : Grid) : ℤ :=
  g 0 2 + g 1 1 + g 2 0

/-- A grid is magic if all rows, columns, and diagonals sum to 12 -/
def is_magic (g : Grid) : Prop :=
  (∀ i : Fin 3, row_sum g i = 12) ∧
  (∀ j : Fin 3, col_sum g j = 12) ∧
  main_diag_sum g = 12 ∧
  anti_diag_sum g = 12

theorem corner_sum_is_sixteen (g : Grid) 
  (h_magic : is_magic g)
  (h_corners : g 0 0 = 4 ∧ g 0 2 = 3 ∧ g 2 0 = 5 ∧ g 2 2 = 4) :
  g 0 0 + g 0 2 + g 2 0 + g 2 2 = 16 := by
  sorry

end corner_sum_is_sixteen_l1327_132766


namespace lowercase_count_l1327_132773

/-- Represents the structure of Pat's password -/
structure Password where
  total_length : ℕ
  symbols : ℕ
  lowercase : ℕ
  uppercase_and_numbers : ℕ

/-- Defines the conditions for Pat's password -/
def valid_password (p : Password) : Prop :=
  p.total_length = 14 ∧
  p.symbols = 2 ∧
  p.uppercase_and_numbers = p.lowercase / 2 ∧
  p.total_length = p.lowercase + p.uppercase_and_numbers + p.symbols

/-- Theorem stating that a valid password has 8 lowercase letters -/
theorem lowercase_count (p : Password) (h : valid_password p) : p.lowercase = 8 := by
  sorry

#check lowercase_count

end lowercase_count_l1327_132773


namespace exists_unobserved_planet_l1327_132796

/-- Represents a planet in the system -/
structure Planet where
  id : Nat

/-- Represents the system of planets -/
structure PlanetSystem where
  planets : Finset Planet
  distance : Planet → Planet → ℝ
  nearest_neighbor : Planet → Planet
  num_planets_odd : Odd (Finset.card planets)
  different_distances : ∀ p q r s : Planet, p ≠ q → r ≠ s → (p, q) ≠ (r, s) → distance p q ≠ distance r s
  nearest_is_nearest : ∀ p q : Planet, p ≠ q → distance p (nearest_neighbor p) ≤ distance p q

/-- The main theorem: In a system with an odd number of planets, where each planet has an astronomer
    observing the nearest planet and all inter-planet distances are unique, there exists at least
    one planet that is not being observed. -/
theorem exists_unobserved_planet (sys : PlanetSystem) :
  ∃ p : Planet, p ∈ sys.planets ∧ ∀ q : Planet, q ∈ sys.planets → sys.nearest_neighbor q ≠ p :=
sorry

end exists_unobserved_planet_l1327_132796


namespace continuous_finite_preimage_implies_smp_l1327_132703

open Set

/-- Definition of "smp" property for a function -/
def IsSmp (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ (n : ℕ) (c : Fin (n + 1) → ℝ),
    c 0 = a ∧ c (Fin.last n) = b ∧
    (∀ i : Fin n, c i < c (i + 1)) ∧
    (∀ i : Fin n, ∀ x ∈ Ioo (c i) (c (i + 1)),
      (f (c i) < f x ∧ f x < f (c (i + 1))) ∨
      (f (c i) > f x ∧ f x > f (c (i + 1))))

/-- Main theorem statement -/
theorem continuous_finite_preimage_implies_smp
  (f : ℝ → ℝ) (a b : ℝ) (h_cont : ContinuousOn f (Icc a b))
  (h_finite : ∀ v : ℝ, Set.Finite {x ∈ Icc a b | f x = v}) :
  IsSmp f a b :=
sorry

end continuous_finite_preimage_implies_smp_l1327_132703


namespace square_root_of_2m_minus_n_is_2_l1327_132785

theorem square_root_of_2m_minus_n_is_2 
  (m n : ℝ) 
  (eq1 : m * 2 + n * 1 = 8) 
  (eq2 : n * 2 - m * 1 = 1) : 
  Real.sqrt (2 * m - n) = 2 := by
  sorry

end square_root_of_2m_minus_n_is_2_l1327_132785


namespace cos_pi_half_plus_alpha_l1327_132770

theorem cos_pi_half_plus_alpha (α : ℝ) (h : Real.sin (-α) = Real.sqrt 5 / 3) :
  Real.cos (π / 2 + α) = Real.sqrt 5 / 3 := by
  sorry

end cos_pi_half_plus_alpha_l1327_132770


namespace gp_solution_and_sum_l1327_132716

/-- Given a real number x, returns true if 10+x, 30+x, and 90+x form a geometric progression -/
def isGeometricProgression (x : ℝ) : Prop :=
  (30 + x)^2 = (10 + x) * (90 + x)

/-- Computes the sum of the terms in the progression for a given x -/
def sumOfProgression (x : ℝ) : ℝ :=
  (10 + x) + (30 + x) + (90 + x)

theorem gp_solution_and_sum :
  ∃! x : ℝ, isGeometricProgression x ∧ sumOfProgression x = 130 :=
sorry

end gp_solution_and_sum_l1327_132716


namespace layla_apples_l1327_132737

theorem layla_apples (maggie : ℕ) (kelsey : ℕ) (layla : ℕ) :
  maggie = 40 →
  kelsey = 28 →
  (maggie + kelsey + layla) / 3 = 30 →
  layla = 22 :=
by sorry

end layla_apples_l1327_132737


namespace geometric_sequence_common_ratio_l1327_132751

theorem geometric_sequence_common_ratio (a : ℕ → ℚ) :
  (∀ n, a (n + 1) = a n * (a 2 / a 1)) →  -- Definition of geometric sequence
  a 1 = 64 →                             -- First term condition
  a 2 = 8 →                              -- Second term condition
  a 2 / a 1 = 1 / 8 :=                   -- Conclusion: common ratio q = 1/8
by
  sorry

end geometric_sequence_common_ratio_l1327_132751


namespace arithmetic_sequence_sum_condition_l1327_132733

def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem arithmetic_sequence_sum_condition
  (a₁ d : ℝ) (n : ℕ)
  (h1 : a₁ = 2)
  (h2 : d = 3)
  (h3 : arithmetic_sequence a₁ d n + arithmetic_sequence a₁ d (n + 2) = 28) :
  n = 4 := by
sorry

end arithmetic_sequence_sum_condition_l1327_132733


namespace intersection_points_line_slope_l1327_132793

theorem intersection_points_line_slope :
  ∀ (s : ℝ) (x y : ℝ),
    (2 * x - 3 * y = 4 * s + 6) →
    (2 * x + y = 3 * s + 1) →
    y = -2/13 * x - 14/13 := by
  sorry

end intersection_points_line_slope_l1327_132793


namespace man_walking_speed_percentage_l1327_132791

/-- Proves that a man is walking at 70% of his usual speed given the conditions -/
theorem man_walking_speed_percentage (usual_time distance : ℝ) 
  (h1 : usual_time = 56)
  (h2 : distance > 0)
  (h3 : distance = usual_time * (distance / usual_time)) -- Speed * Time = Distance
  (h4 : distance = 80 * (distance / (56 + 24))) -- New time is 80 minutes
  : (distance / (56 + 24)) / (distance / usual_time) = 0.7 := by
  sorry

end man_walking_speed_percentage_l1327_132791


namespace factorial_difference_l1327_132712

theorem factorial_difference : Nat.factorial 12 - Nat.factorial 11 = 439084800 := by
  sorry

end factorial_difference_l1327_132712


namespace cow_calf_ratio_l1327_132772

def total_cost : ℕ := 990
def cow_cost : ℕ := 880
def calf_cost : ℕ := 110

theorem cow_calf_ratio : 
  ∃ (m : ℕ), m > 0 ∧ cow_cost = m * calf_cost ∧ cow_cost / calf_cost = 8 := by
  sorry

end cow_calf_ratio_l1327_132772


namespace decorative_window_area_ratio_l1327_132742

/-- Represents the dimensions of a decorative window --/
structure WindowDimensions where
  ab : ℝ  -- width of the rectangle and diameter of semicircles
  ad : ℝ  -- length of the rectangle
  h_ab_positive : ab > 0
  h_ad_positive : ad > 0
  h_ratio : ad / ab = 4 / 3

/-- Theorem about the ratio of areas in a decorative window --/
theorem decorative_window_area_ratio 
  (w : WindowDimensions) 
  (h_ab : w.ab = 36) : 
  (w.ad * w.ab) / (π * (w.ab / 2)^2) = 16 / (3 * π) := by
  sorry

end decorative_window_area_ratio_l1327_132742


namespace angle_A_value_max_area_l1327_132777

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def givenCondition (t : Triangle) : Prop :=
  (-t.b + Real.sqrt 2 * t.c) / Real.cos t.B = t.a / Real.cos t.A

-- Theorem 1: Prove that A = π/4
theorem angle_A_value (t : Triangle) (h : givenCondition t) : t.A = π / 4 := by
  sorry

-- Theorem 2: Prove the maximum area when a = 2
theorem max_area (t : Triangle) (h : givenCondition t) (ha : t.a = 2) :
  ∃ (S : ℝ), S = Real.sqrt 2 + 1 ∧ ∀ (S' : ℝ), S' ≤ S := by
  sorry

end angle_A_value_max_area_l1327_132777


namespace arithmetic_sequence_common_difference_l1327_132765

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_first_fifth : a 1 + a 5 = 10
  fourth_term : a 4 = 7

/-- The common difference of the arithmetic sequence is 2 -/
theorem arithmetic_sequence_common_difference (seq : ArithmeticSequence) :
  ∃ d, (∀ n, seq.a (n + 1) - seq.a n = d) ∧ d = 2 := by
  sorry

end arithmetic_sequence_common_difference_l1327_132765


namespace walk_distance_theorem_l1327_132727

/-- Calculates the total distance walked when a person walks at a given speed for a certain time in one direction and then returns along the same path. -/
def totalDistanceWalked (speed : ℝ) (time : ℝ) : ℝ :=
  2 * speed * time

/-- Theorem stating that walking at 2 miles per hour for 3 hours in one direction and returning results in a total distance of 12 miles. -/
theorem walk_distance_theorem :
  totalDistanceWalked 2 3 = 12 := by
  sorry

end walk_distance_theorem_l1327_132727


namespace concentric_circles_radius_l1327_132790

theorem concentric_circles_radius (r : ℝ) (R : ℝ) : 
  r > 0 → 
  (π * R^2) / (π * r^2) = 5 / 2 → 
  R = r * Real.sqrt 2.5 := by
sorry

end concentric_circles_radius_l1327_132790


namespace system_solution_unique_l1327_132720

theorem system_solution_unique :
  ∃! (x y z : ℚ), 
    3 * x - 4 * y = 12 ∧
    -5 * x + 6 * y - z = 9 ∧
    x + 2 * y + 3 * z = 0 ∧
    x = -262/75 ∧
    y = -2075/200 ∧
    z = -105/100 := by
  sorry

end system_solution_unique_l1327_132720


namespace expression_evaluation_l1327_132782

theorem expression_evaluation (c d : ℕ) (hc : c = 4) (hd : d = 2) :
  (c^c - c*(c-d)^c)^c = 136048896 := by
  sorry

end expression_evaluation_l1327_132782


namespace unique_solution_l1327_132792

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: 36018 is the only positive integer m that satisfies 2001 * S(m) = m -/
theorem unique_solution :
  ∀ m : ℕ, m > 0 → (2001 * sumOfDigits m = m) ↔ m = 36018 := by sorry

end unique_solution_l1327_132792


namespace gcd_factorial_8_9_l1327_132710

theorem gcd_factorial_8_9 : Nat.gcd (Nat.factorial 8) (Nat.factorial 9) = Nat.factorial 8 := by
  sorry

end gcd_factorial_8_9_l1327_132710


namespace employee_count_proof_l1327_132767

theorem employee_count_proof : ∃! b : ℕ, 
  80 < b ∧ b < 150 ∧
  b % 4 = 3 ∧
  b % 5 = 3 ∧
  b % 7 = 4 ∧
  b = 143 := by
sorry

end employee_count_proof_l1327_132767


namespace equal_savings_l1327_132776

-- Define the total combined salary
def total_salary : ℝ := 6000

-- Define A's salary
def salary_A : ℝ := 4500

-- Define B's salary
def salary_B : ℝ := total_salary - salary_A

-- Define A's spending rate
def spending_rate_A : ℝ := 0.95

-- Define B's spending rate
def spending_rate_B : ℝ := 0.85

-- Define A's savings
def savings_A : ℝ := salary_A * (1 - spending_rate_A)

-- Define B's savings
def savings_B : ℝ := salary_B * (1 - spending_rate_B)

-- Theorem: A and B have the same savings
theorem equal_savings : savings_A = savings_B := by
  sorry

end equal_savings_l1327_132776


namespace circle_and_line_theorem_l1327_132752

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 1

-- Define the line m
def line_m (x y : ℝ) : Prop := 3*x - 2*y = 0

-- Define the line l
def line_l (k x y : ℝ) : Prop := y = k*x + 2

-- Define points A and B
def point_A : ℝ × ℝ := (1, 3)
def point_B : ℝ × ℝ := (2, 2)

-- Define the dot product of OM and ON
def dot_product (k x₁ x₂ : ℝ) : ℝ := x₁*x₂ + (k*x₁ + 2)*(k*x₂ + 2)

theorem circle_and_line_theorem :
  -- 1. Circle C passes through A and B and is bisected by line m
  (circle_C point_A.1 point_A.2 ∧ circle_C point_B.1 point_B.2) ∧
  (∀ x y, circle_C x y → line_m x y → circle_C (2*2 - x) (2*3 - y)) →
  -- 2. No k exists such that line l intersects C at M and N where OM•ON = 6
  ¬∃ k : ℝ, ∃ x₁ x₂ : ℝ,
    x₁ ≠ x₂ ∧
    circle_C x₁ (k*x₁ + 2) ∧
    circle_C x₂ (k*x₂ + 2) ∧
    dot_product k x₁ x₂ = 6 :=
sorry

end circle_and_line_theorem_l1327_132752


namespace sphere_tangent_angle_theorem_l1327_132722

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a sphere -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Represents a line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Calculate the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point3D) : ℝ := sorry

/-- Check if three lines are parallel -/
def areParallel (l1 l2 l3 : Line3D) : Prop := sorry

/-- Check if a line is tangent to a sphere -/
def isTangent (l : Line3D) (s : Sphere) : Prop := sorry

/-- Calculate the angle between three points -/
def angle (p1 p2 p3 : Point3D) : ℝ := sorry

theorem sphere_tangent_angle_theorem (O K L M : Point3D) (s : Sphere) (l1 l2 l3 : Line3D) :
  s.center = O →
  s.radius = 5 →
  areParallel l1 l2 l3 →
  isTangent l1 s →
  isTangent l2 s →
  isTangent l3 s →
  triangleArea O K L = 12 →
  triangleArea K L M > 30 →
  angle K M L = Real.arccos (3/5) := by
  sorry

end sphere_tangent_angle_theorem_l1327_132722


namespace union_equals_set_iff_subset_l1327_132783

theorem union_equals_set_iff_subset (A B : Set α) : A ∪ B = B ↔ A ⊆ B := by
  sorry

end union_equals_set_iff_subset_l1327_132783


namespace geometric_sequence_ninth_term_l1327_132756

/-- A geometric sequence with first term 1 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- The theorem to be proved -/
theorem geometric_sequence_ninth_term
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_prod : a 7 * a 11 = 100) :
  a 9 = 10 :=
sorry

end geometric_sequence_ninth_term_l1327_132756


namespace division_problem_l1327_132754

theorem division_problem (divisor quotient remainder number : ℕ) : 
  divisor = 12 → 
  quotient = 9 → 
  remainder = 1 → 
  number = divisor * quotient + remainder → 
  number = 109 := by
sorry

end division_problem_l1327_132754


namespace sum_of_i_powers_l1327_132760

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- Theorem: Sum of specific powers of i equals 2 -/
theorem sum_of_i_powers : i^24 + i^29 + i^34 + i^39 + i^44 + i^49 = 2 := by
  sorry

end sum_of_i_powers_l1327_132760


namespace towers_count_l1327_132726

/-- Represents the number of cubes of each color -/
structure CubeSet where
  yellow : Nat
  purple : Nat
  orange : Nat

/-- Calculates the number of different towers that can be built -/
def countTowers (cubes : CubeSet) (towerHeight : Nat) : Nat :=
  sorry

/-- The theorem to be proved -/
theorem towers_count (cubes : CubeSet) (h : cubes = { yellow := 3, purple := 3, orange := 2 }) :
  countTowers cubes 6 = 350 := by
  sorry

end towers_count_l1327_132726


namespace inequality_addition_l1327_132746

theorem inequality_addition (x y : ℝ) (h : x < y) : x + 6 < y + 6 := by
  sorry

end inequality_addition_l1327_132746


namespace power_of_three_difference_l1327_132743

theorem power_of_three_difference : 3^(1+2+3) - (3^1 + 3^2 + 3^3) = 690 := by
  sorry

end power_of_three_difference_l1327_132743


namespace jeanette_practice_weeks_l1327_132706

/-- The number of objects Jeanette can juggle after w weeks -/
def objects_juggled (w : ℕ) : ℕ := 3 + 2 * w

/-- The theorem stating that Jeanette practiced for 5 weeks -/
theorem jeanette_practice_weeks : 
  ∃ w : ℕ, objects_juggled w = 13 ∧ w = 5 := by
  sorry

end jeanette_practice_weeks_l1327_132706


namespace certain_number_is_three_l1327_132755

theorem certain_number_is_three (n : ℝ) (x : ℤ) (h1 : n^(2*x) = 3^(12-x)) (h2 : x = 4) : n = 3 := by
  sorry

end certain_number_is_three_l1327_132755


namespace nested_fraction_evaluation_l1327_132757

theorem nested_fraction_evaluation : 
  1 / (2 + 1 / (3 + 1 / 4)) = 13 / 30 := by sorry

end nested_fraction_evaluation_l1327_132757


namespace largest_integer_l1327_132750

theorem largest_integer (a b c : ℤ) : 
  (2 * a + 3 * b + 4 * c = 225) →
  (a + b + c = 60) →
  (a = 15 ∨ b = 15 ∨ c = 15) →
  (max a (max b c) = 25) :=
by sorry

end largest_integer_l1327_132750


namespace intersection_condition_l1327_132775

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
def B (p : ℝ) : Set ℝ := {x : ℝ | p + 1 ≤ x ∧ x ≤ 2*p - 1}

theorem intersection_condition (p : ℝ) : A ∩ B p = B p ↔ p ≤ 3 := by
  sorry

end intersection_condition_l1327_132775


namespace rosa_flower_count_l1327_132702

/-- Given Rosa's initial flower count and the number of flowers Andre gave her,
    prove that the total number of flowers Rosa has now is equal to the sum of these two quantities. -/
theorem rosa_flower_count (initial_flowers andre_flowers : ℕ) :
  initial_flowers = 67 →
  andre_flowers = 23 →
  initial_flowers + andre_flowers = 90 :=
by sorry

end rosa_flower_count_l1327_132702


namespace sqrt_meaningful_l1327_132771

theorem sqrt_meaningful (x : ℝ) : (∃ y : ℝ, y ^ 2 = 1 - x) ↔ x ≤ 1 := by sorry

end sqrt_meaningful_l1327_132771


namespace no_prime_pair_with_odd_difference_quotient_l1327_132797

theorem no_prime_pair_with_odd_difference_quotient :
  ¬ ∃ (p q : ℕ), Prime p ∧ Prime q ∧ p > q ∧ (∃ (k : ℕ), 2 * k + 1 = (p^2 - q^2) / 4) :=
by sorry

end no_prime_pair_with_odd_difference_quotient_l1327_132797


namespace one_zero_in_interval_l1327_132707

def f (x : ℝ) := 2*x + x^3 - 2

theorem one_zero_in_interval : ∃! x : ℝ, 0 < x ∧ x < 1 ∧ f x = 0 := by
  sorry

end one_zero_in_interval_l1327_132707


namespace john_money_needed_l1327_132749

/-- The amount of money John currently has, in dollars -/
def current_amount : ℚ := 0.75

/-- The additional amount of money John needs, in dollars -/
def additional_amount : ℚ := 1.75

/-- The total amount of money John needs, in dollars -/
def total_amount : ℚ := current_amount + additional_amount

theorem john_money_needed : total_amount = 2.50 := by sorry

end john_money_needed_l1327_132749


namespace bus_distance_traveled_l1327_132780

theorem bus_distance_traveled (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 67 → time = 3 → distance = speed * time → distance = 201 := by sorry

end bus_distance_traveled_l1327_132780


namespace no_solution_exists_l1327_132736

theorem no_solution_exists (w x y z : ℂ) (n : ℕ+) : 
  w ≠ 0 → x ≠ 0 → y ≠ 0 → z ≠ 0 →
  1 / w + 1 / x + 1 / y + 1 / z = 3 →
  w * x + w * y + w * z + x * y + x * z + y * z = 14 →
  (w + x)^3 + (w + y)^3 + (w + z)^3 + (x + y)^3 + (x + z)^3 + (y + z)^3 = 2160 →
  ∃ (r : ℝ), w + x + y + z + Complex.I * Real.sqrt n = r →
  False :=
by sorry

end no_solution_exists_l1327_132736


namespace arithmetic_sequence_common_difference_l1327_132735

def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a4 : a 4 = 9)
  (h_sum : a 3 + a 7 = 20) :
  ∃ d : ℝ, d = 1 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by sorry

end arithmetic_sequence_common_difference_l1327_132735


namespace four_men_absent_l1327_132762

/-- Represents the work scenario with contractors -/
structure WorkScenario where
  totalMen : ℕ
  plannedDays : ℕ
  actualDays : ℕ
  absentMen : ℕ

/-- The work scenario satisfies the given conditions -/
def validScenario (w : WorkScenario) : Prop :=
  w.totalMen = 10 ∧ w.plannedDays = 6 ∧ w.actualDays = 10 ∧
  w.totalMen * w.plannedDays = (w.totalMen - w.absentMen) * w.actualDays

/-- The theorem stating that 4 men were absent -/
theorem four_men_absent :
  ∃ (w : WorkScenario), validScenario w ∧ w.absentMen = 4 := by
  sorry

end four_men_absent_l1327_132762


namespace equality_for_specific_values_l1327_132739

theorem equality_for_specific_values : 
  ∃ (a b c : ℝ), a + b^2 * c = (a^2 + b) * (a + c) :=
sorry

end equality_for_specific_values_l1327_132739


namespace line_symmetry_theorem_l1327_132701

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Check if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

/-- Two lines are symmetric about the y-axis -/
def symmetric_about_y_axis (l₁ l₂ : Line) : Prop :=
  l₁.slope = -l₂.slope ∧ l₁.intercept = l₂.intercept

theorem line_symmetry_theorem (b : ℝ) :
  let l₁ : Line := { slope := -2, intercept := b }
  let l₂ : Line := { slope := 2, intercept := 4 }
  symmetric_about_y_axis l₁ l₂ ∧ l₂.contains 1 6 → b = 4 := by
  sorry

end line_symmetry_theorem_l1327_132701


namespace range_of_a_l1327_132794

theorem range_of_a (a : ℝ) : a > 5 → ∃ x : ℝ, x > -1 ∧ (x^2 + 3*x + 6) / (x + 1) < a := by
  sorry

end range_of_a_l1327_132794


namespace arithmetic_sequence_a12_l1327_132711

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a12 (a : ℕ → ℤ) 
  (h_arith : arithmetic_sequence a) 
  (h_a4 : a 4 = 4) 
  (h_a8 : a 8 = -4) : 
  a 12 = -12 := by
  sorry

end arithmetic_sequence_a12_l1327_132711


namespace debt_payment_proof_l1327_132779

theorem debt_payment_proof (x : ℝ) : 
  (20 * x + 20 * (x + 65)) / 40 = 442.5 → x = 410 := by
  sorry

end debt_payment_proof_l1327_132779


namespace M_intersect_N_l1327_132724

def M : Set ℕ := {0, 2, 4}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

theorem M_intersect_N : M ∩ N = {0, 4} := by
  sorry

end M_intersect_N_l1327_132724


namespace no_even_primes_greater_than_two_l1327_132768

theorem no_even_primes_greater_than_two :
  ∀ n : ℕ, n > 2 → Prime n → ¬Even n :=
by
  sorry

end no_even_primes_greater_than_two_l1327_132768


namespace direct_proportion_condition_l1327_132747

/-- A function f(x) is a direct proportion function if there exists a non-zero constant k such that f(x) = k * x for all x -/
def IsDirectProportionFunction (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x

/-- The function defined by m -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x + (m^2 - 4)

theorem direct_proportion_condition (m : ℝ) :
  IsDirectProportionFunction (f m) ↔ m = -2 := by sorry

end direct_proportion_condition_l1327_132747


namespace max_x_minus_y_l1327_132761

theorem max_x_minus_y (x y z : ℝ) 
  (sum_eq : x + y + z = 2) 
  (prod_eq : x * y + y * z + z * x = 1) : 
  x - y ≤ 2 * Real.sqrt 3 / 3 := by
  sorry

end max_x_minus_y_l1327_132761


namespace geometry_propositions_l1327_132789

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)

-- State the theorem
theorem geometry_propositions 
  (a b : Line) (α β : Plane) :
  (∀ α β : Plane, ∀ a : Line, 
    parallel α β → subset a α → line_parallel a β) ∧
  (∀ a b : Line, ∀ α β : Plane,
    perpendicular a α → parallel α β → line_parallel b β → 
    line_perpendicular a b) := by
  sorry

end geometry_propositions_l1327_132789


namespace min_score_11_l1327_132744

def basketball_problem (scores : List ℕ) (score_11 : ℕ) : Prop :=
  let total_10 := scores.sum + 15 + 22 + 18
  let avg_10 := total_10 / 10
  let avg_7 := (total_10 - (15 + 22 + 18)) / 7
  let total_11 := total_10 + score_11
  let avg_11 := total_11 / 11
  (scores.length = 7) ∧
  (avg_10 > avg_7) ∧
  (avg_11 > 20) ∧
  (∀ s : ℕ, s < score_11 → (total_10 + s) / 11 ≤ 20)

theorem min_score_11 (scores : List ℕ) :
  basketball_problem scores 33 → 
  ∀ n : ℕ, n < 33 → ¬(basketball_problem scores n) :=
by sorry

end min_score_11_l1327_132744


namespace inverse_proportion_k_range_l1327_132798

/-- Prove that for an inverse proportion function y = (4-k)/x with points A(x₁, y₁) and B(x₂, y₂) 
    on its graph, where x₁ < 0 < x₂ and y₁ < y₂, the range of values for k is k < 4. -/
theorem inverse_proportion_k_range (k : ℝ) (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : x₁ < 0) (h2 : 0 < x₂) (h3 : y₁ < y₂)
  (h4 : y₁ = (4 - k) / x₁) (h5 : y₂ = (4 - k) / x₂) :
  k < 4 := by
  sorry

end inverse_proportion_k_range_l1327_132798


namespace income_tax_calculation_l1327_132774

def salary_jan_jun : ℕ := 23000
def salary_jul_dec : ℕ := 25000
def months_per_half_year : ℕ := 6
def prize_value : ℕ := 10000
def non_taxable_prize : ℕ := 4000
def salary_tax_rate : ℚ := 13 / 100
def prize_tax_rate : ℚ := 35 / 100

def total_income_tax : ℕ := 39540

theorem income_tax_calculation :
  let total_salary := salary_jan_jun * months_per_half_year + salary_jul_dec * months_per_half_year
  let salary_tax := (total_salary : ℚ) * salary_tax_rate
  let taxable_prize := prize_value - non_taxable_prize
  let prize_tax := (taxable_prize : ℚ) * prize_tax_rate
  let total_tax := salary_tax + prize_tax
  ⌊total_tax⌋ = total_income_tax := by sorry

end income_tax_calculation_l1327_132774


namespace complement_of_intersection_union_condition_l1327_132715

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2*x - 4 ≥ x - 2}
def C (a : ℝ) : Set ℝ := {x | 2*x + a > 0}

-- Theorem 1: Complement of intersection
theorem complement_of_intersection :
  (Aᶜ ∪ Bᶜ : Set ℝ) = {x | x < 2 ∨ x ≥ 3} := by sorry

-- Theorem 2: Condition for B ∪ C = C
theorem union_condition (a : ℝ) :
  B ∪ C a = C a → a ≥ -4 := by sorry

end complement_of_intersection_union_condition_l1327_132715


namespace distance_to_focus_l1327_132723

/-- Given a parabola y² = 8x and a point P on it, prove that the distance from P to the focus is 10 -/
theorem distance_to_focus (x₀ : ℝ) : 
  8^2 = 8 * x₀ →  -- P(x₀, 8) is on the parabola y² = 8x
  Real.sqrt ((x₀ - 2)^2 + 8^2) = 10 := by sorry

end distance_to_focus_l1327_132723


namespace best_fit_highest_r_squared_l1327_132741

/-- Represents a regression model with its R² value -/
structure RegressionModel where
  id : Nat
  r_squared : Real

/-- Given a list of regression models, the model with the highest R² value has the best fit -/
theorem best_fit_highest_r_squared (models : List RegressionModel) :
  models ≠ [] →
  ∃ best_model : RegressionModel,
    best_model ∈ models ∧
    (∀ model ∈ models, model.r_squared ≤ best_model.r_squared) ∧
    (∀ model ∈ models, model.r_squared = best_model.r_squared → model = best_model) :=
by sorry

end best_fit_highest_r_squared_l1327_132741


namespace largest_prime_factor_of_sum_of_divisors_360_l1327_132731

def sum_of_divisors (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_sum_of_divisors_360 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ sum_of_divisors 360 ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ sum_of_divisors 360 → q ≤ p :=
by sorry

end largest_prime_factor_of_sum_of_divisors_360_l1327_132731


namespace final_sum_after_transformation_l1327_132748

theorem final_sum_after_transformation (a b S : ℝ) (h : a + b = S) :
  3 * (a - 5) + 3 * (b - 5) = 3 * S - 30 := by
  sorry

end final_sum_after_transformation_l1327_132748


namespace joan_found_79_seashells_l1327_132730

/-- The number of seashells Mike gave to Joan -/
def mike_seashells : ℕ := 63

/-- The total number of seashells Joan has -/
def total_seashells : ℕ := 142

/-- The number of seashells Joan found initially -/
def joan_initial_seashells : ℕ := total_seashells - mike_seashells

theorem joan_found_79_seashells : joan_initial_seashells = 79 := by sorry

end joan_found_79_seashells_l1327_132730


namespace weeks_to_afford_bicycle_l1327_132725

def bicycle_cost : ℕ := 600
def birthday_money : ℕ := 165
def weekly_earnings : ℕ := 20

theorem weeks_to_afford_bicycle :
  let total_money : ℕ → ℕ := λ weeks => birthday_money + weekly_earnings * weeks
  ∀ weeks : ℕ, total_money weeks ≥ bicycle_cost → weeks ≥ 22 :=
by
  sorry

end weeks_to_afford_bicycle_l1327_132725


namespace alice_bushes_l1327_132717

/-- The number of bushes Alice needs to buy for her yard -/
def bushes_needed (sides : ℕ) (side_length : ℕ) (bush_length : ℕ) : ℕ :=
  (sides * side_length) / bush_length

/-- Theorem: Alice needs to buy 12 bushes -/
theorem alice_bushes :
  bushes_needed 3 16 4 = 12 := by
  sorry

end alice_bushes_l1327_132717


namespace seventeen_minus_fifteen_factorial_prime_divisors_l1327_132778

-- Define factorial function
def factorial (n : ℕ) : ℕ := Nat.factorial n

-- Define the number of prime divisors function
def num_prime_divisors (n : ℕ) : ℕ := (Nat.factorization n).support.card

-- Theorem statement
theorem seventeen_minus_fifteen_factorial_prime_divisors :
  num_prime_divisors (factorial 17 - factorial 15) = 7 := by
  sorry

end seventeen_minus_fifteen_factorial_prime_divisors_l1327_132778


namespace jason_lost_cards_l1327_132769

/-- The number of Pokemon cards Jason lost at a tournament -/
def cards_lost (initial_cards bought_cards final_cards : ℕ) : ℕ :=
  initial_cards + bought_cards - final_cards

/-- Theorem stating that Jason lost 188 Pokemon cards at the tournament -/
theorem jason_lost_cards : cards_lost 676 224 712 = 188 := by
  sorry

end jason_lost_cards_l1327_132769


namespace combined_weight_is_9500_l1327_132718

def regular_dinosaur_weight : ℕ := 800
def number_of_regular_dinosaurs : ℕ := 5
def barney_extra_weight : ℕ := 1500

def combined_weight : ℕ :=
  (regular_dinosaur_weight * number_of_regular_dinosaurs) + 
  (regular_dinosaur_weight * number_of_regular_dinosaurs + barney_extra_weight)

theorem combined_weight_is_9500 : combined_weight = 9500 := by
  sorry

end combined_weight_is_9500_l1327_132718


namespace product_of_repeating_decimal_and_22_l1327_132745

/-- The repeating decimal 0.454545... --/
def repeating_decimal : ℚ := 5 / 11

theorem product_of_repeating_decimal_and_22 :
  repeating_decimal * 22 = 10 := by sorry

end product_of_repeating_decimal_and_22_l1327_132745


namespace vector_calculation_l1327_132763

theorem vector_calculation : 
  2 • (((3 : ℝ), -2, 5) + ((-1 : ℝ), 6, -7)) = ((4 : ℝ), 8, -4) := by
sorry

end vector_calculation_l1327_132763


namespace line_through_point_equal_intercepts_l1327_132705

/-- A line passing through (2,3) with equal absolute intercepts -/
theorem line_through_point_equal_intercepts :
  ∃ (m c : ℝ), 
    (3 = 2 * m + c) ∧ 
    (|c| = |c / m|) ∧
    (∀ x y : ℝ, y = m * x + c ↔ y = x + 1) := by
  sorry

end line_through_point_equal_intercepts_l1327_132705
