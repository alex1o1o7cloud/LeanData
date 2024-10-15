import Mathlib

namespace NUMINAMATH_CALUDE_jason_lost_cards_l634_63432

/-- The number of Pokemon cards Jason lost at a tournament -/
def cards_lost (initial_cards bought_cards final_cards : ℕ) : ℕ :=
  initial_cards + bought_cards - final_cards

/-- Theorem stating that Jason lost 188 Pokemon cards at the tournament -/
theorem jason_lost_cards : cards_lost 676 224 712 = 188 := by
  sorry

end NUMINAMATH_CALUDE_jason_lost_cards_l634_63432


namespace NUMINAMATH_CALUDE_walk_distance_theorem_l634_63451

/-- Calculates the total distance walked when a person walks at a given speed for a certain time in one direction and then returns along the same path. -/
def totalDistanceWalked (speed : ℝ) (time : ℝ) : ℝ :=
  2 * speed * time

/-- Theorem stating that walking at 2 miles per hour for 3 hours in one direction and returning results in a total distance of 12 miles. -/
theorem walk_distance_theorem :
  totalDistanceWalked 2 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_walk_distance_theorem_l634_63451


namespace NUMINAMATH_CALUDE_system_solution_unique_l634_63496

theorem system_solution_unique :
  ∃! (x y z : ℚ), 
    3 * x - 4 * y = 12 ∧
    -5 * x + 6 * y - z = 9 ∧
    x + 2 * y + 3 * z = 0 ∧
    x = -262/75 ∧
    y = -2075/200 ∧
    z = -105/100 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l634_63496


namespace NUMINAMATH_CALUDE_corner_sum_is_sixteen_l634_63478

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

end NUMINAMATH_CALUDE_corner_sum_is_sixteen_l634_63478


namespace NUMINAMATH_CALUDE_concentric_circles_radius_l634_63416

theorem concentric_circles_radius (r : ℝ) (R : ℝ) : 
  r > 0 → 
  (π * R^2) / (π * r^2) = 5 / 2 → 
  R = r * Real.sqrt 2.5 := by
sorry

end NUMINAMATH_CALUDE_concentric_circles_radius_l634_63416


namespace NUMINAMATH_CALUDE_employee_count_proof_l634_63441

theorem employee_count_proof : ∃! b : ℕ, 
  80 < b ∧ b < 150 ∧
  b % 4 = 3 ∧
  b % 5 = 3 ∧
  b % 7 = 4 ∧
  b = 143 := by
sorry

end NUMINAMATH_CALUDE_employee_count_proof_l634_63441


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l634_63431

theorem ceiling_floor_difference (x : ℝ) : 
  (⌈x⌉ : ℝ) + (⌊x⌋ : ℝ) = 2 * x → (⌈x⌉ : ℝ) - (⌊x⌋ : ℝ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l634_63431


namespace NUMINAMATH_CALUDE_bus_distance_traveled_l634_63483

theorem bus_distance_traveled (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 67 → time = 3 → distance = speed * time → distance = 201 := by sorry

end NUMINAMATH_CALUDE_bus_distance_traveled_l634_63483


namespace NUMINAMATH_CALUDE_race_time_differences_l634_63446

/-- Race competition with three competitors --/
structure RaceCompetition where
  distance : ℝ
  time_A : ℝ
  time_B : ℝ
  time_C : ℝ

/-- Calculate time difference between two competitors --/
def timeDifference (t1 t2 : ℝ) : ℝ := t2 - t1

/-- Theorem stating the time differences between competitors --/
theorem race_time_differences (race : RaceCompetition) 
  (h_distance : race.distance = 250)
  (h_time_A : race.time_A = 40)
  (h_time_B : race.time_B = 50)
  (h_time_C : race.time_C = 55) : 
  (timeDifference race.time_A race.time_B = 10) ∧ 
  (timeDifference race.time_A race.time_C = 15) ∧ 
  (timeDifference race.time_B race.time_C = 5) := by
  sorry

end NUMINAMATH_CALUDE_race_time_differences_l634_63446


namespace NUMINAMATH_CALUDE_min_cubes_surface_area_52_l634_63479

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

end NUMINAMATH_CALUDE_min_cubes_surface_area_52_l634_63479


namespace NUMINAMATH_CALUDE_union_equals_set_iff_subset_l634_63494

theorem union_equals_set_iff_subset (A B : Set α) : A ∪ B = B ↔ A ⊆ B := by
  sorry

end NUMINAMATH_CALUDE_union_equals_set_iff_subset_l634_63494


namespace NUMINAMATH_CALUDE_max_x_minus_y_l634_63435

theorem max_x_minus_y (x y z : ℝ) 
  (sum_eq : x + y + z = 2) 
  (prod_eq : x * y + y * z + z * x = 1) : 
  x - y ≤ 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l634_63435


namespace NUMINAMATH_CALUDE_root_implies_q_value_l634_63425

theorem root_implies_q_value (p q : ℝ) :
  (Complex.I : ℂ) ^ 2 = -1 →
  (3 : ℂ) * (5 + Complex.I) ^ 2 + p * (5 + Complex.I) + q = 0 →
  q = 78 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_q_value_l634_63425


namespace NUMINAMATH_CALUDE_popsicle_stick_ratio_l634_63499

-- Define the number of popsicle sticks for each person
def steve_sticks : ℕ := 12
def total_sticks : ℕ := 108

-- Define the relationship between Sam and Sid's sticks
def sam_sticks (sid_sticks : ℕ) : ℕ := 3 * sid_sticks

-- Theorem to prove
theorem popsicle_stick_ratio :
  ∃ (sid_sticks : ℕ),
    sid_sticks > 0 ∧
    sam_sticks sid_sticks + sid_sticks + steve_sticks = total_sticks ∧
    sid_sticks = 2 * steve_sticks :=
by sorry

end NUMINAMATH_CALUDE_popsicle_stick_ratio_l634_63499


namespace NUMINAMATH_CALUDE_tims_doctor_visit_cost_l634_63438

theorem tims_doctor_visit_cost (tim_total_payment : ℝ) (cat_visit_cost : ℝ) (cat_insurance_coverage : ℝ) (tim_insurance_coverage_percent : ℝ) : 
  tim_total_payment = 135 →
  cat_visit_cost = 120 →
  cat_insurance_coverage = 60 →
  tim_insurance_coverage_percent = 75 →
  ∃ (doctor_visit_cost : ℝ),
    doctor_visit_cost = 300 ∧
    tim_total_payment = (1 - tim_insurance_coverage_percent / 100) * doctor_visit_cost + (cat_visit_cost - cat_insurance_coverage) :=
by sorry

end NUMINAMATH_CALUDE_tims_doctor_visit_cost_l634_63438


namespace NUMINAMATH_CALUDE_current_age_of_D_l634_63469

theorem current_age_of_D (a b c d : ℕ) : 
  a + b + c + d = 108 →
  a - b = 12 →
  c - (a - 34) = 3 * (d - (a - 34)) →
  d = 13 := by
sorry

end NUMINAMATH_CALUDE_current_age_of_D_l634_63469


namespace NUMINAMATH_CALUDE_carltons_shirts_l634_63445

theorem carltons_shirts (shirts : ℕ) (vests : ℕ) (outfits : ℕ) : 
  vests = 2 * shirts → 
  outfits = vests * shirts → 
  outfits = 18 → 
  shirts = 3 := by
sorry

end NUMINAMATH_CALUDE_carltons_shirts_l634_63445


namespace NUMINAMATH_CALUDE_solution_check_l634_63422

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

end NUMINAMATH_CALUDE_solution_check_l634_63422


namespace NUMINAMATH_CALUDE_income_tax_calculation_l634_63410

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

end NUMINAMATH_CALUDE_income_tax_calculation_l634_63410


namespace NUMINAMATH_CALUDE_no_solution_exists_l634_63474

theorem no_solution_exists (w x y z : ℂ) (n : ℕ+) : 
  w ≠ 0 → x ≠ 0 → y ≠ 0 → z ≠ 0 →
  1 / w + 1 / x + 1 / y + 1 / z = 3 →
  w * x + w * y + w * z + x * y + x * z + y * z = 14 →
  (w + x)^3 + (w + y)^3 + (w + z)^3 + (x + y)^3 + (x + z)^3 + (y + z)^3 = 2160 →
  ∃ (r : ℝ), w + x + y + z + Complex.I * Real.sqrt n = r →
  False :=
by sorry

end NUMINAMATH_CALUDE_no_solution_exists_l634_63474


namespace NUMINAMATH_CALUDE_final_sum_after_transformation_l634_63457

theorem final_sum_after_transformation (a b S : ℝ) (h : a + b = S) :
  3 * (a - 5) + 3 * (b - 5) = 3 * S - 30 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_transformation_l634_63457


namespace NUMINAMATH_CALUDE_angle_A_value_max_area_l634_63473

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

end NUMINAMATH_CALUDE_angle_A_value_max_area_l634_63473


namespace NUMINAMATH_CALUDE_shirts_per_minute_l634_63450

/-- Given an industrial machine that makes 12 shirts in 6 minutes,
    prove that it makes 2 shirts per minute. -/
theorem shirts_per_minute :
  let total_shirts : ℕ := 12
  let total_minutes : ℕ := 6
  let shirts_per_minute : ℚ := total_shirts / total_minutes
  shirts_per_minute = 2 := by
  sorry

end NUMINAMATH_CALUDE_shirts_per_minute_l634_63450


namespace NUMINAMATH_CALUDE_ratio_value_l634_63488

theorem ratio_value (a b c d : ℚ) 
  (h1 : a = 4 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 5 * d) : 
  a * c / (b * d) = 20 := by
sorry

end NUMINAMATH_CALUDE_ratio_value_l634_63488


namespace NUMINAMATH_CALUDE_call_processing_ratio_l634_63424

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

end NUMINAMATH_CALUDE_call_processing_ratio_l634_63424


namespace NUMINAMATH_CALUDE_power_equation_solution_l634_63412

theorem power_equation_solution : ∃ K : ℕ, (81 ^ 2) * (27 ^ 3) = 3 ^ K ∧ K = 17 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l634_63412


namespace NUMINAMATH_CALUDE_square_root_of_2m_minus_n_is_2_l634_63463

theorem square_root_of_2m_minus_n_is_2 
  (m n : ℝ) 
  (eq1 : m * 2 + n * 1 = 8) 
  (eq2 : n * 2 - m * 1 = 1) : 
  Real.sqrt (2 * m - n) = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_2m_minus_n_is_2_l634_63463


namespace NUMINAMATH_CALUDE_circle_and_line_theorem_l634_63421

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

end NUMINAMATH_CALUDE_circle_and_line_theorem_l634_63421


namespace NUMINAMATH_CALUDE_sqrt_problem_1_sqrt_problem_2_l634_63460

-- Problem 1
theorem sqrt_problem_1 : Real.sqrt 6 * Real.sqrt 3 - 6 * Real.sqrt (1/2) = 0 := by sorry

-- Problem 2
theorem sqrt_problem_2 : (Real.sqrt 20 + Real.sqrt 5) / Real.sqrt 5 = 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_problem_1_sqrt_problem_2_l634_63460


namespace NUMINAMATH_CALUDE_cos_pi_half_plus_alpha_l634_63427

theorem cos_pi_half_plus_alpha (α : ℝ) (h : Real.sin (-α) = Real.sqrt 5 / 3) :
  Real.cos (π / 2 + α) = Real.sqrt 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_half_plus_alpha_l634_63427


namespace NUMINAMATH_CALUDE_debt_payment_proof_l634_63482

theorem debt_payment_proof (x : ℝ) : 
  (20 * x + 20 * (x + 65)) / 40 = 442.5 → x = 410 := by
  sorry

end NUMINAMATH_CALUDE_debt_payment_proof_l634_63482


namespace NUMINAMATH_CALUDE_problem_solution_l634_63403

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1|

-- Define the set M
def M : Set ℝ := {x | x < -1 ∨ x > 1}

-- Theorem statement
theorem problem_solution :
  (∀ x : ℝ, f x + 1 < |2 * x + 1| ↔ x ∈ M) ∧
  (∀ a b : ℝ, a ∈ M → b ∈ M → |a * b + 1| > |a + b|) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l634_63403


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l634_63420

theorem geometric_sequence_common_ratio (a : ℕ → ℚ) :
  (∀ n, a (n + 1) = a n * (a 2 / a 1)) →  -- Definition of geometric sequence
  a 1 = 64 →                             -- First term condition
  a 2 = 8 →                              -- Second term condition
  a 2 / a 1 = 1 / 8 :=                   -- Conclusion: common ratio q = 1/8
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l634_63420


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l634_63477

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

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l634_63477


namespace NUMINAMATH_CALUDE_jeanette_practice_weeks_l634_63470

/-- The number of objects Jeanette can juggle after w weeks -/
def objects_juggled (w : ℕ) : ℕ := 3 + 2 * w

/-- The theorem stating that Jeanette practiced for 5 weeks -/
theorem jeanette_practice_weeks : 
  ∃ w : ℕ, objects_juggled w = 13 ∧ w = 5 := by
  sorry

end NUMINAMATH_CALUDE_jeanette_practice_weeks_l634_63470


namespace NUMINAMATH_CALUDE_hclo4_moles_required_l634_63444

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

end NUMINAMATH_CALUDE_hclo4_moles_required_l634_63444


namespace NUMINAMATH_CALUDE_sum_of_i_powers_l634_63498

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- Theorem: Sum of specific powers of i equals 2 -/
theorem sum_of_i_powers : i^24 + i^29 + i^34 + i^39 + i^44 + i^49 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_i_powers_l634_63498


namespace NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l634_63442

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

end NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l634_63442


namespace NUMINAMATH_CALUDE_number_calculation_l634_63487

theorem number_calculation : ∃ x : ℚ, x = 2/15 + 1/5 + 1/2 :=
by sorry

end NUMINAMATH_CALUDE_number_calculation_l634_63487


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l634_63486

theorem similar_triangle_perimeter (a b c : ℝ) (h1 : a = 12) (h2 : b = 12) (h3 : c = 15) 
  (h4 : a = b) (h5 : c ≥ a) (h6 : c ≥ b) (long_side : ℝ) (h7 : long_side = 45) : 
  (long_side / c) * (a + b + c) = 117 :=
by sorry

end NUMINAMATH_CALUDE_similar_triangle_perimeter_l634_63486


namespace NUMINAMATH_CALUDE_division_problem_l634_63439

theorem division_problem (divisor quotient remainder number : ℕ) : 
  divisor = 12 → 
  quotient = 9 → 
  remainder = 1 → 
  number = divisor * quotient + remainder → 
  number = 109 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l634_63439


namespace NUMINAMATH_CALUDE_sphere_tangent_angle_theorem_l634_63447

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

end NUMINAMATH_CALUDE_sphere_tangent_angle_theorem_l634_63447


namespace NUMINAMATH_CALUDE_equality_for_specific_values_l634_63429

theorem equality_for_specific_values : 
  ∃ (a b c : ℝ), a + b^2 * c = (a^2 + b) * (a + c) :=
sorry

end NUMINAMATH_CALUDE_equality_for_specific_values_l634_63429


namespace NUMINAMATH_CALUDE_product_of_repeating_decimal_and_22_l634_63453

/-- The repeating decimal 0.454545... --/
def repeating_decimal : ℚ := 5 / 11

theorem product_of_repeating_decimal_and_22 :
  repeating_decimal * 22 = 10 := by sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimal_and_22_l634_63453


namespace NUMINAMATH_CALUDE_combined_molecular_weight_l634_63455

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

end NUMINAMATH_CALUDE_combined_molecular_weight_l634_63455


namespace NUMINAMATH_CALUDE_parabola_midpoint_to_directrix_distance_l634_63402

/-- Given a parabola y² = 4x and a line passing through its focus intersecting the parabola
    at points A(x₁, y₁) and B(x₂, y₂) with |AB| = 7, the distance from the midpoint M of AB
    to the directrix of the parabola is 7/2. -/
theorem parabola_midpoint_to_directrix_distance
  (x₁ y₁ x₂ y₂ : ℝ) :
  y₁^2 = 4*x₁ →
  y₂^2 = 4*x₂ →
  (x₁ - 1)^2 + y₁^2 = (x₂ - 1)^2 + y₂^2 →
  (x₂ - x₁)^2 + (y₂ - y₁)^2 = 49 →
  (x₁ + x₂)/2 + 1 = 7/2 := by sorry

end NUMINAMATH_CALUDE_parabola_midpoint_to_directrix_distance_l634_63402


namespace NUMINAMATH_CALUDE_M_intersect_N_l634_63418

def M : Set ℕ := {0, 2, 4}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

theorem M_intersect_N : M ∩ N = {0, 4} := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_l634_63418


namespace NUMINAMATH_CALUDE_one_zero_in_interval_l634_63492

def f (x : ℝ) := 2*x + x^3 - 2

theorem one_zero_in_interval : ∃! x : ℝ, 0 < x ∧ x < 1 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_one_zero_in_interval_l634_63492


namespace NUMINAMATH_CALUDE_cos_sum_squared_one_solutions_l634_63459

theorem cos_sum_squared_one_solutions (x : ℝ) : 
  (Real.cos x)^2 + (Real.cos (2*x))^2 + (Real.cos (3*x))^2 = 1 ↔ 
  (∃ k : ℤ, x = π/2 + k*π ∨ 
            x = π/4 + 2*k*π ∨ 
            x = 3*π/4 + 2*k*π ∨ 
            x = π/6 + 2*k*π ∨ 
            x = 5*π/6 + 2*k*π) :=
by sorry

end NUMINAMATH_CALUDE_cos_sum_squared_one_solutions_l634_63459


namespace NUMINAMATH_CALUDE_tangent_line_equation_l634_63433

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

end NUMINAMATH_CALUDE_tangent_line_equation_l634_63433


namespace NUMINAMATH_CALUDE_decorative_window_area_ratio_l634_63414

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

end NUMINAMATH_CALUDE_decorative_window_area_ratio_l634_63414


namespace NUMINAMATH_CALUDE_man_walking_speed_percentage_l634_63489

/-- Proves that a man is walking at 70% of his usual speed given the conditions -/
theorem man_walking_speed_percentage (usual_time distance : ℝ) 
  (h1 : usual_time = 56)
  (h2 : distance > 0)
  (h3 : distance = usual_time * (distance / usual_time)) -- Speed * Time = Distance
  (h4 : distance = 80 * (distance / (56 + 24))) -- New time is 80 minutes
  : (distance / (56 + 24)) / (distance / usual_time) = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_man_walking_speed_percentage_l634_63489


namespace NUMINAMATH_CALUDE_min_score_11_l634_63452

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

end NUMINAMATH_CALUDE_min_score_11_l634_63452


namespace NUMINAMATH_CALUDE_intersection_union_ratio_l634_63449

/-- A rhombus with given diagonal lengths -/
structure Rhombus where
  short_diagonal : ℝ
  long_diagonal : ℝ

/-- The rotation of a rhombus by 90 degrees -/
def rotate_90 (r : Rhombus) : Rhombus := r

/-- The intersection of a rhombus and its 90 degree rotation -/
def intersection (r : Rhombus) : Set (ℝ × ℝ) := sorry

/-- The union of a rhombus and its 90 degree rotation -/
def union (r : Rhombus) : Set (ℝ × ℝ) := sorry

/-- The area of a set in 2D space -/
def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: The ratio of the intersection area to the union area is 1/2023 -/
theorem intersection_union_ratio (r : Rhombus) 
  (h1 : r.short_diagonal = 1) 
  (h2 : r.long_diagonal = 2023) : 
  area (intersection r) / area (union r) = 1 / 2023 := by sorry

end NUMINAMATH_CALUDE_intersection_union_ratio_l634_63449


namespace NUMINAMATH_CALUDE_vector_calculation_l634_63461

theorem vector_calculation : 
  2 • (((3 : ℝ), -2, 5) + ((-1 : ℝ), 6, -7)) = ((4 : ℝ), 8, -4) := by
sorry

end NUMINAMATH_CALUDE_vector_calculation_l634_63461


namespace NUMINAMATH_CALUDE_chandelier_illumination_probability_chandelier_illumination_probability_is_correct_l634_63426

/-- The probability of a chandelier with 3 parallel-connected bulbs being illuminated, 
    given that the probability of each bulb working properly is 0.7 -/
theorem chandelier_illumination_probability : ℝ :=
  let p : ℝ := 0.7  -- probability of each bulb working properly
  let num_bulbs : ℕ := 3  -- number of bulbs in parallel connection
  1 - (1 - p) ^ num_bulbs

/-- Proof that the probability of the chandelier being illuminated is 0.973 -/
theorem chandelier_illumination_probability_is_correct : 
  chandelier_illumination_probability = 0.973 := by
  sorry


end NUMINAMATH_CALUDE_chandelier_illumination_probability_chandelier_illumination_probability_is_correct_l634_63426


namespace NUMINAMATH_CALUDE_function_equality_l634_63493

theorem function_equality (f : ℝ → ℝ) :
  (∀ x : ℝ, f (x + 1) = 2 * x - 1) →
  (∀ x : ℝ, f x = 2 * x - 3) :=
by
  sorry

end NUMINAMATH_CALUDE_function_equality_l634_63493


namespace NUMINAMATH_CALUDE_certain_number_is_three_l634_63440

theorem certain_number_is_three (n : ℝ) (x : ℤ) (h1 : n^(2*x) = 3^(12-x)) (h2 : x = 4) : n = 3 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_three_l634_63440


namespace NUMINAMATH_CALUDE_power_of_three_difference_l634_63437

theorem power_of_three_difference : 3^(1+2+3) - (3^1 + 3^2 + 3^3) = 690 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_difference_l634_63437


namespace NUMINAMATH_CALUDE_range_of_a_l634_63466

theorem range_of_a (a : ℝ) : a > 5 → ∃ x : ℝ, x > -1 ∧ (x^2 + 3*x + 6) / (x + 1) < a := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l634_63466


namespace NUMINAMATH_CALUDE_A_subset_B_iff_l634_63405

/-- The set A parameterized by a -/
def A (a : ℝ) : Set ℝ := {x | 1 < a * x ∧ a * x < 2}

/-- The set B -/
def B : Set ℝ := {x | |x| < 1}

/-- Theorem stating the condition for A to be a subset of B -/
theorem A_subset_B_iff (a : ℝ) : A a ⊆ B ↔ |a| ≥ 2 ∨ a = 0 := by sorry

end NUMINAMATH_CALUDE_A_subset_B_iff_l634_63405


namespace NUMINAMATH_CALUDE_direct_proportion_condition_l634_63456

/-- A function f(x) is a direct proportion function if there exists a non-zero constant k such that f(x) = k * x for all x -/
def IsDirectProportionFunction (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x

/-- The function defined by m -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x + (m^2 - 4)

theorem direct_proportion_condition (m : ℝ) :
  IsDirectProportionFunction (f m) ↔ m = -2 := by sorry

end NUMINAMATH_CALUDE_direct_proportion_condition_l634_63456


namespace NUMINAMATH_CALUDE_joan_found_79_seashells_l634_63407

/-- The number of seashells Mike gave to Joan -/
def mike_seashells : ℕ := 63

/-- The total number of seashells Joan has -/
def total_seashells : ℕ := 142

/-- The number of seashells Joan found initially -/
def joan_initial_seashells : ℕ := total_seashells - mike_seashells

theorem joan_found_79_seashells : joan_initial_seashells = 79 := by sorry

end NUMINAMATH_CALUDE_joan_found_79_seashells_l634_63407


namespace NUMINAMATH_CALUDE_probability_rain_given_wind_l634_63404

theorem probability_rain_given_wind (P_rain P_wind P_rain_and_wind : ℝ) 
  (h1 : P_rain = 4/15)
  (h2 : P_wind = 2/5)
  (h3 : P_rain_and_wind = 1/10) :
  P_rain_and_wind / P_wind = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_rain_given_wind_l634_63404


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a12_l634_63491

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a12 (a : ℕ → ℤ) 
  (h_arith : arithmetic_sequence a) 
  (h_a4 : a 4 = 4) 
  (h_a8 : a 8 = -4) : 
  a 12 = -12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a12_l634_63491


namespace NUMINAMATH_CALUDE_exists_unobserved_planet_l634_63454

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

end NUMINAMATH_CALUDE_exists_unobserved_planet_l634_63454


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l634_63443

theorem nested_fraction_evaluation : 
  1 / (2 + 1 / (3 + 1 / 4)) = 13 / 30 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l634_63443


namespace NUMINAMATH_CALUDE_unique_non_range_value_l634_63401

def f (k : ℚ) (x : ℚ) : ℚ := (2 * x + k) / (3 * x + 4)

theorem unique_non_range_value (k : ℚ) :
  (f k 5 = 5) →
  (f k 100 = 100) →
  (∀ x ≠ (-4/3), f k (f k x) = x) →
  ∃! y, ∀ x, f k x ≠ y ∧ y = (-8/13) :=
sorry

end NUMINAMATH_CALUDE_unique_non_range_value_l634_63401


namespace NUMINAMATH_CALUDE_lowercase_count_l634_63409

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

end NUMINAMATH_CALUDE_lowercase_count_l634_63409


namespace NUMINAMATH_CALUDE_sqrt_eighteen_div_sqrt_two_equals_three_l634_63480

theorem sqrt_eighteen_div_sqrt_two_equals_three : 
  Real.sqrt 18 / Real.sqrt 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eighteen_div_sqrt_two_equals_three_l634_63480


namespace NUMINAMATH_CALUDE_layla_apples_l634_63481

theorem layla_apples (maggie : ℕ) (kelsey : ℕ) (layla : ℕ) :
  maggie = 40 →
  kelsey = 28 →
  (maggie + kelsey + layla) / 3 = 30 →
  layla = 22 :=
by sorry

end NUMINAMATH_CALUDE_layla_apples_l634_63481


namespace NUMINAMATH_CALUDE_expression_evaluation_l634_63485

theorem expression_evaluation :
  let x : ℤ := -1
  let y : ℤ := 2
  (2 * x^2 - 2 * y^2) - 3 * (x^2 * y^2 + x^2) + 3 * (x^2 * y^2 + y^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l634_63485


namespace NUMINAMATH_CALUDE_unique_satisfying_pair_satisfying_pair_is_negative_one_zero_l634_63430

/-- Predicate that checks if a pair (m, n) satisfies the condition for all (x, y) -/
def satisfies_condition (m n : ℝ) : Prop :=
  ∀ x y : ℝ, y ≠ 0 → x / y = m → (x + y)^2 = n

/-- Theorem stating that (-1, 0) is the only pair satisfying the condition -/
theorem unique_satisfying_pair :
  ∃! p : ℝ × ℝ, satisfies_condition p.1 p.2 ∧ p = (-1, 0) := by
  sorry

/-- Corollary: If (m, n) satisfies the condition, then m = -1 and n = 0 -/
theorem satisfying_pair_is_negative_one_zero (m n : ℝ) :
  satisfies_condition m n → m = -1 ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_satisfying_pair_satisfying_pair_is_negative_one_zero_l634_63430


namespace NUMINAMATH_CALUDE_f_is_quasi_even_l634_63484

/-- A function is quasi-even if f(-x) = f(x) only for a finite number of non-zero arguments x. -/
def QuasiEven (f : ℝ → ℝ) : Prop :=
  ∃ (S : Finset ℝ), ∀ x ≠ 0, f (-x) = f x ↔ x ∈ S

/-- The function f(x) = x³ - 2x -/
def f (x : ℝ) : ℝ := x^3 - 2*x

/-- Theorem: f(x) = x³ - 2x is a quasi-even function -/
theorem f_is_quasi_even : QuasiEven f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quasi_even_l634_63484


namespace NUMINAMATH_CALUDE_intersection_points_line_slope_l634_63434

theorem intersection_points_line_slope :
  ∀ (s : ℝ) (x y : ℝ),
    (2 * x - 3 * y = 4 * s + 6) →
    (2 * x + y = 3 * s + 1) →
    y = -2/13 * x - 14/13 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_line_slope_l634_63434


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l634_63495

theorem quadratic_equation_solution (c : ℝ) : 
  (∃ x : ℝ, x^2 - x + c = 0 ∧ x = 1) → 
  (∃ x : ℝ, x^2 - x + c = 0 ∧ x = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l634_63495


namespace NUMINAMATH_CALUDE_cos_minus_sin_for_point_l634_63406

theorem cos_minus_sin_for_point (α : Real) :
  (∃ (r : Real), r > 0 ∧ r * Real.cos α = Real.sqrt 3 ∧ r * Real.sin α = -1) →
  Real.cos α - Real.sin α = (Real.sqrt 3 + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_minus_sin_for_point_l634_63406


namespace NUMINAMATH_CALUDE_seventeen_minus_fifteen_factorial_prime_divisors_l634_63448

-- Define factorial function
def factorial (n : ℕ) : ℕ := Nat.factorial n

-- Define the number of prime divisors function
def num_prime_divisors (n : ℕ) : ℕ := (Nat.factorization n).support.card

-- Theorem statement
theorem seventeen_minus_fifteen_factorial_prime_divisors :
  num_prime_divisors (factorial 17 - factorial 15) = 7 := by
  sorry

end NUMINAMATH_CALUDE_seventeen_minus_fifteen_factorial_prime_divisors_l634_63448


namespace NUMINAMATH_CALUDE_rectangle_area_change_l634_63468

theorem rectangle_area_change (L B : ℝ) (h1 : L > 0) (h2 : B > 0) :
  let new_length := L / 2
  let new_area := L * B / 2
  new_length * B = new_area → B = B :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l634_63468


namespace NUMINAMATH_CALUDE_number_problem_l634_63400

theorem number_problem (x : ℝ) : (0.95 * x - 12 = 178) → x = 200 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l634_63400


namespace NUMINAMATH_CALUDE_sqrt_meaningful_l634_63428

theorem sqrt_meaningful (x : ℝ) : (∃ y : ℝ, y ^ 2 = 1 - x) ↔ x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_l634_63428


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l634_63411

theorem scientific_notation_equivalence : 
  274000000 = 2.74 * (10 ^ 8) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l634_63411


namespace NUMINAMATH_CALUDE_expression_evaluation_l634_63476

theorem expression_evaluation (c d : ℕ) (hc : c = 4) (hd : d = 2) :
  (c^c - c*(c-d)^c)^c = 136048896 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l634_63476


namespace NUMINAMATH_CALUDE_tower_surface_area_theorem_l634_63475

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

end NUMINAMATH_CALUDE_tower_surface_area_theorem_l634_63475


namespace NUMINAMATH_CALUDE_opposite_sum_zero_sum_zero_opposite_exists_opposite_not_negative_one_negative_one_ratio_opposite_l634_63497

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

end NUMINAMATH_CALUDE_opposite_sum_zero_sum_zero_opposite_exists_opposite_not_negative_one_negative_one_ratio_opposite_l634_63497


namespace NUMINAMATH_CALUDE_equal_savings_l634_63472

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

end NUMINAMATH_CALUDE_equal_savings_l634_63472


namespace NUMINAMATH_CALUDE_four_men_absent_l634_63436

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

end NUMINAMATH_CALUDE_four_men_absent_l634_63436


namespace NUMINAMATH_CALUDE_no_even_primes_greater_than_two_l634_63464

theorem no_even_primes_greater_than_two :
  ∀ n : ℕ, n > 2 → Prime n → ¬Even n :=
by
  sorry

end NUMINAMATH_CALUDE_no_even_primes_greater_than_two_l634_63464


namespace NUMINAMATH_CALUDE_intersection_condition_l634_63462

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
def B (p : ℝ) : Set ℝ := {x : ℝ | p + 1 ≤ x ∧ x ≤ 2*p - 1}

theorem intersection_condition (p : ℝ) : A ∩ B p = B p ↔ p ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l634_63462


namespace NUMINAMATH_CALUDE_combined_weight_is_9500_l634_63465

def regular_dinosaur_weight : ℕ := 800
def number_of_regular_dinosaurs : ℕ := 5
def barney_extra_weight : ℕ := 1500

def combined_weight : ℕ :=
  (regular_dinosaur_weight * number_of_regular_dinosaurs) + 
  (regular_dinosaur_weight * number_of_regular_dinosaurs + barney_extra_weight)

theorem combined_weight_is_9500 : combined_weight = 9500 := by
  sorry

end NUMINAMATH_CALUDE_combined_weight_is_9500_l634_63465


namespace NUMINAMATH_CALUDE_best_fit_highest_r_squared_l634_63413

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

end NUMINAMATH_CALUDE_best_fit_highest_r_squared_l634_63413


namespace NUMINAMATH_CALUDE_unique_solution_l634_63458

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: 36018 is the only positive integer m that satisfies 2001 * S(m) = m -/
theorem unique_solution :
  ∀ m : ℕ, m > 0 → (2001 * sumOfDigits m = m) ↔ m = 36018 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l634_63458


namespace NUMINAMATH_CALUDE_largest_integer_l634_63419

theorem largest_integer (a b c : ℤ) : 
  (2 * a + 3 * b + 4 * c = 225) →
  (a + b + c = 60) →
  (a = 15 ∨ b = 15 ∨ c = 15) →
  (max a (max b c) = 25) :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_l634_63419


namespace NUMINAMATH_CALUDE_factorial_difference_l634_63423

theorem factorial_difference : Nat.factorial 12 - Nat.factorial 11 = 439084800 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l634_63423


namespace NUMINAMATH_CALUDE_geometry_propositions_l634_63415

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

end NUMINAMATH_CALUDE_geometry_propositions_l634_63415


namespace NUMINAMATH_CALUDE_cow_calf_ratio_l634_63408

def total_cost : ℕ := 990
def cow_cost : ℕ := 880
def calf_cost : ℕ := 110

theorem cow_calf_ratio : 
  ∃ (m : ℕ), m > 0 ∧ cow_cost = m * calf_cost ∧ cow_cost / calf_cost = 8 := by
  sorry

end NUMINAMATH_CALUDE_cow_calf_ratio_l634_63408


namespace NUMINAMATH_CALUDE_gcd_factorial_8_9_l634_63490

theorem gcd_factorial_8_9 : Nat.gcd (Nat.factorial 8) (Nat.factorial 9) = Nat.factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_8_9_l634_63490


namespace NUMINAMATH_CALUDE_john_money_needed_l634_63471

/-- The amount of money John currently has, in dollars -/
def current_amount : ℚ := 0.75

/-- The additional amount of money John needs, in dollars -/
def additional_amount : ℚ := 1.75

/-- The total amount of money John needs, in dollars -/
def total_amount : ℚ := current_amount + additional_amount

theorem john_money_needed : total_amount = 2.50 := by sorry

end NUMINAMATH_CALUDE_john_money_needed_l634_63471


namespace NUMINAMATH_CALUDE_inequality_addition_l634_63467

theorem inequality_addition (x y : ℝ) (h : x < y) : x + 6 < y + 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_addition_l634_63467


namespace NUMINAMATH_CALUDE_distance_to_focus_l634_63417

/-- Given a parabola y² = 8x and a point P on it, prove that the distance from P to the focus is 10 -/
theorem distance_to_focus (x₀ : ℝ) : 
  8^2 = 8 * x₀ →  -- P(x₀, 8) is on the parabola y² = 8x
  Real.sqrt ((x₀ - 2)^2 + 8^2) = 10 := by sorry

end NUMINAMATH_CALUDE_distance_to_focus_l634_63417
