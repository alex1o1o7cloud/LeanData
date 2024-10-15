import Mathlib

namespace NUMINAMATH_CALUDE_melanie_has_41_balloons_l3748_374898

/-- The number of blue balloons Joan has -/
def joan_balloons : ℕ := 40

/-- The total number of blue balloons -/
def total_balloons : ℕ := 81

/-- The number of blue balloons Melanie has -/
def melanie_balloons : ℕ := total_balloons - joan_balloons

theorem melanie_has_41_balloons : melanie_balloons = 41 := by
  sorry

end NUMINAMATH_CALUDE_melanie_has_41_balloons_l3748_374898


namespace NUMINAMATH_CALUDE_gcf_54_81_l3748_374871

theorem gcf_54_81 : Nat.gcd 54 81 = 27 := by
  sorry

end NUMINAMATH_CALUDE_gcf_54_81_l3748_374871


namespace NUMINAMATH_CALUDE_one_valid_placement_l3748_374890

/-- Represents the number of pegs of each color -/
structure PegCounts where
  purple : Nat
  yellow : Nat
  red : Nat
  green : Nat
  blue : Nat

/-- Represents a hexagonal peg board -/
structure HexBoard where
  rows : Nat
  columns : Nat

/-- Counts the number of valid peg placements -/
def countValidPlacements (board : HexBoard) (pegs : PegCounts) : Nat :=
  sorry

/-- Theorem stating that there is exactly one valid placement -/
theorem one_valid_placement (board : HexBoard) (pegs : PegCounts) : 
  board.rows = 6 ∧ board.columns = 6 ∧ 
  pegs.purple = 6 ∧ pegs.yellow = 5 ∧ pegs.red = 4 ∧ pegs.green = 3 ∧ pegs.blue = 2 →
  countValidPlacements board pegs = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_valid_placement_l3748_374890


namespace NUMINAMATH_CALUDE_point_on_line_l3748_374827

/-- Given a line in 3D space defined by the vector equation (x,y,z) = (5,0,3) + t(0,3,0),
    this theorem proves that the point on the line when t = 1/2 has coordinates (5,3/2,3). -/
theorem point_on_line (x y z t : ℝ) : 
  (x, y, z) = (5, 0, 3) + t • (0, 3, 0) → 
  t = 1/2 → 
  (x, y, z) = (5, 3/2, 3) := by
sorry

end NUMINAMATH_CALUDE_point_on_line_l3748_374827


namespace NUMINAMATH_CALUDE_series_sum_l3748_374824

/-- The sum of the infinite series ∑(n=1 to ∞) (5n-1)/(3^n) is equal to 13/6 -/
theorem series_sum : (∑' n : ℕ, (5 * n - 1 : ℝ) / (3 : ℝ) ^ n) = 13 / 6 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_l3748_374824


namespace NUMINAMATH_CALUDE_not_divisible_by_1955_l3748_374863

theorem not_divisible_by_1955 : ∀ n : ℤ, ¬(1955 ∣ (n^2 + n + 1)) := by sorry

end NUMINAMATH_CALUDE_not_divisible_by_1955_l3748_374863


namespace NUMINAMATH_CALUDE_good_quadruple_inequality_l3748_374851

/-- A good quadruple is a set of positive integers (p, a, b, c) satisfying certain conditions. -/
structure GoodQuadruple where
  p : Nat
  a : Nat
  b : Nat
  c : Nat
  p_prime : Nat.Prime p
  p_odd : Odd p
  distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c
  div_ab : p ∣ (a * b + 1)
  div_bc : p ∣ (b * c + 1)
  div_ca : p ∣ (c * a + 1)

/-- The main theorem about good quadruples. -/
theorem good_quadruple_inequality (q : GoodQuadruple) :
  q.p + 2 ≤ (q.a + q.b + q.c) / 3 ∧
  (q.p + 2 = (q.a + q.b + q.c) / 3 ↔ q.a = 2 ∧ q.b = 2 + q.p ∧ q.c = 2 + 2 * q.p) :=
by sorry

end NUMINAMATH_CALUDE_good_quadruple_inequality_l3748_374851


namespace NUMINAMATH_CALUDE_career_preference_proof_l3748_374837

/-- Represents the ratio of boys to girls in a class -/
def boyGirlRatio : ℚ := 2/3

/-- Represents the fraction of the circle graph allocated to a specific career -/
def careerFraction : ℚ := 192/360

/-- Represents the fraction of girls who prefer the specific career -/
def girlPreferenceFraction : ℚ := 2/3

/-- Represents the fraction of boys who prefer the specific career -/
def boyPreferenceFraction : ℚ := 1/3

theorem career_preference_proof :
  let totalStudents := boyGirlRatio + 1
  let boyFraction := boyGirlRatio / totalStudents
  let girlFraction := 1 / totalStudents
  careerFraction = boyFraction * boyPreferenceFraction + girlFraction * girlPreferenceFraction :=
by sorry

end NUMINAMATH_CALUDE_career_preference_proof_l3748_374837


namespace NUMINAMATH_CALUDE_center_locus_is_single_point_l3748_374835

/-- Two fixed points in a plane -/
structure FixedPoints (α : Type*) [NormedAddCommGroup α] where
  P : α
  Q : α

/-- A circle passing through two fixed points with constant radius -/
structure Circle (α : Type*) [NormedAddCommGroup α] where
  center : α
  radius : ℝ
  fixedPoints : FixedPoints α

/-- The locus of centers of circles passing through two fixed points -/
def CenterLocus (α : Type*) [NormedAddCommGroup α] (a : ℝ) : Set α :=
  {C : α | ∃ (circ : Circle α), circ.center = C ∧ circ.radius = a}

/-- The theorem stating that the locus of centers is a single point -/
theorem center_locus_is_single_point
  (α : Type*) [NormedAddCommGroup α] [NormedSpace ℝ α]
  (a : ℝ) (points : FixedPoints α)
  (h : ‖points.P - points.Q‖ = 2 * a) :
  ∃! C, C ∈ CenterLocus α a :=
sorry

end NUMINAMATH_CALUDE_center_locus_is_single_point_l3748_374835


namespace NUMINAMATH_CALUDE_complement_P_intersect_Q_l3748_374861

def P : Set ℝ := {x | x^2 - 2*x ≥ 0}
def Q : Set ℝ := {x | 0 < Real.log x ∧ Real.log x ≤ Real.log 2}

theorem complement_P_intersect_Q : 
  (Set.compl P) ∩ Q = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_complement_P_intersect_Q_l3748_374861


namespace NUMINAMATH_CALUDE_line_arrangement_with_restriction_l3748_374820

def number_of_students : ℕ := 5

def total_arrangements (n : ℕ) : ℕ := n.factorial

def restricted_arrangements (n : ℕ) : ℕ := 
  (n - 1).factorial * 2

theorem line_arrangement_with_restriction :
  total_arrangements number_of_students - restricted_arrangements number_of_students = 72 := by
  sorry

end NUMINAMATH_CALUDE_line_arrangement_with_restriction_l3748_374820


namespace NUMINAMATH_CALUDE_select_and_swap_count_l3748_374856

def num_people : ℕ := 8
def num_selected : ℕ := 3

def ways_to_select_and_swap : ℕ := Nat.choose num_people num_selected * (Nat.factorial 2)

theorem select_and_swap_count :
  ways_to_select_and_swap = Nat.choose num_people num_selected * (Nat.factorial 2) :=
by sorry

end NUMINAMATH_CALUDE_select_and_swap_count_l3748_374856


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3748_374833

theorem min_value_quadratic (x : ℝ) : 
  ∃ (z_min : ℝ), ∀ (z : ℝ), z = x^2 + 16*x + 20 → z ≥ z_min ∧ ∃ (x_min : ℝ), x_min^2 + 16*x_min + 20 = z_min :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3748_374833


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l3748_374865

theorem unique_quadratic_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 30 * x + c = 0) →
  a + c = 45 →
  a < c →
  (a = (45 - 15 * Real.sqrt 5) / 2 ∧ c = (45 + 15 * Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l3748_374865


namespace NUMINAMATH_CALUDE_perpendicular_planes_condition_l3748_374829

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and subset relations
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_condition
  (a b : Line) (α β : Plane)
  (skew : a ≠ b)
  (perp_a_α : perpendicular a α)
  (perp_b_β : perpendicular b β)
  (not_subset_a_β : ¬subset a β)
  (not_subset_b_α : ¬subset b α) :
  perpendicular_planes α β ↔ perpendicular_lines a b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_condition_l3748_374829


namespace NUMINAMATH_CALUDE_smallest_square_partition_l3748_374866

/-- Represents a square partition of a larger square -/
structure SquarePartition where
  side_length : ℕ
  partitions : List ℕ
  partition_count : partitions.length = 15
  all_integer : ∀ n ∈ partitions, n > 0
  sum_areas : (partitions.map (λ x => x * x)).sum = side_length * side_length
  unit_squares : (partitions.filter (λ x => x = 1)).length ≥ 12

/-- The smallest square that satisfies the partition conditions has side length 5 -/
theorem smallest_square_partition :
  ∀ sp : SquarePartition, sp.side_length ≥ 5 ∧
  ∃ sp' : SquarePartition, sp'.side_length = 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_partition_l3748_374866


namespace NUMINAMATH_CALUDE_zoo_trip_theorem_l3748_374860

/-- Calculates the remaining money for lunch and snacks after a zoo trip for two people -/
def zoo_trip_remaining_money (ticket_price : ℚ) (bus_fare_one_way : ℚ) (total_money : ℚ) : ℚ :=
  let num_people : ℚ := 2
  let total_ticket_cost := ticket_price * num_people
  let total_bus_fare := bus_fare_one_way * num_people * 2
  let total_trip_cost := total_ticket_cost + total_bus_fare
  total_money - total_trip_cost

theorem zoo_trip_theorem :
  zoo_trip_remaining_money 5 1.5 40 = 24 := by
  sorry

end NUMINAMATH_CALUDE_zoo_trip_theorem_l3748_374860


namespace NUMINAMATH_CALUDE_computer_music_time_l3748_374838

def total_time : ℕ := 120
def piano_time : ℕ := 30
def reading_time : ℕ := 38
def exerciser_time : ℕ := 27

theorem computer_music_time : 
  total_time - (piano_time + reading_time + exerciser_time) = 25 := by
sorry

end NUMINAMATH_CALUDE_computer_music_time_l3748_374838


namespace NUMINAMATH_CALUDE_least_positive_angle_theorem_l3748_374810

theorem least_positive_angle_theorem (θ : Real) : 
  (θ > 0 ∧ θ ≤ π / 2) → 
  (Real.cos (10 * π / 180) = Real.sin (20 * π / 180) + Real.sin θ) → 
  θ = 40 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_least_positive_angle_theorem_l3748_374810


namespace NUMINAMATH_CALUDE_factory_task_excess_l3748_374839

theorem factory_task_excess (first_half : Rat) (second_half : Rat)
  (h1 : first_half = 2 / 3)
  (h2 : second_half = 3 / 5) :
  first_half + second_half - 1 = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_factory_task_excess_l3748_374839


namespace NUMINAMATH_CALUDE_cubic_function_value_l3748_374826

/-- Given a cubic function f(x) = ax³ + bx + 3 where f(-3) = 10, prove that f(3) = 27a + 3b + 3 -/
theorem cubic_function_value (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^3 + b * x + 3)
  (h2 : f (-3) = 10) :
  f 3 = 27 * a + 3 * b + 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_value_l3748_374826


namespace NUMINAMATH_CALUDE_inscribed_iff_side_length_le_l3748_374864

/-- A regular polygon -/
structure RegularPolygon where
  n : ℕ
  sideLength : ℝ

/-- A circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a regular polygon is inscribed in a circle -/
def isInscribed (p : RegularPolygon) (c : Circle) : Prop :=
  sorry

/-- The side length of an inscribed regular n-gon in a given circle -/
def inscribedSideLength (n : ℕ) (c : Circle) : ℝ :=
  sorry

theorem inscribed_iff_side_length_le
  (n : ℕ) (c : Circle) (p : RegularPolygon) 
  (h1 : p.n = n) :
  isInscribed p c ↔ p.sideLength ≤ inscribedSideLength n c :=
sorry

end NUMINAMATH_CALUDE_inscribed_iff_side_length_le_l3748_374864


namespace NUMINAMATH_CALUDE_domino_tiling_theorem_l3748_374828

/-- Represents a rectangular board -/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a domino placement on a board -/
def Tiling (b : Board) := Set (ℕ × ℕ × Bool)

/-- Checks if a tiling is valid for a given board -/
def is_valid_tiling (b : Board) (t : Tiling b) : Prop := sorry

/-- Checks if a line bisects at least one domino in the tiling -/
def line_bisects_domino (b : Board) (t : Tiling b) (line : ℕ × Bool) : Prop := sorry

/-- Counts the number of internal lines in a board -/
def internal_lines_count (b : Board) : ℕ := 
  b.rows + b.cols - 2

/-- Main theorem statement -/
theorem domino_tiling_theorem :
  (¬ ∃ (t : Tiling ⟨6, 6⟩), 
    is_valid_tiling ⟨6, 6⟩ t ∧ 
    ∀ (line : ℕ × Bool), line_bisects_domino ⟨6, 6⟩ t line) ∧
  (∃ (t : Tiling ⟨5, 6⟩), 
    is_valid_tiling ⟨5, 6⟩ t ∧ 
    ∀ (line : ℕ × Bool), line_bisects_domino ⟨5, 6⟩ t line) :=
sorry

end NUMINAMATH_CALUDE_domino_tiling_theorem_l3748_374828


namespace NUMINAMATH_CALUDE_product_plus_one_composite_l3748_374800

theorem product_plus_one_composite : 
  ∃ (a b : ℤ), b > 1 ∧ 2014 * 2015 * 2016 * 2017 + 1 = a * b := by
  sorry

end NUMINAMATH_CALUDE_product_plus_one_composite_l3748_374800


namespace NUMINAMATH_CALUDE_passengers_from_other_continents_l3748_374896

theorem passengers_from_other_continents 
  (total : ℕ) 
  (h_total : total = 240) 
  (h_na : total / 3 = 80) 
  (h_eu : total / 8 = 30) 
  (h_af : total / 5 = 48) 
  (h_as : total / 6 = 40) : 
  total - (total / 3 + total / 8 + total / 5 + total / 6) = 42 := by
  sorry

end NUMINAMATH_CALUDE_passengers_from_other_continents_l3748_374896


namespace NUMINAMATH_CALUDE_tomatoes_left_l3748_374830

theorem tomatoes_left (initial_tomatoes : ℕ) (eaten_fraction : ℚ) : initial_tomatoes = 21 ∧ eaten_fraction = 1/3 → initial_tomatoes - (initial_tomatoes * eaten_fraction).floor = 14 := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_left_l3748_374830


namespace NUMINAMATH_CALUDE_seventh_group_draw_l3748_374857

/-- Represents the systematic sampling method for a population -/
structure SystematicSampling where
  population_size : Nat
  num_groups : Nat
  sample_size : Nat
  m : Nat

/-- Calculates the number drawn from a specific group -/
def number_drawn (ss : SystematicSampling) (group : Nat) : Nat :=
  let group_size := ss.population_size / ss.num_groups
  let start := (group - 1) * group_size
  let units_digit := (ss.m + group) % 10
  start + units_digit

/-- Theorem stating that the number drawn from the 7th group is 63 -/
theorem seventh_group_draw (ss : SystematicSampling) 
  (h1 : ss.population_size = 100)
  (h2 : ss.num_groups = 10)
  (h3 : ss.sample_size = 10)
  (h4 : ss.m = 6) :
  number_drawn ss 7 = 63 := by
  sorry

end NUMINAMATH_CALUDE_seventh_group_draw_l3748_374857


namespace NUMINAMATH_CALUDE_satellite_forecast_probability_l3748_374853

theorem satellite_forecast_probability (p_a p_b : ℝ) (h_a : p_a = 0.8) (h_b : p_b = 0.75) :
  1 - (1 - p_a) * (1 - p_b) = 0.95 := by
  sorry

end NUMINAMATH_CALUDE_satellite_forecast_probability_l3748_374853


namespace NUMINAMATH_CALUDE_weight_loss_challenge_l3748_374817

theorem weight_loss_challenge (original_weight : ℝ) (x : ℝ) : 
  x > 0 →
  x < 100 →
  let final_weight := original_weight * (1 - x / 100 + 2 / 100)
  let measured_loss_percentage := 13.3
  final_weight = original_weight * (1 - measured_loss_percentage / 100) →
  x = 15.3 := by
sorry

end NUMINAMATH_CALUDE_weight_loss_challenge_l3748_374817


namespace NUMINAMATH_CALUDE_product_correction_l3748_374814

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem product_correction (a b : ℕ) :
  a ≥ 10 ∧ a < 100 →  -- a is a two-digit number
  a > 0 ∧ b > 0 →  -- a and b are positive
  (reverse_digits a) * b = 143 →
  a * b = 341 := by
sorry

end NUMINAMATH_CALUDE_product_correction_l3748_374814


namespace NUMINAMATH_CALUDE_robs_reading_l3748_374882

/-- Given Rob's planned reading time, actual reading time as a fraction of planned time,
    and his reading speed, calculate the number of pages he read. -/
theorem robs_reading (planned_hours : ℝ) (actual_fraction : ℝ) (pages_per_minute : ℝ) : 
  planned_hours = 3 →
  actual_fraction = 3/4 →
  pages_per_minute = 1/15 →
  (planned_hours * actual_fraction * 60) * pages_per_minute = 9 := by
  sorry

end NUMINAMATH_CALUDE_robs_reading_l3748_374882


namespace NUMINAMATH_CALUDE_negative_integer_problem_l3748_374831

theorem negative_integer_problem (n : ℤ) : n < 0 → n * (-3) + 2 = 65 → n = -21 := by
  sorry

end NUMINAMATH_CALUDE_negative_integer_problem_l3748_374831


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3748_374888

theorem complex_magnitude_problem : 
  let z : ℂ := (1 + 3*I) / (3 - I) - 3*I
  Complex.abs z = 2 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3748_374888


namespace NUMINAMATH_CALUDE_fraction_equality_l3748_374843

theorem fraction_equality (a b c d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) 
  (h1 : (a / b)^2 = (c / d)^2) (h2 : a * c < 0) : 
  a / b = -(c / d) := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l3748_374843


namespace NUMINAMATH_CALUDE_ab_leq_one_l3748_374897

theorem ab_leq_one (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hab : a + b = 2) : a * b ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ab_leq_one_l3748_374897


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l3748_374885

theorem sufficient_condition_for_inequality (a x : ℝ) : 
  (-2 < x ∧ x < -1) → (a > 2 → (a + x) * (1 + x) < 0) := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l3748_374885


namespace NUMINAMATH_CALUDE_x_values_difference_l3748_374832

theorem x_values_difference (x : ℝ) : 
  (x + 3)^2 / (3*x + 29) = 2 → ∃ (x₁ x₂ : ℝ), x₁ - x₂ = 14 ∧ 
    ((x₁ + 3)^2 / (3*x₁ + 29) = 2) ∧ ((x₂ + 3)^2 / (3*x₂ + 29) = 2) := by
  sorry

end NUMINAMATH_CALUDE_x_values_difference_l3748_374832


namespace NUMINAMATH_CALUDE_roots_of_cubic_equations_l3748_374872

theorem roots_of_cubic_equations (p q r s : ℂ) (m : ℂ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0)
  (h1 : p * m^3 + q * m^2 + r * m + s = 0)
  (h2 : q * m^3 + r * m^2 + s * m + p = 0) :
  m = 1 ∨ m = -1 ∨ m = Complex.I ∨ m = -Complex.I := by
sorry

end NUMINAMATH_CALUDE_roots_of_cubic_equations_l3748_374872


namespace NUMINAMATH_CALUDE_vector_addition_problem_l3748_374819

theorem vector_addition_problem (a b : ℝ × ℝ) :
  a = (2, -1) → b = (-3, 4) → 2 • a + b = (1, 2) := by sorry

end NUMINAMATH_CALUDE_vector_addition_problem_l3748_374819


namespace NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l3748_374806

theorem factorization_cubic_minus_linear (a : ℝ) : a^3 - 4*a = a*(a+2)*(a-2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l3748_374806


namespace NUMINAMATH_CALUDE_sum_of_six_numbers_l3748_374884

theorem sum_of_six_numbers : (36 : ℕ) + 17 + 32 + 54 + 28 + 3 = 170 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_six_numbers_l3748_374884


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l3748_374852

theorem area_between_concentric_circles 
  (r_outer : ℝ) 
  (r_inner : ℝ) 
  (chord_length : ℝ) 
  (h_r_outer : r_outer = 60) 
  (h_r_inner : r_inner = 36) 
  (h_chord : chord_length = 96) 
  (h_tangent : chord_length / 2 = Real.sqrt (r_outer^2 - r_inner^2)) : 
  π * (r_outer^2 - r_inner^2) = 2304 * π := by
sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l3748_374852


namespace NUMINAMATH_CALUDE_cycling_equation_correct_l3748_374881

/-- Represents the scenario of two employees cycling to work -/
def cycling_scenario (x : ℝ) : Prop :=
  let distance : ℝ := 5000
  let speed_ratio : ℝ := 1.5
  let time_difference : ℝ := 10
  (distance / x) - (distance / (speed_ratio * x)) = time_difference

/-- Proves that the equation correctly represents the cycling scenario -/
theorem cycling_equation_correct :
  ∀ x : ℝ, x > 0 → cycling_scenario x :=
by
  sorry

end NUMINAMATH_CALUDE_cycling_equation_correct_l3748_374881


namespace NUMINAMATH_CALUDE_abc_value_for_specific_factorization_l3748_374834

theorem abc_value_for_specific_factorization (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = (x - 1) * (x - 2)) → a * b * c = -6 := by
  sorry

end NUMINAMATH_CALUDE_abc_value_for_specific_factorization_l3748_374834


namespace NUMINAMATH_CALUDE_square_land_side_length_l3748_374886

theorem square_land_side_length (area : ℝ) (h : area = Real.sqrt 900) :
  ∃ (side : ℝ), side * side = area ∧ side = 30 := by
  sorry

end NUMINAMATH_CALUDE_square_land_side_length_l3748_374886


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3748_374895

/-- An arithmetic sequence with first term a₁ and common difference d -/
structure ArithmeticSequence where
  a₁ : ℤ
  d : ℤ

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * seq.a₁ + (n * (n - 1) / 2) * seq.d

theorem arithmetic_sequence_sum (seq : ArithmeticSequence) :
  seq.a₁ = -2014 →
  (sum_n seq 2012 / 2012 : ℚ) - (sum_n seq 10 / 10 : ℚ) = 2002 →
  sum_n seq 2016 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3748_374895


namespace NUMINAMATH_CALUDE_worker_a_time_l3748_374809

theorem worker_a_time (worker_b_time worker_ab_time worker_a_time : ℝ) : 
  worker_b_time = 12 →
  worker_ab_time = 5.454545454545454 →
  worker_a_time = 10.153846153846153 →
  (1 / worker_a_time + 1 / worker_b_time) * worker_ab_time = 1 :=
by sorry

end NUMINAMATH_CALUDE_worker_a_time_l3748_374809


namespace NUMINAMATH_CALUDE_election_votes_l3748_374811

theorem election_votes (total_votes : ℕ) (invalid_percent : ℚ) (winner_percent : ℚ) : 
  total_votes = 9000 →
  invalid_percent = 30 / 100 →
  winner_percent = 60 / 100 →
  (total_votes : ℚ) * (1 - invalid_percent) * (1 - winner_percent) = 2520 := by
sorry

end NUMINAMATH_CALUDE_election_votes_l3748_374811


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3748_374816

-- Define the sets M and N
def M : Set ℝ := {x | x^2 = x}
def N : Set ℝ := {x | Real.log x ≤ 0}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3748_374816


namespace NUMINAMATH_CALUDE_bug_position_after_2021_jumps_l3748_374870

/-- Represents the seven points on the circle -/
inductive Point
| One | Two | Three | Four | Five | Six | Seven

/-- Determines if a point is prime -/
def isPrime : Point → Bool
  | Point.Two => true
  | Point.Three => true
  | Point.Five => true
  | Point.Seven => true
  | _ => false

/-- Calculates the next point based on the current point -/
def nextPoint (p : Point) : Point :=
  match p with
  | Point.One => Point.Four
  | Point.Two => Point.Four
  | Point.Three => Point.Five
  | Point.Four => Point.Seven
  | Point.Five => Point.Seven
  | Point.Six => Point.Two
  | Point.Seven => Point.Two

/-- Calculates the bug's position after n jumps -/
def bugPosition (start : Point) (n : Nat) : Point :=
  match n with
  | 0 => start
  | n + 1 => nextPoint (bugPosition start n)

/-- The main theorem to prove -/
theorem bug_position_after_2021_jumps :
  bugPosition Point.Seven 2021 = Point.Two :=
sorry

end NUMINAMATH_CALUDE_bug_position_after_2021_jumps_l3748_374870


namespace NUMINAMATH_CALUDE_distinct_collections_eq_33_l3748_374822

/-- Represents the number of each letter in 'MATHEMATICS' -/
def letter_counts : Fin 26 → Nat :=
  fun i => match i with
  | 0  => 2  -- 'A'
  | 4  => 1  -- 'E'
  | 8  => 1  -- 'I'
  | 12 => 2  -- 'M'
  | 19 => 2  -- 'T'
  | 2  => 1  -- 'C'
  | 7  => 1  -- 'H'
  | 18 => 1  -- 'S'
  | _  => 0

/-- The total number of letters -/
def total_letters : Nat := 11

/-- The number of vowels that fall off -/
def vowels_off : Nat := 3

/-- The number of consonants that fall off -/
def consonants_off : Nat := 2

/-- Function to check if a letter is a vowel -/
def is_vowel (i : Fin 26) : Bool :=
  i = 0 ∨ i = 4 ∨ i = 8 ∨ i = 14 ∨ i = 20

/-- Function to calculate the number of distinct collections -/
noncomputable def distinct_collections : Nat :=
  sorry

/-- Theorem stating that the number of distinct collections is 33 -/
theorem distinct_collections_eq_33 : distinct_collections = 33 :=
  sorry

end NUMINAMATH_CALUDE_distinct_collections_eq_33_l3748_374822


namespace NUMINAMATH_CALUDE_nonconvex_quadrilateral_theorem_l3748_374859

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Checks if a quadrilateral is nonconvex -/
def is_nonconvex (q : Quadrilateral) : Prop := sorry

/-- Calculates the angle at a vertex of a quadrilateral -/
def angle_at_vertex (q : Quadrilateral) (v : Point2D) : ℝ := sorry

/-- Finds the intersection point of two lines -/
def line_intersection (p1 p2 q1 q2 : Point2D) : Point2D := sorry

/-- Checks if a point lies on a line segment -/
def point_on_segment (p : Point2D) (a b : Point2D) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point2D) : ℝ := sorry

theorem nonconvex_quadrilateral_theorem (ABCD : Quadrilateral) 
  (hnonconvex : is_nonconvex ABCD)
  (hC_angle : angle_at_vertex ABCD ABCD.C > 180)
  (F : Point2D) (hF : F = line_intersection ABCD.D ABCD.C ABCD.A ABCD.B)
  (E : Point2D) (hE : E = line_intersection ABCD.B ABCD.C ABCD.A ABCD.D)
  (K L J I : Point2D)
  (hK : point_on_segment K ABCD.A ABCD.B)
  (hL : point_on_segment L ABCD.A ABCD.D)
  (hJ : point_on_segment J ABCD.B ABCD.C)
  (hI : point_on_segment I ABCD.C ABCD.D)
  (hDI_CF : distance ABCD.D I = distance ABCD.C F)
  (hBJ_CE : distance ABCD.B J = distance ABCD.C E) :
  distance K J = distance I L := by sorry

end NUMINAMATH_CALUDE_nonconvex_quadrilateral_theorem_l3748_374859


namespace NUMINAMATH_CALUDE_ten_mile_taxi_cost_l3748_374813

/-- Calculates the cost of a taxi ride given the base fare, per-mile rate, and distance traveled. -/
def taxiRideCost (baseFare : ℝ) (perMileRate : ℝ) (distance : ℝ) : ℝ :=
  baseFare + perMileRate * distance

/-- Theorem stating that a 10-mile taxi ride costs $5.00 given the specified base fare and per-mile rate. -/
theorem ten_mile_taxi_cost :
  let baseFare : ℝ := 2.00
  let perMileRate : ℝ := 0.30
  let distance : ℝ := 10
  taxiRideCost baseFare perMileRate distance = 5.00 := by
  sorry


end NUMINAMATH_CALUDE_ten_mile_taxi_cost_l3748_374813


namespace NUMINAMATH_CALUDE_rationalize_and_product_l3748_374873

theorem rationalize_and_product : ∃ (A B C : ℤ),
  (((2 : ℝ) + Real.sqrt 5) / ((3 : ℝ) - 2 * Real.sqrt 5) = A + B * Real.sqrt C) ∧
  A * B * C = -560 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_and_product_l3748_374873


namespace NUMINAMATH_CALUDE_missing_figure_proof_l3748_374844

theorem missing_figure_proof (x : ℝ) : (1.2 / 100) * x = 0.6 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_missing_figure_proof_l3748_374844


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3748_374802

theorem sufficient_but_not_necessary :
  (∀ x : ℝ, |x| < 1 → x^2 + x - 6 < 0) ∧
  (∃ x : ℝ, x^2 + x - 6 < 0 ∧ ¬(|x| < 1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3748_374802


namespace NUMINAMATH_CALUDE_division_problem_l3748_374899

theorem division_problem (a b q : ℕ) (h1 : a - b = 1365) (h2 : a = 1620) (h3 : a = b * q + 15) : q = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3748_374899


namespace NUMINAMATH_CALUDE_vectors_not_collinear_l3748_374869

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the proposition
def proposition (a b : V) : Prop :=
  ∀ k₁ k₂ : ℝ, k₁ • a + k₂ • b = 0 → k₁ ≠ 0 ∧ k₂ ≠ 0

-- State the theorem
theorem vectors_not_collinear (a b : V) :
  proposition a b → ¬(∃ (t : ℝ), a = t • b ∨ b = t • a) :=
by sorry

end NUMINAMATH_CALUDE_vectors_not_collinear_l3748_374869


namespace NUMINAMATH_CALUDE_bird_stork_difference_l3748_374891

theorem bird_stork_difference (initial_birds storks additional_birds : ℕ) :
  initial_birds = 3 →
  storks = 4 →
  additional_birds = 2 →
  (initial_birds + additional_birds) - storks = 1 := by
  sorry

end NUMINAMATH_CALUDE_bird_stork_difference_l3748_374891


namespace NUMINAMATH_CALUDE_inequality_proof_l3748_374887

theorem inequality_proof (p q : ℝ) (m n : ℕ) 
  (h_pos_p : p > 0) (h_pos_q : q > 0) (h_sum : p + q = 1) (h_pos_m : m > 0) (h_pos_n : n > 0) : 
  (1 - p^m)^n + (1 - q^n)^m ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3748_374887


namespace NUMINAMATH_CALUDE_pokemon_cards_bought_l3748_374840

theorem pokemon_cards_bought (initial_cards final_cards : ℕ) 
  (h1 : initial_cards = 676)
  (h2 : final_cards = 900) :
  final_cards - initial_cards = 224 := by
  sorry

end NUMINAMATH_CALUDE_pokemon_cards_bought_l3748_374840


namespace NUMINAMATH_CALUDE_min_radios_sold_l3748_374874

/-- Proves the minimum value of n given the radio sales problem conditions -/
theorem min_radios_sold (n d₁ : ℕ) : 
  0 < n → 
  0 < d₁ → 
  d₁ % n = 0 → 
  10 * n - 30 = 80 → 
  ∀ m : ℕ, 0 < m → 10 * m - 30 = 80 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_min_radios_sold_l3748_374874


namespace NUMINAMATH_CALUDE_motion_equation_l3748_374876

/-- The acceleration function -/
def a (t : ℝ) : ℝ := 6 * t - 2

/-- The velocity function -/
def v (t : ℝ) : ℝ := 3 * t^2 - 2 * t + 1

/-- The position function -/
def s (t : ℝ) : ℝ := t^3 - t^2 + t

theorem motion_equation (t : ℝ) :
  (∀ t, deriv v t = a t) ∧
  (∀ t, deriv s t = v t) ∧
  v 0 = 1 ∧
  s 0 = 0 →
  s t = t^3 - t^2 + t :=
by
  sorry

end NUMINAMATH_CALUDE_motion_equation_l3748_374876


namespace NUMINAMATH_CALUDE_product_of_g_at_roots_of_f_l3748_374815

def f (y : ℝ) : ℝ := y^4 - y^3 + 2*y - 1

def g (y : ℝ) : ℝ := y^2 + y - 3

theorem product_of_g_at_roots_of_f :
  ∀ y₁ y₂ y₃ y₄ : ℝ,
  f y₁ = 0 → f y₂ = 0 → f y₃ = 0 → f y₄ = 0 →
  ∃ result : ℝ, g y₁ * g y₂ * g y₃ * g y₄ = result :=
by sorry

end NUMINAMATH_CALUDE_product_of_g_at_roots_of_f_l3748_374815


namespace NUMINAMATH_CALUDE_equal_weekend_days_count_l3748_374877

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Checks if starting the month on a given day results in equal Saturdays and Sundays -/
def equalWeekendDays (startDay : DayOfWeek) : Bool :=
  sorry

/-- Counts the number of days that result in equal Saturdays and Sundays when used as the start day -/
def countEqualWeekendDays : Nat :=
  sorry

theorem equal_weekend_days_count :
  countEqualWeekendDays = 2 :=
sorry

end NUMINAMATH_CALUDE_equal_weekend_days_count_l3748_374877


namespace NUMINAMATH_CALUDE_sum_of_first_10_terms_equals_560_l3748_374879

def arithmetic_sequence_1 (n : ℕ) : ℕ := 4 * n - 2
def arithmetic_sequence_2 (n : ℕ) : ℕ := 6 * n - 4

def common_sequence (n : ℕ) : ℕ := 12 * n - 10

def sum_of_first_n_terms (n : ℕ) : ℕ := n * (12 * n - 8) / 2

theorem sum_of_first_10_terms_equals_560 :
  sum_of_first_n_terms 10 = 560 := by sorry

end NUMINAMATH_CALUDE_sum_of_first_10_terms_equals_560_l3748_374879


namespace NUMINAMATH_CALUDE_cricket_team_right_handed_players_l3748_374875

theorem cricket_team_right_handed_players 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (h1 : total_players = 58)
  (h2 : throwers = 37)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 3 = 0) -- Ensures non-throwers can be divided into thirds
  : (throwers + (2 * (total_players - throwers) / 3)) = 51 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_right_handed_players_l3748_374875


namespace NUMINAMATH_CALUDE_phil_charlie_difference_l3748_374841

/-- Represents the number of games won by each player -/
structure GamesWon where
  perry : ℕ
  dana : ℕ
  charlie : ℕ
  phil : ℕ

/-- Conditions for the golf game results -/
def golf_conditions (g : GamesWon) : Prop :=
  g.perry = g.dana + 5 ∧
  g.charlie = g.dana - 2 ∧
  g.phil > g.charlie ∧
  g.phil = 12 ∧
  g.perry = g.phil + 4

/-- Theorem stating the difference between Phil's and Charlie's games -/
theorem phil_charlie_difference (g : GamesWon) (h : golf_conditions g) : 
  g.phil - g.charlie = 3 := by
  sorry

end NUMINAMATH_CALUDE_phil_charlie_difference_l3748_374841


namespace NUMINAMATH_CALUDE_susan_homework_time_l3748_374878

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Calculates the difference between two times in minutes -/
def timeDifference (t1 t2 : Time) : Nat :=
  (t2.hours * 60 + t2.minutes) - (t1.hours * 60 + t1.minutes)

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

theorem susan_homework_time : 
  let homeworkStart : Time := { hours := 13, minutes := 59 }
  let homeworkDuration : Nat := 96
  let practiceStart : Time := { hours := 16, minutes := 0 }
  let homeworkEnd := addMinutes homeworkStart homeworkDuration
  timeDifference homeworkEnd practiceStart = 25 := by
  sorry

end NUMINAMATH_CALUDE_susan_homework_time_l3748_374878


namespace NUMINAMATH_CALUDE_x_minus_y_value_l3748_374850

theorem x_minus_y_value (x y : ℝ) (h1 : x + y = 20) (h2 : x^2 - y^2 = 36) : x - y = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l3748_374850


namespace NUMINAMATH_CALUDE_gmat_scores_l3748_374842

theorem gmat_scores (x y z : ℝ) (h1 : x - y = 1/3) (h2 : z = (x + y) / 2) :
  y = x - 1/3 ∧ z = x - 1/6 := by
  sorry

end NUMINAMATH_CALUDE_gmat_scores_l3748_374842


namespace NUMINAMATH_CALUDE_unique_positive_integers_sum_l3748_374804

noncomputable def y : ℝ := Real.sqrt ((Real.sqrt 37) / 3 + 5 / 3)

theorem unique_positive_integers_sum (d e f : ℕ+) : 
  y^50 = 3*y^48 + 10*y^45 + 9*y^43 - y^25 + (d:ℝ)*y^21 + (e:ℝ)*y^19 + (f:ℝ)*y^15 →
  d + e + f = 119 := by sorry

end NUMINAMATH_CALUDE_unique_positive_integers_sum_l3748_374804


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_when_area_twice_perimeter_l3748_374805

/-- A triangle with an inscribed circle -/
structure Triangle :=
  (area : ℝ)
  (perimeter : ℝ)
  (inradius : ℝ)

/-- The theorem stating that for a triangle where the area is twice the perimeter, 
    the radius of the inscribed circle is 4 -/
theorem inscribed_circle_radius_when_area_twice_perimeter (t : Triangle) 
  (h : t.area = 2 * t.perimeter) : t.inradius = 4 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_when_area_twice_perimeter_l3748_374805


namespace NUMINAMATH_CALUDE_rectangular_garden_length_l3748_374847

theorem rectangular_garden_length 
  (perimeter : ℝ) 
  (breadth : ℝ) 
  (h1 : perimeter = 1200) 
  (h2 : breadth = 240) : 
  2 * (breadth + (perimeter / 2 - breadth)) = perimeter := by
  sorry

end NUMINAMATH_CALUDE_rectangular_garden_length_l3748_374847


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_over_four_l3748_374868

theorem tan_alpha_plus_pi_over_four (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - π / 4) = 1 / 4) :
  Real.tan (α + π / 4) = 3 / 22 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_over_four_l3748_374868


namespace NUMINAMATH_CALUDE_chipped_marbles_count_l3748_374845

def marble_counts : List Nat := [17, 20, 22, 24, 26, 35, 37, 40]

def total_marbles : Nat := marble_counts.sum

theorem chipped_marbles_count (jane_count george_count : Nat) 
  (h1 : jane_count = 3 * george_count)
  (h2 : jane_count + george_count = total_marbles - (marble_counts.get! 0 + marble_counts.get! 7))
  (h3 : ∃ (i j : Fin 8), i ≠ j ∧ 
    marble_counts.get! i.val + marble_counts.get! j.val = total_marbles - (jane_count + george_count) ∧
    (marble_counts.get! i.val = 40 ∨ marble_counts.get! j.val = 40)) :
  40 ∈ marble_counts ∧ 
  ∃ (i j : Fin 8), i ≠ j ∧ 
    marble_counts.get! i.val + marble_counts.get! j.val = total_marbles - (jane_count + george_count) ∧
    (marble_counts.get! i.val = 40 ∨ marble_counts.get! j.val = 40) :=
by sorry

end NUMINAMATH_CALUDE_chipped_marbles_count_l3748_374845


namespace NUMINAMATH_CALUDE_race_table_distance_l3748_374808

/-- Given a race with 11 equally spaced tables over 2100 meters, 
    the distance between the first and third table is 420 meters. -/
theorem race_table_distance (total_distance : ℝ) (num_tables : ℕ) :
  total_distance = 2100 →
  num_tables = 11 →
  (2 * (total_distance / (num_tables - 1))) = 420 := by
  sorry

end NUMINAMATH_CALUDE_race_table_distance_l3748_374808


namespace NUMINAMATH_CALUDE_eggs_per_box_l3748_374846

/-- Given that there are 6 eggs in 2 boxes and each box contains some eggs,
    prove that the number of eggs in each box is 3. -/
theorem eggs_per_box (total_eggs : ℕ) (num_boxes : ℕ) (eggs_per_box : ℕ) 
  (h1 : total_eggs = 6)
  (h2 : num_boxes = 2)
  (h3 : eggs_per_box * num_boxes = total_eggs)
  (h4 : eggs_per_box > 0) :
  eggs_per_box = 3 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_box_l3748_374846


namespace NUMINAMATH_CALUDE_avery_theorem_l3748_374818

/-- A shape with a certain number of 90-degree angles -/
structure Shape :=
  (angles : ℕ)

/-- A rectangular park is a shape with 4 90-degree angles -/
def rectangular_park : Shape :=
  ⟨4⟩

/-- Avery's visit to two places -/
structure AverysVisit :=
  (place1 : Shape)
  (place2 : Shape)
  (total_angles : ℕ)

/-- A rectangle or square is a shape with 4 90-degree angles -/
def rectangle_or_square (s : Shape) : Prop :=
  s.angles = 4

/-- The theorem to be proved -/
theorem avery_theorem (visit : AverysVisit) :
  visit.place1 = rectangular_park →
  visit.total_angles = 8 →
  rectangle_or_square visit.place2 :=
sorry

end NUMINAMATH_CALUDE_avery_theorem_l3748_374818


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3748_374854

theorem rectangle_perimeter (long_side : ℝ) (short_side_difference : ℝ) :
  long_side = 1 →
  short_side_difference = 2/8 →
  let short_side := long_side - short_side_difference
  2 * long_side + 2 * short_side = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3748_374854


namespace NUMINAMATH_CALUDE_equal_piles_coin_count_l3748_374836

theorem equal_piles_coin_count (total_coins : ℕ) (num_quarter_piles : ℕ) (num_dime_piles : ℕ) :
  total_coins = 42 →
  num_quarter_piles = 3 →
  num_dime_piles = 3 →
  ∃ (coins_per_pile : ℕ),
    total_coins = coins_per_pile * (num_quarter_piles + num_dime_piles) ∧
    coins_per_pile = 7 :=
by sorry

end NUMINAMATH_CALUDE_equal_piles_coin_count_l3748_374836


namespace NUMINAMATH_CALUDE_first_digit_of_y_in_base_9_l3748_374893

def base_3_num : List Nat := [1, 1, 2, 2, 0, 0, 2, 2, 1, 1, 0, 0, 2, 2, 1, 1, 2, 2, 2, 1]

def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

def y : Nat := to_base_10 base_3_num 3

def first_digit_base_9 (n : Nat) : Nat :=
  if n = 0 then 0 else
  let log_9 := Nat.log n 9
  (n / (9 ^ log_9)) % 9

theorem first_digit_of_y_in_base_9 :
  first_digit_base_9 y = 4 := by sorry

end NUMINAMATH_CALUDE_first_digit_of_y_in_base_9_l3748_374893


namespace NUMINAMATH_CALUDE_lines_cannot_form_triangle_l3748_374862

/-- A line in 2D space represented by ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if three lines intersect at a single point -/
def intersect_at_point (l1 l2 l3 : Line) : Prop :=
  let x := (l1.b * l3.c - l3.b * l1.c) / (l1.a * l3.b - l3.a * l1.b)
  let y := (l3.a * l1.c - l1.a * l3.c) / (l1.a * l3.b - l3.a * l1.b)
  l2.a * x + l2.b * y = l2.c

/-- The main theorem -/
theorem lines_cannot_form_triangle (m : ℝ) : 
  let l1 : Line := ⟨4, 1, 4⟩
  let l2 : Line := ⟨m, 1, 0⟩
  let l3 : Line := ⟨2, -3, 4⟩
  (parallel l1 l2 ∨ parallel l2 l3 ∨ intersect_at_point l1 l2 l3) →
  m = 4 ∨ m = 1/2 ∨ m = -2/3 :=
by sorry


end NUMINAMATH_CALUDE_lines_cannot_form_triangle_l3748_374862


namespace NUMINAMATH_CALUDE_white_stamp_price_is_20_cents_l3748_374825

/-- The price of a white stamp that satisfies the given conditions -/
def white_stamp_price : ℚ :=
  let red_stamps : ℕ := 30
  let white_stamps : ℕ := 80
  let red_stamp_price : ℚ := 1/2
  let sales_difference : ℚ := 1
  (sales_difference + red_stamps * red_stamp_price) / white_stamps

/-- Theorem stating that the white stamp price is 20 cents -/
theorem white_stamp_price_is_20_cents :
  white_stamp_price = 1/5 := by sorry

end NUMINAMATH_CALUDE_white_stamp_price_is_20_cents_l3748_374825


namespace NUMINAMATH_CALUDE_least_four_digit_11_heavy_l3748_374858

def is_11_heavy (n : ℕ) : Prop := n % 11 > 7

theorem least_four_digit_11_heavy : 
  (∀ m : ℕ, m ≥ 1000 ∧ m < 1000 → ¬(is_11_heavy m)) ∧ is_11_heavy 1000 :=
sorry

end NUMINAMATH_CALUDE_least_four_digit_11_heavy_l3748_374858


namespace NUMINAMATH_CALUDE_total_fish_caught_l3748_374880

/-- The number of times Chris goes fishing -/
def chris_trips : ℕ := 10

/-- The number of fish Brian catches per trip -/
def brian_fish_per_trip : ℕ := 400

/-- The ratio of Brian's fishing frequency to Chris's -/
def brian_frequency_ratio : ℚ := 2

/-- The fraction of fish Brian catches compared to Chris per trip -/
def brian_catch_fraction : ℚ := 3/5

theorem total_fish_caught :
  let brian_trips := chris_trips * brian_frequency_ratio
  let chris_fish_per_trip := brian_fish_per_trip / brian_catch_fraction
  let brian_total := brian_trips * brian_fish_per_trip
  let chris_total := chris_trips * chris_fish_per_trip.floor
  brian_total + chris_total = 14660 := by
sorry

end NUMINAMATH_CALUDE_total_fish_caught_l3748_374880


namespace NUMINAMATH_CALUDE_astrophysics_budget_decrease_l3748_374889

def current_year_allocations : List (String × Rat) :=
  [("Microphotonics", 14/100),
   ("Home Electronics", 24/100),
   ("Food Additives", 15/100),
   ("Genetically Modified Microorganisms", 19/100),
   ("Industrial Lubricants", 8/100)]

def previous_year_allocations : List (String × Rat) :=
  [("Microphotonics", 12/100),
   ("Home Electronics", 22/100),
   ("Food Additives", 13/100),
   ("Genetically Modified Microorganisms", 18/100),
   ("Industrial Lubricants", 7/100)]

def calculate_astrophysics_allocation (allocations : List (String × Rat)) : Rat :=
  1 - (allocations.map (fun x => x.2)).sum

def calculate_percentage_change (old_value : Rat) (new_value : Rat) : Rat :=
  (new_value - old_value) / old_value * 100

theorem astrophysics_budget_decrease :
  let current_astrophysics := calculate_astrophysics_allocation current_year_allocations
  let previous_astrophysics := calculate_astrophysics_allocation previous_year_allocations
  let percentage_change := calculate_percentage_change previous_astrophysics current_astrophysics
  percentage_change = -2857/100 := by sorry

end NUMINAMATH_CALUDE_astrophysics_budget_decrease_l3748_374889


namespace NUMINAMATH_CALUDE_divisors_of_square_l3748_374849

theorem divisors_of_square (n : ℕ) : 
  (∃ p : ℕ, Prime p ∧ n = p^3) → 
  (Finset.card (Nat.divisors n) = 4) → 
  (Finset.card (Nat.divisors (n^2)) = 7) := by
sorry

end NUMINAMATH_CALUDE_divisors_of_square_l3748_374849


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l3748_374801

theorem cubic_equation_solutions :
  ∀ m n : ℤ, (n^3 + m^3 + 231 = n^2 * m^2 + n * m) ↔ ((m = 4 ∧ n = 5) ∨ (m = 5 ∧ n = 4)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l3748_374801


namespace NUMINAMATH_CALUDE_ratio_of_300_to_2_l3748_374894

theorem ratio_of_300_to_2 : 
  let certain_number := 300
  300 / 2 = 150 := by sorry

end NUMINAMATH_CALUDE_ratio_of_300_to_2_l3748_374894


namespace NUMINAMATH_CALUDE_image_of_negative_four_two_l3748_374803

/-- The mapping f from R² to R² defined by f(x, y) = (xy, x + y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 * p.2, p.1 + p.2)

/-- Theorem stating that f(-4, 2) = (-8, -2) -/
theorem image_of_negative_four_two :
  f (-4, 2) = (-8, -2) := by
  sorry

end NUMINAMATH_CALUDE_image_of_negative_four_two_l3748_374803


namespace NUMINAMATH_CALUDE_angle_sum_equation_l3748_374883

theorem angle_sum_equation (α β : Real) (h : (1 + Real.sqrt 3 * Real.tan α) * (1 + Real.sqrt 3 * Real.tan β) = 4) :
  α + β = π / 3 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_equation_l3748_374883


namespace NUMINAMATH_CALUDE_midpoint_complex_coordinates_l3748_374855

theorem midpoint_complex_coordinates (A B C : ℂ) :
  A = 6 + 5*I ∧ B = -2 + 3*I ∧ C = (A + B) / 2 →
  C = 2 + 4*I :=
by sorry

end NUMINAMATH_CALUDE_midpoint_complex_coordinates_l3748_374855


namespace NUMINAMATH_CALUDE_cost_system_correct_l3748_374892

/-- Represents the cost of seedlings in yuan -/
def CostSystem (x y : ℝ) : Prop :=
  (4 * x + 3 * y = 180) ∧ (x - y = 10)

/-- The cost system correctly represents the seedling pricing scenario -/
theorem cost_system_correct (x y : ℝ) :
  (4 * x + 3 * y = 180) →
  (y = x - 10) →
  CostSystem x y :=
by sorry

end NUMINAMATH_CALUDE_cost_system_correct_l3748_374892


namespace NUMINAMATH_CALUDE_quadratic_inequality_all_reals_l3748_374821

theorem quadratic_inequality_all_reals
  (a b c : ℝ) :
  (∀ x, (a / 3) * x^2 + 2 * b * x - c < 0) ↔ (a > 0 ∧ 4 * b^2 - (4 / 3) * a * c < 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_all_reals_l3748_374821


namespace NUMINAMATH_CALUDE_pump_fill_time_l3748_374823

/-- The time it takes to fill the tank with the leak present -/
def fill_time_with_leak : ℝ := 3

/-- The time it takes for the leak to drain the full tank -/
def leak_drain_time : ℝ := 5.999999999999999

/-- The time it takes for the pump to fill the tank without the leak -/
def fill_time_without_leak : ℝ := 2

theorem pump_fill_time :
  (1 / fill_time_without_leak) - (1 / leak_drain_time) = (1 / fill_time_with_leak) :=
sorry

end NUMINAMATH_CALUDE_pump_fill_time_l3748_374823


namespace NUMINAMATH_CALUDE_minimum_red_marbles_l3748_374848

theorem minimum_red_marbles (r w g : ℕ) : 
  g ≥ (2 * w) / 3 →
  g ≤ r / 4 →
  w + g ≥ 72 →
  (∀ r' : ℕ, (∃ w' g' : ℕ, g' ≥ (2 * w') / 3 ∧ g' ≤ r' / 4 ∧ w' + g' ≥ 72) → r' ≥ r) →
  r = 120 := by
sorry

end NUMINAMATH_CALUDE_minimum_red_marbles_l3748_374848


namespace NUMINAMATH_CALUDE_thirty_percent_less_than_ninety_l3748_374812

theorem thirty_percent_less_than_ninety (x : ℝ) : x = 50.4 → x + (1/4 * x) = 90 - (30/100 * 90) := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_less_than_ninety_l3748_374812


namespace NUMINAMATH_CALUDE_overlapping_squares_areas_l3748_374867

/-- Represents the side lengths of three overlapping squares -/
structure SquareSides where
  largest : ℝ
  middle : ℝ
  smallest : ℝ

/-- Represents the areas of three overlapping squares -/
structure SquareAreas where
  largest : ℝ
  middle : ℝ
  smallest : ℝ

/-- Calculates the areas of three overlapping squares given their side lengths -/
def calculateAreas (sides : SquareSides) : SquareAreas :=
  { largest := sides.largest ^ 2,
    middle := sides.middle ^ 2,
    smallest := sides.smallest ^ 2 }

/-- Theorem stating the areas of three overlapping squares given specific conditions -/
theorem overlapping_squares_areas :
  ∀ (sides : SquareSides),
    sides.largest = sides.middle + 1 →
    sides.largest = sides.smallest + 2 →
    (sides.largest - 1) * (sides.middle - 1) = 100 →
    (sides.middle - 1) * (sides.smallest - 1) = 64 →
    calculateAreas sides = { largest := 361, middle := 324, smallest := 289 } := by
  sorry


end NUMINAMATH_CALUDE_overlapping_squares_areas_l3748_374867


namespace NUMINAMATH_CALUDE_sum_of_numbers_l3748_374807

theorem sum_of_numbers (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 16) 
  (h4 : 1 / x = 3 * (1 / y)) : x + y = 16 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l3748_374807
