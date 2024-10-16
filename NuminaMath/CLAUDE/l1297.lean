import Mathlib

namespace NUMINAMATH_CALUDE_male_contestants_count_l1297_129744

theorem male_contestants_count (total : ℕ) (female_ratio : ℚ) : 
  total = 18 → female_ratio = 1/3 → (total : ℚ) * (1 - female_ratio) = 12 := by
  sorry

end NUMINAMATH_CALUDE_male_contestants_count_l1297_129744


namespace NUMINAMATH_CALUDE_ball_radius_from_hole_dimensions_l1297_129710

/-- Given a spherical ball partially submerged in a frozen surface,
    where the hole left by the ball has a diameter of 30 cm and a depth of 8 cm,
    prove that the radius of the ball is 18.0625 cm. -/
theorem ball_radius_from_hole_dimensions (diameter : ℝ) (depth : ℝ) (radius : ℝ) :
  diameter = 30 →
  depth = 8 →
  radius = (((diameter / 2) ^ 2 + depth ^ 2) / (2 * depth)).sqrt →
  radius = 18.0625 := by
sorry

end NUMINAMATH_CALUDE_ball_radius_from_hole_dimensions_l1297_129710


namespace NUMINAMATH_CALUDE_vector_on_line_l1297_129780

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

/-- Two distinct vectors p and q define a line. The vector (3/5)*p + (2/5)*q lies on that line. -/
theorem vector_on_line (p q : V) (h : p ≠ q) :
  ∃ t : ℝ, (3/5 : ℝ) • p + (2/5 : ℝ) • q = p + t • (q - p) := by
  sorry

end NUMINAMATH_CALUDE_vector_on_line_l1297_129780


namespace NUMINAMATH_CALUDE_readers_all_three_genres_l1297_129770

/-- Represents the number of readers for each genre and their intersections --/
structure ReaderCounts where
  total : ℕ
  sf : ℕ
  lw : ℕ
  hf : ℕ
  sf_lw : ℕ
  sf_hf : ℕ
  lw_hf : ℕ

/-- The principle of inclusion-exclusion for three sets --/
def inclusionExclusion (r : ReaderCounts) (x : ℕ) : Prop :=
  r.total = r.sf + r.lw + r.hf - r.sf_lw - r.sf_hf - r.lw_hf + x

/-- The theorem stating the number of readers who read all three genres --/
theorem readers_all_three_genres (r : ReaderCounts) 
  (h_total : r.total = 800)
  (h_sf : r.sf = 300)
  (h_lw : r.lw = 600)
  (h_hf : r.hf = 400)
  (h_sf_lw : r.sf_lw = 175)
  (h_sf_hf : r.sf_hf = 150)
  (h_lw_hf : r.lw_hf = 250) :
  ∃ x, inclusionExclusion r x ∧ x = 75 := by
  sorry

end NUMINAMATH_CALUDE_readers_all_three_genres_l1297_129770


namespace NUMINAMATH_CALUDE_minimum_occupied_seats_l1297_129791

/-- Represents a seating arrangement in a cinema row. -/
structure CinemaRow where
  total_seats : ℕ
  occupied_seats : ℕ

/-- Checks if a seating arrangement ensures the next person sits adjacent to someone. -/
def is_valid_arrangement (row : CinemaRow) : Prop :=
  ∀ i : ℕ, i < row.total_seats → 
    ∃ j : ℕ, j < row.total_seats ∧ 
      (j = i - 1 ∨ j = i + 1) ∧ 
      (∃ k : ℕ, k < row.occupied_seats ∧ j = k * 3)

/-- The theorem to be proved. -/
theorem minimum_occupied_seats :
  ∃ (row : CinemaRow), 
    row.total_seats = 150 ∧ 
    row.occupied_seats = 50 ∧ 
    is_valid_arrangement row ∧
    (∀ (other : CinemaRow), 
      other.total_seats = 150 → 
      is_valid_arrangement other → 
      other.occupied_seats ≥ 50) := by
  sorry

end NUMINAMATH_CALUDE_minimum_occupied_seats_l1297_129791


namespace NUMINAMATH_CALUDE_fraction_irreducible_l1297_129729

theorem fraction_irreducible (n m : ℕ) : 
  Nat.gcd (m * (n + 1) + 1) (m * (n + 1) - n) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l1297_129729


namespace NUMINAMATH_CALUDE_pond_draining_time_l1297_129784

/-- The time taken by the first pump to drain one-half of the pond -/
def first_pump_time : ℝ := 5

/-- The time taken by the second pump to drain the entire pond alone -/
def second_pump_time : ℝ := 1.1111111111111112

/-- The time taken by both pumps to drain the remaining half of the pond -/
def combined_time : ℝ := 0.5

theorem pond_draining_time : 
  (1 / (2 * first_pump_time) + 1 / second_pump_time) * combined_time = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_pond_draining_time_l1297_129784


namespace NUMINAMATH_CALUDE_triangle_determinant_l1297_129718

theorem triangle_determinant (A B C : Real) (h1 : A = 45 * π / 180)
    (h2 : B = 75 * π / 180) (h3 : C = 60 * π / 180) :
  let M : Matrix (Fin 3) (Fin 3) Real :=
    ![![Real.tan A, 1, 1],
      ![1, Real.tan B, 1],
      ![1, 1, Real.tan C]]
  Matrix.det M = -1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_determinant_l1297_129718


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1297_129769

-- Define a geometric sequence
def isGeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Define the arithmetic sequence condition
def arithmeticSequenceCondition (a : ℕ → ℝ) : Prop :=
  a 2 + a 1 = a 3

-- Main theorem
theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geometric : isGeometricSequence a q)
  (h_positive : ∀ n : ℕ, a n > 0)
  (h_arithmetic : arithmeticSequenceCondition a) :
  q = (Real.sqrt 5 + 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1297_129769


namespace NUMINAMATH_CALUDE_sector_area_l1297_129757

theorem sector_area (arc_length : ℝ) (central_angle : ℝ) (h1 : arc_length = 6) (h2 : central_angle = 2) :
  let radius : ℝ := arc_length / central_angle
  let area : ℝ := (1/2) * arc_length * radius
  area = 9 := by sorry

end NUMINAMATH_CALUDE_sector_area_l1297_129757


namespace NUMINAMATH_CALUDE_mean_of_numbers_l1297_129702

def numbers : List ℝ := [13, 8, 13, 21, 7, 23]

theorem mean_of_numbers : (numbers.sum / numbers.length : ℝ) = 14.1666667 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_numbers_l1297_129702


namespace NUMINAMATH_CALUDE_inequality_proof_l1297_129727

theorem inequality_proof (a b c : ℝ) (h : a + b + c = 3) :
  1 / (5 * a^2 - 4 * a + 11) + 1 / (5 * b^2 - 4 * b + 11) + 1 / (5 * c^2 - 4 * c + 11) ≤ 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1297_129727


namespace NUMINAMATH_CALUDE_prob_no_adjacent_standing_ten_people_l1297_129734

/-- Represents the number of valid arrangements for n people where no two adjacent people are standing. -/
def validArrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | n + 3 => validArrangements (n + 1) + validArrangements (n + 2)

/-- The number of people seated around the table. -/
def numPeople : ℕ := 10

/-- The total number of possible outcomes when flipping n fair coins. -/
def totalOutcomes (n : ℕ) : ℕ := 2^n

/-- The probability of no two adjacent people standing when n people flip fair coins. -/
def noAdjacentStandingProb (n : ℕ) : ℚ :=
  validArrangements n / totalOutcomes n

theorem prob_no_adjacent_standing_ten_people :
  noAdjacentStandingProb numPeople = 123 / 1024 := by
  sorry

#eval noAdjacentStandingProb numPeople

end NUMINAMATH_CALUDE_prob_no_adjacent_standing_ten_people_l1297_129734


namespace NUMINAMATH_CALUDE_fifth_term_is_five_l1297_129733

def fibonacci_like_sequence : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fibonacci_like_sequence n + fibonacci_like_sequence (n + 1)

theorem fifth_term_is_five :
  fibonacci_like_sequence 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_five_l1297_129733


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l1297_129725

theorem min_value_expression (x y z : ℝ) 
  (hx : -2 < x ∧ x < 2) (hy : -2 < y ∧ y < 2) (hz : -2 < z ∧ z < 2) :
  (1 / ((1 - x^2) * (1 - y^2) * (1 - z^2))) + (1 / ((1 + x^2) * (1 + y^2) * (1 + z^2))) ≥ 2 :=
sorry

theorem min_value_achieved (x y z : ℝ) :
  (1 / ((1 - x^2) * (1 - y^2) * (1 - z^2))) + (1 / ((1 + x^2) * (1 + y^2) * (1 + z^2))) = 2 ↔ x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l1297_129725


namespace NUMINAMATH_CALUDE_ac_squared_gt_bc_squared_implies_a_gt_b_l1297_129723

theorem ac_squared_gt_bc_squared_implies_a_gt_b (a b c : ℝ) :
  a * c^2 > b * c^2 → a > b := by sorry

end NUMINAMATH_CALUDE_ac_squared_gt_bc_squared_implies_a_gt_b_l1297_129723


namespace NUMINAMATH_CALUDE_granola_bars_pack_count_granola_bars_pack_count_is_20_l1297_129760

theorem granola_bars_pack_count : ℕ → Prop :=
  fun total_bars : ℕ =>
    let bars_for_week : ℕ := 7
    let bars_traded : ℕ := 3
    let sisters : ℕ := 2
    let bars_per_sister : ℕ := 5
    total_bars = bars_for_week + bars_traded + (sisters * bars_per_sister)

theorem granola_bars_pack_count_is_20 : granola_bars_pack_count 20 := by
  sorry

end NUMINAMATH_CALUDE_granola_bars_pack_count_granola_bars_pack_count_is_20_l1297_129760


namespace NUMINAMATH_CALUDE_number_of_girls_in_class_l1297_129799

theorem number_of_girls_in_class (total_students : ℕ) (girls_ratio : ℚ) : 
  total_students = 35 →
  girls_ratio = 0.4 →
  ∃ (boys girls : ℕ), 
    boys + girls = total_students ∧ 
    girls = (girls_ratio * boys).floor ∧
    girls = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_of_girls_in_class_l1297_129799


namespace NUMINAMATH_CALUDE_infinitely_many_n_satisfying_conditions_l1297_129785

theorem infinitely_many_n_satisfying_conditions :
  ∀ k : ℕ, k > 0 →
  let n := k * (k + 1)
  ∃ m : ℕ, m^2 < n ∧ n < (m + 1)^2 ∧ n % m = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_n_satisfying_conditions_l1297_129785


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l1297_129796

/-- The diagonal length of a rectangle with sides 30√3 cm and 30 cm is 60 cm. -/
theorem rectangle_diagonal : ℝ → Prop :=
  fun diagonal =>
    let side1 := 30 * Real.sqrt 3
    let side2 := 30
    diagonal ^ 2 = side1 ^ 2 + side2 ^ 2 →
    diagonal = 60

-- The proof is omitted
axiom rectangle_diagonal_proof : rectangle_diagonal 60

#check rectangle_diagonal_proof

end NUMINAMATH_CALUDE_rectangle_diagonal_l1297_129796


namespace NUMINAMATH_CALUDE_pta_spending_ratio_l1297_129797

/-- Proves the ratio of money spent on food for faculty to the amount left after buying school supplies -/
theorem pta_spending_ratio (initial_savings : ℚ) (school_supplies_fraction : ℚ) (final_amount : ℚ)
  (h1 : initial_savings = 400)
  (h2 : school_supplies_fraction = 1/4)
  (h3 : final_amount = 150)
  : (initial_savings * (1 - school_supplies_fraction) - final_amount) / 
    (initial_savings * (1 - school_supplies_fraction)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_pta_spending_ratio_l1297_129797


namespace NUMINAMATH_CALUDE_circle_line_segments_l1297_129703

/-- The number of line segments formed by joining each pair of n distinct points on a circle -/
def lineSegments (n : ℕ) : ℕ := n.choose 2

/-- There are 8 distinct points on a circle -/
def numPoints : ℕ := 8

theorem circle_line_segments :
  lineSegments numPoints = 28 := by
  sorry

end NUMINAMATH_CALUDE_circle_line_segments_l1297_129703


namespace NUMINAMATH_CALUDE_fraction_addition_l1297_129750

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l1297_129750


namespace NUMINAMATH_CALUDE_production_time_reduction_l1297_129794

/-- Represents the time taken to complete a production order given a number of machines -/
def completion_time (num_machines : ℕ) (base_time : ℕ) : ℚ :=
  (num_machines * base_time : ℚ) / num_machines

theorem production_time_reduction :
  let base_machines := 3
  let base_time := 44
  let new_machines := 4
  (completion_time base_machines base_time - completion_time new_machines base_time : ℚ) = 11 := by
  sorry

end NUMINAMATH_CALUDE_production_time_reduction_l1297_129794


namespace NUMINAMATH_CALUDE_unique_pair_l1297_129758

def is_valid_pair (a b : ℕ) : Prop :=
  a ≥ 60 ∧ a < 70 ∧ b ≥ 60 ∧ b < 70 ∧ 
  a % 10 ≠ 6 ∧ b % 10 ≠ 6 ∧
  a * b = (10 * (a % 10) + 6) * (10 * (b % 10) + 6)

theorem unique_pair : 
  ∀ a b : ℕ, is_valid_pair a b → ((a = 69 ∧ b = 64) ∨ (a = 64 ∧ b = 69)) :=
by sorry

end NUMINAMATH_CALUDE_unique_pair_l1297_129758


namespace NUMINAMATH_CALUDE_solutions_equality_l1297_129778

-- Define a as a positive real number
variable (a : ℝ) (ha : a > 0)

-- Define the condition that 10 < a^x < 100 has exactly five solutions in natural numbers
def has_five_solutions (a : ℝ) : Prop :=
  (∃ (s : Finset ℕ), s.card = 5 ∧ ∀ x : ℕ, x ∈ s ↔ (10 < a^x ∧ a^x < 100))

-- Theorem statement
theorem solutions_equality (h : has_five_solutions a) :
  ∃ (s : Finset ℕ), s.card = 5 ∧ ∀ x : ℕ, x ∈ s ↔ (100 < a^x ∧ a^x < 1000) :=
sorry

end NUMINAMATH_CALUDE_solutions_equality_l1297_129778


namespace NUMINAMATH_CALUDE_total_cement_is_15_point_1_l1297_129748

/-- The amount of cement (in tons) used for Lexi's street -/
def lexis_street_cement : ℝ := 10

/-- The amount of cement (in tons) used for Tess's street -/
def tess_street_cement : ℝ := 5.1

/-- The total amount of cement used by Roadster's Paving Company -/
def total_cement : ℝ := lexis_street_cement + tess_street_cement

/-- Theorem stating that the total cement used is 15.1 tons -/
theorem total_cement_is_15_point_1 : total_cement = 15.1 := by sorry

end NUMINAMATH_CALUDE_total_cement_is_15_point_1_l1297_129748


namespace NUMINAMATH_CALUDE_negative_square_cubed_l1297_129706

theorem negative_square_cubed (a : ℝ) : (-a^2)^3 = -a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_cubed_l1297_129706


namespace NUMINAMATH_CALUDE_joes_age_l1297_129747

theorem joes_age (B J E : ℕ) : 
  B = 3 * J →                  -- Billy's age is three times Joe's age
  E = (B + J) / 2 →            -- Emily's age is the average of Billy's and Joe's ages
  B + J + E = 90 →             -- The sum of their ages is 90
  J = 15 :=                    -- Joe's age is 15
by sorry

end NUMINAMATH_CALUDE_joes_age_l1297_129747


namespace NUMINAMATH_CALUDE_finite_fun_primes_l1297_129731

/-- A prime p is fun with respect to positive integers a and b if there exists a positive integer n
    satisfying the given conditions. -/
def IsFunPrime (p a b : ℕ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ 
    p.Prime ∧
    (p ∣ a^(n.factorial) + b) ∧
    (p ∣ a^((n+1).factorial) + b) ∧
    (p < 2*n^2 + 1)

/-- The set of fun primes for given positive integers a and b is finite. -/
theorem finite_fun_primes (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  {p : ℕ | IsFunPrime p a b}.Finite :=
sorry

end NUMINAMATH_CALUDE_finite_fun_primes_l1297_129731


namespace NUMINAMATH_CALUDE_train_length_l1297_129788

/-- The length of a train given its crossing times over a bridge and a lamp post -/
theorem train_length (bridge_length : ℝ) (bridge_time : ℝ) (lamp_time : ℝ)
  (h1 : bridge_length = 150)
  (h2 : bridge_time = 7.5)
  (h3 : lamp_time = 2.5)
  (h4 : bridge_time > 0)
  (h5 : lamp_time > 0) :
  ∃ (train_length : ℝ),
    train_length = 75 ∧
    (train_length + bridge_length) / bridge_time = train_length / lamp_time :=
by sorry

end NUMINAMATH_CALUDE_train_length_l1297_129788


namespace NUMINAMATH_CALUDE_lisas_age_2005_l1297_129754

theorem lisas_age_2005 (lisa_age_2000 grandfather_age_2000 : ℕ) 
  (h1 : lisa_age_2000 * 2 = grandfather_age_2000)
  (h2 : (2000 - lisa_age_2000) + (2000 - grandfather_age_2000) = 3904) :
  lisa_age_2000 + 5 = 37 := by
  sorry

end NUMINAMATH_CALUDE_lisas_age_2005_l1297_129754


namespace NUMINAMATH_CALUDE_average_trees_is_36_l1297_129765

/-- The number of trees planted by class A -/
def trees_A : ℕ := 35

/-- The number of trees planted by class B -/
def trees_B : ℕ := trees_A + 6

/-- The number of trees planted by class C -/
def trees_C : ℕ := trees_A - 3

/-- The average number of trees planted by the three classes -/
def average_trees : ℚ := (trees_A + trees_B + trees_C) / 3

theorem average_trees_is_36 : average_trees = 36 := by
  sorry

end NUMINAMATH_CALUDE_average_trees_is_36_l1297_129765


namespace NUMINAMATH_CALUDE_abc_inequality_l1297_129781

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1297_129781


namespace NUMINAMATH_CALUDE_max_value_implies_t_equals_one_l1297_129704

def f (t : ℝ) (x : ℝ) : ℝ := |x^2 - 2*x - t|

theorem max_value_implies_t_equals_one (t : ℝ) :
  (∀ x ∈ Set.Icc 0 3, f t x ≤ 2) ∧
  (∃ x ∈ Set.Icc 0 3, f t x = 2) →
  t = 1 := by
sorry

end NUMINAMATH_CALUDE_max_value_implies_t_equals_one_l1297_129704


namespace NUMINAMATH_CALUDE_count_integers_eq_1278_l1297_129715

/-- Recursive function to calculate the number of n-digit sequences with no consecutive 1's -/
def a : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | (n + 2) => a (n + 1) + a n

/-- The number of 12-digit positive integers with all digits either 1 or 2 and exactly two consecutive 1's -/
def count_integers : ℕ := 2 * a 10 + 9 * (2 * a 9)

/-- Theorem stating that the count of such integers is 1278 -/
theorem count_integers_eq_1278 : count_integers = 1278 := by sorry

end NUMINAMATH_CALUDE_count_integers_eq_1278_l1297_129715


namespace NUMINAMATH_CALUDE_range_of_a_l1297_129798

theorem range_of_a (a b c : ℝ) 
  (sum_eq : a + b + c = 2)
  (sum_sq_eq : a^2 + b^2 + c^2 = 4)
  (order : a > b ∧ b > c) :
  a ∈ Set.Ioo (2/3) 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1297_129798


namespace NUMINAMATH_CALUDE_intersection_M_N_l1297_129722

def M : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 2}
def N : Set ℝ := {0, 1, 2}

theorem intersection_M_N :
  M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1297_129722


namespace NUMINAMATH_CALUDE_cube_averaging_solution_l1297_129728

/-- Represents a cube with real numbers on its vertices -/
structure Cube where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  F : ℝ
  G : ℝ
  H : ℝ

/-- Checks if the cube satisfies the averaging condition -/
def satisfiesAveraging (c : Cube) : Prop :=
  (c.D + c.E + c.B) / 3 = 6 ∧
  (c.A + c.F + c.C) / 3 = 3 ∧
  (c.D + c.G + c.B) / 3 = 6 ∧
  (c.A + c.C + c.H) / 3 = 4 ∧
  (c.A + c.H + c.F) / 3 = 3 ∧
  (c.E + c.G + c.B) / 3 = 6 ∧
  (c.H + c.F + c.C) / 3 = 5 ∧
  (c.D + c.G + c.E) / 3 = 3

/-- The theorem stating that the given solution is the only one satisfying the averaging condition -/
theorem cube_averaging_solution :
  ∀ c : Cube, satisfiesAveraging c →
    c.A = 0 ∧ c.B = 12 ∧ c.C = 6 ∧ c.D = 3 ∧ c.E = 3 ∧ c.F = 3 ∧ c.G = 3 ∧ c.H = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_averaging_solution_l1297_129728


namespace NUMINAMATH_CALUDE_unique_solution_l1297_129709

def base7_to_decimal (a b : Nat) : Nat := 7 * a + b

theorem unique_solution :
  ∀ P Q R : Nat,
    P ≠ 0 ∧ Q ≠ 0 ∧ R ≠ 0 →
    P < 7 ∧ Q < 7 ∧ R < 7 →
    P ≠ Q ∧ P ≠ R ∧ Q ≠ R →
    base7_to_decimal P Q + R = base7_to_decimal R 0 →
    base7_to_decimal P Q + base7_to_decimal Q P = base7_to_decimal R R →
    P = 4 ∧ Q = 3 ∧ R = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1297_129709


namespace NUMINAMATH_CALUDE_charity_plates_delivered_l1297_129782

/-- The number of plates delivered by a charity given the cost of ingredients and total spent -/
theorem charity_plates_delivered (rice_cost chicken_cost total_spent : ℚ) : 
  rice_cost = 1/10 →
  chicken_cost = 4/10 →
  total_spent = 50 →
  (total_spent / (rice_cost + chicken_cost) : ℚ) = 100 := by
  sorry

end NUMINAMATH_CALUDE_charity_plates_delivered_l1297_129782


namespace NUMINAMATH_CALUDE_product_c_remaining_amount_l1297_129736

/-- Calculate the remaining amount to be paid for a product -/
def remaining_amount (cost deposit discount_rate tax_rate : ℝ) : ℝ :=
  let discounted_price := cost * (1 - discount_rate)
  let total_price := discounted_price * (1 + tax_rate)
  total_price - deposit

/-- Theorem: The remaining amount to be paid for Product C is $3,610 -/
theorem product_c_remaining_amount :
  remaining_amount 3800 380 0 0.05 = 3610 := by
  sorry

end NUMINAMATH_CALUDE_product_c_remaining_amount_l1297_129736


namespace NUMINAMATH_CALUDE_arccos_one_half_equals_pi_third_l1297_129741

theorem arccos_one_half_equals_pi_third : Real.arccos (1/2) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_half_equals_pi_third_l1297_129741


namespace NUMINAMATH_CALUDE_inverse_f_sum_squares_l1297_129787

-- Define the function f
def f (x : ℝ) : ℝ := x * |x|

-- State the theorem
theorem inverse_f_sum_squares : 
  (∃ y₁ y₂ : ℝ, f y₁ = 9 ∧ f y₂ = -49) → 
  (∃ y₁ y₂ : ℝ, f y₁ = 9 ∧ f y₂ = -49 ∧ y₁^2 + y₂^2 = 58) := by
sorry

end NUMINAMATH_CALUDE_inverse_f_sum_squares_l1297_129787


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l1297_129700

theorem gcd_lcm_product (a b : ℕ) (h1 : Nat.gcd a b = 12) (h2 : Nat.lcm a b = 168) :
  a * b = 2016 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l1297_129700


namespace NUMINAMATH_CALUDE_chad_savings_l1297_129753

/-- Represents Chad's financial situation for a year --/
structure ChadFinances where
  savingRate : ℝ
  mowingIncome : ℝ
  birthdayMoney : ℝ
  videoGamesSales : ℝ
  oddJobsIncome : ℝ

/-- Calculates Chad's total savings for the year --/
def totalSavings (cf : ChadFinances) : ℝ :=
  cf.savingRate * (cf.mowingIncome + cf.birthdayMoney + cf.videoGamesSales + cf.oddJobsIncome)

/-- Theorem stating that Chad's savings for the year will be $460 --/
theorem chad_savings :
  ∀ (cf : ChadFinances),
    cf.savingRate = 0.4 ∧
    cf.mowingIncome = 600 ∧
    cf.birthdayMoney = 250 ∧
    cf.videoGamesSales = 150 ∧
    cf.oddJobsIncome = 150 →
    totalSavings cf = 460 :=
by sorry

end NUMINAMATH_CALUDE_chad_savings_l1297_129753


namespace NUMINAMATH_CALUDE_train_length_l1297_129761

/-- The length of a train given its speed, time to cross a bridge, and the bridge's length -/
theorem train_length (v : ℝ) (t : ℝ) (bridge_length : ℝ) (h1 : v = 36) (h2 : t = 27.997760179185665) (h3 : bridge_length = 150) :
  v * (1000 / 3600) * t - bridge_length = 129.97760179185665 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1297_129761


namespace NUMINAMATH_CALUDE_not_perfect_square_floor_sqrt_l1297_129775

theorem not_perfect_square_floor_sqrt (A : ℕ) (h : ∀ k : ℕ, k * k ≠ A) :
  ∃ n : ℕ, A = ⌊(n : ℝ) + Real.sqrt n + 1/2⌋ :=
sorry

end NUMINAMATH_CALUDE_not_perfect_square_floor_sqrt_l1297_129775


namespace NUMINAMATH_CALUDE_deductive_reasoning_correctness_l1297_129767

/-- Represents a deductive reasoning process -/
structure DeductiveReasoning where
  premise : Prop
  form : Prop
  conclusion : Prop

/-- Represents the correctness of a component in the reasoning process -/
def isCorrect (p : Prop) : Prop := p

theorem deductive_reasoning_correctness 
  (dr : DeductiveReasoning) 
  (h_premise : isCorrect dr.premise) 
  (h_form : isCorrect dr.form) : 
  isCorrect dr.conclusion :=
sorry

end NUMINAMATH_CALUDE_deductive_reasoning_correctness_l1297_129767


namespace NUMINAMATH_CALUDE_power_of_one_seventh_l1297_129732

def is_greatest_power_of_2_factor (x : ℕ) : Prop :=
  2^x ∣ 180 ∧ ∀ k > x, ¬(2^k ∣ 180)

def is_greatest_power_of_3_factor (y : ℕ) : Prop :=
  3^y ∣ 180 ∧ ∀ k > y, ¬(3^k ∣ 180)

theorem power_of_one_seventh (x y : ℕ) 
  (h2 : is_greatest_power_of_2_factor x) 
  (h3 : is_greatest_power_of_3_factor y) : 
  (1/7 : ℚ)^(y - x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_one_seventh_l1297_129732


namespace NUMINAMATH_CALUDE_hyperbola_equation_and_minimum_distance_l1297_129762

structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

def on_hyperbola (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

def asymptotic_equation (h : Hyperbola) : Prop :=
  h.b = Real.sqrt 3 * h.a

def point_on_hyperbola (h : Hyperbola) : Prop :=
  on_hyperbola h (Real.sqrt 5) (Real.sqrt 3)

def perpendicular_vectors (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

theorem hyperbola_equation_and_minimum_distance 
  (h : Hyperbola) 
  (h_asymptotic : asymptotic_equation h)
  (h_point : point_on_hyperbola h) :
  (h.a = 2 ∧ h.b = 2 * Real.sqrt 3) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 
    on_hyperbola h x₁ y₁ → 
    on_hyperbola h x₂ y₂ → 
    perpendicular_vectors x₁ y₁ x₂ y₂ → 
    x₁^2 + y₁^2 + x₂^2 + y₂^2 ≥ 24) :=
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_and_minimum_distance_l1297_129762


namespace NUMINAMATH_CALUDE_line_equation_l1297_129792

/-- Given two parallel lines l₁ and l₂, prove that the line passing through H(-1, 1) with its
    midpoint M lying on x - y - 1 = 0 has the equation x + y = 0. -/
theorem line_equation (A B C₁ C₂ : ℝ) (h₁ : C₁ ≠ C₂) (h₂ : A - B + C₁ + C₂ = 0) :
  ∃ (l : ℝ → ℝ → Prop),
    (∀ x y, l x y ↔ x + y = 0) ∧
    l (-1) 1 ∧
    ∃ (M : ℝ × ℝ),
      (M.1 - M.2 - 1 = 0) ∧
      (∃ (t : ℝ), 
        A * (t * (-1) + (1 - t) * M.1) + B * (t * 1 + (1 - t) * M.2) + C₁ = 0 ∧
        A * (t * (-1) + (1 - t) * M.1) + B * (t * 1 + (1 - t) * M.2) + C₂ = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_l1297_129792


namespace NUMINAMATH_CALUDE_power_three_minus_two_plus_three_l1297_129773

theorem power_three_minus_two_plus_three : 2^3 - 2 + 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_power_three_minus_two_plus_three_l1297_129773


namespace NUMINAMATH_CALUDE_multiplication_value_proof_l1297_129740

theorem multiplication_value_proof : 
  let number : ℝ := 5.5
  let divisor : ℝ := 6
  let result : ℝ := 11
  let multiplier : ℝ := 12
  (number / divisor) * multiplier = result :=
by sorry

end NUMINAMATH_CALUDE_multiplication_value_proof_l1297_129740


namespace NUMINAMATH_CALUDE_xy_value_l1297_129730

theorem xy_value (x y : ℝ) : |x - y + 6| + (y + 8)^2 = 0 → x * y = 112 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1297_129730


namespace NUMINAMATH_CALUDE_true_compound_proposition_l1297_129766

-- Define the propositions
def p : Prop := ∀ x : ℝ, x < 0 → x^3 < 0
def q : Prop := ∀ x : ℝ, x > 0 → Real.log x < 0

-- Theorem to prove
theorem true_compound_proposition : (¬p) ∨ (¬q) := by
  sorry

end NUMINAMATH_CALUDE_true_compound_proposition_l1297_129766


namespace NUMINAMATH_CALUDE_triangle_inequality_l1297_129739

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1297_129739


namespace NUMINAMATH_CALUDE_unmarked_trees_l1297_129768

def total_trees : ℕ := 200
def mark_interval_out : ℕ := 5
def mark_interval_back : ℕ := 8

theorem unmarked_trees :
  let marked_out := total_trees / mark_interval_out
  let marked_back := total_trees / mark_interval_back
  let overlap := total_trees / (mark_interval_out * mark_interval_back)
  let total_marked := marked_out + marked_back - overlap
  total_trees - total_marked = 140 := by
  sorry

end NUMINAMATH_CALUDE_unmarked_trees_l1297_129768


namespace NUMINAMATH_CALUDE_savings_difference_is_250_l1297_129749

def window_price : ℕ := 125
def offer_purchase : ℕ := 6
def offer_free : ℕ := 2
def dave_windows : ℕ := 9
def doug_windows : ℕ := 11

def calculate_cost (num_windows : ℕ) : ℕ :=
  let sets := num_windows / (offer_purchase + offer_free)
  let remainder := num_windows % (offer_purchase + offer_free)
  (sets * offer_purchase + min remainder offer_purchase) * window_price

def savings_difference : ℕ :=
  let separate_cost := calculate_cost dave_windows + calculate_cost doug_windows
  let combined_cost := calculate_cost (dave_windows + doug_windows)
  let separate_savings := dave_windows * window_price + doug_windows * window_price - separate_cost
  let combined_savings := (dave_windows + doug_windows) * window_price - combined_cost
  combined_savings - separate_savings

theorem savings_difference_is_250 : savings_difference = 250 := by
  sorry

end NUMINAMATH_CALUDE_savings_difference_is_250_l1297_129749


namespace NUMINAMATH_CALUDE_sum_of_quotients_divisible_by_nine_l1297_129752

theorem sum_of_quotients_divisible_by_nine (n : ℕ) (hn : n > 8) :
  let a : ℕ → ℕ := λ i => (10^(2*i) - 1) / 9
  let q : ℕ → ℕ := λ i => a i / 11
  let s : ℕ → ℕ := λ i => (Finset.range 9).sum (λ j => q (i + j))
  ∀ i : ℕ, i ≤ n - 8 → (s i) % 9 = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_quotients_divisible_by_nine_l1297_129752


namespace NUMINAMATH_CALUDE_gcd_of_specific_powers_of_two_l1297_129779

theorem gcd_of_specific_powers_of_two : Nat.gcd (2^2048 - 1) (2^2035 - 1) = 8191 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_specific_powers_of_two_l1297_129779


namespace NUMINAMATH_CALUDE_inequality_proof_l1297_129795

theorem inequality_proof (a b c k : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hk : k ≥ 1) :
  (a^(k+1) / b^k : ℚ) + (b^(k+1) / c^k : ℚ) + (c^(k+1) / a^k : ℚ) ≥ 
  (a^k / b^(k-1) : ℚ) + (b^k / c^(k-1) : ℚ) + (c^k / a^(k-1) : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1297_129795


namespace NUMINAMATH_CALUDE_inequality_conditions_l1297_129708

theorem inequality_conditions (A B C : ℝ) :
  (∀ x y z : ℝ, A * (x - y) * (x - z) + B * (y - z) * (y - x) + C * (z - x) * (z - y) ≥ 0) ↔
  (A ≥ 0 ∧ B ≥ 0 ∧ C ≥ 0 ∧ A^2 + B^2 + C^2 ≤ 2*(A*B + B*C + C*A)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_conditions_l1297_129708


namespace NUMINAMATH_CALUDE_sanxingdui_jinsha_visitor_l1297_129735

/-- Represents the four people in the problem -/
inductive Person : Type
  | A | B | C | D

/-- Represents the two archaeological sites -/
inductive Site : Type
  | Sanxingdui
  | Jinsha

/-- Predicate to represent if a person visited a site -/
def visited (p : Person) (s : Site) : Prop := sorry

/-- Predicate to represent if a person is telling the truth -/
def telling_truth (p : Person) : Prop := sorry

theorem sanxingdui_jinsha_visitor :
  (∃! p : Person, ∀ s : Site, visited p s) →
  (∃! p : Person, ¬telling_truth p) →
  (¬visited Person.A Site.Sanxingdui ∧ ¬visited Person.A Site.Jinsha) →
  (visited Person.B Site.Sanxingdui ↔ visited Person.A Site.Sanxingdui) →
  (visited Person.C Site.Jinsha ↔ visited Person.B Site.Jinsha) →
  (∀ s : Site, visited Person.D s → ¬visited Person.B s) →
  (∀ s : Site, visited Person.C s) :=
sorry

end NUMINAMATH_CALUDE_sanxingdui_jinsha_visitor_l1297_129735


namespace NUMINAMATH_CALUDE_abs_neg_seven_l1297_129790

theorem abs_neg_seven : |(-7 : ℤ)| = 7 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_seven_l1297_129790


namespace NUMINAMATH_CALUDE_black_region_area_is_56_l1297_129763

/-- The area of the region between two squares, where a smaller square is entirely contained 
    within a larger square. -/
def black_region_area (small_side : ℝ) (large_side : ℝ) : ℝ :=
  large_side ^ 2 - small_side ^ 2

/-- Theorem stating that the area of the black region between two squares with given side lengths
    is 56 square units. -/
theorem black_region_area_is_56 :
  black_region_area 5 9 = 56 := by
  sorry

end NUMINAMATH_CALUDE_black_region_area_is_56_l1297_129763


namespace NUMINAMATH_CALUDE_exists_vertical_line_through_point_l1297_129793

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  slope : Option ℝ
  yIntercept : ℝ

-- Define a function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  match l.slope with
  | some k => p.y = k * p.x + l.yIntercept
  | none => p.x = l.yIntercept

-- Theorem statement
theorem exists_vertical_line_through_point (b : ℝ) :
  ∃ (l : Line2D), pointOnLine ⟨0, b⟩ l ∧ l.slope = none :=
sorry

end NUMINAMATH_CALUDE_exists_vertical_line_through_point_l1297_129793


namespace NUMINAMATH_CALUDE_power_boat_travel_time_l1297_129717

/-- Represents the scenario of a power boat and raft traveling on a river --/
structure RiverJourney where
  r : ℝ  -- Speed of the river current (km/h)
  p : ℝ  -- Speed of the power boat relative to the river (km/h)
  t : ℝ  -- Time for power boat to travel from A to B (hours)
  s : ℝ  -- Stopping time at dock B (hours)

/-- The theorem stating that the time for the power boat to travel from A to B is 5 hours --/
theorem power_boat_travel_time 
  (journey : RiverJourney) 
  (h1 : journey.r > 0)  -- River speed is positive
  (h2 : journey.p > journey.r)  -- Power boat is faster than river current
  (h3 : journey.s = 1)  -- Stopping time is 1 hour
  (h4 : (journey.p + journey.r) * journey.t + (journey.p - journey.r) * (12 - journey.t - journey.s) = 12 * journey.r)  -- Distance equation
  : journey.t = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_boat_travel_time_l1297_129717


namespace NUMINAMATH_CALUDE_coin_split_sum_l1297_129713

/-- Represents the sum of recorded products when splitting coins into piles -/
def recordedSum (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that for 25 coins, the sum of recorded products is 300 -/
theorem coin_split_sum :
  recordedSum 25 = 300 := by
  sorry

end NUMINAMATH_CALUDE_coin_split_sum_l1297_129713


namespace NUMINAMATH_CALUDE_equation_equivalence_l1297_129712

theorem equation_equivalence (a c x y : ℝ) (s t u : ℤ) : 
  (a^8 * x * y - a^7 * y - a^6 * x = a^5 * (c^5 - 1)) →
  ((a^s * x - a^t) * (a^u * y - a^3) = a^5 * c^5) →
  s * t * u = 18 := by
sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1297_129712


namespace NUMINAMATH_CALUDE_four_students_in_all_activities_l1297_129789

/-- The number of students participating in all three activities in a summer camp. -/
def students_in_all_activities (total_students : ℕ) 
  (swimming_students : ℕ) (archery_students : ℕ) (chess_students : ℕ) 
  (at_least_two_activities : ℕ) : ℕ :=
  let a := swimming_students + archery_students + chess_students - at_least_two_activities - total_students
  a

/-- Theorem stating that 4 students participate in all three activities. -/
theorem four_students_in_all_activities : 
  students_in_all_activities 25 15 17 10 12 = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_students_in_all_activities_l1297_129789


namespace NUMINAMATH_CALUDE_p_or_q_false_sufficient_not_necessary_l1297_129714

-- Define propositions p and q
variable (p q : Prop)

-- Define the statement "p or q is false"
def p_or_q_false : Prop := ¬(p ∨ q)

-- Define the statement "not p is true"
def not_p_true : Prop := ¬p

-- Theorem stating that "p or q is false" is sufficient but not necessary for "not p is true"
theorem p_or_q_false_sufficient_not_necessary :
  (p_or_q_false p q → not_p_true p) ∧
  ¬(not_p_true p → p_or_q_false p q) :=
sorry

end NUMINAMATH_CALUDE_p_or_q_false_sufficient_not_necessary_l1297_129714


namespace NUMINAMATH_CALUDE_remainder_problem_l1297_129746

theorem remainder_problem (d : ℕ) (r : ℕ) (h1 : d > 1) 
  (h2 : 1059 % d = r)
  (h3 : 1482 % d = r)
  (h4 : 2340 % d = r) :
  2 * d - r = 6 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1297_129746


namespace NUMINAMATH_CALUDE_sum_of_numbers_with_lcm_and_ratio_l1297_129719

/-- Given three positive integers a, b, and c in the ratio 2:3:5 with LCM 120, their sum is 40 -/
theorem sum_of_numbers_with_lcm_and_ratio (a b c : ℕ+) : 
  (a : ℕ) + b + c = 40 ∧ 
  Nat.lcm a (Nat.lcm b c) = 120 ∧ 
  3 * a = 2 * b ∧ 
  5 * a = 2 * c := by
sorry


end NUMINAMATH_CALUDE_sum_of_numbers_with_lcm_and_ratio_l1297_129719


namespace NUMINAMATH_CALUDE_max_value_of_trig_function_l1297_129726

theorem max_value_of_trig_function :
  let f (x : ℝ) := Real.tan (x + 3 * Real.pi / 4) - Real.tan x + Real.sin (x + Real.pi / 4)
  ∀ x ∈ Set.Icc (-3 * Real.pi / 4) (-Real.pi / 2),
    f x ≤ 0 ∧ ∃ x₀ ∈ Set.Icc (-3 * Real.pi / 4) (-Real.pi / 2), f x₀ = 0 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_trig_function_l1297_129726


namespace NUMINAMATH_CALUDE_sol_earnings_l1297_129716

def candy_sales (day : Nat) : Nat :=
  10 + 4 * (day - 1)

def total_sales : Nat :=
  (List.range 6).map (λ i => candy_sales (i + 1)) |>.sum

def earnings_cents : Nat :=
  total_sales * 10

theorem sol_earnings :
  earnings_cents / 100 = 12 := by sorry

end NUMINAMATH_CALUDE_sol_earnings_l1297_129716


namespace NUMINAMATH_CALUDE_min_difference_l1297_129776

/-- Represents a 4-digit positive integer ABCD -/
def FourDigitNum (a b c d : Nat) : Nat :=
  1000 * a + 100 * b + 10 * c + d

/-- Represents a 2-digit positive integer -/
def TwoDigitNum (x y : Nat) : Nat :=
  10 * x + y

/-- The difference between a 4-digit number and the product of its two 2-digit parts -/
def Difference (a b c d : Nat) : Nat :=
  FourDigitNum a b c d - TwoDigitNum a b * TwoDigitNum c d

theorem min_difference :
  ∀ (a b c d : Nat),
    a ≠ 0 → c ≠ 0 →
    a < 10 → b < 10 → c < 10 → d < 10 →
    Difference a b c d ≥ 109 :=
by sorry

end NUMINAMATH_CALUDE_min_difference_l1297_129776


namespace NUMINAMATH_CALUDE_smallest_b_value_l1297_129737

theorem smallest_b_value (a b : ℕ+) (h1 : a.val - b.val = 8) 
  (h2 : Nat.gcd ((a.val^3 + b.val^3) / (a.val + b.val)) (a.val * b.val) = 16) :
  ∀ k : ℕ+, k.val < b.val → ¬(∃ a' : ℕ+, a'.val - k.val = 8 ∧ 
    Nat.gcd ((a'.val^3 + k.val^3) / (a'.val + k.val)) (a'.val * k.val) = 16) :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l1297_129737


namespace NUMINAMATH_CALUDE_cistern_width_l1297_129738

/-- Calculates the width of a rectangular cistern given its length, depth, and total wet surface area. -/
theorem cistern_width (length depth area : ℝ) (h1 : length = 5) (h2 : depth = 1.25) (h3 : area = 42.5) :
  ∃ width : ℝ, width = 4 ∧ 
  area = length * width + 2 * (depth * length) + 2 * (depth * width) :=
by sorry

end NUMINAMATH_CALUDE_cistern_width_l1297_129738


namespace NUMINAMATH_CALUDE_quiz_score_average_l1297_129764

theorem quiz_score_average (n : ℕ) (initial_avg : ℚ) (dropped_score : ℚ) : 
  n = 16 → 
  initial_avg = 62.5 → 
  dropped_score = 55 → 
  let total_score := n * initial_avg
  let remaining_total := total_score - dropped_score
  let new_avg := remaining_total / (n - 1)
  new_avg = 63 := by sorry

end NUMINAMATH_CALUDE_quiz_score_average_l1297_129764


namespace NUMINAMATH_CALUDE_max_balls_l1297_129774

theorem max_balls (n : ℕ) : 
  (∃ r : ℕ, r ≤ n ∧ 
    (r ≥ 49 ∧ r ≤ 50) ∧ 
    (∀ k : ℕ, k > 0 → (7 * k ≤ r - 49) ∧ (r - 49 < 8 * k)) ∧
    (10 * r ≥ 9 * n)) → 
  n ≤ 210 :=
sorry

end NUMINAMATH_CALUDE_max_balls_l1297_129774


namespace NUMINAMATH_CALUDE_not_both_odd_l1297_129771

theorem not_both_odd (m n : ℕ) (h : (1 : ℚ) / m + (1 : ℚ) / n = (1 : ℚ) / 2020) :
  Even m ∨ Even n :=
sorry

end NUMINAMATH_CALUDE_not_both_odd_l1297_129771


namespace NUMINAMATH_CALUDE_maple_leaf_picking_l1297_129772

theorem maple_leaf_picking (elder_points younger_points : ℕ) 
  (h1 : elder_points = 5)
  (h2 : younger_points = 3)
  (h3 : ∃ (x y : ℕ), elder_points * x + younger_points * y = 102 ∧ x = y + 6) :
  ∃ (x y : ℕ), x = 15 ∧ y = 9 ∧ 
    elder_points * x + younger_points * y = 102 ∧ x = y + 6 := by
  sorry

end NUMINAMATH_CALUDE_maple_leaf_picking_l1297_129772


namespace NUMINAMATH_CALUDE_total_time_calculation_l1297_129745

/-- Calculates the total time to complete an assignment and clean sticky keys. -/
theorem total_time_calculation (assignment_time : ℕ) (num_keys : ℕ) (time_per_key : ℕ) :
  assignment_time = 10 ∧ num_keys = 14 ∧ time_per_key = 3 →
  assignment_time + num_keys * time_per_key = 52 := by
  sorry

end NUMINAMATH_CALUDE_total_time_calculation_l1297_129745


namespace NUMINAMATH_CALUDE_kolya_role_is_collection_agency_l1297_129777

-- Define the actors in the scenario
inductive Actor : Type
| Katya : Actor
| Vasya : Actor
| Kolya : Actor

-- Define the possible roles
inductive Role : Type
| FinancialPyramid : Role
| CollectionAgency : Role
| Bank : Role
| InsuranceCompany : Role

-- Define the scenario
structure BookLendingScenario where
  lender : Actor
  borrower : Actor
  mediator : Actor
  books_lent : ℕ
  return_period : ℕ
  books_not_returned : Bool
  mediator_reward : ℕ

-- Define the characteristics of a collection agency
def is_collection_agency (r : Role) : Prop :=
  r = Role.CollectionAgency

-- Define the function to determine the role based on the scenario
def determine_role (s : BookLendingScenario) : Role :=
  Role.CollectionAgency

-- Theorem statement
theorem kolya_role_is_collection_agency (s : BookLendingScenario) :
  s.lender = Actor.Katya ∧
  s.borrower = Actor.Vasya ∧
  s.mediator = Actor.Kolya ∧
  s.books_lent > 0 ∧
  s.return_period > 0 ∧
  s.books_not_returned = true ∧
  s.mediator_reward > 0 →
  is_collection_agency (determine_role s) :=
sorry

end NUMINAMATH_CALUDE_kolya_role_is_collection_agency_l1297_129777


namespace NUMINAMATH_CALUDE_cube_root_of_negative_eight_l1297_129705

theorem cube_root_of_negative_eight (x : ℝ) : x^3 = -8 ↔ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_eight_l1297_129705


namespace NUMINAMATH_CALUDE_volume_submerged_object_iron_block_volume_l1297_129711

/-- The volume of a submerged object in a cylindrical container --/
theorem volume_submerged_object
  (r h₁ h₂ : ℝ)
  (hr : r > 0)
  (hh : h₂ > h₁) :
  let V := π * r^2 * (h₂ - h₁)
  V = π * r^2 * h₂ - π * r^2 * h₁ :=
by sorry

/-- The volume of the irregular iron block --/
theorem iron_block_volume
  (r h₁ h₂ : ℝ)
  (hr : r = 5)
  (hh₁ : h₁ = 6)
  (hh₂ : h₂ = 8) :
  π * r^2 * (h₂ - h₁) = 50 * π :=
by sorry

end NUMINAMATH_CALUDE_volume_submerged_object_iron_block_volume_l1297_129711


namespace NUMINAMATH_CALUDE_max_circle_area_in_square_l1297_129720

/-- The maximum possible area of a circle fitting inside a square with given measurements -/
theorem max_circle_area_in_square (square_side : ℝ) (error : ℝ) : 
  square_side = 5 → error = 0.2 → 
  ∃ (max_area : ℝ), max_area = π * ((square_side + error) / 2)^2 ∧ 
  ∀ (area : ℝ), (∃ (r : ℝ), r ≤ (square_side + error) / 2 ∧ area = π * r^2) → area ≤ max_area :=
by sorry

end NUMINAMATH_CALUDE_max_circle_area_in_square_l1297_129720


namespace NUMINAMATH_CALUDE_right_triangle_area_l1297_129756

theorem right_triangle_area (a b c : ℝ) (h1 : a = 12) (h2 : c = 13) (h3 : a^2 + b^2 = c^2) : 
  (1/2) * a * b = 30 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1297_129756


namespace NUMINAMATH_CALUDE_triangle_inequality_expression_negative_l1297_129724

/-- Given a triangle with side lengths a, b, and c, 
    the expression a^2 - c^2 - 2ab + b^2 is always negative. -/
theorem triangle_inequality_expression_negative 
  (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a^2 - c^2 - 2*a*b + b^2 < 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_expression_negative_l1297_129724


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l1297_129742

theorem closest_integer_to_cube_root : 
  ∃ (n : ℤ), n = 7 ∧ 
  ∀ (m : ℤ), |n - (5^3 + 7^3)^(1/3)| ≤ |m - (5^3 + 7^3)^(1/3)| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l1297_129742


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_times_two_l1297_129721

/-- The sum of two repeating decimals multiplied by 2 -/
theorem repeating_decimal_sum_times_two :
  2 * ((5 : ℚ) / 9 + (7 : ℚ) / 9) = (8 : ℚ) / 3 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_times_two_l1297_129721


namespace NUMINAMATH_CALUDE_counterexample_37_l1297_129751

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem counterexample_37 : 
  is_prime 37 ∧ ¬(is_prime (37 - 2) ∨ is_prime (37 + 2)) :=
by sorry

end NUMINAMATH_CALUDE_counterexample_37_l1297_129751


namespace NUMINAMATH_CALUDE_students_playing_both_sports_l1297_129743

theorem students_playing_both_sports (total : ℕ) (football : ℕ) (cricket : ℕ) (neither : ℕ) :
  total = 250 →
  football = 160 →
  cricket = 90 →
  neither = 50 →
  (total - neither) = (football + cricket - (football + cricket - (total - neither))) :=
by sorry

end NUMINAMATH_CALUDE_students_playing_both_sports_l1297_129743


namespace NUMINAMATH_CALUDE_power_two_plus_one_div_by_three_l1297_129786

theorem power_two_plus_one_div_by_three (n : ℕ) :
  n > 0 → (3 ∣ 2^n + 1 ↔ n % 2 = 1) := by sorry

end NUMINAMATH_CALUDE_power_two_plus_one_div_by_three_l1297_129786


namespace NUMINAMATH_CALUDE_darnel_sprint_distance_l1297_129701

theorem darnel_sprint_distance (jogged_distance : Real) (additional_sprint : Real) :
  jogged_distance = 0.75 →
  additional_sprint = 0.125 →
  jogged_distance + additional_sprint = 0.875 := by
  sorry

end NUMINAMATH_CALUDE_darnel_sprint_distance_l1297_129701


namespace NUMINAMATH_CALUDE_combined_savings_equal_separate_savings_l1297_129759

/-- Represents the store's window offer -/
structure WindowOffer where
  price : ℕ  -- Price per window
  buy : ℕ    -- Number of windows to buy
  free : ℕ   -- Number of free windows

/-- Calculates the cost for a given number of windows under the offer -/
def calculateCost (offer : WindowOffer) (windowsNeeded : ℕ) : ℕ :=
  let groups := windowsNeeded / (offer.buy + offer.free)
  let remainder := windowsNeeded % (offer.buy + offer.free)
  (groups * offer.buy + min remainder offer.buy) * offer.price

/-- Calculates the savings for a given number of windows under the offer -/
def calculateSavings (offer : WindowOffer) (windowsNeeded : ℕ) : ℕ :=
  windowsNeeded * offer.price - calculateCost offer windowsNeeded

theorem combined_savings_equal_separate_savings 
  (offer : WindowOffer)
  (davesWindows : ℕ)
  (dougsWindows : ℕ)
  (h1 : offer.price = 150)
  (h2 : offer.buy = 8)
  (h3 : offer.free = 2)
  (h4 : davesWindows = 10)
  (h5 : dougsWindows = 16) :
  calculateSavings offer (davesWindows + dougsWindows) = 
  calculateSavings offer davesWindows + calculateSavings offer dougsWindows :=
by sorry

end NUMINAMATH_CALUDE_combined_savings_equal_separate_savings_l1297_129759


namespace NUMINAMATH_CALUDE_percentage_sum_theorem_l1297_129707

theorem percentage_sum_theorem (X Y : ℝ) 
  (hX : 0.45 * X = 270) 
  (hY : 0.35 * Y = 210) : 
  0.75 * X + 0.55 * Y = 780 := by
sorry

end NUMINAMATH_CALUDE_percentage_sum_theorem_l1297_129707


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angles_equal_l1297_129755

/-- An isosceles triangle is a triangle with at least two equal sides -/
structure IsoscelesTriangle where
  side_a : ℝ
  side_b : ℝ
  side_c : ℝ
  is_isosceles : side_a = side_b ∨ side_b = side_c ∨ side_c = side_a

/-- The two base angles of an isosceles triangle are equal -/
theorem isosceles_triangle_base_angles_equal (t : IsoscelesTriangle) :
  ∃ (angle1 angle2 : ℝ), angle1 = angle2 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angles_equal_l1297_129755


namespace NUMINAMATH_CALUDE_balance_equals_132_l1297_129783

/-- Calculates the account balance after two years given an initial deposit,
    annual interest rate, and additional annual deposit. -/
def account_balance_after_two_years (initial_deposit : ℝ) (interest_rate : ℝ) (annual_deposit : ℝ) : ℝ :=
  let balance_after_first_year := initial_deposit * (1 + interest_rate) + annual_deposit
  balance_after_first_year * (1 + interest_rate) + annual_deposit

/-- Theorem stating that given the specified conditions, the account balance
    after two years will be $132. -/
theorem balance_equals_132 :
  account_balance_after_two_years 100 0.1 10 = 132 := by
  sorry

#eval account_balance_after_two_years 100 0.1 10

end NUMINAMATH_CALUDE_balance_equals_132_l1297_129783
