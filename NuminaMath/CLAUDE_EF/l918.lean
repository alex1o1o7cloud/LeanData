import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_one_l918_91882

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 0 then -Real.sqrt 2 * Real.sin x
  else if 0 < x ∧ x ≤ 1 then Real.tan (Real.pi/4 * x)
  else 0  -- Default value for x outside the defined range

-- Theorem statement
theorem f_composition_equals_one :
  f (f (-Real.pi/4)) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_one_l918_91882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l918_91871

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x - Real.cos x

-- State the theorem
theorem range_of_f :
  (∀ y ∈ Set.Icc (-2 : ℝ) (Real.sqrt 3),
    ∃ x ∈ Set.Icc (-Real.pi/2) (Real.pi/2),
      f x = y) ∧
  (∀ x ∈ Set.Icc (-Real.pi/2) (Real.pi/2),
    -2 ≤ f x ∧ f x ≤ Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l918_91871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_n_l918_91832

theorem no_valid_n : ¬∃ (n : ℕ), 
  (let total_matches := (n + 1) * (2 * n + 3);
  -- Total matches is divisible by 5
  total_matches % 5 = 0 ∧
  -- Number of matches won by women and men are integers
  ∃ (x : ℕ), total_matches = 5 * x ∧
  -- n is one of the given options
  (n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 8)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_n_l918_91832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_l918_91859

theorem line_slope (x y : ℝ) :
  (Real.sqrt 3 * x - y + 1 = 0) → (∃ k : ℝ, k = Real.sqrt 3 ∧ y = k * x - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_l918_91859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_contribution_charity_fund_problem_l918_91896

/-- Represents a charity fund with initial contributions and a new contribution --/
structure CharityFund where
  initial_contributions : ℕ -- number of initial contributions
  initial_average : ℚ -- average contribution before the new donation
  new_average : ℚ -- average contribution after the new donation
  new_contribution : ℚ -- the amount of the new contribution

/-- Calculates the new average contribution after adding a new donation --/
def new_average (fund : CharityFund) : ℚ :=
  (fund.initial_contributions * fund.initial_average + fund.new_contribution) / 
  (fund.initial_contributions + 1)

/-- Theorem stating that given the conditions, John's contribution must be $125 --/
theorem johns_contribution (fund : CharityFund) 
  (h1 : fund.initial_contributions = 2)
  (h2 : fund.new_average = 75)
  (h3 : fund.new_average = (3 : ℚ) / 2 * fund.initial_average) : 
  fund.new_contribution = 125 := by
  sorry

/-- Main theorem combining all conditions and the result --/
theorem charity_fund_problem : ∃ (fund : CharityFund),
  fund.initial_contributions = 2 ∧
  fund.new_average = 75 ∧
  fund.new_average = (3 : ℚ) / 2 * fund.initial_average ∧
  fund.new_contribution = 125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_contribution_charity_fund_problem_l918_91896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_power_x_equals_one_eighth_of_two_power_36_l918_91898

theorem eight_power_x_equals_one_eighth_of_two_power_36 (x : ℝ) : 
  (1 / 8 : ℝ) * (2 : ℝ) ^ 36 = (8 : ℝ) ^ x → x = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_power_x_equals_one_eighth_of_two_power_36_l918_91898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_four_white_balls_l918_91845

def total_balls : ℕ := 14
def white_balls : ℕ := 6
def black_balls : ℕ := 8
def drawn_balls : ℕ := 4

def probability_all_white : ℚ := 15 / 1001

theorem probability_four_white_balls :
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = probability_all_white :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_four_white_balls_l918_91845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_matrix_power_18_l918_91878

/-- The rotation matrix for an angle θ in radians -/
noncomputable def rotation_matrix (θ : Real) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ],
    ![Real.sin θ, Real.cos θ]]

/-- The identity matrix -/
def identity_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 0],
    ![0, 1]]

/-- Theorem stating that 18 is the smallest positive integer n such that
    the 140° rotation matrix raised to the power n equals the identity matrix -/
theorem rotation_matrix_power_18 :
  (∀ k : ℕ, k > 0 ∧ k < 18 → (rotation_matrix (140 * Real.pi / 180)) ^ k ≠ identity_matrix) ∧
  (rotation_matrix (140 * Real.pi / 180)) ^ 18 = identity_matrix :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_matrix_power_18_l918_91878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_estimate_l918_91834

/-- The growth rate of the population over 15 years -/
def growth_rate : ℝ := 2

/-- The initial year for the population data -/
def initial_year : ℕ := 2020

/-- The target year for population estimation -/
def target_year : ℕ := 2045

/-- The initial population of Island X -/
def initial_population : ℕ := 500

/-- The number of years between the initial and target year -/
def years_passed : ℕ := target_year - initial_year

/-- The estimated population of Island X in the target year -/
noncomputable def estimated_population : ℝ :=
  (initial_population : ℝ) * growth_rate ^ ((years_passed : ℝ) / 15)

/-- Theorem stating that the estimated population in 2045 is approximately 1600 -/
theorem population_estimate :
  ⌊estimated_population⌋ = 1600 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_estimate_l918_91834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_O_mass_percentage_approx_l918_91824

/-- Molar mass of Carbon in g/mol -/
noncomputable def C_molar_mass : ℝ := 12.01

/-- Molar mass of Hydrogen in g/mol -/
noncomputable def H_molar_mass : ℝ := 1.008

/-- Molar mass of Nitrogen in g/mol -/
noncomputable def N_molar_mass : ℝ := 14.01

/-- Molar mass of Oxygen in g/mol -/
noncomputable def O_molar_mass : ℝ := 16.00

/-- Number of Carbon atoms in the molecule -/
def C_count : ℕ := 20

/-- Number of Hydrogen atoms in the molecule -/
def H_count : ℕ := 25

/-- Number of Nitrogen atoms in the molecule -/
def N_count : ℕ := 3

/-- Number of Oxygen atoms in the molecule -/
def O_count : ℕ := 1

/-- Calculate the total molar mass of the molecule -/
noncomputable def total_molar_mass : ℝ :=
  C_molar_mass * (C_count : ℝ) +
  H_molar_mass * (H_count : ℝ) +
  N_molar_mass * (N_count : ℝ) +
  O_molar_mass * (O_count : ℝ)

/-- Calculate the mass percentage of Oxygen in the molecule -/
noncomputable def O_mass_percentage : ℝ :=
  (O_molar_mass * (O_count : ℝ) / total_molar_mass) * 100

/-- Theorem: The mass percentage of Oxygen in C20H25N3O is approximately 4.95% -/
theorem O_mass_percentage_approx :
  abs (O_mass_percentage - 4.95) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_O_mass_percentage_approx_l918_91824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_b_fills_in_8_minutes_l918_91829

/-- Represents the time taken by Pipe B to fill one-third of the cistern -/
def pipe_b_time (pipe_a_time : ℝ) (fill_time : ℝ) : ℝ :=
  -- The actual calculation is not provided here, as per the instructions
  sorry

/-- Theorem stating the time taken by Pipe B to fill one-third of the cistern -/
theorem pipe_b_fills_in_8_minutes :
  let pipe_a_time : ℝ := 12
  let fill_time : ℝ := 14.4
  pipe_b_time pipe_a_time fill_time = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_b_fills_in_8_minutes_l918_91829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_non_uniform_partition_l918_91821

/-- Represents a cube with integer edge length -/
structure Cube where
  edge : ℕ

/-- The volume of a cube -/
def Cube.volume (c : Cube) : ℕ := c.edge ^ 3

/-- A partition of a larger cube into smaller cubes -/
structure CubePartition where
  originalCube : Cube
  smallerCubes : List Cube

/-- Check if a partition is valid -/
def CubePartition.isValid (p : CubePartition) : Prop :=
  (∀ c ∈ p.smallerCubes, c.edge > 0 ∧ c.edge ≤ p.originalCube.edge) ∧
  (p.smallerCubes.map Cube.volume).sum = p.originalCube.volume

/-- Check if a partition has non-uniform sized cubes -/
def CubePartition.hasNonUniformSizes (p : CubePartition) : Prop :=
  ∃ c1 c2, c1 ∈ p.smallerCubes ∧ c2 ∈ p.smallerCubes ∧ c1.edge ≠ c2.edge

/-- The main theorem -/
theorem max_non_uniform_partition :
  ∃ (p : CubePartition),
    p.originalCube.edge = 4 ∧
    p.isValid ∧
    p.hasNonUniformSizes ∧
    p.smallerCubes.length = 57 ∧
    (∀ (q : CubePartition),
      q.originalCube.edge = 4 →
      q.isValid →
      q.hasNonUniformSizes →
      q.smallerCubes.length ≤ 57) := by
  sorry

#check max_non_uniform_partition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_non_uniform_partition_l918_91821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moment_formula_moment_nonexistent_l918_91855

/-- The probability density function of the random variable X -/
noncomputable def p (x : ℝ) : ℝ :=
  if x ≤ 1 then 0 else 5 / x^6

/-- The k-th moment of the random variable X -/
noncomputable def moment (k : ℝ) : ℝ := ∫ x in Set.Ioi 1, x^k * p x

/-- Theorem: The k-th moment of X is 5 / (5 - k) for k < 5 -/
theorem moment_formula {k : ℝ} (hk : k < 5) :
  moment k = 5 / (5 - k) := by sorry

/-- Theorem: The k-th moment of X does not exist for k ≥ 5 -/
theorem moment_nonexistent {k : ℝ} (hk : k ≥ 5) :
  ¬ ∃ (m : ℝ), moment k = m := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_moment_formula_moment_nonexistent_l918_91855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l918_91870

/-- An arithmetic sequence with common difference d < 0 -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_d_neg : d < 0
  h_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * seq.a 1 + (n - 1) * seq.d)

theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence)
  (h_geometric_mean : (3 * Real.sqrt 5)^2 = (-seq.a 2) * (seq.a 9))
  (h_sum_10 : sum_n seq 10 = 20) :
  seq.d = -39/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l918_91870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l918_91866

/-- The function f(x) = 1 + log₁₀(x) + 9/log₁₀(x) -/
noncomputable def f (x : ℝ) : ℝ := 1 + Real.log x / Real.log 10 + 9 / (Real.log x / Real.log 10)

/-- Theorem stating that the maximum value of f(x) is -5 for 0 < x < 1 -/
theorem f_max_value : ∀ x : ℝ, 0 < x → x < 1 → f x ≤ -5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l918_91866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_male_employees_distribution_l918_91879

/-- Represents the age brackets for male employees -/
inductive AgeBracket
  | Bracket18To29
  | Bracket30To39
  | Bracket40To49
  | Bracket50To59
  | Bracket60Plus

/-- Represents the corporation's employee statistics -/
structure CorporationStats where
  totalEmployees : ℕ
  malePercentage : ℚ
  ageBracketPercentages : AgeBracket → ℚ

/-- Calculates the number of male employees in a given age bracket -/
def malesToBracket (stats : CorporationStats) (bracket : AgeBracket) : ℕ :=
  Int.toNat ⌊(stats.totalEmployees : ℚ) * stats.malePercentage * stats.ageBracketPercentages bracket⌋

/-- Theorem stating that the sum of male employees in all brackets equals the total male employees -/
theorem male_employees_distribution (stats : CorporationStats) 
    (h1 : stats.totalEmployees = 4200)
    (h2 : stats.malePercentage = 35/100)
    (h3 : stats.ageBracketPercentages AgeBracket.Bracket18To29 = 10/100)
    (h4 : stats.ageBracketPercentages AgeBracket.Bracket30To39 = 25/100)
    (h5 : stats.ageBracketPercentages AgeBracket.Bracket40To49 = 35/100)
    (h6 : stats.ageBracketPercentages AgeBracket.Bracket50To59 = 20/100)
    (h7 : stats.ageBracketPercentages AgeBracket.Bracket60Plus = 10/100) :
    malesToBracket stats AgeBracket.Bracket18To29 = 147 ∧
    malesToBracket stats AgeBracket.Bracket30To39 = 367 ∧
    malesToBracket stats AgeBracket.Bracket40To49 = 515 ∧
    malesToBracket stats AgeBracket.Bracket50To59 = 294 ∧
    malesToBracket stats AgeBracket.Bracket60Plus = 147 :=
  by sorry

#eval malesToBracket 
  { totalEmployees := 4200
    malePercentage := 35/100
    ageBracketPercentages := fun
      | AgeBracket.Bracket18To29 => 10/100
      | AgeBracket.Bracket30To39 => 25/100
      | AgeBracket.Bracket40To49 => 35/100
      | AgeBracket.Bracket50To59 => 20/100
      | AgeBracket.Bracket60Plus => 10/100 }
  AgeBracket.Bracket18To29

end NUMINAMATH_CALUDE_ERRORFEEDBACK_male_employees_distribution_l918_91879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l918_91864

-- Define the function f(x) = 3^x - x^2
noncomputable def f (x : ℝ) : ℝ := Real.exp (x * Real.log 3) - x^2

-- Theorem statement
theorem root_in_interval :
  ∃ x ∈ Set.Ioo (-1 : ℝ) 0, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l918_91864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2x_properties_l918_91886

-- Define the function f(x) = tan(2x)
noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x)

-- State the theorem
theorem tan_2x_properties :
  -- The minimum positive period is π/2
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧ T = Real.pi / 2) ∧
  -- f(x) is an odd function
  (∀ (x : ℝ), f (-x) = -f x) ∧
  -- The symmetry centers are (kπ/4, 0) where k ∈ ℤ
  (∀ (k : ℤ), ∀ (x : ℝ), f (k * Real.pi / 4 + x) = -f (k * Real.pi / 4 - x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2x_properties_l918_91886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_triple_f_equation_l918_91897

def f (x : ℕ+) : ℕ+ := x ^ x.val

theorem smallest_n_for_triple_f_equation : 
  (∃ (n : ℕ), ∃ (m : ℕ+), m ≠ 1 ∧ 
    f (f (f m)) = (m : ℕ) ^ ((m : ℕ) ^ (n + 2020))) ∧
  (∀ (k : ℕ), k < 13611 → 
    ¬∃ (m : ℕ+), m ≠ 1 ∧ 
      f (f (f m)) = (m : ℕ) ^ ((m : ℕ) ^ (k + 2020))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_triple_f_equation_l918_91897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salad_chopping_difference_l918_91837

/-- Represents the salad chopping problem with Tom and Tammy --/
structure SaladChoppers where
  tom_rate : ℚ     -- Tom's chopping rate in lb/minute
  tammy_rate : ℚ   -- Tammy's chopping rate in lb/minute
  total_salad : ℚ  -- Total amount of salad chopped together

/-- Calculates the percentage by which Tom's chopped salad quantity is lesser than Tammy's --/
def chop_difference_percentage (sc : SaladChoppers) : ℚ :=
  let total_rate := sc.tom_rate + sc.tammy_rate
  let time := sc.total_salad / total_rate
  let tom_amount := time * sc.tom_rate
  let tammy_amount := time * sc.tammy_rate
  (tammy_amount - tom_amount) / tammy_amount * 100

/-- Theorem stating that the percentage difference in chopped salad quantity is approximately 55.56% --/
theorem salad_chopping_difference 
  (sc : SaladChoppers) 
  (h1 : sc.tom_rate = 2 / 3)   -- Tom chops 2 lb in 3 minutes
  (h2 : sc.tammy_rate = 3 / 2) -- Tammy chops 3 lb in 2 minutes
  (h3 : sc.total_salad = 65)   -- They chop 65 lb together
  : ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ |chop_difference_percentage sc - 5556 / 100| < ε := by
  sorry -- Proof omitted

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salad_chopping_difference_l918_91837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angles_at_specific_points_l918_91889

/-- The angle of inclination of a tangent line to a curve at a point --/
noncomputable def angleOfInclination (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  Real.arctan (deriv f x)

/-- The curve equation --/
def f (x : ℝ) : ℝ := x^2 + 5*x + 3

theorem tangent_angles_at_specific_points :
  (angleOfInclination f (-2) = π/4) ∧
  (angleOfInclination f 0 = Real.arctan 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angles_at_specific_points_l918_91889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ladder_slide_rate_l918_91876

-- Define the ladder's length
def ladderLength : ℝ := 10

-- Define the rate at which the bottom of the ladder moves away from the wall
def bottomRate : ℝ := 1

-- Define the distance of the bottom of the ladder from the wall at the moment of interest
def bottomDistance : ℝ := 6

-- Function to calculate the height of the top of the ladder
noncomputable def topHeight (x : ℝ) : ℝ := 
  Real.sqrt (ladderLength^2 - x^2)

-- Theorem stating the rate at which the top of the ladder slides down
theorem ladder_slide_rate : 
  let x := bottomDistance
  let y := topHeight x
  let dx_dt := bottomRate
  let dy_dt := -((x * dx_dt) / y)
  dy_dt = -3/4
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ladder_slide_rate_l918_91876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_m_inclination_l918_91863

-- Define the parallel lines
def l1 : Set (ℝ × ℝ) := {(x, y) | x + y = 0}
def l2 : Set (ℝ × ℝ) := {(x, y) | x + y + Real.sqrt 6 = 0}

-- Define the line m
def m : Set (ℝ × ℝ) := {(x, y) | ∃ (θ : ℝ), y = Real.tan θ * x + Real.cos θ * Real.sqrt 6}

-- Define the length of the intercepted segment
noncomputable def intercepted_length : ℝ := 2 * Real.sqrt 3

-- Define the possible angles of inclination
def possible_angles : Set ℝ := {105 * Real.pi / 180, 165 * Real.pi / 180}

theorem line_m_inclination :
  ∃ (θ : ℝ), θ ∈ possible_angles ∧
  ∀ (p q : ℝ × ℝ), p ∈ l1 ∧ q ∈ l2 ∧ p ∈ m ∧ q ∈ m →
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = intercepted_length :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_m_inclination_l918_91863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l918_91808

theorem sin_cos_difference (θ : ℝ) 
  (h1 : θ > -π/2 ∧ θ < π/2) 
  (h2 : Real.sin θ + Real.cos θ = 2/3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 14 / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l918_91808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_theorem_l918_91858

def Triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ (D E F : ℝ × ℝ), 
    D = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) ∧
    E = (C.1 / 2, C.2 / 2) ∧
    F = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def Median (A B C D : ℝ × ℝ) (length : ℝ) : Prop :=
  D = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) ∧
  (A.1 - D.1)^2 + (A.2 - D.2)^2 = length^2

def Perpendicular (A B C D : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (D.1 - C.1) + (B.2 - A.2) * (D.2 - C.2) = 0

theorem median_length_theorem (A B C : ℝ × ℝ) :
  Triangle A B C →
  ∃ (D E F : ℝ × ℝ),
    Median A B C D 18 ∧
    Median B A C E 13.5 ∧
    Perpendicular A D B E →
    Median C A B F 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_theorem_l918_91858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_ratio_square_common_ratio_is_half_or_neg_half_l918_91875

/-- A geometric sequence with given terms -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) = a n * (a 2 / a 1)
  a_2 : a 2 = 2
  a_6 : a 6 = 1/8

/-- The common ratio of a geometric sequence -/
noncomputable def common_ratio (seq : GeometricSequence) : ℝ := seq.a 2 / seq.a 1

theorem common_ratio_square (seq : GeometricSequence) :
  (common_ratio seq)^2 = 1/4 := by
  -- Proof steps would go here
  sorry

theorem common_ratio_is_half_or_neg_half (seq : GeometricSequence) :
  common_ratio seq = 1/2 ∨ common_ratio seq = -1/2 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_ratio_square_common_ratio_is_half_or_neg_half_l918_91875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_statements_true_l918_91831

-- Define the geometric objects
variable (Point Line Plane : Type)

-- Define geometric relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (planeparallel : Plane → Plane → Prop)
variable (lineParallelToPlane : Line → Plane → Prop)
variable (linePerpendicularToPlane : Line → Plane → Prop)
variable (planePerpendicularToLine : Plane → Line → Prop)

-- Define the statements
def statement1 (Line Plane : Type) 
  (planeparallel : Plane → Plane → Prop) 
  (lineParallelToPlane : Line → Plane → Prop) : Prop := 
  ∀ (P1 P2 : Plane) (L : Line), 
    lineParallelToPlane L P1 → lineParallelToPlane L P2 → planeparallel P1 P2

def statement2 (Plane : Type) (planeparallel : Plane → Plane → Prop) : Prop := 
  ∀ (P1 P2 P3 : Plane), 
    planeparallel P1 P3 → planeparallel P2 P3 → planeparallel P1 P2

def statement3 (Line : Type) (parallel perpendicular : Line → Line → Prop) : Prop := 
  ∀ (L1 L2 L3 : Line), 
    perpendicular L1 L3 → perpendicular L2 L3 → parallel L1 L2

def statement4 (Line Plane : Type) 
  (parallel : Line → Line → Prop) 
  (linePerpendicularToPlane : Line → Plane → Prop) : Prop := 
  ∀ (L1 L2 : Line) (P : Plane), 
    linePerpendicularToPlane L1 P → linePerpendicularToPlane L2 P → parallel L1 L2

def statement5 (Line Plane : Type) 
  (planeparallel : Plane → Plane → Prop) 
  (planePerpendicularToLine : Plane → Line → Prop) : Prop := 
  ∀ (P1 P2 : Plane) (L : Line), 
    planePerpendicularToLine P1 L → planePerpendicularToLine P2 L → planeparallel P1 P2

-- Theorem stating that exactly 3 of the statements are true
theorem exactly_three_statements_true : 
  (¬statement1 Line Plane planeparallel lineParallelToPlane ∧ 
   statement2 Plane planeparallel ∧ 
   ¬statement3 Line parallel perpendicular ∧ 
   statement4 Line Plane parallel linePerpendicularToPlane ∧ 
   statement5 Line Plane planeparallel planePerpendicularToLine) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_statements_true_l918_91831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l918_91883

noncomputable def a (x : ℝ) : ℝ × ℝ := (3, -Real.sin (2 * x))
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos (2 * x), Real.sqrt 3)
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  (∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ ∃ (x : ℝ), f x = M) ∧
  (∃ (S : Set ℝ), S = {x | ∃ (k : ℤ), x = k * Real.pi - Real.pi / 12 ∧ f x = 2 * Real.sqrt 3}) ∧
  (∃ (α₁ α₂ : ℝ), 0 < α₁ ∧ α₁ < Real.pi ∧ 0 < α₂ ∧ α₂ < Real.pi ∧ 
    f α₁ = -Real.sqrt 3 ∧ f α₂ = -Real.sqrt 3 ∧ 
    α₁ = Real.pi / 4 ∧ α₂ = 7 * Real.pi / 12) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l918_91883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_when_a_is_one_range_of_a_when_f_increasing_l918_91840

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a / (2 * x)

-- Theorem 1: Range of f when a = 1
theorem range_of_f_when_a_is_one :
  ∀ y : ℝ, (∃ x : ℝ, x ∈ Set.Ioo 0 1 ∧ f 1 x = y) ↔ y ∈ Set.Ici (Real.sqrt 2) :=
by sorry

-- Theorem 2: Range of a when f is increasing
theorem range_of_a_when_f_increasing :
  ∀ a : ℝ, (∀ x ∈ Set.Ioo 0 1, ∀ y ∈ Set.Ioo 0 1, x < y → f a x < f a y) → a ∈ Set.Iic 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_when_a_is_one_range_of_a_when_f_increasing_l918_91840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l918_91849

/-- An ellipse with center at the origin, foci on the x-axis, passing through (3, 4),
    and PF₁ perpendicular to PF₂ has the equation x²/45 + y²/20 = 1 -/
theorem ellipse_equation (F₁ F₂ : ℝ × ℝ) : 
  let P : ℝ × ℝ := (3, 4)
  (∃ c : ℝ, F₁ = (-c, 0) ∧ F₂ = (c, 0)) →   -- Foci on x-axis
  ((P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0) →   -- PF₁ ⊥ PF₂
  (∀ x y : ℝ, x^2/45 + y^2/20 = 1 ↔ 
    ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
    x = 3*Real.sqrt 5*(1-t) + 3*Real.sqrt 5*t ∧ 
    y = 4*(1-t) + 4*t) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l918_91849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_greater_than_square_l918_91899

/-- Given a square and a circle with equal perimeters, the area of the circle is greater than the area of the square. -/
theorem circle_area_greater_than_square (r s : ℝ) (hr : r > 0) (hs : s > 0) 
  (h : 2 * Real.pi * r = 4 * s) : Real.pi * r^2 > s^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_greater_than_square_l918_91899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_345_angles_l918_91892

/-- A triangle with sides 3, 4, and 5 has angles 90°, arccos(4/5), and arccos(3/5) -/
theorem triangle_345_angles : 
  ∀ (A B C : ℝ) (a b c : ℝ),
  a = 3 ∧ b = 4 ∧ c = 5 →
  A + B + C = π →
  Real.cos A = (b^2 + c^2 - a^2) / (2*b*c) →
  Real.cos B = (a^2 + c^2 - b^2) / (2*a*c) →
  Real.cos C = (a^2 + b^2 - c^2) / (2*a*b) →
  C = π/2 ∧ A = Real.arccos (4/5) ∧ B = Real.arccos (3/5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_345_angles_l918_91892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_n_constant_f_n_identity_f_n_square_l918_91860

noncomputable def f_n (f : ℝ → ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  Finset.sum (Finset.range (n + 1)) (λ k => (n.choose k : ℝ) * f (k / n) * x^k * (1 - x)^(n - k))

theorem f_n_constant (n : ℕ) (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) :
  f_n (λ _ => 1) n x = 1 := by sorry

theorem f_n_identity (n : ℕ) (x : ℝ) :
  f_n id n x = x := by sorry

theorem f_n_square (n : ℕ) (x : ℝ) (h : n ≥ 2) :
  f_n (λ x => x^2) n x = ((n - 1) / n : ℝ) * x^2 + x / n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_n_constant_f_n_identity_f_n_square_l918_91860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_leftmost_and_rightmost_l918_91865

/-- Represents the order of contestants from left to right -/
inductive Contestant
| A
| B
| C
| D
| E

/-- The number assigned to each contestant -/
def contestant_number : Contestant → ℕ := sorry

/-- The sum of all contestant numbers -/
def total_sum : ℕ := 35

/-- The sum of numbers to the right of a given contestant -/
def sum_to_right : Contestant → ℕ := sorry

/-- The conditions given in the problem -/
axiom sum_to_right_E : sum_to_right Contestant.E = 13
axiom sum_to_right_D : sum_to_right Contestant.D = 31
axiom sum_to_right_A : sum_to_right Contestant.A = 21
axiom sum_to_right_C : sum_to_right Contestant.C = 7

/-- The leftmost contestant -/
def leftmost : Contestant := sorry

/-- The rightmost contestant -/
def rightmost : Contestant := sorry

/-- The main theorem to prove -/
theorem sum_of_leftmost_and_rightmost :
  contestant_number leftmost + contestant_number rightmost = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_leftmost_and_rightmost_l918_91865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_phi_value_l918_91826

noncomputable def f (x φ : ℝ) : ℝ := Real.sqrt 2 * Real.sin (x + Real.pi/4 + φ)

theorem odd_function_phi_value (φ : ℝ) :
  (∀ x, f x φ = -f (-x) φ) →  -- f is an odd function
  φ ∈ Set.Icc (-Real.pi/2) (Real.pi/2) →  -- φ is in the closed interval [-π/2, π/2]
  φ = -Real.pi/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_phi_value_l918_91826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_meeting_problem_l918_91825

/-- Represents the time taken by a train to reach its destination after meeting another train -/
noncomputable def time_after_meeting (speed : ℝ) (remaining_distance : ℝ) : ℝ :=
  remaining_distance / speed

/-- 
Given two trains with speeds in ratio 3:1, prove that if the slower train takes 36 hours 
to reach its destination after they meet, the faster train will take 12 hours.
-/
theorem train_meeting_problem (speed_slow : ℝ) (speed_fast : ℝ) (remaining_distance : ℝ) 
  (h1 : speed_fast = 3 * speed_slow) 
  (h2 : time_after_meeting speed_slow remaining_distance = 36) :
  time_after_meeting speed_fast remaining_distance = 12 := by
  sorry

#check train_meeting_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_meeting_problem_l918_91825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l918_91881

noncomputable def f (x : ℝ) := Real.cos ((1/2) * x + 5 * Real.pi / 12)

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = 4 * Real.pi := by
  sorry

#check smallest_positive_period_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l918_91881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_3_fourth_power_l918_91813

-- Define the functions h and k
noncomputable def h : ℝ → ℝ := sorry
noncomputable def k : ℝ → ℝ := sorry

-- State the conditions
axiom h_k_composition (x : ℝ) : x ≥ 1 → h (k x) = x^3
axiom k_h_composition (x : ℝ) : x ≥ 1 → k (h x) = x^4
axiom k_81 : k 81 = 81

-- State the theorem to be proved
theorem k_3_fourth_power : (k 3)^4 = 81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_3_fourth_power_l918_91813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l918_91861

noncomputable def f (w : ℝ) (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin (w * x + Real.pi / 6) * Real.cos (w * x)

noncomputable def g (w : ℝ) (x : ℝ) : ℝ := f w (x - Real.pi / 6)

theorem function_properties (w : ℝ) (α : ℝ) :
  (0 < w) ∧ (w < 2) ∧ 
  (f w (5 * Real.pi / 12) = Real.sqrt 3 / 2) ∧
  (g w (α / 2) = 5 * Real.sqrt 3 / 6) →
  (w = 1) ∧ 
  (∀ x, f w (x + Real.pi) = f w x) ∧
  (Real.cos (2 * α - Real.pi / 3) = 7 / 9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l918_91861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friday_loaves_l918_91887

/-- Represents the number of loaves baked on each day of the week -/
def BakeryOutput : Fin 7 → ℕ := sorry

/-- The consistent daily increase in the number of loaves baked -/
def DailyIncrease : ℕ → ℕ := sorry

theorem friday_loaves 
  (wed : BakeryOutput 0 = 5)
  (thu : BakeryOutput 1 = 7)
  (sat : BakeryOutput 3 = 14)
  (sun : BakeryOutput 4 = 19)
  (mon : BakeryOutput 5 = 25)
  (pattern : ∀ i : Fin 6, DailyIncrease (i + 1) = DailyIncrease i + 1)
  (output_pattern : ∀ i : Fin 6, BakeryOutput (i + 1) = BakeryOutput i + DailyIncrease i) :
  BakeryOutput 2 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_friday_loaves_l918_91887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tiles_is_four_l918_91895

/-- Represents an L-shaped tile -/
structure LTile :=
  (size : Nat)
  (h_size : size = 3)

/-- Represents a larger L-shaped figure composed of L-shaped tiles -/
structure LargerLFigure :=
  (tiles : Nat)

/-- Predicate to check if two figures are similar -/
def IsSimilar (a b : Type) : Prop := sorry

/-- Predicate to check if there's no overlap in the figure -/
def NoOverlap (a : Type) : Prop := sorry

/-- The minimum number of L-shaped tiles required to form a larger similar L-shaped figure -/
def min_tiles : Nat := 4

/-- Theorem stating that the minimum number of L-shaped tiles required is 4 -/
theorem min_tiles_is_four :
  ∀ (f : LargerLFigure), f.tiles ≥ min_tiles := by
  sorry

/-- Additional property for LargerLFigure to ensure similarity and no overlap -/
def LargerLFigure.valid (f : LargerLFigure) : Prop :=
  IsSimilar LTile LargerLFigure ∧ NoOverlap LargerLFigure

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tiles_is_four_l918_91895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walker_speed_is_pi_over_three_l918_91856

/-- A track with straight sides and semicircular ends -/
structure Track where
  inner_radius : ℝ
  width : ℝ
  straight_length : ℝ

/-- Walker's constant speed around the track -/
noncomputable def walker_speed (track : Track) : ℝ :=
  Real.pi / 3

/-- Time difference between walking outer and inner edges -/
noncomputable def time_difference (track : Track) : ℝ :=
  (2 * track.straight_length + 2 * Real.pi * (track.inner_radius + track.width)) / (walker_speed track) -
  (2 * track.straight_length + 2 * Real.pi * track.inner_radius) / (walker_speed track)

theorem walker_speed_is_pi_over_three (track : Track) 
  (h_width : track.width = 6)
  (h_time_diff : time_difference track = 36) :
  walker_speed track = Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_walker_speed_is_pi_over_three_l918_91856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagoras_schools_count_l918_91806

/-- Represents a student in the math competition --/
structure Student where
  name : String
  rank : Nat

/-- Represents a school in Pythagoras --/
structure School where
  team : Fin 3 → Student

/-- The math competition in Pythagoras --/
structure MathCompetition where
  schools : List School
  emma : Student
  sarah : Student
  zoe : Student

theorem pythagoras_schools_count (comp : MathCompetition) : 
  (comp.emma.rank < comp.sarah.rank) →
  (comp.sarah.rank < comp.zoe.rank) →
  (comp.emma.rank < 39) →
  (comp.sarah.rank = 39) →
  (comp.zoe.rank = 78) →
  (∀ s : School, s ∈ comp.schools → 
    ∃ i : Fin 3, (s.team i).rank = comp.emma.rank ∨ 
                 (s.team i).rank = comp.sarah.rank ∨ 
                 (s.team i).rank = comp.zoe.rank) →
  (∀ s1 s2 : Student, s1 ∈ (comp.schools.map (λ s => s.team 0) ++ 
                            comp.schools.map (λ s => s.team 1) ++ 
                            comp.schools.map (λ s => s.team 2)) →
                      s2 ∈ (comp.schools.map (λ s => s.team 0) ++ 
                            comp.schools.map (λ s => s.team 1) ++ 
                            comp.schools.map (λ s => s.team 2)) →
                      s1 ≠ s2 → s1.rank ≠ s2.rank) →
  (comp.emma.rank = (comp.schools.length * 3 + 1) / 2) →
  comp.schools.length = 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagoras_schools_count_l918_91806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l918_91853

/-- Given a sequence {a_n} where S_n (sum of first n terms) satisfies S_n = 1 + ka_n (k ≠ 0, k ≠ 1) -/
noncomputable def sequence_a (k : ℝ) (n : ℕ) : ℝ := -k^(n-1) / (k-1)^n

/-- Sum of squares of terms when k = -1 -/
noncomputable def sum_squares (n : ℕ) : ℝ := (1/3) * (1 - 1/4^n)

theorem sequence_properties (k : ℝ) (h1 : k ≠ 0) (h2 : k ≠ 1) :
  /- The general term of the sequence -/
  (∀ n : ℕ, n > 0 → sequence_a k n = -k^(n-1) / (k-1)^n) ∧
  /- The sum of squares when k = -1 -/
  (k = -1 → ∀ n : ℕ, n > 0 → 
    Finset.sum (Finset.range n) (λ i => (sequence_a (-1) (i+1))^2) = sum_squares n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l918_91853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_is_parallelogram_l918_91802

/-- A quadrilateral with side lengths a, b, c, and d is a parallelogram if (a-c)^2 + (b-d)^2 = 0 -/
theorem quadrilateral_is_parallelogram 
  (a b c d : ℝ) 
  (h : (a - c)^2 + (b - d)^2 = 0) : 
  a = c ∧ b = d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_is_parallelogram_l918_91802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_circle_l918_91820

-- Define the circles
noncomputable def circle_A_diameter : ℝ := 10
noncomputable def circle_B_radius : ℝ := 4
noncomputable def circle_C_diameter_radius_sum : ℝ := 9

-- Define the radii of circles A and C
noncomputable def circle_A_radius : ℝ := circle_A_diameter / 2
noncomputable def circle_C_radius : ℝ := circle_C_diameter_radius_sum / 3

-- Theorem statement
theorem smallest_circle :
  circle_C_radius < circle_A_radius ∧ circle_C_radius < circle_B_radius :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_circle_l918_91820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exchange_rate_approx_l918_91884

/-- The exchange rate between Japanese Yen (JPY) and Canadian Dollars (CAD) -/
noncomputable def exchange_rate : ℝ := 5000 / 60

/-- The exchange rate for 1 CAD in JPY -/
noncomputable def exchange_for_one_cad : ℝ := exchange_rate

/-- Theorem stating that the exchange rate for 1 CAD is approximately 83 JPY -/
theorem exchange_rate_approx :
  |exchange_for_one_cad - 83| < 1 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exchange_rate_approx_l918_91884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_passes_through_point_l918_91847

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-1) + 2

theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_passes_through_point_l918_91847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l918_91836

/-- The complex number z -/
noncomputable def z : ℂ := (2 * Complex.I) / (2 - Complex.I)

/-- Theorem: The point corresponding to z is in the second quadrant -/
theorem z_in_second_quadrant : 
  (z.re < 0) ∧ (z.im > 0) := by
  -- Simplify z
  have h : z = -2/5 + 4/5 * Complex.I := by
    -- Proof of simplification goes here
    sorry
  
  -- Show real part is negative
  have h_re : z.re = -2/5 := by
    -- Proof that real part is -2/5 goes here
    sorry
  
  -- Show imaginary part is positive
  have h_im : z.im = 4/5 := by
    -- Proof that imaginary part is 4/5 goes here
    sorry
  
  -- Conclude that z is in the second quadrant
  exact ⟨by linarith, by linarith⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l918_91836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_properties_l918_91838

-- Define the functions f and g as noncomputable
noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x)
noncomputable def g (x : ℝ) : ℝ := x^2 + 2

-- State the theorem
theorem function_composition_properties :
  (f 2 = 1/3 ∧ g 2 = 6) ∧
  (f (g 2) = 1/7) ∧
  (∀ x : ℝ, x ≠ -1 → f (g x) = 1 / (x^2 + 3)) ∧
  (∀ x : ℝ, x ≠ -1 → g (f x) = (1 / (1 + x))^2 + 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_properties_l918_91838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_six_occurrences_l918_91835

/-- The number of occurrences of a specific digit in a range of integers -/
def countDigitOccurrences (start : Nat) (stop : Nat) (digit : Nat) : Nat :=
  sorry

/-- The number of occurrences of the digit 6 in the integers from 100 to 999 inclusive -/
theorem count_six_occurrences :
  countDigitOccurrences 100 999 6 = 280 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_six_occurrences_l918_91835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_theorem_l918_91873

-- Define the given constants
noncomputable def train_speed_kmph : ℝ := 63
noncomputable def bridge_length : ℝ := 150
noncomputable def time_to_cross : ℝ := 14.284571519992687

-- Define the conversion factor from kmph to m/s
noncomputable def kmph_to_ms : ℝ := 1000 / 3600

-- Define the train speed in m/s
noncomputable def train_speed_ms : ℝ := train_speed_kmph * kmph_to_ms

-- Define the theorem
theorem train_length_theorem :
  let total_distance := train_speed_ms * time_to_cross
  let train_length := total_distance - bridge_length
  ∃ ε > 0, |train_length - 99.98| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_theorem_l918_91873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_exponent_l918_91893

/-- If x^(2a-1) + x = 5 is a quadratic equation in x, then a = 3/2 -/
theorem quadratic_equation_exponent (a : ℝ) : 
  (∀ x : ℝ, ∃ b c : ℝ, x^(2*a-1) + x = b*x^2 + c*x + 5) → a = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_exponent_l918_91893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersections_circle_triangle_rectangle_l918_91803

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a triangle in a plane -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Represents a rectangle in a plane -/
structure Rectangle where
  corners : Fin 4 → ℝ × ℝ

/-- Computes the maximum number of intersection points between two geometric objects -/
def max_intersections (α β : Type) : ℕ := sorry

/-- The maximum number of intersection points between a circle and a triangle -/
def circle_triangle_intersections : ℕ := 6

/-- The maximum number of intersection points between a circle and a rectangle -/
def circle_rectangle_intersections : ℕ := 8

/-- The maximum number of intersection points between a triangle and a rectangle -/
def triangle_rectangle_intersections : ℕ := 24

/-- Theorem stating the maximum number of intersection points between a circle, a triangle, and a rectangle -/
theorem max_intersections_circle_triangle_rectangle :
  max_intersections Circle Triangle + 
  max_intersections Circle Rectangle + 
  max_intersections Triangle Rectangle = 38 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersections_circle_triangle_rectangle_l918_91803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_exponent_l918_91816

/-- Power function -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^a

/-- Theorem: If f(9) = 3 for the power function f(x) = x^a, then a = 1/2 -/
theorem power_function_exponent (a : ℝ) : f a 9 = 3 → a = 1/2 := by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_exponent_l918_91816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_chord_lengths_l918_91817

/-- The circle C with center (t, 1-t) and radius √8 -/
def Circle (t : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - t)^2 + (p.2 + t - 1)^2 = 8}

/-- The chord length intercepted by an axis -/
noncomputable def ChordLength (t : ℝ) (axis : ℝ → ℝ) : ℝ :=
  2 * Real.sqrt (8 - (axis t)^2)

/-- Theorem: The chord lengths intercepted by the coordinate axes are equal iff t = 1/2 -/
theorem equal_chord_lengths (t : ℝ) :
  ChordLength t (fun x => x) = ChordLength t (fun x => 1 - x) ↔ t = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_chord_lengths_l918_91817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_240_degree_angle_l918_91880

/-- 
Given a point A(x,y) on the terminal side of an angle of 240°, 
where A is distinct from the origin, prove that y/x = √3.
-/
theorem point_on_240_degree_angle (x y : ℝ) : 
  (x ≠ 0 ∨ y ≠ 0) →  -- A is distinct from the origin
  (x, y) ∈ {p : ℝ × ℝ | ∃ r : ℝ, r > 0 ∧ p.1 = -r * (1/2) ∧ p.2 = -r * (Real.sqrt 3/2)} → -- A is on the terminal side of 240°
  y / x = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_240_degree_angle_l918_91880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_benito_juarez_birth_year_l918_91818

-- Define the range of the 19th century
def nineteenth_century : Set ℕ := {n | 1801 ≤ n ∧ n ≤ 1900}

-- Define the condition for Benito Juárez's age
def age_condition (birth_year : ℕ) : Prop :=
  ∃ x : ℕ, (birth_year + x) = (birth_year + x)^2 ∧ (birth_year + x) ∈ nineteenth_century

-- Theorem statement
theorem benito_juarez_birth_year :
  ∃ birth_year : ℕ,
    birth_year ∈ nineteenth_century ∧
    age_condition birth_year ∧
    birth_year = 1806 := by
  -- Proof goes here
  sorry

#check benito_juarez_birth_year

end NUMINAMATH_CALUDE_ERRORFEEDBACK_benito_juarez_birth_year_l918_91818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_properties_l918_91868

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x - Real.exp (x + m)

theorem extremum_properties (m : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f m x ≤ f m 1) →
  (m = -1 ∧
   (∀ x y, x ∈ Set.Ioo 0 1 → y ∈ Set.Ioo 0 1 → x ≤ y → f m x ≤ f m y) ∧
   (∀ x y, x ∈ Set.Ioi 1 → y ∈ Set.Ioi 1 → x < y → f m x > f m y)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_properties_l918_91868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_downstream_time_l918_91891

/-- Represents the time taken for a boat to travel downstream -/
noncomputable def time_downstream (boat_speed : ℝ) (current_speed : ℝ) (distance : ℝ) : ℝ :=
  distance / (boat_speed + current_speed)

/-- Theorem: A boat with speed 42 km/hr in still water, travelling downstream
    with a current of 3 km/hr for a distance of 33 km, takes 44 minutes. -/
theorem boat_downstream_time :
  let boat_speed : ℝ := 42
  let current_speed : ℝ := 3
  let distance : ℝ := 33
  let time_hours := time_downstream boat_speed current_speed distance
  time_hours * 60 = 44 := by
  sorry

#eval Float.round ((33 / (42 + 3)) * 60) -- This will evaluate to 44.0

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_downstream_time_l918_91891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_to_mean_l918_91814

theorem median_to_mean (m : ℝ) (h : m + 7 = 12) : 
  let s := {m, m + 4, m + 7, m + 10, m + 18}
  (Finset.sum s id) / (Finset.card s : ℝ) = 12.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_to_mean_l918_91814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_formula_l918_91839

def mySequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 2
  | n + 1 => mySequence n / (1 + mySequence n)

theorem mySequence_formula (n : ℕ) : mySequence n = 2 / (2 * n.succ + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_formula_l918_91839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_theorem_l918_91867

noncomputable def f (x : ℝ) : ℝ := x^4 - 5*x^3 + 13*x^2 - 19*x + 8
noncomputable def g (x : ℝ) : ℝ := x^2 - 3*x + 5/2
noncomputable def r (x : ℝ) : ℝ := 2*x - 13/4

theorem polynomial_division_theorem :
  ∃ q : ℝ → ℝ, ∀ x, f x = g x * q x + r x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_theorem_l918_91867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_pairs_l918_91810

def satisfies_equation (a b : ℕ) : Prop :=
  (a : ℚ) + (1 : ℚ) / b = 17 * ((1 : ℚ) / a + b)

def satisfies_inequality (a b : ℕ) : Prop :=
  a + b ≤ 150

def valid_pair (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ satisfies_equation a b ∧ satisfies_inequality a b

theorem count_valid_pairs :
  ∃! (pairs : Finset (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ pairs ↔ valid_pair p.1 p.2) ∧
    pairs.card = 8 := by sorry

#check count_valid_pairs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_pairs_l918_91810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_six_zero_l918_91822

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence (changed to ℚ for computability)
  d : ℚ      -- Common difference (changed to ℚ)
  h : d ≠ 0  -- The common difference is non-zero

/-- The sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.a 1 + ((n : ℚ) - 1) * seq.d)

/-- The theorem statement -/
theorem arithmetic_sequence_sum_six_zero (seq : ArithmeticSequence) :
  (seq.a 2)^2 + (seq.a 3)^2 = (seq.a 4)^2 + (seq.a 5)^2 → S seq 6 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_six_zero_l918_91822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_circle_M_radius_l918_91885

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (-2 + (1/2) * t, (Real.sqrt 3 / 2) * t)

-- Define circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 10

-- Define point P
def point_P : ℝ × ℝ := (-2, 0)

-- Define circle M
def circle_M (a : ℝ) (x y : ℝ) : Prop := (x - a)^2 + y^2 = a^2

-- Theorem 1: Product of distances from P to intersection points
theorem intersection_distance_product :
  ∃ A B : ℝ × ℝ,
    (∃ t₁ t₂ : ℝ, line_l t₁ = A ∧ line_l t₂ = B) ∧
    circle_C A.1 A.2 ∧
    circle_C B.1 B.2 ∧
    (A.1 - point_P.1)^2 + (A.2 - point_P.2)^2 *
    (B.1 - point_P.1)^2 + (B.2 - point_P.2)^2 = 36 := by
  sorry

-- Theorem 2: Radius of circle M
theorem circle_M_radius :
  ∃ a : ℝ,
    a > 0 ∧
    circle_M a 0 0 ∧
    (∃ x y : ℝ, circle_M a x y ∧ line_l ((x + 2) * 2) = (x, y)) ∧
    (∃ x₁ y₁ x₂ y₂ : ℝ,
      circle_M a x₁ y₁ ∧ circle_M a x₂ y₂ ∧
      line_l ((x₁ + 2) * 2) = (x₁, y₁) ∧
      line_l ((x₂ + 2) * 2) = (x₂, y₂) ∧
      (x₁ - x₂)^2 + (y₁ - y₂)^2 = 1) ∧
    a = 13 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_circle_M_radius_l918_91885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_iterative_average_difference_l918_91890

noncomputable def iterative_average (xs : List ℝ) : ℝ :=
  xs.foldl (fun acc x => (acc + x) / 2) xs.head!

def prime_set : List ℝ := [2, 3, 5, 7, 11]

theorem iterative_average_difference :
  let max_value := prime_set.permutations.map iterative_average |>.maximum?
  let min_value := prime_set.permutations.map iterative_average |>.minimum?
  ∀ max min, max_value = some max → min_value = some min →
  max - min = 4.6875 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_iterative_average_difference_l918_91890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_women_in_first_class_l918_91801

def total_passengers : ℕ := 150
def women_percentage : ℚ := 70 / 100
def first_class_percentage : ℚ := 15 / 100

theorem women_in_first_class :
  Int.floor ((total_passengers : ℚ) * women_percentage * first_class_percentage) = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_women_in_first_class_l918_91801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bed_fraction_l918_91848

theorem flower_bed_fraction (yard_length yard_width trapezoid_long_side trapezoid_short_side : ℝ) 
  (h1 : yard_length = 30)
  (h2 : yard_width = 10)
  (h3 : trapezoid_long_side = 40)
  (h4 : trapezoid_short_side = 20) :
  let triangle_side := (trapezoid_long_side - trapezoid_short_side) / 2
  let triangle_area := (Real.sqrt 3 / 4) * triangle_side^2
  let total_flower_bed_area := 2 * triangle_area
  let yard_area := yard_length * yard_width
  total_flower_bed_area / yard_area = 5 * Real.sqrt 3 / 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bed_fraction_l918_91848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vnc_expected_value_l918_91827

/-- The Very Normal Coin (VNC) -/
structure VNC where
  /-- Probability of heads on a normal flip -/
  prob_heads : ℚ
  /-- Probability of tails on a normal flip -/
  prob_tails : ℚ
  /-- Number of consecutive same results that guarantees the opposite result -/
  consecutive_limit : ℕ

/-- Bob's betting strategy -/
def bet (n : ℕ) : ℚ := 2^(-n : ℤ)

/-- Expected value calculation for the VNC game -/
noncomputable def expected_value (coin : VNC) (first_flip : Bool) : ℚ :=
  sorry

/-- The theorem to be proved -/
theorem vnc_expected_value :
  let coin : VNC := ⟨1/2, 1/2, 5⟩
  let first_flip : Bool := true -- true represents heads
  expected_value coin first_flip = 341/683 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vnc_expected_value_l918_91827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_cover_theorem_l918_91815

/-- An increasing function from [0,1] to [0,1] -/
def IncreasingFunction (F : ℝ → ℝ) : Prop :=
  (∀ x, x ∈ Set.Icc 0 1 → F x ∈ Set.Icc 0 1) ∧
  (∀ x y, x ∈ Set.Icc 0 1 → y ∈ Set.Icc 0 1 → x < y → F x < F y)

/-- A rectangle with sides parallel to the axes -/
structure Rectangle where
  x : ℝ
  y : ℝ
  width : ℝ
  height : ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- A covering of the graph of F by rectangles -/
def Covers (F : ℝ → ℝ) (rs : List Rectangle) : Prop :=
  ∀ x, x ∈ Set.Icc 0 1 → ∃ r ∈ rs, x ∈ Set.Icc r.x (r.x + r.width) ∧ F x ∈ Set.Icc r.y (r.y + r.height)

theorem graph_cover_theorem (F : ℝ → ℝ) (h : IncreasingFunction F) :
  ∀ n : ℕ, ∃ N : ℕ, N ≤ n ∧
    ∃ rs : List Rectangle, rs.length = N ∧ Covers F rs ∧
      ∀ r ∈ rs, r.area = 1 / (2 * (n : ℝ)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_cover_theorem_l918_91815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_age_l918_91877

/-- Given the age relationships between James, John, and Tim, prove James's current age. -/
theorem james_age : ∃ (james : ℕ), james = 30 := by
  -- Define the variables
  let john_james_diff : ℕ := 12
  let tim_age : ℕ := 79

  -- Calculate John's age
  have john_age : ℕ := (tim_age + 5) / 2

  -- Calculate James's age
  let james : ℕ := john_age - john_james_diff

  -- Assert that James's age is 30
  have h : james = 30 := by
    -- Proof goes here
    sorry

  -- Conclude the theorem
  exact ⟨james, h⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_age_l918_91877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_assignment_theorem_l918_91828

/-- The number of students --/
def total_students : ℕ := 6

/-- The number of students to be chosen --/
def chosen_students : ℕ := 4

/-- The number of subjects --/
def subjects : ℕ := 4

/-- The number of subjects that students A and B can participate in --/
def subjects_for_ab : ℕ := 3

/-- Represents the assignment problem --/
def assignment_problem (n m k l : ℕ) : Prop :=
  (n ≥ m) ∧ (k ≥ l) ∧ 
  (∃ (ways : ℕ), ways = 
    (Nat.factorial m * Nat.factorial (n - m) / Nat.factorial n) * k * (k - 1) + 
    (l * (l - 1) / 2) * (Nat.factorial (n - 2) * Nat.factorial (m - 2) / Nat.factorial (n - m)) +
    2 * l * (Nat.factorial (n - 1) * Nat.factorial (m - 1) / Nat.factorial (n - m)))

/-- The main theorem --/
theorem assignment_theorem : 
  assignment_problem total_students chosen_students subjects subjects_for_ab → 
  (∃ (ways : ℕ), ways = 240) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_assignment_theorem_l918_91828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ounces_per_jar_is_15_l918_91841

/-- Represents the turtle feeding scenario -/
structure TurtleFeeding where
  food_req : ℚ  -- Food requirement in ounces per pound
  total_weight : ℚ  -- Total weight of turtles in pounds
  jar_cost : ℚ  -- Cost of each jar in dollars
  total_cost : ℚ  -- Total cost to feed the turtles in dollars

/-- Calculates the number of ounces in each jar of turtle food -/
def ounces_per_jar (tf : TurtleFeeding) : ℚ :=
  (tf.food_req * tf.total_weight) / (tf.total_cost / tf.jar_cost)

/-- Theorem stating that the number of ounces in each jar is 15 -/
theorem ounces_per_jar_is_15 (tf : TurtleFeeding) 
    (h1 : tf.food_req = 2)  -- 1 oz per 1/2 lb is equivalent to 2 oz per lb
    (h2 : tf.total_weight = 30)
    (h3 : tf.jar_cost = 2)
    (h4 : tf.total_cost = 8) : 
  ounces_per_jar tf = 15 := by
  sorry

#eval ounces_per_jar { food_req := 2, total_weight := 30, jar_cost := 2, total_cost := 8 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ounces_per_jar_is_15_l918_91841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_PQ_length_l918_91800

-- Define the circle and points
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

-- Define membership for points in a circle
def Point.inCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.1)^2 + (p.y - c.center.2)^2 = c.radius^2

-- Define line segment
def LineSegment (p q : Point) : Set (ℝ × ℝ) :=
  {(x, y) | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ x = p.x + t * (q.x - p.x) ∧ y = p.y + t * (q.y - p.y)}

-- Define length of a line segment
noncomputable def length (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

-- Define the conditions
def isDiameter (c : Circle) (m n : Point) : Prop :=
  m.inCircle c ∧ n.inCircle c ∧ length m n = 2 * c.radius

def sameSide (c : Circle) (m n a b : Point) : Prop := sorry
def oppositeSide (c : Circle) (m n c' : Point) : Prop := sorry
def isMidpoint (c : Circle) (m n a : Point) : Prop := sorry
def intersects (c : Circle) (s1 s2 : Set (ℝ × ℝ)) (p : Point) : Prop := sorry

-- Define the variables
variable (circle : Circle)
variable (A B C M N P Q : Point)

-- State the theorem
theorem max_PQ_length : 
  A.inCircle circle → B.inCircle circle → C.inCircle circle → M.inCircle circle → N.inCircle circle →
  isDiameter circle M N →
  sameSide circle M N A B →
  oppositeSide circle M N C →
  isMidpoint circle M N A →
  intersects circle (LineSegment C A) (LineSegment M N) P →
  intersects circle (LineSegment C B) (LineSegment M N) Q →
  length M N = 1 →
  length M B = 12/13 →
  length P Q ≤ (17 - 4 * Real.sqrt 15) / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_PQ_length_l918_91800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_completion_time_l918_91852

-- Define the workers' individual completion times
variable (A B C D : ℝ)

-- Define the combined completion time for Alpha, Beta, and Delta
noncomputable def h (A B D : ℝ) : ℝ := 1 / (1/A + 1/B + 1/D)

-- State the theorem
theorem workers_completion_time 
  (h1 : 1/A + 1/B + 1/C + 1/D = 1/(A - 8))
  (h2 : 1/A + 1/B + 1/C + 1/D = 1/(B - 2))
  (h3 : 1/A + 1/B + 1/C + 1/D = 3/C)
  (h4 : 1/D = 1/(2*A)) -- Assumption from the solution for simplification
  : h A B D = 16/11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_completion_time_l918_91852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_three_power_minus_nine_power_l918_91823

theorem min_value_three_power_minus_nine_power :
  (∃ x : ℝ, (3 : ℝ)^x - (9 : ℝ)^x = 1/4) ∧ (∀ x : ℝ, (3 : ℝ)^x - (9 : ℝ)^x ≥ 1/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_three_power_minus_nine_power_l918_91823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_origin_to_point_l918_91812

def origin : ℝ × ℝ := (0, 0)
def point : ℝ × ℝ := (9, -12)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_from_origin_to_point :
  distance origin point = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_origin_to_point_l918_91812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_passings_theorem_l918_91804

/-- Represents a swimmer with a given speed -/
structure Swimmer where
  speed : ℕ  -- Changed to ℕ to avoid Real.instReprReal issue
  deriving Repr

/-- Represents the swimming scenario -/
structure SwimmingScenario where
  poolLength : ℕ  -- Changed to ℕ
  swimmerA : Swimmer
  swimmerB : Swimmer
  duration : ℕ  -- Changed to ℕ
  deriving Repr

/-- Calculates the number of times swimmers pass each other -/
def calculatePassings (scenario : SwimmingScenario) : ℕ :=
  sorry

/-- Theorem stating the number of passings in the given scenario -/
theorem swimming_passings_theorem (scenario : SwimmingScenario) 
  (h1 : scenario.poolLength = 120)
  (h2 : scenario.swimmerA.speed = 4)
  (h3 : scenario.swimmerB.speed = 3)
  (h4 : scenario.duration = 15 * 60) : -- 15 minutes in seconds
  calculatePassings scenario = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_passings_theorem_l918_91804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l918_91844

theorem evaluate_expression : ((9 : ℚ)⁻¹ - (6 : ℚ)⁻¹)⁻¹ = -18 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l918_91844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_ratio_l918_91874

/-- Trapezoid with given properties -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  DG : ℝ
  h : ℝ  -- height of the trapezoid

/-- Ratio of areas of trapezoid ABCD to quadrilateral KLMN -/
noncomputable def area_ratio (t : Trapezoid) : ℝ := 
  let NL := t.AB + t.DG + t.CD
  (t.AB + t.CD) / (2 * NL)

/-- Theorem stating the possible ratios -/
theorem trapezoid_area_ratio (t : Trapezoid) 
  (h1 : t.AB = 15) 
  (h2 : t.CD = 19) 
  (h3 : t.DG = 17) : 
  area_ratio t = 2/3 ∨ area_ratio t = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_ratio_l918_91874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_positive_necessary_not_sufficient_for_ax_squared_positive_l918_91833

theorem a_positive_necessary_not_sufficient_for_ax_squared_positive :
  ∃ (a x : ℝ), (∀ x : ℝ, a * x^2 > 0 → a > 0) ∧ ¬(∀ x : ℝ, a > 0 → a * x^2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_positive_necessary_not_sufficient_for_ax_squared_positive_l918_91833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_abc_l918_91846

-- Define the constants as noncomputable
noncomputable def a : ℝ := 2^(Real.log 3)
noncomputable def b : ℝ := 2^(Real.log 2 / Real.log 10)
noncomputable def c : ℝ := (1/4)^(Real.log (1/2) / Real.log (1/3))

-- State the theorem
theorem compare_abc : a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_abc_l918_91846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zero_points_l918_91854

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2^(x-1) - x^2

-- Theorem statement
theorem f_has_three_zero_points :
  ∃ (a b c : ℝ), a < b ∧ b < c ∧
  f a = 0 ∧ f b = 0 ∧ f c = 0 ∧
  ∀ x : ℝ, f x = 0 → x = a ∨ x = b ∨ x = c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zero_points_l918_91854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_line_and_curve_l918_91888

noncomputable def min_distance : ℝ := 1 + (1/4) * Real.log 3

theorem min_distance_between_line_and_curve : ∃ (d : ℝ),
  (∀ (x₁ y₁ x₂ y₂ : ℝ),
    (Real.sqrt 3 * x₁ - y₁ + 1 = 0) →  -- Point (x₁, y₁) is on the line
    (y₂ = Real.log x₂) →      -- Point (x₂, y₂) is on the curve
    d ≤ Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)) ∧
  d = min_distance :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_line_and_curve_l918_91888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equality_l918_91843

/-- Two polynomials of degree n that are equal at n+1 points are identical -/
theorem polynomial_equality {R : Type*} [Field R] (n : ℕ) (f g : Polynomial R) :
  (∃ (S : Finset R), S.card = n + 1 ∧ ∀ x ∈ S, f.eval x = g.eval x) →
  (f.degree ≤ n ∧ g.degree ≤ n) →
  f = g :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equality_l918_91843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l918_91811

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ Set.Ioc (-1) 0 then (1 / (x + 1)) - 3
  else if x ∈ Set.Ioc 0 1 then x
  else 0  -- Define a value for x outside (-1,1] to make f total

-- Define g in terms of f and m
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f x - m * x - m

-- Define the set of m values that satisfy the condition
def valid_m : Set ℝ := {m | ∃ (x y : ℝ), x ∈ Set.Ioc (-1) 1 ∧ y ∈ Set.Ioc (-1) 1 ∧ x ≠ y ∧ 
                              g m x = 0 ∧ g m y = 0 ∧ 
                              ∀ z ∈ Set.Ioc (-1) 1, g m z = 0 → (z = x ∨ z = y)}

-- The theorem to prove
theorem m_range : valid_m = Set.Ioc (-9/4) (-2) ∪ Set.Ioc 0 (1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l918_91811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_l918_91805

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    0 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    n = 100 * a + 10 * b + c ∧
    n = (100 * b + 10 * c + a + 100 * c + 10 * a + b) / 2

theorem valid_numbers :
  ∀ n : ℕ, is_valid_number n ↔ n ∈ ({592, 481, 370, 629, 518, 407} : Finset ℕ) :=
by
  sorry

#check valid_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_l918_91805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_K_sign_conditions_l918_91862

noncomputable def K (m x : ℝ) : ℝ :=
  ((m^2 + 2*m - 24) * x^2 - 6*(m + 6) * x + (m^2 + 10*m + 24)) /
  ((m^2 - 9*m + 18) * x^2 - 8*(m - 6) * x + (m^2 - 3*m - 18))

theorem K_sign_conditions (m : ℝ) :
  (∀ x, K m x > 0) = (abs m > 6) ∧
  (∀ x, K m x < 0) = (5 < abs m ∧ abs m < 6) := by
  sorry

#check K_sign_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_K_sign_conditions_l918_91862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_promenade_and_tornado_parliament_l918_91872

/-- Represents a deputy in the parliament -/
inductive Deputy
| Knight
| Liar
deriving DecidableEq

/-- The parliament of Promenade-and-Tornado -/
structure Parliament where
  members : Finset Deputy
  size_eq : members.card = 2020

/-- The statement made by some deputies -/
def statement (p : Parliament) (d : Deputy) : Prop :=
  2 * (p.members.filter (λ x => x ≠ d ∧ x = Deputy.Liar)).card > p.members.card - 1

/-- The number of deputies who made the statement -/
def num_statement_makers : Nat := 1011

/-- The theorem to be proved -/
theorem promenade_and_tornado_parliament 
  (p : Parliament) 
  (h : ∃ (s : Finset Deputy), s ⊆ p.members ∧ s.card = num_statement_makers ∧ 
       ∀ d ∈ s, (d = Deputy.Knight → statement p d) ∧ (d = Deputy.Liar → ¬statement p d)) :
  (p.members.filter (λ x => x = Deputy.Liar)).card = 1010 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_promenade_and_tornado_parliament_l918_91872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_B_speed_l918_91807

/-- The speed of student B in km/h -/
noncomputable def speed_B : ℝ := 12

/-- The distance from school to the activity location in km -/
noncomputable def distance : ℝ := 12

/-- The time difference of arrival in hours -/
noncomputable def time_diff : ℝ := 1/6

theorem student_B_speed :
  ∀ (speed_A : ℝ),
  speed_A = 1.2 * speed_B →
  distance / speed_B - distance / speed_A = time_diff →
  speed_B = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_B_speed_l918_91807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recycling_points_earned_l918_91850

/-- Conversion rate from kilograms to grams -/
def kg_to_g : ℚ := 1000

/-- Conversion rate from pounds to grams -/
def lb_to_g : ℚ := 453592 / 1000

/-- Conversion rate from pounds to recycling points -/
def lb_to_points : ℚ := 99

/-- Amount of paper Vanessa recycled in kilograms -/
def vanessa_kg : ℚ := 15

/-- Amount of paper Vanessa's first friend recycled in kilograms -/
def friend1_kg : ℚ := 12

/-- Amount of paper Vanessa's second friend recycled in kilograms -/
def friend2_kg : ℚ := 20

/-- Amount of paper Vanessa's third friend recycled in kilograms -/
def friend3_kg : ℚ := 7

/-- Amount of paper Vanessa's fourth friend recycled in grams -/
def friend4_g : ℚ := 25000

/-- Function to convert kilograms to pounds -/
noncomputable def kg_to_lb (x : ℚ) : ℚ := x * kg_to_g / lb_to_g

/-- Function to convert grams to pounds -/
noncomputable def g_to_lb (x : ℚ) : ℚ := x / lb_to_g

/-- Function to calculate recycling points from pounds -/
noncomputable def calculate_points (x : ℚ) : ℕ := (x / lb_to_points).floor.toNat

/-- Theorem stating that Vanessa and her friends earned 1 recycling point -/
theorem recycling_points_earned : 
  calculate_points (kg_to_lb vanessa_kg + kg_to_lb friend1_kg + kg_to_lb friend2_kg + 
                    kg_to_lb friend3_kg + g_to_lb friend4_g) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recycling_points_earned_l918_91850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_l918_91809

/-- The total surface area of a right pyramid with an equilateral triangular base -/
noncomputable def totalSurfaceArea (baseSide height : ℝ) : ℝ :=
  let baseArea := Real.sqrt 3 / 4 * baseSide ^ 2
  let slantHeight := Real.sqrt (height ^ 2 + (Real.sqrt 3 / 2 * baseSide) ^ 2)
  let lateralArea := 3 * (1 / 2 * baseSide * slantHeight)
  baseArea + lateralArea

/-- Theorem: The total surface area of a right pyramid with an equilateral triangular base
    of side length 20 cm and height 17 cm from the center of the base to the peak
    is 100√3 + 30√589 square centimeters. -/
theorem pyramid_surface_area :
  totalSurfaceArea 20 17 = 100 * Real.sqrt 3 + 30 * Real.sqrt 589 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_l918_91809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l918_91830

open Real

/-- The function f(x) defined as sin(x + π/3) + cos(x - π/6) -/
noncomputable def f (x : ℝ) : ℝ := sin (x + π/3) + cos (x - π/6)

/-- Theorem stating that the maximum value of f(x) is 2 -/
theorem max_value_of_f : ∃ M : ℝ, M = 2 ∧ ∀ x : ℝ, f x ≤ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l918_91830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_clearing_time_approx_7_63_l918_91869

-- Define the train properties
def train1_length : ℚ := 121
def train2_length : ℚ := 165
def train1_speed : ℚ := 80
def train2_speed : ℚ := 55

-- Define the function to calculate the time
noncomputable def calculate_clearing_time (l1 l2 s1 s2 : ℚ) : ℚ :=
  let relative_speed := (s1 + s2) * 1000 / 3600
  let total_distance := l1 + l2
  total_distance / relative_speed

-- Theorem statement
theorem trains_clearing_time_approx_7_63 :
  ∃ ε > 0, |calculate_clearing_time train1_length train2_length train1_speed train2_speed - 7.63| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_clearing_time_approx_7_63_l918_91869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_firecracker_explosion_speed_l918_91819

/-- The magnitude of the speed of the second fragment after a firecracker explosion -/
theorem firecracker_explosion_speed 
  (v_initial : ℝ) 
  (t_explosion : ℝ) 
  (g : ℝ) 
  (v_first_fragment : ℝ) 
  (v_second_fragment : ℝ) :
  v_initial = 20 →
  t_explosion = 1 →
  g = 10 →
  v_first_fragment = 48 →
  v_second_fragment = 52 →
  let v_explosion := v_initial - g * t_explosion
  let v_second_x := -v_first_fragment
  let v_second_y := 2 * v_explosion
  v_second_fragment = Real.sqrt (v_second_x^2 + v_second_y^2) := by
    sorry

#check firecracker_explosion_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_firecracker_explosion_speed_l918_91819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l918_91894

/-- The sum of the infinite series ∑(n=1 to ∞) (3n - 2) / (n(n + 1)(n + 3)) -/
noncomputable def infiniteSeries : ℝ := ∑' n, (3 * n - 2) / (n * (n + 1) * (n + 3))

/-- Theorem stating that the infinite series sums to -19/30 -/
theorem infiniteSeriesSum : infiniteSeries = -19/30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l918_91894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_junior_score_is_78_l918_91851

/-- Represents the score of students in a class -/
structure ClassScores where
  juniorPercentage : ℚ
  seniorPercentage : ℚ
  freshmanPercentage : ℚ
  overallAverage : ℚ
  seniorAverage : ℚ
  freshmanAverage : ℚ

/-- Calculates the junior score given the class scores -/
noncomputable def juniorScore (scores : ClassScores) : ℚ :=
  (scores.overallAverage * 100 - scores.seniorPercentage * scores.seniorAverage
    - scores.freshmanPercentage * scores.freshmanAverage) / scores.juniorPercentage

/-- Theorem stating that the junior score is 78 given the specific class scores -/
theorem junior_score_is_78 (scores : ClassScores)
  (h1 : scores.juniorPercentage = 30)
  (h2 : scores.seniorPercentage = 50)
  (h3 : scores.freshmanPercentage = 20)
  (h4 : scores.overallAverage = 80)
  (h5 : scores.seniorAverage = 85)
  (h6 : scores.freshmanAverage = 70) :
  juniorScore scores = 78 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_junior_score_is_78_l918_91851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_is_3_sqrt_2_l918_91857

-- Define the curves C1 and C2
noncomputable def C1 (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 2 * Real.sqrt 3 * Real.sin θ)

noncomputable def C2 (θ : ℝ) : ℝ × ℝ := 
  let ρ := 2 * Real.cos θ - 4 * Real.sin θ
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the point P
def P : ℝ × ℝ := (2, 0)

-- Define the line passing through P with slope 1
def line (x : ℝ) : ℝ := x - 2

-- Define the intersection points A and B
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (-1, -3)

-- Theorem statement
theorem length_AB_is_3_sqrt_2 : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 3 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_is_3_sqrt_2_l918_91857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_miquel_theorem_l918_91842

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the property of a point lying on a circle
variable (lies_on : Point → Circle → Prop)

-- Define the property of two circles intersecting at a point
variable (intersect_at : Circle → Circle → Point → Prop)

-- Define the property of a line passing through two points
variable (line_through : Point → Point → Prop)

-- Define the property of a line intersecting a circle at a point
variable (line_intersects_circle_at : Point → Point → Circle → Point → Prop)

-- Given conditions
variable (O O₁ O₂ O₃ : Circle)
variable (O A₁ A₂ A₃ B₁ B₂ B₃ B₄ : Point)

-- Theorem statement
theorem miquel_theorem 
  (h1 : lies_on O O₁ ∧ lies_on O O₂ ∧ lies_on O O₃)
  (h2 : intersect_at O₁ O₂ A₁ ∧ intersect_at O₂ O₃ A₂ ∧ intersect_at O₃ O₁ A₃)
  (h3 : lies_on B₁ O₁ ∧ B₁ ≠ A₁)
  (h4 : line_intersects_circle_at B₁ A₁ O₂ B₂ ∧ B₂ ≠ A₂)
  (h5 : line_intersects_circle_at B₂ A₂ O₃ B₃ ∧ B₃ ≠ A₃)
  (h6 : line_intersects_circle_at B₃ A₃ O₁ B₄)
  : B₄ = B₁ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_miquel_theorem_l918_91842
