import Mathlib

namespace min_value_of_expression_l882_88250

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_orthogonal : (x - 1) * 1 + 3 * y = 0) :
  (1 / x + 1 / (3 * y)) ≥ 4 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ (x - 1) * 1 + 3 * y = 0 ∧ 1 / x + 1 / (3 * y) = 4 := by
  sorry

end min_value_of_expression_l882_88250


namespace algorithm_design_properties_algorithm_not_endless_algorithm_not_unique_correct_statement_about_algorithms_l882_88237

/-- Represents the properties of an algorithm -/
structure Algorithm where
  finite : Bool
  clearlyDefined : Bool
  nonUnique : Bool
  simple : Bool
  convenient : Bool
  operable : Bool

/-- Defines the correct properties of an algorithm according to computer science -/
def correctAlgorithmProperties : Algorithm :=
  { finite := true
  , clearlyDefined := true
  , nonUnique := true
  , simple := true
  , convenient := true
  , operable := true }

/-- Theorem stating that algorithms should be designed to be simple, convenient, and operable -/
theorem algorithm_design_properties :
  (a : Algorithm) → a = correctAlgorithmProperties → a.simple ∧ a.convenient ∧ a.operable :=
by sorry

/-- Theorem stating that an algorithm cannot run endlessly -/
theorem algorithm_not_endless :
  (a : Algorithm) → a = correctAlgorithmProperties → a.finite :=
by sorry

/-- Theorem stating that there can be multiple algorithms for a task -/
theorem algorithm_not_unique :
  (a : Algorithm) → a = correctAlgorithmProperties → a.nonUnique :=
by sorry

/-- Main theorem proving that the statement about algorithm design properties is correct -/
theorem correct_statement_about_algorithms :
  ∃ (a : Algorithm), a = correctAlgorithmProperties ∧
    (a.simple ∧ a.convenient ∧ a.operable) ∧
    a.finite ∧
    a.nonUnique :=
by sorry

end algorithm_design_properties_algorithm_not_endless_algorithm_not_unique_correct_statement_about_algorithms_l882_88237


namespace first_satellite_launched_by_soviet_union_l882_88214

-- Define a type for countries
inductive Country
| UnitedStates
| SovietUnion
| EuropeanUnion
| Germany

-- Define a structure for a satellite launch event
structure SatelliteLaunch where
  date : Nat × Nat × Nat  -- (day, month, year)
  country : Country

-- Define the first artificial Earth satellite launch
def firstArtificialSatelliteLaunch : SatelliteLaunch :=
  { date := (4, 10, 1957),
    country := Country.SovietUnion }

-- Theorem statement
theorem first_satellite_launched_by_soviet_union :
  firstArtificialSatelliteLaunch.country = Country.SovietUnion :=
by sorry


end first_satellite_launched_by_soviet_union_l882_88214


namespace rectangle_area_l882_88228

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 36 →
  rectangle_width * rectangle_width = square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 108 := by
sorry

end rectangle_area_l882_88228


namespace binomial_divisibility_l882_88277

theorem binomial_divisibility (x n : ℕ) : 
  x = 5 → n = 4 → ∃ k : ℤ, (1 + x)^n - 1 = 7 * k := by
  sorry

end binomial_divisibility_l882_88277


namespace solution_set_and_minimum_t_l882_88223

/-- The set of all numerical values of the real number a -/
def M : Set ℝ := {a | ∀ x : ℝ, a * x^2 + a * x + 2 > 0}

theorem solution_set_and_minimum_t :
  (M = {a : ℝ | 0 ≤ a ∧ a < 4}) ∧
  (∃ t₀ : ℝ, t₀ > 0 ∧ ∀ t : ℝ, t > 0 → (∀ a ∈ M, (a^2 - 2*a) * t ≤ t^2 + 3*t - 46) → t ≥ t₀) ∧
  (∀ t : ℝ, t > 0 → (∀ a ∈ M, (a^2 - 2*a) * t ≤ t^2 + 3*t - 46) → t ≥ 46) :=
by sorry

#check solution_set_and_minimum_t

end solution_set_and_minimum_t_l882_88223


namespace student_take_home_pay_l882_88236

/-- Calculates the take-home pay for a well-performing student at a fast-food chain --/
def takeHomePay (baseSalary bonus taxRate : ℚ) : ℚ :=
  let totalEarnings := baseSalary + bonus
  let taxAmount := totalEarnings * taxRate
  totalEarnings - taxAmount

/-- Theorem stating that the take-home pay for a well-performing student is 26100 rubles --/
theorem student_take_home_pay :
  takeHomePay 25000 5000 (13/100) = 26100 := by
  sorry

#eval takeHomePay 25000 5000 (13/100)

end student_take_home_pay_l882_88236


namespace paper_cutting_theorem_smallest_over_2000_exactly_2005_exists_l882_88234

/-- Represents the number of pieces cut in each step -/
def CutSequence := List Nat

/-- Calculates the total number of pieces after a sequence of cuts -/
def totalPieces (cuts : CutSequence) : Nat :=
  1 + 4 * (1 + cuts.sum)

theorem paper_cutting_theorem (cuts : CutSequence) :
  ∃ (k : Nat), totalPieces cuts = 4 * k + 1 :=
sorry

theorem smallest_over_2000 :
  ∀ (cuts : CutSequence),
    totalPieces cuts > 2000 →
    totalPieces cuts ≥ 2005 :=
sorry

theorem exactly_2005_exists :
  ∃ (cuts : CutSequence), totalPieces cuts = 2005 :=
sorry

end paper_cutting_theorem_smallest_over_2000_exactly_2005_exists_l882_88234


namespace unicorn_tower_theorem_l882_88219

/-- Represents the configuration of a unicorn tethered to a cylindrical tower -/
structure UnicornTower where
  rope_length : ℝ
  tower_radius : ℝ
  unicorn_height : ℝ
  rope_end_distance : ℝ

/-- Calculates the length of rope touching the tower -/
def rope_touching_tower (ut : UnicornTower) : ℝ :=
  ut.rope_length - (ut.rope_end_distance + ut.tower_radius)

/-- Theorem stating the properties of the unicorn-tower configuration -/
theorem unicorn_tower_theorem (ut : UnicornTower) 
  (h_rope : ut.rope_length = 20)
  (h_radius : ut.tower_radius = 8)
  (h_height : ut.unicorn_height = 4)
  (h_distance : ut.rope_end_distance = 4) :
  ∃ (a b c : ℕ), 
    c.Prime ∧ 
    rope_touching_tower ut = (a : ℝ) - Real.sqrt b / c ∧
    a = 60 ∧ b = 750 ∧ c = 3 ∧
    a + b + c = 813 := by
  sorry

end unicorn_tower_theorem_l882_88219


namespace cube_face_sum_l882_88260

theorem cube_face_sum (a b c d e f : ℕ+) : 
  (a * b * c + a * e * c + a * b * f + a * e * f + 
   d * b * c + d * e * c + d * b * f + d * e * f = 1729) → 
  (a + b + c + d + e + f = 39) := by
sorry

end cube_face_sum_l882_88260


namespace outstanding_student_distribution_l882_88281

/-- The number of ways to distribute n indistinguishable objects into k distinguishable containers,
    with each container receiving at least one object. -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n - 1) (k - 1)

/-- Theorem stating that there are 126 ways to distribute 10 indistinguishable objects
    into 6 distinguishable containers, with each container receiving at least one object. -/
theorem outstanding_student_distribution : distribute 10 6 = 126 := by
  sorry

end outstanding_student_distribution_l882_88281


namespace floor_subtraction_inequality_l882_88220

theorem floor_subtraction_inequality (x y : ℝ) : 
  ⌊x - y⌋ ≤ ⌊x⌋ - ⌊y⌋ := by sorry

end floor_subtraction_inequality_l882_88220


namespace baseball_average_calculation_l882_88216

/-- Proves the required average for the remaining games to achieve a target season average -/
theorem baseball_average_calculation
  (total_games : ℕ)
  (completed_games : ℕ)
  (remaining_games : ℕ)
  (current_average : ℚ)
  (target_average : ℚ)
  (h_total : total_games = completed_games + remaining_games)
  (h_completed : completed_games = 20)
  (h_remaining : remaining_games = 10)
  (h_current : current_average = 2)
  (h_target : target_average = 3) :
  (target_average * total_games - current_average * completed_games) / remaining_games = 5 := by
  sorry

#check baseball_average_calculation

end baseball_average_calculation_l882_88216


namespace adjacent_knights_probability_l882_88275

def numKnights : ℕ := 30
def chosenKnights : ℕ := 4

def prob_adjacent_knights : ℚ :=
  1 - (Nat.choose (numKnights - chosenKnights + 1) (chosenKnights - 1) : ℚ) /
      (Nat.choose numKnights chosenKnights : ℚ)

theorem adjacent_knights_probability :
  prob_adjacent_knights = 5 / 11 := by
  sorry

end adjacent_knights_probability_l882_88275


namespace marble_count_theorem_l882_88241

/-- Represents the total number of marbles in a bag given the ratio of colors and the number of green marbles -/
def total_marbles (red blue green yellow : ℕ) (green_count : ℕ) : ℕ :=
  (red + blue + green + yellow) * green_count / green

/-- Theorem stating that given the specific ratio and number of green marbles, the total is 120 -/
theorem marble_count_theorem :
  total_marbles 1 3 2 4 24 = 120 := by
  sorry

end marble_count_theorem_l882_88241


namespace k_domain_l882_88291

noncomputable def k (x : ℝ) : ℝ := 1 / (x + 3) + 1 / (x^2 + 3) + 1 / (x^3 + 3)

def domain_k : Set ℝ := {x | x ≠ -3 ∧ x ≠ -Real.rpow 3 (1/3)}

theorem k_domain :
  {x : ℝ | ∃ y, k x = y} = domain_k :=
sorry

end k_domain_l882_88291


namespace triangle_problem_l882_88231

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Given condition
  (Real.cos C) / (Real.cos B) = (3 * a - c) / b →
  -- Part 1: Value of sin B
  Real.sin B = (2 * Real.sqrt 2) / 3 ∧
  -- Part 2: Area of triangle ABC when b = 4√2 and a = c
  (b = 4 * Real.sqrt 2 ∧ a = c →
    (1/2) * a * c * Real.sin B = 8 * Real.sqrt 2) := by
  sorry

end triangle_problem_l882_88231


namespace group_size_is_correct_l882_88266

/-- The number of people in a group where:
  1. The average weight increases by 2.5 kg when a new person joins.
  2. The person being replaced weighs 45 kg.
  3. The new person weighs 65 kg.
-/
def group_size : ℕ := 8

/-- The weight of the person being replaced -/
def original_weight : ℝ := 45

/-- The weight of the new person joining the group -/
def new_weight : ℝ := 65

/-- The increase in average weight when the new person joins -/
def average_increase : ℝ := 2.5

theorem group_size_is_correct : 
  (new_weight - original_weight) = (average_increase * group_size) :=
sorry

end group_size_is_correct_l882_88266


namespace identify_radioactive_balls_l882_88296

/-- A device that tests two balls for radioactivity -/
structure RadioactivityTester :=
  (test : Fin 100 → Fin 100 → Bool)
  (test_correct : ∀ a b, test a b = true ↔ (a.val < 51 ∧ b.val < 51))

/-- A strategy to identify radioactive balls -/
def IdentificationStrategy := RadioactivityTester → Fin 100 → Bool

/-- The number of tests performed by a strategy -/
def num_tests (strategy : IdentificationStrategy) (tester : RadioactivityTester) : ℕ :=
  sorry

theorem identify_radioactive_balls :
  ∃ (strategy : IdentificationStrategy),
    ∀ (tester : RadioactivityTester),
      (∀ i, strategy tester i = true ↔ i.val < 51) ∧
      num_tests strategy tester ≤ 145 :=
sorry

end identify_radioactive_balls_l882_88296


namespace quadratic_roots_properties_l882_88200

theorem quadratic_roots_properties (p : ℝ) (r₁ r₂ : ℝ) : 
  r₁ ≠ r₂ → 
  (r₁^2 + p*r₁ + 12 = 0) → 
  (r₂^2 + p*r₂ + 12 = 0) → 
  (|r₁ + r₂| > 5 ∧ |r₁ * r₂| > 4) := by
sorry

end quadratic_roots_properties_l882_88200


namespace geometric_sequence_sum_l882_88247

/-- Sum of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sequence_sum :
  let a : ℚ := 1/5
  let r : ℚ := 1/5
  let n : ℕ := 7
  geometric_sum a r n = 78124/312500 := by
sorry

end geometric_sequence_sum_l882_88247


namespace reciprocal_not_one_others_are_l882_88229

theorem reciprocal_not_one_others_are (x : ℝ) (hx : x = -1) : 
  (x⁻¹ ≠ 1) ∧ (-x = 1) ∧ (|x| = 1) ∧ (x^2 = 1) := by
  sorry

end reciprocal_not_one_others_are_l882_88229


namespace gcd_of_256_180_720_l882_88239

theorem gcd_of_256_180_720 : Nat.gcd 256 (Nat.gcd 180 720) = 36 := by
  sorry

end gcd_of_256_180_720_l882_88239


namespace sixth_term_of_arithmetic_sequence_l882_88279

/-- Given an arithmetic sequence where a₁ = -3 and a₂ = 1, prove that a₆ = 17 -/
theorem sixth_term_of_arithmetic_sequence : 
  ∀ (a : ℕ → ℤ), 
    a 1 = -3 → 
    a 2 = 1 → 
    (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) → 
    a 6 = 17 := by
  sorry

end sixth_term_of_arithmetic_sequence_l882_88279


namespace average_of_polynomials_l882_88263

theorem average_of_polynomials (x : ℚ) : 
  (1 / 3 : ℚ) * ((x^2 - 3*x + 2) + (3*x^2 + x - 1) + (2*x^2 - 5*x + 7)) = 2*x^2 + 4 →
  x = -4/7 := by
sorry

end average_of_polynomials_l882_88263


namespace circle_tangent_to_axes_on_line_l882_88272

theorem circle_tangent_to_axes_on_line (x y : ℝ) :
  ∃ (a b r : ℝ),
    (∀ t : ℝ, (2 * t - (2 * t + 6) + 6 = 0)) →  -- Center on the line 2x - y + 6 = 0
    (a = -2 ∨ a = -6) →                         -- Possible x-coordinates of the center
    (b = 2 * a + 6) →                           -- y-coordinate of the center
    (r = |a|) →                                 -- Radius equals the absolute value of x-coordinate
    (r = |b|) →                                 -- Radius equals the absolute value of y-coordinate
    ((x + a)^2 + (y - b)^2 = r^2) →             -- Standard form of circle equation
    (((x + 2)^2 + (y - 2)^2 = 4) ∨ ((x + 6)^2 + (y + 6)^2 = 36)) :=
by sorry

end circle_tangent_to_axes_on_line_l882_88272


namespace repeating_decimal_subtraction_l882_88273

/-- Represents a repeating decimal with a 4-digit repetend -/
def RepeatingDecimal (a b c d : ℕ) : ℚ :=
  (a * 1000 + b * 100 + c * 10 + d) / 9999

theorem repeating_decimal_subtraction :
  RepeatingDecimal 2 3 4 5 - RepeatingDecimal 6 7 8 9 - RepeatingDecimal 1 2 3 4 = -5678 / 9999 := by
  sorry

end repeating_decimal_subtraction_l882_88273


namespace floor_of_expression_equals_eight_l882_88243

theorem floor_of_expression_equals_eight :
  ⌊(2005^3 : ℝ) / (2003 * 2004) - (2003^3 : ℝ) / (2004 * 2005)⌋ = 8 := by
  sorry

end floor_of_expression_equals_eight_l882_88243


namespace simplify_trig_fraction_l882_88240

theorem simplify_trig_fraction (x : Real) :
  let u := Real.sin (x/2) * (Real.cos (x/2) + Real.sin (x/2))
  (2 - Real.sin x + Real.cos x) / (2 + Real.sin x - Real.cos x) = (3 - 2*u) / (1 + 2*u) := by
  sorry

end simplify_trig_fraction_l882_88240


namespace triangle_properties_l882_88295

open Real

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) :
  (t.a * cos t.C + t.c * cos t.A = 2 * t.b * cos t.A) →
  (t.A = π / 3) ∧
  (t.a = Real.sqrt 7 ∧ t.b = 2 →
    (1/2 * t.b * t.c * sin t.A = (3 * Real.sqrt 3) / 2)) :=
by sorry

end triangle_properties_l882_88295


namespace S_4_S_n_l882_88255

-- Define N(n) as the largest odd factor of n
def N (n : ℕ+) : ℕ := sorry

-- Define S(n) as the sum of N(k) from k=1 to 2^n
def S (n : ℕ) : ℕ := sorry

-- Theorem for S(4)
theorem S_4 : S 4 = 86 := by sorry

-- Theorem for S(n)
theorem S_n (n : ℕ) : S n = (4^n + 2) / 3 := by sorry

end S_4_S_n_l882_88255


namespace no_solution_for_equation_l882_88245

theorem no_solution_for_equation : ¬∃ (a b : ℕ), 2 * a^2 + 1 = 4 * b^2 := by sorry

end no_solution_for_equation_l882_88245


namespace arnel_friends_count_l882_88270

/-- Represents the pencil sharing problem --/
def pencil_sharing (num_boxes : ℕ) (pencils_per_box : ℕ) (kept_pencils : ℕ) (pencils_per_friend : ℕ) : ℕ :=
  let total_pencils := num_boxes * pencils_per_box
  let shared_pencils := total_pencils - kept_pencils
  shared_pencils / pencils_per_friend

/-- Proves that Arnel shared pencils with 5 friends --/
theorem arnel_friends_count :
  pencil_sharing 10 5 10 8 = 5 := by
  sorry

end arnel_friends_count_l882_88270


namespace no_solution_equation_l882_88264

theorem no_solution_equation : ¬∃ (x : ℝ), (8 / (x^2 - 4) + 1 = x / (x - 2)) := by
  sorry

end no_solution_equation_l882_88264


namespace grant_total_earnings_l882_88244

/-- Grant's earnings over four months as a freelance math worker -/
def grant_earnings (X Y Z W : ℕ) : ℕ :=
  let month1 := X
  let month2 := 3 * X + Y
  let month3 := 2 * month2 - Z
  let month4 := (month1 + month2 + month3) / 3 + W
  month1 + month2 + month3 + month4

/-- Theorem stating Grant's total earnings over four months -/
theorem grant_total_earnings :
  grant_earnings 350 30 20 50 = 5810 := by
  sorry

end grant_total_earnings_l882_88244


namespace exradii_product_bound_l882_88210

/-- For any triangle with side lengths a, b, c and exradii r_a, r_b, r_c,
    the product of the exradii does not exceed (3√3/8) times the product of the side lengths. -/
theorem exradii_product_bound (a b c r_a r_b r_c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hr_a : r_a > 0) (hr_b : r_b > 0) (hr_c : r_c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_exradii : r_a * (b + c - a) = r_b * (c + a - b) ∧ 
               r_b * (c + a - b) = r_c * (a + b - c)) : 
  r_a * r_b * r_c ≤ (3 * Real.sqrt 3 / 8) * a * b * c := by
  sorry

#check exradii_product_bound

end exradii_product_bound_l882_88210


namespace yellow_balls_count_l882_88232

theorem yellow_balls_count (total : ℕ) (yellow : ℕ) (h1 : total = 15) 
  (h2 : yellow ≤ total) 
  (h3 : (yellow : ℚ) / total * (yellow - 1) / (total - 1) = 1 / 21) : 
  yellow = 5 := by sorry

end yellow_balls_count_l882_88232


namespace parrot_fraction_l882_88246

theorem parrot_fraction (p t : ℝ) : 
  p + t = 1 →                     -- Total fraction of birds
  (2/3 : ℝ) * p + (1/4 : ℝ) * t = (1/2 : ℝ) →  -- Male birds equation
  p = (3/5 : ℝ) := by             -- Fraction of parrots
sorry

end parrot_fraction_l882_88246


namespace quadratic_equation_unique_solution_l882_88201

theorem quadratic_equation_unique_solution (a c : ℝ) : 
  (∃! x : ℝ, a * x^2 + 10 * x + c = 0) →
  a + c = 17 →
  a > c →
  a = 15.375 ∧ c = 1.625 :=
by sorry

end quadratic_equation_unique_solution_l882_88201


namespace parabola_intersection_difference_l882_88257

theorem parabola_intersection_difference (a b c d : ℝ) : 
  (∀ x, 3 * x^2 - 6 * x + 6 = -2 * x^2 - 4 * x + 6 → x = a ∨ x = c) →
  (3 * a^2 - 6 * a + 6 = -2 * a^2 - 4 * a + 6) →
  (3 * c^2 - 6 * c + 6 = -2 * c^2 - 4 * c + 6) →
  c ≥ a →
  c - a = 2/5 := by
sorry

end parabola_intersection_difference_l882_88257


namespace unique_number_property_l882_88285

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 5 := by
  sorry

end unique_number_property_l882_88285


namespace sampling_appropriate_l882_88217

/-- Represents methods of investigation -/
inductive InvestigationMethod
  | Sampling
  | Comprehensive
  | Other

/-- Represents the characteristics of an investigation -/
structure InvestigationCharacteristics where
  isElectronicProduct : Bool
  largeVolume : Bool
  needComprehensive : Bool

/-- Determines the appropriate investigation method based on given characteristics -/
def appropriateMethod (chars : InvestigationCharacteristics) : InvestigationMethod :=
  sorry

/-- Theorem stating that sampling investigation is appropriate for the given conditions -/
theorem sampling_appropriate (chars : InvestigationCharacteristics)
  (h1 : chars.isElectronicProduct = true)
  (h2 : chars.largeVolume = true)
  (h3 : chars.needComprehensive = false) :
  appropriateMethod chars = InvestigationMethod.Sampling :=
sorry

end sampling_appropriate_l882_88217


namespace sphere_equation_l882_88252

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Definition of a sphere in 3D space -/
def Sphere (center : Point3D) (radius : ℝ) : Set Point3D :=
  {p : Point3D | (p.x - center.x)^2 + (p.y - center.y)^2 + (p.z - center.z)^2 = radius^2}

/-- Theorem: The equation (x - x₀)² + (y - y₀)² + (z - z₀)² = r² represents a sphere
    with center (x₀, y₀, z₀) and radius r in a three-dimensional Cartesian coordinate system -/
theorem sphere_equation (center : Point3D) (radius : ℝ) :
  Sphere center radius = {p : Point3D | (p.x - center.x)^2 + (p.y - center.y)^2 + (p.z - center.z)^2 = radius^2} := by
  sorry

end sphere_equation_l882_88252


namespace arrange_five_classes_four_factories_l882_88208

/-- The number of ways to arrange classes into factories -/
def arrange_classes (num_classes : ℕ) (num_factories : ℕ) : ℕ :=
  (num_classes.choose 2) * (num_factories.factorial)

/-- Theorem: The number of ways to arrange 5 classes into 4 factories is 240 -/
theorem arrange_five_classes_four_factories :
  arrange_classes 5 4 = 240 := by
  sorry

end arrange_five_classes_four_factories_l882_88208


namespace high_school_students_l882_88258

theorem high_school_students (m j : ℕ) : 
  m = 4 * j →  -- Maria's school has 4 times as many students as Javier's
  m + j = 2500 →  -- Total students in both schools
  m = 2000 :=  -- Prove that Maria's school has 2000 students
by
  sorry

end high_school_students_l882_88258


namespace autumn_pencils_left_l882_88276

/-- Calculates the number of pencils Autumn has left after various changes --/
def pencils_left (initial : ℕ) (misplaced : ℕ) (broken : ℕ) (found : ℕ) (bought : ℕ) : ℕ :=
  initial - (misplaced + broken) + (found + bought)

/-- Theorem stating that Autumn has 16 pencils left --/
theorem autumn_pencils_left : pencils_left 20 7 3 4 2 = 16 := by
  sorry

end autumn_pencils_left_l882_88276


namespace freds_baseball_cards_l882_88262

/-- Fred's baseball card problem -/
theorem freds_baseball_cards (initial_cards : ℕ) (cards_bought : ℕ) (remaining_cards : ℕ) : 
  initial_cards = 40 → cards_bought = 22 → remaining_cards = initial_cards - cards_bought → 
  remaining_cards = 18 := by
  sorry

end freds_baseball_cards_l882_88262


namespace stratified_sample_size_l882_88207

/-- Represents the population groups in the organization -/
structure Population where
  elderly : Nat
  middleAged : Nat
  young : Nat

/-- Represents a stratified sample -/
structure StratifiedSample where
  total : Nat
  young : Nat

/-- Theorem stating the relationship between the population, sample, and sample size -/
theorem stratified_sample_size 
  (pop : Population)
  (sample : StratifiedSample)
  (h1 : pop.elderly = 20)
  (h2 : pop.middleAged = 120)
  (h3 : pop.young = 100)
  (h4 : sample.young = 10) :
  sample.total = 24 := by
  sorry

#check stratified_sample_size

end stratified_sample_size_l882_88207


namespace grid_coloring_l882_88282

theorem grid_coloring (n : ℕ) (k : ℕ) (h_n_pos : 0 < n) (h_k_bound : k < n^2) :
  (4 * n * k - 2 * n^3 = 50) → (k = 15 ∨ k = 313) := by
  sorry

end grid_coloring_l882_88282


namespace arithmetic_sequence_sum_l882_88286

/-- An arithmetic sequence with positive common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_positive : d > 0
  h_arithmetic : ∀ n, a (n + 1) = a n + d

/-- The sum of the first three terms of the sequence is 15 -/
def sum_first_three (seq : ArithmeticSequence) : Prop :=
  seq.a 1 + seq.a 2 + seq.a 3 = 15

/-- The product of the first three terms of the sequence is 80 -/
def product_first_three (seq : ArithmeticSequence) : Prop :=
  seq.a 1 * seq.a 2 * seq.a 3 = 80

/-- Theorem: If the sum of the first three terms is 15 and their product is 80,
    then the sum of the 11th, 12th, and 13th terms is 135 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence)
  (h_sum : sum_first_three seq) (h_product : product_first_three seq) :
  seq.a 11 + seq.a 12 + seq.a 13 = 135 := by
  sorry

end arithmetic_sequence_sum_l882_88286


namespace two_true_propositions_l882_88238

theorem two_true_propositions (p q : Prop) (h : p ∧ q) :
  (p ∨ q) ∧ p ∧ ¬(¬q) ∧ ¬((¬p) ∨ (¬q)) :=
by sorry

end two_true_propositions_l882_88238


namespace purple_top_implies_violet_bottom_l882_88224

/-- Represents the colors of the cube faces -/
inductive Color
  | R | P | O | Y | G | V

/-- Represents a cube with colored faces -/
structure Cube where
  top : Color
  bottom : Color
  front : Color
  back : Color
  left : Color
  right : Color

/-- Represents the configuration of the six squares before folding -/
structure SquareConfiguration where
  square1 : Color
  square2 : Color
  square3 : Color
  square4 : Color
  square5 : Color
  square6 : Color

/-- Function to fold the squares into a cube -/
def foldIntoCube (config : SquareConfiguration) : Cube :=
  sorry

/-- Theorem stating that if P is on top, V is on the bottom -/
theorem purple_top_implies_violet_bottom (config : SquareConfiguration) :
  let cube := foldIntoCube config
  cube.top = Color.P → cube.bottom = Color.V :=
sorry

end purple_top_implies_violet_bottom_l882_88224


namespace solve_for_A_l882_88249

/-- Given that 3ab · A = 6a²b - 9ab², prove that A = 2a - 3b -/
theorem solve_for_A (a b A : ℝ) (h : 3 * a * b * A = 6 * a^2 * b - 9 * a * b^2) :
  A = 2 * a - 3 * b := by
sorry

end solve_for_A_l882_88249


namespace power_function_through_point_l882_88284

-- Define the power function
def f (x : ℝ) : ℝ := x^(1/3)

-- State the theorem
theorem power_function_through_point (h : f 27 = 3) : f 8 = 2 := by
  sorry

end power_function_through_point_l882_88284


namespace license_plate_increase_l882_88293

-- Define the number of possible letters and digits
def num_letters : ℕ := 26
def num_digits : ℕ := 10

-- Define the number of letters and digits in old and new license plates
def old_num_letters : ℕ := 2
def old_num_digits : ℕ := 3
def new_num_letters : ℕ := 2
def new_num_digits : ℕ := 4

-- Calculate the number of possible old and new license plates
def num_old_plates : ℕ := num_letters^old_num_letters * num_digits^old_num_digits
def num_new_plates : ℕ := num_letters^new_num_letters * num_digits^new_num_digits

-- Theorem: The ratio of new to old license plates is 10
theorem license_plate_increase : num_new_plates / num_old_plates = 10 := by
  sorry

end license_plate_increase_l882_88293


namespace arithmetic_sequence_properties_l882_88205

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := 2 * n - 1

-- Define S_n as the sum of first n terms of a_n
def S (n : ℕ) : ℚ := n * (a 1 + a n) / 2

-- Define b_n
def b (n : ℕ) : ℚ := 1 / (a n * a (n + 1))

-- Define T_n as the sum of first n terms of b_n
def T (n : ℕ) : ℚ := n / (2 * n + 1)

theorem arithmetic_sequence_properties (n : ℕ) :
  (S 3 = a 4 + 2) ∧ 
  (a 3 ^ 2 = a 1 * a 13) ∧ 
  (∀ k : ℕ, a (k + 1) - a k = a 2 - a 1) ∧
  (∀ k : ℕ, b k = 1 / (a k * a (k + 1))) ∧
  (∀ k : ℕ, T k = k / (2 * k + 1)) := by
  sorry

#check arithmetic_sequence_properties

end arithmetic_sequence_properties_l882_88205


namespace card_sum_proof_l882_88213

theorem card_sum_proof (H S D C : ℕ) : 
  (∃ (h₁ h₂ : ℕ), h₁ + h₂ = H ∧ h₁ ≥ 1 ∧ h₂ ≥ 1 ∧ h₁ ≤ 13 ∧ h₂ ≤ 13) →
  (∃ (s₁ s₂ s₃ : ℕ), s₁ + s₂ + s₃ = S ∧ s₁ ≥ 1 ∧ s₂ ≥ 1 ∧ s₃ ≥ 1 ∧ s₁ ≤ 13 ∧ s₂ ≤ 13 ∧ s₃ ≤ 13) →
  (∃ (d₁ d₂ d₃ d₄ : ℕ), d₁ + d₂ + d₃ + d₄ = D ∧ d₁ ≥ 1 ∧ d₂ ≥ 1 ∧ d₃ ≥ 1 ∧ d₄ ≥ 1 ∧ d₁ ≤ 13 ∧ d₂ ≤ 13 ∧ d₃ ≤ 13 ∧ d₄ ≤ 13) →
  (∃ (c₁ c₂ c₃ c₄ c₅ : ℕ), c₁ + c₂ + c₃ + c₄ + c₅ = C ∧ c₁ ≥ 1 ∧ c₂ ≥ 1 ∧ c₃ ≥ 1 ∧ c₄ ≥ 1 ∧ c₅ ≥ 1 ∧ c₁ ≤ 13 ∧ c₂ ≤ 13 ∧ c₃ ≤ 13 ∧ c₄ ≤ 13 ∧ c₅ ≤ 13) →
  S = 11 * H →
  C = D + 45 →
  H + S + D + C = 101 :=
by
  sorry

end card_sum_proof_l882_88213


namespace geometric_sequence_general_term_l882_88222

/-- Given a geometric sequence {a_n} where the first three terms are a-2, a+2, and a+8,
    prove that the general term a_n is equal to 8 · (3/2)^(n-1) -/
theorem geometric_sequence_general_term (a : ℝ) (a_n : ℕ → ℝ) :
  (a_n 1 = a - 2) →
  (a_n 2 = a + 2) →
  (a_n 3 = a + 8) →
  (∀ n : ℕ, n ≥ 1 → a_n (n + 1) / a_n n = a_n 2 / a_n 1) →
  (∀ n : ℕ, n ≥ 1 → a_n n = 8 * (3/2)^(n - 1)) :=
by sorry

end geometric_sequence_general_term_l882_88222


namespace impossible_c_value_l882_88283

theorem impossible_c_value (a b c : ℤ) : 
  (∀ x : ℝ, (x + a) * (x + b) = x^2 + c*x - 8) → c ≠ 4 := by
sorry

end impossible_c_value_l882_88283


namespace rectangle_dimensions_theorem_l882_88256

def rectangle_dimensions (w : ℝ) : Prop :=
  let l := w + 3
  let perimeter := 2 * (w + l)
  let area := w * l
  perimeter = 2 * area ∧ w > 0 ∧ l > 0 → w = 1 ∧ l = 4

theorem rectangle_dimensions_theorem :
  ∃ w : ℝ, rectangle_dimensions w := by
  sorry

end rectangle_dimensions_theorem_l882_88256


namespace basketball_team_selection_count_l882_88280

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose 7 starters from a team of 16 players,
    including a set of 4 quadruplets, where exactly 3 of the quadruplets
    must be in the starting lineup -/
def basketball_team_selection : ℕ :=
  let total_players : ℕ := 16
  let quadruplets : ℕ := 4
  let starters : ℕ := 7
  let quadruplets_in_lineup : ℕ := 3
  (choose quadruplets quadruplets_in_lineup) *
  (choose (total_players - quadruplets) (starters - quadruplets_in_lineup))

theorem basketball_team_selection_count :
  basketball_team_selection = 1980 := by sorry

end basketball_team_selection_count_l882_88280


namespace constant_term_expansion_l882_88215

theorem constant_term_expansion (x : ℝ) : 
  let expression := (Real.sqrt x + 2) * (1 / Real.sqrt x - 1)^5
  ∃ (p : ℝ → ℝ), expression = p x ∧ p 0 = 3 :=
by sorry

end constant_term_expansion_l882_88215


namespace quadratic_root_property_l882_88278

theorem quadratic_root_property (a : ℝ) : 
  a^2 - a - 50 = 0 → a^4 - 101*a = 2550 := by
  sorry

end quadratic_root_property_l882_88278


namespace custom_op_M_T_l882_88290

def custom_op (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∪ B ∧ x ∉ A ∩ B}

def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}
def T : Set ℝ := {x | x < 2}

theorem custom_op_M_T :
  custom_op M T = {x | x < -1 ∨ (2 ≤ x ∧ x ≤ 4)} :=
by sorry

end custom_op_M_T_l882_88290


namespace cherry_weekly_earnings_l882_88211

/-- Represents the charge for a cargo based on its weight range -/
def charge (weight : ℕ) : ℚ :=
  if 3 ≤ weight ∧ weight ≤ 5 then 5/2
  else if 6 ≤ weight ∧ weight ≤ 8 then 4
  else if 9 ≤ weight ∧ weight ≤ 12 then 6
  else if 13 ≤ weight ∧ weight ≤ 15 then 8
  else 0

/-- Calculates the daily earnings based on the number of deliveries for each weight -/
def dailyEarnings (deliveries : List (ℕ × ℕ)) : ℚ :=
  deliveries.foldl (fun acc (weight, count) => acc + charge weight * count) 0

/-- Cherry's daily delivery schedule -/
def cherryDeliveries : List (ℕ × ℕ) := [(5, 4), (8, 2), (10, 3), (14, 1)]

/-- Number of days in a week -/
def daysInWeek : ℕ := 7

/-- Theorem stating that Cherry's weekly earnings equal $308 -/
theorem cherry_weekly_earnings : 
  dailyEarnings cherryDeliveries * daysInWeek = 308 := by
  sorry

end cherry_weekly_earnings_l882_88211


namespace gcd_cube_plus_sixteen_and_plus_four_l882_88289

theorem gcd_cube_plus_sixteen_and_plus_four (n : ℕ) (h : n > 2^4) :
  Nat.gcd (n^3 + 4^2) (n + 4) = 1 := by
  sorry

end gcd_cube_plus_sixteen_and_plus_four_l882_88289


namespace solution_set_theorem_l882_88221

open Set

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : f (-2) = 2013)
variable (h2 : ∀ x : ℝ, deriv f x < 2 * x)

-- Define the solution set
def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x : ℝ | f x > x^2 + 2009}

-- State the theorem
theorem solution_set_theorem (f : ℝ → ℝ) (h1 : f (-2) = 2013) (h2 : ∀ x : ℝ, deriv f x < 2 * x) :
  solution_set f = Iio (-2) :=
sorry

end solution_set_theorem_l882_88221


namespace right_triangle_existence_l882_88206

theorem right_triangle_existence (p q : ℝ) (hp : p > 0) (hq : q > 0) : 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  c + a = p ∧ c + b = q ∧ a^2 + b^2 = c^2 := by
  sorry

end right_triangle_existence_l882_88206


namespace sqrt_19_bounds_l882_88226

theorem sqrt_19_bounds : 4 < Real.sqrt 19 ∧ Real.sqrt 19 < 5 := by
  have h1 : 16 < 19 := by sorry
  have h2 : 19 < 25 := by sorry
  sorry

end sqrt_19_bounds_l882_88226


namespace red_to_yellow_ratio_l882_88259

/-- Represents the number of mugs of each color in Hannah's collection. -/
structure MugCollection where
  red : ℕ
  blue : ℕ
  yellow : ℕ
  other : ℕ

/-- Checks if a mug collection satisfies Hannah's conditions. -/
def isValidCollection (m : MugCollection) : Prop :=
  m.red + m.blue + m.yellow + m.other = 40 ∧
  m.blue = 3 * m.red ∧
  m.yellow = 12 ∧
  m.other = 4

/-- Theorem stating that for any valid mug collection, the ratio of red to yellow mugs is 1:2. -/
theorem red_to_yellow_ratio (m : MugCollection) (h : isValidCollection m) :
  m.red * 2 = m.yellow := by sorry

end red_to_yellow_ratio_l882_88259


namespace inverse_proposition_l882_88299

-- Define the original proposition
def corresponding_angles_equal : Prop := sorry

-- Define the inverse proposition
def equal_angles_corresponding : Prop := sorry

-- Theorem stating that equal_angles_corresponding is the inverse of corresponding_angles_equal
theorem inverse_proposition : 
  (corresponding_angles_equal → equal_angles_corresponding) ∧ 
  (equal_angles_corresponding → corresponding_angles_equal) := by sorry

end inverse_proposition_l882_88299


namespace final_salary_correct_l882_88298

/-- Calculates the final salary after a series of changes -/
def calculate_final_salary (initial_salary : ℝ) (raise_percentage : ℝ) (cut_percentage : ℝ) (deduction : ℝ) : ℝ :=
  let salary_after_raise := initial_salary * (1 + raise_percentage)
  let salary_after_cut := salary_after_raise * (1 - cut_percentage)
  salary_after_cut - deduction

/-- Theorem stating that the final salary matches the expected value -/
theorem final_salary_correct (initial_salary : ℝ) (raise_percentage : ℝ) (cut_percentage : ℝ) (deduction : ℝ) 
    (h1 : initial_salary = 3000)
    (h2 : raise_percentage = 0.1)
    (h3 : cut_percentage = 0.15)
    (h4 : deduction = 100) :
  calculate_final_salary initial_salary raise_percentage cut_percentage deduction = 2705 := by
  sorry

end final_salary_correct_l882_88298


namespace a_range_l882_88269

theorem a_range (a : ℝ) (ha : a > 0) 
  (h : ∀ x : ℝ, x > 0 → 9*x + a^2/x ≥ a^2 + 8) : 
  2 ≤ a ∧ a ≤ 4 := by
sorry

end a_range_l882_88269


namespace fraction_comparison_l882_88287

theorem fraction_comparison (n : ℕ) (hn : n > 0) :
  (n + 1 : ℝ) ^ (n + 3) / (n + 3 : ℝ) ^ (n + 1) > n ^ (n + 2) / (n + 2 : ℝ) ^ n :=
by sorry

end fraction_comparison_l882_88287


namespace negation_of_existence_proposition_l882_88212

theorem negation_of_existence_proposition :
  (¬ ∃ (c : ℝ), c > 0 ∧ ∃ (x : ℝ), x^2 - x + c = 0) ↔
  (∀ (c : ℝ), c > 0 → ¬ ∃ (x : ℝ), x^2 - x + c = 0) :=
by sorry

end negation_of_existence_proposition_l882_88212


namespace equal_area_divide_sum_of_squares_l882_88235

-- Define the region S as a set of points in the plane
def S : Set (ℝ × ℝ) := sorry

-- Define the line m with slope 4
def m : Set (ℝ × ℝ) := {(x, y) | 4 * x = y + c} where c : ℝ := sorry

-- Define the property of m dividing S into two equal areas
def divides_equally (l : Set (ℝ × ℝ)) (r : Set (ℝ × ℝ)) : Prop := sorry

-- Define the equation of line m in the form ax = by + c
def line_equation (a b c : ℕ) : Set (ℝ × ℝ) := {(x, y) | a * x = b * y + c}

-- Main theorem
theorem equal_area_divide_sum_of_squares :
  ∃ (a b c : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    Nat.gcd a (Nat.gcd b c) = 1 ∧
    divides_equally (line_equation a b c) S ∧
    m = line_equation a b c ∧
    a^2 + b^2 + c^2 = 65 := by sorry

end equal_area_divide_sum_of_squares_l882_88235


namespace spinner_probability_l882_88230

theorem spinner_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 1/4 → p_B = 1/3 → p_D = 1/6 → p_A + p_B + p_C + p_D = 1 → p_C = 1/4 :=
by sorry

end spinner_probability_l882_88230


namespace min_distance_curve_line_l882_88271

/-- The minimum distance between a point on y = 2ln(x) and a point on y = 2x + 3 is √5 -/
theorem min_distance_curve_line : 
  let curve := (fun x : ℝ => 2 * Real.log x)
  let line := (fun x : ℝ => 2 * x + 3)
  ∃ (M N : ℝ × ℝ), 
    (M.2 = curve M.1) ∧ 
    (N.2 = line N.1) ∧
    (∀ (P Q : ℝ × ℝ), P.2 = curve P.1 → Q.2 = line Q.1 → 
      Real.sqrt 5 ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)) ∧
    Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = Real.sqrt 5 :=
sorry

end min_distance_curve_line_l882_88271


namespace quadrilateral_area_relation_l882_88227

-- Define the quadrilateral ABCD
variable (A B C D : ℝ × ℝ)

-- Define the intersection point of diagonals
def O : ℝ × ℝ := sorry

-- Define a point P inside triangle AOB
variable (P : ℝ × ℝ)

-- Assume P is inside triangle AOB
axiom P_inside_AOB : sorry

-- Define the area function for triangles
def area (X Y Z : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem quadrilateral_area_relation :
  area P C D - area P A B = area P A C + area P B D := by sorry

end quadrilateral_area_relation_l882_88227


namespace consecutive_terms_iff_equation_l882_88254

/-- Sequence definition -/
def a : ℕ → ℕ → ℕ
  | m, 0 => 0
  | m, 1 => 1
  | m, k + 2 => m * a m (k + 1) - a m k

/-- Main theorem -/
theorem consecutive_terms_iff_equation (m : ℕ) :
  ∀ x y : ℕ, x^2 - m*x*y + y^2 = 1 ↔ ∃ k : ℕ, x = a m k ∧ y = a m (k + 1) :=
by sorry

end consecutive_terms_iff_equation_l882_88254


namespace factorization_x_squared_minus_9x_l882_88253

theorem factorization_x_squared_minus_9x (x : ℝ) : x^2 - 9*x = x*(x - 9) := by
  sorry

end factorization_x_squared_minus_9x_l882_88253


namespace condition_analysis_l882_88233

theorem condition_analysis (a b c : ℝ) : 
  (∀ a b c : ℝ, a * c^2 < b * c^2 → a < b) ∧ 
  (∃ a b c : ℝ, a < b ∧ a * c^2 ≥ b * c^2) :=
by sorry

end condition_analysis_l882_88233


namespace insects_distribution_l882_88242

/-- The number of insects collected by boys -/
def boys_insects : ℕ := 200

/-- The number of insects collected by girls -/
def girls_insects : ℕ := 300

/-- The number of groups the class is divided into -/
def num_groups : ℕ := 4

/-- The total number of insects collected -/
def total_insects : ℕ := boys_insects + girls_insects

/-- The number of insects per group -/
def insects_per_group : ℕ := total_insects / num_groups

theorem insects_distribution :
  insects_per_group = 125 := by sorry

end insects_distribution_l882_88242


namespace magnitude_of_linear_combination_l882_88248

/-- Given two planar unit vectors with a right angle between them, 
    prove that the magnitude of 3 times the first vector plus 4 times the second vector is 5. -/
theorem magnitude_of_linear_combination (m n : ℝ × ℝ) : 
  ‖m‖ = 1 → ‖n‖ = 1 → m • n = 0 → ‖3 • m + 4 • n‖ = 5 := by
  sorry

end magnitude_of_linear_combination_l882_88248


namespace darryl_honeydews_l882_88225

/-- Represents the problem of determining the initial number of honeydews --/
def honeydew_problem (initial_cantaloupes : ℕ) (final_cantaloupes : ℕ) (final_honeydews : ℕ)
  (dropped_cantaloupes : ℕ) (rotten_honeydews : ℕ) (cantaloupe_price : ℕ) (honeydew_price : ℕ)
  (total_revenue : ℕ) : Prop :=
  ∃ (initial_honeydews : ℕ),
    -- Revenue calculation
    (initial_cantaloupes - dropped_cantaloupes - final_cantaloupes) * cantaloupe_price +
    (initial_honeydews - rotten_honeydews - final_honeydews) * honeydew_price = total_revenue

theorem darryl_honeydews :
  honeydew_problem 30 8 9 2 3 2 3 85 →
  ∃ (initial_honeydews : ℕ), initial_honeydews = 27 :=
sorry

end darryl_honeydews_l882_88225


namespace infinite_geometric_series_first_term_l882_88203

theorem infinite_geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (h_r : r = 1/4)
  (h_S : S = 80)
  (h_sum : S = a / (1 - r))
  : a = 60 := by
  sorry

end infinite_geometric_series_first_term_l882_88203


namespace geometric_series_relation_l882_88251

/-- Given two infinite geometric series:
    Series I with first term a₁ = 12 and common ratio r₁ = 1/3
    Series II with first term a₂ = 12 and common ratio r₂ = (4+n)/12
    If the sum of Series II is five times the sum of Series I, then n = 152 -/
theorem geometric_series_relation (n : ℝ) : 
  let a₁ : ℝ := 12
  let r₁ : ℝ := 1/3
  let a₂ : ℝ := 12
  let r₂ : ℝ := (4+n)/12
  (a₁ / (1 - r₁) = a₂ / (1 - r₂) / 5) → n = 152 := by
  sorry

end geometric_series_relation_l882_88251


namespace orange_percentage_l882_88292

/-- Given a box of fruit with initial oranges and kiwis, and additional kiwis added,
    calculate the percentage of oranges in the final mixture. -/
theorem orange_percentage
  (initial_oranges : ℕ)
  (initial_kiwis : ℕ)
  (added_kiwis : ℕ)
  (h1 : initial_oranges = 24)
  (h2 : initial_kiwis = 30)
  (h3 : added_kiwis = 26) :
  (initial_oranges : ℚ) / (initial_oranges + initial_kiwis + added_kiwis) * 100 = 30 := by
  sorry

end orange_percentage_l882_88292


namespace inheritance_calculation_l882_88268

theorem inheritance_calculation (x : ℝ) : 
  let after_charity := 0.95 * x
  let federal_tax := 0.25 * after_charity
  let after_federal := after_charity - federal_tax
  let state_tax := 0.12 * after_federal
  federal_tax + state_tax = 15000 → x = 46400 := by sorry

end inheritance_calculation_l882_88268


namespace all_propositions_false_l882_88265

theorem all_propositions_false : ∃ a b : ℝ,
  (a > b ∧ a^2 ≤ b^2) ∧
  (a^2 > b^2 ∧ a ≤ b) ∧
  (a > b ∧ b/a ≥ 1) ∧
  (a > b ∧ 1/a ≥ 1/b) := by
  sorry

end all_propositions_false_l882_88265


namespace patrol_results_l882_88204

def travel_records : List Int := [10, -8, 6, -13, 7, -12, 3, -1]

def fuel_consumption_rate : ℝ := 0.05

def gas_station_distance : Int := 6

def final_position (records : List Int) : Int :=
  records.sum

def total_distance (records : List Int) : Int :=
  records.map (Int.natAbs) |>.sum

def times_passed_gas_station (records : List Int) (station_dist : Int) : Nat :=
  sorry

theorem patrol_results :
  (final_position travel_records = -8) ∧
  (total_distance travel_records = 60) ∧
  (times_passed_gas_station travel_records gas_station_distance = 4) := by
  sorry

end patrol_results_l882_88204


namespace xyz_inequality_l882_88202

theorem xyz_inequality (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  0 ≤ y*z + z*x + x*y - 2*x*y*z ∧ y*z + z*x + x*y - 2*x*y*z ≤ 7/27 := by
  sorry

end xyz_inequality_l882_88202


namespace mikes_seashells_l882_88218

/-- Given that Joan initially found 79 seashells and has 142 seashells in total after Mike gave her some,
    prove that Mike gave Joan 63 seashells. -/
theorem mikes_seashells (joans_initial : ℕ) (joans_total : ℕ) (mikes_gift : ℕ)
    (h1 : joans_initial = 79)
    (h2 : joans_total = 142)
    (h3 : joans_total = joans_initial + mikes_gift) :
  mikes_gift = 63 := by
  sorry

end mikes_seashells_l882_88218


namespace share_price_increase_l882_88209

theorem share_price_increase (initial_price : ℝ) : 
  let first_quarter_price := initial_price * (1 + 0.2)
  let second_quarter_price := first_quarter_price * (1 + 1/3)
  second_quarter_price = initial_price * (1 + 0.6) := by
sorry

end share_price_increase_l882_88209


namespace pebble_difference_l882_88297

/-- Represents the number of pebbles thrown by each person -/
structure PebbleCount where
  candy : ℚ
  lance : ℚ
  sandy : ℚ

/-- The pebble throwing scenario -/
def pebble_scenario (p : PebbleCount) : Prop :=
  p.lance = p.candy + 10 ∧ 
  5 * p.candy = 2 * p.lance ∧
  4 * p.candy = 2 * p.sandy

theorem pebble_difference (p : PebbleCount) 
  (h : pebble_scenario p) : 
  p.lance + p.sandy - p.candy = 30 := by
  sorry

#check pebble_difference

end pebble_difference_l882_88297


namespace unique_N_exists_l882_88288

theorem unique_N_exists : ∃! N : ℝ, 
  ∃ a b c : ℝ, 
    a + b + c = 120 ∧
    a + 8 = N ∧
    8 * b = N ∧
    c / 8 = N := by
  sorry

end unique_N_exists_l882_88288


namespace second_half_speed_l882_88274

/-- Represents the speed of a car during a trip -/
structure TripSpeed where
  average : ℝ
  firstHalf : ℝ
  secondHalf : ℝ

/-- Theorem stating the speed of the car in the second half of the trip -/
theorem second_half_speed (trip : TripSpeed) (h1 : trip.average = 60) (h2 : trip.firstHalf = 75) :
  trip.secondHalf = 150 := by
  sorry

end second_half_speed_l882_88274


namespace smallest_cube_ending_in_388_l882_88261

def is_cube_ending_in_388 (n : ℕ) : Prop := n^3 % 1000 = 388

theorem smallest_cube_ending_in_388 : 
  (∃ (n : ℕ), is_cube_ending_in_388 n) ∧ 
  (∀ (m : ℕ), m < 16 → ¬is_cube_ending_in_388 m) ∧ 
  is_cube_ending_in_388 16 :=
sorry

end smallest_cube_ending_in_388_l882_88261


namespace shawn_score_shawn_score_is_six_l882_88267

theorem shawn_score (points_per_basket : ℕ) (matthew_points : ℕ) (total_baskets : ℕ) : ℕ :=
  let matthew_baskets := matthew_points / points_per_basket
  let shawn_baskets := total_baskets - matthew_baskets
  let shawn_points := shawn_baskets * points_per_basket
  shawn_points

theorem shawn_score_is_six :
  shawn_score 3 9 5 = 6 := by
  sorry

end shawn_score_shawn_score_is_six_l882_88267


namespace initial_oranges_count_l882_88294

/-- Proves that the initial number of oranges in a bowl is 20, given the specified conditions. -/
theorem initial_oranges_count (apples : ℕ) (removed_oranges : ℕ) (apple_percentage : ℚ) : 
  apples = 14 → removed_oranges = 14 → apple_percentage = 7/10 → 
  ∃ initial_oranges : ℕ, 
    initial_oranges = 20 ∧ 
    (apples : ℚ) / ((apples : ℚ) + (initial_oranges - removed_oranges : ℚ)) = apple_percentage :=
by sorry

end initial_oranges_count_l882_88294
