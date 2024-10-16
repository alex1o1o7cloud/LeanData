import Mathlib

namespace NUMINAMATH_CALUDE_calculation_proof_l289_28958

theorem calculation_proof : (1 / Real.sqrt 3) - (1 / 4)⁻¹ + 4 * Real.sin (60 * π / 180) + |1 - Real.sqrt 3| = (10 / 3) * Real.sqrt 3 - 5 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l289_28958


namespace NUMINAMATH_CALUDE_smallest_equal_packages_l289_28967

theorem smallest_equal_packages (n m : ℕ) : 
  (∀ k l : ℕ, k > 0 ∧ l > 0 ∧ 9 * k = 12 * l → n ≤ k) ∧ 
  (∃ m : ℕ, m > 0 ∧ 9 * n = 12 * m) → 
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_smallest_equal_packages_l289_28967


namespace NUMINAMATH_CALUDE_carpet_area_and_cost_exceed_limits_l289_28917

/-- Represents the dimensions of various room types in Jesse's house -/
structure RoomDimensions where
  rectangular_length : ℝ
  rectangular_width : ℝ
  square_side : ℝ
  triangular_base : ℝ
  triangular_height : ℝ
  trapezoidal_base1 : ℝ
  trapezoidal_base2 : ℝ
  trapezoidal_height : ℝ
  circular_radius : ℝ
  elliptical_major_axis : ℝ
  elliptical_minor_axis : ℝ

/-- Represents the number of each room type in Jesse's house -/
structure RoomCounts where
  rectangular : ℕ
  square : ℕ
  triangular : ℕ
  trapezoidal : ℕ
  circular : ℕ
  elliptical : ℕ

/-- Calculates the total carpet area needed and proves it exceeds 2000 square feet -/
def total_carpet_area_exceeds_2000 (dims : RoomDimensions) (counts : RoomCounts) : Prop :=
  let total_area := 
    counts.rectangular * (dims.rectangular_length * dims.rectangular_width) +
    counts.square * (dims.square_side * dims.square_side) +
    counts.triangular * (dims.triangular_base * dims.triangular_height / 2) +
    counts.trapezoidal * ((dims.trapezoidal_base1 + dims.trapezoidal_base2) / 2 * dims.trapezoidal_height) +
    counts.circular * (Real.pi * dims.circular_radius * dims.circular_radius) +
    counts.elliptical * (Real.pi * (dims.elliptical_major_axis / 2) * (dims.elliptical_minor_axis / 2))
  total_area > 2000

/-- Proves that the total cost exceeds $10,000 when carpet costs $5 per square foot -/
def total_cost_exceeds_budget (dims : RoomDimensions) (counts : RoomCounts) : Prop :=
  let total_area := 
    counts.rectangular * (dims.rectangular_length * dims.rectangular_width) +
    counts.square * (dims.square_side * dims.square_side) +
    counts.triangular * (dims.triangular_base * dims.triangular_height / 2) +
    counts.trapezoidal * ((dims.trapezoidal_base1 + dims.trapezoidal_base2) / 2 * dims.trapezoidal_height) +
    counts.circular * (Real.pi * dims.circular_radius * dims.circular_radius) +
    counts.elliptical * (Real.pi * (dims.elliptical_major_axis / 2) * (dims.elliptical_minor_axis / 2))
  total_area * 5 > 10000

/-- Main theorem combining both conditions -/
theorem carpet_area_and_cost_exceed_limits (dims : RoomDimensions) (counts : RoomCounts) :
  total_carpet_area_exceeds_2000 dims counts ∧ total_cost_exceeds_budget dims counts :=
sorry

end NUMINAMATH_CALUDE_carpet_area_and_cost_exceed_limits_l289_28917


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l289_28978

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l289_28978


namespace NUMINAMATH_CALUDE_final_ethanol_percentage_l289_28960

/-- Calculates the final ethanol percentage in a fuel mixture after adding pure ethanol -/
theorem final_ethanol_percentage
  (initial_volume : ℝ)
  (initial_ethanol_percentage : ℝ)
  (added_ethanol : ℝ)
  (h1 : initial_volume = 27)
  (h2 : initial_ethanol_percentage = 0.05)
  (h3 : added_ethanol = 1.5)
  : (initial_volume * initial_ethanol_percentage + added_ethanol) / (initial_volume + added_ethanol) = 0.1 := by
  sorry

#check final_ethanol_percentage

end NUMINAMATH_CALUDE_final_ethanol_percentage_l289_28960


namespace NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_and_discriminant_multiple_of_four_l289_28957

def polynomial (x : ℂ) : ℂ := x^2 + 6*x + 13

theorem monic_quadratic_with_complex_root_and_discriminant_multiple_of_four :
  (∀ x : ℂ, polynomial x = x^2 + 6*x + 13) ∧
  (polynomial (-3 + 2*I) = 0) ∧
  (∃ k : ℤ, 6^2 - 4*(1:ℝ)*13 = 4*k) :=
by sorry

end NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_and_discriminant_multiple_of_four_l289_28957


namespace NUMINAMATH_CALUDE_original_ratio_of_boarders_to_day_students_l289_28912

theorem original_ratio_of_boarders_to_day_students 
  (initial_boarders : ℕ) 
  (new_boarders : ℕ) 
  (final_ratio_boarders : ℕ) 
  (final_ratio_day_students : ℕ) : 
  initial_boarders = 150 →
  new_boarders = 30 →
  final_ratio_boarders = 1 →
  final_ratio_day_students = 2 →
  ∃ (original_ratio_boarders original_ratio_day_students : ℕ),
    original_ratio_boarders = 5 ∧ 
    original_ratio_day_students = 12 ∧
    (initial_boarders : ℚ) / (initial_boarders + new_boarders : ℚ) * final_ratio_day_students = 
      (original_ratio_boarders : ℚ) / (original_ratio_boarders + original_ratio_day_students : ℚ) :=
by sorry


end NUMINAMATH_CALUDE_original_ratio_of_boarders_to_day_students_l289_28912


namespace NUMINAMATH_CALUDE_probability_of_even_sum_l289_28986

def number_of_balls : ℕ := 12

def is_even_sum (a b : ℕ) : Prop := Even (a + b)

theorem probability_of_even_sum :
  let total_outcomes := number_of_balls * (number_of_balls - 1)
  let favorable_outcomes := (number_of_balls / 2) * ((number_of_balls / 2) - 1) * 2
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_even_sum_l289_28986


namespace NUMINAMATH_CALUDE_rate_squares_sum_l289_28939

theorem rate_squares_sum : ∃ (b j s : ℕ), b + j + s = 34 ∧ b^2 + j^2 + s^2 = 406 := by
  sorry

end NUMINAMATH_CALUDE_rate_squares_sum_l289_28939


namespace NUMINAMATH_CALUDE_series_sum_equals_one_fourth_l289_28907

/-- The sum of the infinite series ∑(n=1 to ∞) [3^n / (1 + 3^n + 3^(n+1) + 3^(2n+1))] is equal to 1/4 -/
theorem series_sum_equals_one_fourth :
  let a : ℕ → ℝ := λ n => (3 : ℝ)^n / (1 + (3 : ℝ)^n + (3 : ℝ)^(n+1) + (3 : ℝ)^(2*n+1))
  ∑' n, a n = 1/4 := by sorry

end NUMINAMATH_CALUDE_series_sum_equals_one_fourth_l289_28907


namespace NUMINAMATH_CALUDE_proportion_third_number_l289_28975

theorem proportion_third_number : 
  ∀ y : ℝ, (0.75 : ℝ) / 0.6 = y / 8 → y = 10 := by
  sorry

end NUMINAMATH_CALUDE_proportion_third_number_l289_28975


namespace NUMINAMATH_CALUDE_least_integer_with_given_remainders_l289_28972

theorem least_integer_with_given_remainders : ∃ x : ℕ, 
  x > 0 ∧
  x % 3 = 2 ∧
  x % 4 = 3 ∧
  x % 5 = 4 ∧
  x % 7 = 6 ∧
  (∀ y : ℕ, y > 0 ∧ y % 3 = 2 ∧ y % 4 = 3 ∧ y % 5 = 4 ∧ y % 7 = 6 → x ≤ y) ∧
  x = 419 :=
by sorry

end NUMINAMATH_CALUDE_least_integer_with_given_remainders_l289_28972


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l289_28992

/-- A geometric sequence with sum of first n terms S_n = 4^n + a has a = -1 -/
theorem geometric_sequence_sum (S : ℕ → ℝ) (a : ℝ) :
  (∀ n : ℕ, S n = 4^n + a) →
  (∃ r : ℝ, ∀ n : ℕ, S (n + 1) - S n = r * (S n - S (n - 1))) →
  a = -1 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l289_28992


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l289_28994

/-- The probability of selecting either a blue or yellow jelly bean from a bag -/
theorem jelly_bean_probability : 
  let red : ℕ := 6
  let green : ℕ := 7
  let yellow : ℕ := 8
  let blue : ℕ := 9
  let total : ℕ := red + green + yellow + blue
  let target : ℕ := yellow + blue
  (target : ℚ) / total = 17 / 30 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l289_28994


namespace NUMINAMATH_CALUDE_ab_equals_six_l289_28961

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_ab_equals_six_l289_28961


namespace NUMINAMATH_CALUDE_charles_earnings_proof_l289_28985

/-- Calculates Charles's earnings after tax from pet care activities -/
def charles_earnings (housesitting_rate : ℝ) (lab_walk_rate : ℝ) (gr_walk_rate : ℝ) (gs_walk_rate : ℝ)
                     (lab_groom_rate : ℝ) (gr_groom_rate : ℝ) (gs_groom_rate : ℝ)
                     (housesitting_time : ℝ) (lab_walk_time : ℝ) (gr_walk_time : ℝ) (gs_walk_time : ℝ)
                     (tax_rate : ℝ) : ℝ :=
  let total_before_tax := housesitting_rate * housesitting_time +
                          lab_walk_rate * lab_walk_time +
                          gr_walk_rate * gr_walk_time +
                          gs_walk_rate * gs_walk_time +
                          lab_groom_rate + gr_groom_rate + gs_groom_rate
  let tax_deduction := tax_rate * total_before_tax
  total_before_tax - tax_deduction

/-- Theorem stating Charles's earnings after tax -/
theorem charles_earnings_proof :
  charles_earnings 15 22 25 30 10 15 20 10 3 2 1.5 0.12 = 313.28 := by
  sorry

end NUMINAMATH_CALUDE_charles_earnings_proof_l289_28985


namespace NUMINAMATH_CALUDE_negation_of_forall_gt_sin_l289_28926

theorem negation_of_forall_gt_sin (P : ℝ → Prop) : 
  (¬ ∀ x > 0, 2 * x > Real.sin x) ↔ (∃ x₀ > 0, 2 * x₀ ≤ Real.sin x₀) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_gt_sin_l289_28926


namespace NUMINAMATH_CALUDE_triangle_area_set_S_is_two_horizontal_lines_l289_28919

/-- The set of points A(x, y) for which the area of triangle ABC is 2,
    where B(1, 0) and C(-1, 0) are fixed points -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p; abs y = 2}

/-- The area of triangle ABC given point A(x, y) and fixed points B(1, 0) and C(-1, 0) -/
def triangleArea (x y : ℝ) : ℝ := abs y

theorem triangle_area_set :
  ∀ (x y : ℝ), (x, y) ∈ S ↔ triangleArea x y = 2 :=
by sorry

theorem S_is_two_horizontal_lines :
  S = {p : ℝ × ℝ | let (x, y) := p; y = 2 ∨ y = -2} :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_set_S_is_two_horizontal_lines_l289_28919


namespace NUMINAMATH_CALUDE_fluffy_carrots_l289_28956

def carrot_sequence (first_day : ℕ) : ℕ → ℕ
  | 0 => first_day
  | n + 1 => 2 * carrot_sequence first_day n

def total_carrots (first_day : ℕ) : ℕ :=
  (carrot_sequence first_day 0) + (carrot_sequence first_day 1) + (carrot_sequence first_day 2)

theorem fluffy_carrots (first_day : ℕ) :
  total_carrots first_day = 84 → carrot_sequence first_day 2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_fluffy_carrots_l289_28956


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_m_values_l289_28927

theorem ellipse_eccentricity_m_values (m : ℝ) :
  (∃ x y : ℝ, x^2 / 9 + y^2 / (m + 9) = 1) →
  (∃ c : ℝ, c^2 / (m + 9) = 1/4) →
  (m = -9/4 ∨ m = 3) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_m_values_l289_28927


namespace NUMINAMATH_CALUDE_puppies_sold_l289_28947

/-- Given a pet store scenario, prove the number of puppies sold -/
theorem puppies_sold (initial_puppies cages_used puppies_per_cage : ℕ) :
  initial_puppies - (cages_used * puppies_per_cage) =
  initial_puppies - cages_used * puppies_per_cage :=
by sorry

end NUMINAMATH_CALUDE_puppies_sold_l289_28947


namespace NUMINAMATH_CALUDE_decimal_365_to_octal_l289_28980

/-- Converts a natural number to its octal representation as a list of digits -/
def toOctal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: toOctal (n / 8)

/-- Theorem: The decimal number 365 is equal to 555₈ in octal representation -/
theorem decimal_365_to_octal :
  toOctal 365 = [5, 5, 5] := by
  sorry

end NUMINAMATH_CALUDE_decimal_365_to_octal_l289_28980


namespace NUMINAMATH_CALUDE_prime_sum_24_l289_28976

theorem prime_sum_24 (a b c : ℕ) : 
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c → a * b + b * c = 119 → a + b + c = 24 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_24_l289_28976


namespace NUMINAMATH_CALUDE_fraction_value_l289_28954

theorem fraction_value (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : 1/a + 1/b = 3) :
  (a + 2*a*b + b) / (2*a*b - a - b) = -5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l289_28954


namespace NUMINAMATH_CALUDE_area_between_curves_l289_28901

theorem area_between_curves : 
  let f (x : ℝ) := x^2
  let g (x : ℝ) := x^3
  ∫ x in (0: ℝ)..(1: ℝ), f x - g x = 1/12 := by sorry

end NUMINAMATH_CALUDE_area_between_curves_l289_28901


namespace NUMINAMATH_CALUDE_min_value_quadratic_ratio_l289_28913

/-- A quadratic function -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The derivative of a quadratic function -/
def quadratic_derivative (a b : ℝ) (x : ℝ) : ℝ := 2 * a * x + b

theorem min_value_quadratic_ratio 
  (a b c : ℝ) 
  (h1 : quadratic_derivative a b 0 > 0)
  (h2 : ∀ x, quadratic a b c x ≥ 0) :
  (quadratic a b c 1) / (quadratic_derivative a b 0) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_ratio_l289_28913


namespace NUMINAMATH_CALUDE_election_win_percentage_l289_28988

/-- In a two-candidate election, if a candidate receives 45% of the total votes,
    they need more than 50% of the total votes to win. -/
theorem election_win_percentage (total_votes : ℕ) (candidate_votes : ℕ) 
    (h1 : candidate_votes = (45 : ℕ) * total_votes / 100) 
    (h2 : total_votes > 0) : 
    ∃ (winning_percentage : ℚ), 
      winning_percentage > (1 : ℚ) / 2 ∧ 
      winning_percentage * total_votes > candidate_votes := by
  sorry

end NUMINAMATH_CALUDE_election_win_percentage_l289_28988


namespace NUMINAMATH_CALUDE_spherical_coordinate_shift_l289_28910

/-- Given a point with rectangular coordinates (3, -2, 5) and spherical coordinates (r, α, β),
    prove that the point with spherical coordinates (r, α+π, β) has rectangular coordinates (-3, 2, 5). -/
theorem spherical_coordinate_shift (r α β : ℝ) : 
  (3 = r * Real.sin β * Real.cos α) → 
  (-2 = r * Real.sin β * Real.sin α) → 
  (5 = r * Real.cos β) → 
  ((-3, 2, 5) : ℝ × ℝ × ℝ) = (
    r * Real.sin β * Real.cos (α + Real.pi),
    r * Real.sin β * Real.sin (α + Real.pi),
    r * Real.cos β
  ) := by sorry

end NUMINAMATH_CALUDE_spherical_coordinate_shift_l289_28910


namespace NUMINAMATH_CALUDE_factors_of_x4_plus_81_l289_28964

theorem factors_of_x4_plus_81 (x : ℝ) : x^4 + 81 = (x^2 + 6*x + 9) * (x^2 - 6*x + 9) := by
  sorry

end NUMINAMATH_CALUDE_factors_of_x4_plus_81_l289_28964


namespace NUMINAMATH_CALUDE_needle_intersection_probability_l289_28993

/-- Represents the experimental data for needle throwing --/
structure NeedleExperiment where
  trials : ℕ
  intersections : ℕ
  frequency : ℚ

/-- The set of experimental data --/
def experimentalData : List NeedleExperiment := [
  ⟨50, 23, 23/50⟩,
  ⟨100, 48, 12/25⟩,
  ⟨200, 83, 83/200⟩,
  ⟨500, 207, 207/500⟩,
  ⟨1000, 404, 101/250⟩,
  ⟨2000, 802, 401/1000⟩
]

/-- The distance between adjacent lines in cm --/
def lineDistance : ℚ := 5

/-- The length of the needle in cm --/
def needleLength : ℚ := 3

/-- The estimated probability of intersection --/
def estimatedProbability : ℚ := 2/5

/-- Theorem stating that the estimated probability approaches 0.4 as trials increase --/
theorem needle_intersection_probability :
  ∀ ε > 0, ∃ N : ℕ, ∀ e ∈ experimentalData,
    e.trials ≥ N → |e.frequency - estimatedProbability| < ε :=
sorry

end NUMINAMATH_CALUDE_needle_intersection_probability_l289_28993


namespace NUMINAMATH_CALUDE_carlos_singles_percentage_l289_28900

/-- Represents the statistics of Carlos's baseball hits -/
structure BaseballStats where
  total_hits : ℕ
  home_runs : ℕ
  triples : ℕ
  doubles : ℕ

/-- Calculates the percentage of singles in Carlos's hits -/
def percentage_singles (stats : BaseballStats) : ℚ :=
  let non_singles := stats.home_runs + stats.triples + stats.doubles
  let singles := stats.total_hits - non_singles
  (singles : ℚ) / stats.total_hits * 100

/-- Carlos's baseball statistics -/
def carlos_stats : BaseballStats :=
  { total_hits := 50
  , home_runs := 3
  , triples := 2
  , doubles := 8 }

/-- Theorem stating that the percentage of singles in Carlos's hits is 74% -/
theorem carlos_singles_percentage :
  percentage_singles carlos_stats = 74 := by
  sorry


end NUMINAMATH_CALUDE_carlos_singles_percentage_l289_28900


namespace NUMINAMATH_CALUDE_max_slope_on_circle_l289_28987

theorem max_slope_on_circle (x y : ℝ) :
  x^2 + y^2 - 2*x - 2 = 0 →
  (∀ a b : ℝ, a^2 + b^2 - 2*a - 2 = 0 → (y + 1) / (x + 1) ≤ (b + 1) / (a + 1)) →
  (y + 1) / (x + 1) = 2 + Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_max_slope_on_circle_l289_28987


namespace NUMINAMATH_CALUDE_set_equality_l289_28942

def A : Set ℤ := {-2, -1, 0, 1, 2}

theorem set_equality : {y : ℤ | ∃ x ∈ A, y = |x + 1|} = {0, 1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l289_28942


namespace NUMINAMATH_CALUDE_count_valid_primes_l289_28990

def isSubnumber (n m : ℕ) : Prop :=
  ∃ (k l : ℕ), n = (m / 10^k) % (10^l)

def hasNonPrimeSubnumber (n : ℕ) : Prop :=
  ∃ (m : ℕ), isSubnumber m n ∧ m > 1 ∧ ¬ Nat.Prime m

def validPrime (n : ℕ) : Prop :=
  Nat.Prime n ∧ n < 1000000000 ∧ ¬ hasNonPrimeSubnumber n

theorem count_valid_primes :
  ∃! (s : Finset ℕ), (∀ n ∈ s, validPrime n) ∧ s.card = 9 :=
sorry

end NUMINAMATH_CALUDE_count_valid_primes_l289_28990


namespace NUMINAMATH_CALUDE_alice_marble_groups_l289_28930

def pink_marble : ℕ := 1
def blue_marble : ℕ := 1
def white_marble : ℕ := 1
def black_marbles : ℕ := 4

def total_marbles : ℕ := pink_marble + blue_marble + white_marble + black_marbles

def different_color_pairs : ℕ := Nat.choose 4 2

theorem alice_marble_groups : 
  (different_color_pairs + 1 : ℕ) = 7 :=
sorry

end NUMINAMATH_CALUDE_alice_marble_groups_l289_28930


namespace NUMINAMATH_CALUDE_least_multiple_25_over_500_l289_28973

theorem least_multiple_25_over_500 : 
  ∀ n : ℕ, n > 0 → 25 * n > 500 → 525 ≤ 25 * n :=
sorry

end NUMINAMATH_CALUDE_least_multiple_25_over_500_l289_28973


namespace NUMINAMATH_CALUDE_partnership_profit_distribution_l289_28965

/-- Partnership profit distribution problem -/
theorem partnership_profit_distribution
  (invest_a invest_b invest_c : ℚ)  -- Investments of A, B, and C
  (total_profit : ℚ)                -- Total profit earned
  (h1 : invest_a = 3 * invest_b)    -- A invests 3 times as much as B
  (h2 : invest_b = (2/3) * invest_c) -- B invests two-thirds of what C invests
  : invest_b / (invest_a + invest_b + invest_c) = 2/7 := by
  sorry

#eval (2/7 : ℚ) * 8800  -- Expected result: approximately 2514.29

end NUMINAMATH_CALUDE_partnership_profit_distribution_l289_28965


namespace NUMINAMATH_CALUDE_eighth_grade_students_l289_28906

theorem eighth_grade_students (total : ℕ) (girls : ℕ) (boys : ℕ) : 
  total = 68 → 
  girls = 28 → 
  boys < 2 * girls →
  boys = total - girls →
  2 * girls - boys = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_eighth_grade_students_l289_28906


namespace NUMINAMATH_CALUDE_range_a_theorem_l289_28996

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀ + (a - 1) * x₀ + 1 < 0

-- Define the range of a
def range_of_a (a : ℝ) : Prop := a > 3 ∨ (a ≥ -1 ∧ a ≤ 1)

-- State the theorem
theorem range_a_theorem (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → range_of_a a :=
by sorry

end NUMINAMATH_CALUDE_range_a_theorem_l289_28996


namespace NUMINAMATH_CALUDE_speed_ratio_is_four_fifths_l289_28977

/-- Two objects A and B moving uniformly along perpendicular paths -/
structure PerpendicularMotion where
  vA : ℝ  -- Speed of object A
  vB : ℝ  -- Speed of object B
  d  : ℝ  -- Initial distance of B from O

/-- Equidistance condition at time t -/
def equidistant (m : PerpendicularMotion) (t : ℝ) : Prop :=
  m.vA * t = |m.d - m.vB * t|

/-- The theorem stating the ratio of speeds given the conditions -/
theorem speed_ratio_is_four_fifths (m : PerpendicularMotion) :
  m.d = 600 ∧ equidistant m 3 ∧ equidistant m 12 → m.vA / m.vB = 4/5 := by
  sorry

#check speed_ratio_is_four_fifths

end NUMINAMATH_CALUDE_speed_ratio_is_four_fifths_l289_28977


namespace NUMINAMATH_CALUDE_gcd_problem_l289_28920

theorem gcd_problem (a : ℤ) (h : ∃ k : ℤ, a = (2*k + 1) * 7767) :
  Int.gcd (6*a^2 + 5*a + 108) (3*a + 9) = 9 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l289_28920


namespace NUMINAMATH_CALUDE_sum_of_arithmetic_sequence_l289_28945

theorem sum_of_arithmetic_sequence (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) (n : ℕ) :
  a₁ = 107 →
  d = 10 →
  aₙ = 447 →
  n = (aₙ - a₁) / d + 1 →
  (n : ℝ) / 2 * (a₁ + aₙ) = 9695 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_arithmetic_sequence_l289_28945


namespace NUMINAMATH_CALUDE_variance_transformed_l289_28902

-- Define a random variable ξ
variable (ξ : ℝ → ℝ)

-- Define the variance operator D
noncomputable def D (X : ℝ → ℝ) : ℝ := sorry

-- Given condition
axiom variance_xi : D ξ = 2

-- Theorem to prove
theorem variance_transformed : D (fun ω => 2 * ξ ω + 3) = 8 := by sorry

end NUMINAMATH_CALUDE_variance_transformed_l289_28902


namespace NUMINAMATH_CALUDE_solution_to_equation_1_no_solution_to_equation_2_l289_28981

-- Problem 1
theorem solution_to_equation_1 (x : ℝ) : 
  (3 / x - 2 / (x - 2) = 0) ↔ (x = 6) :=
sorry

-- Problem 2
theorem no_solution_to_equation_2 :
  ¬∃ (x : ℝ), (3 / (4 - x) + 2 = (1 - x) / (x - 4)) :=
sorry

end NUMINAMATH_CALUDE_solution_to_equation_1_no_solution_to_equation_2_l289_28981


namespace NUMINAMATH_CALUDE_function_properties_l289_28940

def IsAdditive (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

theorem function_properties
    (f : ℝ → ℝ)
    (h_additive : IsAdditive f)
    (h_neg : ∀ x : ℝ, x > 0 → f x < 0)
    (h_f_neg_one : f (-1) = 2) :
    (f 0 = 0 ∧ ∀ x : ℝ, f (-x) = -f x) ∧
    (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂) ∧
    Set.range (fun x => f x) ∩ Set.Icc (-2 : ℝ) 4 = Set.Icc (-8 : ℝ) 4 := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l289_28940


namespace NUMINAMATH_CALUDE_fraction_simplification_l289_28979

theorem fraction_simplification : (240 / 18) * (6 / 135) * (9 / 4) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l289_28979


namespace NUMINAMATH_CALUDE_bus_walk_distance_difference_l289_28998

/-- Craig's route home from school -/
structure Route where
  busA : ℝ
  walk1 : ℝ
  busB : ℝ
  walk2 : ℝ
  busC : ℝ
  walk3 : ℝ

/-- Calculate the total bus distance -/
def totalBusDistance (r : Route) : ℝ :=
  r.busA + r.busB + r.busC

/-- Calculate the total walking distance -/
def totalWalkDistance (r : Route) : ℝ :=
  r.walk1 + r.walk2 + r.walk3

/-- Craig's actual route -/
def craigsRoute : Route :=
  { busA := 1.25
  , walk1 := 0.35
  , busB := 2.68
  , walk2 := 0.47
  , busC := 3.27
  , walk3 := 0.21 }

/-- Theorem: The difference between total bus distance and total walking distance is 6.17 miles -/
theorem bus_walk_distance_difference :
  totalBusDistance craigsRoute - totalWalkDistance craigsRoute = 6.17 := by
  sorry

end NUMINAMATH_CALUDE_bus_walk_distance_difference_l289_28998


namespace NUMINAMATH_CALUDE_prob_same_length_regular_hexagon_l289_28948

/-- The set of all sides and diagonals of a regular hexagon -/
def T : Finset ℝ := sorry

/-- The number of elements in T -/
def n : ℕ := Finset.card T

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of short diagonals in a regular hexagon -/
def num_short_diag : ℕ := 6

/-- The number of long diagonals in a regular hexagon -/
def num_long_diag : ℕ := 3

/-- The probability of selecting two segments of the same length -/
def prob_same_length : ℚ :=
  (num_sides * (num_sides - 1) + num_short_diag * (num_short_diag - 1) + num_long_diag * (num_long_diag - 1)) /
  (n * (n - 1))

theorem prob_same_length_regular_hexagon :
  prob_same_length = 22 / 35 := by sorry

end NUMINAMATH_CALUDE_prob_same_length_regular_hexagon_l289_28948


namespace NUMINAMATH_CALUDE_youngest_sibling_age_l289_28916

/-- The age of the youngest sibling in a family of 6 siblings -/
def youngest_age : ℝ := 17.5

/-- The number of siblings in the family -/
def num_siblings : ℕ := 6

/-- The age differences between the siblings and the youngest sibling -/
def age_differences : List ℝ := [4, 5, 7, 9, 11]

/-- The average age of all siblings -/
def average_age : ℝ := 23.5

/-- Theorem stating that given the conditions, the age of the youngest sibling is 17.5 -/
theorem youngest_sibling_age :
  let ages := youngest_age :: (age_differences.map (· + youngest_age))
  (ages.sum / num_siblings) = average_age ∧
  ages.length = num_siblings :=
by sorry

end NUMINAMATH_CALUDE_youngest_sibling_age_l289_28916


namespace NUMINAMATH_CALUDE_kevin_toad_count_l289_28937

/-- The number of worms each toad eats daily -/
def worms_per_toad : ℕ := 3

/-- The time in minutes it takes Kevin to find one worm -/
def minutes_per_worm : ℕ := 15

/-- The total time in hours Kevin spends finding worms -/
def total_hours : ℕ := 6

/-- The number of toads in Kevin's shoebox -/
def number_of_toads : ℕ := 8

theorem kevin_toad_count : number_of_toads = 8 := by
  sorry

end NUMINAMATH_CALUDE_kevin_toad_count_l289_28937


namespace NUMINAMATH_CALUDE_complex_power_sum_l289_28943

theorem complex_power_sum (z : ℂ) (h : z + z⁻¹ = -Real.sqrt 2) : z^12 + z⁻¹^12 = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l289_28943


namespace NUMINAMATH_CALUDE_certain_number_exists_and_unique_l289_28936

theorem certain_number_exists_and_unique :
  ∃! N : ℚ, (5/6 * N) - (5/16 * N) = 100 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_exists_and_unique_l289_28936


namespace NUMINAMATH_CALUDE_polynomial_multiple_power_coefficients_l289_28949

theorem polynomial_multiple_power_coefficients 
  (p : Polynomial ℝ) (n : ℕ) (hn : n > 0) :
  ∃ q : Polynomial ℝ, q ≠ 0 ∧ 
  ∀ i : ℕ, (p * q).coeff i ≠ 0 → ∃ k : ℕ, i = n * k :=
by sorry

end NUMINAMATH_CALUDE_polynomial_multiple_power_coefficients_l289_28949


namespace NUMINAMATH_CALUDE_horse_saddle_ratio_l289_28934

theorem horse_saddle_ratio : ∀ (total_cost saddle_cost horse_cost : ℕ),
  total_cost = 5000 →
  saddle_cost = 1000 →
  horse_cost = total_cost - saddle_cost →
  (horse_cost : ℚ) / (saddle_cost : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_horse_saddle_ratio_l289_28934


namespace NUMINAMATH_CALUDE_problem_solution_l289_28959

def A : Set ℝ := {1, 2}

def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*a*x + a = 0}

def C (m : ℝ) : Set ℝ := {x : ℝ | x^2 - m*x + 3 > 0}

theorem problem_solution :
  (∀ a : ℝ, (∀ x ∈ B a, x ∈ A) ↔ a ∈ Set.Ioo 0 1) ∧
  (∀ m : ℝ, (A ⊆ C m) ↔ m ∈ Set.Iic (7/2)) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l289_28959


namespace NUMINAMATH_CALUDE_system_solution_ratio_l289_28974

theorem system_solution_ratio (a b x y : ℝ) :
  8 * x - 6 * y = a →
  12 * y - 18 * x = b →
  x ≠ 0 →
  y ≠ 0 →
  b ≠ 0 →
  a / b = -4 / 9 := by sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l289_28974


namespace NUMINAMATH_CALUDE_canoe_row_probability_l289_28951

-- Define the probability of each oar working
def p_left_works : ℚ := 3/5
def p_right_works : ℚ := 3/5

-- Define the event of being able to row the canoe
def can_row : ℚ := 
  p_left_works * p_right_works +  -- both oars work
  p_left_works * (1 - p_right_works) +  -- left works, right breaks
  (1 - p_left_works) * p_right_works  -- left breaks, right works

-- Theorem statement
theorem canoe_row_probability : can_row = 21/25 := by
  sorry

end NUMINAMATH_CALUDE_canoe_row_probability_l289_28951


namespace NUMINAMATH_CALUDE_gcd_5039_3427_l289_28955

theorem gcd_5039_3427 : Nat.gcd 5039 3427 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_5039_3427_l289_28955


namespace NUMINAMATH_CALUDE_marble_distribution_proof_l289_28914

/-- The number of marbles in the jar -/
def total_marbles : ℕ := 312

/-- The number of people in the group today -/
def group_size : ℕ := 24

/-- The number of additional people joining in the future scenario -/
def additional_people : ℕ := 2

/-- The decrease in marbles per person in the future scenario -/
def marble_decrease : ℕ := 1

theorem marble_distribution_proof :
  (total_marbles / group_size = total_marbles / (group_size + additional_people) + marble_decrease) ∧
  (total_marbles % group_size = 0) :=
sorry

end NUMINAMATH_CALUDE_marble_distribution_proof_l289_28914


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l289_28935

theorem decimal_to_fraction : 
  (3.68 : ℚ) = 92 / 25 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l289_28935


namespace NUMINAMATH_CALUDE_inequality_proof_l289_28922

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  x*y/z + y*z/x + x*z/y ≥ Real.sqrt 3 ∧ 
  (x*y/z + y*z/x + x*z/y = Real.sqrt 3 ↔ x = y ∧ y = z ∧ z = Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l289_28922


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_l289_28931

theorem half_abs_diff_squares : (1 / 2 : ℝ) * |23^2 - 19^2| = 84 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_l289_28931


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l289_28925

noncomputable def polynomial_remainder 
  (p : ℝ → ℝ) (a b : ℝ) (h : a ≠ b) : ℝ × ℝ × ℝ :=
  let r := (p a - p b - (b - a) * (deriv p a)) / ((b - a) * (b + a))
  let d := deriv p a - 2 * r * a
  let e := p a - r * a^2 - d * a
  (r, d, e)

theorem polynomial_division_theorem 
  (p : ℝ → ℝ) (a b : ℝ) (h : a ≠ b) :
  ∃ (q : ℝ → ℝ) (r d e : ℝ),
    (∀ x, p x = q x * (x - a)^2 * (x - b) + r * x^2 + d * x + e) ∧
    (r, d, e) = polynomial_remainder p a b h :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l289_28925


namespace NUMINAMATH_CALUDE_square_side_factor_l289_28924

theorem square_side_factor : ∃ f : ℝ, 
  (∀ s : ℝ, s > 0 → s^2 = 20 * (f*s)^2) ∧ f = Real.sqrt 5 / 10 := by
  sorry

end NUMINAMATH_CALUDE_square_side_factor_l289_28924


namespace NUMINAMATH_CALUDE_game_results_l289_28915

/-- Represents a strategy for choosing digits -/
def Strategy := Nat → Nat

/-- Represents the result of the game -/
inductive GameResult
| FirstPlayerWins
| SecondPlayerWins

/-- Determines if a list of digits is divisible by 9 -/
def isDivisibleBy9 (digits : List Nat) : Prop :=
  digits.sum % 9 = 0

/-- Simulates the game for a given k and returns the result -/
def playGame (k : Nat) (firstPlayerStrategy : Strategy) (secondPlayerStrategy : Strategy) : GameResult :=
  sorry

/-- Theorem stating the game results for k = 10 and k = 15 -/
theorem game_results :
  (∀ (firstPlayerStrategy : Strategy),
    ∃ (secondPlayerStrategy : Strategy),
      playGame 10 firstPlayerStrategy secondPlayerStrategy = GameResult.SecondPlayerWins) ∧
  (∃ (firstPlayerStrategy : Strategy),
    ∀ (secondPlayerStrategy : Strategy),
      playGame 15 firstPlayerStrategy secondPlayerStrategy = GameResult.FirstPlayerWins) :=
sorry

end NUMINAMATH_CALUDE_game_results_l289_28915


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l289_28968

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y - x * y = 0) :
  x + 2 * y ≥ 9 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 2 * x + y - x * y = 0 ∧ x + 2 * y = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l289_28968


namespace NUMINAMATH_CALUDE_symmetric_scanning_codes_count_l289_28941

/-- Represents a color of a square in the grid -/
inductive Color
| Black
| White

/-- Represents a square in the 8x8 grid -/
structure Square where
  row : Fin 8
  col : Fin 8
  color : Color

/-- Represents the 8x8 grid -/
def Grid := Array (Array Square)

/-- Checks if a square has at least one adjacent square of each color -/
def hasAdjacentColors (grid : Grid) (square : Square) : Prop :=
  sorry

/-- Checks if the grid is symmetric under 90 degree rotation -/
def isSymmetricUnder90Rotation (grid : Grid) : Prop :=
  sorry

/-- Checks if the grid is symmetric under 180 degree rotation -/
def isSymmetricUnder180Rotation (grid : Grid) : Prop :=
  sorry

/-- Checks if the grid is symmetric under 270 degree rotation -/
def isSymmetricUnder270Rotation (grid : Grid) : Prop :=
  sorry

/-- Checks if the grid is symmetric under reflection across midpoint lines -/
def isSymmetricUnderMidpointReflection (grid : Grid) : Prop :=
  sorry

/-- Checks if the grid is symmetric under reflection across diagonals -/
def isSymmetricUnderDiagonalReflection (grid : Grid) : Prop :=
  sorry

/-- Checks if the grid satisfies all symmetry conditions -/
def isSymmetric (grid : Grid) : Prop :=
  isSymmetricUnder90Rotation grid ∧
  isSymmetricUnder180Rotation grid ∧
  isSymmetricUnder270Rotation grid ∧
  isSymmetricUnderMidpointReflection grid ∧
  isSymmetricUnderDiagonalReflection grid

/-- Counts the number of symmetric scanning codes -/
def countSymmetricCodes : Nat :=
  sorry

/-- The main theorem stating that the number of symmetric scanning codes is 254 -/
theorem symmetric_scanning_codes_count :
  countSymmetricCodes = 254 :=
sorry

end NUMINAMATH_CALUDE_symmetric_scanning_codes_count_l289_28941


namespace NUMINAMATH_CALUDE_trapezoid_ratio_theorem_l289_28991

/-- Represents a trapezoid with bases and a point inside it -/
structure Trapezoid where
  EF : ℝ
  GH : ℝ
  isIsosceles : Bool
  EFGreaterGH : EF > GH

/-- Represents the areas of triangles formed by dividing a trapezoid -/
structure TriangleAreas where
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  area4 : ℝ

/-- Theorem stating the ratio of bases in a trapezoid given specific triangle areas -/
theorem trapezoid_ratio_theorem (T : Trapezoid) (A : TriangleAreas) :
  T.isIsosceles = true ∧
  A.area1 = 3 ∧ A.area2 = 4 ∧ A.area3 = 6 ∧ A.area4 = 7 →
  T.EF / T.GH = 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_ratio_theorem_l289_28991


namespace NUMINAMATH_CALUDE_nonzero_digits_count_l289_28969

-- Define the fraction
def f : ℚ := 80 / (2^4 * 5^9)

-- Define a function to count non-zero digits after decimal point
noncomputable def count_nonzero_digits_after_decimal (q : ℚ) : ℕ := sorry

-- Theorem statement
theorem nonzero_digits_count :
  count_nonzero_digits_after_decimal f = 3 := by sorry

end NUMINAMATH_CALUDE_nonzero_digits_count_l289_28969


namespace NUMINAMATH_CALUDE_first_chapter_has_48_pages_l289_28999

/-- A book with two chapters -/
structure Book where
  total_pages : ℕ
  second_chapter_pages : ℕ

/-- The number of pages in the first chapter of a book -/
def first_chapter_pages (b : Book) : ℕ :=
  b.total_pages - b.second_chapter_pages

/-- Theorem stating that for a book with 94 total pages and 46 pages in the second chapter,
    the first chapter has 48 pages -/
theorem first_chapter_has_48_pages (b : Book)
    (h1 : b.total_pages = 94)
    (h2 : b.second_chapter_pages = 46) :
    first_chapter_pages b = 48 := by
  sorry


end NUMINAMATH_CALUDE_first_chapter_has_48_pages_l289_28999


namespace NUMINAMATH_CALUDE_percentage_relation_l289_28903

theorem percentage_relation (a b c P : ℝ) : 
  (P / 100) * a = 12 →
  (12 / 100) * b = 6 →
  c = b / a →
  c = P / 24 :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l289_28903


namespace NUMINAMATH_CALUDE_balcony_orchestra_difference_l289_28950

/-- Represents the number of tickets sold for a theater performance. -/
structure TicketSales where
  orchestra : ℕ
  balcony : ℕ

/-- Calculates the total number of tickets sold. -/
def TicketSales.total (ts : TicketSales) : ℕ :=
  ts.orchestra + ts.balcony

/-- Calculates the total revenue from ticket sales. -/
def TicketSales.revenue (ts : TicketSales) : ℕ :=
  12 * ts.orchestra + 8 * ts.balcony

/-- Theorem stating the difference between balcony and orchestra ticket sales. -/
theorem balcony_orchestra_difference (ts : TicketSales) 
  (h1 : ts.total = 350)
  (h2 : ts.revenue = 3320) :
  ts.balcony - ts.orchestra = 90 := by
  sorry


end NUMINAMATH_CALUDE_balcony_orchestra_difference_l289_28950


namespace NUMINAMATH_CALUDE_count_valid_assignments_l289_28909

/-- Represents a student --/
inductive Student : Type
| jia : Student
| other : Fin 4 → Student

/-- Represents a dormitory --/
inductive Dormitory : Type
| A : Dormitory
| B : Dormitory
| C : Dormitory

/-- An assignment of students to dormitories --/
def Assignment := Student → Dormitory

/-- Checks if an assignment is valid --/
def isValidAssignment (a : Assignment) : Prop :=
  (∃ s, a s = Dormitory.A) ∧
  (∃ s, a s = Dormitory.B) ∧
  (∃ s, a s = Dormitory.C) ∧
  (a Student.jia ≠ Dormitory.A)

/-- The number of valid assignments --/
def numValidAssignments : ℕ := sorry

theorem count_valid_assignments :
  numValidAssignments = 40 := by sorry

end NUMINAMATH_CALUDE_count_valid_assignments_l289_28909


namespace NUMINAMATH_CALUDE_percent_of_number_l289_28952

theorem percent_of_number (x : ℝ) : (26 / 100) * x = 93.6 → x = 360 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_number_l289_28952


namespace NUMINAMATH_CALUDE_percent_equality_l289_28962

theorem percent_equality (x : ℝ) : 
  (75 / 100) * 600 = (50 / 100) * x → x = 900 := by sorry

end NUMINAMATH_CALUDE_percent_equality_l289_28962


namespace NUMINAMATH_CALUDE_translation_result_l289_28997

/-- Represents a point in 2D Cartesian coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation in 2D space -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Applies a translation to a point -/
def applyTranslation (p : Point) (t : Translation) : Point :=
  ⟨p.x + t.dx, p.y + t.dy⟩

theorem translation_result :
  let A : Point := ⟨-3, 2⟩
  let t : Translation := ⟨3, -2⟩
  applyTranslation A t = ⟨0, 0⟩ := by sorry

end NUMINAMATH_CALUDE_translation_result_l289_28997


namespace NUMINAMATH_CALUDE_softball_players_count_l289_28923

theorem softball_players_count (total : ℕ) (cricket : ℕ) (hockey : ℕ) (football : ℕ) 
  (h1 : total = 50)
  (h2 : cricket = 12)
  (h3 : hockey = 17)
  (h4 : football = 11) :
  total - (cricket + hockey + football) = 10 := by
  sorry

end NUMINAMATH_CALUDE_softball_players_count_l289_28923


namespace NUMINAMATH_CALUDE_sum_of_three_reals_l289_28963

theorem sum_of_three_reals (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + 2*(y-1)*(z-1) = 85)
  (eq2 : y^2 + 2*(z-1)*(x-1) = 84)
  (eq3 : z^2 + 2*(x-1)*(y-1) = 89) :
  x + y + z = 18 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_reals_l289_28963


namespace NUMINAMATH_CALUDE_problem_solution_l289_28911

theorem problem_solution (x y : ℝ) :
  (4 * x + y = 1) →
  (y = 1 - 4 * x) ∧
  (y ≥ 0 → x ≤ 1/4) ∧
  (-1 < y ∧ y ≤ 2 → -1/4 ≤ x ∧ x < 1/2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l289_28911


namespace NUMINAMATH_CALUDE_cherry_popsicles_count_l289_28966

theorem cherry_popsicles_count (total : ℕ) (grape : ℕ) (banana : ℕ) (cherry : ℕ) :
  total = 17 → grape = 2 → banana = 2 → cherry = total - (grape + banana) → cherry = 13 := by
  sorry

end NUMINAMATH_CALUDE_cherry_popsicles_count_l289_28966


namespace NUMINAMATH_CALUDE_vector_sum_components_l289_28905

/-- Given 2D vectors a, b, and c, prove that 3a - 2b + c is equal to
    (3ax - 2bx + cx, 3ay - 2by + cy) where ax, ay, bx, by, cx, and cy
    are the respective x and y components of vectors a, b, and c. -/
theorem vector_sum_components (a b c : ℝ × ℝ) :
  3 • a - 2 • b + c = (3 * a.1 - 2 * b.1 + c.1, 3 * a.2 - 2 * b.2 + c.2) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_components_l289_28905


namespace NUMINAMATH_CALUDE_fifty_cent_items_count_l289_28983

/-- Represents the number of items at each price point -/
structure ItemCounts where
  cents50 : ℕ
  dollars2 : ℕ
  dollars4 : ℕ

/-- The proposition that the given item counts satisfy the problem conditions -/
def satisfiesConditions (counts : ItemCounts) : Prop :=
  counts.cents50 + counts.dollars2 + counts.dollars4 = 50 ∧
  50 * counts.cents50 + 200 * counts.dollars2 + 400 * counts.dollars4 = 5000

/-- The theorem stating that the only solution satisfying the conditions has 36 items at 50 cents -/
theorem fifty_cent_items_count :
  ∀ counts : ItemCounts, satisfiesConditions counts → counts.cents50 = 36 := by
  sorry

end NUMINAMATH_CALUDE_fifty_cent_items_count_l289_28983


namespace NUMINAMATH_CALUDE_arithmetic_expression_equals_two_l289_28938

theorem arithmetic_expression_equals_two :
  10 - 9 + 8 * 7 / 2 - 6 * 5 + 4 - 3 + 2 / 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equals_two_l289_28938


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l289_28908

def A : Set Nat := {1, 3, 5, 7}
def B : Set Nat := {4, 5, 6, 7}

theorem intersection_of_A_and_B : A ∩ B = {5, 7} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l289_28908


namespace NUMINAMATH_CALUDE_equilateral_triangle_cosine_l289_28933

/-- An acute angle in degrees -/
def AcuteAngle (x : ℝ) : Prop := 0 < x ∧ x < 90

/-- Cosine function for angles in degrees -/
noncomputable def cosDeg (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)

/-- Theorem: The only acute angle x (in degrees) that satisfies the conditions for an equilateral triangle with sides cos x, cos x, and cos 5x is 60° -/
theorem equilateral_triangle_cosine (x : ℝ) :
  AcuteAngle x ∧ cosDeg x = cosDeg (5 * x) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_cosine_l289_28933


namespace NUMINAMATH_CALUDE_susans_roses_l289_28984

theorem susans_roses (D : ℚ) : 
  -- Initial number of roses is 12D
  (12 * D : ℚ) > 0 →
  -- Half given to daughter, half placed in vase
  let vase_roses := 6 * D
  -- One-third of vase flowers wilted
  let unwilted_ratio := 2 / 3
  -- 12 flowers remained after removing wilted ones
  unwilted_ratio * vase_roses = 12 →
  -- Prove that D = 3
  D = 3 := by
sorry

end NUMINAMATH_CALUDE_susans_roses_l289_28984


namespace NUMINAMATH_CALUDE_counterexample_square_inequality_l289_28970

theorem counterexample_square_inequality : ∃ a b : ℝ, a > b ∧ a^2 ≤ b^2 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_square_inequality_l289_28970


namespace NUMINAMATH_CALUDE_square_root_sum_equals_ten_l289_28989

theorem square_root_sum_equals_ten : 
  Real.sqrt ((5 - 4 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 4 * Real.sqrt 2) ^ 2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_equals_ten_l289_28989


namespace NUMINAMATH_CALUDE_ibrahim_savings_is_55_l289_28904

/-- The amount of money Ibrahim has in savings -/
def ibrahimSavings (mp3Cost cdCost fatherContribution amountLacking : ℕ) : ℕ :=
  (mp3Cost + cdCost) - fatherContribution - amountLacking

/-- Theorem stating that Ibrahim's savings are 55 euros -/
theorem ibrahim_savings_is_55 :
  ibrahimSavings 120 19 20 64 = 55 := by
  sorry

end NUMINAMATH_CALUDE_ibrahim_savings_is_55_l289_28904


namespace NUMINAMATH_CALUDE_sarah_apples_to_teachers_l289_28953

/-- Calculates the number of apples given to teachers -/
def apples_to_teachers (initial : ℕ) (to_friends : ℕ) (eaten : ℕ) (left : ℕ) : ℕ :=
  initial - to_friends - eaten - left

/-- Theorem stating that Sarah gave 16 apples to teachers -/
theorem sarah_apples_to_teachers :
  apples_to_teachers 25 5 1 3 = 16 := by
  sorry

#eval apples_to_teachers 25 5 1 3

end NUMINAMATH_CALUDE_sarah_apples_to_teachers_l289_28953


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l289_28932

theorem initial_mean_calculation (n : ℕ) (wrong_value correct_value : ℝ) (new_mean : ℝ) :
  n = 40 ∧ 
  wrong_value = 75 ∧ 
  correct_value = 50 ∧ 
  new_mean = 99.075 →
  ∃ initial_mean : ℝ, 
    initial_mean = 98.45 ∧ 
    n * new_mean = n * initial_mean + (wrong_value - correct_value) :=
by sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l289_28932


namespace NUMINAMATH_CALUDE_inequality_proof_l289_28982

theorem inequality_proof (x y z : ℝ) 
  (hx : x ≠ 1) (hy : y ≠ 1) (hz : z ≠ 1) (hxyz : x * y * z = 1) : 
  x^2 / (x - 1)^2 + y^2 / (y - 1)^2 + z^2 / (z - 1)^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l289_28982


namespace NUMINAMATH_CALUDE_remainder_4873_div_29_l289_28971

theorem remainder_4873_div_29 : 4873 % 29 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_4873_div_29_l289_28971


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l289_28995

/-- The breadth of a rectangular plot with specific conditions -/
theorem rectangular_plot_breadth :
  ∀ (b l : ℝ),
  (l * b + (1/2 * (b/2) * (l/3)) = 24 * b) →  -- Area condition
  (l - b = 10) →                              -- Length-breadth difference
  (b = 158/13) :=                             -- Breadth of the plot
by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l289_28995


namespace NUMINAMATH_CALUDE_spending_difference_l289_28946

/- Define the prices and quantities -/
def basketball_price : ℝ := 29
def basketball_quantity : ℕ := 10
def baseball_price : ℝ := 2.5
def baseball_quantity : ℕ := 14
def baseball_bat_price : ℝ := 18

/- Define the total spending for each coach -/
def coach_A_spending : ℝ := basketball_price * basketball_quantity
def coach_B_spending : ℝ := baseball_price * baseball_quantity + baseball_bat_price

/- Theorem statement -/
theorem spending_difference :
  coach_A_spending - coach_B_spending = 237 := by
  sorry

end NUMINAMATH_CALUDE_spending_difference_l289_28946


namespace NUMINAMATH_CALUDE_cube_surface_area_l289_28918

/-- The surface area of a cube with edge length 2 is 24 -/
theorem cube_surface_area : 
  let edge_length : ℝ := 2
  let surface_area_formula (x : ℝ) := 6 * x * x
  surface_area_formula edge_length = 24 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_l289_28918


namespace NUMINAMATH_CALUDE_range_of_m_for_linear_system_l289_28944

/-- Given a system of linear equations and an inequality condition, 
    prove that m must be less than 1. -/
theorem range_of_m_for_linear_system (x y m : ℝ) : 
  3 * x + y = 3 * m + 1 →
  x + 2 * y = 3 →
  2 * x - y < 1 →
  m < 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_for_linear_system_l289_28944


namespace NUMINAMATH_CALUDE_nested_square_root_value_l289_28928

theorem nested_square_root_value :
  ∃ x : ℝ, x = Real.sqrt (3 - x) ∧ x = (-1 + Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_value_l289_28928


namespace NUMINAMATH_CALUDE_date_equality_l289_28929

/-- Given a variable x representing the date behind C, prove that
    the sum of the dates behind S and C equals the sum of the dates behind A and B. -/
theorem date_equality (x : ℤ) : (x - 3) + x = (x - 2) + (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_date_equality_l289_28929


namespace NUMINAMATH_CALUDE_intersection_A_B_complement_union_A_B_l289_28921

-- Define the sets A and B
def A : Set ℝ := {x | (x + 3) / (x - 7) < 0}
def B : Set ℝ := {x | |x - 4| ≤ 6}

-- Theorem for part (1)
theorem intersection_A_B : A ∩ B = {x : ℝ | -2 ≤ x ∧ x < 7} := by sorry

-- Theorem for part (2)
theorem complement_union_A_B : (A ∪ B)ᶜ = {x : ℝ | x ≤ -3 ∨ x > 10} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_complement_union_A_B_l289_28921
