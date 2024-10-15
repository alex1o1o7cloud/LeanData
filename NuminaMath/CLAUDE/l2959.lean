import Mathlib

namespace NUMINAMATH_CALUDE_wire_length_l2959_295901

/-- The length of a wire cut into two pieces, where one piece is 2/3 of the other --/
theorem wire_length (shorter_piece : ℝ) (h : shorter_piece = 27.999999999999993) : 
  ∃ (longer_piece total_length : ℝ),
    longer_piece = (2/3) * shorter_piece ∧
    total_length = shorter_piece + longer_piece ∧
    total_length = 46.66666666666666 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_l2959_295901


namespace NUMINAMATH_CALUDE_hyperbola_line_inclination_l2959_295969

/-- Given a hyperbola with equation x²/m² - y²/n² = 1 and eccentricity 2,
    prove that the angle of inclination of the line mx + ny - 1 = 0
    is either π/6 or 5π/6 -/
theorem hyperbola_line_inclination (m n : ℝ) (h_eccentricity : m^2 + n^2 = 4 * m^2) :
  let θ := Real.arctan (-m / n)
  θ = π / 6 ∨ θ = 5 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_line_inclination_l2959_295969


namespace NUMINAMATH_CALUDE_teachers_not_adjacent_arrangements_l2959_295965

def num_students : ℕ := 3
def num_teachers : ℕ := 2

def arrangement_count : ℕ := 72

theorem teachers_not_adjacent_arrangements :
  (Nat.factorial num_students) * (num_students + 1) * num_teachers = arrangement_count :=
by sorry

end NUMINAMATH_CALUDE_teachers_not_adjacent_arrangements_l2959_295965


namespace NUMINAMATH_CALUDE_train_length_l2959_295955

/-- The length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) : 
  train_speed = 72 * (1000 / 3600) → 
  crossing_time = 30 → 
  bridge_length = 350 → 
  train_speed * crossing_time - bridge_length = 250 := by sorry

end NUMINAMATH_CALUDE_train_length_l2959_295955


namespace NUMINAMATH_CALUDE_angle_between_vectors_l2959_295934

/-- The angle between two vectors given their components and projection. -/
theorem angle_between_vectors (a b : ℝ × ℝ) (h : Real.sqrt 3 * (3 : ℝ) = (b.2 : ℝ)) 
  (proj : (3 : ℝ) * (1 : ℝ) + Real.sqrt 3 * b.2 = 3 * Real.sqrt ((1 : ℝ)^2 + (Real.sqrt 3)^2)) :
  let angle := Real.arccos ((3 : ℝ) * (1 : ℝ) + Real.sqrt 3 * b.2) / 
    (Real.sqrt ((1 : ℝ)^2 + (Real.sqrt 3)^2) * Real.sqrt ((3 : ℝ)^2 + b.2^2))
  angle = π / 6 :=
by sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l2959_295934


namespace NUMINAMATH_CALUDE_fraction_sum_inequality_l2959_295905

theorem fraction_sum_inequality (a b : ℝ) (h : a * b < 0) :
  a / b + b / a ≤ -2 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_inequality_l2959_295905


namespace NUMINAMATH_CALUDE_quadratic_sum_l2959_295958

theorem quadratic_sum (a b c : ℝ) : 
  (∀ x, -3 * x^2 + 15 * x + 75 = a * (x + b)^2 + c) →
  a + b + c = 353/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2959_295958


namespace NUMINAMATH_CALUDE_value_of_M_l2959_295962

theorem value_of_M : ∃ M : ℝ, (0.12 * M = 0.60 * 1500) ∧ (M = 7500) := by
  sorry

end NUMINAMATH_CALUDE_value_of_M_l2959_295962


namespace NUMINAMATH_CALUDE_transformed_square_properties_l2959_295933

/-- A point in the xy-plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- The transformation from xy-plane to uv-plane -/
def transform (p : Point) : Point :=
  { x := p.x^2 + p.y^2,
    y := p.x^2 * p.y^2 }

/-- The unit square PQRST in the xy-plane -/
def unitSquare : Set Point :=
  {p | 0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1}

/-- The image of the unit square under the transformation -/
def transformedSquare : Set Point :=
  {q | ∃ p ∈ unitSquare, q = transform p}

/-- Definition of vertical symmetry -/
def verticallySymmetric (s : Set Point) : Prop :=
  ∀ p ∈ s, ∃ q ∈ s, q.x = p.x ∧ q.y = -p.y

/-- Definition of curved upper boundary -/
def hasCurvedUpperBoundary (s : Set Point) : Prop :=
  ∃ f : ℝ → ℝ, (∀ x, f x ≥ 0) ∧ 
    (∀ p ∈ s, p.y ≤ f p.x) ∧
    (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ ≠ f x₂)

theorem transformed_square_properties :
  verticallySymmetric transformedSquare ∧ 
  hasCurvedUpperBoundary transformedSquare :=
sorry

end NUMINAMATH_CALUDE_transformed_square_properties_l2959_295933


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2959_295999

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 3 + a 11 = 24)
  (h_a4 : a 4 = 3) :
  ∃ d : ℝ, d = 3 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2959_295999


namespace NUMINAMATH_CALUDE_solution_set_correct_l2959_295997

def solution_set := {x : ℝ | 0 < x ∧ x < 1}

theorem solution_set_correct : 
  ∀ x : ℝ, x ∈ solution_set ↔ (x * (x + 2) > 0 ∧ |x| < 1) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_correct_l2959_295997


namespace NUMINAMATH_CALUDE_max_a_value_l2959_295941

/-- A function f defined on the positive reals satisfying certain properties -/
def f : ℝ → ℝ :=
  sorry

/-- The conditions on f -/
axiom f_add (x y : ℝ) : x > 0 → y > 0 → f x + f y = f (x * y)

axiom f_neg (x : ℝ) : x > 1 → f x < 0

axiom f_ineq (x y a : ℝ) : x > 0 → y > 0 → a > 0 → 
  f (Real.sqrt (x^2 + y^2)) ≤ f (a * Real.sqrt (x * y))

/-- The theorem stating the maximum value of a -/
theorem max_a_value : 
  (∀ x y : ℝ, x > 0 → y > 0 → f (Real.sqrt (x^2 + y^2)) ≤ f (a * Real.sqrt (x * y))) →
  a ≤ Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l2959_295941


namespace NUMINAMATH_CALUDE_encryption_decryption_l2959_295994

/-- Given an encryption formula y = a^x - 2, prove that when a^3 - 2 = 6 and y = 14, x = 4 --/
theorem encryption_decryption (a : ℝ) (h1 : a^3 - 2 = 6) (y : ℝ) (h2 : y = 14) :
  ∃ x : ℝ, a^x - 2 = y ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_encryption_decryption_l2959_295994


namespace NUMINAMATH_CALUDE_hyperbola_m_value_l2959_295985

-- Define the hyperbola equation
def hyperbola_equation (x y m : ℝ) : Prop := x^2 - y^2/m^2 = 1

-- Define the condition that the conjugate axis is twice the transverse axis
def conjugate_twice_transverse (m : ℝ) : Prop := abs m = 2

-- State the theorem
theorem hyperbola_m_value : 
  ∀ m : ℝ, (∃ x y : ℝ, hyperbola_equation x y m) → conjugate_twice_transverse m → m = 2 ∨ m = -2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_m_value_l2959_295985


namespace NUMINAMATH_CALUDE_water_bucket_problem_l2959_295912

theorem water_bucket_problem (total_grams : ℕ) : 
  (total_grams % 900 = 200) ∧ 
  (total_grams / 900 = 7) → 
  (total_grams : ℚ) / 1000 = 6.5 := by
sorry

end NUMINAMATH_CALUDE_water_bucket_problem_l2959_295912


namespace NUMINAMATH_CALUDE_find_Y_value_l2959_295984

theorem find_Y_value (Y : ℝ) (h : (200 + 200 / Y) * Y = 18200) : Y = 90 := by
  sorry

end NUMINAMATH_CALUDE_find_Y_value_l2959_295984


namespace NUMINAMATH_CALUDE_expression_value_l2959_295979

theorem expression_value (x : ℝ) (h : x^2 + x - 3 = 0) :
  (x - 1)^2 - x*(x - 3) + (x + 1)*(x - 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2959_295979


namespace NUMINAMATH_CALUDE_magnitude_of_z_l2959_295930

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the given equation
def given_equation (z : ℂ) : Prop := (1 + i) * z = 3 + i

-- State the theorem
theorem magnitude_of_z (z : ℂ) (h : given_equation z) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l2959_295930


namespace NUMINAMATH_CALUDE_log_problem_l2959_295989

theorem log_problem (m : ℝ) : 
  (Real.log 4 / Real.log 3) * (Real.log 8 / Real.log 4) * (Real.log m / Real.log 8) = Real.log 16 / Real.log 4 → 
  m = 9 := by
sorry

end NUMINAMATH_CALUDE_log_problem_l2959_295989


namespace NUMINAMATH_CALUDE_computer_speed_significant_figures_l2959_295996

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Counts the number of significant figures in a scientific notation -/
def countSignificantFigures (n : ScientificNotation) : ℕ :=
  sorry

/-- The given computer speed in scientific notation -/
def computerSpeed : ScientificNotation :=
  { coefficient := 2.09
    exponent := 10 }

/-- Theorem stating that the computer speed has 3 significant figures -/
theorem computer_speed_significant_figures :
  countSignificantFigures computerSpeed = 3 := by
  sorry

end NUMINAMATH_CALUDE_computer_speed_significant_figures_l2959_295996


namespace NUMINAMATH_CALUDE_ice_cream_stacking_problem_l2959_295916

/-- The number of permutations of n distinct objects -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The problem statement -/
theorem ice_cream_stacking_problem :
  permutations 5 = 120 := by sorry

end NUMINAMATH_CALUDE_ice_cream_stacking_problem_l2959_295916


namespace NUMINAMATH_CALUDE_two_std_dev_below_mean_l2959_295903

/-- Represents a normal distribution --/
structure NormalDistribution where
  mean : ℝ
  std_dev : ℝ

/-- Calculates the value that is a given number of standard deviations away from the mean --/
def value_at_std_devs (d : NormalDistribution) (n : ℝ) : ℝ :=
  d.mean - n * d.std_dev

/-- Theorem: For a normal distribution with mean 15 and standard deviation 1.5,
    the value 2 standard deviations below the mean is 12 --/
theorem two_std_dev_below_mean :
  let d : NormalDistribution := { mean := 15, std_dev := 1.5 }
  value_at_std_devs d 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_two_std_dev_below_mean_l2959_295903


namespace NUMINAMATH_CALUDE_appended_ages_digits_l2959_295973

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def append_numbers (a b : ℕ) : ℕ := a * 100 + b

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem appended_ages_digits (j a : ℕ) :
  is_two_digit j →
  is_two_digit a →
  is_perfect_square (append_numbers j a) →
  digit_sum (append_numbers j a) = 7 →
  ∃ n : ℕ, append_numbers j a = n ∧ 1000 ≤ n ∧ n ≤ 9999 :=
sorry

end NUMINAMATH_CALUDE_appended_ages_digits_l2959_295973


namespace NUMINAMATH_CALUDE_evaluate_expression_l2959_295998

theorem evaluate_expression : 48^3 + 3*(48^2)*4 + 3*48*(4^2) + 4^3 = 140608 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2959_295998


namespace NUMINAMATH_CALUDE_initial_fish_count_l2959_295937

/-- The number of fish moved to a different tank -/
def fish_moved : ℕ := 68

/-- The number of fish remaining in the first tank -/
def fish_remaining : ℕ := 144

/-- The initial number of fish in the first tank -/
def initial_fish : ℕ := fish_moved + fish_remaining

theorem initial_fish_count : initial_fish = 212 := by
  sorry

end NUMINAMATH_CALUDE_initial_fish_count_l2959_295937


namespace NUMINAMATH_CALUDE_largest_decimal_l2959_295936

theorem largest_decimal : ∀ (a b c d e : ℝ), 
  a = 0.997 → b = 0.9797 → c = 0.97 → d = 0.979 → e = 0.9709 →
  a > b ∧ a > c ∧ a > d ∧ a > e :=
by sorry

end NUMINAMATH_CALUDE_largest_decimal_l2959_295936


namespace NUMINAMATH_CALUDE_range_of_a_l2959_295953

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 ≥ a) ∧ 
  (∃ x_0 : ℝ, x_0^2 + 2*a*x_0 + 2 - a = 0) → 
  a = 1 ∨ a ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2959_295953


namespace NUMINAMATH_CALUDE_simplify_expression_l2959_295951

theorem simplify_expression (r s : ℝ) : 120*r - 32*r + 50*s - 20*s = 88*r + 30*s := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2959_295951


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l2959_295908

/-- The atomic weight of Aluminum in g/mol -/
def Al_weight : ℝ := 26.98

/-- The atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- The atomic weight of Hydrogen in g/mol -/
def H_weight : ℝ := 1.01

/-- The number of Aluminum atoms in the compound -/
def Al_count : ℕ := 1

/-- The number of Oxygen atoms in the compound -/
def O_count : ℕ := 3

/-- The number of Hydrogen atoms in the compound -/
def H_count : ℕ := 3

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := Al_weight * Al_count + O_weight * O_count + H_weight * H_count

theorem compound_molecular_weight : molecular_weight = 78.01 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l2959_295908


namespace NUMINAMATH_CALUDE_clock_hands_minimum_time_l2959_295966

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDifferenceInMinutes (t1 t2 : Time) : ℕ :=
  (t2.hours * 60 + t2.minutes) - (t1.hours * 60 + t1.minutes)

/-- Converts minutes to Time structure -/
def minutesToTime (m : ℕ) : Time :=
  { hours := m / 60
    minutes := m % 60
    valid := by sorry }

theorem clock_hands_minimum_time :
  let t1 : Time := { hours := 0, minutes := 45, valid := by sorry }
  let t2 : Time := { hours := 3, minutes := 30, valid := by sorry }
  let diff := timeDifferenceInMinutes t1 t2
  let result := minutesToTime diff
  result.hours = 2 ∧ result.minutes = 45 := by sorry

end NUMINAMATH_CALUDE_clock_hands_minimum_time_l2959_295966


namespace NUMINAMATH_CALUDE_incircle_radius_not_less_than_one_l2959_295925

/-- Triangle ABC with sides a, b, c and incircle radius r -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  r : ℝ

/-- The theorem stating that the incircle radius of triangle ABC with BC = 3 and AC = 4 is not less than 1 -/
theorem incircle_radius_not_less_than_one (t : Triangle) (h1 : t.b = 3) (h2 : t.c = 4) : 
  t.r ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_incircle_radius_not_less_than_one_l2959_295925


namespace NUMINAMATH_CALUDE_max_min_f_on_interval_l2959_295915

-- Define the function f(x) = -3x + 1
def f (x : ℝ) : ℝ := -3 * x + 1

-- State the theorem
theorem max_min_f_on_interval :
  (∀ x ∈ Set.Icc 0 1, f x ≤ 1) ∧
  (∃ x ∈ Set.Icc 0 1, f x = 1) ∧
  (∀ x ∈ Set.Icc 0 1, f x ≥ -2) ∧
  (∃ x ∈ Set.Icc 0 1, f x = -2) :=
sorry

end NUMINAMATH_CALUDE_max_min_f_on_interval_l2959_295915


namespace NUMINAMATH_CALUDE_correct_sampling_methods_l2959_295968

/-- Represents different sampling methods --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a survey with its characteristics --/
structure Survey where
  population : ℕ
  sample_size : ℕ
  has_groups : Bool

/-- Determines the most appropriate sampling method for a given survey --/
def best_sampling_method (s : Survey) : SamplingMethod :=
  if s.has_groups then SamplingMethod.Stratified
  else if s.population > 100 then SamplingMethod.Systematic
  else SamplingMethod.SimpleRandom

/-- The yogurt box survey --/
def yogurt_survey : Survey :=
  { population := 10, sample_size := 3, has_groups := false }

/-- The audience survey --/
def audience_survey : Survey :=
  { population := 1280, sample_size := 32, has_groups := false }

/-- The school staff survey --/
def staff_survey : Survey :=
  { population := 160, sample_size := 20, has_groups := true }

theorem correct_sampling_methods :
  best_sampling_method yogurt_survey = SamplingMethod.SimpleRandom ∧
  best_sampling_method audience_survey = SamplingMethod.Systematic ∧
  best_sampling_method staff_survey = SamplingMethod.Stratified :=
sorry

end NUMINAMATH_CALUDE_correct_sampling_methods_l2959_295968


namespace NUMINAMATH_CALUDE_drawing_red_is_certain_l2959_295959

/-- Represents a ball in the box -/
inductive Ball
  | Red

/-- Represents the box containing balls -/
def Box := List Ball

/-- Defines a certain event -/
def CertainEvent (event : Prop) : Prop :=
  ∀ (outcome : Prop), event = outcome

/-- The box contains exactly two red balls -/
def TwoRedBalls (box : Box) : Prop :=
  box = [Ball.Red, Ball.Red]

/-- Drawing a ball from the box -/
def DrawBall (box : Box) : Ball :=
  match box with
  | [] => Ball.Red  -- Default case, should not occur
  | (b :: _) => b

/-- The main theorem: Drawing a red ball from a box with two red balls is a certain event -/
theorem drawing_red_is_certain (box : Box) (h : TwoRedBalls box) :
  CertainEvent (DrawBall box = Ball.Red) := by
  sorry

end NUMINAMATH_CALUDE_drawing_red_is_certain_l2959_295959


namespace NUMINAMATH_CALUDE_captain_times_proof_l2959_295923

-- Define the points and captain times for each boy
def points_A : ℕ := sorry
def points_E : ℕ := sorry
def points_B : ℕ := sorry
def captain_time_A : ℕ := sorry
def captain_time_E : ℕ := sorry
def captain_time_B : ℕ := sorry

-- Define the total travel time
def total_time : ℕ := sorry

-- State the theorem
theorem captain_times_proof :
  -- Conditions
  (points_A = points_B + 3) →
  (points_E + points_B = 15) →
  (total_time / 10 = points_A + points_E + points_B + 25) →
  (captain_time_B = 160) →
  -- Proportionality condition
  (∃ (k : ℚ), 
    captain_time_A = k * points_A ∧
    captain_time_E = k * points_E ∧
    captain_time_B = k * points_B) →
  -- Conclusion
  (captain_time_A = 200 ∧ captain_time_B = 140) :=
by sorry

end NUMINAMATH_CALUDE_captain_times_proof_l2959_295923


namespace NUMINAMATH_CALUDE_sculptures_not_on_display_sculptures_not_on_display_proof_l2959_295961

/-- Represents the total number of art pieces in the gallery -/
def total_art_pieces : ℕ := 3150

/-- Represents the fraction of art pieces on display -/
def fraction_on_display : ℚ := 1/3

/-- Represents the fraction of sculptures among displayed pieces -/
def fraction_sculptures_displayed : ℚ := 1/6

/-- Represents the fraction of paintings among pieces not on display -/
def fraction_paintings_not_displayed : ℚ := 1/3

/-- Represents that some sculptures are not on display -/
axiom some_sculptures_not_displayed : ∃ (n : ℕ), n > 0 ∧ n ≤ total_art_pieces

theorem sculptures_not_on_display : ℕ :=
  1400

theorem sculptures_not_on_display_proof : sculptures_not_on_display = 1400 := by
  sorry

end NUMINAMATH_CALUDE_sculptures_not_on_display_sculptures_not_on_display_proof_l2959_295961


namespace NUMINAMATH_CALUDE_bert_spent_nine_at_dry_cleaners_l2959_295927

/-- Represents Bert's spending problem -/
def BertSpending (initial_amount dry_cleaner_amount : ℚ) : Prop :=
  let hardware_store_amount := initial_amount / 4
  let after_hardware := initial_amount - hardware_store_amount
  let after_dry_cleaner := after_hardware - dry_cleaner_amount
  let grocery_store_amount := after_dry_cleaner / 2
  let final_amount := after_dry_cleaner - grocery_store_amount
  (initial_amount = 52) ∧
  (final_amount = 15) ∧
  (dry_cleaner_amount > 0)

/-- Proves that Bert spent $9 at the dry cleaners -/
theorem bert_spent_nine_at_dry_cleaners :
  ∃ (dry_cleaner_amount : ℚ), BertSpending 52 dry_cleaner_amount ∧ dry_cleaner_amount = 9 := by
  sorry

end NUMINAMATH_CALUDE_bert_spent_nine_at_dry_cleaners_l2959_295927


namespace NUMINAMATH_CALUDE_st_plus_tu_equals_ten_l2959_295906

/-- Represents a polygon PQRSTU -/
structure Polygon where
  area : ℝ
  pq : ℝ
  qr : ℝ
  up : ℝ
  st : ℝ
  tu : ℝ

/-- Theorem stating the sum of ST and TU in the given polygon -/
theorem st_plus_tu_equals_ten (poly : Polygon) 
  (h_area : poly.area = 64)
  (h_pq : poly.pq = 10)
  (h_qr : poly.qr = 10)
  (h_up : poly.up = 6) :
  poly.st + poly.tu = 10 := by
  sorry

end NUMINAMATH_CALUDE_st_plus_tu_equals_ten_l2959_295906


namespace NUMINAMATH_CALUDE_common_solution_l2959_295914

def m_values : List ℤ := [-5, -4, -3, -1, 0, 1, 3, 23, 124, 1000]

def equation (m x y : ℤ) : Prop :=
  (2 * m + 1) * x + (2 - 3 * m) * y + 1 - 5 * m = 0

theorem common_solution :
  ∀ m ∈ m_values, equation m 1 (-1) :=
by sorry

end NUMINAMATH_CALUDE_common_solution_l2959_295914


namespace NUMINAMATH_CALUDE_sum_a_d_equals_one_l2959_295975

theorem sum_a_d_equals_one (a b c d : ℝ) 
  (h1 : a + b = 4) 
  (h2 : b + c = 5) 
  (h3 : c + d = 3) : 
  a + d = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_a_d_equals_one_l2959_295975


namespace NUMINAMATH_CALUDE_frac_2023rd_digit_l2959_295978

-- Define the fraction
def frac : ℚ := 7 / 26

-- Define the length of the repeating decimal
def repeat_length : ℕ := 6

-- Define the position we're interested in
def position : ℕ := 2023

-- Define the function that returns the nth digit after the decimal point
noncomputable def nth_digit (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem frac_2023rd_digit :
  nth_digit position = 5 :=
sorry

end NUMINAMATH_CALUDE_frac_2023rd_digit_l2959_295978


namespace NUMINAMATH_CALUDE_remainder_theorem_remainder_is_72_l2959_295935

def f (x : ℝ) : ℝ := x^4 - 6*x^3 + 11*x^2 + 8*x - 20

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x + a) * q x + f (-a) :=
sorry

theorem remainder_is_72 : 
  ∃ q : ℝ → ℝ, ∀ x, f x = (x + 2) * q x + 72 :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_remainder_is_72_l2959_295935


namespace NUMINAMATH_CALUDE_part1_part2_l2959_295924

-- Define the points
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (2, -2)
def C : ℝ × ℝ := (4, 1)

-- Define vectors
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)

-- Part 1
theorem part1 : ∀ D : ℝ × ℝ, AB = (D.1 - C.1, D.2 - C.2) → D = (5, -4) := by sorry

-- Part 2
theorem part2 : ∀ k : ℝ, (∃ t : ℝ, t ≠ 0 ∧ (k * AB.1 - BC.1, k * AB.2 - BC.2) = (t * (AB.1 + 3 * BC.1), t * (AB.2 + 3 * BC.2))) → k = -1/3 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l2959_295924


namespace NUMINAMATH_CALUDE_smallest_positive_solution_l2959_295995

theorem smallest_positive_solution :
  ∃ (x : ℝ), x > 0 ∧ x/4 + 3/(4*x) = 1 ∧ ∀ (y : ℝ), y > 0 ∧ y/4 + 3/(4*y) = 1 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_l2959_295995


namespace NUMINAMATH_CALUDE_sleep_increase_l2959_295920

theorem sleep_increase (initial_sleep : ℝ) (increase_factor : ℝ) (final_sleep : ℝ) :
  initial_sleep = 6 →
  increase_factor = 1/3 →
  final_sleep = initial_sleep + increase_factor * initial_sleep →
  final_sleep = 8 := by
sorry

end NUMINAMATH_CALUDE_sleep_increase_l2959_295920


namespace NUMINAMATH_CALUDE_problem_proof_l2959_295907

theorem problem_proof (m n : ℕ) (h1 : m + 9 < n) 
  (h2 : (m + (m + 3) + (m + 9) + n + (n + 1) + (2*n - 1)) / 6 = n - 1) 
  (h3 : (m + 9 + n) / 2 = n - 1) : m + n = 21 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l2959_295907


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l2959_295993

theorem simplify_sqrt_expression : 
  Real.sqrt 3 - Real.sqrt 12 + Real.sqrt 27 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l2959_295993


namespace NUMINAMATH_CALUDE_factorization_of_cubic_l2959_295980

theorem factorization_of_cubic (x : ℝ) : 3 * x^3 - 27 * x = 3 * x * (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_cubic_l2959_295980


namespace NUMINAMATH_CALUDE_sequence_sum_l2959_295939

/-- Given a geometric sequence and an arithmetic sequence with specific properties, 
    prove that the sum of two terms in the arithmetic sequence equals 8. -/
theorem sequence_sum (a b : ℕ → ℝ) : 
  (∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) →  -- geometric sequence condition
  (a 3 * a 11 = 4 * a 7) →                              -- given condition for geometric sequence
  (∀ n : ℕ, b (n + 1) - b n = b (n + 2) - b (n + 1)) →  -- arithmetic sequence condition
  (a 7 = b 7) →                                         -- given condition relating both sequences
  b 5 + b 9 = 8 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l2959_295939


namespace NUMINAMATH_CALUDE_peggy_stamp_count_l2959_295967

/-- The number of stamps Peggy has -/
def peggy_stamps : ℕ := 75

/-- The number of stamps Ernie has -/
def ernie_stamps : ℕ := 3 * peggy_stamps

/-- The number of stamps Bert has -/
def bert_stamps : ℕ := 4 * ernie_stamps

theorem peggy_stamp_count : 
  bert_stamps = peggy_stamps + 825 ∧ 
  ernie_stamps = 3 * peggy_stamps ∧ 
  bert_stamps = 4 * ernie_stamps →
  peggy_stamps = 75 := by sorry

end NUMINAMATH_CALUDE_peggy_stamp_count_l2959_295967


namespace NUMINAMATH_CALUDE_monotonicity_condition_l2959_295946

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := 4 * x^2 - k * x - 8

-- Define the monotonicity property on the interval (-∞, 8]
def is_monotonic_on_interval (k : ℝ) : Prop :=
  ∀ x y, x < y ∧ y ≤ 8 → f k x < f k y ∨ f k x > f k y

-- Theorem statement
theorem monotonicity_condition (k : ℝ) :
  is_monotonic_on_interval k ↔ k ≥ 64 := by
  sorry

end NUMINAMATH_CALUDE_monotonicity_condition_l2959_295946


namespace NUMINAMATH_CALUDE_original_serving_size_l2959_295942

/-- Proves that the original serving size was 8 ounces -/
theorem original_serving_size (total_water : ℝ) (current_serving : ℝ) (serving_difference : ℕ) : 
  total_water = 64 →
  current_serving = 16 →
  (total_water / current_serving : ℝ) + serving_difference = total_water / 8 →
  8 = total_water / ((total_water / current_serving : ℝ) + serving_difference : ℝ) := by
sorry

end NUMINAMATH_CALUDE_original_serving_size_l2959_295942


namespace NUMINAMATH_CALUDE_crow_worm_consumption_l2959_295974

theorem crow_worm_consumption 
  (crows_per_hour : ℕ) 
  (worms_per_hour : ℕ) 
  (new_crows : ℕ) 
  (new_hours : ℕ) 
  (h1 : crows_per_hour = 3) 
  (h2 : worms_per_hour = 30) 
  (h3 : new_crows = 5) 
  (h4 : new_hours = 2) : 
  (worms_per_hour / crows_per_hour) * new_crows * new_hours = 100 := by
  sorry

end NUMINAMATH_CALUDE_crow_worm_consumption_l2959_295974


namespace NUMINAMATH_CALUDE_speed_ratio_is_two_l2959_295947

/-- Given a round trip with total distance 60 km, total time 6 hours, and return speed 15 km/h,
    the ratio of return speed to outbound speed is 2. -/
theorem speed_ratio_is_two 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (return_speed : ℝ) 
  (h1 : total_distance = 60) 
  (h2 : total_time = 6) 
  (h3 : return_speed = 15) : 
  return_speed / ((total_distance / 2) / (total_time - total_distance / (2 * return_speed))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_speed_ratio_is_two_l2959_295947


namespace NUMINAMATH_CALUDE_perpendicular_to_plane_implies_parallel_l2959_295963

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_to_plane_implies_parallel
  (m n : Line) (α : Plane) 
  (hm : m ≠ n)
  (hα : perpendicular m α)
  (hβ : perpendicular n α) :
  parallel m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_plane_implies_parallel_l2959_295963


namespace NUMINAMATH_CALUDE_maintenance_check_interval_l2959_295910

theorem maintenance_check_interval (original_interval : ℝ) : 
  (original_interval * 1.2 = 60) → original_interval = 50 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_check_interval_l2959_295910


namespace NUMINAMATH_CALUDE_geometric_sequence_special_ratio_l2959_295909

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Define the problem statement
theorem geometric_sequence_special_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : geometric_sequence a q)
  (h_arith : a 2 - (1/2 * a 3) = (1/2 * a 3) - a 1) :
  q = (Real.sqrt 5 + 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_special_ratio_l2959_295909


namespace NUMINAMATH_CALUDE_inequality_implication_l2959_295991

theorem inequality_implication (m a b : ℝ) : a * m^2 > b * m^2 → a > b := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l2959_295991


namespace NUMINAMATH_CALUDE_min_value_ab_l2959_295911

theorem min_value_ab (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b)
  (h4 : ∀ x y, 0 < x → x < y → a^x + b^x < a^y + b^y) :
  1 ≤ a * b :=
sorry

end NUMINAMATH_CALUDE_min_value_ab_l2959_295911


namespace NUMINAMATH_CALUDE_container_fullness_l2959_295952

theorem container_fullness 
  (capacity : ℝ) 
  (initial_percentage : ℝ) 
  (added_water : ℝ) 
  (h1 : capacity = 120)
  (h2 : initial_percentage = 0.3)
  (h3 : added_water = 54) :
  (initial_percentage * capacity + added_water) / capacity = 0.75 :=
by sorry

end NUMINAMATH_CALUDE_container_fullness_l2959_295952


namespace NUMINAMATH_CALUDE_cosA_value_l2959_295944

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- State the theorem
theorem cosA_value (t : Triangle) 
  (h : (2 * t.b - t.c) * Real.cos t.A = t.a * Real.cos t.C) : 
  Real.cos t.A = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cosA_value_l2959_295944


namespace NUMINAMATH_CALUDE_distribute_6_3_max2_l2959_295921

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute (n : ℕ) (k : ℕ) (max_per_box : ℕ) : ℕ := sorry

/-- The number of ways to distribute 6 distinguishable balls into 3 indistinguishable boxes
    with at most 2 balls per box -/
theorem distribute_6_3_max2 : distribute 6 3 2 = 100 := by sorry

end NUMINAMATH_CALUDE_distribute_6_3_max2_l2959_295921


namespace NUMINAMATH_CALUDE_max_earnings_is_250_l2959_295931

/-- Represents a plumbing job with counts of toilets, showers, and sinks to be fixed -/
structure PlumbingJob where
  toilets : ℕ
  showers : ℕ
  sinks : ℕ

/-- Calculates the earnings for a given plumbing job -/
def jobEarnings (job : PlumbingJob) : ℕ :=
  job.toilets * 50 + job.showers * 40 + job.sinks * 30

/-- The list of available jobs -/
def availableJobs : List PlumbingJob := [
  { toilets := 3, showers := 0, sinks := 3 },
  { toilets := 2, showers := 0, sinks := 5 },
  { toilets := 1, showers := 2, sinks := 3 }
]

/-- Theorem stating that the maximum earnings from the available jobs is $250 -/
theorem max_earnings_is_250 : 
  (availableJobs.map jobEarnings).maximum? = some 250 := by sorry

end NUMINAMATH_CALUDE_max_earnings_is_250_l2959_295931


namespace NUMINAMATH_CALUDE_lcm_gcd_sum_implies_divisibility_l2959_295922

theorem lcm_gcd_sum_implies_divisibility (m n : ℕ) :
  Nat.lcm m n + Nat.gcd m n = m + n → m ∣ n ∨ n ∣ m := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_sum_implies_divisibility_l2959_295922


namespace NUMINAMATH_CALUDE_fencing_cost_for_specific_plot_l2959_295986

/-- Represents a rectangular plot with its dimensions in meters -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ

/-- Calculates the perimeter of a rectangular plot -/
def perimeter (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.breadth)

/-- Calculates the total cost of fencing a plot given the cost per meter -/
def fencingCost (plot : RectangularPlot) (costPerMeter : ℝ) : ℝ :=
  costPerMeter * perimeter plot

/-- Theorem stating the total cost of fencing for a specific rectangular plot -/
theorem fencing_cost_for_specific_plot :
  let plot : RectangularPlot := { length := 60, breadth := 40 }
  let costPerMeter : ℝ := 26.5
  fencingCost plot costPerMeter = 5300 := by
  sorry

#check fencing_cost_for_specific_plot

end NUMINAMATH_CALUDE_fencing_cost_for_specific_plot_l2959_295986


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2959_295919

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the hyperbola
def hyperbola (x y a b : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1

-- Define the asymptote
def asymptote (x y : ℝ) : Prop := Real.sqrt 3 * x + y = 0

-- Theorem statement
theorem hyperbola_equation :
  ∀ (a b : ℝ), a > 0 → b > 0 →
  (∃ (x₀ y₀ : ℝ), parabola x₀ y₀ ∧ hyperbola x₀ y₀ a b) →
  (∃ (x y : ℝ), asymptote x y) →
  ∀ (x y : ℝ), hyperbola x y a b ↔ x^2 - y^2/3 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2959_295919


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l2959_295976

theorem polynomial_identity_sum_of_squares : 
  ∀ (a b c d e f : ℤ), 
  (∀ x : ℝ, 1000 * x^3 + 27 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 11090 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l2959_295976


namespace NUMINAMATH_CALUDE_greatest_real_part_of_cube_l2959_295960

theorem greatest_real_part_of_cube (z₁ z₂ z₃ z₄ z₅ : ℂ) : 
  z₁ = -1 ∧ 
  z₂ = -Real.sqrt 2 + I ∧ 
  z₃ = -1 + Real.sqrt 3 * I ∧ 
  z₄ = 2 * I ∧ 
  z₅ = -1 - Real.sqrt 3 * I → 
  (z₄^3).re ≥ (z₁^3).re ∧ 
  (z₄^3).re ≥ (z₂^3).re ∧ 
  (z₄^3).re ≥ (z₃^3).re ∧ 
  (z₄^3).re ≥ (z₅^3).re :=
by sorry

end NUMINAMATH_CALUDE_greatest_real_part_of_cube_l2959_295960


namespace NUMINAMATH_CALUDE_sams_coins_value_l2959_295929

/-- Represents the value of Sam's coins in dollars -/
def total_value : ℚ :=
  let total_coins : ℕ := 30
  let nickels : ℕ := 12
  let dimes : ℕ := total_coins - nickels
  let nickel_value : ℚ := 5 / 100
  let dime_value : ℚ := 10 / 100
  (nickels : ℚ) * nickel_value + (dimes : ℚ) * dime_value

theorem sams_coins_value : total_value = 2.40 := by
  sorry

end NUMINAMATH_CALUDE_sams_coins_value_l2959_295929


namespace NUMINAMATH_CALUDE_area_fraction_above_line_l2959_295913

/-- The fraction of the area of a square above a line -/
def fraction_above_line (square_vertices : Fin 4 → ℝ × ℝ) (line_point1 line_point2 : ℝ × ℝ) : ℚ :=
  sorry

/-- The theorem statement -/
theorem area_fraction_above_line :
  let square_vertices : Fin 4 → ℝ × ℝ := ![
    (2, 1), (5, 1), (5, 4), (2, 4)
  ]
  let line_point1 : ℝ × ℝ := (2, 3)
  let line_point2 : ℝ × ℝ := (5, 1)
  fraction_above_line square_vertices line_point1 line_point2 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_area_fraction_above_line_l2959_295913


namespace NUMINAMATH_CALUDE_jellybean_distribution_l2959_295990

theorem jellybean_distribution (total_jellybeans : ℕ) (total_recipients : ℕ) 
  (h1 : total_jellybeans = 70) (h2 : total_recipients = 5) :
  total_jellybeans / total_recipients = 14 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_distribution_l2959_295990


namespace NUMINAMATH_CALUDE_xy_equals_one_l2959_295949

theorem xy_equals_one (x y : ℝ) (h : x + y = 1/x + 1/y) (h_neq : x + y ≠ 0) : x * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_one_l2959_295949


namespace NUMINAMATH_CALUDE_caiden_roofing_cost_l2959_295917

def metal_roofing_cost (total_feet : ℕ) (free_feet : ℕ) (cost_per_foot : ℚ) 
  (discount_rate : ℚ) (sales_tax_rate : ℚ) : ℚ :=
  let paid_feet := total_feet - free_feet
  let initial_cost := paid_feet * cost_per_foot
  let discounted_cost := initial_cost * (1 - discount_rate)
  let total_before_tax := discounted_cost
  let total_cost := total_before_tax * (1 + sales_tax_rate)
  total_cost

theorem caiden_roofing_cost :
  metal_roofing_cost 300 250 8 (15/100) (5/100) = 357 := by
  sorry

end NUMINAMATH_CALUDE_caiden_roofing_cost_l2959_295917


namespace NUMINAMATH_CALUDE_eunji_class_size_l2959_295982

/-- The number of lines students stand in --/
def num_lines : ℕ := 3

/-- Eunji's position from the front of the line --/
def position_from_front : ℕ := 3

/-- Eunji's position from the back of the line --/
def position_from_back : ℕ := 6

/-- The total number of students in Eunji's line --/
def students_per_line : ℕ := position_from_front + position_from_back - 1

/-- The total number of students in Eunji's class --/
def total_students : ℕ := num_lines * students_per_line

theorem eunji_class_size : total_students = 24 := by
  sorry

end NUMINAMATH_CALUDE_eunji_class_size_l2959_295982


namespace NUMINAMATH_CALUDE_family_ages_l2959_295938

/-- Represents the ages of a family with a father, mother, and three daughters. -/
structure FamilyAges where
  father : ℕ
  mother : ℕ
  eldest : ℕ
  middle : ℕ
  youngest : ℕ

/-- The family ages satisfy the given conditions. -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  -- Total age is 90
  ages.father + ages.mother + ages.eldest + ages.middle + ages.youngest = 90 ∧
  -- Age difference between daughters is 2 years
  ages.eldest = ages.middle + 2 ∧
  ages.middle = ages.youngest + 2 ∧
  -- Mother's age is 10 years more than sum of daughters' ages
  ages.mother = ages.eldest + ages.middle + ages.youngest + 10 ∧
  -- Age difference between father and mother equals middle daughter's age
  ages.father - ages.mother = ages.middle

/-- The theorem stating the ages of the family members. -/
theorem family_ages : ∃ (ages : FamilyAges), satisfiesConditions ages ∧ 
  ages.father = 38 ∧ ages.mother = 31 ∧ ages.eldest = 9 ∧ ages.middle = 7 ∧ ages.youngest = 5 := by
  sorry

end NUMINAMATH_CALUDE_family_ages_l2959_295938


namespace NUMINAMATH_CALUDE_fraction_value_l2959_295945

theorem fraction_value (w x y z : ℝ) 
  (hx : x = 4 * y) 
  (hy : y = 3 * z) 
  (hz : z = 5 * w) : 
  x * z / (y * w) = 20 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l2959_295945


namespace NUMINAMATH_CALUDE_set_intersection_equality_l2959_295928

-- Define set A
def A : Set ℝ := {y | ∃ x > 1, y = Real.log x}

-- Define set B
def B : Set ℝ := {-2, -1, 1, 2}

-- Theorem statement
theorem set_intersection_equality : (Set.univ \ A) ∩ B = {-2, -1} := by sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l2959_295928


namespace NUMINAMATH_CALUDE_equation_satisfaction_l2959_295948

theorem equation_satisfaction (a b c : ℤ) (h1 : a = c) (h2 : b + 1 = c) :
  a * (b - c) + b * (c - a) + c * (a - b) = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_satisfaction_l2959_295948


namespace NUMINAMATH_CALUDE_phone_plan_monthly_fee_l2959_295902

theorem phone_plan_monthly_fee :
  let first_plan_per_minute : ℚ := 13/100
  let second_plan_monthly_fee : ℚ := 8
  let second_plan_per_minute : ℚ := 18/100
  let equal_minutes : ℕ := 280
  ∃ (F : ℚ),
    F + first_plan_per_minute * equal_minutes = 
    second_plan_monthly_fee + second_plan_per_minute * equal_minutes ∧
    F = 22 := by
  sorry

end NUMINAMATH_CALUDE_phone_plan_monthly_fee_l2959_295902


namespace NUMINAMATH_CALUDE_max_sum_of_product_l2959_295971

theorem max_sum_of_product (a b : ℤ) : 
  a ≠ b → a * b = -132 → a ≤ b → (∀ x y : ℤ, x ≠ y → x * y = -132 → x ≤ y → a + b ≥ x + y) → a + b = -1 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_product_l2959_295971


namespace NUMINAMATH_CALUDE_sequence_divisibility_l2959_295992

def sequence_a : ℕ → ℤ
  | 0 => 1
  | n + 1 => (sequence_a n)^2 + sequence_a n + 1

theorem sequence_divisibility (n : ℕ) : 
  (n ≥ 1) → (sequence_a n)^2 + 1 ∣ (sequence_a (n + 1))^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_divisibility_l2959_295992


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2959_295987

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a * b ≠ 0) → ¬(a ≠ 0 ∧ b ≠ 0)) ↔ ((a = 0 ∨ b = 0) → a * b = 0) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2959_295987


namespace NUMINAMATH_CALUDE_laptop_selection_theorem_l2959_295940

-- Define the number of laptops of each type
def typeA : ℕ := 4
def typeB : ℕ := 5

-- Define the total number of laptops to be selected
def selectTotal : ℕ := 3

-- Define the function to calculate the number of selections
def numSelections : ℕ := 
  Nat.choose typeA 2 * Nat.choose typeB 1 + 
  Nat.choose typeA 1 * Nat.choose typeB 2

-- Theorem statement
theorem laptop_selection_theorem : numSelections = 70 := by
  sorry

end NUMINAMATH_CALUDE_laptop_selection_theorem_l2959_295940


namespace NUMINAMATH_CALUDE_log_3_infinite_sum_equals_4_l2959_295981

theorem log_3_infinite_sum_equals_4 :
  ∃ (x : ℝ), x > 0 ∧ 3^x = x + 81 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_log_3_infinite_sum_equals_4_l2959_295981


namespace NUMINAMATH_CALUDE_person_c_payment_l2959_295904

def personA : ℕ := 560
def personB : ℕ := 350
def personC : ℕ := 180
def totalDuty : ℕ := 100

def totalMoney : ℕ := personA + personB + personC

def proportionalPayment (money : ℕ) : ℚ :=
  (totalDuty : ℚ) * (money : ℚ) / (totalMoney : ℚ)

theorem person_c_payment :
  round (proportionalPayment personC) = 17 := by
  sorry

end NUMINAMATH_CALUDE_person_c_payment_l2959_295904


namespace NUMINAMATH_CALUDE_megan_folders_l2959_295950

theorem megan_folders (initial_files : ℕ) (deleted_files : ℕ) (files_per_folder : ℕ) :
  initial_files = 93 →
  deleted_files = 21 →
  files_per_folder = 8 →
  (initial_files - deleted_files) / files_per_folder = 9 :=
by sorry

end NUMINAMATH_CALUDE_megan_folders_l2959_295950


namespace NUMINAMATH_CALUDE_number_of_pickers_l2959_295926

/-- Given information about grape harvesting, calculate the number of pickers --/
theorem number_of_pickers (drums_per_day : ℕ) (total_drums : ℕ) (total_days : ℕ) 
  (h1 : drums_per_day = 108)
  (h2 : total_drums = 6264)
  (h3 : total_days = 58)
  (h4 : total_drums = drums_per_day * total_days) :
  total_drums / drums_per_day = 58 := by
  sorry

end NUMINAMATH_CALUDE_number_of_pickers_l2959_295926


namespace NUMINAMATH_CALUDE_meaningful_expression_l2959_295988

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = x / Real.sqrt (x - 1)) ↔ x > 1 := by
sorry

end NUMINAMATH_CALUDE_meaningful_expression_l2959_295988


namespace NUMINAMATH_CALUDE_solution_set_when_a_eq_one_range_of_m_when_a_gt_one_l2959_295956

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * |x + 1| - |x - 1|

-- Part 1
theorem solution_set_when_a_eq_one :
  {x : ℝ | f 1 x < 3/2} = {x : ℝ | x < 3/4} := by sorry

-- Part 2
theorem range_of_m_when_a_gt_one (a : ℝ) (m : ℝ) :
  a > 1 →
  (∃ x : ℝ, f a x ≤ -|2*m + 1|) →
  m ∈ Set.Icc (-3/2) 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_eq_one_range_of_m_when_a_gt_one_l2959_295956


namespace NUMINAMATH_CALUDE_third_jumper_height_l2959_295964

/-- The height of Ravi's jump in inches -/
def ravi_jump : ℝ := 39

/-- The height of the first next highest jumper in inches -/
def jumper1 : ℝ := 23

/-- The height of the second next highest jumper in inches -/
def jumper2 : ℝ := 27

/-- The factor by which Ravi can jump higher than the average of the next three highest jumpers -/
def ravi_factor : ℝ := 1.5

/-- The height of the third next highest jumper in inches -/
def jumper3 : ℝ := 28

theorem third_jumper_height :
  ravi_jump = ravi_factor * ((jumper1 + jumper2 + jumper3) / 3) :=
by sorry

end NUMINAMATH_CALUDE_third_jumper_height_l2959_295964


namespace NUMINAMATH_CALUDE_paper_airplane_competition_l2959_295977

theorem paper_airplane_competition
  (a b h v m : ℝ)
  (total : a + b + h + v + m = 41)
  (matyas_least : m ≤ a ∧ m ≤ b ∧ m ≤ h ∧ m ≤ v)
  (andelka_matyas : a = m + 0.9)
  (vlada_andelka : v = a + 0.6)
  (honzik_furthest : h > a ∧ h > b ∧ h > v ∧ h > m)
  (honzik_whole : ∃ n : ℕ, h = n)
  (avg_difference : (a + v + m) / 3 = (a + b + h + v + m) / 5 - 0.2) :
  a = 8.1 ∧ b = 8 ∧ h = 9 ∧ v = 8.7 ∧ m = 7.2 := by
sorry

end NUMINAMATH_CALUDE_paper_airplane_competition_l2959_295977


namespace NUMINAMATH_CALUDE_exists_range_sum_and_even_count_611_l2959_295972

/-- Sum of integers from a to b (inclusive) -/
def sum_range (a b : ℤ) : ℤ := (b - a + 1) * (a + b) / 2

/-- Count of even integers in range [a, b] -/
def count_even (a b : ℤ) : ℤ :=
  if a % 2 = 0 && b % 2 = 0 then
    (b - a) / 2 + 1
  else
    (b - a + 1) / 2

theorem exists_range_sum_and_even_count_611 :
  ∃ a b : ℤ, sum_range a b + count_even a b = 611 :=
sorry

end NUMINAMATH_CALUDE_exists_range_sum_and_even_count_611_l2959_295972


namespace NUMINAMATH_CALUDE_range_of_m_l2959_295957

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x > (1/2) → x^2 - m*x + 4 > 0) → m < 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2959_295957


namespace NUMINAMATH_CALUDE_product_105_95_l2959_295932

theorem product_105_95 : 105 * 95 = 9975 := by
  sorry

end NUMINAMATH_CALUDE_product_105_95_l2959_295932


namespace NUMINAMATH_CALUDE_class_size_l2959_295900

theorem class_size (total : ℕ) (girls : ℕ) (boys : ℕ) :
  girls = total * 52 / 100 →
  girls = boys + 1 →
  total = girls + boys →
  total = 25 :=
by sorry

end NUMINAMATH_CALUDE_class_size_l2959_295900


namespace NUMINAMATH_CALUDE_dog_bath_time_l2959_295943

/-- Represents the time spent on various activities with a dog -/
structure DogCareTime where
  total : ℝ
  walking : ℝ
  bath : ℝ
  blowDry : ℝ

/-- Represents the walking parameters -/
structure WalkingParams where
  distance : ℝ
  speed : ℝ

/-- Theorem stating the bath time given the conditions -/
theorem dog_bath_time (t : DogCareTime) (w : WalkingParams) : 
  t.total = 60 ∧ 
  w.distance = 3 ∧ 
  w.speed = 6 ∧ 
  t.blowDry = t.bath / 2 ∧ 
  t.total = t.walking + t.bath + t.blowDry ∧ 
  t.walking = w.distance / w.speed * 60 →
  t.bath = 20 := by
  sorry


end NUMINAMATH_CALUDE_dog_bath_time_l2959_295943


namespace NUMINAMATH_CALUDE_line_symmetry_l2959_295983

-- Define the lines
def line_l (x y : ℝ) : Prop := x - y - 1 = 0
def line_l1 (x y : ℝ) : Prop := 2*x - y - 2 = 0
def line_l2 (x y : ℝ) : Prop := x - 2*y - 1 = 0

-- Define symmetry with respect to a line
def symmetric_wrt (l1 l2 l : (ℝ → ℝ → Prop)) : Prop :=
  ∀ (x y : ℝ), l1 x y ↔ ∃ (x' y' : ℝ), l2 x' y' ∧ l ((x + x')/2) ((y + y')/2)

-- Theorem statement
theorem line_symmetry :
  symmetric_wrt line_l1 line_l2 line_l :=
sorry

end NUMINAMATH_CALUDE_line_symmetry_l2959_295983


namespace NUMINAMATH_CALUDE_farmer_bean_seedlings_l2959_295970

/-- Represents the farmer's planting scenario -/
structure FarmPlanting where
  bean_seedlings_per_row : ℕ
  pumpkin_seeds : ℕ
  pumpkin_seeds_per_row : ℕ
  radishes : ℕ
  radishes_per_row : ℕ
  rows_per_bed : ℕ
  total_beds : ℕ

/-- Calculates the total number of bean seedlings -/
def total_bean_seedlings (f : FarmPlanting) : ℕ :=
  let total_rows := f.total_beds * f.rows_per_bed
  let pumpkin_rows := f.pumpkin_seeds / f.pumpkin_seeds_per_row
  let radish_rows := f.radishes / f.radishes_per_row
  let bean_rows := total_rows - pumpkin_rows - radish_rows
  bean_rows * f.bean_seedlings_per_row

/-- Theorem stating that the farmer has 64 bean seedlings -/
theorem farmer_bean_seedlings :
  ∀ (f : FarmPlanting),
  f.bean_seedlings_per_row = 8 →
  f.pumpkin_seeds = 84 →
  f.pumpkin_seeds_per_row = 7 →
  f.radishes = 48 →
  f.radishes_per_row = 6 →
  f.rows_per_bed = 2 →
  f.total_beds = 14 →
  total_bean_seedlings f = 64 := by
  sorry

end NUMINAMATH_CALUDE_farmer_bean_seedlings_l2959_295970


namespace NUMINAMATH_CALUDE_linear_program_unbounded_l2959_295918

def objective_function (x₁ x₂ x₃ x₄ : ℝ) : ℝ := x₁ - x₂ + 2*x₃ - x₄

def constraint1 (x₁ x₂ : ℝ) : Prop := x₁ + x₂ = 1
def constraint2 (x₂ x₃ x₄ : ℝ) : Prop := x₂ + x₃ - x₄ = 1
def non_negative (x : ℝ) : Prop := x ≥ 0

theorem linear_program_unbounded :
  ∀ M : ℝ, ∃ x₁ x₂ x₃ x₄ : ℝ,
    constraint1 x₁ x₂ ∧
    constraint2 x₂ x₃ x₄ ∧
    non_negative x₁ ∧
    non_negative x₂ ∧
    non_negative x₃ ∧
    non_negative x₄ ∧
    objective_function x₁ x₂ x₃ x₄ > M :=
by
  sorry


end NUMINAMATH_CALUDE_linear_program_unbounded_l2959_295918


namespace NUMINAMATH_CALUDE_operation_with_96_percent_error_l2959_295954

/-- Given a number N and an operation O(N), if the percentage error between O(N) and 5N is 96%, then O(N) = 0.2N -/
theorem operation_with_96_percent_error (N : ℝ) (O : ℝ → ℝ) :
  (|O N - 5 * N| / (5 * N) = 0.96) → O N = 0.2 * N :=
by sorry

end NUMINAMATH_CALUDE_operation_with_96_percent_error_l2959_295954
