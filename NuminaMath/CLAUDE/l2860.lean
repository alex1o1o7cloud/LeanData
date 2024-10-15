import Mathlib

namespace NUMINAMATH_CALUDE_point_outside_circle_l2860_286070

theorem point_outside_circle (a b : ℝ) :
  (∃ x y, x^2 + y^2 = 1 ∧ a*x + b*y = 1) →
  a^2 + b^2 > 1 :=
sorry

end NUMINAMATH_CALUDE_point_outside_circle_l2860_286070


namespace NUMINAMATH_CALUDE_company_workers_problem_l2860_286071

theorem company_workers_problem (total_workers : ℕ) 
  (h1 : total_workers % 3 = 0)  -- Ensures total_workers is divisible by 3
  (h2 : total_workers ≠ 0)      -- Ensures total_workers is not zero
  (h3 : (total_workers / 3) % 5 = 0)  -- Ensures workers without plan is divisible by 5
  (h4 : (2 * total_workers / 3) % 5 = 0)  -- Ensures workers with plan is divisible by 5
  (h5 : 40 * (2 * total_workers / 3) / 100 = 128)  -- 128 male workers
  : (7 * total_workers / 15 : ℕ) = 224 := by
  sorry

end NUMINAMATH_CALUDE_company_workers_problem_l2860_286071


namespace NUMINAMATH_CALUDE_birthday_party_attendees_l2860_286074

theorem birthday_party_attendees :
  ∀ (n : ℕ),
  (12 * (n + 2) = 16 * n) →
  n = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_birthday_party_attendees_l2860_286074


namespace NUMINAMATH_CALUDE_fraction_is_one_fifth_l2860_286045

/-- The total number of states in the collection -/
def total_states : ℕ := 50

/-- The number of states that joined the union between 1790 and 1809 -/
def states_1790_1809 : ℕ := 10

/-- The fraction of states that joined between 1790 and 1809 -/
def fraction_1790_1809 : ℚ := states_1790_1809 / total_states

theorem fraction_is_one_fifth : fraction_1790_1809 = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_is_one_fifth_l2860_286045


namespace NUMINAMATH_CALUDE_complex_purely_imaginary_l2860_286087

theorem complex_purely_imaginary (a : ℝ) : 
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  ((a : ℂ) + Complex.I) / ((1 : ℂ) + 2 * Complex.I) = Complex.I * ((1 - 2 * a : ℝ) / 5) →
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_complex_purely_imaginary_l2860_286087


namespace NUMINAMATH_CALUDE_quadratic_roots_d_value_l2860_286004

theorem quadratic_roots_d_value (d : ℝ) : 
  (∀ x : ℝ, x^2 + 7*x + d = 0 ↔ x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2) →
  d = 9.8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_d_value_l2860_286004


namespace NUMINAMATH_CALUDE_other_factor_l2860_286009

def f (k : ℝ) (x : ℝ) : ℝ := x^4 - x^3 - 18*x^2 + 52*x + k

theorem other_factor (k : ℝ) : 
  (∃ c : ℝ, ∀ x : ℝ, f k x = (x - 2) * c) → 
  (∃ d : ℝ, ∀ x : ℝ, f k x = (x + 5) * d) :=
sorry

end NUMINAMATH_CALUDE_other_factor_l2860_286009


namespace NUMINAMATH_CALUDE_sequence_difference_l2860_286058

theorem sequence_difference (x : ℕ → ℕ)
  (h1 : x 1 = 1)
  (h2 : ∀ n, x n < x (n + 1))
  (h3 : ∀ n, x (n + 1) ≤ 2 * n) :
  ∀ k : ℕ, k > 0 → ∃ i j, k = x i - x j :=
by sorry

end NUMINAMATH_CALUDE_sequence_difference_l2860_286058


namespace NUMINAMATH_CALUDE_angle_measure_proof_l2860_286033

theorem angle_measure_proof (x : ℝ) : 
  (90 - x = (180 - x) / 2 - 25) → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l2860_286033


namespace NUMINAMATH_CALUDE_existence_of_special_number_l2860_286013

/-- Given a positive integer, returns the sum of its digits. -/
def sum_of_digits (m : ℕ+) : ℕ := sorry

/-- Given a positive integer, returns the number of its digits. -/
def num_digits (m : ℕ+) : ℕ := sorry

/-- Checks if all digits of a positive integer are non-zero. -/
def all_digits_nonzero (m : ℕ+) : Prop := sorry

theorem existence_of_special_number :
  ∀ n : ℕ+, ∃ m : ℕ+,
    (num_digits m = n) ∧
    (all_digits_nonzero m) ∧
    (m.val % sum_of_digits m = 0) :=
sorry

end NUMINAMATH_CALUDE_existence_of_special_number_l2860_286013


namespace NUMINAMATH_CALUDE_f_is_quadratic_l2860_286014

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation -/
def f (x : ℝ) : ℝ := x^2 + 2*x - 5

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

#check f_is_quadratic

end NUMINAMATH_CALUDE_f_is_quadratic_l2860_286014


namespace NUMINAMATH_CALUDE_unique_pair_sum_and_quotient_l2860_286020

theorem unique_pair_sum_and_quotient :
  ∃! (x y : ℕ), x + y = 2015 ∧ ∃ (s : ℕ), x = 25 * y + s ∧ s < y := by
  sorry

end NUMINAMATH_CALUDE_unique_pair_sum_and_quotient_l2860_286020


namespace NUMINAMATH_CALUDE_midpoint_quadrilateral_area_in_regular_hexagon_l2860_286098

/-- Represents a regular hexagon -/
structure RegularHexagon :=
  (side_length : ℝ)

/-- Represents the quadrilateral formed by joining midpoints of non-adjacent sides -/
structure MidpointQuadrilateral :=
  (hexagon : RegularHexagon)

/-- The area of the midpoint quadrilateral in a regular hexagon -/
def midpoint_quadrilateral_area (q : MidpointQuadrilateral) : ℝ :=
  q.hexagon.side_length * q.hexagon.side_length

theorem midpoint_quadrilateral_area_in_regular_hexagon 
  (h : RegularHexagon) 
  (hside : h.side_length = 12) :
  midpoint_quadrilateral_area ⟨h⟩ = 144 := by
  sorry

#check midpoint_quadrilateral_area_in_regular_hexagon

end NUMINAMATH_CALUDE_midpoint_quadrilateral_area_in_regular_hexagon_l2860_286098


namespace NUMINAMATH_CALUDE_equation_solution_l2860_286060

theorem equation_solution :
  ∃! x : ℚ, (x + 2 ≠ 0) ∧ ((x^2 + 2*x + 3) / (x + 2) = x + 4) :=
by
  -- The unique solution is x = -5/4
  use -5/4
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2860_286060


namespace NUMINAMATH_CALUDE_num_valid_sequences_is_377_l2860_286001

/-- Represents a sequence of A's and B's -/
inductive ABSequence
  | A : ABSequence
  | B : ABSequence
  | cons : ABSequence → ABSequence → ABSequence

/-- Returns true if the given sequence satisfies the run length conditions -/
def validSequence : ABSequence → Bool :=
  sorry

/-- Returns the length of the given sequence -/
def sequenceLength : ABSequence → Nat :=
  sorry

/-- Returns true if the given sequence has length 15 and satisfies the run length conditions -/
def validSequenceOfLength15 (s : ABSequence) : Bool :=
  validSequence s ∧ sequenceLength s = 15

/-- The number of valid sequences of length 15 -/
def numValidSequences : Nat :=
  sorry

theorem num_valid_sequences_is_377 : numValidSequences = 377 := by
  sorry

end NUMINAMATH_CALUDE_num_valid_sequences_is_377_l2860_286001


namespace NUMINAMATH_CALUDE_count_valid_numbers_l2860_286064

def is_valid_number (n : ℕ) : Prop :=
  let tens := n / 10
  let units := n % 10
  let a := tens + units
  10 ≤ n ∧ n < 100 ∧
  (3*n % 10 + 5*n % 10 + 7*n % 10 + 9*n % 10 = a)

theorem count_valid_numbers :
  ∃ (S : Finset ℕ), (∀ n ∈ S, is_valid_number n) ∧ S.card = 3 :=
sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l2860_286064


namespace NUMINAMATH_CALUDE_set_equality_l2860_286031

def U := Set ℝ

def M : Set ℝ := {x | x > -1}

def N : Set ℝ := {x | -2 < x ∧ x < 3}

theorem set_equality : {x : ℝ | x ≤ -2} = (M ∪ N)ᶜ := by sorry

end NUMINAMATH_CALUDE_set_equality_l2860_286031


namespace NUMINAMATH_CALUDE_initial_men_employed_is_300_l2860_286023

/-- Represents the highway construction scenario --/
structure HighwayConstruction where
  totalLength : ℝ
  totalDays : ℕ
  initialHoursPerDay : ℕ
  daysWorked : ℕ
  workCompleted : ℝ
  additionalMen : ℕ
  newHoursPerDay : ℕ

/-- Calculates the initial number of men employed --/
def initialMenEmployed (h : HighwayConstruction) : ℕ :=
  sorry

/-- Theorem stating that the initial number of men employed is 300 --/
theorem initial_men_employed_is_300 (h : HighwayConstruction) 
  (h_total_length : h.totalLength = 2)
  (h_total_days : h.totalDays = 50)
  (h_initial_hours : h.initialHoursPerDay = 8)
  (h_days_worked : h.daysWorked = 25)
  (h_work_completed : h.workCompleted = 1/3)
  (h_additional_men : h.additionalMen = 60)
  (h_new_hours : h.newHoursPerDay = 10) :
  initialMenEmployed h = 300 :=
sorry

end NUMINAMATH_CALUDE_initial_men_employed_is_300_l2860_286023


namespace NUMINAMATH_CALUDE_min_tangent_length_l2860_286092

/-- The minimum length of a tangent from a point on the line x - y + 1 = 0 to the circle (x - 2)² + (y + 1)² = 1 is √7 -/
theorem min_tangent_length (x y : ℝ) : 
  let line := {(x, y) | x - y + 1 = 0}
  let circle := {(x, y) | (x - 2)^2 + (y + 1)^2 = 1}
  let tangent_length (p : ℝ × ℝ) := 
    Real.sqrt ((p.1 - 2)^2 + (p.2 + 1)^2 - 1)
  ∃ (p : ℝ × ℝ), p ∈ line ∧ 
    ∀ (q : ℝ × ℝ), q ∈ line → tangent_length p ≤ tangent_length q ∧
    tangent_length p = Real.sqrt 7 :=
by sorry


end NUMINAMATH_CALUDE_min_tangent_length_l2860_286092


namespace NUMINAMATH_CALUDE_existence_of_a_l2860_286069

theorem existence_of_a (p : ℕ) (h_prime : Nat.Prime p) (h_ge_5 : p ≥ 5) :
  ∃ a : ℕ, 1 ≤ a ∧ a ≤ p - 2 ∧
    ¬(p^2 ∣ a^(p-1) - 1) ∧ ¬(p^2 ∣ (a+1)^(p-1) - 1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_a_l2860_286069


namespace NUMINAMATH_CALUDE_num_new_candles_l2860_286095

/-- The amount of wax left in a candle as a percentage of its original weight -/
def waxLeftPercentage : ℚ := 1 / 10

/-- The weight of a large candle in ounces -/
def largeCandle : ℚ := 20

/-- The weight of a medium candle in ounces -/
def mediumCandle : ℚ := 5

/-- The weight of a small candle in ounces -/
def smallCandle : ℚ := 1

/-- The number of large candles -/
def numLargeCandles : ℕ := 5

/-- The number of medium candles -/
def numMediumCandles : ℕ := 5

/-- The number of small candles -/
def numSmallCandles : ℕ := 25

/-- The weight of a new candle to be made in ounces -/
def newCandleWeight : ℚ := 5

/-- Theorem: The number of new candles that can be made is 3 -/
theorem num_new_candles :
  (waxLeftPercentage * (numLargeCandles * largeCandle + 
                        numMediumCandles * mediumCandle + 
                        numSmallCandles * smallCandle)) / newCandleWeight = 3 := by
  sorry

end NUMINAMATH_CALUDE_num_new_candles_l2860_286095


namespace NUMINAMATH_CALUDE_fuel_mixture_problem_l2860_286017

/-- Proves the volume of fuel A in a partially filled tank --/
theorem fuel_mixture_problem (tank_capacity : ℝ) (ethanol_a : ℝ) (ethanol_b : ℝ) (total_ethanol : ℝ) :
  tank_capacity = 212 →
  ethanol_a = 0.12 →
  ethanol_b = 0.16 →
  total_ethanol = 30 →
  ∃ (volume_a : ℝ), volume_a = 98 ∧
    ∃ (volume_b : ℝ), volume_a + volume_b = tank_capacity ∧
      ethanol_a * volume_a + ethanol_b * volume_b = total_ethanol :=
by
  sorry

end NUMINAMATH_CALUDE_fuel_mixture_problem_l2860_286017


namespace NUMINAMATH_CALUDE_simplify_expression_l2860_286044

theorem simplify_expression : 
  (Real.sqrt (Real.sqrt 81) - Real.sqrt (8 + 1/2))^2 = 35/2 - 3 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2860_286044


namespace NUMINAMATH_CALUDE_express_delivery_growth_rate_l2860_286016

/-- Proves that the equation 5000(1+x)^2 = 7500 correctly represents the average annual growth rate
    for an initial value of 5000, a final value of 7500, over a 2-year period. -/
theorem express_delivery_growth_rate (x : ℝ) : 
  (5000 : ℝ) * (1 + x)^2 = 7500 ↔ 
  (∃ (initial final : ℝ) (years : ℕ), 
    initial = 5000 ∧ 
    final = 7500 ∧ 
    years = 2 ∧ 
    final = initial * (1 + x)^years) :=
by sorry

end NUMINAMATH_CALUDE_express_delivery_growth_rate_l2860_286016


namespace NUMINAMATH_CALUDE_cube_surface_area_l2860_286043

/-- Given a cube with volume 729 cubic centimeters, its surface area is 486 square centimeters. -/
theorem cube_surface_area (volume : ℝ) (side : ℝ) : 
  volume = 729 → 
  volume = side ^ 3 → 
  6 * side ^ 2 = 486 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_l2860_286043


namespace NUMINAMATH_CALUDE_intersection_M_N_l2860_286080

def M : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

def N : Set ℕ := {x | Real.sqrt (2^x - 1) < 5}

theorem intersection_M_N : M ∩ N = {1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2860_286080


namespace NUMINAMATH_CALUDE_sine_cosine_values_l2860_286028

def angle_on_line (α : Real) : Prop :=
  ∃ (x y : Real), y = Real.sqrt 3 * x ∧ 
  (Real.cos α = x / Real.sqrt (x^2 + y^2)) ∧
  (Real.sin α = y / Real.sqrt (x^2 + y^2))

theorem sine_cosine_values (α : Real) (h : angle_on_line α) :
  (Real.sin α = Real.sqrt 3 / 2 ∧ Real.cos α = 1 / 2) ∨
  (Real.sin α = -Real.sqrt 3 / 2 ∧ Real.cos α = -1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_values_l2860_286028


namespace NUMINAMATH_CALUDE_count_pairs_eq_738_l2860_286046

/-- The number of pairs (a, b) with 1 ≤ a < b ≤ 57 such that a^2 mod 57 < b^2 mod 57 -/
def count_pairs : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ =>
    let (a, b) := p
    1 ≤ a ∧ a < b ∧ b ≤ 57 ∧ (a^2 % 57 < b^2 % 57))
    (Finset.product (Finset.range 58) (Finset.range 58))).card

theorem count_pairs_eq_738 : count_pairs = 738 := by
  sorry

end NUMINAMATH_CALUDE_count_pairs_eq_738_l2860_286046


namespace NUMINAMATH_CALUDE_sqrt_five_irrational_l2860_286067

theorem sqrt_five_irrational : Irrational (Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_irrational_l2860_286067


namespace NUMINAMATH_CALUDE_mystery_number_sum_l2860_286035

theorem mystery_number_sum : Int → Prop :=
  fun result =>
    let mystery_number : Int := 47
    let added_number : Int := 45
    result = mystery_number + added_number

#check mystery_number_sum 92

end NUMINAMATH_CALUDE_mystery_number_sum_l2860_286035


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2860_286039

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ) : 
  (∀ n, S n = (n * (a 1 + a n)) / 2) →  -- Definition of sum for arithmetic sequence
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- Definition of arithmetic sequence
  m ≥ 2 →
  S (m - 1) = 16 →
  S m = 25 →
  S (m + 2) = 49 →
  m = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2860_286039


namespace NUMINAMATH_CALUDE_parabola_hyperbola_configuration_l2860_286027

/-- Theorem: Value of 'a' for a specific parabola and hyperbola configuration -/
theorem parabola_hyperbola_configuration (p t a : ℝ) : 
  p > 0 → 
  t > 0 → 
  a > 0 → 
  t^2 = 2*p*1 → 
  (1 + p/2)^2 + t^2 = 5^2 → 
  (∃ k : ℝ, k = 4/(1+a) ∧ k = 3/a) → 
  a = 3 :=
by sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_configuration_l2860_286027


namespace NUMINAMATH_CALUDE_all_points_on_line_l2860_286010

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line defined by two other points -/
def isOnLine (p : Point) (p1 : Point) (p2 : Point) : Prop :=
  (p.y - p1.y) * (p2.x - p1.x) = (p2.y - p1.y) * (p.x - p1.x)

theorem all_points_on_line :
  let p1 : Point := ⟨8, 2⟩
  let p2 : Point := ⟨2, -10⟩
  let points : List Point := [⟨5, -4⟩, ⟨4, -6⟩, ⟨10, 6⟩, ⟨0, -14⟩, ⟨1, -12⟩]
  ∀ p ∈ points, isOnLine p p1 p2 := by
  sorry

end NUMINAMATH_CALUDE_all_points_on_line_l2860_286010


namespace NUMINAMATH_CALUDE_m_range_l2860_286000

/-- A circle in a 2D Cartesian coordinate system --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Point A on circle C --/
def A : ℝ × ℝ := (3, 2)

/-- Circle C --/
def C : Circle := { center := (3, 4), radius := 2 }

/-- First fold line equation --/
def foldLine1 (x y : ℝ) : Prop := x - y + 1 = 0

/-- Second fold line equation --/
def foldLine2 (x y : ℝ) : Prop := x + y - 7 = 0

/-- Point M on x-axis --/
def M (m : ℝ) : ℝ × ℝ := (-m, 0)

/-- Point N on x-axis --/
def N (m : ℝ) : ℝ × ℝ := (m, 0)

/-- Theorem stating the range of m --/
theorem m_range : 
  ∀ m : ℝ, 
  (∃ P : ℝ × ℝ, 
    (P.1 - C.center.1)^2 + (P.2 - C.center.2)^2 = C.radius^2 ∧ 
    (P.1 - (M m).1)^2 + (P.2 - (M m).2)^2 = (P.1 - (N m).1)^2 + (P.2 - (N m).2)^2
  ) ↔ 3 ≤ m ∧ m ≤ 7 := by sorry

end NUMINAMATH_CALUDE_m_range_l2860_286000


namespace NUMINAMATH_CALUDE_brendas_age_l2860_286063

/-- Given the ages of Addison, Brenda, and Janet, prove that Brenda is 3 years old -/
theorem brendas_age (A B J : ℕ) 
  (h1 : A = 4 * B)     -- Addison's age is four times Brenda's age
  (h2 : J = B + 9)     -- Janet is nine years older than Brenda
  (h3 : A = J)         -- Addison and Janet are twins (same age)
  : B = 3 := by        -- Prove that Brenda's age (B) is 3
sorry


end NUMINAMATH_CALUDE_brendas_age_l2860_286063


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l2860_286097

theorem minimum_value_theorem (x : ℝ) (h : x > -2) :
  x + 1 / (x + 2) ≥ 0 ∧ ∃ y > -2, y + 1 / (y + 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l2860_286097


namespace NUMINAMATH_CALUDE_arrangements_count_l2860_286038

/-- Represents the number of male students -/
def num_male_students : ℕ := 3

/-- Represents the number of female students -/
def num_female_students : ℕ := 3

/-- Represents the total number of students -/
def total_students : ℕ := num_male_students + num_female_students

/-- Represents whether female students can stand at the ends of the row -/
def female_at_ends : Prop := False

/-- Represents whether female students A and B can be adjacent to female student C -/
def female_AB_adjacent_C : Prop := False

/-- Calculates the number of different arrangements given the conditions -/
def num_arrangements : ℕ := 144

/-- Theorem stating that the number of arrangements is 144 given the conditions -/
theorem arrangements_count :
  num_male_students = 3 ∧
  num_female_students = 3 ∧
  total_students = num_male_students + num_female_students ∧
  ¬female_at_ends ∧
  ¬female_AB_adjacent_C →
  num_arrangements = 144 :=
by sorry

end NUMINAMATH_CALUDE_arrangements_count_l2860_286038


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2860_286037

theorem partial_fraction_decomposition (x : ℝ) 
  (h1 : x ≠ 7/8) (h2 : x ≠ 4/5) (h3 : x ≠ 1/2) :
  (306 * x^2 - 450 * x + 162) / ((8*x-7)*(5*x-4)*(2*x-1)) = 
  9 / (8*x-7) + 6 / (5*x-4) + 3 / (2*x-1) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2860_286037


namespace NUMINAMATH_CALUDE_double_room_cost_is_60_l2860_286084

/-- Represents the hotel booking scenario -/
structure HotelBooking where
  total_rooms : ℕ
  single_room_cost : ℕ
  total_revenue : ℕ
  single_rooms_booked : ℕ

/-- Calculates the cost of a double room given the hotel booking information -/
def double_room_cost (booking : HotelBooking) : ℕ :=
  let double_rooms := booking.total_rooms - booking.single_rooms_booked
  let single_room_revenue := booking.single_rooms_booked * booking.single_room_cost
  let double_room_revenue := booking.total_revenue - single_room_revenue
  double_room_revenue / double_rooms

/-- Theorem stating that the double room cost is 60 for the given scenario -/
theorem double_room_cost_is_60 (booking : HotelBooking) 
  (h1 : booking.total_rooms = 260)
  (h2 : booking.single_room_cost = 35)
  (h3 : booking.total_revenue = 14000)
  (h4 : booking.single_rooms_booked = 64) :
  double_room_cost booking = 60 := by
  sorry

end NUMINAMATH_CALUDE_double_room_cost_is_60_l2860_286084


namespace NUMINAMATH_CALUDE_fifth_flower_is_e_l2860_286012

def flowers := ['a', 'b', 'c', 'd', 'e', 'f', 'g']

theorem fifth_flower_is_e : flowers[4] = 'e' := by
  sorry

end NUMINAMATH_CALUDE_fifth_flower_is_e_l2860_286012


namespace NUMINAMATH_CALUDE_polar_bears_research_l2860_286026

theorem polar_bears_research (time_per_round : ℕ) (sunday_rounds : ℕ) (total_time : ℕ) :
  time_per_round = 30 →
  sunday_rounds = 15 →
  total_time = 780 →
  ∃ (saturday_additional_rounds : ℕ),
    saturday_additional_rounds = 10 ∧
    total_time = time_per_round * (1 + saturday_additional_rounds + sunday_rounds) :=
by sorry

end NUMINAMATH_CALUDE_polar_bears_research_l2860_286026


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_sum_of_numerator_and_denominator_l2860_286061

def repeating_decimal : ℚ := 0.363636

theorem repeating_decimal_as_fraction :
  repeating_decimal = 4 / 11 :=
sorry

theorem sum_of_numerator_and_denominator :
  (4 : ℕ) + 11 = 15 :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_sum_of_numerator_and_denominator_l2860_286061


namespace NUMINAMATH_CALUDE_difference_of_squares_l2860_286089

theorem difference_of_squares (a b : ℝ) : (a + b) * (a - b) = a^2 - b^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2860_286089


namespace NUMINAMATH_CALUDE_books_sold_l2860_286062

theorem books_sold (initial_books final_books : ℕ) 
  (h1 : initial_books = 255)
  (h2 : final_books = 145) :
  initial_books - final_books = 110 := by
  sorry

end NUMINAMATH_CALUDE_books_sold_l2860_286062


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2860_286079

/-- Given a geometric sequence {a_n} where a₄ = 4, prove that a₂ * a₆ = 16 -/
theorem geometric_sequence_product (a : ℕ → ℝ) : 
  (∀ n m : ℕ, a (n + m) = a n * a m) →  -- geometric sequence property
  a 4 = 4 →                            -- given condition
  a 2 * a 6 = 16 :=                    -- theorem to prove
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2860_286079


namespace NUMINAMATH_CALUDE_B_roster_l2860_286052

def A : Set ℤ := {-2, 2, 3, 4}

def B : Set ℤ := {x | ∃ t ∈ A, x = t^2}

theorem B_roster : B = {4, 9, 16} := by
  sorry

end NUMINAMATH_CALUDE_B_roster_l2860_286052


namespace NUMINAMATH_CALUDE_merchant_pricing_strategy_l2860_286003

/-- Represents the pricing strategy of a merchant --/
structure MerchantPricing where
  list_price : ℝ
  purchase_discount : ℝ
  marked_price : ℝ
  selling_discount : ℝ
  profit_margin : ℝ

/-- Calculates the purchase price given the list price and purchase discount --/
def purchase_price (mp : MerchantPricing) : ℝ :=
  mp.list_price * (1 - mp.purchase_discount)

/-- Calculates the selling price given the marked price and selling discount --/
def selling_price (mp : MerchantPricing) : ℝ :=
  mp.marked_price * (1 - mp.selling_discount)

/-- Checks if the pricing strategy satisfies the profit margin requirement --/
def satisfies_profit_margin (mp : MerchantPricing) : Prop :=
  selling_price mp - purchase_price mp = mp.profit_margin * selling_price mp

/-- The main theorem to prove --/
theorem merchant_pricing_strategy (mp : MerchantPricing) 
  (h1 : mp.purchase_discount = 0.3)
  (h2 : mp.selling_discount = 0.2)
  (h3 : mp.profit_margin = 0.2)
  (h4 : satisfies_profit_margin mp) :
  mp.marked_price / mp.list_price = 1.09375 := by
  sorry

end NUMINAMATH_CALUDE_merchant_pricing_strategy_l2860_286003


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2860_286099

theorem inequality_solution_set (x : ℝ) : 
  |5 - 2*x| - 1 > 0 ↔ x < 2 ∨ x > 3 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2860_286099


namespace NUMINAMATH_CALUDE_simplify_sqrt_product_l2860_286068

theorem simplify_sqrt_product : 
  Real.sqrt (5 * 3) * Real.sqrt (3^4 * 5^2) = 225 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_product_l2860_286068


namespace NUMINAMATH_CALUDE_common_number_in_overlapping_lists_l2860_286059

theorem common_number_in_overlapping_lists (list : List ℝ) : 
  list.length = 8 →
  (list.take 5).sum / 5 = 6 →
  (list.drop 3).sum / 5 = 9 →
  list.sum / 8 = 7.5 →
  ∃ x ∈ list.take 5 ∩ list.drop 3, x = 7.5 :=
by sorry

end NUMINAMATH_CALUDE_common_number_in_overlapping_lists_l2860_286059


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l2860_286040

theorem initial_mean_calculation (n : ℕ) (wrong_value correct_value : ℝ) (new_mean : ℝ) :
  n = 50 ∧ 
  wrong_value = 23 ∧ 
  correct_value = 48 ∧ 
  new_mean = 41.5 →
  ∃ (initial_mean : ℝ),
    initial_mean * n + (correct_value - wrong_value) = new_mean * n ∧
    initial_mean = 41 :=
by sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l2860_286040


namespace NUMINAMATH_CALUDE_digit_equation_solution_l2860_286002

theorem digit_equation_solution :
  ∃ (a b c d e : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧
    (5^a) + (100*b + 10*c + 3) = (1000*d + 100*e + 1) := by
  sorry

end NUMINAMATH_CALUDE_digit_equation_solution_l2860_286002


namespace NUMINAMATH_CALUDE_cyclist_speed_north_l2860_286019

/-- The speed of the cyclist going north -/
def speed_north : ℝ := 10

/-- The speed of the cyclist going south -/
def speed_south : ℝ := 40

/-- The time taken -/
def time : ℝ := 1

/-- The distance between the cyclists after the given time -/
def distance : ℝ := 50

/-- Theorem stating that the speed of the cyclist going north is 10 km/h -/
theorem cyclist_speed_north : 
  speed_north + speed_south = distance / time :=
by sorry

end NUMINAMATH_CALUDE_cyclist_speed_north_l2860_286019


namespace NUMINAMATH_CALUDE_factorization_mn_minus_9m_l2860_286078

theorem factorization_mn_minus_9m (m n : ℝ) : m * n - 9 * m = m * (n - 9) := by
  sorry

end NUMINAMATH_CALUDE_factorization_mn_minus_9m_l2860_286078


namespace NUMINAMATH_CALUDE_yard_area_l2860_286021

def yard_length : ℝ := 20
def yard_width : ℝ := 18
def square_cutout_side : ℝ := 4
def rect_cutout_length : ℝ := 2
def rect_cutout_width : ℝ := 5

theorem yard_area : 
  yard_length * yard_width - 
  square_cutout_side * square_cutout_side - 
  rect_cutout_length * rect_cutout_width = 334 := by
sorry

end NUMINAMATH_CALUDE_yard_area_l2860_286021


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_max_area_l2860_286081

/-- A quadrilateral with side lengths a, b, c, d -/
structure Quadrilateral (a b c d : ℝ) where
  angle_sum : ℝ -- Sum of all interior angles
  area : ℝ -- Area of the quadrilateral

/-- Definition of a cyclic quadrilateral -/
def is_cyclic (q : Quadrilateral a b c d) : Prop :=
  q.angle_sum = 2 * Real.pi

/-- Theorem: Among all quadrilaterals with given side lengths, 
    the cyclic quadrilateral has the largest area -/
theorem cyclic_quadrilateral_max_area 
  {a b c d : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  ∀ q : Quadrilateral a b c d, 
    ∃ q_cyclic : Quadrilateral a b c d, 
      is_cyclic q_cyclic ∧ q.area ≤ q_cyclic.area :=
sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_max_area_l2860_286081


namespace NUMINAMATH_CALUDE_square_sum_product_equality_l2860_286090

theorem square_sum_product_equality (a b c d : ℝ) :
  (a^2 + b^2) * (c^2 + d^2) = (a*c + b*d)^2 + (a*d - b*c)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_product_equality_l2860_286090


namespace NUMINAMATH_CALUDE_intersection_area_is_pi_l2860_286056

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- Define set M
def M : Set (ℝ × ℝ) := {p | f p.1 + f p.2 ≤ 0}

-- Define set N
def N : Set (ℝ × ℝ) := {p | f p.1 - f p.2 ≥ 0}

-- Theorem statement
theorem intersection_area_is_pi : MeasureTheory.volume (M ∩ N) = π := by sorry

end NUMINAMATH_CALUDE_intersection_area_is_pi_l2860_286056


namespace NUMINAMATH_CALUDE_line_symmetry_l2860_286018

-- Define the lines
def original_line (x y : ℝ) : Prop := 2*x - y + 3 = 0
def reference_line (x y : ℝ) : Prop := x - y + 2 = 0
def symmetric_line (x y : ℝ) : Prop := x - 2*y + 3 = 0

-- Define symmetry with respect to a line
def symmetric_wrt (l1 l2 l_ref : (ℝ → ℝ → Prop)) : Prop :=
  ∀ (x y : ℝ), l1 x y → ∃ (x' y' : ℝ), l2 x' y' ∧
    (x + x') / 2 = (y + y') / 2 + 2 ∧ -- Point on reference line
    (y' - y) = (x' - x) -- Perpendicular to reference line

-- Theorem statement
theorem line_symmetry :
  symmetric_wrt original_line symmetric_line reference_line :=
sorry

end NUMINAMATH_CALUDE_line_symmetry_l2860_286018


namespace NUMINAMATH_CALUDE_regression_equation_proof_l2860_286076

theorem regression_equation_proof (x y z : ℝ) (b a : ℝ) :
  (y = Real.exp (b * x + a)) →
  (z = Real.log y) →
  (z = 0.25 * x - 2.58) →
  (y = Real.exp (0.25 * x - 2.58)) := by
  sorry

end NUMINAMATH_CALUDE_regression_equation_proof_l2860_286076


namespace NUMINAMATH_CALUDE_dihedral_angle_relation_l2860_286005

/-- Regular quadrilateral prism -/
structure RegularQuadPrism where
  -- We don't need to define the specific geometry, just the existence of the prism
  prism : Unit

/-- Dihedral angle between lateral face and base -/
def lateral_base_angle (p : RegularQuadPrism) : ℝ :=
  sorry

/-- Dihedral angle between adjacent lateral faces -/
def adjacent_lateral_angle (p : RegularQuadPrism) : ℝ :=
  sorry

/-- Theorem stating the relationship between dihedral angles in a regular quadrilateral prism -/
theorem dihedral_angle_relation (p : RegularQuadPrism) :
  Real.cos (adjacent_lateral_angle p) = -(Real.cos (lateral_base_angle p))^2 := by
  sorry

end NUMINAMATH_CALUDE_dihedral_angle_relation_l2860_286005


namespace NUMINAMATH_CALUDE_apple_count_difference_l2860_286077

theorem apple_count_difference (initial_green : ℕ) (red_green_difference : ℕ) (delivered_green : ℕ) : 
  initial_green = 546 →
  red_green_difference = 1850 →
  delivered_green = 2725 →
  (initial_green + delivered_green) - (initial_green + red_green_difference) = 875 :=
by
  sorry

end NUMINAMATH_CALUDE_apple_count_difference_l2860_286077


namespace NUMINAMATH_CALUDE_stratified_sampling_proportion_l2860_286034

/-- Represents the number of athletes selected in a stratified sampling -/
structure StratifiedSample where
  male : ℕ
  female : ℕ

/-- Represents the composition of the track and field team -/
def team : StratifiedSample :=
  { male := 56, female := 42 }

/-- Calculates the ratio of male to female athletes -/
def ratio (s : StratifiedSample) : ℚ :=
  s.male / s.female

/-- Theorem: In a stratified sampling, if 8 male athletes are selected,
    then 6 female athletes should be selected to maintain the same proportion -/
theorem stratified_sampling_proportion :
  ∀ (sample : StratifiedSample),
    sample.male = 8 →
    ratio sample = ratio team →
    sample.female = 6 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_proportion_l2860_286034


namespace NUMINAMATH_CALUDE_nights_with_new_habit_l2860_286057

/-- Represents the number of nights a candle lasts when burned for 1 hour per night -/
def initial_nights_per_candle : ℕ := 8

/-- Represents the number of hours Carmen burns a candle each night after changing her habit -/
def hours_per_night : ℕ := 2

/-- Represents the total number of candles Carmen uses -/
def total_candles : ℕ := 6

/-- Theorem stating the total number of nights Carmen can burn candles with the new habit -/
theorem nights_with_new_habit : 
  (total_candles * initial_nights_per_candle) / hours_per_night = 24 := by
  sorry

end NUMINAMATH_CALUDE_nights_with_new_habit_l2860_286057


namespace NUMINAMATH_CALUDE_midpoint_calculation_l2860_286088

/-- Given two points A and B in a 2D plane, prove that 3x - 5y = -13.5,
    where (x, y) is the midpoint of segment AB. -/
theorem midpoint_calculation (A B : ℝ × ℝ) (h : A = (20, 12) ∧ B = (-4, 3)) :
  let C : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  3 * C.1 - 5 * C.2 = -13.5 := by
sorry

end NUMINAMATH_CALUDE_midpoint_calculation_l2860_286088


namespace NUMINAMATH_CALUDE_rectangle_ratio_is_two_l2860_286029

/-- Configuration of squares and rectangles -/
structure SquareRectConfig where
  inner_square_side : ℝ
  rect_short_side : ℝ
  rect_long_side : ℝ

/-- The configuration satisfies the problem conditions -/
def valid_config (c : SquareRectConfig) : Prop :=
  c.inner_square_side > 0 ∧
  c.rect_short_side > 0 ∧
  c.rect_long_side > 0 ∧
  c.inner_square_side + 2 * c.rect_short_side = 3 * c.inner_square_side ∧
  c.inner_square_side + c.rect_long_side = 3 * c.inner_square_side

theorem rectangle_ratio_is_two (c : SquareRectConfig) (h : valid_config c) :
  c.rect_long_side / c.rect_short_side = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_is_two_l2860_286029


namespace NUMINAMATH_CALUDE_interesting_iff_prime_power_l2860_286022

def is_interesting (n : ℕ) : Prop :=
  n > 1 ∧ ∀ x y : ℕ, (Nat.gcd x n ≠ 1 ∧ Nat.gcd y n ≠ 1) → Nat.gcd (x + y) n ≠ 1

theorem interesting_iff_prime_power (n : ℕ) :
  is_interesting n ↔ ∃ p k : ℕ, Nat.Prime p ∧ k > 0 ∧ n = p^k :=
sorry

end NUMINAMATH_CALUDE_interesting_iff_prime_power_l2860_286022


namespace NUMINAMATH_CALUDE_pure_imaginary_value_l2860_286072

theorem pure_imaginary_value (a : ℝ) : 
  (∀ z : ℂ, z = (a^2 - 3*a + 2 : ℝ) + (a - 2 : ℝ) * I → z.re = 0 ∧ z.im ≠ 0) → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_value_l2860_286072


namespace NUMINAMATH_CALUDE_cube_root_simplification_l2860_286050

theorem cube_root_simplification :
  ∀ (x : ℝ), x > 0 → (x^(1/3) : ℝ) = (154^(1/3) / 9^(1/3) : ℝ) ↔ x = 17 + 1/9 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l2860_286050


namespace NUMINAMATH_CALUDE_equal_roots_coefficients_l2860_286041

def polynomial (x p q : ℝ) : ℝ := x^4 - 10*x^3 + 37*x^2 + p*x + q

theorem equal_roots_coefficients :
  ∀ (p q : ℝ),
  (∃ (x₁ x₃ : ℝ), 
    (∀ x : ℝ, polynomial x p q = 0 ↔ x = x₁ ∨ x = x₃) ∧
    (x₁ + x₃ = 5) ∧
    (x₁ * x₃ = 6)) →
  p = -60 ∧ q = 36 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_coefficients_l2860_286041


namespace NUMINAMATH_CALUDE_square_perimeter_l2860_286085

theorem square_perimeter (s : ℝ) (h1 : s > 0) : 
  (∃ (r : ℝ × ℝ), r.1 = s/2 ∧ r.2 = s ∧ 2*(r.1 + r.2) = 24) → 4*s = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l2860_286085


namespace NUMINAMATH_CALUDE_fans_with_all_items_l2860_286094

def stadium_capacity : ℕ := 5000
def hotdog_interval : ℕ := 75
def soda_interval : ℕ := 45
def popcorn_interval : ℕ := 50
def max_all_items : ℕ := 100

theorem fans_with_all_items :
  let lcm := Nat.lcm (Nat.lcm hotdog_interval soda_interval) popcorn_interval
  min (stadium_capacity / lcm) max_all_items = 11 := by
  sorry

end NUMINAMATH_CALUDE_fans_with_all_items_l2860_286094


namespace NUMINAMATH_CALUDE_total_gold_stars_l2860_286083

def shelby_stars : List Nat := [4, 6, 3, 5, 2, 3, 7]
def alex_stars : List Nat := [5, 3, 6, 4, 7, 2, 5]

theorem total_gold_stars :
  (shelby_stars.sum + alex_stars.sum) = 62 := by
  sorry

end NUMINAMATH_CALUDE_total_gold_stars_l2860_286083


namespace NUMINAMATH_CALUDE_area_of_special_triangle_l2860_286032

/-- A scalene triangle with given properties -/
structure ScaleneTriangle where
  -- A, B, C are the angles of the triangle
  A : ℝ
  B : ℝ
  C : ℝ
  -- rA, rB, rC are the exradii
  rA : ℝ
  rB : ℝ
  rC : ℝ
  -- Conditions
  angle_sum : A + B + C = π
  exradii_condition : 20 * (rB^2 * rC^2 + rC^2 * rA^2 + rA^2 * rB^2) = 19 * (rA * rB * rC)^2
  tan_sum : Real.tan (A/2) + Real.tan (B/2) + Real.tan (C/2) = 2.019
  inradius : ℝ := 1

/-- The area of a scalene triangle with the given properties is 2019/25 -/
theorem area_of_special_triangle (t : ScaleneTriangle) : 
  (2 * t.inradius * (Real.tan (t.A/2) + Real.tan (t.B/2) + Real.tan (t.C/2))) = 2019/25 := by
  sorry

end NUMINAMATH_CALUDE_area_of_special_triangle_l2860_286032


namespace NUMINAMATH_CALUDE_fraction_sum_difference_l2860_286073

theorem fraction_sum_difference : 7/6 + 5/4 - 3/2 = 11/12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_difference_l2860_286073


namespace NUMINAMATH_CALUDE_bank_savings_exceed_50_dollars_l2860_286025

/-- The sum of a geometric sequence with first term 5 and ratio 2, after n terms -/
def geometric_sum (n : ℕ) : ℚ := 5 * (2^n - 1)

/-- The smallest number of days needed for the sum to exceed 5000 cents -/
def smallest_day : ℕ := 10

theorem bank_savings_exceed_50_dollars :
  (∀ k < smallest_day, geometric_sum k ≤ 5000) ∧
  geometric_sum smallest_day > 5000 := by sorry

end NUMINAMATH_CALUDE_bank_savings_exceed_50_dollars_l2860_286025


namespace NUMINAMATH_CALUDE_clothing_distribution_l2860_286096

theorem clothing_distribution (total : ℕ) (first_load : ℕ) (num_small_loads : ℕ) 
  (h1 : total = 59)
  (h2 : first_load = 32)
  (h3 : num_small_loads = 9)
  : (total - first_load) / num_small_loads = 3 := by
  sorry

end NUMINAMATH_CALUDE_clothing_distribution_l2860_286096


namespace NUMINAMATH_CALUDE_intersection_empty_intersection_equals_A_l2860_286049

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x : ℝ | x^2 - 4*x - 12 > 0}

-- Theorem 1: A ∩ B = ∅ iff -2 ≤ a ≤ 3
theorem intersection_empty (a : ℝ) : A a ∩ B = ∅ ↔ -2 ≤ a ∧ a ≤ 3 := by
  sorry

-- Theorem 2: A ∩ B = A iff a < -5 or a > 6
theorem intersection_equals_A (a : ℝ) : A a ∩ B = A a ↔ a < -5 ∨ a > 6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_intersection_equals_A_l2860_286049


namespace NUMINAMATH_CALUDE_percent_relation_l2860_286051

theorem percent_relation (x y : ℝ) (h : 0.2 * (x - y) = 0.15 * (x + y)) : y = x / 7 := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l2860_286051


namespace NUMINAMATH_CALUDE_total_legs_l2860_286030

/-- The number of bees -/
def num_bees : ℕ := 50

/-- The number of ants -/
def num_ants : ℕ := 35

/-- The number of spiders -/
def num_spiders : ℕ := 20

/-- The number of legs a bee has -/
def bee_legs : ℕ := 6

/-- The number of legs an ant has -/
def ant_legs : ℕ := 6

/-- The number of legs a spider has -/
def spider_legs : ℕ := 8

/-- Theorem stating the total number of legs -/
theorem total_legs : 
  num_bees * bee_legs + num_ants * ant_legs + num_spiders * spider_legs = 670 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_l2860_286030


namespace NUMINAMATH_CALUDE_sum_equals_3004_5_l2860_286065

/-- Define the recursive function for the sum -/
def S (n : ℕ) : ℚ :=
  if n = 0 then 2
  else if n = 1 then 3 + (1/3) * 2
  else (2003 - n + 1 : ℚ) + (1/3) * S (n-1)

/-- The main theorem stating that S(2001) equals 3004.5 -/
theorem sum_equals_3004_5 : S 2001 = 3004.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_3004_5_l2860_286065


namespace NUMINAMATH_CALUDE_unique_function_satisfying_equation_l2860_286086

-- Define the property that a function must satisfy
def SatisfiesEquation (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ x y : ℝ, f (x + f (k * y)) = x + y

-- State the theorem
theorem unique_function_satisfying_equation (k : ℝ) (hk : k ≠ 0) :
  ∃! f : ℝ → ℝ, SatisfiesEquation f k ∧ f = id := by sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_equation_l2860_286086


namespace NUMINAMATH_CALUDE_largest_divisors_ratio_l2860_286066

theorem largest_divisors_ratio (N : ℕ) (h1 : N > 1) 
  (h2 : ∃ (a : ℕ), a ∣ N ∧ 6 * a ∣ N ∧ a ≠ 1 ∧ 6 * a ≠ N) :
  (N / 2) / (N / 3) = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_largest_divisors_ratio_l2860_286066


namespace NUMINAMATH_CALUDE_community_service_arrangements_l2860_286053

def number_of_arrangements (n m k : ℕ) (a b : Fin n) : ℕ :=
  let without_ab := Nat.choose m k
  let with_one := 2 * Nat.choose (m - 1) (k - 1)
  without_ab + 2 * with_one

theorem community_service_arrangements :
  number_of_arrangements 8 6 3 0 1 = 80 := by
  sorry

end NUMINAMATH_CALUDE_community_service_arrangements_l2860_286053


namespace NUMINAMATH_CALUDE_van_capacity_l2860_286008

theorem van_capacity (students : ℕ) (adults : ℕ) (vans : ℕ) :
  students = 33 →
  adults = 9 →
  vans = 6 →
  (students + adults) / vans = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_van_capacity_l2860_286008


namespace NUMINAMATH_CALUDE_hyperbola_properties_l2860_286055

/-- Definition of the hyperbola C -/
def C (x y : ℝ) : Prop := y = Real.sqrt 3 * (1 / (2 * x) + x / 3)

/-- C is a hyperbola -/
axiom C_is_hyperbola : ∃ (a b : ℝ), ∀ (x y : ℝ), C x y ↔ (x^2 / a^2) - (y^2 / b^2) = 1

/-- Statement about the asymptote, focus, and intersection properties of C -/
theorem hyperbola_properties :
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x| ∧ |x| < δ → ∃ y, C x y ∧ |y| > 1/ε) ∧ 
  C 1 (Real.sqrt 3) ∧
  (∀ t : ℝ, ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ C x₁ y₁ ∧ C x₂ y₂ ∧ y₁ = x₁ + t ∧ y₂ = x₂ + t) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l2860_286055


namespace NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_eight_satisfies_inequality_exists_no_greater_value_l2860_286015

theorem greatest_value_quadratic_inequality :
  ∀ x : ℝ, x^2 - 12*x + 32 ≤ 0 → x ≤ 8 :=
by
  sorry

theorem eight_satisfies_inequality :
  8^2 - 12*8 + 32 = 0 :=
by
  sorry

theorem exists_no_greater_value :
  ¬∃ y : ℝ, y > 8 ∧ y^2 - 12*y + 32 ≤ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_eight_satisfies_inequality_exists_no_greater_value_l2860_286015


namespace NUMINAMATH_CALUDE_notebooks_ordered_l2860_286054

theorem notebooks_ordered (initial final lost : ℕ) (h1 : initial = 4) (h2 : lost = 2) (h3 : final = 8) :
  ∃ ordered : ℕ, initial + ordered - lost = final ∧ ordered = 6 := by
  sorry

end NUMINAMATH_CALUDE_notebooks_ordered_l2860_286054


namespace NUMINAMATH_CALUDE_prime_condition_characterization_l2860_286048

/-- The set of polynomials with coefficients from {0,1,...,p-1} and degree less than p -/
def K_p (p : ℕ) : Set (Polynomial ℤ) :=
  {f | ∀ i, (f.coeff i < p ∧ f.coeff i ≥ 0) ∧ f.degree < p}

/-- The condition that for all pairs of polynomials P,Q in K_p, 
    if P(Q(n)) ≡ n (mod p) for all integers n, then deg(P) = deg(Q) -/
def condition (p : ℕ) : Prop :=
  ∀ P Q : Polynomial ℤ, P ∈ K_p p → Q ∈ K_p p →
    (∀ n : ℤ, (P.comp Q).eval n ≡ n [ZMOD p]) →
    P.degree = Q.degree

theorem prime_condition_characterization :
  ∀ p : ℕ, p.Prime → (condition p ↔ p ∈ ({2, 3, 5, 7} : Set ℕ)) := by
  sorry

#check prime_condition_characterization

end NUMINAMATH_CALUDE_prime_condition_characterization_l2860_286048


namespace NUMINAMATH_CALUDE_smallest_land_fraction_150_members_l2860_286011

/-- Represents a noble family with land division rules -/
structure NobleFamily :=
  (total_members : ℕ)
  (founder_land : ℝ)
  (divide_land : ℝ → ℕ → ℝ)
  (transfer_to_state : ℝ → ℝ)

/-- The smallest possible fraction of land a family member could receive -/
def smallest_land_fraction (family : NobleFamily) : ℚ :=
  1 / (2 * 3^49)

/-- Theorem stating the smallest possible fraction of land for a family of 150 members -/
theorem smallest_land_fraction_150_members 
  (family : NobleFamily) 
  (h_members : family.total_members = 150) :
  smallest_land_fraction family = 1 / (2 * 3^49) :=
sorry

end NUMINAMATH_CALUDE_smallest_land_fraction_150_members_l2860_286011


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l2860_286036

theorem rectangle_diagonal (a b d : ℝ) : 
  a = 6 → a * b = 48 → d^2 = a^2 + b^2 → d = 10 := by sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l2860_286036


namespace NUMINAMATH_CALUDE_smallest_product_l2860_286007

def S : Set Int := {-10, -3, 0, 2, 6}

theorem smallest_product (a b : Int) (ha : a ∈ S) (hb : b ∈ S) :
  ∃ (x y : Int), x ∈ S ∧ y ∈ S ∧ x * y ≤ a * b ∧ x * y = -60 :=
sorry

end NUMINAMATH_CALUDE_smallest_product_l2860_286007


namespace NUMINAMATH_CALUDE_discriminant_positive_increasing_when_m_le_8_l2860_286047

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - (m - 2) * x - m

-- Theorem 1: The discriminant is always positive
theorem discriminant_positive (m : ℝ) : m^2 + 4 > 0 := by sorry

-- Theorem 2: The function is increasing for x ≥ 3 when m ≤ 8
theorem increasing_when_m_le_8 (m : ℝ) (h : m ≤ 8) :
  ∀ x ≥ 3, ∀ y > x, f m y > f m x := by sorry

end NUMINAMATH_CALUDE_discriminant_positive_increasing_when_m_le_8_l2860_286047


namespace NUMINAMATH_CALUDE_paul_chickens_left_l2860_286075

/-- The number of chickens Paul has left after selling some -/
def chickens_left (initial : ℕ) (sold_neighbor : ℕ) (sold_gate : ℕ) : ℕ :=
  initial - sold_neighbor - sold_gate

/-- Theorem stating that Paul is left with 43 chickens -/
theorem paul_chickens_left : chickens_left 80 12 25 = 43 := by
  sorry

end NUMINAMATH_CALUDE_paul_chickens_left_l2860_286075


namespace NUMINAMATH_CALUDE_function_characterization_l2860_286082

/-- The set of positive rational numbers -/
def PositiveRationals : Set ℚ := {x : ℚ | 0 < x}

/-- The condition on x, y, z -/
def Condition (x y z : ℚ) : Prop := (x + y + z + 1 = 4 * x * y * z) ∧ (x ∈ PositiveRationals) ∧ (y ∈ PositiveRationals) ∧ (z ∈ PositiveRationals)

/-- The property that f must satisfy -/
def SatisfiesProperty (f : ℚ → ℝ) : Prop :=
  ∀ x y z, Condition x y z → f x + f y + f z = 1

/-- The theorem statement -/
theorem function_characterization :
  ∀ f : ℚ → ℝ, (∀ x ∈ PositiveRationals, f x = f x) →
    SatisfiesProperty f →
    ∃ a : ℝ, ∀ x ∈ PositiveRationals, f x = a * (1 / (2 * x + 1)) + (1 - a) * (1 / 3) :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l2860_286082


namespace NUMINAMATH_CALUDE_rental_fee_calculation_l2860_286042

/-- The rental fee for a truck, given the total cost, per-mile charge, and miles driven. -/
def rental_fee (total_cost per_mile_charge miles_driven : ℚ) : ℚ :=
  total_cost - per_mile_charge * miles_driven

/-- Theorem stating that the rental fee is $20.99 under the given conditions. -/
theorem rental_fee_calculation :
  rental_fee 95.74 0.25 299 = 20.99 := by
  sorry

end NUMINAMATH_CALUDE_rental_fee_calculation_l2860_286042


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l2860_286006

theorem geometric_sequence_formula (a : ℕ → ℝ) (q : ℝ) (h1 : q < 1) 
  (h2 : a 2 + a 4 = 5/8) (h3 : a 3 = 1/4) 
  (h4 : ∀ n : ℕ, a (n + 1) = q * a n) : 
  ∀ n : ℕ, a n = (1/2)^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l2860_286006


namespace NUMINAMATH_CALUDE_not_penetrating_function_l2860_286024

/-- Definition of a penetrating function -/
def isPenetratingFunction (f : ℝ → ℝ) : Prop :=
  ∀ a : ℝ, a > 0 ∧ a ≠ 1 → ∀ x : ℝ, f (a * x) = a * f x

/-- The function f(x) = x + 1 -/
def f (x : ℝ) : ℝ := x + 1

/-- Theorem: f(x) = x + 1 is not a penetrating function -/
theorem not_penetrating_function : ¬ isPenetratingFunction f := by
  sorry

end NUMINAMATH_CALUDE_not_penetrating_function_l2860_286024


namespace NUMINAMATH_CALUDE_inverse_composition_problem_l2860_286091

/-- Given functions h and k where k⁻¹ ∘ h = λ z, 7 * z - 4, prove that h⁻¹(k(12)) = 16/7 -/
theorem inverse_composition_problem (h k : ℝ → ℝ) 
  (hk : Function.LeftInverse k⁻¹ h ∧ Function.RightInverse k⁻¹ h) 
  (h_def : ∀ z, k⁻¹ (h z) = 7 * z - 4) : 
  h⁻¹ (k 12) = 16/7 := by
  sorry

end NUMINAMATH_CALUDE_inverse_composition_problem_l2860_286091


namespace NUMINAMATH_CALUDE_quadrilateral_area_l2860_286093

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the angle between three points -/
def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Calculates the area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- Theorem: Area of quadrilateral ABCD is 62.5√3 -/
theorem quadrilateral_area (ABCD : Quadrilateral) :
  angle ABCD.A ABCD.B ABCD.C = π / 2 →
  angle ABCD.A ABCD.C ABCD.D = π / 3 →
  distance ABCD.A ABCD.C = 25 →
  distance ABCD.C ABCD.D = 10 →
  ∃ E : Point, distance ABCD.A E = 15 →
  area ABCD = 62.5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l2860_286093
