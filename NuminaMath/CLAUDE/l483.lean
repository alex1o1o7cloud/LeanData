import Mathlib

namespace NUMINAMATH_CALUDE_number_puzzle_l483_48348

theorem number_puzzle : ∃ x : ℝ, x + (1/5) * x + 1 = 10 ∧ x = 7.5 := by sorry

end NUMINAMATH_CALUDE_number_puzzle_l483_48348


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l483_48307

theorem quadratic_inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, x^2 + m*x + 1 > 0) → -2 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l483_48307


namespace NUMINAMATH_CALUDE_fraction_less_than_two_l483_48357

theorem fraction_less_than_two (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_less_than_two_l483_48357


namespace NUMINAMATH_CALUDE_sum_odd_numbers_100_to_200_l483_48306

def sum_odd_numbers_between (a b : ℕ) : ℕ :=
  let first_odd := if a % 2 = 0 then a + 1 else a
  let last_odd := if b % 2 = 0 then b - 1 else b
  let n := (last_odd - first_odd) / 2 + 1
  n * (first_odd + last_odd) / 2

theorem sum_odd_numbers_100_to_200 :
  sum_odd_numbers_between 100 200 = 7500 := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_numbers_100_to_200_l483_48306


namespace NUMINAMATH_CALUDE_john_shorter_than_rebeca_l483_48317

def height_difference (john_height lena_height rebeca_height : ℕ) : Prop :=
  (john_height = lena_height + 15) ∧
  (john_height < rebeca_height) ∧
  (john_height = 152) ∧
  (lena_height + rebeca_height = 295)

theorem john_shorter_than_rebeca (john_height lena_height rebeca_height : ℕ) :
  height_difference john_height lena_height rebeca_height →
  rebeca_height - john_height = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_john_shorter_than_rebeca_l483_48317


namespace NUMINAMATH_CALUDE_max_value_of_p_l483_48322

theorem max_value_of_p (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x * y * z + x + z = y) :
  ∃ (p : ℝ), p = 2 / (x^2 + 1) - 2 / (y^2 + 1) + 3 / (z^2 + 1) ∧
  p ≤ 10 / 3 ∧
  ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧
    x' * y' * z' + x' + z' = y' ∧
    2 / (x'^2 + 1) - 2 / (y'^2 + 1) + 3 / (z'^2 + 1) = 10 / 3 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_p_l483_48322


namespace NUMINAMATH_CALUDE_base7_to_base10_conversion_l483_48312

-- Define the base-7 number as a list of digits
def base7Number : List Nat := [4, 5, 3, 6]

-- Define the function to convert from base 7 to base 10
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

-- Theorem statement
theorem base7_to_base10_conversion :
  base7ToBase10 base7Number = 1644 := by
  sorry

end NUMINAMATH_CALUDE_base7_to_base10_conversion_l483_48312


namespace NUMINAMATH_CALUDE_cos_150_degrees_l483_48347

theorem cos_150_degrees : Real.cos (150 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l483_48347


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_11_l483_48327

theorem binomial_coefficient_20_11 :
  (Nat.choose 18 9 = 48620) →
  (Nat.choose 18 8 = 43758) →
  (Nat.choose 20 11 = 168168) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_11_l483_48327


namespace NUMINAMATH_CALUDE_rope_cutting_impossibility_l483_48304

theorem rope_cutting_impossibility : ¬ ∃ (n : ℕ), 5 + 4 * n = 2019 := by
  sorry

end NUMINAMATH_CALUDE_rope_cutting_impossibility_l483_48304


namespace NUMINAMATH_CALUDE_no_hamiltonian_cycle_in_circ_2016_2_3_l483_48345

/-- A circulant digraph with n vertices and jump sizes a and b -/
structure CirculantDigraph (n : ℕ) (a b : ℕ) where
  vertices : Fin n

/-- Condition for the existence of a Hamiltonian cycle in a circulant digraph -/
def has_hamiltonian_cycle (G : CirculantDigraph n a b) : Prop :=
  ∃ (s t : ℕ), s + t = Nat.gcd n (a - b) ∧ Nat.gcd n (s * a + t * b) = 1

/-- The main theorem about the non-existence of a Hamiltonian cycle in Circ(2016; 2, 3) -/
theorem no_hamiltonian_cycle_in_circ_2016_2_3 :
  ¬ ∃ (G : CirculantDigraph 2016 2 3), has_hamiltonian_cycle G :=
by sorry

end NUMINAMATH_CALUDE_no_hamiltonian_cycle_in_circ_2016_2_3_l483_48345


namespace NUMINAMATH_CALUDE_cylinder_volume_change_l483_48394

theorem cylinder_volume_change (r h : ℝ) (h1 : r > 0) (h2 : h > 0) : 
  let original_volume := π * r^2 * h
  let new_volume := π * (3*r)^2 * (2*h)
  original_volume = 30 → new_volume = 540 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_change_l483_48394


namespace NUMINAMATH_CALUDE_prime_squared_with_totient_42_l483_48384

theorem prime_squared_with_totient_42 (p : ℕ) (N : ℕ) : 
  Prime p → N = p^2 → Nat.totient N = 42 → N = 49 := by
  sorry

end NUMINAMATH_CALUDE_prime_squared_with_totient_42_l483_48384


namespace NUMINAMATH_CALUDE_root_triple_relation_l483_48349

theorem root_triple_relation (a b c : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = 3 * x) →
  3 * b^2 = 16 * a * c := by
  sorry

end NUMINAMATH_CALUDE_root_triple_relation_l483_48349


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l483_48369

/-- Calculates the total wet surface area of a rectangular cistern -/
def totalWetSurfaceArea (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth + width * depth)

/-- Theorem: The total wet surface area of a cistern with given dimensions -/
theorem cistern_wet_surface_area :
  let length : ℝ := 9
  let width : ℝ := 4
  let depth : ℝ := 1.25
  totalWetSurfaceArea length width depth = 68.5 := by
  sorry

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l483_48369


namespace NUMINAMATH_CALUDE_missing_number_proof_l483_48363

/-- Given a list of 10 numbers with an average of 750, where 9 of the numbers are known,
    prove that the remaining number is 1747. -/
theorem missing_number_proof (numbers : List ℕ) (h1 : numbers.length = 10)
  (h2 : numbers.sum / numbers.length = 750)
  (h3 : numbers.count 744 = 1)
  (h4 : numbers.count 745 = 1)
  (h5 : numbers.count 748 = 1)
  (h6 : numbers.count 749 = 1)
  (h7 : numbers.count 752 = 2)
  (h8 : numbers.count 753 = 1)
  (h9 : numbers.count 755 = 2)
  : numbers.any (· = 1747) := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l483_48363


namespace NUMINAMATH_CALUDE_positive_real_array_inequalities_l483_48333

theorem positive_real_array_inequalities
  (x₁ x₂ x₃ x₄ x₅ : ℝ)
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧ x₅ > 0)
  (h1 : (x₁^2 - x₃*x₅)*(x₂^2 - x₃*x₅) ≤ 0)
  (h2 : (x₂^2 - x₄*x₁)*(x₃^2 - x₄*x₁) ≤ 0)
  (h3 : (x₃^2 - x₅*x₂)*(x₄^2 - x₅*x₂) ≤ 0)
  (h4 : (x₄^2 - x₁*x₃)*(x₅^2 - x₁*x₃) ≤ 0)
  (h5 : (x₅^2 - x₂*x₄)*(x₁^2 - x₂*x₄) ≤ 0) :
  x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄ ∧ x₄ = x₅ := by
sorry

end NUMINAMATH_CALUDE_positive_real_array_inequalities_l483_48333


namespace NUMINAMATH_CALUDE_gas_cost_problem_l483_48325

theorem gas_cost_problem (x : ℝ) : 
  (x / 4 - x / 7 = 15) → x = 140 := by
  sorry

end NUMINAMATH_CALUDE_gas_cost_problem_l483_48325


namespace NUMINAMATH_CALUDE_modulus_of_12_plus_5i_l483_48358

theorem modulus_of_12_plus_5i : Complex.abs (12 + 5 * Complex.I) = 13 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_12_plus_5i_l483_48358


namespace NUMINAMATH_CALUDE_three_digit_sum_relation_l483_48308

/-- Given a three-digit number with tens digit zero, prove the relationship between m and n -/
theorem three_digit_sum_relation (x y m n : ℕ) : 
  (100 * y + x = m * (x + y)) →   -- Original number is m times sum of digits
  (100 * x + y = n * (x + y)) →   -- Swapped number is n times sum of digits
  n = 101 - m := by
  sorry

end NUMINAMATH_CALUDE_three_digit_sum_relation_l483_48308


namespace NUMINAMATH_CALUDE_finite_triples_sum_reciprocals_l483_48320

theorem finite_triples_sum_reciprocals :
  ∃ (S : Finset (ℕ × ℕ × ℕ)), ∀ a b c : ℕ,
    a > 0 ∧ b > 0 ∧ c > 0 →
    (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c = (1 : ℚ) / 1000 →
    (a, b, c) ∈ S :=
by sorry

end NUMINAMATH_CALUDE_finite_triples_sum_reciprocals_l483_48320


namespace NUMINAMATH_CALUDE_student_presentation_time_l483_48366

theorem student_presentation_time 
  (num_students : ℕ) 
  (period_length : ℕ) 
  (num_periods : ℕ) 
  (h1 : num_students = 32) 
  (h2 : period_length = 40) 
  (h3 : num_periods = 4) : 
  (num_periods * period_length) / num_students = 5 := by
  sorry

end NUMINAMATH_CALUDE_student_presentation_time_l483_48366


namespace NUMINAMATH_CALUDE_binary_10101_equals_21_l483_48323

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_10101_equals_21 :
  binary_to_decimal [true, false, true, false, true] = 21 := by
  sorry

end NUMINAMATH_CALUDE_binary_10101_equals_21_l483_48323


namespace NUMINAMATH_CALUDE_circle_theorem_l483_48379

/-- Circle C -/
def circle_C (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

/-- Circle D -/
def circle_D (x y : ℝ) : Prop :=
  (x + 3)^2 + (y + 1)^2 = 16

/-- Line l -/
def line_l (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

/-- Circles C and D are externally tangent -/
def externally_tangent (m : ℝ) : Prop :=
  ∃ x y : ℝ, circle_C x y m ∧ circle_D x y

theorem circle_theorem :
  ∀ m : ℝ,
  (∀ x y : ℝ, circle_C x y m → m < 5) ∧
  (externally_tangent m → m = 4) ∧
  (m = 4 →
    ∃ chord_length : ℝ,
      chord_length = 4 * Real.sqrt 5 / 5 ∧
      ∀ x y : ℝ,
        circle_C x y m ∧ line_l x y →
        ∃ x' y' : ℝ,
          circle_C x' y' m ∧ line_l x' y' ∧
          (x - x')^2 + (y - y')^2 = chord_length^2) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_theorem_l483_48379


namespace NUMINAMATH_CALUDE_average_age_of_boys_l483_48368

def boys_ages (x : ℝ) : Fin 3 → ℝ
| 0 => 3 * x
| 1 => 5 * x
| 2 => 7 * x

theorem average_age_of_boys (x : ℝ) (h1 : boys_ages x 2 = 21) :
  (boys_ages x 0 + boys_ages x 1 + boys_ages x 2) / 3 = 15 := by
  sorry

#check average_age_of_boys

end NUMINAMATH_CALUDE_average_age_of_boys_l483_48368


namespace NUMINAMATH_CALUDE_function_bound_l483_48313

def ContinuousFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 → |f x - f y| < |x - y|

theorem function_bound (f : ℝ → ℝ) (h1 : ContinuousFunction f) (h2 : f 0 = f 1) :
  ∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 → |f x - f y| < (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_bound_l483_48313


namespace NUMINAMATH_CALUDE_ethan_hourly_wage_l483_48382

/-- Represents Ethan's work schedule and earnings --/
structure WorkSchedule where
  hours_per_day : ℕ
  days_per_week : ℕ
  weeks_worked : ℕ
  total_earnings : ℕ

/-- Calculates the hourly wage given a work schedule --/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  schedule.total_earnings / (schedule.hours_per_day * schedule.days_per_week * schedule.weeks_worked)

/-- Theorem stating that Ethan's hourly wage is $18 --/
theorem ethan_hourly_wage :
  let ethan_schedule : WorkSchedule := {
    hours_per_day := 8,
    days_per_week := 5,
    weeks_worked := 5,
    total_earnings := 3600
  }
  hourly_wage ethan_schedule = 18 := by
  sorry

end NUMINAMATH_CALUDE_ethan_hourly_wage_l483_48382


namespace NUMINAMATH_CALUDE_initial_red_orchids_l483_48377

/-- Represents the number of orchids in a vase -/
structure OrchidVase where
  initialRed : ℕ
  initialWhite : ℕ
  addedRed : ℕ
  finalRed : ℕ

/-- Theorem stating the initial number of red orchids in the vase -/
theorem initial_red_orchids (vase : OrchidVase)
  (h1 : vase.initialWhite = 3)
  (h2 : vase.addedRed = 6)
  (h3 : vase.finalRed = 15)
  : vase.initialRed = 9 := by
  sorry

end NUMINAMATH_CALUDE_initial_red_orchids_l483_48377


namespace NUMINAMATH_CALUDE_some_mythical_beings_are_mystical_spirits_l483_48340

-- Define our universe
variable (U : Type)

-- Define our predicates
variable (Dragon : U → Prop)
variable (MythicalBeing : U → Prop)
variable (MysticalSpirit : U → Prop)

-- State the theorem
theorem some_mythical_beings_are_mystical_spirits
  (h1 : ∀ x, Dragon x → MythicalBeing x)
  (h2 : ∃ x, MysticalSpirit x ∧ Dragon x) :
  ∃ x, MythicalBeing x ∧ MysticalSpirit x :=
by sorry

end NUMINAMATH_CALUDE_some_mythical_beings_are_mystical_spirits_l483_48340


namespace NUMINAMATH_CALUDE_percentage_boys_playing_soccer_is_86_percent_l483_48372

/-- Calculates the percentage of boys among students playing soccer -/
def percentage_boys_playing_soccer (total_students : ℕ) (num_boys : ℕ) (students_playing_soccer : ℕ) (girls_not_playing_soccer : ℕ) : ℚ :=
  let total_girls : ℕ := total_students - num_boys
  let girls_playing_soccer : ℕ := total_girls - girls_not_playing_soccer
  let boys_playing_soccer : ℕ := students_playing_soccer - girls_playing_soccer
  (boys_playing_soccer : ℚ) / (students_playing_soccer : ℚ) * 100

/-- Theorem stating that the percentage of boys playing soccer is 86% -/
theorem percentage_boys_playing_soccer_is_86_percent :
  percentage_boys_playing_soccer 420 312 250 73 = 86 := by
  sorry

end NUMINAMATH_CALUDE_percentage_boys_playing_soccer_is_86_percent_l483_48372


namespace NUMINAMATH_CALUDE_ambiguous_date_characterization_max_consecutive_ambiguous_proof_l483_48351

/-- Represents a date with day and month -/
structure Date where
  day : Nat
  month : Nat
  h1 : day ≥ 1 ∧ day ≤ 31
  h2 : month ≥ 1 ∧ month ≤ 12

/-- Defines when a date is ambiguous -/
def is_ambiguous (d : Date) : Prop :=
  d.day ≥ 1 ∧ d.day ≤ 12 ∧ d.day ≠ d.month

/-- The maximum number of consecutive ambiguous dates in any month -/
def max_consecutive_ambiguous : Nat := 11

theorem ambiguous_date_characterization (d : Date) :
  is_ambiguous d ↔ d.day ≥ 1 ∧ d.day ≤ 12 ∧ d.day ≠ d.month :=
sorry

theorem max_consecutive_ambiguous_proof :
  ∀ m : Nat, m ≥ 1 → m ≤ 12 →
    (∃ consecutive : List Date,
      consecutive.length = max_consecutive_ambiguous ∧
      (∀ d ∈ consecutive, d.month = m ∧ is_ambiguous d) ∧
      (∀ d : Date, d.month = m → is_ambiguous d → d ∈ consecutive)) :=
sorry

end NUMINAMATH_CALUDE_ambiguous_date_characterization_max_consecutive_ambiguous_proof_l483_48351


namespace NUMINAMATH_CALUDE_unique_solution_mn_l483_48355

theorem unique_solution_mn : 
  ∃! (m n : ℕ+), 18 * (m : ℝ) * (n : ℝ) = 73 - 9 * (m : ℝ) - 3 * (n : ℝ) ∧ m = 4 ∧ n = 18 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_mn_l483_48355


namespace NUMINAMATH_CALUDE_average_problem_l483_48344

theorem average_problem (x : ℝ) : (0.4 + x) / 2 = 0.2025 → x = 0.005 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l483_48344


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l483_48335

theorem sum_of_three_numbers (a b c : ℝ) 
  (sum_of_squares : a^2 + b^2 + c^2 = 149)
  (sum_of_products : a*b + b*c + c*a = 70) :
  a + b + c = 17 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l483_48335


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l483_48300

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Theorem: In an arithmetic sequence, if a_1 + a_2 = 5 and a_3 + a_4 = 7, then a_5 + a_6 = 9 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a)
    (h_sum_12 : a 1 + a 2 = 5)
    (h_sum_34 : a 3 + a 4 = 7) :
  a 5 + a 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l483_48300


namespace NUMINAMATH_CALUDE_binary_sum_equals_141_l483_48396

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The first binary number $1010101_2$ -/
def binary1 : List Bool := [true, false, true, false, true, false, true]

/-- The second binary number $111000_2$ -/
def binary2 : List Bool := [false, false, false, true, true, true]

/-- Theorem stating that the sum of the two binary numbers is 141 in decimal -/
theorem binary_sum_equals_141 : 
  binary_to_decimal binary1 + binary_to_decimal binary2 = 141 := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_equals_141_l483_48396


namespace NUMINAMATH_CALUDE_cost_reduction_per_meter_l483_48350

/-- Proves that the reduction in cost per meter is 1 Rs -/
theorem cost_reduction_per_meter
  (original_cost : ℝ)
  (original_length : ℝ)
  (new_length : ℝ)
  (h_original_cost : original_cost = 35)
  (h_original_length : original_length = 10)
  (h_new_length : new_length = 14)
  (h_total_cost_unchanged : original_cost = new_length * (original_cost / original_length - x))
  : x = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_cost_reduction_per_meter_l483_48350


namespace NUMINAMATH_CALUDE_invitation_ways_l483_48361

-- Define the total number of classmates
def total_classmates : ℕ := 10

-- Define the number of classmates to invite
def invited_classmates : ℕ := 6

-- Define the number of classmates excluding A and B
def remaining_classmates : ℕ := total_classmates - 2

-- Function to calculate combinations
def combination (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem statement
theorem invitation_ways : 
  combination remaining_classmates (invited_classmates - 2) + 
  combination remaining_classmates invited_classmates = 98 := by
sorry

end NUMINAMATH_CALUDE_invitation_ways_l483_48361


namespace NUMINAMATH_CALUDE_point_coordinates_l483_48390

def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

def distance_to_x_axis (y : ℝ) : ℝ := |y|

def distance_to_y_axis (x : ℝ) : ℝ := |x|

theorem point_coordinates :
  ∀ (x y : ℝ),
    fourth_quadrant x y →
    distance_to_x_axis y = 3 →
    distance_to_y_axis x = 4 →
    (x, y) = (4, -3) :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_l483_48390


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l483_48329

/-- Given a geometric sequence {aₙ} with a₁ + a₂ = -1 and a₁ - a₃ = -3, prove that a₄ = -8 -/
theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)  -- The sequence
  (h_geom : ∃ (q : ℝ), ∀ n, a (n + 1) = a n * q)  -- Geometric sequence condition
  (h_sum : a 1 + a 2 = -1)  -- First condition
  (h_diff : a 1 - a 3 = -3)  -- Second condition
  : a 4 = -8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l483_48329


namespace NUMINAMATH_CALUDE_expand_and_simplify_l483_48391

theorem expand_and_simplify (a b : ℝ) : (3*a - b) * (-3*a - b) = b^2 - 9*a^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l483_48391


namespace NUMINAMATH_CALUDE_annie_village_trick_or_treat_l483_48397

/-- The number of blocks in Annie's village -/
def num_blocks : ℕ := 9

/-- The number of children on each block -/
def children_per_block : ℕ := 6

/-- The total number of children going trick or treating in Annie's village -/
def total_children : ℕ := num_blocks * children_per_block

theorem annie_village_trick_or_treat : total_children = 54 := by
  sorry

end NUMINAMATH_CALUDE_annie_village_trick_or_treat_l483_48397


namespace NUMINAMATH_CALUDE_total_donation_is_1570_l483_48318

/-- Represents the donation amounts to different parks -/
structure Donations where
  treetown_and_forest : ℝ
  forest_reserve : ℝ
  animal_preservation : ℝ

/-- Calculates the total donation to all three parks -/
def total_donation (d : Donations) : ℝ :=
  d.treetown_and_forest + d.animal_preservation

/-- Theorem stating the total donation to all three parks -/
theorem total_donation_is_1570 (d : Donations) 
  (h1 : d.treetown_and_forest = 570)
  (h2 : d.forest_reserve = d.animal_preservation + 140)
  (h3 : d.treetown_and_forest = d.forest_reserve + d.animal_preservation) : 
  total_donation d = 1570 := by
  sorry

#check total_donation_is_1570

end NUMINAMATH_CALUDE_total_donation_is_1570_l483_48318


namespace NUMINAMATH_CALUDE_section_B_seats_l483_48337

-- Define the number of seats in the different subsections of Section A
def seats_subsection_1 : ℕ := 60
def seats_subsection_2 : ℕ := 80
def num_subsection_2 : ℕ := 3

-- Define the total number of seats in Section A
def total_seats_A : ℕ := seats_subsection_1 + seats_subsection_2 * num_subsection_2

-- Define the number of seats in Section B
def seats_B : ℕ := 3 * total_seats_A + 20

-- Theorem statement
theorem section_B_seats : seats_B = 920 := by sorry

end NUMINAMATH_CALUDE_section_B_seats_l483_48337


namespace NUMINAMATH_CALUDE_square_perimeter_l483_48388

theorem square_perimeter (rectangle_length rectangle_width : ℝ)
  (h1 : rectangle_length = 50)
  (h2 : rectangle_width = 10)
  (h3 : rectangle_length > 0)
  (h4 : rectangle_width > 0) :
  let rectangle_area := rectangle_length * rectangle_width
  let square_area := 5 * rectangle_area
  let square_side := Real.sqrt square_area
  let square_perimeter := 4 * square_side
  square_perimeter = 200 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_l483_48388


namespace NUMINAMATH_CALUDE_circle_center_coordinate_sum_l483_48326

theorem circle_center_coordinate_sum (x y : ℝ) : 
  (x^2 + y^2 = 8*x - 6*y - 20) → (∃ h k : ℝ, (x - h)^2 + (y - k)^2 = (h^2 + k^2 + 20) ∧ h + k = 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_coordinate_sum_l483_48326


namespace NUMINAMATH_CALUDE_second_row_sum_is_528_l483_48341

/-- Represents a square grid -/
structure Grid (n : ℕ) :=
  (elements : Matrix (Fin n) (Fin n) ℕ)

/-- Fills the grid with numbers from 1 to n^2 in a clockwise spiral starting from the center -/
def fillGrid (n : ℕ) : Grid n :=
  sorry

/-- Returns the second row from the top of the grid -/
def secondRow (g : Grid 17) : Fin 17 → ℕ :=
  sorry

/-- The greatest number in the second row -/
def maxSecondRow (g : Grid 17) : ℕ :=
  sorry

/-- The least number in the second row -/
def minSecondRow (g : Grid 17) : ℕ :=
  sorry

/-- Theorem stating that the sum of the greatest and least numbers in the second row is 528 -/
theorem second_row_sum_is_528 :
  let g := fillGrid 17
  maxSecondRow g + minSecondRow g = 528 :=
sorry

end NUMINAMATH_CALUDE_second_row_sum_is_528_l483_48341


namespace NUMINAMATH_CALUDE_cubic_function_and_tangent_lines_l483_48356

/-- Given a cubic function f(x) = ax³ + b with a tangent line y = 3x - 1 at x = 1,
    prove that f(x) = x³ + 1 and find the equations of tangent lines passing through (-1, 0) --/
theorem cubic_function_and_tangent_lines 
  (f : ℝ → ℝ) 
  (a b : ℝ) 
  (h1 : ∀ x, f x = a * x^3 + b)
  (h2 : ∃ c d, ∀ x, (3 : ℝ) * x - 1 = c * (x - 1) + d ∧ f 1 = d ∧ (deriv f) 1 = c) :
  (∀ x, f x = x^3 + 1) ∧ 
  (∃ m₁ m₂ : ℝ, 
    (m₁ = 3 ∧ f (-1) = 0 ∧ (deriv f) (-1) = m₁) ∨ 
    (m₂ = 3/4 ∧ f (-1) = 0 ∧ (deriv f) (-1) = m₂)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_and_tangent_lines_l483_48356


namespace NUMINAMATH_CALUDE_john_tax_increase_l483_48302

/-- Calculates the increase in taxes paid given old and new tax rates and incomes -/
def tax_increase (old_rate new_rate : ℚ) (old_income new_income : ℕ) : ℚ :=
  new_rate * new_income - old_rate * old_income

/-- Proves that John's tax increase is $250,000 -/
theorem john_tax_increase :
  let old_rate : ℚ := 1/5
  let new_rate : ℚ := 3/10
  let old_income : ℕ := 1000000
  let new_income : ℕ := 1500000
  tax_increase old_rate new_rate old_income new_income = 250000 := by
  sorry

#eval tax_increase (1/5) (3/10) 1000000 1500000

end NUMINAMATH_CALUDE_john_tax_increase_l483_48302


namespace NUMINAMATH_CALUDE_ant_path_count_l483_48359

/-- The number of paths from A to B -/
def paths_AB : ℕ := 3

/-- The number of paths from B to C -/
def paths_BC : ℕ := 3

/-- The total number of paths from A to C through B -/
def total_paths : ℕ := paths_AB * paths_BC

/-- Theorem stating that the total number of paths from A to C through B is 9 -/
theorem ant_path_count : total_paths = 9 := by
  sorry

end NUMINAMATH_CALUDE_ant_path_count_l483_48359


namespace NUMINAMATH_CALUDE_inequality_system_solution_l483_48365

theorem inequality_system_solution :
  ∀ x : ℝ, (5 * (x - 1) ≤ x + 3 ∧ (x + 1) / 2 < 2 * x) ↔ (1/3 < x ∧ x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l483_48365


namespace NUMINAMATH_CALUDE_triangle_properties_l483_48339

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.c * Real.sin t.B = Real.sqrt 3 * t.b * Real.cos t.C ∧
  t.a^2 - t.c^2 = 2 * t.b^2 ∧
  (1/2) * t.a * t.b * Real.sin t.C = 21 * Real.sqrt 3

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : satisfies_conditions t) :
  t.C = π/3 ∧ t.b = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l483_48339


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_angles_pi_half_l483_48398

/-- Two vectors are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_sum_angles_pi_half
  (α β : ℝ)
  (h_acute_α : 0 < α ∧ α < π / 2)
  (h_acute_β : 0 < β ∧ β < π / 2)
  (a : ℝ × ℝ)
  (b : ℝ × ℝ)
  (h_a : a = (Real.sin α, Real.cos β))
  (h_b : b = (Real.cos α, Real.sin β))
  (h_parallel : parallel a b) :
  α + β = π / 2 := by
sorry


end NUMINAMATH_CALUDE_parallel_vectors_sum_angles_pi_half_l483_48398


namespace NUMINAMATH_CALUDE_least_value_x_minus_y_minus_z_l483_48309

theorem least_value_x_minus_y_minus_z :
  ∀ (x y z : ℕ+), x = 4 → y = 7 → (x : ℤ) - y - z ≥ -4 ∧ ∃ (z : ℕ+), (x : ℤ) - y - z = -4 :=
by sorry

end NUMINAMATH_CALUDE_least_value_x_minus_y_minus_z_l483_48309


namespace NUMINAMATH_CALUDE_max_sum_given_constraints_l483_48392

theorem max_sum_given_constraints (x y : ℝ) 
  (h1 : x^2 + y^2 = 130)
  (h2 : x * y = 45) :
  x + y ≤ 2 * Real.sqrt 55 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_given_constraints_l483_48392


namespace NUMINAMATH_CALUDE_cross_in_square_l483_48370

/-- Given a square with side length S containing a cross made of two large squares
    (each with side length S/2) and two small squares (each with side length S/4),
    if the total area of the cross is 810 cm², then S = 36 cm. -/
theorem cross_in_square (S : ℝ) : 
  (2 * (S/2)^2 + 2 * (S/4)^2 = 810) → S = 36 := by
  sorry

end NUMINAMATH_CALUDE_cross_in_square_l483_48370


namespace NUMINAMATH_CALUDE_remaining_average_l483_48336

theorem remaining_average (n : ℕ) (total_avg : ℚ) (partial_avg : ℚ) :
  n = 10 →
  total_avg = 80 →
  partial_avg = 58 →
  ∃ (m : ℕ), m = 6 ∧
    (n * total_avg - m * partial_avg) / (n - m) = 113 :=
by sorry

end NUMINAMATH_CALUDE_remaining_average_l483_48336


namespace NUMINAMATH_CALUDE_sum_squared_equals_four_l483_48324

theorem sum_squared_equals_four (a b c : ℝ) 
  (h : a^2 + b^2 + c^2 - 2*a + 4*b - 6*c + 14 = 0) : 
  (a + b + c)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_squared_equals_four_l483_48324


namespace NUMINAMATH_CALUDE_mean_calculation_l483_48352

theorem mean_calculation (x : ℝ) : 
  (28 + x + 42 + 78 + 104) / 5 = 62 → 
  (48 + 62 + 98 + 124 + x) / 5 = 78 := by
sorry

end NUMINAMATH_CALUDE_mean_calculation_l483_48352


namespace NUMINAMATH_CALUDE_least_number_with_remainder_l483_48310

theorem least_number_with_remainder (n : ℕ) : n = 115 ↔ 
  (n > 0 ∧ 
   n % 38 = 1 ∧ 
   n % 3 = 1 ∧ 
   ∀ m : ℕ, m > 0 → m % 38 = 1 → m % 3 = 1 → n ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_l483_48310


namespace NUMINAMATH_CALUDE_polygon_diagonals_l483_48334

theorem polygon_diagonals (n : ℕ) (d : ℕ) : n = 17 ∧ d = 104 →
  (n - 1) * (n - 4) / 2 = d := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l483_48334


namespace NUMINAMATH_CALUDE_sum_of_three_nines_power_twenty_l483_48321

theorem sum_of_three_nines_power_twenty (n : ℕ) : 9^20 + 9^20 + 9^20 = 3^41 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_nines_power_twenty_l483_48321


namespace NUMINAMATH_CALUDE_birch_count_l483_48353

/-- Represents the number of trees of each species in the forest -/
structure ForestComposition where
  oak : ℕ
  pine : ℕ
  spruce : ℕ
  birch : ℕ

/-- The total number of trees in the forest -/
def total_trees : ℕ := 4000

/-- The forest composition satisfies the given conditions -/
def is_valid_composition (fc : ForestComposition) : Prop :=
  fc.oak + fc.pine + fc.spruce + fc.birch = total_trees ∧
  fc.spruce = total_trees / 10 ∧
  fc.pine = total_trees * 13 / 100 ∧
  fc.oak = fc.spruce + fc.pine

theorem birch_count (fc : ForestComposition) (h : is_valid_composition fc) : fc.birch = 2160 := by
  sorry

#check birch_count

end NUMINAMATH_CALUDE_birch_count_l483_48353


namespace NUMINAMATH_CALUDE_lcm_gcd_product_36_60_l483_48354

theorem lcm_gcd_product_36_60 : Nat.lcm 36 60 * Nat.gcd 36 60 = 2160 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_36_60_l483_48354


namespace NUMINAMATH_CALUDE_cafeteria_shirt_ratio_l483_48399

/-- Proves that the ratio of people wearing horizontal stripes to people wearing checkered shirts is 4:1 --/
theorem cafeteria_shirt_ratio 
  (total_people : ℕ) 
  (checkered_shirts : ℕ) 
  (vertical_stripes : ℕ) 
  (h1 : total_people = 40)
  (h2 : checkered_shirts = 7)
  (h3 : vertical_stripes = 5) :
  (total_people - checkered_shirts - vertical_stripes) / checkered_shirts = 4 := by
sorry

end NUMINAMATH_CALUDE_cafeteria_shirt_ratio_l483_48399


namespace NUMINAMATH_CALUDE_certain_number_calculation_l483_48387

theorem certain_number_calculation : ∃ (n : ℕ), 9823 + 3377 = n := by
  sorry

end NUMINAMATH_CALUDE_certain_number_calculation_l483_48387


namespace NUMINAMATH_CALUDE_third_side_length_l483_48373

/-- A right-angled isosceles triangle with specific dimensions -/
structure RightIsoscelesTriangle where
  /-- The length of the equal sides -/
  a : ℝ
  /-- The length of the hypotenuse -/
  c : ℝ
  /-- The triangle is right-angled -/
  right_angled : a^2 + a^2 = c^2
  /-- The triangle is isosceles -/
  isosceles : a = 50
  /-- The perimeter of the triangle -/
  perimeter : a + a + c = 160

/-- The theorem stating the length of the third side -/
theorem third_side_length (t : RightIsoscelesTriangle) : t.c = 60 := by
  sorry

end NUMINAMATH_CALUDE_third_side_length_l483_48373


namespace NUMINAMATH_CALUDE_inequality_holds_iff_l483_48385

theorem inequality_holds_iff (x : ℝ) : 
  0 ≤ x ∧ x ≤ 2 * π →
  (2 * Real.cos x ≤ |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| ∧
   |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| ≤ Real.sqrt 2) ↔
  (π / 4 ≤ x ∧ x ≤ 7 * π / 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_l483_48385


namespace NUMINAMATH_CALUDE_vertices_count_l483_48343

/-- Represents a convex polyhedron -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ

/-- Euler's formula for convex polyhedra -/
axiom eulers_formula (p : ConvexPolyhedron) : p.vertices - p.edges + p.faces = 2

/-- A face of a polyhedron -/
inductive Face
| Triangle : Face

/-- Our specific polyhedron -/
def our_polyhedron : ConvexPolyhedron where
  vertices := 12  -- This is what we want to prove
  edges := 30
  faces := 20

/-- All faces of our polyhedron are triangles -/
axiom all_faces_triangular : ∀ f : Face, f = Face.Triangle

/-- The number of vertices in our polyhedron is correct -/
theorem vertices_count : our_polyhedron.vertices = 12 := by sorry

end NUMINAMATH_CALUDE_vertices_count_l483_48343


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l483_48389

theorem modulus_of_complex_fraction :
  let z : ℂ := (-3 + I) / (2 + I)
  Complex.abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l483_48389


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l483_48376

theorem quadratic_equation_solution : ∃ (a b : ℝ), 
  (a^2 - 4*a + 7 = 19) ∧ 
  (b^2 - 4*b + 7 = 19) ∧ 
  (a ≥ b) ∧ 
  (2*a + b = 10) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l483_48376


namespace NUMINAMATH_CALUDE_amount_distributed_l483_48386

theorem amount_distributed (A : ℚ) : 
  (∀ (x : ℚ), x = A / 30 - A / 40 → x = 135.50) →
  A = 16260 := by
sorry

end NUMINAMATH_CALUDE_amount_distributed_l483_48386


namespace NUMINAMATH_CALUDE_expression_simplification_l483_48393

theorem expression_simplification (x y : ℝ) :
  (x + 2*y) * (x - 2*y) - x * (x + 3*y) = -4*y^2 - 3*x*y ∧
  (x - 1 - 3/(x + 1)) / ((x^2 - 4*x + 4) / (x + 1)) = (x + 2) / (x - 2) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l483_48393


namespace NUMINAMATH_CALUDE_permutations_of_six_books_l483_48338

theorem permutations_of_six_books : Nat.factorial 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_six_books_l483_48338


namespace NUMINAMATH_CALUDE_allstar_seating_arrangements_l483_48380

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def num_allstars : ℕ := 9
def num_cubs : ℕ := 3
def num_redsox : ℕ := 3
def num_yankees : ℕ := 2
def num_dodgers : ℕ := 1
def num_teams : ℕ := 4

theorem allstar_seating_arrangements :
  (factorial num_teams) * (factorial num_cubs) * (factorial num_redsox) * 
  (factorial num_yankees) * (factorial num_dodgers) = 1728 := by
  sorry

end NUMINAMATH_CALUDE_allstar_seating_arrangements_l483_48380


namespace NUMINAMATH_CALUDE_largest_x_value_l483_48381

-- Define the probability function
def prob (x y : ℕ) : ℚ :=
  (Nat.choose x 2 + Nat.choose y 2) / Nat.choose (x + y) 2

-- State the theorem
theorem largest_x_value :
  ∀ x y : ℕ,
    x > y →
    x + y ≤ 2008 →
    prob x y = 1/2 →
    x ≤ 990 ∧ (∃ x' y' : ℕ, x' = 990 ∧ y' = 946 ∧ x' > y' ∧ x' + y' ≤ 2008 ∧ prob x' y' = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_largest_x_value_l483_48381


namespace NUMINAMATH_CALUDE_min_distinct_values_l483_48371

/-- A list of positive integers -/
def IntegerList := List ℕ+

/-- The number of occurrences of the most frequent element in a list -/
def modeCount (l : IntegerList) : ℕ := sorry

/-- The number of distinct elements in a list -/
def distinctCount (l : IntegerList) : ℕ := sorry

/-- Theorem: Minimum number of distinct values in a list of 4000 positive integers
    with a unique mode occurring exactly 20 times is 211 -/
theorem min_distinct_values (l : IntegerList) 
  (h1 : l.length = 4000)
  (h2 : ∃! x, modeCount l = x)
  (h3 : modeCount l = 20) :
  distinctCount l ≥ 211 := by sorry

end NUMINAMATH_CALUDE_min_distinct_values_l483_48371


namespace NUMINAMATH_CALUDE_negation_equivalence_l483_48383

theorem negation_equivalence (x : ℝ) :
  ¬(x^2 - x ≥ 0 → x > 2) ↔ (x^2 - x < 0 → x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l483_48383


namespace NUMINAMATH_CALUDE_last_colored_cell_position_l483_48331

/-- Represents a cell position in the grid -/
structure CellPosition where
  row : Nat
  col : Nat

/-- Represents the dimensions of the rectangle -/
structure RectangleDimensions where
  width : Nat
  height : Nat

/-- Represents the coloring process in a spiral pattern -/
def spiralColor (dim : RectangleDimensions) : CellPosition :=
  sorry

/-- Theorem: The last cell colored in a 200x100 rectangle with spiral coloring is at (51, 50) -/
theorem last_colored_cell_position :
  let dim : RectangleDimensions := ⟨200, 100⟩
  spiralColor dim = ⟨51, 50⟩ := by
  sorry

end NUMINAMATH_CALUDE_last_colored_cell_position_l483_48331


namespace NUMINAMATH_CALUDE_half_of_eighteen_is_nine_l483_48360

theorem half_of_eighteen_is_nine : (18 : ℝ) / 2 = 9 := by sorry

end NUMINAMATH_CALUDE_half_of_eighteen_is_nine_l483_48360


namespace NUMINAMATH_CALUDE_singleton_quadratic_set_l483_48303

theorem singleton_quadratic_set (m : ℝ) : 
  (∃! x : ℝ, x^2 - 4*x + m = 0) → m = 4 := by
sorry

end NUMINAMATH_CALUDE_singleton_quadratic_set_l483_48303


namespace NUMINAMATH_CALUDE_count_integer_pairs_l483_48301

theorem count_integer_pairs : 
  ∃ (count : ℕ), 
    (2^2876 < 3^1250 ∧ 3^1250 < 2^2877) →
    count = (Finset.filter 
      (λ (pair : ℕ × ℕ) => 
        let (m, n) := pair
        1 ≤ m ∧ m ≤ 2875 ∧ 3^n < 2^m ∧ 2^m < 2^(m+3) ∧ 2^(m+3) < 3^(n+1))
      (Finset.product (Finset.range 2876) (Finset.range (1250 + 1)))).card ∧
    count = 3750 :=
by sorry

end NUMINAMATH_CALUDE_count_integer_pairs_l483_48301


namespace NUMINAMATH_CALUDE_tv_sale_value_change_l483_48316

theorem tv_sale_value_change 
  (original_price : ℝ) 
  (original_quantity : ℝ) 
  (price_reduction_percent : ℝ) 
  (sales_increase_percent : ℝ) 
  (h1 : price_reduction_percent = 10) 
  (h2 : sales_increase_percent = 85) : 
  let new_price := original_price * (1 - price_reduction_percent / 100)
  let new_quantity := original_quantity * (1 + sales_increase_percent / 100)
  let original_value := original_price * original_quantity
  let new_value := new_price * new_quantity
  (new_value - original_value) / original_value * 100 = 66.5 := by
sorry

end NUMINAMATH_CALUDE_tv_sale_value_change_l483_48316


namespace NUMINAMATH_CALUDE_range_of_a_l483_48395

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 + 2*x - 3 > 0
def q (x a : ℝ) : Prop := (x - a) / (x - a - 1) > 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (∀ x, ¬(q x a) → ¬(p x)) ∧ 
  (∃ x, ¬(p x) ∧ (q x a)) →
  a ∈ Set.Icc (-3) 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l483_48395


namespace NUMINAMATH_CALUDE_horner_method_evaluation_l483_48374

def f (x : ℝ) : ℝ := x^5 + 2*x^4 + 3*x^3 + 4*x^2 + 5*x + 6

theorem horner_method_evaluation : f 5 = 4881 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_evaluation_l483_48374


namespace NUMINAMATH_CALUDE_playground_girls_l483_48367

theorem playground_girls (total_children : ℕ) (boys : ℕ) (girls : ℕ) :
  total_children = 97 → boys = 44 → girls = total_children - boys → girls = 53 := by
  sorry

end NUMINAMATH_CALUDE_playground_girls_l483_48367


namespace NUMINAMATH_CALUDE_money_division_l483_48364

theorem money_division (r s t u : ℝ) (h1 : r / s = 2.5 / 3.5) (h2 : s / t = 3.5 / 7.5) 
  (h3 : t / u = 7.5 / 9.8) (h4 : t - s = 4500) : u - r = 8212.5 := by
  sorry

end NUMINAMATH_CALUDE_money_division_l483_48364


namespace NUMINAMATH_CALUDE_geometric_sequence_154th_term_l483_48378

/-- Represents a geometric sequence with first term a₁ and common ratio r -/
def GeometricSequence (a₁ : ℝ) (r : ℝ) : ℕ → ℝ := fun n => a₁ * r ^ (n - 1)

/-- The 154th term of a geometric sequence with first term 4 and second term 12 -/
theorem geometric_sequence_154th_term :
  let seq := GeometricSequence 4 3
  seq 154 = 4 * 3^153 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_154th_term_l483_48378


namespace NUMINAMATH_CALUDE_cube_sum_inequality_l483_48328

theorem cube_sum_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq : a + b = c + d) (square_sum_gt : a^2 + b^2 > c^2 + d^2) :
  a^3 + b^3 > c^3 + d^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_inequality_l483_48328


namespace NUMINAMATH_CALUDE_average_of_multiples_of_seven_l483_48332

theorem average_of_multiples_of_seven (n : ℕ) : 
  (n / 2 : ℚ) * (7 + 7 * n) / n = 77 → n = 21 := by
  sorry

end NUMINAMATH_CALUDE_average_of_multiples_of_seven_l483_48332


namespace NUMINAMATH_CALUDE_consecutive_product_sum_l483_48314

theorem consecutive_product_sum : ∃ (a b c d e : ℤ),
  (b = a + 1) ∧
  (d = c + 1) ∧
  (e = d + 1) ∧
  (a * b = 990) ∧
  (c * d * e = 990) ∧
  (a + b + c + d + e = 90) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_product_sum_l483_48314


namespace NUMINAMATH_CALUDE_circle_radius_from_chords_l483_48362

/-- Given a circle with two chords of lengths 20 cm and 26 cm starting from the same point
    and forming an angle of 36° 38', the radius of the circle is approximately 24.84 cm. -/
theorem circle_radius_from_chords (chord1 chord2 angle : ℝ) (h1 : chord1 = 20)
    (h2 : chord2 = 26) (h3 : angle = 36 + 38 / 60) : ∃ r : ℝ, 
    abs (r - 24.84) < 0.01 ∧ 
    chord1^2 + chord2^2 - 2 * chord1 * chord2 * Real.cos (angle * Real.pi / 180) = 
    4 * r^2 * Real.sin ((angle * Real.pi / 180) / 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_chords_l483_48362


namespace NUMINAMATH_CALUDE_prize_selection_theorem_l483_48346

/-- Represents the systematic sampling of prizes -/
def systematicSampling (totalPrizes : ℕ) (sampleSize : ℕ) (firstPrize : ℕ) : List ℕ :=
  let interval := totalPrizes / sampleSize
  List.range sampleSize |>.map (fun i => firstPrize + i * interval)

/-- Theorem: Given the conditions of the prize selection, the other four prizes are 46, 86, 126, and 166 -/
theorem prize_selection_theorem (totalPrizes : ℕ) (sampleSize : ℕ) (firstPrize : ℕ) 
    (h1 : totalPrizes = 200)
    (h2 : sampleSize = 5)
    (h3 : firstPrize = 6) :
  systematicSampling totalPrizes sampleSize firstPrize = [6, 46, 86, 126, 166] := by
  sorry

#eval systematicSampling 200 5 6

end NUMINAMATH_CALUDE_prize_selection_theorem_l483_48346


namespace NUMINAMATH_CALUDE_sum_of_roots_is_twelve_l483_48311

/-- A function g: ℝ → ℝ that satisfies g(3+x) = g(3-x) for all real x -/
def SymmetricAboutThree (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (3 + x) = g (3 - x)

/-- The proposition that g has exactly four distinct real roots -/
def HasFourDistinctRoots (g : ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
    (g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0) ∧
    (∀ x : ℝ, g x = 0 → (x = a ∨ x = b ∨ x = c ∨ x = d))

/-- The theorem stating that the sum of roots is 12 -/
theorem sum_of_roots_is_twelve (g : ℝ → ℝ) 
    (h1 : SymmetricAboutThree g) (h2 : HasFourDistinctRoots g) : 
    ∃ (a b c d : ℝ), (g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0) ∧ (a + b + c + d = 12) :=
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_twelve_l483_48311


namespace NUMINAMATH_CALUDE_constant_value_l483_48305

-- Define the function f
def f (x : ℝ) : ℝ := x + 4

-- Define the equation
def equation (c : ℝ) (x : ℝ) : Prop :=
  (3 * f (x - 2)) / f 0 + 4 = f (c * x + 1)

-- Theorem statement
theorem constant_value :
  ∀ c : ℝ, equation c 0.4 → c = 2 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_l483_48305


namespace NUMINAMATH_CALUDE_cos_570_deg_l483_48315

theorem cos_570_deg : Real.cos (570 * π / 180) = - Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_570_deg_l483_48315


namespace NUMINAMATH_CALUDE_jimmy_stair_climb_time_l483_48319

def stairClimbTime (n : ℕ) : ℕ :=
  let baseTime := 25
  let increment := 7
  let flightTimes := List.range n |>.map (λ i => baseTime + i * increment)
  let totalFlightTime := flightTimes.sum
  let stopTime := (n - 1) / 2 * 10
  totalFlightTime + stopTime

theorem jimmy_stair_climb_time :
  stairClimbTime 7 = 342 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_stair_climb_time_l483_48319


namespace NUMINAMATH_CALUDE_domain_of_composite_function_l483_48342

theorem domain_of_composite_function 
  (f : ℝ → ℝ) 
  (h : ∀ k : ℤ, ∀ x : ℝ, x ∈ Set.Icc (2 * k * Real.pi - Real.pi / 6) (2 * k * Real.pi + 2 * Real.pi / 3) → f (Real.cos x) ∈ Set.range f) :
  Set.range f = Set.Icc (-1/2) 1 := by
sorry

end NUMINAMATH_CALUDE_domain_of_composite_function_l483_48342


namespace NUMINAMATH_CALUDE_equation_solution_l483_48375

theorem equation_solution : ∃ x : ℝ, x ≠ 2 ∧ (4*x^2 + 3*x + 2) / (x - 2) = 4*x + 5 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l483_48375


namespace NUMINAMATH_CALUDE_students_liking_both_desserts_l483_48330

/-- Given a class of students, calculate the number who like both apple pie and chocolate cake. -/
theorem students_liking_both_desserts 
  (total_students : ℕ) 
  (like_apple_pie : ℕ) 
  (like_chocolate_cake : ℕ) 
  (like_neither : ℕ) 
  (h1 : total_students = 50)
  (h2 : like_apple_pie = 22)
  (h3 : like_chocolate_cake = 20)
  (h4 : like_neither = 15) :
  like_apple_pie + like_chocolate_cake - (total_students - like_neither) = 7 := by
  sorry

#check students_liking_both_desserts

end NUMINAMATH_CALUDE_students_liking_both_desserts_l483_48330
