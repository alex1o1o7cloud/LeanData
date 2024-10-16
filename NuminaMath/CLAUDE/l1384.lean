import Mathlib

namespace NUMINAMATH_CALUDE_linear_equation_negative_root_m_range_l1384_138458

theorem linear_equation_negative_root_m_range 
  (m : ℝ) 
  (h : ∃ x : ℝ, (3 * x - m + 1 = 2 * x - 1) ∧ (x < 0)) : 
  m < 2 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_negative_root_m_range_l1384_138458


namespace NUMINAMATH_CALUDE_x_value_l1384_138465

theorem x_value : ∃ x : ℝ, (2*x - 3*x + 5*x - x = 120) ∧ (x = 40) := by sorry

end NUMINAMATH_CALUDE_x_value_l1384_138465


namespace NUMINAMATH_CALUDE_smallest_even_number_sum_1194_l1384_138432

/-- Given three consecutive even numbers whose sum is 1194, 
    the smallest of these numbers is 396. -/
theorem smallest_even_number_sum_1194 (x : ℕ) 
  (h1 : x % 2 = 0)  -- x is even
  (h2 : x + (x + 2) + (x + 4) = 1194) : x = 396 := by
  sorry

end NUMINAMATH_CALUDE_smallest_even_number_sum_1194_l1384_138432


namespace NUMINAMATH_CALUDE_initial_fee_calculation_l1384_138401

/-- The initial fee for a taxi trip, given the rate per segment and total charge for a specific distance. -/
theorem initial_fee_calculation (rate_per_segment : ℝ) (total_charge : ℝ) (distance : ℝ) : 
  rate_per_segment = 0.35 →
  distance = 3.6 →
  total_charge = 5.65 →
  ∃ (initial_fee : ℝ), initial_fee = 2.50 ∧ 
    total_charge = initial_fee + (distance / (2/5)) * rate_per_segment :=
by sorry

end NUMINAMATH_CALUDE_initial_fee_calculation_l1384_138401


namespace NUMINAMATH_CALUDE_table_area_proof_l1384_138417

theorem table_area_proof (total_runner_area : ℝ) 
                         (coverage_percentage : ℝ)
                         (two_layer_area : ℝ)
                         (three_layer_area : ℝ) 
                         (h1 : total_runner_area = 220)
                         (h2 : coverage_percentage = 0.80)
                         (h3 : two_layer_area = 24)
                         (h4 : three_layer_area = 28) :
  ∃ (table_area : ℝ), table_area = 275 ∧ 
    coverage_percentage * table_area = total_runner_area := by
  sorry


end NUMINAMATH_CALUDE_table_area_proof_l1384_138417


namespace NUMINAMATH_CALUDE_tank_unoccupied_volume_l1384_138428

/-- Calculates the unoccupied volume in a cube-shaped tank --/
def unoccupied_volume (tank_side : ℝ) (water_fraction : ℝ) (ice_cube_side : ℝ) (num_ice_cubes : ℕ) : ℝ :=
  let tank_volume := tank_side ^ 3
  let water_volume := water_fraction * tank_volume
  let ice_cube_volume := ice_cube_side ^ 3
  let total_ice_volume := (num_ice_cubes : ℝ) * ice_cube_volume
  let occupied_volume := water_volume + total_ice_volume
  tank_volume - occupied_volume

/-- Theorem stating the unoccupied volume in the tank --/
theorem tank_unoccupied_volume :
  unoccupied_volume 12 (1/3) 1.5 15 = 1101.375 := by
  sorry

end NUMINAMATH_CALUDE_tank_unoccupied_volume_l1384_138428


namespace NUMINAMATH_CALUDE_minimum_questionnaires_to_mail_l1384_138408

def response_rate : ℝ := 0.62
def required_responses : ℕ := 300

theorem minimum_questionnaires_to_mail : 
  ∃ n : ℕ, n > 0 ∧ 
  (↑n * response_rate : ℝ) ≥ required_responses ∧
  ∀ m : ℕ, m < n → (↑m * response_rate : ℝ) < required_responses :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_questionnaires_to_mail_l1384_138408


namespace NUMINAMATH_CALUDE_marble_bag_count_l1384_138449

theorem marble_bag_count (blue red white : ℕ) (total : ℕ) : 
  blue = 5 →
  red = 9 →
  total = blue + red + white →
  (red + white : ℚ) / total = 3/4 →
  total = 20 := by
sorry

end NUMINAMATH_CALUDE_marble_bag_count_l1384_138449


namespace NUMINAMATH_CALUDE_f_negative_2017_l1384_138474

noncomputable def f (x : ℝ) : ℝ :=
  ((x + 1)^2 + Real.log (Real.sqrt (1 + 9*x^2) - 3*x) * Real.cos x) / (x^2 + 1)

theorem f_negative_2017 (h : f 2017 = 2016) : f (-2017) = -2014 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_2017_l1384_138474


namespace NUMINAMATH_CALUDE_sin_linear_dependence_l1384_138424

theorem sin_linear_dependence :
  ∃ (α₁ α₂ α₃ : ℝ), (α₁ ≠ 0 ∨ α₂ ≠ 0 ∨ α₃ ≠ 0) ∧
  ∀ x : ℝ, α₁ * Real.sin x + α₂ * Real.sin (x + π/8) + α₃ * Real.sin (x - π/8) = 0 := by
sorry

end NUMINAMATH_CALUDE_sin_linear_dependence_l1384_138424


namespace NUMINAMATH_CALUDE_skew_lines_theorem_l1384_138452

-- Define the concept of a line in 3D space
structure Line3D where
  -- This is a placeholder definition. In a real scenario, we would need to define
  -- what constitutes a line in 3D space, likely using vectors or points.
  mk :: (dummy : Unit)

-- Define what it means for lines to be skew
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Two lines are skew if they are not coplanar and do not intersect
  sorry

-- Define what it means for lines to be parallel
def are_parallel (l1 l2 : Line3D) : Prop :=
  -- Two lines are parallel if they are coplanar and do not intersect
  sorry

-- Define what it means for lines to intersect
def do_intersect (l1 l2 : Line3D) : Prop :=
  -- Two lines intersect if they share a point
  sorry

-- The main theorem
theorem skew_lines_theorem (a b c : Line3D) 
  (h1 : are_skew a b)
  (h2 : are_parallel a c)
  (h3 : ¬do_intersect b c) :
  are_skew b c := by
  sorry

end NUMINAMATH_CALUDE_skew_lines_theorem_l1384_138452


namespace NUMINAMATH_CALUDE_emma_bank_money_l1384_138433

theorem emma_bank_money (X : ℝ) : 
  X > 0 →
  (1/4 : ℝ) * (X - 400) = 400 →
  X = 2000 := by
sorry

end NUMINAMATH_CALUDE_emma_bank_money_l1384_138433


namespace NUMINAMATH_CALUDE_project_hours_difference_l1384_138496

theorem project_hours_difference (total_hours : ℝ) 
  (h_total : total_hours = 350) 
  (h_pat_kate : ∃ k : ℝ, pat = 2 * k ∧ kate = k)
  (h_pat_mark : ∃ m : ℝ, pat = (1/3) * m ∧ mark = m)
  (h_alex_kate : ∃ k : ℝ, alex = 1.5 * k ∧ kate = k)
  (h_sum : pat + kate + mark + alex = total_hours) :
  mark - (kate + alex) = 350/3 :=
sorry

end NUMINAMATH_CALUDE_project_hours_difference_l1384_138496


namespace NUMINAMATH_CALUDE_houses_in_block_is_five_l1384_138459

/-- The number of houses in a block -/
def houses_in_block : ℕ := 5

/-- The number of candies received from each house -/
def candies_per_house : ℕ := 7

/-- The total number of candies received from each block -/
def candies_per_block : ℕ := 35

/-- Theorem: The number of houses in a block is 5 -/
theorem houses_in_block_is_five :
  houses_in_block = candies_per_block / candies_per_house :=
by sorry

end NUMINAMATH_CALUDE_houses_in_block_is_five_l1384_138459


namespace NUMINAMATH_CALUDE_rectangular_field_area_l1384_138438

/-- Given a rectangular field with one side uncovered and three sides fenced, 
    calculate its area. -/
theorem rectangular_field_area 
  (L : ℝ) -- Length of the uncovered side
  (total_fencing : ℝ) -- Total length of fencing used
  (h1 : L = 20) -- The uncovered side is 20 feet long
  (h2 : total_fencing = 64) -- The total fencing required is 64 feet
  : L * ((total_fencing - L) / 2) = 440 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l1384_138438


namespace NUMINAMATH_CALUDE_keiko_speed_l1384_138439

theorem keiko_speed (track_width : ℝ) (time_difference : ℝ) : 
  track_width = 8 → time_difference = 48 → 
  (track_width * π) / time_difference = π / 3 := by
sorry

end NUMINAMATH_CALUDE_keiko_speed_l1384_138439


namespace NUMINAMATH_CALUDE_quadrilateral_perimeter_l1384_138420

/-- A quadrilateral ABCD with the following properties:
  1. AB ⊥ BC
  2. ∠DCB = 135°
  3. AB = 10 cm
  4. DC = 5 cm
  5. BC = 15 cm
-/
structure Quadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  AB_perp_BC : True  -- Represents AB ⊥ BC
  angle_DCB : ℝ
  h_AB : AB = 10
  h_CD : CD = 5
  h_BC : BC = 15
  h_angle_DCB : angle_DCB = 135

/-- The perimeter of the quadrilateral ABCD is 30 + 5√10 cm -/
theorem quadrilateral_perimeter (q : Quadrilateral) : 
  q.AB + q.BC + q.CD + Real.sqrt (q.CD^2 + q.BC^2) = 30 + 5 * Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_quadrilateral_perimeter_l1384_138420


namespace NUMINAMATH_CALUDE_permutation_equation_solution_l1384_138413

theorem permutation_equation_solution (x : ℕ) : 
  (3 * (Nat.factorial 8 / Nat.factorial (8 - x)) = 4 * (Nat.factorial 9 / Nat.factorial (10 - x))) ∧ 
  (1 ≤ x) ∧ (x ≤ 8) → 
  x = 6 := by sorry

end NUMINAMATH_CALUDE_permutation_equation_solution_l1384_138413


namespace NUMINAMATH_CALUDE_translated_line_equation_l1384_138418

/-- Given a line with slope 2 passing through the point (5, 1), prove that its equation is y = 2x - 9 -/
theorem translated_line_equation (x y : ℝ) :
  (y = 2 * x + 3) →  -- Original line equation
  (∃ b, y = 2 * x + b) →  -- Translated line has the same slope but different y-intercept
  (1 = 2 * 5 + b) →  -- The translated line passes through (5, 1)
  (y = 2 * x - 9)  -- The equation of the translated line
  := by sorry

end NUMINAMATH_CALUDE_translated_line_equation_l1384_138418


namespace NUMINAMATH_CALUDE_remainder_5n_mod_3_l1384_138499

theorem remainder_5n_mod_3 (n : ℤ) (h : n % 3 = 2) : (5 * n) % 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_5n_mod_3_l1384_138499


namespace NUMINAMATH_CALUDE_transform_minus_four_plus_two_i_l1384_138485

/-- Applies a 270° counter-clockwise rotation followed by a scaling of 2 to a complex number -/
def transform (z : ℂ) : ℂ := 2 * (z * Complex.I)

/-- The result of applying the transformation to -4 + 2i -/
theorem transform_minus_four_plus_two_i :
  transform (Complex.ofReal (-4) + Complex.I * Complex.ofReal 2) = Complex.ofReal 4 + Complex.I * Complex.ofReal 8 := by
  sorry

#check transform_minus_four_plus_two_i

end NUMINAMATH_CALUDE_transform_minus_four_plus_two_i_l1384_138485


namespace NUMINAMATH_CALUDE_dog_journey_distance_l1384_138463

/-- 
Given a journey where:
- The total time is 2 hours
- The first half of the distance is traveled at 10 km/h
- The second half of the distance is traveled at 5 km/h
Prove that the total distance traveled is 40/3 km
-/
theorem dog_journey_distance : 
  ∀ (total_distance : ℝ),
  (total_distance / 20 + total_distance / 10 = 2) →
  total_distance = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_dog_journey_distance_l1384_138463


namespace NUMINAMATH_CALUDE_m_range_for_third_quadrant_l1384_138444

/-- A complex number z is in the third quadrant if its real and imaginary parts are both negative -/
def in_third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

/-- The theorem stating that if z = (m+4) + (m-2)i is in the third quadrant, 
    then m is in the interval (-∞, -4) -/
theorem m_range_for_third_quadrant (m : ℝ) :
  let z : ℂ := Complex.mk (m + 4) (m - 2)
  in_third_quadrant z → m < -4 := by
  sorry

#check m_range_for_third_quadrant

end NUMINAMATH_CALUDE_m_range_for_third_quadrant_l1384_138444


namespace NUMINAMATH_CALUDE_arrangement_counts_l1384_138493

/-- Represents the number of people in the row -/
def n : ℕ := 5

/-- Calculates the factorial of a natural number -/
def factorial (k : ℕ) : ℕ :=
  match k with
  | 0 => 1
  | k + 1 => (k + 1) * factorial k

/-- The number of arrangements with Person A at the head -/
def arrangements_A_at_head : ℕ := factorial (n - 1)

/-- The number of arrangements with Person A and Person B adjacent -/
def arrangements_A_B_adjacent : ℕ := factorial (n - 1) * 2

/-- The number of arrangements with Person A not at the head and Person B not at the end -/
def arrangements_A_not_head_B_not_end : ℕ := (n - 1) * (n - 2) * factorial (n - 2)

/-- The number of arrangements with Person A to the left of and taller than Person B, and not adjacent -/
def arrangements_A_left_taller_not_adjacent : ℕ := 3 * factorial (n - 2)

theorem arrangement_counts :
  arrangements_A_at_head = 24 ∧
  arrangements_A_B_adjacent = 48 ∧
  arrangements_A_not_head_B_not_end = 72 ∧
  arrangements_A_left_taller_not_adjacent = 18 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_counts_l1384_138493


namespace NUMINAMATH_CALUDE_work_completion_days_l1384_138472

/-- Represents the work scenario with initial workers and additional workers joining later. -/
structure WorkScenario where
  initial_workers : ℕ
  additional_workers : ℕ
  days_saved : ℕ

/-- Calculates the original number of days required to complete the work. -/
def original_days (scenario : WorkScenario) : ℕ :=
  2 * scenario.days_saved

/-- Theorem stating that for the given scenario, the original number of days is 6. -/
theorem work_completion_days (scenario : WorkScenario) 
  (h1 : scenario.initial_workers = 10)
  (h2 : scenario.additional_workers = 10)
  (h3 : scenario.days_saved = 3) :
  original_days scenario = 6 := by
  sorry

#eval original_days { initial_workers := 10, additional_workers := 10, days_saved := 3 }

end NUMINAMATH_CALUDE_work_completion_days_l1384_138472


namespace NUMINAMATH_CALUDE_incorrect_statement_l1384_138423

theorem incorrect_statement : ¬(∀ (p q : Prop), (p ∧ q = False) → (p = False ∧ q = False)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_l1384_138423


namespace NUMINAMATH_CALUDE_total_pencils_count_l1384_138489

/-- The number of pencils each child has -/
def pencils_per_child : ℕ := 4

/-- The number of children -/
def number_of_children : ℕ := 8

/-- The total number of pencils -/
def total_pencils : ℕ := pencils_per_child * number_of_children

theorem total_pencils_count : total_pencils = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_count_l1384_138489


namespace NUMINAMATH_CALUDE_greatest_common_factor_three_digit_palindromes_l1384_138468

-- Define a three-digit palindrome
def three_digit_palindrome (a b : Nat) : Nat :=
  100 * a + 10 * b + a

-- Define the set of all three-digit palindromes
def all_three_digit_palindromes : Set Nat :=
  {n | ∃ (a b : Nat), a ≤ 9 ∧ b ≤ 9 ∧ n = three_digit_palindrome a b}

-- Theorem statement
theorem greatest_common_factor_three_digit_palindromes :
  ∃ (gcf : Nat), gcf = 11 ∧ 
  (∀ (n : Nat), n ∈ all_three_digit_palindromes → gcf ∣ n) ∧
  (∀ (d : Nat), (∀ (n : Nat), n ∈ all_three_digit_palindromes → d ∣ n) → d ≤ gcf) :=
sorry

end NUMINAMATH_CALUDE_greatest_common_factor_three_digit_palindromes_l1384_138468


namespace NUMINAMATH_CALUDE_tax_difference_is_correct_l1384_138427

-- Define the item price
def item_price : ℝ := 50

-- Define the tax rates
def high_tax_rate : ℝ := 0.075
def low_tax_rate : ℝ := 0.05

-- Define the tax difference function
def tax_difference (price : ℝ) (high_rate : ℝ) (low_rate : ℝ) : ℝ :=
  price * high_rate - price * low_rate

-- Theorem statement
theorem tax_difference_is_correct : 
  tax_difference item_price high_tax_rate low_tax_rate = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_tax_difference_is_correct_l1384_138427


namespace NUMINAMATH_CALUDE_quadratic_root_implies_a_l1384_138497

theorem quadratic_root_implies_a (a : ℝ) : 
  (2^2 - a*2 + 6 = 0) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_a_l1384_138497


namespace NUMINAMATH_CALUDE_average_increase_l1384_138400

theorem average_increase (s : Finset ℕ) (f : ℕ → ℝ) :
  s.card = 15 →
  (s.sum f) / s.card = 30 →
  (s.sum (λ x => f x + 15)) / s.card = 45 := by
sorry

end NUMINAMATH_CALUDE_average_increase_l1384_138400


namespace NUMINAMATH_CALUDE_quadratic_root_implies_n_l1384_138481

theorem quadratic_root_implies_n (n : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + n = 0) ∧ (3^2 - 2*3 + n = 0) → n = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_n_l1384_138481


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1384_138486

theorem arithmetic_sequence_ratio (a d : ℚ) : 
  let S : ℕ → ℚ := λ n => n / 2 * (2 * a + (n - 1) * d)
  S 15 = 3 * S 8 → a / d = 7 / 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1384_138486


namespace NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l1384_138455

def first_six_primes : List ℕ := [2, 3, 5, 7, 11, 13]
def seventh_prime : ℕ := 17

theorem sum_first_six_primes_mod_seventh_prime :
  (first_six_primes.sum % seventh_prime) = 7 := by sorry

end NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l1384_138455


namespace NUMINAMATH_CALUDE_inequality_solution_l1384_138441

theorem inequality_solution (n k : ℤ) :
  let x : ℝ := (-1)^n * π/6 + 2*π*n
  let y : ℝ := π/2 + π*k
  4 * Real.sin x - Real.sqrt (Real.cos y) - Real.sqrt (Real.cos y - 16 * (Real.cos x)^2 + 12) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1384_138441


namespace NUMINAMATH_CALUDE_min_sum_four_reals_l1384_138462

theorem min_sum_four_reals (x₁ x₂ x₃ x₄ : ℝ) 
  (h1 : x₁ + x₂ ≥ 12)
  (h2 : x₁ + x₃ ≥ 13)
  (h3 : x₁ + x₄ ≥ 14)
  (h4 : x₃ + x₄ ≥ 22)
  (h5 : x₂ + x₃ ≥ 23)
  (h6 : x₂ + x₄ ≥ 24) :
  x₁ + x₂ + x₃ + x₄ ≥ 37 ∧ ∃ a b c d : ℝ, a + b + c + d = 37 ∧ 
    a + b ≥ 12 ∧ a + c ≥ 13 ∧ a + d ≥ 14 ∧ c + d ≥ 22 ∧ b + c ≥ 23 ∧ b + d ≥ 24 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_four_reals_l1384_138462


namespace NUMINAMATH_CALUDE_inequality_proof_l1384_138495

def f (x : ℝ) := |2*x - 1| + |2*x + 1|

theorem inequality_proof (a b : ℝ) :
  (∀ x, -1 < x ∧ x < 1 → f x < 4) →
  -1 < a ∧ a < 1 →
  -1 < b ∧ b < 1 →
  |a + b| / |a*b + 1| < 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1384_138495


namespace NUMINAMATH_CALUDE_concatenated_number_divisible_by_1980_l1384_138410

def concatenated_number : ℕ :=
  -- Definition of the number A as described in the problem
  sorry

theorem concatenated_number_divisible_by_1980 :
  1980 ∣ concatenated_number :=
by
  sorry

end NUMINAMATH_CALUDE_concatenated_number_divisible_by_1980_l1384_138410


namespace NUMINAMATH_CALUDE_complex_modulus_l1384_138429

theorem complex_modulus (z : ℂ) : z = -6 + (3 - 5/3*I)*I → Complex.abs z = 5*Real.sqrt 10/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l1384_138429


namespace NUMINAMATH_CALUDE_min_cuts_for_20_gons_l1384_138479

/-- Represents a polygon with a given number of sides -/
structure Polygon where
  sides : ℕ

/-- Represents a cut operation on a piece of paper -/
inductive Cut
  | straight : Cut

/-- Represents the state of the paper cutting process -/
structure PaperState where
  pieces : ℕ
  polygons : List Polygon

/-- Defines the initial state with a single rectangular piece of paper -/
def initial_state : PaperState :=
  { pieces := 1, polygons := [⟨4⟩] }

/-- Applies a cut to a paper state -/
def apply_cut (state : PaperState) (cut : Cut) : PaperState :=
  { pieces := state.pieces + 1, polygons := state.polygons }

/-- Checks if the goal of at least 100 20-sided polygons is achieved -/
def goal_achieved (state : PaperState) : Prop :=
  (state.polygons.filter (λ p => p.sides = 20)).length ≥ 100

/-- The main theorem stating the minimum number of cuts required -/
theorem min_cuts_for_20_gons : 
  ∃ (n : ℕ), n = 1699 ∧ 
  (∀ (m : ℕ), m < n → 
    ¬∃ (cuts : List Cut), 
      goal_achieved (cuts.foldl apply_cut initial_state)) ∧
  (∃ (cuts : List Cut), 
    cuts.length = n ∧ 
    goal_achieved (cuts.foldl apply_cut initial_state)) :=
sorry

end NUMINAMATH_CALUDE_min_cuts_for_20_gons_l1384_138479


namespace NUMINAMATH_CALUDE_annulus_area_l1384_138407

/-- The area of an annulus formed by two concentric circles, where the radius of the smaller circle
    is 8 units and the radius of the larger circle is twice that of the smaller, is 192π square units. -/
theorem annulus_area (r₁ r₂ : ℝ) (h₁ : r₁ = 8) (h₂ : r₂ = 2 * r₁) :
  π * r₂^2 - π * r₁^2 = 192 * π := by
  sorry

#check annulus_area

end NUMINAMATH_CALUDE_annulus_area_l1384_138407


namespace NUMINAMATH_CALUDE_locomotive_whistle_distance_l1384_138451

/-- The speed of the locomotive in meters per second -/
def locomotive_speed : ℝ := 20

/-- The speed of sound in meters per second -/
def sound_speed : ℝ := 340

/-- The time difference between hearing the whistle and the train's arrival in seconds -/
def time_difference : ℝ := 4

/-- The distance of the locomotive when it started whistling in meters -/
def whistle_distance : ℝ := 85

theorem locomotive_whistle_distance :
  (whistle_distance / locomotive_speed) - time_difference = whistle_distance / sound_speed :=
by sorry

end NUMINAMATH_CALUDE_locomotive_whistle_distance_l1384_138451


namespace NUMINAMATH_CALUDE_complex_product_l1384_138426

theorem complex_product (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 2) 
  (h2 : Complex.abs z₂ = 3) 
  (h3 : 3 * z₁ - 2 * z₂ = 2 - Complex.I) : 
  z₁ * z₂ = -18/5 + 24/5 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_product_l1384_138426


namespace NUMINAMATH_CALUDE_rebecca_checkerboard_black_squares_l1384_138445

/-- Represents a checkerboard with alternating colors -/
structure Checkerboard where
  size : ℕ
  is_black : ℕ → ℕ → Prop

/-- Defines the properties of Rebecca's checkerboard -/
def rebecca_checkerboard : Checkerboard where
  size := 29
  is_black := fun i j => (i + j) % 2 = 0

/-- Counts the number of black squares in a row -/
def black_squares_in_row (c : Checkerboard) (row : ℕ) : ℕ :=
  (c.size + 1) / 2

/-- Counts the total number of black squares on the checkerboard -/
def total_black_squares (c : Checkerboard) : ℕ :=
  c.size * ((c.size + 1) / 2)

/-- Theorem stating that Rebecca's checkerboard has 435 black squares -/
theorem rebecca_checkerboard_black_squares :
  total_black_squares rebecca_checkerboard = 435 := by
  sorry


end NUMINAMATH_CALUDE_rebecca_checkerboard_black_squares_l1384_138445


namespace NUMINAMATH_CALUDE_equation_solution_l1384_138409

theorem equation_solution (m n x : ℝ) (hm : m > 0) (hn : n < 0) :
  (x - m)^2 - (x - n)^2 = (m - n)^2 → x = m :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1384_138409


namespace NUMINAMATH_CALUDE_constant_term_in_expansion_l1384_138461

theorem constant_term_in_expansion (n : ℕ) (h : n > 0) 
  (h_coeff : (n.choose 2) - (n.choose 1) = 44) : 
  let general_term (r : ℕ) := (n.choose r) * (33 - 11 * r) / 2
  ∃ (k : ℕ), k = 4 ∧ general_term (k - 1) = 0 :=
sorry

end NUMINAMATH_CALUDE_constant_term_in_expansion_l1384_138461


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_is_39_l1384_138447

def expression (x : ℝ) : ℝ :=
  2 * (x^2 - 2*x^3 + 2*x) + 4 * (x + 3*x^3 - 2*x^2 + 2*x^5 - x^3) - 7 * (2 + 2*x - 5*x^3 - x^2)

theorem coefficient_of_x_cubed_is_39 :
  (deriv (deriv (deriv expression))) 0 / 6 = 39 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_is_39_l1384_138447


namespace NUMINAMATH_CALUDE_gcd_m_n_l1384_138492

def m : ℕ := 333333333
def n : ℕ := 9999999999

theorem gcd_m_n : Nat.gcd m n = 9 := by
  sorry

end NUMINAMATH_CALUDE_gcd_m_n_l1384_138492


namespace NUMINAMATH_CALUDE_no_consecutive_prime_roots_l1384_138482

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if two numbers are consecutive primes -/
def areConsecutivePrimes (p q : ℕ) : Prop := sorry

/-- The theorem stating that there are no values of k satisfying the conditions -/
theorem no_consecutive_prime_roots :
  ¬ ∃ (k : ℤ) (p q : ℕ), 
    p < q ∧ 
    areConsecutivePrimes p q ∧ 
    p + q = 65 ∧ 
    p * q = k ∧
    ∀ (x : ℤ), x^2 - 65*x + k = 0 ↔ (x = p ∨ x = q) :=
sorry

end NUMINAMATH_CALUDE_no_consecutive_prime_roots_l1384_138482


namespace NUMINAMATH_CALUDE_polynomial_with_no_integer_roots_but_modular_roots_l1384_138431

/-- Definition of the polynomial P(x) = (x³ + 3)(x² + 1)(x² + 2)(x² - 2) -/
def P (x : ℤ) : ℤ := (x^3 + 3) * (x^2 + 1) * (x^2 + 2) * (x^2 - 2)

/-- Theorem stating the existence of a polynomial with the required properties -/
theorem polynomial_with_no_integer_roots_but_modular_roots :
  (∀ x : ℤ, P x ≠ 0) ∧
  (∀ n : ℕ, n > 0 → ∃ x : ℤ, P x % n = 0) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_with_no_integer_roots_but_modular_roots_l1384_138431


namespace NUMINAMATH_CALUDE_min_value_abc_min_value_equals_one_over_nine_to_nine_l1384_138404

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/b + 1/c = 9) : 
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → 1/x + 1/y + 1/z = 9 → 
  a^4 * b^3 * c^2 ≤ x^4 * y^3 * z^2 :=
by sorry

theorem min_value_equals_one_over_nine_to_nine (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/b + 1/c = 9) : 
  a^4 * b^3 * c^2 = 1 / 9^9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_abc_min_value_equals_one_over_nine_to_nine_l1384_138404


namespace NUMINAMATH_CALUDE_fred_marble_count_l1384_138436

/-- The number of blue marbles Tim has -/
def tim_marbles : ℕ := 5

/-- The factor by which Fred has more marbles than Tim -/
def fred_factor : ℕ := 22

/-- The number of blue marbles Fred has -/
def fred_marbles : ℕ := tim_marbles * fred_factor

theorem fred_marble_count : fred_marbles = 110 := by
  sorry

end NUMINAMATH_CALUDE_fred_marble_count_l1384_138436


namespace NUMINAMATH_CALUDE_power_product_equality_l1384_138419

theorem power_product_equality : (-0.25)^2022 * 4^2022 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l1384_138419


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1384_138467

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ (m : ℝ), m = b / a ∧ m = 2) →
  (∃ (x : ℝ), x^2 + (2*x + 10)^2 = a^2 + b^2) →
  a^2 = 5 ∧ b^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1384_138467


namespace NUMINAMATH_CALUDE_clock_strikes_l1384_138446

/-- If a clock strikes three times in 12 seconds, it will strike six times in 30 seconds. -/
theorem clock_strikes (strike_interval : ℝ) : 
  (3 * strike_interval = 12) → (6 * strike_interval = 30) := by
  sorry

end NUMINAMATH_CALUDE_clock_strikes_l1384_138446


namespace NUMINAMATH_CALUDE_original_magazine_cost_l1384_138460

/-- The original cost of a magazine can be determined from the number of magazines, 
    selling price, and total profit. -/
theorem original_magazine_cost 
  (num_magazines : ℕ) 
  (selling_price : ℚ) 
  (total_profit : ℚ) : 
  num_magazines = 10 → 
  selling_price = 7/2 → 
  total_profit = 5 → 
  (num_magazines : ℚ) * selling_price - total_profit = 30 ∧ 
  ((num_magazines : ℚ) * selling_price - total_profit) / num_magazines = 3 :=
by sorry

end NUMINAMATH_CALUDE_original_magazine_cost_l1384_138460


namespace NUMINAMATH_CALUDE_bigger_part_is_thirteen_l1384_138488

theorem bigger_part_is_thirteen (x y : ℝ) (h1 : x + y = 24) (h2 : 7 * x + 5 * y = 146) 
  (h3 : x > 0) (h4 : y > 0) : max x y = 13 := by
  sorry

end NUMINAMATH_CALUDE_bigger_part_is_thirteen_l1384_138488


namespace NUMINAMATH_CALUDE_jake_final_bitcoin_count_l1384_138476

/-- Calculates the final number of bitcoins Jake has after a series of transactions -/
def final_bitcoin_count (initial : ℕ) (first_donation : ℕ) (second_donation : ℕ) : ℕ :=
  let after_first_donation := initial - first_donation
  let after_giving_to_brother := after_first_donation / 2
  let after_tripling := after_giving_to_brother * 3
  after_tripling - second_donation

/-- Theorem stating that Jake ends up with 80 bitcoins -/
theorem jake_final_bitcoin_count :
  final_bitcoin_count 80 20 10 = 80 := by
  sorry

end NUMINAMATH_CALUDE_jake_final_bitcoin_count_l1384_138476


namespace NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l1384_138403

/-- Acme T-Shirt Company's pricing structure -/
def acme_cost (x : ℕ) : ℕ := 50 + 9 * x

/-- Beta T-shirt Company's pricing structure -/
def beta_cost (x : ℕ) : ℕ := 14 * x

/-- The minimum number of shirts for which Acme is cheaper than Beta -/
def min_shirts_for_acme_cheaper : ℕ := 11

theorem acme_cheaper_at_min_shirts :
  acme_cost min_shirts_for_acme_cheaper < beta_cost min_shirts_for_acme_cheaper ∧
  ∀ n : ℕ, n < min_shirts_for_acme_cheaper →
    acme_cost n ≥ beta_cost n :=
by sorry

end NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l1384_138403


namespace NUMINAMATH_CALUDE_inequality_proof_l1384_138457

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) : 
  a / d < b / c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1384_138457


namespace NUMINAMATH_CALUDE_third_grade_students_l1384_138421

theorem third_grade_students (total : ℕ) (male female : ℕ) : 
  total = 41 → 
  male = female + 3 → 
  total = male + female →
  male = 22 := by
sorry

end NUMINAMATH_CALUDE_third_grade_students_l1384_138421


namespace NUMINAMATH_CALUDE_equation_solution_l1384_138498

theorem equation_solution : 
  {x : ℝ | x + 60 / (x - 3) = -12} = {-3, -6} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1384_138498


namespace NUMINAMATH_CALUDE_yonder_license_plates_l1384_138425

/-- The number of possible letters in a license plate position -/
def num_letters : ℕ := 26

/-- The number of possible digits in a license plate position -/
def num_digits : ℕ := 10

/-- The total number of possible license plates in Yonder -/
def total_license_plates : ℕ := num_letters ^ 3 * num_digits ^ 3

theorem yonder_license_plates :
  total_license_plates = 17576000 := by sorry

end NUMINAMATH_CALUDE_yonder_license_plates_l1384_138425


namespace NUMINAMATH_CALUDE_zero_last_to_appear_l1384_138450

/-- Fibonacci sequence modulo 9 -/
def fib_mod_9 : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => (fib_mod_9 (n + 1) + fib_mod_9 n) % 9

/-- Function to check if a digit has appeared in the sequence up to n -/
def digit_appears (d : ℕ) (n : ℕ) : Prop :=
  ∃ k, k ≤ n ∧ fib_mod_9 k = d

/-- Theorem stating that 0 is the last digit to appear -/
theorem zero_last_to_appear :
  ∃ n, digit_appears 0 n ∧
    ∀ d, d < 9 → d ≠ 0 → ∃ k, k < n ∧ digit_appears d k :=
  sorry

end NUMINAMATH_CALUDE_zero_last_to_appear_l1384_138450


namespace NUMINAMATH_CALUDE_profit_in_scientific_notation_l1384_138411

theorem profit_in_scientific_notation :
  (74.5 : ℝ) * 1000000000 = 7.45 * (10 : ℝ)^9 :=
by sorry

end NUMINAMATH_CALUDE_profit_in_scientific_notation_l1384_138411


namespace NUMINAMATH_CALUDE_watermelon_ratio_is_three_to_one_l1384_138483

/-- The ratio of Clay's watermelon size to Michael's watermelon size -/
def watermelon_ratio (michael_weight john_weight : ℚ) : ℚ :=
  (2 * john_weight) / michael_weight

/-- Proof that the watermelon ratio is 3:1 given the specified conditions -/
theorem watermelon_ratio_is_three_to_one :
  watermelon_ratio 8 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_ratio_is_three_to_one_l1384_138483


namespace NUMINAMATH_CALUDE_parallel_planes_false_l1384_138448

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (belongs_to : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the planes and lines
variable (α β : Plane)
variable (m n : Line)

-- State the theorem
theorem parallel_planes_false :
  (¬ (α = β)) →  -- α and β are non-coincident planes
  (¬ (m = n)) →  -- m and n are non-coincident lines
  ¬ (
    (belongs_to m α ∧ belongs_to n α ∧ 
     parallel_line_plane m β ∧ parallel_line_plane n β) → 
    (parallel α β)
  ) := by sorry

end NUMINAMATH_CALUDE_parallel_planes_false_l1384_138448


namespace NUMINAMATH_CALUDE_low_card_value_is_one_l1384_138416

/-- A card type in the high-low game -/
inductive CardType
| High
| Low

/-- The high-low card game -/
structure HighLowGame where
  total_cards : Nat
  high_cards : Nat
  low_cards : Nat
  high_card_value : Nat
  low_card_value : Nat
  target_points : Nat
  target_low_cards : Nat
  ways_to_reach_target : Nat

/-- Conditions for the high-low game -/
def game_conditions (g : HighLowGame) : Prop :=
  g.total_cards = 52 ∧
  g.high_cards = g.low_cards ∧
  g.high_cards + g.low_cards = g.total_cards ∧
  g.high_card_value = 2 ∧
  g.target_points = 5 ∧
  g.target_low_cards = 3 ∧
  g.ways_to_reach_target = 4

/-- Theorem stating that under the given conditions, the low card value must be 1 -/
theorem low_card_value_is_one (g : HighLowGame) :
  game_conditions g → g.low_card_value = 1 := by
  sorry

end NUMINAMATH_CALUDE_low_card_value_is_one_l1384_138416


namespace NUMINAMATH_CALUDE_g_2187_equals_343_l1384_138414

-- Define the properties of function g
def satisfies_property (g : ℕ → ℝ) : Prop :=
  ∀ (x y m : ℕ), x > 0 → y > 0 → m > 0 → x + y = 3^m → g x + g y = m^3

-- Theorem statement
theorem g_2187_equals_343 (g : ℕ → ℝ) (h : satisfies_property g) : g 2187 = 343 := by
  sorry

end NUMINAMATH_CALUDE_g_2187_equals_343_l1384_138414


namespace NUMINAMATH_CALUDE_foundation_digging_l1384_138434

/-- Represents the work rate for digging a foundation -/
def work_rate (men : ℕ) (days : ℝ) : ℝ := men * days

theorem foundation_digging 
  (men_first_half : ℕ) (days_first_half : ℝ) 
  (men_second_half : ℕ) :
  men_first_half = 10 →
  days_first_half = 6 →
  men_second_half = 20 →
  work_rate men_first_half days_first_half = work_rate men_second_half 3 :=
by sorry

end NUMINAMATH_CALUDE_foundation_digging_l1384_138434


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l1384_138475

/-- A geometric sequence with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_first_term
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geom : geometric_sequence a q)
  (h_q_pos : q > 0)
  (h_condition : a 5 * a 7 = 4 * (a 4)^2)
  (h_a2 : a 2 = 1) :
  a 1 = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l1384_138475


namespace NUMINAMATH_CALUDE_geometric_sequence_solution_l1384_138473

theorem geometric_sequence_solution :
  ∃! (x : ℝ), x > 0 ∧ (∃ (r : ℝ), 12 * r = x ∧ x * r = 2/3) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_solution_l1384_138473


namespace NUMINAMATH_CALUDE_transformation_matrix_correct_l1384_138443

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]
def scaling_matrix (s : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![s, 0; 0, s]

/-- The transformation matrix M represents a 90° counter-clockwise rotation followed by a scaling of factor 3 -/
theorem transformation_matrix_correct :
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![0, -3; 3, 0]
  M = scaling_matrix 3 * rotation_matrix := by sorry

end NUMINAMATH_CALUDE_transformation_matrix_correct_l1384_138443


namespace NUMINAMATH_CALUDE_combined_cost_price_l1384_138422

def usa_stock : ℝ := 100
def uk_stock : ℝ := 150
def germany_stock : ℝ := 200

def usa_discount : ℝ := 0.06
def uk_discount : ℝ := 0.10
def germany_discount : ℝ := 0.07

def usa_brokerage : ℝ := 0.015
def uk_brokerage : ℝ := 0.02
def germany_brokerage : ℝ := 0.025

def usa_transaction : ℝ := 5
def uk_transaction : ℝ := 3
def germany_transaction : ℝ := 2

def maintenance_charge : ℝ := 0.005
def taxation_rate : ℝ := 0.15

def usd_to_gbp : ℝ := 0.75
def usd_to_eur : ℝ := 0.85

theorem combined_cost_price :
  let usa_cost := (usa_stock * (1 - usa_discount) * (1 + usa_brokerage) + usa_transaction) * (1 + maintenance_charge)
  let uk_cost := (uk_stock * (1 - uk_discount) * (1 + uk_brokerage) + uk_transaction) * (1 + maintenance_charge) / usd_to_gbp
  let germany_cost := (germany_stock * (1 - germany_discount) * (1 + germany_brokerage) + germany_transaction) * (1 + maintenance_charge) / usd_to_eur
  let total_cost := usa_cost + uk_cost + germany_cost
  total_cost * (1 + taxation_rate) = 594.75 := by sorry

end NUMINAMATH_CALUDE_combined_cost_price_l1384_138422


namespace NUMINAMATH_CALUDE_daily_fine_is_two_l1384_138466

/-- Calculates the daily fine for absence given the total engagement period, daily wage, total amount received, and number of days absent. -/
def calculate_daily_fine (total_days : ℕ) (daily_wage : ℕ) (total_received : ℕ) (days_absent : ℕ) : ℕ :=
  let days_worked := total_days - days_absent
  let total_earned := days_worked * daily_wage
  let total_fine := total_earned - total_received
  total_fine / days_absent

/-- Theorem stating that the daily fine is 2 given the problem conditions. -/
theorem daily_fine_is_two :
  calculate_daily_fine 30 10 216 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_daily_fine_is_two_l1384_138466


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_coefficient_l1384_138456

theorem quadratic_roots_imply_coefficient (a : ℝ) : 
  (((1 + Real.sqrt 3) / 2 : ℝ) * ((1 - Real.sqrt 3) / 2 : ℝ) = 1 / (2 * a)) ∧
  (((1 + Real.sqrt 3) / 2 : ℝ) + ((1 - Real.sqrt 3) / 2 : ℝ) = 1 / a) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_coefficient_l1384_138456


namespace NUMINAMATH_CALUDE_no_special_sequence_exists_l1384_138454

theorem no_special_sequence_exists : ¬ ∃ (a : ℕ → ℕ),
  (∀ n : ℕ, a n < a (n + 1)) ∧
  (∃ N : ℕ, ∀ m : ℕ, m ≥ N →
    ∃! (i j : ℕ), m = a i + a j) :=
by sorry

end NUMINAMATH_CALUDE_no_special_sequence_exists_l1384_138454


namespace NUMINAMATH_CALUDE_root_implies_a_value_l1384_138478

theorem root_implies_a_value (a : ℝ) : 
  ((3 : ℝ) = 3 ∧ (a - 2) / 3 - 1 / (3 - 2) = 0) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_a_value_l1384_138478


namespace NUMINAMATH_CALUDE_not_prime_1000000027_l1384_138487

theorem not_prime_1000000027 : ¬ Nat.Prime 1000000027 := by
  sorry

end NUMINAMATH_CALUDE_not_prime_1000000027_l1384_138487


namespace NUMINAMATH_CALUDE_no_valid_seating_l1384_138435

/-- A seating arrangement of deputies around a circular table. -/
structure Seating :=
  (deputies : Fin 47 → Fin 12)

/-- The property that any 15 consecutive deputies include all 12 regions. -/
def hasAllRegionsIn15 (s : Seating) : Prop :=
  ∀ start : Fin 47, ∃ (f : Fin 12 → Fin 15), ∀ r : Fin 12,
    ∃ i : Fin 15, s.deputies ((start + i) % 47) = r

/-- Theorem stating that no valid seating arrangement exists. -/
theorem no_valid_seating : ¬ ∃ s : Seating, hasAllRegionsIn15 s := by
  sorry

end NUMINAMATH_CALUDE_no_valid_seating_l1384_138435


namespace NUMINAMATH_CALUDE_tree_height_after_three_good_years_l1384_138480

/-- Represents the growth factor of a tree in different conditions -/
inductive GrowthCondition
| Good
| Bad

/-- Calculates the height of a tree after a given number of years -/
def treeHeight (initialHeight : ℝ) (years : ℕ) (conditions : List GrowthCondition) : ℝ :=
  match years, conditions with
  | 0, _ => initialHeight
  | n+1, [] => initialHeight  -- Default to initial height if no conditions are specified
  | n+1, c::cs => 
    let newHeight := 
      match c with
      | GrowthCondition.Good => 3 * initialHeight
      | GrowthCondition.Bad => 2 * initialHeight
    treeHeight newHeight n cs

/-- Theorem stating the height of the tree after 3 years of good growth -/
theorem tree_height_after_three_good_years :
  let initialHeight : ℝ := treeHeight 1458 3 [GrowthCondition.Bad, GrowthCondition.Bad, GrowthCondition.Bad]
  treeHeight initialHeight 3 [GrowthCondition.Good, GrowthCondition.Good, GrowthCondition.Good] = 1458 :=
by sorry

#eval treeHeight 1458 3 [GrowthCondition.Bad, GrowthCondition.Bad, GrowthCondition.Bad]

end NUMINAMATH_CALUDE_tree_height_after_three_good_years_l1384_138480


namespace NUMINAMATH_CALUDE_square_last_two_digits_averages_l1384_138442

def last_two_digits (n : ℕ) : ℕ := n % 100

def valid_pair (a b : ℕ) : Prop :=
  a ≠ b ∧ 0 < a ∧ a < 50 ∧ 0 < b ∧ b < 50 ∧ last_two_digits (a^2) = last_two_digits (b^2)

def average (a b : ℕ) : ℚ := (a + b : ℚ) / 2

theorem square_last_two_digits_averages :
  {x : ℚ | ∃ a b : ℕ, valid_pair a b ∧ average a b = x} = {10, 15, 20, 25, 30, 35, 40} := by sorry

end NUMINAMATH_CALUDE_square_last_two_digits_averages_l1384_138442


namespace NUMINAMATH_CALUDE_three_digit_integers_with_remainders_l1384_138412

theorem three_digit_integers_with_remainders : 
  ∃! (S : Finset ℕ), 
    (∀ n ∈ S, 100 ≤ n ∧ n < 1000 ∧ 
              n % 7 = 3 ∧ 
              n % 10 = 6 ∧ 
              n % 12 = 9) ∧
    S.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_integers_with_remainders_l1384_138412


namespace NUMINAMATH_CALUDE_integer_part_sqrt_seven_l1384_138464

theorem integer_part_sqrt_seven : ⌊Real.sqrt 7⌋ = 2 := by sorry

end NUMINAMATH_CALUDE_integer_part_sqrt_seven_l1384_138464


namespace NUMINAMATH_CALUDE_books_read_first_week_l1384_138402

/-- The number of books read in the first week of a 7-week reading plan -/
def books_first_week (total_books : ℕ) (second_week : ℕ) (later_weeks : ℕ) : ℕ :=
  total_books - second_week - (later_weeks * 5)

theorem books_read_first_week :
  books_first_week 54 3 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_books_read_first_week_l1384_138402


namespace NUMINAMATH_CALUDE_positive_expressions_l1384_138453

theorem positive_expressions (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 3) : 
  0 < b + b^2 ∧ 0 < b + 3*b^2 := by
  sorry

end NUMINAMATH_CALUDE_positive_expressions_l1384_138453


namespace NUMINAMATH_CALUDE_prob_rain_weekend_l1384_138490

/-- Probability of rain on Saturday -/
def prob_rain_saturday : ℝ := 0.6

/-- Probability of rain on Sunday given it rained on Saturday -/
def prob_rain_sunday_given_rain_saturday : ℝ := 0.7

/-- Probability of rain on Sunday given it didn't rain on Saturday -/
def prob_rain_sunday_given_no_rain_saturday : ℝ := 0.4

/-- Theorem: The probability of rain over the weekend (at least one day) is 76% -/
theorem prob_rain_weekend : 
  1 - (1 - prob_rain_saturday) * (1 - prob_rain_sunday_given_no_rain_saturday) = 0.76 := by
  sorry

end NUMINAMATH_CALUDE_prob_rain_weekend_l1384_138490


namespace NUMINAMATH_CALUDE_division_remainder_composition_l1384_138440

theorem division_remainder_composition (P D Q R D' Q' R' : ℕ) 
  (h1 : P = Q * D + R) 
  (h2 : Q = D' * Q' + R') : 
  ∃ k : ℕ, P = (D * D') * Q' + (R + R' * D) + k * (D * D') := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_composition_l1384_138440


namespace NUMINAMATH_CALUDE_smallest_fitting_polygon_l1384_138437

/-- A regular polygon with n sides that can fit perfectly when rotated by 40° or 60° -/
def FittingPolygon (n : ℕ) : Prop :=
  n > 0 ∧ (40 * n) % 360 = 0 ∧ (60 * n) % 360 = 0

/-- The smallest number of sides for a fitting polygon is 18 -/
theorem smallest_fitting_polygon : ∃ (n : ℕ), FittingPolygon n ∧ ∀ m, FittingPolygon m → n ≤ m :=
  sorry

end NUMINAMATH_CALUDE_smallest_fitting_polygon_l1384_138437


namespace NUMINAMATH_CALUDE_cos_A_value_projection_BA_on_BC_l1384_138484

noncomputable section

variables (A B C : ℝ) (a b c : ℝ)

-- Define the triangle ABC
def triangle_ABC : Prop :=
  2 * (Real.cos ((A - B) / 2))^2 * Real.cos B - Real.sin (A - B) * Real.sin B + Real.cos (A + C) = -3/5

-- Define the side lengths
def side_lengths : Prop :=
  a = 4 * Real.sqrt 2 ∧ b = 5

-- Theorem for part 1
theorem cos_A_value (h : triangle_ABC A B C) : Real.cos A = -3/5 := by sorry

-- Theorem for part 2
theorem projection_BA_on_BC (h1 : triangle_ABC A B C) (h2 : side_lengths a b) :
  ∃ (proj : ℝ), proj = Real.sqrt 2 / 2 ∧ proj = c * Real.cos B := by sorry

end

end NUMINAMATH_CALUDE_cos_A_value_projection_BA_on_BC_l1384_138484


namespace NUMINAMATH_CALUDE_gcd_71_19_l1384_138405

theorem gcd_71_19 : Nat.gcd 71 19 = 1 := by sorry

end NUMINAMATH_CALUDE_gcd_71_19_l1384_138405


namespace NUMINAMATH_CALUDE_unique_non_range_value_l1384_138430

def f (k : ℚ) (x : ℚ) : ℚ := (2 * x + k) / (3 * x + 4)

theorem unique_non_range_value (k : ℚ) :
  (f k 5 = 5) →
  (f k 100 = 100) →
  (∀ x ≠ (-4/3), f k (f k x) = x) →
  ∃! y, ∀ x, f k x ≠ y ∧ y = (-8/13) :=
sorry

end NUMINAMATH_CALUDE_unique_non_range_value_l1384_138430


namespace NUMINAMATH_CALUDE_range_of_a_given_p_necessary_not_sufficient_for_q_l1384_138494

theorem range_of_a_given_p_necessary_not_sufficient_for_q :
  ∀ a : ℝ,
  (∀ x : ℝ, x^2 ≤ 5*x - 4 → x^2 - (a+2)*x + 2*a ≤ 0) ∧
  (∃ x : ℝ, x^2 - (a+2)*x + 2*a ≤ 0 ∧ x^2 > 5*x - 4) →
  1 ≤ a ∧ a ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_given_p_necessary_not_sufficient_for_q_l1384_138494


namespace NUMINAMATH_CALUDE_selection_schemes_count_l1384_138471

/-- The number of people to choose from -/
def total_people : ℕ := 6

/-- The number of pavilions to visit -/
def pavilions : ℕ := 4

/-- The number of people who cannot visit a specific pavilion -/
def restricted_people : ℕ := 2

/-- Calculates the number of ways to select people for pavilions with restrictions -/
def selection_schemes : ℕ :=
  Nat.descFactorial total_people pavilions - 
  restricted_people * Nat.descFactorial (total_people - 1) (pavilions - 1)

/-- The theorem stating the number of selection schemes -/
theorem selection_schemes_count : selection_schemes = 240 := by sorry

end NUMINAMATH_CALUDE_selection_schemes_count_l1384_138471


namespace NUMINAMATH_CALUDE_max_value_7b_5c_l1384_138470

/-- The function f(x) = ax^2 + bx + c -/
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The theorem stating the maximum value of 7b+5c given the conditions -/
theorem max_value_7b_5c (a b c : ℝ) : 
  (∃ a' ∈ Set.Icc 1 2, ∀ x ∈ Set.Icc 1 2, f a' b c x ≤ 1) →
  (∀ y : ℝ, 7 * b + 5 * c ≤ y) → y = -6 := by
  sorry

end NUMINAMATH_CALUDE_max_value_7b_5c_l1384_138470


namespace NUMINAMATH_CALUDE_triangle_property_l1384_138477

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if (a+c)/b = cos(C) + √3*sin(C), then B = 60° and when b = 2, the max area is √3 -/
theorem triangle_property (a b c : ℝ) (A B C : Real) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  (a + c) / b = Real.cos C + Real.sqrt 3 * Real.sin C →
  B = π / 3 ∧ 
  (b = 2 → ∃ (area : ℝ), area ≤ Real.sqrt 3 ∧ 
    ∀ (other_area : ℝ), (∃ (a' c' : ℝ), a' > 0 ∧ c' > 0 ∧ 
      other_area = 1/2 * a' * 2 * Real.sin (π/3)) → other_area ≤ area) := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l1384_138477


namespace NUMINAMATH_CALUDE_marcia_pants_count_l1384_138415

/-- Represents the number of items in Marcia's wardrobe -/
structure Wardrobe where
  skirts : Nat
  blouses : Nat
  pants : Nat

/-- Represents the prices of items and the total budget -/
structure Prices where
  skirt_price : ℕ
  blouse_price : ℕ
  pant_price : ℕ
  total_budget : ℕ

/-- Calculates the cost of pants with the sale applied -/
def pants_cost (n : ℕ) (price : ℕ) : ℕ :=
  if n % 2 = 0 then
    n / 2 * price + n / 2 * (price / 2)
  else
    (n / 2 + 1) * price + (n / 2) * (price / 2)

/-- Theorem stating that Marcia needs to add 2 pairs of pants -/
theorem marcia_pants_count (w : Wardrobe) (p : Prices) : w.pants = 2 :=
  by
    have h1 : w.skirts = 3 := by sorry
    have h2 : w.blouses = 5 := by sorry
    have h3 : p.skirt_price = 20 := by sorry
    have h4 : p.blouse_price = 15 := by sorry
    have h5 : p.pant_price = 30 := by sorry
    have h6 : p.total_budget = 180 := by sorry
    
    have skirt_cost : ℕ := w.skirts * p.skirt_price
    have blouse_cost : ℕ := w.blouses * p.blouse_price
    have remaining_budget : ℕ := p.total_budget - (skirt_cost + blouse_cost)
    
    have pants_fit_budget : pants_cost w.pants p.pant_price = remaining_budget := by sorry
    
    sorry -- Complete the proof here

end NUMINAMATH_CALUDE_marcia_pants_count_l1384_138415


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_30_factorial_l1384_138491

theorem distinct_prime_factors_of_30_factorial :
  (Finset.filter Nat.Prime (Finset.range 31)).card = 10 := by sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_30_factorial_l1384_138491


namespace NUMINAMATH_CALUDE_number_solution_l1384_138469

theorem number_solution : ∃ x : ℝ, 3 * x - 5 = 40 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_number_solution_l1384_138469


namespace NUMINAMATH_CALUDE_problem_statement_l1384_138406

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 1) :
  (ab ≤ 1/8) ∧ (2/(a+1) + 1/b ≥ 3 + 2*Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1384_138406
