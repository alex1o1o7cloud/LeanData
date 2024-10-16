import Mathlib

namespace NUMINAMATH_CALUDE_line_m_plus_b_l1709_170918

/-- A line passing through three given points has m + b = -1 -/
theorem line_m_plus_b (m b : ℝ) : 
  (3 = m * 3 + b) →  -- Line passes through (3, 3)
  (-1 = m * 1 + b) →  -- Line passes through (1, -1)
  (1 = m * 2 + b) →  -- Line passes through (2, 1)
  m + b = -1 := by
sorry

end NUMINAMATH_CALUDE_line_m_plus_b_l1709_170918


namespace NUMINAMATH_CALUDE_election_results_l1709_170953

theorem election_results (total_votes : ℕ) 
  (votes_A votes_B votes_C : ℕ) : 
  votes_A = (35 : ℕ) * total_votes / 100 →
  votes_B = votes_A + 1800 →
  votes_C = votes_A / 2 →
  total_votes = votes_A + votes_B + votes_C →
  total_votes = 14400 ∧
  (votes_A : ℚ) / total_votes = 35 / 100 ∧
  (votes_B : ℚ) / total_votes = 475 / 1000 ∧
  (votes_C : ℚ) / total_votes = 175 / 1000 :=
by
  sorry

#check election_results

end NUMINAMATH_CALUDE_election_results_l1709_170953


namespace NUMINAMATH_CALUDE_sector_max_area_l1709_170997

/-- Given a sector with fixed perimeter P, prove that the maximum area is P^2/16
    and this maximum is achieved when the radius is P/4. -/
theorem sector_max_area (P : ℝ) (h : P > 0) :
  let max_area := P^2 / 16
  let max_radius := P / 4
  ∀ R l, R > 0 → l > 0 → 2 * R + l = P →
    (1/2 * R * l ≤ max_area) ∧
    (1/2 * max_radius * (P - 2 * max_radius) = max_area) :=
by sorry


end NUMINAMATH_CALUDE_sector_max_area_l1709_170997


namespace NUMINAMATH_CALUDE_quadratic_sum_l1709_170979

/-- Given a quadratic function f(x) = 4x^2 - 40x + 100, 
    there exist constants a, b, and c such that 
    f(x) = a(x+b)^2 + c for all x, and a + b + c = -1 -/
theorem quadratic_sum (x : ℝ) : 
  ∃ (a b c : ℝ), (∀ x, 4*x^2 - 40*x + 100 = a*(x+b)^2 + c) ∧ (a + b + c = -1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_l1709_170979


namespace NUMINAMATH_CALUDE_i_times_one_minus_i_squared_eq_two_l1709_170948

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- Theorem stating that i(1-i)² = 2 -/
theorem i_times_one_minus_i_squared_eq_two : i * (1 - i)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_i_times_one_minus_i_squared_eq_two_l1709_170948


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1709_170902

theorem fraction_to_decimal (h : 343 = 7^3) : 7 / 343 = 0.056 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1709_170902


namespace NUMINAMATH_CALUDE_toaster_tax_rate_l1709_170978

/-- Calculates the mandatory state tax rate for a toaster purchase. -/
theorem toaster_tax_rate (msrp : ℝ) (total_paid : ℝ) (insurance_rate : ℝ) : 
  msrp = 30 →
  total_paid = 54 →
  insurance_rate = 0.2 →
  (total_paid - msrp * (1 + insurance_rate)) / (msrp * (1 + insurance_rate)) = 0.5 := by
  sorry

#check toaster_tax_rate

end NUMINAMATH_CALUDE_toaster_tax_rate_l1709_170978


namespace NUMINAMATH_CALUDE_father_age_triple_marika_age_2014_l1709_170975

/-- Represents a person with their birth year -/
structure Person where
  birthYear : ℕ

/-- Marika, born in 1996 -/
def marika : Person := ⟨1996⟩

/-- Marika's father, born in 1961 -/
def father : Person := ⟨1961⟩

/-- The year when Marika was 10 years old -/
def baseYear : ℕ := 2006

/-- Calculates a person's age in a given year -/
def age (p : Person) (year : ℕ) : ℕ :=
  year - p.birthYear

/-- Theorem stating that 2014 is the first year when the father's age is exactly three times Marika's age -/
theorem father_age_triple_marika_age_2014 :
  (∀ y : ℕ, y < 2014 → y ≥ baseYear → age father y ≠ 3 * age marika y) ∧
  age father 2014 = 3 * age marika 2014 :=
sorry

end NUMINAMATH_CALUDE_father_age_triple_marika_age_2014_l1709_170975


namespace NUMINAMATH_CALUDE_tea_containers_theorem_l1709_170994

/-- Given a total amount of tea in gallons, the number of containers Geraldo drank,
    and the amount of tea Geraldo consumed in pints, calculate the total number of
    containers filled with tea. -/
def totalContainers (totalTea : ℚ) (containersDrunk : ℚ) (teaDrunk : ℚ) : ℚ :=
  (totalTea * 8) / (teaDrunk / containersDrunk)

/-- Prove that given 20 gallons of tea, where 3.5 containers contain 7 pints,
    the total number of containers filled is 80. -/
theorem tea_containers_theorem :
  totalContainers 20 (7/2) 7 = 80 := by
  sorry

end NUMINAMATH_CALUDE_tea_containers_theorem_l1709_170994


namespace NUMINAMATH_CALUDE_solve_video_game_problem_l1709_170987

def video_game_problem (initial_players : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : Prop :=
  let remaining_players := total_lives / lives_per_player
  let players_quit := initial_players - remaining_players
  players_quit = 5

theorem solve_video_game_problem :
  video_game_problem 11 5 30 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_video_game_problem_l1709_170987


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l1709_170958

theorem inscribed_cube_volume (large_cube_side : ℝ) (sphere_diameter : ℝ) 
  (small_cube_diagonal : ℝ) (small_cube_side : ℝ) (small_cube_volume : ℝ) :
  large_cube_side = 12 →
  sphere_diameter = large_cube_side →
  small_cube_diagonal = sphere_diameter →
  small_cube_diagonal = small_cube_side * Real.sqrt 3 →
  small_cube_side = 12 / Real.sqrt 3 →
  small_cube_volume = small_cube_side ^ 3 →
  small_cube_volume = 192 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l1709_170958


namespace NUMINAMATH_CALUDE_bella_stamp_difference_l1709_170976

/-- Calculates the difference between truck stamps and rose stamps -/
def stamp_difference (snowflake : ℕ) (truck_surplus : ℕ) (total : ℕ) : ℕ :=
  let truck := snowflake + truck_surplus
  let rose := total - (snowflake + truck)
  truck - rose

/-- Proves that the difference between truck stamps and rose stamps is 13 -/
theorem bella_stamp_difference :
  stamp_difference 11 9 38 = 13 := by
  sorry

end NUMINAMATH_CALUDE_bella_stamp_difference_l1709_170976


namespace NUMINAMATH_CALUDE_exists_quadratic_with_two_n_roots_l1709_170925

-- Define a quadratic polynomial
def QuadraticPolynomial : Type := ℝ → ℝ

-- Define the property of having 2n distinct real roots for n-fold composition
def HasTwoNRoots (f : QuadraticPolynomial) : Prop :=
  ∀ n : ℕ, ∃! (roots : Finset ℝ), (roots.card = 2 * n) ∧ 
    (∀ x ∈ roots, (f^[n]) x = 0) ∧
    (∀ x : ℝ, (f^[n]) x = 0 → x ∈ roots)

-- The theorem to be proved
theorem exists_quadratic_with_two_n_roots :
  ∃ f : QuadraticPolynomial, HasTwoNRoots f :=
sorry


end NUMINAMATH_CALUDE_exists_quadratic_with_two_n_roots_l1709_170925


namespace NUMINAMATH_CALUDE_g_value_at_4_l1709_170957

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^3 - 2*x + 3

-- Define the properties of g
def g_properties (g : ℝ → ℝ) : Prop :=
  (∃ a b c d : ℝ, ∀ x, g x = a*x^3 + b*x^2 + c*x + d) ∧  -- g is a cubic polynomial
  (g 0 = 3) ∧  -- g(0) = 3
  (∀ r : ℝ, f r = 0 → ∃ s : ℝ, g s = 0 ∧ s = r^2)  -- roots of g are squares of roots of f

-- State the theorem
theorem g_value_at_4 (g : ℝ → ℝ) (h : g_properties g) : g 4 = -75 := by
  sorry

end NUMINAMATH_CALUDE_g_value_at_4_l1709_170957


namespace NUMINAMATH_CALUDE_range_of_m_l1709_170933

/-- Set A is defined as the set of real numbers x where -3 ≤ x ≤ 4 -/
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 4}

/-- Set B is defined as the set of real numbers x where 1 < x < m, and m > 1 -/
def B (m : ℝ) : Set ℝ := {x : ℝ | 1 < x ∧ x < m}

/-- The theorem states that if B is a subset of A and m > 1, then 1 < m ≤ 4 -/
theorem range_of_m (m : ℝ) (h1 : B m ⊆ A) (h2 : 1 < m) : 1 < m ∧ m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1709_170933


namespace NUMINAMATH_CALUDE_binary_1001_equals_9_l1709_170922

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1001_equals_9 :
  binary_to_decimal [true, false, false, true] = 9 := by
  sorry

end NUMINAMATH_CALUDE_binary_1001_equals_9_l1709_170922


namespace NUMINAMATH_CALUDE_eight_liter_solution_exists_l1709_170964

/-- Represents the state of the buckets -/
structure BucketState :=
  (bucket10 : ℕ)
  (bucket6 : ℕ)

/-- Represents a valid operation on the buckets -/
inductive BucketOperation
  | FillFrom10To6
  | FillFrom6To10
  | Empty10
  | Empty6
  | Fill10
  | Fill6

/-- Applies a bucket operation to a given state -/
def applyOperation (state : BucketState) (op : BucketOperation) : BucketState :=
  match op with
  | BucketOperation.FillFrom10To6 => 
      let transfer := min state.bucket10 (6 - state.bucket6)
      ⟨state.bucket10 - transfer, state.bucket6 + transfer⟩
  | BucketOperation.FillFrom6To10 => 
      let transfer := min state.bucket6 (10 - state.bucket10)
      ⟨state.bucket10 + transfer, state.bucket6 - transfer⟩
  | BucketOperation.Empty10 => ⟨0, state.bucket6⟩
  | BucketOperation.Empty6 => ⟨state.bucket10, 0⟩
  | BucketOperation.Fill10 => ⟨10, state.bucket6⟩
  | BucketOperation.Fill6 => ⟨state.bucket10, 6⟩

/-- Checks if the given sequence of operations results in 8 liters in one bucket -/
def checkSolution (ops : List BucketOperation) : Bool :=
  let finalState := ops.foldl applyOperation ⟨0, 0⟩
  finalState.bucket10 = 8 ∨ finalState.bucket6 = 8

/-- Theorem: There exists a sequence of operations that results in 8 liters in one bucket -/
theorem eight_liter_solution_exists : ∃ (ops : List BucketOperation), checkSolution ops := by
  sorry


end NUMINAMATH_CALUDE_eight_liter_solution_exists_l1709_170964


namespace NUMINAMATH_CALUDE_distance_C2_C3_eq_sqrt_10m_l1709_170989

/-- Right triangle ABC with given side lengths -/
structure RightTriangleABC where
  AB : ℝ
  AC : ℝ
  BC : ℝ
  right_angle : AB ^ 2 + AC ^ 2 = BC ^ 2
  AB_eq : AB = 80
  AC_eq : AC = 150
  BC_eq : BC = 170

/-- Inscribed circle C1 of triangle ABC -/
def C1 (t : RightTriangleABC) : Circle := sorry

/-- Line DE perpendicular to AC and tangent to C1 -/
def DE (t : RightTriangleABC) : Line := sorry

/-- Line FG perpendicular to AB and tangent to C1 -/
def FG (t : RightTriangleABC) : Line := sorry

/-- Inscribed circle C2 of triangle BDE -/
def C2 (t : RightTriangleABC) : Circle := sorry

/-- Inscribed circle C3 of triangle CFG -/
def C3 (t : RightTriangleABC) : Circle := sorry

/-- The distance between the centers of C2 and C3 -/
def distance_C2_C3 (t : RightTriangleABC) : ℝ := sorry

theorem distance_C2_C3_eq_sqrt_10m (t : RightTriangleABC) :
  distance_C2_C3 t = Real.sqrt (10 * 1057.6) := by sorry

end NUMINAMATH_CALUDE_distance_C2_C3_eq_sqrt_10m_l1709_170989


namespace NUMINAMATH_CALUDE_prob_both_counterfeit_value_l1709_170926

/-- Represents the total number of banknotes --/
def total_notes : ℕ := 20

/-- Represents the number of counterfeit notes --/
def counterfeit_notes : ℕ := 5

/-- Represents the number of notes drawn --/
def drawn_notes : ℕ := 2

/-- Calculates the probability that both drawn notes are counterfeit given that at least one is counterfeit --/
def prob_both_counterfeit : ℚ :=
  (Nat.choose counterfeit_notes drawn_notes) / 
  (Nat.choose counterfeit_notes drawn_notes + 
   Nat.choose counterfeit_notes 1 * Nat.choose (total_notes - counterfeit_notes) 1)

theorem prob_both_counterfeit_value : 
  prob_both_counterfeit = 2 / 17 := by sorry

end NUMINAMATH_CALUDE_prob_both_counterfeit_value_l1709_170926


namespace NUMINAMATH_CALUDE_circle_radius_l1709_170977

/-- A circle with center (0, k) where k < -6 is tangent to y = x, y = -x, and y = -6.
    Its radius is 6√2. -/
theorem circle_radius (k : ℝ) (h : k < -6) :
  let center := (0 : ℝ × ℝ)
  let radius := Real.sqrt 2 * 6
  (∀ p : ℝ × ℝ, (p.1 = p.2 ∨ p.1 = -p.2 ∨ p.2 = -6) →
    ‖p - center‖ = radius) →
  radius = Real.sqrt 2 * 6 :=
by sorry


end NUMINAMATH_CALUDE_circle_radius_l1709_170977


namespace NUMINAMATH_CALUDE_negation_equivalence_l1709_170968

theorem negation_equivalence :
  (¬ ∃ x : ℝ, 5^x + Real.sin x ≤ 0) ↔ (∀ x : ℝ, 5^x + Real.sin x > 0) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1709_170968


namespace NUMINAMATH_CALUDE_min_students_four_correct_is_eight_l1709_170940

-- Define the total number of students
def total_students : ℕ := 100

-- Define the number of students who performed each spell correctly
def spell1_correct : ℕ := 95
def spell2_correct : ℕ := 75
def spell3_correct : ℕ := 97
def spell4_correct : ℕ := 95
def spell5_correct : ℕ := 96

-- Define the function to calculate the minimum number of students who performed exactly 4 out of 5 spells correctly
def min_students_four_correct : ℕ :=
  total_students - spell2_correct - (total_students - spell1_correct) - (total_students - spell3_correct) - (total_students - spell4_correct) - (total_students - spell5_correct)

-- Theorem statement
theorem min_students_four_correct_is_eight :
  min_students_four_correct = 8 :=
sorry

end NUMINAMATH_CALUDE_min_students_four_correct_is_eight_l1709_170940


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l1709_170930

theorem fraction_product_simplification :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l1709_170930


namespace NUMINAMATH_CALUDE_rotation_270_degrees_l1709_170995

theorem rotation_270_degrees (z : ℂ) : z = -8 - 4*I → z * (-I) = -4 + 8*I := by
  sorry

end NUMINAMATH_CALUDE_rotation_270_degrees_l1709_170995


namespace NUMINAMATH_CALUDE_stock_shares_calculation_l1709_170949

/-- Represents the number of shares for each stock --/
structure StockShares where
  v : ℕ
  w : ℕ
  x : ℕ
  y : ℕ
  z : ℕ

/-- Calculates the range of shares --/
def calculateRange (shares : StockShares) : ℕ :=
  max shares.v (max shares.w (max shares.x (max shares.y shares.z))) -
  min shares.v (min shares.w (min shares.x (min shares.y shares.z)))

/-- The main theorem to prove --/
theorem stock_shares_calculation (initial : StockShares) (y : ℕ) :
  initial.v = 68 →
  initial.w = 112 →
  initial.x = 56 →
  initial.z = 45 →
  initial.y = y →
  let final : StockShares := {
    v := initial.v,
    w := initial.w,
    x := initial.x - 20,
    y := initial.y + 23,
    z := initial.z
  }
  calculateRange final - calculateRange initial = 14 →
  y = 50 := by
  sorry

#check stock_shares_calculation

end NUMINAMATH_CALUDE_stock_shares_calculation_l1709_170949


namespace NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_problem_solution_l1709_170941

theorem least_subtrahend_for_divisibility (n : Nat) (d : Nat) (h : d > 0) :
  ∃ (k : Nat), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : Nat), m < k → (n - m) % d ≠ 0 :=
by sorry

theorem problem_solution : 
  ∃ (k : Nat), k < 37 ∧ (1234567 - k) % 37 = 0 ∧ ∀ (m : Nat), m < k → (1234567 - m) % 37 ≠ 0 ∧ k = 13 :=
by sorry

end NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_problem_solution_l1709_170941


namespace NUMINAMATH_CALUDE_slower_train_speed_l1709_170959

/-- Proves the speed of a slower train given specific conditions --/
theorem slower_train_speed
  (train_length : ℝ)
  (faster_speed : ℝ)
  (passing_time : ℝ)
  (h1 : train_length = 60)
  (h2 : faster_speed = 48)
  (h3 : passing_time = 36)
  : ∃ (slower_speed : ℝ),
    slower_speed = 36 ∧
    (faster_speed - slower_speed) * (5 / 18) * passing_time = 2 * train_length :=
by
  sorry

#check slower_train_speed

end NUMINAMATH_CALUDE_slower_train_speed_l1709_170959


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_seven_l1709_170914

theorem floor_ceiling_sum_seven (x : ℝ) : 
  (⌊x⌋ : ℝ) + ⌈x⌉ = 7 ↔ 3 < x ∧ x < 4 :=
sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_seven_l1709_170914


namespace NUMINAMATH_CALUDE_complex_simplification_l1709_170915

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The property of the imaginary unit -/
axiom i_squared : i * i = -1

/-- The theorem to prove -/
theorem complex_simplification :
  3 * (2 - 2 * i) + 2 * i * (3 + i) = (4 : ℂ) := by sorry

end NUMINAMATH_CALUDE_complex_simplification_l1709_170915


namespace NUMINAMATH_CALUDE_one_third_of_270_l1709_170934

theorem one_third_of_270 : (1 / 3 : ℚ) * 270 = 90 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_270_l1709_170934


namespace NUMINAMATH_CALUDE_series_sum_equals_35_over_13_l1709_170928

/-- Definition of the sequence G_n -/
def G : ℕ → ℚ
  | 0 => 1
  | 1 => 2
  | (n + 2) => G (n + 1) + 2 * G n

/-- The sum of the series -/
noncomputable def seriesSum : ℚ := ∑' n, G n / 5^n

/-- Theorem stating that the sum of the series equals 35/13 -/
theorem series_sum_equals_35_over_13 : seriesSum = 35/13 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_35_over_13_l1709_170928


namespace NUMINAMATH_CALUDE_decompose_50900300_l1709_170904

theorem decompose_50900300 :
  ∃ (ten_thousands ones : ℕ),
    50900300 = ten_thousands * 10000 + ones ∧
    ten_thousands = 5090 ∧
    ones = 300 := by
  sorry

end NUMINAMATH_CALUDE_decompose_50900300_l1709_170904


namespace NUMINAMATH_CALUDE_range_of_a_l1709_170905

-- Define the function f
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x =>
  if x > 1 then Real.exp x - a * x^2 + x - 1 else sorry

-- State the theorem
theorem range_of_a :
  (∀ x, f a (-x) = -(f a x)) →  -- f is odd
  (∀ m : ℝ, m ≠ 0 → f a (1/m) * f a m = 1) →  -- property for non-zero m
  (∀ x, x > 1 → f a x = Real.exp x - a * x^2 + x - 1) →  -- definition for x > 1
  (∀ y : ℝ, ∃ x, f a x = y) →  -- range of f is R
  (∀ x, (x - 2) * Real.exp x - x + 4 > 0) →  -- given inequality
  a ∈ Set.Icc (Real.exp 1 - 1) ((Real.exp 2 + 1) / 4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1709_170905


namespace NUMINAMATH_CALUDE_square_sum_representation_l1709_170988

theorem square_sum_representation (x y : ℕ) (h : x ≠ y) :
  ∃ u v : ℕ, x^2 + x*y + y^2 = u^2 + 3*v^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_representation_l1709_170988


namespace NUMINAMATH_CALUDE_sqrt_a_minus_one_is_rational_square_l1709_170956

theorem sqrt_a_minus_one_is_rational_square (a b : ℚ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : a^3 + 4*a^2*b = 4*a^2 + b^4) :
  ∃ q : ℚ, Real.sqrt a - 1 = q^2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_a_minus_one_is_rational_square_l1709_170956


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_positive_l1709_170973

theorem sum_of_reciprocals_positive (a b c d : ℝ) 
  (ha : |a| > 1) (hb : |b| > 1) (hc : |c| > 1) (hd : |d| > 1)
  (h_eq : a * b * c + a * b * d + a * c * d + b * c * d + a + b + c + d = 0) :
  1 / (a - 1) + 1 / (b - 1) + 1 / (c - 1) + 1 / (d - 1) > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_positive_l1709_170973


namespace NUMINAMATH_CALUDE_sum_of_integers_l1709_170924

theorem sum_of_integers : 7 + (-19) + 13 + (-31) = -30 := by sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1709_170924


namespace NUMINAMATH_CALUDE_elderly_sample_size_l1709_170980

/-- Given a population with elderly people, prove the number of elderly to be sampled -/
theorem elderly_sample_size
  (total_population : ℕ)
  (elderly_population : ℕ)
  (sample_size : ℕ)
  (h1 : total_population = 180)
  (h2 : elderly_population = 30)
  (h3 : sample_size = 36)
  : (elderly_population * sample_size) / total_population = 6 := by
  sorry

end NUMINAMATH_CALUDE_elderly_sample_size_l1709_170980


namespace NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l1709_170937

-- Define set A
def A : Set ℝ := {x | |x - 1| ≤ 2}

-- Define set B
def B : Set ℝ := {x | x^2 - 4*x > 0}

-- State the theorem
theorem intersection_of_A_and_complement_of_B :
  A ∩ (Set.univ \ B) = Set.Icc 0 3 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l1709_170937


namespace NUMINAMATH_CALUDE_greatest_number_l1709_170986

theorem greatest_number (x : ℤ) (n : ℕ) : 
  (x ≤ 4) → 
  (2.134 * (n : ℝ)^(x : ℝ) < 210000) → 
  (∀ m : ℕ, m > n → 2.134 * (m : ℝ)^(4 : ℝ) ≥ 210000) → 
  n = 17 := by
sorry

end NUMINAMATH_CALUDE_greatest_number_l1709_170986


namespace NUMINAMATH_CALUDE_richter_frequency_ratio_l1709_170981

/-- Represents the energy released for a given Richter scale reading -/
def energy_released (x : ℝ) : ℝ := sorry

/-- The Richter scale property: a reading of x - 1 indicates one-tenth the released energy as x -/
axiom richter_scale_property (x : ℝ) : energy_released (x - 1) = (1 / 10) * energy_released x

/-- The frequency corresponding to a given Richter scale reading -/
def frequency (x : ℝ) : ℝ := sorry

/-- Theorem stating the relationship between frequencies for Richter scale readings 5 and 3 -/
theorem richter_frequency_ratio : frequency 5 = 100 * frequency 3 := by sorry

end NUMINAMATH_CALUDE_richter_frequency_ratio_l1709_170981


namespace NUMINAMATH_CALUDE_combined_height_of_tamara_and_kim_l1709_170971

/-- Given Tamara's height is 3 times Kim's height less 4 inches and Tamara is 68 inches tall,
    prove that the combined height of Tamara and Kim is 92 inches. -/
theorem combined_height_of_tamara_and_kim (kim_height : ℕ) : 
  (3 * kim_height - 4 = 68) → (68 + kim_height = 92) := by
  sorry

end NUMINAMATH_CALUDE_combined_height_of_tamara_and_kim_l1709_170971


namespace NUMINAMATH_CALUDE_power_of_product_rule_l1709_170992

theorem power_of_product_rule (a b : ℝ) : (a^3 * b)^2 = a^6 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_rule_l1709_170992


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_square_sum_l1709_170970

theorem arithmetic_geometric_mean_square_sum (a b : ℝ) : 
  (a + b) / 2 = 20 → Real.sqrt (a * b) = Real.sqrt 110 → a^2 + b^2 = 1380 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_square_sum_l1709_170970


namespace NUMINAMATH_CALUDE_ali_baba_max_coins_l1709_170943

/-- Represents the game state -/
structure GameState where
  piles : List Nat
  deriving Repr

/-- The initial game state -/
def initialState : GameState :=
  { piles := List.replicate 10 10 }

/-- Ali Baba's strategy -/
def aliBabaStrategy (state : GameState) : GameState :=
  sorry

/-- Thief's strategy -/
def thiefStrategy (state : GameState) : GameState :=
  sorry

/-- Play the game for a given number of rounds -/
def playGame (rounds : Nat) : GameState :=
  sorry

/-- Calculate the maximum number of coins Ali Baba can take -/
def maxCoinsAliBaba (finalState : GameState) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem ali_baba_max_coins :
  ∃ (rounds : Nat), maxCoinsAliBaba (playGame rounds) = 72 :=
sorry

end NUMINAMATH_CALUDE_ali_baba_max_coins_l1709_170943


namespace NUMINAMATH_CALUDE_solution_set_equiv_l1709_170923

/-- The solution set of ax^2 + 2ax > 0 -/
def SolutionSet (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 + 2 * a * x > 0}

/-- The proposition that 0 < a < 1 -/
def q (a : ℝ) : Prop := 0 < a ∧ a < 1

/-- The theorem stating that q is necessary and sufficient for the solution set to be ℝ -/
theorem solution_set_equiv (a : ℝ) : SolutionSet a = Set.univ ↔ q a := by sorry

end NUMINAMATH_CALUDE_solution_set_equiv_l1709_170923


namespace NUMINAMATH_CALUDE_twelve_hash_six_l1709_170990

/-- The # operation for real numbers -/
noncomputable def hash (r s : ℝ) : ℝ :=
  sorry

/-- Axioms for the # operation -/
axiom hash_zero (r : ℝ) : hash r 0 = r
axiom hash_comm (r s : ℝ) : hash r s = hash s r
axiom hash_succ (r s : ℝ) : hash (r + 1) s = hash r s + 2 * s + 1

/-- The main theorem to prove -/
theorem twelve_hash_six : hash 12 6 = 272 := by
  sorry

end NUMINAMATH_CALUDE_twelve_hash_six_l1709_170990


namespace NUMINAMATH_CALUDE_franks_initial_money_l1709_170919

/-- Frank's initial amount of money -/
def initial_money : ℕ := sorry

/-- The amount Frank spent on toys -/
def money_spent : ℕ := 8

/-- The amount Frank had left after spending -/
def money_left : ℕ := 8

/-- Theorem stating that Frank's initial money was $16 -/
theorem franks_initial_money : initial_money = 16 := by sorry

end NUMINAMATH_CALUDE_franks_initial_money_l1709_170919


namespace NUMINAMATH_CALUDE_mean_proportional_segments_l1709_170944

/-- Given that segment b is the mean proportional between segments a and c,
    prove that if a = 2 and b = 4, then c = 8. -/
theorem mean_proportional_segments (a b c : ℝ) 
  (h1 : b^2 = a * c) -- b is the mean proportional between a and c
  (h2 : a = 2)       -- a = 2 cm
  (h3 : b = 4)       -- b = 4 cm
  : c = 8 := by
  sorry

end NUMINAMATH_CALUDE_mean_proportional_segments_l1709_170944


namespace NUMINAMATH_CALUDE_exactly_two_identical_pairs_l1709_170984

/-- Two lines in the xy-plane -/
structure TwoLines where
  a : ℝ
  d : ℝ

/-- The condition for two lines to be identical -/
def are_identical (l : TwoLines) : Prop :=
  (4 / l.a = -l.d / 3) ∧ (l.d / l.a = 6)

/-- The theorem stating that there are exactly two pairs (a, d) that make the lines identical -/
theorem exactly_two_identical_pairs :
  ∃! (s : Finset TwoLines), (∀ l ∈ s, are_identical l) ∧ s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_identical_pairs_l1709_170984


namespace NUMINAMATH_CALUDE_max_log_sum_l1709_170961

theorem max_log_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 4*y = 40) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + 4*b = 40 → Real.log a + Real.log b ≤ Real.log x + Real.log y) →
  Real.log x + Real.log y = 2 := by
sorry

end NUMINAMATH_CALUDE_max_log_sum_l1709_170961


namespace NUMINAMATH_CALUDE_equal_utility_implies_u_equals_four_l1709_170951

def sunday_utility (u : ℝ) : ℝ := 2 * u * (10 - 2 * u)
def monday_utility (u : ℝ) : ℝ := 2 * (4 - 2 * u) * (2 * u + 4)

theorem equal_utility_implies_u_equals_four :
  ∀ u : ℝ, sunday_utility u = monday_utility u → u = 4 :=
by sorry

end NUMINAMATH_CALUDE_equal_utility_implies_u_equals_four_l1709_170951


namespace NUMINAMATH_CALUDE_all_terms_are_perfect_squares_l1709_170955

def a : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => 14 * a (n + 1) - a n - 4

theorem all_terms_are_perfect_squares :
  ∀ n : ℕ, ∃ s : ℤ, a n = s^2 := by
  sorry

end NUMINAMATH_CALUDE_all_terms_are_perfect_squares_l1709_170955


namespace NUMINAMATH_CALUDE_max_value_of_five_numbers_l1709_170916

theorem max_value_of_five_numbers (a b c d e : ℕ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e →  -- five distinct natural numbers
  (a + b + c + d + e) / 5 = 15 →  -- average is 15
  c = 18 →  -- median is 18
  e ≤ 37 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_five_numbers_l1709_170916


namespace NUMINAMATH_CALUDE_largest_number_with_conditions_l1709_170954

def is_valid_digit (d : ℕ) : Prop := d = 4 ∨ d = 5

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def all_digits_valid (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, is_valid_digit d

theorem largest_number_with_conditions :
  ∀ n : ℕ,
    all_digits_valid n →
    digit_sum n = 17 →
    n ≤ 5444 :=
sorry

end NUMINAMATH_CALUDE_largest_number_with_conditions_l1709_170954


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1709_170909

-- Define the conditions
def condition_p (x : ℝ) : Prop := |x + 1| ≤ 4
def condition_q (x : ℝ) : Prop := x^2 - 5*x + 6 ≤ 0

-- Theorem statement
theorem p_necessary_not_sufficient_for_q :
  (∀ x, condition_q x → condition_p x) ∧
  (∃ x, condition_p x ∧ ¬condition_q x) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1709_170909


namespace NUMINAMATH_CALUDE_calculation_proof_l1709_170901

theorem calculation_proof : (0.8 * 60 - 2/5 * 35) * Real.sqrt 144 = 408 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1709_170901


namespace NUMINAMATH_CALUDE_large_number_proof_l1709_170920

/-- A number composed of 80 hundred millions, 5 ten millions, and 6 ten thousands -/
def large_number : ℕ := 80 * 100000000 + 5 * 10000000 + 6 * 10000

/-- The same number expressed in units of ten thousand -/
def large_number_in_ten_thousands : ℕ := large_number / 10000

theorem large_number_proof :
  large_number = 8050060000 ∧ large_number_in_ten_thousands = 805006 := by
  sorry

end NUMINAMATH_CALUDE_large_number_proof_l1709_170920


namespace NUMINAMATH_CALUDE_inequality_proof_l1709_170983

theorem inequality_proof (k n : ℕ) (hk : k > 1) (hn : n > 1) :
  (1 : ℝ) / ((n + 1 : ℝ) ^ (1 / k : ℝ)) + (1 : ℝ) / ((k + 1 : ℝ) ^ (1 / n : ℝ)) > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1709_170983


namespace NUMINAMATH_CALUDE_correct_sampling_order_l1709_170931

-- Define the sampling methods
inductive SamplingMethod
| SimpleRandom
| Stratified
| Systematic

-- Define the characteristics of each scenario
structure Scenario where
  population_size : ℕ
  has_subgroups : Bool
  has_orderly_numbering : Bool

-- Define the three given scenarios
def scenario1 : Scenario := ⟨8, false, false⟩
def scenario2 : Scenario := ⟨2100, true, false⟩
def scenario3 : Scenario := ⟨700, false, true⟩

-- Function to determine the most appropriate sampling method for a given scenario
def appropriate_method (s : Scenario) : SamplingMethod :=
  if s.population_size ≤ 10 && !s.has_subgroups && !s.has_orderly_numbering then
    SamplingMethod.SimpleRandom
  else if s.has_subgroups then
    SamplingMethod.Stratified
  else if s.has_orderly_numbering then
    SamplingMethod.Systematic
  else
    SamplingMethod.SimpleRandom

-- Theorem stating that the given order of sampling methods is correct for the three scenarios
theorem correct_sampling_order :
  (appropriate_method scenario1 = SamplingMethod.SimpleRandom) ∧
  (appropriate_method scenario2 = SamplingMethod.Stratified) ∧
  (appropriate_method scenario3 = SamplingMethod.Systematic) := by
  sorry

end NUMINAMATH_CALUDE_correct_sampling_order_l1709_170931


namespace NUMINAMATH_CALUDE_problem_statement_l1709_170974

def f (x : ℝ) : ℝ := x^2 - 1

theorem problem_statement :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 →
    (∀ m : ℝ, (4 * m^2 * |f x| + 4 * f m ≤ |f (x - 1)| ↔ -1/2 ≤ m ∧ m ≤ 1/2))) ∧
  (∀ a : ℝ, (∀ x₁ : ℝ, 1 ≤ x₁ ∧ x₁ ≤ 2 →
    ∃ x₂ : ℝ, 1 ≤ x₂ ∧ x₂ ≤ 2 ∧ f x₁ = |2 * f x₂ - a * x₂|) ↔
      ((0 ≤ a ∧ a ≤ 3/2) ∨ a = 3)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1709_170974


namespace NUMINAMATH_CALUDE_least_sum_of_exponents_for_520_l1709_170938

theorem least_sum_of_exponents_for_520 (n : ℕ) (h1 : n = 520) :
  ∃ (a b : ℕ), 
    n = 2^a + 2^b ∧ 
    a ≠ b ∧ 
    (a = 3 ∨ b = 3) ∧ 
    ∀ (c d : ℕ), (n = 2^c + 2^d ∧ c ≠ d ∧ (c = 3 ∨ d = 3)) → a + b ≤ c + d :=
by sorry

end NUMINAMATH_CALUDE_least_sum_of_exponents_for_520_l1709_170938


namespace NUMINAMATH_CALUDE_solution_y_l1709_170991

theorem solution_y (y : ℝ) (h : (2 / y) + (3 / y) / (6 / y) = 1.5) : y = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_y_l1709_170991


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_195195_l1709_170945

theorem sum_of_prime_factors_195195 : 
  (Finset.sum (Finset.filter Nat.Prime (Finset.range (195195 + 1))) id = 39) ∧ 
  (∀ p : ℕ, p ∈ Finset.filter Nat.Prime (Finset.range (195195 + 1)) ↔ p.Prime ∧ 195195 % p = 0) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_prime_factors_195195_l1709_170945


namespace NUMINAMATH_CALUDE_twelve_people_in_line_l1709_170998

/-- The number of people in a line with Jeanne, given the number of people in front and behind her -/
def people_in_line (people_in_front : ℕ) (people_behind : ℕ) : ℕ :=
  people_in_front + 1 + people_behind

/-- Theorem stating that there are 12 people in the line -/
theorem twelve_people_in_line :
  people_in_line 4 7 = 12 := by
  sorry

#check twelve_people_in_line

end NUMINAMATH_CALUDE_twelve_people_in_line_l1709_170998


namespace NUMINAMATH_CALUDE_rhombus_properties_l1709_170950

structure Rhombus (O : ℝ × ℝ) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  side_length : ℝ
  OB_length : ℝ
  OD_length : ℝ
  is_rhombus : side_length = 4 ∧ OB_length = 6 ∧ OD_length = 6

def on_semicircle (A : ℝ × ℝ) : Prop :=
  let (x, y) := A
  (x - 2)^2 + y^2 = 4 ∧ 2 ≤ x ∧ x ≤ 4

theorem rhombus_properties (r : Rhombus (0, 0)) :
  (|r.A.1 * r.C.1 + r.A.2 * r.C.2| = 36) ∧
  (∀ A', on_semicircle A' → 
    ∃ C', r.C = C' → (C' = (5, 5) ∨ C' = (5, -5))) :=
sorry

end NUMINAMATH_CALUDE_rhombus_properties_l1709_170950


namespace NUMINAMATH_CALUDE_disinfectant_sales_analysis_l1709_170929

/-- Represents the daily sales quantity as a function of selling price -/
def sales_quantity (x : ℤ) : ℤ := -5 * x + 150

/-- Represents the daily profit as a function of selling price -/
def profit (x : ℤ) : ℤ := (x - 8) * sales_quantity x

theorem disinfectant_sales_analysis 
  (h1 : ∀ x : ℤ, 8 ≤ x → x ≤ 15 → sales_quantity x = -5 * x + 150)
  (h2 : sales_quantity 9 = 105)
  (h3 : sales_quantity 11 = 95)
  (h4 : sales_quantity 13 = 85) :
  (∀ x : ℤ, 8 ≤ x → x ≤ 15 → sales_quantity x = -5 * x + 150) ∧
  (profit 13 = 425) ∧
  (∀ x : ℤ, 8 ≤ x → x ≤ 15 → profit x ≤ 525) ∧
  (profit 15 = 525) := by
sorry


end NUMINAMATH_CALUDE_disinfectant_sales_analysis_l1709_170929


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l1709_170969

theorem negation_of_universal_statement :
  (¬ (∀ x : ℝ, 2^x + x^2 > 0)) ↔ (∃ x₀ : ℝ, 2^x₀ + x₀^2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l1709_170969


namespace NUMINAMATH_CALUDE_peanuts_remaining_l1709_170999

theorem peanuts_remaining (initial_peanuts : ℕ) (brock_fraction : ℚ) (bonita_peanuts : ℕ) : 
  initial_peanuts = 148 →
  brock_fraction = 1/4 →
  bonita_peanuts = 29 →
  initial_peanuts - (initial_peanuts * brock_fraction).floor - bonita_peanuts = 82 :=
by
  sorry

end NUMINAMATH_CALUDE_peanuts_remaining_l1709_170999


namespace NUMINAMATH_CALUDE_allocation_schemes_l1709_170963

/- Define the number of attending physicians -/
def num_attending : Nat := 2

/- Define the number of interns -/
def num_interns : Nat := 4

/- Define the number of groups -/
def num_groups : Nat := 2

/- Define the number of attending physicians per group -/
def attending_per_group : Nat := 1

/- Define the number of interns per group -/
def interns_per_group : Nat := 2

/- Define that one specific intern must be in a particular group -/
def specific_intern_placement : Bool := true

/- Theorem statement -/
theorem allocation_schemes : 
  (num_attending = 2) → 
  (num_interns = 4) → 
  (num_groups = 2) → 
  (attending_per_group = 1) → 
  (interns_per_group = 2) → 
  specific_intern_placement →
  (Nat.choose (num_interns - 1) (interns_per_group - 1) * num_attending = 6) := by
  sorry

end NUMINAMATH_CALUDE_allocation_schemes_l1709_170963


namespace NUMINAMATH_CALUDE_perpendicular_to_two_planes_implies_parallel_l1709_170921

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_to_two_planes_implies_parallel 
  (m : Line) (α β : Plane) (h1 : α ≠ β) :
  perpendicular m α → perpendicular m β → parallel α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_two_planes_implies_parallel_l1709_170921


namespace NUMINAMATH_CALUDE_box_paint_area_l1709_170917

/-- The total area to paint inside a cuboid box -/
def total_paint_area (length width height : ℝ) : ℝ :=
  2 * (length * height + width * height) + length * width

/-- Theorem: The total area to paint inside a cuboid box with dimensions 18 cm long, 10 cm wide, and 2 cm high is 292 square centimeters -/
theorem box_paint_area :
  total_paint_area 18 10 2 = 292 := by
  sorry

end NUMINAMATH_CALUDE_box_paint_area_l1709_170917


namespace NUMINAMATH_CALUDE_cosine_sum_identity_l1709_170946

theorem cosine_sum_identity : 
  Real.cos (42 * π / 180) * Real.cos (18 * π / 180) - 
  Real.cos (48 * π / 180) * Real.sin (18 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_identity_l1709_170946


namespace NUMINAMATH_CALUDE_events_not_independent_l1709_170903

/- Define the sample space -/
def Ω : Type := Fin 10

/- Define the events A and B -/
def A : Set Ω := {ω : Ω | ω.val < 5}
def B : Set Ω := {ω : Ω | ω.val % 2 = 0}

/- Define the probability measure -/
def P : Set Ω → ℝ := sorry

/- State the theorem -/
theorem events_not_independent : ¬(P (A ∩ B) = P A * P B) := by sorry

end NUMINAMATH_CALUDE_events_not_independent_l1709_170903


namespace NUMINAMATH_CALUDE_two_digit_product_777_l1709_170947

theorem two_digit_product_777 :
  ∀ a b : ℕ,
    10 ≤ a ∧ a < 100 →
    10 ≤ b ∧ b < 100 →
    a * b = 777 →
    ((a = 21 ∧ b = 37) ∨ (a = 37 ∧ b = 21)) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_product_777_l1709_170947


namespace NUMINAMATH_CALUDE_angle_properties_l1709_170985

/-- Given that the terminal side of angle α passes through point P(5a, -12a) where a < 0,
    prove that tan α = -12/5 and sin α + cos α = 7/13 -/
theorem angle_properties (a : ℝ) (α : ℝ) (h : a < 0) :
  let x := 5 * a
  let y := -12 * a
  let r := Real.sqrt (x^2 + y^2)
  (Real.tan α = -12/5) ∧ (Real.sin α + Real.cos α = 7/13) := by
sorry

end NUMINAMATH_CALUDE_angle_properties_l1709_170985


namespace NUMINAMATH_CALUDE_extremal_point_implies_k_range_l1709_170932

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (Real.exp x) / (x^2) + 2*k*(Real.log x) - k*x

theorem extremal_point_implies_k_range :
  ∀ k : ℝ, (∀ x : ℝ, x > 0 → (x ≠ 2 → (deriv (f k)) x ≠ 0)) →
  k ∈ Set.Iic ((Real.exp 2) / 4) :=
sorry

end NUMINAMATH_CALUDE_extremal_point_implies_k_range_l1709_170932


namespace NUMINAMATH_CALUDE_A_union_B_eq_A_l1709_170962

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | (x + 1) * (x - 4) < 0}
def B : Set ℝ := {x : ℝ | Real.log x < 1}

-- State the theorem
theorem A_union_B_eq_A : A ∪ B = A := by sorry

end NUMINAMATH_CALUDE_A_union_B_eq_A_l1709_170962


namespace NUMINAMATH_CALUDE_unique_square_divisible_by_12_l1709_170900

theorem unique_square_divisible_by_12 :
  ∃! x : ℕ, (∃ y : ℕ, x = y^2) ∧ 
            (12 ∣ x) ∧ 
            100 ≤ x ∧ x ≤ 200 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_square_divisible_by_12_l1709_170900


namespace NUMINAMATH_CALUDE_divisibility_problem_l1709_170912

theorem divisibility_problem (a b c : ℕ) 
  (ha : a > 1) 
  (hb : b > c) 
  (hc : c > 1) 
  (hdiv : (a * b * c + 1) % (a * b - b + 1) = 0) : 
  b % a = 0 := by
sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1709_170912


namespace NUMINAMATH_CALUDE_eight_ampersand_five_l1709_170908

def ampersand (a b : ℤ) : ℤ := (a + b) * (a - b)

theorem eight_ampersand_five : ampersand 8 5 = 39 := by
  sorry

end NUMINAMATH_CALUDE_eight_ampersand_five_l1709_170908


namespace NUMINAMATH_CALUDE_adult_dog_cost_l1709_170911

/-- The cost to prepare animals for adoption -/
structure AdoptionCost where
  cat : ℕ → ℕ     -- Cost for cats
  dog : ℕ → ℕ     -- Cost for adult dogs
  puppy : ℕ → ℕ   -- Cost for puppies

/-- The theorem stating the cost for each adult dog -/
theorem adult_dog_cost (c : AdoptionCost) 
  (h1 : c.cat 1 = 50)
  (h2 : c.puppy 1 = 150)
  (h3 : c.cat 2 + c.dog 3 + c.puppy 2 = 700) :
  c.dog 1 = 100 := by
  sorry


end NUMINAMATH_CALUDE_adult_dog_cost_l1709_170911


namespace NUMINAMATH_CALUDE_tetris_score_calculation_l1709_170960

/-- Represents the score calculation for a Tetris game with bonus conditions -/
theorem tetris_score_calculation 
  (single_line_score : ℕ)
  (tetris_score_multiplier : ℕ)
  (single_tetris_bonus_multiplier : ℕ)
  (back_to_back_tetris_bonus : ℕ)
  (single_double_triple_bonus : ℕ)
  (singles_count : ℕ)
  (tetrises_count : ℕ)
  (doubles_count : ℕ)
  (triples_count : ℕ)
  (single_tetris_consecutive_count : ℕ)
  (back_to_back_tetris_count : ℕ)
  (single_double_triple_consecutive_count : ℕ)
  (h1 : single_line_score = 1000)
  (h2 : tetris_score_multiplier = 8)
  (h3 : single_tetris_bonus_multiplier = 2)
  (h4 : back_to_back_tetris_bonus = 5000)
  (h5 : single_double_triple_bonus = 3000)
  (h6 : singles_count = 6)
  (h7 : tetrises_count = 4)
  (h8 : doubles_count = 2)
  (h9 : triples_count = 1)
  (h10 : single_tetris_consecutive_count = 1)
  (h11 : back_to_back_tetris_count = 1)
  (h12 : single_double_triple_consecutive_count = 1) :
  singles_count * single_line_score + 
  tetrises_count * (tetris_score_multiplier * single_line_score) +
  single_tetris_consecutive_count * (single_tetris_bonus_multiplier - 1) * (tetris_score_multiplier * single_line_score) +
  back_to_back_tetris_count * back_to_back_tetris_bonus +
  single_double_triple_consecutive_count * single_double_triple_bonus = 54000 := by
  sorry


end NUMINAMATH_CALUDE_tetris_score_calculation_l1709_170960


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1709_170965

theorem quadratic_equation_solution : 
  ∃! x : ℝ, x^2 + 6*x + 8 = -(x + 4)*(x + 6) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1709_170965


namespace NUMINAMATH_CALUDE_solve_system_l1709_170913

theorem solve_system (a b x y : ℝ) 
  (eq1 : a * x + b * y = 16)
  (eq2 : b * x - a * y = -12)
  (sol_x : x = 2)
  (sol_y : y = 4) : 
  a = 4 ∧ b = 2 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l1709_170913


namespace NUMINAMATH_CALUDE_correct_selection_count_l1709_170966

/-- The number of ways to select course representatives from a class -/
def select_representatives (num_boys num_girls num_subjects : ℕ) : ℕ × ℕ × ℕ :=
  let scenario1 := sorry
  let scenario2 := sorry
  let scenario3 := sorry
  (scenario1, scenario2, scenario3)

/-- Theorem stating the correct number of ways to select representatives under different conditions -/
theorem correct_selection_count :
  select_representatives 6 4 5 = (22320, 12096, 1008) := by
  sorry

end NUMINAMATH_CALUDE_correct_selection_count_l1709_170966


namespace NUMINAMATH_CALUDE_twentieth_term_of_sequence_l1709_170952

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem twentieth_term_of_sequence :
  let a₁ := 2
  let a₂ := 7
  let d := a₂ - a₁
  arithmetic_sequence a₁ d 20 = 97 := by sorry

end NUMINAMATH_CALUDE_twentieth_term_of_sequence_l1709_170952


namespace NUMINAMATH_CALUDE_train_overtake_l1709_170996

/-- Proves that Train B overtakes Train A in 120 minutes given the specified conditions -/
theorem train_overtake (speed_a speed_b : ℝ) (head_start : ℝ) (overtake_time : ℝ) : 
  speed_a = 60 →
  speed_b = 80 →
  head_start = 40 / 60 →
  overtake_time = 120 / 60 →
  speed_a * (head_start + overtake_time) = speed_b * overtake_time :=
by sorry

end NUMINAMATH_CALUDE_train_overtake_l1709_170996


namespace NUMINAMATH_CALUDE_inverse_z_minus_z_inv_l1709_170927

/-- Given a complex number z = 1 + i where i² = -1, prove that (z - z⁻¹)⁻¹ = (1 - 3i) / 5 -/
theorem inverse_z_minus_z_inv (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := 1 + i
  (z - z⁻¹)⁻¹ = (1 - 3*i) / 5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_z_minus_z_inv_l1709_170927


namespace NUMINAMATH_CALUDE_square_of_6y_minus_2_l1709_170910

-- Define the condition
def satisfies_equation (y : ℝ) : Prop := 3 * y^2 + 2 = 5 * y + 7

-- State the theorem
theorem square_of_6y_minus_2 (y : ℝ) (h : satisfies_equation y) : (6 * y - 2)^2 = 94 := by
  sorry

end NUMINAMATH_CALUDE_square_of_6y_minus_2_l1709_170910


namespace NUMINAMATH_CALUDE_log_equation_holds_l1709_170967

theorem log_equation_holds (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x / Real.log 3) * (Real.log 5 / Real.log x) = Real.log 5 / Real.log 3 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_holds_l1709_170967


namespace NUMINAMATH_CALUDE_three_digit_square_mod_1000_l1709_170936

theorem three_digit_square_mod_1000 (n : ℕ) : 
  (100 ≤ n ∧ n ≤ 999) → (n^2 ≡ n [ZMOD 1000] ↔ n = 376 ∨ n = 625) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_square_mod_1000_l1709_170936


namespace NUMINAMATH_CALUDE_smallest_divisible_by_10_11_12_13_eight_five_eight_zero_divisible_smallest_positive_integer_divisible_by_10_11_12_13_l1709_170907

theorem smallest_divisible_by_10_11_12_13 : 
  ∀ n : ℕ, n > 0 ∧ 10 ∣ n ∧ 11 ∣ n ∧ 12 ∣ n ∧ 13 ∣ n → n ≥ 8580 := by
  sorry

theorem eight_five_eight_zero_divisible :
  10 ∣ 8580 ∧ 11 ∣ 8580 ∧ 12 ∣ 8580 ∧ 13 ∣ 8580 := by
  sorry

theorem smallest_positive_integer_divisible_by_10_11_12_13 :
  ∃! n : ℕ, n > 0 ∧ 
    (∀ m : ℕ, m > 0 ∧ 10 ∣ m ∧ 11 ∣ m ∧ 12 ∣ m ∧ 13 ∣ m → n ≤ m) ∧
    10 ∣ n ∧ 11 ∣ n ∧ 12 ∣ n ∧ 13 ∣ n ∧ n = 8580 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_10_11_12_13_eight_five_eight_zero_divisible_smallest_positive_integer_divisible_by_10_11_12_13_l1709_170907


namespace NUMINAMATH_CALUDE_incorrect_representation_l1709_170993

/-- Represents a repeating decimal -/
structure RepeatingDecimal where
  nonRepeating : ℕ → ℕ  -- P: mapping from position to digit
  repeating : ℕ → ℕ     -- Q: mapping from position to digit
  r : ℕ                 -- length of non-repeating part
  s : ℕ                 -- length of repeating part

/-- The decimal representation of a RepeatingDecimal -/
def decimalRepresentation (d : RepeatingDecimal) : ℚ :=
  sorry

/-- Theorem stating that the given representation is incorrect -/
theorem incorrect_representation (d : RepeatingDecimal) :
  ∃ (P Q : ℕ), 
    (10^d.r * 10^(2*d.s) * decimalRepresentation d ≠ 
     (P * 100 + Q * 10 + Q : ℚ) + decimalRepresentation d) :=
  sorry

end NUMINAMATH_CALUDE_incorrect_representation_l1709_170993


namespace NUMINAMATH_CALUDE_average_of_abc_is_three_l1709_170906

theorem average_of_abc_is_three (A B C : ℚ) 
  (eq1 : 101 * C - 202 * A = 404)
  (eq2 : 101 * B + 303 * A = 505)
  (eq3 : 101 * A + 101 * B + 101 * C = 303) :
  (A + B + C) / 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_abc_is_three_l1709_170906


namespace NUMINAMATH_CALUDE_range_of_fraction_l1709_170972

theorem range_of_fraction (x y : ℝ) (hx : 1 ≤ x ∧ x ≤ 4) (hy : 3 ≤ y ∧ y ≤ 6) :
  ∃ (z : ℝ), z = x / y ∧ 1/6 ≤ z ∧ z ≤ 4/3 :=
sorry

end NUMINAMATH_CALUDE_range_of_fraction_l1709_170972


namespace NUMINAMATH_CALUDE_number_of_boys_l1709_170935

theorem number_of_boys (girls : ℕ) (groups : ℕ) (members_per_group : ℕ) 
  (h1 : girls = 12)
  (h2 : groups = 7)
  (h3 : members_per_group = 3) :
  groups * members_per_group - girls = 9 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_l1709_170935


namespace NUMINAMATH_CALUDE_area_enclosed_by_curve_l1709_170982

/-- The area enclosed by a curve composed of 12 congruent circular arcs -/
theorem area_enclosed_by_curve (arc_length : Real) (hexagon_side : Real) : 
  arc_length = 5 * Real.pi / 6 →
  hexagon_side = 4 →
  ∃ (area : Real), 
    area = 48 * Real.sqrt 3 + 125 * Real.pi / 2 ∧
    area = (3 * Real.sqrt 3 / 2 * hexagon_side ^ 2) + 
           (12 * (arc_length / (2 * Real.pi)) * Real.pi * (arc_length / Real.pi) ^ 2) :=
by sorry


end NUMINAMATH_CALUDE_area_enclosed_by_curve_l1709_170982


namespace NUMINAMATH_CALUDE_percentage_increase_decrease_l1709_170939

theorem percentage_increase_decrease (p q M : ℝ) 
  (hp : p > 0) (hq : q > 0) (hq_bound : q < 200) (hM : M > 0) :
  M * (1 + p / 100) * (1 - q / 100) > M ↔ p > 100 * q / (100 - q) := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_decrease_l1709_170939


namespace NUMINAMATH_CALUDE_probability_of_drawing_heart_l1709_170942

-- Define the total number of cards
def total_cards : ℕ := 5

-- Define the number of heart cards
def heart_cards : ℕ := 3

-- Define the number of spade cards
def spade_cards : ℕ := 2

-- Theorem statement
theorem probability_of_drawing_heart :
  (heart_cards : ℚ) / total_cards = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_of_drawing_heart_l1709_170942
