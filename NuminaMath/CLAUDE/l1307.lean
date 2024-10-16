import Mathlib

namespace NUMINAMATH_CALUDE_correct_sample_size_l1307_130765

/-- Represents a school in the sampling problem -/
structure School where
  students : ℕ

/-- Represents the sampling data for two schools -/
structure SamplingData where
  schoolA : School
  schoolB : School
  sampleA : ℕ

/-- Calculates the proportional sample size for the second school -/
def calculateSampleB (data : SamplingData) : ℕ :=
  (data.schoolB.students * data.sampleA) / data.schoolA.students

/-- Theorem stating the correct sample size for School B -/
theorem correct_sample_size (data : SamplingData) 
    (h1 : data.schoolA.students = 800)
    (h2 : data.schoolB.students = 500)
    (h3 : data.sampleA = 48) :
  calculateSampleB data = 30 := by
  sorry

#eval calculateSampleB { 
  schoolA := { students := 800 }, 
  schoolB := { students := 500 }, 
  sampleA := 48 
}

end NUMINAMATH_CALUDE_correct_sample_size_l1307_130765


namespace NUMINAMATH_CALUDE_f_sum_reciprocal_l1307_130724

theorem f_sum_reciprocal (x : ℝ) (hx : x > 0) : 
  let f := fun (y : ℝ) => y / (y + 1)
  f x + f (1/x) = 1 := by
sorry

end NUMINAMATH_CALUDE_f_sum_reciprocal_l1307_130724


namespace NUMINAMATH_CALUDE_fake_coin_strategy_exists_find_fake_coin_correct_l1307_130711

/-- Represents a strategy to find a fake coin among 2^(2^k) coins using dogs -/
structure FakeCoinStrategy (k : ℕ) :=
  (num_tests : ℕ)
  (find_fake_coin : Unit → ℕ)

/-- Theorem stating the existence of a strategy to find the fake coin -/
theorem fake_coin_strategy_exists (k : ℕ) :
  ∃ (strategy : FakeCoinStrategy k),
    strategy.num_tests ≤ 2^k + k + 2 ∧
    strategy.find_fake_coin () < 2^(2^k) :=
by sorry

/-- Function to perform a test with selected coins and a dog -/
def perform_test (selected_coins : Finset ℕ) (dog : ℕ) : Bool :=
sorry

/-- Function to select a dog for testing -/
def select_dog : ℕ :=
sorry

/-- Function to implement the strategy and find the fake coin -/
def find_fake_coin (k : ℕ) : ℕ :=
sorry

/-- Theorem proving the correctness of the find_fake_coin function -/
theorem find_fake_coin_correct (k : ℕ) :
  ∃ (num_tests : ℕ),
    num_tests ≤ 2^k + k + 2 ∧
    find_fake_coin k < 2^(2^k) :=
by sorry

end NUMINAMATH_CALUDE_fake_coin_strategy_exists_find_fake_coin_correct_l1307_130711


namespace NUMINAMATH_CALUDE_man_speed_calculation_man_speed_specific_case_l1307_130776

/-- Calculates the speed of a man walking in the same direction as a train, given the train's length, speed, and time to cross the man. -/
theorem man_speed_calculation (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) : Real :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let relative_speed := train_length / crossing_time
  train_speed_ms - relative_speed

/-- The speed of a man walking in the same direction as a train, given specific conditions. -/
theorem man_speed_specific_case : 
  ∃ (ε : Real), ε > 0 ∧ 
  |man_speed_calculation 100 63 5.999520038396929 - 0.831946| < ε :=
sorry

end NUMINAMATH_CALUDE_man_speed_calculation_man_speed_specific_case_l1307_130776


namespace NUMINAMATH_CALUDE_harmonic_mean_leq_geometric_mean_l1307_130780

theorem harmonic_mean_leq_geometric_mean (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  2 / (1/a + 1/b) ≤ Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_harmonic_mean_leq_geometric_mean_l1307_130780


namespace NUMINAMATH_CALUDE_function_derivative_equality_l1307_130720

/-- Given a function f(x) = x(2017 + ln x), prove that if f'(x₀) = 2018, then x₀ = 1 -/
theorem function_derivative_equality (x₀ : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x * (2017 + Real.log x)
  (deriv f x₀ = 2018) → x₀ = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_derivative_equality_l1307_130720


namespace NUMINAMATH_CALUDE_f_monotone_intervals_cos_alpha_value_l1307_130788

noncomputable def f (x : ℝ) : ℝ := 
  1/2 * (Real.sin x + Real.cos x) * (Real.sin x - Real.cos x) + Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_monotone_intervals (k : ℤ) : 
  StrictMonoOn f (Set.Icc (-Real.pi/6 + k * Real.pi) (Real.pi/3 + k * Real.pi)) := by sorry

theorem cos_alpha_value (α : ℝ) 
  (h1 : f (α/2 + Real.pi/4) = Real.sqrt 3 / 3) 
  (h2 : -Real.pi/2 < α) 
  (h3 : α < 0) : 
  Real.cos α = (3 + Real.sqrt 6) / 6 := by sorry

end NUMINAMATH_CALUDE_f_monotone_intervals_cos_alpha_value_l1307_130788


namespace NUMINAMATH_CALUDE_bobs_garden_raised_bed_area_l1307_130731

/-- Represents the dimensions and layout of a garden --/
structure Garden where
  length : ℝ
  width : ℝ
  tilled_ratio : ℝ
  trellis_ratio : ℝ

/-- Calculates the area of the raised bed section in a garden --/
def raised_bed_area (g : Garden) : ℝ :=
  let total_area := g.length * g.width
  let tilled_area := g.tilled_ratio * total_area
  let remaining_area := total_area - tilled_area
  let trellis_area := g.trellis_ratio * remaining_area
  remaining_area - trellis_area

/-- Theorem stating that the raised bed area of the given garden is 8800 square feet --/
theorem bobs_garden_raised_bed_area :
  let g : Garden := {
    length := 220,
    width := 120,
    tilled_ratio := 1/2,
    trellis_ratio := 1/3
  }
  raised_bed_area g = 8800 := by
  sorry

end NUMINAMATH_CALUDE_bobs_garden_raised_bed_area_l1307_130731


namespace NUMINAMATH_CALUDE_problem_proof_l1307_130782

theorem problem_proof : 4⁻¹ - Real.sqrt (1 / 16) + (3 - Real.sqrt 2) ^ 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l1307_130782


namespace NUMINAMATH_CALUDE_popsicle_melting_rate_l1307_130708

/-- Given a sequence of 6 terms where each term is twice the previous term and the first term is 1,
    prove that the last term is equal to 32. -/
theorem popsicle_melting_rate (seq : Fin 6 → ℕ) 
    (h1 : seq 0 = 1)
    (h2 : ∀ i : Fin 5, seq (i.succ) = 2 * seq i) : 
  seq 5 = 32 := by
  sorry

end NUMINAMATH_CALUDE_popsicle_melting_rate_l1307_130708


namespace NUMINAMATH_CALUDE_sum_of_polynomial_values_l1307_130783

/-- The polynomial function P(x) = x^5 - 1.7 * x^3 + 2.5 -/
def P (x : ℝ) : ℝ := x^5 - 1.7 * x^3 + 2.5

/-- Theorem: The sum of P(19.1) and P(-19.1) is equal to 5 -/
theorem sum_of_polynomial_values : P 19.1 + P (-19.1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_polynomial_values_l1307_130783


namespace NUMINAMATH_CALUDE_unique_divisibility_pair_l1307_130794

theorem unique_divisibility_pair : ∀ a b : ℕ+,
  (∃ k : ℤ, (4 * b.val - 1) = k * (3 * a.val + 1)) →
  (∃ m : ℤ, (3 * a.val - 1) = m * (2 * b.val + 1)) →
  a.val = 2 ∧ b.val = 2 := by
sorry

end NUMINAMATH_CALUDE_unique_divisibility_pair_l1307_130794


namespace NUMINAMATH_CALUDE_miles_guitars_l1307_130739

/-- Represents the number of musical instruments Miles owns. -/
structure MilesInstruments where
  guitars : ℕ
  trumpets : ℕ
  trombones : ℕ
  frenchHorns : ℕ

/-- The total number of Miles' fingers. -/
def numFingers : ℕ := 10

/-- The number of Miles' hands. -/
def numHands : ℕ := 2

/-- The number of Miles' heads. -/
def numHeads : ℕ := 1

/-- The total number of musical instruments Miles owns. -/
def totalInstruments : ℕ := 17

/-- Theorem stating the number of guitars Miles owns. -/
theorem miles_guitars :
  ∃ (m : MilesInstruments),
    m.trumpets = numFingers - 3
    ∧ m.trombones = numHeads + 2
    ∧ m.frenchHorns = m.guitars - 1
    ∧ m.guitars = numHands + 2
    ∧ m.trumpets + m.trombones + m.guitars + m.frenchHorns = totalInstruments
    ∧ m.guitars = 4 := by
  sorry

end NUMINAMATH_CALUDE_miles_guitars_l1307_130739


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l1307_130795

/-- 
Given an ellipse with its focus on the x-axis, minor axis length of 16, and eccentricity of 3/5,
prove that its standard equation is x²/100 + y²/64 = 1.
-/
theorem ellipse_standard_equation (b : ℝ) (e : ℝ) :
  b = 8 → e = 3/5 → ∃ (a : ℝ), 
    a > 0 ∧ 
    b > 0 ∧ 
    e = Real.sqrt (1 - b^2 / a^2) ∧
    (∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 100 + y^2 / 64 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l1307_130795


namespace NUMINAMATH_CALUDE_inequality_proof_l1307_130751

theorem inequality_proof (u v x y a b c d : ℝ) 
  (hu : u > 0) (hv : v > 0) (hx : x > 0) (hy : y > 0)
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (u / x + v / y ≥ 4 * (u * y + v * x) / ((x + y) ^ 2)) ∧
  (a / (b + 2 * c + d) + b / (c + 2 * d + a) + c / (d + 2 * a + b) + d / (a + 2 * b + c) ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1307_130751


namespace NUMINAMATH_CALUDE_parallel_vectors_trig_identity_l1307_130735

/-- Given two parallel vectors a and b, prove that cos(2α) + sin(2α) = -7/5 -/
theorem parallel_vectors_trig_identity (α : ℝ) :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (Real.sin α - Real.cos α, Real.sin α + Real.cos α)
  (∃ (k : ℝ), a.1 * b.2 = k * a.2 * b.1) →  -- parallel vectors condition
  Real.cos (2 * α) + Real.sin (2 * α) = -7/5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_trig_identity_l1307_130735


namespace NUMINAMATH_CALUDE_circle_tangent_to_parallel_lines_l1307_130789

/-- A circle is tangent to two parallel lines and its center lies on a third line. -/
theorem circle_tangent_to_parallel_lines (x y : ℚ) :
  -- The circle is tangent to these two lines
  (6 * x - 5 * y = 40 ∨ 6 * x - 5 * y = -20) →
  -- The center lies on this line
  (3 * x + 2 * y = 0) →
  -- The point (20/27, -10/9) is the center of the circle
  x = 20/27 ∧ y = -10/9 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_to_parallel_lines_l1307_130789


namespace NUMINAMATH_CALUDE_f_inequality_l1307_130768

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_increasing : ∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x < f y
axiom f_even_shifted : ∀ x, f (x + 2) = f (-x + 2)

-- State the theorem to be proved
theorem f_inequality : f (5/2) > f 1 ∧ f 1 > f (7/2) := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l1307_130768


namespace NUMINAMATH_CALUDE_sqrt_non_square_irrational_l1307_130764

theorem sqrt_non_square_irrational (a : ℤ) 
  (h : ∀ n : ℤ, n^2 ≠ a) : 
  Irrational (Real.sqrt (a : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_non_square_irrational_l1307_130764


namespace NUMINAMATH_CALUDE_last_two_nonzero_digits_70_factorial_l1307_130791

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := sorry

/-- Function to get the last two nonzero digits of a number -/
def lastTwoNonzeroDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating that the last two nonzero digits of 70! are 44 -/
theorem last_two_nonzero_digits_70_factorial :
  lastTwoNonzeroDigits (factorial 70) = 44 := by sorry

end NUMINAMATH_CALUDE_last_two_nonzero_digits_70_factorial_l1307_130791


namespace NUMINAMATH_CALUDE_smallest_four_digit_in_pascal_l1307_130773

-- Define Pascal's triangle
def pascal_triangle : Nat → Nat → Nat
  | 0, _ => 1
  | n + 1, 0 => 1
  | n + 1, k + 1 => pascal_triangle n k + pascal_triangle n (k + 1)

-- Define a predicate to check if a number is in Pascal's triangle
def in_pascal_triangle (n : Nat) : Prop :=
  ∃ (row col : Nat), pascal_triangle row col = n

-- Theorem statement
theorem smallest_four_digit_in_pascal :
  (∀ n, 1000 ≤ n → n < 10000 → in_pascal_triangle n) →
  (∀ n, n < 1000 → n < 10000 → in_pascal_triangle n) →
  (∃ n, 1000 ≤ n ∧ n < 10000 ∧ in_pascal_triangle n) →
  (∀ n, 1000 ≤ n → n < 10000 → in_pascal_triangle n → 1000 ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_in_pascal_l1307_130773


namespace NUMINAMATH_CALUDE_gift_exchange_equation_l1307_130714

/-- Represents a gathering of people exchanging gifts -/
structure Gathering where
  /-- The number of attendees -/
  attendees : ℕ
  /-- The total number of gifts exchanged -/
  gifts : ℕ
  /-- Each pair of attendees exchanges a different small gift -/
  unique_exchanges : ∀ (a b : Fin attendees), a ≠ b → True

/-- The theorem stating the relationship between attendees and gifts exchanged -/
theorem gift_exchange_equation (g : Gathering) (h : g.gifts = 56) :
  g.attendees * (g.attendees - 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_gift_exchange_equation_l1307_130714


namespace NUMINAMATH_CALUDE_intersection_points_correct_l1307_130756

/-- The number of intersection points of segments joining m distinct points 
    on the positive x-axis to n distinct points on the positive y-axis, 
    where no three segments are concurrent. -/
def intersectionPoints (m n : ℕ) : ℕ :=
  m * n * (m - 1) * (n - 1) / 4

/-- Theorem stating that the number of intersection points is correct. -/
theorem intersection_points_correct (m n : ℕ) :
  intersectionPoints m n = m * n * (m - 1) * (n - 1) / 4 :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_correct_l1307_130756


namespace NUMINAMATH_CALUDE_maximize_sum_with_constraint_l1307_130767

theorem maximize_sum_with_constraint (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_constraint : 9 * a^2 + 4 * b^2 + c^2 = 91) :
  a + 2*b + 3*c ≤ 91/3 :=
by sorry

end NUMINAMATH_CALUDE_maximize_sum_with_constraint_l1307_130767


namespace NUMINAMATH_CALUDE_music_class_size_l1307_130734

theorem music_class_size (total_students : ℕ) (art_music_overlap : ℕ) :
  total_students = 60 →
  art_music_overlap = 8 →
  ∃ (art_only music_only : ℕ),
    total_students = art_only + music_only + art_music_overlap ∧
    art_only + art_music_overlap = music_only + art_music_overlap + 10 →
    music_only + art_music_overlap = 33 :=
by sorry

end NUMINAMATH_CALUDE_music_class_size_l1307_130734


namespace NUMINAMATH_CALUDE_inverse_relationship_scenarios_l1307_130709

/-- Represents a scenario with two variables that may have an inverse relationship -/
structure Scenario where
  x : ℝ
  y : ℝ
  k : ℝ
  h_k_nonzero : k ≠ 0

/-- Checks if a scenario satisfies the inverse relationship y = k/x -/
def has_inverse_relationship (s : Scenario) : Prop :=
  s.y = s.k / s.x

/-- Rectangle scenario with fixed area -/
def rectangle_scenario (area x y : ℝ) (h : area ≠ 0) : Scenario where
  x := x
  y := y
  k := area
  h_k_nonzero := h

/-- Village land scenario with fixed total arable land -/
def village_land_scenario (total_land n S : ℝ) (h : total_land ≠ 0) : Scenario where
  x := n
  y := S
  k := total_land
  h_k_nonzero := h

/-- Car travel scenario with fixed speed -/
def car_travel_scenario (speed s t : ℝ) (h : speed ≠ 0) : Scenario where
  x := t
  y := s
  k := speed
  h_k_nonzero := h

theorem inverse_relationship_scenarios 
  (rect : Scenario) 
  (village : Scenario) 
  (car : Scenario) : 
  has_inverse_relationship rect ∧ 
  has_inverse_relationship village ∧ 
  ¬has_inverse_relationship car := by
  sorry

end NUMINAMATH_CALUDE_inverse_relationship_scenarios_l1307_130709


namespace NUMINAMATH_CALUDE_min_value_expression_l1307_130733

theorem min_value_expression (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 2) (hy : 0 ≤ y ∧ y ≤ 2) (hz : 0 ≤ z ∧ z ≤ 2) : 
  1/((2 - x)*(2 - y)*(2 - z)) + 1/((2 + x)*(2 + y)*(2 + z)) + 1/(1 + (x+y+z)/3) ≥ 2 ∧
  (x = 0 ∧ y = 0 ∧ z = 0 → 1/((2 - x)*(2 - y)*(2 - z)) + 1/((2 + x)*(2 + y)*(2 + z)) + 1/(1 + (x+y+z)/3) = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1307_130733


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l1307_130723

/-- The time required for a train to cross a bridge -/
theorem train_bridge_crossing_time 
  (train_length : Real) 
  (train_speed_kmh : Real) 
  (bridge_length : Real) 
  (h1 : train_length = 100) 
  (h2 : train_speed_kmh = 45) 
  (h3 : bridge_length = 275) : 
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l1307_130723


namespace NUMINAMATH_CALUDE_distance_calculation_l1307_130770

def point : ℝ × ℝ × ℝ := (2, 3, 4)
def line_point : ℝ × ℝ × ℝ := (5, 8, 6)
def line_direction : ℝ × ℝ × ℝ := (4, 3, -3)

def distance_to_line (p : ℝ × ℝ × ℝ) (l_point : ℝ × ℝ × ℝ) (l_dir : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_calculation :
  distance_to_line point line_point line_direction = Real.sqrt 10458 / 34 := by
  sorry

end NUMINAMATH_CALUDE_distance_calculation_l1307_130770


namespace NUMINAMATH_CALUDE_intersection_chord_length_l1307_130798

/-- Given a line and a circle that intersect to form a chord of length √3, 
    prove that the parameter 'a' in the circle equation is 0. -/
theorem intersection_chord_length (a : ℝ) : 
  (∃ (x y : ℝ), (8*x - 6*y - 3 = 0) ∧ 
                (x^2 + y^2 - 2*x + a = 0) ∧ 
                (∃ (x' y' : ℝ), (x' ≠ x ∨ y' ≠ y) ∧ 
                                (8*x' - 6*y' - 3 = 0) ∧ 
                                (x'^2 + y'^2 - 2*x' + a = 0) ∧ 
                                ((x - x')^2 + (y - y')^2 = 3))) →
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_intersection_chord_length_l1307_130798


namespace NUMINAMATH_CALUDE_chairs_moved_by_pat_l1307_130775

theorem chairs_moved_by_pat (total_chairs : ℕ) (careys_chairs : ℕ) (chairs_left : ℕ) 
  (h1 : total_chairs = 74)
  (h2 : careys_chairs = 28)
  (h3 : chairs_left = 17) :
  total_chairs - careys_chairs - chairs_left = 29 := by
  sorry

end NUMINAMATH_CALUDE_chairs_moved_by_pat_l1307_130775


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1307_130722

theorem complex_modulus_problem (z : ℂ) (h : (1 + Complex.I) * z = (1 - Complex.I)^2) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1307_130722


namespace NUMINAMATH_CALUDE_largest_sum_proof_l1307_130738

theorem largest_sum_proof : 
  let sums : List ℚ := [1/4 + 1/2, 1/4 + 1/9, 1/4 + 1/3, 1/4 + 1/10, 1/4 + 1/6]
  (∀ x ∈ sums, x ≤ 1/4 + 1/2) ∧ (1/4 + 1/2 = 3/4) := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_proof_l1307_130738


namespace NUMINAMATH_CALUDE_multiply_powers_l1307_130762

theorem multiply_powers (x y : ℝ) : 3 * x^2 * (-2 * x * y^3) = -6 * x^3 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_multiply_powers_l1307_130762


namespace NUMINAMATH_CALUDE_walk_time_calculation_l1307_130740

/-- The time it takes Mark to walk into the courthouse each day -/
def walk_time : ℕ := sorry

/-- The number of work days in a week -/
def work_days : ℕ := 5

/-- The time it takes to find parking each day -/
def parking_time : ℕ := 5

/-- The time it takes to get through the metal detector on crowded days -/
def crowded_detector_time : ℕ := 30

/-- The time it takes to get through the metal detector on less crowded days -/
def less_crowded_detector_time : ℕ := 10

/-- The number of crowded days per week -/
def crowded_days : ℕ := 2

/-- The number of less crowded days per week -/
def less_crowded_days : ℕ := 3

/-- The total time spent on all activities in a week -/
def total_weekly_time : ℕ := 130

theorem walk_time_calculation : 
  walk_time = 3 ∧
  work_days * (parking_time + walk_time) + 
  crowded_days * crowded_detector_time +
  less_crowded_days * less_crowded_detector_time = 
  total_weekly_time :=
sorry

end NUMINAMATH_CALUDE_walk_time_calculation_l1307_130740


namespace NUMINAMATH_CALUDE_moms_dimes_l1307_130702

/-- Given the initial number of dimes, the number of dimes given by dad, and the final number of dimes,
    proves that the number of dimes given by mom is 4. -/
theorem moms_dimes (initial : ℕ) (from_dad : ℕ) (final : ℕ)
  (h1 : initial = 7)
  (h2 : from_dad = 8)
  (h3 : final = 19) :
  final - (initial + from_dad) = 4 := by
  sorry

end NUMINAMATH_CALUDE_moms_dimes_l1307_130702


namespace NUMINAMATH_CALUDE_C_power_50_l1307_130757

def C : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; -4, -1]

theorem C_power_50 : C^50 = !![101, 50; -200, -99] := by sorry

end NUMINAMATH_CALUDE_C_power_50_l1307_130757


namespace NUMINAMATH_CALUDE_waiter_income_fraction_l1307_130717

theorem waiter_income_fraction (salary tips income : ℚ) : 
  income = salary + tips → 
  tips = (5 : ℚ) / 3 * salary → 
  tips / income = (5 : ℚ) / 8 := by
sorry

end NUMINAMATH_CALUDE_waiter_income_fraction_l1307_130717


namespace NUMINAMATH_CALUDE_sum_of_radii_l1307_130749

/-- A circle with center C(r, r) is tangent to the positive x-axis and y-axis,
    and externally tangent to a circle centered at (4,0) with radius 2. -/
def CircleTangency (r : ℝ) : Prop :=
  r > 0 ∧ (r - 4)^2 + r^2 = (r + 2)^2

/-- The sum of all possible radii of the circle with center C is 12. -/
theorem sum_of_radii :
  ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ CircleTangency r₁ ∧ CircleTangency r₂ ∧ r₁ + r₂ = 12 :=
sorry

end NUMINAMATH_CALUDE_sum_of_radii_l1307_130749


namespace NUMINAMATH_CALUDE_mrs_hilt_marbles_l1307_130779

/-- Calculates the final number of marbles Mrs. Hilt has -/
def final_marbles (initial lost given_away found : ℕ) : ℕ :=
  initial - lost - given_away + found

/-- Theorem stating that Mrs. Hilt's final number of marbles is correct -/
theorem mrs_hilt_marbles :
  final_marbles 38 15 6 8 = 25 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_marbles_l1307_130779


namespace NUMINAMATH_CALUDE_positive_difference_is_zero_l1307_130736

/-- The quadratic equation from the problem -/
def quadratic_equation (x : ℂ) : Prop :=
  x^2 + 5*x + 20 = 2*x + 16

/-- The solutions of the quadratic equation -/
def solutions : Set ℂ :=
  {x : ℂ | quadratic_equation x}

/-- The positive difference between the solutions is 0 -/
theorem positive_difference_is_zero :
  ∃ (x y : ℂ), x ∈ solutions ∧ y ∈ solutions ∧ x ≠ y ∧ |x.re - y.re| = 0 :=
sorry

end NUMINAMATH_CALUDE_positive_difference_is_zero_l1307_130736


namespace NUMINAMATH_CALUDE_circle_equation_l1307_130716

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the line L: x - y - 1 = 0
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 - 1 = 0}

-- Define points A and B
def A : ℝ × ℝ := (4, 1)
def B : ℝ × ℝ := (2, 1)

-- State the theorem
theorem circle_equation :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    -- The circle passes through point A
    A ∈ Circle center radius ∧
    -- The circle is tangent to the line at point B
    B ∈ Circle center radius ∧
    B ∈ Line ∧
    -- The equation of the circle is (x-3)²+y²=2
    center = (3, 0) ∧ radius^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l1307_130716


namespace NUMINAMATH_CALUDE_square_root_of_two_l1307_130774

theorem square_root_of_two : Real.sqrt 2 = (Real.sqrt 2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_two_l1307_130774


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l1307_130799

def M : Set ℝ := {x | x^2 - x - 12 = 0}
def N : Set ℝ := {x | x^2 + 3*x = 0}

theorem union_of_M_and_N : M ∪ N = {0, -3, 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l1307_130799


namespace NUMINAMATH_CALUDE_list_property_l1307_130719

theorem list_property (numbers : List ℝ) (n : ℝ) : 
  numbers.length = 21 →
  n ∈ numbers →
  n = 5 * ((numbers.sum - n) / 20) →
  n = 0.2 * numbers.sum →
  (numbers.filter (λ x => x ≠ n)).length = 20 := by
sorry

end NUMINAMATH_CALUDE_list_property_l1307_130719


namespace NUMINAMATH_CALUDE_max_games_512_3_l1307_130772

/-- Represents a tournament where players must be defeated three times to be eliminated -/
structure Tournament where
  contestants : ℕ
  defeats_to_eliminate : ℕ

/-- Calculates the maximum number of games that could be played in the tournament -/
def max_games (t : Tournament) : ℕ :=
  (t.contestants - 1) * t.defeats_to_eliminate + 2

/-- Theorem stating that for a tournament with 512 contestants and 3 defeats to eliminate,
    the maximum number of games is 1535 -/
theorem max_games_512_3 :
  let t : Tournament := { contestants := 512, defeats_to_eliminate := 3 }
  max_games t = 1535 := by
  sorry

end NUMINAMATH_CALUDE_max_games_512_3_l1307_130772


namespace NUMINAMATH_CALUDE_triangle_projection_inequality_l1307_130721

/-- Given a triangle ABC with sides a, b, c and projections satisfying certain conditions,
    prove that a specific inequality holds. -/
theorem triangle_projection_inequality 
  (a b c : ℝ) 
  (t r μ : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_t : ∃ (A C₁ : ℝ), A > 0 ∧ C₁ > 0 ∧ C₁ = 2 * t * c) 
  (h_r : ∃ (B A₁ : ℝ), B > 0 ∧ A₁ > 0 ∧ A₁ = 2 * r * a) 
  (h_μ : ∃ (C B₁ : ℝ), C > 0 ∧ B₁ > 0 ∧ B₁ = 2 * μ * b) :
  (a^2 / b^2) * (t / (1 - 2*t))^2 + 
  (b^2 / c^2) * (r / (1 - 2*r))^2 + 
  (c^2 / a^2) * (μ / (1 - 2*μ))^2 + 
  16 * t * r * μ ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_projection_inequality_l1307_130721


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l1307_130718

theorem arithmetic_mean_of_special_set (n : ℕ) (h : n > 1) :
  let s : Finset ℕ := Finset.range n
  let f : ℕ → ℝ := λ i => if i = 0 then 1/n + 2/n^2 else 1
  (s.sum f) / n = 1 + 2/n^2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l1307_130718


namespace NUMINAMATH_CALUDE_complex_power_difference_l1307_130793

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference (h : i^2 = -1) : (1 + i)^18 - (1 - i)^18 = 1024 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l1307_130793


namespace NUMINAMATH_CALUDE_largest_triangle_perimeter_l1307_130728

theorem largest_triangle_perimeter (x : ℤ) : 
  (7 : ℝ) + 11 > (x : ℝ) → 
  (7 : ℝ) + (x : ℝ) > 11 → 
  11 + (x : ℝ) > 7 → 
  (∃ (y : ℤ), (7 : ℝ) + 11 + (y : ℝ) ≥ 7 + 11 + (x : ℝ)) ∧ 
  (7 : ℝ) + 11 + (y : ℝ) ≤ 35 :=
by sorry

end NUMINAMATH_CALUDE_largest_triangle_perimeter_l1307_130728


namespace NUMINAMATH_CALUDE_min_nSn_value_l1307_130797

/-- An arithmetic sequence with sum S_n for the first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  S : ℕ → ℚ  -- The sum function
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0
  sum_formula : ∀ n : ℕ, S n = n * (2 * a 0 + (n - 1) * (a 1 - a 0)) / 2

/-- The main theorem stating the minimum value of nS_n -/
theorem min_nSn_value (seq : ArithmeticSequence) 
    (h1 : seq.S 10 = 0) 
    (h2 : seq.S 15 = 25) : 
  ∃ n : ℕ, ∀ m : ℕ, n * seq.S n ≤ m * seq.S m ∧ n * seq.S n = -49 := by
  sorry

end NUMINAMATH_CALUDE_min_nSn_value_l1307_130797


namespace NUMINAMATH_CALUDE_janelle_blue_marble_bags_l1307_130790

/-- Calculates the number of bags of blue marbles Janelle bought -/
def blue_marble_bags (initial_green : ℕ) (marbles_per_bag : ℕ) (green_given : ℕ) (blue_given : ℕ) (total_remaining : ℕ) : ℕ :=
  ((total_remaining + green_given + blue_given) - initial_green) / marbles_per_bag

/-- Proves that Janelle bought 6 bags of blue marbles -/
theorem janelle_blue_marble_bags :
  blue_marble_bags 26 10 6 8 72 = 6 := by
  sorry

#eval blue_marble_bags 26 10 6 8 72

end NUMINAMATH_CALUDE_janelle_blue_marble_bags_l1307_130790


namespace NUMINAMATH_CALUDE_student_sample_size_l1307_130750

theorem student_sample_size :
  ∀ (T : ℝ) (freshmen sophomores juniors seniors : ℝ),
    -- All students are either freshmen, sophomores, juniors, or seniors
    T = freshmen + sophomores + juniors + seniors →
    -- 27% are juniors
    juniors = 0.27 * T →
    -- 75% are not sophomores (which means 25% are sophomores)
    sophomores = 0.25 * T →
    -- There are 160 seniors
    seniors = 160 →
    -- There are 24 more freshmen than sophomores
    freshmen = sophomores + 24 →
    -- Prove that the total number of students is 800
    T = 800 := by
  sorry

end NUMINAMATH_CALUDE_student_sample_size_l1307_130750


namespace NUMINAMATH_CALUDE_point_on_inverse_graph_and_sum_l1307_130754

-- Define a function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
noncomputable def f_inv : ℝ → ℝ := Function.invFun f

-- State the theorem
theorem point_on_inverse_graph_and_sum (h : f 2 = 9) :
  f_inv 9 = 2 ∧ 9 + (2 / 3) = 29 / 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_inverse_graph_and_sum_l1307_130754


namespace NUMINAMATH_CALUDE_original_cost_price_calculation_l1307_130753

/-- Represents the pricing structure of an article -/
structure ArticlePricing where
  cost_price : ℝ
  discount_rate : ℝ
  tax_rate : ℝ
  profit_rate : ℝ
  selling_price : ℝ

/-- Theorem stating the relationship between the original cost price and final selling price -/
theorem original_cost_price_calculation (a : ArticlePricing)
  (h1 : a.discount_rate = 0.10)
  (h2 : a.tax_rate = 0.05)
  (h3 : a.profit_rate = 0.20)
  (h4 : a.selling_price = 1800)
  : a.cost_price = 1500 := by
  sorry

#check original_cost_price_calculation

end NUMINAMATH_CALUDE_original_cost_price_calculation_l1307_130753


namespace NUMINAMATH_CALUDE_condition_relationship_l1307_130704

theorem condition_relationship (x : ℝ) :
  (1 / x > 1 → x < 1) ∧ ¬(x < 1 → 1 / x > 1) := by
  sorry

end NUMINAMATH_CALUDE_condition_relationship_l1307_130704


namespace NUMINAMATH_CALUDE_factorization_equality_l1307_130710

theorem factorization_equality (a b x y : ℝ) :
  (a*x - b*y)^2 + (a*y + b*x)^2 = (x^2 + y^2) * (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1307_130710


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sin_property_l1307_130742

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sin_property (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 + a 5 + a 9 = 5 * Real.pi →
  Real.sin (a 2 + a 8) = -Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sin_property_l1307_130742


namespace NUMINAMATH_CALUDE_minimum_donut_cost_minimum_donut_cost_proof_l1307_130700

/-- The minimum cost to buy at least 550 donuts, given that they are sold in dozens at $7.49 per dozen -/
theorem minimum_donut_cost : ℝ → Prop :=
  fun cost =>
    ∀ n : ℕ,
      (12 * n ≥ 550) →
      (cost ≤ n * 7.49) ∧
      (∃ m : ℕ, (12 * m ≥ 550) ∧ (cost = m * 7.49)) →
      cost = 344.54

/-- Proof of the minimum_donut_cost theorem -/
theorem minimum_donut_cost_proof : minimum_donut_cost 344.54 := by
  sorry

end NUMINAMATH_CALUDE_minimum_donut_cost_minimum_donut_cost_proof_l1307_130700


namespace NUMINAMATH_CALUDE_square_ratio_problem_l1307_130725

theorem square_ratio_problem :
  let area_ratio : ℚ := 18 / 50
  let side_ratio : ℝ := Real.sqrt (area_ratio)
  ∃ (a b c : ℕ), 
    (a : ℝ) * Real.sqrt b / c = side_ratio ∧
    a = 3 ∧ b = 2 ∧ c = 5 ∧
    a + b + c = 10 :=
by sorry

end NUMINAMATH_CALUDE_square_ratio_problem_l1307_130725


namespace NUMINAMATH_CALUDE_third_discount_percentage_l1307_130758

/-- Given an item with successive discounts, prove the third discount percentage --/
theorem third_discount_percentage
  (original_price : ℝ)
  (first_discount second_discount : ℝ)
  (final_price : ℝ)
  (h1 : original_price = 10000)
  (h2 : first_discount = 20)
  (h3 : second_discount = 10)
  (h4 : final_price = 6840)
  : ∃ (third_discount : ℝ),
    final_price = original_price *
      (1 - first_discount / 100) *
      (1 - second_discount / 100) *
      (1 - third_discount / 100) ∧
    third_discount = 5 := by
  sorry

end NUMINAMATH_CALUDE_third_discount_percentage_l1307_130758


namespace NUMINAMATH_CALUDE_absolute_value_minus_half_power_l1307_130743

theorem absolute_value_minus_half_power : |(-3 : ℝ)| - (1/2 : ℝ)^0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_minus_half_power_l1307_130743


namespace NUMINAMATH_CALUDE_japanese_study_fraction_l1307_130769

theorem japanese_study_fraction (j s : ℝ) (x : ℝ) : 
  s = 3 * j →                           -- Senior class is 3 times the junior class
  ((1/3) * s + x * j) / (s + j) = 0.4375 →  -- 0.4375 fraction of all students study Japanese
  x = 3/4 :=                             -- Fraction of juniors studying Japanese
by
  sorry

end NUMINAMATH_CALUDE_japanese_study_fraction_l1307_130769


namespace NUMINAMATH_CALUDE_initial_distance_is_50_l1307_130706

/-- The initial distance between two people walking towards each other -/
def initial_distance (speed : ℝ) (distance_walked : ℝ) : ℝ :=
  2 * distance_walked

/-- Theorem: The initial distance between Fred and Sam is 50 miles -/
theorem initial_distance_is_50 (fred_speed sam_speed : ℝ) (sam_distance : ℝ) :
  fred_speed = 5 →
  sam_speed = 5 →
  sam_distance = 25 →
  initial_distance sam_speed sam_distance = 50 :=
by
  sorry

#check initial_distance_is_50

end NUMINAMATH_CALUDE_initial_distance_is_50_l1307_130706


namespace NUMINAMATH_CALUDE_ball_selection_problem_l1307_130778

/-- The number of ways to select balls from a bag with red and white balls -/
def select_balls (red : ℕ) (white : ℕ) (total : ℕ) (condition : ℕ → ℕ → Bool) : ℕ :=
  sorry

/-- The total score of selected balls -/
def total_score (red : ℕ) (white : ℕ) : ℕ :=
  sorry

theorem ball_selection_problem :
  let red_balls := 4
  let white_balls := 6
  (select_balls red_balls white_balls 4 (fun r w => r ≥ w) = 115) ∧
  (select_balls red_balls white_balls 5 (fun r w => total_score r w ≥ 7) = 186) :=
by sorry

end NUMINAMATH_CALUDE_ball_selection_problem_l1307_130778


namespace NUMINAMATH_CALUDE_sum_sequence_existence_l1307_130705

theorem sum_sequence_existence (n : ℕ) (h : n ≤ 2^1000000) :
  ∃ (k : ℕ) (x : ℕ → ℕ),
    x 0 = 1 ∧
    k ≤ 1100000 ∧
    x k = n ∧
    ∀ i ∈ Finset.range (k + 1), i ≠ 0 →
      ∃ r s, r ≤ s ∧ s < i ∧ x i = x r + x s :=
by sorry

end NUMINAMATH_CALUDE_sum_sequence_existence_l1307_130705


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1307_130786

theorem quadratic_factorization (a : ℤ) : 
  (∃ m n p q : ℤ, (15 : ℤ) * x^2 + a * x + (15 : ℤ) = (m * x + n) * (p * x + q) ∧ 
   Nat.Prime m.natAbs ∧ Nat.Prime p.natAbs) → 
  ∃ k : ℤ, a = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1307_130786


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l1307_130744

theorem regular_polygon_interior_angle_sum 
  (n : ℕ) 
  (h_exterior : (360 : ℝ) / n = 45) : 
  (n - 2 : ℝ) * 180 = 1080 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l1307_130744


namespace NUMINAMATH_CALUDE_one_correct_statement_l1307_130777

-- Define a sequence as a function from natural numbers to real numbers
def Sequence := ℕ → ℝ

-- Statement 1: A sequence represented graphically appears as a group of isolated points
def graphical_representation (s : Sequence) : Prop :=
  ∀ n : ℕ, ∃ ε > 0, ∀ m : ℕ, m ≠ n → |s m - s n| > ε

-- Statement 2: The terms of a sequence are finite
def finite_terms (s : Sequence) : Prop :=
  ∃ N : ℕ, ∀ n : ℕ, n > N → s n = 0

-- Statement 3: If a sequence is decreasing, then the sequence must be finite
def decreasing_implies_finite (s : Sequence) : Prop :=
  (∀ n : ℕ, s (n + 1) ≤ s n) → finite_terms s

-- Theorem stating that only one of the above statements is correct
theorem one_correct_statement :
  (∀ s : Sequence, graphical_representation s) ∧
  (∃ s : Sequence, ¬finite_terms s) ∧
  (∃ s : Sequence, (∀ n : ℕ, s (n + 1) ≤ s n) ∧ ¬finite_terms s) :=
sorry

end NUMINAMATH_CALUDE_one_correct_statement_l1307_130777


namespace NUMINAMATH_CALUDE_combined_mean_of_two_sets_l1307_130784

theorem combined_mean_of_two_sets (set1_count : ℕ) (set1_mean : ℚ) (set2_count : ℕ) (set2_mean : ℚ) :
  set1_count = 5 →
  set1_mean = 15 →
  set2_count = 8 →
  set2_mean = 20 →
  let total_count := set1_count + set2_count
  let total_sum := set1_count * set1_mean + set2_count * set2_mean
  (total_sum / total_count : ℚ) = 235 / 13 := by
  sorry

end NUMINAMATH_CALUDE_combined_mean_of_two_sets_l1307_130784


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1307_130741

/-- Given a geometric sequence {a_n} with common ratio q, 
    if a_5 - a_1 = 15 and a_4 - a_2 = 6, then q = 1/2 or q = 2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- Definition of geometric sequence
  a 5 - a 1 = 15 →              -- Condition 1
  a 4 - a 2 = 6 →               -- Condition 2
  q = 1/2 ∨ q = 2 :=            -- Conclusion
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1307_130741


namespace NUMINAMATH_CALUDE_three_digit_numbers_from_five_l1307_130759

/-- The number of ways to create a three-digit number using five different single-digit numbers -/
def three_digit_combinations (n : ℕ) (r : ℕ) : ℕ :=
  (n.factorial) / ((r.factorial) * ((n - r).factorial))

/-- The number of permutations of r items -/
def permutations (r : ℕ) : ℕ := r.factorial

theorem three_digit_numbers_from_five : 
  three_digit_combinations 5 3 * permutations 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_numbers_from_five_l1307_130759


namespace NUMINAMATH_CALUDE_train_length_calculation_l1307_130730

/-- Calculates the length of a train given its speed, time to cross a bridge, and the bridge length. -/
theorem train_length_calculation (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 80 * (1000 / 3600) →
  crossing_time = 10.889128869690424 →
  bridge_length = 142 →
  ∃ (train_length : ℝ), abs (train_length - 100.222) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1307_130730


namespace NUMINAMATH_CALUDE_valid_n_characterization_l1307_130746

def is_valid_n (n : ℕ) : Prop :=
  ∃ (k : ℤ), (37.5^n + 26.5^n : ℝ) = k ∧ k > 0

theorem valid_n_characterization :
  ∀ n : ℕ, is_valid_n n ↔ n = 1 ∨ n = 3 ∨ n = 5 ∨ n = 7 :=
by sorry

end NUMINAMATH_CALUDE_valid_n_characterization_l1307_130746


namespace NUMINAMATH_CALUDE_special_rectangle_area_l1307_130781

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  /-- The length of the rectangle -/
  length : ℝ
  /-- The width of the rectangle -/
  width : ℝ
  /-- The distance from the intersection of diagonals to the shorter side -/
  diag_dist : ℝ
  /-- Condition: The distance from the intersection of diagonals to the longer side is 2 cm more than to the shorter side -/
  diag_dist_diff : diag_dist + 2 = length / 2
  /-- Condition: The perimeter of the rectangle is 56 cm -/
  perimeter_cond : 2 * (length + width) = 56

/-- The area of a SpecialRectangle is 192 cm² -/
theorem special_rectangle_area (r : SpecialRectangle) : r.length * r.width = 192 := by
  sorry

end NUMINAMATH_CALUDE_special_rectangle_area_l1307_130781


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_seven_l1307_130760

theorem largest_n_divisible_by_seven (n : ℕ) : n < 100000 ∧ 
  (∃ k : ℤ, 6 * (n - 3)^5 - n^2 + 16*n - 36 = 7 * k) →
  n ≤ 99996 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_seven_l1307_130760


namespace NUMINAMATH_CALUDE_yellow_bags_count_l1307_130785

/-- Represents the number of marbles in each type of bag -/
def marbles_per_bag : Fin 3 → ℕ
  | 0 => 10  -- Red bags
  | 1 => 50  -- Blue bags
  | 2 => 100 -- Yellow bags
  | _ => 0   -- This case is unreachable due to Fin 3

/-- The total number of bags -/
def total_bags : ℕ := 12

/-- The total number of marbles -/
def total_marbles : ℕ := 500

theorem yellow_bags_count :
  ∃ (red blue yellow : ℕ),
    red + blue + yellow = total_bags ∧
    red * marbles_per_bag 0 + blue * marbles_per_bag 1 + yellow * marbles_per_bag 2 = total_marbles ∧
    red = blue ∧
    yellow = 2 := by sorry

end NUMINAMATH_CALUDE_yellow_bags_count_l1307_130785


namespace NUMINAMATH_CALUDE_prove_d_value_l1307_130726

def floor_d : ℤ := -9

def frac_d : ℚ := 2/5

theorem prove_d_value :
  let d : ℚ := floor_d + frac_d
  (3 * floor_d^2 + 14 * floor_d - 45 = 0) ∧
  (5 * frac_d^2 - 18 * frac_d + 8 = 0) ∧
  (0 ≤ frac_d ∧ frac_d < 1) →
  d = -43/5 := by sorry

end NUMINAMATH_CALUDE_prove_d_value_l1307_130726


namespace NUMINAMATH_CALUDE_fib_matrix_power_eq_fib_relation_l1307_130752

/-- Fibonacci matrix -/
def fib_matrix : Matrix (Fin 2) (Fin 2) ℕ := !![1, 1; 1, 0]

/-- n-th power of Fibonacci matrix -/
def fib_matrix_power (n : ℕ) : Matrix (Fin 2) (Fin 2) ℕ := fib_matrix ^ n

/-- n-th Fibonacci number -/
def F (n : ℕ) : ℕ := (fib_matrix_power n) 0 1

/-- Theorem stating the relation between Fibonacci numbers and matrix power -/
theorem fib_matrix_power_eq (n : ℕ) :
  fib_matrix_power n = !![F (n + 1), F n; F n, F (n - 1)] := by sorry

/-- Main theorem to prove -/
theorem fib_relation :
  F 1001 * F 1003 - F 1002 * F 1002 = 1 := by sorry

end NUMINAMATH_CALUDE_fib_matrix_power_eq_fib_relation_l1307_130752


namespace NUMINAMATH_CALUDE_sandwich_combinations_l1307_130787

theorem sandwich_combinations (n_meat : Nat) (n_cheese : Nat) : 
  n_meat = 10 → n_cheese = 9 → n_meat * (n_cheese.choose 2) = 360 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l1307_130787


namespace NUMINAMATH_CALUDE_odd_digits_in_560_base9_l1307_130755

/-- Converts a natural number from base 10 to base 9 --/
def toBase9 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of odd digits in a list of natural numbers --/
def countOddDigits (digits : List ℕ) : ℕ :=
  sorry

theorem odd_digits_in_560_base9 :
  countOddDigits (toBase9 560) = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_digits_in_560_base9_l1307_130755


namespace NUMINAMATH_CALUDE_divisors_of_2744_l1307_130748

-- Define 2744 as the number we're interested in
def n : ℕ := 2744

-- Define the function that counts the number of positive divisors
def count_divisors (m : ℕ) : ℕ := (Finset.filter (· ∣ m) (Finset.range (m + 1))).card

-- State the theorem
theorem divisors_of_2744 : count_divisors n = 16 := by sorry

end NUMINAMATH_CALUDE_divisors_of_2744_l1307_130748


namespace NUMINAMATH_CALUDE_inequality_proof_l1307_130715

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  Real.sqrt ((a + c)^2 + (b + d)^2) ≤ Real.sqrt (a^2 + b^2) + Real.sqrt (c^2 + d^2) ∧
  Real.sqrt (a^2 + b^2) + Real.sqrt (c^2 + d^2) ≤ Real.sqrt ((a + c)^2 + (b + d)^2) + (2 * |a * d - b * c|) / Real.sqrt ((a + c)^2 + (b + d)^2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1307_130715


namespace NUMINAMATH_CALUDE_empty_set_condition_single_element_condition_single_element_values_l1307_130727

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 - 3 * x + 2 = 0}

-- Theorem for the empty set condition
theorem empty_set_condition (a : ℝ) : A a = ∅ ↔ a > 9/8 := by sorry

-- Theorem for the single element condition
theorem single_element_condition (a : ℝ) : 
  (∃! x, x ∈ A a) ↔ (a = 0 ∨ a = 9/8) := by sorry

-- Theorem for the specific elements when A has a single element
theorem single_element_values (a : ℝ) :
  (∃! x, x ∈ A a) → 
  ((a = 0 → A a = {2/3}) ∧ (a = 9/8 → A a = {4/3})) := by sorry

end NUMINAMATH_CALUDE_empty_set_condition_single_element_condition_single_element_values_l1307_130727


namespace NUMINAMATH_CALUDE_line_slope_and_intercept_l1307_130771

/-- Given a line expressed as (3, -4) · ((x, y) - (-2, 8)) = 0, 
    prove that its slope is 3/4 and its y-intercept is 9.5 -/
theorem line_slope_and_intercept :
  let line := fun (x y : ℝ) => 3 * (x + 2) + (-4) * (y - 8) = 0
  ∃ (m b : ℝ), m = 3/4 ∧ b = 9.5 ∧ ∀ x y, line x y ↔ y = m * x + b :=
by sorry

end NUMINAMATH_CALUDE_line_slope_and_intercept_l1307_130771


namespace NUMINAMATH_CALUDE_erdos_szekeres_l1307_130713

theorem erdos_szekeres (m n : ℕ) (seq : Fin (m * n + 1) → ℝ) :
  (∃ (subseq : Fin (m + 1) → Fin (m * n + 1)),
    (∀ i j, i < j → seq (subseq i) ≤ seq (subseq j))) ∨
  (∃ (subseq : Fin (n + 1) → Fin (m * n + 1)),
    (∀ i j, i < j → seq (subseq i) ≥ seq (subseq j))) :=
sorry

end NUMINAMATH_CALUDE_erdos_szekeres_l1307_130713


namespace NUMINAMATH_CALUDE_toy_store_inventory_l1307_130761

/-- Calculates the final number of games in a toy store's inventory --/
theorem toy_store_inventory (initial : ℕ) (sold : ℕ) (received : ℕ) :
  initial = 95 →
  sold = 68 →
  received = 47 →
  initial - sold + received = 74 := by
  sorry

end NUMINAMATH_CALUDE_toy_store_inventory_l1307_130761


namespace NUMINAMATH_CALUDE_intersecting_circles_radius_l1307_130763

/-- Two circles on a plane where each passes through the center of the other -/
structure IntersectingCircles where
  O₁ : ℝ × ℝ  -- Center of first circle
  O₂ : ℝ × ℝ  -- Center of second circle
  A : ℝ × ℝ   -- First intersection point
  B : ℝ × ℝ   -- Second intersection point
  radius : ℝ  -- Common radius of both circles
  passes_through_center : dist O₁ O₂ = radius
  on_circle : dist O₁ A = radius ∧ dist O₂ A = radius ∧ dist O₁ B = radius ∧ dist O₂ B = radius

/-- The area of a quadrilateral given its four vertices -/
def quadrilateralArea (p₁ p₂ p₃ p₄ : ℝ × ℝ) : ℝ := sorry

theorem intersecting_circles_radius 
  (circles : IntersectingCircles) 
  (area_condition : quadrilateralArea circles.O₁ circles.A circles.O₂ circles.B = 2 * Real.sqrt 3) :
  circles.radius = 2 := by sorry

end NUMINAMATH_CALUDE_intersecting_circles_radius_l1307_130763


namespace NUMINAMATH_CALUDE_volleyball_team_combinations_l1307_130737

def total_players : ℕ := 16
def num_triplets : ℕ := 3
def num_twins : ℕ := 2
def starters : ℕ := 6

def choose_two_triplets : ℕ := Nat.choose num_triplets 2
def remaining_after_triplets : ℕ := total_players - num_triplets + 1
def choose_rest_with_triplets : ℕ := Nat.choose remaining_after_triplets (starters - 2)

def choose_twins : ℕ := 1
def remaining_after_twins : ℕ := total_players - num_twins
def choose_rest_with_twins : ℕ := Nat.choose remaining_after_twins (starters - 2)

theorem volleyball_team_combinations :
  choose_two_triplets * choose_rest_with_triplets + choose_twins * choose_rest_with_twins = 3146 :=
sorry

end NUMINAMATH_CALUDE_volleyball_team_combinations_l1307_130737


namespace NUMINAMATH_CALUDE_no_integer_pairs_with_square_diff_30_l1307_130745

theorem no_integer_pairs_with_square_diff_30 :
  ¬∃ (m n : ℕ), m ≥ n ∧ m * m - n * n = 30 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_pairs_with_square_diff_30_l1307_130745


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1307_130792

theorem complex_equation_solution (z : ℂ) : z * (2 - I) = 11 + 7 * I → z = 3 + 5 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1307_130792


namespace NUMINAMATH_CALUDE_cookie_distribution_l1307_130732

/-- Given the number of chocolate chip and oatmeal raisin cookies, 
    and the distribution rule, determine the number of guests. -/
theorem cookie_distribution (choc_chip : ℕ) (oatmeal : ℕ) : 
  choc_chip = 22 → oatmeal = 18 → 
  (∀ n : ℕ, n > 0 → 3 * n ≤ choc_chip ∧ 2 * n ≤ oatmeal → n ≤ 9) ∧
  (∃ n : ℕ, n > 0 ∧ 3 * n ≤ choc_chip ∧ 2 * n ≤ oatmeal ∧ n = 9) :=
by sorry

end NUMINAMATH_CALUDE_cookie_distribution_l1307_130732


namespace NUMINAMATH_CALUDE_or_necessary_not_sufficient_for_and_l1307_130712

theorem or_necessary_not_sufficient_for_and (p q : Prop) :
  (∀ (p q : Prop), p ∧ q → p ∨ q) ∧
  (∃ (p q : Prop), p ∨ q ∧ ¬(p ∧ q)) :=
by sorry

end NUMINAMATH_CALUDE_or_necessary_not_sufficient_for_and_l1307_130712


namespace NUMINAMATH_CALUDE_staircase_perimeter_l1307_130729

/-- Represents a staircase-shaped region -/
structure StaircaseRegion where
  congruent_sides : ℕ
  rectangle_length : ℝ
  area : ℝ

/-- The perimeter of a staircase-shaped region -/
def perimeter (region : StaircaseRegion) : ℝ :=
  sorry

/-- Theorem stating the perimeter of the specific staircase region -/
theorem staircase_perimeter : 
  ∀ (region : StaircaseRegion), 
    region.congruent_sides = 12 ∧ 
    region.rectangle_length = 12 ∧ 
    region.area = 85 → 
    perimeter region = 41 :=
by sorry

end NUMINAMATH_CALUDE_staircase_perimeter_l1307_130729


namespace NUMINAMATH_CALUDE_greatest_common_divisor_under_60_l1307_130703

theorem greatest_common_divisor_under_60 : ∃ (d : ℕ), d = 36 ∧ 
  d ∣ 468 ∧ d ∣ 108 ∧ d < 60 ∧ 
  ∀ (x : ℕ), x ∣ 468 ∧ x ∣ 108 ∧ x < 60 → x ≤ d :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_under_60_l1307_130703


namespace NUMINAMATH_CALUDE_total_money_l1307_130747

def sam_money : ℕ := 38
def erica_money : ℕ := 53

theorem total_money : sam_money + erica_money = 91 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l1307_130747


namespace NUMINAMATH_CALUDE_parallelogram_diagonal_intersection_l1307_130796

/-- Given a parallelogram with opposite vertices (2, -3) and (14, 9),
    the diagonals intersect at the point (8, 3). -/
theorem parallelogram_diagonal_intersection :
  let a : ℝ × ℝ := (2, -3)
  let b : ℝ × ℝ := (14, 9)
  let midpoint : ℝ × ℝ := ((a.1 + b.1) / 2, (a.2 + b.2) / 2)
  midpoint = (8, 3) := by sorry

end NUMINAMATH_CALUDE_parallelogram_diagonal_intersection_l1307_130796


namespace NUMINAMATH_CALUDE_total_miles_is_35_l1307_130707

def andrew_daily_miles : ℕ := 2
def peter_extra_miles : ℕ := 3
def days : ℕ := 5

def total_miles : ℕ := 
  (andrew_daily_miles * days) + ((andrew_daily_miles + peter_extra_miles) * days)

theorem total_miles_is_35 : total_miles = 35 := by
  sorry

end NUMINAMATH_CALUDE_total_miles_is_35_l1307_130707


namespace NUMINAMATH_CALUDE_sunflower_height_l1307_130766

/-- The height of sunflowers from Packet B in inches -/
def height_B : ℝ := 160

/-- The percentage difference between Packet A and Packet B sunflowers -/
def percentage_difference : ℝ := 0.2

/-- The height of sunflowers from Packet A in inches -/
def height_A : ℝ := height_B * (1 + percentage_difference)

theorem sunflower_height : height_A = 192 := by
  sorry

end NUMINAMATH_CALUDE_sunflower_height_l1307_130766


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l1307_130701

theorem inequality_proof (a b c d : ℝ) 
  (ha : 1 ≤ a ∧ a ≤ 2) 
  (hb : 1 ≤ b ∧ b ≤ 2) 
  (hc : 1 ≤ c ∧ c ≤ 2) 
  (hd : 1 ≤ d ∧ d ≤ 2) : 
  (a + b) / (b + c) + (c + d) / (d + a) ≤ 4 * (a + c) / (b + d) :=
sorry

theorem equality_condition (a b c d : ℝ) 
  (ha : 1 ≤ a ∧ a ≤ 2) 
  (hb : 1 ≤ b ∧ b ≤ 2) 
  (hc : 1 ≤ c ∧ c ≤ 2) 
  (hd : 1 ≤ d ∧ d ≤ 2) :
  (a + b) / (b + c) + (c + d) / (d + a) = 4 * (a + c) / (b + d) ↔ a = b ∧ b = c ∧ c = d :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l1307_130701
