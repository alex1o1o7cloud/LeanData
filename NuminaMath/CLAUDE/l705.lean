import Mathlib

namespace NUMINAMATH_CALUDE_orchard_theorem_l705_70545

/-- Represents an apple orchard with Fuji and Gala trees -/
structure Orchard where
  total : ℕ
  pure_fuji : ℕ
  pure_gala : ℕ
  cross_pollinated : ℕ

/-- The conditions of the orchard as described in the problem -/
def orchard_conditions (o : Orchard) : Prop :=
  o.cross_pollinated = o.total / 10 ∧
  o.pure_fuji = (o.total * 3) / 4 ∧
  o.pure_gala = 42 ∧
  o.total = o.pure_fuji + o.pure_gala + o.cross_pollinated

/-- The theorem stating that under the given conditions, 
    the number of pure Fuji plus cross-pollinated trees is 238 -/
theorem orchard_theorem (o : Orchard) 
  (h : orchard_conditions o) : o.pure_fuji + o.cross_pollinated = 238 := by
  sorry

end NUMINAMATH_CALUDE_orchard_theorem_l705_70545


namespace NUMINAMATH_CALUDE_framed_painting_ratio_l705_70560

/-- Represents the dimensions and frame properties of a painting --/
structure FramedPainting where
  width : ℝ
  height : ℝ
  side_frame_width : ℝ

/-- Calculates the dimensions of the framed painting --/
def framedDimensions (p : FramedPainting) : ℝ × ℝ :=
  (p.width + 2 * p.side_frame_width, p.height + 6 * p.side_frame_width)

/-- Calculates the area of the framed painting --/
def framedArea (p : FramedPainting) : ℝ :=
  let (w, h) := framedDimensions p
  w * h

/-- Calculates the area of the original painting --/
def paintingArea (p : FramedPainting) : ℝ :=
  p.width * p.height

/-- Theorem statement for the framed painting problem --/
theorem framed_painting_ratio (p : FramedPainting)
  (h1 : p.width = 20)
  (h2 : p.height = 30)
  (h3 : framedArea p = 2 * paintingArea p) :
  let (w, h) := framedDimensions p
  w / h = 1 / 2 := by
  sorry

#check framed_painting_ratio

end NUMINAMATH_CALUDE_framed_painting_ratio_l705_70560


namespace NUMINAMATH_CALUDE_three_common_tangents_implies_a_equals_9_l705_70553

/-- Circle M with equation x^2 + y^2 - 4x + 3 = 0 -/
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3 = 0

/-- Another circle with equation x^2 + y^2 - 4x - 6y + a = 0 -/
def other_circle (x y a : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y + a = 0

/-- Theorem stating that if circle M has exactly three common tangent lines
    with the other circle, then a = 9 -/
theorem three_common_tangents_implies_a_equals_9 :
  ∀ a : ℝ, (∃! (l₁ l₂ l₃ : ℝ → ℝ → Prop),
    (∀ x y, circle_M x y → (l₁ x y ∨ l₂ x y ∨ l₃ x y)) ∧
    (∀ x y, other_circle x y a → (l₁ x y ∨ l₂ x y ∨ l₃ x y))) →
  a = 9 :=
sorry

end NUMINAMATH_CALUDE_three_common_tangents_implies_a_equals_9_l705_70553


namespace NUMINAMATH_CALUDE_eventually_constant_l705_70506

/-- The set of positive integers -/
def PositiveInts : Set ℕ := {n : ℕ | n > 0}

/-- The winning set for (n,S)-nim game -/
def winning_set (S : Set ℕ) : Set ℕ :=
  {n : ℕ | ∃ (strategy : ℕ → ℕ), ∀ (m : ℕ), m < n → strategy m ∈ S ∧ strategy m ≤ n}

/-- The function f that maps a set S to its winning set -/
def f (S : Set ℕ) : Set ℕ := winning_set S

/-- Iterate f k times -/
def iterate_f (S : Set ℕ) : ℕ → Set ℕ
  | 0 => S
  | k + 1 => f (iterate_f S k)

/-- The main theorem: the sequence of iterations of f eventually becomes constant -/
theorem eventually_constant (T : Set ℕ) : ∃ (k : ℕ), iterate_f T k = iterate_f T (k + 1) := by
  sorry

end NUMINAMATH_CALUDE_eventually_constant_l705_70506


namespace NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l705_70572

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_tenth_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + a 19 = -18) :
  a 10 = -9 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l705_70572


namespace NUMINAMATH_CALUDE_original_sum_is_600_l705_70522

/-- Simple interest calculation function -/
def simpleInterest (principal rate time : ℝ) : ℝ :=
  principal * (1 + rate * time)

theorem original_sum_is_600 (P R : ℝ) 
  (h1 : simpleInterest P R 2 = 720)
  (h2 : simpleInterest P R 7 = 1020) : 
  P = 600 := by
  sorry

end NUMINAMATH_CALUDE_original_sum_is_600_l705_70522


namespace NUMINAMATH_CALUDE_largest_attendance_difference_largest_attendance_difference_holds_l705_70518

/-- The largest possible difference between attendances in Chicago and Detroit --/
theorem largest_attendance_difference : ℝ → Prop :=
  fun max_diff =>
  ∀ (chicago_actual detroit_actual : ℝ),
  (chicago_actual ≥ 80000 * 0.95 ∧ chicago_actual ≤ 80000 * 1.05) →
  (detroit_actual ≥ 95000 / 1.15 ∧ detroit_actual ≤ 95000 / 0.85) →
  max_diff = 36000 ∧
  ∀ (diff : ℝ),
  diff ≤ detroit_actual - chicago_actual →
  ⌊diff / 1000⌋ * 1000 ≤ max_diff

/-- The theorem holds --/
theorem largest_attendance_difference_holds :
  largest_attendance_difference 36000 := by sorry

end NUMINAMATH_CALUDE_largest_attendance_difference_largest_attendance_difference_holds_l705_70518


namespace NUMINAMATH_CALUDE_mountain_climb_speeds_l705_70567

theorem mountain_climb_speeds (V₁ V₂ V k m n : ℝ) 
  (hpos : V₁ > 0 ∧ V₂ > 0 ∧ V > 0 ∧ k > 0 ∧ m > 0 ∧ n > 0)
  (hV₂ : V₂ = k * V₁)
  (hVm : V = m * V₁)
  (hVn : V = n * V₂) : 
  m = 2 * k / (1 + k) ∧ m + n = 2 := by
  sorry

end NUMINAMATH_CALUDE_mountain_climb_speeds_l705_70567


namespace NUMINAMATH_CALUDE_total_money_is_84_l705_70516

/-- Represents the money redistribution process among three people. -/
def redistribute (j a t : ℚ) : Prop :=
  ∃ (j₁ a₁ t₁ j₂ a₂ t₂ j₃ a₃ t₃ : ℚ),
    -- Step 1: Jan's redistribution
    j₁ + a₁ + t₁ = j + a + t ∧
    a₁ = 2 * a ∧
    t₁ = 2 * t ∧
    -- Step 2: Toy's redistribution
    j₂ + a₂ + t₂ = j₁ + a₁ + t₁ ∧
    j₂ = 2 * j₁ ∧
    a₂ = 2 * a₁ ∧
    -- Step 3: Amy's redistribution
    j₃ + a₃ + t₃ = j₂ + a₂ + t₂ ∧
    j₃ = 2 * j₂ ∧
    t₃ = 2 * t₂

/-- The main theorem stating the total amount of money. -/
theorem total_money_is_84 :
  ∀ j a t : ℚ, t = 48 → redistribute j a t → j + a + t = 84 :=
by sorry

end NUMINAMATH_CALUDE_total_money_is_84_l705_70516


namespace NUMINAMATH_CALUDE_cube_root_of_three_times_two_to_fifth_l705_70503

theorem cube_root_of_three_times_two_to_fifth (x : ℝ) : 
  x^3 = 2^5 + 2^5 + 2^5 → x = 6 * 6^(2/3) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_of_three_times_two_to_fifth_l705_70503


namespace NUMINAMATH_CALUDE_movie_shelf_distribution_l705_70509

/-- The number of shelves in a movie store given the following conditions:
  * There are 9 movies in total
  * The owner wants to distribute the movies evenly among the shelves
  * The owner needs 1 more movie to achieve an even distribution
-/
def numShelves : ℕ := 4

theorem movie_shelf_distribution (total_movies : ℕ) (movies_needed : ℕ) : 
  total_movies = 9 → movies_needed = 1 → numShelves = 4 := by
  sorry

#check movie_shelf_distribution

end NUMINAMATH_CALUDE_movie_shelf_distribution_l705_70509


namespace NUMINAMATH_CALUDE_butterfly_ratio_is_three_to_one_l705_70507

/-- The ratio of time a butterfly spends as a larva to the time spent in a cocoon -/
def butterfly_development_ratio (total_time cocoon_time : ℕ) : ℚ :=
  (total_time - cocoon_time : ℚ) / cocoon_time

/-- Theorem stating that for a butterfly with 120 days total development time and 30 days in cocoon,
    the ratio of time spent as a larva to time in cocoon is 3:1 -/
theorem butterfly_ratio_is_three_to_one :
  butterfly_development_ratio 120 30 = 3 := by
  sorry

end NUMINAMATH_CALUDE_butterfly_ratio_is_three_to_one_l705_70507


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l705_70565

/-- Simplification of a complex expression involving square roots and exponents -/
theorem simplify_sqrt_expression :
  let x := Real.sqrt 3
  (x - 1) ^ (1 - Real.sqrt 2) / (x + 1) ^ (1 + Real.sqrt 2) = 4 - 2 * x :=
by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l705_70565


namespace NUMINAMATH_CALUDE_sqrt_division_property_l705_70535

theorem sqrt_division_property (x : ℝ) (hx : x > 0) : 2 * Real.sqrt x / Real.sqrt x = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_division_property_l705_70535


namespace NUMINAMATH_CALUDE_triangle_side_length_l705_70523

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a = 3 →
  C = 2 * π / 3 →
  S = (15 * Real.sqrt 3) / 4 →
  S = (1 / 2) * a * b * Real.sin C →
  c^2 = a^2 + b^2 - 2 * a * b * Real.cos C →
  c = 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l705_70523


namespace NUMINAMATH_CALUDE_min_value_of_g_l705_70555

-- Define the properties of odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def IsEven (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- State the theorem
theorem min_value_of_g (f g : ℝ → ℝ) 
  (h_odd : IsOdd f) (h_even : IsEven g) 
  (h_sum : ∀ x, f x + g x = 2^x) : 
  ∃ m, m = 1 ∧ ∀ x, g x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_g_l705_70555


namespace NUMINAMATH_CALUDE_solve_system_l705_70527

theorem solve_system (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (eq1 : x * y = 2 * (x + y))
  (eq2 : y * z = 4 * (y + z))
  (eq3 : x * z = 8 * (x + z)) :
  x = 16 / 3 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l705_70527


namespace NUMINAMATH_CALUDE_tank_capacity_l705_70556

theorem tank_capacity : ∀ (T : ℚ),
  (3/4 : ℚ) * T + 7 = (7/8 : ℚ) * T →
  T = 56 := by sorry

end NUMINAMATH_CALUDE_tank_capacity_l705_70556


namespace NUMINAMATH_CALUDE_expected_occurrences_is_two_l705_70570

/-- The probability of event A occurring -/
def prob_A : ℝ := 0.2

/-- The probability of event B occurring -/
def prob_B : ℝ := 0.4

/-- The number of trials -/
def num_trials : ℕ := 25

/-- The expected number of trials where both events occur simultaneously -/
def expected_occurrences : ℝ := num_trials * (prob_A * prob_B)

/-- Theorem stating that the expected number of occurrences is 2 -/
theorem expected_occurrences_is_two : expected_occurrences = 2 := by
  sorry

end NUMINAMATH_CALUDE_expected_occurrences_is_two_l705_70570


namespace NUMINAMATH_CALUDE_naClOConcentrationDecreases_l705_70550

-- Define the disinfectant solution
structure DisinfectantSolution :=
  (volume : ℝ)
  (naClOConcentration : ℝ)
  (density : ℝ)

-- Define the properties of the initial solution
def initialSolution : DisinfectantSolution :=
  { volume := 480,
    naClOConcentration := 0.25,
    density := 1.19 }

-- Define the property that NaClO absorbs H₂O and CO₂ from air and degrades
axiom naClODegrades : ∀ (t : ℝ), t > 0 → ∃ (δ : ℝ), δ > 0 ∧ δ < initialSolution.naClOConcentration

-- Theorem stating that NaClO concentration decreases over time
theorem naClOConcentrationDecreases :
  ∀ (t : ℝ), t > 0 →
  ∃ (s : DisinfectantSolution),
    s.volume = initialSolution.volume ∧
    s.density = initialSolution.density ∧
    s.naClOConcentration < initialSolution.naClOConcentration :=
sorry

end NUMINAMATH_CALUDE_naClOConcentrationDecreases_l705_70550


namespace NUMINAMATH_CALUDE_smallest_product_l705_70569

def digits : List Nat := [5, 6, 7, 8]

def is_valid_arrangement (a b c d : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def product (a b c d : Nat) : Nat := (10 * a + b) * (10 * c + d)

theorem smallest_product :
  ∀ a b c d : Nat, is_valid_arrangement a b c d →
    product a b c d ≥ 3876 :=
by sorry

end NUMINAMATH_CALUDE_smallest_product_l705_70569


namespace NUMINAMATH_CALUDE_range_of_f_l705_70540

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 12

-- Statement to prove
theorem range_of_f :
  Set.range f = Set.Ici 10 := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l705_70540


namespace NUMINAMATH_CALUDE_cafeteria_pies_l705_70521

theorem cafeteria_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) :
  initial_apples = 75 →
  handed_out = 19 →
  apples_per_pie = 8 →
  (initial_apples - handed_out) / apples_per_pie = 7 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l705_70521


namespace NUMINAMATH_CALUDE_truck_travel_distance_l705_70598

/-- Given a truck that travels 300 kilometers on 5 liters of diesel,
    prove that it can travel 420 kilometers on 7 liters of diesel. -/
theorem truck_travel_distance (initial_distance : ℝ) (initial_fuel : ℝ) (new_fuel : ℝ) :
  initial_distance = 300 ∧ initial_fuel = 5 ∧ new_fuel = 7 →
  (initial_distance / initial_fuel) * new_fuel = 420 := by
  sorry

end NUMINAMATH_CALUDE_truck_travel_distance_l705_70598


namespace NUMINAMATH_CALUDE_clips_ratio_april_to_may_l705_70548

def clips_sold_april : ℕ := 48
def total_clips_sold : ℕ := 72

theorem clips_ratio_april_to_may :
  (clips_sold_april : ℚ) / (total_clips_sold - clips_sold_april : ℚ) = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_clips_ratio_april_to_may_l705_70548


namespace NUMINAMATH_CALUDE_classroom_students_l705_70585

/-- The number of pencils in a dozen -/
def pencils_per_dozen : ℕ := 12

/-- The number of dozens of pencils each student gets -/
def dozens_per_student : ℕ := 4

/-- The total number of pencils to be given out -/
def total_pencils : ℕ := 2208

/-- The number of students in the classroom -/
def num_students : ℕ := total_pencils / (dozens_per_student * pencils_per_dozen)

theorem classroom_students :
  num_students = 46 := by sorry

end NUMINAMATH_CALUDE_classroom_students_l705_70585


namespace NUMINAMATH_CALUDE_power_fraction_equality_l705_70562

theorem power_fraction_equality : (27 ^ 20) / (81 ^ 10) = 3 ^ 20 := by sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l705_70562


namespace NUMINAMATH_CALUDE_incorrect_bracket_expansion_l705_70594

theorem incorrect_bracket_expansion : ∀ x : ℝ, 3 * x^2 - 3 * (x + 6) ≠ 3 * x^2 - 3 * x - 6 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_bracket_expansion_l705_70594


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l705_70538

theorem arithmetic_sequence_sum (n : ℕ) (sum : ℕ) (d : ℕ) : 
  n = 4020 →
  d = 2 →
  sum = 10614 →
  ∃ (a : ℕ), 
    (a + (n - 1) * d / 2) * n = sum ∧
    a + (n / 2 - 1) * (2 * d) = 3297 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l705_70538


namespace NUMINAMATH_CALUDE_area_ratio_theorem_l705_70517

/-- Triangle PQR with points X, Y, Z on its sides -/
structure TriangleWithPoints where
  PQ : ℝ
  QR : ℝ
  PR : ℝ
  u : ℝ
  v : ℝ
  w : ℝ

/-- The theorem statement -/
theorem area_ratio_theorem (t : TriangleWithPoints) 
  (h_PQ : t.PQ = 12)
  (h_QR : t.QR = 16)
  (h_PR : t.PR = 20)
  (h_positive : t.u > 0 ∧ t.v > 0 ∧ t.w > 0)
  (h_sum : t.u + t.v + t.w = 3/4)
  (h_sum_squares : t.u^2 + t.v^2 + t.w^2 = 1/2) :
  let area_PQR := (1/2) * t.PQ * t.QR
  let area_XYZ := area_PQR * (1 - (t.u * (1 - t.w) + t.v * (1 - t.u) + t.w * (1 - t.v)))
  area_XYZ / area_PQR = 9/32 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_theorem_l705_70517


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l705_70587

theorem complex_modulus_problem (z : ℂ) (h : (1 - Complex.I) * z = 4 * Complex.I) : 
  Complex.abs z = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l705_70587


namespace NUMINAMATH_CALUDE_least_sum_of_exponential_equality_l705_70583

theorem least_sum_of_exponential_equality (x y z : ℕ+) 
  (h : (2 : ℕ)^(x : ℕ) = (5 : ℕ)^(y : ℕ) ∧ (5 : ℕ)^(y : ℕ) = (8 : ℕ)^(z : ℕ)) : 
  (∀ a b c : ℕ+, (2 : ℕ)^(a : ℕ) = (5 : ℕ)^(b : ℕ) ∧ (5 : ℕ)^(b : ℕ) = (8 : ℕ)^(c : ℕ) → 
    (x : ℕ) + (y : ℕ) + (z : ℕ) ≤ (a : ℕ) + (b : ℕ) + (c : ℕ)) ∧
  (x : ℕ) + (y : ℕ) + (z : ℕ) = 33 :=
by sorry

end NUMINAMATH_CALUDE_least_sum_of_exponential_equality_l705_70583


namespace NUMINAMATH_CALUDE_certain_number_existence_and_uniqueness_l705_70557

theorem certain_number_existence_and_uniqueness :
  ∃! x : ℚ, x / 3 + x + 3 = 63 :=
by sorry

end NUMINAMATH_CALUDE_certain_number_existence_and_uniqueness_l705_70557


namespace NUMINAMATH_CALUDE_f_minimum_at_neg_nine_halves_l705_70524

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 9*x + 7

-- State the theorem
theorem f_minimum_at_neg_nine_halves :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = -9/2 :=
sorry

end NUMINAMATH_CALUDE_f_minimum_at_neg_nine_halves_l705_70524


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l705_70592

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The statement to be proved -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3)^2 - 6*(a 3) + 8 = 0 →
  (a 15)^2 - 6*(a 15) + 8 = 0 →
  a 7 + a 8 + a 9 + a 10 + a 11 = 15 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l705_70592


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l705_70575

theorem lcm_of_ratio_and_hcf (a b : ℕ+) : 
  (a : ℚ) / b = 3 / 4 → Nat.gcd a b = 5 → Nat.lcm a b = 60 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l705_70575


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l705_70581

theorem triangle_angle_problem (a b : ℝ) (B : ℝ) (A : ℝ) :
  a = Real.sqrt 3 →
  b = 1 →
  B = 30 * π / 180 →
  0 < A →
  A < π →
  (A = π / 3 ∨ A = 2 * π / 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l705_70581


namespace NUMINAMATH_CALUDE_factor_implies_m_value_l705_70578

theorem factor_implies_m_value (m : ℝ) : 
  (∀ x : ℝ, ∃ k : ℝ, x^2 - m*x - 40 = (x + 5) * k) → m = 13 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_m_value_l705_70578


namespace NUMINAMATH_CALUDE_basketball_price_correct_l705_70591

/-- The price of a basketball that satisfies the given conditions -/
def basketball_price : ℚ := 29

/-- The number of basketballs bought by Coach A -/
def basketballs_count : ℕ := 10

/-- The price of each baseball -/
def baseball_price : ℚ := 5/2

/-- The number of baseballs bought by Coach B -/
def baseballs_count : ℕ := 14

/-- The price of the baseball bat -/
def bat_price : ℚ := 18

/-- The difference in spending between Coach A and Coach B -/
def spending_difference : ℚ := 237

theorem basketball_price_correct : 
  basketballs_count * basketball_price = 
  (baseballs_count * baseball_price + bat_price + spending_difference) :=
by sorry

end NUMINAMATH_CALUDE_basketball_price_correct_l705_70591


namespace NUMINAMATH_CALUDE_functions_equal_at_three_l705_70546

open Set

-- Define the open interval (2, 4)
def OpenInterval : Set ℝ := {x : ℝ | 2 < x ∧ x < 4}

-- Define the properties of functions f and g
def SatisfiesConditions (f g : ℝ → ℝ) : Prop :=
  ∀ x ∈ OpenInterval,
    (2 < f x ∧ f x < 4) ∧
    (2 < g x ∧ g x < 4) ∧
    (f (g x) = x) ∧
    (g (f x) = x) ∧
    (f x * g x = x^2)

-- Theorem statement
theorem functions_equal_at_three
  (f g : ℝ → ℝ)
  (h : SatisfiesConditions f g) :
  f 3 = g 3 := by
  sorry

end NUMINAMATH_CALUDE_functions_equal_at_three_l705_70546


namespace NUMINAMATH_CALUDE_return_trip_time_l705_70558

/-- Calculates the return trip time given the outbound trip details -/
theorem return_trip_time (outbound_time : ℝ) (outbound_speed : ℝ) (speed_increase : ℝ) : 
  outbound_time = 6 →
  outbound_speed = 60 →
  speed_increase = 12 →
  (outbound_time * outbound_speed) / (outbound_speed + speed_increase) = 5 := by
  sorry

end NUMINAMATH_CALUDE_return_trip_time_l705_70558


namespace NUMINAMATH_CALUDE_circle_area_above_line_l705_70584

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 - 10*x + y^2 - 16*y + 56 = 0

/-- The line equation -/
def line_eq (y : ℝ) : Prop := y = 4

/-- The area of the circle portion above the line -/
noncomputable def area_above_line : ℝ := 99 * Real.pi / 4

/-- Theorem stating that the area of the circle portion above the line is approximately equal to 99π/4 -/
theorem circle_area_above_line :
  ∃ (ε : ℝ), ε > 0 ∧ 
  (∀ x y : ℝ, circle_eq x y → line_eq y → 
    abs (area_above_line - (Real.pi * 33 * 3 / 4)) < ε) :=
sorry

end NUMINAMATH_CALUDE_circle_area_above_line_l705_70584


namespace NUMINAMATH_CALUDE_mary_added_four_peanuts_l705_70511

/-- The number of peanuts Mary added to the box -/
def peanuts_added (initial final : ℕ) : ℕ := final - initial

/-- Theorem stating that Mary added 4 peanuts to the box -/
theorem mary_added_four_peanuts :
  peanuts_added 4 8 = 4 := by sorry

end NUMINAMATH_CALUDE_mary_added_four_peanuts_l705_70511


namespace NUMINAMATH_CALUDE_simple_interest_years_l705_70537

/-- Given a principal amount and the additional interest earned from a 1% rate increase,
    calculate the number of years the sum was put at simple interest. -/
theorem simple_interest_years (principal : ℝ) (additional_interest : ℝ) : 
  principal = 2400 →
  additional_interest = 72 →
  (principal * 0.01 * (3 : ℝ)) = additional_interest :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_years_l705_70537


namespace NUMINAMATH_CALUDE_tangent_and_inequality_l705_70531

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x^3 - x

theorem tangent_and_inequality (m n : ℝ) :
  (∃ (x : ℝ), x = 1 ∧ f m x = n) →  -- Point N(1, n) on the curve
  (∃ (x : ℝ), x = 1 ∧ (deriv (f m)) x = 1) →  -- Tangent with slope 1 (tan(π/4)) at x = 1
  (m = 2/3 ∧ n = -1/3) ∧  -- Part 1 of the theorem
  (∃ (k : ℕ), k = 2008 ∧ ∀ (x : ℝ), x ∈ Set.Icc (-1) 3 → f m x ≤ k - 1993 ∧
    ∀ (k' : ℕ), k' < k → ∃ (x : ℝ), x ∈ Set.Icc (-1) 3 ∧ f m x > k' - 1993) :=
by sorry

end NUMINAMATH_CALUDE_tangent_and_inequality_l705_70531


namespace NUMINAMATH_CALUDE_systematic_sampling_third_selection_l705_70589

theorem systematic_sampling_third_selection
  (total_students : ℕ)
  (selected_students : ℕ)
  (first_selection : ℕ)
  (h1 : total_students = 100)
  (h2 : selected_students = 10)
  (h3 : first_selection = 3)
  : (first_selection + 2 * (total_students / selected_students)) % 100 = 23 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_third_selection_l705_70589


namespace NUMINAMATH_CALUDE_urn_probability_l705_70502

theorem urn_probability (N : ℚ) : 
  let urn1_green : ℚ := 5
  let urn1_blue : ℚ := 7
  let urn2_green : ℚ := 20
  let urn2_blue : ℚ := N
  let total_probability : ℚ := 65/100
  (urn1_green / (urn1_green + urn1_blue)) * (urn2_green / (urn2_green + urn2_blue)) +
  (urn1_blue / (urn1_green + urn1_blue)) * (urn2_blue / (urn2_green + urn2_blue)) = total_probability →
  N = 280/311 := by
sorry

end NUMINAMATH_CALUDE_urn_probability_l705_70502


namespace NUMINAMATH_CALUDE_sandy_initial_money_l705_70536

def sandy_shopping (initial_money : ℝ) : Prop :=
  let watch_price : ℝ := 50
  let shirt_price : ℝ := 30
  let shoes_price : ℝ := 70
  let shirt_discount : ℝ := 0.1
  let shoes_discount : ℝ := 0.2
  let spent_percentage : ℝ := 0.3
  let money_left : ℝ := 210
  
  let total_cost : ℝ := watch_price + 
    shirt_price * (1 - shirt_discount) + 
    shoes_price * (1 - shoes_discount)
  
  (initial_money * spent_percentage = total_cost) ∧
  (initial_money * (1 - spent_percentage) = money_left)

theorem sandy_initial_money :
  sandy_shopping 300 := by sorry

end NUMINAMATH_CALUDE_sandy_initial_money_l705_70536


namespace NUMINAMATH_CALUDE_expression_simplification_l705_70542

theorem expression_simplification (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ -1) :
  (a - (2*a - 1) / a) + (1 - a^2) / (a^2 + a) = (a^2 - 3*a + 2) / a := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l705_70542


namespace NUMINAMATH_CALUDE_only_four_points_l705_70574

/-- A configuration of n points in the plane with associated real numbers -/
structure PointConfiguration (n : ℕ) where
  points : Fin n → ℝ × ℝ
  r : Fin n → ℝ

/-- The area of a triangle given by three points -/
def triangleArea (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := sorry

/-- Three points are collinear if the area of the triangle they form is zero -/
def collinear (p₁ p₂ p₃ : ℝ × ℝ) : Prop :=
  triangleArea p₁ p₂ p₃ = 0

/-- A valid configuration satisfies the problem conditions -/
def validConfiguration {n : ℕ} (config : PointConfiguration n) : Prop :=
  (n > 3) ∧
  (∀ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    ¬collinear (config.points i) (config.points j) (config.points k)) ∧
  (∀ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    triangleArea (config.points i) (config.points j) (config.points k) =
      config.r i + config.r j + config.r k)

/-- The main theorem: The only valid configuration is for n = 4 -/
theorem only_four_points :
  ∀ n : ℕ, (∃ config : PointConfiguration n, validConfiguration config) → n = 4 :=
sorry

end NUMINAMATH_CALUDE_only_four_points_l705_70574


namespace NUMINAMATH_CALUDE_average_of_numbers_between_40_and_80_divisible_by_3_l705_70510

def numbers_between_40_and_80_divisible_by_3 : List ℕ :=
  (List.range 41).filter (λ n => 40 < n ∧ n ≤ 80 ∧ n % 3 = 0)

theorem average_of_numbers_between_40_and_80_divisible_by_3 :
  (List.sum numbers_between_40_and_80_divisible_by_3) / 
  (List.length numbers_between_40_and_80_divisible_by_3) = 63 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_between_40_and_80_divisible_by_3_l705_70510


namespace NUMINAMATH_CALUDE_student_grade_problem_l705_70551

theorem student_grade_problem (grade2 grade3 overall : ℚ) :
  grade2 = 80 →
  grade3 = 75 →
  overall = 75 →
  ∃ grade1 : ℚ, (grade1 + grade2 + grade3) / 3 = overall ∧ grade1 = 70 :=
by sorry

end NUMINAMATH_CALUDE_student_grade_problem_l705_70551


namespace NUMINAMATH_CALUDE_triangular_grid_4_has_17_triangles_l705_70588

/-- Represents a triangular grid with n rows -/
structure TriangularGrid (n : ℕ) where
  rows : Fin n → ℕ
  row_content : ∀ i : Fin n, rows i = i.val + 1

/-- Counts the number of triangles in a triangular grid -/
def count_triangles (grid : TriangularGrid n) : ℕ :=
  sorry

theorem triangular_grid_4_has_17_triangles :
  ∃ (grid : TriangularGrid 4), count_triangles grid = 17 :=
sorry

end NUMINAMATH_CALUDE_triangular_grid_4_has_17_triangles_l705_70588


namespace NUMINAMATH_CALUDE_octagonal_pyramid_volume_l705_70566

/-- The volume of a regular octagonal pyramid with given dimensions -/
theorem octagonal_pyramid_volume :
  ∀ (base_side_length equilateral_face_side_length : ℝ),
    base_side_length = 5 →
    equilateral_face_side_length = 10 →
    ∃ (volume : ℝ),
      volume = (250 * Real.sqrt 3 * (1 + Real.sqrt 2)) / 3 ∧
      volume = (1 / 3) * (2 * (1 + Real.sqrt 2) * base_side_length^2) * 
               ((equilateral_face_side_length * Real.sqrt 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_octagonal_pyramid_volume_l705_70566


namespace NUMINAMATH_CALUDE_fraction_addition_l705_70579

theorem fraction_addition : (2 : ℚ) / 3 + (1 : ℚ) / 6 = (5 : ℚ) / 6 := by sorry

end NUMINAMATH_CALUDE_fraction_addition_l705_70579


namespace NUMINAMATH_CALUDE_hilt_fountain_trips_l705_70561

/-- The number of trips to the water fountain -/
def number_of_trips (distance_to_fountain : ℕ) (total_distance : ℕ) : ℕ :=
  total_distance / distance_to_fountain

/-- Proof that Mrs. Hilt will go to the water fountain 4 times -/
theorem hilt_fountain_trips :
  let distance_to_fountain : ℕ := 30
  let total_distance : ℕ := 120
  number_of_trips distance_to_fountain total_distance = 4 := by
  sorry

end NUMINAMATH_CALUDE_hilt_fountain_trips_l705_70561


namespace NUMINAMATH_CALUDE_company_theorem_l705_70593

-- Define the type for people
variable {Person : Type}

-- Define the "knows" relation
variable (knows : Person → Person → Prop)

-- Define the company as a finite set of people
variable [Finite Person]

-- State the theorem
theorem company_theorem 
  (h : ∀ (S : Finset Person), S.card = 9 → ∃ (x y : Person), x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ knows x y) :
  ∃ (G : Finset Person), G.card = 8 ∧ 
    ∀ p, p ∉ G → ∃ q ∈ G, knows p q :=
sorry

end NUMINAMATH_CALUDE_company_theorem_l705_70593


namespace NUMINAMATH_CALUDE_dihedral_angle_perpendicular_halfplanes_l705_70597

-- Define dihedral angle
def DihedralAngle : Type := sorry

-- Define half-plane of a dihedral angle
def halfPlane (α : DihedralAngle) : Type := sorry

-- Define perpendicularity of half-planes
def perpendicular (p q : Type) : Prop := sorry

-- Define equality of dihedral angles
def equal (α β : DihedralAngle) : Prop := sorry

-- Define complementary dihedral angles
def complementary (α β : DihedralAngle) : Prop := sorry

-- The theorem
theorem dihedral_angle_perpendicular_halfplanes 
  (α β : DihedralAngle) : 
  perpendicular (halfPlane α) (halfPlane β) → 
  equal α β ∨ complementary α β := by sorry

end NUMINAMATH_CALUDE_dihedral_angle_perpendicular_halfplanes_l705_70597


namespace NUMINAMATH_CALUDE_unique_solution_for_system_l705_70596

theorem unique_solution_for_system :
  ∃! (x y z : ℕ+), 
    (x.val : ℤ)^2 + y.val - z.val = 100 ∧ 
    (x.val : ℤ) + y.val^2 - z.val = 124 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_system_l705_70596


namespace NUMINAMATH_CALUDE_six_digit_number_concatenation_divisibility_l705_70512

theorem six_digit_number_concatenation_divisibility :
  ∃ (A B : ℕ), 
    A ≠ B ∧
    100000 ≤ A ∧ A < 1000000 ∧
    100000 ≤ B ∧ B < 1000000 ∧
    (10^6 * B + A) % (A * B) = 0 :=
by sorry

end NUMINAMATH_CALUDE_six_digit_number_concatenation_divisibility_l705_70512


namespace NUMINAMATH_CALUDE_smallest_abc_cba_divisible_by_11_l705_70563

/-- Represents a six-digit number in the form ABC,CBA -/
def AbcCba (a b c : Nat) : Nat :=
  100000 * a + 10000 * b + 1000 * c + 100 * c + 10 * b + a

theorem smallest_abc_cba_divisible_by_11 :
  ∀ a b c : Nat,
    a ≠ b ∧ b ≠ c ∧ a ≠ c →
    0 < a ∧ a < 10 ∧ b < 10 ∧ c < 10 →
    AbcCba a b c ≥ 123321 ∨ ¬(AbcCba a b c % 11 = 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_abc_cba_divisible_by_11_l705_70563


namespace NUMINAMATH_CALUDE_maximum_mark_calculation_l705_70513

def passing_threshold (max_mark : ℝ) : ℝ := 0.33 * max_mark

theorem maximum_mark_calculation (student_marks : ℝ) (failed_by : ℝ) 
  (h1 : student_marks = 125)
  (h2 : failed_by = 40)
  (h3 : passing_threshold (student_marks + failed_by) = student_marks + failed_by) :
  student_marks + failed_by = 500 := by
  sorry

end NUMINAMATH_CALUDE_maximum_mark_calculation_l705_70513


namespace NUMINAMATH_CALUDE_polynomial_factorization_l705_70580

theorem polynomial_factorization (a b m n : ℝ) : 
  (3 * a^2 - 6 * a * b + 3 * b^2 = 3 * (a - b)^2) ∧ 
  (4 * m^2 - 9 * n^2 = (2 * m - 3 * n) * (2 * m + 3 * n)) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l705_70580


namespace NUMINAMATH_CALUDE_f_derivative_at_neg_one_l705_70529

noncomputable def f' (x : ℝ) : ℝ := 2 * Real.exp x / (Real.exp x - 1)

theorem f_derivative_at_neg_one :
  let f (x : ℝ) := (f' (-1)) * Real.exp x - x^2
  (deriv f) (-1) = 2 * Real.exp 1 / (Real.exp 1 - 1) :=
sorry

end NUMINAMATH_CALUDE_f_derivative_at_neg_one_l705_70529


namespace NUMINAMATH_CALUDE_complement_determines_set_l705_70599

def U : Set ℕ := {0, 1, 2, 3}

theorem complement_determines_set (M : Set ℕ) (h : Set.compl M = {2}) : M = {0, 1, 3} := by
  sorry

end NUMINAMATH_CALUDE_complement_determines_set_l705_70599


namespace NUMINAMATH_CALUDE_graph_is_two_lines_factored_is_two_lines_graph_consists_of_two_intersecting_lines_l705_70525

/-- The equation of the graph -/
def graph_equation (x y : ℝ) : Prop := x * y - 2 * x + 3 * y - 6 = 0

/-- The factored form of the equation -/
def factored_equation (x y : ℝ) : Prop := (x + 3) * (y - 2) = 0

/-- Theorem stating that the graph equation is equivalent to the factored equation -/
theorem graph_is_two_lines :
  ∀ x y : ℝ, graph_equation x y ↔ factored_equation x y :=
by sorry

/-- Theorem stating that the factored equation represents two intersecting lines -/
theorem factored_is_two_lines :
  ∃ a b : ℝ, ∀ x y : ℝ, factored_equation x y ↔ (x = a ∨ y = b) :=
by sorry

/-- Main theorem proving that the graph consists of two intersecting lines -/
theorem graph_consists_of_two_intersecting_lines :
  ∃ a b : ℝ, ∀ x y : ℝ, graph_equation x y ↔ (x = a ∨ y = b) :=
by sorry

end NUMINAMATH_CALUDE_graph_is_two_lines_factored_is_two_lines_graph_consists_of_two_intersecting_lines_l705_70525


namespace NUMINAMATH_CALUDE_solve_system_l705_70559

theorem solve_system (x y : ℤ) (h1 : x + y = 260) (h2 : x - y = 200) : y = 30 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l705_70559


namespace NUMINAMATH_CALUDE_john_candies_proof_l705_70595

/-- The number of candies John has -/
def john_candies : ℕ := 35

/-- The number of candies Mark has -/
def mark_candies : ℕ := 30

/-- The number of candies Peter has -/
def peter_candies : ℕ := 25

/-- The number of candies each person has after sharing equally -/
def shared_candies : ℕ := 30

/-- The number of people sharing the candies -/
def num_people : ℕ := 3

theorem john_candies_proof :
  john_candies = shared_candies * num_people - mark_candies - peter_candies :=
by sorry

end NUMINAMATH_CALUDE_john_candies_proof_l705_70595


namespace NUMINAMATH_CALUDE_cut_rectangle_perimeter_example_l705_70573

/-- The perimeter of a rectangle with squares cut from its corners -/
def cut_rectangle_perimeter (length width cut : ℝ) : ℝ :=
  2 * (length + width)

/-- Theorem: The perimeter of a 12x5 cm rectangle with 2x2 cm squares cut from each corner is 34 cm -/
theorem cut_rectangle_perimeter_example :
  cut_rectangle_perimeter 12 5 2 = 34 := by
  sorry

end NUMINAMATH_CALUDE_cut_rectangle_perimeter_example_l705_70573


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l705_70528

/-- Two vectors in ℝ² are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (x, 2)
  let b : ℝ × ℝ := (1, 6)
  parallel a b → x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l705_70528


namespace NUMINAMATH_CALUDE_company_production_l705_70508

/-- Calculates the total number of bottles produced in one day given the number of cases and bottles per case. -/
def bottles_per_day (cases : ℕ) (bottles_per_case : ℕ) : ℕ :=
  cases * bottles_per_case

/-- Theorem stating that given the specific conditions, the company produces 72,000 bottles per day. -/
theorem company_production : bottles_per_day 7200 10 = 72000 := by
  sorry

end NUMINAMATH_CALUDE_company_production_l705_70508


namespace NUMINAMATH_CALUDE_sum_equals_square_l705_70586

theorem sum_equals_square (k : ℕ) (N : ℕ) : N < 100 →
  (k * (k + 1)) / 2 = N^2 ↔ k = 1 ∨ k = 8 ∨ k = 49 := by sorry

end NUMINAMATH_CALUDE_sum_equals_square_l705_70586


namespace NUMINAMATH_CALUDE_pizza_slices_per_pizza_l705_70547

theorem pizza_slices_per_pizza 
  (num_people : ℕ) 
  (slices_per_person : ℕ) 
  (num_pizzas : ℕ) 
  (h1 : num_people = 10) 
  (h2 : slices_per_person = 2) 
  (h3 : num_pizzas = 5) : 
  (num_people * slices_per_person) / num_pizzas = 4 :=
by sorry

end NUMINAMATH_CALUDE_pizza_slices_per_pizza_l705_70547


namespace NUMINAMATH_CALUDE_brick_surface_area_l705_70501

theorem brick_surface_area :
  let length : ℝ := 8
  let width : ℝ := 4
  let height : ℝ := 2
  let surface_area := 2 * (length * width + length * height + width * height)
  surface_area = 112 :=
by sorry

end NUMINAMATH_CALUDE_brick_surface_area_l705_70501


namespace NUMINAMATH_CALUDE_line_intersects_circle_l705_70539

/-- Given a point M(a, b) outside the unit circle, prove that the line ax + by = 1 intersects the circle -/
theorem line_intersects_circle (a b : ℝ) (h : a^2 + b^2 > 1) :
  ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ a * x + b * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l705_70539


namespace NUMINAMATH_CALUDE_min_value_of_a_l705_70577

theorem min_value_of_a (a : ℝ) : 
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 → 1/x + a/y ≥ 4) → 
  a ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_a_l705_70577


namespace NUMINAMATH_CALUDE_tenth_student_age_l705_70514

theorem tenth_student_age (total_students : ℕ) (students_without_tenth : ℕ) 
  (avg_age_without_tenth : ℕ) (avg_age_increase : ℕ) :
  total_students = 10 →
  students_without_tenth = 9 →
  avg_age_without_tenth = 8 →
  avg_age_increase = 2 →
  (students_without_tenth * avg_age_without_tenth + 
    (total_students * (avg_age_without_tenth + avg_age_increase) - 
     students_without_tenth * avg_age_without_tenth)) = 28 := by
  sorry

end NUMINAMATH_CALUDE_tenth_student_age_l705_70514


namespace NUMINAMATH_CALUDE_gcd_upper_bound_l705_70505

theorem gcd_upper_bound (a b c : ℕ) (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) :
  Nat.gcd (a * b + 1) (Nat.gcd (a * c + 1) (b * c + 1)) ≤ (a + b + c) / 3 :=
sorry

end NUMINAMATH_CALUDE_gcd_upper_bound_l705_70505


namespace NUMINAMATH_CALUDE_students_playing_neither_l705_70532

/-- Theorem: In a class of 39 students, where 26 play football, 20 play tennis, and 17 play both,
    the number of students who play neither football nor tennis is 10. -/
theorem students_playing_neither (N F T B : ℕ) 
  (h_total : N = 39)
  (h_football : F = 26)
  (h_tennis : T = 20)
  (h_both : B = 17) :
  N - (F + T - B) = 10 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_neither_l705_70532


namespace NUMINAMATH_CALUDE_cubic_polynomial_inequality_l705_70554

/-- A cubic polynomial with real coefficients and three non-zero real roots satisfies the inequality 6a^3 + 10(a^2 - 2b)^(3/2) - 12ab ≥ 27c. -/
theorem cubic_polynomial_inequality (a b c : ℝ) (P : ℝ → ℝ) (h1 : P = fun x ↦ x^3 + a*x^2 + b*x + c) 
  (h2 : ∃ (x y z : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ P x = 0 ∧ P y = 0 ∧ P z = 0) :
  6 * a^3 + 10 * (a^2 - 2*b)^(3/2) - 12 * a * b ≥ 27 * c := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_inequality_l705_70554


namespace NUMINAMATH_CALUDE_trig_expression_equals_one_l705_70500

theorem trig_expression_equals_one :
  let expr := (Real.sin (24 * π / 180) * Real.cos (16 * π / 180) + 
               Real.cos (156 * π / 180) * Real.cos (96 * π / 180)) /
              (Real.sin (26 * π / 180) * Real.cos (14 * π / 180) + 
               Real.cos (154 * π / 180) * Real.cos (94 * π / 180))
  expr = 1 := by
sorry

end NUMINAMATH_CALUDE_trig_expression_equals_one_l705_70500


namespace NUMINAMATH_CALUDE_three_digit_square_proof_l705_70519

theorem three_digit_square_proof : 
  ∃! (S : Finset Nat), 
    (∀ n ∈ S, 100 ≤ n ∧ n < 1000) ∧ 
    (∀ n ∈ S, ∃ k : Nat, 1000 * n = n^2 + k ∧ k < 1000) ∧
    S.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_square_proof_l705_70519


namespace NUMINAMATH_CALUDE_tangent_line_equation_l705_70571

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 6

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Theorem statement
theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (3*x + y - 7 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l705_70571


namespace NUMINAMATH_CALUDE_three_brothers_selection_probability_l705_70549

theorem three_brothers_selection_probability 
  (p_ram : ℚ) (p_ravi : ℚ) (p_raj : ℚ) 
  (h_ram : p_ram = 2/7)
  (h_ravi : p_ravi = 1/5)
  (h_raj : p_raj = 3/8) :
  p_ram * p_ravi * p_raj = 3/140 := by
sorry

end NUMINAMATH_CALUDE_three_brothers_selection_probability_l705_70549


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l705_70552

theorem perpendicular_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, x + a * y = 2 → 2 * x + 4 * y = 5 → (1 : ℝ) * (2 : ℝ) + a * (4 : ℝ) = 0) → 
  a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l705_70552


namespace NUMINAMATH_CALUDE_overall_speed_theorem_l705_70526

/-- Given three points A, B, C on a line with AB = BC, and a car traveling from A to B at 40 km/h
    and from B to C at 60 km/h without stopping, the overall speed of the trip from A to C is 48 km/h. -/
theorem overall_speed_theorem (A B C : ℝ) (h1 : A < B) (h2 : B < C) (h3 : B - A = C - B) : 
  let d := B - A
  let t1 := d / 40
  let t2 := d / 60
  let total_time := t1 + t2
  let total_distance := 2 * d
  total_distance / total_time = 48 := by
  sorry

end NUMINAMATH_CALUDE_overall_speed_theorem_l705_70526


namespace NUMINAMATH_CALUDE_kath_friends_count_l705_70533

/-- The number of friends Kath took to the movie --/
def num_friends : ℕ :=
  -- Define this value
  sorry

/-- The number of Kath's siblings --/
def num_siblings : ℕ := 2

/-- The regular admission cost in dollars --/
def regular_cost : ℕ := 8

/-- The discount amount in dollars --/
def discount : ℕ := 3

/-- The total amount Kath paid in dollars --/
def total_paid : ℕ := 30

/-- The actual cost per person after discount --/
def discounted_cost : ℕ := regular_cost - discount

/-- The total number of people in Kath's group --/
def total_people : ℕ := total_paid / discounted_cost

theorem kath_friends_count :
  num_friends = total_people - (num_siblings + 1) ∧
  num_friends = 3 :=
sorry

end NUMINAMATH_CALUDE_kath_friends_count_l705_70533


namespace NUMINAMATH_CALUDE_min_hat_flips_min_hat_flips_1000_l705_70576

theorem min_hat_flips (n : ℕ) (h : n = 1000) : ℕ :=
  let elf_count := n
  let initial_red_count : ℕ := n - 1
  let initial_blue_count : ℕ := 1
  let final_red_count : ℕ := 1
  let final_blue_count : ℕ := n - 1
  let min_flips := initial_red_count - final_red_count
  min_flips

/-- The minimum number of hat flips required for 1000 elves to satisfy the conditions is 998. -/
theorem min_hat_flips_1000 : min_hat_flips 1000 (by rfl) = 998 := by
  sorry

end NUMINAMATH_CALUDE_min_hat_flips_min_hat_flips_1000_l705_70576


namespace NUMINAMATH_CALUDE_probability_sum_four_two_dice_l705_70590

theorem probability_sum_four_two_dice : 
  let dice_count : ℕ := 2
  let faces_per_die : ℕ := 6
  let target_sum : ℕ := 4
  let total_outcomes : ℕ := faces_per_die ^ dice_count
  let favorable_outcomes : ℕ := 3
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_four_two_dice_l705_70590


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_solution_satisfies_conditions_l705_70515

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ m * x^2 + 2 * x = 1 ∧ m * y^2 + 2 * y = 1) ↔ 
  (m > -1 ∧ m ≠ 0) :=
by sorry

theorem solution_satisfies_conditions : 
  1 > -1 ∧ 1 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_solution_satisfies_conditions_l705_70515


namespace NUMINAMATH_CALUDE_stamps_per_page_l705_70534

theorem stamps_per_page (book1 book2 book3 : ℕ) 
  (h1 : book1 = 924) 
  (h2 : book2 = 1386) 
  (h3 : book3 = 1848) : 
  Nat.gcd book1 (Nat.gcd book2 book3) = 462 := by
  sorry

end NUMINAMATH_CALUDE_stamps_per_page_l705_70534


namespace NUMINAMATH_CALUDE_gas_cans_volume_l705_70530

/-- The volume of gas needed to fill a given number of gas cans with a specified capacity. -/
def total_gas_volume (num_cans : ℕ) (can_capacity : ℝ) : ℝ :=
  num_cans * can_capacity

/-- Theorem: The total volume of gas needed to fill 4 gas cans, each with a capacity of 5.0 gallons, is equal to 20.0 gallons. -/
theorem gas_cans_volume :
  total_gas_volume 4 5.0 = 20.0 := by
  sorry

end NUMINAMATH_CALUDE_gas_cans_volume_l705_70530


namespace NUMINAMATH_CALUDE_probability_three_different_suits_l705_70504

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of suits in a standard deck -/
def NumberOfSuits : ℕ := 4

/-- Represents the number of cards in each suit -/
def CardsPerSuit : ℕ := StandardDeck / NumberOfSuits

/-- The probability of selecting three cards of different suits from a standard deck without replacement -/
theorem probability_three_different_suits : 
  (39 : ℚ) / 51 * 24 / 50 = 156 / 425 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_different_suits_l705_70504


namespace NUMINAMATH_CALUDE_equation_solution_l705_70543

theorem equation_solution : 
  {x : ℝ | Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6} = {2, -2} :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l705_70543


namespace NUMINAMATH_CALUDE_largest_quantity_l705_70520

theorem largest_quantity (a b c d : ℝ) (h : a + 1 = b - 3 ∧ a + 1 = c + 4 ∧ a + 1 = d - 2) :
  b ≥ a ∧ b ≥ c ∧ b ≥ d :=
by sorry

end NUMINAMATH_CALUDE_largest_quantity_l705_70520


namespace NUMINAMATH_CALUDE_inequality_solution_l705_70582

/-- The numerator of the inequality -/
def numerator (x : ℝ) : ℝ := |3*x^2 + 8*x - 3| + |3*x^4 + 2*x^3 - 10*x^2 + 30*x - 9|

/-- The denominator of the inequality -/
def denominator (x : ℝ) : ℝ := |x-2| - 2*x - 1

/-- The inequality function -/
def inequality (x : ℝ) : Prop := numerator x / denominator x ≤ 0

/-- The solution set of the inequality -/
def solution_set : Set ℝ := {x | x < 1/3 ∨ x > 1/3}

theorem inequality_solution : 
  ∀ x : ℝ, inequality x ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l705_70582


namespace NUMINAMATH_CALUDE_earl_initial_ascent_l705_70568

def building_height : ℕ := 20

def initial_floor : ℕ := 1

theorem earl_initial_ascent (x : ℕ) : 
  x + 5 = building_height - 9 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_earl_initial_ascent_l705_70568


namespace NUMINAMATH_CALUDE_line_equation_and_distance_l705_70541

-- Define the point P
def P : ℝ × ℝ := (-1, 4)

-- Define line l₂
def l₂ (x y : ℝ) : Prop := 2 * x - y + 5 = 0

-- Define line l₁
def l₁ (x y : ℝ) : Prop := 2 * x - y + 6 = 0

-- Define line l₃
def l₃ (x y m : ℝ) : Prop := 4 * x - 2 * y + m = 0

-- State the theorem
theorem line_equation_and_distance (m : ℝ) : 
  (∀ x y, l₁ x y ↔ l₂ (x + 1/2) (y + 1/2)) ∧ -- l₁ is parallel to l₂
  l₁ P.1 P.2 ∧ -- P lies on l₁
  (∃ d, d = 2 * Real.sqrt 5 ∧ 
   d = |m - 12| / Real.sqrt (4^2 + (-2)^2)) → -- Distance between l₁ and l₃
  (m = -8 ∨ m = 32) := by
sorry

end NUMINAMATH_CALUDE_line_equation_and_distance_l705_70541


namespace NUMINAMATH_CALUDE_no_primes_from_200_l705_70544

def change_one_digit (n : ℕ) : Set ℕ :=
  {m : ℕ | ∃ (i : Fin 3), ∃ (d : Fin 10), 
    m = n + d * (10 ^ i.val) - (n / (10 ^ i.val) % 10) * (10 ^ i.val)}

theorem no_primes_from_200 :
  ∀ n ∈ change_one_digit 200, ¬ Nat.Prime n :=
sorry

end NUMINAMATH_CALUDE_no_primes_from_200_l705_70544


namespace NUMINAMATH_CALUDE_solution_set_l705_70564

theorem solution_set (x : ℝ) : (x^2 - 3*x > 8 ∧ |x| > 2) ↔ x < -2 ∨ x > 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l705_70564
