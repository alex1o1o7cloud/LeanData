import Mathlib

namespace NUMINAMATH_CALUDE_tangents_form_diameter_l3884_388415

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define the circle E
def circle_E (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define a point on the circle E
def point_on_E (P : ℝ × ℝ) : Prop :=
  circle_E P.1 P.2

-- Define tangent lines from P to C
def tangent_to_C (P : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
  ∃ (Q : ℝ × ℝ), ellipse_C Q.1 Q.2 ∧ l Q.1 = Q.2 ∧
  ∀ (x y : ℝ), ellipse_C x y → (y - l x) * (Q.2 - l Q.1) ≥ 0

-- Define intersection points of tangents with E
def intersect_E (P : ℝ × ℝ) (l : ℝ → ℝ) (M : ℝ × ℝ) : Prop :=
  M ≠ P ∧ circle_E M.1 M.2 ∧ l M.1 = M.2

-- Main theorem
theorem tangents_form_diameter (P M N : ℝ × ℝ) (l₁ l₂ : ℝ → ℝ) :
  point_on_E P →
  tangent_to_C P l₁ →
  tangent_to_C P l₂ →
  intersect_E P l₁ M →
  intersect_E P l₂ N →
  (M.1 + N.1 = 0 ∧ M.2 + N.2 = 0) :=
sorry

end NUMINAMATH_CALUDE_tangents_form_diameter_l3884_388415


namespace NUMINAMATH_CALUDE_parabola_focus_l3884_388446

/-- Given a parabola y = ax² passing through (1, 4), its focus is at (0, 1/16) -/
theorem parabola_focus (a : ℝ) : 
  (4 = a * 1^2) → -- Parabola passes through (1, 4)
  let f : ℝ × ℝ := (0, 1/16) -- Define focus coordinates
  (∀ x y : ℝ, y = a * x^2 → -- For all points (x, y) on the parabola
    (x - f.1)^2 = 4 * (1/(4*a)) * (y - f.2)) -- Satisfy the focus-directrix property
  := by sorry

end NUMINAMATH_CALUDE_parabola_focus_l3884_388446


namespace NUMINAMATH_CALUDE_point_not_in_third_quadrant_or_origin_l3884_388493

theorem point_not_in_third_quadrant_or_origin (n : ℝ) : 
  ¬(n ≤ 0 ∧ 1 - n ≤ 0) ∧ ¬(n = 0 ∧ 1 - n = 0) := by
  sorry

end NUMINAMATH_CALUDE_point_not_in_third_quadrant_or_origin_l3884_388493


namespace NUMINAMATH_CALUDE_chord_length_l3884_388457

/-- Given a circle and a line intersecting at two points, 
    prove that the length of the chord formed by these intersection points is 9√5 / 5 -/
theorem chord_length (x y : ℝ) : 
  (x^2 + y^2 + 4*x - 4*y - 10 = 0) →  -- Circle equation
  (2*x - y + 1 = 0) →                -- Line equation
  ∃ (A B : ℝ × ℝ),                   -- Existence of intersection points A and B
    (A.1^2 + A.2^2 + 4*A.1 - 4*A.2 - 10 = 0) ∧ 
    (2*A.1 - A.2 + 1 = 0) ∧
    (B.1^2 + B.2^2 + 4*B.1 - 4*B.2 - 10 = 0) ∧ 
    (2*B.1 - B.2 + 1 = 0) ∧
    (A ≠ B) ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = (9*Real.sqrt 5 / 5)^2) := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l3884_388457


namespace NUMINAMATH_CALUDE_division_problem_l3884_388496

theorem division_problem (x : ℝ) (h : x = 1) : 4 / (1 + 3/x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3884_388496


namespace NUMINAMATH_CALUDE_special_triangle_sum_l3884_388459

/-- Triangle ABC with given side lengths and circles P and Q with specific properties -/
structure SpecialTriangle where
  -- Side lengths of triangle ABC
  AB : ℝ
  AC : ℝ
  BC : ℝ
  -- Radius of circle P
  radiusP : ℝ
  -- Radius of circle Q (to be determined)
  radiusQ : ℝ
  -- Conditions
  isosceles : AB = AC
  tangentP : radiusP < AB ∧ radiusP < BC
  tangentQ : radiusQ < AB ∧ radiusQ < BC
  externalTangent : radiusQ + radiusP < BC
  -- Representation of radiusQ
  m : ℕ
  n : ℕ
  k : ℕ
  radiusQForm : radiusQ = m - n * Real.sqrt k
  kPrime : Nat.Prime k

/-- The main theorem stating the sum of m and nk for the special triangle -/
theorem special_triangle_sum (t : SpecialTriangle) 
  (h1 : t.AB = 130)
  (h2 : t.BC = 150)
  (h3 : t.radiusP = 20) :
  t.m + t.n * t.k = 386 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_sum_l3884_388459


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l3884_388401

def total_votes : ℕ := 8000
def loss_margin : ℕ := 2400

theorem candidate_vote_percentage :
  ∃ (p : ℚ),
    p * total_votes = (total_votes - loss_margin) / 2 ∧
    p = 35 / 100 :=
by sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l3884_388401


namespace NUMINAMATH_CALUDE_min_value_f_inequality_abc_l3884_388439

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * abs (x + 1) + abs (x - 2)

-- Theorem for the minimum value of f(x)
theorem min_value_f : ∃ (m : ℝ), m = 3 ∧ ∀ (x : ℝ), f x ≥ m :=
sorry

-- Theorem for the inequality
theorem inequality_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  b^2 / a + c^2 / b + a^2 / c ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_f_inequality_abc_l3884_388439


namespace NUMINAMATH_CALUDE_perimeter_semicircular_square_l3884_388447

/-- The perimeter of a region bounded by semicircular arcs constructed on the sides of a square with side length 1/π is equal to 2. -/
theorem perimeter_semicircular_square : 
  let side_length : ℝ := 1 / Real.pi
  let semicircle_length : ℝ := Real.pi * side_length / 2
  let num_semicircles : ℕ := 4
  semicircle_length * num_semicircles = 2 := by sorry

end NUMINAMATH_CALUDE_perimeter_semicircular_square_l3884_388447


namespace NUMINAMATH_CALUDE_sum_of_squares_l3884_388468

theorem sum_of_squares (a b c : ℝ) 
  (h1 : a * b + b * c + a * c = 116) 
  (h2 : a + b + c = 22) : 
  a^2 + b^2 + c^2 = 252 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3884_388468


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l3884_388465

open Real

theorem function_inequality_implies_a_bound (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, x^2 * exp x > 3 * exp x + a) →
  a < exp 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l3884_388465


namespace NUMINAMATH_CALUDE_divisors_multiple_of_five_3780_l3884_388460

/-- The number of positive divisors of 3780 that are multiples of 5 -/
def divisors_multiple_of_five (n : ℕ) : ℕ :=
  (Finset.filter (λ d => d % 5 = 0) (Nat.divisors n)).card

/-- Theorem stating that the number of positive divisors of 3780 that are multiples of 5 is 24 -/
theorem divisors_multiple_of_five_3780 :
  divisors_multiple_of_five 3780 = 24 := by
  sorry

end NUMINAMATH_CALUDE_divisors_multiple_of_five_3780_l3884_388460


namespace NUMINAMATH_CALUDE_constant_angle_existence_l3884_388438

-- Define the circle C
def Circle (O : ℝ × ℝ) (r : ℝ) := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2}

-- Define the line L
def Line (a b c : ℝ) := {P : ℝ × ℝ | a * P.1 + b * P.2 + c = 0}

-- Define the condition that L does not intersect C
def DoesNotIntersect (C : Set (ℝ × ℝ)) (L : Set (ℝ × ℝ)) := C ∩ L = ∅

-- Define the circle with diameter MN
def CircleWithDiameter (M N : ℝ × ℝ) := 
  Circle ((M.1 + N.1) / 2, (M.2 + N.2) / 2) ((M.1 - N.1)^2 + (M.2 - N.2)^2)

-- Define the condition that CircleWithDiameter touches C but does not contain it
def TouchesButNotContains (C D : Set (ℝ × ℝ)) := 
  (∃ P, P ∈ C ∧ P ∈ D) ∧ (¬∃ P, P ∈ C ∧ P ∈ interior D)

-- Define the angle MPN
def Angle (M P N : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem constant_angle_existence 
  (C : Set (ℝ × ℝ)) (L : Set (ℝ × ℝ)) (O : ℝ × ℝ) (r : ℝ) 
  (hC : C = Circle O r) (hL : ∃ a b c, L = Line a b c) 
  (hNotIntersect : DoesNotIntersect C L) :
  ∃ P : ℝ × ℝ, ∀ M N : ℝ × ℝ, 
    M ∈ L → N ∈ L → 
    TouchesButNotContains C (CircleWithDiameter M N) →
    ∃ θ : ℝ, Angle M P N = θ :=
sorry

end NUMINAMATH_CALUDE_constant_angle_existence_l3884_388438


namespace NUMINAMATH_CALUDE_students_playing_both_football_and_tennis_l3884_388478

/-- Given a class of students, calculates the number of students playing both football and long tennis. -/
def students_playing_both (total : ℕ) (football : ℕ) (long_tennis : ℕ) (neither : ℕ) : ℕ :=
  football + long_tennis - (total - neither)

/-- Theorem: In a class of 36 students, where 26 play football, 20 play long tennis, and 7 play neither,
    the number of students who play both football and long tennis is 17. -/
theorem students_playing_both_football_and_tennis :
  students_playing_both 36 26 20 7 = 17 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_both_football_and_tennis_l3884_388478


namespace NUMINAMATH_CALUDE_power_of_power_of_two_l3884_388434

theorem power_of_power_of_two :
  let a : ℕ := 2
  a^(a^2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_of_two_l3884_388434


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3884_388487

theorem complex_equation_solution (z : ℂ) : z + Complex.abs z * Complex.I = 3 + 9 * Complex.I → z = 3 + 4 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3884_388487


namespace NUMINAMATH_CALUDE_morgan_red_pens_l3884_388472

def total_pens : ℕ := 168
def blue_pens : ℕ := 45
def black_pens : ℕ := 58

def red_pens : ℕ := total_pens - (blue_pens + black_pens)

theorem morgan_red_pens : red_pens = 65 := by sorry

end NUMINAMATH_CALUDE_morgan_red_pens_l3884_388472


namespace NUMINAMATH_CALUDE_five_students_three_events_outcomes_l3884_388427

/-- The number of different possible outcomes for champions in a sports competition. -/
def championOutcomes (numStudents : ℕ) (numEvents : ℕ) : ℕ :=
  numStudents ^ numEvents

/-- Theorem stating that with 5 students and 3 events, there are 125 possible outcomes. -/
theorem five_students_three_events_outcomes :
  championOutcomes 5 3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_five_students_three_events_outcomes_l3884_388427


namespace NUMINAMATH_CALUDE_jame_tear_frequency_l3884_388477

/-- Represents the number of times Jame tears cards per week -/
def tear_frequency (cards_per_tear : ℕ) (cards_per_deck : ℕ) (num_decks : ℕ) (num_weeks : ℕ) : ℕ :=
  (cards_per_deck * num_decks) / (cards_per_tear * num_weeks)

/-- Theorem stating that Jame tears cards 3 times a week given the conditions -/
theorem jame_tear_frequency :
  let cards_per_tear := 30
  let cards_per_deck := 55
  let num_decks := 18
  let num_weeks := 11
  tear_frequency cards_per_tear cards_per_deck num_decks num_weeks = 3 := by
  sorry


end NUMINAMATH_CALUDE_jame_tear_frequency_l3884_388477


namespace NUMINAMATH_CALUDE_chord_intersection_length_l3884_388494

/-- In a circle with radius R, chord AB of length a, diameter AC, and chord PQ perpendicular to AC
    intersecting AB at M with PM : MQ = 3 : 1, prove that AM = (4R²a) / (16R² - 3a²) -/
theorem chord_intersection_length (R a : ℝ) (h1 : R > 0) (h2 : a > 0) (h3 : a < 2*R) :
  ∃ (AM : ℝ), AM = (4 * R^2 * a) / (16 * R^2 - 3 * a^2) :=
sorry

end NUMINAMATH_CALUDE_chord_intersection_length_l3884_388494


namespace NUMINAMATH_CALUDE_d2_equals_18_l3884_388476

/-- Definition of E(m) -/
def E (m : ℕ) : ℕ :=
  sorry

/-- The polynomial r(x) -/
def r (x : ℕ) : ℕ :=
  sorry

/-- Theorem stating that d₂ = 18 in the polynomial r(x) that satisfies E(m) = r(m) -/
theorem d2_equals_18 :
  ∃ (d₄ d₃ d₂ d₁ d₀ : ℤ),
    (∀ m : ℕ, m ≥ 7 → Odd m → E m = d₄ * m^4 + d₃ * m^3 + d₂ * m^2 + d₁ * m + d₀) →
    d₂ = 18 :=
  sorry

end NUMINAMATH_CALUDE_d2_equals_18_l3884_388476


namespace NUMINAMATH_CALUDE_twelfth_term_of_sequence_l3884_388440

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

theorem twelfth_term_of_sequence (a₁ a₂ a₃ : ℚ) (h₁ : a₁ = 1/2) (h₂ : a₂ = 5/6) (h₃ : a₃ = 7/6) :
  arithmetic_sequence a₁ (a₂ - a₁) 12 = 25/6 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_of_sequence_l3884_388440


namespace NUMINAMATH_CALUDE_cricket_score_problem_l3884_388489

theorem cricket_score_problem :
  ∀ (a b c d e : ℕ),
    -- Average score is 36
    a + b + c + d + e = 36 * 5 →
    -- D scored 5 more than E
    d = e + 5 →
    -- E scored 8 fewer than A
    e = a - 8 →
    -- B scored as many as D and E combined
    b = d + e →
    -- E scored 20 runs
    e = 20 →
    -- Prove that B and C scored 107 runs between them
    b + c = 107 := by
  sorry

end NUMINAMATH_CALUDE_cricket_score_problem_l3884_388489


namespace NUMINAMATH_CALUDE_max_value_of_a_l3884_388492

theorem max_value_of_a (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (sum_squares_one : a^2 + b^2 + c^2 = 1) : 
  ∃ (max_a : ℝ), max_a = Real.sqrt 6 / 3 ∧ 
  ∀ a', (∃ b' c', a' + b' + c' = 0 ∧ a'^2 + b'^2 + c'^2 = 1) → a' ≤ max_a :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l3884_388492


namespace NUMINAMATH_CALUDE_fourth_buoy_distance_l3884_388431

/-- Represents the distance of a buoy from the beach -/
def buoy_distance (n : ℕ) (interval : ℝ) : ℝ := n * interval

theorem fourth_buoy_distance 
  (h1 : buoy_distance 3 interval = 72) 
  (h2 : interval > 0) : 
  buoy_distance 4 interval = 96 :=
sorry

end NUMINAMATH_CALUDE_fourth_buoy_distance_l3884_388431


namespace NUMINAMATH_CALUDE_exist_prime_sum_30_l3884_388485

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- State the theorem
theorem exist_prime_sum_30 : ∃ p q : ℕ, isPrime p ∧ isPrime q ∧ p + q = 30 := by
  sorry

end NUMINAMATH_CALUDE_exist_prime_sum_30_l3884_388485


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3884_388432

/-- A quadratic function f(x) = ax² + bx + c -/
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_properties :
  ∃ (a b c : ℝ),
    (∀ x : ℝ, quadratic_function a b c (-1) = 0 ∧
              quadratic_function a b c (x + 1) - quadratic_function a b c x = 2 * x) →
    (∀ x : ℝ, quadratic_function a b c x = x^2 - x - 2) ∧
    (∀ x : ℝ, quadratic_function a b c x ≥ 0) ∧
    (∀ x : ℝ, quadratic_function a b c (x - 4) = quadratic_function a b c (2 - x)) ∧
    (∀ x : ℝ, 0 ≤ quadratic_function a b c x - x ∧
              quadratic_function a b c x - x ≤ (1/2) * (x - 1)^2) ∧
    a = 1/4 ∧ b = 1/2 ∧ c = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3884_388432


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l3884_388484

/-- A function f is even if f(-x) = f(x) for all x. -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- Given an even function f, prove that f(f(x)) is also even. -/
theorem composition_of_even_is_even (f : ℝ → ℝ) (hf : IsEven f) : IsEven (f ∘ f) := by
  sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l3884_388484


namespace NUMINAMATH_CALUDE_jerry_mowing_fraction_l3884_388482

def total_lawn_area : ℝ := 8
def riding_mower_rate : ℝ := 2
def push_mower_rate : ℝ := 1
def total_mowing_time : ℝ := 5

theorem jerry_mowing_fraction :
  ∃ x : ℝ,
    x ≥ 0 ∧ x ≤ 1 ∧
    (riding_mower_rate * x * total_mowing_time) +
    (push_mower_rate * (1 - x) * total_mowing_time) = total_lawn_area ∧
    x = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_jerry_mowing_fraction_l3884_388482


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l3884_388411

/-- Given that x and y are inversely proportional, x + y = 30, and x - y = 10,
    prove that when x = 3, y = 200/3 -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) 
  (h2 : x + y = 30) (h3 : x - y = 10) : 
  x = 3 → y = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l3884_388411


namespace NUMINAMATH_CALUDE_point_coordinates_l3884_388464

theorem point_coordinates (m n : ℝ) : (m + 3)^2 + Real.sqrt (4 - n) = 0 → m = -3 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l3884_388464


namespace NUMINAMATH_CALUDE_max_value_of_f_l3884_388455

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 1 / x else -x^2 + 2

theorem max_value_of_f :
  ∃ (M : ℝ), M = 2 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3884_388455


namespace NUMINAMATH_CALUDE_original_denominator_proof_l3884_388471

theorem original_denominator_proof (d : ℚ) : 
  (3 : ℚ) / d ≠ (1 : ℚ) / 3 ∧ (3 + 7 : ℚ) / (d + 7) = (1 : ℚ) / 3 → d = 23 := by
  sorry

end NUMINAMATH_CALUDE_original_denominator_proof_l3884_388471


namespace NUMINAMATH_CALUDE_greatest_third_side_proof_l3884_388467

/-- The greatest integer length for the third side of a triangle with sides 5 and 10 -/
def greatest_third_side : ℕ :=
  14

theorem greatest_third_side_proof :
  ∀ (c : ℕ),
  (c > greatest_third_side → ¬(5 < c + 10 ∧ 10 < c + 5 ∧ c < 5 + 10)) ∧
  (c ≤ greatest_third_side → (5 < c + 10 ∧ 10 < c + 5 ∧ c < 5 + 10)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_third_side_proof_l3884_388467


namespace NUMINAMATH_CALUDE_square_difference_equals_one_l3884_388463

theorem square_difference_equals_one : 1.99^2 - 1.98 * 1.99 + 0.99^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_one_l3884_388463


namespace NUMINAMATH_CALUDE_quadratic_negative_on_unit_interval_l3884_388420

/-- Given a quadratic function f(x) = ax² + bx + c with a > b > c and a + b + c = 0,
    prove that f(x) < 0 for all x in the open interval (0, 1). -/
theorem quadratic_negative_on_unit_interval
  (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  ∀ x, x ∈ Set.Ioo 0 1 → a * x^2 + b * x + c < 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_negative_on_unit_interval_l3884_388420


namespace NUMINAMATH_CALUDE_pulley_centers_distance_l3884_388407

theorem pulley_centers_distance (r₁ r₂ d : ℝ) (hr₁ : r₁ = 14) (hr₂ : r₂ = 4) (hd : d = 24) :
  Real.sqrt ((r₁ - r₂)^2 + d^2) = 26 := by
  sorry

end NUMINAMATH_CALUDE_pulley_centers_distance_l3884_388407


namespace NUMINAMATH_CALUDE_car_distance_problem_l3884_388430

theorem car_distance_problem (V : ℝ) (D : ℝ) : 
  V = 50 →
  D / V - D / (V + 25) = 0.5 →
  D = 75 := by
sorry

end NUMINAMATH_CALUDE_car_distance_problem_l3884_388430


namespace NUMINAMATH_CALUDE_x_depends_on_m_and_n_l3884_388469

theorem x_depends_on_m_and_n (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hmn : m ≠ n) :
  ∃ (a b : ℝ → ℝ → ℝ), ∀ (x : ℝ),
    (x = a m n * m + b m n * n) →
    ((x + m)^3 - (x + n)^3 = (m - n)^3) →
    (a m n ≠ 1 ∨ b m n ≠ 1) ∧
    (a m n ≠ -1 ∨ b m n ≠ 1) ∧
    (a m n ≠ 1 ∨ b m n ≠ -1) ∧
    (a m n ≠ -1 ∨ b m n ≠ -1) :=
by sorry

end NUMINAMATH_CALUDE_x_depends_on_m_and_n_l3884_388469


namespace NUMINAMATH_CALUDE_max_radius_circle_in_quartic_region_l3884_388456

/-- The maximum radius of a circle touching the origin and lying in y ≥ x^4 -/
theorem max_radius_circle_in_quartic_region : ∃ r : ℝ,
  (∀ x y : ℝ, x^2 + (y - r)^2 = r^2 → y ≥ x^4) ∧
  (∀ s : ℝ, s > r → ∃ x y : ℝ, x^2 + (y - s)^2 = s^2 ∧ y < x^4) ∧
  r = (3 * Real.rpow 2 (1/3 : ℝ)) / 4 :=
sorry

end NUMINAMATH_CALUDE_max_radius_circle_in_quartic_region_l3884_388456


namespace NUMINAMATH_CALUDE_factor_expression_l3884_388419

theorem factor_expression (x : ℝ) : 54 * x^3 - 135 * x^5 = 27 * x^3 * (2 - 5 * x^2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3884_388419


namespace NUMINAMATH_CALUDE_work_rate_ratio_l3884_388425

/-- Given three workers with work rates, prove the ratio of combined work rates -/
theorem work_rate_ratio 
  (R₁ R₂ R₃ : ℝ) 
  (h₁ : R₂ + R₃ = 2 * R₁) 
  (h₂ : R₁ + R₃ = 3 * R₂) : 
  (R₁ + R₂) / R₃ = 7 / 5 := by
sorry

end NUMINAMATH_CALUDE_work_rate_ratio_l3884_388425


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3884_388448

theorem complex_equation_solution :
  ∀ z : ℂ, z - 3 * I = 3 + z * I → z = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3884_388448


namespace NUMINAMATH_CALUDE_invalid_votes_percentage_l3884_388428

theorem invalid_votes_percentage
  (total_votes : ℕ)
  (candidate_a_percentage : ℚ)
  (candidate_a_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : candidate_a_percentage = 70 / 100)
  (h3 : candidate_a_votes = 333200) :
  (total_votes - (candidate_a_votes / candidate_a_percentage)) / total_votes = 15 / 100 :=
by sorry

end NUMINAMATH_CALUDE_invalid_votes_percentage_l3884_388428


namespace NUMINAMATH_CALUDE_ways_to_pick_one_ball_ways_to_pick_two_different_colored_balls_l3884_388490

-- Define the number of red and white balls
def num_red_balls : ℕ := 8
def num_white_balls : ℕ := 7

-- Theorem for the first question
theorem ways_to_pick_one_ball : 
  num_red_balls + num_white_balls = 15 := by sorry

-- Theorem for the second question
theorem ways_to_pick_two_different_colored_balls : 
  num_red_balls * num_white_balls = 56 := by sorry

end NUMINAMATH_CALUDE_ways_to_pick_one_ball_ways_to_pick_two_different_colored_balls_l3884_388490


namespace NUMINAMATH_CALUDE_ice_cream_preference_l3884_388453

theorem ice_cream_preference (total : ℕ) (vanilla : ℕ) (strawberry : ℕ) (neither : ℕ) 
  (h1 : total = 50)
  (h2 : vanilla = 23)
  (h3 : strawberry = 20)
  (h4 : neither = 14) :
  total - neither - (vanilla + strawberry - (total - neither)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_preference_l3884_388453


namespace NUMINAMATH_CALUDE_right_trapezoid_base_difference_l3884_388429

/-- A right trapezoid with specific properties -/
structure RightTrapezoid where
  /-- The length of the longer leg -/
  longer_leg : ℝ
  /-- The measure of the largest angle in degrees -/
  largest_angle : ℝ
  /-- The length of the longer base -/
  longer_base : ℝ
  /-- The length of the shorter base -/
  shorter_base : ℝ
  /-- The longer leg is positive -/
  longer_leg_pos : longer_leg > 0
  /-- The largest angle is between 90° and 180° -/
  largest_angle_range : 90 < largest_angle ∧ largest_angle < 180
  /-- The longer base is longer than the shorter base -/
  base_order : longer_base > shorter_base

/-- The theorem stating the difference between bases of the specific right trapezoid -/
theorem right_trapezoid_base_difference (t : RightTrapezoid) 
    (h1 : t.longer_leg = 12)
    (h2 : t.largest_angle = 120) :
    t.longer_base - t.shorter_base = 6 := by
  sorry

end NUMINAMATH_CALUDE_right_trapezoid_base_difference_l3884_388429


namespace NUMINAMATH_CALUDE_odd_even_sum_difference_l3884_388421

def sum_odd (n : ℕ) : ℕ := n^2

def sum_even (n : ℕ) : ℕ := n * (n + 1)

theorem odd_even_sum_difference : 
  let n_odd : ℕ := (2023 - 1) / 2 + 1
  let n_even : ℕ := (2022 - 2) / 2 + 1
  sum_odd n_odd - sum_even n_even + 7 - 8 = 47 := by
  sorry

end NUMINAMATH_CALUDE_odd_even_sum_difference_l3884_388421


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3884_388406

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_a5 : a 5 = 8)
  (h_a9 : a 9 = 24) :
  ∃ d : ℝ, d = 4 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3884_388406


namespace NUMINAMATH_CALUDE_alex_silk_distribution_l3884_388413

/-- The amount of silk each friend receives when Alex distributes his remaining silk -/
def silk_per_friend (total_silk : ℕ) (silk_per_dress : ℕ) (num_dresses : ℕ) (num_friends : ℕ) : ℕ :=
  (total_silk - silk_per_dress * num_dresses) / num_friends

/-- Theorem stating that each friend receives 20 meters of silk -/
theorem alex_silk_distribution :
  silk_per_friend 600 5 100 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_alex_silk_distribution_l3884_388413


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3884_388417

def I : Finset Nat := {0, 1, 2, 3, 4}
def M : Finset Nat := {1, 2, 3}
def N : Finset Nat := {0, 3, 4}

theorem complement_intersection_theorem :
  (I \ M) ∩ N = {0, 4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3884_388417


namespace NUMINAMATH_CALUDE_solution_satisfies_inequalities_inequalities_imply_solution_solution_is_correct_l3884_388410

-- Define the system of inequalities
def inequality1 (x : ℝ) : Prop := x + 2 < 3 + 2*x
def inequality2 (x : ℝ) : Prop := 4*x - 3 < 3*x - 1
def inequality3 (x : ℝ) : Prop := 8 + 5*x ≥ 6*x + 7

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 1}

-- Theorem stating that the solution set satisfies all inequalities
theorem solution_satisfies_inequalities :
  ∀ x ∈ solution_set, inequality1 x ∧ inequality2 x ∧ inequality3 x :=
sorry

-- Theorem stating that any real number satisfying all inequalities is in the solution set
theorem inequalities_imply_solution :
  ∀ x : ℝ, inequality1 x ∧ inequality2 x ∧ inequality3 x → x ∈ solution_set :=
sorry

-- Main theorem: The solution set is exactly (-1, 1]
theorem solution_is_correct :
  ∀ x : ℝ, x ∈ solution_set ↔ inequality1 x ∧ inequality2 x ∧ inequality3 x :=
sorry

end NUMINAMATH_CALUDE_solution_satisfies_inequalities_inequalities_imply_solution_solution_is_correct_l3884_388410


namespace NUMINAMATH_CALUDE_sine_cosine_roots_l3884_388409

theorem sine_cosine_roots (θ : Real) (m : Real) : 
  (∃ (x y : Real), x = Real.sin θ ∧ y = Real.cos θ ∧ 
   4 * x^2 + 2 * m * x + m = 0 ∧ 
   4 * y^2 + 2 * m * y + m = 0) →
  m = 1 - Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_sine_cosine_roots_l3884_388409


namespace NUMINAMATH_CALUDE_trajectory_equation_constant_distance_fixed_point_l3884_388486

/-- The trajectory of point P given the conditions -/
def trajectory (x y : ℝ) : Prop :=
  x ≠ 2 ∧ x ≠ -2 ∧ x^2 / 4 + y^2 = 1

/-- The line l intersecting the trajectory -/
def line_l (k m x y : ℝ) : Prop :=
  y = k * x + m

/-- Points M and N are on both the trajectory and line l -/
def intersection_points (x₁ y₁ x₂ y₂ k m : ℝ) : Prop :=
  trajectory x₁ y₁ ∧ trajectory x₂ y₂ ∧
  line_l k m x₁ y₁ ∧ line_l k m x₂ y₂ ∧
  (x₁, y₁) ≠ (x₂, y₂)

/-- OM is perpendicular to ON -/
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

/-- The slopes of BM and BN satisfy the given condition -/
def slope_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (y₁ / (x₁ - 2)) * (y₂ / (x₂ - 2)) = -1/4

theorem trajectory_equation (x y : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ -2) :
  (y / (x + 2)) * (y / (x - 2)) = -1/4 ↔ trajectory x y :=
sorry

theorem constant_distance (k m x₁ y₁ x₂ y₂ : ℝ)
  (h : intersection_points x₁ y₁ x₂ y₂ k m)
  (h_perp : perpendicular x₁ y₁ x₂ y₂) :
  |m| / Real.sqrt (1 + k^2) = 2 * Real.sqrt 5 / 5 :=
sorry

theorem fixed_point (k m x₁ y₁ x₂ y₂ : ℝ)
  (h : intersection_points x₁ y₁ x₂ y₂ k m)
  (h_slope : slope_condition x₁ y₁ x₂ y₂) :
  m = 0 :=
sorry

end NUMINAMATH_CALUDE_trajectory_equation_constant_distance_fixed_point_l3884_388486


namespace NUMINAMATH_CALUDE_consistency_comparison_l3884_388449

/-- Represents a player's performance in a basketball competition -/
structure PlayerPerformance where
  average_score : ℝ
  standard_deviation : ℝ

/-- Determines if a player performed more consistently than another -/
def more_consistent (p1 p2 : PlayerPerformance) : Prop :=
  p1.average_score = p2.average_score ∧ p1.standard_deviation < p2.standard_deviation

/-- Theorem: Given two players with the same average score, 
    the player with the smaller standard deviation performed more consistently -/
theorem consistency_comparison 
  (player_A player_B : PlayerPerformance) 
  (h_avg : player_A.average_score = player_B.average_score) 
  (h_std : player_B.standard_deviation < player_A.standard_deviation) : 
  more_consistent player_B player_A :=
sorry

end NUMINAMATH_CALUDE_consistency_comparison_l3884_388449


namespace NUMINAMATH_CALUDE_mayoral_election_vote_ratio_l3884_388473

theorem mayoral_election_vote_ratio :
  let votes_Z : ℕ := 25000
  let votes_X : ℕ := 22500
  let votes_Y : ℕ := 2 * votes_X / 3
  let fewer_votes : ℕ := votes_Z - votes_Y
  (fewer_votes : ℚ) / votes_Z = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_mayoral_election_vote_ratio_l3884_388473


namespace NUMINAMATH_CALUDE_square_areas_l3884_388470

theorem square_areas (a b : ℝ) (h1 : 4*a - 4*b = 12) (h2 : a^2 - b^2 = 69) :
  (a^2 = 169 ∧ b^2 = 100) :=
sorry

end NUMINAMATH_CALUDE_square_areas_l3884_388470


namespace NUMINAMATH_CALUDE_apples_sold_l3884_388491

/-- Represents an apple variety with its prices and proportion -/
structure AppleVariety where
  light_price : ℚ
  heavy_price : ℚ
  proportion : ℚ

/-- Calculates the average price of an apple variety -/
def average_price (av : AppleVariety) : ℚ :=
  0.6 * av.light_price + 0.4 * av.heavy_price

/-- Calculates the weighted average price of all apple varieties -/
def weighted_average_price (varieties : List AppleVariety) : ℚ :=
  varieties.foldl (λ acc av => acc + av.proportion * average_price av) 0

/-- The main theorem stating the number of apples sold -/
theorem apples_sold (varieties : List AppleVariety) (total_earnings : ℚ) :
  varieties = [
    ⟨0.4, 0.6, 0.4⟩,  -- Sweet apples
    ⟨0.1, 0.15, 0.2⟩, -- Sour apples
    ⟨0.25, 0.35, 0.15⟩, -- Crunchy apples
    ⟨0.15, 0.25, 0.1⟩, -- Soft apples
    ⟨0.2, 0.3, 0.1⟩,  -- Juicy apples
    ⟨0.05, 0.1, 0.05⟩ -- Tangy apples
  ] →
  total_earnings = 120 →
  ⌊total_earnings / weighted_average_price varieties⌋ = 392 := by
  sorry


end NUMINAMATH_CALUDE_apples_sold_l3884_388491


namespace NUMINAMATH_CALUDE_complex_power_sum_l3884_388444

theorem complex_power_sum (z : ℂ) (h : z = (1 - I) / Real.sqrt 2) : 
  z^100 + z^50 + 1 = -I :=
by sorry

end NUMINAMATH_CALUDE_complex_power_sum_l3884_388444


namespace NUMINAMATH_CALUDE_product_sum_fraction_equality_l3884_388480

theorem product_sum_fraction_equality : (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 47 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_fraction_equality_l3884_388480


namespace NUMINAMATH_CALUDE_dropped_student_score_l3884_388424

theorem dropped_student_score 
  (initial_students : ℕ) 
  (initial_average : ℝ) 
  (remaining_students : ℕ) 
  (new_average : ℝ) : ℝ :=
  by
  have h1 : initial_students = 16 := by sorry
  have h2 : initial_average = 60.5 := by sorry
  have h3 : remaining_students = 15 := by sorry
  have h4 : new_average = 64 := by sorry

  -- The score of the dropped student
  let dropped_score := initial_students * initial_average - remaining_students * new_average

  -- Prove that the dropped score is 8
  have h5 : dropped_score = 8 := by sorry

  exact dropped_score

end NUMINAMATH_CALUDE_dropped_student_score_l3884_388424


namespace NUMINAMATH_CALUDE_heather_shared_blocks_l3884_388452

/-- The number of blocks Heather shared with Jose -/
def blocks_shared (initial final : ℕ) : ℕ := initial - final

/-- Theorem: The number of blocks Heather shared is the difference between her initial and final blocks -/
theorem heather_shared_blocks (initial final : ℕ) (h1 : initial = 86) (h2 : final = 45) :
  blocks_shared initial final = 41 := by
  sorry

end NUMINAMATH_CALUDE_heather_shared_blocks_l3884_388452


namespace NUMINAMATH_CALUDE_exists_polynomial_with_negative_coeff_positive_powers_l3884_388461

theorem exists_polynomial_with_negative_coeff_positive_powers :
  ∃ (P : Polynomial ℝ), 
    (∃ (i : ℕ), (P.coeff i) < 0) ∧ 
    (∀ (n : ℕ), n > 1 → ∀ (j : ℕ), ((P ^ n).coeff j) > 0) := by
  sorry

end NUMINAMATH_CALUDE_exists_polynomial_with_negative_coeff_positive_powers_l3884_388461


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l3884_388475

theorem diophantine_equation_solutions :
  ∀ x y z : ℕ, 2^x * 3^y + 1 = 7^z →
    ((x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = 4 ∧ y = 1 ∧ z = 2)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l3884_388475


namespace NUMINAMATH_CALUDE_smallest_k_for_zero_l3884_388450

def f (a b M : ℕ) (n : ℤ) : ℤ :=
  if n ≤ M then n + a else n - b

def iterate_f (a b M : ℕ) : ℕ → ℤ → ℤ
  | 0, n => n
  | k+1, n => f a b M (iterate_f a b M k n)

theorem smallest_k_for_zero (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ b) :
  let M := (a + b) / 2
  ∃ k : ℕ, (∀ j < k, iterate_f a b M j 0 ≠ 0) ∧ 
            iterate_f a b M k 0 = 0 ∧
            k = (a + b) / Nat.gcd a b :=
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_zero_l3884_388450


namespace NUMINAMATH_CALUDE_alice_pears_l3884_388499

/-- The number of pears Alice sold -/
def sold : ℕ := sorry

/-- The number of pears Alice poached -/
def poached : ℕ := sorry

/-- The number of pears Alice canned -/
def canned : ℕ := sorry

/-- The total number of pears -/
def total : ℕ := 42

theorem alice_pears :
  (canned = poached + poached / 5) ∧
  (poached = sold / 2) ∧
  (sold + poached + canned = total) →
  sold = 20 := by sorry

end NUMINAMATH_CALUDE_alice_pears_l3884_388499


namespace NUMINAMATH_CALUDE_pool_filling_rate_l3884_388423

theorem pool_filling_rate (jim_rate sue_rate tony_rate : ℚ) 
  (h_jim : jim_rate = 1 / 30)
  (h_sue : sue_rate = 1 / 45)
  (h_tony : tony_rate = 1 / 90) :
  jim_rate + sue_rate + tony_rate = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_pool_filling_rate_l3884_388423


namespace NUMINAMATH_CALUDE_unique_rational_root_l3884_388445

def polynomial (x : ℚ) : ℚ := 3 * x^4 - 5 * x^3 - 8 * x^2 + 5 * x + 1

theorem unique_rational_root :
  ∀ x : ℚ, polynomial x = 0 ↔ x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_rational_root_l3884_388445


namespace NUMINAMATH_CALUDE_three_digit_probability_l3884_388408

theorem three_digit_probability : 
  let S := Finset.Icc 30 800
  let three_digit := {n : ℕ | 100 ≤ n ∧ n ≤ 800}
  (S.filter (λ n => n ∈ three_digit)).card / S.card = 701 / 771 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_probability_l3884_388408


namespace NUMINAMATH_CALUDE_diagonals_30_sided_polygon_l3884_388403

/-- The number of diagonals in a convex polygon with n sides -/
def numDiagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- Theorem: The number of diagonals in a convex polygon with 30 sides is 375 -/
theorem diagonals_30_sided_polygon :
  numDiagonals 30 = 375 := by sorry

end NUMINAMATH_CALUDE_diagonals_30_sided_polygon_l3884_388403


namespace NUMINAMATH_CALUDE_grading_multiple_l3884_388422

theorem grading_multiple (total_questions : ℕ) (score : ℕ) (correct_responses : ℕ) :
  total_questions = 100 →
  score = 70 →
  correct_responses = 90 →
  ∃ m : ℕ, score = correct_responses - m * (total_questions - correct_responses) →
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_grading_multiple_l3884_388422


namespace NUMINAMATH_CALUDE_floor_sqrt_10_l3884_388433

theorem floor_sqrt_10 : ⌊Real.sqrt 10⌋ = 3 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_10_l3884_388433


namespace NUMINAMATH_CALUDE_probability_two_first_grade_pens_l3884_388442

/-- The probability of selecting 2 first-grade pens from a box of 6 pens, where 3 are first-grade -/
theorem probability_two_first_grade_pens (total_pens : ℕ) (first_grade_pens : ℕ) 
  (h1 : total_pens = 6) (h2 : first_grade_pens = 3) : 
  (Nat.choose first_grade_pens 2 : ℚ) / (Nat.choose total_pens 2) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_first_grade_pens_l3884_388442


namespace NUMINAMATH_CALUDE_cubic_transformation_l3884_388400

theorem cubic_transformation (x z : ℝ) (hz : z = x + 1/x) :
  x^3 - 3*x^2 + x + 2 = 0 ↔ x^2*(z^2 - z - 1) + 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_transformation_l3884_388400


namespace NUMINAMATH_CALUDE_fraction_comparison_l3884_388412

theorem fraction_comparison (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a / b > (a + 1) / (b + 1) := by
sorry

end NUMINAMATH_CALUDE_fraction_comparison_l3884_388412


namespace NUMINAMATH_CALUDE_min_sum_of_product_2310_l3884_388495

theorem min_sum_of_product_2310 (a b c : ℕ+) : 
  a * b * c = 2310 → (∀ x y z : ℕ+, x * y * z = 2310 → a + b + c ≤ x + y + z) → a + b + c = 40 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_product_2310_l3884_388495


namespace NUMINAMATH_CALUDE_binary_10101_is_21_l3884_388426

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_10101_is_21 :
  binary_to_decimal [true, false, true, false, true] = 21 := by
  sorry

end NUMINAMATH_CALUDE_binary_10101_is_21_l3884_388426


namespace NUMINAMATH_CALUDE_sum_8th_10th_is_230_l3884_388402

/-- An arithmetic sequence with given 4th and 6th terms -/
structure ArithmeticSequence where
  term4 : ℤ
  term6 : ℤ
  is_arithmetic : ∃ (a d : ℤ), term4 = a + 3 * d ∧ term6 = a + 5 * d

/-- The sum of the 8th and 10th terms of the arithmetic sequence -/
def sum_8th_10th_terms (seq : ArithmeticSequence) : ℤ :=
  let a : ℤ := seq.term4 - 3 * ((seq.term6 - seq.term4) / 2)
  let d : ℤ := (seq.term6 - seq.term4) / 2
  (a + 7 * d) + (a + 9 * d)

/-- Theorem stating that the sum of the 8th and 10th terms is 230 -/
theorem sum_8th_10th_is_230 (seq : ArithmeticSequence) 
  (h1 : seq.term4 = 25) (h2 : seq.term6 = 61) : 
  sum_8th_10th_terms seq = 230 := by
  sorry

end NUMINAMATH_CALUDE_sum_8th_10th_is_230_l3884_388402


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3884_388414

theorem sufficient_not_necessary (x y : ℝ) :
  (x ≥ 2 ∧ y ≥ 2 → x^2 + y^2 ≥ 4) ∧
  ∃ x y, x^2 + y^2 ≥ 4 ∧ ¬(x ≥ 2 ∧ y ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3884_388414


namespace NUMINAMATH_CALUDE_train_crossing_time_l3884_388474

/-- Prove that a train with given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 200 ∧ train_speed_kmh = 144 →
  (train_length / (train_speed_kmh * 1000 / 3600)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3884_388474


namespace NUMINAMATH_CALUDE_line_slope_angle_l3884_388418

theorem line_slope_angle (x y : ℝ) :
  x + Real.sqrt 3 * y - 2 = 0 →
  ∃ (m : ℝ), y = m * x + (2 * Real.sqrt 3) / 3 ∧
             m = -(Real.sqrt 3) / 3 ∧
             Real.tan (5 * Real.pi / 6) = m :=
by sorry

end NUMINAMATH_CALUDE_line_slope_angle_l3884_388418


namespace NUMINAMATH_CALUDE_least_n_modulo_121_l3884_388497

theorem least_n_modulo_121 : ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 → m < n → ¬(25^m + 16^m) % 121 = 1) ∧ (25^n + 16^n) % 121 = 1 :=
by
  use 32
  sorry

end NUMINAMATH_CALUDE_least_n_modulo_121_l3884_388497


namespace NUMINAMATH_CALUDE_william_tickets_l3884_388436

/-- William's ticket problem -/
theorem william_tickets : ∀ (initial additional : ℕ), 
  initial = 15 → additional = 3 → initial + additional = 18 := by
  sorry

end NUMINAMATH_CALUDE_william_tickets_l3884_388436


namespace NUMINAMATH_CALUDE_fractional_inequality_solution_set_l3884_388435

theorem fractional_inequality_solution_set :
  {x : ℝ | (x + 1) / (x - 3) ≥ 0} = {x : ℝ | x ≤ -1 ∨ x > 3} := by sorry

end NUMINAMATH_CALUDE_fractional_inequality_solution_set_l3884_388435


namespace NUMINAMATH_CALUDE_f_properties_l3884_388437

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  let k : ℤ := ⌊(x + 1) / 2⌋
  (-1: ℝ) ^ k * Real.sqrt (1 - (x - 2 * ↑k) ^ 2)

-- State the theorem
theorem f_properties :
  (∀ x : ℝ, f (x + 4) = f x) ∧
  (∀ x : ℝ, f (x + 2) + f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3884_388437


namespace NUMINAMATH_CALUDE_tank_volume_l3884_388483

-- Define the rates of the pipes
def inlet_rate : ℝ := 3
def outlet_rate_1 : ℝ := 9
def outlet_rate_2 : ℝ := 6

-- Define the time it takes to empty the tank
def emptying_time : ℝ := 4320

-- Define the conversion factor from cubic inches to cubic feet
def cubic_inches_per_cubic_foot : ℝ := 1728

-- State the theorem
theorem tank_volume (net_rate : ℝ) (volume_cubic_inches : ℝ) (volume_cubic_feet : ℝ) 
  (h1 : net_rate = outlet_rate_1 + outlet_rate_2 - inlet_rate)
  (h2 : volume_cubic_inches = net_rate * emptying_time)
  (h3 : volume_cubic_feet = volume_cubic_inches / cubic_inches_per_cubic_foot) :
  volume_cubic_feet = 30 := by
  sorry

end NUMINAMATH_CALUDE_tank_volume_l3884_388483


namespace NUMINAMATH_CALUDE_complementary_angle_supplement_l3884_388443

theorem complementary_angle_supplement (A B : Real) : 
  (A + B = 90) → (180 - A = 90 + B) := by
  sorry

end NUMINAMATH_CALUDE_complementary_angle_supplement_l3884_388443


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3884_388488

theorem geometric_series_sum (a : ℝ) (r : ℝ) (n : ℕ) (h1 : a = 3) (h2 : r = -2) (h3 : a * r^(n-1) = -1536) :
  (a * (1 - r^n)) / (1 - r) = -1023 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3884_388488


namespace NUMINAMATH_CALUDE_eliza_cookies_l3884_388451

theorem eliza_cookies (x : ℚ) 
  (h1 : x + 3*x + 4*(3*x) + 6*(4*(3*x)) = 234) : x = 117/44 := by
  sorry

end NUMINAMATH_CALUDE_eliza_cookies_l3884_388451


namespace NUMINAMATH_CALUDE_medal_award_ways_eq_78_l3884_388454

/-- The number of ways to award medals in a race with American and non-American sprinters. -/
def medalAwardWays (totalSprinters : ℕ) (americanSprinters : ℕ) : ℕ :=
  let nonAmericanSprinters := totalSprinters - americanSprinters
  let noAmericanWins := nonAmericanSprinters * (nonAmericanSprinters - 1)
  let oneAmericanWins := 2 * americanSprinters * nonAmericanSprinters
  noAmericanWins + oneAmericanWins

/-- Theorem stating that the number of ways to award medals in the given scenario is 78. -/
theorem medal_award_ways_eq_78 :
  medalAwardWays 10 4 = 78 := by
  sorry

end NUMINAMATH_CALUDE_medal_award_ways_eq_78_l3884_388454


namespace NUMINAMATH_CALUDE_circle_point_x_coordinate_l3884_388416

theorem circle_point_x_coordinate :
  ∀ x : ℝ,
  let circle_center : ℝ × ℝ := (7, 0)
  let circle_radius : ℝ := 14
  let point_on_circle : ℝ × ℝ := (x, 10)
  (point_on_circle.1 - circle_center.1)^2 + (point_on_circle.2 - circle_center.2)^2 = circle_radius^2 →
  x = 7 + 4 * Real.sqrt 6 ∨ x = 7 - 4 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_point_x_coordinate_l3884_388416


namespace NUMINAMATH_CALUDE_fraction_equality_l3884_388441

theorem fraction_equality (a b : ℝ) : - (a / (b - a)) = a / (a - b) := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l3884_388441


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l3884_388462

theorem sum_of_absolute_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x, (3*x - 1)^8 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8) →
  |a| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| + |a₈| = 4^8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l3884_388462


namespace NUMINAMATH_CALUDE_dans_limes_l3884_388479

theorem dans_limes (limes_picked : ℕ) (limes_given : ℕ) : limes_picked = 9 → limes_given = 4 → limes_picked + limes_given = 13 := by
  sorry

end NUMINAMATH_CALUDE_dans_limes_l3884_388479


namespace NUMINAMATH_CALUDE_triangle_integer_sides_altitudes_even_perimeter_l3884_388481

theorem triangle_integer_sides_altitudes_even_perimeter 
  (a b c : ℕ) 
  (ha hb hc : ℕ) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_altitudes : ha ≠ 0 ∧ hb ≠ 0 ∧ hc ≠ 0) :
  ∃ k : ℕ, a + b + c = 2 * k := by
  sorry

#check triangle_integer_sides_altitudes_even_perimeter

end NUMINAMATH_CALUDE_triangle_integer_sides_altitudes_even_perimeter_l3884_388481


namespace NUMINAMATH_CALUDE_empty_quadratic_inequality_solution_set_l3884_388458

/-- Given a quadratic inequality ax² + bx + c < 0 with a ≠ 0, 
    if the solution set is empty, then a > 0 and Δ ≤ 0, where Δ = b² - 4ac -/
theorem empty_quadratic_inequality_solution_set 
  (a b c : ℝ) (h1 : a ≠ 0) 
  (h2 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) : 
  a > 0 ∧ b^2 - 4*a*c ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_empty_quadratic_inequality_solution_set_l3884_388458


namespace NUMINAMATH_CALUDE_product_prices_and_min_units_l3884_388498

/-- Represents the unit price of product A in yuan -/
def price_A : ℝ := sorry

/-- Represents the unit price of product B in yuan -/
def price_B : ℝ := sorry

/-- Represents the total number of units produced (in thousands) -/
def total_units : ℕ := 80

/-- Represents the relationship between the sales revenue of A and B -/
axiom revenue_relation : 2 * price_A = 3 * price_B

/-- Represents the difference in sales revenue between A and B -/
axiom revenue_difference : 3 * price_A - 2 * price_B = 1500

/-- Represents the minimum number of units of A to be sold (in thousands) -/
def min_units_A : ℕ := sorry

/-- Theorem stating the unit prices of A and B, and the minimum units of A to be sold -/
theorem product_prices_and_min_units : 
  price_A = 900 ∧ price_B = 600 ∧ 
  (∀ m : ℕ, m ≥ min_units_A → 
    900 * m + 600 * (total_units - m) ≥ 54000) ∧
  min_units_A = 2 := by sorry

end NUMINAMATH_CALUDE_product_prices_and_min_units_l3884_388498


namespace NUMINAMATH_CALUDE_exponential_monotonicity_l3884_388404

theorem exponential_monotonicity (a b : ℝ) : a < b → (2 : ℝ) ^ a < (2 : ℝ) ^ b := by
  sorry

end NUMINAMATH_CALUDE_exponential_monotonicity_l3884_388404


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3884_388405

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem statement -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 4 + a 7 = 2 →
  a 5 * a 6 = -8 →
  a 1 + a 10 = -7 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3884_388405


namespace NUMINAMATH_CALUDE_max_dimes_in_piggy_banks_l3884_388466

/-- Represents the number of coins a piggy bank can hold -/
def PiggyBankCapacity : ℕ := 100

/-- Represents the total number of coins in two piggy banks -/
def TotalCoins : ℕ := 200

/-- Represents the total value of coins in cents -/
def TotalValue : ℕ := 1200

/-- Represents the value of a dime in cents -/
def DimeValue : ℕ := 10

/-- Represents the value of a penny in cents -/
def PennyValue : ℕ := 1

/-- Theorem stating the maximum number of dimes that can be held in the piggy banks -/
theorem max_dimes_in_piggy_banks :
  ∃ (d : ℕ), d ≤ 111 ∧
  d * DimeValue + (TotalCoins - d) * PennyValue = TotalValue ∧
  (∀ (x : ℕ), x > d →
    x * DimeValue + (TotalCoins - x) * PennyValue ≠ TotalValue) :=
by sorry

#check max_dimes_in_piggy_banks

end NUMINAMATH_CALUDE_max_dimes_in_piggy_banks_l3884_388466
