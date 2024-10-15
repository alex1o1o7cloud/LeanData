import Mathlib

namespace NUMINAMATH_CALUDE_tree_height_average_l3193_319364

def tree_heights : List ℕ → Prop
  | [h1, h2, h3, h4, h5] =>
    h2 = 6 ∧
    (h1 = 2 * h2 ∨ 2 * h1 = h2) ∧
    (h2 = 2 * h3 ∨ 2 * h2 = h3) ∧
    (h3 = 2 * h4 ∨ 2 * h3 = h4) ∧
    (h4 = 2 * h5 ∨ 2 * h4 = h5)
  | _ => False

theorem tree_height_average :
  ∀ (heights : List ℕ),
    tree_heights heights →
    (heights.sum : ℚ) / heights.length = 66 / 5 := by
  sorry

end NUMINAMATH_CALUDE_tree_height_average_l3193_319364


namespace NUMINAMATH_CALUDE_mutually_inscribed_pentagons_exist_l3193_319305

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a pentagon in a 2D plane -/
structure Pentagon where
  vertices : Fin 5 → Point

/-- Checks if a point lies on a line segment or its extension -/
def pointOnLineSegment (p : Point) (a : Point) (b : Point) : Prop := sorry

/-- Checks if two pentagons are mutually inscribed -/
def areMutuallyInscribed (p1 p2 : Pentagon) : Prop :=
  ∀ (i : Fin 5), 
    (pointOnLineSegment (p1.vertices i) (p2.vertices i) (p2.vertices ((i + 1) % 5))) ∧
    (pointOnLineSegment (p2.vertices i) (p1.vertices i) (p1.vertices ((i + 1) % 5)))

/-- Theorem: For any given pentagon, there exists another pentagon mutually inscribed with it -/
theorem mutually_inscribed_pentagons_exist (p : Pentagon) : 
  ∃ (q : Pentagon), areMutuallyInscribed p q := by sorry

end NUMINAMATH_CALUDE_mutually_inscribed_pentagons_exist_l3193_319305


namespace NUMINAMATH_CALUDE_no_three_digit_even_with_digit_sum_27_l3193_319378

/-- A function that returns the digit sum of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is a 3-digit number -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- A theorem stating that there are no 3-digit even numbers with a digit sum of 27 -/
theorem no_three_digit_even_with_digit_sum_27 :
  ¬ ∃ n : ℕ, is_three_digit n ∧ Even n ∧ digit_sum n = 27 := by sorry

end NUMINAMATH_CALUDE_no_three_digit_even_with_digit_sum_27_l3193_319378


namespace NUMINAMATH_CALUDE_complex_equation_sum_l3193_319318

theorem complex_equation_sum (a b : ℝ) : 
  (3 * b : ℂ) + (2 * a - 2) * I = 1 - I → a + b = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l3193_319318


namespace NUMINAMATH_CALUDE_travel_agency_problem_l3193_319384

/-- Represents the travel agency problem --/
theorem travel_agency_problem 
  (seats_per_bus : ℕ) 
  (incomplete_bus_2005 : ℕ) 
  (increase_2006 : ℕ) 
  (h1 : seats_per_bus = 27)
  (h2 : incomplete_bus_2005 = 19)
  (h3 : increase_2006 = 53) :
  ∃ (k : ℕ),
    (seats_per_bus * k + incomplete_bus_2005 + increase_2006) / seats_per_bus - 
    (seats_per_bus * k + incomplete_bus_2005) / seats_per_bus = 2 ∧
    (seats_per_bus * k + incomplete_bus_2005 + increase_2006) % seats_per_bus = 9 :=
by sorry

end NUMINAMATH_CALUDE_travel_agency_problem_l3193_319384


namespace NUMINAMATH_CALUDE_unique_vector_b_l3193_319351

def a : ℝ × ℝ := (-4, 3)
def c : ℝ × ℝ := (1, 1)

def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

def acute_angle (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 > 0

theorem unique_vector_b :
  ∃! b : ℝ × ℝ,
    collinear b a ∧
    ‖b‖ = 10 ∧
    acute_angle b c ∧
    b = (8, -6) := by sorry

end NUMINAMATH_CALUDE_unique_vector_b_l3193_319351


namespace NUMINAMATH_CALUDE_inverse_j_minus_j_inv_l3193_319370

-- Define the complex number i
def i : ℂ := Complex.I

-- Define j in terms of i
def j : ℂ := i + 1

-- Theorem statement
theorem inverse_j_minus_j_inv :
  (j - j⁻¹)⁻¹ = (-3 * i + 1) / 5 :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_j_minus_j_inv_l3193_319370


namespace NUMINAMATH_CALUDE_problem_solution_l3193_319393

def digits : Finset Nat := {0, 1, 2, 3, 4, 5}

def naturalNumbersWithoutRepeats (d : Finset Nat) : Nat :=
  sorry

def fourDigitEvenWithoutRepeats (d : Finset Nat) : Nat :=
  sorry

def fourDigitGreaterThan4023WithoutRepeats (d : Finset Nat) : Nat :=
  sorry

theorem problem_solution (d : Finset Nat) (h : d = digits) :
  naturalNumbersWithoutRepeats d = 1631 ∧
  fourDigitEvenWithoutRepeats d = 156 ∧
  fourDigitGreaterThan4023WithoutRepeats d = 115 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3193_319393


namespace NUMINAMATH_CALUDE_marks_age_difference_l3193_319321

theorem marks_age_difference (mark_current_age : ℕ) (aaron_current_age : ℕ) : 
  mark_current_age = 28 →
  mark_current_age + 4 = 2 * (aaron_current_age + 4) + 2 →
  (mark_current_age - 3) - 3 * (aaron_current_age - 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_marks_age_difference_l3193_319321


namespace NUMINAMATH_CALUDE_pie_count_correct_l3193_319383

/-- Represents the number of pie slices served in a meal -/
structure MealServing :=
  (apple : ℕ)
  (blueberry : ℕ)
  (cherry : ℕ)
  (pumpkin : ℕ)

/-- Represents the total number of pie slices served over two days -/
structure TotalServing :=
  (apple : ℕ)
  (blueberry : ℕ)
  (cherry : ℕ)
  (pumpkin : ℕ)

def lunch_today : MealServing := ⟨3, 2, 2, 0⟩
def dinner_today : MealServing := ⟨1, 2, 1, 1⟩
def yesterday : MealServing := ⟨8, 8, 0, 0⟩

def total_served : TotalServing := ⟨12, 12, 3, 1⟩

theorem pie_count_correct : 
  lunch_today.apple + dinner_today.apple + yesterday.apple = total_served.apple ∧
  lunch_today.blueberry + dinner_today.blueberry + yesterday.blueberry = total_served.blueberry ∧
  lunch_today.cherry + dinner_today.cherry + yesterday.cherry = total_served.cherry ∧
  lunch_today.pumpkin + dinner_today.pumpkin + yesterday.pumpkin = total_served.pumpkin :=
by sorry

end NUMINAMATH_CALUDE_pie_count_correct_l3193_319383


namespace NUMINAMATH_CALUDE_boat_transport_two_days_l3193_319336

/-- The number of people a boat can transport in multiple days -/
def boat_transport (capacity : ℕ) (trips_per_day : ℕ) (days : ℕ) : ℕ :=
  capacity * trips_per_day * days

/-- Theorem: A boat with capacity 12 making 4 trips per day can transport 96 people in 2 days -/
theorem boat_transport_two_days :
  boat_transport 12 4 2 = 96 := by
  sorry

end NUMINAMATH_CALUDE_boat_transport_two_days_l3193_319336


namespace NUMINAMATH_CALUDE_solution_set_theorem_inequality_theorem_l3193_319332

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Theorem for part I
theorem solution_set_theorem :
  {x : ℝ | f (x - 1) + f (x + 3) ≥ 6} = {x : ℝ | x ≤ -3 ∨ x ≥ 3} := by sorry

-- Theorem for part II
theorem inequality_theorem (a b : ℝ) (ha : |a| < 1) (hb : |b| < 1) (ha_neq_zero : a ≠ 0) :
  f (a * b) > |a| * f (b / a) := by sorry

end NUMINAMATH_CALUDE_solution_set_theorem_inequality_theorem_l3193_319332


namespace NUMINAMATH_CALUDE_sin_2theta_value_l3193_319322

theorem sin_2theta_value (θ : Real) :
  (π < θ ∧ θ < 3*π/2) →  -- θ is in the third quadrant
  (Real.sin θ)^4 + (Real.cos θ)^4 = 5/9 →
  Real.sin (2*θ) = 2*Real.sqrt 2/3 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l3193_319322


namespace NUMINAMATH_CALUDE_append_five_to_two_digit_number_l3193_319376

/-- Given a two-digit number with tens' digit t and units' digit u,
    when the digit 5 is placed after this number,
    the resulting number is equal to 100t + 10u + 5. -/
theorem append_five_to_two_digit_number (t u : ℕ) 
  (h1 : t ≥ 1 ∧ t ≤ 9) (h2 : u ≥ 0 ∧ u ≤ 9) :
  (10 * t + u) * 10 + 5 = 100 * t + 10 * u + 5 := by
  sorry

end NUMINAMATH_CALUDE_append_five_to_two_digit_number_l3193_319376


namespace NUMINAMATH_CALUDE_complex_square_equals_negative_100_minus_64i_l3193_319348

theorem complex_square_equals_negative_100_minus_64i :
  ∀ z : ℂ, z^2 = -100 - 64*I ↔ z = 4 - 8*I ∨ z = -4 + 8*I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_equals_negative_100_minus_64i_l3193_319348


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l3193_319349

/-- 
Given a binomial expansion $(ax^m + bx^n)^{12}$ with specific conditions,
this theorem proves properties about the constant term and the range of $\frac{a}{b}$.
-/
theorem binomial_expansion_properties 
  (a b : ℝ) (m n : ℤ) 
  (ha : a > 0) (hb : b > 0) (hm : m ≠ 0) (hn : n ≠ 0) (hmn : 2*m + n = 0) :
  (∃ (r : ℕ), r = 4 ∧ m*(12 - r) + n*r = 0) ∧ 
  (8/5 ≤ a/b ∧ a/b ≤ 9/4) := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l3193_319349


namespace NUMINAMATH_CALUDE_parallel_lines_m_values_l3193_319361

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  (b1 ≠ 0 ∧ b2 ≠ 0 ∧ a1 / b1 = a2 / b2) ∨ (b1 = 0 ∧ b2 = 0)

/-- The main theorem -/
theorem parallel_lines_m_values (m : ℝ) :
  are_parallel 1 (1+m) (m-2) m 2 6 → m = -2 ∨ m = 1 := by
  sorry


end NUMINAMATH_CALUDE_parallel_lines_m_values_l3193_319361


namespace NUMINAMATH_CALUDE_estimate_greater_than_exact_l3193_319339

theorem estimate_greater_than_exact (a b c d : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (a' b' c' d' : ℕ) 
  (ha' : a' ≥ a) (hb' : b' ≤ b) (hc' : c' ≥ c) (hd' : d' ≥ d) : 
  (a' : ℚ) / b' + c' - d' > (a : ℚ) / b + c - d :=
by sorry

end NUMINAMATH_CALUDE_estimate_greater_than_exact_l3193_319339


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l3193_319303

/-- The quadratic polynomial q(x) -/
def q (x : ℚ) : ℚ := (20/7) * x^2 - (60/7) * x - 360/7

/-- Theorem stating that q(x) satisfies the given conditions -/
theorem q_satisfies_conditions :
  q (-3) = 0 ∧ q 6 = 0 ∧ q (-1) = -40 := by sorry

end NUMINAMATH_CALUDE_q_satisfies_conditions_l3193_319303


namespace NUMINAMATH_CALUDE_sum_of_two_primes_10003_l3193_319374

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

theorem sum_of_two_primes_10003 :
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ p + q = 10003 :=
sorry

end NUMINAMATH_CALUDE_sum_of_two_primes_10003_l3193_319374


namespace NUMINAMATH_CALUDE_shadow_height_calculation_l3193_319337

/-- Given a lamppost and a person casting shadows under the same light source,
    calculate the person's height using the ratio method. -/
theorem shadow_height_calculation
  (lamppost_height : ℝ)
  (lamppost_shadow : ℝ)
  (michael_shadow : ℝ)
  (h_lamppost_height : lamppost_height = 50)
  (h_lamppost_shadow : lamppost_shadow = 25)
  (h_michael_shadow : michael_shadow = 20 / 12)  -- Convert 20 inches to feet
  : ∃ (michael_height : ℝ),
    michael_height = (lamppost_height / lamppost_shadow) * michael_shadow ∧
    michael_height * 12 = 40 := by
  sorry

end NUMINAMATH_CALUDE_shadow_height_calculation_l3193_319337


namespace NUMINAMATH_CALUDE_sum_ratio_simplification_main_result_l3193_319344

def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => n * double_factorial n

def sum_ratio (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i => (double_factorial (2*i+1)) / (double_factorial (2*i+2)))

theorem sum_ratio_simplification (n : ℕ) :
  ∃ (c : ℕ), Odd c ∧ sum_ratio n = c / 2^(2*n - 7) := by sorry

theorem main_result :
  ∃ (c : ℕ), Odd c ∧ sum_ratio 2010 = c / 2^4013 ∧ 4013 / 10 = 401.3 := by sorry

end NUMINAMATH_CALUDE_sum_ratio_simplification_main_result_l3193_319344


namespace NUMINAMATH_CALUDE_clock_strike_duration_clock_strike_six_duration_l3193_319308

-- Define the clock striking behavior
def ClockStrike (strikes : ℕ) (duration : ℝ) : Prop :=
  strikes > 0 ∧ duration > 0 ∧ (strikes - 1) * (duration / (strikes - 1)) = duration

-- Theorem statement
theorem clock_strike_duration (strikes₁ strikes₂ : ℕ) (duration₁ : ℝ) :
  ClockStrike strikes₁ duration₁ →
  strikes₂ > strikes₁ →
  ClockStrike strikes₂ ((strikes₂ - 1) * (duration₁ / (strikes₁ - 1))) :=
by
  sorry

-- The specific problem instance
theorem clock_strike_six_duration :
  ClockStrike 3 12 → ClockStrike 6 30 :=
by
  sorry

end NUMINAMATH_CALUDE_clock_strike_duration_clock_strike_six_duration_l3193_319308


namespace NUMINAMATH_CALUDE_james_sticker_collection_l3193_319342

theorem james_sticker_collection (initial : ℕ) (gift : ℕ) (given_away : ℕ) 
  (h1 : initial = 478) 
  (h2 : gift = 182) 
  (h3 : given_away = 276) : 
  initial + gift - given_away = 384 := by
  sorry

end NUMINAMATH_CALUDE_james_sticker_collection_l3193_319342


namespace NUMINAMATH_CALUDE_equation_solutions_l3193_319307

theorem equation_solutions :
  (∃ x : ℝ, x - 2 * (5 + x) = -4 ∧ x = -6) ∧
  (∃ x : ℝ, (2 * x - 1) / 2 = 1 - (3 - x) / 4 ∧ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3193_319307


namespace NUMINAMATH_CALUDE_tenth_graders_truth_count_l3193_319397

def is_valid_response_count (n : ℕ) (truth_tellers : ℕ) : Prop :=
  n > 0 ∧ truth_tellers ≤ n ∧ 
  (truth_tellers * (n - 1) + (n - truth_tellers) * truth_tellers = 44) ∧
  (truth_tellers * (n - truth_tellers) + (n - truth_tellers) * (n - 1 - truth_tellers) = 28)

theorem tenth_graders_truth_count :
  ∃ (n : ℕ) (t : ℕ), 
    is_valid_response_count n t ∧ 
    (t * (n - 1) = 16 ∨ t * (n - 1) = 56) := by
  sorry

end NUMINAMATH_CALUDE_tenth_graders_truth_count_l3193_319397


namespace NUMINAMATH_CALUDE_construction_materials_cost_l3193_319330

/-- The total cost of materials bought by the construction company -/
def total_cost (gravel_tons sand_tons cement_tons : ℝ) 
               (gravel_price sand_price cement_price : ℝ) : ℝ :=
  gravel_tons * gravel_price + sand_tons * sand_price + cement_tons * cement_price

/-- Theorem stating the total cost of materials -/
theorem construction_materials_cost : 
  total_cost 5.91 8.11 4.35 30.50 40.50 55.60 = 750.57 := by
  sorry

end NUMINAMATH_CALUDE_construction_materials_cost_l3193_319330


namespace NUMINAMATH_CALUDE_point_inside_circle_a_range_l3193_319389

theorem point_inside_circle_a_range (a : ℝ) : 
  (((1 - a)^2 + (1 + a)^2) < 4) → (-1 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_point_inside_circle_a_range_l3193_319389


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_eccentricity_l3193_319399

theorem ellipse_hyperbola_eccentricity (m n : ℝ) (e₁ e₂ : ℝ) : 
  m > 1 → 
  n > 0 → 
  (∀ x y : ℝ, x^2 / m^2 + y^2 = 1 ↔ x^2 / n^2 - y^2 = 1) → 
  e₁ = Real.sqrt (1 - 1 / m^2) → 
  e₂ = Real.sqrt (1 + 1 / n^2) → 
  m > n ∧ e₁ * e₂ > 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_eccentricity_l3193_319399


namespace NUMINAMATH_CALUDE_rod_length_l3193_319380

/-- Given a rod from which 40 pieces of 85 cm each can be cut, prove that its length is 3400 cm. -/
theorem rod_length (num_pieces : ℕ) (piece_length : ℕ) (h1 : num_pieces = 40) (h2 : piece_length = 85) :
  num_pieces * piece_length = 3400 := by
  sorry

end NUMINAMATH_CALUDE_rod_length_l3193_319380


namespace NUMINAMATH_CALUDE_sequence_sum_100_l3193_319311

/-- Sequence sum type -/
def SequenceSum (a : ℕ+ → ℝ) : ℕ+ → ℝ 
  | n => (Finset.range n).sum (fun i => a ⟨i + 1, Nat.succ_pos i⟩)

/-- Main theorem -/
theorem sequence_sum_100 (a : ℕ+ → ℝ) (t : ℝ) : 
  (∀ n : ℕ+, a n > 0) → 
  a 1 = 1 → 
  (∀ n : ℕ+, 2 * SequenceSum a n = a n * (a n + t)) → 
  SequenceSum a 100 = 5050 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_100_l3193_319311


namespace NUMINAMATH_CALUDE_similar_right_triangles_leg_l3193_319363

/-- Given two similar right triangles, where one has legs 12 and 9, and the other has legs y and 6, prove that y = 8 -/
theorem similar_right_triangles_leg (y : ℝ) : 
  (12 : ℝ) / y = 9 / 6 → y = 8 := by sorry

end NUMINAMATH_CALUDE_similar_right_triangles_leg_l3193_319363


namespace NUMINAMATH_CALUDE_projection_vector_l3193_319320

/-- Given two plane vectors a and b, prove that the projection of b onto a is (-1, 2) -/
theorem projection_vector (a b : ℝ × ℝ) : 
  a = (1, -2) → b = (3, 4) → 
  (((a.1 * b.1 + a.2 * b.2) / (a.1^2 + a.2^2)) • a) = (-1, 2) := by
  sorry

end NUMINAMATH_CALUDE_projection_vector_l3193_319320


namespace NUMINAMATH_CALUDE_angle_measure_proof_l3193_319377

/-- Given two supplementary angles C and D, where the measure of angle C is 5 times
    the measure of angle D, prove that the measure of angle C is 150°. -/
theorem angle_measure_proof (C D : ℝ) : 
  C + D = 180 →  -- Angles C and D are supplementary
  C = 5 * D →    -- Measure of angle C is 5 times angle D
  C = 150 := by  -- Measure of angle C is 150°
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l3193_319377


namespace NUMINAMATH_CALUDE_quadratic_root_transformation_l3193_319341

theorem quadratic_root_transformation (p q r u v : ℝ) : 
  (p * u^2 + q * u + r = 0) → 
  (p * v^2 + q * v + r = 0) → 
  ((p^2 * u + q)^2 - q * (p^2 * u + q) + p * r = 0) ∧
  ((p^2 * v + q)^2 - q * (p^2 * v + q) + p * r = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_transformation_l3193_319341


namespace NUMINAMATH_CALUDE_angle_between_vectors_l3193_319306

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/-- Given nonzero vectors a and b such that ||a|| = ||b|| = 2||a + b||,
    the cosine of the angle between them is -7/8 -/
theorem angle_between_vectors (a b : E) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : ‖a‖ = ‖b‖ ∧ ‖a‖ = 2 * ‖a + b‖) : 
  inner a b / (‖a‖ * ‖b‖) = -7/8 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l3193_319306


namespace NUMINAMATH_CALUDE_grid_puzzle_solution_l3193_319381

-- Define the type for our grid cells
def Cell := Fin 16

-- Define our grid
structure Grid :=
  (A B C D E F G H J K L M N P Q R : Cell)

-- Define the conditions
def conditions (g : Grid) : Prop :=
  (g.A.val + g.C.val + g.F.val = 10) ∧
  (g.B.val + g.H.val = g.R.val) ∧
  (g.D.val - g.C.val = 13) ∧
  (g.E.val * g.M.val = 126) ∧
  (g.F.val + g.G.val = 21) ∧
  (g.G.val / g.J.val = 2) ∧
  (g.H.val * g.M.val = 36) ∧
  (g.J.val * g.P.val = 80) ∧
  (g.K.val - g.N.val = g.Q.val) ∧
  (∀ i j : Fin 16, i ≠ j → 
    g.A.val ≠ g.B.val ∧ g.A.val ≠ g.C.val ∧ g.A.val ≠ g.D.val ∧
    g.A.val ≠ g.E.val ∧ g.A.val ≠ g.F.val ∧ g.A.val ≠ g.G.val ∧
    g.A.val ≠ g.H.val ∧ g.A.val ≠ g.J.val ∧ g.A.val ≠ g.K.val ∧
    g.A.val ≠ g.L.val ∧ g.A.val ≠ g.M.val ∧ g.A.val ≠ g.N.val ∧
    g.A.val ≠ g.P.val ∧ g.A.val ≠ g.Q.val ∧ g.A.val ≠ g.R.val ∧
    g.B.val ≠ g.C.val ∧ g.B.val ≠ g.D.val ∧ g.B.val ≠ g.E.val ∧
    g.B.val ≠ g.F.val ∧ g.B.val ≠ g.G.val ∧ g.B.val ≠ g.H.val ∧
    g.B.val ≠ g.J.val ∧ g.B.val ≠ g.K.val ∧ g.B.val ≠ g.L.val ∧
    g.B.val ≠ g.M.val ∧ g.B.val ≠ g.N.val ∧ g.B.val ≠ g.P.val ∧
    g.B.val ≠ g.Q.val ∧ g.B.val ≠ g.R.val ∧ g.C.val ≠ g.D.val ∧
    g.C.val ≠ g.E.val ∧ g.C.val ≠ g.F.val ∧ g.C.val ≠ g.G.val ∧
    g.C.val ≠ g.H.val ∧ g.C.val ≠ g.J.val ∧ g.C.val ≠ g.K.val ∧
    g.C.val ≠ g.L.val ∧ g.C.val ≠ g.M.val ∧ g.C.val ≠ g.N.val ∧
    g.C.val ≠ g.P.val ∧ g.C.val ≠ g.Q.val ∧ g.C.val ≠ g.R.val ∧
    g.D.val ≠ g.E.val ∧ g.D.val ≠ g.F.val ∧ g.D.val ≠ g.G.val ∧
    g.D.val ≠ g.H.val ∧ g.D.val ≠ g.J.val ∧ g.D.val ≠ g.K.val ∧
    g.D.val ≠ g.L.val ∧ g.D.val ≠ g.M.val ∧ g.D.val ≠ g.N.val ∧
    g.D.val ≠ g.P.val ∧ g.D.val ≠ g.Q.val ∧ g.D.val ≠ g.R.val ∧
    g.E.val ≠ g.F.val ∧ g.E.val ≠ g.G.val ∧ g.E.val ≠ g.H.val ∧
    g.E.val ≠ g.J.val ∧ g.E.val ≠ g.K.val ∧ g.E.val ≠ g.L.val ∧
    g.E.val ≠ g.M.val ∧ g.E.val ≠ g.N.val ∧ g.E.val ≠ g.P.val ∧
    g.E.val ≠ g.Q.val ∧ g.E.val ≠ g.R.val ∧ g.F.val ≠ g.G.val ∧
    g.F.val ≠ g.H.val ∧ g.F.val ≠ g.J.val ∧ g.F.val ≠ g.K.val ∧
    g.F.val ≠ g.L.val ∧ g.F.val ≠ g.M.val ∧ g.F.val ≠ g.N.val ∧
    g.F.val ≠ g.P.val ∧ g.F.val ≠ g.Q.val ∧ g.F.val ≠ g.R.val ∧
    g.G.val ≠ g.H.val ∧ g.G.val ≠ g.J.val ∧ g.G.val ≠ g.K.val ∧
    g.G.val ≠ g.L.val ∧ g.G.val ≠ g.M.val ∧ g.G.val ≠ g.N.val ∧
    g.G.val ≠ g.P.val ∧ g.G.val ≠ g.Q.val ∧ g.G.val ≠ g.R.val ∧
    g.H.val ≠ g.J.val ∧ g.H.val ≠ g.K.val ∧ g.H.val ≠ g.L.val ∧
    g.H.val ≠ g.M.val ∧ g.H.val ≠ g.N.val ∧ g.H.val ≠ g.P.val ∧
    g.H.val ≠ g.Q.val ∧ g.H.val ≠ g.R.val ∧ g.J.val ≠ g.K.val ∧
    g.J.val ≠ g.L.val ∧ g.J.val ≠ g.M.val ∧ g.J.val ≠ g.N.val ∧
    g.J.val ≠ g.P.val ∧ g.J.val ≠ g.Q.val ∧ g.J.val ≠ g.R.val ∧
    g.K.val ≠ g.L.val ∧ g.K.val ≠ g.M.val ∧ g.K.val ≠ g.N.val ∧
    g.K.val ≠ g.P.val ∧ g.K.val ≠ g.Q.val ∧ g.K.val ≠ g.R.val ∧
    g.L.val ≠ g.M.val ∧ g.L.val ≠ g.N.val ∧ g.L.val ≠ g.P.val ∧
    g.L.val ≠ g.Q.val ∧ g.L.val ≠ g.R.val ∧ g.M.val ≠ g.N.val ∧
    g.M.val ≠ g.P.val ∧ g.M.val ≠ g.Q.val ∧ g.M.val ≠ g.R.val ∧
    g.N.val ≠ g.P.val ∧ g.N.val ≠ g.Q.val ∧ g.N.val ≠ g.R.val ∧
    g.P.val ≠ g.Q.val ∧ g.P.val ≠ g.R.val ∧ g.Q.val ≠ g.R.val)

-- State the theorem
theorem grid_puzzle_solution (g : Grid) (h : conditions g) : g.L.val = 6 := by
  sorry

end NUMINAMATH_CALUDE_grid_puzzle_solution_l3193_319381


namespace NUMINAMATH_CALUDE_max_cuboid_path_length_l3193_319347

noncomputable def max_path_length (a b c : ℝ) : ℝ :=
  4 * Real.sqrt (a^2 + b^2 + c^2) + 
  4 * Real.sqrt (max a b * max b c) + 
  min a (min b c) + 
  max a (max b c)

theorem max_cuboid_path_length :
  max_path_length 2 2 1 = 12 + 8 * Real.sqrt 2 + 3 := by
  sorry

end NUMINAMATH_CALUDE_max_cuboid_path_length_l3193_319347


namespace NUMINAMATH_CALUDE_coloring_book_shelves_l3193_319357

/-- Given a store's coloring book inventory and sales, calculate the number of shelves needed to display the remaining books. -/
theorem coloring_book_shelves (initial_stock : ℕ) (books_sold : ℕ) (books_per_shelf : ℕ) :
  initial_stock = 86 →
  books_sold = 37 →
  books_per_shelf = 7 →
  (initial_stock - books_sold) / books_per_shelf = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_coloring_book_shelves_l3193_319357


namespace NUMINAMATH_CALUDE_max_tiles_on_floor_l3193_319317

/-- Represents the dimensions of a rectangle -/
structure Dimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℝ := d.length * d.width

/-- Represents the floor and tile dimensions -/
def floor : Dimensions := { length := 400, width := 600 }
def tile : Dimensions := { length := 20, width := 30 }

/-- Theorem stating the maximum number of tiles that can fit on the floor -/
theorem max_tiles_on_floor :
  (area floor / area tile : ℝ) = 400 := by sorry

end NUMINAMATH_CALUDE_max_tiles_on_floor_l3193_319317


namespace NUMINAMATH_CALUDE_arithmetic_geometric_progression_l3193_319350

-- Arithmetic Progression
def arithmetic_sum (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (a + (n - 1) * d / 2)

-- Geometric Progression
def geometric_product (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a^n * r^(n * (n - 1) / 2)

theorem arithmetic_geometric_progression :
  (arithmetic_sum 0 (1/3) 15 = 35) ∧
  (geometric_product 1 (10^(1/3)) 15 = 10^35) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_progression_l3193_319350


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_l3193_319328

theorem sum_of_roots_cubic (x : ℝ) : 
  (x + 2)^2 * (x - 3) = 40 → 
  ∃ (r₁ r₂ r₃ : ℝ), r₁ + r₂ + r₃ = -1 ∧ 
    ((x = r₁) ∨ (x = r₂) ∨ (x = r₃)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_l3193_319328


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l3193_319358

theorem system_of_equations_solution :
  ∃ (x y : ℚ), (4 * x - 3 * y = -14) ∧ (5 * x + 3 * y = -12) ∧ (x = -26/9) ∧ (y = -22/27) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l3193_319358


namespace NUMINAMATH_CALUDE_fraction_comparison_l3193_319395

theorem fraction_comparison (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) : 
  b / (a - c) < a / (b - d) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l3193_319395


namespace NUMINAMATH_CALUDE_quadratic_function_unique_solution_l3193_319386

/-- Given a quadratic function f(x) = ax² + bx + c, prove that if f(-1) = 3, f(0) = 1, and f(1) = 1, then a = 1, b = -1, and c = 1. -/
theorem quadratic_function_unique_solution (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = (fun x => if x = -1 then 3 else if x = 0 then 1 else if x = 1 then 1 else 0) x) →
  a = 1 ∧ b = -1 ∧ c = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_solution_l3193_319386


namespace NUMINAMATH_CALUDE_trailing_zeros_50_factorial_l3193_319340

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Theorem: The number of trailing zeros in 50! is 12 -/
theorem trailing_zeros_50_factorial :
  trailingZeros 50 = 12 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_50_factorial_l3193_319340


namespace NUMINAMATH_CALUDE_seventeen_in_sample_l3193_319365

/-- Represents a systematic sampling with equal intervals -/
structure SystematicSampling where
  start : ℕ
  interval : ℕ
  size : ℕ

/-- Checks if a number is included in the systematic sampling -/
def isInSample (s : SystematicSampling) (n : ℕ) : Prop :=
  ∃ k : ℕ, k < s.size ∧ n = s.start + k * s.interval

/-- Theorem: Given a systematic sampling that includes 5, 23, and 29, it also includes 17 -/
theorem seventeen_in_sample (s : SystematicSampling) 
  (h5 : isInSample s 5) 
  (h23 : isInSample s 23) 
  (h29 : isInSample s 29) : 
  isInSample s 17 := by
  sorry

end NUMINAMATH_CALUDE_seventeen_in_sample_l3193_319365


namespace NUMINAMATH_CALUDE_cos_four_pi_thirds_l3193_319315

theorem cos_four_pi_thirds : Real.cos (4 * Real.pi / 3) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_four_pi_thirds_l3193_319315


namespace NUMINAMATH_CALUDE_birth_rate_calculation_l3193_319300

/-- The average birth rate in the city (people per 2 seconds) -/
def average_birth_rate : ℝ := sorry

/-- The death rate in the city (people per 2 seconds) -/
def death_rate : ℝ := 2

/-- The net population increase in one day -/
def daily_net_increase : ℕ := 172800

/-- The number of 2-second intervals in a day -/
def intervals_per_day : ℕ := 24 * 60 * 60 / 2

theorem birth_rate_calculation :
  average_birth_rate = 6 :=
sorry

end NUMINAMATH_CALUDE_birth_rate_calculation_l3193_319300


namespace NUMINAMATH_CALUDE_seventy_fifth_term_of_sequence_l3193_319359

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem seventy_fifth_term_of_sequence :
  arithmetic_sequence 2 4 75 = 298 := by sorry

end NUMINAMATH_CALUDE_seventy_fifth_term_of_sequence_l3193_319359


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l3193_319385

/-- The side length of an equilateral triangle with inscribed circle and smaller touching circles -/
theorem equilateral_triangle_side_length (r : ℝ) (h : r > 0) : ∃ a : ℝ, 
  a > 0 ∧ 
  (∃ R : ℝ, R > 0 ∧ 
    -- R is the radius of the inscribed circle
    R = (a * Real.sqrt 3) / 6 ∧
    -- Relationship between R, r, and the altitude of the triangle
    R / r = (a * Real.sqrt 3 / 3) / (a * Real.sqrt 3 / 3 - R - r)) ∧
  a = 6 * r * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l3193_319385


namespace NUMINAMATH_CALUDE_ram_work_time_l3193_319396

/-- Ram's efficiency compared to Krish's -/
def ram_efficiency : ℚ := 1/2

/-- Time taken by Ram and Krish working together (in days) -/
def combined_time : ℕ := 7

/-- Time taken by Ram working alone (in days) -/
def ram_alone_time : ℕ := 21

theorem ram_work_time :
  ram_efficiency * combined_time * 2 = ram_alone_time := by
  sorry

end NUMINAMATH_CALUDE_ram_work_time_l3193_319396


namespace NUMINAMATH_CALUDE_boy_travel_time_l3193_319367

/-- Proves that given the conditions of the problem, the boy arrives 8 minutes early on the second day -/
theorem boy_travel_time (distance : ℝ) (speed_day1 speed_day2 : ℝ) (late_time : ℝ) : 
  distance = 2.5 →
  speed_day1 = 5 →
  speed_day2 = 10 →
  late_time = 7 / 60 →
  let time_day1 : ℝ := distance / speed_day1
  let on_time : ℝ := time_day1 - late_time
  let time_day2 : ℝ := distance / speed_day2
  (on_time - time_day2) * 60 = 8 := by sorry

end NUMINAMATH_CALUDE_boy_travel_time_l3193_319367


namespace NUMINAMATH_CALUDE_train_speed_calculation_l3193_319312

/-- Proves that given the conditions of a jogger and a train, the train's speed is 36 km/hr -/
theorem train_speed_calculation (jogger_speed : ℝ) (distance_ahead : ℝ) (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 →
  distance_ahead = 240 →
  train_length = 130 →
  passing_time = 37 →
  ∃ (train_speed : ℝ), train_speed = 36 :=
by
  sorry


end NUMINAMATH_CALUDE_train_speed_calculation_l3193_319312


namespace NUMINAMATH_CALUDE_recurrence_sequence_general_term_l3193_319329

/-- A sequence of positive real numbers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  a 1 = 1 ∧
  ∀ n, (2 * a (n + 1) - 1) * a n - 2 * a (n + 1) = 0

/-- The theorem stating that the sequence satisfying the recurrence relation
    has the general term a_n = 1/(2^(n-1)) -/
theorem recurrence_sequence_general_term (a : ℕ → ℝ) 
    (h : RecurrenceSequence a) : 
    ∀ n, a n = 1 / (2 ^ (n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_recurrence_sequence_general_term_l3193_319329


namespace NUMINAMATH_CALUDE_cuboid_volume_problem_l3193_319326

theorem cuboid_volume_problem (x y : ℕ) : 
  (x > 0) → 
  (y > 0) → 
  (x < 4) → 
  (y < 15) → 
  (15 * 5 * 4 - x * 5 * y = 120) → 
  (x + y = 15) := by
sorry

end NUMINAMATH_CALUDE_cuboid_volume_problem_l3193_319326


namespace NUMINAMATH_CALUDE_min_abs_phi_l3193_319338

/-- Given a function y = 2sin(x + φ), prove that if the abscissa is shortened to 1/3
    and the graph is shifted right by π/4, resulting in symmetry about (π/3, 0),
    then the minimum value of |φ| is π/4. -/
theorem min_abs_phi (φ : Real) : 
  (∀ x, 2 * Real.sin (3 * x + φ - 3 * Real.pi / 4) = 
        2 * Real.sin (3 * (2 * Real.pi / 3 - x) + φ - 3 * Real.pi / 4)) → 
  (∃ k : ℤ, φ = Real.pi / 4 + k * Real.pi) ∧ 
  (∀ ψ : Real, (∃ k : ℤ, ψ = Real.pi / 4 + k * Real.pi) → |ψ| ≥ |φ|) := by
  sorry

end NUMINAMATH_CALUDE_min_abs_phi_l3193_319338


namespace NUMINAMATH_CALUDE_ada_original_seat_l3193_319324

-- Define the type for seats
inductive Seat
| one | two | three | four | five | six

-- Define the type for friends
inductive Friend
| ada | bea | ceci | dee | edie | fana

-- Define the initial seating arrangement
def initial_seating : Friend → Seat := sorry

-- Define the movement function
def move (s : Seat) (n : Int) : Seat := sorry

-- Define the final seating arrangement after movements
def final_seating : Friend → Seat := sorry

-- Theorem to prove
theorem ada_original_seat :
  (∀ f : Friend, f ≠ Friend.ada → final_seating f ≠ initial_seating f) →
  (final_seating Friend.ada = Seat.one ∨ final_seating Friend.ada = Seat.six) →
  initial_seating Friend.ada = Seat.two :=
sorry

end NUMINAMATH_CALUDE_ada_original_seat_l3193_319324


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3193_319382

theorem inequality_solution_set (c : ℝ) : 
  (c / 3 ≤ 2 + c ∧ 2 + c < -2 * (1 + c)) ↔ c ∈ Set.Icc (-3) (-4/3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3193_319382


namespace NUMINAMATH_CALUDE_sqrt_221_between_14_and_15_l3193_319354

theorem sqrt_221_between_14_and_15 : 14 < Real.sqrt 221 ∧ Real.sqrt 221 < 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_221_between_14_and_15_l3193_319354


namespace NUMINAMATH_CALUDE_phone_answer_probability_l3193_319391

theorem phone_answer_probability (p1 p2 p3 p4 : ℚ) 
  (h1 : p1 = 1/10)
  (h2 : p2 = 1/5)
  (h3 : p3 = 3/10)
  (h4 : p4 = 1/10) :
  p1 + p2 + p3 + p4 = 7/10 := by
  sorry

#check phone_answer_probability

end NUMINAMATH_CALUDE_phone_answer_probability_l3193_319391


namespace NUMINAMATH_CALUDE_bus_profit_at_2600_passengers_l3193_319319

/-- Represents the monthly profit of a minibus based on the number of passengers -/
def monthly_profit (passengers : ℕ) : ℤ :=
  2 * (passengers : ℤ) - 5000

/-- Theorem stating that the bus makes a profit with 2600 passengers -/
theorem bus_profit_at_2600_passengers :
  monthly_profit 2600 > 0 := by
  sorry

end NUMINAMATH_CALUDE_bus_profit_at_2600_passengers_l3193_319319


namespace NUMINAMATH_CALUDE_x_fourth_minus_reciprocal_l3193_319356

theorem x_fourth_minus_reciprocal (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/x^4 = 727 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_minus_reciprocal_l3193_319356


namespace NUMINAMATH_CALUDE_cubic_function_properties_l3193_319345

/-- The cubic function f(x) = x³ - kx + k² -/
def f (k : ℝ) (x : ℝ) : ℝ := x^3 - k*x + k^2

theorem cubic_function_properties (k : ℝ) :
  (∀ x y, x < y → f k x < f k y) ∨ 
  ((∃ x y z, x < y ∧ y < z ∧ f k x = 0 ∧ f k y = 0 ∧ f k z = 0) ↔ 0 < k ∧ k < 4/27) :=
sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l3193_319345


namespace NUMINAMATH_CALUDE_g_magnitude_l3193_319373

/-- A quadratic function that is even on a specific interval -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * a

/-- The function g defined as a transformation of f -/
def g (a : ℝ) (x : ℝ) : ℝ := f a (x - 1)

/-- Theorem stating the relative magnitudes of g at specific points -/
theorem g_magnitude (a : ℝ) (h : ∀ x ∈ Set.Icc (-a) (a^2), f a x = f a (-x)) :
  g a (3/2) < g a 0 ∧ g a 0 < g a 3 := by
  sorry

end NUMINAMATH_CALUDE_g_magnitude_l3193_319373


namespace NUMINAMATH_CALUDE_carlas_order_cost_l3193_319310

/-- The original cost of Carla's order at McDonald's -/
def original_cost : ℝ := 7.50

/-- The coupon value -/
def coupon_value : ℝ := 2.50

/-- The senior discount percentage -/
def senior_discount : ℝ := 0.20

/-- The final amount Carla pays -/
def final_payment : ℝ := 4.00

/-- Theorem stating that the original cost is correct given the conditions -/
theorem carlas_order_cost :
  (original_cost - coupon_value) * (1 - senior_discount) = final_payment :=
by sorry

end NUMINAMATH_CALUDE_carlas_order_cost_l3193_319310


namespace NUMINAMATH_CALUDE_divisibility_of_sum_of_powers_l3193_319335

theorem divisibility_of_sum_of_powers (a b : ℤ) (n : ℕ) :
  (a + b) ∣ (a^(2*n + 1) + b^(2*n + 1)) := by sorry

end NUMINAMATH_CALUDE_divisibility_of_sum_of_powers_l3193_319335


namespace NUMINAMATH_CALUDE_area_of_region_R_l3193_319366

/-- A square with side length 3 -/
structure Square :=
  (side_length : ℝ)
  (is_three : side_length = 3)

/-- The region R within the square -/
def region_R (s : Square) := {p : ℝ × ℝ | 
  p.1 ≥ 0 ∧ p.1 ≤ s.side_length ∧ 
  p.2 ≥ 0 ∧ p.2 ≤ s.side_length ∧
  (p.1 - s.side_length)^2 + p.2^2 < p.1^2 + p.2^2 ∧
  (p.1 - s.side_length)^2 + p.2^2 < p.1^2 + (p.2 - s.side_length)^2 ∧
  (p.1 - s.side_length)^2 + p.2^2 < (p.1 - s.side_length)^2 + (p.2 - s.side_length)^2
}

/-- The area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: The area of region R in a square with side length 3 is 9/4 -/
theorem area_of_region_R (s : Square) : area (region_R s) = 9/4 := by sorry

end NUMINAMATH_CALUDE_area_of_region_R_l3193_319366


namespace NUMINAMATH_CALUDE_find_divisor_l3193_319301

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 507 → quotient = 61 → remainder = 19 →
  ∃ (divisor : ℕ), dividend = divisor * quotient + remainder ∧ divisor = 8 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l3193_319301


namespace NUMINAMATH_CALUDE_expression_simplification_l3193_319327

theorem expression_simplification (x : ℝ) : 
  ((7*x + 3) - 3*x*2)*5 + (5 - 2/2)*(8*x - 5) = 37*x - 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3193_319327


namespace NUMINAMATH_CALUDE_ball_return_theorem_l3193_319390

/-- The number of ways a ball returns to the initial person after n passes among m people. -/
def ball_return_ways (m n : ℕ) : ℚ :=
  ((m - 1)^n : ℚ) / m + ((-1)^n : ℚ) * ((m - 1) : ℚ) / m

/-- Theorem: The number of ways a ball returns to the initial person after n passes among m people,
    where m ≥ 2, is given by ((m-1)^n / m) + ((-1)^n * (m-1) / m) -/
theorem ball_return_theorem (m n : ℕ) (h : m ≥ 2) :
  ∃ (a_n : ℕ → ℚ),
    (∀ k, a_n k = ball_return_ways m k) ∧
    (∀ k, a_n k ≥ 0) ∧
    (a_n 0 = 0) ∧
    (a_n 1 = 1) :=
  sorry


end NUMINAMATH_CALUDE_ball_return_theorem_l3193_319390


namespace NUMINAMATH_CALUDE_willam_land_percentage_l3193_319314

/-- Given that farm tax is levied on 40% of cultivated land, prove that Mr. Willam's
    taxable land is 12.5% of the village's total taxable land. -/
theorem willam_land_percentage (total_tax : ℝ) (willam_tax : ℝ)
    (h1 : total_tax = 3840)
    (h2 : willam_tax = 480) :
    willam_tax / total_tax * 100 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_willam_land_percentage_l3193_319314


namespace NUMINAMATH_CALUDE_lines_exist_iff_angle_geq_60_l3193_319371

-- Define the necessary structures
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

structure Plane where
  normal : Point3D
  d : ℝ

-- Define the given point and planes
variable (P : Point3D) -- Given point
variable (givenPlane : Plane) -- Given plane
variable (firstProjectionPlane : Plane) -- First projection plane

-- Define the angle between two planes
def angleBetweenPlanes (p1 p2 : Plane) : ℝ := sorry

-- Define a line passing through a point
structure Line where
  point : Point3D
  direction : Point3D

-- Define the angle between a line and a plane
def angleLinePlane (l : Line) (p : Plane) : ℝ := sorry

-- Define the distance between a point and a plane
def distancePointPlane (point : Point3D) (plane : Plane) : ℝ := sorry

-- Theorem statement
theorem lines_exist_iff_angle_geq_60 :
  (∃ (l1 l2 : Line),
    l1.point = P ∧
    l2.point = P ∧
    angleLinePlane l1 firstProjectionPlane = 60 ∧
    angleLinePlane l2 firstProjectionPlane = 60 ∧
    distancePointPlane l1.point givenPlane = distancePointPlane l2.point givenPlane) ↔
  angleBetweenPlanes givenPlane firstProjectionPlane ≥ 60 :=
sorry

end NUMINAMATH_CALUDE_lines_exist_iff_angle_geq_60_l3193_319371


namespace NUMINAMATH_CALUDE_paths_in_7x8_grid_l3193_319353

/-- The number of paths in a grid moving only up or right -/
def grid_paths (m n : ℕ) : ℕ := Nat.choose (m + n) m

/-- Theorem: The number of paths in a 7x8 grid moving only up or right is 6435 -/
theorem paths_in_7x8_grid : grid_paths 7 8 = 6435 := by
  sorry

end NUMINAMATH_CALUDE_paths_in_7x8_grid_l3193_319353


namespace NUMINAMATH_CALUDE_odometer_seven_count_l3193_319302

/-- A function that counts the number of sevens in a natural number -/
def count_sevens (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a number is six-digit -/
def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

theorem odometer_seven_count (n : ℕ) (h1 : is_six_digit n) (h2 : count_sevens n = 4) :
  count_sevens (n + 900) ≠ 1 := by sorry

end NUMINAMATH_CALUDE_odometer_seven_count_l3193_319302


namespace NUMINAMATH_CALUDE_ratio_seconds_minutes_l3193_319325

theorem ratio_seconds_minutes : ∃ x : ℝ, (12 / x = 6 / (4 * 60)) ∧ x = 480 := by
  sorry

end NUMINAMATH_CALUDE_ratio_seconds_minutes_l3193_319325


namespace NUMINAMATH_CALUDE_power_division_rule_l3193_319388

theorem power_division_rule (x : ℝ) (h : x ≠ 0) : x^10 / x^5 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l3193_319388


namespace NUMINAMATH_CALUDE_choose_three_from_thirteen_l3193_319360

theorem choose_three_from_thirteen : Nat.choose 13 3 = 286 := by sorry

end NUMINAMATH_CALUDE_choose_three_from_thirteen_l3193_319360


namespace NUMINAMATH_CALUDE_twenty_four_game_4888_l3193_319343

/-- The "24 points" game with cards 4, 8, 8, 8 -/
theorem twenty_four_game_4888 :
  let a : ℕ := 4
  let b : ℕ := 8
  let c : ℕ := 8
  let d : ℕ := 8
  (a - (c / d)) * b = 24 :=
by sorry

end NUMINAMATH_CALUDE_twenty_four_game_4888_l3193_319343


namespace NUMINAMATH_CALUDE_potato_price_proof_l3193_319372

/-- The original price of a bag of potatoes in rubles -/
def original_price : ℝ := 250

/-- The number of bags each trader bought -/
def bags_bought : ℕ := 60

/-- Andrey's price increase factor -/
def andrey_increase : ℝ := 2

/-- Boris's first price increase factor -/
def boris_first_increase : ℝ := 1.6

/-- Boris's second price increase factor -/
def boris_second_increase : ℝ := 1.4

/-- Number of bags Boris sold at first price -/
def boris_first_sale : ℕ := 15

/-- Number of bags Boris sold at second price -/
def boris_second_sale : ℕ := 45

/-- The difference in earnings between Boris and Andrey -/
def earnings_difference : ℝ := 1200

theorem potato_price_proof :
  let andrey_earning := bags_bought * original_price * andrey_increase
  let boris_first_earning := boris_first_sale * original_price * boris_first_increase
  let boris_second_earning := boris_second_sale * original_price * boris_first_increase * boris_second_increase
  boris_first_earning + boris_second_earning - andrey_earning = earnings_difference :=
by sorry

end NUMINAMATH_CALUDE_potato_price_proof_l3193_319372


namespace NUMINAMATH_CALUDE_equation_not_equivalent_l3193_319362

theorem equation_not_equivalent (x y : ℝ) 
  (hx1 : x ≠ 0) (hx2 : x ≠ 3) (hy1 : y ≠ 0) (hy2 : y ≠ 5) :
  (3 / x + 2 / y = 1 / 3) ↔ 
  ¬((3*x + 2*y = x*y) ∨ 
    (y = 3*x/(5 - y)) ∨ 
    (x/3 + y/2 = 3) ∨ 
    (3*y/(y - 5) = x)) := by
  sorry

end NUMINAMATH_CALUDE_equation_not_equivalent_l3193_319362


namespace NUMINAMATH_CALUDE_unique_sequence_existence_l3193_319346

theorem unique_sequence_existence :
  ∃! (a : ℕ → ℕ), 
    a 1 = 1 ∧
    a 2 > 1 ∧
    ∀ n : ℕ, n ≥ 1 → 
      (a (n + 1) * (a (n + 1) - 1) : ℚ) = 
        (a n * a (n + 2) : ℚ) / ((a n * a (n + 2) - 1 : ℚ) ^ (1/3) + 1) - 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_sequence_existence_l3193_319346


namespace NUMINAMATH_CALUDE_largest_number_l3193_319323

theorem largest_number (a b c d : ℝ) : 
  a = 1 → b = 0 → c = |-2| → d = -3 → 
  max a (max b (max c d)) = |-2| := by
sorry

end NUMINAMATH_CALUDE_largest_number_l3193_319323


namespace NUMINAMATH_CALUDE_intersection_complement_l3193_319334

def A : Set ℝ := {x | 1 < x ∧ x < 3}
def B : Set ℝ := {x | 2 < x}

theorem intersection_complement : A ∩ (Bᶜ) = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_l3193_319334


namespace NUMINAMATH_CALUDE_tangent_triangle_area_l3193_319309

/-- The area of the triangle formed by the tangent line to y = log₂ x at (1, 0) and the axes -/
theorem tangent_triangle_area : 
  let f (x : ℝ) := Real.log x / Real.log 2
  let tangent_line (x : ℝ) := (1 / Real.log 2) * (x - 1)
  let x_intercept : ℝ := 1
  let y_intercept : ℝ := -1 / Real.log 2
  let triangle_area : ℝ := (1/2) * x_intercept * (-y_intercept)
  triangle_area = 1 / (2 * Real.log 2) := by sorry

end NUMINAMATH_CALUDE_tangent_triangle_area_l3193_319309


namespace NUMINAMATH_CALUDE_expression_simplification_l3193_319387

theorem expression_simplification (x : ℝ) (h : x = 1) : 
  (4 + (4 + x^2) / x) / ((x + 2) / x) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3193_319387


namespace NUMINAMATH_CALUDE_book_page_digits_l3193_319304

/-- Calculate the total number of digits used to number pages in a book -/
def totalDigits (n : ℕ) : ℕ :=
  let singleDigit := min n 9
  let doubleDigit := max 0 (min n 99 - 9)
  let tripleDigit := max 0 (n - 99)
  singleDigit + 2 * doubleDigit + 3 * tripleDigit

/-- The total number of digits used in numbering the pages of a book with 360 pages is 972 -/
theorem book_page_digits :
  totalDigits 360 = 972 := by
  sorry

end NUMINAMATH_CALUDE_book_page_digits_l3193_319304


namespace NUMINAMATH_CALUDE_no_valid_coloring_l3193_319368

theorem no_valid_coloring :
  ¬ ∃ (f : ℕ+ → Fin 3),
    (∀ c : Fin 3, ∃ n : ℕ+, f n = c) ∧
    (∀ a b : ℕ+, f a ≠ f b → f (a * b) ≠ f a ∧ f (a * b) ≠ f b) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_coloring_l3193_319368


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_is_zero_l3193_319392

theorem sum_of_four_numbers_is_zero 
  (x y s t : ℝ) 
  (h_distinct : x ≠ y ∧ x ≠ s ∧ x ≠ t ∧ y ≠ s ∧ y ≠ t ∧ s ≠ t) 
  (h_equality : (x + s) / (x + t) = (y + t) / (y + s)) : 
  x + y + s + t = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_is_zero_l3193_319392


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3193_319394

/-- A positive geometric sequence -/
def IsPositiveGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  IsPositiveGeometricSequence a →
  a 1 * a 19 = 16 →
  a 8 * a 10 * a 12 = 64 := by
    sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3193_319394


namespace NUMINAMATH_CALUDE_stone_pile_division_l3193_319352

/-- Two natural numbers are similar if they differ by no more than twice -/
def similar (a b : ℕ) : Prop := a ≤ b ∧ b ≤ 2 * a

/-- A sequence of operations to combine piles -/
inductive CombineSeq : ℕ → ℕ → Type
  | single : (n : ℕ) → CombineSeq n 1
  | combine : {n m k : ℕ} → (s : CombineSeq n m) → (t : CombineSeq n k) → 
              similar m k → CombineSeq n (m + k)

/-- Any pile of stones can be divided into piles of single stones -/
theorem stone_pile_division (n : ℕ) : CombineSeq n n := by sorry

end NUMINAMATH_CALUDE_stone_pile_division_l3193_319352


namespace NUMINAMATH_CALUDE_total_filled_boxes_is_16_l3193_319333

/-- Represents the types of trading cards --/
inductive CardType
  | Magic
  | Rare
  | Common

/-- Represents the types of boxes --/
inductive BoxType
  | Small
  | Large

/-- Defines the capacity of each box type for each card type --/
def boxCapacity (b : BoxType) (c : CardType) : ℕ :=
  match b, c with
  | BoxType.Small, CardType.Magic => 5
  | BoxType.Small, CardType.Rare => 5
  | BoxType.Small, CardType.Common => 6
  | BoxType.Large, CardType.Magic => 10
  | BoxType.Large, CardType.Rare => 10
  | BoxType.Large, CardType.Common => 15

/-- Calculates the number of fully filled boxes of a given type for a specific card type --/
def filledBoxes (cardCount : ℕ) (b : BoxType) (c : CardType) : ℕ :=
  cardCount / boxCapacity b c

/-- The main theorem stating that the total number of fully filled boxes is 16 --/
theorem total_filled_boxes_is_16 :
  let magicCards := 33
  let rareCards := 28
  let commonCards := 33
  let smallBoxesMagic := filledBoxes magicCards BoxType.Small CardType.Magic
  let smallBoxesRare := filledBoxes rareCards BoxType.Small CardType.Rare
  let smallBoxesCommon := filledBoxes commonCards BoxType.Small CardType.Common
  smallBoxesMagic + smallBoxesRare + smallBoxesCommon = 16 :=
by
  sorry


end NUMINAMATH_CALUDE_total_filled_boxes_is_16_l3193_319333


namespace NUMINAMATH_CALUDE_rower_upstream_speed_l3193_319313

/-- Calculates the upstream speed of a rower given their still water speed and downstream speed -/
def upstream_speed (still_water_speed downstream_speed : ℝ) : ℝ :=
  2 * still_water_speed - downstream_speed

/-- Proves that given a man's speed in still water is 33 kmph and his downstream speed is 41 kmph, 
    his upstream speed is 25 kmph -/
theorem rower_upstream_speed :
  let still_water_speed := (33 : ℝ)
  let downstream_speed := (41 : ℝ)
  upstream_speed still_water_speed downstream_speed = 25 := by
sorry

#eval upstream_speed 33 41

end NUMINAMATH_CALUDE_rower_upstream_speed_l3193_319313


namespace NUMINAMATH_CALUDE_product_remainder_main_theorem_l3193_319316

theorem product_remainder (a b : Nat) : (a * b) % 9 = ((a % 9) * (b % 9)) % 9 := by sorry

theorem main_theorem : (98 * 102) % 9 = 3 := by
  -- The proof would go here, but we're only providing the statement
  sorry

end NUMINAMATH_CALUDE_product_remainder_main_theorem_l3193_319316


namespace NUMINAMATH_CALUDE_sum_base5_digits_2010_l3193_319331

/-- Converts a natural number to its base-5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 5) :: aux (m / 5)
  aux n |>.reverse

/-- Sums the digits in a list of natural numbers -/
def sumDigits (l : List ℕ) : ℕ :=
  l.foldl (·+·) 0

/-- The sum of digits in the base-5 representation of 2010 equals 6 -/
theorem sum_base5_digits_2010 : sumDigits (toBase5 2010) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_base5_digits_2010_l3193_319331


namespace NUMINAMATH_CALUDE_parallel_iff_a_eq_two_l3193_319369

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel (m1 n1 c1 m2 n2 c2 : ℝ) : Prop :=
  m1 * n2 = m2 * n1 ∧ m1 ≠ 0 ∧ m2 ≠ 0

/-- The theorem states that a = 2 is a necessary and sufficient condition
    for the lines 2x - ay + 1 = 0 and (a-1)x - y + a = 0 to be parallel -/
theorem parallel_iff_a_eq_two (a : ℝ) :
  parallel 2 (-a) 1 (a-1) (-1) a ↔ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_iff_a_eq_two_l3193_319369


namespace NUMINAMATH_CALUDE_max_value_theorem_l3193_319375

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

noncomputable def g (x : ℝ) : ℝ := 2 * x * Real.log (2 * x)

theorem max_value_theorem (x₁ x₂ t : ℝ) (h₁ : f x₁ = t) (h₂ : g x₂ = t) (h₃ : t > 0) :
  ∃ (m : ℝ), m = (2 : ℝ) / Real.exp 1 ∧ 
  ∀ (y : ℝ), y = (Real.log t) / (x₁ * x₂) → y ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3193_319375


namespace NUMINAMATH_CALUDE_trees_along_path_l3193_319379

/-- Calculates the total number of trees that can be planted along a path -/
def totalTrees (pathLength : ℕ) (treeSpacing : ℕ) : ℕ :=
  2 * (pathLength / treeSpacing + 1)

/-- Theorem: Given a path of 80 meters with trees planted on both sides every 4 meters,
    including at both ends, the total number of trees that can be planted is 42. -/
theorem trees_along_path :
  totalTrees 80 4 = 42 := by
  sorry

end NUMINAMATH_CALUDE_trees_along_path_l3193_319379


namespace NUMINAMATH_CALUDE_bird_migration_problem_l3193_319398

theorem bird_migration_problem (distance_jim_disney : ℕ) (distance_disney_london : ℕ) (total_distance : ℕ) :
  distance_jim_disney = 50 →
  distance_disney_london = 60 →
  total_distance = 2200 →
  ∃ (num_birds : ℕ), num_birds * (distance_jim_disney + distance_disney_london) = total_distance ∧ num_birds = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_bird_migration_problem_l3193_319398


namespace NUMINAMATH_CALUDE_smallest_divisible_by_12_and_60_l3193_319355

theorem smallest_divisible_by_12_and_60 : Nat.lcm 12 60 = 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_12_and_60_l3193_319355
