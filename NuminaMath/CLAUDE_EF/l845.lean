import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l845_84582

-- Define the ellipse D
def ellipse_D (x y : ℝ) : Prop := x^2 / 50 + y^2 / 25 = 1

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + (y - 5)^2 = 9

-- Define the hyperbola G
def hyperbola_G (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

-- Define the foci of ellipse D
def foci_D : (ℝ × ℝ) × (ℝ × ℝ) := ((-5, 0), (5, 0))

-- Define the condition that G shares foci with D
def shares_foci (G : ℝ → ℝ → Prop) (D : ℝ → ℝ → Prop) : Prop :=
  ∃ (f₁ f₂ : ℝ × ℝ) (k : ℝ), (∀ (x y : ℝ), G x y ↔ 
    (x - f₁.1)^2 + (y - f₁.2)^2 - ((x - f₂.1)^2 + (y - f₂.2)^2) = k) ∧
    (f₁, f₂) = foci_D

-- Define the condition that asymptotes of G are tangent to M
def asymptotes_tangent (G : ℝ → ℝ → Prop) (M : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b : ℝ), (∀ (x y : ℝ), G x y ↔ x^2 / a^2 - y^2 / b^2 = 1) ∧
    (∀ (x y : ℝ), (b * x = a * y ∨ b * x = -a * y) → 
      (M x y → (x - 0)^2 + (y - 5)^2 = 3^2))

-- Theorem statement
theorem hyperbola_equation :
  (∀ (x y : ℝ), ellipse_D x y ↔ x^2 / 50 + y^2 / 25 = 1) →
  (∀ (x y : ℝ), circle_M x y ↔ x^2 + (y - 5)^2 = 9) →
  shares_foci hyperbola_G ellipse_D →
  asymptotes_tangent hyperbola_G circle_M →
  ∀ (x y : ℝ), hyperbola_G x y ↔ x^2 / 9 - y^2 / 16 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l845_84582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_odd_five_prime_factors_l845_84570

theorem smallest_odd_five_prime_factors : 
  ∀ n : ℕ, Odd n ∧ (∃ p₁ p₂ p₃ p₄ p₅ : ℕ, Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ Prime p₅ ∧
    p₁ < p₂ ∧ p₂ < p₃ ∧ p₃ < p₄ ∧ p₄ < p₅ ∧
    n = p₁ * p₂ * p₃ * p₄ * p₅) → n ≥ 15015 := by
  sorry

#check smallest_odd_five_prime_factors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_odd_five_prime_factors_l845_84570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cp_length_when_lambda_third_lambda_value_when_ap_ratio_pb_lambda_range_when_dot_product_inequality_l845_84593

-- Define the equilateral triangle ABC and point P
structure TriangleConfig where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  P : ℝ × ℝ
  side_length : ℝ
  lambda : ℝ

-- Define the conditions
def is_valid_config (config : TriangleConfig) : Prop :=
  let A := config.A
  let B := config.B
  let C := config.C
  let P := config.P
  let side_length := config.side_length
  let lambda := config.lambda
  -- ABC is equilateral
  (‖B - A‖ = side_length) ∧
  (‖C - B‖ = side_length) ∧
  (‖A - C‖ = side_length) ∧
  -- P lies on AB
  (∃ t : ℝ, P = A + t • (B - A) ∧ 0 ≤ t ∧ t ≤ 1) ∧
  -- AP = lambda * AB
  (P - A = lambda • (B - A)) ∧
  -- 0 ≤ lambda ≤ 1
  (0 ≤ lambda ∧ lambda ≤ 1)

-- Theorem 1
theorem cp_length_when_lambda_third (config : TriangleConfig) 
  (h : is_valid_config config) (h_side : config.side_length = 6) (h_lambda : config.lambda = 1/3) :
  ‖config.C - config.P‖ = 2 * Real.sqrt 7 :=
sorry

-- Theorem 2
theorem lambda_value_when_ap_ratio_pb (config : TriangleConfig) 
  (h : is_valid_config config) (h_ratio : config.P - config.A = (3/5) • (config.B - config.P)) :
  config.lambda = 3/8 :=
sorry

-- Theorem 3
theorem lambda_range_when_dot_product_inequality (config : TriangleConfig) 
  (h : is_valid_config config) 
  (h_dot : (config.C - config.P) • (config.B - config.A) ≥ (config.P - config.A) • (config.B - config.P)) :
  (2 - Real.sqrt 2) / 2 ≤ config.lambda ∧ config.lambda ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cp_length_when_lambda_third_lambda_value_when_ap_ratio_pb_lambda_range_when_dot_product_inequality_l845_84593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_delta_calculation_l845_84506

-- Define the Δ operation
def delta (x y : ℝ) : ℝ := x^2 - 2*y

-- State the theorem
theorem delta_calculation : 
  delta (5^(delta 3 4)) (2^(delta 2 3)) = 24.5 := by
  -- Convert natural numbers to reals
  have h1 : (3 : ℝ) = 3 := by norm_num
  have h2 : (4 : ℝ) = 4 := by norm_num
  have h3 : (2 : ℝ) = 2 := by norm_num
  have h4 : (5 : ℝ) = 5 := by norm_num
  
  -- Calculate intermediate steps
  have step1 : delta 3 4 = 1 := by
    rw [delta, h1, h2]
    norm_num
  
  have step2 : delta 2 3 = -2 := by
    rw [delta, h3, h1]
    norm_num
  
  -- Prove the final result
  rw [delta, step1, step2]
  norm_num
  

end NUMINAMATH_CALUDE_ERRORFEEDBACK_delta_calculation_l845_84506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_equation_l845_84501

-- Define the slope of the given line
def m : ℚ := 2 / 5

-- Define the y-intercept of the given line
def b : ℝ := 3

-- Define the distance between the lines
def d : ℝ := 5

-- Define the equation of a line with slope m and y-intercept c
def line_equation (x : ℝ) (c : ℝ) : ℝ := m * x + c

-- Theorem statement
theorem parallel_line_equation :
  ∀ (c : ℝ), 
  (∀ (x : ℝ), line_equation x c = line_equation x (b + Real.sqrt 29) ∨ 
               line_equation x c = line_equation x (b - Real.sqrt 29)) ↔ 
  (abs (c - b) / Real.sqrt (m^2 + 1) = d) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_equation_l845_84501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_le_millionth_l845_84500

def x : ℕ → ℚ
  | 0 => 1  -- Adding case for 0
  | 1 => 1
  | 2 => 1
  | 3 => 2/3
  | n + 4 => (x (n + 3))^2 * x (n + 2) / (2 * (x (n + 2))^2 - x (n + 3) * x (n + 1))

theorem least_n_le_millionth :
  (∀ k < 13, x k > 1/1000000) ∧ x 13 ≤ 1/1000000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_le_millionth_l845_84500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_time_theorem_l845_84530

/-- Represents a race between two runners A and B -/
structure Race where
  distance : ℚ
  lead_distance : ℚ
  lead_time : ℚ

/-- Calculates the time taken by runner A to complete the race -/
def race_time_A (race : Race) : ℚ :=
  race.distance / (race.lead_distance / race.lead_time)

/-- Theorem stating that for the given race conditions, A's time is 35 seconds -/
theorem race_time_theorem (race : Race) 
  (h1 : race.distance = 280)
  (h2 : race.lead_distance = 56)
  (h3 : race.lead_time = 7) : 
  race_time_A race = 35 := by
  sorry

#eval race_time_A { distance := 280, lead_distance := 56, lead_time := 7 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_time_theorem_l845_84530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l845_84573

def sequenceA (n : ℕ) : ℤ :=
  3 - 2^n

theorem sequence_formula :
  (sequenceA 1 = 1) ∧
  (∀ k : ℕ, sequenceA (k + 1) = 2 * sequenceA k - 3) →
  ∀ m : ℕ, m > 0 → sequenceA m = 3 - 2^m :=
by
  intro h
  intro m hm
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l845_84573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_leftover_oranges_l845_84590

theorem max_leftover_oranges :
  ∀ n : ℕ,
  ∃ k : ℕ,
  n = 8 * k + (n % 8) ∧
  n % 8 ≤ 7 ∧
  ∀ m : ℕ, m % 8 ≤ 7 →
  ∃ j : ℕ, j ≤ k ∧ m = 8 * j + (m % 8) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_leftover_oranges_l845_84590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_adjacent_ball_arrangements_l845_84524

/-- Represents the number of ways to arrange balls in a row -/
def arrangement_count (white red yellow : ℕ) : ℕ := sorry

/-- The number of ways to arrange 1 white ball, 1 red ball, and 3 identical yellow balls
    in a row such that the white and red balls are not adjacent -/
theorem non_adjacent_ball_arrangements :
  arrangement_count 1 1 3 = 12 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_adjacent_ball_arrangements_l845_84524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_subsequence_exists_l845_84545

theorem monotone_subsequence_exists (a : ℕ → ℝ) :
  ∃ (j : ℕ → ℕ), StrictMono j ∧
    (Monotone (fun n => a (j n)) ∨ Monotone (fun n => -a (j n))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_subsequence_exists_l845_84545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_line_and_circle_l845_84551

-- Define the curve
def curve (x y : ℝ) : Prop := x^3 + x*y^2 = 2*x

-- Define a line (x = 0)
def line (x : ℝ) : Prop := x = 0

-- Define a circle (x^2 + y^2 = 2)
def circle_curve (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Theorem statement
theorem curve_is_line_and_circle :
  ∀ x y : ℝ, curve x y ↔ (line x ∨ circle_curve x y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_line_and_circle_l845_84551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_l845_84579

theorem find_m (a : ℝ) (m : ℝ) (h1 : a^10 = 1/2) (h2 : a^m = Real.sqrt 2/2) : m = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_l845_84579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_negative_eight_l845_84589

theorem cube_root_of_negative_eight :
  ((-8 : ℝ) ^ (1/3 : ℝ)) = -2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_negative_eight_l845_84589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_triangle_ABC_l845_84538

-- Define the points and line
def A : ℝ × ℝ := (-3, 4)
def y_axis : Set (ℝ × ℝ) := {p | p.1 = 0}
def line_l : Set (ℝ × ℝ) := {p | p.1 + p.2 = 0}

-- Define the perimeter function
noncomputable def perimeter (B : ℝ × ℝ) (C : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) +
  Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) +
  Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)

-- Theorem statement
theorem min_perimeter_triangle_ABC :
  ∃ (B : ℝ × ℝ) (C : ℝ × ℝ),
    B ∈ y_axis ∧ C ∈ line_l ∧
    (∀ (B' : ℝ × ℝ) (C' : ℝ × ℝ),
      B' ∈ y_axis → C' ∈ line_l →
      perimeter B C ≤ perimeter B' C') ∧
    perimeter B C = 5 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_triangle_ABC_l845_84538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_addition_l845_84502

theorem divisibility_addition : 
  ∃ k : ℕ, (897326 + k) % 456 = 0 ∧ k = (456 - (897326 % 456)) % 456 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_addition_l845_84502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_number_reverse_sum_square_l845_84561

def reverse_number (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def satisfies_condition (n : ℕ) : Prop :=
  n ≥ 10 ∧ n ≤ 99 ∧ is_perfect_square (n + reverse_number n)

theorem two_digit_number_reverse_sum_square :
  ∀ n : ℕ, satisfies_condition n ↔ n ∈ ({29, 38, 47, 56, 65, 74, 83, 92} : Finset ℕ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_number_reverse_sum_square_l845_84561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clubsuit_sum_l845_84516

noncomputable def clubsuit (x : ℝ) : ℝ := (x + x^3) / 2

theorem clubsuit_sum : clubsuit 2 + clubsuit 3 + clubsuit 5 = 85 := by
  -- Expand the definition of clubsuit
  unfold clubsuit
  -- Simplify the expressions
  simp [pow_three]
  -- Perform the arithmetic
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clubsuit_sum_l845_84516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_counterpart_sets_l845_84585

-- Define the concept of a "perpendicular counterpart set"
def is_perpendicular_counterpart_set (M : Set (ℝ × ℝ)) : Prop :=
  ∀ (p₁ : ℝ × ℝ), p₁ ∈ M → ∃ (p₂ : ℝ × ℝ), p₂ ∈ M ∧ p₁.1 * p₂.1 + p₁.2 * p₂.2 = 0

-- Define the four sets
def M₁ : Set (ℝ × ℝ) := {p | p.2 = 1 / p.1}
def M₂ : Set (ℝ × ℝ) := {p | p.2 = Real.log p.1 / Real.log 2}
def M₃ : Set (ℝ × ℝ) := {p | p.2 = Real.exp p.1 - 2}
def M₄ : Set (ℝ × ℝ) := {p | p.2 = Real.sin p.1 + 1}

-- State the theorem
theorem perpendicular_counterpart_sets :
  ¬(is_perpendicular_counterpart_set M₁) ∧
  ¬(is_perpendicular_counterpart_set M₂) ∧
  (is_perpendicular_counterpart_set M₃) ∧
  (is_perpendicular_counterpart_set M₄) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_counterpart_sets_l845_84585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_false_dislike_approx_407_l845_84519

/-- Represents the fraction of students who enjoy music -/
noncomputable def enjoy_music : ℝ := 0.7

/-- Represents the fraction of students who do not enjoy music -/
noncomputable def dont_enjoy_music : ℝ := 1 - enjoy_music

/-- Represents the fraction of music-enjoying students who express enjoyment -/
noncomputable def express_enjoyment : ℝ := 0.75

/-- Represents the fraction of non-music-enjoying students who express disinterest -/
noncomputable def express_disinterest : ℝ := 0.85

/-- The fraction of students who claim to dislike music but actually enjoy it -/
noncomputable def fraction_false_dislike : ℝ :=
  (enjoy_music * (1 - express_enjoyment)) / 
  (enjoy_music * (1 - express_enjoyment) + dont_enjoy_music * express_disinterest)

theorem false_dislike_approx_407 :
  |fraction_false_dislike - 0.407| < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_false_dislike_approx_407_l845_84519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_239_times_l845_84587

noncomputable def f (x : ℝ) : ℝ := x / (1 + x / 239)

noncomputable def iterate_f : ℕ → ℝ → ℝ
  | 0, x => x
  | n + 1, x => f (iterate_f n x)

theorem f_239_times (x : ℝ) (h : x ≥ 0) : 
  iterate_f 239 x = x / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_239_times_l845_84587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_map_distance_representation_l845_84537

/-- Represents the scale of a map in kilometers per centimeter -/
noncomputable def map_scale (cm : ℝ) (km : ℝ) : ℝ := km / cm

/-- Theorem: Given a map where 15 cm represents 90 km, 20 cm represents 120 km -/
theorem map_distance_representation (cm₁ km₁ cm₂ : ℝ) 
  (h₁ : cm₁ = 15) 
  (h₂ : km₁ = 90) 
  (h₃ : cm₂ = 20) : 
  cm₂ * (map_scale cm₁ km₁) = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_map_distance_representation_l845_84537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_plus_pi_third_l845_84528

theorem cos_2alpha_plus_pi_third (α : ℝ) 
  (h1 : Real.cos (π/6 - α) + Real.sin (π - α) = -(4 * Real.sqrt 3)/5)
  (h2 : -π/2 < α)
  (h3 : α < 0) :
  Real.cos (2*α + π/3) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_plus_pi_third_l845_84528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_l845_84555

noncomputable def g (x : ℝ) : ℝ := |⌊x + 2⌋| - |⌊3 - x⌋|

theorem g_symmetry (x : ℝ) : g x = g (5 - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_l845_84555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_y_l845_84536

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 + (Real.log x) / (Real.log 3)

-- Define the domain of x
def domain : Set ℝ := Set.Icc 1 9

-- Define the expression y
noncomputable def y (x : ℝ) : ℝ := (f x)^2 + f (x^2)

-- Theorem statement
theorem range_of_y :
  Set.range (fun x => y x) ∩ (Set.image y domain) = Set.Icc 6 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_y_l845_84536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_january_first_is_monday_l845_84511

/-- Represents the days of the week -/
inductive Weekday
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
deriving Repr, DecidableEq

/-- Function to get the next day of the week -/
def nextDay (d : Weekday) : Weekday :=
  match d with
  | Weekday.Sunday => Weekday.Monday
  | Weekday.Monday => Weekday.Tuesday
  | Weekday.Tuesday => Weekday.Wednesday
  | Weekday.Wednesday => Weekday.Thursday
  | Weekday.Thursday => Weekday.Friday
  | Weekday.Friday => Weekday.Saturday
  | Weekday.Saturday => Weekday.Sunday

/-- Function to count occurrences of a specific day in a month -/
def countDaysInMonth (startDay : Weekday) (numDays : Nat) (targetDay : Weekday) : Nat :=
  let rec count (currentDay : Weekday) (daysLeft : Nat) (acc : Nat) : Nat :=
    if daysLeft = 0 then
      acc
    else if currentDay = targetDay then
      count (nextDay currentDay) (daysLeft - 1) (acc + 1)
    else
      count (nextDay currentDay) (daysLeft - 1) acc
  count startDay numDays 0

/-- Theorem: In a 31-day month with exactly five Mondays and five Fridays, the first day must be Monday -/
theorem january_first_is_monday :
  ∀ (startDay : Weekday),
    (countDaysInMonth startDay 31 Weekday.Monday = 5 ∧
     countDaysInMonth startDay 31 Weekday.Friday = 5) →
    startDay = Weekday.Monday :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_january_first_is_monday_l845_84511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_h_value_l845_84552

-- Define the polynomial type
def MyPolynomial := ℝ → ℝ

-- Define the degree of a polynomial
noncomputable def degree (P : MyPolynomial) : ℕ := sorry

-- Define the property that a polynomial vanishes at 0 and 1
def vanishes_at_endpoints (P : MyPolynomial) : Prop :=
  P 0 = 0 ∧ P 1 = 0

-- Define the property for the existence of x₁ and x₂
def exists_equal_values (P : MyPolynomial) (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 1 ∧ x₂ ∈ Set.Icc 0 1 ∧ 
  P x₁ = P x₂ ∧ x₂ - x₁ = a

-- Define the main theorem
theorem max_h_value : 
  (∃ h : ℝ, h > 0 ∧ 
    (∀ a : ℝ, a ∈ Set.Icc 0 h → 
      ∀ P : MyPolynomial, degree P = 99 → 
        vanishes_at_endpoints P → 
          exists_equal_values P a) ∧ 
    (∀ h' : ℝ, h' > h → 
      ∃ a : ℝ, a ∈ Set.Icc 0 h' ∧ 
        ∃ P : MyPolynomial, degree P = 99 ∧ 
          vanishes_at_endpoints P ∧ 
            ¬exists_equal_values P a)) ∧ 
  (∀ h : ℝ, (∃ h : ℝ, h > 0 ∧ 
    (∀ a : ℝ, a ∈ Set.Icc 0 h → 
      ∀ P : MyPolynomial, degree P = 99 → 
        vanishes_at_endpoints P → 
          exists_equal_values P a) ∧ 
    (∀ h' : ℝ, h' > h → 
      ∃ a : ℝ, a ∈ Set.Icc 0 h' ∧ 
        ∃ P : MyPolynomial, degree P = 99 ∧ 
          vanishes_at_endpoints P ∧ 
            ¬exists_equal_values P a)) → 
  h = 1 / 50) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_h_value_l845_84552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l845_84505

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- Given that i² = -1 -/
axiom i_squared : i^2 = -1

/-- The complex fraction to be simplified -/
noncomputable def complex_fraction : ℂ := (3 - 2*i) / (4 - 5*i)

/-- The simplified result -/
noncomputable def simplified_result : ℂ := 2/41 + (7/41)*i

/-- Theorem stating that the complex fraction equals the simplified result -/
theorem complex_fraction_simplification :
  complex_fraction = simplified_result := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l845_84505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l845_84544

-- Define the equation
def equation (x : ℝ) : Prop := x / 50 = Real.sin (2 * x)

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | equation x ∧ -50 ≤ x ∧ x ≤ 50}

-- Theorem statement
theorem equation_solutions :
  ∃ (s : Finset ℝ), s.card = 63 ∧ ∀ x ∈ s, x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l845_84544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_divisibility_l845_84534

theorem sequence_divisibility (p : ℕ) (hp : Nat.Prime p) (hp_mod : p % 4 = 3) :
  ∀ (a : ℕ → ℤ), a 0 = 1 →
  (∀ k : ℕ, k ∈ Finset.range p → (p : ℤ) ∣ a (k - 1) * a k - k) →
  (p : ℤ) ∣ a (p - 1) + 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_divisibility_l845_84534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_arrangement_l845_84510

/-- A function representing an arrangement of natural numbers in an infinite grid -/
def GridArrangement := ℕ → ℕ → ℕ

/-- The sum of numbers in an m × n rectangle starting at position (i, j) -/
def RectangleSum (f : GridArrangement) (m n i j : ℕ) : ℕ :=
  Finset.sum (Finset.range m) (λ x => Finset.sum (Finset.range n) (λ y => f (i + x) (j + y)))

/-- The property that for all m, n > 100, the sum in any m × n rectangle is divisible by m + n -/
def ValidArrangement (f : GridArrangement) : Prop :=
  ∀ m n i j, m > 100 → n > 100 → (m + n) ∣ RectangleSum f m n i j

/-- Theorem stating that no valid arrangement exists -/
theorem no_valid_arrangement : ¬ ∃ f : GridArrangement, ValidArrangement f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_arrangement_l845_84510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sum_is_255_l845_84595

/-- Given a list of 6 different digits, returns the largest possible sum of three 2-digit numbers formed from these digits -/
def largestSumOfThreeTwoDigitNumbers (digits : List Nat) : Nat :=
  if digits.length ≠ 6 then 0
  else if ¬(digits.toFinset.card == 6) then 0
  else if ¬(digits.all (λ d => d < 10)) then 0
  else
    let sortedDigits := digits.toArray.qsort (·>·)
    10 * (sortedDigits[0]! + sortedDigits[1]! + sortedDigits[2]!) +
        (sortedDigits[3]! + sortedDigits[4]! + sortedDigits[5]!)

/-- Theorem stating that the largest possible sum of three 2-digit numbers
    formed from 6 different digits is 255 -/
theorem largest_sum_is_255 :
  ∀ digits : List Nat,
    digits.length = 6 →
    digits.toFinset.card = 6 →
    (∀ d ∈ digits, d < 10) →
    largestSumOfThreeTwoDigitNumbers digits = 255 := by
  sorry

#eval largestSumOfThreeTwoDigitNumbers [9, 8, 7, 6, 5, 4]  -- Should output 255

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sum_is_255_l845_84595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_2_sqrt_3_l845_84513

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
  t.b^2 + t.c^2 = t.a^2 + t.b * t.c ∧
  t.b * t.c * Real.cos t.A = 4

-- Define the area function
noncomputable def area (t : Triangle) : ℝ :=
  1/2 * t.b * t.c * Real.sin t.A

-- Theorem statement
theorem triangle_area_is_2_sqrt_3 (t : Triangle) 
  (h : satisfies_conditions t) : area t = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_2_sqrt_3_l845_84513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_properties_l845_84594

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 3 = 0
def circle_O2 (x y : ℝ) : Prop := x^2 + y^2 - 2*y - 1 = 0

-- Define the intersection points A and B
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Define the line AB
def line_AB (x y : ℝ) : Prop := x - y + 1 = 0

-- Theorem statement
theorem circles_properties :
  -- The circles have two common tangents
  (∃ (t1 t2 : ℝ → ℝ), 
    (∀ x, circle_O1 x (t1 x) ∧ circle_O2 x (t2 x)) ∧
    t1 ≠ t2) ∧
  -- The equation of line AB is x - y + 1 = 0
  (∀ x y, line_AB x y ↔ x - y + 1 = 0) ∧
  -- The maximum distance from a point on circle O₁ to line AB is 2 + √2
  (∃ (max_dist : ℝ), max_dist = 2 + Real.sqrt 2 ∧
    ∀ (p : ℝ × ℝ), circle_O1 p.1 p.2 →
      Real.sqrt ((p.1 - A.1)^2 + (p.2 - A.2)^2) ≤ max_dist) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_properties_l845_84594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shelby_yesterday_stars_l845_84547

/-- Represents the number of gold stars Shelby earned yesterday -/
def yesterday_stars : ℕ := sorry

/-- Represents the number of gold stars Shelby earned today -/
def today_stars : ℕ := 3

/-- Represents the total number of gold stars Shelby has now -/
def total_stars : ℕ := 7

/-- Theorem stating that Shelby earned 4 gold stars yesterday -/
theorem shelby_yesterday_stars : 
  yesterday_stars + today_stars = total_stars → yesterday_stars = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shelby_yesterday_stars_l845_84547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_properties_main_theorem_l845_84560

/-- A pyramid with a parallelogram base -/
structure Pyramid where
  -- Base area
  m : ℝ
  -- Assertion that m > 0
  m_pos : m > 0

/-- The lateral surface area of the pyramid -/
noncomputable def lateralSurfaceArea (p : Pyramid) : ℝ :=
  p.m^2 * (1 + Real.sqrt 2 / 2)

/-- The volume of the pyramid -/
noncomputable def volume (p : Pyramid) : ℝ :=
  p.m^3 * Real.sqrt (Real.sqrt 2) / 6

/-- Theorem stating the properties of the pyramid -/
theorem pyramid_properties (p : Pyramid) :
  (lateralSurfaceArea p = p.m^2 * (1 + Real.sqrt 2 / 2)) ∧
  (volume p = p.m^3 * Real.sqrt (Real.sqrt 2) / 6) := by
  sorry

/-- Main theorem combining all properties -/
theorem main_theorem (p : Pyramid) :
  (lateralSurfaceArea p = p.m^2 * (1 + Real.sqrt 2 / 2)) ∧
  (volume p = p.m^3 * Real.sqrt (Real.sqrt 2) / 6) := by
  exact pyramid_properties p

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_properties_main_theorem_l845_84560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_B_coordinates_l845_84563

-- Define the Point type
structure Point where
  x : ℝ
  y : ℝ

-- Define the problem statement
theorem point_B_coordinates 
  (A : Point) 
  (h1 : A.x = -1 ∧ A.y = 2) 
  (h2 : ∀ (B : Point), B.y = A.y) -- AB is parallel to x-axis
  (h3 : ∀ (B : Point), (B.x - A.x)^2 + (B.y - A.y)^2 = 4^2) -- AB = 4
  : ∃ (B : Point), (B = ⟨3, 2⟩ ∨ B = ⟨-5, 2⟩) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_B_coordinates_l845_84563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_A_to_B_l845_84509

-- Define the constants
noncomputable def track_A_diameter : ℝ := 1000  -- in meters
noncomputable def track_B_length : ℝ := 5000    -- in meters
noncomputable def laps_A : ℝ := 3
noncomputable def trips_B : ℝ := 2
noncomputable def time_A : ℝ := 10              -- in minutes
noncomputable def time_B : ℝ := 5               -- in minutes

-- Define the speeds
noncomputable def speed_A : ℝ := Real.pi * track_A_diameter * laps_A / time_A
noncomputable def speed_B : ℝ := track_B_length * trips_B * 2 / time_B

-- Theorem statement
theorem speed_ratio_A_to_B :
  speed_A / speed_B = 3 * Real.pi / 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_A_to_B_l845_84509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_quadrilateral_problem_l845_84535

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the perimeter of a triangle
noncomputable def triangle_perimeter (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) +
  Real.sqrt ((x₂ - x₃)^2 + (y₂ - y₃)^2) +
  Real.sqrt ((x₃ - x₁)^2 + (y₃ - y₁)^2)

-- Define the dot product of two vectors
def dot_product (x₁ y₁ x₂ y₂ : ℝ) : ℝ := x₁ * x₂ + y₁ * y₂

-- Define the slope of a line
noncomputable def line_slope (x₁ y₁ x₂ y₂ : ℝ) : ℝ := (y₂ - y₁) / (x₂ - x₁)

-- Define the area of a quadrilateral given by the product of its diagonals
noncomputable def quadrilateral_area (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) : ℝ :=
  (1/2) * Real.sqrt ((x₃ - x₁)^2 + (y₃ - y₁)^2) * Real.sqrt ((x₄ - x₂)^2 + (y₄ - y₂)^2)

theorem ellipse_and_quadrilateral_problem
  (a b : ℝ)
  (h₁ : a > b)
  (h₂ : b > 0)
  (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ xp yp xf₁ yf₁ xf₂ yf₂ : ℝ)
  (h₃ : ellipse a b x₁ y₁)
  (h₄ : ellipse a b x₂ y₂)
  (h₅ : ellipse a b x₃ y₃)
  (h₆ : ellipse a b x₄ y₄)
  (h₇ : ellipse a b xp yp)
  (h₈ : triangle_perimeter xp yp xf₁ yf₁ xf₂ yf₂ = 6)
  (h₉ : ∃ (xp' yp' : ℝ), ellipse a b xp' yp' ∧ (xp' - xf₁) * (xf₂ - xf₁) + (yp' - yf₁) * (yf₂ - yf₁) = 0)
  (h₁₀ : dot_product (x₃ - x₁) (y₃ - y₁) (x₄ - x₂) (y₄ - y₂) = 0)
  (h₁₁ : line_slope x₁ y₁ x₃ y₃ = Real.sqrt 3)
  : a = 2 ∧ b = Real.sqrt 3 ∧ quadrilateral_area x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ = 384/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_quadrilateral_problem_l845_84535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_student_average_difference_l845_84599

/-- Represents a school with students and teachers. -/
structure School where
  num_students : ℕ
  num_teachers : ℕ
  class_sizes : List ℕ

/-- Calculates the average number of students in a class as seen by a randomly chosen teacher. -/
def teacher_average (school : School) : ℚ :=
  (school.class_sizes.sum : ℚ) / school.num_teachers

/-- Calculates the average number of students in a class as seen by a randomly chosen student. -/
def student_average (school : School) : ℚ :=
  (school.class_sizes.map (λ size => size * size)).sum / school.num_students

/-- The main theorem stating the difference between teacher and student averages. -/
theorem teacher_student_average_difference (school : School) 
    (h1 : school.num_students = 120)
    (h2 : school.num_teachers = 4)
    (h3 : school.class_sizes = [60, 30, 20, 10])
    (h4 : school.class_sizes.sum = school.num_students) :
    ∃ ε > 0, |teacher_average school - student_average school + 11.67| < ε := by
  sorry

#eval teacher_average { num_students := 120, num_teachers := 4, class_sizes := [60, 30, 20, 10] }
#eval student_average { num_students := 120, num_teachers := 4, class_sizes := [60, 30, 20, 10] }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_student_average_difference_l845_84599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_cos_squared_l845_84543

-- Define the function f(x) = cos²(x)
noncomputable def f (x : ℝ) : ℝ := (Real.cos x) ^ 2

-- State the theorem
theorem smallest_positive_period_of_cos_squared :
  ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = Real.pi := by
  sorry

-- You can add more lemmas or theorems here if needed for the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_cos_squared_l845_84543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_circle_properties_l845_84515

/-- A circle in polar coordinates with equation ρ = 6 cos θ -/
noncomputable def polar_circle (θ : ℝ) : ℝ := 6 * Real.cos θ

theorem polar_circle_properties :
  -- The circle passes through the pole (0, 0)
  polar_circle 0 = 0 ∧
  -- The circle passes through the point (3√2, π/4)
  polar_circle (π / 4) = 3 * Real.sqrt 2 ∧
  -- The center is on the polar axis (implicitly true for ρ = a cos θ equations)
  ∃ a : ℝ, ∀ θ : ℝ, polar_circle θ = a * Real.cos θ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_circle_properties_l845_84515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_l845_84523

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem smallest_difference (a b c : ℕ+) 
  (h1 : (a : ℕ) * b * c = factorial 6) 
  (h2 : a < b) (h3 : b < c) :
  c - a ≥ 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_l845_84523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_diagram_ratio_l845_84557

/-- Represents the sequence of diagrams --/
def DiagramSequence := ℕ → ℕ × ℕ

/-- The sequence of diagrams where the first element is the number of shaded units
    and the second element is the total number of units --/
def geometricDiagrams : DiagramSequence :=
  fun n => (1 * 4^(n-1), 4 * 4^(n-1))

/-- The ratio of shaded units to total units in the nth diagram --/
def shadedRatio (n : ℕ) : ℚ :=
  let (shaded, total) := geometricDiagrams n
  (shaded : ℚ) / total

theorem fourth_diagram_ratio :
  shadedRatio 4 = 1 / 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_diagram_ratio_l845_84557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base5_division_theorem_l845_84574

/-- Represents a number in base 5 --/
structure Base5 where
  value : Nat

/-- Converts a base 5 number to decimal --/
def to_decimal (n : Base5) : Nat := sorry

/-- Converts a decimal number to base 5 --/
def to_base5 (n : Nat) : Base5 := sorry

/-- Performs division in base 5 --/
def div_base5 (a b : Base5) : Base5 × Base5 := sorry

instance : OfNat Base5 n where
  ofNat := Base5.mk n

theorem base5_division_theorem :
  let dividend : Base5 := 3213
  let divisor : Base5 := 14
  let (quotient, remainder) := div_base5 dividend divisor
  quotient = 143 ∧ remainder = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base5_division_theorem_l845_84574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_l845_84597

theorem equation_roots : ∀ x : ℝ, x > 0 →
  (3 * Real.sqrt x + 3 * x^(-(1/2 : ℝ)) = 7) ↔ 
  (x = ((7 + Real.sqrt 13) / 6)^2 ∨ x = ((7 - Real.sqrt 13) / 6)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_l845_84597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_l845_84575

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_equality (a b : V) (lambda1 lambda2 mu1 mu2 : ℝ) 
  (h1 : ¬ ∃ (k : ℝ), b = k • a) 
  (h2 : lambda1 • a + mu1 • b = lambda2 • a + mu2 • b) : 
  lambda1 = lambda2 ∧ mu1 = mu2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_l845_84575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_container_total_weight_l845_84529

/-- A container with variable content -/
structure Container where
  /-- Weight when three-quarters full -/
  weight_three_quarters : ℚ
  /-- Weight when one-third full -/
  weight_one_third : ℚ

/-- Calculate the total weight of a fully loaded container -/
def total_weight (c : Container) : ℚ :=
  (8/5) * c.weight_three_quarters - (3/5) * c.weight_one_third

/-- Theorem: The total weight of a fully loaded container is (8/5)p - (3/5)q -/
theorem container_total_weight (c : Container) :
  total_weight c = (8/5) * c.weight_three_quarters - (3/5) * c.weight_one_third :=
by
  -- Unfold the definition of total_weight
  unfold total_weight
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_container_total_weight_l845_84529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_correct_lines_perpendicular_line2_passes_through_point_l845_84596

noncomputable section

-- Define the slope of the first line
def m1 : ℝ := -3

-- Define the first line
def line1 (x : ℝ) : ℝ := m1 * x + 4

-- Define the slope of the perpendicular line
def m2 : ℝ := -1 / m1

-- Define the point through which the perpendicular line passes
def point : ℝ × ℝ := (3, -2)

-- Define the perpendicular line
def line2 (x : ℝ) : ℝ := m2 * (x - point.1) + point.2

-- Define the intersection point
def intersection_point : ℝ × ℝ := (1.5, -0.5)

-- Theorem statement
theorem intersection_point_correct :
  line1 intersection_point.1 = intersection_point.2 ∧
  line2 intersection_point.1 = intersection_point.2 := by
  sorry

-- Verify that the lines are perpendicular
theorem lines_perpendicular :
  m1 * m2 = -1 := by
  sorry

-- Verify that line2 passes through the given point
theorem line2_passes_through_point :
  line2 point.1 = point.2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_correct_lines_perpendicular_line2_passes_through_point_l845_84596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l845_84549

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 6*x + 17) / Real.log (1/2)

-- State the theorem
theorem range_of_f :
  Set.range f = Set.Iic 3 :=
by
  sorry -- Skip the proof for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l845_84549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_reflections_l845_84508

/-- Represents the number of reflections of a light beam -/
def num_reflections : ℕ := 8

/-- The angle between the two reflecting lines -/
def angle_between_lines : ℝ := 10

/-- The angle of the final reflection -/
def final_reflection_angle : ℝ := 15

/-- The maximum angle possible for a reflection -/
def max_reflection_angle : ℝ := 90

theorem max_reflections :
  ∀ n : ℕ,
  (n : ℝ) * angle_between_lines + (final_reflection_angle - angle_between_lines) ≤ max_reflection_angle →
  num_reflections ≤ n →
  num_reflections = 8 :=
by
  intro n h1 h2
  sorry

#check max_reflections

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_reflections_l845_84508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l845_84564

-- Define the train's length in meters
noncomputable def train_length : ℝ := 360

-- Define the time to cross the pole in seconds
noncomputable def crossing_time : ℝ := 30

-- Define the conversion factor from m/s to km/h
noncomputable def mps_to_kmph : ℝ := 3600 / 1000

-- Define the speed of the train in km/h
noncomputable def train_speed : ℝ := (train_length / crossing_time) * mps_to_kmph

-- Theorem stating the speed of the train
theorem train_speed_calculation : train_speed = 43.2 := by
  -- Unfold definitions
  unfold train_speed train_length crossing_time mps_to_kmph
  -- Perform calculation
  norm_num
  -- Close the proof
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l845_84564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_semicircle_radius_is_three_l845_84514

/-- A right triangle with given side lengths -/
structure RightTriangle where
  /-- Length of side PR -/
  pr : ℝ
  /-- Length of side PQ -/
  pq : ℝ
  /-- PR is positive -/
  pr_pos : 0 < pr
  /-- PQ is positive -/
  pq_pos : 0 < pq

/-- The radius of the inscribed semicircle in a right triangle -/
noncomputable def inscribed_semicircle_radius (t : RightTriangle) : ℝ :=
  (t.pr * t.pq) / (2 * (t.pr + t.pq + Real.sqrt (t.pr^2 + t.pq^2)))

/-- The theorem stating that for a specific right triangle, the inscribed semicircle radius is 3 -/
theorem inscribed_semicircle_radius_is_three :
  ∃ t : RightTriangle, t.pr = 15 ∧ t.pq = 8 ∧ inscribed_semicircle_radius t = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_semicircle_radius_is_three_l845_84514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_locus_is_hyperbola_l845_84550

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A line in 2D space -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle in 2D space -/
structure Circle2D where
  center : Point2D
  radius : ℝ

/-- Distance between a point and a line -/
noncomputable def distPointLine (p : Point2D) (l : Line2D) : ℝ :=
  (abs (l.a * p.x + l.b * p.y + l.c)) / Real.sqrt (l.a^2 + l.b^2)

/-- Angle between a circle and a line at their intersection -/
noncomputable def angleCircleLine (c : Circle2D) (l : Line2D) : ℝ :=
  sorry

/-- The locus of centers of circles passing through a given point and intersecting a given line at a given angle -/
def circleCenterLocus (F : Point2D) (f : Line2D) (α : ℝ) : Set Point2D :=
  {P | ∃ (c : Circle2D), c.center = P ∧ 
                         distPointLine F f = distPointLine P f * Real.cos α ∧
                         angleCircleLine c f = α}

/-- Parameters of a hyperbola -/
structure HyperbolaParams where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A hyperbola in 2D space -/
structure Hyperbola2D where
  focus1 : Point2D
  focus2 : Point2D
  params : HyperbolaParams

/-- Distance between two points -/
noncomputable def distPointPoint (p1 p2 : Point2D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The set of points on a hyperbola -/
def hyperbolaPoints (h : Hyperbola2D) : Set Point2D :=
  {P | abs (distPointPoint P h.focus1 - distPointPoint P h.focus2) = 2 * h.params.a}

theorem circle_center_locus_is_hyperbola 
  (F : Point2D) (f : Line2D) (α : ℝ) 
  (hα : 0 < α ∧ α < Real.pi / 2) :
  ∃ (h : Hyperbola2D), 
    h.focus1 = F ∧ 
    h.params.c = distPointLine F f / (Real.sin α)^2 ∧
    h.params.a = distPointLine F f * Real.cos α / (Real.sin α)^2 ∧
    h.params.b = distPointLine F f / Real.sin α ∧
    circleCenterLocus F f α = hyperbolaPoints h :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_locus_is_hyperbola_l845_84550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_match_run_rate_l845_84571

/-- Represents a cricket match with given parameters -/
structure CricketMatch where
  total_overs : ℕ
  first_part_overs : ℕ
  first_part_run_rate : ℚ
  target : ℕ

/-- Calculates the required run rate for the remaining overs -/
def required_run_rate (m : CricketMatch) : ℚ :=
  let remaining_overs := m.total_overs - m.first_part_overs
  let first_part_runs := m.first_part_run_rate * m.first_part_overs
  let remaining_runs := m.target - first_part_runs
  remaining_runs / remaining_runs

/-- Theorem stating the required run rate for the given match conditions -/
theorem cricket_match_run_rate :
  let m : CricketMatch := {
    total_overs := 50,
    first_part_overs := 10,
    first_part_run_rate := 32/10,
    target := 252
  }
  required_run_rate m = 11/2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_match_run_rate_l845_84571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_satisfactory_fraction_l845_84580

/-- Represents the grade categories --/
inductive Grade
  | A
  | B
  | C
  | D
  | F

/-- Defines whether a grade is satisfactory --/
def is_satisfactory (g : Grade) : Prop :=
  match g with
  | Grade.A => True
  | Grade.B => True
  | Grade.C => True
  | _ => False

/-- Represents the count of students for each grade --/
def grade_count : Grade → ℕ
  | Grade.A => 6
  | Grade.B => 5
  | Grade.C => 4
  | Grade.D => 2
  | Grade.F => 6

/-- The total number of students --/
def total_students : ℕ := 
  grade_count Grade.A + grade_count Grade.B + grade_count Grade.C + 
  grade_count Grade.D + grade_count Grade.F

/-- The number of students with satisfactory grades --/
def satisfactory_count : ℕ :=
  (grade_count Grade.A) + (grade_count Grade.B) + (grade_count Grade.C)

/-- Theorem: The fraction of satisfactory grades is 15/23 --/
theorem satisfactory_fraction :
  (satisfactory_count : ℚ) / (total_students : ℚ) = 15 / 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_satisfactory_fraction_l845_84580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_properties_l845_84532

-- Define the circle C
def circle_C (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2

-- Define the line y = -2x
def line_center (x y : ℝ) : Prop := y = -2 * x

-- Define the point A
def point_A : ℝ × ℝ := (2, -1)

-- Define the tangent line x + y - 1 = 0
def tangent_line (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the point B
def point_B : ℝ × ℝ := (1, 2)

-- State the theorem
theorem circle_C_properties :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ x y, line_center x y → (x, y) = center) ∧
    (∀ x y, circle_C center radius x y) ∧
    circle_C center radius point_A.1 point_A.2 ∧
    (∀ x y, tangent_line x y → 
      (circle_C center radius x y → x = center.1 ∧ y = center.2)) →
    (∀ x y, circle_C (1, -2) (Real.sqrt 2) x y) ∧
    (∃ (k : ℝ), k^2 = 7 ∧
      (k * (point_B.1 - 1) + 2 - point_B.2 = 0 ∨
       -k * (point_B.1 - 1) + 2 - point_B.2 = 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_properties_l845_84532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_trailing_zeros_l845_84507

/-- Count the number of trailing zeros in the factorial of a number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem factorial_trailing_zeros (n : ℕ) (k : ℕ) (h1 : n > 4) :
  (trailingZeros (Nat.factorial n) = k ∧ trailingZeros (Nat.factorial (2*n)) = 3*k) → 
  (n = 8 ∨ n = 9 ∨ n = 13 ∨ n = 14) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_trailing_zeros_l845_84507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_l845_84591

theorem parabola_intersection (n : ℝ) (h1 : n > 0) : 
  let f1 := fun x : ℝ => x^2 + 2*x - n
  let f2 := fun x : ℝ => x^2 - 2*x - n
  let A := -1 - Real.sqrt (n + 1)
  let B := -1 + Real.sqrt (n + 1)
  let C := 1 - Real.sqrt (n + 1)
  let D := 1 + Real.sqrt (n + 1)
  (f1 A = 0 ∧ f1 B = 0 ∧ f2 C = 0 ∧ f2 D = 0 ∧ D - A = 2 * (B - C)) → n = 8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_l845_84591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_to_polar_3_neg3_l845_84588

noncomputable def rect_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 then Real.arctan (y / x)
           else if x < 0 && y ≥ 0 then Real.arctan (y / x) + Real.pi
           else if x < 0 && y < 0 then Real.arctan (y / x) - Real.pi
           else if x = 0 && y > 0 then Real.pi / 2
           else if x = 0 && y < 0 then -Real.pi / 2
           else 0  -- x = 0 and y = 0
  (r, if θ < 0 then θ + 2*Real.pi else θ)

theorem rect_to_polar_3_neg3 :
  let (r, θ) := rect_to_polar 3 (-3)
  r = 3 * Real.sqrt 2 ∧ θ = 7 * Real.pi / 4 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_to_polar_3_neg3_l845_84588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_greater_than_max_point_l845_84553

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^2 / 2 - Real.log x
noncomputable def h (x : ℝ) : ℝ := Real.log x / (2 * x)

-- Define x₁ as the extreme point of f
noncomputable def x₁ : ℝ := sorry

-- Define x₂ as the x-value where h reaches its maximum
noncomputable def x₂ : ℝ := sorry

-- Theorem statement
theorem extreme_point_greater_than_max_point : x₁ > x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_greater_than_max_point_l845_84553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_problem_l845_84583

/-- The number of years for which compound interest is calculated -/
noncomputable def compound_interest_years (principal : ℝ) (rate : ℝ) (interest : ℝ) : ℝ :=
  let future_value := principal + interest
  (Real.log (future_value / principal)) / (Real.log (1 + rate))

/-- Theorem stating that the number of years for the given conditions is 1 -/
theorem compound_interest_problem :
  compound_interest_years 1200 0.20 240 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_problem_l845_84583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_a_range_l845_84581

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Theorem for the minimum value of f(x)
theorem f_minimum_value :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≥ f x ∧ f x = -1 / Real.exp 1 := by
  sorry

-- Theorem for the range of a
theorem a_range (a : ℝ) :
  (∀ (x : ℝ), x ≥ 1 → f x ≥ a * x - 1) ↔ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_a_range_l845_84581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_base_length_l845_84548

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 + 4

-- Define the circle
def on_circle (x y : ℝ) : Prop := x^2 + y^2 = 25

-- Define the triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  let (xA, yA) := t.A
  let (xB, yB) := t.B
  let (xC, yC) := t.C
  -- Vertices on parabola
  yA = parabola xA ∧ yB = parabola xB ∧ yC = parabola xC ∧
  -- Base BC is horizontal
  yB = yC ∧
  -- A is on the circle
  on_circle xA yA ∧
  -- Area of triangle is 36
  abs ((xB - xC) * (yA - yB) / 2) = 36

-- Theorem statement
theorem triangle_base_length (t : Triangle) 
  (h : satisfies_conditions t) : abs (t.B.1 - t.C.1) = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_base_length_l845_84548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_even_sum_is_half_l845_84598

def numbers : Finset ℕ := {1, 2, 3, 5}

def isPairSumEven (pair : ℕ × ℕ) : Bool :=
  (pair.1 + pair.2) % 2 = 0

def allPairs : Finset (ℕ × ℕ) :=
  numbers.product numbers

def validPairs : Finset (ℕ × ℕ) :=
  allPairs.filter (fun pair => pair.1 < pair.2)

theorem probability_even_sum_is_half :
  (validPairs.filter (fun pair => isPairSumEven pair)).card / validPairs.card = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_even_sum_is_half_l845_84598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_max_min_values_l845_84572

noncomputable def y (x a b : ℝ) : ℝ := (Real.cos x)^2 - a * Real.sin x + b

theorem function_max_min_values (a b : ℝ) :
  (∀ x, y x a b ≤ 0) ∧
  (∃ x, y x a b = 0) ∧
  (∀ x, y x a b ≥ -4) ∧
  (∃ x, y x a b = -4) →
  ((a = 2 ∧ b = -2) ∨ (a = -2 ∧ b = -2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_max_min_values_l845_84572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bathtub_fill_time_l845_84586

/-- Represents the time (in minutes) it takes to fill the bathtub with the drain closed -/
def fill_time : ℝ → Prop := sorry

/-- Represents the time (in minutes) it takes to drain the bathtub -/
def drain_time : ℝ → Prop := sorry

/-- Represents the time (in minutes) it takes to fill the bathtub with the drain open -/
def fill_time_open_drain : ℝ → Prop := sorry

theorem bathtub_fill_time 
  (h1 : drain_time 12)
  (h2 : fill_time_open_drain 60) :
  fill_time 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bathtub_fill_time_l845_84586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_equation_solutions_l845_84554

theorem cos_equation_solutions (x : ℝ) : 
  -π ≤ x ∧ x ≤ π → 
  (x = π/2 ∨ x = -π/2) → 
  Real.cos (6*x) + (Real.cos (4*x))^2 + (Real.cos (3*x))^3 + (Real.cos (2*x))^4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_equation_solutions_l845_84554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_k_is_15_l845_84525

/-- The set of all distinct possible values for k, where x^2 - kx + 24 has only positive integer roots -/
def possible_k_values : Finset ℕ :=
  {10, 11, 14, 25}

/-- The average of a finite set of natural numbers -/
def average (s : Finset ℕ) : ℚ :=
  (s.sum id) / s.card

/-- Theorem stating that the average of possible k values is 15 -/
theorem average_k_is_15 : average possible_k_values = 15 := by
  sorry

#eval average possible_k_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_k_is_15_l845_84525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l845_84578

noncomputable def f (x : ℝ) := (Real.exp x - Real.exp (-x)) / 2

theorem m_range (m : ℝ) :
  (∀ θ : ℝ, θ ∈ Set.Ioo 0 (Real.pi / 2) → f (m * Real.sin θ) + f (1 - m) > 0) →
  m ∈ Set.Iic 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l845_84578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_rate_of_change_at_one_l845_84540

-- Define the function f(x) = ln(2x - 1)
noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x - 1)

-- State the theorem
theorem instantaneous_rate_of_change_at_one :
  (deriv f) 1 = 2 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_rate_of_change_at_one_l845_84540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_zero_l845_84556

/-- Converts a base 10 number to base 6 --/
noncomputable def toBase6 (n : ℕ) : ℕ := sorry

/-- Converts a base 6 number to base 10 --/
noncomputable def fromBase6 (n : ℕ) : ℕ := sorry

/-- Checks if a number is a single digit in base 6 --/
def isSingleDigitBase6 (n : ℕ) : Prop := n < 6

theorem absolute_difference_zero (A B : ℕ) 
  (h1 : isSingleDigitBase6 A) 
  (h2 : isSingleDigitBase6 B) 
  (h3 : fromBase6 (100 * B + 10 * B + A) + 
        fromBase6 (100 * 5 + 10 * 2 + B) + 
        fromBase6 (100 * A + 10 * 2 + 4) = 
        fromBase6 (1000 * A + 100 * 2 + 10 * 4 + 2)) :
  toBase6 (Int.natAbs (A - B)) = 0 := by
  sorry

#check absolute_difference_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_zero_l845_84556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_fours_expressions_l845_84565

def four_fours (expr : ℕ → ℕ → ℕ → ℕ → ℚ) : Prop :=
  (∃ a b c d : ℕ, a = 4 ∧ b = 4 ∧ c = 4 ∧ d = 4) ∧
  (expr 4 4 4 4 = 5 ∨ expr 4 4 4 4 = 6 ∨ expr 4 4 4 4 = 7 ∨ expr 4 4 4 4 = 8 ∨ expr 4 4 4 4 = 9)

theorem four_fours_expressions : 
  ∃ expr₁ expr₂ expr₃ expr₄ expr₅ : ℕ → ℕ → ℕ → ℕ → ℚ,
    four_fours expr₁ ∧ 
    four_fours expr₂ ∧ 
    four_fours expr₃ ∧ 
    four_fours expr₄ ∧ 
    four_fours expr₅ ∧ 
    expr₁ 4 4 4 4 = 5 ∧
    expr₂ 4 4 4 4 = 6 ∧
    expr₃ 4 4 4 4 = 7 ∧
    expr₄ 4 4 4 4 = 8 ∧
    expr₅ 4 4 4 4 = 9 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_fours_expressions_l845_84565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l845_84546

theorem area_between_curves : ∫ x in (0 : Real)..(1 : Real), x^2 - x^3 = 1/12 := by
  -- Define the curves
  let f (x : Real) := x^2
  let g (x : Real) := x^3
  
  -- Define the area function
  let area := ∫ x in (0 : Real)..(1 : Real), f x - g x
  
  -- State the theorem
  have h : area = 1/12 := by
    -- The actual proof would go here
    sorry
  
  -- Use the proved equality
  exact h


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l845_84546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_solution_range_part2_l845_84533

-- Define the functions f and g
noncomputable def f (a x : ℝ) : ℝ := |x + 1 - 2*a| + |x - a^2|
noncomputable def g (x : ℝ) : ℝ := x^2 - 2*x - 4 + 4 / (x - 1)^2

-- Part 1: Solution set of f(2a²-1) > 4|a-1|
theorem solution_set_part1 (a : ℝ) : 
  f a (2*a^2 - 1) > 4*|a - 1| ↔ a < -5/3 ∨ a > 1 := by
  sorry

-- Part 2: Range of a where ∃x,y such that f(x) + g(y) ≤ 0
theorem solution_range_part2 (a : ℝ) : 
  (∃ x y : ℝ, f a x + g y ≤ 0) → 0 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_solution_range_part2_l845_84533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orangeade_pricing_l845_84568

/-- Represents the price of orangeade per glass -/
@[reducible] def Price := ℝ

/-- Represents the volume of liquid (orange juice or water) -/
@[reducible] def Volume := ℝ

/-- Calculates the revenue given volume and price -/
def revenue (volume : Volume) (price : Price) : ℝ := volume * price

theorem orangeade_pricing 
  (orange_juice : Volume) 
  (water_day1 : Volume) 
  (price_day2 : Price) :
  orange_juice > 0 →
  water_day1 > 0 →
  orange_juice = water_day1 →
  price_day2 = 0.40 →
  ∃ (price_day1 : Price),
    revenue (orange_juice + water_day1) price_day1 = 
    revenue (orange_juice + 2 * water_day1) price_day2 ∧
    price_day1 = 0.60 :=
by
  intro h_oj_pos h_w1_pos h_oj_eq_w1 h_p2
  use 0.60
  constructor
  · simp [revenue, h_oj_eq_w1, h_p2]
    ring
  · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orangeade_pricing_l845_84568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cards_arrangement_exists_l845_84521

/-- Represents a card with two numbers -/
structure Card where
  front : Nat
  back : Nat

/-- The theorem statement -/
theorem cards_arrangement_exists :
  ∀ (cards : Finset Card),
  (cards.card = 20) →
  (∀ n : Nat, 1 ≤ n ∧ n ≤ 20 → (cards.filter (λ c => c.front = n ∨ c.back = n)).card = 2) →
  ∃ (arrangement : Finset Card),
  arrangement.card = 20 ∧
  (∀ c₁ c₂ : Card, c₁ ∈ arrangement → c₂ ∈ arrangement → c₁ ≠ c₂ → c₁.front ≠ c₂.front) ∧
  (∀ n : Nat, 1 ≤ n ∧ n ≤ 20 → ∃ c : Card, c ∈ arrangement ∧ c.front = n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cards_arrangement_exists_l845_84521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_mint_count_l845_84517

/-- The number of green tea leaves per sprig of mint -/
def green_tea_per_mint : ℕ := 2

/-- The factor by which the new mud reduces ingredient effectiveness -/
def new_mud_factor : ℚ := 1/2

/-- The number of green tea leaves required in the new mud mixture -/
def new_mud_green_tea : ℕ := 12

/-- The number of sprigs of mint in the original mud mixture -/
def original_mint_sprigs : ℕ := 3

theorem original_mint_count :
  original_mint_sprigs = new_mud_green_tea / green_tea_per_mint * (new_mud_factor.num / new_mud_factor.den) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_mint_count_l845_84517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chad_savings_percentage_l845_84559

noncomputable def mowing_income : ℚ := 600
noncomputable def birthday_income : ℚ := 250
noncomputable def videogames_income : ℚ := 150
noncomputable def oddjobs_income : ℚ := 150
noncomputable def amount_saved : ℚ := 460

noncomputable def total_earnings : ℚ := mowing_income + birthday_income + videogames_income + oddjobs_income

noncomputable def savings_percentage : ℚ := (amount_saved / total_earnings) * 100

theorem chad_savings_percentage :
  savings_percentage = 40 := by
  -- Unfold definitions
  unfold savings_percentage
  unfold total_earnings
  unfold mowing_income birthday_income videogames_income oddjobs_income amount_saved
  -- Simplify the expression
  simp [div_mul_eq_mul_div]
  -- Evaluate the fraction
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chad_savings_percentage_l845_84559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_ball_radius_l845_84541

theorem larger_ball_radius (r : ℝ) (h : r = 2) : 
  (6 * (4/3 * Real.pi * r^3))^(1/3) = 48^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_ball_radius_l845_84541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l845_84531

theorem trig_identities (α β : ℝ)
  (h1 : Real.sin α = -3/5)
  (h2 : Real.sin β = 12/13)
  (h3 : α ∈ Set.Ioo π (3*π/2))
  (h4 : β ∈ Set.Ioo (π/2) π) :
  Real.sin (α - β) = 63/65 ∧ Real.cos (2*α) = 7/25 ∧ Real.tan (β/2) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l845_84531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_sum_l845_84569

/-- Three points in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Vector between two points -/
def vector (p q : Point3D) : Point3D :=
  ⟨q.x - p.x, q.y - p.y, q.z - p.z⟩

/-- Collinearity condition for three points -/
def collinear (p q r : Point3D) : Prop :=
  ∃ t : ℝ, vector p q = ⟨t * (r.x - p.x), t * (r.y - p.y), t * (r.z - p.z)⟩

/-- The main theorem -/
theorem collinear_points_sum (a b : ℝ) :
  let A : Point3D := ⟨1, 5, -2⟩
  let B : Point3D := ⟨3, 4, 1⟩
  let C : Point3D := ⟨a, 3, b + 2⟩
  collinear A B C → a + b = 7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_sum_l845_84569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_circle_to_line_l845_84567

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + (Real.sqrt 3 / 2) * t, (1 / 2) * t)

-- Define the distance function from a point to a line
noncomputable def dist_point_to_line (x y : ℝ) : ℝ :=
  abs (x - Real.sqrt 3 * y - 1) / Real.sqrt (1 + 3)

-- Theorem statement
theorem distance_range_circle_to_line :
  ∀ x y : ℝ, circle_C x y →
  ∃ d : ℝ, d = dist_point_to_line x y ∧
  Real.sqrt 3 - 1 ≤ d ∧ d ≤ Real.sqrt 3 + 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_circle_to_line_l845_84567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l845_84576

/-- Given a triangle ABC with sides a and b opposite to angles A and B respectively,
    prove that if a = 5, b = 4, and cos(A-B) = 31/32, then sin B = √7/4 and cos C = 1/8 -/
theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  a = 5 →
  b = 4 →
  Real.cos (A - B) = 31/32 →
  Real.sin B = Real.sqrt 7 / 4 ∧ Real.cos C = 1/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l845_84576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_reflection_point_l845_84558

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a light ray reflection on the x-axis -/
def reflectionPoint (start : Point) (finish : Point) : Point :=
  { x := sorry, y := 0 }

/-- Theorem: The reflection point of a light ray passing through (-3, 4) and (0, 2) is (-1, 0) -/
theorem light_reflection_point :
  let start : Point := { x := -3, y := 4 }
  let finish : Point := { x := 0, y := 2 }
  let reflection := reflectionPoint start finish
  reflection.x = -1 ∧ reflection.y = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_reflection_point_l845_84558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l845_84503

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the focal length
noncomputable def focal_length (a b : ℝ) : ℝ :=
  2 * Real.sqrt (a^2 + b^2)

-- Define the distance from focus to asymptote
def focus_to_asymptote (b : ℝ) : ℝ := b

-- Theorem statement
theorem hyperbola_asymptotes (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : focus_to_asymptote b = (1/4) * focal_length a b) :
  ∃ (k : ℝ), k = Real.sqrt 3 ∧ 
  (∀ (x y : ℝ), hyperbola a b x y ↔ (x = k*y ∨ x = -k*y)) := by
  sorry

#check hyperbola_asymptotes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l845_84503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_of_zeros_l845_84518

noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.log x

theorem min_distance_of_zeros (x₀ x₁ : ℝ) (h₁ : f x₀ = 0) (h₂ : f x₁ = 0) (h₃ : x₀ ≠ x₁) :
  ∃ (y₀ y₁ : ℝ), f y₀ = 0 ∧ f y₁ = 0 ∧ y₀ ≠ y₁ ∧ |y₀ - y₁| = π / 2 - 1 ∧
  ∀ (z₀ z₁ : ℝ), f z₀ = 0 → f z₁ = 0 → z₀ ≠ z₁ → |z₀ - z₁| ≥ π / 2 - 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_of_zeros_l845_84518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_l845_84566

noncomputable def f (a x : ℝ) : ℝ :=
  (1 / (2 * a * Real.sqrt (1 + a^2))) * Real.log ((a + Real.sqrt (1 + a^2) * Real.tanh x) / (a - Real.sqrt (1 + a^2) * Real.tanh x))

theorem derivative_f (a x : ℝ) (ha : a ≠ 0) :
  deriv (f a) x = 1 / (a^2 * Real.cosh x^2 + (1 + a^2) * Real.sinh x^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_l845_84566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_representatives_correct_l845_84592

/-- Represents the number of representatives for a given class size -/
def representatives (x : ℕ) : ℕ := (x + 3) / 10

/-- The election rule for representatives -/
def electionRule (x : ℕ) : ℕ :=
  if x % 10 > 6 then x / 10 + 1 else x / 10

theorem representatives_correct (x : ℕ) :
  representatives x = electionRule x :=
by
  -- Unfold the definitions
  unfold representatives electionRule
  -- Split into cases based on the remainder
  by_cases h : x % 10 > 6
  · -- Case: remainder > 6
    simp [h]
    sorry -- Proof details omitted
  · -- Case: remainder ≤ 6
    simp [h]
    sorry -- Proof details omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_representatives_correct_l845_84592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bertha_ball_loss_rate_l845_84584

/-- Represents the number of games Bertha plays -/
def total_games : ℕ := 20

/-- Represents the number of games after which a ball wears out -/
def wear_out_rate : ℕ := 10

/-- Represents the number of games after which Bertha buys a canister of balls -/
def purchase_rate : ℕ := 4

/-- Represents the number of balls in a canister -/
def balls_per_canister : ℕ := 3

/-- Represents the initial number of balls Bertha has -/
def initial_balls : ℕ := 2

/-- Represents the number of balls Bertha gives away at the start -/
def given_away_balls : ℕ := 1

/-- Represents the number of balls Bertha has after playing the total number of games -/
def final_balls : ℕ := 10

/-- Calculates the number of balls Bertha would have without losing any -/
def balls_without_loss : ℕ := 
  initial_balls - given_away_balls + 
  (total_games / purchase_rate) * balls_per_canister - 
  (total_games / wear_out_rate)

/-- Calculates the number of balls Bertha loses -/
def balls_lost : ℕ := balls_without_loss - final_balls

/-- Represents the approximate number of games after which Bertha loses a ball -/
def loss_rate : ℕ := 7

/-- Theorem stating that Bertha loses a ball approximately every 7 games -/
theorem bertha_ball_loss_rate : 
  (total_games : ℚ) / (balls_lost : ℚ) ≥ (loss_rate : ℚ) ∧ (total_games : ℚ) / (balls_lost : ℚ) < (loss_rate + 1 : ℚ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bertha_ball_loss_rate_l845_84584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_speed_approx_6_l845_84522

/-- Represents a square field with its area and the time taken to cross it diagonally -/
structure SquareField where
  area : ℝ
  crossTime : ℝ

/-- Calculates the walking speed in km/h given a square field -/
noncomputable def walkingSpeed (field : SquareField) : ℝ :=
  let sideLength := Real.sqrt field.area
  let diagonalLength := sideLength * Real.sqrt 2
  let speedMPS := diagonalLength / field.crossTime
  speedMPS * 3.6

/-- Theorem stating that the walking speed is approximately 6 km/h for the given conditions -/
theorem walking_speed_approx_6 (field : SquareField) 
    (h1 : field.area = 112.5) 
    (h2 : field.crossTime = 9) : 
    ∃ ε > 0, |walkingSpeed field - 6| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_speed_approx_6_l845_84522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_is_contrapositive_of_r_l845_84539

-- Define propositions as types
variable (A B : Prop)

-- Define the proposition p
def p (A B : Prop) : Prop := A → B

-- Define q as the inverse proposition of p
def q (A B : Prop) : Prop := B → A

-- Define r as the negation of p
def r (A B : Prop) : Prop := ¬A → ¬B

-- Theorem: q is the contrapositive of r
theorem q_is_contrapositive_of_r (A B : Prop) : q A B ↔ r A B := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_is_contrapositive_of_r_l845_84539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l845_84526

noncomputable def f (ω a x : ℝ) : ℝ := 4 * Real.cos (ω * x) * Real.sin (ω * x + Real.pi / 6) + a

theorem function_properties (ω a : ℝ) (h_ω : ω > 0) :
  (∀ x, f ω a x ≤ 2) ∧
  (∃ x, f ω a x = 2) ∧
  (∀ x y, y - x = Real.pi / ω → (f ω a x = 2 → f ω a y = 2)) →
  a = -1 ∧ ω = 1 ∧
  ∀ x ∈ Set.Icc (Real.pi / 6 : ℝ) (2 * Real.pi / 3),
    ∀ y ∈ Set.Icc (Real.pi / 6 : ℝ) (2 * Real.pi / 3),
      x < y → f ω a y < f ω a x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l845_84526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_power_speed_l845_84512

-- Define the constants and variables
variable (A S ρ v₀ v : ℝ)

-- Define the force function
noncomputable def F (A S ρ v₀ v : ℝ) : ℝ := (A * S * ρ * (v₀ - v)^2) / 2

-- Define the power function
noncomputable def N (A S ρ v₀ v : ℝ) : ℝ := F A S ρ v₀ v * v

-- Theorem statement
theorem max_power_speed (h₀ : v₀ = 4.8) (h₁ : v > 0) (h₂ : v < v₀) :
  (∀ u, u > 0 → u < v₀ → N A S ρ v₀ v ≥ N A S ρ v₀ u) → v = v₀ / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_power_speed_l845_84512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_width_is_40_l845_84504

/-- Calculates the width of a river given its depth, flow rate, and volume of water flowing into the sea per minute. -/
noncomputable def river_width (depth : ℝ) (flow_rate_kmph : ℝ) (volume_per_minute : ℝ) : ℝ :=
  let flow_rate_mpm := flow_rate_kmph * 1000 / 60
  volume_per_minute / (flow_rate_mpm * depth)

/-- Theorem stating that for a river with given parameters, its width is 40 meters. -/
theorem river_width_is_40 :
  let depth : ℝ := 4
  let flow_rate_kmph : ℝ := 4
  let volume_per_minute : ℝ := 10666.666666666666
  river_width depth flow_rate_kmph volume_per_minute = 40 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval river_width 4 4 10666.666666666666

end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_width_is_40_l845_84504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_17_549_l845_84542

/-- The cycle of units digits for powers of 17 -/
def units_cycle : List Nat := [7, 9, 3, 1]

/-- The length of the units digit cycle for powers of 17 -/
def cycle_length : Nat := units_cycle.length

/-- The function to get the units digit of 17^n -/
def units_digit (n : Nat) : Nat :=
  units_cycle[((n - 1) % cycle_length)]'(by
    have h : cycle_length = units_cycle.length := rfl
    rw [h]
    apply Nat.mod_lt
    simp [cycle_length]
    exact Nat.zero_lt_succ _
  )

/-- Theorem: The units digit of 17^549 is 7 -/
theorem units_digit_17_549 : units_digit 549 = 7 := by
  rfl

#eval units_digit 549

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_17_549_l845_84542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_order_l845_84577

noncomputable def m : ℝ := (0.3 : ℝ) ^ (0.2 : ℝ)
noncomputable def n : ℝ := Real.log 3 / Real.log 0.2
noncomputable def p : ℝ := Real.sin 1 + Real.cos 1

theorem decreasing_order : p > m ∧ m > n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_order_l845_84577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_father_vs_mother_optimal_l845_84562

/-- Represents the probability of one player defeating another -/
abbrev WinProbability := ℝ

/-- The tournament structure with three players -/
structure Tournament where
  FM : WinProbability  -- Probability of Father beating Mother
  FS : WinProbability  -- Probability of Father beating Son
  MS : WinProbability  -- Probability of Mother beating Son

/-- Calculate the probability of Father winning the championship when starting with Father vs Mother -/
def probFatherWinsFM (t : Tournament) : ℝ :=
  t.FM * t.FS +
  t.FM * (1 - t.FS) * t.MS * t.FM +
  (1 - t.FM) * (1 - t.MS) * t.FS * t.FM

/-- Calculate the probability of Father winning the championship when starting with Father vs Son -/
def probFatherWinsFS (t : Tournament) : ℝ :=
  t.FS * t.FM +
  t.FS * (1 - t.FM) * (1 - t.MS) * t.FS +
  (1 - t.FS) * t.MS * t.FM * t.FS

/-- The main theorem stating that starting with Father vs Mother is optimal -/
theorem father_vs_mother_optimal (t : Tournament) 
  (h1 : 0 ≤ t.FM ∧ t.FM ≤ 1)
  (h2 : 0 ≤ t.FS ∧ t.FS ≤ 1)
  (h3 : 0 ≤ t.MS ∧ t.MS ≤ 1)
  (h4 : t.FS < t.FM)  -- Son is the strongest player
  : probFatherWinsFM t > probFatherWinsFS t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_father_vs_mother_optimal_l845_84562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_a_value_l845_84520

/-- Profit function for commodity A -/
noncomputable def P (x : ℝ) : ℝ := x / 4

/-- Profit function for commodity B -/
noncomputable def Q (x a : ℝ) : ℝ := (a / 2) * Real.sqrt x

/-- The minimum value of a that satisfies the profit condition -/
noncomputable def min_a : ℝ := Real.sqrt 5

/-- Theorem stating the minimum value of a -/
theorem minimum_a_value :
  ∀ a : ℝ, a > 0 →
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 20 → P (20 - x) + Q x a ≥ 5) →
  a ≥ min_a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_a_value_l845_84520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_of_X_l845_84527

/-- The expected value of a binomial distribution with n trials and probability p -/
noncomputable def binomial_expected_value (n : ℕ) (p : ℝ) : ℝ := n * p

/-- X is a random variable following a binomial distribution B(6, 1/2) -/
noncomputable def X : ℝ := binomial_expected_value 6 (1/2)

/-- Theorem: The expected value of X is 3 -/
theorem expected_value_of_X : X = 3 := by
  -- Unfold the definition of X
  unfold X
  -- Unfold the definition of binomial_expected_value
  unfold binomial_expected_value
  -- Simplify the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_of_X_l845_84527
