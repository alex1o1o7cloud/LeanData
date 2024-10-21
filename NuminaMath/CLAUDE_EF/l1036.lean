import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_point_five_liters_in_pints_l1036_103694

/-- Conversion factor from liters to pints -/
noncomputable def liters_to_pints : ℚ := 2625 / 1250

/-- Proves that 2.5 liters is equal to 5.25 pints -/
theorem two_point_five_liters_in_pints : 
  (5 / 2 : ℚ) * liters_to_pints = 21 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_point_five_liters_in_pints_l1036_103694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_algebra_test_female_students_l1036_103654

theorem algebra_test_female_students 
  (total_average : ℝ) 
  (male_count : ℕ) 
  (male_average : ℝ) 
  (female_average : ℝ) 
  (h1 : total_average = 90)
  (h2 : male_count = 8)
  (h3 : male_average = 85)
  (h4 : female_average = 92) : ℕ := by
  
  -- The proof goes here
  sorry

#check algebra_test_female_students

end NUMINAMATH_CALUDE_ERRORFEEDBACK_algebra_test_female_students_l1036_103654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_fourth_term_l1036_103697

theorem geometric_sequence_fourth_term 
  (a₁ : ℝ) 
  (a₈ : ℝ) 
  (h₁ : a₁ = 125) 
  (h₂ : a₈ = 72) : 
  a₁ * ((a₈ / a₁) ^ (1 / 7))^3 = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_fourth_term_l1036_103697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l1036_103629

theorem divisibility_condition (n : ℕ) (hn : n > 0) :
  (3 ∣ n * 2^n + 1) → (n % 6 = 1 ∨ n % 6 = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l1036_103629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_first_step_l1036_103620

def f (x : ℝ) : ℝ := 7 * x^6 + 6 * x^4 + 3 * x^2 + 2

def horner_first_step (a₆ : ℝ) (x : ℝ) : ℝ := a₆ * x + 0

theorem horner_method_first_step :
  let x : ℝ := 4
  horner_first_step 7 x = 28 :=
by
  simp [horner_first_step]
  norm_num

#eval horner_first_step 7 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_first_step_l1036_103620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_triangle_perimeter_l1036_103681

/-- A right prism with equilateral triangular bases -/
structure RightPrism where
  height : ℝ
  base_side : ℝ

/-- A triangle in 3D space -/
structure Triangle3D where
  X : ℝ × ℝ × ℝ
  Y : ℝ × ℝ × ℝ
  Z : ℝ × ℝ × ℝ

/-- The perimeter of a triangle in 3D space -/
noncomputable def perimeter (t : Triangle3D) : ℝ :=
  let (x1, y1, z1) := t.X
  let (x2, y2, z2) := t.Y
  let (x3, y3, z3) := t.Z
  ((x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2).sqrt +
  ((x2 - x3)^2 + (y2 - y3)^2 + (z2 - z3)^2).sqrt +
  ((x3 - x1)^2 + (y3 - y1)^2 + (z3 - z1)^2).sqrt

theorem prism_triangle_perimeter (p : RightPrism) (t : Triangle3D) :
  p.height = 20 ∧ p.base_side = 10 ∧
  t.X = (0, 0, 0) ∧ t.Y = (5, -5 * Real.sqrt 3, 0) ∧ t.Z = (5, -5 * Real.sqrt 3, -20) →
  perimeter t = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_triangle_perimeter_l1036_103681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_shaded_area_hexagon_triangle_l1036_103661

/-- The area of the non-shaded region in a regular hexagon with an inscribed equilateral triangle --/
theorem non_shaded_area_hexagon_triangle (s : ℝ) (h : s = 12) : ∃ A : ℝ,
  A = (3 * Real.sqrt 3 / 2 * s^2) - (Real.sqrt 3 / 4 * (2 * s)^2) ∧
  A = 288 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_shaded_area_hexagon_triangle_l1036_103661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_ratio_l1036_103666

/-- The parabola C defined by y^2 = 4x --/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The focus of the parabola --/
def F : ℝ × ℝ := (1, 0)

/-- A line with slope 2 passing through F --/
def L : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 2 * (p.1 - 1)}

/-- Points A and B are the intersections of L and C --/
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

/-- A is above the x-axis --/
axiom A_above_x : A.2 > 0

/-- Distance between two points --/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_intersection_ratio :
  distance A F / distance B F = (Real.sqrt 5 + 3) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_ratio_l1036_103666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_29_l1036_103634

def a : ℕ → ℕ
  | 0 => 1  -- Add this case to handle Nat.zero
  | 1 => 1
  | n + 1 => 2 * a n + 3

theorem a_4_equals_29 : a 4 = 29 := by
  -- Unfold the definition of a for the first few steps
  have h1 : a 2 = 2 * a 1 + 3 := rfl
  have h2 : a 3 = 2 * a 2 + 3 := rfl
  have h3 : a 4 = 2 * a 3 + 3 := rfl

  -- Calculate the values step by step
  have h4 : a 1 = 1 := rfl
  have h5 : a 2 = 5 := by simp [h1, h4]
  have h6 : a 3 = 13 := by simp [h2, h5]
  have h7 : a 4 = 29 := by simp [h3, h6]

  -- The final result
  exact h7

#eval a 4  -- This will evaluate a 4 and print the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_29_l1036_103634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_sum_l1036_103663

/-- The sum of an infinite geometric series with first term a and common ratio r -/
noncomputable def geometric_series_sum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- Theorem: The sum of the infinite geometric series 1/2 + 1/4 + 1/8 + 1/16 + ... is 1 -/
theorem infinite_geometric_series_sum :
  let a : ℝ := 1/2  -- first term
  let r : ℝ := 1/2  -- common ratio
  geometric_series_sum a r = 1 := by
  sorry

#check infinite_geometric_series_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_sum_l1036_103663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_container_height_difference_l1036_103673

/-- The radius of each spherical ball in cm -/
def ball_radius : ℝ := 5

/-- The number of balls in each container -/
def num_balls : ℕ := 100

/-- The height of Container A in cm -/
def height_A : ℝ := 2 * ball_radius * num_balls

/-- The height of Container B in cm -/
noncomputable def height_B : ℝ := 10 * ball_radius * Real.sqrt 3

/-- The positive difference in heights between Container A and Container B -/
noncomputable def height_difference : ℝ := height_A - height_B

theorem container_height_difference : 
  height_difference = 50 * (2 - Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_container_height_difference_l1036_103673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_7_30_l1036_103615

-- Define the angle of each hour mark on the clock
noncomputable def hourMarkAngle : ℝ := 360 / 12

-- Define the angle the minute hand moves per minute
noncomputable def minuteHandAnglePerMinute : ℝ := 360 / 60

-- Define the position of the minute hand at 7:30
noncomputable def minuteHandPosition : ℝ := 30 * minuteHandAnglePerMinute

-- Define the position of the hour hand at 7:30
noncomputable def hourHandPosition : ℝ := 7 * hourMarkAngle + (30 / 60) * hourMarkAngle

-- Theorem: The smaller angle between the hour hand and minute hand at 7:30 is 45°
theorem clock_angle_at_7_30 : 
  min (|hourHandPosition - minuteHandPosition|) (360 - |hourHandPosition - minuteHandPosition|) = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_7_30_l1036_103615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fred_remaining_sheets_l1036_103617

theorem fred_remaining_sheets :
  ∀ (fred_initial jane_planned : ℕ),
    fred_initial = 212 →
    jane_planned = 307 →
    let jane_actual := (jane_planned * 3) / 2;
    let fred_total := fred_initial + jane_actual;
    let charles_sheets := (fred_total * 25) / 100;
    fred_total - charles_sheets = 389 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fred_remaining_sheets_l1036_103617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_ray_cos_theta_l1036_103649

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := y^2 = x^2 - x + 1

-- Define a ray from the origin
def ray_from_origin (θ : ℝ) (x y : ℝ) : Prop :=
  y = Real.tan θ * x ∧ x ≥ 0 ∧ y ≥ 0

-- Define tangency condition
def is_tangent (θ : ℝ) : Prop :=
  ∃ (x y : ℝ), hyperbola x y ∧ ray_from_origin θ x y ∧
  ∀ (x' y' : ℝ), x' ≠ x → y' ≠ y → hyperbola x' y' → ¬(ray_from_origin θ x' y')

-- Theorem statement
theorem tangent_ray_cos_theta :
  ∀ θ : ℝ, is_tangent θ → Real.cos θ = 2 / Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_ray_cos_theta_l1036_103649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_unique_submission_scores_20_l1036_103669

/-- The set of Clay Millennium Problems -/
def MillenniumProblems : Set String := sorry

/-- A function that checks if a submission is a valid Millennium Problem -/
def isValidSubmission : String → Bool := sorry

/-- A function that checks if a submission is unique among all team submissions -/
def isUniqueSubmission : String → Bool := sorry

/-- The scoring function for a submission -/
def score (submission : String) : ℕ :=
  if isValidSubmission submission ∧ isUniqueSubmission submission then 20 else 0

/-- Theorem: A correct and unique submission scores 20 points -/
theorem correct_unique_submission_scores_20 (submission : String) :
  isValidSubmission submission = true → isUniqueSubmission submission = true → score submission = 20 :=
by
  intros hValid hUnique
  unfold score
  simp [hValid, hUnique]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_unique_submission_scores_20_l1036_103669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1036_103652

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 2 * x^2 - x = 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | m * x^2 - m * x - 1 = 0}

-- Define the condition ((¬_U A) ∩ B = ∅)
def condition (m : ℝ) : Prop := (Aᶜ ∩ B m) = ∅

-- State the theorem
theorem range_of_m : ∀ m : ℝ, condition m ↔ -4 ≤ m ∧ m ≤ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1036_103652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_winner_l1036_103616

/-- Represents a hex game tournament. -/
structure Tournament (n : ℕ) where
  /-- The result of a match between two players. -/
  result : Fin n → Fin n → Bool
  /-- No draws: for any two distinct players, one beats the other. -/
  no_draws : ∀ i j : Fin n, i ≠ j → result i j ≠ result j i
  /-- A player doesn't play against themselves. -/
  no_self_play : ∀ i : Fin n, result i i = false

/-- A player i has beaten player j either directly or indirectly. -/
inductive has_beaten (t : Tournament n) : Fin n → Fin n → Prop
  | direct (i j : Fin n) : t.result i j → has_beaten t i j
  | indirect (i j k : Fin n) : t.result i k → has_beaten t k j → has_beaten t i j

/-- There exists a player who has beaten all other players either directly or indirectly. -/
theorem exists_winner (n : ℕ) (t : Tournament n) : ∃ i : Fin n, ∀ j : Fin n, i ≠ j → has_beaten t i j := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_winner_l1036_103616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gildas_marbles_l1036_103687

theorem gildas_marbles (M : ℝ) (hM : M > 0) : 
  let remaining_after_pedro : ℝ := M * (1 - 0.3)
  let remaining_after_ebony : ℝ := remaining_after_pedro * (1 - 0.2)
  let remaining_after_jimmy : ℝ := remaining_after_ebony * (1 - 0.15)
  let final_remaining : ℝ := remaining_after_jimmy * (1 - 0.1)
  final_remaining / M = 0.4284 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gildas_marbles_l1036_103687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_configuration_l1036_103623

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Checks if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - q.x) = (r.y - q.y) * (q.x - p.x)

/-- Checks if a point is inside a triangle -/
def point_inside_triangle (p : Point) (t : Triangle) : Prop := sorry

/-- Represents a configuration of blue and red points -/
structure Configuration where
  blue_points : List Point
  red_points : List Point
  outer_triangle : Triangle

/-- Checks if a configuration is valid according to the problem conditions -/
def valid_configuration (config : Configuration) : Prop :=
  config.blue_points.length = 10 ∧
  config.red_points.length = 20 ∧
  (∀ p, p ∈ config.blue_points → point_inside_triangle p config.outer_triangle) ∧
  (∀ p, p ∈ config.red_points → point_inside_triangle p config.outer_triangle) ∧
  (∀ p q r, p ∈ config.blue_points → q ∈ config.blue_points → r ∈ config.blue_points → ¬(collinear p q r)) ∧
  (∀ t : Triangle, (∀ v, v ∈ config.blue_points → (v = t.a ∨ v = t.b ∨ v = t.c)) →
    ∃ r, r ∈ config.red_points ∧ point_inside_triangle r t)

/-- The main theorem stating that no valid configuration exists -/
theorem no_valid_configuration :
  ¬∃ (config : Configuration), valid_configuration config := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_configuration_l1036_103623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_half_l1036_103603

theorem cos_alpha_plus_pi_half (α : Real) 
  (h1 : Real.cos α = 1/4) 
  (h2 : α ∈ Set.Ioo (3*Real.pi/2) (2*Real.pi)) : 
  Real.cos (α + Real.pi/2) = Real.sqrt 15 / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_half_l1036_103603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_functions_property_l1036_103655

def f₁ (x : ℝ) : ℝ := x^2 + x + 5
def f₂ (x : ℝ) : ℝ := x^2 + 3*x - 2

theorem quadratic_functions_property :
  (∀ x y, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 2 → x < y → f₁ x - f₂ x > f₁ y - f₂ y) ∧
  (∀ x, x ∈ Set.Icc 1 2 → f₁ x - f₂ x ≤ 5) ∧
  (∃ x, x ∈ Set.Icc 1 2 ∧ f₁ x - f₂ x = 5) ∧
  (∀ x, x ∈ Set.Icc 1 2 → f₁ x - f₂ x ≥ 3) ∧
  (∃ x, x ∈ Set.Icc 1 2 ∧ f₁ x - f₂ x = 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_functions_property_l1036_103655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sequence_finite_l1036_103630

/-- A sequence of primes where each term is related to the previous term by either adding or subtracting 1 after doubling -/
def PrimeSequence (p : ℕ → ℕ) : Prop :=
  Prime (p 0) ∧ ∀ i : ℕ, Prime (p (i + 1)) ∧
    ((p (i + 1) = 2 * p i - 1) ∨ (p (i + 1) = 2 * p i + 1))

theorem prime_sequence_finite (p : ℕ → ℕ) (h : PrimeSequence p) :
  ∃ n : ℕ, ∀ m : ℕ, m > n → ¬ Prime (p m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sequence_finite_l1036_103630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midsegment_theorem_l1036_103656

/-- Given a triangle ABC, this function returns the length of its midsegment --/
def midsegment_length (A B C : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

/-- Given a triangle ABC, this function returns the length of its side --/
def side_length (A B C : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

/-- Construct a new triangle from 2/3 of the midsegments of the original triangle --/
def construct_new_triangle (A B C : EuclideanSpace ℝ (Fin 2)) : 
  EuclideanSpace ℝ (Fin 2) × EuclideanSpace ℝ (Fin 2) × EuclideanSpace ℝ (Fin 2) := sorry

theorem midsegment_theorem (A B C : EuclideanSpace ℝ (Fin 2)) :
  let (S, T, U) := construct_new_triangle A B C
  midsegment_length S T U = (1/2) * side_length A B C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midsegment_theorem_l1036_103656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l1036_103646

-- Define the ellipse C
noncomputable def ellipse_C (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * x - 1

-- Define the focal distance
def focal_distance : ℝ := 4

-- Define the eccentricity
noncomputable def eccentricity : ℝ := Real.sqrt 2 / 2

-- Define the point P
def point_P : ℝ × ℝ := (0, -1)

-- Theorem statement
theorem ellipse_and_line_properties :
  -- The equation of the ellipse C
  (∀ x y, ellipse_C x y ↔ x^2 / 8 + y^2 / 4 = 1) ∧
  -- The equation of the line l
  (∃ k, k = 3 * Real.sqrt 10 / 10 ∨ k = -3 * Real.sqrt 10 / 10) ∧
  (∀ k, (k = 3 * Real.sqrt 10 / 10 ∨ k = -3 * Real.sqrt 10 / 10) →
    ∃ A B : ℝ × ℝ,
      -- A and B are on the ellipse
      ellipse_C A.1 A.2 ∧ ellipse_C B.1 B.2 ∧
      -- A and B are on the line l
      line_l k A.1 A.2 ∧ line_l k B.1 B.2 ∧
      -- AP = 2PB
      (A.1 - point_P.1)^2 + (A.2 - point_P.2)^2 = 
        4 * ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l1036_103646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_value_l1036_103690

noncomputable def W (R : ℝ) : ℝ := Real.sqrt 3 / (2 * R)

noncomputable def S (W : ℝ) : ℝ := W + 1 / (W + 1 / (W + 1 / W))

theorem S_value (R : ℝ) (h : R ≠ 0) : S (W R) = 1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_value_l1036_103690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1036_103648

/-- A quadratic function f(x) satisfying specific conditions -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Theorem stating the properties of the quadratic function and the range of t -/
theorem quadratic_function_properties (a b c : ℝ) :
  (f a b c (-2) = 0) →
  (∀ x : ℝ, 2 * x ≤ f a b c x ∧ f a b c x ≤ 1/2 * x^2 + 2) →
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → ∀ t : ℝ, f a b c (x + t) < f a b c (x / 3)) →
  (∀ x : ℝ, f a b c x = 1/4 * x^2 + x + 1) ∧
  (∀ t : ℝ, t ∈ Set.Ioo (-8/3) (-2/3) ↔ 
    (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f a b c (x + t) < f a b c (x / 3))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1036_103648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jim_exercises_l1036_103689

def exercises_for_group (group_number : ℕ) : ℕ :=
  group_number * 6

def total_exercises (total_points : ℕ) : ℕ :=
  (List.range (total_points / 6)).map (λ i => exercises_for_group (i + 1)) |>.sum

theorem jim_exercises : total_exercises 30 = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jim_exercises_l1036_103689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_owe_difference_is_30_l1036_103659

/-- Represents the vacation cost-sharing scenario -/
structure VacationCosts where
  tom_paid : ℚ
  dorothy_paid : ℚ
  sammy_paid : ℚ
  nick_paid : ℚ

/-- Calculates the total amount paid by all four people -/
def total_paid (costs : VacationCosts) : ℚ :=
  costs.tom_paid + costs.dorothy_paid + costs.sammy_paid + costs.nick_paid

/-- Calculates the amount each person should have paid -/
def fair_share (costs : VacationCosts) : ℚ :=
  (total_paid costs) / 4

/-- Theorem: The difference between what Tom owes Nick and what Dorothy owes Nick is 30 -/
theorem owe_difference_is_30 (costs : VacationCosts) 
  (h1 : costs.tom_paid = 150)
  (h2 : costs.dorothy_paid = 180)
  (h3 : costs.sammy_paid = 220)
  (h4 : costs.nick_paid = 250) :
  (fair_share costs - costs.tom_paid) - (fair_share costs - costs.dorothy_paid) = 30 := by
  sorry

#eval (200 : ℚ) - (150 : ℚ) - ((200 : ℚ) - (180 : ℚ))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_owe_difference_is_30_l1036_103659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_comparison_l1036_103627

theorem tan_comparison :
  Real.tan (-13 * Real.pi / 7) > Real.tan (-15 * Real.pi / 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_comparison_l1036_103627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_bound_l1036_103637

-- Define a triangle with two known median lengths
structure Triangle where
  median1 : ℝ
  median2 : ℝ
  median3 : ℝ

-- Define the area function for a triangle given its medians
noncomputable def area (t : Triangle) : ℝ :=
  let s_m := (t.median1 + t.median2 + t.median3) / 2
  (4 / 3) * Real.sqrt (s_m * (s_m - t.median1) * (s_m - t.median2) * (s_m - t.median3))

-- Theorem statement
theorem max_area_bound (t : Triangle) (h1 : t.median1 = 15) (h2 : t.median2 = 9) :
  area t ≤ 86 := by
  sorry

#check max_area_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_bound_l1036_103637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisor_pairs_l1036_103664

theorem count_divisor_pairs (n : ℕ) (h : n = 2 * 3^2 * 101) :
  (Finset.filter (fun p : ℕ × ℕ => p.1 * p.2 = n ∧ 0 < p.1 ∧ 0 < p.2) (Finset.range (n + 1) ×ˢ Finset.range (n + 1))).card = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisor_pairs_l1036_103664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_preserves_R_l1036_103686

/-- Region R in the complex plane -/
noncomputable def R : Set ℂ := {z | -2 ≤ z.re ∧ z.re ≤ 2 ∧ -2 ≤ z.im ∧ z.im ≤ 2}

/-- The transformation function -/
noncomputable def T (z : ℂ) : ℂ := (1/2 + 1/2 * Complex.I) * z

/-- Theorem: The transformation T maps every point in R to a point in R -/
theorem transformation_preserves_R : ∀ z ∈ R, T z ∈ R := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_preserves_R_l1036_103686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_l1036_103606

def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (-3, 5)
def vector_c (x : ℝ) : ℝ × ℝ := (4, x)

theorem vector_equation (x lambda : ℝ) :
  vector_a.1 + vector_b.1 = lambda * (vector_c x).1 ∧
  vector_a.2 + vector_b.2 = lambda * (vector_c x).2 →
  lambda + x = -29/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_l1036_103606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_area_l1036_103625

noncomputable section

/-- Parabola structure -/
structure Parabola where
  eq : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Line structure -/
structure Line where
  eq : ℝ → ℝ → Prop

/-- Point structure -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the problem setup -/
def problem_setup (p : Parabola) (l : Line) (A B G : Point) : Prop :=
  p.eq = (fun x y => y^2 = 4*x) ∧
  p.focus = (1, 0) ∧
  l.eq A.x A.y ∧
  l.eq B.x B.y ∧
  l.eq 1 0 ∧
  A.y > 0 ∧
  (A.x - 1, A.y) = (3*(1 - B.x), -3*B.y) ∧
  G.y = 0

/-- Theorem statement -/
theorem parabola_intersection_area 
  (p : Parabola) (l : Line) (A B G : Point) 
  (h : problem_setup p l A B G) : 
  ∃ (area : ℝ), area = (32 * Real.sqrt 3) / 9 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_area_l1036_103625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1036_103640

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6) + 1

theorem f_monotone_increasing :
  StrictMonoOn f (Set.Ioo (-Real.pi/3) (Real.pi/6)) := by
  sorry

#check f_monotone_increasing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1036_103640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_of_sin_sum_diff_l1036_103635

theorem tan_ratio_of_sin_sum_diff (a b : ℝ) :
  Real.sin (a + b) = 5 / 8 →
  Real.sin (a - b) = 3 / 8 →
  0 < a → a < π / 2 →
  0 < b → b < π / 2 →
  Real.tan a / Real.tan b = 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_of_sin_sum_diff_l1036_103635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_share_price_increase_l1036_103660

/-- Given a share price that increases by 30% in the first quarter and 50% in the second quarter
    (both compared to the beginning of the year), prove that the percent increase from the end of
    the first quarter to the end of the second quarter is (200/13)%. -/
theorem share_price_increase (P : ℝ) (h : P > 0) : 
  let first_quarter := 1.30 * P
  let second_quarter := 1.50 * P
  ((second_quarter - first_quarter) / first_quarter) * 100 = 200 / 13 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_share_price_increase_l1036_103660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_congruent_colored_triangles_l1036_103628

/-- A coloring of the integer lattice points -/
def Coloring (n : ℕ) := ℤ × ℤ → Fin n

/-- A triangle represented by its three vertices -/
structure Triangle :=
  (v1 v2 v3 : ℤ × ℤ)

/-- Two triangles are congruent if they have the same side lengths -/
def congruent (t1 t2 : Triangle) : Prop :=
  let d1 := λ (p q : ℤ × ℤ) => (p.1 - q.1)^2 + (p.2 - q.2)^2
  d1 t1.v1 t1.v2 = d1 t2.v1 t2.v2 ∧
  d1 t1.v2 t1.v3 = d1 t2.v2 t2.v3 ∧
  d1 t1.v3 t1.v1 = d1 t2.v3 t2.v1

/-- A side length of a triangle is divisible by a number -/
def has_side_divisible_by (t : Triangle) (d : ℕ) : Prop :=
  ∃ (i j : Fin 3), 
    let vertices := [t.v1, t.v2, t.v3]
    let (x1, y1) := vertices[i]
    let (x2, y2) := vertices[j]
    (∃ k : ℤ, x2 - x1 = d * k) ∨ (∃ k : ℤ, y2 - y1 = d * k)

/-- The main theorem -/
theorem existence_of_congruent_colored_triangles
  (n a b c : ℕ) :
  ∃ (f : Coloring n) (ts : Fin c → Triangle),
    (∀ i j, i ≠ j → congruent (ts i) (ts j)) ∧
    (∀ i, f (ts i).v1 = f (ts i).v2 ∧ f (ts i).v2 = f (ts i).v3) ∧
    (∀ i, has_side_divisible_by (ts i) a ∧ has_side_divisible_by (ts i) b) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_congruent_colored_triangles_l1036_103628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1036_103626

noncomputable def f (x : ℝ) : ℝ := 1 / (2 - x) + Real.sqrt (9 - x^2)

theorem domain_of_f :
  {x : ℝ | -3 ≤ x ∧ x ≤ 3 ∧ x ≠ 2} = {x : ℝ | ∃ y, f x = y} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1036_103626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_congruent_triangles_area_l1036_103674

-- Define the semicircle
def semicircle_diameter : ℝ := 30

-- Define the points on the diameter
def point_B : ℝ × ℝ := (5, 0)
def point_C : ℝ × ℝ := (15, 0)

-- Define the point F on the semicircle
noncomputable def point_F : ℝ × ℝ := (5, 10 * Real.sqrt 2)

-- Define the congruent right triangles
noncomputable def triangle_BCF_area : ℝ := (1/2) * 10 * (10 * Real.sqrt 2)

-- Define m and n
def m : ℕ := 50
def n : ℕ := 2

-- State the theorem
theorem semicircle_congruent_triangles_area :
  triangle_BCF_area = m * Real.sqrt n ∧ m + n = 52 := by
  sorry

#eval m + n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_congruent_triangles_area_l1036_103674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l1036_103631

theorem remainder_theorem : 
  ∀ (x : Polynomial ℤ),
  ∃ (q : Polynomial ℤ), 
  x^2023 + 1 = (x^10 - x^8 + x^6 - x^4 + x^2 - 1) * q + (x^7 + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l1036_103631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_women_in_business_class_l1036_103696

def total_passengers : ℕ := 300
def women_percentage : ℚ := 70 / 100
def business_class_percentage : ℚ := 15 / 100

theorem women_in_business_class :
  ⌊(total_passengers : ℚ) * women_percentage * business_class_percentage⌋₊ = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_women_in_business_class_l1036_103696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_three_eq_neg_twelve_l1036_103695

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => 
  if x ≥ 0 then x^2 + x else -((-x)^2 + (-x))

-- State the theorem
theorem f_neg_three_eq_neg_twelve :
  (∀ x : ℝ, f (-x) = -f x) → -- f is an odd function
  (∀ x : ℝ, x ≥ 0 → f x = x^2 + x) → -- f(x) = x^2 + x for x ≥ 0
  f (-3) = -12 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_three_eq_neg_twelve_l1036_103695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_perimeter_product_l1036_103679

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculate the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

/-- Calculate the perimeter of a triangle given three points -/
noncomputable def trianglePerimeter (p1 p2 p3 : Point) : ℝ :=
  distance p1 p2 + distance p2 p3 + distance p3 p1

theorem triangle_area_perimeter_product :
  let p := Point.mk 0 1
  let q := Point.mk 3 4
  let r := Point.mk 4 1
  (triangleArea p q r) * (trianglePerimeter p q r) = 1.5 * Real.sqrt 2 + 2 + 0.5 * Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_perimeter_product_l1036_103679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_zero_l1036_103698

/-- Triangle ABC with vertices on coordinate axes -/
structure TriangleABC where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0
  hab : a^2 + b^2 = 36
  hbc : b^2 + c^2 = 64
  hca : c^2 + a^2 = 100

/-- Volume of tetrahedron OABC -/
noncomputable def tetrahedronVolume (t : TriangleABC) : ℝ := (1/6) * t.a * t.b * t.c

/-- Theorem: The volume of tetrahedron OABC is 0 -/
theorem tetrahedron_volume_zero (t : TriangleABC) : tetrahedronVolume t = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_zero_l1036_103698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l1036_103685

open Real

-- Define the triangle ABC
def triangle_ABC (a b c A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  0 < A ∧ A < Real.pi ∧
  0 < B ∧ B < Real.pi ∧
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

-- Convert degrees to radians
noncomputable def deg_to_rad (x : ℝ) : ℝ := x * (Real.pi / 180)

theorem triangle_ABC_properties (a b c A B C : ℝ) :
  triangle_ABC a b c A B C →
  c = Real.sqrt 2 →
  A = deg_to_rad 105 →
  C = deg_to_rad 30 →
  b = 2 ∧ 
  (1/2 * b * c * Real.sin A) = (1 + Real.sqrt 3) / 4 := by
  sorry

#check triangle_ABC_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l1036_103685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_area_approx_l1036_103644

/-- Represents the dimensions and specifications of the sign and letters --/
structure SignSpecs where
  height : ℕ
  width : ℕ
  f_vertical : ℕ
  f_horizontal : ℕ
  o_size : ℕ
  d_vertical : ℕ
  d_radius : ℕ

/-- Calculates the white area of the sign after painting the letters --/
noncomputable def whiteArea (specs : SignSpecs) : ℝ :=
  let totalArea := specs.height * specs.width
  let fArea := specs.f_vertical + 2 * specs.f_horizontal
  let oArea := 2 * (4 * specs.o_size - 4)
  let dArea := specs.d_vertical + 2 * Real.pi * (specs.d_radius^2 : ℝ) / 2
  (totalArea : ℝ) - (fArea + oArea + dArea)

/-- Theorem stating that the white area is approximately 110 square units --/
theorem white_area_approx (specs : SignSpecs) 
  (h1 : specs.height = 8)
  (h2 : specs.width = 20)
  (h3 : specs.f_vertical = 6)
  (h4 : specs.f_horizontal = 4)
  (h5 : specs.o_size = 4)
  (h6 : specs.d_vertical = 6)
  (h7 : specs.d_radius = 2) :
  ∃ ε > 0, |whiteArea specs - 110| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_area_approx_l1036_103644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_returns_after_14_throws_l1036_103675

/-- Represents the number of girls in the circle -/
def n : ℕ := 13

/-- Represents the number of positions the ball advances in each throw -/
def k : ℕ := 5

/-- Calculates the position after a throw, wrapping around if necessary -/
def nextPosition (pos : ℕ) : ℕ :=
  (pos + k - 1) % n + 1

/-- Represents the sequence of positions the ball reaches -/
def ballSequence : List ℕ :=
  let rec loop (pos : ℕ) (acc : List ℕ) (fuel : ℕ) : List ℕ :=
    match fuel with
    | 0 => acc.reverse
    | fuel' + 1 =>
      let newPos := nextPosition pos
      if newPos = 1 then (pos :: acc).reverse
      else loop newPos (pos :: acc) fuel'
  loop 1 [] (n + 1)

theorem ball_returns_after_14_throws :
  ballSequence.length = 14 ∧ ballSequence.head? = some 1 ∧ ballSequence.getLast? = some 1 := by
  sorry

#eval ballSequence
#eval ballSequence.length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_returns_after_14_throws_l1036_103675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sixth_power_sum_l1036_103647

theorem sin_cos_sixth_power_sum (θ : ℝ) (h : Real.cos (2 * θ) = 1/4) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 19/64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sixth_power_sum_l1036_103647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kaylin_age_l1036_103682

/-- Given the ages and relationships of several people, calculate Kaylin's age. -/
theorem kaylin_age (freyja alfred olivia liam lucas eli sarah kaylin : ℝ) 
  (h1 : freyja = 9.5)
  (h2 : alfred = freyja / 2.5)
  (h3 : alfred = 0.2 * olivia)
  (h4 : olivia = 3/4 * liam)
  (h5 : lucas = freyja + 9)
  (h6 : eli = Real.sqrt lucas)
  (h7 : sarah = 2 * eli)
  (h8 : kaylin = sarah - 5) :
  ∃ ε > 0, |kaylin - 3.6| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kaylin_age_l1036_103682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_distance_l1036_103600

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflect a point over the x-axis -/
def reflectOverXAxis (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

/-- The main theorem -/
theorem reflection_distance :
  let p : Point := { x := -4, y := 3 }
  let p' : Point := reflectOverXAxis p
  distance p p' = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_distance_l1036_103600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_disjoint_l1036_103624

/-- The line equation -/
def line (x y : ℝ) : Prop := 3 * x + 4 * y + 11 = 0

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 1

/-- The center of the circle -/
def center : ℝ × ℝ := (1, -1)

/-- The radius of the circle -/
def radius : ℝ := 1

/-- The distance from a point to the line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |3 * x + 4 * y + 11| / Real.sqrt (3^2 + 4^2)

/-- Theorem: The line and circle are disjoint -/
theorem line_circle_disjoint :
  distance_to_line center.1 center.2 > radius := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_disjoint_l1036_103624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1036_103619

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + x - a

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (f a x)

-- Theorem statement
theorem problem_solution (a : ℝ) :
  (∃ x₀ y₀ : ℝ, y₀ = Real.cos (2 * x₀) ∧ g a (g a y₀) = y₀) →
  1 ≤ a ∧ a ≤ Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1036_103619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_three_irrational_irrational_implies_infinite_nonrepeating_decimal_sqrt_three_infinite_nonrepeating_decimal_l1036_103665

noncomputable def HasInfiniteNonrepeatingDecimalExpansion (x : ℝ) : Prop := sorry

theorem sqrt_three_irrational : Irrational (Real.sqrt 3) := by sorry

theorem irrational_implies_infinite_nonrepeating_decimal (x : ℝ) :
  Irrational x → HasInfiniteNonrepeatingDecimalExpansion x := by sorry

theorem sqrt_three_infinite_nonrepeating_decimal :
  HasInfiniteNonrepeatingDecimalExpansion (Real.sqrt 3) := by
  apply irrational_implies_infinite_nonrepeating_decimal
  exact sqrt_three_irrational

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_three_irrational_irrational_implies_infinite_nonrepeating_decimal_sqrt_three_infinite_nonrepeating_decimal_l1036_103665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l1036_103622

/-- Given two plane vectors with specified magnitudes and angle between them, 
    prove that the magnitude of their sum is equal to the given value. -/
theorem vector_sum_magnitude (a b : ℝ × ℝ) : 
  (Real.cos (π / 4 : ℝ) = (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) →
  Real.sqrt (a.1^2 + a.2^2) = 2 →
  Real.sqrt (b.1^2 + b.2^2) = 1 →
  Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = Real.sqrt (5 + 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l1036_103622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l1036_103670

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Determines if two circles are intersecting -/
noncomputable def are_intersecting (c1 c2 : Circle) : Prop :=
  let d := Real.sqrt ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2)
  d < c1.radius + c2.radius ∧ d > |c1.radius - c2.radius|

/-- Converts a circle equation of the form x^2 + y^2 + ax + by + c = 0 to a Circle structure -/
noncomputable def equation_to_circle (a b c : ℝ) : Circle :=
  { center := (-a/2, -b/2),
    radius := Real.sqrt ((a/2)^2 + (b/2)^2 - c) }

theorem circles_intersect :
  let c1 := equation_to_circle (-2) 0 0
  let c2 := equation_to_circle 0 4 0
  are_intersecting c1 c2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l1036_103670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_balls_for_three_same_color_l1036_103667

/-- Given a bag with colored balls, this theorem proves the minimum number of balls
    that need to be drawn to guarantee at least 3 balls of the same color. -/
theorem min_balls_for_three_same_color (total_balls : ℕ) (balls_per_color : ℕ) 
    (h1 : total_balls = 60) (h2 : balls_per_color = 6) :
    ∃ (min_draw : ℕ), 
      (∀ (draw : ℕ), draw ≥ min_draw → 
        ∃ (color : ℕ), color ≤ total_balls / balls_per_color ∧ 
          (∃ (same_color_balls : ℕ), same_color_balls ≥ 3)) ∧
      (∀ (draw : ℕ), draw < min_draw → 
        ∃ (coloring : ℕ → ℕ), ∀ (color : ℕ), 
          (color ≤ total_balls / balls_per_color → 
            ∃! (same_color_balls : ℕ), same_color_balls < 3 ∧
              same_color_balls = (Finset.filter (λ i ↦ coloring i = color) (Finset.range draw)).card)) ∧
      min_draw = 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_balls_for_three_same_color_l1036_103667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_heads_expectation_heads_tails_expectation_l1036_103691

-- Define a fair coin toss
noncomputable def fair_coin_toss : ℝ := 1/2

-- Expected number of flips to get two heads in a row
noncomputable def expected_two_heads : ℝ := 6

-- Expected number of flips to get heads followed by tails
noncomputable def expected_heads_tails : ℝ := 4

-- Theorem for two heads in a row
theorem two_heads_expectation :
  let x := expected_two_heads
  x = fair_coin_toss * (x + 1) + fair_coin_toss * fair_coin_toss * 2 + fair_coin_toss * (1 - fair_coin_toss) * (x + 2) :=
by sorry

-- Theorem for heads followed by tails
theorem heads_tails_expectation :
  let y := expected_heads_tails
  y = fair_coin_toss * (y + 1) + fair_coin_toss * (1 + 1/fair_coin_toss) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_heads_expectation_heads_tails_expectation_l1036_103691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_a_range_l1036_103642

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 - a) * x - a / 2 else Real.log x / Real.log a

def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem f_increasing_a_range :
  ∀ a : ℝ, (increasing_function (f a)) ↔ (4/3 ≤ a ∧ a < 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_a_range_l1036_103642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_baking_problem_l1036_103650

/-- The time it takes Jane to bake a cake alone -/
noncomputable def jane_time : ℝ := 4

/-- The time it takes Roy to bake a cake alone -/
noncomputable def roy_time : ℝ := 5

/-- The time Jane and Roy work together -/
noncomputable def together_time : ℝ := 2

/-- The remaining time for Jane to complete the task alone -/
noncomputable def remaining_time : ℝ := 2/5

theorem cake_baking_problem :
  let jane_rate := 1 / jane_time
  let roy_rate := 1 / roy_time
  let combined_rate := jane_rate + roy_rate
  let completed_portion := combined_rate * together_time
  let remaining_portion := 1 - completed_portion
  remaining_time = remaining_portion / jane_rate :=
by
  sorry

#check cake_baking_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_baking_problem_l1036_103650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abcd_problem_l1036_103609

theorem abcd_problem (a b c d : ℕ+) 
  (h1 : a * b * c * d = Nat.factorial 8)
  (h2 : a * b + a + b = 524)
  (h3 : b * c + b + c = 146)
  (h4 : c * d + c + d = 104) :
  a - d = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abcd_problem_l1036_103609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1036_103688

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  b^2 + c^2 = a^2 + b*c →  -- Given condition
  Real.cos B = Real.sqrt 6 / 3 →  -- Given condition
  b = 2 →  -- Given condition
  0 < A ∧ A < Real.pi →  -- Assumption for angle A
  0 < B ∧ B < Real.pi →  -- Assumption for angle B
  0 < C ∧ C < Real.pi →  -- Assumption for angle C
  (A = Real.pi / 3 ∧  -- Conclusion 1: Measure of angle A
   (1/2 : ℝ) * b * c * Real.sin A = (3 * Real.sqrt 2 + Real.sqrt 3) / 2)  -- Conclusion 2: Area of triangle
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1036_103688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_under_arccos_cos_eq_pi_squared_l1036_103639

/-- The area bounded by y = arccos(cos x) and the x-axis on the interval [0, 2π] -/
noncomputable def area_under_arccos_cos : ℝ :=
  ∫ x in (0)..(2 * Real.pi), Real.arccos (Real.cos x)

/-- Theorem stating that the area under y = arccos(cos x) from 0 to 2π is π² -/
theorem area_under_arccos_cos_eq_pi_squared :
  area_under_arccos_cos = Real.pi ^ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_under_arccos_cos_eq_pi_squared_l1036_103639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_motion_properties_l1036_103612

/-- Motion described by x(t) = A sin(ωt) -/
noncomputable def x (A ω t : ℝ) : ℝ := A * Real.sin (ω * t)

/-- Velocity of the motion -/
noncomputable def v (A ω t : ℝ) : ℝ := A * ω * Real.cos (ω * t)

/-- Acceleration of the motion -/
noncomputable def a (A ω t : ℝ) : ℝ := -A * ω^2 * Real.sin (ω * t)

theorem motion_properties (A ω : ℝ) (h₁ : A ≠ 0) (h₂ : ω > 0) :
  let t₀ := 2 * Real.pi / ω
  (∀ t, v A ω t = deriv (x A ω) t) ∧
  (∀ t, a A ω t = deriv (v A ω) t) ∧
  v A ω t₀ = A * ω ∧
  a A ω t₀ = 0 ∧
  ∃ k, ∀ t, a A ω t = k * x A ω t :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_motion_properties_l1036_103612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l1036_103613

noncomputable def f (x : ℝ) : ℝ := Real.exp x

noncomputable def A : ℝ × ℝ := (0, f 0)
noncomputable def B : ℝ × ℝ := (2, f 2)

noncomputable def C : ℝ × ℝ := (2/3 * A.1 + 1/3 * B.1, 2/3 * A.2 + 1/3 * B.2)
noncomputable def D : ℝ × ℝ := (1/3 * A.1 + 2/3 * B.1, 1/3 * A.2 + 2/3 * B.2)

noncomputable def x₃ : ℝ := Real.log (2/3 + 1/3 * Real.exp 2)
noncomputable def x₄ : ℝ := Real.log (1/3 + 2/3 * Real.exp 2)

theorem intersection_points :
  f x₃ = C.2 ∧ f x₄ = D.2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l1036_103613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l1036_103680

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The slope of the line 2x - y = 0 -/
noncomputable def m₁ : ℝ := 2

/-- The slope of the line ax - 2y - 1 = 0 -/
noncomputable def m₂ (a : ℝ) : ℝ := a / 2

/-- Theorem: If two lines 2x - y = 0 and ax - 2y - 1 = 0 are perpendicular, then a = -1 -/
theorem perpendicular_lines (a : ℝ) : perpendicular m₁ (m₂ a) → a = -1 := by
  intro h
  unfold perpendicular at h
  unfold m₁ at h
  unfold m₂ at h
  -- The rest of the proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l1036_103680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_implies_y_l1036_103618

noncomputable def v (y : ℝ) : ℝ × ℝ := (2, y)
def w : ℝ × ℝ := (5, -1)

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_prod := u.1 * v.1 + u.2 * v.2
  let norm_sq := v.1 * v.1 + v.2 * v.2
  ((dot_prod / norm_sq) * v.1, (dot_prod / norm_sq) * v.2)

theorem projection_implies_y (y : ℝ) :
  proj w (v y) = (3, -0.6) → y = -5.6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_implies_y_l1036_103618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_l1036_103657

/-- The present age of Ramesh -/
def R : ℚ := sorry

/-- The present age of Mahesh -/
def M : ℚ := sorry

/-- The ratio of Ramesh's age to Mahesh's age -/
def present_ratio : ℚ := 2 / 5

/-- The ratio of their ages after 10 years -/
def future_ratio : ℚ := 10 / 15

theorem age_difference : 
  R / M = present_ratio ∧ 
  (R + 10) / (M + 10) = future_ratio → 
  M - R = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_l1036_103657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1036_103614

noncomputable def f (x : ℝ) := 2 * Real.cos x * (Real.sin x + Real.cos x) - 1

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  (let a := -π/6
   let b := -π/12
   ∃ (m M : ℝ), (∀ (x : ℝ), a ≤ x → x ≤ b → m ≤ f x ∧ f x ≤ M) ∧
     (∃ (x1 x2 : ℝ), a ≤ x1 ∧ x1 ≤ b ∧ a ≤ x2 ∧ x2 ≤ b ∧ f x1 = m ∧ f x2 = M) ∧
     m + M = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1036_103614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_eccentricity_l1036_103604

-- Define the conic section
def conic_section (x y : ℝ) : Prop :=
  ∃ (m n : ℝ), m * x^2 + n * y^2 = 1

-- Define the points through which the conic section passes
noncomputable def point_A : ℝ × ℝ := (-2, 2 * Real.sqrt 3)
def point_B : ℝ × ℝ := (1, -3)

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt ((a^2 + b^2) / a^2)

theorem conic_section_eccentricity :
  ∃ (m n : ℝ),
    conic_section point_A.1 point_A.2 ∧
    conic_section point_B.1 point_B.2 ∧
    (m < 0 ∧ n > 0) ∧  -- Ensures it's a hyperbola
    eccentricity (Real.sqrt (1/m)) (Real.sqrt (1/n)) = Real.sqrt 2 :=
by
  sorry

#eval "Theorem statement compiled successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_eccentricity_l1036_103604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_set_operations_l1036_103602

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 5*x + 6) / Real.log 10
noncomputable def g (x : ℝ) : ℝ := Real.sqrt ((4/x) - 1)

-- Define the domains A and B
def A : Set ℝ := {x | x > 3 ∨ x < 2}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 4}

-- Theorem statement
theorem domain_and_set_operations :
  (∀ x, x ∈ A ↔ (x > 3 ∨ x < 2)) ∧
  (∀ x, x ∈ B ↔ (0 < x ∧ x ≤ 4)) ∧
  (A ∪ B = Set.univ) ∧
  (A ∩ B = {x | (0 < x ∧ x < 2) ∨ (3 < x ∧ x ≤ 4)}) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_set_operations_l1036_103602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cross_section_area_l1036_103683

variable (a b c : ℝ)

-- Define the parallelepiped with edges a ≤ b ≤ c
def is_parallelepiped (a b c : ℝ) : Prop := 0 < a ∧ a ≤ b ∧ b ≤ c

-- Define the area of a cross-section
noncomputable def cross_section_area (a b c : ℝ) : ℝ → ℝ → ℝ := 
  λ x y ↦ x * Real.sqrt (y^2 + (a^2 + b^2 + c^2 - x^2 - y^2))

-- Theorem statement
theorem max_cross_section_area 
  (h : is_parallelepiped a b c) :
  ∃ (S : ℝ), S = c * Real.sqrt (a^2 + b^2) ∧ 
  ∀ (x y : ℝ), cross_section_area a b c x y ≤ S :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cross_section_area_l1036_103683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_Z_implies_m_eq_neg_two_l1036_103608

-- Define the complex number Z
noncomputable def Z (m : ℝ) : ℂ := (m + 2*Complex.I) / (1 + Complex.I)

-- Define what it means for a complex number to be imaginary
def is_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- Theorem statement
theorem imaginary_Z_implies_m_eq_neg_two :
  ∀ m : ℝ, is_imaginary (Z m) → m = -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_Z_implies_m_eq_neg_two_l1036_103608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_centroid_trajectory_min_MN_value_l1036_103611

noncomputable section

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = 2*x

-- Define the focus F
def focus : ℝ × ℝ := (1/2, 0)

-- Define the directrix
def directrix (x : ℝ) : Prop := x = -1/2

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 2

-- Theorem 1: Equation of parabola C
theorem parabola_equation :
  ∀ x y : ℝ, parabola_C x y ↔ y^2 = 2*x :=
by
  sorry

-- Theorem 2: Trajectory of centroid G
theorem centroid_trajectory :
  ∃ f : ℝ → ℝ, ∀ x y : ℝ,
    (∃ a b : ℝ × ℝ, parabola_C a.1 a.2 ∧ parabola_C b.1 b.2 ∧
      (∃ k : ℝ, (a.2 - focus.2) = k * (a.1 - focus.1) ∧
                (b.2 - focus.2) = k * (b.1 - focus.1)) ∧
      x = (a.1 + b.1) / 3 ∧ y = (a.2 + b.2) / 3) →
    y^2 = (2/3) * x - 2/9 :=
by
  sorry

-- Theorem 3: Minimum value of |MN|
theorem min_MN_value :
  ∃ min_val : ℝ, min_val = 2 * Real.sqrt 30 / 5 ∧
    (∀ x y : ℝ, parabola_C x y →
      ∀ m n : ℝ × ℝ, circle_eq m.1 m.2 ∧ circle_eq n.1 n.2 →
        (m.1 - x)^2 + (m.2 - y)^2 = (n.1 - x)^2 + (n.2 - y)^2 →
          (m.1 - n.1)^2 + (m.2 - n.2)^2 ≥ min_val^2) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_centroid_trajectory_min_MN_value_l1036_103611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_solutions_l1036_103645

theorem exactly_two_solutions (c : ℝ) :
  (∃! (s : Set (ℝ × ℝ)), s.Finite ∧ s.ncard = 2 ∧
    ∀ (x y : ℝ), (x, y) ∈ s ↔ (|x + y| = 99 ∧ |x - y| = c)) ↔
  c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_solutions_l1036_103645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_solid_volume_is_five_sixths_l1036_103699

/-- Represents a cube with unit edge length -/
structure UnitCube where
  vertices : Fin 8 → ℝ × ℝ × ℝ
  is_unit : ∀ (i j : Fin 8), i ≠ j → ‖vertices i - vertices j‖ = 1 ∨ ‖vertices i - vertices j‖ = Real.sqrt 2 ∨ ‖vertices i - vertices j‖ = Real.sqrt 3

/-- Represents a plane passing through two opposite vertices and the midpoint of an adjacent edge -/
structure CuttingPlane (cube : UnitCube) where
  vertex1 : Fin 8
  vertex2 : Fin 8
  midpoint : ℝ × ℝ × ℝ
  is_opposite : ‖cube.vertices vertex1 - cube.vertices vertex2‖ = Real.sqrt 3
  is_midpoint : ∃ (i j : Fin 8), i ≠ j ∧ midpoint = (cube.vertices i + cube.vertices j) / 2

/-- The volume of the larger solid resulting from cutting a unit cube -/
noncomputable def larger_solid_volume (cube : UnitCube) (plane : CuttingPlane cube) : ℝ := 
  sorry

/-- Theorem stating that the volume of the larger solid is 5/6 -/
theorem larger_solid_volume_is_five_sixths (cube : UnitCube) (plane : CuttingPlane cube) :
  larger_solid_volume cube plane = 5/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_solid_volume_is_five_sixths_l1036_103699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_in_third_quadrant_l1036_103651

noncomputable section

-- Define the point A
def A : ℝ × ℝ := (Real.cos (2023 * Real.pi / 180), Real.tan 8)

-- Define the quadrants
def first_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0
def second_quadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 > 0
def third_quadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0
def fourth_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

-- Theorem stating that point A is in the third quadrant
theorem A_in_third_quadrant : third_quadrant A := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_in_third_quadrant_l1036_103651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_in_binomial_expansion_l1036_103607

open Polynomial

theorem coefficient_x_in_binomial_expansion :
  let n : ℕ := 4
  let a : ℚ := 1
  let b : ℚ := -2
  let expansion := (C a + C b * X : ℚ[X])^n
  expansion.coeff 1 = -8 :=
by
  -- Introduce the variables
  intro n a b expansion
  -- Unfold the definitions
  simp [expansion]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_in_binomial_expansion_l1036_103607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_grade_homework_problem_l1036_103636

theorem sixth_grade_homework_problem (A : ℚ) : 
  (0.20 * A + 0.12 * 50 = 0.15 * (A + 50)) → A = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_grade_homework_problem_l1036_103636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1036_103653

def A (a : ℝ) : Set ℝ := {a + 2, (a + 1)^2, a^2 + 3*a + 3}

theorem problem_solution (a : ℝ) (h : 1 ∈ A a) : (2015 : ℝ)^a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1036_103653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_condition_obtuse_condition_l1036_103633

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)

-- Define what it means for a triangle to be equilateral
def isEquilateral (t : Triangle) : Prop :=
  t.A = t.B ∧ t.B = t.C

-- Define what it means for a triangle to be obtuse
def isObtuse (t : Triangle) : Prop :=
  t.A > Real.pi/2 ∨ t.B > Real.pi/2 ∨ t.C > Real.pi/2

-- Theorem 1
theorem equilateral_condition (t : Triangle) :
  Real.cos (t.A - t.B) * Real.cos (t.B - t.C) * Real.cos (t.C - t.A) = 1 →
  isEquilateral t := by
  sorry

-- Theorem 2
theorem obtuse_condition (t : Triangle) :
  Real.cos t.A * Real.cos t.B * Real.cos t.C < 0 →
  isObtuse t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_condition_obtuse_condition_l1036_103633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_calculation_l1036_103668

noncomputable def diamond (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

theorem diamond_calculation : 
  diamond (diamond 7 24) (diamond (-24) (-7)) = 25 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_calculation_l1036_103668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pastry_distribution_theorem_l1036_103671

/-- Represents a multiset of positive integers -/
def PastryBoxes := Multiset ℕ+

/-- Represents the process of removing pastries and creating trays -/
def createTrays (boxes : PastryBoxes) : Multiset ℕ+ :=
  sorry

/-- The theorem stating that the number of distinct elements in the original multiset
    is equal to the number of distinct elements in the transformed multiset -/
theorem pastry_distribution_theorem (boxes : PastryBoxes) :
  (Finset.card (Multiset.toFinset boxes)) = (Finset.card (Multiset.toFinset (createTrays boxes))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pastry_distribution_theorem_l1036_103671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_l1036_103693

-- Define Ket and Dob as structures
structure Ket : Type
structure Dob : Type

-- Define S as the sum type of Ket and Dob
def S : Type := Ket ⊕ Dob

-- Assume Ket and Dob are finite types
variable [Fintype Ket] [Fintype Dob]

-- Define membership for Dob in Ket
def member (d : Dob) (k : Ket) : Prop := sorry

instance : Membership Dob Ket where
  mem := member

-- Axioms
axiom P1 : ∀ k : Ket, ∃ D : Finset Dob, ∀ d : Dob, d ∈ D ↔ d ∈ k

axiom P2 : ∀ k1 k2 k3 : Ket, k1 ≠ k2 ∧ k2 ≠ k3 ∧ k1 ≠ k3 →
  ∃! d : Dob, d ∈ k1 ∧ d ∈ k2 ∧ d ∈ k3

axiom P3 : ∀ d : Dob, ∃! K : Finset Ket, K.card = 3 ∧ ∀ k : Ket, k ∈ K ↔ d ∈ k

axiom P4 : Fintype.card Ket = 5

-- Theorems
def T1 : Prop := Fintype.card Dob = 10

def T3 : Prop := ∀ d : Dob, ∃! d' : Dob, d ≠ d' ∧ ∀ k : Ket, d ∈ k → d' ∉ k

-- Main theorem
theorem main : T1 ∧ T3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_l1036_103693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jump_step_difference_l1036_103601

/-- Represents the number of telephone poles -/
def num_poles : ℕ := 31

/-- Represents the total distance in feet between the first and last pole -/
def total_distance : ℝ := 5280

/-- Represents the number of steps Cleo takes between consecutive poles -/
def cleo_steps_per_gap : ℕ := 36

/-- Represents the number of jumps Zoe takes between consecutive poles -/
def zoe_jumps_per_gap : ℕ := 9

/-- Calculates the length of Cleo's step in feet -/
noncomputable def cleo_step_length : ℝ :=
  total_distance / ((num_poles - 1) * cleo_steps_per_gap)

/-- Calculates the length of Zoe's jump in feet -/
noncomputable def zoe_jump_length : ℝ :=
  total_distance / ((num_poles - 1) * zoe_jumps_per_gap)

/-- Theorem stating the difference between Zoe's jump and Cleo's step length -/
theorem jump_step_difference : 
  ∃ ε > 0, abs (zoe_jump_length - cleo_step_length - 14.7) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jump_step_difference_l1036_103601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l1036_103672

-- Define the real number a
variable (a : ℝ)

-- Define proposition p
def p : Prop := ∀ x : ℝ, ∀ y : ℝ, x < y → (a - 3/2) ^ x > (a - 3/2) ^ y

-- Define proposition q
def q : Prop := ∀ x : ℝ, (1/2 : ℝ) ^ |x - 1| < a

-- State the theorem
theorem a_range :
  (¬(p a ∧ q a) ∧ (p a ∨ q a)) → ((1 < a ∧ a ≤ 3/2) ∨ (a ≥ 5/2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l1036_103672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_floor_area_correct_l1036_103677

-- Define the radius of the larger circle
noncomputable def R : ℝ := 24

-- Define the number of smaller circles
def n : ℕ := 6

-- Define the radius of each smaller circle
noncomputable def r : ℝ := R / 3

-- Theorem statement
theorem area_between_circles :
  π * R^2 - n * π * r^2 = 192 * π := by
  sorry

-- The floor of the area
def area_floor : ℕ := 602

-- Theorem for the floor of the area
theorem floor_area_correct :
  ⌊π * R^2 - n * π * r^2⌋ = area_floor := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_floor_area_correct_l1036_103677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_reciprocal_sum_l1036_103676

/-- Two circles with one common tangent -/
structure TangentCircles where
  a : ℝ
  b : ℝ
  hab : a * b ≠ 0
  h_tangent : (4 * a^2 + b^2) = 1

/-- The sum of reciprocals of squared parameters -/
noncomputable def reciprocal_sum (tc : TangentCircles) : ℝ :=
  1 / tc.a^2 + 1 / tc.b^2

/-- The minimum value of the reciprocal sum is 9 -/
theorem min_reciprocal_sum :
  ∃ (min : ℝ), min = 9 ∧ ∀ (tc : TangentCircles), reciprocal_sum tc ≥ min := by
  sorry

#check min_reciprocal_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_reciprocal_sum_l1036_103676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l1036_103643

noncomputable def m (x : ℝ) : ℝ × ℝ := (2 * Real.cos x + 2 * Real.sqrt 3 * Real.sin x, 1)

noncomputable def n (x y : ℝ) : ℝ × ℝ := (Real.cos x, -y)

noncomputable def f (x : ℝ) : ℝ := 1 + 2 * Real.sin (2 * x + Real.pi / 6)

theorem triangle_area_proof 
  (x : ℝ) 
  (h1 : m x • n x (f x) = 0) 
  (h2 : f (Real.pi / 6) = 3) 
  (h3 : ∃ (a b c : ℝ), a = 2 ∧ b + c = 4 ∧ 
    (∃ (A B C : ℝ), A + B + C = Real.pi ∧ 
      a / Real.sin A = b / Real.sin B ∧ 
      b / Real.sin B = c / Real.sin C)) :
  ∃ (S : ℝ), S = Real.sqrt 3 ∧ 
    (∃ (a b c A B C : ℝ), 
      S = 1/2 * a * b * Real.sin C ∧
      a = 2 ∧ b + c = 4 ∧ 
      A + B + C = Real.pi ∧
      a / Real.sin A = b / Real.sin B ∧
      b / Real.sin B = c / Real.sin C) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l1036_103643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_quotient_conditions_l1036_103605

/-- Complex number representation -/
structure MyComplex where
  re : ℝ
  im : ℝ

/-- The imaginary unit i -/
def i : MyComplex := ⟨0, 1⟩

/-- Addition of complex numbers -/
def MyComplex.add (z w : MyComplex) : MyComplex :=
  ⟨z.re + w.re, z.im + w.im⟩

/-- Multiplication of complex numbers -/
def MyComplex.mul (z w : MyComplex) : MyComplex :=
  ⟨z.re * w.re - z.im * w.im, z.re * w.im + z.im * w.re⟩

/-- Division of complex numbers -/
noncomputable def MyComplex.div (z w : MyComplex) : MyComplex :=
  let denom := w.re * w.re + w.im * w.im
  ⟨(z.re * w.re + z.im * w.im) / denom, (z.im * w.re - z.re * w.im) / denom⟩

/-- A complex number is real if its imaginary part is zero -/
def MyComplex.isReal (z : MyComplex) : Prop := z.im = 0

/-- A complex number is purely imaginary if its real part is zero -/
def MyComplex.isPurelyImaginary (z : MyComplex) : Prop := z.re = 0

theorem complex_quotient_conditions
  (z₁ z₂ : MyComplex)
  (h : z₂ ≠ ⟨0, 0⟩) :
  (MyComplex.isReal (MyComplex.div z₁ z₂) ↔ z₁.re * z₂.im = z₁.im * z₂.re) ∧
  (MyComplex.isPurelyImaginary (MyComplex.div z₁ z₂) ↔ z₁.re * z₂.re = -z₁.im * z₂.im) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_quotient_conditions_l1036_103605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_irreducibility_l1036_103641

/-- Given n ≥ 10 and n pairwise distinct integers, the polynomial
    P(X) = (X - a₁)...(X - aₙ) + 1 is irreducible in ℤ[X]. -/
theorem polynomial_irreducibility (n : ℕ) (a : Fin n → ℤ) 
    (h_n : n ≥ 10) (h_distinct : ∀ i j : Fin n, i ≠ j → a i ≠ a j) :
  Irreducible (((Polynomial.X : Polynomial ℤ) - (Finset.univ.prod (λ i : Fin n ↦ (Polynomial.X - Polynomial.C (a i))))) + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_irreducibility_l1036_103641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_primary_objective_is_economies_of_scale_l1036_103678

/-- Represents a footwear enterprise -/
structure FootwearEnterprise where
  name : String

/-- Represents a location with a footwear industry -/
structure IndustrialLocation where
  name : String
  has_mature_industry : Bool
  has_complete_chain : Bool

/-- Represents the objective of a relocation -/
inductive RelocationObjective
  | ImproveQuality
  | IncreaseQuantity
  | AchieveEconomiesOfScale
  | EnhanceReputation
deriving Repr

/-- Defines the group relocation of footwear enterprises -/
def group_relocation (enterprises : List FootwearEnterprise) (location : IndustrialLocation) : Prop :=
  location.has_mature_industry ∧ location.has_complete_chain

/-- Helper function to determine if an objective is the primary one -/
def is_primary_objective (obj : RelocationObjective) : Bool :=
  match obj with
  | RelocationObjective.AchieveEconomiesOfScale => true
  | _ => false

/-- States that the primary objective of group relocation is to achieve economies of scale -/
theorem primary_objective_is_economies_of_scale 
  (enterprises : List FootwearEnterprise) 
  (location : IndustrialLocation) :
  group_relocation enterprises location →
  is_primary_objective RelocationObjective.AchieveEconomiesOfScale = true :=
by
  intro h
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_primary_objective_is_economies_of_scale_l1036_103678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_sum_l1036_103662

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 = -4*x - 8*y + 8

/-- The center of a circle given by its equation -/
def CircleCenter (x y : ℝ) : Prop :=
  CircleEquation x y ∧ ∀ (a b : ℝ), CircleEquation a b → (x - a)^2 + (y - b)^2 ≤ (x - a)^2 + (y - b)^2

theorem circle_center_sum :
  ∀ (x y : ℝ), CircleCenter x y → x + y = -6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_sum_l1036_103662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_convex_to_convex_improvement_l1036_103610

/-- A figure in a 2D space. -/
class Figure (α : Type*) [MeasurableSpace α] where
  /-- The area of the figure. -/
  area : α → ℝ
  /-- The perimeter of the figure. -/
  perimeter : α → ℝ
  /-- Predicate indicating whether the figure is convex. -/
  is_convex : α → Prop

/-- Theorem stating that for any non-convex figure, there exists a convex figure
    with a smaller perimeter and larger area. -/
theorem non_convex_to_convex_improvement {α : Type*} [MeasurableSpace α] [Figure α] 
  (Ψ : α) (h : ¬Figure.is_convex Ψ) : 
  ∃ (Φ : α), Figure.is_convex Φ ∧ 
             Figure.area Φ ≥ Figure.area Ψ ∧ 
             Figure.perimeter Φ ≤ Figure.perimeter Ψ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_convex_to_convex_improvement_l1036_103610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_diagonals_not_bisect_l1036_103684

/-- The proposition "The diagonals of a trapezoid bisect each other" -/
def p : Prop := sorry

/-- Theorem stating that "The diagonals of a trapezoid do not bisect each other" 
    is logically equivalent to "not p" -/
theorem trapezoid_diagonals_not_bisect : 
  (¬p) ↔ (¬p) := by
  sorry

#check trapezoid_diagonals_not_bisect

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_diagonals_not_bisect_l1036_103684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_2023_minus_one_power_zero_plus_half_power_minus_one_equals_three_l1036_103632

theorem sqrt_2023_minus_one_power_zero_plus_half_power_minus_one_equals_three :
  (Real.sqrt 2023 - 1) ^ 0 + (1 / 2) ^ (-1 : ℤ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_2023_minus_one_power_zero_plus_half_power_minus_one_equals_three_l1036_103632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discriminant_of_equation_l1036_103658

noncomputable def quadratic_equation (x : ℝ) : ℝ := 2 * x^2 + (2 + 1/2) * x + 1/2

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem discriminant_of_equation :
  discriminant 2 (2 + 1/2) (1/2) = 9/4 := by
  -- Expand the definition of discriminant
  unfold discriminant
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_discriminant_of_equation_l1036_103658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_and_b_l1036_103692

-- Define the curve
noncomputable def curve (x a b : ℝ) : ℝ := x * Real.exp x - a * Real.exp x - b * x

-- Define the tangent line
def tangent_line (x : ℝ) : ℝ := x - 1

-- Theorem statement
theorem tangent_line_implies_a_and_b :
  (curve 0 a b = tangent_line 0) →
  ((deriv (fun x ↦ curve x a b)) 0 = (deriv tangent_line) 0) →
  a = 1 ∧ b = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_and_b_l1036_103692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1036_103621

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The focal length of a hyperbola -/
noncomputable def focal_length (h : Hyperbola) : ℝ := Real.sqrt (h.a^2 + h.b^2)

/-- The slope of an asymptote of a hyperbola -/
noncomputable def asymptote_slope (h : Hyperbola) : ℝ := h.b / h.a

theorem hyperbola_equation (h : Hyperbola) 
  (h_focal : focal_length h = 10)
  (h_asymptote : asymptote_slope h = 1/2) :
  h.a^2 = 20 ∧ h.b^2 = 5 := by
  sorry

#check hyperbola_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1036_103621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_range_three_tangent_lines_l1036_103638

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x + 1) * Real.log x - x + 1

-- Define the range of a for which f is increasing
def increasing_range : Set ℝ := { a | a ≥ 1/2 }

-- Define the conditions for the existence of three tangent lines
def tangent_conditions (a : ℝ) : Prop := 0 < a ∧ a < 1/2 ∧ a ≠ 1/Real.exp 1

-- Theorem for the increasing range of f
theorem f_increasing_range (a : ℝ) :
  (∀ x > 0, Monotone (f a)) ↔ a ∈ increasing_range := by
  sorry

-- Theorem for the existence of three tangent lines
theorem three_tangent_lines (a : ℝ) (h : tangent_conditions a) :
  ∃ l₁ l₂ l₃ : ℝ → ℝ,
    (∀ x, l₁ x = Real.log x ∨ l₁ x = a * (x - 1/x)) ∧
    (∀ x, l₂ x = Real.log x ∨ l₂ x = a * (x - 1/x)) ∧
    (∀ x, l₃ x = Real.log x ∨ l₃ x = a * (x - 1/x)) ∧
    l₁ ≠ l₂ ∧ l₁ ≠ l₃ ∧ l₂ ≠ l₃ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_range_three_tangent_lines_l1036_103638
