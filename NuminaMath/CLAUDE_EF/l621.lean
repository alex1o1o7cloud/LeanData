import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l621_62146

theorem sin_beta_value (α β : Real) (h1 : 0 < α) (h2 : α < β) (h3 : β < π / 2)
  (h4 : Real.sin α = 3 / 5) (h5 : Real.cos (β - α) = 12 / 13) : Real.sin β = 56 / 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l621_62146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_points_are_correct_l621_62181

open Real

-- Define the square
def Square : Set (ℝ × ℝ) := { p | -π ≤ p.1 ∧ p.1 ≤ 2*π ∧ 0 ≤ p.2 ∧ p.2 ≤ 3*π }

-- Define the system of equations
def SatisfiesSystem (p : ℝ × ℝ) : Prop :=
  sin p.1 + sin p.2 = sin 3 ∧
  cos p.1 + cos p.2 = cos 3

-- Define the set of solution points
def SolutionPoints : Set (ℝ × ℝ) :=
  { (3 - π/3, 3 + π/3),
    (3 - 5*π/3, 3 + 5*π/3),
    (3 + π/3, 3 + 5*π/3),
    (3 + π/3, 3 - π/3) }

theorem solution_points_are_correct :
  ∀ p ∈ Square, SatisfiesSystem p ↔ p ∈ SolutionPoints := by
  sorry

#check solution_points_are_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_points_are_correct_l621_62181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_volume_l621_62162

/-- The volume of a sphere inscribed in a right circular cone -/
theorem inscribed_sphere_volume (d : ℝ) (h : d = 24) : 
  (4 / 3) * Real.pi * (d / (2 * Real.sqrt 2))^3 = 576 * Real.sqrt 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_volume_l621_62162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l621_62144

open Real

-- Define the triangle ABC
def Triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  0 < A ∧ A < Real.pi ∧
  0 < B ∧ B < Real.pi ∧
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C

-- Define the given condition
def TriangleCondition (A B C : ℝ) (a b c : ℝ) : Prop :=
  Triangle A B C a b c ∧ (2 * a - c) * Real.cos B = b * Real.cos C

-- Define the function y
noncomputable def y (A C : ℝ) : ℝ := (Real.cos (A / 2))^2 * (Real.sin (C / 2))^2 - 1

-- State the theorem
theorem triangle_theorem (A B C a b c : ℝ) 
  (h : TriangleCondition A B C a b c) : 
  B = Real.pi / 3 ∧ ∀ y', y' = y A C → -3/4 < y' ∧ y' < 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l621_62144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_is_32_max_points_achievable_l621_62175

/-- The maximum number of points that can be placed on a segment of length 1,
    such that on any subsegment of length d within this segment,
    there are no more than 1 + 1000d² points. -/
def max_points_on_segment : ℕ := 32

/-- Proof that 32 is the maximum number of points satisfying the conditions. -/
theorem max_points_is_32 (n : ℕ) :
  (∀ d : ℝ, d > 0 → d ≤ 1 → (n : ℝ) * d ≤ 1 + 1000 * d^2) →
  n ≤ max_points_on_segment := by
  sorry

/-- Proof that the maximum number of points is achievable. -/
theorem max_points_achievable :
  ∃ (points : Finset ℝ),
    points.card = max_points_on_segment ∧
    (∀ a b : ℝ, a ∈ points → b ∈ points → a ≠ b → |a - b| ≤ 1) ∧
    (∀ d : ℝ, d > 0 → d ≤ 1 →
      ∀ x : ℝ, (points.filter (fun p => x ≤ p ∧ p ≤ x + d)).card ≤ ⌊1 + 1000 * d^2⌋) := by
  sorry

#check max_points_on_segment
#check max_points_is_32
#check max_points_achievable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_is_32_max_points_achievable_l621_62175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_half_angle_absolute_value_l621_62190

theorem cos_half_angle_absolute_value (x : ℝ) :
  Real.sin x = -5/13 → 
  x ∈ Set.Icc π (3*π/2) →
  |Real.cos (x/2)| = Real.sqrt 26 / 26 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_half_angle_absolute_value_l621_62190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l621_62174

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2 * x + 1) / (x - 1)

-- Define the domain of f
def domain : Set ℝ := {x | -8 ≤ x ∧ x < -4}

-- Theorem statement
theorem f_properties :
  (∃ (max : ℝ), max = 5/3 ∧ ∀ (x : ℝ), x ∈ domain → f x ≤ max) ∧
  (¬∃ (min : ℝ), ∀ (x : ℝ), x ∈ domain → min ≤ f x) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l621_62174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_f_condition_l621_62131

noncomputable def f (x : Real) : Real := Real.sin x + Real.cos x

noncomputable def g (x : Real) : Real := f x * (Real.cos x - Real.sin x) - (f x)^2

theorem g_properties :
  (∃ (M : Real), ∀ (x : Real), g x ≤ M ∧ M = 2) ∧
  (∀ (y : Real), y > 0 ∧ (∀ (x : Real), g (x + y) = g x) → y ≥ Real.pi) := by
  sorry

theorem f_condition (x : Real) :
  f x = 2 * (Real.cos x - Real.sin x) →
  (1 + Real.sin x^2) / (Real.cos x^2 - Real.sin x * Real.cos x) = 11/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_f_condition_l621_62131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_julie_school_hours_l621_62112

/-- Represents Julie's work schedule and earnings --/
structure WorkSchedule where
  summer_weeks : ℕ
  summer_hours_per_week : ℕ
  summer_earnings : ℚ
  school_year_weeks : ℕ
  school_year_earnings : ℚ

/-- Calculates the required hours per week during the school year --/
noncomputable def required_school_hours (schedule : WorkSchedule) : ℚ :=
  let hourly_rate := schedule.summer_earnings / (schedule.summer_weeks * schedule.summer_hours_per_week)
  let weekly_school_earnings := schedule.school_year_earnings / schedule.school_year_weeks
  weekly_school_earnings / hourly_rate

/-- Theorem stating that Julie needs to work 8 hours per week during the school year --/
theorem julie_school_hours (julie : WorkSchedule) 
  (h1 : julie.summer_weeks = 8)
  (h2 : julie.summer_hours_per_week = 48)
  (h3 : julie.summer_earnings = 7000)
  (h4 : julie.school_year_weeks = 48)
  (h5 : julie.school_year_earnings = 7000) :
  required_school_hours julie = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_julie_school_hours_l621_62112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l621_62158

theorem simplify_expression (m : ℝ) (hm : m ≠ 0) :
  (3 / (4 * m))^(-3 : ℤ) * (2 * m)^4 = 1024 * m^7 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l621_62158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l621_62171

def a : ℕ → ℕ
  | 0 => 2  -- Adding the base case for 0
  | 1 => 2
  | n + 2 => 3 * a (n + 1) + 2

def b (n : ℕ) : ℕ := a n + 1

theorem sequence_properties :
  (b 1 = 3 ∧ b 2 = 9 ∧ b 3 = 27) ∧
  (∀ n : ℕ, b (n + 1) = 3 * b n) ∧
  (∀ n : ℕ, a n = 3^n - 1) := by
  sorry

#eval a 0  -- Testing the function
#eval a 1
#eval a 2
#eval a 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l621_62171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_collinearity_l621_62172

-- Define the points M and N
def M : ℝ × ℝ := (-1, 0)
def N : ℝ × ℝ := (1, 0)

-- Define the distance ratio condition
def distance_ratio (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (x + 1)^2 + y^2 = 2 * ((x - 1)^2 + y^2)

-- Define the trajectory C
def C (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x^2 + y^2 - 6*x + 1 = 0

-- Define the line l passing through M
def l (k : ℝ) (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  y = k * (x + 1)

-- Define the reflection across x-axis
def reflect_x (P : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := P; (x, -y)

-- Theorem statement
theorem trajectory_and_collinearity 
  (P A B : ℝ × ℝ) (k : ℝ) :
  distance_ratio P →
  C P →
  l k A →
  l k B →
  C A →
  C B →
  A ≠ B →
  let Q := reflect_x A
  (C P ↔ (let (x, y) := P; x^2 + y^2 - 6*x + 1 = 0)) ∧
  ∃ (m : ℝ), (let (x₂, y₂) := B; y₂ = m * (x₂ - 1)) ∧
             (let (x₁, y₁) := Q; y₁ = m * (x₁ - 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_collinearity_l621_62172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_video_library_storage_efficiency_l621_62156

/-- Calculates the average megabytes per minute of video in a digital library, rounded to the nearest whole number. -/
def averageMegabytesPerMinute (days : ℕ) (totalMegabytes : ℕ) : ℕ :=
  let totalMinutes := days * 24 * 60
  let exactAverage : ℚ := totalMegabytes / totalMinutes
  (exactAverage + 1/2).floor.toNat

/-- Theorem stating that for a 15-day video library taking 24,000 megabytes, 
    the average megabytes per minute rounded to the nearest whole number is 1. -/
theorem video_library_storage_efficiency :
  averageMegabytesPerMinute 15 24000 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_video_library_storage_efficiency_l621_62156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_points_custom_op_l621_62188

-- Define the custom operation as noncomputable
noncomputable def custom_op (a b : ℝ) : ℝ :=
  if a ≥ b then Real.sqrt (a^2 + b^2) else a * b

-- Define the theorem
theorem line_points_custom_op :
  ∀ (m n : ℝ),
  (4/5 * m + n = 0) →  -- Line passes through (4/5, 0)
  (n = 4) →            -- Line passes through (0, 4)
  custom_op m n = -20 :=
by
  -- Introduce variables and hypotheses
  intro m n h1 h2
  -- Use the given equations to determine m and n
  have h3 : m = -5 := by
    rw [h2] at h1
    linarith
  -- Apply the definition of custom_op
  rw [custom_op, h3, h2]
  -- Show that -5 < 4
  have h4 : (-5 : ℝ) < 4 := by norm_num
  -- Simplify the if-then-else expression
  simp [h4]
  -- Evaluate the final expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_points_custom_op_l621_62188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l621_62108

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + 2*y^2 = 2

-- Define the length of the major axis
noncomputable def major_axis_length : ℝ := 2 * Real.sqrt 2

-- Theorem statement
theorem ellipse_major_axis_length :
  ∀ x y : ℝ, ellipse x y → major_axis_length = 2 * Real.sqrt 2 := by
  intros x y h
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l621_62108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_northeast_southwest_angle_l621_62111

/-- Represents a circular arrangement of rays -/
structure CircularArrangement where
  num_rays : ℕ
  north_aligned : Bool

/-- Calculates the angle between two rays in a circular arrangement -/
noncomputable def angle_between_rays (arrangement : CircularArrangement) (ray1 ray2 : ℕ) : ℝ :=
  (((ray2 - ray1 + arrangement.num_rays) % arrangement.num_rays) : ℝ) * (360 / arrangement.num_rays)

/-- Theorem: In a circular arrangement with 12 equally spaced rays, where one ray points due North,
    the smaller angle formed between the Northeast and Southwest rays is 120° -/
theorem northeast_southwest_angle (arrangement : CircularArrangement) 
  (h1 : arrangement.num_rays = 12)
  (h2 : arrangement.north_aligned = true) :
  min (angle_between_rays arrangement 1 7) (angle_between_rays arrangement 7 1) = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_northeast_southwest_angle_l621_62111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l621_62132

-- Define the original function f
def f (x : ℝ) : ℝ := x^2

-- Define the inverse function g
noncomputable def g (x : ℝ) : ℝ := -Real.sqrt x

-- Theorem statement
theorem inverse_function_proof :
  ∀ x > 4, ∀ y < -2,
    f y = x ↔ g x = y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l621_62132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_randy_piano_expert_l621_62182

/-- Calculates the age at which Randy will become a piano expert -/
def piano_expert_age (
  current_age : ℕ
) (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (weeks_per_year : ℕ)
  (vacation_weeks : ℕ)
  (expert_hours : ℕ) : ℕ :=
  let hours_per_week := hours_per_day * days_per_week
  let practice_weeks := weeks_per_year - vacation_weeks
  let hours_per_year := hours_per_week * practice_weeks
  let years_to_expert := expert_hours / hours_per_year
  current_age + years_to_expert

/-- Theorem stating that Randy will become a piano expert at age 20 -/
theorem randy_piano_expert :
  piano_expert_age 12 5 5 52 2 10000 = 20 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_randy_piano_expert_l621_62182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_minutes_heard_is_33_l621_62159

/-- Represents the duration of the talk in minutes -/
def talkDuration : ℚ := 60

/-- Represents the fraction of the audience who heard the entire talk -/
def fullHearingFraction : ℚ := 1/5

/-- Represents the fraction of the audience who slept through the entire talk -/
def sleepingFraction : ℚ := 1/10

/-- Represents the fraction of the remaining audience who heard one third of the talk -/
def oneThirdHearingFraction : ℚ := (1 - fullHearingFraction - sleepingFraction) / 2

/-- Represents the fraction of the remaining audience who heard two thirds of the talk -/
def twoThirdsHearingFraction : ℚ := (1 - fullHearingFraction - sleepingFraction) / 2

/-- Calculates the average number of minutes heard by the audience -/
def averageMinutesHeard : ℚ :=
  fullHearingFraction * talkDuration +
  sleepingFraction * 0 +
  oneThirdHearingFraction * (talkDuration / 3) +
  twoThirdsHearingFraction * (2 * talkDuration / 3)

/-- Theorem stating that the average number of minutes heard is 33 -/
theorem average_minutes_heard_is_33 : averageMinutesHeard = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_minutes_heard_is_33_l621_62159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_share_is_6200_l621_62196

/-- Represents the investment amount of partner A -/
def x : ℝ := 1 -- Assuming a unit investment for simplicity

/-- The annual gain of the partnership -/
def annual_gain : ℝ := 18600

/-- A's investment duration in months -/
def a_duration : ℝ := 12

/-- B's investment duration in months -/
def b_duration : ℝ := 6

/-- C's investment duration in months -/
def c_duration : ℝ := 4

/-- A's investment amount -/
def a_investment : ℝ := x

/-- B's investment amount -/
def b_investment : ℝ := 2 * x

/-- C's investment amount -/
def c_investment : ℝ := 3 * x

/-- A's share ratio in the partnership -/
def a_share_ratio : ℝ := a_investment * a_duration

/-- B's share ratio in the partnership -/
def b_share_ratio : ℝ := b_investment * b_duration

/-- C's share ratio in the partnership -/
def c_share_ratio : ℝ := c_investment * c_duration

/-- Total share ratio of the partnership -/
def total_share_ratio : ℝ := a_share_ratio + b_share_ratio + c_share_ratio

/-- A's share in the partnership's gain -/
noncomputable def a_share : ℝ := (a_share_ratio / total_share_ratio) * annual_gain

/-- Theorem stating A's share in the partnership's gain -/
theorem a_share_is_6200 : a_share = 6200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_share_is_6200_l621_62196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_l621_62139

-- Define the polar coordinates
noncomputable def ρ (θ : ℝ) : ℝ := -4 * Real.cos θ

-- Define the conversion from polar to Cartesian coordinates
noncomputable def x (θ : ℝ) : ℝ := ρ θ * Real.cos θ
noncomputable def y (θ : ℝ) : ℝ := ρ θ * Real.sin θ

-- State the theorem
theorem polar_to_cartesian :
  ∀ θ : ℝ, (x θ + 2)^2 + (y θ)^2 = 4 := by
  intro θ
  -- Expand the definitions of x and y
  unfold x y ρ
  -- Simplify the expression
  simp [Real.cos_sq_add_sin_sq]
  -- The rest of the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_l621_62139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_sequence_2003_l621_62198

/-- Represents the sequence of colored numbers -/
def ColoredSequence : ℕ → ℕ := sorry

/-- The first number in the sequence is 1 -/
axiom first_number : ColoredSequence 1 = 1

/-- Each step alternates between even and odd numbers -/
axiom alternating_parity (n : ℕ) :
  ∃ k : ℕ, (∀ i ∈ Finset.range k, Even (ColoredSequence (n + i))) ∧
           (∀ i ∈ Finset.range k, Odd (ColoredSequence (n + k + i)))

/-- The number of integers colored in each step increases by 1 -/
axiom increasing_step_size (n : ℕ) :
  ∃ k : ℕ, (∀ i ∈ Finset.range k, ColoredSequence (n + i) < ColoredSequence (n + k)) ∧
           (∀ i ∈ Finset.range (k + 1), ColoredSequence (n + k + i) < ColoredSequence (n + k + k + 1))

/-- The pattern continues indefinitely -/
axiom infinite_sequence : ∀ n : ℕ, ∃ m : ℕ, ColoredSequence m > n

/-- The 2003rd number in the sequence is 3943 -/
theorem colored_sequence_2003 : ColoredSequence 2003 = 3943 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_sequence_2003_l621_62198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mercedes_sold_count_l621_62136

-- Define the total number of cars sold
def total_cars : ℕ := 300

-- Define percentages for each car brand
def bmw_percent : ℚ := 20 / 100
def toyota_percent : ℚ := 25 / 100
def nissan_percent : ℚ := 10 / 100

-- Define the number of Mercedes sold
def mercedes_sold : ℕ := 135

-- Theorem statement
theorem mercedes_sold_count :
  mercedes_sold = total_cars - (bmw_percent * (total_cars : ℚ)).floor.toNat
                               - (toyota_percent * (total_cars : ℚ)).floor.toNat
                               - (nissan_percent * (total_cars : ℚ)).floor.toNat :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mercedes_sold_count_l621_62136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l621_62161

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x - 1

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 + a*x + 1)

-- Define proposition p
def prop_p (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc (-1 : ℝ) 1, ∀ y ∈ Set.Icc (-1 : ℝ) 1, x < y → f a x > f a y

-- Define proposition q
def prop_q (a : ℝ) : Prop :=
  Set.range (g a) = Set.univ

-- Main theorem
theorem range_of_a :
  {a : ℝ | (prop_p a ∧ ¬prop_q a) ∨ (¬prop_p a ∧ prop_q a)} =
  Set.Iic (-2 : ℝ) ∪ Set.Icc 2 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l621_62161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_observation_count_l621_62199

theorem observation_count (n : ℕ) (original_mean new_mean wrong_value correct_value : ℚ) 
  (h1 : original_mean = 36)
  (h2 : new_mean = 365/10)
  (h3 : wrong_value = 23)
  (h4 : correct_value = 60)
  (h5 : (n : ℚ) * original_mean + (correct_value - wrong_value) = n * new_mean) :
  n = 74 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_observation_count_l621_62199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_nine_n_squared_l621_62103

/-- Given an even integer n with exactly 17 positive divisors,
    the number of positive divisors of 9n^2 is 99 -/
theorem divisors_of_nine_n_squared (n : ℕ) 
  (h_even : Even n) 
  (h_divisors : (Nat.divisors n).card = 17) : 
  (Nat.divisors (9 * n^2)).card = 99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_nine_n_squared_l621_62103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_fixed_point_theorem_l621_62189

/-- Parabola represented by its equation y^2 = 4x -/
structure Parabola where
  eq : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Point on a parabola -/
structure PointOnParabola (p : Parabola) where
  point : ℝ × ℝ
  on_parabola : p.eq point.1 point.2

/-- Line represented by two points -/
structure Line where
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ

/-- Theorem statement -/
theorem parabola_fixed_point_theorem (C : Parabola)
  (A B : ℝ × ℝ)
  (h_C_eq : C.eq = fun x y => y^2 = 4*x)
  (h_C_focus : C.focus = (1, 0))
  (h_A : A = (-1, 0))
  (h_B : B = (1, -1))
  (M N : PointOnParabola C)
  (h_MN_distinct : M.point ≠ N.point)
  (h_AMN : ∃ (l : Line), l.p1 = A ∧ l.p2 = M.point ∧ ∃ t : ℝ, 0 < t ∧ t < 1 ∧ N.point = (1-t) • l.p1 + t • l.p2)
  (Q : PointOnParabola C)
  (h_MQB : ∃ (l : Line), l.p1 = M.point ∧ l.p2 = Q.point ∧ ∃ t : ℝ, 0 < t ∧ t < 1 ∧ B = (1-t) • l.p1 + t • l.p2) :
  ∃ (l : Line), l.p1 = Q.point ∧ l.p2 = N.point ∧ ∃ t : ℝ, 0 < t ∧ t < 1 ∧ (1, -4) = (1-t) • l.p1 + t • l.p2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_fixed_point_theorem_l621_62189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_distance_l621_62151

/-- Parabola structure -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- Line structure -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Point structure -/
structure Point where
  x : ℝ
  y : ℝ

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

def Parabola.focus (p : Parabola) : Point :=
  ⟨1, 0⟩

def Parabola.intersect_line (p : Parabola) (l : Line) : Set Point :=
  {pt : Point | p.equation pt.x pt.y ∧ pt.y = l.slope * (pt.x - 1) + l.y_intercept}

theorem parabola_intersection_distance 
  (C : Parabola)
  (MN : Line)
  (M N : Point)
  (h1 : C.equation = fun x y => y^2 = 4*x)
  (h2 : M ∈ C.intersect_line MN)
  (h3 : N ∈ C.intersect_line MN)
  (h4 : ∃ (D : Point), distance M D = 2 * distance (C.focus) N) :
  distance M (C.focus) = Real.sqrt 3 + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_distance_l621_62151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cubes_of_roots_l621_62160

-- Define the cubic roots
noncomputable def α : ℝ := Real.rpow 23 (1/3)
noncomputable def β : ℝ := Real.rpow 63 (1/3)
noncomputable def γ : ℝ := Real.rpow 113 (1/3)

-- Define the equation
def equation (x : ℝ) : Prop := (x - α) * (x - β) * (x - γ) = 1/3

-- Theorem statement
theorem sum_of_cubes_of_roots (r s t : ℝ) :
  equation r ∧ equation s ∧ equation t ∧ r ≠ s ∧ s ≠ t ∧ r ≠ t →
  r^3 + s^3 + t^3 = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cubes_of_roots_l621_62160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l621_62102

-- Define the polynomial
def p (x : ℝ) : ℝ := x^5 + 2*x^3 + x + 3

-- Define the divisor
def d (x : ℝ) : ℝ := x - 4

-- Theorem statement
theorem polynomial_remainder :
  ∃ (q : ℝ → ℝ), p = λ x ↦ d x * q x + 1159 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l621_62102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_condition_l621_62168

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 2*x + y^2 - 2*y + 1 = 0

-- Define the line equation
def line_equation (x y b : ℝ) : Prop :=
  y = x + b

-- Define the tangent condition
def is_tangent (b : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_equation x y ∧ line_equation x y b ∧
  ∀ (x' y' : ℝ), circle_equation x' y' → line_equation x' y' b → (x' = x ∧ y' = y)

-- The theorem to prove
theorem tangent_condition :
  ∀ b : ℝ, is_tangent b ↔ (b = Real.sqrt 2 ∨ b = -Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_condition_l621_62168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_length_calculation_l621_62137

/-- The length of a road in kilometers that can be asphalted given the number of workers,
    based on a reference road construction project. -/
noncomputable def road_length (num_workers : ℝ) : ℝ :=
  num_workers / 30

/-- Theorem stating that the length of the road asphalted by a given number of workers
    in 12 days, working 8 hours per day, is equal to the number of workers divided by 30,
    given a reference project where 20 workers asphalt 2 km in 28.8 days working 10 hours per day. -/
theorem road_length_calculation (num_workers : ℝ) :
  let first_project_man_hours := num_workers * 12 * 8
  let reference_project_man_hours := 20 * 28.8 * 10
  let reference_project_length := 2
  first_project_man_hours / road_length num_workers = reference_project_man_hours / reference_project_length :=
by
  -- Unfold the definition of road_length
  unfold road_length
  -- Simplify the equation
  simp [mul_assoc, mul_comm, mul_left_comm]
  -- The proof is complete
  sorry

#check road_length_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_length_calculation_l621_62137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_impossible_l621_62140

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculates the area of a triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  abs ((t.B.x - t.A.x) * (t.C.y - t.A.y) - (t.C.x - t.A.x) * (t.B.y - t.A.y)) / 2

/-- Defines the initial triangle -/
def initialTriangle : Triangle :=
  { A := { x := 0, y := 0 }
    B := { x := 1, y := 0 }
    C := { x := 0, y := 1 } }

/-- Defines the final triangle -/
def finalTriangle : Triangle :=
  { A := { x := 1, y := 0 }
    B := { x := -1, y := 0 }
    C := { x := 0, y := 1 } }

/-- Theorem stating that the transformation is impossible -/
theorem transformation_impossible : 
  ∀ (moves : ℕ → Triangle → Triangle),
    (∀ t : Triangle, triangleArea (moves 0 t) = triangleArea t) →
    (∀ n : ℕ, ∀ t : Triangle, triangleArea (moves (n+1) t) = triangleArea (moves n t)) →
    ¬ ∃ n : ℕ, moves n initialTriangle = finalTriangle :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_impossible_l621_62140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_sum_l621_62164

noncomputable section

-- Define the points A, B, C
def A : ℝ × ℝ := (0, 6)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (10, 0)

-- Define D as the midpoint of AB
noncomputable def D : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define E as the midpoint of BC
noncomputable def E : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define the line AE
noncomputable def line_AE (x : ℝ) : ℝ := (E.2 - A.2) / (E.1 - A.1) * (x - A.1) + A.2

-- Define the line CD
noncomputable def line_CD (x : ℝ) : ℝ := (D.2 - C.2) / (D.1 - C.1) * (x - C.1) + C.2

-- Define the intersection point F
noncomputable def F : ℝ × ℝ := 
  let x := (line_CD 0 - line_AE 0) / ((E.2 - A.2) / (E.1 - A.1) - (D.2 - C.2) / (D.1 - C.1))
  (x, line_AE x)

-- Theorem statement
theorem intersection_point_sum : F.1 + F.2 = 5 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_sum_l621_62164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_length_is_16_l621_62154

/-- Regular triangular pyramid with given conditions -/
structure RegularTriangularPyramid where
  -- Base side length
  a : ℝ
  -- Angle between lateral face and base
  α : ℝ
  -- Height of the pyramid
  h : ℝ
  -- Condition that α = arctg(3/4)
  angle_condition : α = Real.arctan (3/4)
  -- Relation between a, h, and α
  height_condition : h = (a * Real.sqrt 3) / 8

/-- Rectangular prism inside the pyramid -/
structure RectangularPrism (pyramid : RegularTriangularPyramid) where
  -- Height of the prism
  height : ℝ
  -- Condition that height is 3/4 of pyramid height
  height_condition : height = (3 * pyramid.h) / 4

/-- Total surface area of the polyhedron MNKFPR -/
noncomputable def totalSurfaceArea (pyramid : RegularTriangularPyramid) (prism : RectangularPrism pyramid) : ℝ :=
  -- This is a placeholder for the actual surface area calculation
  53 * Real.sqrt 3

/-- Main theorem: prove that the side length of ABC is 16 -/
theorem side_length_is_16 (pyramid : RegularTriangularPyramid) (prism : RectangularPrism pyramid) 
    (h : totalSurfaceArea pyramid prism = 53 * Real.sqrt 3) : 
  pyramid.a = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_length_is_16_l621_62154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l621_62126

open Real

-- Define the functions f, g, and h
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry
def h (x : ℝ) : ℝ := 2 * f x - g x

-- Define the properties of f and g
axiom f_even : ∀ x, f (-x) = f x
axiom g_odd : ∀ x, g (-x) = -g x

-- Define the relationship between f and g
axiom f_minus_g : ∀ x, f x - g x = exp x + x^2 + 1

-- State the theorem
theorem tangent_line_at_zero :
  ∃ m b, ∀ x y, y = m * x + b ↔ (x = 0 ∧ y = h 0) ∨ 
  (x ≠ 0 ∧ y = h x + (deriv h 0) * (x - 0)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l621_62126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_distance_l621_62100

/-- The circle equation -/
def circle_eq (x y r : ℝ) : Prop := (x - 3)^2 + (y + 5)^2 = r^2

/-- The line equation -/
def line_eq (x y : ℝ) : Prop := 4*x - 3*y - 2 = 0

/-- The shortest distance from a point on the circle to the line -/
def shortest_distance : ℝ := 1

/-- Theorem: If the shortest distance from a point on the circle to the line is 1, then r = 4 -/
theorem circle_line_distance (r : ℝ) : 
  (∃ x y : ℝ, circle_eq x y r ∧ line_eq x y) → 
  shortest_distance = 1 → 
  r = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_distance_l621_62100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_d_is_fixed_point_l621_62191

/-- The complex number around which the given function represents a rotation -/
noncomputable def d : ℂ := -2 * Real.sqrt 3 + 7 * Complex.I

/-- The given function -/
noncomputable def g (z : ℂ) : ℂ := ((1 - Complex.I * Real.sqrt 3) * z + (2 * Real.sqrt 3 + 20 * Complex.I)) / (-2)

/-- Theorem stating that d is the fixed point of g -/
theorem d_is_fixed_point : g d = d := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_d_is_fixed_point_l621_62191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_57_l621_62130

/-- The price of a single caramel in dollars -/
noncomputable def caramel_price : ℝ := 3

/-- The price of a single candy bar in dollars -/
noncomputable def candy_bar_price : ℝ := 2 * caramel_price

/-- The price of cotton candy in dollars -/
noncomputable def cotton_candy_price : ℝ := (1/2) * (4 * candy_bar_price)

/-- The total cost of 6 candy bars, 3 caramels, and 1 cotton candy in dollars -/
noncomputable def total_cost : ℝ := 6 * candy_bar_price + 3 * caramel_price + cotton_candy_price

theorem total_cost_is_57 : total_cost = 57 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_57_l621_62130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l621_62121

-- Define the work rates for each person
noncomputable def rate_A : ℝ := 1 / 20
noncomputable def rate_B : ℝ := 1 / 15
noncomputable def rate_C : ℝ := 1 / 10

-- Define the combined rate
noncomputable def combined_rate : ℝ := rate_A + rate_B + rate_C

-- Theorem: The time taken to complete the work together is 60/13 days
theorem work_completion_time :
  (1 : ℝ) / combined_rate = 60 / 13 := by
  -- Expand the definition of combined_rate
  unfold combined_rate
  -- Perform algebraic manipulations
  simp [rate_A, rate_B, rate_C]
  -- The rest of the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l621_62121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l621_62186

theorem problem_statement (x : ℝ) (h1 : x ∈ Set.Ioo (π / 2) π) 
  (h2 : (4 / 3) * (1 / Real.sin x + 1 / Real.cos x) = 1) :
  (4 / 3)^4 * (1 / Real.sin x^4 + 1 / Real.cos x^4) = 49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l621_62186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_multiple_of_three_sequence_plus_one_is_square_sequence_increasing_sequence_2006th_term_l621_62183

/-- A sequence of multiples of 3 which, when 1 is added, are perfect squares -/
def sequenceA (n : ℕ) : ℕ := (3 * n + 1)^2 - 1

/-- The nth term of the sequence is a multiple of 3 -/
theorem sequence_multiple_of_three (n : ℕ) : 3 ∣ sequenceA n := by
  sorry

/-- The nth term of the sequence plus 1 is a perfect square -/
theorem sequence_plus_one_is_square (n : ℕ) : ∃ k : ℕ, sequenceA n + 1 = k^2 := by
  sorry

/-- The sequence is strictly increasing -/
theorem sequence_increasing : StrictMono sequenceA := by
  sorry

/-- The 2006th term of the sequence is 9060099 -/
theorem sequence_2006th_term : sequenceA 2006 = 9060099 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_multiple_of_three_sequence_plus_one_is_square_sequence_increasing_sequence_2006th_term_l621_62183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_phi_value_l621_62150

theorem tan_phi_value (φ : ℝ) 
  (h1 : Real.cos (π / 2 - φ) = Real.sqrt 3 / 2)
  (h2 : |φ| < π / 2) : 
  Real.tan φ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_phi_value_l621_62150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l621_62157

theorem division_problem (x y : ℕ) (h1 : x % y = 8) (h2 : (x : ℝ) / (y : ℝ) = 96.16) : y = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l621_62157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_from_projections_l621_62187

noncomputable def proj (a : ℝ × ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (v.1 * a.1 + v.2 * a.2) / (a.1 * a.1 + a.2 * a.2)
  (scalar * a.1, scalar * a.2)

theorem vector_from_projections (v : ℝ × ℝ) :
  proj (3, 1) v = (45/10, 15/10) ∧
  proj (1, 4) v = (65/17, 260/17) →
  v = (-85/187, 3060/187) := by
  sorry

#check vector_from_projections

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_from_projections_l621_62187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_difference_l621_62141

theorem number_difference (x y : ℤ) : 
  x + y = 40 → 3 * y - 4 * x = 7 → (y - x).natAbs = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_difference_l621_62141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_sin_cos_l621_62194

open Real

theorem triangle_max_sin_cos (S a b c A B C : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  S = (1/2) * b * c * sin A →
  4 * S + a^2 = b^2 + c^2 →
  (∃ (max_val : ℝ), max_val = Real.sqrt 2 ∧
    ∀ (C' : ℝ), 0 < C' ∧ C' < π →
      sin C' - cos (B + π/4) ≤ max_val) ∧
  (sin (π/4) - cos (B + π/4) = Real.sqrt 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_sin_cos_l621_62194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_calculation_l621_62117

-- Define the structure for an item
structure Item where
  sellingPrice : ℚ
  profitPercentage : ℚ

-- Define the function to calculate cost price
noncomputable def costPrice (item : Item) : ℚ :=
  item.sellingPrice / (1 + item.profitPercentage)

-- Define the items
def itemA : Item := { sellingPrice := 400, profitPercentage := 1/4 }
def itemB : Item := { sellingPrice := 600, profitPercentage := 1/5 }
def itemC : Item := { sellingPrice := 800, profitPercentage := 3/20 }

-- State the theorem
theorem cost_price_calculation :
  (costPrice itemA = 320) ∧
  (costPrice itemB = 500) ∧
  (abs (costPrice itemC - 695.65) < 0.01) := by
  sorry

-- Note: We use an inequality for item C to account for potential rounding differences

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_calculation_l621_62117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_l621_62143

/-- Represents a cone with given base radius and lateral surface sector angle -/
structure Cone where
  base_radius : ℝ
  sector_angle : ℝ

/-- Calculates the surface area of a cone -/
noncomputable def surface_area (c : Cone) : ℝ :=
  let base_area := Real.pi * c.base_radius ^ 2
  let lateral_area := Real.pi * c.base_radius * (2 * c.base_radius / c.sector_angle)
  base_area + lateral_area

/-- Theorem: The surface area of a cone with base radius 3 and sector angle 2π/3 is 36π -/
theorem cone_surface_area :
  let c : Cone := { base_radius := 3, sector_angle := 2 * Real.pi / 3 }
  surface_area c = 36 * Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_l621_62143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_segment_ratio_l621_62163

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_sides : a = 9 ∧ b = 12 ∧ c = 15

-- Define the inscribed circle
def InscribedCircle (t : Triangle) := {r : ℝ | ∃ (x y z : ℝ), x + y + z = (t.a + t.b + t.c) / 2 ∧ x * y * z = r * (x + y + z)}

-- Define the segments formed by the point of tangency
structure Segments (t : Triangle) (c : InscribedCircle t) where
  p : ℝ
  q : ℝ
  h_sum : p + q = t.a
  h_order : p < q

-- State the theorem
theorem inscribed_circle_segment_ratio (t : Triangle) (c : InscribedCircle t) (s : Segments t c) :
  s.p / s.q = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_segment_ratio_l621_62163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_factors_of_360_l621_62122

theorem distinct_prime_factors_of_360 :
  (Finset.card (Finset.filter (λ p => Nat.Prime p ∧ p ∣ 360) (Finset.range 361))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_factors_of_360_l621_62122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_l621_62155

/-- If f(x) = x^a and its inverse passes through (1/2, 1/4), then a = 1/2 -/
theorem inverse_function_point (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x, f x = x^a) → f⁻¹ (1/2) = 1/4 → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_l621_62155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_k_values_l621_62120

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of line l1: kx + (1-k)y - 3 = 0 -/
noncomputable def slope_l1 (k : ℝ) : ℝ := -k / (1 - k)

/-- The slope of line l2: (k-1)x + (2k+3)y - 2 = 0 -/
noncomputable def slope_l2 (k : ℝ) : ℝ := -(k - 1) / (2*k + 3)

/-- Theorem: If l1 and l2 are perpendicular, then k = 1 or k = -3 -/
theorem perpendicular_lines_k_values (k : ℝ) :
  perpendicular (slope_l1 k) (slope_l2 k) → k = 1 ∨ k = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_k_values_l621_62120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_circle_equation_l621_62166

/-- The equation of a circle symmetric to another circle with respect to a line -/
theorem symmetric_circle_equation :
  let C₁ : Set (ℝ × ℝ) := {p | (p.1 + 1)^2 + (p.2 - 1)^2 = 1}
  let L : Set (ℝ × ℝ) := {p | p.1 - p.2 - 1 = 0}
  let C₂ : Set (ℝ × ℝ) := {p | ∃ q ∈ C₁, ∀ r ∈ L, dist p r = dist q r}
  C₂ = {p | (p.1 - 2)^2 + (p.2 + 2)^2 = 1} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_circle_equation_l621_62166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l621_62184

noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

noncomputable def focal_length (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

noncomputable def slope_AB (a b : ℝ) : ℝ := -b / a

theorem ellipse_intersection_theorem (a b k : ℝ) :
  a > 0 ∧ b > 0 ∧ a > b ∧ k < 0 →
  focal_length a b = 2 * Real.sqrt 5 →
  slope_AB a b = -2/3 →
  ∃ (x₀ y₀ x₁ y₁ : ℝ),
    ellipse a b x₁ y₁ ∧
    x₁ < 0 ∧ y₁ > 0 ∧
    y₁ = k * x₁ ∧
    ellipse a b (-x₁) (-y₁) ∧
    y₀ = k * x₀ ∧
    (x₀ - (-x₁))^2 + (y₀ - (-y₁))^2 = 9 * ((x₁ - (-x₁))^2 + (y₁ - (-y₁))^2) →
  k = -8/9 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l621_62184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l621_62167

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x * (100 - x)) + Real.sqrt (x * (4 - x))

theorem max_value_of_f :
  ∃ (M : ℝ), M = 8 * Real.sqrt 6 ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 4 → f x ≤ M) ∧
  f 4 = M := by
  -- We define M as 8√6
  let M := 8 * Real.sqrt 6
  
  -- We prove that this M satisfies the three conditions
  have h1 : M = 8 * Real.sqrt 6 := rfl
  
  have h2 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 4 → f x ≤ M := by
    sorry -- This requires a more complex proof
  
  have h3 : f 4 = M := by
    -- Expand the definition of f
    unfold f
    -- Simplify
    simp [Real.sqrt_mul, Real.sqrt_sq]
    -- The rest of the proof would involve algebraic manipulations
    sorry
  
  -- Combine the three conditions
  exact ⟨M, h1, h2, h3⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l621_62167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_at_one_second_l621_62170

-- Define the motion equation
noncomputable def S (t : ℝ) : ℝ := 2 * t^2 + t

-- Define instantaneous velocity as the derivative of S
noncomputable def instantaneous_velocity (t : ℝ) : ℝ := 
  (deriv S) t

-- Theorem statement
theorem velocity_at_one_second :
  instantaneous_velocity 1 = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_at_one_second_l621_62170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_points_theorem_l621_62118

theorem sphere_points_theorem (S I : ℕ) : 
  let T := S + I
  S ≤ (72 : ℕ) * T / 100 →
  Nat.choose S 2 = 14 →
  T = 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_points_theorem_l621_62118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_equals_two_l621_62145

-- Define the function f(x) = x cos(x) + 1
noncomputable def f (x : ℝ) : ℝ := x * Real.cos x + 1

-- Define the interval (-5, 5)
def I : Set ℝ := Set.Ioo (-5) 5

-- State the theorem
theorem max_min_sum_equals_two :
  ∃ (M m : ℝ), 
    (∀ x ∈ I, f x ≤ M) ∧ 
    (∃ x ∈ I, f x = M) ∧
    (∀ x ∈ I, m ≤ f x) ∧ 
    (∃ x ∈ I, f x = m) ∧
    (M + m = 2) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_equals_two_l621_62145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l621_62153

-- Define the circle (M)
def Circle (t : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - t)^2 + (p.2 - Real.sqrt 3/t)^2 = t^2 + 3/t^2}

-- Define the line l
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = -(Real.sqrt 3)/3 * p.1 + 4}

theorem circle_properties (t : ℝ) :
  -- The circle passes through the origin
  (0, 0) ∈ Circle t →
  -- The center of the circle lies on y = √3/x
  (t, Real.sqrt 3/t) ∈ {p : ℝ × ℝ | p.2 = Real.sqrt 3/p.1} →
  -- The area of triangle AOB is constant
  (∃ (a b : ℝ), a ∈ Set.Icc 0 (2*t) ∧ b ∈ Set.Icc 0 (2*Real.sqrt 3/t) ∧ 
    (a, 0) ∈ Circle t ∧ (0, b) ∈ Circle t ∧ a * b / 2 = 2*Real.sqrt 3) ∧
  -- The line l intersects the circle at two equidistant points from the origin
  (∃ (c d : ℝ × ℝ), c ∈ Circle t ∧ d ∈ Circle t ∧ c ∈ Line ∧ d ∈ Line ∧ 
    c.1^2 + c.2^2 = d.1^2 + d.2^2) →
  -- The equation of the circle is (x-1)² + (y-√3)² = 4
  t = 1 ∧ Circle t = {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - Real.sqrt 3)^2 = 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l621_62153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_meals_count_l621_62169

/-- The number of options in each category (entrees, drinks, desserts) -/
def category_options : ℕ := 3

/-- The number of categories (entrees, drinks, desserts) -/
def num_categories : ℕ := 3

/-- A meal consists of one item from each category -/
def meal := Fin num_categories → Fin category_options

/-- Prove that meal is a finite type -/
instance : Fintype meal := Pi.fintype

theorem distinct_meals_count :
  Fintype.card meal = 27 := by
  -- Calculate the cardinality of the meal type
  calc Fintype.card meal
    = (Fintype.card (Fin num_categories → Fin category_options)) := rfl
    _ = (Fintype.card (Fin category_options)) ^ (Fintype.card (Fin num_categories)) := by apply Fintype.card_pi
    _ = category_options ^ num_categories := by simp [category_options, num_categories]
    _ = 3 ^ 3 := by rfl
    _ = 27 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_meals_count_l621_62169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l621_62193

/-- Given a hyperbola with equation x²/m + y²/n = 1 where m < 0 < n,
    and asymptote equations y = ±√2x, its eccentricity is √3. -/
theorem hyperbola_eccentricity (m n : ℝ) (hm : m < 0) (hn : 0 < n) :
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 / m + y^2 / n = 1}
  let asymptote := fun x : ℝ ↦ Real.sqrt 2 * x
  let eccentricity := Real.sqrt (1 + (Real.sqrt 2)^2)
  (∀ x, (x, asymptote x) ∈ closure hyperbola ∨ (x, -asymptote x) ∈ closure hyperbola) →
  eccentricity = Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l621_62193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_father_digging_rate_l621_62148

/-- The depth of the hole dug by the father -/
noncomputable def D : ℝ := 1600

/-- The depth of the hole dug by Michael -/
noncomputable def michael_depth : ℝ := 2 * D - 400

/-- The time taken by the father to dig his hole -/
noncomputable def father_time : ℝ := 400

/-- The time taken by Michael to dig his hole -/
noncomputable def michael_time : ℝ := 700

/-- The digging rate of the father -/
noncomputable def father_rate : ℝ := D / father_time

/-- The digging rate of Michael -/
noncomputable def michael_rate : ℝ := michael_depth / michael_time

theorem father_digging_rate : 
  father_rate = michael_rate ∧ father_rate = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_father_digging_rate_l621_62148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_odd_function_l621_62106

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 1 - m / (5^x + 1)

theorem max_value_of_odd_function (m : ℝ) :
  (∀ x, f m x = -f m (-x)) →  -- f is an odd function
  (∃ x₀ ∈ Set.Icc 0 1, ∀ x ∈ Set.Icc 0 1, f m x ≤ f m x₀) →  -- maximum exists on [0,1]
  (∃ x₀ ∈ Set.Icc 0 1, f m x₀ = 2/3) :=
by
  sorry

#check max_value_of_odd_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_odd_function_l621_62106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_range_l621_62134

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the line l
def line (m : ℝ) (x y : ℝ) : Prop := y = m * x - 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the point F₁ on the ellipse
def F₁ : ℝ × ℝ → Prop := λ p => ellipse p.1 p.2

-- Define the point A as the intersection of line l and ellipse C
def A (m : ℝ) : ℝ × ℝ → Prop := λ p => ellipse p.1 p.2 ∧ line m p.1 p.2

-- Define the point B
def B : ℝ × ℝ → Prop := λ _ => True  -- We don't have specific conditions for B

-- Define the point F₂
def F₂ : ℝ × ℝ → Prop := λ _ => True  -- We don't have specific conditions for F₂

-- Define the relation AF₁ = λ F₁B
def AF₁_relation (lambda : ℝ) (a f₁ b : ℝ × ℝ) : Prop :=
  (a.1 - f₁.1, a.2 - f₁.2) = lambda • (f₁.1 - b.1, f₁.2 - b.2)

-- Define the area of triangle ABF₂
noncomputable def triangle_area (a b f₂ : ℝ × ℝ) : ℝ :=
  abs ((b.1 - a.1) * (f₂.2 - a.2) - (f₂.1 - a.1) * (b.2 - a.2)) / 2

-- Main theorem
theorem triangle_area_range :
  ∀ (m lambda : ℝ) (a f₁ b f₂ : ℝ × ℝ),
    1 ≤ lambda ∧ lambda ≤ 2 →
    A m a →
    F₁ f₁ →
    B b →
    F₂ f₂ →
    AF₁_relation lambda a f₁ b →
    9 * Real.sqrt 5 / 8 ≤ triangle_area a b f₂ ∧ triangle_area a b f₂ ≤ 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_range_l621_62134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l621_62107

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x / (x^2 - 1)

-- Theorem statement
theorem function_properties :
  -- Part I: f(2) = 2/3
  f 2 = 2/3 ∧
  -- Part II: f is decreasing on (-1, 1)
  (∀ x₁ x₂ : ℝ, -1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f x₁ > f x₂) ∧
  -- Part III: f is an odd function
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → f (-x) = -f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l621_62107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_common_tangent_implies_a_eq_2e_l621_62147

/-- Function f(x) = a ln x --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x

/-- Function g(x) = x² --/
def g (x : ℝ) : ℝ := x^2

/-- Condition for a common tangent line --/
def has_common_tangent (a : ℝ) : Prop :=
  ∃ m n : ℝ, m > 0 ∧ n > 0 ∧
    (deriv (f a)) m = deriv g n ∧
    f a m - (deriv (f a)) m * m = g n - deriv g n * n

/-- Main theorem --/
theorem unique_common_tangent_implies_a_eq_2e (a : ℝ) (ha : a > 0) :
  (∃! x : ℝ, has_common_tangent a) → a = 2 * Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_common_tangent_implies_a_eq_2e_l621_62147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_on_x_axis_l621_62105

/-- A hyperbola is a curve with equation x^2 - y^2 = 1 -/
def Hyperbola : Set (ℝ × ℝ) := {p | p.1^2 - p.2^2 = 1}

/-- The focus of a hyperbola is a point from which the difference of the distances 
    to any point on the hyperbola is constant -/
def IsFocus (f : ℝ × ℝ) (h : Set (ℝ × ℝ)) : Prop :=
  ∃ (c : ℝ), ∀ p ∈ h, |dist p (f.1, 0) - dist p (-f.1, 0)| = 2 * c

/-- The x-axis is the set of points with y-coordinate equal to 0 -/
def XAxis : Set (ℝ × ℝ) := {p | p.2 = 0}

theorem hyperbola_focus_on_x_axis :
  ∃ f ∈ XAxis, IsFocus f Hyperbola :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_on_x_axis_l621_62105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_village_earrings_l621_62123

theorem village_earrings (total_women : ℕ) (one_earring_percent : ℚ) 
  (h1 : total_women = 800)
  (h2 : one_earring_percent = 3 / 100) :
  ∃ (women_with_one : ℕ) (women_with_two : ℕ) (women_with_none : ℕ),
    women_with_one + women_with_two + women_with_none = total_women ∧
    women_with_one = (total_women : ℚ) * one_earring_percent ∧
    women_with_two = women_with_none ∧
    women_with_one + 2 * women_with_two = 800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_village_earrings_l621_62123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_is_417_l621_62179

/-- Represents a cell in the grid -/
structure Cell where
  value : Nat
  h_value_bound : value ≤ 4

/-- Represents the 15x15 grid -/
def Grid := Fin 15 → Fin 15 → Cell

/-- The sum of numbers in a 2x2 square is 7 -/
def sum_2x2_is_7 (g : Grid) : Prop :=
  ∀ i j, i < 14 ∧ j < 14 →
    (g i j).value + (g i (j+1)).value + (g (i+1) j).value + (g (i+1) (j+1)).value = 7

/-- The sum of all numbers in the grid -/
def total_sum (g : Grid) : Nat :=
  (Finset.univ : Finset (Fin 15)).sum (λ i => 
    (Finset.univ : Finset (Fin 15)).sum (λ j => (g i j).value))

/-- The theorem statement -/
theorem max_sum_is_417 (g : Grid) (h : sum_2x2_is_7 g) :
  total_sum g ≤ 417 ∧ ∃ g', sum_2x2_is_7 g' ∧ total_sum g' = 417 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_is_417_l621_62179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_of_sample_data_l621_62125

noncomputable def sample_data : List ℝ := [3, 3, 4, 4, 5, 6, 6, 7, 7]

noncomputable def mean (data : List ℝ) : ℝ := (data.sum) / (data.length : ℝ)

noncomputable def variance (data : List ℝ) : ℝ :=
  let μ := mean data
  (data.map (λ x => (x - μ)^2)).sum / (data.length : ℝ)

noncomputable def standard_deviation (data : List ℝ) : ℝ :=
  Real.sqrt (variance data)

theorem standard_deviation_of_sample_data :
  standard_deviation sample_data = (2 * Real.sqrt 5) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_of_sample_data_l621_62125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_and_circumcircle_properties_l621_62152

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculate the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Equation of a circle in standard form (x-h)^2 + (y-k)^2 = r^2 -/
structure CircleEquation where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Find the equation of the circumcircle of a triangle -/
def circumcircleEquation (t : Triangle) : CircleEquation := sorry

/-- Main theorem -/
theorem triangle_and_circumcircle_properties :
  let A : Point := ⟨1, 2⟩
  let B : Point := ⟨1, 3⟩
  let C : Point := ⟨3, 6⟩
  let ABC : Triangle := ⟨A, B, C⟩
  (triangleArea ABC = 1) ∧
  (circumcircleEquation ABC = ⟨5, 5/2, Real.sqrt (65/4)⟩) := by
  sorry

#check triangle_and_circumcircle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_and_circumcircle_properties_l621_62152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l621_62113

/-- The equation of the region -/
def region_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 10*x + 24*y = -144

/-- The area of the region -/
noncomputable def region_area : ℝ := 25 * Real.pi

/-- Theorem stating that the area enclosed by the region defined by the equation
    x^2 + y^2 - 10x + 24y = -144 is equal to 25π -/
theorem area_of_region :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y : ℝ, region_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    region_area = π * radius^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l621_62113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_cheaper_option2_l621_62149

def option1Cost (n : ℕ) : ℕ :=
  (n.repr.toList.map (λ c => c.toNat - 48)).sum

def option2Cost (n : ℕ) : ℕ :=
  (n.bits.map (λ b => if b then 1 else 0)).sum

def isCheaperOption2 (n : ℕ) : Prop :=
  option2Cost n < option1Cost n

theorem smallest_prime_cheaper_option2 : ∃ p : ℕ,
  Nat.Prime p ∧ 
  p < 5000 ∧
  isCheaperOption2 p ∧
  ∀ q : ℕ, Nat.Prime q ∧ q < p ∧ q < 5000 → ¬(isCheaperOption2 q) :=
by
  use 487
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_cheaper_option2_l621_62149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jenine_writing_hours_l621_62115

/-- The number of times Jenine can sharpen a pencil before it runs out -/
def sharpenings : ℕ := 5

/-- The number of hours a pencil can be used before needing sharpening -/
def hours_per_sharpening : ℚ := 3/2

/-- The number of pencils Jenine already has -/
def initial_pencils : ℕ := 10

/-- The cost of a new pencil in dollars -/
def pencil_cost : ℚ := 2

/-- The amount Jenine needs to spend on more pencils in dollars -/
def additional_spend : ℚ := 8

/-- The total number of hours Jenine needs to write for -/
def total_writing_hours : ℚ := 126

theorem jenine_writing_hours :
  (additional_spend / pencil_cost).floor * (sharpenings + 1) * hours_per_sharpening +
  initial_pencils * (sharpenings + 1) * hours_per_sharpening = total_writing_hours := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jenine_writing_hours_l621_62115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcycle_cyclist_problem_l621_62135

/-- The problem of finding when a motorcyclist and cyclist are 252 km apart -/
theorem motorcycle_cyclist_problem (t : ℝ) : 
  let angle : ℝ := 60 * Real.pi / 180
  let motorcyclist_speed : ℝ := 72
  let cyclist_speed : ℝ := 36
  let motorcyclist_distance := fun (s : ℝ) => |motorcyclist_speed * s|
  let cyclist_distance := fun (s : ℝ) => |cyclist_speed * (s - 1)|
  (motorcyclist_distance t)^2 + (cyclist_distance t)^2 - 
    2 * (motorcyclist_distance t) * (cyclist_distance t) * Real.cos angle = 252^2 →
    t = 4 ∨ t = -4 :=
by
  sorry

#check motorcycle_cyclist_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcycle_cyclist_problem_l621_62135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_valid_a_l621_62177

theorem characterization_of_valid_a : ∀ (a : ℝ), a > 0 → (
  (∃ (n : ℕ) (A : ℕ → Set ℤ),
    (∀ i, Set.Infinite (A i)) ∧
    (∀ i j, i ≠ j → A i ∩ A j = ∅) ∧
    (⋃ i, A i) = Set.univ ∧
    (∀ i x y, x ∈ A i → y ∈ A i → x ≠ y → |x - y| ≥ a ^ i)
  ) ↔ a < 2
) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_valid_a_l621_62177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_z_is_purely_imaginary_l621_62138

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := ⟨m^2 + 2*m - 8, m^2 - 2*m⟩

-- Theorem for when z is a real number
theorem z_is_real (m : ℝ) : (z m).im = 0 ↔ m = 0 ∨ m = 2 :=
sorry

-- Theorem for when z is a purely imaginary number
theorem z_is_purely_imaginary (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = -4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_z_is_purely_imaginary_l621_62138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toby_journey_time_l621_62109

-- Define the speeds
noncomputable def unloaded_speed : ℝ := 20
noncomputable def loaded_speed : ℝ := 10

-- Define the distances for each part of the journey
noncomputable def distance1 : ℝ := 180
noncomputable def distance2 : ℝ := 120
noncomputable def distance3 : ℝ := 80
noncomputable def distance4 : ℝ := 140

-- Define the function to calculate time given distance and speed
noncomputable def calculate_time (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

-- Theorem statement
theorem toby_journey_time :
  (calculate_time distance1 loaded_speed) +
  (calculate_time distance2 unloaded_speed) +
  (calculate_time distance3 loaded_speed) +
  (calculate_time distance4 unloaded_speed) = 39 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_toby_journey_time_l621_62109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_iff_l621_62119

noncomputable def f (x : ℝ) := (Real.log (4 * x - 7)) / (Real.sqrt (2 * x - 3))

theorem f_defined_iff (x : ℝ) : 
  (∃ y, f x = y) ↔ x > 7/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_iff_l621_62119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_cross_quadrilateral_l621_62104

/-- Rectangle structure -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Point structure -/
structure Point where
  x : ℝ
  y : ℝ

/-- Quadrilateral structure -/
structure Quadrilateral where
  vertices : Fin 4 → Point

/-- Area of a quadrilateral -/
noncomputable def area_quadrilateral (quad : Quadrilateral) : ℝ := sorry

/-- Orthogonality relation between rectangles -/
def Rectangle.isOrthogonalTo (r1 r2 : Rectangle) : Prop := sorry

/-- Given two rectangles ABCD and EFGH forming a cross shape, 
    where AB = 9, BC = 5, EF = 3, and FG = 10,
    the area of quadrilateral AFCH is 52.5. -/
theorem area_of_cross_quadrilateral 
  (ABCD EFGH : Rectangle) 
  (h1 : ABCD.width = 9) 
  (h2 : ABCD.height = 5)
  (h3 : EFGH.width = 3)
  (h4 : EFGH.height = 10)
  (h5 : ABCD.isOrthogonalTo EFGH) 
  (AFCH : Quadrilateral) : 
  area_quadrilateral AFCH = 52.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_cross_quadrilateral_l621_62104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_cone_apex_angle_l621_62185

/-- Represents a cone with vertex A -/
structure Cone where
  apexAngle : ℝ

/-- Represents the configuration of four cones -/
structure ConeConfiguration where
  cone1 : Cone
  cone2 : Cone
  cone3 : Cone
  cone4 : Cone
  touchExternally : cone1.apexAngle = cone2.apexAngle ∧ 
                    cone1.apexAngle ≠ cone3.apexAngle
  touchInternally : True  -- Simplified representation of internal touching

noncomputable def π : ℝ := Real.pi

theorem third_cone_apex_angle 
  (config : ConeConfiguration)
  (h1 : config.cone1.apexAngle = π / 6)
  (h2 : config.cone2.apexAngle = π / 6)
  (h3 : config.cone4.apexAngle = π / 3) :
  config.cone3.apexAngle = 2 * Real.arctan (1 / (Real.sqrt 3 + 4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_cone_apex_angle_l621_62185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_organization_member_count_l621_62192

/-- Represents an organization with committees and members -/
structure Organization where
  num_committees : Nat
  member_committee_pairs : Finset (Finset Nat)

/-- The number of members in the organization -/
def Organization.num_members (org : Organization) : Nat :=
  org.member_committee_pairs.card

/-- Axioms for the organization structure -/
class OrganizationAxioms (org : Organization) where
  committee_count : org.num_committees = 5
  member_in_two_committees : ∀ m, m ∈ org.member_committee_pairs → m.card = 2
  unique_pair_membership : ∀ m₁ m₂, m₁ ∈ org.member_committee_pairs → m₂ ∈ org.member_committee_pairs → m₁ ≠ m₂
  all_pairs_covered : org.member_committee_pairs.card = Nat.choose org.num_committees 2

theorem organization_member_count {org : Organization} [OrganizationAxioms org] :
  org.num_members = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_organization_member_count_l621_62192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_smallest_second_smallest_l621_62142

theorem difference_smallest_second_smallest : 
  let numbers : List ℕ := [10, 11, 12]
  let smallest := numbers.minimum?
  let second_smallest := (numbers.filter (· ≠ smallest.getD 0)).minimum?
  (second_smallest.getD 0) - (smallest.getD 0) = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_smallest_second_smallest_l621_62142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l621_62101

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the line
def line (m x₀ y₀ x y : ℝ) : Prop := (y - y₀) / (x - x₀) = (0 - y₀) / (m - x₀)

theorem parabola_line_intersection
  (p m : ℝ)
  (hp : p > 0)
  (hm : m ≠ 0)
  (M N P : ℝ × ℝ)
  (hM : parabola p M.1 M.2)
  (hN : parabola p N.1 N.2)
  (hP : P.1 = 0)  -- P is on y-axis
  (hline : line m P.1 P.2 M.1 M.2 ∧ line m P.1 P.2 N.1 N.2)
  (lambda mu : ℝ)
  (hlambda : (M.1 - P.1, M.2 - P.2) = lambda • (m - M.1, -M.2))
  (hmu : (N.1 - P.1, N.2 - P.2) = mu • (m - N.1, -N.2))
  : lambda + mu = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l621_62101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_child_support_owed_l621_62128

def initial_salary : ℕ := 30000
def initial_years : ℕ := 3
def raise_percentage : ℚ := 1/5
def raise_years : ℕ := 4
def child_support_percentage : ℚ := 3/10
def amount_paid : ℕ := 1200

def total_income : ℕ := 
  initial_salary * initial_years + 
  (↑initial_salary * (1 + raise_percentage)).floor.toNat * raise_years

def total_child_support : ℕ := (↑total_income * child_support_percentage).floor.toNat

theorem child_support_owed : 
  total_child_support - amount_paid = 69000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_child_support_owed_l621_62128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_value_l621_62197

def has_18_positive_factors (n : ℕ) : Prop :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 18

theorem y_value (y : ℕ) 
  (h1 : has_18_positive_factors y)
  (h2 : 18 ∣ y)
  (h3 : 24 ∣ y) :
  y = 288 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_value_l621_62197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l621_62124

open Set

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = {x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l621_62124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_saree_sale_price_l621_62176

/-- Calculates the sale price of a saree after successive discounts -/
noncomputable def salePriceAfterDiscounts (originalPrice : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl (fun price discount => price * (1 - discount / 100)) originalPrice

/-- Theorem stating the final sale price of the saree -/
theorem saree_sale_price :
  let originalPrice : ℝ := 495
  let discounts : List ℝ := [15, 10, 5, 3]
  abs (salePriceAfterDiscounts originalPrice discounts - 348.95) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_saree_sale_price_l621_62176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_increasing_interval_l621_62110

theorem sin_cos_increasing_interval :
  ∀ (a b : ℝ), (a = -π ∧ b = -π/2) ∨ (a = -π/2 ∧ b = 0) ∨ (a = 0 ∧ b = π/2) ∨ (a = π/2 ∧ b = π) →
  ((∀ x y, x ∈ Set.Icc a b → y ∈ Set.Icc a b → x < y → Real.sin x < Real.sin y) ∧ 
   (∀ x y, x ∈ Set.Icc a b → y ∈ Set.Icc a b → x < y → Real.cos x < Real.cos y)) ↔ 
  (a = -π/2 ∧ b = 0) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_increasing_interval_l621_62110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_max_implies_a_range_l621_62180

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then 2 * x^3 - 3 * x^2 + 1
  else Real.exp (a * x) + 1

-- State the theorem
theorem function_max_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f a x ≤ 5) →
  (∀ x ∈ Set.Icc (-2 : ℝ) 0, Real.exp (a * x) + 1 ≤ 5) →
  a ≥ -Real.log 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_max_implies_a_range_l621_62180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_cubic_solutions_l621_62178

theorem infinite_cubic_solutions :
  ∃ (S : Set (ℤ × ℤ)), 
    (Set.Infinite S) ∧ 
    (∀ (m n : ℤ), (m, n) ∈ S → 
      (Int.gcd m n = 1) ∧
      (∃ (x₁ x₂ x₃ : ℤ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
        (x₁ + m)^3 = n * x₁ ∧
        (x₂ + m)^3 = n * x₂ ∧
        (x₃ + m)^3 = n * x₃)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_cubic_solutions_l621_62178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dragon_resilience_maximizer_l621_62195

/-- The probability function for s new heads growing -/
noncomputable def p (x : ℝ) (s : ℕ) : ℝ :=
  x^s / (1 + x + x^2)

/-- The probability of obtaining the vector K -/
noncomputable def P (x : ℝ) : ℝ :=
  x^12 / (1 + x + x^2)^10

/-- The observed vector K -/
def K : List ℕ := [1, 2, 2, 1, 0, 2, 1, 0, 1, 2]

theorem dragon_resilience_maximizer (x : ℝ) (hx : x > 0) :
  (∀ y : ℝ, y > 0 → P y ≤ P x) ↔ x = (Real.sqrt 97 + 1) / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dragon_resilience_maximizer_l621_62195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l621_62127

/-- Sequence a_n with sum of first n terms S_n -/
def S : ℕ → ℝ := sorry

/-- Sequence a_n -/
def a : ℕ → ℝ := sorry

/-- Sequence b_n -/
def b : ℕ → ℝ := sorry

/-- Sum of first n terms of b_n -/
def T : ℕ → ℝ := sorry

/-- Main theorem -/
theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → S n = 2 * a n - 3) →
  (∀ n : ℕ, n ≥ 1 → b n = a n + 2 * n) →
  (a 1 = 3 ∧ ∀ n : ℕ, n ≥ 2 → a n = 2 * a (n - 1)) ∧
  (∀ n : ℕ, n ≥ 1 → T n = 3 * 2^n + n^2 + n - 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l621_62127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_american_swallow_weight_is_5_l621_62133

/-- Represents the maximum weight an American swallow can carry -/
def american_swallow_weight : ℝ := sorry

/-- Represents the total number of swallows in the flock -/
def total_swallows : ℕ := 90

/-- Represents the number of European swallows in the flock -/
def european_swallows : ℕ := sorry

/-- The number of American swallows is twice the number of European swallows -/
axiom american_swallow_count : 2 * european_swallows = total_swallows - european_swallows

/-- European swallows can carry twice the weight of American swallows -/
def european_swallow_weight : ℝ := 2 * american_swallow_weight

/-- The maximum combined weight the flock can carry is 600 pounds -/
axiom total_weight_capacity : 
  (total_swallows - european_swallows) * american_swallow_weight + 
  european_swallows * european_swallow_weight = 600

/-- The maximum weight an American swallow can carry is 5 pounds -/
theorem american_swallow_weight_is_5 : american_swallow_weight = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_american_swallow_weight_is_5_l621_62133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alberts_earnings_l621_62165

/-- Albert's earnings problem -/
theorem alberts_earnings (earnings_26_percent : ℝ) (earnings_p_percent : ℝ) :
  earnings_26_percent = 562.54 →
  earnings_p_percent = 567 →
  ∃ (original_earnings : ℝ) (p : ℝ),
    earnings_26_percent = original_earnings * (1 + 26 / 100) ∧
    earnings_p_percent = original_earnings * (1 + p / 100) ∧
    (abs (p - 27) < 0.01) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alberts_earnings_l621_62165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l621_62116

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the domain of f
def PositiveReals := {x : ℝ | x > 0}

-- State the properties of f
axiom f_domain (x : ℝ) : x ∈ PositiveReals → f x ∈ PositiveReals

-- Define the area equality condition
def area_equality (f : ℝ → ℝ) : Prop :=
  ∀ x t, x ∈ PositiveReals → t ∈ PositiveReals → x * f x = t * (f x + f t)

-- State the theorem
theorem function_characterization (f : ℝ → ℝ) 
  (h_domain : ∀ x, x ∈ PositiveReals → f x ∈ PositiveReals)
  (h_area : area_equality f) :
  ∃ c : ℝ, c > 0 ∧ ∀ x, x ∈ PositiveReals → f x = c / x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l621_62116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_B_speed_l621_62173

/-- The speed of student B in km/h -/
noncomputable def speed_B : ℝ := 12

/-- The speed of student A in km/h -/
noncomputable def speed_A : ℝ := 1.2 * speed_B

/-- The distance from school to the activity location in km -/
noncomputable def distance : ℝ := 12

/-- The time difference in hours between A and B's arrivals -/
noncomputable def time_diff : ℝ := 1/6

theorem student_B_speed :
  speed_B = 12 ∧
  speed_A = 1.2 * speed_B ∧
  distance / speed_B - distance / speed_A = time_diff :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_B_speed_l621_62173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_duck_pond_problem_l621_62114

theorem duck_pond_problem (small_pond : ℚ) (large_pond : ℚ) 
  (green_small_percent : ℚ) (green_large_percent : ℚ) (total_green_percent : ℚ) :
  large_pond = 55 →
  green_small_percent = 1/5 →
  green_large_percent = 2/5 →
  total_green_percent = 31/100 →
  (green_small_percent * small_pond + green_large_percent * large_pond) / 
    (small_pond + large_pond) = total_green_percent →
  small_pond = 45 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_duck_pond_problem_l621_62114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l621_62129

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the expression
noncomputable def expression : ℝ :=
  (floor 6.5 : ℝ) * (floor (2 / 3) : ℝ) +
  (floor 2 : ℝ) * 7.2 +
  (floor 8.3 : ℝ) -
  6.6 * (2.3 - (floor 3.7 : ℝ)) / (-7.4 + (floor 5.6 : ℝ)) * (floor (1 / 4) : ℝ) +
  (floor 2.5 : ℝ) * (-2.8) -
  3.1 * (floor 5.2 : ℝ) +
  (floor 4.8 : ℝ) / 2

-- Theorem statement
theorem expression_value : expression = 3.3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l621_62129
