import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_bird_count_l879_87929

def bird_problem (initial_birds : ℕ) (land_first : ℕ) (fly_away : ℕ) (land_second : ℕ) (percent_leave : ℚ) (final_return : ℕ) : ℕ :=
  let total_first := initial_birds + land_first
  let after_changes := total_first - fly_away + land_second
  let doubled := 2 * after_changes
  let after_percent_leave := (doubled : ℤ) - Int.floor (percent_leave * (doubled : ℚ))
  (after_percent_leave + final_return).natAbs

theorem final_bird_count :
  bird_problem 12 8 5 3 (1/4) 2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_bird_count_l879_87929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_b_value_l879_87963

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := b / (3 * x - 4)

theorem unique_b_value :
  ∀ b : ℝ, (f b 3 = (f b).invFun (b + 2)) → b = 2 :=
by
  intro b h
  sorry

#check unique_b_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_b_value_l879_87963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_graph_equation_l879_87926

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem translated_graph_equation (x : ℝ) :
  f (x - Real.pi / 6) = -Real.cos (2 * x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_graph_equation_l879_87926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_of_m_minus_n_l879_87952

theorem square_root_of_m_minus_n (m n : ℝ) : 
  (m * 2 - (-1) = 3) → (3 * 2 + n * (-1) = 14) → Real.sqrt (m - n) = 3 ∨ Real.sqrt (m - n) = -3 :=
by
  intro h1 h2
  have m_val : m = 1 := by
    -- Proof for m = 1
    sorry
  have n_val : n = -8 := by
    -- Proof for n = -8
    sorry
  have m_minus_n : m - n = 9 := by
    -- Proof that m - n = 9
    sorry
  have sqrt_eq : Real.sqrt 9 = 3 := by
    -- Proof that sqrt 9 = 3
    sorry
  have neg_sqrt_eq : -Real.sqrt 9 = -3 := by
    -- Proof that -sqrt 9 = -3
    sorry
  -- Conclude that sqrt (m - n) is either 3 or -3
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_of_m_minus_n_l879_87952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rooms_needed_l879_87901

/-- Represents a group of football fans -/
structure FanGroup where
  team : Fin 3
  gender : Bool
  count : Nat

/-- The problem setup -/
structure FootballFanAccommodation where
  groups : List FanGroup
  sum_constraint : List.sum (groups.map (λ g => g.count)) = 100

/-- Calculates the number of rooms needed for a group -/
def roomsNeeded (g : FanGroup) : Nat :=
  (g.count + 2) / 3

/-- Theorem: The maximum number of rooms needed is 37 -/
theorem max_rooms_needed (setup : FootballFanAccommodation) :
  (List.sum (setup.groups.map roomsNeeded)) ≤ 37 := by
  sorry

#check max_rooms_needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rooms_needed_l879_87901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_intersection_length_l879_87904

/-- Represents a triangle with its medians -/
structure TriangleWithMedians where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  O : ℝ × ℝ

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Assertion that P is the midpoint of BC -/
def is_midpoint_BC (t : TriangleWithMedians) : Prop :=
  t.P = ((t.B.1 + t.C.1) / 2, (t.B.2 + t.C.2) / 2)

/-- Assertion that Q is the midpoint of AB -/
def is_midpoint_AB (t : TriangleWithMedians) : Prop :=
  t.Q = ((t.A.1 + t.B.1) / 2, (t.A.2 + t.B.2) / 2)

/-- Assertion that O is on the median AP -/
def O_on_AP (t : TriangleWithMedians) : Prop :=
  ∃ k : ℝ, 0 ≤ k ∧ k ≤ 1 ∧ t.O = (k * t.A.1 + (1 - k) * t.P.1, k * t.A.2 + (1 - k) * t.P.2)

/-- Assertion that O is on the median CQ -/
def O_on_CQ (t : TriangleWithMedians) : Prop :=
  ∃ k : ℝ, 0 ≤ k ∧ k ≤ 1 ∧ t.O = (k * t.C.1 + (1 - k) * t.Q.1, k * t.C.2 + (1 - k) * t.Q.2)

theorem median_intersection_length (t : TriangleWithMedians) 
  (h1 : is_midpoint_BC t)
  (h2 : is_midpoint_AB t)
  (h3 : O_on_AP t)
  (h4 : O_on_CQ t)
  (h5 : distance t.C t.O = 4) :
  distance t.O t.Q = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_intersection_length_l879_87904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_is_four_l879_87915

def is_valid_representation (a b : ℕ) : Prop :=
  2021 * b.factorial = a.factorial ∧ a > b

theorem smallest_difference_is_four :
  ∀ a b : ℕ,
    is_valid_representation a b →
    ∀ a' b' : ℕ,
      is_valid_representation a' b' →
      a + b ≤ a' + b' →
      (a : ℤ) - (b : ℤ) = 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_is_four_l879_87915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_map_scale_approximation_l879_87979

/-- Represents the scale of a map in inches per mile -/
structure MapScale where
  value : ℝ

/-- Calculates the map scale given the map distance, travel time, and average speed -/
noncomputable def calculate_map_scale (map_distance : ℝ) (travel_time : ℝ) (average_speed : ℝ) : MapScale :=
  ⟨map_distance / (travel_time * average_speed)⟩

/-- The map scale is approximately 0.01282 inches per mile -/
theorem map_scale_approximation :
  let map_distance : ℝ := 5  -- inches
  let travel_time : ℝ := 6.5  -- hours
  let average_speed : ℝ := 60  -- miles per hour
  let calculated_scale : MapScale := calculate_map_scale map_distance travel_time average_speed
  ∃ ε > 0, |calculated_scale.value - 0.01282| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_map_scale_approximation_l879_87979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_elevator_trips_l879_87934

def masses : List Nat := [150, 62, 63, 66, 70, 75, 79, 84, 95, 96, 99]
def capacity : Nat := 190

theorem min_elevator_trips :
  ∃ (trips : List (List Nat)),
    (trips.length = 6) ∧
    (trips.join.toFinset = masses.toFinset) ∧
    (∀ t ∈ trips, t.sum ≤ capacity) ∧
    (∀ t ∈ trips, t.length ≤ 2) ∧
    (∀ (trips' : List (List Nat)),
      (trips'.join.toFinset = masses.toFinset) ∧
      (∀ t' ∈ trips', t'.sum ≤ capacity) →
      trips'.length ≥ 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_elevator_trips_l879_87934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_nine_solution_l879_87959

/-- Given a natural number b > 1, convert a list of digits in base b to a natural number. -/
def toNatBase (b : ℕ) (digits : List ℕ) : ℕ := 
  digits.foldr (fun d acc => d + b * acc) 0

/-- The theorem states that 9 is the unique base that satisfies the given equation. -/
theorem base_nine_solution :
  ∃! b : ℕ, b > 1 ∧ 
    toNatBase b [8, 3, 6, 4] + toNatBase b [5, 7, 2, 3] = toNatBase b [1, 4, 1, 8, 7] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_nine_solution_l879_87959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_is_sqrt_14_l879_87966

-- Define the curves C₁ and C₂
noncomputable def C₁ (t : ℝ) : ℝ × ℝ := (1 - (Real.sqrt 2 / 2) * t, 1 + (Real.sqrt 2 / 2) * t)

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := 
  let ρ := Real.sqrt (2 * Real.cos θ + Real.sqrt (4 * (Real.cos θ)^2 + 12))
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the intersection points A and B
noncomputable def A : ℝ × ℝ := C₁ ((Real.sqrt 6 - Real.sqrt 2) / 2)
noncomputable def B : ℝ × ℝ := C₁ (-(Real.sqrt 6 + Real.sqrt 2) / 2)

-- Theorem statement
theorem length_AB_is_sqrt_14 : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_is_sqrt_14_l879_87966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_echo_distance_result_l879_87958

-- Define the given constants
def car_speed : ℝ := 72 -- km/h
def echo_time : ℝ := 4 -- seconds
def sound_speed : ℝ := 340 -- m/s

-- Define the function to calculate the echo distance
def echo_distance : ℝ := by
  -- Convert car speed to m/s
  let car_speed_ms := car_speed * 1000 / 3600
  
  -- Set up the equation: 2x + vt = ct, where x is the distance
  have h : 2 * echo_distance + car_speed_ms * echo_time = sound_speed * echo_time
  
  -- Solve for echo_distance
  sorry

  -- Return the calculated distance
  exact 640

-- State and prove the result
theorem echo_distance_result : echo_distance = 640 := by
  -- Unfold the definition of echo_distance
  unfold echo_distance
  
  -- The proof is completed by the definition
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_echo_distance_result_l879_87958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thirtieth_term_is_59_l879_87988

/-- A sequence where each term is 2 more than the previous -/
def oddSequence : ℕ → ℕ 
  | 0 => 1
  | n + 1 => oddSequence n + 2

/-- Theorem: The 30th term of the sequence is 59 -/
theorem thirtieth_term_is_59 : oddSequence 29 = 59 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thirtieth_term_is_59_l879_87988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_students_with_A_or_B_l879_87910

/-- The fraction of students who received either A's or B's in Mr. Olsen's math department -/
theorem fraction_of_students_with_A_or_B : ℚ := by
  -- Define the constants
  let class1_students : ℕ := 100
  let class1_A_ratio : ℚ := 4/10
  let class1_B_ratio : ℚ := 3/10

  let class2_students : ℕ := 150
  let class2_A_ratio : ℚ := 1/2
  let class2_B_ratio : ℚ := 1/4

  let class3_students : ℕ := 75
  let class3_A_ratio : ℚ := 6/10
  let class3_B_ratio : ℚ := 1/5

  let total_students : ℕ := class1_students + class2_students + class3_students

  -- Calculate the fraction of students with A's or B's
  let fraction_A_or_B : ℚ :=
    ((class1_students : ℚ) * (class1_A_ratio + class1_B_ratio) +
     (class2_students : ℚ) * (class2_A_ratio + class2_B_ratio) +
     (class3_students : ℚ) * (class3_A_ratio + class3_B_ratio)) /
    (total_students : ℚ)

  -- Prove that the fraction is equal to 97/130
  have fraction_equals_97_130 : fraction_A_or_B = 97/130 := by
    -- The actual proof would go here
    sorry

  -- Return the result
  exact 97/130


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_students_with_A_or_B_l879_87910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_side_length_l879_87998

/-- The side length of an equilateral triangle formed by two internally touching circles -/
noncomputable def triangle_side_length (r R : ℝ) : ℝ :=
  (r * R * Real.sqrt 3) / Real.sqrt (r^2 - r*R + R^2)

/-- Theorem stating the side length of the equilateral triangle -/
theorem equilateral_triangle_side_length
  (r R : ℝ)
  (h_positive : r > 0 ∧ R > 0)
  (h_touch : R > r) :
  ∃ (a : ℝ),
    a > 0 ∧
    a = triangle_side_length r R ∧
    ∃ (A B M : ℝ × ℝ),
      (A.1 - M.1)^2 + (A.2 - M.2)^2 = R^2 ∧
      (B.1 - M.1)^2 + (B.2 - M.2)^2 = r^2 ∧
      (A.1 - B.1)^2 + (A.2 - B.2)^2 = a^2 ∧
      (A.1 - M.1)^2 + (A.2 - M.2)^2 = a^2 ∧
      (B.1 - M.1)^2 + (B.2 - M.2)^2 = a^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_side_length_l879_87998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l879_87921

-- Define the triangle ABC and points D, E, F
structure Triangle :=
  (A B C : ℝ × ℝ)

noncomputable def D (t : Triangle) : ℝ × ℝ := (2/3 * t.B.1 + 1/3 * t.C.1, 2/3 * t.B.2 + 1/3 * t.C.2)
noncomputable def E (t : Triangle) : ℝ × ℝ := (1/3 * t.A.1 + 2/3 * t.C.1, 1/3 * t.A.2 + 2/3 * t.C.2)
noncomputable def F (t : Triangle) : ℝ × ℝ := (2/3 * t.A.1 + 1/3 * t.B.1, 2/3 * t.A.2 + 1/3 * t.B.2)

-- Define the intersection points P, Q, R
noncomputable def P (t : Triangle) : ℝ × ℝ := sorry
noncomputable def Q (t : Triangle) : ℝ × ℝ := sorry
noncomputable def R (t : Triangle) : ℝ × ℝ := sorry

-- Define the area of a triangle given its vertices
noncomputable def area (a b c : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_ratio (t : Triangle) :
  area (P t) (Q t) (R t) / area t.A t.B t.C = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l879_87921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l879_87950

/-- For n ≥ 2, the polynomial P(x) = x^n + a * x^(n-2) is divisible by (x - 2) if and only if a = -4 -/
theorem polynomial_divisibility (n : ℕ) (a : ℝ) (h : n ≥ 2) :
  (∃ Q : Polynomial ℝ, X^n + a • X^(n-2) = (X - 2) * Q) ↔ a = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l879_87950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_l879_87913

theorem trigonometric_inequality : 
  Real.tan (55 * π / 180) > Real.sin (55 * π / 180) ∧ 
  Real.sin (55 * π / 180) > Real.cos (55 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_l879_87913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_positive_integer_root_l879_87941

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Add this case for 0
  | 1 => 1
  | (n + 2) => (1/2) * sequence_a (n + 1) + 1 / (4 * sequence_a (n + 1))

theorem existence_of_positive_integer_root (n : ℕ) (h : n > 1) :
  ∃ b : ℕ+, (b : ℚ)^2 = 2 / (2 * sequence_a n^2 - 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_positive_integer_root_l879_87941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dragon_defeat_possible_l879_87977

/-- Represents the three heroes --/
inductive Hero
| Ilya
| Dobrynya
| Alyosha

/-- The number of heads removed by each hero's strike --/
def headsRemoved (hero : Hero) (heads : Nat) : Nat :=
  match hero with
  | Hero.Ilya => heads / 2 + 1
  | Hero.Dobrynya => heads / 3 + 2
  | Hero.Alyosha => heads / 4 + 3

/-- Predicate to check if a strike is valid (results in an integer number of heads) --/
def isValidStrike (hero : Hero) (heads : Nat) : Prop :=
  heads - headsRemoved hero heads ≥ 0

/-- Helper function to calculate remaining heads after a sequence of strikes --/
def remainingHeads (initial : Nat) (sequence : List Hero) : Nat :=
  sequence.foldl (fun heads hero => heads - headsRemoved hero heads) initial

/-- The main theorem to prove --/
theorem dragon_defeat_possible : ∃ (sequence : List Hero), 
  remainingHeads 41 sequence = 0 ∧ 
  ∀ (i : Nat) (hero : Hero), i < sequence.length → 
    isValidStrike hero (remainingHeads 41 (sequence.take i)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dragon_defeat_possible_l879_87977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_household_income_savings_correlation_l879_87954

/-- Represents the sample data for households -/
structure SampleData where
  n : ℕ
  sum_x : ℝ
  sum_y : ℝ
  sum_xy : ℝ
  sum_x_sq : ℝ

/-- Calculates the slope of the regression line -/
noncomputable def calculate_slope (data : SampleData) : ℝ :=
  (data.sum_xy - (1 / data.n) * data.sum_x * data.sum_y) /
  (data.sum_x_sq - (1 / data.n) * data.sum_x ^ 2)

/-- Calculates the y-intercept of the regression line -/
noncomputable def calculate_intercept (data : SampleData) (slope : ℝ) : ℝ :=
  (1 / data.n) * data.sum_y - slope * (1 / data.n) * data.sum_x

/-- Predicts the y value for a given x using the regression line -/
noncomputable def predict_y (slope : ℝ) (intercept : ℝ) (x : ℝ) : ℝ :=
  slope * x + intercept

/-- Main theorem statement -/
theorem household_income_savings_correlation (data : SampleData)
  (h_n : data.n = 10)
  (h_sum_x : data.sum_x = 80)
  (h_sum_y : data.sum_y = 20)
  (h_sum_xy : data.sum_xy = 184)
  (h_sum_x_sq : data.sum_x_sq = 720) :
  let slope := calculate_slope data
  let intercept := calculate_intercept data slope
  (slope > 0 ∧ predict_y slope intercept 7 = 1.7) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_household_income_savings_correlation_l879_87954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_borya_grisha_different_colors_l879_87961

-- Define the boys and colors
inductive Boy : Type
| Anton | Borya | Vova | Grisha | Dima

inductive Color : Type
| Gray | Brown | Raspberry

-- Define the competitions
inductive Competition : Type
| Buuz | Khinkali | Dumplings

-- Define a function to assign pants color to each boy
variable (pants : Boy → Color)

-- Define a function to determine the winner of each competition
variable (winner : Competition → Boy)

-- Define the conditions
axiom different_colors : ∀ (b1 b2 : Boy), b1 ≠ b2 → pants b1 ≠ pants b2
axiom gray_first : ∀ (c : Competition), pants (winner c) = Color.Gray
axiom brown_second : ∀ (c : Competition), ∃ (b : Boy), pants b = Color.Brown ∧ b ≠ winner c
axiom raspberry_third : ∀ (c : Competition), ∃ (b : Boy), pants b = Color.Raspberry ∧ b ≠ winner c ∧ pants b ≠ Color.Brown
axiom anton_last_buuz : winner Competition.Buuz ≠ Boy.Anton
axiom dima_last_khinkali : winner Competition.Khinkali ≠ Boy.Dima
axiom vova_last_dumplings : winner Competition.Dumplings ≠ Boy.Vova

-- Theorem to prove
theorem borya_grisha_different_colors : pants Boy.Borya ≠ pants Boy.Grisha := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_borya_grisha_different_colors_l879_87961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_reciprocal_sum_constant_l879_87964

/-- A parabola defined by y = x^2 + px + q --/
structure Parabola where
  p : ℝ
  q : ℝ

/-- A point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- A chord of a parabola --/
structure Chord where
  parabola : Parabola
  start : Point
  endPoint : Point  -- Changed 'end' to 'endPoint' to avoid reserved keyword

/-- The distance between two points --/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: There exists a constant r such that for all chords AB of the parabola 
    y = x^2 + px + q passing through the point C = (a, b), the sum of the reciprocals 
    of the distances from C to A and B is equal to r --/
theorem chord_reciprocal_sum_constant 
  (parabola : Parabola) 
  (C : Point) 
  (h : C.y = C.x^2 + parabola.p * C.x + parabola.q) : 
  ∃ (r : ℝ), ∀ (chord : Chord), 
    chord.parabola = parabola → 
    chord.start.y = chord.start.x^2 + parabola.p * chord.start.x + parabola.q →
    chord.endPoint.y = chord.endPoint.x^2 + parabola.p * chord.endPoint.x + parabola.q →
    (∃ (t : ℝ), C.x = t * chord.start.x + (1 - t) * chord.endPoint.x ∧ 
                C.y = t * chord.start.y + (1 - t) * chord.endPoint.y) →
    1 / distance C chord.start + 1 / distance C chord.endPoint = r :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_reciprocal_sum_constant_l879_87964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_equals_2_sqrt_41_l879_87994

noncomputable section

/-- The distance between two points in a 2D plane -/
def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The total distance traveled from (-3, 6) to (6, -3) via (2, 2) -/
def totalDistance : ℝ :=
  distance (-3) 6 2 2 + distance 2 2 6 (-3)

theorem total_distance_equals_2_sqrt_41 :
  totalDistance = 2 * Real.sqrt 41 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_equals_2_sqrt_41_l879_87994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l879_87930

-- Define the sets A and B
def A : Set ℝ := {x | Real.rpow 2 x > 1}
def B : Set ℝ := {x | x^2 - 3*x - 4 > 0}

-- State the theorem
theorem intersection_A_B : A ∩ B = {x : ℝ | x > 4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l879_87930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l879_87985

/-- Representation of a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Definition of the line y = kx + 3 -/
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 3

/-- Definition of the circle (x-1)^2 + (y-2)^2 = 4 -/
def onCircle (p : Point) : Prop :=
  (p.x - 1)^2 + (p.y - 2)^2 = 4

/-- Theorem statement -/
theorem line_circle_intersection (k : ℝ) :
  ∃ (M N : Point),
    onCircle M ∧ onCircle N ∧
    M.y = line k M.x ∧ N.y = line k N.x ∧
    distance M N ≥ 2 * Real.sqrt 3 →
    k ∈ Set.Iic 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l879_87985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_calculations_l879_87953

theorem sqrt_calculations : 
  (Real.sqrt 27 - Real.sqrt 2 * Real.sqrt 6 + 3 * Real.sqrt (1/3) = 2 * Real.sqrt 3) ∧
  ((Real.sqrt 5 + Real.sqrt 2) * (Real.sqrt 5 - Real.sqrt 2) + (Real.sqrt 3 - 1)^2 = 7 - 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_calculations_l879_87953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rank_sum_iff_zero_product_l879_87923

open Matrix

theorem rank_sum_iff_zero_product {n : ℕ} (A B : Matrix (Fin n) (Fin n) ℝ) :
  rank A + rank B ≤ n ↔
  ∃ X : Matrix (Fin n) (Fin n) ℝ, IsUnit X ∧ A * X * B = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rank_sum_iff_zero_product_l879_87923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_probabilities_l879_87962

/-- A box containing red and white balls -/
structure Box where
  total : Nat
  red : Nat
  white : Nat
  h_total : total = red + white

/-- The probability of an event -/
noncomputable def probability (n m : Nat) : Real := (n : Real) / (m : Real)

/-- The box described in the problem -/
def problem_box : Box := {
  total := 6,
  red := 4,
  white := 2,
  h_total := by rfl
}

theorem problem_probabilities (box : Box) 
  (h_box : box = problem_box) : 
  probability box.red box.total = 2/3 ∧ 
  probability box.red (box.total - 1) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_probabilities_l879_87962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_figure_l879_87989

-- Define the bounds of the figure
noncomputable def lower_x_bound : ℝ := -Real.pi/2
noncomputable def upper_x_bound : ℝ := 5*Real.pi/4

-- Define the function representing the curve
noncomputable def f (x : ℝ) : ℝ := Real.sin x

-- Define the area of the figure
noncomputable def area : ℝ := ∫ x in lower_x_bound..upper_x_bound, |f x|

-- Theorem statement
theorem area_of_bounded_figure :
  area = 4 - Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_figure_l879_87989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_failing_students_percentage_l879_87946

noncomputable def current_failing_percentage (n : ℕ) : ℚ := 
  (24 : ℚ) / 100 + 1 / n

theorem failing_students_percentage 
  (n : ℕ) 
  (h1 : n > 0) 
  (h2 : (24 : ℚ) / 100 * n = (25 : ℚ) / 100 * (n - 1)) :
  current_failing_percentage n = 28 / 100 :=
by
  sorry

#check failing_students_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_failing_students_percentage_l879_87946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_derivative_f_l879_87920

noncomputable def f (x : ℝ) := (1/3) * x^3 - x^2 + 8

theorem min_derivative_f :
  ∃ (c : ℝ), c ∈ Set.Icc 0 5 ∧
  ∀ (x : ℝ), x ∈ Set.Icc 0 5 → deriv f c ≤ deriv f x ∧
  deriv f c = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_derivative_f_l879_87920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_year_price_is_correct_optimal_purchase_is_correct_l879_87932

/-- Represents the bicycle shop's sales and pricing data -/
structure BicycleShop where
  last_year_total_sales : ℝ
  price_decrease : ℝ
  sales_decrease_percentage : ℝ
  total_new_units : ℕ
  type_a_purchase_price : ℝ
  type_b_purchase_price : ℝ
  type_b_selling_price : ℝ

/-- Calculates the selling price of type A bicycles last year -/
noncomputable def last_year_price (shop : BicycleShop) : ℝ :=
  shop.last_year_total_sales / (shop.last_year_total_sales * (1 - shop.sales_decrease_percentage) / (shop.last_year_total_sales / shop.last_year_total_sales - shop.price_decrease))

/-- Calculates the optimal number of type A bicycles to purchase -/
noncomputable def optimal_type_a_purchase (shop : BicycleShop) : ℕ :=
  Nat.ceil ((shop.total_new_units : ℝ) / 3)

/-- Theorem stating the correct selling price of type A bicycles last year -/
theorem last_year_price_is_correct (shop : BicycleShop) 
  (h1 : shop.last_year_total_sales = 80000)
  (h2 : shop.price_decrease = 200)
  (h3 : shop.sales_decrease_percentage = 0.1) :
  last_year_price shop = 2000 := by sorry

/-- Theorem stating the optimal purchase quantities for type A and B bicycles -/
theorem optimal_purchase_is_correct (shop : BicycleShop)
  (h1 : shop.total_new_units = 60)
  (h2 : shop.type_a_purchase_price = 1500)
  (h3 : shop.type_b_purchase_price = 1800)
  (h4 : shop.type_b_selling_price = 2400) :
  optimal_type_a_purchase shop = 20 ∧ 
  shop.total_new_units - optimal_type_a_purchase shop = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_year_price_is_correct_optimal_purchase_is_correct_l879_87932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_canoe_weight_theorem_l879_87909

/-- Represents the weight capacity and occupants of a canoe trip --/
structure CanoeTrip where
  max_people : ℕ
  adult_weight : ℚ
  child_weight_ratio : ℚ
  dog_weight_ratio : ℚ
  cat1_weight_ratio : ℚ
  cat2_weight_ratio : ℚ
  num_adults : ℕ
  num_children : ℕ
  has_dog : Bool
  has_cats : Bool

/-- Calculates the total weight and maximum weight limit for a canoe trip --/
def canoe_weight_calculation (trip : CanoeTrip) : ℚ × ℚ :=
  let total_weight := 
    trip.adult_weight * trip.num_adults +
    (trip.adult_weight * trip.child_weight_ratio) * trip.num_children +
    (if trip.has_dog then trip.adult_weight * trip.dog_weight_ratio else 0) +
    (if trip.has_cats then 
      trip.adult_weight * trip.cat1_weight_ratio + 
      trip.adult_weight * trip.cat2_weight_ratio 
    else 0)
  let max_weight := trip.adult_weight * trip.max_people
  (total_weight, max_weight)

theorem canoe_weight_theorem (trip : CanoeTrip) 
  (h1 : trip.max_people = 8)
  (h2 : trip.adult_weight = 150)
  (h3 : trip.child_weight_ratio = 1/2)
  (h4 : trip.dog_weight_ratio = 1/3)
  (h5 : trip.cat1_weight_ratio = 1/10)
  (h6 : trip.cat2_weight_ratio = 1/8)
  (h7 : trip.num_adults = 4)
  (h8 : trip.num_children = 2)
  (h9 : trip.has_dog = true)
  (h10 : trip.has_cats = true) :
  let (total_weight, max_weight) := canoe_weight_calculation trip
  total_weight = 833.75 ∧ max_weight = 1200 := by
  sorry

#eval canoe_weight_calculation {
  max_people := 8,
  adult_weight := 150,
  child_weight_ratio := 1/2,
  dog_weight_ratio := 1/3,
  cat1_weight_ratio := 1/10,
  cat2_weight_ratio := 1/8,
  num_adults := 4,
  num_children := 2,
  has_dog := true,
  has_cats := true
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_canoe_weight_theorem_l879_87909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_minus_pi_over_four_l879_87935

theorem tan_alpha_minus_pi_over_four 
  (α : Real) 
  (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : Real.sin α = 3/5) : 
  (α ∈ Set.Ioo 0 (π/2) → Real.tan (α - π/4) = -1/7) ∧ 
  (α ∈ Set.Ioo (π/2) π → Real.tan (α - π/4) = -7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_minus_pi_over_four_l879_87935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l879_87990

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^3 / (x^3 + 2*y^2 * Real.sqrt (z*x)) +
  y^3 / (y^3 + 2*z^2 * Real.sqrt (x*y)) +
  z^3 / (z^3 + 2*x^2 * Real.sqrt (y*z)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l879_87990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_revenue_is_correct_l879_87948

-- Define the pencil types and their prices
structure PencilType where
  name : String
  price : Float
deriving Inhabited

-- Define the store's inventory and sales
structure StoreInventory where
  pencilTypes : List PencilType
  salesQuantities : List Nat
  eraserPencilDiscount : Float
  mechanicalPencilOffer : Nat

-- Calculate the total revenue
def calculateRevenue (inventory : StoreInventory) : Float :=
  let regularSales := List.sum (List.zipWith (fun p q => p.price * q.toFloat) inventory.pencilTypes inventory.salesQuantities)
  let eraserPencilSales := 
    if (List.head! inventory.salesQuantities) ≥ 100
    then (List.head! inventory.pencilTypes).price * (List.head! inventory.salesQuantities).toFloat * (1 - inventory.eraserPencilDiscount)
    else (List.head! inventory.pencilTypes).price * (List.head! inventory.salesQuantities).toFloat
  let mechanicalPencilSales := 
    (List.get! inventory.pencilTypes 3).price * ((List.get! inventory.salesQuantities 3).toFloat * (inventory.mechanicalPencilOffer.toFloat - 1) / inventory.mechanicalPencilOffer.toFloat)
  regularSales - (List.head! inventory.pencilTypes).price * (List.head! inventory.salesQuantities).toFloat + eraserPencilSales + mechanicalPencilSales

-- Theorem statement
theorem store_revenue_is_correct (inventory : StoreInventory) : 
  inventory.pencilTypes = [
    { name := "Eraser Pencil", price := 0.8 },
    { name := "Regular Pencil", price := 0.5 },
    { name := "Short Pencil", price := 0.4 },
    { name := "Mechanical Pencil", price := 1.2 },
    { name := "Novelty Pencil", price := 1.5 }
  ] ∧
  inventory.salesQuantities = [200, 40, 35, 25, 15] ∧
  inventory.eraserPencilDiscount = 0.1 ∧
  inventory.mechanicalPencilOffer = 4 →
  calculateRevenue inventory = 224.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_revenue_is_correct_l879_87948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l879_87986

theorem sin_minus_cos_value (α : ℝ) 
  (h1 : -π/2 < α) (h2 : α < 0) (h3 : Real.sin α + Real.cos α = 1/5) : 
  Real.sin α - Real.cos α = -7/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l879_87986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_connected_pieces_formula_l879_87907

/-- The number of different connected pieces of stamps obtained from a 2 × n block of 2n different stamps after tearing off 0 to 2n-1 stamps. -/
noncomputable def connected_pieces (n : ℕ) : ℝ :=
  (1/4) * ((1 + Real.sqrt 2)^(n+3) + (1 - Real.sqrt 2)^(n+3)) - 2*n - 7/2

/-- Theorem stating the number of different connected pieces of stamps obtained from a 2 × n block of 2n different stamps after tearing off 0 to 2n-1 stamps. -/
theorem connected_pieces_formula (n : ℕ) :
  connected_pieces n = (1/4) * ((1 + Real.sqrt 2)^(n+3) + (1 - Real.sqrt 2)^(n+3)) - 2*n - 7/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_connected_pieces_formula_l879_87907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_ratio_sum_l879_87991

theorem trig_ratio_sum (a b : ℝ) 
  (h1 : Real.sin a / Real.sin b = 2) 
  (h2 : Real.cos a / Real.cos b = 3) : 
  Real.sin (3 * a) / Real.sin (3 * b) + Real.cos (3 * a) / Real.cos (3 * b) = 29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_ratio_sum_l879_87991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_system_solution_l879_87933

/-- Given real numbers a and b satisfying certain conditions, 
    the system of equations has a unique solution with distinct positive real numbers. -/
theorem equation_system_solution (a b : ℝ) 
    (h1 : 0 < a) (h2 : 0 < b) (h3 : b < a) (h4 : a < Real.sqrt 3 * b) :
  ∃! (x y z : ℝ), 
    x + y + z = a ∧ 
    x^2 + y^2 + z^2 = b^2 ∧ 
    x * y = z^2 ∧
    0 < x ∧ 0 < y ∧ 0 < z ∧
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    ((x = (a^2 + b^2 + Real.sqrt ((3*a^2 - b^2)*(3*b^2 - a^2))) / (4*a) ∧
      y = (a^2 + b^2 - Real.sqrt ((3*a^2 - b^2)*(3*b^2 - a^2))) / (4*a)) ∨
     (y = (a^2 + b^2 + Real.sqrt ((3*a^2 - b^2)*(3*b^2 - a^2))) / (4*a) ∧
      x = (a^2 + b^2 - Real.sqrt ((3*a^2 - b^2)*(3*b^2 - a^2))) / (4*a))) ∧
    z = (a^2 - b^2) / (2*a) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_system_solution_l879_87933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_half_area_l879_87912

-- Define a convex centrally symmetric polygon
structure ConvexCentrallySymmetricPolygon where
  -- Add necessary fields and properties
  area : ℝ
  is_convex : Bool
  is_centrally_symmetric : Bool

-- Define a rhombus
structure Rhombus where
  -- Add necessary fields
  area : ℝ

-- Helper functions (defined as Props)
def point_in_rhombus (r : Rhombus) (point : ℝ × ℝ) : Prop :=
  sorry

def point_in_polygon (p : ConvexCentrallySymmetricPolygon) (point : ℝ × ℝ) : Prop :=
  sorry

-- Theorem statement
theorem rhombus_half_area (p : ConvexCentrallySymmetricPolygon) :
  ∃ (r : Rhombus), r.area = p.area / 2 ∧ 
  (∀ (point : ℝ × ℝ), point_in_rhombus r point → point_in_polygon p point) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_half_area_l879_87912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_side_length_l879_87908

/-- A triangle with side lengths -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- Two triangles are similar if their corresponding sides are proportional -/
def similar_triangles (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ 
    t1.side1 = k * t2.side1 ∧
    t1.side2 = k * t2.side2 ∧
    t1.side3 = k * t2.side3

theorem similar_triangles_side_length 
  (GHI JKL : Triangle)
  (h_similar : similar_triangles GHI JKL)
  (h_GH : GHI.side1 = 10)
  (h_HI : GHI.side2 = 7)
  (h_JK : JKL.side1 = 4) :
  JKL.side2 = 2.8 := by
  sorry

#check similar_triangles_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_side_length_l879_87908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hillary_descending_rate_is_1000_l879_87945

/-- The climbing scenario on Mt. Everest -/
structure ClimbingScenario where
  base_camp_distance : ℝ
  hillary_climb_rate : ℝ
  eddy_climb_rate : ℝ
  hillary_stop_distance : ℝ
  start_time : ℝ
  pass_time : ℝ

/-- Calculate Hillary's descending rate based on the given scenario -/
noncomputable def hillary_descending_rate (scenario : ClimbingScenario) : ℝ :=
  let hillary_climb_distance := scenario.base_camp_distance - scenario.hillary_stop_distance
  let hillary_climb_time := hillary_climb_distance / scenario.hillary_climb_rate
  let eddy_climb_distance := scenario.eddy_climb_rate * (scenario.pass_time - scenario.start_time)
  let hillary_descend_distance := hillary_climb_distance - eddy_climb_distance
  let hillary_descend_time := scenario.pass_time - (scenario.start_time + hillary_climb_time)
  hillary_descend_distance / hillary_descend_time

/-- Theorem stating that Hillary's descending rate is 1000 ft/hr given the specific scenario -/
theorem hillary_descending_rate_is_1000 (scenario : ClimbingScenario) 
  (h1 : scenario.base_camp_distance = 5000)
  (h2 : scenario.hillary_climb_rate = 800)
  (h3 : scenario.eddy_climb_rate = 500)
  (h4 : scenario.hillary_stop_distance = 1000)
  (h5 : scenario.start_time = 6)
  (h6 : scenario.pass_time = 12) :
  hillary_descending_rate scenario = 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hillary_descending_rate_is_1000_l879_87945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_three_extreme_points_range_f_max_m_inequality_l879_87928

-- Define the function f
noncomputable def f (t : ℝ) (x : ℝ) : ℝ := (x^3 - 6*x^2 + 3*x + t) * Real.exp x

-- Part I: Range of t for which f has three extreme points
theorem f_three_extreme_points_range :
  ∀ t : ℝ, (∃ a b c : ℝ, a < b ∧ b < c ∧
    (∀ x : ℝ, x ≠ a ∧ x ≠ b ∧ x ≠ c → (deriv (f t)) x ≠ 0)) ↔
  -8 < t ∧ t < 24 :=
sorry

-- Part II: Maximum m for which f(x) ≤ x holds on [1, m]
theorem f_max_m_inequality (t : ℝ) (h : t ∈ Set.Icc 0 2) :
  (∃ m : ℕ, m > 0 ∧ ∀ x : ℝ, x ∈ Set.Icc 1 (Real.mk m) → f t x ≤ x) ∧
  (∀ n : ℕ, n > 5 → ∃ x : ℝ, x ∈ Set.Icc 1 (Real.mk n) ∧ f t x > x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_three_extreme_points_range_f_max_m_inequality_l879_87928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_column_possibilities_l879_87911

/-- Represents a column in the modified SHORT BINGO card -/
structure BingoColumn where
  range : Finset Nat
  void_position : Nat

/-- The second column of the modified SHORT BINGO card -/
def second_column : BingoColumn :=
  { range := Finset.filter (fun n => 11 ≤ n ∧ n ≤ 16) (Finset.range 17),
    void_position := 2 }

/-- The number of distinct possibilities for the values in a BINGO column -/
def column_possibilities (col : BingoColumn) : Nat :=
  (col.range.card - 1) * (col.range.card - 2)

theorem second_column_possibilities :
  column_possibilities second_column = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_column_possibilities_l879_87911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_fraction_in_cup1_is_one_fourth_l879_87919

/-- Represents a cup with a capacity and current contents of tea and milk -/
structure Cup where
  capacity : ℝ
  tea : ℝ
  milk : ℝ

/-- The process of transferring liquid between cups -/
def transfer (cup1 cup2 : Cup) (fraction : ℝ) : Cup × Cup :=
  sorry

/-- The final state of Cup 1 after the transfers -/
def finalCup1State (initialCup1 initialCup2 : Cup) : Cup :=
  sorry

/-- Theorem stating that the fraction of milk in Cup 1 after transfers is 1/4 -/
theorem milk_fraction_in_cup1_is_one_fourth 
  (cup1 cup2 : Cup)
  (h1 : cup1.capacity = 12)
  (h2 : cup2.capacity = 12)
  (h3 : cup1.tea = 6)
  (h4 : cup1.milk = 0)
  (h5 : cup2.tea = 0)
  (h6 : cup2.milk = 6) :
  let finalCup1 := finalCup1State cup1 cup2
  (finalCup1.milk / (finalCup1.tea + finalCup1.milk)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_fraction_in_cup1_is_one_fourth_l879_87919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_theorem_l879_87957

/-- Represents the number of squares with side length 1/3^n after n iterations --/
def num_squares (n : ℕ) : ℕ := 8^n

/-- Represents the sum of areas of removed squares after n iterations --/
noncomputable def sum_removed_areas (n : ℕ) : ℝ := 1 - (8/9)^n

/-- Theorem stating the properties of the square division problem --/
theorem square_division_theorem :
  ∀ n : ℕ,
  (∃ k : ℕ, num_squares n = 8^n) ∧
  (∀ ε > 0, ∃ N : ℕ, ∀ m ≥ N, |sum_removed_areas m - 1| < ε) :=
by
  intro n
  constructor
  · use n
    rfl
  · intro ε hε
    sorry -- Proof of limit omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_theorem_l879_87957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_profit_percentage_l879_87944

/-- Calculates the profit percentage on the original price of a car given specific buying and selling conditions -/
theorem car_profit_percentage (P : ℝ) (h : P > 0) : 
  let discounted_price := 0.9 * P
  let taxes_and_expenses := 0.05 * discounted_price
  let total_cost := discounted_price + taxes_and_expenses
  let selling_price := total_cost + 0.8 * total_cost
  let profit := selling_price - P
  (profit / P) * 100 = 70.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_profit_percentage_l879_87944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_commercial_arrangements_l879_87916

/-- The number of different commercial advertisements -/
def num_commercial_ads : ℕ := 4

/-- The number of different public service advertisements -/
def num_public_service_ads : ℕ := 2

/-- The total number of commercials in the broadcast -/
def total_commercials : ℕ := 6

/-- The factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- The number of ways to arrange the commercials with the given constraints -/
def num_arrangements : ℕ := num_public_service_ads * factorial num_commercial_ads

theorem commercial_arrangements :
  num_arrangements = 48 :=
by
  rw [num_arrangements, num_public_service_ads, num_commercial_ads]
  rw [factorial]
  norm_num
  rfl

#eval num_arrangements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_commercial_arrangements_l879_87916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l879_87978

open InnerProductSpace

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_sum_magnitude 
  (a b : V) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (norm_a : ‖a‖ = Real.sqrt 7 + 1) 
  (norm_b : ‖b‖ = Real.sqrt 7 - 1) 
  (norm_diff : ‖a - b‖ = 4) : 
  ‖a + b‖ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l879_87978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_b_lt_two_thirds_l879_87975

noncomputable def a : ℕ → ℝ
  | 0 => 2
  | n + 1 => if a n < Real.sqrt 3 then (a n)^2 else (a n)^2 / 3

noncomputable def b : ℕ → ℝ
  | 0 => 0
  | n + 1 => if a n < Real.sqrt 3 then 0 else 1 / (2^(n+1))

theorem sum_b_lt_two_thirds :
  (Finset.range 2020).sum (fun i => b (i + 1)) < 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_b_lt_two_thirds_l879_87975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_equality_iff_congruence_l879_87955

theorem remainder_equality_iff_congruence (a b : ℤ) (d : ℕ) (h : d ≠ 0) :
  (a % d = b % d) ↔ a ≡ b [ZMOD d] :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_equality_iff_congruence_l879_87955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_80_degrees_l879_87992

theorem tan_80_degrees (k : ℝ) (h : Real.cos (-80 * π / 180) = k) :
  Real.tan (80 * π / 180) = Real.sqrt (1 - k^2) / k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_80_degrees_l879_87992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_composite_sum_l879_87956

theorem odd_composite_sum (n : ℕ) (h_odd : Odd n) (h_composite : ¬Prime n) :
  ∃ (k m : ℕ), m ≥ 3 ∧ n = m * (2 * k + m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_composite_sum_l879_87956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base4_repeating_decimal_sum_l879_87927

/-- Represents a base 4 repeating decimal as a rational number -/
def base4RepeatingDecimalToRational (d : ℚ) : Prop :=
  ∃ (a b : ℕ), 
    0 < a ∧ 0 < b ∧
    Nat.Coprime a b ∧
    d = (a : ℚ) / (b : ℚ) ∧
    d = (1 : ℚ) / 4 + (2 : ℚ) / 16 + (1 : ℚ) / 64 + (2 : ℚ) / 256 + (1 : ℚ) / 1024 + (2 : ℚ) / 4096

/-- The main theorem -/
theorem base4_repeating_decimal_sum (d : ℚ) : 
  base4RepeatingDecimalToRational d → 
  ∃ (a b : ℕ), d = (a : ℚ) / (b : ℚ) ∧ Nat.Coprime a b ∧ a + b = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_base4_repeating_decimal_sum_l879_87927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equal_isosceles_sin_cos_inequality_obtuse_tan_sum_positive_acute_l879_87972

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  angle_sum : A + B + C = Real.pi

-- Define isosceles, obtuse, and acute triangles
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.C = t.A

def Triangle.isObtuse (t : Triangle) : Prop :=
  t.A > Real.pi/2 ∨ t.B > Real.pi/2 ∨ t.C > Real.pi/2

def Triangle.isAcute (t : Triangle) : Prop :=
  t.A < Real.pi/2 ∧ t.B < Real.pi/2 ∧ t.C < Real.pi/2

-- State the theorems
theorem sin_equal_isosceles (t : Triangle) :
  Real.sin t.A = Real.sin t.B → t.isIsosceles := by
  sorry

theorem sin_cos_inequality_obtuse (t : Triangle) :
  Real.sin t.A ^ 2 + Real.sin t.B ^ 2 + Real.cos t.C ^ 2 < 1 → t.isObtuse := by
  sorry

theorem tan_sum_positive_acute (t : Triangle) :
  Real.tan t.A + Real.tan t.B + Real.tan t.C > 0 → t.isAcute := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equal_isosceles_sin_cos_inequality_obtuse_tan_sum_positive_acute_l879_87972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_three_similar_parts_l879_87999

theorem impossible_three_similar_parts :
  ∀ (x : ℝ), x > 0 →
  ¬∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  x = a + b + c ∧
  (a ≤ Real.sqrt 2 * b ∧ b ≤ Real.sqrt 2 * a) ∧
  (a ≤ Real.sqrt 2 * c ∧ c ≤ Real.sqrt 2 * a) ∧
  (b ≤ Real.sqrt 2 * c ∧ c ≤ Real.sqrt 2 * b) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_three_similar_parts_l879_87999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l879_87940

/-- Given a point P(x, y) on the unit circle such that x + y = -1/5,
    prove that tan(α + π/4) = ±1/7, where α is the angle formed by
    the positive x-axis and the line from the origin to P. -/
theorem tan_alpha_plus_pi_fourth (x y : ℝ) (α : ℝ) :
  x^2 + y^2 = 1 →
  x + y = -1/5 →
  (∃ θ : ℝ, x = Real.cos θ ∧ y = Real.sin θ ∧ α = θ) →
  Real.tan (α + π/4) = 1/7 ∨ Real.tan (α + π/4) = -1/7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l879_87940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_sum_five_l879_87974

/-- An arithmetic progression with its properties -/
structure ArithmeticProgression where
  a : ℕ → ℚ  -- The sequence (using rationals instead of reals)
  d : ℚ      -- Common difference
  diff_prop : ∀ n, a (n + 1) - a n = d

/-- Sum of first n terms of an arithmetic progression -/
def sum_n (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  n * (2 * ap.a 1 + (n - 1) * ap.d) / 2

/-- Main theorem -/
theorem arithmetic_progression_sum_five 
  (ap : ArithmeticProgression) 
  (h1 : ap.a 2 * ap.a 5 = 3 * ap.a 3)
  (h2 : ap.a 4 - 9 * ap.a 7 = 2) :
  ∃ (S : ℚ), sum_n ap 5 = S := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_sum_five_l879_87974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l879_87936

-- Define the equation
def equation (x : ℂ) : Prop := (121 : ℂ) * x^2 + 54 = 0

-- Define the solutions
noncomputable def solution_pos : ℂ := (3 * Complex.I * Real.sqrt 6) / 11
noncomputable def solution_neg : ℂ := -(3 * Complex.I * Real.sqrt 6) / 11

-- Theorem statement
theorem equation_solutions :
  equation solution_pos ∧ equation solution_neg := by
  sorry

#check equation_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l879_87936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_four_theta_value_l879_87967

theorem cos_four_theta_value (θ : ℝ) :
  (∑' n : ℕ, (Real.cos θ)^(2*n)) = 8 → Real.cos (4*θ) = 1/8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_four_theta_value_l879_87967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graphics_card_pricing_strategy_l879_87960

/-- Represents the price and sales data for a computer graphics card over three years -/
structure GraphicsCardData where
  price_2021 : ℝ
  price_2023 : ℝ
  base_sales : ℝ
  sales_increase_rate : ℝ
  daily_profit : ℝ

/-- Calculates the average annual percentage decrease in price -/
noncomputable def average_annual_decrease (data : GraphicsCardData) : ℝ :=
  1 - (data.price_2023 / data.price_2021) ^ (1 / 2)

/-- Calculates the optimal price reduction to maximize sales while maintaining profit -/
noncomputable def optimal_price_reduction (data : GraphicsCardData) : ℝ :=
  let a := -2
  let b := 2 * (data.price_2021 - data.price_2023) + data.base_sales
  let c := data.base_sales * (data.price_2021 - data.price_2023) - data.daily_profit
  (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)

/-- Theorem stating the properties of the graphics card pricing strategy -/
theorem graphics_card_pricing_strategy (data : GraphicsCardData) 
  (h1 : data.price_2021 = 200)
  (h2 : data.price_2023 = 162)
  (h3 : data.base_sales = 20)
  (h4 : data.sales_increase_rate = 2)
  (h5 : data.daily_profit = 1150) :
  average_annual_decrease data = 0.1 ∧ 
  optimal_price_reduction data = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graphics_card_pricing_strategy_l879_87960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_is_eight_l879_87943

/-- The circle equation -/
def circle_equation (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  2 * x^2 + 2 * y^2 - 16 * x + 8 * y + 36 = 0

/-- The circle is inscribed in a square with sides parallel to axes -/
def inscribed_in_square (c : (ℝ × ℝ) → Prop) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ p, c p ↔ (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2) ∧
    (∀ p, c p → p.1 ≥ center.1 - radius ∧ p.1 ≤ center.1 + radius ∧
                 p.2 ≥ center.2 - radius ∧ p.2 ≤ center.2 + radius)

/-- The main theorem -/
theorem square_area_is_eight :
  inscribed_in_square circle_equation →
  (∃ side : ℝ, side^2 = 8 ∧
    ∀ p, circle_equation p →
      p.1 ≥ -side/2 ∧ p.1 ≤ side/2 ∧ p.2 ≥ -side/2 ∧ p.2 ≤ side/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_is_eight_l879_87943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_center_locus_is_line_l879_87980

/-- A triangle with an inscribed square -/
structure TriangleWithInscribedSquare where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  S : Set (ℝ × ℝ)
  h : ℝ

/-- Predicate to check if a square is inscribed in a triangle -/
def is_inscribed_square_in (S : Set (ℝ × ℝ)) (A B C : ℝ × ℝ) : Prop :=
  sorry  -- Definition to be implemented

/-- Predicate to check if a point is the center of a square -/
def is_center_of (p : ℝ × ℝ) (S : Set (ℝ × ℝ)) : Prop :=
  sorry  -- Definition to be implemented

/-- The locus of centers of inscribed squares -/
def squareCenterLocus (t : TriangleWithInscribedSquare) : Set (ℝ × ℝ) :=
  {center | ∃ (C' : ℝ × ℝ), 
    C'.2 = t.h ∧  -- C' is on a line parallel to AB
    ∃ (S' : Set (ℝ × ℝ)), 
      is_inscribed_square_in S' t.A t.B C' ∧
      is_center_of center S'}

/-- Predicate to check if a set of points forms a line -/
def is_line (s : Set (ℝ × ℝ)) : Prop :=
  sorry  -- Definition to be implemented

/-- Main theorem: The locus of square centers is a line -/
theorem square_center_locus_is_line (t : TriangleWithInscribedSquare) :
  ∃ (l : Set (ℝ × ℝ)), is_line l ∧ squareCenterLocus t = l :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_center_locus_is_line_l879_87980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_cosine_sequence_exists_l879_87982

theorem negative_cosine_sequence_exists : ∃ α : ℝ, ∀ n : ℕ, Real.cos (2^n * α) < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_cosine_sequence_exists_l879_87982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_hats_dont_return_after_odd_switches_l879_87918

/-- Represents the three people involved in the hat switching. -/
inductive Person : Type
  | Lunasa : Person
  | Merlin : Person
  | Lyrica : Person

/-- Represents a hat configuration as a permutation of people. -/
def HatConfiguration := Equiv.Perm Person

/-- Represents a single day's hat switch. -/
def DailySwitch := Person × Person

/-- The number of days that pass in the problem. -/
def numDays : ℕ := 2017

/-- 
  Given a sequence of daily switches, this function returns the probability 
  that all hats are back with their original owners.
-/
noncomputable def probabilityHatsReturnToOwners (switches : List DailySwitch) : ℚ :=
  sorry

/-- 
  Theorem stating that after an odd number of random two-person hat switches,
  the probability of all hats returning to their original owners is 0.
-/
theorem hats_dont_return_after_odd_switches 
  (switches : List DailySwitch) 
  (h : switches.length % 2 = 1) : 
  probabilityHatsReturnToOwners switches = 0 := by
  sorry

#check hats_dont_return_after_odd_switches

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_hats_dont_return_after_odd_switches_l879_87918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l879_87949

theorem exponential_inequality (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (h4 : (2 : ℝ)^x = (3 : ℝ)^y) (h5 : (3 : ℝ)^y = (5 : ℝ)^z) : 3*y < 2*x ∧ 2*x < 5*z :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l879_87949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l879_87965

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (3 - x) + Real.log (x + 2)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Ioo (-2 : ℝ) 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l879_87965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l879_87937

noncomputable def f (x : ℝ) := 2 * (Real.cos (x - Real.pi/4))^2 - Real.sqrt 3 * Real.cos (2*x) + 1

def is_axis_of_symmetry (a : ℝ) :=
  ∀ x, f (a + x) = f (a - x)

theorem f_properties :
  ∃ k : ℤ, is_axis_of_symmetry (k * Real.pi/2 + 5*Real.pi/12) ∧
  (∀ x ∈ Set.Icc (Real.pi/4 : ℝ) (Real.pi/2 : ℝ), f x ≥ 3) ∧
  (∀ x ∈ Set.Icc (Real.pi/4 : ℝ) (Real.pi/2 : ℝ), f x ≤ 4) ∧
  f (Real.pi/4) = 3 ∧
  f (5*Real.pi/12) = 4 ∧
  (∀ m : ℝ, (∀ x ∈ Set.Icc (Real.pi/4 : ℝ) (Real.pi/2 : ℝ), |f x - m| < 2) → 2 < m ∧ m < 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l879_87937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_center_of_symmetry_l879_87951

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := tan (x + π/5)

-- Define the domain of the function
def domain (x : ℝ) : Prop := ∀ k : ℤ, x ≠ k * π + 3 * π / 10

-- State the theorem
theorem tan_center_of_symmetry :
  ∀ x : ℝ, domain x → 
  f (3 * π / 10 - (x - 3 * π / 10)) = -f x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_center_of_symmetry_l879_87951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dots_Z_l879_87947

/-- Represents the number of dots on a given level -/
def dots (n : ℕ) : ℕ := sorry

/-- The number of levels in the sequence -/
def num_levels : ℕ := 26

/-- Level A has 1 dot -/
axiom dots_A : dots 1 = 1

/-- Even-indexed levels (B, D, F, ...) have twice as many dots as the level above -/
axiom dots_even (n : ℕ) (h : 2 ≤ n ∧ n ≤ num_levels ∧ Even n) : dots n = 2 * dots (n - 1)

/-- Odd-indexed levels (C, E, G, ...) have the same number of dots as the level above -/
axiom dots_odd (n : ℕ) (h : 3 ≤ n ∧ n ≤ num_levels ∧ Odd n) : dots n = dots (n - 1)

/-- The main theorem: Level Z (26th level) contains 8192 dots -/
theorem dots_Z : dots num_levels = 8192 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dots_Z_l879_87947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_equal_probability_sum_l879_87971

/-- The number of sides on a standard die -/
def standardDieSides : ℕ := 6

/-- The sum we want to match in probability -/
def targetSum : ℕ := 2027

/-- The smallest number of dice needed to potentially reach the target sum -/
def minDice : ℕ := (targetSum + standardDieSides - 1) / standardDieSides

/-- The transformation function for a single die roll -/
def transformRoll (x : ℕ) : ℕ := standardDieSides + 1 - x

/-- Helper function to represent the probability of getting a certain sum with n dice -/
noncomputable def probability_of_sum (n : ℕ) (sum : ℕ) : ℚ :=
  sorry

/-- The theorem stating the smallest possible value of S -/
theorem smallest_equal_probability_sum : 
  ∃ (S : ℕ), S = minDice * (standardDieSides + 1) - targetSum ∧ 
  S = 339 ∧
  (∀ (S' : ℕ), S' < S → 
    ¬(probability_of_sum minDice S' = probability_of_sum minDice targetSum)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_equal_probability_sum_l879_87971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_rally_accident_probability_l879_87997

theorem car_rally_accident_probability :
  let bridge_fail_prob : ℚ := 1/5
  let turn_fail_prob : ℚ := 3/10
  let tunnel_fail_prob : ℚ := 1/10
  let sand_fail_prob : ℚ := 2/5
  
  let total_accident_prob : ℚ := 1 - (1 - bridge_fail_prob) * (1 - turn_fail_prob) * (1 - tunnel_fail_prob) * (1 - sand_fail_prob)
  
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ abs (total_accident_prob - 7/10) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_rally_accident_probability_l879_87997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_period_and_inequality_l879_87924

noncomputable def f (x : ℝ) := Real.tan (2 * x - Real.pi / 4)

theorem tan_period_and_inequality (x : ℝ) :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ T' > 0, T' < T → ∃ x, f (x + T') ≠ f x) ∧
  (∀ k : ℤ, f x > 1 ↔ k * Real.pi / 2 + Real.pi / 4 < x ∧ x < k * Real.pi / 2 + 3 * Real.pi / 8) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_period_and_inequality_l879_87924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l879_87970

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

theorem min_value_of_f :
  ∃ (m : ℝ), (∀ (x : ℝ), m ≤ f x) ∧ (∃ (x : ℝ), f x = m) ∧ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l879_87970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l879_87984

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then -x + 6 else 3 + (Real.log x) / (Real.log a)

theorem range_of_a (a : ℝ) :
  (a > 0 ∧ a ≠ 1) →
  (∀ x : ℝ, x ≥ 4 → f a x ≥ 4) →
  (1 < a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l879_87984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wang_bo_combined_purchase_l879_87931

-- Define the discount function
noncomputable def discount (amount : ℝ) : ℝ :=
  if amount ≤ 100 then amount
  else if amount ≤ 300 then 0.9 * amount
  else 0.8 * amount

-- Define Wang Bo's purchases
def purchase1 : ℝ := 80
def purchase2 : ℝ := 252

-- Theorem statement
theorem wang_bo_combined_purchase :
  let combined := purchase1 + (purchase2 / 0.9)
  let discounted := discount combined
  discounted = 288 ∨ discounted = 316 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wang_bo_combined_purchase_l879_87931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_deriv_when_pos_l879_87917

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- State the conditions
axiom f_odd : ∀ x, f (-x) + f x = 0
axiom g_even : ∀ x, g x = g (|x|)
axiom f_deriv_neg : ∀ x, x < 0 → deriv f x < 0
axiom g_deriv_pos : ∀ x, x < 0 → deriv g x > 0

-- State the theorem to be proved
theorem f_g_deriv_when_pos :
  ∀ x, x > 0 → deriv f x < 0 ∧ deriv g x < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_deriv_when_pos_l879_87917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_point_implies_a_bound_l879_87981

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := Real.log x + a * x^2 + b * x

-- State the theorem
theorem max_point_implies_a_bound {a b : ℝ} :
  (∀ x > 0, f a b x ≤ f a b 1) →
  a < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_point_implies_a_bound_l879_87981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_billiard_ball_radius_theorem_l879_87922

/-- The radius of a billiard ball in an equilateral triangle arrangement -/
noncomputable def billiard_ball_radius (num_balls : ℕ) (perimeter : ℝ) : ℝ :=
  (perimeter / 3) / (8 + 2 * Real.sqrt 3)

/-- Theorem: Given 15 billiard balls in an equilateral triangle frame with 
    an inner perimeter of 876 mm, the radius of each ball is 146/13 * (4 - √3/3) mm -/
theorem billiard_ball_radius_theorem :
  billiard_ball_radius 15 876 = (146 / 13) * (4 - Real.sqrt 3 / 3) := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_billiard_ball_radius_theorem_l879_87922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_three_iff_l879_87968

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 2^x + 1 else |x|

theorem f_eq_three_iff (x : ℝ) : f x = 3 ↔ x = -3 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_three_iff_l879_87968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l879_87942

/-- Represents a position in the Chocolate game -/
structure ChocolatePosition where
  rows : Nat
  cols : Nat
  markedRow : Nat
  markedCol : Nat

/-- Represents a move in the Chocolate game -/
inductive ChocolateMove
  | HorizontalCut (row : Nat)
  | VerticalCut (col : Nat)

/-- The Chocolate game -/
def ChocolateGame : Type := 
  Σ (pos : ChocolatePosition), List ChocolateMove

/-- Initial position of the Chocolate game -/
def initialPosition : ChocolatePosition :=
  { rows := 6, cols := 8, markedRow := 0, markedCol := 0 }

/-- Determines if a position is a winning position for the current player -/
def isWinningPosition (pos : ChocolatePosition) : Prop :=
  sorry

/-- Represents a strategy for the Chocolate game -/
def Strategy : Type :=
  ChocolatePosition → Option ChocolateMove

/-- Determines if a strategy is a winning strategy for a player -/
def WinningStrategy (game : ChocolateGame) (player : Nat) (strategy : Strategy) : Prop :=
  sorry

/-- The theorem stating that the first player has a winning strategy -/
theorem first_player_wins (game : ChocolateGame) 
  (h : game.1 = initialPosition) : 
  ∃ (strategy : Strategy),
    WinningStrategy game 0 strategy :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l879_87942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_l879_87900

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 / 100 + y^2 / 64 = 1

-- Define a point on the circle
def point_on_circle (P : ℝ × ℝ) : Prop := circle_equation P.1 P.2

-- Define the focus F (we don't know its exact coordinates, so we leave it as a parameter)
variable (F : ℝ × ℝ)

-- Define the perimeter of triangle FMN
noncomputable def perimeter (M N : ℝ × ℝ) : ℝ :=
  Real.sqrt ((M.1 - F.1)^2 + (M.2 - F.2)^2) +
  Real.sqrt ((N.1 - F.1)^2 + (N.2 - F.2)^2) +
  Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2)

-- Theorem statement
theorem max_perimeter :
  ∀ M N : ℝ × ℝ, point_on_circle M → point_on_circle N →
  perimeter F M N ≤ 40 + Real.sqrt 164 :=
by
  sorry

#check max_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_l879_87900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_in_cube_l879_87938

-- Define the cube
noncomputable def cube_edge_length : ℝ := 6

-- Define the sphere
noncomputable def sphere_radius : ℝ := cube_edge_length / 2

-- Calculate the volume of the sphere
noncomputable def sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius ^ 3

-- Theorem statement
theorem sphere_volume_in_cube : sphere_volume = 36 * Real.pi := by
  -- Unfold definitions
  unfold sphere_volume
  unfold sphere_radius
  unfold cube_edge_length
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_in_cube_l879_87938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_estimated_population_is_537_l879_87914

/-- Represents the fish population estimation problem -/
structure FishPopulationEstimation where
  initialMarked : ℕ  -- Number of fish initially marked
  catch1 : ℕ  -- First catch on second day
  marked1 : ℕ  -- Marked fish in first catch
  catch2 : ℕ  -- Second catch on second day
  marked2 : ℕ  -- Marked fish in second catch
  catch3 : ℕ  -- Third catch on second day
  marked3 : ℕ  -- Marked fish in third catch

/-- Calculates the estimated total fish population -/
def estimateTotalPopulation (fpe : FishPopulationEstimation) : ℚ :=
  let totalCaught := fpe.catch1 + fpe.catch2 + fpe.catch3
  let totalMarked := fpe.marked1 + fpe.marked2 + fpe.marked3
  (fpe.initialMarked * totalCaught : ℚ) / totalMarked

/-- Theorem stating that the estimated fish population is approximately 537 -/
theorem estimated_population_is_537 (fpe : FishPopulationEstimation) 
  (h1 : fpe.initialMarked = 50)
  (h2 : fpe.catch1 = 67) (h3 : fpe.marked1 = 6)
  (h4 : fpe.catch2 = 94) (h5 : fpe.marked2 = 10)
  (h6 : fpe.catch3 = 43) (h7 : fpe.marked3 = 3) :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ |estimateTotalPopulation fpe - 537| < ε := by
  sorry

#eval estimateTotalPopulation { 
  initialMarked := 50, 
  catch1 := 67, marked1 := 6, 
  catch2 := 94, marked2 := 10, 
  catch3 := 43, marked3 := 3 
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_estimated_population_is_537_l879_87914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_range_l879_87983

theorem sine_function_range (ω : ℝ) (h_pos : ω > 0) :
  (∀ x : ℝ, x ∈ Set.Icc 0 (2 * Real.pi) →
    (∃! x₁ : ℝ, x₁ ∈ Set.Icc 0 (2 * Real.pi) ∧ Real.sin (ω * x₁ + Real.pi / 3) = 1) ∧
    (∃! x₂ : ℝ, x₂ ∈ Set.Icc 0 (2 * Real.pi) ∧ Real.sin (ω * x₂ + Real.pi / 3) = -1)) →
  7 / 12 ≤ ω ∧ ω < 13 / 12 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_range_l879_87983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_palindromic_14_and_20_l879_87995

/-- A number is palindromic in a given base if it reads the same forwards and backwards in that base. -/
def isPalindromic (n : ℕ) (base : ℕ) : Prop :=
  ∃ (digits : List ℕ), digits.reverse = digits ∧ Nat.digits base n = digits

/-- The smallest natural number greater than 20 that is palindromic in both base 14 and base 20 is 105. -/
theorem smallest_palindromic_14_and_20 :
  ∃ (N : ℕ), N > 20 ∧ isPalindromic N 14 ∧ isPalindromic N 20 ∧
  ∀ (M : ℕ), M > 20 → isPalindromic M 14 → isPalindromic M 20 → N ≤ M :=
by
  use 105
  constructor
  · norm_num
  constructor
  · use [7, 7]
    constructor
    · rfl
    · norm_num
  constructor
  · use [5, 5]
    constructor
    · rfl
    · norm_num
  · intro M hM h14 h20
    sorry

#eval 105  -- To verify the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_palindromic_14_and_20_l879_87995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_even_increasing_function_l879_87976

-- Define the properties of the function f
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def IncreasingOn (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

-- State the theorem
theorem range_of_even_increasing_function (f : ℝ → ℝ) 
  (h_even : EvenFunction f) 
  (h_incr : IncreasingOn f (Set.Iic 0)) : 
  {x : ℝ | f x < f 2} = {x : ℝ | x > 2 ∨ x < -2} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_even_increasing_function_l879_87976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_is_arithmetic_sequence_sum_of_a_sequence_l879_87973

def a : ℕ → ℤ
  | 0 => -3
  | n + 1 => 2 * a n + 2^(n + 1) + 3

def b (n : ℕ) : ℚ := (a n + 3) / 2^n

def S (n : ℕ) : ℤ := (n - 2) * 2^(n + 1) - 3 * n + 4

theorem b_is_arithmetic_sequence :
  ∀ n : ℕ, b (n + 1) - b n = 1 := by sorry

theorem sum_of_a_sequence (n : ℕ) :
  (Finset.range n).sum (λ i => a i) = S n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_is_arithmetic_sequence_sum_of_a_sequence_l879_87973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circus_standing_time_l879_87902

/-- The combined standing time of four entertainers given specific conditions -/
theorem circus_standing_time 
  (pulsar_time polly_time petra_time penny_time : ℕ) :
  pulsar_time = 10 →
  polly_time = 3 * pulsar_time →
  petra_time = polly_time / 6 →
  penny_time = 2 * (pulsar_time + polly_time + petra_time) →
  pulsar_time + polly_time + petra_time + penny_time = 135 := by
  intro h1 h2 h3 h4
  -- The proof steps would go here
  sorry

#check circus_standing_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circus_standing_time_l879_87902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_c_distance_when_a_finishes_l879_87993

/-- Represents the distance of a runner from the start of the race -/
noncomputable def distance_from_start (total_distance : ℝ) (distance_from_finish : ℝ) : ℝ :=
  total_distance - distance_from_finish

/-- The ratio of distances run by two runners -/
noncomputable def distance_ratio (distance1 : ℝ) (distance2 : ℝ) : ℝ :=
  distance1 / distance2

theorem runner_c_distance_when_a_finishes (race_length : ℝ) 
  (b_distance_when_a_finishes : ℝ) (c_distance_when_b_finishes : ℝ) :
  let a_finish_distance := race_length
  let b_start_distance := distance_from_start race_length b_distance_when_a_finishes
  let c_start_distance := distance_from_start race_length c_distance_when_b_finishes
  let ratio_b_to_a := distance_ratio b_start_distance a_finish_distance
  let ratio_c_to_b := distance_ratio c_start_distance race_length
  let ratio_c_to_a := ratio_c_to_b * ratio_b_to_a
  let c_distance_when_a_finishes := race_length * ratio_c_to_a
  race_length = 1000 ∧ 
  b_distance_when_a_finishes = 50 ∧ 
  c_distance_when_b_finishes = 100 →
  race_length - c_distance_when_a_finishes = 145 := by
  sorry

#check runner_c_distance_when_a_finishes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_c_distance_when_a_finishes_l879_87993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bobs_raised_bed_height_l879_87905

/-- Represents the dimensions and properties of a raised bed --/
structure RaisedBed where
  width : ℚ
  length : ℚ
  plankWidth : ℚ
  totalPlanks : ℕ
  numBeds : ℕ

/-- Calculates the height of a raised bed given its properties --/
def calculateHeight (bed : RaisedBed) : ℚ :=
  (bed.totalPlanks : ℚ) / (4 * bed.numBeds : ℚ)

/-- Theorem stating the height of Bob's raised beds --/
theorem bobs_raised_bed_height :
  let bed := RaisedBed.mk 2 8 1 50 10
  calculateHeight bed = 5/4 := by
  -- Proof goes here
  sorry

#eval calculateHeight (RaisedBed.mk 2 8 1 50 10)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bobs_raised_bed_height_l879_87905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangents_perpendicular_l879_87996

theorem tangents_perpendicular (a : ℝ) (ha : a ≠ 0) :
  ∃ x₀ : ℝ, Real.cos x₀ = a * Real.tan x₀ →
  (- Real.sin x₀) * (a / (Real.cos x₀)^2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangents_perpendicular_l879_87996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_can_be_excellent_l879_87969

/-- Represents a young man with height and weight -/
structure YoungMan where
  height : ℝ
  weight : ℝ

/-- A group of 100 young men -/
def YoungMenGroup := Fin 100 → YoungMan

/-- Predicate to check if one young man is not inferior to another -/
def notInferior (a b : YoungMan) : Prop :=
  a.height > b.height ∨ a.weight > b.weight

/-- Predicate to check if a young man is excellent -/
def isExcellent (g : YoungMenGroup) (i : Fin 100) : Prop :=
  ∀ j : Fin 100, j ≠ i → notInferior (g i) (g j)

/-- Theorem stating that it's possible for all 100 men to be excellent -/
theorem all_can_be_excellent :
  ∃ g : YoungMenGroup, ∀ i : Fin 100, isExcellent g i := by
  sorry

#check all_can_be_excellent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_can_be_excellent_l879_87969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l879_87906

/-- The constant term in the expansion of (x^2 - 2/x^3)^5 is 40 -/
theorem constant_term_expansion : ∃ (c : ℤ), c = 40 ∧ 
  (∀ x : ℝ, x ≠ 0 → ∃ (p : ℝ → ℝ), (x^2 - 2/x^3)^5 = c + x * (p x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l879_87906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l879_87939

/-- The function f(x) = cos²x + √3 * sin x * cos x -/
noncomputable def f (x : ℝ) : ℝ := (Real.cos x)^2 + Real.sqrt 3 * Real.sin x * Real.cos x

/-- The domain of x -/
def x_domain : Set ℝ := Set.Ioo 0 (Real.pi / 2)

/-- The interval where f(x) is monotonically increasing -/
def increasing_interval : Set ℝ := Set.Ioc 0 (Real.pi / 6)

/-- Theorem stating that f(x) is monotonically increasing in the specified interval -/
theorem f_monotone_increasing : 
  ∀ x ∈ increasing_interval, ∀ y ∈ increasing_interval, 
    x < y → f x < f y := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l879_87939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_female_officers_count_l879_87903

-- Define the total number of officers on duty
def total_on_duty : ℕ := 225

-- Define the percentage of female officers on duty
def female_percentage_on_duty : ℚ := 62 / 100

-- Define the percentage of total female officers who were on duty
def female_on_duty_percentage : ℚ := 23 / 100

-- Theorem to prove
theorem female_officers_count : 
  ⌊(female_percentage_on_duty * ↑total_on_duty) / female_on_duty_percentage⌋ = 609 := by
  -- Proof steps would go here
  sorry

#eval ⌊(female_percentage_on_duty * ↑total_on_duty) / female_on_duty_percentage⌋

end NUMINAMATH_CALUDE_ERRORFEEDBACK_female_officers_count_l879_87903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l879_87925

theorem trig_inequality : Real.tan (7 * Real.pi / 5) > Real.sin (2 * Real.pi / 5) ∧ 
                          Real.sin (2 * Real.pi / 5) > Real.cos (6 * Real.pi / 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l879_87925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_term_of_sequence_l879_87987

def a : ℕ → ℤ
  | 0 => 4  -- Added case for 0
  | 1 => 4
  | n + 1 => 2 * a n + n^2

theorem general_term_of_sequence (n : ℕ) :
  a n = 5 * 2^n - n^2 - 2*n - 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_term_of_sequence_l879_87987
