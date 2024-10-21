import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_cube_ratio_l82_8211

/-- The ratio of surface areas and volumes of a sphere to its inscribed cube -/
theorem sphere_cube_ratio (R : ℝ) (R_pos : R > 0) :
  (4 * Real.pi * R^2) / (6 * (2 * R / Real.sqrt 3)^2) = Real.pi / 2 ∧
  ((4 / 3) * Real.pi * R^3) / ((2 * R / Real.sqrt 3)^3) = Real.pi * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_cube_ratio_l82_8211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_waiting_probability_l82_8200

/-- The probability of waiting no more than a given time for a bus with uniform arrival distribution -/
noncomputable def waitingProbability (busInterval : ℝ) (maxWaitTime : ℝ) : ℝ :=
  min maxWaitTime busInterval / busInterval

theorem bus_waiting_probability :
  let busInterval : ℝ := 5
  let maxWaitTime : ℝ := 3
  waitingProbability busInterval maxWaitTime = 3/5 := by
  unfold waitingProbability
  simp [min]
  norm_num
  

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_waiting_probability_l82_8200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l82_8286

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function f: ℝ → ℝ is increasing on [a,b] if x ≤ y implies f(x) ≤ f(y) for all x, y in [a,b] -/
def IsIncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y

theorem range_of_t (f : ℝ → ℝ) (t : ℝ)
  (h_odd : IsOdd f)
  (h_incr : IsIncreasingOn f (-1) 1)
  (h_f_neg_one : f (-1) = -1)
  (h_ineq : ∀ x a, x ∈ Set.Icc (-1) 1 → a ∈ Set.Icc (-1) 1 → f x ≤ t^2 - 2*a*t + 1) :
  t ≤ -2 ∨ t ≥ 2 ∨ t = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l82_8286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_with_12_divisors_l82_8209

def has_exactly_n_divisors (n : ℕ) (k : ℕ) : Prop :=
  (Finset.filter (λ m ↦ n % m = 0) (Finset.range (n + 1))).card = k

theorem smallest_with_12_divisors :
  ∀ m : ℕ, m > 0 → has_exactly_n_divisors m 12 → m ≥ 72 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_with_12_divisors_l82_8209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_vectors_l82_8270

noncomputable def angle_between_vectors (a b : ℝ × ℝ) : ℝ :=
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem angle_between_specific_vectors :
  let a : ℝ × ℝ := (3, 3)
  ∃ b : ℝ × ℝ,
    Real.sqrt (b.1^2 + b.2^2) = 6 ∧
    a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0 ∧
    angle_between_vectors a b = π / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_vectors_l82_8270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_congruence_and_isosceles_l82_8284

/-- Triangle structure -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Angle bisector of a triangle -/
noncomputable def angle_bisector (t : Triangle) (v : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Length of a line segment -/
noncomputable def length (p q : ℝ × ℝ) : ℝ := sorry

/-- Angle measure -/
noncomputable def angle (p q r : ℝ × ℝ) : ℝ := sorry

/-- Triangle congruence -/
def congruent (t1 t2 : Triangle) : Prop := sorry

theorem triangle_congruence_and_isosceles 
  (ABC A'B'C' : Triangle) 
  (D : ℝ × ℝ) 
  (A1 C1 : ℝ × ℝ) :
  length ABC.A ABC.C = length A'B'C'.A A'B'C'.C →
  angle ABC.A ABC.B ABC.C = angle A'B'C'.A A'B'C'.B A'B'C'.C →
  length (angle_bisector ABC ABC.B) = length (angle_bisector A'B'C' A'B'C'.B) →
  angle_bisector ABC ABC.B = D →
  length ABC.A A1 = length ABC.C C1 →
  congruent ABC A'B'C' ∧ length ABC.A ABC.B = length ABC.B ABC.C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_congruence_and_isosceles_l82_8284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_transformations_l82_8292

-- Define the square and transformations
def Square : Set (ℝ × ℝ) := {(1, 1), (-1, 1), (-1, -1), (1, -1)}

inductive Transformation
| L  -- 90° counterclockwise rotation
| R  -- 90° clockwise rotation
| S  -- reflection about origin

def apply_transformation : Transformation → (ℝ × ℝ) → (ℝ × ℝ)
| Transformation.L, (x, y) => (-y, x)
| Transformation.R, (x, y) => (y, -x)
| Transformation.S, (x, y) => (-x, -y)

-- Define k-identity transformation
def is_k_identity (seq : List Transformation) : Prop :=
  ∀ p ∈ Square, 
    (seq.foldl (λ acc t => apply_transformation t acc) p) = p

-- Main theorem
theorem identity_transformations :
  (∃ (three_id : List (List Transformation)), 
     (∀ seq ∈ three_id, seq.length = 3 ∧ is_k_identity seq) ∧ 
     three_id.length = 6) ∧
  (∀ n : ℕ, 
     n > 0 → 
     ∃ (n_id : List (List Transformation)),
       (∀ seq ∈ n_id, seq.length = n ∧ is_k_identity seq) ∧
       n_id.length = (3 * ((-1 : ℤ)^n + 3^n)) / 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_transformations_l82_8292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wage_difference_l82_8279

/-- The hourly wage of candidate P -/
def P : ℚ := sorry

/-- The hourly wage of candidate Q -/
def Q : ℚ := sorry

/-- The number of hours candidate P needs to complete the job -/
def h : ℚ := sorry

theorem wage_difference :
  (P = 1.5 * Q) →
  (P * h = 540) →
  (Q * (h + 10) = 540) →
  (P - Q = 9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wage_difference_l82_8279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_fill_time_l82_8208

/-- Represents the filling time of a water tank with three pipes -/
structure FillTime (x y : ℝ) : Prop where
  first_third : 1 / x - 1 / y = 2 / 15
  second_third : 1 / (x + 1) - 1 / y = 1 / 20
  x_positive : x > 0
  y_positive : y > 0

/-- The time to fill the tank with all three pipes open -/
noncomputable def CombinedFillTime (x y : ℝ) : ℝ :=
  1 / (1 / x + 1 / (x + 1) - 1 / y)

theorem water_tank_fill_time :
  ∀ x y : ℝ, FillTime x y → CombinedFillTime x y = 60 / 23 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_fill_time_l82_8208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_a_b_c_l82_8249

theorem compare_a_b_c : 
  (17 : Real)^(1/17) > Real.log 17 / Real.log 16 ∧ 
  Real.log 17 / Real.log 16 > Real.log 4 / Real.log 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_a_b_c_l82_8249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spiral_to_squares_theorem_l82_8237

/-- Represents a configuration of matchsticks --/
structure MatchstickConfiguration where
  matchsticks : List (Int × Int × Int × Int)

/-- Represents a move of a matchstick --/
structure MatchstickMove where
  fromPos : Int × Int × Int × Int
  toPos : Int × Int × Int × Int

/-- Checks if a configuration forms three squares of different sizes --/
def isThreeSquares (config : MatchstickConfiguration) : Prop := sorry

/-- Checks if a list of moves is valid (exactly four moves) --/
def isValidMoveSet (moves : List MatchstickMove) : Prop :=
  moves.length = 4

/-- Applies a list of moves to a configuration --/
def applyMoves (config : MatchstickConfiguration) (moves : List MatchstickMove) : MatchstickConfiguration := sorry

/-- The initial spiral configuration --/
def initialSpiral : MatchstickConfiguration := sorry

/-- Theorem: It is possible to transform the initial spiral into three squares of different sizes by moving exactly four matchsticks --/
theorem spiral_to_squares_theorem :
  ∃ (moves : List MatchstickMove),
    isValidMoveSet moves ∧
    isThreeSquares (applyMoves initialSpiral moves) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spiral_to_squares_theorem_l82_8237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l82_8247

theorem remainder_theorem (e : ℕ) :
  (∃! n : ℕ, n < 180 ∧ n % 8 = 5) →
  e % 13 = 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l82_8247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_monic_poly_degree_3_l82_8280

-- Define a monic polynomial of degree 3
def monicPoly (b c d : ℝ) : ℝ → ℝ := fun x ↦ x^3 + b*x^2 + c*x + d

-- State the theorem
theorem unique_monic_poly_degree_3 :
  ∃! (b c d : ℝ), 
    (monicPoly b c d 0 = 4) ∧ 
    (monicPoly b c d 1 = 10) ∧ 
    (monicPoly b c d (-1) = 2) ∧
    (∀ x, monicPoly b c d x = x^3 + 2*x^2 + 3*x + 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_monic_poly_degree_3_l82_8280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recurrence_sequence_finitely_many_negative_iff_l82_8291

/-- A sequence of non-zero real numbers defined by the recurrence relation x_{n+1} = A - 1/x_n --/
noncomputable def RecurrenceSequence (A : ℝ) : ℕ → ℝ
| 0 => 1  -- arbitrary non-zero initial value
| n + 1 => A - 1 / RecurrenceSequence A n

/-- A sequence has only finitely many negative terms --/
def HasFinitelyManyNegativeTerms (s : ℕ → ℝ) : Prop :=
  ∃ N : ℕ, ∀ n ≥ N, s n ≥ 0

/-- The main theorem: the recurrence sequence has only finitely many negative terms iff A ≥ 2 --/
theorem recurrence_sequence_finitely_many_negative_iff (A : ℝ) :
  HasFinitelyManyNegativeTerms (RecurrenceSequence A) ↔ A ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recurrence_sequence_finitely_many_negative_iff_l82_8291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_pairs_l82_8269

theorem regular_polygon_pairs : ∃ (pairs : List (ℕ × ℕ)), 
  (pairs.length = 4) ∧ 
  (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 > 2 ∧ p.2 > 2) ∧
  (∀ (p : ℕ × ℕ), p ∈ pairs → (p.1 - 2) * p.2 * 3 = (p.2 - 2) * p.1 * 4) ∧
  (∀ (r k : ℕ), r > 2 → k > 2 → (r - 2) * k * 3 = (k - 2) * r * 4 → (r, k) ∈ pairs) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_pairs_l82_8269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sampling_method_is_systematic_l82_8255

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic
  | RandomNumberTable

/-- Represents a class in the grade -/
structure GradeClass where
  students : Finset ℕ
  student_count : students.card = 50
  id_range : ∀ id ∈ students, 1 ≤ id ∧ id ≤ 50

/-- Represents the grade -/
structure Grade where
  classes : Finset GradeClass
  class_count : classes.card = 20

/-- Represents the sampled student IDs -/
def sampled_ids : Finset ℕ := {5, 15, 25, 35, 45}

/-- Main theorem: The sampling method used is Systematic Sampling -/
theorem sampling_method_is_systematic (grade : Grade) : 
  (∀ c ∈ grade.classes, ∀ id ∈ sampled_ids, id ∈ c.students) → 
  SamplingMethod.Systematic = 
    (SamplingMethod.SimpleRandom, SamplingMethod.Stratified, SamplingMethod.Systematic, SamplingMethod.RandomNumberTable).2.2.1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sampling_method_is_systematic_l82_8255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_14_sides_l82_8234

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : n > 2

/-- The sum of exterior angles of a regular polygon is 360° -/
def sumExteriorAngles (n : ℕ) (_ : RegularPolygon n) : ℝ := 360

/-- The measure of each exterior angle of a regular polygon -/
noncomputable def exteriorAngle (n : ℕ) (_ : RegularPolygon n) : ℝ := 360 / n

theorem regular_polygon_14_sides :
  ∀ (p : RegularPolygon 14),
    sumExteriorAngles 14 p = 360 ∧
    exteriorAngle 14 p = 360 / 14 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_14_sides_l82_8234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_l82_8278

theorem function_identity (f : ℝ → ℝ) (h : ∀ x, f (Real.cos x) = Real.cos (17 * x)) :
  ∀ x, (f (Real.cos x))^2 + (f (Real.sin x))^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_l82_8278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_approx_4_08_minutes_l82_8215

/-- The time (in minutes) it takes for two people walking in opposite directions 
    on a circular track to meet for the first time. -/
noncomputable def meetingTime (trackCircumference : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  trackCircumference / (speed1 + speed2)

/-- Theorem stating that two people walking in opposite directions on a 561m track
    at speeds of 4.5 km/hr and 3.75 km/hr will meet in approximately 4.08 minutes. -/
theorem meeting_time_approx_4_08_minutes :
  let trackCircumference : ℝ := 561
  let speed1 : ℝ := 4.5 * 1000 / 60  -- Convert km/hr to m/min
  let speed2 : ℝ := 3.75 * 1000 / 60  -- Convert km/hr to m/min
  abs (meetingTime trackCircumference speed1 speed2 - 4.08) < 0.01 := by
  sorry

-- Use #eval with a function that returns a nat or int instead
def meetingTimeSeconds (trackCircumference : Nat) (speed1 speed2 : Nat) : Nat :=
  (trackCircumference * 60) / (speed1 + speed2)

#eval meetingTimeSeconds 561 75 63  -- speeds in m/min, result in seconds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_approx_4_08_minutes_l82_8215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_problem_l82_8225

def is_three_element_subset_sum (A : Finset ℤ) (s : ℤ) : Prop :=
  ∃ (x y z : ℤ), x ∈ A ∧ y ∈ A ∧ z ∈ A ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x + y + z = s

theorem subset_sum_problem (A : Finset ℤ) :
  A.card = 4 →
  (∀ s, s ∈ ({-1, 3, 5, 8} : Finset ℤ) ↔ is_three_element_subset_sum A s) →
  A = {-3, 0, 2, 6} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_problem_l82_8225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_is_pi_over_24_l82_8267

/-- The ratio of the volume of a right circular cone inscribed in a right prism to the volume of the prism -/
noncomputable def volume_ratio (h r : ℝ) : ℝ :=
  let cone_volume := (1 / 3) * Real.pi * r^2 * h
  let prism_volume := (2 * r)^2 * (2 * h)
  cone_volume / prism_volume

/-- Theorem stating that the volume ratio of an inscribed cone to its containing prism is π/24 -/
theorem volume_ratio_is_pi_over_24 (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  volume_ratio h r = Real.pi / 24 := by
  unfold volume_ratio
  -- The proof steps would go here
  sorry

#check volume_ratio_is_pi_over_24

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_is_pi_over_24_l82_8267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_odd_positive_integer_l82_8241

theorem hundredth_odd_positive_integer : 
  (fun n => 2 * n - 1) 100 = 199 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_odd_positive_integer_l82_8241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_bought_ten_packs_l82_8268

/-- Calculates the number of card packs bought given the total cards per pack,
    fraction of uncommon cards, and total uncommon cards obtained. -/
def calculate_packs_bought (cards_per_pack : ℕ) (uncommon_fraction : ℚ) (total_uncommon : ℕ) : ℚ :=
  (total_uncommon : ℚ) / ((cards_per_pack : ℚ) * uncommon_fraction)

theorem john_bought_ten_packs :
  let cards_per_pack : ℕ := 20
  let uncommon_fraction : ℚ := 1/4
  let total_uncommon : ℕ := 50
  calculate_packs_bought cards_per_pack uncommon_fraction total_uncommon = 10 := by
  sorry

#eval (calculate_packs_bought 20 (1/4) 50).num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_bought_ten_packs_l82_8268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_probability_l82_8265

/-- The number of blue marbles in the hat -/
def blue_marbles : ℕ := 4

/-- The number of yellow marbles in the hat -/
def yellow_marbles : ℕ := 3

/-- The total number of marbles in the hat -/
def total_marbles : ℕ := blue_marbles + yellow_marbles

/-- The probability of drawing all blue marbles before all yellow marbles -/
def prob_all_blue : ℚ := 3 / 7

theorem marble_probability :
  (Nat.choose total_marbles yellow_marbles / 
   Nat.choose (total_marbles - 1) (yellow_marbles - 1) : ℚ) = prob_all_blue := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_probability_l82_8265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_velocity_first_second_instantaneous_velocity_end_first_second_time_velocity_reaches_14_l82_8223

-- Define the distance function
noncomputable def f (x : ℝ) : ℝ := (2/3) * x^3 + x^2 + 2*x

-- Define the velocity function (derivative of f)
noncomputable def v (x : ℝ) : ℝ := 2 * x^2 + 2 * x + 2

-- Theorem for average velocity during the first second
theorem average_velocity_first_second :
  (f 1 - f 0) / (1 - 0) = 3 := by sorry

-- Theorem for instantaneous velocity at the end of the first second
theorem instantaneous_velocity_end_first_second :
  v 1 = 6 := by sorry

-- Theorem for time when velocity reaches 14 m/s
theorem time_velocity_reaches_14 :
  ∃ t : ℝ, t > 0 ∧ v t = 14 ∧ ∀ s, 0 < s ∧ s < t → v s < 14 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_velocity_first_second_instantaneous_velocity_end_first_second_time_velocity_reaches_14_l82_8223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_probability_l82_8240

/-- The probability of drawing two balls of the same color from a pocket containing
    four balls, where one is black and three are white. -/
theorem same_color_probability (total_balls : ℕ) (black_balls : ℕ) (white_balls : ℕ) :
  total_balls = 4 →
  black_balls = 1 →
  white_balls = 3 →
  (Nat.choose white_balls 2 : ℚ) / (Nat.choose total_balls 2) = 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_probability_l82_8240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l82_8218

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6) + Real.cos (2 * x)

theorem triangle_area_proof (A B C : ℝ) (a b c : ℝ) :
  f A = Real.sqrt 3 / 2 →
  a = 2 →
  B = Real.pi / 3 →
  0 < A ∧ A < Real.pi →
  0 < B ∧ B < Real.pi →
  0 < C ∧ C < Real.pi →
  A + B + C = Real.pi →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  (1 / 2) * a * b * Real.sin C = (3 + Real.sqrt 3) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l82_8218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_speed_increase_l82_8290

/-- Calculates the percentage increase in speed for a round trip given the initial speed and average speed -/
noncomputable def percentage_speed_increase (initial_speed average_speed : ℝ) : ℝ :=
  let return_speed := (2 * initial_speed * average_speed) / (2 * initial_speed - average_speed)
  ((return_speed - initial_speed) / initial_speed) * 100

/-- Theorem stating that for a round trip with initial speed 30 km/hr and average speed 34.5 km/hr, 
    the percentage increase in speed for the return trip is approximately 35.294% -/
theorem round_trip_speed_increase :
  abs (percentage_speed_increase 30 34.5 - 35.294) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_speed_increase_l82_8290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_solution_replacement_l82_8210

/-- Represents a sugar solution with a given volume and sugar content -/
structure SugarSolution where
  volume : ℝ
  sugarContent : ℝ

/-- Calculates the percentage of sugar in a solution -/
noncomputable def sugarPercentage (solution : SugarSolution) : ℝ :=
  (solution.sugarContent / solution.volume) * 100

/-- Mixes two sugar solutions -/
def mixSolutions (s1 s2 : SugarSolution) : SugarSolution :=
  { volume := s1.volume + s2.volume,
    sugarContent := s1.sugarContent + s2.sugarContent }

theorem sugar_solution_replacement
  (initialSolution : SugarSolution)
  (replacementSolution : SugarSolution)
  (h1 : sugarPercentage initialSolution = 10)
  (h2 : replacementSolution.volume = initialSolution.volume / 4)
  (h3 : sugarPercentage (mixSolutions
    { volume := initialSolution.volume * 3/4,
      sugarContent := initialSolution.sugarContent * 3/4 }
    replacementSolution) = 17) :
  sugarPercentage replacementSolution = 38 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_solution_replacement_l82_8210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_time_proof_l82_8202

/-- The time it takes for two people to complete a task together,
    given their individual completion times. -/
noncomputable def combined_time (t1 t2 : ℝ) : ℝ :=
  1 / (1 / t1 + 1 / t2)

theorem translation_time_proof :
  let jun_seok_time : ℝ := 4
  let yoon_yeol_time : ℝ := 12
  combined_time jun_seok_time yoon_yeol_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_time_proof_l82_8202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_formula_l82_8248

/-- The volume of a tetrahedron with all faces being equal triangles with side lengths a, b, and c -/
noncomputable def tetrahedron_volume (a b c : ℝ) : ℝ :=
  (1 / (6 * Real.sqrt 2)) * Real.sqrt ((a^2 + b^2 - c^2) * (b^2 + c^2 - a^2) * (c^2 + a^2 - b^2))

/-- Theorem: The volume of a tetrahedron with all faces being equal triangles with side lengths a, b, and c
    is equal to (1 / (6√2)) * √((a² + b² - c²)(b² + c² - a²)(c² + a² - b²)) -/
theorem tetrahedron_volume_formula (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ V : ℝ, V = tetrahedron_volume a b c ∧ 
  V = (1 / (6 * Real.sqrt 2)) * Real.sqrt ((a^2 + b^2 - c^2) * (b^2 + c^2 - a^2) * (c^2 + a^2 - b^2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_formula_l82_8248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_theorem_l82_8245

theorem polynomial_value_theorem (f : ℝ → ℝ) :
  (∃ p : Polynomial ℝ, Polynomial.degree p = 2012 ∧ f = λ x ↦ p.eval x) →
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 2013 → f k = 2 / k) →
  2014 * f 2014 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_theorem_l82_8245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l82_8260

theorem equation_solution : 
  ∃ x : ℝ, (3 / 5 : ℝ) ^ x = (5 / 3 : ℝ) ^ 9 ∧ x = -9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l82_8260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_proof_l82_8256

theorem triangle_angle_proof (A B C a b c : Real) :
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 + c^2 = b^2 + 2*a*c*Real.cos C →
  a = 2*b*Real.sin A →
  A = 2*π/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_proof_l82_8256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sequence_iff_k_positive_general_formula_l82_8242

-- Define the sequence a_n
def a : ℕ → ℝ → ℝ → ℝ
  | 0, a₁, k => a₁  -- Add case for n = 0
  | 1, a₁, _ => a₁
  | n + 2, a₁, k => (a (n + 1) a₁ k)^2 + k

-- Part 1
theorem increasing_sequence_iff_k_positive (a₁ : ℝ) (h : a₁ = 1) :
  (∀ n : ℕ, a (n + 1) a₁ k > a n a₁ k) ↔ k > 0 :=
sorry

-- Part 2
theorem general_formula (n : ℕ) (h : n > 0) :
  a n 3 0 = 3^(2^(n - 1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sequence_iff_k_positive_general_formula_l82_8242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_tangent_product_l82_8250

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b

/-- Represents a circle with radius r -/
structure Circle where
  r : ℝ
  h_r_pos : r > 0

/-- Helper function to determine if a line is tangent to the circle from a point on the ellipse -/
def is_tangent (C : Ellipse) (O : Circle) (P : ℝ × ℝ) (l : ℝ) : Prop :=
  P.1^2 / C.a^2 + P.2^2 / C.b^2 = 1 ∧
  ∃ x y : ℝ, x^2 + y^2 = O.r^2 ∧ (y - P.2) = l * (x - P.1)

/-- The theorem statement -/
theorem ellipse_circle_tangent_product (C : Ellipse) (O : Circle) :
  C.a = 2 →
  C.a^2 - C.b^2 = 8/3 →
  O.r < C.b →
  (∀ P : ℝ × ℝ, P.1^2 / C.a^2 + P.2^2 / C.b^2 = 1 →
    ∃ k : ℝ, ∀ l₁ l₂ : ℝ, 
      (is_tangent C O P l₁ ∧ is_tangent C O P l₂ ∧ l₁ ≠ l₂) → l₁ * l₂ = k) →
  O.r^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_tangent_product_l82_8250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l82_8283

-- Define the triangle and points
variable (A B C P Q : ℂ)
variable (a b c : ℝ)

-- Define the conditions
variable (h1 : Complex.abs (B - C) = a)
variable (h2 : Complex.abs (C - A) = b)
variable (h3 : Complex.abs (A - B) = c)

-- State the theorem
theorem triangle_inequality :
  a * Complex.abs (P - A) * Complex.abs (Q - A) + 
  b * Complex.abs (P - B) * Complex.abs (Q - B) + 
  c * Complex.abs (P - C) * Complex.abs (Q - C) ≥ a * b * c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l82_8283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_g_equation_l82_8221

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 30 / (x + 5)

noncomputable def g (x : ℝ) : ℝ := 3 * (Function.invFun f x)

-- State the theorem
theorem solve_g_equation :
  ∃ x : ℝ, g x = 18 ∧ x = 30 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_g_equation_l82_8221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_equality_l82_8204

/-- Given a triangle ABC with circumradius R, prove that if 
    (a cos A + b cos B + c cos C) / (a sin B + b sin C + c sin A) = (a + b + c) / (9R),
    where a, b, and c are the lengths of the sides opposite to angles A, B, and C respectively,
    then A = B = C = 60°. -/
theorem triangle_angle_equality (A B C : ℝ) (a b c R : ℝ) 
  (h_triangle : A + B + C = Real.pi)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0)
  (h_sides : a = 2 * R * Real.sin A ∧ b = 2 * R * Real.sin B ∧ c = 2 * R * Real.sin C)
  (h_equation : (a * Real.cos A + b * Real.cos B + c * Real.cos C) / (a * Real.sin B + b * Real.sin C + c * Real.sin A) = (a + b + c) / (9 * R)) :
  A = Real.pi/3 ∧ B = Real.pi/3 ∧ C = Real.pi/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_equality_l82_8204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_equation_solution_l82_8203

/-- Define the star operation -/
noncomputable def star (a b : ℝ) : ℝ := Real.sqrt (a + b) / Real.sqrt (a - b)

/-- Theorem: If x ★ 12 = 8, then x = 260/21 -/
theorem star_equation_solution (x : ℝ) (h : star x 12 = 8) : x = 260 / 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_equation_solution_l82_8203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_stationing_ways_l82_8213

-- Define the number of men and women
def num_men : ℕ := 4
def num_women : ℕ := 3

-- Define the number of areas
def num_areas : ℕ := 3

-- Define a type for areas
inductive Area
| A | B | C

-- Define a type for people
inductive Person
| Man (n : ℕ)
| Woman (n : ℕ)

-- Define the function to represent stationing
def stationed : Area → Person → Prop := sorry

-- Define the function to calculate the number of stationing ways
def stationing_ways : ℕ := sorry

-- Theorem statement
theorem correct_stationing_ways :
  (num_men = 4) →
  (num_women = 3) →
  (num_areas = 3) →
  (∀ a : Area, ∃ m w, stationed a (Person.Man m) ∧ stationed a (Person.Woman w)) →
  (stationed Area.A (Person.Man 0)) →
  stationing_ways = 72 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_stationing_ways_l82_8213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_xoy_plane_l82_8263

-- Define a point in 3D space
def Point3D := ℝ × ℝ × ℝ

-- Define the xoy plane
def xoy_plane : Set Point3D := {p : Point3D | p.2.2 = 0}

-- Define symmetry with respect to the xoy plane
def symmetric_point (p : Point3D) : Point3D :=
  (p.1, p.2.1, -p.2.2)

-- Theorem statement
theorem symmetry_xoy_plane :
  let P : Point3D := (1, 3, -5)
  symmetric_point P = (1, 3, 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_xoy_plane_l82_8263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_implies_cos_2x_eq_1_max_value_of_f_is_5_l82_8274

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x + 2 * Real.sqrt 3, Real.sin x)
def c : ℝ × ℝ := (0, 1)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def f (x : ℝ) : ℝ := dot_product (a x) ((b x).1 - 2 * c.1, (b x).2 - 2 * c.2)

theorem perpendicular_implies_cos_2x_eq_1 (x : ℝ) (h : dot_product (a x) c = 0) :
  Real.cos (2 * x) = 1 := by sorry

theorem max_value_of_f_is_5 :
  ∃ (x : ℝ), f x = 5 ∧ ∀ (y : ℝ), f y ≤ 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_implies_cos_2x_eq_1_max_value_of_f_is_5_l82_8274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extreme_points_properties_l82_8264

open Real

/-- The function f(x) = x^2 - 2x + a*ln(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + a * log x

/-- The derivative of f(x) -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 2*x - 2 + a/x

theorem f_extreme_points_properties (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
    f' a x₁ = 0 ∧ f' a x₂ = 0 ∧ 
    (∀ x, x > 0 → (f' a x = 0 → x = x₁ ∨ x = x₂))) →
  (0 < a ∧ a < 1/2) ∧
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
    f' a x₁ = 0 ∧ f' a x₂ = 0 ∧ 
    f a x₁ / x₂ > -3/2 - log 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extreme_points_properties_l82_8264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_is_constant_l82_8288

noncomputable section

/-- Ellipse C with equation x²/4 + y²/3 = 1 -/
def C : Set (ℝ × ℝ) :=
  {p | p.1^2 / 4 + p.2^2 / 3 = 1}

/-- Line with equation y = kx + b -/
def Line (k b : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 = k * p.1 + b}

/-- Distance from a point to a line -/
noncomputable def distancePointToLine (p : ℝ × ℝ) (k b : ℝ) : ℝ :=
  |k * p.1 - p.2 + b| / Real.sqrt (k^2 + 1)

/-- Intersection points of the line and the ellipse -/
def intersectionPoints (k b : ℝ) : Set (ℝ × ℝ) :=
  C ∩ Line k b

/-- Predicate for OA ⟂ OB -/
def perpendicularIntersections (k b : ℝ) : Prop :=
  ∀ A B, A ∈ intersectionPoints k b → B ∈ intersectionPoints k b →
    A.1 * B.1 + A.2 * B.2 = 0

theorem distance_to_line_is_constant :
  ∀ k b : ℝ, perpendicularIntersections k b →
    distancePointToLine (0, 0) k b = 2 * Real.sqrt 21 / 7 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_is_constant_l82_8288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_ones_twelve_dice_l82_8266

/-- The number of dice being rolled -/
def n : ℕ := 12

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The number of dice we want to show a specific value -/
def k : ℕ := 2

/-- The probability of rolling exactly k dice showing 1 out of n dice -/
def prob_k_ones (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / sides) ^ k * ((sides - 1) / sides) ^ (n - k)

/-- Rounds a rational number to the nearest thousandth -/
noncomputable def round_to_thousandth (q : ℚ) : ℚ :=
  (q * 1000).floor / 1000

theorem prob_two_ones_twelve_dice :
  round_to_thousandth (prob_k_ones n k) = 303/1000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_ones_twelve_dice_l82_8266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_center_l82_8219

/-- The center of a hyperbola given by the equation (3y-6)^2 / 8^2 - (4x-5)^2 / 3^2 = 1 is (5/4, 2) -/
theorem hyperbola_center :
  let f : ℝ × ℝ → ℝ := fun (x, y) ↦ (3 * y - 6)^2 / 64 - (4 * x - 5)^2 / 9
  ∃! c : ℝ × ℝ, (∀ p : ℝ × ℝ, f p = 1 → ∃ t : ℝ, p = c + t • (p - c)) ∧ c = (5/4, 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_center_l82_8219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersecting_line_l82_8271

/-- Definition of the parabola C -/
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x

/-- Definition of a point being on the parabola -/
def point_on_parabola (p : ℝ) (x y : ℝ) : Prop := parabola p x y

/-- Definition of a line parallel to OA -/
def parallel_to_OA (m : ℝ) : Prop := m = -2

/-- Definition of the distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (b₁ b₂ : ℝ) : ℝ := 
  |b₁ - b₂| / Real.sqrt 5

/-- Main theorem -/
theorem parabola_intersecting_line (p : ℝ) :
  p > 0 →
  point_on_parabola p 1 (-2) →
  ∃ (m b : ℝ),
    parallel_to_OA m ∧
    (∃ (x y : ℝ), parabola p x y ∧ y = m * x + b) ∧
    distance_between_parallel_lines 0 b = Real.sqrt 5 / 5 ∧
    m = -2 ∧
    b = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersecting_line_l82_8271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sin_plus_cos_ge_sqrt2_div_2_l82_8230

open Real MeasureTheory Measure Set

-- Define the interval [0, π]
def I : Set ℝ := Icc 0 Real.pi

-- Define the event E = {x ∈ [0, π] | sin x + cos x ≥ √2/2}
def E : Set ℝ := {x ∈ I | sin x + cos x ≥ Real.sqrt 2 / 2}

-- State the theorem
theorem probability_sin_plus_cos_ge_sqrt2_div_2 :
  (volume E) / (volume I) = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sin_plus_cos_ge_sqrt2_div_2_l82_8230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_decreasing_l82_8251

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (x^2 - 1) / x

-- State the theorem
theorem f_increasing_decreasing :
  (∀ x > 0, (deriv f) x > 0) ∧ (∀ x < 0, (deriv f) x > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_decreasing_l82_8251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_evaluate_l82_8232

open Real

-- Define m
noncomputable def m : ℝ := tan (60 * π / 180) - 1

-- State the theorem
theorem simplify_and_evaluate :
  (1 - 2 / (m + 1)) / ((m^2 - 2*m + 1) / (m^2 - m)) = (3 - sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_evaluate_l82_8232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_block_length_closest_to_midpoint_l82_8206

-- Define the ruler length
def ruler_length : ℚ := 10

-- Define the block's left edge position
def left_edge : ℚ := 3

-- Define the range for the block's right edge
def right_edge_min : ℚ := 5
def right_edge_max : ℚ := 6

-- Define the given length options
def length_options : List ℚ := [0.24, 4.4, 2.4, 3, 24]

-- Define a function to calculate the midpoint of the block
def block_midpoint : ℚ := (right_edge_min + right_edge_max) / 2 - left_edge

-- Define a function to find the closest value in a list to a given number
def closest_value (list : List ℚ) (target : ℚ) : ℚ :=
  list.foldl (fun acc x => if |x - target| < |acc - target| then x else acc) (list.head!)

-- Theorem stating that 2.4 is the closest to the block's midpoint
theorem block_length_closest_to_midpoint :
  closest_value length_options block_midpoint = 2.4 := by
  -- The proof goes here
  sorry

#eval closest_value length_options block_midpoint

end NUMINAMATH_CALUDE_ERRORFEEDBACK_block_length_closest_to_midpoint_l82_8206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_factor_solutions_k_factor_solutions_l82_8293

/-- The number of positive integer solutions to mn + nr + mr = 2(m + n + r) -/
def solutions_count : ℕ := 7

/-- The equation mn + nr + mr = 2(m + n + r) has exactly 7 positive integer solutions -/
theorem two_factor_solutions :
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 
    (∀ (m n r : ℕ), (m, n, r) ∈ S ↔ m * n + n * r + m * r = 2 * (m + n + r) ∧ m > 0 ∧ n > 0 ∧ r > 0) ∧
    S.card = solutions_count :=
by sorry

/-- For any integer k > 1, mn + nr + mr = k(m + n + r) has at least 3k + 1 positive integer solutions -/
theorem k_factor_solutions (k : ℕ) (h : k > 1) :
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 
    (∀ (m n r : ℕ), (m, n, r) ∈ S → m * n + n * r + m * r = k * (m + n + r) ∧ m > 0 ∧ n > 0 ∧ r > 0) ∧
    S.card ≥ 3 * k + 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_factor_solutions_k_factor_solutions_l82_8293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_point_condition_l82_8235

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * Real.sin x

-- State the theorem
theorem min_point_condition (a : ℝ) :
  (∀ x : ℝ, f a x ≥ f a (Real.pi/3)) ↔ a = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_point_condition_l82_8235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_base_angle_is_arccos_sqrt2_over_4_l82_8252

/-- Regular triangular prism with square lateral faces -/
structure RegularTriangularPrism where
  /-- Edge length of the prism -/
  edge_length : ℝ
  /-- Assumption that edge length is positive -/
  edge_positive : edge_length > 0

/-- The angle between the diagonal of a lateral face and a non-intersecting side of the base
    in a regular triangular prism with square lateral faces -/
noncomputable def diagonalBaseAngle (prism : RegularTriangularPrism) : ℝ :=
  Real.arccos (Real.sqrt 2 / 4)

/-- Theorem stating that the angle between the diagonal of a lateral face and a non-intersecting
    side of the base in a regular triangular prism with square lateral faces is arccos(√2/4) -/
theorem diagonal_base_angle_is_arccos_sqrt2_over_4 (prism : RegularTriangularPrism) :
    diagonalBaseAngle prism = Real.arccos (Real.sqrt 2 / 4) := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_base_angle_is_arccos_sqrt2_over_4_l82_8252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_count_difference_l82_8295

/-- Represents the number of games a person has --/
structure GameCount where
  new : ℕ
  old : ℕ

/-- Calculates the total number of games --/
def totalGames (gc : GameCount) : ℕ := gc.new + gc.old

/-- Katie's game count --/
def katie : GameCount := { new := 57, old := 39 }

/-- Friend 1's game count --/
def friend1 : GameCount := { new := 34, old := 28 }

/-- Friend 2's game count --/
def friend2 : GameCount := { new := 25, old := 32 }

/-- Friend 3's game count --/
def friend3 : GameCount := { new := 12, old := 21 }

/-- Theorem stating the difference in game counts --/
theorem game_count_difference :
  (totalGames katie : ℤ) - ((totalGames friend1 : ℤ) + (totalGames friend2 : ℤ) + (totalGames friend3 : ℤ)) = -56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_count_difference_l82_8295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_derivative_g_l82_8282

-- Function definitions
def f (x : ℝ) : ℝ := x^4 - 3*x^2 - 5*x + 6
noncomputable def g (x : ℝ) : ℝ := x * Real.sin x

-- Theorem statements
theorem derivative_f (x : ℝ) : 
  deriv f x = 4*x^3 - 6*x - 5 := by sorry

theorem derivative_g (x : ℝ) : 
  deriv g x = Real.sin x + x * Real.cos x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_derivative_g_l82_8282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_integer_roots_l82_8273

def quadratic_equation (p q : ℤ) : ℤ → Prop := fun x => x^2 + p*x + q = 0

def increase_coefficients (p q : ℤ) : ℤ × ℤ := (p + 1, q + 1)

def has_integer_roots (p q : ℤ) : Prop :=
  ∃ x y : ℤ, quadratic_equation p q x ∧ quadratic_equation p q y ∧ x ≠ y

theorem quadratic_integer_roots :
  let initial := (3, 2)
  let step1 := increase_coefficients initial.1 initial.2
  let step2 := increase_coefficients step1.1 step1.2
  let step3 := increase_coefficients step2.1 step2.2
  let step4 := increase_coefficients step3.1 step3.2
  has_integer_roots initial.1 initial.2 ∧
  has_integer_roots step1.1 step1.2 ∧
  has_integer_roots step2.1 step2.2 ∧
  has_integer_roots step3.1 step3.2 ∧
  has_integer_roots step4.1 step4.2 :=
by sorry

#check quadratic_integer_roots

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_integer_roots_l82_8273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lending_rate_calculation_l82_8233

noncomputable def borrowed_amount : ℝ := 5000
noncomputable def borrowing_period : ℝ := 2
noncomputable def borrowing_rate : ℝ := 4
noncomputable def lending_period : ℝ := 2
noncomputable def gain_per_year : ℝ := 150

noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem lending_rate_calculation :
  let borrowed_interest := simple_interest borrowed_amount borrowing_rate borrowing_period / borrowing_period
  let total_interest_earned := borrowed_interest + gain_per_year
  let lending_rate := total_interest_earned * 100 / borrowed_amount
  lending_rate = 7 := by
    -- Proof goes here
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lending_rate_calculation_l82_8233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l82_8224

noncomputable section

open Real

theorem trigonometric_identities :
  ∀ α : ℝ,
  (sin α = 5/13 → (cos α = 12/13 ∨ cos α = -12/13) ∧ (tan α = 5/12 ∨ tan α = -5/12)) ∧
  (tan α = 2 → 1 / (2 * sin α * cos α + cos α ^ 2) = 1) :=
by
  intro α
  constructor
  · intro h1
    constructor
    · sorry -- Proof for cos α
    · sorry -- Proof for tan α
  · intro h2
    sorry -- Proof for the second part

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l82_8224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_bailing_rate_l82_8246

/-- Proves the minimum bailing rate required for a leaking boat to reach shore safely --/
theorem minimum_bailing_rate
  (distance_to_shore : ℝ)
  (boat_leak_rate : ℝ)
  (boat_sinking_threshold : ℝ)
  (rowing_speed : ℝ)
  (h1 : distance_to_shore = 2)
  (h2 : boat_leak_rate = 15)
  (h3 : boat_sinking_threshold = 50)
  (h4 : rowing_speed = 4)
  : ∃ (min_bailing_rate : ℝ), 
    (min_bailing_rate ≥ 13.33 ∧ min_bailing_rate < 13.34) ∧ 
    (distance_to_shore / rowing_speed) * 60 * (boat_leak_rate - min_bailing_rate) ≤ boat_sinking_threshold :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_bailing_rate_l82_8246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rate_of_interest_is_12_5_percent_l82_8216

/-- Calculate the rate of interest given principal, simple interest, and time. -/
noncomputable def calculate_rate_of_interest (principal : ℝ) (simple_interest : ℝ) (time : ℝ) : ℝ :=
  (simple_interest * 100) / (principal * time)

/-- Theorem stating that the rate of interest is 0.125 given the specific conditions. -/
theorem rate_of_interest_is_12_5_percent 
  (principal : ℝ) (simple_interest : ℝ) (time : ℝ)
  (h_principal : principal = 400)
  (h_simple_interest : simple_interest = 100)
  (h_time : time = 2) :
  calculate_rate_of_interest principal simple_interest time = 0.125 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rate_of_interest_is_12_5_percent_l82_8216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_period_cos_2x_l82_8227

/-- The minimum positive period of cos(2x) is π -/
theorem min_period_cos_2x :
  ∃ (T : ℝ), T > 0 ∧ 
  (∀ x : ℝ, Real.cos (2 * (x + T)) = Real.cos (2 * x)) ∧
  (∀ T' : ℝ, T' > 0 → (∀ x : ℝ, Real.cos (2 * (x + T')) = Real.cos (2 * x)) → T ≤ T') :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_period_cos_2x_l82_8227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approximation_l82_8253

/-- Given a train traveling at 120 km/hr and crossing a pole in 6 seconds, 
    its length is approximately 200 meters. -/
theorem train_length_approximation (speed : ℝ) (time : ℝ) (length : ℝ) : 
  speed = 120 → -- Speed in km/hr
  time = 6 → -- Time to cross the pole in seconds
  length = speed * (1000 / 3600) * time → -- Convert speed to m/s and calculate length
  abs (length - 200) < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approximation_l82_8253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cauliflower_sales_calculation_l82_8228

noncomputable def total_earnings : ℝ := 380
noncomputable def broccoli_sales : ℝ := 57
noncomputable def carrot_sales : ℝ := 2 * broccoli_sales
noncomputable def spinach_sales : ℝ := (carrot_sales / 2) + 16

theorem cauliflower_sales_calculation :
  total_earnings - (broccoli_sales + carrot_sales + spinach_sales) = 136 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cauliflower_sales_calculation_l82_8228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l82_8299

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  acute : Bool

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : Real.sqrt (t.b + t.c) = Real.sqrt (2 * t.a * Real.sin (t.C + π/6)))
  (h2 : t.acute)
  (h3 : t.c = 2) :
  t.A = π/3 ∧ Real.sqrt 3/2 < t.a * Real.sin t.B ∧ t.a * Real.sin t.B < 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l82_8299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_positive_terms_l82_8254

theorem not_all_positive_terms (a b c d : ℝ) (e f g h : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (he : e < 0) (hf : f < 0) (hg : g < 0) (hh : h < 0) :
  ¬(a * e + b * c > 0 ∧ e * f + c * g > 0 ∧ f * d + g * h > 0 ∧ d * a + h * b > 0) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_positive_terms_l82_8254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l82_8276

noncomputable section

/-- The eccentricity of the ellipse -/
def e : ℝ := 2/3

/-- The x-coordinate of the left vertex -/
def a : ℝ := 3

/-- The semi-minor axis of the ellipse -/
noncomputable def b : ℝ := Real.sqrt (a^2 * (1 - e^2))

/-- The equation of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

/-- The slope of the line l -/
noncomputable def k : ℝ := 5 * Real.sqrt 3 / 3

/-- The equation of line l -/
def is_on_l (x y : ℝ) : Prop := y = k * x

/-- The equation of line l' -/
def is_on_l' (x y : ℝ) : Prop := y = k * (x + a)

/-- Point P is the intersection of l and the ellipse in the first quadrant -/
def P : ℝ × ℝ := sorry

/-- Point Q is the intersection of l' and the ellipse -/
def Q : ℝ × ℝ := sorry

/-- The origin -/
def O : ℝ × ℝ := (0, 0)

/-- The left vertex -/
def A : ℝ × ℝ := (-a, 0)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem ellipse_intersection_theorem :
  e = 2/3 ∧ 
  is_on_ellipse A.1 A.2 ∧
  is_on_ellipse P.1 P.2 ∧
  is_on_l P.1 P.2 ∧
  is_on_ellipse Q.1 Q.2 ∧
  is_on_l' Q.1 Q.2 ∧
  distance A Q = (1/2) * distance O P →
  k = 5 * Real.sqrt 3 / 3 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l82_8276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_form_parabola_l82_8217

-- Define the parametric equations for x and y
noncomputable def x (s : ℝ) : ℝ := 2^s - 3
noncomputable def y (s : ℝ) : ℝ := 4^s - 7 * 2^s - 1

-- Theorem stating that the points (x(s), y(s)) form a parabola
theorem points_form_parabola :
  ∃ (a b c : ℝ), ∀ (s : ℝ), y s = a * (x s)^2 + b * (x s) + c :=
by
  -- We'll use a = 1, b = -1, c = -13 as in the algebraic solution
  use 1, -1, -13
  intro s
  -- Expand the definitions of x and y
  simp [x, y]
  -- The rest of the proof would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_form_parabola_l82_8217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_iff_dot_product_equals_magnitude_product_l82_8207

theorem vector_parallel_iff_dot_product_equals_magnitude_product 
  {n : Type*} [Fintype n] (a b : EuclideanSpace ℝ n) : 
  (∃ (k : ℝ), a = k • b) ↔ |Inner.inner a b| = ‖a‖ * ‖b‖ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_iff_dot_product_equals_magnitude_product_l82_8207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sample_correct_l82_8226

/-- Represents the number of employees in each title category -/
structure TitleCount where
  senior : ℕ
  intermediate : ℕ
  junior : ℕ
deriving Repr

/-- Represents the company's employee structure -/
def company : TitleCount :=
  { senior := 45
  , intermediate := 90
  , junior := 15 }

/-- Total number of employees in the company -/
def totalEmployees : ℕ := company.senior + company.intermediate + company.junior

/-- Size of the stratified sample -/
def sampleSize : ℕ := 30

/-- Calculates the stratified sample count for a given title category -/
def stratifiedSampleCount (categoryCount : ℕ) : ℕ :=
  (sampleSize * categoryCount) / totalEmployees

/-- The expected stratified sample -/
def expectedSample : TitleCount :=
  { senior := stratifiedSampleCount company.senior
  , intermediate := stratifiedSampleCount company.intermediate
  , junior := stratifiedSampleCount company.junior }

/-- Theorem stating that the stratified sample matches the expected counts -/
theorem stratified_sample_correct : 
  expectedSample = { senior := 9, intermediate := 18, junior := 3 } := by
  sorry

#eval expectedSample

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sample_correct_l82_8226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l82_8289

theorem sequence_general_term (n : ℕ) (hn : n > 0) : 
  let a : ℕ → ℕ := fun k => 2^k - 1
  (List.range 5).map a = [1, 3, 7, 15, 31] :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l82_8289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_token_end_probability_l82_8236

/-- A move on the grid can be up, down, left, or right -/
inductive Move
| up
| down
| left
| right

/-- A sequence of moves -/
def MoveSequence := List Move

/-- The final position after a sequence of moves -/
def finalPosition (moves : MoveSequence) : Int × Int :=
  moves.foldl (fun (x, y) move =>
    match move with
    | Move.up => (x, y + 1)
    | Move.down => (x, y - 1)
    | Move.left => (x - 1, y)
    | Move.right => (x + 1, y)
  ) (0, 0)

/-- Check if a position is on the line |y| = |x| -/
def onDiagonal (pos : Int × Int) : Bool :=
  pos.1.natAbs = pos.2.natAbs

/-- The number of moves in each sequence -/
def numMoves : Nat := 6

/-- The total number of possible move sequences -/
def totalSequences : Nat := 4^numMoves

/-- The number of move sequences that end on the diagonal -/
def diagonalSequences : Nat := 884

theorem token_end_probability :
  (diagonalSequences : ℚ) / totalSequences = 221 / 1024 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_token_end_probability_l82_8236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_value_l82_8214

/-- The perpendicular bisector of a line segment is a line that passes through the midpoint of the segment and is perpendicular to it. -/
def is_perp_bisector (l : Set (ℝ × ℝ)) (p₁ p₂ : ℝ × ℝ) : Prop :=
  let midpoint := ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2)
  (midpoint ∈ l) ∧ 
  (∀ p ∈ l, (p.1 - p₁.1)^2 + (p.2 - p₁.2)^2 = (p.1 - p₂.1)^2 + (p.2 - p₂.2)^2)

/-- The line x - y = c is the perpendicular bisector of the line segment from (2,4) to (6,8). -/
def perp_bisector_condition (c : ℝ) : Prop :=
  is_perp_bisector {p : ℝ × ℝ | p.1 - p.2 = c} (2, 4) (6, 8)

theorem perpendicular_bisector_value :
  ∃ c : ℝ, perp_bisector_condition c ∧ c = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_value_l82_8214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_average_speeds_l82_8285

/-- Represents the average speed calculation for Tom's drive -/
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

theorem tom_average_speeds :
  let total_distance : ℝ := 240
  let total_time : ℝ := 8
  let partial_distance : ℝ := 30
  let partial_time : ℝ := 2
  (average_speed total_distance total_time = 30) ∧
  (average_speed partial_distance partial_time = 15) := by
  -- Unfold the definitions
  unfold average_speed
  -- Split the conjunction
  apply And.intro
  -- Prove the first part
  · norm_num
  -- Prove the second part
  · norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_average_speeds_l82_8285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_food_consumption_l82_8262

/-- Calculates the total food consumption for both sides in a war scenario --/
theorem total_food_consumption
  (food_per_soldier_side1 : ℕ)
  (food_difference : ℕ)
  (soldiers_side1 : ℕ)
  (soldier_difference : ℕ)
  (h1 : food_per_soldier_side1 = 10)
  (h2 : food_difference = 2)
  (h3 : soldiers_side1 = 4000)
  (h4 : soldier_difference = 500)
  : (food_per_soldier_side1 * soldiers_side1) +
    ((food_per_soldier_side1 - food_difference) * (soldiers_side1 - soldier_difference)) = 68000 := by
  sorry

-- Remove the #eval line as it's not necessary for building the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_food_consumption_l82_8262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fathers_age_l82_8257

/-- Priya's current age -/
def priya_age : ℕ := sorry

/-- Priya's father's current age -/
def father_age : ℕ := sorry

/-- The difference between Priya's father's age and Priya's age is 31 years -/
axiom age_difference : father_age - priya_age = 31

/-- The sum of their ages after 8 years will be 69 years -/
axiom future_age_sum : (priya_age + 8) + (father_age + 8) = 69

/-- Priya's father's present age is 42 years -/
theorem fathers_age : father_age = 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fathers_age_l82_8257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recreation_spending_ratio_l82_8205

theorem recreation_spending_ratio :
  ∀ (last_week_wages : ℝ),
  last_week_wages > 0 →
  (0.20 * (0.75 * last_week_wages)) / (0.30 * last_week_wages) = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recreation_spending_ratio_l82_8205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_difference_l82_8239

theorem min_abs_difference (x y : ℤ) (hx : x > 0) (hy : y > 0) (h : x * y - 4 * x + 5 * y = 221) :
  ∃ (a b : ℤ), a > 0 ∧ b > 0 ∧ a * b - 4 * a + 5 * b = 221 ∧ 
  |a - b| ≤ |x - y| ∧ 
  |a - b| = 66 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_difference_l82_8239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l82_8243

/-- Given two trains running in opposite directions, calculate the speed of the second train. -/
theorem train_speed_calculation (length1 length2 : ℝ) (speed1 : ℝ) (clear_time : ℝ) 
  (h1 : length1 = 111) 
  (h2 : length2 = 165) 
  (h3 : speed1 = 60) 
  (h4 : clear_time = 6.623470122390208) :
  ∃ speed2 : ℝ, 
    abs (speed2 - 89.916) < 0.001 ∧ 
    abs ((length1 + length2) / 1000 / (clear_time / 3600) - (speed1 + speed2)) < 0.001 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l82_8243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_smallest_absolute_value_l82_8259

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  arithmetic : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  n * (seq.a 1 + seq.a n) / 2

theorem seventh_term_smallest_absolute_value (seq : ArithmeticSequence) 
    (h13 : S seq 13 < 0) (h12 : S seq 12 > 0) :
    ∀ k, k ≠ 7 → |seq.a 7| ≤ |seq.a k| := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_smallest_absolute_value_l82_8259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unequal_probabilities_l82_8297

/-- Represents a normal distribution with mean μ and variance σ² -/
structure NormalDistribution (μ : ℝ) (σ : ℝ) where
  mean : ℝ := μ
  variance : ℝ := σ^2

/-- Represents the probability of a random variable falling within an interval -/
noncomputable def probability (dist : NormalDistribution μ σ) (a b : ℝ) : ℝ := sorry

/-- The physical quantity follows a normal distribution with mean 10 and some variance σ² -/
def physicalQuantity (σ : ℝ) : NormalDistribution 10 σ := ⟨10, σ^2⟩

theorem unequal_probabilities (σ : ℝ) :
  probability (physicalQuantity σ) 9.9 10.2 ≠ probability (physicalQuantity σ) 10 10.3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unequal_probabilities_l82_8297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_arithmetic_sum_l82_8212

/-- Definition of the sum of first n terms of a geometric sequence -/
noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

/-- Theorem: If S₃, S₉, and S₆ form an arithmetic sequence in a geometric sequence with common ratio q, then q³ = -1/2 -/
theorem geometric_sequence_arithmetic_sum (a₁ : ℝ) (q : ℝ) :
  a₁ ≠ 0 → q ≠ 1 → q ≠ 0 →
  let S := geometric_sum a₁ q
  (S 3 + S 6 = 2 * S 9) →
  q^3 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_arithmetic_sum_l82_8212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_value_l82_8261

theorem perpendicular_lines_a_value (a : ℝ) : 
  (∀ x y, a*x + 2*y + 1 = 0 → (3-a)*x - y + a = 0 → 
    (a*x + 2*y + 1 = 0 ∧ (3-a)*x - y + a = 0)) →
  (∃ x1 y1 x2 y2, x1 ≠ x2 ∧ a*x1 + 2*y1 + 1 = 0 ∧ a*x2 + 2*y2 + 1 = 0) →
  (∃ x1 y1 x2 y2, x1 ≠ x2 ∧ (3-a)*x1 - y1 + a = 0 ∧ (3-a)*x2 - y2 + a = 0) →
  (∀ x1 y1 x2 y2, a*x1 + 2*y1 + 1 = 0 ∧ a*x2 + 2*y2 + 1 = 0 ∧ x1 ≠ x2 → 
    (y2 - y1) / (x2 - x1) = -a/2) →
  (∀ x1 y1 x2 y2, (3-a)*x1 - y1 + a = 0 ∧ (3-a)*x2 - y2 + a = 0 ∧ x1 ≠ x2 → 
    (y2 - y1) / (x2 - x1) = 3-a) →
  (∀ x1 y1 x2 y2 u1 v1 u2 v2, 
    a*x1 + 2*y1 + 1 = 0 ∧ a*x2 + 2*y2 + 1 = 0 ∧ x1 ≠ x2 ∧
    (3-a)*u1 - v1 + a = 0 ∧ (3-a)*u2 - v2 + a = 0 ∧ u1 ≠ u2 →
    ((y2 - y1) / (x2 - x1)) * ((v2 - v1) / (u2 - u1)) = -1) →
  a = 1 ∨ a = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_value_l82_8261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_l82_8201

/-- The profit percentage given that the cost price of 40 articles 
    equals the selling price of 32 articles -/
theorem profit_percentage (cost_price selling_price : ℝ) : 
  (40 * cost_price = 32 * selling_price) →
  ((selling_price - cost_price) / cost_price * 100 = 25) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_l82_8201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_team_wins_l82_8229

theorem soccer_team_wins (total_games : ℕ) (win_percentage : ℚ) (h1 : total_games = 158) (h2 : win_percentage = 2/5) : 
  ⌊total_games * win_percentage⌋₊ = 63 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_team_wins_l82_8229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_not_eight_l82_8275

noncomputable def data : List ℝ := [5, 8, 8, 9, 10]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let μ := mean xs
  (xs.map (fun x => (x - μ) ^ 2)).sum / xs.length

theorem variance_not_eight :
  variance data ≠ 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_not_eight_l82_8275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friendly_hands_count_l82_8258

def is_friendly_hand (hand : List Nat) : Bool :=
  hand.length = 4 && hand.sum = 24 && hand.maximum? = some 7

def count_friendly_hands : Nat :=
  (List.sublists [1, 2, 3, 4, 5, 6, 7]).filter is_friendly_hand |>.length

theorem friendly_hands_count : count_friendly_hands = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friendly_hands_count_l82_8258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inflection_point_on_line_l82_8281

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * x + 4 * Real.sin x - Real.cos x

-- Define the first derivative of f
noncomputable def f' (x : ℝ) : ℝ := 3 + 4 * Real.cos x + Real.sin x

-- Define the second derivative of f
noncomputable def f'' (x : ℝ) : ℝ := -4 * Real.sin x + Real.cos x

-- Theorem statement
theorem inflection_point_on_line (x₀ : ℝ) :
  f'' x₀ = 0 → f x₀ = 3 * x₀ := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inflection_point_on_line_l82_8281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_integer_first_two_digits_l82_8220

/-- Given an expression 8k8 + k88 - 16y6 where k and y are non-zero digits and y = 6,
    the first two digits of the third integer in the expression are 16. -/
theorem third_integer_first_two_digits :
  ∀ k : ℕ, 1 ≤ k → k ≤ 9 →
  (let y : ℕ := 6
   let expr := 8 * k * 100 + 8 + k * 100 + k * 10 + 8 - (16 * y * 100 + y * 10 + 6)
   (expr / 100) % 100) = 16 := by
  sorry

#check third_integer_first_two_digits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_integer_first_two_digits_l82_8220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_in_fourth_quadrant_l82_8294

def z₁ : ℂ := Complex.mk 3 1
def z₂ : ℂ := Complex.mk 2 (-1)

theorem product_in_fourth_quadrant :
  let z := z₁ * z₂
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_in_fourth_quadrant_l82_8294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_money_distribution_l82_8277

theorem money_distribution (total : ℕ) (share_A share_B share_C share_D share_E : ℕ) : 
  (share_A + share_B + share_C + share_D + share_E = total) →
  (share_A : ℚ) / (share_B : ℚ) = 5 / 2 →
  (share_A : ℚ) / (share_C : ℚ) = 5 / 4 →
  (share_A : ℚ) / (share_D : ℚ) = 5 / 3 →
  (share_A : ℚ) / (share_E : ℚ) = 5 / 6 →
  share_C = share_D + 500 →
  share_E = 2 * share_B →
  share_A = 2500 := by
  intro h1 h2 h3 h4 h5 h6 h7
  sorry

#check money_distribution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_money_distribution_l82_8277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_zoo_items_l82_8298

/-- Represents the number of pieces of bread or treats -/
@[ext] structure Count where
  value : ℕ

/-- Represents a person's contribution of bread and treats -/
structure Contribution where
  bread : Count
  treats : Count

/-- The total number of items (bread and treats) in a contribution -/
def Contribution.total (c : Contribution) : ℕ := c.bread.value + c.treats.value

/-- Jane's contribution based on Wanda's treats -/
def jane_contribution (wanda_treats : Count) : Contribution := {
  treats := { value := wanda_treats.value / 2 },
  bread := { value := (wanda_treats.value / 2 * 3 / 4) }
}

/-- Wanda's contribution -/
def wanda_contribution : Contribution := {
  bread := { value := 90 },
  treats := { value := 30 }
}

/-- Carla's contribution -/
def carla_contribution : Contribution := {
  bread := { value := 40 },
  treats := { value := 40 * 5 / 2 }
}

/-- Peter's contribution -/
def peter_contribution : Contribution := {
  bread := { value := 140 * 2 / 3 },
  treats := { value := 140 / 3 }
}

/-- The main theorem stating the total number of items brought to the zoo -/
theorem total_zoo_items : 
  (jane_contribution wanda_contribution.treats).total + 
  wanda_contribution.total + 
  carla_contribution.total + 
  peter_contribution.total = 426 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_zoo_items_l82_8298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zoolander_observation_l82_8222

/-- Represents a planet in the Zoolander star system -/
structure Planet where
  id : Nat

/-- The type of the Zoolander star system -/
def ZoolanderSystem := Fin 2015 → Planet

/-- The function representing which planet each astronomer observes -/
def ObservationFunction := Fin 2015 → Fin 2015

/-- Predicate indicating if all distances between planets are different -/
def AllDistancesDifferent (system : ZoolanderSystem) : Prop := sorry

/-- Predicate indicating if each astronomer observes the closest planet -/
def ObserveClosestPlanet (system : ZoolanderSystem) (f : ObservationFunction) : Prop := sorry

theorem zoolander_observation (system : ZoolanderSystem) 
  (h1 : AllDistancesDifferent system) 
  (h2 : ∃ f : ObservationFunction, ObserveClosestPlanet system f) :
  ∃ p : Fin 2015, ∀ q : Fin 2015, (Classical.choose h2) q ≠ p :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zoolander_observation_l82_8222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_example_l82_8238

/-- Represents a square pyramid -/
structure SquarePyramid where
  baseEdge : ℝ
  altitude : ℝ

/-- Calculate the volume of a square pyramid -/
noncomputable def pyramidVolume (p : SquarePyramid) : ℝ :=
  (1 / 3) * p.baseEdge^2 * p.altitude

/-- Calculate the volume of a frustum formed by cutting a square pyramid -/
noncomputable def frustumVolume (original : SquarePyramid) (smaller : SquarePyramid) : ℝ :=
  pyramidVolume original - pyramidVolume smaller

theorem frustum_volume_example :
  let original := SquarePyramid.mk 12 8
  let smaller := SquarePyramid.mk 6 4
  frustumVolume original smaller = 336 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_example_l82_8238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_minus_four_l82_8272

theorem x_squared_minus_four (x : ℝ) :
  (6 * (2 : ℝ)^x = 256) →
  (x + 2) * (x - 2) = 60 - 16 * Real.log 6 / Real.log 2 + (Real.log 6 / Real.log 2)^2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_minus_four_l82_8272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l82_8231

open Real

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_sum_magnitude
  (a b : V)
  (ha : ‖a‖ = 2)
  (hb : ‖b‖ = 1)
  (hab : inner a b = ‖a‖ * ‖b‖ * cos (60 * π / 180)) :
  ‖a + 2 • b‖ = 2 * sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l82_8231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_is_six_l82_8244

/-- The polynomial whose roots form the parallelogram -/
def polynomial (z : ℂ) : ℂ := 3 * z^4 + 9 * Complex.I * z^3 + (-12 + 12 * Complex.I) * z^2 + (-27 - 3 * Complex.I) * z + (4 - 18 * Complex.I)

/-- Predicate to check if four complex numbers form a parallelogram -/
def IsParallelogram (a b c d : ℂ) : Prop :=
  b - a = d - c ∧ c - b = a - d

/-- Function to calculate the area of a parallelogram given by four complex points -/
noncomputable def ParallelogramArea (a b c d : ℂ) : ℝ :=
  Complex.abs ((b - a) * (c - a))

/-- The roots of the polynomial form a parallelogram in the complex plane -/
axiom roots_form_parallelogram : ∃ (a b c d : ℂ), polynomial a = 0 ∧ polynomial b = 0 ∧ polynomial c = 0 ∧ polynomial d = 0 ∧
  IsParallelogram a b c d

/-- The area of the parallelogram formed by the roots of the polynomial is 6 -/
theorem parallelogram_area_is_six :
  ∃ (a b c d : ℂ), polynomial a = 0 ∧ polynomial b = 0 ∧ polynomial c = 0 ∧ polynomial d = 0 ∧
  IsParallelogram a b c d ∧
  ParallelogramArea a b c d = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_is_six_l82_8244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_arrangement_count_l82_8296

def chessboard_size : ℕ := 5
def black_pieces : ℕ := 3
def white_pieces : ℕ := 2

/-- The number of ways to arrange pieces on a chessboard -/
def arrangement_count : ℕ := 1200

/-- Theorem stating that the number of arrangements is correct -/
theorem correct_arrangement_count :
  (∃! (row_b : Fin black_pieces → Fin chessboard_size) 
      (col_b : Fin black_pieces → Fin chessboard_size)
      (row_w : Fin white_pieces → Fin chessboard_size) 
      (col_w : Fin white_pieces → Fin chessboard_size),
    (∀ i j : Fin black_pieces, i ≠ j → 
      (row_b i ≠ row_b j ∧ col_b i ≠ col_b j)) ∧
    (∀ i j : Fin white_pieces, i ≠ j → 
      (row_w i ≠ row_w j ∧ col_w i ≠ col_w j)) ∧
    (∀ i : Fin black_pieces, ∀ j : Fin white_pieces,
      (row_b i ≠ row_w j ∧ col_b i ≠ col_w j))) →
  arrangement_count = 1200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_arrangement_count_l82_8296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_stub_ratio_l82_8287

/-- Represents the length of a candle stub as a function of time -/
noncomputable def candleStubLength (initialLength : ℝ) (burnTime : ℝ) (elapsedTime : ℝ) : ℝ :=
  initialLength * (burnTime - elapsedTime) / burnTime

theorem candle_stub_ratio 
  (initialLength : ℝ) 
  (burnTime1 burnTime2 elapsedTime : ℝ) 
  (hburnTime1 : burnTime1 = 5)
  (hburnTime2 : burnTime2 = 3)
  (helapsedTime : elapsedTime = 2.5) :
  candleStubLength initialLength burnTime1 elapsedTime = 
  3 * candleStubLength initialLength burnTime2 elapsedTime := by
  sorry

#check candle_stub_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_stub_ratio_l82_8287
