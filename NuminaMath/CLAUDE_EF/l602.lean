import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_2_36_l602_60285

/-- The acute angle between clock hands at a given time -/
noncomputable def clockAngle (hours : ℕ) (minutes : ℕ) : ℝ :=
  let hourAngle : ℝ := (hours % 12 + minutes / 60 : ℝ) * 30
  let minuteAngle : ℝ := minutes * 6
  let diff : ℝ := abs (minuteAngle - hourAngle)
  min diff (360 - diff)

/-- Theorem: The acute angle between clock hands at 2:36 is 138° -/
theorem clock_angle_at_2_36 :
  clockAngle 2 36 = 138 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_2_36_l602_60285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_complex_plane_l602_60234

theorem equilateral_triangle_complex_plane (ω : ℂ) (l : ℝ) : 
  Complex.abs ω = 3 →
  l > 2 →
  (l * ω - (ω + 1) = Complex.exp (Complex.I * Real.pi / 3) * ((ω^2 + 2) - (ω + 1))) →
  l = (1 + Real.sqrt 13) / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_complex_plane_l602_60234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_men_tshirt_cost_l602_60216

/-- Represents the cost of a white men's T-shirt -/
def W : ℝ := sorry

/-- The total number of employees -/
def total_employees : ℕ := 40

/-- The cost difference between men's and women's T-shirts -/
def cost_difference : ℝ := 5

/-- The cost of a black men's T-shirt -/
def black_men_cost : ℝ := 18

/-- The total amount spent on T-shirts -/
def total_spent : ℝ := 660

theorem white_men_tshirt_cost :
  (W * (total_employees / 4) + (W - cost_difference) * (total_employees / 4) +
   black_men_cost * (total_employees / 4) + (black_men_cost - cost_difference) * (total_employees / 4) = total_spent) →
  W = 20 := by
  intro h
  sorry

#check white_men_tshirt_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_men_tshirt_cost_l602_60216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_pairs_count_eq_divisors_of_square_l602_60281

theorem lcm_pairs_count_eq_divisors_of_square (n : ℕ) :
  (Finset.filter (fun p : ℕ × ℕ => Nat.lcm p.1 p.2 = n) (Finset.product (Finset.range (n + 1)) (Finset.range (n + 1)))).card = (Nat.divisors (n^2)).card :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_pairs_count_eq_divisors_of_square_l602_60281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_usain_lap_time_l602_60204

/-- Represents the time Usain is closer to a photographer -/
structure CloserTime where
  photographer : String
  duration : ℝ

/-- Represents the circular track -/
structure Track where

/-- Usain's run around the track -/
structure UsainRun where
  track : Track

/-- The total time of Usain's run -/
def totalTime (run : UsainRun) : ℝ := sorry

/-- The sequence of closer times during Usain's run -/
def closerTimes (run : UsainRun) : List CloserTime := sorry

/-- Marina's position is halfway around the track from Arina -/
axiom marina_halfway : ∀ (run : UsainRun), 
  ∃ (t : ℝ), t > 0 ∧ 
    closerTimes run = [
      ⟨"Arina", 4⟩, 
      ⟨"Marina", 21⟩, 
      ⟨"Arina", t⟩
    ]

/-- The main theorem: Usain's lap time is 42 seconds -/
theorem usain_lap_time (run : UsainRun) : 
  totalTime run = 42 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_usain_lap_time_l602_60204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kim_final_score_l602_60287

/-- Represents the points awarded for correct answers in each round -/
def points_per_round : Fin 4 → ℕ
  | 0 => 2  -- Easy
  | 1 => 3  -- Average
  | 2 => 5  -- Hard
  | 3 => 7  -- Expert

/-- Represents Kim's correct answers in each round -/
def correct_answers : Fin 4 → ℕ
  | 0 => 6  -- Easy
  | 1 => 2  -- Average
  | 2 => 4  -- Hard
  | 3 => 3  -- Expert

/-- Represents Kim's incorrect answers in each round -/
def incorrect_answers : Fin 4 → ℕ
  | 0 => 1  -- Easy
  | 1 => 2  -- Average
  | 2 => 2  -- Hard
  | 3 => 3  -- Expert

/-- The number of complex problems Kim solved in the hard round -/
def complex_problems : ℕ := 2

/-- The bonus points awarded for each complex problem -/
def bonus_per_complex : ℕ := 1

/-- The penalty points for each incorrect answer -/
def penalty_per_incorrect : ℕ := 1

theorem kim_final_score :
  (Finset.sum (Finset.range 4) (λ i => points_per_round i * correct_answers i) +
   complex_problems * bonus_per_complex -
   Finset.sum (Finset.range 4) (λ i => incorrect_answers i * penalty_per_incorrect)) = 53 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kim_final_score_l602_60287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_angle_tangent_l602_60200

/-- Given a line y = x + 1 intersecting an ellipse mx^2 + ny^2 = 1 (where m > n > 0) at points A and B,
    if the midpoint of chord AB has an x-coordinate of -1/3, then the tangent of the angle between
    the two asymptotes of the hyperbola x^2/m^2 - y^2/n^2 = 1 is equal to 4/3. -/
theorem asymptote_angle_tangent (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m > n) :
  let line := λ (x : ℝ) ↦ x + 1
  let ellipse := λ (x y : ℝ) ↦ m * x^2 + n * y^2 = 1
  let hyperbola := λ (x y : ℝ) ↦ x^2 / m^2 - y^2 / n^2 = 1
  let intersect_points := {p : ℝ × ℝ | ellipse p.1 p.2 ∧ p.2 = line p.1}
  let midpoint_x := -1/3
  intersect_points.Nonempty ∧ 
  (∃ A B : ℝ × ℝ, A ∈ intersect_points ∧ B ∈ intersect_points ∧ A ≠ B ∧ 
    (A.1 + B.1) / 2 = midpoint_x) →
  let asymptote_angle := 2 * Real.arctan (n / m)
  Real.tan asymptote_angle = 4/3
:= by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_angle_tangent_l602_60200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_required_rd_costs_l602_60245

/-- The R&D costs required to increase the average labor productivity by 1 million rubles per person -/
noncomputable def required_rd_costs (rd_costs : ℝ) (productivity_change : ℝ) : ℝ :=
  rd_costs / productivity_change

/-- Given R&D costs and productivity change, prove the required R&D costs -/
theorem prove_required_rd_costs (rd_costs productivity_change : ℝ) 
  (h1 : rd_costs = 3157.61)
  (h2 : productivity_change = 0.69) :
  required_rd_costs rd_costs productivity_change = 4576 := by
  sorry

/-- Approximate evaluation of the required R&D costs -/
def approximate_required_rd_costs : Float :=
  3157.61 / 0.69

#eval approximate_required_rd_costs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_required_rd_costs_l602_60245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_hours_to_legal_limit_l602_60280

-- Define the legal limit for alcohol content
def legal_limit : ℝ := 0.09

-- Define the initial alcohol content
def initial_content : ℝ := 0.3

-- Define the rate of decrease per hour
def decrease_rate : ℝ := 0.75

-- Define the function for alcohol content after x hours
noncomputable def alcohol_content (x : ℝ) : ℝ := initial_content * decrease_rate ^ x

-- Define an approximation relation
def approx (x y : ℝ) (ε : ℝ) : Prop := abs (x - y) < ε

-- Theorem statement
theorem min_hours_to_legal_limit :
  ∃ x : ℝ, x ≥ 0 ∧ 
  (∀ y : ℝ, y ≥ 0 → alcohol_content y ≤ legal_limit → y ≥ x) ∧
  (approx x 4.2 0.05) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_hours_to_legal_limit_l602_60280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l602_60260

-- Define the lines
def line1 (m : ℝ) (x y : ℝ) : Prop := (3 + m) * x + 4 * y = 5 - 3 * m
def line2 (m : ℝ) (x y : ℝ) : Prop := 2 * x + (5 + m) * y = 8
def line3 (x y : ℝ) : Prop := 3 * x + 4 * y - 3 = 0
def line4 (m : ℝ) (x y : ℝ) : Prop := 6 * x + m * y + 14 = 0

-- Define perpendicularity and parallelism
def perpendicular (m : ℝ) : Prop := (3 + m) * (5 + m) + 4 * 2 = 0
def parallel (m : ℝ) : Prop := 3 * m - 24 = 0

-- Define the distance function
noncomputable def distance (x₀ y₀ : ℝ) : ℝ := |3 * x₀ + 4 * y₀ - 3| / Real.sqrt (3^2 + 4^2)

theorem problem_solution :
  (∃ m : ℝ, perpendicular m ∧ m = -13/3) ∧
  (∃ m : ℝ, parallel m ∧ m = 8) ∧
  distance 0 (-1.5) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l602_60260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_bounds_l602_60244

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 16

-- Define a point on the circle
def point_on_circle (P : ℝ × ℝ) : Prop := circle_eq P.1 P.2

-- Define tangent lines from P to ellipse at A and B
def tangent_lines (P A B : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧
  -- Additional conditions for tangency would be defined here
  True  -- Placeholder for additional conditions

-- Define the dot product of vectors PA and PB
def dot_product (P A B : ℝ × ℝ) : ℝ :=
  (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2)

-- Theorem statement
theorem dot_product_bounds :
  ∀ P A B : ℝ × ℝ,
  point_on_circle P →
  tangent_lines P A B →
  33/4 ≤ dot_product P A B ∧ dot_product P A B ≤ 165/16 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_bounds_l602_60244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_l602_60278

open Real

noncomputable def sequenceLimit (n : ℕ) : ℝ := 
  Real.sqrt (n + 2 : ℝ) * (Real.sqrt (n + 3 : ℝ) - Real.sqrt (n - 4 : ℝ))

theorem sequence_limit : 
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |sequenceLimit n - (7/2)| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_l602_60278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beth_average_speed_l602_60203

/-- Calculates the average speed given distance and time -/
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

theorem beth_average_speed :
  let jerry_speed : ℝ := 40
  let jerry_time : ℝ := 0.5
  let jerry_distance := jerry_speed * jerry_time
  let beth_distance := jerry_distance + 5
  let beth_time := jerry_time + 1/3
  average_speed beth_distance beth_time = 30 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beth_average_speed_l602_60203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_imaginary_axis_length_l602_60231

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 8 = 1

-- Define the length of the imaginary axis
noncomputable def imaginary_axis_length : ℝ := 4 * Real.sqrt 2

-- Theorem statement
theorem hyperbola_imaginary_axis_length :
  ∀ x y : ℝ, hyperbola_equation x y → imaginary_axis_length = 4 * Real.sqrt 2 :=
by
  -- The proof is skipped using 'sorry'
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_imaginary_axis_length_l602_60231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_equals_13_l602_60265

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  a6_minus_a5 : a 6 - a 5 = 2
  a11 : a 11 = 21

/-- Sum of first k terms of an arithmetic sequence -/
def sum_k (seq : ArithmeticSequence) (k : ℕ) : ℚ :=
  (k : ℚ) * seq.a 1 + (k * (k - 1) : ℚ) / 2 * (seq.a 2 - seq.a 1)

/-- Theorem stating that k = 13 for the given conditions -/
theorem k_equals_13 (seq : ArithmeticSequence) :
    sum_k seq 13 = 169 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_equals_13_l602_60265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_between_specific_spheres_l602_60240

/-- The volume of a sphere with radius r -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The volume between two concentric spheres -/
noncomputable def volume_between_spheres (r₁ r₂ : ℝ) : ℝ := sphere_volume r₂ - sphere_volume r₁

theorem volume_between_specific_spheres :
  volume_between_spheres 4 8 = (1792 / 3) * Real.pi := by
  -- Expand the definitions
  unfold volume_between_spheres sphere_volume
  -- Simplify the expressions
  simp [Real.pi]
  -- The rest of the proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_between_specific_spheres_l602_60240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_height_calculation_l602_60214

/-- Right pyramid with square base -/
structure RightPyramid where
  /-- Side length of the square base -/
  baseSide : ℝ
  /-- Distance from apex to any base vertex -/
  apexToVertex : ℝ

/-- Height of the pyramid from apex to center of base -/
noncomputable def pyramidHeight (p : RightPyramid) : ℝ :=
  Real.sqrt (p.apexToVertex^2 - 2 * p.baseSide^2)

theorem pyramid_height_calculation (p : RightPyramid) 
  (h1 : p.baseSide = 8)
  (h2 : p.apexToVertex = 11) : 
  pyramidHeight p = Real.sqrt 89 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_height_calculation_l602_60214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_initial_storks_count_solution_is_correct_l602_60210

def initial_storks_count (initial_birds : ℕ) (joined_storks : ℕ) : ℕ := 3

theorem prove_initial_storks_count (initial_birds : ℕ) (joined_storks : ℕ) : 
  initial_birds = 6 ∧ 
  joined_storks = 2 ∧ 
  initial_birds = (initial_storks_count initial_birds joined_storks + joined_storks) + 1 := by
  -- Define the initial number of birds
  have h1 : initial_birds = 6 := by sorry
  
  -- Define the number of storks that joined
  have h2 : joined_storks = 2 := by sorry
  
  -- Define the relationship between birds and storks after joining
  have h3 : initial_birds = (initial_storks_count initial_birds joined_storks + joined_storks) + 1 := by sorry
  
  -- Combine all conditions
  exact ⟨h1, h2, h3⟩

-- The theorem to prove
theorem solution_is_correct : initial_storks_count 6 2 = 3 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_initial_storks_count_solution_is_correct_l602_60210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l602_60298

theorem max_value_expression (a b c d : ℝ) 
  (ha : a ∈ Set.Icc (-7.5 : ℝ) 7.5) 
  (hb : b ∈ Set.Icc (-7.5 : ℝ) 7.5) 
  (hc : c ∈ Set.Icc (-7.5 : ℝ) 7.5) 
  (hd : d ∈ Set.Icc (-7.5 : ℝ) 7.5) : 
  (∀ x y z w : ℝ, x ∈ Set.Icc (-7.5 : ℝ) 7.5 → y ∈ Set.Icc (-7.5 : ℝ) 7.5 → 
    z ∈ Set.Icc (-7.5 : ℝ) 7.5 → w ∈ Set.Icc (-7.5 : ℝ) 7.5 → 
    x + 2*y + z + 2*w - x*y - y*z - z*w - w*x ≤ 240) ∧ 
  (∃ x y z w : ℝ, x ∈ Set.Icc (-7.5 : ℝ) 7.5 ∧ y ∈ Set.Icc (-7.5 : ℝ) 7.5 ∧ 
    z ∈ Set.Icc (-7.5 : ℝ) 7.5 ∧ w ∈ Set.Icc (-7.5 : ℝ) 7.5 ∧
    x + 2*y + z + 2*w - x*y - y*z - z*w - w*x = 240) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l602_60298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_theorem_l602_60279

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallelLP : Line → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (parallelPP : Plane → Plane → Prop)
variable (projection : Line → Plane → Line → Prop)

-- Theorem statement
theorem geometry_theorem 
  (a b c : Line) (α β : Plane) :
  -- Statement 1
  (∀ c α β, perpendicularLP c α → perpendicularLP c β → parallelPP α β) ∧
  -- Statement 2
  (∀ b c α, subset b α → ¬subset c α → parallelLP c α → parallel b c) ∧
  -- Statement 3
  (∀ a b c β, subset b β → projection a β c → perpendicular b c → perpendicular b a) ∧
  -- Statement 4 (negation, as it's false)
  (∃ b β α, subset b β ∧ perpendicularLP b α ∧ ¬perpendicularPP β α) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_theorem_l602_60279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_yield_is_69_l602_60288

def initial_yield : ℕ := 20
def first_harvest_increase : ℚ := 15 / 100
def second_harvest_increase : ℚ := 20 / 100
def third_harvest_increase : ℚ := 17 / 100
def loss_percentage : ℚ := 5 / 100

def first_harvest_duration : ℕ := 15
def second_harvest_duration : ℕ := 20
def third_harvest_duration : ℕ := 25

def total_duration : ℕ := first_harvest_duration + second_harvest_duration + third_harvest_duration

def calculate_harvest (previous_yield : ℕ) (increase : ℚ) : ℕ :=
  (((previous_yield : ℚ) * (1 + increase) * (1 - loss_percentage)).floor.toNat)

def first_harvest : ℕ := calculate_harvest initial_yield first_harvest_increase
def second_harvest : ℕ := calculate_harvest first_harvest second_harvest_increase
def third_harvest : ℕ := calculate_harvest second_harvest third_harvest_increase

theorem total_yield_is_69 :
  first_harvest + second_harvest + third_harvest = 69 ∧
  total_duration = 60 := by
  sorry

#eval first_harvest
#eval second_harvest
#eval third_harvest
#eval first_harvest + second_harvest + third_harvest
#eval total_duration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_yield_is_69_l602_60288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_four_questions_correct_l602_60299

/-- The probability of correctly answering a single question -/
def p_correct : ℝ := 0.8

/-- The probability of incorrectly answering a single question -/
def p_incorrect : ℝ := 1 - p_correct

/-- The number of preset questions in the competition -/
def total_questions : ℕ := 5

/-- A contestant advances if they answer two consecutive questions correctly -/
def advance_condition (answers : List Bool) : Bool :=
  (List.zip answers (List.drop 1 answers)).any (fun (a, b) => a ∧ b)

/-- The probability of answering exactly 4 questions before advancing -/
def prob_four_questions : ℝ := 0.128

theorem prob_four_questions_correct :
  prob_four_questions = p_correct * p_correct * p_correct * p_incorrect +
                        p_incorrect * p_correct * p_correct * p_correct :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_four_questions_correct_l602_60299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_M_l602_60213

/-- The ellipse defined by x^2/9 + y^2/4 = 1 --/
def Ellipse : Set (ℝ × ℝ) :=
  {p | (p.1^2 / 9) + (p.2^2 / 4) = 1}

/-- The fixed point M(1,0) --/
def M : ℝ × ℝ := (1, 0)

/-- The distance function between two points in ℝ² --/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The minimum distance from any point on the ellipse to M is 4√5/5 --/
theorem min_distance_ellipse_to_M :
  ∃ (d : ℝ), d = 4 * Real.sqrt 5 / 5 ∧
  ∀ (p : ℝ × ℝ), p ∈ Ellipse →
  distance p M ≥ d ∧
  ∃ (q : ℝ × ℝ), q ∈ Ellipse ∧ distance q M = d :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_M_l602_60213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_values_l602_60209

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + 2 * Real.cos x

-- State the theorem
theorem f_max_min_values :
  ∃ (max min : ℝ),
    (∀ x, π / 3 ≤ x ∧ x ≤ 4 * π / 3 → f x ≤ max) ∧
    (∃ x, π / 3 ≤ x ∧ x ≤ 4 * π / 3 ∧ f x = max) ∧
    (∀ x, π / 3 ≤ x ∧ x ≤ 4 * π / 3 → min ≤ f x) ∧
    (∃ x, π / 3 ≤ x ∧ x ≤ 4 * π / 3 ∧ f x = min) ∧
    max = 7 / 4 ∧ min = -2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_values_l602_60209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_160_power_9_l602_60254

noncomputable def rotation_matrix (θ : Real) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ],
    ![Real.sin θ,  Real.cos θ]]

theorem rotation_160_power_9 :
  ∃ (n : ℕ), n > 0 ∧ rotation_matrix (160 * Real.pi / 180) ^ n = 1 ∧
  ∀ (m : ℕ), m > 0 → m < n → rotation_matrix (160 * Real.pi / 180) ^ m ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_160_power_9_l602_60254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_even_shift_l602_60222

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 4

-- Define what it means for a function to be even
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

-- State the theorem
theorem quadratic_even_shift (a : ℝ) :
  is_even (λ x ↦ f a (x + 1)) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_even_shift_l602_60222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_centers_l602_60274

noncomputable def f (x : ℝ) := Real.sin x * Real.cos x + Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3

def is_center_of_symmetry (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x : ℝ, f (p.1 + x) - p.2 = -(f (p.1 - x) - p.2)

theorem f_symmetry_centers (k : ℤ) :
  is_center_of_symmetry f (k * Real.pi / 2 - Real.pi / 6, -Real.sqrt 3 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_centers_l602_60274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_36_l602_60217

def is_factor (n k : ℕ) : Prop := k ∣ n

def factor_sum (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (fun k => k ∣ n) |>.sum id

theorem sum_of_factors_36 : factor_sum 36 = 91 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_36_l602_60217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_product_perfect_square_l602_60270

-- Define factorial function
def factorial (n : ℕ) : ℕ := Nat.factorial n

-- Define is_perfect_square predicate
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

-- State the theorem
theorem factorial_product_perfect_square :
  is_perfect_square (factorial 99 * factorial 100) ∧
  ¬is_perfect_square (factorial 97 * factorial 98) ∧
  ¬is_perfect_square (factorial 97 * factorial 99) ∧
  ¬is_perfect_square (factorial 98 * factorial 99) ∧
  ¬is_perfect_square (factorial 98 * factorial 100) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_product_perfect_square_l602_60270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l602_60247

theorem triangle_abc_properties (A B C : ℝ) (h1 : Real.sin (C - A) = 1) 
  (h2 : Real.sin B = 1/3) (h3 : 0 < A ∧ A < π) (h4 : 0 < B ∧ B < π) (h5 : 0 < C ∧ C < π) 
  (h6 : A + B + C = π) :
  let b := Real.sqrt 6
  Real.sin A = Real.sqrt 3 / 3 ∧ 
  (1/2) * b * (3 * Real.sqrt 2) * (Real.sqrt 6 / 3) = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l602_60247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_probability_l602_60283

/-- The probability of getting heads on a single flip --/
noncomputable def p_heads : ℝ := 1/3

/-- The probability of getting tails on a single flip --/
noncomputable def p_tails : ℝ := 2/3

/-- The probability that all three players get their first head on the n-th flip --/
noncomputable def prob_all_n (n : ℕ) : ℝ := (p_tails ^ (n - 1) * p_heads) ^ 3

/-- The sum of probabilities for all possible n --/
noncomputable def total_prob : ℝ := ∑' n, prob_all_n n

/-- Theorem stating that the probability of all three players flipping the same number of times is 1/19 --/
theorem coin_flip_probability : total_prob = 1/19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_probability_l602_60283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l602_60226

noncomputable def f (θ : ℝ) : ℝ := Real.cos θ / (2 + Real.sin θ)

theorem range_of_f :
  ∀ y : ℝ, (∃ θ : ℝ, f θ = y) → -Real.sqrt 3 / 3 ≤ y ∧ y ≤ Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l602_60226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_g_eq_5_l602_60271

-- Define the piecewise function g(x)
noncomputable def g (x : ℝ) : ℝ :=
  if x < 0 then 4 * x + 8 else 3 * x - 15

-- Theorem stating the solutions of g(x) = 5
theorem solutions_of_g_eq_5 :
  {x : ℝ | g x = 5} = {-3/4, 20/3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_g_eq_5_l602_60271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l602_60232

noncomputable def f (x : ℝ) : ℝ := Real.log (|x| + 1) / Real.log 2 + 2^x + 2^(-x)

theorem f_inequality_range (x : ℝ) : 
  f (x + 1) < f (2 * x) ↔ x < -1/3 ∨ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l602_60232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_distance_theorem_l602_60289

-- Define the parametric equation of line l₁
noncomputable def l₁ (a t : ℝ) : ℝ × ℝ := (1 + t, a + 3 * t)

-- Define the polar equation of line l₂
def l₂_polar (ρ θ : ℝ) : Prop := ρ * Real.sin θ - 3 * ρ * Real.cos θ + 4 = 0

-- Convert l₂ to Cartesian coordinates
def l₂_cartesian (x y : ℝ) : Prop := 3 * x - y - 4 = 0

-- Define the distance between two parallel lines
noncomputable def distance_parallel_lines (a b c₁ c₂ : ℝ) : ℝ := 
  |c₁ - c₂| / Real.sqrt (a^2 + b^2)

theorem line_distance_theorem (a : ℝ) : 
  (∃ t : ℝ, l₁ a t = (1 + t, a + 3 * t)) ∧ 
  (∀ ρ θ : ℝ, l₂_polar ρ θ ↔ l₂_cartesian (ρ * Real.cos θ) (ρ * Real.sin θ)) ∧
  (distance_parallel_lines 3 (-1) (a - 3) (-4) = Real.sqrt 10) →
  a = 9 ∨ a = -11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_distance_theorem_l602_60289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_B_in_compound_X_l602_60212

/-- Represents the ratio of elements A, B, and C in compound X -/
def ratio : Fin 3 → ℕ
| 0 => 2  -- Element A
| 1 => 10 -- Element B
| 2 => 3  -- Element C

/-- Total parts in the ratio -/
def total_parts : ℕ := (List.sum (List.map ratio (List.range 3)))

/-- Total weight of compound X in grams -/
def total_weight : ℕ := 330

/-- Conversion factor from grams to milligrams -/
def mg_per_gram : ℕ := 1000

/-- Weight of element B in milligrams -/
def weight_B_mg : ℕ := (total_weight * ratio 1 * mg_per_gram) / total_parts

theorem weight_of_B_in_compound_X :
  weight_B_mg = 220000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_B_in_compound_X_l602_60212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_l602_60215

/-- A tetrahedron is a three-dimensional geometric shape with four triangular faces -/
structure Tetrahedron where
  surfaceArea : ℝ
  inscribedSphereRadius : ℝ

/-- The volume of a tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ := (1/3) * t.surfaceArea * t.inscribedSphereRadius

/-- Theorem: The volume of a tetrahedron is one-third of the product of its surface area and the radius of its inscribed sphere -/
theorem tetrahedron_volume (t : Tetrahedron) : 
  volume t = (1/3) * t.surfaceArea * t.inscribedSphereRadius := by
  -- Unfold the definition of volume
  unfold volume
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_l602_60215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_spending_l602_60219

/-- Represents the amount of money Mark spent in the first store before the additional $14 -/
noncomputable def first_store_spend : ℚ := 90

/-- The total amount Mark started with -/
noncomputable def total_money : ℚ := 180

/-- The additional amount Mark spent in the first store -/
noncomputable def first_store_extra : ℚ := 14

/-- The amount Mark spent in the second store from his initial money -/
noncomputable def second_store_fraction : ℚ := total_money / 3

/-- The additional amount Mark spent in the second store -/
noncomputable def second_store_extra : ℚ := 16

theorem mark_spending :
  (first_store_spend + first_store_extra) / total_money = 104 / 180 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_spending_l602_60219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vovochka_theorem_l602_60239

/-- Vovochka's addition method for three-digit numbers -/
def vovochka_add (a b : Nat) : Nat :=
  let a1 := a / 100
  let a2 := (a / 10) % 10
  let a3 := a % 10
  let b1 := b / 100
  let b2 := (b / 10) % 10
  let b3 := b % 10
  ((a1 + b1) % 10) * 100 + ((a2 + b2) % 10) * 10 + ((a3 + b3) % 10)

/-- The number of pairs of three-digit numbers where Vovochka's method gives the correct answer -/
def correct_vovochka_count : Nat :=
  81 * 55 * 55

/-- The smallest positive difference between normal addition and Vovochka's method for three-digit numbers -/
def min_vovochka_difference : Nat := 1800

theorem vovochka_theorem :
  (∀ a b : Nat, 100 ≤ a ∧ a < 1000 ∧ 100 ≤ b ∧ b < 1000 →
    (vovochka_add a b = a + b ↔ (a / 100 + b / 100 < 10 ∧
                                 (a / 10 % 10 + b / 10 % 10) < 10 ∧
                                 (a % 10 + b % 10) < 10))) ∧
  correct_vovochka_count = 244620 ∧
  min_vovochka_difference = 1800 ∧
  (∀ a b : Nat, 100 ≤ a ∧ a < 1000 ∧ 100 ≤ b ∧ b < 1000 →
    vovochka_add a b ≠ a + b →
    (vovochka_add a b : Int) - (a + b : Int) ≥ min_vovochka_difference ∨
    (a + b : Int) - (vovochka_add a b : Int) ≥ min_vovochka_difference) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vovochka_theorem_l602_60239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_sqrt23_over_4_l602_60255

/-- Qin Jiushao's formula for triangle area -/
noncomputable def qin_jiushao_area (a b c : ℝ) : ℝ :=
  Real.sqrt (1/4 * (c^2 * a^2 - ((c^2 + a^2 - b^2)/2)^2))

/-- Theorem stating that the area of a triangle with sides √2, √3, and 2 is √23/4 -/
theorem triangle_area_sqrt23_over_4 :
  qin_jiushao_area (Real.sqrt 2) (Real.sqrt 3) 2 = Real.sqrt 23 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_sqrt23_over_4_l602_60255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_polar_l602_60202

/-- The length of the chord formed by the intersection of a line and a circle in polar coordinates -/
theorem chord_length_polar : 
  ∃ A B : ℝ × ℝ, 
    (A.1^2 + A.2^2 = 1) ∧ 
    (B.1^2 + B.2^2 = 1) ∧ 
    (A.2 - A.1 = 1) ∧ 
    (B.2 - B.1 = 1) ∧ 
    A ≠ B ∧
    Real.sqrt 2 = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_polar_l602_60202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_range_l602_60236

-- Define vectors a and b
noncomputable def a (x : Real) : Real × Real := (Real.sin x, -1)
noncomputable def b (x : Real) : Real × Real := (Real.sqrt 3 * Real.cos x, -1/2)

-- Define function f
noncomputable def f (x : Real) : Real := ((a x).1 + (b x).1) * (a x).1 + ((a x).2 + (b x).2) * (a x).2 - 1

-- Triangle ABC
structure Triangle (A B C : Real) (a b c : Real) where
  angle_sum : A + B + C = Real.pi
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  law_of_cosines : a^2 = b^2 + c^2 - 2*b*c*Real.cos A

-- Theorem statement
theorem triangle_side_sum_range (A : Real) (b c : Real) :
  f (A/2) = 3/2 →
  ∃ (B C : Real), Triangle A B C 2 b c →
  2 < b + c ∧ b + c ≤ 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_range_l602_60236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_largest_angle_l602_60276

/-- The sum of interior angles of a hexagon in degrees -/
def hexagon_angle_sum : ℝ := 720

/-- The ratio of angles in the hexagon -/
def angle_ratio : List ℝ := [2, 2, 2, 3, 4, 5]

/-- The measure of the largest angle in the hexagon -/
def largest_angle : ℝ := 200

theorem hexagon_largest_angle :
  let total_ratio : ℝ := angle_ratio.sum
  let angle_unit : ℝ := hexagon_angle_sum / total_ratio
  largest_angle = angle_unit * (angle_ratio.maximum.getD 0) := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_largest_angle_l602_60276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_a_onto_b_l602_60261

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b : ℝ × ℝ := (4, 3)

noncomputable def proj_vector (v w : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (v.1 * w.1 + v.2 * w.2) / (w.1 * w.1 + w.2 * w.2)
  (scalar * w.1, scalar * w.2)

theorem projection_of_a_onto_b :
  proj_vector a b = (8/5, 6/5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_a_onto_b_l602_60261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_l602_60241

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate * (time : ℝ))

noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem interest_difference : 
  let principal := (15000 : ℝ)
  let compound_rate := (0.06 : ℝ)
  let simple_rate := (0.08 : ℝ)
  let time := (10 : ℕ)
  let compound_balance := round_to_nearest (compound_interest principal compound_rate time)
  let simple_balance := round_to_nearest (simple_interest principal simple_rate time)
  (simple_balance - compound_balance : ℤ) = 137 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_l602_60241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l602_60290

def A (a : ℝ) : Set ℝ := {x | 3 ≤ x ∧ x ≤ a + 5}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem problem_solution (a : ℝ) :
  (a = 2 → (Set.univ \ (A 2 ∪ B) = Set.Iic 2 ∪ Set.Ici 10 ∧
            (Set.univ \ A 2) ∩ B = Set.Ioo 2 3 ∪ Set.Ioo 7 10)) ∧
  (A a ∩ B = A a ↔ a < 5) := by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l602_60290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_elements_of_valid_set_l602_60284

def is_valid_set (T : Finset ℕ) : Prop :=
  1 ∈ T ∧
  1989 ∈ T ∧
  ∀ x ∈ T, x ≤ 1989 ∧
  ∀ y ∈ T, (T.sum id - y) % (T.card - 1) = 0

theorem max_elements_of_valid_set (T : Finset ℕ) (h : is_valid_set T) :
  T.card ≤ 29 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_elements_of_valid_set_l602_60284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_values_of_f_l602_60258

noncomputable def f (x : ℝ) := (1/3) * x^3 - 4*x

theorem max_min_values_of_f :
  ∃ (x_max x_min : ℝ),
    x_max ∈ Set.Icc (-3 : ℝ) 3 ∧
    x_min ∈ Set.Icc (-3 : ℝ) 3 ∧
    (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≤ f x_max) ∧
    (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x_min ≤ f x) ∧
    f x_max = 16/3 ∧
    f x_min = -16/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_values_of_f_l602_60258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sequences_l602_60207

def is_valid_sequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n ≤ n * Nat.sqrt n) ∧
  (∀ m n, m ≠ n → (m - n) ∣ (a m - a n))

theorem valid_sequences :
  ∀ a : ℕ → ℕ, is_valid_sequence a ↔ (∀ n, a n = 1) ∨ (∀ n, a n = n) :=
by sorry

#check valid_sequences

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sequences_l602_60207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_triangle_l602_60223

-- Define the triangle ABC
def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = Real.pi

-- Define the sine theorem
def sine_theorem (a b c : ℝ) (A B C : ℝ) : Prop :=
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) ∧
  c / (Real.sin C) = a / (Real.sin A)

-- Define the area of a triangle
noncomputable def triangle_area (a b c : ℝ) (A B C : ℝ) : ℝ :=
  (1/2) * b * c * Real.sin A

-- Theorem statement
theorem area_of_specific_triangle :
  ∀ a b c A B C : ℝ,
  triangle_ABC a b c A B C →
  sine_theorem a b c A B C →
  b = 2 →
  B = Real.pi/6 →
  C = Real.pi/4 →
  triangle_area a b c A B C = Real.sqrt 3 + 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_triangle_l602_60223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_digits_divisible_by_power_of_five_l602_60250

theorem odd_digits_divisible_by_power_of_five (n : ℕ) :
  ∃ m : ℕ,
    (∃ k : ℕ, 10^(n-1) ≤ m ∧ m < 10^n) ∧  -- m has exactly n digits
    m % 5^n = 0 ∧                        -- m is divisible by 5^n
    (∀ d : ℕ, d < n → (m / 10^d) % 10 % 2 = 1)  -- All digits of m are odd
:= by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_digits_divisible_by_power_of_five_l602_60250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prism_volume_l602_60291

-- Define the right prism with triangular bases
structure TriangularPrism where
  base_side : ℝ
  base_angle : ℝ
  height : ℝ

-- Define the conditions
def prism_conditions (p : TriangularPrism) : Prop :=
  p.base_side = 6 ∧ 
  p.base_angle = Real.pi/3 ∧ 
  p.base_side * p.height + 
    (p.base_side * p.height * Real.sin p.base_angle) * p.height + 
    (1/2 * p.base_side * p.base_side * Real.sin p.base_angle) = 36

-- Define the volume of the prism
noncomputable def prism_volume (p : TriangularPrism) : ℝ :=
  1/2 * p.base_side * p.base_side * p.height * Real.sin p.base_angle

-- Theorem statement
theorem max_prism_volume :
  ∀ p : TriangularPrism, prism_conditions p → 
    ∀ q : TriangularPrism, prism_conditions q → 
      prism_volume p ≤ 27 ∧ 
      (∃ r : TriangularPrism, prism_conditions r ∧ prism_volume r = 27) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prism_volume_l602_60291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_intercepts_problem_l602_60224

-- Define the two quadratic functions
def f (h j x : ℝ) := 4 * (x - h)^2 + j
def g (h k x : ℝ) := 5 * (x - h)^2 + k

-- Define the conditions
theorem quadratic_intercepts_problem (h j k : ℝ) 
  (hf_intercept : f h j 0 = 2030)
  (hg_intercept : g h k 0 = 2040)
  (hf_roots : ∃ (x1 x2 : ℕ), x1 > 0 ∧ x2 > 0 ∧ f h j (x1 : ℝ) = 0 ∧ f h j (x2 : ℝ) = 0)
  (hg_roots : ∃ (y1 y2 : ℕ), y1 > 0 ∧ y2 > 0 ∧ g h k (y1 : ℝ) = 0 ∧ g h k (y2 : ℝ) = 0) :
  h = 20.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_intercepts_problem_l602_60224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_inequality_count_l602_60233

theorem sqrt_inequality_count : 
  (Finset.range 10201 \ Finset.range 10001).card = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_inequality_count_l602_60233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l602_60251

theorem line_inclination_angle (x y : ℝ) :
  (Real.sqrt 3 * x - y + 2 = 0) → (Real.arctan (Real.sqrt 3) = π / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l602_60251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seedling_problem_l602_60294

/-- Represents the unit price and quantity of seedlings -/
structure Seedling where
  price : ℝ
  quantity : ℕ

/-- Proves the correct unit prices and minimum quantity of type B seedlings -/
theorem seedling_problem (a b : Seedling) :
  a.price = 1.5 * b.price →
  a.quantity + 10 = b.quantity →
  a.price * (a.quantity : ℝ) = 1200 →
  b.price * (b.quantity : ℝ) = 900 →
  (∃ (m : ℕ), m + (100 - m) = 100 ∧
    b.price * (m : ℝ) + a.price * ((100 - m) : ℝ) ≤ 1314 ∧
    ∀ (n : ℕ), n < m →
      b.price * (n : ℝ) + a.price * ((100 - n) : ℝ) > 1314) →
  a.price = 15 ∧ b.price = 10 ∧ m = 38 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seedling_problem_l602_60294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l602_60269

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (4^x - 2^(x+1))

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l602_60269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_union_l602_60257

def A : Finset Nat := {1, 2}
def B : Finset Nat := {1}

theorem number_of_subsets_union : Finset.card (Finset.powerset (A ∪ B)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_union_l602_60257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_price_proof_l602_60275

-- Define the original price
noncomputable def original_price : ℚ := 64

-- Define the fraction of the original price that was paid
noncomputable def fraction_paid : ℚ := 1 / 8

-- Define the price paid
noncomputable def price_paid : ℚ := 8

-- Theorem statement
theorem bicycle_price_proof :
  fraction_paid * original_price = price_paid ∧ 
  original_price = 64 := by
  constructor
  · -- Prove that fraction_paid * original_price = price_paid
    calc
      fraction_paid * original_price = (1 / 8) * 64 := rfl
      _ = 8 := by norm_num
  · -- Prove that original_price = 64
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_price_proof_l602_60275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jakes_drink_coke_volume_l602_60282

/-- Represents a drink mixture -/
structure Drink where
  coke : ℕ    -- parts of Coke
  sprite : ℕ  -- parts of Sprite
  dew : ℕ     -- parts of Mountain Dew
  total : ℝ   -- total volume in ounces

/-- Calculates the volume of Coke in the drink -/
noncomputable def coke_volume (d : Drink) : ℝ :=
  (d.coke : ℝ) * d.total / ((d.coke + d.sprite + d.dew) : ℝ)

/-- Theorem: The volume of Coke in Jake's drink is 6 ounces -/
theorem jakes_drink_coke_volume :
  let d : Drink := { coke := 2, sprite := 1, dew := 3, total := 18 }
  coke_volume d = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jakes_drink_coke_volume_l602_60282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_solutions_for_f_eq_two_l602_60286

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 4*x + 6
  else |Real.log x / Real.log 8|

theorem three_solutions_for_f_eq_two :
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, f x = 2 ∧
  ∀ y ∉ s, f y ≠ 2 := by
  sorry

#check three_solutions_for_f_eq_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_solutions_for_f_eq_two_l602_60286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_connection_length_final_result_l602_60272

-- Define the points
def red_points : List (ℝ × ℝ) := [(0,0), (1,2), (2,1), (2,2)]
def blue_points : List (ℝ × ℝ) := [(1,0), (2,0), (0,1), (0,2)]

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define a type for valid connections
def ValidConnection := List (Fin 4 × Fin 4)

-- Define a function to calculate the total length of a valid connection
noncomputable def total_length (conn : ValidConnection) : ℝ :=
  conn.map (λ (i, j) => distance (red_points.get! i) (blue_points.get! j)) |>.sum

-- State the theorem
theorem min_connection_length :
  ∃ (conn : ValidConnection), 
    (∀ (other : ValidConnection), total_length conn ≤ total_length other) ∧
    total_length conn = 3 + Real.sqrt 5 := by
  sorry

-- Compute the final result
theorem final_result : 
  ∃ (a b : ℕ), a + Real.sqrt b = 3 + Real.sqrt 5 ∧ 100 * a + b = 305 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_connection_length_final_result_l602_60272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_vertex_coordinates_l602_60221

/-- Given an isosceles triangle ABC with vertices A(12, 10), B(3, -2), and C(x, y),
    where the perpendicular bisector of AB intersects BC at point D(0, 4),
    prove that the coordinates of C are (-3, 10). -/
theorem isosceles_triangle_vertex_coordinates :
  ∀ (x y : ℝ),
  let A : ℝ × ℝ := (12, 10)
  let B : ℝ × ℝ := (3, -2)
  let C : ℝ × ℝ := (x, y)
  let D : ℝ × ℝ := (0, 4)
  (‖A - C‖ = ‖B - C‖) →  -- AB = AC (isosceles triangle)
  (D.1 = (B.1 + C.1) / 2 ∧ D.2 = (B.2 + C.2) / 2) →  -- D is midpoint of BC
  (x = -3 ∧ y = 10) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_vertex_coordinates_l602_60221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_special_fraction_l602_60277

/-- The infinite sum of (3n-2)/(n(n+1)(n+3)) from n=1 to infinity equals 11/24 -/
theorem sum_special_fraction : 
  ∑' (n : ℕ), (3 * (n + 1) - 2) / ((n + 1) * (n + 2) * (n + 4) : ℝ) = 11 / 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_special_fraction_l602_60277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_correct_l602_60273

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2*a*x - 2

noncomputable def min_value (a : ℝ) : ℝ :=
  if a < 0 then -2
  else if a ≤ 2 then -a^2 - 2
  else 2 - 4*a

theorem min_value_correct (a : ℝ) :
  ∀ x ∈ Set.Icc 0 2, f x a ≥ min_value a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_correct_l602_60273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_f_min_value_achieved_f_min_value_is_minimum_l602_60253

/-- The function f(x) = (9x^2 + 27x + 36) / (8(1 + x)) -/
noncomputable def f (x : ℝ) : ℝ := (9 * x^2 + 27 * x + 36) / (8 * (1 + x))

/-- The minimum value of f(x) for x ≥ 0 -/
noncomputable def min_value : ℝ := (9 * Real.sqrt 3) / 4

/-- Theorem: For all x ≥ 0, f(x) ≥ 9√3/4 -/
theorem f_min_value (x : ℝ) (hx : x ≥ 0) : f x ≥ min_value := by
  sorry

/-- Theorem: There exists an x ≥ 0 such that f(x) = 9√3/4 -/
theorem f_min_value_achieved : ∃ x : ℝ, x ≥ 0 ∧ f x = min_value := by
  use Real.sqrt 3 - 1
  constructor
  · -- Prove x ≥ 0
    sorry
  · -- Prove f(x) = min_value
    sorry

/-- Corollary: The minimum value of f(x) for x ≥ 0 is 9√3/4 -/
theorem f_min_value_is_minimum : 
  ∀ x : ℝ, x ≥ 0 → f x ≥ min_value ∧ ∃ y : ℝ, y ≥ 0 ∧ f y = min_value := by
  intro x hx
  constructor
  · exact f_min_value x hx
  · exact f_min_value_achieved

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_f_min_value_achieved_f_min_value_is_minimum_l602_60253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l602_60264

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Point symmetrical to (1, 2, -3) with respect to the y-axis -/
def point_A : Point3D :=
  { x := -1, y := 2, z := 3 }

/-- The other given point -/
def point_B : Point3D :=
  { x := -1, y := -2, z := -1 }

/-- Theorem stating that the distance between point_A and point_B is 4√2 -/
theorem distance_between_points : distance point_A point_B = 4 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l602_60264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_george_max_pencils_l602_60235

/-- The maximum number of pencils George can buy -/
def max_pencils (budget : ℚ) (pencil_cost : ℚ) (discount_threshold : ℕ) (discount_rate : ℚ) : ℕ :=
  max
    (Int.floor (budget / pencil_cost)).toNat
    (Int.floor (budget / (pencil_cost * (1 - discount_rate)))).toNat

theorem george_max_pencils :
  max_pencils 9.30 1.05 8 (1/10) = 9 := by
  sorry

#eval max_pencils 9.30 1.05 8 (1/10)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_george_max_pencils_l602_60235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l602_60293

noncomputable def solution_set : Set ℝ := {3*Real.pi/4, Real.pi, 5*Real.pi/4}

theorem equation_solution (x : ℝ) :
  (Real.sqrt (2 + Real.cos (2*x) - Real.sqrt 3 * Real.tan x) = Real.sin x - Real.sqrt 3 * Real.cos x) ∧
  (|x - 3| < 1) ↔ x ∈ solution_set := by
  sorry

#check equation_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l602_60293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_approximation_of_f_l602_60296

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 + x + 3)

def x₀ : ℝ := 2
def x : ℝ := 1.97

theorem linear_approximation_of_f :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |f x - (f x₀ + ((deriv f) x₀) * (x - x₀))| < ε ∧
  |f x₀ + ((deriv f) x₀) * (x - x₀) - 2.975| < ε := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_approximation_of_f_l602_60296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_slant_angle_l602_60256

noncomputable def curve (x : ℝ) : ℝ := (3 - Real.exp x) / (Real.exp x + 1)

noncomputable def tangent_slope (x : ℝ) : ℝ := -4 / (Real.exp x + Real.exp (-x) + 2)

noncomputable def slant_angle (x : ℝ) : ℝ := Real.arctan (tangent_slope x)

theorem min_slant_angle :
  (∀ x : ℝ, slant_angle x ≥ 3 * Real.pi / 4) ∧
  (∃ x₀ : ℝ, slant_angle x₀ = 3 * Real.pi / 4) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_slant_angle_l602_60256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_properties_l602_60266

noncomputable section

-- Define the points and circle
def P : ℝ × ℝ := (4, 0)
def M : ℝ × ℝ := (0, -2)

-- Define the line that the center of circle C lies on
def center_line (x y : ℝ) : Prop := x - y - 4 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 4

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 4)

-- Define the chord length
noncomputable def chord_length : ℝ := 2 * Real.sqrt 2

-- Main theorem
theorem circle_and_line_properties :
  ∃ (C : ℝ × ℝ),
    center_line C.1 C.2 ∧
    (∀ x y, circle_C x y ↔ ((x - C.1)^2 + (y - C.2)^2 = 4)) ∧
    circle_C M.1 M.2 ∧
    (∃ k₁ k₂ : ℝ, 
      k₁ = 2 + Real.sqrt 3 ∧
      k₂ = 2 - Real.sqrt 3 ∧
      (∀ x y, line_l k₁ x y → (circle_C x y → 
        ∃ x₁ y₁ x₂ y₂, line_l k₁ x₁ y₁ ∧ line_l k₁ x₂ y₂ ∧ circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
        (x₁ - x₂)^2 + (y₁ - y₂)^2 = chord_length^2)) ∧
      (∀ x y, line_l k₂ x y → (circle_C x y → 
        ∃ x₁ y₁ x₂ y₂, line_l k₂ x₁ y₁ ∧ line_l k₂ x₂ y₂ ∧ circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
        (x₁ - x₂)^2 + (y₁ - y₂)^2 = chord_length^2))) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_properties_l602_60266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_football_player_goals_l602_60230

theorem football_player_goals (goals_before : ℝ) : 
  (goals_before * 4 + 2) / 5 = goals_before + 0.1 →
  ∃ (total_goals : ℕ), total_goals = 8 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_football_player_goals_l602_60230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uncle_fyodor_is_11_l602_60242

-- Define Uncle Fyodor's age
def uncle_fyodor_age : ℕ := sorry

-- Define the claims
def claim1 : Prop := uncle_fyodor_age > 11
def claim2 : Prop := uncle_fyodor_age > 10

-- Theorem statement
theorem uncle_fyodor_is_11 : 
  (claim1 ∧ ¬claim2) ∨ (¬claim1 ∧ claim2) → uncle_fyodor_age = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_uncle_fyodor_is_11_l602_60242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_to_vertices_sum_l602_60263

/-- Represents an equilateral triangle with side length 1 -/
structure EquilateralTriangle where
  side_length : ℝ
  is_equilateral : side_length = 1

/-- The length of a median in an equilateral triangle -/
noncomputable def median_length (t : EquilateralTriangle) : ℝ := (Real.sqrt 3) / 2 * t.side_length

/-- The length of a segment from the centroid to a vertex -/
noncomputable def centroid_to_vertex_length (t : EquilateralTriangle) : ℝ := 2 / 3 * median_length t

/-- Theorem: The sum of the lengths of the segments from the centroid to each vertex in an equilateral triangle with side length 1 is √3 -/
theorem centroid_to_vertices_sum (t : EquilateralTriangle) : 
  3 * centroid_to_vertex_length t = Real.sqrt 3 := by
  sorry

#check centroid_to_vertices_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_to_vertices_sum_l602_60263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_divisible_number_l602_60249

/-- The least common multiple of 4, 5, and 6 -/
def lcm456 : ℕ := 60

/-- The first number after 190 that is divisible by 4, 5, and 6 -/
def first_divisible : ℕ := 240

/-- The count of numbers we're looking for -/
def count : ℕ := 6

/-- The starting number -/
def start : ℕ := 190

/-- The ending number we want to prove -/
def end_number : ℕ := 540

theorem sixth_divisible_number : 
  (first_divisible + (count - 1) * lcm456 = end_number) ∧ 
  (∀ n : ℕ, start < n ∧ n ≤ end_number ∧ n % 4 = 0 ∧ n % 5 = 0 ∧ n % 6 = 0 → 
    n ∈ (Finset.range count.succ).image (λ i => first_divisible + i * lcm456)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_divisible_number_l602_60249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_range_l602_60252

-- Define the line l
def line_l (x y : ℝ) : Prop := y = Real.sqrt 3 * x + 4

-- Define the circle O
def circle_O (x y r : ℝ) : Prop := x^2 + y^2 = r^2

-- Define the rhombus ABCD
structure Rhombus (r : ℝ) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  interior_angle : ℝ
  on_line_l : line_l A.1 A.2 ∧ line_l B.1 B.2
  on_circle_O : circle_O C.1 C.2 r ∧ circle_O D.1 D.2 r

-- Define the area of the rhombus
noncomputable def rhombus_area (r : ℝ) (ABCD : Rhombus r) : ℝ := sorry

-- Theorem statement
theorem rhombus_area_range (r : ℝ) (hr : 1 < r ∧ r < 2) (ABCD : Rhombus r) :
  ∃ (S : Set ℝ), S = {x | 0 < x ∧ x < 3 * Real.sqrt 3 / 2 ∨ 3 * Real.sqrt 3 / 2 < x ∧ x < 6 * Real.sqrt 3} ∧
  rhombus_area r ABCD ∈ S := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_range_l602_60252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l602_60205

open Real

-- Define the function f
noncomputable def f (x : ℝ) := log (2 * x + 3) + x^2

-- Define the interval
def a : ℝ := -1
noncomputable def b : ℝ := (exp 2 - 3) / 2

-- State the theorem
theorem f_extrema :
  (∀ x ∈ Set.Icc a b, f (-1/2) ≤ f x) ∧
  (f (-1/2) = log 2 + 1/4) ∧
  (∀ x ∈ Set.Icc a b, f x ≤ f b) ∧
  (f b = 2 + (exp 2 - 3)^2 / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l602_60205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_decreasing_implies_a_range_l602_60208

def is_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem log_function_decreasing_implies_a_range
  (a : ℝ)
  (h : is_decreasing (fun x => Real.log (2 - a * x) / Real.log a) 0 1) :
  1 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_decreasing_implies_a_range_l602_60208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_min_at_neg_point_three_l602_60259

noncomputable def h (z : ℝ) : ℝ := Real.sqrt (1.44 + 0.8 * (z + 0.3)^2)

theorem h_min_at_neg_point_three :
  ∀ z : ℝ, h (-0.3) ≤ h z :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_min_at_neg_point_three_l602_60259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_2_8_plus_4_7_l602_60228

theorem greatest_prime_factor_of_2_8_plus_4_7 :
  (Nat.factors (2^8 + 4^7)).argmax id = some 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_2_8_plus_4_7_l602_60228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_k_equals_one_l602_60237

/-- Triangle ABC with vertices A=(0,2), B=(0,0), and C=(10,0) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The given triangle ABC -/
def triangleABC : Triangle where
  A := (0, 2)
  B := (0, 0)
  C := (10, 0)

/-- The area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ :=
  (1 / 2) * base * height

/-- The value of k that divides the triangle into two equal areas -/
def k : ℝ := 1

/-- Theorem stating that k divides the triangle into two equal areas -/
theorem equal_area_division (t : Triangle) (h : t = triangleABC) :
  let totalArea := triangleArea 10 2
  let upperArea := triangleArea 10 (2 - k)
  upperArea = totalArea / 2 := by
  sorry

/-- Main theorem proving k = 1 -/
theorem k_equals_one : k = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_k_equals_one_l602_60237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_optimal_division_l602_60268

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the centroid of a triangle
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  let (x₁, y₁) := t.A
  let (x₂, y₂) := t.B
  let (x₃, y₃) := t.C
  ((x₁ + x₂ + x₃) / 3, (y₁ + y₂ + y₃) / 3)

-- Define a line passing through a point
structure Line where
  point : ℝ × ℝ
  slope : ℝ

-- Define the area ratio of two parts of a triangle divided by a line
noncomputable def areaRatio (t : Triangle) (l : Line) : ℝ := sorry

-- Theorem statement
theorem centroid_optimal_division (t : Triangle) :
  let m := centroid t
  ∀ l : Line, l.point = m →
    (4 / 5 : ℝ) ≤ areaRatio t l ∧ areaRatio t l ≤ (5 / 4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_optimal_division_l602_60268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_triangle_areas_l602_60206

noncomputable def f (m n : ℕ) : ℝ := sorry

theorem chessboard_triangle_areas (m n : ℕ) :
  (m % 2 = 0 ∧ n % 2 = 0 → f m n = 0) ∧
  (m % 2 = 1 ∧ n % 2 = 1 → f m n = 1/2) ∧
  (f m n ≤ (1/2) * max (m : ℝ) (n : ℝ)) ∧
  (∀ c : ℝ, ∃ m' n' : ℕ, f m' n' ≥ c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_triangle_areas_l602_60206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_properties_l602_60243

-- Define the roots x₁ and x₂ as noncomputable real numbers
noncomputable def x₁ : ℝ := sorry
noncomputable def x₂ : ℝ := sorry

-- Define the equation that x₁ and x₂ satisfy
axiom eq_roots : x₁^2 - 6*x₁ + 1 = 0 ∧ x₂^2 - 6*x₂ + 1 = 0

-- Define the sequence aₙ as noncomputable
noncomputable def a (n : ℕ) : ℝ := x₁^n + x₂^n

-- Theorem statement
theorem a_properties : ∀ n : ℕ, (∃ m : ℤ, a n = m) ∧ (∃ k : ℤ, a n = 5 * k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_properties_l602_60243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l602_60227

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then -x + 1
  else if x < 0 then -x - 1
  else 0

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x > 0, f x = -x + 1) ∧
  f (-2) = 1 ∧
  ∀ x, f x = if x < 0 then -x - 1 else if x > 0 then -x + 1 else 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l602_60227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blake_initial_amount_l602_60297

def initial_amount : ℝ → Prop := sorry
def land_value : ℝ → ℝ := sorry
def half_sale_amount : ℝ := sorry

axiom tripled_value (x : ℝ) : land_value x = 3 * x
axiom half_sale_is_30000 : half_sale_amount = 30000

theorem blake_initial_amount :
  ∀ x : ℝ, initial_amount x →
    land_value x / 2 = half_sale_amount →
    x = 20000 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blake_initial_amount_l602_60297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_pipe_empty_time_l602_60220

/-- Represents a tank with two pipes -/
structure TankWithPipes where
  fillTime1 : ℝ  -- Time for first pipe to fill the tank
  fillTimeBoth : ℝ  -- Time to fill the tank with both pipes open

/-- Calculates the time it takes for the second pipe to empty the tank -/
noncomputable def emptyTimeSecondPipe (tank : TankWithPipes) : ℝ :=
  (tank.fillTime1 * tank.fillTimeBoth) / (tank.fillTimeBoth - tank.fillTime1)

/-- Theorem: Given the conditions, the second pipe empties the tank in 90 minutes -/
theorem second_pipe_empty_time 
  (tank : TankWithPipes) 
  (h1 : tank.fillTime1 = 60) 
  (h2 : tank.fillTimeBoth = 180) : 
  emptyTimeSecondPipe tank = 90 := by
  sorry

#eval "Compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_pipe_empty_time_l602_60220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walkway_time_with_direction_l602_60238

/-- Represents a moving walkway with a person walking on it -/
structure MovingWalkway where
  length : ℝ  -- Length of the walkway in meters
  time_against : ℝ  -- Time to walk against the walkway in seconds
  time_stationary : ℝ  -- Time to walk when the walkway is not moving in seconds

/-- Calculates the time to walk with the direction of the walkway -/
noncomputable def time_with_direction (w : MovingWalkway) : ℝ :=
  w.length / (w.length / w.time_stationary + w.length / w.time_against)

/-- Theorem stating that for the given conditions, the time to walk with the direction is 30 seconds -/
theorem walkway_time_with_direction :
  let w : MovingWalkway := ⟨60, 120, 48⟩
  time_with_direction w = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walkway_time_with_direction_l602_60238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_andrew_total_spent_l602_60201

def game_cost (n : ℕ) : ℚ := 9 + (3.5 * (n - 1))

def total_game_cost (num_games : ℕ) : ℚ :=
  (List.range num_games).map (fun i => game_cost (i + 1)) |>.sum

def discounted_game_cost (total_cost : ℚ) (discount_rate : ℚ) : ℚ :=
  total_cost * (1 - discount_rate)

def discounted_snack_cost (snack_cost : ℚ) (discount_rate : ℚ) : ℚ :=
  snack_cost * (1 - discount_rate)

def taxed_drink_cost (drink_cost : ℚ) (tax_rate : ℚ) : ℚ :=
  drink_cost * (1 + tax_rate)

theorem andrew_total_spent :
  let num_games : ℕ := 7
  let game_discount_rate : ℚ := 0.1
  let snack_cost : ℚ := 25
  let snack_discount_rate : ℚ := 0.05
  let drink_cost : ℚ := 20
  let drink_tax_rate : ℚ := 0.07
  let total_spent := 
    discounted_game_cost (total_game_cost num_games) game_discount_rate +
    discounted_snack_cost snack_cost snack_discount_rate +
    taxed_drink_cost drink_cost drink_tax_rate
  total_spent = 168 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_andrew_total_spent_l602_60201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_power_six_decomposition_l602_60229

open Real

theorem cosine_power_six_decomposition :
  ∃ (b₁ b₂ b₃ b₄ b₅ b₆ : ℝ),
    (∀ θ : ℝ, (cos θ)^6 = b₁ * cos θ + b₂ * cos (2*θ) + b₃ * cos (3*θ) + b₄ * cos (4*θ) + b₅ * cos (5*θ) + b₆ * cos (6*θ)) ∧
    b₁^2 + b₂^2 + b₃^2 + b₄^2 + b₅^2 + b₆^2 = 131/512 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_power_six_decomposition_l602_60229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operations_and_properties_l602_60267

-- Define the vectors
def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (4, 1)

-- Define vector operations
def add_vec (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def scale_vec (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- Define vector properties
def parallel (v w : ℝ × ℝ) : Prop := ∃ (k : ℝ), v = scale_vec k w
def perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

theorem vector_operations_and_properties :
  ∀ (l : ℝ),
  -- Part 1
  (add_vec a (scale_vec l c) = (3 + 4*l, 2 + l)) ∧
  (add_vec (scale_vec 2 b) (scale_vec (-1) a) = (-5, 2)) ∧
  -- Part 2
  (parallel (add_vec a (scale_vec l c)) (add_vec (scale_vec 2 b) (scale_vec (-1) a)) →
    l = -16/13) ∧
  -- Part 3
  (perpendicular (add_vec a (scale_vec l c)) (add_vec (scale_vec 2 b) (scale_vec (-1) a)) →
    l = 11/18) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operations_and_properties_l602_60267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sign_changing_sum_equals_3116_l602_60262

def sign_changing_sum : ℤ := 
  ((-1) + 2 + 3 + 4) + 
  ((-5) + (-6) + (-7) + (-8) + (-9)) + 
  (List.sum (List.range 16 |>.map (fun i => (i + 10 : ℤ)))) + 
  (List.sum (List.range 24 |>.map (fun i => (-(i + 26) : ℤ)))) + 
  (List.sum (List.range 51 |>.map (fun i => (i + 50 : ℤ))))

theorem sign_changing_sum_equals_3116 : sign_changing_sum = 3116 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sign_changing_sum_equals_3116_l602_60262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_fraction_l602_60218

theorem decimal_to_fraction : 
  ∃ (n d : ℤ), d ≠ 0 ∧ 3.675 = (n : ℚ) / (d : ℚ) ∧ Int.gcd n d = 1 ∧ n = 147 ∧ d = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_fraction_l602_60218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_f_l602_60295

noncomputable def f (x : ℝ) := Real.sqrt 3 * Real.sin (2 * x) + Real.cos (2 * x)

theorem symmetry_axis_of_f :
  ∀ x : ℝ, f (2 * π / 3 + x) = f (2 * π / 3 - x) :=
by
  intro x
  simp [f]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_f_l602_60295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_appointment_schemes_l602_60248

def male_teachers : ℕ := 5
def female_teachers : ℕ := 4
def total_advisers : ℕ := 3
def classes : ℕ := 3

theorem appointment_schemes :
  (Nat.choose male_teachers 2 * Nat.choose female_teachers 1 + 
   Nat.choose male_teachers 1 * Nat.choose female_teachers 2) * 
  Nat.factorial classes = 420 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_appointment_schemes_l602_60248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nadal_championship_probability_l602_60211

/-- The probability of winning a single match for Nadal -/
def p : ℚ := 2/3

/-- The number of matches Nadal needs to win to claim the championship -/
def matches_to_win : ℕ := 4

/-- Calculates the probability of Nadal winning the championship -/
def championship_probability : ℚ :=
  (Finset.range matches_to_win).sum (λ k =>
    (Nat.choose (3 + k) k : ℚ) * p^matches_to_win * (1 - p)^k)

theorem nadal_championship_probability :
  championship_probability = 7168/6561 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nadal_championship_probability_l602_60211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_simplification_l602_60225

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define points
variable (A B C D M O : V)

-- Define vectors
def AB (A B : V) : V := B - A
def BC (B C : V) : V := C - B
def CD (C D : V) : V := D - C
def AD (A D : V) : V := D - A
def MB (M B : V) : V := B - M
def CM (C M : V) : V := M - C
def OA (O A : V) : V := A - O
def OC (O C : V) : V := C - O

-- Theorem statement
theorem vector_simplification (A B C D M O : V) :
  (AB A B + CD C D) + BC B C = AD A D ∧
  (AD A D + MB M B) + (BC B C + CM C M) = AD A D ∧
  MB M B + AD A D + (B - M) = AD A D ∧
  OC O C - OA O A - CD C D ≠ AD A D :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_simplification_l602_60225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_90_degrees_equals_zero_l602_60246

theorem cos_90_degrees_equals_zero : Real.cos (π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_90_degrees_equals_zero_l602_60246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_value_l602_60292

theorem trigonometric_expression_value (θ : Real) : 
  (π < θ ∧ θ < 2*π) →  -- θ is in the second quadrant
  (Real.cos (θ/2) < 0) → 
  (Real.sqrt (1 - Real.sin θ)) / (Real.sin (θ/2) - Real.cos (θ/2)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_value_l602_60292
