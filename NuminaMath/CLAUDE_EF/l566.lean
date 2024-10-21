import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l566_56652

noncomputable def f (x : ℝ) := x + 4 / x

theorem min_value_of_f :
  ∀ x : ℝ, x > 0 → f x ≥ 4 ∧ (f x = 4 ↔ x = 2) :=
by
  intro x hx
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l566_56652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_five_equals_eight_l566_56611

theorem g_of_five_equals_eight (g : ℝ → ℝ) (h : ∀ x, g (3*x - 4) = 5*x - 7) : g 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_five_equals_eight_l566_56611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_distribution_theorem_l566_56655

/-- The number of ways to distribute students among universities --/
def distributeStudents (n : ℕ) (k : ℕ) : ℕ :=
  k^n - k * (k-1)^n + if n ≥ k then 0 else Nat.choose k (k-n) * Nat.factorial n

/-- The main theorem --/
theorem student_distribution_theorem :
  distributeStudents 5 3 = 144 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_distribution_theorem_l566_56655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_escalator_faster_l566_56636

/-- Represents the speeds and length involved in the escalator problem -/
structure EscalatorProblem where
  v : ℝ  -- speed of the escalator
  v₁ : ℝ  -- speed of the person going up
  v₂ : ℝ  -- speed of the person going down
  l : ℝ  -- length of the escalator
  h₁ : 0 < v
  h₂ : v < v₁
  h₃ : v₁ < v₂
  h₄ : 0 < l

/-- The total time for ascending and descending on an ascending escalator -/
noncomputable def time_ascending_escalator (p : EscalatorProblem) : ℝ :=
  p.l / (p.v₁ + p.v) + p.l / (p.v₂ - p.v)

/-- The total time for ascending and descending on a descending escalator -/
noncomputable def time_descending_escalator (p : EscalatorProblem) : ℝ :=
  p.l / (p.v₁ - p.v) + p.l / (p.v₂ + p.v)

/-- Theorem stating that ascending and descending on an ascending escalator is faster -/
theorem ascending_escalator_faster (p : EscalatorProblem) :
  time_ascending_escalator p < time_descending_escalator p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_escalator_faster_l566_56636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_intersecting_lines_l566_56610

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Two lines are skew if they are not parallel and do not intersect -/
def are_skew (l1 l2 : Line3D) : Prop := sorry

/-- A line intersects another line -/
def intersects (l1 l2 : Line3D) : Prop := sorry

theorem infinite_intersecting_lines 
  (a b c : Line3D) 
  (h1 : are_skew a b) 
  (h2 : are_skew b c) 
  (h3 : are_skew a c) : 
  ∃ (S : Set Line3D), (∀ l ∈ S, intersects l a ∧ intersects l b ∧ intersects l c) ∧ Infinite S := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_intersecting_lines_l566_56610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l566_56693

theorem inequality_proof (n : ℕ+) (x : ℝ) (h : x > 0) :
  x + (n : ℝ)^(n : ℕ) / x^(n : ℕ) ≥ n + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l566_56693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l566_56663

open Real

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : Differentiable ℝ f)
variable (h2 : ∀ x > 1, f x + deriv f x < x * deriv f x)

-- Define a, b, and c
noncomputable def a : ℝ := f 2
noncomputable def b : ℝ := (1/2) * f 3
noncomputable def c : ℝ := (Real.sqrt 2 + 1) * f (Real.sqrt 2)

-- State the theorem
theorem relationship_abc : c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l566_56663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_passes_donovan_after_six_laps_l566_56628

/-- The length of the circular track in meters -/
noncomputable def track_length : ℝ := 400

/-- Donovan's speed in meters per second -/
noncomputable def donovan_speed : ℝ := track_length / 48

/-- Michael's speed in meters per second -/
noncomputable def michael_speed : ℝ := track_length / 40

/-- The number of laps Michael completes when he first passes Donovan -/
def laps_to_pass : ℕ := 6

theorem michael_passes_donovan_after_six_laps :
  ∃ (t : ℝ), t > 0 ∧ t * michael_speed = track_length * laps_to_pass ∧
             t * michael_speed = t * donovan_speed + track_length := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_passes_donovan_after_six_laps_l566_56628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_not_in_range_of_g_l566_56635

/-- The function g defined by (ax + b) / (cx + d) -/
noncomputable def g (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem unique_not_in_range_of_g (a b c d : ℝ) 
  (ha : a ≠ 0) (hc : c ≠ 0)
  (h23 : g a b c d 23 = 23)
  (h101 : g a b c d 101 = 101)
  (hg : ∀ x, x ≠ -d/c → g a b c d (g a b c d x) = x) :
  ∃! y, (∀ x, g a b c d x ≠ y) ∧ y = 62 := by
  sorry

#check unique_not_in_range_of_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_not_in_range_of_g_l566_56635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_rate_equation_l566_56688

theorem production_rate_equation (x : ℝ) (h : x > 0) : 
  (800 : ℝ) / (x + 30) = 600 / x ↔ 
  ∃ (t : ℝ), t > 0 ∧ 
    800 = (x + 30) * t ∧ 
    600 = x * t :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_rate_equation_l566_56688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_inverse_points_l566_56667

noncomputable def f (a b x : ℝ) : ℝ := 2^(a * x + b)

theorem function_and_inverse_points (a b : ℝ) :
  f a b 2 = 1/2 ∧ f a b (1/2) = 2 → a + b = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_inverse_points_l566_56667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_sum_identity_l566_56678

def f (n : ℕ) : ℚ := (Finset.range n).sum (fun i => 1 / (i + 1 : ℚ))

theorem harmonic_sum_identity (n : ℕ) (h : 2 ≤ n) :
  (Finset.range (n - 1)).sum f = n * (f n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_sum_identity_l566_56678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ac_range_in_obtuse_triangle_l566_56602

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the lengths of sides
noncomputable def side_length (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define an obtuse angle
def is_obtuse_angle (A B C : ℝ × ℝ) : Prop :=
  (side_length A C)^2 > (side_length A B)^2 + (side_length B C)^2

-- Theorem statement
theorem ac_range_in_obtuse_triangle (t : Triangle) 
  (h_obtuse : is_obtuse_angle t.A t.B t.C)
  (h_ab : side_length t.A t.B = 6)
  (h_bc : side_length t.B t.C = 8) :
  10 < side_length t.A t.C ∧ side_length t.A t.C < 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ac_range_in_obtuse_triangle_l566_56602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_equation_solution_range_l566_56624

theorem cosine_equation_solution_range (a : ℝ) : 
  (∃ x : ℝ, (Real.cos x)^2 - 2 * (Real.cos x) - a = 0) ↔ -1 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_equation_solution_range_l566_56624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_factorial_divisible_by_4125_l566_56613

theorem least_factorial_divisible_by_4125 :
  (∀ k < 15, ¬(4125 ∣ Nat.factorial k)) ∧ (4125 ∣ Nat.factorial 15) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_factorial_divisible_by_4125_l566_56613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ones_in_grid_l566_56687

def Grid := Fin 40 → Fin 7 → Bool

def is_valid (g : Grid) : Prop :=
  ∀ i j : Fin 40, i ≠ j → ∃ k : Fin 7, g i k ≠ g j k

def count_ones (g : Grid) : ℕ :=
  (Finset.univ : Finset (Fin 40)).sum (λ i => (Finset.univ : Finset (Fin 7)).sum (λ j => if g i j then 1 else 0))

theorem max_ones_in_grid :
  ∃ g : Grid, is_valid g ∧ count_ones g = 198 ∧
  ∀ h : Grid, is_valid h → count_ones h ≤ 198 := by
  sorry

#check max_ones_in_grid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ones_in_grid_l566_56687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_product_l566_56639

/-- Circle with center (3, -4) and radius 2 -/
def myCircle (x y : ℝ) : Prop := (x - 3)^2 + (y + 4)^2 = 4

/-- Line with slope k passing through the origin -/
def myLine (k x y : ℝ) : Prop := y = k * x

/-- P and Q are the intersection points of the circle and the line -/
def intersectionPoints (k : ℝ) (P Q : ℝ × ℝ) : Prop :=
  myCircle P.1 P.2 ∧ myCircle Q.1 Q.2 ∧ myLine k P.1 P.2 ∧ myLine k Q.1 Q.2 ∧ P ≠ Q

/-- The origin of the coordinate system -/
def O : ℝ × ℝ := (0, 0)

/-- The distance between two points -/
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem circle_line_intersection_product (k : ℝ) (P Q : ℝ × ℝ) :
  intersectionPoints k P Q → distance O P * distance O Q = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_product_l566_56639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_apple_worth_l566_56640

-- Define the types for fruits
def Banana : Type := ℕ
def Orange : Type := ℕ
def Apple : Type := ℕ

-- Define the worth function
def worth : ℕ → ℕ := id

-- Define the conditions
axiom banana_orange_ratio : worth ((3 * 16) / 4) = worth 12
axiom banana_apple_ratio : ∀ (b : ℕ), worth b = worth (2 * b)

-- Theorem to prove
theorem banana_apple_worth : 
  worth ((1 * 9) / 3) = worth 6 :=
by
  -- The proof steps would go here
  sorry

#eval worth ((1 * 9) / 3)
#eval worth 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_apple_worth_l566_56640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_parallel_lines_l566_56608

/-- The distance between two parallel lines -/
noncomputable def distance_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₁ - c₂| / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance between the parallel lines 2x + 3y - 3 = 0 and 2x + 3y + 2 = 0 is 5√13 / 13 -/
theorem distance_specific_parallel_lines :
  distance_parallel_lines 2 3 (-3) 2 = 5 * Real.sqrt 13 / 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_parallel_lines_l566_56608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_theorem_l566_56658

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {x : ℕ | x < 3}

theorem intersection_complement_theorem : A ∩ (Set.univ \ B) = {3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_theorem_l566_56658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_relation_max_binomial_term_sum_of_coefficients_l566_56620

-- Define n as a natural number
def n : ℕ := 6

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Statement 1: Coefficient relation
theorem coefficient_relation :
  2 * binomial n 1 = (1 / 5 : ℚ) * (2^2 : ℚ) * (binomial n 2 : ℚ) := by sorry

-- Statement 2: Maximum binomial coefficient term
theorem max_binomial_term :
  binomial n 3 * 2^3 = 160 := by sorry

-- Statement 3: Sum of coefficients
theorem sum_of_coefficients :
  (Finset.range (n + 1)).sum (λ k => binomial n k) = 64 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_relation_max_binomial_term_sum_of_coefficients_l566_56620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_burrito_calorie_count_l566_56648

/-- The number of calories in each burrito -/
def burrito_calories : ℕ := 120

/-- The number of burritos that can be bought for $6 -/
def burritos_per_6_dollars : ℕ := 10

/-- The number of burgers that can be bought for $8 -/
def burgers_per_8_dollars : ℕ := 5

/-- The number of calories in each burger -/
def burger_calories : ℕ := 400

/-- The difference in calories per dollar between burgers and burritos -/
def calorie_difference_per_dollar : ℕ := 50

theorem burrito_calorie_count :
  burrito_calories * burritos_per_6_dollars = 
  (burger_calories * burgers_per_8_dollars * 6) / 8 - 
  (calorie_difference_per_dollar * 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_burrito_calorie_count_l566_56648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_from_sixth_and_ninth_l566_56604

/-- A geometric sequence with specified 6th and 9th terms -/
structure GeometricSequence where
  a : ℝ  -- first term
  r : ℝ  -- common ratio

/-- The 6th term of a geometric sequence -/
def sixth_term (seq : GeometricSequence) : ℝ := seq.a * seq.r^5

/-- The 9th term of a geometric sequence -/
def ninth_term (seq : GeometricSequence) : ℝ := seq.a * seq.r^8

/-- Theorem stating the relationship between the first term and the 6th and 9th terms -/
theorem first_term_from_sixth_and_ninth :
  ∀ (seq : GeometricSequence),
  sixth_term seq = Nat.factorial 7 →
  ninth_term seq = Nat.factorial 9 →
  seq.a = (Nat.factorial 7 : ℝ) / ((Nat.factorial 9 / Nat.factorial 7 : ℝ) ^ (5/3)) :=
by
  sorry

#check first_term_from_sixth_and_ninth

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_from_sixth_and_ninth_l566_56604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l566_56695

theorem expression_evaluation (x : ℝ) (h1 : 0 < x) (h2 : x ≤ 2) 
  (h3 : x ≠ 2) : 
  (x + 2 + 4 / (x - 2)) / (x^3 / (x^2 - 4*x + 4)) = -1 :=
by
  -- Simplify the expression
  have h4 : (x + 2 + 4 / (x - 2)) / (x^3 / (x^2 - 4*x + 4)) = (x - 2) / x := by
    -- This step would involve algebraic manipulation
    sorry
  
  -- Show that x must be 1
  have h5 : x = 1 := by
    -- This step would involve using the constraints on x
    sorry
  
  -- Substitute x = 1 into the simplified expression
  rw [h5] at h4
  -- Evaluate (1 - 2) / 1
  -- This step would involve arithmetic
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l566_56695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g2_pow_5_l566_56654

-- Define f and g as noncomputable functions
noncomputable section
  def f : ℝ → ℝ := sorry
  def g : ℝ → ℝ := sorry
end

-- State the axioms
axiom fg_condition : ∀ x : ℝ, x ≥ 1 → f (g x) = x^4
axiom gf_condition : ∀ x : ℝ, x ≥ 1 → g (f x) = x^5
axiom g32_condition : g 32 = 32

-- State and prove the theorem
theorem g2_pow_5 : (g 2)^5 = 32^5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g2_pow_5_l566_56654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_transformed_functions_l566_56665

/-- Given a function f, prove that the graphs of y = f(x-19) and y = f(99-x) 
    are symmetric with respect to the line x = 59 -/
theorem symmetry_of_transformed_functions (f : ℝ → ℝ) :
  ∀ x : ℝ, f (x - 19) = f (119 - x) := by
  intro x
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_transformed_functions_l566_56665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_equation_l566_56674

/-- The equation of a circle with its center at the focus of the parabola y^2 = -8x
    and tangent to the directrix of this parabola -/
theorem parabola_circle_equation :
  let parabola := fun (x y : ℝ) ↦ y^2 = -8*x
  let focus : ℝ × ℝ := (-2, 0)
  let directrix := fun (x : ℝ) ↦ x = 2
  let circle := fun (x y : ℝ) ↦ (x+2)^2 + y^2 = 16
  (∀ x y, parabola x y → circle x y) ∧
  (∃ x y, directrix x ∧ circle x y) ∧
  (∀ x y, circle x y → (x - focus.1)^2 + (y - focus.2)^2 = 16) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_equation_l566_56674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_distance_equals_radius_l566_56601

-- Define the circle C
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 2*y + 1 = 0

-- Define the line l
def line_equation (x y : ℝ) : Prop :=
  x/4 - y/3 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (-2, 1)

-- Define the radius of the circle
def circle_radius : ℝ := 2

-- Theorem stating that the line is tangent to the circle
theorem line_tangent_to_circle :
  ∀ (x y : ℝ), circle_equation x y → line_equation x y →
  (x - (-2))^2 + (y - 1)^2 = 4 :=
by
  sorry

-- Theorem stating that the distance from the center to the line equals the radius
theorem distance_equals_radius :
  let (cx, cy) := circle_center
  |(3 * cx - 4 * cy)| / Real.sqrt (3^2 + 4^2) = circle_radius :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_distance_equals_radius_l566_56601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_extreme_points_iff_l566_56683

/-- The function f(x) = x(ln x - ax) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (Real.log x - a * x)

/-- f(x) has two extreme points -/
def has_two_extreme_points (f : ℝ → ℝ) : Prop := 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  (∀ x : ℝ, x ≠ x₁ → x ≠ x₂ → (HasDerivAt f 0 x₁ ∧ HasDerivAt f 0 x₂) → 
    (¬HasDerivAt f 0 x ∨ (¬HasDerivAt (deriv f) 0 x₁ ∧ ¬HasDerivAt (deriv f) 0 x₂)))

/-- Theorem: f(x) has two extreme points if and only if 0 < a < 1/2 -/
theorem f_two_extreme_points_iff (a : ℝ) : 
  has_two_extreme_points (f a) ↔ 0 < a ∧ a < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_extreme_points_iff_l566_56683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bank_queue_problem_l566_56662

/-- Represents a person in the queue with their operation time -/
inductive Person
| Simple : Person
| Long : Person
deriving BEq, Repr

/-- Calculates the total wasted person-minutes for a given queue order -/
def calculate_wasted_time (queue : List Person) : ℕ :=
  sorry

/-- Calculates the expected wasted person-minutes for a random queue order -/
def expected_wasted_time (num_simple num_long : ℕ) : ℚ :=
  sorry

theorem bank_queue_problem (queue : List Person) 
  (h1 : queue.length = 8)
  (h2 : queue.count Person.Simple = 5)
  (h3 : queue.count Person.Long = 3) :
  (∃ (min_queue max_queue : List Person),
    calculate_wasted_time min_queue = 40 ∧
    calculate_wasted_time max_queue = 100) ∧
  expected_wasted_time 5 3 = 84 := by
  sorry

#eval Person.Simple
#eval Person.Long

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bank_queue_problem_l566_56662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_third_altitude_is_six_l566_56680

/-- A scalene triangle with specific altitude properties -/
structure ScaleneTriangle where
  -- The lengths of the sides
  a : ℝ
  b : ℝ
  c : ℝ
  -- The lengths of the altitudes
  h₁ : ℝ
  h₂ : ℝ
  h₃ : ℝ
  -- Conditions
  scalene : a ≠ b ∧ b ≠ c ∧ a ≠ c
  altitude_lengths : h₁ = 3 ∧ h₂ = 9
  third_altitude_integer : ∃ n : ℕ, h₃ = n

/-- The maximum possible integer length of the third altitude -/
def max_third_altitude : ℕ := 6

/-- Theorem stating that 6 is the maximum possible integer length for the third altitude -/
theorem max_third_altitude_is_six (t : ScaleneTriangle) : 
  max_third_altitude = 6 ∧ 
  ∃ n : ℕ, t.h₃ = n ∧ n ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_third_altitude_is_six_l566_56680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_empire_state_model_height_l566_56666

/-- The height of the Empire State Building in feet -/
noncomputable def actual_height : ℝ := 1454

/-- The scale ratio of the model -/
noncomputable def scale_ratio : ℝ := 50

/-- The height of the model before rounding -/
noncomputable def model_height : ℝ := actual_height / scale_ratio

/-- Function to round a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem empire_state_model_height :
  round_to_nearest model_height = 29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_empire_state_model_height_l566_56666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_missed_bus_time_l566_56653

def bus_departure_time : ℕ := 480  -- 8:00 a.m. in minutes since midnight
def travel_time : ℕ := 30  -- 30 minutes
def home_departure_time : ℕ := 470  -- 7:50 a.m. in minutes since midnight

theorem missed_bus_time : 
  home_departure_time + travel_time - bus_departure_time = 20 := by
  -- Unfold the definitions
  unfold home_departure_time travel_time bus_departure_time
  -- Perform the arithmetic
  norm_num

#eval home_departure_time + travel_time - bus_departure_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_missed_bus_time_l566_56653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_derivative_g_l566_56622

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + x * Real.cos x
noncomputable def g (x : ℝ) : ℝ := x / (1 + x)

-- State the theorems
theorem derivative_f (x : ℝ) :
  deriv f x = 6 * x + Real.cos x - x * Real.sin x := by sorry

theorem derivative_g (x : ℝ) (h : x ≠ -1) :
  deriv g x = 1 / (1 + x)^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_derivative_g_l566_56622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_touching_circles_radius_correct_l566_56600

/-- Given a triangle with sides a, b, and c, this function calculates the radius of two touching circles 
    with equal radii that each touch two sides of the triangle. -/
noncomputable def two_touching_circles_radius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  (c * Real.sqrt ((s - a) * (s - b) * (s - c))) / (c * Real.sqrt s + 2 * Real.sqrt ((s - a) * (s - b) * (s - c)))

/-- Theorem stating that the radius calculated by two_touching_circles_radius is correct for any triangle. -/
theorem two_touching_circles_radius_correct (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  let r := two_touching_circles_radius a b c
  ∃ (circle1 circle2 : Set (ℝ × ℝ)), 
    (∃ (x1 y1 x2 y2 : ℝ), 
      circle1 = {p : ℝ × ℝ | (p.1 - x1)^2 + (p.2 - y1)^2 = r^2} ∧
      circle2 = {p : ℝ × ℝ | (p.1 - x2)^2 + (p.2 - y2)^2 = r^2} ∧
      -- The circles touch each other
      (x1 - x2)^2 + (y1 - y2)^2 = (2*r)^2 ∧
      -- Each circle touches two sides of the triangle
      (∃ (p1 p2 q1 q2 : ℝ × ℝ), 
        p1 ∈ circle1 ∧ p2 ∈ circle1 ∧ q1 ∈ circle2 ∧ q2 ∈ circle2 ∧
        (p1 ∈ {p : ℝ × ℝ | p.2 = 0 ∧ 0 ≤ p.1 ∧ p.1 ≤ c} ∨ 
         p1 ∈ {p : ℝ × ℝ | p.1*a + p.2*c = a*c ∧ 0 ≤ p.1 ∧ p.1 ≤ c} ∨
         p1 ∈ {p : ℝ × ℝ | p.1 = 0 ∧ 0 ≤ p.2 ∧ p.2 ≤ b}) ∧
        (p2 ∈ {p : ℝ × ℝ | p.2 = 0 ∧ 0 ≤ p.1 ∧ p.1 ≤ c} ∨ 
         p2 ∈ {p : ℝ × ℝ | p.1*a + p.2*c = a*c ∧ 0 ≤ p.1 ∧ p.1 ≤ c} ∨
         p2 ∈ {p : ℝ × ℝ | p.1 = 0 ∧ 0 ≤ p.2 ∧ p.2 ≤ b}) ∧
        p1 ≠ p2 ∧
        (q1 ∈ {p : ℝ × ℝ | p.2 = 0 ∧ 0 ≤ p.1 ∧ p.1 ≤ c} ∨ 
         q1 ∈ {p : ℝ × ℝ | p.1*a + p.2*c = a*c ∧ 0 ≤ p.1 ∧ p.1 ≤ c} ∨
         q1 ∈ {p : ℝ × ℝ | p.1 = 0 ∧ 0 ≤ p.2 ∧ p.2 ≤ b}) ∧
        (q2 ∈ {p : ℝ × ℝ | p.2 = 0 ∧ 0 ≤ p.1 ∧ p.1 ≤ c} ∨ 
         q2 ∈ {p : ℝ × ℝ | p.1*a + p.2*c = a*c ∧ 0 ≤ p.1 ∧ p.1 ≤ c} ∨
         q2 ∈ {p : ℝ × ℝ | p.1 = 0 ∧ 0 ≤ p.2 ∧ p.2 ≤ b}) ∧
        q1 ≠ q2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_touching_circles_radius_correct_l566_56600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_vector_bounded_norm_l566_56634

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define the dimension of the space
variable (h : FiniteDimensional.finrank ℝ V = 2)

-- Define the unit vectors
variable (a b c : V)
variable (ha : ‖a‖ = 1)
variable (hb : ‖b‖ = 1)
variable (hc : ‖c‖ = 1)

-- Define the theorem
theorem exists_vector_bounded_norm :
  ∃ (sa sb sc : ℝ) (hsa : sa = 1 ∨ sa = -1) (hsb : sb = 1 ∨ sb = -1) (hsc : sc = 1 ∨ sc = -1),
  ‖sa • a + sb • b + sc • c‖ ≤ Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_vector_bounded_norm_l566_56634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_theorem_l566_56671

/-- Represents a parabola y² = 2px with p > 0 -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- Represents a point (x, y) -/
structure Point where
  x : ℝ
  y : ℝ

/-- The focus of a parabola -/
noncomputable def focus (c : Parabola) : Point :=
  ⟨c.p / 2, 0⟩

/-- Check if a point is on the parabola -/
def on_parabola (c : Parabola) (m : Point) : Prop :=
  m.y^2 = 2 * c.p * m.x

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Chord length created by a circle intersecting the y-axis -/
noncomputable def chord_length (center : Point) (radius : ℝ) : ℝ :=
  2 * Real.sqrt (radius^2 - center.x^2)

theorem parabola_theorem (c : Parabola) (m : Point) 
    (h_on_parabola : on_parabola c m)
    (h_y : m.y = 2 * Real.sqrt 2)
    (h_chord : chord_length m (distance m (focus c)) = 2 * Real.sqrt 5) : 
  c.p = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_theorem_l566_56671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_point_l566_56679

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 2

-- Define the derivative of the function
noncomputable def f_derivative (x : ℝ) : ℝ := x^2

-- Theorem statement
theorem tangent_slope_at_point :
  let x₀ : ℝ := 1
  let y₀ : ℝ := -5/3
  f x₀ = y₀ ∧ f_derivative x₀ = 1 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_point_l566_56679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l566_56632

/-- A hyperbola with foci F₁ and F₂ -/
structure Hyperbola where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- A point on a hyperbola -/
def PointOnHyperbola (C : Hyperbola) (P : ℝ × ℝ) : Prop :=
  ∃ a : ℝ, |P.1 - C.F₁.1| - |P.1 - C.F₂.1| = 2 * a

/-- The angle between two vectors -/
noncomputable def angle (v w : ℝ × ℝ) : ℝ := 
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (C : Hyperbola) : ℝ := 
  Real.sqrt ((C.F₁.1 - C.F₂.1)^2 + (C.F₁.2 - C.F₂.2)^2) / (|C.F₁.1 - C.F₂.1| / 2)

/-- Theorem: If there exists a point P on hyperbola C such that ∠F₁PF₂ = 60° and |PF₁| = 3|PF₂|, 
    then the eccentricity of C is √7/2 -/
theorem hyperbola_eccentricity (C : Hyperbola) (P : ℝ × ℝ) 
  (h₁ : PointOnHyperbola C P)
  (h₂ : angle (C.F₁.1 - P.1, C.F₁.2 - P.2) (C.F₂.1 - P.1, C.F₂.2 - P.2) = π / 3)
  (h₃ : Real.sqrt ((P.1 - C.F₁.1)^2 + (P.2 - C.F₁.2)^2) = 3 * Real.sqrt ((P.1 - C.F₂.1)^2 + (P.2 - C.F₂.2)^2)) :
  eccentricity C = Real.sqrt 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l566_56632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_slopes_theorem_l566_56692

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the foci (we don't know their exact coordinates, so we leave them abstract)
variable (F1 F2 : ℝ × ℝ)

-- Define a point on the ellipse
variable (P : ℝ × ℝ)
variable (h_P : is_on_ellipse P.1 P.2)

-- Define the vector from a point to another
def vec (A B : ℝ × ℝ) : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define vector addition
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

-- Define vector magnitude
noncomputable def vec_mag (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Part 1: Minimum value theorem
theorem min_value_theorem :
  vec_mag (vec_add (vec P F1) (vec P F2)) ≥ 2 :=
by sorry

-- Define another point on the ellipse
variable (Q : ℝ × ℝ)
variable (h_Q : is_on_ellipse Q.1 Q.2)

-- Define perpendicularity of vectors
def perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

-- Part 2: Slopes theorem
theorem slopes_theorem (h_perp : perpendicular (vec_add (vec P F1) (vec P F2)) (vec_add (vec Q F1) (vec Q F2)))
  (h_through_C : perpendicular (vec (1, 0) P) (vec (1, 0) Q)) :
  (P.2 / P.1 = Real.sqrt ((2 * Real.sqrt 10 - 5) / 10) ∧ Q.2 / Q.1 = -Real.sqrt ((2 * Real.sqrt 10 - 5) / 10)) ∨
  (P.2 / P.1 = -Real.sqrt ((2 * Real.sqrt 10 - 5) / 10) ∧ Q.2 / Q.1 = Real.sqrt ((2 * Real.sqrt 10 - 5) / 10)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_slopes_theorem_l566_56692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_increasing_power_function_l566_56659

-- Define the type for real numbers greater than 0
def PositiveReal := {x : ℝ | x > 0}

-- Define the function f(x) = x^a
noncomputable def f (a : ℝ) (x : PositiveReal) : ℝ := x.val ^ a

-- Define what it means for a function to be increasing on PositiveReal
def IsIncreasing (g : PositiveReal → ℝ) : Prop :=
  ∀ (x y : PositiveReal), x.val < y.val → g x < g y

-- State the theorem
theorem negation_of_increasing_power_function :
  (¬ ∀ (a : ℝ), a > 1 → IsIncreasing (f a)) ↔
  (∃ (a₀ : ℝ), a₀ > 1 ∧ ¬ IsIncreasing (f a₀)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_increasing_power_function_l566_56659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l566_56609

/-- Given that cos(75° + α) = 1/3 and α is an angle in the third quadrant,
    prove that cos(105° - α) + sin(α - 105°) = (2√2 - 1) / 3 -/
theorem trigonometric_identity (α : ℝ) 
    (h1 : Real.cos (75 * π / 180 + α) = 1/3) 
    (h2 : π < α ∧ α < 3*π/2) : 
    Real.cos (105 * π / 180 - α) + Real.sin (α - 105 * π / 180) = (2 * Real.sqrt 2 - 1) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l566_56609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_rotation_matrix_l566_56675

open Real Matrix

-- Define a 2D rotation matrix
noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![Real.cos θ, -Real.sin θ;
     Real.sin θ,  Real.cos θ]

-- State the theorem
theorem det_rotation_matrix (θ : ℝ) :
  det (rotation_matrix θ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_rotation_matrix_l566_56675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_on_dividing_segment_l566_56668

/-- A triangle -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A line segment dividing a triangle -/
structure DividingSegment (T : Triangle) where
  start : ℝ × ℝ
  endpoint : ℝ × ℝ

/-- The center of the inscribed circle of a triangle -/
noncomputable def incenter (T : Triangle) : ℝ × ℝ := sorry

/-- Predicate to check if a point lies on a line segment -/
def lies_on (p : ℝ × ℝ) (s : DividingSegment T) : Prop := sorry

/-- Predicate to check if two figures have equal perimeters -/
def equal_perimeters (T : Triangle) (s : DividingSegment T) : Prop := sorry

/-- Predicate to check if two figures have equal areas -/
def equal_areas (T : Triangle) (s : DividingSegment T) : Prop := sorry

/-- Main theorem -/
theorem incenter_on_dividing_segment (T : Triangle) (s : DividingSegment T) :
  equal_perimeters T s → equal_areas T s → lies_on (incenter T) s := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_on_dividing_segment_l566_56668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contact_loss_time_l566_56657

/-- The time when two teams lose radio contact -/
noncomputable def time_to_lose_contact (team1_speed team2_speed radio_range : ℝ) : ℝ :=
  radio_range / (team1_speed + team2_speed)

/-- Theorem: The time to lose contact is 2.5 hours under given conditions -/
theorem contact_loss_time :
  time_to_lose_contact 20 30 125 = 2.5 := by
  -- Unfold the definition of time_to_lose_contact
  unfold time_to_lose_contact
  -- Simplify the arithmetic
  norm_num

-- To demonstrate the result without computation
example : time_to_lose_contact 20 30 125 = 2.5 := contact_loss_time

-- If you want to see the numeric result, you can use the following:
-- #eval (time_to_lose_contact 20 30 125 : Float)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contact_loss_time_l566_56657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_black_balls_l566_56606

def total_balls : ℕ := 11
def white_balls : ℕ := 4
def black_balls : ℕ := 7

def total_ways : ℕ := Nat.choose total_balls 2
def black_ways : ℕ := Nat.choose black_balls 2

theorem probability_two_black_balls :
  (black_ways : Rat) / total_ways = 21 / 55 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_black_balls_l566_56606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_rate_calculation_l566_56630

/-- Calculates the walking rate in miles per hour given distance and time -/
noncomputable def walking_rate (distance : ℝ) (hours : ℝ) (minutes : ℝ) : ℝ :=
  distance / (hours + minutes / 60)

/-- Theorem: If a person walks 8 miles in 1 hour and 15 minutes, their walking rate is 6.4 miles per hour -/
theorem walking_rate_calculation :
  walking_rate 8 1 15 = 6.4 := by
  -- Unfold the definition of walking_rate
  unfold walking_rate
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_rate_calculation_l566_56630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_linear_not_direct_proportion_l566_56697

noncomputable def f (x : ℝ) : ℝ := -(x - 1) / 2

theorem f_is_linear_not_direct_proportion :
  (∃ k b : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x + b) ∧
  (¬∃ k : ℝ, ∀ x, f x = k * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_linear_not_direct_proportion_l566_56697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l566_56614

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (1/2)^(x^2 + 2*x - 1)

-- State the theorem
theorem range_of_f :
  ∃ (S : Set ℝ), S = Set.range f ∧ S = Set.Ioo 0 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l566_56614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_specific_terms_l566_56612

def f (x : ℝ) := x^2

def tangent_intersection (a : ℕ → ℝ) : Prop :=
  ∀ k : ℕ, k > 0 → a (k + 1) = (1 / 2) * a k

theorem sum_of_specific_terms (a : ℕ → ℝ) :
  (∀ x : ℝ, x > 0 → Differentiable ℝ (fun x => f x)) →
  (∀ k : ℕ, k > 0 → a k > 0) →
  tangent_intersection a →
  a 1 = 16 →
  a 1 + a 3 + a 5 = 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_specific_terms_l566_56612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l566_56616

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - 5*x + 4 * Real.log x

-- State the theorem
theorem f_properties :
  -- Domain is (0, +∞)
  (∀ x : ℝ, x > 0 ↔ f x ∈ Set.range f) ∧
  -- Monotonicity
  (∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 1 → f x > f y) ∧
  (∀ x y : ℝ, 1 < x ∧ x < y → f x < f y) ∧
  -- Maximum value
  (∀ x : ℝ, x > 0 → f x ≤ f 1) ∧
  f 1 = -9/2 ∧
  -- Minimum value
  (∀ x : ℝ, x > 0 → f x ≥ f 4) ∧
  f 4 = -12 + 4 * Real.log 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l566_56616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_values_of_f_l566_56644

noncomputable def f (x : ℝ) : ℝ := (11 * x^2 - 5 * x + 6) / (x^2 + 5 * x + 6) - x

theorem negative_values_of_f :
  ∀ x : ℝ, f x < 0 ↔ (x > -3 ∧ x < -2) ∨ (x > 1 ∧ x < 2) ∨ (x > 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_values_of_f_l566_56644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_complementary_1999_digit_numbers_l566_56696

/-- Count the occurrences of a digit in a natural number. -/
def digit_count (n : ℕ) (d : Fin 10) : ℕ :=
  sorry

/-- Count the number of digits in a natural number. -/
def num_digits (n : ℕ) : ℕ :=
  sorry

/-- Given two natural numbers with 1999 digits each, having the same decimal digits in different order,
    their sum cannot be 999...999 (1999 digits of 9). -/
theorem no_complementary_1999_digit_numbers : 
  ¬ ∃ (A B : ℕ), 
    (∀ d : Fin 10, (digit_count A d = digit_count B d)) ∧ 
    (num_digits A = 1999) ∧ 
    (num_digits B = 1999) ∧ 
    (A + B = 10^1999 - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_complementary_1999_digit_numbers_l566_56696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_midpoint_vector_l566_56638

/-- Given a parallelogram ABCD with AB = a, AC = b, and E the midpoint of CD, prove EB = (3/2)a - b -/
theorem parallelogram_midpoint_vector (a b : ℝ × ℝ) :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := a
  let C : ℝ × ℝ := b
  let D : ℝ × ℝ := b - a
  let E : ℝ × ℝ := D + (1/2) • (B - D)
  E - B = (3/2) • a - b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_midpoint_vector_l566_56638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spheres_from_given_cylinder_l566_56681

/-- The number of spheres that can be made from a cylinder -/
noncomputable def spheres_from_cylinder (cylinder_diameter : ℝ) (cylinder_height : ℝ) (sphere_diameter : ℝ) : ℝ :=
  let cylinder_volume := Real.pi * (cylinder_diameter / 2)^2 * cylinder_height
  let sphere_volume := (4 / 3) * Real.pi * (sphere_diameter / 2)^3
  cylinder_volume / sphere_volume

/-- Theorem: The number of spheres with diameter 8 cm that can be made from a cylinder
    with diameter 8 cm and height 48 cm is equal to 9 -/
theorem spheres_from_given_cylinder :
  spheres_from_cylinder 8 48 8 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spheres_from_given_cylinder_l566_56681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_work_rate_problem_l566_56690

/-- The fraction of work that two workers can complete together in one day, given their individual work rates. -/
noncomputable def combined_work_rate (time_a : ℝ) (time_b : ℝ) : ℝ :=
  1 / time_a + 1 / time_b

/-- Theorem: Given that A can finish a work in 4 days and B can do the same work in half the time taken by A,
    A and B working together can finish 3/4 of the work in one day. -/
theorem combined_work_rate_problem :
  let time_a : ℝ := 4
  let time_b : ℝ := time_a / 2
  combined_work_rate time_a time_b = 3 / 4 := by
  -- Unfold the definition of combined_work_rate
  unfold combined_work_rate
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_work_rate_problem_l566_56690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triple_second_column_l566_56664

-- Define a 2x2 matrix type
def Matrix2x2 := Fin 2 → Fin 2 → ℝ

-- Define the transformation matrix M
def M : Matrix2x2 := λ i j => if i = j then if i = 0 then 1 else 3 else 0

-- Define matrix multiplication
def Matrix2x2.mul (A B : Matrix2x2) : Matrix2x2 :=
  λ i j => (Finset.sum (Finset.range 2) $ λ k => A i k * B k j)

-- State the theorem
theorem triple_second_column (A : Matrix2x2) :
  ∀ i j, (Matrix2x2.mul M A) i j = if j = 1 then 3 * A i j else A i j :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triple_second_column_l566_56664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_c_l566_56625

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x + Real.sin x, Real.sin x)

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x - Real.sin x, 2 * Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem triangle_side_c (A B C : ℝ) (a b c : ℝ) :
  A = π / 3 →
  f B = 1 →
  Real.sqrt 3 * a + Real.sqrt 2 * b = 10 →
  c = Real.sqrt 6 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_c_l566_56625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_A_fill_time_l566_56647

/-- Represents the time (in minutes) it takes for a pipe to fill or empty a cistern -/
structure PipeTime where
  minutes : ℝ
  minutes_pos : minutes > 0

/-- Represents the rate at which a pipe fills or empties a cistern (in cisterns per minute) -/
noncomputable def fill_rate (t : PipeTime) : ℝ := 1 / t.minutes

theorem pipe_A_fill_time 
  (pipe_A : PipeTime) 
  (pipe_B : PipeTime)
  (pipe_empty : PipeTime)
  (all_pipes : PipeTime)
  (h1 : pipe_B.minutes = 60)
  (h2 : pipe_empty.minutes = 72)
  (h3 : all_pipes.minutes = 40)
  (h4 : fill_rate pipe_A + fill_rate pipe_B - fill_rate pipe_empty = fill_rate all_pipes) :
  pipe_A.minutes = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_A_fill_time_l566_56647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_first_segment_time_fraction_l566_56618

/-- A journey with two segments at different speeds -/
structure Journey where
  totalDistance : ℝ
  totalTime : ℝ
  speed1 : ℝ
  speed2 : ℝ

/-- The fraction of distance covered in the first segment -/
noncomputable def firstSegmentDistanceFraction : ℝ := 2/3

/-- The fraction of distance covered in the second segment -/
noncomputable def secondSegmentDistanceFraction : ℝ := 1/3

/-- Theorem stating that the fraction of time spent on the first segment is 1/2 -/
theorem journey_first_segment_time_fraction (j : Journey) 
  (h1 : j.speed1 = 80)
  (h2 : j.speed2 = 40)
  (h3 : j.totalTime = (firstSegmentDistanceFraction * j.totalDistance / j.speed1) + 
                      (secondSegmentDistanceFraction * j.totalDistance / j.speed2)) :
  (firstSegmentDistanceFraction * j.totalDistance / j.speed1) / j.totalTime = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_first_segment_time_fraction_l566_56618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l566_56650

noncomputable def f (x : ℝ) := Real.cos (Real.sin x)

theorem f_properties : 
  (∀ x, f (-x) = f x) ∧ 
  (∃ p > 0, ∀ x, f (x + p) = f x) ∧
  (∀ q > 0, (∀ x, f (x + q) = f x) → q ≥ π) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l566_56650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_circle_property_l566_56682

/-- A parabola with focus (1,0) and directrix x=-1 -/
structure Parabola where
  -- The equation y^2 = 4x
  eq : ℝ → ℝ → Prop

/-- A tangent line to the parabola -/
structure TangentLine (p : Parabola) where
  k : ℝ
  b : ℝ
  is_tangent : b = 1 / k

/-- The theorem about the parabola and its tangent lines -/
theorem parabola_tangent_circle_property (p : Parabola) (l : TangentLine p) :
  ∃ (x y : ℝ),
    -- (x, y) is on the parabola
    p.eq x y ∧
    -- (x, y) is on the tangent line
    y = l.k * x + l.b ∧
    -- The circle property
    let q_x : ℝ := -1
    let q_y : ℝ := l.k * q_x + l.b
    (x - 1) * (q_x - 1) + y * q_y = 0 :=
by sorry

/-- The specific parabola y^2 = 4x -/
def specific_parabola : Parabola where
  eq := fun x y => y^2 = 4*x

#check parabola_tangent_circle_property specific_parabola

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_circle_property_l566_56682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_circle_quadrilateral_l566_56649

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A quadrilateral in a 2D plane -/
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

/-- A function that counts the number of intersection points between a circle and a line segment -/
def intersectionPoints (c : Circle) (p1 p2 : ℝ × ℝ) : ℕ := sorry

/-- A function that counts the total number of intersection points between a circle and a quadrilateral -/
def totalIntersectionPoints (c : Circle) (q : Quadrilateral) : ℕ :=
  (intersectionPoints c (q.vertices 0) (q.vertices 1)) +
  (intersectionPoints c (q.vertices 1) (q.vertices 2)) +
  (intersectionPoints c (q.vertices 2) (q.vertices 3)) +
  (intersectionPoints c (q.vertices 3) (q.vertices 0))

/-- Theorem stating that the maximum number of intersection points between a circle and a quadrilateral is 8 -/
theorem max_intersection_points_circle_quadrilateral :
  ∀ (c : Circle) (q : Quadrilateral), totalIntersectionPoints c q ≤ 8 ∧ 
  ∃ (c' : Circle) (q' : Quadrilateral), totalIntersectionPoints c' q' = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_circle_quadrilateral_l566_56649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_face_area_ratio_l566_56643

theorem tetrahedron_face_area_ratio (S₁ S₂ S₃ S₄ : ℝ) (S : ℝ) (h_pos : S > 0) 
  (h_max : S = max S₁ (max S₂ (max S₃ S₄))) :
  let lambda := (S₁ + S₂ + S₃ + S₄) / S
  2 < lambda ∧ lambda ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_face_area_ratio_l566_56643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l566_56615

theorem sequence_property (a : ℕ → ℕ) :
  (∀ n : ℕ, n > 0 → ((n ^ 2 + 1) * a n = n * (a (n ^ 2) + 1))) →
  ∀ n : ℕ, n > 0 → a n = n := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l566_56615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_M_l566_56633

def M : Nat := 2^5 * 3^3 * 5^2 * 7^1 * 11^1

theorem number_of_factors_of_M : 
  (Finset.filter (· ∣ M) (Finset.range (M + 1))).card = 288 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_M_l566_56633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_range_l566_56617

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (a - exp (-x))

-- Define the derivative of f
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := a + (x - 1) * exp (-x)

-- Theorem statement
theorem tangent_perpendicular_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    f_derivative a x₁ = 0 ∧ 
    f_derivative a x₂ = 0) →
  a > -exp (-2) ∧ a < 0 :=
by
  sorry

-- Additional lemma to help with the proof
lemma min_value_at_two (a : ℝ) :
  ∀ x : ℝ, (1 - x) * exp (-x) ≥ -exp (-2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_range_l566_56617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_ellipse_properties_l566_56684

/-- Represents an ellipse in standard form -/
structure StandardEllipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- Theorem about a specific ellipse passing through two given points -/
theorem specific_ellipse_properties :
  ∃ (e : StandardEllipse),
    -- The ellipse passes through the given points
    (2^2 / e.a^2 + (-4*Real.sqrt 5/3)^2 / e.b^2 = 1) ∧
    ((-1)^2 / e.a^2 + (8*Real.sqrt 2/3)^2 / e.b^2 = 1) ∧
    -- The ellipse has the specified properties
    e.a = 3 ∧
    e.b = 4 ∧
    -- Foci coordinates
    (∃ (c : ℝ), c^2 = 7 ∧
      ((0, c) ∈ Set.range (λ x : ℝ × ℝ ↦ (x.1, x.2)) ∧
       (0, -c) ∈ Set.range (λ x : ℝ × ℝ ↦ (x.1, x.2)))) ∧
    -- Eccentricity
    (Real.sqrt 7 / 4 = Real.sqrt (1 - e.b^2 / e.a^2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_ellipse_properties_l566_56684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_bound_l566_56607

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

-- Define the lines and points
variable (m₁ m₂ m₃ m₄ : Set V)
variable (O A₁ A₂ A₃ A₄ B : V)

-- State the theorem
theorem intersection_distance_bound
  (h₁ : A₁ ∈ m₁)
  (h₂ : A₂ ∈ m₂)
  (h₃ : A₃ ∈ m₃)
  (h₄ : A₄ ∈ m₄)
  (h₅ : B ∈ m₁)
  (h₆ : O ∈ m₁ ∩ m₂ ∩ m₃ ∩ m₄)
  (h₇ : ∃ k₁ : ℝ, A₂ - A₁ = k₁ • (A₄ - O))
  (h₈ : ∃ k₂ : ℝ, A₃ - A₂ = k₂ • (A₁ - O))
  (h₉ : ∃ k₃ : ℝ, A₄ - A₃ = k₃ • (A₂ - O))
  (h₁₀ : ∃ k₄ : ℝ, B - A₄ = k₄ • (A₃ - O)) :
  ‖B - O‖ ≤ (1/4 : ℝ) * ‖A₁ - O‖ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_bound_l566_56607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_range_l566_56670

theorem derivative_range (θ : Real) (h : θ ∈ Set.Icc 0 (5 * Real.pi / 12)) :
  let f : ℝ → ℝ := λ x => (Real.sin θ / 3) * x^3 + (Real.sqrt 3 * Real.cos θ / 2) * x^2 + Real.tan θ
  let f' : ℝ → ℝ := λ x => Real.sin θ * x^2 + Real.sqrt 3 * Real.cos θ * x
  f' 1 ∈ Set.Icc (Real.sqrt 2) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_range_l566_56670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_decrease_when_area_decreases_75_percent_l566_56672

-- Define the original radius and area
noncomputable def original_radius : ℝ := sorry
noncomputable def original_area : ℝ := Real.pi * original_radius^2

-- Define the new radius and area
noncomputable def new_radius : ℝ := sorry
noncomputable def new_area : ℝ := Real.pi * new_radius^2

-- State the theorem
theorem radius_decrease_when_area_decreases_75_percent :
  new_area = 0.25 * original_area → new_radius = 0.5 * original_radius := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_decrease_when_area_decreases_75_percent_l566_56672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_gain_percentage_l566_56642

/-- Represents a type of cloth with its properties -/
structure ClothType where
  length : ℝ
  costPerMeter : ℝ
  sellingPricePerMeter : ℝ

/-- Calculates the gain percentage given a list of cloth types -/
noncomputable def gainPercentage (clothTypes : List ClothType) : ℝ :=
  let totalCost := (clothTypes.map (fun c => c.length * c.costPerMeter)).sum
  let totalSelling := (clothTypes.map (fun c => c.length * c.sellingPricePerMeter)).sum
  let gain := totalSelling - totalCost
  (gain / totalCost) * 100

/-- The four types of cloth with their properties -/
def clothTypes : List ClothType := [
  ⟨40, 2.5, 3.5⟩,  -- Type A
  ⟨55, 3.0, 4.0⟩,  -- Type B
  ⟨36, 4.5, 5.5⟩,  -- Type C
  ⟨45, 6.0, 7.0⟩   -- Type D
]

theorem overall_gain_percentage :
  abs (gainPercentage clothTypes - 25.25) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_gain_percentage_l566_56642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_interval_l566_56661

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 4*x + 3) / Real.log 10

-- Define the domain of f(x)
def domain (x : ℝ) : Prop := x < 1 ∨ x > 3

-- State the theorem
theorem f_monotonic_increasing_interval :
  ∀ x y, domain x → domain y → x > 3 → y > 3 → x < y → f x < f y :=
by
  -- The proof is skipped for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_interval_l566_56661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_congruence_l566_56677

/-- A line in a 2D plane --/
structure Line where
  -- Define properties of a line
  a : ℝ
  b : ℝ
  c : ℝ

/-- A triangle in a 2D plane --/
structure Triangle where
  -- Define properties of a triangle
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

/-- Two triangles are symmetrical about a line --/
def symmetrical_about (t1 t2 : Triangle) (l : Line) : Prop :=
  -- Define the condition for two triangles to be symmetrical about a line
  sorry

/-- Two triangles are congruent --/
def congruent (t1 t2 : Triangle) : Prop :=
  -- Define the condition for two triangles to be congruent
  sorry

/-- Theorem: If two triangles are symmetrical about a line, then they are congruent --/
theorem symmetry_implies_congruence (t1 t2 : Triangle) (l : Line) :
  symmetrical_about t1 t2 l → congruent t1 t2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_congruence_l566_56677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_earnings_l566_56623

/-- Calculates the total earnings for a worker based on survey completion --/
def calculate_earnings (regular_rate : ℚ) (simple_rate_increase : ℚ) (moderate_rate_increase : ℚ) (complex_rate_increase : ℚ) (bonus_amount : ℚ) (bonus_threshold : ℕ) (simple_surveys : ℕ) (moderate_surveys : ℕ) (complex_surveys : ℕ) (regular_surveys : ℕ) : ℚ :=
  let total_surveys := simple_surveys + moderate_surveys + complex_surveys + regular_surveys
  let simple_rate := regular_rate * (1 + simple_rate_increase)
  let moderate_rate := regular_rate * (1 + moderate_rate_increase)
  let complex_rate := regular_rate * (1 + complex_rate_increase)
  let survey_earnings := 
    regular_rate * regular_surveys +
    simple_rate * simple_surveys +
    moderate_rate * moderate_surveys +
    complex_rate * complex_surveys
  let bonus := (total_surveys / bonus_threshold) * bonus_amount
  survey_earnings + bonus

theorem worker_earnings :
  calculate_earnings 10 (3/10) (1/2) (3/4) 50 25 30 20 10 40 = 1465 := by
  sorry

#eval calculate_earnings 10 (3/10) (1/2) (3/4) 50 25 30 20 10 40

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_earnings_l566_56623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_probability_value_l566_56629

/-- The probability function P(a) -/
noncomputable def P (a : ℝ) : ℝ :=
  let f (x : ℝ) := Real.sin (Real.pi * x)^2 + Real.sin (Real.pi * (x / 2))^2 > 3/2
  (∫ x in Set.Icc 0 (a^2), if f x then 1 else 0) / a^2

/-- The theorem stating the maximum value of P(a) -/
theorem max_probability_value :
  (∀ a : ℝ, 0 ≤ a ∧ a ≤ 1) →
  (∃ a : ℝ, 0 ≤ a ∧ a ≤ 1 ∧ P a = 5/6) ∧
  (∀ a : ℝ, 0 ≤ a ∧ a ≤ 1 → P a ≤ 5/6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_probability_value_l566_56629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_M_is_sqrt_2_l566_56619

noncomputable def f (x : ℝ) : ℝ := Real.log x

theorem min_M_is_sqrt_2 (M : ℝ) :
  (M > 0) →
  (∀ a b c : ℝ, a > M ∧ b > M ∧ c > M →
    (a^2 + b^2 = c^2) →
    (f a + f b > f c)) →
  (M ≥ Real.sqrt 2) ∧ 
  (∀ ε > 0, ∃ a b c : ℝ, 
    a > M - ε ∧ b > M - ε ∧ c > M - ε ∧
    a^2 + b^2 = c^2 ∧
    f a + f b > f c) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_M_is_sqrt_2_l566_56619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_divisibility_l566_56685

theorem arithmetic_sequence_divisibility (n : ℕ) (h : n ≥ 2) :
  (∃ (a : ℕ → ℕ), (∀ i j k : ℕ, a (i + k) - a i = a (j + k) - a j) ∧
    (∀ k : ℕ, k < n → k ∣ a k) ∧
    ¬(n ∣ a n)) →
  ∃ (p : ℕ) (α : ℕ), Nat.Prime p ∧ n = p ^ α :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_divisibility_l566_56685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_discount_percentage_l566_56698

theorem second_discount_percentage (original_price first_discount final_price : ℝ) 
  (h1 : original_price = 480)
  (h2 : first_discount = 15)
  (h3 : final_price = 306) : 
  (let price_after_first_discount := original_price * (1 - first_discount / 100)
   let second_discount := (price_after_first_discount - final_price) / price_after_first_discount * 100
   second_discount) = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_discount_percentage_l566_56698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l566_56686

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := -1/3 * x^3 + x^2 + a*x + b

-- State the theorem
theorem max_value_of_f (a b : ℝ) :
  (∃ (x : ℝ), f a b x = 4 ∧ x = 3) →
  (∃ (max_val : ℝ), max_val = -4/3 ∧
    ∀ (x : ℝ), -2 ≤ x ∧ x ≤ 1 → f a b x ≤ max_val) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l566_56686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l566_56641

/-- A hyperbola with given parameters -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  a_pos : a > 0
  b_pos : b > 0
  asymptote_slope : ℝ
  asymptote_eq : asymptote_slope = Real.sqrt 2

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + h.b^2 / h.a^2)

/-- Theorem: The eccentricity of the given hyperbola is √3 -/
theorem hyperbola_eccentricity (h : Hyperbola) : eccentricity h = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l566_56641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_division_implies_specific_angles_l566_56699

-- Define a right-angled triangle
structure RightTriangle where
  a : ℝ  -- length of one leg
  b : ℝ  -- length of the other leg
  c : ℝ  -- length of the hypotenuse
  right_angle : a^2 + b^2 = c^2  -- Pythagorean theorem

-- Define the circle property
def circle_divides_hypotenuse (t : RightTriangle) : Prop :=
  ∃ (x : ℝ), x > 0 ∧ x < t.c ∧ x / (t.c - x) = 1 / 3

-- Define the acute angles
noncomputable def acute_angles (t : RightTriangle) : Prop :=
  ∃ (θ₁ θ₂ : ℝ), 
    θ₁ > 0 ∧ θ₁ < Real.pi/2 ∧
    θ₂ > 0 ∧ θ₂ < Real.pi/2 ∧
    θ₁ + θ₂ = Real.pi/2 ∧
    (θ₁ = Real.pi/6 ∧ θ₂ = Real.pi/3) ∨ (θ₁ = Real.pi/3 ∧ θ₂ = Real.pi/6)

-- State the theorem
theorem circle_division_implies_specific_angles (t : RightTriangle) :
  circle_divides_hypotenuse t → acute_angles t :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_division_implies_specific_angles_l566_56699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_solution_set_implies_c_value_l566_56627

/-- A quadratic function f(x) = x^2 + ax + b with range [0, +∞) -/
def QuadraticFunction (a b : ℝ) : ℝ → ℝ := λ x ↦ x^2 + a*x + b

/-- The solution set of f(x) < c is an open interval (m, m+6) -/
def SolutionSet (f : ℝ → ℝ) (c m : ℝ) : Prop :=
  ∀ x, f x < c ↔ m < x ∧ x < m + 6

theorem quadratic_solution_set_implies_c_value
  (a b c m : ℝ)
  (h_range : ∀ x, QuadraticFunction a b x ≥ 0)
  (h_solution : SolutionSet (QuadraticFunction a b) c m) :
  c = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_solution_set_implies_c_value_l566_56627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maxwell_brad_meeting_time_l566_56691

/-- The time it takes for Maxwell and Brad to meet up -/
noncomputable def meeting_time (distance : ℝ) (maxwell_speed : ℝ) (brad_speed : ℝ) (head_start : ℝ) : ℝ :=
  let t := (distance - maxwell_speed * head_start) / (maxwell_speed + brad_speed)
  t + head_start

/-- Theorem stating that Maxwell and Brad meet after 2 hours -/
theorem maxwell_brad_meeting_time :
  let distance := (14 : ℝ) -- km
  let maxwell_speed := (4 : ℝ) -- km/h
  let brad_speed := (6 : ℝ) -- km/h
  let head_start := (1 : ℝ) -- hour
  meeting_time distance maxwell_speed brad_speed head_start = 2 := by
  -- Unfold the definition of meeting_time
  unfold meeting_time
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_maxwell_brad_meeting_time_l566_56691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_three_alpha_minus_beta_l566_56626

theorem sin_three_alpha_minus_beta (α β : ℝ) : 
  (0 < α ∧ α < π / 2) →  -- α is an acute angle
  (π / 2 < β ∧ β < π) →  -- β is in the second quadrant
  Real.cos (α - β) = 1 / 2 →
  Real.sin (α + β) = 1 / 2 →
  Real.sin (3 * α - β) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_three_alpha_minus_beta_l566_56626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_constants_and_variables_l566_56689

-- Define the formula for the volume of a cone
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

-- Define the set of constants in the formula
def constants : Set ℝ := {1/3, Real.pi}

-- Define the set of variables in the formula
def variable_names : Set String := {"V", "r", "h"}

-- Theorem statement
theorem cone_volume_constants_and_variables :
  (constants = {1/3, Real.pi}) ∧ 
  (variable_names = {"V", "r", "h"}) := by
  sorry

#check cone_volume_constants_and_variables

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_constants_and_variables_l566_56689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_characterization_of_a_l566_56669

/-- The function f(x) = (x^2 + 2x + a) / x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + 2*x + a) / x

/-- The theorem stating the range of a given the conditions -/
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≥ 1 → f a x > 0) → a > -3 :=
by
  sorry

/-- The theorem stating the complete characterization of a -/
theorem characterization_of_a : 
  ∀ a : ℝ, (∀ x : ℝ, x ≥ 1 → f a x > 0) ↔ a > -3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_characterization_of_a_l566_56669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l566_56676

/-- A sequence satisfying the given sum property -/
def SequenceWithSumProperty (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → (Finset.range n).sum a = n^2

/-- Three terms form an arithmetic sequence -/
def ArithmeticSequence (x y z : ℚ) : Prop :=
  y - x = z - y

theorem sequence_property (a : ℕ → ℕ) (k p r : ℕ)
    (h_seq : SequenceWithSumProperty a)
    (h_order : k < p ∧ p < r)
    (h_arith : ArithmeticSequence (1 / (a k : ℚ)) (1 / (a p : ℚ)) (1 / (a r : ℚ)))
    (h_k_pos : k > 0) (h_p_pos : p > 0) (h_r_pos : r > 0) :
    p = 2 * k - 1 ∧ r = 4 * k^2 - 5 * k + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l566_56676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_n_value_l566_56645

theorem n_value : ∃ n : ℝ, 5 * 16 * 2 * n^2 = Nat.factorial 8 ∧ n = 6 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_n_value_l566_56645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_small_to_large_corral_l566_56673

-- Define the side length of small corrals
def small_side : ℝ := 10

-- Define the number of small corrals
def num_small_corrals : ℕ := 6

-- Define the perimeter of a small corral
def small_perimeter : ℝ := 3 * small_side

-- Define the total perimeter of all small corrals
noncomputable def total_small_perimeter : ℝ := (num_small_corrals : ℝ) * small_perimeter

-- Define the side length of the large corral
noncomputable def large_side : ℝ := total_small_perimeter / 3

-- Define the area of an equilateral triangle
noncomputable def equilateral_triangle_area (side : ℝ) : ℝ := (Real.sqrt 3 / 4) * side ^ 2

-- Theorem statement
theorem area_ratio_small_to_large_corral :
  ((num_small_corrals : ℝ) * equilateral_triangle_area small_side) / 
  (equilateral_triangle_area large_side) = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_small_to_large_corral_l566_56673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_value_l566_56631

def our_sequence (a : ℕ → ℤ) : Prop :=
  (∀ n, a (n + 1) = a (n + 2) + a n) ∧ a 1 = 2 ∧ a 2 = 5

theorem sixth_term_value (a : ℕ → ℤ) (h : our_sequence a) : a 6 = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_value_l566_56631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_separate_and_tangent_line_l566_56646

-- Define the circles
def circle_O : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}
def circle_C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 12*p.1 + 27 = 0}

-- Define the centers and radii
def center_O : ℝ × ℝ := (0, 0)
def center_C : ℝ × ℝ := (6, 0)
def radius_O : ℝ := 2
def radius_C : ℝ := 3

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := Real.sqrt ((center_C.1 - center_O.1)^2 + (center_C.2 - center_O.2)^2)

-- Define the tangent line equation
def tangent_line : ℝ → ℝ → Prop := λ x y => y = 2*Real.sqrt 2*x + 6 ∨ y = -2*Real.sqrt 2*x + 6

theorem circles_separate_and_tangent_line :
  (distance_between_centers > radius_O + radius_C) ∧
  (∀ x y, tangent_line x y → (x, y) ∈ circle_O → (x - center_C.1)^2 + (y - center_C.2)^2 = radius_C^2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_separate_and_tangent_line_l566_56646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_function_value_l566_56603

theorem sin_function_value (φ : ℝ) (ω : ℝ) (f : ℝ → ℝ) :
  Real.sin φ = 3/5 →
  φ > π/2 →
  φ < π →
  ω > 0 →
  (∀ x, f x = Real.sin (ω * x + φ)) →
  (∃ k, ω * (π/2) = 2 * π * k) →
  f (π/4) = -4/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_function_value_l566_56603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_multiple_of_64_l566_56637

def f (n : ℕ) : ℤ := 3^(2*n+2) - 8*n - 9

theorem f_multiple_of_64 : ∀ n : ℕ, ∃ k : ℤ, f n = 64 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_multiple_of_64_l566_56637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_empty_solution_set_l566_56651

theorem quadratic_inequality_empty_solution_set (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - m * x - 1 < 0) ↔ m ∈ Set.Ioc (-4) 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_empty_solution_set_l566_56651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_on_unit_circle_l566_56605

theorem max_distance_on_unit_circle :
  ∀ (α β : ℝ),
  let P := (Real.cos α, Real.sin α)
  let Q := (Real.cos β, Real.sin β)
  let distance := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  distance ≤ 2 ∧ ∃ (α' β' : ℝ), Real.sqrt ((Real.cos α' - Real.cos β')^2 + (Real.sin α' - Real.sin β')^2) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_on_unit_circle_l566_56605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_intersection_ratio_l566_56656

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define points D and E on the sides of the triangle
variable (D E : ℝ × ℝ)

-- Define the intersection point P
variable (P : ℝ × ℝ)

-- Define the ratios
variable (CD DB AE EB CP PE : ℝ)

-- Define a function for creating a line through two points
def line_through (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {r : ℝ × ℝ | ∃ t : ℝ, r = (1 - t) • p + t • q}

-- Theorem statement
theorem triangle_intersection_ratio
  (h1 : CD / DB = 3 / 1)
  (h2 : AE / EB = 3 / 2)
  (h3 : P ∈ line_through C E)
  (h4 : P ∈ line_through A D)
  : CP / PE = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_intersection_ratio_l566_56656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_distance_in_rectangle_l566_56660

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- A rectangle in a 2D plane -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Check if a point is inside a rectangle -/
def isInside (p : Point) (r : Rectangle) : Prop :=
  0 ≤ p.x ∧ p.x ≤ r.width ∧ 0 ≤ p.y ∧ p.y ≤ r.height

/-- Main theorem: In a 4x3 rectangle with 6 points, two points have distance ≤ √5 -/
theorem points_distance_in_rectangle :
  ∀ (points : Finset Point),
    points.card = 6 →
    (∀ p, p ∈ points → isInside p { width := 4, height := 3 : Rectangle }) →
    ∃ p1 p2, p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧ distance p1 p2 ≤ Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_distance_in_rectangle_l566_56660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_region_l566_56694

theorem point_region (m n : ℝ) : (2:ℝ)^m + (2:ℝ)^n < 4 → m + n < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_region_l566_56694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bullet_train_crossing_time_l566_56621

/-- The time taken for two bullet trains to cross each other -/
noncomputable def train_crossing_time (train_length : ℝ) (time1 time2 : ℝ) : ℝ :=
  (2 * train_length) / (train_length / time1 + train_length / time2)

/-- Theorem stating the time taken for two specific bullet trains to cross each other -/
theorem bullet_train_crossing_time :
  let train_length : ℝ := 120
  let time1 : ℝ := 10
  let time2 : ℝ := 12
  abs (train_crossing_time train_length time1 time2 - 10.91) < 0.01 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bullet_train_crossing_time_l566_56621
