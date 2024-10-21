import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_score_needed_l228_22844

noncomputable def current_scores : List ℝ := [95, 85, 75, 85, 90]

noncomputable def current_average : ℝ := (current_scores.sum) / (current_scores.length : ℝ)

def target_increase : ℝ := 5

noncomputable def target_average : ℝ := current_average + target_increase

def next_score : ℝ := 116

theorem minimum_score_needed :
  (((current_scores.sum + next_score) / ((current_scores.length + 1) : ℝ)) ≥ target_average) ∧
  (∀ x : ℝ, x < next_score → ((current_scores.sum + x) / ((current_scores.length + 1) : ℝ)) < target_average) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_score_needed_l228_22844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_bought_is_five_l228_22891

-- Define the price of sugar per kilogram
def sugar_price : ℚ := 3/2

-- Define the total price of 2kg sugar and x kg salt
def total_price_2s_xs : ℚ := 11/2

-- Define the total price of 3kg sugar and 1kg salt
def total_price_3s_1s : ℚ := 5

-- Define the function to calculate the number of kilograms of salt
noncomputable def salt_kg : ℚ := (total_price_2s_xs - 2 * sugar_price) / (total_price_3s_1s - 3 * sugar_price)

-- Theorem statement
theorem salt_bought_is_five : salt_kg = 5 := by
  -- Unfold the definition of salt_kg
  unfold salt_kg
  -- Simplify the expression
  simp [sugar_price, total_price_2s_xs, total_price_3s_1s]
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_bought_is_five_l228_22891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_task_completion_proof_l228_22802

noncomputable section

/-- Represents the number of days person A needs to complete the task alone -/
def deadline : ℝ := 6

/-- Represents the number of days person B needs to complete the task alone -/
def person_b_time : ℝ := deadline + 3

/-- Represents the portion of the task completed when two people work together for 2 days -/
def combined_work : ℝ := 2 / deadline + 2 / person_b_time

/-- Represents the portion of the task completed by person B working alone after the first 2 days -/
def remaining_work : ℝ := (deadline - 2) / person_b_time

theorem task_completion_proof : combined_work + remaining_work = 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_task_completion_proof_l228_22802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_eccentricity_l228_22851

-- Define the points B and C
def B : ℝ × ℝ := (-2, 0)
def C : ℝ × ℝ := (2, 0)

-- Define the perimeter of triangle ABC
noncomputable def perimeter (A : ℝ × ℝ) : ℝ := 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) +
  Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) +
  Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)

-- Define the eccentricity of the locus of point A
noncomputable def eccentricity : ℝ := 2/3

-- Theorem statement
theorem locus_eccentricity :
  ∀ A : ℝ × ℝ, perimeter A = 10 → eccentricity = 2/3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_eccentricity_l228_22851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l228_22898

-- Define the arithmetic sequence
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Define the sum of the first n terms
noncomputable def S (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

-- State the theorem
theorem arithmetic_sequence_problem (a₁ d : ℝ) :
  (arithmetic_sequence a₁ d 1 + arithmetic_sequence a₁ d 2 + arithmetic_sequence a₁ d 3 =
   arithmetic_sequence a₁ d 4 + arithmetic_sequence a₁ d 5) →
  (S a₁ d 5 = 60) →
  (arithmetic_sequence a₁ d 10 = 26) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l228_22898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l228_22877

/-- The function f(x, y, z) defined in the problem -/
noncomputable def f (x y z : ℝ) : ℝ := (3*x^2 - x)/(1 + x^2) + (3*y^2 - y)/(1 + y^2) + (3*z^2 - z)/(1 + z^2)

/-- Theorem stating the minimum value of f(x, y, z) -/
theorem min_value_of_f :
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z = 1 →
  f x y z ≥ 0 ∧ ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 ∧ f a b c = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l228_22877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_heart_king_clubs_l228_22883

/-- A standard deck of cards --/
structure Deck :=
  (cards : Finset (Fin 4 × Fin 13))
  (size : cards.card = 52)

/-- The suit of a card --/
inductive Suit
| Hearts | Diamonds | Clubs | Spades

/-- The rank of a card --/
inductive Rank
| Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- A card in the deck --/
structure Card :=
  (suit : Suit)
  (rank : Rank)

/-- Definition of a standard deck --/
def standardDeck : Deck :=
  { cards := Finset.univ,
    size := by sorry }

/-- The probability of drawing a heart on the first draw and the King of clubs on the second draw --/
def probability (d : Deck) : Rat :=
  (13 : Rat) / 52 * (1 : Rat) / 51

/-- The main theorem --/
theorem probability_heart_king_clubs (d : Deck) :
  d = standardDeck → probability d = 1 / 204 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_heart_king_clubs_l228_22883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fence_cost_per_foot_l228_22859

theorem fence_cost_per_foot (area : ℝ) (total_cost : ℝ) : 
  area = 49 → total_cost = 1624 → (total_cost / (4 * Real.sqrt area)) = 58 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fence_cost_per_foot_l228_22859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_five_l228_22858

-- Define the line 2x + y = 0
def line (x y : ℝ) : Prop := 2 * x + y = 0

-- Define the circle
def circle_set (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the points A and B
def point_A : ℝ × ℝ := (1, 3)
def point_B : ℝ × ℝ := (4, 2)

-- Theorem statement
theorem circle_radius_is_five :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    line center.1 center.2 ∧
    point_A ∈ circle_set center radius ∧
    point_B ∈ circle_set center radius ∧
    radius = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_five_l228_22858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_girl_age_l228_22896

/-- Given two girls with ages that differ by 1 year, one girl being 13 years old,
    and the sum of their ages being 27, prove that the other girl is 14 years old. -/
theorem other_girl_age (age1 age2 : ℤ) : 
  age1 = 13 →
  abs (age1 - age2) = 1 →
  age1 + age2 = 27 →
  age2 = 14 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_girl_age_l228_22896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_a_magnitude_l228_22861

def C : ℝ × ℝ := (1, -1)
def D (x : ℝ) : ℝ × ℝ := (2, x)

def vector_a (x : ℝ) : ℝ × ℝ := (x, 2)

def vector_CD (x : ℝ) : ℝ × ℝ := ((D x).1 - C.1, (D x).2 - C.2)

def opposite_direction (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k < 0 ∧ v = (k * w.1, k * w.2)

theorem vector_a_magnitude (x : ℝ) :
  opposite_direction (vector_a x) (vector_CD x) →
  Real.sqrt ((vector_a x).1^ 2 + (vector_a x).2 ^ 2) = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_a_magnitude_l228_22861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l228_22836

noncomputable def f (x : ℝ) : ℝ := (x + 3) / (x^2 + 4*x + 3)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < -3 ∨ (-3 < x ∧ x < -1) ∨ -1 < x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l228_22836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l228_22855

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2*a - 1)*x + 4*a else -x + 1

theorem decreasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Ici (1/6) ∩ Set.Iio (1/2) := by
  sorry

#check decreasing_f_implies_a_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l228_22855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l228_22837

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sqrt 3 * cos (2 * x - π / 3) - 2 * sin x * cos x

-- State the theorem
theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∃ T > 0, T = π ∧ ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ x ∈ Set.Icc (-π/4) (π/4), f x ≥ -1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l228_22837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l228_22833

noncomputable def f (x : ℝ) : ℝ := 1 / (x - 2)

theorem f_domain : Set ℝ = {x : ℝ | x ≠ 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l228_22833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_prime_ball_l228_22894

def balls : Finset ℕ := {1, 2, 3, 4, 5, 6, 8, 9}

def is_prime (n : ℕ) : Prop := Nat.Prime n

noncomputable def prime_balls : Finset ℕ := balls.filter (fun n => Nat.Prime n)

theorem probability_of_prime_ball :
  (prime_balls.card : ℚ) / (balls.card : ℚ) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_prime_ball_l228_22894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_polar_form_l228_22852

-- We don't need to redefine Complex as it's already defined in Mathlib
-- def Complex := ℂ

-- Define the cis function
noncomputable def cis (θ : ℝ) : ℂ := Complex.exp (θ * Complex.I)

-- Define the polar form of a complex number
structure PolarForm where
  r : ℝ
  θ : ℝ
  r_pos : r > 0
  θ_range : 0 ≤ θ ∧ θ < 2 * Real.pi

-- Define the problem statement
theorem complex_product_polar_form :
  ∃ (result : PolarForm),
    (4 * cis (25 * Real.pi / 180)) * (-3 * cis (48 * Real.pi / 180)) =
    result.r * cis result.θ ∧
    result.r = 12 ∧
    result.θ = 253 * Real.pi / 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_polar_form_l228_22852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_angle_l228_22814

theorem parallel_vectors_angle (α : ℝ) : 
  let a : Fin 2 → ℝ := ![3/2, Real.sin α]
  let b : Fin 2 → ℝ := ![Real.cos α, 1/3]
  (∃ (k : ℝ), a = k • b) → α = π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_angle_l228_22814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_start_behind_l228_22860

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  distance : ℝ

/-- Calculates the distance a runner needs to start behind to finish simultaneously -/
noncomputable def distanceBehind (totalDistance : ℝ) (runner1 runner2 : Runner) : ℝ :=
  totalDistance * (runner1.speed / runner2.speed - 1)

/-- Theorem stating the correct distance Luíza should start behind -/
theorem race_start_behind (totalDistance : ℝ) (ana luiza : Runner) 
  (h1 : totalDistance = 3000)
  (h2 : ana.distance = luiza.distance - 120)
  (h3 : luiza.distance = totalDistance) :
  distanceBehind totalDistance luiza ana = 125 := by
  sorry

#check race_start_behind

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_start_behind_l228_22860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_g_7_l228_22848

-- Define the functions s and g
noncomputable def s (x : ℝ) : ℝ := Real.sqrt (4 * x + 2)
noncomputable def g (x : ℝ) : ℝ := 7 - s x

-- State the theorem
theorem s_of_g_7 : s (g 7) = Real.sqrt (30 - 4 * Real.sqrt 30) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_g_7_l228_22848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_at_two_two_is_solution_l228_22856

noncomputable def f (x : ℝ) : ℝ := 12 / (x^2 + 6)
def g (x : ℝ) : ℝ := 4 - x

theorem intersection_at_two :
  ∃ (x : ℝ), f x = g x ∧ x = 2 := by
  -- We'll use 2 as our witness for the existential quantifier
  use 2
  -- Now we need to prove both parts of the conjunction
  constructor
  -- First, show that f 2 = g 2
  · -- Calculate f 2
    have h1 : f 2 = 12 / (2^2 + 6) := by rfl
    -- Calculate g 2
    have h2 : g 2 = 4 - 2 := by rfl
    -- Show they're equal
    sorry -- This step would require actual calculation
  -- Second part is trivial since we used 2
  · rfl

-- This theorem states that 2 is indeed a solution
theorem two_is_solution :
  f 2 = g 2 := by
  -- Calculate f 2
  have h1 : f 2 = 12 / (2^2 + 6) := by rfl
  -- Calculate g 2
  have h2 : g 2 = 4 - 2 := by rfl
  -- Show they're equal
  sorry -- This step would require actual calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_at_two_two_is_solution_l228_22856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_route_time_difference_l228_22821

/-- Calculates the time in minutes for a given distance and speed -/
noncomputable def time_in_minutes (distance : ℝ) (speed : ℝ) : ℝ :=
  (distance / speed) * 60

/-- Represents the two routes with their respective distances and speeds -/
structure Routes where
  route_a_distance : ℝ
  route_a_speed : ℝ
  route_b_distance1 : ℝ
  route_b_speed1 : ℝ
  route_b_distance2 : ℝ
  route_b_speed2 : ℝ

/-- The main theorem stating the time difference between routes -/
theorem route_time_difference (routes : Routes) 
    (h1 : routes.route_a_distance = 6)
    (h2 : routes.route_a_speed = 30)
    (h3 : routes.route_b_distance1 = 4.5)
    (h4 : routes.route_b_speed1 = 40)
    (h5 : routes.route_b_distance2 = 0.5)
    (h6 : routes.route_b_speed2 = 20) :
    time_in_minutes routes.route_a_distance routes.route_a_speed -
    (time_in_minutes routes.route_b_distance1 routes.route_b_speed1 +
     time_in_minutes routes.route_b_distance2 routes.route_b_speed2) = 3.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_route_time_difference_l228_22821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l228_22875

/-- Calculates the time taken for a train to cross a bridge -/
theorem train_bridge_crossing_time
  (train_length : ℝ)
  (train_speed_kmph : ℝ)
  (bridge_length : ℝ)
  (h1 : train_length = 110)
  (h2 : train_speed_kmph = 36)
  (h3 : bridge_length = 170) :
  (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l228_22875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_max_volume_sphere_max_volume_l228_22803

-- Define a cuboid
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : length > 0
  width_pos : width > 0
  height_pos : height > 0

-- Define a cube as a special case of cuboid
def Cube (s : ℝ) : Cuboid where
  length := s
  width := s
  height := s
  length_pos := by sorry
  width_pos := by sorry
  height_pos := by sorry

-- Define surface area for a cuboid
def surfaceArea (c : Cuboid) : ℝ :=
  2 * (c.length * c.width + c.length * c.height + c.width * c.height)

-- Define volume for a cuboid
def volume (c : Cuboid) : ℝ :=
  c.length * c.width * c.height

-- Define surface area for a sphere
noncomputable def sphereSurfaceArea (r : ℝ) : ℝ :=
  4 * Real.pi * r^2

-- Define volume for a sphere
noncomputable def sphereVolume (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

-- Theorem 1: Among all cuboids with a fixed surface area, the cube has the largest volume
theorem cube_max_volume (S : ℝ) (S_pos : S > 0) :
  ∀ c : Cuboid, surfaceArea c = S →
  volume c ≤ volume (Cube ((S / 6) ^ (1/2))) := by
  sorry

-- Theorem 2: Among all cuboids and spheres with a fixed surface area, the sphere has the largest volume
theorem sphere_max_volume (S : ℝ) (S_pos : S > 0) :
  (∀ c : Cuboid, surfaceArea c = S →
    volume c ≤ sphereVolume ((S / (4 * Real.pi)) ^ (1/2))) ∧
  (∀ r : ℝ, sphereSurfaceArea r = S →
    sphereVolume r ≤ sphereVolume ((S / (4 * Real.pi)) ^ (1/2))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_max_volume_sphere_max_volume_l228_22803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_square_sum_positive_l228_22841

theorem complex_square_sum_positive (Z₁ Z₂ Z₃ : ℂ) :
  (Z₁.re^2 + Z₁.im^2 + Z₂.re^2 + Z₂.im^2 : ℝ) > -(Z₃.re^2 + Z₃.im^2) →
  (Z₁.re^2 + Z₁.im^2 + Z₂.re^2 + Z₂.im^2 + Z₃.re^2 + Z₃.im^2 : ℝ) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_square_sum_positive_l228_22841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_consecutive_odd_members_l228_22870

/-- A sequence of natural numbers where each number after the first is obtained
    by adding the largest digit of the previous number to it. -/
def DigitAddSequence : Type :=
  ℕ → ℕ

/-- The largest digit in a natural number. -/
def max_digit (n : ℕ) : ℕ :=
  sorry

/-- The property that a number in the sequence is obtained by adding
    the largest digit of the previous number. -/
def IsValidSequence (s : DigitAddSequence) : Prop :=
  ∀ n : ℕ, n > 0 → ∃ d : ℕ, d < 10 ∧ d = max_digit (s n) ∧ s (n + 1) = s n + d

/-- The property that a subsequence of length k starting at index i consists of odd numbers. -/
def ConsecutiveOddSubsequence (s : DigitAddSequence) (i k : ℕ) : Prop :=
  ∀ j : ℕ, j < k → Odd (s (i + j))

/-- The main theorem: The maximum number of consecutive odd members in a valid sequence is 5. -/
theorem max_consecutive_odd_members (s : DigitAddSequence) (h : IsValidSequence s) :
  (∃ i k : ℕ, ConsecutiveOddSubsequence s i k) → k ≤ 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_consecutive_odd_members_l228_22870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_building_dimensions_l228_22892

/-- The width of the river -/
noncomputable def river_width : ℝ := 15

/-- The height of the building -/
noncomputable def building_height : ℝ := 15 * Real.sqrt 3

/-- The initial angle of view in radians -/
noncomputable def initial_angle : ℝ := Real.pi / 3

/-- The second angle of view in radians -/
noncomputable def second_angle : ℝ := Real.pi / 6

/-- The distance moved along the bank -/
def distance_moved : ℝ := 30

theorem river_building_dimensions :
  ∃ (x y : ℝ),
    x = river_width ∧
    y = building_height ∧
    y = x * Real.tan initial_angle ∧
    y = (x + distance_moved) * Real.tan second_angle :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_building_dimensions_l228_22892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_quadrilateral_l228_22847

/-- The area of a quadrilateral with vertices (x₁, y₁), (x₂, y₂), (x₃, y₃), (x₄, y₄) -/
noncomputable def quadrilateralArea (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) : ℝ :=
  (1/2) * |x₁*y₂ + x₂*y₃ + x₃*y₄ + x₄*y₁ - (y₁*x₂ + y₂*x₃ + y₃*x₄ + y₄*x₁)|

theorem area_of_specific_quadrilateral :
  quadrilateralArea 1 1 5 6 8 3 2 7 = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_quadrilateral_l228_22847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l228_22864

theorem problem_solution :
  ∀ (a b : ℝ),
    (a + 2 * b = 9) →
    ((|9 - 2 * b| + |a + 1| < 3 → a ∈ Set.Ioo (-2 : ℝ) 1) ∧
    (a > 0 → b > 0 → ∀ z : ℝ, z = a * b^2 → z ≤ 27 ∧ ∃ a₀ b₀ : ℝ, a₀ + 2 * b₀ = 9 ∧ a₀ > 0 ∧ b₀ > 0 ∧ a₀ * b₀^2 = 27)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l228_22864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_prime_factor_sum_l228_22811

def least_prime_factor (n : ℕ) : ℕ :=
  Nat.minFac n

theorem least_prime_factor_sum (a b : ℕ) 
  (ha : least_prime_factor a = 3)
  (hb : least_prime_factor b = 7) :
  least_prime_factor (a + b) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_prime_factor_sum_l228_22811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_leq_x_squared_min_c_for_f_leq_2x_plus_c_l228_22863

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 + 2 * Real.log x

-- Theorem 1: f(x) ≤ x² for all x > 0
theorem f_leq_x_squared {x : ℝ} (hx : x > 0) : f x ≤ x^2 := by sorry

-- Theorem 2: The minimum value of c such that f(x) ≤ 2x + c for all x > 0 is -1
theorem min_c_for_f_leq_2x_plus_c : 
  (∃ c : ℝ, ∀ x > 0, f x ≤ 2*x + c) ∧ 
  (∀ c < -1, ∃ x > 0, f x > 2*x + c) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_leq_x_squared_min_c_for_f_leq_2x_plus_c_l228_22863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_relationships_l228_22879

-- Define the sets A, B, and C
def A : Set ℕ := {0, 1}
def B : Set ℕ := {x ∈ A | x > 0}
def C : Set (Set ℕ) := {x | x ⊆ A}

-- State the theorem
theorem set_relationships :
  (B ⊂ A) ∧ (A ∈ C) ∧ (B ∈ C) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_relationships_l228_22879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_arccos_inequality_l228_22888

theorem arcsin_arccos_inequality (x : ℝ) : 
  Real.arcsin ((5 / (2 * Real.pi)) * Real.arccos x) > Real.arccos ((10 / (3 * Real.pi)) * Real.arcsin x) ↔ 
  (x ∈ Set.Icc (Real.cos (2 * Real.pi / 5)) (Real.cos (8 * Real.pi / 25)) ∪ 
   Set.Ioo (Real.cos (8 * Real.pi / 25)) (Real.cos (Real.pi / 5))) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_arccos_inequality_l228_22888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_max_product_a7_a14_l228_22825

/-- An arithmetic sequence -/
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def arithmetic_sum (a₁ d : ℝ) (n : ℕ) : ℝ := (n : ℝ) * (2 * a₁ + (n - 1 : ℝ) * d) / 2

theorem no_max_product_a7_a14 :
  ∀ a₁ d : ℝ, arithmetic_sum a₁ d 20 = 100 →
  ¬∃ M : ℝ, ∀ a₁' d' : ℝ, arithmetic_sum a₁' d' 20 = 100 →
    arithmetic_sequence a₁' d' 7 * arithmetic_sequence a₁' d' 14 ≤ M :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_max_product_a7_a14_l228_22825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_range_for_cubic_sum_one_l228_22887

theorem sum_range_for_cubic_sum_one (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a^3 + b^3 = 1) : 
  1 < a + b ∧ a + b ≤ Real.rpow 4 (1/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_range_for_cubic_sum_one_l228_22887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l228_22840

/-- The eccentricity of an ellipse with equation x²/a² + y²/b² = 1 is √(1 - b²/a²) -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

/-- The eccentricity of the ellipse x²/9 + y²/5 = 1 is 2/3 -/
theorem ellipse_eccentricity : eccentricity 3 (Real.sqrt 5) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l228_22840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l228_22829

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 / 16 - y^2 / 4 = 1

-- Define the foci F1 and F2
noncomputable def F1 : ℝ × ℝ := sorry
noncomputable def F2 : ℝ × ℝ := sorry

-- Define a point P on the hyperbola
noncomputable def P : ℝ × ℝ := sorry

-- Define the distance function
noncomputable def my_dist (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem hyperbola_properties :
  hyperbola_C P.1 P.2 →
  my_dist P F1 = 4 →
  (∀ (x y : ℝ), y = (1/2) * x ∨ y = -(1/2) * x) ∧
  my_dist P F2 = 12 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l228_22829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_2025_arrangements_l228_22808

def digits : List Nat := [2, 0, 2, 5]

def is_valid_arrangement (arr : List Nat) : Bool :=
  arr.length = 4 && arr.head? ≠ some 0 && arr.toFinset ⊆ digits.toFinset

def count_valid_arrangements : Nat :=
  (digits.permutations.filter is_valid_arrangement).length

theorem count_2025_arrangements : count_valid_arrangements = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_2025_arrangements_l228_22808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_correct_l228_22842

-- Define the propositions
def prop1 : Prop := ∀ p q : Prop, ¬(p ∧ q) → (¬p ∧ ¬q)

def prop2 : Prop := 
  (¬(∀ a b : ℝ, a > b → (2 : ℝ)^a > (2 : ℝ)^b - 1)) ↔ (∀ a b : ℝ, a ≤ b → (2 : ℝ)^a ≤ (2 : ℝ)^b - 1)

def prop3 : Prop := 
  (¬(∀ x : ℝ, x^2 + 1 ≥ 1)) ↔ (∀ x : ℝ, x^2 + 1 < 1)

def prop4 : Prop := 
  ∀ A B : ℝ, (A > B ↔ Real.sin A > Real.sin B)

-- Theorem statement
theorem exactly_two_correct : 
  (¬prop1 ∧ prop2 ∧ ¬prop3 ∧ prop4) ∨
  (¬prop1 ∧ prop2 ∧ prop3 ∧ ¬prop4) ∨
  (¬prop1 ∧ ¬prop2 ∧ prop3 ∧ prop4) ∨
  (prop1 ∧ prop2 ∧ ¬prop3 ∧ ¬prop4) ∨
  (prop1 ∧ ¬prop2 ∧ prop3 ∧ ¬prop4) ∨
  (prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ prop4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_correct_l228_22842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_nonempty_subsets_of_two_element_set_l228_22835

theorem count_nonempty_subsets_of_two_element_set : 
  (∃ (f : Fintype {M : Set ℕ | ∅ ⊂ M ∧ M ⊆ {1, 2}}), 
    Finset.card (Finset.univ : Finset {M : Set ℕ | ∅ ⊂ M ∧ M ⊆ {1, 2}}) = 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_nonempty_subsets_of_two_element_set_l228_22835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_quadratic_l228_22816

theorem max_value_quadratic (f : ℝ → ℝ) (a : ℝ) :
  (f = λ x ↦ -x^2 - 2*x + 3) →
  (∀ x ∈ Set.Icc a 2, f x ≤ 15/4) →
  (∃ x ∈ Set.Icc a 2, f x = 15/4) →
  a = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_quadratic_l228_22816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_inequality_l228_22819

/-- Given a function f(x) = x ln x - ax³ - x, where a is a real number,
    if x₁ and x₂ are two distinct extreme points of f(x) with x₁ < x₂,
    then 3 ln x₁ + ln x₂ > 1 -/
theorem extreme_points_inequality (a : ℝ) (x₁ x₂ : ℝ) : 
  let f := fun x : ℝ => x * Real.log x - a * x^3 - x
  x₁ < x₂ →
  (∀ x, x ≠ x₁ → x ≠ x₂ → (deriv f x) ≠ 0) →
  deriv f x₁ = 0 →
  deriv f x₂ = 0 →
  3 * Real.log x₁ + Real.log x₂ > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_inequality_l228_22819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l228_22839

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^(1 / (x^2 + 1))

-- State the theorem about the range of f
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) → 1 < y ∧ y ≤ 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l228_22839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_quadrilateral_diagonal_l228_22827

/-- A quadrilateral inscribed in a circle -/
structure InscribedQuadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  center : ℝ × ℝ
  radius : ℝ

/-- Angle between three points -/
noncomputable def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem inscribed_quadrilateral_diagonal (ABCD : InscribedQuadrilateral) 
  (h1 : angle ABCD.B ABCD.A ABCD.C = 50 * π / 180)
  (h2 : angle ABCD.A ABCD.D ABCD.B = 55 * π / 180)
  (h3 : distance ABCD.A ABCD.D = 5)
  (h4 : distance ABCD.B ABCD.C = 7) :
  ∃ ε > 0, |distance ABCD.A ABCD.C - 6.4| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_quadrilateral_diagonal_l228_22827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cardioid_arc_length_is_16_l228_22806

open Real MeasureTheory

/-- The arc length of the cardioid ρ = 2(1 + cos φ) for 0 ≤ φ ≤ 2π -/
noncomputable def cardioid_arc_length : ℝ :=
  ∫ φ in Set.Icc 0 (2 * π), Real.sqrt ((2 * Real.sin φ)^2 + (2 * (1 + Real.cos φ))^2)

/-- Theorem: The arc length of the cardioid ρ = 2(1 + cos φ) for 0 ≤ φ ≤ 2π is equal to 16 -/
theorem cardioid_arc_length_is_16 : cardioid_arc_length = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cardioid_arc_length_is_16_l228_22806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_proof_l228_22862

/-- Arithmetic sequence {a_n} -/
noncomputable def a (n : ℕ) : ℝ := 2 * n + 2

/-- Sum of first n terms of {a_n} -/
noncomputable def S (n : ℕ) : ℝ := n * (a 1 + a n) / 2

/-- Sequence {b_n} -/
noncomputable def b (n : ℕ) : ℝ := 1 / (S n + 2)

/-- Sum of first n terms of {b_n} -/
noncomputable def T (n : ℕ) : ℝ := n / (2 * n + 4)

theorem arithmetic_sequence_proof (n : ℕ) :
  (a 1 = 4) ∧
  (∃ r : ℝ, a 2 * (2 * a 7 - 8) = (a 4 + 2)^2 ∧ r > 0) ∧
  (∀ k : ℕ, a (k + 1) - a k = a 2 - a 1) →
  (∀ k : ℕ, a k = 2 * k + 2) ∧
  (T n = n / (2 * n + 4)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_proof_l228_22862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carla_witchcraft_days_l228_22820

/-- The length multiplier for a given day k -/
def dayMultiplier (k : ℕ) : ℚ :=
  if k = 0 then 5/3 else (k + 3 : ℚ) / (k + 2)

/-- The total length multiplier after n days -/
def totalMultiplier (n : ℕ) : ℚ :=
  (List.range (n + 1)).foldl (λ acc k => acc * dayMultiplier k) 1

theorem carla_witchcraft_days (n : ℕ) :
  totalMultiplier n = 200 → n = 357 := by
  sorry

#eval decide (totalMultiplier 357 = 200)  -- This should return true if the theorem is correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carla_witchcraft_days_l228_22820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_sum_three_angles_l228_22884

theorem sin_squared_sum_three_angles (α : ℝ) : 
  Real.sin α ^ 2 + Real.sin (α + Real.pi / 3) ^ 2 + Real.sin (α + 2 * Real.pi / 3) ^ 2 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_sum_three_angles_l228_22884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_implies_c_value_l228_22853

/-- Given two vectors in R², compute the projection of the first onto the second -/
noncomputable def proj (v u : Fin 2 → ℝ) : Fin 2 → ℝ :=
  (((v 0) * (u 0) + (v 1) * (u 1)) / ((u 0)^2 + (u 1)^2)) • u

theorem projection_implies_c_value (c : ℝ) :
  let v : Fin 2 → ℝ := ![- 6, c]
  let u : Fin 2 → ℝ := ![3, 2]
  proj v u = (-20 / 13) • u →
  c = -1 := by
  intros
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_implies_c_value_l228_22853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l228_22880

/-- A point in the 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The parabola y^2 = 4x -/
def IsOnParabola (p : Point2D) : Prop :=
  p.y^2 = 4 * p.x

/-- The focus of the parabola y^2 = 4x -/
def FocusPoint : Point2D :=
  { x := 1, y := 0 }

/-- Distance between two points -/
noncomputable def Distance (p1 p2 : Point2D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Distance from a point to the y-axis -/
def DistanceToYAxis (p : Point2D) : ℝ :=
  |p.x|

theorem parabola_distance_theorem (M : Point2D) 
  (h1 : IsOnParabola M) 
  (h2 : Distance M FocusPoint = 10) : 
  DistanceToYAxis M = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l228_22880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l228_22849

noncomputable def f (x : ℝ) := Real.sqrt (x^2 - 5*x + 6)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≤ 2 ∨ x ≥ 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l228_22849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_two_l228_22886

/-- Hyperbola struct representing the equation x^2/a^2 - y^2/b^2 = 1 --/
structure Hyperbola where
  a : ℝ
  b : ℝ
  ha : a > 0
  hb : b > 0

/-- Point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of eccentricity for a hyperbola --/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + h.b^2 / h.a^2)

/-- The right focus of the hyperbola --/
noncomputable def right_focus (h : Hyperbola) : Point :=
  ⟨Real.sqrt (h.a^2 + h.b^2), 0⟩

/-- Theorem stating that under given conditions, the eccentricity of the hyperbola is 2 --/
theorem hyperbola_eccentricity_is_two (h : Hyperbola) 
  (P : Point) 
  (F₁ : Point) 
  (hP : P.y = (h.b / h.a) * P.x) -- P is on the asymptote
  (hPerp : (P.y - (right_focus h).y) * (P.x - (right_focus h).x) = 
           -(h.a / h.b) * ((P.x - (right_focus h).x)^2 + (P.y - (right_focus h).y)^2)) 
           -- line PF₂ is perpendicular to asymptote
  (hDiff : (P.x - F₁.x)^2 + (P.y - F₁.y)^2 - 
           ((P.x - (right_focus h).x)^2 + (P.y - (right_focus h).y)^2) = 
           (Real.sqrt (h.a^2 + h.b^2))^2) 
           -- |PF₁|^2 - |PF₂|^2 = c^2
  : eccentricity h = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_two_l228_22886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_in_list_l228_22872

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def six_digit_numbers : List ℕ := [301200, 301201, 301202, 301203, 301204, 301205, 301206, 301207, 301208, 301209]

theorem unique_prime_in_list : ∃! n, n ∈ six_digit_numbers ∧ is_prime n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_in_list_l228_22872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l228_22867

noncomputable section

/-- The slope angle of the line 6x - 2y - 5 = 0 -/
def α : ℝ := Real.arctan 3

/-- The expression to be evaluated -/
def expression (θ : ℝ) : ℝ :=
  (Real.sin (Real.pi - θ) + Real.cos (-θ)) / (Real.sin (-θ) - Real.cos (Real.pi + θ))

theorem expression_value :
  expression α = -2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l228_22867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l228_22817

/-- Given two vectors a and b in a real inner product space, 
    with |a| = 1, |b| = 2, and a · b = -√3, 
    prove that the angle between a and b is 5π/6. -/
theorem angle_between_vectors (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b : V) (h1 : ‖a‖ = 1) (h2 : ‖b‖ = 2) (h3 : inner a b = -Real.sqrt 3) :
  Real.arccos (inner a b / (‖a‖ * ‖b‖)) = 5 * Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l228_22817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_identity_l228_22834

/-- Given two parallel vectors a and b, prove that 2 + sin(θ)cos(θ) - cos²(θ) = 5/2 -/
theorem parallel_vectors_identity (θ : ℝ) :
  let a : Fin 2 → ℝ := ![3, 1]
  let b : Fin 2 → ℝ := ![Real.sin θ, Real.cos θ]
  (∃ (k : ℝ), ∀ i, b i = k * a i) →
  2 + Real.sin θ * Real.cos θ - Real.cos θ ^ 2 = 5/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_identity_l228_22834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_score_difference_l228_22843

noncomputable def score_distribution : List (ℚ × ℚ) := [
  (60, 15/100),
  (75, 20/100),
  (85, 25/100),
  (90, 25/100),
  (100, 15/100)
]

noncomputable def mean (dist : List (ℚ × ℚ)) : ℚ :=
  (dist.map (λ (s, p) => s * p)).sum

noncomputable def median (dist : List (ℚ × ℚ)) : ℚ :=
  let cumulative := dist.scanl (λ (_, acc) (s, p) => (s, acc + p)) (0, 0)
  let median_point := cumulative.find? (λ (_, cum) => cum ≥ 1/2)
  match median_point with
  | some (s, _) => s
  | none => 0

theorem exam_score_difference :
  mean score_distribution - median score_distribution = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_score_difference_l228_22843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_slope_product_correct_a_exists_l228_22857

-- Define the slopes of the lines
noncomputable def slope_l1 (a : ℝ) : ℝ := a
noncomputable def slope_l2 (a : ℝ) : ℝ := (1 - 2*a) / a

-- State the theorem
theorem perpendicular_lines_slope_product (a : ℝ) (h : a ≠ 0) :
  slope_l1 a * slope_l2 a = -1 := by
  -- The proof is omitted for now
  sorry

-- Define the possible values of a
def possible_a : Set ℝ := {0, 1, 2}

-- State the theorem for the correct answer
theorem correct_a_exists :
  ∃ a ∈ possible_a, a ≠ 0 ∧ slope_l1 a * slope_l2 a = -1 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_slope_product_correct_a_exists_l228_22857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_intersection_or_disjoint_l228_22868

/-- A line segment represented by its endpoints -/
structure Segment where
  left : ℝ
  right : ℝ
  h : left ≤ right

/-- Two segments intersect if they share at least one point -/
def intersect (s1 s2 : Segment) : Prop :=
  ¬(s1.right < s2.left ∨ s2.right < s1.left)

/-- A set of segments is pairwise disjoint if no two segments intersect -/
def pairwise_disjoint (S : Set Segment) : Prop :=
  ∀ s1 s2, s1 ∈ S → s2 ∈ S → s1 ≠ s2 → ¬(intersect s1 s2)

/-- The main theorem -/
theorem segment_intersection_or_disjoint (segments : Finset Segment) 
    (h_card : segments.card = 50) :
    (∃ S : Finset Segment, S ⊆ segments ∧ S.card = 8 ∧ 
      ∃ p : ℝ, ∀ s ∈ S, s.left ≤ p ∧ p ≤ s.right) ∨
    (∃ S : Finset Segment, S ⊆ segments ∧ S.card = 8 ∧ pairwise_disjoint S) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_intersection_or_disjoint_l228_22868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_false_l228_22804

-- Define the four propositions
def proposition1 : Prop := 
  ∀ α β : Real, (Real.cos α = Real.sin β) ∧ (Real.sin α = Real.cos β) → α = β

def proposition2 : Prop := 
  ∀ a b c : Real, a > 0 ∧ b > 0 ∧ c > 0 → 
    ∃ x y z : Real, x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    x + y + z = 1 ∧ a*x = b*y ∧ b*y = c*z

def proposition3 : Prop := 
  ∀ r θ₁ θ₂ : Real, r > 0 → 
    θ₁ = θ₂ → r * θ₁ = r * θ₂

def proposition4 : Prop := 
  ∀ r c₁ c₂ θ₁ θ₂ : Real, r > 0 → 
    2 * r * Real.sin (c₁/2) = 2 * r * Real.sin (c₂/2) → θ₁ = θ₂

-- Theorem stating that all propositions are false
theorem all_propositions_false : 
  ¬proposition1 ∧ ¬proposition2 ∧ ¬proposition3 ∧ ¬proposition4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_false_l228_22804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decagon_diagonals_l228_22800

/-- A polygon is a closed plane figure with straight sides. -/
structure Polygon where
  sides : ℕ
  sides_pos : sides > 0

/-- A regular polygon is a polygon with all sides and angles equal. -/
structure RegularPolygon extends Polygon

/-- A diagonal of a polygon is a line segment that connects two non-adjacent vertices. -/
def diagonal (p : Polygon) := Unit

/-- The number of diagonals in a polygon. -/
def num_diagonals (p : Polygon) : ℕ := (p.sides * (p.sides - 3)) / 2

/-- A decagon is a polygon with 10 sides. -/
def Decagon : Polygon := ⟨10, by norm_num⟩

theorem decagon_diagonals :
  num_diagonals Decagon = 35 := by
  -- Unfold the definitions and simplify
  unfold num_diagonals Decagon
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decagon_diagonals_l228_22800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_distance_l228_22801

noncomputable section

/-- An ellipse with equation x²/2 + y² = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | p.1^2 / 2 + p.2^2 = 1}

/-- The focus of the ellipse -/
def Focus : ℝ × ℝ := (1, 0)

/-- The eccentricity of the ellipse -/
def Eccentricity : ℝ := Real.sqrt 2 / 2

/-- A point on the major axis of the ellipse -/
def P (t : ℝ) : ℝ × ℝ := (t, 0)

/-- The line passing through P with slope 1 -/
def Line (t : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 = p.1 - t}

/-- The intersection points of the line and the ellipse -/
def IntersectionPoints (t : ℝ) : Set (ℝ × ℝ) :=
  Ellipse ∩ Line t

/-- The squared distance between two points -/
def SquaredDistance (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

/-- The theorem to be proved -/
theorem ellipse_max_distance :
  ∃ (max : ℝ), max = 8/3 ∧
  ∀ (t : ℝ) (A B : ℝ × ℝ),
    A ∈ IntersectionPoints t →
    B ∈ IntersectionPoints t →
    A ≠ B →
    SquaredDistance (P t) A + SquaredDistance (P t) B ≤ max := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_distance_l228_22801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l228_22869

-- Definitions used in the statement
structure Quadrilateral where
  -- Add necessary fields (placeholder)
  dummy : Unit

def DiagonalsEqual (q : Quadrilateral) : Prop :=
  sorry

def IsRectangle (q : Quadrilateral) : Prop :=
  sorry

theorem problem_statement :
  -- Proposition 1
  (∀ k > 0, ∃ x : ℝ, x^2 + 2*x - k = 0) ∧
  -- Proposition 2
  (∀ x y : ℝ, x + y ≠ 8 → x ≠ 2 ∨ y ≠ 6) ∧
  -- Proposition 3 (inverse of "The diagonals of a rectangle are equal")
  (¬ ∀ q : Quadrilateral, DiagonalsEqual q → IsRectangle q) ∧
  -- Proposition 4 (negation of "If xy = 0, then at least one of x or y is 0")
  (∃ x y : ℝ, x * y = 0 ∧ x ≠ 0 ∧ y ≠ 0) :=
by
  constructor
  · -- Proof for Proposition 1
    sorry
  constructor
  · -- Proof for Proposition 2
    sorry
  constructor
  · -- Proof for Proposition 3
    sorry
  · -- Proof for Proposition 4
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l228_22869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_denomination_power_of_two_main_result_l228_22846

def double_factorial (n : ℕ) : ℕ :=
  Finset.prod (Finset.range (n + 1) \ Finset.range (n % 2)) (fun i => n - 2 * i)

def sum_fraction (n : ℕ) : ℚ :=
  (Finset.range n).sum (fun i => (double_factorial (2*i - 1) : ℚ) / (double_factorial (2*i)))

theorem denomination_power_of_two (n : ℕ) :
  ∃ (a b : ℕ), (sum_fraction 2010).den = 2^a * b ∧ b % 2 = 1 :=
by sorry

theorem main_result : 
  ∃ (a b : ℕ), (sum_fraction 2010).den = 2^a * b ∧ b % 2 = 1 ∧ a = 3992 ∧ b = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_denomination_power_of_two_main_result_l228_22846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_square_circle_l228_22809

/-- The probability of a randomly selected point being within two units of the origin
    in a square region with vertices at (±4, ±4) -/
noncomputable def probability_within_circle (square_side : ℝ) (circle_radius : ℝ) : ℝ :=
  (Real.pi * circle_radius^2) / (square_side^2)

theorem probability_in_square_circle : probability_within_circle 8 2 = Real.pi / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_square_circle_l228_22809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_integer_for_diophantine_equation_l228_22845

theorem smallest_positive_integer_for_diophantine_equation
  (a b : ℕ+) (h : Nat.Coprime a.val b.val) :
  ∃ (c₀ : ℕ+),
    (∀ (c : ℕ+), c ≥ c₀ →
      ∃ (x y : ℕ), a * x + b * y = c) ∧
    (∀ (c' : ℕ+), (∀ (c : ℕ+), c ≥ c' →
      ∃ (x y : ℕ), a * x + b * y = c) →
      c' ≥ c₀) ∧
    c₀ = a * b - a - b + 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_integer_for_diophantine_equation_l228_22845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_emu_pens_l228_22830

/-- The number of pens John has for his emus -/
def num_pens : ℕ := 4

/-- The number of emus in each pen -/
def emus_per_pen : ℕ := 6

/-- The fraction of emus that are female -/
def female_ratio : ℚ := 1/2

/-- The number of eggs laid by each female emu per day -/
def eggs_per_female_per_day : ℕ := 1

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total number of eggs collected in a week -/
def total_eggs_per_week : ℕ := 84

theorem john_emu_pens :
  num_pens * (emus_per_pen * (female_ratio.num / female_ratio.den)) * eggs_per_female_per_day * days_in_week = total_eggs_per_week :=
by
  -- Convert rational to float for calculation
  have h1 : (emus_per_pen * (female_ratio.num / female_ratio.den)) = 3 := by sorry
  -- Perform the multiplication
  have h2 : num_pens * 3 * eggs_per_female_per_day * days_in_week = 84 := by sorry
  -- Use the above facts to prove the theorem
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_emu_pens_l228_22830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flag_colors_theorem_l228_22805

theorem flag_colors_theorem (n : ℕ) (colors : Finset (Finset ℕ)) : 
  Odd n → -- number of flags is odd
  n = (colors.biUnion id).card → -- total number of flags
  (∀ c ∈ colors, c.card > 0) → -- each color is used at least once
  colors.card ≥ 2 → -- at least 2 distinct colors are used
  (∀ c ∈ colors, ∃ d : ℕ, c.card * d = n) → -- flags of the same color form a regular polygon
  ∃ (m : ℕ) (cs : Finset (Finset ℕ)), cs ⊆ colors ∧ cs.card ≥ 3 ∧ ∀ c ∈ cs, c.card = m :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flag_colors_theorem_l228_22805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_relations_l228_22866

-- Define the concept of lines
structure Line : Type where
  -- We'll use a placeholder definition for now
  dummy : Unit

-- Define the relation of two lines being parallel
def parallel (l1 l2 : Line) : Prop :=
  sorry

-- Define the relation of two lines intersecting
def intersecting (l1 l2 : Line) : Prop :=
  sorry

-- Define the relation of two lines being skew
def skew (l1 l2 : Line) : Prop :=
  sorry

-- Define the relation of two lines being perpendicular
def perpendicular (l1 l2 : Line) : Prop :=
  sorry

-- Define the concept of a plane
structure Plane : Type where
  -- We'll use a placeholder definition for now
  dummy : Unit

-- Define the concept of a point
structure Point : Type where
  -- We'll use a placeholder definition for now
  dummy : Unit

-- Define the relation of a point being on a line
def on_line (p : Point) (l : Line) : Prop :=
  sorry

-- Define the relation of a line being in a plane
def in_plane (l : Line) (p : Plane) : Prop :=
  sorry

-- Theorem stating that propositions 3 and 4 are correct, while 1 and 2 are not necessarily true
theorem line_relations :
  (∃ l1 l2 : Line, ¬(∃ p : Point, on_line p l1 ∧ on_line p l2) ∧ ¬(parallel l1 l2)) ∧ 
  (∃ l1 l2 : Line, perpendicular l1 l2 ∧ skew l1 l2) ∧
  (∀ l1 l2 : Line, ¬(parallel l1 l2) ∧ ¬(intersecting l1 l2) → skew l1 l2) ∧
  (∀ l1 l2 : Line, (¬∃ p : Plane, in_plane l1 p ∧ in_plane l2 p) → skew l1 l2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_relations_l228_22866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_noetherian_square_id_group_l228_22854

/-- A group where all elements squared equal the identity element -/
class SquareIdentityGroup (G : Type) extends Group G where
  square_id : ∀ a : G, a * a = 1

/-- Definition of a Noetherian group -/
def IsNoetherianGroup (G : Type) [Group G] : Prop :=
  ∀ (S : Set (Subgroup G)), ∃ (M : Subgroup G), M ∈ S ∧ ∀ (H : Subgroup G), H ∈ S → H ≤ M → H = M

/-- There exists a group with square identity property that is not Noetherian -/
theorem exists_non_noetherian_square_id_group :
  ∃ (G : Type) (inst : SquareIdentityGroup G), ¬IsNoetherianGroup G := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_noetherian_square_id_group_l228_22854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_continuity_points_l228_22890

-- Define the piecewise function
noncomputable def f (n : ℝ) (x : ℝ) : ℝ :=
  if x < n then x^2 + 3 else 2*x + 7

-- Theorem statement
theorem sum_of_continuity_points :
  ∃ n₁ n₂ : ℝ, n₁ ≠ n₂ ∧ 
    ContinuousAt (f n₁) n₁ ∧ 
    ContinuousAt (f n₂) n₂ ∧ 
    n₁ + n₂ = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_continuity_points_l228_22890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_w_fourth_power_l228_22889

noncomputable def w : ℂ := (-1 + Complex.I * Real.sqrt 3) / 2

theorem w_fourth_power : w^4 = w := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_w_fourth_power_l228_22889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_l228_22832

open Real

/-- The function f(x) = m(2ln x - x) + 1/x^2 - 1/x --/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * (2 * log x - x) + 1 / x^2 - 1 / x

/-- The derivative of f(x) --/
noncomputable def f_deriv (m : ℝ) (x : ℝ) : ℝ := (x - 2) * (-m * x^2 + 1) / x^3

theorem f_has_two_zeros (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 0 < x₁ ∧ 0 < x₂ ∧ f m x₁ = 0 ∧ f m x₂ = 0 ∧
    ∀ x : ℝ, 0 < x → f m x = 0 → x = x₁ ∨ x = x₂) ↔
  1 / (8 * (log 2 - 1)) < m ∧ m < 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_l228_22832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l228_22871

/-- The eccentricity of a hyperbola with given conditions -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1) ∧ 
  (∃ (x y : ℝ), (x - 2)^2 + y^2 = 2) ∧
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    ((x₁ - 2)^2 + y₁^2 = 2 ∧ (x₂ - 2)^2 + y₂^2 = 2) ∧
    ((y₁ = (b/a) * x₁ ∨ y₁ = -(b/a) * x₁) ∧ (y₂ = (b/a) * x₂ ∨ y₂ = -(b/a) * x₂)) ∧
    ((x₁ - x₂)^2 + (y₁ - y₂)^2 = 4)) →
  Real.sqrt (1 + (b/a)^2) = 2 * Real.sqrt 3 / 3 := by
  sorry

#check hyperbola_eccentricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l228_22871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_angle_trig_identity_l228_22876

-- Problem 1
theorem tan_half_angle (α : Real) (h1 : α ∈ Set.Icc π (3*π/2)) (h2 : Real.sin α = -5/13) :
  Real.tan (α/2) = -5 := by sorry

-- Problem 2
theorem trig_identity (α : Real) (h : Real.tan α = 2) :
  (Real.sin (π - α))^2 + 2*(Real.sin (3*π/2 + α))*(Real.cos (π/2 + α)) = 8/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_angle_trig_identity_l228_22876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_is_one_l228_22822

/-- Represents a face of the cube --/
structure Face where
  value : Fin 6
  h_value : value.val + 1 ≥ 1 ∧ value.val + 1 ≤ 6

/-- Represents the cube with 6 faces --/
structure Cube where
  faces : Fin 6 → Face
  h_sum : ∀ i j, i ≠ j → (faces i).value.val + 1 + (faces j).value.val + 1 = 7

/-- Represents the net of the cube --/
structure Net where
  horizontal : Fin 4 → Fin 6
  vertical : Fin 4 → Fin 6
  h_from_cube : ∃ (c : Cube), 
    (horizontal 0 = (c.faces 0).value) ∧
    (horizontal 3 = (c.faces 5).value) ∧
    (vertical 0 = (c.faces 1).value) ∧
    (vertical 2 = (c.faces 4).value)

/-- The main theorem --/
theorem smallest_difference_is_one (n : Net) : 
  ∃ (h v : Nat), h = (n.horizontal 0).val + 1 + (n.horizontal 1).val + 1 + (n.horizontal 2).val + 1 + (n.horizontal 3).val + 1 ∧
                 v = (n.vertical 0).val + 1 + (n.vertical 1).val + 1 + (n.vertical 2).val + 1 + (n.vertical 3).val + 1 ∧
                 (h : Int) - (v : Int) ≥ 1 ∧
                 (∀ (h' v' : Nat), 
                    h' = (n.horizontal 0).val + 1 + (n.horizontal 1).val + 1 + (n.horizontal 2).val + 1 + (n.horizontal 3).val + 1 ∧
                    v' = (n.vertical 0).val + 1 + (n.vertical 1).val + 1 + (n.vertical 2).val + 1 + (n.vertical 3).val + 1 →
                    (h' : Int) - (v' : Int) ≥ 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_is_one_l228_22822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_day_all_snacks_eaten_l228_22897

/-- Represents a day in March --/
def MarchDay := Fin 31

/-- Feeding schedule for each snack type --/
def catStickSchedule : Nat := 1
def eggYolkSchedule : Nat := 2
def creamSchedule : Nat := 3

/-- The given dates in the problem --/
def march23 : MarchDay := ⟨23, by norm_num⟩
def march25 : MarchDay := ⟨25, by norm_num⟩

/-- Function to check if a snack is eaten on a given day --/
def isSnackEaten (schedule : Nat) (startDay : MarchDay) (day : MarchDay) : Prop :=
  (day.val - startDay.val) % schedule = 0

/-- The theorem to prove --/
theorem first_day_all_snacks_eaten : 
  ∃ (day : MarchDay), 
    day.val ≥ march23.val ∧
    isSnackEaten catStickSchedule march23 day ∧
    isSnackEaten eggYolkSchedule march25 day ∧
    isSnackEaten creamSchedule march23 day ∧
    ∀ (earlier_day : MarchDay), 
      earlier_day.val ≥ march23.val ∧ earlier_day.val < day.val →
        ¬(isSnackEaten catStickSchedule march23 earlier_day ∧
          isSnackEaten eggYolkSchedule march25 earlier_day ∧
          isSnackEaten creamSchedule march23 earlier_day) ∧
    day.val = 29 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_day_all_snacks_eaten_l228_22897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_in_sector_l228_22812

theorem inscribed_circle_in_sector (r R a : ℝ) (hr : r > 0) (hR : R > 0) (ha : a > 0) :
  (1 : ℝ) / r = 1 / R + 1 / a :=
by
  -- We'll outline the proof structure here
  -- Step 1: Set up the geometric configuration
  -- Step 2: Identify similar triangles
  -- Step 3: Set up the ratio of corresponding sides
  -- Step 4: Manipulate the equation to reach the desired form
  sorry -- This is a placeholder for the actual proof

-- You can add helper lemmas or additional definitions here if needed


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_in_sector_l228_22812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sum_division_l228_22865

theorem factorial_sum_division (n : ℕ) : 
  Nat.factorial (n + 1) + Nat.factorial (n + 2) = 80 * Nat.factorial n :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sum_division_l228_22865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_safe_to_drive_time_l228_22899

-- Define the initial alcohol content and the safe driving limit
def initial_alcohol : ℝ := 1
def safe_limit : ℝ := 0.2

-- Define the rate of decrease per hour
def decrease_rate : ℝ := 0.7

-- Function to calculate alcohol content after x hours
noncomputable def alcohol_content (x : ℝ) : ℝ := initial_alcohol * decrease_rate^x

-- Theorem stating that it takes at least 5 hours to reach safe driving limit
theorem safe_to_drive_time :
  ∀ x : ℝ, x < 5 → alcohol_content x > safe_limit := by
  sorry

#check safe_to_drive_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_safe_to_drive_time_l228_22899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_of_M_l228_22818

def A : Finset ℕ := {1, 2, 3}
def B : Finset ℕ := {4, 5}
def M : Finset ℕ := Finset.image (λ (p : ℕ × ℕ) => p.1 + p.2) (A.product B)

theorem cardinality_of_M : Finset.card M = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_of_M_l228_22818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_sum_power_of_two_l228_22878

theorem consecutive_sum_power_of_two (n : ℕ) :
  (∃ k : ℕ, n = 2^k) ↔ ¬(∃ a b : ℕ, a < b ∧ n = (a + b) * (b - a + 1) / 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_sum_power_of_two_l228_22878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_roots_imply_m_range_l228_22831

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.sin x)

noncomputable def vector_b (x m : ℝ) : ℝ × ℝ := (-Real.sin x, m + 1)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def has_three_roots_in_interval (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ x₃, Real.pi/6 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < 5*Real.pi/6 ∧
    f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0

theorem dot_product_roots_imply_m_range (m : ℝ) :
  has_three_roots_in_interval (λ x ↦ dot_product (vector_a x) (vector_b x m)) →
  1/2 < m ∧ m < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_roots_imply_m_range_l228_22831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vectors_angle_l228_22850

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

noncomputable def angle_between_vectors (v w : V) : ℝ := Real.arccos (inner v w / (norm v * norm w))

theorem orthogonal_vectors_angle {a b : V} (h : ‖a - b‖ = ‖b‖) :
  angle_between_vectors (a - 2 • b) a = π / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vectors_angle_l228_22850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l228_22893

theorem binomial_expansion_properties :
  (∀ k : ℕ, k ≤ 6 → (Nat.choose 6 k : ℕ) = (Finset.range 7).sum (λ i => if i = k then 1 else 0)) ∧
  (Nat.choose 6 3 = 20) ∧
  ((Finset.range 7).sum (λ k => Nat.choose 6 k) = 64) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l228_22893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximization_l228_22823

/-- Sales revenue function in million US dollars -/
noncomputable def R (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 40 then 400 - 6*x
  else 7400/x - 40000/(x^2)

/-- Profit function in million US dollars -/
noncomputable def W (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 40 then -6*x^2 + 384*x - 40
  else -40000/x - 16*x + 7360

/-- The production quantity that maximizes profit -/
def x_max : ℝ := 32

/-- The maximum profit -/
def max_profit : ℝ := 6104

/-- Theorem stating that the profit is maximized at x_max -/
theorem profit_maximization :
  ∀ x > 0, W x ≤ W x_max ∧ W x_max = max_profit := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximization_l228_22823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_john_mother_l228_22828

/-- Proves the age difference between John and his mother given the specified conditions -/
theorem age_difference_john_mother :
  -- John's father's age
  ∀ (father_age : ℤ),
  -- John's age is half of his father's
  ∀ (john_age : ℤ),
  -- John's mother's age
  ∀ (mother_age : ℤ),
  -- Conditions
  (father_age = 40) →
  (john_age * 2 = father_age) →
  (father_age = mother_age + 4) →
  -- Conclusion
  |john_age - mother_age| = 16 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_john_mother_l228_22828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integers_between_cubes_l228_22895

theorem integers_between_cubes : 
  (Int.floor ((10.5 : ℝ)^3) - Int.ceil ((10.4 : ℝ)^3) + 1 : ℤ) = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integers_between_cubes_l228_22895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_length_proof_wall_length_meters_l228_22874

/-- Proves that the length of a wall built with 6000 bricks, each measuring 25 cm x 11.25 cm x 6 cm,
    with a height of 6 m and a width of 22.5 cm, is 7.5 meters. -/
theorem wall_length_proof (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ)
                          (wall_height : ℝ) (wall_width : ℝ) (num_bricks : ℕ) :
  brick_length = 25 →
  brick_width = 11.25 →
  brick_height = 6 →
  wall_height = 600 →
  wall_width = 22.5 →
  num_bricks = 6000 →
  ∃ (wall_length : ℝ),
    wall_length = (↑num_bricks * brick_length * brick_width * brick_height) / (wall_height * wall_width) ∧
    wall_length = 750 :=
by
  sorry

/-- Converts the wall length from centimeters to meters. -/
noncomputable def cm_to_meters (length_cm : ℝ) : ℝ :=
  length_cm / 100

/-- Proves that the wall length is 7.5 meters. -/
theorem wall_length_meters (wall_length_cm : ℝ) :
  wall_length_cm = 750 →
  cm_to_meters wall_length_cm = 7.5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_length_proof_wall_length_meters_l228_22874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_price_return_l228_22838

theorem stock_price_return (initial_price : ℝ) (h : initial_price > 0) : 
  let price_after_two_years := initial_price * 1.3 * 1.2
  let decrease_percentage := (price_after_two_years - initial_price) / price_after_two_years
  ∃ ε > 0, |decrease_percentage - 0.35897| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_price_return_l228_22838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_solution_composition_l228_22807

/-- Represents a solution composed of liquid X and water -/
structure Solution where
  total_mass : ℚ
  liquid_x_percentage : ℚ

/-- The initial solution Y -/
def initial_solution_y : Solution :=
  { total_mass := 10,
    liquid_x_percentage := 30 }

/-- Amount of water that evaporates -/
def evaporated_water : ℚ := 2

/-- Amount of solution Y added after evaporation -/
def added_solution_y : ℚ := 2

/-- Calculates the final percentage of liquid X in the solution -/
def final_liquid_x_percentage (y : Solution) : ℚ :=
  let initial_liquid_x := y.total_mass * y.liquid_x_percentage / 100
  let remaining_mass := y.total_mass - evaporated_water
  let added_liquid_x := added_solution_y * y.liquid_x_percentage / 100
  let final_liquid_x := initial_liquid_x + added_liquid_x
  let final_mass := remaining_mass + added_solution_y
  (final_liquid_x / final_mass) * 100

theorem final_solution_composition :
  final_liquid_x_percentage initial_solution_y = 36 := by
  sorry

#eval final_liquid_x_percentage initial_solution_y

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_solution_composition_l228_22807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_one_eq_half_l228_22873

/-- A function f : ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- Definition of the function f -/
noncomputable def f : ℝ → ℝ := fun x => if x ≥ 0 then 1 / 2^x - 1 else -(1 / 2^(-x)) + 1

theorem f_neg_one_eq_half :
  IsOdd f → f (-1) = 1/2 := by
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_one_eq_half_l228_22873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l228_22882

/-- The principal amount invested -/
noncomputable def x : ℝ := sorry

/-- The interest rate as a percentage -/
noncomputable def y : ℝ := sorry

/-- The simple interest earned over two years -/
noncomputable def simple_interest : ℝ := x * y * 2 / 100

/-- The compound interest earned over two years -/
noncomputable def compound_interest : ℝ := x * ((1 + y/100)^2 - 1)

theorem investment_problem : 
  simple_interest = 900 ∧ 
  compound_interest = 922.50 → 
  x = 9000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l228_22882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_derivative_of_y_l228_22826

-- Define the function
noncomputable def y (x : ℝ) : ℝ := x^2 * Real.sin (5*x - 3)

-- State the theorem
theorem third_derivative_of_y (x : ℝ) :
  (deriv^[3] y) x = -150 * x * Real.sin (5*x - 3) + (30 - 125 * x^2) * Real.cos (5*x - 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_derivative_of_y_l228_22826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_returns_in_five_throws_l228_22815

/-- Represents the number of girls in the circle -/
def n : ℕ := 15

/-- Represents the number of girls skipped in each throw -/
def skip : ℕ := 5

/-- Calculates the position of the girl who receives the ball after a throw -/
def next_position (current : ℕ) : ℕ :=
  (current + skip + 1) % n

/-- Represents the sequence of positions the ball reaches -/
def ball_sequence : List ℕ :=
  let rec generate_sequence (current : ℕ) (acc : List ℕ) (fuel : ℕ) : List ℕ :=
    if fuel = 0 then acc
    else if current = 0 ∧ acc.length > 0 then acc
    else generate_sequence (next_position current) (acc ++ [current]) (fuel - 1)
  generate_sequence 0 [] n

/-- The main theorem stating that it takes 5 throws for the ball to return to Ami -/
theorem ball_returns_in_five_throws :
  ball_sequence.length = 5 ∧ ball_sequence.head? = ball_sequence.getLast? := by
  sorry

#eval ball_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_returns_in_five_throws_l228_22815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l228_22813

-- Define the function f(x) = log₄x + x - 7
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 4 + x - 7

-- Theorem statement
theorem root_in_interval :
  ∃ x ∈ Set.Ioo 5 6, f x = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l228_22813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_increase_prism_to_cubes_l228_22881

noncomputable section

/-- Calculates the surface area of a rectangular prism -/
def surface_area_prism (l w h : ℝ) : ℝ :=
  2 * (l * w + l * h + w * h)

/-- Calculates the volume of a rectangular prism -/
def volume_prism (l w h : ℝ) : ℝ :=
  l * w * h

/-- Calculates the surface area of a cube -/
def surface_area_cube (side : ℝ) : ℝ :=
  6 * side^2

/-- Calculates the percentage increase -/
def percentage_increase (original new : ℝ) : ℝ :=
  (new - original) / original * 100

theorem surface_area_increase_prism_to_cubes :
  let l : ℝ := 8
  let w : ℝ := 6
  let h : ℝ := 4
  let cube_side : ℝ := 1
  let original_sa := surface_area_prism l w h
  let prism_volume := volume_prism l w h
  let num_cubes := prism_volume / (cube_side^3)
  let total_cubes_sa := num_cubes * surface_area_cube cube_side
  abs (percentage_increase original_sa total_cubes_sa - 453.85) < 0.01 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_increase_prism_to_cubes_l228_22881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_implies_a_value_l228_22885

/-- Represents a hyperbola with parameter a -/
structure Hyperbola (a : ℝ) :=
  (equation : ∀ (x y : ℝ), x^2 - y^2/a^2 = 1)
  (a_pos : a > 0)

/-- Represents the asymptotic lines of a hyperbola -/
def AsymptoticLines (k : ℝ) : Set (ℝ → ℝ) :=
  {f | ∀ x, f x = k * x ∨ f x = -k * x}

theorem hyperbola_asymptote_implies_a_value
  (a : ℝ)
  (h : Hyperbola a)
  (asymp : AsymptoticLines 2) :
  a = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_implies_a_value_l228_22885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expectation_between_bounds_l228_22824

/-- A continuous random variable with its probability density function -/
structure ContinuousRandomVariable where
  a : ℝ
  b : ℝ
  h_ab : a ≤ b
  f : ℝ → ℝ
  h_nonneg : ∀ x, a ≤ x → x ≤ b → 0 ≤ f x
  h_zero_outside : ∀ x, x < a ∨ b < x → f x = 0
  h_integral_one : ∫ x in a..b, f x = 1

/-- The mathematical expectation of a continuous random variable -/
noncomputable def expectation (X : ContinuousRandomVariable) : ℝ :=
  ∫ x in X.a..X.b, x * X.f x

/-- Theorem: The expectation of a continuous random variable is between its minimum and maximum values -/
theorem expectation_between_bounds (X : ContinuousRandomVariable) :
  X.a ≤ expectation X ∧ expectation X ≤ X.b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expectation_between_bounds_l228_22824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_inequality_l228_22810

-- Define the vectors a and b
noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.cos x + Real.sin x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, Real.sin x - Real.cos x)

-- Define the dot product function
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := dot_product (a x) (b x)

-- State the theorem
theorem range_of_m_for_inequality (x : ℝ) (h : x ∈ Set.Icc (5 * Real.pi / 24) (5 * Real.pi / 12)) :
  {m : ℝ | ∀ t : ℝ, m * t^2 + m * t + 3 ≥ f x} = Set.Icc 0 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_inequality_l228_22810
