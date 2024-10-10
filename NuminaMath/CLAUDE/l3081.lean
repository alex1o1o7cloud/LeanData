import Mathlib

namespace inscribed_sphere_volume_l3081_308131

/-- The volume of a sphere inscribed in a right circular cylinder -/
theorem inscribed_sphere_volume (h : ℝ) (d : ℝ) (h_pos : h > 0) (d_pos : d > 0) :
  let r : ℝ := d / 2
  let cylinder_volume : ℝ := π * r^2 * h
  let sphere_volume : ℝ := (4/3) * π * r^3
  h = 12 ∧ d = 10 → sphere_volume = (500/3) * π := by sorry

end inscribed_sphere_volume_l3081_308131


namespace min_theta_value_l3081_308158

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) + Real.cos (ω * x)

theorem min_theta_value (ω : ℝ) (h_ω_pos : ω > 0) :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f ω (x + p) + |f ω (x + p)| = f ω x + |f ω x| ∧
    ∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f ω (x + q) + |f ω (x + q)| = f ω x + |f ω x|) → p ≤ q) →
  (∃ (θ : ℝ), θ > 0 ∧ ∀ (x : ℝ), f ω x ≥ f ω θ) →
  (∃ (θ_min : ℝ), θ_min > 0 ∧ 
    (∀ (x : ℝ), f ω x ≥ f ω θ_min) ∧
    (∀ (θ : ℝ), θ > 0 → (∀ (x : ℝ), f ω x ≥ f ω θ) → θ_min ≤ θ) ∧
    θ_min = 5 * Real.pi / 8) :=
sorry

end min_theta_value_l3081_308158


namespace arithmetic_sequence_product_l3081_308147

theorem arithmetic_sequence_product (a : ℤ) : 
  (∃ x : ℤ, x * (x + 1) * (x + 2) * (x + 3) = 360) → 
  (a * (a + 1) * (a + 2) * (a + 3) = 360 → (a = 3 ∨ a = -6)) :=
by
  sorry

end arithmetic_sequence_product_l3081_308147


namespace binomial_150_150_equals_1_l3081_308114

theorem binomial_150_150_equals_1 : Nat.choose 150 150 = 1 := by
  sorry

end binomial_150_150_equals_1_l3081_308114


namespace arithmetic_calculations_l3081_308189

theorem arithmetic_calculations :
  (128 + 52 / 13 = 132) ∧
  (132 / 11 * 29 - 178 = 170) ∧
  (45 * (320 / (4 * 5)) = 720) := by
  sorry

end arithmetic_calculations_l3081_308189


namespace arithmetic_progression_x_value_l3081_308153

/-- An arithmetic progression with first three terms x - 3, x + 3, and 3x + 5 has x = 2 -/
theorem arithmetic_progression_x_value (x : ℝ) : 
  let a₁ : ℝ := x - 3
  let a₂ : ℝ := x + 3
  let a₃ : ℝ := 3*x + 5
  (a₂ - a₁ = a₃ - a₂) → x = 2 :=
by sorry

end arithmetic_progression_x_value_l3081_308153


namespace modulus_of_z_l3081_308108

/-- Given a complex number z satisfying (1-i)z = 2i, prove that its modulus is √2 -/
theorem modulus_of_z (z : ℂ) (h : (1 - Complex.I) * z = 2 * Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end modulus_of_z_l3081_308108


namespace cubic_equation_unique_solution_l3081_308164

theorem cubic_equation_unique_solution :
  ∃! x : ℝ, x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3 ∧ x = 6 := by sorry

end cubic_equation_unique_solution_l3081_308164


namespace race_time_l3081_308165

/-- In a 1000-meter race, runner A beats runner B by 48 meters or 6 seconds -/
def Race (t : ℝ) : Prop :=
  -- A's distance in t seconds
  1000 = t * (1000 / t) ∧
  -- B's distance in t seconds
  952 = t * (952 / (t + 6)) ∧
  -- A and B have the same speed
  1000 / t = 952 / (t + 6)

/-- The time taken by runner A to complete the race is 125 seconds -/
theorem race_time : ∃ t : ℝ, Race t ∧ t = 125 := by sorry

end race_time_l3081_308165


namespace dogs_eat_six_cups_l3081_308125

/-- Represents the amount of dog food in various units -/
structure DogFood where
  cups : ℚ
  pounds : ℚ

/-- Represents the feeding schedule and food consumption for dogs -/
structure FeedingSchedule where
  dogsCount : ℕ
  feedingsPerDay : ℕ
  daysInMonth : ℕ
  bagsPerMonth : ℕ
  poundsPerBag : ℚ
  cupWeight : ℚ

/-- Calculates the number of cups of dog food each dog eats at a time -/
def cupsPerFeeding (fs : FeedingSchedule) : ℚ :=
  let totalPoundsPerMonth := fs.bagsPerMonth * fs.poundsPerBag
  let poundsPerDogPerMonth := totalPoundsPerMonth / fs.dogsCount
  let feedingsPerMonth := fs.feedingsPerDay * fs.daysInMonth
  let poundsPerFeeding := poundsPerDogPerMonth / feedingsPerMonth
  poundsPerFeeding / fs.cupWeight

/-- Theorem stating that each dog eats 6 cups of dog food at a time -/
theorem dogs_eat_six_cups
  (fs : FeedingSchedule)
  (h1 : fs.dogsCount = 2)
  (h2 : fs.feedingsPerDay = 2)
  (h3 : fs.daysInMonth = 30)
  (h4 : fs.bagsPerMonth = 9)
  (h5 : fs.poundsPerBag = 20)
  (h6 : fs.cupWeight = 1/4) :
  cupsPerFeeding fs = 6 := by
  sorry

#eval cupsPerFeeding {
  dogsCount := 2,
  feedingsPerDay := 2,
  daysInMonth := 30,
  bagsPerMonth := 9,
  poundsPerBag := 20,
  cupWeight := 1/4
}

end dogs_eat_six_cups_l3081_308125


namespace tree_height_differences_l3081_308117

def pine_height : ℚ := 14 + 1/4
def birch_height : ℚ := 18 + 1/2
def cedar_height : ℚ := 20 + 5/8

theorem tree_height_differences :
  (cedar_height - pine_height = 6 + 3/8) ∧
  (cedar_height - birch_height = 2 + 1/8) := by
  sorry

end tree_height_differences_l3081_308117


namespace geometric_sequence_property_l3081_308124

/-- A positive geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- The common ratio of a geometric sequence -/
def CommonRatio (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h1 : GeometricSequence a)
    (h2 : a 2 + a 4 = 3)
    (h3 : a 3 * a 5 = 1) :
    ∃ q : ℝ, CommonRatio a q ∧ q = Real.sqrt 2 / 2 ∧
    ∀ n : ℕ, a n = 2 ^ ((n + 2 : ℝ) / 2) :=
  sorry

end geometric_sequence_property_l3081_308124


namespace percentage_relation_l3081_308157

theorem percentage_relation (x y : ℕ) (N : ℚ) (hx : Prime x) (hy : Prime y) (hxy : x ≠ y) 
  (h : 70 = (x : ℚ) / 100 * N) : 
  (y : ℚ) / 100 * N = (y * 70 : ℚ) / x := by
  sorry

end percentage_relation_l3081_308157


namespace factor_x8_minus_81_l3081_308178

theorem factor_x8_minus_81 (x : ℝ) : x^8 - 81 = (x^4 + 9) * (x^2 + 3) * (x + Real.sqrt 3) * (x - Real.sqrt 3) := by
  sorry

end factor_x8_minus_81_l3081_308178


namespace f_is_even_f_monotonicity_on_0_1_l3081_308132

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 / (x^2 - 1)

-- Theorem for the even property of f
theorem f_is_even (a : ℝ) (ha : a ≠ 0) :
  ∀ x, x ≠ 1 ∧ x ≠ -1 → f a (-x) = f a x :=
sorry

-- Theorem for the monotonicity of f on (0, 1)
theorem f_monotonicity_on_0_1 (a : ℝ) (ha : a ≠ 0) :
  ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 →
    (a > 0 → f a x₁ > f a x₂) ∧
    (a < 0 → f a x₁ < f a x₂) :=
sorry

end f_is_even_f_monotonicity_on_0_1_l3081_308132


namespace smallest_triangle_perimeter_smallest_triangle_perimeter_proof_l3081_308134

/-- The smallest possible perimeter of a triangle with consecutive integer side lengths,
    where the smallest side is at least 4. -/
theorem smallest_triangle_perimeter : ℕ → Prop :=
  fun p => (∃ n : ℕ, n ≥ 4 ∧ p = n + (n + 1) + (n + 2)) ∧
           (∀ m : ℕ, m ≥ 4 → m + (m + 1) + (m + 2) ≥ p) →
           p = 15

/-- Proof of the smallest_triangle_perimeter theorem -/
theorem smallest_triangle_perimeter_proof : smallest_triangle_perimeter 15 := by
  sorry

end smallest_triangle_perimeter_smallest_triangle_perimeter_proof_l3081_308134


namespace marks_radiator_cost_l3081_308173

/-- The total cost of replacing a car radiator -/
def total_cost (work_duration : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) : ℕ :=
  work_duration * hourly_rate + part_cost

/-- Proof that Mark's total cost for replacing his car radiator is $300 -/
theorem marks_radiator_cost :
  total_cost 2 75 150 = 300 := by
  sorry

end marks_radiator_cost_l3081_308173


namespace term_2007_is_6019_l3081_308149

/-- An arithmetic sequence with first term 1, second term 4, and third term 7 -/
def arithmetic_sequence (n : ℕ) : ℕ :=
  1 + 3 * (n - 1)

/-- Theorem stating that the 2007th term of the sequence is 6019 -/
theorem term_2007_is_6019 : arithmetic_sequence 2007 = 6019 := by
  sorry

end term_2007_is_6019_l3081_308149


namespace hare_leaps_per_dog_leap_is_two_l3081_308197

/-- The number of hare leaps equal to one dog leap -/
def hare_leaps_per_dog_leap : ℕ := 2

/-- The number of dog leaps for a given number of hare leaps -/
def dog_leaps (hare_leaps : ℕ) : ℕ := (5 * hare_leaps : ℕ)

/-- The ratio of dog speed to hare speed -/
def speed_ratio : ℕ := 10

theorem hare_leaps_per_dog_leap_is_two :
  hare_leaps_per_dog_leap = 2 ∧
  (∀ h : ℕ, dog_leaps h = 5 * h) ∧
  speed_ratio = 10 := by
  sorry

end hare_leaps_per_dog_leap_is_two_l3081_308197


namespace roots_equal_condition_l3081_308107

theorem roots_equal_condition (m : ℝ) : 
  (∃! x : ℝ, (x * (x - 1) - (m + 1)) / ((x - 1) * (m - 1)) = x / m) ↔ m = -1/2 := by
  sorry

end roots_equal_condition_l3081_308107


namespace water_tank_capacity_l3081_308144

theorem water_tank_capacity (initial_fraction : Rat) (added_volume : ℝ) (final_fraction : Rat) :
  initial_fraction = 1/3 →
  added_volume = 5 →
  final_fraction = 2/5 →
  ∃ (capacity : ℝ), capacity = 75 ∧ 
    initial_fraction * capacity + added_volume = final_fraction * capacity :=
by sorry

end water_tank_capacity_l3081_308144


namespace square_side_length_average_l3081_308139

theorem square_side_length_average (a b c : ℝ) 
  (ha : a = 36) (hb : b = 64) (hc : c = 144) : 
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / 3 = 26 / 3 := by
sorry

end square_side_length_average_l3081_308139


namespace quilt_shaded_fraction_l3081_308135

/-- Represents a square quilt block -/
structure QuiltBlock where
  size : Nat
  total_squares : Nat
  fully_shaded : Nat
  half_shaded : Nat

/-- Calculates the fraction of shaded area in a quilt block -/
def shaded_fraction (q : QuiltBlock) : Rat :=
  let total_area : Rat := q.total_squares
  let shaded_area : Rat := q.fully_shaded + q.half_shaded / 2
  shaded_area / total_area

/-- Theorem stating the shaded fraction of the specific quilt block -/
theorem quilt_shaded_fraction :
  let q : QuiltBlock := ⟨4, 16, 2, 4⟩
  shaded_fraction q = 1 / 4 := by
  sorry

end quilt_shaded_fraction_l3081_308135


namespace sum_of_roots_times_two_l3081_308137

theorem sum_of_roots_times_two (a b : ℝ) : 
  (a^2 + a - 6 = 0) → (b^2 + b - 6 = 0) → (2*a + 2*b = -2) := by
  sorry

end sum_of_roots_times_two_l3081_308137


namespace linear_equation_solution_range_l3081_308118

theorem linear_equation_solution_range (x k : ℝ) : 
  (2 * x - 5 * k = x + 4) → (x > 0) → (k > -4/5) := by
  sorry

end linear_equation_solution_range_l3081_308118


namespace zero_in_M_l3081_308184

def M : Set ℤ := {-1, 0, 1}

theorem zero_in_M : 0 ∈ M := by
  sorry

end zero_in_M_l3081_308184


namespace complex_fraction_equals_one_tenth_l3081_308155

-- Define the expression
def complex_fraction : ℚ :=
  (⌈(23 / 9 : ℚ) - ⌈(35 / 23 : ℚ)⌉⌉ : ℚ) /
  (⌈(35 / 9 : ℚ) + ⌈(9 * 23 / 35 : ℚ)⌉⌉ : ℚ)

-- State the theorem
theorem complex_fraction_equals_one_tenth : complex_fraction = 1 / 10 := by
  sorry

end complex_fraction_equals_one_tenth_l3081_308155


namespace first_expression_equality_second_expression_equality_l3081_308102

-- First expression
theorem first_expression_equality (a : ℝ) :
  (-2 * a)^6 * (-3 * a^3) + (2 * a)^2 * 3 = -192 * a^9 + 12 * a^2 := by sorry

-- Second expression
theorem second_expression_equality :
  |(-1/8)| + π^3 + (-1/2)^3 - (1/3)^2 = π^3 - 1/9 := by sorry

end first_expression_equality_second_expression_equality_l3081_308102


namespace equation_solution_l3081_308159

theorem equation_solution :
  ∃ (X Y : ℚ), 
    (∀ x : ℚ, x ≠ 5 ∧ x ≠ 6 → 
      (Y * x + 8) / (x^2 - 11*x + 30) = X / (x - 5) + 7 / (x - 6)) →
    X + Y = -22/3 := by
  sorry

end equation_solution_l3081_308159


namespace f_odd_and_decreasing_l3081_308129

-- Define the function f(x) = -x³
def f (x : ℝ) : ℝ := -x^3

-- Theorem stating that f is both odd and decreasing
theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f y < f x) :=
by sorry

end f_odd_and_decreasing_l3081_308129


namespace sum_of_primes_less_than_20_is_77_l3081_308167

def is_prime (n : ℕ) : Prop := sorry

def sum_of_primes_less_than_20 : ℕ := sorry

theorem sum_of_primes_less_than_20_is_77 : 
  sum_of_primes_less_than_20 = 77 := by sorry

end sum_of_primes_less_than_20_is_77_l3081_308167


namespace boat_speed_distance_relationship_l3081_308166

/-- Represents the speed of a boat in various conditions -/
structure BoatSpeed where
  stillWater : ℝ
  downstream : ℝ
  upstream : ℝ

/-- Represents distances traveled by the boat -/
structure BoatDistance where
  downstream : ℝ
  upstream : ℝ

/-- Theorem stating the relationship between boat speed, current speed, and distances traveled -/
theorem boat_speed_distance_relationship 
  (speed : BoatSpeed) 
  (distance : BoatDistance) 
  (currentSpeed : ℝ) :
  speed.stillWater = 12 →
  speed.downstream = speed.stillWater + currentSpeed →
  speed.upstream = speed.stillWater - currentSpeed →
  distance.downstream = speed.downstream * 3 →
  distance.upstream = speed.upstream * 15 :=
by sorry

end boat_speed_distance_relationship_l3081_308166


namespace shaded_region_area_l3081_308175

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a semicircle -/
structure Semicircle where
  center : Point
  radius : ℝ

/-- The shaded region formed by the intersection of three semicircles -/
def ShadedRegion (s1 s2 s3 : Semicircle) : Set Point := sorry

/-- The area of a set of points -/
def area (s : Set Point) : ℝ := sorry

/-- The midpoint of an arc -/
def arcMidpoint (s : Semicircle) : Point := sorry

theorem shaded_region_area 
  (s1 s2 s3 : Semicircle)
  (h1 : s1.radius = 2 ∧ s2.radius = 2 ∧ s3.radius = 2)
  (h2 : arcMidpoint s1 = s3.center)
  (h3 : arcMidpoint s2 = s3.center)
  (h4 : s3.center = arcMidpoint s3) :
  area (ShadedRegion s1 s2 s3) = 8 := by sorry

end shaded_region_area_l3081_308175


namespace complement_union_theorem_l3081_308160

def U : Finset Nat := {1, 2, 3, 4, 5}
def M : Finset Nat := {1, 2}
def N : Finset Nat := {3, 4}

theorem complement_union_theorem :
  (U \ (M ∪ N)) = {5} := by sorry

end complement_union_theorem_l3081_308160


namespace vector_equality_transitivity_l3081_308101

variable {V : Type*} [AddCommGroup V]

theorem vector_equality_transitivity (a b c : V) : a = b → b = c → a = c := by
  sorry

end vector_equality_transitivity_l3081_308101


namespace angle_I_measure_l3081_308133

-- Define the pentagon and its angles
structure Pentagon where
  F : ℝ
  G : ℝ
  H : ℝ
  I : ℝ
  J : ℝ

-- Define the properties of the pentagon
def is_valid_pentagon (p : Pentagon) : Prop :=
  p.F > 0 ∧ p.G > 0 ∧ p.H > 0 ∧ p.I > 0 ∧ p.J > 0 ∧
  p.F + p.G + p.H + p.I + p.J = 540

-- Define the conditions given in the problem
def satisfies_conditions (p : Pentagon) : Prop :=
  p.F = p.G ∧ p.G = p.H ∧
  p.I = p.J ∧
  p.I = p.F + 30

-- Theorem statement
theorem angle_I_measure (p : Pentagon) 
  (h1 : is_valid_pentagon p) 
  (h2 : satisfies_conditions p) : 
  p.I = 126 := by
  sorry

end angle_I_measure_l3081_308133


namespace sara_marbles_count_l3081_308128

/-- The number of black marbles Sara has after receiving marbles from Fred -/
def saras_final_marbles (initial : ℝ) (received : ℝ) : ℝ :=
  initial + received

/-- Theorem: Sara's final number of marbles is 1025.0 -/
theorem sara_marbles_count :
  saras_final_marbles 792.0 233.0 = 1025.0 := by
  sorry

end sara_marbles_count_l3081_308128


namespace percentage_problem_l3081_308110

theorem percentage_problem (x : ℝ) (h : 0.2 * x = 400) : 1.2 * x = 2400 := by
  sorry

end percentage_problem_l3081_308110


namespace geometric_progression_problem_l3081_308172

theorem geometric_progression_problem (b₁ q : ℚ) : 
  (b₁ * q^3 - b₁ * q = -45/32) → 
  (b₁ * q^5 - b₁ * q^3 = -45/512) → 
  ((b₁ = 6 ∧ q = 1/4) ∨ (b₁ = -6 ∧ q = -1/4)) :=
by sorry

end geometric_progression_problem_l3081_308172


namespace ratio_of_trigonometric_equation_l3081_308196

theorem ratio_of_trigonometric_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a * Real.sin (π/5) + b * Real.cos (π/5)) / (a * Real.cos (π/5) - b * Real.sin (π/5)) = Real.tan (8*π/15)) :
  b / a = Real.sqrt 3 := by
  sorry

end ratio_of_trigonometric_equation_l3081_308196


namespace jenny_recycling_l3081_308193

/-- Represents the recycling problem with given weights and prices -/
structure RecyclingProblem where
  bottle_weight : Nat
  can_weight : Nat
  jar_weight : Nat
  max_weight : Nat
  cans_collected : Nat
  bottle_price : Nat
  can_price : Nat
  jar_price : Nat

/-- Calculates the number of jars that can be carried given the remaining weight -/
def max_jars (p : RecyclingProblem) (remaining_weight : Nat) : Nat :=
  remaining_weight / p.jar_weight

/-- Calculates the total money earned from recycling -/
def total_money (p : RecyclingProblem) (cans : Nat) (jars : Nat) (bottles : Nat) : Nat :=
  cans * p.can_price + jars * p.jar_price + bottles * p.bottle_price

/-- States the theorem about Jenny's recycling problem -/
theorem jenny_recycling (p : RecyclingProblem) 
  (h1 : p.bottle_weight = 6)
  (h2 : p.can_weight = 2)
  (h3 : p.jar_weight = 8)
  (h4 : p.max_weight = 100)
  (h5 : p.cans_collected = 20)
  (h6 : p.bottle_price = 10)
  (h7 : p.can_price = 3)
  (h8 : p.jar_price = 12) :
  let remaining_weight := p.max_weight - (p.cans_collected * p.can_weight)
  let jars := max_jars p remaining_weight
  let bottles := 0
  (cans, jars, bottles) = (20, 7, 0) ∧ 
  total_money p p.cans_collected jars bottles = 144 := by
  sorry

end jenny_recycling_l3081_308193


namespace ball_count_in_box_l3081_308111

theorem ball_count_in_box (n : ℕ) (yellow_count : ℕ) (prob_yellow : ℚ) : 
  yellow_count = 9 → prob_yellow = 3/10 → (yellow_count : ℚ) / n = prob_yellow → n = 30 := by
  sorry

end ball_count_in_box_l3081_308111


namespace parallelogram_missing_vertex_l3081_308182

/-- A parallelogram in a 2D coordinate system -/
structure Parallelogram where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- The area of a parallelogram -/
def area (p : Parallelogram) : ℝ := sorry

/-- Theorem: Given a parallelogram with three known vertices and a known area,
    prove that the fourth vertex has specific coordinates -/
theorem parallelogram_missing_vertex 
  (p : Parallelogram)
  (h1 : p.v1 = (4, 4))
  (h2 : p.v3 = (5, 9))
  (h3 : p.v4 = (8, 9))
  (h4 : area p = 5) :
  p.v2 = (3, 4) := by sorry

end parallelogram_missing_vertex_l3081_308182


namespace existence_of_sum_equality_l3081_308130

theorem existence_of_sum_equality (A : Set ℕ) 
  (h : ∀ n : ℕ, ∃ m ∈ A, n ≤ m ∧ m < n + 100) :
  ∃ a b c d : ℕ, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ d ∈ A ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a + b = c + d :=
sorry

end existence_of_sum_equality_l3081_308130


namespace least_consecutive_primes_l3081_308116

/-- Definition of the sequence x_n -/
def x (a b n : ℕ) : ℚ :=
  (a^n - 1) / (b^n - 1)

/-- Main theorem statement -/
theorem least_consecutive_primes (a b : ℕ) (h1 : a > b) (h2 : b > 1) :
  ∃ d : ℕ, d = 3 ∧
  (∀ n : ℕ, ¬(Prime (x a b n) ∧ Prime (x a b (n+1)) ∧ Prime (x a b (n+2)))) ∧
  (∀ d' : ℕ, d' < d →
    ∃ a' b' n' : ℕ, a' > b' ∧ b' > 1 ∧
      Prime (x a' b' n') ∧ Prime (x a' b' (n'+1)) ∧
      (d' = 2 → Prime (x a' b' (n'+2)))) :=
sorry

end least_consecutive_primes_l3081_308116


namespace similar_triangles_shortest_side_l3081_308122

theorem similar_triangles_shortest_side 
  (a b c : ℝ)  -- sides of the first triangle
  (d e f : ℝ)  -- sides of the second triangle
  (h1 : a^2 + b^2 = c^2)  -- first triangle is right-angled
  (h2 : d^2 + e^2 = f^2)  -- second triangle is right-angled
  (h3 : b = 15)  -- given side of first triangle
  (h4 : c = 17)  -- hypotenuse of first triangle
  (h5 : f = 51)  -- hypotenuse of second triangle
  (h6 : (a / d) = (b / e) ∧ (b / e) = (c / f))  -- triangles are similar
  : min d e = 24 := by sorry

end similar_triangles_shortest_side_l3081_308122


namespace geometric_sequence_sum_l3081_308194

/-- Given a geometric sequence {a_n} with common ratio 2 and sum of first four terms equal to 1,
    prove that the sum of the first eight terms is 17. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = 2 * a n) →  -- common ratio is 2
  (a 1 + a 2 + a 3 + a 4 = 1) →  -- sum of first four terms is 1
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 17) := by
sorry

end geometric_sequence_sum_l3081_308194


namespace segment_combination_uniqueness_l3081_308198

theorem segment_combination_uniqueness :
  ∃! (x y : ℕ), 7 * x + 12 * y = 100 :=
by sorry

end segment_combination_uniqueness_l3081_308198


namespace system_solutions_l3081_308187

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  (x + y)^4 = 6*x^2*y^2 - 215 ∧ x*y*(x^2 + y^2) = -78

/-- The set of solutions -/
def solutions : Set (ℝ × ℝ) :=
  {(3, -2), (-2, 3), (-3, 2), (2, -3)}

/-- Theorem stating that the solutions are correct and complete -/
theorem system_solutions :
  ∀ (x y : ℝ), system x y ↔ (x, y) ∈ solutions := by sorry

end system_solutions_l3081_308187


namespace keith_attended_games_l3081_308106

theorem keith_attended_games (total_games missed_games : ℕ) 
  (h1 : total_games = 20)
  (h2 : missed_games = 9) :
  total_games - missed_games = 11 := by
sorry

end keith_attended_games_l3081_308106


namespace remainder_8734_mod_9_l3081_308170

theorem remainder_8734_mod_9 : 8734 ≡ 4 [ZMOD 9] := by sorry

end remainder_8734_mod_9_l3081_308170


namespace sum_and_operations_l3081_308185

/-- Given three numbers a, b, and c, and a value M, such that:
    1. a + b + c = 100
    2. a - 10 = M
    3. b + 10 = M
    4. 10 * c = M
    Prove that M = 1000/21 -/
theorem sum_and_operations (a b c M : ℚ) 
  (sum_eq : a + b + c = 100)
  (a_dec : a - 10 = M)
  (b_inc : b + 10 = M)
  (c_mul : 10 * c = M) :
  M = 1000 / 21 := by
  sorry

end sum_and_operations_l3081_308185


namespace three_digit_numbers_divisible_by_nine_l3081_308181

/-- Given nonzero digits a, b, and c that form 6 distinct three-digit numbers,
    if the sum of these numbers is 5994, then each number is divisible by 9. -/
theorem three_digit_numbers_divisible_by_nine 
  (a b c : ℕ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (hsum : 100 * (a + b + c) + 10 * (a + b + c) + (a + b + c) = 5994) :
  let numbers := [100*a + 10*b + c, 100*a + 10*c + b, 
                  100*b + 10*a + c, 100*b + 10*c + a, 
                  100*c + 10*a + b, 100*c + 10*b + a]
  ∀ n ∈ numbers, n % 9 = 0 :=
by sorry

end three_digit_numbers_divisible_by_nine_l3081_308181


namespace area_smallest_rectangle_radius_6_l3081_308148

/-- The area of the smallest rectangle containing a circle of given radius -/
def smallest_rectangle_area (radius : ℝ) : ℝ :=
  (2 * radius) * (3 * radius)

/-- Theorem: The area of the smallest rectangle containing a circle of radius 6 is 216 -/
theorem area_smallest_rectangle_radius_6 :
  smallest_rectangle_area 6 = 216 := by
  sorry

end area_smallest_rectangle_radius_6_l3081_308148


namespace perfect_squares_from_products_l3081_308150

theorem perfect_squares_from_products (a b c d : ℕ) 
  (h1 : ∃ x : ℕ, a * b * c = x ^ 2)
  (h2 : ∃ x : ℕ, a * c * d = x ^ 2)
  (h3 : ∃ x : ℕ, b * c * d = x ^ 2)
  (h4 : ∃ x : ℕ, a * b * d = x ^ 2) :
  (∃ w : ℕ, a = w ^ 2) ∧ 
  (∃ x : ℕ, b = x ^ 2) ∧ 
  (∃ y : ℕ, c = y ^ 2) ∧ 
  (∃ z : ℕ, d = z ^ 2) := by
  sorry

end perfect_squares_from_products_l3081_308150


namespace trainees_seating_theorem_l3081_308199

/-- Represents the number of trainees and plates -/
def n : ℕ := 67

/-- Represents the number of correct seatings after rotating i positions -/
def correct_seatings (i : ℕ) : ℕ := sorry

theorem trainees_seating_theorem :
  ∃ i : ℕ, i > 0 ∧ i < n ∧ correct_seatings i ≥ 2 :=
sorry

end trainees_seating_theorem_l3081_308199


namespace hit_rate_calculation_l3081_308112

theorem hit_rate_calculation (p₁ p₂ : ℚ) : 
  (p₁ * (1 - p₂) * (1/3 : ℚ) = 1/18) →
  (p₂ * (2/3 : ℚ) = 4/9) →
  p₁ = 1/2 ∧ p₂ = 2/3 := by
  sorry

end hit_rate_calculation_l3081_308112


namespace alex_paper_distribution_l3081_308109

/-- The number of ways to distribute n distinct items to m recipients,
    where each recipient can receive multiple items. -/
def distribution_ways (n m : ℕ) : ℕ := m^n

/-- The problem statement -/
theorem alex_paper_distribution :
  distribution_ways 5 10 = 100000 := by
  sorry

end alex_paper_distribution_l3081_308109


namespace isosceles_triangle_ef_length_l3081_308156

/-- In an isosceles triangle DEF, G is the point where the altitude from D meets EF. -/
structure IsoscelesTriangle where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  is_isosceles : dist D E = dist D F
  altitude : (G.1 - D.1) * (E.1 - F.1) + (G.2 - D.2) * (E.2 - F.2) = 0
  on_base : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ G = (1 - t) • E + t • F

/-- The length of EF in the isosceles triangle DEF. -/
def EF_length (triangle : IsoscelesTriangle) : ℝ :=
  dist triangle.E triangle.F

/-- The theorem stating the length of EF in the specific isosceles triangle. -/
theorem isosceles_triangle_ef_length 
  (triangle : IsoscelesTriangle)
  (de_length : dist triangle.D triangle.E = 5)
  (eg_gf_ratio : dist triangle.E triangle.G = 4 * dist triangle.G triangle.F) :
  EF_length triangle = (5 * Real.sqrt 10) / 4 := by
  sorry

end isosceles_triangle_ef_length_l3081_308156


namespace opposite_of_negative_five_l3081_308191

theorem opposite_of_negative_five : -((-5 : ℤ)) = 5 := by sorry

end opposite_of_negative_five_l3081_308191


namespace number_of_observations_l3081_308186

theorem number_of_observations (initial_mean new_mean : ℝ) 
  (wrong_value correct_value : ℝ) (n : ℕ) : 
  initial_mean = 36 →
  wrong_value = 23 →
  correct_value = 34 →
  new_mean = 36.5 →
  (n : ℝ) * initial_mean + (correct_value - wrong_value) = (n : ℝ) * new_mean →
  n = 22 := by
  sorry

#check number_of_observations

end number_of_observations_l3081_308186


namespace manuscript_cost_theorem_l3081_308145

/-- Calculates the total cost of typing and revising a manuscript. -/
def manuscript_cost (total_pages : ℕ) (first_time_cost : ℕ) (revision_cost : ℕ) 
  (revised_once : ℕ) (revised_twice : ℕ) (revised_thrice : ℕ) : ℕ :=
  total_pages * first_time_cost + 
  revised_once * revision_cost + 
  revised_twice * revision_cost * 2 + 
  revised_thrice * revision_cost * 3

theorem manuscript_cost_theorem : 
  manuscript_cost 500 5 4 200 150 50 = 5100 := by
  sorry

#eval manuscript_cost 500 5 4 200 150 50

end manuscript_cost_theorem_l3081_308145


namespace square_sum_given_product_and_sum_l3081_308162

theorem square_sum_given_product_and_sum (m n : ℝ) 
  (h1 : m * n = 12) 
  (h2 : m + n = 8) : 
  m^2 + n^2 = 40 := by
  sorry

end square_sum_given_product_and_sum_l3081_308162


namespace sphere_volume_from_surface_area_l3081_308168

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
    r > 0 →
    4 * π * r^2 = 256 * π →
    (4 / 3) * π * r^3 = (2048 / 3) * π := by
  sorry

end sphere_volume_from_surface_area_l3081_308168


namespace square_of_rational_difference_l3081_308190

theorem square_of_rational_difference (x y : ℚ) (h : x^5 + y^5 = 2*x^2*y^2) :
  ∃ z : ℚ, 1 - x*y = z^2 := by sorry

end square_of_rational_difference_l3081_308190


namespace limit_hours_proof_l3081_308188

/-- The limit of hours per week for the regular rate -/
def limit_hours : ℕ := sorry

/-- The regular hourly rate in dollars -/
def regular_rate : ℚ := 16

/-- The overtime rate as a percentage increase over the regular rate -/
def overtime_rate_increase : ℚ := 75 / 100

/-- The total hours worked in a week -/
def total_hours : ℕ := 44

/-- The total compensation earned in dollars -/
def total_compensation : ℚ := 752

/-- Calculates the overtime rate based on the regular rate and overtime rate increase -/
def overtime_rate : ℚ := regular_rate * (1 + overtime_rate_increase)

theorem limit_hours_proof :
  regular_rate * limit_hours + 
  overtime_rate * (total_hours - limit_hours) = 
  total_compensation ∧ 
  limit_hours = 40 := by sorry

end limit_hours_proof_l3081_308188


namespace pascal_triangle_row_15_fifth_number_l3081_308154

theorem pascal_triangle_row_15_fifth_number :
  let row := List.map (fun k => Nat.choose 15 k) (List.range 16)
  row[0] = 1 ∧ row[1] = 15 →
  row[4] = Nat.choose 15 4 ∧ Nat.choose 15 4 = 1365 :=
by sorry

end pascal_triangle_row_15_fifth_number_l3081_308154


namespace work_completion_time_l3081_308115

/-- The number of days x needs to finish the work alone -/
def x_days : ℝ := 18

/-- The number of days y worked before leaving -/
def y_worked : ℝ := 5

/-- The number of days x needed to finish the remaining work after y left -/
def x_remaining : ℝ := 12

/-- The number of days y needs to finish the work alone -/
def y_days : ℝ := 15

theorem work_completion_time : 
  (y_worked / y_days) + (x_remaining / x_days) = 1 := by sorry

end work_completion_time_l3081_308115


namespace exactly_four_false_l3081_308152

/-- Represents a statement about the number of false statements -/
inductive Statement
  | one
  | two
  | three
  | four
  | five

/-- Returns true if the statement is consistent with the given number of false statements -/
def isConsistent (s : Statement) (numFalse : Nat) : Bool :=
  match s with
  | .one => numFalse = 1
  | .two => numFalse = 2
  | .three => numFalse = 3
  | .four => numFalse = 4
  | .five => numFalse = 5

/-- The list of all statements on the card -/
def allStatements : List Statement := [.one, .two, .three, .four, .five]

/-- Counts the number of false statements given a predicate -/
def countFalse (pred : Statement → Bool) : Nat :=
  allStatements.filter (fun s => !pred s) |>.length

theorem exactly_four_false :
  ∃ (pred : Statement → Bool),
    (∀ s, pred s ↔ isConsistent s (countFalse pred)) ∧
    countFalse pred = 4 := by
  sorry

end exactly_four_false_l3081_308152


namespace bag_of_balls_l3081_308141

theorem bag_of_balls (white green yellow red purple : ℕ) 
  (h1 : white = 22)
  (h2 : green = 18)
  (h3 : yellow = 8)
  (h4 : red = 5)
  (h5 : purple = 7)
  (h6 : (white + green + yellow : ℝ) / (white + green + yellow + red + purple) = 0.8) :
  white + green + yellow + red + purple = 60 := by
  sorry

end bag_of_balls_l3081_308141


namespace union_equals_real_complement_intersect_B_l3081_308179

-- Define sets A and B
def A : Set ℝ := {x | x - 2 ≥ 0}
def B : Set ℝ := {x | x < 5}

-- Theorem for A ∪ B = ℝ
theorem union_equals_real : A ∪ B = Set.univ := by sorry

-- Theorem for (∁ₐA) ∩ B = {x | x < 2}
theorem complement_intersect_B : 
  (Set.univ \ A) ∩ B = {x : ℝ | x < 2} := by sorry

end union_equals_real_complement_intersect_B_l3081_308179


namespace rectangle_segment_length_l3081_308142

/-- Given a rectangle with dimensions 10 units by 5 units, prove that the total length
    of segments in a new figure formed by removing three sides is 15 units. The remaining
    segments include two full heights and two parts of the width (3 units and 2 units). -/
theorem rectangle_segment_length :
  let original_width : ℕ := 10
  let original_height : ℕ := 5
  let remaining_width_part1 : ℕ := 3
  let remaining_width_part2 : ℕ := 2
  let total_length : ℕ := 2 * original_height + remaining_width_part1 + remaining_width_part2
  total_length = 15 := by sorry

end rectangle_segment_length_l3081_308142


namespace journey_time_calculation_l3081_308104

/-- Proves that if walking twice the distance of running takes 30 minutes,
    then walking one-third and running two-thirds of the same distance takes 24 minutes,
    given that running speed is twice the walking speed. -/
theorem journey_time_calculation (v : ℝ) (S : ℝ) (h1 : v > 0) (h2 : S > 0) :
  (2 * S / v + S / (2 * v) = 30) →
  (S / v + 2 * S / (2 * v) = 24) :=
by sorry

end journey_time_calculation_l3081_308104


namespace middle_part_of_proportional_division_l3081_308138

theorem middle_part_of_proportional_division (total : ℚ) (a b c : ℚ) 
  (h_total : total = 120)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_prop : a = 2 * b ∧ c = (1/2) * b) : 
  b = 240/7 := by
  sorry

end middle_part_of_proportional_division_l3081_308138


namespace length_EC_l3081_308123

-- Define the points
variable (A B C D E : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
variable (h1 : ∃ t : ℝ, C = A + t • (C - A) ∧ D = B + t • (D - B))
variable (h2 : ‖A - E‖ = ‖A - B‖ - 1)
variable (h3 : ‖A - E‖ = ‖D - C‖)
variable (h4 : ‖A - D‖ = ‖B - E‖)
variable (h5 : angle A D C = angle D E C)

-- The theorem to prove
theorem length_EC : ‖E - C‖ = 1 := by
  sorry

end length_EC_l3081_308123


namespace andrews_eggs_l3081_308174

theorem andrews_eggs (initial_eggs bought_eggs final_eggs : ℕ) : 
  bought_eggs = 62 → final_eggs = 70 → initial_eggs + bought_eggs = final_eggs → initial_eggs = 8 := by
  sorry

end andrews_eggs_l3081_308174


namespace no_real_solutions_l3081_308120

theorem no_real_solutions : ∀ x : ℝ, (2*x - 10*x + 24)^2 + 4 ≠ -2*|x| := by
  sorry

end no_real_solutions_l3081_308120


namespace consecutive_integers_sum_of_powers_l3081_308180

theorem consecutive_integers_sum_of_powers (n : ℤ) : 
  (n - 1)^2 + n^2 + (n + 1)^2 = 2450 → 
  (n - 1)^5 + n^5 + (n + 1)^5 = 52070424 := by
  sorry

end consecutive_integers_sum_of_powers_l3081_308180


namespace a1_iff_a2017_positive_l3081_308183

/-- An arithmetic-geometric sequence -/
structure ArithmeticGeometricSequence where
  a : ℕ → ℝ
  q : ℝ

/-- The theorem stating that for an arithmetic-geometric sequence with q = 0,
    a₁ > 0 if and only if a₂₀₁₇ > 0 -/
theorem a1_iff_a2017_positive (seq : ArithmeticGeometricSequence) 
    (h_q : seq.q = 0) :
    seq.a 1 > 0 ↔ seq.a 2017 > 0 := by
  sorry

end a1_iff_a2017_positive_l3081_308183


namespace geometric_sequence_fourth_term_l3081_308195

/-- Given a geometric sequence {aₙ} where the first three terms are x, 2x+2, and 3x+3 respectively,
    prove that the fourth term a₄ = -27/2. -/
theorem geometric_sequence_fourth_term (x : ℝ) (a : ℕ → ℝ) :
  a 1 = x ∧ a 2 = 2*x + 2 ∧ a 3 = 3*x + 3 ∧ 
  (∀ n : ℕ, n ≥ 1 → a (n + 1) / a n = a 2 / a 1) →
  a 4 = -27/2 := by
sorry

end geometric_sequence_fourth_term_l3081_308195


namespace no_charming_numbers_l3081_308113

/-- A two-digit positive integer is charming if it equals the sum of the square of its tens digit
and the product of its digits. -/
def IsCharming (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ ∃ a b : ℕ, n = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = a^2 + a * b

/-- There are no charming two-digit positive integers. -/
theorem no_charming_numbers : ¬∃ n : ℕ, IsCharming n := by
  sorry

end no_charming_numbers_l3081_308113


namespace max_value_and_sum_l3081_308119

theorem max_value_and_sum (x y z v w : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) (pos_v : 0 < v) (pos_w : 0 < w)
  (sum_sq : x^2 + y^2 + z^2 + v^2 + w^2 = 2016) : 
  ∃ (N x_N y_N z_N v_N w_N : ℝ),
    (∀ (a b c d e : ℝ), 0 < a → 0 < b → 0 < c → 0 < d → 0 < e → 
      a^2 + b^2 + c^2 + d^2 + e^2 = 2016 → 
      4*a*c + 3*b*c + 2*c*d + 4*c*e ≤ N) ∧
    (4*x_N*z_N + 3*y_N*z_N + 2*z_N*v_N + 4*z_N*w_N = N) ∧
    (x_N^2 + y_N^2 + z_N^2 + v_N^2 + w_N^2 = 2016) ∧
    (N + x_N + y_N + z_N + v_N + w_N = 78 + 2028 * Real.sqrt 37) := by
  sorry

end max_value_and_sum_l3081_308119


namespace find_A_l3081_308140

theorem find_A : ∀ A : ℕ, (A / 9 = 2 ∧ A % 9 = 6) → A = 24 := by
  sorry

end find_A_l3081_308140


namespace perpendicular_vectors_k_l3081_308121

def a : Fin 2 → ℝ := ![1, 1]
def b : Fin 2 → ℝ := ![1, 2]

theorem perpendicular_vectors_k (k : ℝ) :
  (∀ i : Fin 2, (k * a i - b i) * (b i + a i) = 0) →
  k = 8/5 := by
  sorry

end perpendicular_vectors_k_l3081_308121


namespace total_profit_is_2034_l3081_308126

/-- Represents a group of piglets with their selling and feeding information -/
structure PigletGroup where
  count : Nat
  sellingPrice : Nat
  sellingTime : Nat
  initialFeedCost : Nat
  initialFeedTime : Nat
  laterFeedCost : Nat
  laterFeedTime : Nat

/-- Calculates the profit for a single piglet group -/
def groupProfit (group : PigletGroup) : Int :=
  group.count * group.sellingPrice - 
  group.count * (group.initialFeedCost * group.initialFeedTime + 
                 group.laterFeedCost * group.laterFeedTime)

/-- The farmer's piglet groups -/
def pigletGroups : List PigletGroup := [
  ⟨3, 375, 11, 13, 8, 15, 3⟩,
  ⟨4, 425, 14, 14, 5, 16, 9⟩,
  ⟨2, 475, 18, 15, 10, 18, 8⟩,
  ⟨1, 550, 20, 20, 20, 20, 0⟩
]

/-- Theorem stating the total profit is $2034 -/
theorem total_profit_is_2034 : 
  (pigletGroups.map groupProfit).sum = 2034 := by
  sorry

end total_profit_is_2034_l3081_308126


namespace club_sports_intersection_l3081_308100

/-- Given a club with 310 members, where 138 play tennis, 255 play baseball,
    and 11 play no sports, prove that 94 people play both tennis and baseball. -/
theorem club_sports_intersection (total : ℕ) (tennis : ℕ) (baseball : ℕ) (no_sport : ℕ)
    (h_total : total = 310)
    (h_tennis : tennis = 138)
    (h_baseball : baseball = 255)
    (h_no_sport : no_sport = 11) :
    tennis + baseball - (total - no_sport) = 94 := by
  sorry

end club_sports_intersection_l3081_308100


namespace beaver_group_size_l3081_308105

/-- The number of beavers in the first group -/
def first_group_size : ℕ := 20

/-- The time taken by the first group to build the dam (in hours) -/
def first_group_time : ℕ := 3

/-- The number of beavers in the second group -/
def second_group_size : ℕ := 12

/-- The time taken by the second group to build the dam (in hours) -/
def second_group_time : ℕ := 5

/-- Theorem stating that the first group size is 20 beavers -/
theorem beaver_group_size :
  first_group_size * first_group_time = second_group_size * second_group_time :=
by sorry

end beaver_group_size_l3081_308105


namespace quadratic_equation_roots_range_l3081_308169

theorem quadratic_equation_roots_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) →
  m < -2 ∨ m > 2 := by
sorry

end quadratic_equation_roots_range_l3081_308169


namespace smallest_dual_base_representation_l3081_308127

theorem smallest_dual_base_representation :
  ∃ (n : ℕ) (a b : ℕ), 
    a > 3 ∧ b > 3 ∧
    n = 2 * a + 2 ∧
    n = 3 * b + 3 ∧
    (∀ (m : ℕ) (c d : ℕ), c > 3 → d > 3 → m = 2 * c + 2 → m = 3 * d + 3 → m ≥ n) ∧
    n = 18 := by
  sorry

end smallest_dual_base_representation_l3081_308127


namespace sector_properties_l3081_308192

/-- Given a sector with radius 2 cm and central angle 2 radians, prove that its arc length is 4 cm and its area is 4 cm². -/
theorem sector_properties :
  let r : ℝ := 2  -- radius in cm
  let α : ℝ := 2  -- central angle in radians
  let arc_length : ℝ := r * α
  let sector_area : ℝ := (1/2) * r^2 * α
  (arc_length = 4 ∧ sector_area = 4) :=
by sorry

end sector_properties_l3081_308192


namespace movie_theater_revenue_l3081_308103

theorem movie_theater_revenue
  (adult_price : ℕ)
  (child_price : ℕ)
  (total_tickets : ℕ)
  (adult_tickets : ℕ)
  (h1 : adult_price = 7)
  (h2 : child_price = 4)
  (h3 : total_tickets = 900)
  (h4 : adult_tickets = 500)
  : adult_price * adult_tickets + child_price * (total_tickets - adult_tickets) = 5100 := by
  sorry

end movie_theater_revenue_l3081_308103


namespace unique_solution_square_sum_product_l3081_308163

theorem unique_solution_square_sum_product : 
  ∃! (a b : ℕ+), a^2 + b^2 = a * b * (a + b) := by
sorry

end unique_solution_square_sum_product_l3081_308163


namespace periodic_odd_function_sum_l3081_308161

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem periodic_odd_function_sum (f : ℝ → ℝ) 
  (h_periodic : is_periodic f 3)
  (h_odd : is_odd f)
  (h_f_neg_one : f (-1) = 2) :
  f 2011 + f 2012 = 0 := by
sorry

end periodic_odd_function_sum_l3081_308161


namespace factor_expression_l3081_308177

theorem factor_expression (x : ℝ) : 75 * x^19 + 225 * x^38 = 75 * x^19 * (1 + 3 * x^19) := by
  sorry

end factor_expression_l3081_308177


namespace unique_solution_exists_l3081_308146

theorem unique_solution_exists (x y z : ℝ) : 
  (x / 6) * 12 = 11 ∧ 
  4 * (x - y) + 5 = 11 ∧ 
  Real.sqrt z = (3 * x + y / 2) ^ 2 →
  x = 5.5 ∧ y = 4 ∧ z = 117132.0625 :=
by sorry

end unique_solution_exists_l3081_308146


namespace function_properties_l3081_308176

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := a^x - a + 1

def g (a : ℝ) (x : ℝ) : ℝ := f a (x + 1/2) - 1

def F (a m : ℝ) (x : ℝ) : ℝ := g a (2*x) - m * g a (x - 1)

def h (m : ℝ) : ℝ :=
  if m ≤ 1 then 1 - 2*m
  else if m < 2 then -m^2
  else 4 - 4*m

theorem function_properties (a : ℝ) (ha : a > 0 ∧ a ≠ 1) (hf : f a (1/2) = 2) :
  a = 1/2 ∧
  (∀ x, g a x = (1/2)^x) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 0, F a m x ≥ h m) :=
sorry

end function_properties_l3081_308176


namespace max_distinct_letters_exists_table_with_11_letters_l3081_308136

/-- Represents a 5x5 table of letters -/
def LetterTable := Fin 5 → Fin 5 → Char

/-- Checks if a row contains at most 3 different letters -/
def rowValid (table : LetterTable) (row : Fin 5) : Prop :=
  (Finset.image (λ col => table row col) Finset.univ).card ≤ 3

/-- Checks if a column contains at most 3 different letters -/
def colValid (table : LetterTable) (col : Fin 5) : Prop :=
  (Finset.image (λ row => table row col) Finset.univ).card ≤ 3

/-- Checks if the entire table is valid -/
def tableValid (table : LetterTable) : Prop :=
  (∀ row, rowValid table row) ∧ (∀ col, colValid table col)

/-- Counts the number of different letters in the table -/
def distinctLetters (table : LetterTable) : ℕ :=
  (Finset.image (λ (row, col) => table row col) (Finset.univ.product Finset.univ)).card

/-- The main theorem stating that the maximum number of distinct letters is 11 -/
theorem max_distinct_letters :
  ∀ (table : LetterTable), tableValid table → distinctLetters table ≤ 11 :=
sorry

/-- There exists a valid table with exactly 11 distinct letters -/
theorem exists_table_with_11_letters :
  ∃ (table : LetterTable), tableValid table ∧ distinctLetters table = 11 :=
sorry

end max_distinct_letters_exists_table_with_11_letters_l3081_308136


namespace prob_two_females_is_three_tenths_l3081_308171

-- Define the total number of contestants
def total_contestants : ℕ := 5

-- Define the number of female contestants
def female_contestants : ℕ := 3

-- Define the number of contestants to be chosen
def chosen_contestants : ℕ := 2

-- Define the probability of choosing 2 female contestants
def prob_two_females : ℚ := (female_contestants.choose chosen_contestants) / (total_contestants.choose chosen_contestants)

-- Theorem statement
theorem prob_two_females_is_three_tenths : prob_two_females = 3 / 10 := by
  sorry

end prob_two_females_is_three_tenths_l3081_308171


namespace smallest_cube_box_volume_for_cone_l3081_308143

/-- The volume of the smallest cube-shaped box that can accommodate a cone vertically -/
theorem smallest_cube_box_volume_for_cone (cone_height : ℝ) (cone_base_diameter : ℝ) 
  (h_height : cone_height = 15) 
  (h_diameter : cone_base_diameter = 8) : ℝ := by
  sorry

#check smallest_cube_box_volume_for_cone

end smallest_cube_box_volume_for_cone_l3081_308143


namespace sum_is_integer_l3081_308151

theorem sum_is_integer (x y z : ℝ) 
  (h1 : x^2 = y + 2) 
  (h2 : y^2 = z + 2) 
  (h3 : z^2 = x + 2) : 
  ∃ n : ℤ, (x + y + z : ℝ) = n := by
sorry

end sum_is_integer_l3081_308151
