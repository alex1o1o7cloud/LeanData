import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l72_7237

theorem divisibility_condition (m n : ℕ+) : 
  (∃ k : ℕ, (9^(Int.natAbs (m.val - n.val)) + 3^(Int.natAbs (m.val - n.val)) + 1) = k * m.val) ∧ 
  (∃ l : ℕ, (9^(Int.natAbs (m.val - n.val)) + 3^(Int.natAbs (m.val - n.val)) + 1) = l * n.val) ↔ 
  ((m = 1 ∧ n = 1) ∨ (m = 3 ∧ n = 3)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l72_7237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l72_7232

-- Define the ray l
noncomputable def ray_l (x : ℝ) : ℝ := Real.sqrt 3 * x

-- Define curve C₂
def curve_C2 (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Define curve C₃ in polar coordinates
def curve_C3 (ρ θ : ℝ) : Prop := ρ = 8 * Real.sin θ

-- Define the intersection points
noncomputable def point_M : ℝ × ℝ := sorry
noncomputable def point_N : ℝ × ℝ := sorry

-- Theorem statement
theorem intersection_distance :
  let (x_M, y_M) := point_M
  let (x_N, y_N) := point_N
  curve_C2 x_M y_M ∧
  curve_C3 (Real.sqrt (x_N^2 + y_N^2)) (Real.arctan (y_N / x_N)) ∧
  y_M = ray_l x_M ∧
  y_N = ray_l x_N →
  Real.sqrt ((x_M - x_N)^2 + (y_M - y_N)^2) = 2 * Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l72_7232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_traveled_l72_7297

-- Define the velocity function
def v (t : ℝ) : ℝ := 3 * t^2 + t

-- Define the distance function as the integral of velocity
noncomputable def s (a b : ℝ) : ℝ := ∫ t in a..b, v t

-- Theorem statement
theorem distance_traveled : s 0 4 = 72 := by
  -- The proof steps would go here, but for now we'll use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_traveled_l72_7297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_ultra_squarish_numbers_l72_7230

/-- An ultra-squarish number is an eight-digit number satisfying specific conditions --/
def UltraSquarish (n : ℕ) : Prop :=
  (100000000 ≤ n ∧ n < 1000000000) ∧  -- eight-digit number
  (∀ d, d ∈ Nat.digits 10 n → d ≠ 0) ∧  -- no digit is zero
  (∃ k : ℕ, n = k^2) ∧  -- perfect square
  (∃ a b c d : ℕ, 
    n = 10000000*a + 1000000*b + 100000*c + 10000*d + (n % 10000) ∧
    (10 ≤ a ∧ a < 100 ∧ ∃ k : ℕ, a = k^2) ∧
    (10 ≤ b ∧ b < 100 ∧ ∃ k : ℕ, b = k^2) ∧
    (10 ≤ c ∧ c < 100 ∧ ∃ k : ℕ, c = k^2) ∧
    (10 ≤ d ∧ d < 100 ∧ ∃ k : ℕ, d = k^2))

/-- There are no ultra-squarish numbers --/
theorem no_ultra_squarish_numbers : ¬∃ n : ℕ, UltraSquarish n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_ultra_squarish_numbers_l72_7230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l72_7200

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  area : ℝ
  h1 : Real.sqrt 3 * b * Real.cos A = a * Real.sin B
  h2 : a = 6
  h3 : area = 9 * Real.sqrt 3

/-- Main theorem about the triangle -/
theorem triangle_properties (t : Triangle) : 
  t.A = Real.pi / 3 ∧ t.b = 6 ∧ t.c = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l72_7200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l72_7242

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := 2 * Real.exp x - k * x - 2

theorem function_property (k : ℝ) :
  (∃ m : ℝ, m > 0 ∧ ∀ x : ℝ, 0 < x ∧ x < m → |f k x| > 2 * x) ↔ k > 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l72_7242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_ratio_ellipse_intersections_l72_7269

/-- Ellipse type -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Line type -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ := Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Check if a point is on the ellipse -/
def on_ellipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Check if a point is on a line -/
def on_line (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- The right focus of the ellipse -/
noncomputable def right_focus (e : Ellipse) : Point :=
  { x := Real.sqrt (e.a^2 - e.b^2), y := 0 }

/-- Theorem: Constant ratio for intersecting lines on ellipse -/
theorem constant_ratio_ellipse_intersections (e : Ellipse) 
    (l₁ l₂ : Line) (A B C D : Point) :
  e.a = 2 →
  e.b = Real.sqrt 3 →
  on_line (right_focus e) l₁ →
  l₂.intercept = 0 →
  l₁.slope = l₂.slope →
  on_ellipse A e ∧ on_ellipse B e ∧ on_line A l₁ ∧ on_line B l₁ →
  on_ellipse C e ∧ on_ellipse D e ∧ on_line C l₂ ∧ on_line D l₂ →
  (distance C D)^2 / distance A B = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_ratio_ellipse_intersections_l72_7269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l72_7203

/-- Given two vectors a and b in ℝ², where a = (2, -1) and b = (1, lambda),
    if (a + 2b) is parallel to (2a - b), then lambda = -1/2 -/
theorem parallel_vectors_lambda (lambda : ℝ) : 
  let a : Fin 2 → ℝ := ![2, -1]
  let b : Fin 2 → ℝ := ![1, lambda]
  (∃ (k : ℝ), k ≠ 0 ∧ (a + 2 • b) = k • (2 • a - b)) → lambda = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l72_7203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aquarium_animals_l72_7220

/-- The number of otters in the aquarium -/
def otters : ℕ := sorry

/-- The number of seals in the aquarium -/
def seals : ℕ := sorry

/-- The number of sea lions in the aquarium -/
def sea_lions : ℕ := sorry

/-- There are exactly 7 either otters or seals -/
axiom otters_or_seals : otters = 7 ∨ seals = 7

/-- There are exactly 6 either sea lions or seals -/
axiom sea_lions_or_seals : sea_lions = 6 ∨ seals = 6

/-- There are exactly 5 either otters or sea lions -/
axiom otters_or_sea_lions : otters = 5 ∨ sea_lions = 5

/-- The fewest are either seals or otters -/
axiom fewest_seals_or_otters : (otters ≤ seals ∧ otters ≤ sea_lions) ∨ (seals ≤ otters ∧ seals ≤ sea_lions)

/-- The correct numbers of animals in the aquarium -/
theorem aquarium_animals : otters = 5 ∧ seals = 7 ∧ sea_lions = 6 := by
  sorry

#check aquarium_animals

end NUMINAMATH_CALUDE_ERRORFEEDBACK_aquarium_animals_l72_7220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_roots_characterization_l72_7279

/-- 
Given positive integers m and n, this function returns true if and only if
all roots of (x^2 - mx + n)(x^2 - nx + m) are positive integers.
-/
def all_roots_positive_integers (m n : ℕ+) : Prop :=
  ∀ x : ℝ, (x^2 - (m : ℝ)*x + (n : ℝ) = 0 ∨ x^2 - (n : ℝ)*x + (m : ℝ) = 0) → (∃ k : ℕ+, x = k)

/-- 
Theorem stating that the only positive integer pairs (m, n) satisfying the condition
that all roots of (x^2 - mx + n)(x^2 - nx + m) are positive integers 
are (4, 4), (5, 6), and (6, 5).
-/
theorem roots_characterization :
  ∀ m n : ℕ+, all_roots_positive_integers m n ↔ 
    ((m = 4 ∧ n = 4) ∨ (m = 5 ∧ n = 6) ∨ (m = 6 ∧ n = 5)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_roots_characterization_l72_7279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_eq_one_twentyeight_forty_l72_7217

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- The set of solutions to the equation d(n^3) = n -/
def solution_set : Set ℕ :=
  {n | n > 0 ∧ num_divisors (n^3) = n}

/-- Theorem stating that the solution set contains only 1, 28, and 40 -/
theorem solution_set_eq_one_twentyeight_forty :
    solution_set = {1, 28, 40} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_eq_one_twentyeight_forty_l72_7217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_one_female_student_l72_7247

/-- Represents the total number of students -/
def total_students : ℕ := 75

/-- Represents the number of male students -/
def male_students : ℕ := 45

/-- Represents the number of female students -/
def female_students : ℕ := 30

/-- Represents the size of the interest group -/
def interest_group_size : ℕ := 5

/-- Represents the number of students to be selected for the experiment -/
def experiment_group_size : ℕ := 2

/-- Theorem stating the probability of selecting exactly one female student -/
theorem probability_one_female_student :
  let male_in_group := (male_students : ℚ) * interest_group_size / total_students
  let female_in_group := (female_students : ℚ) * interest_group_size / total_students
  let total_combinations := Nat.choose interest_group_size experiment_group_size
  let favorable_outcomes := Nat.choose (Int.floor female_in_group).toNat 1 * 
                            Nat.choose (Int.floor male_in_group).toNat 1
  (favorable_outcomes : ℚ) / total_combinations = 3 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_one_female_student_l72_7247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixtieth_pair_is_five_six_l72_7264

/-- Definition of our sequence of pairs -/
def our_sequence : ℕ → ℕ × ℕ :=
  sorry

/-- The sum of elements in the nth diagonal -/
def diagonal_sum (n : ℕ) : ℕ :=
  n + 1

/-- The total number of pairs up to and including the nth diagonal -/
def total_pairs_up_to (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- The theorem stating that the 60th pair in the sequence is (5,6) -/
theorem sixtieth_pair_is_five_six :
  our_sequence 60 = (5, 6) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixtieth_pair_is_five_six_l72_7264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l72_7265

noncomputable def f (x : ℝ) := Real.sin (x + Real.pi/4) + Real.cos (x - Real.pi/4)

theorem f_properties (α β : ℝ) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β ≤ Real.pi/2)
  (h4 : Real.cos (β - α) = 1/2) (h5 : Real.cos (β + α) = -1/2) :
  (∃ T : ℝ, T > 0 ∧ (∀ x, f (x + T) = f x) ∧ 
    (∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y)) ∧
  (∀ x, f x ≥ -2) ∧
  (∃ x, f x = -2) ∧
  (f β = -Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l72_7265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_l72_7239

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  f_zero : f 0 = 1
  f_second_deriv : ∀ x : ℝ, (deriv (deriv f)) x < f x + 1

/-- The solution set of the inequality -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x : ℝ | f x + 1 < 2 * Real.exp x}

/-- The theorem stating the solution set for functions satisfying the given conditions -/
theorem solution_set_characterization (sf : SpecialFunction) :
  SolutionSet sf.f = {x : ℝ | x > 1} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_l72_7239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_proof_l72_7257

/-- Represents a right circular cone -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Calculates the volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * Real.pi * c.radius^2 * c.height

/-- Represents the water tank problem -/
def waterTankProblem (tank : Cone) (waterFillPercentage : ℝ) : Prop :=
  let waterHeight := 30 * (4 : ℝ)^(1/3)
  let waterCone : Cone := { radius := tank.radius * (waterHeight / tank.height), height := waterHeight }
  tank.radius = 20 ∧
  tank.height = 60 ∧
  waterFillPercentage = 0.4 ∧
  coneVolume waterCone = waterFillPercentage * coneVolume tank

theorem water_height_proof (tank : Cone) (waterFillPercentage : ℝ) :
  waterTankProblem tank waterFillPercentage → 
  ∃ (a b : ℕ), a = 30 ∧ b = 4 ∧ (tank.height * (waterFillPercentage)^(1/3) : ℝ) = a * (b : ℝ)^(1/3) :=
by
  sorry

#check water_height_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_proof_l72_7257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cosine_inequality_l72_7243

theorem negation_of_cosine_inequality :
  (¬ ∀ x : ℝ, Real.cos x ≤ 1) ↔ (∃ x : ℝ, Real.cos x > 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cosine_inequality_l72_7243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_last_player_wins_is_one_thirty_first_l72_7222

/-- The probability of getting tails on a fair coin flip. -/
noncomputable def p_tails : ℝ := 1 / 2

/-- The number of players in the game. -/
def num_players : ℕ := 4

/-- The probability that the last player wins the coin flipping game. -/
noncomputable def prob_last_player_wins : ℝ :=
  (p_tails ^ num_players) / (1 - p_tails ^ num_players)

/-- Theorem stating that the probability of the last player winning is 1/31. -/
theorem prob_last_player_wins_is_one_thirty_first :
  prob_last_player_wins = 1 / 31 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_last_player_wins_is_one_thirty_first_l72_7222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_centers_l72_7277

-- Define the triangle ABC
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (80, 0)
def C : ℝ × ℝ := (0, 150)

-- Define the lengths of the sides
def AB : ℝ := 80
def AC : ℝ := 150
def BC : ℝ := 170

-- Define the inscribed circle C₁
noncomputable def C₁ : Set (ℝ × ℝ) := sorry

-- Define the line DE
noncomputable def D : ℝ × ℝ := sorry
noncomputable def E : ℝ × ℝ := sorry
noncomputable def DE : Set (ℝ × ℝ) := sorry

-- Define the line FG
noncomputable def F : ℝ × ℝ := sorry
noncomputable def G : ℝ × ℝ := sorry
noncomputable def FG : Set (ℝ × ℝ) := sorry

-- Define the inscribed circles C₂ and C₃
noncomputable def C₂ : Set (ℝ × ℝ) := sorry
noncomputable def C₃ : Set (ℝ × ℝ) := sorry

-- Define the centers of C₂ and C₃
noncomputable def O₂ : ℝ × ℝ := sorry
noncomputable def O₃ : ℝ × ℝ := sorry

-- Define predicates for perpendicular and tangent
def isPerpendicular (l₁ l₂ : Set (ℝ × ℝ)) : Prop := sorry
def isTangentTo (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) : Prop := sorry

-- State the theorem
theorem distance_between_centers : 
  isPerpendicular DE {(x, y) | x = 0} ∧ isTangentTo DE C₁ ∧ 
  isPerpendicular FG {(x, y) | y = 0} ∧ isTangentTo FG C₁ →
  (O₂.1 - O₃.1)^2 + (O₂.2 - O₃.2)^2 = 10 * 2025 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_centers_l72_7277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l72_7253

/-- The function f(x) = (3x^2 + 8x + 12) / (3x + 4) -/
noncomputable def f (x : ℝ) : ℝ := (3 * x^2 + 8 * x + 12) / (3 * x + 4)

/-- The oblique asymptote function g(x) = x + 4/3 -/
noncomputable def g (x : ℝ) : ℝ := x + 4/3

/-- Theorem: The oblique asymptote of f(x) is g(x) -/
theorem oblique_asymptote_of_f : 
  ∀ ε > 0, ∃ M, ∀ x > M, |f x - g x| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l72_7253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_equivalence_l72_7224

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f(x²+1)
def domain_f_x2plus1 : Set ℝ := Set.Icc (-3) 2

-- Define the domain of f(x-1)
def domain_f_xminus1 : Set ℝ := Set.Icc 2 11

-- State the theorem
theorem domain_equivalence :
  (∀ x ∈ domain_f_x2plus1, f (x^2 + 1) = f (x^2 + 1)) →
  (∀ y ∈ domain_f_xminus1, f (y - 1) = f (y - 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_equivalence_l72_7224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_range_of_t_l72_7223

-- Define the circle
def my_circle (m : ℝ) (x y : ℝ) : Prop := (x + m)^2 + (y - m)^2 = 16

-- Define the ellipse C₁
def my_ellipse (m : ℝ) (x y : ℝ) : Prop := x^2 / m^2 + y^2 / (2*m + 8) = 1

-- Define the hyperbola C₂
def my_hyperbola (m t : ℝ) (x y : ℝ) : Prop := x^2 / (m - t) + y^2 / (m - t - 1) = 1

-- Point M
def point_M : ℝ × ℝ := (1, 3)

-- Theorem for the range of m
theorem range_of_m :
  (∀ m : ℝ, (¬ my_circle m point_M.1 point_M.2 ∧ 
    (∃ x y : ℝ, my_ellipse m x y ∧ x ≠ 0)) ↔ 
    (-4 < m ∧ m < -2) ∨ m > 4) :=
sorry

-- Theorem for the range of t
theorem range_of_t :
  (∀ t : ℝ, (∀ m : ℝ, ¬(∃ x y : ℝ, my_hyperbola m t x y)) →
    (¬(∃ x y : ℝ, my_ellipse m x y ∧ x ≠ 0)) ∧
    (∃ m : ℝ, ¬(∃ x y : ℝ, my_ellipse m x y ∧ x ≠ 0) ∧
      (∃ x y : ℝ, my_hyperbola m t x y))) ↔
    (-4 ≤ t ∧ t ≤ -3) ∨ t ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_range_of_t_l72_7223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_metal_waste_calculation_l72_7244

-- Define the rectangle dimensions
noncomputable def rectangle_width : ℝ := 8
noncomputable def rectangle_height : ℝ := 10

-- Define the circle's radius (half of the shorter side)
noncomputable def circle_radius : ℝ := rectangle_width / 2

-- Define the square's side length
noncomputable def square_side : ℝ := circle_radius * Real.sqrt 2

-- Theorem statement
theorem metal_waste_calculation :
  let rectangle_area := rectangle_width * rectangle_height
  let circle_area := Real.pi * circle_radius^2
  let square_area := square_side^2
  let waste := rectangle_area - square_area
  waste = 48 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_metal_waste_calculation_l72_7244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_and_max_value_l72_7249

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  cosine_law_B : b^2 = a^2 + c^2 - 2*a*c*(Real.cos B)

/-- The given condition for the specific triangle -/
def special_condition (t : Triangle) : Prop :=
  t.a^2 + t.c^2 = t.b^2 + Real.sqrt 2 * t.a * t.c

theorem angle_B_and_max_value (t : Triangle) (h : special_condition t) :
  t.B = π/4 ∧ (∀ s : Triangle, special_condition s → Real.sqrt 2 * (Real.cos s.A) + Real.cos s.C ≤ 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_and_max_value_l72_7249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l72_7281

def M : Set ℤ := {-2, -1, 0, 1, 2}

def N : Set ℤ := {x : ℤ | (x : ℝ)^2 - x - 6 ≥ 0}

theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l72_7281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_trip_time_l72_7213

/-- The shortest time (in minutes) for all students to reach the destination -/
def shortest_time : ℝ := 30.4

/-- The number of students -/
def num_students : ℕ := 12

/-- The time (in hours) to walk from the school to the destination -/
def walk_time : ℝ := 2

/-- The maximum number of students the car can take at a time -/
def car_capacity : ℕ := 4

/-- The speed multiplier of the car compared to walking -/
def car_speed_multiplier : ℝ := 15

theorem optimal_trip_time :
  ∀ (t : ℝ),
    t ≥ shortest_time →
    (∃ (plan : List (List ℕ)),
      (∀ group ∈ plan, group.length ≤ car_capacity) ∧
      (plan.join.length = num_students) ∧
      (∃ (schedule : List ℝ),
        schedule.length = plan.length ∧
        (∀ (i : ℕ) (h : i < schedule.length),
          schedule[i] ≤ walk_time / car_speed_multiplier) ∧
        (schedule.sum ≤ t))) ∧
    (∀ (t' : ℝ),
      t' < shortest_time →
      ¬∃ (plan : List (List ℕ)),
        (∀ group ∈ plan, group.length ≤ car_capacity) ∧
        (plan.join.length = num_students) ∧
        (∃ (schedule : List ℝ),
          schedule.length = plan.length ∧
          (∀ (i : ℕ) (h : i < schedule.length),
            schedule[i] ≤ walk_time / car_speed_multiplier) ∧
          (schedule.sum ≤ t'))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_trip_time_l72_7213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_female_to_male_ratio_is_15_to_7_l72_7288

/-- Represents the number of whales in Ishmael's observation --/
structure WhaleCount where
  total : ℕ
  first_trip_male : ℕ
  second_trip_baby : ℕ
  total_eq : total = 178
  first_male_eq : first_trip_male = 28
  second_baby_eq : second_trip_baby = 8

/-- The ratio of female to male whales on the first trip --/
def female_to_male_ratio (w : WhaleCount) : ℚ :=
  sorry

/-- Theorem stating the ratio of female to male whales on the first trip --/
theorem female_to_male_ratio_is_15_to_7 (w : WhaleCount) : female_to_male_ratio w = 15 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_female_to_male_ratio_is_15_to_7_l72_7288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_lengths_l72_7236

/-- Represents the length of a train platform in meters. -/
noncomputable def P : ℝ := sorry

/-- Represents the length of Train A in meters. -/
noncomputable def L_A : ℝ := sorry

/-- Represents the length of Train B in meters. -/
noncomputable def L_B : ℝ := sorry

/-- The speed of Train A in meters per second. -/
noncomputable def speed_A : ℝ := 180 * 1000 / 3600

/-- The speed of Train B in meters per second. -/
noncomputable def speed_B : ℝ := 240 * 1000 / 3600

/-- Theorem stating that the lengths of Train A and Train B are both 1500 meters. -/
theorem train_lengths : 
  (speed_A * 60 = L_A + P) →
  (speed_B * 45 = L_B + P) →
  (L_A + P = 2 * L_B) →
  (L_A = 1500 ∧ L_B = 1500) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_lengths_l72_7236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_zero_in_range_l72_7274

def a : ℕ → ℤ
  | 0 => 0  -- Add this case to handle Nat.zero
  | 1 => 0
  | n+2 => a ((n+2)/2) + (-1)^(((n+2)*((n+2)+1))/2)

theorem exists_zero_in_range (k : ℕ) (h : k ≥ 2) :
  ∃ n : ℕ, 2^k ≤ n ∧ n < 2^(k+1) ∧ a n = 0 :=
by
  sorry  -- Use 'by' and 'sorry' to skip the proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_zero_in_range_l72_7274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_shop_problem_l72_7283

-- Define the types of fruits
inductive Fruit
| A
| B

-- Define the price and quantity functions
def price : Fruit → ℕ
| Fruit.A => 0  -- Placeholder value, will be proven later
| Fruit.B => 0  -- Placeholder value, will be proven later

def quantity : Fruit → ℕ
| Fruit.A => 0  -- Placeholder value
| Fruit.B => 0  -- Placeholder value

-- Define the profit function
def profit : Fruit → ℕ
| Fruit.A => 4
| Fruit.B => 6

-- Define the total cost function
def total_cost (qa qb : ℕ) : ℕ := price Fruit.A * qa + price Fruit.B * qb

-- Define the total profit function
def total_profit (qa qb : ℕ) : ℕ := profit Fruit.A * qa + profit Fruit.B * qb

-- State the theorem
theorem fruit_shop_problem :
  -- Given conditions
  (total_cost 9 10 = 163) →
  (total_cost 12 8 = 164) →
  -- Conclusion
  (price Fruit.A = 7 ∧ price Fruit.B = 10) ∧
  (∀ m : ℕ, m ≤ 13 →
    (quantity Fruit.B = m ∧ quantity Fruit.A = 2 * m + 4) →
    total_profit (quantity Fruit.A) (quantity Fruit.B) ≥ 160 →
    (m = 11 ∨ m = 12 ∨ m = 13)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_shop_problem_l72_7283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peter_reads_three_times_faster_l72_7267

/-- Represents the reading speed ratio between two people -/
noncomputable def reading_speed_ratio (peter_time kristin_time : ℚ) : ℚ :=
  kristin_time / peter_time

/-- Proves that Peter reads 3 times faster than Kristin given the problem conditions -/
theorem peter_reads_three_times_faster :
  let peter_time : ℚ := 18 -- hours for Peter to read one book
  let kristin_half_books : ℚ := 10 -- half of Kristin's books
  let kristin_half_time : ℚ := 540 -- hours for Kristin to read half her books
  let kristin_time : ℚ := kristin_half_time / kristin_half_books -- hours for Kristin to read one book
  reading_speed_ratio peter_time kristin_time = 3 := by
  -- Unfold the definitions
  unfold reading_speed_ratio
  -- Simplify the expression
  simp
  -- Perform the division
  norm_num
  -- QED
  done


end NUMINAMATH_CALUDE_ERRORFEEDBACK_peter_reads_three_times_faster_l72_7267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l72_7245

/-- The function g(x) is defined as the minimum of three linear functions --/
noncomputable def g (x : ℝ) : ℝ := min (3 * x + 3) (min (x + 2) (-1/2 * x + 8))

/-- The maximum value of g(x) over all real numbers is 6 --/
theorem max_value_of_g : ∃ (M : ℝ), M = 6 ∧ ∀ (x : ℝ), g x ≤ M ∧ ∃ (x₀ : ℝ), g x₀ = M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l72_7245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_qr_length_l72_7235

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  area : ℝ
  altitude : ℝ
  pq : ℝ
  rs : ℝ

/-- Calculates the length of QR in a trapezoid with given properties -/
noncomputable def calculate_qr (t : Trapezoid) : ℝ :=
  24 - Real.sqrt 11 - 2 * Real.sqrt 24

/-- Theorem stating that for a trapezoid with the given properties, 
    the length of QR is 24 - √11 - 2√24 -/
theorem trapezoid_qr_length (t : Trapezoid) 
    (h_area : t.area = 240)
    (h_altitude : t.altitude = 10)
    (h_pq : t.pq = 12)
    (h_rs : t.rs = 22) : 
  calculate_qr t = 24 - Real.sqrt 11 - 2 * Real.sqrt 24 := by
  sorry

#check trapezoid_qr_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_qr_length_l72_7235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_triangle_area_l72_7292

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x - 1/2 * Real.cos (2 * x)

-- Theorem for the minimum value and period of f
theorem f_properties : 
  (∀ x : ℝ, f x ≥ -1) ∧ 
  (∀ x : ℝ, f (x + π) = f x) ∧
  (∀ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) → T ≥ π) := by
  sorry

-- Theorem for the area of triangle ABC
theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  f C = 1 →
  B = π / 6 →
  c = 2 * Real.sqrt 3 →
  A + B + C = π →
  1/2 * b * c * Real.sin A = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_triangle_area_l72_7292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_increasing_f_l72_7296

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (4 - a/2)*x + 5

-- State the theorem
theorem range_of_a_for_increasing_f :
  ∀ a : ℝ, (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) ↔ 
  (a ≥ 6 ∧ a < 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_increasing_f_l72_7296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subject_pass_difference_l72_7268

theorem subject_pass_difference (total : ℕ) (english math science : ℕ) 
  (eng_math eng_sci math_sci : ℕ) (all_three : ℕ) :
  total = 60 →
  english = 30 →
  math = 35 →
  science = 28 →
  eng_math = 12 →
  eng_sci = 11 →
  math_sci = 15 →
  all_three = 6 →
  (math - eng_math - math_sci + all_three) + 
  (science - eng_sci - math_sci + all_three) = 
  (english - eng_math - eng_sci + all_three) + 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subject_pass_difference_l72_7268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_not_containing_family_elements_l72_7207

/-- Given a set X of n elements and a family A of 3-element subsets of X
    where any two subsets in A share at most one common element,
    there exists a subset M of X with at least ⌊√(2n)⌋ elements
    that does not contain any subset from A. -/
theorem subset_not_containing_family_elements (n : ℕ) (X : Finset ℕ) (A : Set (Finset ℕ)) 
  (h_X : X = Finset.range n)
  (h_A : ∀ a ∈ A, a.card = 3 ∧ a ⊆ X)
  (h_A_intersection : ∀ a b : Finset ℕ, a ∈ A → b ∈ A → a ≠ b → (a ∩ b).card ≤ 1) :
  ∃ M : Finset ℕ, M ⊆ X ∧ 
    M.card ≥ Int.floor (Real.sqrt (2 * n)) ∧
    ∀ a ∈ A, ¬(a ⊆ M) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_not_containing_family_elements_l72_7207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_zero_given_conditions_l72_7263

/-- Represents a square with side length and area -/
structure Square where
  side : ℝ
  area : ℝ
  area_eq : area = side * side

/-- The probability that a random point in square_a is not in square_b or square_c -/
noncomputable def probability_not_in_squares (square_a square_b square_c : Square) : ℝ :=
  1 - (square_b.area + square_c.area) / square_a.area

/-- Theorem stating that the probability of a point in square A not being in square B or C is 0 -/
theorem probability_zero_given_conditions 
  (square_a square_b square_c : Square)
  (h_a_area : square_a.area = 100)
  (h_b_perim : 4 * square_b.side = 40)
  (h_c_perim : 4 * square_c.side = 24)
  (h_c_in_a : square_c.area ≤ square_a.area)
  (h_b_eq_a : square_b.area = square_a.area) :
  probability_not_in_squares square_a square_b square_c = 0 := by
  sorry

#check probability_zero_given_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_zero_given_conditions_l72_7263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_commutator_sum_zero_l72_7258

variable {α : Type*} [Ring α]
variable {n : Type*} [Fintype n]

def matrix_commutator (A B : Matrix n n α) : Matrix n n α :=
  A * B - B * A

theorem cyclic_commutator_sum_zero 
  (A B C : Matrix n n α) : 
  matrix_commutator A (matrix_commutator B C) + 
  matrix_commutator B (matrix_commutator C A) + 
  matrix_commutator C (matrix_commutator A B) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_commutator_sum_zero_l72_7258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_l72_7225

/-- The parabola defined by the equation y = -(x-2)^2 + 5 has its vertex at (2,5). -/
theorem parabola_vertex (x y : ℝ) : 
  y = -(x-2)^2 + 5 → (2, 5) = (x, y) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_l72_7225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_ball_radius_l72_7276

theorem larger_ball_radius (small_radius : ℝ) (num_small_balls : ℕ) :
  small_radius = 3 →
  num_small_balls = 9 →
  (4 / 3 * Real.pi * small_radius^3 * (num_small_balls : ℝ)) = (4 / 3 * Real.pi * 9^3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_ball_radius_l72_7276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_greater_than_q_l72_7286

theorem p_greater_than_q : Real.sqrt 2 > Real.sqrt 6 - Real.sqrt 2 := by
  sorry

#check p_greater_than_q

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_greater_than_q_l72_7286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_more_than_five_nearest_l72_7246

-- Define a type for cities
def City := ℝ × ℝ

-- Define a function to calculate distance between two cities
noncomputable def distance (a b : City) : ℝ := Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

-- State the theorem
theorem no_more_than_five_nearest (cities : Set City) :
  (∀ a b c d : City, a ∈ cities → b ∈ cities → c ∈ cities → d ∈ cities → 
    a ≠ b → c ≠ d → (a, b) ≠ (c, d) → distance a b ≠ distance c d) →
  ¬∃ center : City, ∃ neighbors : Set City, 
    center ∈ cities ∧ 
    neighbors ⊆ cities ∧ 
    neighbors.ncard > 5 ∧
    (∀ n ∈ neighbors, ∀ c ∈ cities, c ≠ center → distance n center ≤ distance n c) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_more_than_five_nearest_l72_7246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_theta_l72_7266

theorem sin_double_theta (θ : ℝ) (h : Real.cos θ + Real.sin θ = 3/2) : Real.sin (2 * θ) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_theta_l72_7266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curve_and_line_l72_7229

/-- The curve on which point P moves -/
noncomputable def curve (x : ℝ) : ℝ := x * Real.exp (-2 * x)

/-- The line on which point Q moves -/
def line (x : ℝ) : ℝ := x + 2

/-- The minimum distance between points on the curve and the line -/
noncomputable def min_distance : ℝ := Real.sqrt 2

/-- Theorem stating that the minimum distance between a point on the curve and a point on the line is √2 -/
theorem min_distance_between_curve_and_line :
  ∃ (x₁ x₂ : ℝ), ∀ (y₁ y₂ : ℝ), y₁ = curve x₁ → y₂ = line x₂ →
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ≥ min_distance := by
  sorry

#check min_distance_between_curve_and_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curve_and_line_l72_7229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_range_l72_7261

theorem sin_cos_range (x y : ℝ) (h : 2 * Real.sin x ^ 2 + Real.cos y ^ 2 = 1) :
  ∃ (z : ℝ), Real.sin x ^ 2 + Real.cos y ^ 2 = z ∧ 1/2 ≤ z ∧ z ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_range_l72_7261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_layers_correct_l72_7290

/-- Given a stack of steel pipes where:
  * Each layer has one more pipe than the layer below it
  * The bottom layer has m pipes
  * The top layer has n pipes
  This function calculates the number of layers in the stack -/
def num_layers (m n : ℕ) : ℕ := n - m + 1

/-- Function to calculate the number of pipes in a given layer -/
def pipe_count_at_layer (m k : ℕ) : ℕ := m + k

/-- Theorem stating that num_layers correctly calculates the number of layers
    in a stack of steel pipes with the given properties -/
theorem num_layers_correct (m n : ℕ) (h : m ≤ n) :
  ∀ k : ℕ, k ∈ Finset.range (num_layers m n) → 
  (m + k : ℕ) = (pipe_count_at_layer m k) ∧ 
  (pipe_count_at_layer m (num_layers m n - 1) = n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_layers_correct_l72_7290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l72_7271

noncomputable def f (x : Real) := 6 * (Real.cos x) ^ 2 - 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2

theorem f_properties :
  (∃ T : Real, T > 0 ∧ ∀ x, f (x + T) = f x ∧ ∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S) ∧
  (∀ y ∈ Set.range f, 5 - 2 * Real.sqrt 3 ≤ y ∧ y ≤ 5 + 2 * Real.sqrt 3) ∧
  (∀ A B C : Real, 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π ∧ f B = 2 → B = π / 3) :=
by sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l72_7271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cross_sectional_area_cone_l72_7240

/-- Given a cone with generating line length 3 and angle π/3 between the generating line and the axis,
    the maximum cross-sectional area through the vertex is 9/2. -/
theorem max_cross_sectional_area_cone (l : ℝ) (θ : ℝ) :
  l = 3 →
  θ = π / 3 →
  (∃ (S : ℝ), S = (1 / 2) * l^2 * Real.sin θ ∧
    ∀ (φ : ℝ), 0 ≤ φ ∧ φ ≤ 2*π/3 →
      S ≥ (1 / 2) * l^2 * Real.sin φ) →
  (9 : ℝ) / 2 = (1 / 2) * l^2 * 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cross_sectional_area_cone_l72_7240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_side_ratio_bound_l72_7202

/-- A quadrilateral is represented by its four side lengths -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  all_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d

/-- The ratio of the sum of squares of two sides to the sum of squares of the other two sides -/
noncomputable def side_ratio (q : Quadrilateral) : ℝ :=
  (q.a^2 + q.b^2) / (q.c^2 + q.d^2)

/-- The theorem stating that 1 is the greatest lower bound for the side ratio of any quadrilateral -/
theorem quadrilateral_side_ratio_bound :
  ∀ ε > 0, ∃ q : Quadrilateral, side_ratio q < 1 + ε ∧ ∀ q' : Quadrilateral, 1 ≤ side_ratio q' :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_side_ratio_bound_l72_7202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_degree_l72_7231

noncomputable def expr1 (x : ℝ) : ℝ := x^5
noncomputable def expr2 (x : ℝ) : ℝ := x^2 + 1/x^2
noncomputable def expr3 (x : ℝ) : ℝ := 1 + 2/x + 3/x^2

noncomputable def product (x : ℝ) : ℝ := expr1 x * expr2 x * expr3 x

theorem product_degree : 
  ∃ (p : Polynomial ℝ), (∀ x : ℝ, x ≠ 0 → Polynomial.eval x p = product x) ∧ 
  Polynomial.degree p = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_degree_l72_7231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_focus_l72_7284

/-- The x-coordinate of the right focus of an ellipse with equation x^2/a^2 + y^2/b^2 = 1 -/
noncomputable def ellipse_right_focus (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2)

/-- The x-coordinate of the focus of a parabola with equation y^2 = 2px -/
noncomputable def parabola_focus (p : ℝ) : ℝ := p / 2

theorem parabola_ellipse_focus (p : ℝ) : 
  parabola_focus p = ellipse_right_focus (Real.sqrt 6) (Real.sqrt 2) → p = 4 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_focus_l72_7284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_101_equals_52_l72_7254

def a (n : ℕ) : ℚ :=
  match n with
  | 0 => 2
  | n + 1 => a n + 1/2

theorem a_101_equals_52 : a 100 = 52 := by
  show a 100 = 52
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_101_equals_52_l72_7254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_defective_bulb_l72_7226

/-- Given a box of bulbs, calculates the probability of selecting at least one defective bulb -/
def prob_at_least_one_defective (total : ℕ) (defective : ℕ) (chosen : ℕ) : ℚ :=
  1 - (Nat.choose (total - defective) chosen : ℚ) / (Nat.choose total chosen : ℚ)

/-- Theorem: The probability of selecting at least one defective bulb from a box of 20 bulbs 
    with 4 defective ones, when choosing 2 bulbs at random, is 7/19 -/
theorem prob_defective_bulb : 
  prob_at_least_one_defective 20 4 2 = 7 / 19 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_defective_bulb_l72_7226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_equal_implies_a_value_l72_7278

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.log x
def g (a x : ℝ) : ℝ := a * x^2 - a

-- State the theorem
theorem tangent_lines_equal_implies_a_value (a : ℝ) :
  (∀ h : ℝ, h ≠ 0 → (f (1 + h) - f 1) / h = (g a (1 + h) - g a 1) / h) →
  a = 1/2 :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_equal_implies_a_value_l72_7278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_divisor_digit_sum_l72_7282

def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem greatest_prime_divisor_digit_sum (n : Nat) (h : n = 32767) : 
  ∃ p : Nat, 
    Nat.Prime p ∧ 
    p ∣ n ∧ 
    (∀ q : Nat, Nat.Prime q → q ∣ n → q ≤ p) ∧
    (sum_of_digits p = 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_divisor_digit_sum_l72_7282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_probabilities_l72_7210

/-- Probability of shooter A hitting the target in a single shot -/
def prob_A : ℚ := 1/2

/-- Probability of shooter B hitting the target in a single shot -/
def prob_B : ℚ := 1/3

/-- Probability of hitting the target when each shooter takes one shot -/
def prob_one_shot : ℚ := 1 - (1 - prob_A) * (1 - prob_B)

/-- Probability of hitting the target when each shooter takes two shots and at least three shots hit -/
def prob_two_shots : ℚ :=
  (Nat.choose 2 2) * (prob_A^2) * (Nat.choose 2 1) * (prob_B) * (1 - prob_B) +
  (Nat.choose 2 1) * (prob_A) * (1 - prob_A) * (Nat.choose 2 2) * (prob_B^2) +
  (Nat.choose 2 2) * (prob_A^2) * (Nat.choose 2 2) * (prob_B^2)

theorem shooting_probabilities :
  prob_one_shot = 2/3 ∧ prob_two_shots = 7/36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_probabilities_l72_7210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_average_l72_7289

theorem consecutive_integers_average (x : ℤ) (y : ℤ) (h : y = 6*x + 15) :
  (y + (y+1) + (y+2) + (y+3) + (y+4) + (y+5)) / 6 = 6 * x + (35 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_average_l72_7289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_for_real_root_l72_7228

/-- A polynomial of degree 2n with coefficients in [100, 101] -/
def ConstrainedPolynomial (n : ℕ) := 
  {p : Polynomial ℝ // p.degree = 2 * n ∧ ∀ i, 100 ≤ p.coeff i ∧ p.coeff i ≤ 101}

/-- The existence of a real root for a constrained polynomial -/
def HasRealRoot (p : ConstrainedPolynomial n) : Prop :=
  ∃ x : ℝ, p.val.eval x = 0

/-- The minimum n for which a constrained polynomial can have a real root -/
theorem min_n_for_real_root :
  (∃ (p : ConstrainedPolynomial 100), HasRealRoot p) ∧
  (∀ n < 100, ∀ (p : ConstrainedPolynomial n), ¬HasRealRoot p) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_for_real_root_l72_7228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_calculation_train_crossing_time_specific_l72_7227

/-- The time taken for a train to cross a fixed point -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed : ℝ) : ℝ :=
  train_length / train_speed

/-- Theorem: The time taken for a train to cross a fixed point is approximately equal to its length divided by its speed -/
theorem train_crossing_time_calculation (train_length : ℝ) (train_speed : ℝ) 
  (h1 : train_length = 284) 
  (h2 : train_speed = 56.8) :
  abs (train_crossing_time train_length train_speed - 5) < 0.01 := by
  sorry

/-- Corollary: The time taken for the specific train in the problem is approximately 5 seconds -/
theorem train_crossing_time_specific :
  abs (train_crossing_time 284 56.8 - 5) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_calculation_train_crossing_time_specific_l72_7227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_self_root_l72_7298

theorem unique_self_root : ∀ x : ℝ, (x ≥ 0 ∧ Real.sqrt x = x ∧ x^(1/3) = x) ↔ x = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_self_root_l72_7298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_n_for_product_l72_7256

def m : ℕ := 30030

def M : Set ℕ := {d : ℕ | d ∣ m ∧ (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ d = p * q)}

theorem minimal_n_for_product (n : ℕ) : n = 11 ↔ 
  (∀ S : Finset ℕ, ↑S ⊆ M → S.card = n → 
    ∃ a b c : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a * b * c = m) ∧
  (∀ k : ℕ, k < 11 → 
    ∃ T : Finset ℕ, ↑T ⊆ M ∧ T.card = k ∧
      ∀ a b c : ℕ, a ∈ T → b ∈ T → c ∈ T → a * b * c ≠ m) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_n_for_product_l72_7256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_roots_of_equation_l72_7201

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define the equation
def equation (x : ℚ) : Prop :=
  9 * x^2 - 8 * (floor (x : ℝ)) = 1

-- Statement to prove
theorem rational_roots_of_equation :
  {x : ℚ | equation x} = {1, 1/3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_roots_of_equation_l72_7201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_form_triangle_l72_7287

/-- Triangle structure representing a triangle with side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0

/-- Triangle Inequality Theorem: A set of three line segments can form a triangle if and only if 
    the sum of the lengths of any two sides is greater than the length of the remaining side. -/
axiom triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a + b > c ∧ b + c > a ∧ c + a > b) ↔ (∃ t : Triangle, t.a = a ∧ t.b = b ∧ t.c = c)

/-- Theorem: The set of line segments with lengths 3, 5, and 9 cannot form a triangle. -/
theorem cannot_form_triangle : ¬ (∃ t : Triangle, t.a = 3 ∧ t.b = 5 ∧ t.c = 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_form_triangle_l72_7287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_sum_theorem_l72_7295

theorem final_sum_theorem (X m n : ℝ) (h : m + n = X) :
  (3 * (m - 4) + 5) + (3 * (n - 4) + 5) = 3 * X - 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_sum_theorem_l72_7295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workshop_salary_problem_l72_7241

theorem workshop_salary_problem (total_workers : ℕ) (total_avg_salary : ℚ) 
  (technicians : ℕ) (tech_avg_salary : ℚ) :
  total_workers = 14 →
  total_avg_salary = 8000 →
  technicians = 7 →
  tech_avg_salary = 10000 →
  let non_tech_workers := total_workers - technicians
  let total_salary := total_avg_salary * total_workers
  let tech_total_salary := tech_avg_salary * technicians
  let non_tech_total_salary := total_salary - tech_total_salary
  let non_tech_avg_salary := non_tech_total_salary / non_tech_workers
  non_tech_avg_salary = 6000 := by
  intro h1 h2 h3 h4
  -- The proof steps would go here
  sorry

#check workshop_salary_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_workshop_salary_problem_l72_7241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l72_7212

/-- Calculates the time taken for a train to pass a telegraph post -/
noncomputable def time_to_pass (length : ℝ) (speed : ℝ) : ℝ :=
  length / speed

/-- Converts speed from km/h to m/s -/
noncomputable def kmph_to_mps (speed : ℝ) : ℝ :=
  speed * (1000 / 3600)

theorem train_passing_time (length_A length_B speed_A speed_B : ℝ) 
  (h1 : length_A = 40)
  (h2 : length_B = 60)
  (h3 : speed_A = 36)
  (h4 : speed_B = 45) :
  time_to_pass length_A (kmph_to_mps speed_A) + time_to_pass length_B (kmph_to_mps speed_B) = 8.8 :=
by sorry

#check train_passing_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l72_7212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_value_l72_7251

noncomputable def f (x a : ℝ) : ℝ := Real.sin (x + Real.pi/6) + Real.sin (x - Real.pi/6) + Real.cos x + a

theorem triangle_side_value (a : ℝ) :
  (∃ (x : ℝ), f x a = 1) →
  (∀ (x : ℝ), f x a ≤ 1) →
  let A : ℝ := Real.pi/3
  let B : ℝ := Real.pi - A - Real.pi/4
  let c : ℝ := 2
  (f A a = 1) →
  let b : ℝ := c * Real.sin B / Real.sin (Real.pi/4)
  b = Real.sqrt 3 + 1 := by
  sorry

#check triangle_side_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_value_l72_7251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l72_7252

-- Define points A and B
noncomputable def A : ℝ × ℝ := (1, -2)
noncomputable def B : ℝ × ℝ := (5, 6)

-- Define the midpoint M of AB
noncomputable def M : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define a line l
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the property of line l passing through point M
def passes_through (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

-- Define the property of line l having equal intercepts on both axes
def equal_intercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c / l.a = -l.c / l.b

-- Theorem statement
theorem line_equation :
  ∃ (l : Line), passes_through l M ∧ equal_intercepts l ∧
  ((l.a = 2 ∧ l.b = -3 ∧ l.c = 0) ∨ (l.a = 1 ∧ l.b = 1 ∧ l.c = -5)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l72_7252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_power_function_implies_m_equals_one_l72_7259

/-- A function f is even if f(-x) = f(x) for all x in the domain of f -/
def IsEvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The power function f(x) = (m^2 - 3m + 3) * x^(m+1) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  (m^2 - 3*m + 3) * x^(m+1)

/-- If f is an even function, then m = 1 -/
theorem even_power_function_implies_m_equals_one :
  ∀ m : ℝ, IsEvenFunction (f m) → m = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_power_function_implies_m_equals_one_l72_7259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_properties_l72_7215

noncomputable def z (m : ℝ) : ℂ := (m * Complex.I) / (2 - Complex.I)

theorem z_properties (m : ℝ) (h : 0 < m ∧ m ≤ 5) :
  (∃ (x y : ℝ), z m = Complex.mk x y ∧ x < 0 ∧ y > 0) ∧
  (∀ (m' : ℝ), 0 < m' ∧ m' ≤ 5 → Complex.abs (z m') ≤ Real.sqrt 5) ∧
  (∃ (m' : ℝ), 0 < m' ∧ m' ≤ 5 ∧ Complex.abs (z m') = Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_properties_l72_7215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cauchy_problem_solution_l72_7206

-- Define the differential equation
def diff_eq (x : ℝ → ℝ) : Prop :=
  ∀ t, (deriv (deriv x)) t + 4 * x t = Real.cos (2 * t)

-- Define the initial conditions
def initial_conditions (x : ℝ → ℝ) : Prop :=
  x 0 = 1 ∧ (deriv x) 0 = -1

-- Define the solution function
noncomputable def solution (t : ℝ) : ℝ :=
  (1/4) * t * Real.sin (2*t) + Real.cos (2*t) - (1/2) * Real.sin (2*t)

-- State the theorem
theorem cauchy_problem_solution :
  diff_eq solution ∧ initial_conditions solution := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cauchy_problem_solution_l72_7206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sign_l72_7275

theorem alternating_sign (n : ℕ) (F : ℕ → ℝ) (h : ∀ m : ℕ, F m > 0) :
  ((-1 : ℝ)^n * F n > 0) ↔ Even n :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sign_l72_7275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beverage_price_system_l72_7219

/-- Represents the price of a small bottle in cents -/
def small_price : ℝ → ℝ := λ x => x

/-- Represents the price of a medium bottle in cents -/
def medium_price : ℝ → ℝ := λ x => 2 * x - 20

/-- Represents the price of a large bottle in cents -/
def large_price : ℝ → ℝ := λ x => x

theorem beverage_price_system (x y : ℝ) :
  (small_price x + medium_price x + y = 96) ∧
  (y = medium_price x + small_price x + 40) →
  (3 * x + y = 98 ∧ y - 3 * x = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beverage_price_system_l72_7219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_signal_post_l72_7255

/-- Calculates the time (in seconds) it takes for a train to cross a signal post -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  train_length / train_speed_mps

/-- Theorem: A train 150 meters long traveling at 36 km/h takes 15 seconds to cross a signal post -/
theorem train_crossing_signal_post :
  train_crossing_time 150 36 = 15 := by
  -- Unfold the definition of train_crossing_time
  unfold train_crossing_time
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_signal_post_l72_7255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_tripling_fraction_l72_7238

theorem unique_tripling_fraction :
  ∀ a b : ℕ,
  0 < a ∧ 0 < b →
  Nat.Coprime a b →
  (a + 12 : ℚ) / (b + 12) = 3 * (a / b) →
  a = 2 ∧ b = 9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_tripling_fraction_l72_7238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l72_7218

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi - x/2) * Real.cos (x/2) + Real.sqrt 3 * (Real.cos (x/2))^2

theorem f_properties :
  ∃ (T : ℝ),
    (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), 0 < T' → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
    T = 2 * Real.pi ∧
    (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi) 0 → f x ≤ Real.sqrt 3) ∧
    (∃ (x : ℝ), x ∈ Set.Icc (-Real.pi) 0 ∧ f x = Real.sqrt 3) ∧
    (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi) 0 → -1 + Real.sqrt 3 / 2 ≤ f x) ∧
    (∃ (x : ℝ), x ∈ Set.Icc (-Real.pi) 0 ∧ f x = -1 + Real.sqrt 3 / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l72_7218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_translation_theorem_l72_7270

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Angle of inclination of a line in degrees -/
noncomputable def angle_of_inclination (l : Line) : ℝ :=
  (Real.arctan l.slope) * (180 / Real.pi) + 180

/-- Translation of a line by a vector -/
def translate (l : Line) (dx dy : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + dy - l.slope * dx }

/-- The main theorem stating that if a line coincides with itself after
    translation by (-√3, 1), its angle of inclination is 150° -/
theorem line_translation_theorem (l : Line) :
  translate l (-Real.sqrt 3) 1 = l →
  angle_of_inclination l = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_translation_theorem_l72_7270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cloth_length_calculation_l72_7234

/-- The length of cloth colored by a group of women in a given time -/
def clothLength (women : ℕ) (days : ℕ) (length : ℝ) : Prop :=
  (women : ℝ) * days = length

theorem cloth_length_calculation (x : ℝ) :
  clothLength 6 3 x →
  clothLength 5 4 200 →
  x = 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cloth_length_calculation_l72_7234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_subsequence_exists_l72_7248

/-- Given an infinite sequence of positive integers where each prime divides only finitely many terms,
    there exists a subsequence where any two terms are coprime. -/
theorem coprime_subsequence_exists (a : ℕ → ℕ+) 
    (h : ∀ p : ℕ, Nat.Prime p → Set.Finite {i : ℕ | p ∣ (a i).val}) :
  ∃ f : ℕ → ℕ, StrictMono f ∧ ∀ m n : ℕ, m ≠ n → Nat.Coprime (a (f m)).val (a (f n)).val :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_subsequence_exists_l72_7248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_spheres_l72_7272

def sphere_center_1 : ℝ × ℝ × ℝ := (5, -3, 10)
def sphere_radius_1 : ℝ := 24

def sphere_center_2 : ℝ × ℝ × ℝ := (-7, 15, -20)
def sphere_radius_2 : ℝ := 44

theorem max_distance_between_spheres :
  ∃ (p₁ p₂ : ℝ × ℝ × ℝ),
    (‖p₁ - sphere_center_1‖ = sphere_radius_1) ∧
    (‖p₂ - sphere_center_2‖ = sphere_radius_2) ∧
    (∀ (q₁ q₂ : ℝ × ℝ × ℝ),
      (‖q₁ - sphere_center_1‖ = sphere_radius_1) →
      (‖q₂ - sphere_center_2‖ = sphere_radius_2) →
      ‖q₁ - q₂‖ ≤ ‖p₁ - p₂‖) ∧
    ‖p₁ - p₂‖ = 105 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_spheres_l72_7272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_propositions_l72_7280

-- Define the basic geometric entities
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the geometric relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem geometric_propositions :
  (∀ (a b : Line) (α : Plane),
    perpendicular_line_plane a α → perpendicular_line_plane b α → parallel_lines a b) ∧
  (∀ (c : Line) (α β : Plane),
    perpendicular_line_plane c α → perpendicular_line_plane c β → parallel_planes α β) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_propositions_l72_7280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_conditions_l72_7205

theorem quadratic_roots_conditions (a b : ℤ) : 
  (a^2 + 4*b > 12) ∧ 
  (a^2 + 4*b = 12*a - 8) ∧ 
  (a^2 + 4*b < 8*a + 4) := by
  sorry

#check quadratic_roots_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_conditions_l72_7205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l72_7285

noncomputable def simple_interest_rate (principal : ℝ) (final_amount : ℝ) (time : ℝ) : ℝ :=
  (final_amount - principal) / (principal * time)

theorem interest_rate_calculation :
  let principal : ℝ := 12500
  let final_amount : ℝ := 15500
  let time : ℝ := 4
  simple_interest_rate principal final_amount time = 0.06 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l72_7285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_fraction_l72_7216

-- Define the fractions
noncomputable def fraction_A (a : ℝ) := (a + 1) / (a^2 - 1)
noncomputable def fraction_B (a b c : ℝ) := (4*a) / (6*b*c^2)
noncomputable def fraction_C (a : ℝ) := (2*a) / (2 - a)
noncomputable def fraction_D (a b : ℝ) := (a + b) / (a^2 + a*b)

-- Define what it means for a fraction to be in simplest form
def is_simplest_form (n d : ℝ) : Prop :=
  ∀ k : ℝ, k ≠ 0 → (∃ m : ℤ, n / k = m) ∧ (∃ m : ℤ, d / k = m) → k = 1 ∨ k = -1

-- State the theorem
theorem simplest_fraction :
  ∀ a b c : ℝ, a ≠ 2 → a ≠ -1 → a ≠ 1 → b ≠ 0 → c ≠ 0 →
  (¬ is_simplest_form (a + 1) (a^2 - 1)) ∧
  (¬ is_simplest_form (4*a) (6*b*c^2)) ∧
  (is_simplest_form (2*a) (2 - a)) ∧
  (¬ is_simplest_form (a + b) (a^2 + a*b)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_fraction_l72_7216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_outer_boundary_diameter_l72_7299

/-- The diameter of the outer boundary of a circular park with a central pond, seating area, and jogging path. -/
noncomputable def outer_boundary_diameter (pond_diameter seating_width path_width : ℝ) : ℝ :=
  2 * (pond_diameter / 2 + seating_width + path_width)

/-- Theorem: The diameter of the outer boundary of a circular park with a central pond of diameter 18 feet, 
    surrounded by a 10-foot wide seating area and a 7-foot wide jogging path, is 52 feet. -/
theorem park_outer_boundary_diameter : 
  outer_boundary_diameter 18 10 7 = 52 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_outer_boundary_diameter_l72_7299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_entertainment_time_l72_7214

/-- The number of hours Mike watches TV per day -/
def tv_hours_per_day : ℕ := 4

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of days Mike plays video games per week -/
def gaming_days_per_week : ℕ := 3

/-- The ratio of video game playing time to TV watching time -/
def gaming_to_tv_ratio : ℚ := 1/2

theorem total_entertainment_time :
  tv_hours_per_day * days_per_week + 
  (tv_hours_per_day * gaming_to_tv_ratio).floor * gaming_days_per_week = 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_entertainment_time_l72_7214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_speed_approx_80_l72_7233

/-- Calculates the speed of the second train given the parameters of two trains passing each other --/
noncomputable def calculate_second_train_speed (length1 : ℝ) (speed1 : ℝ) (length2 : ℝ) (time : ℝ) : ℝ :=
  let total_distance := length1 + length2
  let speed1_ms := speed1 * (1000 / 3600)
  let relative_speed := total_distance / time
  let speed2_ms := relative_speed - speed1_ms
  speed2_ms * (3600 / 1000)

/-- Theorem stating that given the specific conditions, the speed of the second train is approximately 79.98 km/h --/
theorem second_train_speed_approx_80 :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |calculate_second_train_speed 270 120 230.04 9 - 79.98| < ε :=
by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculate_second_train_speed 270 120 230.04 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_speed_approx_80_l72_7233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l72_7293

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 4)

theorem f_satisfies_conditions :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ 
    ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (x y : ℝ), 0 ≤ x ∧ x < y ∧ y ≤ Real.pi / 4 → f x < f y) ∧
  (∃ (x₀ : ℝ), ∀ (x : ℝ), f x ≤ f x₀ ∧ f x₀ = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l72_7293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_wheel_radius_approximation_wheel_radius_increase_l72_7260

/-- Represents the properties of a vehicle's wheels and trip measurements -/
structure WheelMeasurements where
  original_radius : ℝ
  original_distance : ℝ
  new_distance : ℝ
  inches_per_mile : ℝ

/-- Calculates the new radius of the wheels based on the given measurements -/
noncomputable def calculate_new_radius (w : WheelMeasurements) : ℝ :=
  (w.original_distance * w.original_radius * w.inches_per_mile) / (2 * Real.pi * w.new_distance)

/-- Theorem stating that under the given conditions, the new wheel radius is approximately 12.49 inches -/
theorem new_wheel_radius_approximation (w : WheelMeasurements)
  (h1 : w.original_radius = 12)
  (h2 : w.original_distance = 600)
  (h3 : w.new_distance = 585)
  (h4 : w.inches_per_mile = 63360) :
  ∃ ε > 0, |calculate_new_radius w - 12.49| < ε := by
  sorry

/-- Theorem stating that the increase in wheel radius is approximately 0.49 inches -/
theorem wheel_radius_increase (w : WheelMeasurements)
  (h1 : w.original_radius = 12)
  (h2 : w.original_distance = 600)
  (h3 : w.new_distance = 585)
  (h4 : w.inches_per_mile = 63360) :
  ∃ ε > 0, |calculate_new_radius w - w.original_radius - 0.49| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_wheel_radius_approximation_wheel_radius_increase_l72_7260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_distance_is_48_l72_7262

/-- Represents the swimming scenario with given parameters -/
structure SwimmingScenario where
  upstream_time : ℚ
  downstream_time : ℚ
  upstream_distance : ℚ
  still_water_speed : ℚ

/-- Calculates the downstream distance given a swimming scenario -/
def downstream_distance (s : SwimmingScenario) : ℚ :=
  let upstream_speed := s.upstream_distance / s.upstream_time
  let stream_speed := s.still_water_speed - upstream_speed
  let downstream_speed := s.still_water_speed + stream_speed
  downstream_speed * s.downstream_time

/-- Theorem stating that the downstream distance is 48 km for the given scenario -/
theorem downstream_distance_is_48 :
  let scenario : SwimmingScenario := {
    upstream_time := 3,
    downstream_time := 3,
    upstream_distance := 18,
    still_water_speed := 11
  }
  downstream_distance scenario = 48 := by
  -- Proof goes here
  sorry

#eval downstream_distance {
  upstream_time := 3,
  downstream_time := 3,
  upstream_distance := 18,
  still_water_speed := 11
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_distance_is_48_l72_7262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l72_7250

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 * a - 5) ^ x

-- State the theorem
theorem decreasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a y < f a x) →
  (5 / 2 < a ∧ a < 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l72_7250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l72_7273

-- Define the inequality
def inequality (x : ℝ) : Prop := (1 : ℝ) / (x - 2) ≤ 1

-- Define the solution set
def solution_set : Set ℝ := Set.Iic 2 ∪ Set.Ici 3

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | inequality x} = solution_set :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l72_7273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_union_when_a_is_one_intersection_empty_iff_a_in_range_l72_7294

def A : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}
def B (a : ℝ) : Set ℝ := {x | 2*a < x ∧ x < a+4}

theorem intersection_and_union_when_a_is_one :
  (A ∩ B 1 = {x | 3 ≤ x ∧ x < 5}) ∧
  (A ∪ B 1 = {x | 2 < x ∧ x ≤ 7}) := by sorry

theorem intersection_empty_iff_a_in_range :
  ∀ a : ℝ, A ∩ B a = ∅ ↔ a ∈ Set.Iic (-1) ∪ Set.Ico (7/2) 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_union_when_a_is_one_intersection_empty_iff_a_in_range_l72_7294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_functions_l72_7291

/-- A function satisfying the given functional equation. -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x > 0 → y > 0 → (x + x⁻¹) * f y = f (x * y) + f (y * x⁻¹)

/-- The theorem stating the form of functions satisfying the equation. -/
theorem characterize_functions (f : ℝ → ℝ) (h : SatisfiesEquation f) :
    ∃ C₁ C₂ : ℝ, ∀ x : ℝ, x > 0 → f x = C₁ * x + C₂ * x⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_functions_l72_7291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_circle_l72_7208

/-- The line l₁ with equation y = -3/2 * x + b is tangent to the parabola x² = -16/3 * y at point P -/
def is_tangent_line (b : ℝ) (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  y = -3/2 * x + b ∧ x^2 = -16/3 * y ∧ 
  ∀ x' y', y' = -3/2 * x' + b → (x' = x ∨ x'^2 ≠ -16/3 * y')

/-- The circle C with equation (x+1)² + (y+2)² = 25 -/
def circle_C (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (x + 1)^2 + (y + 2)^2 = 25

/-- A line l₂ passes through point P and is tangent to circle C -/
def is_tangent_to_circle (P : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  l P.1 P.2 ∧ 
  (∃ Q : ℝ × ℝ, Q ≠ P ∧ l Q.1 Q.2 ∧ circle_C Q) ∧
  (∀ R : ℝ × ℝ, l R.1 R.2 ∧ circle_C R → R = P)

/-- The main theorem -/
theorem tangent_line_and_circle :
  ∃ (b : ℝ) (P : ℝ × ℝ) (l₂ : ℝ → ℝ → Prop),
    is_tangent_line b P ∧
    is_tangent_to_circle P l₂ ∧
    b = 3 ∧
    P = (4, -3) ∧
    (∀ x y, l₂ x y ↔ (x = 4 ∨ 12*x - 5*y - 63 = 0)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_circle_l72_7208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_bound_l72_7209

/-- The eccentricity of an ellipse -/
noncomputable def ellipse_eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

/-- The eccentricity of a hyperbola -/
noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + b^2 / a^2)

/-- The focal distance of an ellipse or hyperbola -/
def focal_distance (a e : ℝ) : ℝ := a * e

theorem eccentricity_bound 
  (a₁ b₁ a₂ b₂ : ℝ) 
  (h₁ : a₁ > b₁) (h₂ : b₁ > 0) (h₃ : a₂ > 0) (h₄ : b₂ > 0)
  (h₅ : focal_distance a₁ (ellipse_eccentricity a₁ b₁) = 
        focal_distance a₂ (hyperbola_eccentricity a₂ b₂)) :
  9 * (ellipse_eccentricity a₁ b₁)^2 + (hyperbola_eccentricity a₂ b₂)^2 ≥ 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_bound_l72_7209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_theorem_l72_7204

-- Define the circle C
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define line l₁
def line_l1 (a x y : ℝ) : Prop := a * x - y - 2 = 0

-- Define line l
def line_l (x y : ℝ) : Prop := x - 4 * y + 1 = 0

-- Define line l₂
def line_l2 (x y : ℝ) : Prop := x - 4 * y - 1 = 0

theorem circle_line_theorem :
  -- Line l₁ passes through the center of circle C
  (∃ a : ℝ, line_l1 a 1 0 ∧ my_circle 1 0) →
  -- 1. The value of a in line l₁ is 2
  (∃ a : ℝ, line_l1 a 1 0 ∧ a = 2) ∧
  -- 2. Line l₂ passes through the center of C and is parallel to line l
  (line_l2 1 0 ∧ ∀ x y : ℝ, line_l x y ↔ line_l2 (x + 2) y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_theorem_l72_7204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maserati_odds_against_losing_l72_7211

/-- Represents the odds against an event occurring -/
structure Odds where
  against : ℕ
  inFavor : ℕ

/-- Calculates the probability from given odds -/
def oddsToProb (o : Odds) : ℚ :=
  o.against / (o.against + o.inFavor)

/-- Represents the car brands in the race -/
inductive Brand
  | Jaguar
  | Ford
  | Maserati

/-- Given odds against losing for each brand -/
def oddsAgainstLosing (b : Brand) : Odds :=
  match b with
  | Brand.Jaguar => { against := 5, inFavor := 1 }
  | Brand.Ford => { against := 2, inFavor := 2 }
  | Brand.Maserati => { against := 2, inFavor := 1 }  -- This is what we want to prove

theorem maserati_odds_against_losing :
  let probFordLose := oddsToProb (oddsAgainstLosing Brand.Ford)
  let probJaguarLose := oddsToProb (oddsAgainstLosing Brand.Jaguar)
  let probMaseratiLose := 1 - (1 - probFordLose) - (1 - probJaguarLose)
  probMaseratiLose = 1/3 ∧ 
  oddsAgainstLosing Brand.Maserati = { against := 2, inFavor := 1 } := by
  sorry

#check maserati_odds_against_losing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maserati_odds_against_losing_l72_7211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_single_beautiful_cell_l72_7221

/-- Represents a cell on the board -/
structure Cell where
  x : Fin 100
  y : Fin 100
deriving Fintype, DecidableEq

/-- Represents the state of the board -/
def Board := Cell → Bool

/-- Checks if two cells are adjacent -/
def adjacent (c1 c2 : Cell) : Bool :=
  (c1.x = c2.x ∧ (c1.y = c2.y + 1 ∨ c1.y + 1 = c2.y)) ∨
  (c1.y = c2.y ∧ (c1.x = c2.x + 1 ∨ c1.x + 1 = c2.x))

/-- Counts the number of adjacent cells with chips -/
def countAdjacentChips (board : Board) (c : Cell) : Nat :=
  Finset.card (Finset.filter (fun c' => adjacent c c' ∧ board c') Finset.univ)

/-- Defines a beautiful cell -/
def isBeautiful (board : Board) (c : Cell) : Prop :=
  Even (countAdjacentChips board c)

/-- Theorem: It's impossible to have exactly one beautiful cell on the board -/
theorem no_single_beautiful_cell (board : Board) :
  ¬ (∃! c : Cell, isBeautiful board c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_single_beautiful_cell_l72_7221
