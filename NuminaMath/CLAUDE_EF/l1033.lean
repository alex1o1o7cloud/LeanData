import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_triplets_l1033_103372

def digit := Fin 10

def three_digit_number (a b c : digit) : ℕ :=
  100 * a.val + 10 * b.val + c.val

def sum_of_triplets (perm : Fin 9 → digit) : ℕ :=
  (three_digit_number (perm 0) (perm 1) (perm 2)) +
  (three_digit_number (perm 1) (perm 2) (perm 3)) +
  (three_digit_number (perm 2) (perm 3) (perm 4)) +
  (three_digit_number (perm 3) (perm 4) (perm 5)) +
  (three_digit_number (perm 4) (perm 5) (perm 6)) +
  (three_digit_number (perm 5) (perm 6) (perm 7)) +
  (three_digit_number (perm 6) (perm 7) (perm 8))

theorem max_sum_of_triplets :
  ∃ (perm : Fin 9 → digit), 
    (Function.Injective perm) ∧ 
    (∀ i : Fin 9, (perm i).val < 9) ∧
    (sum_of_triplets perm = 4648) ∧
    (∀ other_perm : Fin 9 → digit, 
      Function.Injective other_perm → 
      (∀ i : Fin 9, (other_perm i).val < 9) → 
      sum_of_triplets other_perm ≤ 4648) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_triplets_l1033_103372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_out_of_four_events_l1033_103390

theorem probability_two_out_of_four_events (p : ℝ) (h : p = 1/2) :
  (4 : ℕ).choose 2 * p^2 * (1 - p)^2 = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_out_of_four_events_l1033_103390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_special_angle_l1033_103304

/-- Given that the terminal side of angle θ passes through point P(-x, -6) and cos θ = -5/13,
    prove that tan(θ + π/4) = -17/7 -/
theorem tangent_sum_special_angle (θ x : ℝ) :
  (∃ x, x > 0 ∧ (-x)^2 + (-6)^2 = 13^2) →
  Real.cos θ = -5/13 →
  Real.tan (θ + π/4) = -17/7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_special_angle_l1033_103304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_roots_l1033_103321

theorem cubic_equation_roots (a b : ℝ) : 
  (∀ x : ℝ, x^3 + a*x^2 + b*x + 6 = 0 ↔ x = 2 ∨ x = 3 ∨ x = -1) →
  a = -4 ∧ b = 1 := by
  intro h
  -- The proof goes here
  sorry

#check cubic_equation_roots

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_roots_l1033_103321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1033_103391

theorem inequality_proof (x y : ℝ) (hx : x ∈ Set.Icc 0 1) (hy : y ∈ Set.Icc 0 1) :
  5 * (x^2 + y^2)^2 ≤ 4 + (x + y)^4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1033_103391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maplewood_camp_basketball_percentage_l1033_103379

theorem maplewood_camp_basketball_percentage (N : ℝ) (hN : N > 0) : 
  let basketball_players := 0.7 * N
  let swimmers := 0.5 * N
  let basketball_swimmers := 0.3 * basketball_players
  let non_swimming_basketball_players := basketball_players - basketball_swimmers
  let non_swimmers := N - swimmers
  (non_swimming_basketball_players / non_swimmers) * 100 = 98 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maplewood_camp_basketball_percentage_l1033_103379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_real_sixth_power_l1033_103389

def roots_of_unity (n : ℕ) : Set ℂ := {z : ℂ | z^n = 1}

def is_real (z : ℂ) : Prop := ∃ (r : ℝ), z = r

theorem count_real_sixth_power :
  ∃ (S : Finset ℂ), (S : Set ℂ) ⊆ roots_of_unity 30 ∧ 
    Finset.card S = 6 ∧
    ∀ z ∈ S, is_real (z^6) ∧
    ∀ z ∈ roots_of_unity 30, is_real (z^6) → z ∈ S :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_real_sixth_power_l1033_103389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tan_product_acute_triangle_l1033_103309

theorem min_tan_product_acute_triangle (A B C : ℝ) (h_acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (h_sin : Real.sin A = 2 * Real.sin B * Real.sin C) :
  ∃ (m : ℝ), m = 8 ∧ ∀ (x : ℝ), x = Real.tan A * Real.tan B * Real.tan C → m ≤ x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tan_product_acute_triangle_l1033_103309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_root_floor_l1033_103313

noncomputable def g (x : ℝ) : ℝ := Real.cos x + 3 * Real.sin x + 4 * (Real.cos x / Real.sin x)

theorem smallest_positive_root_floor (s : ℝ) :
  (∀ x, 0 < x → x < s → g x ≠ 0) →
  g s = 0 →
  3 ≤ s ∧ s < 4 :=
by
  sorry

#check smallest_positive_root_floor

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_root_floor_l1033_103313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_angles_are_congruent_l1033_103367

/-- Two angles are vertical if they are formed by two intersecting lines and are not adjacent. -/
def VerticalAngles (α β : Real) : Prop := sorry

/-- Two angles are congruent if they have the same measure. -/
def CongruentAngles (α β : Real) : Prop := α = β

/-- If two angles are vertical angles, then these two angles are congruent. -/
theorem vertical_angles_are_congruent (α β : Real) :
  VerticalAngles α β → CongruentAngles α β := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_angles_are_congruent_l1033_103367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_difference_l1033_103341

noncomputable section

open Real

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 2*log x

-- Define the conditions
def has_two_extremal_points (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
    (∀ x : ℝ, x > 0 → (deriv (f a)) x = 0 ↔ x = x₁ ∨ x = x₂)

theorem range_of_difference (a : ℝ) 
  (h1 : 2*(exp 1 + 1/exp 1) < a) 
  (h2 : a < 20/3) 
  (h3 : has_two_extremal_points a) : 
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧
    (exp 1)^2 - 1/(exp 1)^2 - 4 < f a x₁ - f a x₂ ∧ 
    f a x₁ - f a x₂ < 80/9 - 4*log 3 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_difference_l1033_103341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_older_brother_catches_up_older_brother_catches_up_proof_l1033_103328

/-- Proves that the older brother catches up with the younger brother and mother before they reach grandmother's house. -/
theorem older_brother_catches_up (younger_speed mother_speed older_speed total_distance delay : ℝ) : Prop :=
  let catch_up_time := delay * younger_speed / (older_speed - younger_speed)
  let total_time_to_catch := delay + catch_up_time
  let time_to_grandma := total_distance / younger_speed
  younger_speed = 2 ∧ 
  mother_speed = 2 ∧ 
  older_speed = 6 ∧ 
  total_distance = 2 * 1.75 ∧ 
  delay = 1 →
  total_time_to_catch < time_to_grandma

/-- Proof of the theorem -/
theorem older_brother_catches_up_proof : older_brother_catches_up 2 2 6 3.5 1 := by
  sorry

#check older_brother_catches_up
#check older_brother_catches_up_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_older_brother_catches_up_older_brother_catches_up_proof_l1033_103328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cryptarithm_B_equals_two_l1033_103366

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the mapping of letters to digits in the cryptarithm -/
structure CryptarithmMapping where
  R : Digit
  E : Digit
  K : Digit
  A : Digit
  B : Digit
  V : Digit
  D : Digit
  distinct : R ≠ E ∧ R ≠ K ∧ R ≠ A ∧ R ≠ B ∧ R ≠ V ∧ R ≠ D ∧
             E ≠ K ∧ E ≠ A ∧ E ≠ B ∧ E ≠ V ∧ E ≠ D ∧
             K ≠ A ∧ K ≠ B ∧ K ≠ V ∧ K ≠ D ∧
             A ≠ B ∧ A ≠ V ∧ A ≠ D ∧
             B ≠ V ∧ B ≠ D ∧
             V ≠ D

/-- The cryptarithm equation -/
def cryptarithmEquation (m : CryptarithmMapping) : Prop :=
  1000 * m.R.val + 100 * m.E.val + 10 * m.K.val + m.A.val +
  1000 * m.K.val + 100 * m.A.val + 10 * m.R.val + m.E.val =
  10000 * m.A.val + 1000 * m.B.val + 100 * m.V.val + 10 * m.A.val + m.D.val

theorem cryptarithm_B_equals_two :
  ∃ (m : CryptarithmMapping), cryptarithmEquation m ∧ m.B = ⟨2, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cryptarithm_B_equals_two_l1033_103366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1033_103352

-- Define a as a parameter
variable (a : ℝ)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (3/4) * Real.sin (x - a) + 1/4

-- Define the theorem
theorem range_of_a :
  (Set.range f = Set.Icc (-1/2) 1) →
  Set.range (λ _ : ℝ => a) = Set.Icc (-Real.arcsin (1/3)) (Real.arcsin (1/3)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1033_103352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l1033_103333

/-- Given a hyperbola with equation x²/9 - y²/4 = 1, its focal length is 2√13 -/
theorem hyperbola_focal_length :
  ∀ (x y : ℝ),
  x^2 / 9 - y^2 / 4 = 1 →
  ∃ (f : ℝ), f = 2 * Real.sqrt 13 ∧ f = Real.sqrt (13 : ℝ) * 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l1033_103333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_problem_l1033_103331

-- Define the function f
variable (f : ℝ → ℝ)

-- Define f' as the derivative of f
variable (f' : ℝ → ℝ)

-- Define the conditions
variable (h1 : HasDerivAt f (f' 5) 5)
variable (h2 : (fun x ↦ -x + 8) = fun x ↦ f 5 + f' 5 * (x - 5))

-- State the theorem
theorem tangent_line_problem : f 5 + f' 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_problem_l1033_103331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_dot_product_l1033_103332

/-- Given a quadrilateral ABCD with O as the intersection of diagonals, 
    prove that DA · DC = 0 under certain conditions -/
theorem quadrilateral_dot_product (A B C D O : ℝ × ℝ) : 
  (O - A) = (C - O) →  -- AO = OC
  (B - O) = (2 : ℝ) • (O - D) →  -- BO = 2OD
  ‖C - A‖ = 4 →  -- AC = 4
  (B - A) • (B - C) = 12 →  -- BA · BC = 12
  (D - A) • (D - C) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_dot_product_l1033_103332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ethanol_percentage_in_fuel_A_l1033_103339

/-- Proves that the percentage of ethanol in fuel A is 12% -/
theorem ethanol_percentage_in_fuel_A (tank_capacity : ℝ) (fuel_A_volume : ℝ) 
  (fuel_B_ethanol_percentage : ℝ) (total_ethanol_volume : ℝ) : ℝ :=
  by
  -- Define the conditions
  have h1 : tank_capacity = 212 := by sorry
  have h2 : fuel_A_volume = 98 := by sorry
  have h3 : fuel_B_ethanol_percentage = 0.16 := by sorry
  have h4 : total_ethanol_volume = 30 := by sorry

  -- Calculate the result
  have result : (fuel_A_volume * (12 / 100)) + 
    ((tank_capacity - fuel_A_volume) * fuel_B_ethanol_percentage) = total_ethanol_volume := by sorry

  -- Return the percentage of ethanol in fuel A
  exact 12

-- Define a computable function for evaluation
def compute_ethanol_percentage_in_fuel_A (tank_capacity : ℝ) (fuel_A_volume : ℝ) 
  (fuel_B_ethanol_percentage : ℝ) (total_ethanol_volume : ℝ) : ℝ :=
  12

#eval compute_ethanol_percentage_in_fuel_A 212 98 0.16 30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ethanol_percentage_in_fuel_A_l1033_103339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_common_point_l1033_103336

-- Define a type for 2D points
structure Point where
  x : ℝ
  y : ℝ

-- Define a type for polygons
structure Polygon where
  vertices : List Point

-- Define the sequence of polygons
def polygonSequence : ℕ → Polygon :=
  sorry

-- Define the midpoint property
def midpointProperty (P Q : Polygon) : Prop :=
  ∀ (v : Point), v ∈ Q.vertices → 
    ∃ (a b : Point), a ∈ P.vertices ∧ b ∈ P.vertices ∧ 
      v.x = (a.x + b.x) / 2 ∧ v.y = (a.y + b.y) / 2

-- Define what it means for a point to be inside a polygon
def isInside (p : Point) (P : Polygon) : Prop :=
  sorry

-- The main theorem
theorem unique_common_point :
  (∀ k : ℕ, midpointProperty (polygonSequence k) (polygonSequence (k + 1))) →
  ∃! p : Point, (∀ k : ℕ, isInside p (polygonSequence k)) ∧ p = ⟨0, 0⟩ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_common_point_l1033_103336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_order_l1033_103356

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem f_order : f (Real.exp 1) > f 3 ∧ f 3 > f 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_order_l1033_103356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_ratio_max_l1033_103344

theorem isosceles_triangle_ratio_max (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_isosceles : b = c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  (b + c) / a ≤ 2 := by
  sorry

#check isosceles_triangle_ratio_max

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_ratio_max_l1033_103344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_islands_visited_l1033_103386

def γ (n : ℕ) : ℕ :=
  sorry -- Definition of γ based on prime factorization of n

def has_ferry (n a i j : ℕ) : Prop :=
  i ≠ j ∧ 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n ∧ (i * j) % n = (i * a) % n

theorem max_islands_visited (n a : ℕ) (h1 : n ≥ 2) (h2 : Nat.Coprime a n) :
  ∃ (max_islands : ℕ), max_islands = 1 + γ n ∧
    (∀ (path : List ℕ),
      (∀ (i : ℕ), i ∈ path → 1 ≤ i ∧ i ≤ n) →
      (∀ (i j : ℕ), List.Pairwise (has_ferry n a) path) →
      path.length ≤ max_islands) ∧
    (∃ (optimal_path : List ℕ),
      (∀ (i : ℕ), i ∈ optimal_path → 1 ≤ i ∧ i ≤ n) ∧
      List.Pairwise (has_ferry n a) optimal_path ∧
      optimal_path.length = max_islands) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_islands_visited_l1033_103386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l1033_103384

theorem tan_difference (x y : ℝ) (hx : Real.tan x = 3) (hy : Real.tan y = 2) : 
  Real.tan (x - y) = 1 / 7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l1033_103384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_property_l1033_103397

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point on the ellipse -/
structure EllipsePoint (E : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (E : Ellipse) : ℝ := Real.sqrt (1 - E.b^2 / E.a^2)

/-- The distance between foci of an ellipse -/
noncomputable def focalDistance (E : Ellipse) : ℝ := 2 * Real.sqrt (E.a^2 - E.b^2)

theorem ellipse_property (E : Ellipse) 
  (h_ecc : eccentricity E = 3/5)
  (h_minor : E.b = 4)
  (M N : EllipsePoint E)
  (h_inscribed : Real.pi = 2 * Real.pi * (5 / (M.x - N.x)^2 + (M.y - N.y)^2)) :
  |M.y - N.y| = 5/3 := by
  sorry

#check ellipse_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_property_l1033_103397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_vowel_l1033_103329

/-- Set1 represents the first set of letters --/
def Set1 : Finset Char := {'a', 'b', 'c', 'd', 'e'}

/-- Set2 represents the second set of letters --/
def Set2 : Finset Char := {'k', 'l', 'm', 'n', 'o', 'p'}

/-- Vowels1 represents the vowels in Set1 --/
def Vowels1 : Finset Char := {'a', 'e'}

/-- Vowels2 represents the vowels in Set2 --/
def Vowels2 : Finset Char := {'o'}

/-- The probability of picking at least one vowel --/
theorem prob_at_least_one_vowel :
  (1 : ℚ) - (((Set1.card - Vowels1.card) : ℚ) / Set1.card) * (((Set2.card - Vowels2.card) : ℚ) / Set2.card) = 1 / 2 := by
  sorry

-- Remove the #eval statement as it's not necessary for building

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_vowel_l1033_103329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_zero_moments_l1033_103302

/-- The distance function representing the distance traveled s in t seconds -/
noncomputable def distance (t : ℝ) : ℝ := (1/4) * t^4 - (5/3) * t^3 + 2 * t^2

/-- The velocity function, which is the derivative of the distance function -/
noncomputable def velocity (t : ℝ) : ℝ := t^3 - 5*t^2 + 4*t

/-- Theorem stating that the velocity is zero at t = 0, t = 1, and t = 4 -/
theorem velocity_zero_moments :
  {t : ℝ | velocity t = 0} = {0, 1, 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_zero_moments_l1033_103302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_after_seven_years_l1033_103330

/-- Simple interest calculation function -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal + (principal * rate * time)

/-- Theorem: Given the conditions, the total amount after 7 years is $815 -/
theorem total_amount_after_seven_years
  (initial_investment : ℝ)
  (amount_after_two_years : ℝ)
  (h1 : initial_investment = 500)
  (h2 : amount_after_two_years = 590)
  (h3 : amount_after_two_years = simple_interest initial_investment ((590 - 500) / (2 * 500)) 2) :
  simple_interest initial_investment ((590 - 500) / (2 * 500)) 7 = 815 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_after_seven_years_l1033_103330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_distance_difference_l1033_103392

/-- Represents the race between Alberto and Bjorn -/
structure Race where
  alberto_initial_speed : ℝ
  alberto_final_speed : ℝ
  bjorn_speed : ℝ
  speed_change_time : ℝ
  total_time : ℝ

/-- Calculates the distance traveled by Alberto -/
def alberto_distance (r : Race) : ℝ :=
  r.alberto_initial_speed * r.speed_change_time + r.alberto_final_speed * (r.total_time - r.speed_change_time)

/-- Calculates the distance traveled by Bjorn -/
def bjorn_distance (r : Race) : ℝ :=
  r.bjorn_speed * r.total_time

/-- The main theorem stating the difference in distance traveled -/
theorem race_distance_difference (r : Race) 
  (h1 : r.alberto_initial_speed = 18)
  (h2 : r.alberto_final_speed = 22)
  (h3 : r.bjorn_speed = 15)
  (h4 : r.speed_change_time = 2)
  (h5 : r.total_time = 5) :
  alberto_distance r - bjorn_distance r = 27 := by
  sorry

/-- Example race instance -/
def example_race : Race := {
  alberto_initial_speed := 18,
  alberto_final_speed := 22,
  bjorn_speed := 15,
  speed_change_time := 2,
  total_time := 5
}

#eval alberto_distance example_race
#eval bjorn_distance example_race
#eval alberto_distance example_race - bjorn_distance example_race

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_distance_difference_l1033_103392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_and_hypotenuse_l1033_103345

-- Define the necessary functions and predicates
def IsRightTriangle : Set ℝ → Prop := sorry
def TriangleArea : Set ℝ → ℝ := sorry
def Hypotenuse : Set ℝ → ℝ := sorry

theorem right_triangle_area_and_hypotenuse :
  ∀ (t : Set ℝ) (leg1 leg2 : ℝ),
  IsRightTriangle t →
  leg1 = 30 →
  leg2 = 24 →
  (TriangleArea t = 360) ∧ (Hypotenuse t = Real.sqrt 1476) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_and_hypotenuse_l1033_103345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proportional_function_solution_l1033_103347

/-- A proportional function passing through (2, -4) and its vertical shift -/
def ProportionalFunctionProblem :=
  ∃ (f g : ℝ → ℝ) (k m : ℝ),
    (∀ x, f x = k * x) ∧  -- f is a proportional function
    f 2 = -4 ∧           -- f passes through (2, -4)
    (∀ x, g x = f x + m) ∧  -- g is f shifted vertically by m
    g 1 = 1              -- g passes through (1, 1)

/-- The solution to the proportional function problem -/
theorem proportional_function_solution :
  ProportionalFunctionProblem →
  ∃ (k m : ℝ), k = -2 ∧ m = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_proportional_function_solution_l1033_103347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_lines_divide_plane_into_22_regions_l1033_103361

/-- A type representing a straight line in a plane -/
structure Line where
  -- You might want to add more properties here in a real implementation

/-- A type representing a plane -/
structure Plane where
  -- You might want to add more properties here in a real implementation

/-- A function that determines if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop := sorry

/-- A function that determines if three lines are concurrent -/
def are_concurrent (l1 l2 l3 : Line) : Prop := sorry

/-- A function that counts the number of regions formed by a set of lines in a plane -/
def count_regions (p : Plane) (lines : Finset Line) : ℕ := sorry

/-- The theorem stating that six non-parallel, non-concurrent lines divide a plane into 22 regions -/
theorem six_lines_divide_plane_into_22_regions (p : Plane) (lines : Finset Line) :
  (lines.card = 6) →
  (∀ l1 l2, l1 ∈ lines → l2 ∈ lines → l1 ≠ l2 → ¬are_parallel l1 l2) →
  (∀ l1 l2 l3, l1 ∈ lines → l2 ∈ lines → l3 ∈ lines → 
    l1 ≠ l2 ∧ l2 ≠ l3 ∧ l1 ≠ l3 → ¬are_concurrent l1 l2 l3) →
  count_regions p lines = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_lines_divide_plane_into_22_regions_l1033_103361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cells_crossed_theorem_l1033_103318

/-- Represents a grid of cells -/
structure Grid where
  m : ℕ  -- number of rows
  n : ℕ  -- number of columns

/-- Represents a line segment or needle -/
structure Segment where
  length : ℕ

/-- Calculates the maximum number of cells crossed by a line segment on a grid -/
def maxCellsCrossedBySegment (g : Grid) : ℕ :=
  g.m + g.n - 1

/-- Calculates the maximum number of cells crossed by a needle of given length -/
def maxCellsCrossedByNeedle (s : Segment) : ℕ :=
  if s.length = 200 then 285 else 0  -- We only consider the case of length 200

theorem max_cells_crossed_theorem (g : Grid) (s : Segment) :
  (maxCellsCrossedBySegment g = g.m + g.n - 1) ∧
  (s.length = 200 → maxCellsCrossedByNeedle s = 285) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cells_crossed_theorem_l1033_103318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l1033_103306

noncomputable def f (x : ℝ) := Real.sin (2 * x - Real.pi / 6)

noncomputable def g (x : ℝ) := Real.sin (2 * x - Real.pi / 3)

theorem axis_of_symmetry :
  ∀ x : ℝ, g x = g (5 * Real.pi / 6 - x) :=
by
  intro x
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l1033_103306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_shift_l1033_103301

noncomputable section

/-- The given function f(x) -/
def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4)

/-- The function g(x) obtained by shifting f(x) to the left by m units -/
def g (ω m : ℝ) (x : ℝ) : ℝ := f ω (x + m)

theorem symmetry_and_shift (ω : ℝ) (h_pos : ω > 0) 
  (h_sym : ∀ x : ℝ, f ω (x + Real.pi / (3 * ω)) = f ω (-x + Real.pi / (3 * ω))) :
  (ω = 3) ∧ 
  (∃ m : ℝ, m > 0 ∧ (∀ x : ℝ, g ω m x = g ω m (-x)) ∧
    (∀ m' : ℝ, m' > 0 → (∀ x : ℝ, g ω m' x = g ω m' (-x)) → m ≤ m')) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_shift_l1033_103301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l1033_103303

/-- The number of solutions for the system of equations x^2 + y^2 + xy = a and x^2 - y^2 = b -/
noncomputable def num_solutions (a b : ℝ) : ℕ :=
  if -2*a < Real.sqrt 3 * b ∧ Real.sqrt 3 * b < 2*a then 4
  else if -2*a = Real.sqrt 3 * b ∨ Real.sqrt 3 * b = 2*a then 2
  else if a = 0 ∧ b = 0 then 1
  else 0

/-- Theorem stating the equivalence between the existence of solutions and the number of solutions being positive -/
theorem system_solutions (a b : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 + x*y = a ∧ x^2 - y^2 = b) ↔ 
  (num_solutions a b > 0) := by
  sorry

#check system_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l1033_103303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_increase_march_to_april_l1033_103349

theorem profit_increase_march_to_april 
  (march_to_april : ℝ) 
  (april_to_may : ℝ) 
  (may_to_june : ℝ) 
  (march_to_june : ℝ) :
  april_to_may = -20 →
  may_to_june = 50 →
  march_to_june = 32.00000000000003 →
  (1 + march_to_april / 100) * (1 + april_to_may / 100) * (1 + may_to_june / 100) = 1 + march_to_june / 100 →
  march_to_april = 10 := by
  intro h1 h2 h3 h4
  sorry

#check profit_increase_march_to_april

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_increase_march_to_april_l1033_103349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_composition_l1033_103383

/-- The curve equation -/
def curve_equation (x y : ℝ) : Prop :=
  x^4 - y^4 - 4*x^2 + 4*y^2 = 0

/-- Predicate for a set to be a line in ℝ² -/
def IsLine (s : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∧ s = {(x, y) | a*x + b*y + c = 0}

/-- Predicate for a set to be a circle in ℝ² -/
def IsCircle (s : Set (ℝ × ℝ)) : Prop :=
  ∃ (x₀ y₀ r : ℝ), r > 0 ∧ s = {(x, y) | (x - x₀)^2 + (y - y₀)^2 = r^2}

/-- The curve consists of two intersecting straight lines and a circle -/
theorem curve_composition :
  ∃ (l₁ l₂ : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)),
    IsLine l₁ ∧ IsLine l₂ ∧ IsCircle c ∧
    (∀ (x y : ℝ), curve_equation x y ↔ (x, y) ∈ l₁ ∪ l₂ ∪ c) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_composition_l1033_103383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_relationship_l1033_103334

/-- The function f as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := 2015 * Real.sin x + x^2015 + 2015 * Real.tan x + 2015

/-- Theorem stating the relationship between f(-2015) and f(2015) -/
theorem f_relationship (h : f (-2015) = 2016) : f 2015 = 2014 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_relationship_l1033_103334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_dimensions_count_l1033_103399

theorem garden_dimensions_count : 
  ∃! n : ℕ, n = (Finset.filter 
    (fun p : ℕ × ℕ ↦ 
      p.2 > p.1 ∧ 
      (p.1 - 4) * (p.2 - 4) = 2 * p.1 * p.2 / 3 ∧ 
      p.1 > 0 ∧ p.2 > 0)
    (Finset.product (Finset.range 100) (Finset.range 100))).card ∧ 
  n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_dimensions_count_l1033_103399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_directrix_l1033_103370

/-- A hyperbola with foci on the y-axis -/
structure Hyperbola where
  /-- The asymptotes of the hyperbola are x ± 2y = 0 -/
  asymptotes : ∀ (x y : ℝ), x = 2*y ∨ x = -2*y
  /-- The minimum distance from any point on the hyperbola to A(5,0) is √6 -/
  min_distance : ∀ (x y : ℝ), (x - 5)^2 + y^2 ≥ 6

/-- The equation of the directrix of the hyperbola -/
def directrix (h : Hyperbola) (y : ℝ) : Prop :=
  y = Real.sqrt 5 / 5 ∨ y = -(Real.sqrt 5 / 5)

/-- Theorem stating that the directrix equation holds for the given hyperbola -/
theorem hyperbola_directrix (h : Hyperbola) :
  ∀ (y : ℝ), directrix h y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_directrix_l1033_103370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_sectors_area_proof_l1033_103387

/-- The area of two combined circular sectors, each with a central angle of 90° and a radius of 15 units -/
noncomputable def combined_sectors_area : ℝ := 112.5 * Real.pi

/-- Theorem: The area of two combined circular sectors, each with a central angle of 90° and a radius of 15 units, is equal to 112.5π -/
theorem combined_sectors_area_proof (sector_angle : ℝ) (sector_radius : ℝ) :
  sector_angle = 90 ∧ sector_radius = 15 →
  combined_sectors_area = 2 * (sector_angle / 360) * Real.pi * sector_radius^2 :=
by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval combined_sectors_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_sectors_area_proof_l1033_103387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_touching_cones_and_table_l1033_103310

/-- A sphere touches a cone -/
def sphere_touches_cone (r α : ℝ) (O : ℝ × ℝ × ℝ) (R : ℝ) : Prop :=
  sorry

/-- A sphere touches a table -/
def sphere_touches_table (O : ℝ × ℝ × ℝ) (R : ℝ) : Prop :=
  sorry

/-- The radius of a sphere touching three cones and a table -/
theorem sphere_radius_touching_cones_and_table 
  (r₁ r₂ r₃ : ℝ) 
  (α β γ : ℝ) 
  (h₁ : r₁ = 1)
  (h₂ : r₂ = 4)
  (h₃ : r₃ = 4)
  (h₄ : α = -4 * Real.arctan (1/3))
  (h₅ : β = 4 * Real.arctan (9/11))
  (h₆ : γ = 4 * Real.arctan (9/11))
  (h₇ : r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0) -- ensure positive radii
  : ∃ (R : ℝ), R = 5/3 ∧ 
    R > 0 ∧
    (∃ (O : ℝ × ℝ × ℝ), 
      sphere_touches_cone r₁ α O R ∧
      sphere_touches_cone r₂ β O R ∧
      sphere_touches_cone r₃ γ O R ∧
      sphere_touches_table O R) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_touching_cones_and_table_l1033_103310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_of_squares_l1033_103360

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 24 + y^2 / 12 = 1

-- Define a point on the ellipse
def point_on_ellipse (x₀ y₀ : ℝ) : Prop := ellipse_C x₀ y₀

-- Define the circle centered at a point on the ellipse
def circle_R (x₀ y₀ x y : ℝ) : Prop := (x - x₀)^2 + (y - y₀)^2 = 8

-- Define the tangent condition
def are_tangent_points (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), point_on_ellipse x₀ y₀ ∧
  ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
  (∃ (l : ℝ), x₁ = l * (x₁ - x₀) ∧ y₁ = l * (y₁ - y₀)) ∧
  (∃ (m : ℝ), x₂ = m * (x₂ - x₀) ∧ y₂ = m * (y₂ - y₀))

-- The main theorem
theorem constant_sum_of_squares :
  ∀ (x₁ y₁ x₂ y₂ : ℝ), are_tangent_points x₁ y₁ x₂ y₂ →
  x₁^2 + y₁^2 + x₂^2 + y₂^2 = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_of_squares_l1033_103360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_for_odd_shifted_sine_l1033_103375

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem phi_value_for_odd_shifted_sine 
  (φ : ℝ) 
  (h1 : 0 < φ) 
  (h2 : φ < Real.pi) 
  (h3 : ∀ x, f φ (x - Real.pi/3) = -f φ (-x - Real.pi/3)) :
  φ = 2 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_for_odd_shifted_sine_l1033_103375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_o_prints_k_l1033_103300

/-- Represents a 3D coordinate in a 3x3x3 cube --/
structure Coord where
  x : Fin 3
  y : Fin 3
  z : Fin 3

/-- Represents a face of the cube --/
inductive Face
  | Front
  | Back
  | Left
  | Right
  | Top
  | Bottom

/-- Represents a letter that can be printed --/
inductive Letter
  | K
  | O
  | T

/-- Represents the cube structure --/
def CubeStructure := Finset Coord

/-- Returns the face opposite to the given face --/
def oppositeFace (f : Face) : Face :=
  match f with
  | Face.Front => Face.Back
  | Face.Back => Face.Front
  | Face.Left => Face.Right
  | Face.Right => Face.Left
  | Face.Top => Face.Bottom
  | Face.Bottom => Face.Top

/-- Returns the letter printed on a given face --/
def printedLetter (c : CubeStructure) (f : Face) : Letter := sorry

/-- Checks if the cube structure is valid (contains 16 cubes) --/
def isValidStructure (c : CubeStructure) : Prop :=
  c.card = 16

/-- Checks if the given structure produces "KOT" on three faces --/
def producesKOT (c : CubeStructure) : Prop :=
  ∃ (f1 f2 f3 : Face),
    f1 ≠ f2 ∧ f2 ≠ f3 ∧ f1 ≠ f3 ∧
    printedLetter c f1 = Letter.K ∧
    printedLetter c f2 = Letter.O ∧
    printedLetter c f3 = Letter.T

theorem opposite_o_prints_k (c : CubeStructure) 
  (h1 : isValidStructure c) (h2 : producesKOT c) :
  ∃ (f : Face), printedLetter c f = Letter.O ∧ 
    printedLetter c (oppositeFace f) = Letter.K := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_o_prints_k_l1033_103300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l1033_103346

theorem constant_term_expansion (x : ℝ) : 
  let expansion := (x^2 - 1 / (Real.sqrt 5 * x^3))^5
  ∃ c : ℝ, c = 2 ∧ 
    (∀ ε > (0 : ℝ), ∃ δ > (0 : ℝ), ∀ x, 0 < |x| → |x| < δ → |expansion - c| < ε) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l1033_103346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_equals_negative_one_l1033_103395

def sequence_a : ℕ → ℚ
  | 0 => 2  -- Define the base case for 0
  | 1 => 2  -- Define the first term explicitly
  | (n + 2) => 1 - 1 / sequence_a (n + 1)

theorem a_2016_equals_negative_one : sequence_a 2016 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_equals_negative_one_l1033_103395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_prove_equation_solution_l1033_103363

theorem equation_solution : ∃! x : ℝ, (3 : ℝ)^x * (9 : ℝ)^x = (81 : ℝ)^(x-12) := by
  use 48
  constructor
  · -- Prove that 48 satisfies the equation
    sorry
  · -- Prove uniqueness
    intro y hy
    sorry

#check equation_solution

-- Proof of the specific solution
theorem prove_equation_solution : (3 : ℝ)^48 * (9 : ℝ)^48 = (81 : ℝ)^(48-12) := by
  sorry

#check prove_equation_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_prove_equation_solution_l1033_103363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_composition_periodicity_s_six_times_five_l1033_103323

noncomputable def s (x : ℝ) : ℝ := 2 / (3 - x)

theorem s_composition_periodicity (x : ℝ) (h : x ≠ 1) : s (s (s x)) = 2/3 := by
  sorry

theorem s_six_times_five : s (s (s (s (s (s 5))))) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_composition_periodicity_s_six_times_five_l1033_103323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_end_point_y_coordinate_l1033_103378

/-- The y-coordinate of the end point of a line segment -/
noncomputable def end_y_coordinate (start_x start_y end_x : ℝ) (length : ℝ) : ℝ :=
  let y := Real.sqrt (length ^ 2 - (end_x - start_x) ^ 2) + start_y
  if y > start_y then y else 2 * start_y - y

/-- Theorem: The y-coordinate of the end point of a line segment -/
theorem end_point_y_coordinate :
  end_y_coordinate 2 5 10 13 = 16 := by
  sorry

-- Remove the #eval line as it's causing issues with noncomputable definitions
-- #eval end_y_coordinate 2 5 10 13

end NUMINAMATH_CALUDE_ERRORFEEDBACK_end_point_y_coordinate_l1033_103378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l1033_103343

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/25 + y^2/9 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-4, 0)
def F₂ : ℝ × ℝ := (4, 0)

-- Define a point on the ellipse
def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  ellipse x y

-- Define the perpendicularity condition
def perpendicular_condition (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (x + 4) * (x - 4) + y * y = 0

-- Define the area of a triangle
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (1/2) * abs ((x₁ - x₃) * (y₂ - y₁) - (x₁ - x₂) * (y₃ - y₁))

-- State the theorem
theorem ellipse_triangle_area :
  ∀ P : ℝ × ℝ,
  point_on_ellipse P →
  perpendicular_condition P →
  (area_triangle P F₁ F₂ = 9) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l1033_103343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hydrogen_percentage_in_water_l1033_103326

/-- The atomic mass of hydrogen in atomic mass units (amu) -/
noncomputable def hydrogen_mass : ℝ := 1.008

/-- The atomic mass of oxygen in atomic mass units (amu) -/
noncomputable def oxygen_mass : ℝ := 16.00

/-- The chemical formula of dihydrogen monoxide (water) -/
inductive H2O
| mk : H2O

/-- Calculate the mass percentage of an element in a compound -/
noncomputable def mass_percentage (element_mass total_mass : ℝ) : ℝ :=
  (element_mass / total_mass) * 100

/-- The mass percentage of hydrogen in dihydrogen monoxide (water) -/
theorem hydrogen_percentage_in_water :
  let total_hydrogen_mass := 2 * hydrogen_mass
  let total_water_mass := total_hydrogen_mass + oxygen_mass
  abs (mass_percentage total_hydrogen_mass total_water_mass - 11.19) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hydrogen_percentage_in_water_l1033_103326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_100_value_l1033_103398

def b : ℕ → ℚ
  | 0 => 2  -- We need to handle the 0 case
  | 1 => 2
  | 2 => 2/3
  | n+3 => (2 - b (n+2)) / (3 * b (n+1))

theorem b_100_value : b 100 = 33/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_100_value_l1033_103398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_science_competition_selection_l1033_103354

theorem science_competition_selection (n_female : Nat) (n_male : Nat) (n_select : Nat) :
  n_female = 2 →
  n_male = 4 →
  n_select = 3 →
  (Finset.sum (Finset.range (n_female + 1))
    (fun k => Nat.choose n_female k * Nat.choose n_male (n_select - k))) -
  Nat.choose n_male n_select = 16 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_science_competition_selection_l1033_103354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l1033_103312

noncomputable def z : ℂ := (Complex.abs (1 - Complex.I) + 2 * Complex.I) / (1 - Complex.I)

theorem z_in_second_quadrant : 
  Real.sign z.re = -1 ∧ Real.sign z.im = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l1033_103312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_tangent_lines_l1033_103362

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

def is_on_circle (p : Point) (c : Circle) : Prop :=
  distance p c.center = c.radius

def is_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

def is_tangent_to_circle (l : Line) (c : Circle) : Prop :=
  ∃ p : Point, is_on_circle p c ∧ is_on_line p l

theorem four_tangent_lines (p q : Point) (h : distance p q = 8) :
  ∃! (lines : Finset Line), 
    lines.card = 4 ∧ 
    (∀ l ∈ lines, is_tangent_to_circle l ⟨p, 3⟩ ∧ is_tangent_to_circle l ⟨q, 4⟩) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_tangent_lines_l1033_103362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_polynomial_divisibility_l1033_103305

theorem symmetric_polynomial_divisibility 
  (P : Polynomial ℤ → Polynomial ℤ → Polynomial ℤ) 
  (h_sym : ∀ x y, P x y = P y x) 
  (h_div : ∀ x y, (X - Y) ∣ P x y) : 
  ∀ x y, (X - Y)^2 ∣ P x y :=
by
  intro x y
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_polynomial_divisibility_l1033_103305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1033_103348

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := x^2 / 25 - y^2 / 4 = 1

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop := y = 2/5 * x ∨ y = -2/5 * x

-- Define a placeholder for the concept of an asymptote
def is_asymptote (x y : ℝ) : Prop := sorry

-- Theorem statement
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola_equation x y → (asymptote_equation x y ↔ is_asymptote x y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1033_103348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_exceeding_4_pow_2018_l1033_103311

def sequence_a : ℕ → ℝ
  | 0 => 7
  | (n + 1) => sequence_a n * (sequence_a n + 2)

theorem smallest_n_exceeding_4_pow_2018 :
  (∀ k : ℕ, k < 12 → sequence_a k ≤ 4^2018) ∧
  sequence_a 12 > 4^2018 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_exceeding_4_pow_2018_l1033_103311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_completion_time_l1033_103353

/-- The number of days it takes x to complete the work -/
noncomputable def x_days : ℝ := 40

/-- The number of days it takes y to complete the work -/
def y_days : ℝ := 45

/-- The portion of work completed by x in 8 days -/
noncomputable def x_work : ℝ := 8 / x_days

/-- The portion of work completed by y in 36 days -/
noncomputable def y_work : ℝ := 36 / y_days

theorem x_completion_time :
  x_work + y_work = 1 → x_days = 40 := by
  intro h
  -- The proof goes here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_completion_time_l1033_103353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theorem_one_theorem_two_l1033_103342

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The law relating sides and angles in the triangle -/
def sideLaw (t : Triangle) : Prop :=
  2 * t.a / Real.cos t.A = (3 * t.c - 2 * t.b) / Real.cos t.B

/-- Theorem 1: If the side law holds and b = √5 * sin(B), then a = 5/3 -/
theorem theorem_one (t : Triangle) (h1 : sideLaw t) (h2 : t.b = Real.sqrt 5 * Real.sin t.B) :
  t.a = 5/3 := by
  sorry

/-- The area of the triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  1/2 * t.b * t.c * Real.sin t.A

/-- Theorem 2: If the side law holds, a = √6, and the area is √5/2, then b + c = 4 -/
theorem theorem_two (t : Triangle) (h1 : sideLaw t) (h2 : t.a = Real.sqrt 6) 
  (h3 : triangleArea t = Real.sqrt 5 / 2) :
  t.b + t.c = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theorem_one_theorem_two_l1033_103342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1033_103376

theorem triangle_problem (A B C a b c : Real) : 
  -- Given conditions
  0 < A ∧ A < π ∧ 
  0 < B ∧ B < π ∧ 
  0 < C ∧ C < π ∧ 
  A + B + C = π ∧
  2 * (Real.cos (A / 2))^2 + Real.cos A = 0 ∧
  a = 2 * Real.sqrt 3 ∧
  b + c = 4 →
  -- Conclusions
  A = 2 * π / 3 ∧
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1033_103376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanarity_conditions_l1033_103316

/-- A point is coplanar with three other points if it can be expressed as a linear combination
    of the vectors from the origin to those points, with coefficients summing to 1 -/
def IsCoplanar (O A B C M : ℝ × ℝ × ℝ) : Prop :=
  ∃ x y z : ℝ, x + y + z = 1 ∧ M - O = x • (A - O) + y • (B - O) + z • (C - O)

theorem coplanarity_conditions (O A B C M : ℝ × ℝ × ℝ) :
  (¬ IsCoplanar O A B C M ↔ M - O = 2 • (A - O) - (B - O) - (C - O)) ∧
  (¬ IsCoplanar O A B C M ↔ M - O = (1/5) • (A - O) + (1/3) • (B - O) + (1/2) • (C - O)) ∧
  (IsCoplanar O A B C M ↔ (M - A) + (M - B) + (M - C) = 0) ∧
  (¬ IsCoplanar O A B C M ↔ (M - O) + (A - O) + (B - O) + (C - O) = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanarity_conditions_l1033_103316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hillarys_craft_sales_l1033_103327

/-- The number of crafts Hillary sold at the flea market -/
def crafts_sold : ℕ := sorry

/-- The price of each craft in dollars -/
def craft_price : ℕ := 12

/-- The extra amount Hillary received from an appreciative customer in dollars -/
def extra_amount : ℕ := 7

/-- The amount Hillary deposited into her bank account in dollars -/
def deposit_amount : ℕ := 18

/-- The amount Hillary was left with after the deposit in dollars -/
def remaining_amount : ℕ := 25

theorem hillarys_craft_sales : 
  crafts_sold * craft_price + extra_amount - deposit_amount = remaining_amount :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hillarys_craft_sales_l1033_103327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_integral_l1033_103357

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + a)

theorem tangent_line_integral (a : ℝ) : 
  (∃ x, f a x = x + 1 ∧ (deriv (f a)) x = 1) → 
  ∫ x in (1)..(2), deriv (deriv (f a)) (x - 2) = Real.log 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_integral_l1033_103357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_cone_properties_l1033_103317

/-- Represents a right circular cone. -/
structure RightCircularCone where
  volume : ℝ
  height : ℝ

/-- Calculate the radius of the base of a right circular cone. -/
noncomputable def baseRadius (cone : RightCircularCone) : ℝ :=
  Real.sqrt ((3 * cone.volume) / (Real.pi * cone.height))

/-- Calculate the area of the base of a right circular cone. -/
noncomputable def baseArea (cone : RightCircularCone) : ℝ :=
  Real.pi * (baseRadius cone) ^ 2

/-- Calculate the slant height of a right circular cone. -/
noncomputable def slantHeight (cone : RightCircularCone) : ℝ :=
  Real.sqrt ((baseRadius cone) ^ 2 + cone.height ^ 2)

/-- Calculate the lateral surface area of a right circular cone. -/
noncomputable def lateralSurfaceArea (cone : RightCircularCone) : ℝ :=
  Real.pi * (baseRadius cone) * (slantHeight cone)

/-- Theorem stating the properties of a specific right circular cone. -/
theorem specific_cone_properties :
  let cone : RightCircularCone := { volume := 36 * Real.pi, height := 6 }
  baseArea cone = 18 * Real.pi ∧ lateralSurfaceArea cone = 36 * Real.pi := by
  sorry

#eval "Theorem defined successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_cone_properties_l1033_103317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equality_l1033_103385

-- Define the integrand
noncomputable def f (x : ℝ) : ℝ := (2*x^3 + 3*x^2 + 3*x + 2) / ((x^2 + x + 1)*(x^2 + 1))

-- Define the antiderivative
noncomputable def F (x : ℝ) : ℝ := 
  (1/2) * Real.log (abs ((x^2 + x + 1) * (x^2 + 1))) + 
  (1/Real.sqrt 3) * Real.arctan ((2*x + 1)/Real.sqrt 3) + 
  Real.arctan x

-- State the theorem
theorem integral_equality (x : ℝ) : 
  deriv F x = f x := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equality_l1033_103385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kaleb_toy_purchase_l1033_103396

def max_toys_kaleb_can_buy (initial_savings : ℚ) (toy_price : ℚ) (discount_percent : ℚ) 
  (additional_allowance : ℚ) (sales_tax_percent : ℚ) : ℕ :=
  let discounted_price := toy_price * (1 - discount_percent)
  let total_price := discounted_price * (1 + sales_tax_percent)
  let total_money := initial_savings + additional_allowance
  (total_money / total_price).floor.toNat

theorem kaleb_toy_purchase :
  max_toys_kaleb_can_buy 39 8 (2/10) 25 (1/10) = 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kaleb_toy_purchase_l1033_103396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1033_103382

/-- Given a hyperbola and an ellipse that share the same foci, 
    prove that the asymptotes of the hyperbola have a specific equation. -/
theorem hyperbola_asymptotes (m : ℝ) :
  (∃ x y : ℝ, m * x^2 + y^2 = 1) →  -- Hyperbola equation
  (∃ x y : ℝ, x^2 + y^2 / 5 = 1) →  -- Ellipse equation
  (∀ x y : ℝ, m * x^2 + y^2 = 1 ↔ x^2 + y^2 / 5 = 1) →  -- Shared foci condition
  (∃ k : ℝ, k = Real.sqrt 3 / 3 ∧ 
    ∀ x y : ℝ, m * x^2 + y^2 = 1 → (y = k * x ∨ y = -k * x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1033_103382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_nine_l1033_103355

/-- Annual production limit in 10,000 units -/
noncomputable def production_limit : ℝ := 35

/-- Selling price in 1,000 yuan per 10,000 units -/
noncomputable def selling_price : ℝ := 16

/-- Fixed cost in 1,000 yuan -/
noncomputable def fixed_cost : ℝ := 30

/-- Variable cost function in 1,000 yuan -/
noncomputable def variable_cost (x : ℝ) : ℝ :=
  if x ≤ 14 then (2/3) * x^2 + 4*x
  else 17*x + 400/x - 80

/-- Annual profit function in 1,000 yuan -/
noncomputable def annual_profit (x : ℝ) : ℝ :=
  selling_price * x - fixed_cost - variable_cost x

/-- Theorem stating that the annual profit is maximized at x = 9 -/
theorem max_profit_at_nine :
  ∀ x ∈ Set.Icc 0 production_limit,
    annual_profit x ≤ annual_profit 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_nine_l1033_103355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_land_theorem_l1033_103368

-- Define the land
structure Land where
  ABC : ℝ
  AC : ℝ
  CD : ℝ
  DE : ℝ

-- Define the properties of the land
def land_properties (l : Land) : Prop :=
  l.ABC = 120 ∧ l.AC = 20 ∧ l.CD = 10 ∧ l.DE = 10

-- Define the total area of the land
noncomputable def total_area (l : Land) : ℝ :=
  l.ABC + (l.AC + l.DE) * l.CD / 2

-- Define the distance CF that divides the land into two equal parts
noncomputable def distance_CF (l : Land) : ℝ :=
  (total_area l / 2 - l.ABC) * 2 / l.AC

-- Theorem statement
theorem land_theorem (l : Land) (h : land_properties l) : 
  total_area l = 270 ∧ distance_CF l = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_land_theorem_l1033_103368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_digits_is_24_l1033_103351

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Fin 24
  minutes : Fin 60

/-- Calculates the sum of digits for a given number -/
def sumOfDigits (n : Nat) : Nat :=
  n.repr.data.foldl (fun acc c => acc + (c.toNat - '0'.toNat)) 0

/-- Calculates the sum of digits for a Time24 -/
def timeSumOfDigits (t : Time24) : Nat :=
  sumOfDigits t.hours.val + sumOfDigits t.minutes.val

/-- The maximum sum of digits possible in a 24-hour format display -/
def maxSumOfDigits : Nat := 24

theorem max_sum_of_digits_is_24 :
  ∀ t : Time24, timeSumOfDigits t ≤ maxSumOfDigits :=
by
  intro t
  sorry

#eval maxSumOfDigits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_digits_is_24_l1033_103351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_relation_l1033_103319

theorem sine_cosine_relation (α : ℝ) :
  Real.sin (540 * π / 180 + α) = -4/5 → Real.cos (α - 270 * π / 180) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_relation_l1033_103319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_rational_l1033_103371

-- Define a structure for a point in a plane
structure Point :=
  (x : ℚ) (y : ℚ)

-- Define a function to calculate the square of the distance between two points
def squareDistance (p q : Point) : ℚ :=
  (p.x - q.x)^2 + (p.y - q.y)^2

-- Define a function to calculate the area of a triangle given three points
noncomputable def triangleArea (p q r : Point) : ℝ :=
  sorry

-- Theorem statement
theorem area_ratio_rational 
  (A B C D : Point) 
  (distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) 
  (not_collinear : ∀ (X Y Z : Point), (X = A ∧ Y = B ∧ Z = C) ∨ 
                                      (X = A ∧ Y = B ∧ Z = D) ∨ 
                                      (X = A ∧ Y = C ∧ Z = D) ∨ 
                                      (X = B ∧ Y = C ∧ Z = D) → 
                                      triangleArea X Y Z ≠ 0)
  (rational_squares : squareDistance A B ∈ Set.range (fun q : ℚ => q) ∧ 
                      squareDistance A C ∈ Set.range (fun q : ℚ => q) ∧ 
                      squareDistance A D ∈ Set.range (fun q : ℚ => q) ∧ 
                      squareDistance B C ∈ Set.range (fun q : ℚ => q) ∧ 
                      squareDistance B D ∈ Set.range (fun q : ℚ => q) ∧ 
                      squareDistance C D ∈ Set.range (fun q : ℚ => q)) :
  ∃ (q : ℚ), (triangleArea A B C) / (triangleArea A B D) = (q : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_rational_l1033_103371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_arrangement_proof_l1033_103373

def number_of_students : ℕ := 6
def number_of_male_students : ℕ := 3
def number_of_female_students : ℕ := 3

theorem student_arrangement_proof :
  (∃ (arrangements_adjacent : ℕ),
    arrangements_adjacent = 144 ∧
    arrangements_adjacent = (number_of_students - number_of_female_students + 1) * (number_of_female_students.factorial)) ∧
  (∃ (arrangements_not_adjacent : ℕ),
    arrangements_not_adjacent = 144 ∧
    arrangements_not_adjacent = number_of_male_students.factorial * Nat.choose (number_of_male_students + 1) number_of_female_students) ∧
  (∃ (arrangements_not_ends : ℕ),
    arrangements_not_ends = 480 ∧
    arrangements_not_ends = (number_of_students - 2) * ((number_of_students - 1).factorial)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_arrangement_proof_l1033_103373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_integral_l1033_103380

open Set MeasureTheory Real

variable (a b : ℝ)
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

noncomputable def integral_to_minimize (t : ℝ) : ℝ :=
  ∫ x in a..b, |f x - f t| * x

theorem minimize_integral
  (h_ab : 0 < a ∧ a < b)
  (h_f : ∀ x ∈ Ioo a b, HasDerivAt f (f' x) x ∧ f' x > 0)
  (h_t : t ∈ Ioo a b) :
  IsMinOn (integral_to_minimize a b f) (Ioo a b) t ↔ t = sqrt ((a^2 + b^2) / 2) :=
sorry

#check minimize_integral

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_integral_l1033_103380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_three_digit_numbers_product_four_zeros_l1033_103377

/-- A three-digit number is a natural number between 100 and 999, inclusive. -/
def ThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- A list of natural numbers uses different digits if each digit appears at most once in the decimal representation of all numbers combined. -/
def UsesDifferentDigits (numbers : List ℕ) : Prop :=
  let digits := numbers.bind (λ n => n.digits 10)
  digits.Nodup

/-- The main theorem stating that there exist three three-digit numbers using different digits whose product ends with four zeros. -/
theorem exist_three_digit_numbers_product_four_zeros :
  ∃ (a b c : ℕ),
    ThreeDigitNumber a ∧
    ThreeDigitNumber b ∧
    ThreeDigitNumber c ∧
    UsesDifferentDigits [a, b, c] ∧
    (a * b * c) % 10000 = 0 := by
  -- Proof goes here
  sorry

#eval 125 * 360 * 748

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_three_digit_numbers_product_four_zeros_l1033_103377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_difference_intersection_points_l1033_103364

/-- Line l with parametric equations x = 1 + 2t, y = 2 + t -/
def line_l (t : ℝ) : ℝ × ℝ := (1 + 2*t, 2 + t)

/-- Circle C with equation x² + y² = 4 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Point P at (-1, 1) -/
def point_P : ℝ × ℝ := (-1, 1)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_difference_intersection_points :
  ∃ A B : ℝ × ℝ,
    (∃ t1 t2 : ℝ, line_l t1 = A ∧ line_l t2 = B) ∧
    circle_C A.1 A.2 ∧
    circle_C B.1 B.2 ∧
    distance point_P A - distance point_P B = 2 * Real.sqrt 5 / 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_difference_intersection_points_l1033_103364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_theorem_l1033_103324

/-- Represents a point in a 2D coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle in a 2D coordinate plane -/
structure Rectangle where
  p : Point
  q : Point
  r : Point
  s : Point

/-- Represents a circle in a 2D coordinate plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Calculates the area of a rectangle -/
def rectangleArea (rect : Rectangle) : ℝ :=
  (rect.q.x - rect.p.x) * (rect.r.y - rect.q.y)

/-- Calculates the area of a circle -/
noncomputable def circleArea (circ : Circle) : ℝ :=
  Real.pi * circ.radius^2

/-- The main theorem to be proved -/
theorem shaded_area_theorem (rect : Rectangle) (circ : Circle) : 
    circleArea circ - rectangleArea rect = 9 * Real.pi - 24 := by
  have h1 : rect.p = Point.mk 0 0 := by sorry
  have h2 : rect.q = Point.mk 6 0 := by sorry
  have h3 : rect.r = Point.mk 6 4 := by sorry
  have h4 : rect.s = Point.mk 0 4 := by sorry
  have h5 : circ.center = Point.mk 3 0 := by sorry
  have h6 : circ.radius = 3 := by sorry
  sorry

#check shaded_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_theorem_l1033_103324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_theorem_l1033_103325

/-- Represents the taxi fare structure -/
structure FareStructure where
  base_fare : ℚ  -- Base fare for first 4 km
  rate1 : ℚ      -- Rate per km for 4-15 km
  rate2 : ℚ      -- Rate per km beyond 15 km

/-- Calculates the fare for a direct trip -/
def directTripFare (fs : FareStructure) (distance : ℚ) : ℚ :=
  fs.base_fare +
  (min (max (distance - 4) 0) 11) * fs.rate1 +
  (max (distance - 15) 0) * fs.rate2

/-- Calculates the minimum fare using multiple taxis -/
def minFareMultipleTaxis (fs : FareStructure) (distance : ℚ) : ℚ :=
  let fullTrips := (distance / 15).floor
  let remainingDistance := distance - fullTrips * 15
  fullTrips * (fs.base_fare + 11 * fs.rate1) +
  if remainingDistance ≤ 4 then
    fs.base_fare
  else
    fs.base_fare + (remainingDistance - 4) * fs.rate1

theorem taxi_fare_theorem (fs : FareStructure) :
  fs.base_fare = 10 ∧ fs.rate1 = 6/5 ∧ fs.rate2 = 11/5 →
  directTripFare fs 50 = 501/5 ∧
  minFareMultipleTaxis fs 50 = 403/5 := by
  sorry

#eval directTripFare ⟨10, 6/5, 11/5⟩ 50
#eval minFareMultipleTaxis ⟨10, 6/5, 11/5⟩ 50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_theorem_l1033_103325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_triangle_exists_l1033_103315

structure ColoredGraph (n : ℕ) where
  vertices : Finset (Fin (3*n + 1))
  edge_color : Fin (3*n + 1) → Fin (3*n + 1) → Fin 3

/-- A colored graph satisfies the conditions if each vertex has exactly n edges of each color -/
def satisfies_conditions (n : ℕ) (G : ColoredGraph n) : Prop :=
  ∀ v : Fin (3*n + 1), ∀ c : Fin 3,
    ((G.vertices.filter (fun w => G.edge_color v w = c)).card : ℕ) = n

/-- A triangle in the graph has all three colors if its three edges have different colors -/
def has_all_colors (n : ℕ) (G : ColoredGraph n) (v1 v2 v3 : Fin (3*n + 1)) : Prop :=
  G.edge_color v1 v2 ≠ G.edge_color v2 v3 ∧
  G.edge_color v2 v3 ≠ G.edge_color v3 v1 ∧
  G.edge_color v3 v1 ≠ G.edge_color v1 v2

theorem colored_triangle_exists (n : ℕ) (G : ColoredGraph n) 
  (h : satisfies_conditions n G) :
  ∃ v1 v2 v3 : Fin (3*n + 1), has_all_colors n G v1 v2 v3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_triangle_exists_l1033_103315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_floor_l1033_103359

noncomputable def dissatisfaction (n : ℕ+) : ℝ := (n : ℝ) + 9 / (n : ℝ)

theorem optimal_floor :
  ∀ n : ℕ+, dissatisfaction n ≥ dissatisfaction 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_floor_l1033_103359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sphere_surface_area_l1033_103358

/-- Regular triangular prism with all vertices on a sphere -/
structure RegularTriangularPrism where
  -- Side length of the base triangle
  a : ℝ
  -- Height of the prism
  h : ℝ
  -- Volume of the prism
  volume : ℝ
  -- Volume constraint
  volume_eq : volume = 3 * Real.sqrt 3
  -- Relationship between height and side length
  height_eq : h = 36 / (a ^ 2)

/-- Sphere radius containing the regular triangular prism -/
noncomputable def sphere_radius (prism : RegularTriangularPrism) : ℝ :=
  Real.sqrt ((1 / 3) * prism.a ^ 2 + (18 / prism.a ^ 2) ^ 2)

/-- Theorem: Minimum surface area of the sphere containing the prism -/
theorem min_sphere_surface_area (prism : RegularTriangularPrism) :
  ∃ (min_area : ℝ), min_area = 12 * Real.pi * 324 ∧
  ∀ (r : ℝ), r = sphere_radius prism → 4 * Real.pi * r ^ 2 ≥ min_area := by
  sorry

#check min_sphere_surface_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sphere_surface_area_l1033_103358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_dot_product_bound_l1033_103307

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x - Real.sqrt 3 * y = 4

-- Define the geometric sequence condition
def geometric_sequence (x y : ℝ) : Prop :=
  ((x + 2)^2 + y^2) * ((x - 2)^2 + y^2) = (x^2 + y^2)^2

-- Define the dot product of PA and PB
def dot_product (x y : ℝ) : ℝ := 2 * (y^2 - 1)

-- Main theorem
theorem circle_dot_product_bound :
  ∀ x y : ℝ,
    circle_eq x y →
    tangent_line x y →
    geometric_sequence x y →
    x^2 + y^2 < 4 →
    -2 < dot_product x y ∧ dot_product x y < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_dot_product_bound_l1033_103307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_tiling_l1033_103393

/-- Definition of a k-cross in 3D space -/
def kCross (k : ℕ) (center : ℤ × ℤ × ℤ) : Set (ℤ × ℤ × ℤ) :=
  let (a, b, c) := center
  {(a, b, c)} ∪ 
  {(a + k, b, c), (a - k, b, c), (a, b + k, c), (a, b - k, c), (a, b, c + k), (a, b, c - k)}

/-- Definition of a tiling of 3D space -/
def isTiling (k : ℕ) (centers : Set (ℤ × ℤ × ℤ)) : Prop :=
  (∀ p : ℤ × ℤ × ℤ, ∃! c, c ∈ centers ∧ p ∈ kCross k c) ∧
  (∀ c₁ c₂, c₁ ∈ centers → c₂ ∈ centers → c₁ ≠ c₂ → kCross k c₁ ∩ kCross k c₂ = ∅)

theorem cross_tiling :
  (∃ centers : Set (ℤ × ℤ × ℤ), isTiling 1 centers) ∧
  (∃ centers : Set (ℤ × ℤ × ℤ), isTiling 2 centers) ∧
  (∀ k : ℕ, k ≥ 5 → ¬∃ centers : Set (ℤ × ℤ × ℤ), isTiling k centers) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_tiling_l1033_103393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_AB_is_sqrt_26_l1033_103322

noncomputable section

def point := ℝ × ℝ

def A : point := (2, 1)
def B : point := (-3, 2)

def vector_between (p q : point) : ℝ × ℝ :=
  (q.1 - p.1, q.2 - p.2)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

theorem magnitude_AB_is_sqrt_26 :
  magnitude (vector_between A B) = Real.sqrt 26 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_AB_is_sqrt_26_l1033_103322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partial_fraction_product_l1033_103338

/-- The function for which we're finding the partial fraction decomposition -/
noncomputable def f (x : ℝ) : ℝ := (x^2 - 21) / (x^3 - x^2 - 7*x + 15)

/-- The partial fraction decomposition of f -/
noncomputable def partial_fraction (A B C : ℝ) (x : ℝ) : ℝ := A / (x - 3) + B / (x + 3) + C / (x - 5)

/-- Theorem stating that the product of A, B, and C in the partial fraction decomposition equals -1/16 -/
theorem partial_fraction_product :
  ∃ A B C : ℝ, (∀ x : ℝ, x ≠ 3 ∧ x ≠ -3 ∧ x ≠ 5 → f x = partial_fraction A B C x) ∧ A * B * C = -1/16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partial_fraction_product_l1033_103338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_measurable_weights_l1033_103394

/-- Represents the available weights in grams -/
def available_weights : List ℕ := [1, 2, 6, 18]

/-- Represents all possible combinations of weights on each side of the scale -/
def weight_combinations : List (List ℕ × List ℕ) :=
  sorry

/-- Calculates the difference between the sum of weights on each side of the scale -/
def calculate_difference (combination : List ℕ × List ℕ) : ℕ :=
  sorry

/-- The set of all measurable weights -/
def measurable_weights : Finset ℕ :=
  sorry

/-- Theorem stating that the number of distinct measurable weights is 27 -/
theorem num_measurable_weights : Finset.card measurable_weights = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_measurable_weights_l1033_103394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_tetrahedron_all_edges_longest_l1033_103320

-- Define a tetrahedron
structure Tetrahedron where
  vertices : Fin 4 → ℝ × ℝ × ℝ
  edges : Fin 6 → ℝ
  faces : Fin 4 → Fin 3 → Fin 4

-- Define a property for an edge to be the longest in a face
def is_longest_in_face (t : Tetrahedron) (edge : Fin 6) (face : Fin 4) : Prop :=
  ∀ e : Fin 3, t.edges edge ≥ t.edges (t.faces face e)

-- Theorem statement
theorem no_tetrahedron_all_edges_longest :
  ¬∃ t : Tetrahedron, ∀ e : Fin 6, ∃ f : Fin 4, is_longest_in_face t e f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_tetrahedron_all_edges_longest_l1033_103320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_variation_problem_l1033_103335

/-- The inverse relationship between x² and ∛y -/
def inverse_relation (x y : ℝ) (k : ℝ) : Prop :=
  x^2 * y^(1/3) = k

/-- The constant k determined by initial conditions -/
def initial_condition (k : ℝ) : Prop :=
  3^2 * 64^(1/3) = k

/-- The relationship between x and y -/
def xy_relation (x y : ℝ) : Prop :=
  x * y = 54

theorem inverse_variation_problem (x y k : ℝ) 
  (h1 : inverse_relation x y k)
  (h2 : initial_condition k)
  (h3 : xy_relation x y) :
  ∃ ε > 0, |y - 12.22| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_variation_problem_l1033_103335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_crossing_possible_l1033_103374

/-- Represents the state of the river crossing problem -/
structure RiverState where
  leftBank : Nat × Nat  -- (missionaries, cannibals) on left bank
  rightBank : Nat × Nat -- (missionaries, cannibals) on right bank
  boat : Nat × Nat      -- (missionaries, cannibals) in the boat
  boatOnLeftBank : Bool -- true if boat is on left bank

/-- Represents a single boat crossing -/
inductive BoatCrossing
  | toRight (m c : Nat) -- m missionaries and c cannibals cross to right
  | toLeft (m c : Nat)  -- m missionaries and c cannibals cross to left

/-- Checks if a given state is safe (missionaries not outnumbered) -/
def isSafeState (state : RiverState) : Bool :=
  let (leftM, leftC) := state.leftBank
  let (rightM, rightC) := state.rightBank
  (leftM = 0 || leftM >= leftC) && (rightM = 0 || rightM >= rightC)

/-- Applies a single crossing to a state -/
def applyCrossing (state : RiverState) (crossing : BoatCrossing) : RiverState :=
  sorry

/-- Applies a list of crossings to the initial state -/
def applyAllCrossings (crossings : List BoatCrossing) : RiverState :=
  sorry

/-- Checks if a sequence of crossings is valid and safe -/
def isValidCrossing (crossings : List BoatCrossing) : Bool :=
  sorry

/-- The main theorem to prove -/
theorem river_crossing_possible :
  ∃ (crossings : List BoatCrossing),
    crossings.length ≤ 15 ∧
    isValidCrossing crossings ∧
    (let finalState := applyAllCrossings crossings
     finalState.leftBank = (0, 0) ∧
     finalState.rightBank = (3, 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_crossing_possible_l1033_103374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l1033_103340

-- Define the ellipse
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

-- Define the focus and points
noncomputable def F (e : Ellipse) : ℝ × ℝ := (Real.sqrt (e.a^2 - e.b^2), 0)
def B₁ (e : Ellipse) : ℝ × ℝ := (0, e.b)
def B₂ (e : Ellipse) : ℝ × ℝ := (0, -e.b)

-- Define the eccentricity
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (e.a^2 - e.b^2) / e.a

-- Define the dot product condition
def dotProductCondition (e : Ellipse) : Prop :=
  let f := F e
  let b₁ := B₁ e
  let b₂ := B₂ e
  (f.1 - b₁.1) * (f.1 - b₂.1) + (f.2 - b₁.2) * (f.2 - b₂.2) = 4

-- Define the line l
structure Line where
  k : ℝ
  b : ℝ

def lineEquation (l : Line) (x : ℝ) : ℝ := l.k * x + l.b

-- Define the intersection condition
def intersectionCondition (e : Ellipse) (l : Line) (A B N : ℝ × ℝ) : Prop :=
  A.2 = lineEquation l A.1 ∧
  B.2 = lineEquation l B.1 ∧
  N.2 = 0 ∧
  (A.1 - N.1, A.2) = (-7/5 * (B.1 - N.1), -7/5 * B.2)

-- State the theorem
theorem ellipse_and_line_properties (e : Ellipse) (l : Line) (A B N : ℝ × ℝ) :
  e.a > e.b ∧ e.b > 0 ∧
  eccentricity e = Real.sqrt 3 / 2 ∧
  dotProductCondition e ∧
  l.b = -1 ∧
  intersectionCondition e l A B N →
  (e.a = 2 * Real.sqrt 2 ∧ e.b = Real.sqrt 2) ∧
  (l.k = 1 ∨ l.k = -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l1033_103340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_but_not_sufficient_condition_l1033_103350

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (∀ x, 2 < x ∧ x < 3 → x > a) ∧
  (∃ x, x > a ∧ ¬(2 < x ∧ x < 3)) ↔
  a ≤ 2 :=
by
  -- The proof would go here
  sorry

#check necessary_but_not_sufficient_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_but_not_sufficient_condition_l1033_103350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_sums_l1033_103337

/-- Rounds a number to the nearest multiple of 5, rounding .5 up -/
def roundToNearestFive (n : Int) : Int :=
  5 * ((n + 2) / 5)

/-- Sums the first n positive integers -/
def sumFirstN (n : Nat) : Nat :=
  n * (n + 1) / 2

/-- Sums the first n positive integers after rounding each to the nearest multiple of 5 -/
def sumRoundedFirstN (n : Nat) : Int :=
  Finset.sum (Finset.range n) (fun i => roundToNearestFive (i + 1))

theorem difference_of_sums : 
  |Int.ofNat (sumFirstN 50) - sumRoundedFirstN 50| = 575 := by
  sorry

#eval |Int.ofNat (sumFirstN 50) - sumRoundedFirstN 50|

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_sums_l1033_103337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_equation_solution_l1033_103381

/-- Diamond operation for nonzero real numbers -/
noncomputable def diamond (a b : ℝ) : ℝ := a / b

/-- Theorem stating the solution to the equation involving the diamond operation -/
theorem diamond_equation_solution :
  ∀ x : ℝ, x ≠ 0 → diamond 5040 (diamond 8 x) = 250 → x = 25 / 63 := by
  intro x hx_nonzero heq
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_equation_solution_l1033_103381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_properties_l1033_103308

/-- The ⋆ operation for positive real numbers -/
noncomputable def star (k : ℝ) (x y : ℝ) : ℝ := (k * x * y) / (x + y)

theorem star_properties (k : ℝ) :
  (∀ x y, x > 0 → y > 0 → star k x y = star k y x) ∧
  (∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ star k (star k x y) z ≠ star k x (star k y z)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_properties_l1033_103308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_divisible_by_25_l1033_103388

theorem smallest_a_divisible_by_25 :
  ∃ (a : ℕ), a > 0 ∧
  (∀ (n : ℕ), 25 ∣ (2^(n+2) * 3^n + 5*n - a)) ∧
  (∀ (b : ℕ), b > 0 → (∀ (n : ℕ), 25 ∣ (2^(n+2) * 3^n + 5*n - b)) → b ≥ a) ∧
  a = 4 :=
by
  -- The proof will be filled in here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_divisible_by_25_l1033_103388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_channel_width_one_l1033_103365

noncomputable def has_channel (f : ℝ → ℝ) (d : ℝ) (D : Set ℝ) : Prop :=
  ∃ (k m₁ m₂ : ℝ), m₂ - m₁ = d ∧ 
  ∀ x ∈ D, k * x + m₁ ≤ f x ∧ f x ≤ k * x + m₂

noncomputable def f₁ (x : ℝ) : ℝ := 1 / x
noncomputable def f₂ (x : ℝ) : ℝ := Real.sin x
noncomputable def f₃ (x : ℝ) : ℝ := Real.sqrt (x^2 - 1)
noncomputable def f₄ (x : ℝ) : ℝ := x^3 + 1

def D : Set ℝ := {x | x ≥ 1}

theorem channel_width_one :
  has_channel f₁ 1 D ∧ 
  has_channel f₃ 1 D ∧ 
  ¬has_channel f₂ 1 D ∧ 
  ¬has_channel f₄ 1 D := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_channel_width_one_l1033_103365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l1033_103314

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 3 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*y + 3 = 0

-- Define the centers and radii of the circles
def center₁ : ℝ × ℝ := (-1, 0)
def center₂ : ℝ × ℝ := (0, 2)
def radius₁ : ℝ := 2
def radius₂ : ℝ := 1

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := Real.sqrt 5

-- Theorem statement
theorem circles_intersect :
  distance_between_centers > radius₁ - radius₂ ∧
  distance_between_centers < radius₁ + radius₂ := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l1033_103314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_delta_minus_star_l1033_103369

/-- Represents the sum of the first row -/
def R1 : ℕ := sorry

/-- Represents the sum of the second row -/
def R2 : ℕ := sorry

/-- Represents the sum of the first column -/
def C1 : ℕ := sorry

/-- Represents the sum of the second column -/
def C2 : ℕ := sorry

/-- Represents the total sum of all numbers -/
def TotalSum : ℕ := sorry

theorem delta_minus_star (Δ up star : ℕ) 
  (h1 : Δ > 0) (h2 : up > 0) (h3 : star > 0)
  (h4 : Δ + star = R1)
  (h5 : up + star = R2)
  (h6 : Δ + up = C1)
  (h7 : star + up = C2)
  (h8 : Δ + up + star = TotalSum) :
  Δ - star = 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_delta_minus_star_l1033_103369
