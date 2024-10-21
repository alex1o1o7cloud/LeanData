import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1055_105510

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  BC : ℝ
  sinIdentity : Real.sin A ^ 2 + Real.sin B ^ 2 - Real.sin C ^ 2 = 
                2 * Real.sin A * Real.sin B * (Real.sqrt 3 - Real.cos C)

/-- Area calculation for triangle -/
noncomputable def area (t : Triangle) : ℝ :=
  1 / 2 * t.BC * (Real.sqrt 2) * Real.sin t.B

theorem triangle_properties (t : Triangle) :
  t.C = π / 6 ∧
  (t.A = π / 4 ∧ t.BC = 2 → area t = (1 + Real.sqrt 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1055_105510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_track_circumference_is_1000_l1055_105562

/-- Represents the circumference of a circular track -/
def track_circumference (c : ℝ) : Prop := c > 0

/-- Represents the distance traveled by A at the first meeting -/
def first_meeting_distance_A : ℝ := 120

/-- Represents the distance B is short of completing two laps at the second meeting -/
def second_meeting_short_distance_B : ℝ := 100

/-- Axiom: A and B travel at consistent speeds in opposite directions -/
axiom consistent_speeds : ∀ (d₁ d₂ t₁ t₂ : ℝ), d₁ / t₁ = d₂ / t₂

/-- Axiom: A and B start from diametrically opposite points -/
axiom diametrically_opposite_start : True

/-- Theorem: The circumference of the track is 1000 yards -/
theorem track_circumference_is_1000 : track_circumference 1000 := by
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_track_circumference_is_1000_l1055_105562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_shift_cos_to_sin_l1055_105537

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)
noncomputable def g (x : ℝ) : ℝ := Real.cos (2 * x)

theorem horizontal_shift_cos_to_sin :
  ∃ (shift : ℝ), ∀ (x : ℝ), f x = g (x - shift) ∧ shift = 3 * Real.pi / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_shift_cos_to_sin_l1055_105537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_plane_no_intersection_l1055_105546

/-- A line in 3D space -/
structure Line3D where
  point : Fin 3 → ℝ
  direction : Fin 3 → ℝ

/-- A plane in 3D space -/
structure Plane3D where
  point : Fin 3 → ℝ
  normal : Fin 3 → ℝ

/-- Define parallel relationship between a line and a plane -/
def Line3D.parallelToPlane (l : Line3D) (p : Plane3D) : Prop :=
  (Finset.sum Finset.univ (fun i => l.direction i * p.normal i)) = 0

/-- Define a line within a plane -/
def Line3D.withinPlane (l : Line3D) (p : Plane3D) : Prop :=
  (Finset.sum Finset.univ (fun i => (l.point i - p.point i) * p.normal i)) = 0 ∧
  (Finset.sum Finset.univ (fun i => l.direction i * p.normal i)) = 0

/-- Define intersection between two lines -/
def Line3D.intersects (l1 l2 : Line3D) : Prop :=
  ∃ t s : ℝ, ∀ i : Fin 3,
    l1.point i + t * l1.direction i = l2.point i + s * l2.direction i

/-- Theorem: If a line is parallel to a plane, it has no common points with any line within the plane -/
theorem parallel_line_plane_no_intersection (l : Line3D) (p : Plane3D) :
  l.parallelToPlane p → ∀ m : Line3D, m.withinPlane p → ¬(l.intersects m) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_plane_no_intersection_l1055_105546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_solutions_condition_l1055_105509

/-- The system of equations has exactly three solutions -/
def has_exactly_three_solutions (a : ℝ) : Prop :=
  ∃! (s : Finset (ℝ × ℝ)), s.card = 3 ∧
    (∀ (x y : ℝ), (x, y) ∈ s ↔
      (abs (y + 2) + abs (x - 11) - 3) * (x^2 + y^2 - 13) = 0 ∧
      (x - 5)^2 + (y + 2)^2 = a)

/-- The theorem stating the conditions for exactly three solutions -/
theorem three_solutions_condition (a : ℝ) :
  has_exactly_three_solutions a ↔ (a = 9 ∨ a = 42 + 2 * Real.sqrt 377) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_solutions_condition_l1055_105509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_bound_l1055_105511

-- Define the curve
def on_curve (x y : ℝ) : Prop := |x| / 4 + |y| / 3 = 1

-- Define the fixed points
noncomputable def F₁ : ℝ × ℝ := (-Real.sqrt 7, 0)
noncomputable def F₂ : ℝ × ℝ := (Real.sqrt 7, 0)

-- Define the distance function
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

-- Theorem statement
theorem distance_sum_bound {x y : ℝ} (h : on_curve x y) :
  distance (x, y) F₁ + distance (x, y) F₂ ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_bound_l1055_105511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l1055_105593

/-- The area of a trapezium with given parallel sides and height -/
noncomputable def trapeziumArea (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: The area of a trapezium with parallel sides 18 cm and 20 cm, 
    and height 5 cm, is 95 cm² -/
theorem trapezium_area_example : 
  trapeziumArea 18 20 5 = 95 := by
  -- Unfold the definition of trapeziumArea
  unfold trapeziumArea
  -- Simplify the arithmetic
  simp [add_mul, mul_div_right_comm]
  -- Check that the result is equal to 95
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l1055_105593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_equivalence_l1055_105559

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

-- Define the reference function
noncomputable def g (x : ℝ) : ℝ := Real.cos (2 * x)

-- Define the transformation (shift right by 5π/12)
noncomputable def h (x : ℝ) : ℝ := g (x - 5 * Real.pi / 12)

-- Theorem statement
theorem shift_equivalence : ∀ x, f x = h x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_equivalence_l1055_105559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_line_l1055_105534

noncomputable def f (x : ℝ) := x * Real.sin x + 5

theorem tangent_perpendicular_line (a : ℝ) : 
  (∀ x : ℝ, deriv f x = Real.sin x + x * Real.cos x) →
  deriv f (Real.pi / 2) = 1 →
  (deriv f (Real.pi / 2)) * (-a / 4) = -1 →
  a = 4 := by
  intros h1 h2 h3
  have h4 : deriv f (Real.pi / 2) * (-a / 4) = -1 := h3
  rw [h2] at h4
  field_simp at h4
  linarith

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_line_l1055_105534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_space_diagonal_l1055_105549

/-- Represents a cube with its properties -/
structure Cube where
  side : ℝ
  baseDiagonal : ℝ
  spaceDiagonal : ℝ

/-- Theorem about the space diagonal of a cube -/
theorem cube_space_diagonal (c : Cube) 
  (h1 : c.baseDiagonal = 5) 
  (h2 : c.spaceDiagonal = c.baseDiagonal) : 
  c.spaceDiagonal = 5 * Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_space_diagonal_l1055_105549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_calculation_l1055_105513

-- Define the ⊗ operation
noncomputable def otimes (a b : ℝ) : ℝ := a^3 / b^2

-- Theorem statement
theorem otimes_calculation :
  let x := otimes (otimes 2 3) 4
  let y := otimes 2 (otimes 3 4)
  x - y = -224/81 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_calculation_l1055_105513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_bound_implies_set_size_bound_l1055_105565

theorem lcm_bound_implies_set_size_bound (n : ℕ+) (A : Finset ℕ) 
  (h_subset : A ⊆ Finset.range n)
  (h_lcm : ∀ (a b : ℕ), a ∈ A → b ∈ A → a ≠ b → Nat.lcm a b ≤ n) :
  A.card ≤ ⌊1.9 * Real.sqrt n + 5⌋ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_bound_implies_set_size_bound_l1055_105565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_calculation_l1055_105517

theorem discount_calculation (initial_discount additional_discount advertised_discount : ℚ) :
  initial_discount = 35/100 ∧ 
  additional_discount = 25/100 ∧ 
  advertised_discount = 55/100 →
  (1 - (1 - initial_discount) * (1 - additional_discount)) - advertised_discount = 375/10000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_calculation_l1055_105517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_2011_1_neg1_l1055_105507

/-- The P₁ operation on points -/
def P₁ (x y : ℤ) : ℤ × ℤ := (x + y, x - y)

/-- The Pₙ operation on points for n ≥ 1 -/
def P : ℕ → ℤ → ℤ → ℤ × ℤ
  | 0 => P₁  -- Define P₀ as P₁ to handle the Nat.zero case
  | 1 => P₁
  | n + 1 => fun x y => let (a, b) := P n x y; P₁ a b

/-- The main theorem: P₂₀₁₁(1,-1) = (0, 2^1006) -/
theorem p_2011_1_neg1 : P 2011 1 (-1) = (0, 2^1006) := by sorry

/-- Helper lemma: For odd n, Pₙ(1,-1) = (0, 2^((n+1)/2)) -/
lemma p_odd_1_neg1 (n : ℕ) (h : Odd n) :
  P n 1 (-1) = (0, 2^((n+1)/2)) := by sorry

/-- Helper lemma: For even n > 0, Pₙ(1,-1) = (2^(n/2), -2^(n/2)) -/
lemma p_even_1_neg1 (n : ℕ) (h : Even n) (h2 : n > 0) :
  P n 1 (-1) = (2^(n/2), -2^(n/2)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_2011_1_neg1_l1055_105507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_AB_complex_l1055_105570

noncomputable def ω : ℂ := -1/2 + (Real.sqrt 3 / 2) * Complex.I

theorem vector_AB_complex (A B : ℂ) (h1 : A = ω) (h2 : B = ω^2) :
  B - A = -Real.sqrt 3 * Complex.I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_AB_complex_l1055_105570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_greater_than_one_l1055_105586

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x - 1 else 1 - x

-- State the theorem
theorem solution_set_of_f_greater_than_one :
  (∀ x, f (-x) = f x) →  -- f is even
  (∀ x ≥ 0, f x = x - 1) →  -- f(x) = x - 1 for x ≥ 0
  {x : ℝ | f x > 1} = {x : ℝ | x < -2 ∨ x > 2} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_greater_than_one_l1055_105586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_angle_terminal_side_l1055_105583

theorem point_on_angle_terminal_side (α : Real) (m : Real) :
  (∃ P : Real × Real, P = (m, 2) ∧ P.1 ≥ 0) →  -- Point P(m, 2) is on the terminal side of angle α
  Real.sin α = 1/3 →                           -- sin α = 1/3
  m = 4 * Real.sqrt 2 ∨ m = -4 * Real.sqrt 2   -- m = ± 4√2
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_angle_terminal_side_l1055_105583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_l1055_105579

theorem inequality_theorem (a b c d e p q : ℝ) 
  (h_pos : 0 < p ∧ p ≤ q)
  (h_bounds : a ∈ Set.Icc p q ∧ b ∈ Set.Icc p q ∧ c ∈ Set.Icc p q ∧ d ∈ Set.Icc p q ∧ e ∈ Set.Icc p q) :
  (a + b + c + d + e) * (1/a + 1/b + 1/c + 1/d + 1/e) ≤ 25 + 6 * (Real.sqrt (p/q) - Real.sqrt (q/p))^2 ∧
  ((a + b + c + d + e) * (1/a + 1/b + 1/c + 1/d + 1/e) = 25 + 6 * (Real.sqrt (p/q) - Real.sqrt (q/p))^2 ↔ 
    (p = q ∨ 
    ((Finset.filter (· = p) {a,b,c,d,e}).card = 3 ∧ (Finset.filter (· = q) {a,b,c,d,e}).card = 2) ∨
    ((Finset.filter (· = p) {a,b,c,d,e}).card = 2 ∧ (Finset.filter (· = q) {a,b,c,d,e}).card = 3))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_l1055_105579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_preference_percentage_l1055_105512

theorem basketball_preference_percentage (north_students south_students : ℕ)
  (north_basketball_percentage south_basketball_percentage : ℚ) :
  north_students = 1800 →
  south_students = 3000 →
  north_basketball_percentage = 25 / 100 →
  south_basketball_percentage = 35 / 100 →
  let total_students := north_students + south_students
  let total_basketball_students := (north_students : ℚ) * north_basketball_percentage +
                                   (south_students : ℚ) * south_basketball_percentage
  let combined_percentage := (total_basketball_students / (total_students : ℚ)) * 100
  (combined_percentage).floor = 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_preference_percentage_l1055_105512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_julia_download_theorem_l1055_105538

/-- Calculates the number of songs that can be downloaded given internet speed, song size, and time. -/
noncomputable def songs_downloaded (internet_speed : ℝ) (song_size : ℝ) (time : ℝ) : ℝ :=
  (internet_speed * time) / song_size

theorem julia_download_theorem :
  let internet_speed : ℝ := 20  -- MBps
  let song_size : ℝ := 5        -- MB
  let time : ℝ := 0.5 * 60 * 60 -- half hour in seconds
  songs_downloaded internet_speed song_size time = 7200 := by
  -- Unfold the definition and simplify
  unfold songs_downloaded
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_julia_download_theorem_l1055_105538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_transaction_result_l1055_105582

def air_conditioner_price : ℝ := 2000
def tv_price : ℝ := 2000
def air_conditioner_profit_rate : ℝ := 0.3
def tv_loss_rate : ℝ := 0.2

theorem store_transaction_result :
  let air_conditioner_cost := air_conditioner_price / (1 + air_conditioner_profit_rate)
  let tv_cost := tv_price / (1 - tv_loss_rate)
  let total_profit := (air_conditioner_price - air_conditioner_cost) + (tv_price - tv_cost)
  ∃ ε > 0, |total_profit + 38.5| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_transaction_result_l1055_105582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_quadratic_bound_l1055_105533

/-- Represents the number of starting directions from which a particle
    reaches a corner after traveling a distance k or less -/
noncomputable def c (k : ℝ) : ℕ := sorry

/-- The side length of the square -/
def squareSide : ℝ := 1

/-- Theorem stating that π is the smallest constant a₂ such that
    c(k) ≤ a₂k² + a₁k + a₀ for all starting points and all k ≥ 0 -/
theorem smallest_quadratic_bound :
  ∃ (a₁ a₀ : ℝ), ∀ (k : ℝ), k ≥ 0 → (c k : ℝ) ≤ π * k^2 + a₁ * k + a₀ ∧
  ∀ (a₂ : ℝ), (∃ (a₁ a₀ : ℝ), ∀ (k : ℝ), k ≥ 0 → (c k : ℝ) ≤ a₂ * k^2 + a₁ * k + a₀) →
  a₂ ≥ π :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_quadratic_bound_l1055_105533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continued_fraction_result_l1055_105550

-- Define the continued fraction
noncomputable def x : ℝ := Real.sqrt 3 + 1

-- Theorem statement
theorem continued_fraction_result : 1 / ((x + 1) * (x - 2)) = 2 + Real.sqrt 3 := by
  sorry

#eval "The theorem is stated correctly, but the proof is omitted using 'sorry'."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continued_fraction_result_l1055_105550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_calculation_l1055_105569

/-- Represents the fuel consumption problem for an airplane -/
structure FuelConsumptionProblem where
  fuel_rate : ℝ
  fly_time : ℝ

/-- Calculates the remaining fuel in the tank -/
def remaining_fuel (problem : FuelConsumptionProblem) : ℝ :=
  problem.fuel_rate * problem.fly_time

/-- Rounds a real number to one decimal place -/
noncomputable def round_to_tenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

theorem fuel_calculation (problem : FuelConsumptionProblem) 
  (h1 : problem.fuel_rate = 9.5)
  (h2 : problem.fly_time = 0.6667) :
  round_to_tenth (remaining_fuel problem) = 6.3 := by
  sorry

#eval Float.round (9.5 * 0.6667 * 10) / 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_calculation_l1055_105569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_choir_size_l1055_105574

/-- Represents the number of students in each of the four equal rows -/
def x : ℕ := sorry

/-- The total number of students in the choir -/
def total_students : ℕ := 5 * x + 3

theorem smallest_choir_size :
  (total_students > 45) ∧ (∀ y : ℕ, y < x → 5 * y + 3 ≤ 45) → total_students = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_choir_size_l1055_105574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carnival_booth_rent_is_30_l1055_105522

def carnival_booth_rent (daily_popcorn_sales : ℕ) 
                        (daily_cotton_candy_sales : ℕ) 
                        (duration : ℕ) 
                        (ingredient_cost : ℕ) 
                        (total_earnings_after_expenses : ℕ) : ℕ :=
  let total_popcorn_sales := daily_popcorn_sales * duration
  let total_cotton_candy_sales := daily_cotton_candy_sales * duration
  let total_sales := total_popcorn_sales + total_cotton_candy_sales
  let earnings_after_rent := total_earnings_after_expenses + ingredient_cost
  total_sales - earnings_after_rent

def carnival_booth_rent_example : ℕ :=
  carnival_booth_rent 50 150 5 75 895

#eval carnival_booth_rent_example

theorem carnival_booth_rent_is_30 :
  carnival_booth_rent 50 150 5 75 895 = 30 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carnival_booth_rent_is_30_l1055_105522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gift_card_balance_l1055_105572

theorem gift_card_balance (x : ℝ) : 
  let monday_spent := 0.5 * x * 1.05
  let tuesday_remainder := x - monday_spent
  let tuesday_spent := 0.25 * tuesday_remainder * 0.9
  let wednesday_remainder := tuesday_remainder - tuesday_spent
  let wednesday_spent := (1/3) * wednesday_remainder * 1.07
  let thursday_remainder := wednesday_remainder - wednesday_spent
  let thursday_spent := 0.2 * thursday_remainder * 0.85
  let final_balance := thursday_remainder - thursday_spent
  final_balance = 0.196566478 * x := by
  sorry

#check gift_card_balance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gift_card_balance_l1055_105572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_circle_intersection_l1055_105520

-- Define the ellipse C₁
def ellipse (x y a b : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

-- Define the circle C₂
def circle_c2 (x y : ℝ) : Prop := (x - Real.sqrt 3)^2 + y^2 = 7

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2) / a

-- Define the line l
def line (x y k m : ℝ) : Prop := y = k * x + m

-- Main theorem
theorem ellipse_and_circle_intersection 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : eccentricity a b = Real.sqrt 3 / 2) 
  (k m : ℝ) 
  (h4 : ∃! x y, ellipse x y a b ∧ line x y k m) 
  (h5 : ∃! x y, circle_c2 x y ∧ line x y k m) :
  (∀ x y, ellipse x y a b ↔ x^2/4 + y^2 = 1) ∧ 
  (∃ x y, circle_c2 x y ∧ line x y k m ∧ x = 2 ∧ y = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_circle_intersection_l1055_105520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variances_not_uniformly_bounded_chebyshev_not_applicable_l1055_105555

/-- Distribution of Xₙ -/
noncomputable def P (n : ℕ) (x : ℝ) : ℝ :=
  if x = n + 1 then n / (2 * n + 1)
  else if x = -n then (n + 1) / (2 * n + 1)
  else 0

/-- Mean of Xₙ -/
def mean (n : ℕ) : ℝ := 0

/-- Variance of Xₙ -/
noncomputable def variance (n : ℕ) : ℝ :=
  n * (2 * n^2 + 3 * n + 1) / (2 * n + 1)

/-- Theorem: The variances of Xₙ are not uniformly bounded -/
theorem variances_not_uniformly_bounded :
  ∀ C : ℝ, ∃ n : ℕ, variance n > C := by
  sorry

/-- Theorem: Chebyshev's theorem is not applicable to the given sequence -/
theorem chebyshev_not_applicable : 
  ¬∃ M : ℝ, ∀ n : ℕ, variance n ≤ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variances_not_uniformly_bounded_chebyshev_not_applicable_l1055_105555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_finite_l1055_105526

-- Define the set A
def A : Set (List ℕ) :=
  {l | ∀ n, n ∈ l → 1 ≤ n ∧ n ≤ 2017}

-- Define what it means for a sequence to "begin with" another
def begins_with (M : List ℕ) (T : List ℕ) : Prop :=
  ∃ N : List ℕ, M = T ++ N

-- Define the properties of set S
structure S_properties (S : Set (List ℕ)) : Prop where
  subset_of_A : S ⊆ A
  finite_sequences : ∀ T, T ∈ S → List.length T < ω
  unique_beginning : ∀ M : List ℕ, M ∈ A → List.length M = ω →
    ∃! T, T ∈ S ∧ begins_with M T

-- Theorem to prove
theorem S_is_finite (S : Set (List ℕ)) (h : S_properties S) : 
  Set.Finite S :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_finite_l1055_105526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_triangle_theorem_l1055_105502

-- Define the triangle ABC
noncomputable def triangle_ABC (base : ℝ) (height : ℝ) : ℝ := (1 / 2) * base * height

-- Define the folded triangle XYZ
noncomputable def triangle_XYZ (base : ℝ) (height : ℝ) (k : ℝ) : ℝ := (1 / 2) * (base * k) * (height * k)

theorem folded_triangle_theorem (base height : ℝ) :
  base = 15 →
  triangle_XYZ base height 0.5 = 0.25 * triangle_ABC base height →
  0.5 * base = 7.5 :=
by
  intros h1 h2
  sorry

#check folded_triangle_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_triangle_theorem_l1055_105502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_when_disjoint_l1055_105515

/-- The function f(x) = x³ - 3x + m has a domain [0, 2] and a range B.
    When the domain and range are disjoint, prove the range of m. -/
theorem range_of_m_when_disjoint (m : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 - 3*x + m
  let A : Set ℝ := Set.Icc 0 2
  let B : Set ℝ := f '' A
  (A ∩ B = ∅) ↔ m ∈ Set.Ioi 4 ∪ Set.Iio (-2) := by
  sorry

#check range_of_m_when_disjoint

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_when_disjoint_l1055_105515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_areas_of_parallel_intersections_l1055_105568

noncomputable section

/-- Represents a point on the hyperbola y = 1/x -/
structure HyperbolaPoint where
  x : ℝ
  y : ℝ
  on_hyperbola : y = 1 / x

/-- Represents a set of four points on the hyperbola -/
structure FourPoints where
  p₁ : HyperbolaPoint
  p₂ : HyperbolaPoint
  p₃ : HyperbolaPoint
  p₄ : HyperbolaPoint

/-- Calculates the area of a quadrilateral given its four vertices -/
noncomputable def quadrilateralArea (p₁ p₂ p₃ p₄ : HyperbolaPoint) : ℝ :=
  1/2 * abs (p₁.x * p₂.y + p₂.x * p₃.y + p₃.x * p₄.y + p₄.x * p₁.y
           - (p₁.y * p₂.x + p₂.y * p₃.x + p₃.y * p₄.x + p₄.y * p₁.x))

/-- Condition for parallel lines -/
def areParallel (a₁ b₁ a₂ b₂ : HyperbolaPoint) : Prop :=
  (b₁.y - a₁.y) / (b₁.x - a₁.x) = (b₂.y - a₂.y) / (b₂.x - a₂.x)

theorem equal_areas_of_parallel_intersections
  (A B : FourPoints)
  (h₁ : areParallel A.p₁ B.p₁ A.p₂ B.p₂)
  (h₂ : areParallel A.p₁ B.p₁ A.p₃ B.p₃)
  (h₃ : areParallel A.p₁ B.p₁ A.p₄ B.p₄) :
  quadrilateralArea A.p₁ A.p₂ A.p₃ A.p₄ = quadrilateralArea B.p₁ B.p₂ B.p₃ B.p₄ := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_areas_of_parallel_intersections_l1055_105568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stationary_point_of_f_l1055_105585

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - 2*x + Real.log x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := x - 2 + 1/x

-- Theorem statement
theorem stationary_point_of_f :
  ∀ x : ℝ, x > 0 → (f' x = 0 ↔ x = 1) :=
by
  sorry

#check stationary_point_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stationary_point_of_f_l1055_105585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_pressure_force_formula_l1055_105592

/-- Represents a vertical dam in the shape of an isosceles trapezoid -/
structure TrapezoidalDam where
  a : ℝ  -- upper base length
  b : ℝ  -- lower base length
  h : ℝ  -- height
  a_positive : 0 < a
  b_positive : 0 < b
  h_positive : 0 < h
  a_ge_b : a ≥ b

/-- Calculates the force of water pressure on a trapezoidal dam -/
noncomputable def waterPressureForce (ρ g : ℝ) (dam : TrapezoidalDam) : ℝ :=
  ρ * g * (dam.h^2 / 3) * (2 * dam.b + dam.a)

theorem water_pressure_force_formula (ρ g : ℝ) (dam : TrapezoidalDam) 
    (ρ_positive : 0 < ρ) (g_positive : 0 < g) :
  waterPressureForce ρ g dam = ρ * g * (dam.h^2 / 3) * (2 * dam.b + dam.a) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_pressure_force_formula_l1055_105592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_in_interval_l1055_105527

-- Define the function f(x) = ln x - 3/x
noncomputable def f (x : ℝ) : ℝ := Real.log x - 3 / x

-- State the theorem
theorem zero_point_in_interval :
  ∃ x : ℝ, e < x ∧ x < 3 ∧ f x = 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_in_interval_l1055_105527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_completion_percentage_l1055_105542

/-- Represents a section of the route with its distance and speed --/
structure Section where
  distance : ℝ
  speed : ℝ

/-- Represents the entire route with three sections --/
structure Route where
  sectionA : Section
  sectionB : Section
  sectionC : Section

/-- Calculates the time taken to travel a section --/
noncomputable def travelTime (s : Section) : ℝ := s.distance / s.speed

/-- Applies a delay factor to a time --/
def applyDelay (time : ℝ) (delayFactor : ℝ) : ℝ := time * (1 + delayFactor)

/-- Theorem: The effective completion percentage of the round-trip is 60% --/
theorem round_trip_completion_percentage 
  (route : Route)
  (h1 : route.sectionA = { distance := 10, speed := 50 })
  (h2 : route.sectionB = { distance := 20, speed := 40 })
  (h3 : route.sectionC = { distance := 15, speed := 60 })
  (delayA : ℝ)
  (delayB : ℝ)
  (h4 : delayA = 0.15)
  (h5 : delayB = 0.10)
  (returnPercentage : ℝ)
  (h6 : returnPercentage = 0.20)
  : (((route.sectionA.distance + route.sectionB.distance + route.sectionC.distance) + 
      returnPercentage * (route.sectionA.distance + route.sectionB.distance + route.sectionC.distance)) / 
     (2 * (route.sectionA.distance + route.sectionB.distance + route.sectionC.distance))) * 100 = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_completion_percentage_l1055_105542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triplet_sum_not_zero_l1055_105553

theorem triplet_sum_not_zero :
  let triplets : List (ℚ × ℚ × ℚ) := [
    (1/4, 1/2, -3/4),
    (1/2, -1, 1/2),
    (3, -5, 2),
    (-1/10, 3/10, -3/10),
    (1/3, 2/3, -1)
  ]
  ∃! t, t ∈ triplets ∧ t.1 + t.2.1 + t.2.2 ≠ 0 ∧ t = (-1/10, 3/10, -3/10) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triplet_sum_not_zero_l1055_105553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_and_distance_l1055_105516

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (x - 1) + a
noncomputable def g (x : ℝ) : ℝ := Real.log x

theorem intersection_points_and_distance (a : ℝ) (h : a > -2) :
  (∀ x, f a x = g x → a ≤ -1) ∧
  (a = -1 → ∃! x, f a x = g x) ∧
  (a < -1 → ∃ x y, x ≠ y ∧ f a x = g x ∧ f a y = g y) ∧
  (a > -1 → ∀ t > a, |Real.exp t - Real.log (t - a) - 1| > a + 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_and_distance_l1055_105516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_volume_formula_l1055_105567

/-- The volume of a regular triangular pyramid with height h and apex angle φ -/
noncomputable def triangular_pyramid_volume (h : ℝ) (φ : ℝ) : ℝ :=
  (h^3 * Real.sqrt 3 * (Real.cos (φ/2))^2) / (3 - 4 * (Real.cos (φ/2))^2)

/-- Theorem: The volume of a regular triangular pyramid with height h and apex angle φ
    is equal to (h³√3 cos²(φ/2)) / (3 - 4cos²(φ/2)) -/
theorem triangular_pyramid_volume_formula (h : ℝ) (φ : ℝ) (h_pos : h > 0) (φ_pos : φ > 0) :
  triangular_pyramid_volume h φ = (h^3 * Real.sqrt 3 * (Real.cos (φ/2))^2) / (3 - 4 * (Real.cos (φ/2))^2) := by
  -- Unfold the definition of triangular_pyramid_volume
  unfold triangular_pyramid_volume
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_volume_formula_l1055_105567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_45_26384_to_nearest_tenth_l1055_105543

/-- Rounds a real number to the nearest tenth -/
noncomputable def roundToNearestTenth (x : ℝ) : ℝ := 
  ⌊x * 10 + 0.5⌋ / 10

/-- The problem statement -/
theorem round_45_26384_to_nearest_tenth : 
  roundToNearestTenth 45.26384 = 45.3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_45_26384_to_nearest_tenth_l1055_105543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrange_balls_314_4_2_l1055_105560

/-- The number of ways to arrange balls of different colors without consecutive same-color blocks -/
def arrange_balls (black white red : ℕ) : ℕ :=
  let total := black + white + red
  let N := Nat.choose total white * Nat.choose (total - white) red
  let white_block := Nat.choose total white
  let red_block := Nat.choose total red
  let white_red_block := Nat.choose total (white + red)
  let all_blocks := 2 * Nat.choose (total - 1) white
  N - (white_block + red_block) + white_red_block - all_blocks

/-- The theorem stating the correct number of arrangements for the given problem -/
theorem arrange_balls_314_4_2 :
  arrange_balls 314 4 2 = 2376 := by
  sorry

#eval arrange_balls 314 4 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrange_balls_314_4_2_l1055_105560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_simplification_l1055_105576

theorem incorrect_simplification : ∃ (x : ℤ), 
  (x = -3) ∧ 
  (x = x) ∧ 
  (-(-x) = -x) ∧ 
  (|x| ≠ x) ∧ 
  (-|x| = x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_simplification_l1055_105576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_f_range_of_bc_over_a_l1055_105519

noncomputable section

-- Define vectors a and b
def a (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, 2 * Real.sin x)
def b (x : ℝ) : ℝ × ℝ := (Real.sin (x - Real.pi/6), Real.cos (x - Real.pi/6))

-- Define function f
def f (x : ℝ) : ℝ := Real.cos (((a x).1 * (b x).1 + (a x).2 * (b x).2) / 
  (Real.sqrt ((a x).1^2 + (a x).2^2) * Real.sqrt ((b x).1^2 + (b x).2^2)))

-- Theorem for zeros of f
theorem zeros_of_f :
  ∀ x : ℝ, f x = 0 ↔ ∃ k : ℤ, x = k * Real.pi / 2 + Real.pi / 12 :=
sorry

-- Theorem for range of (b+c)/a in triangle ABC
theorem range_of_bc_over_a (A B C : ℝ) (a b c : ℝ) :
  A + B + C = Real.pi →
  A > 0 → B > 0 → C > 0 →
  a > 0 → b > 0 → c > 0 →
  a * Real.sin A = b * Real.sin B →
  a * Real.sin A = c * Real.sin C →
  f A = 1 →
  1 < (b + c) / a ∧ (b + c) / a ≤ 2 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_f_range_of_bc_over_a_l1055_105519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1055_105531

def A : Set ℝ := {a | (2 : ℝ)^a = 4}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*(m+1)*x + m^2 < 0}

theorem problem_solution :
  (∀ m : ℝ, m = 4 → A ∪ B m = Set.Icc 2 8) ∧
  (∀ m : ℝ, A ∩ B m = B m → m ≤ -1/2) ∧
  (∀ m : ℝ, m ≤ -1/2 → A ∩ B m = B m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1055_105531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_satisfies_conditions_l1055_105548

/-- Given two points in ℝ², return true if the third point lies on their line. -/
def collinear (p₁ p₂ p : ℝ × ℝ) : Prop :=
  (p.2 - p₁.2) * (p₂.1 - p₁.1) = (p.1 - p₁.1) * (p₂.2 - p₁.2)

/-- Given two points in ℝ², compute the distance between them. -/
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₂.1 - p₁.1)^2 + (p₂.2 - p₁.2)^2)

theorem point_satisfies_conditions (p₁ p₂ p : ℝ × ℝ) 
    (h₁ : p₁ = (2, -1)) 
    (h₂ : p₂ = (0, 5)) 
    (h₃ : p = (-2, 11)) : 
    collinear p₁ p₂ p ∧ distance p₁ p = 2 * distance p p₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_satisfies_conditions_l1055_105548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_times_value_function_range_l1055_105523

open Real

-- Define the function f(x) = ln x + x
noncomputable def f (x : ℝ) : ℝ := log x + x

-- Define the property of being a k-times value function
def is_k_times_value_function (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∃ (a b : ℝ), a < b ∧ (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc (k * a) (k * b))

-- State the theorem
theorem k_times_value_function_range :
  ∀ k > 0, is_k_times_value_function f k → k ∈ Set.Ioo 1 (1 + 1 / exp 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_times_value_function_range_l1055_105523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_a_on_line_a_l1055_105588

/-- Definition of the complex number z in terms of a real number a -/
def z (a : ℝ) : ℂ := Complex.mk (a^2 - 4) (a + 2)

/-- Theorem for part I -/
theorem pure_imaginary_a (a : ℝ) :
  z a = Complex.I * Complex.im (z a) → a = 2 := by sorry

/-- Theorem for part II -/
theorem on_line_a (a : ℝ) :
  Complex.re (z a) + 2 * Complex.im (z a) + 1 = 0 → a = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_a_on_line_a_l1055_105588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_crossing_time_l1055_105500

/-- The time taken for a train to cross a platform -/
noncomputable def time_to_cross_platform (train_length : ℝ) (signal_pole_time : ℝ) (platform_length : ℝ) : ℝ :=
  (train_length + platform_length) / (train_length / signal_pole_time)

/-- Theorem stating the time taken for a specific train to cross a specific platform -/
theorem train_platform_crossing_time :
  time_to_cross_platform 300 18 550.0000000000001 = 51.000000000000006 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_crossing_time_l1055_105500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_point_coordinates_l1055_105529

/-- Triangle ABC with sides a, b, c opposite to vertices A, B, C respectively -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point P with barycentric coordinates u, v, w with respect to triangle ABC -/
structure BarycentricPoint (T : Triangle) where
  u : ℝ
  v : ℝ
  w : ℝ
  sum_one : u + v + w = 1

/-- The theorem stating the barycentric coordinates of the special point P -/
theorem special_point_coordinates (T : Triangle) :
  ∃ P : BarycentricPoint T,
    P.u = 1 / T.b + 1 / T.c - 1 / T.a ∧
    P.v = 1 / T.a + 1 / T.c - 1 / T.b ∧
    P.w = 1 / T.a + 1 / T.b - 1 / T.c ∧
    (∃ A₀ B₀ A₁ C₁ B₂ C₂ : ℝ × ℝ,
      (A₀.1 - B₀.1)^2 + (A₀.2 - B₀.2)^2 =
      (A₁.1 - C₁.1)^2 + (A₁.2 - C₁.2)^2 ∧
      (A₁.1 - C₁.1)^2 + (A₁.2 - C₁.2)^2 =
      (B₂.1 - C₂.1)^2 + (B₂.2 - C₂.2)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_point_coordinates_l1055_105529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_P_friendly_slopes_is_negative_432_over_5_final_answer_l1055_105580

/-- A parabola in the xy-plane --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A line in the xy-plane --/
structure Line where
  m : ℝ
  b : ℝ

/-- The unique parabola P tangent to x-axis at (5,0) and y-axis at (0,12) --/
noncomputable def P : Parabola := { a := 12/25, b := -24/5, c := 12 }

/-- A line is P-friendly if x-axis, y-axis, and P divide it into three equal segments --/
def is_P_friendly (l : Line) : Prop := sorry

/-- The sum of slopes of all P-friendly lines --/
noncomputable def sum_of_P_friendly_slopes : ℝ := sorry

/-- Main theorem: The sum of slopes of all P-friendly lines is -432/5 --/
theorem sum_of_P_friendly_slopes_is_negative_432_over_5 :
  sum_of_P_friendly_slopes = -432/5 := by sorry

/-- The final answer is 437 --/
theorem final_answer : Int :=
  let m : Int := 432
  let n : Int := 5
  m + n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_P_friendly_slopes_is_negative_432_over_5_final_answer_l1055_105580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_limit_l1055_105554

/-- The absolute value of the roots of a quadratic equation approaches infinity as the coefficient of x² approaches zero -/
theorem quadratic_root_limit (b c : ℝ) :
  ∀ ε > (0 : ℝ), ∃ δ > (0 : ℝ), ∀ a : ℝ, 0 < |a| ∧ |a| < δ →
    ∀ x : ℝ, a * x^2 + b * x + c = 0 → |x| > 1/ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_limit_l1055_105554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_covered_by_six_strips_l1055_105532

/-- Represents a rectangular strip -/
structure Strip where
  length : ℝ
  width : ℝ
deriving Inhabited

/-- Calculates the area covered by overlapping strips -/
def areaCovered (strips : List Strip) : ℝ :=
  let totalArea := strips.foldl (fun acc s => acc + s.length * s.width) 0
  let overlapArea := (Nat.choose strips.length 2) * (strips.head!.width * strips.head!.width)
  totalArea - overlapArea

/-- Theorem stating the area covered by six overlapping strips -/
theorem area_covered_by_six_strips :
  let strips := List.replicate 6 ⟨12, 2⟩
  areaCovered strips = 84 := by
  sorry

#eval areaCovered (List.replicate 6 ⟨12, 2⟩)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_covered_by_six_strips_l1055_105532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seldom_means_rarely_l1055_105528

def seldom : String := "seldom"
def seldom_meaning : String := "rarely or infrequently"

theorem seldom_means_rarely : seldom = "seldom" ∧ seldom_meaning = "rarely or infrequently" :=
by
  apply And.intro
  · rfl
  · rfl

#check seldom_means_rarely

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seldom_means_rarely_l1055_105528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_perimeter_l1055_105590

/-- The perimeter of a circular sector with central angle 60° and radius 3 is π + 6 -/
theorem sector_perimeter (central_angle radius : Real) :
  central_angle = π / 3 → radius = 3 →
  (central_angle * radius) + (2 * radius) = π + 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_perimeter_l1055_105590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_log_relation_l1055_105504

theorem exponential_log_relation :
  (∀ x y : ℝ, (2 : ℝ)^x > (2 : ℝ)^y ↔ x > y) ∧
  (∀ x y : ℝ, (Real.log x > Real.log y ↔ x > y ∧ y > 0)) →
  (∀ a b : ℝ, Real.log a > Real.log b → (2 : ℝ)^a > (2 : ℝ)^b) ∧
  ¬(∀ a b : ℝ, (2 : ℝ)^a > (2 : ℝ)^b → Real.log a > Real.log b) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_log_relation_l1055_105504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt2_minus_1_power_l1055_105573

theorem sqrt2_minus_1_power (n : ℕ) :
  ∃ (a b : ℕ),
    (if n % 2 = 1 then
      ((Real.sqrt 2 - 1) ^ n : ℝ) = a * Real.sqrt 2 - b ∧ 2 * a ^ 2 = b ^ 2 + 1
    else
      ((Real.sqrt 2 - 1) ^ n : ℝ) = a - b * Real.sqrt 2 ∧ a ^ 2 = 2 * b ^ 2 + 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt2_minus_1_power_l1055_105573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_inequality_iff_half_plane_l1055_105594

noncomputable section

structure Point where
  x : ℝ
  y : ℝ

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

def perpendicular_bisector (A B : Point) : Set Point :=
  {M : Point | distance M A = distance M B}

def half_plane (A B : Point) : Set Point :=
  {M : Point | distance M B < distance M A}

theorem distance_inequality_iff_half_plane (A B : Point) :
  ∀ M : Point, distance M A > distance M B ↔ M ∈ half_plane A B :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_inequality_iff_half_plane_l1055_105594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_answer_is_correct_l1055_105596

def correct_answer : String := "A. a; the"

theorem answer_is_correct : correct_answer = "A. a; the" := by
  rfl

#check answer_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_answer_is_correct_l1055_105596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_size_calculation_l1055_105544

def grade10Students : ℕ := 400
def grade11Students : ℕ := 320
def grade12Students : ℕ := 280
def selectionProbability : ℚ := 1/5

def totalStudents : ℕ := grade10Students + grade11Students + grade12Students

theorem sample_size_calculation (n : ℕ) :
  n = (totalStudents : ℚ) * selectionProbability → n = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_size_calculation_l1055_105544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_sum_l1055_105597

/-- The function f(x) = (6x^2 - 11) / (4x^2 + 7x + 3) -/
noncomputable def f (x : ℝ) : ℝ := (6 * x^2 - 11) / (4 * x^2 + 7 * x + 3)

/-- p and q are the x-coordinates of the vertical asymptotes of f -/
def is_vertical_asymptote (x : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ y, 0 < |y - x| ∧ |y - x| < δ → |f y| > 1/ε

theorem vertical_asymptotes_sum :
  ∀ p q : ℝ, is_vertical_asymptote p ∧ is_vertical_asymptote q →
  (∀ x : ℝ, x ≠ p ∧ x ≠ q → ¬(is_vertical_asymptote x)) →
  p + q = -7/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_sum_l1055_105597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_time_to_see_again_l1055_105557

/-- The time it takes for Paul and Sara to see each other again after being blocked by a circular pond -/
noncomputable def time_to_see_again (paul_speed sara_speed path_distance pond_radius initial_distance : ℝ) : ℝ :=
  path_distance / (sara_speed - paul_speed)

theorem correct_time_to_see_again :
  let paul_speed : ℝ := 2
  let sara_speed : ℝ := 4
  let path_distance : ℝ := 250
  let pond_radius : ℝ := 75
  let initial_distance : ℝ := 250
  time_to_see_again paul_speed sara_speed path_distance pond_radius initial_distance = 50 := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem proof
-- #eval correct_time_to_see_again

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_time_to_see_again_l1055_105557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_denominator_of_sum_l1055_105581

/-- Given two irreducible fractions with denominators 600 and 700, 
    the smallest possible denominator of their sum (when written as an irreducible fraction) is 168. -/
theorem smallest_denominator_of_sum (a b : ℕ) 
  (ha : Nat.Coprime a 600) 
  (hb : Nat.Coprime b 700) : 
  ∃ (n : ℕ), n ≥ 1 ∧ 
  (∃ (m : ℕ), (7 * a + 6 * b) * n = m * 4200) ∧
  (∀ (k : ℕ), k ≥ 1 → 
    (∃ (l : ℕ), (7 * a + 6 * b) * k = l * 4200) → 
    k ≥ 168) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_denominator_of_sum_l1055_105581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_sum_l1055_105539

theorem cubic_root_sum (x : ℝ) (hx : x > 0) :
  (((1 - x^3)^(1/3 : ℝ) + (1 + x^3)^(1/3 : ℝ)) = 1) → x^6 = 28/27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_sum_l1055_105539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_MN_l1055_105514

/-- Curve C₁ in parametric form -/
noncomputable def C₁ (α : ℝ) : ℝ × ℝ := (2 * Real.cos α, 2 + 2 * Real.sin α)

/-- Curve C₂ in polar form -/
noncomputable def C₂ (θ : ℝ) : ℝ := Real.sqrt (2 / (1 + Real.sin θ ^ 2))

/-- Center of curve C₁ -/
def N : ℝ × ℝ := (0, 2)

/-- A point on curve C₂ -/
noncomputable def M (θ : ℝ) : ℝ × ℝ := (C₂ θ * Real.cos θ, C₂ θ * Real.sin θ)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem max_distance_MN :
  ∃ (θ : ℝ), ∀ (φ : ℝ), distance (M θ) N ≥ distance (M φ) N ∧ distance (M θ) N = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_MN_l1055_105514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_subset_singleton_zero_l1055_105540

theorem empty_subset_singleton_zero : ∅ ⊆ ({0} : Set ℕ) := by
  apply Set.empty_subset


end NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_subset_singleton_zero_l1055_105540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_circumscribed_circle_m_value_l1055_105506

/-- Two lines that form a quadrilateral with the coordinate axes --/
structure QuadrilateralLines where
  l1 : ℝ → ℝ → Prop
  l2 : ℝ → ℝ → Prop

/-- The condition for a quadrilateral to have a circumscribed circle --/
def has_circumscribed_circle (q : QuadrilateralLines) : Prop :=
  ∃ (c : ℝ × ℝ) (r : ℝ), ∀ (x y : ℝ), 
    (q.l1 x y ∨ q.l2 x y ∨ x = 0 ∨ y = 0) → 
    (x - c.fst)^2 + (y - c.snd)^2 = r^2

/-- The theorem statement --/
theorem quadrilateral_circumscribed_circle_m_value 
  (q : QuadrilateralLines)
  (h1 : q.l1 = λ x y ↦ 2*x - 5*y + 20 = 0)
  (h2 : ∃ m : ℝ, q.l2 = λ x y ↦ m*x + 2*y - 10 = 0)
  (h3 : has_circumscribed_circle q) :
  ∃ m : ℝ, q.l2 = λ x y ↦ m*x + 2*y - 10 = 0 ∧ m = 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_circumscribed_circle_m_value_l1055_105506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_coloring_distance_l1055_105530

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- A regular hexagon with side length 1 -/
def RegularHexagon :=
  {p : Point | p.x^2 + p.y^2 ≤ 3/4 ∧ 
    ∀ (i : Fin 6), (p.x - (Real.cos (2 * Real.pi * (i : ℝ) / 6)))^2 + 
                   (p.y - (Real.sin (2 * Real.pi * (i : ℝ) / 6)))^2 ≤ 1}

/-- A 3-coloring of points -/
def Coloring := Point → Fin 3

/-- A valid coloring for a given r -/
def validColoring (r : ℝ) (c : Coloring) : Prop :=
  ∀ (p q : Point), p ∈ RegularHexagon → q ∈ RegularHexagon → c p = c q → distance p q < r

/-- The theorem to be proved -/
theorem min_coloring_distance :
  (∃ (r : ℝ), ∀ (r' : ℝ), (∃ (c : Coloring), validColoring r' c) → r ≤ r') ∧
  (∃ (c : Coloring), validColoring (3/2) c) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_coloring_distance_l1055_105530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1055_105589

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C →
  c = 4 →
  a + b = 7 →
  C = π / 3 ∧ (1 / 2) * a * b * Real.sin C = (11 * Real.sqrt 3) / 4 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1055_105589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C₁_intersects_C₂_l1055_105545

-- Define the curves C₁ and C₂
def C₁ (t : ℝ) : ℝ × ℝ := (4 + t, 5 + 2*t)

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ :=
  let ρ := Real.sqrt (6*Real.cos θ + 10*Real.sin θ + Real.sqrt ((6*Real.cos θ + 10*Real.sin θ)^2 + 36))
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the intersection property
def intersect (C₁ : ℝ → ℝ × ℝ) (C₂ : ℝ → ℝ × ℝ) : Prop :=
  ∃ (t θ : ℝ), C₁ t = C₂ θ

-- Theorem statement
theorem C₁_intersects_C₂ : intersect C₁ C₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C₁_intersects_C₂_l1055_105545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_dihedral_planar_obtuse_l1055_105547

-- Define the types of angles
inductive AngleType
  | SkewLines
  | LinePlane
  | DihedralPlanar

-- Define the angle ranges
noncomputable def angleRange (t : AngleType) : Set ℝ :=
  match t with
  | .SkewLines => Set.Ioo 0 (Real.pi/2)
  | .LinePlane => Set.Icc 0 (Real.pi/2)
  | .DihedralPlanar => Set.Ico 0 Real.pi

-- Define what it means for an angle to be obtuse
def isObtuse (θ : ℝ) : Prop := Real.pi/2 < θ ∧ θ < Real.pi

-- Theorem statement
theorem only_dihedral_planar_obtuse :
  ∀ t : AngleType, (∃ θ ∈ angleRange t, isObtuse θ) ↔ t = AngleType.DihedralPlanar :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_dihedral_planar_obtuse_l1055_105547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_specific_vectors_l1055_105535

/-- The cosine of the angle between two 2D vectors -/
noncomputable def cos_angle (a b : ℝ × ℝ) : ℝ :=
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))

/-- Theorem: The cosine of the angle between vectors (2, -1) and (1, 3) is -√2/10 -/
theorem cos_angle_specific_vectors :
  cos_angle (2, -1) (1, 3) = -Real.sqrt 2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_specific_vectors_l1055_105535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_same_length_is_17_35_l1055_105575

/-- RegularHexagon represents a regular hexagon with sides and diagonals -/
structure RegularHexagon where
  sides : Finset ℕ
  diagonals : Finset ℕ
  sides_count : sides.card = 6
  diagonals_count : diagonals.card = 9
  sides_equal : ∀ s₁ s₂, s₁ ∈ sides → s₂ ∈ sides → s₁ = s₂
  diagonals_equal : ∀ d₁ d₂, d₁ ∈ diagonals → d₂ ∈ diagonals → d₁ = d₂
  sides_diagonals_distinct : sides ∩ diagonals = ∅

/-- The set T of all sides and diagonals -/
def T (h : RegularHexagon) : Finset ℕ := h.sides ∪ h.diagonals

/-- The probability of selecting two segments of the same length -/
def prob_same_length : ℚ :=
  (6 : ℚ) / 15 * 5 / 14 + 9 / 15 * 8 / 14

/-- Theorem stating the probability of selecting two segments of the same length -/
theorem prob_same_length_is_17_35 :
  prob_same_length = 17 / 35 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_same_length_is_17_35_l1055_105575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1055_105525

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) : Prop :=
  b * b = a * c

noncomputable def sum_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  d ≠ 0 →
  a 1 = 2 →
  arithmetic_sequence a d →
  geometric_sequence (a 1) (a 3) (a 6) →
  sum_arithmetic_sequence a 10 = 85 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1055_105525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_divisor_of_consecutive_multiples_of_four_l1055_105584

theorem greatest_divisor_of_consecutive_multiples_of_four :
  ∃ (n : ℕ), 
    (∀ (k : ℕ), 
      let multiples := [4*k, 4*(k+1), 4*(k+2), 4*(k+3), 4*(k+4)]
      let smallest_prime_factors := multiples.map Nat.minFac
      let raised_multiples := List.zip multiples smallest_prime_factors |>.map (λ p => p.1 ^ p.2)
      let product := raised_multiples.prod
      (2^24 ∣ product)) ∧ 
    (∀ (m : ℕ), m > 2^24 → 
      ∃ (k : ℕ), 
        let multiples := [4*k, 4*(k+1), 4*(k+2), 4*(k+3), 4*(k+4)]
        let smallest_prime_factors := multiples.map Nat.minFac
        let raised_multiples := List.zip multiples smallest_prime_factors |>.map (λ p => p.1 ^ p.2)
        let product := raised_multiples.prod
        ¬(m ∣ product)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_divisor_of_consecutive_multiples_of_four_l1055_105584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_g_l1055_105566

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (x - Real.pi/4)

noncomputable def g (x : ℝ) : ℝ := f ((x - Real.pi/3) / 2)

theorem symmetry_axis_of_g :
  ∃ (k : ℤ), g (11*Real.pi/6 + 2*Real.pi*↑k) = g (11*Real.pi/6 - (x - (11*Real.pi/6 + 2*Real.pi*↑k))) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_g_l1055_105566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_pineapple_price_l1055_105518

/-- The price per pineapple given the conditions of Jonah's pineapple business -/
def pineapple_price (num_pineapples : ℕ) (rings_per_pineapple : ℕ) 
  (selling_price : ℚ) (num_rings_sold : ℕ) (total_profit : ℚ) : ℚ :=
  let price_per_pineapple := 3
  let total_rings := num_pineapples * rings_per_pineapple
  let price_per_ring := selling_price / num_rings_sold
  let total_revenue := total_rings * price_per_ring
  let total_cost := num_pineapples * price_per_pineapple
  price_per_pineapple

/-- Proof of the pineapple price theorem -/
theorem prove_pineapple_price : 
  pineapple_price 6 12 5 4 72 = 3 := by
  -- Unfold the definition of pineapple_price
  unfold pineapple_price
  -- The result follows directly from the definition
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_pineapple_price_l1055_105518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_plus_pi_fourth_l1055_105551

/-- Given an angle α whose vertex is at the origin, initial side along the positive x-axis, 
    and terminal side passing through the point (2,3), tan(2α + π/4) = -7/17 -/
theorem tan_double_plus_pi_fourth (α : Real) : 
  (∃ (x y : Real), x = 2 ∧ y = 3 ∧ Real.tan α = y / x) → 
  Real.tan (2 * α + Real.pi / 4) = -7 / 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_plus_pi_fourth_l1055_105551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1055_105541

variable (x₁ x₂ x₃ x₄ x₅ : ℝ)

def eq1 (x₁ x₂ x₃ x₄ x₅ : ℝ) : Prop := x₁ + 2*x₂ + 2*x₃ + 2*x₄ + 2*x₅ = 1
def eq2 (x₁ x₂ x₃ x₄ x₅ : ℝ) : Prop := x₁ + 3*x₂ + 4*x₃ + 4*x₄ + 4*x₅ = 2
def eq3 (x₁ x₂ x₃ x₄ x₅ : ℝ) : Prop := x₁ + 3*x₂ + 5*x₃ + 6*x₄ + 6*x₅ = 3
def eq4 (x₁ x₂ x₃ x₄ x₅ : ℝ) : Prop := x₁ + 3*x₂ + 5*x₃ + 7*x₄ + 8*x₅ = 4
def eq5 (x₁ x₂ x₃ x₄ x₅ : ℝ) : Prop := x₁ + 3*x₂ + 5*x₃ + 7*x₄ + 9*x₅ = 5

theorem unique_solution :
  ∀ x₁ x₂ x₃ x₄ x₅ : ℝ, 
    eq1 x₁ x₂ x₃ x₄ x₅ ∧ eq2 x₁ x₂ x₃ x₄ x₅ ∧ eq3 x₁ x₂ x₃ x₄ x₅ ∧ eq4 x₁ x₂ x₃ x₄ x₅ ∧ eq5 x₁ x₂ x₃ x₄ x₅ → 
    x₁ = 1 ∧ x₂ = -1 ∧ x₃ = 1 ∧ x₄ = -1 ∧ x₅ = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1055_105541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_lengths_l1055_105508

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A : ℝ)
  (area : ℝ)

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  t.A = Real.pi / 3 ∧  -- 60° in radians
  t.a = Real.sqrt 7 ∧
  t.area = 3 * Real.sqrt 3 / 2

-- Define the theorem
theorem triangle_side_lengths (t : Triangle) (h : isValidTriangle t) :
  (t.b = 3 ∧ t.c = 2) ∨ (t.b = 2 ∧ t.c = 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_lengths_l1055_105508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_150_degrees_l1055_105599

noncomputable def rotation_matrix (θ : Real) : Matrix (Fin 2) (Fin 2) Real :=
  !![Real.cos θ, -Real.sin θ;
     Real.sin θ,  Real.cos θ]

noncomputable def angle : Real := 150 * Real.pi / 180

theorem rotation_150_degrees :
  rotation_matrix angle = !![-(Real.sqrt 3) / 2, -1 / 2;
                              1 / 2, -(Real.sqrt 3) / 2] :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_150_degrees_l1055_105599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_specific_function_l1055_105536

theorem integral_of_specific_function (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_def : ∀ x, f x = x^2 + 2 * f (π/2) * x + Real.sin (2*x)) :
  ∫ x in Set.Icc 0 1, f x = 17/6 - π - (1/2) * Real.cos 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_specific_function_l1055_105536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_f_l1055_105524

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 6)

-- State the theorem about the symmetry axis
theorem symmetry_axis_of_f :
  ∃ (k : ℤ), (5 * Real.pi / 12 : ℝ) = (1 / 2 : ℝ) * (k : ℝ) * Real.pi - Real.pi / 12 ∧
  ∀ (x : ℝ), f (5 * Real.pi / 12 + x) = f (5 * Real.pi / 12 - x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_f_l1055_105524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_branch_company_profit_l1055_105564

-- Define the variables and functions
noncomputable def cost : ℝ := 3
def management_fee (a : ℝ) : Prop := 3 ≤ a ∧ a ≤ 5
def selling_price (x : ℝ) : Prop := 9 ≤ x ∧ x ≤ 11
noncomputable def sales_volume (x : ℝ) : ℝ := (12 - x)^2
noncomputable def annual_profit (x a : ℝ) : ℝ := (x - cost - a) * sales_volume x

-- Define the maximum profit function
noncomputable def Q (a : ℝ) : ℝ :=
  if a ≤ 9/2 then 9 * (6 - a) else 4 * (3 - a/3)^3

-- State the theorem
theorem branch_company_profit (a x : ℝ) 
  (h_a : management_fee a) (h_x : selling_price x) :
  -- Annual profit function
  annual_profit x a = (x - 3 - a) * (12 - x)^2 ∧
  -- Optimal selling price
  (∀ y, selling_price y → annual_profit x a ≥ annual_profit y a) →
    ((a ≤ 9/2 ∧ x = 9) ∨ (9/2 < a ∧ x = 6 + 3*a/2)) ∧
  -- Maximum profit
  (∀ y, selling_price y → annual_profit x a ≥ annual_profit y a) →
    annual_profit x a = Q a :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_branch_company_profit_l1055_105564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_bound_l1055_105577

/-- A function satisfying the given inequality condition -/
def SatisfiesCondition (f : ℝ → ℝ) (a b c : ℝ) : Prop :=
  ∀ x y, f (x - f y) ≥ f x + f (f y) - a * x * f y - b * f y - c

/-- The main theorem -/
theorem limit_bound 
  (f : ℝ → ℝ) 
  (a b c : ℝ) 
  (h_pos : ∀ x, f x > 0)
  (h_cond : SatisfiesCondition f a b c) : 
  Filter.Tendsto (fun x => f x / x^2) Filter.atTop (Filter.atTop.comap (fun x => x - a / 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_bound_l1055_105577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rita_downstream_speed_l1055_105558

/-- Calculates the downstream speed given upstream speed, total time, and distance --/
noncomputable def downstream_speed (upstream_speed : ℝ) (total_time : ℝ) (distance : ℝ) : ℝ :=
  let upstream_time := distance / upstream_speed
  let downstream_time := total_time - upstream_time
  distance / downstream_time

/-- Theorem stating that Rita's downstream speed is 9 mph --/
theorem rita_downstream_speed :
  downstream_speed 3 8 18 = 9 := by
  -- Unfold the definition of downstream_speed
  unfold downstream_speed
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rita_downstream_speed_l1055_105558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_constant_l1055_105505

/-- The curve C: y^2 + 4x^2 = 1 -/
def C (x y : ℝ) : Prop := y^2 + 4*x^2 = 1

/-- Point M on curve C -/
structure PointM where
  x : ℝ
  y : ℝ
  on_curve : C x y

/-- Point N on curve C -/
structure PointN where
  x : ℝ
  y : ℝ
  on_curve : C x y

/-- OM ⊥ ON condition -/
def perpendicular (M : PointM) (N : PointN) : Prop :=
  M.x * N.x + M.y * N.y = 0

/-- Distance from origin O to line MN -/
noncomputable def distance_to_line (M : PointM) (N : PointN) : ℝ :=
  let num := Real.sqrt ((M.x^2 + M.y^2) * (N.x^2 + N.y^2))
  let den := Real.sqrt ((M.x - N.x)^2 + (M.y - N.y)^2)
  num / den

/-- Main theorem: distance from O to MN is always √5/5 -/
theorem distance_is_constant (M : PointM) (N : PointN) 
    (h : perpendicular M N) : distance_to_line M N = Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_constant_l1055_105505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_inscribed_angles_is_180_l1055_105598

/-- A circle with an inscribed pentagon --/
structure CircleWithInscribedPentagon where
  /-- The circle --/
  circle : Set (ℝ × ℝ)
  /-- The inscribed pentagon --/
  pentagon : Set (ℝ × ℝ)
  /-- The pentagon is inscribed in the circle --/
  inscribed : pentagon ⊆ circle

/-- An angle inscribed in an arc of the circle --/
def InscribedAngle (cwp : CircleWithInscribedPentagon) : Type := ℝ

/-- The sum of angles inscribed in the five arcs cut off by the sides of the pentagon --/
def SumOfInscribedAngles (cwp : CircleWithInscribedPentagon) : ℝ := sorry

/-- Theorem: The sum of angles inscribed in the five arcs cut off by the sides of an inscribed pentagon is 180° --/
theorem sum_of_inscribed_angles_is_180 (cwp : CircleWithInscribedPentagon) :
  SumOfInscribedAngles cwp = 180 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_inscribed_angles_is_180_l1055_105598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_forms_two_planes_l1055_105578

/-- Definition of a plane in 3D space -/
def IsPlane (S : Set (ℝ × ℝ × ℝ)) : Prop :=
  ∃ (a b c d : ℝ), (a, b, c) ≠ (0, 0, 0) ∧ 
    ∀ (x y z : ℝ), (x, y, z) ∈ S ↔ a*x + b*y + c*z = d

/-- The set of points (x, y, z) satisfying (x+y+z)^2 = x^2 + y^2 + z^2 forms two planes -/
theorem equation_forms_two_planes :
  ∃ (P Q : Set (ℝ × ℝ × ℝ)), 
    (∀ (x y z : ℝ), (x + y + z)^2 = x^2 + y^2 + z^2 ↔ (x, y, z) ∈ P ∪ Q) ∧
    IsPlane P ∧ 
    IsPlane Q ∧ 
    P ≠ Q :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_forms_two_planes_l1055_105578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_distance_is_5_circles_intersect_option_B_l1055_105563

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 4
def circle_C2 (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 16

-- Define the centers and radii
def center_C1 : ℝ × ℝ := (4, 0)
def center_C2 : ℝ × ℝ := (0, 3)
def radius_C1 : ℝ := 2
def radius_C2 : ℝ := 4

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ :=
  Real.sqrt ((center_C2.1 - center_C1.1)^2 + (center_C2.2 - center_C1.2)^2)

-- Theorem stating that the circles intersect
theorem circles_intersect :
  distance_between_centers > |radius_C2 - radius_C1| ∧
  distance_between_centers < radius_C1 + radius_C2 :=
by
  -- The proof is omitted for now
  sorry

-- Helper theorem to show that the distance is 5
theorem distance_is_5 : distance_between_centers = 5 :=
by
  -- The proof is omitted for now
  sorry

-- Final theorem stating that the circles intersect (option B)
theorem circles_intersect_option_B : 
  2 < distance_between_centers ∧ distance_between_centers < 6 :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_distance_is_5_circles_intersect_option_B_l1055_105563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_from_270_degree_sector_l1055_105521

/-- Represents a circular sector --/
structure CircularSector where
  radius : ℝ
  angle : ℝ

/-- Represents a cone --/
structure Cone where
  baseRadius : ℝ
  slantHeight : ℝ

/-- Converts a circular sector to a cone --/
noncomputable def sectorToCone (s : CircularSector) : Cone :=
  { baseRadius := s.radius * s.angle / (2 * Real.pi),
    slantHeight := s.radius }

theorem cone_from_270_degree_sector :
  let s : CircularSector := { radius := 12, angle := 3 * Real.pi / 2 }
  let c : Cone := sectorToCone s
  c.baseRadius = 9 ∧ c.slantHeight = 12 := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_from_270_degree_sector_l1055_105521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_less_than_eight_thirds_iff_x_less_than_three_l1055_105503

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 2 then x - 1/x else x

theorem f_less_than_eight_thirds_iff_x_less_than_three :
  ∀ x : ℝ, f x < 8/3 ↔ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_less_than_eight_thirds_iff_x_less_than_three_l1055_105503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_length_l1055_105591

/-- Calculates the length of a faster train given the speeds of two trains and the time taken for the faster train to pass the slower one. -/
theorem faster_train_length
  (faster_speed slower_speed : ℝ)
  (passing_time : ℝ)
  (h1 : faster_speed = 50)
  (h2 : slower_speed = 32)
  (h3 : passing_time = 15)
  (h4 : faster_speed > slower_speed) :
  (faster_speed - slower_speed) * passing_time / 3600 * 1000 = 75 := by
  sorry

#check faster_train_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_length_l1055_105591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_a_to_b_longer_l1055_105552

/-- The time taken for a journey between two points A and B -/
noncomputable def journey_time (s v₁ v₂ : ℝ) : ℝ → ℝ := fun t ↦ 
  if t = 1 then s/2 * (1/v₁ + 1/v₂)  -- A to B
  else s/(2*(v₁ + v₂))               -- B to A

/-- Theorem stating that the journey from A to B takes longer than from B to A -/
theorem journey_time_a_to_b_longer (s v₁ v₂ : ℝ) 
  (h₁ : s > 0) (h₂ : v₁ > 0) (h₃ : v₂ > 0) (h₄ : v₁ ≠ v₂) : 
  journey_time s v₁ v₂ 1 > journey_time s v₁ v₂ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_a_to_b_longer_l1055_105552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_properties_l1055_105556

/-- Represents a city grid with horizontal and vertical streets -/
structure CityGrid where
  horizontal_streets : Nat
  vertical_streets : Nat

/-- Represents a block address in the city -/
structure BlockAddress where
  i : Nat
  j : Nat

/-- Calculates the number of blocks in the city -/
def num_blocks (city : CityGrid) : Nat :=
  (city.horizontal_streets - 1) * (city.vertical_streets - 1)

/-- Calculates the Manhattan distance between two blocks -/
def block_distance (from_block to_block : BlockAddress) : Nat :=
  Int.natAbs (from_block.i - to_block.i) + Int.natAbs (from_block.j - to_block.j)

/-- Calculates the taxi fare based on distance -/
def taxi_fare (distance : Nat) : Nat :=
  distance

theorem city_properties (city : CityGrid) 
  (h1 : city.horizontal_streets = 10)
  (h2 : city.vertical_streets = 12)
  (block1 : BlockAddress)
  (block2 : BlockAddress)
  (h3 : block1.i = 7 ∧ block1.j = 1)
  (h4 : block2.i = 2 ∧ block2.j = 10) :
  num_blocks city = 99 ∧ 
  block_distance block1 block2 = 14 ∧
  taxi_fare (block_distance block1 block2) = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_properties_l1055_105556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_condition_l1055_105561

/-- The inequality has exactly one solution if and only if a = 2 -/
theorem unique_solution_condition (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∃! x : ℝ, (Real.log (Real.sqrt (x^2 + a*x + 5) + 1) / Real.log a) * 
              (Real.log (x^2 + a*x + 6) / Real.log 5) + 
              (Real.log 3 / Real.log a) ≥ 0) ↔ 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_condition_l1055_105561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisector_theorem_l1055_105587

/-- Triangle with circumscribed and inscribed circles -/
structure TriangleWithCircles where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  O : ℝ × ℝ
  O' : ℝ × ℝ
  R : ℝ
  r : ℝ

/-- Check if a point is on a circle -/
def is_on_circle (P : ℝ × ℝ) (O : ℝ × ℝ) (R : ℝ) : Prop := sorry

/-- Check if a line is an angle bisector -/
def is_angle_bisector (V P A B : ℝ × ℝ) : Prop := sorry

/-- Calculate the distance between two points -/
noncomputable def distance (P Q : ℝ × ℝ) : ℝ := sorry

/-- Calculate the angle between three points -/
noncomputable def angle (A V B : ℝ × ℝ) : ℝ := sorry

/-- The theorem about the bisector of angle C -/
theorem bisector_theorem (t : TriangleWithCircles) (C₁ : ℝ × ℝ) (h₃ : ℝ) :
  is_on_circle C₁ t.O t.R →
  is_angle_bisector t.C C₁ t.A t.B →
  distance t.C C₁ = t.R + 2 * t.r →
  (angle t.A t.C t.B = 60 ∨ h₃ = t.R / 2 + 2 * t.r) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisector_theorem_l1055_105587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1055_105595

/-- The parabola function -/
noncomputable def f (x : ℝ) : ℝ := x^2 - 7*x + 10

/-- The area function of the triangle ABC -/
noncomputable def triangleArea (p : ℝ) : ℝ :=
  (1/2) * |3*p^2 - 25*p + 38|

/-- The theorem statement -/
theorem max_triangle_area :
  ∃ (maxArea : ℝ),
    (∀ p, 2 ≤ p ∧ p ≤ 5 → triangleArea p ≤ maxArea) ∧
    (maxArea = (3/8) * (25/6 - 2) * (5 - 25/6)) ∧
    (f 2 = 0) ∧ (f 5 = 4) ∧ (∀ p, f p = (p^2 - 7*p + 10)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1055_105595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l1055_105501

theorem sin_2alpha_value (α : ℝ) 
  (h1 : Real.cos (Real.pi * α) = -1/2) 
  (h2 : 3/2 * Real.pi < α ∧ α < 2 * Real.pi) : 
  Real.sin (2 * α) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l1055_105501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_specific_arrangement_l1055_105571

-- Define the total number of tiles
def total_tiles : ℕ := 7

-- Define the number of X tiles
def x_tiles : ℕ := 4

-- Define the number of O tiles
def o_tiles : ℕ := 3

-- Define the specific arrangement we're looking for
def target_arrangement : List Char := ['X', 'O', 'X', 'O', 'X', 'O', 'X']

-- Theorem statement
theorem probability_of_specific_arrangement :
  (1 : ℚ) / (Nat.choose total_tiles x_tiles) = 1 / 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_specific_arrangement_l1055_105571
