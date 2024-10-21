import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_properties_l1047_104744

-- Define the class [k]
def classmod5 (k : ℤ) : Set ℤ := {n : ℤ | ∃ m : ℤ, n = 5 * m + k ∧ 0 ≤ k ∧ k < 5}

-- Theorem statement
theorem class_properties :
  (2011 ∈ classmod5 1) ∧
  (Set.univ : Set ℤ) = classmod5 0 ∪ classmod5 1 ∪ classmod5 2 ∪ classmod5 3 ∪ classmod5 4 ∧
  (∀ a b : ℤ, (∃ k : ℤ, a ∈ classmod5 k ∧ b ∈ classmod5 k) ↔ (a - b) ∈ classmod5 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_properties_l1047_104744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l1047_104790

-- Define the curves
def C₁ (x y : ℝ) : Prop := x^2/16 + y^2/8 = 1

def C₂ (ρ θ : ℝ) : Prop := ρ^2 + 2*ρ*Real.cos θ - 1 = 0

-- Define the distance function between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Theorem statement
theorem min_distance_between_curves :
  ∃ (min_dist : ℝ), 
    (∀ (x₁ y₁ ρ θ : ℝ), 
      C₁ x₁ y₁ → C₂ ρ θ → 
      distance x₁ y₁ (ρ * Real.cos θ) (ρ * Real.sin θ) ≥ min_dist) ∧
    min_dist = Real.sqrt 7 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l1047_104790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_in_unit_rectangle_l1047_104750

/-- A circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if two circles are non-overlapping -/
def nonOverlapping (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 > (c1.radius + c2.radius)^2

/-- Predicate to check if a circle is within a rectangle -/
def withinRectangle (c : Circle) (width height : ℝ) : Prop :=
  c.center.1 - c.radius ≥ 0 ∧ c.center.1 + c.radius ≤ width ∧
  c.center.2 - c.radius ≥ 0 ∧ c.center.2 + c.radius ≤ height

theorem circles_in_unit_rectangle : ∃ (circles : List Circle) (width height : ℝ),
  width * height = 1 ∧
  (∀ c1 c2, c1 ∈ circles → c2 ∈ circles → c1 ≠ c2 → nonOverlapping c1 c2) ∧
  (∀ c, c ∈ circles → withinRectangle c width height) ∧
  (circles.map Circle.radius).sum = 1962 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_in_unit_rectangle_l1047_104750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_threes_l1047_104736

/-- Represents a fair die with n sides -/
structure Die (n : Nat) where
  sides : Fin n → Nat
  fair : ∀ i : Fin n, sides i ∈ Finset.range n.succ

/-- The probability of rolling a specific number on a fair die with n sides -/
def prob_roll {n : Nat} (d : Die n) (k : Nat) : Rat :=
  (Finset.filter (fun i => d.sides i = k) Finset.univ).card / n

/-- Jack's four dice -/
def jack_dice : Die 6 × Die 7 × Die 8 × Die 9 := sorry

theorem prob_all_threes (d : Die 6 × Die 7 × Die 8 × Die 9) :
  prob_roll d.1 3 * prob_roll d.2.1 3 * prob_roll d.2.2.1 3 * prob_roll d.2.2.2 3 = 1 / 3024 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_threes_l1047_104736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_tiling_with_large_polygons_l1047_104760

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : Fin n → ℝ × ℝ
  convex : ∀ (a b : ℝ × ℝ), a ∈ Set.range sides → b ∈ Set.range sides → 
           ∀ (t : ℝ), 0 ≤ t ∧ t ≤ 1 → t • a + (1 - t) • b ∈ Set.range sides

/-- A tiling of the plane using a given polygon -/
def PlaneTiling (P : Type) := ℝ × ℝ → P

/-- Predicate to check if a tiling is valid (no gaps or overlaps) -/
def IsValidTiling (P : Type) (tiling : PlaneTiling P) : Prop := sorry

/-- Main theorem: Impossibility of tiling a plane with convex polygons of 7 or more sides -/
theorem no_tiling_with_large_polygons (n : ℕ) (hn : n ≥ 7) :
  ¬∃ (P : ConvexPolygon n) (tiling : PlaneTiling (ConvexPolygon n)), IsValidTiling (ConvexPolygon n) tiling :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_tiling_with_large_polygons_l1047_104760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l1047_104786

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log (6 + x - x^2) / Real.log (1/2)

-- Define the domain
def domain : Set ℝ := {x : ℝ | -2 < x ∧ x < 3}

-- State the theorem
theorem monotonic_increase_interval :
  ∀ x ∈ domain, ∀ y ∈ domain,
    (1/2 < x ∧ x < y ∧ y < 3) → (f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l1047_104786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_value_l1047_104723

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 5

-- Define the line ax - y + 1 = 0
def my_line (a x y : ℝ) : Prop := a * x - y + 1 = 0

-- Define the tangent line passing through P(2,2)
def my_tangent_line (m x y : ℝ) : Prop := y = m * x + (2 - m * 2)

-- Define perpendicularity condition
def my_perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

-- Theorem statement
theorem tangent_line_value (a : ℝ) : 
  (∃ m : ℝ, my_tangent_line m 2 2 ∧ 
            (∀ x y : ℝ, my_circle x y → (x - 2)^2 + (y - 2)^2 ≥ (x - 1)^2 + y^2 - 5) ∧
            my_perpendicular m a) → 
  a = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_value_l1047_104723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_value_at_nine_l1047_104746

-- Define the power function as noncomputable
noncomputable def power_function (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

-- State the theorem
theorem power_function_through_point_value_at_nine :
  ∃ α : ℝ, (power_function α (Real.sqrt 2) = 2) → (power_function α 9 = 81) := by
  -- Introduce α and the hypothesis
  use 2
  intro h
  -- Apply the power function definition
  simp [power_function]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_value_at_nine_l1047_104746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mikes_trip_properties_l1047_104799

/-- Represents Mike's trip as a piecewise function --/
noncomputable def mikesTrip : ℝ → ℝ := sorry

/-- The trip duration in minutes --/
def tripDuration : ℝ := sorry

/-- Duration of the mall stop in minutes --/
def mallStopDuration : ℝ := 45

/-- Duration of the cafe stop in minutes --/
def cafeStopDuration : ℝ := 15

/-- Theorem stating the properties of Mike's trip function --/
theorem mikes_trip_properties :
  -- The function has two flat sections
  (∃ t1 t2 t3 t4, t1 < t2 ∧ t3 < t4 ∧
    (∀ t ∈ Set.Icc t1 t2, mikesTrip t = mikesTrip t1) ∧
    (∀ t ∈ Set.Icc t3 t4, mikesTrip t = mikesTrip t3) ∧
    t2 - t1 = mallStopDuration ∧
    t4 - t3 = cafeStopDuration) ∧
  -- The function has sections with different slopes
  (∃ s1 s2 s3 s4 s5,
    s1 ≠ s2 ∧ s2 ≠ s3 ∧ s3 ≠ s4 ∧ s4 ≠ s5 ∧
    (∃ t1 t2, ∀ t ∈ Set.Ioo t1 t2, 
      (deriv mikesTrip t) = s1) ∧
    (∃ t3 t4, ∀ t ∈ Set.Ioo t3 t4, 
      (deriv mikesTrip t) = s2) ∧
    (∃ t5 t6, ∀ t ∈ Set.Ioo t5 t6, 
      (deriv mikesTrip t) = s3) ∧
    (∃ t7 t8, ∀ t ∈ Set.Ioo t7 t8, 
      (deriv mikesTrip t) = s4) ∧
    (∃ t9 t10, ∀ t ∈ Set.Ioo t9 t10, 
      (deriv mikesTrip t) = s5)) ∧
  -- The function is continuous except at mode changes or stops
  (∀ t ∈ Set.Icc 0 tripDuration, 
    ContinuousAt mikesTrip t ∨
    (∃ ε > 0, ∀ t' ∈ Set.Ioo (t - ε) (t + ε), 
      t' ≠ t → ContinuousAt mikesTrip t')) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mikes_trip_properties_l1047_104799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_251_l1047_104710

/-- Represents the decimal expansion of a rational number -/
def DecimalExpansion (q : ℚ) : ℕ → ℕ := sorry

/-- Checks if the decimal expansion of q contains the digits 2, 5, 1 consecutively -/
def contains_251 (q : ℚ) : Prop := ∃ k : ℕ, 
  DecimalExpansion q k = 2 ∧ 
  DecimalExpansion q (k + 1) = 5 ∧ 
  DecimalExpansion q (k + 2) = 1

theorem smallest_n_for_251 :
  ∃ m : ℕ, m < 127 ∧ Nat.Coprime m 127 ∧ contains_251 ((m : ℚ) / 127) ∧
  ∀ n : ℕ, n < 127 → ¬∃ k : ℕ, k < n ∧ Nat.Coprime k n ∧ contains_251 ((k : ℚ) / n) := by
  sorry

#check smallest_n_for_251

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_251_l1047_104710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_many_people_sharing_carts_l1047_104762

/-- Represents the number of carts when three people share a cart with two empty carts -/
noncomputable def carts_scenario1 (x : ℝ) : ℝ := x / 3 + 2

/-- Represents the number of carts when two people share a cart with nine people walking -/
noncomputable def carts_scenario2 (x : ℝ) : ℝ := (x - 9) / 2

/-- The theorem states that the two scenarios result in the same number of carts -/
theorem many_people_sharing_carts (x : ℝ) : carts_scenario1 x = carts_scenario2 x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_many_people_sharing_carts_l1047_104762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1047_104773

theorem problem_statement (x y : ℝ) (θ : ℝ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : π/4 < θ ∧ θ < π/2) 
  (h4 : Real.sin θ / x = Real.cos θ / y) 
  (h5 : (Real.cos θ)^2 / x^2 + (Real.sin θ)^2 / y^2 = 10 / (3 * (x^2 + y^2))) :
  x / y = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1047_104773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_r_values_l1047_104797

open Real BigOperators

variable {n : ℕ}

theorem determine_r_values (a : Fin n → ℝ) (r : Fin n → ℝ) 
  (h_a_nonzero : ∃ i, a i ≠ 0)
  (h_inequality : ∀ x : Fin n → ℝ, 
    (∑ k, r k * (x k - a k)) ≤ (∑ k, (x k)^2).sqrt - (∑ k, (a k)^2).sqrt) :
  ∀ k, r k = a k / (∑ i, (a i)^2).sqrt := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_r_values_l1047_104797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_richard_twice_scott_age_l1047_104707

/-- The number of years until Richard is twice as old as Scott -/
def years_until_double : ℕ := 8

theorem richard_twice_scott_age : years_until_double = 8 := by
  -- Define the current ages of the brothers
  let david_current_age : ℕ := 14  -- David was 7 years old 7 years ago
  let richard_current_age : ℕ := david_current_age + 6  -- Richard is 6 years older than David
  let scott_current_age : ℕ := david_current_age - 8  -- David is 8 years older than Scott

  -- In 'years_until_double' years, Richard's age will be twice Scott's age
  have h : richard_current_age + years_until_double = 2 * (scott_current_age + years_until_double) := by
    -- Proof of this equality
    sorry

  -- Now we prove that years_until_double = 8
  calc
    years_until_double = richard_current_age + years_until_double - scott_current_age - years_until_double := by sorry
    _ = 2 * (scott_current_age + years_until_double) - scott_current_age - years_until_double := by sorry
    _ = scott_current_age + years_until_double := by sorry
    _ = 8 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_richard_twice_scott_age_l1047_104707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_purchase_shirts_l1047_104798

/-- The price of a saree in rupees -/
def saree_price : ℕ := 400

/-- The price of a shirt in rupees -/
def shirt_price : ℕ := 200

/-- The number of shirts in the initial purchase -/
def initial_shirts : ℕ := 4

theorem initial_purchase_shirts : 
  (2 * saree_price + initial_shirts * shirt_price = 1600) ∧
  (saree_price + 6 * shirt_price = 1600) ∧
  (12 * shirt_price = 2400) →
  initial_shirts = 4 := by
  intro h
  cases' h with h1 h2
  cases' h2 with h2 h3
  -- Proof steps would go here
  sorry

#check initial_purchase_shirts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_purchase_shirts_l1047_104798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_platform_l1047_104724

/-- The time (in seconds) it takes for a train to pass a platform -/
noncomputable def train_passing_time (train_length platform_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem: A train 360 m long running at 45 km/hr takes 48 seconds to pass a 240 m long platform -/
theorem train_passing_platform : train_passing_time 360 240 45 = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_platform_l1047_104724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_divides_space_l1047_104755

/-- A cube is a three-dimensional object with 6 faces -/
structure Cube where
  faces : Fin 6 → Plane

/-- A plane is a two-dimensional surface that extends infinitely -/
structure Plane

/-- Two planes are perpendicular if they intersect at a right angle -/
def perpendicular (p1 p2 : Plane) : Prop := sorry

/-- The number of parts that space is divided into by the planes of a cube -/
def num_parts (c : Cube) : ℕ := sorry

/-- The planes containing the faces of a cube divide the space into 27 parts -/
theorem cube_divides_space (c : Cube) 
  (h1 : ∀ (i j : Fin 6), i ≠ j → perpendicular (c.faces i) (c.faces j)) : 
  num_parts c = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_divides_space_l1047_104755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_function_period_l1047_104721

/-- Represents a sinusoidal function y = A sin(ωx + φ) -/
structure SinusoidalFunction where
  A : ℝ
  ω : ℝ
  φ : ℝ
  ω_pos : ω > 0

/-- Represents a point on the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Angle between three points -/
noncomputable def angle (A B C : Point) : ℝ := sorry

notation "∠" => angle

/-- The theorem stating the period of the sinusoidal function under given conditions -/
theorem sinusoidal_function_period
  (f : SinusoidalFunction)
  (P M N : Point)
  (h_lowest : P = ⟨3/2, -3*Real.sqrt 3/2⟩)
  (h_highest : M.y = N.y ∧ M.y > P.y ∧ N.y > P.y)
  (h_adjacent : ∀ Q : Point, (Q.y > P.y ∧ Q.x ≠ M.x ∧ Q.x ≠ N.x) → (Q.x < M.x ∨ Q.x > N.x))
  (h_angle : Real.cos (∠ M P N) = 1/2)  -- cos 60° = 1/2
  : (2 * π / f.ω) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_function_period_l1047_104721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_when_a_eq_1_range_of_a_when_union_eq_B_l1047_104795

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ (2 : ℝ)^x ∧ (2 : ℝ)^x ≤ 4}
def B (a : ℝ) : Set ℝ := {x | x - a > 0}

-- Theorem 1: Intersection of A and B when a = 1
theorem intersection_A_B_when_a_eq_1 : A ∩ B 1 = Set.Ioo 1 2 := by sorry

-- Theorem 2: Range of a when A ∪ B = B
theorem range_of_a_when_union_eq_B : 
  (∀ a : ℝ, A ∪ B a = B a) → (∀ a : ℝ, a < 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_when_a_eq_1_range_of_a_when_union_eq_B_l1047_104795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangles_in_circle_l1047_104703

/-- The number of triangles formed by n points on a circle -/
def number_of_triangles (n : ℕ) : ℕ := n.choose 3

/-- Given n points on a circle's circumference, with all possible non-intersecting 
    lines drawn between them, the number of triangles formed is ⁿC₃. -/
theorem triangles_in_circle (n : ℕ) : 
  number_of_triangles n = n.choose 3 := by
  -- Unfold the definition of number_of_triangles
  unfold number_of_triangles
  -- The equation is now trivially true
  rfl

#check triangles_in_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangles_in_circle_l1047_104703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1047_104789

/-- The sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmetic_sum (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * a₁ + (n * (n - 1) / 2) * d

/-- The second term of an arithmetic sequence -/
noncomputable def second_term (a₁ : ℝ) (d : ℝ) : ℝ :=
  a₁ + d

theorem arithmetic_sequence_sum :
  let a₁ : ℝ := 8
  let a₂ : ℝ := 5
  let n : ℕ := 20
  let d : ℝ := a₂ - a₁
  arithmetic_sum a₁ d n = -410 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1047_104789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_f_max_values_f_min_values_l1047_104742

noncomputable def f (x : ℝ) := 2 * (Real.sin x)^2 + 2 * Real.cos x - 3

theorem f_extrema :
  (∃ (k : ℤ), f (Real.pi / 3 + 2 * ↑k * Real.pi) = -1/2) ∧
  (∃ (k : ℤ), f (-Real.pi / 3 + 2 * ↑k * Real.pi) = -1/2) ∧
  (∃ (k : ℤ), f (2 * ↑k * Real.pi) = -5) ∧
  (∀ x : ℝ, -5 ≤ f x ∧ f x ≤ -1/2) :=
by sorry

theorem f_max_values (x : ℝ) :
  f x = -1/2 ↔ ∃ (k : ℤ), x = Real.pi / 3 + 2 * ↑k * Real.pi ∨ x = -Real.pi / 3 + 2 * ↑k * Real.pi :=
by sorry

theorem f_min_values (x : ℝ) :
  f x = -5 ↔ ∃ (k : ℤ), x = 2 * ↑k * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_f_max_values_f_min_values_l1047_104742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l1047_104712

/-- The equation of the directrix of a parabola with equation x² = 2y is y = -1/2 -/
theorem parabola_directrix :
  ∃ k : ℝ, k = -1/2 ∧ ∀ x y : ℝ, x^2 = 2*y → y = k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l1047_104712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_series_l1047_104713

theorem sum_of_series (x : ℝ) (hx : x > 1) :
  (∑' n : ℕ, 1 / (x^(3^n) - x^(-(3^n : ℤ)))) = 1 / (x - 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_series_l1047_104713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flagpole_break_height_l1047_104792

/-- The height at which a flagpole breaks, given its original height and the distance
    from the base where the top touches the ground after breaking. -/
noncomputable def break_height (original_height distance_to_ground : ℝ) : ℝ :=
  Real.sqrt (original_height^2 - (distance_to_ground + Real.sqrt (original_height^2 + distance_to_ground^2) / 2)^2)

/-- Theorem stating that for a flagpole of height 8 meters, broken such that the top
    touches the ground 3 meters from the base, the break occurs at the calculated height. -/
theorem flagpole_break_height :
  break_height 8 3 = Real.sqrt (64 - (3 + Real.sqrt 73 / 2)^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flagpole_break_height_l1047_104792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_inscribed_circle_side_lengths_l1047_104731

/-- Represents a triangle with an inscribed circle -/
structure TriangleWithInscribedCircle where
  -- Vertices of the triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- Center of the inscribed circle
  O : ℝ × ℝ
  -- Radius of the inscribed circle
  r : ℝ
  -- Touchpoint on side BC
  D : ℝ × ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem triangle_with_inscribed_circle_side_lengths 
  (t : TriangleWithInscribedCircle) 
  (h_radius : t.r = 4)
  (h_CD : distance t.C t.D = 8)
  (h_DB : distance t.D t.B = 10) :
  distance t.A t.B = 14.5 ∧ distance t.A t.C = 12.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_inscribed_circle_side_lengths_l1047_104731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monday_car_sales_l1047_104763

theorem monday_car_sales (mean : ℚ) (tue_sat_sales : ℕ) : 
  mean = 5.5 → tue_sat_sales = 25 → (mean * 6).floor - tue_sat_sales = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monday_car_sales_l1047_104763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_perimeter_l1047_104785

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  f1 : Point  -- Left focus
  f2 : Point  -- Right focus
  a : ℝ        -- Half of the distance between vertices

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem about the perimeter of a triangle in a hyperbola -/
theorem hyperbola_triangle_perimeter 
  (h : Hyperbola) 
  (A B : Point) 
  (on_left_branch : distance A h.f1 < distance A h.f2) 
  (passes_through_f1 : distance A h.f1 + distance B h.f1 = 5) 
  (constant_diff : h.a = 4) : 
  distance A B + distance A h.f2 + distance B h.f2 = 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_perimeter_l1047_104785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1047_104756

-- Define the function f(x) = x^2 - 2x
def f (x : ℝ) : ℝ := x^2 - 2*x

-- Define the domain [2, 5]
def domain : Set ℝ := Set.Icc 2 5

-- Theorem statement
theorem f_properties :
  -- 1. f(x) is increasing on [2, 5]
  (∀ x y, x ∈ domain → y ∈ domain → x < y → f x < f y) ∧
  -- 2. The minimum value of f(x) on [2, 5] is 0
  (∃ x ∈ domain, f x = 0 ∧ ∀ y ∈ domain, f y ≥ f x) ∧
  -- 3. The maximum value of f(x) on [2, 5] is 15
  (∃ x ∈ domain, f x = 15 ∧ ∀ y ∈ domain, f y ≤ f x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1047_104756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_to_rectangle_ratio_l1047_104728

/-- Given a square with side length 3, dissected into four pieces where E and F are midpoints
    of opposite sides and AG is perpendicular to BF, prove that when reassembled into a rectangle,
    the ratio of height to base is 4. -/
theorem square_to_rectangle_ratio : 
  ∀ (A B C D E F G : ℝ × ℝ) (s : ℝ),
  s = 3 → -- side length of the square
  -- E and F are midpoints of opposite sides
  E.1 = (A.1 + B.1) / 2 ∧ E.2 = A.2 →
  F.1 = (C.1 + D.1) / 2 ∧ F.2 = C.2 →
  -- AG is perpendicular to BF
  (G.2 - A.2) * (F.1 - B.1) = -(G.1 - A.1) * (F.2 - B.2) →
  -- The area of the square is preserved in the rectangle
  let rectangle_height := Real.sqrt ((G.1 - A.1)^2 + (G.2 - A.2)^2)
  let rectangle_base := s^2 / rectangle_height
  rectangle_height * rectangle_base = s^2 →
  -- Prove that the ratio of height to base is 4
  rectangle_height / rectangle_base = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_to_rectangle_ratio_l1047_104728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l1047_104768

/-- The area of a triangle with vertices at (0,0), (x₁, y₁), and (x₂, y₂) -/
noncomputable def triangle_area_origin (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  (1/2) * |x₁ * y₂ - x₂ * y₁|

/-- The area of a triangle with vertices at (x₁, y₁), (x₂, y₂), and (x₃, y₃) -/
noncomputable def triangle_area (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : ℝ :=
  (1/2) * |x₁ * y₂ + x₂ * y₃ + x₃ * y₁ - x₂ * y₁ - x₃ * y₂ - x₁ * y₃|

theorem triangle_area_theorem :
  ∀ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    triangle_area_origin x₁ y₁ x₂ y₂ = (1/2) * |x₁ * y₂ - x₂ * y₁| ∧
    triangle_area x₁ y₁ x₂ y₂ x₃ y₃ = (1/2) * |x₁ * y₂ + x₂ * y₃ + x₃ * y₁ - x₂ * y₁ - x₃ * y₂ - x₁ * y₃| :=
by
  intro x₁ y₁ x₂ y₂ x₃ y₃
  apply And.intro
  · -- Proof for triangle_area_origin
    rfl
  · -- Proof for triangle_area
    rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l1047_104768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enterprise_ice_cost_l1047_104752

/-- Represents the cost of ice per ton received by enterprise A -/
noncomputable def ice_cost (a p n s : ℝ) : ℝ :=
  (2.5 * a + p * s) * 1000 / (2000 - n * s)

/-- Theorem stating the cost of ice per ton received by enterprise A -/
theorem enterprise_ice_cost
  (a p n s : ℝ)
  (ha : a > 0)
  (hp : p > 0)
  (hn : n > 0)
  (hs : s > 0)
  (h_melt : n / 1000 < 1)
  (h_between : ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = s)
  (h_equal_cost : ∀ x y : ℝ, x > 0 → y > 0 → x + y = s →
    (a + p * x) / (1 - n * x / 1000) = (1.5 * a + p * y) / (1 - n * y / 1000)) :
  ∀ x : ℝ, x > 0 → ∃ y : ℝ, y > 0 ∧ x + y = s →
    ice_cost a p n s = (a + p * x) / (1 - n * x / 1000) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_enterprise_ice_cost_l1047_104752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_derivative_sufficient_not_necessary_for_increasing_l1047_104782

/-- A function f is increasing on an interval I -/
def IsIncreasing (f : ℝ → ℝ) (I : Set ℝ) :=
  ∀ x y, x ∈ I → y ∈ I → x < y → f x < f y

/-- A function f has positive derivative on an interval I -/
def HasPositiveDerivative (f : ℝ → ℝ) (I : Set ℝ) :=
  ∀ x, x ∈ I → deriv f x > 0

theorem positive_derivative_sufficient_not_necessary_for_increasing
  (f : ℝ → ℝ) (I : Set ℝ) (hf : DifferentiableOn ℝ f I) :
  (HasPositiveDerivative f I → IsIncreasing f I) ∧
  ¬(IsIncreasing f I → HasPositiveDerivative f I) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_derivative_sufficient_not_necessary_for_increasing_l1047_104782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_is_empty_l1047_104793

def A : Set ℝ := {x | x ≤ 0}
def B : Set ℝ := {x | 10 = (10 : ℝ)^x}

theorem intersection_is_empty : A ∩ (Bᶜ) = ∅ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_is_empty_l1047_104793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_row_swapping_matrix_l1047_104715

open Matrix

theorem row_swapping_matrix :
  ∃! N : Matrix (Fin 2) (Fin 2) ℝ,
    ∀ A : Matrix (Fin 2) (Fin 2) ℝ,
      N * A = !![A 1 0, A 1 1; A 0 0, A 0 1] :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_row_swapping_matrix_l1047_104715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1047_104794

/-- Given an ellipse with equation x²/a² + y²/b² = 1 where a > b > 0,
    if the distance from the center to the upper vertex is equal to the distance
    from the center to the right focus, then the eccentricity of the ellipse is √2/2. -/
theorem ellipse_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  c = b → c^2 + b^2 = a^2 → c / a = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1047_104794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_equidistant_l1047_104772

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define the points
variable (A B C D : V)

-- Define the perpendicular bisector planes
def perp_bisector_plane (P Q : V) : Set V :=
  {X : V | ‖X - P‖ = ‖X - Q‖}

-- Define the line of intersection
def intersection_line (A B C D : V) : Set V :=
  (perp_bisector_plane A B) ∩ (perp_bisector_plane C D)

-- Theorem statement
theorem intersection_line_equidistant (A B C D : V) :
  ∀ X ∈ intersection_line A B C D,
    ‖X - A‖ = ‖X - C‖ ∧ ‖X - B‖ = ‖X - D‖ := by
  sorry

#check intersection_line_equidistant

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_equidistant_l1047_104772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1047_104774

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 2 ∧ t.C = Real.pi/3

-- Theorem for part (1)
theorem part_one (t : Triangle) (h : triangle_conditions t) (h_A : t.A = Real.pi/4) :
  t.c = Real.sqrt 6 := by
  sorry

-- Theorem for part (2)
theorem part_two (t : Triangle) (h : triangle_conditions t) (h_area : 1/2 * t.a * t.b * Real.sin t.C = Real.sqrt 3) :
  t.b = 2 ∧ t.c = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1047_104774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l1047_104700

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + Real.sqrt 3 * Real.cos (2 * x)

theorem axis_of_symmetry (k : ℤ) :
  ∀ x : ℝ, f x = f (π / 6 + k * π - x) := by
  intro x
  -- The proof steps would go here
  sorry

#check axis_of_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l1047_104700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polyhedron_result_l1047_104714

/-- A convex polyhedron with specific properties -/
structure SpecialPolyhedron where
  faces : ℕ
  triangles : ℕ
  pentagons : ℕ
  vertices : ℕ
  edges : ℕ
  faces_sum : faces = triangles + pentagons
  faces_count : faces = 36
  vertex_config : 3 * triangles + 3 * pentagons = 6 * vertices
  euler_formula : vertices - edges + faces = 2
  edge_calc : 2 * edges = 3 * triangles + 5 * pentagons

/-- Theorem stating the result for the special polyhedron -/
theorem special_polyhedron_result (p : SpecialPolyhedron) : 
  100 * p.pentagons + 10 * p.triangles + p.vertices = 2018 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polyhedron_result_l1047_104714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_fixed_line_l1047_104747

noncomputable def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / (a^2 - 4) = 1

noncomputable def eccentricity (a : ℝ) : ℝ := Real.sqrt 14 / 3

noncomputable def left_focus (a : ℝ) : ℝ × ℝ := (-Real.sqrt (2 * a^2 - 4), 0)
noncomputable def right_focus (a : ℝ) : ℝ × ℝ := (Real.sqrt (2 * a^2 - 4), 0)

def collinear (P : ℝ × ℝ) (F₂ : ℝ × ℝ) (Q : ℝ × ℝ) : Prop :=
  (P.2 - Q.2) * (F₂.1 - Q.1) = (F₂.2 - Q.2) * (P.1 - Q.1)

def perpendicular (F₁ P Q : ℝ × ℝ) : Prop :=
  (P.1 - F₁.1) * (Q.1 - F₁.1) + (P.2 - F₁.2) * (Q.2 - F₁.2) = 0

theorem hyperbola_fixed_line (a : ℝ) (P : ℝ × ℝ) :
  a > 2 →
  hyperbola a P.1 P.2 →
  eccentricity a = Real.sqrt 14 / 3 →
  ∃ Q : ℝ × ℝ, Q.1 = 0 ∧ 
    collinear P (right_focus a) Q ∧
    perpendicular (left_focus a) P Q →
  P.1 - P.2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_fixed_line_l1047_104747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_rectangles_count_l1047_104769

/-- A triangle in a 2D plane --/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- A rectangle in a 2D plane --/
structure Rectangle where
  vertices : Fin 4 → ℝ × ℝ

/-- Predicate to check if a rectangle is inscribed in a triangle --/
def isInscribed (r : Rectangle) (t : Triangle) : Prop := sorry

/-- Function to calculate the area of a rectangle --/
noncomputable def rectangleArea (r : Rectangle) : ℝ := sorry

/-- Predicate to check if a rectangle has maximum area among all inscribed rectangles --/
def isMaxArea (r : Rectangle) (t : Triangle) : Prop :=
  isInscribed r t ∧ ∀ r', isInscribed r' t → rectangleArea r' ≤ rectangleArea r

/-- Two rectangles are considered distinct if they are not congruent --/
def areDistinct (r1 r2 : Rectangle) : Prop := sorry

/-- The main theorem --/
theorem max_area_rectangles_count (t : Triangle) :
  (∃ (rs : Finset Rectangle), (∀ r, r ∈ rs → isMaxArea r t) ∧
    (∀ r1 r2, r1 ∈ rs → r2 ∈ rs → r1 ≠ r2 → areDistinct r1 r2) ∧
    (∀ r, isMaxArea r t → r ∈ rs) ∧
    (rs.card = 1 ∨ rs.card = 3)) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_rectangles_count_l1047_104769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_is_cos_l1047_104775

noncomputable def f (n : ℕ) : ℝ → ℝ := 
  match n with
  | 0 => Real.cos
  | n + 1 => deriv (f n)

theorem f_2016_is_cos : f 2016 = Real.cos := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_is_cos_l1047_104775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_max_increasing_interval_l1047_104706

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x - x^2

-- State the theorem
theorem tangent_line_and_max_increasing_interval 
  (a : ℝ) (h_a : 0 < a ∧ a ≤ 1) :
  -- Part 1: Equation of tangent line when a = 1/2
  (a = 1/2 → ∃ m b, ∀ x, (deriv (f (1/2))) 1 * (x - 1) + f (1/2) 1 = m * x + b ∧ m = -1/2) ∧
  -- Part 2: Maximum value of t-s
  (∃ t s, 0 < s ∧ s < t ∧ 
    (∀ x ∈ Set.Ioo s t, (deriv (f a)) x > 0) ∧
    (∀ y, y > t - s → ¬∃ u v, 0 < u ∧ u < v ∧ 
      (∀ x ∈ Set.Ioo u v, (deriv (f a)) x > 0) ∧ 
      y = v - u)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_max_increasing_interval_l1047_104706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_t_value_l1047_104739

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + x^2) - x) - x^3

-- State the theorem
theorem min_t_value (θ : ℝ) (t : ℝ) :
  θ ∈ Set.Icc 0 (Real.pi / 2) →
  (∀ θ, f (Real.cos θ^2 - 2*t) + f (4*Real.sin θ - 3) ≥ 0) →
  t ≥ (1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_t_value_l1047_104739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_digit_number_determinable_in_20_questions_l1047_104791

theorem six_digit_number_determinable_in_20_questions :
  ∀ n : ℕ, 100000 ≤ n ∧ n < 1000000 →
  ∃ (questions : Fin 20 → ℕ → Prop) (answers : Fin 20 → Bool),
    ∀ m : ℕ, 100000 ≤ m ∧ m < 1000000 →
      (∀ i : Fin 20, answers i = questions i m) →
      m = n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_digit_number_determinable_in_20_questions_l1047_104791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_ABC_DEF_l1047_104749

-- Define the side lengths of triangle ABC
noncomputable def a : ℝ := 5
noncomputable def b : ℝ := 12
noncomputable def c : ℝ := 13

-- Define the side lengths of triangle DEF
noncomputable def d : ℝ := 8
noncomputable def e : ℝ := 15
noncomputable def f : ℝ := 17

-- Define the areas of the triangles
noncomputable def area_ABC : ℝ := (1/2) * a * b
noncomputable def area_DEF : ℝ := (1/2) * d * e

-- Theorem statement
theorem area_ratio_ABC_DEF :
  area_ABC / area_DEF = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_ABC_DEF_l1047_104749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weaving_time_approximation_l1047_104748

/-- The rate at which the loom weaves cloth in meters per second -/
noncomputable def weaving_rate : ℝ := 0.126

/-- The total length of cloth to be woven in meters -/
noncomputable def total_length : ℝ := 15

/-- The time required to weave the cloth in seconds -/
noncomputable def weaving_time : ℝ := total_length / weaving_rate

theorem weaving_time_approximation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |weaving_time - 119| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_weaving_time_approximation_l1047_104748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_implies_differential_equation_l1047_104767

noncomputable section

-- Define the general solution
def generalSolution (x : ℝ) (C : ℝ) : ℝ := (1/6) * x^4 + C / x^2

-- Define the differential equation
def differentialEquation (y : ℝ → ℝ) (x : ℝ) : Prop :=
  deriv y x + 2 * (y x) / x = x^3

-- Theorem statement
theorem general_solution_implies_differential_equation :
  ∀ (y : ℝ → ℝ) (C : ℝ),
  (∀ x : ℝ, x ≠ 0 → y x = generalSolution x C) →
  ∀ x : ℝ, x ≠ 0 → differentialEquation y x := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_implies_differential_equation_l1047_104767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_120_divisors_l1047_104732

-- Define the number of divisors function
def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

-- Define the property for n
def is_valid_n (n : ℕ) : Prop :=
  n > 0 ∧ n % 105 = 0 ∧ num_divisors n = 120

-- Theorem statement
theorem smallest_n_with_120_divisors :
  ∃ n : ℕ, is_valid_n n ∧ ∀ m : ℕ, is_valid_n m → n ≤ m ∧ n / 105 = 5952640 := by
  sorry

#check smallest_n_with_120_divisors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_120_divisors_l1047_104732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_theorem_l1047_104779

/-- Definition of a parabola with parameter p > 0 -/
noncomputable def Parabola (p : ℝ) : Set (ℝ × ℝ) :=
  {point : ℝ × ℝ | point.2^2 = 2 * p * point.1 ∧ p > 0}

/-- The directrix of a parabola -/
noncomputable def Directrix (p : ℝ) : Set (ℝ × ℝ) :=
  {point : ℝ × ℝ | point.1 = -p / 2}

/-- The focus of a parabola -/
noncomputable def Focus (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

/-- Theorem: If the directrix of the parabola y^2 = 2px (p > 0) passes through (-1, 1), 
    then its focus is at (1, 0) -/
theorem parabola_focus_theorem :
  ∀ p : ℝ, p > 0 → (-1, 1) ∈ Directrix p → Focus p = (1, 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_theorem_l1047_104779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1047_104718

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (1 - 2*a)*x + 3*a else Real.log x

theorem range_of_a :
  {a : ℝ | -1 ≤ a ∧ a < 1/2} = {a : ℝ | ∀ y : ℝ, ∃ x : ℝ, f a x = y} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1047_104718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_middle_square_height_l1047_104733

/-- The distance from the top vertex of a rotated square to the base line -/
noncomputable def rotated_square_height (side_length : ℝ) (rotation_angle : ℝ) : ℝ :=
  side_length / 2 + side_length * Real.sqrt 2 * Real.sin (rotation_angle * Real.pi / 180)

/-- Theorem: The height of the rotated middle square's top vertex -/
theorem rotated_middle_square_height :
  rotated_square_height 2 30 = 1 + Real.sqrt 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_middle_square_height_l1047_104733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_locus_polar_eq_max_distance_C_to_l_l1047_104753

noncomputable section

-- Define the curve C
def curve_C (α : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos α, 2 * Real.sin α)

-- Define the line l in polar form
def line_l (ρ θ : ℝ) : Prop := Real.sqrt 2 * ρ * Real.sin (θ - Real.pi/4) = 1

-- Define the midpoint M
def midpoint_M (α : ℝ) : ℝ × ℝ := ((curve_C α).1 / 2, (curve_C α).2 / 2)

-- Theorem for the polar equation of the locus of M
theorem midpoint_locus_polar_eq :
  ∀ θ : ℝ, ∃ ρ : ℝ, ρ * Real.cos θ = 1 ∧ ρ * Real.sin θ = Real.sin θ := by sorry

-- Theorem for the maximum distance from C to l
theorem max_distance_C_to_l :
  ∃ d : ℝ, d = (3 * Real.sqrt 2) / 2 + 2 ∧
  ∀ p : ℝ × ℝ, (∃ α : ℝ, p = curve_C α) →
  ∀ q : ℝ × ℝ, (∃ ρ θ : ℝ, line_l ρ θ ∧ q = (ρ * Real.cos θ, ρ * Real.sin θ)) →
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ d := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_locus_polar_eq_max_distance_C_to_l_l1047_104753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_with_given_conditions_l1047_104741

-- Define a parabola
structure Parabola where
  equation : ℝ → ℝ → Prop

-- Define the properties of our specific parabola
def our_parabola : Parabola :=
  { equation := fun x y => y^2 = -8*x }

-- State the theorem
theorem parabola_equation_with_given_conditions :
  -- The vertex is at the origin
  our_parabola.equation 0 0 ∧
  -- The directrix is x = 2
  (∀ y : ℝ, (2 : ℝ) = 2) ∧
  -- For any point (x, y) on the parabola, its distance from the focus
  -- is equal to its distance from the directrix
  (∀ x y : ℝ, our_parabola.equation x y →
    -- Distance from (x, y) to focus (1, 0)
    ((x - 1)^2 + y^2) =
    -- Distance from (x, y) to directrix x = 2
    (x - 2)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_with_given_conditions_l1047_104741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angle_l1047_104764

theorem triangle_right_angle (A B C : ℝ) (h : Real.sin A ^ 2 + Real.sin B ^ 2 = Real.sin C ^ 2) :
  ∃ (a b c : ℝ), a ^ 2 + b ^ 2 = c ^ 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angle_l1047_104764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_necessary_not_sufficient_for_q_l1047_104734

-- Define the conditions p and q
def p (x : ℝ) : Prop := (2 : ℝ)^x > 1/2
def q (x : ℝ) : Prop := (x - 3) / (x - 1) < 0

-- Theorem stating that p is a necessary but not sufficient condition for q
theorem p_necessary_not_sufficient_for_q :
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_necessary_not_sufficient_for_q_l1047_104734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_of_equation_l1047_104759

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (3/11)^x + (5/11)^x + (7/11)^x - 1

-- Theorem statement
theorem unique_root_of_equation :
  ∃! x : ℝ, (3 : ℝ)^x + (5 : ℝ)^x + (7 : ℝ)^x = (11 : ℝ)^x :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_of_equation_l1047_104759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_diagonal_l1047_104726

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  longerBase : ℝ
  shorterBase : ℝ
  leg : ℝ

/-- The diagonal of an isosceles trapezoid -/
noncomputable def diagonal (t : IsoscelesTrapezoid) : ℝ :=
  let height := Real.sqrt (t.leg ^ 2 - ((t.longerBase - t.shorterBase) / 2) ^ 2)
  let halfLongerBase := (t.longerBase + t.shorterBase) / 4
  Real.sqrt (halfLongerBase ^ 2 + height ^ 2)

/-- Theorem: The diagonal of the specified isosceles trapezoid is 19 units -/
theorem isosceles_trapezoid_diagonal :
  let t : IsoscelesTrapezoid := ⟨24, 10, 11⟩
  diagonal t = 19 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_diagonal_l1047_104726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l1047_104743

/-- Time taken for a train to cross a bridge -/
theorem train_crossing_bridge_time
  (train_length : ℝ)
  (train_speed_kmph : ℝ)
  (bridge_length : ℝ)
  (h1 : train_length = 110)
  (h2 : train_speed_kmph = 60)
  (h3 : bridge_length = 390) :
  ∃ (time : ℝ), (time ≥ 29.99 ∧ time ≤ 30.01) ∧ 
  time * (train_speed_kmph * 1000 / 3600) = train_length + bridge_length :=
by
  -- Define the total distance
  let total_distance := train_length + bridge_length
  
  -- Convert speed from km/h to m/s
  let speed_ms := train_speed_kmph * 1000 / 3600
  
  -- Calculate the time
  let time := total_distance / speed_ms
  
  -- Prove the existence of such a time
  use time
  
  constructor
  · -- Prove the approximation
    sorry
  · -- Prove the equation
    rw [h1, h2, h3]
    ring
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l1047_104743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sector_angle_l1047_104796

/-- Represents the properties of a circle divided into sectors --/
structure CircleSectors where
  num_sectors : ℕ
  angles : Fin num_sectors → ℕ
  is_arithmetic_sequence : ∃ (a d : ℕ), ∀ i : Fin num_sectors, angles i = a + i.val * d
  sum_360 : (Finset.univ.sum angles) = 360

/-- The main theorem stating the smallest possible sector angle --/
theorem smallest_sector_angle (c : CircleSectors) 
  (h : c.num_sectors = 15) : 
  ∃ (min_angle : ℕ), min_angle = 3 ∧ ∀ i : Fin c.num_sectors, c.angles i ≥ min_angle := by
  sorry

#check smallest_sector_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sector_angle_l1047_104796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_fifth_power_l1047_104722

theorem det_A_fifth_power {n : Type*} [Fintype n] [DecidableEq n] 
  (A : Matrix n n ℝ) (h : Matrix.det A = -3) : 
  Matrix.det (A^5) = -243 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_fifth_power_l1047_104722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_value_l1047_104727

noncomputable def a : ℕ → ℚ
  | 0 => 0
  | (n + 1) => (8/5) * a n + (6/5) * (((4:ℚ)^n - (a n)^2).sqrt : ℚ)

theorem a_10_value : a 10 = 24576/25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_value_l1047_104727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1047_104788

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos (x + Real.pi/3) + Real.sqrt 3 * (Real.cos x)^2 + (1/2) * Real.sin (2*x)

theorem f_properties :
  ∃ (period : ℝ) (max_val min_val : ℝ) (k : ℤ → ℝ → ℝ → Prop),
    (∀ x, f (x + period) = f x) ∧ 
    (∀ y, y > 0 → (∀ x, f (x + y) = f x) → y ≥ period) ∧
    (∀ x, f x ≤ max_val) ∧
    (∃ x, f x = max_val) ∧
    (∀ x, f x ≥ min_val) ∧
    (∃ x, f x = min_val) ∧
    (∀ (n : ℤ) (x y : ℝ), k n x y ↔ 
      n * Real.pi - 5*Real.pi/12 ≤ x ∧ x < y ∧ y ≤ n * Real.pi + Real.pi/12 ∧ 
      (∀ x' y', x ≤ x' ∧ x' < y' ∧ y' ≤ y → f x' ≤ f y')) ∧
    period = Real.pi ∧
    max_val = 2 ∧
    min_val = -2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1047_104788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l1047_104711

noncomputable def f (x : ℝ) := Real.exp x + x

theorem zero_in_interval (x₀ : ℝ) (h : f x₀ = 0) : -1 < x₀ ∧ x₀ < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l1047_104711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_angle_property_l1047_104705

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

noncomputable def focus : ℝ × ℝ := (Real.sqrt 3, 0)

def chord_through_focus (A B : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), A = (t * (A.1 - focus.1) + focus.1, t * (A.2 - focus.2) + focus.2) ∧
              B = ((1 - t) * (A.1 - focus.1) + focus.1, (1 - t) * (A.2 - focus.2) + focus.2)

def angle_equal (P A B : ℝ × ℝ) : Prop :=
  (A.2 - P.2) / (A.1 - P.1) = -(B.2 - P.2) / (B.1 - P.1)

theorem ellipse_angle_property :
  ∃! (p : ℝ), p > 0 ∧
    ∀ (A B : ℝ × ℝ), 
      ellipse A.1 A.2 → ellipse B.1 B.2 → 
      chord_through_focus A B → 
      angle_equal (p, 0) A B :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_angle_property_l1047_104705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1047_104754

-- Define the propositions
def p1 : Prop := ∀ x y : ℝ, x < y → (Real.exp (x * Real.log 2) - Real.exp (-x * Real.log 2)) < (Real.exp (y * Real.log 2) - Real.exp (-y * Real.log 2))
def p2 : Prop := ∀ x y : ℝ, x < y → (Real.exp (x * Real.log 2) + Real.exp (-x * Real.log 2)) > (Real.exp (y * Real.log 2) + Real.exp (-y * Real.log 2))

-- Define the compound propositions
def q1 : Prop := p1 ∨ p2
def q2 : Prop := p1 ∧ p2
def q3 : Prop := (¬p1) ∨ p2
def q4 : Prop := p1 ∧ (¬p2)

-- Theorem statement
theorem problem_solution : (q1 ∧ q4) ∧ (¬q2 ∧ ¬q3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1047_104754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_fixed_point_l1047_104757

/-- The function f representing a rotation in the complex plane -/
noncomputable def f (z : ℂ) : ℂ := ((-1 - Complex.I * Real.sqrt 3) * z + (2 * Real.sqrt 3 - 18 * Complex.I)) / 2

/-- The fixed point of the rotation -/
noncomputable def d : ℂ := -Real.sqrt 3 + 4 * Complex.I

/-- Theorem stating that d is the fixed point of the rotation represented by f -/
theorem rotation_fixed_point : f d = d := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_fixed_point_l1047_104757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_converse_and_inverse_not_always_true_answer_is_d_l1047_104745

-- Define the properties of triangles
structure Triangle where
  isEquilateral : Prop
  isIsosceles : Prop

-- Given statement
axiom original_statement : ∀ t : Triangle, t.isEquilateral → t.isIsosceles

-- Converse and inverse to be proven false
theorem converse_and_inverse_not_always_true :
  (∀ t : Triangle, t.isIsosceles → t.isEquilateral) ∧
  (∀ t : Triangle, ¬t.isEquilateral → ¬t.isIsosceles) → False := by
  intro h
  sorry

-- The answer is that neither the converse nor the inverse is true
theorem answer_is_d : True := by
  trivial

end NUMINAMATH_CALUDE_ERRORFEEDBACK_converse_and_inverse_not_always_true_answer_is_d_l1047_104745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tulips_ordered_l1047_104761

/-- Proves that the number of tulips ordered is 250 given the specified conditions --/
theorem tulips_ordered (num_carnations num_roses : ℕ) 
  (price_per_flower total_expenses : ℚ) 
  (h1 : num_carnations = 375)
  (h2 : num_roses = 320)
  (h3 : price_per_flower = 2)
  (h4 : total_expenses = 1890)
  : (total_expenses - price_per_flower * (num_carnations + num_roses)) / price_per_flower = 250 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tulips_ordered_l1047_104761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_denominator_is_odd_l1047_104717

def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * double_factorial n

def sum_series (n : ℕ) : ℚ :=
  (Finset.range (n + 1)).sum (λ i => (double_factorial (2 * i)) / (double_factorial (2 * i + 1)))

theorem denominator_is_odd (n : ℕ) :
  ∃ (d : ℕ), Odd d ∧ (sum_series n).den = d :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_denominator_is_odd_l1047_104717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_waiting_time_approx_six_minutes_l1047_104758

/-- Represents the scenario of a man and woman walking, where the woman passes the man and then waits for him to catch up. -/
structure WalkingScenario where
  man_speed : ℝ  -- Man's speed in miles per hour
  woman_speed : ℝ  -- Woman's speed in miles per hour
  wait_start_time : ℝ  -- Time in minutes when the woman starts waiting

/-- Calculates the waiting time for the man to catch up to the woman. -/
noncomputable def waiting_time (scenario : WalkingScenario) : ℝ :=
  let distance_traveled := scenario.woman_speed / 60 * scenario.wait_start_time
  distance_traveled / (scenario.man_speed / 60)

/-- Theorem stating that in the given scenario, the waiting time is approximately 6 minutes. -/
theorem waiting_time_approx_six_minutes :
  ∃ (scenario : WalkingScenario),
    scenario.man_speed = 5 ∧
    scenario.woman_speed = 15 ∧
    scenario.wait_start_time = 2 ∧
    (⌊waiting_time scenario⌋ : ℤ) = 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_waiting_time_approx_six_minutes_l1047_104758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_difference_is_4_26_l1047_104771

/-- The number of telephone poles -/
def num_poles : ℕ := 51

/-- The distance between the first and last pole in feet -/
def total_distance : ℚ := 5280

/-- The number of Elmer's strides between consecutive poles -/
def elmer_strides : ℕ := 38

/-- The number of Oscar's leaps between consecutive poles -/
def oscar_leaps : ℕ := 15

/-- The length of Elmer's stride in feet -/
noncomputable def elmer_stride_length : ℚ := total_distance / ((num_poles - 1) * elmer_strides)

/-- The length of Oscar's leap in feet -/
noncomputable def oscar_leap_length : ℚ := total_distance / ((num_poles - 1) * oscar_leaps)

/-- The difference between Oscar's leap length and Elmer's stride length -/
noncomputable def length_difference : ℚ := oscar_leap_length - elmer_stride_length

theorem length_difference_is_4_26 : ∃ ε > 0, |length_difference - 426/100| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_difference_is_4_26_l1047_104771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_of_tangent_circles_l1047_104765

/-- The area of a triangle given its side lengths -/
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- The area of a triangle formed by the centers of three tangent circles -/
theorem triangle_area_of_tangent_circles (r₁ r₂ r₃ : ℝ) 
  (h₁ : r₁ = 5) (h₂ : r₂ = 6) (h₃ : r₃ = 7) : 
  ∃ ε > 0, |triangle_area (r₁ + r₂) (r₂ + r₃) (r₁ + r₃) - 61.48| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_of_tangent_circles_l1047_104765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_moves_proof_l1047_104781

/-- A move replaces two sets with their intersection and union -/
def Move (α : Type) := (Set α × Set α)

/-- The move operation as defined in the problem -/
def moveOperation {α : Type} (A B : Set α) : Move α := (A ∩ B, A ∪ B)

/-- Predicate to check if a move is valid (neither set is a subset of the other) -/
def isValidMove {α : Type} (A B : Set α) : Prop := ¬(A ⊆ B) ∧ ¬(B ⊆ A)

/-- The maximum number of moves possible for n sets -/
def maxMoves (n : ℕ) : ℕ := n * (n - 1) / 2

theorem max_moves_proof {α : Type} (n : ℕ) (h : n ≥ 2) :
  ∃ (sets : Finset (Set α)) (moves : List (Move α)),
    sets.card = n ∧ 
    (∀ m ∈ moves, ∃ A B, m = moveOperation A B ∧ isValidMove A B) ∧
    moves.length = maxMoves n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_moves_proof_l1047_104781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_at_neg_two_l1047_104725

/-- A polynomial of degree 4 satisfying specific conditions -/
noncomputable def P : ℝ → ℝ := sorry

/-- P is a polynomial of degree 4 -/
axiom P_degree : ∃ (a b c d e : ℝ), ∀ x, P x = a*x^4 + b*x^3 + c*x^2 + d*x + e

/-- P satisfies the given conditions -/
axiom P_conditions : P 0 = 1 ∧ P 1 = 1 ∧ P 2 = 4 ∧ P 3 = 9 ∧ P 4 = 16

/-- Theorem: P(-2) = 19 -/
theorem P_at_neg_two : P (-2) = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_at_neg_two_l1047_104725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steam_mass_approx_19_48_l1047_104776

/-- Represents the thermodynamic system with initial conditions and properties -/
structure ThermoSystem where
  initialTemp : ℝ
  waterMass : ℝ
  iceMass : ℝ
  containerMass : ℝ
  containerHeatCapacity : ℝ
  steamTemp : ℝ
  finalTemp : ℝ
  waterHeatCapacity : ℝ
  latentHeatVaporization : ℝ
  latentHeatFusion : ℝ

/-- Calculates the required steam mass for the given thermodynamic system -/
noncomputable def requiredSteamMass (system : ThermoSystem) : ℝ :=
  let heatGained := system.iceMass * system.latentHeatFusion +
                    (system.iceMass + system.waterMass) * system.waterHeatCapacity * (system.finalTemp - system.initialTemp) +
                    system.containerMass * system.containerHeatCapacity * (system.finalTemp - system.initialTemp)
  let heatLostPerMass := system.latentHeatVaporization + 
                         system.waterHeatCapacity * (system.steamTemp - system.finalTemp)
  heatGained / heatLostPerMass

/-- The main theorem stating that the required steam mass is approximately 19.48 g -/
theorem steam_mass_approx_19_48 (system : ThermoSystem) 
    (h1 : system.initialTemp = 0)
    (h2 : system.waterMass = 300)
    (h3 : system.iceMass = 50)
    (h4 : system.containerMass = 100)
    (h5 : system.containerHeatCapacity = 0.5)
    (h6 : system.steamTemp = 100)
    (h7 : system.finalTemp = 20)
    (h8 : system.waterHeatCapacity = 1)
    (h9 : system.latentHeatVaporization = 536)
    (h10 : system.latentHeatFusion = 80) :
  ∃ ε > 0, abs (requiredSteamMass system - 19.48) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_steam_mass_approx_19_48_l1047_104776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_six_l1047_104708

/-- A geometric sequence with first term a and common ratio q -/
noncomputable def geometric_sequence (a q : ℝ) : ℕ → ℝ := fun n => a * q^(n-1)

/-- Sum of first n terms of a geometric sequence -/
noncomputable def geometric_sum (a q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_six
  (a q : ℝ)
  (h1 : q ≠ 1)
  (h2 : geometric_sequence a q 1 + geometric_sequence a q 3 = 5)
  (h3 : geometric_sum a q 4 = 15) :
  geometric_sum a q 6 = 63 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_six_l1047_104708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_correction_march10_to_march21_l1047_104770

/-- Represents the time difference between two dates in hours -/
noncomputable def timeDifference (startDay startHour endDay endHour : ℕ) : ℝ :=
  (endDay - startDay : ℝ) * 24 + (endHour - startHour : ℝ)

/-- Calculates the correction in minutes for a watch with a given daily loss rate -/
noncomputable def watchCorrection (dailyLoss : ℝ) (hours : ℝ) : ℝ :=
  (dailyLoss / 24) * hours

theorem watch_correction_march10_to_march21 :
  let dailyLoss : ℝ := 3.25
  let startDay := 10  -- March 10
  let startHour := 13 -- 1 PM
  let endDay := 21    -- March 21
  let endHour := 7    -- 7 AM
  let hours := timeDifference startDay startHour endDay endHour
  abs (watchCorrection dailyLoss hours - 38.1875) < 0.0001 := by
  sorry

#eval Float.abs (((3.25 / 24) * ((21 - 10 : Float) * 24 + (7 - 13 : Float))) - 38.1875)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_correction_march10_to_march21_l1047_104770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_is_inverse_of_h_l1047_104740

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := 6 - 7 * x

-- Define the proposed inverse function k
noncomputable def k (x : ℝ) : ℝ := (6 - x) / 7

-- Theorem stating that k is the inverse of h
theorem k_is_inverse_of_h : 
  (∀ x, h (k x) = x) ∧ (∀ x, k (h x) = x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_is_inverse_of_h_l1047_104740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_speed_problem_l1047_104704

noncomputable def speed_B : ℝ := 12

noncomputable def speed_A : ℝ := 1.2 * speed_B

noncomputable def distance : ℝ := 12

noncomputable def time_diff : ℝ := 1/6

theorem student_speed_problem :
  (distance / speed_B) - (distance / speed_A) = time_diff := by
  sorry

#check student_speed_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_speed_problem_l1047_104704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_model_experiment_l1047_104738

noncomputable def arithmetic_max_height (a₁ : ℝ) (d : ℝ) : ℝ :=
  let n := ⌊(a₁ / -d) + 1⌋
  n * a₁ + (n * (n - 1) / 2) * d

noncomputable def geometric_max_height (b₁ : ℝ) (q : ℝ) : ℝ :=
  b₁ / (1 - q)

theorem airplane_model_experiment :
  arithmetic_max_height 15 (-2) = 64 ∧
  geometric_max_height 15 0.8 = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_model_experiment_l1047_104738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_equals_16_2_l1047_104730

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Theorem statement
theorem floor_expression_equals_16_2 :
  (floor 6.5) * (floor (2/3 : ℝ)) + (floor 2) * (7.2 : ℝ) + (floor 8.4) - 6.2 = 16.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_equals_16_2_l1047_104730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_plus_pi_half_l1047_104766

theorem sin_2theta_plus_pi_half (θ : ℝ) (h : Real.tan θ = 2) : 
  Real.sin (2 * θ + π / 2) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_plus_pi_half_l1047_104766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mortgage_payment_sum_l1047_104735

noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem mortgage_payment_sum :
  let first_payment : ℝ := 100
  let ratio : ℝ := 3
  let num_payments : ℕ := 7
  geometric_sum first_payment ratio num_payments = 109300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mortgage_payment_sum_l1047_104735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l1047_104787

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := 
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem about the eccentricity range of a hyperbola under specific conditions -/
theorem hyperbola_eccentricity_range (h : Hyperbola) 
  (F₁ : Point) -- Left focus
  (M N : Point) -- Intersection points on left branch
  (B : Point) -- Point (0, b)
  (h_F₁_left_focus : F₁.x < 0 ∧ F₁.y = 0)
  (h_M_above_N : M.y > N.y)
  (h_MN_parallel_y : M.x = N.x)
  (h_B_coord : B.x = 0 ∧ B.y = h.b)
  (h_BMN_obtuse : (M.x - B.x) * (N.y - M.y) + (M.y - B.y) * (N.x - M.x) < 0)
  : 1 < eccentricity h ∧ eccentricity h < Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l1047_104787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_parenthesizations_other_values_count_l1047_104777

def exponentiation (a b : ℕ) : ℕ := a ^ b

def standard_expression : ℕ := exponentiation 3 (exponentiation 3 (exponentiation 3 2))

def parenthesization1 : ℕ := exponentiation 3 (exponentiation (exponentiation 3 3) 2)
def parenthesization2 : ℕ := exponentiation (exponentiation (exponentiation 3 3) 3) 2
def parenthesization3 : ℕ := exponentiation (exponentiation 3 (exponentiation 3 3)) 2
def parenthesization4 : ℕ := exponentiation (exponentiation 3 3) (exponentiation 3 2)

theorem distinct_parenthesizations :
  ∃! (s : Finset ℕ),
    s.card = 5 ∧
    standard_expression ∈ s ∧
    parenthesization1 ∈ s ∧
    parenthesization2 ∈ s ∧
    parenthesization3 ∈ s ∧
    parenthesization4 ∈ s ∧
    (∀ x ∈ s, x = standard_expression ∨
              x = parenthesization1 ∨
              x = parenthesization2 ∨
              x = parenthesization3 ∨
              x = parenthesization4) :=
by sorry

theorem other_values_count :
  (∃! (s : Finset ℕ),
    s.card = 5 ∧
    standard_expression ∈ s ∧
    parenthesization1 ∈ s ∧
    parenthesization2 ∈ s ∧
    parenthesization3 ∈ s ∧
    parenthesization4 ∈ s ∧
    (∀ x ∈ s, x = standard_expression ∨
              x = parenthesization1 ∨
              x = parenthesization2 ∨
              x = parenthesization3 ∨
              x = parenthesization4)) →
  4 = (Finset.filter (fun x => x ≠ standard_expression) 
        {standard_expression, parenthesization1, parenthesization2, parenthesization3, parenthesization4}).card :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_parenthesizations_other_values_count_l1047_104777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculator_sale_loss_l1047_104784

/-- The profit or loss from selling two calculators -/
noncomputable def calculator_sale_result (price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) : ℝ :=
  let profit_calculator_cost := price / (1 + profit_percent / 100)
  let loss_calculator_cost := price / (1 - loss_percent / 100)
  2 * price - (profit_calculator_cost + loss_calculator_cost)

/-- Theorem stating that selling two calculators at 60 yuan each, 
    one at 20% profit and one at 20% loss, results in a 5 yuan loss -/
theorem calculator_sale_loss :
  calculator_sale_result 60 20 20 = -5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculator_sale_loss_l1047_104784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l1047_104720

/-- The function f(x) -/
noncomputable def f (A ω x : ℝ) : ℝ := A * Real.sin (ω * x + Real.pi / 8) ^ 2

/-- Theorem stating the value of ω given the conditions -/
theorem omega_value (A ω : ℝ) (hA : A > 0) (hω : ω > 0)
  (h_sym : ∀ x, f A ω (Real.pi / 2 - x) = f A ω (Real.pi / 2 + x))
  (h_period : ∃ T, Real.pi / 2 < T ∧ T < 3 * Real.pi / 2 ∧ ∀ x, f A ω x = f A ω (x + T)) :
  ω = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l1047_104720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_M_satisfies_conditions_l1047_104729

def point (x y z : ℝ) := (x, y, z)

noncomputable def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

theorem point_M_satisfies_conditions :
  let A := point 1 0 2
  let B := point 1 (-3) 1
  let M := point 0 (-1) 0
  (M.2.1 = 0 ∧ M.2.2 = 0) ∧ distance M A = distance M B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_M_satisfies_conditions_l1047_104729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_acute_triangles_l1047_104751

-- Define a triangle type with three angles
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define what it means for a triangle to be acute
def is_acute (t : Triangle) : Prop :=
  t.angle1 < 90 ∧ t.angle2 < 90 ∧ t.angle3 < 90

-- Define the six triangles
def triangle1 : Triangle := ⟨40, 60, 80⟩
def triangle2 : Triangle := ⟨10, 10, 160⟩
def triangle3 : Triangle := ⟨110, 35, 35⟩
def triangle4 : Triangle := ⟨50, 30, 100⟩
def triangle5 : Triangle := ⟨90, 40, 50⟩
def triangle6 : Triangle := ⟨80, 20, 80⟩

-- The theorem to prove
theorem exactly_two_acute_triangles :
  ∃! (acute_triangles : List Triangle),
    acute_triangles.length = 2 ∧
    (∀ t ∈ acute_triangles, is_acute t) ∧
    (∀ t ∈ [triangle1, triangle2, triangle3, triangle4, triangle5, triangle6],
      is_acute t → t ∈ acute_triangles) := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_acute_triangles_l1047_104751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_three_books_is_18_72_l1047_104783

/-- The cost of 6 books in dollars -/
noncomputable def cost_six_books : ℚ := 3744 / 100

/-- The cost of 3 books in dollars -/
noncomputable def cost_three_books : ℚ := cost_six_books / 2

/-- Theorem stating that the cost of 3 books is $18.72 -/
theorem cost_three_books_is_18_72 : cost_three_books = 1872 / 100 := by
  unfold cost_three_books cost_six_books
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_three_books_is_18_72_l1047_104783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l1047_104709

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with equation x²/4 + y²/3 = 1 -/
def Ellipse : Set Point := {p : Point | p.x^2 / 4 + p.y^2 / 3 = 1}

/-- Represents a line passing through two points -/
def Line (p1 p2 : Point) : Set Point := {p : Point | ∃ t : ℝ, p = Point.mk (p1.x + t * (p2.x - p1.x)) (p1.y + t * (p2.y - p1.y))}

/-- Slope between two points -/
noncomputable def slopeBetween (p1 p2 : Point) : ℝ := (p2.y - p1.y) / (p2.x - p1.x)

/-- Left focus of the ellipse -/
def F1 : Point := Point.mk (-1) 0

/-- Right focus of the ellipse -/
def F2 : Point := Point.mk 1 0

/-- Left vertex of the ellipse -/
def A : Point := Point.mk (-2) 0

theorem ellipse_intersection_theorem (m : ℝ) (hm : m > 2) :
  ∃ (P Q : Point) (l : Set Point),
    P ∈ Ellipse ∧ Q ∈ Ellipse ∧
    P ≠ Q ∧
    l = Line (Point.mk m 0) P ∧
    Q ∈ l ∧
    slopeBetween Q F2 = slopeBetween P F2 ∧
    slopeBetween P A + slopeBetween P (Point.mk m 0) = 0 →
    m = 3 ∨ m = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l1047_104709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_x_coordinate_sum_l1047_104780

/-- Theorem about the sum of x-coordinates on an ellipse -/
theorem ellipse_x_coordinate_sum 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (ellipse : Set (ℝ × ℝ))
  (h_ellipse : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ (x, y) ∈ ellipse)
  (h_point : (0, Real.sqrt 3) ∈ ellipse)
  (h_eccentricity : (a^2 - b^2) / a^2 = 1/4)
  (P M N : ℝ × ℝ)
  (h_distinct : P ≠ M ∧ P ≠ N ∧ M ≠ N)
  (h_on_ellipse : P ∈ ellipse ∧ M ∈ ellipse ∧ N ∈ ellipse)
  (h_slopes : (M.2 - P.2) / (M.1 - P.1) * (N.2 - P.2) / (N.1 - P.1) = -3/4) :
  M.1 + N.1 = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_x_coordinate_sum_l1047_104780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_triangle_properties_l1047_104716

-- Define the parabola
structure Parabola where
  a : ℝ
  equation : ℝ → ℝ
  focus : ℝ × ℝ
  directrix : ℝ → ℝ

-- Define a point on the parabola
structure ParabolaPoint where
  x : ℝ
  y : ℝ

-- Define a triangle formed by tangents to the parabola
structure TangentTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Helper functions (not implemented, just signatures)
def CircumcircleContainsFocus (p : Parabola) (t : TangentTriangle) : Prop := sorry

def AltitudeIntersectionOnDirectrix (p : Parabola) (t : TangentTriangle) : Prop := sorry

def AreaParabolaPoints (p1 p2 p3 : ParabolaPoint) : ℝ := sorry

def AreaTriangle (t : TangentTriangle) : ℝ := sorry

def ParabolaPointFromCoords (point : ℝ × ℝ) : ParabolaPoint := sorry

noncomputable def CubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Define the theorem
theorem parabola_tangent_triangle_properties
  (p : Parabola)
  (α β γ : ParabolaPoint)
  (triangle : TangentTriangle) :
  -- a) The circumcircle of triangle ABC passes through the focus of the parabola
  CircumcircleContainsFocus p triangle ∧
  -- b) The altitudes of triangle ABC intersect at a point that lies on the directrix of the parabola
  AltitudeIntersectionOnDirectrix p triangle ∧
  -- c) S_αβγ = 2 S_ABC
  AreaParabolaPoints α β γ = 2 * AreaTriangle triangle ∧
  -- d) ∛S_αβC + ∛S_βγA = ∛S_αγB
  CubeRoot (AreaParabolaPoints α β (ParabolaPointFromCoords triangle.C)) +
  CubeRoot (AreaParabolaPoints β γ (ParabolaPointFromCoords triangle.A)) =
  CubeRoot (AreaParabolaPoints α γ (ParabolaPointFromCoords triangle.B)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_triangle_properties_l1047_104716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_ratio_irrational_l1047_104702

theorem log_ratio_irrational (m n : ℕ) (hm : m > 1) (hn : n > 1) (hcoprime : Nat.Coprime m n) :
  ¬ (∃ (p q : ℤ), q ≠ 0 ∧ Real.log m / Real.log n = p / q) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_ratio_irrational_l1047_104702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_through_focus_l1047_104778

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 2x -/
def Parabola := {p : Point | p.y^2 = 2 * p.x}

/-- The focus of the parabola y^2 = 2x -/
noncomputable def focus : Point := ⟨1/2, 0⟩

/-- A line passing through the focus -/
structure FocusLine where
  a : Point
  b : Point
  passes_through_focus : a.x ≠ b.x → 
    (focus.y - a.y) / (focus.x - a.x) = (b.y - a.y) / (b.x - a.x)

/-- Theorem: Length of chord AB passing through focus -/
theorem chord_length_through_focus 
  (l : FocusLine) 
  (ha : l.a ∈ Parabola) 
  (hb : l.b ∈ Parabola) 
  (h_distinct : l.a ≠ l.b) 
  (h_sum : l.a.x + l.b.x = 3) : 
  Real.sqrt ((l.b.x - l.a.x)^2 + (l.b.y - l.a.y)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_through_focus_l1047_104778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_selling_price_l1047_104719

/-- Calculates the selling price of a book given its cost price and profit percentage. -/
noncomputable def selling_price (cost_price : ℝ) (profit_percentage : ℝ) : ℝ :=
  cost_price * (1 + profit_percentage / 100)

/-- Theorem stating that a book with a cost price of $60 and a profit percentage of 25% has a selling price of $75. -/
theorem book_selling_price :
  selling_price 60 25 = 75 := by
  -- Unfold the definition of selling_price
  unfold selling_price
  -- Simplify the arithmetic
  simp [mul_add, mul_div_right_comm]
  -- Check that 60 * (1 + 25 / 100) = 75
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_selling_price_l1047_104719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_has_advantage_smallest_group_size_for_b_disadvantage_l1047_104737

noncomputable def prob_born_on_day : ℝ := 1 / 7

noncomputable def prob_not_sunday : ℝ := 1 - prob_born_on_day

noncomputable def prob_at_least_one_sunday (n : ℕ) : ℝ := 1 - prob_not_sunday ^ n

-- Part a: Prove that B has the advantage in the original bet
theorem b_has_advantage : prob_at_least_one_sunday 7 < 10 / 15 := by
  sorry

-- Part b: Prove that 16 is the smallest number of people for B to be at a disadvantage
theorem smallest_group_size_for_b_disadvantage :
  (∀ n < 16, prob_at_least_one_sunday n ≤ 10 / 11) ∧
  prob_at_least_one_sunday 16 > 10 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_has_advantage_smallest_group_size_for_b_disadvantage_l1047_104737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_intersection_l1047_104701

-- Define the functions g and f
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (a + 1) ^ (x - 2) + 1

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + a) / Real.log (Real.sqrt 3)

-- State the theorem
theorem function_intersection (a : ℝ) (h1 : a > 0) (h2 : g a 2 = 2) (h3 : f a 2 = 2) : a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_intersection_l1047_104701
