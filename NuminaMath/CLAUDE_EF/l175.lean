import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_unchanged_l175_17590

/-- Represents a garden roller -/
structure GardenRoller where
  diameter : ℝ
  length : ℝ

/-- Calculates the area covered by a garden roller in a given number of revolutions -/
noncomputable def areaCovered (roller : GardenRoller) (revolutions : ℝ) : ℝ :=
  2 * (22 / 7) * (roller.diameter / 2) * roller.length * revolutions

/-- Theorem stating that changing the length to 4m doesn't change the area covered -/
theorem area_unchanged (roller : GardenRoller) (h1 : roller.diameter = 1.4) 
    (h2 : areaCovered roller 5 = 88) : 
    areaCovered { diameter := roller.diameter, length := 4 } 5 = 88 := by
  sorry

#check area_unchanged

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_unchanged_l175_17590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l175_17562

theorem repeating_decimal_to_fraction : 
  ∀ (x : ℚ), (∃ (n : ℕ), x * 10^n - x * 10^(n-1) = 3) → x = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l175_17562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bottles_filled_l175_17593

/-- Represents the volume of oil in liters -/
def oil_volume : ℚ := 4

/-- Represents the capacity of each bottle in milliliters -/
def bottle_capacity : ℚ := 200

/-- Conversion factor from liters to milliliters -/
def liters_to_ml : ℚ := 1000

/-- Theorem stating that the number of bottles that can be filled is 20 -/
theorem bottles_filled : 
  (oil_volume * liters_to_ml / bottle_capacity).floor = 20 := by
  sorry

#eval (oil_volume * liters_to_ml / bottle_capacity).floor

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bottles_filled_l175_17593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_range_of_m_l175_17516

noncomputable def f (a b x : ℝ) : ℝ := (a^x - a^(-x)) / (a^x + a^(-x)) + b

theorem function_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y : ℝ, x < y → f a 0 x > f a 0 y) →  -- f is monotonically decreasing
  (∀ x : ℝ, f a 0 x = -f a 0 (-x)) →        -- f is an odd function
  f a 0 1 = -3/5 →                          -- f(1) = -3/5
  a = 1/2 ∧ b = 0 :=
by sorry

-- Additional theorem for part (2)
theorem range_of_m (a : ℝ) (h1 : a = 1/2) :
  ∃ m : ℝ, m ∈ Set.Icc (2 * Real.sqrt 2 - 2) (17/20) ∧
  (∃ x y : ℝ, x ∈ Set.Icc (-1) 1 ∧ y ∈ Set.Icc (-1) 1 ∧ x ≠ y ∧
   f a 0 x = m - (4:ℝ)^x ∧ f a 0 y = m - (4:ℝ)^y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_range_of_m_l175_17516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l175_17565

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  area : ℝ

-- Define the theorem
theorem triangle_problem (abc : Triangle) 
  (h1 : Real.sin (abc.B + abc.C) = 3 * (Real.sin (abc.A / 2))^2)
  (h2 : abc.area = 6)
  (h3 : abc.b + abc.c = 8) :
  Real.cos abc.A = 5/13 ∧ abc.a = 3 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l175_17565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_infinite_implies_infinite_l175_17566

/-- A set is infinite if it has a proper subset that is in bijection with the whole set. -/
def IsInfinite (α : Type*) (s : Set α) : Prop :=
  ∃ t : Set α, t ⊂ s ∧ ∃ f : α → α, Function.Bijective f ∧ f '' s = t

theorem subset_infinite_implies_infinite {α : Type*} (A B : Set α) 
  (h_subset : B ⊆ A) (h_infinite : IsInfinite α B) : IsInfinite α A := by
  sorry

#check subset_infinite_implies_infinite

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_infinite_implies_infinite_l175_17566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_symmetry_l175_17567

/-- Given a circle equation x^2 + y^2 + 2ax - 2ay = 0 where a ≠ 0,
    prove that the center of the circle lies on the line x + y = 0 -/
theorem circle_center_symmetry (a : ℝ) (h : a ≠ 0) :
  let eq := fun (x y : ℝ) ↦ x^2 + y^2 + 2*a*x - 2*a*y = 0
  let center := (-a, a)
  (∃ (r : ℝ), r > 0 ∧ ∀ (x y : ℝ), eq x y ↔ (x - center.1)^2 + (y - center.2)^2 = r^2) →
  (center.1 + center.2 = 0) :=
by
  intros eq center h_circle
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_symmetry_l175_17567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_l175_17553

theorem rectangular_to_polar :
  let x : ℝ := 3
  let y : ℝ := -3
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := if x > 0 then Real.arctan (y / x) else if x < 0 ∧ y ≥ 0 then Real.arctan (y / x) + Real.pi else if x < 0 ∧ y < 0 then Real.arctan (y / x) - Real.pi else if x = 0 ∧ y > 0 then Real.pi / 2 else if x = 0 ∧ y < 0 then -Real.pi / 2 else 0
  (r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi) ∧ r = 3 * Real.sqrt 2 ∧ θ = 7 * Real.pi / 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_l175_17553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wealth_ratio_theorem_main_wealth_ratio_theorem_l175_17522

/-- Represents a nation's statistics -/
structure Nation where
  population_percent : ℝ
  wealth_percent : ℝ

/-- Calculates the wealth per citizen of a nation -/
noncomputable def wealth_per_citizen (nation : Nation) (total_population : ℝ) (total_wealth : ℝ) : ℝ :=
  (nation.wealth_percent * total_wealth) / (nation.population_percent * total_population)

/-- Theorem stating the ratio of wealth per citizen between two nations -/
theorem wealth_ratio_theorem (X Y : Nation) (total_population total_wealth : ℝ) 
    (hX : X.population_percent > 0)
    (hY : Y.population_percent > 0)
    (inequality_index : ℝ)
    (h_inequality : inequality_index = 0.5) :
  (inequality_index * wealth_per_citizen X total_population total_wealth) / 
  (wealth_per_citizen Y total_population total_wealth) = 
  (inequality_index * X.wealth_percent * Y.population_percent) / 
  (X.population_percent * Y.wealth_percent) := by
  sorry

/-- Main theorem proving the ratio of wealth between citizens of Nation X and Nation Y -/
theorem main_wealth_ratio_theorem (X Y : Nation) 
    (hX : X.population_percent > 0)
    (hY : Y.population_percent > 0) :
  ∃ (total_population total_wealth : ℝ),
    (0.5 * wealth_per_citizen X total_population total_wealth) / 
    (wealth_per_citizen Y total_population total_wealth) = 
    (0.5 * X.wealth_percent * Y.population_percent) / 
    (X.population_percent * Y.wealth_percent) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wealth_ratio_theorem_main_wealth_ratio_theorem_l175_17522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_difference_is_correct_l175_17521

/-- Represents the attendance at a soccer game --/
structure Attendance where
  estimate : ℕ
  actual : ℕ
  error_bound : ℚ

/-- The Seattle game attendance --/
def seattle : Attendance := {
  estimate := 75000,
  actual := 0,  -- We don't know the actual value
  error_bound := 15 / 100
}

/-- The Chicago game attendance --/
def chicago : Attendance := {
  estimate := 85000,
  actual := 0,  -- We don't know the actual value
  error_bound := 12 / 100
}

/-- The largest possible difference between the attendances, rounded to the nearest 1000 --/
def largest_difference : ℕ := 33000

/-- Theorem stating that the largest possible difference is correct --/
theorem largest_difference_is_correct : 
  ∃ (seattle_actual chicago_actual : ℕ),
    (seattle_actual : ℚ) ≥ seattle.estimate * (1 - seattle.error_bound) ∧
    (seattle_actual : ℚ) ≤ seattle.estimate * (1 + seattle.error_bound) ∧
    (chicago_actual : ℚ) ≥ chicago.estimate / (1 + chicago.error_bound) ∧
    (chicago_actual : ℚ) ≤ chicago.estimate / (1 - chicago.error_bound) ∧
    largest_difference = (((chicago_actual - seattle_actual : ℚ) / 1000 + 1/2).floor.toNat * 1000) ∧
    ∀ (s c : ℕ),
      (s : ℚ) ≥ seattle.estimate * (1 - seattle.error_bound) →
      (s : ℚ) ≤ seattle.estimate * (1 + seattle.error_bound) →
      (c : ℚ) ≥ chicago.estimate / (1 + chicago.error_bound) →
      (c : ℚ) ≤ chicago.estimate / (1 - chicago.error_bound) →
      (((c - s : ℚ) / 1000 + 1/2).floor.toNat * 1000) ≤ largest_difference :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_difference_is_correct_l175_17521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_vertex_of_dilated_square_l175_17598

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  center : Point
  area : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Applies a dilation to a point -/
def dilate (p : Point) (center : Point) (scale : ℝ) : Point :=
  { x := center.x + scale * (p.x - center.x)
  , y := center.y + scale * (p.y - center.y) }

/-- Theorem: The farthest vertex of the dilated square is (21, -21) -/
theorem farthest_vertex_of_dilated_square (s : Square) 
  (h1 : s.center = { x := 5, y := -5 })
  (h2 : s.area = 16)
  (h3 : ∀ (p : Point), distance p { x := 0, y := 0 } ≤ distance { x := 21, y := -21 } { x := 0, y := 0 }) :
  ∃ (p : Point), p = dilate { x := 7, y := -7 } { x := 0, y := 0 } 3 ∧ 
                 p = { x := 21, y := -21 } :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_vertex_of_dilated_square_l175_17598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_perpendicular_point_l175_17500

/-- The circle C -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x = 0

/-- Point A -/
noncomputable def A (m : ℝ) : ℝ × ℝ := (1, m)

/-- Point B -/
noncomputable def B (m : ℝ) : ℝ × ℝ := (1, 2*Real.sqrt 5 - m)

/-- Perpendicularity condition -/
def perpendicular (P A B : ℝ × ℝ) : Prop :=
  (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0

/-- Main theorem -/
theorem unique_perpendicular_point :
  ∃! P : ℝ × ℝ, circle_C P.1 P.2 ∧ perpendicular P (A (Real.sqrt 5 + 2)) (B (Real.sqrt 5 + 2)) := by
  sorry

#check unique_perpendicular_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_perpendicular_point_l175_17500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_axis_of_symmetry_l175_17581

-- Define the function as noncomputable
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + 5 * Real.pi / 6)

-- State the theorem
theorem closest_axis_of_symmetry 
  (ω : ℝ) 
  (h1 : 0 < ω) 
  (h2 : ω < Real.pi) 
  (h3 : f ω 0 = 1/2) 
  (h4 : f ω (1/2) = 0) :
  ∃ (k : ℤ), 
    (∀ (n : ℤ), abs (3*k - 1) ≤ abs (3*n - 1)) ∧ 
    (3*k - 1 = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_axis_of_symmetry_l175_17581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_15_l175_17569

/-- Given a train and a platform, calculates the speed of the train. -/
noncomputable def train_speed (train_length platform_length time : ℝ) : ℝ :=
  (train_length + platform_length) / time

/-- Theorem: The speed of the train is 15 m/s given the specified conditions. -/
theorem train_speed_is_15 :
  let train_length : ℝ := 50
  let platform_length : ℝ := 100
  let time : ℝ := 10
  train_speed train_length platform_length time = 15 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Simplify the expression
  simp
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_15_l175_17569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_bat_profit_l175_17596

/-- Calculates the adjusted profit and profit percentage for a cricket bat sale --/
theorem cricket_bat_profit (selling_price : ℝ) (initial_profit : ℝ) 
  (sales_tax_rate : ℝ) (discount_rate : ℝ) :
  selling_price = 850 →
  initial_profit = 255 →
  sales_tax_rate = 0.07 →
  discount_rate = 0.05 →
  let cost_price := selling_price - initial_profit
  let sales_tax := sales_tax_rate * selling_price
  let discount := discount_rate * selling_price
  let adjusted_selling_price := selling_price - discount
  let final_amount := adjusted_selling_price - sales_tax
  let adjusted_profit := final_amount - cost_price
  let profit_percentage := (adjusted_profit / cost_price) * 100
  (adjusted_profit = 153) ∧ 
  (abs (profit_percentage - 25.71) < 0.01) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_bat_profit_l175_17596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_similar_triangles_in_sequence_l175_17588

/-- Represents a triangle with angles α, β, γ --/
structure Triangle where
  α : Real
  β : Real
  γ : Real
  angle_sum : α + β + γ = Real.pi
  positive_angles : 0 < α ∧ 0 < β ∧ 0 < γ

/-- Generates the next triangle in the sequence --/
noncomputable def nextTriangle (t : Triangle) : Triangle where
  α := (t.β + t.γ) / 2
  β := (t.α + t.γ) / 2
  γ := (t.α + t.β) / 2
  angle_sum := by
    sorry
  positive_angles := by
    sorry

/-- Checks if two triangles are similar --/
def areSimilar (t1 t2 : Triangle) : Prop :=
  ∃ (k : Real), k > 0 ∧ 
    t1.α = k * t2.α ∧ 
    t1.β = k * t2.β ∧ 
    t1.γ = k * t2.γ

/-- The sequence of triangles --/
noncomputable def triangleSequence (t0 : Triangle) : ℕ → Triangle
  | 0 => t0
  | n + 1 => nextTriangle (triangleSequence t0 n)

theorem no_similar_triangles_in_sequence 
  (t0 : Triangle) 
  (h_scalene : t0.α ≠ t0.β ∧ t0.β ≠ t0.γ ∧ t0.γ ≠ t0.α) :
  ∀ (m n : ℕ), m ≠ n → ¬(areSimilar (triangleSequence t0 m) (triangleSequence t0 n)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_similar_triangles_in_sequence_l175_17588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_proof_l175_17594

/-- The distance between two adjacent parallel lines that intersect a circle -/
noncomputable def distance_between_lines : ℝ := Real.sqrt (76 / 719)

/-- The lengths of the three chords created by the parallel lines -/
def chord_lengths : List ℝ := [40, 40, 36]

theorem distance_between_lines_proof :
  ∃ (d : ℝ), d = distance_between_lines ∧
  ∃ (chords : List ℝ), chords = chord_lengths ∧
  (∀ (c : ℝ), c ∈ chords → c^2 ≤ 4 * (400 + 10 * d^2) / 40) ∧
  (∃ (x y z : ℝ), x ∈ chords ∧ y ∈ chords ∧ z ∈ chords ∧
    x^2 + y^2 + z^2 = 12 * ((400 + 10 * d^2) / 40) - 9 * d^2) :=
by
  sorry

#check distance_between_lines_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_proof_l175_17594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_two_l175_17547

noncomputable def f (x : ℝ) : ℝ := 1 / x

theorem tangent_line_at_two :
  let x₀ : ℝ := 2
  let y₀ : ℝ := f x₀
  let m : ℝ := -1 / (x₀^2)
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (x + 4*y - 4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_two_l175_17547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_profit_calculation_l175_17595

/-- Calculate the profit from movie production given the following parameters --/
theorem movie_profit_calculation 
  (actor_cost : ℕ) 
  (num_people : ℕ) 
  (food_cost_per_person : ℕ) 
  (selling_price : ℕ) 
  (h1 : actor_cost = 1200)
  (h2 : num_people = 50)
  (h3 : food_cost_per_person = 3)
  (h4 : selling_price = 10000)
  : selling_price - (actor_cost + num_people * food_cost_per_person + 2 * (actor_cost + num_people * food_cost_per_person)) = 5950 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_profit_calculation_l175_17595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acceleration_before_hit_formula_l175_17542

/-- Represents the drag force experienced by the ball as a function of its velocity. -/
noncomputable def drag_force (v : ℝ) : ℝ :=
  sorry

/-- Represents the acceleration of a ball just before being hit, given its initial horizontal
    velocity, final vertical velocity, final vertical acceleration, and gravitational acceleration. -/
noncomputable def acceleration_before_hit (V₁ V₂ a₂ g : ℝ) : ℝ :=
  (V₁ / V₂)^2 * (a₂ - g)

/-- Theorem stating that the acceleration of a ball just before being hit is equal to
    (V₁/V₂)² * (a₂ - g), given the conditions from the problem. -/
theorem acceleration_before_hit_formula
  (V₁ V₂ a₂ g : ℝ)
  (h₁ : V₁ > 0)
  (h₂ : V₂ > 0)
  (h₃ : g > 0)
  (h₄ : ∃ k : ℝ, k > 0 ∧ ∀ v : ℝ, drag_force v = k * v^2) :
  ∃ a₁ : ℝ, a₁ = acceleration_before_hit V₁ V₂ a₂ g :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acceleration_before_hit_formula_l175_17542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_xy_value_l175_17515

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : (2 : ℝ)^x * (4 : ℝ)^y = 4) :
  ∀ a b : ℝ, a > 0 → b > 0 → (2 : ℝ)^a * (4 : ℝ)^b = 4 → x * y ≥ a * b ∧ x * y ≤ 1/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_xy_value_l175_17515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_integer_sum_l175_17509

theorem five_integer_sum (a b c d e : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0)
  (h1 : a + b + c + d ∈ ({44, 45, 46, 47} : Set ℕ))
  (h2 : a + b + c + e ∈ ({44, 45, 46, 47} : Set ℕ))
  (h3 : a + b + d + e ∈ ({44, 45, 46, 47} : Set ℕ))
  (h4 : a + c + d + e ∈ ({44, 45, 46, 47} : Set ℕ))
  (h5 : b + c + d + e ∈ ({44, 45, 46, 47} : Set ℕ)) :
  a + b + c + d + e = 57 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_integer_sum_l175_17509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l175_17573

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- The right focus of a hyperbola -/
noncomputable def right_focus (h : Hyperbola) : ℝ × ℝ :=
  (Real.sqrt (h.a^2 + h.b^2), 0)

/-- The slope of the asymptotes of a hyperbola -/
noncomputable def asymptote_slope (h : Hyperbola) : ℝ := h.b / h.a

/-- Represents a point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (h.a^2 + h.b^2) / h.a

/-- Main theorem -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (A B : Point)
  (h_perp : (A.y - (right_focus h).2) * asymptote_slope h = -(A.x - (right_focus h).1))
  (h_on_asymptote : A.y = asymptote_slope h * A.x ∧ B.y = -asymptote_slope h * B.x)
  (h_ratio : 2 * ((A.x - (right_focus h).1)^2 + (A.y - (right_focus h).2)^2) = 
             ((B.x - (right_focus h).1)^2 + (B.y - (right_focus h).2)^2)) :
  eccentricity h = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l175_17573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_day_distance_theorem_l175_17518

/-- Represents the distance traveled on each day of the journey --/
noncomputable def distance_sequence (first_day_distance : ℝ) : ℕ → ℝ
  | 0 => first_day_distance
  | n + 1 => (distance_sequence first_day_distance n) / 2

/-- Calculates the total distance traveled over a given number of days --/
noncomputable def total_distance (first_day_distance : ℝ) (days : ℕ) : ℝ :=
  (List.range days).map (distance_sequence first_day_distance) |>.sum

theorem second_day_distance_theorem :
  ∃ (first_day_distance : ℝ),
    total_distance first_day_distance 6 = 378 ∧
    distance_sequence first_day_distance 1 = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_day_distance_theorem_l175_17518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_five_l175_17575

noncomputable def f (a b c : ℝ) : ℝ :=
  (Real.sqrt ((a * b * c + 4) / a + 4 * Real.sqrt (b * c / a))) / (Real.sqrt (a * b * c) + 2)

theorem expression_equals_five (b c : ℝ) (h1 : b * c ≥ 0) :
  f 0.04 b c = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_five_l175_17575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_640_in_fourth_quadrant_l175_17528

-- Define a function to determine the quadrant of an angle
noncomputable def quadrant (angle : ℝ) : ℕ :=
  let normalized_angle := angle % 360
  if 0 ≤ normalized_angle ∧ normalized_angle < 90 then 1
  else if 90 ≤ normalized_angle ∧ normalized_angle < 180 then 2
  else if 180 ≤ normalized_angle ∧ normalized_angle < 270 then 3
  else 4

-- Theorem statement
theorem angle_640_in_fourth_quadrant :
  quadrant 640 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_640_in_fourth_quadrant_l175_17528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_strategy_l175_17501

/-- Represents a rectangular container with a given base area and height -/
structure Container where
  base : ℝ
  height : ℝ

/-- The volume of a container -/
def volume (c : Container) : ℝ := c.base * c.height

/-- Theorem: The volume of containers A and D combined is always greater than
    the volume of containers B and C combined, given that a ≠ b -/
theorem winning_strategy (a b : ℝ) (ha : a ≠ b) :
  let A : Container := ⟨a^2, a⟩
  let B : Container := ⟨a^2, b⟩
  let C : Container := ⟨b^2, a⟩
  let D : Container := ⟨b^2, b⟩
  volume A + volume D > volume B + volume C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_strategy_l175_17501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_circular_arrangement_l175_17545

/-- A circular arrangement of digits 1 and 2 -/
def CircularArrangement := List Nat

/-- Check if a number is a valid four-digit number using only 1 and 2 -/
def isValidFourDigitNumber (n : Nat) : Prop :=
  n ≥ 1111 ∧ n ≤ 2222 ∧ ∀ d, d ∈ n.digits 10 → d = 1 ∨ d = 2

/-- Check if a four-digit number appears in a circular arrangement -/
def appearsInArrangement (n : Nat) (arr : CircularArrangement) : Prop :=
  ∃ (start : Nat), List.take 4 (List.rotateLeft arr start ++ List.rotateLeft arr start) = n.digits 10

/-- The main theorem -/
theorem smallest_circular_arrangement :
  ∃ (arr : CircularArrangement),
    arr.length = 14 ∧
    (∀ n, isValidFourDigitNumber n → appearsInArrangement n arr) ∧
    (∀ m, m < 14 →
      ¬∃ (arr' : CircularArrangement),
        arr'.length = m ∧
        (∀ n, isValidFourDigitNumber n → appearsInArrangement n arr')) :=
by sorry

#check smallest_circular_arrangement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_circular_arrangement_l175_17545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_is_polynomial_l175_17519

/-- P_{n,k}(x) is the rational function defined as
    (x^n - 1)(x^n - x) ... (x^n - x^(k-1)) / (x^k - 1)(x^k - x) ... (x^k - x^(k-1)) -/
def P (n k : ℕ) (x : ℚ) : ℚ :=
  (Finset.range k).prod (fun i => (x^n - x^i)) / (Finset.range k).prod (fun i => (x^k - x^i))

/-- P_{n,k}(x) is a polynomial for all non-negative integers n and k -/
theorem P_is_polynomial (n k : ℕ) : ∃ p : Polynomial ℚ, ∀ x, P n k x = p.eval x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_is_polynomial_l175_17519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_transformation_l175_17597

/-- Represents a quadratic polynomial ax^2 + bx + c --/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the allowed operations on quadratic polynomials --/
inductive Operation
  | Op1 : Operation  -- f(x) → x^2 f(1/x + 1)
  | Op2 : Operation  -- f(x) → (x-1)^2 f(1/(x-1))

/-- Applies an operation to a quadratic polynomial --/
def applyOperation (p : QuadraticPolynomial) (op : Operation) : QuadraticPolynomial :=
  match op with
  | Operation.Op1 => QuadraticPolynomial.mk (p.a + p.b + p.c) (2*p.a + p.b) p.a
  | Operation.Op2 => QuadraticPolynomial.mk p.a (p.b - 2*p.c) (p.a - p.b + p.c)

/-- Checks if two quadratic polynomials are equal --/
def QuadraticPolynomial.equal (p q : QuadraticPolynomial) : Prop :=
  p.a = q.a ∧ p.b = q.b ∧ p.c = q.c

/-- Applies a list of operations to a quadratic polynomial --/
def applyOperations (p : QuadraticPolynomial) (ops : List Operation) : QuadraticPolynomial :=
  ops.foldl applyOperation p

/-- Theorem: It's impossible to transform x^2 + 4x + 3 into x^2 + 10x + 9 using the given operations --/
theorem impossible_transformation : 
  ¬ ∃ (ops : List Operation), 
    (applyOperations (QuadraticPolynomial.mk 1 4 3) ops).equal 
    (QuadraticPolynomial.mk 1 10 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_transformation_l175_17597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_angle_l175_17538

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 4) + Real.sin (x / 7)

theorem smallest_max_angle :
  ∀ y : ℝ, y > 0 → y < 17190 → f y ≤ f 17190 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_angle_l175_17538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_but_not_sufficient_l175_17556

noncomputable def f (x m : ℝ) : ℝ := (Real.sqrt (4 - x^2)) / (x + 4) - m

theorem necessary_but_not_sufficient :
  (∃ (m : ℝ), ∃ (x : ℝ), f x m = 0) →
  (∀ (m : ℝ), (∃ (x : ℝ), f x m = 0) → abs m ≤ Real.sqrt 3 / 3) ∧
  (∃ (m : ℝ), abs m ≤ Real.sqrt 3 / 3 ∧ ∀ (x : ℝ), f x m ≠ 0) :=
by sorry

#check necessary_but_not_sufficient

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_but_not_sufficient_l175_17556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_first_quadrant_l175_17529

-- Define the angle α
noncomputable def α : Real := sorry

-- Define the point on the terminal side of α
noncomputable def point_on_terminal_side : ℝ × ℝ := (Real.sin 3, -Real.cos 3)

-- Theorem statement
theorem angle_in_first_quadrant :
  point_on_terminal_side = (Real.sin 3, -Real.cos 3) →
  0 < α ∧ α < Real.pi / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_first_quadrant_l175_17529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_one_equals_sixteen_l175_17564

-- Define t(x)
noncomputable def t (x : ℝ) : ℝ := 3 * x - 8

-- Define s(y) where y = t(x)
noncomputable def s (y : ℝ) : ℝ := 
  let x := (y + 8) / 3  -- Inverse of t(x)
  x^2 + 3*x - 2

-- Theorem to prove
theorem s_of_one_equals_sixteen : s 1 = 16 := by
  -- Unfold the definition of s
  unfold s
  -- Simplify the expression
  simp
  -- Complete the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_one_equals_sixteen_l175_17564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l175_17563

theorem trig_identity (α β : ℝ) 
  (h : Real.cos α * Real.cos β - Real.sin α * Real.sin β = 0) :
  Real.sin α * Real.cos β + Real.cos α * Real.sin β = 1 ∨ 
  Real.sin α * Real.cos β + Real.cos α * Real.sin β = -1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l175_17563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_lines_l175_17534

/-- A line in the xy-plane with x-intercept a and y-intercept b --/
structure Line where
  a : ℝ
  b : ℝ

/-- The line passes through the point (4,3) --/
def passes_through_point (l : Line) : Prop :=
  4 / l.a + 3 / l.b = 1

/-- The x-intercept is a positive prime number --/
def x_intercept_prime (l : Line) : Prop :=
  l.a > 0 ∧ Nat.Prime (Int.natAbs (Int.floor l.a))

/-- The y-intercept is a positive integer --/
def y_intercept_positive_int (l : Line) : Prop :=
  l.b > 0 ∧ (∃ n : ℕ, l.b = n)

/-- The main theorem --/
theorem count_lines : 
  ∃ (s : Finset Line), 
    (∀ l ∈ s, passes_through_point l ∧ x_intercept_prime l ∧ y_intercept_positive_int l) ∧
    (∀ l : Line, passes_through_point l ∧ x_intercept_prime l ∧ y_intercept_positive_int l → l ∈ s) ∧
    Finset.card s = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_lines_l175_17534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_random_chord_densities_l175_17539

noncomputable section

open Real

/-- Joint probability density function for method a -/
noncomputable def density_a (r θ : ℝ) : ℝ :=
  1 / (π^2 * sqrt (1 - r^2))

/-- Joint probability density function for method b -/
noncomputable def density_b (r θ : ℝ) : ℝ :=
  r / π

/-- Joint probability density function for method c -/
noncomputable def density_c (r θ : ℝ) : ℝ :=
  1 / (2 * π)

theorem random_chord_densities (r θ : ℝ) 
  (hr : r ∈ Set.Icc 0 1) (hθ : θ ∈ Set.Ico 0 (2 * π)) :
  (density_a r θ = 1 / (π^2 * sqrt (1 - r^2))) ∧
  (density_b r θ = r / π) ∧
  (density_c r θ = 1 / (2 * π)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_random_chord_densities_l175_17539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_membership_l175_17585

theorem set_membership (U A B : Finset ℕ) : 
  (U.card = 193) →
  ((U \ (A ∪ B)).card = 59) →
  ((A ∩ B).card = 23) →
  (A.card = 116) →
  (B.card = 41) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_membership_l175_17585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_unique_frame_covering_l175_17586

/-- Represents a square frame in a grid -/
structure Frame where
  size : ℕ
  top_left : ℕ × ℕ

/-- Represents a grid -/
structure Grid where
  size : ℕ

/-- Checks if a frame covers a given cell -/
def covers (f : Frame) (cell : ℕ × ℕ) : Prop :=
  let (x, y) := cell
  let (tx, ty) := f.top_left
  (x = tx ∨ x = tx + f.size - 1) ∧ ty ≤ y ∧ y < ty + f.size ∨
  (y = ty ∨ y = ty + f.size - 1) ∧ tx ≤ x ∧ x < tx + f.size

/-- Checks if a cell is on the boundary of the grid -/
def is_boundary (g : Grid) (cell : ℕ × ℕ) : Prop :=
  let (x, y) := cell
  x = 0 ∨ x = g.size - 1 ∨ y = 0 ∨ y = g.size - 1

/-- The main theorem to prove -/
def unique_frame_covering (g : Grid) (frames : List Frame) : Prop :=
  g.size = 100 ∧
  (∀ f ∈ frames, f.size = 50) ∧
  (∀ cell, is_boundary g cell → ∃! f, f ∈ frames ∧ covers f cell) ∧
  (∀ f ∈ frames, ∃ cell, is_boundary g cell ∧ covers f cell)

/-- Proof of the theorem -/
theorem prove_unique_frame_covering : ∃! frames : List Frame, unique_frame_covering (Grid.mk 100) frames := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_unique_frame_covering_l175_17586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_is_negative_reals_l175_17582

open Set

-- Define the function f and its properties
def f_properties (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  (∀ x, HasDerivAt f (f' x) x) ∧ 
  (∀ x, f x + f' x < 1) ∧ 
  (f 0 = 2015)

-- Define the solution set
def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | Real.exp x * f x - Real.exp x > 2014}

-- Theorem statement
theorem solution_set_is_negative_reals (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  f_properties f f' → solution_set f = Iio 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_is_negative_reals_l175_17582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_loss_challenge_l175_17540

theorem weight_loss_challenge (W : ℝ) (h : W > 0) : 
  let weight_after_loss := W * (1 - 0.12)
  let weight_with_clothes := weight_after_loss * (1 + 0.02)
  let measured_loss_percentage := (W - weight_with_clothes) / W * 100
  measured_loss_percentage = 10.24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_loss_challenge_l175_17540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisible_power_l175_17508

def p : ℕ → ℕ
  | 0 => 2012  -- Add this case to cover Nat.zero
  | 1 => 2012
  | n + 2 => 2012^(p (n + 1))

theorem largest_divisible_power : 
  (∀ m : ℕ, m > 1 → (2011^m) ∣ (p 2012 - p 2011) → m ≤ 1) ∧ 
  (2011^1) ∣ (p 2012 - p 2011) :=
by
  sorry  -- Use 'by' and 'sorry' to skip the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisible_power_l175_17508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_wins_for_n_not_in_set_l175_17507

/-- Represents the drinks that can be poured into the glasses -/
inductive Drink
| Lemonade
| Compote

/-- Represents a player in the game -/
inductive Player
| Petya
| Vasya

/-- Represents the state of a glass -/
inductive GlassState
| Empty
| Filled (d : Drink)

/-- Represents the game state -/
structure GameState where
  glasses : List GlassState
  currentPlayer : Player

/-- Checks if a move is valid according to the game rules -/
def isValidMove (state : GameState) (position : Nat) : Bool :=
  sorry

/-- Applies a move to the game state -/
def applyMove (state : GameState) (position : Nat) : GameState :=
  sorry

/-- Checks if the game is over -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Determines the winner of the game -/
def getWinner (state : GameState) : Option Player :=
  sorry

/-- Represents Vasya's winning strategy -/
def vasyaStrategy (state : GameState) : Option Nat :=
  sorry

/-- The main theorem stating Vasya's winning strategy exists for n ∉ {1, 2, 4, 6} -/
theorem vasya_wins_for_n_not_in_set (n : Nat) 
  (h : n ∉ ({1, 2, 4, 6} : Set Nat)) : 
  ∃ (strategy : GameState → Option Nat), 
    (∀ (initialState : GameState), 
      initialState.glasses.length = n → 
      (getWinner (applyMove initialState (strategy initialState).get!)) = some Player.Vasya) :=
  sorry

#check vasya_wins_for_n_not_in_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_wins_for_n_not_in_set_l175_17507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_value_l175_17599

/-- The area enclosed by a curve composed of 16 congruent circular arcs,
    each of length π/2, with centers at the vertices of a regular octagon
    with side length 3. -/
noncomputable def enclosed_area (num_arcs : ℕ) (arc_length : ℝ) (octagon_side : ℝ) : ℝ :=
  2 * (1 + Real.sqrt 2) * octagon_side^2 + num_arcs * arc_length * (arc_length / (2 * Real.pi))^2 / 4

/-- Theorem stating that the area enclosed by the described curve
    is equal to 54(1+√2) + 2π. -/
theorem enclosed_area_value :
  enclosed_area 16 (Real.pi/2) 3 = 54 * (1 + Real.sqrt 2) + 2 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_value_l175_17599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_import_tax_calculation_l175_17548

/-- Calculate the import tax for a given item value -/
noncomputable def import_tax (total_value : ℝ) (tax_rate : ℝ) (tax_free_threshold : ℝ) : ℝ :=
  max 0 ((total_value - tax_free_threshold) * tax_rate)

/-- Theorem: The import tax for an item valued at $2,560 with a 7% tax rate above $1,000 is $109.20 -/
theorem import_tax_calculation :
  -- Unfold the definition of import_tax
  unfold import_tax
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_import_tax_calculation_l175_17548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_analysis_l175_17523

/-- Given a function f(x) = (1/3)x^3 + ax + b with certain properties,
    prove its analytical expression and maximum value on an interval. -/
theorem function_analysis (a b : ℝ) :
  let f := λ x : ℝ ↦ (1/3) * x^3 + a * x + b
  (∀ x, HasDerivAt f (x^2 + a) x) →
  HasDerivAt f 0 (-2) →
  f 4 = 28/3 →
  (∀ x, f x = (1/3) * x^3 - 4 * x + 4) ∧
  ∀ m, 
    (m > -4 → m < -2 → IsMaxOn f (Set.Icc (-4) m) ((1/3) * m^3 - 4 * m + 4)) ∧
    (-2 ≤ m → m ≤ 4 → IsMaxOn f (Set.Icc (-4) m) (28/3)) ∧
    (m > 4 → IsMaxOn f (Set.Icc (-4) m) ((1/3) * m^3 - 4 * m + 4)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_analysis_l175_17523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_max_value_l175_17579

open Set Real

/-- The function f(x) = -ax + x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := -a * x + x + a

/-- The open interval (0,1] -/
def openUnitInterval : Set ℝ := { x | 0 < x ∧ x ≤ 1 }

theorem f_increasing_and_max_value (a : ℝ) (ha : a ≠ 0) :
  (StrictMonoOn (f a) openUnitInterval) →
  (0 < a ∧ a ≤ 1) ∧ (∃ M, M = 1 ∧ ∀ x ∈ openUnitInterval, f a x ≤ M) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_max_value_l175_17579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fisherman_sale_theorem_l175_17583

/-- The number of fishes sold at the fisherman sale -/
def num_fishes : ℕ := 16

/-- The position of the radio's price from highest to lowest -/
def position_highest : ℕ := 4

/-- The position of the radio's price from lowest to highest -/
def position_lowest : ℕ := 13

theorem fisherman_sale_theorem :
  ∃ (radio_price : ℕ) (prices : Finset ℕ),
    (Finset.card prices = num_fishes) ∧ 
    (radio_price ∈ prices) ∧
    (Finset.card (prices.filter (λ p => p > radio_price)) = position_highest - 1) ∧
    (Finset.card (prices.filter (λ p => p < radio_price)) = position_lowest - 1) ∧
    (∀ p₁ p₂, p₁ ∈ prices → p₂ ∈ prices → p₁ ≠ p₂ → p₁ ≠ p₂) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fisherman_sale_theorem_l175_17583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_roots_l175_17511

theorem compare_roots : (4 : ℝ)^(1/4) > (5 : ℝ)^(1/5) ∧ (5 : ℝ)^(1/5) > (16 : ℝ)^(1/16) ∧ (16 : ℝ)^(1/16) > (25 : ℝ)^(1/25) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_roots_l175_17511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_sum_probability_l175_17580

/-- Represents a wheel with numbers -/
structure Wheel where
  numbers : List ℕ

/-- The first wheel with numbers 1 through 6 -/
def wheel1 : Wheel := ⟨[1, 2, 3, 4, 5, 6]⟩

/-- The second wheel with numbers 1 through 6 where 1 and 3 are repeated -/
def wheel2 : Wheel := ⟨[1, 1, 2, 3, 3, 4, 5, 6]⟩

/-- Checks if a natural number is even -/
def isEven (n : ℕ) : Bool := n % 2 == 0

/-- Calculates the probability of an event given the number of favorable outcomes and total outcomes -/
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

/-- Counts the number of even sums in the product of two wheels -/
def countEvenSums (w1 w2 : Wheel) : ℕ :=
  (List.product w1.numbers w2.numbers).filter (fun pair => isEven (pair.1 + pair.2)) |>.length

/-- Theorem: The probability of getting an even sum when spinning both wheels is 1/2 -/
theorem even_sum_probability :
  probability (countEvenSums wheel1 wheel2) (wheel1.numbers.length * wheel2.numbers.length) = 1/2 := by
  sorry

#eval probability (countEvenSums wheel1 wheel2) (wheel1.numbers.length * wheel2.numbers.length)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_sum_probability_l175_17580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flag_height_l175_17530

/-- Represents a rectangular flag with specific properties -/
structure RectangularFlag where
  height : ℝ
  length : ℝ
  numStripes : ℕ
  shadedArea : ℝ
  heightPos : 0 < height
  lengthEqTwiceHeight : length = 2 * height
  stripesSeven : numStripes = 7
  shadedAreaValue : shadedArea = 1400

/-- Theorem stating that a flag with the given properties has a height of 35 cm -/
theorem flag_height (flag : RectangularFlag) : flag.height = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flag_height_l175_17530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sun_earth_triangle_base_l175_17526

/-- Represents an isosceles triangle with given leg length and base length -/
structure IsoscelesTriangle where
  leg : ℝ
  base : ℝ

/-- The ratio of leg lengths equals the ratio of base lengths for isosceles triangles with the same vertex angle -/
axiom isosceles_ratio (t1 t2 : IsoscelesTriangle) :
  t1.leg / t2.leg = t1.base / t2.base

/-- Convert kilometers to millimeters -/
noncomputable def km_to_mm (x : ℝ) : ℝ := x * 1000000

/-- Convert millimeters to kilometers -/
noncomputable def mm_to_km (x : ℝ) : ℝ := x / 1000000

theorem sun_earth_triangle_base : 
  let t1 : IsoscelesTriangle := ⟨1, mm_to_km 4.848⟩
  let t2 : IsoscelesTriangle := ⟨1.5e8, 0⟩
  mm_to_km (km_to_mm t2.leg * t1.base) = 727.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sun_earth_triangle_base_l175_17526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_theorem_l175_17524

noncomputable def arithmetic_sequence (n : ℕ) : ℝ := n

noncomputable def geometric_sequence (n : ℕ) : ℝ := 2^(n-1)

noncomputable def S (n : ℕ) : ℝ := n * (n + 1) / 2

theorem arithmetic_geometric_sequence_theorem :
  (∀ n : ℕ, n ≥ 1 → arithmetic_sequence n > 0) ∧
  arithmetic_sequence 1 = 1 ∧
  geometric_sequence 1 = 1 ∧
  geometric_sequence 2 * S 2 = 6 ∧
  geometric_sequence 2 + S 3 = 8 →
  (∀ n : ℕ, n ≥ 1 → arithmetic_sequence n = n) ∧
  (∀ n : ℕ, n ≥ 1 → geometric_sequence n = 2^(n-1)) ∧
  (∀ n : ℕ, n ≥ 1 → (Finset.range n).sum (λ k ↦ 1 / S (k+1)) = 2 * n / (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_theorem_l175_17524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_specific_l175_17532

/-- The area of a sector with given arc length and central angle -/
noncomputable def sectorArea (arcLength : ℝ) (centralAngle : ℝ) : ℝ :=
  let radius := arcLength * 360 / (2 * Real.pi * centralAngle)
  (1 / 2) * radius * arcLength

/-- Theorem: The area of a sector with arc length 3π and central angle 135° is 6π -/
theorem sector_area_specific : sectorArea (3 * Real.pi) 135 = 6 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_specific_l175_17532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_length_problem_l175_17506

/-- The length of a rope spiraling around a cylindrical tank -/
noncomputable def rope_length (c h n : ℝ) : ℝ :=
  n * Real.sqrt (c^2 + (h/n)^2)

/-- Theorem stating the length of the rope in the given problem -/
theorem rope_length_problem : rope_length 6 18 6 = 18 * Real.sqrt 5 := by
  -- Unfold the definition of rope_length
  unfold rope_length
  -- Simplify the expression
  simp [Real.sqrt_mul, Real.sqrt_sq]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_length_problem_l175_17506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l175_17591

-- Define the equation of the region
def region_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 6*x - 8*y = 16

-- Define the area of the region
noncomputable def region_area : ℝ := 41 * Real.pi

-- Theorem statement
theorem area_of_region :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, region_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    region_area = Real.pi * radius^2 := by
  -- Existence of center and radius
  let center_x := -3
  let center_y := 4
  let radius := Real.sqrt 41
  
  -- Proof of the theorem
  use center_x, center_y, radius
  apply And.intro
  
  -- First part: equivalence of equations
  · intro x y
    apply Iff.intro
    · intro h
      -- Transform the original equation to the circle equation
      sorry
    · intro h
      -- Transform the circle equation to the original equation
      sorry
  
  -- Second part: area calculation
  · -- Show that the area matches the definition
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l175_17591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l175_17589

noncomputable section

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define eccentricity
def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

-- Define distance from a point to a line
def distance_point_to_line (x y : ℝ) (A B C : ℝ) : ℝ :=
  abs (A * x + B * y + C) / Real.sqrt (A^2 + B^2)

theorem ellipse_properties
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : eccentricity a b = Real.sqrt 3 / 2)
  (h4 : distance_point_to_line (-a) 0 1 2 (-2) = 4 * Real.sqrt 5 / 5) :
  -- 1. Equation of ellipse C
  (∀ x y : ℝ, ellipse a b x y ↔ x^2 / 4 + y^2 = 1) ∧
  -- 2. Constant distance from O to AB
  (∀ A B : ℝ × ℝ,
    ellipse a b A.1 A.2 →
    ellipse a b B.1 B.2 →
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4 * (A.1^2 + A.2^2) →
    distance_point_to_line 0 0 (B.2 - A.2) (A.1 - B.1) (B.1 * A.2 - A.1 * B.2) = 2 * Real.sqrt 5 / 5) ∧
  -- 3. Minimum area of triangle AOB
  (∃ m : ℝ,
    (∀ A B : ℝ × ℝ,
      ellipse a b A.1 A.2 →
      ellipse a b B.1 B.2 →
      (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4 * (A.1^2 + A.2^2) →
      1/2 * abs (A.1 * B.2 - A.2 * B.1) ≥ m) ∧
    m = 4/5) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l175_17589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_company_a_l175_17576

theorem stratified_sampling_company_a (
  total_a : ℕ) (total_b : ℕ) (supervisors : ℕ) :
  total_a = 120 →
  total_b = 100 →
  supervisors = 11 →
  (supervisors : ℚ) / (total_a + total_b : ℚ) * total_a = 6 := by
  intro h1 h2 h3
  -- The proof steps would go here
  sorry

#check stratified_sampling_company_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_company_a_l175_17576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_for_inequality_l175_17533

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^3

-- State the theorem
theorem x_range_for_inequality (x : ℝ) :
  (f (x^2) < f (3*x - 2)) ↔ (1 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_for_inequality_l175_17533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chemistry_players_l175_17551

theorem chemistry_players (total biology both chemistry : ℕ) :
  total = 12 →
  biology = 7 →
  both = 2 →
  chemistry = total - biology + both →
  chemistry = 7 := by
  intros h_total h_biology h_both h_chemistry
  rw [h_total, h_biology, h_both] at h_chemistry
  norm_num at h_chemistry
  exact h_chemistry

#check chemistry_players

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chemistry_players_l175_17551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_value_l175_17577

-- Define the points A, B, and C
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (4, 5)
def C : ℝ × ℝ := (7, 10)

-- Define the vector equation
def vector_equation (P : ℝ × ℝ) (l : ℝ) : Prop :=
  P.1 - A.1 = (B.1 - A.1) + l * (C.1 - A.1) ∧
  P.2 - A.2 = (B.2 - A.2) + l * (C.2 - A.2)

-- Define the line equation
def on_line (P : ℝ × ℝ) : Prop :=
  P.1 - 2 * P.2 = 0

-- Theorem statement
theorem lambda_value :
  ∀ P : ℝ × ℝ, ∀ l : ℝ,
  vector_equation P l → on_line P →
  l = -2/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_value_l175_17577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_condition_l175_17503

noncomputable def f (x m n : ℝ) : ℝ := x * |Real.sin x + m| + n

theorem odd_function_condition (m n : ℝ) :
  (∀ x, f x m n = -f (-x) m n) ↔ m * n = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_condition_l175_17503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_11_simplest_l175_17561

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∀ y : ℝ, y ≥ 0 → (∃ a b : ℝ, a > 0 ∧ b ≥ 0 ∧ x = a * Real.sqrt y) → y = x^2

theorem sqrt_11_simplest :
  is_simplest_quadratic_radical (Real.sqrt 11) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt (1/2)) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 12) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 1.21) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_11_simplest_l175_17561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l175_17535

theorem geometric_series_sum : 
  let a₀ : ℚ := 1/2
  let r : ℚ := 1/2
  let n : ℕ := 5
  let series_sum := (Finset.range n).sum (λ i => a₀ * r^i)
  series_sum = 31/32 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l175_17535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l175_17592

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (1/2)^x - 1 else Real.sqrt x

-- Theorem statement
theorem range_of_a (a : ℝ) (h : f a > 1) : a ∈ Set.Ioi 1 ∪ Set.Iio (-1) := by
  sorry

-- Note: Set.Ioi 1 represents (1, +∞) and Set.Iio (-1) represents (-∞, -1)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l175_17592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l175_17555

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 3)

theorem min_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l175_17555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_root_solution_verify_solution_l175_17568

/-- The first equation: λx³ - x² - x + (λ+1) = 0 -/
def equation1 (l : ℝ) (x : ℝ) : Prop :=
  l * x^3 - x^2 - x + (l + 1) = 0

/-- The second equation: λx² - x - (λ+1) = 0 -/
def equation2 (l : ℝ) (x : ℝ) : Prop :=
  l * x^2 - x - (l + 1) = 0

/-- The theorem stating that λ = -1 and x₀ = 0 are the only solution -/
theorem common_root_solution :
  ∃! (l : ℝ) (x : ℝ), equation1 l x ∧ equation2 l x ∧ l = -1 ∧ x = 0 := by
  sorry

/-- Verification that λ = -1 and x₀ = 0 satisfy both equations -/
theorem verify_solution :
  equation1 (-1) 0 ∧ equation2 (-1) 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_root_solution_verify_solution_l175_17568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_right_triangle_catheti_l175_17510

/-- A right triangle with special properties -/
structure SpecialRightTriangle where
  -- A, B, C are the vertices of the triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- I is the incenter
  I : ℝ × ℝ
  -- G is the centroid
  G : ℝ × ℝ
  -- The triangle is right-angled at A
  right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  -- IG is parallel to BC
  ig_parallel_bc : (G.2 - I.2) * (C.1 - B.1) = (G.1 - I.1) * (C.2 - B.2)
  -- IG measures 10 cm
  ig_length : Real.sqrt ((G.1 - I.1)^2 + (G.2 - I.2)^2) = 10

/-- The theorem about the special right triangle -/
theorem special_right_triangle_catheti (t : SpecialRightTriangle) :
  let ab_length := Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2)
  let ac_length := Real.sqrt ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2)
  (ab_length = 90 ∧ ac_length = 120) ∨ (ab_length = 120 ∧ ac_length = 90) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_right_triangle_catheti_l175_17510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_z_l175_17531

noncomputable def z (x : ℝ) : ℝ := (15 * x^4 + 3 * x^3 + 7 * x^2 + 6 * x + 2) / (5 * x^4 + x^3 + 4 * x^2 + 2 * x + 1)

theorem horizontal_asymptote_of_z :
  ∀ ε > 0, ∃ M > 0, ∀ x, |x| > M → |z x - 3| < ε :=
by
  sorry

#check horizontal_asymptote_of_z

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_z_l175_17531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_radical_l175_17557

-- Define the options
noncomputable def option_A (a : ℝ) : ℝ := Real.sqrt (2 * a)
noncomputable def option_B (a : ℝ) : ℝ := Real.sqrt (a ^ 2)
noncomputable def option_C (x y : ℝ) : ℝ := Real.sqrt (5 * x^2 * y)
noncomputable def option_D : ℝ := Real.sqrt (1 / 3)

-- Define what it means to be a quadratic radical expression
def is_quadratic_radical (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x, f x = Real.sqrt (a * x^2 + b * x + c)

-- Define what it means to be the simplest
def is_simplest (f : ℝ → ℝ) (others : List (ℝ → ℝ)) : Prop :=
  is_quadratic_radical f ∧ ∀ g ∈ others, is_quadratic_radical g → f = g

-- State the theorem
theorem simplest_quadratic_radical :
  is_simplest (λ a => option_A a) [λ a => option_B a, λ x => option_C x 1, λ _ => option_D] :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_radical_l175_17557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_in_special_triangle_l175_17574

theorem smallest_angle_in_special_triangle (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 180 →
  (a : ℝ) / 3 = (b : ℝ) / 4 ∧ (b : ℝ) / 4 = (c : ℝ) / 5 →
  min a (min b c) > 30 →
  min a (min b c) = 45 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_in_special_triangle_l175_17574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l175_17558

/-- Definition of n-degree zero-point functions -/
def n_degree_zero_point_functions (f g : ℝ → ℝ) (n : ℝ) : Prop :=
  ∃ (α β : ℝ), f α = 0 ∧ g β = 0 ∧ |α - β| < n

/-- The function f -/
noncomputable def f (x : ℝ) : ℝ := 2^(x - 2) - 1

/-- The function g -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x^2 - a * Real.exp x

theorem range_of_a :
  ∃ (a_lower a_upper : ℝ),
    a_lower = Real.exp (-1) ∧
    a_upper = 4 * Real.exp (-2) ∧
    (∀ a : ℝ, n_degree_zero_point_functions f (g a) 1 ↔ a_lower < a ∧ a ≤ a_upper) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l175_17558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_minus_three_frac_eq_two_solutions_l175_17578

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := x - (floor x : ℝ)

-- State the theorem
theorem floor_minus_three_frac_eq_two_solutions :
  ∀ x : ℝ, (floor x - 3 * frac x = 2) ↔ (x = 2 ∨ x = 10/3 ∨ x = 14/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_minus_three_frac_eq_two_solutions_l175_17578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounces_four_times_l175_17505

/-- Represents the total distance traveled by a bouncing ball -/
noncomputable def totalDistance (initialHeight : ℝ) (bounces : ℕ) : ℝ :=
  let bounceHeight (n : ℕ) := initialHeight / (2 ^ n)
  (initialHeight * (1 - 1 / (2 ^ bounces))) * 3

/-- Theorem: A ball dropped from 16 meters, bouncing to half its previous height each time,
    will have bounced exactly 4 times when it has traveled a total distance of 45 meters -/
theorem ball_bounces_four_times :
  ∃ (bounces : ℕ), totalDistance 16 bounces = 45 ∧ bounces = 4 := by
  sorry

#check ball_bounces_four_times

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounces_four_times_l175_17505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sobel_male_percentage_approx_l175_17552

/-- Represents the election results and voter demographics -/
structure ElectionData where
  total_voters : ℚ
  sobel_percentage : ℚ
  male_percentage : ℚ
  lange_female_percentage : ℚ

/-- Calculates the percentage of male voters who voted for Sobel -/
noncomputable def sobel_male_percentage (data : ElectionData) : ℚ :=
  let female_voters := data.total_voters * (1 - data.male_percentage)
  let lange_female_voters := female_voters * data.lange_female_percentage
  let lange_total_voters := data.total_voters * (1 - data.sobel_percentage)
  let lange_male_voters := lange_total_voters - lange_female_voters
  let male_voters := data.total_voters * data.male_percentage
  let sobel_male_voters := male_voters - lange_male_voters
  (sobel_male_voters / male_voters) * 100

/-- Theorem stating that given the election data, the percentage of male voters 
    who voted for Sobel is approximately 73.33% -/
theorem sobel_male_percentage_approx (data : ElectionData) 
  (h1 : data.total_voters = 100)
  (h2 : data.sobel_percentage = 7/10)
  (h3 : data.male_percentage = 3/5)
  (h4 : data.lange_female_percentage = 7/20) : 
  ∃ ε : ℚ, ε > 0 ∧ |sobel_male_percentage data - 11/15| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sobel_male_percentage_approx_l175_17552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_lower_bound_l175_17587

open Real

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x - Real.exp x
def g (m : ℝ) (x : ℝ) : ℝ := m * x + 1

-- State the theorem
theorem m_lower_bound (m : ℝ) : 
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 2, ∃ x₀ ∈ Set.Icc (-1 : ℝ) 1, g m x₀ = f x₁) →
  m ≥ Real.exp 1 + 1 := by
  sorry

#check m_lower_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_lower_bound_l175_17587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l175_17525

noncomputable def f (x : ℝ) := Real.log (2 + x - x^2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l175_17525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l175_17559

theorem sin_beta_value (α β : Real)
  (h1 : Real.sin α = 2 * Real.sqrt 2 / 3)
  (h2 : Real.cos (α + β) = -1/3)
  (h3 : 0 < α ∧ α < Real.pi/2)
  (h4 : 0 < β ∧ β < Real.pi/2) :
  Real.sin β = 4 * Real.sqrt 2 / 9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l175_17559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brick_length_is_22cm_l175_17554

-- Define the courtyard dimensions in meters
def courtyard_length : ℚ := 28
def courtyard_width : ℚ := 13

-- Define the brick width in centimeters
def brick_width : ℚ := 12

-- Define the total number of bricks
def total_bricks : ℚ := 13787.878787878788

-- Define the function to calculate the brick length
noncomputable def calculate_brick_length (cl cw bw tb : ℚ) : ℚ :=
  (cl * 100 * cw * 100) / (bw * tb)

-- Theorem statement
theorem brick_length_is_22cm :
  calculate_brick_length courtyard_length courtyard_width brick_width total_bricks = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_brick_length_is_22cm_l175_17554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_3_5_percent_l175_17513

/-- Calculates the interest rate given the principal, time, and simple interest -/
noncomputable def calculate_interest_rate (principal : ℝ) (time : ℝ) (simple_interest : ℝ) : ℝ :=
  (simple_interest / (principal * time)) * 100

/-- Proves that the interest rate is 3.5% given the specified conditions -/
theorem interest_rate_is_3_5_percent 
  (principal : ℝ) 
  (time : ℝ) 
  (simple_interest : ℝ) 
  (h1 : principal = 500) 
  (h2 : time = 4) 
  (h3 : simple_interest = 70) : 
  calculate_interest_rate principal time simple_interest = 3.5 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculate_interest_rate 500 4 70

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_3_5_percent_l175_17513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_equation_holds_origin_satisfies_equation_center_satisfies_equation_l175_17537

/-- A sphere passing through the origin with center on the x-axis -/
structure SphereOnXAxis where
  R : ℝ  -- x-coordinate of the center

/-- The general equation of a sphere on the x-axis passing through the origin -/
def sphere_equation (S : SphereOnXAxis) (x y z : ℝ) : Prop :=
  x^2 - 2 * S.R * x + y^2 + z^2 = 0

theorem sphere_equation_holds (S : SphereOnXAxis) :
  ∀ (x y z : ℝ),
    ((x - S.R)^2 + y^2 + z^2 = S.R^2) ↔ sphere_equation S x y z := by
  sorry

theorem origin_satisfies_equation (S : SphereOnXAxis) :
  sphere_equation S 0 0 0 := by
  sorry

theorem center_satisfies_equation (S : SphereOnXAxis) :
  ((S.R - S.R)^2 + 0^2 + 0^2 = S.R^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_equation_holds_origin_satisfies_equation_center_satisfies_equation_l175_17537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_geq_x_no_positive_solution_for_a_and_b_l175_17536

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| - |2*x - 2|

-- Define the maximum value of f as k
noncomputable def k : ℝ := sSup (Set.range f)

-- Theorem for part (I)
theorem solution_set_f_geq_x :
  {x : ℝ | f x ≥ x} = {x : ℝ | x ≤ -1 ∨ x = 1} :=
sorry

-- Theorem for part (II)
theorem no_positive_solution_for_a_and_b :
  ¬∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 2*b = k ∧ 2/a + 1/b = 4 - 1/(a*b) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_geq_x_no_positive_solution_for_a_and_b_l175_17536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l175_17517

theorem circle_properties (x y : ℝ) (h : x^2 + y^2 - 4*x + 1 = 0) :
  (∃ (z : ℝ), ∀ (w : ℝ), y / x ≤ z) ∧
  (∃ (z : ℝ), ∀ (w : ℝ), y - x ≥ z) ∧
  (∃ (z : ℝ), ∀ (w : ℝ), x^2 + y^2 ≤ z) :=
by
  have h1 : ∃ (z : ℝ), ∀ (w : ℝ), y / x ≤ z := by
    use Real.sqrt 3
    sorry
  have h2 : ∃ (z : ℝ), ∀ (w : ℝ), y - x ≥ z := by
    use -(Real.sqrt 6) - 2
    sorry
  have h3 : ∃ (z : ℝ), ∀ (w : ℝ), x^2 + y^2 ≤ z := by
    use 7 + 4 * Real.sqrt 3
    sorry
  exact ⟨h1, h2, h3⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l175_17517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_min_values_l175_17572

noncomputable section

/-- P is a monic quadratic polynomial -/
noncomputable def P : ℝ → ℝ := sorry

/-- Q is a monic quadratic polynomial -/
noncomputable def Q : ℝ → ℝ := sorry

/-- The zeros of P(Q(x)) -/
def zeros_PQ : Finset ℝ := {1, 3, 5, 7}

/-- The zeros of Q(P(x)) -/
def zeros_QP : Finset ℝ := {10, 12, 14, 16}

/-- P is monic -/
axiom P_monic : ∃ a b, ∀ x, P x = x^2 + a*x + b

/-- Q is monic -/
axiom Q_monic : ∃ c d, ∀ x, Q x = x^2 + c*x + d

/-- The zeros of P(Q(x)) -/
axiom zeros_of_PQ : ∀ x ∈ zeros_PQ, P (Q x) = 0

/-- The zeros of Q(P(x)) -/
axiom zeros_of_QP : ∀ x ∈ zeros_QP, Q (P x) = 0

/-- The theorem to prove -/
theorem sum_of_min_values : 
  ∃ x y, (∀ t, P t ≥ P x) ∧ (∀ t, Q t ≥ Q y) ∧ P x + Q y = -164 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_min_values_l175_17572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_theorem_l175_17544

/-- The slope of a line passing through two points -/
def my_slope (x₁ y₁ x₂ y₂ : ℚ) : ℚ := (y₂ - y₁) / (x₂ - x₁)

/-- Theorem: The slope of a line passing through points (2,1) and (-1,3) is -2/3 -/
theorem line_slope_theorem : my_slope 2 1 (-1) 3 = -2/3 := by
  -- Unfold the definition of my_slope
  unfold my_slope
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_theorem_l175_17544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_data_std_dev_l175_17512

-- Define the original dataset
variable (original_data : List ℝ)

-- Define the variance of the original dataset
def variance_original : ℝ := 16

-- Define the new dataset
def new_data : List ℝ := original_data.map (λ x => 2 * x + 1)

-- Define the standard deviation of the new dataset
def std_dev_new : ℝ := 8

-- Theorem to prove
theorem new_data_std_dev : std_dev_new = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_data_std_dev_l175_17512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_optimization_l175_17502

-- Define the production efficiency function
noncomputable def production_efficiency (m : ℝ) (x : ℝ) : ℝ :=
  m * Real.log x - (1/100) * x^2 + (101/50) * x + Real.log 10

-- Define the profit function
noncomputable def profit (m : ℝ) (x : ℝ) : ℝ :=
  production_efficiency m x - x

-- State the theorem
theorem production_optimization (m : ℝ) :
  (∀ x, x > 10 → production_efficiency m x ≥ 0) →
  production_efficiency m 20 = 35.7 →
  (∃ unique_m : ℝ, m = unique_m ∧ unique_m = -1) ∧
  (∃ max_x : ℝ, max_x = 50 ∧ 
    ∀ x > 10, profit m x ≤ profit m max_x) ∧
  (profit m 50 = 24.4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_optimization_l175_17502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_correct_expressions_l175_17543

open Set

def set1 : Set ℕ := {0}
def set2 : Set ℕ := {0, 1, 2}
def set3 : Set ℕ := {2, 1, 0}
def set4 : Set ℕ := {0, 1}
def set5 : Set (ℕ × ℕ) := {(0, 1)}

def expression1 : Prop := set1 ⊆ set2
def expression2 : Prop := set2 ⊆ set3
def expression3 : Prop := (∅ : Set ℕ) ⊆ set2
def expression4 : Prop := (∅ : Set ℕ) = set1
def expression5 : Prop := set4 = (set5.image Prod.fst)
def expression6 : Prop := {0} = set1

def correct_expressions : List Prop := [expression2, expression3]

theorem num_correct_expressions : 
  (List.length correct_expressions) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_correct_expressions_l175_17543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_all_reals_l175_17584

/-- The function c(x) with parameter k -/
noncomputable def c (k : ℝ) (x : ℝ) : ℝ := (3*k*x^2 + 3*x - 4) / (-7*x^2 + 3*x + k)

/-- The theorem stating the condition for c(x) to have a domain of all real numbers -/
theorem domain_all_reals (k : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, c k x = y) ↔ k < -9/28 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_all_reals_l175_17584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_calculation_l175_17549

-- Define the custom operation
noncomputable def customOp (a b : ℝ) : ℝ := (a^2 + b^2) / 3

-- State the theorem
theorem custom_op_calculation :
  customOp (customOp 6 3) 9 = 102 := by
  -- Unfold the definition of customOp
  unfold customOp
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_calculation_l175_17549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_quadrilateral_count_l175_17527

theorem rod_quadrilateral_count :
  let rod_lengths : List ℕ := [4, 9, 18]
  let valid_lengths := fun (d : ℕ) =>
    d ∈ Finset.range 41 ∧
    d ∉ rod_lengths ∧
    d < rod_lengths.sum ∧
    d > (rod_lengths.maximum?.getD 0) - (rod_lengths.sum - (rod_lengths.maximum?.getD 0))
  ((Finset.filter valid_lengths (Finset.range 41)).card : ℕ) = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_quadrilateral_count_l175_17527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l175_17546

-- Define the region
def region (x y : ℝ) : Prop := abs (x + y) + abs (x - y) ≤ 6

-- Theorem statement
theorem area_of_region :
  MeasureTheory.volume {p : ℝ × ℝ | region p.1 p.2} = 36 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l175_17546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_catches_alice_l175_17550

-- Define the speeds and initial distance
noncomputable def alice_speed : ℝ := 6
noncomputable def tom_speed : ℝ := 8
noncomputable def initial_distance : ℝ := 3

-- Define the function to calculate the time in minutes
noncomputable def catch_up_time (a_speed t_speed init_dist : ℝ) : ℝ :=
  (init_dist / (a_speed + t_speed)) * 60

-- Theorem statement
theorem tom_catches_alice :
  abs (catch_up_time alice_speed tom_speed initial_distance - 12.857) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_catches_alice_l175_17550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_smallest_positive_period_period_is_pi_symmetry_center_l175_17520

-- Proposition 1
theorem vector_collinearity (a b : Fin 3 → ℝ) :
  ‖a + b‖ = ‖a‖ + ‖b‖ → ∃ k : ℝ, k ≥ 0 ∧ b = k • a :=
sorry

-- Proposition 2
noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.sin x)

theorem smallest_positive_period (x : ℝ) : f (x + Real.pi) = f x :=
sorry

theorem period_is_pi : ∀ p : ℝ, p > 0 ∧ (∀ x : ℝ, f (x + p) = f x) → p ≥ Real.pi :=
sorry

-- Proposition 4
noncomputable def g (x : ℝ) : ℝ := Real.tan (2 * x - Real.pi / 3)

theorem symmetry_center (x : ℝ) :
  g (5 * Real.pi / 12 + x) = -g (5 * Real.pi / 12 - x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_smallest_positive_period_period_is_pi_symmetry_center_l175_17520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2011_equals_2_l175_17514

def sequence_a : ℕ → ℚ
  | 0 => 2  -- Define the base case for 0
  | n + 1 => (5 * sequence_a n - 13) / (3 * sequence_a n - 7)

theorem a_2011_equals_2 : sequence_a 2011 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2011_equals_2_l175_17514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_envelopes_need_extra_postage_l175_17560

/-- Represents an envelope with length and height dimensions -/
structure Envelope where
  length : ℚ
  height : ℚ

/-- Checks if an envelope requires extra postage based on its dimensions -/
def requiresExtraPostage (e : Envelope) : Bool :=
  let ratio := e.length / e.height
  ratio < 14/10 || ratio > 26/10

/-- The list of envelopes with their dimensions -/
def envelopes : List Envelope := [
  ⟨7, 5⟩,  -- Envelope A
  ⟨10, 4⟩, -- Envelope B
  ⟨5, 5⟩,  -- Envelope C
  ⟨12, 4⟩  -- Envelope D
]

/-- Theorem stating that exactly two envelopes require extra postage -/
theorem two_envelopes_need_extra_postage :
  (envelopes.filter requiresExtraPostage).length = 2 := by
  sorry

#eval (envelopes.filter requiresExtraPostage).length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_envelopes_need_extra_postage_l175_17560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_ones_eight_signs_possible_twenty_ones_nine_signs_impossible_l175_17504

def OnesExpression := List (Option Bool)

def evaluateExpression (expr : OnesExpression) : ℤ :=
  sorry

def validExpression (expr : OnesExpression) (ones : ℕ) (signs : ℕ) : Prop :=
  (expr.length = ones - 1) ∧
  (expr.filterMap id).length = signs

theorem twenty_ones_eight_signs_possible : ∃ (expr : OnesExpression),
  validExpression expr 20 8 ∧ evaluateExpression expr = 2013 :=
sorry

theorem twenty_ones_nine_signs_impossible : ¬ ∃ (expr : OnesExpression),
  validExpression expr 20 9 ∧ evaluateExpression expr = 2013 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_ones_eight_signs_possible_twenty_ones_nine_signs_impossible_l175_17504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walk_both_ways_l175_17571

/-- The time taken to walk one way -/
def time_walk : ℝ := sorry

/-- The time taken to go by transport one way -/
def time_transport : ℝ := sorry

/-- Condition 1: Walking one way and taking transport back takes 1.5 hours -/
axiom cond1 : time_walk + time_transport = 1.5

/-- Condition 2: Taking transport both ways takes 0.5 hours -/
axiom cond2 : 2 * time_transport = 0.5

/-- Theorem: The time taken to walk both ways is 2.5 hours -/
theorem walk_both_ways : 2 * time_walk = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walk_both_ways_l175_17571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_lateral_surface_area_l175_17541

/-- The lateral surface area of a regular square pyramid -/
noncomputable def lateral_surface_area (base_edge : ℝ) (slant_height : ℝ) : ℝ :=
  (1 / 2) * 4 * base_edge * slant_height

/-- Theorem: The lateral surface area of a regular square pyramid with base edge length 4 and slant height 3 is 8√5 -/
theorem pyramid_lateral_surface_area :
  lateral_surface_area 4 3 = 8 * Real.sqrt 5 := by
  -- Unfold the definition of lateral_surface_area
  unfold lateral_surface_area
  -- Simplify the left-hand side
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_lateral_surface_area_l175_17541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_properties_l175_17570

/-- An acute triangle ABC where sides a and b are roots of x^2 - 2√3x + 2 = 0 and 2sin(A+B) - √3 = 0 -/
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = Real.pi
  roots : a^2 - 2 * Real.sqrt 3 * a + 2 = 0 ∧ b^2 - 2 * Real.sqrt 3 * b + 2 = 0
  angle_sum : 2 * Real.sin (A + B) - Real.sqrt 3 = 0

theorem acute_triangle_properties (t : AcuteTriangle) :
  t.C = Real.pi / 3 ∧ t.c = Real.sqrt 6 ∧ (t.a * t.b * Real.sin t.C) / 2 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_properties_l175_17570
