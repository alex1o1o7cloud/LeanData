import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_36_equals_4_11_l685_68562

/-- The decimal representation of a repeating decimal with digits 36 --/
noncomputable def repeating_decimal_36 : ℚ := 0.36363636

/-- Theorem stating that the repeating decimal 0.363636... is equal to 4/11 --/
theorem repeating_decimal_36_equals_4_11 : repeating_decimal_36 = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_36_equals_4_11_l685_68562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_surface_area_l685_68528

-- Define the radius of the hemisphere
noncomputable def r : ℝ := Real.sqrt (3 / Real.pi)

-- State the theorem
theorem hemisphere_surface_area : 
  (1/2 : ℝ) * (4 * Real.pi * r^2) + 3 = 9 := by
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_surface_area_l685_68528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt_equals_cosine_l685_68573

noncomputable def nested_sqrt (n : ℕ) : ℝ :=
  match n with
  | 0 => 0
  | n + 1 => Real.sqrt (2 + nested_sqrt n)

theorem nested_sqrt_equals_cosine (n : ℕ) :
  nested_sqrt n = 2 * Real.cos (π / (2 ^ (n + 1))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt_equals_cosine_l685_68573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l685_68560

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x) * Real.cos (ω * x) + 2 * Real.sqrt 3 * (Real.sin (ω * x))^2 - Real.sqrt 3

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x) + 1

theorem function_properties (ω : ℝ) (h_ω : ω > 0) :
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f ω (x + T) = f ω x ∧ ∀ S : ℝ, S > 0 ∧ S < T → ∃ y : ℝ, f ω (y + S) ≠ f ω y) →
  (∀ x : ℝ, f ω x = 2 * Real.sin (2 * x - π / 3)) ∧
  (∀ k : ℤ, ∀ x : ℝ, (k : ℝ) * π - π / 12 ≤ x ∧ x ≤ (k : ℝ) * π + 5 * π / 12 →
    ∀ y : ℝ, (k : ℝ) * π - π / 12 ≤ y ∧ y ≤ x → f ω y ≤ f ω x) ∧
  (∀ x : ℝ, g x = f ω (x + π / 6) + 1) ∧
  (∀ x : ℝ, -π / 12 ≤ x ∧ x ≤ π / 3 → g x ≤ 3) ∧
  (∀ x : ℝ, -π / 12 ≤ x ∧ x ≤ π / 3 → g x ≥ 0) ∧
  (∃ x : ℝ, -π / 12 ≤ x ∧ x ≤ π / 3 ∧ g x = 3) ∧
  (∃ x : ℝ, -π / 12 ≤ x ∧ x ≤ π / 3 ∧ g x = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l685_68560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_approx_equal_to_3_38_l685_68558

noncomputable def expression (x : ℕ) : ℝ := (x + 121 * 3.125) / 121

theorem approx_equal_to_3_38 :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |expression 31 - 3.38| < ε := by
  -- We'll use 0.005 as our ε
  use 0.005
  constructor
  · -- Prove ε > 0
    norm_num
  constructor
  · -- Prove ε < 0.01
    norm_num
  · -- Prove |expression 31 - 3.38| < ε
    unfold expression
    norm_num
    -- The following line would complete the proof, but we'll use sorry for now
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_approx_equal_to_3_38_l685_68558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_length_is_225_meters_l685_68567

-- Define the speeds of the trains in kmph
def slower_train_speed : ℚ := 36
def faster_train_speed : ℚ := 45

-- Define the time taken to pass in seconds
def passing_time : ℚ := 10

-- Convert kmph to m/s
noncomputable def kmph_to_ms (speed : ℚ) : ℚ := speed * (1000 / 3600)

-- Calculate the relative speed in m/s
noncomputable def relative_speed : ℚ := kmph_to_ms (slower_train_speed + faster_train_speed)

-- Calculate the length of the faster train
noncomputable def faster_train_length : ℚ := relative_speed * passing_time

theorem faster_train_length_is_225_meters :
  faster_train_length = 225 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_length_is_225_meters_l685_68567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_lambda_is_perfect_square_l685_68557

theorem floor_lambda_is_perfect_square 
  (l : ℝ) (n : ℕ) 
  (h1 : l ≥ 1) 
  (h2 : n > 0)
  (h3 : ∀ k ∈ Finset.range (3*n) \ Finset.range n, ∃ m : ℕ, ⌊l^(k+1)⌋ = m^2) :
  ∃ m : ℕ, ⌊l⌋ = m^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_lambda_is_perfect_square_l685_68557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_path_length_for_specific_cube_l685_68559

/-- Represents a cube with a dot on its top face -/
structure CubeWithDot where
  edgeLength : ℝ
  dotPosition : ℝ × ℝ

/-- Calculates the path length of the dot when the cube completes a full roll -/
noncomputable def pathLengthOfDot (cube : CubeWithDot) : ℝ :=
  4 * (Real.pi * Real.sqrt 2 / 2)

/-- Theorem stating the path length of the dot for a specific cube configuration -/
theorem dot_path_length_for_specific_cube :
  let cube : CubeWithDot := { edgeLength := 2, dotPosition := (1, 1) }
  pathLengthOfDot cube = 2 * Real.sqrt 2 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_path_length_for_specific_cube_l685_68559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_width_properties_l685_68582

/-- A curve of constant width -/
class ConstantWidthCurve (C : Type*) [TopologicalSpace C] [MetricSpace C] where
  width : ℝ
  is_constant_width : ∀ (p q : C), ∃ (r : C), dist r p = width ∧ dist r q ≤ width

/-- A supporting line to a curve -/
def SupportingLine (C : Type*) [TopologicalSpace C] (L : Set C) :=
  ∃ (p : C), p ∈ L ∧ ∀ (q : C), q ∉ interior L

/-- Theorem about properties of curves of constant width -/
theorem constant_width_properties
  (C : Type*) [TopologicalSpace C] [MetricSpace C] [ConstantWidthCurve C] :
  (∀ (L : Set C), SupportingLine C L → ∃! (p : C), p ∈ L) ∧
  (∀ (L₁ L₂ : Set C), SupportingLine C L₁ → SupportingLine C L₂ → L₁ ≠ L₂ →
    ∃ (A B : C), A ∈ L₁ ∧ B ∈ L₂ ∧ dist A B = ConstantWidthCurve.width C) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_width_properties_l685_68582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_extremum_range_l685_68536

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := a * (x - 2) * Real.exp x + Real.log x - x

-- State the theorem
theorem unique_extremum_range (a : ℝ) :
  (∃! x, ∀ y, f a x ≤ f a y) ∧  -- Unique extremum
  (∃ x, ∀ y, f a x ≤ f a y ∧ f a x < 0) →  -- Extremum is less than 0
  a ∈ Set.Ioc (-1 / Real.exp 1) 0 :=  -- a is in the open interval (-1/e, 0]
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_extremum_range_l685_68536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_3_equals_4_l685_68579

-- Define the domain of the functions
def Domain : Type := Fin 4

-- Define function f
def f : Domain → Domain
  | ⟨0, _⟩ => ⟨1, by norm_num⟩
  | ⟨1, _⟩ => ⟨3, by norm_num⟩
  | ⟨2, _⟩ => ⟨2, by norm_num⟩
  | ⟨3, _⟩ => ⟨0, by norm_num⟩

-- Define function g
def g : Domain → Domain
  | ⟨0, _⟩ => ⟨2, by norm_num⟩
  | ⟨1, _⟩ => ⟨0, by norm_num⟩
  | ⟨2, _⟩ => ⟨1, by norm_num⟩
  | ⟨3, _⟩ => ⟨3, by norm_num⟩

-- Theorem to prove
theorem f_g_3_equals_4 : f (g ⟨2, by norm_num⟩) = ⟨3, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_3_equals_4_l685_68579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_divisor_of_16_squared_plus_63_squared_l685_68569

theorem largest_prime_divisor_of_16_squared_plus_63_squared : 
  (Nat.factors (16^2 + 63^2)).maximum? = some 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_divisor_of_16_squared_plus_63_squared_l685_68569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_profit_is_138_l685_68538

/-- Represents the bakery's production and sales for one hour -/
structure BakeryHour where
  piece_price : ℝ  -- Price of one piece of pie
  pieces_per_pie : ℕ  -- Number of pieces per pie
  pies_per_hour : ℕ  -- Number of pies made per hour
  cost_per_pie : ℝ  -- Cost to make one pie

/-- Calculates the profit for one hour of bakery production -/
def calculate_profit (b : BakeryHour) : ℝ :=
  let revenue := b.piece_price * (b.pieces_per_pie : ℝ) * (b.pies_per_hour : ℝ)
  let cost := b.cost_per_pie * (b.pies_per_hour : ℝ)
  revenue - cost

/-- Theorem stating that the bakery's profit for one hour is $138 -/
theorem bakery_profit_is_138 :
  ∀ b : BakeryHour,
    b.piece_price = 4 →
    b.pieces_per_pie = 3 →
    b.pies_per_hour = 12 →
    b.cost_per_pie = 0.5 →
    calculate_profit b = 138 := by
  sorry

#eval calculate_profit { piece_price := 4, pieces_per_pie := 3, pies_per_hour := 12, cost_per_pie := 0.5 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_profit_is_138_l685_68538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_decreasing_l685_68543

noncomputable def f (x : ℝ) : ℝ := Real.exp (x^2 - 2*x - 3)

-- State the theorem
theorem f_strictly_decreasing :
  ∀ a b : ℝ, a < b ∧ b < 1 → f b < f a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_decreasing_l685_68543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_inside_c_outside_a_b_l685_68524

/-- Circle represented by its center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Area inside a circle -/
noncomputable def area_inside (c : Circle) : ℝ := Real.pi * c.radius^2

/-- Area of intersection between two circles -/
noncomputable def area_intersection (a b : Circle) : ℝ := sorry

/-- Given three circles A, B, and C with radius 1, where A and B are tangent,
    and C is tangent to the midpoint of the line segment connecting the centers of A and B,
    the area inside C but outside A and B is 2. -/
theorem area_inside_c_outside_a_b (a b c : Circle) : 
  a.radius = 1 →
  b.radius = 1 →
  c.radius = 1 →
  (a.center.1 - b.center.1)^2 + (a.center.2 - b.center.2)^2 = 4 →
  c.center = ((a.center.1 + b.center.1) / 2, (a.center.2 + b.center.2) / 2 + 1) →
  (area_inside c - area_inside a - area_inside b + area_intersection a b) = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_inside_c_outside_a_b_l685_68524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l685_68539

noncomputable def g (x : ℝ) : ℝ := (3^x - 1) / (3^x + 1)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  simp [g]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l685_68539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_k_l685_68520

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 5 * x^2 - 1/x + 3*x
noncomputable def g (x k : ℝ) : ℝ := x^2 - k

-- State the theorem
theorem solve_for_k : ∃ k : ℝ, f 3 - g 3 k = 10 ∧ k = -104/3 := by
  -- We use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_k_l685_68520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_shaltaevs_one_boltaev_cost_three_shaltaevs_one_boltaev_cost_a_three_shaltaevs_one_boltaev_cost_b_l685_68531

/-- The price of a shaltaev in kopeks -/
def shaltaev_price : ℕ := sorry

/-- The price of a boltaev in kopeks -/
def boltaev_price : ℕ := sorry

/-- 175 shaltaevs are more expensive than 125 boltaevs -/
axiom shaltaev_lower_bound : 175 * shaltaev_price > 125 * boltaev_price

/-- 175 shaltaevs are cheaper than 126 boltaevs -/
axiom shaltaev_upper_bound : 175 * shaltaev_price < 126 * boltaev_price

/-- Theorem: The cost of three shaltaevs and one boltaev is greater than 100 kopeks -/
theorem three_shaltaevs_one_boltaev_cost :
  3 * shaltaev_price + boltaev_price > 100 := by
  sorry

/-- Theorem: The cost of three shaltaevs and one boltaev is greater than 80 kopeks -/
theorem three_shaltaevs_one_boltaev_cost_a :
  3 * shaltaev_price + boltaev_price > 80 := by
  sorry

/-- Theorem: The cost of three shaltaevs and one boltaev is greater than one ruble (100 kopeks) -/
theorem three_shaltaevs_one_boltaev_cost_b :
  3 * shaltaev_price + boltaev_price > 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_shaltaevs_one_boltaev_cost_three_shaltaevs_one_boltaev_cost_a_three_shaltaevs_one_boltaev_cost_b_l685_68531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_three_pi_half_plus_theta_l685_68554

theorem cos_three_pi_half_plus_theta (θ : ℝ) 
  (h1 : Real.sin (θ - π/6) = 1/4) 
  (h2 : θ ∈ Set.Ioo (π/6) (2*π/3)) : 
  Real.cos (3*π/2 + θ) = (Real.sqrt 15 + Real.sqrt 3) / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_three_pi_half_plus_theta_l685_68554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_disjoint_chords_with_equal_sum_l685_68547

/-- Given 2^500 points on a circle numbered from 1 to 2^500, there exist 100 pairwise disjoint chords
    connecting some of these points, such that the sum of the numbers at the ends of each chord is equal. -/
theorem exist_disjoint_chords_with_equal_sum : ∃ (chords : Finset (Fin (2^500) × Fin (2^500))),
  (chords.card = 100) ∧ 
  (∀ c1 c2, c1 ∈ chords → c2 ∈ chords → c1 ≠ c2 → 
    (c1.1 ≠ c2.1 ∧ c1.1 ≠ c2.2 ∧ c1.2 ≠ c2.1 ∧ c1.2 ≠ c2.2)) ∧
  (∃ s : ℕ, ∀ c, c ∈ chords → c.1.val + c.2.val = s) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_disjoint_chords_with_equal_sum_l685_68547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_reaching_target_l685_68581

def Step := Fin 4

def move (p : ℤ × ℤ) (s : Fin 4) : ℤ × ℤ :=
  match s with
  | ⟨0, _⟩ => (p.1 + 1, p.2) -- right
  | ⟨1, _⟩ => (p.1 - 1, p.2) -- left
  | ⟨2, _⟩ => (p.1, p.2 + 1) -- up
  | ⟨3, _⟩ => (p.1, p.2 - 1) -- down

def reachesTarget (path : List (Fin 4)) : Bool :=
  path.length ≤ 8 ∧ 
  (path.foldl move (0, 0) = (3, 0))

def allPaths (n : ℕ) : List (List (Fin 4)) :=
  if n = 0 then [[]]
  else List.join (List.map (fun path => List.map (fun s => s::path) [0, 1, 2, 3]) (allPaths (n-1)))

theorem probability_of_reaching_target :
  (((allPaths 8).filter reachesTarget).length : ℚ) / ((4^8 : ℕ) : ℚ) = 175 / 16384 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_reaching_target_l685_68581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_DEF_perimeter_l685_68571

/-- Represents a triangle with an incircle -/
structure TriangleWithIncircle where
  /-- The radius of the incircle -/
  r : ℝ
  /-- The length of one segment from a vertex to a tangency point -/
  a : ℝ
  /-- The length of another segment from a vertex to a tangency point -/
  b : ℝ

/-- Calculates the perimeter of a triangle given its incircle properties -/
noncomputable def perimeter (t : TriangleWithIncircle) : ℝ :=
  2 * (t.a + t.b + t.r * (40 + (300 / 57)) / 15)

/-- Theorem stating the perimeter of the specific triangle DEF -/
theorem triangle_DEF_perimeter :
  let t : TriangleWithIncircle := { r := 15, a := 18, b := 22 }
  perimeter t = 80 + 600 / 57 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_DEF_perimeter_l685_68571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_specific_angles_l685_68576

theorem sin_sum_specific_angles (α β : Real) : 
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  Real.sin α = 1/2 → 
  Real.cos β = 1/2 → 
  Real.sin (α + β) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_specific_angles_l685_68576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_seventh_equals_eight_l685_68566

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain : Set ℝ := { x : ℝ | x > 0 }

-- State the theorem
theorem f_one_seventh_equals_eight
  (h_monotonic : Monotone f)
  (h_domain : ∀ x ∈ domain, f (f x - 1/x) = 2) :
  f (1/7) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_seventh_equals_eight_l685_68566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_three_digit_numbers_l685_68592

open BigOperators

theorem sum_of_three_digit_numbers : 
  let digits : Finset ℕ := {0, 1, 2, 3, 4, 5, 8}
  let non_zero_digits : Finset ℕ := digits.filter (λ x => x ≠ 0)
  let all_numbers : ℕ := 
    ∑ h in non_zero_digits, ∑ t in digits, ∑ o in digits, 
      100 * h + 10 * t + o
  all_numbers = 123326 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_three_digit_numbers_l685_68592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_is_36_l685_68584

-- Define the volume of the cone
noncomputable def cone_volume : ℝ := 12288 * Real.pi

-- Define the vertex angle of the vertical cross section
def vertex_angle : ℝ := 90

-- Define the relationship between height and radius for a 90-degree vertex angle
def height_radius_relation (h : ℝ) (r : ℝ) : Prop := h = r

-- Define the volume formula for a cone
noncomputable def cone_volume_formula (h : ℝ) (r : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

-- Theorem statement
theorem cone_height_is_36 :
  ∃ (h : ℝ) (r : ℝ),
    height_radius_relation h r ∧
    cone_volume_formula h r = cone_volume ∧
    h = 36 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_is_36_l685_68584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_time_l685_68594

noncomputable def journey_time (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

noncomputable def meeting_point (total_distance : ℝ) (distance_x : ℝ) : ℝ := total_distance - distance_x

theorem train_journey_time 
  (total_distance : ℝ) 
  (time_y : ℝ) 
  (distance_x : ℝ) :
  total_distance = 140 →
  time_y = 3 →
  distance_x = 60.00000000000001 →
  journey_time total_distance (distance_x / (time_y * distance_x / (total_distance - distance_x))) = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_time_l685_68594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_minus_sine_value_l685_68532

theorem cosine_minus_sine_value (θ : Real) 
  (h1 : θ > π/4) 
  (h2 : θ < π/2) 
  (h3 : Real.sin (2*θ) = 1/16) : 
  Real.cos θ - Real.sin θ = -Real.sqrt 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_minus_sine_value_l685_68532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_l685_68575

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  leg : ℝ
  base1 : ℝ
  base2 : ℝ

/-- Calculate the area of an isosceles trapezoid -/
noncomputable def area (t : IsoscelesTrapezoid) : ℝ :=
  let height := Real.sqrt (t.leg ^ 2 - ((t.base2 - t.base1) / 2) ^ 2)
  (t.base1 + t.base2) / 2 * height

/-- The theorem stating the area of the specific isosceles trapezoid -/
theorem isosceles_trapezoid_area :
  let t : IsoscelesTrapezoid := ⟨5, 10, 16⟩
  area t = 52 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_l685_68575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_tan_theta_l685_68535

theorem perpendicular_vectors_tan_theta (θ : ℝ) : 
  let a : ℝ × ℝ := (Real.cos θ, Real.sin θ)
  let b : ℝ × ℝ := (Real.sqrt 3, 1)
  (a.1 * b.1 + a.2 * b.2 = 0) → Real.tan θ = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_tan_theta_l685_68535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l685_68561

theorem log_inequality (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  Real.log (a^2 + 1) / Real.log a < Real.log (a^3 + 1) / Real.log a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l685_68561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_china_travel_more_cost_effective_l685_68522

/-- Represents a mobile phone card with its fee structure -/
structure PhoneCard where
  name : String
  basicFee : ℚ
  callRate : ℚ

/-- Calculates the call duration (in minutes) for a given expense and card -/
def callDuration (expense : ℚ) (card : PhoneCard) : ℚ :=
  (expense - card.basicFee) / card.callRate

theorem china_travel_more_cost_effective : 
  let globalCard : PhoneCard := { name := "Global", basicFee := 50, callRate := 40/100 }
  let chinaTravelCard : PhoneCard := { name := "China Travel", basicFee := 0, callRate := 60/100 }
  let monthlyBudget : ℚ := 120
  callDuration monthlyBudget chinaTravelCard > callDuration monthlyBudget globalCard := by
  sorry

#eval callDuration 120 { name := "Global", basicFee := 50, callRate := 40/100 }
#eval callDuration 120 { name := "China Travel", basicFee := 0, callRate := 60/100 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_china_travel_more_cost_effective_l685_68522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundred_digit_divisible_by_three_l685_68525

def count_divisible_by_three (n : ℕ) : ℕ :=
  (4^n + 2) / 3

theorem hundred_digit_divisible_by_three :
  count_divisible_by_three 50 = (Finset.filter (fun x => x.digits 2 ⊆ {1, 2} ∧ x % 3 = 0) (Finset.range (10^100))).card :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundred_digit_divisible_by_three_l685_68525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sqrt_sum_l685_68508

noncomputable section

theorem max_value_sqrt_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z = 3 →
    Real.sqrt (3*x + 1) + Real.sqrt (3*y + 1) + Real.sqrt (3*z + 1) ≤ Real.sqrt (3*a + 1) + Real.sqrt (3*b + 1) + Real.sqrt (3*c + 1)) →
  Real.sqrt (3*a + 1) + Real.sqrt (3*b + 1) + Real.sqrt (3*c + 1) = 6 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sqrt_sum_l685_68508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_condition_l685_68509

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.sin (x + φ)

theorem sufficient_but_not_necessary_condition :
  (∀ x, f (π/2) x = f (π/2) (-x)) ∧
  (∃ φ ≠ π/2, ∀ x, f φ x = f φ (-x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_condition_l685_68509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_problem_l685_68570

/-- Given two lines in the xy-plane, if they are perpendicular, then their slopes are negative reciprocals of each other. -/
def Perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The slope of a line in the form ax + by + c = 0 is -a/b when b ≠ 0. -/
noncomputable def slope_from_general_form (a b : ℝ) : ℝ := -a / b

theorem perpendicular_lines_problem (m : ℝ) :
  Perpendicular (slope_from_general_form 1 (-1)) m → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_problem_l685_68570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l685_68546

-- Define set M
def M : Set ℝ := {y | ∃ x, y = Real.exp (x * Real.log 2)}

-- Define set N
def N : Set ℝ := {y | ∃ x, y = x^2}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioi 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l685_68546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l685_68537

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 2 * Real.sqrt 3 * (Real.cos x)^2 + 2 * Real.sin x * Real.cos x - m

-- Define the triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- State the theorem
theorem triangle_problem (m : ℝ) (t : Triangle) : 
  (∀ x, f m x ≤ 2) →                             -- Maximum value of f(x) is 2
  (0 < t.A ∧ t.A < Real.pi/2) →                  -- A is an acute angle
  (f m t.A = 0) →                                -- f(A) = 0
  (Real.sin t.B = 3 * Real.sin t.C) →            -- sin B = 3 sin C
  (1/2 * t.b * t.c * Real.sin t.A = 3 * Real.sqrt 3 / 4) →  -- Area of triangle ABC is 3√3/4
  (m = Real.sqrt 3 ∧ t.a = Real.sqrt 7) := by    -- Prove m = √3 and a = √7
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l685_68537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l685_68540

/-- The distance between the foci of a hyperbola given by the equation 3x^2 - 18x - 9y^2 - 27y = 81 is 2√39 -/
theorem hyperbola_foci_distance :
  ∃ (c : ℝ) (f₁ f₂ : ℝ × ℝ),
    c = Real.sqrt 39 ∧
    let hyperbola := {(x, y) : ℝ × ℝ | 3 * x^2 - 18 * x - 9 * y^2 - 27 * y = 81}
    f₁ ∈ hyperbola ∧ f₂ ∈ hyperbola ∧
    (∀ p ∈ hyperbola, dist p f₁ - dist p f₂ = 2 * c ∨ dist p f₁ - dist p f₂ = -2 * c) ∧
    dist f₁ f₂ = 2 * Real.sqrt 39 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l685_68540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solvers_exist_l685_68504

/-- Represents a mathematics competition -/
structure MathCompetition where
  candidates : Finset ℕ
  problems : Finset ℕ
  solved : ℕ → Finset ℕ  -- For each problem, the set of candidates who solved it

/-- The competition satisfies the given conditions -/
def ValidCompetition (comp : MathCompetition) : Prop :=
  comp.candidates.card = 200 ∧
  comp.problems.card = 6 ∧
  ∀ p ∈ comp.problems, (comp.solved p).card ≥ 120

/-- There exist two candidates who collectively solve all problems -/
def ExistTwoSolvers (comp : MathCompetition) : Prop :=
  ∃ c₁ c₂, c₁ ∈ comp.candidates ∧ c₂ ∈ comp.candidates ∧
    ∀ p ∈ comp.problems, c₁ ∈ comp.solved p ∨ c₂ ∈ comp.solved p

/-- The main theorem -/
theorem two_solvers_exist (comp : MathCompetition) (h : ValidCompetition comp) :
  ExistTwoSolvers comp := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solvers_exist_l685_68504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_son_shoveling_time_l685_68565

/-- Represents the time it takes for the son to shovel one driveway alone -/
noncomputable def son_time (wayne_rate : ℝ) (son_rate : ℝ) : ℝ :=
  1 / son_rate

/-- Represents the time it takes for Wayne and his son to shovel one driveway together -/
noncomputable def combined_time (wayne_rate : ℝ) (son_rate : ℝ) : ℝ :=
  1 / (wayne_rate + son_rate)

theorem son_shoveling_time 
  (wayne_rate : ℝ) (son_rate : ℝ)
  (h1 : wayne_rate = 6 * son_rate) 
  (h2 : combined_time wayne_rate son_rate = 3) :
  son_time wayne_rate son_rate = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_son_shoveling_time_l685_68565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_ratio_theorem_l685_68599

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the angle measures
noncomputable def angle_B (t : Triangle) : ℝ := 45 * Real.pi / 180
noncomputable def angle_C (t : Triangle) : ℝ := 30 * Real.pi / 180

-- Define point D that divides BC in ratio 1:2
noncomputable def point_D (t : Triangle) : ℝ × ℝ :=
  ((2 * t.B.1 + t.C.1) / 3, (2 * t.B.2 + t.C.2) / 3)

-- Define the angles BAD and CAD
noncomputable def angle_BAD (t : Triangle) : ℝ := sorry
noncomputable def angle_CAD (t : Triangle) : ℝ := sorry

-- State the theorem
theorem sin_ratio_theorem (t : Triangle) :
  Real.sin (angle_BAD t) / Real.sin (angle_CAD t) = 1 / (4 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_ratio_theorem_l685_68599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_speed_is_sqrt_34_l685_68598

/-- The speed of a particle moving in a 2D plane -/
noncomputable def particle_speed : ℝ := Real.sqrt 34

/-- The position of the particle at time t -/
def particle_position (t : ℝ) : ℝ × ℝ := (3 * t + 4, 5 * t - 7)

/-- Theorem stating that the speed of the particle is √34 -/
theorem particle_speed_is_sqrt_34 :
  let v := λ t : ℝ => particle_position t
  let velocity := λ t : ℝ => (v (t + 1) - v t)
  let speed := λ t : ℝ => Real.sqrt ((velocity t).1 ^ 2 + (velocity t).2 ^ 2)
  ∀ t : ℝ, speed t = particle_speed :=
by
  intro t
  sorry

#check particle_speed_is_sqrt_34

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_speed_is_sqrt_34_l685_68598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_l685_68530

noncomputable def f (p q x : ℝ) := x^2 + p*x + q
noncomputable def g (x : ℝ) := x + 1/x^2

theorem max_value_f (p q : ℝ) :
  (∃ x₀ ∈ Set.Icc 1 2, 
    (∀ x ∈ Set.Icc 1 2, f p q x₀ ≤ f p q x) ∧
    (∀ x ∈ Set.Icc 1 2, g x₀ ≤ g x) ∧
    f p q x₀ = g x₀) →
  (∃ x_max ∈ Set.Icc 1 2, 
    ∀ x ∈ Set.Icc 1 2, f p q x ≤ f p q x_max ∧ 
    f p q x_max = 4 - (5/2) * Real.rpow 2 (1/3) + Real.rpow 4 (1/3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_l685_68530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_bound_l685_68550

/-- The function f(x) as defined in the problem -/
noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + (a / 2) * x^2 - (a^2 + 1) * x

/-- The minimum value of f(x) when a > 1 -/
noncomputable def g (a : ℝ) : ℝ := f a a

/-- The theorem to be proved -/
theorem min_value_bound (a : ℝ) (h : a > 1) :
  g a < 0 - (1/4) * (2*a^3 - 2*a^2 + 5*a) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_bound_l685_68550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_polynomial_characterization_l685_68514

-- Define the polynomial
def P (a b c : ℤ) (m : ℤ) : ℤ := a * m^3 + b * m^2 + c * m

-- Define the theorem
theorem permutation_polynomial_characterization 
  (a b c p : ℤ) (hp : Nat.Prime p.natAbs) (hp_ge_5 : p ≥ 5) :
  (∀ (x y : ℤ), 0 ≤ x ∧ x < p ∧ 0 ≤ y ∧ y < p ∧ x ≠ y → 
    P a b c x % p ≠ P a b c y % p) ↔ 
  (Nat.gcd a.natAbs p.natAbs = 1 ∧ 
   (b^2 % p = (3 * a * c) % p) ∧
   p % 3 = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_polynomial_characterization_l685_68514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_equality_condition_l685_68506

theorem divisor_equality_condition (n k : ℕ+) (hn : n ≠ k) :
  (∃ s : ℕ+, (Nat.divisors (s * n).val).card = (Nat.divisors (s * k).val).card) ↔ 
  (¬(n ∣ k) ∧ ¬(k ∣ n)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_equality_condition_l685_68506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l685_68593

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_min : min (a * b) (min (b * c) (c * a)) ≥ 1) :
  ((a^2 + 1) * (b^2 + 1) * (c^2 + 1))^(1/3) ≤ ((a + b + c) / 3)^2 + 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l685_68593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l685_68553

/-- Time taken for a train to cross a platform -/
theorem train_crossing_time (train_length platform_length : ℝ) (train_speed_kmh : ℝ) :
  train_length = 480 ∧ 
  platform_length = 620 ∧ 
  train_speed_kmh = 55 →
  ∃ (time : ℝ), (time ≥ 71.9 ∧ time ≤ 72.1) ∧ time = (train_length + platform_length) / (train_speed_kmh * 1000 / 3600) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l685_68553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_ratio_l685_68574

-- Define the original pyramid
noncomputable def original_base_edge : ℝ := 36
noncomputable def original_altitude : ℝ := 15

-- Define the scaling factor
noncomputable def scale_factor : ℝ := 1/3

-- Define the volume of a pyramid
noncomputable def pyramid_volume (base_edge : ℝ) (altitude : ℝ) : ℝ :=
  (1/3) * base_edge^2 * altitude

-- Define the volume ratio of similar solids
noncomputable def volume_ratio (scale : ℝ) : ℝ := scale^3

-- Theorem statement
theorem frustum_volume_ratio :
  let original_volume := pyramid_volume original_base_edge original_altitude
  let smaller_volume := pyramid_volume (scale_factor * original_base_edge) (scale_factor * original_altitude)
  let frustum_volume := original_volume - smaller_volume
  frustum_volume / original_volume = 26/27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_ratio_l685_68574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minute_hand_45_minutes_l685_68572

/-- The distance traveled by the tip of a clock's minute hand -/
noncomputable def minute_hand_distance (hand_length : ℝ) (minutes : ℝ) : ℝ :=
  2 * Real.pi * hand_length * (minutes / 60)

/-- Theorem: The distance traveled by the tip of a 10 cm long minute hand in 45 minutes is 15π cm -/
theorem minute_hand_45_minutes :
  minute_hand_distance 10 45 = 15 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minute_hand_45_minutes_l685_68572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bus_cost_difference_l685_68521

/-- The cost of a train ride from town P to town Q -/
def train_cost : ℝ := sorry

/-- The cost of a bus ride from town P to town Q -/
def bus_cost : ℝ := 1.50

theorem train_bus_cost_difference :
  (train_cost > bus_cost) →
  (train_cost + bus_cost = 9.85) →
  (train_cost - bus_cost = 6.85) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bus_cost_difference_l685_68521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ben_round_trip_time_l685_68597

/-- Represents a round trip journey with walking and cycling components -/
structure RoundTrip where
  distance : ℝ
  walkingTime : ℝ
  cyclingSpeed : ℝ

/-- Calculates the average speed of a round trip -/
noncomputable def averageSpeed (trip : RoundTrip) : ℝ :=
  (2 * trip.distance) / (trip.walkingTime + trip.distance / trip.cyclingSpeed)

/-- Calculates the time taken for a round trip at a given speed -/
noncomputable def roundTripTime (distance : ℝ) (speed : ℝ) : ℝ :=
  (2 * distance) / speed

/-- Theorem stating that Ben's round trip time is 160 minutes -/
theorem ben_round_trip_time : 
  let bexy_trip := RoundTrip.mk 5 1 15
  let ben_speed := averageSpeed bexy_trip / 2
  roundTripTime 5 ben_speed * 60 = 160 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ben_round_trip_time_l685_68597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_is_correct_l685_68507

/-- Represents the financial details of Ramesh's refrigerator purchase and sale --/
structure RefrigeratorPurchase where
  discounted_price : ℚ
  transport_cost : ℚ
  installation_cost : ℚ
  selling_price : ℚ
  profit_percentage : ℚ

/-- Calculates the discount percentage given the purchase details --/
def calculate_discount_percentage (purchase : RefrigeratorPurchase) : ℚ :=
  let labelled_price := purchase.selling_price / (1 + purchase.profit_percentage)
  let discount_amount := labelled_price - purchase.discounted_price
  (discount_amount / labelled_price) * 100

/-- Theorem stating that the calculated discount percentage is approximately 21.43% --/
theorem discount_percentage_is_correct (purchase : RefrigeratorPurchase)
  (h1 : purchase.discounted_price = 16500)
  (h2 : purchase.transport_cost = 125)
  (h3 : purchase.installation_cost = 250)
  (h4 : purchase.selling_price = 23100)
  (h5 : purchase.profit_percentage = 1/10) :
  ∃ (ε : ℚ), ε > 0 ∧ |calculate_discount_percentage purchase - 21427/1000| < ε := by
  sorry

#eval calculate_discount_percentage {
  discounted_price := 16500,
  transport_cost := 125,
  installation_cost := 250,
  selling_price := 23100,
  profit_percentage := 1/10
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_is_correct_l685_68507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_arithmetic_progression_disjoint_l685_68527

def fibonacci : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

def arithmetic_progression (k : ℤ) : ℤ := 4 + 8 * k

theorem fibonacci_arithmetic_progression_disjoint :
  ∀ n k, fibonacci n ≠ arithmetic_progression k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_arithmetic_progression_disjoint_l685_68527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l685_68551

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (-1/2 * t, 2 + (Real.sqrt 3)/2 * t)

noncomputable def circle_C (θ : ℝ) : ℝ := 4 * Real.cos θ

def point_P : ℝ × ℝ := (0, 2)

def trajectory_Q (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 16

theorem intersection_distance_sum :
  ∃ (A B : ℝ × ℝ),
    (∃ (t₁ t₂ : ℝ), line_l t₁ = A ∧ line_l t₂ = B) ∧
    trajectory_Q A.1 A.2 ∧
    trajectory_Q B.1 B.2 ∧
    Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2) +
    Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2) =
    4 + 2 * Real.sqrt 3 := by
  sorry

#check intersection_distance_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l685_68551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_exponential_range_l685_68591

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a^2 - 1)^x

theorem decreasing_exponential_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ > f a x₂) ↔ 
  (a > -Real.sqrt 2 ∧ a < -1) ∨ (a > 1 ∧ a < Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_exponential_range_l685_68591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_x_l685_68555

theorem find_x : ∃ x : ℤ, 
  (¬(3 * x > 30)) ∧ 
  (x ≥ 10) ∧ 
  (x > 5) ∧ 
  (∃ (s₁ s₂ s₃ s₄ s₅ : Prop), 
    s₁ = (3 * x > 30) ∧
    s₂ = (x ≥ 10) ∧
    s₃ = (x > 5) ∧
    s₄ = (x ∈ Set.univ) ∧
    s₅ = (s₁ ∨ s₂ ∨ s₃ ∨ s₄ ∨ s₅) ∧
    (s₁ ↔ False) ∧ (s₂ ↔ True) ∧ (s₃ ↔ True) ∧ (s₄ ↔ True) ∧ (s₅ ↔ False)) ∧
  x = 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_x_l685_68555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laser_path_distance_l685_68577

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents the path of the laser beam -/
def laserPath : List Point :=
  [⟨2, 4⟩, ⟨0, 4⟩, ⟨0, -4⟩, ⟨8, 4⟩]

/-- Theorem stating the total distance of the laser path -/
theorem laser_path_distance :
  (List.zip laserPath laserPath.tail).foldl
    (fun acc pair => acc + distance pair.1 pair.2) 0 = 10 + 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_laser_path_distance_l685_68577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prop_1_prop_2_prop_3_prop_4_l685_68583

/-- Definition of curvature -/
noncomputable def curvature (f : ℝ → ℝ) (x₁ x₂ : ℝ) : ℝ :=
  let y₁ := f x₁
  let y₂ := f x₂
  let k₁ := (deriv f) x₁
  let k₂ := (deriv f) x₂
  |k₁ - k₂| / Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- Proposition 1 -/
theorem prop_1 : ∃ f : ℝ → ℝ, curvature f 1 2 ≤ Real.sqrt 3 ∧ 
  f = fun x ↦ x^3 - x^2 + 1 := by sorry

/-- Proposition 2 -/
theorem prop_2 : ∃ f : ℝ → ℝ, ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → 
  curvature f x₁ x₂ = curvature f 0 1 := by sorry

/-- Proposition 3 -/
theorem prop_3 : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → 
  curvature (fun x ↦ x^2 + 1) x₁ x₂ ≤ 2 := by sorry

/-- Proposition 4 -/
theorem prop_4 : ∃ t : ℝ, t ≥ 1 ∧ ∀ x : ℝ, 
  t * curvature Real.exp x (x + 1) < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prop_1_prop_2_prop_3_prop_4_l685_68583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_when_a_zero_f_nonnegative_iff_a_leq_one_l685_68590

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - (1/3) * a * x^3 - (1/2) * x^2 + 1

-- Part 1: Monotonicity when a = 0
theorem f_increasing_when_a_zero :
  ∀ x y : ℝ, x < y → f 0 x < f 0 y := by sorry

-- Part 2: Condition for f(x) ≥ 0 on [0,+∞)
theorem f_nonnegative_iff_a_leq_one :
  (∀ x : ℝ, x ≥ 0 → f a x ≥ 0) ↔ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_when_a_zero_f_nonnegative_iff_a_leq_one_l685_68590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_d_l685_68503

theorem find_d (x₁ x₂ k d : ℝ) (h1 : x₁ ≠ x₂) 
  (h2 : 4 * x₁^2 - k * x₁ = d ∧ 4 * x₂^2 - k * x₂ = d) 
  (h3 : x₁ + x₂ = 2) : d = -12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_d_l685_68503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_min_distance_value_l685_68568

/-- Parabola defined by y = -1/8(x-4)^2 -/
def parabola (x y : ℝ) : Prop := y = -1/8 * (x - 4)^2

/-- Point on the parabola -/
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

/-- Focus of the parabola -/
noncomputable def F : ℝ × ℝ := (4, 7/2)

/-- Vertex of the parabola -/
noncomputable def E : ℝ × ℝ := (4, -1/2)

/-- Point on the axis of symmetry -/
structure PointOnAxis where
  x : ℝ
  y : ℝ
  on_axis : x = 4

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Point A on the parabola with |AF| = 4 -/
noncomputable def A : PointOnParabola :=
  { x := 8
  , y := -2
  , on_parabola := by
      simp [parabola]
      ring }

theorem min_distance_sum :
  ∃ (P : PointOnAxis), ∀ (Q : PointOnAxis),
    distance (A.x, A.y) (P.x, P.y) + distance (P.x, P.y) E ≤
    distance (A.x, A.y) (Q.x, Q.y) + distance (Q.x, Q.y) E :=
by
  sorry

theorem min_distance_value :
  ∃ (P : PointOnAxis),
    distance (A.x, A.y) (P.x, P.y) + distance (P.x, P.y) E = 2 * Real.sqrt 13 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_min_distance_value_l685_68568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l685_68515

noncomputable def f (x : ℝ) : ℝ := (4 * x^2 + 2 * x + 5) / (x^2 + x + 1)

theorem min_value_of_f : 
  ∀ x > 1, f x ≥ (16 - 2 * Real.sqrt 7) / 3 ∧ 
  ∃ x > 1, f x = (16 - 2 * Real.sqrt 7) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l685_68515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_about_y_equals_x_l685_68586

-- Define the functions f and g as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 4
noncomputable def g (x : ℝ) : ℝ := 4^x

-- State the theorem
theorem symmetric_about_y_equals_x (x y : ℝ) :
  f x = y ↔ g y = x :=
by
  sorry -- Skip the proof for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_about_y_equals_x_l685_68586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l685_68513

-- Define the types for our variables
variable (a b : ℝ)

-- Define the condition from the problem
def condition (a b : ℝ) : Prop :=
  ∀ x, a * x - b > 0 ↔ x < 1

-- Define the theorem
theorem solution_set (a b : ℝ) (h : condition a b) :
  ∀ x, (a*x + b)*(x - 2) > 0 ↔ -1 < x ∧ x < 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l685_68513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_specific_l685_68533

/-- The total surface area of a right pyramid with a rectangular base -/
noncomputable def pyramid_surface_area (base_length : ℝ) (base_width : ℝ) (height : ℝ) : ℝ :=
  let base_area := base_length * base_width
  let half_length := base_length / 2
  let half_width := base_width / 2
  let side_height1 := Real.sqrt (height^2 + half_length^2)
  let side_height2 := Real.sqrt (height^2 + half_width^2)
  base_area + base_length * side_height1 + base_width * side_height2

/-- Theorem stating the specific surface area of the pyramid in the problem -/
theorem pyramid_surface_area_specific : 
  pyramid_surface_area 14 8 15 = 112 + 14 * Real.sqrt 274 + 8 * Real.sqrt 241 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_specific_l685_68533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_two_digit_odd_digits_l685_68500

/-- A function that returns true if a number is odd, false otherwise -/
def is_odd (n : ℕ) : Bool :=
  n % 2 = 1

/-- A function that returns the tens digit of a two-digit number -/
def tens_digit (n : ℕ) : ℕ :=
  n / 10

/-- A function that returns the units digit of a two-digit number -/
def units_digit (n : ℕ) : ℕ :=
  n % 10

/-- A function that returns true if a number is a two-digit number with both digits odd -/
def is_two_digit_odd_digits (n : ℕ) : Bool :=
  n ≥ 10 ∧ n ≤ 99 ∧ is_odd (tens_digit n) ∧ is_odd (units_digit n)

/-- The theorem stating that the sum of all two-digit numbers with both digits odd is 1375 -/
theorem sum_two_digit_odd_digits : 
  (Finset.filter (fun n => is_two_digit_odd_digits n) (Finset.range 100)).sum id = 1375 := by
  sorry

#eval (Finset.filter (fun n => is_two_digit_odd_digits n) (Finset.range 100)).sum id

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_two_digit_odd_digits_l685_68500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_a_l685_68517

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2^x + 1 else x^2 + a*x

-- State the theorem
theorem solve_for_a (a : ℝ) : f a (f a 0) = 4 * a → a = 2 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_a_l685_68517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_cost_per_hour_approx_l685_68549

-- Define the parking cost structure
noncomputable def base_cost : ℝ := 12.00
def base_hours : ℕ := 2
noncomputable def additional_cost_per_hour : ℝ := 1.75

-- Define the parking durations for each vehicle
def vehicle_A_hours : ℕ := 9
def vehicle_B_hours : ℕ := 5
def vehicle_C_hours : ℕ := 12

-- Function to calculate parking cost for a given duration
noncomputable def parking_cost (hours : ℕ) : ℝ :=
  base_cost + (max (hours - base_hours) 0 : ℝ) * additional_cost_per_hour

-- Calculate total parking cost for all vehicles
noncomputable def total_cost : ℝ :=
  parking_cost vehicle_A_hours + parking_cost vehicle_B_hours + parking_cost vehicle_C_hours

-- Calculate total parking hours for all vehicles
def total_hours : ℕ :=
  vehicle_A_hours + vehicle_B_hours + vehicle_C_hours

-- Theorem: The average cost per hour is approximately $2.73
theorem average_cost_per_hour_approx :
  ∃ ε > 0, abs ((total_cost / total_hours : ℝ) - 2.73) < ε := by
  sorry

#eval total_hours

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_cost_per_hour_approx_l685_68549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_I_properties_l685_68523

noncomputable def I (n : ℕ) : ℕ := (Finset.filter (fun p => (50 : ℝ)^n < (7 : ℝ)^p ∧ (7 : ℝ)^p < (50 : ℝ)^(n+1)) (Finset.range (n+2))).card

theorem I_properties :
  (∀ n : ℕ, I n = 2 ∨ I n = 3) ∧
  (∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, I n = 3) ∧
  I 1 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_I_properties_l685_68523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l685_68589

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  d : ℚ      -- Common difference
  is_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

/-- The main theorem -/
theorem arithmetic_sequence_ratio 
  (seq : ArithmeticSequence) 
  (h1 : seq.a 1 ≠ 0) 
  (h2 : S seq 2 = seq.a 4) : 
  seq.a 5 / S seq 3 = 2 / 3 := by
  sorry

#check arithmetic_sequence_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l685_68589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prop_2_prop_3_l685_68580

-- Define the set M
def M : Set (ℝ → ℝ) := {f | ∀ x y, (f x)^2 - (f y)^2 = f (x + y) * f (x - y)}

-- Proposition 2
theorem prop_2 : (λ x : ℝ => 2 * x) ∈ M := by
  intro x y
  simp
  ring

-- Proposition 3
theorem prop_3 (f : ℝ → ℝ) (h : f ∈ M) : ∀ x, f (-x) = -f x := by
  sorry

-- The answer is the set containing propositions 2 and 3
def answer : Set Nat := {2, 3}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prop_2_prop_3_l685_68580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_sum_l685_68595

-- Define the polynomial f(x)
def f (a b c d : ℤ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

-- Define the theorem
theorem polynomial_root_sum (a b c d : ℤ) :
  (∃ r₁ r₂ r₃ r₄ : ℤ, r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧ r₄ > 0 ∧
    ∀ x : ℝ, f a b c d x = (x + r₁ : ℝ) * (x + r₂ : ℝ) * (x + r₃ : ℝ) * (x + r₄ : ℝ)) →
  a + b + c + d = 2023 →
  d = 17020 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_sum_l685_68595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_relations_l685_68596

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem geometric_relations 
  (α β : Plane) (l m : Line) 
  (h_distinct_planes : α ≠ β) 
  (h_distinct_lines : l ≠ m) :
  (∀ (l : Line) (α β : Plane), 
    perpendicular l α ∧ perpendicular l β → parallel_planes α β) ∧
  (∀ (α β : Plane) (l m : Line),
    intersect α β l ∧ parallel_lines m l → 
    parallel_line_plane m α ∨ parallel_line_plane m β) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_relations_l685_68596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_integer_coefficient_l685_68588

-- Define the root type
structure Root (α : Type) where
  value : α

-- Define the problem statement
theorem smallest_positive_integer_coefficient
  (a b c d e : ℤ)
  (p : List ℤ)
  (roots : List (Root ℚ))
  (h1 : p = [a, b, c, d, e])
  (h2 : roots = [⟨-3⟩, ⟨4⟩, ⟨11⟩, ⟨-1/4⟩])
  (h3 : ∀ r ∈ roots, (a * r.value^4 + b * r.value^3 + c * r.value^2 + d * r.value + e : ℚ) = 0)
  (h4 : e > 0) :
  e ≥ 132 ∧ ∃ (a' b' c' d' : ℤ), [a', b', c', d', 132] = p :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_integer_coefficient_l685_68588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_OD₁_l685_68534

/-- Given a cube ABCDEFGA₁B₁C₁D₁ and a sphere with center O and radius 10,
    where the sphere intersects:
    - face AA₁D₁D along a circle of radius 1
    - face A₁B₁C₁D₁ along a circle of radius 1
    - face CDD₁C₁ along a circle of radius 3
    Prove that the length of OD₁ is 17. -/
theorem length_of_OD₁ (O D₁ : EuclideanSpace ℝ (Fin 3))
  (sphere_radius intersect_radius₁ intersect_radius₂ : ℝ)
  (h1 : sphere_radius = 10)
  (h2 : intersect_radius₁ = 1)
  (h3 : intersect_radius₂ = 3) :
  ‖O - D₁‖ = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_OD₁_l685_68534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l685_68529

open Real

noncomputable def a (θ : ℝ) : Fin 2 → ℝ := ![2 * sin θ, 1]
noncomputable def b (θ : ℝ) : Fin 2 → ℝ := ![2 * cos θ, -1]

theorem vector_problem (θ : ℝ) (h : 0 < θ ∧ θ < π/2) :
  (a θ • b θ = 0 → θ = π/12 ∨ θ = 5*π/12) ∧
  (‖a θ - b θ‖ = 2 * ‖b θ‖ → tan θ = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l685_68529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k0_existence_l685_68516

/-- Given a positive even integer n, there exists a smallest positive integer k0 that can be expressed as k0 = f(x)(x+1)^n + g(x)(x^n + 1) for some polynomials f(x) and g(x) with integer coefficients, and k0 = 2^t where n = 2^a * t and t is odd. -/
theorem smallest_k0_existence (n : ℕ) (h_pos : 0 < n) (h_even : Even n) :
  ∃ (k0 : ℕ) (f g : Polynomial ℤ) (a t : ℕ),
    k0 > 0 ∧
    n = 2^a * t ∧
    Odd t ∧
    k0 = 2^t ∧
    (∀ (x : ℤ), k0 = f.eval x * (x + 1)^n + g.eval x * (x^n + 1)) ∧
    (∀ (k : ℕ) (f' g' : Polynomial ℤ),
      (k > 0 ∧ ∀ (x : ℤ), k = f'.eval x * (x + 1)^n + g'.eval x * (x^n + 1))
      → k0 ≤ k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k0_existence_l685_68516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_die_rolls_l685_68556

/-- The probability of rolling a 1 on a fair eight-sided die -/
noncomputable def prob_roll_one : ℝ := 1 / 8

/-- The probability of not rolling a 1 on a fair eight-sided die -/
noncomputable def prob_not_roll_one : ℝ := 7 / 8

/-- The number of days in a non-leap year -/
def days_in_year : ℕ := 365

/-- The expected number of rolls on a single day -/
noncomputable def expected_rolls_per_day : ℝ :=
  1 / (1 - prob_roll_one)

/-- The expected number of rolls in a non-leap year -/
noncomputable def expected_rolls_per_year : ℝ :=
  expected_rolls_per_day * (days_in_year : ℝ)

theorem bob_die_rolls :
  ∃ ε > 0, |expected_rolls_per_year - 417.14| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_die_rolls_l685_68556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_tiling_l685_68544

/-- Represents an L-shaped tile in a grid --/
structure LTile where
  x : ℕ
  y : ℕ

/-- Checks if two L-shaped tiles form a 3 × 2 rectangle --/
def forms_3x2_rectangle (t1 t2 : LTile) : Prop :=
  sorry

/-- Checks if a point is shared by 3 or more L-shaped tiles --/
def shared_by_3_or_more (point : ℕ × ℕ) (tiles : List LTile) : Prop :=
  sorry

/-- Checks if a tiling is valid according to the given conditions --/
def valid_tiling (m n : ℕ) (tiles : List LTile) : Prop :=
  (∀ t1 t2, t1 ∈ tiles → t2 ∈ tiles → t1 ≠ t2 → ¬forms_3x2_rectangle t1 t2) ∧
  (∀ point, ¬shared_by_3_or_more point tiles)

theorem no_valid_tiling :
  ∀ m n : ℕ, ¬∃ tiles : List LTile, valid_tiling m n tiles :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_tiling_l685_68544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l685_68541

/-- The function f(x) with parameters a and b -/
noncomputable def f (a b x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * (a - 1) * x^2 + b * x + 1

/-- The derivative of f(x) -/
noncomputable def f' (a b x : ℝ) : ℝ := x^2 - (a - 1) * x + b

theorem function_properties (a b : ℝ) (ha : a > 0)
  (h_perpendicular : f' a b (-1) = 0) :
  a + b = 0 ∧
  (∀ x : ℝ, x > 0 → f a b x ≤ (1/2) * a + 7/6) ∧
  (∀ x : ℝ, x > 0 → f a b x ≥ -(1/6) * a^3 - (1/2) * a^2 + 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l685_68541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_triangle_area_l685_68545

/-- An ellipse with center at the origin and focus on the x-axis -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : a^2 = 2
  h3 : b^2 = 1

/-- The line passing through a point (0, y₀) -/
structure Line where
  k : ℝ
  y₀ : ℝ

/-- The area of the triangle formed by the intersection of the line and the ellipse -/
noncomputable def triangle_area (e : Ellipse) (l : Line) : ℝ :=
  2 * Real.sqrt 2 * Real.sqrt (2 * l.k^2 - 3) / (1 + 2 * l.k^2)

theorem ellipse_max_triangle_area (e : Ellipse) :
  ∃ (l : Line), l.y₀ = 2 ∧ 
    (∀ (l' : Line), l'.y₀ = 2 → triangle_area e l ≥ triangle_area e l') ∧
    (l.k = Real.sqrt 14 / 2 ∨ l.k = -Real.sqrt 14 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_triangle_area_l685_68545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meaningful_range_l685_68505

-- Define the expression
noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (3 * x + 9)) / (x - 2)

-- State the theorem
theorem meaningful_range (x : ℝ) : 
  (∃ y : ℝ, f x = y) ↔ (x ≥ -3 ∧ x ≠ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meaningful_range_l685_68505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l685_68501

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (-x) * (a * x^2 + a + 1)

-- Define the derivative of f
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.exp (-x) * (-a * x^2 + 2 * a * x - a - 1)

theorem f_properties (a : ℝ) :
  (∀ x, a ≥ 0 → f_deriv a x < 0) ∧
  (a < 0 → ∃ x₁ x₂, ∀ x, 
    (x < x₁ → f_deriv a x > 0) ∧
    (x₁ < x ∧ x < x₂ → f_deriv a x < 0) ∧
    (x > x₂ → f_deriv a x > 0)) ∧
  (∀ x, -1 < a ∧ a < 0 → x ∈ Set.Icc 1 2 → f a x ≥ f a 2) :=
by sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l685_68501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_perimeter_l685_68502

noncomputable def originLine : ℝ → ℝ := λ x ↦ -1/Real.sqrt 3 * x

noncomputable def x1 : ℝ := -2 * Real.sqrt 3

noncomputable def x2 : ℝ := 1 / Real.sqrt 3

noncomputable def sideLength : ℝ := |x1 - x2|

theorem equilateral_triangle_perimeter :
  sideLength = 7 * Real.sqrt 3 / 3 →
  3 * sideLength = 7 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_perimeter_l685_68502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_third_side_range_l685_68542

-- Define Triangle as a structure
structure Triangle where
  sides : Finset ℝ
  side_count : sides.card = 3
  positive_sides : ∀ s ∈ sides, s > 0

-- Triangle inequality theorem
theorem triangle_inequality (a b c : ℝ) : 
  (a > 0 ∧ b > 0 ∧ c > 0) → 
  (a + b > c ∧ b + c > a ∧ c + a > b) ↔ 
  ∃ (t : Triangle), t.sides = {a, b, c} := by sorry

-- Theorem for the range of the third side
theorem third_side_range (s1 s2 s3 : ℝ) :
  s1 = 4 → s2 = 9 → 
  (∃ (t : Triangle), t.sides = {s1, s2, s3}) → 
  (s3 > 5 ∧ s3 < 13) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_third_side_range_l685_68542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_consistency_l685_68510

theorem race_consistency (total_racers : ℕ) (bicycle_fraction : ℚ) (total_wheels : ℕ) 
  (h1 : total_racers = 40)
  (h2 : bicycle_fraction = 3/5)
  (h3 : total_wheels = 96) :
  ∃ (bicycles tricycles : ℕ),
    bicycles + tricycles = total_racers ∧
    bicycles = Int.floor (bicycle_fraction * total_racers) ∧
    tricycles = total_racers - bicycles ∧
    2 * bicycles + 3 * tricycles = total_wheels :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_consistency_l685_68510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_coin_order_l685_68548

-- Define the coin type
inductive Coin : Type
  | A | B | C | D | E | F
  deriving BEq, Repr

-- Define the relation for one coin being above another
def above : Coin → Coin → Prop := sorry

-- State the problem conditions
axiom F_top : ∀ x, x ≠ Coin.F → above Coin.F x
axiom C_above_DEA : above Coin.C Coin.D ∧ above Coin.C Coin.E ∧ above Coin.C Coin.B
axiom D_above_AB : above Coin.D Coin.A ∧ above Coin.D Coin.B
axiom E_above_A : above Coin.E Coin.A
axiom A_above_B : above Coin.A Coin.B

-- Define a function to represent the order of coins
def coinOrder : List Coin :=
  [Coin.F, Coin.C, Coin.D, Coin.E, Coin.A, Coin.B]

-- State the theorem to be proved
theorem correct_coin_order :
  ∀ (x y : Coin), x ≠ y →
  (coinOrder.indexOf x < coinOrder.indexOf y ↔ above x y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_coin_order_l685_68548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_to_line_l685_68578

/-- The ellipse defined by the equation x²/16 + y²/4 = 1 -/
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

/-- The line defined by the equation x + 2y - √2 = 0 -/
def line (x y : ℝ) : Prop := x + 2*y - Real.sqrt 2 = 0

/-- The distance from a point (x, y) to the line x + 2y - √2 = 0 -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (x + 2*y - Real.sqrt 2) / Real.sqrt 5

/-- The maximum distance from any point on the ellipse to the line is √10 -/
theorem max_distance_ellipse_to_line :
  ∃ (x y : ℝ), ellipse x y ∧ 
  ∀ (x' y' : ℝ), ellipse x' y' → distance_to_line x y ≥ distance_to_line x' y' ∧
  distance_to_line x y = Real.sqrt 10 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_to_line_l685_68578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_diff_is_square_l685_68552

theorem gcd_diff_is_square (x y z : ℕ+) (h : (1 : ℚ) / x.val - (1 : ℚ) / y.val = (1 : ℚ) / z.val) :
  ∃ k : ℕ, Nat.gcd x.val (Nat.gcd y.val z.val) * (y.val - x.val) = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_diff_is_square_l685_68552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sally_total_fries_l685_68512

def initial_fries : ℕ := 14
def mark_dozens : ℕ := 3
def mark_fraction : ℚ := 1/3
def jessica_cm : ℕ := 240
def fry_length : ℕ := 5
def jessica_fraction : ℚ := 1/2

theorem sally_total_fries :
  let mark_fries := mark_dozens * 12
  let mark_given := (mark_fries : ℚ) * mark_fraction
  let jessica_fries := jessica_cm / fry_length
  let jessica_given := (jessica_fries : ℚ) * jessica_fraction
  initial_fries + mark_given.floor + jessica_given.floor = 50 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sally_total_fries_l685_68512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lightning_distance_l685_68518

/-- The speed of sound in feet per second -/
noncomputable def speed_of_sound : ℝ := 1100

/-- The number of feet in a mile -/
noncomputable def feet_per_mile : ℝ := 5280

/-- The time in seconds between seeing the lightning and hearing the thunder -/
noncomputable def time_delay : ℝ := 12

/-- Calculates the distance to the lightning strike in miles -/
noncomputable def distance_to_lightning : ℝ := (speed_of_sound * time_delay) / feet_per_mile

/-- Rounds a number to the nearest quarter -/
noncomputable def round_to_nearest_quarter (x : ℝ) : ℝ :=
  (⌊x * 4 + 0.5⌋ : ℝ) / 4

theorem lightning_distance :
  round_to_nearest_quarter distance_to_lightning = 2.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lightning_distance_l685_68518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_and_distributions_properties_l685_68587

noncomputable def game_pass_prob (n : ℕ) (p : ℝ) : ℝ := 1 - (1 - p)^n

noncomputable def hypergeometric_expectation (r w k : ℕ) : ℝ := (r * k : ℝ) / (r + w : ℝ)

noncomputable def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

theorem game_and_distributions_properties :
  (game_pass_prob 5 (1/2) = 31/32) ∧
  (hypergeometric_expectation 3 2 3 = 9/5) ∧
  (∀ k : ℕ, k ≤ 10 → binomial_pmf 10 0.6 k ≤ binomial_pmf 10 0.6 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_and_distributions_properties_l685_68587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_true_l685_68564

-- Define the propositions
def prop1 : Prop := ∀ a b c : ℝ, a > b → a*c^2 > b*c^2

def prop2 : Prop :=
  ∃ F₁ F₂ A B : ℝ × ℝ,
    (F₁.1^2/16 + F₁.2^2/25 = 1 ∧ F₂.1^2/16 + F₂.2^2/25 = 1) →
    (∃ t : ℝ, A = F₁ + t • (B - F₁)) →
    Real.sqrt ((A.1 - F₂.1)^2 + (A.2 - F₂.2)^2) +
    Real.sqrt ((B.1 - F₂.1)^2 + (B.2 - F₂.2)^2) +
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 20

def prop3 : Prop := ∀ p q : Prop, (¬p ∧ (p ∨ q)) → q

def prop4 : Prop := (¬∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0)

-- Theorem statement
theorem exactly_three_true : (¬prop1 ∧ prop2 ∧ prop3 ∧ prop4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_true_l685_68564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_measurements_sufficient_l685_68526

/-- Represents the calibration plate with 15 holes -/
structure CalibrationPlate where
  num_holes : Nat
  first_hole_diameter : ℝ
  diameter_increment : ℝ

/-- Represents a roller to be calibrated -/
structure Roller where
  diameter : ℝ

/-- Function to calculate the diameter of the nth hole -/
def hole_diameter (plate : CalibrationPlate) (n : Nat) : ℝ :=
  plate.first_hole_diameter + (n - 1 : ℝ) * plate.diameter_increment

/-- Function to determine if a roller fits in a hole -/
noncomputable def fits_in_hole (roller : Roller) (hole_diam : ℝ) : Prop :=
  roller.diameter ≤ hole_diam

/-- Theorem stating that 4 measurements are sufficient for calibration -/
theorem four_measurements_sufficient (plate : CalibrationPlate) (roller : Roller) :
  plate.num_holes = 15 ∧ 
  plate.first_hole_diameter = 10 ∧ 
  plate.diameter_increment = 0.04 ∧ 
  10 ≤ roller.diameter ∧ 
  roller.diameter ≤ 10.56 →
  ∃ (measurements : Fin 4 → Nat), 
    ∀ (i : Fin 4), 1 ≤ measurements i ∧ measurements i ≤ 15 ∧
    (∀ (j : Nat), 1 ≤ j ∧ j ≤ 15 → 
      |roller.diameter - hole_diameter plate j| < 0.04 ↔ 
      (∀ (k : Fin 4), fits_in_hole roller (hole_diameter plate (measurements k)) ↔ 
                      fits_in_hole roller (hole_diameter plate j))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_measurements_sufficient_l685_68526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_number_of_points_l685_68585

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A set of n+1 points on a plane -/
def PointSet (n : ℕ) := Fin (n + 1) → Point

/-- Predicate to check if three points are collinear -/
def collinear (p q r : Point) : Prop := sorry

/-- Predicate to check if a point is inside a triangle -/
def insideTriangle (p q r s : Point) : Prop := sorry

/-- The main theorem -/
theorem odd_number_of_points (n : ℕ) (points : PointSet n) (Q : Point) :
  (∀ i j k, i ≠ j → j ≠ k → i ≠ k → ¬collinear (points i) (points j) (points k)) →
  (∀ i j, i ≠ j → ∃ k, k ≠ i ∧ k ≠ j ∧ insideTriangle (points i) (points j) (points k) Q) →
  Odd n :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_number_of_points_l685_68585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_black_probability_l685_68563

def grid_size : Nat := 4

def is_center_square (i j : Fin grid_size) : Prop :=
  i.val ∈ [1, 2] ∧ j.val ∈ [1, 2]

def is_opposite_pair (i₁ j₁ i₂ j₂ : Fin grid_size) : Prop :=
  (i₁.val + i₂.val = 3) ∧ (j₁.val + j₂.val = 3) ∧ ¬(is_center_square i₁ j₁) ∧ ¬(is_center_square i₂ j₂)

def num_opposite_pairs : Nat := 6

def prob_center_black : ℚ := (1 / 2) ^ 4

def prob_opposite_pair_black : ℚ := 1 / 2

theorem grid_black_probability :
  (prob_center_black * prob_opposite_pair_black ^ num_opposite_pairs : ℚ) = 1 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_black_probability_l685_68563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_of_three_numbers_l685_68511

theorem max_of_three_numbers (p q r : ℝ) 
  (sum_eq : p + q + r = 1)
  (sum_prod_eq : p * q + p * r + q * r = -1)
  (prod_eq : p * q * r = 2) :
  max p (max q r) = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_of_three_numbers_l685_68511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_T_is_triangle_l685_68519

/-- The set T defined by the given conditions forms a triangle -/
theorem set_T_is_triangle (b : ℝ) (hb : b > 0) : ∃ (A B C : ℝ × ℝ),
  let T := {p : ℝ × ℝ | 
    b ≤ p.1 ∧ p.1 ≤ 3*b ∧
    b ≤ p.2 ∧ p.2 ≤ 3*b ∧
    p.1 + p.2 ≥ 2*b ∧
    p.1 + 2*b ≥ 2*p.2 ∧
    p.2 + 2*b ≥ 2*p.1}
  A ∈ T ∧ B ∈ T ∧ C ∈ T ∧
  (∀ p ∈ T, p = A ∨ p = B ∨ p = C ∨
    (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
      (p = (1-t) • A + t • B ∨
       p = (1-t) • B + t • C ∨
       p = (1-t) • C + t • A))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_T_is_triangle_l685_68519
