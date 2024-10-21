import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_circle_radius_l91_9170

/-- The radius of a circle centered at a focus of an ellipse and tangent to it --/
theorem ellipse_tangent_circle_radius (a b : ℝ) (ha : a = 7) (hb : b = 6) :
  let c := Real.sqrt (a^2 - b^2)
  let f : ℝ × ℝ := (c, 0)
  let r := Real.sqrt 26
  let ellipse := fun (p : ℝ × ℝ) ↦ (p.1^2 / a^2) + (p.2^2 / b^2) = 1
  let circle := fun (p : ℝ × ℝ) ↦ (p.1 - c)^2 + p.2^2 = r^2
  (∀ p, ellipse p → ¬circle p) ∧
  (∃ p, ellipse p ∧ circle p) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_circle_radius_l91_9170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_blown_westward_l91_9156

/-- A ship's journey with an eastward travel and westward storm --/
structure ShipJourney where
  initial_speed : ℝ
  initial_time : ℝ
  initial_fraction : ℝ
  final_fraction : ℝ

/-- Calculate the distance blown westward by the storm --/
noncomputable def distance_blown (journey : ShipJourney) : ℝ :=
  let initial_distance := journey.initial_speed * journey.initial_time
  let total_distance := initial_distance / journey.initial_fraction
  let final_distance := total_distance * journey.final_fraction
  initial_distance - final_distance

/-- Theorem stating the distance blown westward for the given journey --/
theorem distance_blown_westward :
  let journey : ShipJourney := {
    initial_speed := 30,
    initial_time := 20,
    initial_fraction := 1/2,
    final_fraction := 1/3
  }
  distance_blown journey = 200 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_blown_westward_l91_9156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_lateral_surface_area_l91_9102

/-- Represents a frustum of a regular triangular pyramid -/
structure Frustum where
  topEdge : ℝ
  bottomEdge : ℝ
  height : ℝ

/-- Calculates the lateral surface area of a frustum -/
noncomputable def lateralSurfaceArea (f : Frustum) : ℝ :=
  3 * ((f.topEdge + f.bottomEdge) / 2) * Real.sqrt (f.height^2 + ((f.bottomEdge - f.topEdge) / 2)^2)

theorem frustum_lateral_surface_area :
  let f : Frustum := { topEdge := 3, bottomEdge := 6, height := 3/2 }
  lateralSurfaceArea f = (27 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_lateral_surface_area_l91_9102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_phi_value_l91_9124

theorem tan_phi_value (φ : Real) 
  (h1 : Real.sin (π / 2 + φ) = 1 / 2) 
  (h2 : 0 < φ) 
  (h3 : φ < π) : 
  Real.tan φ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_phi_value_l91_9124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l91_9144

def sequenceA (n : ℕ+) : ℚ :=
  (-1:ℚ)^(n.val - 1) * (2*n.val + 1 : ℚ) / (n.val^2 + 2*n.val : ℚ)

theorem sequence_formula : 
  ∀ n : ℕ+, 
    (n = 1 → sequenceA n = 1) ∧ 
    (n = 2 → sequenceA n = -5/8) ∧ 
    (n = 3 → sequenceA n = 7/15) ∧ 
    (n = 4 → sequenceA n = -9/24) := by
  intro n
  have h1 : sequenceA 1 = 1 := by
    simp [sequenceA]
    norm_num
  have h2 : sequenceA 2 = -5/8 := by
    simp [sequenceA]
    norm_num
  have h3 : sequenceA 3 = 7/15 := by
    simp [sequenceA]
    norm_num
  have h4 : sequenceA 4 = -9/24 := by
    simp [sequenceA]
    norm_num
  exact ⟨λ h => h ▸ h1, λ h => h ▸ h2, λ h => h ▸ h3, λ h => h ▸ h4⟩

#check sequence_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l91_9144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_concentration_formula_l91_9184

/-- Represents the alcohol concentration of a mixture -/
noncomputable def alcohol_concentration (alcohol_volume : ℝ) (total_volume : ℝ) : ℝ :=
  alcohol_volume / total_volume

/-- Calculates the final alcohol concentration of a mixture -/
noncomputable def final_concentration (x y : ℝ) : ℝ :=
  let vessel_a_volume := (2 : ℝ)
  let vessel_a_concentration := (0.3 : ℝ)
  let vessel_b_volume := (6 : ℝ)
  let vessel_b_concentration := (0.45 : ℝ)
  let total_volume := vessel_a_volume + vessel_b_volume + x
  let total_alcohol := vessel_a_volume * vessel_a_concentration +
                       vessel_b_volume * vessel_b_concentration +
                       x * (y / 100)
  alcohol_concentration total_alcohol total_volume

theorem final_concentration_formula (x y : ℝ) :
  final_concentration x y = (0.6 + 2.7 + x * (y / 100)) / (8 + x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_concentration_formula_l91_9184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_days_2006_to_2010_l91_9101

def is_leap_year (year : ℕ) : Bool :=
  year = 2008

def days_in_year (year : ℕ) : ℕ :=
  if is_leap_year year then 366 else 365

def total_days (start_year end_year : ℕ) : ℕ :=
  List.range (end_year - start_year + 1)
  |>.map (fun i => days_in_year (start_year + i))
  |>.sum

theorem total_days_2006_to_2010 :
  total_days 2006 2010 = 1826 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_days_2006_to_2010_l91_9101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_interval_range_l91_9150

/-- The function f(x) = 1/3 * x^3 + x^2 - 2/3 -/
noncomputable def f (x : ℝ) : ℝ := 1/3 * x^3 + x^2 - 2/3

/-- The theorem stating the range of a given the conditions -/
theorem max_value_interval_range (a : ℝ) :
  (∃ x ∈ Set.Ioo a (a + 4), ∀ y ∈ Set.Ioo a (a + 4), f y ≤ f x) →
  -6 < a ∧ a ≤ -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_interval_range_l91_9150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_fourth_plus_alpha_l91_9134

theorem cos_pi_fourth_plus_alpha (α : Real) 
  (h1 : Real.sin α = -3/5) 
  (h2 : α ∈ Set.Icc (3*Real.pi/2) (2*Real.pi)) : 
  Real.cos (Real.pi/4 + α) = (7 * Real.sqrt 2) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_fourth_plus_alpha_l91_9134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_from_f_to_l_l91_9169

/-- The function f(x) = e^(2x) --/
noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x)

/-- The line l: y = 2x --/
def l (x : ℝ) : ℝ := 2 * x

/-- The minimum distance from a point on f to the line l --/
noncomputable def min_distance : ℝ := Real.sqrt 5 / 5

/-- Theorem stating that the minimum distance from any point on f to l is √5/5 --/
theorem min_distance_from_f_to_l :
  ∀ x : ℝ, ∃ d : ℝ, d ≥ 0 ∧ 
    d = |f x - l x| / Real.sqrt (1 + 2^2) ∧
    d ≥ min_distance ∧
    (∃ x₀ : ℝ, |f x₀ - l x₀| / Real.sqrt (1 + 2^2) = min_distance) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_from_f_to_l_l91_9169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_pairs_count_l91_9157

def U : Finset Nat := {1, 2, 3, 4}

theorem subset_pairs_count :
  let pairs := Finset.filter (fun p : Finset Nat × Finset Nat =>
    p.1.Nonempty ∧ p.2.Nonempty ∧ p.1 ⊆ U ∧ p.2 ⊆ U ∧ p.1 ∩ p.2 = ∅)
    (Finset.product (Finset.powerset U) (Finset.powerset U))
  Finset.card pairs = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_pairs_count_l91_9157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_sum_l91_9137

-- Define points A and B
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (2, 1)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem max_distance_sum (M : ℝ × ℝ) :
  (∃ k : ℝ, M.1 + k * M.2 = 0 ∧ k * M.1 - M.2 - 2*k + 1 = 0) →
  distance M A + distance M B ≤ Real.sqrt 10 := by
  sorry

#check max_distance_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_sum_l91_9137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_iff_a_eq_plus_minus_one_l91_9138

/-- The function f(x) = ln(√(x^2 + 1) - ax) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + 1) - a*x)

/-- Theorem: f is an odd function if and only if a = 1 or a = -1 -/
theorem f_odd_iff_a_eq_plus_minus_one (a : ℝ) :
  (∀ x, f a (-x) = -f a x) ↔ (a = 1 ∨ a = -1) := by
  sorry

#check f_odd_iff_a_eq_plus_minus_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_iff_a_eq_plus_minus_one_l91_9138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_theorem_l91_9140

/-- The hyperbola defined by x^2 - y^2 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

/-- A point on the left branch of the hyperbola -/
def left_branch (p : ℝ × ℝ) : Prop :=
  hyperbola p.1 p.2 ∧ p.1 < 0

/-- A point on the right branch of the hyperbola -/
def right_branch (p : ℝ × ℝ) : Prop :=
  hyperbola p.1 p.2 ∧ p.1 > 0

/-- The line passing through two points -/
def line_through (p q : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - p.2) * (q.1 - p.1) = (x - p.1) * (q.2 - p.2)

/-- The intersection point of two lines -/
noncomputable def intersection_point (l1 l2 : ℝ → ℝ → Prop) : ℝ × ℝ :=
  sorry

theorem hyperbola_intersection_theorem
  (A B C : ℝ × ℝ)
  (hA : A = (-1, 0))
  (hB : B = (1, 0))
  (hC : C = (2, 0))
  (D : ℝ × ℝ)
  (hD : left_branch D)
  (hD_ne_A : D ≠ A)
  (E : ℝ × ℝ)
  (hE : right_branch E)
  (hCDE : line_through C D E.1 E.2)
  (P : ℝ × ℝ)
  (hP : P = intersection_point (line_through A D) (line_through B E)) :
  P.1 = 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_theorem_l91_9140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_fuel_efficiency_approx_l91_9130

/-- Represents the fuel efficiency of a car in kilometers per gallon. -/
noncomputable def fuel_efficiency (distance : ℝ) (fuel_used : ℝ) : ℝ :=
  distance / fuel_used

/-- Theorem stating that given a car that uses 6.666666666666667 gallons of gasoline
    to travel 200 kilometers, the car's fuel efficiency is approximately 30 kilometers per gallon. -/
theorem car_fuel_efficiency_approx :
  let distance := (200 : ℝ)
  let fuel_used := (6.666666666666667 : ℝ)
  abs (fuel_efficiency distance fuel_used - 30) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_fuel_efficiency_approx_l91_9130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_result_l91_9111

def initial_vector : Fin 3 → ℝ := ![2, 3, 1]

def rotation_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![-1, 0, 0],
    ![0, -1, 0],
    ![0, 0, 1]]

def rotated_vector : Fin 3 → ℝ := Matrix.mulVec rotation_matrix initial_vector

theorem rotation_result :
  rotated_vector = ![-2, -3, 1] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_result_l91_9111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disneyland_attractions_l91_9116

theorem disneyland_attractions (n : ℕ) (h : n = 5) :
  Nat.factorial (n + 1) / 2 = 360 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_disneyland_attractions_l91_9116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_corner_sum_l91_9142

/-- Represents a face of the cube -/
def Face := Fin 6

/-- The value on a face of the cube -/
def face_value : Face → ℕ := sorry

/-- The sum of opposite faces is 8 -/
axiom opposite_sum (f : Face) : ∃ g : Face, f ≠ g ∧ (face_value f + face_value g = 8)

/-- The face value is between 1 and 7 inclusive -/
axiom face_value_range (f : Face) : 1 ≤ face_value f ∧ face_value f ≤ 7

/-- Three faces meet at a corner -/
def corner_faces : Set (Face × Face × Face) :=
  {c | c.1 ≠ c.2.1 ∧ c.1 ≠ c.2.2 ∧ c.2.1 ≠ c.2.2}

/-- The sum of face values at a corner -/
def corner_sum (c : Face × Face × Face) : ℕ :=
  face_value c.1 + face_value c.2.1 + face_value c.2.2

/-- The maximum sum of face values at any corner is 17 -/
theorem max_corner_sum :
  ∃ c ∈ corner_faces, ∀ c' ∈ corner_faces, corner_sum c' ≤ corner_sum c ∧ corner_sum c = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_corner_sum_l91_9142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_l91_9128

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define a line with slope -2 and y-intercept c
def my_line (x y c : ℝ) : Prop := 2*x + y + c = 0

-- Define what it means for a line to be tangent to the circle
def is_tangent (c : ℝ) : Prop :=
  ∃ (x y : ℝ), my_circle x y ∧ my_line x y c ∧
  ∀ (x' y' : ℝ), my_circle x' y' ∧ my_line x' y' c → (x', y') = (x, y)

-- The main theorem
theorem tangent_lines :
  ∀ c : ℝ, is_tangent c ↔ c = 5 ∨ c = -5 :=
by
  sorry

#check tangent_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_l91_9128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_smallest_a_l91_9103

theorem parabola_smallest_a (a b c : ℝ) (n : ℤ) :
  let vertex_x : ℝ := -1/3
  let vertex_y : ℝ := -4/3
  a > 0 ∧
  (a + b + c = n) ∧
  (∀ x, a*x^2 + b*x + c = a*(x - vertex_x)^2 + vertex_y) →
  a ≥ 3/16 :=
by
  intros vertex_x vertex_y h
  -- The proof steps would go here
  sorry

#check parabola_smallest_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_smallest_a_l91_9103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_1999_2001_closest_to_2000_l91_9162

noncomputable def harmonic_mean (a b : ℝ) : ℝ := 2 * a * b / (a + b)

theorem harmonic_mean_1999_2001_closest_to_2000 :
  let hm := harmonic_mean 1999 2001
  (|hm - 2000| < |hm - 1999|) ∧
  (|hm - 2000| < |hm - 2001|) ∧
  (|hm - 2000| < |hm - 2002|) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_1999_2001_closest_to_2000_l91_9162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_4_l91_9131

-- Define the function f as noncomputable
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x ^ α

-- State the theorem
theorem f_value_at_4 (α : ℝ) :
  f α 2 = Real.sqrt 2 / 2 → f α 4 = 1 / 2 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_4_l91_9131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shape_is_flat_disc_l91_9110

/-- A shape in spherical coordinates -/
structure SphericalShape where
  ρ : ℝ → ℝ → ℝ → ℝ
  θ : ℝ → ℝ → ℝ → ℝ
  φ : ℝ → ℝ → ℝ → ℝ

/-- A flat disc in 3D space -/
structure FlatDisc where
  center : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ
  radius : ℝ

/-- The shape described by the given equations in spherical coordinates -/
noncomputable def givenShape (c : ℝ) : SphericalShape where
  ρ := fun _ _ _ => c
  θ := fun _ _ _ => 0 -- θ is not constrained, so we set it to 0
  φ := fun _ _ _ => Real.pi / 4

theorem shape_is_flat_disc (c : ℝ) (h : c > 0) : 
  ∃ (d : FlatDisc), true := by
  sorry

#check shape_is_flat_disc

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shape_is_flat_disc_l91_9110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_2018_l91_9125

-- Define the sequence a_n
def a : ℕ → ℝ := sorry

-- Define the sum S_n
def S : ℕ → ℝ
  | 0 => 0
  | n + 1 => S n + a (n + 1)

-- State the theorem
theorem sequence_sum_2018 (h1 : a 1 = 1) 
  (h2 : ∀ n ≥ 2, a n + 2 * S (n - 1) = n) : 
  S 2018 = 1009 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_2018_l91_9125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_solution_l91_9120

-- Define the first right triangle
def triangle1 : ℚ × ℚ := (12, 9)

-- Define the second right triangle with unknown leg x
def triangle2 (x : ℚ) : ℚ × ℚ := (x, 6)

-- Define the similarity ratio between the triangles
noncomputable def similarity_ratio (x : ℚ) : ℚ := (12 / x)

-- Theorem statement
theorem similar_triangles_solution (x : ℚ) :
  (similarity_ratio x = 9 / 6) →
  (x = 8 ∧ Real.sqrt ((x : ℝ)^2 + 6^2) = 10) := by
  sorry

#check similar_triangles_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_solution_l91_9120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l91_9194

theorem negation_of_proposition :
  (¬∀ x : ℝ, x^2 + |x| ≥ 0) ↔ (∃ x : ℝ, x^2 + |x| < 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l91_9194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_digits_l91_9186

def a : Nat := 3659893456789325678
def b : Nat := 342973489379256

theorem product_digits : (String.length (toString (a * b))) = 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_digits_l91_9186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_c_fastest_l91_9198

/-- Represents a boat with its speed in still water and the current rate it experiences -/
structure Boat where
  stillSpeed : ℚ
  currentRate : ℚ

/-- Calculates the effective speed of a boat moving downstream -/
def effectiveSpeed (b : Boat) : ℚ := b.stillSpeed + b.currentRate

/-- Calculates the time taken by a boat to travel a given distance -/
def travelTime (b : Boat) (distance : ℚ) : ℚ := distance / effectiveSpeed b

theorem boat_c_fastest (distanceDownstream : ℚ) : 
  distanceDownstream = 60 →
  let boatA : Boat := { stillSpeed := 42, currentRate := 4 }
  let boatB : Boat := { stillSpeed := 36, currentRate := 5 }
  let boatC : Boat := { stillSpeed := 48, currentRate := 6 }
  travelTime boatC distanceDownstream < travelTime boatA distanceDownstream ∧
  travelTime boatC distanceDownstream < travelTime boatB distanceDownstream :=
by
  sorry

#check boat_c_fastest

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_c_fastest_l91_9198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_range_l91_9151

/-- Given vectors a and b, prove that the magnitude of 2a - b is between 0 and 4 inclusive. -/
theorem magnitude_range (θ : ℝ) : 
  let a : Fin 2 → ℝ := ![Real.cos θ, Real.sin θ]
  let b : Fin 2 → ℝ := ![Real.sqrt 3, -1]
  0 ≤ ‖2 • a - b‖ ∧ ‖2 • a - b‖ ≤ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_range_l91_9151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_equals_two_thirds_c_squared_l91_9119

-- Define a right-angled triangle ABC
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define the hypotenuse length
noncomputable def hypotenuse_length (t : RightTriangle) : ℝ :=
  Real.sqrt ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2)

-- Define points C₁ and C₂ that divide AB into three equal parts
noncomputable def C₁ (t : RightTriangle) : ℝ × ℝ :=
  ((2 * t.A.1 + t.B.1) / 3, (2 * t.A.2 + t.B.2) / 3)

noncomputable def C₂ (t : RightTriangle) : ℝ × ℝ :=
  ((t.A.1 + 2 * t.B.1) / 3, (t.A.2 + 2 * t.B.2) / 3)

-- Define the distances CC₁, C₁C₂, and C₂C
noncomputable def CC₁_squared (t : RightTriangle) : ℝ :=
  (t.C.1 - (C₁ t).1)^2 + (t.C.2 - (C₁ t).2)^2

noncomputable def C₁C₂_squared (t : RightTriangle) : ℝ :=
  ((C₂ t).1 - (C₁ t).1)^2 + ((C₂ t).2 - (C₁ t).2)^2

noncomputable def C₂C_squared (t : RightTriangle) : ℝ :=
  (t.C.1 - (C₂ t).1)^2 + (t.C.2 - (C₂ t).2)^2

-- State the theorem
theorem sum_of_squares_equals_two_thirds_c_squared (t : RightTriangle) :
  CC₁_squared t + C₁C₂_squared t + C₂C_squared t = (2/3) * (hypotenuse_length t)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_equals_two_thirds_c_squared_l91_9119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_initial_height_l91_9167

/-- The initial height of a tree, given the growth rates of a boy and the tree,
    and their heights at two different points in time. -/
theorem tree_initial_height
  (boy_initial_height boy_later_height tree_later_height tree_initial_height : ℝ)
  (h1 : boy_initial_height = 24)
  (h2 : boy_later_height = 36)
  (h3 : tree_later_height = 40)
  (h4 : (tree_later_height - tree_initial_height) = 2 * (boy_later_height - boy_initial_height)) :
  tree_initial_height = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_initial_height_l91_9167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_constant_sum_l91_9197

-- Define the ellipse C
structure Ellipse :=
  (center : ℝ × ℝ)
  (semi_major_axis : ℝ)
  (semi_minor_axis : ℝ)
  (eccentricity : ℝ)

-- Define the line l
structure Line :=
  (slope : ℝ)
  (y_intercept : ℝ)

-- Define points A, B, M, and F
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the theorem
theorem ellipse_intersection_constant_sum
  (C : Ellipse)
  (l : Line)
  (A B M F : Point)
  (lambda1 lambda2 : ℝ)
  (h_center : C.center = (0, 0))
  (h_minor : C.semi_minor_axis = 1)
  (h_ecc : C.eccentricity = 2 * Real.sqrt 5 / 5)
  (h_foci : F.y = 0)
  (h_line : l.slope * (A.x - F.x) = A.y ∧ l.slope * (B.x - F.x) = B.y ∧ l.slope * (-F.x) = M.y)
  (h_intersect : A.x^2 / C.semi_major_axis^2 + A.y^2 / C.semi_minor_axis^2 = 1 ∧
                 B.x^2 / C.semi_major_axis^2 + B.y^2 / C.semi_minor_axis^2 = 1)
  (h_lambda1 : M.x - A.x = lambda1 * (A.x - F.x) ∧ M.y - A.y = lambda1 * (A.y - F.y))
  (h_lambda2 : M.x - B.x = lambda2 * (B.x - F.x) ∧ M.y - B.y = lambda2 * (B.y - F.y))
  : lambda1 + lambda2 = -10 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_constant_sum_l91_9197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_downstream_distance_l91_9121

/-- Proves that a boat traveling upstream and downstream with given speeds and distances will cover 64 km downstream -/
theorem boat_downstream_distance 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (upstream_distance : ℝ) 
  (downstream_distance : ℝ)
  (h1 : boat_speed = 18) 
  (h2 : stream_speed = 6) 
  (h3 : upstream_distance = 32) 
  (h4 : (upstream_distance / (boat_speed - stream_speed)) = 
        (downstream_distance / (boat_speed + stream_speed))) : 
  downstream_distance = 64 := by
  sorry

#check boat_downstream_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_downstream_distance_l91_9121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_female_percentage_l91_9139

/-- Given a company's workforce with an initial unknown number of employees,
    prove that after hiring 26 additional male workers, resulting in a total
    of 312 employees with 55% female workers, the initial percentage of
    female workers was approximately 60.14%. -/
theorem initial_female_percentage
  (initial_employees : ℕ)
  (hired_males : ℕ)
  (total_employees : ℕ)
  (final_female_percentage : ℚ)
  (h1 : hired_males = 26)
  (h2 : total_employees = initial_employees + hired_males)
  (h3 : total_employees = 312)
  (h4 : final_female_percentage = 55/100) :
  ∃ (initial_female_percentage : ℚ),
    abs (initial_female_percentage - 6014/10000) < 1/1000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_female_percentage_l91_9139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tim_expenditure_l91_9154

/-- Calculates Tim's total expenditure at the supermarket with discounts and a coupon -/
theorem tim_expenditure 
  (apple_price milk_price milk_discount pineapple_price flour_price chocolate_price coupon_value : ℝ) :
  apple_price = 1 →
  milk_price = 3 →
  milk_discount = 1 →
  pineapple_price = 4 →
  flour_price = 6 →
  chocolate_price = 10 →
  coupon_value = 10 →
  (8 * apple_price + 
   4 * (milk_price - milk_discount) + 
   3 * (pineapple_price / 2) + 
   3 * flour_price + 
   chocolate_price) - 
  (if 8 * apple_price + 
      4 * (milk_price - milk_discount) + 
      3 * (pineapple_price / 2) + 
      3 * flour_price + 
      chocolate_price >= 50 
   then coupon_value 
   else 0) = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tim_expenditure_l91_9154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_lengths_l91_9143

-- Define the triangle and its properties
noncomputable def Triangle (a b c r : ℝ) : Prop :=
  ∃ (k : ℝ), 
    a = 25 * k ∧
    b = 29 * k ∧
    c = 36 * k ∧
    r = 232 ∧
    k > 0

-- Define the semiperimeter
noncomputable def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

-- State the theorem
theorem triangle_side_lengths 
  (a b c r : ℝ) 
  (h : Triangle a b c r) : 
  a = 725 ∧ b = 841 ∧ c = 1044 := by
  sorry

#check triangle_side_lengths

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_lengths_l91_9143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rooks_theorem_l91_9155

/-- Represents a 10x10 chessboard -/
def Board := Fin 10 → Fin 10 → Bool

/-- Checks if a cell is under attack by a rook -/
def is_attacked (board : Board) (row col : Fin 10) : Prop :=
  ∃ (r c : Fin 10), board r c ∧ (r = row ∨ c = col)

/-- Counts the number of rooks on the board -/
def count_rooks (board : Board) : Nat :=
  (Finset.sum (Finset.univ : Finset (Fin 10)) fun r =>
    Finset.sum (Finset.univ : Finset (Fin 10)) fun c =>
      if board r c then 1 else 0)

/-- Checks if the board satisfies the condition after removing any rook -/
def satisfies_condition (board : Board) : Prop :=
  ∀ (row col : Fin 10), board row col →
    ∃ (r c : Fin 10), is_attacked board r c ∧
      ¬is_attacked (fun r' c' => if r' = row ∧ c' = col then False else board r' c') r c

/-- The main theorem stating the maximum number of rooks -/
theorem max_rooks_theorem :
  ∃ (board : Board),
    satisfies_condition board ∧
    count_rooks board = 16 ∧
    ∀ (board' : Board),
      satisfies_condition board' →
      count_rooks board' ≤ 16 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rooks_theorem_l91_9155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_five_l91_9187

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + (h.b / h.a)^2)

/-- Theorem: If the distance from the foci to the asymptotes is equal to the length
    of the real axis, then the eccentricity of the hyperbola is √5 -/
theorem hyperbola_eccentricity_sqrt_five (h : Hyperbola) 
    (h_foci_asymptote : h.b = 2 * h.a) : eccentricity h = Real.sqrt 5 := by
  -- Unfold the definition of eccentricity
  unfold eccentricity
  -- Replace h.b with 2 * h.a
  rw [h_foci_asymptote]
  -- Simplify the expression
  simp [pow_two]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_five_l91_9187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_S_l91_9165

-- Define the solid S
def S : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | let (x, y, z) := p
                   (|x| + |y| + |z| ≤ 1.5) ∧
                   (|x| + |y| ≤ 1) ∧
                   (|z| ≤ 0.5)}

-- State the theorem
theorem volume_of_S : MeasureTheory.volume S = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_S_l91_9165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_equality_condition_l91_9126

theorem min_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  a + Real.sqrt (a * b) + (a * b * c) ^ (1/3 : ℝ) ≥ 3 :=
by sorry

theorem equality_condition (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  a + Real.sqrt (a * b) + (a * b * c) ^ (1/3 : ℝ) = 3 ↔ a = 1 ∧ b = 1 ∧ c = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_equality_condition_l91_9126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l91_9158

/-- The probability distribution for a random variable ξ -/
noncomputable def P (k : ℕ) (c : ℝ) : ℝ := c / (k * (k + 1))

/-- The sum of probabilities for k = 1 to 4 equals 1 -/
axiom prob_sum_one (c : ℝ) : 
  (P 1 c) + (P 2 c) + (P 3 c) + (P 4 c) = 1

/-- The probability that ξ is between 1/2 and 5/2 -/
noncomputable def prob_between (c : ℝ) : ℝ := (P 1 c) + (P 2 c)

theorem probability_theorem : 
  ∃ c : ℝ, prob_between c = 5/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l91_9158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_conditions_l91_9178

/-- Given vectors in ℝ² -/
def a : ℝ × ℝ := (-2, 5)
def b : ℝ × ℝ := (1, -1)
def c (l : ℝ) : ℝ × ℝ := (6, l)

/-- Two vectors are parallel if their cross product is zero -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

/-- The dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Two vectors are perpendicular if their dot product is zero -/
def perpendicular (v w : ℝ × ℝ) : Prop :=
  dot_product v w = 0

/-- Main theorem -/
theorem vector_conditions (l : ℝ) :
  (parallel ((-3, 9)) (c l) → l = -18) ∧
  (perpendicular ((-5, 8)) (c l) → l = 15/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_conditions_l91_9178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_distinct_roots_condition_l91_9104

noncomputable def f (x : ℝ) : ℝ := |x * Real.exp x|

theorem two_distinct_roots_condition (t : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ ^ 2 - 2 * t * f x₁ + 3 = 0 ∧ f x₂ ^ 2 - 2 * t * f x₂ + 3 = 0) ↔
  (t > Real.sqrt 3 ∧ t < (1 + 3 * Real.exp 2) / (2 * Real.exp 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_distinct_roots_condition_l91_9104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l91_9107

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (4 - Real.sqrt (5 - Real.sqrt (6 - x)))

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Icc (-115 : ℝ) 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l91_9107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_integer_function_characterization_l91_9190

/-- The set of odd integers -/
def OddIntegers : Set ℤ := {n : ℤ | ∃ k : ℤ, n = 2 * k + 1}

/-- The functional equation condition -/
def SatisfiesEquation (f : ℤ → ℤ) : Prop :=
  ∀ x y : ℤ, f (x + f x + y) + f (x - f x - y) = f (x + y) + f (x - y)

/-- The theorem statement -/
theorem odd_integer_function_characterization
  (f : ℤ → ℤ)
  (h_odd : ∀ x, f x ∈ OddIntegers)
  (h_eq : SatisfiesEquation f) :
  ∃ (d k : ℤ) (l : Fin (d.natAbs) → ℤ),
    d ≠ 0 ∧
    (∀ i : Fin (d.natAbs), Odd (l i)) ∧
    ∀ (m : ℤ) (i : Fin (d.natAbs)),
      f (m * d + i) = 2 * k * m * d + l i :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_integer_function_characterization_l91_9190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_normal_equations_l91_9171

noncomputable def curve (t : ℝ) : ℝ × ℝ := (3 * Real.cos t, 4 * Real.sin t)

noncomputable def t₀ : ℝ := Real.pi / 4

noncomputable def tangent_slope (t : ℝ) : ℝ := 
  -4 / 3 * (Real.cos t / Real.sin t)

noncomputable def normal_slope (t : ℝ) : ℝ := 
  -1 / tangent_slope t

theorem tangent_and_normal_equations :
  let (x₀, y₀) := curve t₀
  (∀ x y, y = tangent_slope t₀ * (x - x₀) + y₀ ↔ 
    y = -4/3 * x + 4 * Real.sqrt 2) ∧
  (∀ x y, y = normal_slope t₀ * (x - x₀) + y₀ ↔ 
    y = 3/4 * x + 7 * Real.sqrt 2 / 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_normal_equations_l91_9171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_volume_l91_9189

/-- The side length of the squares used to form the polyhedron -/
def square_side : ℝ := 12

/-- A square with side length 12 cm -/
structure Square where
  side : ℝ
  is_twelve : side = square_side

/-- A part of a square, formed by connecting midpoints of adjacent sides -/
structure SquarePart where
  square : Square
  is_half : area = square.side^2 / 2

/-- A regular hexagon -/
structure RegularHexagon

/-- Predicate indicating that a shape is a polyhedron -/
def IsPolyhedron : Type := Unit

/-- The polyhedron formed by folding three squares divided into two parts each -/
structure Polyhedron where
  squares : Fin 3 → Square
  parts : Fin 3 → Fin 2 → SquarePart
  hexagon : RegularHexagon
  folded : IsPolyhedron

/-- The volume of the polyhedron -/
noncomputable def volume (p : Polyhedron) : ℝ := sorry

/-- Theorem stating that the volume of the polyhedron is 864 cm³ -/
theorem polyhedron_volume (p : Polyhedron) : volume p = 864 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_volume_l91_9189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_a_profit_margin_l91_9174

/-- Represents the profit margin of a store selling mobile phones -/
noncomputable def ProfitMargin (purchase_price selling_price : ℝ) : ℝ :=
  (selling_price - purchase_price) / purchase_price

theorem store_a_profit_margin 
  (purchase_price_a selling_price : ℝ) 
  (purchase_price_a_positive : purchase_price_a > 0)
  (selling_price_positive : selling_price > 0) :
  let purchase_price_b := 0.9 * purchase_price_a
  let profit_margin_a := ProfitMargin purchase_price_a selling_price
  let profit_margin_b := ProfitMargin purchase_price_b selling_price
  profit_margin_b = profit_margin_a + 0.12 →
  profit_margin_a = 0.08 := by
  sorry

#check store_a_profit_margin

end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_a_profit_margin_l91_9174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_repeating_decimal_to_hundredth_l91_9161

-- Define the repeating decimal 67.6767...
def repeating_decimal : ℚ := 67 + 67 / 99

-- Define a function to round a rational number to the nearest hundredth
noncomputable def round_to_hundredth (q : ℚ) : ℚ := 
  (q * 100).floor / 100 + if (q * 100 - (q * 100).floor ≥ 1/2) then 1/100 else 0

-- Theorem statement
theorem round_repeating_decimal_to_hundredth :
  round_to_hundredth repeating_decimal = 67 + 68 / 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_repeating_decimal_to_hundredth_l91_9161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toothpicks_10th_stage_l91_9168

def toothpicks : ℕ → ℕ
  | 0 => 5  -- Add this case for 0
  | 1 => 5
  | n+1 => toothpicks n + 3

theorem toothpicks_10th_stage : toothpicks 10 = 32 := by
  -- Unfold the definition and calculate step by step
  have h1 : toothpicks 1 = 5 := rfl
  have h2 : toothpicks 2 = 8 := rfl
  have h3 : toothpicks 3 = 11 := rfl
  have h4 : toothpicks 4 = 14 := rfl
  have h5 : toothpicks 5 = 17 := rfl
  have h6 : toothpicks 6 = 20 := rfl
  have h7 : toothpicks 7 = 23 := rfl
  have h8 : toothpicks 8 = 26 := rfl
  have h9 : toothpicks 9 = 29 := rfl
  have h10 : toothpicks 10 = 32 := rfl
  exact h10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toothpicks_10th_stage_l91_9168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scooter_gain_percent_l91_9191

/-- Calculate the gain percent for a scooter sale -/
theorem scooter_gain_percent 
  (purchase_price repair_cost selling_price : ℝ) 
  (hp : purchase_price > 0) 
  (hr : repair_cost ≥ 0) 
  (hs : selling_price > 0) : 
  ((selling_price - (purchase_price + repair_cost)) / (purchase_price + repair_cost)) * 100 =
  ((selling_price - (purchase_price + repair_cost)) / (purchase_price + repair_cost)) * 100 :=
by
  -- Define total_cost
  let total_cost := purchase_price + repair_cost
  -- Define gain
  let gain := selling_price - total_cost
  -- Define gain_percent
  let gain_percent := (gain / total_cost) * 100
  -- Prove the equality
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_scooter_gain_percent_l91_9191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_4500_simplification_l91_9172

theorem cube_root_4500_simplification :
  ∃ (a b : ℕ+), (a : ℝ) * ((b : ℝ) ^ (1/3)) = (4500 : ℝ) ^ (1/3) ∧
  b = 36 ∧ a = 5 ∧
  ∀ (c d : ℕ+), (c : ℝ) * ((d : ℝ) ^ (1/3)) = (4500 : ℝ) ^ (1/3) → d ≥ b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_4500_simplification_l91_9172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_point_equation_l91_9135

/-- A right-angled triangle with a point on its hypotenuse or its extension -/
structure RightTriangleWithPoint where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  P : ℝ × ℝ
  is_right_angled : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  P_on_AB : ∃ t : ℝ, P = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The main theorem -/
theorem right_triangle_point_equation (triangle : RightTriangleWithPoint) :
  let PA := distance triangle.P triangle.A
  let PB := distance triangle.P triangle.B
  let PC := distance triangle.P triangle.C
  let AB := distance triangle.A triangle.B
  let BC := distance triangle.B triangle.C
  let CA := distance triangle.C triangle.A
  (PA * BC)^2 + (PB * CA)^2 = (PC * AB)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_point_equation_l91_9135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_is_k_l91_9115

/-- The original function h(x) -/
noncomputable def h (x : ℝ) : ℝ := 3 - 7*x

/-- The proposed inverse function k(x) -/
noncomputable def k (x : ℝ) : ℝ := (3 - x) / 7

/-- Theorem stating that k is the inverse of h -/
theorem h_inverse_is_k : Function.LeftInverse k h ∧ Function.RightInverse k h := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_is_k_l91_9115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fabric_cutting_l91_9132

theorem fabric_cutting (fabric_A_width fabric_A_area fabric_B_width fabric_B_area square_width : ℝ)
  (h1 : fabric_A_width = 3)
  (h2 : fabric_A_area = 24)
  (h3 : fabric_B_width = 4)
  (h4 : fabric_B_area = 36)
  (h5 : square_width = 2) :
  fabric_A_area / fabric_A_width = 8 ∧
  (fabric_B_area - square_width ^ 2) / fabric_B_width = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fabric_cutting_l91_9132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_polynomial_with_many_unit_values_l91_9112

theorem no_polynomial_with_many_unit_values (d : ℕ) (hd : d > 3) :
  ¬ ∃ (P : Polynomial ℤ) (S : Finset ℤ),
    (Polynomial.degree P = d) ∧
    (S.card ≥ d + 1) ∧
    (∀ m ∈ S, |P.eval m| = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_polynomial_with_many_unit_values_l91_9112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chi_square_significant_r_equality_r_estimate_l91_9127

-- Define the contingency table
def contingency_table : Fin 2 → Fin 2 → ℕ
| 0, 0 => 30  -- Male, Like
| 0, 1 => 20  -- Male, Dislike
| 1, 0 => 10  -- Female, Like
| 1, 1 => 40  -- Female, Dislike

-- Define the chi-square function
def chi_square (a b c d : ℕ) : ℚ :=
  let n : ℕ := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the conditional probability
noncomputable def conditional_prob (A B : Fin 2 → Prop) [DecidablePred A] [DecidablePred B] : ℚ :=
  (Finset.filter (λ i => A i ∧ B i) (Finset.univ : Finset (Fin 2))).card /
  (Finset.filter B (Finset.univ : Finset (Fin 2))).card

-- Theorem statements
theorem chi_square_significant :
  chi_square 30 10 20 40 > 10.828 := by
  sorry

theorem r_equality :
  let A := λ i : Fin 2 => i = 0  -- Male
  let B := λ i : Fin 2 => contingency_table i 1 > 0  -- Dislike basketball
  (conditional_prob B A / conditional_prob (λ i => ¬ B i) A) /
  (conditional_prob B (λ i => ¬ A i) / conditional_prob (λ i => ¬ B i) (λ i => ¬ A i)) =
  (conditional_prob A B / conditional_prob (λ i => ¬ A i) B) *
  (conditional_prob (λ i => ¬ A i) (λ i => ¬ B i) / conditional_prob A (λ i => ¬ B i)) := by
  sorry

theorem r_estimate :
  let A := λ i : Fin 2 => i = 0  -- Male
  let B := λ i : Fin 2 => contingency_table i 1 > 0  -- Dislike basketball
  (conditional_prob A B / conditional_prob (λ i => ¬ A i) B) *
  (conditional_prob (λ i => ¬ A i) (λ i => ¬ B i) / conditional_prob A (λ i => ¬ B i)) = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chi_square_significant_r_equality_r_estimate_l91_9127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_selection_percentage_l91_9141

theorem exam_selection_percentage (total_candidates : ℕ) 
  (state_b_percentage : ℚ) (difference : ℕ) :
  total_candidates = 8100 →
  state_b_percentage = 7 / 100 →
  difference = 81 →
  (state_b_percentage * total_candidates - difference) / total_candidates = 6 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_selection_percentage_l91_9141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_abc_equals_20_l91_9105

-- Define the piecewise function f
noncomputable def f (a b c : ℕ) (x : ℝ) : ℝ :=
  if x > 0 then 2 * (a : ℝ) * x + 6
  else if x = 0 then (a * b : ℝ)
  else 3 * (b : ℝ) * x + (c : ℝ)

-- State the theorem
theorem sum_abc_equals_20 (a b c : ℕ) :
  f a b c 3 = 24 →
  f a b c 0 = 6 →
  f a b c (-3) = -33 →
  a + b + c = 20 := by
  sorry

#check sum_abc_equals_20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_abc_equals_20_l91_9105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l91_9164

def b : ℕ → ℝ
  | 0 => 2  -- Added case for 0
  | 1 => 2
  | 2 => 3
  | (n + 3) => b (n + 2) + b (n + 1)

theorem infinite_series_sum :
  (∑' n, b n / 3^(n + 1)) = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l91_9164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_zeros_difference_l91_9183

/-- Represents a quadratic function f(x) = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the y-coordinate for a given x-coordinate on the quadratic function -/
def QuadraticFunction.eval (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- Represents the vertex of a parabola -/
structure Vertex where
  x : ℝ
  y : ℝ

/-- Represents a point on the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Returns true if the quadratic function has the given vertex -/
def hasVertex (f : QuadraticFunction) (v : Vertex) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f.eval x = a * (x - v.x)^2 + v.y

/-- Returns true if the quadratic function passes through the given point -/
def passesThrough (f : QuadraticFunction) (p : Point) : Prop :=
  f.eval p.x = p.y

/-- Returns the zeros of the quadratic function -/
def zeros (f : QuadraticFunction) : Set ℝ :=
  {x | f.eval x = 0}

theorem parabola_zeros_difference (f : QuadraticFunction) (v : Vertex) (p : Point) :
  hasVertex f v →
  passesThrough f p →
  v.x = 3 ∧ v.y = -9 →
  p.x = 5 ∧ p.y = 7 →
  ∃ m n : ℝ, m ∈ zeros f ∧ n ∈ zeros f ∧ m > n ∧ m - n = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_zeros_difference_l91_9183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l91_9146

-- Define the hyperbola and its properties
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0

-- Define the points A, F, P, and Q
def A (h : Hyperbola) : ℝ × ℝ := (h.a, 0)

noncomputable def F (h : Hyperbola) : ℝ × ℝ := (Real.sqrt (h.a^2 + h.b^2), 0)

noncomputable def P (h : Hyperbola) : ℝ × ℝ := (Real.sqrt (h.a^2 + h.b^2), h.b^2 / h.a)

noncomputable def Q (h : Hyperbola) : ℝ × ℝ := 
  let c := Real.sqrt (h.a^2 + h.b^2)
  (h.a * (c + h.a) / (h.a - h.b + c), h.b * (c + h.a) / (h.a - h.b + c))

-- Define the vector relation
def vector_relation (h : Hyperbola) : Prop :=
  let ap := (P h).1 - (A h).1
  let aq := (Q h).1 - (A h).1
  ap = (2 - Real.sqrt 2) * aq

-- Theorem statement
theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_vector : vector_relation h) : 
  (F h).1 / h.a = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l91_9146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_proof_l91_9173

/-- Calculates the discount percentage given the purchase price, selling price, and profit percentage. -/
noncomputable def calculate_discount (purchase_price : ℝ) (selling_price : ℝ) (profit_percentage : ℝ) : ℝ :=
  let labelled_price := selling_price / (1 + profit_percentage / 100)
  let discount_fraction := 1 - (purchase_price / labelled_price)
  discount_fraction * 100

/-- Theorem stating that the discount percentage is approximately 21.88% given the problem conditions. -/
theorem discount_percentage_proof :
  let purchase_price : ℝ := 12500
  let selling_price : ℝ := 19200
  let profit_percentage : ℝ := 20
  let calculated_discount := calculate_discount purchase_price selling_price profit_percentage
  ∃ ε > 0, abs (calculated_discount - 21.88) < ε := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof
-- and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_proof_l91_9173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_decreasing_l91_9122

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 - x) / (1 + x))

theorem f_odd_and_decreasing :
  (∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → f (-x) = -f x) ∧
  (∀ x y, x ∈ Set.Ioo (0 : ℝ) 1 → y ∈ Set.Ioo (0 : ℝ) 1 → x < y → f y < f x) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_decreasing_l91_9122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_price_is_100_l91_9196

/-- The marked price of a product, given the discount rate, profit, and cost price. -/
noncomputable def marked_price (discount_rate : ℝ) (profit : ℝ) (cost_price : ℝ) : ℝ :=
  (cost_price + profit) / (1 - discount_rate)

/-- Theorem: The marked price is 100 yuan given the specified conditions. -/
theorem marked_price_is_100 :
  marked_price 0.1 30 60 = 100 := by
  -- Unfold the definition of marked_price
  unfold marked_price
  -- Simplify the expression
  simp
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_price_is_100_l91_9196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_rotation_l91_9199

theorem regular_polygon_rotation (n : ℕ) : n > 0 →
  (∃ (k : ℕ), n = 4 * k) ↔ (n = 12) ∧ 
  (∀ m ∈ ({6, 9, 12, 15} : Set ℕ), (∃ (k : ℕ), m = 4 * k) ↔ m = 12) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_rotation_l91_9199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tuna_can_cost_l91_9188

/-- Calculates the cost of a single can of tuna in cents -/
def cost_per_can (num_cans : ℕ) (num_coupons : ℕ) (coupon_discount : ℕ) 
                 (total_paid : ℚ) (change_received : ℚ) : ℕ :=
  let total_cost : ℚ := total_paid - change_received
  let total_discount : ℚ := (num_coupons * coupon_discount : ℚ) / 100
  let cost_before_coupons : ℚ := total_cost + total_discount
  let cost_per_can_dollars : ℚ := cost_before_coupons / num_cans
  (cost_per_can_dollars * 100).floor.toNat

/-- Proves that the cost of a single can of tuna is 175 cents -/
theorem tuna_can_cost : 
  cost_per_can 9 5 25 20 (11/2) = 175 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tuna_can_cost_l91_9188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l91_9185

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- angle A
  B : ℝ  -- angle B
  C : ℝ  -- angle C
  a : ℝ  -- side opposite to angle A
  b : ℝ  -- side opposite to angle B
  c : ℝ  -- side opposite to angle C

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.C = Real.pi/6 ∧ t.b = 1 ∧ Real.sin t.A = 3 * Real.sin t.B

-- Theorem statement
theorem triangle_area (t : Triangle) 
  (h : triangle_conditions t) : 
  (1/2 * t.a * t.b * Real.sin t.C) = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l91_9185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_extension_length_l91_9181

/-- Given a triangle ABC where ∠BAC = ∠BCA, AB = 5, BC = 6, CA = 6, and D is a point on the 
    extension of AB such that BD = 2.5, prove that CD ≈ 7.4 -/
theorem triangle_extension_length (A B C D : ℝ × ℝ) : 
  let dist := λ (p q : ℝ × ℝ) ↦ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let angle := λ (p q r : ℝ × ℝ) ↦ Real.arccos ((dist p q)^2 + (dist p r)^2 - (dist q r)^2) / (2 * dist p q * dist p r)
  angle B A C = angle B C A →
  dist A B = 5 →
  dist B C = 6 →
  dist C A = 6 →
  D.1 - A.1 = 7.5 * (B.1 - A.1) / 5 ∧ D.2 - A.2 = 7.5 * (B.2 - A.2) / 5 →
  abs (dist C D - 7.4) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_extension_length_l91_9181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minute_hand_movement_5_hours_l91_9100

/-- The number of radians in a full circle -/
noncomputable def full_circle_radians : ℝ := 2 * Real.pi

/-- The number of complete rotations the minute hand makes in 5 hours -/
def rotations_in_5_hours : ℕ := 5

/-- The number of radians the minute hand moves in 5 hours -/
noncomputable def minute_hand_movement (full_circle : ℝ) (rotations : ℕ) : ℝ :=
  -1 * (rotations : ℝ) * full_circle

theorem minute_hand_movement_5_hours :
  minute_hand_movement full_circle_radians rotations_in_5_hours = -10 * Real.pi := by
  unfold minute_hand_movement
  unfold full_circle_radians
  simp [rotations_in_5_hours]
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minute_hand_movement_5_hours_l91_9100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_1197_l91_9153

theorem largest_prime_factor_of_1197 :
  (Nat.factors 1197).maximum? = some 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_1197_l91_9153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_189_is_gray_l91_9166

/-- Represents the color of a marble -/
inductive MarbleColor
  | Gray
  | White
  | Black
  | Blue

/-- The length of one complete cycle in the marble pattern -/
def cycleLength : Nat := 5 + 4 + 3 + 2

/-- Determines the color of a marble at a given position in the sequence -/
def marbleColorAt (position : Nat) : MarbleColor :=
  match position % cycleLength with
  | n => 
    if n < 5 then MarbleColor.Gray
    else if n < 9 then MarbleColor.White
    else if n < 12 then MarbleColor.Black
    else MarbleColor.Blue

/-- Proves that the 189th marble in the sequence is gray -/
theorem marble_189_is_gray :
  marbleColorAt 189 = MarbleColor.Gray := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_189_is_gray_l91_9166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l91_9152

/-- The function g(x) defined as (-x^2 + 5x - 3)e^x -/
noncomputable def g (x : ℝ) : ℝ := (-x^2 + 5*x - 3) * Real.exp x

/-- The derivative of g(x) -/
noncomputable def g' (x : ℝ) : ℝ := Real.exp x * (-x^2 + 3*x + 2)

/-- The point on g(x) at x = 1 -/
noncomputable def point : ℝ × ℝ := (1, Real.exp 1)

/-- The slope of the tangent line at x = 1 -/
noncomputable def tangent_slope : ℝ := 4 * Real.exp 1

/-- The equation of the tangent line -/
noncomputable def tangent_line (x : ℝ) : ℝ := tangent_slope * (x - 1) + Real.exp 1

theorem tangent_line_at_one :
  ∀ x, tangent_line x = 4 * Real.exp 1 * x - 3 * Real.exp 1 := by
  sorry

#check tangent_line_at_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l91_9152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_men_correct_l91_9117

/-- Represents the initial number of men employed by NHAI -/
def initial_men : ℕ := 300

/-- Represents the total length of the highway in km -/
noncomputable def total_length : ℝ := 2

/-- Represents the total number of days to complete the project -/
def total_days : ℕ := 50

/-- Represents the initial number of working hours per day -/
def initial_hours_per_day : ℕ := 8

/-- Represents the fraction of work completed after 25 days -/
noncomputable def work_completed_fraction : ℝ := 1/3

/-- Represents the number of additional men hired -/
def additional_men : ℕ := 60

/-- Represents the new number of working hours per day -/
def new_hours_per_day : ℕ := 10

/-- Represents the number of days worked before additional men were hired -/
def days_before_change : ℕ := 25

theorem initial_men_correct :
  (initial_men : ℝ) * days_before_change * initial_hours_per_day * work_completed_fraction =
  ((initial_men + additional_men) : ℝ) * (total_days - days_before_change) * new_hours_per_day * (1 - work_completed_fraction) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_men_correct_l91_9117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_three_zeros_l91_9149

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := Real.cos (Real.pi * Real.log x + x)

-- State the theorem
theorem g_has_three_zeros :
  ∃ (x₁ x₂ x₃ : ℝ), 
    (1 < x₁ ∧ x₁ < Real.exp 2) ∧
    (1 < x₂ ∧ x₂ < Real.exp 2) ∧
    (1 < x₃ ∧ x₃ < Real.exp 2) ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    g x₁ = 0 ∧ g x₂ = 0 ∧ g x₃ = 0 ∧
    ∀ (x : ℝ), 1 < x ∧ x < Real.exp 2 ∧ g x = 0 → (x = x₁ ∨ x = x₂ ∨ x = x₃) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_three_zeros_l91_9149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_of_squares_l91_9192

/-- Represents a square in a 2D plane --/
structure Square where
  sideLength : ℝ

/-- Represents the configuration of squares WXYZ and JKLM --/
structure SquareConfiguration where
  wxyz : Square
  jklm : Square
  wj : ℝ
  jz : ℝ

/-- The theorem stating the ratio of areas of squares JKLM and WXYZ --/
theorem area_ratio_of_squares (config : SquareConfiguration) :
  config.wxyz.sideLength = 12 ∧ 
  config.wj = 3 * config.jz ∧ 
  config.wj + config.jz = config.wxyz.sideLength →
  (config.jklm.sideLength ^ 2) / (config.wxyz.sideLength ^ 2) = 5 / 8 := by
  sorry

#check area_ratio_of_squares

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_of_squares_l91_9192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_shaded_fraction_l91_9133

/-- Represents a triangle with a base and height -/
structure Triangle where
  base : ℚ
  height : ℚ

/-- Represents a pentagon on a square grid -/
structure Pentagon where
  gridSize : ℚ
  shadedTriangle : Triangle
  unshadedTriangle : Triangle

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℚ := (1/2) * t.base * t.height

/-- Calculates the fraction of the pentagon that is shaded -/
def shadedFraction (p : Pentagon) : ℚ :=
  let gridArea := p.gridSize * p.gridSize
  let pentagonArea := gridArea - triangleArea p.unshadedTriangle
  triangleArea p.shadedTriangle / pentagonArea

theorem pentagon_shaded_fraction :
  let p := Pentagon.mk 6 (Triangle.mk 3 6) (Triangle.mk 3 3)
  shadedFraction p = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_shaded_fraction_l91_9133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_molecular_weight_l91_9177

-- Define atomic weights
def atomic_weight_H : Float := 1.008
def atomic_weight_Cr : Float := 51.996
def atomic_weight_O : Float := 15.999

-- Define the number of atoms for each element
def num_H : Nat := 2
def num_Cr : Nat := 1
def num_O : Nat := 4

-- Define the molecular weight calculation function
def molecular_weight (H_weight Cr_weight O_weight : Float) (H_num Cr_num O_num : Nat) : Float :=
  H_weight * H_num.toFloat + Cr_weight * Cr_num.toFloat + O_weight * O_num.toFloat

-- Theorem statement
theorem compound_molecular_weight :
  (molecular_weight atomic_weight_H atomic_weight_Cr atomic_weight_O num_H num_Cr num_O - 118.008).abs < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_molecular_weight_l91_9177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_product_constraint_l91_9179

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem min_sum_product_constraint (p q r s : ℕ+) 
  (h : (p : ℕ) * q * r * s = factorial 9) : 
  (p : ℕ) + q + r + s ≥ 132 ∧ ∃ (a b c d : ℕ+), (a : ℕ) * b * c * d = factorial 9 ∧ a + b + c + d = 132 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_product_constraint_l91_9179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l91_9195

noncomputable def f (a x : ℝ) : ℝ :=
  if x < 1 then x^2 + 2*a else -x

theorem a_range (a : ℝ) (h1 : a < 0) 
  (h2 : f a (1 - a) ≥ f a (1 + a)) : 
  -2 ≤ a ∧ a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l91_9195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l91_9148

/-- The function g(x) = 5 / (3x^4 + 7) -/
noncomputable def g (x : ℝ) : ℝ := 5 / (3 * x^4 + 7)

/-- g is an even function -/
theorem g_is_even : ∀ x : ℝ, g (-x) = g x := by
  intro x
  unfold g
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l91_9148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_theorem_l91_9160

/-- Represents a square pyramid -/
structure SquarePyramid where
  baseEdge : ℝ
  altitude : ℝ

/-- Represents a frustum of a square pyramid -/
structure Frustum where
  bottomBaseEdge : ℝ
  topBaseEdge : ℝ
  height : ℝ

/-- Calculates the volume of a square pyramid -/
noncomputable def pyramidVolume (p : SquarePyramid) : ℝ :=
  (1 / 3) * p.baseEdge ^ 2 * p.altitude

/-- Calculates the volume of a frustum -/
noncomputable def frustumVolume (f : Frustum) : ℝ :=
  (1 / 3) * f.height * (f.bottomBaseEdge ^ 2 + f.topBaseEdge ^ 2 + f.bottomBaseEdge * f.topBaseEdge)

theorem frustum_volume_theorem (originalPyramid smallerPyramid : SquarePyramid) 
    (frustum : Frustum) :
    originalPyramid.baseEdge = 16 →
    originalPyramid.altitude = 10 →
    smallerPyramid.baseEdge = 8 →
    smallerPyramid.altitude = 5 →
    frustum.height = 5 →
    frustum.bottomBaseEdge = originalPyramid.baseEdge →
    frustum.topBaseEdge = smallerPyramid.baseEdge →
    pyramidVolume originalPyramid - pyramidVolume smallerPyramid = 2240 / 3 := by
  sorry

#check frustum_volume_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_theorem_l91_9160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_value_of_f_l91_9193

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 5) + Real.sin (x / 7)

-- Define the theorem
theorem smallest_max_value_of_f :
  ∃ x : ℝ, x = 5850 * (π / 180) ∧
  (∀ y : ℝ, 0 < y → y < x → f y < f x) ∧
  (∀ z : ℝ, z > 0 → f z ≤ f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_value_of_f_l91_9193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plate_arrangement_theorem_l91_9123

def blue_plates : ℕ := 6
def red_plates : ℕ := 3
def green_plates : ℕ := 4
def yellow_plates : ℕ := 1

def total_plates : ℕ := blue_plates + red_plates + green_plates + yellow_plates

def circular_arrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

def arrangements_with_all_green_adjacent : ℕ := 
  circular_arrangements (total_plates - green_plates + 1)

theorem plate_arrangement_theorem : 
  circular_arrangements total_plates - arrangements_with_all_green_adjacent = 876 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plate_arrangement_theorem_l91_9123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_speed_is_100_l91_9109

/-- Represents the speed and distance characteristics of a car journey -/
structure Journey where
  initial_speed : ℝ
  total_distance : ℝ
  first_reduction : ℝ
  second_reduction : ℝ

/-- Calculates the time difference between traveling at reduced speed and twice-reduced speed -/
noncomputable def time_difference (j : Journey) : ℝ :=
  (j.initial_speed / 2) / (j.initial_speed - j.first_reduction) +
  20 / (j.initial_speed - j.first_reduction - j.second_reduction) -
  20 / (j.initial_speed - j.first_reduction)

/-- The theorem stating the initial speed of the car -/
theorem initial_speed_is_100 (j : Journey) 
  (h1 : j.total_distance = 100)
  (h2 : j.first_reduction = 10)
  (h3 : j.second_reduction = 10)
  (h4 : time_difference j = 5 / 60) :
  j.initial_speed = 100 := by
  sorry

#check initial_speed_is_100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_speed_is_100_l91_9109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_sections_correct_l91_9136

def school_sections (num_boys num_girls : ℕ) : ℕ :=
  let section_size := Nat.gcd num_boys num_girls
  let boy_sections := num_boys / section_size
  let girl_sections := num_girls / section_size
  boy_sections + girl_sections

#eval school_sections 408 264  -- Expected output: 28

theorem school_sections_correct (num_boys num_girls : ℕ) 
  (h_boys : num_boys = 408) 
  (h_girls : num_girls = 264) : 
  school_sections num_boys num_girls = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_sections_correct_l91_9136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sine_product_equality_l91_9108

theorem inverse_sine_product_equality (α : ℝ) : 
  (Real.arcsin (2 * α)) * (Real.arcsin (Real.sin (π / 3) - 2 * α)) * (Real.arcsin (Real.sin (π / 3) + 2 * α)) = 
  4 * (Real.arcsin (6 * α)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sine_product_equality_l91_9108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_money_left_l91_9163

/-- The amount of money John has left after purchases -/
noncomputable def money_left (q : ℝ) : ℝ :=
  let initial_money := 60
  let drink_cost := q
  let medium_pizza_cost := 2 * drink_cost
  let large_pizza_cost := 3 * drink_cost
  let dessert_cost := (1/2) * drink_cost
  let total_cost := 5 * drink_cost + 3 * medium_pizza_cost + 2 * large_pizza_cost + 4 * dessert_cost
  initial_money - total_cost

/-- Theorem stating that John will have 60 - 19q dollars left after purchases -/
theorem john_money_left (q : ℝ) : money_left q = 60 - 19 * q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_money_left_l91_9163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l91_9147

-- Define ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the major axis length of an ellipse
noncomputable def major_axis_length (a b : ℝ) : ℝ := 2 * max a b

-- Define eccentricity of an ellipse
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (min a b / max a b)^2)

-- Define ellipse C1
def ellipse_C1 (x y : ℝ) : Prop := x^2/(25/4) + y^2/(21/4) = 1

theorem ellipse_properties :
  (∀ x y, ellipse_C x y → major_axis_length 2 (Real.sqrt 3) = 4) ∧
  (∀ x y, ellipse_C1 x y →
    eccentricity (5/2) (Real.sqrt (21/4)) = eccentricity 2 (Real.sqrt 3) ∧
    ellipse_C1 2 (-Real.sqrt 3)) := by
  sorry

#check ellipse_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l91_9147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_iff_l91_9113

/-- Definition of the function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + a*x else a*x^2 + x

/-- Theorem stating that f(x) is monotonically decreasing on ℝ if and only if a ≤ -2 -/
theorem f_monotone_decreasing_iff (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) ↔ a ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_iff_l91_9113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equals_negative_f_shifted_l91_9175

-- Define f as an arbitrary real-valued function
variable (f : ℝ → ℝ)

-- Define h as the result of reflecting f vertically and shifting left by 3
def h (f : ℝ → ℝ) (x : ℝ) : ℝ := -(f (x + 3))

-- Theorem stating that h(x) = -f(x+3)
theorem h_equals_negative_f_shifted (f : ℝ → ℝ) (x : ℝ) : h f x = -(f (x + 3)) := by
  -- Unfold the definition of h
  unfold h
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equals_negative_f_shifted_l91_9175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_sum_of_six_l91_9176

/-- Represents a 3x3 table filled with numbers 1 to 9 -/
def Table := Fin 3 → Fin 3 → Fin 9

/-- Checks if a table is valid according to the problem conditions -/
def is_valid_table (t : Table) : Prop :=
  (∀ i j, t i j ≠ 0) ∧
  (∀ i j k l, (i ≠ k ∨ j ≠ l) → t i j ≠ t k l) ∧
  (t 0 0 = 1) ∧ (t 2 0 = 2) ∧ (t 0 2 = 3) ∧ (t 2 2 = 4)

/-- Returns the sum of adjacent numbers to the given position -/
def adjacent_sum (t : Table) (i j : Fin 3) : ℕ :=
  (if i > 0 then (t (i-1) j).val else 0) +
  (if i < 2 then (t (i+1) j).val else 0) +
  (if j > 0 then (t i (j-1)).val else 0) +
  (if j < 2 then (t i (j+1)).val else 0)

/-- Finds the position of a given number in the table -/
def find_position (t : Table) (n : Fin 9) : Option (Fin 3 × Fin 3) :=
  (List.range 3).findSome? (λ i => 
    (List.range 3).findSome? (λ j => 
      if t i j = n then some (i, j) else none))

theorem adjacent_sum_of_six (t : Table) :
  is_valid_table t →
  (∃ i j, t i j = 5 ∧ adjacent_sum t i j = 9) →
  (∃ k l, t k l = 6 ∧ adjacent_sum t k l = 29) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_sum_of_six_l91_9176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_leaves_per_hour_is_967_l91_9106

/-- Represents the number of leaves falling from a tree in each hour -/
structure TreeLeaves where
  hour1 : ℕ
  hour2 : ℕ
  hour3 : ℕ

/-- Calculates the average number of leaves falling per hour across all trees -/
def averageLeavesPerHour (trees : List TreeLeaves) : ℚ :=
  let totalLeaves := (trees.map (λ t => t.hour1 + t.hour2 + t.hour3)).sum
  totalLeaves / (trees.length * 3 : ℕ)

/-- The main theorem stating the average number of leaves falling per hour -/
theorem average_leaves_per_hour_is_967 :
  let tree1 := TreeLeaves.mk 7 12 9
  let tree2 := TreeLeaves.mk 4 4 6
  let tree3 := TreeLeaves.mk 10 20 15
  let trees := [tree1, tree2, tree3]
  averageLeavesPerHour trees = 29/3 := by
  sorry

#eval (29 : ℚ) / 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_leaves_per_hour_is_967_l91_9106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_364_fraction_l91_9118

def is_valid_fraction (m n : ℕ) : Prop :=
  0 < m ∧ m < n ∧ Nat.Coprime m n ∧
  ∃ k : ℕ, (1000 * m : ℤ) = 364 * n + k ∧ 0 < k ∧ k < n

theorem smallest_n_for_364_fraction :
  ∀ n : ℕ, (∃ m : ℕ, is_valid_fraction m n) → n ≥ 8 :=
by
  sorry

#check smallest_n_for_364_fraction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_364_fraction_l91_9118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twin_primes_sum_divisibility_l91_9182

/-- Definition of twin primes -/
def are_twin_primes (a b : ℕ) : Prop := 
  Nat.Prime a ∧ Nat.Prime b ∧ b = a + 2

/-- Definition of floor function -/
def floor (x : ℚ) : ℤ := ⌊x⌋

/-- Definition of the sum S -/
def S (p k : ℕ) : ℚ := 
  1 + (Finset.range k).sum (λ i => (1 : ℚ) / (i + 2) * (Nat.choose (p - 1) (i + 1)))

/-- Main theorem -/
theorem twin_primes_sum_divisibility (p : ℕ) (k : ℕ) 
  (h1 : are_twin_primes (p - 2) p)
  (h2 : k = floor ((2 * p - 1 : ℚ) / 3)) :
  ∃ (n : ℕ), S p k = n ∧ p ∣ n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twin_primes_sum_divisibility_l91_9182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l91_9114

open Real Set

noncomputable def f (x : ℝ) := sin x ^ 2 + Real.sqrt 3 * sin x * cos x

theorem f_properties :
  ∃ (T : ℝ),
    (∀ x, f (x + T) = f x) ∧
    (∀ k, f (k * π / 2 + π / 3 + x) = f (k * π / 2 + π / 3 - x)) ∧
    (∀ x ∈ Icc 0 (π / 2), f x ≤ 3 / 2) ∧
    (∃ x ∈ Icc 0 (π / 2), f x = 3 / 2) ∧
    (∀ x ∈ Icc 0 (π / 2), f x ≥ 0) ∧
    (∃ x ∈ Icc 0 (π / 2), f x = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l91_9114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l91_9145

theorem problem_solution (α β : Real) 
  (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.cos (2*α) = 4/5) 
  (h3 : β ∈ Set.Ioo (π/2) π) 
  (h4 : 5 * Real.sin (2*α + β) = Real.sin β) : 
  (Real.sin α + Real.cos α = 2 * Real.sqrt 10 / 5) ∧ (β = 3*π/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l91_9145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unicorn_tether_sum_l91_9159

/-- The length of the golden rope tethering the unicorn -/
noncomputable def rope_length : ℝ := 30

/-- The radius of the cylindrical tower -/
noncomputable def tower_radius : ℝ := 10

/-- The height at which the rope is attached to the unicorn -/
noncomputable def unicorn_attachment_height : ℝ := 5

/-- The distance from the end of the rope to the nearest point on the tower -/
noncomputable def rope_end_distance : ℝ := 5

/-- The length of the rope touching the tower -/
noncomputable def rope_touching_tower (a b : ℕ) (c : ℕ+) : ℝ := (a - Real.sqrt b) / c

/-- The theorem stating the sum of a, b, and c -/
theorem unicorn_tether_sum (a b : ℕ) (c : ℕ+) 
  (h_prime : Nat.Prime c.val)
  (h_rope_touch : rope_touching_tower a b c = (90 - Real.sqrt 1500) / 3) :
  a + b + c = 1593 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unicorn_tether_sum_l91_9159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_consecutive_integers_l91_9180

variable (k : ℕ)

def sequence_start (k : ℕ) : ℕ := k^2 + 1

def sequence_length (k : ℕ) : ℕ := 3*k + 2

def sequence_sum (k : ℕ) : ℕ := 3*k^3 + 8*k^2 + 6*k + 3

theorem sum_of_consecutive_integers (k : ℕ) :
  (Finset.range (sequence_length k)).sum (λ i => sequence_start k + i) = sequence_sum k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_consecutive_integers_l91_9180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_log_base_3_l91_9129

-- Define the logarithm function
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3

-- Define the exponential function
noncomputable def g (x : ℝ) : ℝ := 3^x

-- State the theorem
theorem inverse_log_base_3 : 
  (∀ x > 0, f (g x) = x) ∧ (∀ x > 0, g (f x) = x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_log_base_3_l91_9129
