import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_solution_l357_35731

/-- Two vectors in ℝ³ -/
def a (x : ℝ) : Fin 3 → ℝ := ![x, 4, 1]
def b (y : ℝ) : Fin 3 → ℝ := ![-2, y, -1]

/-- Definition of parallel vectors in ℝ³ -/
def parallel (u v : Fin 3 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ i, u i = k * v i

/-- Theorem: If a(x) is parallel to b(y), then x = 2 and y = -4 -/
theorem parallel_vectors_solution (x y : ℝ) :
  parallel (a x) (b y) → x = 2 ∧ y = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_solution_l357_35731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_formula_l357_35701

/-- A pyramid with all lateral faces equally inclined to the base plane -/
structure InclinedPyramid where
  /-- The ratio of the area of the section through the center of the inscribed sphere to the base area -/
  k : ℝ
  /-- k is positive -/
  k_pos : k > 0

/-- The dihedral angle at the base of the pyramid -/
noncomputable def dihedral_angle (p : InclinedPyramid) : ℝ :=
  2 * Real.arccos (1 / Real.rpow (4 * p.k) (1/4))

theorem dihedral_angle_formula (p : InclinedPyramid) :
  dihedral_angle p = 2 * Real.arccos (1 / Real.rpow (4 * p.k) (1/4)) := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_formula_l357_35701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l357_35788

/-- Hyperbola type representing the equation x²/a² - y²/b² = 1 --/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- Point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Vector2 in 2D space --/
structure Vector2 where
  x : ℝ
  y : ℝ

/-- Theorem: Eccentricity of a specific hyperbola --/
theorem hyperbola_eccentricity (h : Hyperbola) (F₁ M N : Point) :
  let asymptote : Point → Prop := fun p => h.b * p.x - h.a * p.y = 0
  let on_hyperbola : Point → Prop := fun p => p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1
  let perpendicular : Point → Point → Point → Prop :=
    fun p₁ p₂ p₃ => (p₂.x - p₁.x) * (p₃.x - p₂.x) + (p₂.y - p₁.y) * (p₃.y - p₂.y) = 0
  asymptote M ∧
  on_hyperbola N ∧
  perpendicular F₁ M (Point.mk (M.x + 1) (M.y + (h.a / h.b))) ∧
  let MN : Vector2 := Vector2.mk (N.x - M.x) (N.y - M.y)
  let F₁M : Vector2 := Vector2.mk (M.x - F₁.x) (M.y - F₁.y)
  MN.x = 3 * F₁M.x ∧ MN.y = 3 * F₁M.y →
  h.a / Real.sqrt (h.a^2 + h.b^2) = 5/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l357_35788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_firework_max_height_time_l357_35771

-- Define the height function
noncomputable def h (t : ℝ) : ℝ := -3/4 * t^2 + 12*t - 21

-- Theorem statement
theorem firework_max_height_time : 
  ∃ (t_max : ℝ), t_max = 8 ∧ ∀ (t : ℝ), h t ≤ h t_max := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_firework_max_height_time_l357_35771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_credit_card_discount_l357_35759

/-- Calculates the additional discount percentage for using a store credit card --/
theorem store_credit_card_discount 
  (original_price : ℝ)
  (gift_card_value : ℝ)
  (initial_discount_rate : ℝ)
  (final_price : ℝ)
  (h1 : original_price = 2000)
  (h2 : gift_card_value = 200)
  (h3 : initial_discount_rate = 0.15)
  (h4 : final_price = 1330)
  : ∃ (additional_discount_rate : ℝ), 
    (abs (additional_discount_rate - 0.1133) < 0.0001 ∧ 
     final_price = original_price * (1 - initial_discount_rate) - gift_card_value - 
                   (original_price * (1 - initial_discount_rate) - gift_card_value) * additional_discount_rate) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_credit_card_discount_l357_35759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_in_right_cone_l357_35772

/-- The radius of a sphere inscribed in a right cone -/
noncomputable def inscribedSphereRadius (coneBaseRadius : ℝ) (coneHeight : ℝ) : ℝ :=
  (coneHeight * coneBaseRadius) / (coneBaseRadius + (coneBaseRadius^2 + coneHeight^2).sqrt)

theorem inscribed_sphere_in_right_cone (b d : ℝ) :
  inscribedSphereRadius 15 30 = b * d.sqrt - b →
  b + d = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_in_right_cone_l357_35772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_squared_sum_0_to_90_l357_35782

noncomputable def angle_range : List ℝ := List.range 46 |>.map (λ n => 2 * n * Real.pi / 180)

theorem cosine_squared_sum_0_to_90 : 
  (angle_range.map (λ θ => Real.cos θ ^ 2)).sum = 23 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_squared_sum_0_to_90_l357_35782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_archimedean_triangle_vertex_ordinate_l357_35723

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 4x -/
def Parabola := {p : Point | p.y^2 = 4 * p.x}

/-- Represents a line y = k(x-1) -/
def Line (k : ℝ) := {p : Point | p.y = k * (p.x - 1)}

/-- The focus of the parabola -/
def focus : Point := ⟨1, 0⟩

/-- The length of a line segment between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ := Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem statement -/
theorem archimedean_triangle_vertex_ordinate 
  (k : ℝ) 
  (A B : Point) 
  (h1 : A ∈ Parabola ∩ Line k) 
  (h2 : B ∈ Parabola ∩ Line k) 
  (h3 : distance A B = 8) 
  (h4 : ∃ t : ℝ, focus = ⟨(1-t)*A.x + t*B.x, (1-t)*A.y + t*B.y⟩) :
  ∃ P : Point, P.x = -1 ∧ (P.y = 2 ∨ P.y = -2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_archimedean_triangle_vertex_ordinate_l357_35723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_theorem_l357_35734

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Theorem: The distances between given pairs of points are correct -/
theorem distance_theorem : 
  let A : Point3D := ⟨1, 1, 0⟩
  let B : Point3D := ⟨1, 1, 1⟩
  let C : Point3D := ⟨-3, 1, 5⟩
  let D : Point3D := ⟨0, -2, 3⟩
  distance A B = 1 ∧ distance C D = Real.sqrt 22 := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_theorem_l357_35734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kimiko_video_count_l357_35764

theorem kimiko_video_count : ℕ := by
  -- Define the total time spent watching videos
  let total_time : ℕ := 510

  -- Define the lengths of the videos
  let first_video_length : ℕ := 120
  let second_video_length : ℕ := 270
  let last_video_length : ℕ := 60

  -- Define the number of videos watched
  let num_videos : ℕ := 4

  -- State that the last two videos are equal in length
  have last_two_equal : last_video_length = last_video_length := by rfl

  -- Theorem: The sum of all video lengths equals the total time
  have sum_equals_total : first_video_length + second_video_length + last_video_length + last_video_length = total_time := by
    sorry

  -- Conclusion: The number of videos watched is 4
  exact num_videos

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kimiko_video_count_l357_35764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_h_addition_base_is_eight_l357_35737

/-- Represents a digit in base h --/
def Digit (h : ℕ) := Fin h

/-- Converts a list of digits in base h to a natural number --/
def toNatBase (h : ℕ) (digits : List (Digit h)) : ℕ :=
  digits.foldr (fun d acc => acc * h + d.val) 0

/-- The statement of the problem --/
theorem base_h_addition (h : ℕ) (h_pos : h > 0) : 
  toNatBase h [⟨6, sorry⟩, ⟨4, sorry⟩, ⟨5, sorry⟩, ⟨3, sorry⟩] + 
  toNatBase h [⟨7, sorry⟩, ⟨5, sorry⟩, ⟨1, sorry⟩, ⟨2, sorry⟩] = 
  toNatBase h [⟨1, sorry⟩, ⟨6, sorry⟩, ⟨1, sorry⟩, ⟨6, sorry⟩, ⟨5, sorry⟩] → 
  h = 8 := by
  sorry

/-- Main theorem proving the base is 8 --/
theorem base_is_eight : ∃ h : ℕ, h > 0 ∧ 
  toNatBase h [⟨6, sorry⟩, ⟨4, sorry⟩, ⟨5, sorry⟩, ⟨3, sorry⟩] + 
  toNatBase h [⟨7, sorry⟩, ⟨5, sorry⟩, ⟨1, sorry⟩, ⟨2, sorry⟩] = 
  toNatBase h [⟨1, sorry⟩, ⟨6, sorry⟩, ⟨1, sorry⟩, ⟨6, sorry⟩, ⟨5, sorry⟩] ∧ 
  h = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_h_addition_base_is_eight_l357_35737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l357_35708

/-- Calculates the time (in seconds) for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

theorem train_crossing_bridge :
  train_crossing_time 170 45 205 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l357_35708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_triangle_exists_l357_35713

/-- Represents a place (island or city) --/
inductive Place
| Island : Fin 2 → Place
| City : Fin 7 → Place

/-- Defines whether two cities are adjacent --/
def adjacent (c1 c2 : Fin 7) : Prop :=
  (c1.val + 1) % 7 = c2.val ∨ (c2.val + 1) % 7 = c1.val

/-- Defines whether there's a connection between two places --/
def connected (p1 p2 : Place) : Prop :=
  match p1, p2 with
  | Place.Island _, Place.Island _ => true
  | Place.Island _, Place.City _ => true
  | Place.City _, Place.Island _ => true
  | Place.City c1, Place.City c2 => ¬(adjacent c1 c2)

/-- Represents the two competing shipping companies --/
inductive Company
| A
| B

/-- Assigns a company to each connection --/
def assignment (p1 p2 : Place) : connected p1 p2 → Company := sorry

/-- The main theorem --/
theorem monochromatic_triangle_exists :
  ∃ (p1 p2 p3 : Place) (c12 : connected p1 p2) (c23 : connected p2 p3) (c31 : connected p3 p1),
    assignment p1 p2 c12 = assignment p2 p3 c23 ∧
    assignment p2 p3 c23 = assignment p3 p1 c31 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_triangle_exists_l357_35713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_equals_19_l357_35765

def sequence_a : ℕ → ℤ
  | 0 => 2  -- We define a₀ as 2 to match a₁ in the original problem
  | 1 => 5  -- This matches a₂ in the original problem
  | (n + 2) => sequence_a (n + 1) + sequence_a n

theorem a_5_equals_19 : sequence_a 4 = 19 := by
  -- We use 4 here because Lean's indexing starts at 0
  -- so sequence_a 4 corresponds to a₅ in the original problem
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_equals_19_l357_35765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_properties_l357_35725

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the right-angled triangle MAB
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  M : ℝ × ℝ
  inscribed : parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ parabola M.1 M.2
  right_angle : (A.1 - M.1) * (B.1 - M.1) + (A.2 - M.2) * (B.2 - M.2) = 0

-- Define the fixed point P
def P : ℝ × ℝ := (5, -2)

-- Define the circle (renamed to avoid conflict)
def circleEq (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 8

-- Theorem statement
theorem inscribed_triangle_properties 
  (t : RightTriangle) 
  (h_M : t.M = (1, 2)) :
  (∃ (k : ℝ), t.A.2 - t.B.2 ≠ 0 → 
    (P.1 - t.A.1) / (P.2 - t.A.2) = (t.B.1 - t.A.1) / (t.B.2 - t.A.2)) ∧
  (∀ N : ℝ × ℝ, N.1 ≠ 1 → 
    ((N.1 - t.A.1) * (t.B.2 - t.A.2) = (N.2 - t.A.2) * (t.B.1 - t.A.1)) →
    ((t.M.1 - N.1) * (t.B.1 - t.A.1) + (t.M.2 - N.2) * (t.B.2 - t.A.2) = 0) →
    circleEq N.1 N.2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_properties_l357_35725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_segments_determine_y_l357_35714

/-- Given four points on a Cartesian plane, if two line segments are parallel, 
    then the y-coordinate of the fourth point is determined. -/
theorem parallel_segments_determine_y (k : ℝ) : 
  let A : ℝ × ℝ := (-4, 0)
  let B : ℝ × ℝ := (2, -2)
  let X : ℝ × ℝ := (0, 8)
  let Y : ℝ × ℝ := (18, k)
  (B.2 - A.2) / (B.1 - A.1) = (Y.2 - X.2) / (Y.1 - X.1) → k = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_segments_determine_y_l357_35714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_no_zeros_condition_inequality_condition_l357_35748

-- Define the function f(x) = e^x - ax
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x

-- Theorem 1
theorem tangent_line_condition (a : ℝ) :
  (∀ x, (Real.exp 0 - a) * x + f a 0 = 0 → x = 1) → a = 2 := by sorry

-- Theorem 2
theorem no_zeros_condition (a : ℝ) :
  (∀ x, x > -1 → f a x ≠ 0) → a ∈ Set.Icc (-Real.exp (-1)) (Real.exp 1) := by sorry

-- Theorem 3
theorem inequality_condition (x : ℝ) :
  f 1 x ≥ (1 + x) / (f 1 x + x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_no_zeros_condition_inequality_condition_l357_35748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l357_35793

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x) - (1/4) * x^2

theorem max_value_of_f :
  ∃ (M : ℝ), M = Real.log 2 - 1/4 ∧ 
  (∀ x > -1, f x ≤ M) ∧
  (∀ ε > 0, ∃ x > -1, f x > M - ε) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l357_35793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_solution_l357_35711

theorem no_integer_solution :
  ¬ ∃ (a b c d : ℤ), (4 : ℝ) ^ (a : ℝ) + (5 : ℝ) ^ (b : ℝ) = (2 : ℝ) ^ (c : ℝ) + (2 : ℝ) ^ (d : ℝ) + 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_solution_l357_35711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_l357_35799

/-- Simple interest calculation function -/
noncomputable def simpleInterest (principal rate time : ℝ) : ℝ :=
  principal * (1 + rate * time / 100)

/-- Theorem stating the principal amount given the conditions -/
theorem principal_amount (rate : ℝ) :
  (∃ (P : ℝ), simpleInterest P rate 2 = 720 ∧ simpleInterest P rate 7 = 1020) →
  (∃ (P : ℝ), P = 600 ∧ simpleInterest P rate 2 = 720 ∧ simpleInterest P rate 7 = 1020) :=
by
  sorry

#check principal_amount

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_l357_35799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_perimeter_l357_35726

/-- Definition of an isosceles trapezoid -/
structure IsoscelesTrapezoid (top_base bottom_base height : ℝ) : Prop where
  top_positive : top_base > 0
  bottom_positive : bottom_base > 0
  height_positive : height > 0
  bottom_larger : bottom_base > top_base

/-- Definition of perimeter for a trapezoid -/
def perimeter (top_base bottom_base leg : ℝ) : ℝ :=
  top_base + bottom_base + 2 * leg

/-- Perimeter of an isosceles trapezoid -/
theorem isosceles_trapezoid_perimeter
  (top_base bottom_base height : ℝ)
  (h_top : top_base = 3)
  (h_bottom : bottom_base = 9)
  (h_height : height = 4)
  (h_isosceles : IsoscelesTrapezoid top_base bottom_base height) :
  let leg := Real.sqrt ((bottom_base - top_base)^2 / 4 + height^2)
  perimeter top_base bottom_base leg = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_perimeter_l357_35726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_formation_amount_l357_35781

/-- Represents the amount of a substance in moles -/
structure Mole where
  value : ℝ

/-- Represents mass in grams -/
structure Grams where
  value : ℝ

/-- Represents molar mass in g/mol -/
structure MolarMass where
  value : ℝ

/-- The molar mass of water in g/mol -/
def water_molar_mass : MolarMass := ⟨18.015⟩

/-- Calculates the mass of a substance given its amount in moles and molar mass -/
def mass_from_moles (amount : Mole) (molar_mass : MolarMass) : Grams :=
  ⟨amount.value * molar_mass.value⟩

/-- The amount of water formed when 1 mole of NaOH reacts with 1 mole of HCl -/
def water_formed : Grams :=
  mass_from_moles ⟨1⟩ water_molar_mass

theorem water_formation_amount :
  water_formed.value = 18.015 := by
  unfold water_formed
  unfold mass_from_moles
  unfold water_molar_mass
  simp
  -- The proof is completed by simplification

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_formation_amount_l357_35781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_plus_cos_greater_than_one_l357_35753

theorem cos_squared_plus_cos_greater_than_one (θ : Real) 
  (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.sin θ ^ 2 + Real.sin θ = 1) : 
  Real.cos θ ^ 2 + Real.cos θ > 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_plus_cos_greater_than_one_l357_35753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_desiree_age_proof_l357_35712

/-- Represents Desiree's current age in years -/
noncomputable def desiree_age : ℝ := 2.99999835

/-- Represents Desiree's cousin's current age in years -/
noncomputable def cousin_age : ℝ := desiree_age / 2

/-- The fraction used in the problem statement -/
noncomputable def fraction : ℝ := 0.6666666

theorem desiree_age_proof :
  desiree_age = 2 * cousin_age ∧
  desiree_age + 30 = fraction * (cousin_age + 30) + 14 →
  desiree_age = 2.99999835 := by
  intro h
  exact rfl

#check desiree_age_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_desiree_age_proof_l357_35712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaporation_theorem_l357_35705

/-- Represents the composition of a solution --/
structure Solution where
  total_mass : ℝ
  liquid_x_fraction : ℝ

/-- Represents the evaporation process --/
noncomputable def evaporation_process (initial : Solution) (evaporated_water : ℝ) (added_solution : Solution) : Solution :=
  { total_mass := initial.total_mass - evaporated_water + added_solution.total_mass,
    liquid_x_fraction := (initial.total_mass * initial.liquid_x_fraction + added_solution.total_mass * added_solution.liquid_x_fraction) / (initial.total_mass - evaporated_water + added_solution.total_mass) }

theorem evaporation_theorem (initial : Solution) (evaporated_water : ℝ) (added_solution : Solution) :
  initial.liquid_x_fraction = 0.2 →
  initial.total_mass = 8 →
  added_solution.liquid_x_fraction = 0.2 →
  added_solution.total_mass = 8 →
  (evaporation_process initial evaporated_water added_solution).liquid_x_fraction = 0.25 →
  evaporated_water = 3.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaporation_theorem_l357_35705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minus_one_eq_neg_three_l357_35751

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then 2^x + 2*x + b else -(2^(-x) + 2*(-x) + b)

-- State the theorem
theorem f_minus_one_eq_neg_three :
  ∃ b : ℝ, (∀ x : ℝ, f b x = -(f b (-x))) ∧ f b (-1) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minus_one_eq_neg_three_l357_35751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decrease_6m_notation_l357_35700

/-- Represents the change in water level in meters -/
structure WaterLevelChange where
  value : ℝ

/-- Convention: positive numbers represent rise in water level -/
axiom rise_convention {x : ℝ} : x > 0 → (WaterLevelChange.mk x).value > 0

/-- Represents a decrease in water level by 6 meters -/
def decrease_6m : WaterLevelChange := WaterLevelChange.mk (-6)

theorem decrease_6m_notation : 
  decrease_6m.value = -6 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decrease_6m_notation_l357_35700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l357_35785

theorem log_inequality (a b c : ℝ) (ha : a > b) (hb : b > 0) (hab : a * b = 1) (hc : 0 < c) (hc1 : c < 1) :
  Real.log ((a^2 + b^2) / 2) / Real.log c < Real.log (1 / (Real.sqrt a + Real.sqrt b))^2 / Real.log c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l357_35785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_special_set_exists_l357_35742

theorem no_special_set_exists : ¬∃ (A : Finset ℝ), 
  (A.card = 2016) ∧ 
  (∃ x ∈ A, x ≠ 0) ∧
  (∀ B ⊆ A, B.card = 1008 → 
    ∃ (p : Polynomial ℝ), 
      p.Monic ∧ 
      p.natDegree = 1008 ∧
      (∀ x ∈ B, p.eval x = 0) ∧
      (∃ f : Fin 1008 → ℝ, ∀ i : Fin 1008, f i ∈ A \ B ∧ p.coeff i = f i)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_special_set_exists_l357_35742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_some_squares_are_odd_and_some_primes_not_square_l357_35760

-- Define the property of being prime
def isPrime : Nat → Prop := sorry

-- Define the property of being odd
def isOdd : Nat → Prop := sorry

-- Define the property of being a square integer
def isSquare : Nat → Prop := sorry

-- Condition 1: Every prime number greater than 2 is odd
axiom prime_gt_2_is_odd : ∀ n : Nat, isPrime n → n > 2 → isOdd n

-- Condition 2: Some odd numbers are square integers
axiom some_odd_are_square : ∃ n : Nat, isOdd n ∧ isSquare n

-- Statement to prove
theorem some_squares_are_odd_and_some_primes_not_square :
  (∃ n : Nat, isSquare n ∧ isOdd n) ∧
  (∃ n : Nat, isPrime n ∧ ¬isSquare n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_some_squares_are_odd_and_some_primes_not_square_l357_35760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_pipes_l357_35709

noncomputable section

-- Define the diameters of the pipes
def small_diameter : ℝ := 2
def large_diameter : ℝ := 8

-- Define a function to calculate the area of a circular pipe given its diameter
noncomputable def pipe_area (diameter : ℝ) : ℝ := Real.pi * (diameter / 2) ^ 2

-- State the theorem
theorem equivalent_pipes :
  (pipe_area large_diameter) / (pipe_area small_diameter) = 16 := by
  -- Expand the definitions
  unfold pipe_area
  -- Simplify the expression
  simp [small_diameter, large_diameter]
  -- The actual proof would go here, but we'll use sorry for now
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_pipes_l357_35709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_when_m_3_find_m_for_given_intersection_l357_35790

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 6 / (x + 1) ≥ 1}
def B (m : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*x - m < 0}

-- Theorem 1
theorem intersection_complement_when_m_3 :
  A ∩ (Set.univ \ B 3) = {x : ℝ | 3 ≤ x ∧ x ≤ 5} := by sorry

-- Theorem 2
theorem find_m_for_given_intersection :
  A ∩ B 8 = {x : ℝ | -1 < x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_when_m_3_find_m_for_given_intersection_l357_35790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_height_ratio_l357_35720

/-- Represents the burn time of a candle in hours -/
structure BurnTime where
  hours : ℚ
  positive : hours > 0

/-- Represents a candle with its burn time -/
structure Candle where
  burnTime : BurnTime

/-- The time when the first candle's height is three times the second candle's height -/
noncomputable def timeWhenFirstIsTripleSecond (c1 c2 : Candle) : ℚ :=
  40 / 11

theorem candle_height_ratio (c1 c2 : Candle) 
  (h1 : c1.burnTime.hours = 5)
  (h2 : c2.burnTime.hours = 4) :
  let t := timeWhenFirstIsTripleSecond c1 c2
  (1 - t / c1.burnTime.hours) = 3 * (1 - t / c2.burnTime.hours) := by
  sorry

#check candle_height_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_height_ratio_l357_35720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l357_35719

noncomputable def f1 (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def f2 (x : ℝ) : ℝ := Real.exp x
noncomputable def f3 (x : ℝ) : ℝ := x^2
noncomputable def f4 (x : ℝ) : ℝ := 2^x

def is_intersection (x : ℝ) : Prop :=
  x > 0 ∧ (
    (f1 x = f2 x) ∨ (f1 x = f3 x) ∨ (f1 x = f4 x) ∨
    (f2 x = f3 x) ∨ (f2 x = f4 x) ∨
    (f3 x = f4 x)
  )

theorem intersection_count :
  ∃ (a b : ℝ), a ≠ b ∧ is_intersection a ∧ is_intersection b ∧
  ∀ (c : ℝ), is_intersection c → c = a ∨ c = b := by
  sorry

#check intersection_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l357_35719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_correct_l357_35763

/-- The circle with equation x² + y² = 9 -/
def myCircle : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = 9}

/-- The line with parametric equations x = 1 + 2t, y = 2 + t -/
def myLine : Set (ℝ × ℝ) :=
  {p | ∃ t : ℝ, p.1 = 1 + 2*t ∧ p.2 = 2 + t}

/-- The chord is the intersection of the circle and the line -/
def myChord : Set (ℝ × ℝ) :=
  myCircle ∩ myLine

/-- The length of the chord -/
noncomputable def chord_length : ℝ := 12 * Real.sqrt 5 / 5

theorem chord_length_is_correct :
  ∃ p q : ℝ × ℝ, p ∈ myChord ∧ q ∈ myChord ∧ p ≠ q ∧
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = chord_length := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_correct_l357_35763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_point_sum_l357_35779

noncomputable def A : ℝ × ℝ := (0, 0)
noncomputable def B : ℝ × ℝ := (12, 0)
noncomputable def C : ℝ × ℝ := (5, 8)
noncomputable def P : ℝ × ℝ := (5, 3)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem fermat_point_sum :
  ∃ (m n a b : ℕ), 
    distance A P + distance B P + distance C P = m * Real.sqrt a + n * Real.sqrt b + 5 ∧
    m = 1 ∧ n = 1 ∧ a = 34 ∧ b = 58 ∧ m + n = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_point_sum_l357_35779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_walking_distance_l357_35741

-- Define Tom's walking rate
def walking_rate : ℚ := 1 / 18

-- Define the time Tom walks
def walking_time : ℚ := 15

-- Function to calculate distance
def calculate_distance (rate : ℚ) (time : ℚ) : ℚ := rate * time

-- Function to round to nearest tenth
def round_to_tenth (x : ℚ) : ℚ := 
  ⌊(x * 10 + 1/2)⌋ / 10

-- Theorem statement
theorem tom_walking_distance : 
  round_to_tenth (calculate_distance walking_rate walking_time) = 8/10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_walking_distance_l357_35741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_point_of_f_l357_35761

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - (1/4) * x^4

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := x^2 * (1 - x)

-- Theorem statement
theorem extreme_value_point_of_f :
  ∃! x : ℝ, 0 < x ∧ x < 3 ∧ 
  (∀ y : ℝ, 0 < y ∧ y < 3 → f y ≤ f x ∨ f y ≥ f x) ∧
  x = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_point_of_f_l357_35761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_m_n_is_negative_four_l357_35766

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The tangent line to a function f at a point x -/
noncomputable def TangentLine (f : ℝ → ℝ) (x : ℝ) : Line where
  a := (deriv f) x
  b := -1
  c := f x - x * ((deriv f) x)

theorem product_m_n_is_negative_four (f : ℝ → ℝ) (m n : ℝ) :
  IsEven f →
  (∀ x, f x = x^2 + (m+1)*x + 2*m) →
  TangentLine f 1 = Line.mk (n-2) (-1) (-3) →
  m * n = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_m_n_is_negative_four_l357_35766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_is_closed_interval_from_3_to_infinity_l357_35745

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 1)

-- State the theorem
theorem f_range_is_closed_interval_from_3_to_infinity :
  ∀ y : ℝ, (∃ x : ℝ, x > 1 ∧ f x = y) ↔ y ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_is_closed_interval_from_3_to_infinity_l357_35745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_possible_m_l357_35789

/-- The set P of real numbers x such that |x| · x² = 1 -/
def P : Set ℝ := {x | |x| * x^2 = 1}

/-- The set Q of real numbers x such that m · |x| = 1, where m is a real parameter -/
def Q (m : ℝ) : Set ℝ := {x | m * |x| = 1}

/-- The number of possible values for m such that Q is a subset of P -/
def num_possible_m : ℕ := 3

/-- Theorem stating that there are exactly 3 possible values for m such that Q is a subset of P -/
theorem three_possible_m :
  ∃ (S : Finset ℝ), S.card = num_possible_m ∧
  ∀ m : ℝ, (Q m ⊆ P) ↔ m ∈ S :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_possible_m_l357_35789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l357_35762

/-- The length of a train given its crossing times over two platforms -/
theorem train_length_calculation (platform1_length platform2_length : ℝ)
                                 (time1 time2 : ℝ)
                                 (h1 : platform1_length = 130)
                                 (h2 : platform2_length = 250)
                                 (h3 : time1 = 15)
                                 (h4 : time2 = 20) :
  ∃ train_length : ℝ, 
    (train_length + platform1_length) / time1 = (train_length + platform2_length) / time2 ∧
    train_length = 230 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l357_35762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_plus_2cos6_l357_35777

theorem min_sin6_plus_2cos6 :
  (∀ x : ℝ, Real.sin x ^ 6 + 2 * Real.cos x ^ 6 ≥ 2/3) ∧
  (∃ x : ℝ, Real.sin x ^ 6 + 2 * Real.cos x ^ 6 = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_plus_2cos6_l357_35777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_M_l357_35730

theorem min_value_of_M (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  let M := max (1 / (a * c) + b) (max (1 / a + b * c) (a / b + c))
  2 ≤ M ∧ ∃ (a₀ b₀ c₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧
    (let M₀ := max (1 / (a₀ * c₀) + b₀) (max (1 / a₀ + b₀ * c₀) (a₀ / b₀ + c₀))
     M₀ = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_M_l357_35730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_roll_probability_l357_35724

/-- The probability of no two adjacent people rolling the same number on a six-sided die
    when seven people sit around a circular table. -/
theorem adjacent_roll_probability : (
  /- For the first person, any roll is possible -/
  1 *
  /- For the next 5 people, each has 5 possible rolls to avoid matching the previous person -/
  (5 / 6) ^ 5 *
  /- For the last person, they must avoid matching both their neighbors -/
  (5 / 6 * 4 / 6)
) = 625 / 2799 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_roll_probability_l357_35724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base8_subtraction_theorem_l357_35733

-- Define a custom type for base-8 digits
inductive Base8Digit : Type
| zero | one | two | three | four | five | six | seven

-- Define a function to convert Base8Digit to Nat
def base8ToNat : Base8Digit → Nat
| Base8Digit.zero => 0
| Base8Digit.one => 1
| Base8Digit.two => 2
| Base8Digit.three => 3
| Base8Digit.four => 4
| Base8Digit.five => 5
| Base8Digit.six => 6
| Base8Digit.seven => 7

-- Define a function to convert Nat to Base8Digit
def natToBase8 (n : Nat) : Base8Digit :=
  match n % 8 with
  | 0 => Base8Digit.zero
  | 1 => Base8Digit.one
  | 2 => Base8Digit.two
  | 3 => Base8Digit.three
  | 4 => Base8Digit.four
  | 5 => Base8Digit.five
  | 6 => Base8Digit.six
  | _ => Base8Digit.seven

-- Define base-8 subtraction
def base8Sub (a b : Base8Digit) : Base8Digit :=
  natToBase8 ((base8ToNat a - base8ToNat b + 8) % 8)

-- Define the main theorem
theorem base8_subtraction_theorem (C D : Base8Digit) :
  base8Sub D Base8Digit.six = Base8Digit.three ∧
  base8Sub D Base8Digit.three = Base8Digit.one ∧
  base8Sub C D = Base8Digit.five →
  base8Sub D C = Base8Digit.five := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base8_subtraction_theorem_l357_35733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2angle_BAD_is_zero_l357_35794

noncomputable section

-- Define the points
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the triangles
def triangle_ABC (C : ℝ × ℝ) : Prop :=
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2 ∧  -- isosceles
  (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = 0  -- right angle at C

def triangle_ACD (C D : ℝ × ℝ) : Prop :=
  D.2 = 0 ∧  -- D lies on x-axis
  (C.1 - A.1) * (D.1 - A.1) + (C.2 - A.2) * (D.2 - A.2) = 0 ∧  -- right angle at C
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = (C.1 - D.1)^2 + (C.2 - D.2)^2  -- isosceles

-- Define the angle BAD
noncomputable def angle_BAD (C D : ℝ × ℝ) : ℝ :=
  Real.arccos ((B.1 - A.1) * (D.1 - A.1) / (Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2)))

-- State the theorem
theorem sin_2angle_BAD_is_zero (C D : ℝ × ℝ) :
  triangle_ABC C → triangle_ACD C D → Real.sin (2 * angle_BAD C D) = 0 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2angle_BAD_is_zero_l357_35794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_k_is_15_l357_35728

/-- The equation of line l in the coordinate plane -/
def line_l (x y : ℝ) : Prop := 3 * x - 5 * y + 40 = 0

/-- The point about which the rotation occurs -/
def rotation_point : ℝ × ℝ := (20, 20)

/-- The angle of rotation in radians -/
noncomputable def rotation_angle : ℝ := Real.pi / 4

/-- Line k is obtained by rotating line l counterclockwise by rotation_angle about rotation_point -/
def line_k (x y : ℝ) : Prop :=
  ∃ (x' y' : ℝ), line_l x' y' ∧
  (x - 20) = (x' - 20) * Real.cos rotation_angle - (y' - 20) * Real.sin rotation_angle ∧
  (y - 20) = (x' - 20) * Real.sin rotation_angle + (y' - 20) * Real.cos rotation_angle

/-- The x-intercept of line k -/
def x_intercept_k : ℝ := 15

/-- The main theorem: The x-intercept of line k is 15 -/
theorem x_intercept_of_k_is_15 :
  line_k x_intercept_k 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_k_is_15_l357_35728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_properties_l357_35770

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculate the perimeter of a quadrilateral given its four vertices -/
noncomputable def perimeter (p q r s : Point) : ℝ :=
  distance p q + distance q r + distance r s + distance s p

/-- Calculate the area of a quadrilateral using the shoelace formula -/
def area (p q r s : Point) : ℝ :=
  0.5 * abs ((p.x * q.y + q.x * r.y + r.x * s.y + s.x * p.y) -
             (p.y * q.x + q.y * r.x + r.y * s.x + s.y * p.x))

theorem quadrilateral_properties :
  let p : Point := ⟨1, 2⟩
  let q : Point := ⟨3, 6⟩
  let r : Point := ⟨6, 3⟩
  let s : Point := ⟨8, 1⟩
  (perimeter p q r s = 10 * Real.sqrt 2 + 2 * Real.sqrt 5) ∧
  (area p q r s = 15) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_properties_l357_35770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_sides_l357_35750

noncomputable def is_inscribed_rectangle (r θ x y : ℝ) : Prop :=
  x * y = 2 * r^2 * Real.sin θ ∧
  (x = 2 * y ∨ y = 2 * x) ∧
  x > 0 ∧ y > 0

theorem inscribed_rectangle_sides (r : ℝ) (θ : ℝ) :
  r = 2 * Real.sqrt 5 →
  θ = π / 4 →
  ∃ (x y : ℝ),
    ((x = 2 ∧ y = 4) ∨
    (x = 4 * Real.sqrt (5 / 13) ∧ y = 2 * Real.sqrt (5 / 13))) ∧
    is_inscribed_rectangle r θ x y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_sides_l357_35750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l357_35749

/-- The number of people in the group -/
def groupSize : ℕ := 6

/-- The number of people to be arranged in a row -/
def arrangedSize : ℕ := 4

/-- Calculate the factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Calculate the number of valid arrangements -/
def validArrangements : ℕ :=
  (Nat.choose groupSize arrangedSize) * factorial arrangedSize -
  (Nat.choose groupSize arrangedSize) * (Nat.choose arrangedSize 2) * factorial (arrangedSize - 2) / 2

theorem valid_arrangements_count :
  validArrangements = 288 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l357_35749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_chord_length_l357_35738

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define point Q
def point_Q : ℝ × ℝ := (2, 0)

-- Helper function for Euclidean distance
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the ratio condition for point M
def ratio_condition (M : ℝ × ℝ) : Prop :=
  ∃ (N : ℝ × ℝ), circle_C N.1 N.2 ∧
    (distance M N)^2 / (distance M point_Q)^2 = 2

-- Define the trajectory of M
def trajectory_M (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 7

-- Define the line that intersects the trajectory
def intersecting_line (x y : ℝ) : Prop := y = x - 2

-- Main theorem
theorem trajectory_and_chord_length :
  ∀ (M : ℝ × ℝ),
    ratio_condition M →
    (trajectory_M M.1 M.2) ∧
    (∃ (A B : ℝ × ℝ),
      trajectory_M A.1 A.2 ∧
      trajectory_M B.1 B.2 ∧
      intersecting_line A.1 A.2 ∧
      intersecting_line B.1 B.2 ∧
      distance A B = 2 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_chord_length_l357_35738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_a_not_passing_through_origin_sum_of_logarithms_logarithm_identity_zero_not_in_interval_l357_35795

-- 1. There exists an a such that y = x^a does not pass through (0,0)
theorem existence_of_a_not_passing_through_origin :
  ∃ a : ℝ, ¬(∀ x : ℝ, x^a = 0 → x = 0) :=
by sorry

-- 2. If 10^m = 40 and 10^n = 50, then m + 2n = 5
theorem sum_of_logarithms (m n : ℝ) (h1 : (10 : ℝ)^m = 40) (h2 : (10 : ℝ)^n = 50) :
  m + 2*n = 5 :=
by sorry

-- 3. If lg 2 = a and lg 3 = b, then log_3 6 = (a+b)/b
theorem logarithm_identity (a b : ℝ) (h1 : Real.log 2 / Real.log 10 = a) (h2 : Real.log 3 / Real.log 10 = b) :
  Real.log 6 / Real.log 3 = (a + b) / b :=
by sorry

-- 4. The zero of y = x + ln x is not in the interval (1/4, 1/2)
theorem zero_not_in_interval :
  ¬(∃ x : ℝ, 1/4 < x ∧ x < 1/2 ∧ x + Real.log x = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_a_not_passing_through_origin_sum_of_logarithms_logarithm_identity_zero_not_in_interval_l357_35795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_tiling_l357_35718

noncomputable def internal_angle (n : ℕ) : ℝ := (1 - 2 / n) * Real.pi

noncomputable def vertex_meeting (n : ℕ) : ℝ := 2 + 4 / (n - 2)

noncomputable def vertex_on_side (n : ℕ) : ℝ := 3 + 2 / (n - 2)

def can_tile (n : ℕ) : Prop :=
  (∃ k : ℕ, vertex_meeting n = k) ∨ (∃ k : ℕ, vertex_on_side n = k)

theorem regular_polygon_tiling :
  ∀ n : ℕ, n ≥ 3 → (can_tile n ↔ n = 3 ∨ n = 4 ∨ n = 6) := by
  sorry

#check regular_polygon_tiling

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_tiling_l357_35718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_squares_l357_35715

noncomputable def a : ℝ × ℝ := (0, 1)
noncomputable def b : ℝ × ℝ := (-Real.sqrt 3 / 2, -1 / 2)
noncomputable def c : ℝ × ℝ := (Real.sqrt 3 / 2, -1 / 2)

theorem min_value_of_squares (x y z : ℝ) 
  (h : x • a + y • b + z • c = (1, 2)) :
  ∃ (m : ℝ), (∀ x' y' z' : ℝ, x' • a + y' • b + z' • c = (1, 2) → 
    x'^2 + y'^2 + z'^2 ≥ m) ∧ 
  (∃ x₀ y₀ z₀ : ℝ, x₀ • a + y₀ • b + z₀ • c = (1, 2) ∧ 
    x₀^2 + y₀^2 + z₀^2 = m) ∧ 
  m = 10/3 := by
  sorry

#check min_value_of_squares

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_squares_l357_35715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_l357_35773

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then x^2 + a*x else -x^2 + x

-- State the theorem
theorem odd_function_implies_a_equals_one (a : ℝ) :
  (∀ x, f a (-x) = -(f a x)) → a = 1 :=
by
  intro h
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_l357_35773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_A_l357_35756

/-- Given points A, B, and C in ℝ², where AC:AB = BC:AB = 1:3,
    B = (2, 5), and C = (4, 11), prove that the sum of coordinates of A is -9 -/
theorem sum_of_coordinates_A (A B C : ℝ × ℝ) : 
  (dist A C / dist A B = 1/3) → 
  (dist B C / dist A B = 1/3) → 
  B = (2, 5) → 
  C = (4, 11) → 
  A.1 + A.2 = -9 := by
  sorry

noncomputable def dist (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_A_l357_35756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_l357_35769

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - Real.sqrt 3 * y + 1 = 0

-- Define the inclination angle
def inclination_angle (θ : ℝ) : Prop := θ = 30 * Real.pi / 180

-- Theorem statement
theorem line_inclination :
  ∃ (θ : ℝ), inclination_angle θ ∧
  ∀ (x y : ℝ), line_equation x y → Real.tan θ = Real.sqrt 3 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_l357_35769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationality_preservation_l357_35716

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := ((x - 2) * (x + 1) * (2 * x - 1)) / (x * (x - 1))

-- Theorem statement
theorem rationality_preservation (u v : ℝ) (h_rational_u : ∃ (q : ℚ), u = q) (h_eq : f u = f v) :
  ∃ (q : ℚ), v = q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationality_preservation_l357_35716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_set_size_l357_35787

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def valid_set (s : Finset ℕ) : Prop :=
  ∀ a b c, a ∈ s → b ∈ s → c ∈ s → a ≠ b → b ≠ c → a ≠ c → is_prime (a + b + c)

theorem max_valid_set_size :
  (∃ s : Finset ℕ, valid_set s ∧ s.card = 4) ∧
  (∀ s : Finset ℕ, valid_set s → s.card ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_set_size_l357_35787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unobtainable_value_l357_35796

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (2 - 3*x) / (4*x + 5)

-- State the theorem
theorem unobtainable_value :
  ∀ x : ℝ, x ≠ -5/4 → f x ≠ -3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unobtainable_value_l357_35796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l357_35791

-- Define the function f(x) = x^(-3/5)
noncomputable def f (x : ℝ) : ℝ := x^(-(3/5 : ℝ))

-- State that f is monotonically decreasing
axiom f_decreasing : ∀ x y : ℝ, x < y → f x > f y

-- Define the solution set
def solution_set : Set ℝ := {x | x < -2 ∨ (1 < x ∧ x < 5/2)}

-- Theorem statement
theorem inequality_solution :
  ∀ x : ℝ, (f (x + 2) < f (5 - 2*x)) ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l357_35791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_standard_equation_l357_35740

/-- A parabola with vertex at the origin and axis of symmetry on a coordinate axis. -/
structure Parabola where
  focus : ℝ × ℝ
  axis_of_symmetry : ℝ × ℝ
  vertex_at_origin : focus.1 = 0 ∨ focus.2 = 0
  focus_on_line : 2 * focus.1 - focus.2 - 4 = 0

/-- The standard equation of a parabola. -/
inductive StandardEquation where
  | x_axis : StandardEquation
  | y_axis : StandardEquation

/-- Given a parabola with vertex at the origin, axis of symmetry on a coordinate axis,
    and focus on the line 2x - y - 4 = 0, its standard equation is either y² = 8x or x² = -16y. -/
theorem parabola_standard_equation (p : Parabola) : 
  (∃ (k : ℝ), k = 8 ∧ StandardEquation.x_axis = StandardEquation.x_axis) ∨ 
  (∃ (k : ℝ), k = -16 ∧ StandardEquation.y_axis = StandardEquation.y_axis) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_standard_equation_l357_35740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_range_l357_35757

noncomputable section

/-- Line l: y = √3 x + 4 -/
def line_l (x y : ℝ) : Prop := y = Real.sqrt 3 * x + 4

/-- Circle O: x² + y² = r², where 1 < r < 2 -/
def circle_O (x y r : ℝ) : Prop := x^2 + y^2 = r^2 ∧ 1 < r ∧ r < 2

/-- Rhombus ABCD with one interior angle of 60° -/
def rhombus_ABCD (A B C D : ℝ × ℝ) : Prop :=
  ∃ (angle : ℝ), angle = 60 * Real.pi / 180 ∧
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (C.1 - D.1)^2 + (C.2 - D.2)^2 ∧
  (A.1 - D.1)^2 + (A.2 - D.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2

/-- Area S of rhombus ABCD -/
noncomputable def area_S (A B C D : ℝ × ℝ) : ℝ :=
  Real.sqrt 3 / 2 * ((A.1 - C.1)^2 + (A.2 - C.2)^2)

theorem rhombus_area_range :
  ∀ (A B C D : ℝ × ℝ) (r : ℝ),
    line_l A.1 A.2 → line_l B.1 B.2 →
    circle_O C.1 C.2 r → circle_O D.1 D.2 r →
    rhombus_ABCD A B C D →
    (0 < area_S A B C D ∧ area_S A B C D < 3/2 * Real.sqrt 3) ∨
    (3/2 * Real.sqrt 3 < area_S A B C D ∧ area_S A B C D < 6 * Real.sqrt 3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_range_l357_35757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equality_time_l357_35752

/-- The angle between hour and minute hands at 17:00 -/
def initial_angle : ℚ := 150

/-- Degrees the minute hand moves per minute -/
def minute_hand_speed : ℚ := 6

/-- Degrees the hour hand moves per minute -/
def hour_hand_speed : ℚ := 1/2

/-- The number of minutes after 17:00 when the angle between
    hour and minute hands is the same as at 17:00 -/
noncomputable def time_to_same_angle : ℚ := 54 + 6/11

theorem angle_equality_time :
  let x := time_to_same_angle
  (x * minute_hand_speed - (initial_angle + x * hour_hand_speed)) % 360 = initial_angle :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equality_time_l357_35752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_problem_l357_35767

theorem max_value_problem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c)
    (h4 : a + b + c = 1) (h5 : a * b * c = 1 / 27) :
  a + Real.sqrt (a^2 * b) + (a * b * c)^(1/3) ≤ 101 / 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_problem_l357_35767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heartsuit_sum_l357_35735

noncomputable def heartsuit (x : ℝ) : ℝ := (x + x^2 + x^3) / 3

theorem heartsuit_sum : heartsuit 1 + heartsuit 2 + heartsuit 4 = 101 / 3 := by
  -- Unfold the definition of heartsuit
  unfold heartsuit
  -- Simplify the expressions
  simp
  -- Perform the arithmetic
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_heartsuit_sum_l357_35735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_symmetry_l357_35744

/-- A power function f(x) = (m^2 - m - 1)x^(1-m) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * (x^(1 - m))

/-- Symmetry about y-axis means f(x) = f(-x) for all x -/
def symmetricAboutYAxis (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The theorem states that the power function f is symmetric about the y-axis
    if and only if m = -1 -/
theorem power_function_symmetry :
  ∃ m : ℝ, symmetricAboutYAxis (f m) ↔ m = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_symmetry_l357_35744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_marked_squares_for_9x9_board_l357_35780

/-- Represents a rectangular board --/
structure Board where
  rows : ℕ
  cols : ℕ

/-- Represents a rectangle that can be placed on the board --/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Checks if a position is valid for placing a rectangle on the board --/
def is_valid_position (b : Board) (r : Rectangle) (pos : ℕ × ℕ) : Prop :=
  pos.1 + r.length ≤ b.rows ∧ pos.2 + r.width ≤ b.cols

/-- Returns the set of squares covered by a rectangle at a given position --/
def covered_squares (b : Board) (r : Rectangle) (pos : ℕ × ℕ) : Finset (ℕ × ℕ) :=
  sorry

/-- Checks if a given number of marked squares is sufficient to determine
    the position of a rectangle on the board --/
def is_sufficient_marking (b : Board) (r : Rectangle) (k : ℕ) : Prop :=
  ∀ (marked_squares : Finset (ℕ × ℕ)),
    marked_squares.card = k →
    ∀ (pos1 pos2 : ℕ × ℕ),
      (is_valid_position b r pos1 ∧ is_valid_position b r pos2) →
      (covered_squares b r pos1 ∩ marked_squares ≠
       covered_squares b r pos2 ∩ marked_squares) →
      pos1 = pos2

/-- The main theorem: 40 is the minimum number of squares Petya needs to mark --/
theorem min_marked_squares_for_9x9_board :
  let b : Board := ⟨9, 9⟩
  let r : Rectangle := ⟨1, 4⟩
  ∀ k : ℕ, k < 40 → ¬(is_sufficient_marking b r k) ∧
            is_sufficient_marking b r 40 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_marked_squares_for_9x9_board_l357_35780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l357_35702

-- Define the circle
def myCircle (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 4*y = 0

-- Define the line
def myLine (x y : ℝ) : Prop :=
  x - y - 2 = 0

-- State the theorem
theorem chord_length :
  ∃ (A B : ℝ × ℝ),
    myCircle A.1 A.2 ∧ myCircle B.1 B.2 ∧
    myLine A.1 A.2 ∧ myLine B.1 B.2 ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2)^(1/2 : ℝ) = 2 * (6 : ℝ)^(1/2 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l357_35702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_Z_equals_sqrt5_over_5_quadratic_root_implies_p_and_q_l357_35755

-- Define the complex number Z
noncomputable def Z : ℂ := (1 + 2 * Complex.I) / (3 - 4 * Complex.I)

-- Define the quadratic equation
def quadratic (p q : ℝ) (x : ℂ) : ℂ := x^2 + p*x + q

-- Theorem 1
theorem abs_Z_equals_sqrt5_over_5 :
  Complex.abs Z = Real.sqrt 5 / 5 := by sorry

-- Theorem 2
theorem quadratic_root_implies_p_and_q (p q : ℝ) :
  quadratic p q (2 - 3*Complex.I) = 0 → p = -4 ∧ q = 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_Z_equals_sqrt5_over_5_quadratic_root_implies_p_and_q_l357_35755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_three_l357_35722

/-- The function f as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := x⁻¹ + x⁻¹ / (2 + x⁻¹)

/-- Theorem stating that f(f(3)) = 651/260 -/
theorem f_f_three : f (f 3) = 651 / 260 := by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_three_l357_35722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_masses_l357_35784

/-- The mass of a single green apple in kilograms -/
def G : ℚ := sorry

/-- The mass of a single yellow apple in kilograms -/
def Y : ℚ := sorry

/-- The mass of a single red apple in kilograms -/
def R : ℚ := sorry

/-- The condition that 3 green apples have a whole number mass -/
axiom green_whole : ∃ n : ℕ, 3 * G = n

/-- The condition that 5 yellow apples have a whole number mass -/
axiom yellow_whole : ∃ n : ℕ, 5 * Y = n

/-- The condition that 7 red apples have a whole number mass -/
axiom red_whole : ∃ n : ℕ, 7 * R = n

/-- The condition that the total mass of one of each apple is not a whole number -/
axiom not_whole_sum : ∀ n : ℕ, G + Y + R ≠ n

/-- The condition that the total mass of one of each apple is approximately 1.16 kg -/
axiom approx_sum : abs ((G + Y + R) - 1.16) < 0.01

/-- The theorem to be proved -/
theorem apple_masses : 3 * G = 1 ∧ 5 * Y = 2 ∧ 7 * R = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_masses_l357_35784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_correct_l357_35747

/-- Line l in the Cartesian plane -/
def line_l (x y : ℝ) : Prop := x + 2*y = 0

/-- Circle C in the Cartesian plane -/
noncomputable def circle_C (x y : ℝ) : Prop := (x - Real.sqrt 2 / 2)^2 + (y - Real.sqrt 2 / 2)^2 = 1

/-- The length of the chord formed by the intersection of line l and circle C -/
noncomputable def chord_length : ℝ := Real.sqrt 10 / 5

/-- Theorem stating that the chord length is correct -/
theorem chord_length_is_correct :
  ∃ (A B : ℝ × ℝ),
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2) = chord_length^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_correct_l357_35747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vector_scalar_l357_35774

/-- Given vectors a, b, and c in ℝ², prove that if a + λb is parallel to c, then λ = -1 -/
theorem parallel_vector_scalar (a b c : ℝ × ℝ) (h : a = (-2, 1) ∧ b = (1, 3) ∧ c = (3, 2)) :
  (∃ l : ℝ, ∃ k : ℝ, k ≠ 0 ∧ a.1 + l * b.1 = k * c.1 ∧ a.2 + l * b.2 = k * c.2) →
  (∃ l : ℝ, l = -1 ∧ ∃ k : ℝ, k ≠ 0 ∧ a.1 + l * b.1 = k * c.1 ∧ a.2 + l * b.2 = k * c.2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vector_scalar_l357_35774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l357_35732

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + Real.sin (2 * x)

-- State the theorem
theorem f_properties :
  -- 1. Simplified form of f(x)
  (∀ x, f x = Real.sqrt 2 * Real.sin (2 * x + Real.pi / 4) + 1) ∧
  -- 2. Minimum value of f(x)
  (∃ y_min, ∀ x, f x ≥ y_min ∧ y_min = -Real.sqrt 2 + 1) ∧
  -- 3. x values where f(x) attains its minimum
  (∀ k : ℤ, f (-3 * Real.pi / 8 + k * Real.pi) = -Real.sqrt 2 + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l357_35732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l357_35797

def z : ℂ := sorry

theorem z_in_fourth_quadrant :
  (1 + 2*I : ℂ) * z = Complex.abs (1 + 3*I : ℂ)^2 →
  z.re > 0 ∧ z.im < 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l357_35797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_calculation_l357_35775

theorem election_votes_calculation (percentage : Real) (majority : ℕ) : 
  percentage = 0.84 →
  majority = 476 →
  ∃ total_votes : ℕ,
    percentage * total_votes - (1 - percentage) * total_votes = majority ∧
    total_votes = 700 := by
  sorry

#check election_votes_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_calculation_l357_35775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_ratio_is_one_fourth_l357_35758

/-- Represents the properties of a cylindrical barrel -/
structure Barrel where
  volume : ℝ
  radius : ℝ
  height : ℝ
  iron_price : ℝ

/-- Calculates the cost of the barrel -/
noncomputable def barrel_cost (b : Barrel) : ℝ :=
  b.iron_price * (2 * Real.pi * b.radius * b.height + Real.pi * b.radius^2) +
  3 * b.iron_price * Real.pi * b.radius^2

/-- Theorem stating the optimal ratio of radius to height -/
theorem optimal_ratio_is_one_fourth (b : Barrel) 
  (h_volume : b.volume = Real.pi * b.radius^2 * b.height)
  (h_positive : b.volume > 0 ∧ b.radius > 0 ∧ b.height > 0 ∧ b.iron_price > 0) :
  ∀ b' : Barrel, barrel_cost b ≤ barrel_cost b' → b.radius / b.height = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_ratio_is_one_fourth_l357_35758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_b_value_l357_35710

/-- The number of triangles satisfying the given conditions -/
def num_triangles : ℕ := 1990

/-- The minimal value of b -/
def min_b : ℕ := 1991^2

/-- Theorem stating the minimal value of b for which there exist exactly 1990 triangles
    ABC with integral side-lengths, satisfying ∠ABC = 1/2 * ∠BAC and AC = b, is 1991² -/
theorem minimal_b_value :
  ∀ b : ℕ,
  (∃! triangles : Finset (ℕ × ℕ × ℕ),
    (triangles.card = num_triangles) ∧
    (∀ a c : ℕ, (a, b, c) ∈ triangles →
      ∃ angle_bac : ℚ, angle_bac > 0 ∧ angle_bac < Real.pi ∧
        (2 * (Real.pi - angle_bac/2 - angle_bac) = angle_bac/2) ∧
        (a^2 + b^2 = c^2 + 2*a*b*(Real.cos (angle_bac/2))))) →
  b ≥ min_b :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_b_value_l357_35710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_on_line_trig_identity_l357_35786

theorem angle_on_line_trig_identity (α : Real) :
  (∃ (x y : Real), y = 2 * x ∧ x = Real.cos α ∧ y = Real.sin α) →
  1 + Real.sin α * Real.cos α = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_on_line_trig_identity_l357_35786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_20_eq_neg_sqrt_3_l357_35792

/-- The sequence {aₙ} defined by the given recurrence relation -/
noncomputable def a : ℕ → ℝ
  | 0 => 0  -- Define for n = 0 (which corresponds to a₁ in the problem)
  | (n + 1) => (a n - Real.sqrt 3) / (Real.sqrt 3 * a n + 1)

/-- Theorem stating that the 20th term of the sequence is equal to -√3 -/
theorem a_20_eq_neg_sqrt_3 : a 19 = -Real.sqrt 3 := by
  sorry

/-- Lemma stating the periodicity of the sequence -/
lemma a_periodic (n : ℕ) : a (n + 3) = a n := by
  sorry

/-- Lemma for the first three terms of the sequence -/
lemma a_first_three :
  a 0 = 0 ∧ a 1 = -Real.sqrt 3 ∧ a 2 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_20_eq_neg_sqrt_3_l357_35792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_difference_main_theorem_l357_35706

theorem log_sum_difference (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.log x + Real.log y - Real.log z = Real.log ((x * y) / z) :=
by sorry

theorem main_theorem : 
  Real.log 50 + Real.log 45 - Real.log 9 = 1 + 2 * Real.log 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_difference_main_theorem_l357_35706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_prime_factor_l357_35768

def sequence_a : ℕ → ℕ
  | 0 => 1  -- Define for 0 to cover all natural numbers
  | 1 => 1
  | 2 => 2
  | (n+3) => 2 * sequence_a (n+2) + sequence_a (n+1)

theorem sequence_a_prime_factor (n : ℕ) (h : n ≥ 5) :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ sequence_a n ∧ p % 4 = 1 := by
  sorry

#eval sequence_a 5  -- Optional: to check if the function works correctly

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_prime_factor_l357_35768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_multiple_of_210_l357_35743

def n : ℕ := 2^12 * 3^15 * 5^9 * 7^3

theorem factors_multiple_of_210 : 
  (Finset.filter (fun x => x ∣ n ∧ 210 ∣ x) (Finset.range (n + 1))).card = 4860 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_multiple_of_210_l357_35743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_pie_pieces_l357_35727

theorem minimal_pie_pieces (p q : ℕ) (h_coprime : Nat.Coprime p q) (h_pos_p : p > 0) (h_pos_q : q > 0) :
  ∃ (n : ℕ), n = p + q - 1 ∧
  (∀ (k : ℕ), k < n → (¬(p ∣ k) ∨ ¬(q ∣ k))) ∧
  (p ∣ n ∧ q ∣ n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_pie_pieces_l357_35727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_non_primes_40_to_80_l357_35717

def is_prime (n : ℕ) : Bool :=
  if n ≤ 1 then false
  else
    (List.range (n - 1)).all (fun d => if d > 1 then n % (d + 1) ≠ 0 else true)

def sum_non_primes (a b : ℕ) : ℕ :=
  (List.range (b - a - 1)).map (fun i => a + i + 1)
    |>.filter (fun n => !is_prime n)
    |>.sum

theorem sum_non_primes_40_to_80 :
  sum_non_primes 40 80 = 1746 := by
  sorry

#eval sum_non_primes 40 80

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_non_primes_40_to_80_l357_35717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_functional_equation_solution_unique_l357_35754

theorem functional_equation_solution :
  ∃ f : ℝ → ℝ, ∀ x y : ℝ, f (f x + y) = 2 * x + f (f y - x) :=
sorry

theorem functional_equation_solution_unique (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f x + y) = 2 * x + f (f y - x)) : 
  ∃ a : ℝ, ∀ x : ℝ, f x = x - a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_functional_equation_solution_unique_l357_35754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_circle_equation_max_distance_to_line_l357_35783

/-- The circle equation in polar coordinates -/
def polar_circle (ρ θ : ℝ) : Prop := ρ = 10 * Real.cos (Real.pi/3 - θ)

/-- The line equation -/
def line (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 2 = 0

/-- Theorem stating the equivalent Cartesian equation of the circle -/
theorem cartesian_circle_equation (x y : ℝ) :
  (∃ θ, polar_circle (Real.sqrt (x^2 + y^2)) θ) ↔ x^2 + y^2 - 5*x - 5*Real.sqrt 3*y = 0 :=
sorry

/-- Theorem stating the maximum distance from any point on the circle to the line -/
theorem max_distance_to_line :
  (∃ x y : ℝ, (∃ θ, polar_circle (Real.sqrt (x^2 + y^2)) θ) ∧
    ∀ x' y' : ℝ, (∃ θ', polar_circle (Real.sqrt (x'^2 + y'^2)) θ') →
      Real.sqrt ((x - x')^2 + (y - y')^2) ≤
        |Real.sqrt 3 * x' - y' + 2| / Real.sqrt (3 + 1)) ∧
  (∀ x y : ℝ, (∃ θ, polar_circle (Real.sqrt (x^2 + y^2)) θ) →
    |Real.sqrt 3 * x - y + 2| / Real.sqrt (3 + 1) ≤ 6) ∧
  (∃ x y : ℝ, (∃ θ, polar_circle (Real.sqrt (x^2 + y^2)) θ) ∧
    |Real.sqrt 3 * x - y + 2| / Real.sqrt (3 + 1) = 6) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_circle_equation_max_distance_to_line_l357_35783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_k_sum_l357_35746

/-- Area of a triangle given three points in 2D space --/
def area_triangle (p₁ p₂ p₃ : ℤ × ℤ) : ℚ :=
  let (x₁, y₁) := p₁
  let (x₂, y₂) := p₂
  let (x₃, y₃) := p₃
  (1/2 : ℚ) * ((x₁ * y₂ + x₂ * y₃ + x₃ * y₁ - y₁ * x₂ - y₂ * x₃ - y₃ * x₁).natAbs : ℚ)

/-- The sum of k values that maximize the triangle area --/
theorem max_area_k_sum : ∃ (k₁ k₂ : ℤ),
  (∀ k : ℤ, area_triangle (2, 8) (14, 17) (6, k) ≤ area_triangle (2, 8) (14, 17) (6, k₁)) ∧
  (∀ k : ℤ, area_triangle (2, 8) (14, 17) (6, k) ≤ area_triangle (2, 8) (14, 17) (6, k₂)) ∧
  k₁ + k₂ = 22 := by
  sorry

#eval area_triangle (2, 8) (14, 17) (6, 10)
#eval area_triangle (2, 8) (14, 17) (6, 11)
#eval area_triangle (2, 8) (14, 17) (6, 12)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_k_sum_l357_35746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_towel_area_decrease_l357_35707

theorem towel_area_decrease (L B : ℝ) (hL : L > 0) (hB : B > 0) : 
  (L * B - 0.8 * L * 0.9 * B) / (L * B) = 0.28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_towel_area_decrease_l357_35707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_defective_units_shipped_percentage_l357_35798

/-- Represents the percentage of units with Type A defects -/
def type_a_defect_rate : ℚ := 7/100

/-- Represents the percentage of units with Type B defects -/
def type_b_defect_rate : ℚ := 8/100

/-- Represents the percentage of Type A defective units shipped for sale -/
def type_a_ship_rate : ℚ := 3/100

/-- Represents the percentage of Type B defective units shipped for sale -/
def type_b_ship_rate : ℚ := 6/100

/-- Theorem stating that the percentage of defective units shipped for sale is 1% -/
theorem defective_units_shipped_percentage :
  Int.floor (100 * (type_a_defect_rate * type_a_ship_rate + type_b_defect_rate * type_b_ship_rate)) = 1 := by
  sorry

#check defective_units_shipped_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_defective_units_shipped_percentage_l357_35798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l357_35778

noncomputable def f (x : ℝ) := Real.sqrt (2 * x + 1) + Real.log (3 - 4 * x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -1/2 ≤ x ∧ x < 3/4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l357_35778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_sales_quota_l357_35736

/-- Alice's shoe sales problem -/
theorem alice_sales_quota : ℕ := by
  let adidas_price : ℕ := 45
  let nike_price : ℕ := 60
  let reebok_price : ℕ := 35
  let nike_sold : ℕ := 8
  let adidas_sold : ℕ := 6
  let reebok_sold : ℕ := 9
  let amount_above_goal : ℕ := 65
  let total_sales : ℕ := nike_price * nike_sold + adidas_price * adidas_sold + reebok_price * reebok_sold
  have h : total_sales - amount_above_goal = 1000 := by sorry
  exact 1000


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_sales_quota_l357_35736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_heartsuit_l357_35703

noncomputable def heartsuit (x : ℝ) : ℝ := (x + x^2) / 2

theorem sum_of_heartsuit : heartsuit 2 + heartsuit 3 + heartsuit 4 + heartsuit 5 = 34 := by
  -- Unfold the definition of heartsuit
  unfold heartsuit
  -- Simplify the arithmetic expressions
  simp [add_div, pow_two]
  -- Perform the final calculation
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_heartsuit_l357_35703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ones_minus_twos_is_perfect_square_l357_35704

/-- For any positive integer n, the number formed by 2n ones minus the number formed by n twos is always a perfect square. -/
theorem ones_minus_twos_is_perfect_square (n : ℕ) :
  ∃ k : ℕ, (10^(2*n) - 1) - 2 * (10^n - 1) = k^2 := by
  sorry

#check ones_minus_twos_is_perfect_square

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ones_minus_twos_is_perfect_square_l357_35704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_r_l357_35721

/-- Represents a geometric series with first term a and common ratio r -/
def geometric_series (a r : ℝ) : ℕ → ℝ := λ n ↦ a * r^n

/-- Sum of the geometric series up to infinity, when |r| < 1 -/
noncomputable def geometric_sum (a r : ℝ) : ℝ := a / (1 - r)

/-- Sum of the odd-power terms in the geometric series -/
noncomputable def odd_power_sum (a r : ℝ) : ℝ := (a * r) / (1 - r^2)

theorem geometric_series_r (a r : ℝ) 
  (h1 : geometric_sum a r = 18)
  (h2 : odd_power_sum a r = 8) :
  r = 4/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_r_l357_35721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ginas_facilities_fee_l357_35776

/-- Calculates the facilities fee for Gina's college expenses --/
theorem ginas_facilities_fee 
  (num_credits : ℕ) 
  (cost_per_credit : ℕ) 
  (num_textbooks : ℕ) 
  (cost_per_textbook : ℕ) 
  (total_cost : ℕ) 
  (h1 : num_credits = 14)
  (h2 : cost_per_credit = 450)
  (h3 : num_textbooks = 5)
  (h4 : cost_per_textbook = 120)
  (h5 : total_cost = 7100) :
  total_cost - (num_credits * cost_per_credit + num_textbooks * cost_per_textbook) = 200 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ginas_facilities_fee_l357_35776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_blocks_with_two_differences_l357_35729

/-- Represents the characteristics of a block -/
structure Block where
  material : Fin 2
  size : Fin 3
  color : Fin 4
  shape : Fin 4
  finish : Fin 2
deriving Fintype, DecidableEq

/-- The reference block: 'plastic medium red circle glossy' -/
def referenceBlock : Block := {
  material := 0,
  size := 1,
  color := 2,
  shape := 0,
  finish := 0
}

/-- Counts the number of differences between two blocks -/
def countDifferences (b1 b2 : Block) : Nat :=
  (if b1.material ≠ b2.material then 1 else 0) +
  (if b1.size ≠ b2.size then 1 else 0) +
  (if b1.color ≠ b2.color then 1 else 0) +
  (if b1.shape ≠ b2.shape then 1 else 0) +
  (if b1.finish ≠ b2.finish then 1 else 0)

/-- The set of all possible blocks -/
def allBlocks : Finset Block := Finset.univ

theorem count_blocks_with_two_differences :
  (allBlocks.filter (fun b => countDifferences b referenceBlock = 2)).card = 51 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_blocks_with_two_differences_l357_35729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_vertex_of_square_l357_35739

def square_vertices (a b c d : ℂ) : Prop :=
  (Set.ncard {a, b, c, d} = 4) ∧
  (Complex.abs (a - b) = Complex.abs (b - c)) ∧
  (Complex.abs (b - c) = Complex.abs (c - d)) ∧
  (Complex.abs (c - d) = Complex.abs (d - a)) ∧
  (Complex.abs (a - b) = Complex.abs (b - d))

theorem fourth_vertex_of_square :
  ∃ (d : ℂ), d = (1 - 4*I) ∧ square_vertices (3 + 3*I) (-1 + 4*I) (-3 + I) d := by
  sorry

#check fourth_vertex_of_square

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_vertex_of_square_l357_35739
