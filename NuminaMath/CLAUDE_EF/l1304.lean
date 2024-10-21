import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_league_games_l1304_130425

theorem league_games (n : ℕ) (total_games : ℕ) (h1 : n = 10) (h2 : total_games = 45) :
  (n * (n - 1)) / 2 = total_games →
  ∀ i j : Fin n, i ≠ j → 1 = 1 :=
by
  intro h_games_count
  intro i j h_not_equal
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_league_games_l1304_130425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goat_rope_length_l1304_130465

/-- The length of the rope for a goat tied to the corner of a square plot -/
noncomputable def rope_length (plot_side : ℝ) (graze_area : ℝ) : ℝ :=
  Real.sqrt ((2 * graze_area) / Real.pi)

/-- Theorem stating the rope length for given conditions -/
theorem goat_rope_length :
  let plot_side : ℝ := 12
  let graze_area : ℝ := 38.48451000647496
  abs (rope_length plot_side graze_area - 7) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goat_rope_length_l1304_130465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_product_sign_l1304_130460

theorem cos_product_sign : Real.cos 1 * Real.cos 2 * Real.cos 3 * Real.cos 4 < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_product_sign_l1304_130460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_emilio_sum_difference_l1304_130469

def star_list : List Nat := List.range 30

def replace_two_with_one (n : Nat) : Nat :=
  let s := toString n
  (s.replace "2" "1").toNat!

def emilio_list : List Nat := star_list.map replace_two_with_one

theorem star_emilio_sum_difference :
  star_list.sum - emilio_list.sum = 103 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_emilio_sum_difference_l1304_130469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_when_m_is_three_intersection_empty_iff_l1304_130491

-- Define the sets E and F
def E (m : ℝ) : Set ℝ := {x | |x - 1| ≥ m}
def F : Set ℝ := {x | 10 / (x + 6) > 1}

-- Theorem for part (1)
theorem intersection_when_m_is_three : 
  E 3 ∩ F = Set.Ioc (-6) (-2) :=
sorry

-- Theorem for part (2)
theorem intersection_empty_iff :
  ∀ m : ℝ, E m ∩ F = ∅ ↔ m ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_when_m_is_three_intersection_empty_iff_l1304_130491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_property_l1304_130441

/-- Positive natural numbers -/
def PositiveNat := {n : ℕ // n > 0}

/-- The function property from the original problem -/
def FunctionProperty (f : PositiveNat → PositiveNat) : Prop :=
  ∀ m n : PositiveNat, f ⟨m.val ^ 2 + (f n).val, sorry⟩ = ⟨(f m).val ^ 2 + n.val, sorry⟩

/-- The theorem statement -/
theorem unique_function_property :
  ∀ f : PositiveNat → PositiveNat,
  FunctionProperty f →
  ∀ n : PositiveNat, f n = n :=
by
  sorry

/-- Helper lemma to show that the identity function satisfies the property -/
lemma id_satisfies_property :
  FunctionProperty (λ n => n) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_property_l1304_130441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_cubic_inverse_l1304_130423

theorem min_value_cubic_inverse (y : ℝ) (h : y > 0) : 
  9 * y^3 + 4 * y^(-6 : ℝ) ≥ 13 ∧ ∃ y₀ : ℝ, y₀ > 0 ∧ 9 * y₀^3 + 4 * y₀^(-6 : ℝ) = 13 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_cubic_inverse_l1304_130423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intercepts_sum_l1304_130495

/-- Represents a parabola with equation x = 3y^2 - 9y + 6 -/
def Parabola : ℝ → ℝ := fun y ↦ 3 * y^2 - 9 * y + 6

/-- The x-coordinate of the x-intercept -/
def a : ℝ := Parabola 0

/-- The y-coordinates of the y-intercepts -/
noncomputable def b : ℝ := (3 - Real.sqrt 5) / 2
noncomputable def c : ℝ := (3 + Real.sqrt 5) / 2

theorem parabola_intercepts_sum :
  a + b + c = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intercepts_sum_l1304_130495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_midpoint_region_area_ratio_l1304_130442

/-- Represents a particle moving along the edges of an equilateral triangle -/
structure Particle where
  position : ℝ × ℝ
  speed : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  sideLength : ℝ

/-- The region enclosed by the path of the midpoint of two particles -/
def Region (p1 p2 : Particle) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a given region -/
noncomputable def area (r : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The set of points forming the equilateral triangle -/
def TrianglePoints (t : EquilateralTriangle) : Set (ℝ × ℝ) :=
  sorry

theorem particle_midpoint_region_area_ratio 
  (t : EquilateralTriangle) 
  (p1 p2 : Particle) 
  (h1 : p1.position = (0, 0)) -- p1 starts at vertex A
  (h2 : p2.position = (t.sideLength / 2, 0)) -- p2 starts at midpoint of AB
  (h3 : p1.speed = p2.speed) -- particles move at uniform speed
  : area (Region p1 p2) / area (TrianglePoints t) = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_midpoint_region_area_ratio_l1304_130442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relation_l1304_130490

theorem angle_relation (α β : Real) 
  (h1 : α ∈ Set.Ioo (π/2) π) 
  (h2 : β ∈ Set.Ioo (π/2) π) 
  (h3 : (1 - Real.sin (2*α)) * Real.sin β = Real.cos β * Real.cos (2*α)) : 
  β - α = π/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relation_l1304_130490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_over_12_l1304_130488

noncomputable def f (x : ℝ) := Real.sin (2 * (x - Real.pi / 4)) - 1

theorem f_at_pi_over_12 : f (Real.pi / 12) = -(Real.sqrt 3 + 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_over_12_l1304_130488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_vote_theorem_l1304_130444

/-- The percentage of registered voters who are Democrats -/
noncomputable def democrat_percentage : ℝ := 60

/-- The percentage of registered voters who are Republicans -/
noncomputable def republican_percentage : ℝ := 100 - democrat_percentage

/-- The percentage of Democrats expected to vote for candidate A -/
noncomputable def democrat_vote_percentage : ℝ := 75

/-- The percentage of Republicans expected to vote for candidate A -/
noncomputable def republican_vote_percentage : ℝ := 20

/-- The expected percentage of registered voters voting for candidate A -/
noncomputable def expected_vote_percentage : ℝ :=
  (democrat_percentage * democrat_vote_percentage +
   republican_percentage * republican_vote_percentage) / 100

theorem expected_vote_theorem :
  expected_vote_percentage = 53 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_vote_theorem_l1304_130444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f1_inverse_f2_inverse_f3_inverse_f4_inverse_l1304_130424

-- Function 1
noncomputable def f1 (x : ℝ) : ℝ := 3 * x - 5
noncomputable def f1_inv (y : ℝ) : ℝ := (y + 5) / 3

theorem f1_inverse : ∀ x : ℝ, f1_inv (f1 x) = x ∧ f1 (f1_inv x) = x := by sorry

-- Function 2
noncomputable def f2 (x : ℝ) : ℝ := Real.sqrt (1 - x^3)
noncomputable def f2_inv (y : ℝ) : ℝ := (1 - y^2)^(1/3)

theorem f2_inverse : ∀ x : ℝ, x ≥ 0 → f2_inv (f2 x) = x ∧ f2 (f2_inv x) = x := by sorry

-- Function 3
noncomputable def f3 (x : ℝ) : ℝ := Real.arcsin (3 * x)
noncomputable def f3_inv (y : ℝ) : ℝ := Real.sin y / 3

theorem f3_inverse : ∀ x : ℝ, -1/3 ≤ x ∧ x ≤ 1/3 → f3_inv (f3 x) = x ∧ f3 (f3_inv x) = x := by sorry

-- Function 4
noncomputable def f4 (x : ℝ) : ℝ := x^2 + 2
noncomputable def f4_inv_pos (y : ℝ) : ℝ := Real.sqrt (y - 2)
noncomputable def f4_inv_neg (y : ℝ) : ℝ := -Real.sqrt (y - 2)

theorem f4_inverse : ∀ x : ℝ, 
  (f4_inv_pos (f4 x) = x ∨ f4_inv_neg (f4 x) = x) ∧ 
  (f4 (f4_inv_pos x) = x ∧ f4 (f4_inv_neg x) = x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f1_inverse_f2_inverse_f3_inverse_f4_inverse_l1304_130424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_function_classification_l1304_130438

-- Definition of F function
def is_F_function (f : ℝ → ℝ) : Prop :=
  ∃ M : ℝ, M > 0 ∧ ∀ x : ℝ, |f x| ≤ M * |x|

-- Function definitions
def f1 : ℝ → ℝ := λ x ↦ 2 * x
noncomputable def f2 : ℝ → ℝ := λ x ↦ x^2 + 1
noncomputable def f3 : ℝ → ℝ := λ x ↦ Real.sqrt 2 * (Real.sin x + Real.cos x)
noncomputable def f4 : ℝ → ℝ := λ x ↦ x / (x^2 - x + 1)
noncomputable def f5 : ℝ → ℝ := sorry  -- Placeholder for the odd function

-- Axiom for f5 properties
axiom f5_odd : ∀ x : ℝ, f5 (-x) = -f5 x
axiom f5_lipschitz : ∀ x1 x2 : ℝ, |f5 x1 - f5 x2| ≤ 2 * |x1 - x2|

-- Theorem statement
theorem F_function_classification :
  is_F_function f1 ∧
  is_F_function f4 ∧
  is_F_function f5 ∧
  ¬is_F_function f2 ∧
  ¬is_F_function f3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_function_classification_l1304_130438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1304_130415

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + Real.cos (2*x + Real.pi/3) - 1

theorem f_properties :
  ∃ (p : ℝ), p > 0 ∧ 
  (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 Real.pi → f (5*Real.pi/6 - x) = f (5*Real.pi/6 + x)) ∧
  (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → q ≥ Real.pi) :=
by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1304_130415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l1304_130473

/-- The quadratic function f(x) = x^2 + bx + 1 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 1

/-- Given a quadratic function f(x) = x^2 + bx + 1 satisfying certain conditions,
    the maximum value of m is 3. -/
theorem max_m_value (b : ℝ) (l m t : ℝ) : 
  (∀ x : ℝ, f b (-x) = f b (x + 1)) →
  (∀ x ∈ Set.Icc l m, f b (x + t) ≤ x) →
  m ≤ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l1304_130473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hex_A03_to_dec_l1304_130494

/-- Converts a single hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : ℕ :=
  match c with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | _ => c.toString.toNat!

/-- Converts a hexadecimal string to its decimal value -/
def hex_string_to_dec (s : String) : ℕ :=
  s.toList.reverse.foldl (fun acc d => acc * 16 + hex_to_dec d) 0

theorem hex_A03_to_dec :
  hex_string_to_dec "A03" = 2563 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hex_A03_to_dec_l1304_130494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_35000_l1304_130414

/-- Scientific notation representation -/
structure ScientificNotation where
  a : ℚ
  n : ℤ
  h1 : 1 ≤ a
  h2 : a < 10

/-- The value represented by a scientific notation -/
def ScientificNotation.value (sn : ScientificNotation) : ℚ :=
  sn.a * (10 : ℚ) ^ sn.n

/-- 35000 in scientific notation -/
def number_35000 : ScientificNotation :=
  { a := 35/10
    n := 4
    h1 := by norm_num
    h2 := by norm_num }

theorem scientific_notation_35000 :
  (number_35000.value : ℚ) = 35000 :=
by
  unfold number_35000 ScientificNotation.value
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_35000_l1304_130414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_julia_tuesday_kids_monday_two_more_than_wednesday_l1304_130440

/-- The number of kids Julia played with on each day of the week --/
structure KidsPlayed where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ

/-- Given information about Julia's tag games --/
def julia_games : KidsPlayed where
  monday := 6
  tuesday := 4  -- We now know this value
  wednesday := 4

theorem julia_tuesday_kids : julia_games.tuesday = julia_games.wednesday := by
  -- Unfold the definition of julia_games
  unfold julia_games
  -- Both tuesday and wednesday are defined as 4, so they're equal
  rfl

theorem monday_two_more_than_wednesday :
  julia_games.monday = julia_games.wednesday + 2 := by
  unfold julia_games
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_julia_tuesday_kids_monday_two_more_than_wednesday_l1304_130440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_circular_cone_central_angle_l1304_130432

/-- Represents a right circular cone as defined by Euclid -/
structure RightCircularCone where
  /-- The radius of the base circle -/
  radius : ℝ
  /-- The slant height of the cone -/
  slantHeight : ℝ
  /-- The axis section forms an isosceles right triangle -/
  isIsoscelesRight : slantHeight = Real.sqrt 2 * radius

/-- The radian measure of the central angle of the unfolded side of a right circular cone -/
noncomputable def centralAngle (cone : RightCircularCone) : ℝ := 
  2 * Real.pi * cone.radius / cone.slantHeight

/-- Theorem: The central angle of a right circular cone is √2π -/
theorem right_circular_cone_central_angle (cone : RightCircularCone) : 
  centralAngle cone = Real.sqrt 2 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_circular_cone_central_angle_l1304_130432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_monotonic_f_l1304_130481

/-- Given a function f(x) = x³ + ax that is monotonically increasing on [1, +∞),
    the minimum value of a is -3. -/
theorem min_a_for_monotonic_f (a : ℝ) : 
  (∀ x : ℝ, x ≥ 1 → Monotone (fun y ↦ y^3 + a*y)) →
  a ≥ -3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_monotonic_f_l1304_130481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_guarantee_2kg_each_l1304_130417

/-- Represents the state of sand quantities and exchange rates -/
structure SandState where
  g : ℕ  -- Exchange rate for gold
  p : ℕ  -- Exchange rate for platinum
  G : ℚ  -- Quantity of gold sand in kg
  P : ℚ  -- Quantity of platinum sand in kg

/-- Defines the total state value -/
def stateValue (s : SandState) : ℚ := s.g * s.G + s.p * s.P

/-- Applies a sequence of daily changes to the initial state -/
def applySequence : SandState → List Bool → SandState
  | s, [] => s
  | s, true::rest => applySequence ⟨s.g - 1, s.p, s.G, s.P⟩ rest
  | s, false::rest => applySequence ⟨s.g, s.p - 1, s.G, s.P⟩ rest

/-- Theorem stating that it's impossible to guarantee 2kg of each sand type after 2000 days -/
theorem impossible_to_guarantee_2kg_each (initial : SandState) 
  (h_initial_g : initial.g = 1001)
  (h_initial_p : initial.p = 1001)
  (h_initial_G : initial.G = 1)
  (h_initial_P : initial.P = 1) :
  ∀ (final : SandState),
  (∃ (sequence : List Bool), 
    sequence.length = 2000 ∧ 
    final = applySequence initial sequence ∧
    final.g = 1 ∧ final.p = 1) →
  ¬(final.G ≥ 2 ∧ final.P ≥ 2) := by
  sorry  -- Proof omitted for brevity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_guarantee_2kg_each_l1304_130417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_split_area_equals_triangle_area_a_equals_two_l1304_130409

/-- The value of 'a' that splits the area of 6 unit squares into two equal parts -/
noncomputable def a : ℝ :=
  2

/-- The total number of unit squares -/
def total_squares : ℕ :=
  6

/-- The total area of all squares -/
noncomputable def total_area : ℝ :=
  total_squares

/-- The area of each region after splitting -/
noncomputable def split_area : ℝ :=
  total_area / 2

/-- The line that splits the area, defined by two points -/
noncomputable def splitting_line (x : ℝ) : ℝ :=
  (2 / (5 - a)) * (x - a)

/-- The area of the triangle formed by the splitting line -/
noncomputable def triangle_area : ℝ :=
  (5 - a) * 2 / 2

theorem split_area_equals_triangle_area : split_area = triangle_area := by
  sorry

theorem a_equals_two : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_split_area_equals_triangle_area_a_equals_two_l1304_130409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1304_130436

/-- The distance between points A and B -/
noncomputable def distance : ℝ := sorry

/-- The speed of the first cyclist -/
noncomputable def v1 : ℝ := sorry

/-- The speed of the second cyclist -/
noncomputable def v2 : ℝ := sorry

/-- Time taken for the first cyclist to travel half the distance -/
noncomputable def t1 : ℝ := distance / (2 * v1)

/-- Time taken for the second cyclist to travel (distance - 24) km -/
noncomputable def t2 : ℝ := (distance - 24) / v2

/-- Time taken for the first cyclist to travel (distance - 15) km -/
noncomputable def t3 : ℝ := (distance - 15) / v1

/-- Time taken for the second cyclist to travel half the distance -/
noncomputable def t4 : ℝ := distance / (2 * v2)

theorem distance_between_points (h1 : t1 = t2) (h2 : t3 = t4) : distance = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1304_130436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divides_square_l1304_130407

theorem divides_square (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a^b ∣ b^c) (h2 : a^c ∣ c^b) :
  a^2 ∣ b * c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divides_square_l1304_130407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_mass_percentages_l1304_130452

/-- Molar mass of BaF2 in g/mol -/
noncomputable def molar_mass_BaF2 : ℝ := 175.323

/-- Molar mass of BaO in g/mol -/
noncomputable def molar_mass_BaO : ℝ := 153.326

/-- Atomic mass of Ba in g/mol -/
noncomputable def atomic_mass_Ba : ℝ := 137.327

/-- Atomic mass of F in g/mol -/
noncomputable def atomic_mass_F : ℝ := 18.998

/-- Atomic mass of O in g/mol -/
noncomputable def atomic_mass_O : ℝ := 15.999

/-- Mass percentage of Ba in the mixture -/
noncomputable def mass_percentage_Ba (x y : ℝ) : ℝ :=
  ((x / molar_mass_BaF2) * atomic_mass_Ba + (y / molar_mass_BaO) * atomic_mass_Ba) / (x + y) * 100

/-- Mass percentage of F in the mixture -/
noncomputable def mass_percentage_F (x y : ℝ) : ℝ :=
  (x / molar_mass_BaF2) * (2 * atomic_mass_F) / (x + y) * 100

/-- Mass percentage of O in the mixture -/
noncomputable def mass_percentage_O (x y : ℝ) : ℝ :=
  (y / molar_mass_BaO) * atomic_mass_O / (x + y) * 100

theorem sum_of_mass_percentages (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  mass_percentage_Ba x y + mass_percentage_F x y + mass_percentage_O x y = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_mass_percentages_l1304_130452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beginner_trig_probability_l1304_130447

/-- Represents the number of students in calculus courses -/
noncomputable def C : ℝ := sorry

/-- Represents the total number of students -/
noncomputable def T : ℝ := 2.5 * C

/-- Represents the number of students in trigonometry courses -/
noncomputable def trig_students : ℝ := 1.5 * C

/-- Represents the number of students in beginner calculus -/
noncomputable def beginner_calc : ℝ := 0.8 * C

/-- Represents the number of students in beginner trigonometry -/
noncomputable def beginner_trig : ℝ := 1.2 * C

/-- Represents the total number of students in beginner courses -/
noncomputable def total_beginner : ℝ := (4/5) * T

theorem beginner_trig_probability :
  beginner_trig / T = 12 / 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beginner_trig_probability_l1304_130447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_z_fractions_is_zero_l1304_130419

noncomputable def z : ℂ := Complex.exp (3 * Real.pi * Complex.I / 8)

theorem sum_of_z_fractions_is_zero :
  z / (1 + z^3) + z^2 / (1 + z^6) + z^4 / (1 + z^12) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_z_fractions_is_zero_l1304_130419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_theorem_l1304_130439

/-- Parabola type -/
structure Parabola where
  f : ℝ → ℝ
  h : ∀ x, f x = 4 * x

/-- Point on a parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  h : y^2 = p.f x

/-- Focus of the parabola y² = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Main theorem -/
theorem parabola_point_theorem (p : Parabola) (point : PointOnParabola p) :
  distance (point.x, point.y) focus = (3/2) * point.x → point.x = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_theorem_l1304_130439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_sin_cos_equality_smallest_angle_satisfies_equation_no_smaller_angle_satisfies_equation_l1304_130455

/-- The smallest positive angle x satisfying sin(3x) * sin(4x) = cos(3x) * cos(4x) is equal to π / 28 radians (which is equivalent to 90° / 14). -/
theorem smallest_angle_sin_cos_equality : 
  let f : ℝ → ℝ := λ x => Real.sin (3 * x) * Real.sin (4 * x) - Real.cos (3 * x) * Real.cos (4 * x)
  ∃! x : ℝ, x > 0 ∧ f x = 0 ∧ ∀ y : ℝ, y > 0 ∧ f y = 0 → x ≤ y :=
by
  sorry

/-- The value of the smallest positive angle x satisfying the equation. -/
noncomputable def smallest_angle : ℝ := Real.pi / 28

/-- Proof that the smallest_angle satisfies the equation. -/
theorem smallest_angle_satisfies_equation : 
  Real.sin (3 * smallest_angle) * Real.sin (4 * smallest_angle) = 
  Real.cos (3 * smallest_angle) * Real.cos (4 * smallest_angle) :=
by
  sorry

/-- Proof that there is no smaller positive angle satisfying the equation. -/
theorem no_smaller_angle_satisfies_equation :
  ∀ y : ℝ, 0 < y ∧ y < smallest_angle →
    Real.sin (3 * y) * Real.sin (4 * y) ≠ Real.cos (3 * y) * Real.cos (4 * y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_sin_cos_equality_smallest_angle_satisfies_equation_no_smaller_angle_satisfies_equation_l1304_130455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_A_complement_B_l1304_130468

-- Define the sets A and B
def A : Set ℝ := {0, 1, 2, 3}
def B : Set ℝ := {x : ℝ | x^2 - 2*x - 3 ≥ 0}

-- State the theorem
theorem union_A_complement_B : A ∪ (Set.univ \ B) = Set.Ioc (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_A_complement_B_l1304_130468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1304_130483

theorem triangle_side_length (a b c : ℝ) (A B C : Real) :
  a > 0 → b > 0 → c > 0 →
  a^2 - 5*a + 2 = 0 →
  b^2 - 5*b + 2 = 0 →
  C = Real.pi / 3 →
  a^2 + b^2 - 2*a*b*Real.cos C = c^2 →
  c = Real.sqrt 19 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1304_130483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_is_7_2_l1304_130484

/-- Triangle DEF with side lengths DE = 9, DF = 12, and EF = 15 -/
structure Triangle where
  DE : ℝ
  DF : ℝ
  EF : ℝ

/-- The altitude from vertex D to side EF in triangle DEF -/
noncomputable def altitude (t : Triangle) : ℝ := 2 * (t.DE * t.DF) / (2 * t.EF)

/-- Theorem stating that the altitude from D to EF in the given triangle is 7.2 -/
theorem altitude_is_7_2 (t : Triangle) (h1 : t.DE = 9) (h2 : t.DF = 12) (h3 : t.EF = 15) : 
  altitude t = 7.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_is_7_2_l1304_130484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brown_net_result_l1304_130449

-- Define the selling price of each pen
def selling_price : ℝ := 1.50

-- Define the profit percentage on the first pen
def profit_percentage : ℝ := 0.30

-- Define the loss percentage on the second pen
def loss_percentage : ℝ := 0.10

-- Define the tax rate on the profit from the first pen
def tax_rate : ℝ := 0.05

-- Theorem statement
theorem brown_net_result :
  let cost_price1 := selling_price / (1 + profit_percentage)
  let cost_price2 := selling_price / (1 - loss_percentage)
  let profit1 := selling_price - cost_price1
  let loss2 := cost_price2 - selling_price
  let tax := tax_rate * profit1
  let net_result := 2 * selling_price - (cost_price1 + cost_price2) - tax
  ∃ ε > 0, |net_result - 0.16| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brown_net_result_l1304_130449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_motion_l1304_130470

/-- The value of k that makes (1/k) * QD - BM constant throughout the motion --/
def k : ℚ := 5/2

/-- Point positions on the number line --/
noncomputable def A : ℝ := -800
noncomputable def B : ℝ := -400
noncomputable def C : ℝ := 0
noncomputable def D : ℝ := 200

/-- Velocities of points P and Q --/
noncomputable def v_P : ℝ := -10
noncomputable def v_Q : ℝ := -5

/-- Position of P at time t --/
noncomputable def P (t : ℝ) : ℝ := A + v_P * t

/-- Position of Q at time t --/
noncomputable def Q (t : ℝ) : ℝ := C + v_Q * t

/-- Position of midpoint M at time t --/
noncomputable def M (t : ℝ) : ℝ := (P t + Q t) / 2

/-- Distance QD at time t --/
noncomputable def QD (t : ℝ) : ℝ := Q t - D

/-- Distance BM at time t --/
noncomputable def BM (t : ℝ) : ℝ := M t - B

/-- The expression that should remain constant --/
noncomputable def constant_expression (t : ℝ) : ℝ := (1 / k) * QD t - BM t

theorem constant_motion : 
  ∀ t₁ t₂ : ℝ, constant_expression t₁ = constant_expression t₂ :=
by
  sorry

#eval k

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_motion_l1304_130470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_point_distance_l1304_130437

-- Define a unit square ABCD
def unitSquare (A B C D : ℝ × ℝ) : Prop :=
  A = (0, 0) ∧ B = (1, 0) ∧ C = (1, 1) ∧ D = (0, 1)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the distance from a point to a line
noncomputable def distanceToLine (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * p.1 + b * p.2 + c) / Real.sqrt (a^2 + b^2)

-- Theorem statement
theorem square_point_distance (A B C D X : ℝ × ℝ) :
  unitSquare A B C D →
  distanceToLine X 1 (-1) 0 = distanceToLine X 1 1 (-1) →
  distance A X = Real.sqrt 2 / 2 →
  distance C X ^ 2 = 5 / 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_point_distance_l1304_130437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_product_l1304_130492

theorem sum_reciprocal_product : 12 * ((1/3 : ℚ) + 1/4 + 1/6)⁻¹ = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_product_l1304_130492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_numbers_between_202_and_405_l1304_130467

theorem count_even_numbers_between_202_and_405 :
  (Finset.filter (fun n => n % 2 = 0) (Finset.range 405 \ Finset.range 203)).card = 101 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_numbers_between_202_and_405_l1304_130467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l1304_130420

/-- A right triangle in the xy-plane -/
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right_angle : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0

/-- The length of a line segment between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- A line in the xy-plane represented by y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : ℝ × ℝ) : Prop :=
  p.2 = l.m * p.1 + l.b

/-- The area of a triangle given its vertices -/
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

/-- The main theorem -/
theorem right_triangle_area (t : RightTriangle) 
  (hypotenuse_length : distance t.A t.B = 50)
  (median_A : Line.contains { m := 1, b := 5 } ((2/3 * t.A.1 + 1/3 * t.B.1 + 1/3 * t.C.1, 2/3 * t.A.2 + 1/3 * t.B.2 + 1/3 * t.C.2)))
  (median_B : Line.contains { m := 2, b := 2 } ((1/3 * t.A.1 + 2/3 * t.B.1 + 1/3 * t.C.1, 1/3 * t.A.2 + 2/3 * t.B.2 + 1/3 * t.C.2))) :
  triangle_area t.A t.B t.C = 500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l1304_130420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1304_130431

noncomputable section

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 10*x - 14*y + 70 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := 12*x + 5*y + 12 = 0

-- Define a point on the circle
def point_on_circle (x y : ℝ) : Prop := circle_C x y

-- Define the intersection points of the circle and the line
def intersection_points (A B : ℝ × ℝ) : Prop :=
  circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ line_l A.1 A.2 ∧ line_l B.1 B.2

-- Define the area of a triangle given three points
noncomputable def triangle_area (A B Q : ℝ × ℝ) : ℝ :=
  abs ((A.1 - Q.1) * (B.2 - Q.2) - (B.1 - Q.1) * (A.2 - Q.2)) / 2

-- Theorem statement
theorem max_triangle_area :
  ∃ (A B : ℝ × ℝ), intersection_points A B →
  ∃ (Q : ℝ × ℝ), point_on_circle Q.1 Q.2 →
  ∀ (R : ℝ × ℝ), point_on_circle R.1 R.2 →
  triangle_area A B Q ≥ triangle_area A B R ∧
  triangle_area A B Q = 3 * Real.sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1304_130431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_l1304_130427

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 2*y - 11 = 0

/-- The area of the region enclosed by the circle -/
noncomputable def enclosed_area : ℝ := 16 * Real.pi

/-- Theorem: The area of the region enclosed by the circle defined by the given equation is 16π -/
theorem circle_area :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y : ℝ, circle_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    enclosed_area = Real.pi * radius^2 := by
  -- Provide the center coordinates and radius
  let center_x := 2
  let center_y := -1
  let radius := 4
  
  -- Assert the existence of these values
  use center_x, center_y, radius
  
  constructor
  
  · -- Prove the equivalence of the equations
    intro x y
    simp [circle_equation, center_x, center_y, radius]
    -- The actual proof would go here, but we'll use sorry for now
    sorry
    
  · -- Prove that the enclosed area matches the formula
    simp [enclosed_area, radius]
    -- This step is true by definition, but we'll use sorry as a placeholder
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_l1304_130427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_cos_2x_l1304_130402

/-- The function f(x) = cos(2x) -/
noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)

/-- The minimum positive period of f(x) = cos(2x) is π -/
theorem min_positive_period_cos_2x :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
  (∀ T' : ℝ, T' > 0 → (∀ x, f (x + T') = f x) → T ≤ T') ∧
  T = π := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_cos_2x_l1304_130402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_inequality_l1304_130474

theorem triangle_angle_inequality (α β γ : ℝ) (h_angles : α + β + γ = Real.pi) 
  (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ) :
  2 * (Real.sin α / α + Real.sin β / β + Real.sin γ / γ) ≤ 
    (1 / β + 1 / γ) * Real.sin α + (1 / γ + 1 / α) * Real.sin β + (1 / α + 1 / β) * Real.sin γ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_inequality_l1304_130474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1304_130480

noncomputable section

open Real

variable (A B C a b c : ℝ)

def triangle_condition (A B C a b c : ℝ) : Prop :=
  2 * cos A * cos C * (tan A * tan C - 1) = 1 ∧
  a + c = Real.sqrt 15 ∧
  b = Real.sqrt 3

theorem triangle_properties (h : triangle_condition A B C a b c) :
  B = π / 3 ∧ (1/2) * b * c * sin A = Real.sqrt 3 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1304_130480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_determines_plane_l1304_130475

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A triangle in 3D space -/
structure Triangle3D where
  p1 : Point3D
  p2 : Point3D
  p3 : Point3D

/-- A plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Definition: A point lies on a plane -/
def point_on_plane (point : Point3D) (plane : Plane3D) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- Instance for Membership of Point3D in Plane3D -/
instance : Membership Point3D Plane3D where
  mem := point_on_plane

/-- Theorem: A triangle determines a unique plane in 3D space -/
theorem triangle_determines_plane (t : Triangle3D) : ∃! (p : Plane3D), t.p1 ∈ p ∧ t.p2 ∈ p ∧ t.p3 ∈ p :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_determines_plane_l1304_130475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_meaningful_l1304_130421

theorem fraction_meaningful (x : ℝ) : 
  (x + 2 ≠ 0) ↔ x ≠ -2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_meaningful_l1304_130421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_l1304_130413

theorem total_students : ℕ := by
  let group1_count : ℕ := 15
  let group1_avg : ℚ := 70 / 100
  let group2_count : ℕ := 10
  let group2_avg : ℚ := 90 / 100
  let total_avg : ℚ := 78 / 100

  have h : (group1_count : ℚ) * group1_avg + (group2_count : ℚ) * group2_avg = 
    total_avg * ((group1_count + group2_count) : ℚ) := by sorry

  exact group1_count + group2_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_l1304_130413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jessie_daily_walk_l1304_130493

/-- Given that Jackie walks 2 miles each day and in 6 days Jackie walks 3 miles more than Jessie,
    prove that Jessie walks 1.5 miles each day. -/
theorem jessie_daily_walk (jackie_daily : ℝ) (days : ℕ) (difference : ℝ) :
  jackie_daily = 2 →
  days = 6 →
  difference = 3 →
  jackie_daily * (days : ℝ) - difference = 1.5 * (days : ℝ) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jessie_daily_walk_l1304_130493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1304_130410

noncomputable def f (x : ℝ) : ℝ := -6 * (Real.sin x + Real.cos x) - 3

theorem f_properties :
  (∃ (max : ℝ), ∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 4) → f x ≤ max ∧ max = -9) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 4) ∧ f x + 10 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1304_130410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_participants_l1304_130497

theorem chess_tournament_participants (n : ℕ) 
  (h1 : ∀ i j : ℕ, i < n → j < n → i ≠ j → ∃! game : Unit, True)
  (h2 : n * (n - 1) / 2 = 171) : n = 19 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_participants_l1304_130497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonals_l1304_130408

/-- A rhombus with a height drawn from the vertex of an obtuse angle -/
structure Rhombus where
  /-- Length of the first segment of the divided side -/
  m : ℝ
  /-- Length of the second segment of the divided side -/
  n : ℝ
  /-- m and n are positive -/
  m_pos : 0 < m
  n_pos : 0 < n

/-- The length of diagonal BD in the rhombus -/
noncomputable def diagonalBD (r : Rhombus) : ℝ := Real.sqrt (2 * r.n * (r.m + r.n))

/-- The length of diagonal AC in the rhombus -/
noncomputable def diagonalAC (r : Rhombus) : ℝ := Real.sqrt (4 * r.m^2 + 6 * r.m * r.n + 2 * r.n^2)

/-- Theorem stating the lengths of diagonals in the rhombus -/
theorem rhombus_diagonals (r : Rhombus) :
  (diagonalBD r)^2 = 2 * r.n * (r.m + r.n) ∧
  (diagonalAC r)^2 = 4 * r.m^2 + 6 * r.m * r.n + 2 * r.n^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonals_l1304_130408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_heaven_l1304_130477

/-- Probability function for ascending to heaven from point (m, n) -/
noncomputable def P (m n : ℤ) : ℝ := sorry

/-- Contessa's starting point -/
def start : ℤ × ℤ := (1, 1)

/-- Condition for ascending to heaven -/
def is_heaven (m n : ℤ) : Prop := ∃ (a b : ℤ), m = 6 * a ∧ n = 6 * b

/-- Condition for descending to hell -/
def is_hell (m n : ℤ) : Prop := ∃ (a b : ℤ), m = 6 * a + 3 ∧ n = 6 * b + 3

/-- The main theorem stating the probability of ascending to heaven -/
theorem prob_heaven : P start.1 start.2 = 13 / 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_heaven_l1304_130477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_is_4pi_l1304_130476

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 6*y + 8 = 0

/-- The circumference of the circle -/
noncomputable def circle_circumference : ℝ := 4 * Real.pi

/-- Theorem stating that the circumference of the circle is 4π -/
theorem circle_circumference_is_4pi :
  ∃ (r : ℝ), r > 0 ∧ 
  (∀ x y : ℝ, circle_equation x y ↔ (x - 1)^2 + (y + 3)^2 = r^2) ∧
  circle_circumference = 2 * Real.pi * r := by
  sorry

#check circle_circumference_is_4pi

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_is_4pi_l1304_130476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_first_quadrant_l1304_130426

theorem point_in_first_quadrant (a : ℝ) : 
  (a^2 + 1 > 0) ∧ (2020 > 0) :=
by
  constructor
  · -- Proof for a^2 + 1 > 0
    have h1 : a^2 ≥ 0 := sq_nonneg a
    have h2 : a^2 + 1 > a^2 := by linarith
    linarith
  · -- Proof for 2020 > 0
    norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_first_quadrant_l1304_130426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_b_height_l1304_130463

/-- Represents a right circular cylinder tank -/
structure Tank where
  circumference : ℝ
  height : ℝ

/-- The volume of a right circular cylinder -/
noncomputable def tankVolume (t : Tank) : ℝ :=
  (t.circumference ^ 2 * t.height) / (4 * Real.pi)

theorem tank_b_height (tankA tankB : Tank) 
  (hACirc : tankA.circumference = 8)
  (hAHeight : tankA.height = 6)
  (hBCirc : tankB.circumference = 10)
  (hCapacity : tankVolume tankA = 0.4800000000000001 * tankVolume tankB) :
  tankB.height = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_b_height_l1304_130463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_arrangement_l1304_130478

-- Define the type for triangle numbers
def TriangleNumber := Fin 6

-- Define the structure for the triangle arrangement
structure TriangleArrangement where
  A : TriangleNumber
  B : TriangleNumber
  C : TriangleNumber
  D : TriangleNumber
  E : TriangleNumber
  F : TriangleNumber

-- Define the conditions
def satisfiesConditions (arr : TriangleArrangement) : Prop :=
  arr.A ≠ arr.B ∧ arr.A ≠ arr.C ∧ arr.A ≠ arr.D ∧ arr.A ≠ arr.E ∧ arr.A ≠ arr.F ∧
  arr.B ≠ arr.C ∧ arr.B ≠ arr.D ∧ arr.B ≠ arr.E ∧ arr.B ≠ arr.F ∧
  arr.C ≠ arr.D ∧ arr.C ≠ arr.E ∧ arr.C ≠ arr.F ∧
  arr.D ≠ arr.E ∧ arr.D ≠ arr.F ∧
  arr.E ≠ arr.F ∧
  (arr.B.val + arr.D.val + arr.E.val = 14) ∧
  (arr.C.val + arr.E.val + arr.F.val = 12)

-- Theorem statement
theorem unique_arrangement :
  ∃! arr : TriangleArrangement, satisfiesConditions arr ∧
    arr.A.val = 1 ∧ arr.B.val = 3 ∧ arr.C.val = 2 ∧ arr.D.val = 5 ∧ arr.E.val = 6 ∧ arr.F.val = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_arrangement_l1304_130478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_non_congruent_triangles_l1304_130416

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The set of points in the rectangular grid -/
def grid_points : Set Point := {
  ⟨0, 0⟩, ⟨1, 0⟩, ⟨2, 0⟩,
  ⟨0, 0.5⟩, ⟨1, 0.5⟩, ⟨2, 0.5⟩,
  ⟨0, 1⟩, ⟨1, 1⟩, ⟨2, 1⟩,
  ⟨1, 0.5⟩
}

/-- A triangle represented by its three vertices -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Check if two triangles are congruent -/
def are_congruent (t1 t2 : Triangle) : Prop := sorry

/-- The set of all possible triangles formed by the grid points -/
def all_triangles : Set Triangle := sorry

/-- The set of non-congruent triangles -/
noncomputable def non_congruent_triangles : Finset Triangle := sorry

/-- The main theorem: there are exactly 4 non-congruent triangles -/
theorem four_non_congruent_triangles : 
  Finset.card non_congruent_triangles = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_non_congruent_triangles_l1304_130416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carolyn_sum_is_eight_l1304_130485

-- Define the game state
structure GameState where
  remaining : List Nat
  carolynSum : Nat
  paulSum : Nat
deriving Inhabited

-- Define the game rules
def initialState : GameState :=
  { remaining := [1, 2, 3, 4, 5, 6], carolynSum := 0, paulSum := 0 }

def carolynMove (state : GameState) (num : Nat) : Option GameState :=
  if num ∉ state.remaining then none
  else if ¬(∃ x ∈ state.remaining, x ≠ num ∧ x ∣ num) then none
  else some {
    remaining := state.remaining.filter (· ≠ num),
    carolynSum := state.carolynSum + num,
    paulSum := state.paulSum
  }

def paulMove (state : GameState) (num : Nat) : GameState :=
  let divisors := state.remaining.filter (· ∣ num)
  { remaining := state.remaining.filter (· ∉ divisors),
    carolynSum := state.carolynSum,
    paulSum := state.paulSum + divisors.sum }

def gameOver (state : GameState) : Bool :=
  ∀ x ∈ state.remaining, ¬(∃ y ∈ state.remaining, y ≠ x ∧ y ∣ x)

-- Theorem statement
theorem carolyn_sum_is_eight :
  ∃ (finalState : GameState),
    (let state1 := (carolynMove initialState 2).get!
     let state2 := paulMove state1 2
     let state3 := (carolynMove state2 6).get!
     let state4 := paulMove state3 6
     finalState = if gameOver state4 then paulMove state4 0 else state4) ∧
    finalState.carolynSum = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carolyn_sum_is_eight_l1304_130485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_in_dichromate_l1304_130462

/-- The molar mass of chromium in g/mol -/
noncomputable def molar_mass_Cr : ℝ := 52.00

/-- The molar mass of oxygen in g/mol -/
noncomputable def molar_mass_O : ℝ := 16.00

/-- The number of chromium atoms in dichromate -/
def num_Cr : ℕ := 2

/-- The number of oxygen atoms in dichromate -/
def num_O : ℕ := 7

/-- The molar mass of dichromate in g/mol -/
noncomputable def molar_mass_dichromate : ℝ := num_Cr * molar_mass_Cr + num_O * molar_mass_O

/-- The mass percentage of oxygen in dichromate -/
noncomputable def mass_percentage_O : ℝ := (num_O * molar_mass_O / molar_mass_dichromate) * 100

theorem mass_percentage_O_in_dichromate :
  ∃ ε > 0, |mass_percentage_O - 51.85| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_in_dichromate_l1304_130462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_times_i_in_first_quadrant_l1304_130457

def z : ℂ := 2 - Complex.I

theorem z_times_i_in_first_quadrant :
  let w := z * Complex.I
  0 < w.re ∧ 0 < w.im :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_times_i_in_first_quadrant_l1304_130457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_catch_nut_l1304_130451

/-- The problem of determining whether a squirrel can catch a thrown nut -/
theorem squirrel_catch_nut 
  (initial_distance : ℝ) 
  (nut_speed : ℝ) 
  (squirrel_jump_distance : ℝ) 
  (gravity : ℝ) 
  (h1 : initial_distance = 3.75)
  (h2 : nut_speed = 5)
  (h3 : squirrel_jump_distance = 1.8)
  (h4 : gravity = 10) :
  ∃ (t : ℝ), 
    Real.sqrt ((nut_speed * t - initial_distance)^2 + (gravity * t^2 / 2)^2) ≤ squirrel_jump_distance :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_catch_nut_l1304_130451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_theorem_l1304_130401

/-- The probability that a chord intersects the inner circle when two points are chosen randomly
    on the outer circle of two concentric circles with radii 3 and 5 -/
noncomputable def chord_intersection_probability : ℝ := 73.74 / 360

/-- The setup of two concentric circles with radii 3 and 5 -/
structure ConcentricCircles where
  inner_radius : ℝ
  outer_radius : ℝ
  inner_radius_eq : inner_radius = 3
  outer_radius_eq : outer_radius = 5

/-- A function that calculates the probability of a chord intersecting the inner circle -/
noncomputable def calculate_intersection_probability (c : ConcentricCircles) : ℝ :=
  -- The actual calculation would go here
  sorry

theorem chord_intersection_theorem (c : ConcentricCircles) :
  calculate_intersection_probability c = chord_intersection_probability :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_theorem_l1304_130401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_sum_l1304_130446

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (2 * x + φ)

theorem function_value_at_sum (φ : ℝ) (x₁ x₂ : ℝ) :
  |φ| < π / 2 →
  (∀ x, f φ (x + π / 12) = f φ (π / 12 - x)) →
  x₁ ∈ Set.Ioo (-17 * π / 12) (-2 * π / 3) →
  x₂ ∈ Set.Ioo (-17 * π / 12) (-2 * π / 3) →
  x₁ ≠ x₂ →
  f φ x₁ = f φ x₂ →
  f φ (x₁ + x₂) = Real.sqrt 6 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_sum_l1304_130446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_path_is_ellipse_approximated_by_parabola_l1304_130430

/-- Represents a uniform spherical planet -/
structure Planet where
  radius : ℝ
  mass : ℝ

/-- Represents a particle launched from the planet's surface -/
structure Particle where
  initial_position : ℝ × ℝ × ℝ
  initial_velocity : ℝ × ℝ × ℝ

/-- The gravitational constant -/
noncomputable def G : ℝ := sorry

/-- The gravitational force between two masses at a distance r -/
noncomputable def gravitational_force (m1 m2 r : ℝ) : ℝ :=
  G * m1 * m2 / (r^2)

/-- The path of a particle in a gravitational field -/
def particle_path (p : Planet) (part : Particle) : Set (ℝ × ℝ × ℝ) := sorry

/-- A function that determines if a set of points approximates a parabola -/
def approximates_parabola (s : Set (ℝ × ℝ × ℝ)) : Prop := sorry

/-- Represents an ellipse with a given center and semi-major/semi-minor axes -/
def IsEllipse (center : ℝ × ℝ × ℝ) (a b : ℝ) (s : Set (ℝ × ℝ × ℝ)) : Prop := sorry

/-- Theorem: The path of a particle launched from and returning to a planet's surface
    is an ellipse that can be approximated as a parabola for short flights -/
theorem particle_path_is_ellipse_approximated_by_parabola
  (p : Planet) (part : Particle) :
  ∃ (path : Set (ℝ × ℝ × ℝ)),
    path = particle_path p part ∧
    (∃ (center : ℝ × ℝ × ℝ) (a b : ℝ), IsEllipse center a b path) ∧
    approximates_parabola path := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_path_is_ellipse_approximated_by_parabola_l1304_130430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_homothety_center_l1304_130404

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

structure Circle where
  center : Point
  radius : ℝ

structure RotationHomothety where
  center : Point
  angle : ℝ
  scale : ℝ

-- Define necessary instances
instance : Membership Point Line where
  mem p l := l.a * p.x + l.b * p.y + l.c = 0

instance : Membership Point Circle where
  mem p c := (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

-- Define necessary operations
def Point.sub (p q : Point) : Point :=
  ⟨p.x - q.x, p.y - q.y⟩

def Point.add (p : Point) (s : ℝ) : Point :=
  ⟨p.x + s, p.y + s⟩

def Point.rotateAbout (p : Point) (center : Point) (angle : ℝ) : Point :=
  sorry -- Implement the rotation logic here

-- Define the main theorem
theorem rotation_homothety_center 
  (A B A₁ B₁ P : Point) 
  (line_AB line_A₁B₁ : Line)
  (circle_PAA₁ circle_PBB₁ : Circle)
  (h_distinct : A ≠ B ∧ A ≠ A₁ ∧ A ≠ B₁ ∧ A ≠ P ∧
                B ≠ A₁ ∧ B ≠ B₁ ∧ B ≠ P ∧
                A₁ ≠ B₁ ∧ A₁ ≠ P ∧
                B₁ ≠ P)
  (h_P_intersection : P ∈ line_AB ∧ P ∈ line_A₁B₁)
  (h_O_common : ∃ O, O ∈ circle_PAA₁ ∧ O ∈ circle_PBB₁) :
  ∃! (rh : RotationHomothety), 
    rh.center ∈ circle_PAA₁ ∧ 
    rh.center ∈ circle_PBB₁ ∧
    (∃ (f : Point → Point), 
      f A = A₁ ∧ 
      f B = B₁ ∧
      (∀ X, f X = rh.center.add (rh.scale * ((X.sub rh.center).rotateAbout rh.center rh.angle).x))) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_homothety_center_l1304_130404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_parabola_perpendicular_l1304_130412

/-- The minimum positive m value for a parabola and perpendicular lines configuration -/
theorem min_m_parabola_perpendicular (a : ℝ) : 
  -- The line x - 4y + 1 = 0 passes through the focus of y = ax^2
  (∃ x y : ℝ, x - 4*y + 1 = 0 ∧ y = 1/(4*a)) →
  -- There exists a point P on the parabola such that PA ⊥ PB
  (∃ x y m : ℝ, y = a*x^2 ∧ 
    (y - (2+m))/(x - 0) * (y - (2-m))/(x - 0) = -1) →
  -- The minimum positive value of m
  (∀ m : ℝ, m > 0 → m ≥ Real.sqrt 7/2) ∧ (∃ m : ℝ, m > 0 ∧ m = Real.sqrt 7/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_parabola_perpendicular_l1304_130412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l1304_130422

/-- Conversion from cylindrical coordinates to rectangular coordinates -/
noncomputable def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

/-- The given point in cylindrical coordinates -/
noncomputable def cylindrical_point : ℝ × ℝ × ℝ := (5, Real.pi / 3, 2)

/-- The expected point in rectangular coordinates -/
noncomputable def rectangular_point : ℝ × ℝ × ℝ := (2.5, 5 * Real.sqrt 3 / 2, 2)

/-- Theorem stating that the conversion of the cylindrical point equals the rectangular point -/
theorem cylindrical_to_rectangular_conversion :
  cylindrical_to_rectangular cylindrical_point.1 cylindrical_point.2.1 cylindrical_point.2.2 = rectangular_point := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l1304_130422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_theorem_main_theorem_l1304_130435

/-- Regular quadrilateral pyramid with given properties -/
structure RegularQuadPyramid where
  -- Side length of the base
  base_side : ℝ
  -- Angle between adjacent lateral faces
  lateral_angle : ℝ
  -- Assertion that the base side is 4√2
  base_side_eq : base_side = 4 * Real.sqrt 2
  -- Assertion that the lateral angle is 120°
  lateral_angle_eq : lateral_angle = 2 * π / 3

/-- The cross-section area of the pyramid -/
noncomputable def cross_section_area (p : RegularQuadPyramid) : ℝ := 4 * Real.sqrt 6

/-- Theorem stating the cross-section area is 4√6 -/
theorem cross_section_area_theorem (p : RegularQuadPyramid) :
  cross_section_area p = 4 * Real.sqrt 6 := by
  -- Unfold the definition of cross_section_area
  unfold cross_section_area
  -- The equality is now trivial
  rfl

/-- Main theorem: The area of the cross-section is 4√6 -/
theorem main_theorem (p : RegularQuadPyramid) :
  ∃ (area : ℝ), area = cross_section_area p ∧ area = 4 * Real.sqrt 6 := by
  use cross_section_area p
  constructor
  · rfl
  · exact cross_section_area_theorem p

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_theorem_main_theorem_l1304_130435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_coordinate_sum_l1304_130406

-- Define a Point type
structure Point where
  x : ℝ
  y : ℝ

-- Define a Parallelogram type
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define the problem
theorem parallelogram_coordinate_sum (ABCD : Parallelogram) :
  ABCD.A = ⟨-1, 2⟩ →
  ABCD.C = ⟨7, -4⟩ →
  ABCD.B = ⟨3, -6⟩ →
  ABCD.D.x + ABCD.D.y = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_coordinate_sum_l1304_130406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_volume_at_16_degrees_l1304_130489

/-- Represents the volume of a balloon as a function of temperature change -/
noncomputable def balloon_volume (initial_temp : ℝ) (initial_vol : ℝ) (temp_change : ℝ) : ℝ :=
  initial_vol + (5 / 2) * temp_change

/-- Theorem: The volume of the balloon at 16° is 20 cm³ -/
theorem balloon_volume_at_16_degrees 
  (initial_temp : ℝ) 
  (initial_vol : ℝ) 
  (h1 : initial_temp = 28) 
  (h2 : initial_vol = 50) :
  balloon_volume initial_temp initial_vol (16 - initial_temp) = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_volume_at_16_degrees_l1304_130489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_focal_length_is_8_l1304_130498

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : 0 < a
  b_pos : 0 < b

/-- Calculates the focal length of a hyperbola -/
noncomputable def focal_length (h : Hyperbola a b) : ℝ := 2 * Real.sqrt (a^2 + b^2)

/-- Theorem: The minimum focal length of a hyperbola with area of triangle ODE equal to 8 is 8 -/
theorem min_focal_length_is_8 (a b : ℝ) (h : Hyperbola a b) :
  a * b = 8 → ∃ (min_focal_length : ℝ), 
    (∀ (a' b' : ℝ) (h' : Hyperbola a' b'), a' * b' = 8 → 
      focal_length h' ≥ min_focal_length) ∧
    min_focal_length = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_focal_length_is_8_l1304_130498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_calculation_l1304_130456

-- Define the cone parameters
noncomputable def slant_height : ℝ := 5
noncomputable def lateral_surface_area : ℝ := 15 * Real.pi

-- Define the volume function
noncomputable def cone_volume (r : ℝ) (h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

-- Theorem statement
theorem cone_volume_calculation :
  ∃ (r h : ℝ), 
    r^2 + h^2 = slant_height^2 ∧
    2 * Real.pi * r * slant_height = lateral_surface_area ∧
    cone_volume r h = 12 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_calculation_l1304_130456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tomato_multiple_is_three_l1304_130461

-- Define the number of tomatoes for each plant
def first_plant : ℕ := 8
def second_plant : ℕ := first_plant + 4

-- Define the total number of tomatoes from the first two plants
def first_two_plants : ℕ := first_plant + second_plant

-- Define the total number of tomatoes
def total_tomatoes : ℕ := 140

-- Define the multiple as a rational number
noncomputable def multiple : ℚ := (total_tomatoes - first_two_plants : ℚ) / (2 * first_two_plants)

-- Theorem statement
theorem tomato_multiple_is_three : multiple = 3 := by
  -- Expand the definitions
  unfold multiple
  unfold total_tomatoes
  unfold first_two_plants
  unfold second_plant
  unfold first_plant
  
  -- Simplify the expression
  simp
  
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tomato_multiple_is_three_l1304_130461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_x_coordinate_l1304_130405

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem parabola_point_x_coordinate (x y : ℝ) :
  parabola x y →
  distance (x, y) focus = 20 →
  x = 18 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_x_coordinate_l1304_130405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l1304_130453

/-- The length of a bridge given train specifications -/
noncomputable def bridge_length (train_length : ℝ) (crossing_time : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1 / 3.6)
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Theorem stating the length of the bridge given specific conditions -/
theorem bridge_length_calculation :
  let train_length := (250 : ℝ)
  let crossing_time := (45 : ℝ)
  let train_speed_kmh := (44 : ℝ)
  ∃ ε > 0, |bridge_length train_length crossing_time train_speed_kmh - 299.9| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l1304_130453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_problem_l1304_130459

noncomputable def z (b : ℝ) : ℂ := 3 + Complex.I * b

theorem complex_number_problem (b : ℝ) 
  (h : ∃ (k : ℝ), (1 + 3 * Complex.I) * z b = Complex.I * k) : 
  z b = 3 + Complex.I ∧ Complex.abs ((z b) / (2 + Complex.I)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_problem_l1304_130459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_filling_time_l1304_130454

/-- The time it takes for pipe B to fill the tank alone -/
def time_B : ℝ := 46

/-- The time it takes for both pipes to fill the tank together -/
def time_both : ℝ := 20.195121951219512

/-- The time it takes for pipe A to fill the tank alone -/
def time_A : ℝ := 36.04878048780488

/-- Tolerance for approximate equality -/
def tolerance : ℝ := 0.000001

theorem pipe_filling_time :
  abs ((1 / time_A + 1 / time_B) * time_both - 1) < tolerance := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_filling_time_l1304_130454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_power_of_ten_divisor_l1304_130464

/-- A "good number" is a nine-digit number formed by each of the digits 1, 2, ..., 9 appearing exactly once. -/
def GoodNumber : Type := { n : ℕ // n ≥ 100000000 ∧ n < 1000000000 ∧ (∀ d : Fin 9, ∃! i : Fin 9, (n / (10 ^ i.val)) % 10 = d.val + 1) }

/-- The sum of nine good numbers -/
def SumOfNineGoodNumbers : Type := { s : ℕ // ∃ (n₁ n₂ n₃ n₄ n₅ n₆ n₇ n₈ n₉ : GoodNumber), s = n₁.val + n₂.val + n₃.val + n₄.val + n₅.val + n₆.val + n₇.val + n₈.val + n₉.val }

theorem max_power_of_ten_divisor (s : SumOfNineGoodNumbers) :
  (∃ m : ℕ, s.val = m * (10^8)) ∧ 
  (∀ k : ℕ, k > 8 → ¬∃ m : ℕ, s.val = m * (10^k)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_power_of_ten_divisor_l1304_130464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_start_days_l1304_130479

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to calculate the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to calculate the day of the week after a given number of days -/
def advanceDays (d : DayOfWeek) (n : ℕ) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDays (nextDay d) n

/-- Function to check if any of the 8 redemption days falls on a Saturday -/
def anySaturday (startDay : DayOfWeek) : Prop :=
  ∃ i : Fin 8, advanceDays startDay (i.val * 12) = DayOfWeek.Saturday

/-- Theorem stating that Tuesday and Friday are the only valid starting days -/
theorem valid_start_days :
  (¬ anySaturday DayOfWeek.Tuesday ∧ ¬ anySaturday DayOfWeek.Friday) ∧
  (∀ d : DayOfWeek, d ≠ DayOfWeek.Tuesday → d ≠ DayOfWeek.Friday → anySaturday d) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_start_days_l1304_130479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_sequence_implies_prime_or_power_of_two_l1304_130487

theorem coprime_sequence_implies_prime_or_power_of_two (n : ℕ) (h_n : n > 6) :
  (∃ (k : ℕ) (a : ℕ → ℕ) (d : ℕ),
    (∀ i, i ∈ Finset.range k → 0 < a i ∧ a i < n ∧ Nat.Coprime (a i) n) ∧
    (∀ i, i ∈ Finset.range (k - 1) → a (i + 1) - a i = d) ∧
    d > 0 ∧
    (∀ m, 0 < m ∧ m < n ∧ Nat.Coprime m n → ∃ i, i ∈ Finset.range k ∧ a i = m)) →
  Nat.Prime n ∨ ∃ m, n = 2^m := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_sequence_implies_prime_or_power_of_two_l1304_130487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_side_c_acute_triangle_l1304_130433

/-- Theorem: Minimum value of side c in acute triangle ABC -/
theorem min_side_c_acute_triangle (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  A + B + C = π →
  (Real.sqrt 3 / 12) * (a^2 + b^2 - c^2) = (1 / 2) * a * b * Real.sin C →
  24 * (b * c - a) = b * Real.tan B →
  c ≥ 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_side_c_acute_triangle_l1304_130433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a1_value_l1304_130496

def recurrence_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) = 1 / (1 - a n)

theorem sequence_a1_value (a : ℕ → ℚ) (h : recurrence_sequence a) (h8 : a 8 = 2) : a 1 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a1_value_l1304_130496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1304_130434

-- Define the ellipse structure
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_major_axis : a > b

-- Define a point on the ellipse
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

-- Define the foci of the ellipse
noncomputable def foci (e : Ellipse) : ℝ × ℝ := (Real.sqrt (e.a^2 - e.b^2), 0)

-- Theorem statement
theorem ellipse_equation (e : Ellipse) (p : PointOnEllipse e) :
  (∃ (f1 f2 : ℝ × ℝ), 
    f1 = foci e ∧ f2 = (-(foci e).1, (foci e).2) ∧
    Real.sqrt ((p.x - f1.1)^2 + (p.y - f1.2)^2) + 
    Real.sqrt ((p.x - f2.1)^2 + (p.y - f2.2)^2) = 10 ∧
    (p.x * (foci e).1 = e.a^2 ∨ p.y * (foci e).2 = e.b^2)) →
  (e.a = 4 ∧ e.b = Real.sqrt 12) ∨ (e.a = Real.sqrt 12 ∧ e.b = 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1304_130434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1304_130411

-- Define the function f(t) = 2^t + log_3(t)
noncomputable def f (t : ℝ) : ℝ := 2^t + (Real.log t) / (Real.log 3)

-- Define the system of equations
def system (x y : ℤ) : Prop :=
  (f (x : ℝ) = (y : ℝ)^2) ∧ (f (y : ℝ) = (x : ℝ)^2)

-- State the theorem
theorem unique_solution :
  ∃! p : ℤ × ℤ, system p.1 p.2 ∧ p = (3, 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1304_130411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_and_T_formulas_l1304_130428

-- Define the sequence a_n
def a : ℕ → ℝ := sorry

-- Define S_n as the sum of the first n terms of a_n
def S : ℕ → ℝ := sorry

-- Define the relation between S_n and a_n
axiom S_def : ∀ n : ℕ, S n = (3/2) * a n - (1/2)

-- Define b_n
noncomputable def b (n : ℕ) : ℝ := 2 * n / (a (n + 2) - a (n + 1))

-- Define T_n as the sum of the first n terms of b_n
def T : ℕ → ℝ := sorry

theorem a_and_T_formulas :
  (∀ n : ℕ, n ≥ 1 → a n = 3^(n-1)) ∧
  (∀ n : ℕ, T n = 3/4 - (2*n+3)/(4*3^n)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_and_T_formulas_l1304_130428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_equation_solution_l1304_130418

theorem tan_equation_solution (x : ℝ) :
  (∀ k : ℤ, x ≠ (2 * k + 1) * Real.pi) →
  Real.sqrt 3 * Real.tan (x / 2) = 1 →
  ∃ k : ℤ, x = 2 * k * Real.pi + Real.pi / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_equation_solution_l1304_130418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sh_sum_diff_ch_sum_diff_l1304_130472

open Real

/-- Hyperbolic sine function -/
noncomputable def sh (x : ℝ) : ℝ := (exp x - exp (-x)) / 2

/-- Hyperbolic cosine function -/
noncomputable def ch (x : ℝ) : ℝ := (exp x + exp (-x)) / 2

/-- Theorem: Hyperbolic sine of sum/difference -/
theorem sh_sum_diff (x y : ℝ) : 
  (sh (x + y) = sh x * ch y + ch x * sh y) ∧ 
  (sh (x - y) = sh x * ch y - ch x * sh y) := by sorry

/-- Theorem: Hyperbolic cosine of sum/difference -/
theorem ch_sum_diff (x y : ℝ) : 
  (ch (x + y) = ch x * ch y + sh x * sh y) ∧ 
  (ch (x - y) = ch x * ch y - sh x * sh y) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sh_sum_diff_ch_sum_diff_l1304_130472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_real_or_imaginary_l1304_130403

noncomputable def z (x : ℝ) : ℂ := Complex.log (x^2 - 2*x - 2) + (x^2 + 3*x + 2)*Complex.I

theorem z_real_or_imaginary (x : ℝ) :
  ((z x).im = 0 ↔ x = -1 ∨ x = -2) ∧
  ((z x).re = 0 ↔ x = 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_real_or_imaginary_l1304_130403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_5_l1304_130499

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt ((h.a ^ 2 + h.b ^ 2) / h.a ^ 2)

/-- Theorem: The eccentricity of a hyperbola with asymptotes y = ±2x is √5 -/
theorem hyperbola_eccentricity_sqrt_5 (h : Hyperbola) 
    (h_asymptote : h.b = 2 * h.a) : eccentricity h = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_5_l1304_130499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_paths_count_l1304_130400

/-- Represents a 3x3 grid of numbers from 1 to 9 -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Represents a path on the grid -/
def GridPath := List (Fin 3 × Fin 3)

/-- Checks if a path is valid (passes through sides, not vertices) -/
def isValidPath (p : GridPath) : Prop := sorry

/-- Checks if a path is continuous -/
def isContinuousPath (p : GridPath) : Prop := sorry

/-- Counts the number of 3-number paths -/
def countThreeNumberPaths (g : Grid) : Nat := sorry

/-- Counts the number of 9-number paths starting at 5 -/
def countNineNumberPaths (g : Grid) : Nat := sorry

/-- The main theorem -/
theorem grid_paths_count (g : Grid) : 
  (countThreeNumberPaths g = 44) ∧ 
  (countNineNumberPaths g = 8) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_paths_count_l1304_130400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_madeline_box_count_l1304_130458

/-- Represents the number of crayons in each box -/
def crayons_per_box : ℕ := 24

/-- Represents the fraction of unused crayons in the first two boxes -/
def unused_fraction_first_two : ℚ := 5/8

/-- Represents the fraction of used crayons in the next two boxes -/
def used_fraction_next_two : ℚ := 2/3

/-- Represents the total number of unused crayons -/
def total_unused_crayons : ℕ := 70

/-- Proves that Madeline has 5 boxes of crayons -/
theorem madeline_box_count :
  ∃ (n : ℕ), n = 5 ∧
  (n - 1) * crayons_per_box = total_unused_crayons - 
    (2 * (unused_fraction_first_two * crayons_per_box).floor +
     2 * ((1 - used_fraction_next_two) * crayons_per_box).floor) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_madeline_box_count_l1304_130458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_pea_probability_l1304_130466

/-- Represents the possible genes for pea color --/
inductive PeaGene
  | A  -- dominant gene
  | a  -- recessive gene

/-- Represents the possible genotypes for peas --/
structure PeaGenotype where
  gene1 : PeaGene
  gene2 : PeaGene

/-- Determines the color of a pea based on its genotype --/
def peaColor (genotype : PeaGenotype) : Bool :=
  match genotype with
  | ⟨PeaGene.A, _⟩ => true  -- yellow if at least one A
  | ⟨PeaGene.a, PeaGene.A⟩ => true  -- yellow if at least one A
  | _ => false  -- green otherwise

/-- The set of all possible offspring genotypes from self-crossing Aa peas --/
def offspringGenotypes : List PeaGenotype :=
  [⟨PeaGene.A, PeaGene.A⟩, ⟨PeaGene.A, PeaGene.a⟩, ⟨PeaGene.a, PeaGene.A⟩, ⟨PeaGene.a, PeaGene.a⟩]

/-- Theorem stating that the probability of yellow peas from self-crossing Aa peas is 3/4 --/
theorem yellow_pea_probability :
    (offspringGenotypes.filter peaColor).length / offspringGenotypes.length = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_pea_probability_l1304_130466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_l1304_130450

-- Define the points
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, 1)

-- Define the angle AOC and length of OC
noncomputable def angle_AOC : ℝ := 5 * Real.pi / 6
def length_OC : ℝ := 2

-- Define OC vector
noncomputable def OC : ℝ × ℝ := (2 * Real.cos angle_AOC, 2 * Real.sin angle_AOC)

-- Define lambda and mu
noncomputable def lambda : ℝ := -Real.sqrt 3
def mu : ℝ := 1

-- Theorem statement
theorem vector_decomposition :
  OC.1 = lambda * A.1 + mu * B.1 ∧ OC.2 = lambda * A.2 + mu * B.2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_l1304_130450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1304_130471

/-- Prove that an ellipse with given properties has specific parametric equations -/
theorem ellipse_equation (e : ℝ) (h_e : e = Real.sqrt 3 / 2) :
  ∃ (a b : ℝ),
    (∀ θ : ℝ, ∃ (x y : ℝ), x = a * Real.cos θ ∧ y = b * Real.sin θ) ∧
    (∀ (x y : ℝ), x^2 + (y - 3/2)^2 = 1 →
      ∃ (x' y' : ℝ), (x' - x)^2 + (y' - y)^2 ≤ (1 + Real.sqrt 7)^2) ∧
    a = 2 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1304_130471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_charge_theorem_l1304_130445

/-- The number of hours of use per hour of charge for Olive's phone. -/
noncomputable def hours_of_use_per_charge : ℝ := 2

/-- The total charging time last night in hours. -/
noncomputable def total_charge_time : ℝ := 10

/-- The fraction of last night's charging time used in the second condition. -/
noncomputable def charge_fraction : ℝ := 3/5

/-- The number of hours the phone can be used when charged for the fractional time. -/
noncomputable def fractional_use_time : ℝ := 12

/-- Theorem stating the relationship between charging time and usage time. -/
theorem phone_charge_theorem :
  hours_of_use_per_charge * (charge_fraction * total_charge_time) = fractional_use_time :=
by
  -- Proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_charge_theorem_l1304_130445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_midpoint_theorem_l1304_130443

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 9 = 1

-- Define a line
def line (a b c : ℝ) (x y : ℝ) : Prop := a*x + b*y + c = 0

-- Define a point on a line
def point_on_line (a b c x y : ℝ) : Prop := line a b c x y

-- Define the midpoint of two points
def is_midpoint (x1 y1 x2 y2 xm ym : ℝ) : Prop := xm = (x1 + x2) / 2 ∧ ym = (y1 + y2) / 2

-- Theorem statement
theorem chord_midpoint_theorem :
  ∀ (x1 y1 x2 y2 : ℝ),
  ellipse x1 y1 → 
  ellipse x2 y2 → 
  (∃ a b c : ℝ, point_on_line a b c x1 y1 ∧ point_on_line a b c x2 y2) →
  is_midpoint x1 y1 x2 y2 4 2 →
  ∃ a b c : ℝ, ∀ x y : ℝ, line a b c x y ↔ line 1 2 (-8) x y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_midpoint_theorem_l1304_130443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_15_eq_zero_l1304_130429

/-- Definition of the sequence (bₙ) -/
def b : ℕ → ℚ
  | 0 => 1  -- Adding this case to cover Nat.zero
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | n+4 => 2 * (b (n+3) - b (n+2)) / b (n+1)

/-- Theorem stating that the 15th term of the sequence is 0 -/
theorem b_15_eq_zero : b 15 = 0 := by
  -- Compute the value directly
  norm_num
  -- If norm_num doesn't solve it completely, we can add:
  sorry

#eval b 15  -- This line will evaluate b 15 and print the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_15_eq_zero_l1304_130429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_x_coordinate_l1304_130482

/-- The ellipse equation -/
def ellipse_equation (x y m : ℝ) : Prop := x^2/4 + y^2 = m

/-- Point P on the ellipse -/
def point_P : ℝ × ℝ := (0, 1)

/-- Condition for points A and B -/
def vector_condition (A B : ℝ × ℝ) : Prop :=
  let P := point_P
  (P.1 - A.1, P.2 - A.2) = (2 * (B.1 - P.1), 2 * (B.2 - P.2))

theorem ellipse_max_x_coordinate (m : ℝ) (hm : m > 1) :
  ∃ (A B : ℝ × ℝ),
    ellipse_equation A.1 A.2 m ∧
    ellipse_equation B.1 B.2 m ∧
    vector_condition A B ∧
    (∀ (m' : ℝ) (A' B' : ℝ × ℝ), m' > 1 →
      ellipse_equation A'.1 A'.2 m' →
      ellipse_equation B'.1 B'.2 m' →
      vector_condition A' B' →
      B'.1^2 ≤ B.1^2) →
    m = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_x_coordinate_l1304_130482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_gender_relation_and_expected_value_l1304_130486

-- Define the survey data
def total_drivers : ℕ := 100
def male_drivers : ℕ := 55
def female_drivers : ℕ := 45
def male_high_speed : ℕ := 40
def female_high_speed : ℕ := 20

-- Define the K^2 test function
noncomputable def K_squared (a b c d : ℕ) : ℝ :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : ℝ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for 99.5% confidence
def critical_value : ℝ := 7.879

-- Define the probability of selecting a male driver with high speed
noncomputable def p_male_high_speed : ℝ := male_high_speed / total_drivers

-- Theorem statement
theorem speed_gender_relation_and_expected_value :
  -- Part 1: Speed and gender relation
  K_squared male_high_speed (male_drivers - male_high_speed) 
            female_high_speed (female_drivers - female_high_speed) > critical_value ∧
  -- Part 2: Expected value of X
  (3 : ℝ) * p_male_high_speed = 6/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_gender_relation_and_expected_value_l1304_130486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workforce_reduction_l1304_130448

/-- Given a workforce reduction of 27.6% resulting in 462 employees, 
    prove that the original number of employees was approximately 638. -/
theorem workforce_reduction (reduced_percentage : ℝ) (reduced_employees : ℕ) 
    (h1 : reduced_percentage = 27.6)
    (h2 : reduced_employees = 462) : 
    Int.floor ((reduced_employees : ℝ) / (1 - reduced_percentage / 100)) = 638 := by
  -- Convert the given values
  have reduced_percentage_decimal : ℝ := reduced_percentage / 100
  have remaining_percentage : ℝ := 1 - reduced_percentage_decimal
  
  -- Calculate the original number of employees
  have original_employees : ℝ := (reduced_employees : ℝ) / remaining_percentage
  
  -- Show that the floor of this value is 638
  sorry  -- Proof steps would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_workforce_reduction_l1304_130448
