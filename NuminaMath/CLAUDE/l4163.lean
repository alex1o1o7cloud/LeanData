import Mathlib

namespace NUMINAMATH_CALUDE_debate_panel_probability_l4163_416304

def total_members : ℕ := 20
def boys : ℕ := 8
def girls : ℕ := 12
def panel_size : ℕ := 4

theorem debate_panel_probability :
  let total_combinations := Nat.choose total_members panel_size
  let all_boys := Nat.choose boys panel_size
  let all_girls := Nat.choose girls panel_size
  let prob_complement := (all_boys + all_girls : ℚ) / total_combinations
  1 - prob_complement = 856 / 969 := by sorry

end NUMINAMATH_CALUDE_debate_panel_probability_l4163_416304


namespace NUMINAMATH_CALUDE_right_triangles_problem_l4163_416360

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) := True

-- Define a right triangle
def RightTriangle (A B C : ℝ × ℝ) := Triangle A B C ∧ True

-- Define the length of a line segment
def Length (A B : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem right_triangles_problem 
  (A B C D : ℝ × ℝ) 
  (a : ℝ) 
  (h1 : RightTriangle A B C)
  (h2 : RightTriangle A B D)
  (h3 : Length B C = 3)
  (h4 : Length A C = a)
  (h5 : Length A D = 1) :
  Length B D = Real.sqrt (a^2 + 8) := by
  sorry


end NUMINAMATH_CALUDE_right_triangles_problem_l4163_416360


namespace NUMINAMATH_CALUDE_f_of_g_of_3_l4163_416321

/-- The function f(x) = x^2 + 2 -/
def f (x : ℝ) : ℝ := x^2 + 2

/-- The function g(x) = 3x - 2 -/
def g (x : ℝ) : ℝ := 3*x - 2

/-- Theorem: f(g(3)) = 51 -/
theorem f_of_g_of_3 : f (g 3) = 51 := by
  sorry

end NUMINAMATH_CALUDE_f_of_g_of_3_l4163_416321


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l4163_416308

/-- Given a boat that travels 6 km/hr along a stream and 2 km/hr against the same stream,
    its speed in still water is 4 km/hr. -/
theorem boat_speed_in_still_water (boat_speed : ℝ) (stream_speed : ℝ) : 
  (boat_speed + stream_speed = 6) → 
  (boat_speed - stream_speed = 2) → 
  boat_speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l4163_416308


namespace NUMINAMATH_CALUDE_preimage_of_four_one_l4163_416368

/-- The mapping f from R² to R² defined by f(x,y) = (x+y, x-y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + p.2, p.1 - p.2)

/-- Theorem stating that (2.5, 1.5) is the preimage of (4, 1) under f -/
theorem preimage_of_four_one :
  f (2.5, 1.5) = (4, 1) := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_four_one_l4163_416368


namespace NUMINAMATH_CALUDE_first_girl_siblings_l4163_416348

-- Define the number of girls in the survey
def num_girls : ℕ := 9

-- Define the mean number of siblings
def mean_siblings : ℚ := 5.7

-- Define the list of known sibling counts
def known_siblings : List ℕ := [6, 10, 4, 3, 3, 11, 3, 10]

-- Define the sum of known sibling counts
def sum_known_siblings : ℕ := known_siblings.sum

-- Theorem to prove
theorem first_girl_siblings :
  ∃ (x : ℕ), x + sum_known_siblings = Int.floor (mean_siblings * num_girls) ∧ x = 1 := by
  sorry


end NUMINAMATH_CALUDE_first_girl_siblings_l4163_416348


namespace NUMINAMATH_CALUDE_power_sum_equality_l4163_416332

theorem power_sum_equality (a : ℕ) (h : 2^50 = a) :
  2^50 + 2^51 + 2^52 + 2^53 + 2^54 + 2^55 + 2^56 + 2^57 + 2^58 + 2^59 +
  2^60 + 2^61 + 2^62 + 2^63 + 2^64 + 2^65 + 2^66 + 2^67 + 2^68 + 2^69 +
  2^70 + 2^71 + 2^72 + 2^73 + 2^74 + 2^75 + 2^76 + 2^77 + 2^78 + 2^79 +
  2^80 + 2^81 + 2^82 + 2^83 + 2^84 + 2^85 + 2^86 + 2^87 + 2^88 + 2^89 +
  2^90 + 2^91 + 2^92 + 2^93 + 2^94 + 2^95 + 2^96 + 2^97 + 2^98 + 2^99 + 2^100 = 2*a^2 - a := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l4163_416332


namespace NUMINAMATH_CALUDE_fixed_distance_theorem_l4163_416349

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def is_fixed_distance (p a b : V) : Prop :=
  ∃ (k : ℝ), ∀ (q : V), ‖p - q‖ = k → q = (4/3 : ℝ) • a - (1/3 : ℝ) • b

theorem fixed_distance_theorem (a b p : V) 
  (h : ‖p - b‖ = 2 * ‖p - a‖) : is_fixed_distance p a b := by
  sorry

end NUMINAMATH_CALUDE_fixed_distance_theorem_l4163_416349


namespace NUMINAMATH_CALUDE_trajectory_of_C_l4163_416310

-- Define the triangle ABC
def triangle_ABC (C : ℝ × ℝ) : Prop :=
  let A := (-2, 0)
  let B := (2, 0)
  let perimeter := dist A C + dist B C + dist A B
  perimeter = 10

-- Define the equation of the trajectory
def trajectory_equation (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 / 5 = 1 ∧ y ≠ 0

-- Theorem statement
theorem trajectory_of_C :
  ∀ C : ℝ × ℝ, triangle_ABC C → 
  ∃ x y : ℝ, C = (x, y) ∧ trajectory_equation x y :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_C_l4163_416310


namespace NUMINAMATH_CALUDE_sqrt_3x_minus_1_defined_l4163_416318

theorem sqrt_3x_minus_1_defined (x : ℝ) : Real.sqrt (3 * x - 1) ≥ 0 ↔ x ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3x_minus_1_defined_l4163_416318


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l4163_416370

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a, b := p.b - 2 * p.a * h, c := p.c + p.a * h^2 - p.b * h }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c + v }

theorem parabola_shift_theorem :
  let original := Parabola.mk 1 0 1  -- y = x^2 + 1
  let shifted_left := shift_horizontal original 2
  let final := shift_vertical shifted_left (-3)
  final = Parabola.mk 1 (-4) (-2)  -- y = (x + 2)^2 - 2
  := by sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l4163_416370


namespace NUMINAMATH_CALUDE_emmanuel_jelly_beans_l4163_416388

theorem emmanuel_jelly_beans (total : ℕ) (thomas_percent : ℚ) (barry_ratio : ℕ) (emmanuel_ratio : ℕ) : 
  total = 200 →
  thomas_percent = 1/10 →
  barry_ratio = 4 →
  emmanuel_ratio = 5 →
  (emmanuel_ratio * (total - thomas_percent * total)) / (barry_ratio + emmanuel_ratio) = 100 := by
sorry

end NUMINAMATH_CALUDE_emmanuel_jelly_beans_l4163_416388


namespace NUMINAMATH_CALUDE_priya_speed_calculation_l4163_416398

/-- Priya's speed in km/h -/
def priya_speed : ℝ := 30

/-- Riya's speed in km/h -/
def riya_speed : ℝ := 20

/-- Time traveled in hours -/
def time : ℝ := 0.5

/-- Distance between Riya and Priya after traveling -/
def distance : ℝ := 25

theorem priya_speed_calculation :
  (riya_speed + priya_speed) * time = distance :=
by sorry

end NUMINAMATH_CALUDE_priya_speed_calculation_l4163_416398


namespace NUMINAMATH_CALUDE_fourth_root_over_seventh_root_of_seven_l4163_416347

theorem fourth_root_over_seventh_root_of_seven (x : ℝ) :
  (x^(1/4)) / (x^(1/7)) = x^(3/28) :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_over_seventh_root_of_seven_l4163_416347


namespace NUMINAMATH_CALUDE_M_in_fourth_quadrant_l4163_416381

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def is_in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The given point M -/
def M : Point :=
  { x := 3, y := -2 }

/-- Theorem stating that M is in the fourth quadrant -/
theorem M_in_fourth_quadrant : is_in_fourth_quadrant M := by
  sorry


end NUMINAMATH_CALUDE_M_in_fourth_quadrant_l4163_416381


namespace NUMINAMATH_CALUDE_r_eq_m_times_phi_l4163_416319

/-- The algorithm for writing numbers on intersecting circles -/
def writeNumbers (m : ℕ) (n : ℕ) : Set (ℕ × ℕ) := sorry

/-- The number of appearances of a number on the circles -/
def r (n : ℕ) (m : ℕ) : ℕ := sorry

/-- Euler's totient function -/
def φ : ℕ → ℕ := sorry

/-- Theorem stating the relationship between r(n,m) and φ(n) -/
theorem r_eq_m_times_phi (n : ℕ) (m : ℕ) :
  r n m = m * φ n := by sorry

end NUMINAMATH_CALUDE_r_eq_m_times_phi_l4163_416319


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l4163_416338

theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![x, 1]
  let b : Fin 2 → ℝ := ![1, -1]
  (∃ (k : ℝ), a = k • b) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l4163_416338


namespace NUMINAMATH_CALUDE_prob_red_then_white_l4163_416320

/-- The probability of drawing a red marble first and a white marble second without replacement
    from a bag containing 3 red marbles and 5 white marbles is 15/56. -/
theorem prob_red_then_white (red : ℕ) (white : ℕ) (total : ℕ) (h1 : red = 3) (h2 : white = 5) 
  (h3 : total = red + white) :
  (red / total) * (white / (total - 1)) = 15 / 56 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_then_white_l4163_416320


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l4163_416335

/-- The eccentricity of a hyperbola with equation x²/4 - y² = 1 is √5/2 -/
theorem hyperbola_eccentricity : 
  let a : ℝ := 2
  let b : ℝ := 1
  let c : ℝ := Real.sqrt 5
  let e : ℝ := c / a
  (∀ x y : ℝ, x^2/4 - y^2 = 1 → e = Real.sqrt 5 / 2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l4163_416335


namespace NUMINAMATH_CALUDE_burger_cost_is_87_l4163_416384

/-- The cost of Uri's purchase in cents -/
def uri_cost : ℕ := 385

/-- The cost of Gen's purchase in cents -/
def gen_cost : ℕ := 360

/-- The number of burgers Uri bought -/
def uri_burgers : ℕ := 3

/-- The number of sodas Uri bought -/
def uri_sodas : ℕ := 2

/-- The number of burgers Gen bought -/
def gen_burgers : ℕ := 2

/-- The number of sodas Gen bought -/
def gen_sodas : ℕ := 3

/-- The cost of a burger in cents -/
def burger_cost : ℕ := 87

theorem burger_cost_is_87 :
  uri_burgers * burger_cost + uri_sodas * ((uri_cost - uri_burgers * burger_cost) / uri_sodas) = uri_cost ∧
  gen_burgers * burger_cost + gen_sodas * ((uri_cost - uri_burgers * burger_cost) / uri_sodas) = gen_cost :=
by sorry

end NUMINAMATH_CALUDE_burger_cost_is_87_l4163_416384


namespace NUMINAMATH_CALUDE_function_equality_l4163_416328

open Real

theorem function_equality (f : ℝ → ℝ) (h1 : Differentiable ℝ f) 
  (h2 : ∀ x, f x = x^2 + 2 * (deriv f 1) * x + 3) : 
  f 0 = f 4 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l4163_416328


namespace NUMINAMATH_CALUDE_four_digit_perfect_square_prefix_l4163_416353

theorem four_digit_perfect_square_prefix : ∃ (N : ℕ), 
  (1000 ≤ N ∧ N < 10000) ∧ 
  (∃ (k : ℕ), 4000000 + N = k^2) ∧
  (N = 4001 ∨ N = 8004) := by
  sorry

end NUMINAMATH_CALUDE_four_digit_perfect_square_prefix_l4163_416353


namespace NUMINAMATH_CALUDE_first_satellite_launched_by_soviet_union_l4163_416356

/-- Represents countries capable of space exploration --/
inductive SpaceExploringCountry
  | UnitedStates
  | SovietUnion
  | EuropeanUnion
  | Germany

/-- Represents an artificial Earth satellite --/
structure ArtificialEarthSatellite where
  launchDate : Nat
  launchCountry : SpaceExploringCountry
  usedMultistageRocket : Bool
  markedSpaceAgeBeginning : Bool

/-- The world's first artificial Earth satellite --/
def firstSatellite : ArtificialEarthSatellite :=
  { launchDate := 19571004  -- October 4, 1957
  , launchCountry := SpaceExploringCountry.SovietUnion
  , usedMultistageRocket := true
  , markedSpaceAgeBeginning := true }

theorem first_satellite_launched_by_soviet_union :
  firstSatellite.launchCountry = SpaceExploringCountry.SovietUnion ∧
  firstSatellite.launchDate = 19571004 ∧
  firstSatellite.usedMultistageRocket = true ∧
  firstSatellite.markedSpaceAgeBeginning = true :=
by sorry

end NUMINAMATH_CALUDE_first_satellite_launched_by_soviet_union_l4163_416356


namespace NUMINAMATH_CALUDE_tan_double_angle_l4163_416361

theorem tan_double_angle (α : ℝ) (h : (1 + Real.cos (2 * α)) / Real.sin (2 * α) = 1/2) :
  Real.tan (2 * α) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l4163_416361


namespace NUMINAMATH_CALUDE_inscribed_rectangle_circle_circumference_l4163_416302

/-- Given a rectangle with sides 9 cm and 12 cm inscribed in a circle,
    prove that the circumference of the circle is 15π cm. -/
theorem inscribed_rectangle_circle_circumference :
  ∀ (circle : Real → Real → Prop) (rectangle : Real → Real → Prop),
    (∃ (x y : Real), rectangle x y ∧ x = 9 ∧ y = 12) →
    (∀ (x y : Real), rectangle x y → ∃ (center : Real × Real) (r : Real),
      circle = λ a b => (a - center.1)^2 + (b - center.2)^2 = r^2) →
    (∃ (circumference : Real), circumference = 15 * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_circle_circumference_l4163_416302


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_of_7_pow_1024_l4163_416371

/-- The sum of the tens digit and the units digit in the decimal representation of 7^1024 is 17. -/
theorem sum_of_last_two_digits_of_7_pow_1024 :
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ (7^1024 : ℕ) % 100 = 10 * a + b ∧ a + b = 17 := by
sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_of_7_pow_1024_l4163_416371


namespace NUMINAMATH_CALUDE_fort_blocks_count_l4163_416357

/-- Calculates the number of blocks needed for a fort with given dimensions and wall thickness -/
def fort_blocks (length width height wall_thickness : ℕ) : ℕ :=
  let outer_volume := length * width * height
  let inner_length := length - 2 * wall_thickness
  let inner_width := width - 2 * wall_thickness
  let inner_height := height - wall_thickness
  let inner_volume := inner_length * inner_width * inner_height
  outer_volume - inner_volume

/-- Theorem stating that a fort with given dimensions requires 728 blocks -/
theorem fort_blocks_count :
  fort_blocks 15 12 6 2 = 728 := by sorry

end NUMINAMATH_CALUDE_fort_blocks_count_l4163_416357


namespace NUMINAMATH_CALUDE_total_packs_is_35_l4163_416333

/-- The number of packs sold by Lucy -/
def lucy_packs : ℕ := 19

/-- The number of packs sold by Robyn -/
def robyn_packs : ℕ := 16

/-- The total number of packs sold by Robyn and Lucy -/
def total_packs : ℕ := lucy_packs + robyn_packs

/-- Theorem stating that the total number of packs sold is 35 -/
theorem total_packs_is_35 : total_packs = 35 := by
  sorry

end NUMINAMATH_CALUDE_total_packs_is_35_l4163_416333


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l4163_416363

theorem rectangle_perimeter (a b : ℕ) : 
  b = 3 * a →                   -- One side is three times as long as the other
  a * b = 2 * (a + b) + 12 →    -- Area equals perimeter plus 12
  2 * (a + b) = 32              -- Perimeter is 32 units
  := by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l4163_416363


namespace NUMINAMATH_CALUDE_complement_of_intersection_l4163_416389

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem complement_of_intersection (U M N : Set ℕ) 
  (hU : U = {1, 2, 3, 4}) 
  (hM : M = {1, 2, 3}) 
  (hN : N = {2, 3, 4}) : 
  (M ∩ N)ᶜ = {1, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l4163_416389


namespace NUMINAMATH_CALUDE_negation_of_proposition_l4163_416386

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x ≥ 0 → x^2 - x + 1 ≥ 0)) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^2 - x + 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l4163_416386


namespace NUMINAMATH_CALUDE_largest_four_digit_congruence_l4163_416313

theorem largest_four_digit_congruence :
  ∃ (n : ℕ), 
    n ≤ 9999 ∧ 
    n ≥ 1000 ∧ 
    45 * n ≡ 180 [MOD 315] ∧
    ∀ (m : ℕ), m ≤ 9999 ∧ m ≥ 1000 ∧ 45 * m ≡ 180 [MOD 315] → m ≤ n ∧
    n = 9993 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_congruence_l4163_416313


namespace NUMINAMATH_CALUDE_sum_of_roots_of_quartic_l4163_416307

theorem sum_of_roots_of_quartic (x : ℝ) : 
  (∃ a b c d : ℝ, x^4 - 6*x^3 + 8*x - 3 = (x^2 + a*x + b)*(x^2 + c*x + d)) →
  (∃ r₁ r₂ r₃ r₄ : ℝ, x^4 - 6*x^3 + 8*x - 3 = (x - r₁)*(x - r₂)*(x - r₃)*(x - r₄) ∧ r₁ + r₂ + r₃ + r₄ = 8) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_of_quartic_l4163_416307


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l4163_416390

/-- The focus of the parabola y = 8x^2 has coordinates (0, 1/32) -/
theorem parabola_focus_coordinates :
  let f : ℝ × ℝ → ℝ := fun (x, y) ↦ y - 8 * x^2
  ∃! p : ℝ × ℝ, p = (0, 1/32) ∧ ∀ x y, f (x, y) = 0 → (x - p.1)^2 = 4 * p.2 * (y - p.2) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l4163_416390


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l4163_416326

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 3}
def B : Set Nat := {3, 4, 5}

theorem intersection_complement_equality : A ∩ (U \ B) = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l4163_416326


namespace NUMINAMATH_CALUDE_graduating_class_size_l4163_416397

/-- Given a graduating class where there are 208 boys and 69 more girls than boys,
    prove that the total number of students is 485. -/
theorem graduating_class_size :
  ∀ (boys girls total : ℕ),
  boys = 208 →
  girls = boys + 69 →
  total = boys + girls →
  total = 485 := by
sorry

end NUMINAMATH_CALUDE_graduating_class_size_l4163_416397


namespace NUMINAMATH_CALUDE_specific_pentagon_perimeter_l4163_416392

/-- Pentagon ABCDE with specific side lengths -/
structure Pentagon where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  AE : ℝ

/-- The perimeter of a pentagon -/
def perimeter (p : Pentagon) : ℝ :=
  p.AB + p.BC + p.CD + p.DE + p.AE

/-- Theorem: The perimeter of the specific pentagon is 12 -/
theorem specific_pentagon_perimeter :
  ∃ (p : Pentagon),
    p.AB = 2 ∧ p.BC = 2 ∧ p.CD = 2 ∧ p.DE = 2 ∧
    p.AE ^ 2 = (p.AB + p.BC) ^ 2 + (p.CD + p.DE) ^ 2 ∧
    perimeter p = 12 := by
  sorry


end NUMINAMATH_CALUDE_specific_pentagon_perimeter_l4163_416392


namespace NUMINAMATH_CALUDE_triangle_sine_inequality_l4163_416342

theorem triangle_sine_inequality (A B C : ℝ) (h : A + B + C = π) :
  -2 < Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) ∧
  Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) ≤ 3 * Real.sqrt 3 / 2 ∧
  (Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) = 3 * Real.sqrt 3 / 2 ↔
   A = 7 * π / 9 ∧ B = π / 9 ∧ C = π / 9) :=
by sorry

end NUMINAMATH_CALUDE_triangle_sine_inequality_l4163_416342


namespace NUMINAMATH_CALUDE_line_through_midpoint_l4163_416366

-- Define the lines l1 and l2
def l1 (x y : ℝ) : Prop := x - 3*y + 10 = 0
def l2 (x y : ℝ) : Prop := 2*x + y - 8 = 0

-- Define point P
def P : ℝ × ℝ := (0, 1)

-- Define the line l
def l (x y : ℝ) : Prop := x + 4*y - 4 = 0

-- Theorem statement
theorem line_through_midpoint (A B : ℝ × ℝ) :
  l A.1 A.2 →
  l B.1 B.2 →
  l1 A.1 A.2 →
  l2 B.1 B.2 →
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  ∀ x y, l x y ↔ x + 4*y - 4 = 0 :=
sorry

end NUMINAMATH_CALUDE_line_through_midpoint_l4163_416366


namespace NUMINAMATH_CALUDE_some_club_members_not_committee_members_l4163_416359

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (ClubMember : U → Prop)
variable (CommitteeMember : U → Prop)
variable (Punctual : U → Prop)

-- State the theorem
theorem some_club_members_not_committee_members :
  (∃ x, ClubMember x ∧ ¬Punctual x) →
  (∀ x, CommitteeMember x → Punctual x) →
  ∃ x, ClubMember x ∧ ¬CommitteeMember x :=
by
  sorry


end NUMINAMATH_CALUDE_some_club_members_not_committee_members_l4163_416359


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l4163_416358

theorem trigonometric_simplification (α : ℝ) : 
  (Real.tan ((5 / 4) * Real.pi - 4 * α) * (Real.sin ((5 / 4) * Real.pi + 4 * α))^2) / 
  (1 - 2 * (Real.cos (4 * α))^2) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l4163_416358


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_condition_l4163_416395

/-- The eccentricity of an ellipse with equation x^2 + y^2/m = 1 (m > 0) is greater than 1/2
    if and only if 0 < m < 4/3 or m > 3/4 -/
theorem ellipse_eccentricity_condition (m : ℝ) :
  (m > 0) →
  (∃ (x y : ℝ), x^2 + y^2/m = 1) →
  (∃ (e : ℝ), e > 1/2 ∧ e^2 = 1 - (min 1 m) / (max 1 m)) ↔
  (0 < m ∧ m < 4/3) ∨ m > 3/4 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_condition_l4163_416395


namespace NUMINAMATH_CALUDE_positive_real_as_infinite_sum_representations_l4163_416367

theorem positive_real_as_infinite_sum_representations (k : ℝ) (hk : k > 0) :
  ∃ (f : ℕ → (ℕ → ℕ)), 
    (∀ n : ℕ, ∀ i j : ℕ, i < j → f n i < f n j) ∧ 
    (∀ n : ℕ, k = ∑' i, (1 : ℝ) / (10 ^ (f n i))) :=
sorry

end NUMINAMATH_CALUDE_positive_real_as_infinite_sum_representations_l4163_416367


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l4163_416301

theorem quadratic_equation_solution :
  let a : ℝ := 1
  let b : ℝ := 1
  let c : ℝ := -3
  let x₁ : ℝ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ : ℝ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁^2 + x₁ - 3 = 0 ∧ x₂^2 + x₂ - 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l4163_416301


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l4163_416385

def quadratic_inequality (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c > 0

def solution_set (a b c : ℝ) := {x : ℝ | quadratic_inequality a b c x}

theorem quadratic_inequality_solution_sets
  (a b c : ℝ) (h : solution_set a b c = {x : ℝ | -2 < x ∧ x < 1}) :
  {x : ℝ | c * x^2 + a * x + b ≥ 0} = {x : ℝ | x ≤ -1/2 ∨ x ≥ 1} :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l4163_416385


namespace NUMINAMATH_CALUDE_final_bill_is_520_20_l4163_416375

/-- The final bill amount after applying two consecutive 2% late charges to an original bill of $500 -/
def final_bill_amount (original_bill : ℝ) (late_charge_rate : ℝ) : ℝ :=
  original_bill * (1 + late_charge_rate) * (1 + late_charge_rate)

/-- Theorem stating that the final bill amount is $520.20 -/
theorem final_bill_is_520_20 :
  final_bill_amount 500 0.02 = 520.20 := by sorry

end NUMINAMATH_CALUDE_final_bill_is_520_20_l4163_416375


namespace NUMINAMATH_CALUDE_problem_2023_l4163_416369

theorem problem_2023 : 
  (2023^3 - 2 * 2023^2 * 2024 + 3 * 2023 * 2024^2 - 2024^3 + 1) / (2023 * 2024) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_problem_2023_l4163_416369


namespace NUMINAMATH_CALUDE_pauls_weekend_homework_l4163_416374

/-- Represents Paul's homework schedule for a week -/
structure HomeworkSchedule where
  weeknight_hours : ℕ  -- Hours of homework on a regular weeknight
  practice_nights : ℕ  -- Number of nights with practice (no homework)
  total_weeknights : ℕ -- Total number of weeknights
  average_hours : ℕ   -- Required average hours on non-practice nights

/-- Calculates the weekend homework hours based on Paul's schedule -/
def weekend_homework (schedule : HomeworkSchedule) : ℕ :=
  let non_practice_nights := schedule.total_weeknights - schedule.practice_nights
  let required_hours := non_practice_nights * schedule.average_hours
  let available_weeknight_hours := (schedule.total_weeknights - schedule.practice_nights) * schedule.weeknight_hours
  required_hours - available_weeknight_hours

/-- Theorem stating that Paul's weekend homework is 3 hours -/
theorem pauls_weekend_homework :
  let pauls_schedule : HomeworkSchedule := {
    weeknight_hours := 2,
    practice_nights := 2,
    total_weeknights := 5,
    average_hours := 3
  }
  weekend_homework pauls_schedule = 3 := by sorry

end NUMINAMATH_CALUDE_pauls_weekend_homework_l4163_416374


namespace NUMINAMATH_CALUDE_subset_implies_a_less_than_negative_two_l4163_416322

theorem subset_implies_a_less_than_negative_two (a : ℝ) : 
  let A := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
  let B := {x : ℝ | x > a}
  A ⊆ B → a < -2 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_a_less_than_negative_two_l4163_416322


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4163_416344

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.log (2 * x - x^2)}

-- Define set B
def B : Set ℝ := {y | ∃ x > 0, y = 2^x}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4163_416344


namespace NUMINAMATH_CALUDE_thermal_underwear_sales_l4163_416309

def cost_price : ℕ := 50
def standard_price : ℕ := 70
def price_adjustments : List ℤ := [5, 2, 1, 0, -2]
def sets_sold : List ℕ := [7, 10, 15, 20, 23]

theorem thermal_underwear_sales :
  (List.sum (List.zipWith (· * ·) price_adjustments (List.map Int.ofNat sets_sold)) = 24) ∧
  ((standard_price - cost_price) * (List.sum sets_sold) + 24 = 1524) := by
  sorry

end NUMINAMATH_CALUDE_thermal_underwear_sales_l4163_416309


namespace NUMINAMATH_CALUDE_kim_money_l4163_416378

/-- Given that Kim has 40% more money than Sal, Sal has 20% less money than Phil,
    and Sal and Phil have a combined total of $1.80, prove that Kim has $1.12. -/
theorem kim_money (sal phil kim : ℝ) 
  (h1 : kim = sal * 1.4)
  (h2 : sal = phil * 0.8)
  (h3 : sal + phil = 1.8) : 
  kim = 1.12 := by
sorry

end NUMINAMATH_CALUDE_kim_money_l4163_416378


namespace NUMINAMATH_CALUDE_simple_interest_time_l4163_416365

/-- Given simple interest, principal, and rate, calculate the time in years -/
theorem simple_interest_time (SI P R : ℚ) (h1 : SI = 4016.25) (h2 : P = 16065) (h3 : R = 5) :
  SI = P * R * 5 / 100 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_time_l4163_416365


namespace NUMINAMATH_CALUDE_work_completion_time_proportional_aarti_work_completion_time_l4163_416334

/-- If a person can complete a piece of work in a given number of days,
    then the time to complete a multiple of that work is proportional to the multiple. -/
theorem work_completion_time_proportional
  (original_days : ℕ) (work_multiple : ℕ) :
  original_days * work_multiple = original_days * work_multiple :=
by sorry

/-- Aarti's work completion time -/
theorem aarti_work_completion_time :
  let original_days : ℕ := 6
  let work_multiple : ℕ := 3
  original_days * work_multiple = 18 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_proportional_aarti_work_completion_time_l4163_416334


namespace NUMINAMATH_CALUDE_cubic_decreasing_implies_m_negative_l4163_416399

/-- A cubic function f(x) = mx³ - x --/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^3 - x

/-- The derivative of f(x) --/
def f_deriv (m : ℝ) (x : ℝ) : ℝ := 3 * m * x^2 - 1

/-- A function is decreasing if its derivative is non-positive for all x --/
def is_decreasing (g : ℝ → ℝ) : Prop := ∀ x, g x ≤ 0

theorem cubic_decreasing_implies_m_negative (m : ℝ) :
  is_decreasing (f_deriv m) → m < 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_decreasing_implies_m_negative_l4163_416399


namespace NUMINAMATH_CALUDE_writing_stats_theorem_l4163_416372

/-- Represents the writing statistics of an author -/
structure WritingStats where
  total_words : ℕ
  total_hours : ℕ
  first_half_hours : ℕ
  first_half_words : ℕ

/-- Calculates the average words per hour -/
def average_words_per_hour (words : ℕ) (hours : ℕ) : ℚ :=
  (words : ℚ) / (hours : ℚ)

/-- Theorem about the writing statistics -/
theorem writing_stats_theorem (stats : WritingStats) 
  (h1 : stats.total_words = 60000)
  (h2 : stats.total_hours = 150)
  (h3 : stats.first_half_hours = 50)
  (h4 : stats.first_half_words = stats.total_words / 2) :
  average_words_per_hour stats.total_words stats.total_hours = 400 ∧
  average_words_per_hour stats.first_half_words stats.first_half_hours = 600 := by
  sorry


end NUMINAMATH_CALUDE_writing_stats_theorem_l4163_416372


namespace NUMINAMATH_CALUDE_optimal_price_and_range_l4163_416317

-- Define the linear relationship between quantity and price
def quantity (x : ℝ) : ℝ := -2 * x + 100

-- Define the cost per item
def cost : ℝ := 20

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - cost) * quantity x

-- Statement to prove
theorem optimal_price_and_range :
  -- The price that maximizes profit is 35
  (∃ (x_max : ℝ), x_max = 35 ∧ ∀ (x : ℝ), profit x ≤ profit x_max) ∧
  -- The range of prices that ensures at least 30 items sold and a profit of at least 400
  (∀ (x : ℝ), 30 ≤ x ∧ x ≤ 35 ↔ quantity x ≥ 30 ∧ profit x ≥ 400) :=
by sorry

end NUMINAMATH_CALUDE_optimal_price_and_range_l4163_416317


namespace NUMINAMATH_CALUDE_bianmin_logistics_problem_bianmin_logistics_solution_l4163_416355

/-- The Bianmin Logistics Company problem -/
theorem bianmin_logistics_problem (total_artworks : ℕ) (shipping_cost : ℚ) 
  (compensation_cost : ℚ) (total_profit : ℚ) (broken_artworks : ℕ) : Prop :=
  total_artworks = 2000 ∧
  shipping_cost = 0.2 ∧
  compensation_cost = 2.3 ∧
  total_profit = 390 ∧
  shipping_cost * (total_artworks - broken_artworks : ℚ) - compensation_cost * broken_artworks = total_profit →
  broken_artworks = 4

/-- Proof of the Bianmin Logistics Company problem -/
theorem bianmin_logistics_solution : 
  ∃ (total_artworks : ℕ) (shipping_cost compensation_cost total_profit : ℚ) (broken_artworks : ℕ),
    bianmin_logistics_problem total_artworks shipping_cost compensation_cost total_profit broken_artworks :=
by
  sorry

end NUMINAMATH_CALUDE_bianmin_logistics_problem_bianmin_logistics_solution_l4163_416355


namespace NUMINAMATH_CALUDE_simple_random_sampling_probability_l4163_416380

theorem simple_random_sampling_probability 
  (population_size : ℕ) 
  (sample_size : ℕ) 
  (h1 : population_size = 100) 
  (h2 : sample_size = 30) :
  (sample_size : ℚ) / (population_size : ℚ) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_simple_random_sampling_probability_l4163_416380


namespace NUMINAMATH_CALUDE_road_length_proof_l4163_416393

/-- The total length of the road in meters -/
def total_length : ℝ := 1000

/-- The length repaired in the first week in meters -/
def first_week_repair : ℝ := 0.2 * total_length

/-- The length repaired in the second week in meters -/
def second_week_repair : ℝ := 0.25 * total_length

/-- The length repaired in the third week in meters -/
def third_week_repair : ℝ := 480

/-- The length remaining unrepaired in meters -/
def remaining_length : ℝ := 70

theorem road_length_proof :
  first_week_repair + second_week_repair + third_week_repair + remaining_length = total_length := by
  sorry

end NUMINAMATH_CALUDE_road_length_proof_l4163_416393


namespace NUMINAMATH_CALUDE_range_of_a_satisfying_condition_l4163_416315

/-- The universal set U is the set of real numbers. -/
def U : Set ℝ := Set.univ

/-- Set A is defined as {x | (x - 2)(x - 9) < 0}. -/
def A : Set ℝ := {x | (x - 2) * (x - 9) < 0}

/-- Set B is defined as {x | -2 - x ≤ 0 ≤ 5 - x}. -/
def B : Set ℝ := {x | -2 - x ≤ 0 ∧ 0 ≤ 5 - x}

/-- Set C is defined as {x | a ≤ x ≤ 2 - a}, where a is a real number. -/
def C (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 2 - a}

/-- The theorem states that given the conditions, the range of values for a that satisfies C ∪ (∁ₘB) = R is (-∞, -3]. -/
theorem range_of_a_satisfying_condition :
  ∀ a : ℝ, (C a ∪ (U \ B) = U) ↔ a ≤ -3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_satisfying_condition_l4163_416315


namespace NUMINAMATH_CALUDE_max_y_over_x_l4163_416329

theorem max_y_over_x (x y : ℝ) 
  (h1 : x - 1 ≥ 0) 
  (h2 : x - y ≥ 0) 
  (h3 : x + y - 4 ≤ 0) : 
  ∃ (max : ℝ), max = 3 ∧ ∀ (x' y' : ℝ), 
    x' - 1 ≥ 0 → x' - y' ≥ 0 → x' + y' - 4 ≤ 0 → y' / x' ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_y_over_x_l4163_416329


namespace NUMINAMATH_CALUDE_inequality_system_solution_fractional_equation_no_solution_l4163_416336

-- Part 1: System of inequalities
def inequality_system (x : ℝ) : Prop :=
  (1 - x ≤ 2) ∧ ((x + 1) / 2 + (x - 1) / 3 < 1)

theorem inequality_system_solution :
  {x : ℝ | inequality_system x} = {x : ℝ | -1 ≤ x ∧ x < 1} :=
sorry

-- Part 2: Fractional equation
def fractional_equation (x : ℝ) : Prop :=
  (x - 2) / (x + 2) - 1 = 16 / (x^2 - 4)

theorem fractional_equation_no_solution :
  ¬∃ x : ℝ, fractional_equation x :=
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_fractional_equation_no_solution_l4163_416336


namespace NUMINAMATH_CALUDE_tan_graph_product_l4163_416373

theorem tan_graph_product (a b : ℝ) : 
  a > 0 → b > 0 → 
  (π / b = 2 * π / 3) →
  (a * Real.tan (b * (π / 6)) = 2) →
  a * b = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_graph_product_l4163_416373


namespace NUMINAMATH_CALUDE_two_cars_speed_l4163_416394

/-- Two cars traveling in the same direction with given conditions -/
theorem two_cars_speed (t v₁ S₁ S₂ : ℝ) (h_t : t = 30) (h_v₁ : v₁ = 25)
  (h_S₁ : S₁ = 100) (h_S₂ : S₂ = 400) :
  ∃ v₂ : ℝ, (v₂ = 35 ∨ v₂ = 15) ∧
  (S₂ - S₁) / t = |v₂ - v₁| :=
by sorry

end NUMINAMATH_CALUDE_two_cars_speed_l4163_416394


namespace NUMINAMATH_CALUDE_alexanders_pictures_l4163_416331

theorem alexanders_pictures (total_pencils : ℕ) 
  (new_galleries : ℕ) (pictures_per_new_gallery : ℕ) 
  (pencils_per_picture : ℕ) (pencils_per_exhibition : ℕ) : 
  total_pencils = 88 →
  new_galleries = 5 →
  pictures_per_new_gallery = 2 →
  pencils_per_picture = 4 →
  pencils_per_exhibition = 2 →
  (total_pencils - 
    (new_galleries * pictures_per_new_gallery * pencils_per_picture) - 
    ((new_galleries + 1) * pencils_per_exhibition)) / pencils_per_picture = 9 :=
by sorry

end NUMINAMATH_CALUDE_alexanders_pictures_l4163_416331


namespace NUMINAMATH_CALUDE_probability_rain_given_east_wind_l4163_416383

/-- The probability of an east wind blowing -/
def P_east_wind : ℚ := 3/10

/-- The probability of rain -/
def P_rain : ℚ := 11/30

/-- The probability of both an east wind blowing and rain -/
def P_east_wind_and_rain : ℚ := 4/15

/-- The probability of rain given that there is an east wind blowing -/
def P_rain_given_east_wind : ℚ := P_east_wind_and_rain / P_east_wind

theorem probability_rain_given_east_wind :
  P_rain_given_east_wind = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_probability_rain_given_east_wind_l4163_416383


namespace NUMINAMATH_CALUDE_inequality_proof_l4163_416396

theorem inequality_proof (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_min : min (a + b) (min (b + c) (c + a)) > Real.sqrt 2)
  (h_sum : a^2 + b^2 + c^2 = 3) : 
  a / (b + c - a)^2 + b / (c + a - b)^2 + c / (a + b - c)^2 ≥ 3 / (a * b * c)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4163_416396


namespace NUMINAMATH_CALUDE_original_bill_l4163_416323

theorem original_bill (new_bill : ℝ) (increase_percent : ℝ) (h1 : new_bill = 78) (h2 : increase_percent = 30) :
  ∃ original_bill : ℝ, original_bill * (1 + increase_percent / 100) = new_bill ∧ original_bill = 60 := by
  sorry

end NUMINAMATH_CALUDE_original_bill_l4163_416323


namespace NUMINAMATH_CALUDE_cos_equation_solution_l4163_416325

theorem cos_equation_solution (x : ℝ) : 
  (Real.cos (2 * x) - 2 * Real.cos (4 * x))^2 = 9 + (Real.cos (5 * x))^2 ↔ 
  ∃ k : ℤ, x = π / 2 + k * π :=
sorry

end NUMINAMATH_CALUDE_cos_equation_solution_l4163_416325


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l4163_416343

/-- The minimum distance between a point on y = (1/2)e^x and a point on y = ln(2x) -/
theorem min_distance_between_curves : ∃ (min_dist : ℝ),
  min_dist = Real.sqrt 2 * (1 - Real.log 2) ∧
  ∀ (x₁ x₂ : ℝ),
    let p := (x₁, (1/2) * Real.exp x₁)
    let q := (x₂, Real.log (2 * x₂))
    Real.sqrt ((x₁ - x₂)^2 + ((1/2) * Real.exp x₁ - Real.log (2 * x₂))^2) ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l4163_416343


namespace NUMINAMATH_CALUDE_intersection_line_equation_l4163_416340

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 12 = 0

-- Define the line
def line (x y : ℝ) : Prop := x - 2*y + 6 = 0

-- Theorem statement
theorem intersection_line_equation :
  ∃ (A B : ℝ × ℝ),
    (A.1 ≠ B.1 ∨ A.2 ≠ B.2) ∧
    circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧
    circle2 A.1 A.2 ∧ circle2 B.1 B.2 ∧
    (∀ (x y : ℝ), circle1 x y ∧ circle2 x y → line x y) :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l4163_416340


namespace NUMINAMATH_CALUDE_households_using_both_brands_l4163_416350

/-- Proves that the number of households using both brands of soap is 25 --/
theorem households_using_both_brands (total : ℕ) (neither : ℕ) (only_a : ℕ) (h1 : total = 240) (h2 : neither = 80) (h3 : only_a = 60) : 
  ∃ (both : ℕ), both = 25 ∧ total = neither + only_a + both + 3 * both := by
  sorry

end NUMINAMATH_CALUDE_households_using_both_brands_l4163_416350


namespace NUMINAMATH_CALUDE_factorization_correctness_l4163_416305

theorem factorization_correctness (x : ℝ) : 3 * x^2 - 2*x - 1 = (3*x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_correctness_l4163_416305


namespace NUMINAMATH_CALUDE_students_in_both_band_and_chorus_l4163_416316

theorem students_in_both_band_and_chorus 
  (total : ℕ) 
  (band : ℕ) 
  (chorus : ℕ) 
  (band_or_chorus : ℕ) 
  (h1 : total = 300)
  (h2 : band = 100)
  (h3 : chorus = 120)
  (h4 : band_or_chorus = 195) :
  band + chorus - band_or_chorus = 25 :=
by sorry

end NUMINAMATH_CALUDE_students_in_both_band_and_chorus_l4163_416316


namespace NUMINAMATH_CALUDE_product_of_exponents_l4163_416327

theorem product_of_exponents (p r s : ℕ) : 
  3^p + 3^3 = 36 → 2^r + 18 = 50 → 5^s + 7^2 = 1914 → p * r * s = 40 := by
  sorry

end NUMINAMATH_CALUDE_product_of_exponents_l4163_416327


namespace NUMINAMATH_CALUDE_ellipse_tangent_and_normal_l4163_416346

noncomputable def ellipse (t : ℝ) : ℝ × ℝ := (4 * Real.cos t, 3 * Real.sin t)

theorem ellipse_tangent_and_normal (t : ℝ) :
  let (x₀, y₀) := ellipse (π/3)
  let tangent_slope := -(3 * Real.cos (π/3)) / (4 * Real.sin (π/3))
  let normal_slope := -1 / tangent_slope
  (∀ x y, y - y₀ = tangent_slope * (x - x₀) ↔ y = -Real.sqrt 3 / 4 * x + 2 * Real.sqrt 3) ∧
  (∀ x y, y - y₀ = normal_slope * (x - x₀) ↔ y = 4 / Real.sqrt 3 * x - 7 * Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_tangent_and_normal_l4163_416346


namespace NUMINAMATH_CALUDE_no_m_for_all_x_x_range_for_bounded_m_l4163_416303

-- Define the inequality function
def f (m x : ℝ) : ℝ := m * x^2 - 2*x - m + 1

-- Statement 1
theorem no_m_for_all_x : ¬ ∃ m : ℝ, ∀ x : ℝ, f m x < 0 := by sorry

-- Statement 2
theorem x_range_for_bounded_m :
  ∀ m : ℝ, |m| ≤ 2 →
  ∀ x : ℝ, ((-1 + Real.sqrt 7) / 2 < x ∧ x < (1 + Real.sqrt 3) / 2) →
  f m x < 0 := by sorry

end NUMINAMATH_CALUDE_no_m_for_all_x_x_range_for_bounded_m_l4163_416303


namespace NUMINAMATH_CALUDE_leftSideSeats_l4163_416351

/-- Represents the seating arrangement in a bus -/
structure BusSeats where
  leftSeats : ℕ
  rightSeats : ℕ
  backSeat : ℕ
  peoplePerSeat : ℕ
  totalCapacity : ℕ

/-- The bus seating arrangement satisfies the given conditions -/
def validBusSeats (bus : BusSeats) : Prop :=
  bus.rightSeats = bus.leftSeats - 3 ∧
  bus.peoplePerSeat = 3 ∧
  bus.backSeat = 9 ∧
  bus.totalCapacity = 90

/-- The theorem stating that the number of seats on the left side is 15 -/
theorem leftSideSeats (bus : BusSeats) (h : validBusSeats bus) : 
  bus.leftSeats = 15 := by
  sorry

#check leftSideSeats

end NUMINAMATH_CALUDE_leftSideSeats_l4163_416351


namespace NUMINAMATH_CALUDE_regression_equation_proof_l4163_416311

theorem regression_equation_proof (x y z : ℝ) (b a : ℝ) :
  (y = Real.exp (b * x + a)) →
  (z = Real.log y) →
  (z = 0.25 * x - 2.58) →
  (y = Real.exp (0.25 * x - 2.58)) := by
  sorry

end NUMINAMATH_CALUDE_regression_equation_proof_l4163_416311


namespace NUMINAMATH_CALUDE_range_of_roots_difference_l4163_416337

-- Define the function g
def g (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- Define the derivative of g as f
def f (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

-- State the theorem
theorem range_of_roots_difference
  (a b c d : ℝ)
  (ha : a ≠ 0)
  (hsum : a + 2 * b + 3 * c = 0)
  (hpos : f a b c 0 * f a b c 1 > 0)
  (x₁ x₂ : ℝ)
  (hroot₁ : f a b c x₁ = 0)
  (hroot₂ : f a b c x₂ = 0) :
  ∃ y, y ∈ Set.Icc 0 (2/3) ∧ |x₁ - x₂| = y :=
sorry

end NUMINAMATH_CALUDE_range_of_roots_difference_l4163_416337


namespace NUMINAMATH_CALUDE_solution_to_equation_l4163_416377

theorem solution_to_equation : ∃ x : ℝ, (5 - 3*x)^5 = -1 ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_solution_to_equation_l4163_416377


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l4163_416300

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 2 → 2 / a < 1) ∧
  (∃ a, 2 / a < 1 ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l4163_416300


namespace NUMINAMATH_CALUDE_function_inequality_l4163_416382

theorem function_inequality (a x : ℝ) (h1 : a ≥ Real.exp (-2)) (h2 : x > 0) :
  a * x * Real.exp x - (x + 1)^2 ≥ Real.log x - x^2 - x - 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l4163_416382


namespace NUMINAMATH_CALUDE_equation_solution_l4163_416354

theorem equation_solution (y : ℝ) : (24 / 36 : ℝ) = Real.sqrt (y / 36) → y = 16 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4163_416354


namespace NUMINAMATH_CALUDE_system_solution_l4163_416387

theorem system_solution :
  let f (x y : ℝ) := x * Real.sqrt (1 - y^2) = (Real.sqrt 3 + 1) / 4
  let g (x y : ℝ) := y * Real.sqrt (1 - x^2) = (Real.sqrt 3 - 1) / 4
  ∀ x y : ℝ, (f x y ∧ g x y) ↔ 
    ((x = (Real.sqrt 6 + Real.sqrt 2) / 4 ∧ y = Real.sqrt 2 / 2) ∨
     (x = Real.sqrt 2 / 2 ∧ y = (Real.sqrt 6 - Real.sqrt 2) / 4)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l4163_416387


namespace NUMINAMATH_CALUDE_muffin_banana_cost_ratio_l4163_416376

/-- The cost ratio of a muffin to a banana given Susie and Calvin's purchases -/
theorem muffin_banana_cost_ratio :
  let m := (cost_of_muffin : ℚ)
  let b := (cost_of_banana : ℚ)
  let susie_cost := 5 * m + 2 * b
  let calvin_cost := 3 * m + 12 * b
  calvin_cost = 3 * susie_cost →
  m = (1 / 2 : ℚ) * b := by
sorry

end NUMINAMATH_CALUDE_muffin_banana_cost_ratio_l4163_416376


namespace NUMINAMATH_CALUDE_unique_n_congruence_l4163_416312

theorem unique_n_congruence : ∃! n : ℤ, 3 ≤ n ∧ n ≤ 9 ∧ n ≡ 12473 [ZMOD 7] ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_congruence_l4163_416312


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l4163_416362

def isArithmeticSequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sumIsTerm (a : ℕ → ℕ) : Prop :=
  ∀ p q, ∃ k, a k = a p + a q

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℕ) (d : ℕ) 
  (h1 : isArithmeticSequence a d)
  (h2 : a 1 = 9)
  (h3 : sumIsTerm a) :
  d = 1 ∨ d = 3 ∨ d = 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l4163_416362


namespace NUMINAMATH_CALUDE_b_share_correct_l4163_416345

/-- The share of the total payment for worker b -/
def b_share (a_days b_days c_days d_days total_payment : ℚ) : ℚ :=
  (1 / b_days) / ((1 / a_days) + (1 / b_days) + (1 / c_days) + (1 / d_days)) * total_payment

/-- Theorem stating that b's share is correct given the problem conditions -/
theorem b_share_correct :
  b_share 6 8 12 15 2400 = (1 / 8) / (53 / 120) * 2400 := by
  sorry

#eval b_share 6 8 12 15 2400

end NUMINAMATH_CALUDE_b_share_correct_l4163_416345


namespace NUMINAMATH_CALUDE_equilateral_hyperbola_properties_l4163_416324

/-- An equilateral hyperbola passing through point A(3,-1) with its axes of symmetry lying on the coordinate axes -/
def equilateral_hyperbola (x y : ℝ) : Prop :=
  x^2/8 - y^2/8 = 1

theorem equilateral_hyperbola_properties :
  -- The hyperbola passes through point A(3,-1)
  equilateral_hyperbola 3 (-1) ∧
  -- The axes of symmetry lie on the coordinate axes (implied by the equation form)
  ∀ (x y : ℝ), equilateral_hyperbola x y ↔ equilateral_hyperbola (-x) y ∧
  ∀ (x y : ℝ), equilateral_hyperbola x y ↔ equilateral_hyperbola x (-y) ∧
  -- The hyperbola is equilateral (asymptotes are perpendicular)
  ∃ (a : ℝ), a > 0 ∧ ∀ (x y : ℝ), equilateral_hyperbola x y ↔ x^2/a^2 - y^2/a^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_hyperbola_properties_l4163_416324


namespace NUMINAMATH_CALUDE_comic_book_stacking_order_l4163_416341

theorem comic_book_stacking_order :
  let spiderman_comics := 7
  let archie_comics := 6
  let garfield_comics := 5
  let group_arrangements := 3
  (spiderman_comics.factorial * archie_comics.factorial * garfield_comics.factorial * group_arrangements.factorial) = 248832000 := by
  sorry

end NUMINAMATH_CALUDE_comic_book_stacking_order_l4163_416341


namespace NUMINAMATH_CALUDE_robs_planned_reading_time_l4163_416339

/-- Proves that Rob's planned reading time was 3 hours given the conditions -/
theorem robs_planned_reading_time 
  (pages_read : ℕ) 
  (reading_rate : ℚ)  -- pages per minute
  (actual_time_ratio : ℚ) :
  pages_read = 9 →
  reading_rate = 1 / 15 →
  actual_time_ratio = 3 / 4 →
  (pages_read / reading_rate) / actual_time_ratio / 60 = 3 := by
sorry

end NUMINAMATH_CALUDE_robs_planned_reading_time_l4163_416339


namespace NUMINAMATH_CALUDE_equation_solution_l4163_416314

theorem equation_solution :
  ∃! x : ℚ, (4 * x - 2) / (5 * x - 5) = 3 / 4 ∧ x = -7 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l4163_416314


namespace NUMINAMATH_CALUDE_investment_scientific_notation_l4163_416364

-- Define the total investment in yuan
def total_investment : ℝ := 7.7e9

-- Define the scientific notation representation
def scientific_notation : ℝ := 7.7 * (10 ^ 9)

-- Theorem statement
theorem investment_scientific_notation : total_investment = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_investment_scientific_notation_l4163_416364


namespace NUMINAMATH_CALUDE_max_value_of_expression_l4163_416352

theorem max_value_of_expression (x : ℝ) (h : 0 ≤ x ∧ x ≤ 25) :
  Real.sqrt (x + 64) + Real.sqrt (25 - x) + 2 * Real.sqrt x ≤ 19 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l4163_416352


namespace NUMINAMATH_CALUDE_matrix_solution_l4163_416391

def determinant (a : ℝ) (x : ℝ) : ℝ :=
  (2*x + a) * ((x + a)^2 - x^2) - x * (x*(x + a) - x^2) + x * (x^2 - x*(x + a))

theorem matrix_solution (a : ℝ) (ha : a ≠ 0) :
  {x : ℝ | determinant a x = 0} = {-a/2, a/Real.sqrt 2, -a/Real.sqrt 2} :=
sorry

end NUMINAMATH_CALUDE_matrix_solution_l4163_416391


namespace NUMINAMATH_CALUDE_division_problem_l4163_416379

theorem division_problem (N : ℕ) : 
  (N / 7 = 12) ∧ (N % 7 = 5) → N = 89 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l4163_416379


namespace NUMINAMATH_CALUDE_dodecagon_ratio_l4163_416306

/-- Represents a dodecagon with specific properties -/
structure Dodecagon where
  /-- Total area of the dodecagon -/
  total_area : ℝ
  /-- Area below the bisecting line PQ -/
  area_below_pq : ℝ
  /-- Base of the triangle below PQ -/
  triangle_base : ℝ
  /-- Width of the dodecagon (XQ + QY) -/
  width : ℝ
  /-- Assertion that the dodecagon is made of 12 unit squares -/
  area_is_twelve : total_area = 12
  /-- Assertion that PQ bisects the area -/
  pq_bisects : area_below_pq = total_area / 2
  /-- Assertion about the composition below PQ -/
  below_pq_composition : area_below_pq = 2 + (triangle_base * triangle_base / 12)
  /-- Assertion about the width of the dodecagon -/
  width_is_six : width = 6

/-- Theorem stating that for a dodecagon with given properties, XQ/QY = 2 -/
theorem dodecagon_ratio (d : Dodecagon) : ∃ (xq qy : ℝ), xq / qy = 2 ∧ xq + qy = d.width := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_ratio_l4163_416306


namespace NUMINAMATH_CALUDE_inscribed_square_area_l4163_416330

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 8 = 1

-- Define a square inscribed in the ellipse
def inscribed_square (s : ℝ) : Prop :=
  ∃ (x y : ℝ), ellipse x y ∧ s = 2 * x ∧ s = 2 * y

-- Theorem statement
theorem inscribed_square_area :
  ∃ (s : ℝ), inscribed_square s ∧ s^2 = 32/3 :=
sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l4163_416330
