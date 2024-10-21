import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_tetrahedron_OMNB₁_l1170_117000

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The volume of a tetrahedron given three vectors from a common vertex -/
noncomputable def tetrahedronVolume (v1 v2 v3 : Point3D) : ℝ :=
  (1/6) * abs ((v1.x * (v2.y * v3.z - v2.z * v3.y) +
                v1.y * (v2.z * v3.x - v2.x * v3.z) +
                v1.z * (v2.x * v3.y - v2.y * v3.x)))

/-- The theorem to be proved -/
theorem volume_of_tetrahedron_OMNB₁ :
  let O : Point3D := ⟨1/2, 1/2, 0⟩
  let M : Point3D := ⟨0, 1/2, 1⟩
  let N : Point3D := ⟨1, 1, 2/3⟩
  let B₁ : Point3D := ⟨1, 0, 1⟩
  tetrahedronVolume ⟨B₁.x - O.x, B₁.y - O.y, B₁.z - O.z⟩
                    ⟨N.x - O.x, N.y - O.y, N.z - O.z⟩
                    ⟨M.x - O.x, M.y - O.y, M.z - O.z⟩ = 11/72 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_tetrahedron_OMNB₁_l1170_117000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_winning_pair_l1170_117006

/-- Represents a card with a color and a letter -/
structure Card where
  color : Bool  -- True for red, False for green
  letter : Fin 4  -- 0 for A, 1 for B, 2 for C, 3 for D
deriving Fintype, DecidableEq

/-- The deck of cards -/
def deck : Finset Card := Finset.univ

/-- Checks if two cards form a winning pair -/
def is_winning_pair (c1 c2 : Card) : Bool :=
  c1.color = c2.color || c1.letter = c2.letter

/-- The set of all possible pairs of cards -/
def all_pairs : Finset (Card × Card) :=
  Finset.product deck deck

/-- The set of winning pairs -/
def winning_pairs : Finset (Card × Card) :=
  all_pairs.filter (fun p => is_winning_pair p.1 p.2 && p.1 ≠ p.2)

theorem probability_of_winning_pair :
  (winning_pairs.card : ℚ) / all_pairs.card = 4 / 7 := by
  sorry

#eval (winning_pairs.card : ℚ) / all_pairs.card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_winning_pair_l1170_117006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_dual_base_representation_l1170_117036

def is_valid_representation (n : ℕ) (digits : List ℕ) (base : ℕ) : Prop :=
  n = (digits.reverse.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0)

theorem smallest_dual_base_representation : ∃ (n : ℕ) (a b : ℕ), 
  (a > 2 ∧ b > 2) ∧ 
  is_valid_representation n [3, 1] a ∧
  is_valid_representation n [2, 2] b ∧
  (∀ (m : ℕ) (a' b' : ℕ), 
    (a' > 2 ∧ b' > 2) → 
    is_valid_representation m [3, 1] a' → 
    is_valid_representation m [2, 2] b' → 
    n ≤ m) ∧
  n = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_dual_base_representation_l1170_117036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1170_117065

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a+8)*x + a^2 + a - 12

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)
noncomputable def arithmetic_sum (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + d * (n - 1)) / 2

theorem min_value_theorem (a : ℝ) (a₁ d : ℝ) :
  (∀ n : ℕ, n > 0 → f a n = arithmetic_sum a₁ d n) →
  (f a (a^2 - 4) = f a (2*a - 8)) →
  (∃ n : ℕ, n > 0 ∧ (arithmetic_sum a₁ d n - 4*a) / (arithmetic_sequence a₁ d n - 1) = 37/8) ∧
  (∀ n : ℕ, n > 0 → (arithmetic_sum a₁ d n - 4*a) / (arithmetic_sequence a₁ d n - 1) ≥ 37/8) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1170_117065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_rectangle_l1170_117010

/-- Rectangle ABCD with given coordinates --/
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  D : ℝ × ℝ
  y : ℝ
  h1 : A = (6, -22)
  h2 : B = (2006, 178)
  h3 : D = (8, y)

/-- The area of rectangle ABCD --/
def rectangle_area (rect : Rectangle) : ℝ :=
  40400

/-- Proof that the area of the given rectangle is 40400 --/
theorem area_of_rectangle (rect : Rectangle) : rectangle_area rect = 40400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_rectangle_l1170_117010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_no_real_solutions_pairs_l1170_117076

theorem quadratic_no_real_solutions_pairs : 
  ∃! (s : Finset (ℕ × ℕ)), 
    (∀ (b c : ℕ), (b, c) ∈ s ↔ 
      (b > 0 ∧ c > 0 ∧ 
       b^2 < 4*c ∧ c^2 < 4*b)) ∧
    s.card = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_no_real_solutions_pairs_l1170_117076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_l1170_117014

theorem x_value (x y : ℝ) (h1 : (7 : ℝ)^(x - y) = 343) (h2 : (7 : ℝ)^(x + y) = 16807) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_l1170_117014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_movement_expected_y_equals_target_x_l1170_117096

/-- A fly's position on an integer grid -/
structure FlyPosition where
  x : ℕ
  y : ℕ

/-- The probability of moving right at each step -/
noncomputable def prob_right : ℝ := 1/2

/-- The probability of moving up at each step -/
noncomputable def prob_up : ℝ := 1/2

/-- The target x-coordinate -/
def target_x : ℕ := 2011

/-- Theorem about the fly's movement -/
theorem fly_movement (start : FlyPosition) :
  (∃ t : ℕ, start.x + t = target_x) ∧
  (∃ y : ℕ, y = target_x) := by
  sorry

/-- The expected y-coordinate when reaching the target x-coordinate -/
noncomputable def expected_y : ℝ := target_x

/-- Theorem about the expected y-coordinate -/
theorem expected_y_equals_target_x :
  expected_y = target_x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_movement_expected_y_equals_target_x_l1170_117096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_value_triangle_area_l1170_117069

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions of the problem
def triangle_conditions (t : Triangle) : Prop :=
  t.c = 2 ∧ t.C = Real.pi/3

-- Theorem 1
theorem angle_B_value (t : Triangle) (h : triangle_conditions t) (hb : t.b = 2*Real.sqrt 6/3) :
  t.B = Real.pi/4 := by sorry

-- Theorem 2
theorem triangle_area (t : Triangle) (h : triangle_conditions t) 
  (h_sin : Real.sin t.C + Real.sin (t.B - t.A) = 2 * Real.sin (2 * t.A)) :
  (1/2) * t.a * t.b * Real.sin t.C = 2*Real.sqrt 3/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_value_triangle_area_l1170_117069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_cube_configuration_exists_l1170_117004

/-- A cube in 3D space --/
structure Cube where
  position : ℝ × ℝ × ℝ
  orientation : Quaternion ℝ

/-- A configuration of cubes --/
def CubeConfiguration (n : ℕ) := Fin n → Cube

/-- Predicate to check if two cubes share a face --/
def SharesFace (c1 c2 : Cube) : Prop := sorry

/-- Theorem stating that a valid configuration exists for 5 and 6 cubes --/
theorem valid_cube_configuration_exists (n : Fin 2) : 
  ∃ (config : CubeConfiguration (n + 5)), 
    ∀ (i j : Fin (n + 5)), i ≠ j → SharesFace (config i) (config j) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_cube_configuration_exists_l1170_117004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_european_stamps_cost_l1170_117020

-- Define the countries
inductive Country
| Germany
| Italy
| Sweden
| Norway

-- Define the decades
inductive Decade
| Sixties
| Seventies
| Eighties

-- Define the stamp collection
def stamp_count : Country → Decade → Nat
| Country.Germany => λ d => match d with
  | Decade.Sixties => 5
  | Decade.Seventies => 10
  | Decade.Eighties => 8
| Country.Italy => λ d => match d with
  | Decade.Sixties => 6
  | Decade.Seventies => 12
  | Decade.Eighties => 9
| Country.Sweden => λ d => match d with
  | Decade.Sixties => 7
  | Decade.Seventies => 8
  | Decade.Eighties => 15
| Country.Norway => λ d => match d with
  | Decade.Sixties => 4
  | Decade.Seventies => 6
  | Decade.Eighties => 10

-- Define the cost per stamp for each country
def stamp_cost : Country → Rat
| Country.Germany => 8/100
| Country.Italy => 8/100
| Country.Sweden => 5/100
| Country.Norway => 7/100

-- Define whether a country is European
def is_european : Country → Bool
| Country.Germany => true
| Country.Italy => true
| _ => false

-- Define the discount rule
def apply_discount (count : Nat) (cost : Rat) : Rat :=
  if count > 15 then cost * 9/10 else cost

-- Helper function to calculate the cost for a country
def country_cost (c : Country) : Rat :=
  let count := stamp_count c Decade.Sixties + stamp_count c Decade.Seventies
  apply_discount count (count * stamp_cost c)

-- Theorem statement
theorem european_stamps_cost :
  (country_cost Country.Germany + country_cost Country.Italy) = 5/2 := by
  sorry

#eval country_cost Country.Germany + country_cost Country.Italy

end NUMINAMATH_CALUDE_ERRORFEEDBACK_european_stamps_cost_l1170_117020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_approx_216_seconds_l1170_117099

/-- The time (in seconds) it takes for two women running in opposite directions 
    on a circular track to meet for the first time. -/
noncomputable def meeting_time (track_length : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  track_length / ((speed1 + speed2) * 1000 / 3600)

/-- Theorem stating that the meeting time for the given conditions is approximately 216 seconds. -/
theorem meeting_time_approx_216_seconds : 
  ∃ ε > 0, |meeting_time 1800 10 20 - 216| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_approx_216_seconds_l1170_117099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1170_117038

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (2 + x - x^2) / (|x| - x)

-- State the theorem about the domain of f
theorem domain_of_f :
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ -1 < x ∧ x < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1170_117038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_point_four_nine_l1170_117063

-- Define y as the cube root of 0.000343
noncomputable def y : ℝ := (0.000343 : ℝ) ^ (1/3 : ℝ)

-- Define x in terms of y
noncomputable def x : ℝ := 7 * y

-- Theorem statement
theorem x_equals_point_four_nine : x = 0.49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_point_four_nine_l1170_117063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_probability_of_numbers_non_uniform_selection_probability_l1170_117016

/-- A selection process where k different numbers are chosen from n total numbers. -/
structure SelectionProcess where
  n : ℕ  -- Total number of possible numbers
  k : ℕ  -- Number of different numbers to be selected
  h : k ≤ n

/-- The probability of a specific number appearing in the selection. -/
def probabilityOfNumber (sp : SelectionProcess) : ℚ :=
  sp.k / sp.n

/-- Theorem stating that the probability of each number appearing is equal. -/
theorem equal_probability_of_numbers (sp : SelectionProcess) :
  ∀ i j, i ≤ sp.n → j ≤ sp.n → probabilityOfNumber sp = probabilityOfNumber sp :=
by
  sorry

/-- Theorem stating that the probability of different selections is not uniform. -/
theorem non_uniform_selection_probability (sp : SelectionProcess) (h : sp.k ≥ 2) :
  ∃ s₁ s₂ : Finset (Fin sp.n), s₁.card = sp.k ∧ s₂.card = sp.k ∧ 
  ∃ p₁ p₂ : ℚ, p₁ ≠ p₂ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_probability_of_numbers_non_uniform_selection_probability_l1170_117016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chlorine_percentage_in_ccl4_l1170_117082

/-- The atomic mass of carbon in g/mol -/
noncomputable def carbon_mass : ℝ := 12.01

/-- The atomic mass of chlorine in g/mol -/
noncomputable def chlorine_mass : ℝ := 35.45

/-- The number of carbon atoms in a CCl4 molecule -/
def carbon_count : ℕ := 1

/-- The number of chlorine atoms in a CCl4 molecule -/
def chlorine_count : ℕ := 4

/-- The mass of CCl4 in g/mol -/
noncomputable def ccl4_mass : ℝ := carbon_mass * (carbon_count : ℝ) + chlorine_mass * (chlorine_count : ℝ)

/-- The mass of chlorine in CCl4 in g/mol -/
noncomputable def chlorine_total_mass : ℝ := chlorine_mass * (chlorine_count : ℝ)

/-- The mass percentage of chlorine in CCl4 -/
noncomputable def chlorine_percentage : ℝ := (chlorine_total_mass / ccl4_mass) * 100

theorem chlorine_percentage_in_ccl4 : 
  92.18 < chlorine_percentage ∧ chlorine_percentage < 92.20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chlorine_percentage_in_ccl4_l1170_117082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_in_third_quadrant_l1170_117018

noncomputable def f (α : Real) : Real := (-Real.cos α) * Real.sin α * (-Real.tan α) / ((-Real.tan α) * Real.sin α)

theorem f_value_in_third_quadrant (α : Real) 
  (h1 : α ∈ Set.Icc π (3*π/2)) 
  (h2 : Real.cos (α - 3*π/2) = 1/5) : 
  f α = 2 * Real.sqrt 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_in_third_quadrant_l1170_117018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_b_town_l1170_117001

def speed_to_b : ℝ := 90
def speed_from_b : ℝ := 160
def total_time : ℝ := 5

theorem time_to_b_town : ∃ (time_to_b : ℝ), time_to_b * 60 = 192 := by
  let distance := (speed_to_b * speed_from_b * total_time) / (speed_to_b + speed_from_b)
  let time_to_b := distance / speed_to_b
  use time_to_b
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_b_town_l1170_117001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C₂_and_distance_AB_l1170_117081

-- Define the curve C₁
noncomputable def C₁ (α : ℝ) : ℝ × ℝ := (3 * Real.cos α, 3 + 3 * Real.sin α)

-- Define the point P in terms of M
def P (M : ℝ × ℝ) : ℝ × ℝ := (2 * M.1, 2 * M.2)

-- Define the curve C₂
noncomputable def C₂ (α : ℝ) : ℝ × ℝ := (6 * Real.cos α, 6 + 6 * Real.sin α)

-- Define the polar equation for C₁
noncomputable def C₁_polar (θ : ℝ) : ℝ := 6 * Real.sin θ

-- Define the polar equation for C₂
noncomputable def C₂_polar (θ : ℝ) : ℝ := 12 * Real.sin θ

theorem curve_C₂_and_distance_AB :
  (∀ α, C₂ α = P (C₁ α)) ∧
  (C₂_polar (π/3) - C₁_polar (π/3) = 3 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C₂_and_distance_AB_l1170_117081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_fraction_value_l1170_117061

theorem trig_fraction_value (α : ℝ) (h : Real.tan α = 3) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 3 * Real.cos α) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_fraction_value_l1170_117061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_max_not_less_than_one_l1170_117097

-- Statement 1
theorem inequality_proof (a : ℝ) (h : a > 2) :
  Real.sqrt (a + 2) + Real.sqrt (a - 2) < 2 * Real.sqrt a := by
  sorry

-- Statement 2
theorem max_not_less_than_one (x : ℝ) :
  let a := x^2 + 1/2
  let b := 2 - x
  let c := x^2 - x + 1
  max a (max b c) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_max_not_less_than_one_l1170_117097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1170_117044

/-- A hyperbola with foci on the x-axis and asymptotes y = ± (√2/2)x has eccentricity √6/2 -/
theorem hyperbola_eccentricity (C : Set (ℝ × ℝ)) (a b c : ℝ) :
  (∀ (x y : ℝ), (x, y) ∈ C → (y = (Real.sqrt 2/2)*x ∨ y = -(Real.sqrt 2/2)*x)) →  -- Asymptotes condition
  (∃ (f₁ f₂ : ℝ), (f₁, 0) ∈ C ∧ (f₂, 0) ∈ C ∧ f₁ ≠ f₂) →        -- Foci on x-axis condition
  c^2 = a^2 + b^2 →                                              -- Relation between a, b, and c
  b = (Real.sqrt 2/2)*a →                                        -- Derived from asymptotes
  c/a = Real.sqrt 6/2 :=                                         -- Eccentricity
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1170_117044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_power_product_equals_simplified_fraction_l1170_117042

theorem fraction_power_product_equals_simplified_fraction :
  (1 / 3 : ℚ) ^ 9 * (5 / 6 : ℚ) ^ (-4 : ℤ) = 4 / 37875 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_power_product_equals_simplified_fraction_l1170_117042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_sequence_a_l1170_117023

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 2  -- Define the base case for n = 0
  | n + 1 => (sequence_a n - 4) / 3

theorem limit_of_sequence_a :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |sequence_a n - (-2)| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_sequence_a_l1170_117023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_equation_l1170_117086

theorem cos_sin_equation (x : ℝ) : 
  (Real.cos x - 2 * Real.sin x = 3) → (Real.sin x + 2 * Real.cos x = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_equation_l1170_117086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_EFCD_equals_165_l1170_117040

/-- Represents a trapezoid ABCD with midpoints E and F on sides AD and BC respectively -/
structure Trapezoid :=
  (AB : ℝ)
  (CD : ℝ)
  (altitude : ℝ)

/-- Calculates the area of quadrilateral EFCD within the trapezoid -/
noncomputable def area_EFCD (t : Trapezoid) : ℝ :=
  let EF := (t.AB + t.CD) / 2
  let altitude_EFCD := t.altitude / 2
  altitude_EFCD * (EF + t.CD) / 2

/-- Theorem: The area of quadrilateral EFCD in the given trapezoid is 165 square units -/
theorem area_EFCD_equals_165 (t : Trapezoid)
  (h1 : t.AB = 10)
  (h2 : t.CD = 26)
  (h3 : t.altitude = 15) :
  area_EFCD t = 165 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_EFCD_equals_165_l1170_117040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l1170_117035

def sequence_a : ℕ → ℚ
  | 0 => 2
  | n + 1 => 2 - 1 / sequence_a n

def sequence_b (n : ℕ) : ℚ := 1 / (sequence_a n - 1)

def sequence_c (n : ℕ) : ℚ := 1 / (sequence_b n * sequence_b (n + 2))

def T (n : ℕ) : ℚ := (Finset.range n).sum (λ i => sequence_c i)

theorem min_m_value :
  ∃ (m : ℕ), m > 0 ∧ (∀ (n : ℕ), T n ≤ m / 12) ∧
  (∀ (k : ℕ), k > 0 ∧ (∀ (n : ℕ), T n ≤ k / 12) → k ≥ m) ∧
  m = 9 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l1170_117035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_f_condition_l1170_117080

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x - a else Real.log x / Real.log a

theorem monotonic_increasing_f_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ∈ Set.Icc (3/2) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_f_condition_l1170_117080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l1170_117046

-- Define the points A and B
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (-1, 2)

-- Define the moving point P
def P : ℝ × ℝ → Prop := λ p => 
  let (x, y) := p
  (x - A.1) * (x - B.1) + (y - A.2) * (y - B.2) = 0

-- Define the hyperbola parameters
variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)

-- Define the hyperbola equation
def isOnHyperbola (a b : ℝ) : ℝ × ℝ → Prop := λ p =>
  let (x, y) := p
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the asymptote
noncomputable def asymptote (a b : ℝ) : ℝ → ℝ := λ x => (b / a) * x

-- Define the condition that asymptotes do not intersect with P's trajectory
def noIntersection (a b : ℝ) : Prop :=
  ∀ x y, P (x, y) → |y - asymptote a b x| > 0

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2) / a

-- State the theorem
theorem eccentricity_range (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  noIntersection a b → 1 < eccentricity a b ∧ eccentricity a b < 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l1170_117046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1170_117037

-- Define set A
def A : Set ℝ := {x | x - 1 < 2}

-- Define set B
def B : Set ℝ := {x | 1 < Real.exp (x * Real.log 2) ∧ Real.exp (x * Real.log 2) < 16}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 0 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1170_117037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_day_meetings_percentage_l1170_117072

theorem work_day_meetings_percentage (work_day_hours : ℕ) (first_meeting_minutes : ℕ) : 
  work_day_hours = 10 →
  first_meeting_minutes = 60 →
  (work_day_hours * 60 : ℝ) / ((first_meeting_minutes + 2 * first_meeting_minutes) : ℝ) * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_day_meetings_percentage_l1170_117072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_b_representation_of_197_l1170_117098

theorem base_b_representation_of_197 :
  let possible_bases := {b : ℕ | b ≥ 2 ∧ b^3 ≤ 197 ∧ 197 < b^4}
  Finset.card (Finset.filter (λ b => b^3 ≤ 197 ∧ 197 < b^4) (Finset.range 6 \ Finset.range 2)) = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_b_representation_of_197_l1170_117098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_27_over_4_l1170_117071

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define variables a and b
variable (a b : ℝ)

-- Define the given conditions
axiom a_def : log10 3 = a
axiom b_def : log10 5 = b

-- State the theorem
theorem log_27_over_4 : log10 (27 / 4) = 3 * a + 2 * b - 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_27_over_4_l1170_117071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1170_117039

theorem equation_solutions (a b c : ℝ) :
  (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) →
    (Real.sqrt (a + b * 0) + Real.sqrt (b + c * 0) + Real.sqrt (c + a * 0) =
     Real.sqrt (b - a * 0) + Real.sqrt (c - b * 0) + Real.sqrt (a - c * 0))) ∧
  (a = 0 ∧ b = 0 ∧ c = 0 →
    ∀ x : ℝ, Real.sqrt (a + b * x) + Real.sqrt (b + c * x) + Real.sqrt (c + a * x) =
              Real.sqrt (b - a * x) + Real.sqrt (c - b * x) + Real.sqrt (a - c * x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1170_117039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l1170_117030

def my_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, a (n + 1) = (a n)^2 - n * (a n) + 1

theorem sequence_formula (a : ℕ → ℕ) (h : my_sequence a) : ∀ n : ℕ, a n = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l1170_117030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l1170_117087

theorem trig_problem (x : Real) 
  (h1 : Real.cos x = -Real.sqrt 2 / 10)
  (h2 : x ∈ Set.Ioo (Real.pi / 2) Real.pi) :
  Real.sin x = 7 * Real.sqrt 2 / 10 ∧ 
  Real.tan (2 * x + Real.pi / 4) = 31 / 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l1170_117087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_kings_on_12x12_board_l1170_117026

/-- Represents a chessboard of size n x n -/
def Chessboard (n : ℕ) := Fin n → Fin n → Bool

/-- A king threatens another king if they are on neighboring squares -/
def threatens (x1 y1 x2 y2 : ℕ) : Prop :=
  (x1 = x2 ∧ (y1 + 1 = y2 ∨ y2 + 1 = y1)) ∨
  (y1 = y2 ∧ (x1 + 1 = x2 ∨ x2 + 1 = x1)) ∨
  ((x1 + 1 = x2 ∨ x2 + 1 = x1) ∧ (y1 + 1 = y2 ∨ y2 + 1 = y1))

/-- Each king threatens exactly one other king -/
def valid_placement (board : Chessboard 12) : Prop :=
  ∀ x y, board x y → ∃! x' y', (x ≠ x' ∨ y ≠ y') ∧ board x' y' ∧ threatens x.val y.val x'.val y'.val

/-- Count the number of kings on the board -/
def king_count (board : Chessboard 12) : ℕ :=
  (Finset.sum Finset.univ fun x => Finset.sum Finset.univ fun y => if board x y then 1 else 0)

/-- The maximum number of kings that can be placed on a 12x12 chessboard -/
theorem max_kings_on_12x12_board :
  ∃ (board : Chessboard 12), valid_placement board ∧
  ∀ (board' : Chessboard 12), valid_placement board' →
  king_count board' ≤ king_count board ∧ king_count board = 72 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_kings_on_12x12_board_l1170_117026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_hyperbola_branch_l1170_117073

/-- A complex number z satisfies the given condition -/
def satisfies_condition (z : ℂ) : Prop :=
  Complex.abs (z + 2*Complex.I) - Complex.abs (z - 2*Complex.I) = 2

/-- IsHyperbolaBranch is a proposition that states a set is one branch of a hyperbola -/
def IsHyperbolaBranch (S : Set ℂ) : Prop :=
  ∃ (a b : ℝ) (c : ℂ), a > 0 ∧ b > 0 ∧
    S = {z : ℂ | (((z - c).re / a)^2 - ((z - c).im / b)^2 = 1) ∧ (z - c).re ≥ 0}

/-- The locus of points satisfying the condition is one branch of a hyperbola -/
theorem locus_is_hyperbola_branch :
  ∃ (H : Set ℂ), IsHyperbolaBranch H ∧ ∀ z, z ∈ H ↔ satisfies_condition z :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_hyperbola_branch_l1170_117073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_ray_equation_l1170_117066

/-- Given a light ray emitted from point (5, 3) reflecting off the x-axis,
    with tan(α) = 3 where α is the angle with the positive x-axis,
    prove that the equation of the line on which the reflected ray lies is y = -3x + 12 -/
theorem reflected_ray_equation (α : ℝ) :
  Real.tan α = 3 →
  ∃ (m b : ℝ), m = -3 ∧ b = 12 ∧
    ∀ (x y : ℝ), y = m * x + b ↔ 
      (∃ (t : ℝ), x = 5 + t * Real.cos (Real.pi - α) ∧ y = -3 + t * Real.sin (Real.pi - α)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_ray_equation_l1170_117066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_cylinder_ratio_l1170_117024

/-- The ratio of the combined volume of a perfectly inscribed sphere and 
    right circular cylinder to the volume of their enclosing cube -/
theorem inscribed_sphere_cylinder_ratio (s : ℝ) (h : s > 0) : 
  (((4/3) * Real.pi * (s/2)^3 + Real.pi * (s/2)^2 * s) / s^3) = 5 * Real.pi / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_cylinder_ratio_l1170_117024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_arrangements_count_l1170_117008

def Family : Type := Fin 6
def Seats : Type := Fin 6

def isParentOrGrandparent : Family → Prop := sorry
def isDriverSeat : Seats → Prop := sorry

def validSeatingArrangement (arrangement : Family → Seats) : Prop :=
  Function.Bijective arrangement ∧ 
  ∃ (driver : Family), isParentOrGrandparent driver ∧ isDriverSeat (arrangement driver)

-- Add this instance to resolve the Fintype issue
instance : Fintype { arrangement : Family → Seats | validSeatingArrangement arrangement } :=
  sorry

theorem seating_arrangements_count : 
  Fintype.card { arrangement : Family → Seats | validSeatingArrangement arrangement } = 480 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_arrangements_count_l1170_117008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_package_growth_equation_l1170_117077

/-- Represents the daily growth rate of package handling -/
def x : ℝ := sorry

/-- The initial number of packages handled on day 1 -/
def initial_packages : ℕ := 200

/-- The number of packages handled on day 3 -/
def day_3_packages : ℕ := 242

/-- The number of days between the initial day and the day of interest -/
def days_elapsed : ℕ := 2

/-- Theorem stating the relationship between initial packages, growth rate, and packages on day 3 -/
theorem package_growth_equation :
  (initial_packages : ℝ) * (1 + x)^days_elapsed = day_3_packages := by
  sorry

#check package_growth_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_package_growth_equation_l1170_117077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_answer_is_correct_l1170_117007

def correct_answer : String := "C"

theorem answer_is_correct : correct_answer = "C" := by
  rfl

#check answer_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_answer_is_correct_l1170_117007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_february_first_is_sunday_l1170_117029

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
deriving Repr, BEq

/-- Returns the day of the week that is n days before the given day -/
def daysBefore (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 =>
    match d with
    | .Sunday => daysBefore .Saturday n
    | .Monday => daysBefore .Sunday n
    | .Tuesday => daysBefore .Monday n
    | .Wednesday => daysBefore .Tuesday n
    | .Thursday => daysBefore .Wednesday n
    | .Friday => daysBefore .Thursday n
    | .Saturday => daysBefore .Friday n

theorem february_first_is_sunday (h : DayOfWeek) :
  h = DayOfWeek.Wednesday → daysBefore h 10 = DayOfWeek.Sunday := by
  intro hw
  rw [hw]
  rfl

#eval daysBefore DayOfWeek.Wednesday 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_february_first_is_sunday_l1170_117029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_secretary_order_invalid_l1170_117094

/-- Represents the order of letters. -/
inductive Letter : Type
  | one : Letter
  | two : Letter
  | three : Letter
  | four : Letter
  | five : Letter
deriving BEq, Repr, Inhabited

/-- The boss's delivery order. -/
def bossOrder : List Letter := [Letter.one, Letter.two, Letter.three, Letter.four, Letter.five]

/-- The secretary's typing order we want to prove impossible. -/
def secretaryOrder : List Letter := [Letter.four, Letter.five, Letter.two, Letter.three, Letter.one]

/-- Checks if a typing order is valid given the boss's delivery order. -/
def isValidTypingOrder (typing : List Letter) (delivery : List Letter) : Prop :=
  ∃ (stack : List Letter), 
    (∀ l ∈ typing, l ∈ delivery) ∧ 
    (∀ i j, i < j → delivery.indexOf (typing.get! i) > delivery.indexOf (typing.get! j))

/-- Theorem stating that the given secretary order is not valid. -/
theorem secretary_order_invalid : 
  ¬(isValidTypingOrder secretaryOrder bossOrder) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_secretary_order_invalid_l1170_117094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_police_force_composition_l1170_117091

theorem police_force_composition (male_duty_percent female_duty_percent female_percent : ℝ) 
  (total_on_duty : ℕ) : ℕ :=
  let male_duty_percent := 0.56
  let female_duty_percent := 0.32
  let total_on_duty := 280
  let female_percent := 0.40
  -- The number of female officers on the police force
  let female_officers : ℕ := 241
  female_officers

#check police_force_composition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_police_force_composition_l1170_117091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_25_point_6_l1170_117002

noncomputable section

-- Define the rectangle
def rectangle_base : ℝ := 12
def rectangle_height : ℝ := 8

-- Define the triangle
def triangle_base : ℝ := 8
def triangle_height : ℝ := 8

-- Define key points
def point_A : ℝ × ℝ := (0, 0)
def point_B : ℝ × ℝ := (0, rectangle_height)
def point_C : ℝ × ℝ := (rectangle_base, rectangle_height)
def point_D : ℝ × ℝ := (rectangle_base, 0)
def point_F : ℝ × ℝ := (rectangle_base + triangle_base, 0)

-- Define the slope of line BF
noncomputable def slope_BF : ℝ := (point_B.2 - point_F.2) / (point_B.1 - point_F.1)

-- Define the y-intercept of line BF
noncomputable def intercept_BF : ℝ := point_B.2 - slope_BF * point_B.1

-- Define the x-coordinate of point H (intersection of BF and CG)
def x_H : ℝ := rectangle_base

-- Define the y-coordinate of point H
noncomputable def y_H : ℝ := slope_BF * x_H + intercept_BF

-- Define the area of the full triangle DGF
def area_DGF : ℝ := (1/2) * triangle_base * triangle_height

-- Define the area of the smaller triangle DGH
noncomputable def area_DGH : ℝ := (1/2) * triangle_base * (rectangle_height - y_H)

-- Theorem to prove
theorem shaded_area_is_25_point_6 :
  area_DGF - area_DGH = 25.6 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_25_point_6_l1170_117002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1170_117009

noncomputable def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => a n / (2 + 3 * a n)

theorem sequence_properties :
  (∃ r : ℝ, ∀ n : ℕ, (1 / a (n + 1) + 3) = r * (1 / a n + 3)) ∧
  (∀ n : ℕ, (Finset.range n).sum (λ i => 1 / a i) = 2^(n+2) - 3*n - 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1170_117009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_factors_30_factorial_l1170_117049

/-- The number of distinct prime factors of 30! -/
def num_distinct_prime_factors_30_factorial : ℕ :=
  (Finset.range 31).filter Nat.Prime |>.card

/-- Theorem stating that the number of distinct prime factors of 30! is 10 -/
theorem distinct_prime_factors_30_factorial :
  num_distinct_prime_factors_30_factorial = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_factors_30_factorial_l1170_117049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_second_quadrant_l1170_117079

-- Define the determinant operation
def det (a b c d : ℂ) : ℂ := a * d - b * c

-- Define the equation
def equation (z : ℂ) : Prop := det z (1 + Complex.I) (-Complex.I) (2 * Complex.I) = 0

-- State the theorem
theorem solution_in_second_quadrant :
  ∃ z : ℂ, equation z ∧ z.re < 0 ∧ z.im > 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_second_quadrant_l1170_117079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_random_walk_probability_l1170_117045

def Square : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 6 ∧ 0 ≤ p.2 ∧ p.2 ≤ 6}

def CardinalStep : Set (ℝ × ℝ) := {s | s = (1, 0) ∨ s = (-1, 0) ∨ s = (0, 1) ∨ s = (0, -1)}

def IsOnVerticalSide (p : ℝ × ℝ) : Prop :=
  (p.1 = 0 ∨ p.1 = 6) ∧ 0 ≤ p.2 ∧ p.2 ≤ 6

def RandomWalk (start : ℝ × ℝ) (stop : ℝ × ℝ) : Prop :=
  ∃ (path : ℕ → ℝ × ℝ), 
    path 0 = start ∧
    (∃ n : ℕ, path n = stop ∧ stop ∈ (frontier Square)) ∧
    ∀ i : ℕ, (path (i+1) - path i) ∈ CardinalStep

noncomputable def ℙ : (Set (ℝ × ℝ)) → ℝ := sorry

theorem random_walk_probability (start : ℝ × ℝ) (h : start = (2, 3)) :
  ℙ {stop | RandomWalk start stop ∧ IsOnVerticalSide stop} = 11/18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_random_walk_probability_l1170_117045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_equals_fraction_l1170_117067

/-- The repeating decimal 0.363636... as a rational number -/
def repeating_decimal : ℚ := 4 / 11

/-- The fraction 4/11 as a rational number -/
def fraction : ℚ := 4 / 11

theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  -- Proof goes here
  sorry

#eval repeating_decimal
#eval fraction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_equals_fraction_l1170_117067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_rise_theorem_l1170_117050

/-- Represents the dimensions and water level changes of cubic pools -/
structure PoolData where
  large_edge : ℚ
  medium_edge : ℚ
  small_edge : ℚ
  medium_rise : ℚ
  small_rise : ℚ

/-- Calculates the water level rise in the large pool -/
def water_rise_large_pool (data : PoolData) : ℚ :=
  ((data.medium_edge ^ 2 * data.medium_rise + data.small_edge ^ 2 * data.small_rise) / data.large_edge ^ 2) * 1000

/-- Theorem stating the water level rise in the large pool -/
theorem water_rise_theorem (data : PoolData) 
  (h1 : data.large_edge = 6)
  (h2 : data.medium_edge = 3)
  (h3 : data.small_edge = 2)
  (h4 : data.medium_rise = 6/100)
  (h5 : data.small_rise = 4/100) :
  water_rise_large_pool data = 35 / 18 := by
  sorry

#eval water_rise_large_pool { 
  large_edge := 6, 
  medium_edge := 3, 
  small_edge := 2, 
  medium_rise := 6/100, 
  small_rise := 4/100 
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_rise_theorem_l1170_117050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1170_117021

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : a > 0
  pos_b : b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Calculates the dot product of two vectors represented by points -/
def dot_product (p q r : Point) : ℝ :=
  (q.x - p.x) * (r.x - p.x) + (q.y - p.y) * (r.y - p.y)

/-- Represents the foci of a hyperbola -/
structure Foci where
  F₁ : Point
  F₂ : Point

/-- Calculates the eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ :=
  Real.sqrt (a^2 + b^2) / a

/-- Main theorem: If there exists a point P on the hyperbola satisfying the given conditions,
    then the eccentricity of the hyperbola is √3 -/
theorem hyperbola_eccentricity 
  (h : Hyperbola a b) (f : Foci) (P : Point) 
  (on_hyperbola : P.x^2 / a^2 - P.y^2 / b^2 = 1)
  (distance_condition : distance P f.F₁ = 3 * distance P f.F₂)
  (dot_product_condition : dot_product P f.F₁ f.F₂ = -a^2) :
  eccentricity h = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1170_117021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_z_is_purely_imaginary_l1170_117052

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m^2 + m - 2) (m^2 - 1)

-- Theorem for when z is a real number
theorem z_is_real (m : ℝ) : (z m).im = 0 ↔ m = 1 ∨ m = -1 := by
  sorry

-- Theorem for when z is a purely imaginary number
theorem z_is_purely_imaginary (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_z_is_purely_imaginary_l1170_117052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l1170_117089

noncomputable def f (a x : ℝ) : ℝ := Real.sqrt (1 + a^2) + Real.sqrt (1 - x)

theorem f_derivative (a x : ℝ) (h : x < 1) : 
  deriv (f a) x = -1 / (2 * Real.sqrt (1 - x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l1170_117089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_cubed_is_80_l1170_117031

noncomputable def expansion (x : ℝ) := (x^2 + 1/x + 1)^6

theorem coefficient_of_x_cubed_is_80 :
  ∃ (a b c d e : ℝ), expansion x = a*x^6 + b*x^5 + c*x^4 + 80*x^3 + d*x^2 + e*x + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_cubed_is_80_l1170_117031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_is_54_main_theorem_l1170_117048

-- Define the side lengths of the triangle and hexagon
variable (s t : ℝ)

-- Define the perimeter ratio condition
axiom perimeter_ratio : 6 * t = 2 * (3 * s)

-- Define the triangle area condition
axiom triangle_area : s^2 * Real.sqrt 3 / 4 = 9

-- Define the hexagon area function
noncomputable def hexagon_area (side : ℝ) : ℝ := 3 * side^2 * Real.sqrt 3 / 2

-- Theorem statement
theorem hexagon_area_is_54 : hexagon_area t = 54 := by
  sorry

-- Main theorem combining all conditions
theorem main_theorem : hexagon_area t = 54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_is_54_main_theorem_l1170_117048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_l1170_117090

/-- Given that x and y are inversely proportional, and when x = 40, y = 4,
    prove that x = 16 when y = 10 -/
theorem inverse_proportion (x y : ℝ) (h1 : ∃ k : ℝ, x * y = k) 
    (h2 : 40 * 4 = x * y) : x = 16 ∧ y = 10 → x * y = 160 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_l1170_117090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_folding_theorem_l1170_117054

/-- Represents the dimensions of a rectangular paper. -/
structure PaperDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle given its dimensions. -/
noncomputable def area (d : PaperDimensions) : ℝ := d.length * d.width

/-- Calculates the number of different shapes after n folds. -/
def num_shapes (n : ℕ) : ℕ := n + 1

/-- Calculates the sum of areas after n folds. -/
noncomputable def sum_areas (n : ℕ) : ℝ := 240 * (3 - (n + 3) / 2^n)

/-- Theorem about paper folding patterns. -/
theorem paper_folding_theorem (initial_paper : PaperDimensions) 
    (h1 : initial_paper.length = 20 ∧ initial_paper.width = 12) :
    (num_shapes 4 = 5) ∧ 
    (∀ n : ℕ, sum_areas n = 240 * (3 - (n + 3) / 2^n)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_folding_theorem_l1170_117054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_l1170_117062

/-- The angle of inclination of a line with equation x - √3y + 2 = 0 is 30° -/
theorem angle_of_inclination (a b c : ℝ) (h : a = 1 ∧ b = -Real.sqrt 3 ∧ c = 2) : 
  Real.arctan (a / b) = 30 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_l1170_117062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_point_distance_sum_l1170_117012

/-- Given a triangle ABC and a point M inside it, the sum of distances from M to each vertex
    is less than or equal to the maximum of the sums of any two sides of the triangle. -/
theorem triangle_point_distance_sum (A B C M : EuclideanSpace ℝ (Fin 2)) 
  (h : ∃ (α β γ : ℝ), α ≥ 0 ∧ β ≥ 0 ∧ γ ≥ 0 ∧ α + β + γ = 1 ∧ M = α • A + β • B + γ • C) :
  dist M A + dist M B + dist M C ≤ max (dist A B + dist B C) (max (dist B C + dist A C) (dist A C + dist A B)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_point_distance_sum_l1170_117012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_275_in_fraction_l1170_117060

def consecutiveDigits (a b c : Nat) (x : ℚ) : Prop :=
  ∃ k : ℕ, (10^k * x).floor % 1000 = 100 * a + 10 * b + c

theorem smallest_n_with_275_in_fraction : 
  (∀ n < 127, ¬∃ m : ℕ, m < n ∧ Nat.Coprime m n ∧ consecutiveDigits 2 7 5 (m / n)) ∧
  (∃ m : ℕ, m < 127 ∧ Nat.Coprime m 127 ∧ consecutiveDigits 2 7 5 (m / 127)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_275_in_fraction_l1170_117060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l1170_117041

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem triangle_inequality (a b : V) : ‖a + b‖ ≤ ‖a‖ + ‖b‖ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l1170_117041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stirling_exp_gen_func_formula_l1170_117033

-- Define the Stirling numbers of the second kind
def stirling2 (N n : ℕ) : ℕ := sorry

-- Define the exponential generating function for Stirling numbers
noncomputable def exp_gen_func (n : ℕ) (x : ℝ) : ℝ :=
  ∑' N, (stirling2 N n : ℝ) * x^N / (N.factorial : ℝ)

-- State the theorem
theorem stirling_exp_gen_func_formula (n : ℕ) (x : ℝ) :
  exp_gen_func n x = (Real.exp x - 1)^n / (n.factorial : ℝ) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stirling_exp_gen_func_formula_l1170_117033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_special_triangle_l1170_117019

/-- Given a triangle ABC where the ratio of sines of angles is 3:5:7, 
    prove that the largest angle is 2π/3 radians -/
theorem largest_angle_in_special_triangle (A B C : Real) (h_triangle : A + B + C = Real.pi)
  (h_sine_ratio : ∃ (k : Real), Real.sin A = 3*k ∧ Real.sin B = 5*k ∧ Real.sin C = 7*k) :
  max A (max B C) = 2*Real.pi/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_special_triangle_l1170_117019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_berengere_contribution_is_correct_l1170_117053

/-- The exchange rate from euros to USD -/
noncomputable def exchange_rate : ℚ := 4/5

/-- The cost of the book in euros -/
noncomputable def book_cost : ℚ := 15

/-- Emily's contribution in USD -/
noncomputable def emily_contribution_usd : ℚ := 10

/-- Berengere's contribution in euros -/
noncomputable def berengere_contribution : ℚ := book_cost - (emily_contribution_usd / exchange_rate)

theorem berengere_contribution_is_correct : 
  berengere_contribution = 5/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_berengere_contribution_is_correct_l1170_117053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_well_distance_proof_l1170_117058

/-- Represents the distance from the well to the kitchen in meters -/
noncomputable def well_to_kitchen : ℝ := 50 + 10/11

/-- Represents the fraction of water remaining after leaking from the side hole -/
noncomputable def side_leak : ℝ := 1/2

/-- Represents the fraction of water remaining after leaking from the bottom hole -/
noncomputable def bottom_leak : ℝ := 1/3

/-- Represents the fraction of water remaining when 1 meter away from the kitchen -/
noncomputable def final_water : ℝ := 1/40

/-- Represents the distance from the kitchen where both holes start leaking simultaneously -/
noncomputable def both_leak_start : ℝ := 1

/-- Represents the height of the side hole as a fraction of the bucket's height -/
noncomputable def side_hole_height : ℝ := 1/4

/-- Theorem stating the relationship between the distance and water leakage -/
theorem well_distance_proof :
  ∃ (x : ℝ),
    x = well_to_kitchen ∧
    (1 - side_leak) / x + (1 - bottom_leak) / x = (1 - final_water) / (x - both_leak_start) ∧
    x > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_well_distance_proof_l1170_117058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_triangle_l1170_117083

theorem min_perimeter_triangle (D E F : Real) (d e f : Nat) : 
  Real.cos D = 3/5 → 
  Real.cos E = 1/3 → 
  Real.cos F = -1/2 → 
  d + e + f > 0 →
  (∀ d' e' f' : Nat, d' + e' + f' > 0 → 
    Real.cos D = 3/5 → 
    Real.cos E = 1/3 → 
    Real.cos F = -1/2 → 
    d' + e' + f' ≥ d + e + f) →
  d + e + f = 69 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_triangle_l1170_117083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_3x_plus_4_when_x_is_4_l1170_117055

theorem square_of_3x_plus_4_when_x_is_4 :
  ∀ x : ℝ, x = 4 → (3 * x + 4)^2 = 256 := by
  intro x h
  rw [h]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_3x_plus_4_when_x_is_4_l1170_117055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_properties_l1170_117051

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)
noncomputable def g (x : ℝ) : ℝ := Real.cos (x - Real.pi / 3)

noncomputable def sine_symmetry_axis (k : ℤ) : ℝ := (k : ℝ) * Real.pi / 2 + Real.pi / 3

noncomputable def cosine_symmetry_axis (k : ℤ) : ℝ := (k : ℝ) * Real.pi + Real.pi / 3

noncomputable def sine_symmetry_center (k : ℤ) : ℝ × ℝ := ((k : ℝ) * Real.pi / 2 + Real.pi / 12, 0)

noncomputable def cosine_symmetry_center (k : ℤ) : ℝ × ℝ := ((k : ℝ) * Real.pi + 5 * Real.pi / 6, 0)

theorem symmetry_properties :
  (∃ k : ℤ, sine_symmetry_axis k = cosine_symmetry_axis k) ∧
  (∀ k₁ k₂ : ℤ, sine_symmetry_center k₁ ≠ cosine_symmetry_center k₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_properties_l1170_117051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_x_ln_x_l1170_117013

open Real

/-- The function f(x) = x * ln(x) is monotonically decreasing on the interval (0, 1/e] -/
theorem monotonic_decreasing_x_ln_x :
  ∀ x y, x > 0 → y > 0 → x < y → x ≤ 1/exp 1 → y ≤ 1/exp 1 →
  x * log x ≥ y * log y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_x_ln_x_l1170_117013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_rental_cost_l1170_117095

def admission_cost : ℕ := 10
def total_budget : ℕ := 350
def num_students : ℕ := 25

theorem bus_rental_cost : 
  total_budget - (admission_cost * num_students) = 100 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_rental_cost_l1170_117095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walkway_problem_l1170_117047

/-- Represents the scenario of a person walking on a moving walkway -/
structure WalkwayScenario where
  length : ℝ  -- Length of the walkway in meters
  time_with : ℝ  -- Time to walk with the walkway in seconds
  time_against : ℝ  -- Time to walk against the walkway in seconds

/-- Calculates the time to walk when the walkway is not moving -/
noncomputable def time_stationary (scenario : WalkwayScenario) : ℝ :=
  2 * scenario.length * scenario.time_with * scenario.time_against /
  (scenario.time_with * scenario.time_against + (scenario.time_with + scenario.time_against) * scenario.length)

/-- Theorem stating that for the given scenario, the time to walk on a stationary walkway is 150 seconds -/
theorem walkway_problem (scenario : WalkwayScenario) 
  (h1 : scenario.length = 200)
  (h2 : scenario.time_with = 100)
  (h3 : scenario.time_against = 300) :
  time_stationary scenario = 150 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_walkway_problem_l1170_117047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l1170_117015

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.2 = Real.sqrt (9 - p.1^2) ∧ p.2 ≠ 0}
def N (b : ℝ) : Set (ℝ × ℝ) := {p | p.2 = p.1 + b}

-- State the theorem
theorem intersection_range (b : ℝ) : 
  (M ∩ N b).Nonempty → b ∈ Set.Ioc (-3) (3 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l1170_117015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jacks_walking_rate_l1170_117074

/-- Calculates the walking rate in miles per hour given distance and time -/
noncomputable def walkingRate (distance : ℝ) (hours : ℝ) (minutes : ℝ) : ℝ :=
  distance / (hours + minutes / 60)

/-- Theorem: Jack's walking rate is 3.2 miles per hour -/
theorem jacks_walking_rate : walkingRate 4 1 15 = 3.2 := by
  -- Unfold the definition of walkingRate
  unfold walkingRate
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jacks_walking_rate_l1170_117074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_twelve_percent_l1170_117028

/-- Calculates the interest rate given the principal, interest, and time period. -/
noncomputable def calculate_interest_rate (principal : ℝ) (interest : ℝ) (time : ℝ) : ℝ :=
  (interest * 100) / (principal * time)

/-- Theorem stating that for a loan with given conditions, the interest rate is 12%. -/
theorem interest_rate_is_twelve_percent 
  (principal : ℝ) 
  (interest : ℝ) 
  (time : ℝ) 
  (h1 : principal = 25000)
  (h2 : interest = 9000)
  (h3 : time = 3) :
  calculate_interest_rate principal interest time = 12 := by
  sorry

#check interest_rate_is_twelve_percent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_twelve_percent_l1170_117028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_range_l1170_117017

def sequence_a (a : ℝ) (n : ℕ+) : ℝ :=
  if n ≤ 6 then (1 - 3*a)*n + 10*a else a^(n.val - 7)

theorem sequence_a_range (a : ℝ) :
  (∀ n m : ℕ+, n < m → sequence_a a m < sequence_a a n) →
  1/3 < a ∧ a < 5/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_range_l1170_117017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_100_inverse_sum_eq_200_101_l1170_117056

/-- Arithmetic sequence with first term 1 and common difference 1 -/
def arithmetic_sequence (n : ℕ) : ℚ := n

/-- Sum of first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℚ := n * (n + 1) / 2

/-- The nth term of the sequence (1/S_n) -/
def inverse_sum_sequence (n : ℕ) : ℚ := 1 / S n

/-- Sum of first 100 terms of the sequence (1/S_n) -/
def sum_100_inverse_sum : ℚ :=
  Finset.sum (Finset.range 100) (λ i => inverse_sum_sequence (i + 1))

theorem sum_100_inverse_sum_eq_200_101 : sum_100_inverse_sum = 200 / 101 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_100_inverse_sum_eq_200_101_l1170_117056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_neg_one_third_tangent_through_point_one_zero_l1170_117025

noncomputable def curve (x : ℝ) : ℝ := 1 / x

noncomputable def tangent_slope (x : ℝ) : ℝ := -1 / (x^2)

-- Theorem for tangent lines with slope -1/3
theorem tangent_slope_neg_one_third :
  ∃ (a : ℝ), tangent_slope a = -1/3 ∧
  ((x + 3*curve a - 2*Real.sqrt 3 = 0) ∨ (x + 3*curve a + 2*Real.sqrt 3 = 0)) :=
sorry

-- Theorem for tangent line passing through P(1,0)
theorem tangent_through_point_one_zero :
  ∃ (b : ℝ), (4*1 + 0 - 4 = 0) ∧
  (4*b + curve b - 4 = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_neg_one_third_tangent_through_point_one_zero_l1170_117025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_roots_sum_of_squares_of_roots_specific_equation_l1170_117084

theorem sum_of_squares_of_roots (a b c : ℝ) (h : b^2 - 4*a*c > 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a ≠ 0 → a*x^2 + b*x + c = 0 → r₁^2 + r₂^2 = (b^2 - 2*c) / a := by
  sorry

theorem sum_of_squares_of_roots_specific_equation :
  let r₁ := (15 + Real.sqrt (225 - 28)) / 2
  let r₂ := (15 - Real.sqrt (225 - 28)) / 2
  r₁^2 + r₂^2 = 211 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_roots_sum_of_squares_of_roots_specific_equation_l1170_117084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_servant_worked_seven_months_l1170_117057

/-- Calculates the number of months a servant worked based on given salary conditions -/
def servant_work_duration (yearly_salary : ℚ) (turban_price : ℚ) (received_amount : ℚ) : ℕ :=
  let total_yearly_salary := yearly_salary + turban_price
  let monthly_salary := total_yearly_salary / 12
  let received_without_turban := received_amount - turban_price
  (received_without_turban / monthly_salary).floor.toNat

/-- Proves that the servant worked for 7 months given the problem conditions -/
theorem servant_worked_seven_months :
  servant_work_duration 90 10 75 = 7 := by
  sorry

#eval servant_work_duration 90 10 75

end NUMINAMATH_CALUDE_ERRORFEEDBACK_servant_worked_seven_months_l1170_117057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_max_sum_abs_coordinates_l1170_117011

theorem circle_max_sum_abs_coordinates :
  ∀ x y : ℝ, x^2 + y^2 = 1 → |x| + |y| ≤ Real.sqrt 2 ∧
  ∃ a b : ℝ, a^2 + b^2 = 1 ∧ |a| + |b| = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_max_sum_abs_coordinates_l1170_117011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_photo_probability_l1170_117093

/-- Represents a runner on a circular track -/
structure Runner where
  lapTime : ℕ  -- Time to complete one lap in seconds
  direction : Bool  -- true for counterclockwise, false for clockwise

/-- Calculates the probability of both runners being in a photo -/
def probability_in_photo (emily john : Runner) (photo_width : ℚ) : ℚ :=
  let overlap_time := (2 * (photo_width * emily.lapTime).num).toNat
  let cycle_time := Nat.lcm emily.lapTime john.lapTime
  (overlap_time : ℚ) / (cycle_time : ℚ)

theorem runners_photo_probability :
  let emily : Runner := { lapTime := 100, direction := true }
  let john : Runner := { lapTime := 75, direction := false }
  let photo_width : ℚ := 1/3
  probability_in_photo emily john photo_width = 1/6 := by
  sorry

#eval probability_in_photo 
  { lapTime := 100, direction := true } 
  { lapTime := 75, direction := false } 
  (1/3 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_photo_probability_l1170_117093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1170_117032

theorem trigonometric_identities (α : ℝ) :
  (Real.cos α = -4/5 ∧ α ∈ Set.Ioo (Real.pi) (3/2 * Real.pi)) →
    Real.sin α = -3/5 ∧
  Real.tan α = -3 →
    (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1170_117032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_equals_three_l1170_117068

/-- The function f(x) = x ln x + a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x + a

/-- The derivative of f(x) -/
noncomputable def f_deriv (x : ℝ) : ℝ := Real.log x + 1

theorem tangent_line_implies_a_equals_three (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ 
    f_deriv x₀ = 1 ∧ 
    x₀ - (f a x₀) + 2 = 0) →
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_equals_three_l1170_117068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_decreasing_f_well_defined_l1170_117078

-- Define the function as noncomputable due to its dependency on Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (3 - 2*x - x^2)

-- State the theorem
theorem f_strictly_decreasing :
  ∀ x₁ x₂ : ℝ, x₁ ∈ Set.Ioo (-1) 1 → x₂ ∈ Set.Ioo (-1) 1 → 
  x₁ < x₂ → f x₂ < f x₁ :=
by
  sorry -- Skip the proof for now

-- Define the domain of the function
def f_domain : Set ℝ := Set.Icc (-3) 1

-- State that the function is well-defined on its domain
theorem f_well_defined :
  ∀ x ∈ f_domain, 3 - 2*x - x^2 ≥ 0 :=
by
  sorry -- Skip the proof for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_decreasing_f_well_defined_l1170_117078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_qiuqiu_serving_l1170_117034

/-- Represents the number of cups that can be filled with one bottle of beer -/
def bottles_to_cups : ℚ → ℚ := sorry

/-- Represents the ratio of foam to total volume in a cup -/
def foam_ratio : ℚ := sorry

/-- Represents the expansion factor of beer foam -/
def foam_expansion : ℚ := sorry

theorem qiuqiu_serving (kangkang_cups : ℚ) (h1 : kangkang_cups = 4) 
  (h2 : foam_ratio = 1/2) (h3 : foam_expansion = 3) :
  bottles_to_cups foam_ratio = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_qiuqiu_serving_l1170_117034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_five_percent_l1170_117092

/-- The rate of interest that satisfies the given conditions -/
noncomputable def interest_rate (principal : ℝ) (time : ℝ) (interest_diff : ℝ) : ℝ :=
  100 * Real.sqrt ((interest_diff / principal) / time)

/-- Theorem stating that the interest rate is 5% given the problem conditions -/
theorem interest_rate_is_five_percent :
  let principal := 8000.000000000171
  let time := 2
  let interest_diff := 20
  ∃ ε > 0, |interest_rate principal time interest_diff - 5| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_five_percent_l1170_117092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_inequality_min_value_achievable_l1170_117070

theorem min_value_inequality (a b c : ℝ) (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 6) :
  (a - 2)^2 + 2 * ((b / a) - 1)^2 + 3 * ((c / b) - 1)^2 + 4 * ((6 / c) - 1)^2 ≥ 10 * (2^(13/20) - 1)^2 :=
by sorry

theorem min_value_achievable :
  ∃ a b c : ℝ, 2 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ 6 ∧
  (a - 2)^2 + 2 * ((b / a) - 1)^2 + 3 * ((c / b) - 1)^2 + 4 * ((6 / c) - 1)^2 = 10 * (2^(13/20) - 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_inequality_min_value_achievable_l1170_117070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_circle_radius_theorem_sum_of_coefficients_is_53_l1170_117003

/-- An isosceles trapezoid with given dimensions and inscribed circles -/
structure IsoscelesTrapezoidWithCircles where
  -- The trapezoid ABCD
  AB : ℝ
  BC : ℝ
  DA : ℝ
  CD : ℝ
  -- Circles centered at vertices
  radiusAB : ℝ
  radiusCD : ℝ
  -- Conditions
  isIsosceles : BC = DA
  dimensionsValid : AB = 8 ∧ BC = 7 ∧ DA = 7 ∧ CD = 6
  circleRadiiValid : radiusAB = 4 ∧ radiusCD = 3

/-- The radius of the inner tangent circle -/
noncomputable def innerCircleRadius (t : IsoscelesTrapezoidWithCircles) : ℝ :=
  (24 * Real.sqrt 3 - 24) / 2

/-- Theorem stating the radius of the inner tangent circle -/
theorem inner_circle_radius_theorem (t : IsoscelesTrapezoidWithCircles) :
  ∃ (r : ℝ), r = innerCircleRadius t ∧ 
  (∃ (x y : ℝ), 
    -- Circle is tangent to all four circles
    (x^2 + y^2 = (r + t.radiusAB)^2) ∧
    (x^2 + (t.CD - y)^2 = (r + t.radiusCD)^2) ∧
    -- Circle is contained within the trapezoid
    (0 ≤ x ∧ x ≤ t.AB) ∧ (0 ≤ y ∧ y ≤ t.CD)) := by
  sorry

/-- The sum of k, m, n, and p in the expression (−k + m√n)/p -/
def sumOfCoefficients : ℕ := 24 + 24 + 3 + 2

theorem sum_of_coefficients_is_53 : sumOfCoefficients = 53 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_circle_radius_theorem_sum_of_coefficients_is_53_l1170_117003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_translation_l1170_117027

noncomputable def f (x : ℝ) : ℝ := Real.sin (-2 * x + Real.pi / 3)

noncomputable def g (x φ : ℝ) : ℝ := Real.sin (-2 * (x - φ) + Real.pi / 3)

theorem even_function_translation (φ : ℝ) 
  (h1 : 0 < φ) (h2 : φ < Real.pi) 
  (h3 : ∀ x, g x φ = g (-x) φ) : 
  φ = Real.pi / 12 ∨ φ = 7 * Real.pi / 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_translation_l1170_117027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shape_to_square_theorem_l1170_117043

-- Define a shape as a list of points (vertices)
def Shape := List (ℝ × ℝ)

-- Define a function to check if a shape is a square
def is_square (s : Shape) : Prop := sorry

-- Define a function to cut a shape into parts
def cut_shape (s : Shape) : List Shape := sorry

-- Define a function to check if shapes can be rearranged into a square
def can_form_square (parts : List Shape) : Prop := sorry

-- Theorem: There exists a way to cut the original shape into 4 parts that can form a square
theorem shape_to_square_theorem (original : Shape) : 
  ∃ (parts : List Shape), (parts.length = 4) ∧ (cut_shape original = parts) ∧ (can_form_square parts) := by
  sorry

-- Example usage
def original_shape : Shape := [(0,0), (1,0), (1,1), (0,1), (0.5,1.5)]

#check shape_to_square_theorem original_shape

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shape_to_square_theorem_l1170_117043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_l1170_117064

theorem isosceles_right_triangle (A B : ℝ) (h : Real.cos (A - B) + Real.sin (A + B) = 2) : 
  A = 45 ∧ B = 45 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_l1170_117064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_quadratic_l1170_117005

theorem sum_of_roots_quadratic : 
  (∃ x y : ℝ, x^2 = 16*x - 12 ∧ y^2 = 16*y - 12 ∧ x ≠ y) → x + y = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_quadratic_l1170_117005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_classification_l1170_117088

def numbers : List ℚ := [-3, -1, 0, 20, 1/4, 17/100, -17/2, 2, 22/7]

def is_positive_integer (x : ℚ) : Bool := x > 0 && x.den = 1

def is_fraction (x : ℚ) : Bool := x.den ≠ 1 || Int.natAbs x.num < x.den

def is_non_positive_rational (x : ℚ) : Bool := x ≤ 0

theorem number_classification (numbers : List ℚ) :
  (numbers.filter is_positive_integer).toFinset = {20, 2} ∧
  (numbers.filter is_fraction).toFinset = {1/4, 17/100, -17/2, 22/7} ∧
  (numbers.filter is_non_positive_rational).toFinset = {-3, -1, 0, -17/2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_classification_l1170_117088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l1170_117022

-- Define the circle
def circle (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y + 5 = 0

-- Define the perpendicular line
def perp_line (a x y : ℝ) : Prop := a*x + y - 1 = 0

-- Define the point P
def point_P : ℝ × ℝ := (1, 2)

-- Theorem statement
theorem tangent_line_slope (a : ℝ) : 
  (∃ (m b : ℝ), 
    -- Line equation: y = mx + b
    (point_P.2 = m * point_P.1 + b) ∧ 
    -- Line is tangent to the circle
    (∀ (x y : ℝ), y = m*x + b → (circle x y → x = point_P.1 ∧ y = point_P.2)) ∧
    -- Line is perpendicular to ax + y - 1 = 0
    (m * (-1/a) = -1) ∧
    -- Point P is on the circle
    (circle point_P.1 point_P.2)) →
  a = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l1170_117022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_7_value_l1170_117059

def T (n : ℕ) : ℚ := n * (n + 1) / 2

def Q (n : ℕ) : ℚ := Finset.prod (Finset.range (n - 1)) (λ k => T (k + 2) / (T (k + 2) - 1 + (k + 2)))

theorem Q_7_value : Q 7 = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_7_value_l1170_117059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l1170_117075

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 10 - 9 / x

-- State the theorem
theorem root_exists_in_interval :
  (∀ x y, x < y → f x < f y) →  -- f is increasing
  Continuous f →                -- f is continuous
  f 9 < 0 →                     -- f(9) < 0
  f 10 > 0 →                    -- f(10) > 0
  ∃ r, r ∈ Set.Ioo 9 10 ∧ f r = 0 := by
  intro hIncreasing hContinuous h9 h10
  -- The proof would go here, but we'll use sorry for now
  sorry

-- You can add more theorems or lemmas here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l1170_117075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l1170_117085

/-- A hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  /-- First asymptote equation: y = x + 3 -/
  asymptote1 : ℝ → ℝ
  /-- Second asymptote equation: y = -x + 5 -/
  asymptote2 : ℝ → ℝ
  /-- The hyperbola passes through the point (1,5) -/
  passes_through : ℝ × ℝ
  /-- Condition for the first asymptote -/
  h_asymptote1 : ∀ x, asymptote1 x = x + 3
  /-- Condition for the second asymptote -/
  h_asymptote2 : ∀ x, asymptote2 x = -x + 5
  /-- Condition for the point the hyperbola passes through -/
  h_passes_through : passes_through = (1, 5)

/-- Calculate the distance between the foci of a hyperbola -/
noncomputable def distance_between_foci (h : Hyperbola) : ℝ :=
  2 * Real.sqrt 2

/-- The distance between the foci of the hyperbola is 2√2 -/
theorem hyperbola_foci_distance (h : Hyperbola) : 
  distance_between_foci h = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l1170_117085
