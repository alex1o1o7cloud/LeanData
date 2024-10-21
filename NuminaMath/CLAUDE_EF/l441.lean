import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_log_function_l441_44166

-- Define the function f(x) = lg(x - 1)
noncomputable def f (x : ℝ) : ℝ := Real.log (x - 1)

-- Define the domain of f
def domain_of_f : Set ℝ := {x | x > 1}

-- Theorem statement
theorem domain_of_log_function :
  {x : ℝ | ∃ y, f x = y} = domain_of_f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_log_function_l441_44166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_solution_l441_44181

-- Define the nested radical function
noncomputable def nestedRadical (x : ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => Real.sqrt (x + nestedRadical x n)

-- State the theorem
theorem nested_radical_solution :
  ∀ x y : ℤ, (nestedRadical (x : ℝ) 1964 : ℝ) = y → x = 0 ∧ y = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_solution_l441_44181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l441_44158

/-- A line passing through (4, -3) and tangent to (x+1)^2 + (y+2)^2 = 25 has equation x = 4 or 12x - 5y - 63 = 0 -/
theorem tangent_line_equation (l : Set (ℝ × ℝ)) (P : ℝ × ℝ) (C : Set (ℝ × ℝ)) :
  P = (4, -3) →
  ∃ (center : ℝ × ℝ) (radius : ℝ), center = (-1, -2) ∧ radius = 5 ∧
    C = {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2} →
  P ∈ l →
  ∃ (p : ℝ × ℝ), p ∈ l ∩ C ∧ ∀ q ∈ l, q ∉ C ∨ q = p →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ x = 4) ∨
  (∀ (x y : ℝ), (x, y) ∈ l ↔ 12*x - 5*y - 63 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l441_44158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_distribution_l441_44121

def number_of_distributions (n m : ℕ) : ℕ :=
  -- The number of ways to distribute n identical tickets among m delegates,
  -- with each delegate receiving at most one ticket and all tickets being distributed
  sorry

theorem ticket_distribution (n m : ℕ) :
  n = 4 ∧ m = 5 →
  number_of_distributions n m = m :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_distribution_l441_44121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_use_crying_over_spilt_milk_l441_44157

/-- A theorem representing the futility of lamenting past events -/
theorem no_use_crying_over_spilt_milk (P : Prop) (H : ¬P) : P → False := by
  intro h
  exact H h

#check no_use_crying_over_spilt_milk

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_use_crying_over_spilt_milk_l441_44157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameters_satisfy_equation_l441_44130

/-- A line parameterized by t with parameters s and m -/
structure ParameterizedLine where
  s : ℚ
  m : ℚ

/-- The equation of the line y = 5x + 7 -/
def lineEquation (x y : ℚ) : Prop := y = 5 * x + 7

/-- The parameterized form of the line -/
def parameterizedForm (l : ParameterizedLine) (t : ℚ) : ℚ × ℚ :=
  (l.s + 2 * t, -4 + l.m * t)

/-- Theorem stating that the parameters s and m satisfy the line equation -/
theorem line_parameters_satisfy_equation (l : ParameterizedLine) : 
  (l.s = -11/5 ∧ l.m = 10) ↔ 
  (∀ t, let (x, y) := parameterizedForm l t; lineEquation x y) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameters_satisfy_equation_l441_44130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l441_44142

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

noncomputable def angle_between (a b : E) : ℝ := Real.arccos (inner a b / (norm a * norm b))

theorem perpendicular_vectors (a b : E) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : ‖a + 2 • b‖ = ‖a - 2 • b‖) : 
  angle_between a b = Real.pi / 2 := by
  sorry

#check perpendicular_vectors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l441_44142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_workers_needed_is_30_l441_44115

/-- Road construction project parameters -/
structure RoadProject where
  totalLength : ℚ
  totalDays : ℚ
  initialWorkers : ℚ
  completedLength : ℚ
  completedDays : ℚ

/-- Calculate the number of extra workers needed to complete the project on time -/
noncomputable def extraWorkersNeeded (project : RoadProject) : ℚ :=
  let workRate := project.completedLength / (project.completedDays * project.initialWorkers)
  let remainingLength := project.totalLength - project.completedLength
  let remainingDays := project.totalDays - project.completedDays
  let manDaysNeeded := remainingLength / workRate
  let totalWorkersNeeded := manDaysNeeded / remainingDays
  totalWorkersNeeded - project.initialWorkers

/-- Theorem stating that 30 extra workers are needed for the given project -/
theorem extra_workers_needed_is_30 (project : RoadProject) 
  (h1 : project.totalLength = 10)
  (h2 : project.totalDays = 60)
  (h3 : project.initialWorkers = 30)
  (h4 : project.completedLength = 2)
  (h5 : project.completedDays = 20) :
  extraWorkersNeeded project = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_workers_needed_is_30_l441_44115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_two_l441_44113

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + 9 * x^2) - 3 * x) + 1

-- State the theorem
theorem f_sum_equals_two : f (Real.log 2 / Real.log 2) + f (Real.log (1/2) / Real.log 2) = 2 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_two_l441_44113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l441_44141

noncomputable def f (x : Real) : Real := Real.cos x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_range :
  ∀ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 4),
    0 ≤ f x ∧ f x ≤ 3 / 2 ∧
    (∃ x₁ ∈ Set.Icc (-Real.pi / 6) (Real.pi / 4), f x₁ = 0) ∧
    (∃ x₂ ∈ Set.Icc (-Real.pi / 6) (Real.pi / 4), f x₂ = 3 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l441_44141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_angle_of_sector_l441_44127

-- Define the sector
structure Sector where
  perimeter : ℝ
  radius : ℝ
  h_pos : perimeter > 0 ∧ radius > 0

-- Define the central angle of a sector
noncomputable def CentralAngle (s : Sector) : ℝ :=
  (s.perimeter - 2 * s.radius) / s.radius

-- Theorem statement
theorem central_angle_of_sector (s : Sector) 
  (h1 : s.perimeter = 4) (h2 : s.radius = 1) : 
  CentralAngle s = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_angle_of_sector_l441_44127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_circle_theorem_l441_44153

-- Define the parabola C
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y ∧ p > 0

-- Define the line l
def line_l (p : ℝ) (x y : ℝ) : Prop := y = x + p/2

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Define the circle P
def circle_P (x₀ y₀ x y : ℝ) : Prop := (x - x₀)^2 + (y - y₀)^2 = x₀^2 + (y₀ - 4)^2

-- Main theorem
theorem parabola_and_circle_theorem :
  ∀ p : ℝ,
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    parabola p x₁ y₁ ∧ parabola p x₂ y₂ ∧
    line_l p x₁ y₁ ∧ line_l p x₂ y₂ ∧
    distance x₁ y₁ x₂ y₂ = 16) →
  (∀ x₀ y₀ x_A x_B : ℝ,
    parabola p x₀ y₀ →
    circle_P x₀ y₀ x_A 0 →
    circle_P x₀ y₀ x_B 0 →
    distance 0 4 x_A 0 < distance 0 4 x_B 0) →
  (p = 4 ∧
   (∀ x₀ y₀ x_A x_B : ℝ,
     parabola 4 x₀ y₀ →
     circle_P x₀ y₀ x_A 0 →
     circle_P x₀ y₀ x_B 0 →
     distance 0 4 x_A 0 < distance 0 4 x_B 0 →
     distance 0 4 x_A 0 / distance 0 4 x_B 0 ≥ Real.sqrt 2 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_circle_theorem_l441_44153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_squared_distances_l441_44169

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem: Maximum value of |OM|^2 + |ON|^2 -/
theorem max_sum_squared_distances :
  ∃ (max : ℝ),
    max = 4 + 2 * Real.sqrt 2 ∧
    ∀ (M N : Point),
      M.y = 0 ∧ M.x > 0 ∧   -- M is on positive x-axis
      N.y = N.x ∧ N.x > 0 ∧ -- N is on y = x, x > 0
      distance M N = Real.sqrt 2 →
        M.x^2 + N.x^2 + N.y^2 ≤ max :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_squared_distances_l441_44169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_ps_length_l441_44199

-- Define the quadrilateral PQRS
structure Quadrilateral :=
  (P Q R S : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_right_angled_quadrilateral (quad : Quadrilateral) : Prop :=
  let (xQ, yQ) := quad.Q
  let (xR, yR) := quad.R
  (xQ = xR) ∧ (yQ = yR)

noncomputable def segment_length (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem quadrilateral_ps_length 
  (quad : Quadrilateral)
  (h1 : is_right_angled_quadrilateral quad)
  (h2 : segment_length quad.P quad.Q = 7)
  (h3 : segment_length quad.Q quad.R = 10)
  (h4 : segment_length quad.R quad.S = 25) :
  segment_length quad.P quad.S = 5 * Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_ps_length_l441_44199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l441_44105

/-- An ellipse with equation x²/4 + y²/2 = 1 -/
structure Ellipse where
  x : ℝ
  y : ℝ
  eq : x^2 / 4 + y^2 / 2 = 1

/-- The focal distance of an ellipse -/
noncomputable def focal_distance (e : Ellipse) : ℝ :=
  2 * Real.sqrt 2

/-- Theorem: The focal distance of the given ellipse is 2√2 -/
theorem ellipse_focal_distance (e : Ellipse) :
  focal_distance e = 2 * Real.sqrt 2 := by
  -- Unfold the definition of focal_distance
  unfold focal_distance
  -- The equality is now trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l441_44105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l441_44148

noncomputable def f (x : ℝ) : ℝ := 4 * Real.tan x * Real.sin (Real.pi / 2 - x) * Real.cos (x - Real.pi / 3) - Real.sqrt 3

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem f_properties :
  let domain := {x : ℝ | ∀ k : ℤ, x ≠ k * Real.pi + Real.pi / 2}
  (∀ T > 0, is_periodic f T → T ≥ Real.pi) ∧
  (∀ x ∈ domain ∩ Set.Icc (-Real.pi/12) (Real.pi/4), ∀ y ∈ domain ∩ Set.Icc (-Real.pi/12) (Real.pi/4), x < y → f x < f y) ∧
  (∀ x ∈ domain ∩ Set.Icc (-Real.pi/4) (-Real.pi/12), ∀ y ∈ domain ∩ Set.Icc (-Real.pi/4) (-Real.pi/12), x < y → f x > f y) ∧
  (∀ x ∈ domain ∩ Set.Icc (-Real.pi/4) (Real.pi/4), f x ≥ -2) ∧
  (f (-Real.pi/12) = -2) ∧
  (∀ x ∈ domain ∩ Set.Icc (-Real.pi/4) (Real.pi/4), f x ≤ 1) ∧
  (f (Real.pi/4) = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l441_44148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_increasing_l441_44178

-- Define the interval [0, 0.5]
def I : Set ℝ := Set.Icc 0 0.5

-- Define the functions
variable (f φ a : ℝ → ℝ)

-- State the conditions
axiom f_decreasing : ∀ {x y}, x ∈ I → y ∈ I → x < y → f x > f y
axiom φ_increasing : ∀ {x y}, x ∈ I → y ∈ I → x < y → φ x < φ y
axiom f_endpoints : f 0 = 3 ∧ f 0.5 = 2.5
axiom φ_endpoints : φ 0 = -6 ∧ φ 0.5 = -4

-- Define the relationship between a, f, and φ
axiom a_def : ∀ x, x ∈ I → ∃ k, a x = k * f x + φ x ∧ k > 0

-- State the theorem
theorem a_increasing : ∀ {x y}, x ∈ I → y ∈ I → x < y → a x < a y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_increasing_l441_44178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_length_l441_44167

/-- Calculates the length of a train given the speeds of two trains running in opposite directions, the time they take to cross each other, and the length of the other train. -/
noncomputable def train_length (speed1 speed2 : ℝ) (crossing_time : ℝ) (other_train_length : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let relative_speed_mps := relative_speed * (1000 / 3600)
  let total_distance := relative_speed_mps * crossing_time
  total_distance - other_train_length

/-- The length of the first train is 300 meters. -/
theorem first_train_length :
  train_length 120 80 9 200.04 = 300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_length_l441_44167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_x_minus_one_negative_l441_44117

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x - 1 else -(x - 1)

-- State the theorem
theorem f_x_minus_one_negative (x : ℝ) :
  (∀ y : ℝ, y ≠ 0 → f y = -f (-y)) →  -- f is odd
  (∀ y : ℝ, y > 0 → f y = y - 1) →    -- f(x) = x - 1 for x > 0
  (f (x - 1) < 0 ↔ 1 < x ∧ x < 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_x_minus_one_negative_l441_44117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twist_sequence_theorem_l441_44154

def twist (k s : ℕ) : ℕ :=
  let a := k / s
  let b := k % s
  b * s + a

def sequence_contains_one (n s : ℕ) : Prop :=
  ∃ i : ℕ, (Nat.iterate (twist s) i n) = 1

theorem twist_sequence_theorem (n s : ℕ) (h : s ≥ 2) :
  sequence_contains_one n s ↔ n % (s^2 - 1) = 1 ∨ n % (s^2 - 1) = s :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twist_sequence_theorem_l441_44154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_counterfeit_in_two_weighings_l441_44195

/-- Represents the state of the broken balance scale -/
inductive ScaleState
  | LeftHeavier
  | Balanced
  | RightHeavier

/-- Represents a coin -/
structure Coin where
  id : Nat
  isCounterfeit : Bool

/-- Represents a weighing on the broken balance scale -/
def weighing (left : List Coin) (right : List Coin) : ScaleState :=
  sorry

/-- Represents the strategy to find the counterfeit coin -/
def findCounterfeitStrategy : List Coin → Option Coin :=
  sorry

/-- Theorem stating that the strategy can identify the counterfeit coin in two weighings -/
theorem find_counterfeit_in_two_weighings 
  (coins : List Coin) 
  (h1 : coins.length = 7) 
  (h2 : ∃! c, c ∈ coins ∧ c.isCounterfeit) : 
  ∃ (strategy : List Coin → Option Coin),
    (∀ c, c ∈ coins → c.isCounterfeit → strategy coins = some c) ∧
    (∃ (weighings : List (List Coin × List Coin)), weighings.length ≤ 2) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_counterfeit_in_two_weighings_l441_44195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_in_U_l441_44177

def U : Set Int := {-2, -1, 0, 1, 2}

def A : Set Int := {x : Int | x ∈ Set.range (Nat.cast : ℕ → ℤ) ∧ -2 < x ∧ x < 3}

theorem complement_A_in_U : (U \ A) = {-2, -1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_in_U_l441_44177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_seven_digit_palindromes_prove_ten_palindromes_l441_44197

/-- Represents the available digits for forming palindromes -/
def available_digits : Multiset ℕ := {1, 2, 2, 4, 4, 4, 4}

/-- Represents a 7-digit palindrome -/
structure SevenDigitPalindrome where
  first : ℕ
  second : ℕ
  third : ℕ
  middle : ℕ
  inv_third : ℕ
  inv_second : ℕ
  inv_first : ℕ
  is_palindrome : first = inv_first ∧ second = inv_second ∧ third = inv_third
  digits_valid : Multiset.count first available_digits +
                 Multiset.count second available_digits +
                 Multiset.count third available_digits +
                 Multiset.count middle available_digits = 7

/-- The number of valid 7-digit palindromes -/
def num_valid_palindromes : ℕ := 10

/-- Theorem stating the number of valid 7-digit palindromes -/
theorem count_seven_digit_palindromes : num_valid_palindromes = 10 := by
  rfl

/-- Proof that there are exactly 10 valid 7-digit palindromes -/
theorem prove_ten_palindromes : 
  ∃ (palindromes : Finset SevenDigitPalindrome), palindromes.card = 10 ∧ 
  (∀ p : SevenDigitPalindrome, p ∈ palindromes) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_seven_digit_palindromes_prove_ten_palindromes_l441_44197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_non_negative_integers_l441_44111

theorem min_non_negative_integers (s : Finset ℝ) (avg : ℝ) : 
  s.card = 20 →
  5 ≤ avg ∧ avg ≤ 10 →
  avg = (s.sum (λ x => x)) / s.card →
  (∃ (n : ℕ), (s.filter (λ x => 0 ≤ x)).card = n ∧ 
    ∀ (m : ℕ), (s.filter (λ x => 0 ≤ x)).card = m → n ≤ m) →
  1 ≤ (s.filter (λ x => 0 ≤ x)).card :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_non_negative_integers_l441_44111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l441_44147

/-- The volume of a pyramid with a specific right triangular base and lateral edge inclination -/
theorem pyramid_volume (c : ℝ) (h : c > 0) : 
  (1/3 : ℝ) * ((c^2 * Real.sqrt 3) / 8) * (c / 2) = (c^3 * Real.sqrt 3) / 48 := by
  sorry

#check pyramid_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l441_44147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_range_for_cos_l441_44140

theorem arcsin_range_for_cos (α : Real) (x : Real) (h1 : x = Real.cos α) (h2 : α ∈ Set.Icc (-π/4) (3*π/4)) :
  Real.arcsin x ∈ Set.Icc (-π/4) (π/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_range_for_cos_l441_44140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_value_l441_44129

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 6)

noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := f (x - φ)

def is_odd (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = -h x

theorem min_phi_value :
  ∃ φ : ℝ, φ > 0 ∧ is_odd (g φ) ∧ ∀ ψ : ℝ, (ψ > 0 ∧ is_odd (g ψ)) → φ ≤ ψ ∧ φ = Real.pi / 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_value_l441_44129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_intersection_probability_l441_44185

/-- The probability that a regular 4n-gon with radius 1 intersects the edge of a square
    on a chessboard with squares of side length 4. -/
noncomputable def intersection_probability (n : ℕ) : ℝ :=
  (4 * n / Real.pi) * Real.sin (Real.pi / (4 * n : ℝ)) -
  (n / (8 * Real.pi)) * Real.sin (Real.pi / (2 * n : ℝ)) -
  1 / 8

/-- Theorem stating the probability of intersection for a regular 4n-gon on a chessboard. -/
theorem regular_polygon_intersection_probability (n : ℕ) :
  let p := intersection_probability n
  0 ≤ p ∧ p ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_intersection_probability_l441_44185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l441_44190

def a : ℝ × ℝ := (-2, -1)
def b (lambda : ℝ) : ℝ × ℝ := (lambda, 1)

def angle_obtuse (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 < 0

theorem lambda_range (lambda : ℝ) :
  angle_obtuse a (b lambda) → lambda > -1/2 ∧ lambda ≠ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l441_44190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_is_four_l441_44174

/-- The function f(x) = sin(ωx + φ) -/
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem min_omega_is_four :
  ∀ ω φ : ℝ,
  ω > 0 →
  (∀ x : ℝ, f ω φ (x - π/16) = f ω φ (π/16 - x)) →
  f ω φ (-π/16) = 0 →
  (∃ x₀ : ℝ, ∀ x : ℝ, f ω φ x₀ ≤ f ω φ x ∧ f ω φ x ≤ f ω φ (x₀ + π/4)) →
  ω ≥ 4 ∧ (ω = 4 → ∃ φ : ℝ, 
    (∀ x : ℝ, f 4 φ (x - π/16) = f 4 φ (π/16 - x)) ∧
    f 4 φ (-π/16) = 0 ∧
    (∃ x₀ : ℝ, ∀ x : ℝ, f 4 φ x₀ ≤ f 4 φ x ∧ f 4 φ x ≤ f 4 φ (x₀ + π/4))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_is_four_l441_44174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_law_firm_associates_distribution_l441_44102

theorem law_firm_associates_distribution 
  (second_year_percentage : ℝ) 
  (not_first_year_percentage : ℝ) 
  (h1 : second_year_percentage = 25)
  (h2 : not_first_year_percentage = 75) : 
  100 - second_year_percentage - (100 - not_first_year_percentage) = 50 := by
  -- Substitute the given values
  rw [h1, h2]
  -- Simplify the expression
  ring
  -- The proof is complete
  done

#check law_firm_associates_distribution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_law_firm_associates_distribution_l441_44102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_animal_shelter_arithmetic_sequence_l441_44118

theorem animal_shelter_arithmetic_sequence 
  (c d r : ℕ) 
  (hc : c = 645) 
  (hd : d = 567) 
  (hr : r = 316) 
  (hseq : ∃ k : ℤ, d = c - k ∧ r = d - k) :
  ∃ (k a : ℤ) (S : ℕ), 
    k = 78 ∧ 
    a = 238 ∧ 
    S = 1766 ∧
    a = r - k ∧
    S = (c + d + r + a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_animal_shelter_arithmetic_sequence_l441_44118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l441_44133

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x ^ a

theorem power_function_through_point (a : ℝ) :
  f a 2 = Real.sqrt 2 / 2 → f a 4 = 1 / 2 := by
  intro h
  -- The proof steps would go here
  sorry

#check power_function_through_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l441_44133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l441_44163

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) := Real.log (-x^2 - 2*x + 3)

-- State the theorem
theorem f_increasing_interval :
  ∃ (a b : ℝ), a = -3 ∧ b = -1 ∧
  (∀ x y, a < x ∧ x < y ∧ y < b → f x < f y) ∧
  (∀ x, x < a → ¬(∀ y, x < y → f x < f y)) ∧
  (∀ x, b < x → ¬(∀ y, x < y → f x < f y)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l441_44163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l441_44145

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def smallest_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  has_period f p ∧ ∀ q, 0 < q → q < p → ¬(has_period f q)

theorem max_m_value (f : ℝ → ℝ) (m : ℝ) :
  is_odd f →
  smallest_period f 5 →
  f 2 ≥ 2 →
  f 3 = (2^(m+1) - 3) / (2^m + 1) →
  m ≤ -2 ∧ ∃ m₀ : ℝ, m₀ ≤ -2 ∧ f 3 = (2^(m₀+1) - 3) / (2^m₀ + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l441_44145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_l441_44184

/-- The cost of fencing a circular field -/
theorem fencing_cost (diameter : ℝ) (cost_per_meter : ℝ) (π : ℝ) 
  (h_diameter : diameter = 36) (h_cost : cost_per_meter = 3.5) (h_pi : π = 3.14159) :
  ∃ (total_cost : ℝ), (abs (total_cost - (π * diameter * cost_per_meter)) < 0.01) ∧ 
  (abs (total_cost - 395.85) < 0.01) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_l441_44184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l441_44139

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else 3^x

-- State the theorem
theorem f_composition_value : f (f (1/9)) = 1/9 := by
  -- Evaluate f(1/9)
  have h1 : f (1/9) = -2 := by
    sorry
  
  -- Evaluate f(-2)
  have h2 : f (-2) = 1/9 := by
    sorry
  
  -- Combine the results
  rw [h1, h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l441_44139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_characterization_l441_44135

noncomputable def f (x : ℝ) : ℝ := 
  1 / (x^2 + 9*x - 12) + 1 / (x^2 + 4*x - 5) + 1 / (x^2 - 15*x - 12)

theorem solution_characterization (x : ℝ) : 
  f x = 0 ↔ x = 3 ∨ x = 1 ∨ x = -4 ∨ x = -5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_characterization_l441_44135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_i_part_ii_l441_44151

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
  t.C = Real.pi / 3 ∧ t.c = 4

-- Theorem for part I
theorem part_i (t : Triangle) (h : triangle_conditions t) (h_sin_A : Real.sin t.A = 3/4) :
  t.a = 2 * Real.sqrt 3 := by
  sorry

-- Theorem for part II
theorem part_ii (t : Triangle) (h : triangle_conditions t) 
  (h_area : (1/2) * t.a * t.b * Real.sin t.C = 4 * Real.sqrt 3) :
  t.a = 4 ∧ t.b = 4 := by
  sorry

#check part_i
#check part_ii

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_i_part_ii_l441_44151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l441_44180

/-- A function f(x) with given properties -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (a * x + b) / (x^2 + 1)

/-- Main theorem combining all conditions and results -/
theorem main_theorem (a b : ℝ) :
  (∀ x, f a b x = -f a b (-x)) →  -- f is odd
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f a b x < f a b y) →  -- f is increasing on (-1, 1)
  (f a b (1/2) = 2/5) →  -- f(1/2) = 2/5
  (∀ x, f a b x = x / (x^2 + 1)) ∧  -- Explicit formula
  (∀ t, f a b (t-1) + f a b t < 0 ↔ 0 < t ∧ t < 1/2) :=  -- Solution to inequality
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l441_44180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_to_line_proof_l441_44182

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (t - 3, Real.sqrt 3 * t)

def circle_C (ρ θ : ℝ) : Prop := ρ^2 - 4*ρ*(Real.cos θ) + 3 = 0

noncomputable def distance_center_to_line : ℝ := (5 * Real.sqrt 3) / 2

theorem distance_center_to_line_proof :
  ∃ (x₀ y₀ : ℝ), 
    (∀ θ, circle_C x₀ θ → x₀ * Real.cos θ = 2 ∧ x₀ * Real.sin θ = 0) →
    (∀ t, (Real.sqrt 3) * (line_l t).1 - (line_l t).2 + 3 * Real.sqrt 3 = 0) →
    distance_center_to_line = 
      (|(Real.sqrt 3) * x₀ - y₀ + 3 * Real.sqrt 3| / Real.sqrt ((Real.sqrt 3)^2 + 1^2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_to_line_proof_l441_44182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anns_fare_is_231_l441_44194

/-- Represents the fare calculation for a taxi ride -/
structure TaxiFare where
  /-- The distance traveled in miles -/
  distance : ℚ
  /-- The booking fee in dollars -/
  bookingFee : ℚ
  /-- The fare for a 50-mile ride without booking fee -/
  fare50Miles : ℚ

/-- Calculates the total fare for a taxi ride -/
def totalFare (t : TaxiFare) : ℚ :=
  (t.fare50Miles / 50) * t.distance + t.bookingFee

/-- Theorem stating that the total fare for Ann's trip is $231 -/
theorem anns_fare_is_231 :
  let t : TaxiFare := { distance := 90, bookingFee := 15, fare50Miles := 120 }
  totalFare t = 231 := by
  sorry

#eval totalFare { distance := 90, bookingFee := 15, fare50Miles := 120 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_anns_fare_is_231_l441_44194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l441_44108

open Real

/-- The natural logarithm function -/
noncomputable def f (x : ℝ) : ℝ := log x

/-- The function g(x) = a/x where a > 0 -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a / x

/-- The function F(x) = f(x) + g(x) -/
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := f x + g a x

/-- The derivative of F(x) -/
noncomputable def F_deriv (a : ℝ) (x : ℝ) : ℝ := (x - a) / (x^2)

theorem min_a_value (a : ℝ) :
  (a > 0) →
  (∀ x ∈ Set.Ioo 0 3, F_deriv a x ≤ (1/2)) →
  a ≥ (1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l441_44108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l441_44122

theorem exponential_inequality (a b : ℝ) (h : a > b) : (2 : ℝ)^a > (2 : ℝ)^b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l441_44122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_roots_l441_44103

/-- A function satisfying certain symmetry properties -/
def SymmetricFunction (g : ℝ → ℝ) : Prop :=
  (∀ x, g (3 + x) = g (3 - x)) ∧ (∀ x, g (5 + x) = g (5 - x))

/-- The set of roots of g in the interval [-1000, 1000] -/
def RootsInInterval (g : ℝ → ℝ) : Set ℝ :=
  {x | -1000 ≤ x ∧ x ≤ 1000 ∧ g x = 0}

/-- The main theorem -/
theorem symmetric_function_roots
  (g : ℝ → ℝ)
  (h₁ : SymmetricFunction g)
  (h₂ : g 1 = 0) :
  ∃ (s : Finset ℝ), s.card ≥ 250 ∧ ∀ x ∈ s, x ∈ RootsInInterval g :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_roots_l441_44103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_discount_percentage_l441_44156

/-- Calculates the discount percentage given profit percentages with and without discount -/
noncomputable def discount_percentage (profit_with_discount : ℝ) (profit_without_discount : ℝ) : ℝ :=
  ((1 + profit_without_discount) - (1 + profit_with_discount)) / (1 + profit_without_discount) * 100

/-- Theorem stating that the discount percentage for the given problem is approximately 48.18% -/
theorem book_discount_percentage :
  let profit_with_discount : ℝ := 0.14
  let profit_without_discount : ℝ := 1.20
  abs (discount_percentage profit_with_discount profit_without_discount - 48.18) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_discount_percentage_l441_44156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_computation_problem_l441_44107

theorem computation_problem :
  (((9 : ℝ) / 4) ^ (-(1 : ℝ) / 2)) = 2 / 3 ∧
  (2 : ℝ) ^ (Real.log 3 / Real.log 2) + Real.log (1 / 100) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_computation_problem_l441_44107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_sum_inequality_l441_44112

-- Define the functions f, g, and h
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x
def g (a : ℝ) (x : ℝ) : ℝ := (a + 1) * x^2 - 1
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := 2 * f a x - g a x

-- State the theorem
theorem zeros_sum_inequality (a : ℝ) (x₁ x₂ : ℝ) 
  (h_a : -1 < a ∧ a < 0) 
  (h_zeros : h a x₁ = 0 ∧ h a x₂ = 0) 
  (h_distinct : x₁ ≠ x₂) :
  x₁ + x₂ > 2 / (a + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_sum_inequality_l441_44112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_greatest_average_speed_l441_44160

/-- Represents a palindromic number -/
def IsPalindrome (n : ℕ) : Prop := sorry

/-- Calculates the average speed given distance and time -/
noncomputable def AverageSpeed (distance time : ℝ) : ℝ := distance / time

theorem james_greatest_average_speed 
  (start_odometer end_odometer : ℕ) 
  (drive_time : ℝ) 
  (speed_limit : ℝ) :
  IsPalindrome start_odometer →
  IsPalindrome end_odometer →
  drive_time = 5 →
  speed_limit = 60 →
  start_odometer = 12321 →
  end_odometer > start_odometer →
  (end_odometer : ℝ) - (start_odometer : ℝ) ≤ speed_limit * drive_time →
  ∀ (possible_speed : ℝ), 
    AverageSpeed ((end_odometer : ℝ) - (start_odometer : ℝ)) drive_time ≤ possible_speed → 
    possible_speed ≤ speed_limit →
    AverageSpeed ((end_odometer : ℝ) - (start_odometer : ℝ)) drive_time ≤ 60 := by
  sorry

#check james_greatest_average_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_greatest_average_speed_l441_44160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frank_pie_consumption_l441_44104

theorem frank_pie_consumption (erik_pie : Real) (difference : Real) (frank_pie : Real)
  (h1 : erik_pie = 0.67)
  (h2 : difference = 0.34)
  (h3 : erik_pie = frank_pie + difference) : 
  frank_pie = 0.33 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frank_pie_consumption_l441_44104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_proof_l441_44171

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1

theorem tangent_line_proof (a : ℝ) : 
  (∃ m : ℝ, 
    -- Tangent line equation
    (λ x ↦ f a 1 + m * (x - 1)) 2 = 7 ∧ 
    -- Slope of tangent line equals the derivative at x = 1
    m = f_derivative a 1) → 
  a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_proof_l441_44171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_consecutive_digit_square_swap_l441_44110

/-- A function that checks if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- A function that swaps the hundreds and tens digits of a four-digit number -/
def swapHundredsTens (n : ℕ) : ℕ :=
  let thousands := n / 1000
  let hundreds := (n / 100) % 10
  let tens := (n / 10) % 10
  let ones := n % 10
  thousands * 1000 + tens * 100 + hundreds * 10 + ones

/-- A function that checks if four digits are consecutive -/
def areConsecutive (a b c d : ℕ) : Prop :=
  (b = a + 1) ∧ (c = b + 1) ∧ (d = c + 1)

theorem no_consecutive_digit_square_swap :
  ¬ ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧
    (let a := n / 1000
     let b := (n / 100) % 10
     let c := (n / 10) % 10
     let d := n % 10
     areConsecutive a b c d ∧ isPerfectSquare (swapHundredsTens n)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_consecutive_digit_square_swap_l441_44110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_wrt_x_axis_solution_symmetry_l441_44109

/-- Definition of symmetry with respect to the x-axis -/
def is_symmetrical_wrt_x_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = p2.1 ∧ p1.2 = -p2.2

/-- Given a point M with coordinates (x, y), its symmetrical point M' with respect to the x-axis has coordinates (x, -y) -/
theorem symmetry_wrt_x_axis (x y : ℝ) : 
  let M : ℝ × ℝ := (x, y)
  let M' : ℝ × ℝ := (x, -y)
  is_symmetrical_wrt_x_axis M M' := by
  sorry

/-- The point (3, 4) is symmetrical to (3, -4) with respect to the x-axis -/
theorem solution_symmetry : 
  is_symmetrical_wrt_x_axis (3, -4) (3, 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_wrt_x_axis_solution_symmetry_l441_44109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_positive_function_l441_44125

open Set Real

theorem decreasing_positive_function
  {f : ℝ → ℝ} {f' : ℝ → ℝ} {a b : ℝ} (h1 : a < b)
  (h2 : ∀ x ∈ Ioo a b, HasDerivAt f (f' x) x)
  (h3 : ∀ x ∈ Ioo a b, f' x < 0)
  (h4 : f b > 0) :
  ∀ x ∈ Ioo a b, f x > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_positive_function_l441_44125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l441_44164

/-- A rational function with two specified vertical asymptotes -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (x + 4) / (x^2 + a*x + b)

/-- The property of having vertical asymptotes at x=1 and x=-2 -/
def has_asymptotes (a b : ℝ) : Prop :=
  ∀ x, x^2 + a*x + b = 0 ↔ (x = 1 ∨ x = -2)

/-- Theorem stating that if f has the specified asymptotes, then a + b = -1 -/
theorem sum_of_coefficients (a b : ℝ) (h : has_asymptotes a b) : a + b = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l441_44164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_percent_increase_q11_to_q12_l441_44191

-- Define the list of question values
def questionValues : List ℕ := [100, 300, 600, 900, 1200, 1700, 2700, 5000, 8000, 12000, 16000, 21000, 27000, 34000, 50000]

-- Define a function to calculate percent increase
def percentIncrease (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

-- Theorem statement
theorem smallest_percent_increase_q11_to_q12 :
  ∀ i ∈ Finset.range 14,
    i ≠ 10 →
    percentIncrease (questionValues.get! i) (questionValues.get! (i + 1)) ≥
    percentIncrease (questionValues.get! 10) (questionValues.get! 11) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_percent_increase_q11_to_q12_l441_44191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_abc_l441_44155

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem smallest_difference_abc (a b c : ℕ+) : 
  a * b * c = factorial 9 → a < b → b < c → 
  (∀ a' b' c' : ℕ+, a' * b' * c' = factorial 9 → a' < b' → b' < c' → c' - a' ≥ c - a) →
  c - a = 216 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_abc_l441_44155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_10_terms_l441_44150

/-- Sequence a_n defined recursively -/
def a : ℕ → ℕ
  | 0 => 1  -- Adding the base case for 0
  | 1 => 1
  | n + 1 => 2 * a n + n - 1

/-- Sum of the first n terms of sequence a_n -/
def S (n : ℕ) : ℕ :=
  (List.range n).map (fun i => a (i + 1)) |>.sum

/-- The sum of the first 10 terms of the sequence is 1991 -/
theorem sum_of_first_10_terms : S 10 = 1991 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_10_terms_l441_44150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_on_unit_circle_l441_44165

theorem dot_product_on_unit_circle (x₁ y₁ x₂ y₂ θ : ℝ) : 
  (x₁^2 + y₁^2 = 1) →
  (x₂^2 + y₂^2 = 1) →
  (π/2 < θ) →
  (θ < π) →
  (Real.sin (θ + π/4) = 3/5) →
  (x₁*x₂ + y₁*y₂ = -Real.sqrt 2 / 10) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_on_unit_circle_l441_44165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_amount_proof_l441_44176

/-- Simple interest calculation -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem investment_amount_proof (interest_rate time interest : ℝ) 
  (h1 : interest_rate = 17.5)
  (h2 : time = 2.5)
  (h3 : interest = 3150) :
  ∃ principal : ℝ, 
    simple_interest principal interest_rate time = interest ∧ 
    principal = 7200 := by
  use 7200
  constructor
  · -- Prove that simple_interest 7200 17.5 2.5 = 3150
    rw [simple_interest, h1, h2, h3]
    norm_num
  · -- Prove that 7200 = 7200
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_amount_proof_l441_44176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_true_propositions_l441_44123

-- Define a proposition type
inductive Proposition
| prop1
| prop2
| prop3
| prop4
| prop5

-- Define a function to check if a proposition is true
def is_true (p : Proposition) : Bool :=
  match p with
  | Proposition.prop1 => false  -- Proposition ① is false
  | Proposition.prop2 => true   -- Proposition ② is true
  | Proposition.prop3 => false  -- Proposition ③ is false
  | Proposition.prop4 => true   -- Proposition ④ is true
  | Proposition.prop5 => true   -- Proposition ⑤ is true

-- Define a function to count the number of true propositions
def count_true_propositions : Nat :=
  let props := [Proposition.prop1, Proposition.prop2, Proposition.prop3, Proposition.prop4, Proposition.prop5]
  props.filter is_true |>.length

-- Theorem stating that the number of true propositions is 3
theorem num_true_propositions : count_true_propositions = 3 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_true_propositions_l441_44123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chicken_cage_problem_l441_44138

theorem chicken_cage_problem (x : ℕ) : 
  (∃ n : ℕ, n = 4 * x + 1) ∧ 
  (∃ m : ℕ, m = 5 * (x - 1)) ∧ 
  (4 * x + 1 = 5 * (x - 1)) →
  6 ≤ x ∧ x ≤ 10 := by
  intro h
  sorry

#check chicken_cage_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chicken_cage_problem_l441_44138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_one_l441_44173

/-- A regular triangular prism with base ABC and top A₁B₁C₁ -/
structure RegularTriangularPrism where
  -- Base side length
  base_side : ℝ
  -- Prism side length (height)
  prism_side : ℝ

/-- A point D on the base of the prism -/
structure MidpointD where
  -- D is the midpoint of BC

/-- The volume of the triangular pyramid A-B₁DC₁ -/
def pyramid_volume (prism : RegularTriangularPrism) (d : MidpointD) : ℝ :=
  -- Definition of the pyramid volume
  1 -- Placeholder value, replace with actual calculation when implementing

theorem pyramid_volume_is_one (prism : RegularTriangularPrism) (d : MidpointD) 
    (h1 : prism.base_side = 2)
    (h2 : prism.prism_side = Real.sqrt 3) :
    pyramid_volume prism d = 1 := by
  sorry

#check pyramid_volume_is_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_one_l441_44173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l441_44106

/-- The area of a triangle with vertices (2, 2, 0), (5, 6, 1), and (1, 0, 3) is 5√3 -/
theorem triangle_area : ∃ area : ℝ, area = 5 * Real.sqrt 3 := by
  -- Define the vertices of the triangle
  let a : Fin 3 → ℝ := ![2, 2, 0]
  let b : Fin 3 → ℝ := ![5, 6, 1]
  let c : Fin 3 → ℝ := ![1, 0, 3]

  -- Calculate the area of the triangle
  let area := (1/2 : ℝ) * (((b 0 - a 0) * (c 1 - a 1) - (b 1 - a 1) * (c 0 - a 0))^2 +
                           ((b 1 - a 1) * (c 2 - a 2) - (b 2 - a 2) * (c 1 - a 1))^2 +
                           ((b 2 - a 2) * (c 0 - a 0) - (b 0 - a 0) * (c 2 - a 2))^2).sqrt

  -- Prove that the calculated area is equal to 5√3
  have h : area = 5 * Real.sqrt 3 := by sorry

  -- Return the result
  exact ⟨area, h⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l441_44106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_2016_equals_one_l441_44168

-- Define the sequence a_n
def a : ℕ → ℚ
  | 0 => 3  -- We define a_0 as 3 to match a_1 in the original problem
  | n + 1 => (a n - 1) / (a n)

-- Define A_n as the product of the first n terms of a_n
def A (n : ℕ) : ℚ := (Finset.range n).prod (λ i => a i)

-- State the main theorem
theorem A_2016_equals_one : A 2016 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_2016_equals_one_l441_44168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_terminal_side_l441_44101

-- Define the angle α
noncomputable def α : Real := Real.arcsin (Real.sqrt 13 / 13)

-- Define the point P
noncomputable def P (y : Real) : Real × Real := (-Real.sqrt 3, y)

-- Theorem statement
theorem point_on_terminal_side (y : Real) :
  (P y).1 = -Real.sqrt 3 ∧
  (P y).2 = y ∧
  y > 0 ∧
  Real.sin α = Real.sqrt 13 / 13 →
  y = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_terminal_side_l441_44101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_properties_l441_44187

noncomputable def f (x : ℝ) : ℝ := Real.sin x

noncomputable def g (x : ℝ) : ℝ := Real.sin (3 * x - Real.pi / 4)

theorem sin_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x₁ x₂, x₁ + x₂ = Real.pi / 2 → g x₁ = g x₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_properties_l441_44187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_class_size_l441_44196

/-- Represents the number of students in each of the three equal columns -/
def x : ℕ := sorry

/-- The total number of students in the class -/
def total_students : ℕ := 3 * x + (x + 3)

/-- The class size is over 50 students -/
axiom size_over_50 : total_students > 50

/-- The number of students in each column must be a whole number -/
axiom whole_number : x ∈ Set.univ

/-- The smallest possible class size that satisfies all conditions is 51 -/
theorem smallest_class_size : 
  (∀ y : ℕ, y < x → 3 * y + (y + 3) ≤ 50) ∧ total_students = 51 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_class_size_l441_44196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l441_44192

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  (a 1 = 1) ∧ 
  (∀ n : ℕ, n ≥ 2 → a n - a (n-1) = n - 1)

theorem sequence_formula (a : ℕ → ℚ) (h : arithmetic_sequence a) :
  ∀ n : ℕ, n > 0 → a n = (n * (n + 1)) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l441_44192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_cost_for_two_hours_l441_44172

/-- The cost to park a car in a certain parking garage. -/
noncomputable def parking_cost (initial_cost : ℝ) : ℝ → ℝ :=
  λ hours => if hours ≤ 2 then initial_cost else initial_cost + 1.75 * (hours - 2)

/-- The average cost per hour for a given duration. -/
noncomputable def average_cost (initial_cost : ℝ) (hours : ℝ) : ℝ :=
  parking_cost initial_cost hours / hours

theorem parking_cost_for_two_hours :
  ∃ (initial_cost : ℝ), 
    initial_cost = 9 ∧ 
    average_cost initial_cost 9 = 2.361111111111111 := by
  -- Provide the value of initial_cost
  use 9
  constructor
  · -- Prove initial_cost = 9
    rfl
  · -- Prove average_cost 9 9 = 2.361111111111111
    -- We'll use 'sorry' here as the actual computation might be complex
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_cost_for_two_hours_l441_44172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_b_a_plus_one_l441_44186

noncomputable def e : ℝ := Real.exp 1

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x + (1/2) * x^2

noncomputable def g (a b x : ℝ) : ℝ := (1/2) * x^2 + a * x + b

theorem max_value_b_a_plus_one (a b : ℝ) (h : ∀ x, f x ≥ g a b x) :
  b * (a + 1) ≤ e / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_b_a_plus_one_l441_44186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_advertising_effectiveness_l441_44179

-- Define the sales amount as a function of advertising cost
noncomputable def sales_amount (x : ℝ) : ℝ := 100 * Real.sqrt x

-- Define the advertising effectiveness as a function of advertising cost
noncomputable def advertising_effectiveness (x : ℝ) : ℝ := sales_amount x - x

-- Theorem stating the maximum advertising effectiveness
theorem max_advertising_effectiveness :
  ∀ x : ℝ, x ≥ 0 → advertising_effectiveness x ≤ advertising_effectiveness 2500 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_advertising_effectiveness_l441_44179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_max_score_l441_44159

def player_score : ℕ → ℕ
| _ => 0  -- Default implementation, can be refined later

def max_player_score (n : ℕ) (total_points : ℕ) (min_points : ℕ) : ℕ :=
  total_points - (n - 1) * min_points

theorem basketball_max_score (n : ℕ) (total_points : ℕ) (min_points : ℕ) 
  (h1 : n = 12)
  (h2 : total_points = 100)
  (h3 : min_points = 7)
  (h4 : ∀ i, i < n → min_points ≤ player_score i) :
  max_player_score n total_points min_points = 23 ∧ 
  ∃ i, i < n ∧ player_score i = max_player_score n total_points min_points :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_max_score_l441_44159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_D_coordinates_l441_44198

noncomputable section

/-- Semicircle C in polar coordinates -/
def C (θ : ℝ) : ℝ := 2 * Real.cos θ

/-- Parametric equation of C -/
def C_param (t : ℝ) : ℝ × ℝ :=
  (1 + Real.cos t, Real.sin t)

/-- Line l -/
noncomputable def l (x : ℝ) : ℝ := Real.sqrt 3 * x + 2

theorem point_D_coordinates :
  ∃ t : ℝ, t ∈ Set.Icc 0 Real.pi ∧
    let (x, y) := C_param t
    (x, y) ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 = (C (Real.arccos (p.1 / 2)))^2} ∧
    (y - l x) * Real.sqrt 3 = -1 ∧
    x = 3/2 ∧ y = Real.sqrt 3 / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_D_coordinates_l441_44198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_sum_angle_l441_44144

theorem roots_sum_angle (a : ℝ) (α β : ℝ) : 
  a > 2 →
  α ∈ Set.Ioo (-π/2 : ℝ) (π/2 : ℝ) →
  β ∈ Set.Ioo (-π/2 : ℝ) (π/2 : ℝ) →
  (Real.tan α)^2 + 3*a*(Real.tan α) + 3*a + 1 = 0 →
  (Real.tan β)^2 + 3*a*(Real.tan β) + 3*a + 1 = 0 →
  α + β = π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_sum_angle_l441_44144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l441_44162

noncomputable def f (x : ℝ) : ℝ := Real.cos x + (Real.cos (2 * x)) / 2 + (Real.cos (4 * x)) / 4

theorem f_properties :
  (∀ x, f x = f (-x)) ∧ 
  (∀ x, f x = f (2 * Real.pi - x)) ∧ 
  (∀ x, deriv f x < 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l441_44162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pages_and_min_leftover_l441_44152

/-- Represents the number of digits required to print a page number. -/
def digitsForPage (page : Nat) : Nat :=
  if page < 10 then 1
  else if page < 100 then 2
  else 3

/-- Calculates the total number of digits required to print all pages up to a given page. -/
def totalDigitsUpTo (page : Nat) : Nat :=
  List.range (page + 1) |>.drop 1 |>.map digitsForPage |>.sum

/-- The total number of available printing letters. -/
def totalLetters : Nat := 2011

/-- Theorem stating the maximum number of pages and minimum leftover letters. -/
theorem max_pages_and_min_leftover :
  ∃ (maxPages : Nat) (minLeftover : Nat),
    maxPages = 706 ∧
    minLeftover = 1 ∧
    totalDigitsUpTo maxPages + minLeftover = totalLetters ∧
    ∀ p, p > maxPages → totalDigitsUpTo p > totalLetters := by
  sorry

#eval totalDigitsUpTo 706 + 1 -- Should output 2011

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pages_and_min_leftover_l441_44152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jonathan_additional_payment_l441_44128

-- Define the conversion rate
noncomputable def usd_to_gbp_rate : ℝ := 1 / 1.35

-- Define the cost of the pens
noncomputable def pen_cost : ℝ := 15

-- Define Hiro's USD amount
noncomputable def hiro_usd : ℝ := 20

-- Define the amount Jonathan has already spent
noncomputable def jonathan_spent : ℝ := 3

-- Theorem statement
theorem jonathan_additional_payment :
  max (pen_cost - jonathan_spent - hiro_usd * usd_to_gbp_rate) 0 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jonathan_additional_payment_l441_44128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2019_eq_l441_44161

/-- The function f(x) = x^2 / (2x - 1) -/
noncomputable def f (x : ℝ) : ℝ := x^2 / (2*x - 1)

/-- The sequence of functions defined by f₁(x) = f(x) and f_{n+1}(x) = f(f_n(x)) -/
noncomputable def f_seq : ℕ → (ℝ → ℝ)
  | 0 => id
  | 1 => f
  | (n+2) => f ∘ f_seq (n+1)

/-- The theorem stating that the 2019th iteration of f is equal to the given expression -/
theorem f_2019_eq (x : ℝ) (h : x ≠ 0 ∧ x ≠ 1) : 
  f_seq 2019 x = x^(2^2019) / (x^(2^2019) - (x-1)^(2^2019)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2019_eq_l441_44161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_EOF_l441_44136

-- Define the line L: x - 2y - 3 = 0
def L (x y : ℝ) : Prop := x - 2*y - 3 = 0

-- Define the circle C: (x - 2)² + (y + 3)² = 9
def C (x y : ℝ) : Prop := (x - 2)^2 + (y + 3)^2 = 9

-- Define points E and F as intersection points of L and C
noncomputable def E : ℝ × ℝ := sorry
noncomputable def F : ℝ × ℝ := sorry

-- Define O as the origin
def O : ℝ × ℝ := (0, 0)

-- State the theorem
theorem area_of_triangle_EOF :
  ∃ (E F : ℝ × ℝ), 
    L E.1 E.2 ∧ C E.1 E.2 ∧
    L F.1 F.2 ∧ C F.1 F.2 ∧
    E ≠ F ∧
    let area := abs ((E.1 - O.1) * (F.2 - O.2) - (F.1 - O.1) * (E.2 - O.2)) / 2
    area = 6 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_EOF_l441_44136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_ten_l441_44170

theorem sum_of_solutions_is_ten :
  ∃ (S : Finset ℝ), (∀ x ∈ S, |x^2 - 10*x + 29| = 5) ∧
                    (∀ x : ℝ, |x^2 - 10*x + 29| = 5 → x ∈ S) ∧
                    (S.sum id = 10) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_ten_l441_44170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_multiplicative_l441_44119

-- Define the function f(x) = 3^x
noncomputable def f (x : ℝ) : ℝ := (3 : ℝ) ^ x

-- State the theorem
theorem f_multiplicative (x y : ℝ) : f (x + y) = f x * f y := by
  -- Unfold the definition of f
  unfold f
  -- Use the properties of exponents
  simp [Real.rpow_add]
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_multiplicative_l441_44119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_properties_l441_44100

-- Define set A
def A : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}

-- Define set B
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2 + 1}

-- Statement of the theorem
theorem set_properties :
  (1, 2) ∈ B ∧
  ¬(∀ y, y ∈ A ↔ (∃ x, (x, y) ∈ B)) ∧
  0 ∉ A ∧
  (0, 0) ∉ B :=
by
  sorry

#print axioms set_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_properties_l441_44100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2x0_minus_pi_12_l441_44193

noncomputable def a (x : Real) : Real × Real := (2 * Real.cos x, Real.sqrt 3 / 2)

noncomputable def b (x : Real) : Real × Real := (Real.sin (x - Real.pi / 3), 1)

noncomputable def f (x : Real) : Real := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem tan_2x0_minus_pi_12 (x₀ : Real) (h1 : x₀ ∈ Set.Icc (5 * Real.pi / 12) (2 * Real.pi / 3)) 
  (h2 : f x₀ = 4 / 5) : Real.tan (2 * x₀ - Real.pi / 12) = -1 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2x0_minus_pi_12_l441_44193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rightmost_box_contents_l441_44124

/-- Represents a box containing red and blue balls -/
structure Box where
  red : Nat
  blue : Nat

/-- The problem setup -/
structure BoxProblem where
  boxes : Vector Box 10
  total_red : Nat
  total_blue : Nat
  non_empty : ∀ i, (boxes.get i).red + (boxes.get i).blue > 0
  non_decreasing : ∀ i j, i < j → (boxes.get i).red + (boxes.get i).blue ≤ (boxes.get j).red + (boxes.get j).blue
  unique_combinations : ∀ i j, i ≠ j → (boxes.get i).red ≠ (boxes.get j).red ∨ (boxes.get i).blue ≠ (boxes.get j).blue
  red_sum : (boxes.toList.map Box.red).sum = total_red
  blue_sum : (boxes.toList.map Box.blue).sum = total_blue

theorem rightmost_box_contents (p : BoxProblem) (h1 : p.total_red = 11) (h2 : p.total_blue = 13) :
  (p.boxes.get 9).red = 1 ∧ (p.boxes.get 9).blue = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rightmost_box_contents_l441_44124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l441_44189

-- Define the function f
def f (x : ℝ) : ℝ := 8 * x^3 - 6 * x

-- State the theorem
theorem tangent_slope_at_one :
  (∀ x, f (x/2) = x^3 - 3*x) →
  (deriv f) 1 = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l441_44189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equals_sin_of_sin_zero_solutions_l441_44188

theorem sin_equals_sin_of_sin_zero_solutions 
  (h : ∀ θ : Real, 0 < θ → θ < Real.pi / 2 → Real.sin θ > θ) :
  ¬ ∃ x : Real, 0 ≤ x ∧ x < Real.pi / 2 ∧ Real.sin x = Real.sin (Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equals_sin_of_sin_zero_solutions_l441_44188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_result_l441_44183

theorem power_sum_result (m n : ℕ) (h1 : (2 : ℝ)^m = 5) (h2 : (2 : ℝ)^n = 6) : (2 : ℝ)^(m+n) = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_result_l441_44183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_relations_l441_44116

-- Define a cyclic quadrilateral
structure CyclicQuadrilateral where
  A : Real
  B : Real
  C : Real
  D : Real
  cyclic : A + C = B + D

-- Define the four relations
def relation1 (q : CyclicQuadrilateral) : Prop := Real.sin q.A = Real.sin q.C
def relation2 (q : CyclicQuadrilateral) : Prop := Real.sin q.A + Real.sin q.C = 0
def relation3 (q : CyclicQuadrilateral) : Prop := Real.cos q.B + Real.cos q.D = 0
def relation4 (q : CyclicQuadrilateral) : Prop := Real.cos q.B = Real.cos q.D

-- Theorem statement
theorem cyclic_quadrilateral_relations (q : CyclicQuadrilateral) :
  (relation1 q ∧ relation3 q) ∧ ¬(relation2 q ∨ relation4 q) := by
  sorry

-- Helper lemmas that might be useful for the proof
lemma sin_supplementary_angles (x : Real) : Real.sin x = Real.sin (π - x) := by
  sorry

lemma cos_supplementary_angles (x : Real) : Real.cos x = -Real.cos (π - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_relations_l441_44116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_inequality_solution_l441_44120

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 - 2 / (2^x + 1)

-- Theorem for the parity of f
theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

-- Theorem for the solution of the inequality
theorem inequality_solution : 
  ∀ t : ℝ, f t + f (t^2 - t - 1) < 0 ↔ -1 < t ∧ t < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_inequality_solution_l441_44120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l441_44149

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  Real.sin A = 2 * Real.sqrt 3 * Real.sin B * Real.sin C →
  b * c = 4 →
  a = 2 * Real.sqrt 3 →
  A = π / 3 ∧ a + b + c = 2 * (Real.sqrt 3 + Real.sqrt 6) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l441_44149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l441_44175

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (2 * x - 3)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x > 3/2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l441_44175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_circumradii_l441_44114

/-- A quadrilateral with an inscribed circle. -/
structure Quadrilateral :=
  (A B C D : EuclideanSpace ℝ (Fin 2))
  (has_inscribed_circle : sorry) -- Placeholder for the inscribed circle condition

/-- The intersection point of the diagonals of a quadrilateral. -/
noncomputable def diagonal_intersection (q : Quadrilateral) : EuclideanSpace ℝ (Fin 2) :=
  sorry -- Placeholder for the actual intersection calculation

/-- The circumradius of a triangle given by three points. -/
noncomputable def circumradius (p1 p2 p3 : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  sorry -- Placeholder for the actual circumradius calculation

/-- Theorem about the circumradii of triangles formed by the diagonals of a quadrilateral. -/
theorem quadrilateral_circumradii 
  (q : Quadrilateral) 
  (P : EuclideanSpace ℝ (Fin 2)) 
  (h_P : P = diagonal_intersection q) 
  (R1 R2 R3 R4 : ℝ) 
  (h_R1 : R1 = circumradius q.A P q.B)
  (h_R2 : R2 = circumradius q.B P q.C)
  (h_R3 : R3 = circumradius q.C P q.D)
  (h_R4 : R4 = circumradius q.D P q.A)
  (h_R1_val : R1 = 31)
  (h_R2_val : R2 = 24)
  (h_R3_val : R3 = 12) :
  R4 = 19 := by
    sorry -- Placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_circumradii_l441_44114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_of_8_pow_12_times_3_pow_25_is_23_l441_44126

noncomputable def digits_of_8_pow_12_times_3_pow_25 : ℕ :=
  let n : ℝ := (8 ^ 12) * (3 ^ 25)
  let log_n := Real.log n / Real.log 10
  ⌊log_n⌋.toNat + 1

theorem digits_of_8_pow_12_times_3_pow_25_is_23 :
  digits_of_8_pow_12_times_3_pow_25 = 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_of_8_pow_12_times_3_pow_25_is_23_l441_44126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_circle_intersection_l441_44146

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    and eccentricity √5, if a line asymptotic to C intersects the circle
    (x-2)² + (y-3)² = 1 at points A and B, then the length of |AB| is 4√5/5. -/
theorem hyperbola_asymptote_circle_intersection
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (C : Set (ℝ × ℝ))
  (hC : C = {p : ℝ × ℝ | p.1^2/a^2 - p.2^2/b^2 = 1})
  (hecc : Real.sqrt (1 + b^2/a^2) = Real.sqrt 5)
  (circle : Set (ℝ × ℝ))
  (hcircle : circle = {p : ℝ × ℝ | (p.1-2)^2 + (p.2-3)^2 = 1})
  (asymptote : Set (ℝ × ℝ))
  (hasymptote : asymptote ⊆ {p : ℝ × ℝ | p.1^2/a^2 - p.2^2/b^2 = 0})
  (A B : ℝ × ℝ)
  (hAB : A ∈ asymptote ∩ circle ∧ B ∈ asymptote ∩ circle ∧ A ≠ B) :
  ‖A - B‖ = 4 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_circle_intersection_l441_44146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_odd_zeros_l441_44132

/-- If y = cos(x + φ) is an odd function, then all its zeros are of the form kπ, where k ∈ ℤ -/
theorem cos_odd_zeros (φ : ℝ) :
  (∀ x, Real.cos (x + φ) = -Real.cos (-x + φ)) →
  (∀ x, Real.cos (x + φ) = 0 ↔ ∃ k : ℤ, x = k * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_odd_zeros_l441_44132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_for_composite_polynomial_l441_44143

def is_composite (n : ℕ) : Prop := ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a * b = n

theorem smallest_a_for_composite_polynomial :
  (∀ (x : ℤ), is_composite (Int.natAbs (x^4) + 12^2)) ∧
  (∀ (a : ℕ), 8 < a → a < 12 → ∃ (x : ℤ), ¬is_composite (Int.natAbs (x^4) + a^2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_for_composite_polynomial_l441_44143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_one_minus_x_squared_l441_44137

theorem integral_sqrt_one_minus_x_squared :
  ∫ x in (-1:ℝ)..1, Real.sqrt (1 - x^2) = π / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_one_minus_x_squared_l441_44137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_lines_l441_44134

-- Define the polar equation
noncomputable def polar_equation (θ : Real) : Real :=
  1 / (1 + Real.sin θ)

-- Define the curve in Cartesian coordinates
def curve (x y : Real) : Prop :=
  y = 0 ∨ y = -2

-- Theorem statement
theorem polar_to_cartesian_lines :
  ∀ (x y θ : Real),
    x = polar_equation θ * Real.cos θ ∧
    y = polar_equation θ * Real.sin θ →
    curve x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_lines_l441_44134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_condition_l441_44131

-- Define the function f(x)
noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  ((k + 1) * x^2 + (k + 3) * x + (2 * k - 8)) / ((2 * k - 1) * x^2 + (k + 1) * x + (k - 4))

-- Define the domain of f(x)
def domain (k : ℝ) : Set ℝ :=
  {x | (2 * k - 1) * x^2 + (k + 1) * x + (k - 4) ≠ 0}

-- State the theorem
theorem f_positive_condition (k : ℝ) :
  (∀ x ∈ domain k, f k x > 0) ↔ 
  (k = 1 ∨ k > (15 + 16 * Real.sqrt 2) / 7 ∨ k < (15 - 16 * Real.sqrt 2) / 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_condition_l441_44131
