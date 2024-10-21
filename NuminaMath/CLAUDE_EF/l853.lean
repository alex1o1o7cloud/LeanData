import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_lower_side_length_l853_85353

theorem trapezoid_lower_side_length 
  (area : ℝ) 
  (height : ℝ) 
  (upper_side_difference : ℝ) 
  (h1 : area = 100.62) 
  (h2 : height = 5.2) 
  (h3 : upper_side_difference = 3.4) : 
  ∃ (lower_side : ℝ), 
    area = (1 / 2) * (lower_side + (lower_side + upper_side_difference)) * height ∧ 
    abs (lower_side - 17.65) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_lower_side_length_l853_85353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_quadratic_sum_of_roots_specific_equation_l853_85316

noncomputable def quadraticRoots (a b c : ℝ) : ℝ × ℝ :=
  let d := b^2 - 4*a*c
  let r1 := (-b + Real.sqrt d) / (2*a)
  let r2 := (-b - Real.sqrt d) / (2*a)
  (r1, r2)

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let (r₁, r₂) := quadraticRoots a b c
  r₁ + r₂ = -b / a := by sorry

theorem sum_of_roots_specific_equation :
  let (r₁, r₂) := quadraticRoots 1 1992 (-1993)
  r₁ + r₂ = -1992 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_quadratic_sum_of_roots_specific_equation_l853_85316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_and_coefficient_sum_l853_85324

-- Define the set of circle centers
def circleCenters : Set (ℤ × ℤ) :=
  {(x, y) | 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3}

-- Define the region S as the union of unit circles centered at these points
def S : Set (ℝ × ℝ) :=
  {(x, y) | ∃ (cx cy : ℤ), (cx, cy) ∈ circleCenters ∧ (x - cx)^2 + (y - cy)^2 ≤ 1}

-- Define the line m
def m : ℝ → ℝ
  | x => 2 * x

-- State the theorem
theorem equal_area_division_and_coefficient_sum :
  (∃ (A : Set (ℝ × ℝ)), A ⊆ S ∧
    (∀ (x y : ℝ), (x, y) ∈ A ↔ (x, y) ∈ S ∧ y ≤ m x)) ∧
  2^2 + 1^2 + 0^2 = 5 := by
  sorry

#check equal_area_division_and_coefficient_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_and_coefficient_sum_l853_85324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l853_85336

def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

theorem cubic_function_properties (a b c : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f a b c x = f a b c x) →
  f a b c 0 = 0 →
  (deriv (f a b c)) 1 = -1 →
  (deriv (f a b c)) (-1) = -1 →
  (∃ a' b' c', 
    (∀ x, f a' b' c' x = x^3 - 4*x) ∧
    (∃! x y : ℝ, x ≠ y ∧ (deriv (f a' b' c')) x = 0 ∧ (deriv (f a' b' c')) y = 0) ∧
    (∃ xmax xmin : ℝ, 
      (∀ x ∈ Set.Icc (-2 : ℝ) 2, f a' b' c' x ≤ f a' b' c' xmax) ∧
      (∀ x ∈ Set.Icc (-2 : ℝ) 2, f a' b' c' xmin ≤ f a' b' c' x) ∧
      f a' b' c' xmax + f a' b' c' xmin = 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l853_85336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_approximately_17_78_l853_85331

/-- The equation that defines the problem -/
def problem_equation (n : ℝ) : Prop :=
  0.1 * 0.3 * ((4^2 / 100) * ((50 / 100) * n)^3 / 4)^2 = 90

/-- The theorem stating that the solution to the equation is approximately 17.78 -/
theorem solution_approximately_17_78 :
  ∃ n : ℝ, problem_equation n ∧ |n - 17.78| < 0.01 := by
  sorry

#check solution_approximately_17_78

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_approximately_17_78_l853_85331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l853_85397

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.exp (x - 1) + 1

-- Define the proposed inverse function
noncomputable def g (x : ℝ) : ℝ := 1 + Real.log (x - 1)

-- Theorem statement
theorem inverse_function_proof :
  (∀ x : ℝ, x > 1 → f (g x) = x) ∧
  (∀ y : ℝ, g (f y) = y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l853_85397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_area_is_24_percent_l853_85304

/-- Represents a square flag with a symmetric cross and central circle -/
structure FlagWithCross where
  side : ℝ
  cross_and_circle_ratio : ℝ
  circle_ratio : ℝ
  cross_and_circle_ratio_valid : cross_and_circle_ratio = 0.44
  circle_ratio_valid : circle_ratio = 0.20

/-- Calculates the green area ratio of the flag -/
def green_area_ratio (flag : FlagWithCross) : ℝ :=
  flag.cross_and_circle_ratio - flag.circle_ratio

/-- Theorem stating that the green area ratio is 24% of the total flag area -/
theorem green_area_is_24_percent (flag : FlagWithCross) :
  green_area_ratio flag = 0.24 := by
  unfold green_area_ratio
  simp [flag.cross_and_circle_ratio_valid, flag.circle_ratio_valid]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_area_is_24_percent_l853_85304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l853_85359

/-- Calculates the time (in seconds) for a train to cross a bridge -/
noncomputable def time_to_cross_bridge (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem stating that a train of length 170 meters traveling at 45 km/hr 
    takes 30 seconds to cross a bridge of length 205 meters -/
theorem train_crossing_bridge :
  time_to_cross_bridge 170 45 205 = 30 := by
  -- Unfold the definition of time_to_cross_bridge
  unfold time_to_cross_bridge
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l853_85359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_neg_reals_l853_85307

-- Define the function f(x) = 2^x
noncomputable def f (x : ℝ) : ℝ := 2^x

-- State the theorem
theorem f_increasing_on_neg_reals : 
  ∀ x y : ℝ, x < y → x < 0 → y < 0 → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_neg_reals_l853_85307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_c_proof_l853_85332

/-- The function f(x) = x^2 - 5x + c -/
noncomputable def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 5*x + c

/-- 1 is in the range of f(x) -/
def one_in_range (c : ℝ) : Prop := ∃ x, f c x = 1

/-- The largest value of c such that 1 is in the range of f(x) -/
noncomputable def largest_c : ℝ := 29/4

theorem largest_c_proof :
  (∀ c > largest_c, ¬(one_in_range c)) ∧
  (one_in_range largest_c) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_c_proof_l853_85332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bound_exists_l853_85395

noncomputable def a : ℕ → ℝ
  | 0 => 1  -- Add case for 0
  | 1 => 1
  | n + 1 => Real.sqrt (a n ^ 2 + 1 / a n)

theorem sequence_bound_exists : 
  ∃ α : ℝ, α > 0 ∧ ∀ n : ℕ, n ≥ 1 → 1/2 ≤ a n / n^α ∧ a n / n^α ≤ 2 := by
  -- We'll use α = 1/3
  use 1/3
  constructor
  · -- Prove α > 0
    norm_num
  · -- Prove the main inequality
    intro n hn
    sorry  -- Placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bound_exists_l853_85395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_power_speed_l853_85351

variable (A S ρ v₀ : ℝ)
variable (hA : A > 0)
variable (hS : S > 0)
variable (hρ : ρ > 0)
variable (hv₀ : v₀ > 0)

noncomputable def F (v : ℝ) : ℝ := (A * S * ρ * (v₀ - v)^2) / 2

noncomputable def N (v : ℝ) : ℝ := F A S ρ v₀ v * v

theorem max_power_speed (A S ρ v₀ : ℝ) (hA : A > 0) (hS : S > 0) (hρ : ρ > 0) (hv₀ : v₀ > 0) :
  ∃ (v : ℝ), v = v₀ / 3 ∧ ∀ (u : ℝ), N A S ρ v₀ v ≥ N A S ρ v₀ u := by
  sorry

#check max_power_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_power_speed_l853_85351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_theorem_l853_85378

/-- A regular quadrilateral pyramid with an inscribed sphere -/
structure RegularPyramidWithSphere where
  /-- The slant height (apothem) of the pyramid -/
  a : ℝ
  /-- The sphere touches the base and all lateral faces -/
  sphere_touches_all_faces : Prop
  /-- The sphere divides the height of the pyramid in the ratio 1:3 from the apex -/
  height_ratio : Prop

/-- The volume of the pyramid -/
noncomputable def pyramid_volume (p : RegularPyramidWithSphere) : ℝ := 48 * p.a^3 / 125

/-- Theorem stating the volume of the pyramid -/
theorem pyramid_volume_theorem (p : RegularPyramidWithSphere) :
  pyramid_volume p = 48 * p.a^3 / 125 := by
  -- Unfold the definition of pyramid_volume
  unfold pyramid_volume
  -- The equation is now trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_theorem_l853_85378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orangeade_price_theorem_l853_85376

/-- Represents the price and volume data for orangeade sales over two days -/
structure OrangeadeData where
  orange_juice : ℚ  -- Amount of orange juice used each day
  water_day1 : ℚ    -- Amount of water used on day 1
  water_day2 : ℚ    -- Amount of water used on day 2
  price_day2 : ℚ    -- Price per glass on day 2

/-- Calculates the price per glass on the first day given the orangeade data -/
def calculate_price_day1 (data : OrangeadeData) : ℚ :=
  (data.price_day2 * (data.orange_juice + data.water_day2)) / (data.orange_juice + data.water_day1)

/-- Theorem stating the conditions and the result to be proved -/
theorem orangeade_price_theorem (data : OrangeadeData) 
  (h1 : data.water_day1 = data.orange_juice)          -- Equal amounts of orange juice and water on day 1
  (h2 : data.water_day2 = 2 * data.water_day1)        -- Twice the amount of water on day 2
  (h3 : data.price_day2 = 2/5)                        -- Price on day 2 is $0.40
  : calculate_price_day1 data = 3/5 := by             -- Price on day 1 is $0.60
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orangeade_price_theorem_l853_85376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_elements_for_unique_determination_l853_85335

/-- A matrix satisfying the property that the sum of opposite corners of any rectangle is equal -/
def SpecialMatrix (m n : ℕ) := Matrix (Fin m) (Fin n) ℚ

/-- The property that the sum of opposite corners of any rectangle is equal -/
def has_equal_corner_sums (A : SpecialMatrix m n) : Prop :=
  ∀ (i j k l : ℕ) (hi : i < m) (hj : j < n) (hk : k < m) (hl : l < n),
    A ⟨i, hi⟩ ⟨j, hj⟩ + A ⟨k, hk⟩ ⟨l, hl⟩ = A ⟨i, hi⟩ ⟨l, hl⟩ + A ⟨k, hk⟩ ⟨j, hj⟩

/-- The theorem stating that at least (n+m-1) elements are required to uniquely determine the matrix -/
theorem min_elements_for_unique_determination (m n : ℕ) :
  ∀ (A : SpecialMatrix m n), has_equal_corner_sums A →
  ∀ (S : Finset ((Fin m) × (Fin n))), (∀ (i : Fin m) (j : Fin n), (i, j) ∉ S → A i j = 0) →
  S.card ≥ n + m - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_elements_for_unique_determination_l853_85335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l853_85389

noncomputable def f : ℕ → ℝ → ℝ
  | 0, x => 8
  | n + 1, x => Real.sqrt (x^2 + 6 * f n x)

theorem unique_solution (n : ℕ) (x : ℝ) (h : x ≥ 0) :
  f n x = 2 * x ↔ x = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l853_85389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_focus_coincidence_l853_85342

/-- The x-coordinate of the focus of a parabola y^2 = ax -/
noncomputable def parabola_focus (a : ℝ) : ℝ := a / 4

/-- The x-coordinate of the right focus of a hyperbola x^2/a - y^2/b = 1 -/
noncomputable def hyperbola_right_focus (a b : ℝ) : ℝ := Real.sqrt (a + b)

/-- 
If the focus of the parabola y^2 = 8x coincides with one focus of the hyperbola x^2/3 - y^2/n = 1,
then n = 1
-/
theorem parabola_hyperbola_focus_coincidence :
  ∀ n : ℝ, parabola_focus 8 = hyperbola_right_focus 3 n → n = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_focus_coincidence_l853_85342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l853_85379

noncomputable def original_function (x : ℝ) : ℝ := Real.sin (4 * x - Real.pi / 6)

noncomputable def transformed_function (x : ℝ) : ℝ := Real.sin (2 * (x + Real.pi / 4) - Real.pi / 6)

theorem axis_of_symmetry :
  ∃ (k : ℤ), transformed_function (Real.pi / 12 + k * Real.pi / 2) = transformed_function (Real.pi / 12 - k * Real.pi / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l853_85379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_eq_sin_l853_85391

open Real

/-- Recursive definition of the function sequence -/
noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => cos
  | n + 1 => deriv (f n)

/-- The 2016th function in the sequence equals sine -/
theorem f_2016_eq_sin : f 2016 = sin := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_eq_sin_l853_85391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_super_number_properties_l853_85329

/-- Super number is a sequence of digits with infinitely many digits to the left -/
def SuperNumber := ℕ → Fin 10

/-- Addition of super numbers -/
def add (a b : SuperNumber) : SuperNumber :=
  λ i ↦ (a i + b i).val % 10

/-- Multiplication of super numbers -/
noncomputable def mul (a b : SuperNumber) : SuperNumber :=
  sorry  -- Definition of multiplication for super numbers is complex and left as sorry

/-- Zero super number (all digits are zero) -/
def zero : SuperNumber := λ _ ↦ 0

/-- One super number (all digits are zero except the last one which is 1) -/
def one : SuperNumber := λ i ↦ if i = 0 then 1 else 0

theorem super_number_properties :
  (∀ a : SuperNumber, ∃ b : SuperNumber, add a b = zero) ∧
  (∀ a b : SuperNumber, mul a b = one ↔ a = one ∧ b = one) ∧
  (∀ a b : SuperNumber, mul a b = zero → a = zero ∨ b = zero) := by
  sorry

#check super_number_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_super_number_properties_l853_85329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_implies_sum_l853_85384

noncomputable section

/-- The function f(x) reaching an extreme value of 0 at x = 1 -/
def f (a b x : ℝ) : ℝ := (a * x^3) / 3 - b * x^2 + a^2 * x - 1/3

/-- The derivative of f(x) -/
def f' (a b x : ℝ) : ℝ := a * x^2 - 2 * b * x + a^2

theorem extreme_value_implies_sum (a b : ℝ) :
  (f a b 1 = 0) →  -- f(1) = 0
  (f' a b 1 = 0) →  -- f'(1) = 0 (condition for extreme value)
  (a + b = -7/9) := by
    sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_implies_sum_l853_85384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C₁_equation_min_distance_C₁_to_C₂_l853_85383

noncomputable section

-- Define the lines l₁ and l₂
def l₁ (t k : ℝ) : ℝ × ℝ := (t - Real.sqrt 3, k * t)
def l₂ (m k : ℝ) : ℝ × ℝ := (Real.sqrt 3 - m, m / (3 * k))

-- Define the curve C₁
def C₁ : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 3 + p.2^2 = 1}

-- Define the curve C₂ in polar coordinates
def C₂ : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ (θ : ℝ), p.1 = 4 * Real.sqrt 2 * Real.cos θ / Real.sin (θ + Real.pi/4) ∧
                                               p.2 = 4 * Real.sqrt 2 * Real.sin θ / Real.sin (θ + Real.pi/4)}

theorem curve_C₁_equation : 
  ∀ (k : ℝ), k ≠ 0 → 
  (∃ (t m : ℝ), l₁ t k = l₂ m k) → 
  C₁ = {p : ℝ × ℝ | p.1^2 / 3 + p.2^2 = 1} :=
by
  sorry

theorem min_distance_C₁_to_C₂ : 
  ∀ (p : ℝ × ℝ), p ∈ C₁ → 
  (∃ (q : ℝ × ℝ), q ∈ C₂ ∧ 
    ∀ (r : ℝ × ℝ), r ∈ C₂ → Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ Real.sqrt ((p.1 - r.1)^2 + (p.2 - r.2)^2)) →
  ∃ (q : ℝ × ℝ), q ∈ C₂ ∧ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 3 * Real.sqrt 2 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C₁_equation_min_distance_C₁_to_C₂_l853_85383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_possible_l853_85308

/-- Represents a location --/
inductive Location
| Maikop
| Belorechensk

/-- Represents a mode of transportation --/
inductive TransportMode
| Walking
| Cycling

/-- Represents a person --/
structure Person where
  name : String
  current_location : Location
  destination : Location
  transport_mode : TransportMode

/-- Represents the state of the journey --/
structure JourneyState where
  people : List Person
  bicycle_location : Location
  time_elapsed : ℚ

def distance : ℚ := 24

def min_walking_speed : ℚ := 6

def min_cycling_speed : ℚ := 18

def max_journey_time : ℚ := 2 + 40 / 60

/-- The main theorem to prove --/
theorem journey_possible (initial_state : JourneyState) : 
  ∃ (final_state : JourneyState), 
    (∀ p ∈ final_state.people, p.current_location = p.destination) ∧ 
    final_state.time_elapsed ≤ max_journey_time := by
  sorry

#check journey_possible

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_possible_l853_85308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_theorem_l853_85319

/-- The hyperbola C: x^2 - y^2/3 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

/-- The line l passing through point P(0,1) with slope k -/
def line (k x y : ℝ) : Prop := y = k * x + 1

/-- The right focus F2 of the hyperbola -/
def right_focus : ℝ × ℝ := (2, 0)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The theorem statement -/
theorem hyperbola_intersection_theorem (k : ℝ) :
  (∃ A B : ℝ × ℝ, 
    hyperbola A.1 A.2 ∧ 
    hyperbola B.1 B.2 ∧ 
    line k A.1 A.2 ∧ 
    line k B.1 B.2 ∧ 
    distance A right_focus + distance B right_focus = 6) →
  (k ∈ Set.Ioo (-2 : ℝ) (-Real.sqrt 3) ∪ 
   Set.Ioo (-Real.sqrt 3) (Real.sqrt 3) ∪ 
   Set.Ioo (Real.sqrt 3) 2) ∧
  (k = 1 ∨ k = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_theorem_l853_85319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_factorial_l853_85368

theorem divisibility_of_factorial (n : ℕ+) : ∃ k : ℕ, (3 * n.val).factorial = k * (6^n.val * n.val.factorial) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_factorial_l853_85368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_extension_l853_85303

-- Define an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the function for x < 0
noncomputable def f_neg (x : ℝ) : ℝ :=
  3 * Real.sin x + 4 * Real.cos x + 1

-- Theorem statement
theorem odd_function_extension :
  ∀ f : ℝ → ℝ,
  odd_function f →
  (∀ x < 0, f x = f_neg x) →
  ∀ x > 0, f x = 3 * Real.sin x - 4 * Real.cos x - 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_extension_l853_85303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_theorem_l853_85344

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the circle
def my_circle (x y r : ℝ) : Prop := (x - 1)^2 + (y - r)^2 = r^2

-- Define the theorem
theorem circle_radius_theorem :
  ∃! (r : ℝ),
    (∃! (x y : ℝ), parabola x y ∧ my_circle x y r) ∧  -- Exactly one common point
    (my_circle 1 0 r) ∧                               -- Circle tangent to x-axis at (1,0)
    (r > 0) ∧                                         -- Radius is positive
    (r = (4 * Real.sqrt 3) / 9) :=                    -- Radius value
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_theorem_l853_85344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_with_same_foci_and_minor_axis_l853_85346

/-- Given an ellipse with equation 9x^2 + 4y^2 = 36, prove that the ellipse
    with equation x^2/25 + y^2/20 = 1 has the same foci and a minor axis length of 4√5 -/
theorem ellipse_with_same_foci_and_minor_axis :
  ∃ (given_a given_b new_a new_b : ℝ),
    (∀ x y : ℝ, 9 * x^2 + 4 * y^2 = 36 ↔ x^2 / given_a^2 + y^2 / given_b^2 = 1) ∧
    (∀ x y : ℝ, x^2 / 25 + y^2 / 20 = 1 ↔ x^2 / new_a^2 + y^2 / new_b^2 = 1) ∧
    (∃ c : ℝ, c^2 = given_a^2 - given_b^2 ∧ c^2 = new_a^2 - new_b^2) ∧
    2 * new_b = 4 * Real.sqrt 5 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_with_same_foci_and_minor_axis_l853_85346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_solid_figures_l853_85310

/-- A shape is either a solid figure or not -/
inductive Shape
  | Circle
  | Square
  | Cone
  | Cuboid
  | LineSegment
  | Sphere
  | TriangularPrism
  | RightAngledTriangle

/-- Predicate to determine if a shape is a solid figure -/
def isSolidFigure : Shape → Bool
  | Shape.Cone => true
  | Shape.Cuboid => true
  | Shape.Sphere => true
  | Shape.TriangularPrism => true
  | _ => false

/-- The list of all shapes mentioned in the problem -/
def allShapes : List Shape :=
  [Shape.Circle, Shape.Square, Shape.Cone, Shape.Cuboid, Shape.LineSegment,
   Shape.Sphere, Shape.TriangularPrism, Shape.RightAngledTriangle]

/-- Theorem stating that the number of solid figures among the given shapes is 4 -/
theorem count_solid_figures :
  (allShapes.filter isSolidFigure).length = 4 := by
  sorry

#eval (allShapes.filter isSolidFigure).length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_solid_figures_l853_85310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sum_l853_85388

/-- Arithmetic sequence with first term 1 and common difference d -/
def arithmetic_seq (d : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => arithmetic_seq d n + d

/-- Geometric sequence with first term 1 and common ratio r -/
def geometric_seq (r : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => r * geometric_seq r n

/-- Sum of corresponding terms in arithmetic and geometric sequences -/
def c_seq (d r : ℕ) (n : ℕ) : ℕ := arithmetic_seq d n + geometric_seq r n

theorem arithmetic_geometric_sum (d r k : ℕ) :
  (∀ n : ℕ, n > 1 → arithmetic_seq d n > arithmetic_seq d (n - 1)) →
  (∀ n : ℕ, n > 1 → geometric_seq r n > geometric_seq r (n - 1)) →
  c_seq d r (k - 1) = 50 →
  c_seq d r (k + 1) = 200 →
  c_seq d r k = 90 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sum_l853_85388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_minimum_perimeter_triangle_side_c_l853_85360

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle given two sides and the included angle -/
noncomputable def triangleArea (t : Triangle) : ℝ := (1/2) * t.b * t.c * Real.sin t.A

/-- The perimeter of a triangle -/
def trianglePerimeter (t : Triangle) : ℝ := t.a + t.b + t.c

theorem triangle_minimum_perimeter (t : Triangle) 
  (h1 : t.a = 5)
  (h2 : Real.sin t.A = Real.sqrt 5 / 5)
  (h3 : triangleArea t = Real.sqrt 5) :
  ∃ (l : ℝ), l = 2 * Real.sqrt 10 + 5 ∧ ∀ (t' : Triangle), t'.a = 5 → triangleArea t' = Real.sqrt 5 → trianglePerimeter t' ≥ l := by
  sorry

theorem triangle_side_c (t : Triangle)
  (h1 : t.a = 5)
  (h2 : Real.sin t.A = Real.sqrt 5 / 5)
  (h3 : Real.cos t.B = 3/5) :
  t.c = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_minimum_perimeter_triangle_side_c_l853_85360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_second_quadrant_l853_85392

/-- An angle in the fourth quadrant -/
def fourth_quadrant_angle (θ : ℝ) : Prop :=
  3 * Real.pi / 2 < θ ∧ θ < 2 * Real.pi

/-- A point in the second quadrant -/
def second_quadrant_point (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

/-- The main theorem -/
theorem point_in_second_quadrant (θ : ℝ) :
  fourth_quadrant_angle θ →
  second_quadrant_point (Real.sin (Real.sin θ)) (Real.cos (Real.sin θ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_second_quadrant_l853_85392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygonal_chain_theorem_l853_85341

/-- Represents a polygonal chain inside a unit square --/
structure PolygonalChain where
  -- The chain is represented as a list of points
  points : List (Real × Real)
  -- All points are inside the unit square
  points_in_square : ∀ p, p ∈ points → 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1
  -- Each line parallel to a side intersects the chain in no more than one point
  single_intersection : ∀ x y : Real, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 →
    (∃! p, p ∈ points ∧ p.1 = x) ∧ (∃! p, p ∈ points ∧ p.2 = y)

/-- The length of a polygonal chain --/
noncomputable def chainLength (chain : PolygonalChain) : Real :=
  sorry

/-- The main theorem --/
theorem polygonal_chain_theorem (chain : PolygonalChain) :
  chainLength chain < 2 ∧
  ∀ l : Real, l < 2 → ∃ chain' : PolygonalChain, chainLength chain' = l := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygonal_chain_theorem_l853_85341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l853_85380

noncomputable def f (x : ℝ) := 2 * Real.sin x * (Real.sin x + Real.cos x)

theorem f_properties :
  ∃ (period : ℝ) (monotonic_intervals : Set (Set ℝ)) (x₁ x₂ : ℝ),
    (∀ x, f (x + period) = f x) ∧ 
    (period > 0) ∧
    (∀ other_period, other_period > 0 → (∀ x, f (x + other_period) = f x) → period ≤ other_period) ∧
    (∀ i ∈ monotonic_intervals, ∃ k : ℤ, i = Set.Icc (-Real.pi/8 + k*Real.pi) (3*Real.pi/8 + k*Real.pi)) ∧
    (∀ i ∈ monotonic_intervals, ∀ x y, x ∈ i → y ∈ i → x < y → f x < f y) ∧
    (x₁ ∈ Set.Icc 0 (Real.pi/2)) ∧
    (x₂ ∈ Set.Icc 0 (Real.pi/2)) ∧
    (f x₁ = 2) ∧
    (f x₂ = 2) ∧
    (Real.sin (2*x₁ + 2*x₂) = -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l853_85380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_implies_a_value_l853_85333

def f (a : ℝ) (x : ℝ) : ℝ := (2 - a * x)^6

theorem coefficient_implies_a_value (a : ℝ) :
  (∃ c : ℝ, c = -160 ∧ 
   ∀ x : ℝ, f a x = c * x^3 + (f a x - c * x^3)) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_implies_a_value_l853_85333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_squared_norms_l853_85305

theorem vector_sum_squared_norms (a b m : ℝ × ℝ) : 
  m = (4, 8) →
  m = ((a.1 + b.1) / 2, (a.2 + b.2) / 2) →
  a.1 * b.1 + a.2 * b.2 = 8 →
  (a.1^2 + a.2^2) + (b.1^2 + b.2^2) = 304 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_squared_norms_l853_85305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_specific_pyramid_l853_85311

/-- A regular octagon -/
structure RegularOctagon where
  side_length : ℝ

/-- A right pyramid with a regular octagon base -/
structure RightPyramid where
  base : RegularOctagon
  height : ℝ

/-- An equilateral triangle -/
structure EquilateralTriangle where
  side_length : ℝ

/-- Calculate the volume of a right pyramid with a regular octagon base -/
noncomputable def volume_right_pyramid (p : RightPyramid) : ℝ :=
  (1 / 3) * (2 * (1 + Real.sqrt 2) * p.base.side_length ^ 2) * p.height

/-- The main theorem -/
theorem volume_specific_pyramid :
  ∃ (p : RightPyramid) (t : EquilateralTriangle),
    t.side_length = 10 ∧
    p.base.side_length = 5 * Real.sqrt 3 ∧
    p.height = 5 * Real.sqrt 3 ∧
    volume_right_pyramid p = 750 * Real.sqrt 3 + 750 * Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_specific_pyramid_l853_85311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_equivalence_l853_85362

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(2x+1)
def domain_f_2x_plus_1 : Set ℝ := Set.Icc (-3) 3

-- Define the domain of f(x-1)
def domain_f_x_minus_1 : Set ℝ := Set.Icc (-4) 8

-- Theorem statement
theorem domain_equivalence :
  (∀ x, x ∈ domain_f_2x_plus_1 ↔ (2*x + 1) ∈ Set.Icc (-5) 7) →
  (∀ x, x ∈ domain_f_x_minus_1 ↔ (x - 1) ∈ Set.Icc (-5) 7) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_equivalence_l853_85362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greeting_card_envelope_fit_l853_85317

theorem greeting_card_envelope_fit (card_area envelope_area : ℝ) 
  (h_card_area : card_area = 144)
  (h_envelope_area : envelope_area = 180) : 
  Real.sqrt card_area > Real.sqrt (3 * envelope_area / 16) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greeting_card_envelope_fit_l853_85317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_l853_85349

noncomputable def spherical_to_cartesian (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem circle_radius (θ : ℝ) :
  let (x, y, z) := spherical_to_cartesian 2 θ (π / 4)
  (x^2 + y^2 : ℝ) = 2 ∧ z = Real.sqrt 2 := by
  sorry

#check circle_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_l853_85349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_filled_is_three_fourths_l853_85312

/-- Represents the filling capacity of a pickup truck -/
structure PickupTruck where
  /-- The rate at which the truck fills water cans (in gallons per hour) -/
  fill_rate : ℝ

/-- Represents a water can -/
structure WaterCan where
  /-- The full capacity of the can in gallons -/
  full_capacity : ℝ

/-- The fraction of a water can's capacity that a pickup truck fills in three hours -/
noncomputable def fraction_filled (truck : PickupTruck) (can : WaterCan) : ℝ :=
  (3 * truck.fill_rate) / can.full_capacity

theorem fraction_filled_is_three_fourths (truck : PickupTruck) (can : WaterCan) :
  (20 * fraction_filled truck can * can.full_capacity = 3 * truck.fill_rate) →
  (25 * can.full_capacity = 5 * truck.fill_rate) →
  fraction_filled truck can = 3/4 := by
  sorry

#check fraction_filled_is_three_fourths

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_filled_is_three_fourths_l853_85312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_construction_correct_l853_85366

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  center : Point
  realAxis : Line
  point1 : Point
  point2 : Point

/-- Represents an asymptote of a hyperbola -/
structure Asymptote where
  line : Line

/-- Function to construct asymptotes of a hyperbola -/
noncomputable def constructAsymptotes (h : Hyperbola) : Asymptote × Asymptote :=
  sorry

/-- Theorem stating the correctness of the asymptote construction -/
theorem asymptote_construction_correct (h : Hyperbola) :
  let (a1, a2) := constructAsymptotes h
  let O := h.center
  let x := h.realAxis
  let P1 := h.point1
  let P2 := h.point2
  true :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_construction_correct_l853_85366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_projection_theorem_l853_85369

noncomputable def vector_a : ℝ × ℝ := (4, 3)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (x, 1)

noncomputable def projection (a b : ℝ × ℝ) : ℝ :=
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2)

theorem vector_projection_theorem (x : ℝ) :
  projection vector_a (vector_b x) = 5 * Real.sqrt 2 / 2 →
  (∃ θ : ℝ, θ = π / 4 ∧ 
   Real.cos θ * Real.sqrt (vector_a.1^2 + vector_a.2^2) = 5 * Real.sqrt 2 / 2) ∧
  x = 1 / 7 := by
  sorry

#check vector_projection_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_projection_theorem_l853_85369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_smallest_period_is_pi_l853_85350

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x ^ 2 + Real.sin x * Real.cos x - Real.sqrt 3 / 2

/-- The smallest positive period of f(x) -/
noncomputable def smallest_positive_period : ℝ := Real.pi

/-- Theorem stating that the smallest positive period of f(x) is π -/
theorem f_period : 
  (∀ x : ℝ, f (x + smallest_positive_period) = f x) ∧
  (∀ p : ℝ, 0 < p → p < smallest_positive_period → ∃ y : ℝ, f (y + p) ≠ f y) :=
by
  sorry

/-- Proof that the smallest positive period is indeed π -/
theorem smallest_period_is_pi : smallest_positive_period = Real.pi :=
by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_smallest_period_is_pi_l853_85350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_curve_l853_85322

-- Define the curve in polar coordinates
def polar_curve (p θ : ℝ) : Prop := p * (Real.cos θ)^2 = 4 * Real.sin θ

-- Define the focus in polar coordinates
noncomputable def focus : ℝ × ℝ := (1, Real.pi / 2)

-- Theorem statement
theorem focus_of_curve :
  ∀ p θ : ℝ, p ≥ 0 → 0 ≤ θ → θ < 2 * Real.pi → polar_curve p θ → 
  ∃ r φ : ℝ, (r, φ) = focus ∧ polar_curve r φ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_curve_l853_85322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jogger_distance_l853_85347

/-- A jogger's actual and hypothetical speeds, and the additional distance covered at the higher speed -/
structure JoggerData where
  actual_speed : ℝ
  hypothetical_speed : ℝ
  additional_distance : ℝ

/-- The actual distance jogged given the jogger's data -/
noncomputable def actual_distance (data : JoggerData) : ℝ :=
  (data.additional_distance * data.actual_speed) / (data.hypothetical_speed - data.actual_speed)

/-- Theorem stating that under the given conditions, the actual distance jogged is 30 km -/
theorem jogger_distance (data : JoggerData)
  (h1 : data.actual_speed = 12)
  (h2 : data.hypothetical_speed = 16)
  (h3 : data.additional_distance = 10) :
  actual_distance data = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jogger_distance_l853_85347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_socks_count_l853_85337

theorem blue_socks_count (total : ℕ) (white_fraction : ℚ) (blue_socks : ℕ) : 
  total = 180 → 
  white_fraction = 2/3 → 
  blue_socks = total - (white_fraction * ↑total).floor → 
  blue_socks = 60 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_socks_count_l853_85337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_bed_fraction_is_sqrt3_over_24_l853_85394

/-- Represents a rectangular park with triangular garden beds and a trapezoidal walkway -/
structure ParkWithGardenBeds where
  trapezoid_short_side : ℝ
  trapezoid_long_side : ℝ
  (short_side_positive : 0 < trapezoid_short_side)
  (long_side_positive : 0 < trapezoid_long_side)
  (short_less_than_long : trapezoid_short_side < trapezoid_long_side)

/-- The fraction of the park occupied by the garden beds -/
noncomputable def garden_bed_fraction (park : ParkWithGardenBeds) : ℝ :=
  Real.sqrt 3 / 24

theorem garden_bed_fraction_is_sqrt3_over_24
  (park : ParkWithGardenBeds)
  (h1 : park.trapezoid_short_side = 20)
  (h2 : park.trapezoid_long_side = 30) :
  garden_bed_fraction park = Real.sqrt 3 / 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_bed_fraction_is_sqrt3_over_24_l853_85394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_product_l853_85306

theorem cos_sum_product (a b : ℝ) : Real.cos (a + b) + Real.cos (a - b) = 2 * Real.cos a * Real.cos b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_product_l853_85306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_complex_l853_85330

/-- Predicate indicating that three complex numbers form an equilateral triangle -/
def IsEquilateralTriangle (z₁ z₂ z₃ : ℂ) : Prop :=
  Complex.abs (z₁ - z₂) = Complex.abs (z₂ - z₃) ∧ 
  Complex.abs (z₂ - z₃) = Complex.abs (z₃ - z₁)

/-- Function to calculate the side length of a triangle formed by three complex numbers -/
noncomputable def TriangleSideLength (z₁ z₂ z₃ : ℂ) : ℝ :=
  Complex.abs (z₁ - z₂)

/-- Given complex numbers z₁, z₂, z₃ forming an equilateral triangle with side length 24
    and |z₁ + z₂ + z₃| = 48, prove that |z₁z₂ + z₂z₃ + z₃z₁| = 768 -/
theorem equilateral_triangle_complex (z₁ z₂ z₃ : ℂ) 
  (h_equilateral : IsEquilateralTriangle z₁ z₂ z₃)
  (h_side_length : TriangleSideLength z₁ z₂ z₃ = 24)
  (h_sum_magnitude : Complex.abs (z₁ + z₂ + z₃) = 48) :
  Complex.abs (z₁ * z₂ + z₂ * z₃ + z₃ * z₁) = 768 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_complex_l853_85330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_players_l853_85390

theorem cricket_players (total : ℕ) (football : ℕ) (neither : ℕ) (both : ℕ) 
  (h1 : total = 250)
  (h2 : football = 160)
  (h3 : neither = 50)
  (h4 : both = 50) :
  ∃ cricket : ℕ, cricket = 90 ∧ 
  cricket = total - neither - (football - both) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_players_l853_85390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_points_l853_85365

/-- The number of intersection points inside an ellipse given n points on it,
    assuming each pair of points forms a chord and no three chords intersect
    at the same point inside the ellipse (excluding endpoints) -/
def number_of_intersection_points (n : ℕ) : ℕ :=
  Nat.choose n 4

/-- The number of intersection points inside an ellipse given 8 points on it -/
theorem ellipse_intersection_points (n : ℕ) (h1 : n = 8) : 
  number_of_intersection_points n = 70 :=
by
  rw [number_of_intersection_points]
  rw [h1]
  rfl

#eval number_of_intersection_points 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_points_l853_85365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_in_range_l853_85399

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x + 2) * Real.exp x

-- State the theorem
theorem inequality_holds_iff_a_in_range :
  ∀ a : ℝ, (∀ x : ℝ, x ≥ 0 → (f x - Real.exp x) / (a * x + 1) ≥ 1) ↔ (0 ≤ a ∧ a ≤ 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_in_range_l853_85399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_second_iteration_l853_85374

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2^x + x - 5

-- Define the interval
def a : ℝ := 1
def b : ℝ := 2

-- Define x₁ as the midpoint of [a, b]
noncomputable def x₁ : ℝ := (a + b) / 2

-- Define x₂ as the midpoint of [x₁, b]
noncomputable def x₂ : ℝ := (x₁ + b) / 2

-- Theorem statement
theorem bisection_second_iteration :
  f a * f b < 0 → f x₁ * f b < 0 → x₂ = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_second_iteration_l853_85374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_with_given_conditions_l853_85386

-- Define the slope
def m : ℚ := 3/4

-- Define the area of the triangle
def area : ℝ := 6

-- Define the general form of the line equation
def line_equation (b : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ y = m * x + b

-- Define the condition for the line to form a triangle with area 6
def forms_triangle_with_area (b : ℝ) : Prop :=
  abs (b * (- (4/3) * b) / 2) = area

-- Theorem statement
theorem line_equation_with_given_conditions :
  ∃ b₁ b₂ : ℝ, b₁ ≠ b₂ ∧
    forms_triangle_with_area b₁ ∧
    forms_triangle_with_area b₂ ∧
    (∀ x y : ℝ, line_equation b₁ x y ↔ 3 * x - 4 * y + 12 = 0) ∧
    (∀ x y : ℝ, line_equation b₂ x y ↔ 3 * x - 4 * y - 12 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_with_given_conditions_l853_85386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l853_85393

/-- Given a triangle ABC where b = 3a and c = 2, the maximum area of the triangle is √2/2 -/
theorem triangle_max_area (a b c : ℝ) (h1 : b = 3*a) (h2 : c = 2) :
  ∃ (A : ℝ), 0 < A ∧ A < Real.pi ∧ 
  (∀ (A' : ℝ), 0 < A' ∧ A' < Real.pi → 
    (1/2) * b * c * Real.sin A' ≤ (1/2) * b * c * Real.sin A) ∧
  (1/2) * b * c * Real.sin A = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l853_85393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_alpha_l853_85387

theorem cos_two_alpha (α β : ℝ) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.cos (α + β) = 5/13) (h4 : Real.sin (α - β) = -4/5) :
  Real.cos (2 * α) = 63/65 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_alpha_l853_85387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_iff_phi_value_l853_85327

/-- The function f(x) defined in terms of φ -/
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.sin (x + φ) - Real.sin (x + 7 * φ)

/-- A function is odd if f(-x) = -f(x) for all x -/
def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g x

/-- The main theorem stating the conditions for f to be an odd function -/
theorem f_is_odd_iff_phi_value (φ : ℝ) :
  is_odd_function (f φ) ↔ (φ = Real.pi / 8 ∨ φ = 3 * Real.pi / 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_iff_phi_value_l853_85327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_f_odd_iff_m_eq_neg_one_l853_85355

/-- The function f(x) defined as 2 / (3^x + 1) + m -/
noncomputable def f (x m : ℝ) : ℝ := 2 / (3^x + 1) + m

/-- f is monotonically decreasing on ℝ for all m ∈ ℝ -/
theorem f_monotone_decreasing (m : ℝ) : 
  ∀ x y, x < y → f x m > f y m := by sorry

/-- f is an odd function if and only if m = -1 -/
theorem f_odd_iff_m_eq_neg_one (m : ℝ) : 
  (∀ x, f (-x) m = -(f x m)) ↔ m = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_f_odd_iff_m_eq_neg_one_l853_85355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_sum_l853_85367

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 + Real.sin (2 * x)

-- State the theorem
theorem range_sum (k : ℝ) (m n : ℝ) (h : k > 0) :
  (∀ x ∈ Set.Icc (-k) k, m ≤ f x ∧ f x ≤ n) ∧
  (∀ y ∈ Set.Icc m n, ∃ x ∈ Set.Icc (-k) k, f x = y) →
  m + n = 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_sum_l853_85367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_sum_l853_85357

/-- A hexagon with equal interior angles -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ
  equal_angles : ∀ i : Fin 6, ∃ angle, angle = 2 * π / 3

/-- The area of a triangle formed by three alternate sides of the hexagon -/
noncomputable def alternate_triangle_area (h : RegularHexagon) (i : Fin 2) : ℝ := sorry

/-- The perimeter of the hexagon -/
noncomputable def perimeter (h : RegularHexagon) : ℝ := sorry

theorem hexagon_perimeter_sum (h : RegularHexagon) 
  (area1 : alternate_triangle_area h 0 = 192 * Real.sqrt 3)
  (area2 : alternate_triangle_area h 1 = 324 * Real.sqrt 3) :
  ∃ (m n p : ℕ), 
    perimeter h = m + n * Real.sqrt p ∧ 
    (∀ q : ℕ, q ≠ p → Real.sqrt p ≠ Real.sqrt q) ∧
    m + n + p = 55 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_sum_l853_85357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_n_l853_85300

/-- A function that checks if n! can be expressed as the product of n - 4 consecutive positive integers -/
def is_valid (n : ℕ) : Prop :=
  ∃ (k : ℕ), Nat.factorial n = (k * (k + 1) * (k + 2) * (k + 3))

/-- Theorem stating that 119 is the largest positive integer satisfying the condition -/
theorem largest_valid_n : 
  (is_valid 119) ∧ (∀ m : ℕ, m > 119 → ¬(is_valid m)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_n_l853_85300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_inequality_l853_85361

noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) * Real.exp x

def g (a x : ℝ) : ℝ := a * x - a

/-- A function is tangent to another function at a point if they intersect at that point and have the same derivative there. -/
def IsTangentLineAt (g f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  g x₀ = f x₀ ∧ deriv g x₀ = deriv f x₀

theorem tangent_line_and_inequality (a : ℝ) :
  (∃ x₀ : ℝ, IsTangentLineAt (g a) f x₀) →
  (a = 1 ∨ a = 4 * Real.exp (3/2)) ∧
  (a < 1 ∧ (∃! x₀ : ℤ, f x₀ < g a x₀)) →
  (3 / (2 * Real.exp 1) ≤ a ∧ a < 1) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_inequality_l853_85361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_parallelogram_section_l853_85320

/-- A convex polyhedral angle -/
structure ConvexPolyhedralAngle where
  -- Add necessary fields here
  convex : Bool

/-- A plane in 3D space -/
structure Plane where
  -- Add necessary fields here

/-- Represents a point in 3D space -/
structure Point where
  -- Add necessary fields here

/-- Represents the intersection of a plane with a convex polyhedral angle -/
def intersection (angle : ConvexPolyhedralAngle) (plane : Plane) : Set Point := sorry

/-- Predicate to check if a set of points forms a parallelogram -/
def isParallelogram (s : Set Point) : Prop := sorry

/-- Theorem: For any convex polyhedral angle, there exists a plane that intersects 
    the angle such that the intersection is a parallelogram -/
theorem exists_parallelogram_section (angle : ConvexPolyhedralAngle) : 
  ∃ (plane : Plane), isParallelogram (intersection angle plane) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_parallelogram_section_l853_85320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_theorem_l853_85309

/-- The circle equation -/
noncomputable def circle_equation (x y r : ℝ) : Prop := (x - 3)^2 + (y + 5)^2 = r^2

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := 4*x - 3*y - 2 = 0

/-- Distance from a point to the line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |4*x - 3*y - 2| / Real.sqrt (4^2 + (-3)^2)

/-- Two points on circle with distance 1 from line -/
def two_points_condition (r : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_equation x₁ y₁ r ∧ 
    circle_equation x₂ y₂ r ∧
    distance_to_line x₁ y₁ = 1 ∧
    distance_to_line x₂ y₂ = 1 ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    ∀ (x y : ℝ), circle_equation x y r ∧ distance_to_line x y = 1 → 
      (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)

theorem circle_line_intersection_theorem (r : ℝ) :
  r > 0 → (two_points_condition r ↔ 4 < r ∧ r < 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_theorem_l853_85309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_percentage_approx_l853_85338

/-- The false weight used by the shopkeeper in grams -/
noncomputable def false_weight : ℝ := 892

/-- The actual weight of a kilogram in grams -/
noncomputable def actual_weight : ℝ := 1000

/-- The gain percentage for each type of pulse -/
noncomputable def gain_percentage : ℝ := (1 - false_weight / actual_weight) / (false_weight / actual_weight) * 100

/-- Theorem stating that the gain percentage is approximately 12.107% -/
theorem gain_percentage_approx :
  abs (gain_percentage - 12.107) < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_percentage_approx_l853_85338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_intersection_l853_85302

/-- The area of the shaded region formed by the intersection of two rectangles and a triangle -/
theorem shaded_area_intersection
  (rect1_width rect1_height rect2_width rect2_height tri_leg overlap_rect_area overlap_tri_rect1 overlap_tri_rect2 : ℝ) :
  rect1_width = 4 →
  rect1_height = 12 →
  rect2_width = 5 →
  rect2_height = 7 →
  tri_leg = 3 →
  overlap_rect_area = 12 →
  overlap_tri_rect1 = 2 →
  overlap_tri_rect2 = 1 →
  (rect1_width * rect1_height) + (rect2_width * rect2_height) + (1/2 * tri_leg * tri_leg) - overlap_rect_area - overlap_tri_rect1 - overlap_tri_rect2 = 72.5 := by
  sorry

#check shaded_area_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_intersection_l853_85302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_square_l853_85358

/-- Represents a triangle with sides a, b, c and corresponding heights m₁, m₂, m₃ -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  m₁ : ℝ
  m₂ : ℝ
  m₃ : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < m₁ ∧ 0 < m₂ ∧ 0 < m₃
  h_order : a > b ∧ b > c

/-- Calculates the side length of a square inscribed on a given side of the triangle -/
noncomputable def inscribedSquareSide (t : Triangle) (side : ℝ) (height : ℝ) : ℝ :=
  (side * height) / (side + height)

/-- Theorem: The largest inscribed square in a triangle is on its smallest side -/
theorem largest_inscribed_square (t : Triangle) :
  let x := inscribedSquareSide t t.a t.m₁
  let y := inscribedSquareSide t t.b t.m₂
  let z := inscribedSquareSide t t.c t.m₃
  z > y ∧ y > x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_square_l853_85358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_expressions_l853_85339

theorem equal_expressions (x : ℕ) (hx : x > 0) : 
  (∃! e : ℕ, (e = x! * x^x ∨ e = x^(x+1) ∨ e = (x!)^x ∨ e = x^(x!)) ∧ e = x^x * x!) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_expressions_l853_85339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mural_cost_example_l853_85340

/-- Calculates the total cost of painting a mural given its dimensions, paint cost per square meter,
    artist's painting rate, and artist's hourly rate. -/
noncomputable def mural_cost (length width paint_cost_per_sqm painting_rate_sqm_per_hour artist_hourly_rate : ℝ) : ℝ :=
  let area := length * width
  let paint_cost := area * paint_cost_per_sqm
  let time_required := area / painting_rate_sqm_per_hour
  let labor_cost := time_required * artist_hourly_rate
  paint_cost + labor_cost

/-- Theorem stating that the total cost of painting a 6m by 3m mural with the given conditions is $192. -/
theorem mural_cost_example : mural_cost 6 3 4 1.5 10 = 192 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mural_cost_example_l853_85340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_given_system_is_linear_l853_85375

/-- A linear equation in two variables is of the form ax + by = c, where a, b, and c are constants and x and y are variables. -/
def is_linear_equation (eq : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, ∀ x y : ℝ, eq x y ↔ a * x + b * y = c

/-- A system of linear equations is a set of two or more linear equations. -/
def is_system_of_linear_equations (sys : List (ℝ → ℝ → Prop)) : Prop :=
  sys.length ≥ 2 ∧ ∀ eq ∈ sys, is_linear_equation eq

/-- The given system of equations. -/
def given_system : List (ℝ → ℝ → Prop) :=
  [λ x y ↦ x + y = 11, λ x y ↦ 5 * x - 3 * y = -7]

theorem given_system_is_linear : is_system_of_linear_equations given_system := by
  sorry

#check given_system_is_linear

end NUMINAMATH_CALUDE_ERRORFEEDBACK_given_system_is_linear_l853_85375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l853_85323

noncomputable section

-- Define the ellipse C
def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2) / a

-- Define the area of triangle OAB
def triangle_area (a b : ℝ) : ℝ := a * b / 2

-- Define the constant product
noncomputable def constant_product (x₀ y₀ a b : ℝ) : ℝ :=
  |2 + x₀ / (y₀ - 1)| * |1 + 2 * y₀ / (x₀ - a)|

theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : eccentricity a b = Real.sqrt 3 / 2) (h4 : triangle_area a b = 1) :
  (∀ x y, ellipse x y a b ↔ ellipse x y 2 1) ∧
  (∀ x₀ y₀, ellipse x₀ y₀ a b → constant_product x₀ y₀ a b = 4) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l853_85323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_minus_two_eq_neg_one_l853_85318

-- Define the function g (moved before f to resolve the unknown identifier error)
noncomputable def g (x : ℝ) : ℝ :=
  if x < 0 then -2^(-x) + 3 else 0  -- We define g explicitly for x < 0

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2^x - 3 else g x

-- Define the property of f being an odd function
axiom f_odd : ∀ x, f (-x) = -f x

-- Theorem to prove
theorem f_g_minus_two_eq_neg_one : f (g (-2)) = -1 := by
  -- Expand the definition of g(-2)
  have h1 : g (-2) = 1 := by
    -- Prove that g(-2) = 1
    sorry
  
  -- Now use the definition of f(1)
  have h2 : f 1 = -1 := by
    -- Prove that f(1) = -1
    sorry

  -- Combine the results
  calc
    f (g (-2)) = f 1 := by rw [h1]
    _          = -1  := by rw [h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_minus_two_eq_neg_one_l853_85318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_properties_l853_85314

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y - 1 = k * (x - 2)

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 1

-- Define the center of circle C
def center_C : ℝ × ℝ := (1, -2)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem min_area_triangle_properties (k : ℝ) (Q : ℝ × ℝ) :
  line_l k Q.1 Q.2 →  -- Q lies on line l
  (∃ M, circle_C M.1 M.2 ∧ distance Q M = Real.sqrt ((distance Q center_C)^2 - 1)) →  -- QM is tangent to circle C
  distance Q center_C * Real.sqrt ((distance Q center_C)^2 - 1) / 2 = Real.sqrt 2 →  -- Area of triangle QMC is minimized to √2
  distance Q center_C = 3 ∧  -- |CQ| = 3
  (∀ E F, line_l k E.1 E.2 → circle_C F.1 F.2 → distance E F ≥ 2) ∧  -- Minimum distance between points on l and C is 2
  (∃ E F, line_l k E.1 E.2 ∧ circle_C F.1 F.2 ∧ distance E F = 2)  -- This minimum distance is achievable
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_properties_l853_85314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_perimeter_triangles_l853_85377

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def perimeter (a b c : ℕ) : ℕ :=
  a + b + c

noncomputable def area (a b c : ℕ) : ℝ :=
  let s : ℝ := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

def equal_area_perimeter (a b c : ℕ) : Prop :=
  area a b c = perimeter a b c

theorem equal_area_perimeter_triangles :
  ∀ a b c : ℕ,
    is_triangle a b c →
    equal_area_perimeter a b c →
    (a = 6 ∧ b = 25 ∧ c = 29) ∨
    (a = 7 ∧ b = 15 ∧ c = 20) ∨
    (a = 9 ∧ b = 10 ∧ c = 17) ∨
    (a = 5 ∧ b = 12 ∧ c = 13) ∨
    (a = 6 ∧ b = 8 ∧ c = 10) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_perimeter_triangles_l853_85377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_average_is_90_l853_85371

/-- Represents a class of students with their test scores -/
structure ClassStats where
  boys_count : ℕ
  girls_count : ℕ
  boys_total_score : ℕ
  girls_total_score : ℕ

/-- The average score of a group given their total score and count -/
def average (total : ℕ) (count : ℕ) : ℚ :=
  (total : ℚ) / (count : ℚ)

/-- Theorem stating that under given conditions, the boys' average score is 90 -/
theorem boys_average_is_90 (c : ClassStats) 
  (h1 : average (c.boys_total_score + c.girls_total_score) (c.boys_count + c.girls_count) = 94)
  (h2 : average c.girls_total_score c.girls_count = 96)
  (h3 : (c.boys_count : ℚ) / (c.girls_count : ℚ) = 1/2) :
  average c.boys_total_score c.boys_count = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_average_is_90_l853_85371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_lambda_l853_85382

theorem greatest_lambda : 
  (∃ (lambda : ℝ), lambda = (3/2) ∧ 
    (∀ (a b c : ℝ), a ≥ 0 → b ≥ 0 → c ≥ 0 → 
      a^2 + b^2 + c^2 ≥ a*b + lambda*b*c + c*a) ∧
    (∀ (lambda' : ℝ), lambda' > lambda → 
      ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ 
        a^2 + b^2 + c^2 < a*b + lambda'*b*c + c*a)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_lambda_l853_85382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l853_85315

noncomputable def a : ℕ → ℝ
| 0 => 1
| n + 1 => Real.sqrt ((a n)^2 - 2*(a n) + 2) - 1

theorem sequence_inequality :
  ∃ c : ℝ, c = 1/4 ∧ ∀ n : ℕ, a (2*n) < c ∧ c < a (2*n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l853_85315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_approx_l853_85348

-- Define the radius of the circle
def radius : ℝ := 8

-- Define pi as a constant (we'll use Lean's built-in pi)
noncomputable def circle_pi : ℝ := Real.pi

-- Define the circumference formula
noncomputable def circumference (r : ℝ) : ℝ := 2 * circle_pi * r

-- State the theorem
theorem circle_circumference_approx :
  abs (circumference radius - 50.27) < 0.01 := by
  -- Unfold the definitions
  unfold circumference circle_pi
  -- Simplify the expression
  simp [Real.pi]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_approx_l853_85348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slower_train_speed_problem_l853_85372

/-- The speed of the slower train given the conditions of the problem -/
noncomputable def slower_train_speed (faster_train_speed : ℝ) (faster_train_length : ℝ) (passing_time : ℝ) : ℝ :=
  let relative_speed := (faster_train_length / passing_time) * (18 / 5)
  relative_speed - faster_train_speed

theorem slower_train_speed_problem :
  let faster_train_speed : ℝ := 45
  let faster_train_length : ℝ := 270.0216
  let passing_time : ℝ := 12
  ∃ ε > 0, |slower_train_speed faster_train_speed faster_train_length passing_time - 117.013| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slower_train_speed_problem_l853_85372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_angles_l853_85313

/-- The angles at which the parabola y = x^2 - x intersects the x-axis --/
theorem parabola_intersection_angles :
  let f : ℝ → ℝ := λ x ↦ x^2 - x
  let x₁ : ℝ := 0
  let x₂ : ℝ := 1
  let α₁ := Real.arctan (deriv f x₁)
  let α₂ := Real.arctan (deriv f x₂)
  α₁ = 3*π/4 ∧ α₂ = π/4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_angles_l853_85313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_matrix_add_five_to_second_column_l853_85385

open Matrix

theorem no_matrix_add_five_to_second_column :
  ¬∃ (M : Matrix (Fin 2) (Fin 2) ℝ),
    ∀ (A : Matrix (Fin 2) (Fin 2) ℝ),
      M * A = Matrix.of (λ i j => 
        if j = 1 then A i j + 5 else A i j) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_matrix_add_five_to_second_column_l853_85385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_prime_14n_plus_19_l853_85370

theorem not_prime_14n_plus_19 (n : ℕ) : ¬ (Nat.Prime (14^n + 19)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_prime_14n_plus_19_l853_85370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_c_value_max_perimeter_l853_85325

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  t.C = 2 * Real.pi / 3

def is_arithmetic_sequence (t : Triangle) : Prop :=
  t.b - t.a = 2 ∧ t.c - t.b = 2

-- Theorem 1
theorem side_c_value (t : Triangle) 
  (h1 : is_valid_triangle t) 
  (h2 : is_arithmetic_sequence t) : 
  t.c = 7 := by sorry

-- Theorem 2
theorem max_perimeter (t : Triangle) 
  (h1 : is_valid_triangle t) 
  (h2 : t.c = Real.sqrt 3) : 
  (∃ p : ℝ, p = t.a + t.b + t.c ∧ 
   ∀ q : ℝ, q = t.a + t.b + t.c → q ≤ p) → 
  p = 2 + Real.sqrt 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_c_value_max_perimeter_l853_85325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_upstream_time_l853_85328

/-- Represents the time taken for a boat to travel upstream given its speed in still water,
    the stream speed, and the time taken to travel the same distance downstream. -/
noncomputable def upstreamTime (boatSpeed streamSpeed downstreamTime : ℝ) : ℝ :=
  downstreamTime * (boatSpeed + streamSpeed) / (boatSpeed - streamSpeed)

/-- Theorem stating that for a boat with speed 15 kmph in still water and a stream speed of 3 kmph,
    if it takes 1 hour to travel downstream, it will take 1.5 hours to travel the same distance upstream. -/
theorem boat_upstream_time :
  upstreamTime 15 3 1 = 1.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_upstream_time_l853_85328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reach_factorial_18_l853_85321

def board_operation (a : ℕ) (d : ℕ) : Prop :=
  10 ≤ d ∧ d ≤ 20 ∧ Nat.Coprime a d

def can_reach (target : ℕ) : Prop :=
  ∃ (n : ℕ) (sequence : Fin (n + 1) → ℕ),
    sequence 0 = 1 ∧
    (∀ i : Fin n, board_operation (sequence i) (sequence (i.succ) - sequence i)) ∧
    sequence n = target

theorem reach_factorial_18 : can_reach (Nat.factorial 18) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reach_factorial_18_l853_85321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_over_one_plus_cos_is_odd_l853_85345

/-- The function f(x) = tan(x) / (1 + cos(x)) is odd -/
theorem tan_over_one_plus_cos_is_odd : 
  ∀ x : ℝ, Real.tan x / (1 + Real.cos x) = -((Real.tan (-x)) / (1 + Real.cos (-x))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_over_one_plus_cos_is_odd_l853_85345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_sin_4x_l853_85364

noncomputable def f (x : ℝ) : ℝ := Real.sin (4 * x)

theorem min_positive_period_of_sin_4x :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) ∧
  T = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_sin_4x_l853_85364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_length_l853_85398

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

-- Define the line passing through the origin with slope angle 60°
def line_equation (x y : ℝ) : Prop := y = x * Real.tan (60 * Real.pi / 180)

-- Define the intersection points
def intersection (x y : ℝ) : Prop := circle_equation x y ∧ line_equation x y

-- Define the chord length
noncomputable def chord_length : ℝ := 2 * Real.sqrt 3

-- Theorem statement
theorem chord_intersection_length :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    intersection x₁ y₁ ∧ intersection x₂ y₂ ∧
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = chord_length :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_length_l853_85398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_problem_l853_85373

theorem divisibility_problem (x : ℕ) (h : x = 166) :
  (∀ d : ℕ, d ∈ [2, 3, 4, 5, 6, 7, 8, 9, 10, 11] → (x^2 + 164) % d = 0) ∧
  (∀ y : ℕ, y < x → ∃ d : ℕ, d ∈ [2, 3, 4, 5, 6, 7, 8, 9, 10] ∧ (y^2 + 164) % d ≠ 0) :=
by
  sorry

#eval (166^2 + 164) % 11

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_problem_l853_85373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_and_min_area_l853_85326

/-- The locus S of points P satisfying |PM| - |PN| = 2√2 where M(-2,0) and N(2,0) -/
noncomputable def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 - p.2^2 = 2 ∧ p.1 > 0}

/-- The area of triangle OAB where A and B are intersection points of a line through (2,0) with S -/
noncomputable def triangleArea (a b : ℝ × ℝ) : ℝ :=
  abs (a.1 * b.2 - a.2 * b.1) / 2

theorem locus_and_min_area :
  (∀ p : ℝ × ℝ, Real.sqrt ((p.1 + 2)^2 + p.2^2) - Real.sqrt ((p.1 - 2)^2 + p.2^2) = 2 * Real.sqrt 2 ↔ p ∈ S) ∧
  (∃ min_area : ℝ, min_area = 2 * Real.sqrt 2 ∧
    ∀ a b : ℝ × ℝ, a ∈ S → b ∈ S →
      (∃ k : ℝ, a.2 = k * (a.1 - 2) ∧ b.2 = k * (b.1 - 2)) →
      triangleArea a b ≥ min_area) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_and_min_area_l853_85326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_problem_l853_85352

theorem graph_translation_problem (θ φ : ℝ) 
  (h1 : |θ| < π / 2) 
  (h2 : 0 < φ) 
  (h3 : φ < π) 
  (h4 : Real.sin θ = 1 / 2) 
  (h5 : Real.sin (-2 * φ + θ) = 1 / 2) : φ = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_problem_l853_85352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l853_85354

-- Define the triangle
def Triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

-- Define the acute triangle condition
def AcuteTriangle (A B C : ℝ) : Prop :=
  0 < A ∧ A < Real.pi/2 ∧ 0 < B ∧ B < Real.pi/2 ∧ 0 < C ∧ C < Real.pi/2

-- Define the parallel vector condition
def ParallelVectors (A B C : ℝ) (a b c : ℝ) : Prop :=
  (Real.sin A + Real.sin B - Real.sin C) * (b + c - a) = c * Real.sin A

-- State the theorem
theorem triangle_theorem (A B C a b c : ℝ) 
  (h1 : Triangle A B C a b c) 
  (h2 : AcuteTriangle A B C) 
  (h3 : ParallelVectors A B C a b c) : 
  B = Real.pi/3 ∧ 3/2 < Real.sin A + Real.sin C ∧ Real.sin A + Real.sin C ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l853_85354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_Q_l853_85334

theorem solve_for_Q : ∃ Q : ℝ, (Q^3).sqrt = 18 * (64^(1/6)) ∧ Q = 4 * 2^(1/3) * 3 * 3^(1/3) := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_Q_l853_85334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_equilateral_l853_85363

noncomputable def f (x : ℝ) := Real.sin (2*x + Real.pi/3) - Real.cos (2*x + Real.pi/6) - Real.sqrt 3 * Real.cos (2*x)

theorem triangle_abc_equilateral (B : ℝ) (hB : 0 < B ∧ B < Real.pi/2) :
  let A := Real.pi/3
  let C := Real.pi/3
  f B = Real.sqrt 3 →
  (Real.sin A + Real.sin B + Real.sin C) * 2 * (Real.sqrt 3 / 3) = 3 * Real.sqrt 3 →
  A = B ∧ B = C ∧ 2 * (Real.sqrt 3 / 3) * Real.sin (A / 2) = Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_equilateral_l853_85363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l853_85301

-- Define the base 10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem inequality_proof (x : ℝ) (h : x ≥ 10^(-(3/2 : ℝ))) :
  (lg x + 3)^7 + (lg x)^7 + lg (x^2) + 3 ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l853_85301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_multiples_of_four_l853_85356

theorem count_multiples_of_four : 
  Finset.card (Finset.filter (fun n => 4 ∣ n) (Finset.range 1001 \ {0})) = 250 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_multiples_of_four_l853_85356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_eq_one_solutions_l853_85343

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 1 - x else Real.log x

-- State the theorem
theorem f_f_eq_one_solutions (a : ℝ) :
  f (f a) = 1 ↔ (a = 1 ∨ a = Real.exp (Real.exp 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_eq_one_solutions_l853_85343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l853_85396

/-- Parabola with equation y^2 = -12x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = -12 * p.1}

/-- Focus of the parabola -/
def Focus : ℝ × ℝ := (-3, 0)

/-- Point P on the parabola -/
def P (m : ℝ) : ℝ × ℝ := (-4, m)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_focus_distance (m : ℝ) :
  P m ∈ Parabola → distance (P m) Focus = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l853_85396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l853_85381

-- Define the hyperbola C
def hyperbola (m : ℝ) (x y : ℝ) : Prop := x^2 / m - y^2 = 1

-- Define the asymptote of C
def asymptote (m : ℝ) (x y : ℝ) : Prop := Real.sqrt 3 * x + m * y = 0

-- Define the focal length
noncomputable def focal_length (m : ℝ) : ℝ := 2 * Real.sqrt (m + 1)

-- Theorem statement
theorem hyperbola_focal_length (m : ℝ) (h1 : m > 0) 
  (h2 : ∀ x y : ℝ, hyperbola m x y ↔ asymptote m x y) : 
  focal_length m = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l853_85381
