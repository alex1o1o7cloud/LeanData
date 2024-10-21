import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pauline_meat_purchase_l1172_117226

/-- The cost of taco shells in dollars -/
def taco_shell_cost : ℚ := 5

/-- The number of bell peppers bought -/
def bell_pepper_count : ℕ := 4

/-- The cost of each bell pepper in dollars -/
def bell_pepper_unit_cost : ℚ := 3/2

/-- The cost of meat per pound in dollars -/
def meat_cost_per_pound : ℚ := 3

/-- The total amount spent in dollars -/
def total_spent : ℚ := 17

/-- The amount of meat bought in pounds -/
noncomputable def meat_amount : ℚ := (total_spent - (taco_shell_cost + bell_pepper_count * bell_pepper_unit_cost)) / meat_cost_per_pound

theorem pauline_meat_purchase :
  meat_amount = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pauline_meat_purchase_l1172_117226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_IJKL_WXYZ_l1172_117206

/-- Square WXYZ with side length s -/
noncomputable def Square (s : ℝ) : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ s ∧ 0 ≤ p.2 ∧ p.2 ≤ s}

/-- Point I on side WZ of square WXYZ -/
noncomputable def I (s : ℝ) : ℝ × ℝ := (3*s/4, 0)

/-- Square IJKL with vertices on sides of WXYZ -/
noncomputable def IJKL (s : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), ((x = 3*s/4 ∧ y = 0) ∨ 
                    (x = s ∧ y = s/4) ∨ 
                    (x = s/4 ∧ y = s) ∨ 
                    (x = 0 ∧ y = 3*s/4)) ∧
                    p = (x, y)}

/-- Area of a square -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem stating the area ratio of IJKL to WXYZ -/
theorem area_ratio_IJKL_WXYZ (s : ℝ) (hs : s > 0) :
  area (IJKL s) / area (Square s) = 1/8 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_IJKL_WXYZ_l1172_117206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_passengers_for_given_route_l1172_117285

/-- Represents a bus route with given number of stops and bus capacity -/
structure BusRoute where
  num_stops : Nat
  capacity : Nat

/-- Represents whether a passenger gets on at a specific stop -/
def get_on (p : Nat) (stop : Nat) : Prop := sorry

/-- Represents whether a passenger gets off at a specific stop -/
def get_off (p : Nat) (stop : Nat) : Prop := sorry

/-- Represents the maximum number of passengers that can be carried -/
def max_passengers (route : BusRoute) : Nat := sorry

/-- The theorem stating the maximum number of passengers for the given conditions -/
theorem max_passengers_for_given_route :
  ∀ (route : BusRoute),
    route.num_stops = 12 →
    route.capacity = 20 →
    (∀ (i j : Nat), i < route.num_stops → j < route.num_stops → i ≠ j →
      (∃ (p : Nat), p ≤ max_passengers route ∧
        (get_on p i ∧ get_off p j ∨ get_on p j ∧ get_off p i))) →
    max_passengers route = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_passengers_for_given_route_l1172_117285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quadrangular_pyramid_dihedral_angle_l1172_117230

/-- A regular quadrangular pyramid is a pyramid with a square base and all lateral edges congruent -/
structure RegularQuadrangularPyramid where
  base_side : ℝ
  lateral_edge : ℝ
  apex_angle : ℝ
  lateral_base_angle : ℝ
  apex_angle_eq_lateral_base : apex_angle = lateral_base_angle

/-- The dihedral angle between adjacent lateral faces of a regular quadrangular pyramid -/
noncomputable def dihedral_angle (p : RegularQuadrangularPyramid) : ℝ := Real.arccos (2 - Real.sqrt 5)

/-- Theorem: In a regular quadrangular pyramid where the plane angle at the apex 
    is equal to the angle between the lateral edge and the base plane, 
    the dihedral angle between adjacent lateral faces is arccos(2-√5) -/
theorem regular_quadrangular_pyramid_dihedral_angle 
  (p : RegularQuadrangularPyramid) : 
  dihedral_angle p = Real.arccos (2 - Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quadrangular_pyramid_dihedral_angle_l1172_117230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_non_grey_is_five_sixths_l1172_117257

/-- Represents the different colors of balls in the box -/
inductive BallColor
  | Grey
  | White
  | Black

/-- Represents the box containing the balls -/
structure Box where
  grey : ℕ
  white : ℕ
  black : ℕ

/-- Calculates the total number of balls in the box -/
def totalBalls (box : Box) : ℕ :=
  box.grey + box.white + box.black

/-- Calculates the number of non-grey balls in the box -/
def nonGreyBalls (box : Box) : ℕ :=
  box.white + box.black

/-- Calculates the probability of selecting a non-grey ball -/
def probNonGrey (box : Box) : ℚ :=
  ↑(nonGreyBalls box) / ↑(totalBalls box)

/-- The main theorem to be proved -/
theorem prob_non_grey_is_five_sixths (box : Box) 
  (h1 : box.grey = 1) 
  (h2 : box.white = 2) 
  (h3 : box.black = 3) : 
  probNonGrey box = 5 / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_non_grey_is_five_sixths_l1172_117257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chopped_cube_height_l1172_117294

/-- A unit cube with a corner chopped off -/
structure ChoppedCube where
  /-- The side length of the equilateral triangle formed by the cut -/
  triangle_side : ℝ
  /-- The volume of the chopped-off portion (pyramid) -/
  pyramid_volume : ℝ

/-- The height of the remaining part of the chopped cube when placed on the cut face -/
noncomputable def remaining_height (c : ChoppedCube) : ℝ :=
  1 - Real.sqrt 3 / 3

/-- Theorem stating the height of the remaining part of the chopped cube -/
theorem chopped_cube_height (c : ChoppedCube)
  (h1 : c.triangle_side = Real.sqrt 2)
  (h2 : c.pyramid_volume = 1/6) :
  remaining_height c = 1 - Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chopped_cube_height_l1172_117294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_x_coordinate_l1172_117215

noncomputable def f (x : ℝ) : ℝ := Real.exp x / x

theorem tangent_point_x_coordinate 
  (t : ℝ) (h_t : t ≠ 0) :
  ∃ (x : ℝ), x > 0 ∧ 
  (∃ (y : ℝ), x - t * y - 2 = 0 ∧
              f x = y ∧
              (deriv f x) = 1 / t) ∧
  (x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_x_coordinate_l1172_117215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1172_117245

-- Define the ⊕ operation
noncomputable def oplus (a b : ℝ) : ℝ :=
  if a ≥ b then a else b^2

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  (oplus 1 x) * x - (oplus 2 x)

-- Theorem statement
theorem max_value_of_f :
  ∃ (c : ℝ), c = 6 ∧
  ∀ (x : ℝ), x ∈ Set.Icc (-2) 2 → f x ≤ c := by
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1172_117245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_and_monotonicity_l1172_117201

/-- The function f(x) = x^2 + 2ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2

/-- The domain of x -/
def domain : Set ℝ := Set.Icc (-5) 5

theorem f_extrema_and_monotonicity :
  (∀ x ∈ domain, f (-1) x ≥ 1 ∧ f (-1) x ≤ 37 ∧ 
   (∃ x₁ x₂, x₁ ∈ domain ∧ x₂ ∈ domain ∧ f (-1) x₁ = 1 ∧ f (-1) x₂ = 37)) ∧
  (∀ a : ℝ, (∀ x y, x ∈ domain → y ∈ domain → x < y → (f a x < f a y ∨ f a x > f a y)) ↔ 
   (a ≤ -5 ∨ a ≥ 5)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_and_monotonicity_l1172_117201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_reversal_l1172_117255

/-- Represents a weight on the scale -/
structure Weight where
  value : ℝ
  students : Finset ℕ

/-- Represents the state of the scale -/
structure Scale where
  left : List Weight
  right : List Weight

/-- Calculates the total weight on a side of the scale -/
def totalWeight (side : List Weight) : ℝ :=
  side.foldl (fun acc w => acc + w.value) 0

/-- Determines if the left side is heavier than the right side -/
noncomputable def isLeftHeavier (s : Scale) : Bool :=
  totalWeight s.left > totalWeight s.right

/-- Simulates a student entering and moving weights -/
def studentEnters (s : Scale) (student : ℕ) : Scale :=
  let moveWeight (acc : Scale) (w : Weight) :=
    if student ∈ w.students then
      { left := w :: acc.right, right := acc.left }
    else
      { left := w :: acc.left, right := acc.right }
  s.left.foldl moveWeight { left := [], right := s.right }

/-- The main theorem to prove -/
theorem exists_reversal (initial : Scale) (h : isLeftHeavier initial = true) :
  ∃ (students : List ℕ), isLeftHeavier (students.foldl studentEnters initial) = false := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_reversal_l1172_117255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermats_little_theorem_l1172_117297

theorem fermats_little_theorem (p : ℕ) (a : ℕ) (h_prime : Nat.Prime p) (h_not_div : ¬(p ∣ a)) :
  a ^ (p - 1) ≡ 1 [MOD p] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermats_little_theorem_l1172_117297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_area_greater_half_l1172_117295

/-- A right triangle with altitude to hypotenuse equal to 1 -/
structure RightTriangle where
  /-- The angle between the altitude and one of the legs -/
  α : ℝ
  /-- Assumption that α is between 0 and π/2 -/
  α_range : 0 < α ∧ α < Real.pi / 2

/-- The radius of the inscribed circle in a right triangle -/
noncomputable def inscribed_circle_radius (t : RightTriangle) : ℝ :=
  1 / (1 + Real.sqrt 2 * Real.cos (t.α - Real.pi / 4))

/-- The area of the inscribed circle in a right triangle -/
noncomputable def inscribed_circle_area (t : RightTriangle) : ℝ :=
  Real.pi * (inscribed_circle_radius t) ^ 2

/-- Theorem: The area of the inscribed circle in any right triangle
    with altitude to hypotenuse equal to 1 is greater than 0.5 -/
theorem inscribed_circle_area_greater_half (t : RightTriangle) :
  inscribed_circle_area t > 0.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_area_greater_half_l1172_117295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_l1172_117239

-- Proposition 1
def trajectory_hyperbola (z : ℂ) : Prop :=
  Complex.abs (z - 2) - Complex.abs (z + 2) = 1

-- Proposition 2
def trajectory_parabola (z : ℂ) (a : ℝ) : Prop :=
  z = a^2 + a * Complex.I

-- Proposition 3
def sequence_increasing_necessary_not_sufficient 
  (f : ℝ → ℝ) (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, a n = f n) ∧
  (∀ n : ℕ, a n ≤ a (n + 1)) ∧
  ¬(∀ x y : ℝ, x < y → f x < f y)

-- Proposition 4
def translate_curve (g : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, g (x - 1) (y - 2) ↔ g x y

-- Proposition 5
def ellipse_to_circle (F : ℝ → ℝ → Prop) : Prop :=
  (∃ a b : ℝ, a ≠ b ∧ ∀ x y : ℝ, F x y ↔ (x^2 / a^2 + y^2 / b^2 = 1)) →
  ∃ p q : ℝ, ∀ x y : ℝ, F (p * x) (q * y) ↔ (x^2 + y^2 = 1)

theorem propositions_truth : 
  (¬∀ z : ℂ, trajectory_hyperbola z) ∧
  (∀ a : ℝ, ∃ z : ℂ, trajectory_parabola z a) ∧
  (∃ f : ℝ → ℝ, ∃ a : ℕ → ℝ, sequence_increasing_necessary_not_sufficient f a) ∧
  (∀ g : ℝ → ℝ → Prop, translate_curve g) ∧
  (∀ F : ℝ → ℝ → Prop, ellipse_to_circle F) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_l1172_117239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_max_value_l1172_117202

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x - x

theorem tangent_line_and_max_value :
  (∃ (m b : ℝ), ∀ x, m * x + b = 0 ∧ 
    (∀ h : ℝ, h ≠ 0 → |f h - (m * h + b)| ≤ (fun r => r) h * |h|)) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ Real.exp (Real.pi / 2) - Real.pi / 2) ∧
  (f (Real.pi / 2) = Real.exp (Real.pi / 2) - Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_max_value_l1172_117202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l1172_117278

/-- The distance from the focus to the directrix of the parabola y² = 5x is 5/2 -/
theorem parabola_focus_directrix_distance :
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 5*x}
  let a : ℝ := 5/4
  let focus : ℝ × ℝ := (a, 0)
  let directrix : Set (ℝ × ℝ) := {(x, y) : ℝ × ℝ | x = -a}
  (Set.Icc 0 1).Nonempty →
  ∃ p ∈ parabola, ∃ d ∈ directrix,
    dist p focus + dist p d = 5/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l1172_117278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_hot_day_price_l1172_117250

/-- Calculates the price of a cup of lemonade on a hot day given the conditions of Sarah's lemonade stand. -/
theorem lemonade_hot_day_price :
  ∀ (regular_price : ℚ) (total_profit : ℚ) (total_days : ℕ) (hot_days : ℕ) 
    (cups_per_day : ℕ) (cost_per_cup : ℚ),
  total_profit = 210 →
  total_days = 10 →
  hot_days = 3 →
  cups_per_day = 32 →
  cost_per_cup = 3/4 →
  total_profit = (cups_per_day : ℚ) * regular_price * ((total_days - hot_days : ℚ) + 
                  1.25 * hot_days) - (cups_per_day : ℚ) * total_days * cost_per_cup →
  ∃ (hot_price : ℚ), 
    hot_price = 1.25 * regular_price ∧ 
    (⌊hot_price * 100⌋ : ℚ) / 100 = 164 / 100 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_hot_day_price_l1172_117250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1172_117200

noncomputable def ellipse_C (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

noncomputable def point_A : ℝ × ℝ := (Real.sqrt 6, 2 * Real.sqrt 6 / 3)

noncomputable def sum_distances_6 (F₁ F₂ : ℝ × ℝ) : Prop :=
  Real.sqrt ((point_A.1 - F₁.1)^2 + (point_A.2 - F₁.2)^2) +
  Real.sqrt ((point_A.1 - F₂.1)^2 + (point_A.2 - F₂.2)^2) = 6

theorem ellipse_properties :
  ∃ (a b : ℝ) (F₁ F₂ : ℝ × ℝ),
    a > b ∧ b > 0 ∧
    ellipse_C point_A.1 point_A.2 a b ∧
    sum_distances_6 F₁ F₂ ∧
    (∀ x y, ellipse_C x y a b ↔ x^2 / 9 + y^2 / 8 = 1) ∧
    F₁ = (-1, 0) ∧ F₂ = (1, 0) ∧
    (∀ x y, (∃ x₁ y₁, ellipse_C x₁ y₁ a b ∧ x = (x₁ - F₁.1) / 2 ∧ y = (y₁ - F₁.2) / 2) ↔
      (2*x + 1)^2 / 9 + y^2 / 2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1172_117200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_song_duration_l1172_117253

/-- Represents the duration of a song in minutes -/
def SongDuration := ℕ

/-- Represents a playlist of songs -/
structure Playlist where
  songs : List SongDuration
  total_duration : ℕ

instance : OfNat SongDuration n where
  ofNat := n

/-- Given a playlist with three songs, where two songs have known durations,
    prove that the third song's duration is the difference between the total
    duration and the sum of the known durations. -/
theorem third_song_duration (p : Playlist) 
  (h1 : p.songs.length = 3)
  (h2 : p.total_duration = 8)
  (h3 : p.songs[0] = 2)  -- "Raise the Roof"
  (h4 : p.songs[1] = 3)  -- "Rap Battle"
  : p.songs[2] = 3 := by  -- "The Best Day"
  sorry

#check third_song_duration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_song_duration_l1172_117253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1172_117208

theorem problem_solution (k a b m : ℕ) 
  (h1 : k % (a^2) = 0) 
  (h2 : k % (b^2) = 0)
  (h3 : k / (a^2) = m)
  (h4 : k / (b^2) = m + 116) :
  (Nat.Coprime a b → 
    (Nat.Coprime (a^2 - b^2) (a^2) ∧ 
     Nat.Coprime (a^2 - b^2) (b^2))) ∧
  (Nat.Coprime a b → k = 176400) ∧
  (Nat.gcd a b = 5 → k = 4410000) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1172_117208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leibniz_series_converges_to_pi_over_four_odd_squares_series_converges_to_pi_squared_over_eight_l1172_117216

/-- The alternating Leibniz series converges to π/4 -/
theorem leibniz_series_converges_to_pi_over_four :
  ∑' (n : ℕ), ((-1 : ℝ)^n) / (2*n + 1 : ℝ) = π / 4 := by sorry

/-- The series of reciprocals of odd squares converges to π²/8 -/
theorem odd_squares_series_converges_to_pi_squared_over_eight :
  ∑' (n : ℕ), (1 : ℝ) / ((2*n + 1 : ℝ)^2) = π^2 / 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leibniz_series_converges_to_pi_over_four_odd_squares_series_converges_to_pi_squared_over_eight_l1172_117216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_arrangements_l1172_117205

/-- The number of ways to arrange 4 boys and 2 girls in a line. -/
def arrangement_count (n_boys n_girls : ℕ) : ℕ := Nat.factorial (n_boys + n_girls)

/-- The number of arrangements with girls at the ends. -/
def girls_at_ends (n_boys n_girls : ℕ) : ℕ := Nat.factorial n_girls * Nat.factorial n_boys

/-- The number of arrangements with girls not adjacent. -/
def girls_not_adjacent (n_boys n_girls : ℕ) : ℕ := 
  Nat.factorial n_boys * Nat.choose (n_boys + 1) n_girls

/-- The number of arrangements with one girl to the right of the other. -/
def girl_right_of_other (n_boys n_girls : ℕ) : ℕ := 
  Nat.choose (n_boys + n_girls) n_boys

theorem photo_arrangements (n_boys n_girls : ℕ) 
  (h_boys : n_boys = 4) (h_girls : n_girls = 2) : 
  girls_at_ends n_boys n_girls = 48 ∧ 
  girls_not_adjacent n_boys n_girls = 480 ∧ 
  girl_right_of_other n_boys n_girls = 360 := by
  sorry

#eval girls_at_ends 4 2
#eval girls_not_adjacent 4 2
#eval girl_right_of_other 4 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_arrangements_l1172_117205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_daily_profit_l1172_117282

def cost_price : ℚ := 30

def sales_price (x : ℤ) : ℚ :=
  if 1 ≤ x ∧ x ≤ 30 then (1/2) * x + 35
  else if 31 ≤ x ∧ x ≤ 60 then 50
  else 0

def sales_volume (x : ℤ) : ℚ :=
  if 1 ≤ x ∧ x ≤ 60 then 124 - 2 * x
  else 0

def daily_profit (x : ℤ) : ℚ :=
  (sales_price x - cost_price) * sales_volume x

theorem max_daily_profit :
  ∃ (max_profit : ℚ) (max_day : ℤ),
    max_profit = 1296 ∧ max_day = 26 ∧
    ∀ (x : ℤ), 1 ≤ x ∧ x ≤ 60 → daily_profit x ≤ max_profit :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_daily_profit_l1172_117282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_height_l1172_117290

/-- Represents a trapezoid with given diagonal lengths and midline length -/
structure Trapezoid where
  diagonal1 : ℝ
  diagonal2 : ℝ
  midline : ℝ

/-- Calculates the height of a trapezoid given its properties -/
noncomputable def trapezoid_height (t : Trapezoid) : ℝ :=
  2 * (t.diagonal1 * t.diagonal2) / (2 * t.midline)

/-- Theorem stating the height of a specific trapezoid -/
theorem specific_trapezoid_height :
  let t : Trapezoid := { diagonal1 := 6, diagonal2 := 8, midline := 5 }
  trapezoid_height t = 4.8 := by
  -- The proof goes here
  sorry

#check specific_trapezoid_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_height_l1172_117290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_non_positive_probability_l1172_117241

def S : Finset Int := {-3, -7, 0, 5, 3, -1}

theorem product_non_positive_probability :
  let pairs := (S.powerset.filter (fun p => p.card = 2))
  (pairs.filter (fun p => (p.toList.prod ≤ 0))).card / pairs.card = 11 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_non_positive_probability_l1172_117241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_A_equal_functions_C_equal_functions_B_different_domain_functions_D_different_domain_l1172_117286

-- Define the functions
noncomputable def f_A (x : ℝ) : ℝ := x
noncomputable def g_A (x : ℝ) : ℝ := (x^3)^(1/3)

noncomputable def f_B (x : ℝ) : ℝ := x + 1
noncomputable def g_B (x : ℝ) : ℝ := (x^2 - 1) / (x - 1)

noncomputable def f_C (x : ℝ) : ℝ := Real.sqrt x + 1/x
noncomputable def g_C (t : ℝ) : ℝ := Real.sqrt t + 1/t

noncomputable def f_D (x : ℝ) : ℝ := Real.sqrt (x^2 - 1)
noncomputable def g_D (x : ℝ) : ℝ := Real.sqrt (x + 1) * Real.sqrt (x - 1)

-- Theorem statements
theorem functions_A_equal : ∀ x : ℝ, f_A x = g_A x := by sorry

theorem functions_C_equal : ∀ x : ℝ, x > 0 → f_C x = g_C x := by sorry

theorem functions_B_different_domain : 
  {x : ℝ | ∃ y, f_B x = y} ≠ {x : ℝ | ∃ y, g_B x = y} := by sorry

theorem functions_D_different_domain : 
  {x : ℝ | ∃ y, f_D x = y} ≠ {x : ℝ | ∃ y, g_D x = y} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_A_equal_functions_C_equal_functions_B_different_domain_functions_D_different_domain_l1172_117286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cereal_difference_approx_52_l1172_117210

/-- Represents the outcome of rolling an eight-sided die, excluding 8 --/
inductive DieOutcome
| One | Two | Three | Four | Five | Six | Seven

/-- Determines if a die outcome results in sweetened cereal --/
def is_sweetened (outcome : DieOutcome) : Bool :=
  match outcome with
  | DieOutcome.One | DieOutcome.Three | DieOutcome.Five | DieOutcome.Seven => true
  | _ => false

/-- The number of days in a leap year --/
def leap_year_days : ℕ := 366

/-- The probability of rolling an odd number (excluding 8) --/
def prob_odd : ℚ := 4/7

/-- The probability of rolling an even number (excluding 8) --/
def prob_even : ℚ := 3/7

/-- The expected difference in days between sweetened and unsweetened cereal consumption --/
def expected_difference : ℚ := leap_year_days * (prob_odd - prob_even)

theorem cereal_difference_approx_52 :
  ⌊expected_difference⌋ = 52 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cereal_difference_approx_52_l1172_117210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_k_values_l1172_117277

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def Lines_are_perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- Definition of line l₁ -/
def l₁ (k : ℝ) : ℝ → ℝ → Prop :=
  fun x y ↦ (k - 3) * x + (k + 4) * y + 1 = 0

/-- Definition of line l₂ -/
def l₂ (k : ℝ) : ℝ → ℝ → Prop :=
  fun x y ↦ (k + 1) * x + 2 * (k - 3) * y + 3 = 0

/-- Slope of line l₁ -/
noncomputable def m₁ (k : ℝ) : ℝ := -(k - 3) / (k + 4)

/-- Slope of line l₂ -/
noncomputable def m₂ (k : ℝ) : ℝ := -(k + 1) / (2 * (k - 3))

/-- Main theorem -/
theorem perpendicular_lines_k_values :
  ∀ k : ℝ, Lines_are_perpendicular (m₁ k) (m₂ k) → k = 3 ∨ k = -3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_k_values_l1172_117277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_75_and_sqrt_1_plus_sin_2_l1172_117279

theorem sin_75_and_sqrt_1_plus_sin_2 :
  (Real.sin (75 * π / 180) = (Real.sqrt 2 + Real.sqrt 6) / 4) ∧
  (Real.sqrt (1 + Real.sin 2) = Real.sin 1 + Real.cos 1) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_75_and_sqrt_1_plus_sin_2_l1172_117279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l1172_117284

theorem tan_alpha_plus_pi_fourth (α : Real) 
  (h1 : Real.cos α = 3/5) (h2 : 0 < α) (h3 : α < Real.pi) : 
  Real.tan (α + Real.pi/4) = -7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l1172_117284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_is_eight_l1172_117232

-- Define the parallelogram as a structure
structure Parallelogram where
  base : ℝ
  height : ℝ

-- Define the area function for a parallelogram
def area (p : Parallelogram) : ℝ := p.base * p.height

-- Theorem statement
theorem parallelogram_area_is_eight :
  ∀ p : Parallelogram, p.base = 4 ∧ p.height = 2 → area p = 8 :=
by
  intro p ⟨base_eq, height_eq⟩
  unfold area
  rw [base_eq, height_eq]
  norm_num

-- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_is_eight_l1172_117232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l1172_117247

-- Define the concepts as functions instead of types
def standard_deviation : ℝ → ℝ := sorry
def sample_data_fluctuation : ℝ → ℝ := sorry
def regression_analysis : ℝ → ℝ → ℝ := sorry
def forecast_variable : ℝ → ℝ := sorry
def explanatory_variable : ℝ → ℝ := sorry
def random_error : ℝ → ℝ := sorry
def correlation_index : ℝ → ℝ := sorry
def sum_of_squared_residuals : ℝ → ℝ := sorry
def regression_model_fitting_effect : ℝ → ℝ := sorry

-- Define the statements
def statement_2 : Prop :=
  ∀ x y : ℝ, x < y → sample_data_fluctuation (standard_deviation x) < sample_data_fluctuation (standard_deviation y)

def statement_4 : Prop :=
  ∀ x : ℝ, forecast_variable x = explanatory_variable x + random_error x

def statement_5 : Prop :=
  ∀ x y : ℝ, x < y → 
    sum_of_squared_residuals (correlation_index x) > sum_of_squared_residuals (correlation_index y) ∧ 
    regression_model_fitting_effect (correlation_index x) < regression_model_fitting_effect (correlation_index y)

-- Theorem to prove
theorem correct_statements :
  statement_2 ∧ statement_4 ∧ statement_5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l1172_117247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_problem_l1172_117298

/-- Represents a person's walking speed and direction --/
structure WalkingPace where
  speed : ℝ  -- Speed in miles per minute
  direction : String  -- Direction of movement

/-- Calculates the distance traveled given a walking pace and time --/
def distanceTraveled (pace : WalkingPace) (time : ℝ) : ℝ :=
  pace.speed * time

theorem walking_problem (jay_pace paul_pace anne_pace : WalkingPace) 
  (h_jay : jay_pace = { speed := 1.1 / 20, direction := "east" })
  (h_paul : paul_pace = { speed := 3.1 / 45, direction := "west" })
  (h_anne : anne_pace = { speed := 0.9 / 30, direction := "north" })
  (total_time : ℝ) (h_time : total_time = 3 * 60) : 
  let jay_distance := distanceTraveled jay_pace total_time
  let paul_distance := distanceTraveled paul_pace total_time
  let anne_distance := distanceTraveled anne_pace total_time
  (jay_distance + paul_distance = 22.3) ∧ 
  (anne_distance = 5.4) := by
  sorry

#check walking_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_problem_l1172_117298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sumOfFirst10CommonElements_l1172_117271

def arithmeticProgression (n : ℕ) : ℕ := 4 + 3 * n

def geometricProgression (k : ℕ) : ℕ := 10 * 2^k

def commonElements (n : ℕ) : Prop :=
  ∃ k : ℕ, arithmeticProgression n = geometricProgression k

def first10CommonElements : List ℕ :=
  List.map (λ i => geometricProgression (2 * i)) (List.range 10)

theorem sumOfFirst10CommonElements :
  (first10CommonElements.sum) = 3495250 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sumOfFirst10CommonElements_l1172_117271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_bag_theorem_l1172_117272

theorem coin_bag_theorem (p : ℕ+) : 
  let pennies := p
  let nickels := 3 * p
  let quarters := 12 * p
  let total_cents := pennies + 5 * nickels + 25 * quarters
  total_cents = 316 * p ∧ 
  (∃ (n : ℕ+), n * 316 = 316 ∨ n * 316 = 632 ∨ n * 316 = 948 ∨ n * 316 = 1264) := by
  sorry

#check coin_bag_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_bag_theorem_l1172_117272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_reciprocal_sum_constant_l1172_117262

/-- A hyperbola with focus F, eccentricity e, and focal parameter p -/
structure Hyperbola where
  F : EuclideanSpace ℝ (Fin 2)
  e : ℝ
  p : ℝ
  e_pos : e > 0
  p_pos : p > 0

/-- A point on the hyperbola -/
def PointOnHyperbola (H : Hyperbola) (P : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ (D : EuclideanSpace ℝ (Fin 2)), ‖P - H.F‖ / ‖P - D‖ = H.e

/-- Predicate to check if a point lies on a line segment between two other points -/
def SegmentThrough (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ B = t • A + (1 - t) • C

theorem hyperbola_reciprocal_sum_constant (H : Hyperbola) 
  (A B : EuclideanSpace ℝ (Fin 2)) (hA : PointOnHyperbola H A) (hB : PointOnHyperbola H B) 
  (hAFB : SegmentThrough A H.F B) : 
  1 / ‖A - H.F‖ + 1 / ‖B - H.F‖ = 2 / (H.e * H.p) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_reciprocal_sum_constant_l1172_117262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anna_marble_probability_l1172_117203

/-- The number of marbles in the bag -/
def total_marbles : ℕ := 800

/-- The number of colors -/
def num_colors : ℕ := 100

/-- The number of marbles of each color -/
def marbles_per_color : ℕ := 8

/-- The number of marbles drawn before the probability is calculated -/
def marbles_drawn : ℕ := 699

/-- The condition for Anna to stop drawing -/
def stop_condition : ℕ := 8

/-- The probability of stopping after drawing the next marble -/
def stop_probability : ℚ := 99 / 101

/-- Function to represent the count of marbles of a specific color -/
def count_color (drawn : ℕ) (color : ℕ) : ℕ := sorry

/-- Function to represent the probability of stopping on the next draw -/
def probability_of_stopping_next_draw (drawn : ℕ) : ℚ := sorry

theorem anna_marble_probability :
  (total_marbles = num_colors * marbles_per_color) →
  (marbles_drawn < total_marbles) →
  (∀ c : ℕ, c ≤ num_colors → count_color marbles_drawn c < stop_condition) →
  probability_of_stopping_next_draw marbles_drawn = stop_probability :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_anna_marble_probability_l1172_117203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_iris_to_tulip_ratio_l1172_117273

/-- The ratio of iris bulbs to tulip bulbs planted by Jane -/
theorem iris_to_tulip_ratio :
  let tulip_bulbs : ℕ := 20
  let daffodil_bulbs : ℕ := 30
  let crocus_bulbs : ℕ := 3 * daffodil_bulbs
  let total_earnings : ℚ := 75
  let earnings_per_bulb : ℚ := 1/2
  let total_bulbs : ℕ := (total_earnings / earnings_per_bulb).floor.toNat
  let iris_bulbs : ℕ := total_bulbs - tulip_bulbs - daffodil_bulbs - crocus_bulbs
  (iris_bulbs : ℚ) / tulip_bulbs = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_iris_to_tulip_ratio_l1172_117273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base5_1234_equals_194_l1172_117221

/-- Converts a base-5 number represented as a list of digits to its base-10 equivalent -/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- The base-5 number 1234 is equal to 194 in base-10 -/
theorem base5_1234_equals_194 :
  base5ToBase10 [1, 2, 3, 4] = 194 := by
  rfl

#eval base5ToBase10 [1, 2, 3, 4]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base5_1234_equals_194_l1172_117221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1172_117254

open Set

noncomputable section

variables {f : ℝ → ℝ}

axiom f_derivative_exists : Differentiable ℝ f

axiom f_equation (x : ℝ) : f x = 4 * x^2 - f (-x)

axiom f_derivative_inequality (x : ℝ) (h : x < 0) : 
  deriv f x + 1/2 < 4 * x

theorem range_of_m : 
  {m : ℝ | f (m + 1) ≤ f (-m) + 3 * m + 3/2} = Ici (-1/2) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1172_117254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_and_range_l1172_117209

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 < x ∧ x ≤ 3}
def B : Set ℝ := {x | 1 < x ∧ x < 3}
def C (m : ℝ) : Set ℝ := {x | m ≤ x}

-- State the theorem
theorem set_operations_and_range :
  ((Set.univ \ A) ∩ B = {x | 1 < x ∧ x ≤ 2}) ∧
  (∀ m : ℝ, (A ∪ B) ∩ C m ≠ ∅ → m ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_and_range_l1172_117209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_on_circle_max_value_achieved_l1172_117264

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1

-- Define the function to maximize
def f (x y : ℝ) : ℝ := 2*x + 3*y

-- Theorem statement
theorem max_value_on_circle :
  ∀ x y : ℝ, circle_eq x y → f x y ≤ Real.sqrt 13 - 2 :=
by
  sorry

-- Optionally, we can add a theorem for the existence of a point that achieves the maximum
theorem max_value_achieved :
  ∃ x y : ℝ, circle_eq x y ∧ f x y = Real.sqrt 13 - 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_on_circle_max_value_achieved_l1172_117264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_digit_divisibility_l1172_117236

def is_divisible (n m : ℕ) : Prop := m ∣ n

theorem unique_digit_divisibility :
  ∃! B : ℕ, B < 10 ∧
    (let number := 4627200 + B;
     is_divisible number 2 ∧
     is_divisible number 3 ∧
     is_divisible number 4 ∧
     is_divisible number 5 ∧
     is_divisible number 6 ∧
     is_divisible number 8 ∧
     is_divisible number 9) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_digit_divisibility_l1172_117236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l1172_117252

theorem expression_equality : (π - 4)^0 + |3 - Real.tan (π / 3)| - (1/2)^(-2 : Int) + Real.sqrt 27 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l1172_117252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l1172_117248

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Define the original expression
noncomputable def original_expr : ℝ := 4 / (cubeRoot 5 - cubeRoot 2)

-- Define the rationalized expression
noncomputable def rationalized_expr : ℝ := (cubeRoot 100 + cubeRoot 40 + cubeRoot 16) / 3

-- Theorem statement
theorem rationalize_denominator :
  original_expr = rationalized_expr := by
  sorry

#check rationalize_denominator

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l1172_117248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_cost_change_l1172_117233

/-- Calculates the final cost of an article after a price increase followed by a price decrease -/
noncomputable def finalCost (originalCost : ℝ) (increasePercentage : ℝ) (decreasePercentage : ℝ) : ℝ :=
  let increasedCost := originalCost * (1 + increasePercentage / 100)
  increasedCost * (1 - decreasePercentage / 100)

/-- Theorem stating that the final cost of an article originally priced at 75, 
    after a 20% increase and then a 20% decrease, is 72 -/
theorem article_cost_change : finalCost 75 20 20 = 72 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_cost_change_l1172_117233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_stream_and_home_l1172_117213

-- Define the coordinates of the cowboy's initial position, stream, and cabin
def cowboy_initial : ℝ × ℝ := (0, 5)
def stream_y : ℝ := 0
def cabin : ℝ × ℝ := (-10, 11)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem shortest_distance_to_stream_and_home :
  let cowboy_reflection : ℝ × ℝ := (cowboy_initial.1, -cowboy_initial.2)
  5 + distance cowboy_reflection cabin = 5 + Real.sqrt 356 := by
  sorry

#check shortest_distance_to_stream_and_home

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_stream_and_home_l1172_117213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_induction_step_expression_l1172_117269

theorem induction_step_expression (k : ℕ) : 
  (∀ n : ℕ, n > 0 → (Finset.prod (Finset.range n) (λ i => n + i + 1)) = 
    2^n * Finset.prod (Finset.range (2*n - 1)) (λ i => i + 1)) →
  ((Finset.prod (Finset.range (k+1)) (λ i => k + i + 2)) / 
   (Finset.prod (Finset.range k) (λ i => k + i + 1))) = 2 * (2 * k + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_induction_step_expression_l1172_117269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l1172_117287

noncomputable def Parabola := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

def Directrix := {p : ℝ × ℝ | p.1 = -1}

def Focus : ℝ × ℝ := (1, 0)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_triangle_area 
  (P : ℝ × ℝ) 
  (h1 : P ∈ Parabola) 
  (h2 : ∃ M ∈ Directrix, distance P M = 5) :
  let M := Classical.choose h2
  (1/2) * distance M Focus * distance P Focus = 10 := by
  sorry

#check parabola_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l1172_117287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_ratio_constant_l1172_117214

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents an ellipse in 2D space -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- The dot product of two vectors represented by points -/
def dotProduct (p1 p2 : Point) : ℝ := p1.x * p2.x + p1.y * p2.y

/-- Check if a point is on the ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Check if a point is on a line -/
def isOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem about the ratio of chord lengths on an ellipse -/
theorem ellipse_chord_ratio_constant
  (e : Ellipse)
  (l : Line)
  (m n a b : Point)
  (h1 : e.a = 2 ∧ e.b = Real.sqrt 3)
  (h2 : l.slope = Real.sqrt 2 ∨ l.slope = -Real.sqrt 2)
  (h3 : l.intercept = -l.slope)
  (h4 : isOnEllipse m e ∧ isOnEllipse n e)
  (h5 : isOnLine m l ∧ isOnLine n l)
  (h6 : dotProduct m n = -2)
  (h7 : isOnEllipse a e ∧ isOnEllipse b e)
  (h8 : a.x * b.y = a.y * b.x)  -- AB passes through origin
  (h9 : (m.y - n.y) * (a.x - b.x) = (m.x - n.x) * (a.y - b.y))  -- MN parallel to AB
  : (distance a b)^2 / distance m n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_ratio_constant_l1172_117214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1172_117268

/-- Definition of the ellipse E -/
noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of eccentricity -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

/-- Distance from focus to directrix line -/
noncomputable def focus_to_directrix (a b : ℝ) : ℝ :=
  let c := Real.sqrt (a^2 - b^2)
  |b * c - a * b| / Real.sqrt (a^2 + b^2)

theorem ellipse_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (he : eccentricity a b = 1/2)
  (hd : focus_to_directrix a b = Real.sqrt 21 / 7) :
  (∀ x y, ellipse a b x y ↔ ellipse 2 (Real.sqrt 3) x y) ∧
  (∀ A B : ℝ × ℝ, 
    ellipse 2 (Real.sqrt 3) A.1 A.2 → 
    ellipse 2 (Real.sqrt 3) B.1 B.2 → 
    A.1 * B.1 + A.2 * B.2 = 0 → 
    (2 * Real.sqrt 21 / 7)^2 * (B.2 - A.2)^2 = 
      ((B.1 - A.1)^2 + (B.2 - A.2)^2) * (A.1 * B.2 - A.2 * B.1)^2 / 
        ((B.1 - A.1)^2 + (B.2 - A.2)^2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1172_117268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_faces_two_dice_l1172_117211

/-- Represents a fair die with a given number of faces -/
structure Die where
  faces : ℕ
  h_min_faces : faces ≥ 6

/-- The probability of rolling a specific sum with two dice -/
noncomputable def prob_sum (d1 d2 : Die) (sum : ℕ) : ℚ :=
  (Finset.filter (fun (p : ℕ × ℕ) => p.1 + p.2 = sum) (Finset.product (Finset.range d1.faces) (Finset.range d2.faces))).card /
  (d1.faces * d2.faces : ℚ)

/-- The theorem statement -/
theorem min_faces_two_dice :
  ∀ d1 d2 : Die,
  prob_sum d1 d2 9 = (2/3 : ℚ) * prob_sum d1 d2 11 →
  prob_sum d1 d2 11 = (1/9 : ℚ) →
  d1.faces + d2.faces ≥ 19 ∧
  ∃ (d1' d2' : Die), d1'.faces + d2'.faces = 19 ∧
    prob_sum d1' d2' 9 = (2/3 : ℚ) * prob_sum d1' d2' 11 ∧
    prob_sum d1' d2' 11 = (1/9 : ℚ) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_faces_two_dice_l1172_117211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equality_l1172_117280

/-- Given real numbers a, b, and c satisfying the first series equation,
    prove that the second series sums to 1 -/
theorem series_sum_equality (a b c : ℝ) 
  (h : (a / b) + (a / b^2) + (2 * a / b^3) + (a / b^4) / (1 - 1/b) = 5) :
  (a / (a + c)) + (a / (a + c)^2) + (a / (a + c)^3) + 
    (a / (a + c)^4) / (1 - 1/(a + c)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equality_l1172_117280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_winning_strategy_l1172_117218

/-- Represents a game state with chosen numbers -/
structure GameState where
  numbers : Finset Nat

/-- Checks if three numbers form an arithmetic progression -/
def isArithmeticProgression (a b c : Nat) : Prop :=
  2 * b = a + c

/-- Checks if a game state contains an arithmetic progression -/
def containsArithmeticProgression (state : GameState) : Prop :=
  ∃ a b c, a ∈ state.numbers ∧ b ∈ state.numbers ∧ c ∈ state.numbers ∧
    a < b ∧ b < c ∧ isArithmeticProgression a b c

/-- Represents a strategy for the game -/
def Strategy := GameState → Nat

/-- Checks if a number is valid for the game -/
def isValidMove (n : Nat) (state : GameState) : Prop :=
  n ≤ 2018 ∧ n ∉ state.numbers

/-- Represents a winning strategy for the second player -/
def isWinningStrategyForSecondPlayer (strategy : Strategy) : Prop :=
  ∀ (firstMove : Nat),
    isValidMove firstMove { numbers := ∅ } →
    ∀ (gameState : GameState),
      gameState.numbers.card % 2 = 1 →
      isValidMove (strategy gameState) gameState →
      ¬containsArithmeticProgression { numbers := gameState.numbers ∪ {strategy gameState} }

/-- There exists a winning strategy for the second player -/
theorem second_player_winning_strategy :
  ∃ (strategy : Strategy), isWinningStrategyForSecondPlayer strategy := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_winning_strategy_l1172_117218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_1_003_rounded_l1172_117263

-- Define the function to round a number to 3 decimal places
noncomputable def round_to_3dp (x : ℝ) : ℝ :=
  (⌊x * 1000 + 0.5⌋ : ℝ) / 1000

-- State the theorem
theorem power_of_1_003_rounded : round_to_3dp (1.003^4) = 1.012 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_1_003_rounded_l1172_117263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_condition_range_condition_l1172_117224

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - a*x + 3*a) / Real.log 0.5

-- Theorem for the domain condition
theorem domain_condition (a : ℝ) :
  (∀ x, ∃ y, f a x = y) ↔ (0 < a ∧ a < 12) := by
  sorry

-- Theorem for the range condition
theorem range_condition (a : ℝ) :
  (∀ y, ∃ x, f a x = y) ↔ (a ≤ 0 ∨ a ≥ 12) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_condition_range_condition_l1172_117224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_monotonicity_condition_l1172_117238

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (x^2 + 1) - a * x

-- Theorem for part 1
theorem inequality_solution (a : ℝ) (h : a > 0) :
  ∀ x, f a x ≤ 1 ↔ 
    ((0 < a ∧ a < 1 ∧ 0 ≤ x ∧ x ≤ 2*a / (1 - a^2)) ∨
     (a ≥ 1 ∧ x ≥ 0)) := by sorry

-- Theorem for part 2
theorem monotonicity_condition (a : ℝ) :
  (∀ x y, 0 ≤ x → 0 ≤ y → x < y → f a x < f a y) ∨
  (∀ x y, 0 ≤ x → 0 ≤ y → x < y → f a x > f a y)
  ↔ (a ≥ 1 ∨ a ≤ 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_monotonicity_condition_l1172_117238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_EFCD_is_270_l1172_117275

/-- Represents a trapezoid ABCD with points E and F dividing the non-parallel sides -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  altitude : ℝ
  ratio_AD : ℝ
  ratio_BC : ℝ

/-- Calculates the area of quadrilateral EFCD in the given trapezoid -/
noncomputable def area_EFCD (t : Trapezoid) : ℝ :=
  let EF := t.ratio_AD * t.AB + (1 - t.ratio_AD) * t.CD
  let altitude_EFCD := (1 - t.ratio_AD) * t.altitude
  (EF + t.CD) * altitude_EFCD / 2

/-- Theorem stating that the area of EFCD is 270 square units for the given trapezoid -/
theorem area_EFCD_is_270 (t : Trapezoid) 
    (h1 : t.AB = 10) 
    (h2 : t.CD = 26) 
    (h3 : t.altitude = 15) 
    (h4 : t.ratio_AD = 1/4) 
    (h5 : t.ratio_BC = 1/4) : 
  area_EFCD t = 270 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_EFCD_is_270_l1172_117275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1172_117222

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2

-- Define the point of tangency
noncomputable def point : ℝ × ℝ := (1, 1/2)

-- State the theorem
theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ 2 * x - 2 * y - 1 = 0) ∧
    (m = (f (point.1 + 1) - f point.1) / 1) ∧
    (point.2 = m * point.1 + b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1172_117222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_fourth_in_expansion_l1172_117227

theorem coefficient_x_fourth_in_expansion :
  let n : ℕ := 6
  let k : ℕ := 4
  let coeff := (n.choose k) * ((-1 : ℤ)^(n - k))
  coeff = 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_fourth_in_expansion_l1172_117227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_isosceles_l1172_117220

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The length of a side of a triangle -/
noncomputable def sideLength (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The perimeter of a triangle -/
noncomputable def perimeter (t : Triangle) : ℝ :=
  sideLength t.A t.B + sideLength t.B t.C + sideLength t.C t.A

/-- The angle at vertex C of a triangle -/
noncomputable def angleC (t : Triangle) : ℝ :=
  Real.arccos ((sideLength t.A t.C)^2 + (sideLength t.B t.C)^2 - (sideLength t.A t.B)^2) / (2 * sideLength t.A t.C * sideLength t.B t.C)

/-- Theorem: The triangle with maximum perimeter is isosceles -/
theorem max_perimeter_isosceles (fixedAngle : ℝ) (fixedSideLength : ℝ) :
  ∃ (t : Triangle),
    angleC t = fixedAngle ∧
    sideLength t.A t.B = fixedSideLength ∧
    (∀ (t' : Triangle),
      angleC t' = fixedAngle →
      sideLength t'.A t'.B = fixedSideLength →
      perimeter t' ≤ perimeter t) →
    sideLength t.A t.C = sideLength t.B t.C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_isosceles_l1172_117220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_books_to_pens_ratio_l1172_117258

def total_stationery : ℕ := 400
def num_books : ℕ := 280

def num_pens : ℕ := total_stationery - num_books

theorem books_to_pens_ratio :
  (num_books / Nat.gcd num_books num_pens : ℚ) = 7 ∧ 
  (num_pens / Nat.gcd num_books num_pens : ℚ) = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_books_to_pens_ratio_l1172_117258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_ratio_l1172_117251

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle ABCD -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ :=
  (1/2) * base * height

/-- Main theorem -/
theorem rectangle_ratio (ABCD : Rectangle) (P : Point) :
  (ABCD.B.x - ABCD.A.x > ABCD.C.y - ABCD.B.y) →  -- AB > BC
  (triangleArea (ABCD.B.x - ABCD.A.x) (P.y - ABCD.A.y) = 3) →  -- Area APB = 3
  (triangleArea (ABCD.C.y - ABCD.B.y) (P.x - ABCD.B.x) = 4) →  -- Area BPC = 4
  (triangleArea (ABCD.B.x - ABCD.A.x) (ABCD.C.y - P.y) = 5) →  -- Area CPD = 5
  (triangleArea (ABCD.C.y - ABCD.B.y) (ABCD.A.x - P.x) = 6) →  -- Area DPA = 6
  ((ABCD.B.x - ABCD.A.x) / (ABCD.C.y - ABCD.B.y) = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_ratio_l1172_117251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_length_l1172_117288

/-- Calculates the length of a river given paddling conditions -/
theorem river_length (paddling_speed still_water_speed current_speed time : ℝ) 
  (h1 : paddling_speed > current_speed)
  (h2 : still_water_speed > 0)
  (h3 : current_speed > 0)
  (h4 : time > 0) :
  let effective_speed := still_water_speed - current_speed
  paddling_speed * time = effective_speed * time :=
by
  sorry

#check river_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_length_l1172_117288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_result_l1172_117242

/-- Recurrence sequence a_n -/
def a : ℕ → ℤ
  | 0 => 2
  | n + 1 => 2 * (a n)^2 - 1

theorem divisibility_result (N : ℕ) (p : ℕ) (x : ℤ) 
  (h1 : N ≥ 1) 
  (h2 : Nat.Prime p) 
  (h3 : (p : ℤ) ∣ a N) 
  (h4 : x^2 ≡ 3 [ZMOD p]) :
  (2^(N+2) : ℕ) ∣ (p - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_result_l1172_117242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_cosine_sum_l1172_117240

theorem chord_cosine_sum (r : ℝ) (γ δ : ℝ) : 
  0 < r →
  0 < γ →
  0 < δ →
  γ + δ < π →
  (5 : ℝ) = 2 * r * Real.sin (γ / 2) →
  (12 : ℝ) = 2 * r * Real.sin (δ / 2) →
  (13 : ℝ) = 2 * r * Real.sin ((γ + δ) / 2) →
  0 < Real.cos γ →
  ∃ (a b : ℕ), 
    Real.cos γ = (a : ℝ) / (b : ℝ) ∧ 
    Nat.Coprime a b ∧ 
    a + b = 288 := by
  sorry

#check chord_cosine_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_cosine_sum_l1172_117240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_club_members_l1172_117243

-- Define the sets of members
variable (A B C : Finset Nat)

-- Define the conditions
axiom club_condition : ∀ x, x ∈ (A ∪ B ∪ C)
axiom size_A : Finset.card A = 8
axiom size_B : Finset.card B = 7
axiom size_C : Finset.card C = 11
axiom intersection_AB : Finset.card (A ∩ B) ≥ 2
axiom intersection_BC : Finset.card (B ∩ C) ≥ 3
axiom intersection_AC : Finset.card (A ∩ C) ≥ 4

-- Theorem to prove
theorem max_club_members :
  Finset.card (A ∪ B ∪ C) ≤ 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_club_members_l1172_117243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2017_is_neg_sin_l1172_117283

open Real

-- Define the function f and its derivatives
noncomputable def f (x : ℝ) : ℝ := cos x

noncomputable def f_deriv : ℕ → (ℝ → ℝ)
| 0 => f
| (n + 1) => deriv (f_deriv n)

-- State the theorem
theorem f_2017_is_neg_sin :
  f_deriv 2017 = fun x => -sin x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2017_is_neg_sin_l1172_117283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_half_unit_square_l1172_117237

noncomputable section

-- Define the coordinate plane and the 3x3 grid
def grid_size : ℝ := 3

-- Define the lines
noncomputable def line1 (c : ℝ) (x : ℝ) : ℝ := (4 / (4 - c)) * (x - c)
noncomputable def line2 (d : ℝ) (x : ℝ) : ℝ := -d / 4 * x + d

-- Define the intersection point of the two lines
noncomputable def intersection (c d : ℝ) : ℝ × ℝ :=
  let x := (4 * d * (4 - c) - 16 * c) / (4 * (4 - c) + d)
  (x, line1 c x)

-- Define the area of the enclosed region
noncomputable def enclosed_area (c d : ℝ) : ℝ :=
  let (x, y) := intersection c d
  1/2 * x * y

-- Theorem statement
theorem enclosed_area_half_unit_square (c : ℝ) :
  enclosed_area c 2 = 1/2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_half_unit_square_l1172_117237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_F_l1172_117234

-- Define the integrand function
noncomputable def f (x : ℝ) : ℝ :=
  ((1 - Real.cos x) / (1 + Real.cos x))^2 / (3 * Real.sin x + 4 * Real.cos x + 5)

-- Define the antiderivative function
noncomputable def F (x : ℝ) : ℝ :=
  2 * (1/3 * Real.tan (x/2)^3 - 3 * Real.tan (x/2)^2 + 27 * Real.tan (x/2) - 
       108 * Real.log (abs (Real.tan (x/2) + 3)) - 81 / (Real.tan (x/2) + 3))

-- State the theorem
theorem integral_f_equals_F : 
  ∀ x : ℝ, ∃ C : ℝ, ∀ a b : ℝ, 
    ∫ t in a..b, f t = F b - F a + C := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_F_l1172_117234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l1172_117229

/-- The area of a rectangle with sides 5.9 cm and 3 cm is 17.7 cm² -/
theorem rectangle_area : ∃ (length width area : Real), 
  length = 5.9 ∧ width = 3 ∧ area = length * width ∧ area = 17.7 := by
  -- Introduce the variables
  let length : Real := 5.9
  let width : Real := 3
  let area : Real := length * width
  
  -- Prove the existence
  use length, width, area
  
  -- Prove the conjunction
  apply And.intro
  · rfl  -- length = 5.9
  apply And.intro
  · rfl  -- width = 3
  apply And.intro
  · rfl  -- area = length * width
  · -- Prove area = 17.7
    calc
      area = 5.9 * 3 := rfl
      _ = 17.7 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l1172_117229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_set_infinite_l1172_117217

/-- A set of points in the plane where each point is the midpoint of two others -/
def MidpointSet (E : Set (ℝ × ℝ)) : Prop :=
  ∀ x ∈ E, ∃ y z, y ∈ E ∧ z ∈ E ∧ y ≠ z ∧ x = ((y.1 + z.1) / 2, (y.2 + z.2) / 2)

/-- Theorem stating that a MidpointSet is infinite -/
theorem midpoint_set_infinite (E : Set (ℝ × ℝ)) (h : MidpointSet E) : Set.Infinite E := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_set_infinite_l1172_117217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_l1172_117260

def n : ℕ := 4^5 * 5^2 * 6^3 * Nat.factorial 7

theorem number_of_factors : (Finset.card (Finset.filter (· ∣ n) (Finset.range (n + 1)))) = 864 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_l1172_117260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_range_l1172_117274

theorem triangle_side_range (A B C : ℝ) (a b c : ℝ) :
  B = π / 3 →
  a = 4 →
  (∃ (A₁ A₂ : ℝ), A₁ ≠ A₂ ∧ 
    Real.sin A₁ = a * Real.sin B / b ∧
    Real.sin A₂ = a * Real.sin B / b ∧
    A₁ + B + C = π ∧
    A₂ + B + C = π) →
  2 * Real.sqrt 3 < b ∧ b < 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_range_l1172_117274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_both_readers_count_l1172_117292

/-- Represents the number of workers who have read both Saramago's and Kureishi's books -/
def both_readers (total workers saramago_readers kureishi_readers neither_readers : ℕ) : ℕ :=
  total - (saramago_readers - (total - saramago_readers - kureishi_readers - neither_readers)) -
          (kureishi_readers - (total - saramago_readers - kureishi_readers - neither_readers)) -
          neither_readers

/-- Theorem stating the number of workers who have read both books -/
theorem both_readers_count
  (total workers : ℕ) (saramago_readers kureishi_readers neither_readers : ℕ)
  (h_total : total = 150)
  (h_saramago : saramago_readers = total / 2)
  (h_kureishi : kureishi_readers = total / 6)
  (h_neither : neither_readers = saramago_readers - both_readers total workers saramago_readers kureishi_readers neither_readers - 1) :
  both_readers total workers saramago_readers kureishi_readers neither_readers = 24 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_both_readers_count_l1172_117292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completed_proof_l1172_117291

/-- The fraction of work completed by two workers in one day -/
noncomputable def work_completed_together (time_A : ℝ) (time_B : ℝ) : ℝ :=
  1 / time_A + 1 / time_B

theorem work_completed_proof (time_A time_B : ℝ) 
  (h1 : time_A = 4)
  (h2 : time_B = time_A / 2) :
  work_completed_together time_A time_B = 3/4 := by
  sorry

#check work_completed_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completed_proof_l1172_117291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_speed_l1172_117225

/-- The speed of the stream given the conditions of the rowing problem -/
theorem stream_speed (boat_speed : ℝ) (h : boat_speed = 54) :
  ∃ stream_speed : ℝ, 
    stream_speed = 18 ∧ 
    (boat_speed + stream_speed) = 2 * (boat_speed - stream_speed) := by
  use 18
  constructor
  · rfl
  · rw [h]
    norm_num

#eval 54 + 18 == 2 * (54 - 18)  -- This should evaluate to true

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_speed_l1172_117225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_racing_magic_time_is_360_l1172_117276

/-- The time (in seconds) it takes for the racing magic to circle the track once. -/
def racing_magic_time : ℕ := 360

/-- The number of rounds the charging bull makes in an hour. -/
def charging_bull_rounds_per_hour : ℕ := 40

/-- The time (in minutes) it takes for the racing magic and charging bull to meet at the starting point for the second time. -/
def time_to_second_meeting : ℕ := 6

/-- The racing magic is slower than the charging bull. -/
axiom racing_magic_slower : racing_magic_time > 3600 / charging_bull_rounds_per_hour

theorem racing_magic_time_is_360 : racing_magic_time = 360 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_racing_magic_time_is_360_l1172_117276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2014_eq_l1172_117244

/-- The function f(x) = x / (1 + x) for x ≥ 0 -/
noncomputable def f (x : ℝ) : ℝ := x / (1 + x)

/-- The nth iteration of f, defined recursively -/
noncomputable def f_n : ℕ → (ℝ → ℝ)
  | 0 => f
  | n + 1 => fun x => f (f_n n x)

/-- Main theorem: The 2014th iteration of f is x / (1 + 2014x) -/
theorem f_2014_eq (x : ℝ) (h : x ≥ 0) : f_n 2014 x = x / (1 + 2014 * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2014_eq_l1172_117244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_oranges_l1172_117259

theorem shopkeeper_oranges (oranges : ℕ) : 
  let bananas : ℕ := 400
  let orange_good_ratio : ℚ := 85 / 100
  let banana_good_ratio : ℚ := 94 / 100
  let total_good_ratio : ℚ := 886 / 1000
  (orange_good_ratio * oranges + banana_good_ratio * bananas) / (oranges + bananas) = total_good_ratio →
  oranges = 600 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_oranges_l1172_117259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_distance_approximation_l1172_117270

-- Define constants for the wheel properties
def wheel1_radius : ℝ := 22.4
def wheel1_revolutions : ℕ := 750
def wheel2_radius : ℝ := 15.8
def wheel2_revolutions : ℕ := 950

-- Define a function to calculate the circumference of a wheel
noncomputable def circumference (radius : ℝ) : ℝ := 2 * Real.pi * radius

-- Define a function to calculate the distance covered by a wheel
noncomputable def distance_covered (radius : ℝ) (revolutions : ℕ) : ℝ :=
  (circumference radius) * (revolutions : ℝ)

-- Theorem statement
theorem combined_distance_approximation :
  let d1 := distance_covered wheel1_radius wheel1_revolutions
  let d2 := distance_covered wheel2_radius wheel2_revolutions
  abs ((d1 + d2) - 199896.96) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_distance_approximation_l1172_117270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_domain_and_range_l1172_117212

-- Define the function h
noncomputable def h : ℝ → ℝ := sorry

-- Define the properties of h
axiom h_domain : ∀ x ∈ Set.Icc 1 4, h x ∈ Set.Icc 2 5

-- Define the function p
noncomputable def p (x : ℝ) : ℝ := 2 - h (x - 2)

theorem p_domain_and_range :
  (∀ x, x ∈ Set.Icc 3 6 ↔ p x ∈ Set.Icc (-3) 0) ∧
  (∀ y, y ∈ Set.Icc (-3) 0 → ∃ x ∈ Set.Icc 3 6, p x = y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_domain_and_range_l1172_117212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_traveled_l1172_117231

/-- The distance function of a landing plane with respect to time -/
noncomputable def distance (t : ℝ) : ℝ := 60 * t - (6/5) * t^2

/-- The theorem stating the maximum distance traveled by the plane -/
theorem max_distance_traveled : 
  ∃ (t : ℝ), ∀ (s : ℝ), distance s ≤ distance t ∧ distance t = 750 := by
  -- We'll use t = 25 as the maximum point
  use 25
  intro s
  have h1 : distance 25 = 750 := by
    simp [distance]
    norm_num
  
  have h2 : ∀ (s : ℝ), distance s ≤ distance 25 := by
    intro s
    -- The proof of this inequality requires more advanced techniques
    -- For now, we'll use sorry to skip the detailed proof
    sorry

  exact ⟨h2 s, h1⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_traveled_l1172_117231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_for_f_4_l1172_117299

def f (x : ℕ) : ℕ :=
  if x % 3 = 0 ∧ x % 5 = 0 then x / 15
  else if x % 3 = 0 then 5 * x
  else if x % 5 = 0 then 3 * x
  else x + 5

def f_iter : ℕ → ℕ → ℕ
  | 0, x => x
  | n + 1, x => f (f_iter n x)

theorem smallest_a_for_f_4 :
  (∃ a : ℕ, a > 1 ∧ f_iter a 4 = f 4) ∧
  (∀ a : ℕ, a > 1 ∧ a < 4 → f_iter a 4 ≠ f 4) ∧
  f_iter 4 4 = f 4 :=
by sorry

#eval f_iter 4 4
#eval f 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_for_f_4_l1172_117299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_packet_weight_is_correct_l1172_117228

-- Define the conversion rates and given quantities
noncomputable def pounds_per_ton : ℚ := 2100
noncomputable def ounces_per_pound : ℚ := 16
noncomputable def num_packets : ℚ := 1680
noncomputable def bag_capacity_tons : ℚ := 13

-- Define the weight of each packet
noncomputable def packet_weight : ℚ := (bag_capacity_tons * pounds_per_ton) / num_packets

-- Theorem to prove
theorem packet_weight_is_correct : packet_weight = 16.25 := by
  -- Unfold the definition of packet_weight
  unfold packet_weight
  -- Perform the calculation
  norm_num
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_packet_weight_is_correct_l1172_117228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1172_117204

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos (2 * x - Real.pi / 3)

theorem f_properties :
  (∀ k : ℤ, ∃ x₀ : ℝ, x₀ = 5 * Real.pi / 12 + k * Real.pi / 2 ∧
    (∀ x : ℝ, f (2 * x₀ - x) = f x)) ∧
  (∀ x : ℝ, f (x + Real.pi / 6) = 4 * Real.cos (2 * x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1172_117204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_ACED_l1172_117219

-- Define the equilateral triangle ABC
def ABC : Set (ℝ × ℝ) := sorry

-- Define the side length of ABC
def side_length_ABC : ℝ := 4

-- Define the right isosceles triangle DBE
def DBE : Set (ℝ × ℝ) := sorry

-- Define the side length of DBE
noncomputable def side_length_DBE : ℝ := Real.sqrt 2

-- Define the quadrilateral ACED
def ACED : Set (ℝ × ℝ) := sorry

-- Define necessary predicates
def Equilateral (s : Set (ℝ × ℝ)) : Prop := sorry
def SideLength (s : Set (ℝ × ℝ)) : ℝ := sorry
def RightIsosceles (s : Set (ℝ × ℝ)) : Prop := sorry
def IsSubset (s t : Set (ℝ × ℝ)) : Prop := sorry
def Perimeter (s : Set (ℝ × ℝ)) : ℝ := sorry

-- State the theorem
theorem perimeter_of_ACED :
  ∀ (ABC : Set (ℝ × ℝ)) (DBE : Set (ℝ × ℝ)) (ACED : Set (ℝ × ℝ)),
    Equilateral ABC ∧
    SideLength ABC = side_length_ABC ∧
    RightIsosceles DBE ∧
    SideLength DBE = side_length_DBE ∧
    IsSubset DBE ABC ∧
    ACED = ABC \ DBE →
    Perimeter ACED = 10 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_ACED_l1172_117219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_digit_numbers_not_product_of_five_digit_l1172_117223

theorem ten_digit_numbers_not_product_of_five_digit : 
  (90000 : ℕ) * (90000 + 1) < 9 * 10^9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_digit_numbers_not_product_of_five_digit_l1172_117223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_and_quotient_imply_divisor_l1172_117249

theorem remainder_and_quotient_imply_divisor (k n : ℕ) (hk : k > 0) (hn : n > 0) : 
  k % n = 11 ∧ (k : ℝ) / (n : ℝ) = 71.2 → n = 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_and_quotient_imply_divisor_l1172_117249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_valid_movements_is_81_l1172_117289

/-- Represents a cube with flies on its vertices -/
structure CubeWithFlies where
  vertices : Fin 8 → Bool

/-- Two vertices are diagonally opposite on a face of the cube -/
def DiagonallyOpposite (v w : Fin 8) : Prop := sorry

/-- Represents a valid movement of flies on the cube -/
def ValidMovement (initial final : CubeWithFlies) : Prop :=
  ∀ v : Fin 8, 
    initial.vertices v → 
    ∃! w : Fin 8, final.vertices w ∧ DiagonallyOpposite v w

/-- The number of valid movements of flies on the cube -/
def NumberOfValidMovements : ℕ := sorry

/-- Main theorem: The number of valid movements is 81 -/
theorem number_of_valid_movements_is_81 : 
  NumberOfValidMovements = 81 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_valid_movements_is_81_l1172_117289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cotton_per_tshirt_is_four_l1172_117207

/-- The number of feet of cotton needed for one tee-shirt, given that 15 tee-shirts can be made with 60 feet of material. -/
noncomputable def cotton_per_tshirt : ℚ :=
  60 / 15

/-- Theorem stating that the number of feet of cotton needed for one tee-shirt is 4. -/
theorem cotton_per_tshirt_is_four : cotton_per_tshirt = 4 := by
  -- Unfold the definition of cotton_per_tshirt
  unfold cotton_per_tshirt
  -- Simplify the fraction
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cotton_per_tshirt_is_four_l1172_117207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2018_equals_negative_18_l1172_117296

noncomputable def f (a b α β : ℝ) (x : ℝ) : ℝ := 
  a * Real.sin (Real.pi / 2 * x + α) + b * Real.cos (Real.pi / 2 * x + β)

theorem f_2018_equals_negative_18 (a b α β : ℝ) (h : f a b α β 8 = 18) : 
  f a b α β 2018 = -18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2018_equals_negative_18_l1172_117296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitive_parallel_unique_scalar_l1172_117261

/-- Represent a plane vector -/
structure PlaneVector where
  x : ℝ
  y : ℝ

/-- Define parallelism for plane vectors -/
def parallel (a b : PlaneVector) : Prop :=
  ∃ (k : ℝ), b = PlaneVector.mk (k * a.x) (k * a.y)

/-- Theorem for transitivity of parallelism -/
theorem parallel_transitive (a b c : PlaneVector) 
  (ha : a ≠ PlaneVector.mk 0 0) 
  (hb : b ≠ PlaneVector.mk 0 0) 
  (hc : c ≠ PlaneVector.mk 0 0) 
  (hab : parallel a b) 
  (hbc : parallel b c) : 
  parallel a c := by sorry

/-- Theorem for unique scalar multiple for parallel vectors -/
theorem parallel_unique_scalar (a b : PlaneVector) 
  (ha : a ≠ PlaneVector.mk 0 0) 
  (hab : parallel a b) : 
  ∃! k : ℝ, b = PlaneVector.mk (k * a.x) (k * a.y) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitive_parallel_unique_scalar_l1172_117261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flea_game_theorem_l1172_117265

/-- A flea's position on a line --/
structure Flea where
  position : ℝ

/-- A configuration of n fleas on a line --/
structure FleaConfiguration (n : ℕ) where
  fleas : Fin n → Flea
  not_all_same : ∃ i j, i ≠ j ∧ (fleas i).position ≠ (fleas j).position

/-- The flea movement game --/
def FleaGame (n : ℕ) (lambda : ℝ) :=
  lambda > 0 ∧ n ≥ 2

/-- The winning condition for the flea game --/
def CanMoveAllRight (n : ℕ) (lambda : ℝ) : Prop :=
  ∀ (M : ℝ) (config : FleaConfiguration n),
    ∃ (k : ℕ), ∀ i, ((config.fleas i).position > M)

/-- The main theorem about the flea game --/
theorem flea_game_theorem (n : ℕ) (lambda : ℝ) (h : FleaGame n lambda) :
  CanMoveAllRight n lambda ↔ lambda ≥ 1 / (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flea_game_theorem_l1172_117265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_series_l1172_117267

/-- The sum of an infinite geometric series with first term a and common ratio r -/
noncomputable def infiniteGeometricSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- Proof that the first term of an infinite geometric series with common ratio 1/4 and sum 40 is 30 -/
theorem first_term_of_geometric_series : ∃ a : ℝ, 
  infiniteGeometricSum a (1/4 : ℝ) = 40 ∧ a = 30 := by
  use 30
  constructor
  · simp [infiniteGeometricSum]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_series_l1172_117267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_formula_isosceles_trapezoid_solvability_isosceles_trapezoid_area_special_case_l1172_117266

/-- Isosceles trapezoid with parallel sides a and c, where a > c -/
structure IsoscelesTrapezoid (a c : ℝ) : Type where
  a_gt_c : a > c
  b : ℝ
  height_eq : b - c = (a + c) * (a - 3 * c) / (8 * c)

/-- Area of an isosceles trapezoid -/
noncomputable def area (a c : ℝ) (t : IsoscelesTrapezoid a c) : ℝ :=
  (a + c)^2 * (a - 3 * c) / (16 * c)

theorem isosceles_trapezoid_area_formula (a c : ℝ) (t : IsoscelesTrapezoid a c) :
  area a c t = (a + c)^2 * (a - 3 * c) / (16 * c) := by sorry

theorem isosceles_trapezoid_solvability (a c : ℝ) (t : IsoscelesTrapezoid a c) :
  a > 3 * c := by sorry

theorem isosceles_trapezoid_area_special_case (a : ℝ) (t : IsoscelesTrapezoid a (a / 6)) :
  area a (a / 6) t = 49 * a^2 / 192 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_formula_isosceles_trapezoid_solvability_isosceles_trapezoid_area_special_case_l1172_117266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alberts_earnings_increase_l1172_117235

/-- Given Albert's initial monthly earnings and two scenarios of increased earnings,
    calculate the new percentage increase in his earnings. -/
theorem alberts_earnings_increase (E : ℝ) (P : ℝ) : 
  E + 0.14 * E = 678 →
  E + P * E = 683.95 →
  abs (P - 0.1496) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alberts_earnings_increase_l1172_117235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1172_117256

noncomputable def f (α : ℝ) (x : ℕ) : ℝ := Real.sin α ^ x + Real.cos α ^ x

theorem f_range (k : ℕ) (h : k > 0) :
  ∀ α, 1 / 2^(k - 1) ≤ f α (2 * k) ∧ f α (2 * k) ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1172_117256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l1172_117281

noncomputable def z : ℂ := (Complex.I * (3 - 4 * Complex.I)) / (1 - Complex.I)

theorem z_in_first_quadrant : Real.sign z.re = 1 ∧ Real.sign z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l1172_117281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_transformation_theorem_l1172_117293

/-- The transformed cosine function -/
noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x + 2 * Real.pi / 3)

/-- Condition: g has exactly two zeros in [0, 2π/3] -/
def has_two_zeros (ω : ℝ) : Prop :=
  ∃ x₁ x₂, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 2 * Real.pi / 3 ∧
    g ω x₁ = 0 ∧ g ω x₂ = 0 ∧
    ∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi / 3 ∧ g ω x = 0 → x = x₁ ∨ x = x₂

/-- Condition: g is strictly decreasing in [-π/12, π/12] -/
def is_strictly_decreasing (ω : ℝ) : Prop :=
  ∀ x₁ x₂, -Real.pi/12 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ Real.pi/12 → g ω x₁ > g ω x₂

/-- The main theorem -/
theorem cosine_transformation_theorem :
  {ω : ℝ | ω > 0 ∧ has_two_zeros ω ∧ is_strictly_decreasing ω} = {x | 11/4 ≤ x ∧ x ≤ 4} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_transformation_theorem_l1172_117293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_even_function_l1172_117246

open Real Set

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem periodic_even_function 
  (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_period : has_period f π) 
  (h_def : ∀ x ∈ Icc 0 (π/2), f x = 1 - sin x) :
  ∀ x ∈ Icc (5/2*π) (3*π), f x = 1 - sin x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_even_function_l1172_117246
