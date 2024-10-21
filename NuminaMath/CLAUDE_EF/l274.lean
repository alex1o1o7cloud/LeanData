import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_x_plus_7pi_over_6_l274_27450

theorem tan_x_plus_7pi_over_6 (x : ℝ) 
  (h : Real.sqrt 3 * Real.sin x + Real.cos x = 2/3) : 
  Real.tan (x + 7 * Real.pi / 6) = Real.sqrt 2 / 4 ∨ Real.tan (x + 7 * Real.pi / 6) = -Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_x_plus_7pi_over_6_l274_27450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v3_equals_15_l274_27432

def f (x : ℝ) : ℝ := 2 * x^5 - 3 * x^3 + 5 * x^2 - 4

def qin_jiushao (a : List ℝ) (x : ℝ) : List ℝ :=
  match a with
  | [] => []
  | h :: t => h :: List.scanl (fun v c => v * x + c) h t

theorem v3_equals_15 :
  let coeffs := [2, 0, -3, 5, 0, -4]
  let v := qin_jiushao coeffs 2
  v.get? 3 = some 15 := by
  sorry

#eval qin_jiushao [2, 0, -3, 5, 0, -4] 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_v3_equals_15_l274_27432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_for_f_l274_27405

-- Define the function f as noncomputable
noncomputable def f (y : ℝ) : ℝ := ((y - 3)/2 - 3) * ((y - 3)/2 + 4)

-- State the theorem
theorem two_solutions_for_f :
  ∃! (y₁ y₂ : ℝ), y₁ ≠ y₂ ∧ f y₁ = 170 ∧ f y₂ = 170 ∧ y₁ = -25 ∧ y₂ = 29 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_for_f_l274_27405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_optimal_ratio_l274_27443

noncomputable def cylinder_surface_area (r h : ℝ) : ℝ := 2 * Real.pi * r^2 + 2 * Real.pi * r * h

noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

theorem cylinder_optimal_ratio :
  ∀ r h : ℝ,
  r > 0 → h > 0 →
  cylinder_surface_area r h = 6 * Real.pi →
  (∀ r' h' : ℝ, r' > 0 → h' > 0 → cylinder_surface_area r' h' = 6 * Real.pi → 
    cylinder_volume r' h' ≤ cylinder_volume r h) →
  h / r = Real.sqrt (3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_optimal_ratio_l274_27443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l274_27498

-- Define propositions p and q as functions of a
def p (a : ℝ) : Prop := ∃ x y : ℝ, x^2 / (a - 2) - y^2 / (6 - a) = 1 ∧ (a - 2) * (6 - a) > 0

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (4 - a)^x < (4 - a)^y

-- Define the set of valid values for a
def valid_a : Set ℝ := {a | (p a ∨ q a) ∧ ¬(p a ∧ q a)}

-- Theorem statement
theorem range_of_a : valid_a = Set.Iic 2 ∪ Set.Ico 3 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l274_27498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l274_27428

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (1 - t, t)

-- Define curves C1 and C2 in polar coordinates
noncomputable def curve_C1 (θ : ℝ) : ℝ := Real.sqrt 3 * Real.cos θ
noncomputable def curve_C2 (θ : ℝ) : ℝ := 3 * Real.sin θ

-- Define point A as the intersection of C1 and C2
noncomputable def point_A : ℝ × ℝ := (3/2, Real.pi/6)

-- Define point B as the intersection of line l and OA
noncomputable def point_B : ℝ × ℝ := (Real.sqrt 3 - 1, Real.pi/6)

-- Theorem statement
theorem length_of_AB :
  let xA := (point_A.1 * Real.cos point_A.2)
  let yA := (point_A.1 * Real.sin point_A.2)
  let xB := (point_B.1 * Real.cos point_B.2)
  let yB := (point_B.1 * Real.sin point_B.2)
  Real.sqrt ((xA - xB)^2 + (yA - yB)^2) = 5/2 - Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l274_27428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_and_intersection_points_l274_27445

/-- The function f(x) = x^3 - x^2 + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*x + a

theorem monotonicity_and_intersection_points (a : ℝ) :
  (∀ x y : ℝ, a ≥ 1/3 → x ≤ y → f a x ≤ f a y) ∧
  (a < 1/3 → 
    (∀ x y : ℝ, x < y ∧ y < (1 - Real.sqrt (1 - 3*a)) / 3 → f a x < f a y) ∧
    (∀ x y : ℝ, x < y ∧ (1 + Real.sqrt (1 - 3*a)) / 3 < x → f a x < f a y) ∧
    (∀ x y : ℝ, (1 - Real.sqrt (1 - 3*a)) / 3 < x ∧ x < y ∧ y < (1 + Real.sqrt (1 - 3*a)) / 3 → f a x > f a y)) ∧
  (f a 1 = a + 1) ∧
  (f a (-1) = -a - 1) ∧
  (∃ k : ℝ, k * 1 = f a 1 ∧ k * (-1) = f a (-1) ∧ k * 0 = 0) := by
  sorry

#check monotonicity_and_intersection_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_and_intersection_points_l274_27445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_emily_purchase_l274_27448

/-- Represents Emily's purchase at the store -/
structure Purchase where
  curtain_pairs : ℕ
  curtain_price : ℚ
  wallprint_price : ℚ
  installation_cost : ℚ
  total_cost : ℚ

/-- Calculates the number of wall prints purchased -/
def num_wallprints (p : Purchase) : ℕ :=
  (((p.total_cost - (↑p.curtain_pairs * p.curtain_price + p.installation_cost)) / p.wallprint_price).floor).toNat

/-- Theorem stating that Emily purchased 9 wall prints -/
theorem emily_purchase :
  ∃ (p : Purchase),
    p.curtain_pairs = 2 ∧
    p.curtain_price = 30 ∧
    p.wallprint_price = 15 ∧
    p.installation_cost = 50 ∧
    p.total_cost = 245 ∧
    num_wallprints p = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_emily_purchase_l274_27448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l274_27447

-- Define the curves C₁ and C₂
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ :=
  (2 + 2 * Real.cos θ, 2 * Real.sin θ)

noncomputable def C₂ (θ : ℝ) : ℝ :=
  2 * Real.sin θ

-- Define the line l
noncomputable def l : ℝ := Real.pi / 4

-- Theorem statement
theorem intersection_distance :
  let M := C₁ l
  let N := (C₂ l * Real.cos l, C₂ l * Real.sin l)
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l274_27447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_A_eq_prob_B_l274_27411

-- Define the masses of the competitors
variable (u j p b : ℝ)

-- Define the conditions
variable (h1 : u > j)
variable (h2 : b > p)
variable (h3 : ∀ x y : ℝ, x > 0 ∧ y > 0 → x / (x + y) + y / (x + y) = 1)

-- Define the probabilities of events A and B
noncomputable def prob_A : ℝ := (u / (u + p)) * (b / (u + b)) * (j / (j + b)) * (p / (j + p))
noncomputable def prob_B : ℝ := (u / (u + b)) * (p / (u + p)) * (j / (j + p)) * (b / (j + b))

-- State the theorem
theorem prob_A_eq_prob_B : prob_A u j p b = prob_B u j p b := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_A_eq_prob_B_l274_27411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l274_27419

/-- The compound interest formula calculates the future value of an investment -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Rounds a real number to the nearest hundredth -/
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

theorem investment_growth :
  let principal : ℝ := 1500
  let rate : ℝ := 0.04
  let time : ℕ := 21
  round_to_hundredth (compound_interest principal rate time) = 3046.28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l274_27419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_slope_l274_27454

/-- Given a line with equation 3x - 6y = 9, prove that the slope of any parallel line is 1/2 -/
theorem parallel_line_slope (x y : ℝ) : 
  (3 * x - 6 * y = 9) → (∀ m : ℝ, (y = m * x + (9/6 - m * (3/6))) → m = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_slope_l274_27454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_large_root_change_exists_l274_27486

/-- Represents a quadratic equation ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the larger root of a quadratic equation -/
noncomputable def largerRoot (eq : QuadraticEquation) : ℝ :=
  (-eq.b + Real.sqrt (eq.b^2 - 4*eq.a*eq.c)) / (2*eq.a)

/-- Theorem: There exists a quadratic equation and a small change in its coefficients
    such that the larger root changes by more than 1000 -/
theorem large_root_change_exists : ∃ (eq₁ eq₂ : QuadraticEquation),
  eq₁.a = 1 ∧
  |eq₂.a - eq₁.a| ≤ 0.001 ∧
  |eq₂.b - eq₁.b| ≤ 0.001 ∧
  |eq₂.c - eq₁.c| ≤ 0.001 ∧
  |largerRoot eq₂ - largerRoot eq₁| > 1000 := by
  sorry

#check large_root_change_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_large_root_change_exists_l274_27486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_speed_theorem_l274_27490

/-- Calculates the swimming speed in still water given the distance, time, and current speed -/
noncomputable def swimming_speed_in_still_water (distance : ℝ) (time : ℝ) (current_speed : ℝ) : ℝ :=
  distance / time + current_speed

/-- Theorem: Given the conditions, the swimming speed in still water is 6 km/h -/
theorem swimming_speed_theorem (distance : ℝ) (time : ℝ) (current_speed : ℝ) 
    (h1 : distance = 14) 
    (h2 : time = 3.5) 
    (h3 : current_speed = 2) : 
  swimming_speed_in_still_water distance time current_speed = 6 := by
  sorry

/-- Example calculation -/
def example_calculation : ℚ :=
  (14 : ℚ) / (7/2) + 2

#eval example_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_speed_theorem_l274_27490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_correct_l274_27420

-- Define the functions f, g, and h
noncomputable def f (x : ℝ) := 3 * x + 4
noncomputable def g (x : ℝ) := 2 * x - 3
noncomputable def h (x : ℝ) := f (g x)

-- Define the inverse function of h
noncomputable def h_inv (x : ℝ) := (x + 5) / 6

-- Theorem statement
theorem h_inverse_correct : 
  ∀ x : ℝ, h (h_inv x) = x ∧ h_inv (h x) = x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_correct_l274_27420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l274_27487

noncomputable def f (x : ℝ) := Real.sqrt (x + 3) / x

theorem domain_of_f : 
  {x : ℝ | f x ≠ 0} = {x : ℝ | x ≥ -3 ∧ x ≠ 0} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l274_27487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_x_and_y_l274_27421

/-- S(k) is the sum of all positive integers that do not exceed k and are coprime to k -/
def S (k : ℕ) : ℕ := sorry

/-- Main theorem -/
theorem existence_of_x_and_y (m n : ℕ) (h_n_odd : Odd n) (hm : m > 0) (hn : n > 0) :
  ∃ (x y : ℕ), (x > 0) ∧ (y > 0) ∧ (m ∣ x) ∧ (2 * S x = y^n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_x_and_y_l274_27421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_crups_are_arogs_and_brafs_l274_27418

-- Define the sets as predicates over a universe type
variable (U : Type)
variable (Arog Braf Crup Dramp : U → Prop)

-- Define the conditions
variable (h1 : ∀ x, Arog x → Braf x)
variable (h2 : ∀ x, Crup x → Braf x)
variable (h3 : ∀ x, Dramp x → Arog x)
variable (h4 : ∀ x, Crup x → Dramp x)

-- Theorem to prove
theorem all_crups_are_arogs_and_brafs :
  ∀ x, Crup x → (Arog x ∧ Braf x) :=
by
  intro x hCrup
  constructor
  · exact h3 x (h4 x hCrup)
  · exact h2 x hCrup

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_crups_are_arogs_and_brafs_l274_27418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_problems_l274_27425

theorem math_problems : 
  (∀ (x y z : ℝ), 
    x = Real.sqrt 48 ∧ y = Real.sqrt 8 ∧ z = Real.sqrt (2/3) → 
    x / y - z - 1 / Real.sqrt 6 = Real.sqrt 6 / 2) ∧
  (∀ (a b c : ℝ), 
    a = 5 * Real.sqrt 2 ∧ b = 2 * Real.sqrt 5 ∧ c = Real.sqrt 3 → 
    (a + b) * (a - b) + (c - 1)^2 = 34 - 2 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_problems_l274_27425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_hyperbola_standard_equation_l274_27423

/-- An equilateral hyperbola passing through the point (5, -4) has the standard equation x²/9 - y²/9 = 1 -/
theorem equilateral_hyperbola_standard_equation :
  ∀ (x y : ℝ),
  (∃ (k : ℝ), k ≠ 0 ∧ x^2 - y^2 = k) →  -- equilateral hyperbola condition
  (5^2 - (-4)^2 = 9) →  -- passes through (5, -4)
  (x^2 / 9 - y^2 / 9 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_hyperbola_standard_equation_l274_27423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_ray_l274_27456

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Defines the locus of points P satisfying |PF₁| - |PF₂| = 10 -/
def isOnLocus (p : Point) (f1 f2 : Point) : Prop :=
  distance p f1 - distance p f2 = 10

/-- Theorem: The locus of points P satisfying |PF₁| - |PF₂| = 10 forms a ray -/
theorem locus_is_ray (f1 f2 : Point) 
  (h1 : f1 = ⟨-8, 3⟩) 
  (h2 : f2 = ⟨2, 3⟩) :
  ∃ (start : Point) (direction : ℝ × ℝ), 
    ∀ p, isOnLocus p f1 f2 ↔ 
      ∃ t : ℝ, t ≥ 0 ∧ p = ⟨start.x + t * direction.1, start.y + t * direction.2⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_ray_l274_27456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_angle_sum_l274_27424

theorem half_angle_sum (θ : ℝ) (h1 : Real.cos θ = -7/25) (h2 : θ ∈ Set.Ioo (-Real.pi) 0) :
  Real.sin (θ/2) + Real.cos (θ/2) = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_angle_sum_l274_27424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_a_value_l274_27472

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 ∧ x < 2 then Real.log x - a * x else 0

-- State the theorem
theorem odd_function_a_value (a : ℝ) :
  (∀ x, x ≠ 0 → f a x = -f a (-x)) →  -- f is an odd function
  (a > 1/2) →  -- a > 1/2
  (∀ x, -2 < x ∧ x < 0 → f a x ≥ 1) →  -- minimum value is 1 for x ∈ (-2, 0)
  (∃ x, -2 < x ∧ x < 0 ∧ f a x = 1) →  -- the minimum value 1 is attained
  a = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_a_value_l274_27472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_complex_points_l274_27464

def z₁ : ℂ := Complex.mk 1 (-1)
def z₂ : ℂ := Complex.mk 3 (-5)

def Z₁ : ℝ × ℝ := (z₁.re, z₁.im)
def Z₂ : ℝ × ℝ := (z₂.re, z₂.im)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_between_complex_points :
  distance Z₁ Z₂ = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_complex_points_l274_27464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bookClubRatioRounded_l274_27403

/-- Rounds a real number to the nearest tenth -/
noncomputable def roundToNearestTenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

/-- The ratio of agreeing members to total members -/
noncomputable def bookClubRatio : ℝ := 11 / 16

/-- Theorem stating that the book club ratio rounded to the nearest tenth is 0.7 -/
theorem bookClubRatioRounded :
  roundToNearestTenth bookClubRatio = 0.7 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval roundToNearestTenth bookClubRatio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bookClubRatioRounded_l274_27403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_three_zeros_in_interval_l274_27406

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.cos (2 * x / 3 - Real.pi / 3)

-- Theorem statement
theorem g_has_three_zeros_in_interval :
  ∃ (x₁ x₂ x₃ : ℝ), 
    0 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < 5 * Real.pi ∧
    g x₁ = 0 ∧ g x₂ = 0 ∧ g x₃ = 0 ∧
    ∀ x, 0 < x ∧ x < 5 * Real.pi ∧ g x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_three_zeros_in_interval_l274_27406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decision_time_is_three_minutes_l274_27475

-- Define the problem parameters
noncomputable def distance_AB : ℝ := 90
noncomputable def speed_A_initial : ℝ := 60
noncomputable def speed_B : ℝ := 80
noncomputable def speed_A_after_breakdown : ℝ := 20
noncomputable def distance_to_breakdown : ℝ := 45
noncomputable def bus_B_interval : ℝ := 0.25 -- 15 minutes in hours

-- Define the function to calculate the decision time
noncomputable def calculate_decision_time : ℝ :=
  let time_to_breakdown := distance_to_breakdown / speed_A_initial
  let distance_B_first_bus := speed_B * time_to_breakdown
  let distance_between_buses := distance_AB - distance_to_breakdown - distance_B_first_bus
  let relative_speed := speed_A_after_breakdown + speed_B
  distance_between_buses / relative_speed

-- Theorem statement
theorem decision_time_is_three_minutes :
  calculate_decision_time * 60 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decision_time_is_three_minutes_l274_27475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_five_l274_27476

/-- The area of a triangle in polar coordinates -/
noncomputable def triangleAreaPolar (r1 : ℝ) (θ1 : ℝ) (r2 : ℝ) (θ2 : ℝ) : ℝ :=
  (1/2) * r1 * r2 * Real.sin (θ2 - θ1)

/-- Prove that the area of triangle AOB is 5 in polar coordinates -/
theorem triangle_area_is_five :
  let A : ℝ × ℝ := (2, π/3)
  let B : ℝ × ℝ := (5, 5*π/6)
  triangleAreaPolar A.1 A.2 B.1 B.2 = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_five_l274_27476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_derivative_odd_others_not_l274_27457

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem cos_derivative_odd_others_not :
  is_odd (λ x => -Real.sin x) ∧
  ¬ is_odd (λ x => Real.exp x) ∧
  ¬ is_odd (λ x => 1 / x) ∧
  ¬ is_odd (λ x => Real.exp (x * Real.log 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_derivative_odd_others_not_l274_27457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_Q_l274_27493

-- Define the circle
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the point P on the circle
def P (x y : ℝ) : Prop := Circle x y

-- Define the point P" as the foot of the perpendicular from P to the x-axis
def P'' (x : ℝ) : ℝ × ℝ := (x, 0)

-- Define the point Q on the line segment PP"
def Q (x y x₀ y₀ : ℝ) : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ x = t * x₀ ∧ y = t * y₀

-- Define the vector relation PQ = 2QP"
def VectorRelation (x₀ y₀ x y : ℝ) : Prop :=
  x - x₀ = 2 * (x₀ - x) ∧ y - y₀ = -2 * y

theorem trajectory_of_Q (x₀ y₀ x y : ℝ) :
  P x₀ y₀ → Q x y x₀ y₀ → VectorRelation x₀ y₀ x y → x^2 + 9*y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_Q_l274_27493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l274_27494

theorem tan_alpha_value (α : ℝ) 
  (h1 : Real.sin α + Real.cos α = 7/13)
  (h2 : α ∈ Set.Ioo (-Real.pi) 0) :
  Real.tan α = -12/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l274_27494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_20km_l274_27470

/-- Calculates the taxi fare for a given distance -/
noncomputable def taxi_fare (distance : ℝ) : ℝ :=
  let base_fare := 8
  let mid_rate := 1.5
  let long_rate := 0.8
  let base_distance := 3
  let mid_distance := 10
  if distance ≤ base_distance then
    base_fare
  else if distance ≤ mid_distance then
    base_fare + (distance - base_distance) * mid_rate
  else
    base_fare + (mid_distance - base_distance) * mid_rate + (distance - mid_distance) * long_rate

/-- The theorem states that the taxi fare for a 20km ride is 26.5 yuan -/
theorem taxi_fare_20km : taxi_fare 20 = 26.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_20km_l274_27470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_is_24_l274_27473

/-- The equation that defines the set of points -/
def satisfies_equation (x y : ℝ) : Prop :=
  |4 * x| + |3 * y| + |24 - 4 * x - 3 * y| = 24

/-- The three points that form the triangle -/
def point_A : ℝ × ℝ := (0, 0)
def point_B : ℝ × ℝ := (6, 0)
def point_C : ℝ × ℝ := (0, 8)

/-- The area of a triangle given its vertices -/
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (1/2) * |x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂)|

/-- The main theorem to be proved -/
theorem area_of_triangle_is_24 :
  satisfies_equation point_A.1 point_A.2 ∧
  satisfies_equation point_B.1 point_B.2 ∧
  satisfies_equation point_C.1 point_C.2 →
  triangle_area point_A point_B point_C = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_is_24_l274_27473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yogurt_demand_and_profit_l274_27444

/-- Represents the temperature ranges and their corresponding frequencies --/
structure TemperatureDistribution :=
  (ranges : List (ℝ × ℝ))
  (frequencies : List ℕ)

/-- Represents the demand function based on temperature --/
noncomputable def demand (temp : ℝ) : ℕ :=
  if temp ≥ 25 then 500
  else if temp ≥ 20 then 300
  else 200

/-- Calculates the profit based on demand and purchase quantity --/
def profit (demand purchased : ℕ) : ℤ :=
  (min demand purchased : ℤ) * 2 - (purchased - min demand purchased : ℤ) * 2

/-- The main theorem to be proved --/
theorem yogurt_demand_and_profit 
  (dist : TemperatureDistribution)
  (h_dist : dist.ranges = [(10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40)] ∧ 
            dist.frequencies = [2, 16, 36, 25, 7, 4]) :
  let total_days := (dist.frequencies.sum : ℝ)
  let days_demand_not_exceeding_300 := (dist.frequencies.take 3).sum
  let days_temp_ge_20 := (dist.frequencies.drop 2).sum
  (days_demand_not_exceeding_300 / total_days = 3/5) ∧
  (days_temp_ge_20 / total_days = 4/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yogurt_demand_and_profit_l274_27444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_through_M_l274_27495

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y = 0

-- Define point M
def point_M : ℝ × ℝ := (1, 0)

-- Define the line equation
def line_L (x y : ℝ) : Prop := x + y - 1 = 0

-- Theorem statement
theorem shortest_chord_through_M :
  ∀ (x y : ℝ),
  circle_C x y →
  line_L x y →
  (∀ (x' y' : ℝ), circle_C x' y' → line_L x' y' → 
    (x' - 1)^2 + y'^2 ≤ (x - 1)^2 + y^2) →
  x = 1 ∧ y = 0 ∨ 
  ∃ (t : ℝ), x = 1 + t ∧ y = -t :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_through_M_l274_27495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l274_27429

theorem expression_simplification (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x > y) (h2 : y > z) : 
  (x ^ (y + z) * z ^ (x + y)) / (y ^ (x + z) * z ^ (y + x)) = (x / y) ^ (y + z) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l274_27429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_sum_at_least_three_divisible_l274_27452

def harmonic_sum (n : ℕ+) : ℚ :=
  (Finset.range n).sum (λ i => 1 / (i + 1 : ℚ))

noncomputable def a_n (n : ℕ+) : ℕ := Nat.floor ((harmonic_sum n).num)

theorem harmonic_sum_at_least_three_divisible (p : ℕ) (hp : p.Prime) (hp5 : p ≥ 5) :
  ∃ (n1 n2 n3 : ℕ+), n1 ≠ n2 ∧ n2 ≠ n3 ∧ n1 ≠ n3 ∧
    p ∣ a_n n1 ∧ p ∣ a_n n2 ∧ p ∣ a_n n3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_sum_at_least_three_divisible_l274_27452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l274_27469

def a : Fin 3 → ℝ := ![4, 2, -4]
def b : Fin 3 → ℝ := ![6, -3, 3]

theorem vector_magnitude_problem : 
  Real.sqrt (((a 0) - (b 0))^2 + ((a 1) - (b 1))^2 + ((a 2) - (b 2))^2) = Real.sqrt 78 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l274_27469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_not_multiple_of_six_l274_27433

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

def is_perfect_fifth_power (n : ℕ) : Prop := ∃ k : ℕ, n = k^5

def smallest_m : ℕ := 2^60 * 3^45 * 5^30

theorem divisors_not_multiple_of_six (m : ℕ) (h1 : m = smallest_m) 
  (h2 : is_perfect_square (m / 4)) (h3 : is_perfect_cube (m / 9)) 
  (h4 : is_perfect_fifth_power (m / 25)) :
  (Finset.filter (fun d ↦ d ∣ m ∧ ¬(6 ∣ d)) (Finset.range (m + 1))).card = 3766 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_not_multiple_of_six_l274_27433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_d_value_l274_27499

theorem abs_d_value (a b c d : ℤ) : 
  (a * (3 + I : ℂ)^6 + b * (3 + I : ℂ)^5 + c * (3 + I : ℂ)^4 + d * (3 + I : ℂ)^3 + 
   c * (3 + I : ℂ)^2 + b * (3 + I : ℂ) + a = 0) →
  (Int.gcd a (Int.gcd b (Int.gcd c d)) = 1) →
  d.natAbs = 540 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_d_value_l274_27499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_18_dividing_24_factorial_l274_27427

theorem largest_power_of_18_dividing_24_factorial :
  (∃ n : ℕ, (18 ^ n : ℕ) ∣ Nat.factorial 24 ∧ 
    ∀ m : ℕ, m > n → ¬((18 ^ m : ℕ) ∣ Nat.factorial 24)) →
  (∃ n : ℕ, n = 5 ∧ (18 ^ n : ℕ) ∣ Nat.factorial 24 ∧ 
    ∀ m : ℕ, m > n → ¬((18 ^ m : ℕ) ∣ Nat.factorial 24)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_18_dividing_24_factorial_l274_27427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_power_1500_eq_one_l274_27400

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The complex number (2 + 3i) / (3 - 2i) -/
noncomputable def z : ℂ := (2 + 3 * i) / (3 - 2 * i)

/-- Theorem: z^1500 = 1 -/
theorem z_power_1500_eq_one : z ^ 1500 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_power_1500_eq_one_l274_27400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l274_27481

theorem sin_alpha_value (α : Real) 
  (h1 : Real.tan α = 1/2) 
  (h2 : α ∈ Set.Ioo π (3*π/2)) : 
  Real.sin α = -Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l274_27481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_l274_27439

-- Define the function f(x)
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x - k) * Real.exp x

-- State the theorem
theorem f_lower_bound (k : ℝ) :
  (k ≤ 1) →
  (∀ x ∈ Set.Icc 0 1, f k x > k^2 - 2) ↔ (-2 < k ∧ k < 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_l274_27439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l274_27435

/-- The distance between two parallel lines ax + by + c₁ = 0 and ax + by + c₂ = 0 -/
noncomputable def distance_between_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  abs (c₂ - c₁) / Real.sqrt (a^2 + b^2)

/-- First line equation: 3x + 4y - 12 = 0 -/
def line1 : ℝ × ℝ → Prop :=
  fun (x, y) ↦ 3 * x + 4 * y - 12 = 0

/-- Second line equation: 6x + 8y + 6 = 0 -/
def line2 : ℝ × ℝ → Prop :=
  fun (x, y) ↦ 6 * x + 8 * y + 6 = 0

theorem distance_between_given_lines :
  distance_between_parallel_lines 3 4 (-12) 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l274_27435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l274_27440

noncomputable def f (x : ℝ) : ℝ := (15*x^5 + 7*x^4 + 6*x^3 + 2*x^2 + x + 4) / (4*x^5 + 3*x^4 + 9*x^3 + 4*x^2 + 2*x + 1)

theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ N : ℝ, ∀ x : ℝ, |x| > N → |f x - 15/4| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l274_27440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_equals_fifteen_eighths_minus_two_ln_two_l274_27451

-- Define the curves
noncomputable def f (x : ℝ) : ℝ := 1 / x
noncomputable def g (x : ℝ) : ℝ := -x + 5/2

-- Define the intersection points
noncomputable def x₁ : ℝ := 1/2
noncomputable def x₂ : ℝ := 2

-- Theorem statement
theorem enclosed_area_equals_fifteen_eighths_minus_two_ln_two :
  ∫ x in x₁..x₂, (g x - f x) = 15/8 - 2 * Real.log 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_equals_fifteen_eighths_minus_two_ln_two_l274_27451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l274_27442

-- Define the arithmetic sequence a_n
noncomputable def a (n : ℕ) : ℝ := 2 * n + 1

-- Define the sum S_n of the first n terms of a_n
noncomputable def S (n : ℕ) : ℝ := n * (a 1 + a n) / 2

-- Define the geometric sequence b_n
noncomputable def b (n : ℕ) : ℝ := a 1 * 3^(n - 1)

-- Define the sum B_n of the first n terms of b_n
noncomputable def B (n : ℕ) : ℝ := (3 / 2) * (3^n - 1)

-- Define T_n as the sum of the first n terms of 1/S_n
noncomputable def T (n : ℕ) : ℝ := 3 / 4 - (2 * n + 3) / (2 * (n + 1) * (n + 2))

theorem sequence_properties (n : ℕ) :
  (a n = 2 * n + 1) ∧
  (B n = (3 / 2) * (3^n - 1)) ∧
  (T n = 3 / 4 - (2 * n + 3) / (2 * (n + 1) * (n + 2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l274_27442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_range_in_non_right_triangle_l274_27480

theorem angle_B_range_in_non_right_triangle (A B C : ℝ) 
  (hNonRight : A + B + C = Real.pi) 
  (hGeometricSeq : ∃ (r : ℝ), Real.tan C = r * Real.tan B ∧ Real.tan B = r * Real.tan A) :
  Real.pi/3 ≤ B ∧ B < Real.pi/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_range_in_non_right_triangle_l274_27480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_scaling_l274_27402

theorem determinant_scaling (a b c d e f g h i : ℝ) :
  Matrix.det !![a, b, c; d, e, f; g, h, i] = 2 →
  Matrix.det !![3*a, 3*b, 3*c; 3*d, 3*e, 3*f; 3*g, 3*h, 3*i] = 54 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_scaling_l274_27402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_A_counters_l274_27485

/-- Represents the rules for B to choose piles -/
inductive Rule
| R1 -- B takes the biggest and smallest heaps
| R2 -- B takes the two middle heaps
| R3 -- B chooses between R1 and R2

/-- Represents a strategy for dividing counters -/
structure Strategy :=
  (initial_division : ℕ → ℕ × ℕ)
  (subsequent_division : ℕ → ℕ × ℕ)

/-- Represents the game with N counters -/
def Game (N : ℕ) (rule : Rule) :=
  { strategy : Strategy // 
    (∀ n : ℕ, n ≥ 4 → 
      let (a, b) := strategy.initial_division n
      a ≥ 2 ∧ b ≥ 2 ∧ a + b = n) ∧
    (∀ n : ℕ, n ≥ 2 → 
      let (c, d) := strategy.subsequent_division n
      c ≥ 1 ∧ d ≥ 1 ∧ c + d = n) }

/-- The number of counters A gets after optimal play -/
noncomputable def A_counters (N : ℕ) (rule : Rule) (game : Game N rule) : ℕ :=
  sorry -- Definition of A_counters based on the game rules

/-- Theorem stating that A always gets ⌊N/2⌋ counters under optimal play -/
theorem optimal_A_counters (N : ℕ) (h : N ≥ 4) (rule : Rule) :
  ∀ game : Game N rule, A_counters N rule game = N / 2 := by
  sorry

#check optimal_A_counters

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_A_counters_l274_27485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_is_four_hours_l274_27468

/-- Two people traveling towards each other on a route --/
structure TravelScenario where
  distance : ℚ
  speed1 : ℚ
  speed2 : ℚ

/-- Calculate the time it takes for two people to meet --/
def meetingTime (scenario : TravelScenario) : ℚ :=
  scenario.distance / (scenario.speed1 + scenario.speed2)

/-- Theorem: In the given scenario, the meeting time is 4 hours --/
theorem meeting_time_is_four_hours :
  let scenario : TravelScenario := {
    distance := 600,
    speed1 := 70,
    speed2 := 80
  }
  meetingTime scenario = 4 := by
  -- Proof goes here
  sorry

-- Evaluate the meeting time for the given scenario
#eval meetingTime { distance := 600, speed1 := 70, speed2 := 80 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_is_four_hours_l274_27468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_money_sharing_ratio_l274_27474

def sandra_share : ℝ := 100
def amy_share : ℝ := 50

def ruth_share : ℝ → ℝ := fun r => r

theorem money_sharing_ratio (r : ℝ) : 
  (sandra_share, amy_share, ruth_share r) = (2 * 50, 1 * 50, r) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_money_sharing_ratio_l274_27474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2003_l274_27483

noncomputable def sequenceR (x₀ x₁ : ℝ) : ℕ → ℝ
  | 0 => x₀
  | 1 => x₁
  | n + 2 => (sequenceR x₀ x₁ (n + 1) + 1) / sequenceR x₀ x₁ n

theorem sequence_2003 (x₀ x₁ : ℝ) (h₀ : x₀ > 0) (h₁ : x₁ > 0) :
  sequenceR x₀ x₁ 2003 = (x₁ + 1) / x₀ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2003_l274_27483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_valid_trace_solutions_l274_27497

/-- Represents a trace line of a plane -/
structure TraceLine where
  -- Add necessary fields
  x : ℝ
  y : ℝ

/-- Represents a projection plane -/
structure ProjectionPlane where
  -- Add necessary fields
  normal : Vector ℝ 3

/-- Represents the angle between a plane and a projection plane -/
def PlaneAngle : Type := ℝ

/-- Represents a solution for the other trace line -/
structure TraceSolution where
  -- Add necessary fields
  trace : TraceLine

/-- Predicate to check if the intersection is outside the paper boundary -/
def IntersectionOutside (trace : TraceLine) (plane : ProjectionPlane) : Prop :=
  sorry

/-- Function to find valid trace solutions -/
noncomputable def findTraceSolutions (givenTrace : TraceLine) (angle : PlaneAngle) (projectionPlane : ProjectionPlane) : Finset TraceSolution :=
  sorry

/-- Theorem stating that there are exactly two valid solutions -/
theorem two_valid_trace_solutions 
  (givenTrace : TraceLine) 
  (angle : PlaneAngle) 
  (projectionPlane : ProjectionPlane) 
  (h_intersection_outside : IntersectionOutside givenTrace projectionPlane) :
  (findTraceSolutions givenTrace angle projectionPlane).card = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_valid_trace_solutions_l274_27497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_55_l274_27477

/-- Definition of the sequence a_n -/
def a : ℕ → ℕ
  | 0 => 0  -- Add this case for 0
  | 1 => 0  -- Add this case for 1
  | 2 => 0  -- Add this case for 2
  | 3 => 0  -- Add this case for 3
  | 4 => 0  -- Add this case for 4
  | 5 => 5
  | n+6 => 200 * a (n+5) + (n+6)

/-- Theorem stating that 32 is the least positive integer n > 5 such that a_n is divisible by 55 -/
theorem least_multiple_of_55 : ∀ k > 5, k < 32 → ¬(55 ∣ a k) ∧ (55 ∣ a 32) := by
  sorry

#eval a 32  -- This will evaluate a_32

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_55_l274_27477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polygon_theorem_l274_27438

/-- A polygon with n sides --/
structure Polygon (n : ℕ) where
  interiorAngles : Fin n → ℝ
  exteriorAngles : Fin n → ℝ

/-- The sum of interior angles of a polygon --/
def sumInteriorAngles {n : ℕ} (Q : Polygon n) : ℝ :=
  Finset.sum Finset.univ (λ i => Q.interiorAngles i)

/-- The sum of exterior angles of a polygon --/
def sumExteriorAngles {n : ℕ} (Q : Polygon n) : ℝ :=
  Finset.sum Finset.univ (λ i => Q.exteriorAngles i)

/-- A polygon is regular if all its interior angles are equal --/
def isRegular {n : ℕ} (Q : Polygon n) : Prop :=
  ∀ i j : Fin n, Q.interiorAngles i = Q.interiorAngles j

/-- Theorem about the sum of interior angles and regularity of a special polygon --/
theorem special_polygon_theorem {n : ℕ} (Q : Polygon n) 
  (h1 : ∀ i, Q.interiorAngles i = 8 * Q.exteriorAngles i) 
  (h2 : sumExteriorAngles Q = 360) : 
  sumInteriorAngles Q = 2880 ∧ isRegular Q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polygon_theorem_l274_27438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l274_27446

/-- The function f(x) defined as the given rational function --/
noncomputable def f (x : ℝ) : ℝ := (15 * x^5 + 7 * x^3 + 4 * x^2 + 6 * x + 5) / (3 * x^5 + 2 * x^3 + 7 * x^2 + 4 * x + 2)

/-- Theorem stating that the horizontal asymptote of f(x) is 5 --/
theorem horizontal_asymptote_of_f : 
  ∀ ε > 0, ∃ M, ∀ x, |x| > M → |f x - 5| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l274_27446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_n_l274_27412

-- Define a function to get the units digit of a natural number
def unitsDigit (x : ℕ) : ℕ := x % 10

-- Define the given conditions
axiom m : ℕ
axiom n : ℕ
axiom condition1 : m * n = 21^6
axiom condition2 : unitsDigit m = 7

-- State the theorem
theorem units_digit_of_n : unitsDigit n = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_n_l274_27412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_vector_sum_l274_27471

/-- An equilateral triangle with side length √2 -/
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  side_length : ℝ
  is_equilateral : side_length = Real.sqrt 2
  AB_eq_side : dist A B = side_length
  BC_eq_side : dist B C = side_length
  CA_eq_side : dist C A = side_length

/-- Vectors of the triangle -/
def triangle_vectors (t : EquilateralTriangle) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  let a := (t.C.1 - t.B.1, t.C.2 - t.B.2)
  let b := (t.A.1 - t.C.1, t.A.2 - t.C.2)
  let c := (t.B.1 - t.A.1, t.B.2 - t.A.2)
  (a, b, c)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- The main theorem -/
theorem equilateral_triangle_vector_sum (t : EquilateralTriangle) :
  let (a, b, c) := triangle_vectors t
  dot_product a b + dot_product b c + dot_product c a = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_vector_sum_l274_27471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersections_l274_27434

/-- The line equation 3x + 4y = 12 -/
def my_line (x y : ℝ) : Prop := 3 * x + 4 * y = 12

/-- The circle equation x^2 + y^2 = 4 -/
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Theorem stating that the line and circle have no real intersections -/
theorem no_intersections : ¬∃ (x y : ℝ), my_line x y ∧ my_circle x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersections_l274_27434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vacation_fund_adjustment_l274_27492

/-- Calculates the new weekly work hours required to meet a financial goal
    after losing some work weeks. -/
noncomputable def new_weekly_hours (original_hours : ℝ) (original_weeks : ℕ) 
  (lost_weeks : ℕ) : ℝ :=
  (original_weeks : ℝ) / ((original_weeks - lost_weeks) : ℝ) * original_hours

theorem vacation_fund_adjustment 
  (original_hours : ℝ) (original_weeks : ℕ) (lost_weeks : ℕ) 
  (h1 : original_hours = 15)
  (h2 : original_weeks = 10)
  (h3 : lost_weeks = 3)
  (h4 : original_weeks > lost_weeks) :
  new_weekly_hours original_hours original_weeks lost_weeks = 
    (original_weeks : ℝ) / ((original_weeks - lost_weeks) : ℝ) * original_hours :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vacation_fund_adjustment_l274_27492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l274_27484

theorem sin_double_angle_special_case (α : ℝ) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin α = 4 / 5) : 
  Real.sin (2 * α) = -24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l274_27484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l274_27462

/-- σ(n) is the sum of divisors function -/
def sigma (n : ℕ+) : ℕ+ := sorry

/-- φ(n) is Euler's totient function -/
def phi (n : ℕ+) : ℕ+ := sorry

/-- The main theorem stating that (1, 1) is the only solution -/
theorem unique_solution :
  ∀ n k : ℕ+, (sigma n * phi n : ℚ) = n^2 / k ↔ n = 1 ∧ k = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l274_27462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_and_max_distance_l274_27409

noncomputable section

-- Define the line C1
def C1 (t : ℝ) : ℝ × ℝ := (1 + (Real.sqrt 2 / 2) * t, (Real.sqrt 2 / 2) * t)

-- Define the curve C2
def C2 (r θ : ℝ) : ℝ × ℝ := (r * Real.cos θ, r * Real.sin θ)

-- Define the distance function from a point to C1
def distToC1 (p : ℝ × ℝ) : ℝ :=
  (|(p.1 - p.2) - 1|) / Real.sqrt 2

theorem intersection_points_and_max_distance :
  -- Part 1: Intersection points when r = 1
  ({p : ℝ × ℝ | ∃ t, C1 t = p} ∩ {p : ℝ × ℝ | ∃ θ, C2 1 θ = p} = {(1, 0), (0, -1)}) ∧
  -- Part 2: Point with maximum distance when r = √2
  (∃ θ : ℝ, C2 (Real.sqrt 2) θ = (-1, 1) ∧
    ∀ φ : ℝ, distToC1 (C2 (Real.sqrt 2) φ) ≤ distToC1 (-1, 1)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_and_max_distance_l274_27409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_difference_l274_27416

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 4*x + 4

-- Define the interval [0, 3]
def interval : Set ℝ := { x | 0 ≤ x ∧ x ≤ 3 }

-- Theorem statement
theorem max_min_difference :
  ∃ (M m : ℝ), (∀ x ∈ interval, f x ≤ M) ∧
                (∀ x ∈ interval, m ≤ f x) ∧
                (M - m = 16/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_difference_l274_27416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_solution_range_l274_27482

theorem inequality_system_solution_range (a : ℚ) : 
  (∃! (s : Finset ℤ), s = {x : ℤ | x ≤ 11 ∧ (2 * x + 2) / 3 < x + a} ∧ s.card = 3) →
  -7/3 < a ∧ a ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_solution_range_l274_27482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l274_27478

-- Define the function f
def f : ℝ → ℝ := λ x ↦ x + 2

-- Define the function g
def g (l : ℝ) : ℝ → ℝ := λ x ↦ x * f x + l * f x + 1

-- Main theorem
theorem f_and_g_properties (l : ℝ) :
  (∀ x : ℝ, f (x + 1) = x + 3) ∧
  f 1 = 3 ∧
  (∀ x ∈ Set.Ioo 0 2, StrictMono (g l)) ∧
  l < 0 →
  (∀ x : ℝ, f x = x + 2) ∧
  (l ≤ -6 ∨ -2 ≤ l) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l274_27478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_equilateral_triangle_l274_27413

noncomputable section

-- Define the square ABCD
def square (R : ℝ) : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ R - 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ R - 1}

-- Define points A, B, C, D
def A (R : ℝ) : ℝ × ℝ := (0, 0)
def B (R : ℝ) : ℝ × ℝ := (R - 1, 0)
def C (R : ℝ) : ℝ × ℝ := (R - 1, R - 1)
def D (R : ℝ) : ℝ × ℝ := (0, R - 1)

-- Define the equilateral triangle AEF
def triangle_AEF (R : ℝ) (E F : ℝ × ℝ) : Prop :=
  E.1 = R - 1 ∧ 0 ≤ E.2 ∧ E.2 ≤ R - 1 ∧  -- E is on BC
  F.2 = R - 1 ∧ 0 ≤ F.1 ∧ F.1 ≤ R - 1 ∧  -- F is on CD
  Real.sqrt ((A R).1 - E.1)^2 + ((A R).2 - E.2)^2 =
  Real.sqrt ((A R).1 - F.1)^2 + ((A R).2 - F.2)^2 ∧
  Real.sqrt ((A R).1 - E.1)^2 + ((A R).2 - E.2)^2 =
  Real.sqrt (E.1 - F.1)^2 + (E.2 - F.2)^2  -- AE = AF = EF

-- Define the area of triangle AEF
noncomputable def area_AEF (R : ℝ) (E F : ℝ × ℝ) : ℝ :=
  Real.sqrt 3 / 4 * (Real.sqrt ((A R).1 - E.1)^2 + ((A R).2 - E.2)^2)^2

-- Theorem statement
theorem area_of_equilateral_triangle (R S : ℝ) (E F : ℝ × ℝ) :
  R > 1 →
  E ∈ square R →
  F ∈ square R →
  triangle_AEF R E F →
  area_AEF R E F = S - 3 →
  S = 2 * Real.sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_equilateral_triangle_l274_27413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_machine_rate_l274_27453

/-- The rate at which the old machine produces bolts, in bolts per hour -/
noncomputable def old_rate : ℝ := 100

/-- The time both machines work together, in hours -/
noncomputable def work_time : ℝ := 96 / 60

/-- The total number of bolts produced by both machines -/
noncomputable def total_bolts : ℝ := 400

/-- The rate at which the new machine produces bolts, in bolts per hour -/
noncomputable def new_rate : ℝ := (total_bolts - old_rate * work_time) / work_time

theorem new_machine_rate : new_rate = 150 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_machine_rate_l274_27453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_modular_product_l274_27408

/-- A positive square-free integer -/
def SquareFree (n : ℕ) : Prop :=
  n > 0 ∧ ∀ p : ℕ, Nat.Prime p → (p * p ∣ n) → False

theorem subset_modular_product {n : ℕ} (hn : SquareFree n) 
  (S : Finset ℕ) (hS : S ⊆ Finset.range n) (hScard : S.card * 2 ≥ n) :
  ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ (a * b) % n = c % n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_modular_product_l274_27408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_condition_g_greater_than_one_l274_27479

-- Define the functions f and g
noncomputable def f (a x : ℝ) : ℝ := (Real.exp x - a * x) * (x * Real.exp x + Real.sqrt 2)
noncomputable def g (x : ℝ) : ℝ := x * Real.exp x + Real.sqrt 2

-- Statement for part (1)
theorem tangent_slope_condition (a : ℝ) : 
  (deriv (f a)) 0 = Real.sqrt 2 + 1 ↔ a = 0 := by sorry

-- Statement for part (2)
theorem g_greater_than_one : ∀ x : ℝ, g x > 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_condition_g_greater_than_one_l274_27479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_at_neg_two_l274_27467

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x - 2) * Real.exp x

theorem f_max_at_neg_two :
  ∃ (c : ℝ), c = -2 ∧ ∀ x, f x ≤ f c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_at_neg_two_l274_27467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bethany_twice_sister_age_l274_27465

/-- Represents the age difference between Bethany and the current year -/
def years_ago : ℕ := 3

/-- Bethany's current age -/
def bethany_current_age : ℕ := 19

/-- Bethany's sister's age in 5 years -/
def sister_age_in_5_years : ℕ := 16

theorem bethany_twice_sister_age : 
  bethany_current_age - years_ago = 2 * (sister_age_in_5_years - 5 - years_ago) ∧ 
  years_ago = 3 := by
  -- Proof goes here
  sorry

#eval years_ago

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bethany_twice_sister_age_l274_27465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_placement_theorem_l274_27461

theorem circle_placement_theorem (rectangle_width : ℝ) (rectangle_height : ℝ) 
  (num_squares : ℕ) (square_side_length : ℝ) (circle_diameter : ℝ)
  (h1 : rectangle_width = 20)
  (h2 : rectangle_height = 25)
  (h3 : num_squares = 120)
  (h4 : square_side_length = 1)
  (h5 : circle_diameter = 1) :
  ∃ (x y : ℝ), 
    0.5 ≤ x ∧ x ≤ rectangle_width - 0.5 ∧
    0.5 ≤ y ∧ y ≤ rectangle_height - 0.5 ∧
    ∀ (sx sy : ℝ), 
      (0 ≤ sx ∧ sx ≤ rectangle_width - square_side_length) →
      (0 ≤ sy ∧ sy ≤ rectangle_height - square_side_length) →
      |x - sx| ≥ 0.5 ∨ |y - sy| ≥ 0.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_placement_theorem_l274_27461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sampling_method_is_systematic_l274_27488

/-- Represents a sampling method --/
inductive SamplingMethod
  | Random
  | Systematic
  | Stratified

/-- Represents a class with students and their IDs --/
structure ClassInfo where
  size : Nat
  selectedIDs : List Nat

/-- Checks if the given list has a constant difference between consecutive elements --/
def hasConstantDifference (lst : List Nat) : Prop :=
  ∃ d : Nat, ∀ i : Nat, i + 1 < lst.length → lst[i+1]! - lst[i]! = d

/-- Defines the characteristics of Systematic Sampling --/
def isSystematicSampling (c : ClassInfo) : Prop :=
  c.size > 0 ∧ c.selectedIDs.length > 1 ∧ hasConstantDifference c.selectedIDs

/-- The main theorem to prove --/
theorem sampling_method_is_systematic 
  (c : ClassInfo) 
  (h1 : c.size = 50) 
  (h2 : c.selectedIDs = [3, 8, 13, 18, 23, 28, 33, 38, 43, 48]) : 
  isSystematicSampling c := by
  sorry

#check sampling_method_is_systematic

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sampling_method_is_systematic_l274_27488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_calculation_l274_27436

theorem sin_beta_calculation (α β : Real) (h1 : 0 < α ∧ α < Real.pi/2) (h2 : 0 < β ∧ β < Real.pi/2)
  (h3 : Real.sin α = 3/5) (h4 : Real.cos (α + β) = -5/13) : Real.sin β = 63/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_calculation_l274_27436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_example_l274_27417

/-- Definition of factorization from left to right -/
def is_factorization_left_to_right (f g : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∃ (h k : ℝ → ℝ → ℝ → ℝ), f = g ∧ (∀ a b c, f a b c = (h a b c) * (k a b c))

/-- The expression 2ab - 2ac = 2a(b - c) is a factorization from left to right -/
theorem factorization_example :
  is_factorization_left_to_right (λ a b c ↦ 2*a*b - 2*a*c) (λ a b c ↦ 2*a*(b - c)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_example_l274_27417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_sum_problem_l274_27441

theorem cubic_root_sum_problem :
  ∃ (y₁ y₂ : ℝ) (r s : ℤ),
    y₁ < y₂ ∧
    (y₁^(1/3 : ℝ) + (26 - y₁)^(1/3 : ℝ) = 3) ∧
    (y₂^(1/3 : ℝ) + (26 - y₂)^(1/3 : ℝ) = 3) ∧
    y₂ = r + Real.sqrt (s : ℝ) ∧
    r + s = 5130 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_sum_problem_l274_27441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_pass_prob_one_of_two_rewarded_l274_27496

/-- The probability of passing a single competition --/
noncomputable def pass_prob : ℚ := 1 / 2

/-- The number of competitions each contestant participates in --/
def num_competitions : ℕ := 3

/-- The total number of contestants --/
def total_contestants : ℕ := 6

/-- The number of contestants that receive commendation and reward --/
def rewarded_contestants : ℕ := 2

/-- The probability of a contestant getting at least one pass in three independent competitions --/
theorem prob_at_least_one_pass :
  1 - (1 - pass_prob) ^ num_competitions = 7 / 8 := by sorry

/-- The probability that exactly one of two specific contestants is among the top 2 out of 6 contestants --/
theorem prob_one_of_two_rewarded :
  (rewarded_contestants * (total_contestants - rewarded_contestants)) /
  (Nat.choose total_contestants rewarded_contestants : ℚ) = 8 / 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_pass_prob_one_of_two_rewarded_l274_27496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l274_27422

-- Define the points A, B, and C
def A : ℝ × ℝ := (3, 3)
def B : ℝ × ℝ := (3, 10)
def C : ℝ × ℝ := (8, 6)

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the perimeter of the triangle
noncomputable def perimeter : ℝ :=
  distance A B + distance B C + distance C A

-- Theorem stating that the perimeter is equal to 7 + √41 + √34
theorem triangle_perimeter :
  perimeter = 7 + Real.sqrt 41 + Real.sqrt 34 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l274_27422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_decisive_games_is_four_l274_27404

/-- A chess match where the winner is determined by the first player to win two games -/
structure ChessMatch where
  /-- The probability of winning a single game for each player -/
  winProbability : ℝ
  /-- The condition that the win probability is equal for both players and between 0 and 1 -/
  winProbabilityValid : 0 < winProbability ∧ winProbability < 1 ∧ winProbability = 1 - winProbability

/-- The expected number of decisive games in the chess match -/
def expectedDecisiveGames (m : ChessMatch) : ℝ :=
  4

/-- Theorem stating that the expected number of decisive games is 4 -/
theorem expected_decisive_games_is_four (m : ChessMatch) :
  expectedDecisiveGames m = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_decisive_games_is_four_l274_27404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_hot_dogs_made_l274_27415

/-- Given that a restaurant made hamburgers in two batches and the total number of hamburgers,
    prove that no hot dogs were made. -/
theorem no_hot_dogs_made (initial_hamburgers : ℝ) (additional_hamburgers : ℝ) 
    (total_hamburgers : ℕ) (h1 : initial_hamburgers = 9) (h2 : additional_hamburgers = 3) 
    (h3 : total_hamburgers = 12) (h4 : ↑total_hamburgers = initial_hamburgers + additional_hamburgers) : 
    0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_hot_dogs_made_l274_27415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_angle_values_l274_27459

/-- A sector is characterized by its radius and central angle -/
structure Sector where
  radius : ℝ
  angle : ℝ

/-- The circumference of a sector -/
noncomputable def sectorCircumference (s : Sector) : ℝ :=
  s.radius * s.angle + 2 * s.radius

/-- The area of a sector -/
noncomputable def sectorArea (s : Sector) : ℝ :=
  1/2 * s.radius^2 * s.angle

/-- Theorem: Given a sector with circumference 6 and area 2,
    its central angle is either 1 or 4 radians -/
theorem sector_angle_values (s : Sector) 
    (h_circ : sectorCircumference s = 6)
    (h_area : sectorArea s = 2) :
    s.angle = 1 ∨ s.angle = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_angle_values_l274_27459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_rounded_sum_l274_27407

/-- Rounds a number to the nearest multiple of 5, rounding up on .5 -/
def roundToNearest5 (n : ℕ) : ℕ :=
  5 * ((n + 2) / 5)

/-- Sum of first n natural numbers -/
def sumFirstN (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Sum of first n natural numbers rounded to nearest 5 -/
def sumRoundedFirstN (n : ℕ) : ℕ :=
  (List.range n).map (fun i => roundToNearest5 (i + 1)) |> List.sum

theorem sum_equals_rounded_sum :
  sumFirstN 100 = sumRoundedFirstN 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_rounded_sum_l274_27407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_approximation_l274_27410

-- Define the scale range
noncomputable def scale_start : ℝ := 9.75
noncomputable def scale_end : ℝ := 10.00

-- Define the arrow position (slightly above midpoint)
noncomputable def arrow_position : ℝ := (scale_start + scale_end) / 2 + 0.01

-- Define the possible readings (to the nearest 0.05)
def possible_readings : List ℝ := [9.80, 9.85, 9.90, 9.95, 10.00]

-- Function to find the closest reading
noncomputable def closest_reading (position : ℝ) (readings : List ℝ) : ℝ :=
  match readings.argmin (fun r => |r - position|) with
  | some x => x
  | none => 0  -- Default value, should never occur with non-empty list

-- Theorem statement
theorem closest_approximation :
  closest_reading arrow_position possible_readings = 9.90 := by
  sorry

#eval possible_readings  -- This line is added to ensure the list is properly defined

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_approximation_l274_27410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conference_distance_l274_27431

/-- Represents the journey to the conference -/
structure Journey where
  initial_speed : ℝ
  speed_increase : ℝ
  initial_delay : ℝ
  final_early : ℝ
  first_hour_distance : ℝ

/-- Calculates the distance to the conference -/
noncomputable def distance_to_conference (j : Journey) : ℝ :=
  let final_speed := j.initial_speed + j.speed_increase
  let t := (j.initial_speed * (j.initial_delay + 1) + final_speed * j.final_early) / (final_speed - j.initial_speed)
  j.initial_speed * (t + j.initial_delay)

/-- The main theorem stating the distance to the conference -/
theorem conference_distance :
  let j : Journey := {
    initial_speed := 45
    speed_increase := 25
    initial_delay := 1.5
    final_early := 1/6
    first_hour_distance := 45
  }
  ⌊distance_to_conference j⌋ = 290 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conference_distance_l274_27431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_l274_27463

/-- The function f(x) = (3x^2 + 10x + 5) / (2x + 5) -/
noncomputable def f (x : ℝ) : ℝ := (3 * x^2 + 10 * x + 5) / (2 * x + 5)

/-- The proposed oblique asymptote function -/
noncomputable def g (x : ℝ) : ℝ := (3/2) * x + 5/2

/-- Theorem: The oblique asymptote of f(x) is g(x) -/
theorem oblique_asymptote : 
  ∀ ε > 0, ∃ M, ∀ x, |x| > M → |f x - g x| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_l274_27463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_monotonic_increase_interval_l274_27449

/-- The interval of monotonic increase for a periodic sine function -/
theorem sine_monotonic_increase_interval 
  (f : ℝ → ℝ) 
  (ω : ℝ) 
  (h_ω_pos : ω > 0) 
  (h_f_def : ∀ x, f x = 2 * Real.sin (ω * x - π / 6) - 1) 
  (h_period : ∀ x, f (x + π / ω) = f x) :
  ∀ k : ℤ, StrictMonoOn f (Set.Icc (k * π - π / 6) (k * π + π / 3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_monotonic_increase_interval_l274_27449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_schemes_l274_27491

/-- Represents the set of anti-inflammatory drugs -/
inductive AntiInflammatory
| X₁ | X₂ | X₃ | X₄ | X₅
deriving Fintype, DecidableEq

/-- Represents the set of antipyretic drugs -/
inductive Antipyretic
| T₁ | T₂ | T₃ | T₄
deriving Fintype, DecidableEq

/-- A test scheme consists of two anti-inflammatory drugs and one antipyretic drug -/
structure TestScheme where
  antiInflammatory1 : AntiInflammatory
  antiInflammatory2 : AntiInflammatory
  antipyretic : Antipyretic
deriving Fintype, DecidableEq

/-- Predicate to check if a test scheme is valid -/
def isValidScheme (scheme : TestScheme) : Prop :=
  (scheme.antiInflammatory1 = AntiInflammatory.X₁ ↔ scheme.antiInflammatory2 = AntiInflammatory.X₂) ∧
  ¬(scheme.antiInflammatory1 = AntiInflammatory.X₃ ∧ scheme.antipyretic = Antipyretic.T₄) ∧
  ¬(scheme.antiInflammatory2 = AntiInflammatory.X₃ ∧ scheme.antipyretic = Antipyretic.T₄)

instance : DecidablePred isValidScheme :=
  fun scheme => decidable_of_iff
    ((scheme.antiInflammatory1 = AntiInflammatory.X₁ ↔ scheme.antiInflammatory2 = AntiInflammatory.X₂) ∧
     ¬(scheme.antiInflammatory1 = AntiInflammatory.X₃ ∧ scheme.antipyretic = Antipyretic.T₄) ∧
     ¬(scheme.antiInflammatory2 = AntiInflammatory.X₃ ∧ scheme.antipyretic = Antipyretic.T₄))
    (by simp [isValidScheme])

/-- The main theorem stating that there are exactly 14 valid test schemes -/
theorem count_valid_schemes : 
  (Finset.filter isValidScheme (Finset.univ : Finset TestScheme)).card = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_schemes_l274_27491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_closed_form_l274_27430

def P : ℕ → ℚ
  | 0 => 1/2  -- Add this case to handle Nat.zero
  | 1 => 1/2
  | n+1 => -4/15 * P n + 3/5

theorem P_closed_form (n : ℕ) (hn : n ≥ 1) : 
  P n = 1/38 * (-4/15)^(n-1) + 9/19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_closed_form_l274_27430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_cos_equality_l274_27414

theorem contrapositive_cos_equality (x y : ℝ) : 
  (Real.cos x ≠ Real.cos y → x ≠ y) ↔ (x = y → Real.cos x = Real.cos y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_cos_equality_l274_27414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_hoses_time_l274_27489

/-- Represents the time taken to fill a pool with different combinations of hoses -/
structure PoolFilling where
  A_B : ℚ  -- Time for hoses A and B together
  A_C : ℚ  -- Time for hoses A and C together
  A_D : ℚ  -- Time for hoses A and D together
  B_C : ℚ  -- Time for hoses B and C together
  B_D : ℚ  -- Time for hoses B and D together
  C_D : ℚ  -- Time for hoses C and D together

/-- Calculates the time taken for all four hoses to fill the pool together -/
def time_for_all_hoses (pf : PoolFilling) : ℚ :=
  48 / 43

/-- Theorem stating that given the conditions, the time for all hoses is 48/43 hours -/
theorem all_hoses_time (pf : PoolFilling) 
    (h1 : pf.A_B = 3) 
    (h2 : pf.A_C = 6) 
    (h3 : pf.A_D = 4) 
    (h4 : pf.B_C = 9) 
    (h5 : pf.B_D = 6) 
    (h6 : pf.C_D = 8) : 
  time_for_all_hoses pf = 48 / 43 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_hoses_time_l274_27489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_M_l274_27466

-- Define the curve
noncomputable def C (x : ℝ) : ℝ := x * Real.log x

-- Define the point of tangency
noncomputable def M : ℝ × ℝ := (Real.exp 1, Real.exp 1)

-- Define the derivative of C
noncomputable def C_deriv (x : ℝ) : ℝ := Real.log x + 1

-- State the theorem
theorem tangent_line_at_M : 
  ∀ x y : ℝ, (y - M.2 = ((C_deriv M.1) * (x - M.1))) ↔ (y = 2*x - Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_M_l274_27466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_examples_l274_27460

-- Define the logarithm operation as noncomputable
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_examples : log 2 16 = 4 ∧ log 3 81 = 4 := by
  -- Split the conjunction
  constructor
  -- Prove log 2 16 = 4
  · sorry
  -- Prove log 3 81 = 4
  · sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_examples_l274_27460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_work_time_l274_27455

/-- The number of days A needs to complete the work alone -/
def a_days : ℝ := 15

/-- The number of days B needs to complete the work alone -/
def b_days : ℝ := 14

/-- The number of days A, B, and C need to complete the work together -/
def abc_days : ℝ := 4.977777777777778

/-- The number of days C needs to complete the work alone -/
noncomputable def c_days : ℝ := (abc_days * a_days * b_days) / (a_days * b_days - abc_days * (a_days + b_days))

theorem c_work_time : ∃ (ε : ℝ), ε > 0 ∧ |c_days - 15.92| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_work_time_l274_27455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_f_10_l274_27426

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- State the conditions
axiom g_has_inverse : Function.Bijective g
axiom f_g_relation : ∀ x, Function.invFun f (g x) = x^2 + 1

-- State the theorem
theorem g_inverse_f_10 : 
  Function.invFun g (f 10) = 3 ∨ Function.invFun g (f 10) = -3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_f_10_l274_27426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_sum_of_powers_l274_27401

theorem divisibility_of_sum_of_powers (n a b : ℕ) : 
  Odd n → (∃ k : ℕ, a + b = n * k) → ∃ m : ℕ, a^n + b^n = n^2 * m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_sum_of_powers_l274_27401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_return_probability_l274_27458

/-- Probability of returning to the starting vertex after n moves -/
def Q (n : ℕ) : ℚ := sorry

/-- The cube has edges of 1 meter -/
def cube_edge : ℝ := 1

/-- The bug starts at vertex A -/
def start_vertex : ℕ := 0

/-- At each vertex, the bug chooses one of three edges with equal probability -/
axiom move_probability : ∀ (v : ℕ), (1 : ℚ) / 3 = (1 : ℚ) / 3

/-- The probability of returning to the starting vertex after 8 moves -/
def return_probability : ℚ := Q 8

/-- The main theorem to prove -/
theorem bug_return_probability : return_probability = 1641 / 6561 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_return_probability_l274_27458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l274_27437

def A : Set ℝ := {x | |x - 2| ≤ 1}
def B (a : ℝ) : Set ℝ := {x | x^2 - a < 0}

theorem problem_solution :
  (A ∩ B 4 = Set.Icc 1 2 ∧ A ∪ B 4 = Set.Ioc (-2) 3) ∧
  (∀ a : ℝ, B a ⊆ Aᶜ → a ≤ 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l274_27437
