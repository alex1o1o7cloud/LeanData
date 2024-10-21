import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_f_nonpositive_range_l688_68897

noncomputable section

variable (a : ℝ)

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x - a * Real.exp (x - 1)

-- Theorem for monotonicity of f(x)
theorem f_monotonicity (a : ℝ) :
  (a ≤ 0 → Monotone (f a)) ∧
  (a > 0 → (∀ x y, x < y → x < 1 - Real.log a → f a x < f a y) ∧
           (∀ x y, x < y → x > 1 - Real.log a → f a x > f a y)) := by sorry

-- Theorem for the range of a where f(x) ≤ 0 for all x
theorem f_nonpositive_range (a : ℝ) :
  (∀ x, f a x ≤ 0) ↔ a ∈ Set.Ici 1 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_f_nonpositive_range_l688_68897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_difference_l688_68883

/-- A linear function g satisfying g(d+1) - g(d) = 5 for all real d -/
noncomputable def g : ℝ → ℝ := sorry

/-- g is a linear function -/
axiom g_linear : IsLinearMap ℝ g

/-- g satisfies g(d+1) - g(d) = 5 for all real d -/
axiom g_property (d : ℝ) : g (d + 1) - g d = 5

/-- Theorem: g(2) - g(7) = -25 -/
theorem g_difference : g 2 - g 7 = -25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_difference_l688_68883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l688_68863

theorem trig_identity (α : ℝ) : 
  4 * Real.cos (α - π/2) * (Real.sin (π/2 + α))^3 - 
  4 * Real.sin (5*π/2 - α) * (Real.cos (3*π/2 + α))^3 = 
  Real.sin (4*α) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l688_68863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_number_l688_68828

theorem find_number (A B : ℕ) (h1 : Nat.lcm A B = 2310) (h2 : Nat.gcd A B = 30) (h3 : B = 180) :
  A = 385 := by
  sorry

#check find_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_number_l688_68828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_two_equal_l688_68825

theorem at_least_two_equal (a b c : ℕ+) 
  (k l m : ℕ)
  (hk : k = (b : ℕ) ^ (c : ℕ) + a)
  (hl : l = (a : ℕ) ^ (b : ℕ) + c)
  (hm : m = (c : ℕ) ^ (a : ℕ) + b)
  (pk : Nat.Prime k)
  (pl : Nat.Prime l)
  (pm : Nat.Prime m) :
  (k = l) ∨ (k = m) ∨ (l = m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_two_equal_l688_68825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_construction_condition_l688_68859

open Real

/-- Theorem: A right triangle can be constructed given the radii of the inscribed
    and exscribed (in the right angle) circles if and only if R ≥ r(3 + 2√2),
    where R is the radius of the exscribed circle and r is the radius of the inscribed circle. -/
theorem right_triangle_construction_condition (r R : ℝ) (hr : r > 0) (hR : R > 0) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2 ∧
  r = (a + b - c) / 2 ∧ R = (a + b + c) / 2 ↔ R ≥ r * (3 + 2 * sqrt 2) := by
  sorry

/-- Lemma: The distance between the centers of the inscribed and exscribed circles
    is equal to r + R, where r is the radius of the inscribed circle and R is the
    radius of the exscribed circle. -/
lemma distance_between_centers (r R : ℝ) (hr : r > 0) (hR : R > 0) :
  ∃ (d : ℝ), d = r + R ∧ d > 0 := by
  sorry

/-- Lemma: The radius of the exscribed circle is always greater than
    the radius of the inscribed circle. -/
lemma excircle_radius_greater (r R : ℝ) (hr : r > 0) (hR : R > 0) :
  R > r := by
  sorry

/-- Lemma: The hypotenuse of the right triangle is a common tangent
    to both the inscribed and exscribed circles. -/
lemma hypotenuse_common_tangent (r R : ℝ) (hr : r > 0) (hR : R > 0) :
  ∃ (c : ℝ), c > 0 ∧ c = R - r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_construction_condition_l688_68859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_output_per_capita_l688_68834

/-- Oil output per capita calculation for different regions --/
theorem oil_output_per_capita 
  (west_population : ℝ) (west_output : ℝ)
  (non_west_population : ℝ) (non_west_output : ℝ)
  (russia_population : ℝ) (russia_partial_output : ℝ) 
  (russia_partial_percentage : ℝ) :
  west_population = 1 →
  west_output = 55.084 →
  non_west_population = 6.9 →
  non_west_output = 1480.689 →
  russia_population = 147 →
  russia_partial_output = 13737.1 →
  russia_partial_percentage = 9 →
  ∃ (west_per_capita non_west_per_capita russia_per_capita : ℝ),
    (abs (west_per_capita - 55.084) < 0.01) ∧ 
    (abs (non_west_per_capita - 214.59) < 0.01) ∧ 
    (abs (russia_per_capita - 1038.33) < 0.01) ∧
    west_per_capita = west_output / west_population ∧
    non_west_per_capita = non_west_output / non_west_population ∧
    russia_per_capita = (russia_partial_output * 100 / russia_partial_percentage) / russia_population :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_output_per_capita_l688_68834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_uniqueness_l688_68854

theorem intersection_uniqueness (a : ℝ) : 
  let M : Set ℝ := {1, 2, a^2 - 3*a - 1}
  let N : Set ℝ := {-1, a, 3}
  (M ∩ N = {3}) → a = 4 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_uniqueness_l688_68854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_semicircle_theorem_l688_68868

/-- Represents a trapezoid with a semicircle inscribed on its base -/
structure TrapezoidWithSemicircle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  O : ℝ × ℝ
  r : ℝ

/-- The semicircle's diameter is on AB and tangent to BC, CD, and DA -/
def is_valid_configuration (t : TrapezoidWithSemicircle) : Prop :=
  -- Add appropriate geometric conditions here
  True

/-- Length of a line segment between two points -/
noncomputable def length (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Main theorem -/
theorem trapezoid_semicircle_theorem (t : TrapezoidWithSemicircle) 
  (h_valid : is_valid_configuration t)
  (h_BC : length t.B t.C = 2)
  (h_DA : length t.D t.A = 3) :
  length t.A t.B = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_semicircle_theorem_l688_68868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_when_m_is_one_C₁_not_contained_in_C₂_l688_68827

-- Define the circles
def C₁ (m : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2*m*x + 4*y + m^2 - 5 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 + 2*x = 0

-- Define the center and radius of C₁
def center_C₁ (m : ℝ) : ℝ × ℝ := (m, -2)
def radius_C₁ : ℝ := 3

-- Define the center and radius of C₂
def center_C₂ : ℝ × ℝ := (-1, 0)
def radius_C₂ : ℝ := 1

-- Define the distance between centers
noncomputable def distance_between_centers (m : ℝ) : ℝ :=
  Real.sqrt ((m + 1)^2 + 2^2)

-- Theorem 1: C₁ and C₂ intersect when m = 1
theorem circles_intersect_when_m_is_one :
  let m := 1
  |radius_C₁ - radius_C₂| < distance_between_centers m ∧
  distance_between_centers m < radius_C₁ + radius_C₂ := by
  sorry

-- Theorem 2: C₁ is never contained within C₂
theorem C₁_not_contained_in_C₂ :
  ∀ m : ℝ, ¬(∀ x y : ℝ, C₁ m x y → C₂ x y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_when_m_is_one_C₁_not_contained_in_C₂_l688_68827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_rental_options_l688_68804

theorem bus_rental_options (total_people : Nat) (bus_seats : Nat) (minibus_seats : Nat)
  (h1 : total_people = 482)
  (h2 : bus_seats = 42)
  (h3 : minibus_seats = 20) :
  ∃ solutions : Finset (Nat × Nat),
    (∀ (x y : Nat), (x, y) ∈ solutions ↔ bus_seats * x + minibus_seats * y = total_people) ∧
    Finset.card solutions = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_rental_options_l688_68804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_closer_to_six_is_four_sevenths_l688_68805

/-- The probability that a randomly selected point from the interval [0, 7] 
    is closer to 6 than to 0 -/
noncomputable def probability_closer_to_six : ℝ := 4 / 7

/-- Theorem stating that the probability of a randomly selected point 
    from [0, 7] being closer to 6 than to 0 is 4/7 -/
theorem probability_closer_to_six_is_four_sevenths : 
  probability_closer_to_six = 4 / 7 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_closer_to_six_is_four_sevenths_l688_68805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_village_y_population_is_42000_l688_68861

def village_x_population : ℕ := 76000
def village_x_decrease_rate : ℕ := 1200
def village_y_increase_rate : ℕ := 800
def years : ℕ := 17

def village_y_population : ℕ :=
  village_x_population - years * village_x_decrease_rate - years * village_y_increase_rate

#eval village_y_population

theorem village_y_population_is_42000 : village_y_population = 42000 := by
  rfl

#check village_y_population_is_42000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_village_y_population_is_42000_l688_68861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_g_equals_one_ninth_l688_68884

/-- Definition of g(n) -/
noncomputable def g (n : ℕ) : ℝ := ∑' k, (1 : ℝ) / (k + 2 : ℝ) ^ n

/-- The sum of g(n) from n = 3 to infinity equals 1/9 -/
theorem sum_g_equals_one_ninth : ∑' n, g (n + 3) = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_g_equals_one_ninth_l688_68884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bottle_cap_cost_l688_68855

/-- Given that 6 bottle caps cost $12, prove that each bottle cap costs $2. -/
theorem bottle_cap_cost (total_cost : ℝ) : 
  total_cost = 12 →
  ∃ (single_cost : ℝ),
    single_cost * 6 = total_cost ∧
    single_cost = 2 := by
  intro h
  use 2
  constructor
  · rw [h]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bottle_cap_cost_l688_68855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l688_68853

theorem simplify_expression : (64 : ℝ) ^ (-(2 : ℝ) ^ (-(2 : ℝ))) = Real.sqrt 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l688_68853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_point_f_on_interval_l688_68811

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x + 2 * Real.cos x

-- State the theorem
theorem min_point_f_on_interval :
  ∃ (x_min : ℝ), x_min ∈ Set.Icc 0 Real.pi ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 Real.pi → f x_min ≤ f x) ∧
  x_min = 5 * Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_point_f_on_interval_l688_68811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_theorem_l688_68865

/-- Represents a rectangular parallelepiped with diagonal length d -/
structure RectangularParallelepiped where
  d : ℝ
  d_pos : d > 0

/-- Represents a plane cutting the parallelepiped -/
structure CuttingPlane where
  β : ℝ  -- Angle between the plane and the base
  γ : ℝ  -- Angle between AC₁ and the plane
  β_eq : β = 30 * π / 180
  γ_eq : γ = 45 * π / 180

/-- The area of the cross-section formed by the cutting plane -/
noncomputable def cross_section_area (p : RectangularParallelepiped) (c : CuttingPlane) : ℝ :=
  (2 * Real.sqrt 5 * p.d^2) / 12

/-- Theorem stating that the cross-section area is correct -/
theorem cross_section_area_theorem (p : RectangularParallelepiped) (c : CuttingPlane) :
  cross_section_area p c = (2 * Real.sqrt 5 * p.d^2) / 12 := by
  -- Unfold the definition of cross_section_area
  unfold cross_section_area
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_theorem_l688_68865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_odd_square_dissection_l688_68833

/-- A plus figure consisting of 5 unit squares --/
def Plus : ℕ := 5

/-- A minus figure consisting of 2 unit squares --/
def Minus : ℕ := 2

/-- Represents a dissection of a square into pluses and minuses --/
def Dissection := List (Bool × ℕ)

/-- The sum of areas in a dissection --/
def dissectionArea (d : Dissection) : ℕ :=
  d.foldr (fun (b, count) acc => acc + count * (if b then Plus else Minus)) 0

/-- Predicate for a valid dissection of an n × n square --/
def isValidDissection (n : ℕ) (d : Dissection) : Prop :=
  dissectionArea d = n * n

theorem no_odd_square_dissection (n : ℕ) (hn : Odd n) :
  ¬ ∃ (d : Dissection), isValidDissection n d :=
sorry

#check no_odd_square_dissection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_odd_square_dissection_l688_68833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l688_68878

noncomputable section

open Real Set

-- Define the function f
def f (ω : ℝ) (x : ℝ) : ℝ := cos (ω * x + π / 3)

-- State the theorem
theorem omega_range (ω : ℝ) :
  (ω > 0) →
  (∀ x ∈ Icc 0 π, f ω x ∈ Icc (-1) (1/2)) →
  ω ∈ Icc (2/3) (4/3) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l688_68878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diver_descent_time_l688_68806

/-- The time taken for a diver to reach a certain depth -/
noncomputable def time_to_reach_depth (depth : ℝ) (descent_rate : ℝ) : ℝ :=
  depth / descent_rate

/-- Theorem: The time taken for a diver to reach a depth of 3600 feet,
    descending at a rate of 60 feet per minute, is equal to 60 minutes -/
theorem diver_descent_time :
  time_to_reach_depth 3600 60 = 60 := by
  -- Unfold the definition of time_to_reach_depth
  unfold time_to_reach_depth
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diver_descent_time_l688_68806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_in_triangle_l688_68857

-- Define the equilateral triangle ABC
noncomputable def Triangle (A B C : ℝ × ℝ) : Prop :=
  let d := Real.sqrt 3
  dist A B = 2*d ∧ dist B C = 2*d ∧ dist C A = 2*d

-- Define the vector operations
def vec (A B : ℝ × ℝ) : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

noncomputable def vec_length (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- State the theorem
theorem max_distance_in_triangle (A B C P : ℝ × ℝ) :
  Triangle A B C →
  vec_length (vec A P - vec A B - vec A C) = 1 →
  vec_length (vec A P) ≤ 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_in_triangle_l688_68857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_most_accurate_D_l688_68862

/-- The constant D with its central value and error margin -/
def D : ℝ × ℝ := (1.68529, 0.00247)

/-- The lower bound of the range for D -/
def D_lower : ℝ := D.1 - D.2

/-- The upper bound of the range for D -/
def D_upper : ℝ := D.1 + D.2

/-- A function to round a real number to the nearest tenth -/
noncomputable def round_to_tenth (x : ℝ) : ℝ := 
  ⌊x * 10 + 0.5⌋ / 10

/-- The theorem stating that 1.7 is the most accurate value for D -/
theorem most_accurate_D : 
  round_to_tenth D_lower = round_to_tenth D_upper ∧ 
  round_to_tenth D_lower = 1.7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_most_accurate_D_l688_68862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_free_throws_count_l688_68892

/-- Represents the number of two-point shots made by the basketball team. -/
def two_point_shots : ℕ := 0

/-- Represents the number of three-point shots made by the basketball team. -/
def three_point_shots : ℕ := 0

/-- Represents the number of free throws made by the basketball team. -/
def free_throws : ℕ := 0

/-- The points from three-point shots are triple the points from two-point shots. -/
axiom three_point_relation : 3 * three_point_shots = 2 * two_point_shots

/-- The number of free throws is twice the number of two-point shots. -/
axiom free_throw_relation : free_throws = 2 * two_point_shots

/-- The total score of the team is 72 points. -/
axiom total_score : 2 * two_point_shots + 3 * three_point_shots + free_throws = 72

/-- Theorem: Given the conditions, the number of free throws made by the team is 24. -/
theorem free_throws_count : free_throws = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_free_throws_count_l688_68892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l688_68876

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the sides and angle of the triangle
noncomputable def Triangle.a (t : Triangle) : ℝ := sorry
noncomputable def Triangle.b (t : Triangle) : ℝ := sorry
noncomputable def Triangle.c (t : Triangle) : ℝ := sorry
noncomputable def Triangle.γ (t : Triangle) : ℝ := sorry

-- State the theorem
theorem triangle_inequality (t : Triangle) : 
  Real.sin (t.γ / 2) ≤ t.c / (t.a + t.b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l688_68876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lulu_remaining_cash_l688_68860

noncomputable def initial_amount : ℝ := 65
noncomputable def ice_cream_cost : ℝ := 5
noncomputable def tshirt_fraction : ℝ := 1/2
noncomputable def deposit_fraction : ℝ := 1/5

theorem lulu_remaining_cash : 
  let after_ice_cream := initial_amount - ice_cream_cost
  let after_tshirt := after_ice_cream - (tshirt_fraction * after_ice_cream)
  let final_amount := after_tshirt - (deposit_fraction * after_tshirt)
  final_amount = 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lulu_remaining_cash_l688_68860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_meeting_time_is_seven_and_half_minutes_l688_68896

/-- Represents the race between George and Henry in a pool --/
structure RaceInPool where
  poolLength : ℝ
  firstMeetingDistance : ℝ
  georgeSpeed : ℝ
  henrySpeed : ℝ

/-- Calculates the time of the second meeting --/
noncomputable def secondMeetingTime (race : RaceInPool) : ℝ :=
  let firstMeetingTime := race.firstMeetingDistance / race.georgeSpeed
  let georgeFullLapTime := race.poolLength / race.georgeSpeed
  let henryFullLapTime := race.poolLength / race.henrySpeed
  henryFullLapTime + (race.poolLength - race.firstMeetingDistance) / (race.georgeSpeed + race.henrySpeed)

/-- The theorem to be proved --/
theorem second_meeting_time_is_seven_and_half_minutes 
  (race : RaceInPool) 
  (h1 : race.poolLength = 120)
  (h2 : race.firstMeetingDistance = 80)
  (h3 : race.georgeSpeed = 80 / 1.5)
  (h4 : race.henrySpeed = 40 / 1.5) : 
  secondMeetingTime race = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_meeting_time_is_seven_and_half_minutes_l688_68896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_a_2017_l688_68820

/-- The sequence a_n defined as (√2 + 1)^n - (√2 - 1)^n for n ∈ ℕ -/
noncomputable def a (n : ℕ) : ℝ := (Real.sqrt 2 + 1)^n - (Real.sqrt 2 - 1)^n

/-- The floor function, denoting the greatest integer less than or equal to a real number -/
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

/-- The units digit of an integer -/
def units_digit (z : ℤ) : ℕ := z.natAbs % 10

theorem units_digit_a_2017 :
  units_digit (floor (a 2017)) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_a_2017_l688_68820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_component_area_at_least_four_ninths_l688_68829

/-- An isosceles right-angled triangle with a point on its hypotenuse -/
structure IsoscelesRightTriangleWithPoint where
  /-- The length of a leg of the triangle -/
  leg : ℝ
  /-- The position of point P on the hypotenuse, normalized to [0, 1] -/
  k : ℝ
  /-- k is between 0 and 1 -/
  h_k_bounds : 0 ≤ k ∧ k ≤ 1

/-- The areas of the component figures -/
noncomputable def componentAreas (t : IsoscelesRightTriangleWithPoint) : Fin 3 → ℝ
  | 0 => t.k * t.leg^2 / 2  -- Area of triangle XBP
  | 1 => (1 - t.k) * t.leg^2 / 2  -- Area of triangle YPC
  | 2 => t.k * (1 - t.k) * t.leg^2  -- Area of rectangle APXY

/-- The original triangle's area -/
noncomputable def originalArea (t : IsoscelesRightTriangleWithPoint) : ℝ := t.leg^2 / 2

/-- Theorem: At least one component figure has an area ≥ 4/9 of the original triangle -/
theorem component_area_at_least_four_ninths (t : IsoscelesRightTriangleWithPoint) :
  ∃ i : Fin 3, componentAreas t i ≥ (4/9) * originalArea t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_component_area_at_least_four_ninths_l688_68829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_range_a_l688_68817

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then a * x + 1 - a
  else if 1 < x ∧ x ≤ 2 then 2^(x^2 - a*x)
  else 0  -- undefined for x outside [0, 2]

-- State the theorem
theorem f_increasing_range_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 2 → x₂ ∈ Set.Icc 0 2 → x₁ ≠ x₂ →
    (f a x₂ - f a x₁) / (x₂ - x₁) > 0) ↔
  a ∈ Set.Ioo 0 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_range_a_l688_68817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_driver_worked_50_hours_l688_68835

/-- Represents the bus driver's work schedule and compensation --/
structure BusDriverSchedule where
  regularRate : ℚ
  overtimeRateMultiplier : ℚ
  regularHoursLimit : ℕ
  totalCompensation : ℚ

/-- Calculates the total hours worked by the bus driver --/
def totalHoursWorked (schedule : BusDriverSchedule) : ℚ :=
  let overtimeRate := schedule.regularRate * (1 + schedule.overtimeRateMultiplier)
  let regularHours := schedule.regularHoursLimit
  let overtimeHours := (schedule.totalCompensation - schedule.regularRate * regularHours) / overtimeRate
  regularHours + overtimeHours

/-- Theorem stating that the bus driver worked 50 hours given the specified conditions --/
theorem bus_driver_worked_50_hours (schedule : BusDriverSchedule)
  (h1 : schedule.regularRate = 16)
  (h2 : schedule.overtimeRateMultiplier = 3/4)
  (h3 : schedule.regularHoursLimit = 40)
  (h4 : schedule.totalCompensation = 920) :
  totalHoursWorked schedule = 50 := by
  sorry

#eval totalHoursWorked { regularRate := 16, overtimeRateMultiplier := 3/4, regularHoursLimit := 40, totalCompensation := 920 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_driver_worked_50_hours_l688_68835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_hand_distance_l688_68816

/-- The length of the second hand on the clock in centimeters -/
def second_hand_length : ℝ := 8

/-- The number of minutes in the given time period -/
def time_period : ℕ := 45

/-- The distance traveled by the tip of the second hand in centimeters -/
noncomputable def distance_traveled : ℝ := 720 * Real.pi

theorem second_hand_distance :
  2 * Real.pi * second_hand_length * (time_period : ℝ) = distance_traveled := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_hand_distance_l688_68816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_perpendicular_k_function_min_k_value_l688_68885

noncomputable def a : ℝ × ℝ := (Real.sqrt 3, -1)
noncomputable def b : ℝ × ℝ := (1/2, Real.sqrt 3 / 2)

noncomputable def x (t : ℝ) : ℝ × ℝ := t • a + (t^2 - t - 5) • b
noncomputable def y (k t : ℝ) : ℝ × ℝ := -k • a + 4 • b

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem vectors_perpendicular : dot_product a b = 0 := by sorry

theorem k_function (t : ℝ) (h : t ≠ -2) :
  ∃ k, dot_product (x t) (y k t) = 0 ↔ k = (t^2 - t - 5) / (t + 2) := by sorry

theorem min_k_value :
  ∃ t₀ ∈ Set.Ioo (-2 : ℝ) 2, ∀ t ∈ Set.Ioo (-2 : ℝ) 2,
    (t^2 - t - 5) / (t + 2) ≥ (t₀^2 - t₀ - 5) / (t₀ + 2) ∧
    (t₀^2 - t₀ - 5) / (t₀ + 2) = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_perpendicular_k_function_min_k_value_l688_68885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_sqrt5_div_2_l688_68847

/-- The polynomial whose roots form the sides of the quadrilateral -/
noncomputable def f (x : ℝ) : ℝ := x^4 - 5*x^3 + 8*x^2 - 5*x + 1/2

/-- The roots of the polynomial f -/
noncomputable def roots : Finset ℝ := sorry

/-- The area of the quadrilateral with sides equal to the roots of f -/
noncomputable def quadrilateral_area : ℝ := sorry

/-- Theorem stating that the area of the quadrilateral is √5/2 -/
theorem quadrilateral_area_is_sqrt5_div_2 : quadrilateral_area = Real.sqrt 5 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_sqrt5_div_2_l688_68847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_M_intersect_N_l688_68893

-- Define the sets M and N
def M : Set ℝ := {x | Real.log x < 0}
def N : Set ℝ := {x | (1/2)^x ≥ Real.sqrt 2 / 2}

-- State the theorem
theorem complement_M_intersect_N :
  (Set.univ \ M) ∩ N = Set.Iic 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_M_intersect_N_l688_68893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pencils_to_sell_for_profit_l688_68899

/-- The number of pencils that must be sold to make a profit of exactly $180.00,
    given that 2000 pencils were bought at $0.15 each and will be sold at $0.30 each. -/
theorem pencils_to_sell_for_profit (total_pencils : ℕ) (buy_price sell_price : ℚ) 
    (desired_profit : ℚ) : 
  total_pencils = 2000 →
  buy_price = 15 / 100 →
  sell_price = 30 / 100 →
  desired_profit = 180 →
  (↑total_pencils * buy_price + desired_profit) / sell_price = 1600 := by
  intros h1 h2 h3 h4
  sorry

#eval (2000 : ℕ) * (15 : ℚ) / 100 + 180  -- Total cost + desired profit
#eval ((2000 : ℕ) * (15 : ℚ) / 100 + 180) / (30 / 100)  -- Number of pencils to sell

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pencils_to_sell_for_profit_l688_68899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_growth_and_total_employees_l688_68837

/-- Represents a company with its employee count and growth rate -/
structure Company where
  name : String
  december_employees : ℕ
  growth_rate : ℚ

/-- Calculates the number of employees in January given the December count and growth rate -/
def january_employees (c : Company) : ℕ :=
  Int.toNat (⌊(c.december_employees : ℚ) / (1 + c.growth_rate)⌋)

/-- Calculates the number of employees gained from January to December -/
def employees_gained (c : Company) : ℕ :=
  c.december_employees - january_employees c

theorem company_growth_and_total_employees 
  (p : Company) 
  (q : Company) 
  (r : Company) 
  (hp : p.name = "P" ∧ p.december_employees = 515 ∧ p.growth_rate = 15/100)
  (hq : q.name = "Q" ∧ q.december_employees = 558 ∧ q.growth_rate = 105/1000)
  (hr : r.name = "R" ∧ r.december_employees = 611 ∧ r.growth_rate = 195/1000) :
  (employees_gained r ≥ employees_gained p ∧ employees_gained r ≥ employees_gained q) ∧
  (january_employees p + january_employees q + january_employees r = 1464) := by
  sorry

#eval january_employees {name := "P", december_employees := 515, growth_rate := 15/100}
#eval january_employees {name := "Q", december_employees := 558, growth_rate := 105/1000}
#eval january_employees {name := "R", december_employees := 611, growth_rate := 195/1000}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_growth_and_total_employees_l688_68837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_after_steps_l688_68800

/-- Represents the state of the sheets after k steps -/
structure SheetState (m : ℕ) where
  k : ℕ
  numbers : Fin (2^m) → ℕ

/-- The operation of choosing two sheets and updating their values -/
def update_sheets (s : SheetState m) (i j : Fin (2^m)) : SheetState m :=
  { k := s.k + 1,
    numbers := fun k => if k = i ∨ k = j then s.numbers i + s.numbers j else s.numbers k }

/-- The sum of all numbers on the sheets -/
def sum_of_sheets (s : SheetState m) : ℕ :=
  (Finset.univ : Finset (Fin (2^m))).sum s.numbers

/-- The initial state of the sheets -/
def initial_state (m : ℕ) : SheetState m :=
  { k := 0, numbers := fun _ => 1 }

/-- The final state after m * 2^(m-1) steps -/
noncomputable def final_state (m : ℕ) : SheetState m :=
  sorry  -- The implementation of the steps is not required for the theorem statement

/-- The main theorem: sum after steps is at least 4^m -/
theorem sum_after_steps (m : ℕ) :
  sum_of_sheets (final_state m) ≥ 4^m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_after_steps_l688_68800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_hexagon_dots_l688_68812

/-- The number of dots in a hexagon layer -/
def hexagon_layer_dots (n : ℕ) : ℕ := 12 * n

/-- The total number of dots in a hexagon of order n -/
def hexagon_dots (n : ℕ) : ℕ :=
  1 + (Finset.sum (Finset.range n) (fun i => hexagon_layer_dots (i + 1)))

/-- The fourth hexagon has 85 dots -/
theorem fourth_hexagon_dots :
  hexagon_dots 4 = 85 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_hexagon_dots_l688_68812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l688_68856

/-- Represents an ellipse in the Cartesian coordinate system -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0
  h_a_gt_b : a > b
  h_c_lt_a : c < a

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the 2D plane -/
structure Line where
  k : ℝ
  m : ℝ

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := e.c / e.a

/-- Predicate to check if a point lies on an ellipse -/
def on_ellipse (p : Point) (e : Ellipse) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- Predicate to check if a point lies on a line -/
def on_line (p : Point) (l : Line) : Prop :=
  p.x = l.k * p.y + l.m

/-- Predicate to check if two points are perpendicular with respect to the origin -/
def perpendicular_to_origin (p1 p2 : Point) : Prop :=
  p1.x * p2.x + p1.y * p2.y = 0

/-- The main theorem -/
theorem ellipse_eccentricity_range (e : Ellipse) :
  (∃ l : Line, ∃ A B : Point,
    on_line (Point.mk e.c 0) l ∧
    on_ellipse A e ∧
    on_ellipse B e ∧
    on_line A l ∧
    on_line B l ∧
    perpendicular_to_origin A B) →
  ((Real.sqrt 5 - 1) / 2 ≤ eccentricity e) ∧ (eccentricity e < 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l688_68856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_function_satisfies_inequality_l688_68809

theorem no_function_satisfies_inequality : 
  ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, |f (x + y) + Real.sin x + Real.sin y| < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_function_satisfies_inequality_l688_68809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_difference_l688_68882

def repeating_decimal_567 : ℚ := 567 / 999
def repeating_decimal_345 : ℚ := 345 / 999
def repeating_decimal_234 : ℚ := 234 / 999

theorem repeating_decimal_sum_difference :
  repeating_decimal_567 + repeating_decimal_345 - repeating_decimal_234 = 226 / 333 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_difference_l688_68882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_gravel_sale_amount_l688_68803

/-- Represents the sale of gravel over time --/
noncomputable def gravelSale (initialPrice : ℝ) (weeklyReduction : ℝ) (weeklySaleRatio : ℝ) : ℝ :=
  let firstWeekSale := initialPrice * weeklySaleRatio
  let commonRatio := (1 - weeklySaleRatio) * (1 - weeklyReduction)
  firstWeekSale / (1 - commonRatio)

/-- The total amount collected from the gravel sale --/
theorem total_gravel_sale_amount :
  gravelSale 3200 0.1 0.6 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_gravel_sale_amount_l688_68803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l688_68894

theorem expression_evaluation : 
  Real.sqrt 5 * (5 : ℝ)^(1/2 : ℝ) + 15 / 5 * 3 - (9 : ℝ)^(3/2 : ℝ) = -13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l688_68894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_incircle_area_l688_68895

noncomputable def hyperbola_equation (x y : ℝ) : Prop := x^2 - y^2/24 = 1

def is_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

def point_on_hyperbola (P : ℝ × ℝ) : Prop :=
  hyperbola_equation P.1 P.2 ∧ is_first_quadrant P.1 P.2

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

def focal_ratio (P F1 F2 : ℝ × ℝ) : Prop :=
  distance P F1 / distance P F2 = 5/4

theorem hyperbola_incircle_area (P F1 F2 : ℝ × ℝ) :
  point_on_hyperbola P →
  focal_ratio P F1 F2 →
  F1.1 < F2.1 →
  F1.2 = F2.2 →
  (π * (distance P F1 + distance P F2 + distance F1 F2)^2) /
  (4 * (distance P F1 + distance P F2 + distance F1 F2) * (distance P F1 * distance P F2 * Real.sin (Real.arccos ((distance P F1^2 + distance P F2^2 - distance F1 F2^2) / (2 * distance P F1 * distance P F2)))))^(1/2) = 48*π/7 := by
  sorry

#check hyperbola_incircle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_incircle_area_l688_68895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_diameter_theorem_l688_68858

-- Define necessary types
def Point : Type := ℝ × ℝ

-- Define necessary functions and predicates
def is_diameter (A B : Point) (circle : Set Point) : Prop := sorry
def is_tangent (A B : Point) (circle : Set Point) : Prop := sorry
def on_circle (P : Point) (circle : Set Point) : Prop := sorry
def is_intersection (P A C B D : Point) : Prop := sorry
def segment_length (A B : Point) : ℝ := sorry
def diameter_length (A B : Point) : ℝ := sorry

theorem circle_diameter_theorem (A B C D P : Point) (circle : Set Point) (a b c : ℝ) :
  is_diameter A B circle →
  is_tangent A D circle →
  is_tangent B C circle →
  on_circle P circle →
  is_intersection P A C B D →
  segment_length A D = a →
  segment_length B C = b →
  a^2 + b^2 = c^2 →
  diameter_length A B = Real.sqrt 2 * c :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_diameter_theorem_l688_68858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_ln_abs_x_l688_68890

-- Define the curve
noncomputable def curve (x : ℝ) : ℝ := Real.log (abs x)

-- Define the two lines
def line1 (x y : ℝ) : Prop := x - Real.exp 1 * y = 0
def line2 (x y : ℝ) : Prop := x + Real.exp 1 * y = 0

theorem tangent_lines_to_ln_abs_x :
  ∀ (x y : ℝ),
  (y = curve x) →
  (line1 x y ∨ line2 x y) →
  (∃ (ε : ℝ), ε > 0 ∧
    ∀ (h : ℝ), 0 < |h| ∧ |h| < ε →
      (curve (x + h) - y) / h > (x + h - x) / h ∨
      (curve (x + h) - y) / h < (x + h - x) / h) ∧
  (line1 0 0 ∨ line2 0 0) := by
  sorry

#check tangent_lines_to_ln_abs_x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_ln_abs_x_l688_68890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equal_remainder_distribution_l688_68843

theorem no_equal_remainder_distribution (a : ℕ) : 
  ¬ (∀ r : Fin 100, (Finset.filter (λ n : Fin 1000 => a % (n.val + 1) = r.val) Finset.univ).card = 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equal_remainder_distribution_l688_68843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_distance_theorem_l688_68879

/-- The total distance covered by two trains traveling in opposite directions -/
noncomputable def total_distance (speed_a speed_b : ℝ) (time : ℝ) : ℝ :=
  (speed_a * time / 60) + (speed_b * time / 60)

/-- Theorem: The total distance covered by two trains in 25 minutes is 137.5 km -/
theorem train_distance_theorem :
  let speed_a : ℝ := 150
  let speed_b : ℝ := 180
  let time : ℝ := 25
  total_distance speed_a speed_b time = 137.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_distance_theorem_l688_68879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sound_pressure_comparison_l688_68889

noncomputable def sound_pressure_level (p p₀ : ℝ) : ℝ := 20 * Real.log (p / p₀) / Real.log 10

theorem sound_pressure_comparison 
  (p₀ p₁ p₂ p₃ : ℝ) 
  (h_p₀ : p₀ > 0)
  (h_p₁_lower : 60 ≤ sound_pressure_level p₁ p₀)
  (h_p₁_upper : sound_pressure_level p₁ p₀ ≤ 90)
  (h_p₂_lower : 50 ≤ sound_pressure_level p₂ p₀)
  (h_p₂_upper : sound_pressure_level p₂ p₀ ≤ 60)
  (h_p₃ : sound_pressure_level p₃ p₀ = 40) :
  p₂ ≤ p₁ ∧ p₁ ≤ 100 * p₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sound_pressure_comparison_l688_68889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_condition_l688_68814

theorem necessary_not_sufficient_condition : 
  (∀ x : ℝ, |x - 1| ≤ 2 → 3 - x ≥ 0) ∧ 
  (∃ x : ℝ, 3 - x ≥ 0 ∧ |x - 1| > 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_condition_l688_68814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l688_68875

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  sorry

/-- Check if a point is on the ellipse -/
def is_on_ellipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- The left focus of the ellipse -/
noncomputable def left_focus (e : Ellipse) : Point :=
  sorry

/-- The left vertex of the ellipse -/
noncomputable def left_vertex (e : Ellipse) : Point :=
  sorry

/-- Check if a point bisects a line segment -/
def bisects (p1 p2 p3 : Point) (m : Point) : Prop :=
  sorry

theorem ellipse_eccentricity (e : Ellipse) (B C : Point) 
  (h_B : is_on_ellipse B e)
  (h_C : is_on_ellipse C e)
  (h_B_quad1 : B.x > 0 ∧ B.y > 0)
  (h_bisect : ∃ M, bisects (left_vertex e) C M (left_focus e)) :
  eccentricity e = 1/3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l688_68875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_solution_theorem_l688_68832

/-- Represents the salt solution experiment -/
structure SaltSolution where
  n : ℝ  -- Fraction of solution poured out (1/n)
  p : ℝ  -- Percentage increase in salt content
  y : ℝ  -- Initial volume of solution

/-- Calculates the initial salt concentration -/
noncomputable def initial_concentration (s : SaltSolution) : ℝ :=
  (s.p * (s.n + 1)) / (s.n - 1)

/-- Calculates the fraction to pour out for 1.5 times increase -/
def fraction_for_1_5_increase (s : SaltSolution) : ℚ :=
  1 / 3

/-- Main theorem about the salt solution experiment -/
theorem salt_solution_theorem (s : SaltSolution) :
  initial_concentration s = 2 * s.p ∧
  fraction_for_1_5_increase s = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_solution_theorem_l688_68832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_valid_colorings_l688_68813

/-- A coloring of a cube's faces with white and black right-angled triangles -/
structure CubeColoring where
  /-- The coloring function: maps each face and triangle to a color -/
  coloring : Fin 6 → Fin 2 → Bool

/-- Sum of white angles at a vertex -/
def sum_white_angles (v : Fin 8) (c : CubeColoring) : ℕ :=
  sorry

/-- Sum of black angles at a vertex -/
def sum_black_angles (v : Fin 8) (c : CubeColoring) : ℕ :=
  sorry

/-- Predicate to check if a coloring satisfies the angle sum condition -/
def satisfies_angle_sum (c : CubeColoring) : Prop :=
  ∀ v, sum_white_angles v c = sum_black_angles v c

/-- Cube rotations -/
inductive CubeRotation
| identity : CubeRotation
| rotation_x : CubeRotation
| rotation_y : CubeRotation
| rotation_z : CubeRotation

/-- Predicate to check if two colorings are equivalent under cube rotations -/
def equivalent_under_rotation (c1 c2 : CubeColoring) : Prop :=
  ∃ r : CubeRotation, ∀ f t, c1.coloring f t = c2.coloring (sorry) t

/-- The main theorem stating that there are exactly two non-equivalent valid colorings -/
theorem two_valid_colorings :
  ∃! (c1 c2 : CubeColoring),
    satisfies_angle_sum c1 ∧
    satisfies_angle_sum c2 ∧
    ¬equivalent_under_rotation c1 c2 ∧
    ∀ c, satisfies_angle_sum c →
      (equivalent_under_rotation c c1 ∨ equivalent_under_rotation c c2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_valid_colorings_l688_68813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l688_68877

-- Define the circle
def circle_equation (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - a*x + 2 = 0

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop :=
  x + y - 4 = 0

-- Define the point A
def point_A : ℝ × ℝ := (3, 1)

-- Theorem statement
theorem tangent_line_equation (a : ℝ) :
  circle_equation a point_A.1 point_A.2 →
  (∃ (x y : ℝ), circle_equation a x y ∧ tangent_line x y) →
  tangent_line point_A.1 point_A.2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l688_68877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_equivalence_l688_68872

/-- Simple interest calculation function -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem interest_rate_equivalence (P : ℝ) (h : simple_interest P 4 5 = 1680) :
  simple_interest P 5 4 = 1680 := by
  -- Unfold the definition of simple_interest
  unfold simple_interest at h ⊢
  
  -- Use the hypothesis to derive the value of P
  have P_eq : P = 1680 * 100 / (4 * 5) := by
    rw [← h]
    field_simp
    ring
  
  -- Substitute the value of P and simplify
  rw [P_eq]
  field_simp
  ring
  
  -- The proof is complete
  done


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_equivalence_l688_68872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l688_68831

-- Define the function (marked as noncomputable due to Real.log)
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 3*x + 2) / Real.log (1/2)

-- Define the domain of the function
def domain : Set ℝ := {x | x < 1 ∨ x > 2}

-- State the theorem
theorem monotonic_increasing_interval :
  ∀ x ∈ domain, ∀ y ∈ domain,
  x < y ∧ y < 1 → f x < f y :=
by
  -- Proof is omitted
  sorry

#check monotonic_increasing_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l688_68831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l688_68810

/-- The time (in seconds) it takes for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (bridge_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  total_distance / train_speed_mps

/-- Theorem: A train 100 meters long, traveling at 72 kmph, will take 12.5 seconds to cross a bridge 150 meters long -/
theorem train_crossing_bridge :
  train_crossing_time 100 150 72 = 12.5 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof and may cause issues
-- #eval train_crossing_time 100 150 72

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l688_68810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_system_properties_l688_68823

/-- Represents the properties of an ideal monoatomic gas in a cylinder with a light piston and spring -/
structure GasSystem where
  k : ℝ  -- spring stiffness coefficient
  S : ℝ  -- cross-sectional area of the cylinder
  ν : ℝ  -- number of moles of gas
  R : ℝ  -- universal gas constant
  μ : ℝ  -- molar mass of the gas

/-- The equation of state for the gas system -/
noncomputable def equation_of_state (sys : GasSystem) (V : ℝ) : ℝ :=
  (sys.k / sys.S^2) * V

/-- The heat capacity of the gas -/
noncomputable def heat_capacity (sys : GasSystem) : ℝ :=
  2 * sys.ν * sys.R

/-- The specific heat capacity of the gas -/
noncomputable def specific_heat_capacity (sys : GasSystem) : ℝ :=
  2 * sys.R / sys.μ

/-- Theorem stating the heat capacity and specific heat capacity for the given gas system -/
theorem gas_system_properties (sys : GasSystem) :
  heat_capacity sys = 2 * sys.ν * sys.R ∧
  specific_heat_capacity sys = 2 * sys.R / sys.μ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_system_properties_l688_68823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_prime_differences_l688_68887

/-- Definition of the sequence a_n -/
def a (k : ℕ) (n : ℕ) : ℕ :=
  if n < k then 0  -- undefined for n < k
  else if n = k then 2 * k
  else if Nat.gcd (a k (n - 1)) n = 1 then a k (n - 1) + 1
  else 2 * n

/-- Statement of the theorem -/
theorem infinitely_many_prime_differences (k : ℕ) (h : k ≥ 3) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, Nat.Prime (a k (n + 1) - a k n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_prime_differences_l688_68887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_properties_l688_68821

-- Define the circle C
def circle_C (x y : ℝ) (D E : ℝ) : Prop :=
  x^2 + y^2 + D*x + E*y + 3 = 0

-- Define the symmetry line
def symmetry_line (x y : ℝ) : Prop :=
  x + y - 1 = 0

-- Define the chord line
def chord_line (x y : ℝ) : Prop :=
  2*x + 4*y - 1 = 0

-- Define the second quadrant
def second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

theorem circle_C_properties (D E : ℝ) :
  (∀ x y, circle_C x y D E ↔ circle_C x y D E) ∧
  (∀ x y, circle_C x y D E → symmetry_line x y → circle_C y x D E) ∧
  (∃ x y, circle_C x y D E ∧ second_quadrant x y) ∧
  (∃ x y r, circle_C x y D E ∧ r = Real.sqrt 2 ∧ 
    ∀ u v, (u - x)^2 + (v - y)^2 = r^2 → circle_C u v D E) →
  (D = 2 ∧ E = -4) ∧
  (∃ l, l = Real.sqrt 3 ∧ 
    ∀ x₁ y₁ x₂ y₂, circle_C x₁ y₁ D E ∧ circle_C x₂ y₂ D E ∧
      chord_line x₁ y₁ ∧ chord_line x₂ y₂ → 
      (x₁ - x₂)^2 + (y₁ - y₂)^2 = l^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_properties_l688_68821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_fixed_point_l688_68886

/-- A set of non-constant linear functions closed under composition and inversion -/
structure FunctionSet where
  G : Set (ℝ → ℝ)
  linear : ∀ f ∈ G, ∃ a b : ℝ, a ≠ 1 ∧ ∀ x, f x = a * x + b
  closed_comp : ∀ {f g}, f ∈ G → g ∈ G → (f ∘ g) ∈ G
  closed_inv : ∀ f ∈ G, ∃ g ∈ G, ∀ x, g (f x) = x ∧ f (g x) = x

/-- The fixed point of a linear function f(x) = ax + b is b/(1-a) -/
noncomputable def fixedPoint (a b : ℝ) : ℝ := b / (1 - a)

/-- Theorem: All functions in the set have a common fixed point -/
theorem common_fixed_point (S : FunctionSet) :
  ∀ f g, f ∈ S.G → g ∈ S.G → ∃ a b c d : ℝ,
    (∀ x, f x = a * x + b) ∧
    (∀ x, g x = c * x + d) ∧
    fixedPoint a b = fixedPoint c d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_fixed_point_l688_68886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_starters_count_l688_68870

/-- Represents a basketball team with twins and triplets -/
structure BasketballTeam where
  total_players : Nat
  twins : Nat
  triplets : Nat

/-- Calculates the number of ways to choose starters with given conditions -/
def choose_starters (team : BasketballTeam) : Nat :=
  let remaining_players := team.total_players - 5
  let one_twin_choices := Nat.choose 2 1 * Nat.choose remaining_players 1
  let both_twins_choices := Nat.choose 2 2 * Nat.choose (remaining_players - 1) 0
  one_twin_choices + both_twins_choices

/-- The main theorem to prove -/
theorem starters_count (team : BasketballTeam) 
  (h1 : team.total_players = 16) : choose_starters team = 23 := by
  sorry

#eval choose_starters ⟨16, 2, 3⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_starters_count_l688_68870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_line_relationship_l688_68850

/-- Proves that for points A(-2,m) and B(3,n) lying on the line y = -2x + b, m > n -/
theorem points_on_line_relationship (m n b : ℝ) : 
  ((-2 : ℝ), m) ∈ {p : ℝ × ℝ | p.2 = -2 * p.1 + b} →
  ((3 : ℝ), n) ∈ {p : ℝ × ℝ | p.2 = -2 * p.1 + b} →
  m > n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_line_relationship_l688_68850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tensor_product_result_l688_68824

def tensor_product (A B : Set ℝ) : Set ℝ :=
  {z | ∃ x y, x ∈ A ∧ y ∈ B ∧ z = x * y}

def A : Set ℝ := {0, 2}

def B : Set ℝ := {x : ℝ | x^2 - 3*x + 2 = 0}

theorem tensor_product_result : tensor_product A B = {0, 2, 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tensor_product_result_l688_68824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_iodine_mass_percentage_in_calcium_iodide_l688_68801

/-- Calculates the mass percentage of an element in a compound -/
noncomputable def massPercentage (elementMass : ℝ) (compoundMass : ℝ) : ℝ :=
  (elementMass / compoundMass) * 100

/-- Represents the chemical compound Calcium Iodide (CaI2) -/
structure CalciumIodide where
  ca : ℝ
  i : ℝ

/-- Theorem: The mass percentage of iodine in calcium iodide is approximately 86.36% -/
theorem iodine_mass_percentage_in_calcium_iodide :
  let ca_mass : ℝ := 40.08
  let i_mass : ℝ := 126.90
  let cai2 : CalciumIodide := { ca := ca_mass, i := 2 * i_mass }
  let compound_mass : ℝ := cai2.ca + cai2.i
  let i_percentage : ℝ := massPercentage cai2.i compound_mass
  ∃ ε > 0, |i_percentage - 86.36| < ε := by
  sorry

-- Note: We can't use #eval with real numbers, so we'll omit it


end NUMINAMATH_CALUDE_ERRORFEEDBACK_iodine_mass_percentage_in_calcium_iodide_l688_68801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l688_68898

/-- Represents a hyperbola with parameters a and b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0

/-- The eccentricity of the hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := (3 * h.a - h.b) / h.a

/-- The asymptote equation of the hyperbola -/
def asymptote_equation (h : Hyperbola) : ℝ → ℝ → Prop :=
  λ x y ↦ 4 * x = 3 * y ∨ 4 * x = -3 * y

/-- Theorem: The asymptote equation of a hyperbola with the given eccentricity -/
theorem hyperbola_asymptote (h : Hyperbola) 
  (h_ecc : eccentricity h = (3 * h.a - h.b) / h.a) :
  ∀ x y, (x^2 / h.a^2 - y^2 / h.b^2 = 1) → asymptote_equation h x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l688_68898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_and_reflection_l688_68867

noncomputable section

/-- Definition of the ellipse C -/
def ellipse (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 = 1

/-- Definition of a point being in the first quadrant -/
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- Definition of a tangent line to the ellipse at point (x₀, y₀) -/
def tangent_line (a x₀ y₀ x y : ℝ) : Prop := x₀ * x / a^2 + y₀ * y = 1

/-- Definition of the distance between x and y intercepts of a line -/
noncomputable def intercept_distance (a x₀ y₀ : ℝ) : ℝ := Real.sqrt ((a^2 / x₀)^2 + (1 / y₀)^2)

/-- Definition of the reflected ellipse C' -/
def reflected_ellipse (a x y : ℝ) : Prop := 
  ∃ (x' y' : ℝ), ellipse a x' y' ∧ 
  (x + y / Real.sqrt a = x' + y' / Real.sqrt a) ∧
  (x - y / Real.sqrt a = x' - y' / Real.sqrt a)

theorem ellipse_tangent_and_reflection (a : ℝ) :
  a > 0 →
  (∃ (x₀ y₀ : ℝ), 
    ellipse a x₀ y₀ ∧ 
    first_quadrant x₀ y₀ ∧
    tangent_line a x₀ y₀ x₀ y₀ ∧
    (∀ (x' y' : ℝ), ellipse a x' y' ∧ first_quadrant x' y' ∧ tangent_line a x' y' x' y' →
      intercept_distance a x' y' ≥ intercept_distance a x₀ y₀) ∧
    (x₀ + Real.sqrt a * y₀ = Real.sqrt (a * (1 + a)))) ∧
  (0 < a ∧ a < 1/3 ↔ 
    ∃ (x : ℝ), reflected_ellipse a x 0) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_and_reflection_l688_68867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_sqrt_5655_l688_68838

/-- A triangle with vertices on positive coordinate axes -/
structure AxisAlignedTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0

/-- The volume of a tetrahedron formed by the origin and an axis-aligned triangle -/
noncomputable def tetrahedronVolume (t : AxisAlignedTriangle) : ℝ :=
  (1 / 6) * t.a * t.b * t.c

/-- The side lengths of the triangle -/
noncomputable def sideLengths (t : AxisAlignedTriangle) : Fin 3 → ℝ
| 0 => Real.sqrt (t.a^2 + t.b^2)
| 1 => Real.sqrt (t.b^2 + t.c^2)
| 2 => Real.sqrt (t.c^2 + t.a^2)

theorem tetrahedron_volume_sqrt_5655 :
  ∃ t : AxisAlignedTriangle,
    (sideLengths t 0 = 7 ∧ sideLengths t 1 = 9 ∧ sideLengths t 2 = 10) ∧
    tetrahedronVolume t = Real.sqrt 5655 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_sqrt_5655_l688_68838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_zero_l688_68874

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then x^2 * (Real.cos (11 / x))^2 else 0

theorem derivative_at_zero (f : ℝ → ℝ) :
  (∀ x ≠ 0, f x = x^2 * (Real.cos (11 / x))^2) →
  f 0 = 0 →
  deriv f 0 = 0 := by
  intros h1 h2
  sorry

#check derivative_at_zero f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_zero_l688_68874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_limit_negative_infinity_l688_68830

/-- The sequence defined by (n+1)³ - (n+1)² / ((n-1)³ - (n+1)³) for natural numbers n -/
def my_sequence (n : ℕ) : ℚ :=
  (n + 1)^3 - (n + 1)^2 / ((n - 1)^3 - (n + 1)^3)

/-- The limit of the sequence as n approaches infinity is negative infinity -/
theorem my_sequence_limit_negative_infinity :
  Filter.Tendsto my_sequence Filter.atTop Filter.atBot := by
  sorry

#check my_sequence_limit_negative_infinity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_limit_negative_infinity_l688_68830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_k_for_divisible_elements_l688_68881

def S : Finset Nat := Finset.range 32

def divides (a b : Nat) : Prop := ∃ k, b = a * k

def hasThreeDivisibleElements (subset : Finset Nat) : Prop :=
  ∃ a b c, a ∈ subset ∧ b ∈ subset ∧ c ∈ subset ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ divides a b ∧ divides b c

theorem minimum_k_for_divisible_elements :
  ∀ k : Nat, (∀ subset : Finset Nat, subset ⊆ S → subset.card = k → hasThreeDivisibleElements subset) ↔ k ≥ 25 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_k_for_divisible_elements_l688_68881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_plus_beta_value_l688_68802

theorem alpha_plus_beta_value (α β : ℝ) (h1 : 0 < α ∧ α < π) (h2 : 0 < β ∧ β < π)
    (h3 : Real.cos α = -(3 * Real.sqrt 10) / 10) (h4 : Real.sin (2 * α + β) = (1 / 2) * Real.sin β) :
  α + β = (5 / 4) * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_plus_beta_value_l688_68802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_mapping_with_involution_l688_68839

-- Define the set of lattice points
def LatticePoint := ℤ × ℤ

-- Define the set of neighbors for a lattice point
def neighbors (p : LatticePoint) : Set LatticePoint :=
  {(p.1 - 1, p.2), (p.1 + 1, p.2), (p.1, p.2 - 1), (p.1, p.2 + 1)}

-- Define a perfect mapping
def isPerfectMapping {S : Set LatticePoint} (f : S → S) : Prop :=
  Function.Injective f ∧ Function.Surjective f ∧
  ∀ p : S, (f p : LatticePoint) ∈ neighbors (p : LatticePoint)

-- The main theorem
theorem perfect_mapping_with_involution
  {S : Set LatticePoint} (hS : Set.Finite S) (f : S → S) (hf : isPerfectMapping f) :
  ∃ g : S → S, isPerfectMapping g ∧ ∀ p : S, g (g p) = p :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_mapping_with_involution_l688_68839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonals_equal_exists_rhombus_unequal_diagonals_l688_68873

/-- A rectangle is a quadrilateral with four right angles. -/
structure Rectangle :=
  (sides : Fin 4 → ℝ)
  (angles_are_right : ∀ i : Fin 4, (i + 1 : Fin 4) ≠ i → sides i ≠ 0 ∧ sides (i + 1) ≠ 0)

/-- A rhombus is a quadrilateral with four equal sides. -/
structure Rhombus :=
  (side : ℝ)
  (side_positive : side > 0)

/-- The length of a diagonal in a quadrilateral. -/
noncomputable def diagonal_length (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

theorem rectangle_diagonals_equal (r : Rectangle) :
  diagonal_length (r.sides 0) (r.sides 1) = diagonal_length (r.sides 2) (r.sides 3) :=
by sorry

theorem exists_rhombus_unequal_diagonals :
  ∃ (rh : Rhombus), diagonal_length rh.side rh.side ≠ diagonal_length rh.side (Real.sqrt 3 * rh.side) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonals_equal_exists_rhombus_unequal_diagonals_l688_68873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eleventh_term_is_671_l688_68852

def mySequence (n : ℕ) : ℕ := sorry

theorem eleventh_term_is_671 :
  (mySequence 1 = 11) →
  (∀ n : ℕ, n ≥ 1 → mySequence (n + 1) - mySequence n = 12 * n) →
  mySequence 11 = 671 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eleventh_term_is_671_l688_68852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_special_angle_l688_68826

/-- In a right triangle ABC with ∠A = 90°, ∠B = α, and ∠C the right angle opposite to A:
    - tan(α/2) = √3
    - BL bisects the hypotenuse AC
    - BM is the median from B
    - θ is the angle between BL and BM
Then tan θ = (-1-√3)/3 -/
theorem right_triangle_special_angle (α θ : ℝ) : 
  0 < α → α < π/2 →  -- α is an acute angle
  Real.tan (α/2) = Real.sqrt 3 →
  Real.tan θ = (-1 - Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_special_angle_l688_68826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_amount_l688_68848

/-- Represents the initial investment amount in dollars -/
def P : ℝ := 650

/-- Represents the annual interest rate as a decimal -/
def r : ℝ := 0.05  -- Assuming a 5% interest rate for this example

/-- Peter's investment duration in years -/
def t₁ : ℝ := 3

/-- David's investment duration in years -/
def t₂ : ℝ := 4

/-- Peter's final amount after investment -/
def A₁ : ℝ := 815

/-- David's final amount after investment -/
def A₂ : ℝ := 870

/-- Simple interest formula: A = P(1 + rt) where A is the final amount, 
    P is the principal (initial investment), r is the interest rate, 
    and t is the time in years -/
axiom simple_interest_formula (P r t A : ℝ) : A = P * (1 + r * t)

theorem investment_amount : P = 650 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_amount_l688_68848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fifth_roots_property_l688_68845

/-- A fifth root of unity -/
noncomputable def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)

/-- Sum of powers of ω -/
noncomputable def z : ℂ := ω + ω^2 + ω^3 + ω^4

/-- Theorem: z^2 + z + 1 = 1 -/
theorem sum_of_fifth_roots_property : z^2 + z + 1 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fifth_roots_property_l688_68845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sequence_length_l688_68846

-- Define a property for the sequence
def ValidSequence (s : List Nat) : Prop :=
  -- All numbers in the sequence are odd
  (∀ n, n ∈ s → Odd n) ∧
  -- No number in the sequence divides another
  (∀ a b, a ∈ s → b ∈ s → a ≠ b → ¬(a ∣ b)) ∧
  -- For any three numbers, two sum to a multiple of the third
  (∀ a b c, a ∈ s → b ∈ s → c ∈ s → a ≠ b → b ≠ c → a ≠ c → 
    (a + b) % c = 0 ∨ (a + c) % b = 0 ∨ (b + c) % a = 0)

-- Theorem statement
theorem max_sequence_length :
  (∃ s : List Nat, ValidSequence s ∧ s.length = 5) ∧
  (∀ s : List Nat, ValidSequence s → s.length ≤ 5) := by
  sorry

#check max_sequence_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sequence_length_l688_68846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_k_value_l688_68871

/-- The equation of a circle that touches the y-axis and has a radius of 5 -/
def circle_equation (k : ℝ) : Prop :=
  ∃ x y, x^2 + 8*x + y^2 + 4*y - k = 0

/-- The circle touches the y-axis -/
def touches_y_axis (k : ℝ) : Prop :=
  ∃ y, circle_equation k ∧ (0^2 + 8*0 + y^2 + 4*y - k = 0)

/-- The circle has a radius of 5 -/
def has_radius_5 (k : ℝ) : Prop :=
  ∀ x y, (x^2 + 8*x + y^2 + 4*y - k = 0) ↔ ((x + 4)^2 + (y + 2)^2 = 25)

/-- The value of k for which the equation represents a circle touching the y-axis with radius 5 -/
theorem circle_k_value : 
  ∃! k, circle_equation k ∧ touches_y_axis k ∧ has_radius_5 k ∧ k = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_k_value_l688_68871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_true_l688_68808

theorem contrapositive_true : 
  (∀ a : ℝ, a^2 ≤ 9 → a < 4) ↔ (∀ a : ℝ, a ≥ 4 → a^2 > 9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_true_l688_68808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_proof_l688_68844

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 6*y + 8 = 0

-- Define the circumference of the circle
noncomputable def circle_circumference : ℝ := 2 * Real.sqrt 2 * Real.pi

-- Theorem statement
theorem circle_circumference_proof :
  ∃ (r : ℝ), r > 0 ∧
  (∀ (x y : ℝ), circle_equation x y ↔ (x - 1)^2 + (y + 3)^2 = r^2) ∧
  circle_circumference = 2 * Real.pi * r := by
  -- Proof goes here
  sorry

#check circle_circumference_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_proof_l688_68844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_tan_is_even_and_smallest_period_l688_68866

-- Define the property of being an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Define the property of having a period
def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

-- Define the absolute value of tangent function
noncomputable def abs_tan (x : ℝ) : ℝ := |Real.tan x|

-- State the theorem
theorem abs_tan_is_even_and_smallest_period :
  is_even_function abs_tan ∧
  has_period abs_tan π ∧
  (∀ p : ℝ, 0 < p → has_period abs_tan p → π ≤ p) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_tan_is_even_and_smallest_period_l688_68866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tobys_girl_friends_l688_68815

/-- The proportion of Toby's friends who are boys -/
def boysProportion : ℚ := 55 / 100

/-- The number of Toby's friends who are boys -/
def numBoys : ℕ := 33

/-- The total number of Toby's friends -/
noncomputable def totalFriends : ℕ := 
  (numBoys * 100 / boysProportion.num).toNat

/-- The number of Toby's friends who are girls -/
noncomputable def numGirls : ℕ := totalFriends - numBoys

theorem tobys_girl_friends : numGirls = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tobys_girl_friends_l688_68815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_prism_coloring_l688_68818

-- Define a triangular prism
def TriangularPrism := Fin 6

-- Define the number of available colors
def NumColors := 5

-- Define a coloring function
def Coloring := TriangularPrism → Fin NumColors

-- Define the adjacency relation for a triangular prism
def IsAdjacent (i j : TriangularPrism) : Prop :=
  (i.val + 1) % 6 = j.val ∨ 
  (j.val + 1) % 6 = i.val ∨ 
  (i.val + 3) % 6 = j.val

-- Define a predicate for valid colorings
def ValidColoring (c : Coloring) : Prop :=
  ∀ i j : TriangularPrism, IsAdjacent i j → c i ≠ c j

-- Provide instances for Fintype and DecidablePred
instance : Fintype Coloring := by sorry

instance : DecidablePred ValidColoring := by sorry

-- State the theorem
theorem triangular_prism_coloring :
  (Finset.filter ValidColoring (Finset.univ : Finset Coloring)).card = 1920 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_prism_coloring_l688_68818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_result_theorem_l688_68807

/-- Represents the number of votes for each candidate and the total votes -/
structure ElectionResult where
  total_votes : ℕ
  valid_votes : ℕ
  votes_A : ℕ
  votes_B : ℕ
  votes_C : ℕ
  votes_D : ℕ

/-- Defines the conditions of the election result -/
def valid_election_result (e : ElectionResult) : Prop :=
  e.total_votes = 15000 ∧
  e.valid_votes = (75 * e.total_votes) / 100 ∧
  e.votes_A = e.votes_B + (20 * e.total_votes) / 100 ∧
  e.votes_C = (95 * e.votes_B) / 100 ∧
  e.votes_D = (92 * e.votes_A) / 100 ∧
  e.valid_votes = e.votes_A + e.votes_B + e.votes_C + e.votes_D

/-- Defines an approximate equality for natural numbers -/
def approx_equal (a b : ℕ) (ε : ℕ := 1) : Prop :=
  (a > b → a - b ≤ ε) ∧ (b > a → b - a ≤ ε)

theorem election_result_theorem (e : ElectionResult) 
  (h : valid_election_result e) : 
  approx_equal (e.votes_B + e.votes_C) 3731 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_result_theorem_l688_68807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l688_68891

/-- The function f(x) = x / (1 + |x|) -/
noncomputable def f (x : ℝ) : ℝ := x / (1 + abs x)

/-- Recursive definition of f_n -/
noncomputable def f_n : ℕ → (ℝ → ℝ)
| 0 => f
| n + 1 => f ∘ f_n n

theorem f_properties :
  (∀ y ∈ Set.range f, -1 < y ∧ y < 1) ∧ 
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 0) ∧
  f_n 9 (1/2) = 1/12 := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l688_68891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_six_l688_68841

theorem g_of_six (g : ℝ → ℝ) (h : ∀ x : ℝ, g (4*x - 2) = x^2 - x + 2) :
  g 6 = 4.375 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_six_l688_68841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_result_l688_68864

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then Real.log x / Real.log (1/2)
  else 2^(x-1)

-- State the theorem
theorem f_composition_result : f (f (1/16)) = 8 := by
  -- Proof steps will go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_result_l688_68864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_expansion_of_five_thirty_thirds_ninety_second_digit_of_five_thirty_thirds_l688_68869

theorem decimal_expansion_of_five_thirty_thirds (n : ℕ) : 
  n % 2 = 0 → n > 0 → 
  (((5 : ℚ) / 33 * 10^n).floor % 10 : ℤ) = 5 :=
by sorry

theorem ninety_second_digit_of_five_thirty_thirds : 
  (((5 : ℚ) / 33 * 10^92).floor % 10 : ℤ) = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_expansion_of_five_thirty_thirds_ninety_second_digit_of_five_thirty_thirds_l688_68869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acid_solution_percentage_l688_68822

/-- Given a solution with pure acid and total volume, calculate the percentage of pure acid -/
noncomputable def acid_percentage (pure_acid_volume : ℝ) (total_volume : ℝ) : ℝ :=
  (pure_acid_volume / total_volume) * 100

/-- Theorem: The percentage of pure acid in the given solution is 25% -/
theorem acid_solution_percentage : acid_percentage 2.5 10 = 25 := by
  -- Unfold the definition of acid_percentage
  unfold acid_percentage
  -- Evaluate the expression
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acid_solution_percentage_l688_68822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l688_68840

-- Define the constants
noncomputable def a : ℝ := 0.6^4
noncomputable def b : ℝ := Real.log 3 / Real.log 2
noncomputable def c : ℝ := 0.6^5

-- State the theorem
theorem order_of_abc : b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l688_68840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_not_below_x_axis_PQRS_l688_68851

/-- A parallelogram in a 2D plane --/
structure Parallelogram where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ

/-- The specific parallelogram PQRS from the problem --/
def PQRS : Parallelogram :=
  { P := (5, 5)
    Q := (-1, -5)
    R := (-11, -5)
    S := (-5, 5) }

/-- A point is not below the x-axis if its y-coordinate is non-negative --/
def notBelowXAxis (p : ℝ × ℝ) : Prop := p.2 ≥ 0

/-- The area of a region in the plane --/
noncomputable def area (region : Set (ℝ × ℝ)) : ℝ := sorry

/-- The region enclosed by a parallelogram --/
def parallelogramRegion (p : Parallelogram) : Set (ℝ × ℝ) := sorry

/-- The probability of an event occurring when selecting a point randomly from a region --/
noncomputable def probability (region : Set (ℝ × ℝ)) (event : (ℝ × ℝ) → Prop) : ℝ :=
  (area {p ∈ region | event p}) / (area region)

/-- The theorem stating the probability of selecting a point not below the x-axis in PQRS --/
theorem prob_not_below_x_axis_PQRS : 
  probability (parallelogramRegion PQRS) notBelowXAxis = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_not_below_x_axis_PQRS_l688_68851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l688_68888

/-- 
  Given a pyramid with the following properties:
  - The base is an isosceles trapezoid
  - The base has an acute angle α
  - The base has an area of S
  - All lateral faces form the same angle β with the plane of the base
  This theorem states that the volume of the pyramid is (S * tan(β) * sqrt(S * sin(α))) / 6
-/
noncomputable def volume_of_pyramid_with_isosceles_trapezoid_base (S α β : ℝ) : ℝ :=
  (S * Real.tan β * Real.sqrt (S * Real.sin α)) / 6

theorem pyramid_volume (S α β : ℝ) (h_S : S > 0) (h_α : 0 < α ∧ α < π/2) (h_β : 0 < β ∧ β < π/2) :
  volume_of_pyramid_with_isosceles_trapezoid_base S α β = 
    (S * Real.tan β * Real.sqrt (S * Real.sin α)) / 6 :=
by
  -- Unfold the definition of volume_of_pyramid_with_isosceles_trapezoid_base
  unfold volume_of_pyramid_with_isosceles_trapezoid_base
  -- The equality is now trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l688_68888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_expression_equality_second_expression_equality_l688_68849

-- First expression
theorem first_expression_equality :
  ((9/4 : ℝ) ^ (1/2 : ℝ)) - 1 - ((1/16 : ℝ) ^ (3/4 : ℝ)) = 3/8 := by sorry

-- Second expression
theorem second_expression_equality :
  (4 : ℝ) ^ (Real.log 5 / Real.log 4) - 5 + Real.log 500 / Real.log 10 + Real.log 2 / Real.log 10 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_expression_equality_second_expression_equality_l688_68849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_theorem_l688_68819

open Real

theorem tan_sum_theorem (x y : ℝ) 
  (h1 : (sin x / cos y) + (sin y / cos x) = 2)
  (h2 : (cos x / sin y) + (cos y / sin x) = 4) :
  (tan x / tan y) + (tan y / tan x) = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_theorem_l688_68819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l688_68836

/-- The function f(x) = sin(ωx) + √3 * cos(ωx) -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) + Real.sqrt 3 * Real.cos (ω * x)

/-- Theorem: If f(α) = -2, f(β) = 0, and the minimum value of |α - β| is 3π/4, then ω = 2/3 -/
theorem omega_value (ω α β : ℝ) (h_pos : ω > 0) 
  (h_alpha : f ω α = -2) 
  (h_beta : f ω β = 0) 
  (h_min : ∀ (k : ℤ), |α - β| ≤ |α - β + (2 * k * Real.pi) / ω|) 
  (h_diff : |α - β| = (3 * Real.pi) / 4) : 
  ω = 2/3 := by
  sorry

#check omega_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l688_68836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_close_points_l688_68842

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define a rectangle
structure Rectangle where
  width : ℝ
  height : ℝ

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define a function to check if a point is within a rectangle
def isWithinRectangle (p : Point) (r : Rectangle) : Prop :=
  0 ≤ p.1 ∧ p.1 ≤ r.width ∧ 0 ≤ p.2 ∧ p.2 ≤ r.height

-- Theorem statement
theorem exists_close_points
  (r : Rectangle)
  (h_width : r.width = 3)
  (h_height : r.height = 4)
  (points : Finset Point)
  (h_card : points.card = 6)
  (h_within : ∀ p ∈ points, isWithinRectangle p r) :
  ∃ p1 p2 : Point, p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧ distance p1 p2 ≤ Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_close_points_l688_68842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_retailer_shirts_count_l688_68880

/-- The total number of shirts the retailer is selling -/
def total_shirts : ℕ := 10

/-- The prices of the first three shirts sold -/
def first_three_prices : List ℝ := [82, 100, 90]

/-- The number of remaining shirts -/
def remaining_shirts : ℕ := 7

/-- The minimum average price of the remaining shirts -/
def min_avg_remaining : ℝ := 104

/-- The desired overall average price -/
def desired_avg : ℝ := 100

theorem retailer_shirts_count :
  (List.sum first_three_prices + remaining_shirts * min_avg_remaining) / total_shirts ≥ desired_avg := by
  sorry

#eval total_shirts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_retailer_shirts_count_l688_68880
