import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l486_48605

-- Define the angle θ
noncomputable def θ : Real := Real.arctan (-Real.sqrt 2)

-- Define the angle α
noncomputable def α : Real := Real.arcsin (Real.sqrt 6 / 3)

-- Theorem for part 1
theorem part_one :
  (-Real.cos (3 * Real.pi / 2 + θ) + Real.sqrt 2 * Real.sin (Real.pi / 2 + θ)) /
  (Real.sin (2 * Real.pi - θ) - 2 * Real.sqrt 2 * Real.cos (-θ)) = -2 := by
  sorry

-- Theorem for part 2
theorem part_two :
  Real.sin (α - Real.pi / 6) = (3 * Real.sqrt 2 - Real.sqrt 3) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l486_48605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_2016_l486_48612

theorem divisible_by_2016 (S : Finset ℕ) : 
  S.card = 65 → (∀ n ∈ S, n ≤ 2016) → 
  ∃ a b c d, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (a + b - c - d) % 2016 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_2016_l486_48612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_CD_l486_48649

/-- The line equation mx + y + 3m - √3 = 0 -/
def line_equation (m x y : ℝ) : Prop := m * x + y + 3 * m - Real.sqrt 3 = 0

/-- The circle equation x^2 + y^2 = 12 -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 12

/-- A and B are intersection points of the line and circle -/
def intersection_points (m x_a y_a x_b y_b : ℝ) : Prop :=
  line_equation m x_a y_a ∧ circle_equation x_a y_a ∧
  line_equation m x_b y_b ∧ circle_equation x_b y_b

/-- C and D are points on the x-axis -/
def points_on_x_axis (x_c x_d : ℝ) : Prop := True

/-- AC and BD are perpendicular to the x-axis -/
def perpendicular_to_x_axis (x_a x_b x_c x_d : ℝ) : Prop :=
  x_a = x_c ∧ x_b = x_d

/-- The length of AB is 2√3 -/
def length_AB (x_a y_a x_b y_b : ℝ) : Prop :=
  (x_a - x_b)^2 + (y_a - y_b)^2 = 12

/-- The length of CD -/
def length_CD (x_c x_d : ℝ) : ℝ := |x_c - x_d|

theorem length_of_CD
  (m x_a y_a x_b y_b x_c x_d : ℝ)
  (h1 : intersection_points m x_a y_a x_b y_b)
  (h2 : points_on_x_axis x_c x_d)
  (h3 : perpendicular_to_x_axis x_a x_b x_c x_d)
  (h4 : length_AB x_a y_a x_b y_b) :
  length_CD x_c x_d = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_CD_l486_48649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_right_is_four_l486_48696

/-- Represents a 5x5 grid of integers -/
def Grid := Fin 5 → Fin 5 → Fin 5

/-- Check if a grid is valid according to the rules -/
def is_valid_grid (g : Grid) : Prop :=
  (∀ i j k, i ≠ j → g i k ≠ g j k) ∧  -- Each column contains distinct numbers
  (∀ i j k, j ≠ k → g i j ≠ g i k) ∧  -- Each row contains distinct numbers
  (∀ i j, g i j < 5)                  -- All numbers are between 0 and 4

/-- The initial grid setup -/
def initial_grid : Grid :=
  λ i j =>
    if i = 0 ∧ j = 0 then 0     -- 1 in top-left corner
    else if i = 0 ∧ j = 2 then 2 -- 3 in top row, third column
    else if i = 1 ∧ j = 0 then 2 -- 3 in second row, first column
    else if i = 1 ∧ j = 1 then 3 -- 4 in second row, second column
    else if i = 2 ∧ j = 3 then 4 -- 5 in third row, fourth column
    else 4  -- placeholder for empty cells

theorem lower_right_is_four (g : Grid) :
  is_valid_grid g → (∀ i j, g i j = initial_grid i j ∨ initial_grid i j = 4) →
  g 4 4 = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_right_is_four_l486_48696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_implies_m_equals_two_l486_48657

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![4, 2]

-- Define vector c as a function of m
def c (m : ℝ) : Fin 2 → ℝ := ![m + 4, 2*m + 2]

-- Define the dot product of two 2D vectors
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- Define the magnitude of a 2D vector
noncomputable def magnitude (v : Fin 2 → ℝ) : ℝ := Real.sqrt ((v 0)^2 + (v 1)^2)

-- Theorem statement
theorem angle_bisector_implies_m_equals_two :
  ∀ m : ℝ, 
    (dot_product a (c m) / (magnitude a * magnitude (c m)) = 
     dot_product b (c m) / (magnitude b * magnitude (c m))) 
    → m = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_implies_m_equals_two_l486_48657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l486_48611

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x)

-- Define the monotonically increasing interval
def mono_increasing_interval (k : ℤ) : Set ℝ := 
  Set.Icc (k * Real.pi - 5 * Real.pi / 12) (k * Real.pi + Real.pi / 12)

-- Theorem statement
theorem f_monotone_increasing :
  ∀ k : ℤ, StrictMonoOn f (mono_increasing_interval k) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l486_48611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_channel_bottom_width_l486_48624

/-- Represents the cross-section of a water channel -/
structure WaterChannel where
  topWidth : ℝ
  bottomWidth : ℝ
  depth : ℝ
  area : ℝ

/-- Calculates the area of a trapezoidal cross-section -/
noncomputable def trapezoidArea (c : WaterChannel) : ℝ :=
  (1/2) * (c.topWidth + c.bottomWidth) * c.depth

/-- Theorem: Given the specifications of the water channel, the bottom width is 8 meters -/
theorem water_channel_bottom_width :
  ∃ (c : WaterChannel),
    c.topWidth = 14 ∧
    c.depth = 80 ∧
    c.area = 880 ∧
    trapezoidArea c = c.area ∧
    c.bottomWidth = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_channel_bottom_width_l486_48624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_A_B_intersect_complement_A_B_l486_48661

-- Define sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

-- Define the complement of A in ℝ
def C_R_A : Set ℝ := {x | x < 3 ∨ x > 7}

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = Set.Icc 3 7 := by sorry

-- Theorem for the intersection of complement of A and B
theorem intersect_complement_A_B : C_R_A ∩ B = {x | (2 < x ∧ x < 3) ∨ (7 < x ∧ x < 10)} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_A_B_intersect_complement_A_B_l486_48661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_m_l486_48630

open Real

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 4*x

-- State the theorem
theorem min_value_of_m :
  ∃ (m_min : ℝ), ∀ (θ : ℝ), 
    (f (2 * cos θ - 1) ≥ m_min) ∧ 
    (∃ (θ_min : ℝ), f (2 * cos θ_min - 1) = m_min ∧ m_min = -4) :=
by
  -- We'll prove this later
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_m_l486_48630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_or_swap_l486_48686

/-- A tree is a connected acyclic graph -/
structure MyTree (α : Type*) where
  vertices : Set α
  edges : Set (α × α)
  is_acyclic : Bool
  is_connected : Bool

/-- An isomorphism between trees -/
structure TreeIsomorphism {α : Type*} (T : MyTree α) where
  to_fun : α → α
  is_bijective : Bool
  preserves_edges : Bool

theorem fixed_point_or_swap {α : Type*} (T : MyTree α) (f : TreeIsomorphism T) :
  (∃ a, a ∈ T.vertices ∧ f.to_fun a = a) ∨
  (∃ a b, a ∈ T.vertices ∧ b ∈ T.vertices ∧ (a, b) ∈ T.edges ∧ f.to_fun a = b ∧ f.to_fun b = a) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_or_swap_l486_48686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_product_l486_48646

theorem polynomial_roots_product (k : ℝ) : 
  let p : ℝ → ℝ := λ x => x^4 - 18*x^3 + k*x^2 + 200*x - 1984
  (∃ (x₁ x₂ x₃ x₄ : ℝ), 
    (∀ x, p x = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) ∧
    (∃ (i j : Fin 4), i ≠ j ∧ 
      let roots := [x₁, x₂, x₃, x₄]
      roots[i.val] * roots[j.val] = -32)) →
  k = 86 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_product_l486_48646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_fraction_l486_48663

/-- Given two positive integers a and b where a ≠ b, prove that (a * b^2) / (a + b) is a prime number if and only if (a, b) = (6, 2) -/
theorem unique_prime_fraction (a b : ℕ+) (h : a ≠ b) :
  (↑a * ↑b^2 : ℚ) / (↑a + ↑b) ∈ Set.range (Nat.cast : ℕ → ℚ) ∧ 
  Nat.Prime (((↑a * ↑b^2 : ℚ) / (↑a + ↑b)).num.toNat) ↔ 
  (a, b) = (6, 2) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_fraction_l486_48663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_square_perimeter_l486_48619

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents the square and folded triangle -/
structure FoldedSquare where
  A : Point
  B : Point
  C : Point
  D : Point
  C' : Point
  E : Point
  h_square : A.x = 0 ∧ A.y = 1 ∧ B.x = 0 ∧ B.y = 0 ∧ C.x = 1 ∧ C.y = 0 ∧ D.x = 1 ∧ D.y = 1
  h_C'_on_AD : C'.x = 1 ∧ C'.y = 1/4
  h_E_on_AB : E.y = 0
  h_E_on_AC' : E.y = -3/4 * E.x + 1

/-- The main theorem to prove -/
theorem folded_square_perimeter (fs : FoldedSquare) :
  distance fs.A fs.E + distance fs.E fs.C' + distance fs.C' fs.A = 10/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_square_perimeter_l486_48619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recover_investment_pens_l486_48636

/-- The number of pens that need to be sold to recover the initial investment -/
def pens_sold_to_recover (total_pens : ℕ) (profit_percentage : ℚ) : ℚ :=
  (total_pens : ℚ) / (1 + profit_percentage)

/-- Theorem stating that 20 pens need to be sold to recover the initial investment -/
theorem recover_investment_pens : 
  ⌊pens_sold_to_recover 30 (1/2)⌋ = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recover_investment_pens_l486_48636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_1000_eq_1003_l486_48638

/-- The sequence bₙ as defined in the problem -/
noncomputable def b : ℕ → ℕ
  | 0 => 2  -- Added case for 0
  | 1 => 2
  | 2 => 2
  | 3 => 2
  | n + 4 => Nat.card {y : ℝ | y^3 - 3 * b n * y + b (n - 1) * b (n - 2) = 0}

/-- The sum of the first 1000 terms of the sequence bₙ -/
noncomputable def sum_1000 : ℕ := (Finset.range 1000).sum (fun i => b (i + 1))

/-- The main theorem stating that the sum of the first 1000 terms of bₙ is 1003 -/
theorem sum_1000_eq_1003 : sum_1000 = 1003 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_1000_eq_1003_l486_48638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_is_1950_l486_48647

/-- Represents a car with its fuel consumption rate -/
structure Car where
  fuelConsumption : ℝ  -- Fuel consumption per 100 km

/-- Calculates the distance a car can travel with a given amount of fuel -/
noncomputable def distanceTraveled (car : Car) (fuelAmount : ℝ) : ℝ :=
  (fuelAmount / car.fuelConsumption) * 100

/-- Theorem: The total distance traveled by all four cars is 1950 km -/
theorem total_distance_is_1950 (carU carV carW carX : Car)
  (h1 : carU.fuelConsumption = 20)
  (h2 : carV.fuelConsumption = 25)
  (h3 : carW.fuelConsumption = 5)
  (h4 : carX.fuelConsumption = 10)
  (fuelAmount : ℝ)
  (h5 : fuelAmount = 50) :
  distanceTraveled carU fuelAmount +
  distanceTraveled carV fuelAmount +
  distanceTraveled carW fuelAmount +
  distanceTraveled carX fuelAmount = 1950 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_is_1950_l486_48647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_alpha_value_l486_48674

/-- The acceleration due to gravity -/
noncomputable def g : ℝ := 10

/-- The initial velocity -/
noncomputable def V₀ : ℝ := 20

/-- The distance from the origin to the puck shooter -/
noncomputable def d : ℝ := 16

/-- The distance from the origin to the goal line -/
noncomputable def goal_distance : ℝ := 25

/-- The trajectory of the puck -/
noncomputable def trajectory (α : ℝ) (t : ℝ) : ℝ × ℝ :=
  (d + V₀ * t * Real.cos α, V₀ * t * Real.sin α - g * t^2 / 2)

/-- The maximum value of tan(α) -/
noncomputable def max_tan_α : ℝ := (5 + 2 * Real.sqrt 121.2) / 121

/-- Theorem stating the maximum value of tan(α) -/
theorem max_tan_alpha_value :
  ∃ (α : ℝ), ∀ (β : ℝ),
    (∀ (x : ℝ), x ∈ Set.Icc d goal_distance →
      (trajectory α ((x - d) / (V₀ * Real.cos α))).2 ≤ x / goal_distance) →
    Real.tan α ≥ Real.tan β := by
  sorry

#check max_tan_alpha_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_alpha_value_l486_48674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_sum_l486_48677

/-- The equation of the ellipse -/
noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  3 * x^2 + 2 * x * y + 4 * y^2 - 18 * x - 28 * y + 50 = 0

/-- The set of points (x, y) on the ellipse -/
noncomputable def ellipse_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ellipse_eq p.1 p.2}

/-- The ratio y/x for a point (x, y) -/
noncomputable def ratio (p : ℝ × ℝ) : ℝ := p.2 / p.1

/-- Theorem stating the existence of min and max ratios and their sum -/
theorem ellipse_ratio_sum :
  ∃ (min_ratio max_ratio : ℝ),
    (∀ p ∈ ellipse_points, min_ratio ≤ ratio p ∧ ratio p ≤ max_ratio) ∧
    min_ratio + max_ratio = 13 := by
  sorry

#check ellipse_ratio_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_sum_l486_48677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_radical_l486_48602

-- Define the concept of a quadratic radical
def QuadraticRadical (x : ℝ) : Prop := ∃ (y : ℝ), x = Real.sqrt y

-- Define the concept of simplest quadratic radical
def SimplestQuadraticRadical (x : ℝ) : Prop :=
  QuadraticRadical x ∧ ∀ y : ℝ, QuadraticRadical y → (∃ z : ℝ, z ≠ 1 ∧ x = z * y) → False

-- Define the given expressions
noncomputable def expr1 : ℝ := Real.sqrt (2/3)
noncomputable def expr2 : ℝ := Real.sqrt 3
noncomputable def expr3 : ℝ := Real.sqrt 9
noncomputable def expr4 : ℝ := Real.sqrt 12

-- State the theorem
theorem simplest_quadratic_radical :
  SimplestQuadraticRadical expr2 ∧
  ¬SimplestQuadraticRadical expr1 ∧
  ¬SimplestQuadraticRadical expr3 ∧
  ¬SimplestQuadraticRadical expr4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_radical_l486_48602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l486_48664

noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 then 2/x - 1
  else if x < 0 then 2/(-x) - 1
  else 1

theorem f_properties : 
  (∀ x, f x = f (-x)) ∧ 
  (f (-1) = 1) ∧
  (∀ a b, 0 < b ∧ b < a → f a < f b) ∧
  (∀ x, x < 0 → f x = 2/(-x) - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l486_48664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_distance_30N_60E_diff_l486_48667

/-- The spherical distance between two points on a sphere --/
noncomputable def spherical_distance (R : ℝ) (lat : ℝ) (long_diff : ℝ) : ℝ :=
  R * Real.arccos ((3 * Real.cos lat ^ 2 + 1) / 4)

/-- Theorem: Spherical distance between two points at 30°N latitude and 60° longitude difference --/
theorem spherical_distance_30N_60E_diff (R : ℝ) (h : R > 0) :
  spherical_distance R (π/6) (π/3) = R * Real.arccos (5/8) := by
  sorry

#check spherical_distance_30N_60E_diff

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_distance_30N_60E_diff_l486_48667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_to_y_equals_one_l486_48666

theorem x_to_y_equals_one (x y : ℝ) (h : Real.sqrt (x + 1) + |y - 2| = 0) : x^y = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_to_y_equals_one_l486_48666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l486_48645

noncomputable section

-- Define the parabola E: x^2 = 2py (p > 0)
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y ∧ p > 0

-- Define the line l: y = kx + p/2
def line (k p : ℝ) (x y : ℝ) : Prop := y = k*x + p/2

-- Define the distance between two points
def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Define the slope of a line through two points
def lineSlope (x1 y1 x2 y2 : ℝ) : ℝ := (y2 - y1) / (x2 - x1)

-- Main theorem
theorem parabola_intersection_theorem (p k : ℝ) :
  (∃ x1 y1 x2 y2 : ℝ, 
    parabola p x1 y1 ∧ 
    parabola p x2 y2 ∧ 
    line 1 p x1 y1 ∧ 
    line 1 p x2 y2 ∧ 
    distance x1 y1 x2 y2 = 8) →
  (∃ xp yp xf yf : ℝ,
    parabola p xf yf ∧  -- F is on the parabola
    (∀ x y : ℝ, parabola p x y → (x - xf)^2 + (y - yf)^2 ≤ (x - xp)^2 + (y - yp)^2) ∧  -- F is the focus
    lineSlope xp yp xf yf + k = -3/2) →
  p = 2 ∧ (k = -2 ∨ k = 1/2) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l486_48645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a4_l486_48609

def satisfies_conditions (a : Fin 10 → ℕ+) : Prop :=
  (∀ i j, i ≠ j → a i ≠ a j) ∧
  a 2 = a 1 + a 5 ∧
  a 3 = a 2 + a 6 ∧
  a 4 = a 3 + a 7 ∧
  a 6 = a 5 + a 8 ∧
  a 7 = a 6 + a 9 ∧
  a 9 = a 8 + a 10

theorem smallest_a4 (a : Fin 10 → ℕ+) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h2 : a 2 = a 1 + a 5)
  (h3 : a 3 = a 2 + a 6)
  (h4 : a 4 = a 3 + a 7)
  (h6 : a 6 = a 5 + a 8)
  (h7 : a 7 = a 6 + a 9)
  (h9 : a 9 = a 8 + a 10) :
  20 ≤ a 4 ∧ ∃ a', satisfies_conditions a' ∧ a' 4 = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a4_l486_48609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_for_doubling_in_20_years_l486_48652

/-- Simple interest calculation function -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ := (principal * rate * time) / 100

/-- The problem statement -/
theorem interest_rate_for_doubling_in_20_years :
  ∀ (principal : ℝ), principal > 0 →
  ∃ (rate : ℝ), 
    simple_interest principal rate 20 = principal ∧ 
    rate = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_for_doubling_in_20_years_l486_48652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triple_sum_eq_one_over_1771_l486_48632

/-- The sum of 1 / (2^a * 4^b * 6^c) over all positive integer triples (a,b,c) where 1 ≤ a < b < c -/
noncomputable def triple_sum : ℝ :=
  ∑' (a : ℕ), ∑' (b : ℕ), ∑' (c : ℕ), 
    if 1 ≤ a ∧ a < b ∧ b < c then 1 / ((2:ℝ)^a * 4^b * 6^c) else 0

/-- The theorem stating that the triple sum equals 1/1771 -/
theorem triple_sum_eq_one_over_1771 : triple_sum = 1 / 1771 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triple_sum_eq_one_over_1771_l486_48632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_14_solutions_l486_48665

/-- The number of real solutions to the equation x/50 = cos(x) -/
def num_solutions : ℕ := 14

/-- The equation we're considering -/
def equation (x : ℝ) : Prop := x / 50 = Real.cos x

/-- The set of solutions to the equation -/
def solution_set : Set ℝ := {x : ℝ | equation x}

theorem equation_has_14_solutions :
  ∃ (S : Set ℝ), (∀ x ∈ S, equation x) ∧ (Finite S) ∧ (Nat.card S = num_solutions) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_14_solutions_l486_48665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distances_condition_l486_48614

-- Define the line l: x + y + 1 = 0
def line_l (x y : ℝ) : Prop := x + y + 1 = 0

-- Define the distance formula from a point to a line
noncomputable def distance_to_line (x y : ℝ) : ℝ := |x + y + 1| / Real.sqrt 2

-- Define the points A and B
def point_A (a : ℝ) : ℝ × ℝ := (a, 1)
def point_B : ℝ × ℝ := (4, 8)

-- State the theorem
theorem equal_distances_condition (a : ℝ) : 
  distance_to_line (point_A a).1 (point_A a).2 = distance_to_line point_B.1 point_B.2 ↔ 
  a = 11 ∨ a = -15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distances_condition_l486_48614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mian_number_l486_48618

-- Define the property of being irrational (cannot be expressed as an exact square root)
def IsIrrational (x : ℝ) : Prop := ∀ (a b : ℤ), b ≠ 0 → x ≠ (a : ℝ) / (b : ℝ)

-- Define the numbers we're considering
noncomputable def n1 : ℝ := Real.sqrt 3
noncomputable def n2 : ℝ := Real.sqrt 4
noncomputable def n3 : ℝ := Real.sqrt 9
noncomputable def n4 : ℝ := Real.sqrt 16

-- State the theorem
theorem mian_number :
  IsIrrational n1 ∧ ¬IsIrrational n2 ∧ ¬IsIrrational n3 ∧ ¬IsIrrational n4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mian_number_l486_48618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l486_48690

-- Define the function (marked as noncomputable due to dependency on Real.sqrt)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) + 1 / (2 - x)

-- Define the domain
def domain : Set ℝ := Set.Ici (-1) ∪ Set.Ioi 2

-- Theorem statement
theorem f_domain : 
  {x : ℝ | x + 1 ≥ 0 ∧ 2 - x ≠ 0} = domain := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l486_48690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lshaped_room_ratio_l486_48607

/-- Represents the dimensions of an L-shaped room --/
structure LShapedRoom where
  rectangleLength : ℚ
  rectangleWidth : ℚ
  squareSide : ℚ

/-- Calculates the ratio of total length to perimeter for an L-shaped room --/
def lengthToPerimeterRatio (room : LShapedRoom) : ℚ :=
  let totalLength := room.rectangleLength
  let totalWidth := room.rectangleWidth + room.squareSide
  let perimeter := 2 * (totalLength + totalWidth)
  totalLength / perimeter

/-- Theorem stating that for the given L-shaped room, the ratio of total length to perimeter is 1/4 --/
theorem lshaped_room_ratio :
  let room := LShapedRoom.mk 23 15 8
  lengthToPerimeterRatio room = 1/4 := by
  sorry

#eval lengthToPerimeterRatio (LShapedRoom.mk 23 15 8)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lshaped_room_ratio_l486_48607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l486_48678

theorem solution_set (x : ℝ) : x^2 - x ≤ 4 ∧ x > 1 - 2*x → x ∈ Set.Ioo (1/3) 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l486_48678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_roots_of_f_l486_48673

/-- The polynomial function f(x) = (x-3)(x^2 + 2x + 1)(x^3 - x + 2) -/
def f (x : ℝ) : ℝ := (x - 3) * (x^2 + 2*x + 1) * (x^3 - x + 2)

/-- The number of distinct real roots of f(x) = 0 -/
def num_distinct_roots : ℕ := 2

theorem distinct_roots_of_f :
  ∃ (S : Finset ℝ), S.card = num_distinct_roots ∧ 
  (∀ x ∈ S, f x = 0) ∧
  (∀ x : ℝ, f x = 0 → x ∈ S) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_roots_of_f_l486_48673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_function_l486_48613

theorem min_value_of_function :
  (∀ x : ℝ, -Real.cos x ^ 2 + 2 * Real.sin x + 2 ≥ 0) ∧
  (∃ x : ℝ, -Real.cos x ^ 2 + 2 * Real.sin x + 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_function_l486_48613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_average_speed_l486_48682

/-- Represents the speed and distance characteristics of a train journey segment -/
structure Segment where
  distance : ℝ
  speed : ℝ

/-- Calculates the time taken for a segment -/
noncomputable def segmentTime (s : Segment) : ℝ := s.distance / s.speed

/-- Represents the entire train journey -/
noncomputable def TrainJourney (S D : ℝ) : List Segment :=
  [ { distance := D,     speed := S },
    { distance := 3*D,   speed := 2*S },
    { distance := D/2,   speed := 0.75*S },
    { distance := 1.5*D, speed := 0.8*S },
    { distance := 2.5*D, speed := 0.75*S },
    { distance := 4*D,   speed := 1.056*S },
    { distance := 1.2*D, speed := 0.5*S } ]

/-- Calculates the total distance of the journey -/
noncomputable def totalDistance (journey : List Segment) : ℝ :=
  (journey.map (λ s => s.distance)).sum

/-- Calculates the total time of the journey -/
noncomputable def totalTime (journey : List Segment) : ℝ :=
  (journey.map segmentTime).sum

/-- Theorem stating that the average speed of the train journey is approximately S / 1.002 -/
theorem train_journey_average_speed (S D : ℝ) (hS : S > 0) (hD : D > 0) :
  let journey := TrainJourney S D
  let avgSpeed := totalDistance journey / totalTime journey
  ∃ ε > 0, |avgSpeed - S / 1.002| < ε := by
  sorry

#eval "Compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_average_speed_l486_48682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chemical_transportation_l486_48616

/-- The amount of chemical raw materials a worker can carry per hour (in kg) -/
def worker_capacity : ℝ := 30

/-- The amount of chemical raw materials a robot can carry per hour (in kg) -/
def robot_capacity : ℝ := 450

/-- The number of additional workers needed -/
def additional_workers : ℕ := 15

/-- The total amount of chemical raw materials to be transported (in kg) -/
def total_materials : ℝ := 3600

/-- The number of robots involved in the transportation -/
def num_robots : ℕ := 3

/-- The time limit for the transportation (in hours) -/
def time_limit : ℝ := 2

theorem chemical_transportation :
  (robot_capacity = worker_capacity + 420) ∧
  (900 / robot_capacity = 600 / (10 * worker_capacity)) ∧
  (robot_capacity * ↑num_robots * time_limit + 
   worker_capacity * ↑additional_workers * time_limit ≥ total_materials) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chemical_transportation_l486_48616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_length_sum_l486_48601

/-- A triangle with vertices X, Y, Z -/
structure Triangle (X Y Z : ℝ × ℝ) : Prop where
  xy_length : dist X Y = 27
  yz_length : dist Y Z = 35
  xz_length : dist X Z = 30

/-- An inscribed triangle with vertices J, K, L inside triangle XYZ -/
structure InscribedTriangle (X Y Z J K L : ℝ × ℝ) : Prop where
  j_on_yz : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ J = (1 - t) • Y + t • Z
  k_on_xz : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ K = (1 - t) • X + t • Z
  l_on_xy : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ L = (1 - t) • X + t • Y

/-- The condition that certain arcs are equal -/
structure EqualArcs (X Y Z J K L : ℝ × ℝ) : Prop where
  yj_eq_lz : dist Y J = dist L Z
  xj_eq_ky : dist X J = dist K Y
  kx_eq_lj : dist K X = dist L J

/-- The main theorem -/
theorem inscribed_triangle_length_sum (X Y Z J K L : ℝ × ℝ)
  (tri : Triangle X Y Z)
  (ins : InscribedTriangle X Y Z J K L)
  (arcs : EqualArcs X Y Z J K L) :
  ∃ p q : ℕ, Nat.Coprime p q ∧ dist L J = p / q ∧ p + q = 29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_length_sum_l486_48601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandy_walks_36km_l486_48629

/-- The distance Sandy walks before meeting Ed -/
noncomputable def distance_sandy_walks (total_distance : ℝ) (sandy_speed : ℝ) (ed_speed : ℝ) (head_start : ℝ) : ℝ :=
  let initial_distance := sandy_speed * head_start
  let remaining_distance := total_distance - initial_distance
  let meeting_time := remaining_distance / (sandy_speed + ed_speed)
  initial_distance + sandy_speed * meeting_time

/-- Theorem stating that Sandy walks 36 km before meeting Ed -/
theorem sandy_walks_36km :
  distance_sandy_walks 52 6 4 2 = 36 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandy_walks_36km_l486_48629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friendship_number_assignment_exists_l486_48606

-- Define the assignment function type
def Assignment (V : Type) := V → ℕ+

-- Main theorem
theorem friendship_number_assignment_exists (V : Type) (G : SimpleGraph V) :
  ∃ (N : ℕ+) (f : Assignment V), ∀ (u v : V),
    (G.Adj u v) ↔ (N ∣ (f u * f v)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friendship_number_assignment_exists_l486_48606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_bijective_f_l486_48697

/-- A bijective function from ℕ to ℕ satisfying the given property -/
noncomputable def f : ℕ → ℕ := sorry

/-- The property that f must satisfy for all m and n -/
axiom f_property (m n : ℕ) : f (3 * m * n + m + n) = 4 * f m * f n + f m + f n

/-- f is injective -/
axiom f_injective : Function.Injective f

/-- f is surjective -/
axiom f_surjective : Function.Surjective f

/-- The main theorem stating the existence of f with the required properties -/
theorem exists_bijective_f : 
  ∃ (f : ℕ → ℕ), Function.Bijective f ∧ 
    (∀ m n : ℕ, f (3 * m * n + m + n) = 4 * f m * f n + f m + f n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_bijective_f_l486_48697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solutions_l486_48669

-- Define the function type
def RealFunction : Type := ℝ → ℝ

-- Define the functional equation
def SatisfiesEquation (f : RealFunction) : Prop :=
  ∀ x y : ℝ, f (x^2 + f x * f y) = x * f (x + y)

-- Define the possible solutions
def IsZeroFunction (f : RealFunction) : Prop :=
  ∀ x : ℝ, f x = 0

def IsIdentityFunction (f : RealFunction) : Prop :=
  ∀ x : ℝ, f x = x

def IsNegativeIdentityFunction (f : RealFunction) : Prop :=
  ∀ x : ℝ, f x = -x

-- State the theorem
theorem functional_equation_solutions (f : RealFunction) :
  SatisfiesEquation f → IsZeroFunction f ∨ IsIdentityFunction f ∨ IsNegativeIdentityFunction f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solutions_l486_48669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_sequence_properties_radius_of_2016th_circle_l486_48608

/-- Represents a circle in the Cartesian plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the sequence of circles inside the parabola -/
noncomputable def circle_sequence (a : ℝ) : ℕ → Circle
  | 0 => { center := (0, 1/(2*a)), radius := 1/(2*a) }
  | n+1 => sorry

/-- The parabola y = ax² -/
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2

/-- Defines what it means for a circle to be tangent to the parabola -/
def is_tangent_to_parabola (a : ℝ) (c : Circle) : Prop := sorry

/-- Defines what it means for two circles to be externally tangent -/
def is_externally_tangent (c1 c2 : Circle) : Prop := sorry

/-- Defines what it means for a circle to pass through the vertex of the parabola -/
def passes_through_vertex (a : ℝ) (c : Circle) : Prop := sorry

theorem circle_sequence_properties (a : ℝ) (h : a > 0) :
  let seq := circle_sequence a
  (∀ n : ℕ, (seq n).center.1 = 0) ∧ 
  (∀ n : ℕ, n > 0 → is_tangent_to_parabola a (seq n)) ∧
  (∀ n : ℕ, n > 0 → is_externally_tangent (seq n) (seq (n-1))) ∧
  ((seq 0).radius = 1/(2*a)) ∧
  passes_through_vertex a (seq 0) := by
  sorry

theorem radius_of_2016th_circle (a : ℝ) (h : a > 0) :
  (circle_sequence a 2015).radius = 4031 / (2*a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_sequence_properties_radius_of_2016th_circle_l486_48608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_men_required_theorem_l486_48693

/-- Calculates the number of men required for a new job given the parameters of an initial job and the new job. -/
def men_required_for_new_job (
  initial_men : ℕ
  ) (initial_days : ℕ
  ) (initial_hours_per_day : ℕ
  ) (new_job_size_multiplier : ℕ
  ) (new_days : ℕ
  ) (new_hours_per_day : ℕ
  ) (productivity_ratio : ℚ
  ) : ℕ :=
  let initial_man_hours := initial_men * initial_days * initial_hours_per_day
  let new_man_hours := new_job_size_multiplier * initial_man_hours
  let adjusted_man_hours := (new_man_hours : ℚ) / productivity_ratio
  let hours_per_man := new_days * new_hours_per_day
  Int.natAbs (Int.ceil (adjusted_man_hours / hours_per_man))

/-- Theorem stating that given the specified conditions, 600 men are required for the new job. -/
theorem men_required_theorem :
  men_required_for_new_job 250 16 8 3 20 10 (4/5) = 600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_men_required_theorem_l486_48693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_wins_l486_48615

/-- Represents a runner in the race -/
structure Runner where
  speed : ℚ

/-- Represents the race scenario -/
structure RaceScenario where
  petya : Runner
  vasya : Runner
  race_distance : ℚ
  initial_gap : ℚ

/-- The given race scenario based on the problem conditions -/
def given_scenario : RaceScenario where
  petya := ⟨60⟩
  vasya := ⟨51⟩
  race_distance := 60
  initial_gap := 9

/-- Calculates the final difference between Petya and Vasya in the second race -/
def final_difference (scenario : RaceScenario) : ℚ :=
  scenario.race_distance - (scenario.race_distance * scenario.vasya.speed / scenario.petya.speed + scenario.initial_gap * scenario.vasya.speed / scenario.petya.speed)

/-- Theorem stating that Petya finishes 1.35 meters ahead of Vasya in the second race -/
theorem petya_wins (ε : ℚ) (h : ε > 0) : 
  |final_difference given_scenario - 135/100| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_wins_l486_48615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cube_volume_ratio_l486_48628

/-- The volume of a regular tetrahedron with edge length a -/
noncomputable def tetrahedron_volume (a : ℝ) : ℝ := (Real.sqrt 2 * a^3) / 12

/-- The volume of a cube with edge length a -/
def cube_volume (a : ℝ) : ℝ := a^3

theorem truncated_cube_volume_ratio :
  let cube_edge_length : ℝ := 2
  let tetrahedron_edge_length : ℝ := 1
  let num_vertices : ℕ := 8
  let original_volume := cube_volume cube_edge_length
  let removed_volume := num_vertices * tetrahedron_volume tetrahedron_edge_length
  let truncated_volume := original_volume - removed_volume
  truncated_volume / original_volume = 5/6 := by
  sorry

#eval (5:ℚ) / 6 -- This will output the rational number 5/6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cube_volume_ratio_l486_48628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_l486_48625

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Define the point of tangency
def p : ℝ × ℝ := (1, 0)

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := x - y - 1 = 0

-- Theorem statement
theorem tangent_line_at_point :
  ∀ x y : ℝ, tangent_line x y ↔ 
  (∃ m : ℝ, HasDerivAt f m p.1 ∧ y - p.2 = m * (x - p.1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_l486_48625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_larger_set_l486_48634

theorem gcd_of_larger_set (n : ℕ+) (S T : Finset ℕ) (d : ℕ) :
  S.Nonempty →
  S ⊆ Finset.range n →
  (Finset.gcd S id ≠ 1) →
  d > 1 →
  d = Nat.minFac (Finset.gcd S id) →
  S ⊆ T →
  T ⊆ Finset.range n →
  T.card ≥ 1 + n / d →
  Finset.gcd T id = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_larger_set_l486_48634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l486_48653

/-- The function f(x) defined for x > 0 -/
noncomputable def f (x : ℝ) : ℝ := (-x^2 + x - 4) / x

/-- Theorem stating the maximum value of f(x) and where it occurs -/
theorem f_max_value :
  (∀ x > 0, f x ≤ -3) ∧ 
  (f 2 = -3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l486_48653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_min_value_part2_l486_48676

-- Define the function f
def f (x a b : ℝ) : ℝ := |x + a| + |x - b|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f x 1 1 ≤ 4} = {x : ℝ | -2 ≤ x ∧ x ≤ 2} := by
  sorry

-- Part 2
theorem min_value_part2 (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hmin : ∀ x, f x a b ≥ 2) (hexists : ∃ x, f x a b = 2) :
  (∀ x : ℝ, 1/a + 2/b ≥ 3) ∧ (∃ x : ℝ, 1/a + 2/b = 3) := by
  sorry

#check solution_set_part1
#check min_value_part2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_min_value_part2_l486_48676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_A_l486_48691

-- Define the points
def A : ℝ × ℝ := (0, 10)
def B : ℝ × ℝ := (0, 15)
def C : ℝ × ℝ := (3, 9)

-- Define the line y = x
def line_y_eq_x (p : ℝ × ℝ) : Prop := p.2 = p.1

-- Define A' and B' as points on the line y = x
def A' : ℝ × ℝ := (7.5, 7.5)
def B' : ℝ × ℝ := (5, 5)

-- Define the property that AA' and BB' intersect at C
def intersect_at_C (p q : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
  C = (A.1 + t * (p.1 - A.1), A.2 + t * (p.2 - A.2)) ∧
  ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ 
  C = (B.1 + s * (q.1 - B.1), B.2 + s * (q.2 - B.2))

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- The theorem to prove
theorem length_of_A'B' :
  line_y_eq_x A' ∧ line_y_eq_x B' ∧ intersect_at_C A' B' →
  distance A' B' = 2.5 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_A_l486_48691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_specific_l486_48699

-- Define the pyramid
structure Pyramid where
  base_length : ℝ
  base_width : ℝ
  edge_length : ℝ

-- Define the volume calculation function
noncomputable def pyramidVolume (p : Pyramid) : ℝ := 
  let base_area := (1/2) * p.base_length * p.base_width
  let height := Real.sqrt (p.edge_length^2 - (1/4) * (p.base_length^2 + p.base_width^2))
  (1/3) * base_area * height

-- Theorem statement
theorem pyramid_volume_specific : 
  let p : Pyramid := { base_length := 5, base_width := 12, edge_length := 15 }
  pyramidVolume p = 10 * Real.sqrt 182.75 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_specific_l486_48699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closed_set_properties_l486_48643

-- Define a closed set
def ClosedSet {α : Type*} [Add α] (A : Set α) : Prop :=
  ∀ a b : α, a ∈ A → b ∈ A → (a + b) ∈ A

-- Define the set of positive integers
def PositiveIntegers : Set ℤ :=
  {n : ℤ | n > 0}

-- Define the set of multiples of 3
def MultiplesOfThree : Set ℤ :=
  {n : ℤ | ∃ k : ℤ, n = 3 * k}

theorem closed_set_properties :
  (ClosedSet PositiveIntegers) ∧
  (ClosedSet MultiplesOfThree) ∧
  (∀ A₁ A₂ : Set ℝ, ClosedSet A₁ → ClosedSet A₂ → ∃ c : ℝ, c ∉ (A₁ ∪ A₂)) :=
by
  sorry

#check closed_set_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closed_set_properties_l486_48643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l486_48622

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem unique_solution :
  ∃! x : ℕ, ∃ p : ℕ, is_prime p ∧ 2^x + x^2 + 25 = p^3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l486_48622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_tangent_to_x_axis_l486_48680

/-- The quadratic function g(x) -/
noncomputable def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + ((-6)^2 / (4 * 3))

/-- The discriminant of g(x) -/
noncomputable def discriminant : ℝ := (-6)^2 - 4 * 3 * ((-6)^2 / (4 * 3))

/-- The x-coordinate of the vertex of g(x) -/
noncomputable def vertex_x : ℝ := -(-6) / (2 * 3)

/-- Theorem: The graph of y = g(x) is tangent to the x-axis -/
theorem graph_tangent_to_x_axis : 
  discriminant = 0 ∧ g vertex_x = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_tangent_to_x_axis_l486_48680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_K_l486_48640

/-- The radius of the larger circle D -/
def R : ℝ := 40

/-- The number of smaller circles in the ring -/
def n : ℕ := 8

/-- The area of the region inside circle D and outside all n smaller circles -/
noncomputable def K' (r : ℝ) : ℝ := Real.pi * R^2 - n * Real.pi * r^2

/-- Theorem stating that the floor of K' is 12320 -/
theorem floor_K'_eq_12320 :
  ∃ r : ℝ, r > 0 ∧ 3 * r = R ∧ ⌊K' r⌋ = 12320 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_K_l486_48640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_comparison_l486_48639

theorem sin_comparison : Real.sin 3 < Real.sin 1 ∧ Real.sin 1 < Real.sin 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_comparison_l486_48639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lamp_post_break_height_is_sqrt_10_l486_48684

/-- The height of a broken lamp post -/
noncomputable def lamp_post_break_height (total_height : ℝ) (distance_to_ground : ℝ) : ℝ :=
  Real.sqrt ((total_height ^ 2 + distance_to_ground ^ 2) / 4)

/-- Theorem: A 6-meter lamp post breaking with its top touching 2 meters away breaks at √10 meters -/
theorem lamp_post_break_height_is_sqrt_10 :
  lamp_post_break_height 6 2 = Real.sqrt 10 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lamp_post_break_height_is_sqrt_10_l486_48684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l486_48692

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 2 → x₂ ∈ Set.Icc 0 2 → |f m x₂ - f m x₁| ≤ 9) →
  m ∈ Set.Icc (-5/2) (13/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l486_48692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_ratio_l486_48688

theorem triangle_sine_ratio (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC with sides a, b, c and angles A, B, C
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧
  -- Sides are consecutive integers
  ((b = a + 1 ∧ c = b + 1) ∨ (a = b + 1 ∧ b = c + 1) ∨ (c = a + 1 ∧ a = b + 1)) ∧
  -- A > B > C
  (A > B) ∧ (B > C) ∧
  -- 3b = 20a*cos(A)
  (3*b = 20*a*(Real.cos A)) ∧
  -- Law of sines
  (a/(Real.sin A) = b/(Real.sin B)) ∧ (b/(Real.sin B) = c/(Real.sin C)) →
  -- Conclusion: sin(A) : sin(B) : sin(C) = 6 : 5 : 4
  ∃ (k : ℝ), k > 0 ∧ Real.sin A = 6*k ∧ Real.sin B = 5*k ∧ Real.sin C = 4*k :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_ratio_l486_48688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_subset_singleton_zero_l486_48670

theorem empty_subset_singleton_zero : ∅ ⊆ ({0} : Set ℕ) := by
  simp

end NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_subset_singleton_zero_l486_48670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_equals_one_l486_48600

/-- The set of all nonzero real numbers -/
def S : Set ℝ := {x : ℝ | x ≠ 0}

/-- The main theorem -/
theorem f_one_equals_one (k : ℝ) (hk : k ≠ 0) (f : ℝ → ℝ)
  (hf : ∀ x, x ≠ 0 → f x ≠ 0)
  (h1 : ∀ x, x ≠ 0 → f (1 / x) = Real.cos (k * x) * x * f x)
  (h2 : ∀ x y, x ≠ 0 → y ≠ 0 → x + y ≠ 0 → f (1 / x) + f (1 / y) = 1 + f (1 / (x + y))) :
  f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_equals_one_l486_48600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_relationships_l486_48642

/-- Given system of equations -/
theorem system (x y : ℝ) : Prop :=
  x^3 - y^5 = 2882 ∧ x - y = 2

/-- Known solutions -/
noncomputable def x₁ : ℝ := 5
noncomputable def y₁ : ℝ := 3
noncomputable def x₃ : ℝ := 1 + 3 * Real.rpow (-2) (1/3)

/-- Theorem stating the relationships between solutions -/
theorem solution_relationships :
  ∃ (x₂ y₂ y₃ x₄ y₄ : ℝ),
    (x₂ = -y₁ ∧ y₂ = -x₁) ∧
    y₃ = -x₃ ∧
    (x₄ = -y₃ ∧ y₄ = -x₃) ∧
    system x₁ y₁ ∧
    system x₂ y₂ ∧
    system x₃ y₃ ∧
    system x₄ y₄ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_relationships_l486_48642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_sequence_theorem_l486_48627

/-- The modified sequence of odd integers where each odd positive integer k appears k+2 times -/
def b : ℕ → ℕ := sorry

/-- The main theorem -/
theorem modified_sequence_theorem :
  ∃ (p q r : ℤ), 
    (∀ n : ℕ, b n = p * Int.floor (Real.sqrt (n + q : ℝ)) + r) ∧
    (p + q + r = 2) :=
by
  sorry

#check modified_sequence_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_sequence_theorem_l486_48627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_power_of_two_l486_48671

theorem perfect_square_power_of_two (n : ℕ) : 2^n + 1 = m^2 ↔ n = 3 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_power_of_two_l486_48671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_in_interval_l486_48689

noncomputable def f (x : ℝ) : ℝ := x^2 - 2/x

theorem zero_point_in_interval :
  ∃ (c : ℝ), c ∈ Set.Ioo (5/4 : ℝ) (3/2 : ℝ) ∧ f c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_in_interval_l486_48689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_completion_theorem_combined_work_rate_l486_48621

/-- The time it takes for two people to complete a project together -/
noncomputable def project_completion_time (x y : ℝ) : ℝ :=
  (x * y) / (x + y)

/-- Theorem: The time it takes for two people to complete a project together
    is (xy)/(x+y) days, given their individual completion times x and y. -/
theorem project_completion_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  project_completion_time x y = (x * y) / (x + y) :=
by
  -- Unfold the definition of project_completion_time
  unfold project_completion_time
  -- The equality holds by definition
  rfl

/-- Proof that the combined work rate is the sum of individual work rates -/
theorem combined_work_rate (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (1 : ℝ) / project_completion_time x y = 1 / x + 1 / y :=
by
  sorry -- We'll skip the detailed proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_completion_theorem_combined_work_rate_l486_48621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l486_48662

noncomputable def point := ℝ × ℝ

noncomputable def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_between_points : distance (0, 12) (5, 0) = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l486_48662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt_solutions_l486_48603

/-- Defines the nested square root function with n levels -/
noncomputable def nestedSqrt (x : ℝ) : ℕ → ℝ
  | 0 => Real.sqrt (3 * x)
  | n + 1 => Real.sqrt (x + 2 * nestedSqrt x n)

/-- The theorem stating that 0 and 3 are the only real solutions to the nested square root equation -/
theorem nested_sqrt_solutions (n : ℕ) :
  ∀ x : ℝ, nestedSqrt x n = x ↔ x = 0 ∨ x = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt_solutions_l486_48603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l486_48660

theorem area_between_curves : ∫ (x : ℝ) in Set.Icc 0 1, (x - x^3) = 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l486_48660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_simplification_and_sum_l486_48604

noncomputable def f (x : ℝ) : ℝ := (x^3 + 5*x^2 + 8*x + 4) / (x + 1)

def g (x : ℝ) : ℝ := x^2 + 4*x + 4

def D : ℝ := -1

theorem function_simplification_and_sum :
  (∀ x : ℝ, x ≠ D → f x = g x) ∧
  (1 + 4 + 4 + D = 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_simplification_and_sum_l486_48604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_formation_possibility_l486_48648

/-- Represents a geometric shape -/
structure Shape where
  area : ℝ

/-- Represents a square -/
structure Square extends Shape

/-- Represents a set of shapes that can form a square -/
def CanFormSquare (shapes : List Shape) : Prop :=
  ∃ (s : Square), s.area = (shapes.map Shape.area).sum

theorem square_formation_possibility
  (vika_shapes : List Shape)
  (alina_square polina_square : Square)
  (h1 : vika_shapes.length = 4)
  (h2 : alina_square.area ≠ polina_square.area)
  (h3 : CanFormSquare (alina_square.toShape :: vika_shapes)) :
  ∃ (new_square : Square),
    new_square.area = polina_square.area + (vika_shapes.map Shape.area).sum := by
  sorry

#check square_formation_possibility

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_formation_possibility_l486_48648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l486_48617

/-- The area of a rectangle bounded by specific lines -/
theorem rectangle_area (a b c d m k : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (hm : m > 0) (hk : k > 0) 
  (h_km : k * m ≠ 1) :
  ∃ A : ℝ, A = (4 * m * c * k * a) / ((1 - k * m)^2) ∧
    A = (2 * m * c / (1 - k * m)) * (2 * k * a / (1 - k * m)) :=
by
  -- Define the lines
  let line1 := fun x => m * x + a
  let line2 := fun x => m * x - b
  let line3 := fun y => k * y + c
  let line4 := fun y => k * y - d
  
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l486_48617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_process_time_l486_48637

/-- Represents the candy-making process with given temperatures and rates -/
structure CandyProcess where
  initial_temp : ℝ
  target_temp : ℝ
  heating_rate1 : ℝ
  heating_rate2 : ℝ
  stirring_time : ℝ
  cooling_target : ℝ
  cooling_rate1 : ℝ
  cooling_rate2 : ℝ

/-- Calculates the total time for the candy-making process -/
noncomputable def total_time (p : CandyProcess) : ℝ :=
  let heating_time1 := 120 / p.heating_rate1
  let heating_time2 := (p.target_temp - (p.initial_temp + 120)) / p.heating_rate2
  let cooling_time1 := 40 / p.cooling_rate1
  let cooling_time2 := (p.target_temp - p.cooling_target - 40) / p.cooling_rate2
  heating_time1 + heating_time2 + p.stirring_time + cooling_time1 + cooling_time2

/-- Theorem stating that the total time for the given process is approximately 54.71 minutes -/
theorem candy_process_time : 
  let p : CandyProcess := {
    initial_temp := 60
    target_temp := 240
    heating_rate1 := 5
    heating_rate2 := 8
    stirring_time := 10
    cooling_target := 170
    cooling_rate1 := 7
    cooling_rate2 := 4
  }
  abs (total_time p - 54.71) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_process_time_l486_48637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l486_48656

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x + 4)) / x

-- State the theorem
theorem f_domain : 
  {x : ℝ | f x = f x} = {x : ℝ | x ≥ -4 ∧ x ≠ 0} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l486_48656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_weight_l486_48654

/-- The weight of a round weight -/
def w_round : ℝ := sorry

/-- The weight of a triangular weight -/
def w_triangular : ℝ := sorry

/-- The weight of the rectangular weight -/
def w_rectangular : ℝ := 90

/-- First balance equation -/
axiom balance1 : w_round + w_triangular = 3 * w_round

/-- Second balance equation -/
axiom balance2 : 4 * w_round + w_triangular = w_triangular + w_round + w_rectangular

/-- Theorem: The weight of a triangular weight is 60 grams -/
theorem triangular_weight : w_triangular = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_weight_l486_48654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l486_48655

/-- The sum of the infinite series ∑(n=1 to ∞) (3n+2)/(n(n+1)(n+3)) -/
noncomputable def infiniteSeries : ℝ := ∑' n : ℕ+, (3 * n + 2) / (n * (n + 1) * (n + 3))

/-- Theorem stating that the infinite series sums to 5/6 -/
theorem infiniteSeriesSum : infiniteSeries = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l486_48655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sum_parity_difference_l486_48633

/-- The number of positive integer divisors of n, including 1 and n -/
def τ (n : ℕ+) : ℕ := sorry

/-- The sum of τ(k) for k from 1 to n -/
def S (n : ℕ+) : ℕ := sorry

/-- The number of positive integers n ≤ 3000 with S(n) odd -/
def c : ℕ := sorry

/-- The number of positive integers n ≤ 3000 with S(n) even -/
def d : ℕ := sorry

theorem divisor_sum_parity_difference : |Int.ofNat c - Int.ofNat d| = 1733 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sum_parity_difference_l486_48633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_vector_combination_l486_48668

def a : ℝ × ℝ × ℝ := (3, 5, 1)
def b : ℝ × ℝ × ℝ := (2, 2, 3)

theorem magnitude_of_vector_combination : Real.sqrt ((2 * a.1 - 3 * b.1)^2 + (2 * a.2.1 - 3 * b.2.1)^2 + (2 * a.2.2 - 3 * b.2.2)^2) = Real.sqrt 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_vector_combination_l486_48668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_simplification_A_special_value_l486_48685

-- Define the expression A
noncomputable def A (x : ℝ) : ℝ := (x^2 - 1) / (x^2 - 2*x + 1) / ((x + 1) / x) + 1 / (x - 1)

-- Define the special value of x
noncomputable def special_x : ℝ := (Real.sqrt 12 - Real.sqrt (4/3)) * Real.sqrt 3

-- Theorem 1: Simplification of A
theorem A_simplification (x : ℝ) (h : x ≠ 1) : A x = (x + 1) / (x - 1) := by
  sorry

-- Theorem 2: Value of A for the special x
theorem A_special_value : A special_x = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_simplification_A_special_value_l486_48685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_commutative_star_not_associative_l486_48626

/-- Operation ∗ for positive real numbers -/
noncomputable def star (k : ℝ) (x y : ℝ) : ℝ := (x * y + k) / (x + y + k)

/-- Commutativity of ∗ operation -/
theorem star_commutative (k : ℝ) (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  star k x y = star k y x := by
  -- Proof of commutativity
  sorry

/-- Non-associativity of ∗ operation -/
theorem star_not_associative :
  ∃ (k x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0),
    star k (star k x y) z ≠ star k x (star k y z) := by
  -- Proof of non-associativity
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_commutative_star_not_associative_l486_48626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_ellipse_l486_48694

/-- The locus of points M(x,y) satisfying the given conditions -/
def locus (x y : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ),
    (x₀^2 + y₀^2 = 1) ∧  -- P is on the circle
    (x = x₀) ∧  -- PP₀ is perpendicular to x-axis
    (y = Real.sqrt 3 / 2 * y₀)  -- Vector MP₀ = (√3/2) * Vector PP₀

/-- The theorem stating that the locus is an ellipse -/
theorem locus_is_ellipse :
  ∀ (x y : ℝ), locus x y ↔ x^2 + y^2 / (3/4) = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_ellipse_l486_48694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_center_sum_l486_48675

-- Define the square ABCD
structure Square where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define the condition that the square lies in the first quadrant
def inFirstQuadrant (s : Square) : Prop :=
  s.A.1 ≥ 0 ∧ s.A.2 ≥ 0 ∧
  s.B.1 ≥ 0 ∧ s.B.2 ≥ 0 ∧
  s.C.1 ≥ 0 ∧ s.C.2 ≥ 0 ∧
  s.D.1 ≥ 0 ∧ s.D.2 ≥ 0

-- Define the condition that points lie on the sides of the square
def pointsOnSides (s : Square) : Prop :=
  ∃ (t₁ t₂ t₃ t₄ : ℝ),
    0 ≤ t₁ ∧ t₁ ≤ 1 ∧
    0 ≤ t₂ ∧ t₂ ≤ 1 ∧
    0 ≤ t₃ ∧ t₃ ≤ 1 ∧
    0 ≤ t₄ ∧ t₄ ≤ 1 ∧
    (1 - t₁) * s.A.1 + t₁ * s.D.1 = 4 ∧
    (1 - t₂) * s.B.1 + t₂ * s.C.1 = 6 ∧
    (1 - t₃) * s.A.1 + t₃ * s.B.1 = 9 ∧
    (1 - t₄) * s.C.1 + t₄ * s.D.1 = 14

-- Define the center of the square
noncomputable def center (s : Square) : ℝ × ℝ :=
  ((s.A.1 + s.B.1 + s.C.1 + s.D.1) / 4, (s.A.2 + s.B.2 + s.C.2 + s.D.2) / 4)

-- Theorem statement
theorem square_center_sum (s : Square) 
  (h1 : inFirstQuadrant s) 
  (h2 : pointsOnSides s) : 
  (center s).1 + (center s).2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_center_sum_l486_48675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_combinations_l486_48620

/-- Represents the types of food items available --/
inductive FoodItem
  | Sandwich
  | Roll
  | Pastry
deriving DecidableEq

/-- Represents the types of drink items available --/
inductive DrinkItem
  | JuicePack
  | SmallSoda
  | LargeSoda
deriving DecidableEq

/-- The price of a food item --/
def foodPrice (item : FoodItem) : Rat :=
  match item with
  | .Sandwich => 4/5
  | .Roll => 3/5
  | .Pastry => 1

/-- The price of a drink item --/
def drinkPrice (item : DrinkItem) : Rat :=
  match item with
  | .JuicePack => 1/2
  | .SmallSoda => 3/4
  | .LargeSoda => 5/4

/-- A combination of food and drink items --/
structure Combination :=
  (food : FoodItem)
  (drink : DrinkItem)
deriving DecidableEq

/-- The total budget available --/
def budget : Rat := 25/2

/-- The minimum amount to be spent on food --/
def minFoodSpend : Rat := 10

/-- Theorem: The maximum number of unique combinations possible within the given constraints is 5 --/
theorem max_combinations : 
  ∃ (combinations : List Combination),
    (combinations.length = 5) ∧ 
    (combinations.toFinset.card = 5) ∧
    (combinations.map (λ c => foodPrice c.food)).sum ≥ minFoodSpend ∧
    (combinations.map (λ c => foodPrice c.food + drinkPrice c.drink)).sum ≤ budget ∧
    ¬∃ (extraCombination : Combination),
      (extraCombination ∉ combinations.toFinset) ∧
      ((combinations.map (λ c => foodPrice c.food)).sum + foodPrice extraCombination.food ≥ minFoodSpend) ∧
      ((combinations.map (λ c => foodPrice c.food + drinkPrice c.drink)).sum + 
        foodPrice extraCombination.food + drinkPrice extraCombination.drink ≤ budget) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_combinations_l486_48620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_range_m_l486_48698

theorem sine_range_m (x : ℝ) (m : ℝ) (h : Real.sin x = m - 1) :
  0 ≤ m ∧ m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_range_m_l486_48698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_foci_distances_l486_48644

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/9 + y^2/4 = 1

-- Define the distance function
noncomputable def dist (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- State the theorem
theorem max_product_foci_distances (x y xF₁ yF₁ xF₂ yF₂ : ℝ) :
  ellipse x y →
  (∀ x' y', ellipse x' y' → dist x' y' xF₁ yF₁ + dist x' y' xF₂ yF₂ = 6) →
  dist x y xF₁ yF₁ * dist x y xF₂ yF₂ ≤ 9 ∧
  ∃ x' y', ellipse x' y' ∧ dist x' y' xF₁ yF₁ * dist x' y' xF₂ yF₂ = 9 := by
  sorry

#check max_product_foci_distances

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_foci_distances_l486_48644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_family_age_calculation_l486_48651

/-- Represents the average age of grandparents in a family -/
def average_age_grandparents : ℝ → Prop := fun x => True

/-- Theorem stating the average age of grandparents in a specific family scenario -/
theorem family_age_calculation :
  ∀ (x : ℝ),
  average_age_grandparents x →
  (2 * x + 2 * 39 + 3 * 6) / 7 = 32 →
  x = 64 :=
by
  intro x h1 h2
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_family_age_calculation_l486_48651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_interior_angle_measure_l486_48635

/-- The measure of each interior angle of a regular n-sided polygon -/
noncomputable def interior_angle_measure (n : ℕ) : ℝ :=
  (n - 2 : ℝ) * 180 / n

/-- Theorem: The measure of each interior angle of a regular n-sided polygon is (n-2) * 180° / n -/
theorem regular_polygon_interior_angle_measure (n : ℕ) (h : n ≥ 3) :
  interior_angle_measure n = (n - 2 : ℝ) * 180 / n := by
  -- Unfold the definition of interior_angle_measure
  unfold interior_angle_measure
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_interior_angle_measure_l486_48635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_PM_is_55_41_l486_48681

-- Define the parametric equations of line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + (3/5) * t, 1 + (4/5) * t)

-- Define the polar equation of curve C
noncomputable def curve_C (θ : ℝ) : ℝ := Real.sqrt (2 / (1 + Real.sin θ ^ 2))

-- Define point P
def point_P : ℝ × ℝ := (1, 1)

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem distance_PM_is_55_41 :
  ∃ t : ℝ, distance point_P (line_l t) = 55/41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_PM_is_55_41_l486_48681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_log_sum_l486_48610

theorem max_log_sum (a b : ℝ) (h1 : a > b) (h2 : b ≥ 2) :
  (∀ x y : ℝ, x > y ∧ y ≥ 2 → Real.log (x^2 / y) / Real.log x + Real.log (y^2 / x) / Real.log y ≤ 2) ∧
  (∃ x y : ℝ, x > y ∧ y ≥ 2 ∧ Real.log (x^2 / y) / Real.log x + Real.log (y^2 / x) / Real.log y = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_log_sum_l486_48610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_extension_l486_48631

-- Define the function f on the non-negative real numbers
noncomputable def f_nonneg (x : ℝ) : ℝ := x^2 * (1 - Real.sqrt x)

-- State the theorem
theorem odd_function_extension 
  (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x) 
  (h_nonneg : ∀ x ≥ 0, f x = f_nonneg x) :
  ∀ x < 0, f x = -((-x)^2 * (1 - Real.sqrt (-x))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_extension_l486_48631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_comparison_volume_ratio_l486_48641

/-- Given an inscribable isosceles triangle with circumradius r, 
    v₁ is the volume when rotated around its base,
    v₂ is the volume when rotated around a tangent through the vertex -/
noncomputable def v₁ (r : ℝ) : ℝ := (50 * Real.sqrt 5 / 81) * Real.pi * r^3
noncomputable def v₂ (r : ℝ) : ℝ := (100 * Real.sqrt 5 / 81) * Real.pi * r^3

/-- The volume v₂ is greater than v₁ for all positive real r -/
theorem volume_comparison (r : ℝ) (hr : r > 0) : v₂ r > v₁ r := by
  sorry

/-- The volume v₂ is exactly twice v₁ for all non-zero real r -/
theorem volume_ratio (r : ℝ) (hr : r ≠ 0) : v₂ r = 2 * v₁ r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_comparison_volume_ratio_l486_48641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l486_48695

/-- The distance between the foci of an ellipse with equation x²/36 + y²/9 = 5 is 2√5.4 -/
theorem ellipse_foci_distance : 
  ∀ (x y : ℝ), x^2/36 + y^2/9 = 5 → 
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (f₁ ∈ Set.univ) ∧ 
    (f₂ ∈ Set.univ) ∧ 
    (dist f₁ f₂ = 2 * Real.sqrt 5.4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l486_48695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_B_l486_48659

def B : Set ℕ := {n : ℕ | ∀ p, Nat.Prime p → p ∣ n → p ∈ ({2, 3, 5, 7} : Set ℕ)}

theorem sum_of_reciprocals_B : ∑' (n : B), (1 : ℚ) / (n : ℚ) = 105 / 16 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_B_l486_48659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_104_exists_l486_48687

def S : Set ℕ := {n | ∃ k : ℕ, k ≤ 33 ∧ n = 3 * k + 1}

theorem sum_104_exists (T : Finset ℕ) (h_subset : ↑T ⊆ S) (h_card : T.card = 20) :
  ∃ x y : ℕ, x ∈ T ∧ y ∈ T ∧ x ≠ y ∧ x + y = 104 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_104_exists_l486_48687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_K_value_l486_48658

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt (-x^2 + x + 2)

-- Define the function f_K(x)
noncomputable def f_K (K : ℝ) (x : ℝ) : ℝ :=
  if f x ≤ K then f x else K

-- Define the domain of f(x)
def domain_f : Set ℝ := {x : ℝ | -x^2 + x + 2 ≥ 0}

-- State the theorem
theorem min_K_value :
  ∀ K : ℝ, K > 0 →
  (∀ x ∈ domain_f, f_K K x = f x) →
  K ≥ 2 * Real.sqrt 2 :=
by
  sorry

#check min_K_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_K_value_l486_48658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_exists_l486_48679

theorem special_triangle_exists : ∃ (a b c : ℝ) (α β γ : ℝ),
  a = 4 ∧ b = 5 ∧ c = 6 ∧
  0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = Real.pi ∧
  γ = 2 * α ∧
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos α) ∧
  b^2 = a^2 + c^2 - 2*a*c*(Real.cos β) ∧
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos γ) ∧
  Real.cos γ = 1/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_exists_l486_48679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_25000_l486_48623

-- Define the expression
noncomputable def expression : ℝ := (10^(-1 : ℤ) * 5^2) / (10^(-4 : ℤ))

-- State the theorem
theorem expression_equals_25000 : expression = 25000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_25000_l486_48623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l486_48683

-- Define the function f(x)
noncomputable def f (k a x : ℝ) : ℝ := k * (a ^ x) - (a ^ (-x))

-- State the theorem
theorem function_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  -- f is an odd function
  (∀ x : ℝ, f 1 a (-x) = -f 1 a x) →
  -- k = 1
  (∃ k : ℝ, ∀ x : ℝ, f k a x = f 1 a x) ∧
  -- If a > 1, then f is strictly increasing on ℝ
  (a > 1 → ∀ x y : ℝ, x < y → f 1 a x < f 1 a y) ∧
  -- If a > 1, then the solution to f(x^2) + f(2x-1) < 0 is (-1-√2, -1+√2)
  (a > 1 → ∀ x : ℝ, f 1 a (x^2) + f 1 a (2*x-1) < 0 ↔ -1-Real.sqrt 2 < x ∧ x < -1+Real.sqrt 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l486_48683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l486_48650

open Real

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * (x + 1)
noncomputable def g (x : ℝ) : ℝ := x + Real.log x

-- Define the distance function between two points
def distance (m n : ℝ) : ℝ := |m - n|

-- Theorem statement
theorem min_distance_between_curves :
  ∃ (min_dist : ℝ), min_dist = 3/2 ∧
  ∀ (m n : ℝ), n > 0 → f m = g n →
  distance m n ≥ min_dist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l486_48650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_womans_train_speed_l486_48672

/-- Calculates the speed of a train given the parameters of an opposing train passing by --/
theorem womans_train_speed 
  (goods_train_speed : ℝ) 
  (goods_train_length : ℝ) 
  (passing_time : ℝ) : ℝ :=
by
  have h1 : goods_train_speed = 51.99424046076314 := by sorry
  have h2 : goods_train_length = 300 := by sorry
  have h3 : passing_time = 15 := by sorry
  
  let womans_train_speed := (goods_train_length / passing_time - goods_train_speed * 1000 / 3600) * 3600 / 1000
  
  have speed_close : abs (womans_train_speed - 20.038) < 0.001 := by sorry
  
  exact womans_train_speed


end NUMINAMATH_CALUDE_ERRORFEEDBACK_womans_train_speed_l486_48672
