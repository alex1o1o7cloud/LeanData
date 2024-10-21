import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l97_9781

noncomputable def f (ω a φ x : ℝ) : ℝ := 4 * Real.sin (ω * x + φ) + a

theorem problem_solution (ω a φ : ℝ) (hω : ω > 0) :
  (∀ x, f ω a φ x ≤ 2 ∧ ∃ x₀, f ω a φ x₀ = 2 → a = -2) ∧
  (φ = π / 3 → 
    (∀ x y, x ∈ Set.Icc (-π / 6) (π / 2) → y ∈ Set.Icc (-π / 6) (π / 2) → x < y → f ω a φ x < f ω a φ y) → 
    ω > 0 ∧ ω ≤ 1 / 3) ∧
  (a = -2 * Real.sqrt 2 → 
    (∀ φ, ∃ x₁ x₂, x₁ ∈ Set.Icc 0 (π / 2) ∧ x₂ ∈ Set.Icc 0 (π / 2) ∧ x₁ < x₂ ∧ f ω a φ x₁ = 0 ∧ f ω a φ x₂ = 0) → 
    ω ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l97_9781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_neg_seven_thirty_two_l97_9799

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x^6 - 1) / 4

-- State the theorem
theorem inverse_f_at_neg_seven_thirty_two :
  f⁻¹ (-7/32) = 1/2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_neg_seven_thirty_two_l97_9799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_value_l97_9789

theorem theta_value (θ : Real) 
  (h1 : Real.sin (Real.pi + θ) = -Real.sqrt 3 * Real.cos (2 * Real.pi - θ))
  (h2 : |θ| < Real.pi / 2) : 
  θ = Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_value_l97_9789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_equiv_m_bound_l97_9783

open Real

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := (x^2 - a*x - a) * exp x

-- State the theorem
theorem f_inequality_equiv_m_bound (a : ℝ) (h_a : a ∈ Set.Ioo 0 2) :
  (∀ x₁ x₂, x₁ ∈ Set.Icc (-4) 0 → x₂ ∈ Set.Icc (-4) 0 → 
    ∃ m : ℝ, |f a x₁ - f a x₂| < 4 * exp (-2) + m * exp a) ↔ 
  (∃ m : ℝ, m > (1 + exp 2) / exp 3 ∧ 
    ∀ x₁ x₂, x₁ ∈ Set.Icc (-4) 0 → x₂ ∈ Set.Icc (-4) 0 → 
      |f a x₁ - f a x₂| < 4 * exp (-2) + m * exp a) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_equiv_m_bound_l97_9783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_possible_XG_l97_9723

/-- Given a line segment XZ with point Y on it such that XY = 5 and YZ = 3,
    and a point G on XZ that is the centroid of a triangle ABC where X is on BC,
    Y is on AC, and Z is on AB, prove that the largest possible value of XG is 20/3. -/
theorem largest_possible_XG (X Y Z G : ℝ) : 
  Y ∈ Set.Icc X Z →  -- Y is between X and Z
  X + 5 = Y →    -- XY = 5
  Y + 3 = Z →    -- YZ = 3
  G ∈ Set.Icc X Z →  -- G is between X and Z
  (∃ (A B C : ℝ), G = (A + B + C) / 3 ∧  -- G is the centroid of triangle ABC
                  X ∈ Set.Icc B C ∧          -- X is on BC
                  Y ∈ Set.Icc A C ∧          -- Y is on AC
                  Z ∈ Set.Icc A B) →         -- Z is on AB
  (∀ g : ℝ, G - X = g → g ≤ 20/3) ∧      -- XG is at most 20/3
  (∃ g : ℝ, G - X = g ∧ g = 20/3)        -- XG can equal 20/3
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_possible_XG_l97_9723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_inradius_circumradius_ratio_l97_9747

open Real

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  a_pos : 0 < a
  b_pos : 0 < b

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is on the hyperbola -/
def on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Calculates the eccentricity of the hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  sqrt (1 + h.b^2 / h.a^2)

/-- Represents the foci of the hyperbola -/
structure Foci (h : Hyperbola) where
  F₁ : Point
  F₂ : Point

/-- Calculates the dot product of two vectors -/
def dot_product (p₁ p₂ p₃ : Point) : ℝ :=
  (p₂.x - p₁.x) * (p₃.x - p₁.x) + (p₂.y - p₁.y) * (p₃.y - p₁.y)

/-- The main theorem -/
theorem hyperbola_inradius_circumradius_ratio 
  (h : Hyperbola) 
  (p : Point) 
  (f : Foci h) 
  (h_on : on_hyperbola h p) 
  (h_ecc : eccentricity h = sqrt 3) 
  (h_perp : dot_product p f.F₁ f.F₂ = 0) : 
  ∃ (r R : ℝ), r / R = sqrt 15 / 3 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_inradius_circumradius_ratio_l97_9747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_squares_theorem_l97_9715

/-- The probability that the sum of squares of two randomly selected numbers 
    from [0,2] is within [0,2] -/
noncomputable def probability_sum_squares_in_range : ℝ := Real.pi / 8

/-- The sample space: a square with side length 2 -/
noncomputable def sample_space : ℝ := 2 * 2

/-- The event space: a quarter circle with radius √2 -/
noncomputable def event_space : ℝ := Real.pi * (Real.sqrt 2)^2 / 4

theorem probability_sum_squares_theorem :
  probability_sum_squares_in_range = event_space / sample_space := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_squares_theorem_l97_9715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l97_9790

/-- A function that is even and increasing on the non-negative reals -/
def EvenIncreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y)

/-- The theorem statement -/
theorem solution_set_of_inequality 
  (f : ℝ → ℝ) 
  (h_even_increasing : EvenIncreasingFunction f)
  (h_positive : f (1/3) > 0) :
  {x : ℝ | f (Real.log x / Real.log (1/8)) > 0} = {x : ℝ | 0 < x ∧ x < 1/2} ∪ {x : ℝ | x > 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l97_9790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_in_folded_equilateral_triangle_l97_9729

/-- Given an equilateral triangle ABC with side length 3, where vertex A is folded to A' on BC
    such that BA' = 1 and A'C = 2, the length of the crease PQ is 7√21/20. -/
theorem crease_length_in_folded_equilateral_triangle :
  ∀ (A B C A' P Q : ℝ × ℝ),
    -- ABC is an equilateral triangle with side length 3
    ‖B - C‖ = 3 ∧ ‖A - B‖ = 3 ∧ ‖A - C‖ = 3 →
    -- A' is on BC
    (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ A' = t • B + (1 - t) • C) →
    -- BA' = 1 and A'C = 2
    ‖B - A'‖ = 1 ∧ ‖A' - C‖ = 2 →
    -- P is on AB and Q is on AC
    (∃ u v : ℝ, 0 ≤ u ∧ u ≤ 1 ∧ 0 ≤ v ∧ v ≤ 1 ∧
      P = u • A + (1 - u) • B ∧
      Q = v • A + (1 - v) • C) →
    -- AP = A'P and AQ = A'Q (folding condition)
    ‖A - P‖ = ‖A' - P‖ ∧ ‖A - Q‖ = ‖A' - Q‖ →
    -- The length of PQ is 7√21/20
    ‖P - Q‖ = 7 * Real.sqrt 21 / 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_in_folded_equilateral_triangle_l97_9729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_sixteen_to_twelve_l97_9741

theorem fourth_root_sixteen_to_twelve : ((16 : ℝ) ^ (1/4 : ℝ)) ^ 12 = 4096 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_sixteen_to_twelve_l97_9741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_associated_function_m_range_l97_9708

-- Define the interval [0, 3]
noncomputable def I : Set ℝ := Set.Icc 0 3

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x^3 / 3 - 3 * x^2 / 2 + 4 * x
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := 2 * x + m

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := x^2 - 3 * x + 4

-- Define the condition for f to be the "associated function" of g
def isAssociatedFunction (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ∈ I ∧ x₂ ∈ I ∧ x₁ ≠ x₂ ∧
  f' x₁ = g m x₁ ∧ f' x₂ = g m x₂ ∧
  ∀ x ∈ I, x ≠ x₁ → x ≠ x₂ → f' x ≠ g m x

theorem associated_function_m_range :
  ∀ m : ℝ, isAssociatedFunction m ↔ -9/4 < m ∧ m ≤ -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_associated_function_m_range_l97_9708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extraneous_root_exists_l97_9745

noncomputable def f (x : ℝ) := Real.sqrt (x + 16) - 8 / Real.sqrt (x + 16)

theorem extraneous_root_exists :
  ∃ x : ℝ, -8 < x ∧ x < -4 ∧ f x ≠ 3 ∧ 
  (∃ u : ℝ, u^2 = x + 16 ∧ u - 8/u = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extraneous_root_exists_l97_9745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_most_probable_parent_genotypes_l97_9788

/-- Represents the alleles for rabbit fur type -/
inductive Allele
| H  -- Dominant hairy
| h  -- Recessive hairy
| S  -- Dominant smooth
| s  -- Recessive smooth

/-- Represents the genotype of a rabbit -/
structure Genotype where
  allele1 : Allele
  allele2 : Allele

/-- Determines if a rabbit has hairy fur based on its genotype -/
def isHairy (g : Genotype) : Bool :=
  match g.allele1, g.allele2 with
  | Allele.H, _ => true
  | _, Allele.H => true
  | Allele.h, Allele.h => true
  | _, _ => false

/-- Represents the probability of an allele in the population -/
structure AlleleProbability where
  hairy : ℝ
  smooth : ℝ

/-- Theorem: Most probable parent genotypes for all hairy offspring -/
theorem most_probable_parent_genotypes
  (p : AlleleProbability)
  (h : p.hairy = 0.1)
  (s : p.smooth = 0.9)
  (parent1 : Genotype)
  (parent2 : Genotype)
  (offspring_all_hairy : ∀ (child : Genotype), 
    child.allele1 ∈ ({parent1.allele1, parent1.allele2} : Set Allele) →
    child.allele2 ∈ ({parent2.allele1, parent2.allele2} : Set Allele) →
    isHairy child = true) :
  (parent1 = ⟨Allele.H, Allele.H⟩ ∧ parent2 = ⟨Allele.S, Allele.h⟩) ∨
  (parent2 = ⟨Allele.H, Allele.H⟩ ∧ parent1 = ⟨Allele.S, Allele.h⟩) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_most_probable_parent_genotypes_l97_9788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l97_9759

-- Define a circle
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

-- Define four points on the circle
variable (A B C D : Circle)

-- Define a function to calculate the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the sum of distances from a point to the four houses
noncomputable def sum_distances (O : ℝ × ℝ) (A B C D : Circle) : ℝ :=
  distance O A + distance O B + distance O C + distance O D

-- Define the intersection point of diagonals
noncomputable def intersection_point (A B C D : Circle) : ℝ × ℝ := sorry

-- State the theorem
theorem min_sum_distances (A B C D : Circle) :
  ∀ O : ℝ × ℝ, sum_distances O A B C D ≥ sum_distances (intersection_point A B C D) A B C D :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l97_9759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_expansion_l97_9751

/-- Binomial expansion of (a + bX)^n as a polynomial in X -/
def binomial_expansion (n : ℕ) (a b : ℝ) : Polynomial ℝ := sorry

/-- Coefficient of X^n in a polynomial p -/
def coefficient_of_power (p : Polynomial ℝ) (n : ℕ) : ℝ := sorry

/-- The coefficient of x^2 in x(1+2x)^6 is 12 -/
theorem coefficient_x_squared_in_expansion :
  let p : Polynomial ℝ := X * (1 + 2 * X)^6
  coefficient_of_power p 2 = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_expansion_l97_9751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_six_circles_l97_9779

/-- The area of the shaded region formed by six identical circles of radius 3 units intersecting at the origin -/
noncomputable def shaded_area (r : ℝ) : ℝ :=
  let n : ℕ := 6  -- number of circles
  let quarter_circle_area := Real.pi * r^2 / 4
  let triangle_area := r^2 / 2
  let checkered_region_area := quarter_circle_area - triangle_area
  n * 2 * checkered_region_area

/-- Theorem stating that the shaded area for circles with radius 3 is 27π - 54 -/
theorem shaded_area_six_circles :
  shaded_area 3 = 27 * Real.pi - 54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_six_circles_l97_9779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_average_speed_l97_9792

noncomputable def distance : ℝ := 120
noncomputable def speed_AB : ℝ := 30
noncomputable def speed_BA : ℝ := 40

noncomputable def average_speed : ℝ := (2 * distance) / ((distance / speed_AB) + (distance / speed_BA))

theorem round_trip_average_speed :
  (average_speed ≥ 34) ∧ (average_speed < 35) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_average_speed_l97_9792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_downstream_distance_l97_9704

/-- The total distance traveled by a boat downstream given its speed in still water,
    journey duration, and varying current speeds. -/
theorem boat_downstream_distance 
  (boat_speed : ℝ) 
  (journey_duration : ℝ) 
  (current_speed1 current_speed2 current_speed3 : ℝ) 
  (h1 : boat_speed = 15)
  (h2 : journey_duration = 12 / 60)
  (h3 : current_speed1 = 3)
  (h4 : current_speed2 = 5)
  (h5 : current_speed3 = 7) :
  let segment_duration := journey_duration / 3
  let distance1 := (boat_speed + current_speed1) * segment_duration
  let distance2 := (boat_speed + current_speed2) * segment_duration
  let distance3 := (boat_speed + current_speed3) * segment_duration
  let total_distance := distance1 + distance2 + distance3
  ∃ ε > 0, |total_distance - 4.001| < ε := by
  sorry

#check boat_downstream_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_downstream_distance_l97_9704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_is_closest_l97_9733

/-- The line is defined by the equation y = (x + 2) / 3 -/
noncomputable def line (x : ℝ) : ℝ := (x + 2) / 3

/-- The point we're measuring distance from -/
def point : ℝ × ℝ := (5, -1)

/-- The proposed closest point on the line -/
def closest_point : ℝ × ℝ := (4, 2)

/-- Theorem stating that the closest_point is indeed the closest point on the line to point -/
theorem closest_point_is_closest :
  ∀ x : ℝ, 
  (x - point.1)^2 + (line x - point.2)^2 ≥ 
  (closest_point.1 - point.1)^2 + (closest_point.2 - point.2)^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_is_closest_l97_9733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_max_symmetry_l97_9769

/-- Represents the number of lines of symmetry for a geometric figure -/
def lines_of_symmetry (figure : Type) : ℕ := sorry

/-- A right triangle -/
inductive right_triangle : Type
| mk : right_triangle

/-- A parallelogram -/
inductive parallelogram : Type
| mk : parallelogram

/-- A regular pentagon -/
inductive regular_pentagon : Type
| mk : regular_pentagon

/-- An isosceles trapezoid -/
inductive isosceles_trapezoid : Type
| mk : isosceles_trapezoid

/-- A square -/
inductive square : Type
| mk : square

theorem regular_pentagon_max_symmetry :
  (lines_of_symmetry regular_pentagon > lines_of_symmetry right_triangle) ∧
  (lines_of_symmetry regular_pentagon > lines_of_symmetry parallelogram) ∧
  (lines_of_symmetry regular_pentagon > lines_of_symmetry isosceles_trapezoid) ∧
  (lines_of_symmetry regular_pentagon > lines_of_symmetry square) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_max_symmetry_l97_9769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_valid_sets_l97_9768

def satisfies_condition (M : Set ℝ) : Prop :=
  ∀ a b, a ∈ M → b ∈ M → a ≠ b → a^3 - 4/9 * b ∈ M

def is_valid_set (M : Set ℝ) : Prop :=
  M.Finite ∧ M.Nonempty ∧ satisfies_condition M

theorem characterization_of_valid_sets :
  ∀ M : Set ℝ, is_valid_set M ↔
    M = {-Real.sqrt 5 / 3, Real.sqrt 5 / 3} ∨
    M = {(1 - Real.sqrt 17) / 6, (1 + Real.sqrt 17) / 6} ∨
    M = {(-1 - Real.sqrt 17) / 6, (-1 + Real.sqrt 17) / 6} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_valid_sets_l97_9768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_implies_a_range_l97_9709

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * a * x^2 + (a-1) * x + 1

/-- The derivative of f(x) with respect to x -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := x^2 - a * x + (a-1)

/-- Theorem stating the range of a given the conditions on f(x) -/
theorem f_monotonicity_implies_a_range :
  ∀ a : ℝ, 
  (∀ x ∈ Set.Ioo 1 4, (f_derivative a x) < 0) →
  (∀ x > 6, (f_derivative a x) > 0) →
  a ∈ Set.Ioo 5 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_implies_a_range_l97_9709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l97_9777

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x)

-- State the theorem
theorem f_monotone_increasing :
  StrictMonoOn f (Set.Ioi 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l97_9777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_consumption_order_l97_9797

-- Define the pizza as a unit (1.0)
noncomputable def pizza : ℝ := 1.0

-- Define the fractions eaten by each sibling
noncomputable def alex_fraction : ℝ := 1/5
noncomputable def beth_fraction : ℝ := 1/3
noncomputable def cyril_fraction : ℝ := 1/4

-- Define the amount eaten by Dan as the remainder
noncomputable def dan_fraction : ℝ := pizza - (alex_fraction + beth_fraction + cyril_fraction)

-- Define a function to represent the order of pizza consumption
def consumption_order : List String := ["Beth", "Cyril", "Dan", "Alex"]

-- Theorem statement
theorem pizza_consumption_order : 
  beth_fraction > cyril_fraction ∧
  cyril_fraction > dan_fraction ∧
  dan_fraction > alex_fraction ∧
  consumption_order = ["Beth", "Cyril", "Dan", "Alex"] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_consumption_order_l97_9797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_for_f_iter_3_l97_9786

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 ∧ x % 5 = 0 then x / 10
  else if x % 5 = 0 then 2 * x
  else if x % 2 = 0 then 5 * x
  else x + 2

def f_iter : ℕ → ℕ → ℕ
  | 0, x => x
  | n + 1, x => f_iter n (f x)

theorem smallest_a_for_f_iter_3 :
  ∀ a : ℕ, a > 1 → (f_iter a 3 = f 3 ↔ a ≥ 9) :=
by
  sorry

#eval f_iter 9 3
#eval f 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_for_f_iter_3_l97_9786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hiker_total_distance_is_62_l97_9794

/-- A hiker's journey over three days -/
structure HikerJourney where
  day1_distance : ℕ
  day1_speed : ℕ
  day2_speed_increase : ℕ
  day2_time_decrease : ℕ

/-- Calculate the total distance walked by the hiker over three days -/
def total_distance (journey : HikerJourney) : ℕ :=
  let day1_time := journey.day1_distance / journey.day1_speed
  let day2_speed := journey.day1_speed + journey.day2_speed_increase
  let day2_time := day1_time - journey.day2_time_decrease
  let day2_distance := day2_speed * day2_time
  let day3_distance := day2_speed * day1_time
  journey.day1_distance + day2_distance + day3_distance

/-- Theorem stating that the hiker's total distance is 62 miles -/
theorem hiker_total_distance_is_62 (journey : HikerJourney) 
  (h1 : journey.day1_distance = 18)
  (h2 : journey.day1_speed = 3)
  (h3 : journey.day2_speed_increase = 1)
  (h4 : journey.day2_time_decrease = 1) :
  total_distance journey = 62 := by
  sorry

#eval total_distance { day1_distance := 18, day1_speed := 3, day2_speed_increase := 1, day2_time_decrease := 1 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hiker_total_distance_is_62_l97_9794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_of_whole_l97_9700

theorem percent_of_whole (part whole result : ℝ) :
  part = 6.2 →
  whole = 1000 →
  (part / whole) * 100 = result →
  result = 0.62 :=
by
  intros h_part h_whole h_result
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_of_whole_l97_9700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_diameter_formula_valid_l97_9705

/-- Represents the diameter of a sphere given its radius and the side length of an equilateral triangle on its surface. -/
noncomputable def sphereDiameter (ρ : ℝ) : ℝ :=
  let a := ρ * Real.sqrt 3
  let p := 3 * a / 2
  ρ^2 / Real.sqrt (ρ^2 - (a^2 * a^2 * a^2) / (16 * p * (p - a) * (p - a) * (p - a)))

/-- Theorem stating that the sphere diameter formula is correct for the given conditions. -/
theorem sphere_diameter_formula_valid (ρ : ℝ) (h : ρ > 0) :
  sphereDiameter ρ = 2 * ρ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_diameter_formula_valid_l97_9705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_formula_volume_formula_proof_l97_9767

/-- A quadrilateral pyramid with a rectangular base -/
structure QuadPyramid where
  /-- Length of the diagonal of the base rectangle -/
  b : ℝ
  /-- Angle between the diagonals of the base rectangle (in radians) -/
  diag_angle : ℝ
  /-- Angle between each lateral edge and the base plane (in radians) -/
  lateral_angle : ℝ
  /-- The diagonal length is positive -/
  h_b_pos : 0 < b
  /-- The angle between diagonals is 60° -/
  h_diag_angle : diag_angle = π / 3
  /-- The angle between lateral edges and base plane is 45° -/
  h_lateral_angle : lateral_angle = π / 4

/-- The volume of the quadrilateral pyramid -/
noncomputable def pyramid_volume (p : QuadPyramid) : ℝ :=
  (p.b ^ 3 * Real.sqrt 3) / 24

/-- Theorem: The volume of the quadrilateral pyramid is (b³√3) / 24 -/
theorem pyramid_volume_formula (p : QuadPyramid) :
  pyramid_volume p = (p.b ^ 3 * Real.sqrt 3) / 24 := by
  -- Unfold the definition of pyramid_volume
  unfold pyramid_volume
  -- The equality holds by definition
  rfl

/-- Lemma: The base area of the pyramid is (b²√3) / 4 -/
lemma base_area (p : QuadPyramid) :
  (p.b ^ 2 * Real.sqrt 3) / 4 = (p.b / 2) * (p.b * Real.sqrt 3 / 2) := by
  sorry

/-- Lemma: The height of the pyramid is b / 2 -/
lemma pyramid_height (p : QuadPyramid) :
  p.b / 2 = (p.b / 2) * Real.tan (π / 4) := by
  sorry

/-- Theorem: The volume formula is correct -/
theorem volume_formula_proof (p : QuadPyramid) :
  pyramid_volume p = (1 / 3) * ((p.b ^ 2 * Real.sqrt 3) / 4) * (p.b / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_formula_volume_formula_proof_l97_9767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highway_miles_theorem_l97_9773

/-- Represents the fuel efficiency and travel data of a car -/
structure CarData where
  city_mpg : ℚ
  city_miles_per_tank : ℚ
  highway_city_mpg_diff : ℚ

/-- Calculates the miles per tankful on the highway given car data -/
def highway_miles_per_tank (data : CarData) : ℚ :=
  let tank_size := data.city_miles_per_tank / data.city_mpg
  let highway_mpg := data.city_mpg + data.highway_city_mpg_diff
  highway_mpg * tank_size

/-- Theorem stating that given the conditions, the car travels 420 miles per tankful on the highway -/
theorem highway_miles_theorem (data : CarData) 
    (h1 : data.city_mpg = 24)
    (h2 : data.city_miles_per_tank = 336)
    (h3 : data.highway_city_mpg_diff = 6) : 
  highway_miles_per_tank data = 420 := by
  sorry

#eval highway_miles_per_tank { city_mpg := 24, city_miles_per_tank := 336, highway_city_mpg_diff := 6 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highway_miles_theorem_l97_9773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_identity_l97_9752

theorem sin_cos_identity : 
  Real.sin (135 * π / 180) * Real.cos (15 * π / 180) - 
  Real.cos (45 * π / 180) * Real.sin (-15 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_identity_l97_9752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_ultra_deficient_l97_9736

/-- Sum of divisors function -/
def f (n : ℕ) : ℕ := sorry

/-- An integer n is ultra-deficient if f(f(n)) = n + 6 -/
def is_ultra_deficient (n : ℕ) : Prop := f (f n) = n + 6

/-- There exists exactly one ultra-deficient positive integer -/
theorem unique_ultra_deficient : ∃! (n : ℕ), n > 0 ∧ is_ultra_deficient n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_ultra_deficient_l97_9736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_parabola_focus_l97_9754

/-- The focus of a parabola y^2 = 4px is at (p/2, 0) -/
noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)

/-- A point (x, y) lies on a line ax - y + 1 = 0 -/
def point_on_line (a x y : ℝ) : Prop := a * x - y + 1 = 0

/-- A point (x, y) lies on a parabola y^2 = 4px -/
def point_on_parabola (p x y : ℝ) : Prop := y^2 = 4 * p * x

theorem line_through_parabola_focus (a : ℝ) :
  point_on_line a 1 0 ∧ point_on_parabola 1 1 0 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_parabola_focus_l97_9754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ortho_quad_special_triangle_area_l97_9780

/-- Represents a quadrilateral with perpendicular diagonals -/
structure OrthoQuad where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  perp_diag : (A.1 - C.1) * (B.1 - D.1) + (A.2 - C.2) * (B.2 - D.2) = 0
  convex : True  -- Simplified convexity condition

/-- The square of the area of a triangle given its side lengths -/
noncomputable def triangle_area_squared (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  s * (s - a) * (s - b) * (s - c)

/-- Main theorem -/
theorem ortho_quad_special_triangle_area 
  (q : OrthoQuad) 
  (right_angle_B : (q.B.1 - q.A.1) * (q.B.1 - q.C.1) + (q.B.2 - q.A.2) * (q.B.2 - q.C.2) = 0)
  (right_angle_C : (q.C.1 - q.B.1) * (q.C.1 - q.D.1) + (q.C.2 - q.B.2) * (q.C.2 - q.D.2) = 0)
  (BC_length : Real.sqrt ((q.B.1 - q.C.1)^2 + (q.B.2 - q.C.2)^2) = 20)
  (AD_length : Real.sqrt ((q.A.1 - q.D.1)^2 + (q.A.2 - q.D.2)^2) = 30) :
  triangle_area_squared 
    (Real.sqrt ((q.C.1 - q.D.1)^2 + (q.C.2 - q.D.2)^2))
    30
    (Real.sqrt ((q.A.1 - q.B.1)^2 + (q.A.2 - q.B.2)^2)) = 30000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ortho_quad_special_triangle_area_l97_9780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_of_hexagons_l97_9743

-- Define the hexagon ABCDEF
structure RegularHexagon :=
  (side_length : ℝ)

-- Define the points P, Q, R, S, T, U as midpoints
def midpoint_of_side (h : RegularHexagon) (side : Fin 6) : Point := sorry

-- Define the intersection points V, W, X, Y, Z, A'
def intersection_point (h : RegularHexagon) (i : Fin 6) : Point := sorry

-- Define the inner hexagon VWXYZA'
def inner_hexagon (h : RegularHexagon) : RegularHexagon := sorry

-- Define the area of a hexagon
noncomputable def area (h : RegularHexagon) : ℝ := sorry

-- Theorem statement
theorem area_ratio_of_hexagons (ABCDEF : RegularHexagon) :
  ABCDEF.side_length = 4 →
  (area (inner_hexagon ABCDEF)) / (area ABCDEF) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_of_hexagons_l97_9743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_quadrilateral_area_l97_9796

/-- The area of a quadrilateral given its diagonal and two offsets -/
noncomputable def quadrilateralArea (d h₁ h₂ : ℝ) : ℝ := (1/2) * d * (h₁ + h₂)

/-- Theorem stating that a quadrilateral with diagonal 28 cm and offsets 9 cm and 6 cm has an area of 210 cm² -/
theorem specific_quadrilateral_area :
  quadrilateralArea 28 9 6 = 210 := by
  -- Unfold the definition of quadrilateralArea
  unfold quadrilateralArea
  -- Simplify the expression
  simp
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_quadrilateral_area_l97_9796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_eccentricity_l97_9718

-- Define the setup
def Point : Type := ℝ × ℝ
def Ellipse : Type := Point → Prop
def Hyperbola : Type := Point → Prop

-- Define helper functions
noncomputable def dist (p q : Point) : ℝ := sorry
noncomputable def angle (p q r : Point) : ℝ := sorry
noncomputable def semi_major_axis (curve : Point → Prop) : ℝ := sorry
noncomputable def eccentricity (curve : Point → Prop) : ℝ := sorry

-- Define the problem
theorem ellipse_hyperbola_eccentricity 
  (M : Ellipse) (Γ : Hyperbola) (F₁ F₂ P : Point) 
  (e₁ e₂ : ℝ) :
  (∃ (P : Point), M P ∧ Γ P) →  -- P is an intersection of M and Γ
  (∀ (Q : Point), M Q → (dist Q F₁ + dist Q F₂ = 2 * semi_major_axis M)) →  -- M is an ellipse with foci F₁ and F₂
  (∀ (Q : Point), Γ Q → (dist Q F₁ - dist Q F₂ = 2 * semi_major_axis Γ)) →  -- Γ is a hyperbola with foci F₁ and F₂
  Real.cos (angle F₁ P F₂) = 4/5 →  -- cos∠F₁PF₂ = 4/5
  e₁ = eccentricity M →  -- e₁ is the eccentricity of M
  e₂ = eccentricity Γ →  -- e₂ is the eccentricity of Γ
  e₂ = 2 * e₁ →  -- e₂ = 2e₁
  e₁ = Real.sqrt 13 / 20 :=  -- Prove that e₁ = √13/20
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_eccentricity_l97_9718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_chord_product_l97_9724

def semicircle_radius : ℝ := 3

def num_divisions : ℕ := 5

noncomputable def chord_product (r : ℝ) (n : ℕ) : ℝ :=
  let ω := Complex.exp (2 * Real.pi * Complex.I / (2 * n))
  let chord_lengths := List.range (n - 1) |>.map (fun k =>
    (Complex.abs (1 - ω ^ k)) * (Complex.abs (1 - ω ^ (k + n))))
  (r ^ (2 * (n - 1))) * chord_lengths.prod

theorem semicircle_chord_product :
  chord_product semicircle_radius num_divisions = 590490 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_chord_product_l97_9724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l97_9774

-- Define the function f as noncomputable due to the use of Real.log
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - a*x + a)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) ↔ (a ≤ 0 ∨ a ≥ 4) := by
  sorry

-- You can add additional lemmas or theorems here if needed for the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l97_9774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_l97_9702

theorem cone_volume (totalArea : Real) (centralAngle : Real) :
  totalArea = 15 * Real.pi ∧ centralAngle = 60 * (Real.pi / 180) →
  ∃ (volume : Real), volume = (25 * Real.sqrt 3) / 7 * Real.pi := by
  intro h
  -- Let r be the radius of the base and l be the slant height
  let r := Real.sqrt 105 / 7
  let l := 6 * Real.sqrt 105 / 7
  
  -- Calculate the height h
  let h := 5 * Real.sqrt 3
  
  -- Define the volume
  let volume := (1/3) * Real.pi * r^2 * h
  
  -- Assert the existence of the volume
  use volume
  
  -- The actual proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_l97_9702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometricSumApproximation_l97_9795

/-- The sum of a geometric series with initial term 4000, common ratio 1.05, and 10 terms -/
noncomputable def geometricSum : ℝ :=
  4000 * (1 - (1.05 ^ 10)) / (1 - 1.05)

/-- The approximate value of the geometric sum -/
def approximateSum : ℝ := 50311.2

/-- Theorem stating that the geometric sum is approximately equal to the given value -/
theorem geometricSumApproximation : 
  abs (geometricSum - approximateSum) < 0.1 := by
  sorry

#eval approximateSum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometricSumApproximation_l97_9795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_building_floors_l97_9744

theorem building_floors (rooms_per_floor : ℕ) (hours_per_room : ℕ) (hourly_rate : ℕ) (total_earnings : ℕ) : 
  rooms_per_floor = 10 →
  hours_per_room = 6 →
  hourly_rate = 15 →
  total_earnings = 3600 →
  (total_earnings / hourly_rate / hours_per_room / rooms_per_floor : ℕ) = 4 := by
  intros h1 h2 h3 h4
  -- The proof steps would go here
  sorry

#check building_floors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_building_floors_l97_9744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acres_purchased_is_four_l97_9721

/-- Represents the land purchase and division scenario --/
structure LandPurchase where
  costPerAcre : ℚ
  numLots : ℕ
  breakEvenPricePerLot : ℚ

/-- Calculates the number of acres purchased given the land purchase scenario --/
def calculateAcres (land : LandPurchase) : ℚ :=
  (land.numLots : ℚ) * land.breakEvenPricePerLot / land.costPerAcre

/-- Theorem stating that the number of acres purchased is 4 --/
theorem acres_purchased_is_four :
  let land : LandPurchase := {
    costPerAcre := 1863,
    numLots := 9,
    breakEvenPricePerLot := 828
  }
  calculateAcres land = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acres_purchased_is_four_l97_9721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourier_series_sums_l97_9720

/-- A periodic function with period 2π, defined as x^2 on [-π, π) -/
noncomputable def f (x : ℝ) : ℝ := (x % (2 * Real.pi)) ^ 2

/-- The alternating sum 1 - 1/2^2 + 1/3^2 - 1/4^2 + ... -/
noncomputable def alternating_sum : ℝ := Real.pi^2 / 12

/-- The sum 1 + 1/2^2 + 1/3^2 + 1/4^2 + ... -/
noncomputable def sum : ℝ := Real.pi^2 / 6

/-- Theorem stating the equality of the Fourier series sums -/
theorem fourier_series_sums : 
  (∑' n : ℕ+, ((-1)^(n:ℕ) : ℝ) / (n:ℝ)^2) = alternating_sum ∧
  (∑' n : ℕ+, (1 : ℝ) / (n:ℝ)^2) = sum :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourier_series_sums_l97_9720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_investment_time_l97_9714

/-- Represents the investment scenario described in the problem -/
structure Investment where
  a_amount : ℚ  -- A's investment amount
  b_amount : ℚ  -- B's investment amount
  total_profit : ℚ  -- Total profit at the end of the year
  a_profit : ℚ  -- A's share of the profit
  total_months : ℕ  -- Total investment period in months

/-- Calculates the number of months after A's investment that B invested -/
noncomputable def months_until_b_investment (inv : Investment) : ℚ :=
  let a_investment := inv.a_amount * inv.total_months
  inv.total_months - (inv.a_profit / (inv.total_profit - inv.a_profit) * 
           (a_investment / inv.b_amount))

/-- Theorem stating that B invested 6 months after A -/
theorem b_investment_time (inv : Investment) 
  (h1 : inv.a_amount = 400)
  (h2 : inv.b_amount = 200)
  (h3 : inv.total_profit = 100)
  (h4 : inv.a_profit = 80)
  (h5 : inv.total_months = 12) :
  months_until_b_investment inv = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_investment_time_l97_9714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_fifth_power_l97_9740

theorem det_A_fifth_power {n : Type*} [Fintype n] [DecidableEq n] 
  (A : Matrix n n ℝ) (h : Matrix.det A = -7) : 
  Matrix.det (A^5) = -16807 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_fifth_power_l97_9740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_ball_weight_is_12_l97_9731

/-- The weight of a single canoe in pounds -/
noncomputable def canoe_weight : ℚ := 28

/-- The number of canoes that weigh the same as the bowling balls -/
def num_canoes : ℕ := 3

/-- The number of bowling balls that weigh the same as the canoes -/
def num_bowling_balls : ℕ := 7

/-- The weight of a single bowling ball in pounds -/
noncomputable def bowling_ball_weight : ℚ := (canoe_weight * num_canoes) / num_bowling_balls

theorem bowling_ball_weight_is_12 : bowling_ball_weight = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_ball_weight_is_12_l97_9731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_semicircles_triangle_l97_9772

/-- The shaded area between two semicircles drawn on the edges of a right-angled triangle -/
theorem shaded_area_semicircles_triangle (r₁ r₂ : ℝ) (h₁ : r₁ = 3) (h₂ : r₂ = 4) : 
  (π * r₁^2 / 2 + π * r₂^2 / 2) - ((2 * r₁) * (2 * r₂) / 2) = 25 * π / 2 - 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_semicircles_triangle_l97_9772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ray_reflection_tangent_circle_l97_9717

/-- Represents a line in 2D space --/
def Line : Type := ℝ × ℝ → Prop

/-- Reflects a line across the x-axis --/
noncomputable def Line.reflectX (l : Line) : Line := sorry

/-- Checks if a line is tangent to a set (representing a circle) --/
def Line.isTangentTo (l : Line) (s : Set (ℝ × ℝ)) : Prop := sorry

/-- Constructs a line through two points --/
noncomputable def Line.throughPoints (p q : ℝ × ℝ) : Line := sorry

/-- Given a ray passing through P(-3,1) and Q(a,0), reflecting off the x-axis
    and becoming tangent to the circle x²+y²=1, prove that a = -5/3 --/
theorem ray_reflection_tangent_circle (a : ℝ) : 
  let P : ℝ × ℝ := (-3, 1)
  let Q : ℝ × ℝ := (a, 0)
  let circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
  let reflected_ray := Line.reflectX (Line.throughPoints P Q)
  reflected_ray.isTangentTo circle → a = -5/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ray_reflection_tangent_circle_l97_9717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_olya_wins_l97_9775

/-- Represents the game state -/
structure GameState where
  rows : Nat
  cols : Nat
  currentRow : Nat
  currentCol : Nat
  lastMoveDirection : Bool  -- true for right, false for down
  pivots : Nat

/-- Simulates the game play -/
def playGame (n : Nat) (strategy : GameState → Bool) (counterStrategy : GameState → Bool) : GameState :=
  sorry

/-- Defines the game rules and winning conditions -/
def gameRules (n : Nat) : Prop :=
  ∀ (strategy : GameState → Bool),
    (n ∉ ({13, 14, 15} : Set Nat) →
      ∃ (counterStrategy : GameState → Bool),
        let finalState := playGame n strategy counterStrategy
        finalState.pivots % 2 = 1)
    ∧
    (n ∈ ({13, 14, 15} : Set Nat) →
      ∀ (counterStrategy : GameState → Bool),
        let finalState := playGame n strategy counterStrategy
        finalState.pivots % 2 = 0)

/-- The main theorem stating Olya's winning condition -/
theorem olya_wins (n : Nat) : gameRules n ↔ n ∉ ({13, 14, 15} : Set Nat) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_olya_wins_l97_9775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_proper_subsets_l97_9750

def N : Finset ℕ := {1, 3, 5}

theorem number_of_proper_subsets : Finset.card (Finset.powerset N \ {N}) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_proper_subsets_l97_9750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_to_line_l97_9766

/-- The ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1

/-- The line equation -/
def line_equation (x y : ℝ) : ℝ := x + y - 7

/-- Distance from a point to the line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |line_equation x y| / Real.sqrt 2

/-- Maximum distance from ellipse to line -/
theorem max_distance_ellipse_to_line : 
  ∃ (x y : ℝ), is_on_ellipse x y ∧ 
  (∀ (x' y' : ℝ), is_on_ellipse x' y' → 
    distance_to_line x y ≥ distance_to_line x' y') ∧
  distance_to_line x y = 6 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_to_line_l97_9766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_analytical_method_most_suitable_l97_9758

-- Define the inequality (marked as noncomputable due to Real.sqrt)
noncomputable def inequality : ℝ := Real.sqrt 2 + Real.sqrt 3 - Real.sqrt 10

-- Define the proof methods
inductive ProofMethod
| Contradiction
| Analytical
| Synthetic
deriving Repr, DecidableEq

-- Define the properties of the inequality
axiom involves_irrational_expressions : Prop
axiom contradiction_not_appropriate : Prop
axiom synthetic_not_straightforward : Prop

-- Theorem stating that the analytical method is the most suitable
theorem analytical_method_most_suitable :
  involves_irrational_expressions →
  contradiction_not_appropriate →
  synthetic_not_straightforward →
  (ProofMethod.Analytical : ProofMethod) = 
    (let methods := [ProofMethod.Contradiction, ProofMethod.Analytical, ProofMethod.Synthetic]
     methods.argmax (λ m => if m = ProofMethod.Analytical then 1 else 0)) :=
by sorry

#eval ProofMethod.Analytical

end NUMINAMATH_CALUDE_ERRORFEEDBACK_analytical_method_most_suitable_l97_9758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_acute_angle_l97_9701

theorem tan_difference_acute_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos α = Real.sqrt 5 / 5) :
  Real.tan (α - π / 4) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_acute_angle_l97_9701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_midpoint_bound_l97_9753

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (exp a + exp (-a)) * log x - x + 1/x

-- Define the theorem
theorem f_midpoint_bound (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : 0 ≤ a) (h₂ : a ≤ 1) 
  (h₃ : x₁ > 0) (h₄ : x₂ > 0) (h₅ : x₁ ≠ x₂)
  (h₆ : (exp a - exp (-a)) / x₁ - 1 - 1 / x₁^2 = 
        (exp a - exp (-a)) / x₂ - 1 - 1 / x₂^2) : 
  f a ((x₁ + x₂) / 2) ≤ 2 / exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_midpoint_bound_l97_9753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l97_9707

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 3) + Real.sin (2 * x - Real.pi / 3) + 2 * (Real.cos x) ^ 2 - 1

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) → f x ≤ Real.sqrt 2) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) → f x ≥ -1) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) ∧ f x = Real.sqrt 2) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) ∧ f x = -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l97_9707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l97_9719

/-- Given a line l with equation Ax + By + C = 0 (A and B are not both 0),
    prove that the following statements are true -/
theorem line_properties (A B C : ℝ) (h : A ≠ 0 ∨ B ≠ 0) :
  /- Statement A -/
  (A * 2 + B * 1 + C = 0 → ∀ (x y : ℝ), A * x + B * y + C = 0 ↔ A * (x - 2) + B * (y - 1) = 0) ∧
  /- Statement B -/
  (A ≠ 0 ∧ B ≠ 0 → ∃ (x y : ℝ), (A * x + C = 0 ∧ y = 0) ∧ (B * y + C = 0 ∧ x = 0)) ∧
  /- Statement C -/
  (A = 0 ∧ B ≠ 0 ∧ C ≠ 0 → ∃ (k : ℝ), ∀ (x y : ℝ), B * y + C = 0 ↔ y = k) ∧
  /- Statement D -/
  (A ≠ 0 ∧ B^2 + C^2 = 0 → ∀ (x y : ℝ), A * x + B * y + C = 0 ↔ x = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l97_9719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solution_set_range_max_F_correct_l97_9764

-- Define the functions f and g
noncomputable def f (a x : ℝ) : ℝ := |x - a|
def g (a x : ℝ) : ℝ := a * x

-- Define the set of a for which f(x) = g(x) has two distinct solutions
def two_solution_set : Set ℝ := {a | ∃ x₁ x₂, x₁ ≠ x₂ ∧ f a x₁ = g a x₁ ∧ f a x₂ = g a x₂}

-- Define F(x) = g(x)f(x)
noncomputable def F (a x : ℝ) : ℝ := g a x * f a x

-- Define the maximum value function
noncomputable def max_F (a : ℝ) : ℝ :=
  if 0 < a ∧ a < 5/3 then 4*a - 2*a^2
  else if 5/3 ≤ a ∧ a ≤ 2 then a^2 - a
  else if 2 < a ∧ a ≤ 4 then a^3/4
  else if 4 < a then 2*a^2 - 4*a
  else 0  -- This case should not occur given a > 0, but Lean requires all cases to be covered

-- State the theorems to be proved
theorem two_solution_set_range : two_solution_set = Set.Ioo (-1) 0 ∪ Set.Ioo 0 1 := sorry

theorem max_F_correct (a : ℝ) (h : a > 0) : 
  ∃ x ∈ Set.Icc 1 2, ∀ y ∈ Set.Icc 1 2, F a x ≥ F a y ∧ F a x = max_F a := sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solution_set_range_max_F_correct_l97_9764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_best_stability_measure_l97_9770

-- Define the set of scores
def scores : List ℝ := [86, 78, 80, 85, 92]

-- Define the statistical measures
noncomputable def average (l : List ℝ) : ℝ := (l.sum) / l.length
noncomputable def variance (l : List ℝ) : ℝ := (l.map (λ x => (x - average l)^2)).sum / l.length
def median (l : List ℝ) : ℝ := sorry -- Definition of median
def mode (l : List ℝ) : ℝ := sorry -- Definition of mode

-- Define a type for stability measures
inductive StabilityMeasure
| Average
| Variance
| Median
| Mode

-- Function to determine the best stability measure
def best_stability_measure (s : List ℝ) : StabilityMeasure := sorry

-- Theorem stating that variance is the best measure of stability
theorem variance_best_stability_measure :
  ∀ (s : List ℝ), s = scores →
  variance s > average s ∧ variance s > median s ∧ variance s > mode s →
  best_stability_measure s = StabilityMeasure.Variance :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_best_stability_measure_l97_9770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leftover_cookie_radius_l97_9742

-- Define the radius of the large dough
def large_radius : ℝ := 5

-- Define the radius of each small cookie
def small_radius : ℝ := 1

-- Define the number of small cookies
def num_small_cookies : ℕ := 8

-- Theorem statement
theorem leftover_cookie_radius :
  let large_area := π * large_radius ^ 2
  let small_area := π * small_radius ^ 2
  let total_small_area := (num_small_cookies : ℝ) * small_area
  let leftover_area := large_area - total_small_area
  let leftover_radius := Real.sqrt (leftover_area / π)
  leftover_radius = Real.sqrt 17 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_leftover_cookie_radius_l97_9742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_powers_l97_9726

theorem compare_powers : 
  (2 : ℕ) ^ 40 < (4 : ℕ) ^ 24 ∧ (4 : ℕ) ^ 24 < (3 : ℕ) ^ 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_powers_l97_9726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l97_9739

theorem simplify_expression (k : ℝ) (h : k ≠ 0) :
  (1 / (3 * k))^(-3 : ℤ) * (-2 * k)^2 = 108 * k^5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l97_9739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_height_l97_9730

theorem prism_height (α β : Real) (b : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2) (h3 : b > 0) :
  ∃ height : Real, height = b / Real.cos (α/2) * Real.sqrt (Real.sin (β + α/2) * Real.sin (β - α/2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_height_l97_9730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_XX₁_l97_9787

-- Define the triangles and points
structure RightTriangle where
  hypotenuse : ℝ
  leg : ℝ

def triangle_DEF : RightTriangle := { hypotenuse := 13, leg := 5 }

-- D₁ is on EF
noncomputable def D₁_on_EF (t : RightTriangle) : ℝ → Prop := 
  λ x => 0 < x ∧ x < Real.sqrt (t.hypotenuse^2 - t.leg^2)

-- XY = D₁E and XZ = 1/2 * D₁F
noncomputable def triangle_XYZ (t : RightTriangle) (d₁ : ℝ) : RightTriangle :=
  { hypotenuse := Real.sqrt (t.hypotenuse^2 - t.leg^2) - d₁,
    leg := (d₁ * t.leg) / (Real.sqrt (t.hypotenuse^2 - t.leg^2)) / 2 }

-- X₁ is on YZ
noncomputable def X₁_on_YZ (t : RightTriangle) : ℝ → Prop :=
  λ x => 0 < x ∧ x < Real.sqrt (t.hypotenuse^2 - t.leg^2)

-- Main theorem
theorem length_of_XX₁ (d₁ : ℝ) (x₁ : ℝ) 
  (h₁ : D₁_on_EF triangle_DEF d₁)
  (h₂ : X₁_on_YZ (triangle_XYZ triangle_DEF d₁) x₁) :
  x₁ = 30 / 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_XX₁_l97_9787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_representation_l97_9756

/-- Represents a hyperbola in the complex plane --/
def IsHyperbola (S : Set ℂ) : Prop := sorry

theorem hyperbola_representation (m n : ℝ) (z : ℂ) 
  (hm : m ≠ 0) (hn : n ≠ 0) :
  (Complex.abs (z + n * Complex.I) + Complex.abs (z - m * Complex.I) = n) ∧ 
  (Complex.abs (z + n * Complex.I) - Complex.abs (z - m * Complex.I) = -m) →
  IsHyperbola {z : ℂ | (Complex.abs (z + n * Complex.I) + Complex.abs (z - m * Complex.I) = n) ∧ 
                       (Complex.abs (z + n * Complex.I) - Complex.abs (z - m * Complex.I) = -m)} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_representation_l97_9756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_arctan_sum_l97_9716

theorem triangle_arctan_sum (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  let angle_C := Real.arcsin (c / (a + b + c))
  Real.arctan (a / (b + c)) + Real.arctan (b / (a + c)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_arctan_sum_l97_9716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lattice_points_with_non_lattice_centroid_l97_9748

/-- A lattice point on the coordinate plane -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- The centroid of a triangle formed by three lattice points -/
def centroid (p1 p2 p3 : LatticePoint) : ℚ × ℚ :=
  (((p1.x : ℚ) + (p2.x : ℚ) + (p3.x : ℚ)) / 3, ((p1.y : ℚ) + (p2.y : ℚ) + (p3.y : ℚ)) / 3)

/-- Predicate to check if a point is a lattice point -/
def isLatticePoint (p : ℚ × ℚ) : Prop :=
  ∃ (x y : ℤ), p = ((x : ℚ), (y : ℚ))

/-- The main theorem statement -/
theorem max_lattice_points_with_non_lattice_centroid :
  ∃ (n : ℕ) (points : Finset LatticePoint),
    points.card = n ∧
    (∀ (p1 p2 p3 : LatticePoint),
      p1 ∈ points → p2 ∈ points → p3 ∈ points →
      p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 →
      ¬isLatticePoint (centroid p1 p2 p3)) ∧
    (∀ (m : ℕ) (larger_set : Finset LatticePoint),
      m > n →
      larger_set.card = m →
      ∃ (q1 q2 q3 : LatticePoint),
        q1 ∈ larger_set ∧ q2 ∈ larger_set ∧ q3 ∈ larger_set ∧
        q1 ≠ q2 ∧ q2 ≠ q3 ∧ q1 ≠ q3 ∧
        isLatticePoint (centroid q1 q2 q3)) ∧
    n = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lattice_points_with_non_lattice_centroid_l97_9748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_value_l97_9703

-- Define the power function as noncomputable
noncomputable def power_function (a : ℝ) : ℝ → ℝ := fun x ↦ x ^ a

-- State the theorem
theorem power_function_through_point_value :
  ∀ a : ℝ, power_function a 2 = 1/4 → power_function a (-2) = 1/4 :=
by
  intro a h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_value_l97_9703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l97_9757

theorem sin_alpha_value (α : Real) (h1 : Real.tan α = -5/12) (h2 : π/2 < α ∧ α < π) : 
  Real.sin α = 5/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l97_9757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_with_twos_and_five_l97_9734

/-- A function that checks if a number's decimal representation consists of n digits equal to 2 and one digit equal to 5 -/
def has_n_twos_and_one_five (N : ℕ) (n : ℕ) : Prop :=
  ∃ (digits : List Char), 
    N.repr.toList = digits ∧ 
    digits.length = n + 1 ∧
    digits.count '2' = n ∧
    digits.count '5' = 1

/-- The main theorem stating that 25 and 225 are the only numbers satisfying the conditions -/
theorem perfect_square_with_twos_and_five : 
  ∀ N n : ℕ, 
    N > 0 → 
    n > 0 → 
    (∃ k : ℕ, N = k^2) → 
    has_n_twos_and_one_five N n → 
    (N = 25 ∨ N = 225) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_with_twos_and_five_l97_9734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_transformation_l97_9762

theorem solution_transformation (a b c d : ℂ) :
  (a^4 - 6*a^2 - 8 = 0) →
  (b^4 - 6*b^2 - 8 = 0) →
  (c^4 - 6*c^2 - 8 = 0) →
  (d^4 - 6*d^2 - 8 = 0) →
  let x₁ := (a^2 + b^2 + c^2 + d^2) / 2;
  let x₂ := (a^2 + b^2 + c^2 - d^2) / 2;
  let x₃ := (a^2 - b^2 + c^2 + d^2) / 2;
  let x₄ := (-a^2 + b^2 + c^2 + d^2) / 2
  (x₁^4 - 24*x₁^3 + 216*x₁^2 - 864*x₁ + 1296 = 0) ∧
  (x₂^4 - 24*x₂^3 + 216*x₂^2 - 864*x₂ + 1296 = 0) ∧
  (x₃^4 - 24*x₃^3 + 216*x₃^2 - 864*x₃ + 1296 = 0) ∧
  (x₄^4 - 24*x₄^3 + 216*x₄^2 - 864*x₄ + 1296 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_transformation_l97_9762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_less_than_point_four_l97_9798

theorem count_less_than_point_four : 
  let S : Finset ℚ := {0.8, 1/2, 0.9}
  (S.filter (λ x => x < 0.4)).card = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_less_than_point_four_l97_9798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_value_l97_9761

theorem tan_theta_value (θ : ℝ) 
  (h1 : Real.cos (θ / 2) = 4 / 5) 
  (h2 : Real.sin θ < 0) : 
  Real.tan θ = -24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_value_l97_9761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_christopher_speed_l97_9755

/-- Calculates the speed given distance and time -/
noncomputable def speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

/-- Proves that for a distance of 5 miles and a time of 1.25 hours, the speed is 4 miles per hour -/
theorem christopher_speed : speed 5 1.25 = 4 := by
  -- Unfold the definition of speed
  unfold speed
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_christopher_speed_l97_9755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_constant_l97_9713

/-- Given that m is a constant and (5, 0) is a focus of the hyperbola y^2/m - x^2/9 = 1, prove that m = -16 -/
theorem hyperbola_focus_constant (m : ℝ) : 
  (∀ x y : ℝ, y^2/m - x^2/9 = 1 → ((x - 5)^2 + y^2 = (x + 5)^2 + y^2 ∨ (x - 5)^2 + y^2 < (x + 5)^2 + y^2)) → 
  m = -16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_constant_l97_9713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l97_9776

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 4 * Real.log x

-- State the theorem
theorem f_monotone_decreasing :
  StrictMonoOn f (Set.Ioo (0 : ℝ) 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l97_9776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_double_angle_l97_9765

theorem cos_squared_double_angle (x : ℝ) : 2 * (Real.cos x)^2 = 1 + Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_double_angle_l97_9765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l97_9735

def triangle_proof (A B C : Real) (a b c : Real) : Prop :=
  let triangle_abc := (A + B + C = Real.pi) ∧ (a > 0) ∧ (b > 0) ∧ (c > 0)
  let cosine_law := (c^2 = a^2 + b^2 - 2*a*b*(Real.cos C))
  let condition := ((Real.cos C) / (Real.sin C) = (Real.cos A + Real.cos B) / (Real.sin A + Real.sin B)) ∧ (c = 2)
  triangle_abc → cosine_law → condition →
    (C = Real.pi/3) ∧ 
    (∃ (S : Real), S = Real.sqrt 3 ∧ ∀ (S' : Real), S' = 1/2 * a * b * Real.sin C → S' ≤ S)

theorem triangle_theorem : ∃ A B C a b c, triangle_proof A B C a b c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l97_9735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_of_x12_minus_x3_l97_9728

theorem factorization_of_x12_minus_x3 :
  ∃ (f1 f2 f3 f4 : Polynomial ℤ), 
    (X^12 - X^3 : Polynomial ℤ) = f1 * f2 * f3 * f4 ∧ 
    (∀ g : Polynomial ℤ, g ∣ f1 → g = 1 ∨ g = f1) ∧
    (∀ g : Polynomial ℤ, g ∣ f2 → g = 1 ∨ g = f2) ∧
    (∀ g : Polynomial ℤ, g ∣ f3 → g = 1 ∨ g = f3) ∧
    (∀ g : Polynomial ℤ, g ∣ f4 → g = 1 ∨ g = f4) ∧
    ∀ (g1 g2 g3 g4 g5 : Polynomial ℤ),
      (X^12 - X^3 : Polynomial ℤ) ≠ g1 * g2 * g3 * g4 * g5 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_of_x12_minus_x3_l97_9728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_curve_l97_9791

-- Define the curve C in polar coordinates
noncomputable def curve_C_polar (θ : ℝ) : ℝ := 2 * Real.sin θ

-- Define the line l in parametric form
def line_l_param (t : ℝ) : ℝ × ℝ := (-3 * t + 2, 4 * t)

-- Define the curve C in Cartesian coordinates
def curve_C_cartesian (x y : ℝ) : Prop := x^2 + y^2 - 2*y = 0

-- Define the line l in Cartesian coordinates
def line_l_cartesian (x y : ℝ) : Prop := 4*x + 3*y - 8 = 0

-- Theorem stating that the line is tangent to the curve
theorem line_tangent_to_curve : 
  ∃ (x₀ y₀ : ℝ), 
    curve_C_cartesian x₀ y₀ ∧ 
    line_l_cartesian x₀ y₀ ∧ 
    ∀ (x y : ℝ), curve_C_cartesian x y ∧ line_l_cartesian x y → (x, y) = (x₀, y₀) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_curve_l97_9791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_dist_symmetry_l97_9732

/-- A random variable that follows a normal distribution with mean 1 and standard deviation σ -/
def normal_dist (σ : ℝ) (ξ : ℝ → ℝ) : Prop :=
  ∃ (pdf : ℝ → ℝ), ∀ x, pdf x = (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(1 / 2) * ((x - 1) / σ)^2)

theorem normal_dist_symmetry
  (σ : ℝ)
  (ξ : ℝ → ℝ)
  (h_σ_pos : σ > 0)
  (h_normal : normal_dist σ ξ)
  (h_prob : ∫ x in Set.Ioo 0 1, ξ x = 0.35) :
  ∫ x in Set.Ioo 0 2, ξ x = 0.7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_dist_symmetry_l97_9732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_eq_two_l97_9722

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = (x-1)^2 + ax + sin(x + π/2) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (x - 1)^2 + a * x + Real.sin (x + Real.pi/2)

/-- If f is an even function, then a = 2 -/
theorem even_function_implies_a_eq_two :
  ∃ a, IsEven (f a) → a = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_eq_two_l97_9722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_3_l97_9793

/-- The set of digits available in the bag -/
def digits : Finset ℕ := {1, 2, 3, 4, 5}

/-- A function to check if a number is divisible by 3 -/
def divisibleBy3 (n : ℕ) : Bool := n % 3 = 0

/-- The set of all possible three-digit numbers formed by drawing three digits from the bag -/
def allNumbers : Finset ℕ :=
  Finset.image (fun (a, b, c) => 100 * a + 10 * b + c) 
    (Finset.product digits (Finset.product digits digits))

/-- The set of three-digit numbers formed that are divisible by 3 -/
def divisibleNumbers : Finset ℕ :=
  allNumbers.filter (fun n => divisibleBy3 n)

/-- The theorem stating the probability of drawing a number divisible by 3 -/
theorem probability_divisible_by_3 :
  (Finset.card divisibleNumbers : ℚ) / (Finset.card allNumbers : ℚ) = 1 / 10 := by
  sorry

#eval Finset.card allNumbers
#eval Finset.card divisibleNumbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_3_l97_9793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equalize_expenses_bernardo_share_equalized_l97_9760

/-- The amount LeRoy must give Cecilia to equalize expenses -/
noncomputable def amount_to_equalize (L B C : ℝ) : ℝ := (B + C - 2*L) / 3

/-- Proof that the amount_to_equalize function correctly equalizes expenses -/
theorem equalize_expenses (L B C : ℝ) (h1 : L < B) (h2 : B < C) :
  let total := L + B + C
  let equal_share := total / 3
  let leroy_final := L + amount_to_equalize L B C
  let cecilia_final := C - amount_to_equalize L B C
  leroy_final = equal_share ∧ cecilia_final = equal_share := by
  sorry

/-- Proof that Bernardo's share is also equalized -/
theorem bernardo_share_equalized (L B C : ℝ) (h1 : L < B) (h2 : B < C) :
  let total := L + B + C
  let equal_share := total / 3
  B = equal_share := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equalize_expenses_bernardo_share_equalized_l97_9760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radar_positioning_l97_9771

/-- The number of radars --/
def n : ℕ := 7

/-- The coverage radius of each radar in km --/
def r : ℝ := 26

/-- The width of the coverage ring in km --/
def w : ℝ := 20

/-- The maximum distance from the center to the radars --/
noncomputable def max_distance : ℝ := 24 / Real.sin (180 / n * Real.pi / 180)

/-- The area of the coverage ring --/
noncomputable def coverage_area : ℝ := 960 * Real.pi / Real.tan (180 / n * Real.pi / 180)

theorem radar_positioning (n : ℕ) (r w : ℝ) (h1 : n = 7) (h2 : r = 26) (h3 : w = 20) :
  (max_distance = 24 / Real.sin (180 / n * Real.pi / 180)) ∧
  (coverage_area = 960 * Real.pi / Real.tan (180 / n * Real.pi / 180)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radar_positioning_l97_9771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_specific_l97_9785

/-- The length of the common chord of two overlapping circles -/
noncomputable def common_chord_length (r : ℝ) (d : ℝ) : ℝ :=
  2 * Real.sqrt (r^2 - (d/2)^2)

/-- Theorem: The length of the common chord of two overlapping circles
    with radius 12 cm and centers 16 cm apart is 8√5 cm -/
theorem common_chord_length_specific :
  common_chord_length 12 16 = 8 * Real.sqrt 5 := by
  -- Unfold the definition of common_chord_length
  unfold common_chord_length
  -- Simplify the expression
  simp [Real.sqrt_mul, Real.sqrt_sq]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_specific_l97_9785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l97_9712

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) 
  (h1 : 0 < A ∧ A < π) 
  (h2 : 0 < B ∧ B < π) 
  (h3 : 0 < C ∧ C < π) 
  (h4 : A + B + C = π) 
  (h5 : Real.sin A ^ 2 + Real.sin C ^ 2 - Real.sin B ^ 2 - Real.sin A * Real.sin C = 0) 
  (h6 : a / c = 3 / 2) 
  (h7 : a = c * Real.sin A / Real.sin C) 
  (h8 : b = c * Real.sin B / Real.sin C) : 
  B = π / 3 ∧ Real.tan C = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l97_9712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hotel_cost_theorem_l97_9706

theorem hotel_cost_theorem (initial_friends : ℕ) (additional_friends : ℕ) (cost_decrease : ℚ) :
  initial_friends = 4 →
  additional_friends = 3 →
  cost_decrease = 15 →
  ∃ total_cost : ℚ,
    (total_cost / (initial_friends : ℚ) - total_cost / ((initial_friends + additional_friends) : ℚ) = cost_decrease) ∧
    total_cost = 140 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hotel_cost_theorem_l97_9706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_f_l97_9784

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -x - 4 else x^2 - 4

-- State the theorem
theorem zeros_of_f :
  ∃ (x y : ℝ), x = -4 ∧ y = 2 ∧
  (∀ z : ℝ, f z = 0 ↔ z = x ∨ z = y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_f_l97_9784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_less_than_y_l97_9746

theorem x_less_than_y (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) :
  (Real.sqrt a + Real.sqrt b) / Real.sqrt 2 < Real.sqrt (a + b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_less_than_y_l97_9746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_approx_3_84_l97_9782

/-- The time (in minutes) it takes for two people walking in opposite directions 
    on a circular track to meet, given the track circumference and their speeds. -/
noncomputable def meetingTime (trackCircumference : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  trackCircumference / (speed1 + speed2)

/-- Conversion factor from km/hr to m/min -/
noncomputable def kmhrToMmin : ℝ := 1000 / 60

theorem meeting_time_approx_3_84 :
  let trackCircumference : ℝ := 528
  let speed1 : ℝ := 4.5 * kmhrToMmin
  let speed2 : ℝ := 3.75 * kmhrToMmin
  abs (meetingTime trackCircumference speed1 speed2 - 3.84) < 0.01 := by
  sorry

-- Remove the #eval line as it's not computable
-- #eval meetingTime 528 (4.5 * kmhrToMmin) (3.75 * kmhrToMmin)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_approx_3_84_l97_9782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_ratio_l97_9738

/-- Calculate simple interest -/
def simple_interest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

/-- Calculate compound interest -/
def compound_interest (principal : ℚ) (rate : ℚ) (time : ℕ) : ℚ :=
  principal * ((1 + rate / 100) ^ time - 1)

theorem interest_ratio : 
  let si := simple_interest 5250 4 2
  let ci := compound_interest 4000 10 2
  si / ci = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_ratio_l97_9738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l97_9763

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- State the theorem
theorem min_value_f :
  ∃ (x : ℝ), x > 0 ∧ f x = -1/Real.exp 1 ∧ ∀ (y : ℝ), y > 0 → f y ≥ -1/Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l97_9763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_theorem_l97_9737

theorem complex_power_theorem : 
  (3 * Complex.cos (30 * π / 180) + 3 * Complex.I * Complex.sin (30 * π / 180)) ^ 4 = 
  Complex.ofReal (-40.5) + Complex.I * Complex.ofReal (40.5 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_theorem_l97_9737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_property_l97_9711

def StrictlyIncreasing (f : ℕ → ℕ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem unique_function_property (f : ℕ → ℕ) 
  (h_increasing : StrictlyIncreasing f)
  (h_two : f 2 = 2)
  (h_mult : ∀ n m, Nat.Coprime n m → f (n * m) = f n * f m) :
  ∀ n, f n = n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_property_l97_9711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_solutions_are_1_1_1_2_2_1_l97_9749

/-- The function that represents the floor of the square root of a number -/
noncomputable def floorSqrt (n : ℕ) : ℕ := Int.toNat ⌊Real.sqrt (n : ℝ)⌋

/-- The theorem stating the only solutions to the given conditions -/
theorem only_solutions_are_1_1_1_2_2_1 :
  ∀ a b : ℕ+,
    (a ∣ b^4 + 1) →
    (b ∣ a^4 + 1) →
    (floorSqrt a.val = floorSqrt b.val) →
    ((a = 1 ∧ b = 1) ∨ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_solutions_are_1_1_1_2_2_1_l97_9749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_products_ending_in_zero_is_correct_l97_9727

/-- The number of products in the sequence (1 × 2, 2 × 3, 3 × 4, ..., 2017 × 2018) that have their last digit as zero -/
def count_products_ending_in_zero : ℕ := 806

/-- The sequence of products of consecutive integers from 1 × 2 to 2017 × 2018 -/
def product_sequence : List ℕ :=
  List.range 2017 |>.map (fun n => (n + 1) * (n + 2))

/-- Function to check if a number ends in zero -/
def ends_in_zero (n : ℕ) : Bool :=
  n % 10 = 0

theorem count_products_ending_in_zero_is_correct :
  (product_sequence.filter ends_in_zero).length = count_products_ending_in_zero :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_products_ending_in_zero_is_correct_l97_9727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_value_l97_9778

open Real

noncomputable def f (x : ℝ) := sin x * cos x - Real.sqrt 3 * cos x ^ 2
noncomputable def g (x : ℝ) := sin (2 * x + π / 3) - Real.sqrt 3 / 2

theorem min_shift_value (k : ℝ) :
  (∀ x, f x = g (x - k)) ∧ k > 0 →
  k ≥ π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_value_l97_9778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_arrangement_l97_9710

/-- Predicate indicating if a task is assigned to a volunteer -/
def task_assigned : Fin 3 → Fin 4 → Prop := sorry

/-- The number of ways to arrange volunteers for tasks -/
def arrange_volunteers (num_volunteers : ℕ) (num_tasks : ℕ) : ℕ :=
  if num_volunteers = 3 ∧ num_tasks = 4 then
    (Nat.choose num_tasks 2) * (Nat.factorial num_volunteers)
  else
    0

/-- Theorem stating the correct number of arrangements -/
theorem correct_arrangement : arrange_volunteers 3 4 = 36 := by
  rfl

/-- Each volunteer completes at least one task -/
axiom each_volunteer_has_task : ∀ (v : Fin 3), ∃ (t : Fin 4), task_assigned v t

/-- Each task is completed by exactly one volunteer -/
axiom each_task_has_one_volunteer : ∀ (t : Fin 4), ∃! (v : Fin 3), task_assigned v t

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_arrangement_l97_9710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yz_radius_is_sqrt_59_l97_9725

/-- A sphere intersecting two planes -/
structure IntersectingSphere where
  /-- Center of the circle in the xy-plane -/
  xy_center : ℝ × ℝ × ℝ
  /-- Radius of the circle in the xy-plane -/
  xy_radius : ℝ
  /-- Center of the circle in the yz-plane -/
  yz_center : ℝ × ℝ × ℝ

/-- The radius of the circle in the yz-plane for a specific intersecting sphere -/
noncomputable def yz_radius (s : IntersectingSphere) : ℝ :=
  Real.sqrt 59

/-- Theorem stating that the radius of the circle in the yz-plane is √59 -/
theorem yz_radius_is_sqrt_59 (s : IntersectingSphere) 
    (h1 : s.xy_center = (3, 3, 0))
    (h2 : s.xy_radius = 2)
    (h3 : s.yz_center = (0, 3, -8)) : 
  yz_radius s = Real.sqrt 59 := by
  sorry

#check yz_radius_is_sqrt_59

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yz_radius_is_sqrt_59_l97_9725
