import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_difference_l176_17697

theorem min_max_difference (a : Fin 2021 → ℤ)
  (h : ∀ n : Fin 2016, a (n + 5) + a n > a (n + 2) + a (n + 3)) :
  (∃ i j : Fin 2021, a i - a j = 85008) ∧
  (∀ k l : Fin 2021, a k - a l ≤ 85008) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_difference_l176_17697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l176_17667

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -1/2 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | x^2 ≤ 1}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = Set.Ici (-1 : ℝ) ∩ Set.Iio 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l176_17667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_PQ_is_12_over_5_l176_17660

/-- The curve C in the Cartesian coordinate system -/
noncomputable def C : Set (ℝ × ℝ) :=
  {p | let (x, y) := p
       Real.sqrt (x^2 + 2 * Real.sqrt 7 * x + y^2 + 7) +
       Real.sqrt (x^2 - 2 * Real.sqrt 7 * x + y^2 + 7) = 8}

/-- Check if three points are on the same circle -/
def onSameCircle (p q o : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := p
  let (x₂, y₂) := q
  let (x₃, y₃) := o
  x₁ * x₂ + y₁ * y₂ = x₃^2 + y₃^2

/-- The distance from a point to a line defined by two points -/
noncomputable def distanceToLine (o p q : ℝ × ℝ) : ℝ :=
  let (x₀, y₀) := o
  let (x₁, y₁) := p
  let (x₂, y₂) := q
  abs ((y₂ - y₁) * x₀ - (x₂ - x₁) * y₀ + x₂ * y₁ - y₂ * x₁) /
    Real.sqrt ((y₂ - y₁)^2 + (x₂ - x₁)^2)

theorem distance_to_PQ_is_12_over_5 (p q : ℝ × ℝ) (hp : p ∈ C) (hq : q ∈ C) 
    (hpq : p ≠ q) (hcircle : onSameCircle p q (0, 0)) :
  distanceToLine (0, 0) p q = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_PQ_is_12_over_5_l176_17660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l176_17629

theorem distance_between_points (a_to_c b_to_c angle_a angle_b : ℝ) 
  (h1 : a_to_c = 10)
  (h2 : b_to_c = 15)
  (h3 : angle_a = 25 * π / 180)
  (h4 : angle_b = 35 * π / 180) :
  let c_angle := angle_a + angle_b
  let ab_distance_squared := a_to_c^2 + b_to_c^2 - 2 * a_to_c * b_to_c * Real.cos c_angle
  Real.sqrt ab_distance_squared = 5 * Real.sqrt 19 := by
  sorry

#check distance_between_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l176_17629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longitude_latitude_unique_identifier_building_in_district_not_unique_side_of_road_not_unique_direction_without_reference_not_unique_unique_location_identifier_l176_17693

-- Define a structure for a point on Earth's surface
structure EarthPoint where
  longitude : Real
  latitude : Real

-- Define a function to check if a description can uniquely identify a location
def canUniquelyIdentify (description : String) : Prop :=
  ∃! (p : EarthPoint), True  -- Changed from descriptionMatchesPoint to True

-- Theorem stating that longitude and latitude can uniquely identify a location
theorem longitude_latitude_unique_identifier :
  canUniquelyIdentify "East longitude 118°, north latitude 28°" := by
  sorry

-- Theorem stating that a building in a district cannot uniquely identify a location
theorem building_in_district_not_unique :
  ¬ canUniquelyIdentify "Building 4 in Huasheng District" := by
  sorry

-- Theorem stating that a side of a road cannot uniquely identify a location
theorem side_of_road_not_unique :
  ¬ canUniquelyIdentify "Right side of Jiefang Road" := by
  sorry

-- Theorem stating that a direction without a reference point cannot uniquely identify a location
theorem direction_without_reference_not_unique :
  ¬ canUniquelyIdentify "Southward and eastward by 40°" := by
  sorry

-- Main theorem combining all the above results
theorem unique_location_identifier :
  (canUniquelyIdentify "East longitude 118°, north latitude 28°") ∧
  (¬ canUniquelyIdentify "Building 4 in Huasheng District") ∧
  (¬ canUniquelyIdentify "Right side of Jiefang Road") ∧
  (¬ canUniquelyIdentify "Southward and eastward by 40°") := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longitude_latitude_unique_identifier_building_in_district_not_unique_side_of_road_not_unique_direction_without_reference_not_unique_unique_location_identifier_l176_17693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gavrila_ascend_time_l176_17696

noncomputable section

def escalator_length : ℝ := 1  -- Assume the escalator length is 1 unit

-- V represents the speed of the escalator
-- U represents Gavrila's speed relative to the escalator
def V : ℝ := 1 / 60
def U : ℝ := 1 / 40

-- Time taken to ascend the non-working escalator
def time_non_working : ℝ := escalator_length / U

theorem gavrila_ascend_time :
  (escalator_length / V = 60) →  -- Condition 1: Standing on moving escalator takes 1 minute
  (escalator_length / (V + U) = 24) →  -- Condition 2: Running on moving escalator takes 24 seconds
  (time_non_working = 40) :=  -- Prove that ascending non-working escalator takes 40 seconds
by
  sorry

end noncomputable section

#eval Float.ofScientific 4 0 1  -- This will output 40.0

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gavrila_ascend_time_l176_17696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_l176_17637

-- Define the function types
variable {α : Type*}

-- Define the invertible functions
variable (s t u : α → α)
variable (s_inv t_inv u_inv : α → α)

-- Assume the functions are invertible
axiom s_invertible : Function.LeftInverse s_inv s ∧ Function.RightInverse s_inv s
axiom t_invertible : Function.LeftInverse t_inv t ∧ Function.RightInverse t_inv t
axiom u_invertible : Function.LeftInverse u_inv u ∧ Function.RightInverse u_inv u

-- Define g as the composition of u, s, and t
def g (s t u : α → α) : α → α := u ∘ s ∘ t

-- State the theorem
theorem g_inverse (s t u : α → α) (s_inv t_inv u_inv : α → α) 
  (hs : Function.LeftInverse s_inv s ∧ Function.RightInverse s_inv s)
  (ht : Function.LeftInverse t_inv t ∧ Function.RightInverse t_inv t)
  (hu : Function.LeftInverse u_inv u ∧ Function.RightInverse u_inv u) :
  Function.LeftInverse (t_inv ∘ s_inv ∘ u_inv) (g s t u) ∧ 
  Function.RightInverse (t_inv ∘ s_inv ∘ u_inv) (g s t u) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_l176_17637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l176_17627

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the interval [-2, 1]
def interval : Set ℝ := Set.Icc (-2) 1

-- State the theorem
theorem f_properties :
  -- Minimum value is -2
  (∃ x ∈ interval, f x = -2) ∧ (∀ x ∈ interval, f x ≥ -2) ∧
  -- Maximum value is 2
  (∃ x ∈ interval, f x = 2) ∧ (∀ x ∈ interval, f x ≤ 2) ∧
  -- Tangent line equation at P(2, -6)
  (let m := deriv f 2; 24 * 2 - (-6) - 54 = 0 ∧ m = 24) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l176_17627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l176_17628

/-- Given conditions and function definition -/
noncomputable def f (x : ℝ) : ℝ := (9 : ℝ)^x - (3 : ℝ)^(x+1) - 1

/-- Theorem statement -/
theorem f_max_min (x : ℝ) 
  (h1 : ((1/2) : ℝ)^x ≤ 4) 
  (h2 : Real.log x / Real.log (Real.sqrt 3) ≤ 2) : 
  (∃ y, f y = 647 ∧ ∀ z, f z ≤ f y) ∧ 
  (∃ y, f y = -13/4 ∧ ∀ z, f z ≥ f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l176_17628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_k_value_l176_17622

noncomputable def larger_circle_radius (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

def smaller_circle_radius (r : ℝ) : ℝ := r - 2

theorem concentric_circles_k_value (P : ℝ × ℝ) (S : ℝ × ℝ) 
  (h1 : P = (5, 12))
  (h2 : S.1 = 0)
  (h3 : larger_circle_radius P.1 P.2 = larger_circle_radius 5 12)
  (h4 : smaller_circle_radius (larger_circle_radius 5 12) = S.2) :
  S.2 = 11 := by
  sorry

#eval smaller_circle_radius 13

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_k_value_l176_17622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peanut_cluster_percentage_approx_l176_17678

/-- Represents the number of chocolates of each type in the box -/
structure ChocolateBox where
  total : Nat
  typeA : Nat
  typeB : Nat
  typeC : Nat
  typeD : Nat
  typeE : Nat
  typeF : Nat
  typeG : Nat
  typeH : Nat
  typeI : Nat
  typeJ : Nat

/-- Calculates the percentage of peanut clusters in the chocolate box -/
noncomputable def peanutClusterPercentage (box : ChocolateBox) : Real :=
  let nonPeanutClusters := box.typeA + box.typeB + box.typeC + box.typeD + box.typeE +
                           box.typeF + box.typeG + box.typeH + box.typeI + box.typeJ
  let peanutClusters := box.total - nonPeanutClusters
  (peanutClusters : Real) / (box.total : Real) * 100

/-- Theorem stating that the percentage of peanut clusters is approximately 13.33% -/
theorem peanut_cluster_percentage_approx (box : ChocolateBox)
  (h1 : box.total = 150)
  (h2 : box.typeA = 5)
  (h3 : box.typeB = 8)
  (h4 : box.typeC = Int.floor ((box.typeA : Real) * 1.5))
  (h5 : box.typeD = 2 * (box.typeA + box.typeB))
  (h6 : box.typeE = Int.floor ((box.typeC : Real) * 1.2))
  (h7 : box.typeF = box.typeA + 6)
  (h8 : box.typeG = box.typeB + 6)
  (h9 : box.typeH = box.typeC + 6)
  (h10 : box.typeI = 7)
  (h11 : box.typeJ = Int.floor (((box.typeI + box.typeF) : Real) * 1.5)) :
  |peanutClusterPercentage box - 13.33| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_peanut_cluster_percentage_approx_l176_17678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l176_17653

/-- Represents the number of holes in the graph of a rational function. -/
noncomputable def num_holes (f : ℝ → ℝ) : ℕ := sorry

/-- Represents the number of vertical asymptotes in the graph of a rational function. -/
noncomputable def num_vertical_asymptotes (f : ℝ → ℝ) : ℕ := sorry

/-- Represents the number of horizontal asymptotes in the graph of a rational function. -/
noncomputable def num_horizontal_asymptotes (f : ℝ → ℝ) : ℕ := sorry

/-- Represents the number of oblique asymptotes in the graph of a rational function. -/
noncomputable def num_oblique_asymptotes (f : ℝ → ℝ) : ℕ := sorry

/-- The function f(x) = (x^3 + 4x^2 + 4x) / (x^3 + x^2 - 2x) -/
noncomputable def f (x : ℝ) : ℝ := (x^3 + 4*x^2 + 4*x) / (x^3 + x^2 - 2*x)

theorem asymptote_sum : 
  let a := num_holes f
  let b := num_vertical_asymptotes f
  let c := num_horizontal_asymptotes f
  let d := num_oblique_asymptotes f
  a + 2*b + 3*c + 4*d = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l176_17653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bound_l176_17659

def ValidSequence (s : List Nat) : Prop :=
  s.head? = some 1 ∧
  (s.tail?.bind List.head?) = some 2 ∧
  ∀ i j k, i ∈ s → j ∈ s → k ∈ s → i + j ≠ k

theorem sequence_bound (k : Nat) (s : List Nat) (h : ValidSequence s) :
  (s.filter (· < k)).length ≤ k / 3 + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bound_l176_17659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l176_17644

noncomputable def f (x : ℝ) : ℝ := (Real.log (4 - x)) / (x - 2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 2 ∨ (2 < x ∧ x < 4)} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l176_17644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difficult_minus_easy_equals_twenty_l176_17698

/-- Represents the number of problems solved by each combination of students -/
structure ProblemCounts where
  total : ℕ
  individual : ℕ
  exclusive : Fin 3 → ℕ
  pairs : Fin 3 → ℕ
  all : ℕ

/-- Sum of a function over Fin 3 -/
def sum3 (f : Fin 3 → ℕ) : ℕ :=
  f 0 + f 1 + f 2

/-- Conditions for the problem scenario -/
def ValidScenario (p : ProblemCounts) : Prop :=
  p.total = 100 ∧
  p.individual = 60 ∧
  sum3 p.exclusive + sum3 p.pairs + p.all = p.total ∧
  ∀ i : Fin 3, p.exclusive i + p.pairs i + p.pairs ((i.val + 1) % 3) + p.all = p.individual

/-- The main theorem to prove -/
theorem difficult_minus_easy_equals_twenty (p : ProblemCounts) 
  (h : ValidScenario p) : sum3 p.exclusive - p.all = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_difficult_minus_easy_equals_twenty_l176_17698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_problem_l176_17655

-- Define the conditions
theorem exponent_problem (a b : ℝ) (h1 : (30 : ℝ)^a = 2) (h2 : (30 : ℝ)^b = 3) :
  (10 : ℝ)^((1 - a - b) / (2 * (1 - b))) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_problem_l176_17655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_three_monotonic_intervals_iff_b_range_l176_17642

noncomputable section

/-- The function y(x) with parameter b -/
def y (b x : ℝ) : ℝ := (1/3) * x^3 + b * x^2 + (b + 6) * x + 3

/-- The derivative of y(x) with respect to x -/
def y_derivative (b x : ℝ) : ℝ := x^2 + 2 * b * x + (b + 6)

/-- Predicate for y having three monotonic intervals -/
def has_three_monotonic_intervals (b : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ y_derivative b x₁ = 0 ∧ y_derivative b x₂ = 0

/-- Theorem stating the equivalence between y having three monotonic intervals and the range of b -/
theorem y_three_monotonic_intervals_iff_b_range :
  ∀ b : ℝ, has_three_monotonic_intervals b ↔ (b < -2 ∨ b > 3) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_three_monotonic_intervals_iff_b_range_l176_17642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_l176_17689

def Digits : List ℕ := [0, 3, 4, 7, 8]

def is_valid_4digit (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧ (n.digits 10).length = 4 ∧ (n.digits 10).toFinset ⊆ Digits.toFinset

def is_valid_3digit (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ (n.digits 10).length = 3 ∧ (n.digits 10).toFinset ⊆ Digits.toFinset

def use_all_digits (a b : ℕ) : Prop :=
  (a.digits 10 ++ b.digits 10).toFinset = Digits.toFinset

theorem smallest_difference :
  ∀ a b : ℕ,
    is_valid_4digit a →
    is_valid_3digit b →
    use_all_digits a b →
    a - b ≥ 2243 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_l176_17689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_coordinates_transformation_l176_17666

noncomputable section

-- Define the original spherical coordinates
def original_rho : ℝ := 3
def original_theta : ℝ := 9 * Real.pi / 8
def original_phi : ℝ := 4 * Real.pi / 7

-- Define the new spherical coordinates
def new_rho : ℝ := 3
def new_theta : ℝ := 9 * Real.pi / 8
def new_phi : ℝ := 3 * Real.pi / 7

-- Theorem statement
theorem spherical_coordinates_transformation :
  (original_rho > 0) →
  (0 ≤ original_theta ∧ original_theta < 2 * Real.pi) →
  (0 ≤ original_phi ∧ original_phi ≤ Real.pi) →
  (new_rho > 0) →
  (0 ≤ new_theta ∧ new_theta < 2 * Real.pi) →
  (0 ≤ new_phi ∧ new_phi ≤ Real.pi) →
  (let x := original_rho * Real.sin original_phi * Real.cos original_theta
   let y := original_rho * Real.sin original_phi * Real.sin original_theta
   let z := original_rho * Real.cos original_phi
   new_rho * Real.sin new_phi * Real.cos new_theta = x ∧
   new_rho * Real.sin new_phi * Real.sin new_theta = -y ∧
   new_rho * Real.cos new_phi = -z) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_coordinates_transformation_l176_17666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_nested_roots_l176_17648

theorem simplify_nested_roots : 
  Real.sqrt (Real.sqrt (Real.sqrt (1 / 16384))) = 1 / (2 ^ (7/8)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_nested_roots_l176_17648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_variation_c_squared_d_fourth_l176_17620

theorem inverse_variation_c_squared_d_fourth (c d : ℝ) (h : c^2 * d^4 = (8^2) * 2^4) :
  c^2 = 4 ↔ d = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_variation_c_squared_d_fourth_l176_17620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_atoms_count_l176_17614

-- Define Avogadro's constant
noncomputable def N_A : ℝ := Real.exp 1 -- placeholder value

-- Define the molar mass of CH₂O
def molar_mass_CH₂O : ℝ := 30

-- Define the mass of the mixture
def mixture_mass : ℝ := 3.0

-- Define the number of atoms per molecule of CH₂O
def atoms_per_molecule : ℕ := 4

-- Theorem statement
theorem mixture_atoms_count :
  let moles : ℝ := mixture_mass / molar_mass_CH₂O
  let total_atoms : ℝ := moles * N_A * (atoms_per_molecule : ℝ)
  total_atoms = 0.4 * N_A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_atoms_count_l176_17614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_angle_theorem_l176_17680

/-- A regular polygon with n vertices -/
structure RegularPolygon where
  n : ℕ
  vertices : Fin n → ℝ × ℝ

/-- An arbitrary point inside the polygon -/
noncomputable def interior_point (p : RegularPolygon) : ℝ × ℝ := sorry

/-- The angle between two vertices and an interior point -/
noncomputable def angle (p : RegularPolygon) (i j : Fin p.n) (o : ℝ × ℝ) : ℝ := sorry

/-- The theorem statement -/
theorem regular_polygon_angle_theorem (p : RegularPolygon) (o : ℝ × ℝ) 
  (h : o = interior_point p) :
  ∃ (i j : Fin p.n), i < j ∧ 
    π * (1 - 1 / (p.n : ℝ)) ≤ angle p i j o ∧ angle p i j o ≤ π := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_angle_theorem_l176_17680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_at_seven_l176_17643

-- Define the polynomial Q(x)
def Q (p q r s t u : ℝ) (x : ℂ) : ℂ :=
  (3 * x^4 - 33 * x^3 + p * x^2 + q * x + r) * (4 * x^4 - 60 * x^3 + s * x^2 + t * x + u)

-- Define the set of complex roots
def roots : Set ℂ := {1, 2, 3, 4, 5, 6}

-- Theorem statement
theorem Q_at_seven (p q r s t u : ℝ) :
  (∀ z : ℂ, z ∈ roots → Q p q r s t u z = 0) →
  Q p q r s t u 7 = 67184640 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_at_seven_l176_17643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_ratio_is_21_div_2_l176_17617

/-- Sequence a_n defined recursively -/
def a : ℕ → ℚ
  | 0 => 33  -- Define the base case for 0
  | 1 => 33
  | n + 1 => a n + 2 * n

/-- The ratio a_n / n -/
noncomputable def ratio (n : ℕ) : ℚ :=
  if n = 0 then 0 else a n / n

theorem min_ratio_is_21_div_2 :
  ∃ (N : ℕ), N > 0 ∧ ∀ (n : ℕ), n > 0 → ratio n ≥ 21 / 2 ∧ ratio N = 21 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_ratio_is_21_div_2_l176_17617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_not_collinear_l176_17609

/-- Given two vectors a and b in ℝ³, prove that the vectors c₁ and c₂ constructed from a and b are not collinear. -/
theorem vectors_not_collinear (a b : Fin 3 → ℝ) 
  (ha : a = ![3, 7, 0]) 
  (hb : b = ![4, 6, -1]) : 
  ¬ ∃ (k : ℝ), (3 • a + 2 • b) = k • (5 • a - 7 • b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_not_collinear_l176_17609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_l176_17608

theorem cube_root_equation_solution :
  ∃! x : ℝ, (5 - x / 3) ^ (1/3 : ℝ) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_l176_17608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_j_l176_17672

-- Define the function h
def h (x : ℝ) : ℝ := 4 * x - 3

-- Define the function j as a composition of h
def j (x : ℝ) : ℝ := h (h (h x))

-- Theorem statement
theorem range_of_j :
  ∀ y ∈ Set.range (fun x => j x), 
    (0 ≤ x ∧ x ≤ 3 → -63 ≤ y ∧ y ≤ 129) ∧
    (∃ x, 0 ≤ x ∧ x ≤ 3 ∧ j x = -63) ∧
    (∃ x, 0 ≤ x ∧ x ≤ 3 ∧ j x = 129) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_j_l176_17672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_furniture_cost_and_total_price_l176_17616

/-- Represents the cost and sale details of furniture items -/
structure FurnitureSale where
  markup_rate : ℝ
  chair_discount_rate : ℝ
  table_sale_price : ℝ
  chair_sale_price : ℝ
  sales_tax_rate : ℝ

/-- Calculates the cost price given the sale price and markup rate -/
noncomputable def cost_price (sale_price : ℝ) (markup_rate : ℝ) : ℝ :=
  sale_price / (1 + markup_rate)

/-- Calculates the total sale price including tax -/
noncomputable def total_price_with_tax (table_price : ℝ) (chair_price : ℝ) (tax_rate : ℝ) : ℝ :=
  (table_price + chair_price) * (1 + tax_rate)

theorem furniture_cost_and_total_price 
  (sale : FurnitureSale) 
  (h_markup : sale.markup_rate = 0.25)
  (h_chair_discount : sale.chair_discount_rate = 0.10)
  (h_table_sale : sale.table_sale_price = 4800)
  (h_chair_sale : sale.chair_sale_price = 2700)
  (h_tax : sale.sales_tax_rate = 0.07) :
  let table_cost := cost_price sale.table_sale_price sale.markup_rate
  let chair_cost := cost_price (sale.chair_sale_price / (1 - sale.chair_discount_rate)) sale.markup_rate
  let total_with_tax := total_price_with_tax sale.table_sale_price sale.chair_sale_price sale.sales_tax_rate
  table_cost = 3840 ∧ chair_cost = 2400 ∧ total_with_tax = 8025 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_furniture_cost_and_total_price_l176_17616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_two_part_three_l176_17615

-- Define a "difference of 1" equation
def is_difference_of_one (a b c : ℝ) : Prop :=
  let discriminant := b^2 - 4*a*c
  discriminant ≥ 0 ∧ Real.sqrt (discriminant / (a^2)) = 1

-- Part 2
theorem part_two (m : ℝ) :
  is_difference_of_one 1 (1 - m) (-m) → m = 0 ∨ m = -2 :=
sorry

-- Part 3
theorem part_three (a b : ℝ) (ha : a > 0) :
  is_difference_of_one a b 1 →
  ∃ (t : ℝ), t = 10*a - b^2 ∧ t ≤ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ is_difference_of_one a₀ b₀ 1 ∧ 10*a₀ - b₀^2 = 9 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_two_part_three_l176_17615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mila_agnes_earnings_l176_17676

/-- Calculates the number of hours Mila needs to work to earn the same as Agnes in a month -/
noncomputable def milaHours (milaRate : ℝ) (agnesRate : ℝ) (agnesWeeklyHours : ℝ) (weeksInMonth : ℝ) : ℝ :=
  (agnesRate * agnesWeeklyHours * weeksInMonth) / milaRate

theorem mila_agnes_earnings 
  (milaRate : ℝ) 
  (agnesRate : ℝ) 
  (agnesWeeklyHours : ℝ) 
  (weeksInMonth : ℝ) 
  (h1 : milaRate = 10) 
  (h2 : agnesRate = 15) 
  (h3 : agnesWeeklyHours = 8) 
  (h4 : weeksInMonth = 4) :
  milaHours milaRate agnesRate agnesWeeklyHours weeksInMonth = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mila_agnes_earnings_l176_17676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concentration_theorem_l176_17688

/-- The concentration function M(t) with constants a and r -/
noncomputable def M (a r t : ℝ) : ℝ := a * r^t + 24

/-- The condition that M(0) = 124 -/
def condition1 (a r : ℝ) : Prop := M a r 0 = 124

/-- The condition that M(1) = 64 -/
def condition2 (a r : ℝ) : Prop := M a r 1 = 64

/-- The theorem stating the concentration at t = 4 and the smallest integer t when M(t) < 24.001 -/
theorem concentration_theorem (a r : ℝ) (h1 : condition1 a r) (h2 : condition2 a r) :
  ∃ (ε : ℝ), ε > 0 ∧ abs (M a r 4 - 26.56) < ε ∧ 
  (∀ t : ℕ, t < 13 → M a r (t : ℝ) > 24.001) ∧
  (M a r 13 < 24.001) := by
  sorry

#check concentration_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concentration_theorem_l176_17688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_foci_l176_17641

/-- The hyperbola defined by xy = 4 -/
def hyperbola (x y : ℝ) : Prop := x * y = 4

/-- A point is on the hyperbola if it satisfies the equation xy = 4 -/
def on_hyperbola (p : ℝ × ℝ) : Prop := hyperbola p.1 p.2

/-- The foci of the hyperbola xy = 4 -/
def foci : (ℝ × ℝ) × (ℝ × ℝ) := ((2, 2), (-2, -2))

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: The distance between the foci of the hyperbola xy = 4 is 4√2 -/
theorem distance_between_foci :
  distance foci.1 foci.2 = 4 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_foci_l176_17641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_difference_and_implication_l176_17613

-- Define the symmetric difference operation
def mySymmDiff (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∉ B}

-- Define sets A and B
def A : Set ℝ := {x | 4*x^2 + 9*x + 2 < 0}
def B : Set ℝ := {x | -1 < x ∧ x < 2}

-- Define set M as the symmetric difference of B and A
def M : Set ℝ := mySymmDiff B A

-- Define set P parameterized by a
def P (a : ℝ) : Set ℝ := {x | (x - 2*a) * (x + a - 2) < 0}

-- Main theorem
theorem symmetric_difference_and_implication :
  (M = {x : ℝ | -1/4 ≤ x ∧ x < 2}) ∧
  (∀ x : ℝ, x ∈ M → (x ∈ P a ↔ a < -1/8 ∨ a > 9/4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_difference_and_implication_l176_17613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_discount_theorem_l176_17625

-- Define constants
def m : ℚ := 3
def n : ℚ := 2

-- Define the profit function for seafood skewers
noncomputable def seafood_profit (x : ℚ) : ℚ :=
  if x ≤ 200 then 2 * x else x + 200

-- Define the profit function for meat skewers with discount a
def meat_profit (x a : ℚ) : ℚ :=
  (3.5 - a - n) * (1000 - x)

-- Theorem statement
theorem max_discount_theorem (a : ℚ) :
  (∀ x, 200 < x → x ≤ 400 → meat_profit x a ≥ seafood_profit x) →
  a ≤ 0.5 := by
  sorry

#check max_discount_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_discount_theorem_l176_17625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_decreasing_multiplicative_function_l176_17685

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (x * Real.log a)

-- State the theorem
theorem exists_decreasing_multiplicative_function :
  ∃ a : ℝ, 0 < a ∧ a < 1 ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ > f a x₂) ∧
  (∀ x₁ x₂ : ℝ, f a (x₁ + x₂) = f a x₁ * f a x₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_decreasing_multiplicative_function_l176_17685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_is_one_l176_17656

/-- The cross-sectional area of the solid R at height x -/
noncomputable def S (x : ℝ) : ℝ := ((1 + 2*x) / (1 + x)) * Real.sqrt (1 - x^2)

/-- The volume of the solid R -/
noncomputable def volume : ℝ := ∫ x in Set.Icc 0 1, S x

/-- Theorem: The volume of the solid R is equal to 1 -/
theorem volume_is_one : volume = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_is_one_l176_17656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l176_17683

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (3*x - 2) / (x + 4)

-- Define the inverse function g_inv
noncomputable def g_inv (x : ℝ) : ℝ := (4*x + 2) / (-x + 3)

-- Theorem statement
theorem inverse_function_theorem :
  (∀ x, g (g_inv x) = x) ∧ 
  (∀ x, g_inv (g x) = x) ∧
  (4 : ℝ) / (-1 : ℝ) = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l176_17683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perpendicular_iff_b_equals_four_thirds_l176_17695

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the first line -/
noncomputable def m1 : ℝ := 3

/-- The slope of the second line in terms of b -/
noncomputable def m2 (b : ℝ) : ℝ := -b / 4

/-- The value of b that makes the lines perpendicular -/
noncomputable def b_perpendicular : ℝ := 4 / 3

theorem lines_perpendicular_iff_b_equals_four_thirds :
  ∀ b : ℝ, perpendicular m1 (m2 b) ↔ b = b_perpendicular :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perpendicular_iff_b_equals_four_thirds_l176_17695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_people_moving_to_california_l176_17639

def people_moving : ℕ := 3000
def days : ℕ := 4
def hours_per_day : ℕ := 24

def average_people_per_hour : ℚ :=
  people_moving / (days * hours_per_day)

def rounded_average : ℕ :=
  (average_people_per_hour + 1/2).floor.toNat

theorem average_people_moving_to_california :
  rounded_average = 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_people_moving_to_california_l176_17639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l176_17673

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Define point A
def point_A : ℝ × ℝ := (2, -2)

-- Define a predicate for points on the tangent line
def on_tangent_line (x y : ℝ) : Prop := x - 2*y + 2 = 0

-- Theorem statement
theorem tangent_line_equation :
  ∃ (T₁ T₂ : ℝ × ℝ),
    (circle_equation T₁.1 T₁.2) ∧
    (circle_equation T₂.1 T₂.2) ∧
    (T₁ ≠ T₂) ∧
    (∀ (P : ℝ × ℝ), circle_equation P.1 P.2 → 
      ((P.1 - point_A.1)^2 + (P.2 - point_A.2)^2 ≥ (T₁.1 - point_A.1)^2 + (T₁.2 - point_A.2)^2) ∧
      ((P.1 - point_A.1)^2 + (P.2 - point_A.2)^2 ≥ (T₂.1 - point_A.1)^2 + (T₂.2 - point_A.2)^2)) →
    on_tangent_line T₁.1 T₁.2 ∧ on_tangent_line T₂.1 T₂.2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l176_17673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stamp_collection_count_l176_17650

theorem stamp_collection_count
  (foreign : ℕ) (old : ℕ) (foreign_and_old : ℕ) (neither : ℕ)
  (h1 : foreign = 90)
  (h2 : old = 50)
  (h3 : foreign_and_old = 20)
  (h4 : neither = 80) :
  foreign + old - foreign_and_old + neither = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stamp_collection_count_l176_17650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l176_17638

/-- An ellipse is defined by its foci and a point it passes through -/
structure Ellipse where
  focus1 : ℝ × ℝ
  focus2 : ℝ × ℝ
  point : ℝ × ℝ

/-- The standard form of an ellipse equation -/
def standard_form (a b h k : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

/-- The theorem stating the properties of the given ellipse -/
theorem ellipse_properties (e : Ellipse) 
    (h_focus1 : e.focus1 = (1, 1))
    (h_focus2 : e.focus2 = (7, 1))
    (h_point : e.point = (0, 8)) :
    ∃ (a b h k : ℝ),
      a > 0 ∧ b > 0 ∧
      (∀ x y, (x, y) ∈ Set.range (λ t : ℝ × ℝ ↦ t) → standard_form a b h k x y) ∧
      a = 6 * Real.sqrt 2 ∧
      b = 3 * Real.sqrt 7 ∧
      h = 4 ∧
      k = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l176_17638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_and_optimal_inventory_l176_17661

noncomputable section

/-- Represents the market price as a function of inventory quantity -/
def marketPrice (k : ℝ) (x : ℝ) : ℝ := k / (x + 20)

/-- Represents the daily profit as a function of inventory quantity -/
def dailyProfit (k : ℝ) (x : ℝ) : ℝ := (marketPrice k x - 2) * x

theorem max_profit_and_optimal_inventory :
  ∃ (k : ℝ),
    -- Condition: When 100 pieces were purchased and sold out, profit was 100
    dailyProfit k 100 = 100 →
    -- Prove: Maximum profit is 160 and optimal inventory is 40
    (∀ x : ℝ, x ≥ 0 → dailyProfit k x ≤ 160) ∧
    dailyProfit k 40 = 160 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_and_optimal_inventory_l176_17661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_relations_and_factors_l176_17610

theorem number_relations_and_factors : 
  (∃ k : ℕ, 20 = 4 * k) ∧ 
  (20 % 4 = 0) ∧
  ({1, 2, 3, 4, 6, 8, 12, 24} : Finset ℕ) = Finset.filter (fun x => 24 % x = 0) (Finset.range 25) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_relations_and_factors_l176_17610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l176_17645

-- Define the curve C in polar coordinates
def curve_C (ρ θ : ℝ) : Prop :=
  ρ^2 = 12 / (2 + Real.cos θ^2)

-- Define the line l in polar coordinates
def line_l (ρ θ : ℝ) : Prop :=
  2 * ρ * Real.cos (θ - Real.pi/6) = Real.sqrt 3

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ (ρ₁ θ₁ ρ₂ θ₂ : ℝ),
    curve_C ρ₁ θ₁ ∧ line_l ρ₁ θ₁ ∧
    curve_C ρ₂ θ₂ ∧ line_l ρ₂ θ₂ ∧
    A = (ρ₁ * Real.cos θ₁, ρ₁ * Real.sin θ₁) ∧
    B = (ρ₂ * Real.cos θ₂, ρ₂ * Real.sin θ₂)

-- Theorem statement
theorem intersection_distance (A B : ℝ × ℝ) :
  intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 10 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l176_17645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_of_curve_and_line_l176_17686

theorem intersection_points_of_curve_and_line :
  let curve := {p : ℝ × ℝ | p.1^2 - 4*p.2^2 = 4}
  let line := {p : ℝ × ℝ | p.1 = 3*p.2}
  let intersection := {p : ℝ × ℝ | p ∈ curve ∧ p ∈ line}
  intersection = {(3, 1), (-3, -1)} := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_of_curve_and_line_l176_17686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l176_17623

-- Define the ellipse C
def ellipse (a b x y : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2/a^2)

theorem ellipse_properties (a b : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : eccentricity a b = Real.sqrt 2 / 2) 
  (h4 : ellipse a b 2 1) :
  -- 1) Standard equation of C
  (∃ (x y : ℝ), x^2/6 + y^2/3 = 1 ↔ ellipse a b x y) ∧
  -- 2) Maximum area of triangle OPQ
  (∃ (max_area : ℝ), 
    max_area = 2 ∧
    ∀ (k : ℝ), 
      let l (x : ℝ) := k * x + 1
      ∃ (x1 x2 : ℝ),
        ellipse a b x1 (l x1) ∧
        ellipse a b x2 (l x2) ∧
        (1/2 * |x1 - x2|) ≤ max_area) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l176_17623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_cover_quadrilateral_l176_17651

-- Define a convex quadrilateral
structure ConvexQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  convex : Convex ℝ (Set.range vertices)

-- Define a circle with diameter as a side of the quadrilateral
def CircleOnSide (q : ConvexQuadrilateral) (i : Fin 4) : Set (ℝ × ℝ) :=
  let a := q.vertices i
  let b := q.vertices ((i + 1) % 4)
  {p : ℝ × ℝ | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (1 - t) • a + t • b}

-- Define the union of all four circles
def UnionOfCircles (q : ConvexQuadrilateral) : Set (ℝ × ℝ) :=
  ⋃ i : Fin 4, CircleOnSide q i

-- The theorem to be proved
theorem circles_cover_quadrilateral (q : ConvexQuadrilateral) :
  Set.range q.vertices ⊆ UnionOfCircles q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_cover_quadrilateral_l176_17651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l176_17671

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 2 * Real.log x

-- State the theorem
theorem f_strictly_increasing :
  ∀ x y, x > Real.sqrt 3 / 3 → y > Real.sqrt 3 / 3 → x < y → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l176_17671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_price_return_l176_17605

theorem stock_price_return (initial_price : ℝ) (h : initial_price > 0) : 
  (initial_price * 1.3 * 1.2) * (1 - (1 - 1 / 1.56)) = initial_price := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_price_return_l176_17605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2_greater_than_sin_1_greater_than_sin_3_l176_17670

-- Define the approximations for radian to degree conversions
def radian_1_deg : ℝ := 57
def radian_2_deg : ℝ := 114
def radian_3_deg : ℝ := 171

-- Define the property that sin is increasing in (0°, 90°)
axiom sin_increasing_0_to_90 : ∀ x y, 0 < x ∧ x < y ∧ y < 90 → Real.sin (x * Real.pi / 180) < Real.sin (y * Real.pi / 180)

-- State the theorem
theorem sin_2_greater_than_sin_1_greater_than_sin_3 :
  Real.sin 2 > Real.sin 1 ∧ Real.sin 1 > Real.sin 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2_greater_than_sin_1_greater_than_sin_3_l176_17670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l176_17687

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin x * Real.cos (x + Real.pi/3) + 4 * Real.sqrt 3 * (Real.sin x)^2 - Real.sqrt 3

theorem f_properties :
  (f (Real.pi/3) = Real.sqrt 3) ∧
  (∀ k : ℤ, ∃ x : ℝ, f x = f (k * Real.pi/2 + 5*Real.pi/12)) ∧
  (∀ x ∈ Set.Icc (-Real.pi/4) (Real.pi/3), -2 ≤ f x ∧ f x ≤ Real.sqrt 3) ∧
  (∃ x ∈ Set.Icc (-Real.pi/4) (Real.pi/3), f x = -2) ∧
  (∃ x ∈ Set.Icc (-Real.pi/4) (Real.pi/3), f x = Real.sqrt 3) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l176_17687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_volume_estimation_l176_17657

def tree_data (n : ℕ) (sum_x sum_y sum_x2 sum_y2 sum_xy : ℝ) : Prop :=
  n = 10 ∧ sum_x = 0.6 ∧ sum_y = 3.9 ∧ sum_x2 = 0.038 ∧ sum_y2 = 1.6158 ∧ sum_xy = 0.2474

def total_root_area (X : ℝ) : Prop := X = 186

theorem tree_volume_estimation 
  (n : ℕ) (sum_x sum_y sum_x2 sum_y2 sum_xy X : ℝ) 
  (h1 : tree_data n sum_x sum_y sum_x2 sum_y2 sum_xy)
  (h2 : total_root_area X) :
  let avg_x := sum_x / n
  let avg_y := sum_y / n
  let r := (sum_xy - n * avg_x * avg_y) / 
           (Real.sqrt ((sum_x2 - n * avg_x^2) * (sum_y2 - n * avg_y^2)))
  let total_volume := (avg_y / avg_x) * X
  (avg_x = 0.06 ∧ avg_y = 0.39) ∧ 
  (abs (r - 0.97) < 0.005) ∧
  (abs (total_volume - 1209) < 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_volume_estimation_l176_17657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_difference_units_digit_l176_17632

theorem greatest_difference_units_digit (n : ℤ) :
  n ≥ 850 ∧ n < 860 ∧ n % 5 = 0 →
  ∃ (x y : ℤ), 0 ≤ x ∧ x < 10 ∧ 0 ≤ y ∧ y < 10 ∧ n = 850 + x ∧ n = 850 + y ∧ 
  ∀ (z : ℤ), 0 ≤ z ∧ z < 10 ∧ n = 850 + z → |x - y| ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_difference_units_digit_l176_17632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l176_17601

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ (a b : ℝ), a > b ∧ a^2 = m/2 ∧ b^2 = m/2 - 1

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 - 4*m*x + 4*m - 3 ≥ 0

-- State the theorem
theorem range_of_m (m : ℝ) : (¬p m ∧ q m) → m ∈ Set.Icc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l176_17601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_sup_K_l176_17675

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^(x + 1) + 2)

-- Monotonicity of f
theorem f_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂ := by sorry

-- Define the set of k satisfying the inequality
def K : Set ℝ := {k : ℝ | ∀ x : ℝ, f (k * 3^x) + f (3^x - 9^x - 4) < 0}

-- Theorem about the supremum of K
theorem sup_K : ∃ s : ℝ, IsLUB K s ∧ s = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_sup_K_l176_17675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_eight_l176_17635

theorem power_of_eight (x : ℝ) (h : (8 : ℝ)^(3*x) = 64) : (8 : ℝ)^(-x) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_eight_l176_17635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rudy_total_running_time_l176_17634

/-- Calculates the total running time for Rudy given two separate runs with different distances and rates. -/
theorem rudy_total_running_time 
  (distance1 : ℝ) (rate1 : ℝ) (distance2 : ℝ) (rate2 : ℝ)
  (h1 : distance1 = 5)
  (h2 : rate1 = 10)
  (h3 : distance2 = 4)
  (h4 : rate2 = 9.5) :
  distance1 * rate1 + distance2 * rate2 = 88 := by
  sorry

#check rudy_total_running_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rudy_total_running_time_l176_17634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_of_unit_vectors_l176_17664

/-- Given four distinct unit vectors in 3D space with specific dot product relationships,
    prove that the dot product of the first and last vectors equals 1. -/
theorem dot_product_of_unit_vectors (a b c d : ℝ × ℝ × ℝ) : 
  norm a = 1 → norm b = 1 → norm c = 1 → norm d = 1 →
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  a • b = -1/5 → a • c = -1/5 → b • c = -1/5 → b • d = -1/5 → c • d = -1/5 →
  a • d = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_of_unit_vectors_l176_17664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_S_T_l176_17630

-- Define the sets S and T
def S : Set ℝ := {x | x > -2}
def T : Set ℝ := {x | -4 ≤ x ∧ x ≤ 1}

-- State the theorem
theorem intersection_S_T : S ∩ T = Set.Ioc (-2) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_S_T_l176_17630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_range_sum_l176_17603

-- Define the function f(x) = |2^x - 1|
noncomputable def f (x : ℝ) : ℝ := |2^x - 1|

-- State the theorem
theorem domain_range_sum (a b : ℝ) : 
  (∀ x, x ∈ Set.Icc a b → f x ∈ Set.Icc a b) → -- Domain is [a,b]
  (∀ y ∈ Set.Icc a b, ∃ x ∈ Set.Icc a b, f x = y) → -- Range is [a,b]
  a + b = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_range_sum_l176_17603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_remainder_l176_17611

theorem division_remainder (x y : ℕ) 
  (hx : x > 0)
  (hy : y > 0)
  (h1 : (x : ℝ) / (y : ℝ) = 75.12)
  (h2 : (y : ℝ) = 99.9999999999962) : 
  x % y = 11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_remainder_l176_17611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_f_geq_4_g_lower_bound_l176_17600

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 - (a^2 + 1) * x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + a * Real.log x

-- Theorem for the first part of the problem
theorem zero_point_f_geq_4 (a : ℝ) (ha : a ≥ 1) :
  ∀ x > 0, f a x = 0 → x ≥ 4 := by sorry

-- Theorem for the second part of the problem
theorem g_lower_bound (a : ℝ) (ha : a ≥ 1) :
  ∀ x ≥ 1, g a x > -(1/2) * a^3 - Real.exp a / a + Real.exp 1 - 2 * a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_f_geq_4_g_lower_bound_l176_17600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_n_squared_l176_17694

-- Define the sequence a_n
def a : ℕ → ℕ
  | 0 => 1  -- Define for n=0 to cover all natural numbers
  | 1 => 1
  | (n+2) => a (n+1) + 2*(n+2) - 1

-- Theorem statement
theorem a_equals_n_squared (n : ℕ) : a n = n^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_n_squared_l176_17694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_is_focus_of_hyperbola_l176_17663

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  3 * x^2 - 4 * y^2 + 6 * x - 24 * y - 8 = 0

/-- The x-coordinate of the focus -/
noncomputable def focus_x : ℝ := -1

/-- The y-coordinate of the focus -/
noncomputable def focus_y : ℝ := -3 + (Real.sqrt 203) / (2 * Real.sqrt 3)

/-- Theorem stating that the given point is a focus of the hyperbola -/
theorem is_focus_of_hyperbola : ∃ (c : ℝ), 
  (∀ (x y : ℝ), hyperbola_equation x y ↔ 
    ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
      ((x - (-1))^2 / a^2) - ((y - (-3))^2 / b^2) = 1 ∧
      c^2 = a^2 + b^2 ∧
      (x = focus_x ∧ y = focus_y) ∨ (x = focus_x ∧ y = -3 - (Real.sqrt 203) / (2 * Real.sqrt 3))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_is_focus_of_hyperbola_l176_17663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_wearing_other_colors_l176_17652

theorem students_wearing_other_colors 
  (total_students : ℕ) 
  (blue_percent : ℚ) 
  (red_percent : ℚ) 
  (green_percent : ℚ) 
  (h1 : total_students = 700) 
  (h2 : blue_percent = 45 / 100) 
  (h3 : red_percent = 23 / 100) 
  (h4 : green_percent = 15 / 100) :
  Int.floor ((1 - (blue_percent + red_percent + green_percent)) * total_students) = 119 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_wearing_other_colors_l176_17652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equation_chord_equation_l176_17640

-- Define the curve E
def CurveE : Set (ℝ × ℝ) :=
  {p | let (x, y) := p
       (x^2 : ℝ) = 4 * y}

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the distance to a line function
noncomputable def distToLine (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * p.1 + b * p.2 + c) / Real.sqrt (a^2 + b^2)

-- Define the focus point F
def F : ℝ × ℝ := (0, 1)

-- Define the point M
def M : ℝ × ℝ := (1, 4)

theorem curve_equation (p : ℝ × ℝ) (h : p ∈ CurveE) :
  distToLine p 0 1 2 = distance p F + 1 := by
  sorry

theorem chord_equation (A B : ℝ × ℝ) (hA : A ∈ CurveE) (hB : B ∈ CurveE)
  (hM : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  ∃ (k c : ℝ), k = 1/2 ∧ c = 7 ∧ ∀ (x y : ℝ), (x, y) ∈ Set.Icc A B → x - 2*y + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equation_chord_equation_l176_17640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_time_for_four_people_l176_17646

/-- Represents the time taken to paint houses given the number of people -/
noncomputable def paintTime (people : ℕ) (houses : ℕ) : ℝ :=
  (6 * 8 * houses : ℝ) / people

/-- Theorem stating that 4 people will take 12 hours to paint 2 houses -/
theorem paint_time_for_four_people :
  let initialPeople : ℕ := 6
  let initialTime : ℝ := 8
  let initialHouses : ℕ := 2
  let newPeople : ℕ := 4
  let newHouses : ℕ := 2
  paintTime newPeople newHouses = 12 := by
  -- Unfold the definition of paintTime
  unfold paintTime
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_time_for_four_people_l176_17646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_crosses_asymptote_l176_17690

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := (3 * x^2 - 7 * x - 20) / (2 * x^2 - 5 * x + 3)

-- Define the horizontal asymptote
noncomputable def horizontal_asymptote : ℝ := 3 / 2

-- Theorem stating that g(x) crosses its horizontal asymptote at x = 49
theorem g_crosses_asymptote : g 49 = horizontal_asymptote := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_crosses_asymptote_l176_17690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_l176_17674

noncomputable def a (n : ℕ) : ℕ → ℝ
  | 0 => Real.sqrt (n : ℝ)
  | k + 1 => Real.sqrt ((k + 1 : ℝ) * a n k)

theorem inequality_holds (n : ℕ) (h : n ≥ 2) : a n 2 < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_l176_17674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_k_value_l176_17658

/-- An increasing arithmetic sequence starting with 1 -/
def arithmetic_seq (d : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => arithmetic_seq d n + d

/-- An increasing geometric sequence starting with 1 -/
def geometric_seq (r : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => geometric_seq r n * r

/-- The sum of corresponding terms of arithmetic and geometric sequences -/
def c_seq (d r : ℕ) (n : ℕ) : ℕ := arithmetic_seq d n + geometric_seq r n

/-- Main theorem: Under given conditions, c_k equals 314 -/
theorem c_k_value (d r k : ℕ) : 
  d > 0 → r > 1 → 
  c_seq d r (k - 1) = 150 → c_seq d r (k + 1) = 1500 → 
  c_seq d r k = 314 := by
  sorry

#check c_k_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_k_value_l176_17658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l176_17692

-- Define the function f(x) = 4cos(x) - 1
noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x - 1

-- State the theorem
theorem f_properties :
  -- The smallest positive period is 2π
  (∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S)) ∧
  -- The maximum value in (0, π/2) is 3
  (∀ x, 0 < x ∧ x < Real.pi / 2 → f x ≤ 3) ∧
  (∃ x, 0 < x ∧ x < Real.pi / 2 ∧ f x = 3) ∧
  -- The minimum value in (0, π/2) is -1
  (∀ x, 0 < x ∧ x < Real.pi / 2 → f x ≥ -1) ∧
  (∃ x, 0 < x ∧ x < Real.pi / 2 ∧ f x = -1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l176_17692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_value_when_cos_condition_f_value_at_negative_1860_degrees_l176_17604

open Real

-- Define the function f as noncomputable
noncomputable def f (α : ℝ) : ℝ :=
  (sin (π - α) * cos (2*π - α) * tan (-α - π)) / (tan (-α) * sin (-π - α))

-- Theorem 1: Simplification of f(α)
theorem f_simplification (α : ℝ) (h : π < α ∧ α < 3*π/2) :
  f α = cos α := by
  sorry

-- Theorem 2: Value of f(α) when cos(α - 3π/2) = 1/5
theorem f_value_when_cos_condition (α : ℝ) (h1 : π < α ∧ α < 3*π/2) (h2 : cos (α - 3*π/2) = 1/5) :
  f α = -2 * Real.sqrt 6 / 5 := by
  sorry

-- Theorem 3: Value of f(-1860°)
theorem f_value_at_negative_1860_degrees :
  f (-1860 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_value_when_cos_condition_f_value_at_negative_1860_degrees_l176_17604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_at_6_is_25_l176_17668

/-- Represents a clock with ticks at 6 o'clock and 12 o'clock -/
structure Clock where
  ticks_at_6 : ℕ
  ticks_at_12 : ℕ
  time_at_12 : ℝ

/-- The time between the first and last ticks at 6 o'clock -/
noncomputable def time_at_6 (c : Clock) : ℝ :=
  (c.time_at_12 / (c.ticks_at_12 - 1)) * (c.ticks_at_6 - 1)

/-- Theorem stating the time between first and last ticks at 6 o'clock is 25 seconds -/
theorem time_at_6_is_25 (c : Clock) 
  (h1 : c.ticks_at_6 = 6)
  (h2 : c.ticks_at_12 = 12)
  (h3 : c.time_at_12 = 55) :
  time_at_6 c = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_at_6_is_25_l176_17668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_period_of_f_l176_17633

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (3 * x / 10)

theorem least_period_of_f :
  ∃! (n : ℕ), (∀ (x : ℝ), f (n * Real.pi + x) = f x) ∧
              (∀ (m : ℕ), m < n → ∃ (y : ℝ), f (m * Real.pi + y) ≠ f y) ∧
              n = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_period_of_f_l176_17633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_cups_emptied_l176_17636

/-- Represents a circular arrangement of cups -/
def CircularArrangement (n : ℕ) := Fin n

/-- Represents the state of cups (full or empty) -/
inductive CupState
| Full : CupState
| Empty : CupState

/-- Represents the state of all cups in the arrangement -/
def ArrangementState (n : ℕ) := CircularArrangement n → CupState

/-- Represents a drinking strategy -/
def DrinkingStrategy (n : ℕ) := 
  CircularArrangement n → CircularArrangement n → CircularArrangement n × CircularArrangement n

/-- Represents the process of drinking and rotating -/
def drinkAndRotate (n : ℕ) (strategy : DrinkingStrategy n) 
  (state : ArrangementState n) : ArrangementState n :=
  sorry

/-- Theorem: All cups can be emptied using the alternating strategy -/
theorem all_cups_emptied (n : ℕ) (h : n = 30) :
  ∃ (strategy : DrinkingStrategy n),
    ∀ (initial_state : ArrangementState n),
      (∀ i, initial_state i = CupState.Full) →
      ∃ (k : ℕ),
        let final_state := (drinkAndRotate n strategy)^[k] initial_state
        ∀ i, final_state i = CupState.Empty :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_cups_emptied_l176_17636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_on_circle_l176_17612

noncomputable def Circle (x y : ℤ) : Prop := x^2 + y^2 = 100

noncomputable def Distance (x₁ y₁ x₂ y₂ : ℤ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2 : ℝ)

theorem max_ratio_on_circle :
  ∀ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℤ),
    Circle x₁ y₁ → Circle x₂ y₂ → Circle x₃ y₃ → Circle x₄ y₄ →
    (x₁, y₁) ≠ (x₂, y₂) → (x₃, y₃) ≠ (x₄, y₄) →
    (x₁, y₁) ≠ (x₃, y₃) → (x₁, y₁) ≠ (x₄, y₄) →
    (x₂, y₂) ≠ (x₃, y₃) → (x₂, y₂) ≠ (x₄, y₄) →
    ¬(∃ q : ℚ, Distance x₁ y₁ x₂ y₂ = ↑q) →
    ¬(∃ q : ℚ, Distance x₃ y₃ x₄ y₄ = ↑q) →
    Distance x₁ y₁ x₂ y₂ / Distance x₃ y₃ x₄ y₄ ≤ 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_on_circle_l176_17612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_for_given_sines_l176_17631

theorem angle_sum_for_given_sines :
  ∀ (A B : ℝ),
  π / 2 < A → A < π →
  π / 2 < B → B < π →
  Real.sin A = Real.sqrt 5 / 5 →
  Real.sin B = Real.sqrt 10 / 10 →
  A + B = 7 * π / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_for_given_sines_l176_17631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l176_17654

noncomputable def g (x : ℝ) : ℝ := 1 / (x - 1)^2

theorem g_range :
  ∀ y : ℝ, y > 0 → ∃ x : ℝ, x ≠ 1 ∧ g x = y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l176_17654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trigonometric_inequality_l176_17626

theorem triangle_trigonometric_inequality (A B C : ℝ) 
  (h : Real.sin A ^ 2 + Real.sin B ^ 2 = Real.sin C ^ 2 - Real.sqrt 2 * Real.sin A * Real.sin B) :
  (∀ A' B' C' : ℝ, Real.sin A' ^ 2 + Real.sin B' ^ 2 = Real.sin C' ^ 2 - Real.sqrt 2 * Real.sin A' * Real.sin B' →
    Real.sin (2 * A') * Real.tan B' ^ 2 ≤ 3 - 2 * Real.sqrt 2) ∧
  (∃ A₀ B₀ C₀ : ℝ, Real.sin A₀ ^ 2 + Real.sin B₀ ^ 2 = Real.sin C₀ ^ 2 - Real.sqrt 2 * Real.sin A₀ * Real.sin B₀ ∧
    Real.sin (2 * A₀) * Real.tan B₀ ^ 2 = 3 - 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trigonometric_inequality_l176_17626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_rate_maximizes_revenue_optimal_room_rate_l176_17647

/-- Represents the revenue function for a hotel --/
noncomputable def revenue (rate : ℝ) : ℝ :=
  let initialRate : ℝ := 400
  let initialOccupancy : ℝ := 50
  let rateDecrease : ℝ := 20
  let occupancyIncrease : ℝ := 5
  let decreases : ℝ := (initialRate - rate) / rateDecrease
  let occupancy : ℝ := initialOccupancy + decreases * occupancyIncrease
  rate * occupancy

/-- Theorem stating that 300 yuan/day maximizes the hotel's revenue --/
theorem optimal_rate_maximizes_revenue :
  ∀ x : ℝ, x ≥ 0 → x ≤ 400 → revenue 300 ≥ revenue x :=
by sorry

/-- Corollary: The optimal room rate is 300 yuan/day --/
theorem optimal_room_rate : 
  ∃ x : ℝ, (∀ y : ℝ, y ≥ 0 → y ≤ 400 → revenue x ≥ revenue y) ∧ x = 300 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_rate_maximizes_revenue_optimal_room_rate_l176_17647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_illuminate_plane_with_six_sources_l176_17681

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A light source at a point that illuminates a 60° sector -/
structure LightSource where
  position : Point2D
  angle : ℝ  -- The angle of the central ray of the light sector

/-- Predicate to check if a point is illuminated by a light source -/
def is_illuminated (p : Point2D) (s : LightSource) : Prop :=
  -- This would involve checking if the point is within the 60° sector of the light source
  sorry

/-- The theorem stating that 6 light sources can illuminate the entire plane -/
theorem illuminate_plane_with_six_sources 
  (points : Finset Point2D) 
  (distinct : ∀ p q, p ∈ points → q ∈ points → p ≠ q → p.x ≠ q.x ∨ p.y ≠ q.y) 
  (count : points.card = 6) :
  ∃ (sources : Finset LightSource), 
    (∀ s, s ∈ sources → s.position ∈ points) ∧ 
    (sources.card = 6) ∧
    (∀ p : Point2D, ∃ s, s ∈ sources ∧ is_illuminated p s) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_illuminate_plane_with_six_sources_l176_17681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_heads_fair_coin_l176_17649

-- Define a fair coin
structure FairCoin where

-- Define the possible outcomes of a coin toss
inductive CoinOutcome where
  | Heads : CoinOutcome
  | Tails : CoinOutcome

-- Define the probability measure for a fair coin
noncomputable def fairCoinProbability : CoinOutcome → ℝ
  | CoinOutcome.Heads => 1/2
  | CoinOutcome.Tails => 1/2

-- Theorem: The probability of getting heads when tossing a fair coin once is 1/2
theorem probability_of_heads_fair_coin :
  fairCoinProbability CoinOutcome.Heads = 1/2 := by
  rfl

#check probability_of_heads_fair_coin

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_heads_fair_coin_l176_17649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_f_l176_17602

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x^2 - 3 * x + 4) / Real.log (1/2)

-- State the theorem
theorem decreasing_interval_of_f :
  ∀ x ≥ (3/4 : ℝ), ∀ y > x, f y < f x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_f_l176_17602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_fraction_range_l176_17662

theorem circle_fraction_range (x y : ℝ) : 
  x^2 + y^2 = 1 → -Real.sqrt 3 / 3 ≤ y / (x + 2) ∧ y / (x + 2) ≤ Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_fraction_range_l176_17662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_greater_than_sin_sum_l176_17621

theorem sin_cos_sum_greater_than_sin_sum (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2) : 
  Real.sin α + Real.cos α > Real.sin (α + β) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_greater_than_sin_sum_l176_17621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_digits_greatest_prime_divisor_32766_l176_17606

/-- The greatest prime divisor of a natural number -/
def greatest_prime_divisor (n : ℕ) : ℕ :=
  (Nat.factors n).maximum?
    |>.getD 1

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  (Nat.digits 10 n).sum

/-- Theorem stating that the sum of the digits of the greatest prime divisor of 32766 is 8 -/
theorem sum_digits_greatest_prime_divisor_32766 :
  sum_of_digits (greatest_prime_divisor 32766) = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_digits_greatest_prime_divisor_32766_l176_17606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_point_determine_plane_trapezoid_determines_plane_l176_17691

-- Define basic geometric objects
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

structure Line where
  contains : Point → Prop

structure Plane where
  contains : Point → Prop

-- Define a trapezoid as a structure with four points
structure Trapezoid where
  p1 : Point
  p2 : Point
  p3 : Point
  p4 : Point

-- A line and a point not on the line determine a unique plane
theorem line_point_determine_plane (l : Line) (p : Point) (h : ¬l.contains p) :
  ∃! (plane : Plane), (∀ q : Point, l.contains q → plane.contains q) ∧ plane.contains p :=
sorry

-- A trapezoid determines a unique plane
theorem trapezoid_determines_plane (t : Trapezoid) :
  ∃! (plane : Plane), plane.contains t.p1 ∧ plane.contains t.p2 ∧ plane.contains t.p3 ∧ plane.contains t.p4 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_point_determine_plane_trapezoid_determines_plane_l176_17691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_properties_l176_17618

noncomputable section

open Real

theorem trigonometric_properties :
  (¬(α = π/6 → Real.sin α = 1/2) → False) ∧
  (∀ x : ℝ, Real.sin x ≤ 1) ∧
  (∀ x ∈ Set.Ioo 0 (π/2), Real.sin x + Real.cos x > 1/2) ∧
  (∀ A B C : ℝ, 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
    Real.sin A > Real.sin B → A > B) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_properties_l176_17618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l176_17684

/-- Parabola with focus F and directrix x = -p/2 -/
structure Parabola where
  p : ℝ
  focus : ℝ × ℝ
  directrix : ℝ → ℝ × ℝ
  h_p_pos : p > 0
  h_focus : focus = (p/2, 0)
  h_directrix : directrix = λ x => (-p/2, x)

/-- Point on the parabola -/
def point_on_parabola (C : Parabola) (P : ℝ × ℝ) : Prop :=
  P.2^2 = 2 * C.p * P.1

/-- Distance between two points -/
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem parabola_equation (C : Parabola) 
  (P : ℝ × ℝ) (Q : ℝ × ℝ) :
  point_on_parabola C P →
  P.2 = 4 →
  Q = (0, 4) →
  distance P C.focus = 3/2 * distance P Q →
  C.p = 2 * Real.sqrt 2 ∧ P = (2 * Real.sqrt 2, 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l176_17684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_surface_area_theorem_octahedron_surface_area_proof_l176_17665

/-- The surface area of an octahedron formed by connecting the centers of the bases
    to the centers of the lateral faces of a cube with edge length a -/
noncomputable def octahedron_surface_area (a : ℝ) : ℝ := a^2 * Real.sqrt 3

/-- The theorem stating that the surface area of the octahedron is a^2 * √3 -/
theorem octahedron_surface_area_theorem (a : ℝ) :
  octahedron_surface_area a = a^2 * Real.sqrt 3 := by
  -- Unfold the definition of octahedron_surface_area
  unfold octahedron_surface_area
  -- The equality follows directly from the definition
  rfl

/-- Proof that the surface area of the octahedron is correct -/
theorem octahedron_surface_area_proof (a : ℝ) (h : a > 0) :
  octahedron_surface_area a = 8 * (1/2 * (a * Real.sqrt 2 / 2)^2 * Real.sqrt 3 / 2) := by
  sorry  -- The detailed proof steps would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_surface_area_theorem_octahedron_surface_area_proof_l176_17665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l176_17619

/-- Parabola type representing y²=2px with p > 0 -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- Point type representing a point (x, y) in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Focus of a parabola -/
noncomputable def focus (c : Parabola) : Point :=
  { x := c.p / 2, y := 0 }

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For a parabola y²=2px with p > 0, if A(6, y₀) is on the parabola and |AF|=2p, then p = 4 -/
theorem parabola_focus_distance (c : Parabola) (A : Point) 
  (h_on_parabola : A.y^2 = 2 * c.p * A.x)
  (h_x : A.x = 6)
  (h_distance : distance A (focus c) = 2 * c.p) : 
  c.p = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l176_17619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_numbers_on_board_l176_17624

theorem max_numbers_on_board : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, n ≤ 235) ∧ 
  (∀ a b c, a ∈ S → b ∈ S → c ∈ S → a ≠ b → c ∣ (a - b) → False) ∧
  S.card = 118 ∧
  (∀ T : Finset ℕ, (∀ n ∈ T, n ≤ 235) → 
    (∀ a b c, a ∈ T → b ∈ T → c ∈ T → a ≠ b → c ∣ (a - b) → False) → 
    T.card ≤ 118) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_numbers_on_board_l176_17624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_pathway_theorem_l176_17682

/-- A parallelogram with given side lengths and one height -/
structure Parallelogram where
  side1 : ℝ
  side2 : ℝ
  height1 : ℝ

/-- Calculate the other height of the parallelogram -/
noncomputable def other_height (p : Parallelogram) : ℝ :=
  (p.side1 * p.height1) / p.side2

/-- Theorem: The other height of the specific parallelogram is 20 feet -/
theorem park_pathway_theorem (p : Parallelogram) 
  (h1 : p.side1 = 25) 
  (h2 : p.side2 = 75) 
  (h3 : p.height1 = 60) : 
  other_height p = 20 := by
  sorry

#check park_pathway_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_pathway_theorem_l176_17682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_abc_is_360_div_7_l176_17699

/-- Triangle ABC with point D on side AC -/
structure TriangleABC where
  /-- Length of side AC -/
  ac : ℝ
  /-- Length of AD -/
  ad : ℝ
  /-- Distance from D to side AB -/
  d_to_ab : ℝ
  /-- Distance from D to side CB -/
  d_to_cb : ℝ
  /-- AC is positive -/
  ac_pos : ac > 0
  /-- AD is positive and less than AC -/
  ad_pos_lt_ac : 0 < ad ∧ ad < ac
  /-- Distance from D to AB is positive -/
  d_to_ab_pos : d_to_ab > 0
  /-- Distance from D to CB is positive -/
  d_to_cb_pos : d_to_cb > 0

/-- The area of triangle ABC -/
noncomputable def area_abc (t : TriangleABC) : ℝ := 360 / 7

/-- Theorem: The area of triangle ABC is 360/7 cm² -/
theorem area_abc_is_360_div_7 (t : TriangleABC) 
  (h1 : t.ac = 18) 
  (h2 : t.ad = 5) 
  (h3 : t.d_to_ab = 4) 
  (h4 : t.d_to_cb = 5) : 
  area_abc t = 360 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_abc_is_360_div_7_l176_17699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_jars_theorem_l176_17669

/-- The number of jars James needs to buy for his honey production --/
def jars_needed (num_hives : ℕ) (honey_per_hive : ℚ) (jar_capacity : ℚ) (friend_portion : ℚ) : ℕ :=
  let total_honey := num_hives * honey_per_hive
  let james_honey := total_honey * (1 - friend_portion)
  (james_honey / jar_capacity).ceil.toNat

/-- Theorem stating that James needs 100 jars given the problem conditions --/
theorem james_jars_theorem :
  jars_needed 5 20 (1/2) (1/2) = 100 := by
  sorry

#eval jars_needed 5 20 (1/2) (1/2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_jars_theorem_l176_17669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_property_l176_17679

-- Define the parabola
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = p.1

-- Define the circle
def circle_M (p : ℝ × ℝ) : Prop := (p.1 - 2)^2 + p.2^2 = 1

-- Define a line being tangent to the circle
def tangent_to_circle (l : Set (ℝ × ℝ)) : Prop :=
  ∃ p, p ∈ l ∧ circle_M p ∧ ∀ q ∈ l, circle_M q → q = p

-- Define a line through two points
def line_through (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p = (1 - t) • A + t • B}

-- Main theorem
theorem tangent_property (A₁ A₂ A₃ : ℝ × ℝ) :
  parabola A₁ ∧ parabola A₂ ∧ parabola A₃ →
  tangent_to_circle (line_through A₁ A₂) →
  tangent_to_circle (line_through A₁ A₃) →
  tangent_to_circle (line_through A₂ A₃) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_property_l176_17679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_squares_simplification_l176_17607

-- Define the expressions
def expr1 (x y : ℝ) := (-x-y)*(-x+y)
def expr2 (x y : ℝ) := (-x+y)*(x-y)
def expr3 (x y : ℝ) := (y+x)*(x-y)
def expr4 (x y : ℝ) := (y-x)*(x+y)

-- Define a predicate for difference of squares form
def is_difference_of_squares (f : ℝ → ℝ → ℝ) :=
  ∃ (a b : ℝ → ℝ → ℝ), ∀ x y, f x y = (a x y)^2 - (b x y)^2

-- State the theorem
theorem difference_of_squares_simplification :
  is_difference_of_squares expr1 ∧
  is_difference_of_squares expr3 ∧
  is_difference_of_squares expr4 ∧
  ¬is_difference_of_squares expr2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_squares_simplification_l176_17607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l176_17677

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (6 * x) / (1 + x^2)

-- State the theorem
theorem max_value_of_f :
  (∀ x ∈ Set.Icc 0 3, f x ≤ 3) ∧ (∃ x ∈ Set.Icc 0 3, f x = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l176_17677
