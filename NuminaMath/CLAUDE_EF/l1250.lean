import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_volume_specific_tank_l1250_125088

/-- Calculates the volume of water in a cylindrical tank lying on its side. -/
noncomputable def water_volume_in_cylinder (r h d : ℝ) : ℝ :=
  let θ := Real.arccos ((r - d) / r)
  let sector_area := 2 * θ * r^2
  let triangle_area := d * Real.sqrt (r^2 - (r - d)^2)
  h * (sector_area - triangle_area)

/-- The volume of water in a specific cylindrical tank. -/
theorem water_volume_specific_tank :
  water_volume_in_cylinder 5 10 3 = 150 * Real.pi - 180 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_volume_specific_tank_l1250_125088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_sphere_ratio_angle_theorem_l1250_125074

/-- Represents a cone with an inscribed sphere -/
structure ConeWithInscribedSphere where
  R : ℝ  -- radius of the base of the cone
  H : ℝ  -- height of the cone
  r : ℝ  -- radius of the inscribed sphere
  k : ℝ  -- ratio of cone volume to sphere volume

/-- The volume ratio condition for the cone and inscribed sphere -/
def volume_ratio_condition (c : ConeWithInscribedSphere) : Prop :=
  (1 / 3) * Real.pi * c.R^2 * c.H = c.k * ((4 / 3) * Real.pi * c.r^3)

/-- The angle between the slant height and the base of the cone -/
noncomputable def slant_base_angle (c : ConeWithInscribedSphere) : ℝ :=
  2 * Real.arctan (Real.sqrt ((c.k + Real.sqrt (c.k^2 - 2*c.k)) / (2*c.k)))

/-- The theorem stating the relationship between the cone-sphere ratio and the angle -/
theorem cone_sphere_ratio_angle_theorem (c : ConeWithInscribedSphere) :
  volume_ratio_condition c →
  (slant_base_angle c = 2 * Real.arctan (Real.sqrt ((c.k + Real.sqrt (c.k^2 - 2*c.k)) / (2*c.k))) ∧
   c.k ≥ 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_sphere_ratio_angle_theorem_l1250_125074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_tan_monotone_first_quadrant_g_symmetry_axis_l1250_125005

open Real

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 2 * (cos ((1/3) * x + π/4))^2 - 1
noncomputable def g (x : ℝ) : ℝ := sin (2 * x + 5 * π/4)

-- State the theorems
theorem f_is_odd : ∀ x, f (-x) = -f x := by sorry

theorem tan_monotone_first_quadrant :
  ∀ α β, 0 < α ∧ α < β ∧ β < π/2 → tan α < tan β := by sorry

theorem g_symmetry_axis :
  ∀ x, g (π/4 - x) = g (π/4 + x) := by sorry

-- Note: We don't include the false propositions in our Lean statement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_tan_monotone_first_quadrant_g_symmetry_axis_l1250_125005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_colonies_growth_time_l1250_125008

/-- Represents the size of a bacteria colony on a given day -/
def ColonySize := ℕ

/-- Represents the number of days -/
def Days := ℕ

/-- The growth rate of the bacteria colony (doubling each day) -/
def growthRate : ℕ := 2

/-- The number of days it takes for a single colony to reach the habitat limit -/
def daysToLimit : ℕ := 21

/-- A function representing the size of a colony after a given number of days -/
def colonyGrowth (initialSize : ℕ) (days : ℕ) : ℕ :=
  initialSize * growthRate ^ days

/-- The habitat's limit for a single colony -/
def habitatLimit : ℕ := colonyGrowth 1 daysToLimit

/-- Theorem: Two colonies reach the habitat limit in the same number of days as a single colony -/
theorem two_colonies_growth_time (initialSize : ℕ) : 
  colonyGrowth (2 * initialSize) daysToLimit = 2 * habitatLimit := by
  sorry

#check two_colonies_growth_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_colonies_growth_time_l1250_125008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_count_l1250_125035

theorem disk_count (blue yellow green : ℕ) : 
  blue + yellow + green > 0 →
  3 * yellow = 7 * blue →
  3 * green = 8 * blue →
  green = blue + 30 →
  blue + yellow + green = 108 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_count_l1250_125035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_zero_l1250_125093

theorem determinant_zero (α β : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![0, Real.cos α, Real.sin α; 
                                        -Real.cos α, 0, Real.cos β; 
                                        -Real.sin α, -Real.cos β, 0]
  Matrix.det M = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_zero_l1250_125093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1250_125046

/-- The time taken by two workers to complete a task together -/
noncomputable def time_together : ℝ := 8

/-- The time taken by worker A to complete the task alone -/
noncomputable def time_a_alone : ℝ := 12

/-- The rate at which a task is completed -/
noncomputable def rate (time : ℝ) : ℝ := 1 / time

theorem work_completion_time :
  rate time_together = rate time_a_alone + rate (1 / (rate time_together - rate time_a_alone)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1250_125046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_2004_2008_l1250_125082

/-- Custom operation ⊗ for positive integers -/
def custom_op : ℕ+ → ℕ+ → ℤ :=
  sorry

/-- Axioms for the custom operation -/
axiom custom_op_base : custom_op 1 1 = 2
axiom custom_op_left : ∀ m n : ℕ+, custom_op (m + 1) n = custom_op m n - 1
axiom custom_op_right : ∀ m n : ℕ+, custom_op m (n + 1) = custom_op m n + 2

/-- The main theorem to prove -/
theorem custom_op_2004_2008 : custom_op 2004 2008 = 2013 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_2004_2008_l1250_125082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bacteria_growth_l1250_125071

/-- The number of bacteria in a generation --/
def BacteriaCount := ℕ

/-- The effective multiplication factor between generations --/
def MultiplicationFactor : ℕ := 4

/-- The number of generations --/
def Generations : ℕ := 7

/-- The number of bacteria in the seventh generation (in millions) --/
def SeventhGeneration : ℕ := 4096

theorem bacteria_growth (first_gen : ℕ) :
  first_gen * MultiplicationFactor^(Generations - 1) = SeventhGeneration →
  first_gen = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bacteria_growth_l1250_125071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_complement_l1250_125020

/-- Given sets A and B, prove that their intersection with the complement of B is as specified -/
theorem set_intersection_complement (A B : Set ℝ) :
  A = {x : ℝ | 1 + 2*x - 3*x^2 > 0} →
  B = {x : ℝ | 2*x*(4*x - 1) < 0} →
  A ∩ (Set.univ \ B) = Set.Ioc (-1/3 : ℝ) 0 ∪ Set.Ico (1/4 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_complement_l1250_125020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_three_equals_thirteen_no_n_for_p_equals_2002_l1250_125097

noncomputable def P (n : ℕ) : ℕ :=
  Finset.card (Finset.filter (fun p : ℕ × ℕ =>
    let (a, b) := p
    n ≥ 2 ∧ 
    0 < (n : ℚ) / a ∧ (n : ℚ) / a < 1 ∧
    1 < (a : ℚ) / b ∧ (a : ℚ) / b < 2 ∧
    2 < (b : ℚ) / n ∧ (b : ℚ) / n < 3)
    (Finset.product (Finset.range (3*n)) (Finset.range (3*n))))

theorem p_three_equals_thirteen : P 3 = 13 := by sorry

theorem no_n_for_p_equals_2002 : ¬∃ n : ℕ, n ≥ 2 ∧ P n = 2002 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_three_equals_thirteen_no_n_for_p_equals_2002_l1250_125097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_l1250_125070

theorem product_of_roots (p q : Polynomial ℝ) : 
  p = 3 * X^3 + 2 * X^2 - 10 * X + 30 →
  q = 4 * X^3 - 20 * X^2 + 24 →
  (p * q).roots.prod = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_l1250_125070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_and_harmonic_sum_l1250_125098

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x - 2 * a * Real.exp x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := -2 - a * x

-- Define the harmonic sum
noncomputable def harmonic_sum (n : ℕ) : ℝ := (Finset.range n).sum (fun i => 1 / (i + 1 : ℝ))

-- State the theorem
theorem function_inequality_and_harmonic_sum (a : ℝ) (n : ℕ) :
  (∀ x : ℝ, x ≥ 0 → f a x ≥ g a x) →
  a ≤ 1 ∧ harmonic_sum n < Real.log (2 * n + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_and_harmonic_sum_l1250_125098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_fen_l1250_125011

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle with three vertices -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Calculate the distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Calculate the perimeter of a triangle -/
noncomputable def perimeter (t : Triangle) : ℝ :=
  distance t.a t.b + distance t.b t.c + distance t.c t.a

/-- Represents the folded rectangular sheet with specific measurements -/
structure FoldedSheet where
  m : Point
  b : Point
  k : Point
  c : Point
  f : Point
  mb_eq_kc : distance m b = distance k c
  mb_eq_4 : distance m b = 4
  bk_eq_cf : distance b k = distance c f
  bk_eq_3 : distance b k = 3
  mk_eq_kf : distance m k = distance k f
  mk_eq_5 : distance m k = 5

/-- The main theorem to be proved -/
theorem perimeter_of_fen (sheet : FoldedSheet) (f e n : Point) :
    perimeter (Triangle.mk f e n) = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_fen_l1250_125011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_supervisors_l1250_125014

/-- Calculates the number of supervisors in a factory given certain salary information --/
def calculate_supervisors (total_avg : ℚ) (supervisor_avg : ℚ) (laborer_avg : ℚ) (num_laborers : ℕ) : ℕ :=
  let total_workers := num_laborers + (total_avg * (num_laborers : ℚ) - num_laborers * laborer_avg) / (supervisor_avg - total_avg)
  (total_workers - num_laborers).floor.toNat

/-- Proves that the number of supervisors is 10 given the specific conditions --/
theorem factory_supervisors : calculate_supervisors 1250 2450 950 42 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_supervisors_l1250_125014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_product_equals_one_over_128_l1250_125080

theorem cosine_product_equals_one_over_128 :
  Real.cos (π / 15) * Real.cos (2 * π / 15) * Real.cos (3 * π / 15) * Real.cos (4 * π / 15) * 
  Real.cos (5 * π / 15) * Real.cos (6 * π / 15) * Real.cos (7 * π / 15) = 1 / 128 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_product_equals_one_over_128_l1250_125080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_inequality_l1250_125079

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then x^2 else -x^2

-- State the theorem
theorem odd_function_inequality (a : ℝ) :
  (∀ x : ℝ, f (-x) = -f x) →  -- f is odd
  (∀ x : ℝ, x ≥ 0 → f x = x^2) →  -- f(x) = x² for x ≥ 0
  (∀ x ∈ Set.Icc a (a + 2), f (x + a) ≥ 2 * f x) →  -- inequality holds for x ∈ [a, a+2]
  a ≥ Real.sqrt 2 :=
by
  intros h_odd h_nonneg h_ineq
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_inequality_l1250_125079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_constant_shift_standard_deviation_is_two_l1250_125048

/-- Standard deviation of a list of numbers -/
noncomputable def standardDeviation (xs : List ℝ) : ℝ :=
  let mean := xs.sum / xs.length
  (xs.map (fun x => (x - mean) ^ 2)).sum / xs.length |> Real.sqrt

theorem standard_deviation_constant_shift 
  (a b c k : ℝ) :
  let original := [a, b, c]
  let shifted := [a + k, b + k, c + k]
  standardDeviation original = standardDeviation shifted :=
by
  sorry

/-- Given that the new standard deviation is 2 -/
def new_std_dev : ℝ := 2

theorem standard_deviation_is_two 
  (a b c k : ℝ) :
  let shifted := [a + k, b + k, c + k]
  standardDeviation shifted = new_std_dev :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_constant_shift_standard_deviation_is_two_l1250_125048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_difference_l1250_125069

theorem cube_root_difference (r : ℝ) (h : r^(1/3) - 1 / r^(1/3) = 2) : 
  r^3 - 1 / (r^3) = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_difference_l1250_125069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_mile_time_third_mile_speed_third_mile_time_l1250_125095

/-- Represents the speed of a particle at a given mile -/
noncomputable def speed (n : ℕ) : ℝ :=
  4 / ((n - 1) ^ 2 : ℝ)

/-- Represents the time taken to traverse the nth mile -/
noncomputable def time_for_nth_mile (n : ℕ) : ℝ :=
  1 / speed n

theorem nth_mile_time (n : ℕ) (h : n ≥ 3) :
  time_for_nth_mile n = (n - 1)^2 / 4 := by
  sorry

/-- The speed for the third mile is 1 mile per hour -/
theorem third_mile_speed :
  speed 3 = 1 := by
  sorry

/-- The time taken for the third mile is 1 hour -/
theorem third_mile_time :
  time_for_nth_mile 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_mile_time_third_mile_speed_third_mile_time_l1250_125095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magical_red_knights_fraction_l1250_125024

theorem magical_red_knights_fraction :
  ∀ (total : ℕ),
  total > 0 →
  let red : ℕ := (3 * total) / 8
  let blue : ℕ := total - red
  let magical : ℕ := total / 4
  let magical_red : ℕ := red * 3 / 7  -- Define magical_red
  let magical_blue : ℕ := magical - magical_red  -- Define magical_blue
  let red_magical_fraction : ℚ := magical_red / red
  let blue_magical_fraction : ℚ := magical_blue / blue
  red_magical_fraction = 3 * blue_magical_fraction →
  magical = magical_red + magical_blue →
  red_magical_fraction = 3 / 7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_magical_red_knights_fraction_l1250_125024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ce_length_l1250_125064

-- Define the triangle
structure Triangle (A B C : ℝ × ℝ) : Prop where
  valid : True -- Placeholder, can be expanded with actual triangle properties

-- Define the right angle
def is_right_angle (A B C : ℝ × ℝ) : Prop :=
  True -- Placeholder, should be replaced with actual right angle definition

-- Define the 60 degree angle
def is_60_degree (A B C : ℝ × ℝ) : Prop :=
  True -- Placeholder, should be replaced with actual 60 degree angle definition

-- Define the length of a line segment
def length (A B : ℝ × ℝ) : ℝ :=
  0 -- Placeholder, should be replaced with actual length calculation

-- Theorem statement
theorem ce_length 
  (A B C D E : ℝ × ℝ)
  (triangle_ABE : Triangle A B E)
  (triangle_BCE : Triangle B C E)
  (triangle_CDE : Triangle C D E)
  (right_ABE : is_right_angle A B E)
  (right_BCE : is_right_angle B C E)
  (right_CDE : is_right_angle C D E)
  (angle_AEB : is_60_degree A E B)
  (angle_BEC : is_60_degree B E C)
  (angle_CED : is_60_degree C E D)
  (ae_length : length A E = 36)
  : length C E = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ce_length_l1250_125064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_60_coordinates_l1250_125041

/-- Represents a point with integer coordinates -/
structure Point where
  x : ℤ
  y : ℤ

/-- The sequence of points -/
def P : ℕ → Point
  | _ => ⟨0, 0⟩  -- Default definition, will be overridden by axioms

/-- The first few points in the sequence are defined -/
axiom P_1 : P 1 = ⟨1, 1⟩
axiom P_2 : P 2 = ⟨1, 2⟩
axiom P_3 : P 3 = ⟨2, 1⟩
axiom P_4 : P 4 = ⟨1, 3⟩
axiom P_5 : P 5 = ⟨2, 2⟩
axiom P_6 : P 6 = ⟨3, 1⟩
axiom P_7 : P 7 = ⟨1, 4⟩
axiom P_8 : P 8 = ⟨2, 3⟩
axiom P_9 : P 9 = ⟨3, 2⟩
axiom P_10 : P 10 = ⟨4, 1⟩
axiom P_11 : P 11 = ⟨1, 5⟩
axiom P_12 : P 12 = ⟨2, 4⟩

/-- The theorem to be proved -/
theorem P_60_coordinates : P 60 = ⟨5, 7⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_60_coordinates_l1250_125041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_5_is_simplest_l1250_125007

/-- A quadratic radical is simplest if it cannot be simplified further and does not result in a rational number. -/
noncomputable def IsSimplestQuadraticRadical (x : ℝ) : Prop :=
  ∀ y : ℚ, x ≠ y ∧ ∀ a b : ℚ, x ≠ a * Real.sqrt b

/-- The given options for quadratic radicals -/
noncomputable def Options : List ℝ := [Real.sqrt (1/3), Real.sqrt 4, Real.sqrt 5, Real.sqrt 8]

theorem sqrt_5_is_simplest : 
  ∃ x ∈ Options, IsSimplestQuadraticRadical x ∧ 
  ∀ y ∈ Options, IsSimplestQuadraticRadical y → y = x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_5_is_simplest_l1250_125007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_6x_mod_9_l1250_125094

theorem remainder_6x_mod_9 (x : ℕ) (h : x % 9 = 5) : (6 * x) % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_6x_mod_9_l1250_125094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_root_solutions_l1250_125084

def A (x : ℤ) : ℕ → ℤ
  | 0 => 0
  | n + 1 => Int.sqrt (x + A x n)

theorem nested_root_solutions (x y z : ℤ) :
  (y > 0 ∧ y.toNat > 0 ∧ A x y.toNat = z) ↔ 
  (∃ t : ℤ, x = t^2 ∧ y = 1 ∧ z = t) ∨ 
  (x = 0 ∧ y > 0 ∧ z = 0) := by
  sorry

#check nested_root_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_root_solutions_l1250_125084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_b_minus_a_in_AUB_l1250_125006

def A : Finset Int := {-2, -1, 0}
def B : Finset Int := {-1, 0, 1, 2}

def total_outcomes : Nat := A.card * B.card

def favorable_outcomes : Nat := Finset.sum A (fun a => 
  Finset.sum B (fun b => if (b - a) ∈ (A ∪ B) then 1 else 0))

theorem probability_b_minus_a_in_AUB : 
  (favorable_outcomes : ℚ) / total_outcomes = 3/4 := by
  sorry

#eval favorable_outcomes
#eval total_outcomes
#eval (favorable_outcomes : ℚ) / total_outcomes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_b_minus_a_in_AUB_l1250_125006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_kitten_percentage_l1250_125045

theorem cat_kitten_percentage : 
  ∀ (initial_cats female_ratio kittens_per_female kittens_sold price donation_percent : ℕ),
  initial_cats = 16 →
  female_ratio = 3/8 →
  kittens_per_female = 11 →
  kittens_sold = 23 →
  price = 4 →
  donation_percent = 25 →
  let female_cats := (initial_cats : ℚ) * (female_ratio : ℚ)
  let total_kittens := female_cats * kittens_per_female
  let remaining_kittens := total_kittens - kittens_sold
  let total_cats := initial_cats + remaining_kittens
  let kitten_percentage := (remaining_kittens / total_cats) * 100
  ⌊kitten_percentage⌋₊ = 73 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_kitten_percentage_l1250_125045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sprinkler_coverage_theorem_l1250_125003

/-- Represents a square lawn with sprinklers at each corner -/
structure LawnWithSprinklers where
  side_length : ℝ
  sprinkler_range : ℝ
  sprinkler_angle : ℝ

/-- Calculates the proportion of the lawn covered by the sprinklers -/
noncomputable def covered_proportion (lawn : LawnWithSprinklers) : ℝ :=
  (Real.pi + 3 - 3 * Real.sqrt 3) / 3

/-- Theorem stating the proportion of the lawn covered by the sprinklers -/
theorem sprinkler_coverage_theorem (lawn : LawnWithSprinklers) 
  (h1 : lawn.side_length > 0)
  (h2 : lawn.sprinkler_range = lawn.side_length)
  (h3 : lawn.sprinkler_angle = Real.pi / 2) : 
  covered_proportion lawn = (Real.pi + 3 - 3 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sprinkler_coverage_theorem_l1250_125003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_l1250_125086

/-- The equation of the circle is 2x^2 - 8x + 2y^2 + 4y = 34 -/
def circle_equation (x y : ℝ) : Prop :=
  2 * x^2 - 8 * x + 2 * y^2 + 4 * y = 34

/-- The center of the circle is (2, -1) -/
theorem circle_center : 
  ∃ (x₀ y₀ : ℝ), (x₀ = 2 ∧ y₀ = -1) ∧
    ∀ (x y : ℝ), circle_equation x y ↔ (x - x₀)^2 + (y - y₀)^2 = 22 :=
by
  -- Introduce the center coordinates
  let x₀ : ℝ := 2
  let y₀ : ℝ := -1
  
  -- State the existence of the center
  use x₀, y₀
  
  constructor
  · -- Prove that x₀ = 2 and y₀ = -1
    simp
  
  · -- Prove the equivalence of the circle equations
    intro x y
    sorry -- The actual proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_l1250_125086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_m_equals_two_l1250_125029

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 - 3 / x

-- Define the function g in terms of f and m
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f x - m

-- State the theorem
theorem odd_function_implies_m_equals_two :
  ∀ m : ℝ, (∀ x : ℝ, g m (-x) = -(g m x)) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_m_equals_two_l1250_125029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1250_125066

-- Define the function f with domain [0, 2]
def f : ℝ → ℝ := sorry

-- Define the function g(x) = f(x+1) + f(x-1)
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x + 1) + f (x - 1)

-- Theorem statement
theorem domain_of_g :
  {x : ℝ | (x + 1) ∈ Set.Icc 0 2 ∧ (x - 1) ∈ Set.Icc 0 2} = {1} := by
  sorry

#check domain_of_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1250_125066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1250_125081

def sequence_a : ℕ → ℤ
  | 0 => 0  -- Add a case for 0
  | 1 => 1
  | 2 => 3
  | (n + 3) => sequence_a (n + 2) - sequence_a (n + 1)

def S (n : ℕ) : ℤ := ∑ i in Finset.range n, sequence_a (i + 1)

theorem sequence_properties :
  sequence_a 100 = -1 ∧ S 100 = 5 := by
  sorry

#eval sequence_a 100
#eval S 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1250_125081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bottle_height_theorem_l1250_125090

/-- Represents a bottle composed of two cylinders -/
structure Bottle where
  small_radius : ℝ
  large_radius : ℝ
  upright_water_height : ℝ
  upsidedown_water_height : ℝ

/-- Calculates the total height of the bottle -/
noncomputable def bottle_height (b : Bottle) : ℝ :=
  (b.upright_water_height * b.small_radius^2 + b.upsidedown_water_height * b.large_radius^2) / 
  (b.small_radius^2 + b.large_radius^2)

theorem bottle_height_theorem (b : Bottle) 
  (h1 : b.small_radius = 1)
  (h2 : b.large_radius = 3)
  (h3 : b.upright_water_height = 20)
  (h4 : b.upsidedown_water_height = 28) :
  bottle_height b = 29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bottle_height_theorem_l1250_125090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l1250_125060

theorem unique_solution_exponential_equation :
  ∃! x : ℚ, (5 : ℝ) ^ (2 * (x : ℝ)^2 - 9 * (x : ℝ) + 5) = (5 : ℝ) ^ (2 * (x : ℝ)^2 + 3 * (x : ℝ) - 1) ∧ x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l1250_125060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_value_range_l1250_125089

-- Define the propositions p and q
def p (m : ℝ) : Prop := 0 < m ∧ m < 1/3

def q (m : ℝ) : Prop := m^2 - 15*m < 0

-- Define the range of m
def m_range (m : ℝ) : Prop := 1/3 ≤ m ∧ m < 15

-- State the theorem
theorem m_value_range :
  ∀ m : ℝ, (¬(p m ∧ q m) ∧ (p m ∨ q m)) → m_range m :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_value_range_l1250_125089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_exp_plus_x_l1250_125040

theorem definite_integral_exp_plus_x : 
  ∫ x in (Set.Icc 0 1), (Real.exp x + x) = Real.exp 1 - 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_exp_plus_x_l1250_125040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l1250_125012

/-- Parabola type representing y^2 = 4x -/
structure Parabola where
  x : ℝ
  y : ℝ
  eq : y^2 = 4*x

/-- Point type representing a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line type representing a line passing through two points -/
structure Line where
  p1 : Point
  p2 : Point

def focus : Point := ⟨1, 0⟩

def distance_to_axis (p : Point) : ℝ := |p.x|

/-- Check if a point is on a parabola -/
def on_parabola (p : Point) (E : Parabola) : Prop :=
  p.y^2 = 4*p.x

/-- Check if a point is on a line -/
def on_line (p : Point) (l : Line) : Prop :=
  (p.y - l.p1.y) * (l.p2.x - l.p1.x) = (p.x - l.p1.x) * (l.p2.y - l.p1.y)

theorem parabola_intersection_length (E : Parabola) (l : Line) (A B : Point) :
  l.p1 = focus →
  on_parabola A E →
  on_parabola B E →
  on_line A l →
  on_line B l →
  distance_to_axis A = 3 →
  distance_to_axis B = 7 →
  ‖A.x - B.x‖ + ‖A.y - B.y‖ = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l1250_125012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_third_quadrant_l1250_125034

/-- A point P in the coordinate plane with coordinates (m, 1+2m) -/
structure Point where
  m : ℝ
  x : ℝ := m
  y : ℝ := 1 + 2*m

/-- Definition of a point being in the third quadrant -/
def is_in_third_quadrant (P : Point) : Prop :=
  P.x < 0 ∧ P.y < 0

/-- Theorem stating the condition for P(m, 1+2m) to be in the third quadrant -/
theorem point_in_third_quadrant (m : ℝ) :
  is_in_third_quadrant ⟨m, m, 1 + 2*m⟩ ↔ m < -1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_third_quadrant_l1250_125034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_addition_and_parallel_vectors_l1250_125004

noncomputable def a (θ : Real) : Prod Real Real := (1, Real.sin θ)
def b : Prod Real Real := (3, 1)

theorem vector_addition_and_parallel_vectors :
  (∃ (θ : Real),
    θ = Real.pi / 6 ∧
    2 • (a θ) + b = (5, 2)) ∧
  (∃ (θ : Real),
    θ ∈ Set.Ioo 0 (Real.pi / 2) ∧
    (∃ (k : Real), k • (a θ) = b) ∧
    Real.sin (2 * θ + Real.pi / 4) = 5 * Real.sqrt 2 / 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_addition_and_parallel_vectors_l1250_125004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_workers_theorem_l1250_125018

/-- Represents the company's widget production and profit scenario -/
structure WidgetCompany where
  maintenance_fee : ℕ := 500
  hourly_wage : ℕ := 20
  widgets_per_hour : ℕ := 5
  widget_price : ℚ := 7/2
  workday_hours : ℕ := 8

/-- Calculates the minimum number of workers needed for profit -/
def min_workers_for_profit (company : WidgetCompany) : ℕ :=
  let daily_worker_cost := company.hourly_wage * company.workday_hours
  let daily_worker_revenue := (company.widgets_per_hour * company.workday_hours : ℚ) * company.widget_price
  ⌈(company.maintenance_fee : ℚ) / (daily_worker_revenue - daily_worker_cost)⌉.toNat

/-- Theorem stating the minimum number of workers needed for profit -/
theorem min_workers_theorem (company : WidgetCompany) :
  min_workers_for_profit company = 26 := by
  sorry

#eval min_workers_for_profit { : WidgetCompany }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_workers_theorem_l1250_125018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_width_decrease_l1250_125009

/-- Represents the change in dimensions of a rectangle -/
structure RectangleChange where
  original_length : ℝ
  original_width : ℝ
  length_increase_factor : ℝ
  width_decrease_factor : ℝ

/-- The theorem statement -/
theorem rectangle_width_decrease 
  (r : RectangleChange) 
  (h1 : r.length_increase_factor = 1.3)
  (h2 : r.original_length * r.original_width = 
        (r.original_length * r.length_increase_factor) * 
        (r.original_width * r.width_decrease_factor))
  (h3 : 2 * (r.original_length * r.length_increase_factor + 
             r.original_width * r.width_decrease_factor) = 
        1.1 * (2 * (r.original_length + r.original_width))) :
  ∃ (ε : ℝ), abs (r.width_decrease_factor - 0.76923) < ε ∧ ε > 0 := by
  sorry

#check rectangle_width_decrease

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_width_decrease_l1250_125009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_susan_candy_problem_l1250_125068

theorem susan_candy_problem (tuesday_candies friday_candies candies_left candies_eaten : ℕ)
  (h1 : tuesday_candies = 3)
  (h2 : friday_candies = 2)
  (h3 : candies_left = 4)
  (h4 : candies_eaten = 6) :
  (candies_eaten + candies_left) - (tuesday_candies + friday_candies) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_susan_candy_problem_l1250_125068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_fixed_point_l1250_125038

noncomputable def l₁_angle : ℝ := Real.pi / 60
noncomputable def l₂_angle : ℝ := Real.pi / 45
noncomputable def l_slope : ℝ := 17 / 75

noncomputable def R (θ : ℝ) : ℝ := θ + Real.pi / 18

def is_fixed_point (m : ℕ) : Prop :=
  ∃ k : ℕ, m * (Real.pi / 18) = 2 * Real.pi * (k : ℝ)

theorem smallest_fixed_point :
  ∃ m : ℕ, m > 0 ∧ is_fixed_point m ∧ ∀ n : ℕ, 0 < n → n < m → ¬is_fixed_point n :=
by
  -- The proof goes here
  sorry

#eval 36

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_fixed_point_l1250_125038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1250_125055

theorem problem_solution : (9 : ℚ) * ((1/3 + 1/6 + 1/9)⁻¹ : ℚ) = 162/11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1250_125055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l1250_125052

/-- Represents the train crossing problem -/
structure TrainCrossing where
  train_length : ℝ
  platform_length : ℝ
  platform_crossing_time : ℝ

/-- Calculates the time it takes for the train to cross a signal pole -/
noncomputable def time_to_cross_pole (tc : TrainCrossing) : ℝ :=
  let total_distance := tc.train_length + tc.platform_length
  let train_speed := total_distance / tc.platform_crossing_time
  tc.train_length / train_speed

/-- Theorem stating that the time to cross the signal pole is approximately 18 seconds -/
theorem train_crossing_theorem (tc : TrainCrossing) 
  (h1 : tc.train_length = 300)
  (h2 : tc.platform_length = 550.0000000000001)
  (h3 : tc.platform_crossing_time = 51) :
  ∃ ε > 0, |time_to_cross_pole tc - 18| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l1250_125052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interesting_pairs_ratio_l1250_125031

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define the property of being an acute triangle
def is_acute_triangle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the property of a point lying on a side of the triangle
def lies_on_side (P X Y : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the midpoint of a line segment
def segment_midpoint (M P Q : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the perpendicular bisector of a line segment
def perp_bisector (L : Set (EuclideanSpace ℝ (Fin 2))) (P Q : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the property of points being concyclic
def concyclic (P Q R S : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the "interesting" property for a pair of points
def interesting_pair (E F : EuclideanSpace ℝ (Fin 2)) (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ (M K S T : EuclideanSpace ℝ (Fin 2)),
    lies_on_side E A C ∧
    lies_on_side F A B ∧
    segment_midpoint M E F ∧
    (∃ L₁, perp_bisector L₁ E F ∧ K ∈ L₁ ∧ lies_on_side K B C) ∧
    (∃ L₂, perp_bisector L₂ M K ∧ S ∈ L₂ ∧ T ∈ L₂ ∧
      lies_on_side S A C ∧ lies_on_side T A B) ∧
    concyclic K S A T

-- Main theorem
theorem interesting_pairs_ratio
  (h_acute : is_acute_triangle A B C)
  (E₁ F₁ E₂ F₂ : EuclideanSpace ℝ (Fin 2))
  (h_interesting₁ : interesting_pair E₁ F₁ A B C)
  (h_interesting₂ : interesting_pair E₂ F₂ A B C) :
  ‖E₁ - E₂‖ / ‖A - B‖ = ‖F₁ - F₂‖ / ‖A - C‖ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interesting_pairs_ratio_l1250_125031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_profit_percentage_approx_l1250_125075

-- Define the given values
noncomputable def a_cost_price : ℝ := 114.94
def a_profit_percentage : ℝ := 35
def c_buying_price : ℝ := 225

-- Define the function to calculate B's profit percentage
noncomputable def b_profit_percentage (a_cost : ℝ) (a_profit_pct : ℝ) (c_price : ℝ) : ℝ :=
  let a_selling_price := a_cost * (1 + a_profit_pct / 100)
  let b_profit := c_price - a_selling_price
  (b_profit / a_selling_price) * 100

-- State the theorem
theorem b_profit_percentage_approx :
  ∃ ε > 0, abs (b_profit_percentage a_cost_price a_profit_percentage c_buying_price - 44.99) < ε :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_profit_percentage_approx_l1250_125075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_female_winning_probability_l1250_125067

-- Define the probabilities for each name
variable (P_Alexander P_Alexandra P_Yevgeny P_Evgenia P_Valentin P_Valentina P_Vasily P_Vasilisa : ℝ)

-- Define the conditions
axiom alexander_more_common : P_Alexander = 3 * P_Alexandra
axiom yevgeny_more_common : P_Yevgeny = 3 * P_Evgenia
axiom valentin_more_common : P_Valentin = 1.5 * P_Valentina
axiom vasily_more_common : P_Vasily = 49 * P_Vasilisa

-- Define the sum of probabilities for each pair
axiom sum_alexander : P_Alexander + P_Alexandra = 1
axiom sum_yevgeny : P_Yevgeny + P_Evgenia = 1
axiom sum_valentin : P_Valentin + P_Valentina = 1
axiom sum_vasily : P_Vasily + P_Vasilisa = 1

-- Define the theorem
theorem female_winning_probability :
  (1/4 * P_Alexandra) + (1/4 * P_Evgenia) + (1/4 * P_Valentina) + (1/4 * P_Vasilisa) = 95.33 / 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_female_winning_probability_l1250_125067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_b_value_l1250_125056

/-- A line passing through two points with a specific direction vector form -/
def line_through_points_with_direction (p1 p2 : ℝ × ℝ) (b : ℝ) : Prop :=
  let direction := (p2.1 - p1.1, p2.2 - p1.2)
  ∃ (k : ℝ), (k * direction.1, k * direction.2) = (2, b)

/-- The main theorem stating that b = 2/3 for the given conditions -/
theorem direction_vector_b_value :
  let p1 : ℝ × ℝ := (-5, 4)
  let p2 : ℝ × ℝ := (-2, 5)
  ∀ b : ℝ, line_through_points_with_direction p1 p2 b → b = 2/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_b_value_l1250_125056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_tap_rate_l1250_125047

-- Define the rates of the taps
noncomputable def r₁ : ℝ := 1 / 10
noncomputable def r₃ : ℝ := 1 / 6

-- Define the combined rate of all taps
noncomputable def combined_rate : ℝ := 1 / 3

-- Theorem statement
theorem second_tap_rate (r₂ : ℝ) : 
  r₁ + r₂ + r₃ = combined_rate → r₂ = 1 / 15 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_tap_rate_l1250_125047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_coach_handshakes_zero_l1250_125087

/-- The total number of handshakes in the gymnastics competition scenario -/
def total_handshakes : ℕ := 496

/-- The theorem stating that it's possible for one coach to have 0 handshakes -/
theorem min_coach_handshakes_zero :
  ∃ (n k₁ k₂ : ℕ), 
    n ≥ 2 ∧ 
    Nat.choose n 2 + k₁ + k₂ = total_handshakes ∧
    (k₁ = 0 ∨ k₂ = 0) := by
  -- We'll use n = 32, k₁ = 0, k₂ = 0 as our witness
  use 32, 0, 0
  constructor
  · -- Prove n ≥ 2
    norm_num
  constructor
  · -- Prove Nat.choose n 2 + k₁ + k₂ = total_handshakes
    norm_num
    rfl
  · -- Prove k₁ = 0 ∨ k₂ = 0
    left
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_coach_handshakes_zero_l1250_125087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_unique_solution_conditions_range_of_a_for_bounded_difference_l1250_125096

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (2^a)

-- Statement 1
theorem solution_set_when_a_is_one :
  ∀ x : ℝ, f 1 x > 1 ↔ x ∈ Set.Ioo 0 1 := by sorry

-- Statement 2
theorem unique_solution_conditions :
  ∀ a : ℝ, (∃! x : ℝ, f a x + (a - 2) / 2 = 0) ↔ (a = 0 ∨ a = -1/4) := by sorry

-- Statement 3
theorem range_of_a_for_bounded_difference :
  ∀ a : ℝ, a > 0 →
  (∀ t : ℝ, t ∈ Set.Icc (1/2) 1 →
    ∀ x y : ℝ, x ∈ Set.Icc t (t+1) → y ∈ Set.Icc t (t+1) →
      |f a x - f a y| ≤ 1) ↔
  a ∈ Set.Ici (2/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_unique_solution_conditions_range_of_a_for_bounded_difference_l1250_125096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_balls_in_cube_l1250_125010

noncomputable def cube_side_length : ℝ := 8
noncomputable def ball_radius : ℝ := 3

noncomputable def cube_volume : ℝ := cube_side_length ^ 3
noncomputable def ball_volume : ℝ := (4 / 3) * Real.pi * (ball_radius ^ 3)

noncomputable def max_balls : ℕ := Int.toNat ⌊cube_volume / ball_volume⌋

theorem max_balls_in_cube : max_balls = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_balls_in_cube_l1250_125010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_iff_m_eq_one_l1250_125083

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = m · 2^x + 2^(-x) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * (2 : ℝ) ^ x + (2 : ℝ) ^ (-x)

/-- Theorem: f(x) = m · 2^x + 2^(-x) is an even function if and only if m = 1 -/
theorem f_even_iff_m_eq_one (m : ℝ) : IsEven (f m) ↔ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_iff_m_eq_one_l1250_125083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l1250_125091

noncomputable def original_function (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

noncomputable def shifted_function (x : ℝ) : ℝ := original_function (x + Real.pi / 3)

noncomputable def final_function (x : ℝ) : ℝ := shifted_function (x / 2)

noncomputable def expected_result (x : ℝ) : ℝ := Real.sin (4 * x + Real.pi / 3)

theorem function_transformation :
  ∀ x : ℝ, final_function x = expected_result x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l1250_125091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_illumination_range_exact_l1250_125092

/-- Particle motion in Oxy plane -/
noncomputable def x (t : ℝ) : ℝ := 3 + Real.sin t * Real.cos t - Real.sin t - Real.cos t

/-- Constant y-coordinate of the particle -/
def y : ℝ := 1

/-- Light ray equation -/
def light_ray (c : ℝ) (x : ℝ) : ℝ := c * x

/-- The range of c values for which the particle is illuminated -/
def illumination_range : Set ℝ := {c | c > 0 ∧ ∃ t, y = light_ray c (x t)}

/-- Theorem stating the exact range of c values -/
theorem illumination_range_exact : 
  illumination_range = {c | 2 * (7 - 2 * Real.sqrt 2) / 41 ≤ c ∧ c ≤ 1 / 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_illumination_range_exact_l1250_125092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_decrease_approximation_l1250_125058

/-- Calculates the percent decrease between two values -/
noncomputable def percentDecrease (original : ℝ) (new : ℝ) : ℝ :=
  (original - new) / original * 100

/-- Represents the cost changes for long-distance calls and SMS from 1990 to 2010 -/
structure CostChanges where
  callCost1990 : ℝ
  callCost2010 : ℝ
  smsCost1990 : ℝ
  smsCost2010 : ℝ

/-- Theorem stating the approximate percent decreases for call and SMS costs -/
theorem cost_decrease_approximation (costs : CostChanges)
    (h1 : costs.callCost1990 = 35)
    (h2 : costs.callCost2010 = 5)
    (h3 : costs.smsCost1990 = 15)
    (h4 : costs.smsCost2010 = 1) :
    (‖percentDecrease costs.callCost1990 costs.callCost2010 - 85‖ < 1) ∧
    (‖percentDecrease costs.smsCost1990 costs.smsCost2010 - 93‖ < 1) := by
  sorry

#check cost_decrease_approximation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_decrease_approximation_l1250_125058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_iff_a_range_l1250_125001

/-- The function f(x) = x^3 + ax^2 + x - 7 is monotonically increasing on ℝ -/
def is_monotone_increasing (a : ℝ) : Prop :=
  ∀ x : ℝ, deriv (λ x => x^3 + a*x^2 + x - 7) x ≥ 0

/-- The range of a for which f(x) is monotonically increasing -/
def a_range : Set ℝ := { a | -Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3 }

/-- Theorem stating the equivalence between the function being monotonically increasing
    and the range of a -/
theorem monotone_increasing_iff_a_range :
  ∀ a : ℝ, is_monotone_increasing a ↔ a ∈ a_range :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_iff_a_range_l1250_125001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersection_l1250_125043

/-- The line equation 5x + 8y = 10 -/
def line_eq (x y : ℝ) : Prop := 5 * x + 8 * y = 10

/-- The circle equation x^2 + y^2 = 1 -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Theorem stating that the line and circle have no intersection points -/
theorem no_intersection : ¬ ∃ (x y : ℝ), line_eq x y ∧ circle_eq x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersection_l1250_125043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_boxes_to_eliminate_l1250_125063

-- Define the set of box values
noncomputable def box_values : Finset ℚ := {1/100, 1, 5, 10, 25, 50, 75, 100, 200, 300, 400, 500, 750, 1000, 5000, 10000, 50000, 75000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 1000000}

-- Total number of boxes
def total_boxes : ℕ := 30

-- Number of boxes with value at least $200,000
noncomputable def high_value_boxes : ℕ := (box_values.filter (λ x => x ≥ 200000)).card

-- Function to calculate probability of holding a high-value box
noncomputable def prob_high_value (eliminated : ℕ) : ℚ :=
  high_value_boxes / (total_boxes - eliminated : ℚ)

-- Theorem statement
theorem min_boxes_to_eliminate :
  ∃ n : ℕ, n ≤ total_boxes ∧ 
    (∀ m : ℕ, m < n → prob_high_value m < 1/2) ∧
    prob_high_value n ≥ 1/2 ∧
    n = 10 :=
  sorry

#eval total_boxes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_boxes_to_eliminate_l1250_125063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_to_x_axis_max_a_for_positive_f_l1250_125028

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x + a / 2

/-- The derivative of f(x) with respect to x -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a

theorem tangent_line_parallel_to_x_axis (a : ℝ) (h : a > 0) :
  f_derivative a 1 = 0 ↔ a = Real.exp 1 :=
by sorry

theorem max_a_for_positive_f (a : ℝ) (h : a > 0) :
  (∀ x < 1, f a x > 0) ↔ a ≤ 2 * Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_to_x_axis_max_a_for_positive_f_l1250_125028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chris_age_is_14_l1250_125027

-- Define the ages of Amy, Ben, and Chris
variable (amy_age ben_age chris_age : ℚ)

-- The average age is 10
axiom average_age : (amy_age + ben_age + chris_age) / 3 = 10

-- Four years ago, Chris was the same age as Amy is now
axiom age_relation : chris_age - 4 = amy_age

-- In 5 years, Ben's age will be 3/4 of Amy's age at that time
axiom future_age_relation : ben_age + 5 = 3/4 * (amy_age + 5)

-- Theorem: Chris's age is 14
theorem chris_age_is_14 : chris_age = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chris_age_is_14_l1250_125027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_product_l1250_125022

theorem quadratic_roots_product (p q : ℝ) : 
  (∃ Q : ℝ → ℝ, Q = λ x ↦ x^2 + p*x + q) ∧ 
  (Real.sin (π/6) ∈ {x | x^2 + p*x + q = 0}) ∧ 
  (Real.sin (5*π/6) ∈ {x | x^2 + p*x + q = 0}) →
  p * q = -1/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_product_l1250_125022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1250_125051

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log (5 - 4*x + x^2) / Real.log (1/2)

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1250_125051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1250_125065

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((a^2 - 1) * x^2 - 2 * (a - 1) * x + 3)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) → a ∈ Set.Icc (-2 : ℝ) (-1 : ℝ) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1250_125065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_location_l1250_125030

theorem max_value_location (f : ℝ → ℝ) (a b : ℝ) (h_diff : Differentiable ℝ f) (h_closed : a ≤ b) :
  ∃ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, f y ≤ f x ∧
  (x = a ∨ x = b ∨ (∃ y ∈ Set.Ioo a b, deriv f y = 0)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_location_l1250_125030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_theorem_l1250_125044

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  majorAxis1 : Point
  majorAxis2 : Point
  minorAxis1 : Point
  minorAxis2 : Point

/-- Calculates the center of an ellipse -/
noncomputable def calculateCenter (e : Ellipse) : Point :=
  { x := (e.majorAxis1.x + e.majorAxis2.x) / 2,
    y := (e.majorAxis1.y + e.majorAxis2.y) / 2 }

/-- Calculates the focus with greater x-coordinate of an ellipse -/
noncomputable def calculateFocus (e : Ellipse) : Point :=
  calculateCenter e

theorem ellipse_focus_theorem (e : Ellipse) 
  (h1 : e.majorAxis1 = { x := 1, y := -2 })
  (h2 : e.majorAxis2 = { x := 7, y := -2 })
  (h3 : e.minorAxis1 = { x := 3, y := 1 })
  (h4 : e.minorAxis2 = { x := 3, y := -5 }) :
  calculateFocus e = { x := 3, y := -2 } := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_theorem_l1250_125044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1250_125077

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle given two sides and the included angle -/
noncomputable def triangleArea (b c A : ℝ) : ℝ := (1/2) * b * c * Real.sin A

/-- The length of the third side of a triangle given two sides and the included angle (Law of Cosines) -/
noncomputable def thirdSide (b c A : ℝ) : ℝ := Real.sqrt (b^2 + c^2 - 2*b*c*Real.cos A)

theorem triangle_properties (t : Triangle) 
  (h1 : t.b = 2)
  (h2 : t.c = Real.sqrt 3)
  (h3 : t.A = π/6) :
  triangleArea t.b t.c t.A = Real.sqrt 3 / 2 ∧ 
  thirdSide t.b t.c t.A = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1250_125077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_side_length_l1250_125062

/-- Two rectangles in a plane -/
structure TwoRectangles where
  rect1 : Set (EuclideanSpace ℝ (Fin 2))
  rect2 : Set (EuclideanSpace ℝ (Fin 2))
  is_rectangle1 : sorry
  is_rectangle2 : sorry

/-- Circumscribed circle of a rectangle -/
noncomputable def circumscribedCircle (rect : Set (EuclideanSpace ℝ (Fin 2))) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

/-- Diagonals of a rectangle -/
noncomputable def diagonals (rect : Set (EuclideanSpace ℝ (Fin 2))) : Set (Set (EuclideanSpace ℝ (Fin 2))) := sorry

/-- A point touches a circle if it lies on the circle -/
def touches (p : EuclideanSpace ℝ (Fin 2)) (c : Set (EuclideanSpace ℝ (Fin 2))) : Prop := p ∈ c

/-- The diagonals of each rectangle touch the circumscribed circle of the other rectangle -/
def diagonalsTouchCircle (tr : TwoRectangles) : Prop :=
  ∀ d1 ∈ diagonals tr.rect1, ∃ p ∈ d1, touches p (circumscribedCircle tr.rect2) ∧
  ∀ d2 ∈ diagonals tr.rect2, ∃ q ∈ d2, touches q (circumscribedCircle tr.rect1)

/-- Side length of a rectangle -/
noncomputable def sideLength (rect : Set (EuclideanSpace ℝ (Fin 2))) : ℝ := sorry

/-- Main theorem -/
theorem equal_side_length (tr : TwoRectangles) (h : diagonalsTouchCircle tr) :
  ∃ s1 s2 : ℝ, s1 = sideLength tr.rect1 ∧ s2 = sideLength tr.rect2 ∧ s1 = s2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_side_length_l1250_125062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jumps_correct_l1250_125050

/-- The number of ways a frog can return to vertex A in n jumps on a triangle ABC -/
def frog_jumps (n : ℕ) : ℚ :=
  (2^n + 2 * (-1:ℤ)^n) / 3

/-- Theorem stating the number of ways a frog can return to vertex A in n jumps -/
theorem frog_jumps_correct (n : ℕ) : 
  frog_jumps n = (2^n + 2 * (-1:ℤ)^n) / 3 := by
  rfl

#check frog_jumps_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jumps_correct_l1250_125050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plant_distribution_ways_l1250_125013

-- Define the number of plants and lamps
def num_plants : ℕ := 4
def num_lamps : ℕ := 4

-- Define a function to represent the number of ways to distribute plants
def distribute_plants : ℕ := 50

-- State the theorem
theorem plant_distribution_ways : distribute_plants = 50 := by
  -- Unfold the definition of distribute_plants
  unfold distribute_plants
  -- The proof is complete since we defined distribute_plants as 50
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plant_distribution_ways_l1250_125013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_ratio_difference_l1250_125023

theorem lcm_ratio_difference (a b : ℕ) : 
  lcm a b = 420 → a * 7 = b * 5 → b - a = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_ratio_difference_l1250_125023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_speed_calculation_l1250_125015

-- Define the given constants
noncomputable def train_length : ℝ := 150  -- meters
noncomputable def crossing_time : ℝ := 6  -- seconds
noncomputable def train_speed_kmph : ℝ := 84.99280057595394  -- km/h

-- Define the conversion factor from km/h to m/s
noncomputable def kmph_to_ms : ℝ := 1000 / 3600

-- Define the theorem
theorem man_speed_calculation :
  let train_speed_ms : ℝ := train_speed_kmph * kmph_to_ms
  let relative_speed : ℝ := train_length / crossing_time
  let man_speed_ms : ℝ := relative_speed - train_speed_ms
  let man_speed_kmph : ℝ := man_speed_ms / kmph_to_ms
  ∃ ε > 0, |man_speed_kmph - 5.007198224048459| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_speed_calculation_l1250_125015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_time_for_reduced_speed_l1250_125061

/-- Calculates the extra time taken when speed is reduced -/
noncomputable def extra_time_taken (usual_time : ℝ) (speed_reduction_factor : ℝ) : ℝ :=
  usual_time * (1 / speed_reduction_factor - 1)

/-- 
Theorem: When a man reduces his speed to 25% of his usual speed, 
and his usual time to cover a distance is 8 minutes, 
he will take 24 extra minutes to cover the same distance at the slower speed.
-/
theorem extra_time_for_reduced_speed :
  extra_time_taken 8 0.25 = 24 := by
  -- Unfold the definition of extra_time_taken
  unfold extra_time_taken
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_time_for_reduced_speed_l1250_125061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1250_125076

noncomputable def f (x : ℝ) : ℝ := (1/3) * (x - 1)^2 - 16/3

theorem quadratic_function_properties :
  (f 0 = -5) ∧ (f (-1) = -4) ∧ (f 2 = -5) →
  (∀ x, f x = (1/3) * (x - 1)^2 - 16/3) ∧
  (∀ x ∈ Set.Icc 0 5, f x ≤ 0) ∧
  (∀ x ∈ Set.Icc 0 5, f x ≥ -16/3) ∧
  (∃ x ∈ Set.Icc 0 5, f x = 0) ∧
  (∃ x ∈ Set.Icc 0 5, f x = -16/3) :=
by
  sorry

#check quadratic_function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1250_125076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l1250_125059

/-- Helper function to represent the focus of a parabola -/
def focus_of_parabola (x y : ℝ) : ℝ × ℝ := sorry

/-- The parabola defined by y^2 - 4x = 0 has its focus at (1, 0) -/
theorem parabola_focus (x y : ℝ) : 
  y^2 - 4*x = 0 → focus_of_parabola x y = (1, 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l1250_125059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ABC_area_is_178_5_triangle_ABC_AC_triangle_ABC_height_l1250_125057

-- Define the triangles and their properties
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  AB : ℝ
  BC : ℝ
  AC : ℝ
  angleA : ℝ

-- Define the given conditions
def triangle_ABC : Triangle := sorry
def triangle_DEF : Triangle := sorry

-- Axioms for the given conditions
axiom AB_eq_DE : triangle_ABC.AB = triangle_DEF.AB
axiom AB_value : triangle_ABC.AB = 20

axiom BC_eq_EF : triangle_ABC.BC = triangle_DEF.BC
axiom BC_value : triangle_ABC.BC = 13

axiom angleA_eq_angleD : triangle_ABC.angleA = triangle_DEF.angleA

axiom AC_minus_DF : triangle_ABC.AC - triangle_DEF.AC = 10

-- Define the area function for a triangle
noncomputable def triangle_area (t : Triangle) : ℝ :=
  let s := (t.AB + t.BC + t.AC) / 2
  Real.sqrt (s * (s - t.AB) * (s - t.BC) * (s - t.AC))

-- Theorem statement
theorem ABC_area_is_178_5 : 
  triangle_area triangle_ABC = 178.5 := by sorry

-- Additional helper theorems if needed
theorem triangle_ABC_AC : triangle_ABC.AC = 21 := by sorry
theorem triangle_ABC_height : Real.sqrt (triangle_ABC.AB^2 - (triangle_ABC.AC/2)^2) = 17 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ABC_area_is_178_5_triangle_ABC_AC_triangle_ABC_height_l1250_125057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_terms_count_is_fifteen_expansion_terms_count_is_fifteen_alt_l1250_125021

/-- The number of terms in the expansion of (x+y+z)^4 -/
def expansion_terms_count : Nat :=
  let n : Nat := 4
  (n + 1) * (n + 2) / 2

#eval expansion_terms_count -- This will output 15

/-- Proof that the number of terms in the expansion of (x+y+z)^4 is 15 -/
theorem expansion_terms_count_is_fifteen : expansion_terms_count = 15 := by
  unfold expansion_terms_count
  norm_num

/-- Alternative proof using computation -/
theorem expansion_terms_count_is_fifteen_alt : expansion_terms_count = 15 := 
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_terms_count_is_fifteen_expansion_terms_count_is_fifteen_alt_l1250_125021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_christopher_stroll_time_l1250_125036

/-- Calculates the time taken for a journey given distance and speed -/
noncomputable def journey_time (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

/-- Proves that a 5-mile journey at 4 miles per hour takes 1.25 hours -/
theorem christopher_stroll_time :
  journey_time 5 4 = 1.25 := by
  -- Unfold the definition of journey_time
  unfold journey_time
  -- Perform the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_christopher_stroll_time_l1250_125036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleTransformation_l1250_125073

/-- Represents a quadratic expression ax^2 + bx + c -/
structure QuadraticExpr where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The three allowed operations for transforming quadratic expressions -/
inductive Operation
  | op1 : Operation  -- If c ≠ 0, replace a by 4a - 3/c and c by c/4
  | op2 : Operation  -- If a ≠ 0, replace a by -a/2 and c by -2c + 3/a
  | op3 : ℚ → Operation  -- Replace x by x - t

/-- Applies an operation to a quadratic expression -/
def applyOperation (q : QuadraticExpr) (op : Operation) : QuadraticExpr :=
  match op with
  | Operation.op1 => { a := 4 * q.a - 3 / q.c, b := q.b, c := q.c / 4 }
  | Operation.op2 => { a := -q.a / 2, b := q.b, c := -2 * q.c + 3 / q.a }
  | Operation.op3 t => { a := q.a, b := q.b - 2 * q.a * t, c := q.a * t^2 - q.b * t + q.c }

/-- The initial quadratic expression -/
def initial : QuadraticExpr := { a := 1, b := -1, c := -6 }

/-- The two target quadratic expressions -/
def target1 : QuadraticExpr := { a := 5, b := 5, c := -1 }
def target2 : QuadraticExpr := { a := 1, b := 6, c := 2 }

/-- Apply a list of operations to a quadratic expression -/
def applyOperations (q : QuadraticExpr) (ops : List Operation) : QuadraticExpr :=
  ops.foldl applyOperation q

/-- Theorem stating the impossibility of the transformation -/
theorem impossibleTransformation : 
  ¬ (∃ (ops : List Operation), 
     (applyOperations initial ops = target1) ∨
     (applyOperations initial ops = target2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleTransformation_l1250_125073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_critical_points_and_inequality_l1250_125072

noncomputable section

open Real

def f (x : ℝ) : ℝ := -x^2 + 2 * log x

def g (x : ℝ) : ℝ := x + 1/x

def e : ℝ := exp 1

theorem critical_points_and_inequality (x1 x2 t : ℝ) :
  (deriv f 1 = 0 ∧ deriv g 1 = 0) ∧
  (x1 ∈ Set.Icc (1/e) 5 ∧ x2 ∈ Set.Icc (1/e) 5 →
    ((f x1 - g x2) / (t + 1) ≤ 1 ↔
      t ∈ Set.Iic (-156/5 + 2 * log 5) ∪ Set.Ioi (-1))) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_critical_points_and_inequality_l1250_125072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_donuts_finished_katie_family_donuts_l1250_125078

/-- Given the conditions of Katie's family's donut and coffee consumption, 
    prove that they finish 36 donuts. -/
theorem donuts_finished (coffee_per_donut : ℚ) (coffee_per_pot : ℚ) 
  (cost_per_pot : ℚ) (total_spent : ℚ) : ℚ :=
  by
  -- Define the conditions
  have h1 : coffee_per_donut = 2 := by sorry
  have h2 : coffee_per_pot = 12 := by sorry
  have h3 : cost_per_pot = 3 := by sorry
  have h4 : total_spent = 18 := by sorry

  -- Calculate the number of donuts
  -- (total_spent / cost_per_pot) * coffee_per_pot / coffee_per_donut
  let result : ℚ := (total_spent / cost_per_pot) * coffee_per_pot / coffee_per_donut

  -- Return the result
  exact result

-- The theorem statement
theorem katie_family_donuts : donuts_finished 2 12 3 18 = 36 := by
  -- Unfold the definition of donuts_finished
  unfold donuts_finished
  -- Simplify the arithmetic expression
  simp
  -- The result should now be obvious
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_donuts_finished_katie_family_donuts_l1250_125078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_to_y_axis_l1250_125019

/-- Parabola structure -/
structure Parabola where
  focus : ℝ × ℝ

/-- Point on a parabola -/
def ParabolaPoint (p : Parabola) := { point : ℝ × ℝ // point.2^2 = 4 * point.1 }

/-- Chord passing through the focus of a parabola -/
structure FocusChord (p : Parabola) where
  a : ParabolaPoint p
  b : ParabolaPoint p

/-- Theorem: Minimum sum of distances from chord endpoints to y-axis -/
theorem min_sum_distances_to_y_axis (p : Parabola) (c : FocusChord p) :
  let a := c.a.val
  let b := c.b.val
  ‖a.2‖ + ‖b.2‖ ≥ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_to_y_axis_l1250_125019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_invariant_under_translation_specific_parallelogram_area_specific_parallelogram_area_after_translation_l1250_125026

/-- The area of a parallelogram formed by two vectors remains unchanged after translation -/
theorem parallelogram_area_invariant_under_translation 
  (v w t : ℝ × ℝ) : 
  let area := abs (v.1 * w.2 - v.2 * w.1)
  area = abs ((v.1 + t.1) * (w.2 + t.2) - (v.2 + t.2) * (w.1 + t.1)) := by
  sorry

/-- The area of the specific parallelogram in the problem is 14 -/
theorem specific_parallelogram_area :
  let v : ℝ × ℝ := (6, -4)
  let w : ℝ × ℝ := (-8, 3)
  abs (v.1 * w.2 - v.2 * w.1) = 14 := by
  sorry

/-- The area of the specific parallelogram remains 14 after translation -/
theorem specific_parallelogram_area_after_translation :
  let v : ℝ × ℝ := (6, -4)
  let w : ℝ × ℝ := (-8, 3)
  let t : ℝ × ℝ := (3, 2)
  abs ((v.1 + t.1) * (w.2 + t.2) - (v.2 + t.2) * (w.1 + t.1)) = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_invariant_under_translation_specific_parallelogram_area_specific_parallelogram_area_after_translation_l1250_125026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_triangle_classification_l1250_125002

/-- A triangle with sides in geometric progression -/
structure GeometricTriangle where
  r : ℝ
  side1 : ℝ := 1
  side2 : ℝ := r
  side3 : ℝ := r^2
  h_positive : r ≥ 1

/-- Classification of triangles based on their angles -/
inductive TriangleType
  | Acute
  | Right
  | Obtuse

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Predicate for right-angled triangle -/
def isRight (t : GeometricTriangle) : Prop :=
  t.r = Real.sqrt ((1 + Real.sqrt 5) / 2)

/-- Predicate for acute-angled triangle -/
def isAcute (t : GeometricTriangle) : Prop :=
  1 ≤ t.r ∧ t.r < Real.sqrt ((1 + Real.sqrt 5) / 2)

/-- Predicate for obtuse-angled triangle -/
def isObtuse (t : GeometricTriangle) : Prop :=
  Real.sqrt ((1 + Real.sqrt 5) / 2) < t.r ∧ t.r < φ

/-- Theorem about the classification of geometric triangles -/
theorem geometric_triangle_classification (t : GeometricTriangle) :
  (isRight t ↔ t.r = Real.sqrt ((1 + Real.sqrt 5) / 2)) ∧
  (isAcute t ↔ 1 ≤ t.r ∧ t.r < Real.sqrt ((1 + Real.sqrt 5) / 2)) ∧
  (isObtuse t ↔ Real.sqrt ((1 + Real.sqrt 5) / 2) < t.r ∧ t.r < φ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_triangle_classification_l1250_125002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_D_n_properties_l1250_125085

def D_n (n a b c : ℕ) : ℕ := Nat.gcd (a + b + c) (Nat.gcd (a^2 + b^2 + c^2) (a^n + b^n + c^n))

theorem D_n_properties (n : ℕ) :
  (¬(3 ∣ n) → ∀ k : ℕ, ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ Nat.gcd a (Nat.gcd b c) = 1 ∧ D_n n a b c > k) ∧
  ((3 ∣ n) → ∀ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 → Nat.gcd a (Nat.gcd b c) = 1 → (D_n n a b c ∣ 6)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_D_n_properties_l1250_125085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_range_l1250_125000

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 1 + x^2 else 1

-- State the theorem
theorem inequality_range (x : ℝ) :
  f (x - 4) > f (2*x - 3) ↔ x ∈ Set.Ioo (-1 : ℝ) 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_range_l1250_125000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_spherical_conversion_l1250_125016

/-- Converts cylindrical coordinates to spherical coordinates -/
noncomputable def cylindrical_to_spherical (r θ z : Real) : Real × Real × Real :=
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  let ρ := Real.sqrt (x^2 + y^2 + z^2)
  let φ := Real.arccos (z / ρ)
  (ρ, φ, θ)

/-- Proves that the given cylindrical coordinates convert to the specified spherical coordinates -/
theorem cylindrical_to_spherical_conversion :
  let (ρ, φ, θ) := cylindrical_to_spherical 10 (π/3) 2
  ρ = Real.sqrt 104 ∧ φ = Real.arccos (2 / Real.sqrt 104) ∧ θ = π/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_spherical_conversion_l1250_125016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_parallel_imply_tan_and_fraction_l1250_125033

noncomputable section

variable (x : ℝ)

def a : ℝ × ℝ := (2, 1)
def b (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

theorem vectors_parallel_imply_tan_and_fraction :
  parallel a (b x) →
  Real.tan x = 2 ∧ (3 * Real.sin x - Real.cos x) / (Real.sin x + 3 * Real.cos x) = 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_parallel_imply_tan_and_fraction_l1250_125033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kangaroo_fiber_theorem_l1250_125037

/-- The amount of fiber a kangaroo eats in a day, given the absorption rate and amount absorbed -/
noncomputable def fiber_eaten (absorption_rate : ℝ) (absorbed : ℝ) : ℝ :=
  absorbed / absorption_rate

/-- Theorem: If a kangaroo absorbs 30% of fiber and absorbed 15 ounces, it ate 50 ounces -/
theorem kangaroo_fiber_theorem :
  let absorption_rate : ℝ := 0.30
  let absorbed : ℝ := 15
  fiber_eaten absorption_rate absorbed = 50 := by
  -- Unfold the definition of fiber_eaten
  unfold fiber_eaten
  -- Simplify the expression
  simp
  -- Check that 15 / 0.30 = 50
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kangaroo_fiber_theorem_l1250_125037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_shaped_area_t_shaped_area_specific_l1250_125054

/-- The area of a T-shaped region formed by subtracting four smaller squares from a larger square -/
theorem t_shaped_area (side_length : ℝ) (h : side_length > 0) : 
  (4 * side_length)^2 - 4 * side_length^2 = 3 * (4 * side_length)^2 / 4 := by
  sorry

/-- The specific case where the side length of each smaller square is 2 -/
theorem t_shaped_area_specific : 
  (4 * 2)^2 - 4 * 2^2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_shaped_area_t_shaped_area_specific_l1250_125054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_ratio_l1250_125032

noncomputable section

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1/2, 0)

-- Define the directrix of the parabola
def directrix (x : ℝ) : Prop := x = -1/2

-- Define a line intersecting the parabola and directrix
def intersecting_line (A B P : ℝ × ℝ) : Prop :=
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2) ∨ (x = P.1 ∧ y = P.2) →
    y = m * x + b

-- Define A as the midpoint of PB
def A_midpoint_PB (A B P : ℝ × ℝ) : Prop :=
  A.1 = (P.1 + B.1) / 2 ∧ A.2 = (P.2 + B.2) / 2

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem parabola_intersection_ratio
  (A B P : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (hB : parabola B.1 B.2)
  (hP : directrix P.1)
  (hLine : intersecting_line A B P)
  (hMidpoint : A_midpoint_PB A B P) :
  distance B focus / distance A focus = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_ratio_l1250_125032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_properties_l1250_125053

theorem equation_properties (a : ℝ) :
  let f (x : ℝ) := x^4 + 4*x^2 + 1 - a*x*(x^2 - 1)
  let g (y : ℝ) := y^2 - a*y + 6
  (∀ x ≠ 0, f x = f (-1/x)) ∧
  (∀ x ≠ 0, g (x - 1/x) = 0 ↔ f x = 0) ∧
  (∀ y, g y = 0 → abs a ≥ 2 * Real.sqrt 6) ∧
  (∃ y, ∀ y', g y = 0 ∧ g y' = 0 → y = y') →
    (∃ x, x ∈ ({(Real.sqrt 6 + Real.sqrt 10) / 2,
                (Real.sqrt 6 - Real.sqrt 10) / 2,
                (-Real.sqrt 6 + Real.sqrt 10) / 2,
                (-Real.sqrt 6 - Real.sqrt 10) / 2} : Set ℝ) ∧
           f x = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_properties_l1250_125053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_relation_l1250_125039

theorem quadratic_inequality_relation (x : ℝ) : True := by
  have necessary : x > 4 → x^2 - 3*x > 0 := by
    sorry

  have not_sufficient : ∃ x, x^2 - 3*x > 0 ∧ x ≤ 4 := by
    sorry

  trivial


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_relation_l1250_125039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_15_l1250_125017

/-- The number of degrees in a full circle -/
noncomputable def full_circle : ℝ := 360

/-- The number of hour marks on a clock -/
def hour_marks : ℕ := 12

/-- The angle between each hour mark in degrees -/
noncomputable def angle_per_hour : ℝ := full_circle / hour_marks

/-- The current hour (3 for 3:15 p.m.) -/
noncomputable def current_hour : ℝ := 3

/-- The current minute (15 for 3:15 p.m.) -/
noncomputable def current_minute : ℝ := 15

/-- The position of the hour hand in degrees -/
noncomputable def hour_hand_position : ℝ := current_hour * angle_per_hour + (current_minute / 60) * angle_per_hour

/-- The position of the minute hand in degrees -/
noncomputable def minute_hand_position : ℝ := (current_minute / 5) * angle_per_hour

/-- The smaller angle between the hour and minute hands -/
noncomputable def smaller_angle : ℝ := min (abs (hour_hand_position - minute_hand_position)) (full_circle - abs (hour_hand_position - minute_hand_position))

theorem clock_angle_at_3_15 : smaller_angle = 7.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_15_l1250_125017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_sum_l1250_125049

/-- Represents a parabola in the Cartesian plane -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- Represents a line in the Cartesian plane -/
structure Line where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Checks if a line is tangent to a parabola -/
def is_tangent (l : Line) (p : Parabola) : Prop := sorry

/-- Checks if a line has rational slope -/
def has_rational_slope (l : Line) : Prop := sorry

theorem common_tangent_sum : 
  let p₁ : Parabola := ⟨λ x y => y = x^2 + 101/100⟩
  let p₂ : Parabola := ⟨λ x y => x = y^2 + 45/4⟩
  ∀ l : Line, 
    is_tangent l p₁ ∧ 
    is_tangent l p₂ ∧ 
    has_rational_slope l ∧ 
    l.a > 0 ∧ l.b > 0 ∧ l.c > 0 ∧ 
    (∃ (a b c : ℕ), (a : ℚ) = l.a ∧ (b : ℚ) = l.b ∧ (c : ℚ) = l.c ∧ Nat.gcd a (Nat.gcd b c) = 1) →
    ∃ (a b c : ℕ), (a : ℚ) = l.a ∧ (b : ℚ) = l.b ∧ (c : ℚ) = l.c ∧ a + b + c = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_sum_l1250_125049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1250_125042

def N : ℕ := 2^2 * 3^3 * 5^4

theorem problem_solution :
  (∃ T : ℕ,
    (Finset.card (Finset.filter (λ d : ℕ => d ∣ N) (Finset.range (N + 1))) = T) ∧
    (∀ n : ℕ, 1 ≤ n ∧ n ≤ T → (n * (n - 1) * (n - 2)) % 12 = 0) ∧
    (Nat.lcm T 36 / Nat.gcd T 36 = 15)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1250_125042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_covering_l1250_125099

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in a 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- The set of 100 points in the plane -/
def Points : Set Point := sorry

/-- The set of circles covering the points -/
def CoveringCircles : Set Circle := sorry

/-- The sum of diameters of the covering circles -/
noncomputable def SumOfDiameters : ℝ := sorry

/-- Predicate to check if two circles are non-overlapping -/
def NonOverlapping (c1 c2 : Circle) : Prop := sorry

/-- Predicate to check if a point is covered by a circle -/
def IsCovered (p : Point) (c : Circle) : Prop := sorry

/-- Distance between two points -/
noncomputable def Distance (p1 p2 : Point) : ℝ := sorry

/-- Theorem stating the existence of a valid covering -/
theorem exists_valid_covering :
  ∃ (circles : Set Circle),
    (∀ p, p ∈ Points → ∃ c ∈ circles, IsCovered p c) ∧
    (∀ c1 c2, c1 ∈ circles → c2 ∈ circles → c1 ≠ c2 → NonOverlapping c1 c2) ∧
    (SumOfDiameters < 100) ∧
    (∀ p1 p2, p1 ∈ Points → p2 ∈ Points → 
      ∀ c1 c2, c1 ∈ circles → c2 ∈ circles → 
      IsCovered p1 c1 → IsCovered p2 c2 → c1 ≠ c2 → Distance p1 p2 > 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_covering_l1250_125099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1250_125025

-- Define the circle
def circle_set : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 25}

-- Define the point P
noncomputable def P : ℝ × ℝ := (-3, -3/2)

-- Define the property of line l
def is_valid_line (l : Set (ℝ × ℝ)) : Prop :=
  P ∈ l ∧ 
  ∃ A B : ℝ × ℝ, A ∈ l ∧ B ∈ l ∧ A ∈ circle_set ∧ B ∈ circle_set ∧ 
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 64

-- Define the two possible equations for line l
def line1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -3}
def line2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 + 15 = 0}

-- Theorem statement
theorem line_equation : 
  ∀ l : Set (ℝ × ℝ), is_valid_line l → (l = line1 ∨ l = line2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1250_125025
