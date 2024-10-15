import Mathlib

namespace NUMINAMATH_CALUDE_ruth_school_days_l982_98296

/-- Ruth's school schedule -/
def school_schedule (days_per_week : ℝ) : Prop :=
  let hours_per_day : ℝ := 8
  let math_class_fraction : ℝ := 0.25
  let math_hours_per_week : ℝ := 10
  (hours_per_day * days_per_week * math_class_fraction = math_hours_per_week)

theorem ruth_school_days : ∃ (d : ℝ), school_schedule d ∧ d = 5 := by
  sorry

end NUMINAMATH_CALUDE_ruth_school_days_l982_98296


namespace NUMINAMATH_CALUDE_intersecting_chords_theorem_chord_intersection_equality_l982_98252

/-- A circle in a 2D plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a 2D plane. -/
def Point := ℝ × ℝ

/-- The distance between two points. -/
def distance (p q : Point) : ℝ := sorry

/-- Checks if a point lies on a circle. -/
def onCircle (c : Circle) (p : Point) : Prop := 
  distance c.center p = c.radius

/-- Represents a chord of a circle. -/
structure Chord (c : Circle) where
  p1 : Point
  p2 : Point
  h1 : onCircle c p1
  h2 : onCircle c p2

/-- Theorem: For two intersecting chords and a line through their intersection point,
    the product of the distances from the intersection point to the endpoints of one chord
    is equal to the product of the distances from the intersection point to the endpoints of the other chord. -/
theorem intersecting_chords_theorem (c : Circle) (ab cd : Chord c) (e f g h i : Point) : 
  onCircle c f ∧ onCircle c g ∧ onCircle c h ∧ onCircle c i →
  distance e f * distance e g = distance e h * distance e i :=
sorry

/-- Main theorem to prove -/
theorem chord_intersection_equality (c : Circle) (ab cd : Chord c) (e f g h i : Point) : 
  onCircle c f ∧ onCircle c g ∧ onCircle c h ∧ onCircle c i →
  distance f g = distance h i :=
sorry

end NUMINAMATH_CALUDE_intersecting_chords_theorem_chord_intersection_equality_l982_98252


namespace NUMINAMATH_CALUDE_two_digit_product_problem_l982_98298

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def swap_digits (n : ℕ) : ℕ := (n % 10) * 10 + (n / 10)

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10

def units_digit (n : ℕ) : ℕ := n % 10

theorem two_digit_product_problem :
  ∃ (x y z : ℕ),
    is_two_digit x ∧
    is_two_digit y ∧
    y = swap_digits x ∧
    x ≠ y ∧
    z = x * y ∧
    100 ≤ z ∧ z < 1000 ∧
    hundreds_digit z = units_digit z ∧
    ((x = 12 ∧ y = 21) ∨ (x = 21 ∧ y = 12)) ∧
    z = 252 :=
by
  sorry

end NUMINAMATH_CALUDE_two_digit_product_problem_l982_98298


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l982_98208

/-- An arithmetic sequence. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem stating that a_15 = 24 for the given arithmetic sequence. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arithmetic : ArithmeticSequence a)
    (h_sum : a 3 + a 13 = 20)
    (h_a2 : a 2 = -2) : 
  a 15 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l982_98208


namespace NUMINAMATH_CALUDE_abs_negative_six_l982_98267

theorem abs_negative_six : |(-6 : ℤ)| = 6 := by sorry

end NUMINAMATH_CALUDE_abs_negative_six_l982_98267


namespace NUMINAMATH_CALUDE_three_bushes_same_flowers_l982_98255

theorem three_bushes_same_flowers (garden : Finset ℕ) (flower_count : ℕ → ℕ) :
  garden.card = 201 →
  (∀ bush ∈ garden, 1 ≤ flower_count bush ∧ flower_count bush ≤ 100) →
  ∃ n : ℕ, ∃ bush₁ bush₂ bush₃ : ℕ,
    bush₁ ∈ garden ∧ bush₂ ∈ garden ∧ bush₃ ∈ garden ∧
    bush₁ ≠ bush₂ ∧ bush₁ ≠ bush₃ ∧ bush₂ ≠ bush₃ ∧
    flower_count bush₁ = n ∧ flower_count bush₂ = n ∧ flower_count bush₃ = n :=
by sorry

end NUMINAMATH_CALUDE_three_bushes_same_flowers_l982_98255


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l982_98286

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > 2 ∧ b > 2 → a + b > 4) ∧
  (∃ a b : ℝ, a + b > 4 ∧ ¬(a > 2 ∧ b > 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l982_98286


namespace NUMINAMATH_CALUDE_max_teams_with_10_points_l982_98234

/-- Represents a football tournament with the given conditions -/
structure FootballTournament where
  total_teams : Nat
  points_per_win : Nat
  points_per_draw : Nat
  points_per_loss : Nat
  target_points : Nat

/-- The maximum number of teams that can achieve the target points -/
def max_teams_with_target_points (tournament : FootballTournament) : Nat :=
  sorry

/-- Theorem stating the maximum number of teams that can score exactly 10 points -/
theorem max_teams_with_10_points :
  let tournament := FootballTournament.mk 17 3 1 0 10
  max_teams_with_target_points tournament = 11 := by
  sorry

end NUMINAMATH_CALUDE_max_teams_with_10_points_l982_98234


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_exists_l982_98276

/-- A positive single-digit integer is a natural number between 1 and 9, inclusive. -/
def PositiveSingleDigit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

/-- The quadratic equation x^2 - (2A)x + AB = 0 has positive integer solutions. -/
def HasPositiveIntegerSolutions (A B : ℕ) : Prop :=
  ∃ x : ℕ, x > 0 ∧ x^2 - (2 * A) * x + A * B = 0

theorem quadratic_equation_solution_exists :
  ∃ A B : ℕ, PositiveSingleDigit A ∧ PositiveSingleDigit B ∧ HasPositiveIntegerSolutions A B := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_exists_l982_98276


namespace NUMINAMATH_CALUDE_coin_sequence_count_l982_98229

/-- Represents a coin toss sequence -/
def CoinSequence := List Bool

/-- Counts the number of specific subsequences in a coin sequence -/
def countSubsequences (seq : CoinSequence) : Nat × Nat × Nat × Nat :=
  sorry

/-- Checks if a coin sequence has the required number of subsequences -/
def hasRequiredSubsequences (seq : CoinSequence) : Bool :=
  let (hh, ht, th, tt) := countSubsequences seq
  hh = 3 ∧ ht = 2 ∧ th = 5 ∧ tt = 6

/-- Generates all possible 17-toss coin sequences -/
def allSequences : List CoinSequence :=
  sorry

/-- Counts the number of sequences with required subsequences -/
def countValidSequences : Nat :=
  (allSequences.filter hasRequiredSubsequences).length

theorem coin_sequence_count : countValidSequences = 840 := by
  sorry

end NUMINAMATH_CALUDE_coin_sequence_count_l982_98229


namespace NUMINAMATH_CALUDE_equation_solution_l982_98230

theorem equation_solution (x : ℚ) : 1 / (x + 1/5) = 5/3 → x = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l982_98230


namespace NUMINAMATH_CALUDE_total_highlighters_l982_98259

theorem total_highlighters (pink : ℕ) (yellow : ℕ) (blue : ℕ)
  (h1 : pink = 9)
  (h2 : yellow = 8)
  (h3 : blue = 5) :
  pink + yellow + blue = 22 := by
  sorry

end NUMINAMATH_CALUDE_total_highlighters_l982_98259


namespace NUMINAMATH_CALUDE_tetrahedron_dihedral_angle_l982_98211

/-- Regular tetrahedron with given dimensions -/
structure RegularTetrahedron where
  base_side_length : ℝ
  side_edge_length : ℝ

/-- Plane that divides the tetrahedron's volume equally -/
structure DividingPlane where
  tetrahedron : RegularTetrahedron
  passes_through_AB : Bool
  divides_volume_equally : Bool

/-- The cosine of the dihedral angle between the dividing plane and the base -/
def dihedral_angle_cosine (plane : DividingPlane) : ℝ :=
  sorry

theorem tetrahedron_dihedral_angle 
  (t : RegularTetrahedron) 
  (p : DividingPlane) 
  (h1 : t.base_side_length = 1) 
  (h2 : t.side_edge_length = 2) 
  (h3 : p.tetrahedron = t) 
  (h4 : p.passes_through_AB = true) 
  (h5 : p.divides_volume_equally = true) : 
  dihedral_angle_cosine p = 2 * Real.sqrt 15 / 15 :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_dihedral_angle_l982_98211


namespace NUMINAMATH_CALUDE_wall_volume_theorem_l982_98216

/-- Calculates the volume of a rectangular wall given its width and height-to-width and length-to-height ratios -/
def wall_volume (width : ℝ) (height_ratio : ℝ) (length_ratio : ℝ) : ℝ :=
  width * (height_ratio * width) * (length_ratio * height_ratio * width)

/-- Theorem: The volume of a wall with width 4m, height 6 times its width, and length 7 times its height is 16128 cubic meters -/
theorem wall_volume_theorem :
  wall_volume 4 6 7 = 16128 := by
  sorry

end NUMINAMATH_CALUDE_wall_volume_theorem_l982_98216


namespace NUMINAMATH_CALUDE_inequality_proof_l982_98231

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) : 
  (3*x^2 - x)/(1 + x^2) + (3*y^2 - y)/(1 + y^2) + (3*z^2 - z)/(1 + z^2) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l982_98231


namespace NUMINAMATH_CALUDE_rationalize_denominator_l982_98254

theorem rationalize_denominator :
  ∀ (x : ℝ), x > 0 → (5 / (x^(1/3) + (27 * x)^(1/3))) = (5 * (9 * x)^(1/3)) / 12 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l982_98254


namespace NUMINAMATH_CALUDE_happy_children_count_is_30_l982_98284

/-- Represents the number of children in different categories -/
structure ChildrenCount where
  total : Nat
  sad : Nat
  neither : Nat
  boys : Nat
  girls : Nat
  happyBoys : Nat
  sadGirls : Nat
  neitherBoys : Nat

/-- Calculates the number of happy children given the conditions -/
def happyChildrenCount (c : ChildrenCount) : Nat :=
  c.total - c.sad - c.neither

/-- Theorem stating that the number of happy children is 30 -/
theorem happy_children_count_is_30 (c : ChildrenCount) 
  (h1 : c.total = 60)
  (h2 : c.sad = 10)
  (h3 : c.neither = 20)
  (h4 : c.boys = 22)
  (h5 : c.girls = 38)
  (h6 : c.happyBoys = 6)
  (h7 : c.sadGirls = 4)
  (h8 : c.neitherBoys = 10) :
  happyChildrenCount c = 30 := by
  sorry

#check happy_children_count_is_30

end NUMINAMATH_CALUDE_happy_children_count_is_30_l982_98284


namespace NUMINAMATH_CALUDE_digit_sum_eleven_l982_98242

/-- Represents a digit in base 10 -/
def Digit := Fin 10

/-- Defines a two-digit number -/
def TwoDigitNumber (a b : Digit) : ℕ := 10 * a.val + b.val

/-- Defines a three-digit number -/
def ThreeDigitNumber (c d e : Digit) : ℕ := 100 * c.val + 10 * d.val + e.val

/-- Checks if three digits are consecutive and increasing -/
def ConsecutiveIncreasing (c d e : Digit) : Prop :=
  d.val = c.val + 1 ∧ e.val = d.val + 1

theorem digit_sum_eleven 
  (a b c d e : Digit) 
  (h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (h2 : TwoDigitNumber a b * TwoDigitNumber c b = ThreeDigitNumber c d e)
  (h3 : ConsecutiveIncreasing c d e) :
  a.val + b.val + c.val + d.val + e.val = 11 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_eleven_l982_98242


namespace NUMINAMATH_CALUDE_max_value_of_e_l982_98251

def b (n : ℕ) : ℤ := (10^n - 1) / 7

def e (n : ℕ) : ℕ := Nat.gcd (Int.natAbs (b n)) (Int.natAbs (b (n + 2)))

theorem max_value_of_e : ∀ n : ℕ, e n ≤ 99 ∧ ∃ m : ℕ, e m = 99 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_e_l982_98251


namespace NUMINAMATH_CALUDE_parallel_to_plane_not_always_parallel_l982_98253

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallelLine : Line → Line → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- State the theorem
theorem parallel_to_plane_not_always_parallel :
  ∃ (m n : Line) (α : Plane),
    parallelLinePlane m α ∧ parallelLinePlane n α ∧ ¬ parallelLine m n := by
  sorry

end NUMINAMATH_CALUDE_parallel_to_plane_not_always_parallel_l982_98253


namespace NUMINAMATH_CALUDE_dye_mixture_amount_l982_98236

/-- The total amount of mixture obtained by combining a fraction of water and a fraction of vinegar -/
def mixture_amount (water_total : ℚ) (vinegar_total : ℚ) (water_fraction : ℚ) (vinegar_fraction : ℚ) : ℚ :=
  water_fraction * water_total + vinegar_fraction * vinegar_total

/-- Theorem stating that the mixture amount for the given problem is 27 liters -/
theorem dye_mixture_amount :
  mixture_amount 20 18 (3/5) (5/6) = 27 := by
  sorry

end NUMINAMATH_CALUDE_dye_mixture_amount_l982_98236


namespace NUMINAMATH_CALUDE_coin_probability_theorem_l982_98280

theorem coin_probability_theorem (p q : ℝ) : 
  p + q = 1 →
  0 ≤ p ∧ p ≤ 1 →
  0 ≤ q ∧ q ≤ 1 →
  (Nat.choose 10 5 : ℝ) * p^5 * q^5 = (Nat.choose 10 6 : ℝ) * p^6 * q^4 →
  p = 6/11 :=
by sorry

end NUMINAMATH_CALUDE_coin_probability_theorem_l982_98280


namespace NUMINAMATH_CALUDE_kishore_rent_expense_l982_98281

def monthly_salary (savings : ℕ) : ℕ := savings * 10

def total_expenses_excluding_rent : ℕ := 1500 + 4500 + 2500 + 2000 + 5200

def rent_expense (salary savings : ℕ) : ℕ :=
  salary - (total_expenses_excluding_rent + savings)

theorem kishore_rent_expense :
  rent_expense (monthly_salary 2300) 2300 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_kishore_rent_expense_l982_98281


namespace NUMINAMATH_CALUDE_sqrt_product_quotient_sqrt_27_times_sqrt_32_div_sqrt_6_l982_98244

theorem sqrt_product_quotient :
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
  Real.sqrt (a * b) / Real.sqrt c = Real.sqrt a * Real.sqrt b / Real.sqrt c :=
by sorry

theorem sqrt_27_times_sqrt_32_div_sqrt_6 :
  Real.sqrt 27 * Real.sqrt 32 / Real.sqrt 6 = 12 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_quotient_sqrt_27_times_sqrt_32_div_sqrt_6_l982_98244


namespace NUMINAMATH_CALUDE_sum_200th_row_l982_98279

/-- Represents the sum of numbers in the nth row of the triangular array -/
def f (n : ℕ) : ℕ := sorry

/-- The triangular array has the following properties:
    1. The sides contain numbers 0, 1, 2, 3, ...
    2. Each interior number is the sum of two adjacent numbers in the previous row -/
axiom array_properties : True

/-- The sum of numbers in the nth row follows the recurrence relation:
    f(n) = 2 * f(n-1) + 2 for n ≥ 2 -/
axiom recurrence_relation (n : ℕ) (h : n ≥ 2) : f n = 2 * f (n-1) + 2

/-- The sum of numbers in the 200th row of the triangular array is 2^200 - 2 -/
theorem sum_200th_row : f 200 = 2^200 - 2 := by sorry

end NUMINAMATH_CALUDE_sum_200th_row_l982_98279


namespace NUMINAMATH_CALUDE_sqrt_2_times_sqrt_12_minus_2_between_2_and_3_l982_98246

theorem sqrt_2_times_sqrt_12_minus_2_between_2_and_3 :
  2 < Real.sqrt 2 * Real.sqrt 12 - 2 ∧ Real.sqrt 2 * Real.sqrt 12 - 2 < 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2_times_sqrt_12_minus_2_between_2_and_3_l982_98246


namespace NUMINAMATH_CALUDE_sum_770_product_not_divisible_l982_98225

theorem sum_770_product_not_divisible (a b : ℕ) : 
  a + b = 770 → ¬(770 ∣ (a * b)) := by
sorry

end NUMINAMATH_CALUDE_sum_770_product_not_divisible_l982_98225


namespace NUMINAMATH_CALUDE_equation_solution_l982_98226

theorem equation_solution (x : ℝ) :
  x ≠ 2/3 →
  ((4*x + 3) / (3*x^2 + 4*x - 4) = 3*x / (3*x - 2)) ↔
  (x = (-1 + Real.sqrt 10) / 3 ∨ x = (-1 - Real.sqrt 10) / 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l982_98226


namespace NUMINAMATH_CALUDE_part_one_part_two_l982_98237

-- Part 1
theorem part_one (f : ℝ → ℝ) (a : ℝ) 
  (h : ∀ x > 0, f x = x - a * Real.log x)
  (h1 : ∀ x > 0, f x ≥ 1) : a = 1 := by
  sorry

-- Part 2
theorem part_two (x₁ x₂ : ℝ) 
  (h1 : x₁ > 0)
  (h2 : x₂ > 0)
  (h3 : Real.exp x₁ + Real.log x₂ > x₁ + x₂) :
  Real.exp x₁ + x₂ > 2 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l982_98237


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l982_98210

-- Define the quadratic function f(x)
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

-- Define g(x) in terms of f(x) and m
def g (m : ℝ) (x : ℝ) : ℝ := m * f x + 1

-- Theorem statement
theorem quadratic_function_properties :
  (∀ x : ℝ, f x ≥ -4) ∧
  (f (-2) = -3) ∧
  (f 0 = -3) ∧
  (∀ m : ℝ, m < 0 → ∃! x : ℝ, x ≤ 1 ∧ g m x = 0) ∧
  (∀ m : ℝ, m > 0 →
    (m ≤ 8/7 → (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3/2 → |g m x| ≤ 9*m/4 + 1)) ∧
    (m > 8/7 → (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3/2 → |g m x| ≤ 4*m - 1))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l982_98210


namespace NUMINAMATH_CALUDE_sequence_a_properties_l982_98232

def sequence_a (n : ℕ) : ℝ := sorry

def sum_S (n : ℕ) : ℝ := sorry

axiom arithmetic_mean (n : ℕ) : sequence_a n = (sum_S n + 2) / 2

theorem sequence_a_properties :
  (sequence_a 1 = 2 ∧ sequence_a 2 = 4) ∧
  (∀ n : ℕ, n ≥ 1 → sequence_a n = 2^n) := by sorry

end NUMINAMATH_CALUDE_sequence_a_properties_l982_98232


namespace NUMINAMATH_CALUDE_yard_area_l982_98290

/-- The area of a rectangular yard with square cutouts -/
theorem yard_area (length width cutout_side : ℕ) (num_cutouts : ℕ) : 
  length = 20 → 
  width = 18 → 
  cutout_side = 4 → 
  num_cutouts = 2 → 
  length * width - num_cutouts * cutout_side * cutout_side = 328 := by
  sorry

end NUMINAMATH_CALUDE_yard_area_l982_98290


namespace NUMINAMATH_CALUDE_perpendicular_bisector_b_value_l982_98272

/-- A line that is a perpendicular bisector of a line segment -/
structure PerpendicularBisector where
  a : ℝ
  b : ℝ
  c : ℝ
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ
  is_perpendicular_bisector : True  -- This is a placeholder for the actual condition

/-- The theorem stating that b = 12 for the given perpendicular bisector -/
theorem perpendicular_bisector_b_value :
  ∀ (pb : PerpendicularBisector), 
  pb.a = 1 ∧ pb.b = 1 ∧ pb.p1 = (2, 4) ∧ pb.p2 = (8, 10) → 
  pb.c = 12 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_b_value_l982_98272


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l982_98224

theorem right_triangle_side_length 
  (west_distance : ℝ) 
  (total_distance : ℝ) 
  (h1 : west_distance = 10) 
  (h2 : total_distance = 14.142135623730951) : 
  ∃ (north_distance : ℝ), 
    north_distance^2 + west_distance^2 = total_distance^2 ∧ 
    north_distance = 10 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l982_98224


namespace NUMINAMATH_CALUDE_a_value_l982_98258

theorem a_value (a : ℝ) : 3 ∈ ({1, -a^2, a-1} : Set ℝ) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_a_value_l982_98258


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l982_98249

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 6*x + 5
  ∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l982_98249


namespace NUMINAMATH_CALUDE_hidden_dots_count_l982_98282

/-- The sum of numbers on a single die -/
def dieDots : ℕ := 21

/-- The number of dice stacked -/
def numDice : ℕ := 4

/-- The visible numbers on the stacked dice -/
def visibleNumbers : List ℕ := [1, 2, 2, 3, 3, 5, 6]

/-- The number of visible faces -/
def visibleFaces : ℕ := 7

/-- The number of hidden faces -/
def hiddenFaces : ℕ := 17

theorem hidden_dots_count : 
  numDice * dieDots - visibleNumbers.sum = 62 :=
sorry

end NUMINAMATH_CALUDE_hidden_dots_count_l982_98282


namespace NUMINAMATH_CALUDE_triangle_inequality_l982_98207

theorem triangle_inequality (x y z : ℝ) 
  (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (h_sum : x + y + z = 1) : 
  x^2 + y^2 + z^2 ≥ x^3 + y^3 + z^3 + 6*x*y*z := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l982_98207


namespace NUMINAMATH_CALUDE_balloons_bought_at_park_l982_98299

theorem balloons_bought_at_park (allan_balloons jake_initial_balloons : ℕ) 
  (h1 : allan_balloons = 6)
  (h2 : jake_initial_balloons = 3)
  (h3 : ∃ (x : ℕ), jake_initial_balloons + x = allan_balloons + 1) :
  ∃ (x : ℕ), x = 4 ∧ jake_initial_balloons + x = allan_balloons + 1 := by
sorry

end NUMINAMATH_CALUDE_balloons_bought_at_park_l982_98299


namespace NUMINAMATH_CALUDE_circle_tangent_to_two_lines_through_point_circle_through_two_points_tangent_to_line_circle_tangent_to_two_lines_and_circle_l982_98277

-- Define the basic types
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define the tangency and passing through relations
def tangent_to_line (c : Circle) (l : Line) : Prop := sorry

def passes_through (c : Circle) (p : Point) : Prop := sorry

def tangent_to_circle (c1 : Circle) (c2 : Circle) : Prop := sorry

-- Part a
theorem circle_tangent_to_two_lines_through_point 
  (l1 l2 : Line) (A : Point) : 
  ∃ (S : Circle), tangent_to_line S l1 ∧ tangent_to_line S l2 ∧ passes_through S A := by
  sorry

-- Part b
theorem circle_through_two_points_tangent_to_line 
  (A B : Point) (l : Line) :
  ∃ (S : Circle), passes_through S A ∧ passes_through S B ∧ tangent_to_line S l := by
  sorry

-- Part c
theorem circle_tangent_to_two_lines_and_circle 
  (l1 l2 : Line) (S_bar : Circle) :
  ∃ (S : Circle), tangent_to_line S l1 ∧ tangent_to_line S l2 ∧ tangent_to_circle S S_bar := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_to_two_lines_through_point_circle_through_two_points_tangent_to_line_circle_tangent_to_two_lines_and_circle_l982_98277


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_S_l982_98214

def S : Finset ℕ := {8, 88, 888, 8888, 88888, 888888, 8888888, 88888888, 888888888}

def arithmetic_mean (s : Finset ℕ) : ℚ :=
  (s.sum id) / s.card

def digits (n : ℕ) : Finset ℕ :=
  sorry

theorem arithmetic_mean_of_S :
  arithmetic_mean S = 109728268 ∧
  ∀ d : ℕ, d < 10 → (d ∉ digits 109728268 ↔ d = 4) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_S_l982_98214


namespace NUMINAMATH_CALUDE_prob_sum_greater_than_four_is_five_sixths_l982_98203

/-- The number of possible outcomes when tossing two dice -/
def total_outcomes : ℕ := 36

/-- The number of outcomes where the sum is 4 or less -/
def outcomes_sum_4_or_less : ℕ := 6

/-- The probability of getting a sum greater than four when tossing two dice -/
def prob_sum_greater_than_four : ℚ := 5 / 6

theorem prob_sum_greater_than_four_is_five_sixths :
  prob_sum_greater_than_four = 1 - (outcomes_sum_4_or_less : ℚ) / total_outcomes :=
by sorry

end NUMINAMATH_CALUDE_prob_sum_greater_than_four_is_five_sixths_l982_98203


namespace NUMINAMATH_CALUDE_customers_who_left_l982_98274

/-- Proves that 12 customers left a waiter's section given the initial and final conditions -/
theorem customers_who_left (initial_customers : ℕ) (people_per_table : ℕ) (remaining_tables : ℕ) : 
  initial_customers = 44 → people_per_table = 8 → remaining_tables = 4 →
  initial_customers - (people_per_table * remaining_tables) = 12 :=
by sorry

end NUMINAMATH_CALUDE_customers_who_left_l982_98274


namespace NUMINAMATH_CALUDE_original_number_l982_98297

theorem original_number : ∃ x : ℚ, 213 * x = 3408 ∧ x = 16 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l982_98297


namespace NUMINAMATH_CALUDE_circle_area_l982_98292

theorem circle_area (circumference : ℝ) (area : ℝ) : 
  circumference = 36 → area = 324 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circle_area_l982_98292


namespace NUMINAMATH_CALUDE_bucket_weight_calculation_l982_98287

/-- Given an initial weight of shells and an additional weight of shells,
    calculate the total weight of shells in the bucket. -/
def total_weight (initial_weight additional_weight : ℕ) : ℕ :=
  initial_weight + additional_weight

/-- Theorem stating that given 5 pounds of initial weight and 12 pounds of additional weight,
    the total weight of shells in the bucket is 17 pounds. -/
theorem bucket_weight_calculation :
  total_weight 5 12 = 17 := by
  sorry

end NUMINAMATH_CALUDE_bucket_weight_calculation_l982_98287


namespace NUMINAMATH_CALUDE_no_primes_divisible_by_77_l982_98285

theorem no_primes_divisible_by_77 : ¬∃ p : ℕ, Nat.Prime p ∧ 77 ∣ p := by
  sorry

end NUMINAMATH_CALUDE_no_primes_divisible_by_77_l982_98285


namespace NUMINAMATH_CALUDE_common_point_of_alternating_ap_lines_l982_98283

/-- Represents a line in 2D space with equation ax + by = c --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point (x, y) lies on a given line --/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y = l.c

/-- Defines an alternating arithmetic progression for a, b, c --/
def is_alternating_ap (a b c : ℝ) : Prop :=
  ∃ d : ℝ, b = a - d ∧ c = a + d

theorem common_point_of_alternating_ap_lines :
  ∀ l : Line, is_alternating_ap l.a l.b l.c → l.contains 1 (-1) :=
sorry

end NUMINAMATH_CALUDE_common_point_of_alternating_ap_lines_l982_98283


namespace NUMINAMATH_CALUDE_concert_attendance_l982_98294

theorem concert_attendance (num_buses : ℕ) (students_per_bus : ℕ) 
  (h1 : num_buses = 8) (h2 : students_per_bus = 45) : 
  num_buses * students_per_bus = 360 := by
  sorry

end NUMINAMATH_CALUDE_concert_attendance_l982_98294


namespace NUMINAMATH_CALUDE_stationery_problem_l982_98266

theorem stationery_problem (georgia lorene : ℕ) 
  (h1 : lorene = 3 * georgia) 
  (h2 : georgia = lorene - 50) : 
  georgia = 25 := by
sorry

end NUMINAMATH_CALUDE_stationery_problem_l982_98266


namespace NUMINAMATH_CALUDE_barbie_coconuts_l982_98219

theorem barbie_coconuts (total_coconuts : ℕ) (trips : ℕ) (bruno_capacity : ℕ) 
  (h1 : total_coconuts = 144)
  (h2 : trips = 12)
  (h3 : bruno_capacity = 8) :
  ∃ barbie_capacity : ℕ, 
    barbie_capacity * trips + bruno_capacity * trips = total_coconuts ∧ 
    barbie_capacity = 4 := by
  sorry

end NUMINAMATH_CALUDE_barbie_coconuts_l982_98219


namespace NUMINAMATH_CALUDE_roberts_chocolates_l982_98268

theorem roberts_chocolates (nickel_chocolates : ℕ) (robert_extra : ℕ) : 
  nickel_chocolates = 4 → robert_extra = 9 → nickel_chocolates + robert_extra = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_roberts_chocolates_l982_98268


namespace NUMINAMATH_CALUDE_total_amount_l982_98289

/-- Represents the distribution of money among three parties -/
structure MoneyDistribution where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The problem statement -/
def money_problem (d : MoneyDistribution) : Prop :=
  d.y = 45 ∧                    -- y's share is 45 rupees
  d.y = 0.45 * d.x ∧            -- y gets 0.45 rupees for each rupee x gets
  d.z = 0.50 * d.x              -- z gets 0.50 rupees for each rupee x gets

/-- The theorem to prove -/
theorem total_amount (d : MoneyDistribution) :
  money_problem d → d.x + d.y + d.z = 195 :=
by
  sorry


end NUMINAMATH_CALUDE_total_amount_l982_98289


namespace NUMINAMATH_CALUDE_book_reading_increase_l982_98235

theorem book_reading_increase (matt_last_year matt_this_year pete_last_year pete_this_year : ℕ) 
  (h1 : pete_last_year = 2 * matt_last_year)
  (h2 : pete_this_year = 2 * pete_last_year)
  (h3 : pete_last_year + pete_this_year = 300)
  (h4 : matt_this_year = 75) :
  (matt_this_year - matt_last_year) * 100 / matt_last_year = 50 := by
  sorry

end NUMINAMATH_CALUDE_book_reading_increase_l982_98235


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_property_l982_98212

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Sum of three consecutive terms in a sequence -/
def SumOfThree (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  a n + a (n + 1) + a (n + 2)

theorem geometric_sequence_sum_property
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_sum1 : SumOfThree a 1 = 8)
  (h_sum2 : SumOfThree a 4 = -4) :
  SumOfThree a 7 = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_property_l982_98212


namespace NUMINAMATH_CALUDE_scientific_notation_130_billion_l982_98295

theorem scientific_notation_130_billion : 130000000000 = 1.3 * (10 ^ 11) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_130_billion_l982_98295


namespace NUMINAMATH_CALUDE_square_ratio_theorem_l982_98233

theorem square_ratio_theorem : ∃ (a b c : ℕ), 
  (a > 0 ∧ b > 0 ∧ c > 0) ∧ 
  (180 : ℝ) / 45 = (a * (b.sqrt : ℝ) / c : ℝ)^2 ∧ 
  a + b + c = 4 := by
sorry

end NUMINAMATH_CALUDE_square_ratio_theorem_l982_98233


namespace NUMINAMATH_CALUDE_sum_O_eq_1000_l982_98221

/-- O(n) is the sum of the odd digits of n -/
def O (n : ℕ) : ℕ := sorry

/-- The sum of O(n) for n from 1 to 200 -/
def sum_O : ℕ := (Finset.range 200).sum (fun i => O (i + 1))

/-- Theorem: The sum of O(n) for n from 1 to 200 is equal to 1000 -/
theorem sum_O_eq_1000 : sum_O = 1000 := by sorry

end NUMINAMATH_CALUDE_sum_O_eq_1000_l982_98221


namespace NUMINAMATH_CALUDE_distance_AB_l982_98261

noncomputable def C₁ (θ : Real) : Real := 2 * Real.sqrt 3 * Real.cos θ + 2 * Real.sin θ

noncomputable def C₂ (θ : Real) : Real := 2 * Real.cos θ + 2 * Real.sqrt 3 * Real.sin θ

theorem distance_AB : 
  let θ := Real.pi / 3
  let ρ₁ := C₁ θ
  let ρ₂ := C₂ θ
  abs (ρ₁ - ρ₂) = 4 - 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_distance_AB_l982_98261


namespace NUMINAMATH_CALUDE_sqrt_144000_l982_98278

theorem sqrt_144000 : Real.sqrt 144000 = 120 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_144000_l982_98278


namespace NUMINAMATH_CALUDE_equation_solution_l982_98200

theorem equation_solution : ∃ x : ℝ, 
  Real.sqrt (9 - Real.sqrt (16 + 8*x)) + Real.sqrt (5 - Real.sqrt (5 + x)) = 3 + Real.sqrt 5 :=
by
  use 4
  sorry

end NUMINAMATH_CALUDE_equation_solution_l982_98200


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l982_98206

def quadratic_inequality (a : ℝ) (x : ℝ) : Prop :=
  (a - 1) * x^2 + 2 * (a - 1) * x - 4 ≥ 0

def empty_solution_set (a : ℝ) : Prop :=
  ∀ x : ℝ, ¬(quadratic_inequality a x)

theorem quadratic_inequality_range :
  ∀ a : ℝ, empty_solution_set a ↔ -3 < a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l982_98206


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l982_98227

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_product (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 4 * a 5 * a 6 = 27 →
  a 1 * a 9 = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l982_98227


namespace NUMINAMATH_CALUDE_distance_between_runners_l982_98238

-- Define the race length in kilometers
def race_length_km : ℝ := 1

-- Define Arianna's position in meters when Ethan finished
def arianna_position : ℝ := 184

-- Theorem to prove the distance between Ethan and Arianna
theorem distance_between_runners : 
  (race_length_km * 1000) - arianna_position = 816 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_runners_l982_98238


namespace NUMINAMATH_CALUDE_product_remainder_by_10_l982_98256

theorem product_remainder_by_10 : (2583 * 7462 * 93215) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_by_10_l982_98256


namespace NUMINAMATH_CALUDE_monthly_fee_is_two_l982_98271

/-- Represents the monthly phone bill structure -/
structure PhoneBill where
  monthlyFee : ℝ
  perMinuteRate : ℝ
  minutesUsed : ℕ
  totalBill : ℝ

/-- Proves that the monthly fee is $2 given the specified conditions -/
theorem monthly_fee_is_two (bill : PhoneBill) 
    (h1 : bill.totalBill = bill.monthlyFee + bill.perMinuteRate * bill.minutesUsed)
    (h2 : bill.perMinuteRate = 0.12)
    (h3 : bill.totalBill = 23.36)
    (h4 : bill.minutesUsed = 178) :
    bill.monthlyFee = 2 := by
  sorry


end NUMINAMATH_CALUDE_monthly_fee_is_two_l982_98271


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l982_98204

theorem expression_simplification_and_evaluation :
  ∀ a : ℤ, -1 < a ∧ a < Real.sqrt 5 ∧ a ≠ 0 ∧ a ≠ 1 →
  let expr := ((a + 1) / (2 * a - 2) - 5 / (2 * a^2 - 2) - (a + 3) / (2 * a + 2)) / (a^2 / (a^2 - 1))
  expr = -1 / (2 * a^2) ∧
  (a = 2 → expr = -1/8) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l982_98204


namespace NUMINAMATH_CALUDE_bubble_gum_cost_l982_98257

theorem bubble_gum_cost (total_pieces : ℕ) (total_cost : ℕ) (cost_per_piece : ℕ) : 
  total_pieces = 136 → total_cost = 2448 → cost_per_piece = 18 → 
  total_cost = total_pieces * cost_per_piece :=
by sorry

end NUMINAMATH_CALUDE_bubble_gum_cost_l982_98257


namespace NUMINAMATH_CALUDE_max_product_constraint_l982_98239

theorem max_product_constraint (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 2*b = 1) :
  (∀ a' b' : ℝ, 0 < a' → 0 < b' → a' + 2*b' = 1 → a'*b' ≤ a*b) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constraint_l982_98239


namespace NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l982_98265

theorem unique_solution_for_exponential_equation :
  ∀ x y : ℕ, x ≥ 1 → y ≥ 1 → (3^x - 2^y = 7 ↔ x = 2 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l982_98265


namespace NUMINAMATH_CALUDE_richard_david_age_difference_l982_98223

-- Define the ages of the three sons
def david_age : ℕ := 14
def scott_age : ℕ := david_age - 8
def richard_age : ℕ := scott_age * 2 + 8

-- Define the conditions
theorem richard_david_age_difference :
  richard_age - david_age = 6 :=
by
  -- Proof goes here
  sorry

#check richard_david_age_difference

end NUMINAMATH_CALUDE_richard_david_age_difference_l982_98223


namespace NUMINAMATH_CALUDE_first_player_wins_n_9_first_player_wins_n_10_l982_98241

/-- Represents the state of the game board -/
inductive BoardState
  | Minuses (n : ℕ)
  | Pluses (n : ℕ)

/-- Represents a move in the game -/
inductive Move
  | ChangeOne
  | ChangeTwo

/-- Defines the game rules and winning condition -/
def gameRules (n : ℕ) (player : ℕ) (board : BoardState) (move : Move) : Prop :=
  match board with
  | BoardState.Minuses m =>
      (move = Move.ChangeOne ∧ m > 0) ∨
      (move = Move.ChangeTwo ∧ m > 1)
  | BoardState.Pluses _ => false

/-- Defines the winning condition -/
def isWinningState (board : BoardState) : Prop :=
  match board with
  | BoardState.Minuses 0 => true
  | _ => false

/-- Theorem: The first player has a winning strategy for n = 9 -/
theorem first_player_wins_n_9 :
  ∃ (strategy : ℕ → BoardState → Move),
    ∀ (opponent_strategy : ℕ → BoardState → Move),
      isWinningState (BoardState.Minuses 0) ∧
      (∀ (t : ℕ),
        gameRules 9 (t % 2) (BoardState.Minuses (9 - t)) (strategy t (BoardState.Minuses (9 - t)))) :=
sorry

/-- Theorem: The first player has a winning strategy for n = 10 -/
theorem first_player_wins_n_10 :
  ∃ (strategy : ℕ → BoardState → Move),
    ∀ (opponent_strategy : ℕ → BoardState → Move),
      isWinningState (BoardState.Minuses 0) ∧
      (∀ (t : ℕ),
        gameRules 10 (t % 2) (BoardState.Minuses (10 - t)) (strategy t (BoardState.Minuses (10 - t)))) :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_n_9_first_player_wins_n_10_l982_98241


namespace NUMINAMATH_CALUDE_factorization_proof_l982_98262

theorem factorization_proof (m x y a : ℝ) : 
  (-3 * m^3 + 12 * m = -3 * m * (m + 2) * (m - 2)) ∧ 
  (2 * x^2 * y - 8 * x * y + 8 * y = 2 * y * (x - 2)^2) ∧ 
  (a^4 + 3 * a^2 - 4 = (a^2 + 4) * (a + 1) * (a - 1)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l982_98262


namespace NUMINAMATH_CALUDE_circumcircle_area_l982_98263

/-- An isosceles triangle with two sides of length 6 and a base of length 4 -/
structure IsoscelesTriangle where
  side : ℝ
  base : ℝ
  is_isosceles : side = 6 ∧ base = 4

/-- A circle passing through the vertices of an isosceles triangle -/
def CircumCircle (t : IsoscelesTriangle) : ℝ → Prop :=
  fun area => area = 16 * Real.pi

/-- The theorem stating that the area of the circumcircle of the given isosceles triangle is 16π -/
theorem circumcircle_area (t : IsoscelesTriangle) : 
  ∃ area, CircumCircle t area :=
sorry

end NUMINAMATH_CALUDE_circumcircle_area_l982_98263


namespace NUMINAMATH_CALUDE_fair_die_probability_at_least_one_six_l982_98217

theorem fair_die_probability_at_least_one_six (n : ℕ) (p : ℚ) : 
  n = 3 → p = 1/6 → (1 : ℚ) - (1 - p)^n = 91/216 := by
  sorry

end NUMINAMATH_CALUDE_fair_die_probability_at_least_one_six_l982_98217


namespace NUMINAMATH_CALUDE_percentage_of_number_l982_98213

theorem percentage_of_number (x : ℚ) (y : ℕ) (z : ℕ) :
  (x / 100) * y = z → x = 33 + 1/3 → y = 210 → z = 70 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_number_l982_98213


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l982_98243

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum_odd : a 1 + a 3 + a 5 = 105)
  (h_sum_even : a 2 + a 4 + a 6 = 99) :
  ∃ d : ℝ, d = -2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l982_98243


namespace NUMINAMATH_CALUDE_min_cubes_for_valid_config_l982_98270

/-- Represents a cube with two opposite sides having protruding snaps and four sides with receptacle holes. -/
structure SpecialCube where
  snaps : Fin 2 → Bool
  holes : Fin 4 → Bool

/-- A configuration of special cubes. -/
def CubeConfiguration := List SpecialCube

/-- Checks if a configuration has no visible protruding snaps and only shows receptacle holes on visible surfaces. -/
def isValidConfiguration (config : CubeConfiguration) : Bool :=
  sorry

/-- The theorem stating that 6 is the minimum number of cubes required for a valid configuration. -/
theorem min_cubes_for_valid_config :
  ∃ (config : CubeConfiguration),
    config.length = 6 ∧ isValidConfiguration config ∧
    ∀ (smallerConfig : CubeConfiguration),
      smallerConfig.length < 6 → ¬isValidConfiguration smallerConfig :=
  sorry

end NUMINAMATH_CALUDE_min_cubes_for_valid_config_l982_98270


namespace NUMINAMATH_CALUDE_triangle_side_value_l982_98275

noncomputable section

-- Define the triangle ABC
def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  -- Conditions for a valid triangle
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a < b + c ∧ b < a + c ∧ c < a + b

-- Theorem statement
theorem triangle_side_value
  (A B C : ℝ) (a b c : ℝ)
  (h_triangle : triangle A B C a b c)
  (h_angle : A = 2 * C)
  (h_side_c : c = 2)
  (h_side_a : a^2 = 4*b - 4) :
  a = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_value_l982_98275


namespace NUMINAMATH_CALUDE_inequality_proof_l982_98215

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b)^2 / 2 + (a + b) / 4 ≥ a * Real.sqrt b + b * Real.sqrt a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l982_98215


namespace NUMINAMATH_CALUDE_monopoly_houses_theorem_l982_98240

structure Player where
  name : String
  initialHouses : ℕ
  deriving Repr

def seanTransactions (houses : ℕ) : ℕ :=
  houses - 15 + 18

def karenTransactions (houses : ℕ) : ℕ :=
  0 + 10 + 8 + 15

def markTransactions (houses : ℕ) : ℕ :=
  houses + 12 - 25 - 15

def lucyTransactions (houses : ℕ) : ℕ :=
  houses - 8 + 6 - 20

def finalHouses (player : Player) : ℕ :=
  match player.name with
  | "Sean" => seanTransactions player.initialHouses
  | "Karen" => karenTransactions player.initialHouses
  | "Mark" => markTransactions player.initialHouses
  | "Lucy" => lucyTransactions player.initialHouses
  | _ => player.initialHouses

theorem monopoly_houses_theorem (sean karen mark lucy : Player)
  (h1 : sean.name = "Sean" ∧ sean.initialHouses = 45)
  (h2 : karen.name = "Karen" ∧ karen.initialHouses = 30)
  (h3 : mark.name = "Mark" ∧ mark.initialHouses = 55)
  (h4 : lucy.name = "Lucy" ∧ lucy.initialHouses = 35) :
  finalHouses sean = 48 ∧
  finalHouses karen = 33 ∧
  finalHouses mark = 27 ∧
  finalHouses lucy = 13 := by
  sorry

end NUMINAMATH_CALUDE_monopoly_houses_theorem_l982_98240


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l982_98218

theorem quadratic_roots_sum_product (m n : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - m * x + n = 0 ∧ 3 * y^2 - m * y + n = 0 ∧ x ≠ y) →
  (∃ x y : ℝ, 3 * x^2 - m * x + n = 0 ∧ 3 * y^2 - m * y + n = 0 ∧ x + y = 9 ∧ x * y = 20) →
  m + n = 87 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l982_98218


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l982_98205

/-- Given an arithmetic sequence {a_n} where a_2 = 10 and a_4 = 18, 
    the common difference d equals 4. -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) -- a is a sequence of real numbers indexed by natural numbers
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) -- a is arithmetic
  (h_a2 : a 2 = 10) -- a_2 = 10
  (h_a4 : a 4 = 18) -- a_4 = 18
  : a 3 - a 2 = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l982_98205


namespace NUMINAMATH_CALUDE_complex_intersection_l982_98222

theorem complex_intersection (z : ℂ) (k : ℝ) : 
  k > 0 → 
  Complex.abs (z - 4) = 3 * Complex.abs (z + 4) →
  Complex.abs z = k →
  (∃! z', Complex.abs (z' - 4) = 3 * Complex.abs (z' + 4) ∧ Complex.abs z' = k) →
  k = 4 ∨ k = 14 := by
sorry

end NUMINAMATH_CALUDE_complex_intersection_l982_98222


namespace NUMINAMATH_CALUDE_trapezoid_area_l982_98264

theorem trapezoid_area (top_base bottom_base height : ℝ) 
  (h1 : top_base = 4)
  (h2 : bottom_base = 8)
  (h3 : height = 3) :
  (top_base + bottom_base) * height / 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l982_98264


namespace NUMINAMATH_CALUDE_carpet_area_calculation_l982_98291

/-- Calculates the required carpet area in square yards for a rectangular bedroom and square closet, including wastage. -/
theorem carpet_area_calculation 
  (bedroom_length : ℝ) 
  (bedroom_width : ℝ) 
  (closet_side : ℝ) 
  (wastage_rate : ℝ) 
  (feet_per_yard : ℝ) 
  (h1 : bedroom_length = 15)
  (h2 : bedroom_width = 10)
  (h3 : closet_side = 6)
  (h4 : wastage_rate = 0.1)
  (h5 : feet_per_yard = 3) :
  let bedroom_area := (bedroom_length / feet_per_yard) * (bedroom_width / feet_per_yard)
  let closet_area := (closet_side / feet_per_yard) ^ 2
  let total_area := bedroom_area + closet_area
  let required_area := total_area * (1 + wastage_rate)
  required_area = 22.715 := by
  sorry


end NUMINAMATH_CALUDE_carpet_area_calculation_l982_98291


namespace NUMINAMATH_CALUDE_clips_for_huahuas_handkerchiefs_l982_98260

/-- The number of clips needed to hang handkerchiefs on clotheslines -/
def clips_needed (handkerchiefs : ℕ) (clotheslines : ℕ) : ℕ :=
  -- We define this function without implementation, as the problem doesn't provide the exact formula
  sorry

/-- Theorem stating the number of clips needed for the given scenario -/
theorem clips_for_huahuas_handkerchiefs :
  clips_needed 40 3 = 43 := by
  sorry

end NUMINAMATH_CALUDE_clips_for_huahuas_handkerchiefs_l982_98260


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l982_98248

/-- The perimeter of a rectangle with length 6 cm and width 4 cm is 20 cm. -/
theorem rectangle_perimeter : 
  let length : ℝ := 6
  let width : ℝ := 4
  let perimeter := 2 * (length + width)
  perimeter = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l982_98248


namespace NUMINAMATH_CALUDE_inverse_half_plus_sqrt_four_log_sum_minus_power_inverse_sum_with_sqrt_three_l982_98247

-- Part 1
theorem inverse_half_plus_sqrt_four (x y : ℝ) (h1 : x = 0.5) (h2 : y = 4) :
  x⁻¹ + y^(1/2) = 4 := by sorry

-- Part 2
theorem log_sum_minus_power (x y z : ℝ) (h1 : x = 2) (h2 : y = 5) (h3 : z = π / 23) :
  Real.log x / Real.log 10 + Real.log y / Real.log 10 - z^0 = 0 := by sorry

-- Part 3
theorem inverse_sum_with_sqrt_three (x : ℝ) (h : x = 3) :
  (2 - Real.sqrt x)⁻¹ + (2 + Real.sqrt x)⁻¹ = 4 := by sorry

end NUMINAMATH_CALUDE_inverse_half_plus_sqrt_four_log_sum_minus_power_inverse_sum_with_sqrt_three_l982_98247


namespace NUMINAMATH_CALUDE_square_area_with_inscribed_triangle_l982_98269

theorem square_area_with_inscribed_triangle (d : ℝ) (h : d = 16) : 
  let s := d / Real.sqrt 2
  let square_area := s^2
  square_area = 128 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_inscribed_triangle_l982_98269


namespace NUMINAMATH_CALUDE_bicycle_sales_theorem_l982_98250

/-- Represents the sales and pricing data for bicycle types A and B -/
structure BicycleSales where
  lastYearTotalSalesA : ℕ
  priceIncreaseA : ℕ
  purchasePriceA : ℕ
  purchasePriceB : ℕ
  sellingPriceB : ℕ
  totalPurchase : ℕ

/-- Calculates the selling price of type A bicycles this year -/
def sellingPriceA (data : BicycleSales) : ℕ :=
  sorry

/-- Calculates the optimal purchase plan to maximize profit -/
def optimalPurchasePlan (data : BicycleSales) : ℕ × ℕ :=
  sorry

/-- Main theorem stating the selling price of type A bicycles and the optimal purchase plan -/
theorem bicycle_sales_theorem (data : BicycleSales) 
  (h1 : data.lastYearTotalSalesA = 32000)
  (h2 : data.priceIncreaseA = 400)
  (h3 : data.purchasePriceA = 1100)
  (h4 : data.purchasePriceB = 1400)
  (h5 : data.sellingPriceB = 2400)
  (h6 : data.totalPurchase = 50)
  (h7 : ∀ (x y : ℕ), x + y = data.totalPurchase → y ≤ 2 * x) :
  sellingPriceA data = 2000 ∧ optimalPurchasePlan data = (17, 33) :=
sorry

end NUMINAMATH_CALUDE_bicycle_sales_theorem_l982_98250


namespace NUMINAMATH_CALUDE_selling_price_fraction_l982_98201

theorem selling_price_fraction (cost_price : ℝ) (original_selling_price : ℝ) : 
  original_selling_price = cost_price * (1 + 0.275) →
  ∃ (f : ℝ), f * original_selling_price = cost_price * (1 - 0.15) ∧ f = 17 / 25 :=
by
  sorry

end NUMINAMATH_CALUDE_selling_price_fraction_l982_98201


namespace NUMINAMATH_CALUDE_peanuts_in_box_l982_98293

/-- The number of peanuts in a box after adding more -/
def total_peanuts (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem: If a box initially contains 10 peanuts and 8 more peanuts are added,
    the total number of peanuts in the box is 18. -/
theorem peanuts_in_box : total_peanuts 10 8 = 18 := by
  sorry

end NUMINAMATH_CALUDE_peanuts_in_box_l982_98293


namespace NUMINAMATH_CALUDE_opposite_of_2023_l982_98245

-- Define the concept of opposite for integers
def opposite (n : ℤ) : ℤ := -n

-- Theorem statement
theorem opposite_of_2023 : opposite 2023 = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l982_98245


namespace NUMINAMATH_CALUDE_correct_side_for_significant_figures_l982_98288

/-- Represents the side from which we start counting significant figures -/
inductive Side
  | Left
  | Right
  | Front
  | Back

/-- Definition of significant figures for an approximate number -/
def significantFigures (number : ℕ) (startSide : Side) : ℕ :=
  sorry

/-- Theorem stating that the correct side to start from for significant figures is the left side -/
theorem correct_side_for_significant_figures :
  ∀ (number : ℕ), significantFigures number Side.Left = significantFigures number Side.Left :=
  sorry

end NUMINAMATH_CALUDE_correct_side_for_significant_figures_l982_98288


namespace NUMINAMATH_CALUDE_xixi_apples_count_l982_98220

/-- The number of students in Teacher Xixi's class -/
def xixi_students : ℕ := 12

/-- The number of students in Teacher Shanshan's class -/
def shanshan_students : ℕ := xixi_students

/-- The number of apples Teacher Xixi prepared -/
def xixi_apples : ℕ := 72

/-- The number of oranges Teacher Shanshan prepared -/
def shanshan_oranges : ℕ := 60

theorem xixi_apples_count : xixi_apples = 72 := by
  have h1 : xixi_apples = shanshan_students * 6 := sorry
  have h2 : shanshan_oranges = xixi_students * 3 + 12 := sorry
  have h3 : shanshan_oranges = shanshan_students * 5 := sorry
  sorry

end NUMINAMATH_CALUDE_xixi_apples_count_l982_98220


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l982_98228

theorem scientific_notation_equality : 21500000 = 2.15 * (10 ^ 7) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l982_98228


namespace NUMINAMATH_CALUDE_same_number_on_four_dice_l982_98202

theorem same_number_on_four_dice : 
  let n : ℕ := 6  -- number of sides on each die
  let k : ℕ := 4  -- number of dice
  (1 : ℚ) / n^(k-1) = (1 : ℚ) / 216 :=
by sorry

end NUMINAMATH_CALUDE_same_number_on_four_dice_l982_98202


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l982_98209

open Set

-- Define the sets A and B
def A : Set ℝ := {x | x < 5}
def B : Set ℝ := {x | x < 2}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (Bᶜ) = {x : ℝ | 2 ≤ x ∧ x < 5} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l982_98209


namespace NUMINAMATH_CALUDE_conditions_satisfied_l982_98273

-- Define the points and lengths
variable (P Q R S : ℝ) -- Representing points as real numbers for simplicity
variable (a b c k : ℝ)

-- State the conditions
axiom distinct_collinear : P < Q ∧ Q < R ∧ R < S
axiom positive_lengths : a > 0 ∧ b > 0 ∧ c > 0 ∧ k > 0
axiom length_PQ : Q - P = a
axiom length_PR : R - P = b
axiom length_PS : S - P = c
axiom b_relation : b = a + k

-- Triangle inequality conditions
axiom triangle_inequality1 : a + (b - a) > c - b
axiom triangle_inequality2 : (b - a) + (c - b) > a
axiom triangle_inequality3 : a + (c - b) > b - a

-- Theorem to prove
theorem conditions_satisfied :
  a < c / 2 ∧ b < 2 * a + c / 2 :=
sorry

end NUMINAMATH_CALUDE_conditions_satisfied_l982_98273
