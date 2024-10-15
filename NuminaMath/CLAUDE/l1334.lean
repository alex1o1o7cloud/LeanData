import Mathlib

namespace NUMINAMATH_CALUDE_symmetry_of_f_l1334_133419

def arithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def f (x a₁ a₂ : ℝ) : ℝ := |x - a₁| + |x - a₂|

theorem symmetry_of_f (a : ℕ → ℝ) (d : ℝ) (h : d ≠ 0) :
  arithmeticSequence a d →
  ∀ x : ℝ, f (((a 1) + (a 2)) / 2 - x) ((a 1) : ℝ) ((a 2) : ℝ) = 
           f (((a 1) + (a 2)) / 2 + x) ((a 1) : ℝ) ((a 2) : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_of_f_l1334_133419


namespace NUMINAMATH_CALUDE_inequality_proof_l1334_133465

theorem inequality_proof (a b c d : ℝ) :
  (a + b + c + d) / ((1 + a^2) * (1 + b^2) * (1 + c^2) * (1 + d^2)) < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1334_133465


namespace NUMINAMATH_CALUDE_corridor_length_is_95_meters_l1334_133484

/-- Represents the scale of a blueprint in meters per centimeter. -/
def blueprint_scale : ℝ := 10

/-- Represents the length of the corridor in the blueprint in centimeters. -/
def blueprint_corridor_length : ℝ := 9.5

/-- Calculates the real-life length of the corridor in meters. -/
def real_life_corridor_length : ℝ := blueprint_scale * blueprint_corridor_length

/-- Theorem stating that the real-life length of the corridor is 95 meters. -/
theorem corridor_length_is_95_meters : real_life_corridor_length = 95 := by
  sorry

end NUMINAMATH_CALUDE_corridor_length_is_95_meters_l1334_133484


namespace NUMINAMATH_CALUDE_circle_point_distance_relation_l1334_133413

/-- Given a circle with radius r and a point F constructed as described in the problem,
    prove the relationship between distances u and v from F to specific lines. -/
theorem circle_point_distance_relation (r u v : ℝ) : v^2 = u^3 / (2*r - u) := by
  sorry

end NUMINAMATH_CALUDE_circle_point_distance_relation_l1334_133413


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l1334_133494

theorem abs_sum_inequality (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) ↔ (-6.5 < x ∧ x < 3.5) :=
sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l1334_133494


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l1334_133403

theorem system_of_equations_solution (x y : ℚ) : 
  2 * x - 3 * y = 24 ∧ x + 2 * y = 15 → y = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l1334_133403


namespace NUMINAMATH_CALUDE_polar_equivalence_l1334_133401

/-- Two points in polar coordinates are equivalent if they represent the same point in the plane. -/
def polar_equivalent (r1 : ℝ) (θ1 : ℝ) (r2 : ℝ) (θ2 : ℝ) : Prop :=
  r1 * (Real.cos θ1) = r2 * (Real.cos θ2) ∧ r1 * (Real.sin θ1) = r2 * (Real.sin θ2)

/-- The theorem stating that (-3, 7π/6) is equivalent to (3, π/6) in polar coordinates. -/
theorem polar_equivalence :
  polar_equivalent (-3) (7 * Real.pi / 6) 3 (Real.pi / 6) ∧ 
  3 > 0 ∧ 
  0 ≤ Real.pi / 6 ∧ 
  Real.pi / 6 < 2 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_polar_equivalence_l1334_133401


namespace NUMINAMATH_CALUDE_sin_graph_shift_l1334_133455

noncomputable def f (x : ℝ) := Real.sin (2 * x)
noncomputable def g (x : ℝ) := Real.sin (2 * x + 1)

theorem sin_graph_shift :
  ∀ x : ℝ, g x = f (x + 1/2) := by sorry

end NUMINAMATH_CALUDE_sin_graph_shift_l1334_133455


namespace NUMINAMATH_CALUDE_no_integer_tangent_length_l1334_133474

theorem no_integer_tangent_length (t : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ 2 * π * r = 8 * π) →  -- Circle with circumference 8π
  (t^2 = (8*π/3) * π) →                    -- Tangent-secant relationship
  ¬(∃ (n : ℤ), t = n) :=                   -- No integer solution for t
by sorry

end NUMINAMATH_CALUDE_no_integer_tangent_length_l1334_133474


namespace NUMINAMATH_CALUDE_root_product_theorem_l1334_133491

theorem root_product_theorem (n r : ℝ) (c d : ℝ) : 
  c^2 - n*c + 3 = 0 → 
  d^2 - n*d + 3 = 0 → 
  (c + 2/d)^2 - r*(c + 2/d) + s = 0 → 
  (d + 2/c)^2 - r*(d + 2/c) + s = 0 → 
  s = 25/3 := by sorry

end NUMINAMATH_CALUDE_root_product_theorem_l1334_133491


namespace NUMINAMATH_CALUDE_carrot_harvest_calculation_l1334_133407

/-- Calculates the expected carrot harvest from a rectangular backyard --/
theorem carrot_harvest_calculation 
  (length_paces width_paces : ℕ) 
  (pace_to_feet : ℝ) 
  (carrot_yield_per_sqft : ℝ) : 
  length_paces = 25 → 
  width_paces = 30 → 
  pace_to_feet = 2.5 → 
  carrot_yield_per_sqft = 0.5 → 
  (length_paces : ℝ) * pace_to_feet * (width_paces : ℝ) * pace_to_feet * carrot_yield_per_sqft = 2343.75 := by
  sorry

#check carrot_harvest_calculation

end NUMINAMATH_CALUDE_carrot_harvest_calculation_l1334_133407


namespace NUMINAMATH_CALUDE_bromine_extraction_l1334_133467

-- Define the solubility of a substance in a solvent
def solubility (substance solvent : Type) : ℝ := sorry

-- Define the property of being immiscible
def immiscible (solvent1 solvent2 : Type) : Prop := sorry

-- Define the extraction process
def can_extract (substance from_solvent to_solvent : Type) : Prop := sorry

-- Define the substances and solvents
def bromine : Type := sorry
def water : Type := sorry
def benzene : Type := sorry
def soybean_oil : Type := sorry

-- Theorem statement
theorem bromine_extraction :
  (solubility bromine benzene > solubility bromine water) →
  (solubility bromine soybean_oil > solubility bromine water) →
  immiscible benzene water →
  immiscible soybean_oil water →
  (can_extract bromine water benzene ∨ can_extract bromine water soybean_oil) :=
by sorry

end NUMINAMATH_CALUDE_bromine_extraction_l1334_133467


namespace NUMINAMATH_CALUDE_complement_union_equals_set_l1334_133431

open Set

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {3, 4, 5}
def B : Set Nat := {1, 3, 6}

theorem complement_union_equals_set : 
  (U \ (A ∪ B)) = {2, 7} := by sorry

end NUMINAMATH_CALUDE_complement_union_equals_set_l1334_133431


namespace NUMINAMATH_CALUDE_dans_cards_after_purchase_l1334_133444

/-- The number of baseball cards Dan has after Sam's purchase -/
def dans_remaining_cards (initial_cards sam_bought : ℕ) : ℕ :=
  initial_cards - sam_bought

/-- Theorem: Dan's remaining cards is the difference between his initial cards and those Sam bought -/
theorem dans_cards_after_purchase (initial_cards sam_bought : ℕ) 
  (h : sam_bought ≤ initial_cards) : 
  dans_remaining_cards initial_cards sam_bought = initial_cards - sam_bought := by
  sorry

end NUMINAMATH_CALUDE_dans_cards_after_purchase_l1334_133444


namespace NUMINAMATH_CALUDE_joes_first_lift_weight_l1334_133468

theorem joes_first_lift_weight (total_weight first_lift second_lift : ℕ) 
  (h1 : total_weight = 900)
  (h2 : first_lift + second_lift = total_weight)
  (h3 : 2 * first_lift = second_lift + 300) :
  first_lift = 400 := by
  sorry

end NUMINAMATH_CALUDE_joes_first_lift_weight_l1334_133468


namespace NUMINAMATH_CALUDE_parabola_triangle_area_l1334_133412

/-- Given a parabola y² = 4x with focus F(1,0), and points A and B on the parabola
    such that FA = 2BF, the area of triangle OAB is 3√2/2. -/
theorem parabola_triangle_area (A B : ℝ × ℝ) :
  let C : ℝ × ℝ → Prop := λ p => p.2^2 = 4 * p.1
  let F : ℝ × ℝ := (1, 0)
  let O : ℝ × ℝ := (0, 0)
  C A ∧ C B ∧ 
  (∃ (t : ℝ), A = F + t • (A - F) ∧ B = F + t • (B - F)) ∧
  (A - F) = 2 • (F - B) →
  abs ((A.1 * B.2 - A.2 * B.1) / 2) = 3 * Real.sqrt 2 / 2 :=
by sorry


end NUMINAMATH_CALUDE_parabola_triangle_area_l1334_133412


namespace NUMINAMATH_CALUDE_square_area_proof_l1334_133441

theorem square_area_proof (x : ℝ) (h1 : 4 * x - 15 = 20 - 3 * x) : 
  (4 * x - 15) ^ 2 = 25 := by
sorry

end NUMINAMATH_CALUDE_square_area_proof_l1334_133441


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1334_133423

theorem inequality_solution_set (k : ℝ) : 
  (∀ x : ℝ, (1 + k / (x - 1) ≤ 0 ↔ x ∈ Set.Ici (-2) ∩ Set.Iio 1)) → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1334_133423


namespace NUMINAMATH_CALUDE_initial_average_production_l1334_133409

theorem initial_average_production (n : ℕ) (A : ℝ) (today_production : ℝ) (new_average : ℝ)
  (h1 : n = 5)
  (h2 : today_production = 90)
  (h3 : new_average = 65)
  (h4 : (n * A + today_production) / (n + 1) = new_average) :
  A = 60 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_production_l1334_133409


namespace NUMINAMATH_CALUDE_slopes_equal_implies_parallel_false_l1334_133481

/-- A line in a 2D plane --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Two lines are parallel if they have the same slope --/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

/-- Statement: If two lines have the same slope, they are parallel --/
theorem slopes_equal_implies_parallel_false :
  ¬ (∀ (l1 l2 : Line), l1.slope = l2.slope → parallel l1 l2) := by
  sorry

end NUMINAMATH_CALUDE_slopes_equal_implies_parallel_false_l1334_133481


namespace NUMINAMATH_CALUDE_graph_shift_l1334_133427

/-- Given a function f : ℝ → ℝ, prove that the graph of y = f(x + 2) - 1 
    is equivalent to shifting the graph of y = f(x) 2 units left and 1 unit down. -/
theorem graph_shift (f : ℝ → ℝ) (x y : ℝ) :
  y = f (x + 2) - 1 ↔ ∃ x₀ y₀ : ℝ, y₀ = f x₀ ∧ x = x₀ - 2 ∧ y = y₀ - 1 :=
sorry

end NUMINAMATH_CALUDE_graph_shift_l1334_133427


namespace NUMINAMATH_CALUDE_base7_to_base10_76543_l1334_133430

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base 7 number 76543 --/
def base7Number : List Nat := [3, 4, 5, 6, 7]

/-- Theorem: The base 10 equivalent of 76543 in base 7 is 19141 --/
theorem base7_to_base10_76543 :
  base7ToBase10 base7Number = 19141 := by
  sorry

end NUMINAMATH_CALUDE_base7_to_base10_76543_l1334_133430


namespace NUMINAMATH_CALUDE_cubic_one_real_root_l1334_133414

theorem cubic_one_real_root (c : ℝ) :
  ∃! x : ℝ, x^3 - 4*x^2 + 9*x + c = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_cubic_one_real_root_l1334_133414


namespace NUMINAMATH_CALUDE_function_maximum_l1334_133425

/-- The function f(x) = x + 4/x for x < 0 has a maximum value of -4 -/
theorem function_maximum (x : ℝ) (h : x < 0) : 
  x + 4 / x ≤ -4 :=
sorry

end NUMINAMATH_CALUDE_function_maximum_l1334_133425


namespace NUMINAMATH_CALUDE_cut_prism_faces_cut_prism_faces_proof_l1334_133451

/-- A triangular prism with 9 edges -/
structure TriangularPrism :=
  (edges : ℕ)
  (edges_eq : edges = 9)

/-- The result of cutting a triangular prism parallel to its base from the midpoints of its side edges -/
structure CutPrism extends TriangularPrism :=
  (additional_faces : ℕ)
  (additional_faces_eq : additional_faces = 3)

/-- The theorem stating that a cut triangular prism has 8 faces in total -/
theorem cut_prism_faces (cp : CutPrism) : ℕ :=
  8

#check cut_prism_faces

/-- Proof of the theorem -/
theorem cut_prism_faces_proof (cp : CutPrism) : cut_prism_faces cp = 8 := by
  sorry

end NUMINAMATH_CALUDE_cut_prism_faces_cut_prism_faces_proof_l1334_133451


namespace NUMINAMATH_CALUDE_sum_of_digits_of_square_l1334_133464

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a number is a ten-digit number -/
def isTenDigitNumber (n : ℕ) : Prop := sorry

theorem sum_of_digits_of_square (N : ℕ) :
  isTenDigitNumber N →
  sumOfDigits N = 4 →
  (sumOfDigits (N^2) = 7 ∨ sumOfDigits (N^2) = 16) := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_square_l1334_133464


namespace NUMINAMATH_CALUDE_symmetry_properties_l1334_133487

/-- A point in a 2D plane. -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the x-axis. -/
def symmetryOX (p : Point2D) : Point2D :=
  ⟨p.x, -p.y⟩

/-- Symmetry with respect to the y-axis. -/
def symmetryOY (p : Point2D) : Point2D :=
  ⟨-p.x, p.y⟩

theorem symmetry_properties (p : Point2D) : 
  (symmetryOX p = ⟨p.x, -p.y⟩) ∧ (symmetryOY p = ⟨-p.x, p.y⟩) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_properties_l1334_133487


namespace NUMINAMATH_CALUDE_walking_rate_l1334_133469

/-- Given a distance of 4 miles and a time of 1.25 hours, the rate of travel is 3.2 miles per hour -/
theorem walking_rate (distance : ℝ) (time : ℝ) (rate : ℝ) 
    (h1 : distance = 4)
    (h2 : time = 1.25)
    (h3 : rate = distance / time) : 
  rate = 3.2 := by
  sorry

end NUMINAMATH_CALUDE_walking_rate_l1334_133469


namespace NUMINAMATH_CALUDE_combined_average_marks_average_marks_two_classes_l1334_133416

/-- Given two classes with specified number of students and average marks,
    calculate the average mark of all students combined. -/
theorem combined_average_marks (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 > 0 → n2 > 0 →
  let total_students := n1 + n2
  let total_marks := n1 * avg1 + n2 * avg2
  total_marks / total_students = (n1 * avg1 + n2 * avg2) / (n1 + n2) :=
by sorry

/-- The average marks of all students from two classes. -/
theorem average_marks_two_classes :
  let class1_students : ℕ := 30
  let class2_students : ℕ := 50
  let class1_avg : ℚ := 40
  let class2_avg : ℚ := 70
  let total_students := class1_students + class2_students
  let total_marks := class1_students * class1_avg + class2_students * class2_avg
  total_marks / total_students = 58.75 :=
by sorry

end NUMINAMATH_CALUDE_combined_average_marks_average_marks_two_classes_l1334_133416


namespace NUMINAMATH_CALUDE_newspapers_collected_l1334_133462

/-- The number of newspapers collected by Chris and Lily -/
theorem newspapers_collected (chris_newspapers lily_newspapers : ℕ) 
  (h1 : chris_newspapers = 42)
  (h2 : lily_newspapers = 23) :
  chris_newspapers + lily_newspapers = 65 := by
  sorry

end NUMINAMATH_CALUDE_newspapers_collected_l1334_133462


namespace NUMINAMATH_CALUDE_kho_kho_only_count_l1334_133489

/-- The number of people who play kho kho only -/
def kho_kho_only : ℕ := 30

/-- The total number of players -/
def total_players : ℕ := 40

/-- The number of people who play kabadi -/
def kabadi_players : ℕ := 10

/-- The number of people who play both kabadi and kho kho -/
def both_players : ℕ := 5

/-- Theorem stating that the number of people who play kho kho only is 30 -/
theorem kho_kho_only_count :
  kho_kho_only = total_players - kabadi_players + both_players :=
by sorry

end NUMINAMATH_CALUDE_kho_kho_only_count_l1334_133489


namespace NUMINAMATH_CALUDE_equation_solution_l1334_133497

theorem equation_solution : 
  ∃ x : ℚ, (2 * x + 1) / 3 - (x - 1) / 6 = 2 ∧ x = 3 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1334_133497


namespace NUMINAMATH_CALUDE_linear_equation_power_l1334_133477

/-- If $2x^{n-3}-\frac{1}{3}y^{2m+1}=0$ is a linear equation in $x$ and $y$, then $n^m = 1$. -/
theorem linear_equation_power (n m : ℕ) :
  (∀ x y : ℝ, ∃ a b c : ℝ, 2 * x^(n-3) - (1/3) * y^(2*m+1) = a * x + b * y + c) →
  n^m = 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_power_l1334_133477


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l1334_133486

theorem inequality_and_equality_condition (x y : ℝ) 
  (hx : x > -1) (hy : y > -1) (hsum : x + y = 1) :
  (x / (y + 1) + y / (x + 1) ≥ 2 / 3) ∧
  (x / (y + 1) + y / (x + 1) = 2 / 3 ↔ x = 1 / 2 ∧ y = 1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l1334_133486


namespace NUMINAMATH_CALUDE_athlete_running_time_l1334_133448

/-- Represents the calories burned per minute while running -/
def running_rate : ℝ := 10

/-- Represents the calories burned per minute while walking -/
def walking_rate : ℝ := 4

/-- Represents the total calories burned -/
def total_calories : ℝ := 450

/-- Represents the total time spent exercising in minutes -/
def total_time : ℝ := 60

/-- Theorem stating that the athlete spends 35 minutes running -/
theorem athlete_running_time :
  ∃ (r w : ℝ),
    r + w = total_time ∧
    running_rate * r + walking_rate * w = total_calories ∧
    r = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_athlete_running_time_l1334_133448


namespace NUMINAMATH_CALUDE_parabola_equation_l1334_133408

/-- A parabola is defined by the equation y = ax^2 + bx + c where a, b, and c are real numbers and a ≠ 0 -/
def Parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- A parabola opens upwards if a > 0 -/
def OpensUpwards (a b c : ℝ) : Prop := a > 0

/-- A parabola intersects the y-axis at the point (0, y) if y = c -/
def IntersectsYAxisAt (a b c y : ℝ) : Prop := c = y

theorem parabola_equation : ∃ (a b : ℝ), 
  OpensUpwards a b 2 ∧ 
  IntersectsYAxisAt a b 2 2 ∧ 
  (∀ x, Parabola a b 2 x = x^2 + 2) := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1334_133408


namespace NUMINAMATH_CALUDE_smallest_n_for_125_l1334_133479

/-- The sequence term defined as a function of n -/
def a (n : ℕ) : ℤ := 2 * n^2 - 3

/-- The proposition that 8 is the smallest positive integer n for which a(n) = 125 -/
theorem smallest_n_for_125 : ∀ n : ℕ, n > 0 → a n = 125 → n ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_125_l1334_133479


namespace NUMINAMATH_CALUDE_rhombus_matches_l1334_133454

/-- Represents the number of matches needed for a rhombus -/
def matches_for_rhombus (s : ℕ) : ℕ := s * (s + 3)

/-- Theorem: The number of matches needed for a rhombus with side length s,
    divided into unit triangles, is s(s+3) -/
theorem rhombus_matches (s : ℕ) : 
  matches_for_rhombus s = s * (s + 3) := by
  sorry

#eval matches_for_rhombus 10  -- Should evaluate to 320

end NUMINAMATH_CALUDE_rhombus_matches_l1334_133454


namespace NUMINAMATH_CALUDE_no_pythagorean_triple_with_3_l1334_133461

theorem no_pythagorean_triple_with_3 :
  ¬∃ (a b c : ℤ), a^2 + b^2 = 3 * c^2 ∧ Int.gcd a (Int.gcd b c) = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_pythagorean_triple_with_3_l1334_133461


namespace NUMINAMATH_CALUDE_range_of_x_for_inequality_l1334_133458

-- Define the function f(x, a)
def f (x a : ℝ) : ℝ := x^2 + (a - 4) * x + 4 - 2 * a

-- State the theorem
theorem range_of_x_for_inequality (a : ℝ) (h : a ∈ Set.Icc (-1) 1) :
  {x : ℝ | ∀ a ∈ Set.Icc (-1) 1, f x a > 0} = Set.Iio 1 ∪ Set.Ioi 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_for_inequality_l1334_133458


namespace NUMINAMATH_CALUDE_three_digit_geometric_progression_exists_l1334_133410

/-- Represents a three-digit number as a tuple of its digits -/
def ThreeDigitNumber := (Nat × Nat × Nat)

/-- Checks if three numbers form a geometric progression -/
def is_geometric_progression (a b c : Nat) : Prop :=
  b * b = a * c

/-- Converts a ThreeDigitNumber to its decimal representation -/
def to_decimal (n : ThreeDigitNumber) : Nat :=
  100 * n.1 + 10 * n.2.1 + n.2.2

/-- The main theorem statement -/
theorem three_digit_geometric_progression_exists : ∃! (n : ThreeDigitNumber),
  (is_geometric_progression n.1 n.2.1 n.2.2) ∧
  (to_decimal (n.2.2, n.2.1, n.1) = to_decimal n - 594) ∧
  (10 * n.2.2 + n.2.1 = 10 * n.2.1 + n.2.2 - 18) ∧
  (to_decimal n = 842) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_geometric_progression_exists_l1334_133410


namespace NUMINAMATH_CALUDE_joan_remaining_apples_l1334_133496

/-- Given that Joan picked 43 apples and gave 27 to Melanie, prove that she now has 16 apples. -/
theorem joan_remaining_apples : 
  ∀ (initial_apples given_apples remaining_apples : ℕ), 
    initial_apples = 43 → 
    given_apples = 27 → 
    remaining_apples = initial_apples - given_apples → 
    remaining_apples = 16 := by
  sorry

end NUMINAMATH_CALUDE_joan_remaining_apples_l1334_133496


namespace NUMINAMATH_CALUDE_f_odd_and_decreasing_l1334_133411

-- Define the function f(x) = -x|x|
def f (x : ℝ) : ℝ := -x * abs x

-- Theorem stating that f is both odd and decreasing
theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f y < f x) :=
sorry

end NUMINAMATH_CALUDE_f_odd_and_decreasing_l1334_133411


namespace NUMINAMATH_CALUDE_quadratic_inequality_boundary_l1334_133478

theorem quadratic_inequality_boundary (c : ℝ) : 
  (∀ x : ℝ, x * (3 * x + 1) < c ↔ -5/2 < x ∧ x < 3) ↔ c = 30 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_boundary_l1334_133478


namespace NUMINAMATH_CALUDE_product_simplification_l1334_133428

theorem product_simplification (x : ℝ) (hx : x ≠ 0) :
  (10 * x^3) * (8 * x^2) * (1 / (4*x)^3) = (5/4) * x^2 := by
  sorry

end NUMINAMATH_CALUDE_product_simplification_l1334_133428


namespace NUMINAMATH_CALUDE_dice_probability_l1334_133475

theorem dice_probability : 
  let n : ℕ := 8  -- total number of dice
  let k : ℕ := 4  -- number of dice showing even
  let p : ℚ := 1/2  -- probability of rolling even (or odd) on a single die
  Nat.choose n k * p^n = 35/128 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l1334_133475


namespace NUMINAMATH_CALUDE_trajectory_and_fixed_point_l1334_133404

-- Define the moving circle M
structure MovingCircle where
  center : ℝ × ℝ
  passes_through : center.1 - 1 ^ 2 + center.2 ^ 2 = (center.1 + 1) ^ 2
  tangent_to_line : center.1 + 1 = ((center.1 - 1) ^ 2 + center.2 ^ 2).sqrt

-- Define the trajectory C
def trajectory (x y : ℝ) : Prop := y ^ 2 = 4 * x

-- Define a point on the trajectory
structure PointOnTrajectory where
  point : ℝ × ℝ
  on_trajectory : trajectory point.1 point.2
  not_origin : point ≠ (0, 0)

-- Theorem statement
theorem trajectory_and_fixed_point 
  (M : MovingCircle) 
  (A B : PointOnTrajectory) 
  (h : A.point.1 * B.point.1 + A.point.2 * B.point.2 = 0) :
  ∃ (t : ℝ), 
    t * A.point.2 + (1 - t) * B.point.2 = 0 ∧ 
    t * A.point.1 + (1 - t) * B.point.1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_and_fixed_point_l1334_133404


namespace NUMINAMATH_CALUDE_billy_soda_theorem_l1334_133435

def billy_soda_distribution (num_sisters : ℕ) (soda_pack : ℕ) : Prop :=
  let num_brothers := 2 * num_sisters
  let total_siblings := num_brothers + num_sisters
  let sodas_per_sibling := soda_pack / total_siblings
  (num_sisters = 2) ∧ (soda_pack = 12) → (sodas_per_sibling = 2)

theorem billy_soda_theorem : billy_soda_distribution 2 12 := by
  sorry

end NUMINAMATH_CALUDE_billy_soda_theorem_l1334_133435


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1334_133470

/-- An increasing geometric sequence -/
def IsIncreasingGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 1 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geom : IsIncreasingGeometricSequence a)
  (h_sum : a 1 + a 4 = 9)
  (h_prod : a 2 * a 3 = 8) :
  ∃ q : ℝ, q = 2 ∧ ∀ n, a (n + 1) = a n * q :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1334_133470


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1334_133426

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  increasing : ∀ n, a n < a (n + 1)
  arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  first_term : a 1 = 2
  geometric : (a 2)^2 = a 1 * a 5

/-- The common difference of the arithmetic sequence is 4 -/
theorem arithmetic_sequence_common_difference (seq : ArithmeticSequence) :
  ∃ d, (∀ n, seq.a (n + 1) - seq.a n = d) ∧ d = 4 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1334_133426


namespace NUMINAMATH_CALUDE_f_expression_l1334_133422

/-- A linear function f satisfying specific conditions -/
def f (x : ℝ) : ℝ := sorry

/-- f is a linear function -/
axiom f_linear : ∃ (a b : ℝ), ∀ x, f x = a * x + b

/-- f(-2) = -1 -/
axiom f_neg_two : f (-2) = -1

/-- f(0) + f(2) = 10 -/
axiom f_sum : f 0 + f 2 = 10

/-- Theorem: f(x) = 2x + 3 -/
theorem f_expression : ∀ x, f x = 2 * x + 3 := by sorry

end NUMINAMATH_CALUDE_f_expression_l1334_133422


namespace NUMINAMATH_CALUDE_spider_return_probability_l1334_133452

/-- Probability of the spider being at the starting corner after n moves -/
def P : ℕ → ℚ
| 0 => 1
| n + 1 => (1 - P n) / 3

/-- The probability of returning to the starting corner on the eighth move -/
theorem spider_return_probability : P 8 = 547 / 2187 := by
  sorry

end NUMINAMATH_CALUDE_spider_return_probability_l1334_133452


namespace NUMINAMATH_CALUDE_modulus_equality_necessary_not_sufficient_l1334_133420

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem modulus_equality_necessary_not_sufficient :
  (∀ (u v : E), u ≠ 0 ∧ v ≠ 0 → (‖u‖ = ‖v‖ → u = v) ↔ false) ∧
  (∀ (u v : E), u ≠ 0 ∧ v ≠ 0 → (u = v → ‖u‖ = ‖v‖)) :=
by sorry

end NUMINAMATH_CALUDE_modulus_equality_necessary_not_sufficient_l1334_133420


namespace NUMINAMATH_CALUDE_n_value_equality_l1334_133439

theorem n_value_equality (n : ℕ) : 3 * (Nat.choose (n - 3) (n - 7)) = 5 * (Nat.factorial (n - 4) / Nat.factorial (n - 6)) → n = 11 := by
  sorry

end NUMINAMATH_CALUDE_n_value_equality_l1334_133439


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_divisibility_l1334_133446

/-- Given three numbers in an arithmetic sequence with common difference d,
    where one of the numbers is divisible by d, their product is divisible by 6d³ -/
theorem arithmetic_sequence_product_divisibility
  (a b c d : ℤ) -- a, b, c are the three numbers, d is the common difference
  (h_arithmetic : b - a = d ∧ c - b = d) -- arithmetic sequence condition
  (h_divisible : a % d = 0 ∨ b % d = 0 ∨ c % d = 0) -- one number divisible by d
  : (6 * d^3) ∣ (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_divisibility_l1334_133446


namespace NUMINAMATH_CALUDE_triangle_area_ratio_l1334_133445

theorem triangle_area_ratio (K J : ℝ) (x : ℝ) (h_positive : 0 < x) (h_less_than_one : x < 1)
  (h_ratio : J / K = x) : x = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_ratio_l1334_133445


namespace NUMINAMATH_CALUDE_ratio_a_to_b_l1334_133415

/-- A geometric sequence with first four terms a, x, b, 2x -/
structure GeometricSequence (α : Type*) [Field α] where
  a : α
  x : α
  b : α

/-- The ratio between consecutive terms in a geometric sequence is constant -/
def is_geometric_sequence {α : Type*} [Field α] (seq : GeometricSequence α) : Prop :=
  seq.x / seq.a = seq.b / seq.x ∧ seq.b / seq.x = 2

theorem ratio_a_to_b {α : Type*} [Field α] (seq : GeometricSequence α) 
  (h : is_geometric_sequence seq) : seq.a / seq.b = 1 / 4 := by
  sorry

#check ratio_a_to_b

end NUMINAMATH_CALUDE_ratio_a_to_b_l1334_133415


namespace NUMINAMATH_CALUDE_max_value_of_complex_expression_l1334_133499

theorem max_value_of_complex_expression (a b : ℝ) : 
  let z : ℂ := Complex.mk a b
  (Complex.abs z = 2) →
  (∃ (m : ℝ), m = 9 ∧ ∀ (x y : ℝ), 
    let w : ℂ := Complex.mk x y
    Complex.abs w = 2 → 
    Complex.abs ((w - 1) * (w + 1)^2) ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_complex_expression_l1334_133499


namespace NUMINAMATH_CALUDE_circleplus_two_three_l1334_133421

-- Define the operation ⊕
def circleplus (a b : ℝ) : ℝ := 4 * a + 5 * b

-- State the theorem
theorem circleplus_two_three : circleplus 2 3 = 23 := by
  sorry

end NUMINAMATH_CALUDE_circleplus_two_three_l1334_133421


namespace NUMINAMATH_CALUDE_sum_of_squares_quadratic_roots_sum_of_squares_specific_quadratic_l1334_133443

theorem sum_of_squares_quadratic_roots (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * x^2 + b * x + c = 0 → r₁^2 + r₂^2 = (b^2 - 2*a*c) / a^2 :=
by sorry

theorem sum_of_squares_specific_quadratic :
  let r₁ := (10 + Real.sqrt 36) / 2
  let r₂ := (10 - Real.sqrt 36) / 2
  r₁^2 + r₂^2 = 68 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_quadratic_roots_sum_of_squares_specific_quadratic_l1334_133443


namespace NUMINAMATH_CALUDE_new_student_weight_l1334_133432

theorem new_student_weight (initial_count : ℕ) (replaced_weight : ℝ) (avg_decrease : ℝ) :
  initial_count = 4 →
  replaced_weight = 96 →
  avg_decrease = 8 →
  ∃ (new_weight : ℝ),
    new_weight = initial_count * avg_decrease + replaced_weight ∧
    new_weight = 160 :=
by sorry

end NUMINAMATH_CALUDE_new_student_weight_l1334_133432


namespace NUMINAMATH_CALUDE_downstream_distance_man_downstream_distance_l1334_133417

/-- Calculates the downstream distance given swimming conditions --/
theorem downstream_distance (time : ℝ) (upstream_distance : ℝ) (still_speed : ℝ) : ℝ :=
  let stream_speed := still_speed - (upstream_distance / time)
  let downstream_speed := still_speed + stream_speed
  downstream_speed * time

/-- Proves that the downstream distance is 45 km given the specific conditions --/
theorem man_downstream_distance : 
  downstream_distance 5 25 7 = 45 := by
  sorry

end NUMINAMATH_CALUDE_downstream_distance_man_downstream_distance_l1334_133417


namespace NUMINAMATH_CALUDE_money_distribution_l1334_133449

theorem money_distribution (a b c total : ℕ) : 
  a + b + c = 9 →
  b = 3 →
  1200 * 3 = total →
  total = 3600 := by sorry

end NUMINAMATH_CALUDE_money_distribution_l1334_133449


namespace NUMINAMATH_CALUDE_ronas_age_l1334_133418

theorem ronas_age (rona rachel collete : ℕ) 
  (h1 : rachel = 2 * rona)
  (h2 : collete = rona / 2)
  (h3 : rachel - collete = 12) : 
  rona = 12 := by
sorry

end NUMINAMATH_CALUDE_ronas_age_l1334_133418


namespace NUMINAMATH_CALUDE_square_of_98_l1334_133456

theorem square_of_98 : (98 : ℕ) ^ 2 = 9604 := by sorry

end NUMINAMATH_CALUDE_square_of_98_l1334_133456


namespace NUMINAMATH_CALUDE_jinas_teddies_l1334_133476

/-- Proves that the initial number of teddies is 5 given the conditions in Jina's mascot collection problem -/
theorem jinas_teddies :
  ∀ (initial_teddies : ℕ),
  let bunnies := 3 * initial_teddies
  let additional_teddies := 2 * bunnies
  let total_mascots := initial_teddies + bunnies + additional_teddies + 1
  total_mascots = 51 →
  initial_teddies = 5 := by
sorry

end NUMINAMATH_CALUDE_jinas_teddies_l1334_133476


namespace NUMINAMATH_CALUDE_trigonometric_product_equals_one_l1334_133402

theorem trigonometric_product_equals_one :
  let x : Real := 40 * π / 180
  let y : Real := 50 * π / 180
  (1 - 1 / Real.cos x) * (1 + 1 / Real.sin y) * (1 - 1 / Real.sin x) * (1 + 1 / Real.cos y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_product_equals_one_l1334_133402


namespace NUMINAMATH_CALUDE_no_intersection_and_in_circle_l1334_133440

theorem no_intersection_and_in_circle : ¬∃ (a b : ℝ), 
  (∃ (n : ℤ), ∃ (m : ℤ), n = m ∧ a * n + b = 3 * m^2 + 15) ∧ 
  (a^2 + b^2 ≤ 144) := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_and_in_circle_l1334_133440


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l1334_133480

-- Define a line in slope-intercept form (y = mx + b)
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

def Line.equation (l : Line) (x y : ℝ) : Prop :=
  y = l.m * x + l.b

-- Define parallel lines
def parallel (l1 l2 : Line) : Prop :=
  l1.m = l2.m

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define a line passing through a point
def passes_through (l : Line) (p : Point) : Prop :=
  l.equation p.x p.y

-- The given line
def given_line : Line :=
  { m := -2, b := 3 }

-- The point (0, 1)
def point : Point :=
  { x := 0, y := 1 }

-- The theorem to prove
theorem parallel_line_through_point :
  ∃! l : Line, parallel l given_line ∧ passes_through l point ∧ l.equation 0 1 :=
sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l1334_133480


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1334_133457

theorem arithmetic_sequence_common_difference 
  (n : ℕ) 
  (total_sum : ℝ) 
  (even_sum : ℝ) 
  (h1 : n = 20) 
  (h2 : total_sum = 75) 
  (h3 : even_sum = 25) : 
  (even_sum - (total_sum - even_sum)) / 10 = -2.5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1334_133457


namespace NUMINAMATH_CALUDE_number_difference_l1334_133400

theorem number_difference (L S : ℕ) (h1 : L = 1584) (h2 : L = 6 * S + 15) : L - S = 1323 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l1334_133400


namespace NUMINAMATH_CALUDE_equal_share_theorem_l1334_133490

/-- Represents the number of candies each person has initially -/
structure CandyDistribution :=
  (mark : ℕ)
  (peter : ℕ)
  (john : ℕ)

/-- Calculates the number of candies each person gets after sharing equally -/
def share_candies (dist : CandyDistribution) : ℕ :=
  (dist.mark + dist.peter + dist.john) / 3

/-- Theorem: Given the initial candy distribution, prove that each person gets 30 candies after sharing -/
theorem equal_share_theorem (dist : CandyDistribution) 
  (h1 : dist.mark = 30)
  (h2 : dist.peter = 25)
  (h3 : dist.john = 35) :
  share_candies dist = 30 := by
  sorry

end NUMINAMATH_CALUDE_equal_share_theorem_l1334_133490


namespace NUMINAMATH_CALUDE_alicia_art_collection_l1334_133434

/-- The number of medieval art pieces Alicia donated -/
def donated : ℕ := 46

/-- The number of medieval art pieces Alicia had left after donating -/
def left_after_donating : ℕ := 24

/-- The original number of medieval art pieces Alicia had -/
def original_pieces : ℕ := donated + left_after_donating

theorem alicia_art_collection : original_pieces = 70 := by
  sorry

end NUMINAMATH_CALUDE_alicia_art_collection_l1334_133434


namespace NUMINAMATH_CALUDE_farm_cows_l1334_133485

/-- Represents the number of bags of husk eaten by some cows in 45 days -/
def total_bags : ℕ := 45

/-- Represents the number of bags of husk eaten by one cow in 45 days -/
def bags_per_cow : ℕ := 1

/-- Calculates the number of cows on the farm -/
def num_cows : ℕ := total_bags / bags_per_cow

/-- Proves that the number of cows on the farm is 45 -/
theorem farm_cows : num_cows = 45 := by
  sorry

end NUMINAMATH_CALUDE_farm_cows_l1334_133485


namespace NUMINAMATH_CALUDE_circle_equation_solution_l1334_133482

theorem circle_equation_solution :
  ∃! (x y : ℝ), (x - 11)^2 + (y - 12)^2 + (x - y)^2 = 1/3 ∧ x = 11 + 1/3 ∧ y = 11 + 2/3 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_solution_l1334_133482


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l1334_133433

theorem shaded_area_calculation (R : ℝ) (r : ℝ) (h1 : R = 9) (h2 : r = R / 4) :
  π * R^2 - 2 * (π * r^2) = 70.875 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l1334_133433


namespace NUMINAMATH_CALUDE_divisible_by_eight_probability_l1334_133498

theorem divisible_by_eight_probability (n : ℕ) : 
  (Finset.filter (λ k => (k * (k + 1)) % 8 = 0) (Finset.range 100)).card / 100 = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_divisible_by_eight_probability_l1334_133498


namespace NUMINAMATH_CALUDE_power_of_seven_roots_l1334_133471

theorem power_of_seven_roots (x : ℝ) (h : x > 0) :
  (x^(1/4)) / (x^(1/7)) = x^(3/28) := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_roots_l1334_133471


namespace NUMINAMATH_CALUDE_beads_left_in_container_l1334_133463

/-- The number of beads left in a container after some are removed -/
theorem beads_left_in_container (green brown red removed : ℕ) : 
  green = 1 → brown = 2 → red = 3 → removed = 2 →
  green + brown + red - removed = 4 := by
  sorry

end NUMINAMATH_CALUDE_beads_left_in_container_l1334_133463


namespace NUMINAMATH_CALUDE_train_crossing_tree_time_l1334_133436

/-- Given a train and a platform with specified lengths and the time it takes for the train to pass the platform, 
    calculate the time it takes for the train to cross a tree. -/
theorem train_crossing_tree_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (time_to_pass_platform : ℝ) 
  (h1 : train_length = 1200) 
  (h2 : platform_length = 300) 
  (h3 : time_to_pass_platform = 150) : 
  (train_length / ((train_length + platform_length) / time_to_pass_platform)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_tree_time_l1334_133436


namespace NUMINAMATH_CALUDE_cos_pi_sixth_plus_alpha_l1334_133438

theorem cos_pi_sixth_plus_alpha (α : ℝ) (h : Real.sin (π / 3 - α) = 1 / 6) :
  Real.cos (π / 6 + α) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_sixth_plus_alpha_l1334_133438


namespace NUMINAMATH_CALUDE_stamp_problem_solution_l1334_133493

def stamp_problem (aj kj cj : ℕ) (m : ℚ) : Prop :=
  aj = 370 ∧
  kj = aj / 2 ∧
  aj + kj + cj = 930 ∧
  cj = m * kj + 5

theorem stamp_problem_solution :
  ∃ (aj kj cj : ℕ) (m : ℚ), stamp_problem aj kj cj m ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_stamp_problem_solution_l1334_133493


namespace NUMINAMATH_CALUDE_rotation_center_l1334_133424

theorem rotation_center (f : ℂ → ℂ) (c : ℂ) : 
  (f = fun z ↦ ((1 - Complex.I * Real.sqrt 2) * z + (-4 * Real.sqrt 2 + 6 * Complex.I)) / 2) →
  (c = (2 * Real.sqrt 2) / 3 - (2 * Complex.I) / 3) →
  f c = c := by
sorry

end NUMINAMATH_CALUDE_rotation_center_l1334_133424


namespace NUMINAMATH_CALUDE_amy_chocolate_bars_l1334_133447

/-- The number of chocolate bars Amy has -/
def chocolate_bars : ℕ := sorry

/-- The number of M&Ms Amy has -/
def m_and_ms : ℕ := 7 * chocolate_bars

/-- The number of marshmallows Amy has -/
def marshmallows : ℕ := 6 * m_and_ms

/-- The total number of candies Amy has -/
def total_candies : ℕ := chocolate_bars + m_and_ms + marshmallows

/-- The number of baskets Amy fills -/
def num_baskets : ℕ := 25

/-- The number of candies in each basket -/
def candies_per_basket : ℕ := 10

theorem amy_chocolate_bars : 
  chocolate_bars = 5 ∧ 
  total_candies = num_baskets * candies_per_basket := by
  sorry

end NUMINAMATH_CALUDE_amy_chocolate_bars_l1334_133447


namespace NUMINAMATH_CALUDE_sphere_surface_area_l1334_133472

theorem sphere_surface_area (V : ℝ) (r : ℝ) (S : ℝ) :
  V = 48 * Real.pi →
  V = (4 / 3) * Real.pi * r^3 →
  S = 4 * Real.pi * r^2 →
  S = 144 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l1334_133472


namespace NUMINAMATH_CALUDE_triangle_side_length_l1334_133429

theorem triangle_side_length 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h_area : (1/2) * a * c * Real.sin B = Real.sqrt 3)
  (h_angle : B = Real.pi / 3)
  (h_sides : a^2 + c^2 = 3*a*c) :
  b = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1334_133429


namespace NUMINAMATH_CALUDE_remainder_1632_times_2024_div_400_l1334_133405

theorem remainder_1632_times_2024_div_400 : (1632 * 2024) % 400 = 368 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1632_times_2024_div_400_l1334_133405


namespace NUMINAMATH_CALUDE_exists_area_preserving_projection_l1334_133492

-- Define the concept of a plane
def Plane : Type := sorry

-- Define the concept of a triangle
structure Triangle (P : Plane) :=
  (area : ℝ)

-- Define the concept of parallel projection
def parallel_projection (P Q : Plane) (T : Triangle P) : Triangle Q := sorry

-- Theorem statement
theorem exists_area_preserving_projection
  (P Q : Plane)
  (intersect : P ≠ Q)
  (T : Triangle P) :
  ∃ (proj : Triangle Q), proj = parallel_projection P Q T ∧ proj.area = T.area :=
sorry

end NUMINAMATH_CALUDE_exists_area_preserving_projection_l1334_133492


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1334_133495

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 0 ∧ ∀ n, a n > 0 ∧ a (n + 1) = a n * q

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom : GeometricSequence a q)
  (h_sum : a 2 + a 4 = 3)
  (h_prod : a 3 * a 5 = 2) :
  q = Real.sqrt ((3 * Real.sqrt 2 + 2) / 7) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1334_133495


namespace NUMINAMATH_CALUDE_octal_addition_l1334_133406

/-- Addition of octal numbers -/
def octal_add (a b c : ℕ) : ℕ :=
  (a * 8^2 + (a / 8) * 8 + (a % 8)) +
  (b * 8^2 + (b / 8) * 8 + (b % 8)) +
  (c * 8^2 + (c / 8) * 8 + (c % 8))

/-- Conversion from decimal to octal -/
def to_octal (n : ℕ) : ℕ :=
  (n / 8^2) * 100 + ((n / 8) % 8) * 10 + (n % 8)

theorem octal_addition :
  to_octal (octal_add 176 725 63) = 1066 := by
  sorry

end NUMINAMATH_CALUDE_octal_addition_l1334_133406


namespace NUMINAMATH_CALUDE_first_equation_value_l1334_133488

theorem first_equation_value (x y a : ℝ) 
  (eq1 : 2 * x + y = a) 
  (eq2 : x + 2 * y = 10) 
  (eq3 : (x + y) / 3 = 4) : 
  a = 12 := by
sorry

end NUMINAMATH_CALUDE_first_equation_value_l1334_133488


namespace NUMINAMATH_CALUDE_population_increase_theorem_l1334_133460

/-- Given birth and death rates per 1000 people, calculate the percentage increase in population rate -/
def population_increase_percentage (birth_rate death_rate : ℚ) : ℚ :=
  (birth_rate - death_rate) * 100 / 1000

theorem population_increase_theorem (birth_rate death_rate : ℚ) 
  (h1 : birth_rate = 32)
  (h2 : death_rate = 11) : 
  population_increase_percentage birth_rate death_rate = (21 : ℚ) / 10 :=
by sorry

end NUMINAMATH_CALUDE_population_increase_theorem_l1334_133460


namespace NUMINAMATH_CALUDE_minimize_m_l1334_133483

theorem minimize_m (x y : ℝ) :
  let m := 4 * x^2 - 12 * x * y + 10 * y^2 + 4 * y + 9
  ∀ a b : ℝ, m ≤ (4 * a^2 - 12 * a * b + 10 * b^2 + 4 * b + 9) ∧
  m = 5 ∧ x = -3 ∧ y = -2 := by
  sorry

end NUMINAMATH_CALUDE_minimize_m_l1334_133483


namespace NUMINAMATH_CALUDE_kannon_apples_last_night_l1334_133453

/-- The number of apples Kannon had last night -/
def apples_last_night : ℕ := sorry

/-- The number of bananas Kannon had last night -/
def bananas_last_night : ℕ := 1

/-- The number of oranges Kannon had last night -/
def oranges_last_night : ℕ := 4

/-- The number of apples Kannon will have today -/
def apples_today : ℕ := apples_last_night + 4

/-- The number of bananas Kannon will have today -/
def bananas_today : ℕ := 10 * bananas_last_night

/-- The number of oranges Kannon will have today -/
def oranges_today : ℕ := 2 * apples_today

theorem kannon_apples_last_night :
  apples_last_night = 3 ∧
  (apples_last_night + bananas_last_night + oranges_last_night +
   apples_today + bananas_today + oranges_today = 39) :=
by sorry

end NUMINAMATH_CALUDE_kannon_apples_last_night_l1334_133453


namespace NUMINAMATH_CALUDE_f_negative_range_x_range_for_negative_f_l1334_133442

/-- The function f(x) = mx^2 - mx - 6 + m -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - m * x - 6 + m

theorem f_negative_range (m : ℝ) (x : ℝ) :
  (∀ x ∈ Set.Icc 1 3, f m x < 0) ↔ m < 6/7 :=
sorry

theorem x_range_for_negative_f (m : ℝ) (x : ℝ) :
  (∀ m ∈ Set.Icc (-2) 2, f m x < 0) ↔ -1 < x ∧ x < 2 :=
sorry

end NUMINAMATH_CALUDE_f_negative_range_x_range_for_negative_f_l1334_133442


namespace NUMINAMATH_CALUDE_two_small_triangles_exist_l1334_133437

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Calculate the area of a triangle -/
def area (t : Triangle) : ℝ :=
  sorry

/-- Check if a point is inside a triangle -/
def isInside (p : Point) (t : Triangle) : Prop :=
  sorry

/-- The unit triangle -/
def unitTriangle : Triangle :=
  sorry

/-- Theorem: Given 5 points in a unit triangle, there exist at least two distinct
    triangles formed by these points, each with an area not exceeding 1/4 -/
theorem two_small_triangles_exist (points : Finset Point)
    (h1 : points.card = 5)
    (h2 : ∀ p ∈ points, isInside p unitTriangle) :
    ∃ t1 t2 : Triangle,
      t1.a ∈ points ∧ t1.b ∈ points ∧ t1.c ∈ points ∧
      t2.a ∈ points ∧ t2.b ∈ points ∧ t2.c ∈ points ∧
      t1 ≠ t2 ∧
      area t1 ≤ 1/4 ∧ area t2 ≤ 1/4 :=
  sorry

end NUMINAMATH_CALUDE_two_small_triangles_exist_l1334_133437


namespace NUMINAMATH_CALUDE_min_d_value_l1334_133466

theorem min_d_value (a b c d : ℕ+) (h_order : a < b ∧ b < c ∧ c < d) 
  (h_unique : ∃! (x y : ℝ), x + 2*y = 2023 ∧ y = |x - a| + |x - b| + |x - c| + |x - d|) :
  d ≥ 1010 ∧ ∃ (a' b' c' : ℕ+), a' < b' ∧ b' < c' ∧ c' < 1010 ∧
    ∃! (x y : ℝ), x + 2*y = 2023 ∧ y = |x - a'| + |x - b'| + |x - c'| + |x - 1010| :=
by sorry

end NUMINAMATH_CALUDE_min_d_value_l1334_133466


namespace NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l1334_133473

-- Define the conditions P and Q
def P (x : ℝ) : Prop := |2*x - 3| < 1
def Q (x : ℝ) : Prop := x*(x - 3) < 0

-- Theorem stating that P is sufficient but not necessary for Q
theorem P_sufficient_not_necessary_for_Q :
  (∀ x : ℝ, P x → Q x) ∧ 
  (∃ x : ℝ, Q x ∧ ¬(P x)) :=
sorry

end NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l1334_133473


namespace NUMINAMATH_CALUDE_crate_height_determination_l1334_133450

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a cylindrical gas tank -/
structure GasTank where
  radius : ℝ
  height : ℝ

/-- Checks if a gas tank fits inside a crate -/
def tankFitsInCrate (tank : GasTank) (crate : CrateDimensions) : Prop :=
  2 * tank.radius ≤ min crate.length (min crate.width crate.height) ∧
  tank.height ≤ max crate.length (max crate.width crate.height)

theorem crate_height_determination
  (crate : CrateDimensions)
  (tank : GasTank)
  (h_crate_dims : crate.length = 6 ∧ crate.width = 8)
  (h_tank_radius : tank.radius = 4)
  (h_tank_fits : tankFitsInCrate tank crate)
  (h_max_volume : ∀ (other_tank : GasTank),
    tankFitsInCrate other_tank crate →
    tank.radius * tank.radius * tank.height ≥ other_tank.radius * other_tank.radius * other_tank.height) :
  crate.height = 6 :=
sorry

end NUMINAMATH_CALUDE_crate_height_determination_l1334_133450


namespace NUMINAMATH_CALUDE_prime_neighbor_divisible_by_six_l1334_133459

theorem prime_neighbor_divisible_by_six (p : ℕ) (hp : Prime p) (hp_gt_3 : p > 3) :
  6 ∣ (p - 1) ∨ 6 ∣ (p + 1) := by
  sorry

#check prime_neighbor_divisible_by_six

end NUMINAMATH_CALUDE_prime_neighbor_divisible_by_six_l1334_133459
