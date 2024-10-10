import Mathlib

namespace all_gp_lines_through_origin_l1066_106633

/-- A line in the 2D plane represented by the equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if three real numbers form a geometric progression -/
def isGeometricProgression (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = a * r^2

/-- The point (0, 0) in the 2D plane -/
def origin : ℝ × ℝ := (0, 0)

/-- Checks if a point lies on a line -/
def pointOnLine (p : ℝ × ℝ) (l : Line) : Prop :=
  l.a * p.1 + l.b * p.2 = l.c

theorem all_gp_lines_through_origin :
  ∀ l : Line, isGeometricProgression l.a l.b l.c → pointOnLine origin l :=
sorry

end all_gp_lines_through_origin_l1066_106633


namespace quadratic_inequality_l1066_106619

-- Define the quadratic function
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- Define the solution set condition
def solution_set (b c : ℝ) : Set ℝ := {x | x > 2 ∨ x < 1}

-- Theorem statement
theorem quadratic_inequality (b c : ℝ) :
  (∀ x, x ∈ solution_set b c ↔ f b c x > 0) →
  (b = -3 ∧ c = 2) ∧
  (∀ x, x ∈ {x | 1/2 ≤ x ∧ x ≤ 1} ↔ 2*x^2 - 3*x + 1 ≤ 0) :=
sorry

end quadratic_inequality_l1066_106619


namespace negation_of_existence_is_forall_not_l1066_106659

theorem negation_of_existence_is_forall_not :
  (¬ ∃ x : ℝ, x^2 + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 + 1 > 0) := by
  sorry

end negation_of_existence_is_forall_not_l1066_106659


namespace monotonicity_of_f_l1066_106661

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x / (x - 1)

theorem monotonicity_of_f (a : ℝ) (h_a : a ≠ 0) :
  (∀ x₁ x₂ : ℝ, -1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → 
    (a > 0 → f a x₁ > f a x₂) ∧ 
    (a < 0 → f a x₁ < f a x₂)) :=
by sorry

end monotonicity_of_f_l1066_106661


namespace smallest_n_same_factors_l1066_106656

/-- Count the number of factors of a natural number -/
def countFactors (n : ℕ) : ℕ := sorry

/-- Check if three consecutive numbers have the same number of factors -/
def sameFactorCount (n : ℕ) : Prop :=
  countFactors n = countFactors (n + 1) ∧ countFactors n = countFactors (n + 2)

/-- 33 is the smallest natural number n such that n, n+1, and n+2 have the same number of factors -/
theorem smallest_n_same_factors : 
  (∀ m : ℕ, m < 33 → ¬(sameFactorCount m)) ∧ sameFactorCount 33 := by sorry

end smallest_n_same_factors_l1066_106656


namespace lulu_poptarts_count_l1066_106600

/-- Represents the number of pastries baked by Lola and Lulu -/
structure PastryCounts where
  lola_cupcakes : ℕ
  lola_poptarts : ℕ
  lola_pies : ℕ
  lulu_cupcakes : ℕ
  lulu_poptarts : ℕ
  lulu_pies : ℕ

/-- The total number of pastries baked by Lola and Lulu -/
def total_pastries (counts : PastryCounts) : ℕ :=
  counts.lola_cupcakes + counts.lola_poptarts + counts.lola_pies +
  counts.lulu_cupcakes + counts.lulu_poptarts + counts.lulu_pies

/-- Theorem stating that Lulu baked 12 pop tarts -/
theorem lulu_poptarts_count (counts : PastryCounts) 
  (h1 : counts.lola_cupcakes = 13)
  (h2 : counts.lola_poptarts = 10)
  (h3 : counts.lola_pies = 8)
  (h4 : counts.lulu_cupcakes = 16)
  (h5 : counts.lulu_pies = 14)
  (h6 : total_pastries counts = 73) :
  counts.lulu_poptarts = 12 := by
  sorry

end lulu_poptarts_count_l1066_106600


namespace smallest_sum_mn_l1066_106642

theorem smallest_sum_mn (m n : ℕ) (h : 3 * n^3 = 5 * m^2) : 
  ∃ (m' n' : ℕ), 3 * n'^3 = 5 * m'^2 ∧ m' + n' ≤ m + n ∧ m' + n' = 60 :=
sorry

end smallest_sum_mn_l1066_106642


namespace book_pages_book_has_120_pages_l1066_106679

theorem book_pages : ℕ → Prop :=
  fun total_pages =>
    let pages_yesterday : ℕ := 12
    let pages_today : ℕ := 2 * pages_yesterday
    let pages_read : ℕ := pages_yesterday + pages_today
    let pages_tomorrow : ℕ := 42
    let remaining_pages : ℕ := 2 * pages_tomorrow
    total_pages = pages_read + remaining_pages ∧ total_pages = 120

-- The proof of the theorem
theorem book_has_120_pages : ∃ (n : ℕ), book_pages n := by
  sorry

end book_pages_book_has_120_pages_l1066_106679


namespace greatest_third_term_arithmetic_sequence_l1066_106683

theorem greatest_third_term_arithmetic_sequence :
  ∀ (a d : ℕ+), 
  (a : ℕ) + (a + d : ℕ) + (a + 2 * d : ℕ) + (a + 3 * d : ℕ) = 58 →
  ∀ (b e : ℕ+),
  (b : ℕ) + (b + e : ℕ) + (b + 2 * e : ℕ) + (b + 3 * e : ℕ) = 58 →
  (a + 2 * d : ℕ) ≤ 19 :=
by sorry

end greatest_third_term_arithmetic_sequence_l1066_106683


namespace probability_is_four_twentysevenths_l1066_106687

/-- A regular tetrahedron with painted stripes -/
structure StripedTetrahedron where
  /-- The number of faces in a tetrahedron -/
  num_faces : Nat
  /-- The number of possible stripe configurations per face -/
  stripes_per_face : Nat
  /-- The total number of possible stripe configurations -/
  total_configurations : Nat
  /-- The number of configurations that form a continuous stripe -/
  continuous_configurations : Nat

/-- The probability of a continuous stripe encircling the tetrahedron -/
def probability_continuous_stripe (t : StripedTetrahedron) : Rat :=
  t.continuous_configurations / t.total_configurations

/-- Theorem stating the probability of a continuous stripe encircling the tetrahedron -/
theorem probability_is_four_twentysevenths (t : StripedTetrahedron) 
  (h1 : t.num_faces = 4)
  (h2 : t.stripes_per_face = 3)
  (h3 : t.total_configurations = t.stripes_per_face ^ t.num_faces)
  (h4 : t.continuous_configurations = 12) : 
  probability_continuous_stripe t = 4 / 27 := by
  sorry

end probability_is_four_twentysevenths_l1066_106687


namespace undefined_values_count_l1066_106693

theorem undefined_values_count : ∃! (s : Finset ℤ), 
  (∀ x ∈ s, (x^2 - x - 6) * (x - 4) = 0) ∧ 
  (∀ x ∉ s, (x^2 - x - 6) * (x - 4) ≠ 0) ∧ 
  s.card = 3 := by
  sorry

end undefined_values_count_l1066_106693


namespace paper_clip_distribution_l1066_106678

theorem paper_clip_distribution (total_clips : ℕ) (clips_per_box : ℕ) (boxes_needed : ℕ) :
  total_clips = 81 →
  clips_per_box = 9 →
  total_clips = clips_per_box * boxes_needed →
  boxes_needed = 9 := by
  sorry

end paper_clip_distribution_l1066_106678


namespace mango_problem_l1066_106648

theorem mango_problem (alexis_mangoes : ℕ) (dilan_ashley_mangoes : ℕ) : 
  alexis_mangoes = 60 → 
  alexis_mangoes = 4 * dilan_ashley_mangoes → 
  alexis_mangoes + dilan_ashley_mangoes = 75 := by
sorry

end mango_problem_l1066_106648


namespace quiche_volume_l1066_106622

theorem quiche_volume (raw_spinach : ℝ) (cooked_spinach_ratio : ℝ) (cream_cheese : ℝ) (eggs : ℝ)
  (h1 : raw_spinach = 40)
  (h2 : cooked_spinach_ratio = 0.2)
  (h3 : cream_cheese = 6)
  (h4 : eggs = 4) :
  raw_spinach * cooked_spinach_ratio + cream_cheese + eggs = 18 :=
by sorry

end quiche_volume_l1066_106622


namespace real_part_of_z_l1066_106616

theorem real_part_of_z (z : ℂ) (h : Complex.I * (z - 3) = -1 + 3 * Complex.I) : 
  z.re = 6 := by
  sorry

end real_part_of_z_l1066_106616


namespace skittles_distribution_l1066_106605

theorem skittles_distribution (total_skittles : ℕ) (num_friends : ℕ) (skittles_per_friend : ℕ) : 
  total_skittles = 40 → num_friends = 5 → skittles_per_friend = total_skittles / num_friends → skittles_per_friend = 8 := by
  sorry

end skittles_distribution_l1066_106605


namespace rachel_homework_l1066_106620

/-- Rachel's homework problem -/
theorem rachel_homework (math_pages reading_pages total_pages : ℕ) : 
  math_pages = 8 ∧ 
  math_pages = reading_pages + 3 →
  total_pages = math_pages + reading_pages ∧
  total_pages = 13 := by
  sorry

end rachel_homework_l1066_106620


namespace evaluate_expression_l1066_106611

theorem evaluate_expression : (2^13 : ℚ) / (5 * 4^3) = 128 / 5 := by
  sorry

end evaluate_expression_l1066_106611


namespace x_squared_minus_4x_geq_m_l1066_106606

theorem x_squared_minus_4x_geq_m (m : ℝ) : 
  (∀ x ∈ Set.Icc 3 4, x^2 - 4*x ≥ m) → m ≤ -3 := by
  sorry

end x_squared_minus_4x_geq_m_l1066_106606


namespace total_rainfall_sum_l1066_106613

/-- The total rainfall recorded over three days equals the sum of individual daily rainfall amounts. -/
theorem total_rainfall_sum (monday tuesday wednesday : Real) 
  (h1 : monday = 0.17)
  (h2 : tuesday = 0.42)
  (h3 : wednesday = 0.08) :
  monday + tuesday + wednesday = 0.67 := by
  sorry

end total_rainfall_sum_l1066_106613


namespace perpendicular_bisector_c_value_l1066_106644

/-- The value of c for which the line 3x - y = c is a perpendicular bisector
    of the line segment from (2,4) to (6,8) -/
theorem perpendicular_bisector_c_value :
  ∃ c : ℝ,
    (∀ x y : ℝ, 3 * x - y = c → 
      ((x - 4) ^ 2 + (y - 6) ^ 2 = 8) ∧ 
      (3 * (x - 4) + (y - 6) = 0)) →
    c = 6 := by
  sorry

end perpendicular_bisector_c_value_l1066_106644


namespace quadratic_minimum_positive_l1066_106660

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 5

-- Theorem statement
theorem quadratic_minimum_positive :
  ∃ (x_min : ℝ), x_min > 0 ∧ 
  (∀ (x : ℝ), f x ≥ f x_min) ∧
  (∀ (ε : ℝ), ε > 0 → ∃ (x : ℝ), x ≠ x_min ∧ |x - x_min| < ε ∧ f x > f x_min) :=
by sorry

end quadratic_minimum_positive_l1066_106660


namespace snail_return_whole_hours_l1066_106675

/-- Represents the snail's movement on a 2D plane -/
structure SnailMovement where
  speed : ℝ
  turnInterval : ℝ
  turnAngle : ℝ

/-- Represents the snail's position on a 2D plane -/
structure Position where
  x : ℝ
  y : ℝ

/-- Calculates the snail's position after a given time -/
def snailPosition (movement : SnailMovement) (time : ℝ) : Position :=
  sorry

/-- Theorem: The snail returns to its starting point only after a whole number of hours -/
theorem snail_return_whole_hours (movement : SnailMovement) 
    (h1 : movement.speed > 0)
    (h2 : movement.turnInterval = 1/4)
    (h3 : movement.turnAngle = π/2) :
  ∀ t : ℝ, snailPosition movement t = snailPosition movement 0 → ∃ n : ℕ, t = n :=
  sorry

end snail_return_whole_hours_l1066_106675


namespace a_plus_b_equals_zero_l1066_106684

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M (a : ℝ) : Set ℝ := {x | x^2 + a*x ≤ 0}

-- Define the complement of M in U
def C_U_M (a b : ℝ) : Set ℝ := {x | x > b ∨ x < 0}

-- Theorem statement
theorem a_plus_b_equals_zero (a b : ℝ) : 
  (∀ x, x ∈ M a ↔ x ∉ C_U_M a b) → a + b = 0 := by
  sorry

end a_plus_b_equals_zero_l1066_106684


namespace pythagorean_triple_for_odd_integer_l1066_106628

theorem pythagorean_triple_for_odd_integer (x : ℕ) 
  (h1 : x > 1) 
  (h2 : Odd x) : 
  ∃ y z : ℕ, 
    y > 0 ∧ 
    z > 0 ∧ 
    y = (x^2 - 1) / 2 ∧ 
    z = (x^2 + 1) / 2 ∧ 
    x^2 + y^2 = z^2 := by
  sorry

end pythagorean_triple_for_odd_integer_l1066_106628


namespace count_valid_labelings_l1066_106623

/-- A labeling of the edges of a rectangular prism with 0s and 1s. -/
def Labeling := Fin 12 → Fin 2

/-- The set of faces of a rectangular prism. -/
def Face := Fin 6

/-- The edges that make up each face of the rectangular prism. -/
def face_edges : Face → Finset (Fin 12) :=
  sorry

/-- The sum of labels on a given face for a given labeling. -/
def face_sum (l : Labeling) (f : Face) : Nat :=
  (face_edges f).sum (fun e => l e)

/-- A labeling is valid if the sum of labels on each face is exactly 2. -/
def is_valid_labeling (l : Labeling) : Prop :=
  ∀ f : Face, face_sum l f = 2

/-- The set of all valid labelings. -/
def valid_labelings : Finset Labeling :=
  sorry

theorem count_valid_labelings :
  valid_labelings.card = 16 :=
sorry

end count_valid_labelings_l1066_106623


namespace quadratic_function_value_l1066_106621

/-- Given a quadratic function f(x) = ax^2 + bx + c where f(1) = 3 and f(2) = 12, prove that f(3) = 21 -/
theorem quadratic_function_value (a b c : ℝ) (f : ℝ → ℝ) 
  (h_quad : ∀ x, f x = a * x^2 + b * x + c)
  (h_f1 : f 1 = 3)
  (h_f2 : f 2 = 12) : 
  f 3 = 21 := by
  sorry

end quadratic_function_value_l1066_106621


namespace rectangular_field_area_l1066_106691

/-- 
Given a rectangular field with one side uncovered and three sides fenced,
prove that the area of the field is 720 square feet when the uncovered side
is 20 feet and the total fencing is 92 feet.
-/
theorem rectangular_field_area (L W : ℝ) : 
  L = 20 →  -- The uncovered side is 20 feet
  2 * W + L = 92 →  -- Total fencing equation
  L * W = 720  -- Area of the field
:= by sorry

end rectangular_field_area_l1066_106691


namespace exists_special_quadrilateral_l1066_106604

/-- Represents a quadrilateral with its properties -/
structure Quadrilateral where
  sides : Fin 4 → ℕ
  diagonals : Fin 2 → ℕ
  area : ℕ
  radius : ℕ

/-- Predicate to check if the quadrilateral is cyclic -/
def isCyclic (q : Quadrilateral) : Prop := sorry

/-- Predicate to check if the side lengths are pairwise distinct -/
def hasPairwiseDistinctSides (q : Quadrilateral) : Prop :=
  ∀ i j, i ≠ j → q.sides i ≠ q.sides j

/-- Theorem stating the existence of a quadrilateral with the required properties -/
theorem exists_special_quadrilateral :
  ∃ q : Quadrilateral,
    isCyclic q ∧
    hasPairwiseDistinctSides q :=
  sorry

end exists_special_quadrilateral_l1066_106604


namespace infinite_male_lineage_l1066_106667

/-- Represents a male person --/
structure Male where
  name : String

/-- Represents the son relationship between two males --/
def is_son_of (son father : Male) : Prop := sorry

/-- Adam, the first male --/
def adam : Male := ⟨"Adam"⟩

/-- An infinite sequence of males --/
def male_sequence : ℕ → Male
| 0 => adam
| n + 1 => sorry

/-- Theorem stating the existence of an infinite male lineage starting from Adam --/
theorem infinite_male_lineage :
  (∀ n : ℕ, is_son_of (male_sequence (n + 1)) (male_sequence n)) ∧
  (∀ n : ℕ, ∃ m : ℕ, m > n) :=
sorry

end infinite_male_lineage_l1066_106667


namespace fraction_equality_l1066_106607

theorem fraction_equality : (5 + 4 - 3) / (5 + 4 + 3) = 1 / 2 := by
  sorry

end fraction_equality_l1066_106607


namespace room_dimension_l1066_106696

/-- Proves that a square room with an area of 14400 square inches has sides of length 10 feet, given that there are 12 inches in a foot. -/
theorem room_dimension (inches_per_foot : ℕ) (area_sq_inches : ℕ) : 
  inches_per_foot = 12 → 
  area_sq_inches = 14400 → 
  ∃ (side_length : ℕ), side_length * side_length * (inches_per_foot * inches_per_foot) = area_sq_inches ∧ 
                        side_length = 10 := by
  sorry

end room_dimension_l1066_106696


namespace binary_multiplication_l1066_106632

def binary_to_nat : List Bool → Nat
  | [] => 0
  | b::bs => (if b then 1 else 0) + 2 * binary_to_nat bs

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then
    []
  else
    (n % 2 = 1) :: nat_to_binary (n / 2)

def binary_num1 : List Bool := [true, true, false, true, true]
def binary_num2 : List Bool := [true, true, true, true]
def binary_result : List Bool := [true, false, true, true, true, true, false, true]

theorem binary_multiplication :
  binary_to_nat binary_num1 * binary_to_nat binary_num2 = binary_to_nat binary_result := by
  sorry

end binary_multiplication_l1066_106632


namespace odd_function_properties_l1066_106665

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define an increasing function on an interval
def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- Define the minimum value of a function on an interval
def MinValueOn (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → m ≤ f x) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = m)

-- Define the maximum value of a function on an interval
def MaxValueOn (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → f x ≤ m) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = m)

-- Theorem statement
theorem odd_function_properties (f : ℝ → ℝ) :
  OddFunction f →
  IncreasingOn f 3 7 →
  MinValueOn f 3 7 1 →
  IncreasingOn f (-7) (-3) ∧ MaxValueOn f (-7) (-3) (-1) :=
by sorry

end odd_function_properties_l1066_106665


namespace apex_angle_of_regular_quad_pyramid_l1066_106669

-- Define a regular quadrilateral pyramid
structure RegularQuadPyramid where
  -- We don't need to specify all properties, just the relevant ones
  apex_angle : ℝ
  dihedral_angle : ℝ

-- State the theorem
theorem apex_angle_of_regular_quad_pyramid 
  (pyramid : RegularQuadPyramid)
  (h : pyramid.dihedral_angle = 2 * pyramid.apex_angle) :
  pyramid.apex_angle = Real.arccos ((Real.sqrt 5 - 1) / 2) := by
  sorry


end apex_angle_of_regular_quad_pyramid_l1066_106669


namespace grinder_purchase_price_l1066_106612

theorem grinder_purchase_price 
  (x : ℝ) -- purchase price of grinder
  (h1 : 0.96 * x + 9200 = x + 8600) -- equation representing the overall transaction
  : x = 15000 := by
  sorry

end grinder_purchase_price_l1066_106612


namespace element_in_set_l1066_106697

open Set

universe u

def U : Set ℕ := {1, 3, 5, 7, 9}

theorem element_in_set (M : Set ℕ) (h : (U \ M) = {1, 3, 5}) : 7 ∈ M := by
  sorry

end element_in_set_l1066_106697


namespace journey_times_equal_l1066_106670

-- Define the variables
def distance1 : ℝ := 120
def distance2 : ℝ := 240

-- Define the theorem
theorem journey_times_equal (speed1 : ℝ) (h1 : speed1 > 0) :
  distance1 / speed1 = distance2 / (2 * speed1) :=
by sorry

end journey_times_equal_l1066_106670


namespace sufficient_not_necessary_and_negation_l1066_106627

theorem sufficient_not_necessary_and_negation :
  (∀ a : ℝ, a > 1 → (1 / a < 1)) ∧
  (∃ a : ℝ, 1 / a < 1 ∧ a ≤ 1) ∧
  (¬(∀ x : ℝ, x^2 + x + 1 < 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≥ 0)) :=
by sorry

end sufficient_not_necessary_and_negation_l1066_106627


namespace dodecagon_diagonals_l1066_106625

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

theorem dodecagon_diagonals :
  num_diagonals dodecagon_sides = 54 := by sorry

end dodecagon_diagonals_l1066_106625


namespace average_increase_l1066_106699

theorem average_increase (s : Finset ℕ) (f : ℕ → ℝ) :
  s.card = 15 →
  (s.sum f) / s.card = 30 →
  (s.sum (λ x => f x + 15)) / s.card = 45 := by
sorry

end average_increase_l1066_106699


namespace grid_sum_equality_l1066_106685

theorem grid_sum_equality (row1 row2 : List ℕ) (x : ℕ) :
  row1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1050] →
  row2 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, x] →
  row1.sum = row2.sum →
  x = 950 := by
  sorry

end grid_sum_equality_l1066_106685


namespace vector_equation_solution_l1066_106652

/-- Given real numbers a and b satisfying a vector equation, prove they equal specific values. -/
theorem vector_equation_solution :
  ∀ (a b : ℝ),
  (![3, 2] : Fin 2 → ℝ) + a • ![6, -4] = ![(-1), 1] + b • ![(-3), 5] →
  a = -23/18 ∧ b = 5/9 := by
  sorry

end vector_equation_solution_l1066_106652


namespace wholesale_price_proof_l1066_106650

def retail_price : ℝ := 144

theorem wholesale_price_proof :
  ∃ (wholesale_price : ℝ),
    wholesale_price = 108 ∧
    retail_price = 144 ∧
    retail_price * 0.9 = wholesale_price + wholesale_price * 0.2 :=
by sorry

end wholesale_price_proof_l1066_106650


namespace triangle_square_side_ratio_l1066_106637

theorem triangle_square_side_ratio (t s : ℝ) : 
  t > 0 ∧ s > 0 ∧ 3 * t = 12 ∧ 4 * s = 12 → t / s = 4 / 3 := by
  sorry

end triangle_square_side_ratio_l1066_106637


namespace today_is_thursday_l1066_106677

-- Define the days of the week
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

-- Define the vehicles
inductive Vehicle
  | A
  | B
  | C
  | D
  | E

def is_weekday (d : Day) : Prop :=
  d ≠ Day.Saturday ∧ d ≠ Day.Sunday

def next_day (d : Day) : Day :=
  match d with
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday
  | Day.Sunday => Day.Monday

def can_operate (v : Vehicle) (d : Day) : Prop := sorry

theorem today_is_thursday 
  (h1 : ∀ (d : Day), is_weekday d → ∃ (v : Vehicle), ¬can_operate v d)
  (h2 : ∀ (d : Day), is_weekday d → (∃ (v1 v2 v3 v4 : Vehicle), can_operate v1 d ∧ can_operate v2 d ∧ can_operate v3 d ∧ can_operate v4 d))
  (h3 : ¬can_operate Vehicle.E Day.Thursday)
  (h4 : ¬can_operate Vehicle.B (next_day today))
  (h5 : ∀ (d : Day), d = today ∨ d = next_day today ∨ d = next_day (next_day today) ∨ d = next_day (next_day (next_day today)) → can_operate Vehicle.A d ∧ can_operate Vehicle.C d)
  (h6 : can_operate Vehicle.E (next_day today))
  : today = Day.Thursday :=
sorry

end today_is_thursday_l1066_106677


namespace systematic_sampling_milk_powder_l1066_106668

/-- Represents a systematic sampling selection -/
def SystematicSample (totalItems : ℕ) (sampleSize : ℕ) : List ℕ :=
  let interval := totalItems / sampleSize
  List.range sampleSize |>.map (fun i => (i + 1) * interval)

/-- The problem statement -/
theorem systematic_sampling_milk_powder :
  SystematicSample 50 5 = [5, 15, 25, 35, 45] := by
  sorry

end systematic_sampling_milk_powder_l1066_106668


namespace sqrt_five_decomposition_l1066_106676

theorem sqrt_five_decomposition (a : ℤ) (b : ℝ) 
  (h1 : Real.sqrt 5 = a + b) 
  (h2 : 0 < b) 
  (h3 : b < 1) : 
  (a - b) * (4 + Real.sqrt 5) = 11 := by
  sorry

end sqrt_five_decomposition_l1066_106676


namespace net_pay_calculation_l1066_106681

/-- Calculate net pay given gross pay and tax paid -/
def netPay (grossPay : ℕ) (taxPaid : ℕ) : ℕ :=
  grossPay - taxPaid

theorem net_pay_calculation (grossPay : ℕ) (taxPaid : ℕ) 
  (h1 : grossPay = 450)
  (h2 : taxPaid = 135) :
  netPay grossPay taxPaid = 315 := by
  sorry

end net_pay_calculation_l1066_106681


namespace team_selection_ways_l1066_106617

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem team_selection_ways : 
  let group_size := 6
  let selection_size := 3
  let num_groups := 2
  (choose group_size selection_size) ^ num_groups = 400 := by sorry

end team_selection_ways_l1066_106617


namespace problem_statement_l1066_106682

theorem problem_statement (a b : ℝ) (h1 : a > 1) (h2 : a * b = a + b + 8) :
  (b > 1) ∧ 
  (∀ x y : ℝ, x > 1 ∧ x * y = x + y + 8 → a + b ≤ x + y) ∧ 
  (∀ x y : ℝ, x > 1 ∧ x * y = x + y + 8 → a * b ≤ x * y) :=
by sorry

end problem_statement_l1066_106682


namespace petes_number_l1066_106688

theorem petes_number : ∃ x : ℝ, 4 * (2 * x + 20) = 200 → x = 15 := by
  sorry

end petes_number_l1066_106688


namespace cubic_decreasing_l1066_106651

-- Define the cubic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 1

-- State the theorem
theorem cubic_decreasing (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) ↔ a < 0 :=
by sorry

end cubic_decreasing_l1066_106651


namespace function_equality_l1066_106641

theorem function_equality (f g h : ℕ → ℕ) 
  (h_injective : Function.Injective h)
  (g_surjective : Function.Surjective g)
  (f_def : ∀ n, f n = g n - h n + 1) :
  ∀ n, f n = 1 :=
sorry

end function_equality_l1066_106641


namespace bernardo_wins_l1066_106658

theorem bernardo_wins (N : ℕ) : N = 63 ↔ 
  N ≤ 1999 ∧ 
  (∀ m : ℕ, m < N → 
    (3*m < 3000 ∧
     3*m + 100 < 3000 ∧
     9*m + 300 < 3000 ∧
     9*m + 400 < 3000 ∧
     27*m + 1200 < 3000 ∧
     27*m + 1300 < 3000)) ∧
  (3*N < 3000 ∧
   3*N + 100 < 3000 ∧
   9*N + 300 < 3000 ∧
   9*N + 400 < 3000 ∧
   27*N + 1200 < 3000 ∧
   27*N + 1300 ≥ 3000) :=
by sorry

end bernardo_wins_l1066_106658


namespace max_expression_value_l1066_106608

def expression (a b c d : ℕ) : ℕ := c * a^b - d

theorem max_expression_value :
  ∃ (a b c d : ℕ), 
    a ∈ ({0, 1, 2, 3} : Set ℕ) ∧ 
    b ∈ ({0, 1, 2, 3} : Set ℕ) ∧ 
    c ∈ ({0, 1, 2, 3} : Set ℕ) ∧ 
    d ∈ ({0, 1, 2, 3} : Set ℕ) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    expression a b c d = 9 ∧
    ∀ (w x y z : ℕ), 
      w ∈ ({0, 1, 2, 3} : Set ℕ) → 
      x ∈ ({0, 1, 2, 3} : Set ℕ) → 
      y ∈ ({0, 1, 2, 3} : Set ℕ) → 
      z ∈ ({0, 1, 2, 3} : Set ℕ) →
      w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
      expression w x y z ≤ 9 :=
by sorry

end max_expression_value_l1066_106608


namespace inequality_proof_l1066_106629

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a^4 + b^4 ≥ a^3 * b + a * b^3 := by
  sorry

end inequality_proof_l1066_106629


namespace inequality_solution_range_l1066_106618

theorem inequality_solution_range (k : ℝ) : 
  (∀ x : ℝ, -x^2 + k*x - 4 < 0) → -4 < k ∧ k < 4 := by sorry

end inequality_solution_range_l1066_106618


namespace parallelogram_area_l1066_106610

/-- Represents a parallelogram with given dimensions -/
structure Parallelogram where
  base : ℝ
  height : ℝ
  shift : ℝ

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := p.base * p.height

/-- Theorem: The area of the specific parallelogram is 140 square feet -/
theorem parallelogram_area :
  let p := Parallelogram.mk 20 7 8
  area p = 140 := by sorry

end parallelogram_area_l1066_106610


namespace smallest_number_with_remainders_l1066_106640

theorem smallest_number_with_remainders : ∃ n : ℕ, 
  (n > 0) ∧ 
  (n % 4 = 1) ∧ 
  (n % 3 = 2) ∧ 
  (n % 5 = 2) ∧
  (∀ m : ℕ, m > 0 ∧ m % 4 = 1 ∧ m % 3 = 2 ∧ m % 5 = 2 → m ≥ n) ∧
  n = 17 :=
by
  sorry

end smallest_number_with_remainders_l1066_106640


namespace min_value_of_quadratic_expression_l1066_106614

theorem min_value_of_quadratic_expression (x y : ℝ) :
  2 * x^2 + 3 * y^2 - 12 * x + 9 * y + 35 ≥ 41 / 4 := by
  sorry

end min_value_of_quadratic_expression_l1066_106614


namespace prime_arithmetic_progression_difference_divisibility_l1066_106649

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def arithmetic_progression (a : ℕ → ℕ) (d : ℕ) :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem prime_arithmetic_progression_difference_divisibility
  (p : ℕ → ℕ)
  (d : ℕ)
  (h_prime : ∀ n, n ∈ Finset.range 15 → is_prime (p n))
  (h_increasing : ∀ n, n ∈ Finset.range 14 → p n < p (n + 1))
  (h_arith_prog : arithmetic_progression p d) :
  ∃ k : ℕ, d = k * (2 * 3 * 5 * 7 * 11 * 13) :=
sorry

end prime_arithmetic_progression_difference_divisibility_l1066_106649


namespace salesman_commission_l1066_106662

/-- Calculates the total commission for a salesman given the commission rates and bonus amount. -/
theorem salesman_commission
  (base_commission_rate : Real)
  (bonus_commission_rate : Real)
  (bonus_threshold : Real)
  (bonus_amount : Real) :
  base_commission_rate = 0.09 →
  bonus_commission_rate = 0.03 →
  bonus_threshold = 10000 →
  bonus_amount = 120 →
  ∃ (total_sales : Real),
    total_sales > bonus_threshold ∧
    bonus_commission_rate * (total_sales - bonus_threshold) = bonus_amount ∧
    base_commission_rate * total_sales + bonus_amount = 1380 :=
by sorry

end salesman_commission_l1066_106662


namespace scientific_notation_equivalence_l1066_106657

-- Define the original number
def original_number : ℕ := 141260

-- Define the scientific notation representation
def scientific_notation : ℝ := 1.4126 * (10 ^ 5)

-- Theorem to prove
theorem scientific_notation_equivalence :
  (original_number : ℝ) = scientific_notation :=
sorry

end scientific_notation_equivalence_l1066_106657


namespace prime_expressions_l1066_106680

theorem prime_expressions (p : ℤ) : 
  Prime p ∧ Prime (2*p + 1) ∧ Prime (4*p + 1) ∧ Prime (6*p + 1) ↔ p = -2 ∨ p = -3 ∨ p = 3 :=
sorry

end prime_expressions_l1066_106680


namespace cauchy_schwarz_inequality_l1066_106634

theorem cauchy_schwarz_inequality (a b a₁ b₁ : ℝ) :
  (a * a₁ + b * b₁)^2 ≤ (a^2 + b^2) * (a₁^2 + b₁^2) := by
  sorry

end cauchy_schwarz_inequality_l1066_106634


namespace page_shoe_collection_l1066_106698

theorem page_shoe_collection (initial_shoes : ℕ) (donation_percentage : ℚ) (new_shoes : ℕ) : 
  initial_shoes = 120 →
  donation_percentage = 45 / 100 →
  new_shoes = 15 →
  initial_shoes - (initial_shoes * donation_percentage).floor + new_shoes = 81 :=
by sorry

end page_shoe_collection_l1066_106698


namespace expression_evaluation_l1066_106695

theorem expression_evaluation : 
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / 
  (2 - 4 + 6 - 8 + 10 - 12 + 14 - 16 + 18) = 3 / 5 := by
  sorry

end expression_evaluation_l1066_106695


namespace f_neg_nine_eq_neg_one_l1066_106672

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the properties of the function f
def is_valid_f (f : ℝ → ℝ) : Prop :=
  ∃ b : ℝ, 
    (∀ x : ℝ, f (-x) = -f x) ∧ 
    (∀ x : ℝ, x ≥ 0 → f x = lg (x + 1) - b)

-- State the theorem
theorem f_neg_nine_eq_neg_one (f : ℝ → ℝ) (h : is_valid_f f) : f (-9) = -1 := by
  sorry

end f_neg_nine_eq_neg_one_l1066_106672


namespace impossibleTo2012_l1066_106639

/-- Represents a 5x5 board with integer values -/
def Board := Fin 5 → Fin 5 → ℤ

/-- Checks if two cells are adjacent (share a common side) -/
def adjacent (i j i' j' : Fin 5) : Bool :=
  (i = i' ∧ (j.val + 1 = j'.val ∨ j.val = j'.val + 1)) ∨
  (j = j' ∧ (i.val + 1 = i'.val ∨ i.val = i'.val + 1))

/-- Represents a single move on the board -/
def move (b : Board) (i j : Fin 5) : Board :=
  fun i' j' => if i' = i ∧ j' = j ∨ adjacent i j i' j' then b i' j' + 1 else b i' j'

/-- Checks if all cells in the board have the value 2012 -/
def allCells2012 (b : Board) : Prop :=
  ∀ i j, b i j = 2012

/-- The initial board with all cells set to zero -/
def initialBoard : Board :=
  fun _ _ => 0

theorem impossibleTo2012 : ¬ ∃ (moves : List (Fin 5 × Fin 5)), 
  allCells2012 (moves.foldl (fun b (i, j) => move b i j) initialBoard) :=
sorry

end impossibleTo2012_l1066_106639


namespace triple_sharp_fifty_l1066_106692

-- Define the # operation
def sharp (N : ℝ) : ℝ := 0.6 * N - 2

-- Theorem statement
theorem triple_sharp_fifty : sharp (sharp (sharp 50)) = 6.88 := by
  sorry

end triple_sharp_fifty_l1066_106692


namespace complex_product_magnitude_l1066_106690

-- Define complex numbers a and b
variable (a b : ℂ)

-- Define real number t
variable (t : ℝ)

-- State the theorem
theorem complex_product_magnitude 
  (h1 : Complex.abs a = 2)
  (h2 : Complex.abs b = 5)
  (h3 : a * b = t - 3 * Complex.I)
  (h4 : t > 0) :
  t = Real.sqrt 91 := by
sorry

end complex_product_magnitude_l1066_106690


namespace initial_toys_count_l1066_106689

/-- 
Given that Emily sold some toys and has some left, this theorem proves
the initial number of toys she had.
-/
theorem initial_toys_count 
  (sold : ℕ) -- Number of toys sold
  (remaining : ℕ) -- Number of toys remaining
  (h1 : sold = 3) -- Condition: Emily sold 3 toys
  (h2 : remaining = 4) -- Condition: Emily now has 4 toys left
  : sold + remaining = 7 := by
  sorry

end initial_toys_count_l1066_106689


namespace sons_age_l1066_106635

theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 26 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 24 := by
sorry

end sons_age_l1066_106635


namespace min_abs_z_l1066_106626

theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 16) + Complex.abs (z - 8 * Complex.I) = 18) :
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ Complex.abs (w - 16) + Complex.abs (w - 8 * Complex.I) = 18 ∧ Complex.abs w = 64 / 9 :=
by sorry

end min_abs_z_l1066_106626


namespace polynomial_root_sum_l1066_106601

theorem polynomial_root_sum (a b : ℝ) : 
  (Complex.I * Real.sqrt 2 + 2 : ℂ) ^ 3 + a * (Complex.I * Real.sqrt 2 + 2) + b = 0 → 
  a + b = 14 := by sorry

end polynomial_root_sum_l1066_106601


namespace combined_average_age_l1066_106645

theorem combined_average_age (people_c : ℕ) (avg_c : ℚ) (people_d : ℕ) (avg_d : ℚ) :
  people_c = 8 →
  avg_c = 35 →
  people_d = 6 →
  avg_d = 30 →
  (people_c * avg_c + people_d * avg_d) / (people_c + people_d : ℚ) = 33 := by
  sorry

end combined_average_age_l1066_106645


namespace karen_tagalongs_sales_l1066_106671

/-- The number of cases Karen picked up -/
def num_cases : ℕ := 3

/-- The number of boxes in each case -/
def boxes_per_case : ℕ := 12

/-- The total number of boxes Karen sold -/
def total_boxes : ℕ := num_cases * boxes_per_case

theorem karen_tagalongs_sales : total_boxes = 36 := by
  sorry

end karen_tagalongs_sales_l1066_106671


namespace manufacturing_sector_degrees_l1066_106694

theorem manufacturing_sector_degrees (total_degrees : ℝ) (total_percent : ℝ) 
  (manufacturing_percent : ℝ) (h1 : total_degrees = 360) 
  (h2 : total_percent = 100) (h3 : manufacturing_percent = 60) : 
  (manufacturing_percent / total_percent) * total_degrees = 216 := by
  sorry

end manufacturing_sector_degrees_l1066_106694


namespace average_minutes_theorem_l1066_106653

/-- Represents the distribution of attendees and their listening durations --/
structure LectureAttendance where
  total_attendees : ℕ
  full_listeners : ℕ
  sleepers : ℕ
  half_listeners : ℕ
  quarter_listeners : ℕ
  lecture_duration : ℕ

/-- Calculates the average minutes heard by attendees --/
def average_minutes_heard (attendance : LectureAttendance) : ℚ :=
  let full_minutes := attendance.full_listeners * attendance.lecture_duration
  let half_minutes := attendance.half_listeners * (attendance.lecture_duration / 2)
  let quarter_minutes := attendance.quarter_listeners * (attendance.lecture_duration / 4)
  let total_minutes := full_minutes + half_minutes + quarter_minutes
  (total_minutes : ℚ) / attendance.total_attendees

/-- The theorem stating the average minutes heard is 59.1 --/
theorem average_minutes_theorem (attendance : LectureAttendance) 
  (h1 : attendance.lecture_duration = 120)
  (h2 : attendance.full_listeners = attendance.total_attendees * 30 / 100)
  (h3 : attendance.sleepers = attendance.total_attendees * 15 / 100)
  (h4 : attendance.half_listeners = (attendance.total_attendees - attendance.full_listeners - attendance.sleepers) * 40 / 100)
  (h5 : attendance.quarter_listeners = attendance.total_attendees - attendance.full_listeners - attendance.sleepers - attendance.half_listeners) :
  average_minutes_heard attendance = 591/10 := by
  sorry

end average_minutes_theorem_l1066_106653


namespace complex_equation_solution_l1066_106663

theorem complex_equation_solution (a b : ℝ) 
  (h : (a - 1 : ℂ) + 2*a*I = -4 + b*I) : b = -6 := by
  sorry

end complex_equation_solution_l1066_106663


namespace new_person_weight_l1066_106624

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 85 →
  ∃ (new_weight : ℝ), new_weight = 105 ∧
    (initial_count : ℝ) * weight_increase = new_weight - replaced_weight :=
by sorry

end new_person_weight_l1066_106624


namespace generatable_pairs_theorem_l1066_106603

/-- Given two positive integers and the operations of sum, product, and integer ratio,
    this function determines which pairs of positive integers can be generated. -/
def generatable_pairs (m n : ℕ+) : Set (ℕ+ × ℕ+) :=
  if m = 1 ∧ n = 1 then Set.univ
  else Set.univ \ {(1, 1)}

/-- The main theorem stating which pairs can be generated based on the initial values -/
theorem generatable_pairs_theorem (m n : ℕ+) :
  (∀ (a b : ℕ+), (a, b) ∈ generatable_pairs m n) ∨
  (∀ (a b : ℕ+), (a, b) ≠ (1, 1) → (a, b) ∈ generatable_pairs m n) :=
sorry

end generatable_pairs_theorem_l1066_106603


namespace trailing_zeros_302_factorial_l1066_106666

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: The number of trailing zeros in 302! is 74 -/
theorem trailing_zeros_302_factorial :
  trailingZeros 302 = 74 := by
  sorry

end trailing_zeros_302_factorial_l1066_106666


namespace triangle_side_length_l1066_106631

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = 60 * π / 180 →  -- Convert 60° to radians
  b = 1 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  c = 4 := by
  sorry

end triangle_side_length_l1066_106631


namespace unique_square_with_property_l1066_106646

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def all_digits_less_than_7 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ (n.digits 10) → d < 7

def add_3_to_digits (n : ℕ) : ℕ :=
  n.digits 10
   |> List.map (· + 3)
   |> List.foldl (λ acc d => acc * 10 + d) 0

theorem unique_square_with_property :
  ∃! N : ℕ,
    1000 ≤ N ∧ N < 10000 ∧
    is_perfect_square N ∧
    all_digits_less_than_7 N ∧
    is_perfect_square (add_3_to_digits N) ∧
    N = 1156 :=
sorry

end unique_square_with_property_l1066_106646


namespace cylinder_volume_change_l1066_106647

theorem cylinder_volume_change (r h V : ℝ) : 
  V = π * r^2 * h → (π * (3*r)^2 * (4*h) = 36 * V) := by sorry

end cylinder_volume_change_l1066_106647


namespace tv_clients_count_l1066_106609

def total_clients : ℕ := 180
def radio_clients : ℕ := 110
def magazine_clients : ℕ := 130
def tv_and_magazine_clients : ℕ := 85
def tv_and_radio_clients : ℕ := 75
def radio_and_magazine_clients : ℕ := 95
def all_three_clients : ℕ := 80

theorem tv_clients_count :
  ∃ (tv_clients : ℕ),
    tv_clients = total_clients + all_three_clients - radio_clients - magazine_clients + 
                 tv_and_magazine_clients + tv_and_radio_clients + radio_and_magazine_clients ∧
    tv_clients = 130 := by
  sorry

end tv_clients_count_l1066_106609


namespace isosceles_right_triangle_max_area_l1066_106673

/-- Given a right triangle with legs of length a and hypotenuse of length c,
    the area of the triangle is maximized when the legs are equal. -/
theorem isosceles_right_triangle_max_area (a c : ℝ) (h1 : 0 < a) (h2 : 0 < c) :
  let area := (1/2) * a * (c^2 - a^2).sqrt
  ∀ b, 0 < b → b^2 + a^2 = c^2 → area ≥ (1/2) * a * b :=
sorry

end isosceles_right_triangle_max_area_l1066_106673


namespace expected_value_is_negative_half_l1066_106643

/-- A three-sided coin with probabilities and payoffs -/
structure ThreeSidedCoin where
  prob_heads : ℚ
  prob_tails : ℚ
  prob_edge : ℚ
  payoff_heads : ℚ
  payoff_tails : ℚ
  payoff_edge : ℚ

/-- Expected value of winnings for a three-sided coin -/
def expected_value (coin : ThreeSidedCoin) : ℚ :=
  coin.prob_heads * coin.payoff_heads +
  coin.prob_tails * coin.payoff_tails +
  coin.prob_edge * coin.payoff_edge

/-- Theorem: Expected value of winnings for the given coin is -1/2 -/
theorem expected_value_is_negative_half :
  let coin : ThreeSidedCoin := {
    prob_heads := 1/4,
    prob_tails := 2/4,
    prob_edge := 1/4,
    payoff_heads := 4,
    payoff_tails := -3,
    payoff_edge := 0
  }
  expected_value coin = -1/2 := by
  sorry

end expected_value_is_negative_half_l1066_106643


namespace scalene_polygon_existence_l1066_106686

theorem scalene_polygon_existence (n : ℕ) : 
  (n ≥ 13) → 
  (∀ (S : Finset ℝ), 
    (S.card = n) → 
    (∀ x ∈ S, 1 ≤ x ∧ x ≤ 2013) → 
    ∃ (a b c : ℝ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ 
      a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
      a + b > c ∧ b + c > a ∧ a + c > b) ∧
  (n = 13) :=
sorry

end scalene_polygon_existence_l1066_106686


namespace yeast_population_growth_l1066_106664

/-- The yeast population growth problem -/
theorem yeast_population_growth
  (initial_population : ℕ)
  (growth_factor : ℕ)
  (time_increments : ℕ)
  (h1 : initial_population = 30)
  (h2 : growth_factor = 3)
  (h3 : time_increments = 3) :
  initial_population * growth_factor ^ time_increments = 810 :=
by sorry

end yeast_population_growth_l1066_106664


namespace sum_of_digits_l1066_106655

theorem sum_of_digits (a₁ a₂ b c : ℕ) :
  a₁ < 10 → a₂ < 10 → b < 10 → c < 10 →
  100 * (10 * a₁ + a₂) + 10 * b + 7 * c = 2024 →
  a₁ + a₂ + b + c = 5 :=
by sorry

end sum_of_digits_l1066_106655


namespace marie_stamps_giveaway_l1066_106630

theorem marie_stamps_giveaway (notebooks : Nat) (stamps_per_notebook : Nat)
  (binders : Nat) (stamps_per_binder : Nat) (keep_percentage : Rat) :
  notebooks = 30 →
  stamps_per_notebook = 120 →
  binders = 7 →
  stamps_per_binder = 210 →
  keep_percentage = 35 / 100 →
  (notebooks * stamps_per_notebook + binders * stamps_per_binder : Nat) -
    (((notebooks * stamps_per_notebook + binders * stamps_per_binder : Nat) : Rat) *
      keep_percentage).floor.toNat = 3296 := by
  sorry

end marie_stamps_giveaway_l1066_106630


namespace change_received_l1066_106638

-- Define the number of apples
def num_apples : ℕ := 5

-- Define the cost per apple in cents
def cost_per_apple : ℕ := 30

-- Define the amount paid in dollars
def amount_paid : ℚ := 10

-- Define the function to calculate change
def calculate_change (num_apples : ℕ) (cost_per_apple : ℕ) (amount_paid : ℚ) : ℚ :=
  amount_paid - (num_apples * cost_per_apple : ℚ) / 100

-- Theorem statement
theorem change_received :
  calculate_change num_apples cost_per_apple amount_paid = 8.5 := by
  sorry

end change_received_l1066_106638


namespace triangle_ABC_is_right_l1066_106615

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line passing through (5,-2)
def line_through_point (x y : ℝ) : Prop := ∃ (m b : ℝ), y = m*x + b ∧ -2 = m*5 + b

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a right triangle
def is_right_triangle (t : Triangle) : Prop :=
  let (x₁, y₁) := t.A
  let (x₂, y₂) := t.B
  let (x₃, y₃) := t.C
  (x₂ - x₁) * (x₃ - x₁) + (y₂ - y₁) * (y₃ - y₁) = 0

-- Theorem statement
theorem triangle_ABC_is_right :
  ∀ (B C : ℝ × ℝ),
  parabola B.1 B.2 →
  parabola C.1 C.2 →
  line_through_point B.1 B.2 →
  line_through_point C.1 C.2 →
  is_right_triangle { A := (1, 2), B := B, C := C } :=
by sorry

end triangle_ABC_is_right_l1066_106615


namespace trigonometric_equality_l1066_106674

theorem trigonometric_equality (θ α β γ x y z : ℝ) 
  (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) 
  (h4 : x ≠ y) (h5 : y ≠ z) (h6 : z ≠ x)
  (h7 : Real.tan (θ + α) / x = Real.tan (θ + β) / y)
  (h8 : Real.tan (θ + β) / y = Real.tan (θ + γ) / z) : 
  (x + y) / (x - y) * Real.sin (α - β) ^ 2 + 
  (y + z) / (y - z) * Real.sin (β - γ) ^ 2 + 
  (z + x) / (z - x) * Real.sin (γ - α) ^ 2 = 0 := by
  sorry

end trigonometric_equality_l1066_106674


namespace capital_ratio_l1066_106602

-- Define the partners' capitals
variable (a b c : ℝ)

-- Define the total profit and b's share
def total_profit : ℝ := 16500
def b_share : ℝ := 6000

-- State the theorem
theorem capital_ratio (h1 : b = 4 * c) (h2 : b_share / total_profit = b / (a + b + c)) :
  a / b = 17 / 4 := by
  sorry

end capital_ratio_l1066_106602


namespace x_power_expression_l1066_106636

theorem x_power_expression (x : ℝ) (h : x + 1/x = 3) :
  x^7 - 5*x^5 + 3*x^3 = 126*x - 48 := by
  sorry

end x_power_expression_l1066_106636


namespace subtraction_of_reciprocals_l1066_106654

theorem subtraction_of_reciprocals (p q : ℚ) : 
  (4 / p = 8) → (4 / q = 18) → (p - q = 5 / 18) := by
  sorry

end subtraction_of_reciprocals_l1066_106654
