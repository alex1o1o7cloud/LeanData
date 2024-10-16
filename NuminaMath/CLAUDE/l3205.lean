import Mathlib

namespace NUMINAMATH_CALUDE_parallelogram_area_l3205_320501

/-- The area of a parallelogram with base 32 cm and height 15 cm is 480 cm². -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
  base = 32 → height = 15 → area = base * height → area = 480 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3205_320501


namespace NUMINAMATH_CALUDE_union_of_given_sets_l3205_320543

theorem union_of_given_sets :
  let A : Set ℕ := {0, 1}
  let B : Set ℕ := {1, 2}
  A ∪ B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_given_sets_l3205_320543


namespace NUMINAMATH_CALUDE_convex_polyhedron_has_32_faces_l3205_320542

/-- A convex polyhedron with pentagonal and hexagonal faces -/
structure ConvexPolyhedron where
  /-- Number of pentagonal faces -/
  pentagonFaces : ℕ
  /-- Number of hexagonal faces -/
  hexagonFaces : ℕ
  /-- Three faces meet at each vertex -/
  threeAtVertex : True
  /-- Each pentagon shares edges with 5 hexagons -/
  pentagonSharing : pentagonFaces * 5 = hexagonFaces * 3
  /-- Each hexagon shares edges with 3 pentagons -/
  hexagonSharing : hexagonFaces * 3 = pentagonFaces * 5

/-- The total number of faces in the polyhedron -/
def ConvexPolyhedron.totalFaces (p : ConvexPolyhedron) : ℕ :=
  p.pentagonFaces + p.hexagonFaces

/-- Theorem: The convex polyhedron has exactly 32 faces -/
theorem convex_polyhedron_has_32_faces (p : ConvexPolyhedron) :
  p.totalFaces = 32 := by
  sorry

#eval ConvexPolyhedron.totalFaces ⟨12, 20, trivial, rfl, rfl⟩

end NUMINAMATH_CALUDE_convex_polyhedron_has_32_faces_l3205_320542


namespace NUMINAMATH_CALUDE_total_watch_time_l3205_320545

/-- Calculate the total watch time for John's videos in a week -/
theorem total_watch_time
  (short_video_length : ℕ)
  (long_video_multiplier : ℕ)
  (short_videos_per_day : ℕ)
  (long_videos_per_day : ℕ)
  (days_per_week : ℕ)
  (retention_rate : ℝ)
  (h1 : short_video_length = 2)
  (h2 : long_video_multiplier = 6)
  (h3 : short_videos_per_day = 2)
  (h4 : long_videos_per_day = 1)
  (h5 : days_per_week = 7)
  (h6 : 0 < retention_rate)
  (h7 : retention_rate ≤ 100)
  : ℝ :=
by
  sorry

#check total_watch_time

end NUMINAMATH_CALUDE_total_watch_time_l3205_320545


namespace NUMINAMATH_CALUDE_element_n3_l3205_320559

/-- Represents a right triangular number array where each column forms an arithmetic sequence
    and each row (starting from the third row) forms a geometric sequence with a constant common ratio. -/
structure TriangularArray where
  -- a[i][j] represents the element in the i-th row and j-th column
  a : Nat → Nat → Rat
  -- Each column forms an arithmetic sequence
  column_arithmetic : ∀ i j k, i ≥ j → k ≥ j → a (i+1) j - a i j = a (k+1) j - a k j
  -- Each row forms a geometric sequence (starting from the third row)
  row_geometric : ∀ i j, i ≥ 3 → j < i → a i (j+1) / a i j = a i (j+2) / a i (j+1)

/-- The element a_{n3} in the n-th row and 3rd column is equal to n/16 -/
theorem element_n3 (arr : TriangularArray) (n : Nat) :
  arr.a n 3 = n / 16 := by
  sorry

end NUMINAMATH_CALUDE_element_n3_l3205_320559


namespace NUMINAMATH_CALUDE_intersection_implies_solution_l3205_320598

/-- Two lines intersecting at a point imply the solution to a related system of equations -/
theorem intersection_implies_solution (b k : ℝ) : 
  (∃ (x y : ℝ), y = -3*x + b ∧ y = -k*x + 1 ∧ x = 1 ∧ y = -2) →
  (∀ (x y : ℝ), 3*x + y = b ∧ k*x + y = 1 ↔ x = 1 ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_implies_solution_l3205_320598


namespace NUMINAMATH_CALUDE_carrot_count_proof_l3205_320520

def total_carrots (initial : ℕ) (thrown_out : ℕ) (additional : ℕ) : ℕ :=
  initial - thrown_out + additional

theorem carrot_count_proof (initial : ℕ) (thrown_out : ℕ) (additional : ℕ) 
  (h1 : initial ≥ thrown_out) : 
  total_carrots initial thrown_out additional = initial - thrown_out + additional :=
by
  sorry

end NUMINAMATH_CALUDE_carrot_count_proof_l3205_320520


namespace NUMINAMATH_CALUDE_correct_league_members_l3205_320569

/-- The number of members in the Valleyball Soccer League --/
def league_members : ℕ := 110

/-- The cost of a pair of socks in dollars --/
def sock_cost : ℕ := 6

/-- The additional cost of a T-shirt compared to a pair of socks in dollars --/
def tshirt_additional_cost : ℕ := 8

/-- The total expenditure of the league in dollars --/
def total_expenditure : ℕ := 3740

/-- Theorem stating that the number of members in the league is correct given the conditions --/
theorem correct_league_members :
  let tshirt_cost : ℕ := sock_cost + tshirt_additional_cost
  let member_cost : ℕ := sock_cost + 2 * tshirt_cost
  total_expenditure = league_members * member_cost :=
by sorry

end NUMINAMATH_CALUDE_correct_league_members_l3205_320569


namespace NUMINAMATH_CALUDE_translate_linear_function_l3205_320576

/-- A linear function in the Cartesian coordinate system. -/
def LinearFunction (a b : ℝ) : ℝ → ℝ := λ x ↦ a * x + b

/-- Vertical translation of a function. -/
def VerticalTranslate (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x ↦ f x - k

theorem translate_linear_function :
  let f := LinearFunction 5 0
  let g := VerticalTranslate f 5
  ∀ x, g x = 5 * x - 5 := by sorry

end NUMINAMATH_CALUDE_translate_linear_function_l3205_320576


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l3205_320502

theorem gcd_of_three_numbers (A B C : ℕ+) 
  (h_lcm : Nat.lcm A.val (Nat.lcm B.val C.val) = 1540)
  (h_prod : A.val * B.val * C.val = 1230000) :
  Nat.gcd A.val (Nat.gcd B.val C.val) = 20 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l3205_320502


namespace NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l3205_320586

/-- Given information about children's emotions and genders -/
structure ChildrenInfo where
  total : Nat
  happy : Nat
  sad : Nat
  neither : Nat
  boys : Nat
  girls : Nat
  happyBoys : Nat
  sadGirls : Nat

/-- Theorem: The number of boys who are neither happy nor sad is 6 -/
theorem boys_neither_happy_nor_sad (info : ChildrenInfo)
  (h1 : info.total = 60)
  (h2 : info.happy = 30)
  (h3 : info.sad = 10)
  (h4 : info.neither = 20)
  (h5 : info.boys = 18)
  (h6 : info.girls = 42)
  (h7 : info.happyBoys = 6)
  (h8 : info.sadGirls = 4)
  (h9 : info.total = info.happy + info.sad + info.neither)
  (h10 : info.total = info.boys + info.girls) :
  info.boys - (info.happyBoys + (info.sad - info.sadGirls)) = 6 :=
by sorry

end NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l3205_320586


namespace NUMINAMATH_CALUDE_intersection_condition_l3205_320546

/-- Given functions f, g, f₁, g₁ and their coefficients, prove that if their graphs intersect
    at a single point with a negative x-coordinate and ac ≠ 0, then bc = ad. -/
theorem intersection_condition (a b c d : ℝ) (x₀ : ℝ) :
  let f := fun x : ℝ => a * x^2 + b * x
  let g := fun x : ℝ => c * x^2 + d * x
  let f₁ := fun x : ℝ => a * x + b
  let g₁ := fun x : ℝ => c * x + d
  (∀ x ≠ x₀, f x ≠ g x ∧ f x ≠ f₁ x ∧ f x ≠ g₁ x ∧
             g x ≠ f₁ x ∧ g x ≠ g₁ x ∧ f₁ x ≠ g₁ x) →
  (f x₀ = g x₀ ∧ f x₀ = f₁ x₀ ∧ f x₀ = g₁ x₀) →
  x₀ < 0 →
  a * c ≠ 0 →
  b * c = a * d :=
by sorry

end NUMINAMATH_CALUDE_intersection_condition_l3205_320546


namespace NUMINAMATH_CALUDE_subset_union_of_product_zero_l3205_320505

variable {X : Type*}
variable (f g : X → ℝ)

def M (f : X → ℝ) := {x : X | f x = 0}
def N (g : X → ℝ) := {x : X | g x = 0}
def P (f g : X → ℝ) := {x : X | f x * g x = 0}

theorem subset_union_of_product_zero (hM : M f ≠ ∅) (hN : N g ≠ ∅) (hP : P f g ≠ ∅) :
  P f g ⊆ M f ∪ N g := by
  sorry

end NUMINAMATH_CALUDE_subset_union_of_product_zero_l3205_320505


namespace NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l3205_320593

-- Define the equations
def equation1 (x : ℝ) : Prop := x^2 + 4*x - 4 = 0
def equation2 (x : ℝ) : Prop := (x - 1)^2 = 2*(x - 1)

-- Theorem for the first equation
theorem solutions_equation1 :
  ∃ (x1 x2 : ℝ), 
    equation1 x1 ∧ equation1 x2 ∧ 
    x1 = -2 + 2 * Real.sqrt 2 ∧ 
    x2 = -2 - 2 * Real.sqrt 2 :=
sorry

-- Theorem for the second equation
theorem solutions_equation2 :
  ∃ (x1 x2 : ℝ), 
    equation2 x1 ∧ equation2 x2 ∧ 
    x1 = 1 ∧ x2 = 3 :=
sorry

end NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l3205_320593


namespace NUMINAMATH_CALUDE_greater_number_proof_l3205_320552

theorem greater_number_proof (x y : ℝ) (h1 : x > y) (h2 : x * y = 2688) (h3 : x + y - (x - y) = 64) : x = 84 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_proof_l3205_320552


namespace NUMINAMATH_CALUDE_bert_crossword_theorem_l3205_320544

/-- Represents a crossword puzzle --/
structure Crossword where
  size : Nat × Nat
  words : Nat

/-- Represents Bert's crossword solving habits --/
structure CrosswordHabit where
  puzzlesPerDay : Nat
  daysToUsePencil : Nat
  wordsPerPencil : Nat

/-- Calculate the average words per puzzle --/
def avgWordsPerPuzzle (habit : CrosswordHabit) : Nat :=
  habit.wordsPerPencil / (habit.puzzlesPerDay * habit.daysToUsePencil)

/-- Calculate the estimated words for a given puzzle size --/
def estimatedWords (baseSize : Nat × Nat) (baseWords : Nat) (newSize : Nat × Nat) : Nat :=
  let baseArea := baseSize.1 * baseSize.2
  let newArea := newSize.1 * newSize.2
  (baseWords * newArea) / baseArea

/-- Main theorem about Bert's crossword habits --/
theorem bert_crossword_theorem (habit : CrosswordHabit)
  (h1 : habit.puzzlesPerDay = 1)
  (h2 : habit.daysToUsePencil = 14)
  (h3 : habit.wordsPerPencil = 1050) :
  avgWordsPerPuzzle habit = 75 ∧
  estimatedWords (15, 15) 75 (21, 21) - 75 = 72 := by
  sorry

end NUMINAMATH_CALUDE_bert_crossword_theorem_l3205_320544


namespace NUMINAMATH_CALUDE_motel_rent_is_400_l3205_320516

/-- The total rent charged by a motel on a Saturday night. -/
def totalRent (r50 r60 : ℕ) : ℝ := 50 * r50 + 60 * r60

/-- The rent after changing 10 rooms from $60 to $50. -/
def newRent (r50 r60 : ℕ) : ℝ := 50 * (r50 + 10) + 60 * (r60 - 10)

/-- The theorem stating that the total rent is $400. -/
theorem motel_rent_is_400 (r50 r60 : ℕ) : 
  (∃ (r50 r60 : ℕ), totalRent r50 r60 = 400 ∧ 
    newRent r50 r60 = 0.75 * totalRent r50 r60) := by
  sorry

end NUMINAMATH_CALUDE_motel_rent_is_400_l3205_320516


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l3205_320514

/-- Represents a cricket team with its age-related properties -/
structure CricketTeam where
  totalMembers : ℕ
  averageAge : ℝ
  captainAgeDiff : ℝ
  remainingAverageAgeDiff : ℝ

/-- Theorem stating that the average age of the cricket team is 30 years -/
theorem cricket_team_average_age
  (team : CricketTeam)
  (h1 : team.totalMembers = 20)
  (h2 : team.averageAge = 30)
  (h3 : team.captainAgeDiff = 5)
  (h4 : team.remainingAverageAgeDiff = 3)
  : team.averageAge = 30 := by
  sorry

#check cricket_team_average_age

end NUMINAMATH_CALUDE_cricket_team_average_age_l3205_320514


namespace NUMINAMATH_CALUDE_sale_price_is_63_percent_l3205_320536

/-- The sale price of an item after two successive discounts -/
def sale_price (original_price : ℝ) : ℝ :=
  let first_discount := 0.1
  let second_discount := 0.3
  let price_after_first_discount := original_price * (1 - first_discount)
  price_after_first_discount * (1 - second_discount)

/-- Theorem stating that the sale price is 63% of the original price -/
theorem sale_price_is_63_percent (x : ℝ) : sale_price x = 0.63 * x := by
  sorry

end NUMINAMATH_CALUDE_sale_price_is_63_percent_l3205_320536


namespace NUMINAMATH_CALUDE_count_valid_three_digit_numbers_l3205_320573

/-- The count of three-digit numbers with specific exclusions -/
def valid_three_digit_numbers : ℕ :=
  let total_three_digit_numbers := 900
  let numbers_with_two_same_nonadjacent_digits := 81
  let numbers_with_increasing_digits := 28
  total_three_digit_numbers - (numbers_with_two_same_nonadjacent_digits + numbers_with_increasing_digits)

/-- Theorem stating the count of valid three-digit numbers -/
theorem count_valid_three_digit_numbers :
  valid_three_digit_numbers = 791 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_three_digit_numbers_l3205_320573


namespace NUMINAMATH_CALUDE_chris_age_l3205_320582

/-- The ages of Amy, Ben, and Chris -/
structure Ages where
  amy : ℝ
  ben : ℝ
  chris : ℝ

/-- The conditions given in the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  -- The average of their ages is 10
  (ages.amy + ages.ben + ages.chris) / 3 = 10 ∧
  -- Five years ago, Chris was twice Amy's age
  ages.chris - 5 = 2 * (ages.amy - 5) ∧
  -- In 5 years, Ben's age will be half of Amy's age
  ages.ben + 5 = (ages.amy + 5) / 2

/-- The theorem to prove -/
theorem chris_age (ages : Ages) (h : satisfies_conditions ages) : 
  ∃ (ε : ℝ), ages.chris = 16 + ε ∧ abs ε < 1 := by
  sorry

end NUMINAMATH_CALUDE_chris_age_l3205_320582


namespace NUMINAMATH_CALUDE_smallest_k_for_binomial_divisibility_l3205_320506

theorem smallest_k_for_binomial_divisibility (k : ℕ) : 
  (k ≥ 25 ∧ 49 ∣ Nat.choose (2 * k) k) ∧ 
  (∀ m : ℕ, m < 25 → ¬(49 ∣ Nat.choose (2 * m) m)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_binomial_divisibility_l3205_320506


namespace NUMINAMATH_CALUDE_nabla_example_l3205_320524

-- Define the nabla operation
def nabla (a b : ℕ) : ℕ := 3 + b^a

-- State the theorem
theorem nabla_example : nabla (nabla 2 3) 2 = 4099 := by
  sorry

end NUMINAMATH_CALUDE_nabla_example_l3205_320524


namespace NUMINAMATH_CALUDE_gcd_power_sum_l3205_320517

theorem gcd_power_sum (n : ℕ) (h : n > 32) :
  Nat.gcd (n^5 + 5^3) (n + 5) = if n % 5 = 0 then 5 else 1 := by sorry

end NUMINAMATH_CALUDE_gcd_power_sum_l3205_320517


namespace NUMINAMATH_CALUDE_opposite_of_cube_root_eight_l3205_320535

theorem opposite_of_cube_root_eight (x : ℝ) : x^3 = 8 → -x = -2 := by sorry

end NUMINAMATH_CALUDE_opposite_of_cube_root_eight_l3205_320535


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3205_320500

/-- The asymptotes of the hyperbola x^2 - y^2/3 = 1 are y = ±√3 x -/
theorem hyperbola_asymptotes :
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2 - y^2/3 = 1
  ∃ (k : ℝ), k^2 = 3 ∧
    (∀ x y, h x y → (y = k*x ∨ y = -k*x))
    ∧ (∀ ε > 0, ∃ δ > 0, ∀ x y, h x y → (|x| > δ → min (|y - k*x|) (|y + k*x|) < ε * |x|)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3205_320500


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l3205_320565

theorem inequality_system_solution_set :
  ∀ x : ℝ, (x + 1 ≥ 0 ∧ x - 2 < 0) ↔ (-1 ≤ x ∧ x < 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l3205_320565


namespace NUMINAMATH_CALUDE_min_value_quadratic_sum_l3205_320591

theorem min_value_quadratic_sum (x y z : ℝ) (h : x + y + z = 2) :
  ∃ (m : ℝ), m = 24 / 11 ∧ ∀ (a b c : ℝ), a + b + c = 2 → 2 * a^2 + 3 * b^2 + c^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_sum_l3205_320591


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3205_320596

theorem polynomial_evaluation (x : ℝ) (h1 : x > 0) (h2 : x^2 + 3*x - 9 = 0) :
  x^3 + 3*x^2 - 9*x - 5 = 22 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3205_320596


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l3205_320594

theorem tangent_circle_radius (R : ℝ) (chord_length : ℝ) (ratio : ℝ) :
  R = 5 →
  chord_length = 8 →
  ratio = 1/3 →
  ∃ (r₁ r₂ : ℝ), (r₁ = 8/9 ∧ r₂ = 32/9) ∧
    (∀ (r : ℝ), (r = r₁ ∨ r = r₂) ↔
      (∃ (C : ℝ × ℝ),
        C.1^2 + C.2^2 = R^2 ∧
        C.1^2 + (C.2 - chord_length * ratio)^2 = r^2 ∧
        (R - r)^2 = (r + C.2)^2 + C.1^2)) :=
by sorry


end NUMINAMATH_CALUDE_tangent_circle_radius_l3205_320594


namespace NUMINAMATH_CALUDE_b_age_is_four_l3205_320531

-- Define the ages as natural numbers
def a : ℕ := sorry
def b : ℕ := sorry
def c : ℕ := sorry

-- State the theorem
theorem b_age_is_four :
  (a = b + 2) →  -- a is two years older than b
  (b = 2 * c) →  -- b is twice as old as c
  (a + b + c = 12) →  -- The total of the ages is 12
  b = 4 := by
  sorry

end NUMINAMATH_CALUDE_b_age_is_four_l3205_320531


namespace NUMINAMATH_CALUDE_min_value_product_squares_l3205_320553

theorem min_value_product_squares (a b c d e f g h i j k l m n o p : ℝ) 
  (h1 : a * b * c * d = 16)
  (h2 : e * f * g * h = 16)
  (h3 : i * j * k * l = 16)
  (h4 : m * n * o * p = 16) :
  (a * e * i * m)^2 + (b * f * j * n)^2 + (c * g * k * o)^2 + (d * h * l * p)^2 ≥ 1024 :=
sorry

end NUMINAMATH_CALUDE_min_value_product_squares_l3205_320553


namespace NUMINAMATH_CALUDE_oranges_per_box_l3205_320578

theorem oranges_per_box (total_oranges : ℕ) (total_boxes : ℕ) 
  (h1 : total_oranges = 2650) 
  (h2 : total_boxes = 265) :
  total_oranges / total_boxes = 10 := by
sorry

end NUMINAMATH_CALUDE_oranges_per_box_l3205_320578


namespace NUMINAMATH_CALUDE_quad_sum_is_six_l3205_320547

/-- A quadrilateral with given properties --/
structure Quadrilateral where
  a : ℤ
  c : ℤ
  a_pos : 0 < a
  c_pos : 0 < c
  a_gt_c : c < a
  symmetric : True  -- Represents symmetry about origin
  equal_diagonals : True  -- Represents equal diagonal lengths
  area : (2 * (a - c).natAbs * (a + c).natAbs : ℤ) = 24

/-- The sum of a and c in a quadrilateral with given properties is 6 --/
theorem quad_sum_is_six (q : Quadrilateral) : q.a + q.c = 6 := by
  sorry

end NUMINAMATH_CALUDE_quad_sum_is_six_l3205_320547


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l3205_320503

theorem hemisphere_surface_area (base_area : ℝ) (h : base_area = 225 * Real.pi) :
  let radius := Real.sqrt (base_area / Real.pi)
  let curved_area := 2 * Real.pi * radius^2
  let total_area := curved_area + base_area
  total_area = 675 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l3205_320503


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l3205_320561

/-- The distance between the foci of the hyperbola x^2 - 4x - 9y^2 - 18y = 56 -/
theorem hyperbola_foci_distance :
  let h : ℝ → ℝ → ℝ := λ x y => x^2 - 4*x - 9*y^2 - 18*y - 56
  ∃ c : ℝ, c > 0 ∧ (∀ x y : ℝ, h x y = 0 → 
    ∃ p q : ℝ, (x - p)^2 / (c^2) - (y - q)^2 / ((c^2) / 9) = 1) ∧
  2 * c = 2 * Real.sqrt (170 / 3) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l3205_320561


namespace NUMINAMATH_CALUDE_union_complement_problem_l3205_320574

def U : Set ℤ := {x : ℤ | |x| < 3}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {-2, -1, 2}

theorem union_complement_problem :
  A ∪ (U \ B) = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_complement_problem_l3205_320574


namespace NUMINAMATH_CALUDE_perimeter_division_ratio_l3205_320557

/-- A square with a point on its diagonal and a line passing through that point. -/
structure SquareWithLine where
  /-- Side length of the square -/
  a : ℝ
  /-- Point M divides diagonal AC in ratio 2:1 -/
  m : ℝ × ℝ
  /-- The line divides the square's area in ratio 9:31 -/
  areaRatio : ℝ × ℝ
  /-- Conditions -/
  h1 : a > 0
  h2 : m = (2*a/3, 2*a/3)
  h3 : areaRatio = (9, 31)

/-- The theorem to be proved -/
theorem perimeter_division_ratio (s : SquareWithLine) :
  let p1 := (9 : ℝ) / 10 * (4 * s.a)
  let p2 := (31 : ℝ) / 10 * (4 * s.a)
  (p1, p2) = (9, 31) := by sorry

end NUMINAMATH_CALUDE_perimeter_division_ratio_l3205_320557


namespace NUMINAMATH_CALUDE_x_less_than_2_necessary_not_sufficient_l3205_320572

theorem x_less_than_2_necessary_not_sufficient :
  (∃ x : ℝ, x^2 < 4 ∧ ¬(x < 2)) ∧
  (∀ x : ℝ, x^2 < 4 → x < 2) :=
by sorry

end NUMINAMATH_CALUDE_x_less_than_2_necessary_not_sufficient_l3205_320572


namespace NUMINAMATH_CALUDE_find_number_l3205_320563

theorem find_number : ∃! x : ℝ, 22 * (x - 36) = 748 ∧ x = 70 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3205_320563


namespace NUMINAMATH_CALUDE_parabola_chord_length_l3205_320558

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (x y : ℝ) : Prop := ∃ t : ℝ, x = 1 + t ∧ y = t

-- Define the intersection points
structure IntersectionPoint where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y
  on_line : line_through_focus x y

-- Theorem statement
theorem parabola_chord_length 
  (A B : IntersectionPoint) 
  (sum_condition : A.x + B.x = 6) : 
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_chord_length_l3205_320558


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l3205_320580

theorem tan_alpha_plus_pi_fourth (α : Real) (h : Real.tan α = 2) : 
  Real.tan (α + π/4) = -3 := by sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l3205_320580


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3205_320541

theorem least_subtraction_for_divisibility (n : ℕ) : 
  ∃ (k : ℕ), k ≤ 9 ∧ (427751 - k) % 10 = 0 ∧ 
  ∀ (m : ℕ), m < k → (427751 - m) % 10 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3205_320541


namespace NUMINAMATH_CALUDE_abs_frac_inequality_l3205_320534

theorem abs_frac_inequality (x : ℝ) : 
  |((3 * x - 2) / (x - 2))| < 3 ↔ 4/3 < x ∧ x < 2 :=
sorry

end NUMINAMATH_CALUDE_abs_frac_inequality_l3205_320534


namespace NUMINAMATH_CALUDE_triangle_properties_l3205_320595

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  a + b + c = 10 →
  Real.sin B + Real.sin C = 4 * Real.sin A →
  b * c = 16 →
  (a = 2 ∧ Real.cos A = 7/8) := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l3205_320595


namespace NUMINAMATH_CALUDE_oreo_shop_combinations_l3205_320577

/-- Represents the number of flavors for each product type -/
structure Flavors where
  oreos : Nat
  milk : Nat
  cookies : Nat

/-- Represents the purchasing rules for Alpha and Gamma -/
structure PurchaseRules where
  alpha_max_items : Nat
  alpha_allows_repeats : Bool
  gamma_allowed_products : List String
  gamma_allows_repeats : Bool

/-- Calculates the number of ways to purchase items given the rules and flavors -/
def purchase_combinations (flavors : Flavors) (rules : PurchaseRules) (total_items : Nat) : Nat :=
  sorry

/-- The main theorem stating the number of purchase combinations -/
theorem oreo_shop_combinations :
  let flavors := Flavors.mk 5 3 2
  let rules := PurchaseRules.mk 2 false ["oreos", "cookies"] true
  purchase_combinations flavors rules 4 = 2100 := by
  sorry

end NUMINAMATH_CALUDE_oreo_shop_combinations_l3205_320577


namespace NUMINAMATH_CALUDE_cylinder_radius_proof_l3205_320518

/-- The radius of a cylinder with given conditions -/
def cylinder_radius : ℝ := 12

theorem cylinder_radius_proof (h : ℝ) (r : ℝ) :
  h = 4 →
  (π * (r + 4)^2 * h = π * r^2 * (h + 4)) →
  r = cylinder_radius := by
  sorry

end NUMINAMATH_CALUDE_cylinder_radius_proof_l3205_320518


namespace NUMINAMATH_CALUDE_chocolate_bar_cost_l3205_320562

/-- The cost of each chocolate bar in dollars -/
def bar_cost : ℚ := 2

/-- The total number of chocolate bars in the box -/
def total_bars : ℕ := 13

/-- The number of unsold chocolate bars -/
def unsold_bars : ℕ := 4

/-- The total amount made from selling chocolate bars in dollars -/
def total_made : ℚ := 18

/-- Theorem stating that the cost of each chocolate bar is $2 -/
theorem chocolate_bar_cost :
  bar_cost * (total_bars - unsold_bars) = total_made :=
sorry

end NUMINAMATH_CALUDE_chocolate_bar_cost_l3205_320562


namespace NUMINAMATH_CALUDE_f_bounds_l3205_320508

/-- A function that returns the size of the largest subfamily of sets that doesn't contain a union -/
noncomputable def f (n : ℕ) : ℕ :=
  sorry

/-- The main theorem stating the bounds for f(n) -/
theorem f_bounds (n : ℕ) : Real.sqrt (2 * n) - 1 ≤ f n ∧ f n ≤ 2 * Real.sqrt n + 1 := by
  sorry

end NUMINAMATH_CALUDE_f_bounds_l3205_320508


namespace NUMINAMATH_CALUDE_box_2_neg1_3_neg2_l3205_320539

/-- Definition of the box operation for integers a, b, c, d -/
def box (a b c d : ℤ) : ℚ := a^b - b^c + c^a + d^a

/-- Theorem stating that box(2,-1,3,-2) = 12.5 -/
theorem box_2_neg1_3_neg2 : box 2 (-1) 3 (-2) = 25/2 := by
  sorry

end NUMINAMATH_CALUDE_box_2_neg1_3_neg2_l3205_320539


namespace NUMINAMATH_CALUDE_total_earnings_theorem_l3205_320564

/-- Represents the investment and return ratios for three investors -/
structure InvestmentData where
  a_invest : ℝ
  b_invest : ℝ
  c_invest : ℝ
  a_return : ℝ
  b_return : ℝ
  c_return : ℝ

/-- Calculates the total earnings given investment data -/
def totalEarnings (data : InvestmentData) : ℝ :=
  data.a_invest * data.a_return +
  data.b_invest * data.b_return +
  data.c_invest * data.c_return

/-- Theorem stating the total earnings under given conditions -/
theorem total_earnings_theorem (data : InvestmentData) :
  data.a_invest = 3 ∧
  data.b_invest = 4 ∧
  data.c_invest = 5 ∧
  data.a_return = 6 ∧
  data.b_return = 5 ∧
  data.c_return = 4 ∧
  data.b_invest * data.b_return = data.a_invest * data.a_return + 200 →
  totalEarnings data = 58000 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_theorem_l3205_320564


namespace NUMINAMATH_CALUDE_pentagon_area_theorem_l3205_320575

/-- A pentagon is a polygon with 5 sides -/
structure Pentagon where
  sides : Fin 5 → ℝ

/-- The area of a pentagon -/
noncomputable def Pentagon.area (p : Pentagon) : ℝ := sorry

/-- Theorem: There exists a pentagon with sides 18, 25, 30, 28, and 25 units, and its area is 950 square units -/
theorem pentagon_area_theorem : 
  ∃ (p : Pentagon), 
    p.sides 0 = 18 ∧ 
    p.sides 1 = 25 ∧ 
    p.sides 2 = 30 ∧ 
    p.sides 3 = 28 ∧ 
    p.sides 4 = 25 ∧ 
    p.area = 950 := by sorry

end NUMINAMATH_CALUDE_pentagon_area_theorem_l3205_320575


namespace NUMINAMATH_CALUDE_coprime_sum_not_divides_power_sum_l3205_320567

theorem coprime_sum_not_divides_power_sum
  (x y n : ℕ)
  (h_coprime : Nat.Coprime x y)
  (h_positive : 0 < x ∧ 0 < y)
  (h_not_one : x * y ≠ 1)
  (h_even : Even n)
  (h_pos : 0 < n) :
  ¬ (x + y ∣ x^n + y^n) :=
sorry

end NUMINAMATH_CALUDE_coprime_sum_not_divides_power_sum_l3205_320567


namespace NUMINAMATH_CALUDE_bubble_gum_cost_l3205_320548

/-- Given a number of bubble gum pieces and a total cost in cents,
    calculate the cost per piece of bubble gum. -/
def cost_per_piece (num_pieces : ℕ) (total_cost : ℕ) : ℕ :=
  total_cost / num_pieces

/-- Theorem stating that the cost per piece of bubble gum is 18 cents
    given the specific conditions of the problem. -/
theorem bubble_gum_cost :
  cost_per_piece 136 2448 = 18 := by
  sorry

end NUMINAMATH_CALUDE_bubble_gum_cost_l3205_320548


namespace NUMINAMATH_CALUDE_village_population_l3205_320588

theorem village_population (P : ℕ) : 
  (P : ℝ) * 0.9 * 0.8 = 4554 → P = 6325 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l3205_320588


namespace NUMINAMATH_CALUDE_retailer_profit_percent_l3205_320554

/-- Calculates the profit percent for a retailer given the purchase price, overhead expenses, and selling price. -/
theorem retailer_profit_percent
  (purchase_price : ℚ)
  (overhead_expenses : ℚ)
  (selling_price : ℚ)
  (h1 : purchase_price = 225)
  (h2 : overhead_expenses = 15)
  (h3 : selling_price = 300) :
  (selling_price - (purchase_price + overhead_expenses)) / (purchase_price + overhead_expenses) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_retailer_profit_percent_l3205_320554


namespace NUMINAMATH_CALUDE_right_triangular_pyramid_surface_area_l3205_320549

/-- A regular triangular pyramid with right-angled lateral faces -/
structure RightTriangularPyramid where
  base_edge : ℝ
  is_regular : Bool
  lateral_faces_right_angled : Bool

/-- The total surface area of a right triangular pyramid -/
def total_surface_area (p : RightTriangularPyramid) : ℝ :=
  sorry

/-- Theorem: The total surface area of a regular triangular pyramid with 
    right-angled lateral faces and base edge length 2 is 3 + √3 -/
theorem right_triangular_pyramid_surface_area :
  ∀ (p : RightTriangularPyramid), 
    p.base_edge = 2 → 
    p.is_regular = true → 
    p.lateral_faces_right_angled = true → 
    total_surface_area p = 3 + Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangular_pyramid_surface_area_l3205_320549


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3205_320521

/-- Given a hyperbola and a parabola with the following properties:
    - The hyperbola has the general equation x^2/a^2 - y^2/b^2 = 1, where a > 0 and b > 0
    - One focus of the hyperbola coincides with the focus of the parabola y^2 = 20x
    - The eccentricity of the hyperbola is √5
    Prove that the equation of the hyperbola is x^2/5 - y^2/20 = 1 -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) ∧
  (∃ (x y : ℝ), y^2 = 20*x) ∧
  (∃ (c : ℝ), c/a = Real.sqrt 5) →
  a^2 = 5 ∧ b^2 = 20 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3205_320521


namespace NUMINAMATH_CALUDE_min_square_sum_l3205_320583

theorem min_square_sum (y₁ y₂ y₃ : ℝ) 
  (h_pos : y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0) 
  (h_sum : 2*y₁ + 3*y₂ + 4*y₃ = 120) : 
  y₁^2 + y₂^2 + y₃^2 ≥ 14400/29 := by
  sorry

end NUMINAMATH_CALUDE_min_square_sum_l3205_320583


namespace NUMINAMATH_CALUDE_sum_of_star_tip_angles_l3205_320526

/-- The angle measurement of one tip of an 8-pointed star formed by connecting
    eight evenly spaced points on a circle -/
def star_tip_angle : ℝ := 67.5

/-- The number of tips in an 8-pointed star -/
def num_tips : ℕ := 8

/-- Theorem: The sum of the angle measurements of the eight tips of an 8-pointed star,
    formed by connecting eight evenly spaced points on a circle, is equal to 540° -/
theorem sum_of_star_tip_angles :
  (num_tips : ℝ) * star_tip_angle = 540 := by sorry

end NUMINAMATH_CALUDE_sum_of_star_tip_angles_l3205_320526


namespace NUMINAMATH_CALUDE_part_one_part_two_l3205_320599

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Part 1
theorem part_one (a : ℝ) (h : a ≤ 2) :
  {x : ℝ | f a x ≥ 2} = {x : ℝ | x ≤ 1/2 ∨ x ≥ 5/2} := by sorry

-- Part 2
theorem part_two :
  {a : ℝ | a > 1 ∧ ∀ x, f a x + |x - 1| ≥ 1} = {a : ℝ | a ≥ 2} := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3205_320599


namespace NUMINAMATH_CALUDE_increasing_perfect_powers_sum_l3205_320525

def s (n : ℕ+) : ℕ := sorry

theorem increasing_perfect_powers_sum (x : ℝ) :
  ∃ N : ℕ, ∀ n > N, (Finset.range n).sup (fun i => s ⟨i + 1, Nat.succ_pos i⟩) / n > x := by
  sorry

end NUMINAMATH_CALUDE_increasing_perfect_powers_sum_l3205_320525


namespace NUMINAMATH_CALUDE_second_student_marks_l3205_320507

/-- Proves that given two students' marks satisfying specific conditions, 
    the student with the lower score obtained 33 marks. -/
theorem second_student_marks : 
  ∀ (x y : ℝ), 
  x = y + 9 →  -- First student scored 9 marks more
  x = 0.56 * (x + y) →  -- Higher score is 56% of sum
  y = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_second_student_marks_l3205_320507


namespace NUMINAMATH_CALUDE_mutually_exclusive_but_not_opposite_l3205_320585

/-- A bag containing red and black balls -/
structure Bag where
  red : Nat
  black : Nat

/-- The possible outcomes when drawing two balls -/
inductive Outcome
  | OneBlack
  | TwoBlack
  | NoBlack

/-- Two events are mutually exclusive if they cannot occur simultaneously -/
def mutuallyExclusive (e1 e2 : Set Outcome) : Prop :=
  e1 ∩ e2 = ∅

/-- Two events are opposite if their union is the entire sample space -/
def opposite (e1 e2 : Set Outcome) (Ω : Set Outcome) : Prop :=
  e1 ∪ e2 = Ω

/-- The theorem to be proved -/
theorem mutually_exclusive_but_not_opposite (bag : Bag) 
    (h1 : bag.red = 2) (h2 : bag.black = 2) : 
    ∃ (e1 e2 : Set Outcome) (Ω : Set Outcome), 
      mutuallyExclusive e1 e2 ∧ ¬opposite e1 e2 Ω := by
  sorry

end NUMINAMATH_CALUDE_mutually_exclusive_but_not_opposite_l3205_320585


namespace NUMINAMATH_CALUDE_complex_sum_equals_negative_two_l3205_320504

theorem complex_sum_equals_negative_two (z : ℂ) 
  (h1 : z = Complex.exp (6 * Real.pi * Complex.I / 11))
  (h2 : z^11 = 1) : 
  (z / (1 + z^3)) + (z^2 / (1 + z^6)) + (z^4 / (1 + z^9)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equals_negative_two_l3205_320504


namespace NUMINAMATH_CALUDE_aprils_roses_l3205_320550

theorem aprils_roses (initial_roses : ℕ) 
  (rose_price : ℕ) 
  (total_earnings : ℕ) 
  (roses_left : ℕ) : 
  rose_price = 4 → 
  total_earnings = 36 → 
  roses_left = 4 → 
  initial_roses = 13 := by
  sorry

end NUMINAMATH_CALUDE_aprils_roses_l3205_320550


namespace NUMINAMATH_CALUDE_polynomial_root_relation_l3205_320527

/-- Given two polynomial equations:
    1. x^2 - ax + b = 0 with roots α and β
    2. x^2 - px + q = 0 with roots α^2 + β^2 and αβ
    Prove that p = a^2 - b -/
theorem polynomial_root_relation (a b p q α β : ℝ) : 
  (∀ x, x^2 - a*x + b = 0 ↔ x = α ∨ x = β) →
  (∀ x, x^2 - p*x + q = 0 ↔ x = α^2 + β^2 ∨ x = α*β) →
  p = a^2 - b := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_relation_l3205_320527


namespace NUMINAMATH_CALUDE_total_rulers_l3205_320556

/-- Given an initial number of rulers and a number of rulers added, 
    the total number of rulers is equal to their sum. -/
theorem total_rulers (initial_rulers added_rulers : ℕ) :
  initial_rulers + added_rulers = initial_rulers + added_rulers :=
by sorry

end NUMINAMATH_CALUDE_total_rulers_l3205_320556


namespace NUMINAMATH_CALUDE_inscribed_circle_chord_length_l3205_320570

theorem inscribed_circle_chord_length (a b : Real) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = 1) :
  let r := (a + b - 1) / 2
  let chord_length := Real.sqrt (1 - 2 * r^2)
  chord_length = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_chord_length_l3205_320570


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l3205_320587

theorem regular_polygon_interior_angle_sum (exterior_angle : ℝ) : 
  exterior_angle = 72 → (360 / exterior_angle - 2) * 180 = 540 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l3205_320587


namespace NUMINAMATH_CALUDE_binary_101101_equals_octal_55_l3205_320509

-- Define a function to convert binary to decimal
def binary_to_decimal (binary : List Bool) : Nat :=
  binary.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

-- Define a function to convert decimal to octal
def decimal_to_octal (n : Nat) : List Nat :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

-- Theorem statement
theorem binary_101101_equals_octal_55 :
  let binary := [true, false, true, true, false, true]
  let octal := [5, 5]
  binary_to_decimal binary = (octal.foldr (fun digit acc => 8 * acc + digit) 0) := by
  sorry

end NUMINAMATH_CALUDE_binary_101101_equals_octal_55_l3205_320509


namespace NUMINAMATH_CALUDE_molecular_weight_proof_l3205_320533

-- Define atomic weights
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_O : ℝ := 16.00
def atomic_weight_Fe : ℝ := 55.85
def atomic_weight_H : ℝ := 1.01

-- Define the number of atoms for each element
def num_Al : ℕ := 2
def num_O : ℕ := 3
def num_Fe : ℕ := 2
def num_H : ℕ := 4

-- Define the molecular weight calculation function
def molecular_weight : ℝ :=
  (num_Al : ℝ) * atomic_weight_Al +
  (num_O : ℝ) * atomic_weight_O +
  (num_Fe : ℝ) * atomic_weight_Fe +
  (num_H : ℝ) * atomic_weight_H

-- Theorem statement
theorem molecular_weight_proof :
  molecular_weight = 217.70 := by sorry

end NUMINAMATH_CALUDE_molecular_weight_proof_l3205_320533


namespace NUMINAMATH_CALUDE_product_minimum_value_l3205_320568

-- Define the functions h and k
def h : ℝ → ℝ := sorry
def k : ℝ → ℝ := sorry

-- State the theorem
theorem product_minimum_value (x : ℝ) :
  (∀ x, -3 ≤ h x ∧ h x ≤ 4) →
  (∀ x, -1 ≤ k x ∧ k x ≤ 3) →
  -12 ≤ h x * k x :=
sorry

end NUMINAMATH_CALUDE_product_minimum_value_l3205_320568


namespace NUMINAMATH_CALUDE_polar_curve_is_circle_l3205_320590

/-- The curve defined by the polar equation r = 1 / (sin θ + cos θ) is a circle. -/
theorem polar_curve_is_circle :
  ∀ θ : ℝ, ∃ r : ℝ, r = 1 / (Real.sin θ + Real.cos θ) → ∃ c x₀ y₀ : ℝ, 
    (r * Real.cos θ - x₀)^2 + (r * Real.sin θ - y₀)^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_polar_curve_is_circle_l3205_320590


namespace NUMINAMATH_CALUDE_no_real_roots_composite_l3205_320589

-- Define the function f
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem no_real_roots_composite (b c : ℝ) :
  (∀ x : ℝ, f b c x ≠ x) →
  (∀ x : ℝ, f b c (f b c x) ≠ x) :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_composite_l3205_320589


namespace NUMINAMATH_CALUDE_division_vs_multiplication_error_l3205_320560

theorem division_vs_multiplication_error (x : ℝ) (h : x > 0) :
  ∃ (ε : ℝ), abs (ε - 98) < 1 ∧
  (abs ((8 * x) - (x / 8)) / (8 * x)) * 100 = ε :=
sorry

end NUMINAMATH_CALUDE_division_vs_multiplication_error_l3205_320560


namespace NUMINAMATH_CALUDE_reciprocal_sum_l3205_320519

theorem reciprocal_sum : (1 / (1 / 4 + 1 / 6) : ℚ) = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_l3205_320519


namespace NUMINAMATH_CALUDE_game_ends_after_33_rounds_l3205_320523

/-- Represents a player in the token redistribution game -/
inductive Player
| P
| Q
| R

/-- State of the game, tracking token counts for each player and the number of rounds played -/
structure GameState where
  tokens : Player → ℕ
  rounds : ℕ

/-- Determines if the game has ended (any player has 0 tokens) -/
def gameEnded (state : GameState) : Prop :=
  ∃ p : Player, state.tokens p = 0

/-- Simulates one round of the game -/
def playRound (state : GameState) : GameState :=
  sorry

/-- The initial state of the game -/
def initialState : GameState :=
  { tokens := λ p => match p with
    | Player.P => 12
    | Player.Q => 10
    | Player.R => 8,
    rounds := 0 }

/-- The main theorem: the game ends after 33 rounds -/
theorem game_ends_after_33_rounds :
  ∃ finalState : GameState,
    finalState.rounds = 33 ∧
    gameEnded finalState ∧
    (∀ n : ℕ, n < 33 → ¬gameEnded ((playRound^[n]) initialState)) :=
  sorry

end NUMINAMATH_CALUDE_game_ends_after_33_rounds_l3205_320523


namespace NUMINAMATH_CALUDE_ellipse_properties_line_through_focus_l3205_320555

/-- Ellipse C defined by the equation x²/2 + y² = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/2 + y^2 = 1

/-- Point on ellipse C -/
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse_C x y

/-- Line passing through a point with slope k -/
def line (k : ℝ) (x₀ y₀ : ℝ) (x y : ℝ) : Prop :=
  y - y₀ = k * (x - x₀)

theorem ellipse_properties :
  let a := Real.sqrt 2
  let b := 1
  let c := 1
  let e := c / a
  let left_focus := (-1, 0)
  (∀ x y, ellipse_C x y → x^2 / (a^2) + y^2 / (b^2) = 1) ∧
  (2 * a = 2 * Real.sqrt 2) ∧
  (2 * b = 2) ∧
  (e = Real.sqrt 2 / 2) ∧
  (left_focus.1 = -c ∧ left_focus.2 = 0) :=
by sorry

theorem line_through_focus (k : ℝ) :
  let left_focus := (-1, 0)
  ∃ A B : PointOnEllipse,
    line k left_focus.1 left_focus.2 A.x A.y ∧
    line k left_focus.1 left_focus.2 B.x B.y ∧
    (A.x - B.x)^2 + (A.y - B.y)^2 = (8 * Real.sqrt 2 / 7)^2 →
    k = Real.sqrt 3 ∨ k = -Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_line_through_focus_l3205_320555


namespace NUMINAMATH_CALUDE_find_a_l3205_320512

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 2^x else a + 2*x

-- State the theorem
theorem find_a : ∃ a : ℝ, (f a (f a (-1)) = 2) ∧ (a = 1) := by
  sorry

end NUMINAMATH_CALUDE_find_a_l3205_320512


namespace NUMINAMATH_CALUDE_circle_diameter_endpoint_l3205_320529

/-- Given a circle with center (4, 6) and one endpoint of a diameter at (1, 2),
    the other endpoint of the diameter is at (7, 10). -/
theorem circle_diameter_endpoint (P : Set (ℝ × ℝ)) : 
  (∃ (r : ℝ), P = {p | ∃ (x y : ℝ), p = (x, y) ∧ (x - 4)^2 + (y - 6)^2 = r^2}) →
  ((1, 2) ∈ P) →
  ((7, 10) ∈ P) ∧ 
  (∀ (x y : ℝ), (x, y) ∈ P → (x - 4)^2 + (y - 6)^2 = (7 - 4)^2 + (10 - 6)^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_diameter_endpoint_l3205_320529


namespace NUMINAMATH_CALUDE_bird_families_migration_l3205_320584

theorem bird_families_migration (total : ℕ) (difference : ℕ) (flew_away : ℕ) : 
  total = 87 → difference = 73 → flew_away + (flew_away + difference) = total → flew_away = 7 := by
  sorry

end NUMINAMATH_CALUDE_bird_families_migration_l3205_320584


namespace NUMINAMATH_CALUDE_surface_generates_solid_by_rotation_l3205_320597

/-- A right-angled triangle -/
structure RightTriangle where
  /-- The triangle has a right angle -/
  has_right_angle : Bool

/-- A cone -/
structure Cone where
  /-- The cone is formed by rotation -/
  formed_by_rotation : Bool

/-- Rotation of a triangle around one of its perpendicular sides -/
def rotate_triangle (t : RightTriangle) : Cone :=
  { formed_by_rotation := true }

/-- A theorem stating that rotating a right-angled triangle around one of its perpendicular sides
    demonstrates that a surface can generate a solid through rotation -/
theorem surface_generates_solid_by_rotation (t : RightTriangle) :
  ∃ (c : Cone), c = rotate_triangle t ∧ c.formed_by_rotation :=
by sorry

end NUMINAMATH_CALUDE_surface_generates_solid_by_rotation_l3205_320597


namespace NUMINAMATH_CALUDE_mean_salary_calculation_l3205_320528

def total_employees : ℕ := 100
def salary_group_1 : ℕ := 6000
def salary_group_2 : ℕ := 4000
def salary_group_3 : ℕ := 2500
def employees_group_1 : ℕ := 5
def employees_group_2 : ℕ := 15
def employees_group_3 : ℕ := 80

theorem mean_salary_calculation :
  (salary_group_1 * employees_group_1 + salary_group_2 * employees_group_2 + salary_group_3 * employees_group_3) / total_employees = 2900 := by
  sorry

end NUMINAMATH_CALUDE_mean_salary_calculation_l3205_320528


namespace NUMINAMATH_CALUDE_cost_price_calculation_cost_price_is_15000_l3205_320515

theorem cost_price_calculation (selling_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : ℝ :=
  let discounted_price := selling_price * (1 - discount_rate)
  let cost_price := discounted_price / (1 + profit_rate)
  cost_price

theorem cost_price_is_15000 : 
  cost_price_calculation 18000 0.1 0.08 = 15000 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_cost_price_is_15000_l3205_320515


namespace NUMINAMATH_CALUDE_complex_calculation_equality_l3205_320532

theorem complex_calculation_equality : 
  (2 * (15^2 + 35^2 + 21^2) - (3^4 + 5^4 + 7^4)) / (3 + 5 + 7) = 45 := by
  sorry

end NUMINAMATH_CALUDE_complex_calculation_equality_l3205_320532


namespace NUMINAMATH_CALUDE_fraction_of_fraction_one_eighth_of_one_third_l3205_320530

theorem fraction_of_fraction (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) := by sorry

theorem one_eighth_of_one_third :
  (1 / 8 : ℚ) / (1 / 3 : ℚ) = 3 / 8 := by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_one_eighth_of_one_third_l3205_320530


namespace NUMINAMATH_CALUDE_fiscal_revenue_scientific_notation_l3205_320511

/-- Converts a number in billions to scientific notation -/
def to_scientific_notation (x : ℝ) : ℝ × ℤ :=
  (1.14, 9)

/-- The fiscal revenue in billions -/
def fiscal_revenue : ℝ := 1.14

theorem fiscal_revenue_scientific_notation :
  to_scientific_notation fiscal_revenue = (1.14, 9) := by
  sorry

end NUMINAMATH_CALUDE_fiscal_revenue_scientific_notation_l3205_320511


namespace NUMINAMATH_CALUDE_mary_james_seating_probability_l3205_320510

/-- The number of chairs in the row -/
def total_chairs : ℕ := 10

/-- The number of chairs Mary can choose from -/
def mary_choices : ℕ := 9

/-- The number of chairs James can choose from -/
def james_choices : ℕ := 10

/-- The probability that Mary and James do not sit next to each other -/
def prob_not_adjacent : ℚ := 8/9

theorem mary_james_seating_probability :
  prob_not_adjacent = 1 - (mary_choices.pred / (mary_choices * james_choices)) :=
by sorry

end NUMINAMATH_CALUDE_mary_james_seating_probability_l3205_320510


namespace NUMINAMATH_CALUDE_production_days_calculation_l3205_320537

theorem production_days_calculation (n : ℕ) : 
  (∀ k : ℕ, k > 0 → (60 * n + 90) / (n + 1) = 65) → n = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_production_days_calculation_l3205_320537


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l3205_320540

def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, x = y^2

def is_perfect_cube (x : ℕ) : Prop := ∃ y : ℕ, x = y^3

theorem smallest_n_square_and_cube :
  let n := 54
  (∀ m : ℕ, m > 0 ∧ m < n → ¬(is_perfect_square (3*m) ∧ is_perfect_cube (4*m))) ∧
  is_perfect_square (3*n) ∧ is_perfect_cube (4*n) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l3205_320540


namespace NUMINAMATH_CALUDE_three_numbers_proof_l3205_320592

theorem three_numbers_proof (a b c : ℕ) (h1 : a < b) (h2 : b < c) 
  (h3 : (a + b + c) / 3 = b) (h4 : c - a = 321) (h5 : a + c = 777) : 
  a = 228 ∧ b = 549 ∧ c = 870 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_proof_l3205_320592


namespace NUMINAMATH_CALUDE_emily_calculation_l3205_320571

theorem emily_calculation (a b c : ℝ) 
  (h1 : a - (2*b - c) = 15) 
  (h2 : a - 2*b - c = 5) : 
  a - 2*b = 10 := by sorry

end NUMINAMATH_CALUDE_emily_calculation_l3205_320571


namespace NUMINAMATH_CALUDE_M_equals_N_l3205_320522

def M : Set ℤ := {u | ∃ m n l : ℤ, u = 12*m + 8*n + 4*l}
def N : Set ℤ := {u | ∃ p q r : ℤ, u = 20*p + 16*q + 12*r}

theorem M_equals_N : M = N := by sorry

end NUMINAMATH_CALUDE_M_equals_N_l3205_320522


namespace NUMINAMATH_CALUDE_smallest_common_factor_l3205_320513

theorem smallest_common_factor (n : ℕ) : 
  (∀ m : ℕ, m < 43 → Nat.gcd (8*m - 3) (5*m + 2) = 1) ∧ 
  Nat.gcd (8*43 - 3) (5*43 + 2) > 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_common_factor_l3205_320513


namespace NUMINAMATH_CALUDE_factorial_ratio_simplification_l3205_320551

theorem factorial_ratio_simplification (N : ℕ) (h : N ≥ 2) :
  (Nat.factorial (N - 2) * (N - 1) * N) / Nat.factorial (N + 2) = 1 / ((N + 2) * (N + 1)) :=
by sorry

end NUMINAMATH_CALUDE_factorial_ratio_simplification_l3205_320551


namespace NUMINAMATH_CALUDE_palindromic_not_end_zero_two_digit_palindromic_count_three_digit_palindromic_count_four_digit_palindromic_count_ten_digit_palindromic_count_l3205_320566

/-- A number is palindromic if it reads the same backward as forward. -/
def IsPalindromic (n : ℕ) : Prop := sorry

/-- The count of palindromic numbers with a given number of digits. -/
def PalindromicCount (digits : ℕ) : ℕ := sorry

/-- Palindromic numbers with more than two digits cannot end in 0. -/
theorem palindromic_not_end_zero (n : ℕ) (h : n > 99) (h_pal : IsPalindromic n) : n % 10 ≠ 0 := sorry

/-- There are 9 two-digit palindromic numbers. -/
theorem two_digit_palindromic_count : PalindromicCount 2 = 9 := sorry

/-- There are 90 three-digit palindromic numbers. -/
theorem three_digit_palindromic_count : PalindromicCount 3 = 90 := sorry

/-- There are 90 four-digit palindromic numbers. -/
theorem four_digit_palindromic_count : PalindromicCount 4 = 90 := sorry

/-- The main theorem: There are 90000 ten-digit palindromic numbers. -/
theorem ten_digit_palindromic_count : PalindromicCount 10 = 90000 := sorry

end NUMINAMATH_CALUDE_palindromic_not_end_zero_two_digit_palindromic_count_three_digit_palindromic_count_four_digit_palindromic_count_ten_digit_palindromic_count_l3205_320566


namespace NUMINAMATH_CALUDE_florist_roses_problem_l3205_320538

/-- Proves that the initial number of roses was 37 given the conditions of the problem. -/
theorem florist_roses_problem (initial_roses : ℕ) : 
  (initial_roses - 16 + 19 = 40) → initial_roses = 37 := by
  sorry

end NUMINAMATH_CALUDE_florist_roses_problem_l3205_320538


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3205_320581

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def sum_arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * a₁ + n * (n - 1) / 2 * d

theorem arithmetic_sequence_sum (a₁ : ℤ) (d : ℤ) :
  (sum_arithmetic_sequence a₁ d 12 / 12 : ℚ) - (sum_arithmetic_sequence a₁ d 10 / 10 : ℚ) = 2 →
  sum_arithmetic_sequence a₁ d 2018 = -2018 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3205_320581


namespace NUMINAMATH_CALUDE_det_sum_of_matrices_l3205_320579

def A : Matrix (Fin 2) (Fin 2) ℤ := !![5, -2; 3, 4]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![1, 3; -1, 2]

theorem det_sum_of_matrices : Matrix.det (A + B) = 34 := by sorry

end NUMINAMATH_CALUDE_det_sum_of_matrices_l3205_320579
