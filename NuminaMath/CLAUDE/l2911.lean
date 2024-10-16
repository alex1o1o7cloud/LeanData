import Mathlib

namespace NUMINAMATH_CALUDE_circle_area_half_radius_l2911_291137

/-- The area of a circle with radius 1/2 is π/4 -/
theorem circle_area_half_radius : 
  let r : ℚ := 1/2
  π * r^2 = π/4 := by sorry

end NUMINAMATH_CALUDE_circle_area_half_radius_l2911_291137


namespace NUMINAMATH_CALUDE_k_value_theorem_l2911_291126

theorem k_value_theorem (k : ℝ) : 
  (∀ x : ℝ, x^2 + k*x + 1 = (x + 1)^2) → k = 2 := by
sorry

end NUMINAMATH_CALUDE_k_value_theorem_l2911_291126


namespace NUMINAMATH_CALUDE_quadratic_properties_l2911_291170

/-- A quadratic function f(x) = ax² + bx + c with specific properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_properties (a b c d : ℝ) :
  (∀ x, QuadraticFunction a b c x = a * x^2 + b * x + c) →
  QuadraticFunction a b c 0 = 3 →
  QuadraticFunction a b c (-1/2) = 0 →
  QuadraticFunction a b c 3 = 0 →
  (∃ x, QuadraticFunction a b c x = x + d ∧ 
        ∀ y, y ≠ x → QuadraticFunction a b c y > y + d) →
  a = -2 ∧ b = 5 ∧ c = 3 ∧ d = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2911_291170


namespace NUMINAMATH_CALUDE_can_determine_native_types_l2911_291175

/-- Represents the type of native: Knight or Liar -/
inductive NativeType
| Knight
| Liar

/-- Represents a native on the island -/
structure Native where
  type : NativeType
  leftNeighborAge : ℕ
  rightNeighborAge : ℕ

/-- The circle of natives -/
def NativeCircle := Vector Native 50

/-- Represents the statements made by a native -/
structure Statement where
  declaredLeftAge : ℕ
  declaredRightAge : ℕ

/-- Function to get the statements of all natives -/
def getAllStatements (circle : NativeCircle) : Vector Statement 50 := sorry

/-- Predicate to check if a native's statement is consistent with their type -/
def isConsistentStatement (native : Native) (statement : Statement) : Prop :=
  match native.type with
  | NativeType.Knight => 
      statement.declaredLeftAge = native.leftNeighborAge ∧ 
      statement.declaredRightAge = native.rightNeighborAge
  | NativeType.Liar => 
      (statement.declaredLeftAge = native.leftNeighborAge + 1 ∧ 
       statement.declaredRightAge = native.rightNeighborAge - 1) ∨
      (statement.declaredLeftAge = native.leftNeighborAge - 1 ∧ 
       statement.declaredRightAge = native.rightNeighborAge + 1)

/-- Main theorem: It's always possible to determine the identity of each native -/
theorem can_determine_native_types (circle : NativeCircle) :
  ∃ (determinedTypes : Vector NativeType 50),
    ∀ (i : Fin 50), 
      (circle.get i).type = determinedTypes.get i ∧
      isConsistentStatement (circle.get i) ((getAllStatements circle).get i) :=
sorry

end NUMINAMATH_CALUDE_can_determine_native_types_l2911_291175


namespace NUMINAMATH_CALUDE_isosceles_iff_equal_angle_bisectors_l2911_291178

/-- Given a triangle with sides a, b, c, and angle bisectors l_α and l_β, 
    prove that the triangle is isosceles (a = b) if and only if l_α = l_β -/
theorem isosceles_iff_equal_angle_bisectors 
  (a b c : ℝ) 
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c)
  (l_α : ℝ := (1 / (b + c)) * Real.sqrt (b * c * ((b + c)^2 - a^2)))
  (l_β : ℝ := (1 / (c + a)) * Real.sqrt (c * a * ((c + a)^2 - b^2))) :
  a = b ↔ l_α = l_β := by
  sorry

end NUMINAMATH_CALUDE_isosceles_iff_equal_angle_bisectors_l2911_291178


namespace NUMINAMATH_CALUDE_base6_division_l2911_291144

/-- Converts a base 6 number to base 10 --/
def toBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (6 ^ i)) 0

/-- Converts a base 10 number to base 6 --/
def toBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
  aux n []

/-- The quotient of 2134₆ divided by 14₆ is equal to 81₆ in base 6 --/
theorem base6_division :
  toBase6 (toBase10 [4, 3, 1, 2] / toBase10 [4, 1]) = [1, 8] := by
  sorry

end NUMINAMATH_CALUDE_base6_division_l2911_291144


namespace NUMINAMATH_CALUDE_cylinder_radii_ratio_l2911_291176

/-- Given two cylinders of the same height, this theorem proves that if their volumes are 40 cc
    and 360 cc respectively, then the ratio of their radii is 1:3. -/
theorem cylinder_radii_ratio (h : ℝ) (r₁ r₂ : ℝ) 
  (h_pos : h > 0) (r₁_pos : r₁ > 0) (r₂_pos : r₂ > 0) :
  π * r₁^2 * h = 40 → π * r₂^2 * h = 360 → r₁ / r₂ = 1 / 3 := by
  sorry

#check cylinder_radii_ratio

end NUMINAMATH_CALUDE_cylinder_radii_ratio_l2911_291176


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2911_291109

theorem inequality_solution_range (a : ℝ) : 
  ((3 - a) * (3 + 2*a - 1)^2 * (3 - 3*a) ≤ 0) →
  (a = -1 ∨ (1 ≤ a ∧ a ≤ 3)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2911_291109


namespace NUMINAMATH_CALUDE_ufo_convention_attendees_l2911_291103

/-- UFO Convention Attendees Problem -/
theorem ufo_convention_attendees :
  ∀ (male_attendees female_attendees : ℕ),
  male_attendees = 62 →
  male_attendees = female_attendees + 4 →
  male_attendees + female_attendees = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_ufo_convention_attendees_l2911_291103


namespace NUMINAMATH_CALUDE_art_dealer_earnings_l2911_291101

/-- Calculates the total money made from selling etchings -/
def total_money_made (total_etchings : ℕ) (first_group_count : ℕ) (first_group_price : ℕ) (second_group_price : ℕ) : ℕ :=
  let second_group_count := total_etchings - first_group_count
  (first_group_count * first_group_price) + (second_group_count * second_group_price)

/-- Proves that the art dealer made $630 from selling the etchings -/
theorem art_dealer_earnings : total_money_made 16 9 35 45 = 630 := by
  sorry

end NUMINAMATH_CALUDE_art_dealer_earnings_l2911_291101


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_minimum_a_for_f_geq_two_l2911_291182

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a*x - 1| + |x + 2|

-- Part I
theorem solution_set_when_a_is_one :
  ∀ x : ℝ, f 1 x ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2 :=
sorry

-- Part II
theorem minimum_a_for_f_geq_two :
  (∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, f a x ≥ 2) ∧
  (∀ ε > 0, ∃ a x : ℝ, 0 < a ∧ a < 1/2 + ε ∧ f a x < 2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_minimum_a_for_f_geq_two_l2911_291182


namespace NUMINAMATH_CALUDE_paint_room_combinations_l2911_291162

theorem paint_room_combinations (n : ℕ) (k : ℕ) (h1 : n = 9) (h2 : k = 2) :
  (Nat.choose n k) * k.factorial = 72 := by
  sorry

end NUMINAMATH_CALUDE_paint_room_combinations_l2911_291162


namespace NUMINAMATH_CALUDE_percentage_calculation_l2911_291171

theorem percentage_calculation (x : ℝ) (h : 0.2 * x = 1000) : 1.2 * x = 6000 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2911_291171


namespace NUMINAMATH_CALUDE_fruit_bowl_problem_l2911_291189

theorem fruit_bowl_problem (initial_oranges : ℕ) : 
  (14 : ℝ) / ((14 : ℝ) + (initial_oranges - 19)) = 0.7 → 
  initial_oranges = 25 := by
  sorry

end NUMINAMATH_CALUDE_fruit_bowl_problem_l2911_291189


namespace NUMINAMATH_CALUDE_cyclic_identity_l2911_291196

theorem cyclic_identity (a b c : ℝ) : 
  a * (a - c)^2 + b * (b - c)^2 - (a - c) * (b - c) * (a + b - c) = 
  b * (b - a)^2 + c * (c - a)^2 - (b - a) * (c - a) * (b + c - a) ∧
  b * (b - a)^2 + c * (c - a)^2 - (b - a) * (c - a) * (b + c - a) = 
  c * (c - b)^2 + a * (a - b)^2 - (c - b) * (a - b) * (c + a - b) ∧
  a * (a - c)^2 + b * (b - c)^2 - (a - c) * (b - c) * (a + b - c) = 
  a^3 + b^3 + c^3 - (a^2 * b + a * b^2 + b^2 * c + b * c^2 + c^2 * a + a^2 * c) + 3 * a * b * c :=
by sorry

end NUMINAMATH_CALUDE_cyclic_identity_l2911_291196


namespace NUMINAMATH_CALUDE_quadratic_root_property_l2911_291142

theorem quadratic_root_property (a : ℝ) : 
  a^2 + 3*a - 1010 = 0 → 2*a^2 + 6*a + 4 = 2024 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l2911_291142


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l2911_291163

theorem shaded_area_calculation (square_side : ℝ) (triangle1_base triangle1_height : ℝ) (triangle2_base triangle2_height : ℝ) :
  square_side = 40 →
  triangle1_base = 15 →
  triangle1_height = 20 →
  triangle2_base = 15 →
  triangle2_height = 10 →
  square_side * square_side - (0.5 * triangle1_base * triangle1_height + 0.5 * triangle2_base * triangle2_height) = 1375 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l2911_291163


namespace NUMINAMATH_CALUDE_max_students_distribution_l2911_291166

theorem max_students_distribution (pens pencils : ℕ) (h1 : pens = 1230) (h2 : pencils = 920) :
  (∃ (students : ℕ), students > 0 ∧ 
   pens % students = 0 ∧ 
   pencils % students = 0 ∧
   ∀ (n : ℕ), n > students → (pens % n ≠ 0 ∨ pencils % n ≠ 0)) ↔ 
  (Nat.gcd pens pencils = 10) :=
sorry

end NUMINAMATH_CALUDE_max_students_distribution_l2911_291166


namespace NUMINAMATH_CALUDE_neighborhood_cable_cost_l2911_291107

/-- Calculates the total cost of power cable for a neighborhood with the given specifications. -/
theorem neighborhood_cable_cost
  (ew_streets : ℕ)
  (ew_length : ℝ)
  (ns_streets : ℕ)
  (ns_length : ℝ)
  (cable_per_mile : ℝ)
  (cable_cost : ℝ)
  (h1 : ew_streets = 18)
  (h2 : ew_length = 2)
  (h3 : ns_streets = 10)
  (h4 : ns_length = 4)
  (h5 : cable_per_mile = 5)
  (h6 : cable_cost = 2000) :
  (ew_streets * ew_length + ns_streets * ns_length) * cable_per_mile * cable_cost = 760000 := by
  sorry

end NUMINAMATH_CALUDE_neighborhood_cable_cost_l2911_291107


namespace NUMINAMATH_CALUDE_draw_inferior_pencil_probability_l2911_291172

/-- The total number of pencils -/
def total_pencils : ℕ := 10

/-- The number of good quality pencils -/
def good_pencils : ℕ := 8

/-- The number of inferior quality pencils -/
def inferior_pencils : ℕ := 2

/-- The number of pencils drawn -/
def drawn_pencils : ℕ := 2

/-- The probability of drawing at least one inferior pencil -/
def prob_at_least_one_inferior : ℚ := 17 / 45

theorem draw_inferior_pencil_probability :
  (1 : ℚ) - (Nat.choose good_pencils drawn_pencils : ℚ) / (Nat.choose total_pencils drawn_pencils : ℚ) = prob_at_least_one_inferior :=
sorry

end NUMINAMATH_CALUDE_draw_inferior_pencil_probability_l2911_291172


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_magnitude_l2911_291167

/-- Two parallel planar vectors have a specific sum magnitude -/
theorem parallel_vectors_sum_magnitude :
  ∀ (x : ℝ),
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![x, -3]
  (∃ (k : ℝ), ∀ (i : Fin 2), a i = k * b i) →
  ‖(a + b)‖ = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_magnitude_l2911_291167


namespace NUMINAMATH_CALUDE_max_area_rectangular_fence_l2911_291188

/-- Represents a rectangular fence with given constraints -/
structure RectangularFence where
  length : ℝ
  width : ℝ
  perimeter_constraint : length + width = 200
  length_constraint : length ≥ 100
  width_constraint : width ≥ 50

/-- Calculates the area of a rectangular fence -/
def area (fence : RectangularFence) : ℝ :=
  fence.length * fence.width

/-- Theorem stating the maximum area of the rectangular fence -/
theorem max_area_rectangular_fence :
  ∃ (fence : RectangularFence), ∀ (other : RectangularFence), area fence ≥ area other ∧ area fence = 10000 := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangular_fence_l2911_291188


namespace NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l2911_291102

theorem cos_x_plus_2y_equals_one 
  (x y : ℝ) 
  (a : ℝ) 
  (hx : x ∈ Set.Icc (-Real.pi/4) (Real.pi/4))
  (hy : y ∈ Set.Icc (-Real.pi/4) (Real.pi/4))
  (eq1 : x^3 + Real.sin x - 2*a = 0)
  (eq2 : 4*y^3 + Real.sin y * Real.cos y + a = 0) :
  Real.cos (x + 2*y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l2911_291102


namespace NUMINAMATH_CALUDE_reasoning_is_inductive_l2911_291193

/-- Represents different types of reasoning methods -/
inductive ReasoningMethod
  | Analogical
  | Inductive
  | Deductive
  | Analytical

/-- Represents a metal -/
structure Metal where
  name : String

/-- Represents the property of conducting electricity -/
def conductsElectricity (m : Metal) : Prop := sorry

/-- The set of metals mentioned in the statement -/
def mentionedMetals : List Metal := [
  { name := "Gold" },
  { name := "Silver" },
  { name := "Copper" },
  { name := "Iron" }
]

/-- The statement that all mentioned metals conduct electricity -/
def allMentionedMetalsConduct : Prop :=
  ∀ m ∈ mentionedMetals, conductsElectricity m

/-- The conclusion that all metals conduct electricity -/
def allMetalsConduct : Prop :=
  ∀ m : Metal, conductsElectricity m

/-- The reasoning method used in the given statement -/
def reasoningMethodUsed : ReasoningMethod := sorry

/-- Theorem stating that the reasoning method used is inductive -/
theorem reasoning_is_inductive :
  allMentionedMetalsConduct →
  reasoningMethodUsed = ReasoningMethod.Inductive :=
sorry

end NUMINAMATH_CALUDE_reasoning_is_inductive_l2911_291193


namespace NUMINAMATH_CALUDE_mrs_hilts_snow_amount_l2911_291160

def snow_at_mrs_hilts_house : ℕ := 29
def snow_at_brecknock_school : ℕ := 17

theorem mrs_hilts_snow_amount : snow_at_mrs_hilts_house = 29 := by sorry

end NUMINAMATH_CALUDE_mrs_hilts_snow_amount_l2911_291160


namespace NUMINAMATH_CALUDE_lemonade_second_intermission_l2911_291190

theorem lemonade_second_intermission 
  (total : ℝ) 
  (first : ℝ) 
  (third : ℝ) 
  (h1 : total = 0.9166666666666666) 
  (h2 : first = 0.25) 
  (h3 : third = 0.25) : 
  total - (first + third) = 0.4166666666666666 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_second_intermission_l2911_291190


namespace NUMINAMATH_CALUDE_simplify_square_roots_l2911_291105

theorem simplify_square_roots : 
  2 * Real.sqrt 12 - Real.sqrt 27 - Real.sqrt 3 * Real.sqrt (1/9) = (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l2911_291105


namespace NUMINAMATH_CALUDE_complement_B_intersect_A_range_of_a_l2911_291165

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x - 18 ≥ 0}
def B : Set ℝ := {x | (x+5)/(x-14) ≤ 0}
def C (a : ℝ) : Set ℝ := {x | 2*a < x ∧ x < a+1}

-- Theorem for part (1)
theorem complement_B_intersect_A : 
  (Set.univ \ B) ∩ A = Set.Iic (-5) ∪ Set.Ici 14 := by sorry

-- Theorem for part (2)
theorem range_of_a (a : ℝ) : 
  (B ∩ C a = C a) ↔ a ≥ -5/2 := by sorry

end NUMINAMATH_CALUDE_complement_B_intersect_A_range_of_a_l2911_291165


namespace NUMINAMATH_CALUDE_income_calculation_l2911_291199

/-- Given a person's income and expenditure ratio, and their savings amount, 
    calculate their income. -/
theorem income_calculation (income_ratio : ℕ) (expenditure_ratio : ℕ) (savings : ℕ) : 
  income_ratio = 5 → expenditure_ratio = 4 → savings = 3200 → 
  income_ratio * savings / (income_ratio - expenditure_ratio) = 16000 := by
  sorry

#check income_calculation

end NUMINAMATH_CALUDE_income_calculation_l2911_291199


namespace NUMINAMATH_CALUDE_roses_per_bush_calculation_l2911_291133

/-- The number of rose petals needed to make one ounce of perfume -/
def petals_per_ounce : ℕ := 320

/-- The number of petals produced by each rose -/
def petals_per_rose : ℕ := 8

/-- The number of bushes harvested -/
def bushes_harvested : ℕ := 800

/-- The number of bottles of perfume to be made -/
def bottles_to_make : ℕ := 20

/-- The number of ounces in each bottle of perfume -/
def ounces_per_bottle : ℕ := 12

/-- The number of roses per bush -/
def roses_per_bush : ℕ := 12

theorem roses_per_bush_calculation :
  roses_per_bush * bushes_harvested * petals_per_rose =
  bottles_to_make * ounces_per_bottle * petals_per_ounce :=
by sorry

end NUMINAMATH_CALUDE_roses_per_bush_calculation_l2911_291133


namespace NUMINAMATH_CALUDE_lock_min_moves_l2911_291100

/-- Represents a combination lock with n discs, each having d digits -/
structure CombinationLock (n : ℕ) (d : ℕ) where
  discs : Fin n → Fin d

/-- Represents a move on the lock -/
def move (lock : CombinationLock n d) (disc : Fin n) (direction : Bool) : CombinationLock n d :=
  sorry

/-- Checks if a combination is valid (for part b) -/
def is_valid_combination (lock : CombinationLock n d) : Bool :=
  sorry

/-- The number of moves required to ensure finding the correct combination -/
def min_moves (n : ℕ) (d : ℕ) (initial : CombinationLock n d) (valid : CombinationLock n d → Bool) : ℕ :=
  sorry

theorem lock_min_moves :
  let n : ℕ := 6
  let d : ℕ := 10
  let initial : CombinationLock n d := sorry
  let valid_a : CombinationLock n d → Bool := λ _ => true
  let valid_b : CombinationLock n d → Bool := is_valid_combination
  (∀ (i : Fin n), initial.discs i = 0) →
  min_moves n d initial valid_a = 999998 ∧
  min_moves n d initial valid_b = 999998 :=
sorry

end NUMINAMATH_CALUDE_lock_min_moves_l2911_291100


namespace NUMINAMATH_CALUDE_rectangle_100_101_diagonal_segments_l2911_291127

/-- The number of segments a diagonal is divided into by grid lines in a rectangle -/
def diagonal_segments (width : ℕ) (height : ℕ) : ℕ :=
  width + height - Nat.gcd width height

/-- Theorem: In a 100 × 101 rectangle, the diagonal is divided into 200 segments by grid lines -/
theorem rectangle_100_101_diagonal_segments :
  diagonal_segments 100 101 = 200 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_100_101_diagonal_segments_l2911_291127


namespace NUMINAMATH_CALUDE_smallest_common_multiple_9_15_gt_50_l2911_291168

theorem smallest_common_multiple_9_15_gt_50 : ∃ n : ℕ, n = 90 ∧ 
  (∀ m : ℕ, m < n → (m % 9 = 0 ∧ m % 15 = 0 → m ≤ 50)) ∧
  n % 9 = 0 ∧ n % 15 = 0 ∧ n > 50 := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_9_15_gt_50_l2911_291168


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l2911_291153

theorem quadratic_roots_sum (a b : ℝ) (ha : a^2 - 8*a + 5 = 0) (hb : b^2 - 8*b + 5 = 0) (hab : a ≠ b) :
  (b - 1) / (a - 1) + (a - 1) / (b - 1) = -20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l2911_291153


namespace NUMINAMATH_CALUDE_select_and_arrange_five_three_unique_descending_arrangement_select_three_from_five_descending_l2911_291119

/-- The number of ways to select and arrange 3 people from 5 in descending height order -/
def select_and_arrange (n m : ℕ) : ℕ :=
  Nat.choose n m

theorem select_and_arrange_five_three :
  select_and_arrange 5 3 = Nat.choose 5 3 := by
  sorry

/-- The number of ways to arrange 3 people in descending height order -/
def arrange_descending (k : ℕ) : ℕ := 1

theorem unique_descending_arrangement (k : ℕ) :
  arrange_descending k = 1 := by
  sorry

/-- The main theorem: selecting and arranging 3 from 5 equals C(5,3) -/
theorem select_three_from_five_descending :
  select_and_arrange 5 3 = Nat.choose 5 3 := by
  sorry

end NUMINAMATH_CALUDE_select_and_arrange_five_three_unique_descending_arrangement_select_three_from_five_descending_l2911_291119


namespace NUMINAMATH_CALUDE_logarithm_expression_equality_l2911_291151

-- Define the logarithm function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the logarithm with arbitrary base
noncomputable def log (b x : ℝ) : ℝ := Real.log x / Real.log b

-- State the theorem
theorem logarithm_expression_equality :
  (lg 5)^2 + lg 2 * lg 50 - log 8 9 * log 27 32 = -1/9 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equality_l2911_291151


namespace NUMINAMATH_CALUDE_all_zeros_not_pronounced_l2911_291159

/-- Represents a natural number in decimal notation --/
def DecimalNumber : Type := List Nat

/-- Rules for reading integers --/
structure ReadingRules where
  readHighestToLowest : Bool
  skipEndZeros : Bool
  readConsecutiveZerosAsOne : Bool

/-- Function to determine if a digit should be pronounced --/
def shouldPronounce (rules : ReadingRules) (num : DecimalNumber) (index : Nat) : Bool :=
  sorry

/-- The number 3,406,000 in decimal notation --/
def number : DecimalNumber := [3, 4, 0, 6, 0, 0, 0]

/-- The rules for reading integers as described in the problem --/
def integerReadingRules : ReadingRules := {
  readHighestToLowest := true,
  skipEndZeros := true,
  readConsecutiveZerosAsOne := true
}

/-- Theorem stating that all zeros in 3,406,000 are not pronounced --/
theorem all_zeros_not_pronounced : 
  ∀ i, i ∈ [2, 4, 5, 6] → ¬(shouldPronounce integerReadingRules number i) :=
sorry

end NUMINAMATH_CALUDE_all_zeros_not_pronounced_l2911_291159


namespace NUMINAMATH_CALUDE_eg_length_l2911_291129

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_valid_quadrilateral (q : Quadrilateral) : Prop :=
  let (ex, ey) := q.E
  let (fx, fy) := q.F
  let (gx, gy) := q.G
  let (hx, hy) := q.H
  -- EF = 6
  (ex - fx)^2 + (ey - fy)^2 = 36 ∧
  -- FG = 18
  (fx - gx)^2 + (fy - gy)^2 = 324 ∧
  -- GH = 6
  (gx - hx)^2 + (gy - hy)^2 = 36 ∧
  -- HE = 10
  (hx - ex)^2 + (hy - ey)^2 = 100 ∧
  -- Angle EFG is a right angle
  (ex - fx) * (gx - fx) + (ey - fy) * (gy - fy) = 0

-- Theorem statement
theorem eg_length (q : Quadrilateral) (h : is_valid_quadrilateral q) :
  let (ex, ey) := q.E
  let (gx, gy) := q.G
  (ex - gx)^2 + (ey - gy)^2 = 360 := by
  sorry

end NUMINAMATH_CALUDE_eg_length_l2911_291129


namespace NUMINAMATH_CALUDE_continuity_at_three_l2911_291120

def f (x : ℝ) : ℝ := 2 * x^2 - 4

theorem continuity_at_three :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 3| < δ → |f x - f 3| < ε := by
  sorry

end NUMINAMATH_CALUDE_continuity_at_three_l2911_291120


namespace NUMINAMATH_CALUDE_polynomial_expansion_equality_l2911_291115

theorem polynomial_expansion_equality (p q : ℝ) : 
  p > 0 ∧ q > 0 ∧ p + q = 2 ∧ 
  (55 * p^9 * q^2 = 165 * p^8 * q^3) → 
  p = 3/2 :=
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_equality_l2911_291115


namespace NUMINAMATH_CALUDE_cookie_circle_radius_l2911_291118

theorem cookie_circle_radius (x y : ℝ) :
  (∃ (h k r : ℝ), ∀ x y : ℝ, x^2 + y^2 - 12*x + 16*y + 64 = 0 ↔ (x - h)^2 + (y - k)^2 = r^2) →
  (∃ (r : ℝ), r = 6 ∧ ∀ x y : ℝ, x^2 + y^2 - 12*x + 16*y + 64 = 0 ↔ (x - 6)^2 + (y + 8)^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_cookie_circle_radius_l2911_291118


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2911_291149

theorem arithmetic_sequence_sum (k : ℕ) : 
  let a : ℕ → ℕ := λ n => 1 + 2 * (n - 1)
  let S : ℕ → ℕ := λ n => n * (2 * a 1 + (n - 1) * 2) / 2
  S (k + 2) - S k = 24 → k = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2911_291149


namespace NUMINAMATH_CALUDE_eq_length_is_40_l2911_291194

/-- Represents a trapezoid with a circle inscribed in it -/
structure InscribedCircleTrapezoid where
  -- Lengths of the trapezoid sides
  ef : ℝ
  fg : ℝ
  gh : ℝ
  he : ℝ
  -- Ensure EF is parallel to GH (represented by their lengths being different)
  ef_parallel_gh : ef ≠ gh
  -- Circle center Q is on EF
  eq : ℝ
  -- Circle is tangent to FG and HE (implicitly assumed by the structure)

/-- The specific trapezoid from the problem -/
def problemTrapezoid : InscribedCircleTrapezoid where
  ef := 100
  fg := 60
  gh := 22
  he := 80
  ef_parallel_gh := by norm_num
  eq := 40  -- This is what we want to prove

/-- The main theorem: EQ = 40 in the given trapezoid -/
theorem eq_length_is_40 : problemTrapezoid.eq = 40 := by
  sorry

#eval problemTrapezoid.eq  -- Should output 40

end NUMINAMATH_CALUDE_eq_length_is_40_l2911_291194


namespace NUMINAMATH_CALUDE_cookie_distribution_l2911_291187

/-- Represents the number of cookie boxes in Sonny's distribution problem -/
structure CookieBoxes where
  total : ℕ
  tobrother : ℕ
  tocousin : ℕ
  kept : ℕ
  tosister : ℕ

/-- Theorem stating the relationship between the number of cookie boxes -/
theorem cookie_distribution (c : CookieBoxes) 
  (h1 : c.total = 45)
  (h2 : c.tobrother = 12)
  (h3 : c.tocousin = 7)
  (h4 : c.kept = 17) :
  c.tosister = c.total - (c.tobrother + c.tocousin + c.kept) :=
by sorry

end NUMINAMATH_CALUDE_cookie_distribution_l2911_291187


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2911_291156

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x ∈ Set.Icc m (m + 1), x^2 + m*x - 1 < 0) → 
  m ∈ Set.Ioo (-Real.sqrt 2 / 2) 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2911_291156


namespace NUMINAMATH_CALUDE_consecutive_binomial_ratio_sum_n_plus_k_l2911_291198

theorem consecutive_binomial_ratio (n k : ℕ) : 
  (Nat.choose n k : ℚ) / (Nat.choose n (k + 1) : ℚ) = 1 / 3 ∧
  (Nat.choose n (k + 1) : ℚ) / (Nat.choose n (k + 2) : ℚ) = 3 / 6 →
  n = 11 ∧ k = 2 :=
by sorry

theorem sum_n_plus_k (n k : ℕ) :
  (Nat.choose n k : ℚ) / (Nat.choose n (k + 1) : ℚ) = 1 / 3 ∧
  (Nat.choose n (k + 1) : ℚ) / (Nat.choose n (k + 2) : ℚ) = 3 / 6 →
  n + k = 13 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_binomial_ratio_sum_n_plus_k_l2911_291198


namespace NUMINAMATH_CALUDE_noah_jelly_beans_l2911_291130

-- Define the total number of jelly beans
def total_jelly_beans : ℝ := 600

-- Define the percentages for Thomas and Sarah
def thomas_percentage : ℝ := 0.06
def sarah_percentage : ℝ := 0.10

-- Define the ratio for Barry, Emmanuel, and Miguel
def barry_ratio : ℝ := 4
def emmanuel_ratio : ℝ := 5
def miguel_ratio : ℝ := 6

-- Define the percentages for Chloe and Noah
def chloe_percentage : ℝ := 0.40
def noah_percentage : ℝ := 0.30

-- Theorem to prove
theorem noah_jelly_beans :
  let thomas_share := total_jelly_beans * thomas_percentage
  let sarah_share := total_jelly_beans * sarah_percentage
  let remaining_jelly_beans := total_jelly_beans - (thomas_share + sarah_share)
  let total_ratio := barry_ratio + emmanuel_ratio + miguel_ratio
  let emmanuel_share := (emmanuel_ratio / total_ratio) * remaining_jelly_beans
  let noah_share := emmanuel_share * noah_percentage
  noah_share = 50.4 := by
  sorry

end NUMINAMATH_CALUDE_noah_jelly_beans_l2911_291130


namespace NUMINAMATH_CALUDE_ellipse_and_line_properties_l2911_291131

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0
  eccentricity : ℝ
  h_eccentricity : eccentricity = Real.sqrt 6 / 3
  triangle_area : ℝ
  h_triangle_area : triangle_area = 5 * Real.sqrt 2 / 3

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) : ℝ × ℝ → Prop :=
  fun (x, y) ↦ x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The moving line that intersects the ellipse -/
def moving_line (k : ℝ) : ℝ → ℝ :=
  fun x ↦ k * (x + 1)

/-- Theorem stating the properties of the ellipse and the intersecting line -/
theorem ellipse_and_line_properties (e : Ellipse) :
  (∀ x y, ellipse_equation e (x, y) ↔ x^2 / 5 + y^2 / (5/3) = 1) ∧
  (∃ k : ℝ, k = Real.sqrt 3 / 3 ∨ k = -Real.sqrt 3 / 3 ∧
    ∃ x₁ x₂ : ℝ, 
      ellipse_equation e (x₁, moving_line k x₁) ∧
      ellipse_equation e (x₂, moving_line k x₂) ∧
      (x₁ + x₂) / 2 = -1/2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_line_properties_l2911_291131


namespace NUMINAMATH_CALUDE_odd_function_property_l2911_291155

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Theorem statement
theorem odd_function_property (f : ℝ → ℝ) (a : ℝ) 
  (h_odd : OddFunction f) (h_fa : f a = 11) : f (-a) = -11 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l2911_291155


namespace NUMINAMATH_CALUDE_initial_amount_was_21_l2911_291143

/-- The initial amount of money in the cookie jar -/
def initial_amount : ℕ := sorry

/-- The amount Doris spent -/
def doris_spent : ℕ := 6

/-- The amount Martha spent -/
def martha_spent : ℕ := doris_spent / 2

/-- The amount left in the cookie jar after spending -/
def amount_left : ℕ := 12

/-- Theorem stating that the initial amount in the cookie jar was 21 dollars -/
theorem initial_amount_was_21 : initial_amount = 21 := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_was_21_l2911_291143


namespace NUMINAMATH_CALUDE_circle_C_and_min_chord_length_l2911_291157

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 8)^2 + y^2 = 5

-- Define the lines
def line_1 (x y : ℝ) : Prop := y = 2*x - 21
def line_2 (x y : ℝ) : Prop := y = 2*x - 11
def center_line (x y : ℝ) : Prop := x + y = 8

-- Define the intersecting line
def line_l (x y a : ℝ) : Prop := 2*x + a*y + 6*a = a*x + 14

-- Theorem statement
theorem circle_C_and_min_chord_length :
  ∃ (x₀ y₀ : ℝ),
    -- Center of C lies on the center line
    center_line x₀ y₀ ∧
    -- C is tangent to line_1 and line_2
    (∃ (x₁ y₁ : ℝ), circle_C x₁ y₁ ∧ line_1 x₁ y₁) ∧
    (∃ (x₂ y₂ : ℝ), circle_C x₂ y₂ ∧ line_2 x₂ y₂) ∧
    -- The equation of circle C
    (∀ (x y : ℝ), circle_C x y ↔ (x - 8)^2 + y^2 = 5) ∧
    -- Minimum length of chord MN
    (∀ (a : ℝ),
      (∃ (x₁ y₁ x₂ y₂ : ℝ),
        x₁ ≠ x₂ ∧ circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
        line_l x₁ y₁ a ∧ line_l x₂ y₂ a) →
      ∃ (m n : ℝ), m ≤ (x₁ - x₂)^2 + (y₁ - y₂)^2 ∧ m = 12 ∧ n^2 = m) :=
sorry

end NUMINAMATH_CALUDE_circle_C_and_min_chord_length_l2911_291157


namespace NUMINAMATH_CALUDE_miguels_wall_paint_area_l2911_291152

/-- The area to be painted on a wall with given dimensions and a window -/
def area_to_paint (wall_height wall_length window_side : ℝ) : ℝ :=
  wall_height * wall_length - window_side * window_side

/-- Theorem stating the area to be painted for Miguel's wall -/
theorem miguels_wall_paint_area :
  area_to_paint 10 15 3 = 141 := by
  sorry

end NUMINAMATH_CALUDE_miguels_wall_paint_area_l2911_291152


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2911_291191

/-- The equation of a hyperbola sharing foci with a given ellipse and passing through a specific point -/
theorem hyperbola_equation (x y : ℝ) : 
  (∃ (a b : ℝ), (x^2 / 9 + y^2 / 5 = 1) ∧ 
   (x^2 / a^2 - y^2 / b^2 = 1) ∧
   (3^2 / a^2 - 2 / b^2 = 1) ∧
   (a^2 + b^2 = 4)) →
  (x^2 / 3 - y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2911_291191


namespace NUMINAMATH_CALUDE_equation_with_positive_root_l2911_291128

theorem equation_with_positive_root (x m : ℝ) : 
  ((x - 2) / (x + 1) = m / (x + 1) ∧ x > 0) → m = -3 :=
by sorry

end NUMINAMATH_CALUDE_equation_with_positive_root_l2911_291128


namespace NUMINAMATH_CALUDE_maria_fair_spending_l2911_291141

def fair_spending (initial_amount spent_on_rides discounted_ride_cost discount_percent
                   borrowed won food_cost found_money lent_money final_amount : ℚ) : Prop :=
  let discounted_ride_spending := discounted_ride_cost * (1 - discount_percent / 100)
  let net_amount := initial_amount - spent_on_rides - discounted_ride_spending + borrowed + won - food_cost + found_money - lent_money
  net_amount - final_amount = 41

theorem maria_fair_spending :
  fair_spending 87 25 4 25 15 10 12 5 20 16 := by sorry

end NUMINAMATH_CALUDE_maria_fair_spending_l2911_291141


namespace NUMINAMATH_CALUDE_arithmetic_sequence_shared_prime_factor_l2911_291110

theorem arithmetic_sequence_shared_prime_factor (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  ∃ (p : ℕ) (hp : Prime p), ∀ n : ℕ, ∃ k ≥ n, p ∣ (a * k + b) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_shared_prime_factor_l2911_291110


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l2911_291122

theorem rectangle_dimension_change (L B : ℝ) (h : L > 0 ∧ B > 0) :
  let L' := L * (1 + 30 / 100)
  let B' := B * (1 - 20 / 100)
  L' * B' = (L * B) * (1 + 4.0000000000000036 / 100) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l2911_291122


namespace NUMINAMATH_CALUDE_continued_fraction_sum_l2911_291181

theorem continued_fraction_sum (x y z : ℕ+) :
  (30 : ℚ) / 7 = x + 1 / (y + 1 / z) →
  x + y + z = 9 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_sum_l2911_291181


namespace NUMINAMATH_CALUDE_problem_statement_l2911_291136

theorem problem_statement (x y : ℝ) 
  (h1 : 2 * x - y = 1) 
  (h2 : x * y = 2) : 
  4 * x^3 * y - 4 * x^2 * y^2 + x * y^3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2911_291136


namespace NUMINAMATH_CALUDE_ricciana_long_jump_l2911_291132

/-- Ricciana's long jump problem -/
theorem ricciana_long_jump (R : ℕ) : R = 20 :=
  let ricciana_jump := 4
  let margarita_run := 18
  let margarita_jump := 2 * ricciana_jump - 1
  let ricciana_total := R + ricciana_jump
  let margarita_total := margarita_run + margarita_jump
  have h1 : margarita_total = ricciana_total + 1 := by sorry
  sorry

#check ricciana_long_jump

end NUMINAMATH_CALUDE_ricciana_long_jump_l2911_291132


namespace NUMINAMATH_CALUDE_no_strictly_monotonic_pair_l2911_291117

theorem no_strictly_monotonic_pair :
  ¬∃ (f g : ℕ → ℕ),
    (∀ x y, x < y → f x < f y) ∧
    (∀ x y, x < y → g x < g y) ∧
    (∀ n, f (g (g n)) < g (f n)) :=
by sorry

end NUMINAMATH_CALUDE_no_strictly_monotonic_pair_l2911_291117


namespace NUMINAMATH_CALUDE_mulch_cost_theorem_l2911_291147

/-- The cost of mulch in dollars per cubic foot -/
def mulch_cost_per_cubic_foot : ℝ := 5

/-- The conversion factor from cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet : ℝ := 27

/-- The volume of mulch in cubic yards -/
def mulch_volume_cubic_yards : ℝ := 8

/-- Theorem: The cost of 8 cubic yards of mulch is 1080 dollars -/
theorem mulch_cost_theorem :
  mulch_volume_cubic_yards * cubic_yards_to_cubic_feet * mulch_cost_per_cubic_foot = 1080 := by
  sorry

end NUMINAMATH_CALUDE_mulch_cost_theorem_l2911_291147


namespace NUMINAMATH_CALUDE_taya_jenna_meet_l2911_291169

/-- The floor where Taya and Jenna meet -/
def meeting_floor : ℕ := 32

/-- The starting floor -/
def start_floor : ℕ := 22

/-- Time Jenna waits for the elevator (in seconds) -/
def wait_time : ℕ := 120

/-- Time Taya takes to go up one floor (in seconds) -/
def taya_time_per_floor : ℕ := 15

/-- Time the elevator takes to go up one floor (in seconds) -/
def elevator_time_per_floor : ℕ := 3

/-- Theorem stating that Taya and Jenna arrive at the meeting floor at the same time -/
theorem taya_jenna_meet :
  taya_time_per_floor * (meeting_floor - start_floor) =
  wait_time + elevator_time_per_floor * (meeting_floor - start_floor) :=
by sorry

end NUMINAMATH_CALUDE_taya_jenna_meet_l2911_291169


namespace NUMINAMATH_CALUDE_salary_percentage_difference_l2911_291146

theorem salary_percentage_difference (raja_salary : ℝ) (ram_salary : ℝ) :
  ram_salary = raja_salary * 1.25 →
  (raja_salary - ram_salary) / ram_salary = -0.2 := by
sorry

end NUMINAMATH_CALUDE_salary_percentage_difference_l2911_291146


namespace NUMINAMATH_CALUDE_equation_solution_l2911_291179

theorem equation_solution : ∃ x : ℝ, 15 * 2 = 3 + x ∧ x = 27 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2911_291179


namespace NUMINAMATH_CALUDE_G₁_intersects_x_axis_range_of_n_minus_m_plus_a_right_triangle_BNB_l2911_291197

-- Define the parabola G₁
def G₁ (a x : ℝ) : ℝ := x^2 - 2*a*x + a^2 - 4

-- Define the coordinates of point N
def N : ℝ × ℝ := (0, -4)

-- Theorem 1: G₁ intersects x-axis at two points
theorem G₁_intersects_x_axis (a : ℝ) :
  ∃ m n : ℝ, m < n ∧ G₁ a m = 0 ∧ G₁ a n = 0 := by sorry

-- Theorem 2: Range of n - m + a when NA ≥ 5
theorem range_of_n_minus_m_plus_a (a m n : ℝ) :
  m < n → G₁ a m = 0 → G₁ a n = 0 →
  Real.sqrt ((m - N.1)^2 + (N.2)^2) ≥ 5 →
  (n - m + a ≥ 9 ∨ n - m + a ≤ 3) := by sorry

-- Define the parabola G₂ (symmetric to G₁ with respect to A)
def G₂ (a x : ℝ) : ℝ := G₁ a (2*a - 2 - x)

-- Theorem 3: Conditions for right triangle BNB'
theorem right_triangle_BNB' (a : ℝ) :
  (∃ m n b : ℝ, m < n ∧ G₁ a m = 0 ∧ G₁ a n = 0 ∧ G₂ a b = 0 ∧ b ≠ m ∧
   (n - N.1)^2 + N.2^2 + (b - N.1)^2 + N.2^2 = (n - b)^2) ↔
  (a = 2 ∨ a = -2 ∨ a = 6) := by sorry

end NUMINAMATH_CALUDE_G₁_intersects_x_axis_range_of_n_minus_m_plus_a_right_triangle_BNB_l2911_291197


namespace NUMINAMATH_CALUDE_arithmetic_sequence_60th_term_l2911_291134

/-- An arithmetic sequence is defined by its first term and common difference. -/
structure ArithmeticSequence where
  a₁ : ℚ
  d : ℚ

/-- The nth term of an arithmetic sequence. -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  seq.a₁ + (n - 1 : ℚ) * seq.d

theorem arithmetic_sequence_60th_term
  (seq : ArithmeticSequence)
  (h₁ : seq.a₁ = 6)
  (h₁₃ : seq.nthTerm 13 = 32) :
  seq.nthTerm 60 = 803 / 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_60th_term_l2911_291134


namespace NUMINAMATH_CALUDE_coins_per_stack_l2911_291174

theorem coins_per_stack (total_coins : ℕ) (num_stacks : ℕ) (coins_per_stack : ℕ) : 
  total_coins = 15 → num_stacks = 5 → total_coins = num_stacks * coins_per_stack → coins_per_stack = 3 := by
  sorry

end NUMINAMATH_CALUDE_coins_per_stack_l2911_291174


namespace NUMINAMATH_CALUDE_visit_either_not_both_l2911_291186

def probability_chile : ℝ := 0.5
def probability_madagascar : ℝ := 0.5

theorem visit_either_not_both :
  probability_chile + probability_madagascar - 2 * (probability_chile * probability_madagascar) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_visit_either_not_both_l2911_291186


namespace NUMINAMATH_CALUDE_prob_10_or_9_prob_less_than_7_l2911_291192

-- Define the probabilities
def p_10 : ℝ := 0.21
def p_9 : ℝ := 0.23
def p_8 : ℝ := 0.25
def p_7 : ℝ := 0.28

-- Theorem for the first question
theorem prob_10_or_9 : p_10 + p_9 = 0.44 := by sorry

-- Theorem for the second question
theorem prob_less_than_7 : 1 - (p_10 + p_9 + p_8 + p_7) = 0.03 := by sorry

end NUMINAMATH_CALUDE_prob_10_or_9_prob_less_than_7_l2911_291192


namespace NUMINAMATH_CALUDE_first_day_of_month_l2911_291140

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def day_after_n_days (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => next_day (day_after_n_days d n)

theorem first_day_of_month (d : DayOfWeek) :
  day_after_n_days d 27 = DayOfWeek.Tuesday → d = DayOfWeek.Wednesday :=
by sorry

end NUMINAMATH_CALUDE_first_day_of_month_l2911_291140


namespace NUMINAMATH_CALUDE_line_properties_l2911_291183

/-- Represents a line in the form ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the slope angle of a line in degrees --/
def slope_angle (l : Line) : ℝ :=
  sorry

/-- Calculates the y-intercept of a line --/
def y_intercept (l : Line) : ℝ :=
  sorry

/-- The line x + y + 1 = 0 --/
def line : Line :=
  { a := 1, b := 1, c := 1 }

theorem line_properties :
  slope_angle line = 135 ∧ y_intercept line = -1 := by
  sorry

end NUMINAMATH_CALUDE_line_properties_l2911_291183


namespace NUMINAMATH_CALUDE_number_equation_l2911_291161

theorem number_equation (x : ℝ) : (40 / 100) * x = (10 / 100) * 70 → x = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l2911_291161


namespace NUMINAMATH_CALUDE_no_nonzero_solution_l2911_291104

theorem no_nonzero_solution (x y z : ℝ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 →
  (x^2 + x = y^2 - y ∧ 
   y^2 + y = z^2 - z ∧ 
   z^2 + z = x^2 - x) → 
  False :=
sorry

end NUMINAMATH_CALUDE_no_nonzero_solution_l2911_291104


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2911_291106

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1 where a > 0, b > 0,
    and one of its asymptotes is y = √2 x, prove that its eccentricity is √3. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ x : ℝ, y = Real.sqrt 2 * x) →
  Real.sqrt (1 + b^2 / a^2) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2911_291106


namespace NUMINAMATH_CALUDE_m_range_l2911_291154

/-- The function f(x) = x³ - ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x + 2

/-- The function g(x) = f(x) + mx -/
def g (a m : ℝ) (x : ℝ) : ℝ := f a x + m*x

theorem m_range (a m : ℝ) : 
  (∃ x₀, ∀ x, f a x ≤ f a x₀ ∧ f a x₀ = 4) →
  (∃ x₁ ∈ Set.Ioo (-3) (a - 1), ∀ x ∈ Set.Ioo (-3) (a - 1), g a m x₁ ≤ g a m x ∧ g a m x₁ ≤ m - 1) →
  -9 < m ∧ m ≤ -15/4 := by sorry

end NUMINAMATH_CALUDE_m_range_l2911_291154


namespace NUMINAMATH_CALUDE_pairwise_sums_not_distinct_l2911_291185

theorem pairwise_sums_not_distinct (n : ℕ+) (A : Finset (ZMod n)) :
  A.card > 1 + Real.sqrt (n + 4) →
  ∃ (a b c d : ZMod n), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ d ∈ A ∧ 
    (a, b) ≠ (c, d) ∧ (a, b) ≠ (d, c) ∧ a + b = c + d :=
by sorry

end NUMINAMATH_CALUDE_pairwise_sums_not_distinct_l2911_291185


namespace NUMINAMATH_CALUDE_intersection_P_Q_l2911_291121

def P : Set ℝ := {x : ℝ | 1 < x ∧ x < 4}
def Q : Set ℝ := {x : ℝ | 2 < x ∧ x < 3}

theorem intersection_P_Q : P ∩ Q = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l2911_291121


namespace NUMINAMATH_CALUDE_trig_simplification_l2911_291138

theorem trig_simplification (x y : ℝ) :
  (Real.cos (x + π/4))^2 + (Real.cos (x + y + π/2))^2 - 
  2 * Real.cos (x + π/4) * Real.cos (y + π/4) * Real.cos (x + y + π/2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l2911_291138


namespace NUMINAMATH_CALUDE_water_evaporation_rate_l2911_291180

/-- Given a bowl with water that experiences evaporation over time, 
    calculate the amount of water evaporated per day. -/
theorem water_evaporation_rate 
  (initial_water : ℝ) 
  (evaporation_period : ℕ) 
  (evaporation_percentage : ℝ) 
  (h1 : initial_water = 10)
  (h2 : evaporation_period = 50)
  (h3 : evaporation_percentage = 0.03)
  : (initial_water * evaporation_percentage) / evaporation_period = 0.06 := by
  sorry


end NUMINAMATH_CALUDE_water_evaporation_rate_l2911_291180


namespace NUMINAMATH_CALUDE_function_values_imply_parameters_l2911_291123

theorem function_values_imply_parameters 
  (f : ℝ → ℝ) 
  (a θ : ℝ) 
  (h1 : ∀ x, f x = Real.sin (x + θ) + a * Real.cos (x + 2 * θ))
  (h2 : θ > -Real.pi / 2 ∧ θ < Real.pi / 2)
  (h3 : f (Real.pi / 2) = 0)
  (h4 : f Real.pi = 1) :
  a = -1 ∧ θ = -Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_function_values_imply_parameters_l2911_291123


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l2911_291112

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then x^2 * Real.exp (abs x) * Real.sin (1 / x^2) else 0

theorem f_derivative_at_zero :
  deriv f 0 = 0 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l2911_291112


namespace NUMINAMATH_CALUDE_vector_operation_equals_two_l2911_291111

-- Define the vectors
def a : ℝ × ℝ := (4, 2)
def b : ℝ × ℝ := (1, -1)

-- Define the dot product operation
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Define vector scalar multiplication
def scalar_mult (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (c * v.1, c * v.2)

-- Define vector subtraction
def vector_sub (v w : ℝ × ℝ) : ℝ × ℝ :=
  (v.1 - w.1, v.2 - w.2)

-- Theorem statement
theorem vector_operation_equals_two :
  dot_product (vector_sub (scalar_mult 2 a) b) b = 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_equals_two_l2911_291111


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2911_291124

theorem absolute_value_inequality (x : ℝ) : |x - 2| ≥ |x| ↔ x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2911_291124


namespace NUMINAMATH_CALUDE_boys_share_is_14_l2911_291113

/-- The amount of money each boy makes from selling shrimp -/
def boys_share (victor_shrimp : ℕ) (austin_diff : ℕ) (price : ℚ) (per_shrimp : ℕ) : ℚ :=
  let austin_shrimp := victor_shrimp - austin_diff
  let victor_austin_total := victor_shrimp + austin_shrimp
  let brian_shrimp := victor_austin_total / 2
  let total_shrimp := victor_shrimp + austin_shrimp + brian_shrimp
  let total_money := (total_shrimp / per_shrimp) * price
  total_money / 3

/-- Theorem stating that each boy's share is $14 given the problem conditions -/
theorem boys_share_is_14 :
  boys_share 26 8 7 11 = 14 := by
  sorry

end NUMINAMATH_CALUDE_boys_share_is_14_l2911_291113


namespace NUMINAMATH_CALUDE_base_b_divisibility_l2911_291173

theorem base_b_divisibility (b : ℤ) : b = 7 ↔ ¬(5 ∣ (b^2 * (3*b - 2))) ∧ 
  (b = 6 ∨ b = 8 ∨ b = 9 ∨ b = 10 → 5 ∣ (b^2 * (3*b - 2))) := by
  sorry

end NUMINAMATH_CALUDE_base_b_divisibility_l2911_291173


namespace NUMINAMATH_CALUDE_sandy_shopping_money_l2911_291135

theorem sandy_shopping_money (original_amount : ℝ) : 
  original_amount * 0.7 = 210 → original_amount = 300 :=
by sorry

end NUMINAMATH_CALUDE_sandy_shopping_money_l2911_291135


namespace NUMINAMATH_CALUDE_parallelogram_acute_angle_iff_diagonal_equation_l2911_291184

/-- A parallelogram with side lengths a and b, and diagonal lengths m and n -/
structure Parallelogram where
  a : ℝ
  b : ℝ
  m : ℝ
  n : ℝ
  ha : a > 0
  hb : b > 0
  hm : m > 0
  hn : n > 0

/-- The acute angle of a parallelogram -/
def acute_angle (p : Parallelogram) : ℝ := sorry

theorem parallelogram_acute_angle_iff_diagonal_equation (p : Parallelogram) :
  p.a^4 + p.b^4 = p.m^2 * p.n^2 ↔ acute_angle p = π/4 := by sorry

end NUMINAMATH_CALUDE_parallelogram_acute_angle_iff_diagonal_equation_l2911_291184


namespace NUMINAMATH_CALUDE_possible_total_students_l2911_291150

/-- Represents the possible total number of students -/
inductive TotalStudents
  | seventySix
  | eighty

/-- Checks if a number is a valid group size given the constraints -/
def isValidGroupSize (size : ℕ) : Prop :=
  size = 12 ∨ size = 13 ∨ size = 14

/-- Represents the distribution of students into groups -/
structure StudentDistribution where
  groupSizes : Fin 6 → ℕ
  validSizes : ∀ i, isValidGroupSize (groupSizes i)
  fourGroupsOf13 : (Finset.filter (fun i => groupSizes i = 13) Finset.univ).card = 4
  totalStudents : TotalStudents

/-- The main theorem stating the possible total number of students -/
theorem possible_total_students (d : StudentDistribution) :
    d.totalStudents = TotalStudents.seventySix ∨
    d.totalStudents = TotalStudents.eighty :=
  sorry

end NUMINAMATH_CALUDE_possible_total_students_l2911_291150


namespace NUMINAMATH_CALUDE_min_edges_theorem_l2911_291164

/-- A simple graph with 19998 vertices -/
structure Graph :=
  (vertices : Finset Nat)
  (edges : Finset (Nat × Nat))
  (simple : ∀ e ∈ edges, e.1 ≠ e.2)
  (vertex_count : vertices.card = 19998)

/-- A subgraph of G with 9999 vertices -/
def Subgraph (G : Graph) :=
  {G' : Graph | G'.vertices ⊆ G.vertices ∧ G'.edges ⊆ G.edges ∧ G'.vertices.card = 9999}

/-- The condition that any subgraph with 9999 vertices has at least 9999 edges -/
def SubgraphEdgeCondition (G : Graph) :=
  ∀ G' ∈ Subgraph G, G'.edges.card ≥ 9999

/-- The theorem stating that G has at least 49995 edges -/
theorem min_edges_theorem (G : Graph) (h : SubgraphEdgeCondition G) :
  G.edges.card ≥ 49995 := by
  sorry

end NUMINAMATH_CALUDE_min_edges_theorem_l2911_291164


namespace NUMINAMATH_CALUDE_minimum_postage_l2911_291116

/-- Calculates the postage for a given weight in grams -/
def calculatePostage (weight : ℕ) : ℚ :=
  if weight ≤ 100 then
    (((weight - 1) / 20 + 1) * 8) / 10
  else
    4 + (((weight - 101) / 100 + 1) * 2)

/-- Calculates the total postage for two envelopes -/
def totalPostage (x : ℕ) : ℚ :=
  calculatePostage (12 * x + 4) + calculatePostage (12 * (11 - x) + 4)

theorem minimum_postage :
  ∃ x : ℕ, x ≤ 11 ∧ totalPostage x = 56/10 ∧ ∀ y : ℕ, y ≤ 11 → totalPostage y ≥ 56/10 :=
sorry

end NUMINAMATH_CALUDE_minimum_postage_l2911_291116


namespace NUMINAMATH_CALUDE_east_bus_speed_l2911_291114

/-- The speed of a bus traveling east, given that it and another bus traveling
    west at 60 mph end up 460 miles apart after 4 hours. -/
theorem east_bus_speed : ℝ := by
  -- Define the speed of the west-traveling bus
  let west_speed : ℝ := 60
  -- Define the time of travel
  let time : ℝ := 4
  -- Define the total distance between buses after travel
  let total_distance : ℝ := 460
  -- Define the speed of the east-traveling bus
  let east_speed : ℝ := (total_distance / time) - west_speed
  -- Assert that the east_speed is equal to 55
  have h : east_speed = 55 := by sorry
  -- Return the speed of the east-traveling bus
  exact east_speed

end NUMINAMATH_CALUDE_east_bus_speed_l2911_291114


namespace NUMINAMATH_CALUDE_raisin_nut_mixture_cost_fraction_l2911_291148

theorem raisin_nut_mixture_cost_fraction :
  ∀ (R : ℚ),
  R > 0 →
  let raisin_pounds : ℚ := 3
  let nut_pounds : ℚ := 4
  let raisin_cost_per_pound : ℚ := R
  let nut_cost_per_pound : ℚ := 4 * R
  let total_raisin_cost : ℚ := raisin_pounds * raisin_cost_per_pound
  let total_nut_cost : ℚ := nut_pounds * nut_cost_per_pound
  let total_mixture_cost : ℚ := total_raisin_cost + total_nut_cost
  (total_raisin_cost / total_mixture_cost) = 3 / 19 :=
by
  sorry

end NUMINAMATH_CALUDE_raisin_nut_mixture_cost_fraction_l2911_291148


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2911_291158

theorem simplify_and_evaluate (a : ℝ) (h : a = -2) :
  (1 - 1 / (a + 1)) / ((a^2 - 2*a + 1) / (a^2 - 1)) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2911_291158


namespace NUMINAMATH_CALUDE_sapphire_percentage_l2911_291139

def total_gems : ℕ := 12000
def diamonds : ℕ := 1800
def rubies : ℕ := 4000
def emeralds : ℕ := 3500

def sapphires : ℕ := total_gems - (diamonds + rubies + emeralds)

theorem sapphire_percentage :
  (sapphires : ℚ) / total_gems * 100 = 22.5 := by sorry

end NUMINAMATH_CALUDE_sapphire_percentage_l2911_291139


namespace NUMINAMATH_CALUDE_solve_for_a_l2911_291177

theorem solve_for_a (a b : ℚ) (h1 : b / a = 4) (h2 : b = 18 - 6 * a) : a = 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l2911_291177


namespace NUMINAMATH_CALUDE_student_average_less_than_true_average_l2911_291145

theorem student_average_less_than_true_average 
  (w x y z : ℝ) (h : w < x ∧ x < y ∧ y < z) : 
  (2*w + 2*x + y + z) / 6 < (w + x + y + z) / 4 := by
sorry

end NUMINAMATH_CALUDE_student_average_less_than_true_average_l2911_291145


namespace NUMINAMATH_CALUDE_sedan_count_l2911_291125

theorem sedan_count (trucks sedans motorcycles : ℕ) : 
  trucks * 7 = sedans * 3 →
  sedans * 2 = motorcycles * 7 →
  motorcycles = 2600 →
  sedans = 9100 := by
sorry

end NUMINAMATH_CALUDE_sedan_count_l2911_291125


namespace NUMINAMATH_CALUDE_no_unique_five_day_august_l2911_291195

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a calendar month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- Given a month, returns the number of occurrences of each day of the week -/
def countDaysInMonth (m : Month) : DayOfWeek → Nat :=
  sorry

/-- July has five Tuesdays and 30 days -/
def july : Month :=
  { days := 30,
    firstDay := sorry }

/-- August follows July and has 30 days -/
def august : Month :=
  { days := 30,
    firstDay := sorry }

/-- There is no unique day that occurs five times in August -/
theorem no_unique_five_day_august :
  ¬ ∃! (d : DayOfWeek), countDaysInMonth august d = 5 :=
sorry

end NUMINAMATH_CALUDE_no_unique_five_day_august_l2911_291195


namespace NUMINAMATH_CALUDE_equivalent_functions_l2911_291108

theorem equivalent_functions (x : ℝ) : x^2 = (x^6)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_functions_l2911_291108
