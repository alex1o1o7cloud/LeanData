import Mathlib

namespace NUMINAMATH_CALUDE_expression_simplification_l4066_406635

theorem expression_simplification (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 2) (h3 : x ≠ -1) :
  (x + 1) / (x^2 - 2*x) / (1 + 1/x) = 1 / (x - 2) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l4066_406635


namespace NUMINAMATH_CALUDE_least_multiple_24_greater_450_l4066_406695

theorem least_multiple_24_greater_450 : ∃ n : ℕ, 24 * n = 456 ∧ 456 > 450 ∧ ∀ m : ℕ, 24 * m > 450 → 24 * m ≥ 456 :=
sorry

end NUMINAMATH_CALUDE_least_multiple_24_greater_450_l4066_406695


namespace NUMINAMATH_CALUDE_peach_fraction_proof_l4066_406665

theorem peach_fraction_proof (martine_peaches benjy_peaches gabrielle_peaches : ℕ) : 
  martine_peaches = 16 →
  gabrielle_peaches = 15 →
  martine_peaches = 2 * benjy_peaches + 6 →
  (benjy_peaches : ℚ) / gabrielle_peaches = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_peach_fraction_proof_l4066_406665


namespace NUMINAMATH_CALUDE_remainder_theorem_l4066_406669

theorem remainder_theorem (P D Q R D' Q' R' D'' Q'' R'' : ℕ) 
  (h1 : P = D * Q + R) 
  (h2 : Q = D' * Q' + R') 
  (h3 : Q' = D'' * Q'' + R'') : 
  P % (D * D' * D'') = D' * D * R'' + D * R' + R :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l4066_406669


namespace NUMINAMATH_CALUDE_sons_age_l4066_406607

/-- Given a father and son, where the father is 46 years older than the son,
    and in two years the father's age will be twice the son's age,
    prove that the son's current age is 44 years. -/
theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 46 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 44 := by sorry

end NUMINAMATH_CALUDE_sons_age_l4066_406607


namespace NUMINAMATH_CALUDE_max_candies_theorem_l4066_406636

/-- Represents the distribution of candies among students -/
structure CandyDistribution where
  num_students : ℕ
  total_candies : ℕ
  min_candies : ℕ
  max_candies : ℕ

/-- The greatest number of candies one student could have taken -/
def max_student_candies (d : CandyDistribution) : ℕ :=
  min d.max_candies (d.total_candies - (d.num_students - 1) * d.min_candies)

/-- Theorem stating the maximum number of candies one student could have taken -/
theorem max_candies_theorem (d : CandyDistribution) 
    (h1 : d.num_students = 50)
    (h2 : d.total_candies = 50 * 7)
    (h3 : d.min_candies = 1)
    (h4 : d.max_candies = 20) :
    max_student_candies d = 20 := by
  sorry

#eval max_student_candies { num_students := 50, total_candies := 350, min_candies := 1, max_candies := 20 }

end NUMINAMATH_CALUDE_max_candies_theorem_l4066_406636


namespace NUMINAMATH_CALUDE_hidden_faces_sum_l4066_406601

/-- The sum of numbers on a single die -/
def die_sum : ℕ := 21

/-- The number of dice -/
def num_dice : ℕ := 4

/-- The visible numbers on the stacked dice -/
def visible_numbers : List ℕ := [1, 2, 2, 3, 3, 4, 5, 6]

/-- The sum of visible numbers -/
def visible_sum : ℕ := visible_numbers.sum

theorem hidden_faces_sum :
  (num_dice * die_sum) - visible_sum = 58 := by
  sorry

end NUMINAMATH_CALUDE_hidden_faces_sum_l4066_406601


namespace NUMINAMATH_CALUDE_least_positive_integer_with_specific_remainders_l4066_406645

theorem least_positive_integer_with_specific_remainders : ∃ n : ℕ, 
  (n > 0) ∧ 
  (n % 4 = 3) ∧ 
  (n % 5 = 4) ∧ 
  (n % 6 = 5) ∧ 
  (n % 7 = 6) ∧ 
  (n % 11 = 10) ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 4 = 3 ∧ m % 5 = 4 ∧ m % 6 = 5 ∧ m % 7 = 6 ∧ m % 11 = 10 → m ≥ n) ∧
  n = 4619 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_specific_remainders_l4066_406645


namespace NUMINAMATH_CALUDE_triangle_sides_l4066_406615

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (3 * Real.pi + x) * Real.cos (Real.pi - x) + (Real.cos (Real.pi / 2 + x))^2

theorem triangle_sides (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < Real.pi →
  f A = 3/2 →
  a = 2 →
  b + c = 4 →
  b = 2 ∧ c = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_sides_l4066_406615


namespace NUMINAMATH_CALUDE_y_axis_intersection_uniqueness_l4066_406616

theorem y_axis_intersection_uniqueness (f : ℝ → ℝ) : 
  ∃! y, f 0 = y :=
sorry

end NUMINAMATH_CALUDE_y_axis_intersection_uniqueness_l4066_406616


namespace NUMINAMATH_CALUDE_time_to_reach_destination_l4066_406621

/-- Calculates the time needed to reach a destination given initial movement and remaining distance -/
theorem time_to_reach_destination (initial_distance : ℝ) (initial_time : ℝ) (remaining_distance_yards : ℝ) : 
  initial_distance > 0 ∧ initial_time > 0 ∧ remaining_distance_yards > 0 →
  (remaining_distance_yards * 3) / (initial_distance / initial_time) = 75 :=
by
  sorry

#check time_to_reach_destination 80 20 100

end NUMINAMATH_CALUDE_time_to_reach_destination_l4066_406621


namespace NUMINAMATH_CALUDE_square_of_87_l4066_406671

theorem square_of_87 : 87^2 = 7569 := by sorry

end NUMINAMATH_CALUDE_square_of_87_l4066_406671


namespace NUMINAMATH_CALUDE_routes_3x2_grid_l4066_406634

/-- The number of routes in a grid from top-left to bottom-right -/
def numRoutes (width height : ℕ) : ℕ :=
  Nat.choose (width + height) width

theorem routes_3x2_grid : numRoutes 3 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_routes_3x2_grid_l4066_406634


namespace NUMINAMATH_CALUDE_subset_implies_a_geq_four_l4066_406684

theorem subset_implies_a_geq_four (a : ℝ) :
  let A : Set ℝ := {x | 1 < x ∧ x < 2}
  let B : Set ℝ := {x | x^2 - a*x + 3 ≤ 0}
  A ⊆ B → a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_geq_four_l4066_406684


namespace NUMINAMATH_CALUDE_sufficient_condition_implies_range_l4066_406639

theorem sufficient_condition_implies_range (a : ℝ) : 
  (∀ x, (x - 1) * (x - 2) < 0 → x - a < 0) → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_implies_range_l4066_406639


namespace NUMINAMATH_CALUDE_triangle_angle_inequalities_l4066_406675

theorem triangle_angle_inequalities (α β γ : Real) 
  (h_triangle : α + β + γ = Real.pi) : 
  (Real.sin (α/2) * Real.sin (β/2) * Real.sin (γ/2) ≤ 1/8) ∧ 
  (Real.cos α * Real.cos β * Real.cos γ ≤ 1/8) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_inequalities_l4066_406675


namespace NUMINAMATH_CALUDE_triangle_side_length_l4066_406696

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a = 2√3, b = 2, and the area S_ABC = √3, then c = 2 or c = 2√7. -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) (S_ABC : ℝ) :
  a = 2 * Real.sqrt 3 →
  b = 2 →
  S_ABC = Real.sqrt 3 →
  (c = 2 ∨ c = 2 * Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4066_406696


namespace NUMINAMATH_CALUDE_uncertain_relationship_l4066_406640

/-- Represents a line in 3D space -/
structure Line3D where
  -- We don't need to define the internal structure of a line for this problem

/-- Represents the possible relationships between two lines -/
inductive LineRelationship
  | Parallel
  | Perpendicular
  | Skew

/-- Perpendicularity of two lines -/
def perpendicular (l1 l2 : Line3D) : Prop := sorry

/-- The relationship between two lines -/
def relationship (l1 l2 : Line3D) : LineRelationship := sorry

theorem uncertain_relationship 
  (l1 l2 l3 l4 : Line3D) 
  (h_distinct : l1 ≠ l2 ∧ l1 ≠ l3 ∧ l1 ≠ l4 ∧ l2 ≠ l3 ∧ l2 ≠ l4 ∧ l3 ≠ l4)
  (h12 : perpendicular l1 l2)
  (h23 : perpendicular l2 l3)
  (h34 : perpendicular l3 l4) :
  ∃ (r : LineRelationship), relationship l1 l4 = r ∧ 
    (r = LineRelationship.Parallel ∨ 
     r = LineRelationship.Perpendicular ∨ 
     r = LineRelationship.Skew) :=
by sorry

end NUMINAMATH_CALUDE_uncertain_relationship_l4066_406640


namespace NUMINAMATH_CALUDE_original_number_proof_l4066_406685

theorem original_number_proof (r : ℝ) : 
  r * (1 + 0.125) - r * (1 - 0.25) = 30 → r = 80 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l4066_406685


namespace NUMINAMATH_CALUDE_bobby_candy_remaining_l4066_406682

def candy_problem (initial_count : ℕ) (first_eaten : ℕ) (second_eaten : ℕ) : ℕ :=
  initial_count - (first_eaten + second_eaten)

theorem bobby_candy_remaining :
  candy_problem 36 17 15 = 4 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_remaining_l4066_406682


namespace NUMINAMATH_CALUDE_apples_at_first_store_l4066_406606

def first_store_price : ℝ := 3
def second_store_price : ℝ := 4
def second_store_apples : ℝ := 10
def savings_per_apple : ℝ := 0.1

theorem apples_at_first_store :
  let second_store_price_per_apple := second_store_price / second_store_apples
  let first_store_price_per_apple := second_store_price_per_apple + savings_per_apple
  first_store_price / first_store_price_per_apple = 6 := by sorry

end NUMINAMATH_CALUDE_apples_at_first_store_l4066_406606


namespace NUMINAMATH_CALUDE_standard_deviation_of_applicant_ages_l4066_406631

def average_age : ℕ := 10
def num_different_ages : ℕ := 17

theorem standard_deviation_of_applicant_ages :
  ∃ (s : ℕ),
    s > 0 ∧
    (average_age + s) - (average_age - s) + 1 = num_different_ages ∧
    s = 8 := by
  sorry

end NUMINAMATH_CALUDE_standard_deviation_of_applicant_ages_l4066_406631


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l4066_406619

/-- A quadratic function with the given properties -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_symmetry (a b c : ℝ) :
  f a b c (-2) = -13/2 ∧
  f a b c (-1) = -4 ∧
  f a b c 0 = -5/2 ∧
  f a b c 1 = -2 ∧
  f a b c 2 = -5/2 →
  f a b c 3 = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l4066_406619


namespace NUMINAMATH_CALUDE_student_lecture_selections_l4066_406662

/-- The number of different selection methods for students choosing lectures -/
def selection_methods (num_students : ℕ) (num_lectures : ℕ) : ℕ :=
  num_lectures ^ num_students

/-- Theorem: Given 4 students and 3 lectures, the number of different selection methods is 81 -/
theorem student_lecture_selections :
  selection_methods 4 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_student_lecture_selections_l4066_406662


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l4066_406683

theorem triangle_max_perimeter :
  ∀ x : ℕ,
    x > 0 →
    x < 17 →
    x + 4*x > 17 →
    x + 17 > 4*x →
    ∀ y : ℕ,
      y > 0 →
      y < 17 →
      y + 4*y > 17 →
      y + 17 > 4*y →
      x + 4*x + 17 ≥ y + 4*y + 17 →
      x + 4*x + 17 ≤ 42 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l4066_406683


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l4066_406658

theorem nested_fraction_equality : 1 + 1 / (1 + 1 / (2 + 1)) = 7 / 4 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l4066_406658


namespace NUMINAMATH_CALUDE_min_value_sum_l4066_406605

theorem min_value_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + 3)⁻¹ + (b + 3)⁻¹ = (1 : ℝ) / 4) :
  ∀ x y : ℝ, x > 0 → y > 0 → (x + 3)⁻¹ + (y + 3)⁻¹ = (1 : ℝ) / 4 → 
  a + 3 * b ≤ x + 3 * y ∧ a + 3 * b = 4 + 8 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_l4066_406605


namespace NUMINAMATH_CALUDE_inequality_theorem_l4066_406641

theorem inequality_theorem :
  (∀ (x y z : ℝ), x^2 + 2*y^2 + 3*z^2 ≥ Real.sqrt 3 * (x*y + y*z + z*x)) ∧
  (∃ (k : ℝ), k > Real.sqrt 3 ∧ ∀ (x y z : ℝ), x^2 + 2*y^2 + 3*z^2 ≥ k * (x*y + y*z + z*x)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l4066_406641


namespace NUMINAMATH_CALUDE_initial_nails_l4066_406652

theorem initial_nails (found_nails : ℕ) (nails_to_buy : ℕ) (total_nails : ℕ) 
  (h1 : found_nails = 144)
  (h2 : nails_to_buy = 109)
  (h3 : total_nails = 500)
  : total_nails = found_nails + nails_to_buy + 247 := by
  sorry

end NUMINAMATH_CALUDE_initial_nails_l4066_406652


namespace NUMINAMATH_CALUDE_triangle_problem_l4066_406617

theorem triangle_problem (A B C : Real) (a b c : Real) (D : Real × Real) :
  (2 * a^2 * Real.sin B * Real.sin C = Real.sqrt 3 * (a^2 + b^2 - c^2) * Real.sin A) →
  (a = 1) →
  (b = 2) →
  (D = ((A + B) / 2, 0)) →  -- Assuming A and B are coordinates on the x-axis
  (C = Real.pi / 3) ∧
  (Real.sqrt ((C - D.1)^2 + D.2^2) = Real.sqrt 7 / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l4066_406617


namespace NUMINAMATH_CALUDE_smallest_even_divisible_by_20_and_60_l4066_406638

theorem smallest_even_divisible_by_20_and_60 : ∃ n : ℕ, n > 0 ∧ Even n ∧ 20 ∣ n ∧ 60 ∣ n ∧ ∀ m : ℕ, m > 0 → Even m → 20 ∣ m → 60 ∣ m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_even_divisible_by_20_and_60_l4066_406638


namespace NUMINAMATH_CALUDE_marble_ratio_proof_l4066_406649

theorem marble_ratio_proof (initial_red : ℕ) (initial_blue : ℕ) (red_taken : ℕ) (total_left : ℕ) :
  initial_red = 20 →
  initial_blue = 30 →
  red_taken = 3 →
  total_left = 35 →
  ∃ (blue_taken : ℕ),
    blue_taken * red_taken = 4 * red_taken ∧
    initial_red + initial_blue = total_left + red_taken + blue_taken :=
by
  sorry

end NUMINAMATH_CALUDE_marble_ratio_proof_l4066_406649


namespace NUMINAMATH_CALUDE_cube_root_of_19683_l4066_406686

theorem cube_root_of_19683 (x : ℝ) (h1 : x > 0) (h2 : x^3 = 19683) : x = 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_19683_l4066_406686


namespace NUMINAMATH_CALUDE_inequality_proof_l4066_406622

theorem inequality_proof (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
  ((2 + x) / (1 + x))^2 + ((2 + y) / (1 + y))^2 ≥ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4066_406622


namespace NUMINAMATH_CALUDE_shape_to_square_cut_l4066_406667

/-- Represents a shape with a given area -/
structure Shape :=
  (area : ℝ)

/-- Represents a cut of a shape into three parts -/
structure Cut (s : Shape) :=
  (part1 : Shape)
  (part2 : Shape)
  (part3 : Shape)
  (sum_area : part1.area + part2.area + part3.area = s.area)

/-- Predicate to check if three shapes can form a square -/
def CanFormSquare (p1 p2 p3 : Shape) : Prop :=
  ∃ (side : ℝ), side > 0 ∧ p1.area + p2.area + p3.area = side * side

/-- Theorem stating that any shape can be cut into three parts that form a square -/
theorem shape_to_square_cut (s : Shape) : 
  ∃ (c : Cut s), CanFormSquare c.part1 c.part2 c.part3 := by
  sorry

#check shape_to_square_cut

end NUMINAMATH_CALUDE_shape_to_square_cut_l4066_406667


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_four_l4066_406680

theorem reciprocal_of_negative_four :
  ∃ x : ℚ, x * (-4) = 1 ∧ x = -1/4 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_four_l4066_406680


namespace NUMINAMATH_CALUDE_first_month_sales_l4066_406603

def sales_month_2 : ℕ := 5744
def sales_month_3 : ℕ := 5864
def sales_month_4 : ℕ := 6122
def sales_month_5 : ℕ := 6588
def sales_month_6 : ℕ := 4916
def average_sale : ℕ := 5750

theorem first_month_sales :
  sales_month_2 + sales_month_3 + sales_month_4 + sales_month_5 + sales_month_6 + 5266 = 6 * average_sale :=
by sorry

end NUMINAMATH_CALUDE_first_month_sales_l4066_406603


namespace NUMINAMATH_CALUDE_treasure_value_is_3049_l4066_406689

/-- Converts a list of digits in base 7 to its decimal (base 10) equivalent -/
def base7ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

/-- The total value of the treasures in base 10 -/
def totalTreasureValue : Nat :=
  base7ToDecimal [4, 1, 2, 3] + -- 3214₇
  base7ToDecimal [2, 5, 6, 1] + -- 1652₇
  base7ToDecimal [1, 3, 4, 2] + -- 2431₇
  base7ToDecimal [4, 5, 6]      -- 654₇

/-- Theorem stating that the total value of the treasures is 3049 -/
theorem treasure_value_is_3049 : totalTreasureValue = 3049 := by
  sorry


end NUMINAMATH_CALUDE_treasure_value_is_3049_l4066_406689


namespace NUMINAMATH_CALUDE_existence_of_special_integers_l4066_406610

theorem existence_of_special_integers : ∃ (a b : ℕ+), 
  (¬ (7 ∣ (a.val * b.val * (a.val + b.val)))) ∧ 
  ((7^7 : ℕ) ∣ ((a.val + b.val)^7 - a.val^7 - b.val^7)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_integers_l4066_406610


namespace NUMINAMATH_CALUDE_min_fence_posts_is_22_l4066_406629

/-- Calculates the number of fence posts needed for a rectangular grazing area -/
def fence_posts (length width post_spacing : ℕ) : ℕ :=
  let long_side_posts := (length / post_spacing) + 1
  let short_side_posts := (width / post_spacing) + 1
  (2 * long_side_posts) + short_side_posts - 2

/-- The minimum number of fence posts for the given dimensions is 22 -/
theorem min_fence_posts_is_22 :
  fence_posts 80 50 10 = 22 :=
by sorry

end NUMINAMATH_CALUDE_min_fence_posts_is_22_l4066_406629


namespace NUMINAMATH_CALUDE_quiz_winning_probability_l4066_406697

/-- The number of questions in the quiz -/
def num_questions : ℕ := 4

/-- The number of choices for each question -/
def num_choices : ℕ := 4

/-- The minimum number of correct answers needed to win -/
def min_correct : ℕ := 3

/-- The probability of answering a single question correctly -/
def prob_correct : ℚ := 1 / num_choices

/-- The probability of answering a single question incorrectly -/
def prob_incorrect : ℚ := 1 - prob_correct

/-- The probability of winning the quiz -/
def prob_winning : ℚ := (num_questions.choose min_correct) * (prob_correct ^ min_correct * prob_incorrect ^ (num_questions - min_correct)) +
                        (num_questions.choose num_questions) * (prob_correct ^ num_questions)

theorem quiz_winning_probability :
  prob_winning = 13 / 256 := by
  sorry

end NUMINAMATH_CALUDE_quiz_winning_probability_l4066_406697


namespace NUMINAMATH_CALUDE_ellipse_equation_line_equation_l4066_406647

-- Define the ellipse
structure Ellipse where
  center : ℝ × ℝ
  a : ℝ
  b : ℝ
  foci_on_x_axis : Bool

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define a line
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define the problem
def ellipse_problem (e : Ellipse) (B : Point) (Q : Point) (F : Point) : Prop :=
  e.center = (0, 0) ∧
  e.foci_on_x_axis = true ∧
  B = ⟨0, 1⟩ ∧
  Q = ⟨0, 3/2⟩ ∧
  F.y = 0 ∧
  F.x > 0 ∧
  (F.x - 0 + 2 * Real.sqrt 2) / Real.sqrt 2 = 3

-- Theorem for the ellipse equation
theorem ellipse_equation (e : Ellipse) (B : Point) (Q : Point) (F : Point) :
  ellipse_problem e B Q F →
  ∀ x y : ℝ, (x^2 / 3 + y^2 = 1) ↔ (x^2 / e.a^2 + y^2 / e.b^2 = 1) :=
sorry

-- Theorem for the line equation
theorem line_equation (e : Ellipse) (B : Point) (Q : Point) (F : Point) (l : Line) :
  ellipse_problem e B Q F →
  (∃ M N : Point,
    M ≠ N ∧
    (M.x^2 / 3 + M.y^2 = 1) ∧
    (N.x^2 / 3 + N.y^2 = 1) ∧
    M.y = l.slope * M.x + l.intercept ∧
    N.y = l.slope * N.x + l.intercept ∧
    (M.x - B.x)^2 + (M.y - B.y)^2 = (N.x - B.x)^2 + (N.y - B.y)^2) →
  (l.slope = Real.sqrt 6 / 3 ∧ l.intercept = 3/2) ∨
  (l.slope = -Real.sqrt 6 / 3 ∧ l.intercept = 3/2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_line_equation_l4066_406647


namespace NUMINAMATH_CALUDE_f_min_value_l4066_406600

/-- The quadratic function f(x) = 2x^2 - 8x + 9 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 9

/-- The minimum value of f(x) is 1 -/
theorem f_min_value : ∃ (m : ℝ), m = 1 ∧ ∀ (x : ℝ), f x ≥ m := by
  sorry

end NUMINAMATH_CALUDE_f_min_value_l4066_406600


namespace NUMINAMATH_CALUDE_sequence_identity_l4066_406646

def IsIncreasing (a : ℕ → ℕ) : Prop :=
  ∀ i j, i ≤ j → a i ≤ a j

def DivisorCountEqual (a : ℕ → ℕ) : Prop :=
  ∀ i j, (Nat.divisors (i + j)).card = (Nat.divisors (a i + a j)).card

theorem sequence_identity (a : ℕ → ℕ) 
    (h1 : IsIncreasing a) 
    (h2 : DivisorCountEqual a) : 
    ∀ n : ℕ, a n = n := by
  sorry

end NUMINAMATH_CALUDE_sequence_identity_l4066_406646


namespace NUMINAMATH_CALUDE_angle_half_quadrant_l4066_406661

-- Define the angle α
def α : ℝ := sorry

-- Define the integer k
def k : ℤ := sorry

-- Define the condition for α
axiom α_condition : 40 + k * 360 < α ∧ α < 140 + k * 360

-- Define the first quadrant
def first_quadrant (θ : ℝ) : Prop := 0 ≤ θ ∧ θ < 90

-- Define the third quadrant
def third_quadrant (θ : ℝ) : Prop := 180 ≤ θ ∧ θ < 270

-- State the theorem
theorem angle_half_quadrant : 
  first_quadrant (α / 2) ∨ third_quadrant (α / 2) := by sorry

end NUMINAMATH_CALUDE_angle_half_quadrant_l4066_406661


namespace NUMINAMATH_CALUDE_rain_probability_in_tel_aviv_l4066_406643

/-- The probability of exactly k successes in n independent trials,
    where the probability of success in each trial is p. -/
def binomialProbability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The probability of rain on any given day in Tel Aviv -/
def probabilityOfRain : ℝ := 0.5

/-- The number of randomly chosen days -/
def totalDays : ℕ := 6

/-- The number of rainy days we're interested in -/
def rainyDays : ℕ := 4

theorem rain_probability_in_tel_aviv :
  binomialProbability totalDays rainyDays probabilityOfRain = 0.234375 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_in_tel_aviv_l4066_406643


namespace NUMINAMATH_CALUDE_final_expression_l4066_406677

/-- Given a real number b, prove that doubling b, adding 4, subtracting 4b, and dividing by 2 results in -b + 2 -/
theorem final_expression (b : ℝ) : ((2 * b + 4) - 4 * b) / 2 = -b + 2 := by
  sorry

end NUMINAMATH_CALUDE_final_expression_l4066_406677


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l4066_406604

theorem max_value_trig_expression (a b φ : ℝ) :
  (∀ θ : ℝ, a * Real.cos θ + b * Real.sin (θ + φ) ≤ Real.sqrt (a^2 + 2*a*b * Real.sin φ + b^2)) ∧
  (∃ θ : ℝ, a * Real.cos θ + b * Real.sin (θ + φ) = Real.sqrt (a^2 + 2*a*b * Real.sin φ + b^2)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l4066_406604


namespace NUMINAMATH_CALUDE_inequality_solution_l4066_406699

theorem inequality_solution (x : ℝ) : (x^2 - 49) / (x + 7) < 0 ↔ x < -7 ∨ (-7 < x ∧ x < 7) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l4066_406699


namespace NUMINAMATH_CALUDE_simplified_fraction_numerator_problem_solution_l4066_406632

theorem simplified_fraction_numerator (a : ℕ) (h : a > 0) : 
  ((a + 1 : ℚ) / a - a / (a + 1)) * (a * (a + 1)) = 2 * a + 1 :=
by sorry

theorem problem_solution : 
  ((2024 : ℚ) / 2023 - 2023 / 2024) * (2023 * 2024) = 4047 :=
by sorry

end NUMINAMATH_CALUDE_simplified_fraction_numerator_problem_solution_l4066_406632


namespace NUMINAMATH_CALUDE_tree_support_uses_triangle_stability_l4066_406655

/-- A triangle formed by two supporting sticks and a tree -/
structure TreeSupport where
  stickOne : ℝ × ℝ  -- Coordinates of the first stick's base
  stickTwo : ℝ × ℝ  -- Coordinates of the second stick's base
  treeTop : ℝ × ℝ   -- Coordinates of the tree's top

/-- The property of a triangle that provides support -/
def triangleProperty : String := "stability"

/-- 
  Theorem: The property of triangles applied when using two wooden sticks 
  to support a falling tree is stability.
-/
theorem tree_support_uses_triangle_stability (support : TreeSupport) : 
  triangleProperty = "stability" := by
  sorry

end NUMINAMATH_CALUDE_tree_support_uses_triangle_stability_l4066_406655


namespace NUMINAMATH_CALUDE_max_value_of_f_on_interval_l4066_406679

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

-- Define the interval
def interval : Set ℝ := Set.Icc (-1) 1

-- Theorem statement
theorem max_value_of_f_on_interval :
  ∃ (c : ℝ), c ∈ interval ∧ f c = 2 ∧ ∀ x ∈ interval, f x ≤ f c :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_interval_l4066_406679


namespace NUMINAMATH_CALUDE_card_distribution_count_l4066_406651

/-- The number of ways to distribute 6 cards into 3 envelopes -/
def card_distribution : ℕ :=
  let n_cards : ℕ := 6
  let n_envelopes : ℕ := 3
  let cards_per_envelope : ℕ := 2
  let n_free_cards : ℕ := n_cards - 2  -- A and B are treated as one unit
  let ways_to_distribute_remaining : ℕ := Nat.choose n_free_cards cards_per_envelope
  let envelope_choices_for_ab : ℕ := n_envelopes
  ways_to_distribute_remaining * envelope_choices_for_ab

/-- Theorem stating that the number of card distributions is 18 -/
theorem card_distribution_count : card_distribution = 18 := by
  sorry

end NUMINAMATH_CALUDE_card_distribution_count_l4066_406651


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l4066_406674

-- Define an isosceles triangle with side lengths a, b, and c
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : (a = b ∧ c ≠ a) ∨ (b = c ∧ a ≠ b) ∨ (a = c ∧ b ≠ a)
  validTriangle : a + b > c ∧ b + c > a ∧ a + c > b

-- Define the perimeter of a triangle
def perimeter (t : IsoscelesTriangle) : ℝ := t.a + t.b + t.c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, 
    ((t.a = 3 ∧ t.b = 7) ∨ (t.a = 7 ∧ t.b = 3) ∨ 
     (t.b = 3 ∧ t.c = 7) ∨ (t.b = 7 ∧ t.c = 3) ∨ 
     (t.a = 3 ∧ t.c = 7) ∨ (t.a = 7 ∧ t.c = 3)) →
    perimeter t = 17 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l4066_406674


namespace NUMINAMATH_CALUDE_product_of_roots_plus_one_l4066_406691

theorem product_of_roots_plus_one (a b c : ℂ) : 
  (x^3 - 15*x^2 + 22*x - 8 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (1 + a) * (1 + b) * (1 + c) = 46 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_plus_one_l4066_406691


namespace NUMINAMATH_CALUDE_problem_solution_l4066_406613

theorem problem_solution (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_prod : x * y * z = 1)
  (h_eq1 : x + 1 / z = 6)
  (h_eq2 : y + 1 / x = 30) :
  z + 1 / y = 38 / 179 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l4066_406613


namespace NUMINAMATH_CALUDE_tank_capacity_l4066_406602

theorem tank_capacity (x : ℝ) (h : 0.5 * x = 75) : x = 150 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l4066_406602


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l4066_406672

theorem complex_magnitude_problem (w : ℂ) (h : w^2 = 45 - 21*I) : 
  Complex.abs w = (2466 : ℝ)^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l4066_406672


namespace NUMINAMATH_CALUDE_percentage_problem_l4066_406698

theorem percentage_problem (x : ℝ) (p : ℝ) : 
  (0.4 * x = 160) → (p * x = 120) → p = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l4066_406698


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l4066_406625

theorem quadratic_equation_roots : ∃ x y : ℝ, x ≠ y ∧ 
  x^2 - 6*x + 1 = 0 ∧ y^2 - 6*y + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l4066_406625


namespace NUMINAMATH_CALUDE_sum_of_first_5n_integers_l4066_406654

theorem sum_of_first_5n_integers (n : ℕ) : 
  (3*n*(3*n + 1))/2 = (n*(n + 1))/2 + 210 → 
  (5*n*(5*n + 1))/2 = 630 := by
sorry

end NUMINAMATH_CALUDE_sum_of_first_5n_integers_l4066_406654


namespace NUMINAMATH_CALUDE_son_work_time_l4066_406653

theorem son_work_time (man_time son_time combined_time : ℚ) : 
  man_time = 6 →
  combined_time = 3 →
  1 / man_time + 1 / son_time = 1 / combined_time →
  son_time = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_son_work_time_l4066_406653


namespace NUMINAMATH_CALUDE_parents_selection_count_l4066_406630

def number_of_students : ℕ := 6
def number_of_parents : ℕ := 12
def parents_to_choose : ℕ := 4

theorem parents_selection_count : 
  (number_of_students.choose 1) * ((number_of_parents - 2).choose 1) * ((number_of_parents - 4).choose 1) = 480 :=
by sorry

end NUMINAMATH_CALUDE_parents_selection_count_l4066_406630


namespace NUMINAMATH_CALUDE_bakery_pie_production_l4066_406670

/-- The number of pies a bakery can make in one hour, given specific pricing and profit conditions. -/
theorem bakery_pie_production (piece_price : ℚ) (pieces_per_pie : ℕ) (pie_cost : ℚ) (total_profit : ℚ) 
  (h1 : piece_price = 4)
  (h2 : pieces_per_pie = 3)
  (h3 : pie_cost = 1/2)
  (h4 : total_profit = 138) :
  (total_profit / (piece_price * ↑pieces_per_pie - pie_cost) : ℚ) = 12 := by
  sorry

end NUMINAMATH_CALUDE_bakery_pie_production_l4066_406670


namespace NUMINAMATH_CALUDE_solution_satisfies_equations_l4066_406694

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := (x^2 + 11) * Real.sqrt (21 + y^2) = 180
def equation2 (y z : ℝ) : Prop := (y^2 + 21) * Real.sqrt (z^2 - 33) = 100
def equation3 (z x : ℝ) : Prop := (z^2 - 33) * Real.sqrt (11 + x^2) = 96

-- Define the solution set
def solutionSet : Set (ℝ × ℝ × ℝ) :=
  {(5, 2, 7), (5, 2, -7), (5, -2, 7), (5, -2, -7),
   (-5, 2, 7), (-5, 2, -7), (-5, -2, 7), (-5, -2, -7)}

-- Theorem stating that all elements in the solution set satisfy the system of equations
theorem solution_satisfies_equations :
  ∀ (x y z : ℝ), (x, y, z) ∈ solutionSet →
    equation1 x y ∧ equation2 y z ∧ equation3 z x :=
by sorry

end NUMINAMATH_CALUDE_solution_satisfies_equations_l4066_406694


namespace NUMINAMATH_CALUDE_ten_point_circle_chords_l4066_406642

/-- The number of chords that can be drawn between n points on a circle's circumference,
    where no two adjacent points can be connected. -/
def restricted_chords (n : ℕ) : ℕ :=
  Nat.choose n 2 - n

theorem ten_point_circle_chords :
  restricted_chords 10 = 35 := by
  sorry

end NUMINAMATH_CALUDE_ten_point_circle_chords_l4066_406642


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l4066_406687

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- Given a geometric sequence a with a₁ = 1 and a₅ = 16, prove that a₃ = 4 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geom : IsGeometricSequence a) 
    (h_a1 : a 1 = 1) 
    (h_a5 : a 5 = 16) : 
  a 3 = 4 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_problem_l4066_406687


namespace NUMINAMATH_CALUDE_candy_box_price_increase_l4066_406608

theorem candy_box_price_increase (new_price : ℝ) (increase_rate : ℝ) (original_price : ℝ) :
  new_price = 20 ∧ increase_rate = 0.25 ∧ new_price = original_price * (1 + increase_rate) →
  original_price = 16 := by
  sorry

end NUMINAMATH_CALUDE_candy_box_price_increase_l4066_406608


namespace NUMINAMATH_CALUDE_ceiling_sqrt_count_l4066_406657

theorem ceiling_sqrt_count : 
  (Finset.range 226 \ Finset.range 197).card = 29 := by sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_count_l4066_406657


namespace NUMINAMATH_CALUDE_sandwich_not_vegetable_percentage_l4066_406648

def sandwich_weight : ℝ := 180
def vegetable_weight : ℝ := 50

theorem sandwich_not_vegetable_percentage :
  let non_vegetable_weight := sandwich_weight - vegetable_weight
  let percentage := (non_vegetable_weight / sandwich_weight) * 100
  ∃ ε > 0, |percentage - 72.22| < ε :=
sorry

end NUMINAMATH_CALUDE_sandwich_not_vegetable_percentage_l4066_406648


namespace NUMINAMATH_CALUDE_unripe_orange_harvest_l4066_406656

/-- The number of sacks of unripe oranges harvested per day -/
def daily_unripe_harvest : ℕ := 65

/-- The number of days of harvest -/
def harvest_days : ℕ := 6

/-- The total number of sacks of unripe oranges harvested over the harvest period -/
def total_unripe_harvest : ℕ := daily_unripe_harvest * harvest_days

theorem unripe_orange_harvest : total_unripe_harvest = 390 := by
  sorry

end NUMINAMATH_CALUDE_unripe_orange_harvest_l4066_406656


namespace NUMINAMATH_CALUDE_tempo_insurance_fraction_l4066_406666

/-- The fraction of the original value that a tempo is insured for -/
def insured_fraction (premium_rate : ℚ) (premium_amount : ℚ) (original_value : ℚ) : ℚ :=
  (premium_amount / premium_rate) / original_value

/-- Theorem stating that given the specific conditions, the insured fraction is 4/5 -/
theorem tempo_insurance_fraction :
  let premium_rate : ℚ := 13 / 1000
  let premium_amount : ℚ := 910
  let original_value : ℚ := 87500
  insured_fraction premium_rate premium_amount original_value = 4 / 5 := by
sorry


end NUMINAMATH_CALUDE_tempo_insurance_fraction_l4066_406666


namespace NUMINAMATH_CALUDE_symmetry_about_xOy_plane_l4066_406609

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xOy plane in 3D space -/
def xOyPlane : Set Point3D := {p : Point3D | p.z = 0}

/-- Symmetry about the xOy plane -/
def symmetricAboutXOY (p : Point3D) : Point3D :=
  ⟨p.x, p.y, -p.z⟩

theorem symmetry_about_xOy_plane (p : Point3D) :
  symmetricAboutXOY p = ⟨p.x, p.y, -p.z⟩ := by
  sorry

#check symmetry_about_xOy_plane

end NUMINAMATH_CALUDE_symmetry_about_xOy_plane_l4066_406609


namespace NUMINAMATH_CALUDE_bills_rats_l4066_406628

theorem bills_rats (total : ℕ) (ratio : ℕ) (h1 : total = 70) (h2 : ratio = 6) : 
  (ratio * total) / (ratio + 1) = 60 := by
  sorry

end NUMINAMATH_CALUDE_bills_rats_l4066_406628


namespace NUMINAMATH_CALUDE_victors_work_hours_l4066_406688

theorem victors_work_hours (hourly_rate : ℝ) (total_earnings : ℝ) (h : ℝ) : 
  hourly_rate = 6 → 
  total_earnings = 60 → 
  2 * (hourly_rate * h) = total_earnings → 
  h = 5 := by
sorry

end NUMINAMATH_CALUDE_victors_work_hours_l4066_406688


namespace NUMINAMATH_CALUDE_polynomial_root_implies_coefficients_l4066_406668

theorem polynomial_root_implies_coefficients 
  (a b : ℝ) 
  (h : (2 - 3*I : ℂ) = -a/3 - (Complex.I * Real.sqrt (Complex.normSq (2 - 3*I) - a^2/9))) :
  a = -3/2 ∧ b = 65/2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_implies_coefficients_l4066_406668


namespace NUMINAMATH_CALUDE_reaction_compound_is_chloramine_l4066_406637

/-- Represents a chemical compound --/
structure Compound where
  formula : String

/-- Represents a chemical reaction --/
structure Reaction where
  reactant : Compound
  water_amount : ℝ
  hcl_product : ℝ
  nh4oh_product : ℝ

/-- The molecular weight of water in g/mol --/
def water_molecular_weight : ℝ := 18

/-- Checks if a compound is chloramine --/
def is_chloramine (c : Compound) : Prop :=
  c.formula = "NH2Cl"

/-- Theorem stating that the compound in the reaction is chloramine --/
theorem reaction_compound_is_chloramine (r : Reaction) : 
  r.water_amount = water_molecular_weight ∧ 
  r.hcl_product = 1 ∧ 
  r.nh4oh_product = 1 → 
  is_chloramine r.reactant :=
by
  sorry


end NUMINAMATH_CALUDE_reaction_compound_is_chloramine_l4066_406637


namespace NUMINAMATH_CALUDE_class_book_count_l4066_406690

/-- Calculates the final number of books after a series of additions and subtractions. -/
def finalBookCount (initial given_away received_later traded_away received_in_trade additional : ℕ) : ℕ :=
  initial - given_away + received_later - traded_away + received_in_trade + additional

/-- Theorem stating that given the specified book counts, the final count is 93. -/
theorem class_book_count : 
  finalBookCount 54 16 23 12 9 35 = 93 := by
  sorry

end NUMINAMATH_CALUDE_class_book_count_l4066_406690


namespace NUMINAMATH_CALUDE_length_of_AC_l4066_406664

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D O : ℝ × ℝ)  -- Points in 2D plane

-- Define the conditions
def satisfies_conditions (q : Quadrilateral) : Prop :=
  let d := (λ p1 p2 : ℝ × ℝ => ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2).sqrt)
  d q.O q.A = 5 ∧
  d q.O q.C = 12 ∧
  d q.O q.B = 6 ∧
  d q.O q.D = 5 ∧
  d q.B q.D = 11

-- State the theorem
theorem length_of_AC (q : Quadrilateral) :
  satisfies_conditions q →
  let d := (λ p1 p2 : ℝ × ℝ => ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2).sqrt)
  d q.A q.C = 3 * (71 : ℝ).sqrt :=
by sorry

end NUMINAMATH_CALUDE_length_of_AC_l4066_406664


namespace NUMINAMATH_CALUDE_decimal_6_to_binary_l4066_406659

def binary_representation (n : ℕ) : List Bool :=
  sorry

theorem decimal_6_to_binary :
  binary_representation 6 = [true, true, false] :=
sorry

end NUMINAMATH_CALUDE_decimal_6_to_binary_l4066_406659


namespace NUMINAMATH_CALUDE_exists_non_integer_root_l4066_406623

theorem exists_non_integer_root (a b c : ℤ) : ∃ n : ℕ+, ¬ ∃ m : ℤ, (m : ℝ)^2 = (n : ℝ)^3 + (a : ℝ) * (n : ℝ)^2 + (b : ℝ) * (n : ℝ) + (c : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_exists_non_integer_root_l4066_406623


namespace NUMINAMATH_CALUDE_infiniteSum_equals_power_l4066_406626

/-- Number of paths from (0,0) to (k,n) satisfying the given conditions -/
def C (k n : ℕ) : ℕ := sorry

/-- The sum of C_{100j+19,17} for j from 0 to infinity -/
def infiniteSum : ℕ := sorry

/-- Theorem stating that the infinite sum equals 100^17 -/
theorem infiniteSum_equals_power : infiniteSum = 100^17 := by sorry

end NUMINAMATH_CALUDE_infiniteSum_equals_power_l4066_406626


namespace NUMINAMATH_CALUDE_pie_apples_ratio_l4066_406627

def total_apples : ℕ := 62
def refrigerator_apples : ℕ := 25
def muffin_apples : ℕ := 6

def pie_apples : ℕ := total_apples - refrigerator_apples - muffin_apples

theorem pie_apples_ratio :
  (pie_apples : ℚ) / total_apples = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_pie_apples_ratio_l4066_406627


namespace NUMINAMATH_CALUDE_orange_flower_count_l4066_406660

/-- Represents the number of flowers of each color in a garden -/
structure FlowerGarden where
  orange : ℕ
  red : ℕ
  yellow : ℕ
  pink : ℕ
  purple : ℕ

/-- Theorem stating the conditions and the result to be proved -/
theorem orange_flower_count (g : FlowerGarden) : 
  g.orange + g.red + g.yellow + g.pink + g.purple = 105 →
  g.red = 2 * g.orange →
  g.yellow = g.red - 5 →
  g.pink = g.purple →
  g.pink + g.purple = 30 →
  g.orange = 16 := by
  sorry


end NUMINAMATH_CALUDE_orange_flower_count_l4066_406660


namespace NUMINAMATH_CALUDE_trigonometric_values_l4066_406693

theorem trigonometric_values (x : ℝ) 
  (h1 : -π/2 < x ∧ x < 0) 
  (h2 : Real.sin x + Real.cos x = 7/13) : 
  (Real.sin x - Real.cos x = -17/13) ∧ 
  (4 * Real.sin x * Real.cos x - Real.cos x^2 = -384/169) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_values_l4066_406693


namespace NUMINAMATH_CALUDE_shirt_boxes_per_roll_l4066_406614

-- Define the variables
def xl_boxes_per_roll : ℕ := 3
def shirt_boxes_to_wrap : ℕ := 20
def xl_boxes_to_wrap : ℕ := 12
def cost_per_roll : ℚ := 4
def total_cost : ℚ := 32

-- Define the theorem
theorem shirt_boxes_per_roll :
  ∃ (s : ℕ), 
    s * ((total_cost / cost_per_roll) - (xl_boxes_to_wrap / xl_boxes_per_roll)) = shirt_boxes_to_wrap ∧ 
    s = 5 := by
  sorry

end NUMINAMATH_CALUDE_shirt_boxes_per_roll_l4066_406614


namespace NUMINAMATH_CALUDE_shirley_sold_54_boxes_l4066_406650

/-- The number of cases Shirley needs to deliver -/
def num_cases : ℕ := 9

/-- The number of boxes in each case -/
def boxes_per_case : ℕ := 6

/-- The total number of boxes Shirley sold -/
def total_boxes : ℕ := num_cases * boxes_per_case

theorem shirley_sold_54_boxes : total_boxes = 54 := by
  sorry

end NUMINAMATH_CALUDE_shirley_sold_54_boxes_l4066_406650


namespace NUMINAMATH_CALUDE_boys_percentage_in_class_l4066_406692

theorem boys_percentage_in_class (total_students : ℕ) (boys_ratio girls_ratio : ℕ) 
  (h1 : total_students = 42)
  (h2 : boys_ratio = 3)
  (h3 : girls_ratio = 4) :
  (boys_ratio * total_students : ℚ) / ((boys_ratio + girls_ratio) * 100) = 42857 / 100000 :=
by sorry

end NUMINAMATH_CALUDE_boys_percentage_in_class_l4066_406692


namespace NUMINAMATH_CALUDE_race_outcomes_count_l4066_406624

/-- The number of participants in the race -/
def num_participants : Nat := 5

/-- The number of podium positions (1st, 2nd, 3rd) -/
def num_podium_positions : Nat := 3

/-- Calculate the number of permutations of k items chosen from n items -/
def permutations (n k : Nat) : Nat :=
  Nat.factorial n / Nat.factorial (n - k)

/-- The theorem stating that the number of different 1st-2nd-3rd place outcomes
    in a race with 5 participants and no ties is equal to 60 -/
theorem race_outcomes_count : 
  permutations num_participants num_podium_positions = 60 := by
  sorry


end NUMINAMATH_CALUDE_race_outcomes_count_l4066_406624


namespace NUMINAMATH_CALUDE_negation_constant_geometric_sequence_l4066_406633

theorem negation_constant_geometric_sequence :
  ¬(∀ (a : ℕ → ℝ), (∀ n : ℕ, a n = a 0) → (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n)) ↔
  (∃ (a : ℕ → ℝ), (∀ n : ℕ, a n = a 0) ∧ ¬(∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n)) :=
by sorry

end NUMINAMATH_CALUDE_negation_constant_geometric_sequence_l4066_406633


namespace NUMINAMATH_CALUDE_emmas_age_l4066_406678

/-- Given the ages and relationships between Jose, Zack, Inez, and Emma, prove Emma's age --/
theorem emmas_age (jose_age : ℕ) (zack_age : ℕ) (inez_age : ℕ) (emma_age : ℕ)
  (h1 : jose_age = 20)
  (h2 : zack_age = jose_age + 4)
  (h3 : inez_age = zack_age - 12)
  (h4 : emma_age = jose_age + 5) :
  emma_age = 25 := by
  sorry


end NUMINAMATH_CALUDE_emmas_age_l4066_406678


namespace NUMINAMATH_CALUDE_age_difference_proof_l4066_406681

/-- Given the ages of Katie's daughter, Lavinia's daughter, and Lavinia's son, prove that Lavinia's son is 22 years older than Lavinia's daughter. -/
theorem age_difference_proof (katie_daughter_age lavinia_daughter_age lavinia_son_age : ℕ) :
  katie_daughter_age = 12 →
  lavinia_daughter_age = katie_daughter_age - 10 →
  lavinia_son_age = 2 * katie_daughter_age →
  lavinia_son_age - lavinia_daughter_age = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l4066_406681


namespace NUMINAMATH_CALUDE_roses_cut_l4066_406673

theorem roses_cut (initial_roses initial_orchids final_roses final_orchids : ℕ) 
  (h1 : initial_roses = 13)
  (h2 : initial_orchids = 84)
  (h3 : final_roses = 14)
  (h4 : final_orchids = 91) :
  final_roses - initial_roses = 1 := by
  sorry

end NUMINAMATH_CALUDE_roses_cut_l4066_406673


namespace NUMINAMATH_CALUDE_integer_solution_exists_l4066_406618

theorem integer_solution_exists (n : ℤ) : ∃ (a b c : ℤ), 
  a ≠ 0 ∧ 
  (a = b + c ∨ b = a + c ∨ c = a + b) ∧
  a * n + b = c :=
by sorry

end NUMINAMATH_CALUDE_integer_solution_exists_l4066_406618


namespace NUMINAMATH_CALUDE_rectangle_y_value_l4066_406620

/-- Given a rectangle with vertices (-3, y), (1, y), (1, -2), and (-3, -2),
    if the area of the rectangle is 12, then y = 1. -/
theorem rectangle_y_value (y : ℝ) : 
  let vertices := [(-3, y), (1, y), (1, -2), (-3, -2)]
  let length := 1 - (-3)
  let height := y - (-2)
  let area := length * height
  area = 12 → y = 1 := by
sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l4066_406620


namespace NUMINAMATH_CALUDE_nabla_problem_l4066_406611

-- Define the ∇ operation
def nabla (a b : ℕ) : ℕ := 3 + b^a

-- Theorem statement
theorem nabla_problem : nabla (nabla 2 3) 4 = 16777219 := by
  sorry

end NUMINAMATH_CALUDE_nabla_problem_l4066_406611


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l4066_406644

/-- Given a polynomial P(z) = 4z^3 - 5z^2 - 19z + 4, when divided by 4z + 6
    with quotient z^2 - 4z + 1, prove that the remainder is 5z^2 + z - 2. -/
theorem polynomial_division_remainder
  (z : ℂ)
  (P : ℂ → ℂ)
  (D : ℂ → ℂ)
  (Q : ℂ → ℂ)
  (h1 : P z = 4 * z^3 - 5 * z^2 - 19 * z + 4)
  (h2 : D z = 4 * z + 6)
  (h3 : Q z = z^2 - 4 * z + 1)
  : ∃ R : ℂ → ℂ, P z = D z * Q z + R z ∧ R z = 5 * z^2 + z - 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l4066_406644


namespace NUMINAMATH_CALUDE_sum_of_even_factors_720_l4066_406676

def sum_of_even_factors (n : ℕ) : ℕ := sorry

theorem sum_of_even_factors_720 : sum_of_even_factors 720 = 2340 := by sorry

end NUMINAMATH_CALUDE_sum_of_even_factors_720_l4066_406676


namespace NUMINAMATH_CALUDE_math_club_team_probability_l4066_406612

theorem math_club_team_probability :
  let total_girls : ℕ := 8
  let total_boys : ℕ := 6
  let team_size : ℕ := 4
  let girls_in_team : ℕ := 2
  let boys_in_team : ℕ := 2

  (Nat.choose total_girls girls_in_team * Nat.choose total_boys boys_in_team) /
  Nat.choose (total_girls + total_boys) team_size = 60 / 143 :=
by sorry

end NUMINAMATH_CALUDE_math_club_team_probability_l4066_406612


namespace NUMINAMATH_CALUDE_boxes_to_fill_l4066_406663

theorem boxes_to_fill (total_boxes filled_boxes : ℝ) 
  (h1 : total_boxes = 25.75) 
  (h2 : filled_boxes = 17.5) : 
  total_boxes - filled_boxes = 8.25 := by
  sorry

end NUMINAMATH_CALUDE_boxes_to_fill_l4066_406663
