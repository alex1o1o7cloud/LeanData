import Mathlib

namespace NUMINAMATH_CALUDE_unbroken_matches_count_l2872_287246

def dozen : ℕ := 12
def boxes_count : ℕ := 5 * dozen
def matches_per_box : ℕ := 20
def broken_matches_per_box : ℕ := 3

theorem unbroken_matches_count :
  boxes_count * (matches_per_box - broken_matches_per_box) = 1020 :=
by sorry

end NUMINAMATH_CALUDE_unbroken_matches_count_l2872_287246


namespace NUMINAMATH_CALUDE_f_property_l2872_287211

theorem f_property (f : ℝ → ℝ) 
  (h : ∀ x y z : ℝ, f (x^2 + y * f z + z) = x * f x + z * f y + f z) :
  (∃ a b : ℝ, (∀ x : ℝ, f x = a ∨ f x = b) ∧ f 5 = a ∨ f 5 = b) ∧
  (∃ s : ℝ, s = (f 5 + f 5) ∧ s = 5) :=
sorry

end NUMINAMATH_CALUDE_f_property_l2872_287211


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l2872_287292

theorem decimal_to_fraction : (3.56 : ℚ) = 89 / 25 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l2872_287292


namespace NUMINAMATH_CALUDE_set_operations_l2872_287263

-- Define the sets A and B
def A : Set ℝ := {x | x - 2 < 0}
def B : Set ℝ := {x | -1 < x ∧ x < 1}

-- Theorem statement
theorem set_operations :
  (A ∩ B = B) ∧
  (B ⊆ A) ∧
  (A \ B = {x | x ≤ -1 ∨ (1 ≤ x ∧ x < 2)}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2872_287263


namespace NUMINAMATH_CALUDE_incorrect_statement_l2872_287226

-- Define propositions P and Q
def P : Prop := 2 + 2 = 5
def Q : Prop := 3 > 2

-- Theorem stating that the incorrect statement is "'P and Q' is false, 'not P' is false"
theorem incorrect_statement :
  ¬((P ∧ Q → False) ∧ (¬P → False)) :=
by
  sorry


end NUMINAMATH_CALUDE_incorrect_statement_l2872_287226


namespace NUMINAMATH_CALUDE_triangle_problem_l2872_287251

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Given conditions
  A = π / 4 →
  b = Real.sqrt 6 →
  (1 / 2) * b * c * Real.sin A = (3 + Real.sqrt 3) / 2 →
  -- Definitions from cosine rule
  a = Real.sqrt (b^2 + c^2 - 2*b*c*(Real.cos A)) →
  Real.cos B = (a^2 + c^2 - b^2) / (2*a*c) →
  -- Conclusion
  c = 1 + Real.sqrt 3 ∧ B = π / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l2872_287251


namespace NUMINAMATH_CALUDE_video_game_enemies_l2872_287213

/-- The number of points earned per defeated enemy -/
def points_per_enemy : ℕ := 3

/-- The number of enemies left undefeated -/
def undefeated_enemies : ℕ := 2

/-- The total points earned -/
def total_points : ℕ := 12

/-- The total number of enemies in the level -/
def total_enemies : ℕ := 6

theorem video_game_enemies :
  total_enemies = (total_points / points_per_enemy) + undefeated_enemies :=
sorry

end NUMINAMATH_CALUDE_video_game_enemies_l2872_287213


namespace NUMINAMATH_CALUDE_different_colors_probability_l2872_287247

structure Box where
  red : ℕ
  black : ℕ
  white : ℕ
  yellow : ℕ

def boxA : Box := { red := 3, black := 3, white := 3, yellow := 0 }
def boxB : Box := { red := 0, black := 2, white := 2, yellow := 2 }

def totalBalls (box : Box) : ℕ := box.red + box.black + box.white + box.yellow

def probabilityDifferentColors (boxA boxB : Box) : ℚ :=
  let totalA := totalBalls boxA
  let totalB := totalBalls boxB
  let sameColor := boxA.black * boxB.black + boxA.white * boxB.white
  (totalA * totalB - sameColor) / (totalA * totalB)

theorem different_colors_probability :
  probabilityDifferentColors boxA boxB = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_different_colors_probability_l2872_287247


namespace NUMINAMATH_CALUDE_f_composition_of_one_l2872_287202

def f (x : ℤ) : ℤ :=
  if x % 2 = 0 then x / 3 else 4 * x + 1

theorem f_composition_of_one : f (f (f (f 1))) = 341 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_of_one_l2872_287202


namespace NUMINAMATH_CALUDE_drawing_red_ball_certain_l2872_287214

/-- A bag containing only red balls -/
structure RedBallBag where
  num_balls : ℕ
  all_red : True

/-- The probability of drawing a red ball from a bag of red balls -/
def prob_draw_red (bag : RedBallBag) : ℝ :=
  1

/-- An event is certain if its probability is 1 -/
def is_certain_event (p : ℝ) : Prop :=
  p = 1

/-- Theorem: Drawing a red ball from a bag containing only 5 red balls is a certain event -/
theorem drawing_red_ball_certain (bag : RedBallBag) (h : bag.num_balls = 5) :
    is_certain_event (prob_draw_red bag) := by
  sorry

end NUMINAMATH_CALUDE_drawing_red_ball_certain_l2872_287214


namespace NUMINAMATH_CALUDE_quadratic_inequality_equiv_interval_l2872_287299

theorem quadratic_inequality_equiv_interval (x : ℝ) :
  x^2 - 8*x + 15 < 0 ↔ 3 < x ∧ x < 5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equiv_interval_l2872_287299


namespace NUMINAMATH_CALUDE_trisection_intersection_l2872_287257

/-- Given two points on the natural logarithm curve, prove that the x-coordinate of the 
    intersection point between a horizontal line through the first trisection point and 
    the curve is 2^(7/3). -/
theorem trisection_intersection (A B C : ℝ × ℝ) : 
  A.1 = 2 → 
  A.2 = Real.log 2 →
  B.1 = 32 → 
  B.2 = Real.log 32 →
  C.2 = (2 / 3) * A.2 + (1 / 3) * B.2 →
  ∃ (x : ℝ), x > 0 ∧ Real.log x = C.2 →
  x = 2^(7/3) := by
sorry

end NUMINAMATH_CALUDE_trisection_intersection_l2872_287257


namespace NUMINAMATH_CALUDE_inscribed_tetrahedron_volume_bound_l2872_287295

/-- A right circular cylinder with volume 1 -/
structure RightCircularCylinder where
  radius : ℝ
  height : ℝ
  volume_eq_one : π * radius^2 * height = 1

/-- A tetrahedron inscribed in a right circular cylinder -/
structure InscribedTetrahedron (c : RightCircularCylinder) where
  volume : ℝ
  is_inscribed : volume ≤ π * c.radius^2 * c.height

/-- The volume of any tetrahedron inscribed in a right circular cylinder 
    with volume 1 does not exceed 2/(3π) -/
theorem inscribed_tetrahedron_volume_bound 
  (c : RightCircularCylinder) 
  (t : InscribedTetrahedron c) : 
  t.volume ≤ 2 / (3 * π) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_tetrahedron_volume_bound_l2872_287295


namespace NUMINAMATH_CALUDE_success_arrangements_l2872_287241

def word_length : ℕ := 7
def s_count : ℕ := 3
def c_count : ℕ := 2
def u_count : ℕ := 1
def e_count : ℕ := 1

theorem success_arrangements : 
  (word_length.factorial) / (s_count.factorial * c_count.factorial * u_count.factorial * e_count.factorial) = 420 :=
sorry

end NUMINAMATH_CALUDE_success_arrangements_l2872_287241


namespace NUMINAMATH_CALUDE_jerry_lawn_mowing_money_l2872_287229

/-- The amount of money Jerry made mowing lawns -/
def M : ℝ := sorry

/-- The amount of money Jerry made from weed eating -/
def weed_eating_money : ℝ := 31

/-- The number of weeks Jerry's money would last -/
def weeks : ℝ := 9

/-- The amount Jerry would spend per week -/
def weekly_spending : ℝ := 5

theorem jerry_lawn_mowing_money :
  M = 14 :=
by
  have total_money : M + weed_eating_money = weeks * weekly_spending := by sorry
  sorry

end NUMINAMATH_CALUDE_jerry_lawn_mowing_money_l2872_287229


namespace NUMINAMATH_CALUDE_odd_operations_l2872_287261

theorem odd_operations (a b : ℤ) (h_even : Even a) (h_odd : Odd b) :
  Odd (a + b) ∧ Odd (a - b) ∧ Odd ((a + b)^2) ∧ 
  ¬(∀ (a b : ℤ), Even a → Odd b → Odd (a * b)) ∧
  ¬(∀ (a b : ℤ), Even a → Odd b → Odd ((a + b) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_odd_operations_l2872_287261


namespace NUMINAMATH_CALUDE_range_of_a_l2872_287298

/-- The inequality (a-3)x^2 + 2(a-3)x - 4 < 0 has a solution set of all real numbers for x -/
def has_all_real_solutions (a : ℝ) : Prop :=
  ∀ x : ℝ, (a - 3) * x^2 + 2 * (a - 3) * x - 4 < 0

/-- The main theorem stating the range of a -/
theorem range_of_a (a : ℝ) : 
  has_all_real_solutions a ↔ -1 < a ∧ a < 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2872_287298


namespace NUMINAMATH_CALUDE_tetrahedron_surface_area_l2872_287210

/-- Given a regular tetrahedron with a square cross-section of area m², 
    its surface area is 4m²√3. -/
theorem tetrahedron_surface_area (m : ℝ) (h : m > 0) : 
  let square_area : ℝ := m^2
  let tetrahedron_surface_area : ℝ := 4 * m^2 * Real.sqrt 3
  square_area = m^2 → tetrahedron_surface_area = 4 * m^2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_surface_area_l2872_287210


namespace NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l2872_287291

theorem unique_solution_for_exponential_equation :
  ∃! (x : ℝ), x ≠ 0 ∧ (9 * x)^18 = (27 * x)^9 ∧ x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l2872_287291


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l2872_287244

/-- Given a line segment with midpoint (3, 4) and one endpoint (7, 10), 
    prove that the other endpoint is (-1, -2). -/
theorem line_segment_endpoint (M A B : ℝ × ℝ) : 
  M = (3, 4) → A = (7, 10) → M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → B = (-1, -2) := by
  sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l2872_287244


namespace NUMINAMATH_CALUDE_flower_pots_distance_l2872_287208

/-- Given 8 equally spaced points on a line, if the distance between
    the first and fifth points is 100, then the distance between
    the first and eighth points is 175. -/
theorem flower_pots_distance (points : Fin 8 → ℝ) 
    (equally_spaced : ∀ i j k : Fin 8, i.val < j.val → j.val < k.val → 
      points k - points j = points j - points i)
    (dist_1_5 : points 4 - points 0 = 100) :
    points 7 - points 0 = 175 := by
  sorry


end NUMINAMATH_CALUDE_flower_pots_distance_l2872_287208


namespace NUMINAMATH_CALUDE_stability_comparison_l2872_287204

/-- Represents an athlete's performance in a series of tests -/
structure AthletePerformance where
  average_score : ℝ
  variance : ℝ

/-- Defines stability of performance based on variance -/
def more_stable (a b : AthletePerformance) : Prop :=
  a.variance < b.variance

/-- Theorem: Given two athletes with the same average score,
    the one with lower variance has more stable performance -/
theorem stability_comparison 
  (athlete_A athlete_B : AthletePerformance)
  (h_same_average : athlete_A.average_score = athlete_B.average_score)
  (h_A_variance : athlete_A.variance = 1.2)
  (h_B_variance : athlete_B.variance = 1) :
  more_stable athlete_B athlete_A :=
sorry

end NUMINAMATH_CALUDE_stability_comparison_l2872_287204


namespace NUMINAMATH_CALUDE_f_properties_l2872_287220

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|

-- Theorem statement
theorem f_properties : 
  (∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min ∧ f x_min = 4) ∧ 
  (∀ (a : ℝ), (∃ (x : ℝ), f x < a) ↔ a > 4) ∧
  (∀ (a b : ℝ), (∀ (x : ℝ), f x < a ↔ b < x ∧ x < 7/2) → a + b = 3.5) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2872_287220


namespace NUMINAMATH_CALUDE_bridge_length_l2872_287274

/-- The length of a bridge given train parameters and crossing time -/
theorem bridge_length
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 125)
  (h2 : train_speed_kmh = 45)
  (h3 : crossing_time = 30)
  : ∃ (bridge_length : ℝ), bridge_length = 250 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l2872_287274


namespace NUMINAMATH_CALUDE_no_solution_exists_l2872_287224

-- Define the system of equations
def system (a b c d : ℝ) : Prop :=
  a^3 + c^3 = 2 ∧
  a^2*b + c^2*d = 0 ∧
  b^3 + d^3 = 1 ∧
  a*b^2 + c*d^2 = -6

-- Theorem stating that no solution exists
theorem no_solution_exists : ¬∃ (a b c d : ℝ), system a b c d := by
  sorry


end NUMINAMATH_CALUDE_no_solution_exists_l2872_287224


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l2872_287279

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant -/
def isInFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Definition of the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Theorem: If A(m,n) is in the first quadrant, then B(-m,-n) is in the third quadrant -/
theorem point_in_third_quadrant
  (A : Point)
  (hA : isInFirstQuadrant A) :
  isInThirdQuadrant (Point.mk (-A.x) (-A.y)) :=
by sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l2872_287279


namespace NUMINAMATH_CALUDE_circle_center_is_3_neg4_l2872_287268

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 + 8*y - 12 = 0

/-- The center of a circle -/
def circle_center (h k : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = (x^2 - 6*x + y^2 + 8*y - 12 + 37) / 2

theorem circle_center_is_3_neg4 : circle_center 3 (-4) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_is_3_neg4_l2872_287268


namespace NUMINAMATH_CALUDE_consecutive_zeros_in_prime_power_l2872_287273

theorem consecutive_zeros_in_prime_power (p : Nat) (h : Nat.Prime p) :
  ∃ n : Nat, n > 0 ∧ p^n % 10^2002 = 0 ∧ p^n % 10^2003 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_zeros_in_prime_power_l2872_287273


namespace NUMINAMATH_CALUDE_square_root_of_sixteen_l2872_287237

theorem square_root_of_sixteen : 
  {x : ℝ | x^2 = 16} = {4, -4} := by sorry

end NUMINAMATH_CALUDE_square_root_of_sixteen_l2872_287237


namespace NUMINAMATH_CALUDE_pet_store_puppies_l2872_287248

theorem pet_store_puppies (initial_kittens : ℕ) (sold_puppies sold_kittens remaining_pets : ℕ) 
  (h1 : initial_kittens = 6)
  (h2 : sold_puppies = 2)
  (h3 : sold_kittens = 3)
  (h4 : remaining_pets = 8) :
  ∃ initial_puppies : ℕ, 
    initial_puppies - sold_puppies + initial_kittens - sold_kittens = remaining_pets ∧ 
    initial_puppies = 7 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_puppies_l2872_287248


namespace NUMINAMATH_CALUDE_intersection_area_greater_than_half_l2872_287228

/-- Represents a rectangle in a 2D plane -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents the intersection of two rectangles -/
structure Intersection (r1 r2 : Rectangle) where
  area : ℝ

/-- Theorem: Given two equal rectangles whose contours intersect at 8 points,
    the area of their intersection is greater than half the area of each rectangle -/
theorem intersection_area_greater_than_half 
  (r1 r2 : Rectangle) 
  (h_equal : r1 = r2) 
  (h_intersect : ∃ (pts : Finset (ℝ × ℝ)), pts.card = 8) 
  (i : Intersection r1 r2) : 
  i.area > (1/2) * r1.area := by
  sorry

end NUMINAMATH_CALUDE_intersection_area_greater_than_half_l2872_287228


namespace NUMINAMATH_CALUDE_historical_fiction_new_release_fraction_is_four_sevenths_l2872_287206

/-- Represents the inventory of a bookstore -/
structure BookstoreInventory where
  total_books : ℕ
  historical_fiction_ratio : ℚ
  historical_fiction_new_release_ratio : ℚ
  other_new_release_ratio : ℚ

/-- Calculates the fraction of new releases that are historical fiction -/
def historical_fiction_new_release_fraction (inventory : BookstoreInventory) : ℚ :=
  let historical_fiction := inventory.total_books * inventory.historical_fiction_ratio
  let other_books := inventory.total_books * (1 - inventory.historical_fiction_ratio)
  let historical_fiction_new_releases := historical_fiction * inventory.historical_fiction_new_release_ratio
  let other_new_releases := other_books * inventory.other_new_release_ratio
  historical_fiction_new_releases / (historical_fiction_new_releases + other_new_releases)

/-- Theorem stating that the fraction of new releases that are historical fiction is 4/7 -/
theorem historical_fiction_new_release_fraction_is_four_sevenths
  (inventory : BookstoreInventory)
  (h1 : inventory.historical_fiction_ratio = 2/5)
  (h2 : inventory.historical_fiction_new_release_ratio = 2/5)
  (h3 : inventory.other_new_release_ratio = 1/5) :
  historical_fiction_new_release_fraction inventory = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_historical_fiction_new_release_fraction_is_four_sevenths_l2872_287206


namespace NUMINAMATH_CALUDE_complex_number_problem_l2872_287234

theorem complex_number_problem (a b : ℝ) (i : ℂ) : 
  (a - 2*i) * i = b - i → a^2 + b^2 = 5 := by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2872_287234


namespace NUMINAMATH_CALUDE_fraction_difference_equals_one_l2872_287231

theorem fraction_difference_equals_one (x y : ℝ) (h : x ≠ y) :
  x / (x - y) - y / (x - y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_equals_one_l2872_287231


namespace NUMINAMATH_CALUDE_triangle_side_ratio_maximum_l2872_287289

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and the area of the triangle is (1/2)c^2, the maximum value of (a^2 + b^2 + c^2) / (ab) is 2√2. -/
theorem triangle_side_ratio_maximum (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  (1/2) * c^2 = (1/2) * a * b * Real.sin C →
  (∃ (x : ℝ), (a^2 + b^2 + c^2) / (a * b) ≤ x) ∧
  (∀ (x : ℝ), (a^2 + b^2 + c^2) / (a * b) ≤ x → x ≥ 2 * Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_triangle_side_ratio_maximum_l2872_287289


namespace NUMINAMATH_CALUDE_group_average_score_l2872_287287

theorem group_average_score (class_average : ℝ) (differences : List ℝ) : 
  class_average = 80 →
  differences = [2, 3, -3, -5, 12, 14, 10, 4, -6, 4, -11, -7, 8, -2] →
  (class_average + (differences.sum / differences.length)) = 81.64 := by
sorry

end NUMINAMATH_CALUDE_group_average_score_l2872_287287


namespace NUMINAMATH_CALUDE_mat_weavers_problem_l2872_287217

/-- Given that 4 mat-weavers can weave 4 mats in 4 days, prove that 12 mat-weavers
    are needed to weave 36 mats in 12 days. -/
theorem mat_weavers_problem (weavers_group1 mats_group1 days_group1 : ℕ)
                             (mats_group2 days_group2 : ℕ) :
  weavers_group1 = 4 →
  mats_group1 = 4 →
  days_group1 = 4 →
  mats_group2 = 36 →
  days_group2 = 12 →
  (weavers_group1 * mats_group2 * days_group1 = mats_group1 * days_group2 * 12) :=
by sorry

end NUMINAMATH_CALUDE_mat_weavers_problem_l2872_287217


namespace NUMINAMATH_CALUDE_probability_of_sum_25_l2872_287245

/-- Represents a die with numbered faces and a blank face -/
structure Die where
  faces : ℕ
  numbers : List ℕ
  blank_faces : ℕ

/-- Calculates the probability of a specific sum when rolling two dice -/
def probability_of_sum (die1 die2 : Die) (target_sum : ℕ) : ℚ :=
  sorry

/-- The first die with 19 faces numbered 1 through 18 and one blank face -/
def first_die : Die :=
  { faces := 20,
    numbers := List.range 18,
    blank_faces := 1 }

/-- The second die with 19 faces numbered 2 through 9 and 11 through 21 and one blank face -/
def second_die : Die :=
  { faces := 20,
    numbers := (List.range 8).map (· + 2) ++ (List.range 11).map (· + 11),
    blank_faces := 1 }

/-- Theorem stating the probability of rolling a sum of 25 with the given dice -/
theorem probability_of_sum_25 :
  probability_of_sum first_die second_die 25 = 3 / 80 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_sum_25_l2872_287245


namespace NUMINAMATH_CALUDE_line_through_points_l2872_287282

/-- Given a line y = ax + b passing through points (3, 2) and (7, 26), prove that a - b = 22 -/
theorem line_through_points (a b : ℝ) : 
  (2 : ℝ) = a * 3 + b ∧ (26 : ℝ) = a * 7 + b → a - b = 22 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l2872_287282


namespace NUMINAMATH_CALUDE_divisibility_implication_l2872_287270

theorem divisibility_implication (a b : ℕ) (h1 : a < 1000) (h2 : b^10 ∣ a^21) : b ∣ a^2 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implication_l2872_287270


namespace NUMINAMATH_CALUDE_first_sequence_6th_7th_terms_l2872_287267

def first_sequence : ℕ → ℕ
  | 0 => 3
  | n + 1 => 2 * first_sequence n + 1

theorem first_sequence_6th_7th_terms :
  first_sequence 5 = 127 ∧ first_sequence 6 = 255 := by
  sorry

end NUMINAMATH_CALUDE_first_sequence_6th_7th_terms_l2872_287267


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l2872_287240

theorem right_triangle_perimeter (a b c r : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 →
  c = 10 → r = 1 →
  a^2 + b^2 = c^2 →
  (a + b - c) * r = a * b / 2 →
  a + b + c = 24 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l2872_287240


namespace NUMINAMATH_CALUDE_persimmons_count_l2872_287296

theorem persimmons_count (tangerines : ℕ) (total : ℕ) (h1 : tangerines = 19) (h2 : total = 37) :
  total - tangerines = 18 := by
  sorry

end NUMINAMATH_CALUDE_persimmons_count_l2872_287296


namespace NUMINAMATH_CALUDE_curve_C₂_equation_l2872_287221

-- Define the parabola C₁
def C₁ (x y : ℝ) : Prop := y = (1/20) * x^2

-- Define the focus F of C₁
def F : ℝ × ℝ := (0, 5)

-- Define point E symmetric to F with respect to the origin
def E : ℝ × ℝ := (0, -5)

-- Define the property of points on C₂
def on_C₂ (x y : ℝ) : Prop :=
  abs (Real.sqrt ((x - E.1)^2 + (y - E.2)^2) - Real.sqrt ((x - F.1)^2 + (y - F.2)^2)) = 6

-- Theorem statement
theorem curve_C₂_equation :
  ∀ x y : ℝ, on_C₂ x y ↔ y^2/9 - x^2/16 = 1 :=
sorry

end NUMINAMATH_CALUDE_curve_C₂_equation_l2872_287221


namespace NUMINAMATH_CALUDE_equation_solution_l2872_287238

theorem equation_solution (c d : ℝ) : 
  (∃ x : ℝ, x^2 - 6*x + 15 = 27 ∧ (x = c ∨ x = d)) →
  c ≥ d →
  3*c - d = 6 + 4*Real.sqrt 21 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2872_287238


namespace NUMINAMATH_CALUDE_peach_difference_l2872_287258

theorem peach_difference (jill_peaches steven_peaches jake_peaches : ℕ) : 
  jill_peaches = 87 →
  steven_peaches = jill_peaches + 18 →
  jake_peaches = jill_peaches + 13 →
  steven_peaches - jake_peaches = 5 := by
sorry

end NUMINAMATH_CALUDE_peach_difference_l2872_287258


namespace NUMINAMATH_CALUDE_circle_and_line_intersection_l2872_287233

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 4)^2 = 25

-- Define the moving line l
def line_l (m x y : ℝ) : Prop :=
  (m + 2) * x + (2 * m + 1) * y - 7 * m - 8 = 0

-- Theorem statement
theorem circle_and_line_intersection :
  -- Given conditions
  (circle_C (-2) 1) ∧  -- Circle C passes through A(-2, 1)
  (circle_C 5 0) ∧     -- Circle C passes through B(5, 0)
  (∃ x y : ℝ, circle_C x y ∧ y = 2 * x) →  -- Center of C is on y = 2x
  -- Conclusions
  ((∀ x y : ℝ, circle_C x y ↔ (x - 2)^2 + (y - 4)^2 = 25) ∧
   (∃ min_PQ : ℝ, 
     (min_PQ = 4 * Real.sqrt 5) ∧
     (∀ m x1 y1 x2 y2 : ℝ,
       (circle_C x1 y1 ∧ circle_C x2 y2 ∧ 
        line_l m x1 y1 ∧ line_l m x2 y2) →
       ((x1 - x2)^2 + (y1 - y2)^2 ≥ min_PQ^2))))
  := by sorry

end NUMINAMATH_CALUDE_circle_and_line_intersection_l2872_287233


namespace NUMINAMATH_CALUDE_square_area_on_parabola_l2872_287252

/-- The area of a square with one side on y = 5 and endpoints on y = x^2 + 3x + 2 is 21 -/
theorem square_area_on_parabola : ∃ (x₁ x₂ : ℝ),
  (x₁^2 + 3*x₁ + 2 = 5) ∧
  (x₂^2 + 3*x₂ + 2 = 5) ∧
  (x₁ ≠ x₂) ∧
  ((x₂ - x₁)^2 = 21) := by
  sorry

end NUMINAMATH_CALUDE_square_area_on_parabola_l2872_287252


namespace NUMINAMATH_CALUDE_sum_of_d_μ_M_equals_59_22_l2872_287232

-- Define the data distribution
def data_distribution : List (Nat × Nat) :=
  (List.range 28).map (fun x => (x + 1, 24)) ++
  [(29, 22), (30, 22), (31, 14)]

-- Define the total number of data points
def total_count : Nat :=
  data_distribution.foldl (fun acc (_, count) => acc + count) 0

-- Define the median of modes
def d : ℝ := 14.5

-- Define the median of the entire dataset
def M : ℝ := 29

-- Define the mean of the entire dataset
noncomputable def μ : ℝ :=
  let sum := data_distribution.foldl (fun acc (value, count) => acc + value * count) 0
  (sum : ℝ) / total_count

-- Theorem statement
theorem sum_of_d_μ_M_equals_59_22 :
  d + μ + M = 59.22 := by sorry

end NUMINAMATH_CALUDE_sum_of_d_μ_M_equals_59_22_l2872_287232


namespace NUMINAMATH_CALUDE_matrix_power_four_l2872_287255

/-- Given a 2x2 matrix A, prove that A^4 equals the given result. -/
theorem matrix_power_four (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A = !![3 * Real.sqrt 2, -3; 3, 3 * Real.sqrt 2] →
  A ^ 4 = !![-81, 0; 0, -81] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_four_l2872_287255


namespace NUMINAMATH_CALUDE_parallel_line_plane_condition_l2872_287269

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes and lines
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the subset relation for lines and planes
variable (subset_line_plane : Line → Plane → Prop)

-- State the theorem
theorem parallel_line_plane_condition 
  (α β : Plane) (m : Line) :
  parallel_planes α β → 
  ¬subset_line_plane m β → 
  parallel_line_plane m α :=
sorry

end NUMINAMATH_CALUDE_parallel_line_plane_condition_l2872_287269


namespace NUMINAMATH_CALUDE_intersection_line_equation_l2872_287242

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def line_through_intersections (c1 c2 : Circle) : ℝ → ℝ → Prop :=
  fun x y => x + y = 26/3

theorem intersection_line_equation :
  let c1 : Circle := ⟨(2, -3), 10⟩
  let c2 : Circle := ⟨(-4, 7), 6⟩
  ∀ x y : ℝ,
    (x - c1.center.1)^2 + (y - c1.center.2)^2 = c1.radius^2 ∧
    (x - c2.center.1)^2 + (y - c2.center.2)^2 = c2.radius^2 →
    line_through_intersections c1 c2 x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l2872_287242


namespace NUMINAMATH_CALUDE_sum_of_digits_main_expression_l2872_287297

/-- Represents a string of digits --/
structure DigitString :=
  (length : ℕ)
  (digit : ℕ)
  (digit_valid : digit < 10)

/-- Calculates the product of two DigitStrings --/
def multiply_digit_strings (a b : DigitString) : ℕ := sorry

/-- Calculates the sum of digits in a natural number --/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Represents the expression (80 eights × 80 fives + 80 ones) --/
def main_expression : ℕ :=
  let eights : DigitString := ⟨80, 8, by norm_num⟩
  let fives : DigitString := ⟨80, 5, by norm_num⟩
  let ones : DigitString := ⟨80, 1, by norm_num⟩
  multiply_digit_strings eights fives + ones.length * ones.digit

/-- The main theorem to be proved --/
theorem sum_of_digits_main_expression :
  sum_of_digits main_expression = 400 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_main_expression_l2872_287297


namespace NUMINAMATH_CALUDE_sin_120_degrees_l2872_287262

theorem sin_120_degrees : Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_120_degrees_l2872_287262


namespace NUMINAMATH_CALUDE_min_abs_GB_is_392_l2872_287283

-- Define the Revolution polynomial
def Revolution (U S A : ℤ) (x : ℤ) : ℤ := x^3 + U*x^2 + S*x + A

-- State the theorem
theorem min_abs_GB_is_392 
  (U S A G B : ℤ) 
  (h1 : U + S + A + 1 = 1773)
  (h2 : ∀ x, Revolution U S A x = 0 ↔ x = G ∨ x = B)
  (h3 : G ≠ B)
  (h4 : G ≠ 0)
  (h5 : B ≠ 0) :
  ∃ (G' B' : ℤ), G' * B' = 392 ∧ 
    ∀ (G'' B'' : ℤ), G'' ≠ 0 ∧ B'' ≠ 0 ∧ 
      (∀ x, Revolution U S A x = 0 ↔ x = G'' ∨ x = B'') → 
      abs (G'' * B'') ≥ 392 :=
sorry

end NUMINAMATH_CALUDE_min_abs_GB_is_392_l2872_287283


namespace NUMINAMATH_CALUDE_subset_implies_a_range_l2872_287254

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2*x < 0}
def N (a : ℝ) : Set ℝ := {x | x < a}

-- State the theorem
theorem subset_implies_a_range (a : ℝ) : M ⊆ N a → a ∈ Set.Ici 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_range_l2872_287254


namespace NUMINAMATH_CALUDE_intersection_of_intervals_solution_interval_l2872_287243

theorem intersection_of_intervals : Set.Ioo (1/2 : ℝ) (3/5 : ℝ) = 
  Set.inter 
    (Set.Ioo (1/2 : ℝ) (3/4 : ℝ)) 
    (Set.Ioo (2/5 : ℝ) (3/5 : ℝ)) := by sorry

theorem solution_interval (x : ℝ) : 
  (2 < 4 * x ∧ 4 * x < 3) ∧ (2 < 5 * x ∧ 5 * x < 3) ↔ 
  x ∈ Set.Ioo (1/2 : ℝ) (3/5 : ℝ) := by sorry

end NUMINAMATH_CALUDE_intersection_of_intervals_solution_interval_l2872_287243


namespace NUMINAMATH_CALUDE_x_plus_y_value_l2872_287277

theorem x_plus_y_value (x y : ℝ) (h1 : x + Real.cos y = 3005) 
  (h2 : x + 3005 * Real.sin y = 3004) (h3 : 0 ≤ y) (h4 : y ≤ π) : 
  x + y = 3004 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l2872_287277


namespace NUMINAMATH_CALUDE_g_in_M_l2872_287266

-- Define the set M
def M : Set (ℝ → ℝ) :=
  {f | ∀ x₁ x₂, |x₁| ≤ 1 → |x₂| ≤ 1 → |f x₁ - f x₂| ≤ 4 * |x₁ - x₂|}

-- Define the function g
def g : ℝ → ℝ := λ x ↦ x^2 + 2*x - 1

-- Theorem statement
theorem g_in_M : g ∈ M := by
  sorry

end NUMINAMATH_CALUDE_g_in_M_l2872_287266


namespace NUMINAMATH_CALUDE_movie_theater_popcorn_ratio_l2872_287260

/-- Movie theater revenue calculation and customer ratio --/
theorem movie_theater_popcorn_ratio 
  (matinee_price evening_price opening_price popcorn_price : ℕ)
  (matinee_customers evening_customers opening_customers : ℕ)
  (total_revenue : ℕ)
  (h_matinee : matinee_price = 5)
  (h_evening : evening_price = 7)
  (h_opening : opening_price = 10)
  (h_popcorn : popcorn_price = 10)
  (h_matinee_cust : matinee_customers = 32)
  (h_evening_cust : evening_customers = 40)
  (h_opening_cust : opening_customers = 58)
  (h_total_rev : total_revenue = 1670) :
  (total_revenue - (matinee_price * matinee_customers + 
                    evening_price * evening_customers + 
                    opening_price * opening_customers)) / popcorn_price = 
  (matinee_customers + evening_customers + opening_customers) / 2 :=
by sorry

end NUMINAMATH_CALUDE_movie_theater_popcorn_ratio_l2872_287260


namespace NUMINAMATH_CALUDE_car_speed_calculation_l2872_287219

theorem car_speed_calculation (v : ℝ) : v > 0 → (1 / v) * 3600 = (1 / 80) * 3600 + 10 → v = 3600 / 55 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_calculation_l2872_287219


namespace NUMINAMATH_CALUDE_tropical_fish_count_l2872_287272

theorem tropical_fish_count (total : ℕ) (koi : ℕ) (h1 : total = 52) (h2 : koi = 37) :
  total - koi = 15 := by
  sorry

end NUMINAMATH_CALUDE_tropical_fish_count_l2872_287272


namespace NUMINAMATH_CALUDE_apples_left_l2872_287280

theorem apples_left (frank_apples susan_apples : ℕ) : 
  frank_apples = 36 →
  susan_apples = 3 * frank_apples →
  (frank_apples - frank_apples / 3) + (susan_apples - susan_apples / 2) = 78 :=
by
  sorry

end NUMINAMATH_CALUDE_apples_left_l2872_287280


namespace NUMINAMATH_CALUDE_cos_555_degrees_l2872_287203

theorem cos_555_degrees : 
  Real.cos (555 * Real.pi / 180) = -(Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_555_degrees_l2872_287203


namespace NUMINAMATH_CALUDE_gina_hourly_rate_l2872_287288

/-- Gina's painting rates and order details -/
structure PaintingJob where
  rose_rate : ℕ  -- Cups with roses painted per hour
  lily_rate : ℕ  -- Cups with lilies painted per hour
  rose_order : ℕ  -- Number of rose cups ordered
  lily_order : ℕ  -- Number of lily cups ordered
  total_payment : ℕ  -- Total payment for the order in dollars

/-- Calculate Gina's hourly rate for a given painting job -/
def hourly_rate (job : PaintingJob) : ℚ :=
  job.total_payment / (job.rose_order / job.rose_rate + job.lily_order / job.lily_rate)

/-- Theorem: Gina's hourly rate for the given job is $30 -/
theorem gina_hourly_rate :
  let job : PaintingJob := {
    rose_rate := 6,
    lily_rate := 7,
    rose_order := 6,
    lily_order := 14,
    total_payment := 90
  }
  hourly_rate job = 30 := by
  sorry

end NUMINAMATH_CALUDE_gina_hourly_rate_l2872_287288


namespace NUMINAMATH_CALUDE_not_neighboring_root_eq1_is_neighboring_root_eq2_neighboring_root_eq3_l2872_287230

/-- Definition of a neighboring root equation -/
def is_neighboring_root_equation (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ (x - y = 1 ∨ y - x = 1)

/-- Theorem for the first equation -/
theorem not_neighboring_root_eq1 : ¬is_neighboring_root_equation 1 1 (-6) :=
sorry

/-- Theorem for the second equation -/
theorem is_neighboring_root_eq2 : is_neighboring_root_equation 2 (-2 * Real.sqrt 5) 2 :=
sorry

/-- Theorem for the third equation -/
theorem neighboring_root_eq3 (m : ℝ) : 
  is_neighboring_root_equation 1 (-(m-2)) (-2*m) ↔ m = -1 ∨ m = -3 :=
sorry

end NUMINAMATH_CALUDE_not_neighboring_root_eq1_is_neighboring_root_eq2_neighboring_root_eq3_l2872_287230


namespace NUMINAMATH_CALUDE_students_liking_both_desserts_l2872_287264

/-- Proves the number of students liking both apple pie and chocolate cake in a class --/
theorem students_liking_both_desserts 
  (total_students : ℕ) 
  (apple_pie_fans : ℕ) 
  (chocolate_cake_fans : ℕ) 
  (neither_fans : ℕ) 
  (only_cookies_fans : ℕ) 
  (h1 : total_students = 50)
  (h2 : apple_pie_fans = 22)
  (h3 : chocolate_cake_fans = 20)
  (h4 : neither_fans = 10)
  (h5 : only_cookies_fans = 5)
  : ∃ (both_fans : ℕ), both_fans = 7 := by
  sorry


end NUMINAMATH_CALUDE_students_liking_both_desserts_l2872_287264


namespace NUMINAMATH_CALUDE_divisibility_property_not_true_for_p_2_l2872_287205

theorem divisibility_property (p a n : ℕ) : 
  Nat.Prime p → p ≠ 2 → a > 0 → n > 0 → (p^n ∣ a^p - 1) → (p^(n-1) ∣ a - 1) :=
by sorry

-- The statement is not true for p = 2
theorem not_true_for_p_2 : 
  ∃ (a n : ℕ), a > 0 ∧ n > 0 ∧ (2^n ∣ a^2 - 1) ∧ ¬(2^(n-1) ∣ a - 1) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_property_not_true_for_p_2_l2872_287205


namespace NUMINAMATH_CALUDE_min_area_rectangle_l2872_287216

theorem min_area_rectangle (l w : ℕ) : 
  l > 0 ∧ w > 0 ∧ 2 * (l + w) = 120 → l * w ≥ 59 := by
  sorry

end NUMINAMATH_CALUDE_min_area_rectangle_l2872_287216


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l2872_287271

theorem cubic_sum_minus_product (x y z : ℝ) 
  (h1 : x + y + z = 10) 
  (h2 : x*y + x*z + y*z = 30) : 
  x^3 + y^3 + z^3 - 3*x*y*z = 100 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l2872_287271


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2872_287250

-- Problem 1
theorem problem_1 (a : ℝ) (h : Real.sqrt a + 1 / Real.sqrt a = 3) :
  (a^2 + 1/a^2 + 3) / (4*a + 1/(4*a)) = 10 * Real.sqrt 5 := by sorry

-- Problem 2
theorem problem_2 :
  (1 - Real.log 3 / Real.log 6)^2 + (Real.log 2 / Real.log 6) * (Real.log 18 / Real.log 6) * (Real.log 6 / Real.log 4) = 1 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2872_287250


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l2872_287286

/-- Given a hyperbola with equation x²/p² - y²/q² = 1 where p > q,
    if the angle between its asymptotes is 45°, then p/q = √2 - 1 -/
theorem hyperbola_asymptote_angle (p q : ℝ) (h1 : p > q) (h2 : q > 0) :
  (∃ (x y : ℝ → ℝ), ∀ t, (x t)^2 / p^2 - (y t)^2 / q^2 = 1) →
  (∃ (m : ℝ), m = q / p ∧ 
    Real.tan (45 * π / 180) = |((m - (-m)) / (1 + m * (-m)))|) →
  p / q = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l2872_287286


namespace NUMINAMATH_CALUDE_b_age_is_ten_l2872_287293

/-- Given the ages of three people a, b, and c, prove that b is 10 years old. -/
theorem b_age_is_ten (a b c : ℕ) : 
  a = b + 2 → 
  b = 2 * c → 
  a + b + c = 27 → 
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_b_age_is_ten_l2872_287293


namespace NUMINAMATH_CALUDE_nickel_probability_is_one_fourth_l2872_287207

/-- Represents the types of coins in the jar -/
inductive Coin
  | Dime
  | Nickel
  | Penny

/-- The value of each coin type in cents -/
def coinValue : Coin → ℚ
  | Coin.Dime => 10
  | Coin.Nickel => 5
  | Coin.Penny => 1

/-- The total value of each coin type in the jar in cents -/
def totalValue : Coin → ℚ
  | Coin.Dime => 500
  | Coin.Nickel => 250
  | Coin.Penny => 100

/-- The number of coins of each type in the jar -/
def coinCount (c : Coin) : ℚ :=
  totalValue c / coinValue c

/-- The total number of coins in the jar -/
def totalCoins : ℚ :=
  coinCount Coin.Dime + coinCount Coin.Nickel + coinCount Coin.Penny

/-- The probability of choosing a nickel from the jar -/
def nickelProbability : ℚ :=
  coinCount Coin.Nickel / totalCoins

theorem nickel_probability_is_one_fourth :
  nickelProbability = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_nickel_probability_is_one_fourth_l2872_287207


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2872_287276

theorem polynomial_simplification (x : ℝ) :
  (3 * x^3 + 4 * x^2 + 9 * x - 5) - (2 * x^3 + x^2 + 6 * x - 8) = x^3 + 3 * x^2 + 3 * x + 3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2872_287276


namespace NUMINAMATH_CALUDE_initial_truck_distance_l2872_287278

/-- 
Given two trucks on opposite sides of a highway, where:
- Driver A starts driving at 90 km/h
- Driver B starts 1 hour later at 80 km/h
- When they meet, Driver A has driven 140 km farther than Driver B

This theorem proves that the initial distance between the trucks is 940 km.
-/
theorem initial_truck_distance :
  ∀ (t : ℝ) (d_a d_b : ℝ),
  d_a = 90 * (t + 1) →
  d_b = 80 * t →
  d_a = d_b + 140 →
  d_a + d_b = 940 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_truck_distance_l2872_287278


namespace NUMINAMATH_CALUDE_grandfather_pension_increase_l2872_287239

/-- Represents the percentage increase in family income when a member's income is doubled -/
structure IncomeIncrease where
  masha : ℝ
  mother : ℝ
  father : ℝ

/-- Calculates the percentage increase in family income when grandfather's pension is doubled -/
def grandfather_increase (i : IncomeIncrease) : ℝ :=
  100 - (i.masha + i.mother + i.father)

/-- Theorem stating that given the specified income increases for Masha, mother, and father,
    doubling grandfather's pension will increase the family income by 55% -/
theorem grandfather_pension_increase (i : IncomeIncrease) 
  (h1 : i.masha = 5)
  (h2 : i.mother = 15)
  (h3 : i.father = 25) :
  grandfather_increase i = 55 := by
  sorry

#eval grandfather_increase { masha := 5, mother := 15, father := 25 }

end NUMINAMATH_CALUDE_grandfather_pension_increase_l2872_287239


namespace NUMINAMATH_CALUDE_highest_score_percentage_l2872_287201

/-- The percentage of correct answers on an exam with a given number of questions -/
def examPercentage (correctAnswers : ℕ) (totalQuestions : ℕ) : ℚ :=
  (correctAnswers : ℚ) / (totalQuestions : ℚ) * 100

theorem highest_score_percentage
  (totalQuestions : ℕ)
  (hannahsTarget : ℕ)
  (otherStudentWrong : ℕ)
  (hTotal : totalQuestions = 40)
  (hHannah : hannahsTarget = 39)
  (hOther : otherStudentWrong = 3)
  : examPercentage (totalQuestions - otherStudentWrong - 1) totalQuestions = 95 := by
  sorry

end NUMINAMATH_CALUDE_highest_score_percentage_l2872_287201


namespace NUMINAMATH_CALUDE_polynomial_expansion_coefficient_l2872_287209

theorem polynomial_expansion_coefficient (x : ℝ) : 
  ∃ (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ), 
    (x^10 = a + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + 
            a₅*(x-1)^5 + a₆*(x-1)^6 + a₇*(x-1)^7 + a₈*(x-1)^8 + 
            a₉*(x-1)^9 + a₁₀*(x-1)^10) ∧ 
    a₇ = 120 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_coefficient_l2872_287209


namespace NUMINAMATH_CALUDE_solve_for_y_l2872_287225

theorem solve_for_y (x y : ℤ) (h1 : x^2 = y - 2) (h2 : x = -6) : y = 38 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2872_287225


namespace NUMINAMATH_CALUDE_no_odd_multiples_of_18_24_36_between_1500_3000_l2872_287200

theorem no_odd_multiples_of_18_24_36_between_1500_3000 :
  ∀ n : ℕ, 1500 < n ∧ n < 3000 ∧ n % 2 = 1 →
    ¬(18 ∣ n ∧ 24 ∣ n ∧ 36 ∣ n) :=
by sorry

end NUMINAMATH_CALUDE_no_odd_multiples_of_18_24_36_between_1500_3000_l2872_287200


namespace NUMINAMATH_CALUDE_league_face_count_l2872_287259

/-- The number of games in a single round-robin tournament with n teams -/
def roundRobinGames (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of times each team faces another in a league -/
def faceCount (totalTeams : ℕ) (totalGames : ℕ) : ℕ :=
  totalGames / roundRobinGames totalTeams

theorem league_face_count :
  faceCount 14 455 = 5 := by sorry

end NUMINAMATH_CALUDE_league_face_count_l2872_287259


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l2872_287256

theorem regular_polygon_exterior_angle (n : ℕ) (exterior_angle : ℝ) :
  n ≥ 3 → exterior_angle = 36 → n = 10 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l2872_287256


namespace NUMINAMATH_CALUDE_unique_number_with_pairable_divisors_l2872_287212

def is_own_divisor (d n : ℕ) : Prop :=
  d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def has_pairable_own_divisors (n : ℕ) : Prop :=
  ∃ (f : ℕ → ℕ), 
    (∀ d, is_own_divisor d n → is_own_divisor (f d) n) ∧
    (∀ d, is_own_divisor d n → f (f d) = d) ∧
    (∀ d, is_own_divisor d n → (f d = d + 545 ∨ f d = d - 545))

theorem unique_number_with_pairable_divisors :
  ∃! n : ℕ, has_pairable_own_divisors n ∧ n = 1094 :=
sorry

end NUMINAMATH_CALUDE_unique_number_with_pairable_divisors_l2872_287212


namespace NUMINAMATH_CALUDE_solution_set_l2872_287223

theorem solution_set (x : ℝ) : 
  (x / 4 ≤ 3 + x ∧ 3 + x < -3 * (1 + x)) ↔ x ∈ Set.Icc (-4) (-3/2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_l2872_287223


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l2872_287235

theorem square_area_from_diagonal (d : ℝ) (h : d = 12) : 
  let side := d / Real.sqrt 2
  side ^ 2 = 72 := by
sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l2872_287235


namespace NUMINAMATH_CALUDE_total_wool_calculation_l2872_287249

/-- The number of scarves Aaron makes -/
def aaron_scarves : ℕ := 10

/-- The number of sweaters Aaron makes -/
def aaron_sweaters : ℕ := 5

/-- The number of sweaters Enid makes -/
def enid_sweaters : ℕ := 8

/-- The number of balls of wool used for one scarf -/
def wool_per_scarf : ℕ := 3

/-- The number of balls of wool used for one sweater -/
def wool_per_sweater : ℕ := 4

/-- The total number of balls of wool used by Enid and Aaron -/
def total_wool_used : ℕ :=
  aaron_scarves * wool_per_scarf +
  aaron_sweaters * wool_per_sweater +
  enid_sweaters * wool_per_sweater

theorem total_wool_calculation : total_wool_used = 82 := by
  sorry

end NUMINAMATH_CALUDE_total_wool_calculation_l2872_287249


namespace NUMINAMATH_CALUDE_man_speed_on_bridge_l2872_287253

/-- Calculates the speed of a man crossing a bridge -/
theorem man_speed_on_bridge (bridge_length : ℝ) (crossing_time : ℝ) : 
  bridge_length = 2500 →  -- bridge length in meters
  crossing_time = 15 →    -- crossing time in minutes
  bridge_length / (crossing_time / 60) / 1000 = 10 := by
  sorry

end NUMINAMATH_CALUDE_man_speed_on_bridge_l2872_287253


namespace NUMINAMATH_CALUDE_completing_square_result_l2872_287222

theorem completing_square_result (x : ℝ) : 
  (x^2 - 4*x + 2 = 0) ↔ ((x - 2)^2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_result_l2872_287222


namespace NUMINAMATH_CALUDE_seven_hash_three_l2872_287236

-- Define the # operation
noncomputable def hash (r s : ℝ) : ℝ :=
  sorry

-- Axioms for the # operation
axiom hash_zero (r : ℝ) : hash r 0 = r + 1
axiom hash_comm (r s : ℝ) : hash r s = hash s r
axiom hash_succ (r s : ℝ) : hash (r + 2) s = hash r s + s + 2

-- Theorem to prove
theorem seven_hash_three : hash 7 3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_seven_hash_three_l2872_287236


namespace NUMINAMATH_CALUDE_remainder_problem_l2872_287218

theorem remainder_problem (n : ℕ) : 
  n % 101 = 0 ∧ n / 101 = 347 → n % 89 = 70 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2872_287218


namespace NUMINAMATH_CALUDE_power_equation_solution_l2872_287290

def solution_set : Set ℝ := {-3, 1, 2}

theorem power_equation_solution (x : ℝ) : 
  (2*x - 3)^(x + 3) = 1 ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2872_287290


namespace NUMINAMATH_CALUDE_cube_root_5832_l2872_287281

theorem cube_root_5832 : ∃ (a b : ℕ+), (a.val : ℝ) * (b.val : ℝ)^(1/3) = 5832^(1/3) ∧ 
  (∀ (c d : ℕ+), (c.val : ℝ) * (d.val : ℝ)^(1/3) = 5832^(1/3) → b ≤ d) → 
  a.val + b.val = 19 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_5832_l2872_287281


namespace NUMINAMATH_CALUDE_angle_D_is_100_l2872_287294

-- Define the triangle DEF
structure Triangle :=
  (D E F : ℝ)

-- Define the properties of the triangle
def is_right_triangle (t : Triangle) : Prop :=
  t.D + t.E + t.F = 180

def angle_E_is_30 (t : Triangle) : Prop :=
  t.E = 30

def angle_D_twice_F (t : Triangle) : Prop :=
  t.D = 2 * t.F

-- Theorem statement
theorem angle_D_is_100 (t : Triangle) 
  (h1 : is_right_triangle t) 
  (h2 : angle_E_is_30 t) 
  (h3 : angle_D_twice_F t) : 
  t.D = 100 :=
sorry

end NUMINAMATH_CALUDE_angle_D_is_100_l2872_287294


namespace NUMINAMATH_CALUDE_mushroom_soup_production_l2872_287284

theorem mushroom_soup_production (total_required : ℕ) (team1_production : ℕ) (team2_production : ℕ) 
  (h1 : total_required = 280)
  (h2 : team1_production = 90)
  (h3 : team2_production = 120) :
  total_required - (team1_production + team2_production) = 70 :=
by sorry

end NUMINAMATH_CALUDE_mushroom_soup_production_l2872_287284


namespace NUMINAMATH_CALUDE_stratified_sample_female_count_l2872_287227

/-- Represents the number of female athletes in a stratified sample -/
def female_athletes_in_sample (total_athletes : ℕ) (female_athletes : ℕ) (sample_size : ℕ) : ℕ :=
  (female_athletes * sample_size) / total_athletes

theorem stratified_sample_female_count :
  female_athletes_in_sample 98 42 28 = 12 := by
  sorry

#eval female_athletes_in_sample 98 42 28

end NUMINAMATH_CALUDE_stratified_sample_female_count_l2872_287227


namespace NUMINAMATH_CALUDE_distance_to_point_l2872_287265

-- Define the point
def point : ℝ × ℝ := (-12, 5)

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem distance_to_point : Real.sqrt ((point.1 - origin.1)^2 + (point.2 - origin.2)^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_point_l2872_287265


namespace NUMINAMATH_CALUDE_tricycle_count_l2872_287215

theorem tricycle_count (num_bicycles : ℕ) (bicycle_wheels : ℕ) (tricycle_wheels : ℕ) (total_wheels : ℕ) :
  num_bicycles = 16 →
  bicycle_wheels = 2 →
  tricycle_wheels = 3 →
  total_wheels = 53 →
  ∃ num_tricycles : ℕ, num_bicycles * bicycle_wheels + num_tricycles * tricycle_wheels = total_wheels ∧ num_tricycles = 7 :=
by sorry

end NUMINAMATH_CALUDE_tricycle_count_l2872_287215


namespace NUMINAMATH_CALUDE_irreducible_fraction_l2872_287275

theorem irreducible_fraction (n : ℕ) (hn : n > 0) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_irreducible_fraction_l2872_287275


namespace NUMINAMATH_CALUDE_sequence_non_positive_l2872_287285

theorem sequence_non_positive (n : ℕ) (a : ℕ → ℝ) 
  (h_n : n ≥ 3)
  (h_start : a 1 = 0)
  (h_end : a n = 0)
  (h_ineq : ∀ k : ℕ, 2 ≤ k ∧ k ≤ n - 1 → a (k - 1) + a (k + 1) ≥ 2 * a k) :
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a i ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_sequence_non_positive_l2872_287285
