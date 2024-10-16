import Mathlib

namespace NUMINAMATH_CALUDE_solve_for_a_l2126_212605

theorem solve_for_a : ∃ a : ℝ, (3 * 3 - 2 * a = 5) ∧ a = 2 := by sorry

end NUMINAMATH_CALUDE_solve_for_a_l2126_212605


namespace NUMINAMATH_CALUDE_expression_value_l2126_212634

theorem expression_value (x y : ℝ) (hx : x = 2) (hy : y = -3) :
  ((2 * x - y)^2 - (x - y) * (x + y) - 2 * y^2) / x = 18 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2126_212634


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2126_212603

theorem cube_root_equation_solution :
  ∃! x : ℝ, (4 - x / 3) ^ (1/3 : ℝ) = -2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2126_212603


namespace NUMINAMATH_CALUDE_system_solution_l2126_212654

def solution_set : Set (ℕ × ℕ × ℕ × ℕ) :=
  {(1, 5, 2, 3), (1, 5, 3, 2), (5, 1, 2, 3), (5, 1, 3, 2), (2, 2, 2, 2),
   (2, 3, 1, 5), (2, 3, 5, 1), (3, 2, 1, 5), (3, 2, 5, 1)}

theorem system_solution (a b c d : ℕ) :
  (a * b = c + d ∧ c * d = a + b) ↔ (a, b, c, d) ∈ solution_set := by
  sorry

#check system_solution

end NUMINAMATH_CALUDE_system_solution_l2126_212654


namespace NUMINAMATH_CALUDE_basketball_free_throws_l2126_212684

theorem basketball_free_throws (total_players : Nat) (goalkeepers : Nat) : 
  total_players = 18 → goalkeepers = 2 → 
  (total_players - goalkeepers) * goalkeepers = 34 := by
  sorry

end NUMINAMATH_CALUDE_basketball_free_throws_l2126_212684


namespace NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l2126_212675

theorem sphere_hemisphere_volume_ratio (r : ℝ) (h : r > 0) :
  (4 / 3 * Real.pi * r^3) / (1 / 2 * 4 / 3 * Real.pi * (r / 3)^3) = 54 := by
  sorry

end NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l2126_212675


namespace NUMINAMATH_CALUDE_total_flowers_and_stems_l2126_212688

def roses : ℕ := 12
def carnations : ℕ := 15
def lilies : ℕ := 10
def tulips : ℕ := 8
def daisies : ℕ := 5
def orchids : ℕ := 3
def babys_breath : ℕ := 10

theorem total_flowers_and_stems :
  roses + carnations + lilies + tulips + daisies + orchids + babys_breath = 63 := by
  sorry

end NUMINAMATH_CALUDE_total_flowers_and_stems_l2126_212688


namespace NUMINAMATH_CALUDE_binomial_expansion_theorem_l2126_212648

theorem binomial_expansion_theorem (a b c k n : ℝ) :
  (n ≥ 2) →
  (a ≠ b) →
  (a * b ≠ 0) →
  (a = k * b + c) →
  (k > 0) →
  (c ≠ 0) →
  (c ≠ b * (k - 1)) →
  (∃ (x y : ℝ), (x + y)^n = (a - b)^n ∧ x + y = 0) →
  (n = -b * (k - 1) / c) := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_theorem_l2126_212648


namespace NUMINAMATH_CALUDE_orange_price_theorem_l2126_212627

/-- The cost of fruits and the discount policy at a store --/
structure FruitStore where
  apple_cost : ℚ
  banana_cost : ℚ
  discount_per_five : ℚ

/-- A customer's purchase of fruits --/
structure Purchase where
  apples : ℕ
  oranges : ℕ
  bananas : ℕ

/-- Calculate the total cost of a purchase given the store's prices and an orange price --/
def totalCost (store : FruitStore) (purchase : Purchase) (orange_price : ℚ) : ℚ :=
  store.apple_cost * purchase.apples +
  orange_price * purchase.oranges +
  store.banana_cost * purchase.bananas -
  store.discount_per_five * ((purchase.apples + purchase.oranges + purchase.bananas) / 5)

/-- The theorem stating the price of oranges based on Mary's purchase --/
theorem orange_price_theorem (store : FruitStore) (purchase : Purchase) :
  store.apple_cost = 1 →
  store.banana_cost = 3 →
  store.discount_per_five = 1 →
  purchase.apples = 5 →
  purchase.oranges = 3 →
  purchase.bananas = 2 →
  totalCost store purchase (8/3) = 15 :=
by sorry

end NUMINAMATH_CALUDE_orange_price_theorem_l2126_212627


namespace NUMINAMATH_CALUDE_marker_cost_l2126_212607

theorem marker_cost (total_students : ℕ) (total_cost : ℕ) 
  (h_total_students : total_students = 40)
  (h_total_cost : total_cost = 3388) :
  ∃ (s n c : ℕ),
    s > total_students / 2 ∧
    s ≤ total_students ∧
    n > 1 ∧
    c > n ∧
    s * n * c = total_cost ∧
    c = 11 := by
  sorry

end NUMINAMATH_CALUDE_marker_cost_l2126_212607


namespace NUMINAMATH_CALUDE_kaydence_age_l2126_212609

/-- Kaydence's family ages problem -/
theorem kaydence_age (total_age father_age mother_age brother_age sister_age kaydence_age : ℕ) :
  total_age = 200 ∧
  father_age = 60 ∧
  mother_age = father_age - 2 ∧
  brother_age = father_age / 2 ∧
  sister_age = 40 ∧
  total_age = father_age + mother_age + brother_age + sister_age + kaydence_age →
  kaydence_age = 12 := by
  sorry

end NUMINAMATH_CALUDE_kaydence_age_l2126_212609


namespace NUMINAMATH_CALUDE_inequality_represents_lower_right_l2126_212618

/-- Represents a point in the 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- The line defined by the equation x - 2y + 6 = 0 -/
def line (p : Point) : Prop :=
  p.x - 2 * p.y + 6 = 0

/-- The area defined by the inequality x - 2y + 6 > 0 -/
def inequality_area (p : Point) : Prop :=
  p.x - 2 * p.y + 6 > 0

/-- A point is on the lower right side of the line if it satisfies the inequality -/
def is_lower_right (p : Point) : Prop :=
  inequality_area p

theorem inequality_represents_lower_right :
  ∀ p : Point, is_lower_right p ↔ inequality_area p :=
sorry

end NUMINAMATH_CALUDE_inequality_represents_lower_right_l2126_212618


namespace NUMINAMATH_CALUDE_no_snow_probability_l2126_212636

/-- The probability of no snow for five consecutive days, given the probability of snow each day is 2/3 -/
theorem no_snow_probability (p : ℚ) (h : p = 2/3) : (1 - p)^5 = 1/243 := by
  sorry

end NUMINAMATH_CALUDE_no_snow_probability_l2126_212636


namespace NUMINAMATH_CALUDE_yellow_balloons_total_l2126_212601

/-- The total number of yellow balloons Sam and Mary have -/
def total_balloons (sam_initial : ℝ) (sam_given : ℝ) (mary : ℝ) : ℝ :=
  (sam_initial - sam_given) + mary

/-- Theorem stating the total number of yellow balloons Sam and Mary have -/
theorem yellow_balloons_total :
  total_balloons 6.0 5.0 7.0 = 8.0 := by
  sorry

end NUMINAMATH_CALUDE_yellow_balloons_total_l2126_212601


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l2126_212622

theorem fractional_equation_solution_range (m : ℝ) (x : ℝ) : 
  (m / (2 * x - 1) + 2 = 0) → (x > 0) → (m < 2 ∧ m ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l2126_212622


namespace NUMINAMATH_CALUDE_perimeter_is_22_l2126_212608

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 14 - y^2 / 11 = 1

-- Define the focus F₁
def F₁ : ℝ × ℝ := sorry

-- Define the line l
def l : Set (ℝ × ℝ) := sorry

-- Define points P and Q
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

-- State that P and Q are on the hyperbola
axiom P_on_hyperbola : hyperbola P.1 P.2
axiom Q_on_hyperbola : hyperbola Q.1 Q.2

-- State that P and Q are on line l
axiom P_on_l : P ∈ l
axiom Q_on_l : Q ∈ l

-- State that line l passes through the origin
axiom l_through_origin : (0, 0) ∈ l

-- State that PF₁ · QF₁ = 0
axiom PF₁_perp_QF₁ : (P.1 - F₁.1, P.2 - F₁.2) • (Q.1 - F₁.1, Q.2 - F₁.2) = 0

-- Define the perimeter of triangle PF₁Q
def perimeter_PF₁Q : ℝ := sorry

-- Theorem to prove
theorem perimeter_is_22 : perimeter_PF₁Q = 22 := sorry

end NUMINAMATH_CALUDE_perimeter_is_22_l2126_212608


namespace NUMINAMATH_CALUDE_triangle_nth_part_area_l2126_212679

theorem triangle_nth_part_area (b h n : ℝ) (h_pos : 0 < h) (n_pos : 0 < n) :
  let original_area := (1 / 2) * b * h
  let cut_height := h / Real.sqrt n
  let cut_area := (1 / 2) * b * cut_height
  cut_area = (1 / n) * original_area := by
  sorry

end NUMINAMATH_CALUDE_triangle_nth_part_area_l2126_212679


namespace NUMINAMATH_CALUDE_square_area_is_169_l2126_212677

/-- Square with intersecting segments --/
structure SquareWithIntersection where
  -- Side length of the square
  s : ℝ
  -- Length of BR
  br : ℝ
  -- Length of PR
  pr : ℝ
  -- Length of CQ
  cq : ℝ
  -- Conditions
  br_positive : br > 0
  pr_positive : pr > 0
  cq_positive : cq > 0
  right_angle : True  -- Represents that BP and CQ intersect at right angles
  br_eq : br = 8
  pr_eq : pr = 5
  cq_eq : cq = 12

/-- The area of the square is 169 --/
theorem square_area_is_169 (square : SquareWithIntersection) : square.s^2 = 169 := by
  sorry

end NUMINAMATH_CALUDE_square_area_is_169_l2126_212677


namespace NUMINAMATH_CALUDE_students_without_glasses_l2126_212658

theorem students_without_glasses (total : ℕ) (with_glasses_percent : ℚ) 
  (h1 : total = 325) 
  (h2 : with_glasses_percent = 40 / 100) : 
  ↑total * (1 - with_glasses_percent) = 195 := by
  sorry

end NUMINAMATH_CALUDE_students_without_glasses_l2126_212658


namespace NUMINAMATH_CALUDE_max_expected_expenditure_l2126_212690

/-- Linear regression model for fiscal revenue and expenditure -/
def fiscal_model (x y a b ε : ℝ) : Prop :=
  y = a + b * x + ε

/-- Theorem: Maximum expected expenditure given fiscal revenue -/
theorem max_expected_expenditure
  (a b x y ε : ℝ)
  (model : fiscal_model x y a b ε)
  (h_a : a = 2)
  (h_b : b = 0.8)
  (h_ε : |ε| ≤ 0.5)
  (h_x : x = 10) :
  y ≤ 10.5 := by
  sorry

#check max_expected_expenditure

end NUMINAMATH_CALUDE_max_expected_expenditure_l2126_212690


namespace NUMINAMATH_CALUDE_max_area_rectangle_with_fixed_perimeter_l2126_212686

/-- The maximum area of a rectangle with integer side lengths and perimeter 160 feet -/
theorem max_area_rectangle_with_fixed_perimeter :
  ∃ (w h : ℕ), 
    (2 * w + 2 * h = 160) ∧ 
    (∀ (x y : ℕ), (2 * x + 2 * y = 160) → (x * y ≤ w * h)) ∧
    (w * h = 1600) := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangle_with_fixed_perimeter_l2126_212686


namespace NUMINAMATH_CALUDE_abc_sum_product_bounds_l2126_212646

theorem abc_sum_product_bounds (a b c : ℝ) (h : a + b + c = 1) :
  ∀ ε > 0, ∃ x : ℝ, x = a * b + a * c + b * c ∧ x ≤ 1/3 ∧ ∃ y : ℝ, y = a * b + a * c + b * c ∧ y < -ε :=
by sorry

end NUMINAMATH_CALUDE_abc_sum_product_bounds_l2126_212646


namespace NUMINAMATH_CALUDE_new_person_weight_l2126_212613

theorem new_person_weight (n : ℕ) (old_weight avg_increase : ℝ) :
  n = 8 ∧ 
  old_weight = 70 ∧ 
  avg_increase = 3 →
  (n * avg_increase + old_weight : ℝ) = 94 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l2126_212613


namespace NUMINAMATH_CALUDE_stating_prize_distribution_orders_l2126_212657

/-- Represents the number of players in the tournament -/
def num_players : ℕ := 6

/-- Represents the number of games played in the tournament -/
def num_games : ℕ := num_players - 1

/-- 
Theorem stating that the number of possible prize distribution orders
in a tournament with 6 players following the described elimination format is 32
-/
theorem prize_distribution_orders :
  (2 : ℕ) ^ num_games = 32 := by
  sorry

end NUMINAMATH_CALUDE_stating_prize_distribution_orders_l2126_212657


namespace NUMINAMATH_CALUDE_average_difference_l2126_212619

/-- The number of students in the school -/
def num_students : ℕ := 120

/-- The number of teachers in the school -/
def num_teachers : ℕ := 4

/-- The list of class sizes -/
def class_sizes : List ℕ := [40, 30, 30, 20]

/-- Average number of students per class from a teacher's perspective -/
def t : ℚ := (num_students : ℚ) / num_teachers

/-- Average number of students per class from a student's perspective -/
def s : ℚ := (List.sum (List.map (λ x => x * x) class_sizes) : ℚ) / num_students

theorem average_difference : t - s = -167/100 := by sorry

end NUMINAMATH_CALUDE_average_difference_l2126_212619


namespace NUMINAMATH_CALUDE_people_born_in_country_l2126_212667

/-- The number of people who immigrated to the country last year -/
def immigrants : ℕ := 16320

/-- The total number of new people who began living in the country last year -/
def new_residents : ℕ := 106491

/-- The number of people born in the country last year -/
def births : ℕ := new_residents - immigrants

theorem people_born_in_country : births = 90171 := by
  sorry

end NUMINAMATH_CALUDE_people_born_in_country_l2126_212667


namespace NUMINAMATH_CALUDE_jasmine_bouquet_cost_l2126_212683

/-- The cost of a bouquet with a given number of jasmines -/
def bouquet_cost (num_jasmines : ℕ) : ℚ :=
  24 * (num_jasmines : ℚ) / 8

/-- The discounted cost of a bouquet -/
def discounted_cost (original_cost : ℚ) (discount_percent : ℚ) : ℚ :=
  original_cost * (1 - discount_percent / 100)

theorem jasmine_bouquet_cost :
  discounted_cost (bouquet_cost 50) 10 = 135 := by sorry

end NUMINAMATH_CALUDE_jasmine_bouquet_cost_l2126_212683


namespace NUMINAMATH_CALUDE_quadratic_property_l2126_212610

def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_property (a b c : ℝ) (h1 : a ≠ 0) :
  (quadratic a b c 2 = 0.35) →
  (quadratic a b c 4 = 0.35) →
  (quadratic a b c 5 = 3) →
  (a + b + c) * (-b / a) = 18 := by
sorry

end NUMINAMATH_CALUDE_quadratic_property_l2126_212610


namespace NUMINAMATH_CALUDE_book_arrangement_count_l2126_212626

/-- The number of ways to arrange books on a shelf -/
def arrange_books : ℕ := 48

/-- The number of math books -/
def num_math_books : ℕ := 4

/-- The number of English books -/
def num_english_books : ℕ := 5

/-- Theorem stating the number of ways to arrange books on a shelf -/
theorem book_arrangement_count :
  arrange_books = 
    (Nat.factorial 2) * (Nat.factorial num_math_books) * 1 :=
by sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l2126_212626


namespace NUMINAMATH_CALUDE_six_balls_four_boxes_l2126_212642

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 84 ways to distribute 6 indistinguishable balls into 4 distinguishable boxes -/
theorem six_balls_four_boxes : distribute_balls 6 4 = 84 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_four_boxes_l2126_212642


namespace NUMINAMATH_CALUDE_solve_for_x_l2126_212637

def U (x : ℝ) : Set ℝ := {2, 3, x^2 + 2*x - 3}
def A (x : ℝ) : Set ℝ := {2, |x + 7|}

theorem solve_for_x : 
  ∃ x : ℝ, (U x \ A x = {5}) ∧ x = -4 :=
sorry

end NUMINAMATH_CALUDE_solve_for_x_l2126_212637


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l2126_212693

theorem complex_arithmetic_equality : ((-1 : ℤ) ^ 2024) + (-10 : ℤ) / (1 / 2 : ℚ) * 2 + (2 - (-3 : ℤ) ^ 3) = -10 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l2126_212693


namespace NUMINAMATH_CALUDE_surface_area_greater_when_contained_l2126_212697

/-- A convex polyhedron in 3D space -/
structure ConvexPolyhedron where
  -- Add necessary fields here
  surfaceArea : ℝ

/-- States that one polyhedron is completely contained within another -/
def IsContainedIn (inner outer : ConvexPolyhedron) : Prop :=
  sorry

/-- Theorem: The surface area of the outer polyhedron is greater than
    the surface area of the inner polyhedron when one is contained in the other -/
theorem surface_area_greater_when_contained
  (inner outer : ConvexPolyhedron)
  (h : IsContainedIn inner outer) :
  outer.surfaceArea > inner.surfaceArea :=
sorry

end NUMINAMATH_CALUDE_surface_area_greater_when_contained_l2126_212697


namespace NUMINAMATH_CALUDE_correct_num_tripodasauruses_l2126_212614

/-- Represents the number of tripodasauruses in a flock -/
def num_tripodasauruses : ℕ := 5

/-- Represents the number of legs a tripodasaurus has -/
def legs_per_tripodasaurus : ℕ := 3

/-- Represents the number of heads a tripodasaurus has -/
def heads_per_tripodasaurus : ℕ := 1

/-- Represents the total number of heads and legs in the flock -/
def total_heads_and_legs : ℕ := 20

/-- Theorem stating that the number of tripodasauruses in the flock is correct -/
theorem correct_num_tripodasauruses : 
  num_tripodasauruses * (legs_per_tripodasaurus + heads_per_tripodasaurus) = total_heads_and_legs :=
by sorry

end NUMINAMATH_CALUDE_correct_num_tripodasauruses_l2126_212614


namespace NUMINAMATH_CALUDE_tennis_tournament_matches_l2126_212669

theorem tennis_tournament_matches (total_players : ℕ) (advanced_players : ℕ) 
  (h1 : total_players = 128)
  (h2 : advanced_players = 20)
  (h3 : total_players > advanced_players) :
  (total_players - 1 : ℕ) = 127 := by
sorry

end NUMINAMATH_CALUDE_tennis_tournament_matches_l2126_212669


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2126_212633

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 - 3*x + a = 0}
def B (b : ℝ) : Set ℝ := {x | x^2 + b = 0}

-- State the theorem
theorem union_of_A_and_B (a b : ℝ) :
  (∃ (x : ℝ), A a ∩ B b = {x}) →
  (∃ (y z : ℝ), A a ∪ B b = {y, z, 2}) :=
sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2126_212633


namespace NUMINAMATH_CALUDE_trig_problem_l2126_212602

theorem trig_problem (a : Real) (h1 : 0 < a) (h2 : a < Real.pi) (h3 : Real.tan a = -2) :
  (Real.cos a = -Real.sqrt 5 / 5) ∧
  (2 * Real.sin a ^ 2 - Real.sin a * Real.cos a + Real.cos a ^ 2 = 11 / 5) := by
  sorry

end NUMINAMATH_CALUDE_trig_problem_l2126_212602


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_nineteen_fifteenths_l2126_212640

theorem sqrt_sum_equals_nineteen_fifteenths (w x z : ℝ) 
  (hw : w = 4) (hx : x = 9) (hz : z = 25) : 
  Real.sqrt (w / x) + Real.sqrt (x / z) = 19 / 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_nineteen_fifteenths_l2126_212640


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2126_212652

theorem quadratic_equation_solution (p q : ℤ) :
  (∃ x : ℝ, 36 * x^2 - 4 * (p^2 + 11) * x + 135 * (p + q) + 576 = 0) →
  p + q = 20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2126_212652


namespace NUMINAMATH_CALUDE_shortest_distance_proof_l2126_212670

/-- Given a body moving on a horizontal plane, prove that with displacements of 4 meters
    along the x-axis and 3 meters along the y-axis, the shortest distance between
    the initial and final points is 5 meters. -/
theorem shortest_distance_proof (x y : ℝ) (hx : x = 4) (hy : y = 3) :
  Real.sqrt (x^2 + y^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_proof_l2126_212670


namespace NUMINAMATH_CALUDE_interior_perimeter_is_14_l2126_212632

/-- Represents a rectangular picture frame -/
structure PictureFrame where
  outerWidth : ℝ
  outerHeight : ℝ
  frameWidth : ℝ

/-- Calculates the area of just the frame -/
def frameArea (frame : PictureFrame) : ℝ :=
  frame.outerWidth * frame.outerHeight - (frame.outerWidth - 2 * frame.frameWidth) * (frame.outerHeight - 2 * frame.frameWidth)

/-- Calculates the sum of the lengths of the four interior edges -/
def interiorPerimeter (frame : PictureFrame) : ℝ :=
  2 * (frame.outerWidth - 2 * frame.frameWidth) + 2 * (frame.outerHeight - 2 * frame.frameWidth)

/-- Theorem: Given the conditions, the sum of interior edges is 14 inches -/
theorem interior_perimeter_is_14 (frame : PictureFrame) 
  (h1 : frame.frameWidth = 1)
  (h2 : frameArea frame = 18)
  (h3 : frame.outerWidth = 5) :
  interiorPerimeter frame = 14 := by
  sorry

end NUMINAMATH_CALUDE_interior_perimeter_is_14_l2126_212632


namespace NUMINAMATH_CALUDE_exists_points_with_midpoint_l2126_212631

/-- Definition of the hyperbola -/
def is_on_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2/9 = 1

/-- Definition of midpoint -/
def is_midpoint (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₀ = (x₁ + x₂)/2 ∧ y₀ = (y₁ + y₂)/2

/-- Theorem statement -/
theorem exists_points_with_midpoint :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    is_on_hyperbola x₁ y₁ ∧
    is_on_hyperbola x₂ y₂ ∧
    is_midpoint (-1) (-4) x₁ y₁ x₂ y₂ :=
by sorry

end NUMINAMATH_CALUDE_exists_points_with_midpoint_l2126_212631


namespace NUMINAMATH_CALUDE_broadway_ticket_sales_l2126_212695

theorem broadway_ticket_sales
  (num_adults : ℕ)
  (num_children : ℕ)
  (adult_ticket_price : ℝ)
  (h1 : num_adults = 400)
  (h2 : num_children = 200)
  (h3 : adult_ticket_price = 32)
  (h4 : adult_ticket_price = 2 * (adult_ticket_price / 2)) :
  num_adults * adult_ticket_price + num_children * (adult_ticket_price / 2) = 16000 := by
sorry

end NUMINAMATH_CALUDE_broadway_ticket_sales_l2126_212695


namespace NUMINAMATH_CALUDE_chocolate_box_problem_l2126_212604

theorem chocolate_box_problem (total : ℕ) (remaining : ℕ) : 
  (remaining = 36) →
  (total = (remaining : ℚ) * 3 / (1 - (4/15 : ℚ))) →
  (total = 98) := by
sorry

end NUMINAMATH_CALUDE_chocolate_box_problem_l2126_212604


namespace NUMINAMATH_CALUDE_stability_comparison_A_more_stable_than_B_l2126_212649

structure Student where
  name : String
  variance : ℝ

def more_stable (a b : Student) : Prop :=
  a.variance < b.variance

theorem stability_comparison (a b : Student) 
  (h_mean : a.variance ≠ b.variance) :
  more_stable a b ∨ more_stable b a := by
  sorry

-- Define the specific students from the problem
def student_A : Student := ⟨"A", 1.4⟩
def student_B : Student := ⟨"B", 2.5⟩

-- Theorem for the specific case in the problem
theorem A_more_stable_than_B : 
  more_stable student_A student_B := by
  sorry

end NUMINAMATH_CALUDE_stability_comparison_A_more_stable_than_B_l2126_212649


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l2126_212665

theorem trigonometric_simplification :
  (Real.sin (15 * π / 180) + Real.sin (30 * π / 180) + Real.sin (45 * π / 180) + 
   Real.sin (60 * π / 180) + Real.sin (75 * π / 180)) / 
  (Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * Real.cos (30 * π / 180)) = 
  (Real.sqrt 2 * (4 * Real.cos (22.5 * π / 180) * Real.cos (7.5 * π / 180) + 1)) / 
  (Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * Real.cos (30 * π / 180)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l2126_212665


namespace NUMINAMATH_CALUDE_contrapositive_inequality_l2126_212689

theorem contrapositive_inequality (a b c : ℝ) :
  (¬(a < b) → ¬(a + c < b + c)) ↔ (a + c ≥ b + c → a ≥ b) := by sorry

end NUMINAMATH_CALUDE_contrapositive_inequality_l2126_212689


namespace NUMINAMATH_CALUDE_scarf_cost_l2126_212628

theorem scarf_cost (initial_amount : ℕ) (toy_car_cost : ℕ) (num_toy_cars : ℕ) (beanie_cost : ℕ) (remaining_amount : ℕ) : 
  initial_amount = 53 →
  toy_car_cost = 11 →
  num_toy_cars = 2 →
  beanie_cost = 14 →
  remaining_amount = 7 →
  initial_amount - (num_toy_cars * toy_car_cost + beanie_cost + remaining_amount) = 10 := by
sorry

end NUMINAMATH_CALUDE_scarf_cost_l2126_212628


namespace NUMINAMATH_CALUDE_line_through_midpoint_of_ellipse_chord_l2126_212629

/-- The ellipse in the problem -/
def ellipse (x y : ℝ) : Prop := x^2 / 64 + y^2 / 16 = 1

/-- The line we're trying to find -/
def line (x y : ℝ) : Prop := x + 8*y - 17 = 0

/-- Theorem stating that the line passing through the midpoint (1, 2) of a chord of the given ellipse
    has the equation x + 8y - 17 = 0 -/
theorem line_through_midpoint_of_ellipse_chord :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧  -- The endpoints of the chord lie on the ellipse
    (x₁ + x₂) / 2 = 1 ∧ (y₁ + y₂) / 2 = 2 ∧  -- (1, 2) is the midpoint of the chord
    ∀ (x y : ℝ), line x y ↔ y - 2 = (-1/8) * (x - 1) :=  -- The line equation is correct
by sorry

end NUMINAMATH_CALUDE_line_through_midpoint_of_ellipse_chord_l2126_212629


namespace NUMINAMATH_CALUDE_jose_weekly_earnings_l2126_212668

/-- Calculates Jose's weekly earnings from his swimming pool. -/
theorem jose_weekly_earnings :
  let kid_price : ℕ := 3
  let adult_price : ℕ := 2 * kid_price
  let kids_per_day : ℕ := 8
  let adults_per_day : ℕ := 10
  let days_per_week : ℕ := 7
  
  (kid_price * kids_per_day + adult_price * adults_per_day) * days_per_week = 588 :=
by sorry

end NUMINAMATH_CALUDE_jose_weekly_earnings_l2126_212668


namespace NUMINAMATH_CALUDE_student_tickets_sold_l2126_212660

theorem student_tickets_sold (adult_price student_price total_tickets total_amount : ℚ)
  (h1 : adult_price = 4)
  (h2 : student_price = (5/2))
  (h3 : total_tickets = 59)
  (h4 : total_amount = (445/2))
  (h5 : ∃ (adult_tickets student_tickets : ℚ),
    adult_tickets + student_tickets = total_tickets ∧
    adult_price * adult_tickets + student_price * student_tickets = total_amount) :
  ∃ (student_tickets : ℚ), student_tickets = 9 := by
sorry

end NUMINAMATH_CALUDE_student_tickets_sold_l2126_212660


namespace NUMINAMATH_CALUDE_frustum_cone_altitude_l2126_212638

theorem frustum_cone_altitude (h : ℝ) (A_lower A_upper : ℝ) :
  h = 24 →
  A_lower = 225 * Real.pi →
  A_upper = 25 * Real.pi →
  ∃ x : ℝ, x = 12 ∧ x = (1/3) * (3/2 * h) :=
by sorry

end NUMINAMATH_CALUDE_frustum_cone_altitude_l2126_212638


namespace NUMINAMATH_CALUDE_last_digit_of_large_prime_l2126_212615

theorem last_digit_of_large_prime (n : ℕ) (h : n = 859433) :
  (2^n - 1) % 10 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_large_prime_l2126_212615


namespace NUMINAMATH_CALUDE_triangle_angle_relation_l2126_212666

theorem triangle_angle_relation (A B C : Real) : 
  A + B + C = Real.pi →
  0 < A ∧ A < Real.pi →
  0 < B ∧ B < Real.pi →
  0 < C ∧ C < Real.pi →
  Real.sin A = Real.cos B →
  Real.sin A = Real.tan C →
  Real.cos A ^ 3 + Real.cos A ^ 2 - Real.cos A = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_relation_l2126_212666


namespace NUMINAMATH_CALUDE_three_black_reachable_l2126_212662

structure UrnState :=
  (black : ℕ)
  (white : ℕ)

def initial_state : UrnState :=
  ⟨100, 120⟩

inductive Operation
  | replace_3b_with_2b
  | replace_2b1w_with_1b1w
  | replace_1b2w_with_2w
  | replace_3w_with_1b1w

def apply_operation (state : UrnState) (op : Operation) : UrnState :=
  match op with
  | Operation.replace_3b_with_2b => ⟨state.black - 1, state.white⟩
  | Operation.replace_2b1w_with_1b1w => ⟨state.black - 1, state.white⟩
  | Operation.replace_1b2w_with_2w => ⟨state.black - 1, state.white⟩
  | Operation.replace_3w_with_1b1w => ⟨state.black + 1, state.white - 2⟩

def reachable (target : UrnState) : Prop :=
  ∃ (n : ℕ) (ops : Fin n → Operation),
    (List.foldl apply_operation initial_state (List.ofFn ops)) = target

theorem three_black_reachable :
  reachable ⟨3, 120⟩ :=
sorry

end NUMINAMATH_CALUDE_three_black_reachable_l2126_212662


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_seven_l2126_212687

theorem largest_n_divisible_by_seven : 
  ∀ n : ℕ, n < 50000 → 
  (6 * (n - 3)^3 - n^2 + 10*n - 15) % 7 = 0 → 
  n ≤ 49999 ∧ 
  (6 * (49999 - 3)^3 - 49999^2 + 10*49999 - 15) % 7 = 0 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_seven_l2126_212687


namespace NUMINAMATH_CALUDE_consecutive_integers_divisibility_l2126_212653

theorem consecutive_integers_divisibility (a₁ a₂ a₃ : ℕ) 
  (h1 : a₁ + 1 = a₂) 
  (h2 : a₂ + 1 = a₃) 
  (h3 : 0 < a₁) : 
  a₂^3 ∣ (a₁ * a₂ * a₃ + a₂) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_divisibility_l2126_212653


namespace NUMINAMATH_CALUDE_bread_leftover_l2126_212616

theorem bread_leftover (total_length : Real) (jimin_eats_cm : Real) (taehyung_eats_m : Real) :
  total_length = 30 ∧ jimin_eats_cm = 150 ∧ taehyung_eats_m = 1.65 →
  total_length - (jimin_eats_cm / 100 + taehyung_eats_m) = 26.85 := by
  sorry

end NUMINAMATH_CALUDE_bread_leftover_l2126_212616


namespace NUMINAMATH_CALUDE_distance_covered_l2126_212644

/-- Proves that the total distance covered is 10 km given the specified conditions --/
theorem distance_covered (walking_speed running_speed : ℝ) (total_time : ℝ) 
  (h1 : walking_speed = 4)
  (h2 : running_speed = 8)
  (h3 : total_time = 3.75)
  (h4 : (total_distance / 2) / walking_speed + (total_distance / 2) / running_speed = total_time) :
  total_distance = 10 :=
by sorry

#check distance_covered

end NUMINAMATH_CALUDE_distance_covered_l2126_212644


namespace NUMINAMATH_CALUDE_proper_divisor_of_two_square_representations_l2126_212625

theorem proper_divisor_of_two_square_representations (n s t u v : ℕ) 
  (h1 : n = s^2 + t^2)
  (h2 : n = u^2 + v^2)
  (h3 : s ≥ t)
  (h4 : t ≥ 0)
  (h5 : u ≥ v)
  (h6 : v ≥ 0)
  (h7 : s > u) :
  1 < Nat.gcd (s * u - t * v) n ∧ Nat.gcd (s * u - t * v) n < n :=
by sorry

end NUMINAMATH_CALUDE_proper_divisor_of_two_square_representations_l2126_212625


namespace NUMINAMATH_CALUDE_linear_function_property_l2126_212659

/-- A linear function is a function f such that f(x) = mx + b for some constants m and b. -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

/-- Given a linear function g such that g(10) - g(4) = 24, prove that g(16) - g(4) = 48. -/
theorem linear_function_property (g : ℝ → ℝ) 
  (h_linear : LinearFunction g) 
  (h_condition : g 10 - g 4 = 24) : 
  g 16 - g 4 = 48 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_property_l2126_212659


namespace NUMINAMATH_CALUDE_inequality_proof_l2126_212651

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2*a + b + c)^2 / (2*a^2 + (b + c)^2) +
  (2*b + a + c)^2 / (2*b^2 + (c + a)^2) +
  (2*c + a + b)^2 / (2*c^2 + (a + b)^2) ≤ 8 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2126_212651


namespace NUMINAMATH_CALUDE_middle_school_count_l2126_212635

structure School where
  total_students : ℕ
  sample_size : ℕ
  middle_school_in_sample : ℕ

def middle_school_students (s : School) : ℕ :=
  s.total_students * s.middle_school_in_sample / s.sample_size

theorem middle_school_count (s : School) 
  (h1 : s.total_students = 2000)
  (h2 : s.sample_size = 400)
  (h3 : s.middle_school_in_sample = 180) :
  middle_school_students s = 900 := by
  sorry

end NUMINAMATH_CALUDE_middle_school_count_l2126_212635


namespace NUMINAMATH_CALUDE_even_cubic_implies_odd_factor_l2126_212621

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function g: ℝ → ℝ is odd if g(-x) = -g(x) for all x ∈ ℝ -/
def IsOdd (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g x

/-- Given f(x) = x³ * g(x) is an even function, prove that g(x) is odd -/
theorem even_cubic_implies_odd_factor
    (g : ℝ → ℝ) (f : ℝ → ℝ)
    (h1 : ∀ x, f x = x^3 * g x)
    (h2 : IsEven f) :
  IsOdd g :=
by sorry

end NUMINAMATH_CALUDE_even_cubic_implies_odd_factor_l2126_212621


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l2126_212672

theorem sum_of_absolute_coefficients (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (1 - 2*x)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  |a₀| + |a₁| + |a₂| + |a₃| + |a₄| = 81 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l2126_212672


namespace NUMINAMATH_CALUDE_fourth_day_temperature_l2126_212606

def temperature_problem (t1 t2 t3 avg : ℚ) : Prop :=
  let sum3 := t1 + t2 + t3
  let sum4 := 4 * avg
  sum4 - sum3 = -36

theorem fourth_day_temperature :
  temperature_problem 13 (-15) (-10) (-12) := by sorry

end NUMINAMATH_CALUDE_fourth_day_temperature_l2126_212606


namespace NUMINAMATH_CALUDE_negation_of_forall_square_ge_self_l2126_212685

theorem negation_of_forall_square_ge_self :
  (¬ ∀ x : ℕ, x^2 ≥ x) ↔ (∃ x : ℕ, x^2 < x) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_square_ge_self_l2126_212685


namespace NUMINAMATH_CALUDE_limit_f_at_zero_l2126_212692

open Real
open Filter
open Topology

noncomputable def f (x : ℝ) : ℝ := Real.log ((Real.exp (x^2) - Real.cos x) * Real.cos (1/x) + Real.tan (x + π/3))

theorem limit_f_at_zero : 
  Tendsto f (𝓝 0) (𝓝 ((1/2) * Real.log 3)) := by sorry

end NUMINAMATH_CALUDE_limit_f_at_zero_l2126_212692


namespace NUMINAMATH_CALUDE_parakeets_per_cage_l2126_212674

theorem parakeets_per_cage (num_cages : ℕ) (parrots_per_cage : ℕ) (total_birds : ℕ) :
  num_cages = 6 →
  parrots_per_cage = 6 →
  total_birds = 48 →
  (total_birds - num_cages * parrots_per_cage) / num_cages = 2 := by
  sorry

end NUMINAMATH_CALUDE_parakeets_per_cage_l2126_212674


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l2126_212624

theorem arithmetic_expression_evaluation : 15 - 2 + 4 / 1 / 2 * 8 = 29 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l2126_212624


namespace NUMINAMATH_CALUDE_equation_one_solutions_l2126_212620

theorem equation_one_solutions (x : ℝ) : x * (x - 2) = x - 2 ↔ x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_one_solutions_l2126_212620


namespace NUMINAMATH_CALUDE_train_speed_problem_l2126_212630

/-- Calculates the speed of train A given the conditions of the problem -/
theorem train_speed_problem (length_A length_B : ℝ) (speed_B : ℝ) (crossing_time : ℝ) :
  length_A = 150 →
  length_B = 150 →
  speed_B = 36 →
  crossing_time = 12 →
  (length_A + length_B) / crossing_time * 3.6 - speed_B = 54 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l2126_212630


namespace NUMINAMATH_CALUDE_pigs_in_blanket_calculation_l2126_212623

/-- The number of appetizers per guest -/
def appetizers_per_guest : ℕ := 6

/-- The number of guests -/
def number_of_guests : ℕ := 30

/-- The number of dozen deviled eggs -/
def dozen_deviled_eggs : ℕ := 3

/-- The number of dozen kebabs -/
def dozen_kebabs : ℕ := 2

/-- The additional number of dozen appetizers to make -/
def additional_dozen_appetizers : ℕ := 8

/-- The number of items in a dozen -/
def items_per_dozen : ℕ := 12

theorem pigs_in_blanket_calculation : 
  let total_appetizers := appetizers_per_guest * number_of_guests
  let made_appetizers := dozen_deviled_eggs * items_per_dozen + dozen_kebabs * items_per_dozen
  let remaining_appetizers := total_appetizers - made_appetizers
  let planned_additional_appetizers := additional_dozen_appetizers * items_per_dozen
  let pigs_in_blanket := remaining_appetizers - planned_additional_appetizers
  (pigs_in_blanket / items_per_dozen : ℕ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_pigs_in_blanket_calculation_l2126_212623


namespace NUMINAMATH_CALUDE_vector_magnitude_l2126_212661

theorem vector_magnitude (a b : ℝ × ℝ) :
  let angle := 60 * π / 180
  (a.1^2 + a.2^2 = 4) →
  (b.1^2 + b.2^2 = 1) →
  (a.1 * b.1 + a.2 * b.2 = 2 * Real.cos angle) →
  ((a.1 - 2*b.1)^2 + (a.2 - 2*b.2)^2 = 4) :=
by
  sorry

#check vector_magnitude

end NUMINAMATH_CALUDE_vector_magnitude_l2126_212661


namespace NUMINAMATH_CALUDE_intersection_M_N_l2126_212645

def U : Finset ℕ := {0, 1, 2, 3, 4}
def M : Finset ℕ := {0, 2, 3}
def complement_N : Finset ℕ := {1, 2, 4}

theorem intersection_M_N :
  (M ∩ (U \ complement_N) : Finset ℕ) = {0, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2126_212645


namespace NUMINAMATH_CALUDE_money_problem_l2126_212694

/-- Given three people A, B, and C with certain amounts of money, 
    prove that A and C together have 300 rupees. -/
theorem money_problem (a b c : ℕ) : 
  a + b + c = 700 →
  b + c = 600 →
  c = 200 →
  a + c = 300 := by
  sorry

end NUMINAMATH_CALUDE_money_problem_l2126_212694


namespace NUMINAMATH_CALUDE_point_quadrant_l2126_212612

/-- If a point A(a,b) is in the first quadrant, then the point B(a,-b) is in the fourth quadrant. -/
theorem point_quadrant (a b : ℝ) (h : a > 0 ∧ b > 0) : a > 0 ∧ -b < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_quadrant_l2126_212612


namespace NUMINAMATH_CALUDE_weekly_profit_calculation_l2126_212643

def planned_daily_sales : ℕ := 10

def daily_differences : List ℤ := [4, -3, -2, 7, -6, 18, -5]

def selling_price : ℕ := 65

def num_workers : ℕ := 3

def daily_expense_per_worker : ℕ := 80

def packaging_fee : ℕ := 5

def total_days : ℕ := 7

theorem weekly_profit_calculation :
  let total_sales := planned_daily_sales * total_days + daily_differences.sum
  let revenue := total_sales * (selling_price - packaging_fee)
  let expenses := num_workers * daily_expense_per_worker * total_days
  let profit := revenue - expenses
  profit = 3300 := by sorry

end NUMINAMATH_CALUDE_weekly_profit_calculation_l2126_212643


namespace NUMINAMATH_CALUDE_probability_one_girl_two_boys_l2126_212680

/-- The probability of having a boy or a girl for each child -/
def child_probability : ℝ := 0.5

/-- The number of children in the family -/
def num_children : ℕ := 3

/-- The number of ways to arrange 1 girl and 2 boys in 3 positions -/
def num_arrangements : ℕ := 3

/-- Theorem: The probability of having exactly 1 girl and 2 boys in a family with 3 children,
    where each child has an equal probability of being a boy or a girl, is 0.375 -/
theorem probability_one_girl_two_boys :
  (child_probability ^ num_children) * num_arrangements = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_girl_two_boys_l2126_212680


namespace NUMINAMATH_CALUDE_sin_c_special_triangle_l2126_212641

/-- Given a right triangle ABC where A is the right angle, if the logarithms of 
    the side lengths form an arithmetic sequence with a negative common difference, 
    then sin C equals (√5 - 1)/2 -/
theorem sin_c_special_triangle (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_right_angle : a^2 = b^2 + c^2)
  (h_arithmetic_seq : ∃ d : ℝ, d < 0 ∧ Real.log a - Real.log b = d ∧ Real.log b - Real.log c = d) :
  Real.sin (Real.arccos (c / a)) = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_c_special_triangle_l2126_212641


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l2126_212682

theorem absolute_value_simplification : |(-4^2 + (5 - 2))| = 13 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l2126_212682


namespace NUMINAMATH_CALUDE_equations_represent_problem_l2126_212664

/-- Represents the money each person brought -/
structure Money where
  a : ℝ  -- Amount A brought
  b : ℝ  -- Amount B brought

/-- Checks if the given equations satisfy the conditions of the problem -/
def satisfies_conditions (m : Money) : Prop :=
  (m.a + (1/2) * m.b = 50) ∧ (m.b + (2/3) * m.a = 50)

/-- Theorem stating that the equations correctly represent the problem -/
theorem equations_represent_problem :
  ∃ (m : Money), satisfies_conditions m :=
sorry

end NUMINAMATH_CALUDE_equations_represent_problem_l2126_212664


namespace NUMINAMATH_CALUDE_ln_squared_plus_ln_inequality_l2126_212650

theorem ln_squared_plus_ln_inequality (x : ℝ) :
  x > 0 → (Real.log x ^ 2 + Real.log x < 0 ↔ Real.exp (-1) < x ∧ x < 1) := by
  sorry

end NUMINAMATH_CALUDE_ln_squared_plus_ln_inequality_l2126_212650


namespace NUMINAMATH_CALUDE_max_balls_in_cube_l2126_212647

theorem max_balls_in_cube (cube_volume ball_volume : ℝ) (h1 : cube_volume = 1000) (h2 : ball_volume = 36 * Real.pi) :
  ⌊cube_volume / ball_volume⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_balls_in_cube_l2126_212647


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2126_212656

theorem imaginary_part_of_complex_fraction : Complex.im ((2 + Complex.I) / (1 - 2 * Complex.I)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2126_212656


namespace NUMINAMATH_CALUDE_sales_growth_rate_l2126_212617

theorem sales_growth_rate (initial_sales final_sales : ℝ) 
  (h1 : initial_sales = 2000000)
  (h2 : final_sales = 2880000)
  (h3 : ∃ r : ℝ, initial_sales * (1 + r)^2 = final_sales) :
  ∃ r : ℝ, initial_sales * (1 + r)^2 = final_sales ∧ r = 0.2 :=
sorry

end NUMINAMATH_CALUDE_sales_growth_rate_l2126_212617


namespace NUMINAMATH_CALUDE_ducks_in_lake_l2126_212698

/-- The number of ducks initially in the lake -/
def initial_ducks : ℕ := 13

/-- The number of ducks that joined the lake -/
def joining_ducks : ℕ := 20

/-- The total number of ducks in the lake -/
def total_ducks : ℕ := initial_ducks + joining_ducks

theorem ducks_in_lake : total_ducks = 33 := by
  sorry

end NUMINAMATH_CALUDE_ducks_in_lake_l2126_212698


namespace NUMINAMATH_CALUDE_action_figures_per_shelf_l2126_212639

theorem action_figures_per_shelf 
  (total_figures : ℕ) 
  (num_shelves : ℕ) 
  (h1 : total_figures = 80) 
  (h2 : num_shelves = 8) : 
  total_figures / num_shelves = 10 := by
  sorry

end NUMINAMATH_CALUDE_action_figures_per_shelf_l2126_212639


namespace NUMINAMATH_CALUDE_tree_planting_l2126_212699

theorem tree_planting (road_length : ℕ) (tree_spacing : ℕ) (h1 : road_length = 42) (h2 : tree_spacing = 7) : 
  road_length / tree_spacing + 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_l2126_212699


namespace NUMINAMATH_CALUDE_seventeen_flavors_l2126_212678

/-- Represents the number of different flavors possible given blue and orange candies. -/
def number_of_flavors (blue : ℕ) (orange : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that given 5 blue candies and 4 orange candies, 
    the number of different possible flavors is 17. -/
theorem seventeen_flavors : number_of_flavors 5 4 = 17 := by
  sorry

end NUMINAMATH_CALUDE_seventeen_flavors_l2126_212678


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2126_212673

theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (2^a * 2^b)) : 
  (∀ x y, x > 0 → y > 0 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2126_212673


namespace NUMINAMATH_CALUDE_trajectory_theorem_l2126_212696

/-- The trajectory of point M -/
def trajectory_M (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 4 * ((x - 4)^2 + y^2)

/-- The trajectory of point P -/
def trajectory_P (x y : ℝ) : Prop :=
  (x - 1/2)^2 + y^2 = 1

/-- The main theorem -/
theorem trajectory_theorem :
  (∀ x y : ℝ, trajectory_M x y ↔ x^2 + y^2 = 4) ∧
  (∀ x y : ℝ, (∃ a b : ℝ, trajectory_M a b ∧ x = (a + 1) / 2 ∧ y = b / 2) → trajectory_P x y) :=
sorry

end NUMINAMATH_CALUDE_trajectory_theorem_l2126_212696


namespace NUMINAMATH_CALUDE_sqrt_diff_inequality_l2126_212600

theorem sqrt_diff_inequality (n : ℕ) (h : n ≥ 2) :
  Real.sqrt (n - 1) - Real.sqrt n < Real.sqrt n - Real.sqrt (n + 1) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_diff_inequality_l2126_212600


namespace NUMINAMATH_CALUDE_sticks_in_yard_l2126_212691

theorem sticks_in_yard (picked_up left : ℕ) 
  (h1 : picked_up = 38) 
  (h2 : left = 61) : 
  picked_up + left = 99 := by
  sorry

end NUMINAMATH_CALUDE_sticks_in_yard_l2126_212691


namespace NUMINAMATH_CALUDE_equation_solution_l2126_212663

theorem equation_solution : 
  ∀ x : ℝ, x * (x - 1) = x ↔ x = 0 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2126_212663


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l2126_212655

theorem sum_of_fourth_powers (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = 1) : x^4 + y^4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l2126_212655


namespace NUMINAMATH_CALUDE_four_lighthouses_cover_plane_l2126_212611

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a 90-degree angle in the plane -/
inductive Quadrant
  | NE
  | SE
  | SW
  | NW

/-- Represents a lighthouse with its position and illumination direction -/
structure Lighthouse where
  position : Point
  direction : Quadrant

/-- Checks if a point is illuminated by a lighthouse -/
def isIlluminated (p : Point) (l : Lighthouse) : Prop :=
  sorry

/-- The main theorem: four lighthouses can illuminate the entire plane -/
theorem four_lighthouses_cover_plane (a b c d : Point) :
  ∃ (la lb lc ld : Lighthouse),
    la.position = a ∧ lb.position = b ∧ lc.position = c ∧ ld.position = d ∧
    ∀ p : Point, isIlluminated p la ∨ isIlluminated p lb ∨ isIlluminated p lc ∨ isIlluminated p ld :=
  sorry


end NUMINAMATH_CALUDE_four_lighthouses_cover_plane_l2126_212611


namespace NUMINAMATH_CALUDE_bakers_cake_inventory_l2126_212671

/-- Baker's cake inventory problem -/
theorem bakers_cake_inventory (cakes_made cakes_bought cakes_sold : ℕ) :
  cakes_made = 8 →
  cakes_bought = 139 →
  cakes_sold = 145 →
  cakes_sold - cakes_bought = 6 := by
  sorry

end NUMINAMATH_CALUDE_bakers_cake_inventory_l2126_212671


namespace NUMINAMATH_CALUDE_jones_trip_time_comparison_l2126_212681

/-- Proves that the time taken for the third trip is three times the time taken for the first trip
    given the conditions of Jones' three trips. -/
theorem jones_trip_time_comparison 
  (v : ℝ) -- Original speed
  (h1 : v > 0) -- Assumption that speed is positive
  (d1 : ℝ) (h2 : d1 = 40) -- Distance of first trip
  (d2 : ℝ) (h3 : d2 = 200) -- Distance of second trip
  (d3 : ℝ) (h4 : d3 = 480) -- Distance of third trip
  (v2 : ℝ) (h5 : v2 = 2 * v) -- Speed of second trip
  (v3 : ℝ) (h6 : v3 = 2 * v2) -- Speed of third trip
  : (d3 / v3) = 3 * (d1 / v) := by
  sorry

end NUMINAMATH_CALUDE_jones_trip_time_comparison_l2126_212681


namespace NUMINAMATH_CALUDE_complex_equality_implies_ratio_l2126_212676

theorem complex_equality_implies_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b * Complex.I) ^ 4 = (a - b * Complex.I) ^ 4 → b / a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_implies_ratio_l2126_212676
