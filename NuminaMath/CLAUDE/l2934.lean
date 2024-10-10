import Mathlib

namespace max_value_product_l2934_293455

theorem max_value_product (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_sum : x/y + y/z + z/x + y/x + z/y + x/z = 9) :
  (x/y + y/z + z/x) * (y/x + z/y + x/z) ≤ 81/4 :=
by sorry

end max_value_product_l2934_293455


namespace train_speed_problem_l2934_293474

/-- Proves the speed of the second train given the conditions of the problem -/
theorem train_speed_problem (train_length : ℝ) (train1_speed : ℝ) (passing_time : ℝ) :
  train_length = 210 →
  train1_speed = 90 →
  passing_time = 8.64 →
  ∃ train2_speed : ℝ,
    train2_speed = 85 ∧
    (train_length * 2) / passing_time * 3.6 = train1_speed + train2_speed :=
by sorry

end train_speed_problem_l2934_293474


namespace f_2017_equals_neg_2_l2934_293406

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem f_2017_equals_neg_2 (f : ℝ → ℝ) 
  (h1 : is_odd_function f)
  (h2 : is_even_function (fun x ↦ f (x + 1)))
  (h3 : f (-1) = 2) : 
  f 2017 = -2 := by
  sorry

end f_2017_equals_neg_2_l2934_293406


namespace space_diagonals_of_Q_l2934_293469

/-- A convex polyhedron with specific properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  let total_line_segments := Q.vertices.choose 2
  let face_diagonals := 2 * Q.quadrilateral_faces
  total_line_segments - Q.edges - face_diagonals

/-- The specific polyhedron Q described in the problem -/
def Q : ConvexPolyhedron :=
  { vertices := 30
  , edges := 72
  , faces := 44
  , triangular_faces := 30
  , quadrilateral_faces := 14 }

theorem space_diagonals_of_Q :
  space_diagonals Q = 335 := by sorry

end space_diagonals_of_Q_l2934_293469


namespace prob_one_head_two_tails_l2934_293408

/-- The probability of getting one head and two tails when tossing three fair coins -/
theorem prob_one_head_two_tails : ℝ := by
  -- Define the number of possible outcomes when tossing three fair coins
  let total_outcomes : ℕ := 2^3

  -- Define the number of ways to get one head and two tails
  let favorable_outcomes : ℕ := 3

  -- Define the probability as the ratio of favorable outcomes to total outcomes
  let probability : ℝ := favorable_outcomes / total_outcomes

  -- Prove that this probability equals 3/8
  sorry

end prob_one_head_two_tails_l2934_293408


namespace statement_analysis_l2934_293486

-- Define the types of statements
inductive StatementType
  | Universal
  | Existential

-- Define a structure to represent a statement
structure Statement where
  content : String
  type : StatementType
  isTrue : Bool

-- Define the statements
def statement1 : Statement := {
  content := "The diagonals of a square are perpendicular bisectors of each other",
  type := StatementType.Universal,
  isTrue := true
}

def statement2 : Statement := {
  content := "All Chinese people speak Chinese",
  type := StatementType.Universal,
  isTrue := false
}

def statement3 : Statement := {
  content := "Some numbers are greater than their squares",
  type := StatementType.Existential,
  isTrue := true
}

def statement4 : Statement := {
  content := "Some real numbers have irrational square roots",
  type := StatementType.Existential,
  isTrue := true
}

-- Theorem to prove
theorem statement_analysis : 
  (statement1.type = StatementType.Universal ∧ statement1.isTrue) ∧
  (statement2.type = StatementType.Universal ∧ ¬statement2.isTrue) ∧
  (statement3.type = StatementType.Existential ∧ statement3.isTrue) ∧
  (statement4.type = StatementType.Existential ∧ statement4.isTrue) := by
  sorry


end statement_analysis_l2934_293486


namespace reciprocal_of_negative_four_point_five_l2934_293433

theorem reciprocal_of_negative_four_point_five :
  ((-4.5)⁻¹ : ℝ) = -2/9 := by sorry

end reciprocal_of_negative_four_point_five_l2934_293433


namespace parabola_focus_distance_l2934_293409

/-- Represents a parabola y^2 = 2px with p > 0 -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- A point on the parabola -/
structure PointOnParabola (para : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * para.p * x

theorem parabola_focus_distance (para : Parabola) 
  (A : PointOnParabola para) (h_x : A.x = 2) (h_dist : Real.sqrt ((A.x - para.p/2)^2 + A.y^2) = 6) :
  para.p = 8 := by
  sorry

end parabola_focus_distance_l2934_293409


namespace student_rank_l2934_293430

theorem student_rank (total : Nat) (rank_right : Nat) (rank_left : Nat) : 
  total = 21 → rank_right = 17 → rank_left = total - rank_right + 1 → rank_left = 5 := by
  sorry

end student_rank_l2934_293430


namespace part_one_part_two_l2934_293497

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- Part 1
theorem part_one (x : ℝ) (h : p x 1 ∧ q x) : 2 < x ∧ x < 3 := by sorry

-- Part 2
theorem part_two (a : ℝ) (h : ∀ x, ¬(p x a) → ¬(q x)) 
  (h_not_necessary : ∃ x, ¬(p x a) ∧ q x) : 1 < a ∧ a ≤ 2 := by sorry

end part_one_part_two_l2934_293497


namespace sarah_bottle_caps_l2934_293404

/-- The total number of bottle caps Sarah has at the end of the week -/
def total_bottle_caps (initial : ℕ) (day1 : ℕ) (day2 : ℕ) (day3 : ℕ) : ℕ :=
  initial + day1 + day2 + day3

/-- Theorem stating that Sarah's total bottle caps at the end of the week
    is equal to her initial count plus all purchased bottle caps -/
theorem sarah_bottle_caps : 
  total_bottle_caps 450 175 95 220 = 940 := by
  sorry

end sarah_bottle_caps_l2934_293404


namespace two_books_selection_ways_l2934_293407

/-- The number of ways to select two books of different subjects from three shelves -/
def select_two_books (chinese_books : ℕ) (math_books : ℕ) (english_books : ℕ) : ℕ :=
  chinese_books * math_books + chinese_books * english_books + math_books * english_books

/-- Theorem stating that selecting two books of different subjects from the given shelves results in 242 ways -/
theorem two_books_selection_ways :
  select_two_books 10 9 8 = 242 := by
  sorry

end two_books_selection_ways_l2934_293407


namespace fourteen_sided_figure_area_l2934_293465

/-- A fourteen-sided figure constructed on a 1 cm × 1 cm grid -/
structure FourteenSidedFigure where
  /-- The number of full unit squares inside the figure -/
  full_squares : ℕ
  /-- The number of small right-angled triangles along the boundaries -/
  boundary_triangles : ℕ
  /-- The figure has 14 sides -/
  sides : ℕ
  sides_eq : sides = 14

/-- The area of the fourteen-sided figure is 16 cm² -/
theorem fourteen_sided_figure_area (f : FourteenSidedFigure) : 
  f.full_squares + f.boundary_triangles / 2 = 16 := by
  sorry

end fourteen_sided_figure_area_l2934_293465


namespace number_times_24_equals_173_times_240_l2934_293443

theorem number_times_24_equals_173_times_240 : ∃ x : ℕ, x * 24 = 173 * 240 ∧ x = 1730 := by
  sorry

end number_times_24_equals_173_times_240_l2934_293443


namespace triangle_radii_inequality_l2934_293479

theorem triangle_radii_inequality (a b c r r_a r_b r_c : ℝ) 
    (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ r_a > 0 ∧ r_b > 0 ∧ r_c > 0)
    (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
    (h_inradius : r = (a * b * c) / (4 * (a + b + c) * (a * b + b * c + c * a - a^2 - b^2 - c^2)^(1/2)))
    (h_exradius_a : r_a = (a * (b + c - a)) / (4 * (a + b + c) * (a * b + b * c + c * a - a^2 - b^2 - c^2)^(1/2)))
    (h_exradius_b : r_b = (b * (c + a - b)) / (4 * (a + b + c) * (a * b + b * c + c * a - a^2 - b^2 - c^2)^(1/2)))
    (h_exradius_c : r_c = (c * (a + b - c)) / (4 * (a + b + c) * (a * b + b * c + c * a - a^2 - b^2 - c^2)^(1/2))) :
  (a + b + c) / (a^2 + b^2 + c^2)^(1/2) ≤ 2 * (r_a^2 + r_b^2 + r_c^2)^(1/2) / (r_a + r_b + r_c - 3 * r) := by
sorry

end triangle_radii_inequality_l2934_293479


namespace quadratic_floor_existence_l2934_293491

theorem quadratic_floor_existence (x : ℝ) : 
  (∃ a b : ℤ, ∀ x : ℝ, x^2 + a*x + b ≠ 0 ∧ ∃ y : ℝ, ⌊y^2⌋ + a*y + b = 0) ∧
  (¬∃ a b : ℤ, ∀ x : ℝ, x^2 + 2*a*x + b ≠ 0 ∧ ∃ y : ℝ, ⌊y^2⌋ + 2*a*y + b = 0) :=
by sorry

end quadratic_floor_existence_l2934_293491


namespace fraction_sum_zero_implies_one_zero_l2934_293442

theorem fraction_sum_zero_implies_one_zero (a b c : ℝ) :
  (a - b) / (1 + a * b) + (b - c) / (1 + b * c) + (c - a) / (1 + c * a) = 0 →
  (a - b) / (1 + a * b) = 0 ∨ (b - c) / (1 + b * c) = 0 ∨ (c - a) / (1 + c * a) = 0 := by
  sorry

end fraction_sum_zero_implies_one_zero_l2934_293442


namespace twenty_photos_needed_l2934_293434

/-- The minimum number of non-overlapping rectangular photos required to form a square -/
def min_photos_for_square (width : ℕ) (length : ℕ) : ℕ :=
  let square_side := Nat.lcm width length
  (square_side * square_side) / (width * length)

/-- Theorem stating that 20 photos of 12cm x 15cm are needed for the smallest square -/
theorem twenty_photos_needed : min_photos_for_square 12 15 = 20 := by
  sorry

end twenty_photos_needed_l2934_293434


namespace arithmetic_sequence_sum_l2934_293466

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the third and fourth terms is 1/2 -/
def third_fourth_sum (a : ℕ → ℚ) : Prop :=
  a 3 + a 4 = 1/2

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a → third_fourth_sum a → a 1 + a 6 = 1/2 := by
  sorry

end arithmetic_sequence_sum_l2934_293466


namespace regular_polygon_perimeter_l2934_293475

/-- A regular polygon with side length 7 units and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 0 ∧ 
  side_length = 7 ∧ 
  exterior_angle = 90 ∧ 
  exterior_angle = 360 / n →
  n * side_length = 28 := by
  sorry

end regular_polygon_perimeter_l2934_293475


namespace cleaning_event_calculation_l2934_293428

def total_members : ℕ := 2000
def adult_men_percentage : ℚ := 30 / 100
def senior_percentage : ℚ := 5 / 100
def child_teen_ratio : ℚ := 3 / 2
def child_collection_rate : ℚ := 3 / 2
def teen_collection_rate : ℕ := 3
def senior_collection_rate : ℕ := 1

theorem cleaning_event_calculation :
  let adult_men := (adult_men_percentage * total_members).floor
  let adult_women := 2 * adult_men
  let seniors := (senior_percentage * total_members).floor
  let children_and_teens := total_members - (adult_men + adult_women + seniors)
  let children := ((child_teen_ratio * children_and_teens) / (1 + child_teen_ratio)).floor
  let teenagers := children_and_teens - children
  ∃ (children teenagers : ℕ) (recyclable mixed various : ℚ),
    children = 60 ∧
    teenagers = 40 ∧
    recyclable = child_collection_rate * children ∧
    mixed = teen_collection_rate * teenagers ∧
    various = senior_collection_rate * seniors ∧
    recyclable = 90 ∧
    mixed = 120 ∧
    various = 100 :=
by
  sorry

end cleaning_event_calculation_l2934_293428


namespace problem_1_problem_2_problem_3_l2934_293402

-- Problem 1
theorem problem_1 : Real.sqrt 9 + |-2| - (-3)^2 + (π - 100)^0 = -3 := by sorry

-- Problem 2
theorem problem_2 (x : ℝ) : 
  (x^2 + 1 = 5) ↔ (x = 2 ∨ x = -2) := by sorry

-- Problem 3
theorem problem_3 (x : ℝ) :
  (x^2 = (x - 2)^2 + 7) ↔ (x = 11/4) := by sorry

end problem_1_problem_2_problem_3_l2934_293402


namespace chord_existence_l2934_293460

/-- A continuous curve in a 2D plane -/
def ContinuousCurve := Set (ℝ × ℝ)

/-- Defines if a curve connects two points -/
def connects (curve : ContinuousCurve) (A B : ℝ × ℝ) : Prop := sorry

/-- Defines if a curve has a chord of a given length parallel to a line segment -/
def has_parallel_chord (curve : ContinuousCurve) (A B : ℝ × ℝ) (length : ℝ) : Prop := sorry

/-- The distance between two points in 2D space -/
def distance (A B : ℝ × ℝ) : ℝ := sorry

theorem chord_existence (n : ℕ) (hn : n > 0) (A B : ℝ × ℝ) (curve : ContinuousCurve) :
  distance A B = 1 →
  connects curve A B →
  has_parallel_chord curve A B (1 / n) := by sorry

end chord_existence_l2934_293460


namespace book_arrangement_proof_l2934_293473

theorem book_arrangement_proof :
  let total_books : ℕ := 11
  let geometry_books : ℕ := 5
  let number_theory_books : ℕ := 6
  Nat.choose total_books geometry_books = 462 := by
  sorry

end book_arrangement_proof_l2934_293473


namespace betty_order_total_payment_l2934_293417

/-- Calculates the total payment for Betty's order including shipping -/
def totalPayment (
  slipperPrice : Float) (slipperWeight : Float) (slipperCount : Nat)
  (lipstickPrice : Float) (lipstickWeight : Float) (lipstickCount : Nat)
  (hairColorPrice : Float) (hairColorWeight : Float) (hairColorCount : Nat)
  (sunglassesPrice : Float) (sunglassesWeight : Float) (sunglassesCount : Nat)
  (tshirtPrice : Float) (tshirtWeight : Float) (tshirtCount : Nat)
  : Float :=
  let totalCost := 
    slipperPrice * slipperCount.toFloat +
    lipstickPrice * lipstickCount.toFloat +
    hairColorPrice * hairColorCount.toFloat +
    sunglassesPrice * sunglassesCount.toFloat +
    tshirtPrice * tshirtCount.toFloat
  let totalWeight :=
    slipperWeight * slipperCount.toFloat +
    lipstickWeight * lipstickCount.toFloat +
    hairColorWeight * hairColorCount.toFloat +
    sunglassesWeight * sunglassesCount.toFloat +
    tshirtWeight * tshirtCount.toFloat
  let shippingCost :=
    if totalWeight ≤ 5 then 2
    else if totalWeight ≤ 10 then 4
    else 6
  totalCost + shippingCost

theorem betty_order_total_payment :
  totalPayment 2.5 0.3 6 1.25 0.05 4 3 0.2 8 5.75 0.1 3 12.25 0.5 4 = 114.25 := by
  sorry

end betty_order_total_payment_l2934_293417


namespace power_inequality_l2934_293445

theorem power_inequality : 81^31 > 27^41 ∧ 27^41 > 9^61 := by sorry

end power_inequality_l2934_293445


namespace best_of_three_match_probability_l2934_293456

/-- The probability of player A winning a single game against player B. -/
def p_win_game : ℚ := 1/3

/-- The probability of player A winning a best-of-three match against player B. -/
def p_win_match : ℚ := 7/27

/-- Theorem stating that if the probability of player A winning each game is 1/3,
    then the probability of A winning a best-of-three match is 7/27. -/
theorem best_of_three_match_probability :
  p_win_game = 1/3 → p_win_match = 7/27 := by sorry

end best_of_three_match_probability_l2934_293456


namespace sin_m_theta_bound_l2934_293420

theorem sin_m_theta_bound (θ : ℝ) (m : ℕ) : 
  |Real.sin (m * θ)| ≤ m * |Real.sin θ| := by
  sorry

end sin_m_theta_bound_l2934_293420


namespace solve_for_a_l2934_293461

theorem solve_for_a : ∃ a : ℝ, (1/2 * 2 + a = -1) ∧ a = -2 := by
  sorry

end solve_for_a_l2934_293461


namespace sufficient_not_necessary_l2934_293405

theorem sufficient_not_necessary (a b : ℝ) : 
  (∀ a b, b ≥ 0 → a^2 + b ≥ 0) ∧ 
  (∃ a b, a^2 + b ≥ 0 ∧ b < 0) :=
by sorry

end sufficient_not_necessary_l2934_293405


namespace junghyeon_stickers_l2934_293453

/-- Given a total of 25 stickers shared between Junghyeon and Yejin, 
    where Junghyeon has 1 more sticker than twice Yejin's, 
    prove that Junghyeon will have 17 stickers. -/
theorem junghyeon_stickers : 
  ∀ (junghyeon_stickers yejin_stickers : ℕ),
  junghyeon_stickers + yejin_stickers = 25 →
  junghyeon_stickers = 2 * yejin_stickers + 1 →
  junghyeon_stickers = 17 := by
sorry

end junghyeon_stickers_l2934_293453


namespace monomial_division_l2934_293450

theorem monomial_division (x : ℝ) : 2 * x^3 / x^2 = 2 * x := by sorry

end monomial_division_l2934_293450


namespace system_solution_l2934_293440

theorem system_solution (x y : ℝ) : 
  (1 / (x^2 + y^2) + x^2 * y^2 = 5/4) ∧ 
  (2 * x^4 + 2 * y^4 + 5 * x^2 * y^2 = 9/4) ↔ 
  ((x = 1 / Real.sqrt 2 ∧ (y = 1 / Real.sqrt 2 ∨ y = -1 / Real.sqrt 2)) ∨
   (x = -1 / Real.sqrt 2 ∧ (y = 1 / Real.sqrt 2 ∨ y = -1 / Real.sqrt 2))) :=
by sorry

end system_solution_l2934_293440


namespace cottage_cheese_production_l2934_293492

/-- Represents the fat content balance in milk processing -/
def fat_balance (milk_mass : ℝ) (milk_fat : ℝ) (cheese_fat : ℝ) (whey_fat : ℝ) (cheese_mass : ℝ) : Prop :=
  milk_mass * milk_fat = cheese_mass * cheese_fat + (milk_mass - cheese_mass) * whey_fat

/-- Proves the amount of cottage cheese produced from milk -/
theorem cottage_cheese_production (milk_mass : ℝ) (milk_fat : ℝ) (cheese_fat : ℝ) (whey_fat : ℝ) 
  (h_milk_mass : milk_mass = 1)
  (h_milk_fat : milk_fat = 0.05)
  (h_cheese_fat : cheese_fat = 0.155)
  (h_whey_fat : whey_fat = 0.005) :
  ∃ cheese_mass : ℝ, cheese_mass = 0.3 ∧ fat_balance milk_mass milk_fat cheese_fat whey_fat cheese_mass :=
by
  sorry

#check cottage_cheese_production

end cottage_cheese_production_l2934_293492


namespace max_underwear_is_four_l2934_293403

/-- Represents the washing machine and clothing weights --/
structure WashingMachine where
  limit : Nat
  sock_weight : Nat
  underwear_weight : Nat
  shirt_weight : Nat
  shorts_weight : Nat
  pants_weight : Nat

/-- Represents the clothes Tony is washing --/
structure ClothesInWash where
  pants : Nat
  shirts : Nat
  shorts : Nat
  socks : Nat

/-- Calculates the maximum number of additional pairs of underwear that can be added --/
def max_additional_underwear (wm : WashingMachine) (clothes : ClothesInWash) : Nat :=
  let current_weight := 
    clothes.pants * wm.pants_weight +
    clothes.shirts * wm.shirt_weight +
    clothes.shorts * wm.shorts_weight +
    clothes.socks * wm.sock_weight
  let remaining_weight := wm.limit - current_weight
  remaining_weight / wm.underwear_weight

/-- Theorem stating that the maximum number of additional pairs of underwear is 4 --/
theorem max_underwear_is_four :
  let wm : WashingMachine := {
    limit := 50,
    sock_weight := 2,
    underwear_weight := 4,
    shirt_weight := 5,
    shorts_weight := 8,
    pants_weight := 10
  }
  let clothes : ClothesInWash := {
    pants := 1,
    shirts := 2,
    shorts := 1,
    socks := 3
  }
  max_additional_underwear wm clothes = 4 := by
  sorry

end max_underwear_is_four_l2934_293403


namespace snow_leopard_arrangements_l2934_293435

theorem snow_leopard_arrangements (n : ℕ) (h : n = 8) :
  2 * Nat.factorial (n - 2) = 1440 := by
  sorry

end snow_leopard_arrangements_l2934_293435


namespace house_rooms_count_l2934_293488

/-- The number of rooms with 4 walls -/
def rooms_with_four_walls : ℕ := 5

/-- The number of rooms with 5 walls -/
def rooms_with_five_walls : ℕ := 4

/-- The number of walls each person should paint -/
def walls_per_person : ℕ := 8

/-- The number of people in Amanda's family -/
def family_members : ℕ := 5

/-- The total number of rooms in the house -/
def total_rooms : ℕ := rooms_with_four_walls + rooms_with_five_walls

theorem house_rooms_count : total_rooms = 9 := by
  sorry

end house_rooms_count_l2934_293488


namespace triangle_property_l2934_293418

-- Define the necessary types and structures
structure Point := (x y : ℝ)

structure Triangle :=
  (A B C : Point)

structure Line :=
  (p1 p2 : Point)

-- Define the given conditions
def isAcute (t : Triangle) : Prop := sorry

def isOrthocenter (H : Point) (t : Triangle) : Prop := sorry

def lieOnSide (P : Point) (l : Line) : Prop := sorry

def angleEquals (A B C : Point) (angle : ℝ) : Prop := sorry

def intersectsAt (l1 l2 : Line) (P : Point) : Prop := sorry

def isCircumcenter (O : Point) (t : Triangle) : Prop := sorry

def sameSideAs (P Q : Point) (l : Line) : Prop := sorry

def collinear (P Q R : Point) : Prop := sorry

-- Define the theorem
theorem triangle_property 
  (ABC : Triangle) 
  (H M N P Q O E : Point) :
  isAcute ABC →
  isOrthocenter H ABC →
  lieOnSide M (Line.mk ABC.A ABC.B) →
  lieOnSide N (Line.mk ABC.A ABC.C) →
  angleEquals H M ABC.B (60 : ℝ) →
  angleEquals H N ABC.C (60 : ℝ) →
  intersectsAt (Line.mk H M) (Line.mk ABC.C ABC.A) P →
  intersectsAt (Line.mk H N) (Line.mk ABC.B ABC.A) Q →
  isCircumcenter O (Triangle.mk H M N) →
  angleEquals E ABC.B ABC.C (60 : ℝ) →
  sameSideAs E ABC.A (Line.mk ABC.B ABC.C) →
  collinear E O H →
  (Line.mk O H).p1 = (Line.mk P Q).p1 ∧ -- OH ⊥ PQ
  (Triangle.mk E ABC.B ABC.C).A = (Triangle.mk E ABC.B ABC.C).B ∧ 
  (Triangle.mk E ABC.B ABC.C).B = (Triangle.mk E ABC.B ABC.C).C -- Triangle EBC is equilateral
  := by sorry

end triangle_property_l2934_293418


namespace min_value_theorem_l2934_293483

theorem min_value_theorem (x : ℝ) (h : x > 0) : 
  (x^2 + 3*x + 2) / x ≥ 2 * Real.sqrt 2 + 3 := by
  sorry

end min_value_theorem_l2934_293483


namespace rational_cube_sum_representation_l2934_293437

theorem rational_cube_sum_representation (r : ℚ) (hr : 0 < r) :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ r = (a^3 + b^3 : ℚ) / (c^3 + d^3 : ℚ) := by
  sorry

end rational_cube_sum_representation_l2934_293437


namespace triangle_angle_inequality_l2934_293495

def f (x : ℝ) : ℝ := x^2014

theorem triangle_angle_inequality (α β : ℝ) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : α + β > π/2) : 
  f (Real.sin α) > f (Real.cos β) := by
  sorry

end triangle_angle_inequality_l2934_293495


namespace delta_phi_composition_l2934_293482

/-- Given two functions δ and φ, prove that δ(φ(x)) = 3 if and only if x = -19/20 -/
theorem delta_phi_composition (δ φ : ℝ → ℝ) (h1 : ∀ x, δ x = 4 * x + 6) (h2 : ∀ x, φ x = 5 * x + 4) :
  (∃ x, δ (φ x) = 3) ↔ (∃ x, x = -19/20) :=
by sorry

end delta_phi_composition_l2934_293482


namespace prob_red_then_black_custom_deck_l2934_293412

/-- A deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (black_cards : ℕ)

/-- The probability of drawing a red card first and then a black card from a shuffled deck -/
def prob_red_then_black (d : Deck) : ℚ :=
  (d.red_cards : ℚ) * (d.black_cards : ℚ) / ((d.total_cards : ℚ) * (d.total_cards - 1 : ℚ))

/-- The theorem stating the probability for the given deck -/
theorem prob_red_then_black_custom_deck :
  let d : Deck := ⟨60, 30, 30⟩
  prob_red_then_black d = 15 / 59 := by
  sorry

end prob_red_then_black_custom_deck_l2934_293412


namespace stream_speed_calculation_l2934_293462

/-- Represents the speed of a boat in still water -/
def boat_speed : ℝ := 15

/-- Represents the speed of the stream -/
def stream_speed : ℝ := 10

/-- Represents the downstream distance traveled -/
def downstream_distance : ℝ := 100

/-- Represents the upstream distance traveled -/
def upstream_distance : ℝ := 75

/-- Represents the time taken for downstream travel -/
def downstream_time : ℝ := 4

/-- Represents the time taken for upstream travel -/
def upstream_time : ℝ := 15

theorem stream_speed_calculation :
  (downstream_distance / downstream_time = boat_speed + stream_speed) ∧
  (upstream_distance / upstream_time = boat_speed - stream_speed) →
  stream_speed = 10 := by sorry

end stream_speed_calculation_l2934_293462


namespace red_rose_value_l2934_293490

def total_flowers : ℕ := 400
def tulips : ℕ := 120
def white_roses : ℕ := 80
def selling_price : ℚ := 75

def roses : ℕ := total_flowers - tulips
def red_roses : ℕ := roses - white_roses
def roses_to_sell : ℕ := red_roses / 2

theorem red_rose_value (total_flowers tulips white_roses selling_price : ℕ) 
  (h1 : total_flowers = 400)
  (h2 : tulips = 120)
  (h3 : white_roses = 80)
  (h4 : selling_price = 75) :
  (selling_price : ℚ) / roses_to_sell = 3/4 := by
  sorry

#eval (75 : ℚ) / 100  -- To verify the result is indeed 0.75

end red_rose_value_l2934_293490


namespace time_ratio_l2934_293441

def minutes_to_seconds (m : ℕ) : ℕ := m * 60

def hours_to_seconds (h : ℕ) : ℕ := h * 3600

def time_period_1 : ℕ := minutes_to_seconds 37 + 48

def time_period_2 : ℕ := hours_to_seconds 2 + minutes_to_seconds 13 + 15

theorem time_ratio : 
  time_period_1 * 7995 = time_period_2 * 2268 := by sorry

end time_ratio_l2934_293441


namespace french_exam_vocab_study_l2934_293485

/-- Represents the French exam vocabulary problem -/
theorem french_exam_vocab_study (total_words : ℕ) (recall_rate : ℚ) (guess_rate : ℚ) (target_score : ℚ) :
  let min_words : ℕ := 712
  total_words = 800 ∧ recall_rate = 1 ∧ guess_rate = 1/10 ∧ target_score = 9/10 →
  (↑min_words : ℚ) + guess_rate * (total_words - min_words) ≥ target_score * total_words ∧
  ∀ (x : ℕ), x < min_words →
    (↑x : ℚ) + guess_rate * (total_words - x) < target_score * total_words :=
by sorry

end french_exam_vocab_study_l2934_293485


namespace find_constant_k_l2934_293413

theorem find_constant_k (c : ℝ) (k : ℝ) :
  c = 2 →
  (∀ x : ℝ, -x^2 - (k + 10)*x - 8 = -(x - c)*(x - 4)) →
  k = -16 := by
  sorry

end find_constant_k_l2934_293413


namespace correct_students_joined_l2934_293444

/-- The number of students who joined Beth's class -/
def students_joined : ℕ := 30

/-- The initial number of students -/
def initial_students : ℕ := 150

/-- The number of students who left in the final year -/
def students_left : ℕ := 15

/-- The final number of students -/
def final_students : ℕ := 165

/-- Theorem stating that the number of students who joined is correct -/
theorem correct_students_joined :
  initial_students + students_joined - students_left = final_students :=
by sorry

end correct_students_joined_l2934_293444


namespace ball_arrangement_count_l2934_293463

/-- The number of ways to arrange 8 balls in a row, with 5 red balls (3 of which must be consecutive) and 3 white balls. -/
def ball_arrangements : ℕ := 30

/-- The total number of balls -/
def total_balls : ℕ := 8

/-- The number of red balls -/
def red_balls : ℕ := 5

/-- The number of white balls -/
def white_balls : ℕ := 3

/-- The number of consecutive red balls -/
def consecutive_red_balls : ℕ := 3

theorem ball_arrangement_count : 
  ball_arrangements = (Nat.choose (total_balls - consecutive_red_balls + 1) white_balls) * 
                      (Nat.choose (total_balls - white_balls - consecutive_red_balls + 1) 1) / 
                      (Nat.factorial (red_balls - consecutive_red_balls)) :=
sorry

end ball_arrangement_count_l2934_293463


namespace weight_of_doubled_cube_l2934_293487

/-- Given a cubical block of metal weighing 8 pounds, proves that another cube of the same metal with sides twice as long will weigh 64 pounds. -/
theorem weight_of_doubled_cube (s : ℝ) (h : s > 0) : 
  let original_weight : ℝ := 8
  let original_volume : ℝ := s^3
  let density : ℝ := original_weight / original_volume
  let new_side_length : ℝ := 2 * s
  let new_volume : ℝ := new_side_length^3
  let new_weight : ℝ := density * new_volume
  new_weight = 64 := by
  sorry

end weight_of_doubled_cube_l2934_293487


namespace special_line_equation_l2934_293489

/-- A line passing through point A(-3, 4) with x-intercept twice the y-intercept -/
structure SpecialLine where
  -- The slope-intercept form of the line: y = mx + b
  slope : ℝ
  y_intercept : ℝ
  -- The line passes through (-3, 4)
  point_condition : 4 = slope * (-3) + y_intercept
  -- The x-intercept is twice the y-intercept
  intercept_condition : -2 * y_intercept = y_intercept / slope

/-- The equation of the special line is either 3y + 4x = 0 or 2x - y - 5 = 0 -/
theorem special_line_equation (L : SpecialLine) :
  (3 * L.slope + 4 = 0 ∧ 3 * L.y_intercept = 0) ∨
  (2 = L.slope ∧ -5 = L.y_intercept) :=
sorry

end special_line_equation_l2934_293489


namespace number_of_boys_in_school_l2934_293439

theorem number_of_boys_in_school : 
  ∃ (x : ℕ), 
    (x + (x * 900 / 100) = 900) ∧ 
    (x = 90) := by
  sorry

end number_of_boys_in_school_l2934_293439


namespace smallest_n_for_divisibility_l2934_293452

theorem smallest_n_for_divisibility (x y z : ℕ+) 
  (h1 : x ∣ y^3) (h2 : y ∣ z^3) (h3 : z ∣ x^3) :
  (∀ n : ℕ, n < 13 → ¬(x * y * z ∣ (x + y + z)^n)) ∧ 
  (x * y * z ∣ (x + y + z)^13) :=
sorry

end smallest_n_for_divisibility_l2934_293452


namespace smallest_m_divisibility_l2934_293436

theorem smallest_m_divisibility : ∃! m : ℕ,
  (∀ n : ℕ, Odd n → (148^n + m * 141^n) % 2023 = 0) ∧
  (∀ k : ℕ, k < m → ∃ n : ℕ, Odd n ∧ (148^n + k * 141^n) % 2023 ≠ 0) ∧
  m = 1735 :=
by sorry

end smallest_m_divisibility_l2934_293436


namespace duck_average_l2934_293499

theorem duck_average (adelaide ephraim kolton : ℕ) : 
  adelaide = 30 →
  adelaide = 2 * ephraim →
  kolton = ephraim + 45 →
  (adelaide + ephraim + kolton) / 3 = 35 := by
sorry

end duck_average_l2934_293499


namespace geometric_sequence_sum_l2934_293471

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 2 = (1 : ℝ) / 4 →
  a 2 * a 8 = 4 * (a 5 - 1) →
  a 4 + a 5 + a 6 + a 7 + a 8 = 31 := by
  sorry

end geometric_sequence_sum_l2934_293471


namespace circle_area_diameter_increase_l2934_293432

theorem circle_area_diameter_increase : 
  ∀ (A D A' D' : ℝ), 
  A > 0 → D > 0 → 
  A = (Real.pi / 4) * D^2 →
  A' = 4 * A →
  A' = (Real.pi / 4) * D'^2 →
  D' / D - 1 = 1 :=
by
  sorry

end circle_area_diameter_increase_l2934_293432


namespace intersection_A_B_union_A_complement_B_l2934_293454

-- Define the universal set U
def U : Set ℝ := {x | x^2 - (5/2)*x + 1 ≥ 0}

-- Define set A
def A : Set ℝ := {x | |x - 1| > 1}

-- Define set B
def B : Set ℝ := {x | (x + 1)/(x - 2) ≥ 0}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x | x ≤ -1 ∨ x > 2} := by sorry

-- Theorem for A ∪ (CᵤB)
theorem union_A_complement_B : A ∪ (U \ B) = U := by sorry

end intersection_A_B_union_A_complement_B_l2934_293454


namespace arithmetic_sequence_length_l2934_293422

/-- An arithmetic sequence with first term 7, second term 11, and last term 95 has 23 terms. -/
theorem arithmetic_sequence_length : 
  ∀ (a : ℕ → ℕ), 
    (∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence
    a 0 = 7 →                                -- first term is 7
    a 1 = 11 →                               -- second term is 11
    (∃ m : ℕ, a m = 95 ∧ ∀ k > m, a k > 95) →  -- last term is 95
    ∃ n : ℕ, n = 23 ∧ a (n - 1) = 95 := by
  sorry


end arithmetic_sequence_length_l2934_293422


namespace quadratic_inequality_problem_l2934_293493

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 5 * x - 2

-- Define the solution set of f(x) > 0
def solution_set (a : ℝ) : Set ℝ := {x | f a x > 0}

-- Define the second quadratic function
def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 5 * x + a^2 - 1

-- State the theorem
theorem quadratic_inequality_problem (a : ℝ) :
  solution_set a = {x | 1/2 < x ∧ x < 2} →
  (a = -2 ∧ {x | g a x > 0} = {x | -3 < x ∧ x < 1/2}) :=
by sorry

end quadratic_inequality_problem_l2934_293493


namespace workshop_average_salary_l2934_293467

theorem workshop_average_salary 
  (num_technicians : ℕ)
  (num_total_workers : ℕ)
  (avg_salary_technicians : ℚ)
  (avg_salary_others : ℚ)
  (h1 : num_technicians = 7)
  (h2 : num_total_workers = 22)
  (h3 : avg_salary_technicians = 1000)
  (h4 : avg_salary_others = 780) :
  (num_technicians * avg_salary_technicians + (num_total_workers - num_technicians) * avg_salary_others) / num_total_workers = 850 := by
sorry

end workshop_average_salary_l2934_293467


namespace simplify_expression_l2934_293476

theorem simplify_expression (p : ℝ) (h1 : 1 < p) (h2 : p < 2) :
  Real.sqrt ((1 - p)^2) + (Real.sqrt (2 - p))^2 = 1 := by
  sorry

end simplify_expression_l2934_293476


namespace battle_station_staffing_l2934_293449

theorem battle_station_staffing (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 5) :
  (n.factorial / (n - k).factorial) = 30240 := by
  sorry

end battle_station_staffing_l2934_293449


namespace triangle_angle_and_max_area_l2934_293429

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition for the triangle -/
def triangleCondition (t : Triangle) : Prop :=
  cos t.B / cos t.C = -t.b / (2 * t.a + t.c)

theorem triangle_angle_and_max_area (t : Triangle) 
  (h : triangleCondition t) : 
  t.B = 2 * π / 3 ∧ 
  (t.b = 3 → ∃ (maxArea : ℝ), maxArea = 3 * sqrt 3 / 4 ∧ 
    ∀ (area : ℝ), area ≤ maxArea) := by
  sorry


end triangle_angle_and_max_area_l2934_293429


namespace cleaning_payment_l2934_293451

theorem cleaning_payment (payment_per_room : ℚ) (rooms_cleaned : ℚ) (discount_rate : ℚ) :
  payment_per_room = 13/3 →
  rooms_cleaned = 5/2 →
  discount_rate = 1/10 →
  (payment_per_room * rooms_cleaned) * (1 - discount_rate) = 39/4 := by
  sorry

end cleaning_payment_l2934_293451


namespace hockey_league_teams_l2934_293401

/-- The number of teams in a hockey league. -/
def num_teams : ℕ := 16

/-- The number of times each team faces every other team. -/
def games_per_pair : ℕ := 10

/-- The total number of games played in the season. -/
def total_games : ℕ := 1200

/-- Theorem stating that the number of teams is correct given the conditions. -/
theorem hockey_league_teams :
  (num_teams * (num_teams - 1) * games_per_pair) / 2 = total_games :=
sorry

end hockey_league_teams_l2934_293401


namespace negative_discriminant_implies_no_real_roots_l2934_293457

/-- Represents a quadratic equation of the form ax² + bx + c = 0 -/
structure QuadraticEquation (α : Type*) [Field α] where
  a : α
  b : α
  c : α

/-- Calculates the discriminant of a quadratic equation -/
def discriminant {α : Type*} [Field α] (eq : QuadraticEquation α) : α :=
  eq.b ^ 2 - 4 * eq.a * eq.c

/-- Represents the property of having real roots -/
def has_real_roots {α : Type*} [Field α] (eq : QuadraticEquation α) : Prop :=
  ∃ x : α, eq.a * x ^ 2 + eq.b * x + eq.c = 0

theorem negative_discriminant_implies_no_real_roots 
  {k : ℝ} (eq : QuadraticEquation ℝ) 
  (h_eq : eq = { a := 3, b := -4 * Real.sqrt 3, c := k }) 
  (h_discr : discriminant eq < 0) : 
  ¬ has_real_roots eq :=
sorry

end negative_discriminant_implies_no_real_roots_l2934_293457


namespace log_x2y2_value_l2934_293494

theorem log_x2y2_value (x y : ℝ) (h1 : Real.log (x * y^2) = 2) (h2 : Real.log (x^3 * y) = 2) :
  Real.log (x^2 * y^2) = 12/5 := by sorry

end log_x2y2_value_l2934_293494


namespace competition_result_competition_result_proof_l2934_293427

-- Define the type for students
inductive Student : Type
  | A | B | C | D | E

-- Define a type for the competition order
def CompetitionOrder := List Student

-- Define the first person's prediction
def firstPrediction : CompetitionOrder :=
  [Student.A, Student.B, Student.C, Student.D, Student.E]

-- Define the second person's prediction
def secondPrediction : CompetitionOrder :=
  [Student.D, Student.A, Student.E, Student.C, Student.B]

-- Function to check if a student is in the correct position
def correctPosition (actual : CompetitionOrder) (predicted : CompetitionOrder) (index : Nat) : Prop :=
  actual.get? index = predicted.get? index

-- Function to check if adjacent pairs are correct
def correctAdjacentPair (actual : CompetitionOrder) (predicted : CompetitionOrder) (index : Nat) : Prop :=
  actual.get? index = predicted.get? index ∧ actual.get? (index + 1) = predicted.get? (index + 1)

-- Main theorem
theorem competition_result (actual : CompetitionOrder) : Prop :=
  (actual.length = 5) ∧
  (∀ i, i < 5 → ¬correctPosition actual firstPrediction i) ∧
  (∀ i, i < 4 → ¬correctAdjacentPair actual firstPrediction i) ∧
  ((correctPosition actual secondPrediction 0 ∧ correctPosition actual secondPrediction 1) ∨
   (correctPosition actual secondPrediction 1 ∧ correctPosition actual secondPrediction 2) ∨
   (correctPosition actual secondPrediction 2 ∧ correctPosition actual secondPrediction 3) ∨
   (correctPosition actual secondPrediction 3 ∧ correctPosition actual secondPrediction 4)) ∧
  ((correctAdjacentPair actual secondPrediction 0 ∧ correctAdjacentPair actual secondPrediction 2) ∨
   (correctAdjacentPair actual secondPrediction 0 ∧ correctAdjacentPair actual secondPrediction 3) ∨
   (correctAdjacentPair actual secondPrediction 1 ∧ correctAdjacentPair actual secondPrediction 3)) ∧
  (actual = [Student.E, Student.D, Student.A, Student.C, Student.B])

-- Proof of the theorem
theorem competition_result_proof : ∃ actual, competition_result actual := by
  sorry


end competition_result_competition_result_proof_l2934_293427


namespace min_sum_squares_l2934_293472

theorem min_sum_squares (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 3) :
  ∃ (m : ℝ), m = 9 ∧ ∀ (a b c : ℝ), a ≥ 0 → b ≥ 0 → c ≥ 0 → a + b + c = 3 → a^2 + b^2 + c^2 ≥ m :=
by sorry

end min_sum_squares_l2934_293472


namespace m_equals_one_sufficient_not_necessary_l2934_293414

def A (m : ℝ) : Set ℝ := {0, m^2}
def B : Set ℝ := {1, 2}

theorem m_equals_one_sufficient_not_necessary :
  (∃ m : ℝ, A m ∩ B = {1} ∧ m ≠ 1) ∧
  (∀ m : ℝ, m = 1 → A m ∩ B = {1}) :=
sorry

end m_equals_one_sufficient_not_necessary_l2934_293414


namespace cubic_solution_sum_l2934_293419

theorem cubic_solution_sum (a b c : ℝ) : 
  (a^3 - 4*a^2 + 7*a = 15) ∧ 
  (b^3 - 4*b^2 + 7*b = 15) ∧ 
  (c^3 - 4*c^2 + 7*c = 15) →
  a*b/c + b*c/a + c*a/b = 49/15 := by
sorry

end cubic_solution_sum_l2934_293419


namespace complex_abs_power_six_l2934_293478

theorem complex_abs_power_six : Complex.abs ((2 : ℂ) + 2 * Complex.I * Real.sqrt 3) ^ 6 = 4096 := by
  sorry

end complex_abs_power_six_l2934_293478


namespace tangent_parallel_implies_a_equals_5_l2934_293411

-- Define the curve
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x

-- Define the derivative of the curve
def f' (a : ℝ) (x : ℝ) : ℝ := 2*x + a

theorem tangent_parallel_implies_a_equals_5 (a : ℝ) :
  f' a 1 = 7 → a = 5 := by
  sorry

#check tangent_parallel_implies_a_equals_5

end tangent_parallel_implies_a_equals_5_l2934_293411


namespace seminar_chairs_l2934_293480

/-- Converts a number from base 6 to base 10 -/
def base6ToDecimal (n : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 36 + tens * 6 + ones

/-- Calculates the number of chairs needed given the number of participants and participants per chair -/
def calculateChairs (participants : Nat) (participantsPerChair : Nat) : Nat :=
  (participants + participantsPerChair - 1) / participantsPerChair

theorem seminar_chairs :
  let participantsBase6 : Nat := 315
  let participantsPerChair : Nat := 3
  let participantsDecimal := base6ToDecimal participantsBase6
  calculateChairs participantsDecimal participantsPerChair = 40 := by
  sorry

end seminar_chairs_l2934_293480


namespace no_zero_roots_l2934_293423

theorem no_zero_roots : 
  (∀ x : ℝ, 5 * x^2 - 3 = 50 → x ≠ 0) ∧
  (∀ x : ℝ, (3*x - 1)^2 = (x - 2)^2 → x ≠ 0) ∧
  (∀ x : ℝ, x^2 - 9 ≥ 0 → 2*x - 2 ≥ 0 → x^2 - 9 = 2*x - 2 → x ≠ 0) := by
  sorry


end no_zero_roots_l2934_293423


namespace base_three_to_decimal_l2934_293421

/-- Converts a digit in base 3 to its decimal value -/
def toDecimal (d : Nat) : Nat :=
  if d < 3 then d else 0

/-- Calculates the value of a base-3 number given its digits -/
def baseThreeToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + toDecimal d * 3^i) 0

/-- The decimal representation of 10212 in base 3 -/
def baseThreeNumber : Nat :=
  baseThreeToDecimal [2, 1, 2, 0, 1]

theorem base_three_to_decimal :
  baseThreeNumber = 104 := by sorry

end base_three_to_decimal_l2934_293421


namespace encounter_twelve_trams_l2934_293438

/-- Represents the tram system with given parameters -/
structure TramSystem where
  departure_interval : ℕ  -- Interval between tram departures in minutes
  journey_duration : ℕ    -- Duration of a full journey in minutes

/-- Calculates the number of trams encountered during a journey -/
def count_encountered_trams (system : TramSystem) : ℕ :=
  2 * (system.journey_duration / system.departure_interval)

/-- Theorem stating that in the given tram system, a passenger will encounter 12 trams -/
theorem encounter_twelve_trams (system : TramSystem) 
  (h1 : system.departure_interval = 10)
  (h2 : system.journey_duration = 60) : 
  count_encountered_trams system = 12 := by
  sorry

#eval count_encountered_trams ⟨10, 60⟩

end encounter_twelve_trams_l2934_293438


namespace inscribed_squares_ratio_l2934_293477

/-- A right triangle with sides 5, 12, and 13 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  sides : a = 5 ∧ b = 12 ∧ c = 13

/-- A square inscribed in a right triangle with a vertex at the right angle -/
def squareAtRightAngle (t : RightTriangle) (x : ℝ) : Prop :=
  x > 0 ∧ x < t.a ∧ x < t.b ∧ x / t.a = x / t.b

/-- A square inscribed in a right triangle with a side along the hypotenuse -/
def squareAlongHypotenuse (t : RightTriangle) (y : ℝ) : Prop :=
  y > 0 ∧ y < t.c ∧ (t.a - y)^2 + (t.b - y)^2 = y^2

theorem inscribed_squares_ratio (t1 t2 : RightTriangle) (x y : ℝ)
  (h1 : squareAtRightAngle t1 x) (h2 : squareAlongHypotenuse t2 y) :
  x / y = 144 / 85 := by
  sorry

end inscribed_squares_ratio_l2934_293477


namespace modular_home_cost_l2934_293458

/-- Calculates the cost of a modular home given specific module costs and sizes. -/
theorem modular_home_cost 
  (kitchen_size : ℕ) (kitchen_cost : ℕ)
  (bathroom_size : ℕ) (bathroom_cost : ℕ)
  (other_cost_per_sqft : ℕ)
  (total_size : ℕ) (num_bathrooms : ℕ) : 
  kitchen_size = 400 →
  kitchen_cost = 20000 →
  bathroom_size = 150 →
  bathroom_cost = 12000 →
  other_cost_per_sqft = 100 →
  total_size = 2000 →
  num_bathrooms = 2 →
  (kitchen_cost + num_bathrooms * bathroom_cost + 
   (total_size - kitchen_size - num_bathrooms * bathroom_size) * other_cost_per_sqft) = 174000 :=
by sorry

end modular_home_cost_l2934_293458


namespace inequality_implies_lower_bound_l2934_293470

theorem inequality_implies_lower_bound (a : ℝ) : 
  (∀ x : ℝ, x > 0 → x / (x^2 + 3*x + 1) ≤ a) → a ≥ 1/5 :=
by
  sorry

end inequality_implies_lower_bound_l2934_293470


namespace min_value_of_3a_plus_2_l2934_293426

theorem min_value_of_3a_plus_2 (a : ℝ) (h : 8 * a^2 + 10 * a + 6 = 2) :
  ∃ (m : ℝ), m = -1 ∧ ∀ x, (8 * x^2 + 10 * x + 6 = 2) → (3 * x + 2 ≥ m) := by
  sorry

end min_value_of_3a_plus_2_l2934_293426


namespace arithmetic_sequence_sum_l2934_293410

def arithmetic_sequence (a : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a + i * d)

def last_term (seq : List ℕ) : ℕ :=
  match seq.getLast? with
  | some x => x
  | none => 0

theorem arithmetic_sequence_sum (a d : ℕ) :
  ∀ seq : List ℕ, seq = arithmetic_sequence a d seq.length →
  last_term seq = 50 →
  seq.sum = 442 := by
  sorry

end arithmetic_sequence_sum_l2934_293410


namespace sqrt_144000_simplification_l2934_293431

theorem sqrt_144000_simplification : Real.sqrt 144000 = 120 * Real.sqrt 10 := by
  sorry

end sqrt_144000_simplification_l2934_293431


namespace octal_to_decimal_l2934_293468

theorem octal_to_decimal (r : ℕ) : 175 = 120 + r → r = 5 := by
  sorry

end octal_to_decimal_l2934_293468


namespace inverse_proportion_k_value_l2934_293484

/-- Given an inverse proportion function y = (k+1)/x passing through the point (1, -2),
    prove that the value of k is -3. -/
theorem inverse_proportion_k_value (k : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, x ≠ 0 → f x = (k + 1) / x) ∧ f 1 = -2) → k = -3 :=
by sorry

end inverse_proportion_k_value_l2934_293484


namespace post_office_problem_l2934_293400

theorem post_office_problem (total_spent : ℚ) (letter_cost : ℚ) (package_cost : ℚ) 
  (h1 : total_spent = 449/100)
  (h2 : letter_cost = 37/100)
  (h3 : package_cost = 88/100)
  : ∃ (letters packages : ℕ), 
    letters = packages + 2 ∧ 
    letter_cost * letters + package_cost * packages = total_spent ∧
    letters = 5 := by
  sorry

end post_office_problem_l2934_293400


namespace wins_to_losses_ratio_l2934_293446

/-- Represents the statistics of a baseball team's season. -/
structure BaseballSeason where
  total_games : ℕ
  wins : ℕ
  losses : ℕ

/-- Defines the conditions for the baseball season. -/
def validSeason (s : BaseballSeason) : Prop :=
  s.total_games = 130 ∧
  s.wins = s.losses + 14 ∧
  s.wins = 101

/-- Theorem stating the ratio of wins to losses for the given conditions. -/
theorem wins_to_losses_ratio (s : BaseballSeason) (h : validSeason s) :
  s.wins = 101 ∧ s.losses = 87 := by
  sorry

#check wins_to_losses_ratio

end wins_to_losses_ratio_l2934_293446


namespace running_time_difference_l2934_293498

/-- The time difference for running 5 miles between new and old shoes -/
theorem running_time_difference 
  (old_shoe_time : ℕ) -- Time to run one mile in old shoes
  (new_shoe_time : ℕ) -- Time to run one mile in new shoes
  (distance : ℕ) -- Distance to run in miles
  (h1 : old_shoe_time = 10)
  (h2 : new_shoe_time = 13)
  (h3 : distance = 5) :
  new_shoe_time * distance - old_shoe_time * distance = 15 :=
by sorry

end running_time_difference_l2934_293498


namespace fraction_equivalence_l2934_293425

theorem fraction_equivalence (k : ℝ) (h : k ≠ -5) :
  (k + 3) / (k + 5) = 3 / 5 ↔ k = 0 := by
sorry

end fraction_equivalence_l2934_293425


namespace abs_equation_solution_difference_l2934_293448

theorem abs_equation_solution_difference : ∃ x₁ x₂ : ℝ, 
  (|x₁ - 3| = 15 ∧ |x₂ - 3| = 15) ∧ 
  x₁ ≠ x₂ ∧ 
  |x₁ - x₂| = 30 := by
sorry

end abs_equation_solution_difference_l2934_293448


namespace negation_of_universal_proposition_l2934_293459

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) ↔
  (∃ x : ℝ, x > 0 ∧ (x + 1) * Real.exp x ≤ 1) := by
  sorry

end negation_of_universal_proposition_l2934_293459


namespace smallest_absolute_value_of_z_l2934_293481

theorem smallest_absolute_value_of_z (z : ℂ) (h : Complex.abs (z - 8) + Complex.abs (z - Complex.I * 7) = 15) :
  ∃ (w : ℂ), Complex.abs w = 56 / 15 ∧ ∀ (v : ℂ), Complex.abs (v - 8) + Complex.abs (v - Complex.I * 7) = 15 → Complex.abs w ≤ Complex.abs v :=
by sorry

end smallest_absolute_value_of_z_l2934_293481


namespace parallelepiped_properties_l2934_293496

/-- Properties of a parallelepiped -/
structure Parallelepiped where
  projection : ℝ
  height : ℝ
  rhombus_area : ℝ
  rhombus_diagonal : ℝ

/-- Calculate the lateral surface area of the parallelepiped -/
def lateral_surface_area (p : Parallelepiped) : ℝ := sorry

/-- Calculate the volume of the parallelepiped -/
def volume (p : Parallelepiped) : ℝ := sorry

/-- Theorem stating the lateral surface area and volume of the given parallelepiped -/
theorem parallelepiped_properties (p : Parallelepiped) 
  (h1 : p.projection = 5)
  (h2 : p.height = 12)
  (h3 : p.rhombus_area = 24)
  (h4 : p.rhombus_diagonal = 8) :
  lateral_surface_area p = 260 ∧ volume p = 312 := by sorry

end parallelepiped_properties_l2934_293496


namespace condition_implies_increasing_l2934_293464

def IsIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem condition_implies_increasing (a : ℕ → ℝ) 
    (h : ∀ n : ℕ, a (n + 1) > |a n|) : 
  IsIncreasing a := by
  sorry

end condition_implies_increasing_l2934_293464


namespace vehicles_meeting_time_l2934_293447

-- Define the vehicles
structure Vehicle where
  id : Nat
  speed : ℝ

-- Define the meeting points
structure MeetingPoint where
  vehicle1 : Vehicle
  vehicle2 : Vehicle
  time : ℝ

-- Define the problem
theorem vehicles_meeting_time
  (v1 v2 v3 v4 : Vehicle)
  (m12 m13 m14 m24 m34 : MeetingPoint)
  (h1 : m12.vehicle1 = v1 ∧ m12.vehicle2 = v2 ∧ m12.time = 0)
  (h2 : m13.vehicle1 = v1 ∧ m13.vehicle2 = v3 ∧ m13.time = 220)
  (h3 : m14.vehicle1 = v1 ∧ m14.vehicle2 = v4 ∧ m14.time = 280)
  (h4 : m24.vehicle1 = v2 ∧ m24.vehicle2 = v4 ∧ m24.time = 240)
  (h5 : m34.vehicle1 = v3 ∧ m34.vehicle2 = v4 ∧ m34.time = 130)
  (h_constant_speed : ∀ v : Vehicle, v.speed > 0)
  : ∃ m23 : MeetingPoint, m23.vehicle1 = v2 ∧ m23.vehicle2 = v3 ∧ m23.time = 200 :=
sorry

end vehicles_meeting_time_l2934_293447


namespace geometric_sequence_product_l2934_293424

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 1 * a 99 = 16 →
  a 1 + a 99 = 10 →
  a 20 * a 50 * a 80 = 64 := by
  sorry

end geometric_sequence_product_l2934_293424


namespace shaded_square_area_l2934_293415

/- Define the structure of the lawn -/
structure Lawn :=
  (total_area : ℝ)
  (rectangle_area : ℝ)
  (is_square : Bool)
  (has_four_rectangles : Bool)
  (has_square_in_rectangle : Bool)

/- Define the properties of the lawn -/
def lawn_properties (l : Lawn) : Prop :=
  l.is_square ∧ 
  l.has_four_rectangles ∧ 
  l.rectangle_area = 40 ∧
  l.has_square_in_rectangle

/- Theorem statement -/
theorem shaded_square_area (l : Lawn) :
  lawn_properties l →
  ∃ (square_area : ℝ), square_area = 2500 / 441 :=
by
  sorry

end shaded_square_area_l2934_293415


namespace integer_points_on_line_l2934_293416

theorem integer_points_on_line (n : ℕ) (initial_sum final_sum shift : ℤ) 
  (h1 : initial_sum = 25)
  (h2 : final_sum = -35)
  (h3 : shift = 5)
  (h4 : final_sum = initial_sum - n * shift) : n = 12 := by
  sorry

end integer_points_on_line_l2934_293416
