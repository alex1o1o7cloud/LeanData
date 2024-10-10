import Mathlib

namespace five_at_ten_equals_ten_thirds_l2043_204378

-- Define the @ operation for positive integers
def at_operation (a b : ℕ+) : ℚ := (a * b : ℚ) / (a + b : ℚ)

-- State the theorem
theorem five_at_ten_equals_ten_thirds : 
  at_operation 5 10 = 10 / 3 := by sorry

end five_at_ten_equals_ten_thirds_l2043_204378


namespace no_real_solution_for_sqrt_equation_l2043_204307

theorem no_real_solution_for_sqrt_equation :
  ¬∃ (x : ℝ), Real.sqrt (4 - 5*x) = 9 - x := by
sorry

end no_real_solution_for_sqrt_equation_l2043_204307


namespace binomial_14_11_l2043_204305

theorem binomial_14_11 : Nat.choose 14 11 = 364 := by
  sorry

end binomial_14_11_l2043_204305


namespace intersection_of_A_and_B_l2043_204317

def A : Set Int := {-2, -1}
def B : Set Int := {-1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {-1} := by sorry

end intersection_of_A_and_B_l2043_204317


namespace hyperbola_equation_tangent_line_perpendicular_intersection_l2043_204365

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop :=
  abs (((x + 2)^2 + y^2).sqrt - ((x - 2)^2 + y^2).sqrt) = 2 * Real.sqrt 3

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

-- Define the modified hyperbola C' for part 3
def hyperbola_C' (x y : ℝ) : Prop :=
  abs (((x + 2)^2 + y^2).sqrt - ((x - 2)^2 + y^2).sqrt) = 2

-- Define the line for part 3
def line_part3 (k : ℝ) (x y : ℝ) : Prop := y = k * x + 2

-- Theorem statements
theorem hyperbola_equation :
  ∀ x y : ℝ, hyperbola_C x y ↔ x^2 / 3 - y^2 = 1 := by sorry

theorem tangent_line :
  ∀ k : ℝ, (∃! p : ℝ × ℝ, hyperbola_C p.1 p.2 ∧ line_l k p.1 p.2) ↔
    k = Real.sqrt 3 / 3 ∨ k = -Real.sqrt 3 / 3 ∨ k = 2 ∨ k = -2 := by sorry

theorem perpendicular_intersection :
  ∀ k : ℝ, (∃ A B : ℝ × ℝ,
    hyperbola_C' A.1 A.2 ∧ hyperbola_C' B.1 B.2 ∧
    line_part3 k A.1 A.2 ∧ line_part3 k B.1 B.2 ∧
    A.1 * B.1 + A.2 * B.2 = 0) ↔
  k = Real.sqrt 2 ∨ k = -Real.sqrt 2 := by sorry

end hyperbola_equation_tangent_line_perpendicular_intersection_l2043_204365


namespace constant_value_l2043_204343

theorem constant_value (x : ℝ) (some_constant a k n : ℝ) :
  (3 * x + some_constant) * (2 * x - 7) = a * x^2 + k * x + n →
  a - n + k = 3 →
  some_constant = 2 := by
  sorry

end constant_value_l2043_204343


namespace solution_set_of_inequality_l2043_204322

noncomputable def f (x : ℝ) : ℝ := Real.exp (-x) - Real.exp x - 5 * x

theorem solution_set_of_inequality :
  {x : ℝ | f (x^2) + f (-x-6) < 0} = {x : ℝ | x < -2 ∨ x > 3} := by sorry

end solution_set_of_inequality_l2043_204322


namespace expression_value_l2043_204384

theorem expression_value (x : ℝ) (h : x = 5) : 3 * x + 4 = 19 := by
  sorry

end expression_value_l2043_204384


namespace bbq_guests_count_l2043_204368

/-- Represents the BBQ scenario with given parameters -/
structure BBQ where
  cook_time_per_side : ℕ  -- cooking time for one side of a burger in minutes
  grill_capacity : ℕ      -- number of burgers that can be cooked simultaneously
  total_cook_time : ℕ     -- total time spent cooking all burgers in minutes

/-- Calculates the number of guests at the BBQ -/
def number_of_guests (bbq : BBQ) : ℕ :=
  let total_burgers := (bbq.total_cook_time / (2 * bbq.cook_time_per_side)) * bbq.grill_capacity
  (2 * total_burgers) / 3

/-- Theorem stating that the number of guests at the BBQ is 30 -/
theorem bbq_guests_count (bbq : BBQ) 
  (h1 : bbq.cook_time_per_side = 4)
  (h2 : bbq.grill_capacity = 5)
  (h3 : bbq.total_cook_time = 72) :
  number_of_guests bbq = 30 := by
  sorry

#eval number_of_guests ⟨4, 5, 72⟩

end bbq_guests_count_l2043_204368


namespace map_coloring_theorem_l2043_204361

/-- Represents a map with regions -/
structure Map where
  regions : Nat
  adjacency : List (Nat × Nat)

/-- The minimum number of colors needed to color a map -/
def minColors (m : Map) : Nat :=
  sorry

/-- Theorem: The minimum number of colors needed for a 26-region map is 4 -/
theorem map_coloring_theorem (m : Map) (h1 : m.regions = 26) 
  (h2 : ∀ (i j : Nat), (i, j) ∈ m.adjacency → i ≠ j) 
  (h3 : minColors m > 3) : 
  minColors m = 4 :=
sorry

end map_coloring_theorem_l2043_204361


namespace brownie_pieces_fit_l2043_204346

/-- Represents the dimensions of a rectangle -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents the pan and brownie piece dimensions -/
def pan : Dimensions := ⟨24, 15⟩
def piece : Dimensions := ⟨3, 2⟩

/-- The number of brownie pieces that fit in the pan -/
def num_pieces : ℕ := area pan / area piece

theorem brownie_pieces_fit :
  num_pieces = 60 ∧
  area pan = num_pieces * area piece :=
sorry

end brownie_pieces_fit_l2043_204346


namespace same_gender_probability_same_school_probability_l2043_204306

/-- Represents a school with a certain number of male and female teachers --/
structure School where
  male_teachers : ℕ
  female_teachers : ℕ

/-- The total number of teachers in a school --/
def School.total_teachers (s : School) : ℕ := s.male_teachers + s.female_teachers

/-- The schools in the problem --/
def school_A : School := { male_teachers := 2, female_teachers := 1 }
def school_B : School := { male_teachers := 1, female_teachers := 2 }

/-- The total number of teachers in both schools --/
def total_teachers : ℕ := school_A.total_teachers + school_B.total_teachers

/-- Theorem for the probability of selecting two teachers of the same gender --/
theorem same_gender_probability :
  (school_A.male_teachers * school_B.male_teachers + school_A.female_teachers * school_B.female_teachers) / 
  (school_A.total_teachers * school_B.total_teachers) = 4 / 9 := by sorry

/-- Theorem for the probability of selecting two teachers from the same school --/
theorem same_school_probability :
  (school_A.total_teachers * (school_A.total_teachers - 1) + school_B.total_teachers * (school_B.total_teachers - 1)) / 
  (total_teachers * (total_teachers - 1)) = 2 / 5 := by sorry

end same_gender_probability_same_school_probability_l2043_204306


namespace cosine_law_acute_triangle_condition_l2043_204309

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  -- Assume the triangle is valid
  valid : a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π

-- Theorem 1: c = a cos B + b cos A
theorem cosine_law (t : Triangle) : t.c = t.a * Real.cos t.B + t.b * Real.cos t.A := by
  sorry

-- Theorem 2: If a³ + b³ = c³, then a² + b² > c²
theorem acute_triangle_condition (t : Triangle) : 
  t.a^3 + t.b^3 = t.c^3 → t.a^2 + t.b^2 > t.c^2 := by
  sorry

end cosine_law_acute_triangle_condition_l2043_204309


namespace quadratic_transformation_sum_l2043_204395

/-- Given a quadratic function y = x^2 - 4x - 12, when transformed into the form y = (x - m)^2 + p,
    the sum of m and p equals -14. -/
theorem quadratic_transformation_sum (x : ℝ) :
  ∃ (m p : ℝ), (∀ x, x^2 - 4*x - 12 = (x - m)^2 + p) ∧ (m + p = -14) := by
  sorry

end quadratic_transformation_sum_l2043_204395


namespace right_triangle_square_equal_area_l2043_204339

theorem right_triangle_square_equal_area (s h : ℝ) (s_pos : s > 0) : 
  (1/2 * s * h = s^2) → h = 2*s := by
  sorry

end right_triangle_square_equal_area_l2043_204339


namespace employed_females_percentage_l2043_204336

theorem employed_females_percentage (population : ℝ) 
  (h1 : population > 0)
  (employed : ℝ) 
  (h2 : employed = 0.6 * population)
  (employed_males : ℝ) 
  (h3 : employed_males = 0.42 * population) :
  (employed - employed_males) / employed = 0.3 := by
sorry

end employed_females_percentage_l2043_204336


namespace intersection_not_empty_implies_a_value_l2043_204330

theorem intersection_not_empty_implies_a_value (a : ℤ) : 
  let M : Set ℤ := {a, 0}
  let N : Set ℤ := {x | 2 * x^2 - 5 * x < 0}
  (M ∩ N).Nonempty → a = 1 ∨ a = 2 := by
sorry

end intersection_not_empty_implies_a_value_l2043_204330


namespace a_plus_b_equals_10_l2043_204303

-- Define the logarithm function (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem a_plus_b_equals_10 (a b : ℝ) 
  (ha : a + log10 a = 10) 
  (hb : b + 10^b = 10) : 
  a + b = 10 := by sorry

end a_plus_b_equals_10_l2043_204303


namespace f_composition_value_l2043_204359

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x else Real.exp (x + 1) - 2

theorem f_composition_value : f (f (1 / Real.exp 1)) = -1 := by
  sorry

end f_composition_value_l2043_204359


namespace volumes_not_equal_sufficient_not_necessary_for_areas_not_always_equal_l2043_204323

/-- Represents a geometric body -/
structure GeometricBody where
  volume : ℝ
  crossSectionArea : ℝ → ℝ  -- Function mapping height to cross-sectional area

/-- The Gougu Principle -/
axiom gougu_principle {A B : GeometricBody} (h : ∀ (height : ℝ), A.crossSectionArea height = B.crossSectionArea height) :
  A.volume = B.volume

/-- Two geometric bodies have the same height -/
def same_height (A B : GeometricBody) : Prop := true

theorem volumes_not_equal_sufficient_not_necessary_for_areas_not_always_equal
  (A B : GeometricBody) (h : same_height A B) :
  (∃ (height : ℝ), A.crossSectionArea height ≠ B.crossSectionArea height) ↔ 
  (A.volume ≠ B.volume ∨ (A.volume = B.volume ∧ ∃ (height : ℝ), A.crossSectionArea height ≠ B.crossSectionArea height)) :=
sorry

end volumes_not_equal_sufficient_not_necessary_for_areas_not_always_equal_l2043_204323


namespace janet_ticket_count_l2043_204315

/-- The number of tickets needed for Janet's amusement park rides -/
def total_tickets (roller_coaster_tickets : ℕ) (giant_slide_tickets : ℕ) 
                  (roller_coaster_rides : ℕ) (giant_slide_rides : ℕ) : ℕ :=
  roller_coaster_tickets * roller_coaster_rides + giant_slide_tickets * giant_slide_rides

/-- Proof that Janet needs 47 tickets for her planned rides -/
theorem janet_ticket_count : 
  total_tickets 5 3 7 4 = 47 := by
  sorry

end janet_ticket_count_l2043_204315


namespace subset_X_l2043_204380

def X : Set ℤ := {x | -2 ≤ x ∧ x ≤ 2}

theorem subset_X : {0} ⊆ X := by
  sorry

end subset_X_l2043_204380


namespace water_pumped_in_30_min_l2043_204321

/-- 
Given a pump that operates at a rate of 540 gallons per hour, 
this theorem proves that the volume of water pumped in 30 minutes is 270 gallons.
-/
theorem water_pumped_in_30_min (pump_rate : ℝ) (time : ℝ) : 
  pump_rate = 540 → time = 0.5 → pump_rate * time = 270 := by
  sorry

#check water_pumped_in_30_min

end water_pumped_in_30_min_l2043_204321


namespace wall_building_time_l2043_204385

/-- Given that 60 workers can build a wall in 3 days, prove that 30 workers 
    will take 6 days to build the same wall, assuming consistent work rate and conditions. -/
theorem wall_building_time 
  (workers_initial : ℕ) 
  (days_initial : ℕ) 
  (workers_new : ℕ) 
  (h1 : workers_initial = 60) 
  (h2 : days_initial = 3) 
  (h3 : workers_new = 30) :
  (workers_initial * days_initial) / workers_new = 6 := by
sorry

end wall_building_time_l2043_204385


namespace intersection_point_sum_l2043_204386

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a line passing through two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Calculates the area of a quadrilateral given its four vertices -/
def quadrilateralArea (a b c d : Point) : ℚ :=
  sorry

/-- Calculates the area of a triangle given its three vertices -/
def triangleArea (a b c : Point) : ℚ :=
  sorry

/-- Checks if a point is on a line -/
def isPointOnLine (p : Point) (l : Line) : Prop :=
  sorry

/-- Represents the intersection point of a line with CD -/
structure IntersectionPoint where
  p : ℕ
  q : ℕ
  r : ℕ
  s : ℕ

/-- The main theorem -/
theorem intersection_point_sum (a b c d : Point) (l : Line) (i : IntersectionPoint) :
  a = Point.mk 0 0 →
  b = Point.mk 2 4 →
  c = Point.mk 6 6 →
  d = Point.mk 8 0 →
  l.p1 = a →
  isPointOnLine (Point.mk (i.p / i.q) (i.r / i.s)) l →
  isPointOnLine (Point.mk (i.p / i.q) (i.r / i.s)) (Line.mk c d) →
  triangleArea a (Point.mk (i.p / i.q) (i.r / i.s)) d = (1/3) * quadrilateralArea a b c d →
  i.p + i.q + i.r + i.s = 28 :=
by
  sorry

end intersection_point_sum_l2043_204386


namespace prime_power_sum_l2043_204316

theorem prime_power_sum (w x y z : ℕ) : 
  2^w * 3^x * 5^y * 7^z = 13230 → 3*w + 2*x + 6*y + 4*z = 23 := by
  sorry

end prime_power_sum_l2043_204316


namespace chocolate_distribution_chocolate_squares_per_student_l2043_204370

theorem chocolate_distribution (gerald_bars : Nat) (squares_per_bar : Nat) (teacher_multiplier : Nat) (num_students : Nat) : Nat :=
  let total_bars := gerald_bars + gerald_bars * teacher_multiplier
  let total_squares := total_bars * squares_per_bar
  total_squares / num_students

-- The main theorem
theorem chocolate_squares_per_student :
  chocolate_distribution 7 8 2 24 = 7 := by
  sorry

end chocolate_distribution_chocolate_squares_per_student_l2043_204370


namespace book_loss_percentage_l2043_204327

/-- Given that the cost price of 5 books equals the selling price of 20 books,
    prove that the loss percentage is 75%. -/
theorem book_loss_percentage : ∀ (C S : ℝ), 
  C > 0 → S > 0 →  -- Ensure positive prices
  5 * C = 20 * S →  -- Given condition
  (C - S) / C * 100 = 75 := by  -- Loss percentage formula
  sorry

end book_loss_percentage_l2043_204327


namespace digit_distribution_exists_l2043_204345

theorem digit_distribution_exists : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 2 = 0 ∧ 
  n % 5 = 0 ∧ 
  n % 10 = 0 ∧ 
  n / 2 + 2 * (n / 5) + n / 10 = n :=
sorry

end digit_distribution_exists_l2043_204345


namespace unique_solution_condition_l2043_204328

theorem unique_solution_condition (a b : ℝ) : 
  (∃! x : ℝ, 4 * x - 6 + a = (b + 1) * x + 2) ↔ b ≠ 3 := by
  sorry

end unique_solution_condition_l2043_204328


namespace classroom_pairing_probability_l2043_204337

/-- The probability of two specific students being paired in a classroom. -/
def pairProbability (n : ℕ) : ℚ :=
  1 / (n - 1)

/-- Theorem: In a classroom of 24 students where each student is randomly paired
    with another, the probability of a specific student being paired with
    another specific student is 1/23. -/
theorem classroom_pairing_probability :
  pairProbability 24 = 1 / 23 := by
  sorry

#eval pairProbability 24

end classroom_pairing_probability_l2043_204337


namespace locus_C_equation_point_N_coordinates_l2043_204301

-- Define the circle and its properties
def Circle (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∀ p : ℝ × ℝ, (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2

-- Define the locus C
def LocusC (p : ℝ × ℝ) : Prop :=
  ∃ (r : ℝ), Circle p r ∧ 
  (p.1 - 1)^2 + p.2^2 = r^2 ∧  -- Tangent to F(1,0)
  (p.1 + 1)^2 = r^2            -- Tangent to x = -1

-- Define point A
def PointA : ℝ × ℝ := (4, 4)

-- Define point B
def PointB : ℝ × ℝ := (0, 4)

-- Define point M
def PointM : ℝ × ℝ := (0, 2)

-- Define point F
def PointF : ℝ × ℝ := (1, 0)

-- Theorem for the equation of locus C
theorem locus_C_equation : 
  ∀ p : ℝ × ℝ, LocusC p ↔ p.2^2 = 4 * p.1 := by sorry

-- Theorem for the coordinates of point N
theorem point_N_coordinates :
  ∃ N : ℝ × ℝ, N.1 = 8/5 ∧ N.2 = 4/5 ∧
  (N.2 - PointM.2) / (N.1 - PointM.1) = -3/4 ∧  -- MN perpendicular to FA
  (PointA.2 - PointF.2) / (PointA.1 - PointF.1) = 4/3 := by sorry

end locus_C_equation_point_N_coordinates_l2043_204301


namespace factorization_equality_l2043_204379

theorem factorization_equality (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) := by
  sorry

end factorization_equality_l2043_204379


namespace problems_left_to_grade_l2043_204394

def problems_per_worksheet : ℕ := 4
def total_worksheets : ℕ := 9
def graded_worksheets : ℕ := 5

theorem problems_left_to_grade :
  (total_worksheets - graded_worksheets) * problems_per_worksheet = 16 := by
  sorry

end problems_left_to_grade_l2043_204394


namespace sum_of_roots_l2043_204360

theorem sum_of_roots (p q : ℝ) : 
  (∀ x, x^2 - p*x + q = 0 ↔ (x = p ∨ x = q)) →
  2*p + 3*q = 6 →
  p + q = p :=
by sorry

end sum_of_roots_l2043_204360


namespace intersection_M_complement_N_l2043_204376

-- Define the set M
def M : Set ℝ := {x | 0 < x ∧ x < 27}

-- Define the set N
def N : Set ℝ := {x | x < -1 ∨ x > 5}

-- Theorem statement
theorem intersection_M_complement_N :
  M ∩ (Set.univ \ N) = Set.Ioo 0 5 := by sorry

end intersection_M_complement_N_l2043_204376


namespace probability_not_perfect_power_l2043_204331

/-- A number is a perfect power if it can be expressed as x^y where x and y are integers and y > 1 -/
def IsPerfectPower (n : ℕ) : Prop :=
  ∃ x y : ℕ, y > 1 ∧ n = x^y

/-- The count of numbers from 1 to 200 that are not perfect powers -/
def CountNotPerfectPower : ℕ := 179

theorem probability_not_perfect_power :
  (CountNotPerfectPower : ℚ) / 200 = 179 / 200 :=
sorry

end probability_not_perfect_power_l2043_204331


namespace factor_expression_l2043_204344

theorem factor_expression (x : ℝ) : 63 * x^2 + 54 = 9 * (7 * x^2 + 6) := by
  sorry

end factor_expression_l2043_204344


namespace perfect_cube_from_sum_l2043_204397

theorem perfect_cube_from_sum (a b c : ℤ) (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (h_sum : ∃ (n : ℤ), a / b + b / c + c / a = n) : 
  ∃ (m : ℤ), a * b * c = m^3 := by
  sorry

end perfect_cube_from_sum_l2043_204397


namespace x_value_when_y_72_l2043_204372

/-- Given positive numbers x and y, where x^2 * y is constant, y = 8 when x = 3,
    and x^2 has increased by a factor of 4, prove that x = 1 when y = 72 -/
theorem x_value_when_y_72 (x y : ℝ) (z : ℝ) (h1 : x > 0) (h2 : y > 0)
  (h3 : ∃ k : ℝ, ∀ x y, x^2 * y = k)
  (h4 : 3^2 * 8 = 8 * 3^2)
  (h5 : z = 4)
  (h6 : y = 72)
  (h7 : x^2 = 3^2 * z) :
  x = 1 := by
sorry

end x_value_when_y_72_l2043_204372


namespace sum_xy_equals_two_l2043_204335

theorem sum_xy_equals_two (w x y z : ℝ) 
  (eq1 : w + x + y = 3)
  (eq2 : x + y + z = 4)
  (eq3 : w + x + y + z = 5) :
  x + y = 2 := by
sorry

end sum_xy_equals_two_l2043_204335


namespace five_balls_four_boxes_l2043_204333

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (balls : ℕ) (boxes : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 56 := by
  sorry

end five_balls_four_boxes_l2043_204333


namespace sqrt_seven_to_sixth_l2043_204393

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end sqrt_seven_to_sixth_l2043_204393


namespace oreo_multiple_l2043_204340

theorem oreo_multiple (total : Nat) (jordan : Nat) (m : Nat) : 
  total = 36 → jordan = 11 → jordan + (jordan * m + 3) = total → m = 2 := by
  sorry

end oreo_multiple_l2043_204340


namespace cylinder_height_l2043_204348

/-- Represents a right cylinder with given dimensions -/
structure RightCylinder where
  radius : ℝ
  height : ℝ
  lateralSurfaceArea : ℝ
  endArea : ℝ

/-- Theorem stating the height of a specific cylinder -/
theorem cylinder_height (c : RightCylinder) 
  (h_radius : c.radius = 2)
  (h_lsa : c.lateralSurfaceArea = 16 * Real.pi)
  (h_ea : c.endArea = 8 * Real.pi) :
  c.height = 4 := by
  sorry

end cylinder_height_l2043_204348


namespace vector_operation_l2043_204353

def vector_a : ℝ × ℝ := (2, -1)
def vector_b : ℝ × ℝ := (1, 3)

theorem vector_operation :
  (-2 : ℝ) • vector_a + (3 : ℝ) • vector_b = (-1, 11) := by sorry

end vector_operation_l2043_204353


namespace age_difference_l2043_204357

/-- Given the ages of Patrick, Michael, and Monica satisfying certain ratios and sum, 
    prove that the difference between Monica's and Patrick's ages is 33 years. -/
theorem age_difference (patrick michael monica : ℕ) : 
  patrick * 5 = michael * 3 →  -- Patrick and Michael's ages are in ratio 3:5
  michael * 4 = monica * 3 →   -- Michael and Monica's ages are in ratio 3:4
  patrick + michael + monica = 132 →  -- Sum of their ages is 132
  monica - patrick = 33 := by  -- Difference between Monica's and Patrick's ages is 33
sorry  -- Proof omitted

end age_difference_l2043_204357


namespace ladder_length_l2043_204391

theorem ladder_length : ∃ L : ℝ, 
  L > 0 ∧ 
  (4/5 * L)^2 + 4^2 = L^2 ∧ 
  L = 20/3 :=
by sorry

end ladder_length_l2043_204391


namespace no_real_roots_implies_not_first_quadrant_l2043_204318

theorem no_real_roots_implies_not_first_quadrant (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - m ≠ 0) →
  ∀ x y : ℝ, y = (m + 1) * x + (m - 1) → (x > 0 ∧ y > 0 → False) :=
by sorry

end no_real_roots_implies_not_first_quadrant_l2043_204318


namespace combination_sum_equals_84_l2043_204313

theorem combination_sum_equals_84 : Nat.choose 8 2 + Nat.choose 8 3 = 84 := by
  sorry

end combination_sum_equals_84_l2043_204313


namespace perpendicular_unit_vector_to_BC_l2043_204347

def AB : ℝ × ℝ := (-1, 3)
def BC : ℝ → ℝ × ℝ := λ k => (3, k)
def CD : ℝ → ℝ × ℝ := λ k => (k, 2)

def AC (k : ℝ) : ℝ × ℝ := (AB.1 + (BC k).1, AB.2 + (BC k).2)

def parallel (v w : ℝ × ℝ) : Prop := v.1 * w.2 = v.2 * w.1

def perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

def is_unit_vector (v : ℝ × ℝ) : Prop := v.1^2 + v.2^2 = 1

theorem perpendicular_unit_vector_to_BC (k : ℝ) :
  parallel (AC k) (CD k) →
  ∃ v : ℝ × ℝ, perpendicular v (BC k) ∧ is_unit_vector v ∧
    (v = (Real.sqrt 10 / 10, -3 * Real.sqrt 10 / 10) ∨
     v = (-Real.sqrt 10 / 10, 3 * Real.sqrt 10 / 10)) :=
by sorry

end perpendicular_unit_vector_to_BC_l2043_204347


namespace cosine_sum_specific_angles_l2043_204308

theorem cosine_sum_specific_angles : 
  Real.cos (π/3) * Real.cos (π/6) - Real.sin (π/3) * Real.sin (π/6) = 0 := by
  sorry

end cosine_sum_specific_angles_l2043_204308


namespace carolyn_sum_is_24_l2043_204383

def game_list : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def is_removable (n : Nat) (l : List Nat) : Bool :=
  ∃ m ∈ l, m ≠ n ∧ n % m = 0

def remove_divisors (n : Nat) (l : List Nat) : List Nat :=
  l.filter (fun m => m = n ∨ n % m ≠ 0)

def carolyn_moves (l : List Nat) : List Nat :=
  let after_first_move := l.filter (· ≠ 8)
  let after_paul_first := remove_divisors 8 after_first_move
  let second_move := after_paul_first.filter (· ≠ 10)
  let after_paul_second := remove_divisors 10 second_move
  let third_move := after_paul_second.filter (· ≠ 6)
  third_move

theorem carolyn_sum_is_24 :
  let carolyn_removed := [8, 10, 6]
  carolyn_removed.sum = 24 ∧
  (∀ n ∈ carolyn_moves game_list, ¬is_removable n (carolyn_moves game_list)) := by
  sorry

end carolyn_sum_is_24_l2043_204383


namespace squares_below_line_count_l2043_204314

/-- The number of squares below the line 5x + 45y = 225 in the first quadrant -/
def squares_below_line : ℕ :=
  let x_intercept : ℕ := 45
  let y_intercept : ℕ := 5
  let total_squares : ℕ := x_intercept * y_intercept
  let diagonal_squares : ℕ := x_intercept + y_intercept - 1
  let non_diagonal_squares : ℕ := total_squares - diagonal_squares
  non_diagonal_squares / 2

theorem squares_below_line_count : squares_below_line = 88 := by
  sorry

end squares_below_line_count_l2043_204314


namespace negative_real_inequality_l2043_204363

theorem negative_real_inequality (x y z : ℝ) (hx : x < 0) (hy : y < 0) (hz : z < 0) :
  x * y * z / ((1 + 5*x) * (4*x + 3*y) * (5*y + 6*z) * (z + 18)) ≤ 1 / 5120 := by
  sorry

end negative_real_inequality_l2043_204363


namespace negation_false_l2043_204326

/-- A multi-digit number ends in 0 -/
def EndsInZero (n : ℕ) : Prop := n % 10 = 0 ∧ n ≥ 10

/-- A number is a multiple of 5 -/
def MultipleOfFive (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k

theorem negation_false : 
  ¬(∀ n : ℕ, EndsInZero n → MultipleOfFive n) → 
  (∃ n : ℕ, EndsInZero n ∧ ¬MultipleOfFive n) :=
by sorry

end negation_false_l2043_204326


namespace line_intersects_ellipse_l2043_204341

/-- The set of possible slopes for a line with y-intercept (0, -3) intersecting the ellipse 4x^2 + 25y^2 = 100 -/
def possible_slopes : Set ℝ :=
  {m : ℝ | m ≤ -Real.sqrt (1/20) ∨ m ≥ Real.sqrt (1/20)}

/-- The equation of the line with slope m and y-intercept (0, -3) -/
def line_equation (m : ℝ) (x : ℝ) : ℝ := m * x - 3

/-- The equation of the ellipse 4x^2 + 25y^2 = 100 -/
def ellipse_equation (x y : ℝ) : Prop := 4 * x^2 + 25 * y^2 = 100

theorem line_intersects_ellipse (m : ℝ) : 
  m ∈ possible_slopes ↔ 
  ∃ x : ℝ, ellipse_equation x (line_equation m x) := by
  sorry

#check line_intersects_ellipse

end line_intersects_ellipse_l2043_204341


namespace cider_production_l2043_204389

theorem cider_production (golden_per_pint pink_per_pint : ℕ)
  (num_farmhands work_hours : ℕ) (total_pints : ℕ) :
  golden_per_pint = 20 →
  pink_per_pint = 40 →
  num_farmhands = 6 →
  work_hours = 5 →
  total_pints = 120 →
  (∃ (apples_per_hour : ℕ),
    apples_per_hour * num_farmhands * work_hours = 
      (golden_per_pint + pink_per_pint) * total_pints ∧
    3 * (golden_per_pint * total_pints) = 
      (golden_per_pint + pink_per_pint) * total_pints ∧
    apples_per_hour = 240) :=
by sorry

end cider_production_l2043_204389


namespace min_n_plus_d_l2043_204352

/-- An arithmetic sequence with positive integer terms -/
structure ArithmeticSequence where
  n : ℕ+  -- number of terms
  d : ℕ+  -- common difference
  first_term : ℕ+ := 1  -- first term
  last_term : ℕ+ := 51  -- last term

/-- The property that the sequence follows the arithmetic sequence formula -/
def is_valid (seq : ArithmeticSequence) : Prop :=
  seq.first_term + (seq.n - 1) * seq.d = seq.last_term

/-- The theorem stating the minimum value of n + d -/
theorem min_n_plus_d (seq : ArithmeticSequence) (h : is_valid seq) : 
  (∀ seq' : ArithmeticSequence, is_valid seq' → seq.n + seq.d ≤ seq'.n + seq'.d) → 
  seq.n + seq.d = 16 := by
  sorry

end min_n_plus_d_l2043_204352


namespace divisibility_of_factorial_l2043_204325

theorem divisibility_of_factorial (n : ℕ+) :
  (2011^2011 ∣ n!) → (2011^2012 ∣ n!) :=
by sorry

end divisibility_of_factorial_l2043_204325


namespace solution_set_theorem_range_of_k_theorem_l2043_204366

-- Define the function f
def f (x k : ℝ) : ℝ := |x + 1| + |2 - x| - k

-- Theorem 1: Solution set of f(x) < 0 when k = 4
theorem solution_set_theorem :
  {x : ℝ | f x 4 < 0} = Set.Ioo (-3/2) (5/2) := by sorry

-- Theorem 2: Range of k for f(x) ≥ √(k+3) for all x ∈ ℝ
theorem range_of_k_theorem :
  ∀ k : ℝ, (∀ x : ℝ, f x k ≥ Real.sqrt (k + 3)) ↔ k ∈ Set.Iic 1 := by sorry

end solution_set_theorem_range_of_k_theorem_l2043_204366


namespace range_of_m_for_equation_l2043_204375

/-- Given that the equation e^(mx) = x^2 has two distinct real roots in the interval (0, 16),
    prove that the range of values for the real number m is (ln(2)/2, 2/e). -/
theorem range_of_m_for_equation (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 16 ∧ 
   Real.exp (m * x₁) = x₁^2 ∧ Real.exp (m * x₂) = x₂^2) →
  (Real.log 2 / 2 < m ∧ m < 2 / Real.exp 1) :=
by sorry

end range_of_m_for_equation_l2043_204375


namespace equation_solution_l2043_204350

theorem equation_solution (x : ℝ) : 3 - 1 / (2 - x) = 1 / (2 - x) → x = 4 / 3 := by
  sorry

end equation_solution_l2043_204350


namespace right_triangle_x_coordinate_l2043_204312

/-- Given points P, Q, and R forming a right triangle with ∠PQR = 90°, prove that the x-coordinate of R is 13. -/
theorem right_triangle_x_coordinate :
  let P : ℝ × ℝ := (2, 0)
  let Q : ℝ × ℝ := (11, -3)
  let R : ℝ × ℝ := (x, 3)
  ∀ x : ℝ,
  (Q.1 - P.1) * (R.1 - Q.1) + (Q.2 - P.2) * (R.2 - Q.2) = 0 →
  x = 13 :=
by sorry

end right_triangle_x_coordinate_l2043_204312


namespace hyperbola_ellipse_condition_l2043_204392

/-- Represents the condition for a hyperbola -/
def is_hyperbola (m : ℝ) : Prop :=
  (m + 3) * (2*m + 1) < 0

/-- Represents the condition for an ellipse with foci on y-axis -/
def is_ellipse_y_foci (m : ℝ) : Prop :=
  -(2*m - 1) > m + 2 ∧ m + 2 > 0

/-- The necessary but not sufficient condition -/
def necessary_condition (m : ℝ) : Prop :=
  -2 < m ∧ m < -1/3

theorem hyperbola_ellipse_condition :
  (∀ m, is_hyperbola m ∧ is_ellipse_y_foci m → necessary_condition m) ∧
  ¬(∀ m, necessary_condition m → is_hyperbola m ∧ is_ellipse_y_foci m) :=
sorry

end hyperbola_ellipse_condition_l2043_204392


namespace angle_ABC_measure_l2043_204319

theorem angle_ABC_measure (angle_CBD angle_ABD angle_sum : ℝ) : 
  angle_CBD = 90 → angle_ABD = 60 → angle_sum = 200 → 
  ∃ (angle_ABC : ℝ), angle_ABC = 50 ∧ angle_ABC + angle_ABD + angle_CBD = angle_sum :=
by sorry

end angle_ABC_measure_l2043_204319


namespace equation_solution_l2043_204332

theorem equation_solution : ∃ x : ℝ, 4*x + 6*x = 360 - 10*(x - 4) ∧ x = 20 := by
  sorry

end equation_solution_l2043_204332


namespace africa_asia_difference_l2043_204387

/-- The number of bird families living near the mountain -/
def mountain_families : ℕ := 8

/-- The number of bird families that flew to Africa -/
def africa_families : ℕ := 42

/-- The number of bird families that flew to Asia -/
def asia_families : ℕ := 31

/-- Theorem: The difference between the number of bird families that flew to Africa
    and the number of bird families that flew to Asia is 11 -/
theorem africa_asia_difference : africa_families - asia_families = 11 := by
  sorry

end africa_asia_difference_l2043_204387


namespace plus_sign_square_has_90_degree_symmetry_l2043_204354

/-- Represents a square with markings -/
structure MarkedSquare where
  markings : Set (ℝ × ℝ)

/-- Defines 90-degree rotational symmetry for a marked square -/
def has_90_degree_rotational_symmetry (s : MarkedSquare) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ s.markings ↔ (-y, x) ∈ s.markings

/-- Represents a square with vertical and horizontal midlines crossed (plus sign) -/
def plus_sign_square : MarkedSquare :=
  { markings := {(x, y) | x = 0 ∨ y = 0} }

/-- Theorem: A square with both vertical and horizontal midlines crossed (plus sign) has 90-degree rotational symmetry -/
theorem plus_sign_square_has_90_degree_symmetry :
  has_90_degree_rotational_symmetry plus_sign_square :=
sorry

end plus_sign_square_has_90_degree_symmetry_l2043_204354


namespace area_of_triangle_ABC_l2043_204300

/-- The area of triangle ABC given the total area of small triangles and subtracted areas. -/
theorem area_of_triangle_ABC (total_area : ℝ) (subtracted_area : ℝ) 
  (h1 : total_area = 24)
  (h2 : subtracted_area = 14) :
  total_area - subtracted_area = 10 := by
  sorry

end area_of_triangle_ABC_l2043_204300


namespace arithmetic_sequence_n_eq_16_l2043_204382

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  a_4_eq_7 : a 4 = 7
  a_3_plus_a_6_eq_16 : a 3 + a 6 = 16
  exists_n : ∃ n : ℕ, a n = 31

/-- The theorem stating that n = 16 for the given arithmetic sequence -/
theorem arithmetic_sequence_n_eq_16 (seq : ArithmeticSequence) :
  ∃ n : ℕ, seq.a n = 31 ∧ n = 16 := by
  sorry


end arithmetic_sequence_n_eq_16_l2043_204382


namespace quanxing_max_difference_l2043_204338

/-- Represents the mass of a bottle of Quanxing mineral water in mL -/
structure QuanxingBottle where
  mass : ℝ
  h : abs (mass - 450) ≤ 1

/-- The maximum difference in mass between any two Quanxing bottles is 2 mL -/
theorem quanxing_max_difference (bottle1 bottle2 : QuanxingBottle) :
  abs (bottle1.mass - bottle2.mass) ≤ 2 := by
  sorry


end quanxing_max_difference_l2043_204338


namespace asha_granny_gift_l2043_204329

/-- The amount of money Asha was gifted by her granny --/
def granny_gift (brother_loan mother_loan father_loan savings spent_fraction remaining : ℚ) : ℚ :=
  (remaining / (1 - spent_fraction)) - (brother_loan + mother_loan + father_loan + savings)

/-- Theorem stating the amount gifted by Asha's granny --/
theorem asha_granny_gift :
  granny_gift 20 30 40 100 (3/4) 65 = 70 := by sorry

end asha_granny_gift_l2043_204329


namespace absolute_value_inequality_implies_a_geq_two_l2043_204399

theorem absolute_value_inequality_implies_a_geq_two :
  (∀ x : ℝ, |x + 3| - |x + 1| - 2*a + 2 < 0) → a ≥ 2 :=
by
  sorry

end absolute_value_inequality_implies_a_geq_two_l2043_204399


namespace inequality_solution_set_minimum_m_value_minimum_fraction_value_l2043_204320

def f (x : ℝ) := |x + 1| + |x - 1|

theorem inequality_solution_set :
  {x : ℝ | f x < 2*x + 3} = {x : ℝ | x > -1/2} := by sorry

theorem minimum_m_value :
  (∃ (x₀ : ℝ), f x₀ ≤ 2) ∧ 
  (∀ (m : ℝ), (∃ (x : ℝ), f x ≤ m) → m ≥ 2) := by sorry

theorem minimum_fraction_value :
  ∀ (a b : ℝ), a > 0 → b > 0 → 3*a + b = 2 →
  1/(2*a) + 1/(a+b) ≥ 2 := by sorry

end inequality_solution_set_minimum_m_value_minimum_fraction_value_l2043_204320


namespace possible_distances_l2043_204369

/-- Three points on a line -/
structure PointsOnLine where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The distance between two points on a line -/
def distance (x y : ℝ) : ℝ := |x - y|

theorem possible_distances (p : PointsOnLine) 
  (h1 : distance p.A p.B = 1)
  (h2 : distance p.B p.C = 3) :
  distance p.A p.C = 4 ∨ distance p.A p.C = 2 := by
  sorry

end possible_distances_l2043_204369


namespace compare_n_squared_and_two_to_n_l2043_204374

theorem compare_n_squared_and_two_to_n (n : ℕ+) :
  (n = 1 → n.val^2 < 2^n.val) ∧
  (n = 2 → n.val^2 = 2^n.val) ∧
  (n = 3 → n.val^2 > 2^n.val) ∧
  (n = 4 → n.val^2 = 2^n.val) ∧
  (n ≥ 5 → n.val^2 < 2^n.val) := by
  sorry

end compare_n_squared_and_two_to_n_l2043_204374


namespace meals_neither_kosher_nor_vegan_l2043_204388

theorem meals_neither_kosher_nor_vegan 
  (total_clients : ℕ) 
  (vegan_clients : ℕ) 
  (kosher_clients : ℕ) 
  (both_vegan_and_kosher : ℕ) 
  (h1 : total_clients = 30) 
  (h2 : vegan_clients = 7) 
  (h3 : kosher_clients = 8) 
  (h4 : both_vegan_and_kosher = 3) : 
  total_clients - (vegan_clients + kosher_clients - both_vegan_and_kosher) = 18 :=
by sorry

end meals_neither_kosher_nor_vegan_l2043_204388


namespace travel_theorem_l2043_204304

/-- Represents the speeds of Butch, Sundance, and Sparky in miles per hour -/
structure Speeds where
  butch : ℝ
  sundance : ℝ
  sparky : ℝ

/-- Represents the distance traveled and time taken -/
structure TravelData where
  distance : ℕ  -- in miles
  time : ℕ      -- in minutes

/-- The main theorem representing the problem -/
theorem travel_theorem (speeds : Speeds) (h1 : speeds.butch = 4)
    (h2 : speeds.sundance = 2.5) (h3 : speeds.sparky = 6) : 
    ∃ (data : TravelData), data.distance = 19 ∧ data.time = 330 ∧ 
    data.distance + data.time = 349 := by
  sorry

#check travel_theorem

end travel_theorem_l2043_204304


namespace grade_distribution_equals_total_total_students_is_100_l2043_204351

-- Define the total number of students
def total_students : ℕ := 100

-- Define the number of students who received each grade
def a_students : ℕ := total_students / 5
def b_students : ℕ := total_students / 4
def c_students : ℕ := total_students / 2
def d_students : ℕ := 5

-- Theorem stating that the sum of students in each grade category equals the total number of students
theorem grade_distribution_equals_total :
  a_students + b_students + c_students + d_students = total_students := by
  sorry

-- Theorem proving that the total number of students is 100
theorem total_students_is_100 : total_students = 100 := by
  sorry

end grade_distribution_equals_total_total_students_is_100_l2043_204351


namespace earnings_difference_l2043_204371

def bert_phones : ℕ := 8
def bert_price : ℕ := 18
def tory_guns : ℕ := 7
def tory_price : ℕ := 20

theorem earnings_difference : bert_phones * bert_price - tory_guns * tory_price = 4 := by
  sorry

end earnings_difference_l2043_204371


namespace sum_of_digits_mod_9_C_mod_9_eq_5_l2043_204398

/-- The sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A is the sum of digits of 4568^7777 -/
def A : ℕ := sumOfDigits (4568^7777)

/-- B is the sum of digits of A -/
def B : ℕ := sumOfDigits A

/-- C is the sum of digits of B -/
def C : ℕ := sumOfDigits B

/-- Theorem stating that the sum of digits of a number is congruent to the number modulo 9 -/
theorem sum_of_digits_mod_9 (n : ℕ) : sumOfDigits n ≡ n [MOD 9] := sorry

/-- Main theorem to prove -/
theorem C_mod_9_eq_5 : C ≡ 5 [MOD 9] := by sorry

end sum_of_digits_mod_9_C_mod_9_eq_5_l2043_204398


namespace balloon_problem_solution_l2043_204302

def balloon_problem (initial_balloons : ℕ) (given_to_girl : ℕ) (floated_away : ℕ) (given_away_later : ℕ) (final_balloons : ℕ) : ℕ :=
  final_balloons - (initial_balloons - given_to_girl - floated_away - given_away_later)

theorem balloon_problem_solution :
  balloon_problem 50 1 12 9 39 = 11 := by
  sorry

end balloon_problem_solution_l2043_204302


namespace sqrt_two_irrational_l2043_204334

theorem sqrt_two_irrational : Irrational (Real.sqrt 2) := by
  sorry

end sqrt_two_irrational_l2043_204334


namespace problem_solution_l2043_204356

theorem problem_solution (x : ℝ) : ((x / 4) * 5 + 10 - 12 = 48) → x = 40 := by
  sorry

end problem_solution_l2043_204356


namespace smallest_n_with_6474_l2043_204355

def concatenate (a b c : ℕ) : List ℕ :=
  (a.digits 10) ++ (b.digits 10) ++ (c.digits 10)

def contains_subseq (l : List ℕ) (s : List ℕ) : Prop :=
  ∃ i, l.drop i = s ++ l.drop (i + s.length)

theorem smallest_n_with_6474 :
  ∀ n : ℕ, n < 46 →
    ¬ (contains_subseq (concatenate n (n + 1) (n + 2)) [6, 4, 7, 4]) ∧
  contains_subseq (concatenate 46 47 48) [6, 4, 7, 4] :=
sorry

end smallest_n_with_6474_l2043_204355


namespace l_shape_area_is_55_l2043_204358

/-- The area of an "L" shape formed by cutting a smaller rectangle from a larger rectangle -/
def l_shape_area (large_width large_height small_width small_height : ℕ) : ℕ :=
  large_width * large_height - small_width * small_height

/-- Theorem: The area of the "L" shape is 55 square units -/
theorem l_shape_area_is_55 :
  l_shape_area 10 7 5 3 = 55 := by
  sorry

#eval l_shape_area 10 7 5 3

end l_shape_area_is_55_l2043_204358


namespace expression_result_l2043_204364

theorem expression_result : 
  (7899665 : ℝ) - 12 * 3 * 2 + (7^3) / Real.sqrt 144 = 7899621.5833 := by
sorry

end expression_result_l2043_204364


namespace solution_set_linear_inequalities_l2043_204377

theorem solution_set_linear_inequalities :
  let S := {x : ℝ | x - 2 > 1 ∧ x < 4}
  S = {x : ℝ | 3 < x ∧ x < 4} := by
  sorry

end solution_set_linear_inequalities_l2043_204377


namespace partial_fraction_decomposition_l2043_204367

theorem partial_fraction_decomposition (C D : ℚ) :
  (∀ x : ℚ, x ≠ 11 ∧ x ≠ -5 →
    (7 * x - 4) / (x^2 - 6 * x - 55) = C / (x - 11) + D / (x + 5)) →
  C = 73 / 16 ∧ D = 39 / 16 := by
sorry

end partial_fraction_decomposition_l2043_204367


namespace projection_orthogonal_vectors_l2043_204381

/-- Given two orthogonal vectors a and b in ℝ², prove that if the projection of (4, -2) onto a
    is (1/2, 1), then the projection of (4, -2) onto b is (7/2, -3). -/
theorem projection_orthogonal_vectors (a b : ℝ × ℝ) : 
  a.1 * b.1 + a.2 * b.2 = 0 →  -- a and b are orthogonal
  (∃ k : ℝ, k • a = (1/2, 1) ∧ k * (a.1 * 4 + a.2 * (-2)) = a.1^2 + a.2^2) →  -- proj_a (4, -2) = (1/2, 1)
  (∃ m : ℝ, m • b = (7/2, -3) ∧ m * (b.1 * 4 + b.2 * (-2)) = b.1^2 + b.2^2)  -- proj_b (4, -2) = (7/2, -3)
  := by sorry

end projection_orthogonal_vectors_l2043_204381


namespace smallest_four_digit_divisible_by_4_and_5_l2043_204349

theorem smallest_four_digit_divisible_by_4_and_5 :
  ∀ n : ℕ, 1000 ≤ n → n < 10000 → n % 4 = 0 → n % 5 = 0 → 1000 ≤ n :=
by sorry

end smallest_four_digit_divisible_by_4_and_5_l2043_204349


namespace arithmetic_sum_equals_expression_l2043_204362

/-- The sum of an arithmetic sequence with given parameters -/
def arithmetic_sum (k : ℕ) : ℚ :=
  let n : ℕ := 2 * k - 1
  let a₁ : ℚ := k^2 - 1
  let d : ℚ := 1
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Theorem stating the sum of the arithmetic sequence equals the given expression -/
theorem arithmetic_sum_equals_expression (k : ℕ) :
  arithmetic_sum k = 2 * k^3 + k^2 - 4 * k + 3/2 := by
  sorry

end arithmetic_sum_equals_expression_l2043_204362


namespace smallest_zero_201_l2043_204342

/-- A sequence defined by the given recurrence relation -/
def a : ℕ → ℚ
  | 0 => 134
  | 1 => 150
  | (k + 2) => a k - (k + 1) / a (k + 1)

/-- The property that a_n = 0 -/
def sequence_zero (n : ℕ) : Prop := a n = 0

/-- Theorem stating that 201 is the smallest positive integer n for which a_n = 0 -/
theorem smallest_zero_201 : 
  (∀ m : ℕ, m < 201 → ¬ sequence_zero m) ∧ sequence_zero 201 := by sorry

end smallest_zero_201_l2043_204342


namespace road_distance_ratio_l2043_204311

/-- Represents the distance between two cities --/
structure CityDistance where
  total : ℕ
  deriving Repr

/-- Represents a pole with distances to two cities --/
structure Pole where
  distanceA : ℕ
  distanceB : ℕ
  deriving Repr

/-- The configuration of poles between two cities --/
structure RoadConfiguration where
  distance : CityDistance
  pole1 : Pole
  pole2 : Pole
  pole3 : Pole
  deriving Repr

/-- The theorem to be proved --/
theorem road_distance_ratio 
  (config : RoadConfiguration) 
  (h1 : config.pole1.distanceB = 3 * config.pole1.distanceA)
  (h2 : config.pole2.distanceB = 3 * config.pole2.distanceA)
  (h3 : config.pole1.distanceA + config.pole1.distanceB = config.distance.total)
  (h4 : config.pole2.distanceA + config.pole2.distanceB = config.distance.total)
  (h5 : config.pole2.distanceA = config.pole1.distanceA + 40)
  (h6 : config.pole3.distanceA = config.pole2.distanceA + 10)
  (h7 : config.pole3.distanceB = config.pole2.distanceB - 10) :
  (max config.pole3.distanceA config.pole3.distanceB) / 
  (min config.pole3.distanceA config.pole3.distanceB) = 7 := by
  sorry

end road_distance_ratio_l2043_204311


namespace junior_teachers_sampled_count_l2043_204310

/-- Represents the number of teachers in each category -/
structure TeacherCounts where
  total : Nat
  senior : Nat
  intermediate : Nat
  junior : Nat

/-- Represents the sample size for stratified sampling -/
def SampleSize : Nat := 50

/-- Calculates the number of junior teachers in a stratified sample -/
def juniorTeachersSampled (counts : TeacherCounts) (sampleSize : Nat) : Nat :=
  (sampleSize * counts.junior) / counts.total

/-- Theorem: The number of junior teachers sampled is 20 -/
theorem junior_teachers_sampled_count 
  (counts : TeacherCounts) 
  (h1 : counts.total = 200)
  (h2 : counts.senior = 20)
  (h3 : counts.intermediate = 100)
  (h4 : counts.junior = 80) :
  juniorTeachersSampled counts SampleSize = 20 := by
  sorry

#eval juniorTeachersSampled { total := 200, senior := 20, intermediate := 100, junior := 80 } SampleSize

end junior_teachers_sampled_count_l2043_204310


namespace total_pencils_l2043_204396

/-- Given that each child has 2 pencils and there are 8 children, 
    prove that the total number of pencils is 16. -/
theorem total_pencils (pencils_per_child : ℕ) (num_children : ℕ) 
  (h1 : pencils_per_child = 2) 
  (h2 : num_children = 8) : 
  pencils_per_child * num_children = 16 := by
  sorry

end total_pencils_l2043_204396


namespace expression_value_l2043_204324

theorem expression_value : 
  let x : ℕ := 3
  x + x * (x ^ (x ^ 2)) = 59052 := by sorry

end expression_value_l2043_204324


namespace quadratic_roots_sum_of_squares_l2043_204373

theorem quadratic_roots_sum_of_squares (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 - m*x₁ + 2*m - 1 = 0 ∧ 
    x₂^2 - m*x₂ + 2*m - 1 = 0 ∧
    x₁^2 + x₂^2 = 7) → 
  m = 5 :=
by sorry

end quadratic_roots_sum_of_squares_l2043_204373


namespace salary_comparison_l2043_204390

theorem salary_comparison (a b : ℝ) (h : a = 0.8 * b) : b = 1.25 * a := by
  sorry

end salary_comparison_l2043_204390
