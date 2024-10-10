import Mathlib

namespace number_difference_theorem_l1405_140591

theorem number_difference_theorem (x : ℝ) : x - (3 / 5) * x = 64 → x = 160 := by
  sorry

end number_difference_theorem_l1405_140591


namespace smallest_three_types_sixty_nine_includes_three_types_l1405_140531

/-- Represents a type of tree in the grove -/
inductive TreeType
  | Birch
  | Spruce
  | Pine
  | Aspen

/-- Represents the grove of trees -/
structure Grove :=
  (trees : Finset ℕ)
  (type : ℕ → TreeType)
  (total_count : trees.card = 100)
  (four_types_in_85 : ∀ s : Finset ℕ, s ⊆ trees → s.card = 85 → 
    (∃ i ∈ s, type i = TreeType.Birch) ∧
    (∃ i ∈ s, type i = TreeType.Spruce) ∧
    (∃ i ∈ s, type i = TreeType.Pine) ∧
    (∃ i ∈ s, type i = TreeType.Aspen))

/-- The main theorem stating the smallest number of trees that must include at least three types -/
theorem smallest_three_types (g : Grove) : 
  ∀ n < 69, ∃ s : Finset ℕ, s ⊆ g.trees ∧ s.card = n ∧ 
    (∃ t1 t2 : TreeType, ∀ i ∈ s, g.type i = t1 ∨ g.type i = t2) :=
by sorry

/-- The theorem stating that 69 trees always include at least three types -/
theorem sixty_nine_includes_three_types (g : Grove) :
  ∀ s : Finset ℕ, s ⊆ g.trees → s.card = 69 → 
    ∃ t1 t2 t3 : TreeType, t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧
    (∃ i ∈ s, g.type i = t1) ∧ (∃ i ∈ s, g.type i = t2) ∧ (∃ i ∈ s, g.type i = t3) :=
by sorry

end smallest_three_types_sixty_nine_includes_three_types_l1405_140531


namespace chocolate_price_after_discount_l1405_140507

/-- The final price of a chocolate after discount -/
def final_price (original_cost discount : ℚ) : ℚ :=
  original_cost - discount

/-- Theorem: The final price of a chocolate with original cost $2 and discount $0.57 is $1.43 -/
theorem chocolate_price_after_discount :
  final_price 2 0.57 = 1.43 := by
  sorry

end chocolate_price_after_discount_l1405_140507


namespace exist_positive_integers_with_nonzero_integer_roots_l1405_140524

theorem exist_positive_integers_with_nonzero_integer_roots :
  ∃ (a b c : ℕ+), 
    (∀ (x : ℤ), (x ≠ 0 ∧ (a:ℤ) * x^2 + (b:ℤ) * x + (c:ℤ) = 0) → 
      ∃ (y : ℤ), y ≠ 0 ∧ y ≠ x ∧ (a:ℤ) * y^2 + (b:ℤ) * y + (c:ℤ) = 0) ∧
    (∀ (x : ℤ), (x ≠ 0 ∧ (a:ℤ) * x^2 + (b:ℤ) * x - (c:ℤ) = 0) → 
      ∃ (y : ℤ), y ≠ 0 ∧ y ≠ x ∧ (a:ℤ) * y^2 + (b:ℤ) * y - (c:ℤ) = 0) ∧
    (∀ (x : ℤ), (x ≠ 0 ∧ (a:ℤ) * x^2 - (b:ℤ) * x + (c:ℤ) = 0) → 
      ∃ (y : ℤ), y ≠ 0 ∧ y ≠ x ∧ (a:ℤ) * y^2 - (b:ℤ) * y + (c:ℤ) = 0) ∧
    (∀ (x : ℤ), (x ≠ 0 ∧ (a:ℤ) * x^2 - (b:ℤ) * x - (c:ℤ) = 0) → 
      ∃ (y : ℤ), y ≠ 0 ∧ y ≠ x ∧ (a:ℤ) * y^2 - (b:ℤ) * y - (c:ℤ) = 0) :=
by sorry

end exist_positive_integers_with_nonzero_integer_roots_l1405_140524


namespace triangle_side_length_l1405_140527

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  (1/2 * a * c * Real.sin B = Real.sqrt 3) →  -- Area of triangle is √3
  (B = Real.pi / 3) →  -- B = 60°
  (a^2 + c^2 = 3*a*c) →  -- Given condition
  (b = 2 * Real.sqrt 2) :=  -- Conclusion to prove
by sorry

end triangle_side_length_l1405_140527


namespace circle_line_intersection_sum_l1405_140597

/-- Given a circle with radius 4 centered at the origin and a line y = √3x - 4
    intersecting the circle at points A and B, the sum of the length of segment AB
    and the length of the larger arc AB is (16π/3) + 4√3. -/
theorem circle_line_intersection_sum (A B : ℝ × ℝ) : 
  let r : ℝ := 4
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = r^2}
  let line := {(x, y) : ℝ × ℝ | y = Real.sqrt 3 * x - 4}
  A ∈ circle ∧ A ∈ line ∧ B ∈ circle ∧ B ∈ line ∧ A ≠ B →
  let segment_length := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let angle := Real.arccos ((2 * r^2 - segment_length^2) / (2 * r^2))
  let arc_length := (2 * π - angle) * r
  segment_length + arc_length = (16 * π / 3) + 4 * Real.sqrt 3 := by
  sorry

end circle_line_intersection_sum_l1405_140597


namespace geometric_transformations_l1405_140530

-- Define the basic geometric entities
structure Point

structure Line

structure Surface

structure Body

-- Define the movement operation
def moves (a : Type) (b : Type) : Prop :=
  ∃ (x : a), ∃ (y : b), true

-- Theorem statement
theorem geometric_transformations :
  (moves Point Line) ∧
  (moves Line Surface) ∧
  (moves Surface Body) := by
  sorry

end geometric_transformations_l1405_140530


namespace equation_implies_difference_l1405_140577

theorem equation_implies_difference (m n : ℝ) :
  (∀ x : ℝ, (x - m) * (x + 2) = x^2 + n*x - 8) →
  m - n = 6 := by
  sorry

end equation_implies_difference_l1405_140577


namespace total_potatoes_l1405_140541

theorem total_potatoes (nancy_potatoes sandy_potatoes : ℕ) 
  (h1 : nancy_potatoes = 6) 
  (h2 : sandy_potatoes = 7) : 
  nancy_potatoes + sandy_potatoes = 13 := by
  sorry

end total_potatoes_l1405_140541


namespace line_plane_relations_l1405_140576

/-- Represents a plane in 3D space -/
structure Plane where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ

/-- Represents a line in 3D space -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  z₁ : ℝ
  m : ℝ
  n : ℝ
  p : ℝ

/-- Determines if a line is parallel to a plane -/
def isParallel (plane : Plane) (line : Line) : Prop :=
  plane.A * line.m + plane.B * line.n + plane.C * line.p = 0

/-- Determines if a line is perpendicular to a plane -/
def isPerpendicular (plane : Plane) (line : Line) : Prop :=
  plane.A / line.m = plane.B / line.n ∧ plane.B / line.n = plane.C / line.p

theorem line_plane_relations (plane : Plane) (line : Line) :
  (isParallel plane line ↔ plane.A * line.m + plane.B * line.n + plane.C * line.p = 0) ∧
  (isPerpendicular plane line ↔ plane.A / line.m = plane.B / line.n ∧ plane.B / line.n = plane.C / line.p) :=
sorry

end line_plane_relations_l1405_140576


namespace men_in_business_class_l1405_140589

def total_passengers : ℕ := 160
def men_percentage : ℚ := 3/4
def business_class_percentage : ℚ := 1/4

theorem men_in_business_class : 
  ⌊(total_passengers : ℚ) * men_percentage * business_class_percentage⌋ = 30 := by
  sorry

end men_in_business_class_l1405_140589


namespace deepak_age_l1405_140567

/-- Proves that Deepak's present age is 21 years given the conditions -/
theorem deepak_age (rahul_future_age : ℕ) (years_difference : ℕ) (ratio_rahul : ℕ) (ratio_deepak : ℕ) :
  rahul_future_age = 34 →
  years_difference = 6 →
  ratio_rahul = 4 →
  ratio_deepak = 3 →
  (rahul_future_age - years_difference) * ratio_deepak = 21 * ratio_rahul :=
by
  sorry

#check deepak_age

end deepak_age_l1405_140567


namespace area_of_triangle_NOI_l1405_140553

/-- Triangle PQR with given side lengths -/
structure TrianglePQR where
  PQ : ℝ
  PR : ℝ
  QR : ℝ
  side_lengths : PQ = 15 ∧ PR = 8 ∧ QR = 17

/-- Point O is the circumcenter of triangle PQR -/
def is_circumcenter (O : ℝ × ℝ) (t : TrianglePQR) : Prop :=
  sorry

/-- Point I is the incenter of triangle PQR -/
def is_incenter (I : ℝ × ℝ) (t : TrianglePQR) : Prop :=
  sorry

/-- Point N is the center of a circle tangent to sides PQ, PR, and the circumcircle -/
def is_tangent_circle_center (N : ℝ × ℝ) (t : TrianglePQR) (O : ℝ × ℝ) : Prop :=
  sorry

/-- Calculate the area of a triangle given three points -/
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  sorry

/-- Main theorem: The area of triangle NOI is 5 -/
theorem area_of_triangle_NOI (t : TrianglePQR) (O I N : ℝ × ℝ) 
  (hO : is_circumcenter O t) 
  (hI : is_incenter I t)
  (hN : is_tangent_circle_center N t O) : 
  triangle_area N O I = 5 :=
sorry

end area_of_triangle_NOI_l1405_140553


namespace mel_age_when_katherine_is_two_dozen_l1405_140506

/-- The age difference between Katherine and Mel -/
def age_difference : ℕ := 3

/-- Katherine's age when she is two dozen years old -/
def katherine_age : ℕ := 24

/-- Mel's age when Katherine is two dozen years old -/
def mel_age : ℕ := katherine_age - age_difference

theorem mel_age_when_katherine_is_two_dozen : mel_age = 21 := by
  sorry

end mel_age_when_katherine_is_two_dozen_l1405_140506


namespace total_money_collected_is_960_l1405_140551

/-- Calculates the total money collected from admission receipts for a play. -/
def totalMoneyCollected (totalPeople : Nat) (adultPrice : Nat) (childPrice : Nat) (numAdults : Nat) : Nat :=
  let numChildren := totalPeople - numAdults
  adultPrice * numAdults + childPrice * numChildren

/-- Theorem stating that the total money collected is 960 dollars given the specified conditions. -/
theorem total_money_collected_is_960 :
  totalMoneyCollected 610 2 1 350 = 960 := by
  sorry

end total_money_collected_is_960_l1405_140551


namespace rectangular_prism_diagonal_l1405_140557

/-- A rectangular prism with given surface area and total edge length has a specific interior diagonal length -/
theorem rectangular_prism_diagonal (a b c : ℝ) 
  (h_surface_area : 2 * (a * b + a * c + b * c) = 54)
  (h_edge_length : 4 * (a + b + c) = 40) :
  ∃ d : ℝ, d^2 = a^2 + b^2 + c^2 ∧ d = Real.sqrt 46 := by
  sorry

end rectangular_prism_diagonal_l1405_140557


namespace optimal_chair_removal_l1405_140588

theorem optimal_chair_removal :
  let chairs_per_row : ℕ := 15
  let initial_chairs : ℕ := 150
  let expected_attendees : ℕ := 125
  let removed_chairs : ℕ := 45

  -- All rows are complete
  (initial_chairs - removed_chairs) % chairs_per_row = 0 ∧
  -- At least one row is empty
  (initial_chairs - removed_chairs) / chairs_per_row < initial_chairs / chairs_per_row ∧
  -- Remaining chairs are sufficient for attendees
  initial_chairs - removed_chairs ≥ expected_attendees ∧
  -- Minimizes empty seats
  ∀ (x : ℕ), x < removed_chairs →
    (initial_chairs - x) % chairs_per_row ≠ 0 ∨
    (initial_chairs - x) / chairs_per_row ≥ initial_chairs / chairs_per_row ∨
    initial_chairs - x < expected_attendees :=
by
  sorry

end optimal_chair_removal_l1405_140588


namespace fraction_equality_l1405_140510

theorem fraction_equality (w x y : ℝ) 
  (h1 : w / x = 1 / 6)
  (h2 : (x + y) / y = 2.2) :
  w / y = 0.2 := by
  sorry

end fraction_equality_l1405_140510


namespace geometric_sequence_problem_l1405_140579

/-- Represents a geometric sequence. -/
structure GeometricSequence where
  a : ℕ → ℝ
  r : ℝ
  h1 : ∀ n : ℕ, a (n + 1) = a n * r

/-- Given a geometric sequence {a_n} with a_1 = 2 and a_3a_5 = 4a_6^2, prove that a_3 = 1. -/
theorem geometric_sequence_problem (seq : GeometricSequence)
  (h2 : seq.a 1 = 2)
  (h3 : seq.a 3 * seq.a 5 = 4 * (seq.a 6)^2) :
  seq.a 3 = 1 := by
  sorry

end geometric_sequence_problem_l1405_140579


namespace sphere_radius_from_hole_l1405_140502

/-- Given a spherical hole in ice with a diameter of 30 cm at the surface and a depth of 10 cm,
    the radius of the sphere that created this hole is 16.25 cm. -/
theorem sphere_radius_from_hole (diameter : ℝ) (depth : ℝ) (radius : ℝ) :
  diameter = 30 ∧ depth = 10 ∧ radius = (diameter / 2)^2 / (4 * depth) + depth / 4 →
  radius = 16.25 :=
by sorry

end sphere_radius_from_hole_l1405_140502


namespace max_value_in_region_D_l1405_140585

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define the asymptotes
def asymptote1 (x y : ℝ) : Prop := y = x
def asymptote2 (x y : ℝ) : Prop := y = -x

-- Define the bounding line
def boundingLine (x : ℝ) : Prop := x = 3

-- Define the region D
def regionD (x y : ℝ) : Prop :=
  x ≤ 3 ∧ y ≤ x ∧ y ≥ -x

-- Define the objective function
def objectiveFunction (x y : ℝ) : ℝ := x + 4*y

-- Theorem statement
theorem max_value_in_region_D :
  ∃ (x y : ℝ), regionD x y ∧
  ∀ (x' y' : ℝ), regionD x' y' →
  objectiveFunction x y ≥ objectiveFunction x' y' ∧
  objectiveFunction x y = 15 :=
sorry

end max_value_in_region_D_l1405_140585


namespace fraction_of_men_left_l1405_140549

/-- Represents the movie screening scenario -/
structure MovieScreening where
  total_guests : ℕ
  women : ℕ
  men : ℕ
  children : ℕ
  children_left : ℕ
  people_stayed : ℕ

/-- The specific movie screening instance from the problem -/
def problem_screening : MovieScreening :=
  { total_guests := 50
  , women := 25
  , men := 15
  , children := 10
  , children_left := 4
  , people_stayed := 43
  }

/-- Theorem stating that the fraction of men who left is 1/5 -/
theorem fraction_of_men_left (s : MovieScreening) 
  (h1 : s.total_guests = 50)
  (h2 : s.women = s.total_guests / 2)
  (h3 : s.men = 15)
  (h4 : s.children = s.total_guests - s.women - s.men)
  (h5 : s.children_left = 4)
  (h6 : s.people_stayed = 43) :
  (s.total_guests - s.people_stayed - s.children_left) / s.men = 1 / 5 := by
  sorry

end fraction_of_men_left_l1405_140549


namespace pea_patch_problem_l1405_140540

theorem pea_patch_problem (radish_patch : ℝ) (pea_patch : ℝ) :
  radish_patch = 15 →
  pea_patch = 2 * radish_patch →
  pea_patch / 6 = 5 := by
  sorry

end pea_patch_problem_l1405_140540


namespace baking_powder_difference_l1405_140574

def baking_powder_yesterday : ℝ := 0.4
def baking_powder_today : ℝ := 0.3

theorem baking_powder_difference :
  baking_powder_yesterday - baking_powder_today = 0.1 := by
  sorry

end baking_powder_difference_l1405_140574


namespace race_finish_order_l1405_140587

-- Define the athletes
inductive Athlete : Type
| Grisha : Athlete
| Sasha : Athlete
| Lena : Athlete

-- Define the race
structure Race where
  start_order : List Athlete
  finish_order : List Athlete
  overtakes : Athlete → Nat
  no_triple_overtake : Bool

-- Define the specific race conditions
def race_conditions (r : Race) : Prop :=
  r.start_order = [Athlete.Grisha, Athlete.Sasha, Athlete.Lena] ∧
  r.overtakes Athlete.Grisha = 10 ∧
  r.overtakes Athlete.Lena = 6 ∧
  r.overtakes Athlete.Sasha = 4 ∧
  r.no_triple_overtake = true ∧
  r.finish_order.length = 3 ∧
  r.finish_order.Nodup

-- Theorem statement
theorem race_finish_order (r : Race) :
  race_conditions r →
  r.finish_order = [Athlete.Grisha, Athlete.Sasha, Athlete.Lena] :=
by sorry

end race_finish_order_l1405_140587


namespace present_age_of_B_prove_present_age_of_B_l1405_140533

theorem present_age_of_B : ℕ → ℕ → Prop :=
  fun a b =>
    (a + 30 = 2 * (b - 30)) →  -- In 30 years, A will be twice as old as B was 30 years ago
    (a = b + 5) →              -- A is now 5 years older than B
    (b = 95)                   -- B's current age is 95 years

-- The proof of the theorem
theorem prove_present_age_of_B : ∃ a b : ℕ, present_age_of_B a b :=
  sorry

end present_age_of_B_prove_present_age_of_B_l1405_140533


namespace no_integer_list_with_mean_6_35_l1405_140544

theorem no_integer_list_with_mean_6_35 :
  ¬ ∃ (lst : List ℤ), lst.length = 35 ∧ (lst.sum : ℚ) / 35 = 35317 / 5560 := by
  sorry

end no_integer_list_with_mean_6_35_l1405_140544


namespace line_through_point_with_equal_intercepts_l1405_140503

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ
  nonzero : a ≠ 0 ∨ b ≠ 0

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has equal intercepts on both axes
def equalIntercepts (l : Line2D) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c / l.a = l.c / l.b

-- Theorem statement
theorem line_through_point_with_equal_intercepts :
  ∀ (l : Line2D),
    pointOnLine { x := 1, y := 2 } l →
    equalIntercepts l →
    (l.a = 2 ∧ l.b = -1 ∧ l.c = 0) ∨ (l.a = 1 ∧ l.b = 1 ∧ l.c = -3) :=
sorry

end line_through_point_with_equal_intercepts_l1405_140503


namespace janine_reading_ratio_l1405_140521

/-- The number of books Janine read last month -/
def books_last_month : ℕ := 5

/-- The number of pages in each book -/
def pages_per_book : ℕ := 10

/-- The total number of pages Janine read in two months -/
def total_pages : ℕ := 150

/-- The number of books Janine read this month -/
def books_this_month : ℕ := (total_pages - books_last_month * pages_per_book) / pages_per_book

/-- The ratio of books read this month to last month -/
def book_ratio : ℚ := books_this_month / books_last_month

theorem janine_reading_ratio :
  book_ratio = 2 := by
  sorry

end janine_reading_ratio_l1405_140521


namespace expression_evaluation_l1405_140528

theorem expression_evaluation : -30 + 5 * (9 / (3 + 3)) = -22.5 := by
  sorry

end expression_evaluation_l1405_140528


namespace exists_valid_painting_33_exists_valid_painting_32_l1405_140593

/-- Represents a cell on the board -/
structure Cell :=
  (x : Fin 7)
  (y : Fin 7)

/-- Checks if two cells are adjacent -/
def adjacent (c1 c2 : Cell) : Prop :=
  (c1.x = c2.x ∧ c1.y.val + 1 = c2.y.val) ∨
  (c1.x = c2.x ∧ c1.y.val = c2.y.val + 1) ∨
  (c1.x.val + 1 = c2.x.val ∧ c1.y = c2.y) ∨
  (c1.x.val = c2.x.val + 1 ∧ c1.y = c2.y)

/-- A valid painting sequence -/
def ValidPainting (seq : List Cell) : Prop :=
  seq.length > 0 ∧
  ∀ i j, 0 < i ∧ i < seq.length → 0 ≤ j ∧ j < i - 1 →
    (adjacent (seq.get ⟨i, sorry⟩) (seq.get ⟨i-1, sorry⟩) ∧
     ¬adjacent (seq.get ⟨i, sorry⟩) (seq.get ⟨j, sorry⟩))

/-- Main theorem: There exists a valid painting of 33 cells -/
theorem exists_valid_painting_33 :
  ∃ (seq : List Cell), seq.length = 33 ∧ ValidPainting seq :=
sorry

/-- Corollary: There exists a valid painting of 32 cells -/
theorem exists_valid_painting_32 :
  ∃ (seq : List Cell), seq.length = 32 ∧ ValidPainting seq :=
sorry

end exists_valid_painting_33_exists_valid_painting_32_l1405_140593


namespace inequality_solution_l1405_140525

theorem inequality_solution (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  {x : ℝ | (x - a) * (x - 1/a) < 0} = {x : ℝ | a < x ∧ x < 1/a} := by
sorry

end inequality_solution_l1405_140525


namespace correct_proposition_l1405_140516

theorem correct_proposition :
  ∀ (p q : Prop),
    (p ∨ q) →
    ¬(p ∧ q) →
    ¬p →
    (p ↔ (5 + 2 = 6)) →
    (q ↔ (6 > 2)) →
    (¬p ∧ q) :=
by sorry

end correct_proposition_l1405_140516


namespace triangle_vector_parallel_l1405_140569

/-- Given a triangle ABC with sides a, b, c, if the vector (sin B - sin A, √3a + c) 
    is parallel to the vector (sin C, a + b), then angle B = 5π/6 -/
theorem triangle_vector_parallel (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angles : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π)
  (h_sum : A + B + C = π)
  (h_law_of_sines : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C)
  (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ 
    k * (Real.sin B - Real.sin A) = Real.sin C ∧
    k * (Real.sqrt 3 * a + c) = a + b) :
  B = 5 * π / 6 := by
  sorry

end triangle_vector_parallel_l1405_140569


namespace nell_card_count_l1405_140552

/-- The number of cards Nell has after receiving cards from Jeff -/
def total_cards (initial : Float) (received : Float) : Float :=
  initial + received

/-- Theorem stating that Nell's total cards is the sum of her initial cards and received cards -/
theorem nell_card_count (initial : Float) (received : Float) :
  total_cards initial received = initial + received := by sorry

end nell_card_count_l1405_140552


namespace not_all_datasets_have_regression_equation_l1405_140595

-- Define a type for datasets
def Dataset : Type := Set (ℝ × ℝ)

-- Define a predicate for whether a dataset has a regression equation
def has_regression_equation (d : Dataset) : Prop := sorry

-- Theorem stating that not every dataset has a regression equation
theorem not_all_datasets_have_regression_equation : 
  ¬ (∀ d : Dataset, has_regression_equation d) := by sorry

end not_all_datasets_have_regression_equation_l1405_140595


namespace normal_distribution_std_dev_l1405_140512

theorem normal_distribution_std_dev (μ σ : ℝ) : 
  μ = 55 → μ - 3 * σ > 48 → σ < 7/3 := by sorry

end normal_distribution_std_dev_l1405_140512


namespace odot_four_three_l1405_140500

-- Define the binary operation ⊙
def odot (a b : ℝ) : ℝ := 5 * a + 2 * b

-- Theorem statement
theorem odot_four_three : odot 4 3 = 26 := by
  sorry

end odot_four_three_l1405_140500


namespace common_ratio_equation_l1405_140568

/-- A geometric progression with positive terms where the first term is equal to the sum of the next three terms -/
structure GeometricProgression where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  a_pos : a > 0
  r_pos : r > 0
  sum_condition : a = a * r + a * r^2 + a * r^3

/-- The common ratio of the geometric progression satisfies the equation r^3 + r^2 + r - 1 = 0 -/
theorem common_ratio_equation (gp : GeometricProgression) : gp.r^3 + gp.r^2 + gp.r - 1 = 0 := by
  sorry

end common_ratio_equation_l1405_140568


namespace pepperjack_cheese_probability_l1405_140513

theorem pepperjack_cheese_probability :
  let cheddar : ℕ := 15
  let mozzarella : ℕ := 30
  let pepperjack : ℕ := 45
  let total : ℕ := cheddar + mozzarella + pepperjack
  (pepperjack : ℚ) / (total : ℚ) = 1/2 := by
  sorry

end pepperjack_cheese_probability_l1405_140513


namespace expected_value_of_12_sided_die_l1405_140543

/-- A fair 12-sided die -/
def fair_12_sided_die : Finset ℕ := Finset.range 12

/-- The probability of each outcome for a fair 12-sided die -/
def prob (n : ℕ) : ℚ := if n ∈ fair_12_sided_die then 1 / 12 else 0

/-- The expected value of a roll of a fair 12-sided die -/
def expected_value : ℚ := (fair_12_sided_die.sum (λ x => x * prob x)) / 1

theorem expected_value_of_12_sided_die : expected_value = 13/2 := by sorry

end expected_value_of_12_sided_die_l1405_140543


namespace decimal_sum_to_fraction_l1405_140564

theorem decimal_sum_to_fraction : 
  (0.2 : ℚ) + 0.04 + 0.006 + 0.0008 + 0.00001 = 24681 / 100000 := by
  sorry

end decimal_sum_to_fraction_l1405_140564


namespace hyperbola_circle_intersection_sum_of_squares_l1405_140590

/-- A hyperbola centered at the origin -/
structure Hyperbola where
  a : ℝ
  equation : ∀ (x y : ℝ), x^2 - y^2 = a^2

/-- A circle with center at the origin -/
structure Circle where
  r : ℝ
  equation : ∀ (x y : ℝ), x^2 + y^2 = r^2

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance of a point from the origin -/
def distance_from_origin (p : Point) : ℝ := p.x^2 + p.y^2

theorem hyperbola_circle_intersection_sum_of_squares 
  (h : Hyperbola) (c : Circle) (P Q R S : Point) :
  (P.x^2 - P.y^2 = h.a^2) →
  (Q.x^2 - Q.y^2 = h.a^2) →
  (R.x^2 - R.y^2 = h.a^2) →
  (S.x^2 - S.y^2 = h.a^2) →
  (P.x^2 + P.y^2 = c.r^2) →
  (Q.x^2 + Q.y^2 = c.r^2) →
  (R.x^2 + R.y^2 = c.r^2) →
  (S.x^2 + S.y^2 = c.r^2) →
  distance_from_origin P + distance_from_origin Q + 
  distance_from_origin R + distance_from_origin S = 4 * c.r^2 := by
  sorry

end hyperbola_circle_intersection_sum_of_squares_l1405_140590


namespace special_rectangle_dimensions_l1405_140598

/-- A rectangle with the given properties -/
structure SpecialRectangle where
  width : ℝ
  length : ℝ
  perimeter_area_relation : 2 * (width + length) = 3 * (width * length)
  length_width_relation : length = 2 * width

/-- The dimensions of the special rectangle are 1 inch width and 2 inches length -/
theorem special_rectangle_dimensions (rect : SpecialRectangle) : rect.width = 1 ∧ rect.length = 2 := by
  sorry

#check special_rectangle_dimensions

end special_rectangle_dimensions_l1405_140598


namespace circular_sum_equivalence_l1405_140559

/-- 
Given integers n > m > 1 arranged in a circle, s_i is the sum of m integers 
starting at the i-th position moving clockwise, and t_i is the sum of the 
remaining n-m integers. f(a, b) is the number of elements i in {1, 2, ..., n} 
such that s_i ≡ a (mod 4) and t_i ≡ b (mod 4).
-/
def f (n m : ℕ) (a b : ℕ) : ℕ := sorry

/-- The main theorem to be proved -/
theorem circular_sum_equivalence (n m : ℕ) (h1 : n > m) (h2 : m > 1) :
  f n m 1 3 ≡ f n m 3 1 [MOD 4] ↔ Even (f n m 2 2) := by sorry

end circular_sum_equivalence_l1405_140559


namespace morgan_experiment_correct_l1405_140520

/-- Statements about biological experiments and research -/
inductive BiologicalStatement
| A : BiologicalStatement -- Ovary of locusts for observing animal cell meiosis
| B : BiologicalStatement -- Morgan's fruit fly experiment
| C : BiologicalStatement -- Hydrogen peroxide as substrate in enzyme activity experiment
| D : BiologicalStatement -- Investigating red-green color blindness incidence

/-- Predicate to determine if a biological statement is correct -/
def is_correct : BiologicalStatement → Prop
| BiologicalStatement.A => False
| BiologicalStatement.B => True
| BiologicalStatement.C => False
| BiologicalStatement.D => False

/-- Theorem stating that Morgan's fruit fly experiment statement is correct -/
theorem morgan_experiment_correct :
  is_correct BiologicalStatement.B :=
by sorry

end morgan_experiment_correct_l1405_140520


namespace joker_king_probability_l1405_140534

/-- A deck of cards with jokers -/
structure Deck :=
  (total_cards : ℕ)
  (num_jokers : ℕ)
  (num_kings : ℕ)

/-- The probability of drawing a joker first and a king second -/
def joker_king_prob (d : Deck) : ℚ :=
  (d.num_jokers : ℚ) / d.total_cards * (d.num_kings : ℚ) / (d.total_cards - 1)

/-- The modified 54-card deck -/
def modified_deck : Deck :=
  { total_cards := 54,
    num_jokers := 2,
    num_kings := 4 }

theorem joker_king_probability :
  joker_king_prob modified_deck = 8 / 1431 := by
  sorry

end joker_king_probability_l1405_140534


namespace sam_mystery_books_l1405_140517

/-- Represents the number of books in each category --/
structure BookCount where
  adventure : ℕ
  mystery : ℕ
  used : ℕ
  new : ℕ

/-- The total number of books is the sum of used and new books --/
def total_books (b : BookCount) : ℕ := b.used + b.new

/-- Theorem stating the number of mystery books Sam bought --/
theorem sam_mystery_books :
  ∃ (b : BookCount),
    b.adventure = 13 ∧
    b.used = 15 ∧
    b.new = 15 ∧
    total_books b = b.adventure + b.mystery ∧
    b.mystery = 17 := by
  sorry

end sam_mystery_books_l1405_140517


namespace bumper_car_queue_count_l1405_140539

theorem bumper_car_queue_count : ∀ (initial leaving joining : ℕ),
  initial = 9 →
  leaving = 6 →
  joining = 3 →
  initial + joining = 12 :=
by
  sorry

end bumper_car_queue_count_l1405_140539


namespace cube_volume_from_face_perimeter_l1405_140565

/-- Given a cube with face perimeter of 40 cm, its volume is 1000 cubic centimeters. -/
theorem cube_volume_from_face_perimeter (face_perimeter : ℝ) (volume : ℝ) :
  face_perimeter = 40 → volume = 1000 := by
  sorry

end cube_volume_from_face_perimeter_l1405_140565


namespace appropriate_sampling_methods_l1405_140556

structure Survey where
  population_size : ℕ
  sample_size : ℕ
  has_distinct_subgroups : Bool

def is_large_population (s : Survey) : Bool :=
  s.population_size ≥ 1000

def is_small_population (s : Survey) : Bool :=
  s.population_size < 100

def stratified_sampling_appropriate (s : Survey) : Bool :=
  is_large_population s ∧ s.has_distinct_subgroups

def simple_random_sampling_appropriate (s : Survey) : Bool :=
  is_small_population s ∧ ¬s.has_distinct_subgroups

theorem appropriate_sampling_methods 
  (survey_A survey_B : Survey)
  (h_A : survey_A.population_size = 20000 ∧ survey_A.sample_size = 200 ∧ survey_A.has_distinct_subgroups = true)
  (h_B : survey_B.population_size = 15 ∧ survey_B.sample_size = 3 ∧ survey_B.has_distinct_subgroups = false) :
  stratified_sampling_appropriate survey_A ∧ simple_random_sampling_appropriate survey_B :=
sorry

end appropriate_sampling_methods_l1405_140556


namespace limit_of_polynomial_at_two_l1405_140545

theorem limit_of_polynomial_at_two :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → |4*x^2 - 6*x + 3 - 7| < ε :=
by sorry

end limit_of_polynomial_at_two_l1405_140545


namespace ellipse_distance_theorem_l1405_140555

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- Theorem about the distance AF₂ in the given ellipse problem -/
theorem ellipse_distance_theorem (E : Ellipse) (F₁ F₂ A B : Point) : 
  -- F₁ and F₂ are the foci of E
  (∀ P : Point, distance P F₁ + distance P F₂ = 2 * E.a) →
  -- Line through F₁ intersects E at A and B
  (∃ t : ℝ, A = ⟨t * F₁.x, t * F₁.y⟩ ∧ B = ⟨(1 - t) * F₁.x, (1 - t) * F₁.y⟩) →
  -- |AF₁| = 3|F₁B|
  distance A F₁ = 3 * distance F₁ B →
  -- |AB| = 4
  distance A B = 4 →
  -- Perimeter of triangle ABF₂ is 16
  distance A B + distance B F₂ + distance F₂ A = 16 →
  -- Then |AF₂| = 5
  distance A F₂ = 5 := by sorry

end ellipse_distance_theorem_l1405_140555


namespace function_derivative_at_midpoint_negative_l1405_140580

open Real

theorem function_derivative_at_midpoint_negative 
  (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a > 0) 
  (hx : 0 < x₁ ∧ x₁ < x₂) 
  (hf : ∀ x, f x = log x - a * x + 1) 
  (hz : f x₁ = 0 ∧ f x₂ = 0) : 
  deriv f ((x₁ + x₂) / 2) < 0 := by
  sorry


end function_derivative_at_midpoint_negative_l1405_140580


namespace quadratic_increasing_negative_l1405_140505

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * x^2

-- Theorem statement
theorem quadratic_increasing_negative (x₁ x₂ : ℝ) :
  x₁ < x₂ ∧ x₂ < 0 → f x₁ < f x₂ := by
  sorry

end quadratic_increasing_negative_l1405_140505


namespace largest_integer_inequality_l1405_140547

theorem largest_integer_inequality : ∀ y : ℤ, y ≤ 3 ↔ (y : ℚ) / 4 + 6 / 7 < 7 / 4 := by sorry

end largest_integer_inequality_l1405_140547


namespace prob_not_same_group_three_groups_l1405_140586

/-- The probability that two students are not in the same interest group -/
def prob_not_same_group (num_groups : ℕ) : ℚ :=
  if num_groups = 0 then 0
  else (num_groups - 1 : ℚ) / num_groups

theorem prob_not_same_group_three_groups :
  prob_not_same_group 3 = 2/3 := by sorry

end prob_not_same_group_three_groups_l1405_140586


namespace geometric_sequence_formula_l1405_140571

/-- Given a geometric sequence {a_n} where a_5 = 7 and a_8 = 56, 
    prove that the general formula is a_n = (7/32) * 2^n -/
theorem geometric_sequence_formula (a : ℕ → ℝ) 
  (h1 : a 5 = 7) 
  (h2 : a 8 = 56) 
  (h_geom : ∀ n m : ℕ, a (n + m) = a n * (a (n + 1) / a n) ^ m) :
  ∃ q : ℝ, ∀ n : ℕ, a n = (7 / 32) * 2^n :=
sorry

end geometric_sequence_formula_l1405_140571


namespace factorization_proof_l1405_140572

theorem factorization_proof (x y : ℝ) : 
  (x^2 - 9*y^2 = (x+3*y)*(x-3*y)) ∧ 
  (x^2*y - 6*x*y + 9*y = y*(x-3)^2) ∧ 
  (9*(x+2*y)^2 - 4*(x-y)^2 = (5*x+4*y)*(x+8*y)) ∧ 
  ((x-1)*(x-3) + 1 = (x-2)^2) := by
  sorry

end factorization_proof_l1405_140572


namespace finite_difference_polynomial_l1405_140522

/-- The finite difference operator -/
def finite_difference (f : ℕ → ℚ) : ℕ → ℚ := λ x => f (x + 1) - f x

/-- The n-th finite difference -/
def nth_finite_difference (n : ℕ) (f : ℕ → ℚ) : ℕ → ℚ :=
  match n with
  | 0 => f
  | n + 1 => finite_difference (nth_finite_difference n f)

/-- Polynomial of degree m -/
def polynomial_degree_m (m : ℕ) (coeffs : Fin (m + 1) → ℚ) : ℕ → ℚ :=
  λ x => (Finset.range (m + 1)).sum (λ i => coeffs i * x^i)

theorem finite_difference_polynomial (m n : ℕ) (coeffs : Fin (m + 1) → ℚ) :
  (m < n → ∀ x, nth_finite_difference n (polynomial_degree_m m coeffs) x = 0) ∧
  (∀ x, nth_finite_difference m (polynomial_degree_m m coeffs) x = m.factorial * coeffs m) :=
sorry

end finite_difference_polynomial_l1405_140522


namespace thirty_factorial_trailing_zeros_l1405_140526

def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

theorem thirty_factorial_trailing_zeros :
  trailing_zeros 30 = 7 := by
  sorry

end thirty_factorial_trailing_zeros_l1405_140526


namespace problem_solution_l1405_140537

theorem problem_solution (a b : ℝ) 
  (h1 : 5 + a = 6 - b) 
  (h2 : 6 + b = 9 + a) : 
  5 - a = 6 := by
sorry

end problem_solution_l1405_140537


namespace charity_dinner_cost_l1405_140596

/-- The total cost of dinners given the number of plates and the cost of rice and chicken per plate -/
def total_cost (num_plates : ℕ) (rice_cost chicken_cost : ℚ) : ℚ :=
  num_plates * (rice_cost + chicken_cost)

/-- Theorem stating that the total cost for 100 plates with rice costing $0.10 and chicken costing $0.40 per plate is $50.00 -/
theorem charity_dinner_cost :
  total_cost 100 (10 / 100) (40 / 100) = 50 := by
  sorry

end charity_dinner_cost_l1405_140596


namespace rectangular_garden_width_l1405_140536

theorem rectangular_garden_width (width length area : ℝ) : 
  length = 3 * width → 
  area = length * width → 
  area = 432 → 
  width = 12 := by
sorry

end rectangular_garden_width_l1405_140536


namespace linear_function_condition_l1405_140583

/-- A linear function f(x) = ax + b satisfying f⁽¹⁰⁾(x) ≥ 1024x + 1023 
    must have a = 2 and b ≥ 1, or a = -2 and b ≤ -3 -/
theorem linear_function_condition (a b : ℝ) (h : ∀ x, a^10 * x + b * (a^10 - 1) / (a - 1) ≥ 1024 * x + 1023) :
  (a = 2 ∧ b ≥ 1) ∨ (a = -2 ∧ b ≤ -3) := by sorry

end linear_function_condition_l1405_140583


namespace arithmetic_sequence_sum_l1405_140508

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a, prove that if a₃ + a₅ = 12 - a₇, then a₁ + a₉ = 8 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : ArithmeticSequence a) 
    (h1 : a 3 + a 5 = 12 - a 7) : a 1 + a 9 = 8 := by
  sorry

end arithmetic_sequence_sum_l1405_140508


namespace sock_pair_count_l1405_140578

/-- The number of ways to choose a pair of socks with different colors -/
def different_color_pairs (white brown blue red : ℕ) : ℕ :=
  white * brown + white * blue + white * red +
  brown * blue + brown * red +
  blue * red

/-- Theorem: Given 5 white socks, 5 brown socks, 2 blue socks, and 1 red sock,
    the number of ways to choose a pair of socks with different colors is 57 -/
theorem sock_pair_count :
  different_color_pairs 5 5 2 1 = 57 := by
  sorry

end sock_pair_count_l1405_140578


namespace salary_after_changes_l1405_140532

/-- Given an original salary, calculate the final salary after a raise and a reduction -/
def finalSalary (originalSalary : ℚ) (raisePercentage : ℚ) (reductionPercentage : ℚ) : ℚ :=
  let salaryAfterRaise := originalSalary * (1 + raisePercentage / 100)
  salaryAfterRaise * (1 - reductionPercentage / 100)

theorem salary_after_changes : 
  finalSalary 5000 10 5 = 5225 := by sorry

end salary_after_changes_l1405_140532


namespace intersection_A_B_min_value_fraction_l1405_140519

-- Define the parameters b and c based on the given inequality
def b : ℝ := 3
def c : ℝ := 6

-- Define the solution set of the original inequality
def original_solution_set : Set ℝ := {x | 2 < x ∧ x < 3}

-- Define set B
def B : Set ℝ := {x | -2 ≤ x ∧ x < 2}

-- Define the solution set of bx^2 - (c+1)x - c > 0
def A : Set ℝ := {x | b * x^2 - (c + 1) * x - c > 0}

-- Theorem 1: Intersection of A and B
theorem intersection_A_B : A ∩ B = {x | -2 ≤ x ∧ x < -2/3} := by sorry

-- Theorem 2: Minimum value of the fraction
theorem min_value_fraction :
  ∀ x > 1, (x^2 - b*x + c) / (x - 1) ≥ 3 ∧
  ∃ x > 1, (x^2 - b*x + c) / (x - 1) = 3 := by sorry

end intersection_A_B_min_value_fraction_l1405_140519


namespace octal_to_decimal_fraction_l1405_140566

theorem octal_to_decimal_fraction (c d : ℕ) : 
  (5 * 8^2 + 4 * 8 + 7 = 300 + 10 * c + d) → 
  (0 ≤ c) → (c ≤ 9) → (0 ≤ d) → (d ≤ 9) →
  (c * d) / 12 = 5 / 4 := by sorry

end octal_to_decimal_fraction_l1405_140566


namespace total_crosswalk_lines_l1405_140523

/-- Given 5 intersections, 4 crosswalks per intersection, and 20 lines per crosswalk,
    the total number of lines in all crosswalks is 400. -/
theorem total_crosswalk_lines
  (num_intersections : ℕ)
  (crosswalks_per_intersection : ℕ)
  (lines_per_crosswalk : ℕ)
  (h1 : num_intersections = 5)
  (h2 : crosswalks_per_intersection = 4)
  (h3 : lines_per_crosswalk = 20) :
  num_intersections * crosswalks_per_intersection * lines_per_crosswalk = 400 :=
by sorry

end total_crosswalk_lines_l1405_140523


namespace pigsy_fruits_l1405_140529

def process (n : ℕ) : ℕ := 
  (n / 2 + 2) / 2

theorem pigsy_fruits : ∃ x : ℕ, process (process (process (process x))) = 5 ∧ x = 20 := by
  sorry

end pigsy_fruits_l1405_140529


namespace sterilization_tank_solution_l1405_140592

/-- Represents the sterilization tank problem --/
def sterilization_tank_problem (initial_volume : ℝ) (drained_volume : ℝ) (final_concentration : ℝ) (initial_concentration : ℝ) : Prop :=
  let remaining_volume := initial_volume - drained_volume
  remaining_volume * initial_concentration + drained_volume = initial_volume * final_concentration

/-- Theorem stating the solution to the sterilization tank problem --/
theorem sterilization_tank_solution :
  sterilization_tank_problem 100 3.0612244898 0.05 0.02 := by
  sorry

end sterilization_tank_solution_l1405_140592


namespace quadratic_equation_from_means_l1405_140550

theorem quadratic_equation_from_means (a b : ℝ) 
  (h_arithmetic_mean : (a + b) / 2 = 10)
  (h_geometric_mean : Real.sqrt (a * b) = 24) :
  ∀ x, x^2 - 20*x + 576 = 0 ↔ (x = a ∨ x = b) :=
by sorry

end quadratic_equation_from_means_l1405_140550


namespace monroe_collection_legs_l1405_140599

/-- Represents the number of legs for each type of creature -/
structure CreatureLegs where
  ant : Nat
  spider : Nat
  beetle : Nat
  centipede : Nat

/-- Represents the count of each type of creature in the collection -/
structure CreatureCount where
  ants : Nat
  spiders : Nat
  beetles : Nat
  centipedes : Nat

/-- Calculates the total number of legs in the collection -/
def totalLegs (legs : CreatureLegs) (count : CreatureCount) : Nat :=
  legs.ant * count.ants + 
  legs.spider * count.spiders + 
  legs.beetle * count.beetles + 
  legs.centipede * count.centipedes

/-- Theorem: The total number of legs in Monroe's collection is 726 -/
theorem monroe_collection_legs : 
  let legs : CreatureLegs := { ant := 6, spider := 8, beetle := 6, centipede := 100 }
  let count : CreatureCount := { ants := 12, spiders := 8, beetles := 15, centipedes := 5 }
  totalLegs legs count = 726 := by
  sorry

end monroe_collection_legs_l1405_140599


namespace race_head_start_l1405_140582

theorem race_head_start (L : ℝ) (Va Vb : ℝ) (h : Va = (30 / 17) * Vb) :
  ∃ H : ℝ, H = (13 / 30) * L ∧ L / Va = (L - H) / Vb :=
by sorry

end race_head_start_l1405_140582


namespace apple_problem_l1405_140584

theorem apple_problem (initial_apples : ℕ) (sold_to_jill_percent : ℚ) (sold_to_june_percent : ℚ) (given_to_teacher : ℕ) : 
  initial_apples = 150 →
  sold_to_jill_percent = 30 / 100 →
  sold_to_june_percent = 20 / 100 →
  given_to_teacher = 2 →
  initial_apples - 
    (↑initial_apples * sold_to_jill_percent).floor - 
    ((↑initial_apples - (↑initial_apples * sold_to_jill_percent).floor) * sold_to_june_percent).floor - 
    given_to_teacher = 82 :=
by sorry

end apple_problem_l1405_140584


namespace max_stamps_for_50_dollars_l1405_140570

/-- The maximum number of stamps that can be purchased with a given budget and stamp price -/
def maxStamps (budget : ℕ) (stampPrice : ℕ) : ℕ :=
  (budget / stampPrice : ℕ)

/-- Theorem stating the maximum number of stamps that can be purchased with $50 when stamps cost 45 cents each -/
theorem max_stamps_for_50_dollars : maxStamps 5000 45 = 111 := by
  sorry

end max_stamps_for_50_dollars_l1405_140570


namespace sequence_minimum_l1405_140511

theorem sequence_minimum (n : ℤ) : ∃ (m : ℤ), ∀ (n : ℤ), n^2 - 8*n + 5 ≥ m ∧ ∃ (k : ℤ), k^2 - 8*k + 5 = m :=
sorry

end sequence_minimum_l1405_140511


namespace intersection_A_B_quadratic_inequality_solution_l1405_140538

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B : Set ℝ := {x | x^2 + x - 6 > 0}

-- Theorem for part (1)
theorem intersection_A_B : A ∩ B = {x | 2 < x ∧ x < 3} := by sorry

-- Theorem for part (2)
theorem quadratic_inequality_solution (a b : ℝ) :
  ({x : ℝ | x^2 + a*x + b < 0} = {x | 2 < x ∧ x < 3}) ↔ (a = -5 ∧ b = 6) := by sorry

end intersection_A_B_quadratic_inequality_solution_l1405_140538


namespace book_price_proof_l1405_140554

theorem book_price_proof (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 75)
  (h2 : profit_percentage = 25) :
  ∃ original_price : ℝ, 
    original_price * (1 + profit_percentage / 100) = selling_price ∧ 
    original_price = 60 :=
by
  sorry

end book_price_proof_l1405_140554


namespace subtraction_multiplication_equality_l1405_140504

theorem subtraction_multiplication_equality : (3.625 - 1.047) * 4 = 10.312 := by
  sorry

end subtraction_multiplication_equality_l1405_140504


namespace annual_interest_calculation_l1405_140563

/-- Calculates the simple interest for a given principal, rate, and time. -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- The loan amount in dollars -/
def loan_amount : ℝ := 9000

/-- The annual interest rate as a decimal -/
def interest_rate : ℝ := 0.09

/-- The time period in years -/
def time_period : ℝ := 1

theorem annual_interest_calculation :
  simple_interest loan_amount interest_rate time_period = 810 := by
  sorry

end annual_interest_calculation_l1405_140563


namespace triangles_containing_center_201_l1405_140560

/-- Given a regular 201-sided polygon inscribed in a circle with center C,
    this function computes the number of triangles formed by connecting
    any three vertices of the polygon such that C lies inside the triangle. -/
def triangles_containing_center (n : ℕ) : ℕ :=
  if n = 201 then
    let vertex_count := n
    let half_vertex_count := (vertex_count - 1) / 2
    let triangles_per_vertex := half_vertex_count * (half_vertex_count + 1) / 2
    vertex_count * triangles_per_vertex / 3
  else
    0

/-- Theorem stating that the number of triangles containing the center
    for a regular 201-sided polygon is 338350. -/
theorem triangles_containing_center_201 :
  triangles_containing_center 201 = 338350 := by
  sorry

end triangles_containing_center_201_l1405_140560


namespace pool_capacity_l1405_140575

theorem pool_capacity (C : ℝ) 
  (h1 : 0.8 * C - 0.5 * C = 300) : C = 1000 := by
  sorry

end pool_capacity_l1405_140575


namespace distance_product_l1405_140518

noncomputable def f (x : ℝ) : ℝ := 2 * x + 5 / x

theorem distance_product (x : ℝ) (hx : x ≠ 0) :
  let P : ℝ × ℝ := (x, f x)
  let d₁ : ℝ := |f x - 2 * x| / Real.sqrt 5
  let d₂ : ℝ := |x|
  d₁ * d₂ = Real.sqrt 5 := by
  sorry

end distance_product_l1405_140518


namespace exactly_three_solutions_l1405_140594

-- Define the greatest integer function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then (floor x : ℝ) / x else 0

-- State the theorem
theorem exactly_three_solutions (a : ℝ) (h : 3/4 < a ∧ a ≤ 4/5) :
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, x > 0 ∧ f x = a :=
sorry

end exactly_three_solutions_l1405_140594


namespace power_of_seven_mod_hundred_l1405_140542

theorem power_of_seven_mod_hundred : 7^700 % 100 = 1 := by
  sorry

end power_of_seven_mod_hundred_l1405_140542


namespace lisa_spoons_count_l1405_140573

/-- The number of spoons Lisa has after combining all sets -/
def total_spoons (num_children : ℕ) (baby_spoons_per_child : ℕ) (decorative_spoons : ℕ)
  (large_spoons : ℕ) (dessert_spoons : ℕ) (soup_spoons : ℕ) (teaspoons : ℕ) : ℕ :=
  num_children * baby_spoons_per_child + decorative_spoons +
  large_spoons + dessert_spoons + soup_spoons + teaspoons

/-- Theorem stating that Lisa has 98 spoons in total -/
theorem lisa_spoons_count :
  total_spoons 6 4 4 20 10 15 25 = 98 := by
  sorry

end lisa_spoons_count_l1405_140573


namespace production_cost_correct_l1405_140509

/-- The production cost per performance for Steve's circus investment -/
def production_cost_per_performance : ℝ := 7000

/-- The overhead cost for the circus production -/
def overhead_cost : ℝ := 81000

/-- The income from a single sold-out performance -/
def sold_out_income : ℝ := 16000

/-- The number of sold-out performances needed to break even -/
def break_even_performances : ℕ := 9

/-- Theorem stating that the production cost per performance is correct -/
theorem production_cost_correct :
  production_cost_per_performance * break_even_performances + overhead_cost =
  sold_out_income * break_even_performances :=
by sorry

end production_cost_correct_l1405_140509


namespace meeting_time_and_bridge_location_l1405_140501

/-- Represents a time of day in hours and minutes -/
structure TimeOfDay where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents a journey between two villages -/
structure Journey where
  startTime : TimeOfDay
  endTime : TimeOfDay
  deriving Repr

/-- Calculates the duration of a journey in minutes -/
def journeyDuration (j : Journey) : Nat :=
  (j.endTime.hours - j.startTime.hours) * 60 + j.endTime.minutes - j.startTime.minutes

/-- Theorem: Meeting time and bridge location -/
theorem meeting_time_and_bridge_location
  (womanJourney : Journey)
  (manJourney : Journey)
  (hWoman : womanJourney = ⟨⟨10, 31⟩, ⟨13, 43⟩⟩)
  (hMan : manJourney = ⟨⟨9, 13⟩, ⟨11, 53⟩⟩)
  (hSameRoad : True)  -- They travel on the same road
  (hConstantSpeed : True)  -- Both travel at constant speeds
  (hBridgeCrossing : True)  -- Woman crosses bridge 1 minute later than man
  : ∃ (meetingTime : TimeOfDay) (bridgeFromA bridgeFromB : Nat),
    meetingTime = ⟨11, 13⟩ ∧
    bridgeFromA = 7 ∧
    bridgeFromB = 24 :=
by sorry

end meeting_time_and_bridge_location_l1405_140501


namespace sticker_distribution_l1405_140562

theorem sticker_distribution (n m : ℕ) (hn : n = 5) (hm : m = 5) :
  (Nat.choose (n + m - 1) (m - 1) : ℕ) = 126 := by
  sorry

end sticker_distribution_l1405_140562


namespace square_minus_product_l1405_140561

theorem square_minus_product (a : ℝ) : (a - 1)^2 - a*(a - 1) = -a + 1 := by
  sorry

end square_minus_product_l1405_140561


namespace both_miss_probability_l1405_140548

/-- The probability that both shooters miss the target given their individual hit probabilities -/
theorem both_miss_probability (p_hit_A p_hit_B : ℝ) (h_A : p_hit_A = 0.85) (h_B : p_hit_B = 0.8) :
  (1 - p_hit_A) * (1 - p_hit_B) = 0.03 := by
  sorry

end both_miss_probability_l1405_140548


namespace odd_function_value_l1405_140581

-- Define an odd function f
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem odd_function_value :
  (∀ x : ℝ, f (-x) = -f x) →  -- f is odd
  (∀ x < 0, f x = 2^x) →      -- f(x) = 2^x for x < 0
  f (Real.log 9 / Real.log 4) = -1/3 := by
sorry

end odd_function_value_l1405_140581


namespace inverse_equals_original_at_three_l1405_140514

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x - 9

-- Define the property of being an inverse function
def is_inverse (f g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = x ∧ g (f x) = x

-- Theorem statement
theorem inverse_equals_original_at_three :
  ∃ g_inv : ℝ → ℝ, is_inverse g g_inv ∧
  ∀ x : ℝ, g x = g_inv x ↔ x = 3 :=
sorry

end inverse_equals_original_at_three_l1405_140514


namespace intersection_of_A_and_B_l1405_140535

def A : Set ℝ := {x | x^2 - 1 ≥ 0}
def B : Set ℝ := {x | 0 < x ∧ x < 4}

theorem intersection_of_A_and_B : A ∩ B = {x | 1 ≤ x ∧ x < 4} := by sorry

end intersection_of_A_and_B_l1405_140535


namespace arccos_cos_three_l1405_140558

theorem arccos_cos_three : Real.arccos (Real.cos 3) = 3 - 2 * Real.pi := by
  sorry

end arccos_cos_three_l1405_140558


namespace fence_price_per_foot_l1405_140546

/-- Given a square plot with area and total fencing cost, calculate the price per foot of fencing --/
theorem fence_price_per_foot 
  (area : ℝ) 
  (total_cost : ℝ) 
  (h1 : area = 289) 
  (h2 : total_cost = 3740) : 
  total_cost / (4 * Real.sqrt area) = 55 := by
  sorry

end fence_price_per_foot_l1405_140546


namespace frank_breakfast_shopping_l1405_140515

/-- The cost of one bun in dollars -/
def bun_cost : ℚ := 1/10

/-- The cost of one bottle of milk in dollars -/
def milk_cost : ℚ := 2

/-- The number of bottles of milk Frank bought -/
def milk_bottles : ℕ := 2

/-- The cost of a carton of eggs in dollars -/
def egg_cost : ℚ := 3 * milk_cost

/-- The total amount Frank paid in dollars -/
def total_paid : ℚ := 11

/-- The number of buns Frank bought -/
def buns_bought : ℕ := 10

theorem frank_breakfast_shopping :
  buns_bought * bun_cost + milk_bottles * milk_cost + egg_cost = total_paid :=
by sorry

end frank_breakfast_shopping_l1405_140515
