import Mathlib

namespace NUMINAMATH_CALUDE_valid_partition_exists_l652_65272

/-- Represents a person in the group -/
structure Person where
  id : Nat

/-- Represents the friendship and enmity relations in the group -/
structure Relations (P : Type) where
  friend : P → P
  enemy : P → P

/-- Represents a partition of the group into two subsets -/
structure Partition (P : Type) where
  set1 : Set P
  set2 : Set P
  partition_complete : set1 ∪ set2 = Set.univ
  partition_disjoint : set1 ∩ set2 = ∅

/-- The main theorem stating that a valid partition exists -/
theorem valid_partition_exists (P : Type) [Finite P] (r : Relations P) 
  (friend_injective : Function.Injective r.friend)
  (enemy_injective : Function.Injective r.enemy)
  (friend_enemy_distinct : ∀ p : P, r.friend p ≠ r.enemy p) :
  ∃ (part : Partition P), 
    (∀ p ∈ part.set1, r.friend p ∉ part.set1 ∧ r.enemy p ∉ part.set1) ∧
    (∀ p ∈ part.set2, r.friend p ∉ part.set2 ∧ r.enemy p ∉ part.set2) :=
  sorry

end NUMINAMATH_CALUDE_valid_partition_exists_l652_65272


namespace NUMINAMATH_CALUDE_car_city_efficiency_l652_65280

/-- Represents the fuel efficiency of a car -/
structure CarEfficiency where
  highway : ℝ  -- Miles per gallon on highway
  city : ℝ     -- Miles per gallon in city
  tank : ℝ     -- Tank size in gallons

/-- Theorem stating the car's fuel efficiency in the city given the conditions -/
theorem car_city_efficiency (car : CarEfficiency) 
  (highway_distance : car.highway * car.tank = 900)
  (city_distance : car.city * car.tank = 600)
  (efficiency_difference : car.city = car.highway - 5) :
  car.city = 10 := by sorry

end NUMINAMATH_CALUDE_car_city_efficiency_l652_65280


namespace NUMINAMATH_CALUDE_class_composition_l652_65278

theorem class_composition (girls boys : ℕ) : 
  girls * 6 = boys * 5 →  -- Initial ratio of girls to boys is 5:6
  (girls - 20) * 3 = boys * 2 →  -- New ratio after 20 girls leave is 2:3
  boys = 120 := by  -- The number of boys in the class is 120
sorry

end NUMINAMATH_CALUDE_class_composition_l652_65278


namespace NUMINAMATH_CALUDE_equal_vectors_have_equal_magnitudes_l652_65246

theorem equal_vectors_have_equal_magnitudes {V : Type*} [NormedAddCommGroup V] 
  {a b : V} (h : a = b) : ‖a‖ = ‖b‖ := by
  sorry

end NUMINAMATH_CALUDE_equal_vectors_have_equal_magnitudes_l652_65246


namespace NUMINAMATH_CALUDE_tv_price_change_l652_65235

theorem tv_price_change (x : ℝ) : 
  (1 - x / 100) * (1 + 40 / 100) = 1 + 12 / 100 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_change_l652_65235


namespace NUMINAMATH_CALUDE_min_ratio_bounds_l652_65268

/-- An equiangular hexagon with alternating side lengths 1 and a -/
structure EquiangularHexagon :=
  (a : ℝ)

/-- A circle intersecting the hexagon at 12 distinct points -/
structure IntersectingCircle (h : EquiangularHexagon) :=
  (exists_intersection : True)

/-- The bounds M and N for the side length a -/
structure Bounds (h : EquiangularHexagon) (c : IntersectingCircle h) :=
  (M N : ℝ)
  (lower_bound : M < h.a)
  (upper_bound : h.a < N)

/-- The theorem stating the minimum possible value of N/M -/
theorem min_ratio_bounds 
  (h : EquiangularHexagon) 
  (c : IntersectingCircle h) 
  (b : Bounds h c) : 
  ∃ (M N : ℝ), M < h.a ∧ h.a < N ∧ 
  ∀ (M' N' : ℝ), M' < h.a → h.a < N' → (3 * Real.sqrt 3 + 3) / 2 ≤ N' / M' :=
sorry

end NUMINAMATH_CALUDE_min_ratio_bounds_l652_65268


namespace NUMINAMATH_CALUDE_ellipse_parabola_intersection_hyperbola_l652_65270

/-- Given an ellipse and a parabola that intersect, prove that the radius of the circumcircle
    of the triangle formed by their intersection points and the origin, along with the parameter
    of the parabola, satisfy a hyperbolic equation. -/
theorem ellipse_parabola_intersection_hyperbola (p r : ℝ) (hp : p > 0) (hr : r > 0) :
  (∃ x y : ℝ, x^2/4 + y^2/2 = 1 ∧ y^2 = 2*p*x) →
  (∃ x₀ y₀ : ℝ, x₀^2/4 + y₀^2/2 = 1 ∧ y₀^2 = 2*p*x₀ ∧ x₀^2 + y₀^2 = r^2) →
  r^2 - p^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_parabola_intersection_hyperbola_l652_65270


namespace NUMINAMATH_CALUDE_theater_ticket_difference_l652_65210

theorem theater_ticket_difference :
  ∀ (orchestra_tickets balcony_tickets : ℕ),
  orchestra_tickets + balcony_tickets = 370 →
  12 * orchestra_tickets + 8 * balcony_tickets = 3320 →
  balcony_tickets - orchestra_tickets = 190 :=
by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_difference_l652_65210


namespace NUMINAMATH_CALUDE_flower_count_l652_65277

theorem flower_count (red_green : ℕ) (red_yellow : ℕ) (green_yellow : ℕ)
  (h1 : red_green = 62)
  (h2 : red_yellow = 49)
  (h3 : green_yellow = 77) :
  ∃ (red green yellow : ℕ),
    red + green = red_green ∧
    red + yellow = red_yellow ∧
    green + yellow = green_yellow ∧
    red = 17 ∧ green = 45 ∧ yellow = 32 := by
  sorry

end NUMINAMATH_CALUDE_flower_count_l652_65277


namespace NUMINAMATH_CALUDE_sum_of_ages_l652_65236

/-- Given that Jed is 10 years older than Matt and in 10 years, Jed will be 25 years old,
    prove that the sum of their present ages is 20. -/
theorem sum_of_ages (jed_age matt_age : ℕ) : 
  jed_age = matt_age + 10 →  -- Jed is 10 years older than Matt
  jed_age + 10 = 25 →        -- In 10 years, Jed will be 25 years old
  jed_age + matt_age = 20 :=   -- The sum of their present ages is 20
by sorry

end NUMINAMATH_CALUDE_sum_of_ages_l652_65236


namespace NUMINAMATH_CALUDE_school_club_revenue_l652_65284

/-- Represents the revenue from full-price tickets in a school club event. -/
def revenue_full_price (total_tickets : ℕ) (total_revenue : ℚ) : ℚ :=
  let full_price : ℚ := 30
  let full_price_tickets : ℕ := 45
  full_price * full_price_tickets

/-- Proves that the revenue from full-price tickets is $1350 given the conditions. -/
theorem school_club_revenue 
  (total_tickets : ℕ) 
  (total_revenue : ℚ) 
  (h_tickets : total_tickets = 160)
  (h_revenue : total_revenue = 2500) :
  revenue_full_price total_tickets total_revenue = 1350 := by
  sorry

#eval revenue_full_price 160 2500

end NUMINAMATH_CALUDE_school_club_revenue_l652_65284


namespace NUMINAMATH_CALUDE_cat_eye_movement_l652_65269

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the transformation (moving 3 units to the right)
def moveRight (p : Point) : Point :=
  (p.1 + 3, p.2)

-- Define the original points
def eye1 : Point := (-4, 3)
def eye2 : Point := (-2, 3)

-- State the theorem
theorem cat_eye_movement :
  (moveRight eye1 = (-1, 3)) ∧ (moveRight eye2 = (1, 3)) := by
  sorry

end NUMINAMATH_CALUDE_cat_eye_movement_l652_65269


namespace NUMINAMATH_CALUDE_bells_lcm_l652_65228

/-- The time intervals at which the bells toll -/
def bell_intervals : List ℕ := [5, 8, 11, 15, 20]

/-- The theorem stating that the least common multiple of the bell intervals is 1320 -/
theorem bells_lcm : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 5 8) 11) 15) 20 = 1320 := by
  sorry

end NUMINAMATH_CALUDE_bells_lcm_l652_65228


namespace NUMINAMATH_CALUDE_tutors_next_common_workday_l652_65220

def tim_schedule : ℕ := 5
def uma_schedule : ℕ := 6
def victor_schedule : ℕ := 9
def xavier_schedule : ℕ := 8

theorem tutors_next_common_workday : 
  lcm (lcm (lcm tim_schedule uma_schedule) victor_schedule) xavier_schedule = 360 := by
  sorry

end NUMINAMATH_CALUDE_tutors_next_common_workday_l652_65220


namespace NUMINAMATH_CALUDE_milburg_children_l652_65206

/-- The number of children in Milburg -/
def children : ℕ := 8243 - 5256

/-- The total population of Milburg -/
def total_population : ℕ := 8243

/-- The number of grown-ups in Milburg -/
def grown_ups : ℕ := 5256

theorem milburg_children : children = 2987 := by
  sorry

end NUMINAMATH_CALUDE_milburg_children_l652_65206


namespace NUMINAMATH_CALUDE_flight_departure_requirement_l652_65245

/-- The minimum number of people required for the flight to depart -/
def min_required : ℕ := 16

/-- The number of people currently on the plane -/
def current_people : ℕ := 9

/-- The number of additional people needed to board before departure -/
def additional_people : ℕ := min_required - current_people

theorem flight_departure_requirement :
  min_required > 15 ∧ current_people = 9 → additional_people = 7 := by
  sorry

end NUMINAMATH_CALUDE_flight_departure_requirement_l652_65245


namespace NUMINAMATH_CALUDE_nested_diamond_result_l652_65260

/-- Diamond operation for real numbers -/
def diamond (a b : ℝ) : ℝ := a^2 + b^2 - a * b

/-- Theorem stating the result of the nested diamond operations -/
theorem nested_diamond_result :
  diamond (diamond 3 8) (diamond 8 (-3)) = 7057 := by
  sorry

end NUMINAMATH_CALUDE_nested_diamond_result_l652_65260


namespace NUMINAMATH_CALUDE_drama_club_two_skills_l652_65297

/-- Represents the number of students with a particular combination of skills -/
structure SkillCount where
  write : Nat
  direct : Nat
  produce : Nat
  write_direct : Nat
  write_produce : Nat
  direct_produce : Nat

/-- Represents the constraints of the drama club problem -/
def drama_club_constraints (sc : SkillCount) : Prop :=
  sc.write + sc.direct + sc.produce + sc.write_direct + sc.write_produce + sc.direct_produce = 150 ∧
  sc.write + sc.write_direct + sc.write_produce = 90 ∧
  sc.direct + sc.write_direct + sc.direct_produce = 60 ∧
  sc.produce + sc.write_produce + sc.direct_produce = 110

/-- The main theorem stating that under the given constraints, 
    the number of students with exactly two skills is 110 -/
theorem drama_club_two_skills (sc : SkillCount) 
  (h : drama_club_constraints sc) : 
  sc.write_direct + sc.write_produce + sc.direct_produce = 110 := by
  sorry

end NUMINAMATH_CALUDE_drama_club_two_skills_l652_65297


namespace NUMINAMATH_CALUDE_unique_three_digit_cube_sum_l652_65293

def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem unique_three_digit_cube_sum : ∃! n : ℕ, 
  is_three_digit_number n ∧ n = (digit_sum n)^3 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_cube_sum_l652_65293


namespace NUMINAMATH_CALUDE_multiple_of_smaller_number_l652_65231

theorem multiple_of_smaller_number (S L k : ℤ) : 
  S = 14 → 
  L = k * S - 3 → 
  S + L = 39 → 
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_smaller_number_l652_65231


namespace NUMINAMATH_CALUDE_fixed_point_of_function_l652_65214

theorem fixed_point_of_function (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ (a - 1) * 2^x - 2*a
  f 1 = -2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_function_l652_65214


namespace NUMINAMATH_CALUDE_hotdogs_sold_l652_65263

theorem hotdogs_sold (initial : ℕ) (final : ℕ) (h1 : initial = 99) (h2 : final = 97) :
  initial - final = 2 := by
  sorry

end NUMINAMATH_CALUDE_hotdogs_sold_l652_65263


namespace NUMINAMATH_CALUDE_harry_initial_bid_was_500_l652_65261

/-- Represents the auction scenario with given conditions --/
structure Auction where
  startingBid : ℕ
  harryFirstBid : ℕ
  harryFinalBid : ℕ
  finalBidDifference : ℕ

/-- Calculates the second bidder's bid --/
def secondBid (a : Auction) : ℕ := a.startingBid + 2 * a.harryFirstBid

/-- Calculates the third bidder's bid --/
def thirdBid (a : Auction) : ℕ := a.startingBid + 5 * a.harryFirstBid

/-- Theorem stating that Harry's initial bid increment was $500 --/
theorem harry_initial_bid_was_500 (a : Auction) 
  (h1 : a.startingBid = 300)
  (h2 : a.harryFinalBid = 4000)
  (h3 : a.finalBidDifference = 1500)
  (h4 : a.harryFinalBid = thirdBid a + a.finalBidDifference) :
  a.harryFirstBid = 500 := by
  sorry


end NUMINAMATH_CALUDE_harry_initial_bid_was_500_l652_65261


namespace NUMINAMATH_CALUDE_negation_equivalence_l652_65250

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 2*x + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2*x + 5 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l652_65250


namespace NUMINAMATH_CALUDE_james_tree_problem_l652_65291

/-- Represents the number of trees James initially has -/
def initial_trees : ℕ := 2

/-- Represents the percentage of seeds planted -/
def planting_rate : ℚ := 60 / 100

/-- Represents the number of new trees planted -/
def new_trees : ℕ := 24

/-- Represents the number of plants per tree -/
def plants_per_tree : ℕ := 20

theorem james_tree_problem :
  plants_per_tree * initial_trees * planting_rate = new_trees :=
sorry

end NUMINAMATH_CALUDE_james_tree_problem_l652_65291


namespace NUMINAMATH_CALUDE_supplement_complement_difference_l652_65266

/-- An acute angle is between 0° and 90° -/
def is_acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < 90

/-- The supplement of an angle θ -/
def supplement (θ : ℝ) : ℝ := 180 - θ

/-- The complement of an angle θ -/
def complement (θ : ℝ) : ℝ := 90 - θ

/-- For any acute angle, the difference between its supplement and complement is 90° -/
theorem supplement_complement_difference (θ : ℝ) (h : is_acute_angle θ) :
  supplement θ - complement θ = 90 := by
  sorry

end NUMINAMATH_CALUDE_supplement_complement_difference_l652_65266


namespace NUMINAMATH_CALUDE_intersection_M_N_l652_65244

def M : Set ℝ := {x | x ≤ 0}
def N : Set ℝ := {-2, 0, 1}

theorem intersection_M_N : M ∩ N = {-2, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l652_65244


namespace NUMINAMATH_CALUDE_real_solutions_quadratic_l652_65259

theorem real_solutions_quadratic (x : ℝ) :
  (∃ y : ℝ, 9 * y^2 + 6 * x * y + 2 * x + 1 = 0) ↔ (x < 2 - Real.sqrt 6 ∨ x > 2 + Real.sqrt 6) :=
sorry

end NUMINAMATH_CALUDE_real_solutions_quadratic_l652_65259


namespace NUMINAMATH_CALUDE_unique_equidistant_point_l652_65288

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  center : Point3D
  sideLength : ℝ

/-- Checks if a point is inside or on the diagonal face BDD₁B₁ of the cube -/
def isOnDiagonalFace (c : Cube) (p : Point3D) : Prop := sorry

/-- Calculates the distance from a point to a plane -/
def distToPlane (p : Point3D) (plane : Point3D → Prop) : ℝ := sorry

/-- The plane ABC of the cube -/
def planeABC (c : Cube) : Point3D → Prop := sorry

/-- The plane ABA₁ of the cube -/
def planeABA1 (c : Cube) : Point3D → Prop := sorry

/-- The plane ADA₁ of the cube -/
def planeADA1 (c : Cube) : Point3D → Prop := sorry

theorem unique_equidistant_point (c : Cube) : 
  ∃! p : Point3D, 
    isOnDiagonalFace c p ∧ 
    distToPlane p (planeABC c) = distToPlane p (planeABA1 c) ∧
    distToPlane p (planeABC c) = distToPlane p (planeADA1 c) :=
sorry

end NUMINAMATH_CALUDE_unique_equidistant_point_l652_65288


namespace NUMINAMATH_CALUDE_max_cross_section_area_l652_65273

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a triangular prism -/
structure TriangularPrism where
  base : List Point3D
  heights : List ℝ

def crossSectionArea (prism : TriangularPrism) (plane : Plane) : ℝ := sorry

/-- The main theorem statement -/
theorem max_cross_section_area :
  let prism : TriangularPrism := {
    base := [
      { x := 4, y := 0, z := 0 },
      { x := -2, y := 2 * Real.sqrt 3, z := 0 },
      { x := -2, y := -2 * Real.sqrt 3, z := 0 }
    ],
    heights := [2, 4, 3]
  }
  let plane : Plane := { a := 5, b := -3, c := 2, d := 30 }
  let area := crossSectionArea prism plane
  ∃ ε > 0, abs (area - 104.25) < ε := by
  sorry

end NUMINAMATH_CALUDE_max_cross_section_area_l652_65273


namespace NUMINAMATH_CALUDE_number_game_l652_65242

theorem number_game (x : ℤ) : 3 * (3 * (x + 3) - 3) = 3 * (3 * x) := by
  sorry

end NUMINAMATH_CALUDE_number_game_l652_65242


namespace NUMINAMATH_CALUDE_solution_set_g_range_of_a_l652_65294

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - a| + |2*x + 5|
def g (x : ℝ) : ℝ := |x - 1| - |2*x|

-- Theorem for part I
theorem solution_set_g (x : ℝ) : g x > -4 ↔ -5 < x ∧ x < -3 := by sorry

-- Theorem for part II
theorem range_of_a (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, f a x₁ = g x₂) → -6 ≤ a ∧ a ≤ -4 := by sorry

end NUMINAMATH_CALUDE_solution_set_g_range_of_a_l652_65294


namespace NUMINAMATH_CALUDE_library_books_count_l652_65283

/-- Given the conditions of the library bookshelves, calculate the total number of books -/
theorem library_books_count (num_shelves : ℕ) (floors_per_shelf : ℕ) (books_after_removal : ℕ) : 
  num_shelves = 28 → 
  floors_per_shelf = 6 → 
  books_after_removal = 20 → 
  (num_shelves * floors_per_shelf * (books_after_removal + 2) = 3696) :=
by
  sorry

#check library_books_count

end NUMINAMATH_CALUDE_library_books_count_l652_65283


namespace NUMINAMATH_CALUDE_specific_quilt_shaded_fraction_l652_65234

/-- Represents a square quilt composed of unit squares -/
structure Quilt :=
  (total_squares : ℕ)
  (whole_squares : ℕ)
  (half_shaded_squares : ℕ)
  (fully_half_shaded_squares : ℕ)

/-- Calculates the shaded fraction of a quilt -/
def shaded_fraction (q : Quilt) : ℚ :=
  let shaded_area := (q.half_shaded_squares : ℚ) / 2 + (q.fully_half_shaded_squares : ℚ) / 2
  shaded_area / q.total_squares

/-- Theorem stating that for a specific quilt configuration, the shaded fraction is 2.5/9 -/
theorem specific_quilt_shaded_fraction :
  let q := Quilt.mk 9 4 1 4
  shaded_fraction q = 5/18 := by sorry

end NUMINAMATH_CALUDE_specific_quilt_shaded_fraction_l652_65234


namespace NUMINAMATH_CALUDE_square_sectors_semicircle_differences_l652_65279

/-- Given a square with side length 300 cm, containing two right-angle sectors and a semicircle,
    prove the difference in area between the two shaded regions and the difference in their perimeters. -/
theorem square_sectors_semicircle_differences (π : ℝ) (h_π : π = 3.14) :
  let square_side : ℝ := 300
  let quarter_circle_area : ℝ := 1/4 * π * square_side^2
  let semicircle_area : ℝ := 1/2 * π * (square_side/2)^2
  let quarter_circle_perimeter : ℝ := 1/2 * π * square_side
  let semicircle_perimeter : ℝ := π * square_side/2 + square_side
  let area_difference : ℝ := 2 * quarter_circle_area - square_side^2 - semicircle_area
  let perimeter_difference : ℝ := 2 * quarter_circle_perimeter - semicircle_perimeter
  area_difference = 15975 ∧ perimeter_difference = 485 :=
by sorry

end NUMINAMATH_CALUDE_square_sectors_semicircle_differences_l652_65279


namespace NUMINAMATH_CALUDE_composite_divisibility_l652_65271

theorem composite_divisibility (k : ℕ) (p_k : ℕ) (n : ℕ) 
  (h1 : k ≥ 14)
  (h2 : Nat.Prime p_k)
  (h3 : p_k < k)
  (h4 : ∀ p, Nat.Prime p → p < k → p ≤ p_k)
  (h5 : p_k ≥ 3 * k / 4)
  (h6 : ¬ Nat.Prime n) :
  (n = 2 * p_k → ¬ (n ∣ Nat.factorial (n - k))) ∧
  (n > 2 * p_k → n ∣ Nat.factorial (n - k)) := by
  sorry

end NUMINAMATH_CALUDE_composite_divisibility_l652_65271


namespace NUMINAMATH_CALUDE_total_maggots_is_twenty_l652_65213

/-- The number of maggots served in the first attempt -/
def first_attempt : ℕ := 10

/-- The number of maggots served in the second attempt -/
def second_attempt : ℕ := 10

/-- The total number of maggots served -/
def total_maggots : ℕ := first_attempt + second_attempt

theorem total_maggots_is_twenty : total_maggots = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_maggots_is_twenty_l652_65213


namespace NUMINAMATH_CALUDE_identical_differences_l652_65290

theorem identical_differences (a : Fin 20 → ℕ) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) 
  (h_bound : ∀ i, a i < 70) : 
  ∃ (d : ℕ) (i j k l : Fin 19), 
    i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
    a (i.succ) - a i = d ∧ 
    a (j.succ) - a j = d ∧ 
    a (k.succ) - a k = d ∧ 
    a (l.succ) - a l = d :=
sorry

end NUMINAMATH_CALUDE_identical_differences_l652_65290


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l652_65292

-- Problem 1
theorem problem_1 : 
  (2 + 1/4)^(1/2) - (-0.96)^0 - (3 + 3/8)^(-2/3) + (3/2)^(-2) = 1/2 := by sorry

-- Problem 2
theorem problem_2 (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) : 
  (2 * (a^2)^(1/3) * b^(1/2)) * (-6 * a^(1/2) * b^(1/3)) / (-3 * a^(1/6) * b^(5/6)) = 4*a := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l652_65292


namespace NUMINAMATH_CALUDE_triangle_side_length_l652_65212

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively
  (A + B + C = π) →
  -- Condition: √3 sin A + cos A = 2
  (Real.sqrt 3 * Real.sin A + Real.cos A = 2) →
  -- Condition: a = 3
  (a = 3) →
  -- Condition: C = 5π/12
  (C = 5 * π / 12) →
  -- Conclusion: b = √6
  (b = Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l652_65212


namespace NUMINAMATH_CALUDE_integer_fraction_pairs_l652_65254

theorem integer_fraction_pairs : 
  {p : ℕ × ℕ | p.1 > 0 ∧ p.2 > 0 ∧ (∃ k : ℤ, (p.2^3 + 1 : ℤ) = k * (p.1 * p.2 - 1))} = 
  {(2,1), (3,1), (2,2), (5,2), (5,3), (2,5), (3,5)} := by
sorry

end NUMINAMATH_CALUDE_integer_fraction_pairs_l652_65254


namespace NUMINAMATH_CALUDE_calculate_expression_l652_65223

theorem calculate_expression : (1000 * 0.09999) / 10 * 999 = 998001 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l652_65223


namespace NUMINAMATH_CALUDE_solve_for_b_l652_65289

/-- Given two functions p and q, prove that if p(q(3)) = 31, then b has two specific values. -/
theorem solve_for_b (p q : ℝ → ℝ) (b : ℝ) 
  (hp : ∀ x, p x = 2 * x^2 - 7)
  (hq : ∀ x, q x = 4 * x - b)
  (h_pq3 : p (q 3) = 31) :
  b = 12 + Real.sqrt 19 ∨ b = 12 - Real.sqrt 19 := by
  sorry

#check solve_for_b

end NUMINAMATH_CALUDE_solve_for_b_l652_65289


namespace NUMINAMATH_CALUDE_vershoks_in_arshin_l652_65219

/-- The number of vershoks in one arshin -/
def vershoks_per_arshin : ℕ := sorry

/-- Length of a plank in arshins -/
def plank_length : ℕ := 6

/-- Width of a plank in vershoks -/
def plank_width : ℕ := 6

/-- Side length of the room in arshins -/
def room_side : ℕ := 12

/-- Number of planks needed to cover the floor -/
def num_planks : ℕ := 64

theorem vershoks_in_arshin : 
  vershoks_per_arshin = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_vershoks_in_arshin_l652_65219


namespace NUMINAMATH_CALUDE_min_value_theorem_l652_65295

theorem min_value_theorem (a : ℝ) (x₁ x₂ : ℝ) 
  (h_a : a > 0)
  (h_solution : ∀ x, x^2 - 4*a*x + 3*a^2 < 0 ↔ x ∈ Set.Ioo x₁ x₂) :
  ∀ y, x₁ + x₂ + a / (x₁ * x₂) ≥ y → y ≤ 4 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l652_65295


namespace NUMINAMATH_CALUDE_motel_payment_savings_l652_65256

/-- Calculates the savings when choosing monthly payments over weekly payments for a motel stay. -/
theorem motel_payment_savings 
  (weeks_per_month : ℕ) 
  (total_months : ℕ) 
  (weekly_rate : ℕ) 
  (monthly_rate : ℕ) 
  (h1 : weeks_per_month = 4) 
  (h2 : total_months = 3) 
  (h3 : weekly_rate = 280) 
  (h4 : monthly_rate = 1000) : 
  (total_months * weeks_per_month * weekly_rate) - (total_months * monthly_rate) = 360 := by
  sorry

#check motel_payment_savings

end NUMINAMATH_CALUDE_motel_payment_savings_l652_65256


namespace NUMINAMATH_CALUDE_arithmetic_progression_properties_l652_65211

/-- An arithmetic progression with given first and tenth terms -/
def ArithmeticProgression (a : ℕ → ℤ) : Prop :=
  a 1 = 21 ∧ a 10 = 3 ∧ ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_progression_properties (a : ℕ → ℤ) (h : ArithmeticProgression a) :
  (∀ n : ℕ, a n = -2 * n + 23) ∧
  (Finset.sum (Finset.range 11) a = 121) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_properties_l652_65211


namespace NUMINAMATH_CALUDE_win_sector_area_l652_65241

theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 7) (h2 : p = 3/8) :
  p * π * r^2 = 147 * π / 8 := by
  sorry

end NUMINAMATH_CALUDE_win_sector_area_l652_65241


namespace NUMINAMATH_CALUDE_symmetric_axis_of_shifted_function_l652_65230

/-- Given a function f(x) = √3 * sin(2x) - cos(2x), prove that when shifted right by π/3 units,
    one of its symmetric axes is given by the equation x = π/6 -/
theorem symmetric_axis_of_shifted_function :
  ∃ (f : ℝ → ℝ) (g : ℝ → ℝ),
    (∀ x, f x = Real.sqrt 3 * Real.sin (2 * x) - Real.cos (2 * x)) ∧
    (∀ x, g x = f (x - π / 3)) ∧
    (∀ x, g x = g (π / 3 - x)) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_axis_of_shifted_function_l652_65230


namespace NUMINAMATH_CALUDE_correct_bucket_size_l652_65252

/-- The size of the bucket needed to collect leaking fluid -/
def bucket_size (leak_rate : ℝ) (max_time : ℝ) : ℝ :=
  2 * leak_rate * max_time

/-- Theorem stating the correct bucket size for the given conditions -/
theorem correct_bucket_size :
  bucket_size 1.5 12 = 36 := by
  sorry

end NUMINAMATH_CALUDE_correct_bucket_size_l652_65252


namespace NUMINAMATH_CALUDE_g_2010_value_l652_65262

-- Define the property of the function g
def g_property (g : ℕ → ℝ) : Prop :=
  ∀ x y m : ℕ, x > 0 → y > 0 → m > 0 → x + y = 2^m → g x + g y = ((m + 1) : ℝ)^2

-- Theorem statement
theorem g_2010_value (g : ℕ → ℝ) (h : g_property g) : g 2010 = 126 := by
  sorry

end NUMINAMATH_CALUDE_g_2010_value_l652_65262


namespace NUMINAMATH_CALUDE_handshake_theorem_l652_65276

def num_people : ℕ := 8

def handshake_arrangements (n : ℕ) : ℕ :=
  if n = 0 then 1
  else (n - 1) * handshake_arrangements (n - 2)

theorem handshake_theorem :
  handshake_arrangements num_people = 105 :=
by sorry

end NUMINAMATH_CALUDE_handshake_theorem_l652_65276


namespace NUMINAMATH_CALUDE_existence_of_x_y_l652_65200

theorem existence_of_x_y : ∃ (x y : ℝ), 3*x + y > 0 ∧ 4*x + y > 0 ∧ 6*x + 5*y < 0 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_x_y_l652_65200


namespace NUMINAMATH_CALUDE_initial_tree_height_l652_65251

/-- Represents the growth of a tree over time -/
def TreeGrowth (initial_height growth_rate years final_height : ℝ) : Prop :=
  initial_height + growth_rate * years = final_height

theorem initial_tree_height : 
  ∃ (h : ℝ), TreeGrowth h 5 5 29 ∧ h = 4 := by sorry

end NUMINAMATH_CALUDE_initial_tree_height_l652_65251


namespace NUMINAMATH_CALUDE_janet_sock_purchase_l652_65226

theorem janet_sock_purchase : 
  ∀ (x y z : ℕ),
  -- Total number of pairs
  x + y + z = 18 →
  -- Total cost
  2*x + 5*y + 7*z = 60 →
  -- Exactly 3 pairs of $7 socks
  z = 3 →
  -- x represents the number of $2 socks
  x = 12 := by
sorry

end NUMINAMATH_CALUDE_janet_sock_purchase_l652_65226


namespace NUMINAMATH_CALUDE_coloring_book_shelves_l652_65257

theorem coloring_book_shelves (initial_stock : ℕ) (books_sold : ℕ) (books_per_shelf : ℕ) : 
  initial_stock = 87 → books_sold = 33 → books_per_shelf = 6 → 
  (initial_stock - books_sold) / books_per_shelf = 9 := by
sorry

end NUMINAMATH_CALUDE_coloring_book_shelves_l652_65257


namespace NUMINAMATH_CALUDE_triangle_existence_condition_l652_65209

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the conditions
def angle_ABC (t : Triangle) : ℝ := sorry
def side_AC (t : Triangle) : ℝ := sorry
def side_BC (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem triangle_existence_condition (t : Triangle) (k : ℝ) :
  (∃! t, angle_ABC t = π/3 ∧ side_AC t = 12 ∧ side_BC t = k) ↔
  (0 < k ∧ k ≤ 12) ∨ k = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_existence_condition_l652_65209


namespace NUMINAMATH_CALUDE_statement_1_statement_3_statement_4_l652_65248

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (skew_lines : Line → Line → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- Define the planes and lines
variable (a b : Plane)
variable (l m n : Line)

-- Define the non-coincidence conditions
variable (planes_non_coincident : a ≠ b)
variable (lines_non_coincident : l ≠ m ∧ m ≠ n ∧ l ≠ n)

-- Theorem for statement 1
theorem statement_1 :
  parallel_planes a b →
  line_in_plane l a →
  parallel_line_plane l b :=
sorry

-- Theorem for statement 3
theorem statement_3 :
  parallel_line_plane l a →
  perpendicular_line_plane l b →
  perpendicular_planes a b :=
sorry

-- Theorem for statement 4
theorem statement_4 :
  skew_lines m n →
  parallel_line_plane m a →
  parallel_line_plane n a →
  perpendicular_lines l m →
  perpendicular_lines l n →
  perpendicular_line_plane l a :=
sorry

end NUMINAMATH_CALUDE_statement_1_statement_3_statement_4_l652_65248


namespace NUMINAMATH_CALUDE_complex_equation_solution_l652_65224

theorem complex_equation_solution :
  ∃ z : ℂ, (z - Complex.I) * Complex.I = 2 + Complex.I ∧ z = 1 - Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l652_65224


namespace NUMINAMATH_CALUDE_hash_solution_l652_65265

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem hash_solution :
  ∃ (A : ℝ), (hash A 7 = 225) ∧ (A = 7 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_hash_solution_l652_65265


namespace NUMINAMATH_CALUDE_boat_current_rate_l652_65227

/-- Proves that the rate of the current is 5 km/hr given the conditions of the boat problem -/
theorem boat_current_rate 
  (boat_speed : ℝ) 
  (distance : ℝ) 
  (time_minutes : ℝ) 
  (h1 : boat_speed = 20) 
  (h2 : distance = 11.25) 
  (h3 : time_minutes = 27) : 
  ∃ current_rate : ℝ, 
    current_rate = 5 ∧ 
    distance = (boat_speed + current_rate) * (time_minutes / 60) :=
by
  sorry


end NUMINAMATH_CALUDE_boat_current_rate_l652_65227


namespace NUMINAMATH_CALUDE_least_n_satisfying_inequality_five_satisfies_inequality_least_n_is_five_l652_65274

theorem least_n_satisfying_inequality :
  ∀ n : ℕ+, n < 5 → (1 : ℚ) / n.val - (1 : ℚ) / (n.val + 2) ≥ (1 : ℚ) / 15 :=
by sorry

theorem five_satisfies_inequality :
  (1 : ℚ) / 5 - (1 : ℚ) / 7 < (1 : ℚ) / 15 :=
by sorry

theorem least_n_is_five :
  ∃! (n : ℕ+), 
    ((1 : ℚ) / n.val - (1 : ℚ) / (n.val + 2) < (1 : ℚ) / 15) ∧
    (∀ m : ℕ+, m < n → (1 : ℚ) / m.val - (1 : ℚ) / (m.val + 2) ≥ (1 : ℚ) / 15) :=
by sorry

end NUMINAMATH_CALUDE_least_n_satisfying_inequality_five_satisfies_inequality_least_n_is_five_l652_65274


namespace NUMINAMATH_CALUDE_insulation_cost_for_given_tank_l652_65281

/-- Calculates the surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ :=
  2 * (l * w + l * h + w * h)

/-- Calculates the cost of insulating a rectangular tank -/
def insulation_cost (l w h cost_per_sqft : ℝ) : ℝ :=
  surface_area l w h * cost_per_sqft

/-- Theorem: The cost of insulating a rectangular tank with given dimensions -/
theorem insulation_cost_for_given_tank :
  insulation_cost 7 3 2 20 = 1640 := by
  sorry

end NUMINAMATH_CALUDE_insulation_cost_for_given_tank_l652_65281


namespace NUMINAMATH_CALUDE_all_numbers_on_diagonal_l652_65255

/-- Represents a 15x15 table with numbers 1 to 15 -/
def Table := Fin 15 → Fin 15 → Fin 15

/-- The property that each number appears exactly once in each row -/
def row_property (t : Table) : Prop :=
  ∀ i j₁ j₂, j₁ ≠ j₂ → t i j₁ ≠ t i j₂

/-- The property that each number appears exactly once in each column -/
def column_property (t : Table) : Prop :=
  ∀ i₁ i₂ j, i₁ ≠ i₂ → t i₁ j ≠ t i₂ j

/-- The property that symmetrically placed numbers are identical -/
def symmetry_property (t : Table) : Prop :=
  ∀ i j, t i j = t j i

/-- The main theorem stating that all numbers appear on the main diagonal -/
theorem all_numbers_on_diagonal (t : Table)
  (h_row : row_property t)
  (h_col : column_property t)
  (h_sym : symmetry_property t) :
  ∀ n : Fin 15, ∃ i : Fin 15, t i i = n :=
sorry

end NUMINAMATH_CALUDE_all_numbers_on_diagonal_l652_65255


namespace NUMINAMATH_CALUDE_father_age_twice_marika_l652_65208

/-- Marika's birth year -/
def marika_birth_year : ℕ := 1996

/-- Marika's father's birth year -/
def father_birth_year : ℕ := 1956

/-- The year when the father's age is twice Marika's age -/
def target_year : ℕ := 2036

theorem father_age_twice_marika (year : ℕ) :
  year = target_year ↔ 
  (year - father_birth_year = 2 * (year - marika_birth_year)) ∧
  (year > marika_birth_year) ∧
  (year > father_birth_year) := by
sorry

end NUMINAMATH_CALUDE_father_age_twice_marika_l652_65208


namespace NUMINAMATH_CALUDE_curve_touches_x_axis_and_area_l652_65264

noncomputable def curve (a : ℝ) (t : ℝ) : ℝ × ℝ :=
  (t + Real.exp (a * t), -t + Real.exp (a * t))

theorem curve_touches_x_axis_and_area (a : ℝ) (h : a > 0) :
  (∃ t : ℝ, (curve a t).2 = 0 ∧ 
    (∀ s : ℝ, s ≠ t → (curve a s).2 ≠ 0 ∨ (curve a s).2 < 0)) →
  a = 1 / Real.exp 1 ∧
  (∫ t in (0)..(Real.exp 1), (curve a t).2 - min (curve a t).1 (curve a t).2) = Real.exp 2 / 2 - Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_curve_touches_x_axis_and_area_l652_65264


namespace NUMINAMATH_CALUDE_toy_purchase_cost_l652_65267

theorem toy_purchase_cost (yoyo_cost whistle_cost : ℕ) 
  (h1 : yoyo_cost = 24) 
  (h2 : whistle_cost = 14) : 
  yoyo_cost + whistle_cost = 38 := by
  sorry

end NUMINAMATH_CALUDE_toy_purchase_cost_l652_65267


namespace NUMINAMATH_CALUDE_parabola_vertex_l652_65253

/-- The parabola defined by y = (x-1)^2 + 2 -/
def parabola (x : ℝ) : ℝ := (x - 1)^2 + 2

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, 2)

/-- Theorem: The vertex of the parabola y = (x-1)^2 + 2 has coordinates (1, 2) -/
theorem parabola_vertex : 
  ∀ (x : ℝ), parabola x ≥ parabola (vertex.1) ∧ parabola (vertex.1) = vertex.2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l652_65253


namespace NUMINAMATH_CALUDE_coffee_maker_price_l652_65217

/-- The final price of a coffee maker after applying a discount -/
def final_price (original_price discount : ℕ) : ℕ :=
  original_price - discount

/-- Theorem: The customer pays $70 for a coffee maker with original price $90 and a $20 discount -/
theorem coffee_maker_price :
  final_price 90 20 = 70 := by
  sorry

end NUMINAMATH_CALUDE_coffee_maker_price_l652_65217


namespace NUMINAMATH_CALUDE_treasure_chest_gems_l652_65243

theorem treasure_chest_gems (total_gems rubies : ℕ) 
  (h1 : total_gems = 5155)
  (h2 : rubies = 5110)
  (h3 : total_gems ≥ rubies) :
  total_gems - rubies = 45 := by
  sorry

end NUMINAMATH_CALUDE_treasure_chest_gems_l652_65243


namespace NUMINAMATH_CALUDE_solution_sum_l652_65233

-- Define the solution set for |2x-3| ≤ 1
def solution_set (m n : ℝ) : Prop :=
  ∀ x, |2*x - 3| ≤ 1 ↔ m ≤ x ∧ x ≤ n

-- Theorem statement
theorem solution_sum (m n : ℝ) : solution_set m n → m + n = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_sum_l652_65233


namespace NUMINAMATH_CALUDE_marksman_hit_rate_l652_65229

theorem marksman_hit_rate (p : ℝ) : 
  (0 ≤ p ∧ p ≤ 1) →  -- p is a probability
  (1 - (1 - p)^4 = 80/81) →  -- probability of hitting at least once in 4 shots
  p = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_marksman_hit_rate_l652_65229


namespace NUMINAMATH_CALUDE_range_of_a_l652_65286

def A : Set ℝ := {x : ℝ | x^2 + 4*x = 0}

def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*(a-1)*x + a^2 - 1 = 0}

theorem range_of_a (a : ℝ) : (A ∩ B a = B a) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l652_65286


namespace NUMINAMATH_CALUDE_wendy_extraction_cost_l652_65249

/-- The cost of a dental cleaning in dollars -/
def cleaning_cost : ℕ := 70

/-- The cost of a dental filling in dollars -/
def filling_cost : ℕ := 120

/-- The number of fillings Wendy had -/
def num_fillings : ℕ := 2

/-- The total cost of Wendy's dental bill in dollars -/
def total_bill : ℕ := 5 * filling_cost

/-- The cost of Wendy's tooth extraction in dollars -/
def extraction_cost : ℕ := total_bill - (cleaning_cost + num_fillings * filling_cost)

theorem wendy_extraction_cost : extraction_cost = 290 := by
  sorry

end NUMINAMATH_CALUDE_wendy_extraction_cost_l652_65249


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l652_65298

theorem arithmetic_sequence_sum : ∀ (a₁ a_last d n : ℕ),
  a₁ = 1 →
  a_last = 23 →
  d = 2 →
  n = (a_last - a₁) / d + 1 →
  (n : ℝ) * (a₁ + a_last) / 2 = 144 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l652_65298


namespace NUMINAMATH_CALUDE_binary_of_34_l652_65296

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- Theorem: The binary representation of 34 is 100010 -/
theorem binary_of_34 : toBinary 34 = [false, true, false, false, false, true] := by
  sorry

end NUMINAMATH_CALUDE_binary_of_34_l652_65296


namespace NUMINAMATH_CALUDE_triangle_abc_is_obtuse_l652_65222

theorem triangle_abc_is_obtuse (A B C : ℝ) (h1 : A = 2 * B) (h2 : A = 3 * C) 
  (h3 : A + B + C = 180) : A > 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_is_obtuse_l652_65222


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l652_65247

theorem fraction_to_decimal : (5 : ℚ) / 8 = 0.625 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l652_65247


namespace NUMINAMATH_CALUDE_complex_equation_solutions_l652_65225

theorem complex_equation_solutions :
  let f : ℂ → ℂ := λ z => (z^3 + 2*z^2 + z - 2) / (z^2 - 3*z + 2)
  ∃! (s : Finset ℂ), s.card = 2 ∧ ∀ z ∈ s, f z = 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l652_65225


namespace NUMINAMATH_CALUDE_equation_solution_l652_65299

theorem equation_solution : ∃ x : ℚ, (1 / 4 : ℚ) + 1 / x = 7 / 8 ∧ x = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l652_65299


namespace NUMINAMATH_CALUDE_coke_drinking_days_l652_65201

/-- Calculates the remaining days to finish drinking Coke -/
def remaining_days (total_volume : ℕ) (daily_consumption : ℕ) (days_consumed : ℕ) : ℕ :=
  (total_volume * 1000 / daily_consumption) - days_consumed

/-- Proves that it takes 7 more days to finish the Coke -/
theorem coke_drinking_days : remaining_days 2 200 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_coke_drinking_days_l652_65201


namespace NUMINAMATH_CALUDE_stratified_sampling_group_c_l652_65239

/-- Represents the number of cities to be selected from a group in a stratified sampling -/
def citiesSelected (totalSelected : ℕ) (totalCities : ℕ) (groupCities : ℕ) : ℕ :=
  (totalSelected * groupCities) / totalCities

theorem stratified_sampling_group_c (totalCities : ℕ) (groupACities : ℕ) (groupBCities : ℕ) 
    (totalSelected : ℕ) (hTotal : totalCities = 48) (hA : groupACities = 8) (hB : groupBCities = 24) 
    (hSelected : totalSelected = 12) :
    citiesSelected totalSelected totalCities (totalCities - groupACities - groupBCities) = 4 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_group_c_l652_65239


namespace NUMINAMATH_CALUDE_equation1_solutions_equation2_solution_l652_65287

-- Define the equations
def equation1 (x : ℝ) : Prop := (x - 1)^2 - 1 = 15
def equation2 (x : ℝ) : Prop := (1/3) * (x + 3)^3 - 9 = 0

-- Theorem for equation 1
theorem equation1_solutions : 
  (∃ x : ℝ, equation1 x) ↔ (equation1 5 ∧ equation1 (-3)) :=
sorry

-- Theorem for equation 2
theorem equation2_solution : 
  (∃ x : ℝ, equation2 x) ↔ equation2 0 :=
sorry

end NUMINAMATH_CALUDE_equation1_solutions_equation2_solution_l652_65287


namespace NUMINAMATH_CALUDE_bella_stamps_count_l652_65282

/-- The number of snowflake stamps Bella bought -/
def snowflake_stamps : ℕ := 11

/-- The number of truck stamps Bella bought -/
def truck_stamps : ℕ := snowflake_stamps + 9

/-- The number of rose stamps Bella bought -/
def rose_stamps : ℕ := truck_stamps - 13

/-- The total number of stamps Bella bought -/
def total_stamps : ℕ := snowflake_stamps + truck_stamps + rose_stamps

theorem bella_stamps_count : total_stamps = 38 := by
  sorry

end NUMINAMATH_CALUDE_bella_stamps_count_l652_65282


namespace NUMINAMATH_CALUDE_probability_of_specific_draw_l652_65202

/-- The number of blue balls in the bag -/
def blue_balls : ℕ := 8

/-- The number of green balls in the bag -/
def green_balls : ℕ := 7

/-- The number of blue balls to be drawn -/
def blue_draw : ℕ := 3

/-- The number of green balls to be drawn -/
def green_draw : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := blue_balls + green_balls

/-- The total number of balls to be drawn -/
def total_draw : ℕ := blue_draw + green_draw

/-- The probability of drawing 3 blue balls followed by 2 green balls without replacement -/
theorem probability_of_specific_draw :
  (Nat.choose blue_balls blue_draw * Nat.choose green_balls green_draw : ℚ) /
  (Nat.choose total_balls total_draw : ℚ) = 1176 / 3003 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_specific_draw_l652_65202


namespace NUMINAMATH_CALUDE_unique_n_for_consecutive_product_l652_65240

theorem unique_n_for_consecutive_product : ∃! (n : ℕ), 
  n > 0 ∧ ∃ (k : ℕ), k > 0 ∧ 
  (n^6 + 5*n^3 + 4*n + 116 = k * (k + 1) ∨ 
   n^6 + 5*n^3 + 4*n + 116 = k * (k + 1) * (k + 2) ∨
   n^6 + 5*n^3 + 4*n + 116 = k * (k + 1) * (k + 2) * (k + 3)) ∧
  n = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_n_for_consecutive_product_l652_65240


namespace NUMINAMATH_CALUDE_polynomial_division_degree_l652_65237

theorem polynomial_division_degree (f d q r : Polynomial ℝ) : 
  Polynomial.degree f = 15 →
  Polynomial.degree q = 8 →
  Polynomial.degree r = 2 →
  f = d * q + r →
  Polynomial.degree r < Polynomial.degree d →
  Polynomial.degree d = 7 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_degree_l652_65237


namespace NUMINAMATH_CALUDE_not_prime_special_expression_l652_65207

theorem not_prime_special_expression (n : ℕ) (h : n > 2) :
  ¬ Nat.Prime (n^(n^n) - 4*n^n + 3) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_special_expression_l652_65207


namespace NUMINAMATH_CALUDE_sqrt_21_times_sqrt_7_minus_sqrt_3_l652_65203

theorem sqrt_21_times_sqrt_7_minus_sqrt_3 :
  Real.sqrt 21 * Real.sqrt 7 - Real.sqrt 3 = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_21_times_sqrt_7_minus_sqrt_3_l652_65203


namespace NUMINAMATH_CALUDE_number_calculation_l652_65275

theorem number_calculation (number : ℝ) : 
  (number / 0.3 = 0.03) → number = 0.009 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l652_65275


namespace NUMINAMATH_CALUDE_vampire_consumption_l652_65221

/-- Represents the number of people consumed by the vampire and werewolf -/
structure Consumption where
  vampire : ℕ
  werewolf : ℕ

/-- The total consumption over a given number of weeks -/
def total_consumption (c : Consumption) (weeks : ℕ) : ℕ :=
  weeks * (c.vampire + c.werewolf)

theorem vampire_consumption (village_population : ℕ) (duration_weeks : ℕ) (c : Consumption) :
  village_population = 72 →
  duration_weeks = 9 →
  c.werewolf = 5 →
  total_consumption c duration_weeks = village_population →
  c.vampire = 3 := by
  sorry

end NUMINAMATH_CALUDE_vampire_consumption_l652_65221


namespace NUMINAMATH_CALUDE_connie_savings_theorem_connie_savings_value_l652_65238

/-- The amount of money Connie saved up -/
def connie_savings : ℕ := sorry

/-- The cost of the watch -/
def watch_cost : ℕ := 55

/-- The additional amount Connie needs -/
def additional_needed : ℕ := 16

/-- Theorem stating that Connie's savings plus the additional amount needed equals the watch cost -/
theorem connie_savings_theorem : connie_savings + additional_needed = watch_cost := by sorry

/-- Theorem proving that Connie's savings equal $39 -/
theorem connie_savings_value : connie_savings = 39 := by sorry

end NUMINAMATH_CALUDE_connie_savings_theorem_connie_savings_value_l652_65238


namespace NUMINAMATH_CALUDE_aira_rubber_bands_l652_65204

theorem aira_rubber_bands (samantha aira joe : ℕ) : 
  samantha = aira + 5 →
  joe = aira + 1 →
  samantha + aira + joe = 18 →
  aira = 4 := by
sorry

end NUMINAMATH_CALUDE_aira_rubber_bands_l652_65204


namespace NUMINAMATH_CALUDE_square_equation_solution_l652_65218

theorem square_equation_solution : ∃! (M : ℕ), M > 0 ∧ 16^2 * 40^2 = 20^2 * M^2 ∧ M = 32 := by sorry

end NUMINAMATH_CALUDE_square_equation_solution_l652_65218


namespace NUMINAMATH_CALUDE_combined_average_score_l652_65258

theorem combined_average_score (score_a score_b score_c : ℝ)
  (ratio_a ratio_b ratio_c : ℕ) :
  score_a = 65 →
  score_b = 90 →
  score_c = 77 →
  ratio_a = 4 →
  ratio_b = 6 →
  ratio_c = 5 →
  (ratio_a * score_a + ratio_b * score_b + ratio_c * score_c) / (ratio_a + ratio_b + ratio_c) = 79 := by
  sorry

end NUMINAMATH_CALUDE_combined_average_score_l652_65258


namespace NUMINAMATH_CALUDE_x_equals_neg_x_valid_l652_65285

/-- Represents a variable in a programming context -/
structure Variable where
  name : String

/-- Represents an expression in a programming context -/
inductive Expression where
  | Var : Variable → Expression
  | Num : Int → Expression
  | Neg : Expression → Expression
  | Add : Expression → Expression → Expression
  | Str : String → Expression

/-- Represents an assignment statement -/
structure Assignment where
  lhs : Expression
  rhs : Expression

/-- Predicate to check if an assignment is valid -/
def is_valid_assignment (a : Assignment) : Prop :=
  match a.lhs with
  | Expression.Var _ => True
  | _ => False

/-- Theorem stating that x = -x is a valid assignment -/
theorem x_equals_neg_x_valid :
  ∃ (x : Variable),
    is_valid_assignment { lhs := Expression.Var x, rhs := Expression.Neg (Expression.Var x) } ∧
    ¬is_valid_assignment { lhs := Expression.Num 5, rhs := Expression.Str "M" } ∧
    ¬is_valid_assignment { lhs := Expression.Add (Expression.Var ⟨"x"⟩) (Expression.Var ⟨"y"⟩), rhs := Expression.Num 0 } :=
by
  sorry


end NUMINAMATH_CALUDE_x_equals_neg_x_valid_l652_65285


namespace NUMINAMATH_CALUDE_arrangement_remainder_l652_65232

/-- The number of green marbles --/
def green_marbles : ℕ := 7

/-- The maximum number of blue marbles satisfying the arrangement condition --/
def max_blue_marbles : ℕ := 19

/-- The total number of marbles --/
def total_marbles : ℕ := green_marbles + max_blue_marbles

/-- The number of ways to arrange the marbles --/
def arrangement_count : ℕ := Nat.choose total_marbles green_marbles

/-- Theorem stating the remainder when the number of arrangements is divided by 500 --/
theorem arrangement_remainder : arrangement_count % 500 = 30 := by sorry

end NUMINAMATH_CALUDE_arrangement_remainder_l652_65232


namespace NUMINAMATH_CALUDE_divides_a_iff_divides_n_l652_65216

/-- Sequence defined by a(n) = 2a(n-1) + a(n-2) for n > 1, with a(0) = 0 and a(1) = 1 -/
def a : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 2 * a (n + 1) + a n

/-- For all natural numbers k and n, 2^k divides a(n) if and only if 2^k divides n -/
theorem divides_a_iff_divides_n (k n : ℕ) : (2^k : ℤ) ∣ a n ↔ 2^k ∣ n := by sorry

end NUMINAMATH_CALUDE_divides_a_iff_divides_n_l652_65216


namespace NUMINAMATH_CALUDE_x_intercept_implies_m_slope_implies_m_l652_65205

/-- The equation of line l -/
def line_equation (m x y : ℝ) : Prop :=
  (m^2 - 2*m - 3)*x + (2*m^2 + m - 1)*y = 2*m - 6

/-- The x-intercept of line l is -3 -/
def x_intercept (m : ℝ) : Prop :=
  line_equation m (-3) 0

/-- The slope of line l is -1 -/
def slope_negative_one (m : ℝ) : Prop :=
  m^2 - 2*m - 3 = -(2*m^2 + m - 1) ∧ m^2 - 2*m - 3 ≠ 0

theorem x_intercept_implies_m (m : ℝ) :
  x_intercept m → m = -5/3 :=
sorry

theorem slope_implies_m (m : ℝ) :
  slope_negative_one m → m = 4/3 :=
sorry

end NUMINAMATH_CALUDE_x_intercept_implies_m_slope_implies_m_l652_65205


namespace NUMINAMATH_CALUDE_solutions_to_equation_unique_solutions_l652_65215

-- Define the equation
def equation (s : ℝ) : ℝ := 12 * s^2 + 2 * s

-- Theorem stating that 0.5 and -2/3 are solutions to the equation when t = 4
theorem solutions_to_equation :
  equation (1/2) = 4 ∧ equation (-2/3) = 4 :=
by sorry

-- Theorem stating that these are the only solutions
theorem unique_solutions (s : ℝ) :
  equation s = 4 ↔ s = 1/2 ∨ s = -2/3 :=
by sorry

end NUMINAMATH_CALUDE_solutions_to_equation_unique_solutions_l652_65215
