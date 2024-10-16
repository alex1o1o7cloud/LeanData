import Mathlib

namespace NUMINAMATH_CALUDE_projection_implies_y_value_l498_49844

/-- Given two vectors v and w in ℝ², prove that if the projection of v onto w
    is [-8, -12], then the y-coordinate of v must be -56/3. -/
theorem projection_implies_y_value (v w : ℝ × ℝ) (y : ℝ) 
    (h1 : v = (2, y))
    (h2 : w = (4, 6))
    (h3 : (v • w / (w • w)) • w = (-8, -12)) :
  y = -56/3 := by
  sorry

end NUMINAMATH_CALUDE_projection_implies_y_value_l498_49844


namespace NUMINAMATH_CALUDE_simplify_and_ratio_l498_49864

theorem simplify_and_ratio (m : ℝ) : 
  (6*m + 12) / 6 = m + 2 ∧ (1 : ℝ) / 2 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_ratio_l498_49864


namespace NUMINAMATH_CALUDE_pyramid_surface_area_l498_49854

/-- Given a pyramid with its base coinciding with a face of a cube and its apex at the center
    of the opposite face, the surface area of the pyramid can be expressed in terms of the
    cube's edge length. -/
theorem pyramid_surface_area (a : ℝ) (h : a > 0) :
  ∃ (S : ℝ), S = (a * (3 * Real.sqrt (4 * a^2 - a^2) + a * Real.sqrt 3)) / 36 :=
sorry

end NUMINAMATH_CALUDE_pyramid_surface_area_l498_49854


namespace NUMINAMATH_CALUDE_average_age_after_leaving_l498_49875

theorem average_age_after_leaving (initial_people : ℕ) (initial_avg : ℚ) 
  (leaving_age : ℕ) (remaining_people : ℕ) :
  initial_people = 5 →
  initial_avg = 30 →
  leaving_age = 18 →
  remaining_people = 4 →
  (initial_people * initial_avg - leaving_age) / remaining_people = 33 := by
  sorry

end NUMINAMATH_CALUDE_average_age_after_leaving_l498_49875


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l498_49884

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + (a-1)*x + 1 < 0) → (a > 3 ∨ a < -1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l498_49884


namespace NUMINAMATH_CALUDE_triangle_sine_inequality_l498_49805

theorem triangle_sine_inequality (A B C : Real) (h_triangle : A + B + C = Real.pi) :
  -2 < Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) ∧
  Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) ≤ (3 / 2) * Real.sqrt 3 ∧
  (Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) = (3 / 2) * Real.sqrt 3 ↔
   A = 7 * Real.pi / 9 ∧ B = Real.pi / 9 ∧ C = Real.pi / 9) :=
by sorry

end NUMINAMATH_CALUDE_triangle_sine_inequality_l498_49805


namespace NUMINAMATH_CALUDE_parabola_point_distance_l498_49836

/-- Given a parabola y² = x with focus at (1/4, 0), prove that a point on the parabola
    with distance 1 from the focus has x-coordinate 3/4 -/
theorem parabola_point_distance (x y : ℝ) : 
  y^2 = x →                                           -- Point (x, y) is on the parabola
  (x - 1/4)^2 + y^2 = 1 →                             -- Distance from (x, y) to focus (1/4, 0) is 1
  x = 3/4 := by sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l498_49836


namespace NUMINAMATH_CALUDE_bags_used_by_kid4_l498_49855

def hours : ℕ := 5
def ears_per_row : ℕ := 85
def seeds_per_bag : ℕ := 48
def seeds_per_ear : ℕ := 2
def rows_per_hour_kid4 : ℕ := 5

def bags_used_kid4 : ℕ :=
  let rows := hours * rows_per_hour_kid4
  let ears := rows * ears_per_row
  let seeds := ears * seeds_per_ear
  (seeds + seeds_per_bag - 1) / seeds_per_bag

theorem bags_used_by_kid4 : bags_used_kid4 = 89 := by sorry

end NUMINAMATH_CALUDE_bags_used_by_kid4_l498_49855


namespace NUMINAMATH_CALUDE_smallest_square_box_for_cards_l498_49822

/-- Represents the dimensions of a business card -/
structure BusinessCard where
  width : ℕ
  length : ℕ

/-- Represents a square box -/
structure SquareBox where
  side : ℕ

/-- Checks if a square box can fit a whole number of business cards without overlapping -/
def canFitCards (box : SquareBox) (card : BusinessCard) : Prop :=
  (box.side % card.width = 0) ∧ (box.side % card.length = 0)

/-- Theorem: The smallest square box that can fit business cards of 5x7 cm has sides of 35 cm -/
theorem smallest_square_box_for_cards :
  let card := BusinessCard.mk 5 7
  let box := SquareBox.mk 35
  (canFitCards box card) ∧
  (∀ (smallerBox : SquareBox), smallerBox.side < box.side → ¬(canFitCards smallerBox card)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_box_for_cards_l498_49822


namespace NUMINAMATH_CALUDE_farmer_plant_beds_l498_49809

theorem farmer_plant_beds (bean_seedlings : ℕ) (bean_per_row : ℕ) 
  (pumpkin_seeds : ℕ) (pumpkin_per_row : ℕ) 
  (radishes : ℕ) (radish_per_row : ℕ) 
  (rows_per_bed : ℕ) : 
  bean_seedlings = 64 → 
  bean_per_row = 8 → 
  pumpkin_seeds = 84 → 
  pumpkin_per_row = 7 → 
  radishes = 48 → 
  radish_per_row = 6 → 
  rows_per_bed = 2 → 
  (bean_seedlings / bean_per_row + 
   pumpkin_seeds / pumpkin_per_row + 
   radishes / radish_per_row) / rows_per_bed = 14 := by
  sorry

#check farmer_plant_beds

end NUMINAMATH_CALUDE_farmer_plant_beds_l498_49809


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l498_49891

theorem rectangle_perimeter (a b : ℝ) (h1 : a + b = 7) (h2 : 2 * a + b = 9.5) :
  2 * (a + b) = 14 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l498_49891


namespace NUMINAMATH_CALUDE_statement_is_universal_l498_49867

-- Define the concept of a line
def Line : Type := sorry

-- Define the concept of a plane
def Plane : Type := sorry

-- Define the property of two lines intersecting
def intersect (l1 l2 : Line) : Prop := sorry

-- Define the property of a plane passing through two lines
def passes_through (p : Plane) (l1 l2 : Line) : Prop := sorry

-- Define the statement as a proposition
def statement : Prop :=
  ∀ l1 l2 : Line, intersect l1 l2 → ∃! p : Plane, passes_through p l1 l2

-- Theorem to prove that the statement is a universal proposition
theorem statement_is_universal : 
  (∀ l1 l2 : Line, intersect l1 l2 → ∃! p : Plane, passes_through p l1 l2) ↔ statement :=
sorry

end NUMINAMATH_CALUDE_statement_is_universal_l498_49867


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l498_49838

def is_simplest_quadratic_radical (x : ℝ → ℝ) (others : List (ℝ → ℝ)) : Prop :=
  ∀ y ∈ others, ∃ k : ℝ, k ≠ 0 ∧ ∀ a : ℝ, (x a) = k * (y a) → k = 1

theorem simplest_quadratic_radical :
  let x : ℝ → ℝ := λ a => Real.sqrt (a^2 + 1)
  let y₁ : ℝ → ℝ := λ _ => Real.sqrt 8
  let y₂ : ℝ → ℝ := λ _ => 1 / Real.sqrt 3
  let y₃ : ℝ → ℝ := λ _ => Real.sqrt 0.5
  is_simplest_quadratic_radical x [y₁, y₂, y₃] :=
sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l498_49838


namespace NUMINAMATH_CALUDE_max_value_abc_fraction_l498_49857

theorem max_value_abc_fraction (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b * c * (a + b + c)) / ((a + b)^3 * (b + c)^3) ≤ (1 : ℝ) / 4 ∧
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a * b * c * (a + b + c)) / ((a + b)^3 * (b + c)^3) = (1 : ℝ) / 4 :=
sorry

end NUMINAMATH_CALUDE_max_value_abc_fraction_l498_49857


namespace NUMINAMATH_CALUDE_similar_triangle_sum_l498_49848

theorem similar_triangle_sum (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a / 3 = b / 5) (h5 : b / 5 = c / 7) (h6 : c = 21) : a + b = 24 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangle_sum_l498_49848


namespace NUMINAMATH_CALUDE_existence_of_powers_of_seven_with_difference_divisible_by_2021_l498_49886

theorem existence_of_powers_of_seven_with_difference_divisible_by_2021 :
  ∃ (n m : ℕ), n > m ∧ (7^n - 7^m) % 2021 = 0 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_powers_of_seven_with_difference_divisible_by_2021_l498_49886


namespace NUMINAMATH_CALUDE_line_through_intersection_and_perpendicular_l498_49831

-- Define the lines l₁, l₂, and l₃
def l₁ (x y : ℝ) : Prop := x - 2*y + 3 = 0
def l₂ (x y : ℝ) : Prop := 2*x + 3*y - 8 = 0
def l₃ (x y : ℝ) : Prop := 3*x - y + 1 = 0

-- Define the line l (the answer)
def l (x y : ℝ) : Prop := x + 3*y - 7 = 0

-- Define the intersection point M
def M : ℝ × ℝ := (1, 2)

-- Theorem statement
theorem line_through_intersection_and_perpendicular :
  (l₁ M.1 M.2 ∧ l₂ M.1 M.2) ∧  -- M is the intersection of l₁ and l₂
  (∀ x y : ℝ, l x y → l₃ x y → (x - M.1) * 3 + (y - M.2) * (-1) = 0) ∧  -- l is perpendicular to l₃
  l M.1 M.2  -- l passes through M
  := by sorry

end NUMINAMATH_CALUDE_line_through_intersection_and_perpendicular_l498_49831


namespace NUMINAMATH_CALUDE_inequality_holds_iff_first_quadrant_l498_49841

theorem inequality_holds_iff_first_quadrant (θ : Real) :
  (∀ x : Real, x ∈ Set.Icc 0 1 →
    x^2 * Real.cos θ - 3 * x * (1 - x) + (1 - x)^2 * Real.sin θ > 0) ↔
  θ ∈ Set.Ioo 0 (Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_first_quadrant_l498_49841


namespace NUMINAMATH_CALUDE_book_selection_count_l498_49897

/-- Represents the number of books in each genre -/
def genre_books : Fin 4 → ℕ
  | 0 => 4  -- Mystery novels
  | 1 => 3  -- Fantasy novels
  | 2 => 3  -- Biographies
  | 3 => 3  -- Science fiction novels

/-- The number of ways to choose three books from three different genres -/
def book_combinations : ℕ := 4 * 3 * 3 * 3

theorem book_selection_count :
  book_combinations = 108 :=
sorry

end NUMINAMATH_CALUDE_book_selection_count_l498_49897


namespace NUMINAMATH_CALUDE_greatest_b_quadratic_inequality_l498_49815

theorem greatest_b_quadratic_inequality :
  ∃ b : ℝ, b^2 - 14*b + 45 ≤ 0 ∧
  ∀ x : ℝ, x^2 - 14*x + 45 ≤ 0 → x ≤ b ∧
  b = 9 :=
sorry

end NUMINAMATH_CALUDE_greatest_b_quadratic_inequality_l498_49815


namespace NUMINAMATH_CALUDE_thirtieth_digit_of_sum_l498_49802

def fraction_sum (a b c : ℚ) : ℚ := a + b + c

def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ :=
  sorry

theorem thirtieth_digit_of_sum :
  nth_digit_after_decimal (fraction_sum (1/7) (1/3) (1/11)) 30 = 9 :=
sorry

end NUMINAMATH_CALUDE_thirtieth_digit_of_sum_l498_49802


namespace NUMINAMATH_CALUDE_one_language_speakers_l498_49893

theorem one_language_speakers (total : ℕ) (latin french spanish : ℕ) (none : ℕ) 
  (latin_french latin_spanish french_spanish : ℕ) (all_three : ℕ) 
  (h1 : total = 40)
  (h2 : latin = 20)
  (h3 : french = 22)
  (h4 : spanish = 15)
  (h5 : none = 5)
  (h6 : latin_french = 8)
  (h7 : latin_spanish = 6)
  (h8 : french_spanish = 4)
  (h9 : all_three = 3) :
  total - none - (latin_french + latin_spanish + french_spanish - 2 * all_three) - all_three = 20 := by
  sorry

#check one_language_speakers

end NUMINAMATH_CALUDE_one_language_speakers_l498_49893


namespace NUMINAMATH_CALUDE_pencils_per_row_l498_49863

/-- Given 12 pencils distributed equally among 3 rows, prove that there are 4 pencils in each row. -/
theorem pencils_per_row (total_pencils : ℕ) (num_rows : ℕ) (pencils_per_row : ℕ) 
  (h1 : total_pencils = 12) 
  (h2 : num_rows = 3) 
  (h3 : total_pencils = num_rows * pencils_per_row) : 
  pencils_per_row = 4 := by
  sorry

end NUMINAMATH_CALUDE_pencils_per_row_l498_49863


namespace NUMINAMATH_CALUDE_natural_number_representation_l498_49829

/-- Binomial coefficient -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem natural_number_representation (n : ℕ) :
  ∃ x y z : ℕ, n = choose x 1 + choose y 2 + choose z 3 ∧
    ((0 ≤ x ∧ x < y ∧ y < z) ∨ (0 = x ∧ x = y ∧ y < z)) :=
  sorry

end NUMINAMATH_CALUDE_natural_number_representation_l498_49829


namespace NUMINAMATH_CALUDE_cuboid_volume_l498_49883

/-- Given a cuboid with edges a, b, and c, if the shortest distances between its space diagonal
    and three distinct edges are 2√5, 30/√13, and 15/√10, then its volume is 750. -/
theorem cuboid_volume (a b c : ℝ) (h₁ : a * b / Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 5)
  (h₂ : b * c / Real.sqrt (b^2 + c^2) = 30 / Real.sqrt 13)
  (h₃ : a * c / Real.sqrt (a^2 + c^2) = 15 / Real.sqrt 10) :
  a * b * c = 750 :=
by sorry

end NUMINAMATH_CALUDE_cuboid_volume_l498_49883


namespace NUMINAMATH_CALUDE_infinitely_many_pairs_l498_49834

theorem infinitely_many_pairs : 
  Set.Infinite {p : ℕ × ℕ | 2019 < (2 : ℝ)^p.1 / (3 : ℝ)^p.2 ∧ (2 : ℝ)^p.1 / (3 : ℝ)^p.2 < 2020} :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_pairs_l498_49834


namespace NUMINAMATH_CALUDE_distance_O_to_MN_l498_49862

/-- The hyperbola C₁: 2x² - y² = 1 -/
def C₁ (x y : ℝ) : Prop := 2 * x^2 - y^2 = 1

/-- The ellipse C₂: 4x² + y² = 1 -/
def C₂ (x y : ℝ) : Prop := 4 * x^2 + y^2 = 1

/-- M is a point on C₁ -/
def M : ℝ × ℝ := sorry

/-- N is a point on C₂ -/
def N : ℝ × ℝ := sorry

/-- O is the origin -/
def O : ℝ × ℝ := (0, 0)

/-- OM is perpendicular to ON -/
def OM_perp_ON : Prop := sorry

/-- The distance from a point to a line -/
noncomputable def distancePointToLine (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ := sorry

/-- The line MN -/
def lineMN : Set (ℝ × ℝ) := sorry

/-- Main theorem: The distance from O to MN is √3/3 -/
theorem distance_O_to_MN :
  C₁ M.1 M.2 → C₂ N.1 N.2 → OM_perp_ON →
  distancePointToLine O lineMN = Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_distance_O_to_MN_l498_49862


namespace NUMINAMATH_CALUDE_sphere_radius_is_zero_l498_49880

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- The configuration of points and lines in the problem -/
structure Configuration where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  m : Line3D
  n : Line3D
  a : ℝ
  b : ℝ
  θ : ℝ

/-- Checks if two points are distinct -/
def are_distinct (p q : Point3D) : Prop :=
  p ≠ q

/-- Checks if a line is perpendicular to another line -/
def is_perpendicular (l1 l2 : Line3D) : Prop :=
  sorry

/-- Checks if a point is on a line -/
def point_on_line (p : Point3D) (l : Line3D) : Prop :=
  sorry

/-- Calculates the distance between two points -/
def distance (p q : Point3D) : ℝ :=
  sorry

/-- Calculates the angle between two lines -/
def angle_between_lines (l1 l2 : Line3D) : ℝ :=
  sorry

/-- Calculates the radius of a sphere passing through four points -/
def sphere_radius (p1 p2 p3 p4 : Point3D) : ℝ :=
  sorry

/-- The main theorem stating that the radius of the sphere is zero -/
theorem sphere_radius_is_zero (config : Configuration) :
  are_distinct config.A config.B ∧
  is_perpendicular config.m (Line3D.mk config.A config.B) ∧
  is_perpendicular config.n (Line3D.mk config.A config.B) ∧
  point_on_line config.C config.m ∧
  are_distinct config.A config.C ∧
  point_on_line config.D config.n ∧
  are_distinct config.B config.D ∧
  distance config.A config.B = config.a ∧
  distance config.C config.D = config.b ∧
  angle_between_lines config.m config.n = config.θ
  →
  sphere_radius config.A config.B config.C config.D = 0 :=
sorry

end NUMINAMATH_CALUDE_sphere_radius_is_zero_l498_49880


namespace NUMINAMATH_CALUDE_launderette_machine_count_l498_49849

/-- Represents a laundry machine with quarters and dimes -/
structure LaundryMachine where
  quarters : ℕ
  dimes : ℕ

/-- Calculates the value of a laundry machine in cents -/
def machine_value (m : LaundryMachine) : ℕ :=
  m.quarters * 25 + m.dimes * 10

/-- Represents the launderette -/
structure Launderette where
  machine : LaundryMachine
  total_value : ℕ
  machine_count : ℕ

/-- Theorem: The number of machines in the launderette is 3 -/
theorem launderette_machine_count (l : Launderette) 
  (h1 : l.machine.quarters = 80)
  (h2 : l.machine.dimes = 100)
  (h3 : l.total_value = 9000) -- $90 in cents
  (h4 : l.machine_count * machine_value l.machine = l.total_value) :
  l.machine_count = 3 := by
  sorry

end NUMINAMATH_CALUDE_launderette_machine_count_l498_49849


namespace NUMINAMATH_CALUDE_radio_cost_price_l498_49868

theorem radio_cost_price (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1275)
  (h2 : loss_percentage = 15) : 
  ∃ (cost_price : ℝ), 
    cost_price = 1500 ∧ 
    selling_price = cost_price * (1 - loss_percentage / 100) := by
sorry

end NUMINAMATH_CALUDE_radio_cost_price_l498_49868


namespace NUMINAMATH_CALUDE_min_value_implies_a_eq_four_l498_49858

/-- Given a function f(x) = 4x + a²/x where x > 0 and x ∈ ℝ, 
    if f attains its minimum value at x = 2, then a = 4. -/
theorem min_value_implies_a_eq_four (a : ℝ) :
  (∀ x : ℝ, x > 0 → ∃ (f : ℝ → ℝ), f x = 4*x + a^2/x) →
  (∃ (f : ℝ → ℝ), ∀ x : ℝ, x > 0 → f x ≥ f 2) →
  a = 4 := by
  sorry


end NUMINAMATH_CALUDE_min_value_implies_a_eq_four_l498_49858


namespace NUMINAMATH_CALUDE_father_son_age_difference_l498_49811

/-- Represents the age difference between a father and son -/
def AgeDifference (fatherAge sonAge : ℕ) : ℕ := fatherAge - sonAge

/-- The problem statement -/
theorem father_son_age_difference :
  ∀ (fatherAge sonAge : ℕ),
  fatherAge > sonAge →
  fatherAge + 2 = 2 * (sonAge + 2) →
  sonAge = 28 →
  AgeDifference fatherAge sonAge = 30 := by
  sorry

end NUMINAMATH_CALUDE_father_son_age_difference_l498_49811


namespace NUMINAMATH_CALUDE_symmetry_axis_of_f_l498_49801

/-- The quadratic function f(x) = -2(x-1)^2 + 3 -/
def f (x : ℝ) : ℝ := -2 * (x - 1)^2 + 3

/-- The axis of symmetry for the quadratic function f -/
def axis_of_symmetry : ℝ := 1

/-- Theorem: The axis of symmetry of f(x) = -2(x-1)^2 + 3 is x = 1 -/
theorem symmetry_axis_of_f :
  ∀ x : ℝ, f (axis_of_symmetry + x) = f (axis_of_symmetry - x) :=
by
  sorry


end NUMINAMATH_CALUDE_symmetry_axis_of_f_l498_49801


namespace NUMINAMATH_CALUDE_unique_matrix_solution_l498_49807

theorem unique_matrix_solution (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ) 
  (h : A^3 = 0) : 
  ∃! X : Matrix (Fin n) (Fin n) ℝ, X + A * X + X * A^2 = A ∧ 
  X = A * (1 + A + A^2)⁻¹ := by
sorry

end NUMINAMATH_CALUDE_unique_matrix_solution_l498_49807


namespace NUMINAMATH_CALUDE_intersection_at_origin_l498_49818

/-- A line in the coordinate plane --/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The origin point (0, 0) --/
def origin : ℝ × ℝ := (0, 0)

/-- Check if a point lies on a line --/
def pointOnLine (l : Line) (p : ℝ × ℝ) : Prop :=
  p.2 = l.slope * p.1 + (l.point.2 - l.slope * l.point.1)

theorem intersection_at_origin 
  (k : Line)
  (l : Line)
  (hk_slope : k.slope = 1/2)
  (hk_origin : pointOnLine k origin)
  (hl_slope : l.slope = -2)
  (hl_point : l.point = (-2, 4)) :
  ∃ (p : ℝ × ℝ), pointOnLine k p ∧ pointOnLine l p ∧ p = origin :=
sorry

#check intersection_at_origin

end NUMINAMATH_CALUDE_intersection_at_origin_l498_49818


namespace NUMINAMATH_CALUDE_count_ordered_pairs_l498_49835

theorem count_ordered_pairs (n : ℕ) (hn : n > 1) :
  (Finset.sum (Finset.range (n - 1)) (fun k => n - k)) = (n - 1) * n / 2 := by
  sorry

end NUMINAMATH_CALUDE_count_ordered_pairs_l498_49835


namespace NUMINAMATH_CALUDE_smallest_sum_of_sequence_l498_49889

theorem smallest_sum_of_sequence (A B C D : ℕ) : 
  A > 0 → B > 0 → C > 0 → 
  (C - B = B - A) →  -- arithmetic sequence condition
  (C * C = B * D) →  -- geometric sequence condition
  (C : ℚ) / B = 7 / 4 →
  A + B + C + D ≥ 97 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_sequence_l498_49889


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l498_49851

theorem cyclic_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b) / (a + b) + (b * c) / (b + c) + (c * a) / (c + a) ≤ 3 * (a * b + b * c + c * a) / (2 * (a + b + c)) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l498_49851


namespace NUMINAMATH_CALUDE_first_part_second_part_l498_49800

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 + 2*a*x - 3

-- Theorem for the first part of the problem
theorem first_part (a : ℝ) : f a (a + 1) - f a a = 9 → a = 2 := by
  sorry

-- Theorem for the second part of the problem
theorem second_part (a : ℝ) : 
  (∀ x, f a x ≥ -4) ∧ (∃ x, f a x = -4) ↔ (a = 1 ∨ a = -1) := by
  sorry

end NUMINAMATH_CALUDE_first_part_second_part_l498_49800


namespace NUMINAMATH_CALUDE_isosceles_triangles_12_similar_l498_49833

/-- An isosceles triangle with side ratio 1:2 -/
structure IsoscelesTriangle12 where
  a : ℝ  -- Length of one side
  b : ℝ  -- Length of another side
  h : a = 2 * b ∨ b = 2 * a  -- Condition for 1:2 ratio

/-- Similarity of isosceles triangles with 1:2 side ratio -/
theorem isosceles_triangles_12_similar (t1 t2 : IsoscelesTriangle12) :
  ∃ (k : ℝ), k > 0 ∧ 
    (t1.a = k * t2.a ∧ t1.b = k * t2.b) ∨
    (t1.a = k * t2.b ∧ t1.b = k * t2.a) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangles_12_similar_l498_49833


namespace NUMINAMATH_CALUDE_probability_A_selected_l498_49852

/-- The number of people in the group -/
def group_size : ℕ := 4

/-- The number of representatives to be selected -/
def representatives : ℕ := 2

/-- The probability of person A being selected as a representative -/
def prob_A_selected : ℚ := 1/2

/-- Theorem stating the probability of person A being selected as a representative -/
theorem probability_A_selected :
  prob_A_selected = (representatives : ℚ) / group_size :=
by sorry

end NUMINAMATH_CALUDE_probability_A_selected_l498_49852


namespace NUMINAMATH_CALUDE_fourth_to_third_l498_49839

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is in the fourth quadrant -/
def in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Predicate to check if a point is in the third quadrant -/
def in_third_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Theorem stating that if P(a,b) is in the fourth quadrant, 
    then Q(-a,b-1) is in the third quadrant -/
theorem fourth_to_third (a b : ℝ) :
  in_fourth_quadrant ⟨a, b⟩ → in_third_quadrant ⟨-a, b-1⟩ := by
  sorry


end NUMINAMATH_CALUDE_fourth_to_third_l498_49839


namespace NUMINAMATH_CALUDE_hundred_billion_scientific_notation_l498_49821

theorem hundred_billion_scientific_notation :
  (100000000000 : ℕ) = 1 * 10^11 :=
sorry

end NUMINAMATH_CALUDE_hundred_billion_scientific_notation_l498_49821


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_and_line_in_plane_l498_49888

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- Theorem statement
theorem line_perpendicular_to_plane_and_line_in_plane
  (m n : Line) (α : Plane)
  (h1 : perpendicular m α)
  (h2 : contains α n) :
  perpendicularLines m n :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_and_line_in_plane_l498_49888


namespace NUMINAMATH_CALUDE_largest_710_triple_l498_49806

/-- Converts a base-10 number to its base-7 representation as a list of digits -/
def toBase7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

/-- Converts a list of digits in base-7 to a base-10 number -/
def fromBase7ToBase10 (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => acc * 10 + d) 0

/-- Checks if a number is a 7-10 triple -/
def is710Triple (n : ℕ) : Prop :=
  let base7Digits := toBase7 n
  fromBase7ToBase10 base7Digits = 3 * n

/-- States that 335 is the largest 7-10 triple -/
theorem largest_710_triple :
  (is710Triple 335) ∧ (∀ m : ℕ, m > 335 → ¬(is710Triple m)) := by
  sorry

end NUMINAMATH_CALUDE_largest_710_triple_l498_49806


namespace NUMINAMATH_CALUDE_evaluate_expression_l498_49840

theorem evaluate_expression (c d : ℝ) (h : c^2 ≠ d^2) :
  (c^4 - d^4) / (2 * (c^2 - d^2)) = (c^2 + d^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l498_49840


namespace NUMINAMATH_CALUDE_smallest_w_l498_49877

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w (w : ℕ) : 
  w > 0 → 
  is_factor (2^4) (1452 * w) → 
  is_factor (3^3) (1452 * w) → 
  is_factor (13^3) (1452 * w) → 
  w ≥ 79132 :=
sorry

end NUMINAMATH_CALUDE_smallest_w_l498_49877


namespace NUMINAMATH_CALUDE_sunlight_rice_yield_correlation_l498_49812

/-- Definition of a correlation relationship -/
def isCorrelation (X Y : Type) (relationship : X → Y → Prop) : Prop :=
  ∃ (pattern : X → Y → Prop),
    (∀ x : X, ∃ y : Y, relationship x y) ∧
    (∀ x : X, ∃ y₁ y₂ : Y, y₁ ≠ y₂ ∧ relationship x y₁ ∧ relationship x y₂) ∧
    (∀ x : X, ∀ y : Y, relationship x y → pattern x y)

/-- Amount of sunlight -/
def Sunlight : Type := ℝ

/-- Per-acre yield of rice -/
def RiceYield : Type := ℝ

/-- The relationship between sunlight and rice yield -/
def sunlightRiceYieldRelation : Sunlight → RiceYield → Prop := sorry

theorem sunlight_rice_yield_correlation :
  isCorrelation Sunlight RiceYield sunlightRiceYieldRelation := by sorry

end NUMINAMATH_CALUDE_sunlight_rice_yield_correlation_l498_49812


namespace NUMINAMATH_CALUDE_difference_ones_zeros_is_six_l498_49885

-- Define the number in base 10
def base_10_num : Nat := 253

-- Define a function to convert a number to its binary representation
def to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : Nat) (acc : List Bool) : List Bool :=
    if m = 0 then acc else aux (m / 2) ((m % 2 = 1) :: acc)
  aux n []

-- Define functions to count zeros and ones in a binary representation
def count_zeros (binary : List Bool) : Nat :=
  binary.filter (· = false) |>.length

def count_ones (binary : List Bool) : Nat :=
  binary.filter (· = true) |>.length

-- Theorem statement
theorem difference_ones_zeros_is_six :
  let binary := to_binary base_10_num
  let y := count_ones binary
  let x := count_zeros binary
  y - x = 6 := by sorry

end NUMINAMATH_CALUDE_difference_ones_zeros_is_six_l498_49885


namespace NUMINAMATH_CALUDE_problem_solution_l498_49813

/-- Given the conditions of the problem, prove that x · z = 4.5 -/
theorem problem_solution :
  ∀ x y z : ℝ,
  (∃ x₀ y₀ z₀ : ℝ, x₀ = 2*y₀ ∧ z₀ = x₀ ∧ x₀*y₀ = y₀*z₀) →  -- Initial condition
  z = x/2 →
  x*y = y^2 →
  y = 3 →
  x*z = 4.5 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l498_49813


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l498_49894

structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  h_p_pos : p > 0
  h_eq : ∀ x y, eq x y ↔ y^2 = 2*p*x
  h_focus : focus = (1, 0)

structure Line where
  m : ℝ
  b : ℝ
  eq : ℝ → ℝ → Prop
  h_eq : ∀ x y, eq x y ↔ y = m*x + b

def intersect (C : Parabola) (l : Line) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | C.eq p.1 p.2 ∧ l.eq p.1 p.2}

def perpendicular (A B O : ℝ × ℝ) : Prop :=
  (A.1 - O.1) * (B.1 - O.1) + (A.2 - O.2) * (B.2 - O.2) = 0

theorem parabola_intersection_theorem (C : Parabola) (l : Line) 
  (A B : ℝ × ℝ) (h_AB : A ∈ intersect C l ∧ B ∈ intersect C l) 
  (h_perp : perpendicular A B (0, 0)) :
  ∃ (T : ℝ × ℝ), 
    (∃ (k : ℝ), ∀ (X : ℝ × ℝ), X ∈ intersect C l → 
      (X.2 / (X.1 - 4) + X.2 / (X.1 - T.1) = k)) ∧
    T = (-4, 0) ∧ 
    k = 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l498_49894


namespace NUMINAMATH_CALUDE_negative_root_condition_l498_49827

theorem negative_root_condition (p : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ x^4 - 4*p*x^3 + x^2 - 4*p*x + 1 = 0) ↔ p ≥ -3/8 := by sorry

end NUMINAMATH_CALUDE_negative_root_condition_l498_49827


namespace NUMINAMATH_CALUDE_triangle_properties_l498_49820

-- Define the triangle ABC
def A : ℝ × ℝ := (0, 2)
def B : ℝ × ℝ := (6, 4)
def C : ℝ × ℝ := (4, 0)

-- Define the perpendicular bisector equation
def perpendicular_bisector (x y : ℝ) : Prop :=
  2 * x - y - 3 = 0

-- Define the circumcircle equation
def circumcircle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 3)^2 = 10

-- Theorem statement
theorem triangle_properties :
  (∀ x y : ℝ, perpendicular_bisector x y ↔ 
    (x - (A.1 + C.1) / 2)^2 + (y - (A.2 + C.2) / 2)^2 = 
    ((A.1 - C.1)^2 + (A.2 - C.2)^2) / 4) ∧
  (∀ x y : ℝ, circumcircle x y ↔ 
    (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 ∧
    (x - B.1)^2 + (y - B.2)^2 = (x - C.1)^2 + (y - C.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l498_49820


namespace NUMINAMATH_CALUDE_max_contribution_l498_49810

theorem max_contribution 
  (total : ℝ) 
  (num_people : ℕ) 
  (min_contribution : ℝ) 
  (h1 : total = 20) 
  (h2 : num_people = 12) 
  (h3 : min_contribution = 1) 
  (h4 : ∀ p, p ≤ num_people → p • min_contribution ≤ total) : 
  ∃ max_contrib : ℝ, max_contrib = 9 ∧ 
    ∀ individual_contrib, 
      individual_contrib ≤ max_contrib ∧ 
      (num_people - 1) • min_contribution + individual_contrib = total :=
sorry

end NUMINAMATH_CALUDE_max_contribution_l498_49810


namespace NUMINAMATH_CALUDE_functional_equation_solution_l498_49869

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x * f y / f (x * y)

/-- Theorem stating the possible forms of f -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  (∀ x : ℝ, f x = 0) ∨ ∃ c : ℝ, c ≠ 0 ∧ ∀ x : ℝ, f x = c :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l498_49869


namespace NUMINAMATH_CALUDE_equation_proof_l498_49804

theorem equation_proof (h : Real.sqrt 27 = 3 * Real.sqrt 3) :
  -2 * Real.sqrt 3 + Real.sqrt 27 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l498_49804


namespace NUMINAMATH_CALUDE_completing_square_solution_l498_49873

theorem completing_square_solution (x : ℝ) :
  (x^2 - 4*x + 3 = 0) ↔ ((x - 2)^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_solution_l498_49873


namespace NUMINAMATH_CALUDE_bears_permutations_l498_49879

theorem bears_permutations :
  Finset.card (Finset.univ.image (fun σ : Equiv.Perm (Fin 5) => σ)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_bears_permutations_l498_49879


namespace NUMINAMATH_CALUDE_compound_weight_proof_l498_49828

/-- Atomic weight of Aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- Atomic weight of Bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.90

/-- Number of Aluminum atoms in the compound -/
def num_Al : ℕ := 1

/-- Number of Bromine atoms in the compound -/
def num_Br : ℕ := 3

/-- Molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 267

/-- Theorem stating that the molecular weight of the compound is approximately 267 g/mol -/
theorem compound_weight_proof :
  ∃ ε > 0, abs (molecular_weight - (num_Al * atomic_weight_Al + num_Br * atomic_weight_Br)) < ε :=
sorry

end NUMINAMATH_CALUDE_compound_weight_proof_l498_49828


namespace NUMINAMATH_CALUDE_integers_abs_leq_three_l498_49843

theorem integers_abs_leq_three :
  {x : ℤ | |x| ≤ 3} = {-3, -2, -1, 0, 1, 2, 3} := by
sorry

end NUMINAMATH_CALUDE_integers_abs_leq_three_l498_49843


namespace NUMINAMATH_CALUDE_overlapping_strips_l498_49824

theorem overlapping_strips (total_length width : ℝ) 
  (left_length right_length : ℝ) 
  (left_area right_area : ℝ) : 
  total_length = 16 →
  left_length = 9 →
  right_length = 7 →
  left_length + right_length = total_length →
  left_area = 27 →
  right_area = 18 →
  (left_area + (left_length * width)) / (right_area + (right_length * width)) = left_length / right_length →
  ∃ overlap_area : ℝ, overlap_area = 13.5 ∧ 
    (left_area + overlap_area) / (right_area + overlap_area) = left_length / right_length :=
by sorry

end NUMINAMATH_CALUDE_overlapping_strips_l498_49824


namespace NUMINAMATH_CALUDE_mod_power_seventeen_l498_49881

theorem mod_power_seventeen (n : ℕ) : 17^1501 ≡ 4 [ZMOD 19] := by
  sorry

end NUMINAMATH_CALUDE_mod_power_seventeen_l498_49881


namespace NUMINAMATH_CALUDE_world_cup_2006_group_stage_matches_l498_49892

/-- The number of matches in a single round-robin tournament with n teams -/
def matches_in_group (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The total number of matches in a tournament with groups -/
def total_matches (total_teams : ℕ) (num_groups : ℕ) : ℕ :=
  let teams_per_group := total_teams / num_groups
  num_groups * matches_in_group teams_per_group

theorem world_cup_2006_group_stage_matches :
  total_matches 32 8 = 48 := by
  sorry

end NUMINAMATH_CALUDE_world_cup_2006_group_stage_matches_l498_49892


namespace NUMINAMATH_CALUDE_hanna_money_spent_l498_49850

theorem hanna_money_spent (rose_price : ℚ) (jenna_fraction : ℚ) (imma_fraction : ℚ) (total_given : ℕ) : 
  rose_price = 2 →
  jenna_fraction = 1/3 →
  imma_fraction = 1/2 →
  total_given = 125 →
  (jenna_fraction + imma_fraction) * (total_given / (jenna_fraction + imma_fraction)) * rose_price = 300 := by
sorry

end NUMINAMATH_CALUDE_hanna_money_spent_l498_49850


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l498_49817

theorem nested_fraction_equality : 
  (2 : ℚ) / (2 + 2 / (3 + 1 / 4)) = 13 / 17 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l498_49817


namespace NUMINAMATH_CALUDE_ball_count_after_50_moves_l498_49845

/-- Represents the state of the boxes --/
structure BoxState :=
  (A : ℕ)
  (B : ℕ)
  (C : ℕ)
  (D : ℕ)

/-- Performs one iteration of the ball-moving process --/
def moveOnce (state : BoxState) : BoxState :=
  sorry

/-- Performs n iterations of the ball-moving process --/
def moveNTimes (n : ℕ) (state : BoxState) : BoxState :=
  sorry

/-- The initial state of the boxes --/
def initialState : BoxState :=
  { A := 8, B := 6, C := 3, D := 1 }

theorem ball_count_after_50_moves :
  (moveNTimes 50 initialState).A = 6 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_after_50_moves_l498_49845


namespace NUMINAMATH_CALUDE_polar_curve_is_line_and_circle_l498_49847

/-- The curve represented by the polar equation ρsin(θ) = sin(2θ) -/
def polar_curve (ρ θ : ℝ) : Prop :=
  ρ * Real.sin θ = Real.sin (2 * θ)

/-- The line part of the curve -/
def line_part (x y : ℝ) : Prop :=
  y = 0

/-- The circle part of the curve -/
def circle_part (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1

/-- Theorem stating that the polar curve consists of a line and a circle -/
theorem polar_curve_is_line_and_circle :
  ∀ ρ θ x y : ℝ, polar_curve ρ θ → 
  (∃ ρ' θ', x = ρ' * Real.cos θ' ∧ y = ρ' * Real.sin θ') →
  (line_part x y ∨ circle_part x y) :=
sorry

end NUMINAMATH_CALUDE_polar_curve_is_line_and_circle_l498_49847


namespace NUMINAMATH_CALUDE_intersection_value_l498_49887

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The equation of the first line -/
def line1 (a c : ℝ) (x y : ℝ) : Prop := a * x - 3 * y = c

/-- The equation of the second line -/
def line2 (b c : ℝ) (x y : ℝ) : Prop := 3 * x + b * y = -c

/-- The theorem stating that c = 39 given the conditions -/
theorem intersection_value (a b c : ℝ) : 
  perpendicular (a / 3) (-3 / b) →
  line1 a c 2 (-3) →
  line2 b c 2 (-3) →
  c = 39 := by sorry

end NUMINAMATH_CALUDE_intersection_value_l498_49887


namespace NUMINAMATH_CALUDE_max_area_rectangle_l498_49895

def is_valid_rectangle (x y : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ x ≥ y

def cost (x y : ℕ) : ℕ :=
  2 * (3 * x + 5 * y)

def area (x y : ℕ) : ℕ :=
  x * y

theorem max_area_rectangle :
  ∃ (x y : ℕ), is_valid_rectangle x y ∧ cost x y ≤ 100 ∧
  area x y = 40 ∧
  ∀ (a b : ℕ), is_valid_rectangle a b → cost a b ≤ 100 → area a b ≤ 40 :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l498_49895


namespace NUMINAMATH_CALUDE_time_after_1007_hours_l498_49830

def clock_add (current_time hours_elapsed : ℕ) : ℕ :=
  (current_time + hours_elapsed) % 12

theorem time_after_1007_hours :
  let current_time := 5
  let hours_elapsed := 1007
  clock_add current_time hours_elapsed = 4 := by
sorry

end NUMINAMATH_CALUDE_time_after_1007_hours_l498_49830


namespace NUMINAMATH_CALUDE_trioball_playing_time_l498_49898

theorem trioball_playing_time (total_children : ℕ) (playing_children : ℕ) (total_time : ℕ) 
  (h1 : total_children = 6)
  (h2 : playing_children = 3)
  (h3 : total_time = 180) :
  (total_time * playing_children) / total_children = 90 := by
  sorry

end NUMINAMATH_CALUDE_trioball_playing_time_l498_49898


namespace NUMINAMATH_CALUDE_ellipse_semi_minor_axis_l498_49814

/-- Given an ellipse with specified center, focus, and endpoint of semi-major axis,
    prove that its semi-minor axis has length √3 -/
theorem ellipse_semi_minor_axis 
  (center : ℝ × ℝ) 
  (focus : ℝ × ℝ) 
  (semi_major_endpoint : ℝ × ℝ) 
  (h1 : center = (-3, 1)) 
  (h2 : focus = (-3, 0)) 
  (h3 : semi_major_endpoint = (-3, 3)) : 
  Real.sqrt ((center.2 - semi_major_endpoint.2)^2 - (center.2 - focus.2)^2) = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ellipse_semi_minor_axis_l498_49814


namespace NUMINAMATH_CALUDE_notebook_duration_example_l498_49832

/-- The number of days notebooks last given the number of notebooks, pages per notebook, and daily page usage. -/
def notebook_duration (num_notebooks : ℕ) (pages_per_notebook : ℕ) (pages_per_day : ℕ) : ℕ :=
  (num_notebooks * pages_per_notebook) / pages_per_day

/-- Theorem stating that 5 notebooks with 40 pages each, used at a rate of 4 pages per day, last for 50 days. -/
theorem notebook_duration_example : notebook_duration 5 40 4 = 50 := by
  sorry

end NUMINAMATH_CALUDE_notebook_duration_example_l498_49832


namespace NUMINAMATH_CALUDE_cos_2x_minus_pi_6_eq_sin_2x_plus_pi_3_l498_49856

theorem cos_2x_minus_pi_6_eq_sin_2x_plus_pi_3 (x : ℝ) :
  Real.cos (2 * x - π / 6) = Real.sin (2 * (x + π / 6)) := by
  sorry

end NUMINAMATH_CALUDE_cos_2x_minus_pi_6_eq_sin_2x_plus_pi_3_l498_49856


namespace NUMINAMATH_CALUDE_middleton_marching_band_max_members_l498_49846

theorem middleton_marching_band_max_members :
  ∀ n : ℕ,
  (30 * n % 21 = 9) →
  (30 * n < 1500) →
  (∀ m : ℕ, (30 * m % 21 = 9) → (30 * m < 1500) → (30 * m ≤ 30 * n)) →
  30 * n = 1470 :=
by sorry

end NUMINAMATH_CALUDE_middleton_marching_band_max_members_l498_49846


namespace NUMINAMATH_CALUDE_volume_of_cut_cone_l498_49874

/-- The volume of the cone cut to form a frustum, given the frustum's properties -/
theorem volume_of_cut_cone (r R h H : ℝ) : 
  (R = 3 * r) →  -- Area of one base is 9 times the other
  (H = 3 * h) →  -- Height ratio follows from radius ratio
  (π * R^2 * H / 3 - π * r^2 * h / 3 = 52) →  -- Volume of frustum is 52
  (π * r^2 * h / 3 = 54) :=  -- Volume of cut cone is 54
by sorry

end NUMINAMATH_CALUDE_volume_of_cut_cone_l498_49874


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l498_49870

theorem matrix_equation_solution :
  ∀ (M : Matrix (Fin 2) (Fin 2) ℝ),
  M^3 - 5 • M^2 + 6 • M = !![16, 8; 24, 12] →
  M = !![4, 2; 6, 3] := by
sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l498_49870


namespace NUMINAMATH_CALUDE_triangle_property_l498_49826

theorem triangle_property (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  A < π/2 ∧ B < π/2 ∧ C < π/2 →
  S > 0 →
  S = (1/2) * b * c * Real.sin A →
  (b - c) * Real.sin B = b * Real.sin (A - C) →
  A = π/3 ∧ 
  4 * Real.sqrt 3 ≤ (a^2 + b^2 + c^2) / S ∧ 
  (a^2 + b^2 + c^2) / S < 16 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_property_l498_49826


namespace NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l498_49861

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_small_prime_factors (n : ℕ) : Prop :=
  ∀ p, Nat.Prime p → p < 15 → ¬(p ∣ n)

theorem smallest_composite_no_small_factors :
  is_composite 289 ∧
  has_no_small_prime_factors 289 ∧
  ∀ m, m < 289 → ¬(is_composite m ∧ has_no_small_prime_factors m) :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l498_49861


namespace NUMINAMATH_CALUDE_property_price_calculation_l498_49899

/-- The price of the property in dollars given the price per square foot, house size, and barn size. -/
def property_price (price_per_sqft : ℚ) (house_size : ℚ) (barn_size : ℚ) : ℚ :=
  price_per_sqft * (house_size + barn_size)

/-- Theorem stating that the property price is $333,200 given the specified conditions. -/
theorem property_price_calculation :
  property_price 98 2400 1000 = 333200 := by
  sorry

end NUMINAMATH_CALUDE_property_price_calculation_l498_49899


namespace NUMINAMATH_CALUDE_smallest_y_for_cube_l498_49860

theorem smallest_y_for_cube (y : ℕ+) (M : ℤ) : 
  (∀ k : ℕ+, k < y → ¬∃ N : ℤ, 2520 * k = N^3) → 
  (∃ N : ℤ, 2520 * y = N^3) → 
  y = 3675 := by
sorry

end NUMINAMATH_CALUDE_smallest_y_for_cube_l498_49860


namespace NUMINAMATH_CALUDE_parabola_intersection_value_l498_49823

theorem parabola_intersection_value (m : ℝ) : 
  (m^2 - m - 1 = 0) → (m^2 - m + 2008 = 2009) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_value_l498_49823


namespace NUMINAMATH_CALUDE_eraser_price_correct_l498_49825

/-- The price of an eraser given the following conditions:
  1. 3 erasers and 5 pencils cost 10.6 yuan
  2. 4 erasers and 4 pencils cost 12 yuan -/
def eraser_price : ℝ := 2.2

/-- The price of a pencil (to be determined) -/
def pencil_price : ℝ := sorry

theorem eraser_price_correct :
  3 * eraser_price + 5 * pencil_price = 10.6 ∧
  4 * eraser_price + 4 * pencil_price = 12 :=
by sorry

end NUMINAMATH_CALUDE_eraser_price_correct_l498_49825


namespace NUMINAMATH_CALUDE_not_prime_sum_minus_one_l498_49865

theorem not_prime_sum_minus_one (m n : ℤ) 
  (hm : m > 1) 
  (hn : n > 1) 
  (h_divides : (m + n - 1) ∣ (m^2 + n^2 - 1)) : 
  ¬(Nat.Prime (m + n - 1).natAbs) := by
sorry

end NUMINAMATH_CALUDE_not_prime_sum_minus_one_l498_49865


namespace NUMINAMATH_CALUDE_exchange_result_l498_49837

/-- Represents the number of exchanges between Xiao Zhang and Xiao Li -/
def exchanges : ℕ := 4

/-- Xiao Zhang's initial number of pencils -/
def zhang_initial_pencils : ℕ := 200

/-- Xiao Li's initial number of pens -/
def li_initial_pens : ℕ := 20

/-- Number of pencils Xiao Zhang gives in each exchange -/
def pencils_per_exchange : ℕ := 6

/-- Number of pens Xiao Li gives in each exchange -/
def pens_per_exchange : ℕ := 1

/-- Xiao Zhang's pencils after exchanges -/
def zhang_final_pencils : ℕ := zhang_initial_pencils - exchanges * pencils_per_exchange

/-- Xiao Li's pens after exchanges -/
def li_final_pens : ℕ := li_initial_pens - exchanges * pens_per_exchange

theorem exchange_result : zhang_final_pencils = 11 * li_final_pens := by
  sorry

end NUMINAMATH_CALUDE_exchange_result_l498_49837


namespace NUMINAMATH_CALUDE_problem_clock_integer_distances_l498_49882

/-- Represents a clock with specified hand lengths -/
structure Clock where
  hour_hand_length : ℝ
  minute_hand_length : ℝ

/-- Calculates the number of integer distance occurrences for a given clock over a 12-hour period -/
def integer_distance_occurrences (c : Clock) : ℕ :=
  sorry

/-- The specific clock described in the problem -/
def problem_clock : Clock :=
  { hour_hand_length := 3
  , minute_hand_length := 4 }

/-- Theorem stating the number of integer distance occurrences for the problem clock -/
theorem problem_clock_integer_distances :
  integer_distance_occurrences problem_clock = 132 :=
sorry

end NUMINAMATH_CALUDE_problem_clock_integer_distances_l498_49882


namespace NUMINAMATH_CALUDE_complex_in_third_quadrant_l498_49866

def complex_number (x : ℝ) : ℂ := Complex.mk (x^2 - 6*x + 5) (x - 2)

def in_third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

theorem complex_in_third_quadrant (x : ℝ) :
  in_third_quadrant (complex_number x) ↔ 1 < x ∧ x < 2 := by sorry

end NUMINAMATH_CALUDE_complex_in_third_quadrant_l498_49866


namespace NUMINAMATH_CALUDE_triangles_in_200_sided_polygon_l498_49890

/-- The number of sides in the regular polygon -/
def n : ℕ := 200

/-- The number of vertices to select for each triangle -/
def k : ℕ := 3

/-- The number of triangles that can be formed from a regular n-sided polygon -/
def num_triangles (n : ℕ) : ℕ := Nat.choose n k

theorem triangles_in_200_sided_polygon :
  num_triangles n = 1313400 := by
  sorry

end NUMINAMATH_CALUDE_triangles_in_200_sided_polygon_l498_49890


namespace NUMINAMATH_CALUDE_smallest_value_for_x_between_zero_and_one_l498_49876

theorem smallest_value_for_x_between_zero_and_one (x : ℝ) (h : 0 < x ∧ x < 1) :
  x^3 < x^2 ∧ x^2 < x ∧ x < 2*x ∧ 2*x < 3*x := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_for_x_between_zero_and_one_l498_49876


namespace NUMINAMATH_CALUDE_train_boarding_probability_l498_49853

theorem train_boarding_probability 
  (cycle_time : ℝ) 
  (favorable_window : ℝ) 
  (h1 : cycle_time = 5) 
  (h2 : favorable_window = 0.5) 
  (h3 : 0 < favorable_window) 
  (h4 : favorable_window < cycle_time) :
  (favorable_window / cycle_time) = (1 / 10) := by
sorry

end NUMINAMATH_CALUDE_train_boarding_probability_l498_49853


namespace NUMINAMATH_CALUDE_exam_duration_l498_49896

/-- Proves that the examination time is 30 hours given the specified conditions -/
theorem exam_duration (total_questions : ℕ) (type_a_questions : ℕ) (type_a_time : ℝ) :
  total_questions = 200 →
  type_a_questions = 10 →
  type_a_time = 17.142857142857142 →
  (total_questions - type_a_questions) * (type_a_time / 2) + type_a_questions * type_a_time = 30 * 60 := by
  sorry

end NUMINAMATH_CALUDE_exam_duration_l498_49896


namespace NUMINAMATH_CALUDE_no_real_roots_composition_l498_49816

/-- Given a quadratic function f(x) = ax^2 + bx + c where a ≠ 0,
    if f(x) = x has no real roots, then f(f(x)) = x has no real roots -/
theorem no_real_roots_composition (a b c : ℝ) (ha : a ≠ 0) :
  ((b - 1)^2 - 4*a*c < 0) →
  (a^2 * ((b + 1)^2 - 4*(a*c + b + 1)) < 0) :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_composition_l498_49816


namespace NUMINAMATH_CALUDE_problem_statement_l498_49859

open Real

noncomputable def f (x : ℝ) := log x
noncomputable def g (a : ℝ) (x : ℝ) := a * (x - 1) / (x + 1)
noncomputable def h (a : ℝ) (x : ℝ) := f x - g a x

theorem problem_statement :
  (∀ x > 1, f x > g 2 x) ∧
  (∀ a ≤ 2, StrictMono (h a)) ∧
  (∀ a > 2, ∃ x y, x < y ∧ IsLocalMax (h a) x ∧ IsLocalMin (h a) y) ∧
  (∀ x > 0, f (x + 1) > x^2 / (exp x - 1)) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l498_49859


namespace NUMINAMATH_CALUDE_nina_running_distance_l498_49872

theorem nina_running_distance :
  let d1 : ℚ := 0.08333333333333333
  let d2 : ℚ := 0.08333333333333333
  let d3 : ℚ := 0.6666666666666666
  d1 + d2 + d3 = 0.8333333333333333 := by sorry

end NUMINAMATH_CALUDE_nina_running_distance_l498_49872


namespace NUMINAMATH_CALUDE_second_discount_percentage_l498_49819

theorem second_discount_percentage (original_price : ℝ) (first_discount : ℝ) (final_price : ℝ) : 
  original_price = 400 →
  first_discount = 10 →
  final_price = 331.2 →
  ∃ (second_discount : ℝ),
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) ∧
    second_discount = 8 :=
by sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l498_49819


namespace NUMINAMATH_CALUDE_log_equation_solution_l498_49842

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x^3 / Real.log 3 + Real.log x / Real.log (1/3) = 6 → x = 27 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l498_49842


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l498_49871

open Set

theorem complement_intersection_theorem (M N : Set ℝ) :
  M = {x | x > 1} →
  N = {x | |x| ≤ 2} →
  (𝓤 \ M) ∩ N = Icc (-2) 1 := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l498_49871


namespace NUMINAMATH_CALUDE_weight_loss_challenge_l498_49808

theorem weight_loss_challenge (initial_weight : ℝ) (clothes_weight_percentage : ℝ) 
  (h1 : clothes_weight_percentage > 0) 
  (h2 : initial_weight > 0) : 
  (0.85 * initial_weight + clothes_weight_percentage * 0.85 * initial_weight) / initial_weight = 0.867 → 
  clothes_weight_percentage = 0.02 := by
sorry

end NUMINAMATH_CALUDE_weight_loss_challenge_l498_49808


namespace NUMINAMATH_CALUDE_j_mod_2_not_zero_l498_49878

theorem j_mod_2_not_zero (x j : ℤ) (h : 2 * x - j = 11) : j % 2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_j_mod_2_not_zero_l498_49878


namespace NUMINAMATH_CALUDE_u_5_value_l498_49803

def sequence_u (u : ℕ → ℝ) : Prop :=
  ∀ n, u (n + 2) = 3 * u (n + 1) + 2 * u n

theorem u_5_value (u : ℕ → ℝ) (h : sequence_u u) (h3 : u 3 = 10) (h6 : u 6 = 256) :
  u 5 = 808 / 11 := by
  sorry

end NUMINAMATH_CALUDE_u_5_value_l498_49803
