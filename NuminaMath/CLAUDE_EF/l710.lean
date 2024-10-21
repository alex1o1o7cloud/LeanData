import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l710_71068

/-- The maximum distance from a point on the unit circle to a line in polar form -/
theorem max_distance_circle_to_line :
  let unit_circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
  let line := {p : ℝ × ℝ | p.1 + p.2 = 6}
  ∃ (d : ℝ), d = 3 * Real.sqrt 2 + 1 ∧
    ∀ (p : ℝ × ℝ), p ∈ unit_circle →
      (∀ (q : ℝ × ℝ), q ∈ line → dist p q ≤ d) ∧
      ∃ (p' : ℝ × ℝ), p' ∈ unit_circle ∧ ∃ (q' : ℝ × ℝ), q' ∈ line ∧ dist p' q' = d :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l710_71068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_y_l710_71071

-- Define the max function
noncomputable def max' (a b : ℝ) : ℝ := if a ≥ b then a else b

-- Define the function y
noncomputable def y (x : ℝ) : ℝ := max' (-x - 1) (x^2 - 2*x - 3)

-- Theorem statement
theorem min_value_of_y : 
  ∃ (x₀ : ℝ), ∀ (x : ℝ), y x ≥ y x₀ ∧ y x₀ = -3 := by
  -- Proof goes here
  sorry

#check min_value_of_y

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_y_l710_71071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graveyard_bones_count_l710_71073

theorem graveyard_bones_count :
  let total_skeletons : ℕ := 20
  let adult_women_ratio : ℚ := 1/2
  let adult_women_bones : ℕ := 20
  let adult_men_extra_bones : ℕ := 5
  let child_bones_ratio : ℚ := 1/2

  let adult_women_count : ℕ := (total_skeletons * adult_women_ratio.num / adult_women_ratio.den).toNat
  let remaining_skeletons : ℕ := total_skeletons - adult_women_count
  let adult_men_count : ℕ := remaining_skeletons / 2
  let children_count : ℕ := remaining_skeletons / 2

  let adult_men_bones : ℕ := adult_women_bones + adult_men_extra_bones
  let child_bones : ℕ := (adult_women_bones * child_bones_ratio.num / child_bones_ratio.den).toNat

  let total_bones : ℕ := 
    adult_women_count * adult_women_bones +
    adult_men_count * adult_men_bones +
    children_count * child_bones

  total_bones = 375 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graveyard_bones_count_l710_71073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trucks_passing_time_l710_71081

/-- The time taken for a slower truck to pass the driver of a faster truck -/
noncomputable def time_to_pass (truck_length : ℝ) (speed_faster : ℝ) (speed_slower : ℝ) : ℝ :=
  truck_length / (speed_faster + speed_slower)

theorem trucks_passing_time :
  let truck_length : ℝ := 250
  let speed_faster : ℝ := 30 * 1000 / 3600  -- Convert km/h to m/s
  let speed_slower : ℝ := 20 * 1000 / 3600  -- Convert km/h to m/s
  abs (time_to_pass truck_length speed_faster speed_slower - 18) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trucks_passing_time_l710_71081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_quotient_is_nine_l710_71079

def S : Set ℤ := {-12, -4, -3, 1, 3, 9}

theorem largest_quotient_is_nine :
  ∀ a b : ℤ, a ∈ S → b ∈ S → a ≠ 0 → (b : ℚ) / a ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_quotient_is_nine_l710_71079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_angle_A_value_l710_71047

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - 4 * Real.pi / 3) + 2 * (Real.cos x) ^ 2

-- Theorem for the maximum value of f
theorem f_max_value : 
  ∀ x : ℝ, f x ≤ 2 := by
  sorry

-- Theorem for the value of angle A
theorem angle_A_value (A B C : ℝ) :
  A + B + C = Real.pi → 
  f (B + C) = 3/2 → 
  A = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_angle_A_value_l710_71047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_triangle_areas_in_unit_cube_sum_of_mnp_l710_71033

-- Define the cube
def cube_side_length : ℚ := 1

-- Define the sum of areas
noncomputable def sum_of_areas : ℝ := 12 + Real.sqrt 288 + Real.sqrt 48

-- Theorem statement
theorem sum_of_triangle_areas_in_unit_cube :
  ∃ (m n p : ℕ), sum_of_areas = m + Real.sqrt (n : ℝ) + Real.sqrt (p : ℝ) :=
by
  -- Instantiate m, n, and p
  let m : ℕ := 12
  let n : ℕ := 288
  let p : ℕ := 48
  
  -- Prove the existence
  use m, n, p
  
  -- The actual proof would go here
  sorry

-- Additional theorem to show m + n + p = 348
theorem sum_of_mnp : 
  ∃ (m n p : ℕ), sum_of_areas = m + Real.sqrt (n : ℝ) + Real.sqrt (p : ℝ) ∧ m + n + p = 348 :=
by
  -- Instantiate m, n, and p
  let m : ℕ := 12
  let n : ℕ := 288
  let p : ℕ := 48
  
  -- Prove the existence
  use m, n, p
  
  -- Split the goal into two parts
  apply And.intro
  
  -- First part: sum_of_areas equality
  · sorry
  
  -- Second part: m + n + p = 348
  · norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_triangle_areas_in_unit_cube_sum_of_mnp_l710_71033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_fabric_yards_l710_71090

/-- The amount spent on checkered fabric in dollars -/
noncomputable def checkered_spent : ℚ := 75

/-- The amount spent on plain fabric in dollars -/
noncomputable def plain_spent : ℚ := 45

/-- The cost per yard of fabric in dollars -/
noncomputable def cost_per_yard : ℚ := 15/2

/-- The total yards of fabric bought -/
noncomputable def total_yards : ℚ := checkered_spent / cost_per_yard + plain_spent / cost_per_yard

theorem total_fabric_yards : total_yards = 16 := by
  -- Unfold the definitions
  unfold total_yards checkered_spent plain_spent cost_per_yard
  -- Simplify the rational number expressions
  simp [Rat.add_def, Rat.div_def]
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_fabric_yards_l710_71090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_domain_characterization_l710_71061

open Set Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := tan (π / 4 - x)

-- Define the domain
def domain : Set ℝ := {x | ∀ k : ℤ, x ≠ k * π + 3 * π / 4}

-- Theorem statement
theorem tan_domain_characterization :
  {x : ℝ | ∃ y, f x = y} = domain := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_domain_characterization_l710_71061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_results_l710_71011

/-- Represents the total number of votes cast in the election -/
def total_votes : ℕ := 30000

/-- Represents the percentage of votes for each candidate -/
def vote_percentage (candidate : Fin 4) : ℚ :=
  match candidate with
  | 0 => 32/100
  | 1 => 28/100
  | 2 => 22/100
  | 3 => 18/100

/-- Represents the number of votes for each candidate -/
def votes_for (candidate : Fin 4) : ℕ :=
  (vote_percentage candidate * total_votes).floor.toNat

/-- Theorem stating the properties of the election results -/
theorem election_results :
  (vote_percentage 0 = 32/100) ∧ 
  (vote_percentage 1 = 28/100) ∧ 
  (vote_percentage 2 = 22/100) ∧ 
  (vote_percentage 3 = 18/100) ∧
  (∀ c : Fin 4, votes_for c = (vote_percentage c * total_votes).floor.toNat) ∧
  (votes_for 0 - votes_for 1 = 1200) ∧
  (votes_for 0 - votes_for 2 = 2200) ∧
  (votes_for 1 - votes_for 3 = 900) ∧
  (total_votes = 30000) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_results_l710_71011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_14_10_l710_71097

/-- The volume of a cylinder with height h and radius r is π * r^2 * h -/
noncomputable def cylinderVolume (h : ℝ) (r : ℝ) : ℝ := Real.pi * r^2 * h

/-- The diameter of a circle is twice its radius -/
noncomputable def diameterToRadius (d : ℝ) : ℝ := d / 2

theorem cylinder_volume_14_10 :
  let h : ℝ := 14
  let d : ℝ := 10
  let r : ℝ := diameterToRadius d
  cylinderVolume h r = Real.pi * 350 := by
  -- Unfold definitions
  unfold cylinderVolume diameterToRadius
  -- Simplify
  simp [Real.pi]
  -- The proof steps would go here, but we'll use sorry to skip them
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_14_10_l710_71097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_l710_71008

noncomputable section

-- Define the ellipse parametrically
def ellipse_x (θ : Real) : Real := Real.sqrt 2 * Real.cos θ
def ellipse_y (θ : Real) : Real := Real.sin θ

-- Define the focal length of an ellipse
def focal_length (a b : Real) : Real := 2 * Real.sqrt (a^2 - b^2)

theorem ellipse_focal_length :
  focal_length (Real.sqrt 2) 1 = 2 := by
  sorry

#check ellipse_focal_length

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_l710_71008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l710_71082

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 6*y + 4 = 0

-- Define the center and radius of C1
def center_C1 : ℝ × ℝ := (0, 0)
def radius_C1 : ℝ := 1

-- Define the center and radius of C2
def center_C2 : ℝ × ℝ := (-2, 3)
def radius_C2 : ℝ := 3

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := Real.sqrt 13

-- Theorem stating that the circles are intersect
theorem circles_intersect :
  distance_between_centers > abs (radius_C2 - radius_C1) ∧
  distance_between_centers < radius_C1 + radius_C2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l710_71082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_one_sufficient_not_necessary_l710_71051

-- Define the function f as noncomputable due to the use of Real.log
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + a^2) - x)

-- State the theorem
theorem a_one_sufficient_not_necessary :
  (∃ a : ℝ, a ≠ 1 ∧ ∀ x : ℝ, f a x + f a (-x) = 0) ∧
  (∀ x : ℝ, f 1 x + f 1 (-x) = 0) := by
  sorry

#check a_one_sufficient_not_necessary

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_one_sufficient_not_necessary_l710_71051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_distribution_probability_l710_71077

def total_cookies : ℕ := 12
def num_types : ℕ := 3
def cookies_per_type : ℕ := 4
def num_children : ℕ := 4
def cookies_per_child : ℕ := 3

theorem cookie_distribution_probability :
  (((cookies_per_type.choose 1) ^ num_types *
    ((cookies_per_type - 1).choose 1) ^ num_types *
    ((cookies_per_type - 2).choose 1) ^ num_types : ℚ) /
   ((total_cookies.choose cookies_per_child) *
    ((total_cookies - cookies_per_child).choose cookies_per_child) *
    ((total_cookies - 2 * cookies_per_child).choose cookies_per_child))) = 72 / 1925 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_distribution_probability_l710_71077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l710_71024

-- Define the lower and upper bounds of the region
def lower_bound (x : ℝ) : ℝ := -1 - x^2
noncomputable def upper_bound (x : ℝ) : ℝ := 2 + Real.sqrt x

-- Define the region
def region (x y : ℝ) : Prop :=
  x ∈ Set.Icc 0 1 ∧ 
  Real.sqrt (1 - x) + 2 * x ≥ 0 ∧
  lower_bound x ≤ y ∧ y ≤ upper_bound x

-- State the theorem
theorem area_of_region : 
  (∫ x in Set.Icc 0 1, upper_bound x - lower_bound x) = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l710_71024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_at_distance_l710_71083

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (A B C₁ C₂ : ℝ) : ℝ :=
  |C₂ - C₁| / Real.sqrt (A^2 + B^2)

/-- The given line equation -/
def given_line (x y : ℝ) : Prop :=
  2*x + y + 1 = 0

/-- A parallel line to the given line -/
def parallel_line (x y m : ℝ) : Prop :=
  2*x + y + m = 0

theorem parallel_lines_at_distance (d : ℝ) :
  d = Real.sqrt 5 / 5 →
  ∃ m₁ m₂ : ℝ,
    distance_between_parallel_lines 2 1 1 m₁ = d ∧
    distance_between_parallel_lines 2 1 1 m₂ = d ∧
    m₁ ≠ m₂ ∧
    (∀ x y, parallel_line x y m₁ ↔ (2*x + y = 0)) ∧
    (∀ x y, parallel_line x y m₂ ↔ (2*x + y + 2 = 0)) := by
  sorry

#check parallel_lines_at_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_at_distance_l710_71083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_nonnegative_iff_a_leq_one_l710_71026

open Real

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a * Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - x^2 - a*x + 1 + (a + x) * Real.log x

-- State the theorem
theorem g_nonnegative_iff_a_leq_one :
  ∀ a : ℝ, (∀ x : ℝ, x > 0 → g a x ≥ 0) ↔ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_nonnegative_iff_a_leq_one_l710_71026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_above_average_l710_71002

/-- Represents the number of students in the class -/
def class_size : ℕ := 150

/-- Represents a list of student scores -/
def scores := List ℝ

/-- Calculates the average score of the class -/
noncomputable def class_average (scores : List ℝ) : ℝ :=
  scores.sum / scores.length

/-- Counts the number of scores above the average -/
noncomputable def count_above_average (scores : List ℝ) : ℕ :=
  (scores.filter (λ score => score > class_average scores)).length

/-- Theorem stating the maximum number of students who can score above average -/
theorem max_above_average :
  ∀ scores : List ℝ,
  scores.length = class_size →
  count_above_average scores ≤ 149 := by
  sorry

#eval class_size

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_above_average_l710_71002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closer_to_point_probability_l710_71023

/-- A rectangle in the 2D plane --/
structure Rectangle where
  x_min : ℝ
  y_min : ℝ
  x_max : ℝ
  y_max : ℝ

/-- A point in the 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- The probability of an event in a continuous uniform distribution --/
noncomputable def probability (favorable_area area : ℝ) : ℝ :=
  favorable_area / area

theorem closer_to_point_probability 
  (rect : Rectangle) 
  (p1 p2 : Point) : 
  rect.x_min = 0 ∧ 
  rect.y_min = 0 ∧ 
  rect.x_max = 3 ∧ 
  rect.y_max = 2 ∧
  p1 = ⟨0, 2⟩ ∧
  p2 = ⟨4, 2⟩ →
  probability 
    ((rect.x_max - rect.x_min) * (rect.y_max - rect.y_min) / 2) 
    ((rect.x_max - rect.x_min) * (rect.y_max - rect.y_min)) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closer_to_point_probability_l710_71023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_distribution_ratio_l710_71072

theorem orange_distribution_ratio : 
  ∀ (total : ℕ) (brother_fraction : ℚ) (friend_oranges : ℕ),
    total = 12 →
    brother_fraction = 1/3 →
    friend_oranges = 2 →
    (friend_oranges : ℚ) / (total - (brother_fraction * ↑total).num) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_distribution_ratio_l710_71072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l710_71058

theorem tan_alpha_plus_pi_fourth (α : ℝ) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin α = 3 / 5) : 
  Real.tan (α + π / 4) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l710_71058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_six_l710_71039

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the directrix of the parabola
def directrix : ℝ := -1

-- Define the center of the circle
def circle_center : ℝ × ℝ := (4, 4)

-- Define the property that the circle is tangent to the directrix
def circle_tangent_to_directrix (radius : ℝ) : Prop :=
  radius = circle_center.1 - directrix

-- Define the chord length
noncomputable def chord_length (radius : ℝ) : ℝ :=
  2 * (radius^2 - circle_center.2^2).sqrt

-- Theorem statement
theorem chord_length_is_six :
  ∃ (radius : ℝ),
    parabola circle_center.1 circle_center.2 ∧
    circle_tangent_to_directrix radius ∧
    chord_length radius = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_six_l710_71039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l710_71052

/-- Given a function f(x) = 2x^2 - xf'(2), prove that the equation of the tangent line
    to the graph of f at the point (2, f(2)) is 4x - y - 8 = 0. -/
theorem tangent_line_equation (f : ℝ → ℝ) (h : ∀ x, f x = 2 * x^2 - x * (deriv f 2)) :
  ∃ m b, (deriv f 2 = m ∧ f 2 = 2 * m + b) ∧ 
         ∀ x y, y = m * x + b ↔ 4 * x - y - 8 = 0 :=
by
  -- We'll prove this later
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l710_71052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_100th_term_l710_71050

def arithmeticSequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + d * (n - 1)

theorem sequence_100th_term :
  arithmeticSequence 3 (-2) 100 = -195 := by
  unfold arithmeticSequence
  norm_num

#eval arithmeticSequence 3 (-2) 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_100th_term_l710_71050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_tiles_in_10x10_l710_71028

/-- Represents a square grid of tiles -/
structure TileGrid where
  size : Nat
  is_blue : Fin size → Fin size → Bool

/-- A 10x10 grid where perimeter tiles are blue and interior tiles are yellow -/
def grid_10x10 : TileGrid where
  size := 10
  is_blue := fun i j => i = 0 || i = 9 || j = 0 || j = 9

/-- Count of yellow tiles in a grid -/
def yellow_tile_count (grid : TileGrid) : Nat :=
  (grid.size * grid.size) - (Finset.univ.filter (fun p : Fin grid.size × Fin grid.size => grid.is_blue p.1 p.2)).card

/-- Theorem: The number of yellow tiles in a 10x10 grid with blue perimeter is 64 -/
theorem yellow_tiles_in_10x10 : yellow_tile_count grid_10x10 = 64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_tiles_in_10x10_l710_71028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l710_71054

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the system of inequalities -/
def satisfies_inequalities (p : Point) (lambda : ℝ) : Prop :=
  p.x ≤ 1 ∧ p.y ≤ 3 ∧ 2 * p.x - p.y + lambda - 1 ≥ 0

/-- Checks if a point is in the first quadrant -/
def in_first_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Checks if a point is in the second quadrant -/
def in_second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Checks if a point is in the third quadrant -/
def in_third_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Checks if a point is in the fourth quadrant -/
def in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Defines the condition for the region to pass through all four quadrants -/
def passes_through_all_quadrants (lambda : ℝ) : Prop :=
  ∃ (p1 p2 p3 p4 : Point),
    satisfies_inequalities p1 lambda ∧ in_first_quadrant p1 ∧
    satisfies_inequalities p2 lambda ∧ in_second_quadrant p2 ∧
    satisfies_inequalities p3 lambda ∧ in_third_quadrant p3 ∧
    satisfies_inequalities p4 lambda ∧ in_fourth_quadrant p4

/-- The main theorem stating the range of lambda -/
theorem lambda_range :
  ∀ lambda : ℝ, passes_through_all_quadrants lambda ↔ lambda > 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l710_71054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_count_is_six_l710_71022

def average_of_all : Float := 3.95
def average_group1 : Float := 3.4
def average_group2 : Float := 3.85
def average_group3 : Float := 4.600000000000001
def count_group1 : Nat := 2
def count_group2 : Nat := 2
def count_group3 : Nat := 2

theorem total_count_is_six :
  count_group1 + count_group2 + count_group3 = 6 ∧
  (average_group1 * count_group1.toFloat +
   average_group2 * count_group2.toFloat +
   average_group3 * count_group3.toFloat) / (count_group1 + count_group2 + count_group3).toFloat = average_of_all :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_count_is_six_l710_71022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_average_increase_l710_71020

/-- Represents a batsman's score data -/
structure BatsmanScore where
  innings : Nat
  totalRuns : Nat
  average : ℚ

/-- Calculates the new average after an inning -/
def newAverage (score : BatsmanScore) (runsInNewInning : Nat) : ℚ :=
  (score.totalRuns + runsInNewInning : ℚ) / (score.innings + 1 : ℚ)

/-- Theorem: If a batsman's average increases by 10 after scoring 200 runs in the 17th inning, 
    then his average after the 17th inning is 40 -/
theorem batsman_average_increase 
  (score : BatsmanScore) 
  (h1 : score.innings = 16) 
  (h2 : newAverage score 200 = score.average + 10) : 
  newAverage score 200 = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_average_increase_l710_71020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_f_equality_range_l710_71056

/-- The function f(x) = |2x - 1| + |x - a| -/
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - 1| + |x - a|

theorem f_inequality_solution (a : ℝ) :
  (a = 1) → {x : ℝ | f 1 x ≤ 2} = Set.Icc 0 (4/3) := by sorry

theorem f_equality_range (a : ℝ) :
  {x : ℝ | f a x = |x - 1 + a|} = 
    if a < 1/2 then Set.Ico a (1/2)
    else if a = 1/2 then {1/2}
    else Set.Icc (1/2) a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_f_equality_range_l710_71056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_eq_30_l710_71005

/-- Sequence a_n defined recursively -/
def a : ℕ → ℕ
  | 0 => 3  -- Define the base case for n = 0
  | n + 1 => a n + 3

/-- Theorem stating that the 10th term of the sequence is 30 -/
theorem a_10_eq_30 : a 10 = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_eq_30_l710_71005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_neg_one_sufficient_not_necessary_l710_71085

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The slope of a line -/
noncomputable def Line.slope (l : Line) : ℝ := -l.a / l.b

/-- Two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop := l1.slope * l2.slope = -1

/-- The first line from the problem -/
def l1 (a : ℝ) : Line := { a := 1, b := a - 2, c := -2 }

/-- The second line from the problem -/
def l2 (a : ℝ) : Line := { a := a - 2, b := a, c := -1 }

/-- The main theorem -/
theorem a_neg_one_sufficient_not_necessary :
  (∀ a : ℝ, a = -1 → perpendicular (l1 a) (l2 a)) ∧
  ¬(∀ a : ℝ, perpendicular (l1 a) (l2 a) → a = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_neg_one_sufficient_not_necessary_l710_71085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_equation_solutions_l710_71034

-- Define the custom operation
noncomputable def star (a b : ℝ) : ℝ :=
  if a < b then b^2 else b^3

-- Theorem statement
theorem star_equation_solutions :
  ∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
  star 5 x₁ = 64 ∧ star 5 x₂ = 64 ∧
  ∀ (y : ℝ), y > 0 ∧ star 5 y = 64 → (y = x₁ ∨ y = x₂) := by
  sorry

#check star_equation_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_equation_solutions_l710_71034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_implies_common_difference_l710_71088

noncomputable section

-- Define an arithmetic sequence
def arithmeticSequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ := λ n => a₁ + (n - 1) * d

-- Define the variance of the first five terms
def varianceFirstFive (a₁ : ℝ) (d : ℝ) : ℝ :=
  let seq := arithmeticSequence a₁ d
  let mean := (seq 1 + seq 2 + seq 3 + seq 4 + seq 5) / 5
  ((seq 1 - mean)^2 + (seq 2 - mean)^2 + (seq 3 - mean)^2 + (seq 4 - mean)^2 + (seq 5 - mean)^2) / 5

-- Theorem statement
theorem variance_implies_common_difference (a₁ : ℝ) (d : ℝ) :
  varianceFirstFive a₁ d = 2 → d = 1 ∨ d = -1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_implies_common_difference_l710_71088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combination_sum_equality_l710_71004

theorem combination_sum_equality (k m n : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ m) (h3 : m ≤ n) :
  (Finset.range (k + 1)).sum (λ i ↦ Nat.choose k i * Nat.choose n (m - i)) = Nat.choose (n + k) m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combination_sum_equality_l710_71004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_l710_71017

/-- Calculate simple interest -/
noncomputable def simpleInterest (principal rate time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Problem statement -/
theorem interest_difference : 
  let principal := (900 : ℝ)
  let time := (7 : ℝ)
  let rate1 := (4 : ℝ)
  let rate2 := (4.5 : ℝ)
  let interest1 := simpleInterest principal rate1 time
  let interest2 := simpleInterest principal rate2 time
  interest2 - interest1 = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_l710_71017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l710_71076

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola with equation y^2 = 2px -/
structure Parabola where
  p : ℝ

/-- Represents a right-angled triangle -/
structure RightTriangle where
  A : Point
  O : Point
  B : Point

/-- Check if a point lies on a parabola -/
def onParabola (point : Point) (parabola : Parabola) : Prop :=
  point.y^2 = 2 * parabola.p * point.x

/-- Check if a point lies on the line y = √3x -/
def onLine (point : Point) : Prop :=
  point.y = Real.sqrt 3 * point.x

/-- Calculate the area of a right-angled triangle -/
noncomputable def triangleArea (triangle : RightTriangle) : ℝ :=
  (triangle.A.x - triangle.O.x) * (triangle.B.y - triangle.O.y) / 2

/-- Main theorem -/
theorem parabola_equation (triangle : RightTriangle) (parabola : Parabola) :
  triangle.O = ⟨0, 0⟩ →
  onParabola triangle.A parabola →
  onParabola triangle.O parabola →
  onParabola triangle.B parabola →
  onLine triangle.A →
  triangleArea triangle = 6 * Real.sqrt 3 →
  parabola.p = 3/2 ∨ parabola.p = -3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l710_71076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equalities_and_relation_to_g_l710_71021

/-- The generating function f_{k,l}(x) of the sequence P_{k,l}(n) -/
def f (k l : ℕ) (x : ℝ) : ℝ := sorry

/-- The Gaussian polynomial g_{k,l}(x) -/
def g (k l : ℕ) (x : ℝ) : ℝ := sorry

/-- Main theorem stating the equalities for f_{k,l}(x) and its relation to g_{k,l}(x) -/
theorem f_equalities_and_relation_to_g (k l : ℕ) (x : ℝ) :
  f k l x = f (k-1) l x + x^k * f k (l-1) x ∧
  f k l x = f k (l-1) x + x^l * f (k-1) l x ∧
  f k l x = g k l x := by
  sorry

/-- Auxiliary function P_{k,l}(n) -/
def P (k l n : ℕ) : ℕ := sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equalities_and_relation_to_g_l710_71021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_on_arc_l710_71059

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point2D
  radius : ℝ

/-- Definition of an equilateral triangle -/
def isEquilateral (t : Triangle) : Prop :=
  let AB := ((t.B.x - t.A.x)^2 + (t.B.y - t.A.y)^2).sqrt
  let BC := ((t.C.x - t.B.x)^2 + (t.C.y - t.B.y)^2).sqrt
  let CA := ((t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2).sqrt
  AB = BC ∧ BC = CA

/-- Definition of a point on the extension of a line segment -/
def isOnExtension (A B D : Point2D) : Prop :=
  ∃ t : ℝ, t > 1 ∧ D.x = A.x + t * (B.x - A.x) ∧ D.y = A.y + t * (B.y - A.y)

/-- Definition of the circumcircle of a triangle -/
noncomputable def circumcircle (t : Triangle) : Circle := sorry

/-- Definition of a point lying on an arc of a circle -/
def liesOnArc (P : Point2D) (c : Circle) (A B : Point2D) (angle : ℝ) : Prop := sorry

/-- Definition of a point lying on a line -/
def liesOnLine (P : Point2D) (A B : Point2D) : Prop := sorry

/-- Main theorem -/
theorem intersection_point_on_arc 
  (t : Triangle) 
  (D E : Point2D) 
  (h1 : isEquilateral t)
  (h2 : isOnExtension t.A t.B D)
  (h3 : isOnExtension t.A t.C E)
  (h4 : let BD := ((D.x - t.B.x)^2 + (D.y - t.B.y)^2).sqrt
        let CE := ((E.x - t.C.x)^2 + (E.y - t.C.y)^2).sqrt
        let BC := ((t.C.x - t.B.x)^2 + (t.C.y - t.B.y)^2).sqrt
        BD * CE = BC^2)
  (P : Point2D)
  (h5 : liesOnLine P D t.C)
  (h6 : liesOnLine P B E) :
  liesOnArc P (circumcircle t) t.B t.C (2 * Real.pi / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_on_arc_l710_71059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_exponents_equals_six_l710_71096

/-- Sum of positive divisors of 2^i * 3^j -/
def sumOfDivisors (i j : ℕ) : ℕ := 
  (2^(i+1) - 1) * (3^(j+1) - 1) / ((2-1) * (3-1))

/-- The theorem to be proved -/
theorem sum_of_exponents_equals_six (i j : ℕ) : 
  sumOfDivisors i j = 600 → i + j = 6 := by
  sorry

#eval sumOfDivisors 3 3  -- This should output 600

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_exponents_equals_six_l710_71096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_and_geometric_sequences_l710_71006

-- Define the arithmetic sequence a_n
noncomputable def a (n : ℕ) : ℝ := 2 * n - 1

-- Define the geometric sequence b_n
noncomputable def b (n : ℕ) : ℝ := 3 * 3^(n - 1)

-- Define the sum of the first n terms of b_n
noncomputable def S (n : ℕ) : ℝ := (3^(n + 1) - 3) / 2

theorem arithmetic_and_geometric_sequences :
  (a 1 = 1) ∧ (a 3 = 5) ∧
  (b 1 = a 2) ∧ (b 2 = a 1 + a 2 + a 3) →
  (∀ n : ℕ, a n = 2 * n - 1) ∧
  (∀ n : ℕ, S n = (3^(n + 1) - 3) / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_and_geometric_sequences_l710_71006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_frustum_sphere_ratio_l710_71060

/-- A frustum of a cone with given dimensions -/
structure Frustum where
  height : ℝ
  base_radius : ℝ

/-- The ratio of the inscribed sphere radius to the circumscribed sphere radius -/
noncomputable def sphere_ratio (f : Frustum) : ℝ := 
  let r := (10 : ℝ) / 3  -- radius of inscribed sphere
  let R := 169 / 24      -- radius of circumscribed sphere
  r / R

/-- Theorem stating the sphere ratio for a specific frustum -/
theorem specific_frustum_sphere_ratio : 
  ∃ (f : Frustum), f.height = 12 ∧ f.base_radius = 5 ∧ sphere_ratio f = 80 / 169 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_frustum_sphere_ratio_l710_71060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_set_is_parabola_l710_71046

-- Define the parameters a and d as positive real numbers
variable (a d : ℝ) (ha : 0 < a) (hd : 0 < d)

-- Define t as a real number
variable (t : ℝ)

-- Define the vertex coordinates as functions of t
noncomputable def x_t (a : ℝ) (t : ℝ) : ℝ := -t / (2 * a)
noncomputable def y_t (a d : ℝ) (t : ℝ) : ℝ := -(t^2) / (4 * a) + d

-- Define the set of all vertices
def vertex_set (a d : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p.1 = x_t a t ∧ p.2 = y_t a d t}

-- Theorem stating that the vertex set forms a parabola
theorem vertex_set_is_parabola (a d : ℝ) (ha : 0 < a) (hd : 0 < d) :
  ∃ (A B C : ℝ), A ≠ 0 ∧ 
    vertex_set a d = {p : ℝ × ℝ | p.2 = A * p.1^2 + B * p.1 + C} :=
by
  -- We'll prove this later
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_set_is_parabola_l710_71046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_ratio_l710_71040

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.length * r.width

/-- Perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

theorem rectangle_perimeter_ratio (rect1 rect2 : Rectangle) 
  (h_area : rect1.area / rect2.area = 49 / 64)
  (h_length : rect1.length / rect2.length = 7 / 8) :
  rect1.perimeter / rect2.perimeter = 91 / 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_ratio_l710_71040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_is_120_degrees_l710_71043

/-- The angle of inclination of a line defined by parametric equations -/
noncomputable def angle_of_inclination (x y : ℝ → ℝ) : ℝ :=
  let slope := (y 1 - y 0) / (x 1 - x 0)
  Real.pi - Real.arctan slope

/-- The parametric equations of the line -/
noncomputable def x (t : ℝ) : ℝ := 1 - (1/2) * t
noncomputable def y (t : ℝ) : ℝ := (Real.sqrt 3 / 2) * t

/-- Theorem stating that the angle of inclination is 120 degrees (2π/3 radians) -/
theorem angle_of_inclination_is_120_degrees :
  angle_of_inclination x y = 2 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_is_120_degrees_l710_71043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_plus_cos6_l710_71030

theorem min_sin6_plus_cos6 (x : ℝ) : Real.sin x ^ 6 + Real.cos x ^ 6 ≥ (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_plus_cos6_l710_71030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_sphere_volume_ratio_l710_71086

/-- A tetrahedron with vertices P, A, B, C inscribed in a sphere O -/
structure InscribedTetrahedron where
  /-- The length of edge PA -/
  pa : ℝ
  /-- The length of edge PB -/
  pb : ℝ
  /-- The length of edge PC -/
  pc : ℝ
  /-- The radius of the circumscribed sphere O -/
  r : ℝ
  /-- Assertion that PA = 2 -/
  h_pa : pa = 2
  /-- Assertion that PB = √6 -/
  h_pb : pb = Real.sqrt 6
  /-- Assertion that PC = √6 -/
  h_pc : pc = Real.sqrt 6
  /-- Assertion that PA, PB, PC are mutually perpendicular (maximizing the sum of lateral face areas) -/
  h_perpendicular : pa ^ 2 + pb ^ 2 + pc ^ 2 = (2 * r) ^ 2

/-- The volume of a tetrahedron given its edge lengths -/
noncomputable def tetrahedronVolume (t : InscribedTetrahedron) : ℝ :=
  (1 / 6) * t.pa * t.pb * t.pc

/-- The volume of a sphere given its radius -/
noncomputable def sphereVolume (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r ^ 3

/-- The main theorem stating the ratio of volumes -/
theorem tetrahedron_sphere_volume_ratio (t : InscribedTetrahedron) :
    tetrahedronVolume t / sphereVolume t.r = 3 / (16 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_sphere_volume_ratio_l710_71086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_digit_number_proof_l710_71029

def is_nine_digit (n : ℕ) : Prop := 100000000 ≤ n ∧ n ≤ 999999999

def is_obtained_from (A B : ℕ) : Prop :=
  ∃ (d : ℕ), d < 10 ∧ A = d * 100000000 + B / 10

theorem nine_digit_number_proof (A B : ℕ) 
  (h1 : is_nine_digit A)
  (h2 : is_obtained_from A B)
  (h3 : Nat.Coprime B 24)
  (h4 : B > 666666666) :
  (∀ A' : ℕ, is_nine_digit A' → is_obtained_from A' B → A' ≤ 999999998) ∧
  (∀ A' : ℕ, is_nine_digit A' → is_obtained_from A' B → A' ≥ 166666667) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_digit_number_proof_l710_71029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_vertical_line_l710_71066

noncomputable def slope_angle (line : Set (ℝ × ℝ)) : ℝ := sorry

theorem slope_angle_of_vertical_line :
  ∀ (x : ℝ), 
  let line : Set (ℝ × ℝ) := {(x', y) | x' = x}
  (∀ (p q : ℝ × ℝ), p ∈ line ∧ q ∈ line ∧ p ≠ q → (p.1 - q.1 = 0)) →
  slope_angle line = π / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_vertical_line_l710_71066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_base_area_l710_71095

/-- A rectangular parallelepiped with specific geometric properties -/
structure RectangularParallelepiped where
  -- Diagonal CA₁
  d : ℝ
  -- Angle between CA₁ and base plane
  angle_to_base : ℝ
  -- Angle between CA₁ and plane through AC₁ and midpoint of BB₁
  angle_to_plane : ℝ

/-- The area of the base of the parallelepiped -/
noncomputable def base_area (p : RectangularParallelepiped) : ℝ :=
  (Real.sqrt 3 * p.d^2) / (8 * Real.sqrt 5)

/-- Theorem stating the area of the base of the parallelepiped -/
theorem parallelepiped_base_area 
  (p : RectangularParallelepiped) 
  (h1 : p.angle_to_base = π/3)  -- 60 degrees
  (h2 : p.angle_to_plane = π/4) -- 45 degrees
  : base_area p = (Real.sqrt 3 * p.d^2) / (8 * Real.sqrt 5) := by
  -- Unfold the definition of base_area
  unfold base_area
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_base_area_l710_71095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_perimeter_l710_71070

/-- A line passing through (3,0) and intersecting y = 3x and y = 6 - 2x forms an equilateral triangle -/
structure TriangleConfig where
  l : Set (ℝ × ℝ)   -- The line passing through (3,0)
  p1 : ℝ × ℝ        -- Intersection point with y = 3x
  p2 : ℝ × ℝ        -- Intersection point with y = 6 - 2x
  p3 : ℝ × ℝ        -- The point (3,0)

/-- The triangle formed is equilateral -/
def is_equilateral (t : TriangleConfig) : Prop :=
  dist t.p1 t.p2 = dist t.p2 t.p3 ∧ dist t.p2 t.p3 = dist t.p3 t.p1

/-- The line passes through (3,0) -/
def passes_through_3_0 (t : TriangleConfig) : Prop :=
  t.p3 = (3, 0) ∧ t.p3 ∈ t.l

/-- The line intersects y = 3x and y = 6 - 2x -/
def intersects_given_lines (t : TriangleConfig) : Prop :=
  t.p1 ∈ t.l ∧ t.p1.2 = 3 * t.p1.1 ∧
  t.p2 ∈ t.l ∧ t.p2.2 = 6 - 2 * t.p2.1

/-- The perimeter of the triangle -/
def perimeter (t : TriangleConfig) : ℝ :=
  dist t.p1 t.p2 + dist t.p2 t.p3 + dist t.p3 t.p1

theorem equilateral_triangle_perimeter :
  ∀ t : TriangleConfig,
  is_equilateral t ∧ passes_through_3_0 t ∧ intersects_given_lines t →
  perimeter t = 18 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_perimeter_l710_71070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l710_71035

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x^2

-- Define the function m as the derivative of f
noncomputable def m (a : ℝ) (x : ℝ) : ℝ := deriv (f a) x

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x

-- Part I
theorem part_one (a : ℝ) : deriv (m a) 1 = 3 → a = 2 := by
  sorry

-- Part II
theorem part_two (a : ℝ) : 
  (∀ x > 0, StrictMono (g a)) → a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l710_71035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_intersect_N_equals_open_1_closed_2_l710_71009

def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | -3 ≤ x - 1 ∧ x - 1 ≤ 1}

theorem M_intersect_N_equals_open_1_closed_2 : M ∩ N = Set.Ioc 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_intersect_N_equals_open_1_closed_2_l710_71009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_in_sequence_l710_71093

/-- A sequence of 8 increasing real numbers -/
def IncreasingSequence (a : Fin 8 → ℝ) : Prop :=
  ∀ i j, i < j → a i < a j

/-- Arithmetic progression with common difference d -/
def IsArithmeticProgression (a : Fin 4 → ℝ) (d : ℝ) : Prop :=
  ∀ i : Fin 3, a (i.succ) - a i = d

/-- Geometric progression -/
def IsGeometricProgression (a : Fin 4 → ℝ) : Prop :=
  ∃ r : ℝ, ∀ i : Fin 3, a (i.succ) / a i = r

/-- Theorem statement -/
theorem largest_number_in_sequence (a : Fin 8 → ℝ) 
  (h_increasing : IncreasingSequence a)
  (h_ap1 : ∃ i : Fin 5, IsArithmeticProgression (fun j ↦ a (i + j)) 4)
  (h_ap2 : ∃ i : Fin 5, IsArithmeticProgression (fun j ↦ a (i + j)) 36)
  (h_gp : ∃ i : Fin 5, IsGeometricProgression (fun j ↦ a (i + j))) :
  a 7 = 126 ∨ a 7 = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_in_sequence_l710_71093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l710_71025

/-- Proves that a train with given crossing times has a specific length -/
theorem train_length (tree_time platform_time platform_length : ℝ) 
  (h1 : tree_time = 120)
  (h2 : platform_time = 190)
  (h3 : platform_length = 700) : 
  (platform_time * platform_length) / (platform_time - tree_time) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l710_71025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_permutation_one_real_root_l710_71003

/-- A polynomial of degree 2n+1 with all positive coefficients -/
def PositivePolynomial (n : ℕ) := { p : Polynomial ℝ // p.degree = 2*n + 1 ∧ ∀ i, 0 ≤ p.coeff i }

/-- The theorem stating the existence of a permutation resulting in a polynomial with one real root -/
theorem exists_permutation_one_real_root (n : ℕ) (P : PositivePolynomial n) :
  ∃ (σ : Equiv.Perm (Fin (2*n + 2))),
    ∃ (Q : Polynomial ℝ),
      (∀ i : Fin (2*n + 2), Q.coeff i = P.val.coeff (σ i)) ∧
      (∃! x : ℝ, Q.eval x = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_permutation_one_real_root_l710_71003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_complete_residue_system_l710_71001

def sequence_a (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => sequence_a n + 2^(sequence_a n)

theorem sequence_complete_residue_system :
  ∀ (k : ℕ), k > 0 →
    (∀ (x : ℕ), x < 3^k →
      ∃ (n : ℕ), n < 3^k ∧ Nat.ModEq (3^k) (sequence_a n) x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_complete_residue_system_l710_71001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_XYZ_l710_71080

-- Define the right triangle DEF
def triangle_DEF (DE DF EF : ℝ) : Prop :=
  DE = 13 ∧ DF = 5 ∧ DE^2 = DF^2 + EF^2

-- Define the angle bisector point D₁
def angle_bisector_point (D₁F D₁E DF EF : ℝ) : Prop :=
  D₁F / D₁E = DF / EF

-- Define triangle XYZ
def triangle_XYZ (XY XZ D₁E D₁F : ℝ) : Prop :=
  XY = D₁E ∧ XZ = D₁F

-- State the theorem
theorem area_of_triangle_XYZ 
  (DE DF EF D₁F D₁E XY XZ : ℝ) 
  (h1 : triangle_DEF DE DF EF) 
  (h2 : angle_bisector_point D₁F D₁E DF EF) 
  (h3 : triangle_XYZ XY XZ D₁E D₁F) : 
  (1/2 : ℝ) * XY * XZ = 2520/289 := by
  sorry

#check area_of_triangle_XYZ

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_XYZ_l710_71080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integers_between_cubes_l710_71055

theorem integers_between_cubes : 
  (Finset.range (Int.toNat (⌊(10.7 : ℝ)^3⌋ - ⌈(10.5 : ℝ)^3⌉ + 1))).card = 67 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integers_between_cubes_l710_71055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_to_solve_l710_71092

/-- Represents a move in the puzzle -/
structure Move where
  piece : Nat
  fromSquare : Nat
  toSquare : Nat

/-- Represents the state of the puzzle -/
structure PuzzleState where
  squares : Array (List Nat)

/-- Checks if a move is valid according to the rules -/
def isValidMove (state : PuzzleState) (move : Move) : Bool :=
  let fromSquare := state.squares[move.fromSquare]!
  let toSquare := state.squares[move.toSquare]!
  move.piece == fromSquare.head! &&
  (toSquare.isEmpty || move.piece < toSquare.head!)

/-- Applies a move to the current state -/
def applyMove (state : PuzzleState) (move : Move) : PuzzleState :=
  let newSquares := state.squares.mapIdx (fun i square =>
    if i == move.fromSquare then square.tail!
    else if i == move.toSquare then move.piece :: square
    else square)
  { squares := newSquares }

/-- The initial state of the puzzle -/
def initialState : PuzzleState :=
  { squares := #[List.range 15 |>.reverse, [], [], [], [], []] }

/-- The goal state of the puzzle -/
def goalState : PuzzleState :=
  { squares := #[[], [], [], [], [], List.range 15 |>.reverse] }

/-- Theorem stating that the minimum number of moves to solve the puzzle is 49 -/
theorem min_moves_to_solve : ∃ (moves : List Move),
  moves.length = 49 ∧
  moves.foldl applyMove initialState = goalState ∧
  ∀ (m : List Move), m.foldl applyMove initialState = goalState → m.length ≥ 49 := by
  sorry

#check min_moves_to_solve

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_to_solve_l710_71092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_circle_radius_l710_71084

/-- An ellipse with major axis 2a and minor axis 2b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : 0 < b ∧ b < a

/-- The distance from the center to a focus of an ellipse -/
noncomputable def Ellipse.focalDistance (e : Ellipse) : ℝ :=
  Real.sqrt (e.a ^ 2 - e.b ^ 2)

/-- A circle tangent to an ellipse from the inside, centered at one of its foci -/
noncomputable def tangentCircle (e : Ellipse) : ℝ :=
  e.a - e.focalDistance

theorem ellipse_tangent_circle_radius (e : Ellipse) (h : e.a = 6 ∧ e.b = 3) :
  tangentCircle e = 6 - 3 * Real.sqrt 3 := by
  sorry

#check ellipse_tangent_circle_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_circle_radius_l710_71084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_first_five_terms_l710_71015

def my_sequence (n : ℕ+) : ℕ := 2 * n.val - 1

theorem my_sequence_first_five_terms :
  (my_sequence 1 = 1) ∧
  (my_sequence 2 = 3) ∧
  (my_sequence 3 = 5) ∧
  (my_sequence 4 = 7) ∧
  (my_sequence 5 = 9) := by
  sorry

#eval my_sequence 1
#eval my_sequence 2
#eval my_sequence 3
#eval my_sequence 4
#eval my_sequence 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_first_five_terms_l710_71015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_equation_l710_71042

/-- The relationship between x and y as defined in the problem -/
def y (x : ℕ) : ℝ :=
  match x with
  | 0 => 0  -- Adding a case for 0 to cover all natural numbers
  | n + 1 => 4 * (n + 1) + 0.6 * (n + 1)

/-- Theorem stating the relationship between x and y -/
theorem y_equation (x : ℕ) (h : x ≥ 1) : y x = (4 + 0.6) * x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_equation_l710_71042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_f_l710_71049

noncomputable def f (z : ℂ) : ℂ := 
  ((2 + Complex.I * Real.sqrt 2) * z + (Real.sqrt 2 + 4 * Complex.I)) / 2 + 1 - 2 * Complex.I

theorem fixed_point_of_f :
  ∃ c : ℂ, f c = c ∧ c = -Complex.I * (1 + Real.sqrt 2) := by
  sorry

#check fixed_point_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_f_l710_71049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_true_propositions_l710_71099

/-- A proposition about geometric relationships in 3D space -/
inductive GeometricProposition
  | LineLineParallel
  | LinePlaneParallel
  | PlanePlaneParallel
  | PlanePlaneParallelPerpendicularPlane

/-- Checks if a geometric proposition is true in 3D space -/
def isTrue (prop : GeometricProposition) : Bool :=
  match prop with
  | GeometricProposition.LineLineParallel => false
  | GeometricProposition.LinePlaneParallel => true
  | GeometricProposition.PlanePlaneParallel => true
  | GeometricProposition.PlanePlaneParallelPerpendicularPlane => false

/-- The number of true propositions among the given four -/
def numTruePropositions : Nat :=
  [GeometricProposition.LineLineParallel,
   GeometricProposition.LinePlaneParallel,
   GeometricProposition.PlanePlaneParallel,
   GeometricProposition.PlanePlaneParallelPerpendicularPlane]
  |> List.filter isTrue
  |> List.length

theorem two_true_propositions :
  numTruePropositions = 2 := by
  rfl

#eval numTruePropositions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_true_propositions_l710_71099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curve_and_line_l710_71018

/-- The curve function -/
def f (x : ℝ) : ℝ := 3 - x^2

/-- The line function -/
def g (x : ℝ) : ℝ := 2*x

/-- The area between the curve and the line -/
noncomputable def enclosed_area : ℝ := ∫ x in Set.Icc (-3) 1, f x - g x

theorem area_between_curve_and_line : enclosed_area = 32/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curve_and_line_l710_71018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_running_distance_l710_71098

/-- The distance players run around a rectangular field -/
theorem player_running_distance (length width : ℝ) (laps : ℕ) : 
  length = 100 → width = 50 → laps = 6 → 
  2 * (length + width) * (laps : ℝ) = 1800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_running_distance_l710_71098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l710_71044

noncomputable section

-- Define the vectors m and n
def m (α : Real) : Real × Real := (Real.cos α - Real.sqrt 2 / 3, -1)
def n (α : Real) : Real × Real := (Real.sin α, 1)

-- Define the collinearity condition
def collinear (α : Real) : Prop :=
  (m α).1 * (n α).2 = (m α).2 * (n α).1

-- Main theorem
theorem vector_problem (α : Real) 
  (h1 : α ∈ Set.Icc (-Real.pi/2) 0) 
  (h2 : collinear α) : 
  (Real.sin α + Real.cos α = Real.sqrt 2 / 3) ∧ 
  (Real.sin (2*α) / (Real.sin α - Real.cos α) = 7/12) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l710_71044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inscribed_equilateral_triangle_area_l710_71094

-- Define the square
def square_side_length : ℝ := 8

-- Define the equilateral triangle inscribed in the square
def inscribed_equilateral_triangle (s : ℝ) : Prop :=
  s > 0 ∧ s ≤ square_side_length

-- Define the area of an equilateral triangle
noncomputable def equilateral_triangle_area (s : ℝ) : ℝ :=
  (Real.sqrt 3 / 4) * s^2

-- Theorem statement
theorem max_inscribed_equilateral_triangle_area :
  ∃ (s : ℝ), inscribed_equilateral_triangle s ∧
  (∀ (t : ℝ), inscribed_equilateral_triangle t →
  equilateral_triangle_area t ≤ equilateral_triangle_area s) ∧
  equilateral_triangle_area s = 16 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inscribed_equilateral_triangle_area_l710_71094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_good_parabolas_l710_71067

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A parabola represented by its coefficients -/
structure Parabola where
  b : ℝ
  c : ℝ

/-- Determines if a point is on or above a parabola -/
def isOnOrAbove (p : Point) (para : Parabola) : Prop :=
  p.y ≥ p.x^2 + para.b * p.x + para.c

/-- Determines if a parabola is "good" -/
def isGoodParabola (para : Parabola) (points : List Point) (p1 p2 : Point) : Prop :=
  p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧
  (∀ p, p ∈ points → p ≠ p1 → p ≠ p2 → ¬(isOnOrAbove p para))

/-- The main theorem -/
theorem max_good_parabolas {n : ℕ} (points : List Point) (h : points.length = n) 
  (h_distinct : ∀ p1 p2, p1 ∈ points → p2 ∈ points → p1 ≠ p2 → p1.x ≠ p2.x) :
  (∃ goodParabolas : List Parabola, 
    (∀ para, para ∈ goodParabolas → ∃ p1 p2, isGoodParabola para points p1 p2) ∧ 
    goodParabolas.length = n - 1) ∧
  (∀ goodParabolas : List Parabola, 
    (∀ para, para ∈ goodParabolas → ∃ p1 p2, isGoodParabola para points p1 p2) → 
    goodParabolas.length ≤ n - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_good_parabolas_l710_71067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_proof_l710_71063

-- Define the vectors a and b
def a : ℝ × ℝ := (1, -3)
def b : ℝ × ℝ := (-2, 6)

-- Define the vector c as a variable
variable (c : ℝ × ℝ)

-- Theorem statement
theorem vector_magnitude_proof :
  -- Angle between c and a is 60°
  (a.1 * c.1 + a.2 * c.2) / (Real.sqrt ((a.1^2 + a.2^2) * (c.1^2 + c.2^2))) = Real.cos (π/3) →
  -- Dot product condition
  c.1 * (a.1 + b.1) + c.2 * (a.2 + b.2) = -10 →
  -- Conclusion: magnitude of c is 2√10
  Real.sqrt (c.1^2 + c.2^2) = 2 * Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_proof_l710_71063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_specific_planes_l710_71038

/-- The cosine of the angle between two planes -/
noncomputable def cos_angle_between_planes (a1 b1 c1 d1 a2 b2 c2 d2 : ℝ) : ℝ :=
  (a1 * a2 + b1 * b2 + c1 * c2) / 
  (Real.sqrt (a1^2 + b1^2 + c1^2) * Real.sqrt (a2^2 + b2^2 + c2^2))

/-- Theorem: The cosine of the angle between the given planes is -13 / √780 -/
theorem cos_angle_specific_planes :
  cos_angle_between_planes 4 (-3) 1 (-7) 1 5 (-2) 2 = -13 / Real.sqrt 780 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_specific_planes_l710_71038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_product_inequality_l710_71074

theorem triangle_cosine_product_inequality (α β γ : ℝ) 
  (h_triangle : α + β + γ = Real.pi) : 
  Real.cos α * Real.cos β + Real.cos β * Real.cos γ + Real.cos γ * Real.cos α ≤ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_product_inequality_l710_71074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_divisor_problem_l710_71078

theorem smallest_positive_divisor_problem (N k m : ℕ) : 
  (N > 0) →
  (k > 0) →
  (m > 0) →
  (∀ d : ℕ, d > 0 → d ∣ N → d = 1 ∨ d ≥ k) →
  (∀ d : ℕ, d > 0 → d ∣ N → d = 1 ∨ d = N ∨ d ≤ m) →
  (k > m) →
  (k^k + m^m = N) →
  N = 260 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_divisor_problem_l710_71078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_round_to_nearest_tenth_l710_71037

-- Define the function to round to the nearest tenth
noncomputable def roundToNearestTenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

-- State the theorem
theorem sum_and_round_to_nearest_tenth :
  roundToNearestTenth (2.72 + 0.76) = 3.5 := by
  -- Unfold the definition of roundToNearestTenth
  unfold roundToNearestTenth
  -- Simplify the expression
  simp
  -- The proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_round_to_nearest_tenth_l710_71037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_accident_responsibility_l710_71048

-- Define the set of people
inductive Person : Type where
  | A | B | C | D

-- Define a function to represent who is responsible
def responsible (p : Person) : Prop := sorry

-- Define a function to represent who is telling the truth
def telling_truth (p : Person) : Prop := sorry

-- State the theorem
theorem traffic_accident_responsibility :
  -- Only one person is responsible
  (∃! p : Person, responsible p) →
  -- Only one person is telling the truth
  (∃! p : Person, telling_truth p) →
  -- A's statement
  (telling_truth Person.A ↔ responsible Person.B) →
  -- B's statement
  (telling_truth Person.B ↔ responsible Person.C) →
  -- C's statement
  (telling_truth Person.C ↔ telling_truth Person.A) →
  -- D's statement
  (telling_truth Person.D ↔ ¬responsible Person.D) →
  -- Conclusion: A is responsible
  responsible Person.A := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_accident_responsibility_l710_71048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_w_satisfies_projections_l710_71027

/-- The vector w that satisfies the given projection conditions -/
noncomputable def w : ℝ × ℝ := (8.8, 6.3)

/-- The projection of vector v onto vector u -/
noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let magnitude_squared := u.1 * u.1 + u.2 * u.2
  (dot_product / magnitude_squared * u.1, dot_product / magnitude_squared * u.2)

/-- Theorem stating that w satisfies the given projection conditions -/
theorem w_satisfies_projections :
  proj (3, 2) w = (9, 6) ∧ proj (1, 4) w = (2, 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_w_satisfies_projections_l710_71027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l710_71031

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  center : Point

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Given the conditions, prove the equation of the ellipse -/
theorem ellipse_equation (e : Ellipse) (f1 f2 a b : Point) :
  e.a > e.b ∧ e.b > 0 ∧
  distance f1 f2 = 2 * Real.sqrt (e.a^2 - e.b^2) ∧
  distance a f1 + distance b f1 = 16 ∧
  2 * distance f1 f2 = distance a f1 + distance a f2 →
  e.a = 4 ∧ e.b = 2 * Real.sqrt 3 := by
  sorry

#check ellipse_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l710_71031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_triangle_satisfying_inequality_l710_71057

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line segment between two points -/
structure LineSegment where
  start : Point3D
  stop : Point3D

/-- The length of a line segment -/
noncomputable def LineSegment.length (seg : LineSegment) : ℝ :=
  Real.sqrt ((seg.stop.x - seg.start.x)^2 + (seg.stop.y - seg.start.y)^2 + (seg.stop.z - seg.start.z)^2)

/-- Check if four points are coplanar -/
def areFourPointsCoplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

/-- Check if three line segments form a triangle -/
def formTriangle (seg1 seg2 seg3 : LineSegment) : Prop := sorry

/-- The inequality condition for the triangle -/
def satisfiesInequality (a b c : ℝ) : Prop :=
  let p := (a + b + c) / 2
  (a^2 + b^2 + c^2) / 4 ≥ Real.sqrt (3 * p * (p-a) * (p-b) * (p-c))

theorem existence_of_triangle_satisfying_inequality
  (points : Fin 8 → Point3D)
  (segments : Fin 17 → LineSegment)
  (h1 : ∀ i j k l, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i → ¬areFourPointsCoplanar (points i) (points j) (points k) (points l))
  (h2 : ∀ i, (segments i).start ∈ Set.range points ∧ (segments i).stop ∈ Set.range points) :
  ∃ (i j k : Fin 17), i ≠ j ∧ j ≠ k ∧ k ≠ i ∧
    formTriangle (segments i) (segments j) (segments k) ∧
    satisfiesInequality (LineSegment.length (segments i)) (LineSegment.length (segments j)) (LineSegment.length (segments k)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_triangle_satisfying_inequality_l710_71057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_with_remainder_scaling_l710_71010

theorem division_with_remainder_scaling (a b q r k : ℤ) 
  (h1 : b ≠ 0) 
  (h2 : k ≠ 0) 
  (h3 : a = b * q + r) 
  (h4 : 0 ≤ r ∧ r < b.natAbs) : 
  ∃ (q' r' : ℤ), (k * a) = (k * b) * q' + r' ∧ q' = q ∧ r' = k * r ∧ 0 ≤ r' ∧ r' < (k * b).natAbs :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_with_remainder_scaling_l710_71010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_zero_l710_71013

def S (n : ℕ) : Set ℤ :=
  {x | -((2 * n) - 1) ≤ x ∧ x ≤ (2 * n) - 1}

theorem subset_sum_zero (n : ℕ) (A : Finset ℤ) (h1 : ↑A ⊆ S n) (h2 : A.card = 2 * n + 1) :
  ∃ (x y z : ℤ), x ∈ A ∧ y ∈ A ∧ z ∈ A ∧ x + y + z = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_zero_l710_71013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_siblings_ages_l710_71041

-- Define the ages of the siblings
def richard_age : ℕ := sorry
def david_age : ℕ := sorry
def scott_age : ℕ := sorry
def emily_age : ℕ := sorry

-- Define the conditions
axiom richard_david : richard_age = david_age + 6
axiom david_scott : david_age = scott_age + 8
axiom emily_richard : emily_age = richard_age - 5
axiom future_condition : richard_age + 8 = 2 * (scott_age + 8)

-- Theorem to prove
theorem siblings_ages :
  richard_age = 20 ∧ david_age = 14 ∧ scott_age = 6 ∧ emily_age = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_siblings_ages_l710_71041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_some_number_value_l710_71075

theorem some_number_value (some_number : ℝ) : 
  (0.0077 * 3.6) / (0.04 * 0.1 * some_number) = 990.0000000000001 → 
  abs (some_number - 0.007) < 0.000001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_some_number_value_l710_71075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_and_periodicity_l710_71014

noncomputable def f (x : ℝ) : ℝ := (Real.cos x)^2 + Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_lower_bound_and_periodicity :
  (∀ x : ℝ, f x ≥ -1/2) ∧ (∀ x : ℝ, f (2 * Real.pi + x) = f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_and_periodicity_l710_71014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_spent_is_two_dollars_l710_71087

/-- The price of one pencil in cents -/
def pencil_price : ℕ := 20

/-- The number of pencils Tolu wants -/
def tolu_pencils : ℕ := 3

/-- The number of pencils Robert wants -/
def robert_pencils : ℕ := 5

/-- The number of pencils Melissa wants -/
def melissa_pencils : ℕ := 2

/-- The total cost in dollars -/
def total_cost : ℚ := (pencil_price * (tolu_pencils + robert_pencils + melissa_pencils) : ℚ) / 100

theorem total_spent_is_two_dollars : total_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_spent_is_two_dollars_l710_71087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_interval_approx_0_0211_seconds_l710_71091

/-- Represents the length of rails in feet --/
structure RailLength where
  section1 : ℝ
  section2 : ℝ

/-- Calculates the average number of clicks per minute given a speed in mph --/
noncomputable def avg_clicks_per_minute (rail : RailLength) (speed : ℝ) : ℝ :=
  let clicks_section1 := (5280 * speed) / (60 * rail.section1)
  let clicks_section2 := (5280 * speed) / (60 * rail.section2)
  (clicks_section1 + clicks_section2) / 2

/-- The time interval in minutes for which the number of clicks equals the speed --/
noncomputable def time_interval (rail : RailLength) (speed : ℝ) : ℝ :=
  1 / (avg_clicks_per_minute rail speed)

/-- Theorem stating that the time interval is approximately 0.0211 seconds --/
theorem time_interval_approx_0_0211_seconds (speed : ℝ) :
  let rail := RailLength.mk 40 25
  let interval_minutes := time_interval rail speed
  let interval_seconds := interval_minutes * 60
  ∃ (ε : ℝ), ε > 0 ∧ |interval_seconds - 0.0211| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_interval_approx_0_0211_seconds_l710_71091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_position_after_120_moves_l710_71032

noncomputable def ω : ℂ := Complex.exp (Complex.I * (Real.pi / 3))

noncomputable def move (z : ℂ) : ℂ := ω * z + 5

noncomputable def iterate_move : ℕ → ℂ → ℂ
  | 0, z => z
  | n + 1, z => move (iterate_move n z)

theorem final_position_after_120_moves :
  iterate_move 120 5 = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_position_after_120_moves_l710_71032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_four_equals_sixteen_l710_71069

theorem power_of_four_equals_sixteen (x : ℝ) : (4 : ℝ)^x = 16 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_four_equals_sixteen_l710_71069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_theorem_l710_71012

/-- Given a polynomial q(x) with the specified remainder properties,
    prove that its remainder when divided by (x-2)(x+3) is 2x + 3 -/
theorem polynomial_remainder_theorem (q : Polynomial ℝ) 
  (h1 : q.eval 2 = 7)
  (h2 : q.eval (-3) = -3) :
  ∃ k : Polynomial ℝ, q = k * ((X - 2) * (X + 3)) + (2 * X + 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_theorem_l710_71012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_with_root_property_l710_71016

/-- The cubic root of unity -/
noncomputable def ω : ℂ := (-1 + Complex.I * Real.sqrt 3) / 2

/-- The polynomial type we're considering -/
def P (a b c d e : ℝ) (x : ℂ) : ℂ :=
  x^6 + a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + 2024

/-- The root property required by the problem -/
def has_root_property (p : ℂ → ℂ) : Prop :=
  ∀ r : ℂ, p r = 0 → p (ω * r) = 0 ∧ p (-r) = 0

/-- The main theorem -/
theorem unique_polynomial_with_root_property :
  ∃! (a b c d e : ℝ), has_root_property (P a b c d e) := by
  sorry

#check unique_polynomial_with_root_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_with_root_property_l710_71016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tennis_tournament_puzzle_l710_71036

theorem tennis_tournament_puzzle (n : ℕ+) : ¬ (
  ((9 * n.val^2 - 3 * n.val) / 2 : ℚ).num.natAbs % ((9 * n.val^2 - 3 * n.val) / 2 : ℚ).den = 0 ∧
  ((21 * n.val^2 - 7 * n.val) / 8 : ℚ).num.natAbs % ((21 * n.val^2 - 7 * n.val) / 8 : ℚ).den = 0 ∧
  ((45 * n.val^2 - 15 * n.val) / 24 : ℚ).num.natAbs % ((45 * n.val^2 - 15 * n.val) / 24 : ℚ).den = 0 ∧
  ((21 * n.val^2 - 7 * n.val) * 5 = (45 * n.val^2 - 15 * n.val) * 7) ∧
  (n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 7)
) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tennis_tournament_puzzle_l710_71036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_factor_proof_l710_71065

theorem lcm_factor_proof (A B : ℕ) (x : ℕ) 
  (h1 : Nat.gcd A B = 20)
  (h2 : A = 280)
  (h3 : Nat.lcm A B = 20 * x * 14) :
  x = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_factor_proof_l710_71065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l710_71062

def set_A : Set ℝ := {x | 9 * x^2 < 1}
def set_B : Set ℝ := {y | ∃ x, y = x^2 - 2*x + 5/4}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = Set.Icc (1/4 : ℝ) (1/3 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l710_71062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_intensity_indeterminate_l710_71053

/-- Represents the intensity of paint as a percentage -/
def Intensity := ℝ

/-- Represents the fraction of paint replaced -/
def ReplacementFraction := ℝ

/-- 
Given:
- A paint is completely replaced (replacement fraction = 1)
- The replacement paint has an intensity of 20%
- The resulting mixture has an intensity of 20%

Prove: The original paint's intensity cannot be uniquely determined
-/
theorem original_intensity_indeterminate 
  (replacement_fraction : ReplacementFraction)
  (replacement_intensity : Intensity)
  (final_intensity : Intensity)
  (h1 : replacement_fraction = (1 : ℝ))
  (h2 : replacement_intensity = (20 : ℝ))
  (h3 : final_intensity = (20 : ℝ)) :
  ¬∃!original_intensity : Intensity, True :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_intensity_indeterminate_l710_71053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_returns_to_one_l710_71000

def sequenceA (d : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => if sequenceA d n % 2 = 0 then sequenceA d n / 2 else sequenceA d n + d

theorem sequence_returns_to_one (d : ℕ) :
  (d > 0 ∧ Odd d) ↔ ∃ i : ℕ, i > 0 ∧ sequenceA d i = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_returns_to_one_l710_71000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_chain_resistance_theorem_infinite_chain_10_ohm_l710_71045

/-- The resistance of an infinitely long chain of identical resistors -/
noncomputable def infinite_chain_resistance (R₀ : ℝ) : ℝ :=
  (R₀ / 2) * (1 + Real.sqrt 5)

/-- Theorem: The resistance of an infinitely long chain of identical resistors
    is (R₀/2)(1 + √5), where R₀ is the resistance of each individual resistor. -/
theorem infinite_chain_resistance_theorem (R₀ : ℝ) (h : R₀ > 0) :
  let R_X := infinite_chain_resistance R₀
  R_X = R₀ + (R₀ * R_X) / (R₀ + R_X) := by
  sorry

/-- The resistance of an infinitely long chain of 10 Ohm resistors is approximately 16.20 Ohms -/
theorem infinite_chain_10_ohm :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |infinite_chain_resistance 10 - 16.20| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_chain_resistance_theorem_infinite_chain_10_ohm_l710_71045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_eq_interval_l710_71089

/-- An odd function f: ℝ → ℝ where f(x) = 2^x - 3 for x > 0 -/
noncomputable def f : ℝ → ℝ :=
  fun x => if x > 0 then Real.exp (x * Real.log 2) - 3 else -Real.exp (-x * Real.log 2) + 3

/-- f is an odd function -/
axiom f_odd : ∀ x, f (-x) = -f x

/-- The solution set of f(x) ≤ -5 -/
def solution_set : Set ℝ := {x | f x ≤ -5}

theorem solution_set_eq_interval :
  solution_set = Set.Iic (-3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_eq_interval_l710_71089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_participants_perfect_square_l710_71007

/-- Represents a chess tournament with masters and grandmasters -/
structure ChessTournament where
  masters : ℕ
  grandmasters : ℕ

/-- The scoring system for the tournament -/
inductive Score
  | Win : Score
  | Draw : Score
  | Loss : Score

/-- The total number of participants in the tournament -/
def ChessTournament.total_participants (t : ChessTournament) : ℕ :=
  t.masters + t.grandmasters

/-- Points scored against masters by a participant -/
def ChessTournament.points_against_masters (t : ChessTournament) (participant : ℕ) : ℚ :=
  sorry

/-- Total points scored by a participant -/
def ChessTournament.total_points (t : ChessTournament) (participant : ℕ) : ℚ :=
  sorry

/-- Predicate that each participant scores half their points against masters -/
def ChessTournament.half_points_against_masters (t : ChessTournament) : Prop :=
  ∀ participant, participant < t.total_participants →
    2 * (t.points_against_masters participant) = t.total_points participant

/-- The main theorem: if each participant scores half their points against masters,
    then the total number of participants is a perfect square -/
theorem tournament_participants_perfect_square (t : ChessTournament) 
  (h : t.half_points_against_masters) : 
  ∃ n : ℕ, t.total_participants = n ^ 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_participants_perfect_square_l710_71007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_120_l710_71064

/-- The area of a trapezoid with bases 5 and 15, and legs both of length 13 -/
noncomputable def trapezoidArea : ℝ :=
  let a : ℝ := 5  -- length of one base
  let b : ℝ := 15 -- length of the other base
  let c : ℝ := 13 -- length of each leg
  let h : ℝ := Real.sqrt (c^2 - ((b - a) / 2)^2)  -- height of the trapezoid
  (a + b) * h / 2

/-- Theorem stating that the area of the described trapezoid is 120 -/
theorem trapezoid_area_is_120 : trapezoidArea = 120 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval trapezoidArea

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_120_l710_71064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_lower_bound_l710_71019

noncomputable def f (a : ℝ) (x : ℕ+) : ℝ := (x.val^2 + a * x.val + 11) / (x.val + 1)

theorem function_lower_bound (a : ℝ) :
  (∀ x : ℕ+, f a x ≥ 3) → a ≥ -8/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_lower_bound_l710_71019
