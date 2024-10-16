import Mathlib

namespace NUMINAMATH_CALUDE_factor_x12_minus_4096_l1668_166831

theorem factor_x12_minus_4096 (x : ℝ) :
  x^12 - 4096 = (x^6 + 64) * (x + 2) * (x^2 - 2*x + 4) * (x - 2) * (x^2 + 2*x + 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_x12_minus_4096_l1668_166831


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2017th_term_l1668_166868

/-- An arithmetic sequence is monotonically increasing if its common difference is positive -/
def is_monotonically_increasing_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

/-- Three terms form a geometric sequence if the ratio between consecutive terms is constant -/
def is_geometric_sequence (x y z : ℝ) : Prop :=
  ∃ r : ℝ, y = x * r ∧ z = y * r

theorem arithmetic_sequence_2017th_term
  (a : ℕ → ℝ)
  (h_incr : is_monotonically_increasing_arithmetic a)
  (h_first : a 1 = 2)
  (h_geom : is_geometric_sequence (a 1 - 1) (a 3) (a 5 + 5)) :
  a 2017 = 1010 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2017th_term_l1668_166868


namespace NUMINAMATH_CALUDE_initial_mixture_volume_l1668_166899

/-- Given a mixture of milk and water with an initial ratio of 2:1, 
    prove that if 60 litres of water is added to change the ratio to 1:2, 
    the initial volume of the mixture was 60 litres. -/
theorem initial_mixture_volume 
  (initial_milk : ℝ) 
  (initial_water : ℝ) 
  (h1 : initial_milk = 2 * initial_water) 
  (h2 : initial_milk = (initial_water + 60) / 2) : 
  initial_milk + initial_water = 60 := by
  sorry

#check initial_mixture_volume

end NUMINAMATH_CALUDE_initial_mixture_volume_l1668_166899


namespace NUMINAMATH_CALUDE_calculate_expression_l1668_166817

theorem calculate_expression : 5 + 4 * (4 - 9)^3 = -495 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1668_166817


namespace NUMINAMATH_CALUDE_trick_decks_total_spent_l1668_166889

/-- The total amount spent by Frank and his friend on trick decks -/
def total_spent (deck_price : ℕ) (frank_decks : ℕ) (friend_decks : ℕ) : ℕ :=
  deck_price * frank_decks + deck_price * friend_decks

/-- Theorem stating the total amount spent by Frank and his friend -/
theorem trick_decks_total_spent :
  total_spent 7 3 2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_trick_decks_total_spent_l1668_166889


namespace NUMINAMATH_CALUDE_batsman_average_increase_l1668_166828

/-- Represents a batsman's performance -/
structure Batsman where
  innings : Nat
  totalScore : Nat
  notOutCount : Nat

/-- Calculate the batting average -/
def battingAverage (b : Batsman) : Rat :=
  b.totalScore / (b.innings - b.notOutCount)

/-- The increase in average after a new innings -/
def averageIncrease (b : Batsman) (newScore : Nat) : Rat :=
  battingAverage { innings := b.innings + 1, totalScore := b.totalScore + newScore, notOutCount := b.notOutCount } -
  battingAverage b

/-- Theorem: The batsman's average increase is 2 runs -/
theorem batsman_average_increase :
  ∀ (b : Batsman),
    b.innings = 11 →
    b.notOutCount = 0 →
    (battingAverage { innings := b.innings + 1, totalScore := b.totalScore + 80, notOutCount := b.notOutCount } = 58) →
    averageIncrease b 80 = 2 :=
by sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l1668_166828


namespace NUMINAMATH_CALUDE_max_stamps_for_50_dollars_l1668_166865

/-- The maximum number of stamps that can be purchased with a given budget and stamp price -/
def maxStamps (budget : ℕ) (stampPrice : ℕ) : ℕ :=
  (budget / stampPrice : ℕ)

/-- Theorem stating the maximum number of stamps that can be purchased with $50 when stamps cost 45 cents each -/
theorem max_stamps_for_50_dollars : maxStamps 5000 45 = 111 := by
  sorry

end NUMINAMATH_CALUDE_max_stamps_for_50_dollars_l1668_166865


namespace NUMINAMATH_CALUDE_prime_from_divisibility_condition_l1668_166867

-- Define the divisibility condition
def divisibility_condition (n : ℤ) : Prop :=
  ∀ d : ℤ, d ∣ n → (d + 1) ∣ (n + 1)

-- Theorem statement
theorem prime_from_divisibility_condition (n : ℤ) :
  divisibility_condition n → Nat.Prime (Int.natAbs n) :=
by
  sorry

end NUMINAMATH_CALUDE_prime_from_divisibility_condition_l1668_166867


namespace NUMINAMATH_CALUDE_infinitely_many_benelux_couples_l1668_166807

/-- Definition of a Benelux couple -/
def is_benelux_couple (m n : ℕ) : Prop :=
  1 < m ∧ m < n ∧
  (∀ p : ℕ, Nat.Prime p → (p ∣ m ↔ p ∣ n)) ∧
  (∀ p : ℕ, Nat.Prime p → (p ∣ (m + 1) ↔ p ∣ (n + 1)))

/-- Theorem: There exist infinitely many Benelux couples -/
theorem infinitely_many_benelux_couples :
  ∀ N : ℕ, ∃ m n : ℕ, N < m ∧ is_benelux_couple m n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_benelux_couples_l1668_166807


namespace NUMINAMATH_CALUDE_circle_cartesian_and_center_l1668_166896

-- Define the circle C in polar coordinates
def C (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

-- Theorem statement
theorem circle_cartesian_and_center :
  ∃ (x y : ℝ), 
    (∀ (ρ θ : ℝ), C ρ θ ↔ x^2 - 2*x + y^2 = 0) ∧
    (x = 1 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_circle_cartesian_and_center_l1668_166896


namespace NUMINAMATH_CALUDE_parabola_opens_upwards_l1668_166809

/-- For a parabola y = (2-m)x^2 + 1 to open upwards, m must be less than 2 -/
theorem parabola_opens_upwards (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = (2 - m) * x^2 + 1) → 
  (∀ a b : ℝ, a < b → ((2 - m) * a^2 + 1) < ((2 - m) * b^2 + 1)) →
  m < 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_opens_upwards_l1668_166809


namespace NUMINAMATH_CALUDE_triangle_circles_area_sum_l1668_166856

/-- Represents a right triangle with circles centered at its vertices -/
structure TriangleWithCircles where
  /-- The length of the shortest side of the triangle -/
  a : ℝ
  /-- The length of the middle side of the triangle -/
  b : ℝ
  /-- The length of the hypotenuse of the triangle -/
  c : ℝ
  /-- The radius of the circle centered at the vertex opposite to side a -/
  r : ℝ
  /-- The radius of the circle centered at the vertex opposite to side b -/
  s : ℝ
  /-- The radius of the circle centered at the vertex opposite to side c -/
  t : ℝ
  /-- The triangle is a right triangle -/
  right_triangle : a^2 + b^2 = c^2
  /-- The circles are mutually externally tangent -/
  tangent_circles : r + s = a ∧ r + t = b ∧ s + t = c

/-- The theorem stating that for a 6-8-10 right triangle with mutually externally tangent 
    circles centered at its vertices, the sum of the areas of these circles is 56π -/
theorem triangle_circles_area_sum (triangle : TriangleWithCircles) 
    (h1 : triangle.a = 6) (h2 : triangle.b = 8) (h3 : triangle.c = 10) : 
    π * (triangle.r^2 + triangle.s^2 + triangle.t^2) = 56 * π :=
  sorry

end NUMINAMATH_CALUDE_triangle_circles_area_sum_l1668_166856


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1668_166862

/-- The length of the major axis of an ellipse with equation x^2/25 + y^2/16 = 1 is 10 -/
theorem ellipse_major_axis_length : 
  ∀ x y : ℝ, x^2/25 + y^2/16 = 1 → 
  ∃ a b : ℝ, a ≥ b ∧ a^2 = 25 ∧ b^2 = 16 ∧ 2*a = 10 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1668_166862


namespace NUMINAMATH_CALUDE_group_size_proof_l1668_166841

/-- The number of men in a group where replacing one man increases the average weight by 2.5 kg, 
    and the difference between the new man's weight and the replaced man's weight is 25 kg. -/
def number_of_men : ℕ := 10

/-- The increase in average weight when one man is replaced. -/
def average_weight_increase : ℚ := 5/2

/-- The difference in weight between the new man and the replaced man. -/
def weight_difference : ℕ := 25

theorem group_size_proof : 
  number_of_men * average_weight_increase = weight_difference := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l1668_166841


namespace NUMINAMATH_CALUDE_correlation_theorem_l1668_166844

-- Define the relation between x and y
def relation (x y : ℝ) : Prop := y = -0.1 * x + 1

-- Define positive correlation
def positively_correlated (a b : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → a x < a y ∧ b x < b y

-- Define negative correlation
def negatively_correlated (a b : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → a x > a y ∧ b x < b y

-- The main theorem
theorem correlation_theorem (x y z : ℝ → ℝ) 
  (h1 : ∀ t, relation (x t) (y t))
  (h2 : positively_correlated y z) :
  negatively_correlated x y ∧ negatively_correlated x z := by
  sorry

end NUMINAMATH_CALUDE_correlation_theorem_l1668_166844


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1668_166843

theorem inequality_system_solution (x : ℝ) : 
  (5 * x - 1 > 3 * (x + 1) ∧ (1/2) * x - 1 ≤ 7 - (3/2) * x) ↔ (2 < x ∧ x ≤ 4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1668_166843


namespace NUMINAMATH_CALUDE_rectangle_perimeter_rectangle_perimeter_400_l1668_166852

/-- A rectangle divided into four identical squares with a given area has a specific perimeter -/
theorem rectangle_perimeter (area : ℝ) (h_area : area > 0) : 
  ∃ (side : ℝ), 
    side > 0 ∧ 
    4 * side^2 = area ∧ 
    8 * side = 80 :=
by
  sorry

/-- The perimeter of a rectangle with area 400 square centimeters, 
    divided into four identical squares, is 80 centimeters -/
theorem rectangle_perimeter_400 : 
  ∃ (side : ℝ), 
    side > 0 ∧ 
    4 * side^2 = 400 ∧ 
    8 * side = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_rectangle_perimeter_400_l1668_166852


namespace NUMINAMATH_CALUDE_equal_angles_in_special_quadrilateral_l1668_166801

/-- A point on the Cartesian plane -/
structure Point := (x : ℝ) (y : ℝ)

/-- A quadrilateral on the Cartesian plane -/
structure Quadrilateral := (A B C D : Point)

/-- Checks if a point is on the hyperbola y = 1/x -/
def on_hyperbola (p : Point) : Prop := p.y = 1 / p.x

/-- Checks if a point is on the negative branch of the hyperbola -/
def on_negative_branch (p : Point) : Prop := on_hyperbola p ∧ p.x < 0

/-- Checks if a point is on the positive branch of the hyperbola -/
def on_positive_branch (p : Point) : Prop := on_hyperbola p ∧ p.x > 0

/-- Checks if a point is to the left of another point -/
def left_of (p1 p2 : Point) : Prop := p1.x < p2.x

/-- Checks if a line segment passes through the origin -/
def passes_through_origin (p1 p2 : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ t * p1.x + (1 - t) * p2.x = 0 ∧ t * p1.y + (1 - t) * p2.y = 0

/-- Calculates the angle between two lines given three points -/
noncomputable def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- Main theorem -/
theorem equal_angles_in_special_quadrilateral (ABCD : Quadrilateral) :
  on_negative_branch ABCD.A →
  on_negative_branch ABCD.D →
  on_positive_branch ABCD.B →
  on_positive_branch ABCD.C →
  left_of ABCD.B ABCD.C →
  passes_through_origin ABCD.A ABCD.C →
  angle ABCD.B ABCD.A ABCD.D = angle ABCD.B ABCD.C ABCD.D :=
by sorry

end NUMINAMATH_CALUDE_equal_angles_in_special_quadrilateral_l1668_166801


namespace NUMINAMATH_CALUDE_loan_principal_calculation_l1668_166875

theorem loan_principal_calculation (principal : ℝ) : 
  principal * 0.05 * 5 = principal - 2250 → principal = 3000 := by
  sorry

end NUMINAMATH_CALUDE_loan_principal_calculation_l1668_166875


namespace NUMINAMATH_CALUDE_officer_selection_ways_l1668_166838

/-- The number of ways to select distinct officers from a group -/
def selectOfficers (n m : ℕ) : ℕ :=
  (n - 0) * (n - 1) * (n - 2) * (n - 3) * (n - 4)

/-- Theorem: Selecting 5 distinct officers from 12 people results in 95,040 ways -/
theorem officer_selection_ways :
  selectOfficers 12 5 = 95040 := by
  sorry

end NUMINAMATH_CALUDE_officer_selection_ways_l1668_166838


namespace NUMINAMATH_CALUDE_max_product_min_sum_l1668_166853

theorem max_product_min_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∀ x y, x > 0 → y > 0 → x + y = 2 → x * y ≤ a * b → x * y ≤ 1) ∧
  (∀ x y, x > 0 → y > 0 → x + y = 2 → 2/x + 8/y ≥ 2/a + 8/b → 2/x + 8/y ≥ 9) := by
sorry

end NUMINAMATH_CALUDE_max_product_min_sum_l1668_166853


namespace NUMINAMATH_CALUDE_luna_budget_sum_l1668_166825

def luna_budget (house_rental food phone_bill : ℝ) : Prop :=
  food = 0.6 * house_rental ∧
  phone_bill = 0.1 * food ∧
  house_rental + food + phone_bill = 249

theorem luna_budget_sum (house_rental food phone_bill : ℝ) :
  luna_budget house_rental food phone_bill →
  house_rental + food = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_luna_budget_sum_l1668_166825


namespace NUMINAMATH_CALUDE_system_solution_l1668_166815

theorem system_solution (x y : ℝ) : 
  (x^2 + x*y + y^2 = 37 ∧ x^4 + x^2*y^2 + y^4 = 481) ↔ 
  ((x = -4 ∧ y = -3) ∨ (x = -3 ∧ y = -4) ∨ (x = 3 ∧ y = 4) ∨ (x = 4 ∧ y = 3)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1668_166815


namespace NUMINAMATH_CALUDE_field_trip_students_l1668_166808

theorem field_trip_students (adult_chaperones : ℕ) (student_fee adult_fee total_cost : ℚ) : 
  adult_chaperones = 4 →
  student_fee = 5 →
  adult_fee = 6 →
  total_cost = 199 →
  ∃ (num_students : ℕ), (num_students : ℚ) * student_fee + (adult_chaperones : ℚ) * adult_fee = total_cost ∧ 
    num_students = 35 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_students_l1668_166808


namespace NUMINAMATH_CALUDE_distance_to_reflection_over_x_axis_l1668_166892

/-- The distance between a point and its reflection over the x-axis --/
theorem distance_to_reflection_over_x_axis 
  (D : ℝ × ℝ) -- Point D in the plane
  (h : D = (3, -2)) -- D has coordinates (3, -2)
  : ‖D - (D.1, -D.2)‖ = 4 := by
  sorry


end NUMINAMATH_CALUDE_distance_to_reflection_over_x_axis_l1668_166892


namespace NUMINAMATH_CALUDE_saltwater_animals_per_aquarium_l1668_166860

theorem saltwater_animals_per_aquarium 
  (num_aquariums : ℕ) 
  (total_animals : ℕ) 
  (h1 : num_aquariums = 26) 
  (h2 : total_animals = 52) 
  (h3 : total_animals % num_aquariums = 0) :
  total_animals / num_aquariums = 2 := by
sorry

end NUMINAMATH_CALUDE_saltwater_animals_per_aquarium_l1668_166860


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l1668_166888

/-- The lateral surface area of a cone with base radius 2 and height 1 is 2√5π -/
theorem cone_lateral_surface_area :
  let r : ℝ := 2  -- base radius
  let h : ℝ := 1  -- height
  let l : ℝ := Real.sqrt (r^2 + h^2)  -- slant height
  r * l * Real.pi = 2 * Real.sqrt 5 * Real.pi := by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l1668_166888


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1668_166894

theorem absolute_value_equation_solution :
  ∃ x : ℚ, (|x - 1| = |x - 2|) ∧ (x = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1668_166894


namespace NUMINAMATH_CALUDE_additional_buffaloes_count_l1668_166816

/-- Represents the daily fodder consumption of one buffalo -/
def buffalo_consumption : ℚ := 1

/-- Represents the daily fodder consumption of one cow -/
def cow_consumption : ℚ := 3/4 * buffalo_consumption

/-- Represents the daily fodder consumption of one ox -/
def ox_consumption : ℚ := 3/2 * buffalo_consumption

/-- Represents the initial number of buffaloes -/
def initial_buffaloes : ℕ := 15

/-- Represents the initial number of oxen -/
def initial_oxen : ℕ := 8

/-- Represents the initial number of cows -/
def initial_cows : ℕ := 24

/-- Represents the initial duration of fodder in days -/
def initial_duration : ℕ := 24

/-- Represents the number of additional cows -/
def additional_cows : ℕ := 60

/-- Represents the new duration of fodder in days -/
def new_duration : ℕ := 9

/-- Theorem stating that the number of additional buffaloes is 30 -/
theorem additional_buffaloes_count : 
  ∃ (x : ℕ), 
    (initial_buffaloes * buffalo_consumption + 
     initial_oxen * ox_consumption + 
     initial_cows * cow_consumption) * initial_duration =
    ((initial_buffaloes + x) * buffalo_consumption + 
     initial_oxen * ox_consumption + 
     (initial_cows + additional_cows) * cow_consumption) * new_duration ∧
    x = 30 := by sorry

end NUMINAMATH_CALUDE_additional_buffaloes_count_l1668_166816


namespace NUMINAMATH_CALUDE_max_k_on_unit_circle_l1668_166814

theorem max_k_on_unit_circle (k : ℤ) : 
  (0 ≤ k ∧ k ≤ 2019) →
  (∀ m : ℤ, 0 ≤ m ∧ m ≤ 2019 → 
    Complex.abs (Complex.exp (2 * Real.pi * Complex.I * (↑k / 2019)) - 1) ≥ 
    Complex.abs (Complex.exp (2 * Real.pi * Complex.I * (↑m / 2019)) - 1)) →
  k = 1010 := by
  sorry

end NUMINAMATH_CALUDE_max_k_on_unit_circle_l1668_166814


namespace NUMINAMATH_CALUDE_decreasing_linear_function_conditions_l1668_166890

/-- A linear function y = kx - b where y decreases as x increases
    and intersects the y-axis above the x-axis -/
def DecreasingLinearFunction (k b : ℝ) : Prop :=
  k < 0 ∧ b > 0

/-- Theorem stating that for a linear function y = kx - b,
    if y decreases as x increases and intersects the y-axis above the x-axis,
    then k < 0 and b > 0 -/
theorem decreasing_linear_function_conditions (k b : ℝ) :
  DecreasingLinearFunction k b ↔ k < 0 ∧ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_linear_function_conditions_l1668_166890


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l1668_166800

/-- A point (x, y) on the parabola y^2 = 4x that maintains equal distance from (1, 0) and the line x = -1 -/
def Parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- A line passing through (-2, 0) with slope k -/
def Line (k x y : ℝ) : Prop := y = k*(x + 2)

/-- The set of k values for which the line intersects the parabola -/
def IntersectionSet : Set ℝ := {k : ℝ | ∃ x y : ℝ, Parabola x y ∧ Line k x y}

theorem parabola_line_intersection :
  IntersectionSet = {k : ℝ | k ∈ Set.Icc (-Real.sqrt 2 / 2) (Real.sqrt 2 / 2)} := by sorry

#check parabola_line_intersection

end NUMINAMATH_CALUDE_parabola_line_intersection_l1668_166800


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l1668_166872

theorem coefficient_x_cubed_in_expansion (x : ℝ) : 
  (Finset.range 6).sum (fun k => (Nat.choose 5 k) * (2^k) * x^k * (1^(5-k))) = 
  80 * x^3 + (Finset.range 6).sum (fun k => if k ≠ 3 then (Nat.choose 5 k) * (2^k) * x^k * (1^(5-k)) else 0) := by
sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l1668_166872


namespace NUMINAMATH_CALUDE_delta_max_success_ratio_l1668_166818

/-- Represents a participant's score for a single day -/
structure DayScore where
  scored : ℕ
  attempted : ℕ

/-- Represents a participant's scores for two days -/
structure TwoDayScore where
  day1 : DayScore
  day2 : DayScore

def gamma : TwoDayScore := {
  day1 := { scored := 210, attempted := 350 }
  day2 := { scored := 150, attempted := 250 }
}

def total_attempted (score : TwoDayScore) : ℕ :=
  score.day1.attempted + score.day2.attempted

def total_scored (score : TwoDayScore) : ℕ :=
  score.day1.scored + score.day2.scored

def success_ratio (score : DayScore) : ℚ :=
  score.scored / score.attempted

def overall_success_ratio (score : TwoDayScore) : ℚ :=
  (total_scored score) / (total_attempted score)

theorem delta_max_success_ratio :
  ∀ delta : TwoDayScore,
    total_attempted delta = 600 →
    delta.day1.scored > 0 →
    delta.day2.scored > 0 →
    success_ratio delta.day1 < success_ratio gamma.day1 →
    success_ratio delta.day2 < success_ratio gamma.day2 →
    overall_success_ratio delta ≤ 359 / 600 :=
by sorry

end NUMINAMATH_CALUDE_delta_max_success_ratio_l1668_166818


namespace NUMINAMATH_CALUDE_angle_bisector_ratio_not_determine_shape_l1668_166871

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The ratio of an angle bisector to its corresponding side length --/
def angleBisectorToSideRatio (t : Triangle) : ℝ := sorry

/-- Two triangles are similar if they have the same shape --/
def areSimilar (t1 t2 : Triangle) : Prop := sorry

/-- Theorem: The ratio of an angle bisector to the corresponding side's length
    does not uniquely determine the shape of a triangle --/
theorem angle_bisector_ratio_not_determine_shape :
  ∃ (t1 t2 : Triangle), angleBisectorToSideRatio t1 = angleBisectorToSideRatio t2 ∧ ¬ areSimilar t1 t2 := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_ratio_not_determine_shape_l1668_166871


namespace NUMINAMATH_CALUDE_carmen_cats_given_up_l1668_166839

/-- The number of cats Carmen gave up for adoption -/
def cats_given_up : ℕ := 3

/-- The initial number of cats Carmen had -/
def initial_cats : ℕ := 28

/-- The number of dogs Carmen has -/
def dogs : ℕ := 18

theorem carmen_cats_given_up :
  initial_cats - cats_given_up = dogs + 7 :=
by sorry

end NUMINAMATH_CALUDE_carmen_cats_given_up_l1668_166839


namespace NUMINAMATH_CALUDE_expansion_coefficients_l1668_166832

/-- The coefficient of x^n in the expansion of (1 + x^5 + x^7)^20 -/
def coeff (n : ℕ) : ℕ :=
  (Finset.range 21).sum (fun k =>
    (Finset.range (21 - k)).sum (fun m =>
      if 5 * k + 7 * m == n && k + m ≤ 20
      then Nat.choose 20 k * Nat.choose (20 - k) m
      else 0))

theorem expansion_coefficients :
  coeff 17 = 3420 ∧ coeff 18 = 0 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficients_l1668_166832


namespace NUMINAMATH_CALUDE_team_d_wins_l1668_166803

-- Define the teams
inductive Team
| A
| B
| C
| D

-- Define the students
inductive Student
| XiaoZhang
| XiaoWang
| XiaoLi
| XiaoZhao

-- Define a prediction function
def prediction (s : Student) (t : Team) : Prop :=
  match s, t with
  | Student.XiaoZhang, Team.A => True
  | Student.XiaoZhang, Team.B => True
  | Student.XiaoWang, Team.D => True
  | Student.XiaoLi, Team.B => False
  | Student.XiaoLi, Team.C => False
  | Student.XiaoZhao, Team.A => True
  | _, _ => False

-- Define the theorem
theorem team_d_wins (winner : Team) :
  (∃! t : Team, t = winner) →
  (∃! (s1 s2 : Student), s1 ≠ s2 ∧ 
    prediction s1 winner = true ∧ 
    prediction s2 winner = true ∧
    ∀ s, s ≠ s1 → s ≠ s2 → prediction s winner = false) →
  winner = Team.D :=
by sorry

end NUMINAMATH_CALUDE_team_d_wins_l1668_166803


namespace NUMINAMATH_CALUDE_walnut_trees_remaining_l1668_166885

/-- The number of walnut trees remaining in the park after cutting down damaged trees. -/
def remaining_walnut_trees (initial : ℕ) (cut_down : ℕ) : ℕ :=
  initial - cut_down

/-- Theorem stating that 29 walnut trees remain after cutting down 13 from the initial 42. -/
theorem walnut_trees_remaining : remaining_walnut_trees 42 13 = 29 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_remaining_l1668_166885


namespace NUMINAMATH_CALUDE_book_page_numbering_l1668_166805

theorem book_page_numbering (total_digits : Nat) (total_pages : Nat) : 
  total_digits = 1392 ∧ total_pages = 500 →
  (9 * 1) + (90 * 2) + ((total_pages - 99) * 3) = total_digits :=
by sorry

end NUMINAMATH_CALUDE_book_page_numbering_l1668_166805


namespace NUMINAMATH_CALUDE_triangle_angle_contradiction_l1668_166849

theorem triangle_angle_contradiction (α β γ : ℝ) : 
  (α > 60 ∧ β > 60 ∧ γ > 60) → 
  (α + β + γ = 180) → 
  False :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_contradiction_l1668_166849


namespace NUMINAMATH_CALUDE_distance_after_two_hours_l1668_166810

/-- The distance between two people walking in opposite directions after a given time -/
def distanceApart (jaySpeed : Real) (paulSpeed : Real) (time : Real) : Real :=
  (jaySpeed + paulSpeed) * time

theorem distance_after_two_hours :
  let jaySpeed : Real := 1 / 20 -- miles per minute
  let paulSpeed : Real := 3 / 40 -- miles per minute
  let time : Real := 2 * 60 -- 2 hours in minutes
  distanceApart jaySpeed paulSpeed time = 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_after_two_hours_l1668_166810


namespace NUMINAMATH_CALUDE_blanket_thickness_after_four_foldings_l1668_166883

/-- Represents the thickness of a blanket after a certain number of foldings -/
def blanketThickness (initialThickness : ℕ) (numFoldings : ℕ) : ℕ :=
  initialThickness * 2^numFoldings

/-- Proves that a blanket with initial thickness 3 inches will be 48 inches thick after 4 foldings -/
theorem blanket_thickness_after_four_foldings :
  blanketThickness 3 4 = 48 := by
  sorry

#eval blanketThickness 3 4

end NUMINAMATH_CALUDE_blanket_thickness_after_four_foldings_l1668_166883


namespace NUMINAMATH_CALUDE_sum_not_zero_l1668_166897

theorem sum_not_zero (a b c d : ℝ) 
  (eq1 : a * b * c * d - d = 1)
  (eq2 : b * c * d - a = 2)
  (eq3 : c * d * a - b = 3)
  (eq4 : d * a * b - c = -6) :
  a + b + c + d ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_sum_not_zero_l1668_166897


namespace NUMINAMATH_CALUDE_colored_disk_overlap_l1668_166837

/-- Represents a disk with colored sectors -/
structure ColoredDisk :=
  (total_sectors : ℕ)
  (colored_sectors : ℕ)
  (h_total : total_sectors > 0)
  (h_colored : colored_sectors ≤ total_sectors)

/-- Counts the number of positions with at most k overlapping colored sectors -/
def count_low_overlap_positions (d1 d2 : ColoredDisk) (k : ℕ) : ℕ :=
  sorry

theorem colored_disk_overlap (d1 d2 : ColoredDisk) 
  (h1 : d1.total_sectors = 1985) (h2 : d2.total_sectors = 1985)
  (h3 : d1.colored_sectors = 200) (h4 : d2.colored_sectors = 200) :
  count_low_overlap_positions d1 d2 20 ≥ 80 := by
  sorry

end NUMINAMATH_CALUDE_colored_disk_overlap_l1668_166837


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1668_166866

theorem sum_of_coefficients (a b c d : ℤ) :
  (∀ x, (x^2 + a*x + b) * (x^2 + c*x + d) = x^4 + x^3 - 2*x^2 + 17*x + 15) →
  a + b + c + d = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1668_166866


namespace NUMINAMATH_CALUDE_correct_result_l1668_166851

theorem correct_result (mistaken_result : ℕ) 
  (ones_digit_mistake : ℕ) (tens_digit_mistake : ℕ) : 
  mistaken_result = 387 ∧ 
  ones_digit_mistake = 8 - 3 ∧ 
  tens_digit_mistake = 90 - 50 → 
  mistaken_result - ones_digit_mistake + tens_digit_mistake = 422 :=
by sorry

end NUMINAMATH_CALUDE_correct_result_l1668_166851


namespace NUMINAMATH_CALUDE_shirt_tie_combination_count_l1668_166830

/-- The number of possible shirt-and-tie combinations given:
  * total_shirts: The total number of shirts
  * total_ties: The total number of ties
  * incompatible_shirts: The number of shirts that are incompatible with some ties
  * incompatible_ties: The number of ties that are incompatible with some shirts
-/
def shirt_tie_combinations (total_shirts : ℕ) (total_ties : ℕ) 
  (incompatible_shirts : ℕ) (incompatible_ties : ℕ) : ℕ :=
  total_shirts * total_ties - incompatible_shirts * incompatible_ties

/-- Theorem stating that with 8 shirts, 7 ties, and 1 shirt incompatible with 2 ties,
    the total number of possible shirt-and-tie combinations is 54. -/
theorem shirt_tie_combination_count :
  shirt_tie_combinations 8 7 1 2 = 54 := by
  sorry

end NUMINAMATH_CALUDE_shirt_tie_combination_count_l1668_166830


namespace NUMINAMATH_CALUDE_square_ratio_proof_l1668_166820

theorem square_ratio_proof (area_ratio : ℚ) (a b c : ℕ) :
  area_ratio = 75 / 128 →
  (a : ℚ) * Real.sqrt b / c = Real.sqrt (area_ratio) →
  a = 5 ∧ b = 6 ∧ c = 16 →
  a + b + c = 27 :=
by sorry

end NUMINAMATH_CALUDE_square_ratio_proof_l1668_166820


namespace NUMINAMATH_CALUDE_robie_cards_count_l1668_166802

theorem robie_cards_count (cards_per_box : ℕ) (unboxed_cards : ℕ) (boxes_given_away : ℕ) (boxes_remaining : ℕ) : 
  cards_per_box = 25 →
  unboxed_cards = 11 →
  boxes_given_away = 6 →
  boxes_remaining = 12 →
  cards_per_box * (boxes_given_away + boxes_remaining) + unboxed_cards = 461 := by
sorry

end NUMINAMATH_CALUDE_robie_cards_count_l1668_166802


namespace NUMINAMATH_CALUDE_island_closed_path_theorem_l1668_166878

/-- Represents a rectangular county with a diagonal road --/
structure County where
  has_diagonal_road : Bool

/-- Represents a rectangular island composed of counties --/
structure Island where
  counties : List County
  is_rectangular : Bool

/-- Checks if the roads in the counties form a closed path without self-intersection --/
def forms_closed_path (island : Island) : Bool := sorry

/-- Theorem stating that a rectangular island with an odd number of counties can form a closed path
    if and only if it has at least 9 counties --/
theorem island_closed_path_theorem (island : Island) :
  island.is_rectangular ∧ 
  island.counties.length % 2 = 1 ∧
  island.counties.length ≥ 9 ∧
  (∀ c ∈ island.counties, c.has_diagonal_road) →
  forms_closed_path island :=
sorry

end NUMINAMATH_CALUDE_island_closed_path_theorem_l1668_166878


namespace NUMINAMATH_CALUDE_quadratic_inequality_three_integer_solutions_l1668_166859

theorem quadratic_inequality_three_integer_solutions (α : ℝ) : 
  (∃ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (∀ (w : ℤ), 2 * (w : ℝ)^2 - 17 * (w : ℝ) + α ≤ 0 ↔ w = x ∨ w = y ∨ w = z)) →
  -33 ≤ α ∧ α < -30 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_three_integer_solutions_l1668_166859


namespace NUMINAMATH_CALUDE_coin_value_difference_l1668_166882

/-- Represents the number of coins of each type -/
structure CoinCount where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Calculates the total value in cents for a given coin count -/
def totalValue (coins : CoinCount) : ℕ :=
  coins.pennies + 5 * coins.nickels + 10 * coins.dimes

/-- The total number of coins Maria has -/
def totalCoins : ℕ := 3030

/-- Predicate to check if a coin count is valid for Maria -/
def isValidCount (coins : CoinCount) : Prop :=
  coins.pennies ≥ 1 ∧ coins.nickels ≥ 1 ∧ coins.dimes ≥ 1 ∧
  coins.pennies + coins.nickels + coins.dimes = totalCoins

/-- Theorem stating the difference between max and min possible values -/
theorem coin_value_difference :
  ∃ (maxCoins minCoins : CoinCount),
    isValidCount maxCoins ∧ isValidCount minCoins ∧
    (∀ c, isValidCount c → totalValue c ≤ totalValue maxCoins) ∧
    (∀ c, isValidCount c → totalValue c ≥ totalValue minCoins) ∧
    totalValue maxCoins - totalValue minCoins = 27243 :=
  sorry

end NUMINAMATH_CALUDE_coin_value_difference_l1668_166882


namespace NUMINAMATH_CALUDE_intersection_of_lines_l1668_166884

/-- The intersection point of two lines in 2D space -/
structure IntersectionPoint where
  x : ℚ
  y : ℚ

/-- Represents a line in 2D space of the form ax + by = c -/
structure Line where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Check if a point lies on a given line -/
def pointOnLine (p : IntersectionPoint) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y = l.c

theorem intersection_of_lines (line1 line2 : Line)
  (h1 : line1 = ⟨6, -9, 18⟩)
  (h2 : line2 = ⟨8, 2, 20⟩) :
  ∃! p : IntersectionPoint, pointOnLine p line1 ∧ pointOnLine p line2 ∧ p = ⟨18/7, -2/7⟩ := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l1668_166884


namespace NUMINAMATH_CALUDE_amoeba_count_after_week_l1668_166835

/-- Calculates the number of amoebas on a given day -/
def amoeba_count (day : ℕ) : ℕ :=
  if day = 0 then 1
  else if day % 2 = 1 then 3 * amoeba_count (day - 1)
  else (3 * amoeba_count (day - 1)) / 2

/-- The number of amoebas after 7 days is 243 -/
theorem amoeba_count_after_week : amoeba_count 7 = 243 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_count_after_week_l1668_166835


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l1668_166887

theorem min_value_and_inequality (a b x₁ x₂ : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) (hab : a + b = 1) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ / a + x₂ / b + 2 / (x₁ * x₂) ≥ 6) ∧
  (a * x₁ + b * x₂) * (a * x₂ + b * x₁) ≥ x₁ * x₂ := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l1668_166887


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l1668_166834

theorem cube_sum_reciprocal (x : ℝ) (h : x + 1/x = 3) : x^3 + 1/x^3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l1668_166834


namespace NUMINAMATH_CALUDE_lu_daokui_scholarship_winners_l1668_166806

theorem lu_daokui_scholarship_winners 
  (total_winners : ℕ) 
  (first_prize_amount : ℕ) 
  (second_prize_amount : ℕ) 
  (total_prize_money : ℕ) 
  (h1 : total_winners = 28)
  (h2 : first_prize_amount = 10000)
  (h3 : second_prize_amount = 2000)
  (h4 : total_prize_money = 80000) :
  ∃ (first_prize_winners second_prize_winners : ℕ),
    first_prize_winners + second_prize_winners = total_winners ∧
    first_prize_winners * first_prize_amount + second_prize_winners * second_prize_amount = total_prize_money ∧
    first_prize_winners = 3 ∧
    second_prize_winners = 25 := by
  sorry

end NUMINAMATH_CALUDE_lu_daokui_scholarship_winners_l1668_166806


namespace NUMINAMATH_CALUDE_rectangle_perimeter_area_ratio_bound_l1668_166880

/-- A function that checks if a number is prime --/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- The theorem statement --/
theorem rectangle_perimeter_area_ratio_bound :
  ∀ l w : ℕ,
    l < 100 →
    w < 100 →
    l ≠ w →
    isPrime l →
    isPrime w →
    (2 * l + 2 * w)^2 / (l * w : ℚ) ≥ 82944 / 5183 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_area_ratio_bound_l1668_166880


namespace NUMINAMATH_CALUDE_right_triangle_existence_condition_l1668_166898

/-- A right triangle with hypotenuse c and median s_a to one of the legs. -/
structure RightTriangle (c s_a : ℝ) :=
  (hypotenuse_positive : c > 0)
  (median_positive : s_a > 0)
  (right_angle : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = c^2)
  (median_property : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = c^2 ∧ s_a^2 = (a/2)^2 + (c/2)^2)

/-- The existence condition for a right triangle with given hypotenuse and median. -/
theorem right_triangle_existence_condition (c s_a : ℝ) :
  (∃ (t : RightTriangle c s_a), True) ↔ (c/2 < s_a ∧ s_a < c) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_existence_condition_l1668_166898


namespace NUMINAMATH_CALUDE_circumcircle_tangency_l1668_166881

-- Define the points
variable (A B C D E F : EuclideanPlane)

-- Define the parallelogram ABCD
def is_parallelogram (A B C D : EuclideanPlane) : Prop := sorry

-- Define that E is on BC
def point_on_segment (P Q R : EuclideanPlane) : Prop := sorry

-- Define that F is on AD
-- (We can reuse the point_on_segment definition)

-- Define the circumcircle of a triangle
def circumcircle (P Q R : EuclideanPlane) : Set EuclideanPlane := sorry

-- Define a line being tangent to a circle
def is_tangent (line : Set EuclideanPlane) (circle : Set EuclideanPlane) : Prop := sorry

-- Define a line segment
def line_segment (P Q : EuclideanPlane) : Set EuclideanPlane := sorry

-- The main theorem
theorem circumcircle_tangency 
  (h_parallelogram : is_parallelogram A B C D)
  (h_E_on_BC : point_on_segment B E C)
  (h_F_on_AD : point_on_segment A F D)
  (h_ABE_tangent_CF : is_tangent (line_segment C F) (circumcircle A B E)) :
  is_tangent (line_segment A E) (circumcircle C D F) := by
  sorry

end NUMINAMATH_CALUDE_circumcircle_tangency_l1668_166881


namespace NUMINAMATH_CALUDE_erased_number_l1668_166833

theorem erased_number (a : ℤ) (b : ℤ) (h1 : -4 ≤ b ∧ b ≤ 4) 
  (h2 : 8 * a - b = 1703) : a + b = 214 := by
  sorry

end NUMINAMATH_CALUDE_erased_number_l1668_166833


namespace NUMINAMATH_CALUDE_sum_reciprocals_and_diff_squares_l1668_166869

theorem sum_reciprocals_and_diff_squares (x y : ℝ) 
  (sum_eq : x + y = 12) 
  (prod_eq : x * y = 32) : 
  (1 / x + 1 / y = 3 / 8) ∧ (x^2 - y^2 = 48 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_and_diff_squares_l1668_166869


namespace NUMINAMATH_CALUDE_shopkeeper_additional_cards_l1668_166855

/-- The number of cards in a standard deck -/
def standard_deck : ℕ := 52

/-- The number of complete decks the shopkeeper has -/
def complete_decks : ℕ := 6

/-- The total number of cards the shopkeeper has -/
def total_cards : ℕ := 319

/-- The number of additional cards the shopkeeper has -/
def additional_cards : ℕ := total_cards - (complete_decks * standard_deck)

theorem shopkeeper_additional_cards : additional_cards = 7 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_additional_cards_l1668_166855


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l1668_166874

/-- Represents a hyperbola -/
structure Hyperbola where
  /-- The asymptotic equations of the hyperbola are y = ± (slope * x) -/
  slope : ℝ
  /-- The focal length of the hyperbola -/
  focal_length : ℝ

/-- Checks if the given equation is a valid standard form for the hyperbola -/
def is_standard_equation (h : Hyperbola) (eq : ℝ → ℝ → ℝ) : Prop :=
  (∀ x y, eq x y = 0 ↔ x^2 / 4 - y^2 = 1) ∨
  (∀ x y, eq x y = 0 ↔ y^2 - x^2 / 4 = 1)

/-- The main theorem stating the standard equation of the hyperbola -/
theorem hyperbola_standard_equation (h : Hyperbola) 
  (h_slope : h.slope = 1/2) 
  (h_focal : h.focal_length = 2 * Real.sqrt 5) :
  ∃ eq : ℝ → ℝ → ℝ, is_standard_equation h eq :=
sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l1668_166874


namespace NUMINAMATH_CALUDE_extreme_value_inequality_l1668_166847

/-- A function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4 * Real.log x - (1/2) * a * x^2 + (4-a) * x

/-- The derivative of f(x) -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 4 / x - a * x + (4-a)

theorem extreme_value_inequality (a : ℝ) (x₀ x₁ x₂ : ℝ) 
  (ha : a > 0) 
  (hx₀ : x₀ > 0) 
  (hx₁ : x₁ > 0) 
  (hx₂ : x₂ > 0) 
  (h_order : x₁ < x₂) 
  (h_extreme : ∃ x, x > 0 ∧ ∀ y, y > 0 → f a x ≥ f a y) 
  (h_mean_value : f a x₁ - f a x₂ = f_deriv a x₀ * (x₁ - x₂)) :
  x₁ + x₂ > 2 * x₀ := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_inequality_l1668_166847


namespace NUMINAMATH_CALUDE_car_insurance_present_value_l1668_166840

/-- Calculate the present value of a series of payments with annual growth and inflation --/
theorem car_insurance_present_value
  (initial_payment : ℝ)
  (insurance_growth_rate : ℝ)
  (inflation_rate : ℝ)
  (years : ℕ)
  (h1 : initial_payment = 3000)
  (h2 : insurance_growth_rate = 0.05)
  (h3 : inflation_rate = 0.02)
  (h4 : years = 10) :
  ∃ (pv : ℝ), abs (pv - ((initial_payment * ((1 + insurance_growth_rate) ^ years - 1) / insurance_growth_rate) / (1 + inflation_rate) ^ years)) < 0.01 ∧ 
  30954.87 < pv ∧ pv < 30954.89 :=
by
  sorry

end NUMINAMATH_CALUDE_car_insurance_present_value_l1668_166840


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l1668_166879

theorem unique_three_digit_number : ∃! n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧ 
  (n / 11 : ℚ) = (n / 100 : ℕ) + ((n / 10) % 10 : ℕ) + (n % 10 : ℕ) ∧
  n = 198 := by
sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l1668_166879


namespace NUMINAMATH_CALUDE_natural_number_pairs_l1668_166822

theorem natural_number_pairs : 
  ∀ x y : ℕ, 
    x + y > 40 ∧ 
    2^(10*x + 24*y - 493) + 1 = 9 * 2^(5*x + 12*y - 248) → 
    ((x = 4 ∧ y = 36) ∨ (x = 49 ∧ y = 0) ∨ (x = 37 ∧ y = 7)) :=
by sorry

end NUMINAMATH_CALUDE_natural_number_pairs_l1668_166822


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1668_166858

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  right_angle : (R.1 - P.1) * (R.1 - Q.1) + (R.2 - P.2) * (R.2 - Q.2) = 0

-- Define the squares PQXY and PRWZ
structure Square (A B C D : ℝ × ℝ) : Prop where
  side_length_eq : (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2
  right_angle : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0

-- Define points on a circle
def OnCircle (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop :=
  (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2

-- Main theorem
theorem triangle_perimeter 
  (P Q R X Y Z W : ℝ × ℝ) 
  (h_triangle : Triangle P Q R)
  (h_pq_length : (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = 100)
  (h_square_pq : Square P Q X Y)
  (h_square_pr : Square P R W Z)
  (h_circle : ∃ (center : ℝ × ℝ) (radius : ℝ), 
    OnCircle center radius X ∧ 
    OnCircle center radius Y ∧ 
    OnCircle center radius Z ∧ 
    OnCircle center radius W) :
  (Q.1 - P.1)^2 + (Q.2 - P.2)^2 + 
  (R.1 - P.1)^2 + (R.2 - P.2)^2 + 
  (R.1 - Q.1)^2 + (R.2 - Q.2)^2 = (10 + 10 * Real.sqrt 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1668_166858


namespace NUMINAMATH_CALUDE_dataset_growth_percentage_l1668_166842

theorem dataset_growth_percentage (initial_size : ℕ) (final_size : ℕ) : 
  initial_size = 200 →
  final_size = 180 →
  ∃ (growth_percentage : ℚ),
    growth_percentage = 20 ∧
    (3/4 : ℚ) * (initial_size + initial_size * (growth_percentage / 100)) = final_size :=
by sorry

end NUMINAMATH_CALUDE_dataset_growth_percentage_l1668_166842


namespace NUMINAMATH_CALUDE_track_length_not_approximately_200mm_l1668_166846

/-- Represents the length of a school's track and field in millimeters -/
def track_length : ℝ := 200000 -- Assuming 200 meters = 200000 mm

/-- Represents a reasonable range for "approximately 200 mm" -/
def approximate_range : Set ℝ := {x | 190 ≤ x ∧ x ≤ 210}

theorem track_length_not_approximately_200mm : 
  track_length ∉ approximate_range := by sorry

end NUMINAMATH_CALUDE_track_length_not_approximately_200mm_l1668_166846


namespace NUMINAMATH_CALUDE_A_and_D_independent_l1668_166829

-- Define the sample space
def Ω : Type := Fin 6 × Fin 6

-- Define the probability measure
noncomputable def P : Set Ω → ℝ := sorry

-- Define events A and D
def A : Set Ω := {ω | ω.1 = 0}
def D : Set Ω := {ω | ω.1.val + ω.2.val + 2 = 7}

-- State the theorem
theorem A_and_D_independent : 
  P (A ∩ D) = P A * P D := by sorry

end NUMINAMATH_CALUDE_A_and_D_independent_l1668_166829


namespace NUMINAMATH_CALUDE_intermediate_circle_radius_l1668_166893

theorem intermediate_circle_radius 
  (r₁ r₂ r₃ : ℝ) 
  (h₁ : r₁ = 5)
  (h₂ : r₃ = 13)
  (h₃ : π * r₁^2 = π * r₃^2 - π * r₂^2) :
  r₂ = 12 := by
sorry

end NUMINAMATH_CALUDE_intermediate_circle_radius_l1668_166893


namespace NUMINAMATH_CALUDE_team_average_score_l1668_166811

theorem team_average_score (lefty_score : ℕ) (righty_score : ℕ) (other_score : ℕ) :
  lefty_score = 20 →
  righty_score = lefty_score / 2 →
  other_score = 6 * righty_score →
  (lefty_score + righty_score + other_score) / 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_team_average_score_l1668_166811


namespace NUMINAMATH_CALUDE_new_person_age_l1668_166823

theorem new_person_age (n : ℕ) (initial_avg : ℝ) (new_avg : ℝ) (new_person_age : ℝ) :
  n = 9 →
  initial_avg = 15 →
  new_avg = 17 →
  (n * initial_avg + new_person_age) / (n + 1) = new_avg →
  new_person_age = 35 := by
  sorry

end NUMINAMATH_CALUDE_new_person_age_l1668_166823


namespace NUMINAMATH_CALUDE_male_contestants_l1668_166857

theorem male_contestants (total : ℕ) (female_ratio : ℚ) (h1 : total = 18) (h2 : female_ratio = 1/3) :
  (1 - female_ratio) * total = 12 := by
  sorry

end NUMINAMATH_CALUDE_male_contestants_l1668_166857


namespace NUMINAMATH_CALUDE_intersection_of_sets_l1668_166886

theorem intersection_of_sets : 
  let A : Set ℕ := {1, 3, 4}
  let B : Set ℕ := {0, 1, 3}
  A ∩ B = {1, 3} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l1668_166886


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l1668_166848

theorem arithmetic_sequence_sum_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)) →  -- arithmetic sequence sum formula
  (S 6 / S 3 = 4) →                                         -- given condition
  (S 9 / S 6 = 9 / 4) :=                                    -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l1668_166848


namespace NUMINAMATH_CALUDE_bottom_row_bricks_l1668_166861

/-- Represents a pyramidal brick wall -/
structure PyramidalWall where
  rows : ℕ
  totalBricks : ℕ
  bottomRowBricks : ℕ

/-- Calculates the total number of bricks in a pyramidal wall -/
def calculateTotalBricks (wall : PyramidalWall) : ℕ :=
  (wall.rows : ℕ) * (2 * wall.bottomRowBricks - wall.rows + 1) / 2

theorem bottom_row_bricks (wall : PyramidalWall) 
  (h1 : wall.rows = 15)
  (h2 : wall.totalBricks = 300)
  (h3 : calculateTotalBricks wall = wall.totalBricks) :
  wall.bottomRowBricks = 27 := by
  sorry

end NUMINAMATH_CALUDE_bottom_row_bricks_l1668_166861


namespace NUMINAMATH_CALUDE_triangle_side_length_l1668_166873

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  (1/2 * a * c * Real.sin B = Real.sqrt 3) →  -- Area of triangle is √3
  (B = Real.pi / 3) →  -- B = 60°
  (a^2 + c^2 = 3*a*c) →  -- Given condition
  (b = 2 * Real.sqrt 2) :=  -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1668_166873


namespace NUMINAMATH_CALUDE_recycling_project_weight_l1668_166876

-- Define the number of items collected by each person
def marcus_bottles : ℕ := 25
def marcus_cans : ℕ := 30
def john_bottles : ℕ := 20
def john_cans : ℕ := 25
def sophia_bottles : ℕ := 15
def sophia_cans : ℕ := 35

-- Define the weight of each item
def bottle_weight : ℚ := 0.5
def can_weight : ℚ := 0.025

-- Define the total weight function
def total_weight : ℚ :=
  (marcus_bottles + john_bottles + sophia_bottles) * bottle_weight +
  (marcus_cans + john_cans + sophia_cans) * can_weight

-- Theorem statement
theorem recycling_project_weight :
  total_weight = 32.25 := by sorry

end NUMINAMATH_CALUDE_recycling_project_weight_l1668_166876


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1668_166826

theorem algebraic_expression_value (x : ℝ) : 
  2 * x^2 + 3 * x + 7 = 8 → 2 * x^2 + 3 * x - 7 = -6 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1668_166826


namespace NUMINAMATH_CALUDE_birthday_candles_ratio_l1668_166804

def simplify_ratio (a b : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd a b
  (a / gcd, b / gcd)

theorem birthday_candles_ratio : simplify_ratio 45 27 = (5, 3) := by
  sorry

end NUMINAMATH_CALUDE_birthday_candles_ratio_l1668_166804


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l1668_166819

theorem cubic_sum_theorem (p q r : ℝ) (h_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (h_eq : (p^3 + 7) / p = (q^3 + 7) / q ∧ (q^3 + 7) / q = (r^3 + 7) / r) : 
  p^3 + q^3 + r^3 = -21 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l1668_166819


namespace NUMINAMATH_CALUDE_triangle_vector_parallel_l1668_166864

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

end NUMINAMATH_CALUDE_triangle_vector_parallel_l1668_166864


namespace NUMINAMATH_CALUDE_train_length_l1668_166870

/-- The length of a train given its speed and time to cross a stationary observer -/
theorem train_length (speed_kmh : ℝ) (time_seconds : ℝ) : 
  speed_kmh = 48 → time_seconds = 12 → speed_kmh * (5/18) * time_seconds = 480 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1668_166870


namespace NUMINAMATH_CALUDE_equation_solution_l1668_166891

theorem equation_solution (x : ℝ) :
  (∀ y : ℝ, 10 * x * y - 15 * y + 2 * x - 3 = 0) ↔ x = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1668_166891


namespace NUMINAMATH_CALUDE_largest_angle_measure_l1668_166827

/-- A convex heptagon with interior angles as specified -/
structure ConvexHeptagon where
  x : ℝ
  angle1 : ℝ := x + 2
  angle2 : ℝ := 2 * x
  angle3 : ℝ := 3 * x
  angle4 : ℝ := 4 * x
  angle5 : ℝ := 5 * x
  angle6 : ℝ := 6 * x - 2
  angle7 : ℝ := 7 * x - 3

/-- The sum of interior angles of a heptagon is 900 degrees -/
axiom heptagon_angle_sum (h : ConvexHeptagon) : 
  h.angle1 + h.angle2 + h.angle3 + h.angle4 + h.angle5 + h.angle6 + h.angle7 = 900

/-- The measure of the largest angle in the specified convex heptagon is 222.75 degrees -/
theorem largest_angle_measure (h : ConvexHeptagon) : 
  max h.angle1 (max h.angle2 (max h.angle3 (max h.angle4 (max h.angle5 (max h.angle6 h.angle7))))) = 222.75 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_measure_l1668_166827


namespace NUMINAMATH_CALUDE_book_selection_ways_l1668_166824

def number_of_books : ℕ := 3

def ways_to_choose (n : ℕ) : ℕ := 2^n - 1

theorem book_selection_ways :
  ways_to_choose number_of_books = 7 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_ways_l1668_166824


namespace NUMINAMATH_CALUDE_three_digit_square_ends_with_self_l1668_166845

theorem three_digit_square_ends_with_self (A : ℕ) : 
  (100 ≤ A ∧ A ≤ 999) ∧ (A^2 % 1000 = A) ↔ (A = 376 ∨ A = 625) := by
sorry

end NUMINAMATH_CALUDE_three_digit_square_ends_with_self_l1668_166845


namespace NUMINAMATH_CALUDE_smallest_k_divisible_by_500_l1668_166850

theorem smallest_k_divisible_by_500 : 
  ∀ k : ℕ+, k.val < 3000 → ¬(500 ∣ (k.val * (k.val + 1) * (2 * k.val + 1) / 6)) ∧ 
  (500 ∣ (3000 * 3001 * 6001 / 6)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_divisible_by_500_l1668_166850


namespace NUMINAMATH_CALUDE_chord_slope_l1668_166877

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 9 = 1

/-- Definition of the midpoint of the chord -/
def is_midpoint (x y : ℝ) : Prop := x = 4 ∧ y = 2

/-- Theorem: The slope of the chord is -1/2 -/
theorem chord_slope (x1 y1 x2 y2 : ℝ) :
  is_on_ellipse x1 y1 → is_on_ellipse x2 y2 →
  is_midpoint ((x1 + x2) / 2) ((y1 + y2) / 2) →
  (y2 - y1) / (x2 - x1) = -1/2 := by sorry

end NUMINAMATH_CALUDE_chord_slope_l1668_166877


namespace NUMINAMATH_CALUDE_max_product_constraint_l1668_166836

theorem max_product_constraint (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 30) :
  x^5 * y^3 ≤ (18.75 : ℝ)^5 * (11.25 : ℝ)^3 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constraint_l1668_166836


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1668_166854

theorem arithmetic_calculation : 8 / 4 - 3 - 9 + 3 * 7 - 2^2 = 7 := by sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1668_166854


namespace NUMINAMATH_CALUDE_cone_volume_l1668_166863

/-- The volume of a cone whose lateral surface unfolds to a semicircle with radius 2 -/
theorem cone_volume (r : Real) (h : Real) : 
  r = 1 → h = Real.sqrt 3 → (1/3 : Real) * π * r^2 * h = (Real.sqrt 3 / 3) * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l1668_166863


namespace NUMINAMATH_CALUDE_anns_shopping_problem_l1668_166812

theorem anns_shopping_problem (total_spent : ℝ) (shorts_price : ℝ) (shorts_count : ℕ) 
  (shoes_price : ℝ) (shoes_count : ℕ) (tops_count : ℕ) :
  total_spent = 75 →
  shorts_price = 7 →
  shorts_count = 5 →
  shoes_price = 10 →
  shoes_count = 2 →
  tops_count = 4 →
  (total_spent - (shorts_price * shorts_count + shoes_price * shoes_count)) / tops_count = 5 := by
sorry

end NUMINAMATH_CALUDE_anns_shopping_problem_l1668_166812


namespace NUMINAMATH_CALUDE_rate_of_profit_l1668_166813

theorem rate_of_profit (cost_price selling_price : ℝ) : 
  cost_price = 50 → selling_price = 100 → 
  (selling_price - cost_price) / cost_price * 100 = 100 := by
  sorry

end NUMINAMATH_CALUDE_rate_of_profit_l1668_166813


namespace NUMINAMATH_CALUDE_fraction_ordering_l1668_166895

theorem fraction_ordering : (6 : ℚ) / 29 < 8 / 25 ∧ 8 / 25 < 10 / 31 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l1668_166895


namespace NUMINAMATH_CALUDE_remainder_3_100_mod_7_l1668_166821

theorem remainder_3_100_mod_7 : 3^100 % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_100_mod_7_l1668_166821
