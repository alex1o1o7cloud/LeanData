import Mathlib

namespace kindergarten_count_l377_37766

/-- Given the ratio of boys to girls and girls to teachers in a kindergarten,
    along with the number of boys, prove the total number of students and teachers. -/
theorem kindergarten_count (boys girls teachers : ℕ) : 
  (boys : ℚ) / girls = 3 / 4 →
  (girls : ℚ) / teachers = 5 / 2 →
  boys = 18 →
  boys + girls + teachers = 53 := by
sorry

end kindergarten_count_l377_37766


namespace circle_equation_diameter_circle_equation_points_line_l377_37730

-- Define points and line
def P₁ : ℝ × ℝ := (4, 9)
def P₂ : ℝ × ℝ := (6, 3)
def A : ℝ × ℝ := (2, -3)
def B : ℝ × ℝ := (-2, -5)
def l (x y : ℝ) : Prop := x - 2*y - 3 = 0

-- Theorem for the first circle
theorem circle_equation_diameter (x y : ℝ) : 
  (x - 5)^2 + (y - 6)^2 = 10 ↔ 
  ∃ (t : ℝ), (x, y) = (1 - t) • P₁ + t • P₂ ∧ 0 ≤ t ∧ t ≤ 1 :=
sorry

-- Theorem for the second circle
theorem circle_equation_points_line (x y : ℝ) :
  x^2 + y^2 + 2*x + 4*y - 5 = 0 ↔
  (∃ (cx cy : ℝ), (x - cx)^2 + (y - cy)^2 = ((x - 2)^2 + (y + 3)^2) ∧
                  (x - (-2))^2 + (y - (-5))^2 = ((x - 2)^2 + (y + 3)^2) ∧
                  l cx cy) :=
sorry

end circle_equation_diameter_circle_equation_points_line_l377_37730


namespace exam_probabilities_l377_37752

/-- Represents the probabilities of scoring in different ranges in a math exam --/
structure ExamProbabilities where
  above90 : ℝ
  between80and89 : ℝ
  between70and79 : ℝ
  between60and69 : ℝ

/-- Calculates the probability of scoring 80 or above --/
def prob_80_or_above (p : ExamProbabilities) : ℝ :=
  p.above90 + p.between80and89

/-- Calculates the probability of failing the exam (scoring below 60) --/
def prob_fail (p : ExamProbabilities) : ℝ :=
  1 - (p.above90 + p.between80and89 + p.between70and79 + p.between60and69)

/-- Theorem stating the probabilities of scoring 80 or above and failing the exam --/
theorem exam_probabilities 
  (p : ExamProbabilities) 
  (h1 : p.above90 = 0.18) 
  (h2 : p.between80and89 = 0.51) 
  (h3 : p.between70and79 = 0.15) 
  (h4 : p.between60and69 = 0.09) : 
  prob_80_or_above p = 0.69 ∧ prob_fail p = 0.07 := by
  sorry

end exam_probabilities_l377_37752


namespace rectangle_length_width_difference_l377_37755

theorem rectangle_length_width_difference 
  (length width : ℝ) 
  (h1 : length = 6)
  (h2 : width = 4)
  (h3 : 2 * (length + width) = 20)
  (h4 : ∃ d : ℝ, length = width + d) : 
  length - width = 2 := by
  sorry

end rectangle_length_width_difference_l377_37755


namespace tan_sum_quarter_pi_l377_37735

theorem tan_sum_quarter_pi (α β : Real) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β - π/4) = 1/4) : 
  Real.tan (α + π/4) = 3/22 := by
  sorry

end tan_sum_quarter_pi_l377_37735


namespace prob_sum_five_l377_37774

/-- A uniformly dense cubic die -/
structure Die :=
  (faces : Fin 6)

/-- The result of throwing a die twice -/
def TwoThrows := Die × Die

/-- The sum of points from two throws -/
def sum_points (t : TwoThrows) : ℕ :=
  t.1.faces.val + 1 + t.2.faces.val + 1

/-- The set of all possible outcomes when throwing a die twice -/
def all_outcomes : Finset TwoThrows :=
  sorry

/-- The set of outcomes where the sum of points is 5 -/
def sum_five : Finset TwoThrows :=
  sorry

/-- The probability of an event occurring when throwing a die twice -/
def prob (event : Finset TwoThrows) : ℚ :=
  (event.card : ℚ) / (all_outcomes.card : ℚ)

theorem prob_sum_five :
  prob sum_five = 1 / 9 :=
sorry

end prob_sum_five_l377_37774


namespace seat_representation_l377_37791

/-- Represents a seat in a movie theater -/
structure Seat :=
  (row : ℕ)
  (column : ℕ)

/-- The notation for representing seats in the movie theater -/
def seat_notation (r : ℕ) (c : ℕ) : Seat := ⟨r, c⟩

/-- Theorem stating that if (5, 2) represents the seat in the 5th row and 2nd column,
    then (7, 3) represents the seat in the 7th row and 3rd column -/
theorem seat_representation :
  (seat_notation 5 2 = ⟨5, 2⟩) →
  (seat_notation 7 3 = ⟨7, 3⟩) :=
by sorry

end seat_representation_l377_37791


namespace triangle_inequality_with_semiperimeter_l377_37729

theorem triangle_inequality_with_semiperimeter (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  Real.sqrt (b + c - a) + Real.sqrt (c + a - b) + Real.sqrt (a + b - c) ≤ 
  Real.sqrt a + Real.sqrt b + Real.sqrt c ∧ 
  (Real.sqrt (b + c - a) + Real.sqrt (c + a - b) + Real.sqrt (a + b - c) = 
   Real.sqrt a + Real.sqrt b + Real.sqrt c ↔ a = b ∧ b = c) :=
by sorry

end triangle_inequality_with_semiperimeter_l377_37729


namespace mono_decreasing_g_l377_37754

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of f being monotonically increasing on [1, 2]
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 2 → x ≤ y → f x ≤ f y

-- Define the function g(x) = f(1-x)
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (1 - x)

-- State the theorem
theorem mono_decreasing_g (h : MonoIncreasing f) :
  ∀ x y, x ∈ Set.Icc (-1) 0 → y ∈ Set.Icc (-1) 0 → x ≤ y → g f y ≤ g f x :=
sorry

end mono_decreasing_g_l377_37754


namespace jemma_price_calculation_l377_37780

/-- The price at which Jemma sells each frame -/
def jemma_price : ℝ := 5

/-- The number of frames Jemma sold -/
def jemma_frames : ℕ := 400

/-- The total revenue made by both Jemma and Dorothy -/
def total_revenue : ℝ := 2500

theorem jemma_price_calculation :
  (jemma_price * jemma_frames : ℝ) + 
  (jemma_price / 2 * (jemma_frames / 2) : ℝ) = total_revenue :=
by sorry

end jemma_price_calculation_l377_37780


namespace three_intersections_iff_zero_l377_37725

/-- The number of distinct intersection points between the curves x^2 - y^2 = a^2 and (x-1)^2 + y^2 = 1 -/
def intersection_count (a : ℝ) : ℕ :=
  sorry

/-- The condition for exactly three distinct intersection points -/
def has_three_intersections (a : ℝ) : Prop :=
  intersection_count a = 3

theorem three_intersections_iff_zero (a : ℝ) :
  has_three_intersections a ↔ a = 0 :=
sorry

end three_intersections_iff_zero_l377_37725


namespace circle_intersection_distance_l377_37717

-- Define the circles and their properties
variable (r R : ℝ)
variable (d : ℝ)

-- Hypotheses
variable (h1 : r > 0)
variable (h2 : R > 0)
variable (h3 : r < R)
variable (h4 : d > 0)

-- Define the intersection property
variable (intersection : ∃ (x : ℝ × ℝ), (x.1^2 + x.2^2 = r^2) ∧ ((x.1 - d)^2 + x.2^2 = R^2))

-- Theorem statement
theorem circle_intersection_distance : R - r < d ∧ d < r + R := by
  sorry

end circle_intersection_distance_l377_37717


namespace martha_juice_bottles_l377_37781

/-- Calculates the number of juice bottles left after a week -/
def bottles_left (initial_refrigerator : ℕ) (initial_pantry : ℕ) (bought : ℕ) (drunk : ℕ) : ℕ :=
  initial_refrigerator + initial_pantry + bought - drunk

/-- Proves that given the initial conditions and actions, 10 bottles are left -/
theorem martha_juice_bottles : bottles_left 4 4 5 3 = 10 := by
  sorry

end martha_juice_bottles_l377_37781


namespace unique_bijective_function_satisfying_equation_l377_37751

-- Define the property that a function f must satisfy
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + f y) = f x + y

-- State the theorem
theorem unique_bijective_function_satisfying_equation :
  ∃! f : ℝ → ℝ, Function.Bijective f ∧ SatisfiesEquation f ∧ (∀ x : ℝ, f x = x) :=
sorry

end unique_bijective_function_satisfying_equation_l377_37751


namespace other_number_proof_l377_37740

/-- Prove that given two positive integers, 24 and x, if their HCF (h) is 17 and their LCM (l) is 312, then x = 221. -/
theorem other_number_proof (x : ℕ) (h l : ℕ) : 
  x > 0 ∧ h > 0 ∧ l > 0 ∧ 
  h = Nat.gcd 24 x ∧ 
  l = Nat.lcm 24 x ∧ 
  h = 17 ∧ 
  l = 312 → 
  x = 221 := by
sorry

end other_number_proof_l377_37740


namespace shortest_path_on_sphere_intersection_l377_37784

/-- The shortest path on a sphere's surface between the two most distant points of its intersection with a plane --/
theorem shortest_path_on_sphere_intersection (R d : ℝ) (h1 : R = 2) (h2 : d = 1) :
  let r := Real.sqrt (R^2 - d^2)
  let θ := 2 * Real.arccos (d / R)
  θ / (2 * Real.pi) * (2 * Real.pi * r) = Real.pi / 3 :=
by sorry

end shortest_path_on_sphere_intersection_l377_37784


namespace sum_of_extreme_prime_factors_of_1540_l377_37798

theorem sum_of_extreme_prime_factors_of_1540 :
  ∃ (smallest largest : ℕ),
    smallest.Prime ∧ 
    largest.Prime ∧
    smallest ∣ 1540 ∧ 
    largest ∣ 1540 ∧
    (∀ p : ℕ, p.Prime → p ∣ 1540 → p ≥ smallest) ∧
    (∀ p : ℕ, p.Prime → p ∣ 1540 → p ≤ largest) ∧
    smallest + largest = 13 :=
by sorry

end sum_of_extreme_prime_factors_of_1540_l377_37798


namespace constrained_line_generates_surface_l377_37769

/-- A line parallel to the plane y=z, intersecting two parabolas -/
structure ConstrainedLine where
  /-- The line is parallel to the plane y=z -/
  parallel_to_yz : ℝ → ℝ → ℝ → Prop
  /-- The line intersects the parabola 2x=y², z=0 -/
  meets_parabola1 : ℝ → ℝ → ℝ → Prop
  /-- The line intersects the parabola 3x=z², y=0 -/
  meets_parabola2 : ℝ → ℝ → ℝ → Prop

/-- The surface generated by the constrained line -/
def generated_surface (x y z : ℝ) : Prop :=
  x = (y - z) * (y / 2 - z / 3)

/-- Theorem stating that the constrained line generates the specified surface -/
theorem constrained_line_generates_surface (L : ConstrainedLine) :
  ∀ x y z, L.parallel_to_yz x y z → L.meets_parabola1 x y z → L.meets_parabola2 x y z →
  generated_surface x y z :=
sorry

end constrained_line_generates_surface_l377_37769


namespace last_score_is_90_l377_37796

def scores : List Nat := [72, 77, 85, 90, 94]

def isValidOrder (order : List Nat) : Prop :=
  order.length = 5 ∧
  order.toFinset = scores.toFinset ∧
  ∀ k : Fin 5, (order.take k.val.succ).sum % k.val.succ = 0

theorem last_score_is_90 :
  ∀ order : List Nat, isValidOrder order → order.getLast? = some 90 := by
  sorry

end last_score_is_90_l377_37796


namespace min_width_proof_l377_37701

/-- The minimum width of a rectangular area satisfying given conditions -/
def min_width : ℝ := 5

/-- The length of the rectangular area in terms of its width -/
def length (w : ℝ) : ℝ := 2 * w + 10

/-- The area of the rectangular region -/
def area (w : ℝ) : ℝ := w * length w

theorem min_width_proof :
  (∀ w : ℝ, w > 0 → area w ≥ 120 → w ≥ min_width) ∧
  (area min_width ≥ 120) ∧
  (min_width > 0) :=
sorry

end min_width_proof_l377_37701


namespace penelope_savings_l377_37758

/-- The amount of money Penelope saves daily, in dollars. -/
def daily_savings : ℕ := 24

/-- The number of days in a year (assuming it's not a leap year). -/
def days_in_year : ℕ := 365

/-- The total amount Penelope saves in a year. -/
def total_savings : ℕ := daily_savings * days_in_year

/-- Theorem: Penelope's total savings after one year is $8,760. -/
theorem penelope_savings : total_savings = 8760 := by
  sorry

end penelope_savings_l377_37758


namespace wedge_volume_l377_37742

/-- The volume of a wedge cut from a cylindrical log --/
theorem wedge_volume (d h θ : ℝ) (hd : d = 20) (hh : h = 20) (hθ : θ = 30 * π / 180) :
  let r := d / 2
  let cylinder_volume := π * r^2 * h
  let wedge_volume := (θ / (2 * π)) * cylinder_volume
  wedge_volume = 250 * π := by sorry

end wedge_volume_l377_37742


namespace instantaneous_velocity_at_3_l377_37795

/-- Represents the position function of a particle -/
def S (t : ℝ) : ℝ := 2 * t^3

/-- Represents the velocity function of a particle -/
def V (t : ℝ) : ℝ := 6 * t^2

theorem instantaneous_velocity_at_3 :
  V 3 = 54 :=
sorry

end instantaneous_velocity_at_3_l377_37795


namespace tetrahedron_volume_ratio_sum_l377_37787

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron where
  -- Add necessary fields here
  dummy : Unit

/-- Represents the smaller tetrahedron formed by the centers of the faces of a regular tetrahedron -/
def smaller_tetrahedron (t : RegularTetrahedron) : RegularTetrahedron :=
  sorry

/-- The volume ratio of the smaller tetrahedron to the original tetrahedron -/
def volume_ratio (t : RegularTetrahedron) : ℚ :=
  sorry

/-- States that m and n are relatively prime positive integers -/
def are_relatively_prime (m n : ℕ) : Prop :=
  sorry

theorem tetrahedron_volume_ratio_sum (t : RegularTetrahedron) (m n : ℕ) :
  volume_ratio t = m / n →
  are_relatively_prime m n →
  m + n = 28 :=
sorry

end tetrahedron_volume_ratio_sum_l377_37787


namespace sum_of_binary_numbers_l377_37753

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 101₍₂₎ -/
def binary_101 : List Bool := [true, false, true]

/-- The binary representation of 110₍₂₎ -/
def binary_110 : List Bool := [false, true, true]

theorem sum_of_binary_numbers :
  binary_to_decimal binary_101 + binary_to_decimal binary_110 = 11 := by
  sorry


end sum_of_binary_numbers_l377_37753


namespace worker_count_l377_37797

theorem worker_count (work_amount : ℝ) : ∃ (workers : ℕ), 
  (workers : ℝ) * 75 = work_amount ∧ 
  (workers + 10 : ℝ) * 65 = work_amount ∧ 
  workers = 65 := by
sorry

end worker_count_l377_37797


namespace sams_dimes_proof_l377_37765

/-- The number of dimes Sam's dad gave him -/
def dimes_from_dad (initial_dimes final_dimes : ℕ) : ℕ :=
  final_dimes - initial_dimes

theorem sams_dimes_proof (initial_dimes final_dimes : ℕ) 
  (h1 : initial_dimes = 9)
  (h2 : final_dimes = 16) : 
  dimes_from_dad initial_dimes final_dimes = 7 := by
  sorry

end sams_dimes_proof_l377_37765


namespace quadratic_always_nonnegative_implies_a_geq_four_l377_37786

theorem quadratic_always_nonnegative_implies_a_geq_four (a : ℝ) :
  (∀ x : ℝ, x^2 - 4*x + a ≥ 0) → a ≥ 4 := by
  sorry

end quadratic_always_nonnegative_implies_a_geq_four_l377_37786


namespace intersection_empty_implies_a_greater_than_neg_four_l377_37710

def A (a : ℝ) : Set ℝ := {x | x^2 + (a + 2) * x + 1 = 0}
def B : Set ℝ := {x | x > 0}

theorem intersection_empty_implies_a_greater_than_neg_four (a : ℝ) :
  A a ∩ B = ∅ → a > -4 := by sorry

end intersection_empty_implies_a_greater_than_neg_four_l377_37710


namespace sara_height_l377_37785

/-- Proves that Sara's height is 45 inches given the relative heights of Sara, Joe, Roy, Mark, and Julie. -/
theorem sara_height (
  julie_height : ℕ)
  (mark_taller_than_julie : ℕ)
  (roy_taller_than_mark : ℕ)
  (joe_taller_than_roy : ℕ)
  (sara_taller_than_joe : ℕ)
  (h_julie : julie_height = 33)
  (h_mark : mark_taller_than_julie = 1)
  (h_roy : roy_taller_than_mark = 2)
  (h_joe : joe_taller_than_roy = 3)
  (h_sara : sara_taller_than_joe = 6) :
  julie_height + mark_taller_than_julie + roy_taller_than_mark + joe_taller_than_roy + sara_taller_than_joe = 45 := by
  sorry

end sara_height_l377_37785


namespace problem_triangle_integer_segments_l377_37768

/-- Represents a right triangle with integer side lengths -/
structure RightTriangle where
  de : ℕ
  ef : ℕ

/-- Counts the number of distinct integer lengths of line segments
    from vertex E to points on the hypotenuse DF -/
def countIntegerSegments (t : RightTriangle) : ℕ :=
  sorry

/-- The specific right triangle in the problem -/
def problemTriangle : RightTriangle :=
  { de := 24, ef := 25 }

/-- The main theorem stating that the number of distinct integer lengths
    of line segments from E to DF in the problem triangle is 14 -/
theorem problem_triangle_integer_segments :
  countIntegerSegments problemTriangle = 14 := by
  sorry

end problem_triangle_integer_segments_l377_37768


namespace a_six_plus_seven_l377_37764

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the sequence a_n
def a (n : ℕ) : ℝ := f n

-- State the properties of f
axiom f_odd (x : ℝ) : f (-x) = -f x
axiom f_periodic (x : ℝ) : f (x + 3) = f x
axiom f_neg_two : f (-2) = -3

-- State the theorem
theorem a_six_plus_seven : a f 6 + a f 7 = -3 := by
  sorry

end a_six_plus_seven_l377_37764


namespace smallest_k_no_real_roots_l377_37714

theorem smallest_k_no_real_roots : 
  ∃ (k : ℤ), k = 2 ∧ 
  (∀ x : ℝ, 2 * x * (k * x - 4) - x^2 + 6 ≠ 0) ∧
  (∀ m : ℤ, m < k → ∃ x : ℝ, 2 * x * (m * x - 4) - x^2 + 6 = 0) :=
sorry

end smallest_k_no_real_roots_l377_37714


namespace probability_at_least_one_girl_l377_37750

def number_of_boys : ℕ := 3
def number_of_girls : ℕ := 2
def total_students : ℕ := number_of_boys + number_of_girls
def students_selected : ℕ := 2

theorem probability_at_least_one_girl :
  (Nat.choose total_students students_selected - Nat.choose number_of_boys students_selected) /
  Nat.choose total_students students_selected = 7 / 10 := by
  sorry

end probability_at_least_one_girl_l377_37750


namespace correct_result_l377_37747

/-- Represents a five-digit number -/
structure FiveDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  e : Nat
  h1 : a ≥ 1 ∧ a ≤ 9
  h2 : b ≥ 0 ∧ b ≤ 9
  h3 : c ≥ 0 ∧ c ≤ 9
  h4 : d ≥ 0 ∧ d ≤ 9
  h5 : e ≥ 0 ∧ e ≤ 9

def reverseNumber (n : FiveDigitNumber) : Nat :=
  n.e * 10000 + n.d * 1000 + n.c * 100 + n.b * 10 + n.a

def originalNumber (n : FiveDigitNumber) : Nat :=
  n.a * 10000 + n.b * 1000 + n.c * 100 + n.d * 10 + n.e

theorem correct_result (n : FiveDigitNumber) 
  (h : reverseNumber n - originalNumber n = 34056) :
  n.e > n.a ∧ 
  n.e - n.a = 3 ∧ 
  (n.a - n.e) % 10 = 6 ∧ 
  n.b > n.d :=
sorry

end correct_result_l377_37747


namespace x_squared_plus_y_squared_l377_37726

theorem x_squared_plus_y_squared (x y : ℝ) 
  (h1 : x * y = 12)
  (h2 : x^2 * y + x * y^2 + x + y = 99) : 
  x^2 + y^2 = 5745 / 169 := by
sorry

end x_squared_plus_y_squared_l377_37726


namespace point_B_coordinates_l377_37706

-- Define the point A and vector a
def A : ℝ × ℝ := (2, 4)
def a : ℝ × ℝ := (3, 4)

-- Define the relation between AB and a
def AB_relation (B : ℝ × ℝ) : Prop :=
  (B.1 - A.1, B.2 - A.2) = (2 * a.1, 2 * a.2)

-- Theorem stating that B has coordinates (8, 12)
theorem point_B_coordinates :
  ∃ B : ℝ × ℝ, AB_relation B ∧ B = (8, 12) := by sorry

end point_B_coordinates_l377_37706


namespace stock_price_change_l377_37760

theorem stock_price_change (x : ℝ) : 
  (1 - x / 100) * 1.10 = 1.012 → x = 8 := by
  sorry

end stock_price_change_l377_37760


namespace poles_count_l377_37704

/-- The number of telephone poles given the interval distance and total distance -/
def num_poles (interval : ℕ) (total_distance : ℕ) : ℕ :=
  (total_distance / interval) + 1

/-- Theorem stating that the number of poles is 61 given the specific conditions -/
theorem poles_count : num_poles 25 1500 = 61 := by
  sorry

end poles_count_l377_37704


namespace tables_needed_for_children_twenty_tables_needed_l377_37744

theorem tables_needed_for_children (num_children : ℕ) (table_capacity : ℕ) (num_tables : ℕ) : Prop :=
  num_children > 0 ∧ 
  table_capacity > 0 ∧ 
  num_tables * table_capacity ≥ num_children ∧ 
  (num_tables - 1) * table_capacity < num_children

theorem twenty_tables_needed : tables_needed_for_children 156 8 20 := by
  sorry

end tables_needed_for_children_twenty_tables_needed_l377_37744


namespace smallest_angle_in_special_right_triangle_l377_37782

theorem smallest_angle_in_special_right_triangle :
  ∀ (a b : ℝ), 
  0 < a ∧ 0 < b →  -- Angles are positive
  a + b = 90 →     -- Sum of acute angles in a right triangle
  a / b = 3 / 2 →  -- Ratio of angles is 3:2
  min a b = 36 :=  -- The smallest angle is 36°
by
  sorry

end smallest_angle_in_special_right_triangle_l377_37782


namespace cubic_extreme_values_l377_37734

/-- Given a cubic function f(x) = x^3 - px^2 - qx that passes through (1,0),
    prove that its maximum value is 4/27 and its minimum value is 0. -/
theorem cubic_extreme_values (p q : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^3 - p*x^2 - q*x
  (f 1 = 0) →
  (∃ x, f x = 4/27) ∧ (∀ y, f y ≤ 4/27) ∧ (∃ z, f z = 0) ∧ (∀ w, f w ≥ 0) :=
by sorry


end cubic_extreme_values_l377_37734


namespace racket_price_l377_37700

theorem racket_price (total_spent sneakers_cost outfit_cost : ℕ) 
  (h1 : total_spent = 750)
  (h2 : sneakers_cost = 200)
  (h3 : outfit_cost = 250) :
  total_spent - sneakers_cost - outfit_cost = 300 := by
  sorry

end racket_price_l377_37700


namespace angle_range_given_sine_l377_37708

theorem angle_range_given_sine (α : Real) (h1 : 0 < α ∧ α < Real.pi / 2) (h2 : Real.sin α = 0.58) :
  Real.pi / 6 < α ∧ α < Real.pi / 4 := by
  sorry

end angle_range_given_sine_l377_37708


namespace chord_length_theorem_l377_37711

theorem chord_length_theorem (k : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = 1 ∧ x + y = 2*k - 1) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 + y₁^2 = 1 ∧ 
    x₂^2 + y₂^2 = 1 ∧ 
    x₁ + y₁ = 2*k - 1 ∧ 
    x₂ + y₂ = 2*k - 1 ∧ 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 2) →
  k = 0 ∨ k = 1 := by
sorry

end chord_length_theorem_l377_37711


namespace existence_of_four_integers_l377_37757

theorem existence_of_four_integers (a : Fin 97 → ℕ+) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) : 
  ∃ w x y z : Fin 97, w ≠ x ∧ y ≠ z ∧ 1984 ∣ ((a w).val - (a x).val) * ((a y).val - (a z).val) := by
  sorry

end existence_of_four_integers_l377_37757


namespace sqrt_three_not_in_P_l377_37790

-- Define the set P
def P : Set ℝ := {x | x^2 - Real.sqrt 2 * x ≤ 0}

-- State the theorem
theorem sqrt_three_not_in_P : Real.sqrt 3 ∉ P := by
  sorry

end sqrt_three_not_in_P_l377_37790


namespace quadratic_root_and_coefficient_l377_37702

theorem quadratic_root_and_coefficient (m : ℝ) :
  (∃ x, x^2 + m*x + 2 = 0 ∧ x = -2) →
  (∃ y, y^2 + m*y + 2 = 0 ∧ y = -1) ∧ m = 3 := by
sorry

end quadratic_root_and_coefficient_l377_37702


namespace product_expansion_sum_l377_37777

theorem product_expansion_sum (a b c d : ℝ) :
  (∀ x, (2*x^2 - 3*x + 5)*(5 - x) = a*x^3 + b*x^2 + c*x + d) →
  a + b + c + d = 16 := by
sorry

end product_expansion_sum_l377_37777


namespace negation_of_existence_negation_of_quadratic_inequality_l377_37763

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_quadratic_inequality : 
  (¬ ∃ x₀ : ℝ, x₀^2 - x₀ + (1/4 : ℝ) ≤ 0) ↔ 
  (∀ x : ℝ, x^2 - x + (1/4 : ℝ) > 0) :=
by sorry

end negation_of_existence_negation_of_quadratic_inequality_l377_37763


namespace zoo_visitors_l377_37745

theorem zoo_visitors (num_cars : ℝ) (people_per_car : ℝ) 
  (h1 : num_cars = 3.0) 
  (h2 : people_per_car = 63.0) : 
  num_cars * people_per_car = 189.0 := by
  sorry

end zoo_visitors_l377_37745


namespace multiply_by_fraction_l377_37743

theorem multiply_by_fraction (a b c : ℝ) (h : a * b = c) :
  (b / 10) * a = c / 10 := by
  sorry

end multiply_by_fraction_l377_37743


namespace two_language_speakers_l377_37709

/-- Represents the number of students who can speak a given language -/
structure LanguageSpeakers where
  gujarati : ℕ
  hindi : ℕ
  marathi : ℕ

/-- Represents the number of students who can speak exactly two languages -/
structure BilingualStudents where
  gujarati_hindi : ℕ
  gujarati_marathi : ℕ
  hindi_marathi : ℕ

/-- The theorem to be proved -/
theorem two_language_speakers
  (total_students : ℕ)
  (speakers : LanguageSpeakers)
  (trilingual : ℕ)
  (h_total : total_students = 22)
  (h_gujarati : speakers.gujarati = 6)
  (h_hindi : speakers.hindi = 15)
  (h_marathi : speakers.marathi = 6)
  (h_trilingual : trilingual = 1)
  : ∃ (bilingual : BilingualStudents),
    bilingual.gujarati_hindi + bilingual.gujarati_marathi + bilingual.hindi_marathi = 6 ∧
    total_students = speakers.gujarati + speakers.hindi + speakers.marathi -
      (bilingual.gujarati_hindi + bilingual.gujarati_marathi + bilingual.hindi_marathi) +
      trilingual :=
by sorry

end two_language_speakers_l377_37709


namespace transformed_is_ellipse_l377_37721

-- Define the original circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the scaling transformation
def scaling_transformation (x y : ℝ) : ℝ × ℝ := (5*x, 4*y)

-- Define the resulting equation after transformation
def transformed_equation (x' y' : ℝ) : Prop :=
  ∃ x y, circle_equation x y ∧ scaling_transformation x y = (x', y')

-- Statement to prove
theorem transformed_is_ellipse :
  ∃ a b, a > b ∧ a = 5 ∧
  ∀ x' y', transformed_equation x' y' ↔ (x'^2 / a^2) + (y'^2 / b^2) = 1 :=
sorry

end transformed_is_ellipse_l377_37721


namespace exists_hyperbola_segment_with_midpoint_l377_37728

/-- The hyperbola equation -/
def on_hyperbola (x y : ℝ) : Prop := x^2 - y^2/9 = 1

/-- The midpoint of two points -/
def is_midpoint (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₀ = (x₁ + x₂)/2 ∧ y₀ = (y₁ + y₂)/2

theorem exists_hyperbola_segment_with_midpoint :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    on_hyperbola x₁ y₁ ∧
    on_hyperbola x₂ y₂ ∧
    is_midpoint (-1) (-4) x₁ y₁ x₂ y₂ :=
  sorry

end exists_hyperbola_segment_with_midpoint_l377_37728


namespace jason_egg_consumption_l377_37759

/-- The number of eggs Jason consumes in two weeks -/
def eggs_consumed_in_two_weeks : ℕ :=
  let eggs_per_omelet : ℕ := 3
  let days_in_two_weeks : ℕ := 14
  eggs_per_omelet * days_in_two_weeks

/-- Theorem stating that Jason consumes 42 eggs in two weeks -/
theorem jason_egg_consumption :
  eggs_consumed_in_two_weeks = 42 := by
  sorry

end jason_egg_consumption_l377_37759


namespace triangle_arithmetic_sequence_l377_37756

theorem triangle_arithmetic_sequence (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- c cos A, b cos B, a cos C form an arithmetic sequence
  2 * b * Real.cos B = c * Real.cos A + a * Real.cos C →
  -- Given conditions
  a + c = 3 * Real.sqrt 3 / 2 →
  b = Real.sqrt 3 →
  -- Conclusions
  B = π / 3 ∧
  (1 / 2 * a * c * Real.sin B = 5 * Real.sqrt 3 / 16) :=
by sorry

end triangle_arithmetic_sequence_l377_37756


namespace power_inequality_l377_37705

theorem power_inequality (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  a^6 + b^6 ≥ a*b*(a^4 + b^4) := by
  sorry

end power_inequality_l377_37705


namespace base_ratio_l377_37722

/-- An isosceles trapezoid with bases a and b (a > b) and height h -/
structure IsoscelesTrapezoid where
  a : ℝ
  b : ℝ
  h : ℝ
  a_gt_b : a > b
  h_gt_zero : h > 0

/-- The property that the height divides the larger base in ratio 1:3 -/
def height_divides_base (t : IsoscelesTrapezoid) : Prop :=
  ∃ x : ℝ, x > 0 ∧ (3 * x = t.a - x)

/-- The theorem stating the ratio of bases -/
theorem base_ratio (t : IsoscelesTrapezoid) 
  (h : height_divides_base t) : t.a / t.b = 3 := by
  sorry

end base_ratio_l377_37722


namespace high_school_ratio_problem_l377_37724

theorem high_school_ratio_problem (initial_boys initial_girls : ℕ) 
  (boys_left girls_left : ℕ) (final_boys final_girls : ℕ) : 
  (initial_boys : ℚ) / initial_girls = 3 / 4 →
  girls_left = 2 * boys_left →
  boys_left = 10 →
  (final_boys : ℚ) / final_girls = 4 / 5 →
  final_boys = initial_boys - boys_left →
  final_girls = initial_girls - girls_left →
  initial_boys = 90 := by
sorry


end high_school_ratio_problem_l377_37724


namespace flamingo_tail_feathers_l377_37719

/-- The number of tail feathers per flamingo given the conditions for making feather boas --/
theorem flamingo_tail_feathers 
  (num_boas : ℕ) 
  (feathers_per_boa : ℕ) 
  (num_flamingoes : ℕ) 
  (safe_pluck_percentage : ℚ) : ℕ :=
  sorry

#check flamingo_tail_feathers 12 200 480 (1/4) = 20

end flamingo_tail_feathers_l377_37719


namespace shirt_ratio_l377_37762

theorem shirt_ratio (brian_shirts andrew_shirts steven_shirts : ℕ) :
  brian_shirts = 3 →
  andrew_shirts = 6 * brian_shirts →
  steven_shirts = 72 →
  steven_shirts / andrew_shirts = 4 :=
by sorry

end shirt_ratio_l377_37762


namespace grade_assignments_count_l377_37789

/-- The number of possible grades a professor can assign to each student. -/
def num_grades : ℕ := 4

/-- The number of students in the class. -/
def num_students : ℕ := 15

/-- The number of ways to assign grades to all students in the class. -/
def num_grade_assignments : ℕ := num_grades ^ num_students

/-- Theorem stating that the number of ways to assign grades is 4^15. -/
theorem grade_assignments_count :
  num_grade_assignments = 1073741824 := by
  sorry

end grade_assignments_count_l377_37789


namespace problem_solution_l377_37775

-- Define the set of possible values
def S : Set ℕ := {0, 1, 3}

-- Define the properties
def prop1 (a b c : ℕ) : Prop := a ≠ 3
def prop2 (a b c : ℕ) : Prop := b = 3
def prop3 (a b c : ℕ) : Prop := c ≠ 0

theorem problem_solution (a b c : ℕ) :
  {a, b, c} = S →
  (prop1 a b c ∨ prop2 a b c ∨ prop3 a b c) →
  (¬(prop1 a b c ∧ prop2 a b c) ∧ ¬(prop1 a b c ∧ prop3 a b c) ∧ ¬(prop2 a b c ∧ prop3 a b c)) →
  100 * a + 10 * b + c = 301 := by
  sorry

end problem_solution_l377_37775


namespace sum_after_transformation_l377_37712

theorem sum_after_transformation (S a b : ℝ) (h : a + b = S) :
  3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 := by
  sorry

end sum_after_transformation_l377_37712


namespace principal_amount_proof_l377_37741

-- Define the parameters of the investment
def interest_rate : ℚ := 5 / 100
def investment_duration : ℕ := 5
def final_amount : ℚ := 10210.25

-- Define the compound interest formula
def compound_interest (principal : ℚ) : ℚ :=
  principal * (1 + interest_rate) ^ investment_duration

-- State the theorem
theorem principal_amount_proof :
  ∃ (principal : ℚ), 
    compound_interest principal = final_amount ∧ 
    (principal ≥ 7999.5 ∧ principal ≤ 8000.5) := by
  sorry

end principal_amount_proof_l377_37741


namespace folded_rectangle_perimeter_l377_37776

/-- Represents a rectangular piece of paper --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- The perimeter of the rectangle when folded along its width --/
def perimeterFoldedWidth (r : Rectangle) : ℝ := 2 * r.length + r.width

/-- The perimeter of the rectangle when folded along its length --/
def perimeterFoldedLength (r : Rectangle) : ℝ := 2 * r.width + r.length

/-- The area of the rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem folded_rectangle_perimeter 
  (r : Rectangle) 
  (h1 : area r = 140)
  (h2 : perimeterFoldedWidth r = 34) :
  perimeterFoldedLength r = 38 := by
  sorry

#check folded_rectangle_perimeter

end folded_rectangle_perimeter_l377_37776


namespace triangle_area_with_median_l377_37716

/-- Given a triangle PQR with side lengths and median, calculate its area -/
theorem triangle_area_with_median (P Q R M : ℝ × ℝ) : 
  let PQ := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  let PR := Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2)
  let PM := Real.sqrt ((M.1 - P.1)^2 + (M.2 - P.2)^2)
  let QR := Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2)
  let s := (PQ + PR + QR) / 2
  let area := Real.sqrt (s * (s - PQ) * (s - PR) * (s - QR))
  PQ = 9 →
  PR = 17 →
  PM = 13 →
  M = ((Q.1 + R.1) / 2, (Q.2 + R.2) / 2) →
  area = A :=
by sorry

#check triangle_area_with_median

end triangle_area_with_median_l377_37716


namespace sum_4_inclusive_numbers_eq_1883_l377_37707

/-- Returns true if the number contains the digit 4 -/
def contains4 (n : ℕ) : Bool :=
  n.repr.contains '4'

/-- Returns true if the number is 4-inclusive (multiple of 4 or contains 4) -/
def is4Inclusive (n : ℕ) : Bool :=
  n % 4 = 0 || contains4 n

/-- The sum of all 4-inclusive numbers in the range [0, 100] -/
def sum4InclusiveNumbers : ℕ :=
  (List.range 101).filter is4Inclusive |>.sum

theorem sum_4_inclusive_numbers_eq_1883 : sum4InclusiveNumbers = 1883 := by
  sorry

end sum_4_inclusive_numbers_eq_1883_l377_37707


namespace extreme_value_condition_negative_interval_condition_l377_37749

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

-- Part I
theorem extreme_value_condition (a b : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), 0 < |x - 1| ∧ |x - 1| < ε → f a b x ≤ f a b 1) ∧
  f a b 1 = 10 →
  a = 4 ∧ b = -11 :=
sorry

-- Part II
theorem negative_interval_condition (b : ℝ) :
  (∀ (x : ℝ), x ∈ Set.Icc 1 2 → f (-1) b x < 0) →
  b < -5/2 :=
sorry

end extreme_value_condition_negative_interval_condition_l377_37749


namespace min_sum_given_product_l377_37799

theorem min_sum_given_product (x y : ℤ) (h : x * y = 144) : 
  ∀ a b : ℤ, a * b = 144 → x + y ≤ a + b ∧ ∃ c d : ℤ, c * d = 144 ∧ c + d = -145 :=
by sorry

end min_sum_given_product_l377_37799


namespace least_positive_integer_with_property_l377_37727

/-- Represents a three-digit number as 100a + 10b + c -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  a_nonzero : a ≠ 0
  b_digit : b < 10
  c_digit : c < 10

/-- The value of a ThreeDigitNumber -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.b + n.c

/-- The value of a ThreeDigitNumber with the leftmost digit removed -/
def ThreeDigitNumber.valueWithoutLeftmost (n : ThreeDigitNumber) : Nat :=
  10 * n.b + n.c

theorem least_positive_integer_with_property :
  ∃ (n : ThreeDigitNumber),
    n.value = 725 ∧
    n.valueWithoutLeftmost = n.value / 29 ∧
    ∀ (m : ThreeDigitNumber), m.valueWithoutLeftmost = m.value / 29 → n.value ≤ m.value :=
sorry

end least_positive_integer_with_property_l377_37727


namespace incorrect_expression_l377_37723

theorem incorrect_expression (x y : ℝ) (h : x / y = 2 / 5) :
  (x + 3 * y) / x ≠ 17 / 2 := by
  sorry

end incorrect_expression_l377_37723


namespace prime_diff_cubes_sum_squares_l377_37761

theorem prime_diff_cubes_sum_squares (p : ℕ) (a b : ℕ) :
  Prime p → p = a^3 - b^3 → ∃ (c d : ℤ), p = c^2 + 3 * d^2 := by
  sorry

end prime_diff_cubes_sum_squares_l377_37761


namespace suraj_innings_l377_37718

/-- Represents the cricket problem for Suraj's innings --/
def cricket_problem (n : ℕ) : Prop :=
  let A : ℚ := 10  -- Initial average (derived from the new average minus the increase)
  let new_average : ℚ := 16  -- New average after the last innings
  let runs_increase : ℚ := 6  -- Increase in average
  let last_innings_runs : ℕ := 112  -- Runs scored in the last innings
  
  -- The equation representing the new average
  (n * A + last_innings_runs) / (n + 1) = new_average ∧
  -- The equation representing the increase in average
  new_average = A + runs_increase

/-- Theorem stating that the number of innings before the last one is 16 --/
theorem suraj_innings : cricket_problem 16 := by sorry

end suraj_innings_l377_37718


namespace acute_triangle_side_range_l377_37792

/-- Given an acute triangle ABC with side lengths a = 2 and b = 3, 
    prove that the side length c satisfies √5 < c < √13 -/
theorem acute_triangle_side_range (a b c : ℝ) : 
  a = 2 → b = 3 → 
  (a^2 + b^2 > c^2) → (a^2 + c^2 > b^2) → (b^2 + c^2 > a^2) →
  Real.sqrt 5 < c ∧ c < Real.sqrt 13 := by
  sorry

end acute_triangle_side_range_l377_37792


namespace divisible_by_six_l377_37748

theorem divisible_by_six (a : ℤ) : ∃ k : ℤ, a^3 + 11*a = 6*k := by
  sorry

end divisible_by_six_l377_37748


namespace inequality_holds_iff_p_in_range_l377_37767

theorem inequality_holds_iff_p_in_range :
  ∀ p : ℝ, p ≥ 0 →
  (∀ q : ℝ, q > 0 → (4 * (p * q^2 + p^2 * q + 4 * q^2 + 4 * p * q)) / (p + q) > 3 * p^2 * q) ↔
  p ∈ Set.Ici 0 ∩ Set.Iio 4 := by
sorry

end inequality_holds_iff_p_in_range_l377_37767


namespace wooden_statue_cost_l377_37703

/-- The cost of a wooden statue given Theodore's production and earnings. -/
theorem wooden_statue_cost :
  let stone_statues : ℕ := 10
  let wooden_statues : ℕ := 20
  let stone_cost : ℚ := 20
  let tax_rate : ℚ := 1/10
  let total_earnings : ℚ := 270
  ∃ (wooden_cost : ℚ),
    (1 - tax_rate) * (stone_statues * stone_cost + wooden_statues * wooden_cost) = total_earnings ∧
    wooden_cost = 5 := by
  sorry

end wooden_statue_cost_l377_37703


namespace nikkas_stamp_collection_l377_37778

theorem nikkas_stamp_collection :
  ∀ (total_stamps : ℕ) 
    (chinese_percentage : ℚ) 
    (us_percentage : ℚ) 
    (japanese_stamps : ℕ),
  chinese_percentage = 35 / 100 →
  us_percentage = 20 / 100 →
  japanese_stamps = 45 →
  (1 - chinese_percentage - us_percentage) * total_stamps = japanese_stamps →
  total_stamps = 100 := by
sorry

end nikkas_stamp_collection_l377_37778


namespace tims_manicure_cost_l377_37779

/-- The total cost of a manicure with tip -/
def total_cost (base_cost : ℝ) (tip_percentage : ℝ) : ℝ :=
  base_cost * (1 + tip_percentage)

/-- Theorem: Tim's total payment for a $30 manicure with a 30% tip is $39 -/
theorem tims_manicure_cost :
  total_cost 30 0.3 = 39 := by
  sorry

end tims_manicure_cost_l377_37779


namespace f_positive_range_min_k_for_f_plus_k_positive_l377_37713

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x - a * x

theorem f_positive_range (a : ℝ) :
  (∀ x, f a x > 0 ↔ x > 0) ∨
  (∀ x, f a x > 0 ↔ (x > 0 ∨ x < Real.log a)) ∨
  (∀ x, f a x > 0 ↔ (x > Real.log a ∨ x < 0)) :=
sorry

theorem min_k_for_f_plus_k_positive :
  ∃! k : ℕ, k > 0 ∧ ∀ x, f 2 x + k > 0 ∧ ∀ m : ℕ, m < k → ∃ y, f 2 y + m ≤ 0 :=
sorry

end f_positive_range_min_k_for_f_plus_k_positive_l377_37713


namespace fraction_sum_theorem_l377_37772

theorem fraction_sum_theorem (a b c : ℝ) 
  (h : a / (30 - a) + b / (70 - b) + c / (75 - c) = 8) :
  6 / (30 - a) + 14 / (70 - b) + 15 / (75 - c) = 11 / 5 := by
  sorry

end fraction_sum_theorem_l377_37772


namespace areas_theorem_l377_37720

-- Define the areas A, B, and C
def A : ℝ := sorry
def B : ℝ := sorry
def C : ℝ := sorry

-- State the theorem
theorem areas_theorem :
  -- Condition for A: square with diagonal 2√2
  (∃ (s : ℝ), s * s = A ∧ s * Real.sqrt 2 = 2 * Real.sqrt 2) →
  -- Condition for B: rectangle with given vertices
  (∃ (w h : ℝ), w * h = B ∧ w = 4 ∧ h = 2) →
  -- Condition for C: triangle formed by axes and line y = -x/2 + 2
  (∃ (base height : ℝ), (1/2) * base * height = C ∧ base = 4 ∧ height = 2) →
  -- Conclusion
  A = 4 ∧ B = 8 ∧ C = 4 := by
sorry


end areas_theorem_l377_37720


namespace two_digit_divisible_by_3_and_4_with_tens_greater_than_ones_l377_37788

/-- A function that returns true if a natural number is a two-digit number -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A function that returns the tens digit of a natural number -/
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

/-- A function that returns the ones digit of a natural number -/
def ones_digit (n : ℕ) : ℕ := n % 10

/-- The main theorem to be proved -/
theorem two_digit_divisible_by_3_and_4_with_tens_greater_than_ones :
  ∃! (s : Finset ℕ), 
    s.card = 4 ∧ 
    (∀ n ∈ s, 
      n > 0 ∧ 
      is_two_digit n ∧ 
      n % 3 = 0 ∧ 
      n % 4 = 0 ∧ 
      tens_digit n > ones_digit n) := by
  sorry

end two_digit_divisible_by_3_and_4_with_tens_greater_than_ones_l377_37788


namespace a_divides_2b_l377_37783

theorem a_divides_2b (a b : ℕ+) 
  (h : ∃ (S : Set (ℕ+ × ℕ+)), Set.Infinite S ∧ 
    ∀ (p : ℕ+ × ℕ+), p ∈ S → 
      ∃ (r s : ℕ+), (p.1.val ^ 2 + a.val * p.2.val + b.val = r.val ^ 2) ∧ 
                    (p.2.val ^ 2 + a.val * p.1.val + b.val = s.val ^ 2)) : 
  a.val ∣ (2 * b.val) :=
sorry

end a_divides_2b_l377_37783


namespace open_box_volume_formula_l377_37732

/-- The volume of an open box constructed from a rectangular sheet of metal -/
def openBoxVolume (length width x : ℝ) : ℝ :=
  (length - 2*x) * (width - 2*x) * x

theorem open_box_volume_formula :
  ∀ x : ℝ, openBoxVolume 14 10 x = 140*x - 48*x^2 + 4*x^3 :=
by sorry

end open_box_volume_formula_l377_37732


namespace decimal_addition_subtraction_l377_37773

theorem decimal_addition_subtraction : 0.5 + 0.03 - 0.004 + 0.007 = 0.533 := by
  sorry

end decimal_addition_subtraction_l377_37773


namespace parallel_case_perpendicular_case_l377_37733

-- Define the vectors
def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 1)

-- Define parallel condition
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

-- Define perpendicular condition
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- Theorem for parallel case
theorem parallel_case :
  ∃ (x : ℝ), parallel (a + 2 • (b x)) (2 • a - b x) ∧ x = 1/2 := by sorry

-- Theorem for perpendicular case
theorem perpendicular_case :
  ∃ (x : ℝ), perpendicular (a + 2 • (b x)) (2 • a - b x) ∧ (x = -2 ∨ x = 7/2) := by sorry

end parallel_case_perpendicular_case_l377_37733


namespace competition_configs_l377_37771

/-- Represents a valid competition configuration -/
structure CompetitionConfig where
  n : ℕ
  k : ℕ
  h_n_ge_2 : n ≥ 2
  h_k_ge_1 : k ≥ 1
  h_total_score : k * (n * (n + 1) / 2) = 26 * n

/-- The set of all valid competition configurations -/
def ValidConfigs : Set CompetitionConfig := {c | c.n ≥ 2 ∧ c.k ≥ 1 ∧ c.k * (c.n * (c.n + 1) / 2) = 26 * c.n}

/-- The theorem stating the possible values of (n, k) -/
theorem competition_configs : ValidConfigs = {⟨25, 2, by norm_num, by norm_num, by norm_num⟩, 
                                              ⟨12, 4, by norm_num, by norm_num, by norm_num⟩, 
                                              ⟨3, 13, by norm_num, by norm_num, by norm_num⟩} := by
  sorry

end competition_configs_l377_37771


namespace equation_solution_l377_37737

theorem equation_solution : ∀ x y : ℕ, 
  (x - 1) / (1 + (x - 1) * y) + (y - 1) / (2 * y - 1) = x / (x + 1) → 
  x = 2 ∧ y = 2 :=
by
  sorry

end equation_solution_l377_37737


namespace route_choice_and_expected_value_l377_37736

-- Define the data types
structure RouteData where
  good : ℕ
  average : ℕ

structure GenderRouteData where
  male : ℕ
  female : ℕ

-- Define the constants
def total_tourists : ℕ := 300
def route_a : RouteData := { good := 50, average := 75 }
def route_b : RouteData := { good := 75, average := 100 }
def gender_data : GenderRouteData := { male := 120, female := 180 }

-- Define the K^2 formula
def k_squared (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for K^2 at 0.001 significance level
def k_critical : ℚ := 10.828

-- Define the expected value calculation
def expected_value (good_prob : ℚ) : ℚ :=
  let good_score := 5
  let avg_score := 2
  (1 - good_prob)^3 * (3 * avg_score) +
  3 * good_prob * (1 - good_prob)^2 * (2 * avg_score + good_score) +
  3 * good_prob^2 * (1 - good_prob) * (avg_score + 2 * good_score) +
  good_prob^3 * (3 * good_score)

-- Theorem statement
theorem route_choice_and_expected_value :
  let k_value := k_squared gender_data.male (gender_data.female - gender_data.male)
                            (total_tourists - gender_data.male - gender_data.female) gender_data.female
  let prob_a := (route_a.good : ℚ) / (route_a.good + route_a.average)
  let prob_b := (route_b.good : ℚ) / (route_b.good + route_b.average)
  k_value > k_critical ∧ expected_value prob_a > expected_value prob_b := by
  sorry

end route_choice_and_expected_value_l377_37736


namespace no_common_solution_l377_37794

theorem no_common_solution : ¬∃ x : ℝ, (263 - x = 108) ∧ (25 * x = 1950) ∧ (x / 15 = 64) := by
  sorry

end no_common_solution_l377_37794


namespace optimal_garden_max_area_l377_37770

/-- Represents a rectangular garden with given constraints --/
structure Garden where
  length : ℝ
  width : ℝ
  perimeter_constraint : length + width = 200
  length_min : length ≥ 100
  width_min : width ≥ 50
  length_width_diff : length ≥ width + 20

/-- The area of a garden --/
def garden_area (g : Garden) : ℝ := g.length * g.width

/-- The optimal garden dimensions and area --/
def optimal_garden : Garden := {
  length := 120,
  width := 80,
  perimeter_constraint := by sorry,
  length_min := by sorry,
  width_min := by sorry,
  length_width_diff := by sorry
}

/-- Theorem stating that the optimal garden has the maximum area --/
theorem optimal_garden_max_area :
  ∀ g : Garden, garden_area g ≤ garden_area optimal_garden := by sorry

end optimal_garden_max_area_l377_37770


namespace quadratic_solution_property_l377_37715

theorem quadratic_solution_property (x₁ x₂ : ℝ) : 
  x₁^2 - 5*x₁ + 2 = 0 → x₂^2 - 5*x₂ + 2 = 0 → 2*x₁ - x₁*x₂ + 2*x₂ = 8 := by
  sorry

end quadratic_solution_property_l377_37715


namespace binomial_12_choose_10_l377_37738

theorem binomial_12_choose_10 : Nat.choose 12 10 = 66 := by sorry

end binomial_12_choose_10_l377_37738


namespace smallest_prime_perimeter_scalene_triangle_l377_37793

-- Define a scalene triangle with prime side lengths
def ScaleneTriangleWithPrimeSides (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c

-- Define a function to check if the perimeter is prime
def HasPrimePerimeter (a b c : ℕ) : Prop :=
  Nat.Prime (a + b + c)

-- Theorem statement
theorem smallest_prime_perimeter_scalene_triangle :
  ∀ a b c : ℕ,
    ScaleneTriangleWithPrimeSides a b c →
    HasPrimePerimeter a b c →
    a + b + c ≥ 23 :=
by sorry

end smallest_prime_perimeter_scalene_triangle_l377_37793


namespace three_fourths_to_fifth_power_l377_37739

theorem three_fourths_to_fifth_power :
  (3 / 4 : ℚ) ^ 5 = 243 / 1024 := by
  sorry

end three_fourths_to_fifth_power_l377_37739


namespace problem_solution_l377_37731

/-- Given that 4x^5 + 3x^3 - 2x + 1 + g(x) = 7x^3 - 5x^2 + 4x - 3,
    prove that g(x) = -4x^5 + 4x^3 - 5x^2 + 6x - 4 -/
theorem problem_solution (x : ℝ) (g : ℝ → ℝ) 
    (h : ∀ x, 4*x^5 + 3*x^3 - 2*x + 1 + g x = 7*x^3 - 5*x^2 + 4*x - 3) : 
  g x = -4*x^5 + 4*x^3 - 5*x^2 + 6*x - 4 := by
  sorry

end problem_solution_l377_37731


namespace hotel_assignment_count_l377_37746

/-- Represents the number of rooms in the hotel -/
def num_rooms : ℕ := 4

/-- Represents the number of friends arriving -/
def num_friends : ℕ := 6

/-- Represents the maximum number of friends allowed per room -/
def max_per_room : ℕ := 3

/-- Calculates the number of ways to assign friends to rooms -/
def num_assignments : ℕ :=
  -- The actual calculation is not provided here
  1560

/-- Theorem stating that the number of assignments is 1560 -/
theorem hotel_assignment_count :
  num_assignments = 1560 :=
sorry

end hotel_assignment_count_l377_37746
