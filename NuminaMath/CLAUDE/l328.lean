import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequalities_l328_32858

/-- Given a quadratic inequality and its solution set, prove the value of the coefficient and the solution set of a related inequality -/
theorem quadratic_inequalities (a : ℝ) :
  (∀ x : ℝ, (a * x^2 + 3 * x - 1 > 0) ↔ (1/2 < x ∧ x < 1)) →
  (a = -2 ∧ 
   ∀ x : ℝ, (a * x^2 - 3 * x + a^2 + 1 > 0) ↔ (-5/2 < x ∧ x < 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequalities_l328_32858


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l328_32857

-- Define a structure for a rectangular solid
structure RectangularSolid where
  length : ℕ
  width : ℕ
  height : ℕ

-- Define the properties of the rectangular solid
def isPrime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem rectangular_solid_surface_area 
  (solid : RectangularSolid) 
  (prime_edges : isPrime solid.length ∧ isPrime solid.width ∧ isPrime solid.height) 
  (volume_constraint : solid.length * solid.width * solid.height = 105) :
  2 * (solid.length * solid.width + solid.width * solid.height + solid.height * solid.length) = 142 := by
  sorry


end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l328_32857


namespace NUMINAMATH_CALUDE_area_PQR_is_sqrt_35_l328_32839

/-- Represents a square pyramid with given dimensions and points -/
structure SquarePyramid where
  base_side : ℝ
  altitude : ℝ
  p_ratio : ℝ
  q_ratio : ℝ
  r_ratio : ℝ

/-- Calculates the area of triangle PQR in the square pyramid -/
def area_PQR (pyramid : SquarePyramid) : ℝ :=
  sorry

/-- Theorem stating that the area of triangle PQR is √35 for the given pyramid -/
theorem area_PQR_is_sqrt_35 :
  let pyramid := SquarePyramid.mk 4 8 (1/4) (1/4) (3/4)
  area_PQR pyramid = Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_area_PQR_is_sqrt_35_l328_32839


namespace NUMINAMATH_CALUDE_inscribed_semicircle_radius_l328_32806

/-- An isosceles triangle with a semicircle inscribed in its base -/
structure IsoscelesTriangleWithSemicircle where
  /-- The base of the isosceles triangle -/
  base : ℝ
  /-- The height of the isosceles triangle -/
  height : ℝ
  /-- The radius of the inscribed semicircle -/
  radius : ℝ
  /-- The base is positive -/
  base_pos : 0 < base
  /-- The height is positive -/
  height_pos : 0 < height
  /-- The radius is positive -/
  radius_pos : 0 < radius
  /-- The diameter of the semicircle is equal to the base of the triangle -/
  diameter_eq_base : 2 * radius = base

/-- The radius of the inscribed semicircle in the given isosceles triangle -/
theorem inscribed_semicircle_radius 
    (t : IsoscelesTriangleWithSemicircle) 
    (h1 : t.base = 20) 
    (h2 : t.height = 21) : 
    t.radius = 210 / Real.sqrt 541 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_semicircle_radius_l328_32806


namespace NUMINAMATH_CALUDE_quadratic_exponent_condition_l328_32805

theorem quadratic_exponent_condition (a : ℝ) : 
  (∀ x, ∃ p q r : ℝ, x^(a^2 - 7) - 3*x - 2 = p*x^2 + q*x + r) → 
  (a = 3 ∨ a = -3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_exponent_condition_l328_32805


namespace NUMINAMATH_CALUDE_train_speed_problem_l328_32890

theorem train_speed_problem (x : ℝ) (h : x > 0) :
  let total_distance := 3 * x
  let first_distance := x
  let second_distance := 2 * x
  let second_speed := 20
  let average_speed := 26
  let time_first := first_distance / V
  let time_second := second_distance / second_speed
  let total_time := time_first + time_second
  average_speed = total_distance / total_time →
  V = 65 := by
sorry

end NUMINAMATH_CALUDE_train_speed_problem_l328_32890


namespace NUMINAMATH_CALUDE_ground_beef_cost_l328_32814

/-- The cost of ground beef in dollars per kilogram -/
def price_per_kg : ℝ := 5

/-- The quantity of ground beef in kilograms -/
def quantity : ℝ := 12

/-- The total cost of ground beef -/
def total_cost : ℝ := price_per_kg * quantity

theorem ground_beef_cost : total_cost = 60 := by
  sorry

end NUMINAMATH_CALUDE_ground_beef_cost_l328_32814


namespace NUMINAMATH_CALUDE_partnership_share_theorem_l328_32881

/-- Represents a partner in the partnership --/
structure Partner where
  name : String
  investment : ℕ

/-- Represents the partnership --/
structure Partnership where
  partners : List Partner
  duration : ℕ  -- in months

/-- Calculates the share of a partner based on the total profit and investments --/
def calculateShare (partnership : Partnership) (totalProfit : ℕ) (partner : Partner) : ℕ :=
  let totalInvestment := partnership.partners.map Partner.investment |>.sum
  (partner.investment * totalProfit) / totalInvestment

theorem partnership_share_theorem (a b c : Partner) (partnership : Partnership) 
    (h1 : a.investment = 15000)
    (h2 : b.investment = 21000)
    (h3 : c.investment = 27000)
    (h4 : partnership.partners = [a, b, c])
    (h5 : partnership.duration = 8)
    (h6 : calculateShare partnership 4620 b = 1540) :
  calculateShare partnership 4620 a = 1100 := by
  sorry

#check partnership_share_theorem

end NUMINAMATH_CALUDE_partnership_share_theorem_l328_32881


namespace NUMINAMATH_CALUDE_carnival_ticket_cost_l328_32800

/-- The cost of carnival tickets -/
theorem carnival_ticket_cost :
  ∀ (cost_12 : ℚ) (cost_4 : ℚ),
  cost_12 = 3 →
  12 * cost_4 = cost_12 →
  cost_4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_carnival_ticket_cost_l328_32800


namespace NUMINAMATH_CALUDE_friend_team_assignments_l328_32873

theorem friend_team_assignments (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 4) :
  k ^ n = 65536 := by
  sorry

end NUMINAMATH_CALUDE_friend_team_assignments_l328_32873


namespace NUMINAMATH_CALUDE_quadratic_equation_from_root_properties_l328_32872

theorem quadratic_equation_from_root_properties (a b c : ℝ) :
  (∀ x y : ℝ, x + y = 10 ∧ x * y = 24 → a * x^2 + b * x + c = 0) →
  a = 1 ∧ b = -10 ∧ c = 24 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_root_properties_l328_32872


namespace NUMINAMATH_CALUDE_jim_distance_in_24_steps_l328_32876

-- Define the number of steps for Carly and Jim to cover the same distance
def carly_steps : ℕ := 3
def jim_steps : ℕ := 4

-- Define the length of Carly's step in meters
def carly_step_length : ℚ := 1/2

-- Define the number of Jim's steps we're interested in
def jim_target_steps : ℕ := 24

-- Theorem to prove
theorem jim_distance_in_24_steps :
  (jim_target_steps : ℚ) * (carly_steps * carly_step_length) / jim_steps = 9 := by
  sorry

end NUMINAMATH_CALUDE_jim_distance_in_24_steps_l328_32876


namespace NUMINAMATH_CALUDE_unique_solution_is_one_l328_32867

theorem unique_solution_is_one (n : ℕ) (hn : n ≥ 1) :
  (∃ (a b : ℕ), 
    (∀ (p : ℕ), Prime p → ¬(p^3 ∣ (a^2 + b + 3))) ∧
    ((a * b + 3 * b + 8) : ℚ) / (a^2 + b + 3 : ℚ) = n) 
  ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_is_one_l328_32867


namespace NUMINAMATH_CALUDE_slope_of_tan_45_degrees_line_l328_32897

theorem slope_of_tan_45_degrees_line (x : ℝ) :
  let f : ℝ → ℝ := λ x => Real.tan (45 * π / 180)
  (deriv f) x = 0 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_tan_45_degrees_line_l328_32897


namespace NUMINAMATH_CALUDE_number_puzzle_l328_32826

theorem number_puzzle (x : ℝ) : 2 * x = 18 → x - 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l328_32826


namespace NUMINAMATH_CALUDE_janabel_widget_sales_l328_32828

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem janabel_widget_sales : arithmetic_sequence_sum 2 3 15 = 345 := by
  sorry

end NUMINAMATH_CALUDE_janabel_widget_sales_l328_32828


namespace NUMINAMATH_CALUDE_rhombus_area_l328_32871

-- Define the rhombus
def Rhombus (perimeter : ℝ) (diagonal1 : ℝ) : Prop :=
  perimeter > 0 ∧ diagonal1 > 0

-- Theorem statement
theorem rhombus_area 
  (perimeter : ℝ) 
  (diagonal1 : ℝ) 
  (h : Rhombus perimeter diagonal1) 
  (h_perimeter : perimeter = 80) 
  (h_diagonal : diagonal1 = 36) : 
  ∃ (area : ℝ), area = 72 * Real.sqrt 19 :=
sorry

end NUMINAMATH_CALUDE_rhombus_area_l328_32871


namespace NUMINAMATH_CALUDE_surface_area_of_specific_solid_l328_32856

/-- A right prism with equilateral triangle bases -/
structure RightPrism where
  height : ℝ
  base_side : ℝ

/-- Midpoint of an edge -/
structure Midpoint where
  edge : String

/-- The solid formed by slicing off the top part of the prism -/
structure SlicedSolid where
  prism : RightPrism
  x : Midpoint
  y : Midpoint
  z : Midpoint

/-- Calculate the surface area of the sliced solid -/
noncomputable def surface_area (solid : SlicedSolid) : ℝ :=
  sorry

/-- Theorem stating the surface area of the specific sliced solid -/
theorem surface_area_of_specific_solid :
  let prism := RightPrism.mk 20 10
  let x := Midpoint.mk "AC"
  let y := Midpoint.mk "BC"
  let z := Midpoint.mk "DF"
  let solid := SlicedSolid.mk prism x y z
  surface_area solid = 100 + (25 * Real.sqrt 3) / 4 + (5 * Real.sqrt 418.75) / 2 :=
sorry

end NUMINAMATH_CALUDE_surface_area_of_specific_solid_l328_32856


namespace NUMINAMATH_CALUDE_sun_energy_china_equivalence_l328_32829

/-- The energy received from the sun in one year on 1 square kilometer of land,
    measured in kilograms of coal equivalent -/
def energy_per_sq_km : ℝ := 1.3 * 10^8

/-- The approximate land area of China in square kilometers -/
def china_area : ℝ := 9.6 * 10^6

/-- The total energy received from the sun on China's land area,
    measured in kilograms of coal equivalent -/
def total_energy : ℝ := energy_per_sq_km * china_area

theorem sun_energy_china_equivalence :
  total_energy = 1.248 * 10^15 := by
  sorry

end NUMINAMATH_CALUDE_sun_energy_china_equivalence_l328_32829


namespace NUMINAMATH_CALUDE_arithmetic_sequence_average_l328_32818

/-- The average value of an arithmetic sequence with 5 terms, starting at 0 and with a common difference of 3x, is 6x. -/
theorem arithmetic_sequence_average (x : ℝ) : 
  let sequence := [0, 3*x, 6*x, 9*x, 12*x]
  (sequence.sum / sequence.length : ℝ) = 6*x := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_average_l328_32818


namespace NUMINAMATH_CALUDE_largest_angle_of_special_hexagon_l328_32803

-- Define a hexagon type
structure Hexagon where
  angles : Fin 6 → ℝ
  is_convex : True
  consecutive_integers : ∀ i : Fin 5, ∃ n : ℤ, angles i.succ = angles i + 1
  sum_720 : (Finset.univ.sum angles) = 720

-- Theorem statement
theorem largest_angle_of_special_hexagon (h : Hexagon) :
  Finset.max' (Finset.univ.image h.angles) (by sorry) = 122.5 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_of_special_hexagon_l328_32803


namespace NUMINAMATH_CALUDE_count_valid_n_l328_32854

def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, y * y = x

def valid_n (n : ℕ) : Prop :=
  n > 0 ∧
  is_perfect_square (1 * 4 + 2112) ∧
  is_perfect_square (1 * n + 2112) ∧
  is_perfect_square (4 * n + 2112)

theorem count_valid_n :
  ∃ (S : Finset ℕ), (∀ n ∈ S, valid_n n) ∧ S.card = 7 ∧ (∀ n, valid_n n → n ∈ S) :=
sorry

end NUMINAMATH_CALUDE_count_valid_n_l328_32854


namespace NUMINAMATH_CALUDE_train_length_proof_l328_32804

/-- Proves that the length of each train is 50 meters given the specified conditions -/
theorem train_length_proof (v_fast v_slow : ℝ) (t : ℝ) (h1 : v_fast = 46) (h2 : v_slow = 36) (h3 : t = 36) :
  let v_rel := (v_fast - v_slow) * (5 / 18)  -- Convert km/hr to m/s
  let l := v_rel * t / 2                     -- Length of one train
  l = 50 := by sorry

end NUMINAMATH_CALUDE_train_length_proof_l328_32804


namespace NUMINAMATH_CALUDE_calculator_exam_duration_l328_32860

theorem calculator_exam_duration 
  (full_battery : ℝ) 
  (remaining_battery : ℝ) 
  (exam_duration : ℝ) :
  full_battery = 60 →
  remaining_battery = 13 →
  exam_duration = (1/4 * full_battery) - remaining_battery →
  exam_duration = 2 :=
by sorry

end NUMINAMATH_CALUDE_calculator_exam_duration_l328_32860


namespace NUMINAMATH_CALUDE_number_of_people_liking_apple_l328_32827

/-- The number of people who like apple -/
def like_apple : ℕ := 40

/-- The number of people who like orange and mango but dislike apple -/
def like_orange_mango_not_apple : ℕ := 7

/-- The number of people who like mango and apple but dislike orange -/
def like_mango_apple_not_orange : ℕ := 10

/-- The number of people who like all three fruits -/
def like_all : ℕ := 4

/-- Theorem stating that the number of people who like apple is 40 -/
theorem number_of_people_liking_apple : 
  like_apple = 40 := by sorry

end NUMINAMATH_CALUDE_number_of_people_liking_apple_l328_32827


namespace NUMINAMATH_CALUDE_factorization_equality_l328_32855

theorem factorization_equality (x : ℝ) : 
  (x - 3) * (x - 1) * (x - 2) * (x + 4) + 24 = (x - 2) * (x + 3) * (x^2 + x - 8) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l328_32855


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l328_32831

-- Problem 1
theorem problem_1 (m : ℝ) : 
  let A : Set ℝ := {x | x^2 + 3*x + 2 = 0}
  let B : Set ℝ := {x | x^2 + (m+1)*x + m = 0}
  A ∩ B = B → m = 1 ∨ m = 2 := by sorry

-- Problem 2
theorem problem_2 (n : ℝ) :
  let A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
  let B : Set ℝ := {x | n+1 ≤ x ∧ x ≤ 2*n-1}
  B ⊆ A → n ≤ 3 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l328_32831


namespace NUMINAMATH_CALUDE_tenth_term_is_neg_512_l328_32875

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℤ  -- The sequence (a₁, a₂, a₃, ...)
  is_geometric : ∀ n : ℕ, n ≥ 2 → a (n + 1) / a n = a 2 / a 1
  product_25 : a 2 * a 5 = -32
  sum_34 : a 3 + a 4 = 4
  integer_ratio : ∃ q : ℤ, ∀ n : ℕ, n ≥ 1 → a (n + 1) = q * a n

/-- The 10th term of the geometric sequence is -512 -/
theorem tenth_term_is_neg_512 (seq : GeometricSequence) : seq.a 10 = -512 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_is_neg_512_l328_32875


namespace NUMINAMATH_CALUDE_remaining_debt_l328_32849

def total_debt : ℕ := 50
def payment_two_months_ago : ℕ := 12
def payment_last_month : ℕ := payment_two_months_ago + 3

theorem remaining_debt :
  total_debt - (payment_two_months_ago + payment_last_month) = 23 :=
by sorry

end NUMINAMATH_CALUDE_remaining_debt_l328_32849


namespace NUMINAMATH_CALUDE_pen_notebook_cost_l328_32869

theorem pen_notebook_cost : ∃ (p n : ℕ), 
  p > 0 ∧ n > 0 ∧ 
  15 * p + 5 * n = 13000 ∧ 
  p > n ∧ 
  p + n = 10 :=
by sorry

end NUMINAMATH_CALUDE_pen_notebook_cost_l328_32869


namespace NUMINAMATH_CALUDE_tangent_slope_at_point_l328_32892

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 2

theorem tangent_slope_at_point :
  let x₀ : ℝ := 1
  let y₀ : ℝ := -5/3
  let slope : ℝ := deriv f x₀
  (f x₀ = y₀) ∧ (slope = 1) := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_point_l328_32892


namespace NUMINAMATH_CALUDE_total_bird_wings_l328_32883

/-- The number of birds in the sky -/
def num_birds : ℕ := 13

/-- The number of wings each bird has -/
def wings_per_bird : ℕ := 2

/-- Theorem: The total number of bird wings in the sky is 26 -/
theorem total_bird_wings : num_birds * wings_per_bird = 26 := by
  sorry

end NUMINAMATH_CALUDE_total_bird_wings_l328_32883


namespace NUMINAMATH_CALUDE_circle_equations_from_line_intersections_l328_32816

-- Define the line
def line (x y : ℝ) : Prop := 2 * x + y - 4 = 0

-- Define the two intersection points
def point_A : ℝ × ℝ := (0, 4)
def point_B : ℝ × ℝ := (2, 0)

-- Define the two circle equations
def circle1 (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 20
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 20

-- Theorem statement
theorem circle_equations_from_line_intersections :
  (∀ x y : ℝ, line x y → (x = 0 ∧ y = 4) ∨ (x = 2 ∧ y = 0)) ∧
  (circle1 (point_A.1) (point_A.2) ∧ circle1 (point_B.1) (point_B.2)) ∧
  (circle2 (point_A.1) (point_A.2) ∧ circle2 (point_B.1) (point_B.2)) :=
sorry

end NUMINAMATH_CALUDE_circle_equations_from_line_intersections_l328_32816


namespace NUMINAMATH_CALUDE_teddy_bears_per_shelf_l328_32817

theorem teddy_bears_per_shelf (total_bears : ℕ) (num_shelves : ℕ) 
  (h1 : total_bears = 98) (h2 : num_shelves = 14) :
  (total_bears / num_shelves : ℕ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_teddy_bears_per_shelf_l328_32817


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l328_32874

theorem largest_prime_factor_of_1729 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 1729 → q ≤ p :=
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l328_32874


namespace NUMINAMATH_CALUDE_sqrt_6_simplest_l328_32886

-- Define the criteria for simplest square root form
def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y : ℚ, x ≠ y^2 ∧ (∀ z : ℕ, z > 1 → ¬ (∃ w : ℕ, x = z * w^2))

-- Define the set of square roots to compare
def sqrt_set : Set ℝ := {Real.sqrt 0.2, Real.sqrt (1/2), Real.sqrt 6, Real.sqrt 12}

-- Theorem statement
theorem sqrt_6_simplest :
  ∀ x ∈ sqrt_set, x ≠ Real.sqrt 6 → ¬(is_simplest_sqrt x) ∧ is_simplest_sqrt (Real.sqrt 6) :=
sorry

end NUMINAMATH_CALUDE_sqrt_6_simplest_l328_32886


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l328_32859

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 7 + a 13 = 20 →
  a 9 + a 10 + a 11 = 30 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l328_32859


namespace NUMINAMATH_CALUDE_chord_length_is_twenty_l328_32840

/-- A circle with a chord that is the perpendicular bisector of its diameter -/
structure CircleWithChord where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The chord is a perpendicular bisector of a diameter -/
  chord_is_perp_bisector : Bool

/-- Theorem: The length of a chord that is the perpendicular bisector of a diameter 
    in a circle with radius 10 is 20 -/
theorem chord_length_is_twenty (c : CircleWithChord) 
  (h1 : c.radius = 10) 
  (h2 : c.chord_is_perp_bisector = true) : 
  ∃ (chord_length : ℝ), chord_length = 20 := by
  sorry


end NUMINAMATH_CALUDE_chord_length_is_twenty_l328_32840


namespace NUMINAMATH_CALUDE_factorial_combination_l328_32863

theorem factorial_combination : Nat.factorial 10 / (Nat.factorial 7 * Nat.factorial 3) = 120 := by
  sorry

end NUMINAMATH_CALUDE_factorial_combination_l328_32863


namespace NUMINAMATH_CALUDE_marbles_remaining_l328_32894

theorem marbles_remaining (initial : ℝ) (lost : ℝ) (given_away : ℝ) (found : ℝ) : 
  initial = 150 → 
  lost = 58.5 → 
  given_away = 37.2 → 
  found = 10.8 → 
  initial - lost - given_away + found = 65.1 := by
sorry

end NUMINAMATH_CALUDE_marbles_remaining_l328_32894


namespace NUMINAMATH_CALUDE_chord_length_l328_32822

theorem chord_length (r : ℝ) (h : r = 10) : 
  let chord_length := 2 * (r^2 - (r/2)^2).sqrt
  chord_length = 10 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l328_32822


namespace NUMINAMATH_CALUDE_classroom_arrangements_l328_32861

theorem classroom_arrangements (n : Nat) (h : n = 6) : 
  (Finset.range (n + 1)).sum (fun k => Nat.choose n k) - Nat.choose n 1 - Nat.choose n 0 = 57 := by
  sorry

end NUMINAMATH_CALUDE_classroom_arrangements_l328_32861


namespace NUMINAMATH_CALUDE_infinite_special_integers_l328_32824

theorem infinite_special_integers : 
  ∃ f : ℕ → ℕ, Infinite {n : ℕ | ∃ m : ℕ, 
    n = m * (m + 1) + 2 ∧ 
    ∀ p : ℕ, Prime p → p ∣ (n^2 + 3) → 
      ∃ k : ℕ, k^2 < n ∧ p ∣ (k^2 + 3)} :=
sorry

end NUMINAMATH_CALUDE_infinite_special_integers_l328_32824


namespace NUMINAMATH_CALUDE_total_students_l328_32895

theorem total_students (girls : ℕ) (boys : ℕ) (total : ℕ) : 
  girls = 160 →
  5 * boys = 8 * girls →
  total = girls + boys →
  total = 416 := by
sorry

end NUMINAMATH_CALUDE_total_students_l328_32895


namespace NUMINAMATH_CALUDE_blue_marble_difference_l328_32888

theorem blue_marble_difference (total_green : ℕ) 
  (ratio_a_blue ratio_a_green ratio_b_blue ratio_b_green : ℕ) : 
  total_green = 162 →
  ratio_a_blue = 5 →
  ratio_a_green = 3 →
  ratio_b_blue = 4 →
  ratio_b_green = 1 →
  ∃ (a b : ℕ), 
    ratio_a_green * a + ratio_b_green * b = total_green ∧
    (ratio_a_blue + ratio_a_green) * a = (ratio_b_blue + ratio_b_green) * b ∧
    ratio_b_blue * b - ratio_a_blue * a = 49 :=
by
  sorry

#check blue_marble_difference

end NUMINAMATH_CALUDE_blue_marble_difference_l328_32888


namespace NUMINAMATH_CALUDE_smallest_n_for_probability_threshold_l328_32893

theorem smallest_n_for_probability_threshold (n : ℕ) : 
  (∀ k, k < n → 1 / (k * (k + 1)) ≥ 1 / 2010) ∧
  1 / (n * (n + 1)) < 1 / 2010 →
  n = 45 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_probability_threshold_l328_32893


namespace NUMINAMATH_CALUDE_worker_ant_ratio_l328_32865

theorem worker_ant_ratio (total_ants : ℕ) (female_worker_ants : ℕ) 
  (h1 : total_ants = 110)
  (h2 : female_worker_ants = 44)
  (h3 : (female_worker_ants : ℚ) / (female_worker_ants / 0.8 : ℚ) = 0.8) :
  (female_worker_ants / 0.8 : ℚ) / (total_ants : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_worker_ant_ratio_l328_32865


namespace NUMINAMATH_CALUDE_distance_AB_DB1_is_12_div_5_l328_32877

/-- A rectangular prism with given dimensions -/
structure RectangularPrism where
  AB : ℝ
  BC : ℝ
  BB1 : ℝ

/-- The distance between AB and DB₁ in a rectangular prism -/
def distance_AB_DB1 (prism : RectangularPrism) : ℝ := sorry

theorem distance_AB_DB1_is_12_div_5 (prism : RectangularPrism) 
  (h1 : prism.AB = 5)
  (h2 : prism.BC = 4)
  (h3 : prism.BB1 = 3) :
  distance_AB_DB1 prism = 12 / 5 := by sorry

end NUMINAMATH_CALUDE_distance_AB_DB1_is_12_div_5_l328_32877


namespace NUMINAMATH_CALUDE_modified_sequence_last_term_l328_32899

def sequence_rule (n : ℕ) : ℕ → ℕ
  | 0 => 1
  | i + 1 => 
    let prev := sequence_rule n i
    if prev < 10 then
      2 * prev
    else
      (prev % 10) + 5

def modified_sequence (n : ℕ) (m : ℕ) : ℕ → ℕ
  | i => if i = 99 then sequence_rule n i + m else sequence_rule n i

theorem modified_sequence_last_term (n : ℕ) :
  ∃ m : ℕ, m < 10 ∧ modified_sequence 2012 m 2011 = 5 → m = 8 := by
  sorry

end NUMINAMATH_CALUDE_modified_sequence_last_term_l328_32899


namespace NUMINAMATH_CALUDE_no_rectangular_prism_exists_l328_32846

theorem no_rectangular_prism_exists : ¬∃ (a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a + b + c = 12 ∧ 
  a * b + b * c + c * a = 1 ∧ 
  a * b * c = 12 := by
  sorry

end NUMINAMATH_CALUDE_no_rectangular_prism_exists_l328_32846


namespace NUMINAMATH_CALUDE_george_marbles_count_l328_32862

/-- The total number of marbles George collected -/
def total_marbles : ℕ := 50

/-- The number of yellow marbles -/
def yellow_marbles : ℕ := 12

/-- The number of red marbles -/
def red_marbles : ℕ := 7

/-- The number of green marbles -/
def green_marbles : ℕ := yellow_marbles / 2

/-- The number of white marbles -/
def white_marbles : ℕ := total_marbles / 2

theorem george_marbles_count :
  total_marbles = white_marbles + yellow_marbles + green_marbles + red_marbles :=
by sorry

end NUMINAMATH_CALUDE_george_marbles_count_l328_32862


namespace NUMINAMATH_CALUDE_sum_of_even_numbers_1_to_200_l328_32882

theorem sum_of_even_numbers_1_to_200 : 
  (Finset.filter (fun n => Even n) (Finset.range 201)).sum id = 10100 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_even_numbers_1_to_200_l328_32882


namespace NUMINAMATH_CALUDE_f_of_f_2_l328_32866

def f (x : ℝ) : ℝ := 2 * x^3 - 4 * x^2 + 3 * x - 1

theorem f_of_f_2 : f (f 2) = 164 := by sorry

end NUMINAMATH_CALUDE_f_of_f_2_l328_32866


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l328_32835

theorem quadratic_root_implies_k (k : ℝ) : 
  ((k - 3) * (-1)^2 + 6 * (-1) + k^2 - k = 0) → k = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l328_32835


namespace NUMINAMATH_CALUDE_jane_payment_l328_32853

/-- The amount Jane paid with, given the cost of the apple and the change received. -/
def amount_paid (apple_cost change : ℚ) : ℚ :=
  apple_cost + change

/-- Theorem stating that Jane paid with $5.00, given the conditions of the problem. -/
theorem jane_payment :
  let apple_cost : ℚ := 75 / 100
  let change : ℚ := 425 / 100
  amount_paid apple_cost change = 5 := by
  sorry

end NUMINAMATH_CALUDE_jane_payment_l328_32853


namespace NUMINAMATH_CALUDE_solve_equation_l328_32852

theorem solve_equation : 
  let y : ℚ := 45 / (8 - 3/7)
  y = 315 / 53 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l328_32852


namespace NUMINAMATH_CALUDE_equation_solution_l328_32848

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), (x₁ = 5 ∧ x₂ = -1) ∧ 
  (∀ x : ℝ, (x - 2)^2 = 9 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l328_32848


namespace NUMINAMATH_CALUDE_f_neg_one_eq_neg_one_l328_32830

/-- Given a function f(x) = -2x^2 + 1, prove that f(-1) = -1 -/
theorem f_neg_one_eq_neg_one :
  let f : ℝ → ℝ := fun x ↦ -2 * x^2 + 1
  f (-1) = -1 := by sorry

end NUMINAMATH_CALUDE_f_neg_one_eq_neg_one_l328_32830


namespace NUMINAMATH_CALUDE_angle_between_lines_l328_32864

theorem angle_between_lines (k₁ k₂ : ℝ) (h₁ : 6 * k₁^2 + k₁ - 1 = 0) (h₂ : 6 * k₂^2 + k₂ - 1 = 0) :
  let θ := Real.arctan ((k₁ - k₂) / (1 + k₁ * k₂))
  θ = π / 4 ∨ θ = -π / 4 :=
sorry

end NUMINAMATH_CALUDE_angle_between_lines_l328_32864


namespace NUMINAMATH_CALUDE_min_value_quadratic_expression_l328_32842

theorem min_value_quadratic_expression (x y : ℝ) :
  x^2 + y^2 - 4*x + 6*y + 20 ≥ 7 := by
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_expression_l328_32842


namespace NUMINAMATH_CALUDE_simplify_fraction_l328_32813

theorem simplify_fraction : 4 * (14 / 5) * (20 / -42) = -(4 / 15) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l328_32813


namespace NUMINAMATH_CALUDE_correlation_coefficient_measures_linear_relationship_l328_32833

/-- The correlation coefficient is a statistical measure. -/
def correlation_coefficient : Type := sorry

/-- A measure of the strength of a linear relationship between two variables. -/
def linear_relationship_strength : Type := sorry

/-- The correlation coefficient measures the strength of the linear relationship between two variables. -/
theorem correlation_coefficient_measures_linear_relationship :
  correlation_coefficient → linear_relationship_strength :=
sorry

end NUMINAMATH_CALUDE_correlation_coefficient_measures_linear_relationship_l328_32833


namespace NUMINAMATH_CALUDE_igor_travel_time_l328_32832

/-- Represents the ski lift system with its properties and functions -/
structure SkiLift where
  total_cabins : Nat
  igor_cabin : Nat
  first_alignment : Nat
  second_alignment : Nat
  alignment_time : Nat

/-- Calculates the time for Igor to reach the top of the mountain -/
def time_to_top (lift : SkiLift) : Nat :=
  let total_distance := lift.total_cabins - lift.igor_cabin + lift.second_alignment
  let speed := (lift.first_alignment - lift.second_alignment) / lift.alignment_time
  (total_distance / 2) * (1 / speed)

/-- Theorem stating that Igor will reach the top in 1035 seconds -/
theorem igor_travel_time (lift : SkiLift) 
  (h1 : lift.total_cabins = 99)
  (h2 : lift.igor_cabin = 42)
  (h3 : lift.first_alignment = 13)
  (h4 : lift.second_alignment = 12)
  (h5 : lift.alignment_time = 15) :
  time_to_top lift = 1035 := by
  sorry

end NUMINAMATH_CALUDE_igor_travel_time_l328_32832


namespace NUMINAMATH_CALUDE_two_digit_number_puzzle_l328_32838

theorem two_digit_number_puzzle : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 1000 + 100 * (n / 10) + 10 * (n % 10) + 1 = 23 * n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_two_digit_number_puzzle_l328_32838


namespace NUMINAMATH_CALUDE_six_digit_number_exists_l328_32851

/-- A six-digit number is between 100000 and 999999 -/
def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

/-- A five-digit number is between 10000 and 99999 -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

/-- The result of removing one digit from a six-digit number -/
def remove_digit (n : ℕ) : ℕ := n / 10

theorem six_digit_number_exists : 
  ∃! n : ℕ, is_six_digit n ∧ 
    ∃ m : ℕ, is_five_digit m ∧ 
      m = remove_digit n ∧ 
      n - m = 654321 :=
sorry

end NUMINAMATH_CALUDE_six_digit_number_exists_l328_32851


namespace NUMINAMATH_CALUDE_min_value_sum_l328_32801

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 27) :
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x * y * z = 27 → a + 3 * b + 9 * c ≤ x + 3 * y + 9 * z :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_l328_32801


namespace NUMINAMATH_CALUDE_one_true_proposition_l328_32887

theorem one_true_proposition :
  (∃! i : Fin 4, 
    (i = 0 ∧ (∀ x y : ℝ, ¬(x = -y) → x + y ≠ 0)) ∨
    (i = 1 ∧ (∀ a b : ℝ, a^2 > b^2 → a > b)) ∨
    (i = 2 ∧ (∃ x : ℝ, x ≤ -3 ∧ x^2 - x - 6 ≤ 0)) ∨
    (i = 3 ∧ (∀ a b : ℝ, Irrational a ∧ Irrational b → Irrational (a^b)))) :=
sorry

end NUMINAMATH_CALUDE_one_true_proposition_l328_32887


namespace NUMINAMATH_CALUDE_tangent_circle_circumference_l328_32844

-- Define the geometric configuration
structure GeometricConfig where
  -- Centers of the arcs
  A : Point
  B : Point
  -- Points on the arcs
  C : Point
  -- Radii of the arcs
  r1 : ℝ
  r2 : ℝ
  -- Angle subtended by arc AC at center B
  angle_ACB : ℝ
  -- Length of arc BC
  length_BC : ℝ
  -- Radius of the tangent circle
  r : ℝ

-- State the theorem
theorem tangent_circle_circumference (config : GeometricConfig) 
  (h1 : config.angle_ACB = 75 * π / 180)
  (h2 : config.length_BC = 18)
  (h3 : config.r1 = 54 / π)
  (h4 : config.r2 = 216 / (5 * π))
  (h5 : config.r = 30 / π) : 
  2 * π * config.r = 60 := by
  sorry


end NUMINAMATH_CALUDE_tangent_circle_circumference_l328_32844


namespace NUMINAMATH_CALUDE_container_capacity_prove_container_capacity_l328_32870

theorem container_capacity : ℝ → Prop :=
  fun capacity =>
    (0.5 * capacity + 20 = 0.75 * capacity) →
    capacity = 80

-- The proof of the theorem
theorem prove_container_capacity : container_capacity 80 := by
  sorry

end NUMINAMATH_CALUDE_container_capacity_prove_container_capacity_l328_32870


namespace NUMINAMATH_CALUDE_power_of_ten_square_l328_32836

theorem power_of_ten_square (k : ℕ) (N : ℕ) : 
  (10^(k-1) ≤ N) ∧ (N < 10^k) ∧ 
  (∃ m : ℕ, N^2 = N * 10^k + m ∧ m < N * 10^k) → 
  N = 10^(k-1) :=
by sorry

end NUMINAMATH_CALUDE_power_of_ten_square_l328_32836


namespace NUMINAMATH_CALUDE_store_display_arrangement_l328_32847

def stripe_arrangement (n : ℕ) : ℕ := 
  match n with
  | 0 => 0
  | 1 => 2
  | 2 => 2
  | (n + 3) => stripe_arrangement (n + 1) + stripe_arrangement (n + 2)

theorem store_display_arrangement : 
  stripe_arrangement 10 = 110 := by sorry

end NUMINAMATH_CALUDE_store_display_arrangement_l328_32847


namespace NUMINAMATH_CALUDE_probability_white_ball_l328_32807

/-- The probability of drawing a white ball from a bag containing 2 red balls and 1 white ball is 1/3. -/
theorem probability_white_ball (red_balls white_balls total_balls : ℕ) : 
  red_balls = 2 → white_balls = 1 → total_balls = red_balls + white_balls →
  (white_balls : ℚ) / total_balls = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_white_ball_l328_32807


namespace NUMINAMATH_CALUDE_sum_of_fractions_l328_32896

theorem sum_of_fractions : 
  (3 / 15 : ℚ) + (6 / 15 : ℚ) + (9 / 15 : ℚ) + (12 / 15 : ℚ) + (1 : ℚ) + 
  (18 / 15 : ℚ) + (21 / 15 : ℚ) + (24 / 15 : ℚ) + (27 / 15 : ℚ) + (5 : ℚ) = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l328_32896


namespace NUMINAMATH_CALUDE_cafeteria_pies_l328_32809

theorem cafeteria_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) : 
  initial_apples = 62 → 
  handed_out = 8 → 
  apples_per_pie = 9 → 
  (initial_apples - handed_out) / apples_per_pie = 6 := by
sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l328_32809


namespace NUMINAMATH_CALUDE_find_M_and_N_convex_polygon_diagonals_calculate_y_l328_32845

-- Part 1 and 2
theorem find_M_and_N :
  ∃ (M N : ℕ),
    M < 10 ∧ N < 10 ∧
    258024 * 10 + M * 10 + 8 * 9 = 2111110 * N * 11 ∧
    M = 9 ∧ N = 2 := by sorry

-- Part 3
theorem convex_polygon_diagonals (n : ℕ) (h : n = 20) :
  (n * (n - 3)) / 2 = 170 := by sorry

-- Part 4
theorem calculate_y (a b : ℕ) (h1 : a = 99) (h2 : b = 49) :
  a * b + a + b + 1 = 4999 := by sorry

end NUMINAMATH_CALUDE_find_M_and_N_convex_polygon_diagonals_calculate_y_l328_32845


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_for_minimum_2_l328_32815

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Theorem for the first part of the problem
theorem solution_set_when_a_is_3 :
  {x : ℝ | f 3 x ≥ 4} = {x : ℝ | x ≤ 0 ∨ x ≥ 4} := by sorry

-- Theorem for the second part of the problem
theorem range_of_a_for_minimum_2 :
  {a : ℝ | ∀ x₁ : ℝ, f a x₁ ≥ 2} = {a : ℝ | a ≥ 3 ∨ a ≤ -1} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_for_minimum_2_l328_32815


namespace NUMINAMATH_CALUDE_sum_reciprocals_bound_l328_32823

theorem sum_reciprocals_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  1/a + 1/b ≥ 4 ∧ ∀ M : ℝ, ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 ∧ 1/a + 1/b > M :=
by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_bound_l328_32823


namespace NUMINAMATH_CALUDE_shopping_mall_sales_l328_32843

/-- Shopping mall sales problem -/
theorem shopping_mall_sales
  (initial_cost : ℝ)
  (initial_price : ℝ)
  (january_sales : ℝ)
  (march_sales : ℝ)
  (price_decrease : ℝ)
  (sales_increase : ℝ)
  (desired_profit : ℝ)
  (h1 : initial_cost = 60)
  (h2 : initial_price = 80)
  (h3 : january_sales = 64)
  (h4 : march_sales = 100)
  (h5 : price_decrease = 0.5)
  (h6 : sales_increase = 5)
  (h7 : desired_profit = 2160) :
  ∃ (growth_rate : ℝ) (optimal_price : ℝ),
    growth_rate = 0.25 ∧
    optimal_price = 72 ∧
    (1 + growth_rate)^2 * january_sales = march_sales ∧
    (optimal_price - initial_cost) * (march_sales + (sales_increase / price_decrease) * (initial_price - optimal_price)) = desired_profit :=
by sorry

end NUMINAMATH_CALUDE_shopping_mall_sales_l328_32843


namespace NUMINAMATH_CALUDE_couple_consistency_l328_32825

-- Define the possible types of people
inductive PersonType
  | Knight
  | Liar
  | Normal

-- Define a couple
structure Couple :=
  (husband : PersonType)
  (wife : PersonType)

-- Define the statement made by a person about their spouse
def makeStatement (speaker : PersonType) (spouse : PersonType) : Prop :=
  spouse ≠ PersonType.Normal

-- Define the consistency of statements with reality
def isConsistent (couple : Couple) : Prop :=
  match couple.husband, couple.wife with
  | PersonType.Knight, _ => makeStatement PersonType.Knight couple.wife
  | PersonType.Liar, _ => ¬(makeStatement PersonType.Liar couple.wife)
  | PersonType.Normal, PersonType.Knight => makeStatement PersonType.Normal couple.wife
  | PersonType.Normal, PersonType.Liar => ¬(makeStatement PersonType.Normal couple.wife)
  | PersonType.Normal, PersonType.Normal => True

-- Theorem stating that the only consistent solution is both being normal people
theorem couple_consistency :
  ∀ (couple : Couple),
    isConsistent couple ∧
    makeStatement couple.husband couple.wife ∧
    makeStatement couple.wife couple.husband →
    couple.husband = PersonType.Normal ∧
    couple.wife = PersonType.Normal :=
sorry

end NUMINAMATH_CALUDE_couple_consistency_l328_32825


namespace NUMINAMATH_CALUDE_sum_of_polynomials_l328_32889

/-- Given three polynomial functions f, g, and h, prove their sum equals a specific polynomial. -/
theorem sum_of_polynomials (x : ℝ) : 
  let f := fun (x : ℝ) => -4*x^2 + 2*x - 5
  let g := fun (x : ℝ) => -6*x^2 + 4*x - 9
  let h := fun (x : ℝ) => 6*x^2 + 6*x + 2
  f x + g x + h x = -4*x^2 + 12*x - 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_polynomials_l328_32889


namespace NUMINAMATH_CALUDE_tan_alpha_value_l328_32812

theorem tan_alpha_value (α β : ℝ) 
  (h1 : Real.tan (α + β) = 3/5) 
  (h2 : Real.tan β = 1/3) : 
  Real.tan α = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l328_32812


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l328_32868

/-- Given two vectors a and b in ℝ², where a = (1, 2) and b = (x, -2),
    if a and b are perpendicular, then x = 4. -/
theorem perpendicular_vectors_x_value :
  ∀ (x : ℝ),
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![x, -2]
  (∀ i, i < 2 → a i * b i = 0) →
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l328_32868


namespace NUMINAMATH_CALUDE_imaginary_sum_zero_l328_32841

theorem imaginary_sum_zero (i : ℂ) (h : i^2 = -1) :
  i^1234 + i^1235 + i^1236 + i^1237 = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_sum_zero_l328_32841


namespace NUMINAMATH_CALUDE_test_questions_count_l328_32885

theorem test_questions_count (total_questions : ℕ) 
  (correct_answers : ℕ) (final_score : ℚ) :
  correct_answers = 104 →
  final_score = 100 →
  (correct_answers : ℚ) + ((total_questions - correct_answers : ℕ) : ℚ) * (-1/4) = final_score →
  total_questions = 120 :=
by sorry

end NUMINAMATH_CALUDE_test_questions_count_l328_32885


namespace NUMINAMATH_CALUDE_unique_root_of_equation_l328_32884

theorem unique_root_of_equation (a b c d : ℝ) 
  (h1 : a + d = 2016)
  (h2 : b + c = 2016)
  (h3 : a ≠ c) :
  ∃! x : ℝ, (x - a) * (x - b) = (x - c) * (x - d) ∧ x = 1008 :=
by sorry

end NUMINAMATH_CALUDE_unique_root_of_equation_l328_32884


namespace NUMINAMATH_CALUDE_bills_age_l328_32811

theorem bills_age (caroline_age : ℝ) 
  (h1 : caroline_age + (2 * caroline_age - 1) + (caroline_age - 4) = 45) : 
  2 * caroline_age - 1 = 24 := by
  sorry

#check bills_age

end NUMINAMATH_CALUDE_bills_age_l328_32811


namespace NUMINAMATH_CALUDE_triathlon_problem_l328_32802

/-- Triathlon problem -/
theorem triathlon_problem (v1 v2 v3 : ℝ) 
  (h1 : 1 / v1 + 25 / v2 + 4 / v3 = 5 / 4)
  (h2 : v1 / 16 + v2 / 49 + v3 / 49 = 5 / 4) :
  4 / v3 = 2 / 7 ∧ v3 = 14 := by sorry

end NUMINAMATH_CALUDE_triathlon_problem_l328_32802


namespace NUMINAMATH_CALUDE_exam_scores_l328_32819

theorem exam_scores (total_items : Nat) (lowella_percentage : Nat) (pamela_increase : Nat) :
  total_items = 100 →
  lowella_percentage = 35 →
  pamela_increase = 20 →
  let lowella_score := total_items * lowella_percentage / 100
  let pamela_score := lowella_score + lowella_score * pamela_increase / 100
  let mandy_score := 2 * pamela_score
  mandy_score = 84 := by sorry

end NUMINAMATH_CALUDE_exam_scores_l328_32819


namespace NUMINAMATH_CALUDE_expected_boy_girl_adjacencies_l328_32837

/-- The expected number of boy-girl adjacencies in a row of 6 boys and 14 girls -/
theorem expected_boy_girl_adjacencies :
  let num_boys : ℕ := 6
  let num_girls : ℕ := 14
  let total_people : ℕ := num_boys + num_girls
  let num_adjacencies : ℕ := total_people - 1
  let prob_boy_girl : ℚ := (num_boys : ℚ) / total_people * (num_girls : ℚ) / (total_people - 1)
  let expected_adjacencies : ℚ := 2 * prob_boy_girl * num_adjacencies
  expected_adjacencies = 798 / 95 := by
  sorry

end NUMINAMATH_CALUDE_expected_boy_girl_adjacencies_l328_32837


namespace NUMINAMATH_CALUDE_two_x_less_than_one_necessary_not_sufficient_l328_32898

theorem two_x_less_than_one_necessary_not_sufficient :
  (∀ x : ℝ, -1 < x ∧ x < 0 → 2*x < 1) ∧
  (∃ x : ℝ, 2*x < 1 ∧ ¬(-1 < x ∧ x < 0)) :=
by sorry

end NUMINAMATH_CALUDE_two_x_less_than_one_necessary_not_sufficient_l328_32898


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l328_32821

open Set

theorem intersection_A_complement_B (U A B : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) 
  (hA : A = {2, 4}) (hB : B = {4, 5}) : A ∩ (U \ B) = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l328_32821


namespace NUMINAMATH_CALUDE_claudia_water_amount_l328_32880

/-- The amount of water in ounces that Claudia had initially -/
def initial_water : ℕ := 122

/-- The capacity of a 5-ounce glass in ounces -/
def five_ounce_glass : ℕ := 5

/-- The capacity of an 8-ounce glass in ounces -/
def eight_ounce_glass : ℕ := 8

/-- The capacity of a 4-ounce glass in ounces -/
def four_ounce_glass : ℕ := 4

/-- The number of 5-ounce glasses filled -/
def num_five_ounce : ℕ := 6

/-- The number of 8-ounce glasses filled -/
def num_eight_ounce : ℕ := 4

/-- The number of 4-ounce glasses that can be filled with the remaining water -/
def num_four_ounce : ℕ := 15

theorem claudia_water_amount :
  initial_water = 
    num_five_ounce * five_ounce_glass + 
    num_eight_ounce * eight_ounce_glass + 
    num_four_ounce * four_ounce_glass := by
  sorry

end NUMINAMATH_CALUDE_claudia_water_amount_l328_32880


namespace NUMINAMATH_CALUDE_olive_flea_fraction_is_half_l328_32850

/-- The fraction of fleas Olive has compared to Gertrude -/
def olive_flea_fraction (gertrude_fleas maud_fleas olive_fleas total_fleas : ℕ) : ℚ :=
  olive_fleas / gertrude_fleas

theorem olive_flea_fraction_is_half :
  ∀ (gertrude_fleas maud_fleas olive_fleas total_fleas : ℕ),
    gertrude_fleas = 10 →
    maud_fleas = 5 * olive_fleas →
    total_fleas = 40 →
    gertrude_fleas + maud_fleas + olive_fleas = total_fleas →
    olive_flea_fraction gertrude_fleas maud_fleas olive_fleas total_fleas = 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_olive_flea_fraction_is_half_l328_32850


namespace NUMINAMATH_CALUDE_crazy_silly_school_series_l328_32820

theorem crazy_silly_school_series (total_movies : ℕ) (books_read : ℕ) (movies_watched : ℕ) (movies_to_watch : ℕ) :
  total_movies = 17 →
  books_read = 19 →
  movies_watched + movies_to_watch = total_movies →
  (∃ (different_books : ℕ), different_books = books_read) :=
by sorry

end NUMINAMATH_CALUDE_crazy_silly_school_series_l328_32820


namespace NUMINAMATH_CALUDE_two_alarms_parallel_reliability_l328_32891

/-- The reliability of a single alarm -/
def single_alarm_reliability : ℝ := 0.90

/-- The reliability of two independent alarms connected in parallel -/
def parallel_reliability (p : ℝ) : ℝ := 1 - (1 - p) * (1 - p)

theorem two_alarms_parallel_reliability :
  parallel_reliability single_alarm_reliability = 0.99 := by
  sorry

end NUMINAMATH_CALUDE_two_alarms_parallel_reliability_l328_32891


namespace NUMINAMATH_CALUDE_orchard_sections_l328_32834

/-- Given the daily harvest from each orchard section and the total daily harvest,
    calculate the number of orchard sections. -/
theorem orchard_sections 
  (sacks_per_section : ℕ) 
  (total_sacks : ℕ) 
  (h1 : sacks_per_section = 45)
  (h2 : total_sacks = 360) :
  total_sacks / sacks_per_section = 8 := by
  sorry

end NUMINAMATH_CALUDE_orchard_sections_l328_32834


namespace NUMINAMATH_CALUDE_integer_power_sum_l328_32808

theorem integer_power_sum (x : ℝ) (h1 : x ≠ 0) (h2 : ∃ k : ℤ, x + 1/x = k) :
  ∀ n : ℕ, ∃ m : ℤ, x^n + 1/(x^n) = m :=
sorry

end NUMINAMATH_CALUDE_integer_power_sum_l328_32808


namespace NUMINAMATH_CALUDE_sunzi_wood_measurement_l328_32810

theorem sunzi_wood_measurement 
  (x y : ℝ) 
  (h1 : y - x = 4.5) 
  (h2 : (1/2) * y < x) 
  (h3 : x < (1/2) * y + 1) : 
  y - x = 4.5 ∧ (1/2) * y = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_sunzi_wood_measurement_l328_32810


namespace NUMINAMATH_CALUDE_molecular_weight_CH3COOH_is_60_l328_32879

/-- The molecular weight of CH3COOH in grams per mole -/
def molecular_weight_CH3COOH : ℝ := 60

/-- The number of moles in the given sample -/
def sample_moles : ℝ := 6

/-- The total weight of the sample in grams -/
def sample_weight : ℝ := 360

/-- Theorem stating that the molecular weight of CH3COOH is 60 grams/mole -/
theorem molecular_weight_CH3COOH_is_60 :
  molecular_weight_CH3COOH = sample_weight / sample_moles :=
by sorry

end NUMINAMATH_CALUDE_molecular_weight_CH3COOH_is_60_l328_32879


namespace NUMINAMATH_CALUDE_AB_vector_l328_32878

def OA : ℝ × ℝ := (2, 1)
def OB : ℝ × ℝ := (-3, 4)

theorem AB_vector : 
  let AB := (OB.1 - OA.1, OB.2 - OA.2)
  AB = (-5, 3) := by sorry

end NUMINAMATH_CALUDE_AB_vector_l328_32878
