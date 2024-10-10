import Mathlib

namespace quadratic_two_distinct_roots_l2768_276805

theorem quadratic_two_distinct_roots (c : ℝ) (h : c < 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + c = 0 ∧ x₂^2 + 2*x₂ + c = 0 :=
sorry

end quadratic_two_distinct_roots_l2768_276805


namespace max_good_sequences_75_each_l2768_276839

/-- Represents a string of beads -/
structure BeadString :=
  (blue : ℕ)
  (red : ℕ)
  (green : ℕ)

/-- Defines a "good" sequence of beads -/
def is_good_sequence (seq : List Char) : Bool :=
  seq.length = 5 ∧ 
  seq.count 'G' = 3 ∧ 
  seq.count 'R' = 1 ∧ 
  seq.count 'B' = 1

/-- Calculates the maximum number of "good" sequences in a bead string -/
def max_good_sequences (s : BeadString) : ℕ :=
  min (s.green * 5 / 3) (min s.red s.blue)

/-- Theorem stating the maximum number of "good" sequences for the given bead string -/
theorem max_good_sequences_75_each (s : BeadString) 
  (h1 : s.blue = 75) (h2 : s.red = 75) (h3 : s.green = 75) : 
  max_good_sequences s = 123 := by
  sorry

end max_good_sequences_75_each_l2768_276839


namespace intersection_equality_iff_t_range_l2768_276899

/-- The set M -/
def M : Set ℝ := {x | -2 < x ∧ x < 5}

/-- The set N parameterized by t -/
def N (t : ℝ) : Set ℝ := {x | 2 - t < x ∧ x < 2 * t + 1}

/-- Theorem stating the equivalence between M ∩ N = N and t ∈ (-∞, 2] -/
theorem intersection_equality_iff_t_range :
  ∀ t : ℝ, (M ∩ N t = N t) ↔ t ≤ 2 := by sorry

end intersection_equality_iff_t_range_l2768_276899


namespace figure_to_square_partition_l2768_276894

/-- Represents a point on a 2D grid --/
structure GridPoint where
  x : Int
  y : Int

/-- Represents a planar figure on a grid --/
def PlanarFigure := Set GridPoint

/-- Represents a transformation that can be applied to a set of points --/
structure Transformation where
  rotate : ℤ → GridPoint → GridPoint
  translate : ℤ → ℤ → GridPoint → GridPoint

/-- Checks if a set of points forms a square --/
def is_square (s : Set GridPoint) : Prop := sorry

/-- The main theorem --/
theorem figure_to_square_partition 
  (F : PlanarFigure) 
  (G : Set GridPoint) -- The grid
  (T : Transformation) -- Available transformations
  : 
  ∃ (S1 S2 S3 : Set GridPoint),
    (S1 ∪ S2 ∪ S3 = F) ∧ 
    (S1 ∩ S2 ≠ ∅) ∧ 
    (S2 ∩ S3 ≠ ∅) ∧ 
    (S3 ∩ S1 ≠ ∅) ∧
    ∃ (S : Set GridPoint), 
      is_square S ∧ 
      ∃ (f1 f2 f3 : Set GridPoint → Set GridPoint),
        (∀ p ∈ S1, ∃ q, q = T.rotate r1 (T.translate dx1 dy1 p) ∧ f1 {p} = {q}) ∧
        (∀ p ∈ S2, ∃ q, q = T.rotate r2 (T.translate dx2 dy2 p) ∧ f2 {p} = {q}) ∧
        (∀ p ∈ S3, ∃ q, q = T.rotate r3 (T.translate dx3 dy3 p) ∧ f3 {p} = {q}) ∧
        (f1 S1 ∪ f2 S2 ∪ f3 S3 = S)
  := by sorry

end figure_to_square_partition_l2768_276894


namespace dot_product_sum_equilateral_triangle_l2768_276846

-- Define the equilateral triangle
def EquilateralTriangle (A B C : ℝ × ℝ) : Prop :=
  (dist A B = 1) ∧ (dist B C = 1) ∧ (dist C A = 1)

-- Define vectors a, b, c
def a (B C : ℝ × ℝ) : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)
def b (A C : ℝ × ℝ) : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
def c (A B : ℝ × ℝ) : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem dot_product_sum_equilateral_triangle (A B C : ℝ × ℝ) 
  (h : EquilateralTriangle A B C) : 
  dot_product (a B C) (b A C) + dot_product (b A C) (c A B) + dot_product (c A B) (a B C) = 1/2 := by
  sorry

end dot_product_sum_equilateral_triangle_l2768_276846


namespace trig_inequality_l2768_276870

theorem trig_inequality : ∀ a b c : ℝ,
  a = Real.sin (21 * π / 180) →
  b = Real.cos (72 * π / 180) →
  c = Real.tan (23 * π / 180) →
  c > a ∧ a > b :=
by sorry

end trig_inequality_l2768_276870


namespace seashell_collection_l2768_276835

/-- Calculates the total number of seashells after Leo gives away a quarter of his collection -/
theorem seashell_collection (henry paul total : ℕ) (h1 : henry = 11) (h2 : paul = 24) (h3 : total = 59) :
  let leo := total - henry - paul
  let leo_remaining := leo - (leo / 4)
  henry + paul + leo_remaining = 53 := by sorry

end seashell_collection_l2768_276835


namespace length_AB_on_parabola_l2768_276807

/-- Parabola type -/
structure Parabola where
  a : ℝ
  C : ℝ × ℝ → Prop
  focus : ℝ × ℝ

/-- Point on a parabola -/
structure PointOnParabola (p : Parabola) where
  point : ℝ × ℝ
  on_parabola : p.C point

/-- Tangent line to a parabola at a point -/
def tangent_line (p : Parabola) (pt : PointOnParabola p) : ℝ × ℝ → Prop := sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Theorem: Length of AB on parabola y² = 6x -/
theorem length_AB_on_parabola (p : Parabola) 
  (h_eq : p.C = fun (x, y) ↦ y^2 = 6*x) 
  (A B : PointOnParabola p) 
  (F : ℝ × ℝ) 
  (h_focus : F = p.focus)
  (h_collinear : ∃ (m : ℝ), A.point.1 = m * A.point.2 + F.1 ∧ 
                             B.point.1 = m * B.point.2 + F.1)
  (P : ℝ × ℝ)
  (h_tangent_intersect : (tangent_line p A) P ∧ (tangent_line p B) P)
  (h_PF_distance : distance P F = 2 * Real.sqrt 3) :
  distance A.point B.point = 8 := by sorry

end length_AB_on_parabola_l2768_276807


namespace marble_distribution_l2768_276848

/-- Represents a distribution of marbles into bags -/
def Distribution := List Nat

/-- Checks if a distribution is valid for a given number of children -/
def isValidDistribution (d : Distribution) (numChildren : Nat) : Prop :=
  d.sum = 77 ∧ d.length ≥ numChildren ∧ (77 % numChildren = 0)

/-- The minimum number of bags needed -/
def minBags : Nat := 17

theorem marble_distribution :
  (∀ d : Distribution, d.length < minBags → ¬(isValidDistribution d 7 ∧ isValidDistribution d 11)) ∧
  (∃ d : Distribution, d.length = minBags ∧ isValidDistribution d 7 ∧ isValidDistribution d 11) :=
sorry

#check marble_distribution

end marble_distribution_l2768_276848


namespace beginner_course_fraction_l2768_276825

theorem beginner_course_fraction :
  ∀ (total_students : ℕ) (calculus_students : ℕ) (trigonometry_students : ℕ) 
    (beginner_calculus : ℕ) (beginner_trigonometry : ℕ),
  total_students > 0 →
  calculus_students + trigonometry_students = total_students →
  trigonometry_students = (3 * calculus_students) / 2 →
  beginner_calculus = (4 * calculus_students) / 5 →
  (beginner_trigonometry : ℚ) / total_students = 48 / 100 →
  (beginner_calculus + beginner_trigonometry : ℚ) / total_students = 4 / 5 :=
by sorry

end beginner_course_fraction_l2768_276825


namespace repeating_decimal_sum_l2768_276864

/-- Represents a repeating decimal of the form 0.abcabc... where abc is a finite sequence of digits -/
def RepeatingDecimal (numerator denominator : ℕ) : ℚ := numerator / denominator

theorem repeating_decimal_sum : 
  RepeatingDecimal 4 33 + RepeatingDecimal 2 999 + RepeatingDecimal 2 99999 = 12140120 / 99999 := by
  sorry

end repeating_decimal_sum_l2768_276864


namespace two_planes_division_l2768_276838

/-- Represents the possible configurations of two planes in 3D space -/
inductive PlaneConfiguration
  | Parallel
  | Intersecting

/-- Represents the number of parts that two planes divide the space into -/
def spaceDivisions (config : PlaneConfiguration) : Nat :=
  match config with
  | PlaneConfiguration.Parallel => 3
  | PlaneConfiguration.Intersecting => 4

/-- Theorem stating that two planes divide space into either 3 or 4 parts -/
theorem two_planes_division :
  ∀ (config : PlaneConfiguration), 
    (spaceDivisions config = 3 ∨ spaceDivisions config = 4) :=
by sorry

end two_planes_division_l2768_276838


namespace binary_arithmetic_equality_l2768_276808

/-- Converts a binary number (represented as a list of bits) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- Represents a binary number as a list of bits (true for 1, false for 0) -/
def binary_10110 : List Bool := [true, false, true, true, false]
def binary_1101 : List Bool := [true, true, false, true]
def binary_110 : List Bool := [true, true, false]
def binary_101 : List Bool := [true, false, true]
def binary_1010 : List Bool := [true, false, true, false]

/-- The main theorem to prove -/
theorem binary_arithmetic_equality :
  binary_to_decimal binary_10110 - binary_to_decimal binary_1101 +
  binary_to_decimal binary_110 - binary_to_decimal binary_101 =
  binary_to_decimal binary_1010 := by
  sorry

end binary_arithmetic_equality_l2768_276808


namespace function_relationship_l2768_276884

def main (f : ℝ → ℝ) : Prop :=
  (∀ x, f (2 - x) = f x) ∧
  (∀ x, f (x + 2) = f (x - 2)) ∧
  (∀ x₁ x₂, x₁ ∈ Set.Icc 1 3 → x₂ ∈ Set.Icc 1 3 → x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0) →
  f 2016 = f 2014 ∧ f 2014 > f 2015

theorem function_relationship : main f :=
sorry

end function_relationship_l2768_276884


namespace class_size_is_50_l2768_276896

def original_average : ℝ := 87.26
def incorrect_score : ℝ := 89
def correct_score : ℝ := 98
def new_average : ℝ := 87.44

theorem class_size_is_50 : 
  ∃ n : ℕ, n > 0 ∧ 
  (n : ℝ) * new_average = (n : ℝ) * original_average + (correct_score - incorrect_score) ∧
  n = 50 :=
sorry

end class_size_is_50_l2768_276896


namespace pure_imaginary_complex_number_l2768_276867

/-- Given that z = (1-mi)/(2+i) is a pure imaginary number, prove that m = 2 --/
theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := (1 - m * Complex.I) / (2 + Complex.I)
  (∃ (y : ℝ), z = y * Complex.I) → m = 2 := by
  sorry

end pure_imaginary_complex_number_l2768_276867


namespace product_of_fractions_l2768_276873

theorem product_of_fractions : 
  let f (n : ℕ) := (n^4 - 1) / (n^4 + 1)
  (f 3) * (f 4) * (f 5) * (f 6) * (f 7) = 880 / 91 := by
  sorry

end product_of_fractions_l2768_276873


namespace pattern_proof_l2768_276850

theorem pattern_proof (n : ℕ) (h : n > 0) : 
  Real.sqrt (n - n / (n^2 + 1)) = n * Real.sqrt (n / (n^2 + 1)) := by
  sorry

end pattern_proof_l2768_276850


namespace no_solution_double_inequality_l2768_276878

theorem no_solution_double_inequality :
  ¬∃ x : ℝ, (3 * x - 2 < (x + 2)^2) ∧ ((x + 2)^2 < 9 * x - 5) := by
  sorry

end no_solution_double_inequality_l2768_276878


namespace gnome_count_after_removal_l2768_276874

/-- The number of gnomes in each forest and the total remaining after removal --/
theorem gnome_count_after_removal :
  let westerville : ℕ := 20
  let ravenswood : ℕ := 4 * westerville
  let greenwood : ℕ := ravenswood + ravenswood / 4
  let remaining_westerville : ℕ := westerville - westerville * 3 / 10
  let remaining_ravenswood : ℕ := ravenswood - ravenswood * 2 / 5
  let remaining_greenwood : ℕ := greenwood - greenwood / 2
  remaining_westerville + remaining_ravenswood + remaining_greenwood = 112 := by
sorry

end gnome_count_after_removal_l2768_276874


namespace bhanu_petrol_expenditure_l2768_276837

theorem bhanu_petrol_expenditure (income : ℝ) 
  (h1 : income > 0)
  (h2 : 0.14 * (income - 0.3 * income) = 98) : 
  0.3 * income = 300 := by
  sorry

end bhanu_petrol_expenditure_l2768_276837


namespace angle_B_measure_max_area_l2768_276881

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

/-- The given condition a^2 + c^2 = b^2 - ac -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.a^2 + t.c^2 = t.b^2 - t.a * t.c

theorem angle_B_measure (t : Triangle) (h : satisfiesCondition t) :
  t.B = 2 * π / 3 := by sorry

theorem max_area (t : Triangle) (h1 : satisfiesCondition t) (h2 : t.b = 2 * Real.sqrt 3) :
  (t.a * t.c * Real.sin t.B) / 2 ≤ Real.sqrt 3 := by sorry

end angle_B_measure_max_area_l2768_276881


namespace strawberry_calories_is_4_l2768_276892

/-- The number of strawberries Zoe ate -/
def num_strawberries : ℕ := 12

/-- The amount of yogurt Zoe ate in ounces -/
def yogurt_ounces : ℕ := 6

/-- The number of calories per ounce of yogurt -/
def yogurt_calories_per_ounce : ℕ := 17

/-- The total calories Zoe ate -/
def total_calories : ℕ := 150

/-- The number of calories in each strawberry -/
def strawberry_calories : ℕ := (total_calories - yogurt_ounces * yogurt_calories_per_ounce) / num_strawberries

theorem strawberry_calories_is_4 : strawberry_calories = 4 := by
  sorry

end strawberry_calories_is_4_l2768_276892


namespace expression_value_l2768_276800

theorem expression_value (a b : ℚ) (ha : a = -1) (hb : b = 1/4) :
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) = 1 := by sorry

end expression_value_l2768_276800


namespace local_taxes_in_cents_l2768_276819

/-- The hourly wage in dollars -/
def hourly_wage : ℝ := 25

/-- The local tax rate as a decimal -/
def tax_rate : ℝ := 0.024

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- Theorem: The amount of local taxes paid in cents per hour is 60 -/
theorem local_taxes_in_cents : 
  (hourly_wage * tax_rate * cents_per_dollar : ℝ) = 60 := by sorry

end local_taxes_in_cents_l2768_276819


namespace total_birds_count_l2768_276820

def birds_monday : ℕ := 70

def birds_tuesday : ℕ := birds_monday / 2

def birds_wednesday : ℕ := birds_tuesday + 8

theorem total_birds_count : birds_monday + birds_tuesday + birds_wednesday = 148 := by
  sorry

end total_birds_count_l2768_276820


namespace greatest_value_quadratic_inequality_l2768_276877

theorem greatest_value_quadratic_inequality :
  ∃ (a : ℝ), a^2 - 10*a + 21 ≤ 0 ∧ ∀ (x : ℝ), x^2 - 10*x + 21 ≤ 0 → x ≤ a :=
by
  -- The proof goes here
  sorry

end greatest_value_quadratic_inequality_l2768_276877


namespace geometric_progression_ratio_l2768_276851

def is_valid_ratio (a r : ℕ+) : Prop :=
  a * r^2 + a * r^4 + a * r^6 = 819 * 6^2016

theorem geometric_progression_ratio :
  ∃ (a : ℕ+), is_valid_ratio a 1 ∧ is_valid_ratio a 2 ∧ is_valid_ratio a 3 ∧ is_valid_ratio a 4 ∧
  ∀ (r : ℕ+), r ≠ 1 ∧ r ≠ 2 ∧ r ≠ 3 ∧ r ≠ 4 → ¬(∃ (b : ℕ+), is_valid_ratio b r) :=
sorry

end geometric_progression_ratio_l2768_276851


namespace kelly_bought_five_more_paper_l2768_276861

/-- Calculates the number of additional pieces of construction paper Kelly bought --/
def additional_construction_paper (students : ℕ) (paper_per_student : ℕ) (glue_bottles : ℕ) (final_supplies : ℕ) : ℕ :=
  let initial_supplies := students * paper_per_student + glue_bottles
  let remaining_supplies := initial_supplies / 2
  final_supplies - remaining_supplies

/-- Proves that Kelly bought 5 additional pieces of construction paper --/
theorem kelly_bought_five_more_paper : 
  additional_construction_paper 8 3 6 20 = 5 := by
  sorry

#eval additional_construction_paper 8 3 6 20

end kelly_bought_five_more_paper_l2768_276861


namespace complex_in_second_quadrant_l2768_276875

/-- A complex number z is in the second quadrant if its real part is negative and its imaginary part is positive -/
def in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

/-- Given that (2a+2i)/(1+i) is purely imaginary for some real a, 
    prove that 2a+2i is in the second quadrant -/
theorem complex_in_second_quadrant (a : ℝ) 
    (h : (Complex.I : ℂ).re * ((2*a + 2*Complex.I) / (1 + Complex.I)).im = 
         (Complex.I : ℂ).im * ((2*a + 2*Complex.I) / (1 + Complex.I)).re) : 
    in_second_quadrant (2*a + 2*Complex.I) := by
  sorry


end complex_in_second_quadrant_l2768_276875


namespace sequence_general_term_l2768_276826

theorem sequence_general_term (a : ℕ → ℚ) :
  a 1 = -1 ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 1) * a n = a (n + 1) - a n) →
  ∀ n : ℕ, n ≥ 1 → a n = -1 / n :=
sorry

end sequence_general_term_l2768_276826


namespace power_product_equality_l2768_276817

theorem power_product_equality : (-4 : ℝ)^2013 * (-0.25 : ℝ)^2014 = -0.25 := by
  sorry

end power_product_equality_l2768_276817


namespace dan_final_marbles_l2768_276868

/-- The number of marbles Dan has after giving some away and receiving more. -/
def final_marbles (initial : ℕ) (given_mary : ℕ) (given_peter : ℕ) (received : ℕ) : ℕ :=
  initial - given_mary - given_peter + received

/-- Theorem stating that Dan has 98 marbles at the end. -/
theorem dan_final_marbles :
  final_marbles 128 24 16 10 = 98 := by
  sorry

end dan_final_marbles_l2768_276868


namespace tickets_spent_on_beanie_l2768_276829

/-- Proves the number of tickets spent on a beanie given initial tickets, additional tickets won, and final ticket count. -/
theorem tickets_spent_on_beanie 
  (initial_tickets : ℕ) 
  (additional_tickets : ℕ) 
  (final_tickets : ℕ) 
  (h1 : initial_tickets = 49)
  (h2 : additional_tickets = 6)
  (h3 : final_tickets = 30)
  : initial_tickets - (initial_tickets - final_tickets + additional_tickets) = 25 := by
  sorry

end tickets_spent_on_beanie_l2768_276829


namespace quadratic_roots_sum_of_inverses_squared_l2768_276802

theorem quadratic_roots_sum_of_inverses_squared (p q : ℝ) : 
  (3 * p^2 - 5 * p + 2 = 0) → 
  (3 * q^2 - 5 * q + 2 = 0) → 
  (1 / p^2 + 1 / q^2 = 13 / 4) := by
  sorry

end quadratic_roots_sum_of_inverses_squared_l2768_276802


namespace tangent_line_at_x_1_l2768_276857

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 - 3*x + 3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 2*x - 3

-- Theorem statement
theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (y = -2*x + 2) :=
by sorry

end tangent_line_at_x_1_l2768_276857


namespace medication_C_consumption_l2768_276890

def days_in_july : ℕ := 31

def doses_per_day_C : ℕ := 3

def missed_days_C : ℕ := 2

theorem medication_C_consumption :
  days_in_july * doses_per_day_C - missed_days_C * doses_per_day_C = 87 := by
  sorry

end medication_C_consumption_l2768_276890


namespace car_can_climb_slope_l2768_276841

theorem car_can_climb_slope (car_max_angle : Real) (slope_gradient : Real) : 
  car_max_angle = 60 * Real.pi / 180 →
  slope_gradient = 1.5 →
  Real.tan car_max_angle > slope_gradient := by
  sorry

end car_can_climb_slope_l2768_276841


namespace circle_properties_l2768_276816

/-- For a circle with area 4π, prove its diameter is 4 and circumference is 4π -/
theorem circle_properties (r : ℝ) (h : r^2 * π = 4 * π) : 
  2 * r = 4 ∧ 2 * π * r = 4 * π :=
sorry

end circle_properties_l2768_276816


namespace arithmetic_sequence_11th_term_l2768_276858

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_11th_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_2 : a 2 = 3)
  (h_6 : a 6 = 7) :
  a 11 = 12 := by
sorry

end arithmetic_sequence_11th_term_l2768_276858


namespace seating_theorem_l2768_276810

/-- The number of desks in a row -/
def num_desks : ℕ := 6

/-- The number of students to be seated -/
def num_students : ℕ := 2

/-- The minimum number of empty desks required between students -/
def min_gap : ℕ := 1

/-- The number of ways to seat students in desks with the given constraints -/
def seating_arrangements (n_desks n_students min_gap : ℕ) : ℕ :=
  sorry

theorem seating_theorem :
  seating_arrangements num_desks num_students min_gap = 9 := by
  sorry

end seating_theorem_l2768_276810


namespace smallest_non_factor_product_l2768_276822

theorem smallest_non_factor_product (a b : ℕ+) : 
  a ≠ b →
  a ∣ 48 →
  b ∣ 48 →
  ¬(a * b ∣ 48) →
  (∀ c d : ℕ+, c ≠ d → c ∣ 48 → d ∣ 48 → ¬(c * d ∣ 48) → a * b ≤ c * d) →
  a * b = 32 := by
sorry

end smallest_non_factor_product_l2768_276822


namespace smartphone_price_problem_l2768_276876

theorem smartphone_price_problem (store_a_price : ℝ) (store_a_discount : ℝ) (store_b_discount : ℝ) :
  store_a_price = 125 →
  store_a_discount = 0.08 →
  store_b_discount = 0.10 →
  store_a_price * (1 - store_a_discount) = store_b_price * (1 - store_b_discount) - 2 →
  store_b_price = 130 :=
by
  sorry

#check smartphone_price_problem

end smartphone_price_problem_l2768_276876


namespace jackson_score_l2768_276840

/-- Given a basketball team's scoring information, calculate Jackson's score. -/
theorem jackson_score (total_score : ℕ) (other_players : ℕ) (avg_score : ℕ) (h1 : total_score = 72) (h2 : other_players = 7) (h3 : avg_score = 6) : total_score - other_players * avg_score = 30 := by
  sorry

end jackson_score_l2768_276840


namespace four_more_laps_needed_l2768_276811

/-- Calculates the number of additional laps needed to reach a total distance -/
def additional_laps_needed (total_distance : ℕ) (track_length : ℕ) (laps_run_per_person : ℕ) (num_people : ℕ) : ℕ :=
  let total_laps_run := laps_run_per_person * num_people
  let distance_covered := total_laps_run * track_length
  let remaining_distance := total_distance - distance_covered
  remaining_distance / track_length

/-- Theorem: Given the problem conditions, 4 additional laps are needed -/
theorem four_more_laps_needed :
  additional_laps_needed 2400 150 6 2 = 4 := by
  sorry

#eval additional_laps_needed 2400 150 6 2

end four_more_laps_needed_l2768_276811


namespace alloy_chromium_percentage_l2768_276891

/-- The percentage of chromium in an alloy mixture -/
def chromium_percentage (m1 m2 p1 p2 p3 : ℝ) : Prop :=
  m1 * p1 / 100 + m2 * p2 / 100 = (m1 + m2) * p3 / 100

/-- The problem statement -/
theorem alloy_chromium_percentage :
  ∃ (x : ℝ),
    chromium_percentage 15 30 12 x 9.333333333333334 ∧
    x = 8 := by sorry

end alloy_chromium_percentage_l2768_276891


namespace simon_blueberry_theorem_l2768_276862

/-- The number of blueberries Simon picked from his own bushes -/
def own_blueberries : ℕ := 100

/-- The number of blueberries needed for each pie -/
def blueberries_per_pie : ℕ := 100

/-- The number of pies Simon can make -/
def number_of_pies : ℕ := 3

/-- The number of blueberries Simon picked from nearby bushes -/
def nearby_blueberries : ℕ := number_of_pies * blueberries_per_pie - own_blueberries

theorem simon_blueberry_theorem : nearby_blueberries = 200 := by
  sorry

end simon_blueberry_theorem_l2768_276862


namespace school_population_theorem_l2768_276815

/-- Represents the school population statistics -/
structure SchoolPopulation where
  y : ℕ  -- Total number of students
  x : ℚ  -- Percentage of boys that 162 students represent
  z : ℚ  -- Percentage of girls in the school

/-- The conditions given in the problem -/
def school_conditions (pop : SchoolPopulation) : Prop :=
  (162 : ℚ) = pop.x / 100 * (1/2 : ℚ) * pop.y ∧ 
  pop.z = 100 - 50

/-- The theorem to be proved -/
theorem school_population_theorem (pop : SchoolPopulation) 
  (h : school_conditions pop) : 
  pop.z = 50 ∧ pop.x = 32400 / pop.y := by
  sorry


end school_population_theorem_l2768_276815


namespace trip_duration_l2768_276809

/-- Represents the duration of a car trip with varying speeds -/
def car_trip (initial_hours : ℝ) (initial_speed : ℝ) (additional_speed : ℝ) (average_speed : ℝ) : Prop :=
  ∃ (additional_hours : ℝ),
    let total_hours := initial_hours + additional_hours
    let total_distance := initial_hours * initial_speed + additional_hours * additional_speed
    (total_distance / total_hours = average_speed) ∧
    (total_hours = 15)

/-- The main theorem stating that under given conditions, the trip duration is 15 hours -/
theorem trip_duration :
  car_trip 5 30 42 38 := by
  sorry

end trip_duration_l2768_276809


namespace lanas_nickels_l2768_276836

theorem lanas_nickels (num_stacks : ℕ) (nickels_per_stack : ℕ) 
  (h1 : num_stacks = 9) (h2 : nickels_per_stack = 8) : 
  num_stacks * nickels_per_stack = 72 := by
  sorry

end lanas_nickels_l2768_276836


namespace tin_weight_in_water_l2768_276827

theorem tin_weight_in_water (total_weight : ℝ) (weight_lost : ℝ) (tin_silver_ratio : ℝ) 
  (tin_loss : ℝ) (silver_weight : ℝ) (silver_loss : ℝ) :
  total_weight = 60 →
  weight_lost = 6 →
  tin_silver_ratio = 2/3 →
  tin_loss = 1.375 →
  silver_weight = 5 →
  silver_loss = 0.375 →
  ∃ (tin_weight : ℝ), tin_weight * (weight_lost / total_weight) = tin_loss ∧ 
    tin_weight = 13.75 := by
  sorry

end tin_weight_in_water_l2768_276827


namespace specific_grid_area_l2768_276806

/-- A rectangular grid formed by perpendicular lines -/
structure RectangularGrid where
  num_boundary_lines : ℕ
  perimeter : ℝ
  is_rectangular : Bool
  has_perpendicular_lines : Bool

/-- The area of a rectangular grid -/
def grid_area (grid : RectangularGrid) : ℝ :=
  sorry

/-- Theorem stating the area of the specific rectangular grid -/
theorem specific_grid_area :
  ∀ (grid : RectangularGrid),
    grid.num_boundary_lines = 36 ∧
    grid.perimeter = 72 ∧
    grid.is_rectangular = true ∧
    grid.has_perpendicular_lines = true →
    grid_area grid = 84 :=
  sorry

end specific_grid_area_l2768_276806


namespace difference_of_squares_factorization_l2768_276824

theorem difference_of_squares_factorization (x : ℝ) : 9 - 4 * x^2 = (3 - 2*x) * (3 + 2*x) := by
  sorry

end difference_of_squares_factorization_l2768_276824


namespace regular_polygon_sides_l2768_276803

theorem regular_polygon_sides (n : ℕ) (h : n > 2) :
  ((n - 2) * 180 : ℝ) / n = 160 → n = 18 := by
  sorry

end regular_polygon_sides_l2768_276803


namespace a_sequence_square_values_l2768_276801

def a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | (n + 3) => a (n + 2) + a (n + 1) + a n

theorem a_sequence_square_values (n : ℕ) : 
  (n > 0 ∧ a (n - 1) = n^2) ↔ (n = 1 ∨ n = 9) := by
  sorry

#check a_sequence_square_values

end a_sequence_square_values_l2768_276801


namespace factoring_expression_l2768_276854

theorem factoring_expression (x y : ℝ) :
  5 * x * (x + 4) + 2 * (x + 4) * (y + 2) = (x + 4) * (5 * x + 2 * y + 4) := by
  sorry

end factoring_expression_l2768_276854


namespace prob_white_second_is_half_l2768_276887

/-- Represents the number of black balls initially in the bag -/
def initial_black_balls : ℕ := 4

/-- Represents the number of white balls initially in the bag -/
def initial_white_balls : ℕ := 3

/-- Represents the total number of balls initially in the bag -/
def total_balls : ℕ := initial_black_balls + initial_white_balls

/-- Represents the probability of drawing a white ball on the second draw,
    given that a black ball was drawn on the first draw -/
def prob_white_second_given_black_first : ℚ :=
  initial_white_balls / (total_balls - 1)

theorem prob_white_second_is_half :
  prob_white_second_given_black_first = 1/2 := by
  sorry

end prob_white_second_is_half_l2768_276887


namespace angle_in_second_quadrant_l2768_276833

theorem angle_in_second_quadrant : 
  let θ := (29 * Real.pi) / 6
  0 < θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < Real.pi :=
by
  sorry

end angle_in_second_quadrant_l2768_276833


namespace ellipse_inscribed_circle_max_area_l2768_276847

/-- The ellipse with equation x²/4 + y²/3 = 1 -/
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- The left focus of the ellipse -/
def F₁ : ℝ × ℝ := (-1, 0)

/-- The right focus of the ellipse -/
def F₂ : ℝ × ℝ := (1, 0)

/-- A line passing through F₂ -/
def line_through_F₂ (m : ℝ) (x y : ℝ) : Prop := y = m * (x - 1)

/-- The area of the inscribed circle in triangle F₁MN -/
def inscribed_circle_area (m : ℝ) : ℝ := sorry

theorem ellipse_inscribed_circle_max_area :
  ∃ (max_area : ℝ),
    (∀ m : ℝ, inscribed_circle_area m ≤ max_area) ∧
    (max_area = 9 * Real.pi / 16) ∧
    (∀ m : ℝ, inscribed_circle_area m = max_area ↔ m = 0) :=
  sorry

end ellipse_inscribed_circle_max_area_l2768_276847


namespace diophantine_equation_solutions_l2768_276886

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 19 ↔ 
  (x = 20 ∧ y = 380) ∨ (x = 380 ∧ y = 20) ∨ 
  (x = 18 ∧ y = -342) ∨ (x = -342 ∧ y = 18) ∨ 
  (x = 38 ∧ y = 38) :=
by sorry

end diophantine_equation_solutions_l2768_276886


namespace part_one_part_two_l2768_276823

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 ≤ 4}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- Part I
theorem part_one (m : ℝ) : A m ∩ B = {x | 0 ≤ x ∧ x ≤ 3} → m = 2 := by sorry

-- Part II
theorem part_two (m : ℝ) : B ⊆ (Set.univ \ A m) → m > 5 ∨ m < -3 := by sorry

end part_one_part_two_l2768_276823


namespace interest_rate_is_30_percent_l2768_276843

-- Define the compound interest function
def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := P * (1 + r) ^ t

-- State the theorem
theorem interest_rate_is_30_percent 
  (P : ℝ) 
  (h1 : compound_interest P r 2 = 17640) 
  (h2 : compound_interest P r 3 = 22932) : 
  r = 0.3 := by
  sorry


end interest_rate_is_30_percent_l2768_276843


namespace eighth_grade_girls_l2768_276818

theorem eighth_grade_girls (total_students : ℕ) (boys girls : ℕ) : 
  total_students = 68 →
  boys = 2 * girls - 16 →
  total_students = boys + girls →
  girls = 28 := by
sorry

end eighth_grade_girls_l2768_276818


namespace ring_arrangements_l2768_276880

theorem ring_arrangements (n k f : ℕ) (hn : n = 10) (hk : k = 7) (hf : f = 5) :
  (n.choose k) * k.factorial * ((k + f - 1).choose (f - 1)) = 200160000 :=
sorry

end ring_arrangements_l2768_276880


namespace triangle_angle_measure_l2768_276860

theorem triangle_angle_measure (P Q R : ℝ) (h1 : P = 2 * Q) (h2 : R = 5 * Q) (h3 : P + Q + R = 180) : P = 45 := by
  sorry

end triangle_angle_measure_l2768_276860


namespace boys_neither_happy_nor_sad_l2768_276832

theorem boys_neither_happy_nor_sad (total_children : ℕ) (happy_children : ℕ) (sad_children : ℕ)
  (total_boys : ℕ) (total_girls : ℕ) (happy_boys : ℕ) (sad_girls : ℕ)
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : sad_children = 10)
  (h4 : total_boys = 22)
  (h5 : total_girls = 38)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4)
  (h8 : total_children = total_boys + total_girls)
  (h9 : sad_children ≥ sad_girls) :
  total_boys - happy_boys - (sad_children - sad_girls) = 10 := by
  sorry

end boys_neither_happy_nor_sad_l2768_276832


namespace car_rental_rates_equal_l2768_276849

/-- The daily rate of Safety Rent-a-Car in dollars -/
def safety_daily_rate : ℝ := 21.95

/-- The per-mile rate of Safety Rent-a-Car in dollars -/
def safety_mile_rate : ℝ := 0.19

/-- The per-mile rate of the second company in dollars -/
def second_mile_rate : ℝ := 0.21

/-- The number of miles driven -/
def miles_driven : ℝ := 150

/-- The daily rate of the second company in dollars -/
def second_daily_rate : ℝ := 18.95

theorem car_rental_rates_equal :
  safety_daily_rate + safety_mile_rate * miles_driven =
  second_daily_rate + second_mile_rate * miles_driven :=
by sorry

end car_rental_rates_equal_l2768_276849


namespace f_of_five_equals_102_l2768_276834

/-- Given a function f(x) = 2x^2 + y where f(2) = 60, prove that f(5) = 102 -/
theorem f_of_five_equals_102 (f : ℝ → ℝ) (y : ℝ) 
  (h1 : ∀ x, f x = 2 * x^2 + y)
  (h2 : f 2 = 60) :
  f 5 = 102 := by
  sorry

end f_of_five_equals_102_l2768_276834


namespace courtyard_length_proof_l2768_276813

/-- Proves that the length of a rectangular courtyard is 15 m given specific conditions -/
theorem courtyard_length_proof (width : ℝ) (stone_length : ℝ) (stone_width : ℝ) (total_stones : ℕ) :
  width = 6 →
  stone_length = 3 →
  stone_width = 2 →
  total_stones = 15 →
  (width * (width * total_stones * stone_length * stone_width / width / stone_length / stone_width)) = 15 := by
  sorry

end courtyard_length_proof_l2768_276813


namespace fraction_problem_l2768_276852

theorem fraction_problem (x : ℚ) : (3/4 : ℚ) * x * (2/3 : ℚ) = (2/5 : ℚ) → x = (4/5 : ℚ) := by
  sorry

end fraction_problem_l2768_276852


namespace bush_height_after_two_years_l2768_276869

/-- The height of a bush after a given number of years -/
def bush_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * 4^years

/-- Theorem stating the height of the bush after 2 years -/
theorem bush_height_after_two_years
  (h : bush_height (bush_height 1 0) 4 = 64) :
  bush_height (bush_height 1 0) 2 = 4 :=
by
  sorry

#check bush_height_after_two_years

end bush_height_after_two_years_l2768_276869


namespace complex_power_equality_l2768_276898

theorem complex_power_equality : (3 * Complex.cos (π / 6) + 3 * Complex.I * Complex.sin (π / 6)) ^ 4 = -40.5 + 40.5 * Complex.I * Real.sqrt 3 := by
  sorry

end complex_power_equality_l2768_276898


namespace no_real_solution_l2768_276871

theorem no_real_solution :
  ¬∃ x : ℝ, Real.sqrt (x + 9) - Real.sqrt (x - 2) + 2 = 0 := by
sorry

end no_real_solution_l2768_276871


namespace f_not_odd_nor_even_f_minimum_value_l2768_276893

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 + |x - 2| - 1

-- Theorem for the parity of f(x)
theorem f_not_odd_nor_even :
  ¬(∀ x, f x = f (-x)) ∧ ¬(∀ x, f x = -f (-x)) :=
sorry

-- Theorem for the minimum value of f(x)
theorem f_minimum_value :
  ∀ x, f x ≥ 3 ∧ ∃ y, f y = 3 :=
sorry

end f_not_odd_nor_even_f_minimum_value_l2768_276893


namespace taxi_charge_calculation_l2768_276879

/-- Taxi service charge calculation -/
theorem taxi_charge_calculation 
  (initial_fee : ℝ) 
  (total_charge : ℝ) 
  (trip_distance : ℝ) 
  (segment_length : ℝ) : 
  initial_fee = 2.25 →
  total_charge = 4.5 →
  trip_distance = 3.6 →
  segment_length = 2/5 →
  (total_charge - initial_fee) / (trip_distance / segment_length) = 0.25 := by
  sorry

end taxi_charge_calculation_l2768_276879


namespace homework_probability_l2768_276804

theorem homework_probability (p : ℚ) (h : p = 5 / 9) :
  1 - p = 4 / 9 := by
  sorry

end homework_probability_l2768_276804


namespace third_month_sale_l2768_276866

theorem third_month_sale
  (average : ℕ)
  (month1 month2 month4 month5 month6 : ℕ)
  (h1 : average = 6800)
  (h2 : month1 = 6435)
  (h3 : month2 = 6927)
  (h4 : month4 = 7230)
  (h5 : month5 = 6562)
  (h6 : month6 = 6791)
  : ∃ month3 : ℕ, 
    month3 = 6855 ∧ 
    (month1 + month2 + month3 + month4 + month5 + month6) / 6 = average :=
by sorry

end third_month_sale_l2768_276866


namespace tree_height_after_two_years_l2768_276830

/-- The height of a tree after n years, given its initial height and growth rate -/
def tree_height (initial_height : ℝ) (growth_rate : ℝ) (n : ℕ) : ℝ :=
  initial_height * growth_rate ^ n

/-- Theorem: A tree that triples its height every year and reaches 243 feet after 5 years
    will be 9 feet tall after 2 years -/
theorem tree_height_after_two_years
  (h1 : ∃ initial_height : ℝ, tree_height initial_height 3 5 = 243)
  (h2 : ∀ n : ℕ, tree_height initial_height 3 (n + 1) = 3 * tree_height initial_height 3 n) :
  tree_height initial_height 3 2 = 9 :=
by
  sorry

end tree_height_after_two_years_l2768_276830


namespace recreation_spending_ratio_l2768_276897

theorem recreation_spending_ratio : 
  ∀ (last_week_wages : ℝ),
  last_week_wages > 0 →
  let last_week_recreation := 0.20 * last_week_wages
  let this_week_wages := 0.80 * last_week_wages
  let this_week_recreation := 0.40 * this_week_wages
  (this_week_recreation / last_week_recreation) * 100 = 160 :=
by
  sorry

end recreation_spending_ratio_l2768_276897


namespace triangle_properties_l2768_276821

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC with angles A, B, C and opposite sides a, b, c
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  -- B is an obtuse angle
  π/2 < B ∧ B < π ∧
  -- √3a = 2b sin A
  Real.sqrt 3 * a = 2 * b * Real.sin A →
  -- 1. B = 2π/3
  B = 2 * π / 3 ∧
  -- 2. If the area is 15√3/4 and b = 7, then a + c = 8
  (1/2 * a * c * Real.sin B = 15 * Real.sqrt 3 / 4 ∧ b = 7 → a + c = 8) ∧
  -- 3. If b = 6, the maximum area is 3√3
  (b = 6 → ∀ (a' c' : ℝ), 1/2 * a' * c' * Real.sin B ≤ 3 * Real.sqrt 3) :=
by sorry

end triangle_properties_l2768_276821


namespace small_cube_side_length_l2768_276845

theorem small_cube_side_length (large_cube_side : ℝ) (num_small_cubes : ℕ) (small_cube_side : ℝ) :
  large_cube_side = 1 →
  num_small_cubes = 1000 →
  large_cube_side ^ 3 = num_small_cubes * small_cube_side ^ 3 →
  small_cube_side = 0.1 := by
sorry

end small_cube_side_length_l2768_276845


namespace felix_weight_ratio_l2768_276865

/-- The weight ratio of Felix's brother to Felix -/
theorem felix_weight_ratio :
  let felix_lift_ratio : ℝ := 1.5
  let brother_lift_ratio : ℝ := 3
  let felix_lift_weight : ℝ := 150
  let brother_lift_weight : ℝ := 600
  let felix_weight := felix_lift_weight / felix_lift_ratio
  let brother_weight := brother_lift_weight / brother_lift_ratio
  brother_weight / felix_weight = 2 := by
sorry


end felix_weight_ratio_l2768_276865


namespace quadratic_term_elimination_l2768_276828

/-- The polynomial in question -/
def polynomial (x m : ℝ) : ℝ := 3*x^2 - 10 - 2*x - 4*x^2 + m*x^2

/-- The coefficient of x^2 in the polynomial -/
def x_squared_coefficient (m : ℝ) : ℝ := 3 - 4 + m

theorem quadratic_term_elimination :
  ∃ (m : ℝ), x_squared_coefficient m = 0 ∧ m = 1 := by sorry

end quadratic_term_elimination_l2768_276828


namespace initial_amount_theorem_l2768_276853

def poster_cost : ℕ := 5
def notebook_cost : ℕ := 4
def bookmark_cost : ℕ := 2

def num_posters : ℕ := 2
def num_notebooks : ℕ := 3
def num_bookmarks : ℕ := 2

def leftover : ℕ := 14

def total_cost : ℕ := num_posters * poster_cost + num_notebooks * notebook_cost + num_bookmarks * bookmark_cost

theorem initial_amount_theorem : total_cost + leftover = 40 := by
  sorry

end initial_amount_theorem_l2768_276853


namespace no_linear_term_condition_l2768_276842

theorem no_linear_term_condition (m : ℝ) : 
  (∀ x : ℝ, ∃ a b : ℝ, (x + m) * (x - 4) = a * x^2 + b) ↔ m = 4 := by
  sorry

end no_linear_term_condition_l2768_276842


namespace area_equality_function_unique_l2768_276863

/-- A function satisfying the given area equality property -/
def AreaEqualityFunction (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ * f x₂ = (x₂ - x₁) * (f x₁ + f x₂)

theorem area_equality_function_unique
  (f : ℝ → ℝ)
  (h₁ : ∀ x, x > 0 → f x > 0)
  (h₂ : AreaEqualityFunction f)
  (h₃ : f 1 = 4) :
  (∀ x, x > 0 → f x = 4 / x) ∧ f 4 = 1 := by
sorry

end area_equality_function_unique_l2768_276863


namespace strawberry_distribution_l2768_276831

theorem strawberry_distribution (num_girls : ℕ) (strawberries_per_girl : ℕ) 
  (h1 : num_girls = 8) (h2 : strawberries_per_girl = 6) :
  num_girls * strawberries_per_girl = 48 := by
  sorry

end strawberry_distribution_l2768_276831


namespace rectangular_box_volume_l2768_276872

theorem rectangular_box_volume (a b c : ℝ) 
  (h1 : a * b = 48) 
  (h2 : b * c = 20) 
  (h3 : c * a = 15) : 
  a * b * c = 120 := by
sorry

end rectangular_box_volume_l2768_276872


namespace johns_donation_is_260_average_increase_70_percent_new_average_is_85_five_initial_contributions_l2768_276812

/-- The size of John's donation to the charity fund -/
def johns_donation (initial_average : ℝ) (num_initial_contributions : ℕ) : ℝ :=
  let new_average := 85
  let num_total_contributions := num_initial_contributions + 1
  let total_initial_amount := initial_average * num_initial_contributions
  new_average * num_total_contributions - total_initial_amount

/-- Proof that John's donation is $260 given the conditions -/
theorem johns_donation_is_260 :
  let initial_average := 50
  let num_initial_contributions := 5
  johns_donation initial_average num_initial_contributions = 260 :=
by sorry

/-- The average contribution size increases by 70% after John's donation -/
theorem average_increase_70_percent (initial_average : ℝ) (num_initial_contributions : ℕ) :
  let new_average := 85
  new_average = initial_average * 1.7 :=
by sorry

/-- The new average contribution size is $85 per person -/
theorem new_average_is_85 (initial_average : ℝ) (num_initial_contributions : ℕ) :
  let new_average := 85
  let num_total_contributions := num_initial_contributions + 1
  let total_amount := initial_average * num_initial_contributions + johns_donation initial_average num_initial_contributions
  total_amount / num_total_contributions = new_average :=
by sorry

/-- There were 5 other contributions made before John's -/
theorem five_initial_contributions :
  let num_initial_contributions := 5
  num_initial_contributions = 5 :=
by sorry

end johns_donation_is_260_average_increase_70_percent_new_average_is_85_five_initial_contributions_l2768_276812


namespace three_conical_planet_models_l2768_276859

/-- Represents a model of a conical planet --/
structure ConicalPlanetModel where
  /-- The type of coordinate lines in the model --/
  CoordinateLine : Type
  /-- Predicate for whether two coordinate lines intersect --/
  intersects : CoordinateLine → CoordinateLine → Prop
  /-- Predicate for whether a coordinate line self-intersects --/
  self_intersects : CoordinateLine → Prop
  /-- Predicate for whether the constant direction principle holds --/
  constant_direction : Prop

/-- Cylindrical projection model --/
def cylindrical_model : ConicalPlanetModel := sorry

/-- Traditional conical projection model --/
def conical_model : ConicalPlanetModel := sorry

/-- Hybrid model --/
def hybrid_model : ConicalPlanetModel := sorry

/-- Properties of the hybrid model --/
axiom hybrid_model_properties :
  ∀ (l1 l2 : hybrid_model.CoordinateLine),
    l1 ≠ l2 → (hybrid_model.intersects l1 l2 ∧ hybrid_model.intersects l2 l1) ∧
    hybrid_model.self_intersects l1 ∧
    hybrid_model.constant_direction

/-- Theorem stating the existence of three distinct conical planet models --/
theorem three_conical_planet_models :
  ∃ (m1 m2 m3 : ConicalPlanetModel),
    m1 ≠ m2 ∧ m2 ≠ m3 ∧ m1 ≠ m3 ∧
    (m1 = cylindrical_model ∨ m1 = conical_model ∨ m1 = hybrid_model) ∧
    (m2 = cylindrical_model ∨ m2 = conical_model ∨ m2 = hybrid_model) ∧
    (m3 = cylindrical_model ∨ m3 = conical_model ∨ m3 = hybrid_model) := by
  sorry

end three_conical_planet_models_l2768_276859


namespace arithmetic_progression_rth_term_l2768_276883

/-- Definition of the sum function for the arithmetic progression -/
def S (n : ℕ) : ℝ := 4 * n + 5 * n^2

/-- The rth term of the arithmetic progression -/
def a (r : ℕ) : ℝ := 10 * r - 1

/-- Theorem stating that a(r) is the rth term of the arithmetic progression -/
theorem arithmetic_progression_rth_term (r : ℕ) :
  a r = S r - S (r - 1) :=
by sorry

end arithmetic_progression_rth_term_l2768_276883


namespace final_sum_theorem_l2768_276888

theorem final_sum_theorem (a b S : ℝ) (h : a + b = S) :
  3 * (a + 4) + 3 * (b + 4) = 3 * S + 24 := by
  sorry

end final_sum_theorem_l2768_276888


namespace problem_statement_l2768_276844

theorem problem_statement (a b : ℝ) (ha : a > 0) (hcond : Real.exp a + Real.log b = 1) :
  (a + Real.log b < 0) ∧ (Real.exp a + b > 2) ∧ (a + b > 1) := by
  sorry

end problem_statement_l2768_276844


namespace complex_product_equals_one_l2768_276885

theorem complex_product_equals_one (x : ℂ) (h : x = Complex.exp (Complex.I * Real.pi / 7)) :
  (x^2 + x^4) * (x^4 + x^8) * (x^6 + x^12) * (x^8 + x^16) * (x^10 + x^20) * (x^12 + x^24) = 1 := by
  sorry

end complex_product_equals_one_l2768_276885


namespace fraction_evaluation_l2768_276814

theorem fraction_evaluation : (4 * Nat.factorial 7 + 28 * Nat.factorial 6) / Nat.factorial 8 = 1 := by
  sorry

end fraction_evaluation_l2768_276814


namespace total_distance_traveled_l2768_276856

/-- Calculates the total distance traveled given walking and running speeds and durations, with a break. -/
theorem total_distance_traveled
  (total_time : ℝ)
  (walking_time : ℝ)
  (walking_speed : ℝ)
  (running_time : ℝ)
  (running_speed : ℝ)
  (break_time : ℝ)
  (h1 : total_time = 2)
  (h2 : walking_time = 1)
  (h3 : walking_speed = 3.5)
  (h4 : running_time = 0.75)
  (h5 : running_speed = 8)
  (h6 : break_time = 0.25)
  (h7 : total_time = walking_time + running_time + break_time) :
  walking_time * walking_speed + running_time * running_speed = 9.5 := by
  sorry


end total_distance_traveled_l2768_276856


namespace arithmetic_geometric_seq_l2768_276882

/-- An arithmetic sequence with common difference 3 -/
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + 3

/-- a_1, a_3, and a_4 form a geometric sequence -/
def geometric_subseq (a : ℕ → ℤ) : Prop :=
  (a 3) ^ 2 = (a 1) * (a 4)

theorem arithmetic_geometric_seq (a : ℕ → ℤ) 
  (h1 : arithmetic_seq a) 
  (h2 : geometric_subseq a) : 
  a 2 = -9 := by
sorry

end arithmetic_geometric_seq_l2768_276882


namespace july_production_l2768_276855

/-- Calculates the mask production after a given number of months, 
    starting from an initial production and doubling each month. -/
def maskProduction (initialProduction : ℕ) (months : ℕ) : ℕ :=
  initialProduction * 2^months

/-- Theorem stating that the mask production in July (4 months after March) 
    is 48000, given an initial production of 3000 in March. -/
theorem july_production : maskProduction 3000 4 = 48000 := by
  sorry

end july_production_l2768_276855


namespace arithmetic_sequence_sum_l2768_276895

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 80 -/
def SumCondition (a : ℕ → ℝ) : Prop :=
  a 2 + a 4 + a 6 + a 8 + a 10 = 80

theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h1 : ArithmeticSequence a)
  (h2 : SumCondition a) :
  a 5 + (1/4) * a 10 = 20 := by
  sorry

end arithmetic_sequence_sum_l2768_276895


namespace subset_intersection_bound_l2768_276889

theorem subset_intersection_bound (m n k : ℕ) (F : Fin k → Finset (Fin m)) :
  m ≥ n →
  n > 1 →
  (∀ i, (F i).card = n) →
  (∀ i j, i < j → (F i ∩ F j).card ≤ 1) →
  k ≤ m * (m - 1) / (n * (n - 1)) :=
by sorry

end subset_intersection_bound_l2768_276889
