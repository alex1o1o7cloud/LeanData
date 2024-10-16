import Mathlib

namespace NUMINAMATH_CALUDE_davids_english_marks_l3791_379169

/-- Calculates the marks in English given the marks in other subjects and the average -/
def marks_in_english (math physics chemistry biology : ℕ) (average : ℚ) : ℚ :=
  5 * average - (math + physics + chemistry + biology)

/-- Proves that David's marks in English are 72 -/
theorem davids_english_marks :
  marks_in_english 60 35 62 84 (62.6) = 72 := by
  sorry

end NUMINAMATH_CALUDE_davids_english_marks_l3791_379169


namespace NUMINAMATH_CALUDE_green_balloons_l3791_379195

theorem green_balloons (total : ℕ) (red : ℕ) (green : ℕ) : 
  total = 17 → red = 8 → green = total - red → green = 9 := by sorry

end NUMINAMATH_CALUDE_green_balloons_l3791_379195


namespace NUMINAMATH_CALUDE_new_student_weight_l3791_379182

theorem new_student_weight
  (n : ℕ)
  (initial_avg : ℝ)
  (new_avg : ℝ)
  (h1 : n = 29)
  (h2 : initial_avg = 28)
  (h3 : new_avg = 27.3) :
  (n + 1) * new_avg - n * initial_avg = 7 :=
by sorry

end NUMINAMATH_CALUDE_new_student_weight_l3791_379182


namespace NUMINAMATH_CALUDE_lipstick_ratio_l3791_379129

/-- Given information about students wearing lipstick, prove the ratio of red to colored lipstick wearers --/
theorem lipstick_ratio (total_students : ℕ) (blue_lipstick : ℕ) 
  (h1 : total_students = 200)
  (h2 : blue_lipstick = 5)
  (h3 : 2 * (total_students / 2) = total_students)  -- Half of students wore colored lipstick
  (h4 : 5 * blue_lipstick = red_lipstick) :  -- One-fifth as many blue as red
  (red_lipstick : ℕ) → (red_lipstick : ℚ) / (total_students / 2 : ℚ) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_lipstick_ratio_l3791_379129


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3791_379132

theorem quadratic_function_properties (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 + a*x + b
  (∀ x ∈ Set.Icc 0 1, f x ∈ Set.Icc 1 3 → a = 1 ∧ b = 1) ∧
  ((a = 0 ∧ b = 0) ∨ (a = -2 ∧ b = 1) → ∀ x ∈ Set.Icc 0 1, f x ∈ Set.Icc 0 1) ∧
  (∀ x, |x| ≥ 2 → f x ≥ 0) ∧
  (∀ x ∈ Set.Ioc 2 3, f x ≤ 1) ∧
  (f 3 = 1) →
  (32 : ℝ) ≤ a^2 + b^2 ∧ a^2 + b^2 ≤ 74 :=
by sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l3791_379132


namespace NUMINAMATH_CALUDE_cubic_root_sum_squares_l3791_379110

theorem cubic_root_sum_squares (a b c : ℂ) : 
  (a^3 + 3*a^2 - 10*a + 5 = 0) →
  (b^3 + 3*b^2 - 10*b + 5 = 0) →
  (c^3 + 3*c^2 - 10*c + 5 = 0) →
  a^2*b^2 + b^2*c^2 + c^2*a^2 = 70 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_squares_l3791_379110


namespace NUMINAMATH_CALUDE_five_volunteers_four_events_l3791_379162

/-- The number of ways to allocate volunteers to events --/
def allocationSchemes (volunteers : ℕ) (events : ℕ) : ℕ :=
  (volunteers.choose 2) * (events.factorial)

/-- Theorem stating the number of allocation schemes for 5 volunteers and 4 events --/
theorem five_volunteers_four_events :
  allocationSchemes 5 4 = 240 := by
  sorry

#eval allocationSchemes 5 4

end NUMINAMATH_CALUDE_five_volunteers_four_events_l3791_379162


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3791_379121

theorem arithmetic_sequence_problem (x : ℚ) : 
  let a₁ : ℚ := 1/3
  let a₂ : ℚ := x - 2
  let a₃ : ℚ := 4*x
  (a₂ - a₁ = a₃ - a₂) → x = -13/6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3791_379121


namespace NUMINAMATH_CALUDE_ten_thousand_equals_10000_l3791_379197

theorem ten_thousand_equals_10000 : (10 * 1000 : ℕ) = 10000 := by
  sorry

end NUMINAMATH_CALUDE_ten_thousand_equals_10000_l3791_379197


namespace NUMINAMATH_CALUDE_courtyard_length_l3791_379143

theorem courtyard_length 
  (breadth : ℝ) 
  (brick_length : ℝ) 
  (brick_width : ℝ) 
  (num_bricks : ℝ) 
  (h1 : breadth = 12)
  (h2 : brick_length = 0.15)
  (h3 : brick_width = 0.13)
  (h4 : num_bricks = 11076.923076923076) :
  (num_bricks * brick_length * brick_width) / breadth = 18 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_length_l3791_379143


namespace NUMINAMATH_CALUDE_all_propositions_false_l3791_379105

-- Define the correlation coefficient
def correlation_coefficient : ℝ → ℝ := sorry

-- Define the degree of linear correlation
def linear_correlation_degree : ℝ → ℝ := sorry

-- Define the cubic function
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + b*x + a^2

-- Define what it means for f to have an extreme value at x = -1
def has_extreme_value_at_neg_one (a b : ℝ) : Prop :=
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), 0 < |x + 1| ∧ |x + 1| < ε →
    (f a b (-1) - f a b x) * (f a b (-1) - f a b (-1 - (x + 1))) > 0

theorem all_propositions_false :
  -- Proposition 1
  (∀ r₁ r₂ : ℝ, |r₁| < |r₂| → linear_correlation_degree r₁ < linear_correlation_degree r₂) ∧
  -- Proposition 2
  (¬(∃ x : ℝ, x^2 + x + 1 < 0) ↔ ∀ x : ℝ, x^2 + x + 1 > 0) ∧
  -- Proposition 3
  (∀ p q : Prop, (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q)) ∧
  -- Proposition 4
  (∀ a b : ℝ, has_extreme_value_at_neg_one a b → (a = 1 ∧ b = 9))
  → False := by sorry

end NUMINAMATH_CALUDE_all_propositions_false_l3791_379105


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3791_379100

-- Define the quadratic function f(x)
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

-- Define g(x) in terms of f(x) and m
def g (m : ℝ) (x : ℝ) : ℝ := m * f x + 1

-- Theorem statement
theorem quadratic_function_properties :
  (∀ x : ℝ, f x ≥ -4) ∧
  (f (-2) = -3) ∧
  (f 0 = -3) ∧
  (∀ m : ℝ, m < 0 → ∃! x : ℝ, x ≤ 1 ∧ g m x = 0) ∧
  (∀ m : ℝ, m > 0 →
    (m ≤ 8/7 → (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3/2 → |g m x| ≤ 9*m/4 + 1)) ∧
    (m > 8/7 → (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3/2 → |g m x| ≤ 4*m - 1))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3791_379100


namespace NUMINAMATH_CALUDE_equation_solution_l3791_379193

theorem equation_solution (a : ℚ) : 
  (∀ x : ℚ, (2*a*x + 3) / (a - x) = 3/4 ↔ x = 1) → a = -3 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3791_379193


namespace NUMINAMATH_CALUDE_max_values_l3791_379185

theorem max_values (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b^2 = 1) :
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y^2 = 1 ∧ b * Real.sqrt a ≤ x * Real.sqrt y) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x + y^2 = 1 → b * Real.sqrt a ≤ 1/2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y^2 = 1 ∧ Real.sqrt a + b ≤ Real.sqrt x + y) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x + y^2 = 1 → Real.sqrt a + b ≤ Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_max_values_l3791_379185


namespace NUMINAMATH_CALUDE_fourth_number_unit_digit_l3791_379122

def unit_digit (n : ℕ) : ℕ := n % 10

def product_unit_digit (a b c d : ℕ) : ℕ :=
  unit_digit (unit_digit a * unit_digit b * unit_digit c * unit_digit d)

theorem fourth_number_unit_digit :
  ∃ (x : ℕ), product_unit_digit 624 708 463 x = 8 ∧ unit_digit x = 3 :=
by sorry

end NUMINAMATH_CALUDE_fourth_number_unit_digit_l3791_379122


namespace NUMINAMATH_CALUDE_parabola_x_intercepts_l3791_379144

theorem parabola_x_intercepts :
  let f (x : ℝ) := 3 * x^2 + 5 * x - 8
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ∧
  (∀ (x₁ x₂ x₃ : ℝ), f x₁ = 0 → f x₂ = 0 → f x₃ = 0 → x₁ = x₂ ∨ x₁ = x₃ ∨ x₂ = x₃) := by
  sorry

end NUMINAMATH_CALUDE_parabola_x_intercepts_l3791_379144


namespace NUMINAMATH_CALUDE_no_real_solutions_exponential_equation_l3791_379192

theorem no_real_solutions_exponential_equation :
  ∀ x : ℝ, (2 : ℝ)^(5*x+2) * (4 : ℝ)^(2*x+4) ≠ (8 : ℝ)^(3*x+7) := by
sorry

end NUMINAMATH_CALUDE_no_real_solutions_exponential_equation_l3791_379192


namespace NUMINAMATH_CALUDE_machines_count_l3791_379147

/-- Given that n machines produce x units in 6 days and 12 machines produce 3x units in 6 days,
    where all machines work at an identical constant rate, prove that n = 4. -/
theorem machines_count (n : ℕ) (x : ℝ) (h1 : x > 0) :
  (n * x / 6 = x / 6) →
  (12 * (3 * x) / 6 = 3 * x / 6) →
  (n * x / (6 * n) = 12 * (3 * x) / (6 * 12)) →
  n = 4 :=
sorry

end NUMINAMATH_CALUDE_machines_count_l3791_379147


namespace NUMINAMATH_CALUDE_initial_value_problem_l3791_379159

theorem initial_value_problem (x : ℤ) : x + 335 = 456 * (x + 335) / 456 → x = 121 :=
by sorry

end NUMINAMATH_CALUDE_initial_value_problem_l3791_379159


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3791_379117

theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, (0 < x ∧ x < 2) ↔ ((m - 1) * x < Real.sqrt (4 * x) - x^2)) → 
  m = 1 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3791_379117


namespace NUMINAMATH_CALUDE_max_value_x_5_minus_4x_l3791_379103

theorem max_value_x_5_minus_4x (x : ℝ) (h1 : 0 < x) (h2 : x < 5/4) :
  x * (5 - 4*x) ≤ 25/16 := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_5_minus_4x_l3791_379103


namespace NUMINAMATH_CALUDE_min_packs_needed_l3791_379107

/-- Represents the number of cans in each pack type -/
def pack_sizes : Fin 3 → ℕ
  | 0 => 8
  | 1 => 15
  | 2 => 18

/-- The total number of cans needed -/
def total_cans : ℕ := 95

/-- The maximum number of packs allowed for each type -/
def max_packs : ℕ := 4

/-- A function to calculate the total number of cans from a given combination of packs -/
def total_from_packs (x y z : ℕ) : ℕ :=
  x * pack_sizes 0 + y * pack_sizes 1 + z * pack_sizes 2

/-- The main theorem to prove -/
theorem min_packs_needed :
  ∃ (x y z : ℕ),
    x ≤ max_packs ∧ y ≤ max_packs ∧ z ≤ max_packs ∧
    total_from_packs x y z = total_cans ∧
    x + y + z = 6 ∧
    (∀ (a b c : ℕ),
      a ≤ max_packs → b ≤ max_packs → c ≤ max_packs →
      total_from_packs a b c = total_cans →
      a + b + c ≥ 6) :=
sorry

end NUMINAMATH_CALUDE_min_packs_needed_l3791_379107


namespace NUMINAMATH_CALUDE_largest_quotient_is_30_l3791_379153

def S : Set Int := {-30, -5, -1, 0, 3, 9}

theorem largest_quotient_is_30 : 
  ∀ a b : Int, a ∈ S → b ∈ S → b ≠ 0 → a / b ≤ 30 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_quotient_is_30_l3791_379153


namespace NUMINAMATH_CALUDE_shortest_path_in_sqrt2_octahedron_l3791_379164

/-- A regular octahedron -/
structure RegularOctahedron where
  edgeLength : ℝ
  edgeLength_pos : edgeLength > 0

/-- The shortest path between midpoints of non-adjacent edges -/
def shortestPathBetweenMidpoints (o : RegularOctahedron) : ℝ :=
  sorry

/-- Theorem: In a regular octahedron with edge length √2, the shortest path
    between midpoints of non-adjacent edges is √2 -/
theorem shortest_path_in_sqrt2_octahedron :
  let o : RegularOctahedron := ⟨ Real.sqrt 2, sorry ⟩
  shortestPathBetweenMidpoints o = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_shortest_path_in_sqrt2_octahedron_l3791_379164


namespace NUMINAMATH_CALUDE_lattice_points_on_segment_l3791_379155

/-- The number of lattice points on a line segment with given endpoints -/
def latticePointCount (x1 y1 x2 y2 : Int) : Nat :=
  sorry

/-- Theorem stating that the number of lattice points on the given line segment is 6 -/
theorem lattice_points_on_segment : latticePointCount 5 26 40 146 = 6 := by
  sorry

end NUMINAMATH_CALUDE_lattice_points_on_segment_l3791_379155


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l3791_379137

def reciprocal (x : ℚ) : ℚ := 1 / x

theorem reciprocal_of_negative_fraction :
  reciprocal (-5/4) = -4/5 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l3791_379137


namespace NUMINAMATH_CALUDE_cosine_sine_identity_l3791_379120

theorem cosine_sine_identity (α : Real) :
  (∃ (x y : Real), x = 2 ∧ y = 1 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.cos α ^ 2 + Real.sin (2 * α) = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_identity_l3791_379120


namespace NUMINAMATH_CALUDE_cos_2017pi_minus_2alpha_l3791_379134

theorem cos_2017pi_minus_2alpha (α : Real) 
  (h1 : π/2 < α ∧ α < π) -- α is in the second quadrant
  (h2 : Real.sin α + Real.cos α = Real.sqrt 3 / 2) :
  Real.cos (2017 * π - 2 * α) = 1/2 := by sorry

end NUMINAMATH_CALUDE_cos_2017pi_minus_2alpha_l3791_379134


namespace NUMINAMATH_CALUDE_initial_speed_satisfies_conditions_l3791_379111

/-- Represents the initial speed of the car in km/h -/
def V : ℝ := 60

/-- Represents the distance from A to B in km -/
def distance : ℝ := 300

/-- Represents the increase in speed on the return journey in km/h -/
def speed_increase : ℝ := 16

/-- Represents the time after which the speed was increased on the return journey in hours -/
def time_before_increase : ℝ := 1.2

/-- Represents the time difference between the outward and return journeys in hours -/
def time_difference : ℝ := 0.8

/-- Theorem stating that the initial speed satisfies the given conditions -/
theorem initial_speed_satisfies_conditions :
  (distance / V - time_difference = 
   time_before_increase + (distance - V * time_before_increase) / (V + speed_increase)) := by
  sorry

end NUMINAMATH_CALUDE_initial_speed_satisfies_conditions_l3791_379111


namespace NUMINAMATH_CALUDE_gcd_3465_10780_l3791_379157

theorem gcd_3465_10780 : Nat.gcd 3465 10780 = 385 := by
  sorry

end NUMINAMATH_CALUDE_gcd_3465_10780_l3791_379157


namespace NUMINAMATH_CALUDE_parabola_shift_correct_l3791_379140

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola y = 4x^2 -/
def original_parabola : Parabola := { a := 4, b := 0, c := 0 }

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h
  , c := p.a * h^2 + p.c + v }

/-- The resulting parabola after shifting -/
def shifted_parabola : Parabola := shift_parabola original_parabola 9 6

theorem parabola_shift_correct :
  shifted_parabola = { a := 4, b := -72, c := 330 } := by sorry

end NUMINAMATH_CALUDE_parabola_shift_correct_l3791_379140


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l3791_379198

theorem quadratic_equation_unique_solution (a : ℝ) (h1 : a ≠ 0) 
  (h2 : ∃! x, a * x^2 + 12 * x + 9 = 0) :
  ∃ x, a * x^2 + 12 * x + 9 = 0 ∧ x = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l3791_379198


namespace NUMINAMATH_CALUDE_equation_solution_range_l3791_379131

theorem equation_solution_range (x m : ℝ) : 
  (2 * x + 4 = m - x) → (x < 0) → (m < 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_range_l3791_379131


namespace NUMINAMATH_CALUDE_smallest_x_value_l3791_379184

theorem smallest_x_value (x : ℚ) : 
  (7 * (8 * x^2 + 8 * x + 11) = x * (8 * x - 45)) → x ≥ -7/3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l3791_379184


namespace NUMINAMATH_CALUDE_right_isosceles_triangle_circle_segment_area_l3791_379126

theorem right_isosceles_triangle_circle_segment_area :
  let hypotenuse : ℝ := 10
  let radius : ℝ := hypotenuse / 2
  let sector_angle : ℝ := 45 -- in degrees
  let sector_area : ℝ := (sector_angle / 360) * π * radius^2
  let triangle_area : ℝ := (1 / 2) * radius^2
  let shaded_area : ℝ := sector_area - triangle_area
  let a : ℝ := 25
  let b : ℝ := 50
  let c : ℝ := 1
  (shaded_area = a * π - b * Real.sqrt c) ∧ (a + b + c = 76) := by
  sorry

end NUMINAMATH_CALUDE_right_isosceles_triangle_circle_segment_area_l3791_379126


namespace NUMINAMATH_CALUDE_fourth_term_is_nine_l3791_379174

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- The theorem stating that the 4th term of the arithmetic sequence is 9 -/
theorem fourth_term_is_nine (seq : ArithmeticSequence) 
    (first_term : seq.a 1 = 3)
    (sum_three : seq.S 3 = 15) : 
  seq.a 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_nine_l3791_379174


namespace NUMINAMATH_CALUDE_harmonic_sum_denominator_not_div_by_five_l3791_379151

/-- The sum of reciprocals from 1 to n -/
def harmonic_sum (n : ℕ+) : ℚ :=
  Finset.sum (Finset.range n) (λ m => 1 / (m + 1 : ℚ))

/-- The set of positive integers n for which 5 does not divide the denominator
    of the harmonic sum when expressed in lowest terms -/
def D : Set ℕ+ :=
  {n | ¬ (5 ∣ (harmonic_sum n).den)}

/-- The theorem stating that D is exactly the given set -/
theorem harmonic_sum_denominator_not_div_by_five :
  D = {1, 2, 3, 4, 20, 21, 22, 23, 24, 100, 101, 102, 103, 104, 120, 121, 122, 123, 124} := by
  sorry


end NUMINAMATH_CALUDE_harmonic_sum_denominator_not_div_by_five_l3791_379151


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3791_379181

/-- The sum of the coordinates of the midpoint of a segment with endpoints (10, 3) and (4, -3) is 7. -/
theorem midpoint_coordinate_sum : 
  let p₁ : ℝ × ℝ := (10, 3)
  let p₂ : ℝ × ℝ := (4, -3)
  let midpoint := ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2)
  (midpoint.1 + midpoint.2 : ℝ) = 7 := by
sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3791_379181


namespace NUMINAMATH_CALUDE_root_difference_implies_k_value_l3791_379179

theorem root_difference_implies_k_value (k : ℝ) : 
  (∃ r s : ℝ, r^2 + k*r + 10 = 0 ∧ s^2 + k*s + 10 = 0 ∧ 
   (r+2)^2 - k*(r+2) + 10 = 0 ∧ (s+2)^2 - k*(s+2) + 10 = 0) → k = 2 :=
by sorry

end NUMINAMATH_CALUDE_root_difference_implies_k_value_l3791_379179


namespace NUMINAMATH_CALUDE_ending_number_is_989_l3791_379146

/-- A function that counts the number of integers between 0 and n (inclusive) 
    that do not contain the digit 1 in their decimal representation -/
def count_no_one (n : ℕ) : ℕ := sorry

/-- The theorem stating that 989 is the smallest positive integer n such that 
    there are exactly 728 integers between 0 and n (inclusive) that do not 
    contain the digit 1 -/
theorem ending_number_is_989 : 
  (∀ m : ℕ, m < 989 → count_no_one m < 728) ∧ count_no_one 989 = 728 :=
sorry

end NUMINAMATH_CALUDE_ending_number_is_989_l3791_379146


namespace NUMINAMATH_CALUDE_puppy_adoption_ratio_l3791_379145

theorem puppy_adoption_ratio :
  let first_week : ℕ := 20
  let second_week : ℕ := (2 * first_week) / 5
  let fourth_week : ℕ := first_week + 10
  let total_puppies : ℕ := 74
  let third_week : ℕ := total_puppies - (first_week + second_week + fourth_week)
  (third_week : ℚ) / second_week = 2 := by
  sorry

end NUMINAMATH_CALUDE_puppy_adoption_ratio_l3791_379145


namespace NUMINAMATH_CALUDE_slope_dividing_area_l3791_379133

-- Define the vertices of the L-shaped region
def vertices : List (ℝ × ℝ) := [(0, 0), (0, 4), (4, 4), (4, 2), (7, 2), (7, 0)]

-- Define the L-shaped region
def l_shape (x y : ℝ) : Prop :=
  (0 ≤ x ∧ x ≤ 7 ∧ 0 ≤ y ∧ y ≤ 4) ∧
  (x ≤ 4 ∨ y ≤ 2)

-- Define the area of the L-shaped region
def area_l_shape : ℝ := 22

-- Define a line through the origin
def line_through_origin (m : ℝ) (x y : ℝ) : Prop :=
  y = m * x

-- Define the area above the line
def area_above_line (m : ℝ) : ℝ := 11

-- Theorem: The slope of the line that divides the area in half is -0.375
theorem slope_dividing_area :
  ∃ (m : ℝ), m = -0.375 ∧
    area_above_line m = area_l_shape / 2 ∧
    ∀ (x y : ℝ), l_shape x y → line_through_origin m x y →
      (y ≥ m * x → area_above_line m ≥ area_l_shape / 2) ∧
      (y ≤ m * x → area_above_line m ≤ area_l_shape / 2) :=
by sorry

end NUMINAMATH_CALUDE_slope_dividing_area_l3791_379133


namespace NUMINAMATH_CALUDE_tan_double_alpha_l3791_379194

theorem tan_double_alpha (α : ℝ) (h : (2 * Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3) :
  Real.tan (2 * α) = -8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_alpha_l3791_379194


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3791_379112

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, Real.exp x > x^2) ↔ (∃ x : ℝ, Real.exp x ≤ x^2) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3791_379112


namespace NUMINAMATH_CALUDE_initial_number_of_people_l3791_379196

theorem initial_number_of_people (avg_weight_increase : ℝ) (weight_difference : ℝ) : 
  avg_weight_increase = 2.5 →
  weight_difference = 20 →
  avg_weight_increase * (weight_difference / avg_weight_increase) = weight_difference →
  (weight_difference / avg_weight_increase : ℝ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_of_people_l3791_379196


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3791_379130

theorem quadratic_inequality (x : ℝ) : x^2 - 8*x + 12 < 0 ↔ 2 < x ∧ x < 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3791_379130


namespace NUMINAMATH_CALUDE_sum_of_cumulative_sums_geometric_sequence_l3791_379180

/-- The sum of cumulative sums of a geometric sequence -/
theorem sum_of_cumulative_sums_geometric_sequence (a₁ q : ℝ) (h : |q| < 1) :
  ∃ (S : ℕ → ℝ), (∀ n, S n = a₁ * (1 - q^n) / (1 - q)) ∧
  (∑' n, S n) = a₁ / (1 - q)^2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cumulative_sums_geometric_sequence_l3791_379180


namespace NUMINAMATH_CALUDE_hot_day_price_correct_l3791_379135

/-- Represents the lemonade stand operation --/
structure LemonadeStand where
  totalDays : ℕ
  hotDays : ℕ
  cupsPerDay : ℕ
  costPerCup : ℚ
  totalProfit : ℚ
  hotDayPriceIncrease : ℚ

/-- Calculates the price of a cup on a hot day --/
def hotDayPrice (stand : LemonadeStand) : ℚ :=
  let regularPrice := (stand.totalProfit + stand.totalDays * stand.cupsPerDay * stand.costPerCup) /
    (stand.cupsPerDay * (stand.totalDays + stand.hotDays * stand.hotDayPriceIncrease))
  regularPrice * (1 + stand.hotDayPriceIncrease)

/-- Theorem stating that the hot day price is correct --/
theorem hot_day_price_correct (stand : LemonadeStand) : 
  stand.totalDays = 10 ∧ 
  stand.hotDays = 4 ∧ 
  stand.cupsPerDay = 32 ∧ 
  stand.costPerCup = 3/4 ∧ 
  stand.totalProfit = 200 ∧
  stand.hotDayPriceIncrease = 1/4 →
  hotDayPrice stand = 25/16 := by
  sorry

#eval hotDayPrice {
  totalDays := 10
  hotDays := 4
  cupsPerDay := 32
  costPerCup := 3/4
  totalProfit := 200
  hotDayPriceIncrease := 1/4
}

end NUMINAMATH_CALUDE_hot_day_price_correct_l3791_379135


namespace NUMINAMATH_CALUDE_icosahedron_edge_ratio_l3791_379170

/-- An icosahedron with edge length a -/
structure Icosahedron where
  a : ℝ
  a_pos : 0 < a

/-- A regular octahedron -/
structure RegularOctahedron where
  edge_length : ℝ
  edge_length_pos : 0 < edge_length

/-- Given two icosahedrons, this function returns true if six vertices 
    can be chosen from them to form a regular octahedron -/
def can_form_octahedron (i1 i2 : Icosahedron) : Prop := sorry

theorem icosahedron_edge_ratio 
  (i1 i2 : Icosahedron) 
  (h : can_form_octahedron i1 i2) : 
  i1.a / i2.a = (Real.sqrt 5 + 1) / 2 := by sorry

end NUMINAMATH_CALUDE_icosahedron_edge_ratio_l3791_379170


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l3791_379161

theorem quadratic_roots_relation (p q : ℝ) : 
  (∀ x, x^2 - p^2*x + p*q = 0 ↔ ∃ y, y^2 + p*y + q = 0 ∧ x = y + 1) →
  ((p = -1 ∧ q = -1) ∨ (p = 2 ∧ q = -1)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l3791_379161


namespace NUMINAMATH_CALUDE_marble_pairs_l3791_379160

-- Define the set of marbles
def Marble : Type := 
  Sum (Fin 1) (Sum (Fin 1) (Sum (Fin 1) (Sum (Fin 3) (Fin 2))))

-- Define the function to count distinct pairs
def countDistinctPairs (s : Finset Marble) : ℕ := sorry

-- State the theorem
theorem marble_pairs : 
  let s : Finset Marble := sorry
  countDistinctPairs s = 12 := by sorry

end NUMINAMATH_CALUDE_marble_pairs_l3791_379160


namespace NUMINAMATH_CALUDE_non_pine_trees_l3791_379125

theorem non_pine_trees (total : ℕ) (pine_percentage : ℚ) : 
  total = 350 → pine_percentage = 70 / 100 → 
  total - (total * pine_percentage).floor = 105 := by
  sorry

end NUMINAMATH_CALUDE_non_pine_trees_l3791_379125


namespace NUMINAMATH_CALUDE_merry_go_round_revolutions_l3791_379118

theorem merry_go_round_revolutions 
  (distance_A : ℝ) 
  (distance_B : ℝ) 
  (revolutions_A : ℝ) 
  (h1 : distance_A = 36) 
  (h2 : distance_B = 12) 
  (h3 : revolutions_A = 40) 
  (h4 : distance_A * revolutions_A = distance_B * revolutions_B) : 
  revolutions_B = 120 := by
  sorry

end NUMINAMATH_CALUDE_merry_go_round_revolutions_l3791_379118


namespace NUMINAMATH_CALUDE_four_integer_b_values_l3791_379168

/-- A function that checks if a given integer b results in integer roots for the quadratic equation x^2 + bx + 7b = 0 -/
def has_integer_roots (b : ℤ) : Prop :=
  ∃ p q : ℤ, p + q = -b ∧ p * q = 7 * b

/-- The theorem stating that there are exactly 4 integer values of b for which the quadratic equation x^2 + bx + 7b = 0 always has integer roots -/
theorem four_integer_b_values :
  ∃! (s : Finset ℤ), s.card = 4 ∧ ∀ b : ℤ, has_integer_roots b ↔ b ∈ s :=
sorry

end NUMINAMATH_CALUDE_four_integer_b_values_l3791_379168


namespace NUMINAMATH_CALUDE_number_of_children_l3791_379183

/-- Given a person with some children and money to distribute, prove the number of children. -/
theorem number_of_children (total_money : ℕ) (share_d_and_e : ℕ) (children : List String) : 
  total_money = 12000 → share_d_and_e = 4800 → 
  children = ["a", "b", "c", "d", "e"] → 
  children.length = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_of_children_l3791_379183


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3791_379186

theorem algebraic_expression_value (a b : ℝ) (h : a - b = 2) :
  a^3 - 2*a^2*b + a*b^2 - 4*a = 0 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3791_379186


namespace NUMINAMATH_CALUDE_special_sequence_characterization_l3791_379109

/-- A sequence of real numbers satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, a n ≤ a (n + 1)) ∧ 
  (∀ m n : ℕ, a (m^2 + n^2) = (a m)^2 + (a n)^2)

/-- The theorem stating the only possible sequences satisfying the conditions -/
theorem special_sequence_characterization (a : ℕ → ℝ) :
  SpecialSequence a →
  ((∀ n, a n = 0) ∨ (∀ n, a n = 1/2) ∨ (∀ n, a n = n)) :=
by sorry

end NUMINAMATH_CALUDE_special_sequence_characterization_l3791_379109


namespace NUMINAMATH_CALUDE_cos_plus_sin_value_l3791_379165

theorem cos_plus_sin_value (α : Real) (k : Real) :
  (∃ x y : Real, x * y = 1 ∧ 
    x^2 - k*x + k^2 - 3 = 0 ∧ 
    y^2 - k*y + k^2 - 3 = 0 ∧ 
    x = Real.tan α ∧ 
    y = 1 / Real.tan α) →
  3 * Real.pi < α ∧ α < 7/2 * Real.pi →
  Real.cos α + Real.sin α = -Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_cos_plus_sin_value_l3791_379165


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l3791_379148

theorem min_value_theorem (x : ℝ) : (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 9 / Real.sqrt 5 := by
  sorry

theorem min_value_achievable : ∃ x : ℝ, (x^2 + 9) / Real.sqrt (x^2 + 5) = 9 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l3791_379148


namespace NUMINAMATH_CALUDE_cans_needed_for_35_rooms_l3791_379173

/-- Represents the number of rooms that can be painted with the available paint -/
def initial_rooms : ℕ := 45

/-- Represents the number of paint cans lost -/
def lost_cans : ℕ := 5

/-- Represents the number of rooms that can be painted after losing some paint cans -/
def remaining_rooms : ℕ := 35

/-- Represents that each can must be used entirely (no partial cans) -/
def whole_cans_only : Prop := True

/-- Theorem stating that 18 cans are needed to paint 35 rooms given the conditions -/
theorem cans_needed_for_35_rooms : 
  ∃ (cans_per_room : ℚ),
    cans_per_room * (initial_rooms - remaining_rooms) = lost_cans ∧
    ∃ (cans_needed : ℕ),
      cans_needed = ⌈(remaining_rooms : ℚ) / cans_per_room⌉ ∧
      cans_needed = 18 :=
sorry

end NUMINAMATH_CALUDE_cans_needed_for_35_rooms_l3791_379173


namespace NUMINAMATH_CALUDE_max_sum_of_square_roots_l3791_379113

theorem max_sum_of_square_roots (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 7) :
  Real.sqrt (3 * x + 2) + Real.sqrt (3 * y + 2) + Real.sqrt (3 * z + 2) ≤ 9 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_square_roots_l3791_379113


namespace NUMINAMATH_CALUDE_sandy_shirt_cost_l3791_379166

/-- The amount Sandy spent on shorts -/
def shorts_cost : ℝ := 13.99

/-- The amount Sandy received for returning a jacket -/
def jacket_return : ℝ := 7.43

/-- The net amount Sandy spent on clothes -/
def net_spend : ℝ := 18.7

/-- The amount Sandy spent on the shirt -/
def shirt_cost : ℝ := net_spend + jacket_return - shorts_cost

theorem sandy_shirt_cost : shirt_cost = 12.14 := by sorry

end NUMINAMATH_CALUDE_sandy_shirt_cost_l3791_379166


namespace NUMINAMATH_CALUDE_max_value_4x_3y_l3791_379150

theorem max_value_4x_3y (x y : ℝ) : 
  x^2 + y^2 = 16*x + 8*y + 10 → 
  (4*x + 3*y ≤ (82.47 : ℝ) / 18) ∧ 
  ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 16*x₀ + 8*y₀ + 10 ∧ 4*x₀ + 3*y₀ = (82.47 : ℝ) / 18 :=
by sorry

end NUMINAMATH_CALUDE_max_value_4x_3y_l3791_379150


namespace NUMINAMATH_CALUDE_coeff_x4_when_sum_64_l3791_379172

-- Define the binomial coefficient function
def binomial_coeff (n k : ℕ) : ℕ := sorry

-- Define the function to calculate the sum of binomial coefficients
def sum_binomial_coeff (n : ℕ) : ℕ := sorry

-- Define the function to calculate the coefficient of x^4 in the expansion
def coeff_x4 (n : ℕ) : ℤ := sorry

-- Theorem statement
theorem coeff_x4_when_sum_64 (n : ℕ) :
  sum_binomial_coeff n = 64 → coeff_x4 n = -12 := by sorry

end NUMINAMATH_CALUDE_coeff_x4_when_sum_64_l3791_379172


namespace NUMINAMATH_CALUDE_f_and_g_properties_l3791_379177

-- Define the function f and its derivative
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Define g as the derivative of f
def g : ℝ → ℝ := f'

-- Axioms based on the problem conditions
axiom f_diff : ∀ x, HasDerivAt f (f' x) x
axiom f_even : ∀ x, f (3/2 - 2*x) = f (3/2 + 2*x)
axiom g_even : ∀ x, g (2 + x) = g (2 - x)

-- Theorem to prove
theorem f_and_g_properties :
  f (-1) = f 4 ∧ g (-1/2) = 0 :=
sorry

end NUMINAMATH_CALUDE_f_and_g_properties_l3791_379177


namespace NUMINAMATH_CALUDE_number_theory_statements_l3791_379116

theorem number_theory_statements :
  (∃ n : ℕ, 20 = 4 * n) ∧
  (∃ n : ℕ, 209 = 19 * n) ∧ ¬(∃ n : ℕ, 63 = 19 * n) ∧
  ¬(∃ n : ℕ, 75 = 12 * n) ∧ ¬(∃ n : ℕ, 29 = 12 * n) ∧
  (∃ n : ℕ, 33 = 11 * n) ∧ ¬(∃ n : ℕ, 64 = 11 * n) ∧
  (∃ n : ℕ, 180 = 9 * n) := by
sorry

end NUMINAMATH_CALUDE_number_theory_statements_l3791_379116


namespace NUMINAMATH_CALUDE_sphere_diameter_count_l3791_379114

theorem sphere_diameter_count (total_points : ℕ) (surface_percentage : ℚ) 
  (h1 : total_points = 39)
  (h2 : surface_percentage ≤ 72/100)
  : ∃ (surface_points : ℕ), 
    surface_points ≤ ⌊(surface_percentage * total_points)⌋ ∧ 
    (surface_points.choose 2) = 378 := by
  sorry

end NUMINAMATH_CALUDE_sphere_diameter_count_l3791_379114


namespace NUMINAMATH_CALUDE_exists_special_subset_l3791_379178

/-- Definition of arithmetic mean -/
def arithmetic_mean (S : Finset ℕ) : ℚ :=
  (S.sum id) / S.card

/-- Definition of perfect power -/
def is_perfect_power (n : ℚ) : Prop :=
  ∃ (a : ℚ) (k : ℕ), k > 1 ∧ n = a ^ k

/-- Main theorem -/
theorem exists_special_subset :
  ∃ (A : Finset ℕ), A.card = 2022 ∧
    ∀ (B : Finset ℕ), B ⊆ A →
      is_perfect_power (arithmetic_mean B) :=
sorry

end NUMINAMATH_CALUDE_exists_special_subset_l3791_379178


namespace NUMINAMATH_CALUDE_parabola_decreasing_right_of_axis_l3791_379163

-- Define the parabola function
def f (b c x : ℝ) : ℝ := -x^2 + b*x + c

-- State the theorem
theorem parabola_decreasing_right_of_axis (b c : ℝ) :
  (∀ x, f b c x = f b c (6 - x)) →  -- Axis of symmetry at x = 3
  ∀ x > 3, ∀ y > x, f b c y < f b c x :=
sorry

end NUMINAMATH_CALUDE_parabola_decreasing_right_of_axis_l3791_379163


namespace NUMINAMATH_CALUDE_evaluate_expression_l3791_379199

theorem evaluate_expression : (0.5^4 / 0.05^3) + 3 = 503 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3791_379199


namespace NUMINAMATH_CALUDE_triangle_side_values_l3791_379124

theorem triangle_side_values (A B C : Real) (a b c : Real) :
  c = Real.sqrt 3 →
  C = π / 3 →
  Real.sin B = 2 * Real.sin A →
  a ^ 2 + b ^ 2 - a * b = 3 →
  (a = 1 ∧ b = 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_values_l3791_379124


namespace NUMINAMATH_CALUDE_molecular_weight_4_moles_BaBr2_l3791_379187

-- Define atomic weights
def atomic_weight_Ba : ℝ := 137.33
def atomic_weight_Br : ℝ := 79.90

-- Define molecular weight of BaBr2
def molecular_weight_BaBr2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_Br

-- Define the number of moles
def moles : ℝ := 4

-- Theorem statement
theorem molecular_weight_4_moles_BaBr2 :
  moles * molecular_weight_BaBr2 = 1188.52 := by sorry

end NUMINAMATH_CALUDE_molecular_weight_4_moles_BaBr2_l3791_379187


namespace NUMINAMATH_CALUDE_trig_simplification_l3791_379127

theorem trig_simplification :
  (Real.cos (40 * π / 180)) / (Real.cos (25 * π / 180) * Real.sqrt (1 - Real.sin (40 * π / 180))) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l3791_379127


namespace NUMINAMATH_CALUDE_prob_exactly_one_red_ball_l3791_379158

def total_balls : ℕ := 5
def red_balls : ℕ := 2
def white_balls : ℕ := 3
def drawn_balls : ℕ := 2

theorem prob_exactly_one_red_ball : 
  (Nat.choose red_balls 1 * Nat.choose white_balls 1) / Nat.choose total_balls drawn_balls = 3 / 5 :=
sorry

end NUMINAMATH_CALUDE_prob_exactly_one_red_ball_l3791_379158


namespace NUMINAMATH_CALUDE_tetrahedron_inequality_l3791_379102

/-- Given a tetrahedron with product of opposite edges equal to 1,
    angles α, β, γ between opposite edges, and face circumradii R₁, R₂, R₃, R₄,
    prove that sin²α + sin²β + sin²γ ≥ 1/√(R₁R₂R₃R₄) -/
theorem tetrahedron_inequality
  (α β γ R₁ R₂ R₃ R₄ : ℝ)
  (h_positive : R₁ > 0 ∧ R₂ > 0 ∧ R₃ > 0 ∧ R₄ > 0)
  (h_product : ∀ (i j k l : Fin 4), i ≠ j ∧ k ≠ l ∧ i ≠ k ∧ j ≠ l → 
    ∃ (a_ij a_kl : ℝ), a_ij * a_kl = 1) :
  Real.sin α ^ 2 + Real.sin β ^ 2 + Real.sin γ ^ 2 ≥ 1 / Real.sqrt (R₁ * R₂ * R₃ * R₄) := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_inequality_l3791_379102


namespace NUMINAMATH_CALUDE_bucket_weight_l3791_379136

/-- Given a bucket with weight p when three-quarters full and weight q when one-third full,
    prove that its weight when full is (8p - 7q) / 5 -/
theorem bucket_weight (p q : ℝ) : ℝ :=
  let x := (5 * q - 4 * p) / 5  -- weight of empty bucket
  let y := (12 * (p - q)) / 5   -- weight of water when bucket is full
  let weight_three_quarters := x + 3/4 * y
  let weight_one_third := x + 1/3 * y
  have h1 : weight_three_quarters = p := by sorry
  have h2 : weight_one_third = q := by sorry
  (8 * p - 7 * q) / 5

#check bucket_weight

end NUMINAMATH_CALUDE_bucket_weight_l3791_379136


namespace NUMINAMATH_CALUDE_tetrahedron_dihedral_angle_l3791_379101

/-- Regular tetrahedron with given dimensions -/
structure RegularTetrahedron where
  base_side_length : ℝ
  side_edge_length : ℝ

/-- Plane that divides the tetrahedron's volume equally -/
structure DividingPlane where
  tetrahedron : RegularTetrahedron
  passes_through_AB : Bool
  divides_volume_equally : Bool

/-- The cosine of the dihedral angle between the dividing plane and the base -/
def dihedral_angle_cosine (plane : DividingPlane) : ℝ :=
  sorry

theorem tetrahedron_dihedral_angle 
  (t : RegularTetrahedron) 
  (p : DividingPlane) 
  (h1 : t.base_side_length = 1) 
  (h2 : t.side_edge_length = 2) 
  (h3 : p.tetrahedron = t) 
  (h4 : p.passes_through_AB = true) 
  (h5 : p.divides_volume_equally = true) : 
  dihedral_angle_cosine p = 2 * Real.sqrt 15 / 15 :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_dihedral_angle_l3791_379101


namespace NUMINAMATH_CALUDE_quadratic_function_range_l3791_379128

/-- Given a quadratic function f(x) = x^2 - 2x + 3 on the closed interval [0, m],
    if the maximum value of f(x) is 3 and the minimum value is 2,
    then m is in the interval [1, 2]. -/
theorem quadratic_function_range (m : ℝ) : 
  (∀ x ∈ Set.Icc 0 m, x^2 - 2*x + 3 ≤ 3) ∧ 
  (∀ x ∈ Set.Icc 0 m, x^2 - 2*x + 3 ≥ 2) →
  m ∈ Set.Icc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l3791_379128


namespace NUMINAMATH_CALUDE_largest_number_value_l3791_379171

theorem largest_number_value
  (a b c : ℝ)
  (h_order : a < b ∧ b < c)
  (h_sum : a + b + c = 100)
  (h_larger_diff : c - b = 10)
  (h_smaller_diff : b - a = 5) :
  c = 125 / 3 := by
sorry

end NUMINAMATH_CALUDE_largest_number_value_l3791_379171


namespace NUMINAMATH_CALUDE_median_in_70_79_interval_l3791_379156

/-- Represents a score interval with its lower bound and number of students -/
structure ScoreInterval :=
  (lower_bound : ℕ)
  (count : ℕ)

/-- The distribution of scores for 100 students -/
def score_distribution : List ScoreInterval :=
  [⟨90, 22⟩, ⟨80, 18⟩, ⟨70, 20⟩, ⟨60, 15⟩, ⟨50, 25⟩]

/-- The total number of students -/
def total_students : ℕ := 100

/-- The position of the median in the sorted list of scores -/
def median_position : ℕ := total_students / 2

/-- Finds the interval containing the median score -/
def find_median_interval (distribution : List ScoreInterval) (total : ℕ) (median_pos : ℕ) : ScoreInterval :=
  sorry

/-- Theorem stating that the interval 70-79 contains the median score -/
theorem median_in_70_79_interval :
  find_median_interval score_distribution total_students median_position = ⟨70, 20⟩ :=
sorry

end NUMINAMATH_CALUDE_median_in_70_79_interval_l3791_379156


namespace NUMINAMATH_CALUDE_keith_picked_no_pears_l3791_379175

-- Define the number of apples picked by each person
def mike_apples : ℕ := 7
def nancy_apples : ℕ := 3
def keith_apples : ℕ := 6

-- Define the total number of apples picked
def total_apples : ℕ := 16

-- Define Keith's pears as a variable
def keith_pears : ℕ := sorry

-- Theorem to prove
theorem keith_picked_no_pears : keith_pears = 0 := by
  sorry

end NUMINAMATH_CALUDE_keith_picked_no_pears_l3791_379175


namespace NUMINAMATH_CALUDE_prime_from_phi_and_omega_l3791_379141

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- Number of prime divisors of n -/
def omega (n : ℕ) : ℕ := sorry

/-- A number is prime if it has exactly two divisors -/
def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem prime_from_phi_and_omega (n : ℕ) 
  (h1 : phi n ∣ (n - 1)) 
  (h2 : omega n ≤ 3) : 
  is_prime n :=
sorry

end NUMINAMATH_CALUDE_prime_from_phi_and_omega_l3791_379141


namespace NUMINAMATH_CALUDE_not_difference_of_squares_l3791_379152

/-- The difference of squares formula cannot be directly applied to (-x+y)(x-y) -/
theorem not_difference_of_squares (x y : ℝ) : 
  ¬ ∃ (a b : ℝ), (-x + y) * (x - y) = a^2 - b^2 :=
sorry

end NUMINAMATH_CALUDE_not_difference_of_squares_l3791_379152


namespace NUMINAMATH_CALUDE_candidate_a_support_l3791_379119

/-- Represents the percentage of registered voters in each category -/
structure VoterDistribution :=
  (democrats : ℝ)
  (republicans : ℝ)
  (independents : ℝ)
  (undecided : ℝ)

/-- Represents the percentage of voters in each category supporting candidate A -/
structure SupportDistribution :=
  (democrats : ℝ)
  (republicans : ℝ)
  (independents : ℝ)
  (undecided : ℝ)

/-- Calculates the total percentage of registered voters supporting candidate A -/
def calculateTotalSupport (vd : VoterDistribution) (sd : SupportDistribution) : ℝ :=
  vd.democrats * sd.democrats +
  vd.republicans * sd.republicans +
  vd.independents * sd.independents +
  vd.undecided * sd.undecided

theorem candidate_a_support :
  let vd : VoterDistribution := {
    democrats := 0.45,
    republicans := 0.30,
    independents := 0.20,
    undecided := 0.05
  }
  let sd : SupportDistribution := {
    democrats := 0.75,
    republicans := 0.25,
    independents := 0.50,
    undecided := 0.50
  }
  calculateTotalSupport vd sd = 0.5375 := by
  sorry

end NUMINAMATH_CALUDE_candidate_a_support_l3791_379119


namespace NUMINAMATH_CALUDE_carrie_highlighters_l3791_379149

/-- The total number of highlighters in Carrie's desk drawer -/
def total_highlighters (y p b o g : ℕ) : ℕ := y + p + b + o + g

/-- Theorem stating the total number of highlighters in Carrie's desk drawer -/
theorem carrie_highlighters : ∃ (y p b o g : ℕ),
  y = 7 ∧
  p = y + 7 ∧
  b = p + 5 ∧
  o + g = 21 ∧
  o * 7 = g * 3 ∧
  total_highlighters y p b o g = 61 :=
by sorry

end NUMINAMATH_CALUDE_carrie_highlighters_l3791_379149


namespace NUMINAMATH_CALUDE_vine_paint_time_l3791_379108

/-- Time to paint different flowers and total painting time -/
def paint_problem (lily_time rose_time orchid_time vine_time : ℕ) 
  (total_time lily_count rose_count orchid_count vine_count : ℕ) : Prop :=
  lily_time * lily_count + rose_time * rose_count + 
  orchid_time * orchid_count + vine_time * vine_count = total_time

/-- Theorem stating the time to paint a vine -/
theorem vine_paint_time : 
  ∃ (vine_time : ℕ), 
    paint_problem 5 7 3 vine_time 213 17 10 6 20 ∧ 
    vine_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_vine_paint_time_l3791_379108


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l3791_379188

theorem log_sum_equals_two : 2 * Real.log 10 / Real.log 2 + Real.log 0.04 / Real.log 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l3791_379188


namespace NUMINAMATH_CALUDE_fourth_grade_students_end_of_year_l3791_379138

/-- Calculates the total number of students at the end of the year given the initial number,
    students added during the year, and new students who came to school. -/
def total_students (initial : ℝ) (added : ℝ) (new_students : ℝ) : ℝ :=
  initial + added + new_students

/-- Proves that given the specific numbers in the problem, the total number of students
    at the end of the year is 56.0. -/
theorem fourth_grade_students_end_of_year :
  total_students 10.0 4.0 42.0 = 56.0 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_students_end_of_year_l3791_379138


namespace NUMINAMATH_CALUDE_equation_solution_inequalities_solution_l3791_379190

-- Part 1: Equation solution
theorem equation_solution :
  ∀ x : ℝ, x * (x - 4) = x - 6 ↔ x = 2 ∨ x = 3 := by sorry

-- Part 2: System of inequalities solution
theorem inequalities_solution :
  ∀ x : ℝ, (4 * x - 2 ≥ 3 * (x - 1) ∧ (x - 5) / 2 + 1 > x - 3) ↔ -1 ≤ x ∧ x < 3 := by sorry

end NUMINAMATH_CALUDE_equation_solution_inequalities_solution_l3791_379190


namespace NUMINAMATH_CALUDE_factors_180_l3791_379167

/-- The number of positive factors of 180 -/
def num_factors_180 : ℕ :=
  (Finset.filter (· ∣ 180) (Finset.range 181)).card

/-- Theorem stating that the number of positive factors of 180 is 18 -/
theorem factors_180 : num_factors_180 = 18 := by
  sorry

end NUMINAMATH_CALUDE_factors_180_l3791_379167


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_l3791_379123

theorem geometric_progression_ratio (a b c d : ℝ) : 
  0 < a → a < b → b < c → c < d → d = 2*a →
  (d - a) * (a^2 / (b - a) + b^2 / (c - b) + c^2 / (d - c)) = (a + b + c)^2 →
  b * c * d / a^3 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_l3791_379123


namespace NUMINAMATH_CALUDE_number_of_installments_l3791_379176

def cash_price : ℕ := 8000
def deposit : ℕ := 3000
def monthly_installment : ℕ := 300
def cash_saving : ℕ := 4000

theorem number_of_installments : 
  (cash_price + cash_saving - deposit) / monthly_installment = 30 := by
  sorry

end NUMINAMATH_CALUDE_number_of_installments_l3791_379176


namespace NUMINAMATH_CALUDE_secret_reaches_1093_on_sunday_l3791_379104

def secret_spread (n : ℕ) : ℕ := (3^(n+1) - 1) / 2

theorem secret_reaches_1093_on_sunday : 
  ∃ n : ℕ, secret_spread n = 1093 ∧ n = 6 :=
sorry

end NUMINAMATH_CALUDE_secret_reaches_1093_on_sunday_l3791_379104


namespace NUMINAMATH_CALUDE_cake_ingredient_difference_l3791_379154

/-- Given a cake recipe and partially added ingredients, calculate the difference
    between remaining flour and required sugar. -/
theorem cake_ingredient_difference
  (total_sugar : ℕ)
  (total_flour : ℕ)
  (added_flour : ℕ)
  (h1 : total_sugar = 6)
  (h2 : total_flour = 9)
  (h3 : added_flour = 2)
  : total_flour - added_flour - total_sugar = 1 := by
  sorry

end NUMINAMATH_CALUDE_cake_ingredient_difference_l3791_379154


namespace NUMINAMATH_CALUDE_binary_multiplication_l3791_379139

-- Define binary numbers as natural numbers
def binary_1101 : ℕ := 13  -- 1101₂ in decimal
def binary_111 : ℕ := 7    -- 111₂ in decimal

-- Define the expected result
def expected_result : ℕ := 79  -- 1001111₂ in decimal

-- Theorem statement
theorem binary_multiplication :
  binary_1101 * binary_111 = expected_result := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_l3791_379139


namespace NUMINAMATH_CALUDE_equation_solution_difference_l3791_379189

theorem equation_solution_difference : ∃ x₁ x₂ : ℝ,
  (x₁ + 3)^2 / (3 * x₁ + 29) = 2 ∧
  (x₂ + 3)^2 / (3 * x₂ + 29) = 2 ∧
  x₁ ≠ x₂ ∧
  x₂ - x₁ = 14 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_difference_l3791_379189


namespace NUMINAMATH_CALUDE_ratio_problem_l3791_379191

theorem ratio_problem (a b c : ℝ) (h1 : b/a = 2) (h2 : c/b = 3) : (a + b) / (b + c) = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3791_379191


namespace NUMINAMATH_CALUDE_area_of_bcd_l3791_379142

-- Define the right triangular prism
structure RightTriangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  x : ℝ
  y : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_abc : x = (1/2) * a * b
  h_adc : y = (1/2) * b * c

-- Theorem statement
theorem area_of_bcd (prism : RightTriangularPrism) : 
  (1/2) * prism.b * prism.c = prism.y := by
  sorry

end NUMINAMATH_CALUDE_area_of_bcd_l3791_379142


namespace NUMINAMATH_CALUDE_moon_speed_in_km_per_hour_l3791_379115

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- The moon's speed in kilometers per second -/
def moon_speed_km_per_sec : ℚ := 9/10

/-- Converts a speed from kilometers per second to kilometers per hour -/
def km_per_sec_to_km_per_hour (speed : ℚ) : ℚ :=
  speed * seconds_per_hour

theorem moon_speed_in_km_per_hour :
  km_per_sec_to_km_per_hour moon_speed_km_per_sec = 3240 := by
  sorry

end NUMINAMATH_CALUDE_moon_speed_in_km_per_hour_l3791_379115


namespace NUMINAMATH_CALUDE_intersection_point_is_correct_l3791_379106

-- Define the slope of the first line
def m₁ : ℚ := 2

-- Define the first line: y = 2x + 3
def line₁ (x y : ℚ) : Prop := y = m₁ * x + 3

-- Define the slope of the perpendicular line
def m₂ : ℚ := -1 / m₁

-- Define the point that the perpendicular line passes through
def point : ℚ × ℚ := (3, 8)

-- Define the perpendicular line passing through (3, 8)
def line₂ (x y : ℚ) : Prop :=
  y - point.2 = m₂ * (x - point.1)

-- Define the intersection point
def intersection_point : ℚ × ℚ := (13/5, 41/5)

-- Theorem statement
theorem intersection_point_is_correct :
  line₁ intersection_point.1 intersection_point.2 ∧
  line₂ intersection_point.1 intersection_point.2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_is_correct_l3791_379106
