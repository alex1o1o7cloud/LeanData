import Mathlib

namespace NUMINAMATH_CALUDE_rachel_rona_age_ratio_l2616_261681

/-- Given the ages of Rachel, Rona, and Collete, prove that the ratio of Rachel's age to Rona's age is 2:1 -/
theorem rachel_rona_age_ratio (rachel_age rona_age collete_age : ℕ) : 
  rachel_age > rona_age →
  collete_age = rona_age / 2 →
  rona_age = 8 →
  rachel_age - collete_age = 12 →
  rachel_age / rona_age = 2 := by
  sorry


end NUMINAMATH_CALUDE_rachel_rona_age_ratio_l2616_261681


namespace NUMINAMATH_CALUDE_friends_at_reception_l2616_261669

def wedding_reception (total_guests : ℕ) (bride_couples : ℕ) (groom_couples : ℕ) : ℕ :=
  total_guests - 2 * (bride_couples + groom_couples)

theorem friends_at_reception :
  wedding_reception 300 30 30 = 180 := by
  sorry

end NUMINAMATH_CALUDE_friends_at_reception_l2616_261669


namespace NUMINAMATH_CALUDE_sick_days_per_year_l2616_261695

/-- Represents the number of hours in a workday -/
def hoursPerDay : ℕ := 8

/-- Represents the number of hours remaining after using half of the allotment -/
def remainingHours : ℕ := 80

/-- Theorem stating that the number of sick days per year is 20 -/
theorem sick_days_per_year :
  ∀ (sickDays vacationDays : ℕ),
  sickDays = vacationDays →
  sickDays + vacationDays = 2 * (remainingHours / hoursPerDay) →
  sickDays = 20 := by sorry

end NUMINAMATH_CALUDE_sick_days_per_year_l2616_261695


namespace NUMINAMATH_CALUDE_tinplate_allocation_l2616_261688

theorem tinplate_allocation (total_tinplates : ℕ) 
  (bodies_per_tinplate : ℕ) (bottoms_per_tinplate : ℕ) 
  (bodies_to_bottoms_ratio : ℚ) :
  total_tinplates = 36 →
  bodies_per_tinplate = 25 →
  bottoms_per_tinplate = 40 →
  bodies_to_bottoms_ratio = 1/2 →
  ∃ (bodies_tinplates bottoms_tinplates : ℕ),
    bodies_tinplates + bottoms_tinplates = total_tinplates ∧
    bodies_tinplates * bodies_per_tinplate * 2 = bottoms_tinplates * bottoms_per_tinplate ∧
    bodies_tinplates = 16 ∧
    bottoms_tinplates = 20 :=
by sorry

end NUMINAMATH_CALUDE_tinplate_allocation_l2616_261688


namespace NUMINAMATH_CALUDE_largest_common_divisor_m_squared_minus_n_squared_plus_two_l2616_261611

theorem largest_common_divisor_m_squared_minus_n_squared_plus_two
  (m n : ℤ) (h : n < m) :
  ∃ (k : ℤ), m^2 - n^2 + 2 = 2 * k ∧
  ∀ (d : ℤ), (∀ (a b : ℤ), b < a → ∃ (l : ℤ), a^2 - b^2 + 2 = d * l) → d ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_largest_common_divisor_m_squared_minus_n_squared_plus_two_l2616_261611


namespace NUMINAMATH_CALUDE_expression_simplification_l2616_261638

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 7) :
  (2 / (x - 3) - 1 / (x + 3)) / ((x^2 + 9*x) / (x^2 - 9)) = Real.sqrt 7 / 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2616_261638


namespace NUMINAMATH_CALUDE_triangle_side_sharing_l2616_261653

/-- A point on a circle -/
structure Point

/-- A triangle formed by three points -/
structure Triangle (Point : Type) where
  p1 : Point
  p2 : Point
  p3 : Point

/-- A side of a triangle -/
structure Side (Point : Type) where
  p1 : Point
  p2 : Point

/-- Definition of 8 points on a circle -/
def circle_points : Finset Point := sorry

/-- Definition of all possible triangles formed by the 8 points -/
def all_triangles : Finset (Triangle Point) := sorry

/-- Definition of all possible sides formed by the 8 points -/
def all_sides : Finset (Side Point) := sorry

/-- Function to get the sides of a triangle -/
def triangle_sides (t : Triangle Point) : Finset (Side Point) := sorry

theorem triangle_side_sharing :
  ∀ (triangles : Finset (Triangle Point)),
    triangles ⊆ all_triangles →
    triangles.card = 9 →
    ∃ (t1 t2 : Triangle Point) (s : Side Point),
      t1 ∈ triangles ∧ t2 ∈ triangles ∧ t1 ≠ t2 ∧
      s ∈ triangle_sides t1 ∧ s ∈ triangle_sides t2 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_sharing_l2616_261653


namespace NUMINAMATH_CALUDE_circle_tangent_lines_l2616_261680

/-- Given a circle with equation (x-1)^2 + (y+3)^2 = 4 and a point (-1, -1),
    the tangent lines from this point to the circle have equations x = -1 or y = -1 -/
theorem circle_tangent_lines (x y : ℝ) :
  let circle := (x - 1)^2 + (y + 3)^2 = 4
  let point := ((-1 : ℝ), (-1 : ℝ))
  let tangent1 := x = -1
  let tangent2 := y = -1
  (∃ (t : ℝ), circle ∧ (tangent1 ∨ tangent2) ∧
    (point.1 = t ∧ point.2 = -1) ∨ (point.1 = -1 ∧ point.2 = t)) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_lines_l2616_261680


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l2616_261692

theorem smallest_n_satisfying_conditions : ∃ (n : ℕ), 
  n > 20 ∧ 
  n % 6 = 4 ∧ 
  n % 7 = 3 ∧ 
  n % 8 = 5 ∧ 
  (∀ (m : ℕ), m > 20 ∧ m % 6 = 4 ∧ m % 7 = 3 ∧ m % 8 = 5 → m ≥ n) ∧
  n = 136 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l2616_261692


namespace NUMINAMATH_CALUDE_complex_magnitude_l2616_261675

theorem complex_magnitude (z : ℂ) : z * (1 - Complex.I) = Complex.I → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2616_261675


namespace NUMINAMATH_CALUDE_fence_cost_square_plot_l2616_261600

/-- The cost of building a fence around a square plot -/
theorem fence_cost_square_plot (area : ℝ) (cost_per_foot : ℝ) (total_cost : ℝ) :
  area = 289 →
  cost_per_foot = 59 →
  total_cost = 4 * Real.sqrt area * cost_per_foot →
  total_cost = 4012 := by
  sorry

#check fence_cost_square_plot

end NUMINAMATH_CALUDE_fence_cost_square_plot_l2616_261600


namespace NUMINAMATH_CALUDE_no_numbers_seven_times_digit_sum_l2616_261616

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem no_numbers_seven_times_digit_sum : 
  ∀ n : ℕ, n > 0 ∧ n < 10000 → n ≠ 7 * (sum_of_digits n) :=
by
  sorry

end NUMINAMATH_CALUDE_no_numbers_seven_times_digit_sum_l2616_261616


namespace NUMINAMATH_CALUDE_johnny_distance_l2616_261660

/-- The distance between Q and Y in kilometers -/
def total_distance : ℝ := 45

/-- Matthew's walking rate in kilometers per hour -/
def matthew_rate : ℝ := 3

/-- Johnny's walking rate in kilometers per hour -/
def johnny_rate : ℝ := 4

/-- The time difference in hours between when Matthew and Johnny start walking -/
def time_difference : ℝ := 1

/-- The theorem stating that Johnny walked 24 km when they met -/
theorem johnny_distance : ℝ := by
  sorry

end NUMINAMATH_CALUDE_johnny_distance_l2616_261660


namespace NUMINAMATH_CALUDE_donation_start_age_l2616_261668

def annual_donation : ℕ := 8000
def current_age : ℕ := 71
def total_donation : ℕ := 440000

theorem donation_start_age :
  ∃ (start_age : ℕ),
    start_age = current_age - (total_donation / annual_donation) ∧
    start_age = 16 := by
  sorry

end NUMINAMATH_CALUDE_donation_start_age_l2616_261668


namespace NUMINAMATH_CALUDE_abc_inequality_l2616_261621

theorem abc_inequality (a b c x y z : ℝ) 
  (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1)
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : a^x = b*c) (eq2 : b^y = c*a) (eq3 : c^z = a*b) :
  1/(2+x) + 1/(2+y) + 1/(2+z) ≤ 3/4 := by sorry

end NUMINAMATH_CALUDE_abc_inequality_l2616_261621


namespace NUMINAMATH_CALUDE_dog_tail_length_l2616_261622

/-- Represents the length of a dog's body parts and total length --/
structure DogMeasurements where
  body : ℝ
  head : ℝ
  tail : ℝ
  total : ℝ

/-- Theorem stating the tail length of a dog given specific proportions --/
theorem dog_tail_length (d : DogMeasurements) 
  (h1 : d.tail = d.body / 2)
  (h2 : d.head = d.body / 6)
  (h3 : d.total = d.body + d.head + d.tail)
  (h4 : d.total = 30) : 
  d.tail = 9 := by
  sorry

end NUMINAMATH_CALUDE_dog_tail_length_l2616_261622


namespace NUMINAMATH_CALUDE_inclination_angle_60_degrees_l2616_261685

def line (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 1 = 0

theorem inclination_angle_60_degrees :
  line (Real.sqrt 3) 4 →
  ∃ θ : ℝ, θ = 60 * π / 180 ∧ Real.tan θ = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_inclination_angle_60_degrees_l2616_261685


namespace NUMINAMATH_CALUDE_max_value_of_g_l2616_261607

/-- Definition of the function f --/
def f (n : ℕ+) : ℕ := 70 + n^2

/-- Definition of the function g --/
def g (n : ℕ+) : ℕ := Nat.gcd (f n) (f (n + 1))

/-- Theorem stating the maximum value of g(n) --/
theorem max_value_of_g :
  ∃ (m : ℕ+), ∀ (n : ℕ+), g n ≤ g m ∧ g m = 281 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_g_l2616_261607


namespace NUMINAMATH_CALUDE_smallest_number_with_gcd_six_l2616_261664

theorem smallest_number_with_gcd_six : ∃ (n : ℕ), 
  (70 ≤ n ∧ n ≤ 90) ∧ 
  Nat.gcd n 24 = 6 ∧ 
  (∀ m, (70 ≤ m ∧ m < n) → Nat.gcd m 24 ≠ 6) ∧
  n = 78 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_with_gcd_six_l2616_261664


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2616_261679

/-- A function satisfying the given functional equation for all integers -/
def SatisfiesFunctionalEq (f : ℤ → ℤ) : Prop :=
  ∀ m n : ℤ, f (f m + n) + f m = f n + f (3 * m) + 2014

/-- The theorem stating that any function satisfying the functional equation
    must be of the form f(n) = 2n + 1007 -/
theorem functional_equation_solution :
  ∀ f : ℤ → ℤ, SatisfiesFunctionalEq f → ∀ n : ℤ, f n = 2 * n + 1007 :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2616_261679


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2616_261636

/-- The sum of the first n terms of a geometric series with first term a and common ratio r -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- Theorem: The sum of the first nine terms of a geometric series with first term 1/3 and common ratio 2/3 is 19171/19683 -/
theorem geometric_series_sum :
  geometricSum (1/3) (2/3) 9 = 19171/19683 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2616_261636


namespace NUMINAMATH_CALUDE_alternate_seating_l2616_261684

theorem alternate_seating (B : ℕ) :
  (∃ (G : ℕ), G = 1 ∧ B > 0 ∧ B - 1 = 24) → B = 25 := by
  sorry

end NUMINAMATH_CALUDE_alternate_seating_l2616_261684


namespace NUMINAMATH_CALUDE_angle_terminal_side_formula_l2616_261627

/-- Given a point P(-4,3) on the terminal side of angle α, prove that 2sin α + cos α = 2/5 -/
theorem angle_terminal_side_formula (α : Real) (P : ℝ × ℝ) : 
  P = (-4, 3) → 2 * Real.sin α + Real.cos α = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_angle_terminal_side_formula_l2616_261627


namespace NUMINAMATH_CALUDE_certain_number_proof_l2616_261604

theorem certain_number_proof : ∃ x : ℤ, (287^2 : ℤ) + x^2 - 2*287*x = 324 ∧ x = 269 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2616_261604


namespace NUMINAMATH_CALUDE_cloth_cost_price_l2616_261629

theorem cloth_cost_price (meters : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ) :
  meters = 85 →
  selling_price = 8925 →
  profit_per_meter = 35 →
  (selling_price - meters * profit_per_meter) / meters = 70 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l2616_261629


namespace NUMINAMATH_CALUDE_notebook_cost_l2616_261674

theorem notebook_cost (total_students : ℕ) 
  (buyers : ℕ) 
  (notebooks_per_buyer : ℕ) 
  (notebook_cost : ℕ) 
  (total_cost : ℕ) :
  total_students = 40 →
  buyers > total_students / 2 →
  notebooks_per_buyer > 2 →
  notebook_cost > 2 * notebooks_per_buyer →
  buyers * notebooks_per_buyer * notebook_cost = total_cost →
  total_cost = 4515 →
  notebook_cost = 35 := by
sorry

end NUMINAMATH_CALUDE_notebook_cost_l2616_261674


namespace NUMINAMATH_CALUDE_sin_90_degrees_l2616_261632

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_90_degrees_l2616_261632


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l2616_261690

theorem fraction_zero_implies_x_equals_one :
  ∀ x : ℝ, (x^2 - 1) / (x + 1) = 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l2616_261690


namespace NUMINAMATH_CALUDE_coefficient_of_P_equals_30_l2616_261609

/-- The generating function P as described in the problem -/
def P (x : Fin 6 → ℚ) : ℚ :=
  (1 / 24) * (
    (x 0 + x 1 + x 2 + x 3 + x 4 + x 5)^6 +
    6 * (x 0 + x 1 + x 2 + x 3 + x 4 + x 5)^2 * (x 0^4 + x 1^4 + x 2^4 + x 3^4 + x 4^4 + x 5^4) +
    3 * (x 0 + x 1 + x 2 + x 3 + x 4 + x 5)^2 * (x 0^2 + x 1^2 + x 2^2 + x 3^2 + x 4^2 + x 5^2)^2 +
    6 * (x 0^2 + x 1^2 + x 2^2 + x 3^2 + x 4^2 + x 5^2)^3 +
    8 * (x 0^3 + x 1^3 + x 2^3 + x 3^3 + x 4^3 + x 5^3)^2
  )

/-- The coefficient of x₁x₂x₃x₄x₅x₆ in the generating function P -/
def coefficient_x1x2x3x4x5x6 (P : (Fin 6 → ℚ) → ℚ) : ℚ :=
  sorry  -- Definition of how to extract the coefficient

theorem coefficient_of_P_equals_30 :
  coefficient_x1x2x3x4x5x6 P = 30 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_P_equals_30_l2616_261609


namespace NUMINAMATH_CALUDE_apple_stack_count_l2616_261633

def pyramid_stack (base_length : ℕ) (base_width : ℕ) : ℕ :=
  let layers := List.range (base_length - 1)
  let regular_layers := layers.map (λ i => (base_length - i) * (base_width - i))
  let top_layer := 2
  regular_layers.sum + top_layer

theorem apple_stack_count : pyramid_stack 6 9 = 156 := by
  sorry

end NUMINAMATH_CALUDE_apple_stack_count_l2616_261633


namespace NUMINAMATH_CALUDE_g_value_at_4_l2616_261697

/-- The cubic polynomial f(x) = x^3 - 2x + 5 -/
def f (x : ℝ) : ℝ := x^3 - 2*x + 5

/-- g is a cubic polynomial such that g(0) = 1 and its roots are the squares of the roots of f -/
def g : ℝ → ℝ :=
  sorry

theorem g_value_at_4 : g 4 = -9/25 := by
  sorry

end NUMINAMATH_CALUDE_g_value_at_4_l2616_261697


namespace NUMINAMATH_CALUDE_girls_average_score_l2616_261619

-- Define the variables
def num_girls : ℝ := 1
def num_boys : ℝ := 1.8 * num_girls
def class_average : ℝ := 75
def girls_score_ratio : ℝ := 1.2

-- Theorem statement
theorem girls_average_score :
  ∃ (girls_score : ℝ),
    girls_score * num_girls + (girls_score / girls_score_ratio) * num_boys = 
    class_average * (num_girls + num_boys) ∧
    girls_score = 84 := by
  sorry

end NUMINAMATH_CALUDE_girls_average_score_l2616_261619


namespace NUMINAMATH_CALUDE_degree_of_g_l2616_261651

/-- Given polynomials f and g, where h(x) = f(g(x)) + g(x), 
    the degree of h(x) is 6, and the degree of f(x) is 3, 
    then the degree of g(x) is 2. -/
theorem degree_of_g (f g h : Polynomial ℝ) :
  (∀ x, h.eval x = (f.comp g).eval x + g.eval x) →
  h.degree = 6 →
  f.degree = 3 →
  g.degree = 2 := by
sorry

end NUMINAMATH_CALUDE_degree_of_g_l2616_261651


namespace NUMINAMATH_CALUDE_equation_solution_l2616_261694

theorem equation_solution : 
  ∃ x₁ x₂ : ℝ, (x₁ = 4 ∧ x₂ = -6) ∧ 
  (∀ x : ℝ, 2 * (x + 1)^2 - 49 = 1 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2616_261694


namespace NUMINAMATH_CALUDE_infinitely_many_odd_n_composite_l2616_261656

theorem infinitely_many_odd_n_composite (n : ℕ) : 
  ∃ S : Set ℕ, (Set.Infinite S) ∧ 
  (∀ n ∈ S, Odd n ∧ ¬(Nat.Prime (2^n + n - 1))) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_odd_n_composite_l2616_261656


namespace NUMINAMATH_CALUDE_ratio_x_y_is_four_to_one_l2616_261601

theorem ratio_x_y_is_four_to_one 
  (x y : ℝ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : 2 * Real.log (x - 2*y) = Real.log x + Real.log y) : 
  x / y = 4 := by
sorry

end NUMINAMATH_CALUDE_ratio_x_y_is_four_to_one_l2616_261601


namespace NUMINAMATH_CALUDE_school_size_calculation_l2616_261657

/-- Represents a school with stratified sampling -/
structure School where
  total_students : ℕ
  grade11_students : ℕ
  sample_size : ℕ
  grade10_sample : ℕ
  grade12_sample : ℕ

/-- The theorem stating the conditions and the conclusion to be proved -/
theorem school_size_calculation (s : School)
  (h1 : s.grade11_students = 600)
  (h2 : s.sample_size = 50)
  (h3 : s.grade10_sample = 15)
  (h4 : s.grade12_sample = 20) :
  s.total_students = 2000 := by
  sorry


end NUMINAMATH_CALUDE_school_size_calculation_l2616_261657


namespace NUMINAMATH_CALUDE_triangle_inequality_for_specific_triangle_l2616_261652

/-- A triangle with sides of length 3, 4, and x is valid if and only if 1 < x < 7 -/
theorem triangle_inequality_for_specific_triangle (x : ℝ) :
  (3 + 4 > x ∧ 3 + x > 4 ∧ 4 + x > 3) ↔ (1 < x ∧ x < 7) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_for_specific_triangle_l2616_261652


namespace NUMINAMATH_CALUDE_cost_increase_when_b_doubled_l2616_261630

theorem cost_increase_when_b_doubled (t : ℝ) (b : ℝ) :
  let original_cost := t * b^4
  let new_cost := t * (2*b)^4
  new_cost = 16 * original_cost :=
by sorry

end NUMINAMATH_CALUDE_cost_increase_when_b_doubled_l2616_261630


namespace NUMINAMATH_CALUDE_fourth_group_frequency_l2616_261617

theorem fourth_group_frequency 
  (groups : Fin 6 → ℝ) 
  (first_three_sum : (groups 0) + (groups 1) + (groups 2) = 0.65)
  (last_two_sum : (groups 4) + (groups 5) = 0.32)
  (all_sum_to_one : (groups 0) + (groups 1) + (groups 2) + (groups 3) + (groups 4) + (groups 5) = 1) :
  groups 3 = 0.03 := by
  sorry

end NUMINAMATH_CALUDE_fourth_group_frequency_l2616_261617


namespace NUMINAMATH_CALUDE_girls_on_same_team_probability_l2616_261661

/-- The probability of all three girls being on the same team when five boys and three girls
    are randomly divided into two four-person teams is 1/7. -/
theorem girls_on_same_team_probability :
  let total_children : ℕ := 8
  let num_boys : ℕ := 5
  let num_girls : ℕ := 3
  let team_size : ℕ := 4
  let total_ways : ℕ := (Nat.choose total_children team_size) / 2
  let favorable_ways : ℕ := Nat.choose num_boys 1
  ↑favorable_ways / ↑total_ways = 1 / 7 :=
by sorry

end NUMINAMATH_CALUDE_girls_on_same_team_probability_l2616_261661


namespace NUMINAMATH_CALUDE_plan_b_more_economical_l2616_261650

/-- Proves that Plan B (fixed money spent) is more economical than Plan A (fixed amount of gasoline) for two refuelings with different prices. -/
theorem plan_b_more_economical (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) :
  (2 * x * y) / (x + y) < (x + y) / 2 := by
  sorry

#check plan_b_more_economical

end NUMINAMATH_CALUDE_plan_b_more_economical_l2616_261650


namespace NUMINAMATH_CALUDE_intersection_point_of_problem_lines_l2616_261641

/-- Two lines in a 2D plane -/
structure TwoLines where
  line1 : ℝ → ℝ → Prop
  line2 : ℝ → ℝ → Prop

/-- The specific two lines from the problem -/
def problemLines : TwoLines where
  line1 := λ x y ↦ x + y + 3 = 0
  line2 := λ x y ↦ x - 2*y + 3 = 0

/-- Definition of an intersection point -/
def isIntersectionPoint (lines : TwoLines) (x y : ℝ) : Prop :=
  lines.line1 x y ∧ lines.line2 x y

/-- Theorem stating that (-3, 0) is the intersection point of the given lines -/
theorem intersection_point_of_problem_lines :
  isIntersectionPoint problemLines (-3) 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_problem_lines_l2616_261641


namespace NUMINAMATH_CALUDE_firewood_per_log_l2616_261602

/-- Calculates the number of pieces of firewood per log -/
def piecesPerLog (totalPieces : ℕ) (totalTrees : ℕ) (logsPerTree : ℕ) : ℚ :=
  totalPieces / (totalTrees * logsPerTree)

theorem firewood_per_log :
  piecesPerLog 500 25 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_firewood_per_log_l2616_261602


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_tangency_l2616_261644

/-- An ellipse with equation x^2 + 9y^2 = 9 is tangent to a hyperbola with equation x^2 - m(y - 2)^2 = 4 -/
theorem ellipse_hyperbola_tangency (m : ℝ) : 
  (∃ x y : ℝ, x^2 + 9*y^2 = 9 ∧ x^2 - m*(y - 2)^2 = 4 ∧ 
   ∀ x' y' : ℝ, (x' ≠ x ∨ y' ≠ y) → 
   (x'^2 + 9*y'^2 - 9) * (x'^2 - m*(y' - 2)^2 - 4) > 0) → 
  m = 45/31 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_tangency_l2616_261644


namespace NUMINAMATH_CALUDE_inequality_proof_l2616_261654

theorem inequality_proof (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m + n = 1) :
  (m + 1/m) * (n + 1/n) ≥ 25/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2616_261654


namespace NUMINAMATH_CALUDE_fruit_vendor_sales_l2616_261615

/-- Calculates the total sales for a fruit vendor given the prices and quantities sold --/
theorem fruit_vendor_sales
  (apple_price : ℚ)
  (orange_price : ℚ)
  (morning_apples : ℕ)
  (morning_oranges : ℕ)
  (afternoon_apples : ℕ)
  (afternoon_oranges : ℕ)
  (h1 : apple_price = 3/2)
  (h2 : orange_price = 1)
  (h3 : morning_apples = 40)
  (h4 : morning_oranges = 30)
  (h5 : afternoon_apples = 50)
  (h6 : afternoon_oranges = 40) :
  let morning_sales := apple_price * morning_apples + orange_price * morning_oranges
  let afternoon_sales := apple_price * afternoon_apples + orange_price * afternoon_oranges
  morning_sales + afternoon_sales = 205 :=
by sorry

end NUMINAMATH_CALUDE_fruit_vendor_sales_l2616_261615


namespace NUMINAMATH_CALUDE_only_zero_point_eight_greater_than_zero_point_seven_l2616_261670

theorem only_zero_point_eight_greater_than_zero_point_seven :
  let numbers : List ℝ := [0.07, -0.41, 0.8, 0.35, -0.9]
  ∀ x ∈ numbers, x > 0.7 ↔ x = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_only_zero_point_eight_greater_than_zero_point_seven_l2616_261670


namespace NUMINAMATH_CALUDE_reflect_y_of_neg_five_two_l2616_261672

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

theorem reflect_y_of_neg_five_two :
  reflect_y (-5, 2) = (5, 2) := by sorry

end NUMINAMATH_CALUDE_reflect_y_of_neg_five_two_l2616_261672


namespace NUMINAMATH_CALUDE_danny_initial_caps_l2616_261663

/-- The number of bottle caps Danny had initially -/
def initial_caps : ℕ := sorry

/-- The number of bottle caps Danny threw away -/
def thrown_away : ℕ := 60

/-- The number of new bottle caps Danny found -/
def found : ℕ := 58

/-- The number of bottle caps Danny traded away -/
def traded_away : ℕ := 15

/-- The number of bottle caps Danny received in trade -/
def received : ℕ := 25

/-- The number of bottle caps Danny has now -/
def final_caps : ℕ := 67

/-- Theorem stating that Danny initially had 59 bottle caps -/
theorem danny_initial_caps : 
  initial_caps = 59 ∧
  final_caps = initial_caps - thrown_away + found - traded_away + received :=
sorry

end NUMINAMATH_CALUDE_danny_initial_caps_l2616_261663


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2616_261626

theorem imaginary_part_of_complex_fraction : Complex.im ((1 + 3*Complex.I) / (3 - Complex.I)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2616_261626


namespace NUMINAMATH_CALUDE_x_squared_coefficient_of_product_l2616_261689

/-- The coefficient of x^2 in the expansion of (3x^2 + 4x + 5)(6x^2 + 7x + 8) is 82 -/
theorem x_squared_coefficient_of_product : 
  let p₁ : Polynomial ℝ := 3 * X^2 + 4 * X + 5
  let p₂ : Polynomial ℝ := 6 * X^2 + 7 * X + 8
  (p₁ * p₂).coeff 2 = 82 := by
sorry

end NUMINAMATH_CALUDE_x_squared_coefficient_of_product_l2616_261689


namespace NUMINAMATH_CALUDE_two_digit_number_property_l2616_261649

theorem two_digit_number_property (N : ℕ) : 
  (N ≥ 10 ∧ N ≤ 99) →  -- N is a two-digit number
  (4 * (N / 10) + 2 * (N % 10) = N / 2) →  -- Property condition
  (N = 32 ∨ N = 64 ∨ N = 96) := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l2616_261649


namespace NUMINAMATH_CALUDE_marks_initial_trees_l2616_261639

theorem marks_initial_trees (total_after_planting : ℕ) (trees_to_plant : ℕ) : 
  total_after_planting = 25 → trees_to_plant = 12 → total_after_planting - trees_to_plant = 13 := by
  sorry

end NUMINAMATH_CALUDE_marks_initial_trees_l2616_261639


namespace NUMINAMATH_CALUDE_sum_inequality_l2616_261686

theorem sum_inequality (t1 t2 t3 t4 t5 : ℝ) :
  (1 - t1) * Real.exp t1 +
  (1 - t2) * Real.exp (t1 + t2) +
  (1 - t3) * Real.exp (t1 + t2 + t3) +
  (1 - t4) * Real.exp (t1 + t2 + t3 + t4) +
  (1 - t5) * Real.exp (t1 + t2 + t3 + t4 + t5) ≤ Real.exp (Real.exp (Real.exp (Real.exp 1))) := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l2616_261686


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2616_261671

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 - 4*y + m = 0 → y = x) → 
  m = 4 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2616_261671


namespace NUMINAMATH_CALUDE_waldo_puzzles_per_book_l2616_261603

theorem waldo_puzzles_per_book 
  (num_books : ℕ) 
  (minutes_per_puzzle : ℕ) 
  (total_minutes : ℕ) 
  (h1 : num_books = 15)
  (h2 : minutes_per_puzzle = 3)
  (h3 : total_minutes = 1350) :
  total_minutes / minutes_per_puzzle / num_books = 30 := by
  sorry

end NUMINAMATH_CALUDE_waldo_puzzles_per_book_l2616_261603


namespace NUMINAMATH_CALUDE_carpet_reconstruction_l2616_261605

theorem carpet_reconstruction (original_length original_width cut_length cut_width new_side : ℝ) 
  (h1 : original_length = 12)
  (h2 : original_width = 9)
  (h3 : cut_length = 8)
  (h4 : cut_width = 1)
  (h5 : new_side = 10) :
  original_length * original_width - cut_length * cut_width = new_side * new_side := by
sorry

end NUMINAMATH_CALUDE_carpet_reconstruction_l2616_261605


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2616_261642

theorem min_value_sum_reciprocals (p q r : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hsum : p + q + r = 3) :
  (1 / (p + 3*q) + 1 / (q + 3*r) + 1 / (r + 3*p)) ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2616_261642


namespace NUMINAMATH_CALUDE_system_of_equations_solution_transformed_system_solution_l2616_261682

theorem system_of_equations_solution (x y : ℝ) :
  x + 2*y = 9 ∧ 2*x + y = 6 → x - y = -3 ∧ x + y = 5 := by sorry

theorem transformed_system_solution (m n x y : ℝ) :
  (m = 5 ∧ n = 4 ∧ 2*m - 3*n = -2 ∧ 3*m + 5*n = 35) →
  (2*(x+2) - 3*(y-1) = -2 ∧ 3*(x+2) + 5*(y-1) = 35 → x = 3 ∧ y = 5) := by sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_transformed_system_solution_l2616_261682


namespace NUMINAMATH_CALUDE_exponent_division_l2616_261659

theorem exponent_division (x : ℝ) : x^10 / x^2 = x^8 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2616_261659


namespace NUMINAMATH_CALUDE_coins_collected_in_hours_2_3_l2616_261665

/-- Represents the number of coins collected in each hour -/
structure CoinCollection where
  hour1 : ℕ
  hour2_3 : ℕ
  hour4 : ℕ
  given_away : ℕ
  total : ℕ

/-- The coin collection scenario for Joanne -/
def joannes_collection : CoinCollection where
  hour1 := 15
  hour2_3 := 0  -- This is what we need to prove
  hour4 := 50
  given_away := 15
  total := 120

/-- Theorem stating that Joanne collected 70 coins in hours 2 and 3 -/
theorem coins_collected_in_hours_2_3 :
  joannes_collection.hour2_3 = 70 :=
by sorry

end NUMINAMATH_CALUDE_coins_collected_in_hours_2_3_l2616_261665


namespace NUMINAMATH_CALUDE_quadrilaterals_on_circle_l2616_261696

/-- The number of convex quadrilaterals formed by 15 points on a circle -/
theorem quadrilaterals_on_circle (n : ℕ) (h : n = 15) : 
  Nat.choose n 4 = 1365 :=
by sorry

end NUMINAMATH_CALUDE_quadrilaterals_on_circle_l2616_261696


namespace NUMINAMATH_CALUDE_max_sphere_in_cones_l2616_261614

/-- Right circular cone -/
structure Cone :=
  (base_radius : ℝ)
  (height : ℝ)

/-- Configuration of two intersecting cones -/
structure ConePair :=
  (cone : Cone)
  (intersection_distance : ℝ)

/-- The maximum squared radius of a sphere fitting in both cones -/
def max_sphere_radius_squared (cp : ConePair) : ℝ :=
  sorry

/-- Theorem statement -/
theorem max_sphere_in_cones :
  let cp := ConePair.mk (Cone.mk 5 12) 4
  max_sphere_radius_squared cp = 1600 / 169 := by
  sorry

end NUMINAMATH_CALUDE_max_sphere_in_cones_l2616_261614


namespace NUMINAMATH_CALUDE_team_a_more_uniform_than_team_b_l2616_261640

/-- Represents a team in the gymnastics competition -/
structure Team where
  name : String
  variance : ℝ

/-- Determines if one team has more uniform heights than another -/
def hasMoreUniformHeights (team1 team2 : Team) : Prop :=
  team1.variance < team2.variance

/-- Theorem stating that Team A has more uniform heights than Team B -/
theorem team_a_more_uniform_than_team_b 
  (team_a team_b : Team)
  (h_team_a : team_a.name = "Team A" ∧ team_a.variance = 1.5)
  (h_team_b : team_b.name = "Team B" ∧ team_b.variance = 2.8) :
  hasMoreUniformHeights team_a team_b :=
by
  sorry

#check team_a_more_uniform_than_team_b

end NUMINAMATH_CALUDE_team_a_more_uniform_than_team_b_l2616_261640


namespace NUMINAMATH_CALUDE_max_values_f_and_g_l2616_261662

noncomputable def f (θ : ℝ) := (1 + Real.cos θ) * (1 + Real.sin θ)
noncomputable def g (θ : ℝ) := (1/2 + Real.cos θ) * (Real.sqrt 3/2 + Real.sin θ)

theorem max_values_f_and_g :
  (∃ (θ : ℝ), θ ∈ Set.Ioo 0 (Real.pi/2) ∧ f θ = (3 + 2 * Real.sqrt 2)/2) ∧
  (∀ (θ : ℝ), θ ∈ Set.Ioo 0 (Real.pi/2) → f θ ≤ (3 + 2 * Real.sqrt 2)/2) ∧
  (∃ (θ : ℝ), θ ∈ Set.Ioo 0 (Real.pi/2) ∧ g θ = Real.sqrt 3/4 + 3/2 * Real.sin (5*Real.pi/9)) ∧
  (∀ (θ : ℝ), θ ∈ Set.Ioo 0 (Real.pi/2) → g θ ≤ Real.sqrt 3/4 + 3/2 * Real.sin (5*Real.pi/9)) :=
by sorry

end NUMINAMATH_CALUDE_max_values_f_and_g_l2616_261662


namespace NUMINAMATH_CALUDE_digit_reversal_value_l2616_261643

theorem digit_reversal_value (x y : ℕ) : 
  x * 10 + y = 24 →  -- The original number is 24
  x * y = 8 →        -- The product of digits is 8
  x < 10 ∧ y < 10 →  -- The number is two-digit
  ∃ (a : ℕ), y * 10 + x = x * 10 + y + a ∧ a = 18 -- Value added to reverse digits is 18
  := by sorry

end NUMINAMATH_CALUDE_digit_reversal_value_l2616_261643


namespace NUMINAMATH_CALUDE_complex_purely_imaginary_l2616_261676

theorem complex_purely_imaginary (a : ℝ) :
  (a = 1 → ∃ (z : ℂ), z = (a - 1) * (a + 2) + (a + 3) * I ∧ z.re = 0) ∧
  (∃ (b : ℝ), b ≠ 1 ∧ ∃ (z : ℂ), z = (b - 1) * (b + 2) + (b + 3) * I ∧ z.re = 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_purely_imaginary_l2616_261676


namespace NUMINAMATH_CALUDE_work_completion_time_l2616_261620

theorem work_completion_time (x : ℝ) 
  (h1 : x > 0) 
  (h2 : 1/x + 1/18 = 1/6) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2616_261620


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2616_261646

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℤ, x^2021 + 1 = (x^12 - x^9 + x^6 - x^3 + 1) * q + (-x^4 + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2616_261646


namespace NUMINAMATH_CALUDE_cubic_factorization_l2616_261624

theorem cubic_factorization (a : ℝ) : a^3 - 4*a = a*(a+2)*(a-2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l2616_261624


namespace NUMINAMATH_CALUDE_square_sum_from_diff_and_product_l2616_261647

theorem square_sum_from_diff_and_product (x y : ℝ) 
  (h1 : x - y = 18) 
  (h2 : x * y = 9) : 
  x^2 + y^2 = 342 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_diff_and_product_l2616_261647


namespace NUMINAMATH_CALUDE_unique_correct_answers_l2616_261677

/-- Scoring rules for the Intermediate Maths Challenge -/
structure ScoringRules where
  totalQuestions : Nat
  easyQuestions : Nat
  hardQuestions : Nat
  easyMarks : Nat
  hardMarks : Nat
  easyPenalty : Nat
  hardPenalty : Nat

/-- Calculate the total score based on the number of correct answers -/
def calculateScore (rules : ScoringRules) (correctAnswers : Nat) : Int :=
  sorry

/-- Theorem stating that given the scoring rules and a total score of 80,
    the only possible number of correct answers is 16 -/
theorem unique_correct_answers (rules : ScoringRules) :
  rules.totalQuestions = 25 →
  rules.easyQuestions = 15 →
  rules.hardQuestions = 10 →
  rules.easyMarks = 5 →
  rules.hardMarks = 6 →
  rules.easyPenalty = 1 →
  rules.hardPenalty = 2 →
  ∃! (correctAnswers : Nat), calculateScore rules correctAnswers = 80 ∧ correctAnswers = 16 :=
sorry

end NUMINAMATH_CALUDE_unique_correct_answers_l2616_261677


namespace NUMINAMATH_CALUDE_equal_share_of_candles_total_divisible_by_four_l2616_261687

/- Define the number of candles for each person -/
def ambika_candles : ℕ := 4
def aniyah_candles : ℕ := 6 * ambika_candles
def bree_candles : ℕ := 2 * aniyah_candles
def caleb_candles : ℕ := bree_candles + (bree_candles / 2)

/- Define the total number of candles -/
def total_candles : ℕ := ambika_candles + aniyah_candles + bree_candles + caleb_candles

/- The theorem to prove -/
theorem equal_share_of_candles : total_candles / 4 = 37 := by
  sorry

/- Additional helper theorem to show the total is divisible by 4 -/
theorem total_divisible_by_four : total_candles % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equal_share_of_candles_total_divisible_by_four_l2616_261687


namespace NUMINAMATH_CALUDE_pie_not_crust_percentage_l2616_261631

/-- Given a pie weighing 200 grams with a crust of 50 grams,
    prove that 75% of the pie is not crust. -/
theorem pie_not_crust_percentage :
  let total_weight : ℝ := 200
  let crust_weight : ℝ := 50
  let non_crust_weight : ℝ := total_weight - crust_weight
  let non_crust_percentage : ℝ := (non_crust_weight / total_weight) * 100
  non_crust_percentage = 75 := by
  sorry


end NUMINAMATH_CALUDE_pie_not_crust_percentage_l2616_261631


namespace NUMINAMATH_CALUDE_hulk_jump_exceeds_2km_l2616_261623

def hulk_jump (n : ℕ) : ℝ :=
  if n = 0 then 0.5 else 2^(n - 1)

theorem hulk_jump_exceeds_2km : 
  (∀ k < 13, hulk_jump k ≤ 2000) ∧ 
  hulk_jump 13 > 2000 := by sorry

end NUMINAMATH_CALUDE_hulk_jump_exceeds_2km_l2616_261623


namespace NUMINAMATH_CALUDE_airplane_seats_l2616_261634

theorem airplane_seats (total_seats : ℕ) (first_class : ℕ) (coach : ℕ) 
  (h1 : total_seats = 387)
  (h2 : first_class + coach = total_seats)
  (h3 : coach = 4 * first_class + 2) :
  first_class = 77 := by
  sorry

end NUMINAMATH_CALUDE_airplane_seats_l2616_261634


namespace NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l2616_261613

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (line_intersect : Line → Line → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_parallel_plane
  (α β : Plane) (m n : Line)
  (h_distinct_planes : α ≠ β)
  (h_planes_parallel : parallel α β)
  (h_m_parallel_α : line_parallel_plane m α)
  (h_n_intersect_m : line_intersect n m)
  (h_n_not_in_β : ¬ line_in_plane n β) :
  line_parallel_plane n β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l2616_261613


namespace NUMINAMATH_CALUDE_distance_between_points_l2616_261648

/-- The distance between two points A(-1, 2) and B(-4, 6) is 5. -/
theorem distance_between_points : 
  let A : ℝ × ℝ := (-1, 2)
  let B : ℝ × ℝ := (-4, 6)
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2616_261648


namespace NUMINAMATH_CALUDE_exist_similar_numbers_l2616_261635

/-- A function that generates a number by repeating a given 3-digit number n times -/
def repeatDigits (d : Nat) (n : Nat) : Nat :=
  (d * (Nat.pow 10 (3 * n) - 1)) / 999

/-- Theorem stating the existence of three similar 1995-digit numbers with the required property -/
theorem exist_similar_numbers : ∃ (A B C : Nat),
  (A = repeatDigits 459 665) ∧
  (B = repeatDigits 495 665) ∧
  (C = repeatDigits 954 665) ∧
  (A + B = C) ∧
  (A ≠ 0) ∧ (B ≠ 0) ∧ (C ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_exist_similar_numbers_l2616_261635


namespace NUMINAMATH_CALUDE_candy_distribution_l2616_261691

theorem candy_distribution (total_candy : ℕ) (num_bags : ℕ) (candy_per_bag : ℕ) :
  total_candy = 858 →
  num_bags = 26 →
  total_candy = num_bags * candy_per_bag →
  candy_per_bag = 33 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l2616_261691


namespace NUMINAMATH_CALUDE_exactly_one_and_two_red_mutually_exclusive_non_opposing_l2616_261666

/-- Represents the color of a ball -/
inductive BallColor
| Red
| White

/-- Represents the outcome of drawing three balls -/
structure DrawOutcome :=
  (red_count : Nat)
  (white_count : Nat)
  (h_total : red_count + white_count = 3)

/-- The bag of balls -/
def bag : Multiset BallColor :=
  Multiset.replicate 5 BallColor.Red + Multiset.replicate 3 BallColor.White

/-- The event of drawing exactly one red ball -/
def exactly_one_red (outcome : DrawOutcome) : Prop :=
  outcome.red_count = 1

/-- The event of drawing exactly two red balls -/
def exactly_two_red (outcome : DrawOutcome) : Prop :=
  outcome.red_count = 2

/-- Two events are mutually exclusive -/
def mutually_exclusive (e1 e2 : DrawOutcome → Prop) : Prop :=
  ∀ outcome, ¬(e1 outcome ∧ e2 outcome)

/-- Two events are non-opposing -/
def non_opposing (e1 e2 : DrawOutcome → Prop) : Prop :=
  ∃ outcome, e1 outcome ∨ e2 outcome

theorem exactly_one_and_two_red_mutually_exclusive_non_opposing :
  mutually_exclusive exactly_one_red exactly_two_red ∧
  non_opposing exactly_one_red exactly_two_red :=
sorry

end NUMINAMATH_CALUDE_exactly_one_and_two_red_mutually_exclusive_non_opposing_l2616_261666


namespace NUMINAMATH_CALUDE_consecutive_missing_factors_l2616_261637

theorem consecutive_missing_factors (n : ℕ) (h1 : n > 30) : 
  (∀ (k : ℕ), 1 ≤ k ∧ k ≤ 30 → (k ≠ 16 ∧ k ≠ 17 → n % k = 0)) →
  (∃ (m : ℕ), m ≥ 1 ∧ m < 30 ∧ n % m ≠ 0 ∧ n % (m + 1) ≠ 0) →
  (∀ (j : ℕ), j ≥ 1 ∧ j < 30 ∧ n % j ≠ 0 ∧ n % (j + 1) ≠ 0 → j = 16) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_missing_factors_l2616_261637


namespace NUMINAMATH_CALUDE_units_digit_of_product_l2616_261612

theorem units_digit_of_product (n : ℕ) : 
  (2^2021 * 5^2022 * 7^2023) % 10 = 0 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l2616_261612


namespace NUMINAMATH_CALUDE_surface_area_of_six_cubes_l2616_261683

/-- Represents the configuration of 6 cubes fastened together -/
structure CubeConfiguration where
  numCubes : Nat
  edgeLength : ℝ
  numConnections : Nat

/-- Calculates the total surface area of the cube configuration -/
def totalSurfaceArea (config : CubeConfiguration) : ℝ :=
  (config.numCubes * 6 - 2 * config.numConnections) * config.edgeLength ^ 2

/-- Theorem stating that the total surface area of the given configuration is 26 square units -/
theorem surface_area_of_six_cubes :
  ∀ (config : CubeConfiguration),
    config.numCubes = 6 ∧
    config.edgeLength = 1 ∧
    config.numConnections = 10 →
    totalSurfaceArea config = 26 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_six_cubes_l2616_261683


namespace NUMINAMATH_CALUDE_intersection_area_greater_than_half_l2616_261699

/-- Two identical rectangles with sides a and b -/
structure Rectangle where
  a : ℝ
  b : ℝ
  pos_a : 0 < a
  pos_b : 0 < b

/-- The configuration of two intersecting rectangles -/
structure IntersectingRectangles where
  rect : Rectangle
  intersection_points : ℕ
  eight_intersections : intersection_points = 8

/-- The area of intersection of two rectangles -/
def intersectionArea (ir : IntersectingRectangles) : ℝ := sorry

/-- The area of a single rectangle -/
def rectangleArea (r : Rectangle) : ℝ := r.a * r.b

/-- Theorem: The area of intersection is greater than half the area of each rectangle -/
theorem intersection_area_greater_than_half (ir : IntersectingRectangles) :
  intersectionArea ir > (1/2) * rectangleArea ir.rect :=
sorry

end NUMINAMATH_CALUDE_intersection_area_greater_than_half_l2616_261699


namespace NUMINAMATH_CALUDE_solve_family_income_problem_l2616_261698

def family_income_problem (initial_members : ℕ) (initial_average : ℝ) 
  (final_members : ℕ) (final_average : ℝ) : Prop :=
  let initial_total := initial_members * initial_average
  let final_total := final_members * final_average
  let deceased_income := initial_total - final_total
  initial_members = 4 ∧ 
  final_members = 3 ∧ 
  initial_average = 782 ∧ 
  final_average = 650 ∧ 
  deceased_income = 1178

theorem solve_family_income_problem : 
  ∃ (initial_members final_members : ℕ) (initial_average final_average : ℝ),
    family_income_problem initial_members initial_average final_members final_average :=
by
  sorry

end NUMINAMATH_CALUDE_solve_family_income_problem_l2616_261698


namespace NUMINAMATH_CALUDE_clea_escalator_ride_time_l2616_261618

/-- Represents the escalator scenario for Clea -/
structure EscalatorScenario where
  /-- Time (in seconds) for Clea to walk down a stationary escalator -/
  stationary_time : ℝ
  /-- Time (in seconds) for Clea to walk down a moving escalator -/
  moving_time : ℝ
  /-- Slowdown factor for the escalator during off-peak hours -/
  slowdown_factor : ℝ

/-- Calculates the time for Clea to ride the slower escalator without walking -/
def ride_time (scenario : EscalatorScenario) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that given the specific scenario, the ride time is 60 seconds -/
theorem clea_escalator_ride_time :
  let scenario : EscalatorScenario :=
    { stationary_time := 80
      moving_time := 30
      slowdown_factor := 0.8 }
  ride_time scenario = 60 := by
  sorry

end NUMINAMATH_CALUDE_clea_escalator_ride_time_l2616_261618


namespace NUMINAMATH_CALUDE_number_calculation_l2616_261628

theorem number_calculation (x : ℝ) : (0.8 * 90 = 0.7 * x + 30) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l2616_261628


namespace NUMINAMATH_CALUDE_x_value_l2616_261608

theorem x_value : 
  let x := 98 * (1 + 20 / 100)
  x = 117.6 := by sorry

end NUMINAMATH_CALUDE_x_value_l2616_261608


namespace NUMINAMATH_CALUDE_specialist_time_calculation_l2616_261625

theorem specialist_time_calculation (days_in_hospital : ℕ) (bed_charge_per_day : ℕ) 
  (specialist_charge_per_hour : ℕ) (ambulance_charge : ℕ) (total_bill : ℕ) : 
  days_in_hospital = 3 →
  bed_charge_per_day = 900 →
  specialist_charge_per_hour = 250 →
  ambulance_charge = 1800 →
  total_bill = 4625 →
  (total_bill - (days_in_hospital * bed_charge_per_day + ambulance_charge)) / 
    (2 * (specialist_charge_per_hour / 60)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_specialist_time_calculation_l2616_261625


namespace NUMINAMATH_CALUDE_non_chihuahua_male_dogs_l2616_261606

theorem non_chihuahua_male_dogs (total_dogs : ℕ) (male_ratio : ℚ) (chihuahua_ratio : ℚ) :
  total_dogs = 32 →
  male_ratio = 5/8 →
  chihuahua_ratio = 3/4 →
  (total_dogs : ℚ) * male_ratio * (1 - chihuahua_ratio) = 5 := by
  sorry

end NUMINAMATH_CALUDE_non_chihuahua_male_dogs_l2616_261606


namespace NUMINAMATH_CALUDE_statement_II_always_true_l2616_261655

-- Define the possible digits
inductive Digit
| two : Digit
| three : Digit
| five : Digit
| six : Digit
| other : Digit

-- Define the statements
def statement_I (d : Digit) : Prop := d = Digit.two
def statement_II (d : Digit) : Prop := d ≠ Digit.three
def statement_III (d : Digit) : Prop := d = Digit.five
def statement_IV (d : Digit) : Prop := d ≠ Digit.six

-- Define the condition that exactly three statements are true
def three_true (d : Digit) : Prop :=
  (statement_I d ∧ statement_II d ∧ statement_III d) ∨
  (statement_I d ∧ statement_II d ∧ statement_IV d) ∨
  (statement_I d ∧ statement_III d ∧ statement_IV d) ∨
  (statement_II d ∧ statement_III d ∧ statement_IV d)

-- Theorem: Statement II is always true given the conditions
theorem statement_II_always_true :
  ∀ d : Digit, three_true d → statement_II d :=
by
  sorry


end NUMINAMATH_CALUDE_statement_II_always_true_l2616_261655


namespace NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_3_divisible_by_11_l2616_261678

def ends_in_3 (n : ℕ) : Prop := n % 10 = 3

theorem smallest_positive_integer_ending_in_3_divisible_by_11 :
  ∃ (n : ℕ), n > 0 ∧ ends_in_3 n ∧ n % 11 = 0 ∧
  ∀ (m : ℕ), m > 0 → ends_in_3 m → m % 11 = 0 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_3_divisible_by_11_l2616_261678


namespace NUMINAMATH_CALUDE_total_wheels_on_floor_l2616_261645

theorem total_wheels_on_floor (num_people : ℕ) (wheels_per_skate : ℕ) (skates_per_person : ℕ) : 
  num_people = 40 → 
  wheels_per_skate = 2 → 
  skates_per_person = 2 → 
  num_people * wheels_per_skate * skates_per_person = 160 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_on_floor_l2616_261645


namespace NUMINAMATH_CALUDE_sum_nine_terms_is_99_l2616_261673

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  (∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  (a 1 + a 4 + a 7 = 35) ∧
  (a 3 + a 6 + a 9 = 27)

/-- The sum of the first n terms of an arithmetic sequence -/
def SumArithmeticSequence (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

/-- Theorem: The sum of the first 9 terms of the specified arithmetic sequence is 99 -/
theorem sum_nine_terms_is_99 (a : ℕ → ℚ) (h : ArithmeticSequence a) :
  SumArithmeticSequence a 9 = 99 := by
  sorry

end NUMINAMATH_CALUDE_sum_nine_terms_is_99_l2616_261673


namespace NUMINAMATH_CALUDE_count_400000_to_500000_by_50_l2616_261658

def count_sequence (start : ℕ) (increment : ℕ) (end_value : ℕ) : ℕ :=
  (end_value - start) / increment + 1

theorem count_400000_to_500000_by_50 :
  count_sequence 400000 50 500000 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_count_400000_to_500000_by_50_l2616_261658


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2616_261667

def A : Set Int := {-1, 2, 4}
def B : Set Int := {-1, 0, 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2616_261667


namespace NUMINAMATH_CALUDE_square_perimeter_l2616_261693

theorem square_perimeter (s : ℝ) (h : s > 0) : 
  (7/3 * s = 42) → (4 * s = 72) := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l2616_261693


namespace NUMINAMATH_CALUDE_f_positive_iff_in_intervals_l2616_261610

def f (x : ℝ) := (x + 1) * (x - 1) * (x - 3)

theorem f_positive_iff_in_intervals (x : ℝ) : 
  f x > 0 ↔ (x > -1 ∧ x < 1) ∨ x > 3 :=
sorry

end NUMINAMATH_CALUDE_f_positive_iff_in_intervals_l2616_261610
