import Mathlib

namespace NUMINAMATH_CALUDE_parallelogram_angle_ratio_l3716_371655

-- Define a parallelogram
structure Parallelogram :=
  (A B C D : Point)

-- Define the angles of the parallelogram
def angle (p : Parallelogram) (v : Fin 4) : ℝ :=
  sorry

-- State the theorem
theorem parallelogram_angle_ratio (p : Parallelogram) :
  ∃ (k : ℝ), k > 0 ∧
    angle p 0 = k ∧
    angle p 1 = 2 * k ∧
    angle p 2 = k ∧
    angle p 3 = 2 * k :=
  sorry

end NUMINAMATH_CALUDE_parallelogram_angle_ratio_l3716_371655


namespace NUMINAMATH_CALUDE_rectangle_area_proof_l3716_371633

/-- Given a rectangle EFGH with vertices E(0, 0), F(0, 5), G(y, 5), and H(y, 0),
    where y > 0 and the area of the rectangle is 45 square units,
    prove that y = 9. -/
theorem rectangle_area_proof (y : ℝ) : y > 0 → y * 5 = 45 → y = 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_proof_l3716_371633


namespace NUMINAMATH_CALUDE_log_inequality_l3716_371696

theorem log_inequality (x₁ x₂ : ℝ) (h : 0 < x₁ ∧ x₁ < x₂) : 
  Real.log (Real.sqrt (x₁ * x₂)) = (Real.log x₁ + Real.log x₂) / 2 ∧
  Real.log (Real.sqrt (x₁ * x₂)) < Real.log ((x₁ + x₂) / 2) := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l3716_371696


namespace NUMINAMATH_CALUDE_opposite_sum_and_sum_opposite_l3716_371629

theorem opposite_sum_and_sum_opposite (a b : ℤ) (h1 : a = -6) (h2 : b = 4) : 
  (-a) + (-b) = 2 ∧ -(a + b) = 2 :=
by sorry

end NUMINAMATH_CALUDE_opposite_sum_and_sum_opposite_l3716_371629


namespace NUMINAMATH_CALUDE_A_3_2_l3716_371632

def A : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => A m 1
  | m + 1, n + 1 => A m (A (m + 1) n)

theorem A_3_2 : A 3 2 = 12 := by sorry

end NUMINAMATH_CALUDE_A_3_2_l3716_371632


namespace NUMINAMATH_CALUDE_original_number_proof_l3716_371663

theorem original_number_proof (x : ℚ) : 1 + 1 / x = 8 / 3 → x = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3716_371663


namespace NUMINAMATH_CALUDE_circle_line_tangent_problem_l3716_371648

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + 1 = 0

-- Define the line l
def line_l (x y : ℝ) (m : ℝ) : Prop :=
  x + m*y + 1 = 0

-- Define the point M
def point_M (m : ℝ) : ℝ × ℝ :=
  (m, m)

-- Define the symmetric property
def symmetric_points_exist (m : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), 
    circle_C x1 y1 ∧ circle_C x2 y2 ∧
    line_l ((x1 + x2)/2) ((y1 + y2)/2) m

-- Define the tangent property
def tangent_exists (m : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_C x y ∧
    ∃ (k : ℝ), ∀ (t : ℝ),
      ¬(circle_C (m + k*t) (m + t))

-- Main theorem
theorem circle_line_tangent_problem (m : ℝ) :
  symmetric_points_exist m → tangent_exists m →
  m = -1 ∧ Real.sqrt ((m - 1)^2 + (m - 2)^2 - 4) = 3 :=
sorry

end NUMINAMATH_CALUDE_circle_line_tangent_problem_l3716_371648


namespace NUMINAMATH_CALUDE_brad_balloons_l3716_371636

theorem brad_balloons (total : ℕ) (red : ℕ) (green : ℕ) (blue : ℕ) 
  (h1 : total = 37)
  (h2 : red = 14)
  (h3 : green = 10)
  (h4 : total = red + green + blue) :
  blue = 13 := by
  sorry

end NUMINAMATH_CALUDE_brad_balloons_l3716_371636


namespace NUMINAMATH_CALUDE_remainder_theorem_l3716_371686

-- Define the polynomial p(x) = 4x^3 - 12x^2 + 16x - 20
def p (x : ℝ) : ℝ := 4 * x^3 - 12 * x^2 + 16 * x - 20

-- Define the divisor d(x) = x - 3
def d (x : ℝ) : ℝ := x - 3

-- Theorem statement
theorem remainder_theorem :
  (p 3 : ℝ) = 28 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3716_371686


namespace NUMINAMATH_CALUDE_circle_equation_l3716_371617

/-- The standard equation of a circle with given center and radius -/
theorem circle_equation (x y : ℝ) : 
  (∃ (C : ℝ × ℝ) (r : ℝ), C = (1, -2) ∧ r = 3) →
  ((x - 1)^2 + (y + 2)^2 = 9) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3716_371617


namespace NUMINAMATH_CALUDE_ice_cream_parlor_distance_l3716_371634

/-- The distance to the ice cream parlor -/
def D : ℝ := sorry

/-- Rita's upstream paddling speed -/
def upstream_speed : ℝ := 3

/-- Rita's downstream paddling speed -/
def downstream_speed : ℝ := 9

/-- Upstream wind speed -/
def upstream_wind : ℝ := 2

/-- Downstream wind speed -/
def downstream_wind : ℝ := 4

/-- Total trip time -/
def total_time : ℝ := 8

/-- Effective upstream speed -/
def effective_upstream_speed : ℝ := upstream_speed - upstream_wind

/-- Effective downstream speed -/
def effective_downstream_speed : ℝ := downstream_speed + downstream_wind

theorem ice_cream_parlor_distance : 
  D / effective_upstream_speed + D / effective_downstream_speed = total_time := by sorry

end NUMINAMATH_CALUDE_ice_cream_parlor_distance_l3716_371634


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l3716_371676

/-- The area of an isosceles right triangle with hypotenuse 6 is 9 -/
theorem isosceles_right_triangle_area (h : ℝ) (A : ℝ) : 
  h = 6 →  -- The hypotenuse is 6 units
  A = h^2 / 4 →  -- Area formula for isosceles right triangle
  A = 9 := by
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l3716_371676


namespace NUMINAMATH_CALUDE_solve_equation_l3716_371669

theorem solve_equation (x : ℝ) : 3034 - (1002 / x) = 3029 → x = 200.4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3716_371669


namespace NUMINAMATH_CALUDE_cube_after_carving_l3716_371652

def cube_side_length : ℝ := 9

-- Volume of the cube after carving the cross-shaped groove
def remaining_volume : ℝ := 639

-- Surface area of the cube after carving the cross-shaped groove
def new_surface_area : ℝ := 510

-- Theorem statement
theorem cube_after_carving (groove_volume : ℝ) (groove_surface_area : ℝ) :
  cube_side_length ^ 3 - groove_volume = remaining_volume ∧
  6 * cube_side_length ^ 2 + groove_surface_area = new_surface_area :=
by sorry

end NUMINAMATH_CALUDE_cube_after_carving_l3716_371652


namespace NUMINAMATH_CALUDE_choir_selection_l3716_371660

theorem choir_selection (boys girls : ℕ) (h1 : boys = 3) (h2 : girls = 5) :
  let total := boys + girls
  (Nat.choose boys 2 * Nat.choose girls 2 = 30) ∧
  (Nat.choose total 4 - Nat.choose girls 4 = 65) :=
by sorry

end NUMINAMATH_CALUDE_choir_selection_l3716_371660


namespace NUMINAMATH_CALUDE_students_liking_both_sports_and_music_l3716_371640

theorem students_liking_both_sports_and_music
  (total : ℕ)
  (sports : ℕ)
  (music : ℕ)
  (neither : ℕ)
  (h_total : total = 55)
  (h_sports : sports = 43)
  (h_music : music = 34)
  (h_neither : neither = 4) :
  ∃ (both : ℕ), both = sports + music - total + neither ∧ both = 26 :=
by sorry

end NUMINAMATH_CALUDE_students_liking_both_sports_and_music_l3716_371640


namespace NUMINAMATH_CALUDE_andrews_age_l3716_371605

theorem andrews_age :
  ∀ (a g : ℝ),
  g = 15 * a →
  g - a = 55 →
  a = 55 / 14 :=
by
  sorry

end NUMINAMATH_CALUDE_andrews_age_l3716_371605


namespace NUMINAMATH_CALUDE_matrix_power_4_l3716_371649

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 1], ![1, 0]]

theorem matrix_power_4 : A^4 = ![![5, 3], ![3, 2]] := by sorry

end NUMINAMATH_CALUDE_matrix_power_4_l3716_371649


namespace NUMINAMATH_CALUDE_square_sum_given_conditions_l3716_371657

theorem square_sum_given_conditions (x y : ℝ) 
  (h1 : (x + y)^2 = 9) 
  (h2 : x * y = -6) : 
  x^2 + y^2 = 21 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_conditions_l3716_371657


namespace NUMINAMATH_CALUDE_total_choices_is_64_l3716_371674

/-- The number of tour routes available -/
def num_routes : ℕ := 4

/-- The number of tour groups -/
def num_groups : ℕ := 3

/-- The total number of different possible choices -/
def total_choices : ℕ := num_routes ^ num_groups

/-- Theorem stating that the total number of different choices is 64 -/
theorem total_choices_is_64 : total_choices = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_choices_is_64_l3716_371674


namespace NUMINAMATH_CALUDE_two_color_plane_division_l3716_371620

/-- A type representing a line in a plane. -/
structure Line where
  -- We don't need to specify the exact properties of a line for this problem

/-- A type representing a region in a plane. -/
structure Region where
  -- We don't need to specify the exact properties of a region for this problem

/-- A type representing a color (either red or blue). -/
inductive Color
  | Red
  | Blue

/-- A function that determines if two regions are adjacent. -/
def adjacent (r1 r2 : Region) : Prop :=
  sorry  -- The exact definition is not important for the statement

/-- A type representing a coloring of regions. -/
def Coloring := Region → Color

/-- A predicate that checks if a coloring is valid (no adjacent regions have the same color). -/
def valid_coloring (c : Coloring) : Prop :=
  ∀ r1 r2 : Region, adjacent r1 r2 → c r1 ≠ c r2

/-- The main theorem stating that for any set of lines dividing a plane,
    there exists a valid two-coloring of the resulting regions. -/
theorem two_color_plane_division (lines : Set Line) :
  ∃ c : Coloring, valid_coloring c :=
sorry

end NUMINAMATH_CALUDE_two_color_plane_division_l3716_371620


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3716_371607

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 = 450) →
  (a 1 + a 9 = 180) :=
by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3716_371607


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3716_371689

/-- The sum of a geometric series with given parameters -/
theorem geometric_series_sum (a₁ : ℝ) (q : ℝ) (aₙ : ℝ) (h₁ : a₁ = 100) (h₂ : q = 1/10) (h₃ : aₙ = 0.01) :
  (a₁ - aₙ * q) / (1 - q) = (10^5 - 1) / 900 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3716_371689


namespace NUMINAMATH_CALUDE_evaluate_expression_l3716_371675

theorem evaluate_expression : (8^5 / 8^2) * 2^12 = 2^21 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3716_371675


namespace NUMINAMATH_CALUDE_partner_b_investment_l3716_371685

/-- Represents the investment and profit share of a partner in a partnership. -/
structure Partner where
  investment : ℝ
  profitShare : ℝ

/-- Represents a partnership with three partners. -/
structure Partnership where
  a : Partner
  b : Partner
  c : Partner

/-- Theorem stating that given the conditions of the problem, partner b's investment is $21000. -/
theorem partner_b_investment (p : Partnership)
  (h1 : p.a.investment = 15000)
  (h2 : p.c.investment = 27000)
  (h3 : p.b.profitShare = 1540)
  (h4 : p.a.profitShare = 1100)
  : p.b.investment = 21000 := by
  sorry

#check partner_b_investment

end NUMINAMATH_CALUDE_partner_b_investment_l3716_371685


namespace NUMINAMATH_CALUDE_circle_power_theorem_l3716_371641

structure Circle where
  a : ℝ
  b : ℝ
  R : ℝ

def power (c : Circle) (x₁ y₁ : ℝ) : ℝ :=
  (x₁ - c.a)^2 + (y₁ - c.b)^2 - c.R^2

def distance_squared (x₁ y₁ a b : ℝ) : ℝ :=
  (x₁ - a)^2 + (y₁ - b)^2

theorem circle_power_theorem (c : Circle) (x₁ y₁ : ℝ) :
  -- 1. Power definition
  power c x₁ y₁ = (x₁ - c.a)^2 + (y₁ - c.b)^2 - c.R^2 ∧
  -- 2. Power sign properties
  (distance_squared x₁ y₁ c.a c.b > c.R^2 → power c x₁ y₁ > 0) ∧
  (distance_squared x₁ y₁ c.a c.b < c.R^2 → power c x₁ y₁ < 0) ∧
  (distance_squared x₁ y₁ c.a c.b = c.R^2 → power c x₁ y₁ = 0) ∧
  -- 3. Tangent length property
  (distance_squared x₁ y₁ c.a c.b > c.R^2 → 
    ∃ p, p^2 = power c x₁ y₁ ∧ p ≥ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_power_theorem_l3716_371641


namespace NUMINAMATH_CALUDE_victors_total_money_l3716_371651

-- Define Victor's initial amount
def initial_amount : ℕ := 10

-- Define Victor's allowance
def allowance : ℕ := 8

-- Theorem stating Victor's total money
theorem victors_total_money : initial_amount + allowance = 18 := by
  sorry

end NUMINAMATH_CALUDE_victors_total_money_l3716_371651


namespace NUMINAMATH_CALUDE_shortened_tripod_height_l3716_371608

/-- Represents a tripod with potentially unequal legs -/
structure Tripod where
  leg1 : ℝ
  leg2 : ℝ
  leg3 : ℝ

/-- Calculates the height of a tripod given its leg lengths -/
def tripodHeight (t : Tripod) : ℝ := sorry

/-- The original tripod with equal legs -/
def originalTripod : Tripod := ⟨5, 5, 5⟩

/-- The height of the original tripod -/
def originalHeight : ℝ := 4

/-- The tripod with one shortened leg -/
def shortenedTripod : Tripod := ⟨4, 5, 5⟩

/-- The theorem to be proved -/
theorem shortened_tripod_height :
  tripodHeight shortenedTripod = 144 / Real.sqrt (5 * 317) := by sorry

end NUMINAMATH_CALUDE_shortened_tripod_height_l3716_371608


namespace NUMINAMATH_CALUDE_max_sum_l3716_371631

/-- An arithmetic sequence {an} with sum Sn -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  sum : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, sum n = n * a 1 + n * (n - 1) / 2 * d

/-- The conditions of the problem -/
def problem_conditions (seq : ArithmeticSequence) : Prop :=
  seq.a 1 + seq.a 2 + seq.a 3 = 156 ∧
  seq.a 2 + seq.a 3 + seq.a 4 = 147

/-- The theorem to prove -/
theorem max_sum (seq : ArithmeticSequence) 
  (h : problem_conditions seq) : 
  ∃ (n : ℕ), n = 19 ∧ 
  ∀ (m : ℕ), m > 0 → seq.sum n ≥ seq.sum m :=
sorry

end NUMINAMATH_CALUDE_max_sum_l3716_371631


namespace NUMINAMATH_CALUDE_decimal_digit_of_fraction_thirteenth_over_seventeen_150th_digit_l3716_371622

theorem decimal_digit_of_fraction (n : ℕ) (a b : ℕ) (h : b ≠ 0) :
  ∃ (d : ℕ), d < 10 ∧ d = (a * 10^n) % b :=
sorry

theorem thirteenth_over_seventeen_150th_digit :
  ∃ (d : ℕ), d < 10 ∧ d = (13 * 10^150) % 17 ∧ d = 1 :=
sorry

end NUMINAMATH_CALUDE_decimal_digit_of_fraction_thirteenth_over_seventeen_150th_digit_l3716_371622


namespace NUMINAMATH_CALUDE_computers_needed_after_increase_problem_solution_l3716_371601

theorem computers_needed_after_increase (initial_students : ℕ) 
  (students_per_computer : ℕ) (additional_students : ℕ) : ℕ :=
  let initial_computers := initial_students / students_per_computer
  let additional_computers := additional_students / students_per_computer
  initial_computers + additional_computers

theorem problem_solution :
  computers_needed_after_increase 82 2 16 = 49 := by
  sorry

end NUMINAMATH_CALUDE_computers_needed_after_increase_problem_solution_l3716_371601


namespace NUMINAMATH_CALUDE_square_roots_problem_l3716_371604

theorem square_roots_problem (x a : ℝ) : 
  x > 0 ∧ (2*a - 3)^2 = x ∧ (5 - a)^2 = x → a = -2 ∧ x = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l3716_371604


namespace NUMINAMATH_CALUDE_scientific_notation_of_111_3_billion_l3716_371688

theorem scientific_notation_of_111_3_billion : ∃ (a : ℝ) (n : ℤ), 
  1 ≤ a ∧ a < 10 ∧ 111300000000 = a * (10 : ℝ) ^ n ∧ a = 1.113 ∧ n = 11 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_111_3_billion_l3716_371688


namespace NUMINAMATH_CALUDE_company_shares_l3716_371642

theorem company_shares (p v s i : Real) : 
  p + v + s + i = 1 → 
  2*p + v + s + i = 1.3 →
  p + 2*v + s + i = 1.4 →
  p + v + 3*s + i = 1.2 →
  ∃ k : Real, k > 3.75 ∧ k * i > 0.75 := by sorry

end NUMINAMATH_CALUDE_company_shares_l3716_371642


namespace NUMINAMATH_CALUDE_expand_and_simplify_powers_of_two_one_more_than_cube_l3716_371650

-- Part (i)
theorem expand_and_simplify (x : ℝ) : (x + 1) * (x^2 - x + 1) = x^3 + 1 := by sorry

-- Part (ii)
theorem powers_of_two_one_more_than_cube : 
  {n : ℕ | ∃ k : ℕ, 2^n = k^3 + 1} = {0, 1} := by sorry

end NUMINAMATH_CALUDE_expand_and_simplify_powers_of_two_one_more_than_cube_l3716_371650


namespace NUMINAMATH_CALUDE_price_reduction_theorem_l3716_371618

/-- Represents the mall's sales and profit model -/
structure MallSales where
  initial_sales : ℕ  -- Initial daily sales
  initial_profit : ℕ  -- Initial profit per item in yuan
  sales_increase_rate : ℕ  -- Additional items sold per yuan of price reduction
  price_reduction : ℕ  -- Price reduction in yuan

/-- Calculates the daily profit given a MallSales structure -/
def daily_profit (m : MallSales) : ℕ :=
  let new_sales := m.initial_sales + m.sales_increase_rate * m.price_reduction
  let new_profit_per_item := m.initial_profit - m.price_reduction
  new_sales * new_profit_per_item

/-- Theorem stating that a price reduction of 20 yuan results in a daily profit of 2100 yuan -/
theorem price_reduction_theorem (m : MallSales) 
  (h1 : m.initial_sales = 30)
  (h2 : m.initial_profit = 50)
  (h3 : m.sales_increase_rate = 2)
  (h4 : m.price_reduction = 20) :
  daily_profit m = 2100 := by
  sorry

#eval daily_profit { initial_sales := 30, initial_profit := 50, sales_increase_rate := 2, price_reduction := 20 }

end NUMINAMATH_CALUDE_price_reduction_theorem_l3716_371618


namespace NUMINAMATH_CALUDE_isosceles_triangle_dimensions_l3716_371661

/-- An isosceles triangle with base b and leg length l -/
structure IsoscelesTriangle where
  b : ℝ
  l : ℝ
  h : ℝ
  isPositive : 0 < b ∧ 0 < l ∧ 0 < h
  isIsosceles : l = b - 1
  areaRelation : (1/2) * b * h = (1/3) * b^2

/-- Theorem about the dimensions of a specific isosceles triangle -/
theorem isosceles_triangle_dimensions (t : IsoscelesTriangle) :
  t.b = 6 ∧ t.l = 5 ∧ t.h = 4 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_dimensions_l3716_371661


namespace NUMINAMATH_CALUDE_stratified_sampling_proportionality_l3716_371611

/-- Represents the number of students selected in stratified sampling -/
structure StratifiedSample where
  total : ℕ
  first_year : ℕ
  second_year : ℕ
  selected_first : ℕ
  selected_second : ℕ

/-- Checks if the stratified sample maintains proportionality -/
def is_proportional (s : StratifiedSample) : Prop :=
  s.selected_first * s.second_year = s.selected_second * s.first_year

theorem stratified_sampling_proportionality :
  ∀ s : StratifiedSample,
    s.total = 70 →
    s.first_year = 30 →
    s.second_year = 40 →
    s.selected_first = 6 →
    s.selected_second = 8 →
    is_proportional s :=
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_proportionality_l3716_371611


namespace NUMINAMATH_CALUDE_harmonic_mean_of_square_sides_l3716_371609

theorem harmonic_mean_of_square_sides (a b c : ℝ) (ha : a = 25) (hb : b = 64) (hc : c = 144) :
  3 / (1 / Real.sqrt a + 1 / Real.sqrt b + 1 / Real.sqrt c) = 360 / 49 := by
  sorry

end NUMINAMATH_CALUDE_harmonic_mean_of_square_sides_l3716_371609


namespace NUMINAMATH_CALUDE_probability_opposite_corner_is_one_third_l3716_371644

/-- Represents a cube with its properties -/
structure Cube where
  vertices : Fin 8
  faces : Fin 6

/-- Represents the ant's position on the cube -/
inductive Position
  | Corner : Fin 8 → Position

/-- Represents a single move of the ant -/
def Move : Type := Position → Position

/-- The probability of the ant ending at the diagonally opposite corner after two moves -/
def probability_opposite_corner (c : Cube) : ℚ :=
  1/3

/-- Theorem stating that the probability of ending at the diagonally opposite corner is 1/3 -/
theorem probability_opposite_corner_is_one_third (c : Cube) :
  probability_opposite_corner c = 1/3 := by sorry

end NUMINAMATH_CALUDE_probability_opposite_corner_is_one_third_l3716_371644


namespace NUMINAMATH_CALUDE_equation_solution_l3716_371698

theorem equation_solution : ∃ c : ℚ, (c - 23) / 2 = (2 * c + 5) / 7 ∧ c = 57 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3716_371698


namespace NUMINAMATH_CALUDE_proportional_function_decreasing_l3716_371628

theorem proportional_function_decreasing (x₁ x₂ : ℝ) (h : x₁ < x₂) : -2 * x₁ > -2 * x₂ := by
  sorry

end NUMINAMATH_CALUDE_proportional_function_decreasing_l3716_371628


namespace NUMINAMATH_CALUDE_transformations_map_correctly_l3716_371672

-- Define points in 2D space
def C : ℝ × ℝ := (3, -2)
def D : ℝ × ℝ := (4, -5)
def C' : ℝ × ℝ := (-3, 2)
def D' : ℝ × ℝ := (-4, 5)

-- Define translation
def translate (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - 6, p.2 + 4)

-- Define 180° clockwise rotation
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- Theorem stating that both transformations map C to C' and D to D'
theorem transformations_map_correctly :
  (translate C = C' ∧ translate D = D') ∧
  (rotate180 C = C' ∧ rotate180 D = D') :=
by sorry

end NUMINAMATH_CALUDE_transformations_map_correctly_l3716_371672


namespace NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_intersection_A_C_nonempty_implies_a_less_than_8_l3716_371692

-- Define the sets A, B, C, and U
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 6}
def C (a : ℝ) : Set ℝ := {x | x > a}
def U : Set ℝ := Set.univ

-- Theorem 1: A ∪ B = {x | 1 ≤ x ∧ x ≤ 8}
theorem union_A_B : A ∪ B = {x : ℝ | 1 ≤ x ∧ x ≤ 8} := by sorry

-- Theorem 2: (∁ₐA) ∩ B = {x | 1 ≤ x ∧ x < 2}
theorem complement_A_intersect_B : (Aᶜ) ∩ B = {x : ℝ | 1 ≤ x ∧ x < 2} := by sorry

-- Theorem 3: If A ∩ C ≠ ∅, then a < 8
theorem intersection_A_C_nonempty_implies_a_less_than_8 (a : ℝ) :
  (A ∩ C a).Nonempty → a < 8 := by sorry

end NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_intersection_A_C_nonempty_implies_a_less_than_8_l3716_371692


namespace NUMINAMATH_CALUDE_ellipse_foci_l3716_371670

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 25 = 1

-- Define the foci coordinates
def foci : Set (ℝ × ℝ) :=
  {(0, 3), (0, -3)}

-- Theorem statement
theorem ellipse_foci :
  ∀ (f : ℝ × ℝ), f ∈ foci ↔
    (∃ (x y : ℝ), ellipse_equation x y ∧
      (x - f.1)^2 + (y - f.2)^2 +
      (x + f.1)^2 + (y + f.2)^2 = 4 * (5^2 + 4^2)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_l3716_371670


namespace NUMINAMATH_CALUDE_cat_food_cans_per_package_l3716_371600

theorem cat_food_cans_per_package (cat_packages : ℕ) (dog_packages : ℕ) (dog_cans_per_package : ℕ) (cat_dog_difference : ℕ) :
  cat_packages = 9 →
  dog_packages = 7 →
  dog_cans_per_package = 5 →
  cat_dog_difference = 55 →
  ∃ (cat_cans_per_package : ℕ),
    cat_cans_per_package * cat_packages = dog_cans_per_package * dog_packages + cat_dog_difference ∧
    cat_cans_per_package = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_cat_food_cans_per_package_l3716_371600


namespace NUMINAMATH_CALUDE_p_toluidine_molecular_weight_l3716_371630

/-- The atomic weight of Carbon in atomic mass units (amu) -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of Hydrogen in atomic mass units (amu) -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of Nitrogen in atomic mass units (amu) -/
def nitrogen_weight : ℝ := 14.01

/-- The chemical formula of p-Toluidine -/
structure ChemicalFormula where
  carbon : ℕ
  hydrogen : ℕ
  nitrogen : ℕ

/-- The chemical formula of p-Toluidine (C7H9N) -/
def p_toluidine : ChemicalFormula := ⟨7, 9, 1⟩

/-- Calculate the molecular weight of a chemical compound given its formula -/
def molecular_weight (formula : ChemicalFormula) : ℝ :=
  formula.carbon * carbon_weight + 
  formula.hydrogen * hydrogen_weight + 
  formula.nitrogen * nitrogen_weight

/-- Theorem: The molecular weight of p-Toluidine is 107.152 amu -/
theorem p_toluidine_molecular_weight : 
  molecular_weight p_toluidine = 107.152 := by
  sorry

end NUMINAMATH_CALUDE_p_toluidine_molecular_weight_l3716_371630


namespace NUMINAMATH_CALUDE_intersection_range_l3716_371679

/-- The function f(x) = 3x - x^3 --/
def f (x : ℝ) : ℝ := 3*x - x^3

/-- The line y = m intersects the graph of f at three distinct points --/
def intersects_at_three_points (m : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    f x₁ = m ∧ f x₂ = m ∧ f x₃ = m

theorem intersection_range (m : ℝ) :
  intersects_at_three_points m → -2 < m ∧ m < 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_range_l3716_371679


namespace NUMINAMATH_CALUDE_total_distance_walked_l3716_371612

def first_part : ℝ := 0.75
def second_part : ℝ := 0.25

theorem total_distance_walked : first_part + second_part = 1 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_walked_l3716_371612


namespace NUMINAMATH_CALUDE_rebecca_eggs_count_l3716_371647

/-- Given that Rebecca wants to split eggs into 3 groups with 5 eggs in each group,
    prove that the total number of eggs is 15. -/
theorem rebecca_eggs_count :
  let num_groups : ℕ := 3
  let eggs_per_group : ℕ := 5
  num_groups * eggs_per_group = 15 := by sorry

end NUMINAMATH_CALUDE_rebecca_eggs_count_l3716_371647


namespace NUMINAMATH_CALUDE_arrangement_count_l3716_371614

def committee_size : ℕ := 12
def num_men : ℕ := 3
def num_women : ℕ := 9

theorem arrangement_count :
  (committee_size.choose num_men) = 220 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l3716_371614


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_no_solution_for_2009_l3716_371616

theorem cubic_equation_solutions (n : ℕ+) :
  (∃ (x y : ℤ), x^3 - 3*x*y^2 + y^3 = n) →
  (∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℤ),
    (x₁^3 - 3*x₁*y₁^2 + y₁^3 = n) ∧
    (x₂^3 - 3*x₂*y₂^2 + y₂^3 = n) ∧
    (x₃^3 - 3*x₃*y₃^2 + y₃^3 = n) ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₃, y₃)) :=
by sorry

theorem no_solution_for_2009 :
  ¬ ∃ (x y : ℤ), x^3 - 3*x*y^2 + y^3 = 2009 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_no_solution_for_2009_l3716_371616


namespace NUMINAMATH_CALUDE_unique_digit_divisibility_l3716_371613

def is_divisible (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

theorem unique_digit_divisibility :
  ∃! B : ℕ,
    B < 10 ∧
    let number := 658274 * 10 + B
    is_divisible number 2 ∧
    is_divisible number 4 ∧
    is_divisible number 5 ∧
    is_divisible number 7 ∧
    is_divisible number 8 :=
by sorry

end NUMINAMATH_CALUDE_unique_digit_divisibility_l3716_371613


namespace NUMINAMATH_CALUDE_angle_C_is_60_degrees_a_and_b_values_l3716_371653

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a + t.b = 5 ∧
  t.c = Real.sqrt 7 ∧
  4 * (Real.sin (t.A + t.B) / 2)^2 - Real.cos (2 * t.C) = 7/2 ∧
  t.A + t.B + t.C = Real.pi

-- Theorem 1: Prove that C = 60°
theorem angle_C_is_60_degrees (t : Triangle) 
  (h : triangle_conditions t) : t.C = Real.pi / 3 := by
  sorry

-- Theorem 2: Prove that if a > b, then a = 3 and b = 2
theorem a_and_b_values (t : Triangle) 
  (h1 : triangle_conditions t) (h2 : t.a > t.b) : t.a = 3 ∧ t.b = 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_is_60_degrees_a_and_b_values_l3716_371653


namespace NUMINAMATH_CALUDE_limit_f_at_infinity_l3716_371646

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ((a^x - 1) / (x * (a - 1)))^(1/x)

theorem limit_f_at_infinity (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ ε > 0, ∃ N, ∀ x ≥ N, |f a x - (if a > 1 then a else 1)| < ε) :=
sorry

end NUMINAMATH_CALUDE_limit_f_at_infinity_l3716_371646


namespace NUMINAMATH_CALUDE_no_real_roots_l3716_371673

def polynomial (x p : ℝ) : ℝ := x^4 + 4*p*x^3 + 6*x^2 + 4*p*x + 1

theorem no_real_roots (p : ℝ) : 
  (∀ x : ℝ, polynomial x p ≠ 0) ↔ p > -Real.sqrt 5 / 2 ∧ p < Real.sqrt 5 / 2 :=
sorry

end NUMINAMATH_CALUDE_no_real_roots_l3716_371673


namespace NUMINAMATH_CALUDE_problem_solution_l3716_371697

def even_squared_sum : ℕ := (2^2) + (4^2) + (6^2) + (8^2) + (10^2)

def prime_count : ℕ := 4

def odd_product : ℕ := 1 * 3 * 5 * 7 * 9

theorem problem_solution :
  let x := even_squared_sum
  let y := prime_count
  let z := odd_product
  x - y + z = 1161 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3716_371697


namespace NUMINAMATH_CALUDE_hockey_helmets_l3716_371624

theorem hockey_helmets (red blue : ℕ) : 
  red = blue + 6 → 
  red * 3 = blue * 5 → 
  red + blue = 24 := by
sorry

end NUMINAMATH_CALUDE_hockey_helmets_l3716_371624


namespace NUMINAMATH_CALUDE_percentage_to_pass_l3716_371621

/-- Calculates the percentage of total marks needed to pass an exam -/
theorem percentage_to_pass (marks_obtained : ℕ) (marks_short : ℕ) (total_marks : ℕ) :
  marks_obtained = 125 →
  marks_short = 40 →
  total_marks = 500 →
  (((marks_obtained + marks_short : ℚ) / total_marks) * 100 : ℚ) = 33 := by
  sorry

end NUMINAMATH_CALUDE_percentage_to_pass_l3716_371621


namespace NUMINAMATH_CALUDE_ceiling_floor_product_range_l3716_371677

theorem ceiling_floor_product_range (y : ℝ) : 
  y < -1 → ⌈y⌉ * ⌊y⌋ = 72 → -9 < y ∧ y < -8 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_product_range_l3716_371677


namespace NUMINAMATH_CALUDE_job_completion_time_l3716_371667

/-- The time taken for a, b, and c to finish a job together, given the conditions. -/
theorem job_completion_time (a b c : ℝ) : 
  (a + b = 1 / 15) →  -- a and b finish the job in 15 days
  (c = 1 / 7.5) →     -- c alone finishes the job in 7.5 days
  (1 / (a + b + c) = 5) :=  -- a, b, and c together finish the job in 5 days
by sorry

end NUMINAMATH_CALUDE_job_completion_time_l3716_371667


namespace NUMINAMATH_CALUDE_matilda_earnings_l3716_371638

/-- Calculates the total earnings for a newspaper delivery job -/
def calculate_earnings (hourly_wage : ℚ) (per_newspaper : ℚ) (newspapers_per_hour : ℕ) (shift_duration : ℕ) : ℚ :=
  let wage_earnings := hourly_wage * shift_duration
  let newspaper_earnings := per_newspaper * newspapers_per_hour * shift_duration
  wage_earnings + newspaper_earnings

/-- Proves that Matilda's earnings for a 3-hour shift equal $40.50 -/
theorem matilda_earnings : 
  calculate_earnings 6 (1/4) 30 3 = 81/2 := by
  sorry

#eval calculate_earnings 6 (1/4) 30 3

end NUMINAMATH_CALUDE_matilda_earnings_l3716_371638


namespace NUMINAMATH_CALUDE_sunflower_seed_distribution_l3716_371682

theorem sunflower_seed_distribution (total_seeds : ℝ) (num_cans : ℝ) (seeds_per_can : ℝ) 
  (h1 : total_seeds = 54.0)
  (h2 : num_cans = 9.0)
  (h3 : seeds_per_can = total_seeds / num_cans) :
  seeds_per_can = 6.0 := by
sorry

end NUMINAMATH_CALUDE_sunflower_seed_distribution_l3716_371682


namespace NUMINAMATH_CALUDE_largest_59_double_l3716_371635

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a number from base 10 to base 9 -/
def base10ToBase9 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 9) ((m % 9) :: acc)
    aux n []

/-- Checks if a number is a 5-9 double -/
def is59Double (m : Nat) : Prop :=
  let base5Digits := base10ToBase9 m
  let base9Value := base5ToBase10 base5Digits
  base9Value = 2 * m

theorem largest_59_double :
  ∀ n : Nat, n > 20 → ¬(is59Double n) ∧ is59Double 20 :=
sorry

end NUMINAMATH_CALUDE_largest_59_double_l3716_371635


namespace NUMINAMATH_CALUDE_complement_of_P_l3716_371683

def P : Set ℝ := {x | |x + 3| + |x + 6| = 3}

theorem complement_of_P : 
  {x : ℝ | x < -6 ∨ x > -3} = (Set.univ : Set ℝ) \ P := by sorry

end NUMINAMATH_CALUDE_complement_of_P_l3716_371683


namespace NUMINAMATH_CALUDE_animals_on_shore_l3716_371659

/-- Proves that given the initial numbers of animals and the conditions about drowning,
    the total number of animals that made it to shore is 35. -/
theorem animals_on_shore (initial_sheep initial_cows initial_dogs : ℕ)
                         (drowned_sheep : ℕ)
                         (h1 : initial_sheep = 20)
                         (h2 : initial_cows = 10)
                         (h3 : initial_dogs = 14)
                         (h4 : drowned_sheep = 3)
                         (h5 : drowned_sheep * 2 = initial_cows - (initial_cows - drowned_sheep * 2)) :
  initial_sheep - drowned_sheep + (initial_cows - drowned_sheep * 2) + initial_dogs = 35 := by
  sorry

#check animals_on_shore

end NUMINAMATH_CALUDE_animals_on_shore_l3716_371659


namespace NUMINAMATH_CALUDE_certain_number_value_value_is_232_l3716_371645

theorem certain_number_value : ℤ → ℤ → Prop :=
  fun n value => 5 * n - 28 = value

theorem value_is_232 (n : ℤ) (value : ℤ) 
  (h1 : n = 52) 
  (h2 : certain_number_value n value) : 
  value = 232 := by
sorry

end NUMINAMATH_CALUDE_certain_number_value_value_is_232_l3716_371645


namespace NUMINAMATH_CALUDE_increasing_function_condition_l3716_371671

/-- The function f(x) = x^2 + ax + 1/x is increasing on (1/3, +∞) if and only if a ≥ 25/3 -/
theorem increasing_function_condition (a : ℝ) :
  (∀ x > 1/3, Monotone (fun x => x^2 + a*x + 1/x)) ↔ a ≥ 25/3 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_condition_l3716_371671


namespace NUMINAMATH_CALUDE_equal_area_segment_property_l3716_371626

/-- A trapezoid with the given properties -/
structure Trapezoid where
  b : ℝ  -- Length of the shorter base
  h : ℝ  -- Height of the trapezoid
  area_ratio : (b + (b + 75)) / (b + 75 + (b + 150)) = 1 / 2  -- Midpoint segment divides areas in ratio 1:2

/-- The length of the segment parallel to bases dividing the trapezoid into equal areas -/
def equal_area_segment (t : Trapezoid) : ℝ :=
  let x : ℝ := sorry
  x

/-- The main theorem -/
theorem equal_area_segment_property (t : Trapezoid) :
  ⌊(2812.5 + 112.5 * equal_area_segment t) / 100⌋ = ⌊(equal_area_segment t)^2 / 100⌋ := by
  sorry

end NUMINAMATH_CALUDE_equal_area_segment_property_l3716_371626


namespace NUMINAMATH_CALUDE_romans_remaining_coins_l3716_371654

/-- Represents the problem of calculating Roman's remaining gold coins --/
theorem romans_remaining_coins 
  (initial_worth : ℕ) 
  (coins_sold : ℕ) 
  (money_after_sale : ℕ) 
  (h1 : initial_worth = 20)
  (h2 : coins_sold = 3)
  (h3 : money_after_sale = 12) :
  initial_worth / (money_after_sale / coins_sold) - coins_sold = 2 :=
sorry

end NUMINAMATH_CALUDE_romans_remaining_coins_l3716_371654


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3716_371639

theorem condition_necessary_not_sufficient :
  (∀ x : ℝ, x > 0 → x ≠ 0) ∧
  (∃ x : ℝ, x ≠ 0 ∧ ¬(x > 0)) := by
  sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3716_371639


namespace NUMINAMATH_CALUDE_range_of_linear_function_l3716_371625

theorem range_of_linear_function (c : ℝ) (h : c ≠ 0) :
  let g : ℝ → ℝ := λ x ↦ c * x + 2
  let domain := Set.Icc (-1 : ℝ) 2
  let range := Set.image g domain
  range = if c > 0 
    then Set.Icc (-c + 2) (2 * c + 2)
    else Set.Icc (2 * c + 2) (-c + 2) := by
  sorry

end NUMINAMATH_CALUDE_range_of_linear_function_l3716_371625


namespace NUMINAMATH_CALUDE_max_sin_x_sin_2x_l3716_371658

theorem max_sin_x_sin_2x (x : Real) (h : 0 < x ∧ x < π / 2) :
  (∀ y : Real, y = Real.sin x * Real.sin (2 * x) → y ≤ 4 * Real.sqrt 3 / 9) ∧
  (∃ y : Real, y = Real.sin x * Real.sin (2 * x) ∧ y = 4 * Real.sqrt 3 / 9) :=
sorry

end NUMINAMATH_CALUDE_max_sin_x_sin_2x_l3716_371658


namespace NUMINAMATH_CALUDE_mountain_loop_trail_length_l3716_371690

/-- The Mountain Loop Trail Theorem -/
theorem mountain_loop_trail_length 
  (x1 x2 x3 x4 x5 : ℝ) 
  (h1 : x1 + x2 = 36)
  (h2 : x2 + x3 = 40)
  (h3 : x3 + x4 + x5 = 45)
  (h4 : x1 + x4 = 38) :
  x1 + x2 + x3 + x4 + x5 = 81 := by
  sorry

#check mountain_loop_trail_length

end NUMINAMATH_CALUDE_mountain_loop_trail_length_l3716_371690


namespace NUMINAMATH_CALUDE_geometric_series_problem_l3716_371684

theorem geometric_series_problem (x y : ℝ) (h : y ≠ 1.375) :
  (∑' n, x / y^n) = 10 →
  (∑' n, x / (x - 2*y)^n) = 5/4 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_problem_l3716_371684


namespace NUMINAMATH_CALUDE_cubic_quadratic_relation_l3716_371602

theorem cubic_quadratic_relation (A B C D : ℝ) (p q r : ℝ) (a b : ℝ) : 
  (A * p^3 + B * p^2 + C * p + D = 0) →
  (A * q^3 + B * q^2 + C * q + D = 0) →
  (A * r^3 + B * r^2 + C * r + D = 0) →
  ((p^2 + q)^2 + a * (p^2 + q) + b = 0) →
  ((q^2 + r)^2 + a * (q^2 + r) + b = 0) →
  ((r^2 + p)^2 + a * (r^2 + p) + b = 0) →
  (A ≠ 0) →
  a = (A * B + 2 * A * C - B^2) / A^2 := by
sorry

end NUMINAMATH_CALUDE_cubic_quadratic_relation_l3716_371602


namespace NUMINAMATH_CALUDE_average_age_when_youngest_born_l3716_371695

theorem average_age_when_youngest_born 
  (n : ℕ) 
  (current_avg : ℝ) 
  (youngest_age : ℝ) 
  (sum_others_at_birth : ℝ) 
  (h1 : n = 7) 
  (h2 : current_avg = 30) 
  (h3 : youngest_age = 6) 
  (h4 : sum_others_at_birth = 150) : 
  (sum_others_at_birth / n : ℝ) = 150 / 7 := by
sorry

end NUMINAMATH_CALUDE_average_age_when_youngest_born_l3716_371695


namespace NUMINAMATH_CALUDE_inequality_solution_l3716_371606

open Set Real

def inequality_holds (x a : ℝ) : Prop :=
  (a + 2) * x - (1 + 2 * a) * (x^2)^(1/3) - 6 * x^(1/3) + a^2 + 4 * a - 5 > 0

theorem inequality_solution :
  ∀ x : ℝ, (∃ a ∈ Icc (-2) 1, inequality_holds x a) ↔ 
  x ∈ Iio (-1) ∪ Ioo (-1) 0 ∪ Ioi 8 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3716_371606


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_reciprocals_l3716_371603

theorem min_value_of_sum_of_reciprocals (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 2) :
  (1 / (x + y) + 1 / (x + z) + 1 / (y + z)) ≥ 9/4 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_reciprocals_l3716_371603


namespace NUMINAMATH_CALUDE_power_division_rule_l3716_371665

theorem power_division_rule (a : ℝ) : a^7 / a^5 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l3716_371665


namespace NUMINAMATH_CALUDE_symmetric_points_on_parabola_l3716_371656

/-- Two points on a parabola, symmetric with respect to a line -/
theorem symmetric_points_on_parabola (x₁ x₂ y₁ y₂ m : ℝ) :
  y₁ = 2 * x₁^2 →                          -- A is on the parabola
  y₂ = 2 * x₂^2 →                          -- B is on the parabola
  (y₂ - y₁) / (x₂ - x₁) = -1 →             -- A and B are symmetric (slope condition)
  (y₂ + y₁) / 2 = (x₂ + x₁) / 2 + m →      -- Midpoint of A and B lies on y = x + m
  x₁ * x₂ = -3/4 →                         -- Given condition
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_on_parabola_l3716_371656


namespace NUMINAMATH_CALUDE_range_sum_bounds_l3716_371662

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + 4*x

-- State the theorem
theorem range_sum_bounds :
  ∃ (m n : ℝ), (∀ x, m ≤ f x ∧ f x ≤ n) ∧
  (m = -5 ∧ n = 4) →
  1 ≤ m + n ∧ m + n ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_range_sum_bounds_l3716_371662


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l3716_371623

theorem no_solution_for_equation : ¬∃ (a b : ℕ+), 
  a * b + 100 = 25 * Nat.lcm a b + 15 * Nat.gcd a b := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l3716_371623


namespace NUMINAMATH_CALUDE_pics_per_album_l3716_371699

-- Define the given conditions
def phone_pics : ℕ := 35
def camera_pics : ℕ := 5
def num_albums : ℕ := 5

-- Define the total number of pictures
def total_pics : ℕ := phone_pics + camera_pics

-- Theorem to prove
theorem pics_per_album : total_pics / num_albums = 8 := by
  sorry

end NUMINAMATH_CALUDE_pics_per_album_l3716_371699


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l3716_371693

def f (x : ℝ) : ℝ := x^3 - 4*x^2 - x + 6

theorem roots_of_polynomial :
  (f (-1) = 0) ∧ (f 1 = 0) ∧ (f 6 = 0) ∧
  (∀ x : ℝ, f x = 0 → x = -1 ∨ x = 1 ∨ x = 6) :=
sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l3716_371693


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3716_371691

theorem sum_of_coefficients (b₆ b₅ b₄ b₃ b₂ b₁ b₀ : ℝ) :
  (∀ x : ℝ, (5*x - 2)^6 = b₆*x^6 + b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₆ + b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 729 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3716_371691


namespace NUMINAMATH_CALUDE_union_M_N_l3716_371666

-- Define the sets M and N
def M : Set ℝ := {x | -2 < x ∧ x < -1}
def N : Set ℝ := {x | (1/2 : ℝ)^x ≤ 4}

-- State the theorem
theorem union_M_N : M ∪ N = {x | x ≥ -2} := by sorry

end NUMINAMATH_CALUDE_union_M_N_l3716_371666


namespace NUMINAMATH_CALUDE_juanita_dessert_cost_l3716_371619

/-- Represents the menu prices and discounts for the brownie dessert --/
structure BrownieMenu where
  brownie_base : ℝ := 2.50
  regular_scoop : ℝ := 1.00
  premium_scoop : ℝ := 1.25
  deluxe_scoop : ℝ := 1.50
  syrup : ℝ := 0.50
  nuts : ℝ := 1.50
  whipped_cream : ℝ := 0.75
  cherry : ℝ := 0.25
  tuesday_discount : ℝ := 0.10
  wednesday_discount : ℝ := 0.50
  sunday_discount : ℝ := 0.25

/-- Represents Juanita's order --/
structure JuanitaOrder where
  regular_scoops : ℕ := 2
  premium_scoops : ℕ := 1
  deluxe_scoops : ℕ := 0
  syrups : ℕ := 2
  has_nuts : Bool := true
  has_whipped_cream : Bool := true
  has_cherry : Bool := true

/-- Calculates the total cost of Juanita's dessert --/
def calculate_total_cost (menu : BrownieMenu) (order : JuanitaOrder) : ℝ :=
  let discounted_brownie := menu.brownie_base * (1 - menu.tuesday_discount)
  let ice_cream_cost := order.regular_scoops * menu.regular_scoop + 
                        order.premium_scoops * menu.premium_scoop + 
                        order.deluxe_scoops * menu.deluxe_scoop
  let syrup_cost := order.syrups * menu.syrup
  let topping_cost := (if order.has_nuts then menu.nuts else 0) +
                      (if order.has_whipped_cream then menu.whipped_cream else 0) +
                      (if order.has_cherry then menu.cherry else 0)
  discounted_brownie + ice_cream_cost + syrup_cost + topping_cost

/-- Theorem stating that Juanita's dessert costs $9.00 --/
theorem juanita_dessert_cost (menu : BrownieMenu) (order : JuanitaOrder) :
  calculate_total_cost menu order = 9.00 := by
  sorry


end NUMINAMATH_CALUDE_juanita_dessert_cost_l3716_371619


namespace NUMINAMATH_CALUDE_expected_value_of_event_A_l3716_371643

theorem expected_value_of_event_A (p : ℝ) (h1 : (1 - p)^4 = 16/81) :
  4 * p = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_of_event_A_l3716_371643


namespace NUMINAMATH_CALUDE_ninth_grade_science_only_l3716_371637

/-- Represents the set of all ninth-grade students -/
def NinthGrade : Finset Nat := sorry

/-- Represents the set of students in the science class -/
def ScienceClass : Finset Nat := sorry

/-- Represents the set of students in the history class -/
def HistoryClass : Finset Nat := sorry

theorem ninth_grade_science_only :
  (NinthGrade.card = 120) →
  (ScienceClass.card = 85) →
  (HistoryClass.card = 75) →
  (NinthGrade = ScienceClass ∪ HistoryClass) →
  ((ScienceClass \ HistoryClass).card = 45) := by
  sorry

end NUMINAMATH_CALUDE_ninth_grade_science_only_l3716_371637


namespace NUMINAMATH_CALUDE_max_sum_squares_l3716_371610

theorem max_sum_squares (m n : ℕ) : 
  m ∈ Finset.range 1982 →
  n ∈ Finset.range 1982 →
  ((n^2 : ℤ) - m*n - m^2)^2 = 1 →
  m^2 + n^2 ≤ 3524578 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_squares_l3716_371610


namespace NUMINAMATH_CALUDE_red_card_count_l3716_371627

theorem red_card_count (red_credit blue_credit total_cards total_credit : ℕ) 
  (h1 : red_credit = 3)
  (h2 : blue_credit = 5)
  (h3 : total_cards = 20)
  (h4 : total_credit = 84) :
  ∃ (red_cards blue_cards : ℕ),
    red_cards + blue_cards = total_cards ∧
    red_credit * red_cards + blue_credit * blue_cards = total_credit ∧
    red_cards = 8 := by
  sorry

end NUMINAMATH_CALUDE_red_card_count_l3716_371627


namespace NUMINAMATH_CALUDE_nth_equation_l3716_371680

theorem nth_equation (n : ℕ) (hn : n > 0) :
  1 / (n + 2 : ℚ) + 2 / (n^2 + 2*n : ℚ) = 1 / n :=
by sorry

end NUMINAMATH_CALUDE_nth_equation_l3716_371680


namespace NUMINAMATH_CALUDE_solve_equation_l3716_371615

theorem solve_equation (x : ℝ) : (2 * x + 7) / 7 = 13 → x = 42 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3716_371615


namespace NUMINAMATH_CALUDE_cost_per_box_is_three_fifty_l3716_371681

/-- The cost per box of wafer cookies -/
def cost_per_box (num_trays : ℕ) (cookies_per_tray : ℕ) (cookies_per_box : ℕ) (total_cost : ℚ) : ℚ :=
  total_cost / (((num_trays * cookies_per_tray) + cookies_per_box - 1) / cookies_per_box)

/-- Theorem stating that the cost per box is $3.50 given the problem conditions -/
theorem cost_per_box_is_three_fifty :
  cost_per_box 3 80 60 14 = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_box_is_three_fifty_l3716_371681


namespace NUMINAMATH_CALUDE_aubrey_distance_to_school_l3716_371678

/-- The distance from Aubrey's home to his school -/
def distance_to_school (journey_time : ℝ) (average_speed : ℝ) : ℝ :=
  journey_time * average_speed

/-- Theorem stating the distance from Aubrey's home to his school -/
theorem aubrey_distance_to_school :
  distance_to_school 4 22 = 88 := by
  sorry

end NUMINAMATH_CALUDE_aubrey_distance_to_school_l3716_371678


namespace NUMINAMATH_CALUDE_line_circle_no_intersection_l3716_371664

theorem line_circle_no_intersection (a : ℝ) :
  (∀ x y : ℝ, x + y = a → x^2 + y^2 ≠ 1) ↔ (a > 1 ∨ a < -1) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_no_intersection_l3716_371664


namespace NUMINAMATH_CALUDE_equation_solution_l3716_371694

theorem equation_solution : ∃ x : ℝ, (2 / x = 1 / (x + 1)) ∧ (x = -2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3716_371694


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_m_value_l3716_371687

theorem quadratic_roots_imply_m_value (m : ℝ) : 
  (∃ x : ℂ, 5 * x^2 - 4 * x + m = 0 ∧ 
   (x = (2 + Complex.I * Real.sqrt 143) / 5 ∨ 
    x = (2 - Complex.I * Real.sqrt 143) / 5)) → 
  m = 7.95 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_m_value_l3716_371687


namespace NUMINAMATH_CALUDE_smallest_positive_b_l3716_371668

/-- A function with period 30 -/
def is_periodic_30 (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x - 30) = g x

/-- The property we want to prove for the smallest positive b -/
def has_property (g : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x, g ((x - b) / 10) = g (x / 10)

theorem smallest_positive_b (g : ℝ → ℝ) (h : is_periodic_30 g) :
  ∃ b : ℝ, b > 0 ∧ has_property g b ∧ ∀ b' : ℝ, 0 < b' ∧ has_property g b' → b ≤ b' :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_b_l3716_371668
