import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_coefficient_l427_42713

theorem quadratic_coefficient (b : ℝ) (n : ℝ) : 
  (∀ x, x^2 + b*x + 56 = (x + n)^2 + 12) → 
  b > 0 → 
  b = 4 * Real.sqrt 11 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l427_42713


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l427_42759

theorem divisibility_equivalence (m n : ℕ) : 
  (((2^m : ℕ) - 1)^2 ∣ ((2^n : ℕ) - 1)) ↔ (m * ((2^m : ℕ) - 1) ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l427_42759


namespace NUMINAMATH_CALUDE_tom_has_24_blue_marbles_l427_42795

/-- The number of blue marbles Jason has -/
def jason_blue_marbles : ℕ := 44

/-- The difference between Jason's and Tom's blue marbles -/
def marble_difference : ℕ := 20

/-- The number of blue marbles Tom has -/
def tom_blue_marbles : ℕ := jason_blue_marbles - marble_difference

theorem tom_has_24_blue_marbles : tom_blue_marbles = 24 := by
  sorry

end NUMINAMATH_CALUDE_tom_has_24_blue_marbles_l427_42795


namespace NUMINAMATH_CALUDE_three_digit_number_problem_l427_42794

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def hundreds_digit (n : ℕ) : ℕ :=
  (n / 100) % 10

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

def units_digit (n : ℕ) : ℕ :=
  n % 10

def swap_hundreds_units (n : ℕ) : ℕ :=
  (units_digit n) * 100 + (tens_digit n) * 10 + (hundreds_digit n)

theorem three_digit_number_problem :
  ∀ n : ℕ, is_three_digit_number n →
    (tens_digit n)^2 = (hundreds_digit n) * (units_digit n) →
    n - (swap_hundreds_units n) = 297 →
    (n = 300 ∨ n = 421) :=
sorry

end NUMINAMATH_CALUDE_three_digit_number_problem_l427_42794


namespace NUMINAMATH_CALUDE_derivative_sin_cos_l427_42719

theorem derivative_sin_cos (x : ℝ) :
  deriv (λ x => 3 * Real.sin x - 4 * Real.cos x) x = 3 * Real.cos x + 4 * Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_cos_l427_42719


namespace NUMINAMATH_CALUDE_sum_opposite_and_sqrt_81_l427_42749

-- Define the function for the sum
def sum_opposite_and_sqrt : ℝ → Set ℝ :=
  λ x => {2 + Real.sqrt 2, Real.sqrt 2 - 4}

-- State the theorem
theorem sum_opposite_and_sqrt_81 :
  sum_opposite_and_sqrt (Real.sqrt 81) = {2 + Real.sqrt 2, Real.sqrt 2 - 4} :=
by sorry

end NUMINAMATH_CALUDE_sum_opposite_and_sqrt_81_l427_42749


namespace NUMINAMATH_CALUDE_first_agency_less_expensive_l427_42784

/-- The number of miles at which the first agency becomes less expensive than the second -/
def miles_threshold : ℝ := 25

/-- The daily rate for the first agency -/
def daily_rate_1 : ℝ := 20.25

/-- The per-mile rate for the first agency -/
def mile_rate_1 : ℝ := 0.14

/-- The daily rate for the second agency -/
def daily_rate_2 : ℝ := 18.25

/-- The per-mile rate for the second agency -/
def mile_rate_2 : ℝ := 0.22

/-- Theorem stating that the first agency is less expensive when miles driven exceed the threshold -/
theorem first_agency_less_expensive (miles : ℝ) (days : ℝ) 
  (h : miles > miles_threshold) : 
  daily_rate_1 * days + mile_rate_1 * miles < daily_rate_2 * days + mile_rate_2 * miles :=
by
  sorry


end NUMINAMATH_CALUDE_first_agency_less_expensive_l427_42784


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l427_42726

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_one : a + b + c + d = 1) :
  (b * c * d / (1 - a)^2) + (a * c * d / (1 - b)^2) + 
  (a * b * d / (1 - c)^2) + (a * b * c / (1 - d)^2) ≤ 1/9 ∧
  ((b * c * d / (1 - a)^2) + (a * c * d / (1 - b)^2) + 
   (a * b * d / (1 - c)^2) + (a * b * c / (1 - d)^2) = 1/9 ↔ 
   a = 1/4 ∧ b = 1/4 ∧ c = 1/4 ∧ d = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l427_42726


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l427_42735

theorem sum_of_reciprocals_of_roots (x : ℝ) : 
  x^2 - 17*x + 8 = 0 → 
  ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ x^2 - 17*x + 8 = (x - r₁) * (x - r₂) ∧ 
  (1 / r₁ + 1 / r₂ = 17 / 8) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l427_42735


namespace NUMINAMATH_CALUDE_hyperbola_equation_l427_42760

/-- A hyperbola with center at the origin, focus at (3,0), and a line passing through
    the focus intersecting the hyperbola at two points whose midpoint is (-12,-15) -/
structure Hyperbola where
  /-- The equation of the hyperbola in the form x²/a² - y²/b² = 1 -/
  equation : ℝ → ℝ → Prop
  /-- The center of the hyperbola is at the origin -/
  center_at_origin : equation 0 0
  /-- One focus of the hyperbola is at (3,0) -/
  focus_at_3_0 : ∃ (x y : ℝ), equation x y ∧ (x - 3)^2 + y^2 = (x + 3)^2 + y^2
  /-- There exists a line passing through (3,0) that intersects the hyperbola at two points -/
  intersecting_line : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    equation x₁ y₁ ∧ equation x₂ y₂ ∧ 
    (y₁ - 0) / (x₁ - 3) = (y₂ - 0) / (x₂ - 3)
  /-- The midpoint of the two intersection points is (-12,-15) -/
  midpoint : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    equation x₁ y₁ ∧ equation x₂ y₂ ∧
    (x₁ + x₂) / 2 = -12 ∧ (y₁ + y₂) / 2 = -15

/-- The equation of the hyperbola is x²/4 - y²/5 = 1 -/
theorem hyperbola_equation (h : Hyperbola) : 
  h.equation = fun x y => x^2 / 4 - y^2 / 5 = 1 := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l427_42760


namespace NUMINAMATH_CALUDE_parallel_non_coincident_lines_l427_42707

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 b1 b2 : ℝ} :
  (∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- Two lines are distinct if and only if their y-intercepts are different -/
axiom distinct_lines_different_intercepts {m b1 b2 : ℝ} :
  (∃ x y : ℝ, y = m * x + b1 ∧ y ≠ m * x + b2) ↔ b1 ≠ b2

theorem parallel_non_coincident_lines (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 6 = 0 ↔ y = -a/2 * x - 3) ∧
  (∀ x y : ℝ, x + (a - 1) * y + (a^2 - 1) = 0 ↔ y = -1/(a-1) * x - (a^2-1)/(a-1)) ∧
  (∀ x y : ℝ, y = -a/2 * x - 3 ↔ y = -1/(a-1) * x - (a^2-1)/(a-1)) ∧
  (∃ x y : ℝ, y = -a/2 * x - 3 ∧ y ≠ -1/(a-1) * x - (a^2-1)/(a-1)) →
  a = -1 := by sorry

end NUMINAMATH_CALUDE_parallel_non_coincident_lines_l427_42707


namespace NUMINAMATH_CALUDE_remainder_123456789012_mod_252_l427_42792

theorem remainder_123456789012_mod_252 : 
  123456789012 % 252 = 156 := by sorry

end NUMINAMATH_CALUDE_remainder_123456789012_mod_252_l427_42792


namespace NUMINAMATH_CALUDE_multiplication_mistake_l427_42740

theorem multiplication_mistake (number : ℕ) (correct_multiplier : ℕ) (mistaken_multiplier : ℕ) :
  number = 138 →
  correct_multiplier = 43 →
  mistaken_multiplier = 34 →
  (number * correct_multiplier) - (number * mistaken_multiplier) = 1242 := by
sorry

end NUMINAMATH_CALUDE_multiplication_mistake_l427_42740


namespace NUMINAMATH_CALUDE_tegwens_family_size_l427_42717

theorem tegwens_family_size :
  ∀ (g b : ℕ),
  g > 0 →  -- At least one girl (Tegwen)
  b = g - 1 →  -- Tegwen has same number of brothers as sisters
  g = (3 * (b - 1)) / 2 →  -- Each brother has 50% more sisters than brothers
  g + b = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_tegwens_family_size_l427_42717


namespace NUMINAMATH_CALUDE_lauren_mail_total_l427_42721

/-- The number of pieces of mail Lauren sent on Monday -/
def monday : ℕ := 65

/-- The number of pieces of mail Lauren sent on Tuesday -/
def tuesday : ℕ := monday + 10

/-- The number of pieces of mail Lauren sent on Wednesday -/
def wednesday : ℕ := tuesday - 5

/-- The number of pieces of mail Lauren sent on Thursday -/
def thursday : ℕ := wednesday + 15

/-- The total number of pieces of mail Lauren sent over four days -/
def total : ℕ := monday + tuesday + wednesday + thursday

theorem lauren_mail_total : total = 295 := by sorry

end NUMINAMATH_CALUDE_lauren_mail_total_l427_42721


namespace NUMINAMATH_CALUDE_envelope_game_properties_l427_42765

/-- A game with envelopes and two evenly matched teams -/
structure EnvelopeGame where
  num_envelopes : ℕ
  win_points : ℕ
  win_probability : ℝ

/-- Calculate the expected number of points for one team in a single game -/
noncomputable def expected_points (game : EnvelopeGame) : ℝ :=
  sorry

/-- Calculate the probability of a specific envelope being chosen in a game -/
noncomputable def envelope_probability (game : EnvelopeGame) : ℝ :=
  sorry

/-- Theorem about the expected points and envelope probability in the specific game -/
theorem envelope_game_properties :
  let game : EnvelopeGame := ⟨13, 6, 1/2⟩
  (100 * expected_points game = 465) ∧
  (envelope_probability game = 12/13) := by
  sorry

end NUMINAMATH_CALUDE_envelope_game_properties_l427_42765


namespace NUMINAMATH_CALUDE_circle_area_greater_than_five_times_triangle_area_l427_42738

theorem circle_area_greater_than_five_times_triangle_area 
  (R r : ℝ) (S : ℝ) (h_R_positive : R > 0) (h_r_positive : r > 0) (h_S_positive : S > 0)
  (h_R_r : R ≥ 2 * r) -- Euler's inequality
  (h_S : S ≤ (3 * Real.sqrt 3 / 2) * R * r) -- Upper bound for triangle area
  : π * (R + r)^2 > 5 * S := by
  sorry

end NUMINAMATH_CALUDE_circle_area_greater_than_five_times_triangle_area_l427_42738


namespace NUMINAMATH_CALUDE_tangent_line_value_l427_42771

/-- A function f: ℝ → ℝ is tangent to the line y = -x + 8 at x = 5 if:
    1. f(5) = -5 + 8
    2. f'(5) = -1
-/
def is_tangent_at_5 (f : ℝ → ℝ) : Prop :=
  f 5 = 3 ∧ deriv f 5 = -1

theorem tangent_line_value (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h_tangent : is_tangent_at_5 f) : f 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_value_l427_42771


namespace NUMINAMATH_CALUDE_seventh_data_entry_is_18_l427_42775

-- Define the given conditions
def total_results : ℕ := 15
def total_average : ℚ := 60
def first_set_count : ℕ := 7
def first_set_average : ℚ := 56
def second_set_count : ℕ := 6
def second_set_average : ℚ := 63
def last_set_count : ℕ := 6
def last_set_average : ℚ := 66

-- Theorem to prove
theorem seventh_data_entry_is_18 :
  ∃ (x : ℚ),
    x = 18 ∧
    total_average * total_results =
      first_set_average * first_set_count +
      second_set_average * second_set_count +
      x +
      (last_set_average * last_set_count - second_set_average * second_set_count - x) :=
by sorry

end NUMINAMATH_CALUDE_seventh_data_entry_is_18_l427_42775


namespace NUMINAMATH_CALUDE_larger_integer_value_l427_42704

theorem larger_integer_value (a b : ℕ+) 
  (h1 : (a : ℝ) / (b : ℝ) = 7 / 3) 
  (h2 : (a : ℕ) * b = 294) : 
  max a b = ⌈7 * Real.sqrt 14⌉ := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_value_l427_42704


namespace NUMINAMATH_CALUDE_reciprocal_difference_fractions_l427_42764

theorem reciprocal_difference_fractions : (1 / (1/4 - 1/5) : ℚ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_difference_fractions_l427_42764


namespace NUMINAMATH_CALUDE_apartment_building_floors_l427_42799

/-- Represents an apartment building with the given specifications. -/
structure ApartmentBuilding where
  floors : ℕ
  apartments_per_floor : ℕ
  people_per_apartment : ℕ
  total_people : ℕ

/-- Calculates the number of people on a full floor. -/
def people_on_full_floor (building : ApartmentBuilding) : ℕ :=
  building.apartments_per_floor * building.people_per_apartment

/-- Calculates the number of people on a half-capacity floor. -/
def people_on_half_capacity_floor (building : ApartmentBuilding) : ℕ :=
  (building.apartments_per_floor / 2) * building.people_per_apartment

/-- Theorem stating that given the conditions, the apartment building has 12 floors. -/
theorem apartment_building_floors
  (building : ApartmentBuilding)
  (h1 : building.apartments_per_floor = 10)
  (h2 : building.people_per_apartment = 4)
  (h3 : building.total_people = 360)
  (h4 : building.total_people = 
    (building.floors / 2 * people_on_full_floor building) + 
    (building.floors / 2 * people_on_half_capacity_floor building)) :
  building.floors = 12 := by
  sorry


end NUMINAMATH_CALUDE_apartment_building_floors_l427_42799


namespace NUMINAMATH_CALUDE_selection_ways_l427_42718

theorem selection_ways (male_count female_count : ℕ) 
  (h1 : male_count = 5)
  (h2 : female_count = 4) :
  male_count + female_count = 9 := by
  sorry

end NUMINAMATH_CALUDE_selection_ways_l427_42718


namespace NUMINAMATH_CALUDE_language_selection_theorem_l427_42748

theorem language_selection_theorem (n : ℕ) :
  ∀ (employees : Finset (Finset ℕ)),
    (employees.card = 500) →
    (∀ e ∈ employees, e ⊆ Finset.range (2 * n)) →
    (∀ e ∈ employees, e.card ≥ n) →
    ∃ (selected : Finset ℕ),
      selected.card = 14 ∧
      selected ⊆ Finset.range (2 * n) ∧
      ∀ e ∈ employees, ∃ l ∈ selected, l ∈ e :=
by sorry

end NUMINAMATH_CALUDE_language_selection_theorem_l427_42748


namespace NUMINAMATH_CALUDE_product_ratio_theorem_l427_42793

theorem product_ratio_theorem (a b c d e f : ℝ) 
  (h1 : a * b * c = 195)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 1000)
  (h4 : d * e * f = 250) :
  (a * f) / (c * d) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_product_ratio_theorem_l427_42793


namespace NUMINAMATH_CALUDE_arrangement_count_is_correct_l427_42750

/-- The number of ways to arrange 3 individuals on 7 steps -/
def arrangement_count : ℕ := 336

/-- The number of steps -/
def num_steps : ℕ := 7

/-- The number of individuals -/
def num_individuals : ℕ := 3

/-- The maximum number of people allowed on each step -/
def max_per_step : ℕ := 2

/-- Theorem stating that the arrangement count is correct -/
theorem arrangement_count_is_correct :
  arrangement_count = 
    (num_steps.choose num_individuals * num_individuals.factorial) + 
    (num_individuals.choose 2 * num_steps.choose 2) := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_is_correct_l427_42750


namespace NUMINAMATH_CALUDE_exponent_division_l427_42776

theorem exponent_division (a : ℝ) : a^6 / a^4 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l427_42776


namespace NUMINAMATH_CALUDE_fraction_equivalence_l427_42734

theorem fraction_equivalence (n : ℚ) : (4 + n) / (7 + n) = 7 / 9 ↔ n = 13 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l427_42734


namespace NUMINAMATH_CALUDE_locus_of_p_forms_two_circles_l427_42716

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on the circle
structure PointOnCircle (c : Circle) where
  point : ℝ × ℝ
  on_circle : (point.1 - c.center.1)^2 + (point.2 - c.center.2)^2 = c.radius^2

-- Define the projection of a point onto a line
def project_point_to_line (p : ℝ × ℝ) (l : ℝ × ℝ → ℝ) : ℝ × ℝ :=
  sorry

-- Define the point P on OQ such that OP = QQ'
def point_p (c : Circle) (q : PointOnCircle c) (diameter : ℝ × ℝ → ℝ) : ℝ × ℝ :=
  sorry

-- Theorem stating that the locus of P forms two circles
theorem locus_of_p_forms_two_circles (c : Circle) (diameter : ℝ × ℝ → ℝ) :
  ∃ (c1 c2 : Circle),
    ∀ (q : PointOnCircle c),
      let p := point_p c q diameter
      (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 = c1.radius^2 ∨
      (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 = c2.radius^2 :=
sorry

end NUMINAMATH_CALUDE_locus_of_p_forms_two_circles_l427_42716


namespace NUMINAMATH_CALUDE_cookies_per_person_l427_42741

/-- The number of cookies in a dozen --/
def dozen : ℕ := 12

/-- The number of batches Beth bakes --/
def batches : ℕ := 4

/-- The number of dozens per batch --/
def dozens_per_batch : ℕ := 2

/-- The number of people sharing the cookies --/
def people : ℕ := 16

/-- Theorem: Each person consumes 6 cookies when 4 batches of 2 dozen cookies are shared equally among 16 people --/
theorem cookies_per_person :
  (batches * dozens_per_batch * dozen) / people = 6 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_person_l427_42741


namespace NUMINAMATH_CALUDE_no_distributive_laws_hold_l427_42709

-- Define the * operation
def star (a b : ℝ) : ℝ := 3 * (a + b)

-- Theorem stating that none of the distributive laws hold
theorem no_distributive_laws_hold :
  (∀ x y z : ℝ, star x (y + z) = star x y + star x z) ∧
  (∀ x y z : ℝ, x + star y z = star (x + y) (x + z)) ∧
  (∀ x y z : ℝ, star x (star y z) = star (star x y) (star x z)) →
  False :=
by sorry

end NUMINAMATH_CALUDE_no_distributive_laws_hold_l427_42709


namespace NUMINAMATH_CALUDE_prob_at_most_sixes_equals_sum_l427_42708

def numDice : ℕ := 10
def maxSixes : ℕ := 3

def probExactlySixes (n : ℕ) : ℚ :=
  (Nat.choose numDice n) * (1/6)^n * (5/6)^(numDice - n)

def probAtMostSixes : ℚ :=
  (Finset.range (maxSixes + 1)).sum probExactlySixes

theorem prob_at_most_sixes_equals_sum :
  probAtMostSixes = 
    probExactlySixes 0 + probExactlySixes 1 + 
    probExactlySixes 2 + probExactlySixes 3 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_most_sixes_equals_sum_l427_42708


namespace NUMINAMATH_CALUDE_truncated_cube_properties_l427_42798

/-- A space-filling cube arrangement --/
structure CubeArrangement where
  /-- The space is filled with equal cubes --/
  space_filled : Bool
  /-- Eight cubes converge at each vertex --/
  eight_cubes_at_vertex : Bool

/-- A truncated cube in the arrangement --/
structure TruncatedCube where
  /-- The number of faces after truncation --/
  num_faces : Nat
  /-- The number of octagonal faces --/
  num_octagonal_faces : Nat
  /-- The number of triangular faces --/
  num_triangular_faces : Nat

/-- The result of truncating and joining cubes in the arrangement --/
def truncate_and_join (arr : CubeArrangement) : TruncatedCube × Rational :=
  sorry

/-- Theorem stating the properties of truncated cubes and space occupation --/
theorem truncated_cube_properties (arr : CubeArrangement) :
  arr.space_filled ∧ arr.eight_cubes_at_vertex →
  let (truncated_cube, octahedra_space) := truncate_and_join arr
  truncated_cube.num_faces = 14 ∧
  truncated_cube.num_octagonal_faces = 6 ∧
  truncated_cube.num_triangular_faces = 8 ∧
  octahedra_space = 5/6 :=
  sorry

end NUMINAMATH_CALUDE_truncated_cube_properties_l427_42798


namespace NUMINAMATH_CALUDE_roots_properties_l427_42715

theorem roots_properties (a b : ℝ) 
  (h1 : a^2 - 6*a + 4 = 0) 
  (h2 : b^2 - 6*b + 4 = 0) 
  (h3 : a > b) : 
  (a > 0 ∧ b > 0) ∧ 
  (((Real.sqrt a - Real.sqrt b) / (Real.sqrt a + Real.sqrt b)) = Real.sqrt 5 / 5) := by
  sorry

end NUMINAMATH_CALUDE_roots_properties_l427_42715


namespace NUMINAMATH_CALUDE_abs_purely_imaginary_complex_l427_42796

/-- Given a complex number z = (a + i) / (1 + i) where a is real,
    if z is purely imaginary, then its absolute value is 1. -/
theorem abs_purely_imaginary_complex (a : ℝ) :
  let z : ℂ := (a + Complex.I) / (1 + Complex.I)
  (z.re = 0 ∧ z.im ≠ 0) → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_purely_imaginary_complex_l427_42796


namespace NUMINAMATH_CALUDE_action_figures_removed_l427_42753

/-- 
Given:
- initial_figures: The initial number of action figures
- added_figures: The number of action figures added
- final_figures: The final number of action figures on the shelf

Prove that the number of removed figures is 1.
-/
theorem action_figures_removed 
  (initial_figures : ℕ) 
  (added_figures : ℕ) 
  (final_figures : ℕ) 
  (h1 : initial_figures = 3)
  (h2 : added_figures = 4)
  (h3 : final_figures = 6) :
  initial_figures + added_figures - final_figures = 1 := by
  sorry

end NUMINAMATH_CALUDE_action_figures_removed_l427_42753


namespace NUMINAMATH_CALUDE_functions_inequality_l427_42773

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- State the theorem
theorem functions_inequality (hf : f 0 = 0) 
  (hg : ∀ x y : ℝ, g (x - y) ≥ f x * f y + g x * g y) :
  ∀ x : ℝ, f x ^ 2008 + g x ^ 2008 ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_functions_inequality_l427_42773


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_quadratic_equation_with_square_l427_42782

theorem quadratic_equation_solutions :
  (∃ x : ℝ, x^2 - 4*x - 8 = 0) ↔ 
  (∃ x : ℝ, x = 2 + 2*Real.sqrt 3 ∨ x = 2 - 2*Real.sqrt 3) :=
sorry

theorem quadratic_equation_with_square :
  (∃ x : ℝ, (x - 2)^2 = 2*x - 4) ↔ 
  (∃ x : ℝ, x = 2 ∨ x = 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_quadratic_equation_with_square_l427_42782


namespace NUMINAMATH_CALUDE_trig_identity_l427_42755

theorem trig_identity : 
  (Real.cos (20 * π / 180) * Real.sin (20 * π / 180)) / 
  (Real.cos (25 * π / 180)^2 - Real.sin (25 * π / 180)^2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l427_42755


namespace NUMINAMATH_CALUDE_company_women_count_l427_42772

theorem company_women_count (total_workers : ℕ) 
  (h1 : total_workers / 3 = total_workers - (2 * total_workers / 3))  -- One-third don't have retirement plan
  (h2 : (total_workers / 3) / 5 = total_workers / 15)  -- 20% of workers without plan are women
  (h3 : (2 * total_workers / 3) * 2 / 5 = (2 * total_workers / 3) - ((2 * total_workers / 3) * 3 / 5))  -- 40% of workers with plan are men
  (h4 : 144 = (2 * total_workers / 3) * 2 / 5)  -- 144 men in the company
  : (total_workers / 15 + (2 * total_workers / 3) * 3 / 5 = 252) := by
  sorry

end NUMINAMATH_CALUDE_company_women_count_l427_42772


namespace NUMINAMATH_CALUDE_cylinder_circumference_l427_42730

/-- Given two right circular cylinders C and B, prove that the circumference of C is 8√5 meters -/
theorem cylinder_circumference (h_C h_B r_B : ℝ) (vol_C vol_B : ℝ) : 
  h_C = 10 →
  h_B = 8 →
  2 * Real.pi * r_B = 10 →
  vol_C = Real.pi * (h_C * r_C^2) →
  vol_B = Real.pi * (h_B * r_B^2) →
  vol_C = 0.8 * vol_B →
  2 * Real.pi * r_C = 8 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_circumference_l427_42730


namespace NUMINAMATH_CALUDE_power_sum_equals_w_minus_one_l427_42723

theorem power_sum_equals_w_minus_one (w : ℂ) (hw : w^2 - w + 1 = 0) : 
  w^98 + w^99 + w^100 + w^101 + w^102 = w - 1 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_w_minus_one_l427_42723


namespace NUMINAMATH_CALUDE_reseating_ways_l427_42732

/-- Represents the number of ways n women can be reseated under the given rules -/
def S : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | 2 => 3
  | n + 3 => S (n + 2) + S (n + 1) + S n

/-- The number of women -/
def num_women : ℕ := 12

/-- The theorem stating that the number of ways 12 women can be reseated is 927 -/
theorem reseating_ways : S num_women = 927 := by
  sorry

end NUMINAMATH_CALUDE_reseating_ways_l427_42732


namespace NUMINAMATH_CALUDE_equalSideToWidthRatio_l427_42724

/-- Represents a rectangle with given width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- Represents an isosceles triangle with two equal sides and a base -/
structure IsoscelesTriangle where
  equalSide : ℝ
  base : ℝ

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.length)

/-- Calculates the perimeter of an isosceles triangle -/
def IsoscelesTriangle.perimeter (t : IsoscelesTriangle) : ℝ := 2 * t.equalSide + t.base

/-- Theorem: The ratio of the equal side of an isosceles triangle to the width of a rectangle
    is 5/2, given that both shapes have a perimeter of 60 and the rectangle's length is twice its width -/
theorem equalSideToWidthRatio :
  ∀ (r : Rectangle) (t : IsoscelesTriangle),
    r.perimeter = 60 →
    t.perimeter = 60 →
    r.length = 2 * r.width →
    t.equalSide / r.width = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equalSideToWidthRatio_l427_42724


namespace NUMINAMATH_CALUDE_village_population_growth_l427_42746

theorem village_population_growth (c d : ℕ) : 
  c^3 + 180 = d^3 + 10 →                     -- Population condition for 2001
  (d + 1)^3 = d^3 + 180 →                    -- Population condition for 2011
  (((d + 1)^3 - c^3) * 100) / c^3 = 101 :=   -- Percent growth over 20 years
by
  sorry

end NUMINAMATH_CALUDE_village_population_growth_l427_42746


namespace NUMINAMATH_CALUDE_largest_n_with_lcm_property_l427_42744

theorem largest_n_with_lcm_property : 
  ∃ (m : ℕ+), Nat.lcm m.val 972 = 3 * m.val * Nat.gcd m.val 972 ∧ 
  ∀ (n : ℕ) (m : ℕ+), n > 972 → n < 1000 → 
    Nat.lcm m.val n ≠ 3 * m.val * Nat.gcd m.val n := by
  sorry

end NUMINAMATH_CALUDE_largest_n_with_lcm_property_l427_42744


namespace NUMINAMATH_CALUDE_opposite_of_negative_five_l427_42774

-- Define the concept of opposite
def opposite (a : ℝ) : ℝ := -a

-- State the theorem
theorem opposite_of_negative_five :
  opposite (-5) = 5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_five_l427_42774


namespace NUMINAMATH_CALUDE_systematic_sampling_selection_l427_42752

/-- Represents a student number in the range [1, 1000] -/
def StudentNumber := Fin 1000

/-- The total number of students -/
def totalStudents : ℕ := 1000

/-- The number of students to be selected -/
def sampleSize : ℕ := 200

/-- The sample interval for systematic sampling -/
def sampleInterval : ℕ := totalStudents / sampleSize

/-- Predicate to check if a student number is selected in the systematic sampling -/
def isSelected (n : StudentNumber) : Prop :=
  n.val % sampleInterval = 122 % sampleInterval

theorem systematic_sampling_selection :
  isSelected ⟨121, by norm_num⟩ → isSelected ⟨926, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_selection_l427_42752


namespace NUMINAMATH_CALUDE_max_cookies_juan_l427_42768

/-- Represents the ingredients required for baking cookies -/
structure Ingredients where
  milk : ℚ
  sugar : ℚ
  flour : ℚ

/-- Represents the storage capacity for ingredients -/
structure StorageCapacity where
  milk : ℚ
  sugar : ℚ
  flour : ℚ

/-- Calculate the maximum number of cookies that can be baked given the ingredients per cookie and storage capacity -/
def max_cookies (ingredients_per_cookie : Ingredients) (storage : StorageCapacity) : ℚ :=
  min (storage.milk / ingredients_per_cookie.milk)
      (min (storage.sugar / ingredients_per_cookie.sugar)
           (storage.flour / ingredients_per_cookie.flour))

/-- Theorem: The maximum number of cookies Juan can bake within storage constraints is 320 -/
theorem max_cookies_juan :
  let ingredients_per_40_cookies : Ingredients := { milk := 10, sugar := 5, flour := 15 }
  let ingredients_per_cookie : Ingredients := {
    milk := ingredients_per_40_cookies.milk / 40,
    sugar := ingredients_per_40_cookies.sugar / 40,
    flour := ingredients_per_40_cookies.flour / 40
  }
  let storage : StorageCapacity := { milk := 80, sugar := 200, flour := 220 }
  max_cookies ingredients_per_cookie storage = 320 := by sorry

end NUMINAMATH_CALUDE_max_cookies_juan_l427_42768


namespace NUMINAMATH_CALUDE_school_population_l427_42758

/-- Given a school with 42 boys and a boy-to-girl ratio of 7:1, 
    prove that the total number of students is 48. -/
theorem school_population (num_boys : ℕ) (ratio : ℚ) : 
  num_boys = 42 → ratio = 7/1 → num_boys + (num_boys / ratio.num) = 48 := by
  sorry

end NUMINAMATH_CALUDE_school_population_l427_42758


namespace NUMINAMATH_CALUDE_marble_sharing_l427_42769

theorem marble_sharing (initial_marbles : ℝ) (initial_marbles_pos : initial_marbles > 0) :
  let remaining_after_lara := initial_marbles * (1 - 0.3)
  let remaining_after_max := remaining_after_lara * (1 - 0.15)
  let remaining_after_ben := remaining_after_max * (1 - 0.2)
  remaining_after_ben / initial_marbles = 0.476 := by
sorry

end NUMINAMATH_CALUDE_marble_sharing_l427_42769


namespace NUMINAMATH_CALUDE_breakfast_cooking_time_l427_42739

theorem breakfast_cooking_time (num_sausages num_eggs egg_time total_time : ℕ) 
  (h1 : num_sausages = 3)
  (h2 : num_eggs = 6)
  (h3 : egg_time = 4)
  (h4 : total_time = 39) :
  ∃ (sausage_time : ℕ), 
    sausage_time * num_sausages + egg_time * num_eggs = total_time ∧ 
    sausage_time = 5 := by
  sorry

end NUMINAMATH_CALUDE_breakfast_cooking_time_l427_42739


namespace NUMINAMATH_CALUDE_book_sale_loss_percentage_l427_42785

/-- Given two books with a total cost of 600, where one is sold at a loss and the other at a 19% gain,
    both sold at the same price, and the cost of the book sold at a loss is 350,
    prove that the loss percentage on the first book is 15%. -/
theorem book_sale_loss_percentage : 
  ∀ (total_cost cost_book1 cost_book2 selling_price gain_percentage : ℚ),
  total_cost = 600 →
  cost_book1 = 350 →
  cost_book2 = total_cost - cost_book1 →
  gain_percentage = 19 →
  selling_price = cost_book2 * (1 + gain_percentage / 100) →
  (cost_book1 - selling_price) / cost_book1 * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_loss_percentage_l427_42785


namespace NUMINAMATH_CALUDE_completing_square_result_l427_42710

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 2*x - 2 = 0

-- Define the completed square form
def completed_square (x n : ℝ) : Prop := (x - 1)^2 = n

-- Theorem statement
theorem completing_square_result : 
  ∃ n : ℝ, (∀ x : ℝ, quadratic_equation x ↔ completed_square x n) ∧ n = 3 :=
sorry

end NUMINAMATH_CALUDE_completing_square_result_l427_42710


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l427_42754

theorem simplify_trig_expression (x : ℝ) :
  (Real.sin x + Real.sin (3 * x)) / (1 + Real.cos x + Real.cos (3 * x)) =
  (4 * Real.sin x * Real.cos x ^ 2) / (1 - 4 * Real.cos x ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l427_42754


namespace NUMINAMATH_CALUDE_problem_solution_l427_42737

theorem problem_solution : 
  ((-1 : ℝ) ^ 2023 + Real.sqrt 9 - 2022 ^ 0 = 1) ∧ 
  ((Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) + Real.sqrt 8 = 2 + 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l427_42737


namespace NUMINAMATH_CALUDE_routes_on_3x2_grid_l427_42727

/-- The number of routes on a grid from (0,0) to (m,n) moving only right or down -/
def numRoutes (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

/-- The width of the grid -/
def gridWidth : ℕ := 3

/-- The height of the grid -/
def gridHeight : ℕ := 2

theorem routes_on_3x2_grid : 
  numRoutes gridWidth gridHeight = 10 := by
  sorry

end NUMINAMATH_CALUDE_routes_on_3x2_grid_l427_42727


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l427_42797

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  max (a + 1/b) (max (b + 1/c) (c + 1/a)) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l427_42797


namespace NUMINAMATH_CALUDE_expected_closest_distance_five_points_l427_42745

/-- The expected distance between the closest pair of points when five points are chosen uniformly at random on a segment of length 1 -/
theorem expected_closest_distance_five_points (segment_length : ℝ) 
  (h_segment : segment_length = 1) : ℝ :=
by
  sorry

end NUMINAMATH_CALUDE_expected_closest_distance_five_points_l427_42745


namespace NUMINAMATH_CALUDE_alex_problem_count_l427_42791

/-- Given that Alex has written 61 problems out of 187 total problems,
    this theorem proves that he needs to write 65 more problems
    to have written half of the total problems. -/
theorem alex_problem_count (alex_initial : ℕ) (total_initial : ℕ)
    (h1 : alex_initial = 61)
    (h2 : total_initial = 187) :
    ∃ x : ℕ, 2 * (alex_initial + x) = total_initial + x ∧ x = 65 := by
  sorry

end NUMINAMATH_CALUDE_alex_problem_count_l427_42791


namespace NUMINAMATH_CALUDE_fraction_simplification_l427_42725

theorem fraction_simplification (a x : ℝ) (h : a^2 + x^2 ≠ 0) :
  (Real.sqrt (a^2 + x^2) + (x^2 - a^2) / Real.sqrt (a^2 + x^2)) / (a^2 + x^2) =
  2 * x^2 / (a^2 + x^2)^(3/2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l427_42725


namespace NUMINAMATH_CALUDE_max_negative_integers_l427_42751

theorem max_negative_integers (a b c d e f : ℤ) (h : a * b + c * d * e * f < 0) :
  (∃ neg_count : ℕ, neg_count ≤ 3 ∧
    neg_count = (if a < 0 then 1 else 0) + (if b < 0 then 1 else 0) +
                (if c < 0 then 1 else 0) + (if d < 0 then 1 else 0) +
                (if e < 0 then 1 else 0) + (if f < 0 then 1 else 0)) ∧
  ¬(∃ neg_count : ℕ, neg_count > 3 ∧
    neg_count = (if a < 0 then 1 else 0) + (if b < 0 then 1 else 0) +
                (if c < 0 then 1 else 0) + (if d < 0 then 1 else 0) +
                (if e < 0 then 1 else 0) + (if f < 0 then 1 else 0)) :=
by sorry

end NUMINAMATH_CALUDE_max_negative_integers_l427_42751


namespace NUMINAMATH_CALUDE_remainder_problem_l427_42761

theorem remainder_problem (x : ℤ) : x % 61 = 24 → x % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l427_42761


namespace NUMINAMATH_CALUDE_pupusa_minimum_l427_42762

theorem pupusa_minimum (a b : ℕ+) (h1 : a < 391) (h2 : Nat.lcm a b > Nat.lcm a 391) : 
  ∀ b' : ℕ+, (∃ a' : ℕ+, a' < 391 ∧ Nat.lcm a' b' > Nat.lcm a' 391) → b' ≥ 18 := by
sorry

end NUMINAMATH_CALUDE_pupusa_minimum_l427_42762


namespace NUMINAMATH_CALUDE_smallest_integer_larger_than_root_sum_fourth_power_l427_42729

theorem smallest_integer_larger_than_root_sum_fourth_power :
  ∃ n : ℕ, n = 248 ∧ (∀ m : ℕ, m < n → (m : ℝ) ≤ (Real.sqrt 5 + Real.sqrt 3)^4) ∧
  (n : ℝ) > (Real.sqrt 5 + Real.sqrt 3)^4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_larger_than_root_sum_fourth_power_l427_42729


namespace NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l427_42770

/-- A circle inscribed in a rectangle --/
structure InscribedCircle where
  radius : ℝ
  rectangle_length : ℝ
  rectangle_width : ℝ
  inscribed : rectangle_width = 2 * radius
  ratio : rectangle_length = 3 * rectangle_width

/-- The area of a rectangle with an inscribed circle of radius 8 and length-to-width ratio of 3:1 is 768 --/
theorem inscribed_circle_rectangle_area (c : InscribedCircle) 
  (h1 : c.radius = 8) : c.rectangle_length * c.rectangle_width = 768 := by
  sorry

#check inscribed_circle_rectangle_area

end NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l427_42770


namespace NUMINAMATH_CALUDE_probability_zero_or_one_excellent_equals_formula_l427_42747

def total_people : ℕ := 12
def excellent_students : ℕ := 5
def selected_people : ℕ := 5

def probability_zero_or_one_excellent : ℚ :=
  (Nat.choose (total_people - excellent_students) selected_people +
   Nat.choose excellent_students 1 * Nat.choose (total_people - excellent_students) (selected_people - 1)) /
  Nat.choose total_people selected_people

theorem probability_zero_or_one_excellent_equals_formula :
  probability_zero_or_one_excellent = 
  (Nat.choose (total_people - excellent_students) selected_people +
   Nat.choose excellent_students 1 * Nat.choose (total_people - excellent_students) (selected_people - 1)) /
  Nat.choose total_people selected_people :=
by sorry

end NUMINAMATH_CALUDE_probability_zero_or_one_excellent_equals_formula_l427_42747


namespace NUMINAMATH_CALUDE_main_tire_mileage_approx_l427_42703

/-- Represents the mileage distribution of a car's tires -/
structure CarTires where
  total_miles : ℕ
  num_main_tires : ℕ
  num_spare_tires : ℕ
  spare_multiplier : ℕ

/-- Calculates the mileage for each main tire -/
def main_tire_mileage (c : CarTires) : ℚ :=
  c.total_miles / (c.num_main_tires + c.spare_multiplier * c.num_spare_tires)

/-- Theorem stating the main tire mileage for the given conditions -/
theorem main_tire_mileage_approx :
  let c : CarTires := {
    total_miles := 40000,
    num_main_tires := 4,
    num_spare_tires := 1,
    spare_multiplier := 2
  }
  ∃ ε > 0, |main_tire_mileage c - 6667| < ε :=
sorry

end NUMINAMATH_CALUDE_main_tire_mileage_approx_l427_42703


namespace NUMINAMATH_CALUDE_percentage_with_repeat_approx_l427_42763

/-- The count of all possible five-digit numbers -/
def total_five_digit_numbers : ℕ := 90000

/-- The count of five-digit numbers without repeated digits -/
def no_repeat_numbers : ℕ := 27216

/-- The count of five-digit numbers with at least one repeated digit -/
def repeat_numbers : ℕ := total_five_digit_numbers - no_repeat_numbers

/-- The percentage of five-digit numbers with at least one repeated digit -/
def percentage_with_repeat : ℚ :=
  (repeat_numbers : ℚ) / (total_five_digit_numbers : ℚ) * 100

theorem percentage_with_repeat_approx :
  ∃ (ε : ℚ), ε > 0 ∧ ε < (1 : ℚ) / 10 ∧ 
  abs (percentage_with_repeat - 698 / 10) < ε :=
sorry

end NUMINAMATH_CALUDE_percentage_with_repeat_approx_l427_42763


namespace NUMINAMATH_CALUDE_somu_age_problem_l427_42787

/-- Somu's age problem -/
theorem somu_age_problem (somu_age father_age : ℕ) : 
  somu_age = father_age / 3 →
  somu_age - 7 = (father_age - 7) / 5 →
  somu_age = 14 := by
  sorry

end NUMINAMATH_CALUDE_somu_age_problem_l427_42787


namespace NUMINAMATH_CALUDE_cricket_average_l427_42720

theorem cricket_average (initial_average : ℝ) (innings : ℕ) (new_score : ℝ) (average_increase : ℝ) :
  innings = 16 →
  new_score = 92 →
  average_increase = 4 →
  (innings * initial_average + new_score) / (innings + 1) = initial_average + average_increase →
  initial_average + average_increase = 28 := by
  sorry

end NUMINAMATH_CALUDE_cricket_average_l427_42720


namespace NUMINAMATH_CALUDE_greatest_c_value_l427_42780

theorem greatest_c_value (c : ℝ) : 
  (∀ x : ℝ, -x^2 + 9*x - 20 ≥ 0 → x ≤ 5) ∧ 
  (-5^2 + 9*5 - 20 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_greatest_c_value_l427_42780


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l427_42756

def repeating_decimal_3 : ℚ := 1 / 3
def repeating_decimal_56 : ℚ := 56 / 99

theorem sum_of_repeating_decimals : 
  repeating_decimal_3 + repeating_decimal_56 = 89 / 99 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l427_42756


namespace NUMINAMATH_CALUDE_y_satisfies_equation_l427_42728

noncomputable def y (a x : ℝ) : ℝ := a * Real.tan (Real.sqrt (a / x - 1))

theorem y_satisfies_equation (a x : ℝ) (h1 : x ≠ 0) (h2 : a / x - 1 ≥ 0) :
  a^2 + (y a x)^2 + 2 * x * Real.sqrt (a * x - x^2) * (deriv (y a) x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_y_satisfies_equation_l427_42728


namespace NUMINAMATH_CALUDE_four_is_eight_percent_of_fifty_l427_42766

theorem four_is_eight_percent_of_fifty :
  (4 : ℝ) / 50 * 100 = 8 := by sorry

end NUMINAMATH_CALUDE_four_is_eight_percent_of_fifty_l427_42766


namespace NUMINAMATH_CALUDE_total_money_found_l427_42705

def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.10
def nickel_value : ℚ := 0.05
def penny_value : ℚ := 0.01

def num_quarters : ℕ := 10
def num_dimes : ℕ := 3
def num_nickels : ℕ := 3
def num_pennies : ℕ := 5

theorem total_money_found :
  (num_quarters : ℚ) * quarter_value +
  (num_dimes : ℚ) * dime_value +
  (num_nickels : ℚ) * nickel_value +
  (num_pennies : ℚ) * penny_value = 3 := by sorry

end NUMINAMATH_CALUDE_total_money_found_l427_42705


namespace NUMINAMATH_CALUDE_minibus_seats_l427_42779

/-- The number of seats in a minibus given specific seating arrangements -/
theorem minibus_seats (total_children : ℕ) (three_child_seats : ℕ) : 
  total_children = 19 →
  three_child_seats = 5 →
  (∃ (two_child_seats : ℕ), 
    total_children = three_child_seats * 3 + two_child_seats * 2 ∧
    three_child_seats + two_child_seats = 7) := by
  sorry

end NUMINAMATH_CALUDE_minibus_seats_l427_42779


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l427_42777

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem seventh_term_of_geometric_sequence
  (a : ℕ → ℝ) (q : ℝ)
  (h_geometric : geometric_sequence a q)
  (h_a4 : a 4 = 8)
  (h_q : q = -2) :
  a 7 = -64 := by
sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l427_42777


namespace NUMINAMATH_CALUDE_complex_magnitude_l427_42757

theorem complex_magnitude (z : ℂ) (h : z^4 = 80 - 96*I) : Complex.abs z = 5^(3/4) := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l427_42757


namespace NUMINAMATH_CALUDE_exists_graph_with_short_paths_l427_42733

/-- A directed graph with n vertices -/
def DirectedGraph (n : ℕ) := Fin n → Fin n → Prop

/-- A path of length at most 2 exists between two vertices in a directed graph -/
def HasPathAtMost2 (G : DirectedGraph n) (u v : Fin n) : Prop :=
  G u v ∨ ∃ w, G u w ∧ G w v

/-- For any n > 4, there exists a directed graph with n vertices
    such that any two vertices have a path of length at most 2 between them -/
theorem exists_graph_with_short_paths (n : ℕ) (h : n > 4) :
  ∃ G : DirectedGraph n, ∀ u v : Fin n, HasPathAtMost2 G u v :=
sorry

end NUMINAMATH_CALUDE_exists_graph_with_short_paths_l427_42733


namespace NUMINAMATH_CALUDE_uki_biscuits_per_day_l427_42722

/-- Represents the daily production and pricing of bakery items -/
structure BakeryData where
  cupcake_price : ℝ
  cookie_price : ℝ
  biscuit_price : ℝ
  cupcakes_per_day : ℕ
  cookie_packets_per_day : ℕ
  total_earnings_five_days : ℝ

/-- Calculates the number of biscuit packets that can be baked daily -/
def biscuits_per_day (data : BakeryData) : ℕ :=
  20

/-- Theorem stating that given the bakery data, Uki can bake 20 packets of biscuits per day -/
theorem uki_biscuits_per_day (data : BakeryData)
    (h1 : data.cupcake_price = 1.5)
    (h2 : data.cookie_price = 2)
    (h3 : data.biscuit_price = 1)
    (h4 : data.cupcakes_per_day = 20)
    (h5 : data.cookie_packets_per_day = 10)
    (h6 : data.total_earnings_five_days = 350) :
    biscuits_per_day data = 20 := by
  sorry

end NUMINAMATH_CALUDE_uki_biscuits_per_day_l427_42722


namespace NUMINAMATH_CALUDE_problem_1_l427_42742

theorem problem_1 (x : ℝ) (h1 : x ≠ 0) (h2 : x^2 - Real.sqrt 5 * x - x - 1 = 0) :
  x^2 + 1/x^2 = 8 + 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l427_42742


namespace NUMINAMATH_CALUDE_xi_eq_4_equiv_events_l427_42731

/-- Represents the outcome of a single die roll -/
def DieRoll : Type := Fin 6

/-- Represents the outcome of rolling two dice -/
def TwoDiceRoll : Type := DieRoll × DieRoll

/-- The sum of the points obtained when rolling two dice -/
def ξ : TwoDiceRoll → Nat :=
  fun (d1, d2) => d1.val + 1 + d2.val + 1

/-- The event where one die shows 3 and the other shows 1 -/
def event_3_1 : Set TwoDiceRoll :=
  {roll | (roll.1.val = 2 ∧ roll.2.val = 0) ∨ (roll.1.val = 0 ∧ roll.2.val = 2)}

/-- The event where both dice show 2 -/
def event_2_2 : Set TwoDiceRoll :=
  {roll | roll.1.val = 1 ∧ roll.2.val = 1}

/-- The theorem stating that ξ = 4 is equivalent to the union of event_3_1 and event_2_2 -/
theorem xi_eq_4_equiv_events :
  {roll : TwoDiceRoll | ξ roll = 4} = event_3_1 ∪ event_2_2 := by
  sorry

end NUMINAMATH_CALUDE_xi_eq_4_equiv_events_l427_42731


namespace NUMINAMATH_CALUDE_sequence_problem_l427_42789

def is_arithmetic_sequence (a b c : ℝ) : Prop :=
  b - a = c - b

def is_geometric_sequence (a b c : ℝ) : Prop :=
  b / a = c / b

theorem sequence_problem (x y : ℝ) :
  is_arithmetic_sequence (2 * x) 1 (y - 1) →
  is_geometric_sequence (y + 3) (|x + 1| + |x - 1|) (Real.cos (Real.arcsin (Real.sqrt (1 - x^2)))) →
  (x + 1) * (y + 1) = 4 ∨ (x + 1) * (y + 1) = 2 * (Real.sqrt 17 - 3) := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l427_42789


namespace NUMINAMATH_CALUDE_job_crop_production_l427_42781

/-- Represents the land allocation of Job's farm in hectares -/
structure FarmLand where
  total : ℕ
  house_and_machinery : ℕ
  future_expansion : ℕ
  cattle : ℕ

/-- Calculates the land used for crop production given a FarmLand allocation -/
def crop_production (farm : FarmLand) : ℕ :=
  farm.total - (farm.house_and_machinery + farm.future_expansion + farm.cattle)

/-- Theorem stating that for Job's specific land allocation, the crop production area is 70 hectares -/
theorem job_crop_production :
  let job_farm := FarmLand.mk 150 25 15 40
  crop_production job_farm = 70 := by
  sorry

end NUMINAMATH_CALUDE_job_crop_production_l427_42781


namespace NUMINAMATH_CALUDE_cost_increase_percentage_l427_42786

theorem cost_increase_percentage 
  (initial_cost_eggs initial_cost_apples : ℝ)
  (h_equal_initial_cost : initial_cost_eggs = initial_cost_apples)
  (egg_price_decrease : ℝ := 0.02)
  (apple_price_increase : ℝ := 0.10) :
  let new_cost_eggs := initial_cost_eggs * (1 - egg_price_decrease)
  let new_cost_apples := initial_cost_apples * (1 + apple_price_increase)
  let total_initial_cost := initial_cost_eggs + initial_cost_apples
  let total_new_cost := new_cost_eggs + new_cost_apples
  (total_new_cost - total_initial_cost) / total_initial_cost = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_cost_increase_percentage_l427_42786


namespace NUMINAMATH_CALUDE_particular_number_problem_l427_42706

theorem particular_number_problem (x : ℝ) :
  4 * (x - 220) = 320 → (5 * x) / 3 = 500 := by
  sorry

end NUMINAMATH_CALUDE_particular_number_problem_l427_42706


namespace NUMINAMATH_CALUDE_altitude_inradius_equality_l427_42788

/-- Triangle ABC with side lengths a, b, c, altitudes h_a, h_b, h_c, and inradius r -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a : ℝ
  h_b : ℝ
  h_c : ℝ
  r : ℝ

/-- The theorem states that the sum of altitudes equals 9 times the inradius 
    if and only if the triangle is equilateral -/
theorem altitude_inradius_equality (t : Triangle) : 
  t.h_a + t.h_b + t.h_c = 9 * t.r ↔ t.a = t.b ∧ t.b = t.c :=
sorry

end NUMINAMATH_CALUDE_altitude_inradius_equality_l427_42788


namespace NUMINAMATH_CALUDE_quiz_ranking_l427_42736

structure Student where
  name : String
  score : ℕ

def Hannah : Student := { name := "Hannah", score := 0 }
def Cassie : Student := { name := "Cassie", score := 0 }
def Bridget : Student := { name := "Bridget", score := 0 }

def is_not_highest (s : Student) (others : List Student) : Prop :=
  ∃ t ∈ others, t.score > s.score

def scored_better_than (s1 s2 : Student) : Prop :=
  s1.score > s2.score

def is_not_lowest (s : Student) (others : List Student) : Prop :=
  ∃ t ∈ others, s.score > t.score

theorem quiz_ranking :
  is_not_highest Hannah [Cassie, Bridget] →
  scored_better_than Bridget Cassie →
  is_not_lowest Cassie [Hannah, Bridget] →
  scored_better_than Bridget Cassie ∧ scored_better_than Cassie Hannah :=
by sorry

end NUMINAMATH_CALUDE_quiz_ranking_l427_42736


namespace NUMINAMATH_CALUDE_unequal_weight_l427_42711

-- Define the shapes as variables
variable (square circle big_circle triangle big_triangle : ℕ)

-- Define the balance conditions
def balance1 : Prop := 4 * square = big_circle + circle
def balance2 : Prop := 2 * circle + big_circle = 2 * triangle

-- Define the weight of the original combination
def original_weight : ℕ := triangle + big_circle + square

-- Define the weight of the option to be proven unequal
def option_d_weight : ℕ := 2 * big_triangle + square

-- Theorem statement
theorem unequal_weight 
  (h1 : balance1 square circle big_circle)
  (h2 : balance2 circle big_circle triangle)
  (h3 : big_triangle = triangle) :
  option_d_weight square big_triangle ≠ original_weight triangle big_circle square :=
sorry

end NUMINAMATH_CALUDE_unequal_weight_l427_42711


namespace NUMINAMATH_CALUDE_blake_change_l427_42767

/-- The amount Blake spent on oranges -/
def orange_cost : ℕ := 40

/-- The amount Blake spent on apples -/
def apple_cost : ℕ := 50

/-- The amount Blake spent on mangoes -/
def mango_cost : ℕ := 60

/-- The initial amount Blake had -/
def initial_amount : ℕ := 300

/-- The change Blake received after shopping -/
def change : ℕ := initial_amount - (orange_cost + apple_cost + mango_cost)

theorem blake_change : change = 150 := by sorry

end NUMINAMATH_CALUDE_blake_change_l427_42767


namespace NUMINAMATH_CALUDE_museum_clock_position_l427_42700

/-- A special clock with the given properties -/
structure SpecialClock where
  positions : ℕ
  jump_interval : ℕ
  jump_distance : ℕ

/-- Calculate the position of the clock hand after a given number of minutes -/
def clock_position (clock : SpecialClock) (initial_position : ℕ) (minutes : ℕ) : ℕ :=
  (initial_position + (minutes / clock.jump_interval) * clock.jump_distance) % clock.positions

theorem museum_clock_position : 
  let clock := SpecialClock.mk 20 7 9
  let minutes_between_8pm_and_8am := 12 * 60
  clock_position clock 9 minutes_between_8pm_and_8am = 2 := by
  sorry

end NUMINAMATH_CALUDE_museum_clock_position_l427_42700


namespace NUMINAMATH_CALUDE_games_played_so_far_l427_42778

/-- Proves that the number of games played so far is 15, given the conditions of the problem -/
theorem games_played_so_far 
  (total_games : ℕ) 
  (current_average : ℚ) 
  (goal_average : ℚ) 
  (required_average : ℚ) 
  (h1 : total_games = 20)
  (h2 : current_average = 26)
  (h3 : goal_average = 30)
  (h4 : required_average = 42)
  : ∃ (x : ℕ), x = 15 ∧ 
    x * current_average + (total_games - x) * required_average = total_games * goal_average := by
  sorry

end NUMINAMATH_CALUDE_games_played_so_far_l427_42778


namespace NUMINAMATH_CALUDE_inequality_proof_l427_42701

theorem inequality_proof (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a * b + b * c + c * a = 1) : 
  (((1 / a + 6 * b) ^ (1/3 : ℝ)) + 
   ((1 / b + 6 * c) ^ (1/3 : ℝ)) + 
   ((1 / c + 6 * a) ^ (1/3 : ℝ))) ≤ 1 / (a * b * c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l427_42701


namespace NUMINAMATH_CALUDE_teacher_engineer_ratio_l427_42783

theorem teacher_engineer_ratio (t e : ℕ) (t_pos : t > 0) (e_pos : e > 0) :
  (40 * t + 55 * e) / (t + e) = 45 →
  t / e = 2 := by
sorry

end NUMINAMATH_CALUDE_teacher_engineer_ratio_l427_42783


namespace NUMINAMATH_CALUDE_jessicas_carrots_l427_42743

theorem jessicas_carrots (joan_carrots : ℕ) (total_carrots : ℕ) 
  (h1 : joan_carrots = 29) 
  (h2 : total_carrots = 40) : 
  total_carrots - joan_carrots = 11 := by
  sorry

end NUMINAMATH_CALUDE_jessicas_carrots_l427_42743


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l427_42712

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 1

-- State the theorem
theorem derivative_f_at_zero :
  (deriv f) 0 = 2 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l427_42712


namespace NUMINAMATH_CALUDE_special_function_property_l427_42790

/-- A function that is monotonically increasing on [0,2] and f(x+2) is even -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 2 → f x ≤ f y) ∧
  (∀ x, f (x + 2) = f (-x + 2))

theorem special_function_property (f : ℝ → ℝ) (h : special_function f) :
  f (7/2) < f 1 ∧ f 1 < f (5/2) := by
  sorry

end NUMINAMATH_CALUDE_special_function_property_l427_42790


namespace NUMINAMATH_CALUDE_vertical_line_not_conic_section_l427_42714

/-- The equation |y-3| = √((x+4)² + (y-3)²) describes a vertical line x = -4 -/
theorem vertical_line_not_conic_section :
  ∀ x y : ℝ, |y - 3| = Real.sqrt ((x + 4)^2 + (y - 3)^2) ↔ x = -4 :=
by sorry

end NUMINAMATH_CALUDE_vertical_line_not_conic_section_l427_42714


namespace NUMINAMATH_CALUDE_complement_of_A_l427_42702

def U : Set Int := Set.univ

def A : Set Int := {x : Int | x^2 - x - 2 ≥ 0}

theorem complement_of_A : Set.compl A = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_l427_42702
