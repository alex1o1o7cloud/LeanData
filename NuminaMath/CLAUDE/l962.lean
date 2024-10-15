import Mathlib

namespace NUMINAMATH_CALUDE_circle_equation_l962_96294

/-- The equation of a circle with center (-1, 2) and radius √5 is x² + y² + 2x - 4y = 0 -/
theorem circle_equation (x y : ℝ) :
  let center : ℝ × ℝ := (-1, 2)
  let radius : ℝ := Real.sqrt 5
  (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔ x^2 + y^2 + 2*x - 4*y = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l962_96294


namespace NUMINAMATH_CALUDE_divisibility_condition_l962_96258

theorem divisibility_condition (N : ℤ) : 
  (7 * N + 55) ∣ (N^2 - 71) ↔ N = 57 ∨ N = -8 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l962_96258


namespace NUMINAMATH_CALUDE_tan_alpha_one_third_implies_cos_2alpha_over_expression_l962_96247

theorem tan_alpha_one_third_implies_cos_2alpha_over_expression (α : Real) 
  (h : Real.tan α = 1/3) : 
  (Real.cos (2*α)) / (2 * Real.sin α * Real.cos α + (Real.cos α)^2) = 8/15 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_one_third_implies_cos_2alpha_over_expression_l962_96247


namespace NUMINAMATH_CALUDE_upsilon_value_l962_96207

theorem upsilon_value (Υ : ℤ) : 5 * (-3) = Υ - 3 → Υ = -12 := by
  sorry

end NUMINAMATH_CALUDE_upsilon_value_l962_96207


namespace NUMINAMATH_CALUDE_cross_section_ratio_cube_l962_96213

theorem cross_section_ratio_cube (a : ℝ) (ha : a > 0) :
  let cube_diagonal := a * Real.sqrt 3
  let min_area := (a / Real.sqrt 2) * cube_diagonal
  let max_area := Real.sqrt 2 * a^2
  max_area / min_area = 2 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_cross_section_ratio_cube_l962_96213


namespace NUMINAMATH_CALUDE_sqrt_fraction_equals_sixteen_l962_96244

theorem sqrt_fraction_equals_sixteen :
  let eight : ℕ := 2^3
  let four : ℕ := 2^2
  ∀ x : ℝ, x = (((eight^10 + four^10) : ℝ) / (eight^4 + four^11 : ℝ))^(1/2) → x = 16 := by
sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equals_sixteen_l962_96244


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l962_96284

def U : Set Int := {x | x^2 ≤ 2*x + 3}
def A : Set Int := {0, 1, 2}

theorem complement_of_A_in_U :
  (U \ A) = {-1, 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l962_96284


namespace NUMINAMATH_CALUDE_divisibility_implies_five_divisor_l962_96278

theorem divisibility_implies_five_divisor (n : ℕ) : 
  n > 1 → (6^n - 1) % n = 0 → n % 5 = 0 := by sorry

end NUMINAMATH_CALUDE_divisibility_implies_five_divisor_l962_96278


namespace NUMINAMATH_CALUDE_determine_fifth_subject_marks_l962_96216

/-- Given the marks of a student in 4 subjects and the average marks of 5 subjects,
    this theorem proves that the marks in the fifth subject can be uniquely determined. -/
theorem determine_fifth_subject_marks
  (english : ℕ)
  (mathematics : ℕ)
  (chemistry : ℕ)
  (biology : ℕ)
  (average : ℚ)
  (h1 : english = 70)
  (h2 : mathematics = 63)
  (h3 : chemistry = 63)
  (h4 : biology = 65)
  (h5 : average = 68.2)
  : ∃! physics : ℕ,
    (english + mathematics + physics + chemistry + biology : ℚ) / 5 = average :=
by sorry

end NUMINAMATH_CALUDE_determine_fifth_subject_marks_l962_96216


namespace NUMINAMATH_CALUDE_P_on_x_axis_P_parallel_to_y_axis_P_second_quadrant_equidistant_l962_96204

-- Define point P
def P (a : ℝ) : ℝ × ℝ := (2*a - 2, a + 5)

-- Theorem 1
theorem P_on_x_axis (a : ℝ) :
  P a = (-12, 0) ↔ (P a).2 = 0 :=
sorry

-- Theorem 2
theorem P_parallel_to_y_axis (a : ℝ) :
  P a = (4, 8) ↔ (P a).1 = 4 :=
sorry

-- Theorem 3
theorem P_second_quadrant_equidistant (a : ℝ) :
  (P a).1 < 0 ∧ (P a).2 > 0 ∧ |(P a).1| = |(P a).2| →
  a^2023 + 2022 = 2021 :=
sorry

end NUMINAMATH_CALUDE_P_on_x_axis_P_parallel_to_y_axis_P_second_quadrant_equidistant_l962_96204


namespace NUMINAMATH_CALUDE_sector_angle_in_unit_circle_l962_96226

theorem sector_angle_in_unit_circle (sector_area : ℝ) (central_angle : ℝ) : 
  sector_area = 1 → central_angle = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_in_unit_circle_l962_96226


namespace NUMINAMATH_CALUDE_negative_integer_solutions_l962_96273

def inequality_system (x : ℤ) : Prop :=
  2 * x + 9 ≥ 3 ∧ (1 + 2 * x) / 3 + 1 > x

def is_negative_integer (x : ℤ) : Prop :=
  x < 0

theorem negative_integer_solutions :
  {x : ℤ | inequality_system x ∧ is_negative_integer x} = {-3, -2, -1} :=
sorry

end NUMINAMATH_CALUDE_negative_integer_solutions_l962_96273


namespace NUMINAMATH_CALUDE_expression_evaluation_l962_96201

theorem expression_evaluation : 5 + 15 / 3 - 2^2 * 4 = -6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l962_96201


namespace NUMINAMATH_CALUDE_solution_set_theorem_min_value_g_min_value_fraction_l962_96227

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1|

-- Define the function g
def g (x : ℝ) : ℝ := f x + f (x - 1)

-- Theorem 1: Solution set of f(x) + |x+1| < 2
theorem solution_set_theorem :
  {x : ℝ | f x + |x + 1| < 2} = {x : ℝ | 0 < x ∧ x < 2/3} :=
sorry

-- Theorem 2: Minimum value of g(x)
theorem min_value_g :
  ∀ x : ℝ, g x ≥ 2 :=
sorry

-- Theorem 3: Minimum value of 4/m + 1/n
theorem min_value_fraction (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m + n = 2) :
  4/m + 1/n ≥ 9/2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_theorem_min_value_g_min_value_fraction_l962_96227


namespace NUMINAMATH_CALUDE_negation_of_universal_real_proposition_l962_96283

theorem negation_of_universal_real_proposition :
  (¬ ∀ x : ℝ, x^2 ≠ x) ↔ (∃ x : ℝ, x^2 = x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_real_proposition_l962_96283


namespace NUMINAMATH_CALUDE_smallest_result_l962_96282

def S : Finset ℕ := {2, 4, 6, 8, 10, 12}

def process (a b c : ℕ) : ℕ := (a + b) * c

def valid_choice (a b c : ℕ) : Prop :=
  a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem smallest_result :
  ∃ (a b c : ℕ), valid_choice a b c ∧
  process a b c = 20 ∧
  ∀ (x y z : ℕ), valid_choice x y z → process x y z ≥ 20 :=
sorry

end NUMINAMATH_CALUDE_smallest_result_l962_96282


namespace NUMINAMATH_CALUDE_march_first_is_thursday_l962_96206

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in March -/
structure MarchDate where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Given that March 15th is a Thursday, prove that March 1st is also a Thursday -/
theorem march_first_is_thursday (march15 : MarchDate) 
    (h : march15.day = 15 ∧ march15.dayOfWeek = DayOfWeek.Thursday) :
    ∃ (march1 : MarchDate), march1.day = 1 ∧ march1.dayOfWeek = DayOfWeek.Thursday :=
  sorry

end NUMINAMATH_CALUDE_march_first_is_thursday_l962_96206


namespace NUMINAMATH_CALUDE_find_y_l962_96211

def v (y : ℝ) : Fin 2 → ℝ := ![1, y]
def w : Fin 2 → ℝ := ![9, 3]
def proj_w_v : Fin 2 → ℝ := ![-6, -2]

theorem find_y : ∃ y : ℝ, v y = v y ∧ w = w ∧ proj_w_v = proj_w_v → y = -23 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l962_96211


namespace NUMINAMATH_CALUDE_polynomial_transformation_l962_96217

theorem polynomial_transformation (x y : ℝ) : 
  x^3 - 6*x^2 + 11*x - 6 = 0 → 
  y = x + 1/x → 
  x^2*(y^2 + y - 6) = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_transformation_l962_96217


namespace NUMINAMATH_CALUDE_science_club_election_l962_96250

theorem science_club_election (total_candidates : Nat) (past_officers : Nat) (positions : Nat) :
  total_candidates = 20 →
  past_officers = 10 →
  positions = 4 →
  (Nat.choose total_candidates positions -
   (Nat.choose (total_candidates - past_officers) positions +
    Nat.choose past_officers 1 * Nat.choose (total_candidates - past_officers) (positions - 1))) = 3435 :=
by sorry

end NUMINAMATH_CALUDE_science_club_election_l962_96250


namespace NUMINAMATH_CALUDE_factorial_quotient_l962_96242

theorem factorial_quotient : (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := by
  sorry

end NUMINAMATH_CALUDE_factorial_quotient_l962_96242


namespace NUMINAMATH_CALUDE_fixed_point_parabola_l962_96248

theorem fixed_point_parabola :
  ∀ (k : ℝ), 3 * (5 : ℝ)^2 + k * 5 - 5 * k = 75 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_parabola_l962_96248


namespace NUMINAMATH_CALUDE_exactly_two_sets_l962_96253

/-- A structure representing a set of consecutive positive integers -/
structure ConsecutiveSet where
  start : ℕ+
  length : ℕ+

/-- The sum of a set of consecutive positive integers -/
def sum_consecutive (s : ConsecutiveSet) : ℕ :=
  (s.length : ℕ) * (2 * (s.start : ℕ) + s.length - 1) / 2

/-- Predicate for a valid set of consecutive integers summing to 256 -/
def is_valid_set (s : ConsecutiveSet) : Prop :=
  s.length ≥ 2 ∧ sum_consecutive s = 256

theorem exactly_two_sets :
  ∃! (sets : Finset ConsecutiveSet), sets.card = 2 ∧ ∀ s ∈ sets, is_valid_set s :=
sorry

end NUMINAMATH_CALUDE_exactly_two_sets_l962_96253


namespace NUMINAMATH_CALUDE_one_in_set_zero_one_l962_96269

theorem one_in_set_zero_one : 1 ∈ ({0, 1} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_one_in_set_zero_one_l962_96269


namespace NUMINAMATH_CALUDE_chocolate_division_l962_96291

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (piles_for_shaina : ℕ) : 
  total_chocolate = 70 / 7 →
  num_piles = 5 →
  piles_for_shaina = 2 →
  (total_chocolate / num_piles) * piles_for_shaina = 4 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_division_l962_96291


namespace NUMINAMATH_CALUDE_four_digit_number_relation_l962_96288

theorem four_digit_number_relation : 
  let n : ℕ := 1197
  let thousands : ℕ := n / 1000
  let hundreds : ℕ := (n / 100) % 10
  let tens : ℕ := (n / 10) % 10
  let units : ℕ := n % 10
  units = hundreds - 2 →
  thousands + hundreds + tens + units = 18 →
  thousands = hundreds - 2 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_number_relation_l962_96288


namespace NUMINAMATH_CALUDE_quadratic_min_iff_m_gt_neg_one_l962_96215

/-- A quadratic function with coefficient (m + 1) has a minimum value if and only if m > -1 -/
theorem quadratic_min_iff_m_gt_neg_one (m : ℝ) :
  (∃ (min : ℝ), ∀ (x : ℝ), (m + 1) * x^2 ≥ min) ↔ m > -1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_min_iff_m_gt_neg_one_l962_96215


namespace NUMINAMATH_CALUDE_accurate_estimation_l962_96214

/-- Represents a scale with a lower and upper bound -/
structure Scale where
  lower : ℝ
  upper : ℝ
  h : lower < upper

/-- Represents the position of an arrow on the scale -/
def ArrowPosition (s : Scale) := {x : ℝ // s.lower ≤ x ∧ x ≤ s.upper}

/-- The set of possible readings -/
def PossibleReadings : Set ℝ := {10.1, 10.2, 10.3, 10.4, 10.5}

/-- Function to determine the most accurate estimation -/
noncomputable def mostAccurateEstimation (s : Scale) (arrow : ArrowPosition s) : ℝ :=
  sorry

/-- Theorem stating that 10.3 is the most accurate estimation -/
theorem accurate_estimation (s : Scale) (arrow : ArrowPosition s) 
    (h1 : s.lower = 10.15) (h2 : s.upper = 10.4) : 
    mostAccurateEstimation s arrow = 10.3 := by
  sorry

end NUMINAMATH_CALUDE_accurate_estimation_l962_96214


namespace NUMINAMATH_CALUDE_sqrt_3_irrational_l962_96287

theorem sqrt_3_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_irrational_l962_96287


namespace NUMINAMATH_CALUDE_no_tiling_with_all_tetrominoes_l962_96232

/-- A tetromino is a shape consisting of 4 squares that can be rotated but not reflected. -/
structure Tetromino :=
  (squares : Fin 4 → (Fin 2 × Fin 2))

/-- There are exactly 7 different tetrominoes. -/
axiom num_tetrominoes : {n : ℕ // n = 7}

/-- A 4 × n rectangle. -/
def Rectangle (n : ℕ) := Fin 4 × Fin n

/-- A tiling of a rectangle with tetrominoes. -/
def Tiling (n : ℕ) := Rectangle n → Tetromino

/-- Theorem: It is impossible to tile a 4 × n rectangle with one copy of each of the 7 different tetrominoes. -/
theorem no_tiling_with_all_tetrominoes (n : ℕ) :
  ¬∃ (t : Tiling n), (∀ tetromino : Tetromino, ∃! (x : Rectangle n), t x = tetromino) :=
sorry

end NUMINAMATH_CALUDE_no_tiling_with_all_tetrominoes_l962_96232


namespace NUMINAMATH_CALUDE_point_on_terminal_side_l962_96210

/-- Proves that for a point P(-√3, y) on the terminal side of angle β, 
    where sin β = √13/13, the value of y is 1/2. -/
theorem point_on_terminal_side (β : ℝ) (y : ℝ) : 
  (∃ P : ℝ × ℝ, P.1 = -Real.sqrt 3 ∧ P.2 = y ∧ 
    Real.sin β = Real.sqrt 13 / 13 ∧ 
    (P.1 ≥ 0 ∨ (P.1 < 0 ∧ P.2 > 0))) → 
  y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_terminal_side_l962_96210


namespace NUMINAMATH_CALUDE_wendi_chicken_count_l962_96298

/-- The number of chickens Wendi has after various changes --/
def final_chicken_count (initial : ℕ) : ℕ :=
  let doubled := initial * 2
  let after_loss := doubled - 1
  let additional := 6
  after_loss + additional

/-- Theorem stating that starting with 4 chickens, Wendi ends up with 13 chickens --/
theorem wendi_chicken_count : final_chicken_count 4 = 13 := by
  sorry

end NUMINAMATH_CALUDE_wendi_chicken_count_l962_96298


namespace NUMINAMATH_CALUDE_lcm_of_15_25_35_l962_96277

theorem lcm_of_15_25_35 : Nat.lcm (Nat.lcm 15 25) 35 = 525 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_15_25_35_l962_96277


namespace NUMINAMATH_CALUDE_number_operations_l962_96229

theorem number_operations (n y : ℝ) : ((2 * n + y) / 2) - n = y / 2 := by
  sorry

end NUMINAMATH_CALUDE_number_operations_l962_96229


namespace NUMINAMATH_CALUDE_correct_transformation_l962_96296

theorem correct_transformation : (-2 : ℚ) * (1/2 : ℚ) * (-5 : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_correct_transformation_l962_96296


namespace NUMINAMATH_CALUDE_min_value_xyz_l962_96281

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 27) :
  x + 3 * y + 9 * z ≥ 27 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 27 ∧ x₀ + 3 * y₀ + 9 * z₀ = 27 :=
by sorry

end NUMINAMATH_CALUDE_min_value_xyz_l962_96281


namespace NUMINAMATH_CALUDE_a_equals_seven_l962_96297

theorem a_equals_seven (A B : Set ℝ) (a : ℝ) : 
  A = {1, 2, a} → B = {1, 7} → B ⊆ A → a = 7 := by
  sorry

end NUMINAMATH_CALUDE_a_equals_seven_l962_96297


namespace NUMINAMATH_CALUDE_sqrt_two_is_quadratic_radical_l962_96245

-- Define what a quadratic radical is
def is_quadratic_radical (x : ℝ) : Prop :=
  ∃ (y : ℝ), x = Real.sqrt y ∧ ¬ (∃ (n : ℤ), x = n)

-- Theorem statement
theorem sqrt_two_is_quadratic_radical : 
  is_quadratic_radical (Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_sqrt_two_is_quadratic_radical_l962_96245


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l962_96267

/-- Given a line y = mx - 3 intersecting the ellipse 4x^2 + 25y^2 = 100,
    the possible slopes m satisfy m^2 ≥ 4/41 -/
theorem line_ellipse_intersection_slopes (m : ℝ) : 
  (∃ x y : ℝ, 4 * x^2 + 25 * y^2 = 100 ∧ y = m * x - 3) → m^2 ≥ 4/41 := by
  sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l962_96267


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l962_96231

theorem complex_number_quadrant : ∃ (z : ℂ), z = Complex.I * (1 - Complex.I) ∧ z.re > 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l962_96231


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l962_96263

theorem binomial_coefficient_equality (m : ℕ) : 
  (Nat.choose 13 (m + 1) = Nat.choose 13 (2 * m - 3)) ↔ (m = 4 ∨ m = 5) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l962_96263


namespace NUMINAMATH_CALUDE_hyperbola_line_slope_l962_96293

/-- Given a hyperbola and a line intersecting it, prove that the slope of the line is 6 -/
theorem hyperbola_line_slope :
  ∀ (A B : ℝ × ℝ),
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let P := (2, 1)
  -- Hyperbola equation
  (x₁^2 - y₁^2/3 = 1) →
  (x₂^2 - y₂^2/3 = 1) →
  -- P is the midpoint of AB
  (2 = (x₁ + x₂)/2) →
  (1 = (y₁ + y₂)/2) →
  -- Slope of AB
  ((y₁ - y₂)/(x₁ - x₂) = 6) :=
by
  sorry


end NUMINAMATH_CALUDE_hyperbola_line_slope_l962_96293


namespace NUMINAMATH_CALUDE_f_increasing_f_sum_zero_l962_96228

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 2 / (2^x + 1)

theorem f_increasing (a : ℝ) : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂ := by sorry

theorem f_sum_zero : 
  f 1 (-5) + f 1 (-3) + f 1 (-1) + f 1 1 + f 1 3 + f 1 5 = 0 := by sorry

end NUMINAMATH_CALUDE_f_increasing_f_sum_zero_l962_96228


namespace NUMINAMATH_CALUDE_high_school_relationships_l962_96259

/-- The number of people in the group -/
def n : ℕ := 12

/-- The number of categories for each pair -/
def categories : ℕ := 3

/-- The number of pairs in a group of n people -/
def pairs (n : ℕ) : ℕ := n.choose 2

/-- The total number of pair categorizations -/
def totalCategorizations (n : ℕ) (categories : ℕ) : ℕ :=
  pairs n * categories

theorem high_school_relationships :
  totalCategorizations n categories = 198 := by sorry

end NUMINAMATH_CALUDE_high_school_relationships_l962_96259


namespace NUMINAMATH_CALUDE_certain_number_problem_l962_96261

theorem certain_number_problem : ∃! x : ℕ+, 220030 = (x + 445) * (2 * (x - 445)) + 30 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l962_96261


namespace NUMINAMATH_CALUDE_mini_van_tank_capacity_l962_96290

/-- Represents the problem of determining the capacity of a mini-van's tank. -/
theorem mini_van_tank_capacity 
  (service_cost : ℝ) 
  (fuel_cost : ℝ) 
  (num_mini_vans : ℕ) 
  (num_trucks : ℕ) 
  (total_cost : ℝ) 
  (truck_tank_ratio : ℝ) 
  (h1 : service_cost = 2.30)
  (h2 : fuel_cost = 0.70)
  (h3 : num_mini_vans = 4)
  (h4 : num_trucks = 2)
  (h5 : total_cost = 396)
  (h6 : truck_tank_ratio = 2.20) : 
  ∃ (mini_van_capacity : ℝ),
    mini_van_capacity = 65 ∧
    total_cost = 
      (num_mini_vans + num_trucks) * service_cost + 
      (num_mini_vans * mini_van_capacity + num_trucks * (truck_tank_ratio * mini_van_capacity)) * fuel_cost :=
by sorry

end NUMINAMATH_CALUDE_mini_van_tank_capacity_l962_96290


namespace NUMINAMATH_CALUDE_pocket_money_problem_l962_96205

/-- Pocket money problem -/
theorem pocket_money_problem (a b c d e : ℕ) : 
  (a + b + c + d + e) / 5 = 2300 →
  (a + b) / 2 = 3000 →
  (b + c) / 2 = 2100 →
  (c + d) / 2 = 2750 →
  a = b + 800 →
  d = 3900 := by
sorry

end NUMINAMATH_CALUDE_pocket_money_problem_l962_96205


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l962_96279

open Set

def U : Set Nat := {1,2,3,4,5,6,7,8}
def A : Set Nat := {1,2,3}
def B : Set Nat := {2,3,4,5}

theorem complement_intersection_theorem : 
  (Aᶜ ∪ Bᶜ) ∩ U = {1,4,5,6,7,8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l962_96279


namespace NUMINAMATH_CALUDE_unique_exponent_solution_l962_96246

theorem unique_exponent_solution :
  ∃! w : ℤ, (3 : ℝ) ^ 6 * (3 : ℝ) ^ w = (3 : ℝ) ^ 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_exponent_solution_l962_96246


namespace NUMINAMATH_CALUDE_stewart_farm_ratio_l962_96256

/-- Represents the Stewart farm with sheep and horses -/
structure Farm where
  sheep : ℕ
  total_horse_food : ℕ
  food_per_horse : ℕ

/-- Calculates the number of horses on the farm -/
def num_horses (f : Farm) : ℕ := f.total_horse_food / f.food_per_horse

/-- Calculates the ratio of sheep to horses as a pair of natural numbers -/
def sheep_to_horse_ratio (f : Farm) : ℕ × ℕ :=
  let gcd := Nat.gcd f.sheep (num_horses f)
  (f.sheep / gcd, num_horses f / gcd)

/-- Theorem stating that for the given farm conditions, the sheep to horse ratio is 2:7 -/
theorem stewart_farm_ratio :
  let f : Farm := ⟨16, 12880, 230⟩
  sheep_to_horse_ratio f = (2, 7) := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_ratio_l962_96256


namespace NUMINAMATH_CALUDE_solution_x_equals_three_l962_96251

theorem solution_x_equals_three : ∃ (f : ℝ → ℝ), f 3 = 0 ∧ (∀ x, f x = 0 → x = 3) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_solution_x_equals_three_l962_96251


namespace NUMINAMATH_CALUDE_selina_leaves_with_30_l962_96220

/-- The amount of money Selina leaves the store with after selling and buying clothes -/
def selina_final_money (pants_price shorts_price shirts_price : ℕ) 
  (pants_sold shorts_sold shirts_sold : ℕ) 
  (shirts_bought new_shirt_price : ℕ) : ℕ :=
  pants_price * pants_sold + shorts_price * shorts_sold + shirts_price * shirts_sold - 
  shirts_bought * new_shirt_price

/-- Theorem stating that Selina leaves the store with $30 -/
theorem selina_leaves_with_30 : 
  selina_final_money 5 3 4 3 5 5 2 10 = 30 := by
  sorry

end NUMINAMATH_CALUDE_selina_leaves_with_30_l962_96220


namespace NUMINAMATH_CALUDE_complement_of_A_l962_96260

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}

theorem complement_of_A :
  (U \ A) = {2, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l962_96260


namespace NUMINAMATH_CALUDE_binomial_20_19_l962_96237

theorem binomial_20_19 : Nat.choose 20 19 = 20 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_19_l962_96237


namespace NUMINAMATH_CALUDE_expression_evaluation_l962_96240

theorem expression_evaluation :
  3 + 2 * Real.sqrt 3 + (3 + 2 * Real.sqrt 3)⁻¹ + (2 * Real.sqrt 3 - 3)⁻¹ = 3 + (16 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l962_96240


namespace NUMINAMATH_CALUDE_f_order_magnitude_l962_96285

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x < y → f x < f y

-- State the theorem
theorem f_order_magnitude 
  (h1 : is_even f) 
  (h2 : is_increasing_on_nonneg f) : 
  f (-π) > f 3 ∧ f 3 > f (-2) :=
sorry

end NUMINAMATH_CALUDE_f_order_magnitude_l962_96285


namespace NUMINAMATH_CALUDE_rosy_fish_count_l962_96233

/-- The number of fish Lilly has -/
def lilly_fish : ℕ := 10

/-- The total number of fish Lilly and Rosy have together -/
def total_fish : ℕ := 21

/-- The number of fish Rosy has -/
def rosy_fish : ℕ := total_fish - lilly_fish

theorem rosy_fish_count : rosy_fish = 11 := by
  sorry

end NUMINAMATH_CALUDE_rosy_fish_count_l962_96233


namespace NUMINAMATH_CALUDE_factorial_difference_l962_96268

theorem factorial_difference : Nat.factorial 10 - Nat.factorial 9 = 3265920 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l962_96268


namespace NUMINAMATH_CALUDE_gcd_powers_of_two_l962_96209

theorem gcd_powers_of_two : Nat.gcd (2^115 - 1) (2^105 - 1) = 2^10 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_powers_of_two_l962_96209


namespace NUMINAMATH_CALUDE_curve_inequality_l962_96224

/-- Given real numbers a, b, c satisfying certain conditions, 
    prove an inequality for points on a specific curve. -/
theorem curve_inequality (a b c : ℝ) 
  (h1 : b^2 - a*c < 0) 
  (h2 : ∀ x y : ℝ, x > 0 → y > 0 → 
    a * (Real.log x)^2 + 2*b*(Real.log x * Real.log y) + c * (Real.log y)^2 = 1 → 
    (x = 10 ∧ y = 1/10) ∨ 
    (-1 / Real.sqrt (a*c - b^2) ≤ Real.log (x*y) ∧ 
     Real.log (x*y) ≤ 1 / Real.sqrt (a*c - b^2))) : 
  ∀ x y : ℝ, x > 0 → y > 0 → 
    a * (Real.log x)^2 + 2*b*(Real.log x * Real.log y) + c * (Real.log y)^2 = 1 → 
    -1 / Real.sqrt (a*c - b^2) ≤ Real.log (x*y) ∧ 
    Real.log (x*y) ≤ 1 / Real.sqrt (a*c - b^2) := by
  sorry

end NUMINAMATH_CALUDE_curve_inequality_l962_96224


namespace NUMINAMATH_CALUDE_expression_simplification_l962_96262

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 + 3) :
  (x^2 - 1) / (x^2 - 6*x + 9) * (1 - x / (x - 1)) / ((x + 1) / (x - 3)) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l962_96262


namespace NUMINAMATH_CALUDE_exists_m_between_alpha_beta_l962_96222

theorem exists_m_between_alpha_beta (α β : ℝ) (h1 : 0 ≤ α) (h2 : α < β) (h3 : β ≤ 1) :
  ∃ m : ℕ, α < (Nat.totient m : ℝ) / m ∧ (Nat.totient m : ℝ) / m < β := by
  sorry

end NUMINAMATH_CALUDE_exists_m_between_alpha_beta_l962_96222


namespace NUMINAMATH_CALUDE_cyclic_wins_count_l962_96265

/-- Represents a round-robin tournament. -/
structure Tournament where
  /-- The number of teams in the tournament. -/
  num_teams : ℕ
  /-- The number of wins for each team. -/
  wins_per_team : ℕ
  /-- The number of losses for each team. -/
  losses_per_team : ℕ
  /-- No ties in the tournament. -/
  no_ties : wins_per_team + losses_per_team = num_teams - 1

/-- The number of sets of three teams {A, B, C} where A beat B, B beat C, and C beat A. -/
def cyclic_wins (t : Tournament) : ℕ := sorry

/-- The main theorem stating the number of cyclic win sets in the given tournament. -/
theorem cyclic_wins_count (t : Tournament) 
  (h1 : t.num_teams = 21)
  (h2 : t.wins_per_team = 10)
  (h3 : t.losses_per_team = 10) :
  cyclic_wins t = 385 := by sorry

end NUMINAMATH_CALUDE_cyclic_wins_count_l962_96265


namespace NUMINAMATH_CALUDE_distance_swam_against_current_l962_96299

/-- Proves that the distance swam against the current is 10 km -/
theorem distance_swam_against_current
  (still_water_speed : ℝ)
  (water_speed : ℝ)
  (time_taken : ℝ)
  (h1 : still_water_speed = 12)
  (h2 : water_speed = 2)
  (h3 : time_taken = 1) :
  still_water_speed - water_speed * time_taken = 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_swam_against_current_l962_96299


namespace NUMINAMATH_CALUDE_water_duration_village_water_duration_l962_96221

/-- Calculates how long water will last in a village given specific conditions. -/
theorem water_duration (water_per_person : ℝ) (small_households : ℕ) (large_households : ℕ) 
  (small_household_size : ℕ) (large_household_size : ℕ) (total_water : ℝ) : ℝ :=
  let water_usage_per_month := 
    (small_households * small_household_size * water_per_person) + 
    (large_households * large_household_size * water_per_person)
  total_water / water_usage_per_month

/-- Proves that the water lasts approximately 4.31 months under given conditions. -/
theorem village_water_duration : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |water_duration 20 7 3 2 5 2500 - 4.31| < ε :=
sorry

end NUMINAMATH_CALUDE_water_duration_village_water_duration_l962_96221


namespace NUMINAMATH_CALUDE_folded_paper_length_l962_96223

/-- Given a rectangle with sides of lengths 1 and √2, where one vertex is folded to touch the opposite side, the length d of the folded edge is √2 - 1. -/
theorem folded_paper_length (a b d : ℝ) : 
  a = 1 → b = Real.sqrt 2 → 
  d = Real.sqrt ((b - d)^2 + a^2) → 
  d = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_folded_paper_length_l962_96223


namespace NUMINAMATH_CALUDE_sqrt_product_property_l962_96236

theorem sqrt_product_property : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_property_l962_96236


namespace NUMINAMATH_CALUDE_sports_meeting_formation_l962_96202

/-- The number of performers in the initial formation -/
def initial_performers : ℕ := sorry

/-- The number of performers after adding 16 -/
def after_addition : ℕ := initial_performers + 16

/-- The number of performers after 15 leave -/
def after_leaving : ℕ := after_addition - 15

theorem sports_meeting_formation :
  (∃ n : ℕ, initial_performers = 8 * n) ∧ 
  (∃ m : ℕ, after_addition = m * m) ∧
  (∃ k : ℕ, after_leaving = k * k) →
  initial_performers = 48 := by sorry

end NUMINAMATH_CALUDE_sports_meeting_formation_l962_96202


namespace NUMINAMATH_CALUDE_sum_other_y_coordinates_specific_parallelogram_l962_96219

/-- A parallelogram with two opposite corners given -/
structure Parallelogram where
  corner1 : ℝ × ℝ
  corner2 : ℝ × ℝ

/-- The sum of y-coordinates of the other two vertices of the parallelogram -/
def sumOtherYCoordinates (p : Parallelogram) : ℝ :=
  (p.corner1.2 + p.corner2.2)

theorem sum_other_y_coordinates_specific_parallelogram :
  let p := Parallelogram.mk (2, 15) (8, -6)
  sumOtherYCoordinates p = 9 := by
  sorry

#check sum_other_y_coordinates_specific_parallelogram

end NUMINAMATH_CALUDE_sum_other_y_coordinates_specific_parallelogram_l962_96219


namespace NUMINAMATH_CALUDE_shelter_cats_l962_96243

theorem shelter_cats (cats dogs : ℕ) : 
  (cats : ℚ) / dogs = 15 / 7 →
  cats / (dogs + 12) = 15 / 11 →
  cats = 45 := by
sorry

end NUMINAMATH_CALUDE_shelter_cats_l962_96243


namespace NUMINAMATH_CALUDE_inequality_proof_l962_96255

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  2 * x + 1 / (x^2 - 2*x*y + y^2) ≥ 2 * y + 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l962_96255


namespace NUMINAMATH_CALUDE_alternating_exponent_inequality_l962_96235

theorem alternating_exponent_inequality (n : ℕ) (h : n ≥ 1) :
  2^(3^n) > 3^(2^(n-1)) := by
  sorry

end NUMINAMATH_CALUDE_alternating_exponent_inequality_l962_96235


namespace NUMINAMATH_CALUDE_watch_sale_gain_percentage_l962_96249

/-- Proves that for a watch with a given cost price, sold at a loss, 
    if the selling price is increased by a certain amount, 
    the resulting gain percentage is as expected. -/
theorem watch_sale_gain_percentage 
  (cost_price : ℝ) 
  (loss_percentage : ℝ) 
  (price_increase : ℝ) : 
  cost_price = 1200 →
  loss_percentage = 10 →
  price_increase = 168 →
  let loss_amount := (loss_percentage / 100) * cost_price
  let initial_selling_price := cost_price - loss_amount
  let new_selling_price := initial_selling_price + price_increase
  let gain_amount := new_selling_price - cost_price
  let gain_percentage := (gain_amount / cost_price) * 100
  gain_percentage = 4 := by
sorry


end NUMINAMATH_CALUDE_watch_sale_gain_percentage_l962_96249


namespace NUMINAMATH_CALUDE_line_translation_down_5_l962_96225

/-- Represents a line in 2D space -/
structure Line where
  slope : ℚ
  intercept : ℚ

/-- Translates a line vertically by a given amount -/
def translateLine (l : Line) (dy : ℚ) : Line :=
  { slope := l.slope, intercept := l.intercept + dy }

theorem line_translation_down_5 :
  let original_line := { slope := -1/2, intercept := 2 : Line }
  let translated_line := translateLine original_line (-5)
  translated_line = { slope := -1/2, intercept := -3 : Line } := by
  sorry

end NUMINAMATH_CALUDE_line_translation_down_5_l962_96225


namespace NUMINAMATH_CALUDE_cube_with_corners_removed_faces_l962_96218

-- Define the properties of the cube
def cube_side_length : ℝ := 3
def small_cube_side_length : ℝ := 1
def initial_faces : ℕ := 6
def corners_in_cube : ℕ := 8
def new_faces_per_corner : ℕ := 3

-- Theorem statement
theorem cube_with_corners_removed_faces :
  initial_faces + corners_in_cube * new_faces_per_corner = 30 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_corners_removed_faces_l962_96218


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_intersections_l962_96239

/-- The number of unit cubes a space diagonal passes through in a rectangular prism -/
def spaceDiagonalIntersections (a b c : ℕ) : ℕ :=
  a + b + c - Nat.gcd a b - Nat.gcd b c - Nat.gcd c a + Nat.gcd a (Nat.gcd b c)

/-- Theorem: For a 150 × 324 × 375 rectangular prism, the space diagonal passes through 768 unit cubes -/
theorem rectangular_prism_diagonal_intersections :
  spaceDiagonalIntersections 150 324 375 = 768 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_intersections_l962_96239


namespace NUMINAMATH_CALUDE_sine_cosine_parity_l962_96254

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem sine_cosine_parity (sine cosine : ℝ → ℝ) 
  (h1 : ∀ x, sine (-x) = -(sine x)) 
  (h2 : ∀ x, cosine (-x) = cosine x) : 
  is_odd_function sine ∧ is_even_function cosine := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_parity_l962_96254


namespace NUMINAMATH_CALUDE_term_206_of_specific_sequence_l962_96241

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r ^ (n - 1)

theorem term_206_of_specific_sequence :
  let a₁ := 10
  let a₂ := -10
  let r := a₂ / a₁
  geometric_sequence a₁ r 206 = -10 := by sorry

end NUMINAMATH_CALUDE_term_206_of_specific_sequence_l962_96241


namespace NUMINAMATH_CALUDE_xyz_maximum_l962_96286

theorem xyz_maximum (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_eq : x * y - z = (x - z) * (y - z)) (h_sum : x + y + z = 1) :
  x * y * z ≤ 1 / 27 :=
sorry

end NUMINAMATH_CALUDE_xyz_maximum_l962_96286


namespace NUMINAMATH_CALUDE_simplify_complex_expression_l962_96208

-- Define the complex number i
def i : ℂ := Complex.I

-- Theorem statement
theorem simplify_complex_expression : i * (1 - i)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_expression_l962_96208


namespace NUMINAMATH_CALUDE_interior_angles_increase_l962_96289

/-- The sum of interior angles of a convex polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

theorem interior_angles_increase (n : ℕ) :
  sum_interior_angles n = 1620 → sum_interior_angles (n + 3) = 2160 := by
  sorry

end NUMINAMATH_CALUDE_interior_angles_increase_l962_96289


namespace NUMINAMATH_CALUDE_beth_age_proof_l962_96271

/-- Beth's current age -/
def beth_age : ℕ := 18

/-- Beth's sister's current age -/
def sister_age : ℕ := 5

/-- Years into the future when Beth will be twice her sister's age -/
def future_years : ℕ := 8

theorem beth_age_proof :
  beth_age = 18 ∧
  sister_age = 5 ∧
  beth_age + future_years = 2 * (sister_age + future_years) :=
by sorry

end NUMINAMATH_CALUDE_beth_age_proof_l962_96271


namespace NUMINAMATH_CALUDE_toms_fruit_bowl_l962_96264

/-- The number of fruits remaining in Tom's fruit bowl after eating some fruits -/
def remaining_fruits (initial_oranges initial_lemons eaten : ℕ) : ℕ :=
  initial_oranges + initial_lemons - eaten

/-- Theorem: Given Tom's fruit bowl with 3 oranges and 6 lemons, after eating 3 fruits, 6 fruits remain -/
theorem toms_fruit_bowl : remaining_fruits 3 6 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_toms_fruit_bowl_l962_96264


namespace NUMINAMATH_CALUDE_largest_reciprocal_l962_96252

theorem largest_reciprocal (a b c d e : ℚ) : 
  a = 3/4 → b = 5/3 → c = -1/6 → d = 7 → e = 3 →
  (1/a > 1/b ∧ 1/a > 1/c ∧ 1/a > 1/d ∧ 1/a > 1/e) := by
  sorry

end NUMINAMATH_CALUDE_largest_reciprocal_l962_96252


namespace NUMINAMATH_CALUDE_power_function_properties_l962_96292

noncomputable def f (x : ℝ) : ℝ := x ^ (1/2)

theorem power_function_properties :
  ∀ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ < x₂ →
    (x₁ * f x₁ < x₂ * f x₂) ∧
    (f x₁ / x₁ > f x₂ / x₂) :=
by sorry

end NUMINAMATH_CALUDE_power_function_properties_l962_96292


namespace NUMINAMATH_CALUDE_improper_fraction_decomposition_l962_96270

theorem improper_fraction_decomposition (x : ℝ) :
  (6 * x^3 + 5 * x^2 + 3 * x - 4) / (x^2 + 4) = 6 * x + 5 - (21 * x + 24) / (x^2 + 4) := by
  sorry

end NUMINAMATH_CALUDE_improper_fraction_decomposition_l962_96270


namespace NUMINAMATH_CALUDE_scientific_notation_of_0_00625_l962_96276

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_0_00625 :
  toScientificNotation 0.00625 = ScientificNotation.mk 6.25 (-3) sorry := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_0_00625_l962_96276


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l962_96230

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := fun x ↦ 2 * x^2 - 2
  ∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = -1 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ 
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l962_96230


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l962_96266

theorem degree_to_radian_conversion (x : Real) : 
  x * (π / 180) = -5 * π / 3 → x = -300 :=
by sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l962_96266


namespace NUMINAMATH_CALUDE_line_passes_through_point_l962_96257

theorem line_passes_through_point (m : ℝ) : m * 1 - 1 + 1 - m = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l962_96257


namespace NUMINAMATH_CALUDE_eight_faucets_fill_time_l962_96272

/-- The time (in seconds) it takes for a given number of faucets to fill a tub of a given volume -/
def fill_time (num_faucets : ℕ) (volume : ℝ) : ℝ :=
  -- Definition to be filled based on the problem conditions
  sorry

theorem eight_faucets_fill_time :
  -- Given conditions
  (fill_time 4 200 = 8 * 60) →  -- 4 faucets fill 200 gallons in 8 minutes (converted to seconds)
  (∀ n v, fill_time n v = fill_time 1 v / n) →  -- All faucets dispense water at the same rate
  -- Conclusion
  (fill_time 8 50 = 60) :=
by
  sorry

end NUMINAMATH_CALUDE_eight_faucets_fill_time_l962_96272


namespace NUMINAMATH_CALUDE_cosine_product_equality_l962_96212

theorem cosine_product_equality : Real.cos (2 * Real.pi / 31) * Real.cos (4 * Real.pi / 31) * Real.cos (8 * Real.pi / 31) * Real.cos (16 * Real.pi / 31) * Real.cos (32 * Real.pi / 31) * 3.418 = 1 / 32 := by
  sorry

end NUMINAMATH_CALUDE_cosine_product_equality_l962_96212


namespace NUMINAMATH_CALUDE_rahul_savings_l962_96234

/-- Rahul's savings problem -/
theorem rahul_savings (nsc ppf : ℚ) : 
  (1/3 : ℚ) * nsc = (1/2 : ℚ) * ppf →
  nsc + ppf = 180000 →
  ppf = 72000 := by
sorry

end NUMINAMATH_CALUDE_rahul_savings_l962_96234


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l962_96275

theorem quadratic_equation_solution : ∃ (x c d : ℕ), 
  (x^2 + 14*x = 84) ∧ 
  (x = Real.sqrt c - d) ∧ 
  (c > 0) ∧ 
  (d > 0) ∧ 
  (c + d = 140) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l962_96275


namespace NUMINAMATH_CALUDE_washing_time_calculation_l962_96280

def clothes_time : ℕ := 30

def towels_time (clothes_time : ℕ) : ℕ := 2 * clothes_time

def sheets_time (towels_time : ℕ) : ℕ := towels_time - 15

def total_washing_time (clothes_time towels_time sheets_time : ℕ) : ℕ :=
  clothes_time + towels_time + sheets_time

theorem washing_time_calculation :
  total_washing_time clothes_time (towels_time clothes_time) (sheets_time (towels_time clothes_time)) = 135 := by
  sorry

end NUMINAMATH_CALUDE_washing_time_calculation_l962_96280


namespace NUMINAMATH_CALUDE_restaurant_cooks_count_l962_96295

theorem restaurant_cooks_count (initial_cooks : ℕ) (initial_waiters : ℕ) 
  (h1 : initial_cooks * 8 = initial_waiters * 3) 
  (h2 : initial_cooks * 4 = (initial_waiters + 12) * 1) : 
  initial_cooks = 9 := by
sorry

end NUMINAMATH_CALUDE_restaurant_cooks_count_l962_96295


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l962_96238

/-- The distance between the foci of the ellipse 9x^2 + y^2 = 900 is 40√2 -/
theorem ellipse_foci_distance :
  let a : ℝ := Real.sqrt 100
  let b : ℝ := Real.sqrt 900
  let c : ℝ := Real.sqrt (b^2 - a^2)
  2 * c = 40 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l962_96238


namespace NUMINAMATH_CALUDE_system_inequalities_solution_range_l962_96274

theorem system_inequalities_solution_range (a : ℚ) : 
  (∃! (s : Finset ℤ), s.card = 5 ∧ 
    (∀ x : ℤ, x ∈ s ↔ (((2 * x + 5 : ℚ) / 3 > x - 5) ∧ ((x + 3 : ℚ) / 2 < x + a)))) →
  (-6 < a ∧ a ≤ -11/2) :=
sorry

end NUMINAMATH_CALUDE_system_inequalities_solution_range_l962_96274


namespace NUMINAMATH_CALUDE_probability_matches_given_l962_96200

def total_pens : ℕ := 8
def defective_pens : ℕ := 3
def pens_bought : ℕ := 2

def probability_no_defective (total : ℕ) (defective : ℕ) (bought : ℕ) : ℚ :=
  (Nat.choose (total - defective) bought : ℚ) / (Nat.choose total bought : ℚ)

theorem probability_matches_given :
  probability_no_defective total_pens defective_pens pens_bought = 5 / 14 :=
by sorry

end NUMINAMATH_CALUDE_probability_matches_given_l962_96200


namespace NUMINAMATH_CALUDE_two_digit_numbers_with_gcd_lcm_l962_96203

theorem two_digit_numbers_with_gcd_lcm (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 →
  Nat.gcd a b = 8 →
  Nat.lcm a b = 96 →
  a + b = 56 := by
sorry

end NUMINAMATH_CALUDE_two_digit_numbers_with_gcd_lcm_l962_96203
