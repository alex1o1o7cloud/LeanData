import Mathlib

namespace NUMINAMATH_CALUDE_expansion_coefficient_constraint_l1119_111920

theorem expansion_coefficient_constraint (k : ℕ+) :
  (15 : ℝ) * (k : ℝ)^4 < 120 → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_constraint_l1119_111920


namespace NUMINAMATH_CALUDE_smallest_n_for_square_root_solution_l1119_111945

def is_square_integer (x : ℚ) : Prop :=
  ∃ m : ℤ, x = m^2

theorem smallest_n_for_square_root (n : ℕ) : Prop :=
  n ≥ 2 ∧ 
  is_square_integer ((n + 1) * (2 * n + 1) / 6) ∧
  ∀ k : ℕ, k ≥ 2 ∧ k < n → ¬is_square_integer ((k + 1) * (2 * k + 1) / 6)

theorem solution : smallest_n_for_square_root 337 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_square_root_solution_l1119_111945


namespace NUMINAMATH_CALUDE_increasing_interval_of_sine_l1119_111955

theorem increasing_interval_of_sine (f : ℝ → ℝ) (h : f = λ x => Real.sin (2 * x + π / 6)) :
  ∀ x ∈ Set.Icc (-π / 3) (π / 6), ∀ y ∈ Set.Icc (-π / 3) (π / 6),
    x < y → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_increasing_interval_of_sine_l1119_111955


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sin_problem_l1119_111900

theorem arithmetic_sequence_sin_problem (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 5 + a 6 = 10 * Real.pi / 3 →                    -- given condition
  Real.sin (a 4 + a 7) = -Real.sqrt 3 / 2 :=        -- conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sin_problem_l1119_111900


namespace NUMINAMATH_CALUDE_ladder_problem_l1119_111984

theorem ladder_problem (ladder_length height base : ℝ) :
  ladder_length = 15 ∧ height = 12 ∧ ladder_length^2 = height^2 + base^2 → base = 9 := by
  sorry

end NUMINAMATH_CALUDE_ladder_problem_l1119_111984


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_l1119_111967

/-- Given a line and a circle with no common points, prove that a line through a point
    on that line intersects a specific ellipse at exactly two points. -/
theorem line_ellipse_intersection (m n : ℝ) : 
  (∀ x y : ℝ, m*x + n*y - 3 = 0 → x^2 + y^2 ≠ 3) →
  0 < m^2 + n^2 →
  m^2 + n^2 < 3 →
  ∃! (p : ℕ), p = 2 ∧ 
    ∃ (x₁ y₁ x₂ y₂ : ℝ), 
      (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
      (∃ (k : ℝ), x₁ = m*k ∧ y₁ = n*k) ∧
      (∃ (k : ℝ), x₂ = m*k ∧ y₂ = n*k) ∧
      x₁^2/7 + y₁^2/3 = 1 ∧
      x₂^2/7 + y₂^2/3 = 1 ∧
      (∀ x y : ℝ, (∃ k : ℝ, x = m*k ∧ y = n*k) → 
        x^2/7 + y^2/3 = 1 → (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
by sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_l1119_111967


namespace NUMINAMATH_CALUDE_cone_volume_from_cylinder_l1119_111954

/-- Given a cylinder with volume 72π cm³, prove that a cone with the same radius
    and half the height has a volume of 12π cm³. -/
theorem cone_volume_from_cylinder (r h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) :
  π * r^2 * h = 72 * π →
  (1/3) * π * r^2 * (h/2) = 12 * π := by
sorry

end NUMINAMATH_CALUDE_cone_volume_from_cylinder_l1119_111954


namespace NUMINAMATH_CALUDE_charlie_snowballs_l1119_111903

theorem charlie_snowballs (lucy_snowballs : ℕ) (charlie_snowballs : ℕ) : 
  lucy_snowballs = 19 → 
  charlie_snowballs = lucy_snowballs + 31 → 
  charlie_snowballs = 50 := by
  sorry

end NUMINAMATH_CALUDE_charlie_snowballs_l1119_111903


namespace NUMINAMATH_CALUDE_no_quadratic_transform_l1119_111974

/-- A polynomial function of degree 2 or less -/
def QuadraticPolynomial (a b c : ℚ) : ℚ → ℚ := λ x => a * x^2 + b * x + c

/-- Theorem stating that no quadratic polynomial can transform (1,4,7) to (1,10,7) -/
theorem no_quadratic_transform :
  ¬ ∃ (a b c : ℚ), 
    (QuadraticPolynomial a b c 1 = 1) ∧ 
    (QuadraticPolynomial a b c 4 = 10) ∧ 
    (QuadraticPolynomial a b c 7 = 7) := by
  sorry

end NUMINAMATH_CALUDE_no_quadratic_transform_l1119_111974


namespace NUMINAMATH_CALUDE_correct_swap_l1119_111971

-- Define the initial values
def a : Int := 2
def b : Int := -6

-- Define the swap operation
def swap (x y : Int) : (Int × Int) :=
  let c := x
  let new_x := y
  let new_y := c
  (new_x, new_y)

-- Theorem statement
theorem correct_swap :
  swap a b = (-6, 2) := by
  sorry

end NUMINAMATH_CALUDE_correct_swap_l1119_111971


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1119_111999

/-- Given an arithmetic sequence {a_n} where a_2 = 1 and a_3 + a_5 = 4,
    the common difference of the sequence is 1/2. -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℚ) -- The sequence as a function from natural numbers to rationals
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) -- Arithmetic sequence condition
  (h_a2 : a 2 = 1) -- Given: a_2 = 1
  (h_sum : a 3 + a 5 = 4) -- Given: a_3 + a_5 = 4
  : ∃ d : ℚ, d = 1/2 ∧ ∀ n : ℕ, a (n + 1) - a n = d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1119_111999


namespace NUMINAMATH_CALUDE_grasshopper_impossibility_l1119_111953

/-- A point in the 2D plane with integer coordinates -/
structure Point where
  x : Int
  y : Int

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- Check if a move from p1 to p2 is parallel to the line segment from p3 to p4 -/
def parallel_move (p1 p2 p3 p4 : Point) : Prop :=
  (p2.x - p1.x) * (p4.y - p3.y) = (p2.y - p1.y) * (p4.x - p3.x)

/-- A valid move in the grasshopper game -/
inductive ValidMove : List Point → List Point → Prop where
  | move (p1 p2 p3 p1' : Point) (rest : List Point) :
      parallel_move p1 p1' p2 p3 →
      ValidMove [p1, p2, p3] (p1' :: rest)

/-- A sequence of valid moves -/
def ValidMoveSequence : List Point → List Point → Prop :=
  Relation.ReflTransGen ValidMove

/-- The main theorem: impossibility of reaching the final configuration -/
theorem grasshopper_impossibility :
  ¬∃ (final : List Point),
    ValidMoveSequence [Point.mk 1 0, Point.mk 0 0, Point.mk 0 1] final ∧
    final = [Point.mk 0 0, Point.mk (-1) (-1), Point.mk 1 1] :=
sorry


end NUMINAMATH_CALUDE_grasshopper_impossibility_l1119_111953


namespace NUMINAMATH_CALUDE_all_positive_l1119_111949

theorem all_positive (a b c : ℝ) 
  (sum_pos : a + b + c > 0) 
  (sum_prod_pos : a * b + b * c + c * a > 0) 
  (prod_pos : a * b * c > 0) : 
  a > 0 ∧ b > 0 ∧ c > 0 := by
  sorry

end NUMINAMATH_CALUDE_all_positive_l1119_111949


namespace NUMINAMATH_CALUDE_hike_attendance_l1119_111914

/-- The number of cars used for the hike -/
def num_cars : ℕ := 7

/-- The number of people in each car -/
def people_per_car : ℕ := 4

/-- The number of taxis used for the hike -/
def num_taxis : ℕ := 10

/-- The number of people in each taxi -/
def people_per_taxi : ℕ := 6

/-- The number of vans used for the hike -/
def num_vans : ℕ := 4

/-- The number of people in each van -/
def people_per_van : ℕ := 5

/-- The number of buses used for the hike -/
def num_buses : ℕ := 3

/-- The number of people in each bus -/
def people_per_bus : ℕ := 20

/-- The number of minibuses used for the hike -/
def num_minibuses : ℕ := 2

/-- The number of people in each minibus -/
def people_per_minibus : ℕ := 8

/-- The total number of people who went on the hike -/
def total_people : ℕ := 
  num_cars * people_per_car + 
  num_taxis * people_per_taxi + 
  num_vans * people_per_van + 
  num_buses * people_per_bus + 
  num_minibuses * people_per_minibus

theorem hike_attendance : total_people = 184 := by
  sorry

end NUMINAMATH_CALUDE_hike_attendance_l1119_111914


namespace NUMINAMATH_CALUDE_longest_segment_in_cylinder_l1119_111985

/-- The longest segment in a cylinder. -/
theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 3) (hh : h = 8) :
  Real.sqrt ((2 * r) ^ 2 + h ^ 2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_longest_segment_in_cylinder_l1119_111985


namespace NUMINAMATH_CALUDE_inverse_proportion_l1119_111902

theorem inverse_proportion (x₁ x₂ y₁ y₂ : ℝ) (h1 : x₁ ≠ 0) (h2 : x₂ ≠ 0) (h3 : y₁ ≠ 0) (h4 : y₂ ≠ 0)
  (h5 : ∃ k : ℝ, ∀ x y : ℝ, x * y = k) (h6 : x₁ / x₂ = 4 / 5) :
  y₁ / y₂ = 5 / 4 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_l1119_111902


namespace NUMINAMATH_CALUDE_sum_squared_equals_sixteen_l1119_111970

theorem sum_squared_equals_sixteen (a b : ℝ) (h : a + b = 4) : a^2 + 2*a*b + b^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_squared_equals_sixteen_l1119_111970


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1119_111958

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 + 5*x - 14 > 0} = {x : ℝ | x < -7 ∨ x > 2} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1119_111958


namespace NUMINAMATH_CALUDE_class_average_mark_l1119_111972

theorem class_average_mark (total_students : ℕ) (excluded_students : ℕ) (excluded_avg : ℚ) (remaining_avg : ℚ) :
  total_students = 33 →
  excluded_students = 3 →
  excluded_avg = 40 →
  remaining_avg = 95 →
  (total_students * (total_students - excluded_students) * remaining_avg +
   total_students * excluded_students * excluded_avg) /
  (total_students * total_students) = 90 := by
  sorry

end NUMINAMATH_CALUDE_class_average_mark_l1119_111972


namespace NUMINAMATH_CALUDE_smallest_common_factor_l1119_111908

theorem smallest_common_factor (n : ℕ) : 
  (∃ (k : ℕ), k > 1 ∧ k ∣ (11*n - 4) ∧ k ∣ (8*n - 5)) ∧ 
  (∀ (m : ℕ), m < n → ¬(∃ (k : ℕ), k > 1 ∧ k ∣ (11*m - 4) ∧ k ∣ (8*m - 5))) → 
  n = 15 :=
sorry

end NUMINAMATH_CALUDE_smallest_common_factor_l1119_111908


namespace NUMINAMATH_CALUDE_unique_solution_in_interval_l1119_111929

theorem unique_solution_in_interval (x : ℝ) :
  x ∈ Set.Icc 0 (Real.pi / 2) →
  ((2 - Real.sin (2 * x)) * Real.sin (x + Real.pi / 4) = 1) ↔
  (x = Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_in_interval_l1119_111929


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_minus_product_l1119_111978

theorem quadratic_roots_sum_minus_product (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ + 2 = 0 → 
  x₂^2 - 3*x₂ + 2 = 0 → 
  x₁ + x₂ - x₁*x₂ = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_minus_product_l1119_111978


namespace NUMINAMATH_CALUDE_base8_addition_problem_l1119_111975

-- Define the base
def base : ℕ := 8

-- Define the addition operation in base 8
def add_base8 (a b : ℕ) : ℕ := (a + b) % base

-- Define the carry operation in base 8
def carry_base8 (a b : ℕ) : ℕ := (a + b) / base

-- The theorem to prove
theorem base8_addition_problem (square : ℕ) :
  square < base →
  add_base8 (add_base8 square square) 4 = 6 →
  add_base8 (add_base8 3 5) square = square →
  add_base8 (add_base8 4 square) (carry_base8 3 5) = 3 →
  square = 1 := by
sorry

end NUMINAMATH_CALUDE_base8_addition_problem_l1119_111975


namespace NUMINAMATH_CALUDE_complex_sum_zero_l1119_111919

theorem complex_sum_zero : 
  let z : ℂ := -1/2 + (Real.sqrt 3 / 2) * Complex.I
  1 + z + z^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_zero_l1119_111919


namespace NUMINAMATH_CALUDE_doris_hourly_rate_l1119_111924

/-- Doris's hourly rate for babysitting -/
def hourly_rate : ℝ := 20

/-- Minimum amount Doris needs to earn in 3 weeks -/
def minimum_earnings : ℝ := 1200

/-- Number of hours Doris babysits on weekdays -/
def weekday_hours : ℝ := 3

/-- Number of hours Doris babysits on Saturdays -/
def saturday_hours : ℝ := 5

/-- Number of weekdays in a week -/
def weekdays_per_week : ℝ := 5

/-- Number of Saturdays in a week -/
def saturdays_per_week : ℝ := 1

/-- Number of weeks Doris needs to work to earn minimum_earnings -/
def weeks_to_earn : ℝ := 3

theorem doris_hourly_rate :
  hourly_rate = minimum_earnings / (weeks_to_earn * (weekdays_per_week * weekday_hours + saturdays_per_week * saturday_hours)) := by
  sorry

end NUMINAMATH_CALUDE_doris_hourly_rate_l1119_111924


namespace NUMINAMATH_CALUDE_course_selection_proof_l1119_111952

def total_courses : ℕ := 9
def courses_to_choose : ℕ := 4
def conflicting_courses : ℕ := 3
def other_courses : ℕ := total_courses - conflicting_courses

def selection_schemes : ℕ := 
  (conflicting_courses.choose 1 * other_courses.choose (courses_to_choose - 1)) +
  (other_courses.choose courses_to_choose)

theorem course_selection_proof : selection_schemes = 75 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_proof_l1119_111952


namespace NUMINAMATH_CALUDE_triangle_circumradius_l1119_111981

/-- Given a triangle with sides a and b, area S, and the median to the third side
    less than half of that side, prove that the radius of the circumcircle is 8 / √15 -/
theorem triangle_circumradius (a b : ℝ) (S : ℝ) (h_a : a = 2) (h_b : b = 3)
  (h_S : S = (3 * Real.sqrt 15) / 4)
  (h_median : ∃ (m : ℝ), m < (a + b) / 4 ∧ m^2 = (2 * (a^2 + b^2) - ((a + b) / 2)^2) / 4) :
  ∃ (R : ℝ), R = 8 / Real.sqrt 15 ∧ R * 2 * S = a * b * (Real.sqrt ((a + b + (a + b)) * (-a + b + (a + b)) * (a - b + (a + b)) * (a + b - (a + b))) / (4 * (a + b))) := by
  sorry

end NUMINAMATH_CALUDE_triangle_circumradius_l1119_111981


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_l1119_111976

theorem sqrt_sum_equals_sqrt (n : ℕ+) :
  (∃ x y : ℕ+, Real.sqrt x + Real.sqrt y = Real.sqrt n) ↔
  (∃ p q : ℕ, p > 1 ∧ n = p^2 * q) :=
sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_l1119_111976


namespace NUMINAMATH_CALUDE_distribute_8_3_non_empty_different_l1119_111931

/-- The number of ways to distribute n different balls into k different boxes --/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute n different balls into k different boxes,
    where each box contains at least one ball --/
def distributeNonEmpty (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute n different balls into k different boxes,
    where each box contains at least one ball and the number of balls in each box is different --/
def distributeNonEmptyDifferent (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: The number of ways to distribute 8 different balls into 3 different boxes,
    where each box contains at least one ball and the number of balls in each box is different,
    is equal to 2688 --/
theorem distribute_8_3_non_empty_different :
  distributeNonEmptyDifferent 8 3 = 2688 := by sorry

end NUMINAMATH_CALUDE_distribute_8_3_non_empty_different_l1119_111931


namespace NUMINAMATH_CALUDE_salary_decrease_increase_l1119_111923

theorem salary_decrease_increase (original_salary : ℝ) (h : original_salary > 0) :
  let decreased_salary := original_salary * 0.5
  let final_salary := decreased_salary * 1.5
  final_salary = original_salary * 0.75 ∧ 
  (original_salary - final_salary) / original_salary = 0.25 :=
by sorry

end NUMINAMATH_CALUDE_salary_decrease_increase_l1119_111923


namespace NUMINAMATH_CALUDE_shifted_parabola_vertex_l1119_111980

/-- Given a parabola y = -2x^2 + 1 shifted 1 unit left and 3 units up, its vertex is at (-1, 4) -/
theorem shifted_parabola_vertex (x y : ℝ) :
  let f : ℝ → ℝ := λ x ↦ -2 * x^2 + 1
  let g : ℝ → ℝ := λ x ↦ f (x + 1) + 3
  g x = y ∧ ∀ t, g t ≤ y → (x = -1 ∧ y = 4) :=
by sorry

end NUMINAMATH_CALUDE_shifted_parabola_vertex_l1119_111980


namespace NUMINAMATH_CALUDE_price_changes_l1119_111918

/-- The original price of an item that, after a 5% decrease and a 40% increase,
    results in a price $1352.06 less than twice the original price. -/
def original_price : ℝ := 2018

theorem price_changes (x : ℝ) (hx : x = original_price) :
  let price_after_decrease := 0.95 * x
  let price_after_increase := price_after_decrease * 1.4
  price_after_increase = 2 * x - 1352.06 := by sorry

end NUMINAMATH_CALUDE_price_changes_l1119_111918


namespace NUMINAMATH_CALUDE_ratio_equality_product_l1119_111968

theorem ratio_equality_product (x : ℝ) : 
  (x + 3) / (2 * x + 3) = (4 * x + 4) / (7 * x + 4) → 
  ∃ y : ℝ, (x = 0 ∨ x = 5) ∧ x * y = 0 := by sorry

end NUMINAMATH_CALUDE_ratio_equality_product_l1119_111968


namespace NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l1119_111921

def is_arithmetic_sequence (seq : Fin 4 → ℝ) : Prop :=
  ∃ (a d : ℝ), seq 0 = a - d ∧ seq 1 = a ∧ seq 2 = a + d ∧ seq 3 = a + 2*d

def sum_is_26 (seq : Fin 4 → ℝ) : Prop :=
  (seq 0) + (seq 1) + (seq 2) + (seq 3) = 26

def middle_product_is_40 (seq : Fin 4 → ℝ) : Prop :=
  (seq 1) * (seq 2) = 40

theorem arithmetic_sequence_theorem (seq : Fin 4 → ℝ) :
  is_arithmetic_sequence seq ∧ sum_is_26 seq ∧ middle_product_is_40 seq →
  (seq 0 = 2 ∧ seq 1 = 5 ∧ seq 2 = 8 ∧ seq 3 = 11) ∨
  (seq 0 = 11 ∧ seq 1 = 8 ∧ seq 2 = 5 ∧ seq 3 = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l1119_111921


namespace NUMINAMATH_CALUDE_printer_fraction_of_total_l1119_111948

/-- The price of the printer as a fraction of the total price with an enhanced computer -/
theorem printer_fraction_of_total (basic_computer_price printer_price enhanced_computer_price total_price_basic total_price_enhanced : ℚ) : 
  total_price_basic = basic_computer_price + printer_price →
  enhanced_computer_price = basic_computer_price + 500 →
  basic_computer_price = 2000 →
  total_price_enhanced = enhanced_computer_price + printer_price →
  printer_price / total_price_enhanced = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_printer_fraction_of_total_l1119_111948


namespace NUMINAMATH_CALUDE_solve_rock_problem_l1119_111994

def rock_problem (joshua_rocks : ℕ) : Prop :=
  let jose_rocks := joshua_rocks - 14
  let albert_rocks := jose_rocks + 28
  let clara_rocks := jose_rocks / 2
  let maria_rocks := clara_rocks + 18
  albert_rocks - joshua_rocks = 14

theorem solve_rock_problem :
  rock_problem 80 := by sorry

end NUMINAMATH_CALUDE_solve_rock_problem_l1119_111994


namespace NUMINAMATH_CALUDE_arithmetic_sequence_remainder_l1119_111969

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The number of terms in the sequence -/
def sequence_length (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

theorem arithmetic_sequence_remainder (a₁ d aₙ : ℕ) (h₁ : a₁ = 2) (h₂ : d = 6) (h₃ : aₙ = 278) :
  (arithmetic_sum a₁ d (sequence_length a₁ d aₙ)) % 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_remainder_l1119_111969


namespace NUMINAMATH_CALUDE_alices_total_distance_l1119_111964

/-- Alice's weekly walking distance to school and back home --/
def alices_weekly_walking_distance (days_per_week : ℕ) (distance_to_school : ℕ) (distance_from_school : ℕ) : ℕ :=
  (days_per_week * distance_to_school) + (days_per_week * distance_from_school)

/-- Theorem: Alice walks 110 miles in a week --/
theorem alices_total_distance :
  alices_weekly_walking_distance 5 10 12 = 110 := by
  sorry

end NUMINAMATH_CALUDE_alices_total_distance_l1119_111964


namespace NUMINAMATH_CALUDE_equation_identity_l1119_111947

theorem equation_identity (a b c x : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  c * ((x - a) * (x - b)) / ((c - a) * (c - b)) +
  b * ((x - a) * (x - c)) / ((b - a) * (b - c)) +
  a * ((x - b) * (x - c)) / ((a - b) * (a - c)) = x := by
  sorry

end NUMINAMATH_CALUDE_equation_identity_l1119_111947


namespace NUMINAMATH_CALUDE_reasonable_prize_distribution_l1119_111925

/-- The most reasonable prize distribution for a math competition problem --/
theorem reasonable_prize_distribution
  (total_prize : ℝ)
  (prob_A : ℝ)
  (prob_B : ℝ)
  (h_total : total_prize = 190)
  (h_prob_A : prob_A = 3/4)
  (h_prob_B : prob_B = 4/5)
  (h_prob_valid : 0 ≤ prob_A ∧ prob_A ≤ 1 ∧ 0 ≤ prob_B ∧ prob_B ≤ 1) :
  let expected_A := (prob_A * (1 - prob_B) * total_prize + prob_A * prob_B * (total_prize / 2))
  let expected_B := (prob_B * (1 - prob_A) * total_prize + prob_A * prob_B * (total_prize / 2))
  expected_A = 90 ∧ expected_B = 100 :=
by sorry


end NUMINAMATH_CALUDE_reasonable_prize_distribution_l1119_111925


namespace NUMINAMATH_CALUDE_odd_function_log_value_l1119_111982

theorem odd_function_log_value (f : ℝ → ℝ) :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x > 0, f x = Real.log x / Real.log 2) →  -- f(x) = log₂(x) for x > 0
  f (-2) = -1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_log_value_l1119_111982


namespace NUMINAMATH_CALUDE_expression_evaluation_l1119_111938

/-- Given a = 4 and b = -3, prove that 2a^2 - 3b^2 + 4ab = -43 -/
theorem expression_evaluation (a b : ℤ) (ha : a = 4) (hb : b = -3) :
  2 * a^2 - 3 * b^2 + 4 * a * b = -43 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1119_111938


namespace NUMINAMATH_CALUDE_sin_210_deg_l1119_111910

/-- The sine of 210 degrees is equal to -1/2 --/
theorem sin_210_deg : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_deg_l1119_111910


namespace NUMINAMATH_CALUDE_polar_coords_of_negative_two_plus_two_i_l1119_111915

/-- The polar coordinates of a complex number z = -(2+2i) -/
theorem polar_coords_of_negative_two_plus_two_i :
  ∃ (r : ℝ) (θ : ℝ) (k : ℤ),
    r = 2 * Real.sqrt 2 ∧
    θ = 5 * Real.pi / 4 + 2 * k * Real.pi ∧
    Complex.exp (θ * Complex.I) * r = -(2 + 2 * Complex.I) :=
by sorry

end NUMINAMATH_CALUDE_polar_coords_of_negative_two_plus_two_i_l1119_111915


namespace NUMINAMATH_CALUDE_clockwise_rotation_240_l1119_111935

/-- The angle formed by rotating a ray clockwise around its endpoint -/
def clockwise_rotation (angle : ℝ) : ℝ := -angle

/-- Theorem: The angle formed by rotating a ray 240° clockwise around its endpoint is -240° -/
theorem clockwise_rotation_240 : clockwise_rotation 240 = -240 := by
  sorry

end NUMINAMATH_CALUDE_clockwise_rotation_240_l1119_111935


namespace NUMINAMATH_CALUDE_ratio_odd_even_divisors_l1119_111993

def M : ℕ := 18 * 18 * 56 * 165

def sum_odd_divisors (n : ℕ) : ℕ := sorry

def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors :
  (sum_odd_divisors M) * 62 = sum_even_divisors M := by sorry

end NUMINAMATH_CALUDE_ratio_odd_even_divisors_l1119_111993


namespace NUMINAMATH_CALUDE_fahrenheit_95_equals_celsius_35_l1119_111912

-- Define the conversion function from Fahrenheit to Celsius
def fahrenheit_to_celsius (f : ℚ) : ℚ := (f - 32) * (5/9)

-- Theorem statement
theorem fahrenheit_95_equals_celsius_35 : fahrenheit_to_celsius 95 = 35 := by
  sorry

end NUMINAMATH_CALUDE_fahrenheit_95_equals_celsius_35_l1119_111912


namespace NUMINAMATH_CALUDE_wilson_sledding_l1119_111983

/-- The number of times Wilson sleds down a tall hill -/
def tall_hill_slides : ℕ := 4

/-- The number of small hills -/
def small_hills : ℕ := 3

/-- The total number of times Wilson sled down all hills -/
def total_slides : ℕ := 14

/-- The number of tall hills Wilson sled down -/
def tall_hills : ℕ := 2

theorem wilson_sledding :
  tall_hills * tall_hill_slides + small_hills * (tall_hill_slides / 2) = total_slides :=
by sorry

end NUMINAMATH_CALUDE_wilson_sledding_l1119_111983


namespace NUMINAMATH_CALUDE_all_days_happy_l1119_111913

theorem all_days_happy (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_all_days_happy_l1119_111913


namespace NUMINAMATH_CALUDE_existence_of_xy_l1119_111942

theorem existence_of_xy (f g : ℝ → ℝ) : ∃ x y : ℝ, 
  x ∈ Set.Icc 0 1 ∧ 
  y ∈ Set.Icc 0 1 ∧ 
  |f x + g y - x * y| ≥ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_xy_l1119_111942


namespace NUMINAMATH_CALUDE_sum_and_cube_sum_divisibility_l1119_111911

theorem sum_and_cube_sum_divisibility (x y : ℤ) :
  (6 ∣ (x + y)) ↔ (6 ∣ (x^3 + y^3)) := by sorry

end NUMINAMATH_CALUDE_sum_and_cube_sum_divisibility_l1119_111911


namespace NUMINAMATH_CALUDE_problem_solution_l1119_111940

/-- f(n) denotes the nth positive integer which is not a perfect square -/
def f (n : ℕ) : ℕ := sorry

/-- Applies the function f n times -/
def iterateF (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | m + 1 => f (iterateF m x)

theorem problem_solution :
  ∃ (n : ℕ), n > 0 ∧ iterateF 2013 n = 2014^2 + 1 ∧ n = 6077248 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1119_111940


namespace NUMINAMATH_CALUDE_smallest_n_for_polygon_cuts_l1119_111933

theorem smallest_n_for_polygon_cuts : ∃ n : ℕ, 
  (∀ m : ℕ, m > 0 → (m - 2) % 31 = 0 ∧ (m - 2) % 65 = 0 → n ≤ m) ∧
  (n - 2) % 31 = 0 ∧ 
  (n - 2) % 65 = 0 ∧ 
  n = 2017 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_polygon_cuts_l1119_111933


namespace NUMINAMATH_CALUDE_square_from_equation_l1119_111943

theorem square_from_equation (x y z : ℕ) 
  (h : x^2 + y^2 + z^2 = 2*(x*y + y*z + z*x)) :
  ∃ (a b c : ℕ), x = a^2 ∧ y = b^2 ∧ z = c^2 := by
  sorry

end NUMINAMATH_CALUDE_square_from_equation_l1119_111943


namespace NUMINAMATH_CALUDE_inequality_not_hold_l1119_111989

theorem inequality_not_hold (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  ¬(1 / (a - b) > 1 / a) := by
sorry

end NUMINAMATH_CALUDE_inequality_not_hold_l1119_111989


namespace NUMINAMATH_CALUDE_linden_birch_problem_l1119_111963

theorem linden_birch_problem :
  ∃ (x y : ℕ), 
    x + y > 14 ∧ 
    y + 18 > 2 * x ∧ 
    x > 2 * y ∧ 
    x = 11 ∧ 
    y = 5 := by
  sorry

end NUMINAMATH_CALUDE_linden_birch_problem_l1119_111963


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1119_111930

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 3*x₁ - 4 = 0) → (x₂^2 - 3*x₂ - 4 = 0) → (x₁ + x₂ = 3) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1119_111930


namespace NUMINAMATH_CALUDE_tan_105_degrees_l1119_111937

theorem tan_105_degrees :
  Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l1119_111937


namespace NUMINAMATH_CALUDE_seedling_problem_l1119_111904

-- Define variables for seedling prices
variable (x y : ℚ)

-- Define conditions
def condition1 : Prop := 3 * x + 2 * y = 12
def condition2 : Prop := x + 3 * y = 11

-- Define total number of seedlings
def total_seedlings : ℕ := 200

-- Define value multiplier
def value_multiplier : ℕ := 100

-- Define minimum total value
def min_total_value : ℕ := 50000

-- Theorem to prove
theorem seedling_problem (h1 : condition1 x y) (h2 : condition2 x y) :
  x = 2 ∧ y = 3 ∧
  ∃ m : ℕ, m ≥ 100 ∧
  m ≤ total_seedlings ∧
  2 * value_multiplier * (total_seedlings - m) + 3 * value_multiplier * m ≥ min_total_value ∧
  ∀ n : ℕ, n < m →
    2 * value_multiplier * (total_seedlings - n) + 3 * value_multiplier * n < min_total_value :=
by
  sorry

end NUMINAMATH_CALUDE_seedling_problem_l1119_111904


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_11_l1119_111996

/-- An arithmetic sequence is a sequence where the difference between
    each consecutive term is constant. -/
structure ArithmeticSequence (α : Type*) [AddCommGroup α] where
  a : ℕ → α
  d : α
  h : ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_11 (a : ArithmeticSequence ℝ) 
  (h : a.a 4 + a.a 8 = 16) : 
  (Finset.range 11).sum a.a = 88 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_11_l1119_111996


namespace NUMINAMATH_CALUDE_train_length_l1119_111928

/-- The length of a train given its passing times over different distances -/
theorem train_length (tree_time platform_time platform_length : ℝ) 
  (h1 : tree_time = 120)
  (h2 : platform_time = 170)
  (h3 : platform_length = 500)
  (h4 : tree_time > 0)
  (h5 : platform_time > 0)
  (h6 : platform_length > 0) :
  let train_length := (platform_time * platform_length) / (platform_time - tree_time)
  train_length = 1200 := by
sorry


end NUMINAMATH_CALUDE_train_length_l1119_111928


namespace NUMINAMATH_CALUDE_zongzi_sales_l1119_111909

/-- The cost and profit calculation for zongzi sales during the Dragon Boat Festival --/
theorem zongzi_sales (x : ℝ) (m : ℝ) : 
  /- Cost price after festival -/
  (∀ y, y > 0 → 240 / y - 4 = 240 / (y + 2) → y = x) →
  /- Total cost constraint -/
  ((12 : ℝ) * m + 10 * (400 - m) ≤ 4600) →
  /- Profit calculation -/
  (∀ w, w = 2 * m + 2400) →
  /- Conclusions -/
  (x = 10 ∧ m = 300 ∧ (2 * 300 + 2400 = 3000)) := by
  sorry


end NUMINAMATH_CALUDE_zongzi_sales_l1119_111909


namespace NUMINAMATH_CALUDE_cos_squared_pi_sixth_plus_half_alpha_l1119_111966

theorem cos_squared_pi_sixth_plus_half_alpha (α : ℝ) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  Real.cos (π / 6 + α / 2) ^ 2 = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_cos_squared_pi_sixth_plus_half_alpha_l1119_111966


namespace NUMINAMATH_CALUDE_forty_students_not_enrolled_l1119_111926

/-- The number of students not enrolled in any language course -/
def students_not_enrolled (total students_french students_german students_spanish
  students_french_german students_french_spanish students_german_spanish
  students_all_three : ℕ) : ℕ :=
  total - (students_french + students_german + students_spanish
           - students_french_german - students_french_spanish - students_german_spanish
           + students_all_three)

/-- Theorem stating that 40 students are not enrolled in any language course -/
theorem forty_students_not_enrolled :
  students_not_enrolled 150 60 50 40 20 15 10 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_forty_students_not_enrolled_l1119_111926


namespace NUMINAMATH_CALUDE_factor_of_polynomial_l1119_111932

theorem factor_of_polynomial (c d : ℤ) : 
  (∀ x : ℝ, x^2 - x - 1 = 0 → c * x^19 + d * x^18 + 1 = 0) ↔ 
  (c = 1597 ∧ d = -2584) :=
by sorry

end NUMINAMATH_CALUDE_factor_of_polynomial_l1119_111932


namespace NUMINAMATH_CALUDE_f_monotonicity_f_shifted_even_f_positive_domain_l1119_111959

-- Define the function f(x) = lg|x-1|
noncomputable def f (x : ℝ) : ℝ := Real.log (abs (x - 1)) / Real.log 10

-- Statement 1: f(x) is monotonically decreasing on (-∞, 1) and increasing on (1, +∞)
theorem f_monotonicity :
  (∀ x y, x < y ∧ y < 1 → f x > f y) ∧
  (∀ x y, 1 < x ∧ x < y → f x < f y) := by sorry

-- Statement 2: f(x+1) is an even function
theorem f_shifted_even :
  ∀ x, f (x + 1) = f (-x + 1) := by sorry

-- Statement 3: If f(a) > 0, then a < 0 or a > 2
theorem f_positive_domain :
  ∀ a, f a > 0 → a < 0 ∨ a > 2 := by sorry

end NUMINAMATH_CALUDE_f_monotonicity_f_shifted_even_f_positive_domain_l1119_111959


namespace NUMINAMATH_CALUDE_sum_of_first_seven_primes_with_units_digit_3_l1119_111944

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def has_units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def first_seven_primes_with_units_digit_3 : List ℕ := [3, 13, 23, 43, 53, 73, 83]

theorem sum_of_first_seven_primes_with_units_digit_3 :
  (∀ n ∈ first_seven_primes_with_units_digit_3, is_prime n ∧ has_units_digit_3 n) →
  (∀ p : ℕ, is_prime p → has_units_digit_3 p → 
    p ∉ first_seven_primes_with_units_digit_3 → 
    p > (List.maximum first_seven_primes_with_units_digit_3).getD 0) →
  List.sum first_seven_primes_with_units_digit_3 = 291 := by
sorry

end NUMINAMATH_CALUDE_sum_of_first_seven_primes_with_units_digit_3_l1119_111944


namespace NUMINAMATH_CALUDE_shirt_probabilities_l1119_111992

structure ShirtPack :=
  (m_shirts : ℕ)
  (count : ℕ)

def total_packs : ℕ := 50

def shirt_distribution : List ShirtPack := [
  ⟨0, 7⟩, ⟨1, 3⟩, ⟨4, 10⟩, ⟨5, 15⟩, ⟨7, 5⟩, ⟨9, 4⟩, ⟨10, 3⟩, ⟨11, 3⟩
]

def count_packs (pred : ShirtPack → Bool) : ℕ :=
  (shirt_distribution.filter pred).foldl (λ acc pack => acc + pack.count) 0

theorem shirt_probabilities :
  (count_packs (λ pack => pack.m_shirts = 0) : ℚ) / total_packs = 7 / 50 ∧
  (count_packs (λ pack => pack.m_shirts < 7) : ℚ) / total_packs = 7 / 10 ∧
  (count_packs (λ pack => pack.m_shirts > 9) : ℚ) / total_packs = 3 / 25 := by
  sorry

end NUMINAMATH_CALUDE_shirt_probabilities_l1119_111992


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1119_111922

/-- A function f is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem solution_set_of_inequality (f : ℝ → ℝ) (hf_odd : OddFunction f)
    (hf_2 : f 2 = 0) (hf_deriv : ∀ x > 0, (x * (deriv f x) - f x) / x^2 < 0) :
    {x : ℝ | x^2 * f x > 0} = {x : ℝ | x < -2 ∨ (0 < x ∧ x < 2)} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1119_111922


namespace NUMINAMATH_CALUDE_complement_of_A_l1119_111951

def A : Set ℝ := {x | x ≥ 3} ∪ {x | x < -1}

theorem complement_of_A : 
  (Set.univ : Set ℝ) \ A = {x : ℝ | -1 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l1119_111951


namespace NUMINAMATH_CALUDE_divisors_of_cube_l1119_111997

/-- 
Given a natural number n with exactly two prime divisors,
if n^2 has 81 divisors, then n^3 has either 160 or 169 divisors.
-/
theorem divisors_of_cube (n : ℕ) : 
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ ∃ α β : ℕ, n = p^α * q^β) →
  (Finset.card (Nat.divisors (n^2)) = 81) →
  (Finset.card (Nat.divisors (n^3)) = 160 ∨ Finset.card (Nat.divisors (n^3)) = 169) :=
by sorry

end NUMINAMATH_CALUDE_divisors_of_cube_l1119_111997


namespace NUMINAMATH_CALUDE_sin_600_degrees_l1119_111960

theorem sin_600_degrees : Real.sin (600 * π / 180) = - (Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_600_degrees_l1119_111960


namespace NUMINAMATH_CALUDE_product_sum_equals_power_l1119_111939

theorem product_sum_equals_power : 
  2 * (3 + 1) * (3^2 + 1) * (3^4 + 1) * (3^8 + 1) * (3^16 + 1) * (3^32 + 1) * (3^64 + 1) + 1 = 3^128 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_equals_power_l1119_111939


namespace NUMINAMATH_CALUDE_a_eq_b_sufficient_not_necessary_for_a_sq_eq_b_sq_l1119_111962

theorem a_eq_b_sufficient_not_necessary_for_a_sq_eq_b_sq :
  (∀ a b : ℝ, a = b → a^2 = b^2) ∧
  (∃ a b : ℝ, a^2 = b^2 ∧ a ≠ b) := by
  sorry

end NUMINAMATH_CALUDE_a_eq_b_sufficient_not_necessary_for_a_sq_eq_b_sq_l1119_111962


namespace NUMINAMATH_CALUDE_right_triangle_square_areas_l1119_111917

theorem right_triangle_square_areas : ∀ (A B C : ℝ),
  (A = 6^2) →
  (B = 8^2) →
  (C = 10^2) →
  A + B = C :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_square_areas_l1119_111917


namespace NUMINAMATH_CALUDE_range_of_M_M_lower_bound_l1119_111934

theorem range_of_M (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a + b + c = 1) :
  let M := (1/a - 1) * (1/b - 1) * (1/c - 1)
  ∀ x : ℝ, x ≥ 8 → ∃ a' b' c' : ℝ, 
    a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 
    a' + b' + c' = 1 ∧
    (1/a' - 1) * (1/b' - 1) * (1/c' - 1) = x :=
by sorry

theorem M_lower_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a + b + c = 1) :
  (1/a - 1) * (1/b - 1) * (1/c - 1) ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_range_of_M_M_lower_bound_l1119_111934


namespace NUMINAMATH_CALUDE_parabola_reflection_difference_l1119_111927

/-- Represents a quadratic function of the form ax^2 + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The function representing the original parabola translated up by 3 units --/
def f (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c + 3

/-- The function representing the reflected parabola translated down by 3 units --/
def g (p : Parabola) (x : ℝ) : ℝ :=
  -p.a * x^2 - p.b * x - p.c - 3

/-- Theorem stating that (f-g)(x) equals 2ax^2 + 2bx + 2c + 6 --/
theorem parabola_reflection_difference (p : Parabola) (x : ℝ) :
  f p x - g p x = 2 * p.a * x^2 + 2 * p.b * x + 2 * p.c + 6 := by
  sorry


end NUMINAMATH_CALUDE_parabola_reflection_difference_l1119_111927


namespace NUMINAMATH_CALUDE_intersection_S_T_l1119_111957

def S : Set ℝ := {x | x + 1 ≥ 2}
def T : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_S_T : S ∩ T = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_S_T_l1119_111957


namespace NUMINAMATH_CALUDE_function_inequality_l1119_111998

/-- Given a function f(x) = ln x - 3x defined on (0, +∞), and for all x ∈ (0, +∞),
    f(x) ≤ x(ae^x - 4) + b, prove that a + b ≥ 0. -/
theorem function_inequality (a b : ℝ) : 
  (∀ x : ℝ, x > 0 → Real.log x - 3 * x ≤ x * (a * Real.exp x - 4) + b) → 
  a + b ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1119_111998


namespace NUMINAMATH_CALUDE_total_chocolate_bars_l1119_111901

/-- The number of chocolate bars in a massive crate -/
def chocolateBarsInCrate (largeBozesPerCrate mediumBoxesPerLarge smallBoxesPerMedium barsPerSmall : ℕ) : ℕ :=
  largeBozesPerCrate * mediumBoxesPerLarge * smallBoxesPerMedium * barsPerSmall

/-- Theorem: The massive crate contains 153,900 chocolate bars -/
theorem total_chocolate_bars :
  chocolateBarsInCrate 10 19 27 30 = 153900 := by
  sorry

#eval chocolateBarsInCrate 10 19 27 30

end NUMINAMATH_CALUDE_total_chocolate_bars_l1119_111901


namespace NUMINAMATH_CALUDE_daniel_animals_legs_l1119_111973

/-- The number of legs an animal has --/
def legs (animal : String) : ℕ :=
  match animal with
  | "horse" => 4
  | "dog" => 4
  | "cat" => 4
  | "turtle" => 4
  | "goat" => 4
  | "snake" => 0
  | "spider" => 8
  | "bird" => 2
  | _ => 0

/-- The number of each type of animal Daniel has --/
def animals : List (String × ℕ) := [
  ("horse", 2),
  ("dog", 5),
  ("cat", 7),
  ("turtle", 3),
  ("goat", 1),
  ("snake", 4),
  ("spider", 2),
  ("bird", 3)
]

/-- The total number of legs of all animals --/
def totalLegs : ℕ := (animals.map (fun (a, n) => n * legs a)).sum

theorem daniel_animals_legs :
  totalLegs = 94 := by sorry

end NUMINAMATH_CALUDE_daniel_animals_legs_l1119_111973


namespace NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l1119_111965

def n : ℕ := 4016

def k : ℕ := n^2 + 2^n

theorem units_digit_of_k_squared_plus_two_to_k (n : ℕ) (k : ℕ) :
  n = 4016 →
  k = n^2 + 2^n →
  (k^2 + 2^k) % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l1119_111965


namespace NUMINAMATH_CALUDE_tangent_condition_l1119_111991

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := x + y + 1 = 0

/-- The circle equation -/
def circle_equation (x y a b : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 2

/-- The line is tangent to the circle -/
def is_tangent (a b : ℝ) : Prop := ∃ x y : ℝ, line_equation x y ∧ circle_equation x y a b ∧
  ∀ x' y' : ℝ, line_equation x' y' → circle_equation x' y' a b → (x', y') = (x, y)

theorem tangent_condition (a b : ℝ) :
  (a + b = 1 → is_tangent a b) ∧
  (∃ a' b' : ℝ, is_tangent a' b' ∧ a' + b' ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_tangent_condition_l1119_111991


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1119_111961

theorem min_value_of_expression (x : ℝ) :
  (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2027 ≥ 2026 ∧
  ∃ y : ℝ, (y + 1) * (y + 2) * (y + 3) * (y + 4) + 2027 = 2026 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1119_111961


namespace NUMINAMATH_CALUDE_middle_number_of_three_consecutive_squares_l1119_111906

theorem middle_number_of_three_consecutive_squares (n : ℕ) : 
  n^2 + (n+1)^2 + (n+2)^2 = 2030 → n + 1 = 26 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_of_three_consecutive_squares_l1119_111906


namespace NUMINAMATH_CALUDE_jake_sausage_spending_l1119_111956

/-- Represents a type of sausage package -/
structure SausagePackage where
  weight : Real
  price_per_pound : Real

/-- Calculates the total cost for a given number of packages of a specific type -/
def total_cost_for_type (package : SausagePackage) (num_packages : Nat) : Real :=
  package.weight * package.price_per_pound * num_packages

/-- Theorem: Jake spends $52 on sausages -/
theorem jake_sausage_spending :
  let type1 : SausagePackage := { weight := 2, price_per_pound := 4 }
  let type2 : SausagePackage := { weight := 1.5, price_per_pound := 5 }
  let type3 : SausagePackage := { weight := 3, price_per_pound := 3.5 }
  let num_packages : Nat := 2
  total_cost_for_type type1 num_packages +
  total_cost_for_type type2 num_packages +
  total_cost_for_type type3 num_packages = 52 := by
  sorry

end NUMINAMATH_CALUDE_jake_sausage_spending_l1119_111956


namespace NUMINAMATH_CALUDE_problem_statement_l1119_111941

theorem problem_statement (x y : ℝ) 
  (h1 : x + y = 3) 
  (h2 : x^2 + y^2 - x*y = 4) : 
  x^4 + y^4 + x^3*y + x*y^3 = 36 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1119_111941


namespace NUMINAMATH_CALUDE_triangular_weight_is_60_l1119_111988

/-- Given a set of weights with specific balancing conditions, prove that the triangular weight is 60 grams. -/
theorem triangular_weight_is_60 :
  ∀ (round_weight triangular_weight : ℝ),
  (round_weight + triangular_weight = 3 * round_weight) →
  (4 * round_weight + triangular_weight = triangular_weight + round_weight + 90) →
  triangular_weight = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangular_weight_is_60_l1119_111988


namespace NUMINAMATH_CALUDE_power_of_two_in_product_l1119_111986

theorem power_of_two_in_product (w : ℕ+) : 
  (∃ k : ℕ, 936 * w = k * 3^3 * 11^2) →  -- The product has 3^3 and 11^2 as factors
  (∀ x : ℕ+, x < 132 → ¬∃ k : ℕ, 936 * x = k * 3^3 * 11^2) →  -- 132 is the smallest possible w
  (∃ m : ℕ, 936 * w = 2^5 * m ∧ m % 2 ≠ 0) :=  -- The highest power of 2 dividing the product is 2^5
by sorry

end NUMINAMATH_CALUDE_power_of_two_in_product_l1119_111986


namespace NUMINAMATH_CALUDE_triangle_inequality_l1119_111916

/-- Proves that for any triangle with side lengths a, b, c, and area S,
    the inequality (ab + ac + bc) / (4S) ≥ √3 holds true. -/
theorem triangle_inequality (a b c S : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ S > 0) 
    (h_triangle : S = 1/4 * Real.sqrt ((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c))) :
  (a * b + a * c + b * c) / (4 * S) ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1119_111916


namespace NUMINAMATH_CALUDE_max_cos_product_l1119_111936

theorem max_cos_product (α β γ : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2) (h3 : 0 < γ ∧ γ < π/2) 
  (h4 : Real.sin α ^ 2 + Real.sin β ^ 2 + Real.sin γ ^ 2 = 1) :
  Real.cos α * Real.cos β * Real.cos γ ≤ 2 * Real.sqrt 6 / 9 := by
  sorry

end NUMINAMATH_CALUDE_max_cos_product_l1119_111936


namespace NUMINAMATH_CALUDE_factorization_proof_l1119_111950

theorem factorization_proof (x : ℝ) : 221 * x^2 + 68 * x + 17 = 17 * (13 * x^2 + 4 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1119_111950


namespace NUMINAMATH_CALUDE_tan_alpha_plus_20_l1119_111995

theorem tan_alpha_plus_20 (α : ℝ) (h : Real.tan (α + 80 * π / 180) = 4 * Real.sin (420 * π / 180)) :
  Real.tan (α + 20 * π / 180) = Real.sqrt 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_20_l1119_111995


namespace NUMINAMATH_CALUDE_cannot_complete_in_five_trips_l1119_111990

def truck_capacity : ℕ := 2000
def rice_sacks : ℕ := 150
def corn_sacks : ℕ := 100
def rice_weight : ℕ := 60
def corn_weight : ℕ := 25
def num_trips : ℕ := 5

theorem cannot_complete_in_five_trips :
  rice_sacks * rice_weight + corn_sacks * corn_weight > num_trips * truck_capacity :=
by sorry

end NUMINAMATH_CALUDE_cannot_complete_in_five_trips_l1119_111990


namespace NUMINAMATH_CALUDE_bird_nest_twigs_l1119_111907

theorem bird_nest_twigs (twigs_in_circle : ℕ) (additional_twigs_per_weave : ℕ) (twigs_still_needed : ℕ) :
  twigs_in_circle = 12 →
  additional_twigs_per_weave = 6 →
  twigs_still_needed = 48 →
  (twigs_in_circle * additional_twigs_per_weave - twigs_still_needed : ℚ) / (twigs_in_circle * additional_twigs_per_weave) = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_bird_nest_twigs_l1119_111907


namespace NUMINAMATH_CALUDE_f_x_plus_3_odd_l1119_111977

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_x_plus_3_odd (f : ℝ → ℝ) 
  (h1 : is_odd (fun x ↦ f (x + 1))) 
  (h2 : is_odd (fun x ↦ f (x - 1))) : 
  is_odd (fun x ↦ f (x + 3)) := by
  sorry

end NUMINAMATH_CALUDE_f_x_plus_3_odd_l1119_111977


namespace NUMINAMATH_CALUDE_car_fuel_efficiency_l1119_111979

theorem car_fuel_efficiency (x : ℝ) : x = 40 :=
  by
  have h1 : x > 0 := sorry
  have h2 : (4 / x + 4 / 20) = (8 / x) * 1.50000000000000014 := sorry
  sorry

end NUMINAMATH_CALUDE_car_fuel_efficiency_l1119_111979


namespace NUMINAMATH_CALUDE_walnut_trees_after_planting_l1119_111987

/-- The number of walnut trees in the park after planting -/
def total_trees (current_trees newly_planted_trees : ℕ) : ℕ :=
  current_trees + newly_planted_trees

/-- Theorem: The total number of walnut trees after planting is the sum of current trees and newly planted trees -/
theorem walnut_trees_after_planting 
  (current_trees : ℕ) 
  (newly_planted_trees : ℕ) :
  total_trees current_trees newly_planted_trees = current_trees + newly_planted_trees :=
by sorry

/-- Given information about the walnut trees in the park -/
def current_walnut_trees : ℕ := 22
def new_walnut_trees : ℕ := 55

/-- The total number of walnut trees after planting -/
def final_walnut_trees : ℕ := total_trees current_walnut_trees new_walnut_trees

#eval final_walnut_trees

end NUMINAMATH_CALUDE_walnut_trees_after_planting_l1119_111987


namespace NUMINAMATH_CALUDE_negative_multiplication_result_l1119_111905

theorem negative_multiplication_result : (-4 : ℚ) * (-3/2 : ℚ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_negative_multiplication_result_l1119_111905


namespace NUMINAMATH_CALUDE_investment_rate_calculation_l1119_111946

theorem investment_rate_calculation (total_investment : ℝ) (invested_at_18_percent : ℝ) (total_interest : ℝ) :
  total_investment = 22000 →
  invested_at_18_percent = 7000 →
  total_interest = 3360 →
  let remaining_investment := total_investment - invested_at_18_percent
  let interest_from_18_percent := invested_at_18_percent * 0.18
  let remaining_interest := total_interest - interest_from_18_percent
  let unknown_rate := remaining_interest / remaining_investment
  unknown_rate = 0.14 := by sorry

end NUMINAMATH_CALUDE_investment_rate_calculation_l1119_111946
