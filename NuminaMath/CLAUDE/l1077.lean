import Mathlib

namespace NUMINAMATH_CALUDE_power_multiplication_l1077_107744

theorem power_multiplication (x : ℝ) : (x^5) * (x^2) = x^7 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1077_107744


namespace NUMINAMATH_CALUDE_inscribed_shape_perimeter_lower_bound_l1077_107718

/-- A shape inscribed in a circle -/
structure InscribedShape where
  -- The radius of the circumscribed circle
  radius : ℝ
  -- The perimeter of the shape
  perimeter : ℝ
  -- Predicate indicating if the center of the circle is inside or on the boundary of the shape
  center_inside : Prop

/-- Theorem: The perimeter of a shape inscribed in a circle is at least 4 times the radius
    if the center of the circle is inside or on the boundary of the shape -/
theorem inscribed_shape_perimeter_lower_bound
  (shape : InscribedShape)
  (h : shape.center_inside) :
  shape.perimeter ≥ 4 * shape.radius :=
sorry

end NUMINAMATH_CALUDE_inscribed_shape_perimeter_lower_bound_l1077_107718


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l1077_107760

/-- Given two perpendicular vectors a and b in ℝ², prove that m = 2 -/
theorem perpendicular_vectors_m_value (a b : ℝ × ℝ) (h : a.1 * b.1 + a.2 * b.2 = 0) :
  a = (-2, 3) → b.1 = 3 → b.2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l1077_107760


namespace NUMINAMATH_CALUDE_polynomial_roots_inequality_l1077_107793

theorem polynomial_roots_inequality (a b c : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    12 * x^3 + a * x^2 + b * x + c = 0 ∧ 
    12 * y^3 + a * y^2 + b * y + c = 0 ∧ 
    12 * z^3 + a * z^2 + b * z + c = 0) →
  (∀ x : ℝ, (x^2 + x + 2001)^3 + a * (x^2 + x + 2001)^2 + b * (x^2 + x + 2001) + c ≠ 0) →
  2001^3 + a * 2001^2 + b * 2001 + c > 1/64 := by
sorry

end NUMINAMATH_CALUDE_polynomial_roots_inequality_l1077_107793


namespace NUMINAMATH_CALUDE_angle_BCA_measure_l1077_107765

-- Define the points
variable (A B C D M O : EuclideanPlane)

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : EuclideanPlane) : Prop := sorry

-- Define M as the midpoint of AD
def is_midpoint (M A D : EuclideanPlane) : Prop := sorry

-- Define the intersection of BM and AC at O
def intersect_at (B M A C O : EuclideanPlane) : Prop := sorry

-- Define the angle measure function
def angle_measure (P Q R : EuclideanPlane) : ℝ := sorry

-- State the theorem
theorem angle_BCA_measure 
  (h_quad : is_convex_quadrilateral A B C D)
  (h_midpoint : is_midpoint M A D)
  (h_intersect : intersect_at B M A C O)
  (h_ABM : angle_measure A B M = 55)
  (h_AMB : angle_measure A M B = 70)
  (h_BOC : angle_measure B O C = 80)
  (h_ADC : angle_measure A D C = 60) :
  angle_measure B C A = 35 := by sorry

end NUMINAMATH_CALUDE_angle_BCA_measure_l1077_107765


namespace NUMINAMATH_CALUDE_floor_sqrt_50_squared_l1077_107704

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_squared_l1077_107704


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1077_107732

theorem sufficient_but_not_necessary :
  (∀ x : ℝ, x^2 > 2012 → x^2 > 2011) ∧
  (∃ x : ℝ, x^2 > 2011 ∧ x^2 ≤ 2012) →
  (∀ x : ℝ, x^2 > 2012 → x^2 > 2011) ∧
  ¬(∀ x : ℝ, x^2 > 2011 → x^2 > 2012) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1077_107732


namespace NUMINAMATH_CALUDE_coupon1_best_in_range_best_price_is_209_95_l1077_107713

def coupon1_discount (x : ℝ) : ℝ := 0.12 * x

def coupon2_discount : ℝ := 25

def coupon3_discount (x : ℝ) : ℝ := 0.15 * (x - 120)

theorem coupon1_best_in_range (x : ℝ) 
  (h1 : 208.33 < x) (h2 : x < 600) : 
  coupon1_discount x > coupon2_discount ∧ 
  coupon1_discount x > coupon3_discount x := by
  sorry

def listed_prices : List ℝ := [189.95, 209.95, 229.95, 249.95, 269.95]

theorem best_price_is_209_95 : 
  ∃ p ∈ listed_prices, p > 208.33 ∧ p < 600 ∧ 
  ∀ q ∈ listed_prices, q > 208.33 ∧ q < 600 → p ≤ q := by
  sorry

end NUMINAMATH_CALUDE_coupon1_best_in_range_best_price_is_209_95_l1077_107713


namespace NUMINAMATH_CALUDE_soccer_substitutions_modulo_l1077_107790

def num_players : ℕ := 22
def starting_players : ℕ := 11
def max_substitutions : ℕ := 4

def substitution_ways : ℕ → ℕ
| 0 => 1
| 1 => starting_players * starting_players
| n+1 => substitution_ways n * (starting_players - n) * (starting_players - n)

def total_substitution_ways : ℕ := 
  (List.range (max_substitutions + 1)).map substitution_ways |> List.sum

theorem soccer_substitutions_modulo :
  total_substitution_ways % 1000 = 722 := by sorry

end NUMINAMATH_CALUDE_soccer_substitutions_modulo_l1077_107790


namespace NUMINAMATH_CALUDE_edward_book_purchase_l1077_107762

theorem edward_book_purchase (total_spent : ℝ) (num_books : ℕ) (cost_per_book : ℝ) : 
  total_spent = 6 ∧ num_books = 2 ∧ total_spent = num_books * cost_per_book → cost_per_book = 3 := by
  sorry

end NUMINAMATH_CALUDE_edward_book_purchase_l1077_107762


namespace NUMINAMATH_CALUDE_common_tangents_M_N_l1077_107773

/-- Circle M defined by the equation x^2 + y^2 - 4y = 0 -/
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

/-- Circle N defined by the equation (x-1)^2 + (y-1)^2 = 1 -/
def circle_N (x y : ℝ) : Prop := (x-1)^2 + (y-1)^2 = 1

/-- The number of common tangent lines between two circles -/
def num_common_tangents (M N : (ℝ → ℝ → Prop)) : ℕ := sorry

/-- Theorem stating that the number of common tangent lines between circles M and N is 2 -/
theorem common_tangents_M_N : num_common_tangents circle_M circle_N = 2 := by sorry

end NUMINAMATH_CALUDE_common_tangents_M_N_l1077_107773


namespace NUMINAMATH_CALUDE_perpendicular_tangent_line_l1077_107770

/-- The equation of a line perpendicular to x + 4y - 4 = 0 and tangent to y = 2x² --/
theorem perpendicular_tangent_line : 
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, a*x + b*y + c = 0) ∧ 
    (∀ x y : ℝ, x + 4*y - 4 = 0 → (a*1 + b*4 = 0)) ∧
    (∃ x₀ : ℝ, a*x₀ + b*(2*x₀^2) + c = 0 ∧ 
              ∀ x : ℝ, a*x + b*(2*x^2) + c ≥ 0) ∧
    (a = 4 ∧ b = -1 ∧ c = -2) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_tangent_line_l1077_107770


namespace NUMINAMATH_CALUDE_jerome_toy_cars_l1077_107761

theorem jerome_toy_cars (original : ℕ) : original = 25 :=
  let last_month := 5
  let this_month := 2 * last_month
  let total := 40
  have h : original + last_month + this_month = total := by sorry
  sorry

end NUMINAMATH_CALUDE_jerome_toy_cars_l1077_107761


namespace NUMINAMATH_CALUDE_ellipse_dot_product_min_l1077_107703

/-- An ellipse with center at origin and left focus at (-1, 0) -/
structure Ellipse where
  x : ℝ
  y : ℝ
  eq : x^2 / 4 + y^2 / 3 = 1

/-- The dot product of OP and FP is always greater than or equal to 2 -/
theorem ellipse_dot_product_min (P : Ellipse) : 
  P.x * (P.x + 1) + P.y * P.y ≥ 2 := by
  sorry

#check ellipse_dot_product_min

end NUMINAMATH_CALUDE_ellipse_dot_product_min_l1077_107703


namespace NUMINAMATH_CALUDE_min_workers_for_job_l1077_107776

/-- Represents a construction job with workers -/
structure ConstructionJob where
  totalDays : ℕ
  elapsedDays : ℕ
  initialWorkers : ℕ
  completedPortion : ℚ
  
/-- Calculates the minimum number of workers needed to complete the job on time -/
def minWorkersNeeded (job : ConstructionJob) : ℕ :=
  job.initialWorkers

/-- Theorem stating that for the given job specifications, 
    the minimum number of workers needed is 10 -/
theorem min_workers_for_job :
  let job := ConstructionJob.mk 40 10 10 (1/4)
  minWorkersNeeded job = 10 := by
  sorry

end NUMINAMATH_CALUDE_min_workers_for_job_l1077_107776


namespace NUMINAMATH_CALUDE_sandy_earnings_l1077_107735

/-- Calculates the total earnings for Sandy given her hourly rate and hours worked each day -/
def total_earnings (hourly_rate : ℕ) (hours_friday : ℕ) (hours_saturday : ℕ) (hours_sunday : ℕ) : ℕ :=
  hourly_rate * (hours_friday + hours_saturday + hours_sunday)

/-- Theorem stating that Sandy's total earnings for the three days is $450 -/
theorem sandy_earnings : 
  total_earnings 15 10 6 14 = 450 := by
  sorry

end NUMINAMATH_CALUDE_sandy_earnings_l1077_107735


namespace NUMINAMATH_CALUDE_problem_solution_l1077_107795

theorem problem_solution : 
  let x : ℚ := 5
  let intermediate : ℚ := x * 12 / (180 / 3)
  intermediate + 80 = 81 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1077_107795


namespace NUMINAMATH_CALUDE_nancy_files_problem_l1077_107721

theorem nancy_files_problem (deleted_files : ℕ) (files_per_folder : ℕ) (final_folders : ℕ) :
  deleted_files = 31 →
  files_per_folder = 6 →
  final_folders = 2 →
  deleted_files + (files_per_folder * final_folders) = 43 :=
by sorry

end NUMINAMATH_CALUDE_nancy_files_problem_l1077_107721


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1077_107719

-- Problem 1
theorem problem_1 (x y : ℝ) :
  3 * x^2 * (-3 * x * y)^2 - x^2 * (x^2 * y^2 - 2 * x) = 26 * x^4 * y^2 + 2 * x^3 :=
by sorry

-- Problem 2
theorem problem_2 (a b c : ℝ) :
  -2 * (-a^2 * b * c)^2 * (1/2) * a * (b * c)^3 - (-a * b * c)^3 * (-a * b * c)^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1077_107719


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1077_107702

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n,
    prove that the common difference d is 2 when (S_2020 / 2020) - (S_20 / 20) = 2000 -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (S : ℕ → ℝ)  -- The sum function
  (h_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0)  -- Arithmetic sequence property
  (h_sum : ∀ n, S n = n * a 0 + n * (n - 1) / 2 * (a 1 - a 0))  -- Sum formula
  (h_condition : S 2020 / 2020 - S 20 / 20 = 2000)  -- Given condition
  : a 1 - a 0 = 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1077_107702


namespace NUMINAMATH_CALUDE_paint_needed_l1077_107797

theorem paint_needed (total_needed existing_paint new_paint : ℕ) 
  (h1 : total_needed = 70)
  (h2 : existing_paint = 36)
  (h3 : new_paint = 23) :
  total_needed - (existing_paint + new_paint) = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_paint_needed_l1077_107797


namespace NUMINAMATH_CALUDE_number_ratio_l1077_107716

theorem number_ratio (first second third : ℚ) : 
  first + second + third = 220 →
  second = 60 →
  third = (1 / 3) * first →
  first / second = 2 := by
sorry

end NUMINAMATH_CALUDE_number_ratio_l1077_107716


namespace NUMINAMATH_CALUDE_books_for_girls_l1077_107741

theorem books_for_girls (num_girls num_boys total_books : ℕ) : 
  num_girls = 15 → 
  num_boys = 10 → 
  total_books = 375 → 
  (num_girls * (total_books / (num_girls + num_boys))) = 225 := by
  sorry

end NUMINAMATH_CALUDE_books_for_girls_l1077_107741


namespace NUMINAMATH_CALUDE_expression_value_l1077_107720

theorem expression_value (a b c d m : ℝ) 
  (h1 : a = -b)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : |m| = 4)  -- The absolute value of m is 4
  : m + c * d + (a + b) / m = 5 ∨ m + c * d + (a + b) / m = -3 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1077_107720


namespace NUMINAMATH_CALUDE_difference_of_squares_401_399_l1077_107731

theorem difference_of_squares_401_399 : 401^2 - 399^2 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_401_399_l1077_107731


namespace NUMINAMATH_CALUDE_joan_bought_six_dozens_l1077_107739

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The total number of eggs Joan bought -/
def total_eggs : ℕ := 72

/-- The number of dozens of eggs Joan bought -/
def dozens_bought : ℕ := total_eggs / eggs_per_dozen

theorem joan_bought_six_dozens : dozens_bought = 6 := by
  sorry

end NUMINAMATH_CALUDE_joan_bought_six_dozens_l1077_107739


namespace NUMINAMATH_CALUDE_johns_money_l1077_107710

/-- Given that John needs a total amount of money and still needs some more,
    prove that the amount he already has is the difference between the total needed and the amount still needed. -/
theorem johns_money (total_needed : ℚ) (still_needed : ℚ) (already_has : ℚ) :
  total_needed = 2.5 →
  still_needed = 1.75 →
  already_has = total_needed - still_needed →
  already_has = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_johns_money_l1077_107710


namespace NUMINAMATH_CALUDE_negative_x_squared_times_x_cubed_l1077_107729

theorem negative_x_squared_times_x_cubed (x : ℝ) : (-x)^2 * x^3 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_negative_x_squared_times_x_cubed_l1077_107729


namespace NUMINAMATH_CALUDE_select_shoes_four_pairs_l1077_107766

/-- The number of ways to select 4 shoes from 4 pairs such that no two form a pair -/
def selectShoes (n : ℕ) : ℕ :=
  if n = 4 then 2^4 else 0

theorem select_shoes_four_pairs :
  selectShoes 4 = 16 :=
by sorry

end NUMINAMATH_CALUDE_select_shoes_four_pairs_l1077_107766


namespace NUMINAMATH_CALUDE_min_value_theorem_l1077_107771

/-- Two lines are perpendicular if the sum of products of their coefficients is zero -/
def perpendicular (a₁ b₁ a₂ b₂ : ℝ) : Prop := a₁ * a₂ + b₁ * b₂ = 0

/-- Definition of the first line l₁: (a-1)x + y - 1 = 0 -/
def line1 (a x y : ℝ) : Prop := (a - 1) * x + y - 1 = 0

/-- Definition of the second line l₂: x + 2by + 1 = 0 -/
def line2 (b x y : ℝ) : Prop := x + 2 * b * y + 1 = 0

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perp : perpendicular (a - 1) 1 1 (2 * b)) :
  (∀ a' b', a' > 0 → b' > 0 → perpendicular (a' - 1) 1 1 (2 * b') → 2 / a + 1 / b ≤ 2 / a' + 1 / b') ∧ 
  2 / a + 1 / b = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1077_107771


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1077_107789

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  1 / a + 4 / b ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ 1 / a₀ + 4 / b₀ = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1077_107789


namespace NUMINAMATH_CALUDE_finite_gcd_lcm_process_terminates_l1077_107748

theorem finite_gcd_lcm_process_terminates 
  (n : ℕ) 
  (a : Fin n → ℕ+) : 
  ∃ (k : ℕ), ∀ (j k : Fin n), j < k → (a j).val ∣ (a k).val :=
sorry

end NUMINAMATH_CALUDE_finite_gcd_lcm_process_terminates_l1077_107748


namespace NUMINAMATH_CALUDE_population_net_increase_l1077_107742

/-- Represents the birth rate in people per two seconds -/
def birth_rate : ℚ := 7

/-- Represents the death rate in people per two seconds -/
def death_rate : ℚ := 2

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- Theorem stating the net increase in population size in one day -/
theorem population_net_increase : 
  (birth_rate - death_rate) / 2 * seconds_per_day = 216000 := by sorry

end NUMINAMATH_CALUDE_population_net_increase_l1077_107742


namespace NUMINAMATH_CALUDE_bd_squared_equals_four_l1077_107749

theorem bd_squared_equals_four (a b c d : ℤ) 
  (eq1 : a - b - c + d = 13) 
  (eq2 : a + b - c - d = 9) : 
  (b - d)^2 = 4 := by sorry

end NUMINAMATH_CALUDE_bd_squared_equals_four_l1077_107749


namespace NUMINAMATH_CALUDE_distance_to_focus_is_13_l1077_107753

/-- Parabola with equation y^2 = 16x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  axis_of_symmetry : ℝ → ℝ

/-- Point on the parabola -/
structure PointOnParabola (p : Parabola) where
  point : ℝ × ℝ
  on_parabola : p.equation point.1 point.2

theorem distance_to_focus_is_13 (p : Parabola) (P : PointOnParabola p) 
  (h_equation : p.equation = fun x y => y^2 = 16*x)
  (h_distance : abs P.point.2 = 12) :
  dist P.point p.focus = 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_focus_is_13_l1077_107753


namespace NUMINAMATH_CALUDE_complex_number_problem_l1077_107709

theorem complex_number_problem (α β : ℂ) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ α + β = x ∧ Complex.I * (2 * α - β) = y) →
  β = 4 + 3 * Complex.I →
  α = 2 - 3 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l1077_107709


namespace NUMINAMATH_CALUDE_books_read_indeterminate_l1077_107787

/-- Represents the 'crazy silly school' series --/
structure CrazySillySchool where
  total_movies : ℕ
  total_books : ℕ
  movies_watched : ℕ
  movies_left : ℕ

/-- Theorem stating that the number of books read cannot be uniquely determined --/
theorem books_read_indeterminate (series : CrazySillySchool)
  (h1 : series.total_movies = 8)
  (h2 : series.total_books = 21)
  (h3 : series.movies_watched = 4)
  (h4 : series.movies_left = 4) :
  ∀ n : ℕ, n ≤ series.total_books → ∃ m : ℕ, m ≠ n ∧ m ≤ series.total_books :=
by sorry

end NUMINAMATH_CALUDE_books_read_indeterminate_l1077_107787


namespace NUMINAMATH_CALUDE_stone_slab_length_l1077_107781

theorem stone_slab_length (num_slabs : ℕ) (total_area : ℝ) (slab_length : ℝ) :
  num_slabs = 30 →
  total_area = 67.5 →
  num_slabs * (slab_length ^ 2) = total_area →
  slab_length = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_stone_slab_length_l1077_107781


namespace NUMINAMATH_CALUDE_amy_baskets_l1077_107747

/-- The number of baskets Amy can fill with candies -/
def num_baskets (chocolate_bars : ℕ) (m_and_ms_ratio : ℕ) (marshmallow_ratio : ℕ) (candies_per_basket : ℕ) : ℕ :=
  let m_and_ms := chocolate_bars * m_and_ms_ratio
  let marshmallows := m_and_ms * marshmallow_ratio
  let total_candies := chocolate_bars + m_and_ms + marshmallows
  total_candies / candies_per_basket

/-- Theorem stating that Amy will fill 25 baskets given the conditions -/
theorem amy_baskets :
  num_baskets 5 7 6 10 = 25 := by
  sorry

end NUMINAMATH_CALUDE_amy_baskets_l1077_107747


namespace NUMINAMATH_CALUDE_log_equation_solution_l1077_107714

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 3 + Real.log x / Real.log 9 = 5 → x = 3^(10/3) := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1077_107714


namespace NUMINAMATH_CALUDE_not_sum_of_three_squares_2015_l1077_107706

theorem not_sum_of_three_squares_2015 : ¬∃ (a b c : ℤ), a^2 + b^2 + c^2 = 2015 := by
  sorry

end NUMINAMATH_CALUDE_not_sum_of_three_squares_2015_l1077_107706


namespace NUMINAMATH_CALUDE_robert_reading_theorem_l1077_107763

/-- Calculates the maximum number of complete books that can be read given the reading speed, book length, and available time. -/
def max_complete_books_read (reading_speed : ℕ) (book_length : ℕ) (available_time : ℕ) : ℕ :=
  (available_time * reading_speed) / book_length

/-- Theorem: Given Robert's reading speed of 120 pages per hour, the maximum number of complete 360-page books he can read in 8 hours is 2. -/
theorem robert_reading_theorem : 
  max_complete_books_read 120 360 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_robert_reading_theorem_l1077_107763


namespace NUMINAMATH_CALUDE_circle_center_on_line_l1077_107799

theorem circle_center_on_line (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*a*x + 4*y - 6 = 0 → 
    ∃ h k : ℝ, (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 2*a*h + 4*k - 6) ∧ 
    h + 2*k + 1 = 0) →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_circle_center_on_line_l1077_107799


namespace NUMINAMATH_CALUDE_kids_on_soccer_field_l1077_107791

/-- The number of kids initially on the soccer field -/
def initial_kids : ℕ := 14

/-- The number of kids who joined the soccer field -/
def joined_kids : ℕ := 22

/-- The total number of kids on the soccer field after more kids joined -/
def total_kids : ℕ := initial_kids + joined_kids

theorem kids_on_soccer_field : total_kids = 36 := by
  sorry

end NUMINAMATH_CALUDE_kids_on_soccer_field_l1077_107791


namespace NUMINAMATH_CALUDE_library_books_theorem_l1077_107700

variable (Library : Type)
variable (is_new_edition : Library → Prop)

theorem library_books_theorem :
  (¬ ∀ (book : Library), is_new_edition book) →
  (∃ (book : Library), ¬ is_new_edition book) ∧
  (¬ ∀ (book : Library), is_new_edition book) :=
by sorry

end NUMINAMATH_CALUDE_library_books_theorem_l1077_107700


namespace NUMINAMATH_CALUDE_correlation_strength_increases_l1077_107750

-- Define the correlation coefficient as a real number between -1 and 1
def correlation_coefficient : Type := {r : ℝ // -1 ≤ r ∧ r ≤ 1}

-- Define a measure of linear correlation strength
def linear_correlation_strength (r : correlation_coefficient) : ℝ := |r.val|

-- Define a notion of "closer to 1"
def closer_to_one (r1 r2 : correlation_coefficient) : Prop :=
  |r1.val - 1| < |r2.val - 1|

-- Statement: As |r| approaches 1, the linear correlation becomes stronger
theorem correlation_strength_increases (r1 r2 : correlation_coefficient) :
  closer_to_one r1 r2 → linear_correlation_strength r1 > linear_correlation_strength r2 :=
sorry

end NUMINAMATH_CALUDE_correlation_strength_increases_l1077_107750


namespace NUMINAMATH_CALUDE_sqrt_product_equals_140_l1077_107755

theorem sqrt_product_equals_140 :
  Real.sqrt (13 + Real.sqrt (28 + Real.sqrt 281)) *
  Real.sqrt (13 - Real.sqrt (28 + Real.sqrt 281)) *
  Real.sqrt (141 + Real.sqrt 281) = 140 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_140_l1077_107755


namespace NUMINAMATH_CALUDE_rectangle_area_l1077_107774

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 16 →
  rectangle_width^2 = square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 48 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1077_107774


namespace NUMINAMATH_CALUDE_remainder_3_pow_2023_mod_5_l1077_107733

theorem remainder_3_pow_2023_mod_5 : 3^2023 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_pow_2023_mod_5_l1077_107733


namespace NUMINAMATH_CALUDE_evans_earnings_l1077_107798

/-- Proves that Evan earned $21 given the conditions of the problem -/
theorem evans_earnings (markese_earnings : ℕ) (total_earnings : ℕ) (earnings_difference : ℕ)
  (h1 : markese_earnings = 16)
  (h2 : total_earnings = 37)
  (h3 : markese_earnings + earnings_difference = total_earnings)
  (h4 : earnings_difference = 5) : 
  total_earnings - markese_earnings = 21 := by
  sorry

end NUMINAMATH_CALUDE_evans_earnings_l1077_107798


namespace NUMINAMATH_CALUDE_solve_linear_equation_l1077_107777

theorem solve_linear_equation (x : ℝ) : x + 1 = 4 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l1077_107777


namespace NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_3_to_m_l1077_107778

def m : ℕ := 2011^2 + 3^2011

theorem units_digit_of_m_squared_plus_3_to_m (m : ℕ := 2011^2 + 3^2011) : 
  (m^2 + 3^m) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_3_to_m_l1077_107778


namespace NUMINAMATH_CALUDE_no_fixed_points_composition_l1077_107794

-- Define the quadratic function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + x

-- Theorem statement
theorem no_fixed_points_composition
  (a b : ℝ)
  (h : ∀ x : ℝ, f a b x ≠ x) :
  ∀ x : ℝ, f a b (f a b x) ≠ x :=
by sorry

end NUMINAMATH_CALUDE_no_fixed_points_composition_l1077_107794


namespace NUMINAMATH_CALUDE_jennifer_future_age_l1077_107711

def jennifer_age_in_10_years : ℕ := 30

def jordana_current_age : ℕ := 80

theorem jennifer_future_age :
  jennifer_age_in_10_years = 30 :=
by
  have h1 : jordana_current_age + 10 = 3 * jennifer_age_in_10_years :=
    sorry
  sorry

#check jennifer_future_age

end NUMINAMATH_CALUDE_jennifer_future_age_l1077_107711


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1077_107728

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1/x + x/y ≥ 3 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ 1/x + x/y = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1077_107728


namespace NUMINAMATH_CALUDE_clara_stickers_l1077_107782

theorem clara_stickers (initial : ℕ) : 
  initial ≥ 10 →
  (initial - 10) % 2 = 0 →
  (initial - 10) / 2 - 45 = 45 →
  initial = 100 := by
sorry

end NUMINAMATH_CALUDE_clara_stickers_l1077_107782


namespace NUMINAMATH_CALUDE_count_equal_S_is_11_l1077_107724

/-- S(n) is the smallest positive integer divisible by each of the positive integers 1, 2, 3, ..., n -/
def S (n : ℕ) : ℕ := sorry

/-- The count of positive integers n with 1 ≤ n ≤ 100 that have S(n) = S(n+4) -/
def count_equal_S : ℕ := sorry

theorem count_equal_S_is_11 : count_equal_S = 11 := by sorry

end NUMINAMATH_CALUDE_count_equal_S_is_11_l1077_107724


namespace NUMINAMATH_CALUDE_lisa_challenge_time_l1077_107740

/-- The time remaining for Lisa to complete the hotdog-eating challenge -/
def timeRemaining (totalHotdogs : ℕ) (hotdogsEaten : ℕ) (eatingRate : ℕ) : ℚ :=
  (totalHotdogs - hotdogsEaten : ℚ) / eatingRate

/-- Theorem stating that Lisa has 5 minutes to complete the challenge -/
theorem lisa_challenge_time : 
  timeRemaining 75 20 11 = 5 := by sorry

end NUMINAMATH_CALUDE_lisa_challenge_time_l1077_107740


namespace NUMINAMATH_CALUDE_sequence_sum_bounded_l1077_107734

theorem sequence_sum_bounded (n : ℕ) (a : ℕ → ℝ) 
  (h_n : n ≥ 2)
  (h_a1 : 0 ≤ a 1)
  (h_a : ∀ i ∈ Finset.range (n - 1), a i ≤ a (i + 1) ∧ a (i + 1) ≤ 2 * a i) :
  ∃ ε : ℕ → ℝ, (∀ i ∈ Finset.range n, ε i = 1 ∨ ε i = -1) ∧ 
    0 ≤ (Finset.range n).sum (λ i => ε i * a (i + 1)) ∧
    (Finset.range n).sum (λ i => ε i * a (i + 1)) ≤ a 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_bounded_l1077_107734


namespace NUMINAMATH_CALUDE_ice_cream_bill_l1077_107701

/-- Calculate the final bill for ice cream sundaes with tip -/
theorem ice_cream_bill (price1 price2 price3 price4 : ℝ) :
  let total_price := price1 + price2 + price3 + price4
  let tip_percentage := 0.20
  let tip := total_price * tip_percentage
  let final_bill := total_price + tip
  final_bill = total_price * (1 + tip_percentage) :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_bill_l1077_107701


namespace NUMINAMATH_CALUDE_complex_expression_equality_l1077_107746

theorem complex_expression_equality (c d : ℂ) (h1 : c = 3 - 2*I) (h2 : d = 2 + 3*I) :
  3*c + 4*d + 2 = 19 + 6*I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l1077_107746


namespace NUMINAMATH_CALUDE_colonization_combinations_count_l1077_107759

/-- Represents the number of Earth-like planets -/
def earth_like_planets : ℕ := 6

/-- Represents the number of Mars-like planets -/
def mars_like_planets : ℕ := 7

/-- Represents the resource cost to colonize an Earth-like planet -/
def earth_like_cost : ℕ := 2

/-- Represents the resource cost to colonize a Mars-like planet -/
def mars_like_cost : ℕ := 1

/-- Represents the total available resources -/
def total_resources : ℕ := 14

/-- Calculates the number of ways to select planets for colonization -/
def colonization_combinations : ℕ := sorry

/-- Theorem stating that the number of colonization combinations is 336 -/
theorem colonization_combinations_count :
  colonization_combinations = 336 := by sorry

end NUMINAMATH_CALUDE_colonization_combinations_count_l1077_107759


namespace NUMINAMATH_CALUDE_polygon_rotation_theorem_l1077_107743

theorem polygon_rotation_theorem (n : ℕ) (h : n ≥ 3) 
  (a : Fin n → Fin n) (h_perm : Function.Bijective a) 
  (h_initial : ∀ i : Fin n, a i ≠ i) :
  ∃ (r : ℕ) (i j : Fin n), i ≠ j ∧ 
    (a i).val - i.val ≡ r [MOD n] ∧
    (a j).val - j.val ≡ r [MOD n] :=
sorry

end NUMINAMATH_CALUDE_polygon_rotation_theorem_l1077_107743


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l1077_107723

theorem inequality_not_always_true (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  ¬ (∀ a b c : ℝ, a > b → b > c → a * c > b * c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l1077_107723


namespace NUMINAMATH_CALUDE_power_function_through_point_l1077_107738

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x^α

-- State the theorem
theorem power_function_through_point (f : ℝ → ℝ) :
  isPowerFunction f →
  f 2 = Real.sqrt 2 / 2 →
  f 4 = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_power_function_through_point_l1077_107738


namespace NUMINAMATH_CALUDE_unique_solution_l1077_107768

theorem unique_solution (a b c : ℝ) 
  (ha : a > 4) (hb : b > 4) (hc : c > 4)
  (heq : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 45) :
  a = 12 ∧ b = 10 ∧ c = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1077_107768


namespace NUMINAMATH_CALUDE_circle_ratio_theorem_l1077_107769

/-- Given two circles ω₁ and ω₂ with centers O₁ and O₂ and radii r₁ and r₂ respectively,
    where O₂ lies on ω₁, A is an intersection point of ω₁ and ω₂, B is an intersection of line O₁O₂ with ω₂,
    and AB = O₁A, prove that r₁/r₂ can only be (√5 - 1)/2 or (√5 + 1)/2 -/
theorem circle_ratio_theorem (r₁ r₂ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ > 0) :
  (∃ (O₁ O₂ A B : ℝ × ℝ),
    (‖O₂ - O₁‖ = r₁) ∧
    (‖A - O₁‖ = r₁) ∧
    (‖A - O₂‖ = r₂) ∧
    (‖B - O₂‖ = r₂) ∧
    (∃ t : ℝ, B = O₁ + t • (O₂ - O₁)) ∧
    (‖A - B‖ = ‖A - O₁‖)) →
  (r₁ / r₂ = (Real.sqrt 5 - 1) / 2 ∨ r₁ / r₂ = (Real.sqrt 5 + 1) / 2) :=
by sorry


end NUMINAMATH_CALUDE_circle_ratio_theorem_l1077_107769


namespace NUMINAMATH_CALUDE_cube_order_l1077_107726

theorem cube_order (a b : ℝ) : a > b → a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_order_l1077_107726


namespace NUMINAMATH_CALUDE_complete_square_sum_l1077_107792

theorem complete_square_sum (x : ℝ) : 
  (∃ d e : ℤ, (x + d:ℝ)^2 = e ∧ x^2 - 10*x + 15 = 0) → 
  (∃ d e : ℤ, (x + d:ℝ)^2 = e ∧ x^2 - 10*x + 15 = 0 ∧ d + e = 5) :=
by sorry

end NUMINAMATH_CALUDE_complete_square_sum_l1077_107792


namespace NUMINAMATH_CALUDE_prime_equation_solution_l1077_107722

theorem prime_equation_solution (p : ℕ) (hp : Prime p) :
  (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x * (y^2 - p) + y * (x^2 - p) = 5 * p) ↔ p = 2 ∨ p = 3 :=
by sorry

end NUMINAMATH_CALUDE_prime_equation_solution_l1077_107722


namespace NUMINAMATH_CALUDE_nagy_birth_and_death_l1077_107764

def birth_year : ℕ := 1849
def death_year : ℕ := 1934
def grandchild_birth_year : ℕ := 1932
def num_grandchildren : ℕ := 24

theorem nagy_birth_and_death :
  (∃ (n : ℕ), birth_year = n^2) ∧
  (birth_year ≥ 1834 ∧ birth_year ≤ 1887) ∧
  (death_year - birth_year = 84) ∧
  (grandchild_birth_year - birth_year = 83) ∧
  (num_grandchildren = 24) :=
by sorry

end NUMINAMATH_CALUDE_nagy_birth_and_death_l1077_107764


namespace NUMINAMATH_CALUDE_optimal_price_achieves_target_profit_l1077_107788

/-- Represents the sales data and profit target for a fruit supermarket --/
structure FruitSales where
  cost_price : ℝ
  initial_price : ℝ
  initial_sales : ℝ
  price_reduction : ℝ
  sales_increase : ℝ
  target_profit : ℝ

/-- Calculates the optimal selling price for the fruit --/
def optimal_selling_price (data : FruitSales) : ℝ :=
  data.initial_price - (data.price_reduction * 3)

/-- Theorem stating that the optimal selling price achieves the target profit --/
theorem optimal_price_achieves_target_profit (data : FruitSales) 
  (h1 : data.cost_price = 22)
  (h2 : data.initial_price = 38)
  (h3 : data.initial_sales = 160)
  (h4 : data.price_reduction = 3)
  (h5 : data.sales_increase = 120)
  (h6 : data.target_profit = 3640) :
  let price := optimal_selling_price data
  let sales := data.initial_sales + data.sales_increase
  let profit_per_kg := price - data.cost_price
  profit_per_kg * sales = data.target_profit ∧ 
  price = 29 :=
by sorry

#eval optimal_selling_price { 
  cost_price := 22, 
  initial_price := 38, 
  initial_sales := 160, 
  price_reduction := 3, 
  sales_increase := 120, 
  target_profit := 3640 
}

end NUMINAMATH_CALUDE_optimal_price_achieves_target_profit_l1077_107788


namespace NUMINAMATH_CALUDE_max_value_a_l1077_107752

open Real

theorem max_value_a (a : ℝ) : 
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, a ≤ (1-x)/x + log x) → 
  a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_max_value_a_l1077_107752


namespace NUMINAMATH_CALUDE_shooting_probabilities_l1077_107783

/-- Represents a shooter with a given probability of hitting the target -/
structure Shooter where
  hit_prob : ℝ
  hit_prob_nonneg : 0 ≤ hit_prob
  hit_prob_le_one : hit_prob ≤ 1

/-- The probability of both shooters hitting the target -/
def both_hit (a b : Shooter) : ℝ := a.hit_prob * b.hit_prob

/-- The probability of at least one shooter hitting the target -/
def at_least_one_hit (a b : Shooter) : ℝ := 1 - (1 - a.hit_prob) * (1 - b.hit_prob)

theorem shooting_probabilities (a b : Shooter) 
  (ha : a.hit_prob = 0.9) (hb : b.hit_prob = 0.8) : 
  both_hit a b = 0.72 ∧ at_least_one_hit a b = 0.98 := by
  sorry

end NUMINAMATH_CALUDE_shooting_probabilities_l1077_107783


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l1077_107779

theorem rectangle_dimensions (x : ℝ) : 
  (x - 3 > 0) →
  (x - 3) * (3 * x + 7) = 11 * x - 4 →
  x = (13 + Real.sqrt 373) / 6 := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l1077_107779


namespace NUMINAMATH_CALUDE_cubic_factorization_l1077_107780

theorem cubic_factorization (x : ℝ) : x^3 - 4*x = x*(x+2)*(x-2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l1077_107780


namespace NUMINAMATH_CALUDE_pumpkin_pies_sold_l1077_107757

/-- Represents the number of pumpkin pies sold -/
def pumpkin_pies : ℕ := sorry

/-- The number of slices in a pumpkin pie -/
def pumpkin_slices : ℕ := 8

/-- The price of a pumpkin pie slice in cents -/
def pumpkin_price : ℕ := 500

/-- The number of slices in a custard pie -/
def custard_slices : ℕ := 6

/-- The price of a custard pie slice in cents -/
def custard_price : ℕ := 600

/-- The number of custard pies sold -/
def custard_pies : ℕ := 5

/-- The total revenue in cents -/
def total_revenue : ℕ := 34000

theorem pumpkin_pies_sold :
  pumpkin_pies * pumpkin_slices * pumpkin_price +
  custard_pies * custard_slices * custard_price = total_revenue →
  pumpkin_pies = 4 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_pies_sold_l1077_107757


namespace NUMINAMATH_CALUDE_max_price_reduction_l1077_107730

/-- The maximum price reduction for a product while maintaining a minimum profit margin -/
theorem max_price_reduction (cost_price selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 1000 →
  selling_price = 1500 →
  min_profit_margin = 0.05 →
  ∃ (max_reduction : ℝ),
    max_reduction = 450 ∧
    selling_price - max_reduction = cost_price * (1 + min_profit_margin) :=
by sorry

end NUMINAMATH_CALUDE_max_price_reduction_l1077_107730


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l1077_107707

def i : ℂ := Complex.I

def z : ℂ := i * (1 + i)

theorem z_in_second_quadrant :
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l1077_107707


namespace NUMINAMATH_CALUDE_jonessas_take_home_pay_l1077_107736

/-- Calculates the take-home pay given the total pay and tax rate -/
def takeHomePay (totalPay : ℝ) (taxRate : ℝ) : ℝ :=
  totalPay * (1 - taxRate)

/-- Proves that Jonessa's take-home pay is $450 -/
theorem jonessas_take_home_pay :
  let totalPay : ℝ := 500
  let taxRate : ℝ := 0.1
  takeHomePay totalPay taxRate = 450 := by sorry

end NUMINAMATH_CALUDE_jonessas_take_home_pay_l1077_107736


namespace NUMINAMATH_CALUDE_square_perimeter_l1077_107727

theorem square_perimeter (t : ℝ) (h1 : t > 0) : 
  (5 / 2 * t = 40) → (4 * t = 64) := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l1077_107727


namespace NUMINAMATH_CALUDE_pages_to_read_thursday_l1077_107767

def book_pages : ℕ := 158
def monday_pages : ℕ := 23
def tuesday_pages : ℕ := 38
def wednesday_pages : ℕ := 61

theorem pages_to_read_thursday (thursday_pages : ℕ) : 
  thursday_pages = 12 ↔ 
  ∃ (friday_pages : ℕ),
    friday_pages = 2 * thursday_pages ∧
    monday_pages + tuesday_pages + wednesday_pages + thursday_pages + friday_pages = book_pages :=
by sorry

end NUMINAMATH_CALUDE_pages_to_read_thursday_l1077_107767


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1077_107754

def M : Set ℕ := {0, 1, 2, 3}
def N : Set ℕ := {2, 3}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1077_107754


namespace NUMINAMATH_CALUDE_harrys_pizza_toppings_l1077_107705

/-- Calculates the number of toppings per pizza given the conditions of Harry's pizza order --/
theorem harrys_pizza_toppings : ∀ (toppings_per_pizza : ℕ),
  (14 : ℚ) * 2 + -- Cost of two large pizzas
  (2 : ℚ) * (2 * toppings_per_pizza) + -- Cost of toppings
  (((14 : ℚ) * 2 + (2 : ℚ) * (2 * toppings_per_pizza)) * (1 / 4)) -- 25% tip
  = 50 →
  toppings_per_pizza = 3 := by
  sorry

#check harrys_pizza_toppings

end NUMINAMATH_CALUDE_harrys_pizza_toppings_l1077_107705


namespace NUMINAMATH_CALUDE_hubei_population_scientific_notation_l1077_107745

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The population of Hubei Province -/
def hubei_population : ℕ := 57000000

/-- Scientific notation for Hubei population -/
def hubei_scientific : ScientificNotation :=
  { coefficient := 5.7
  , exponent := 7
  , h1 := by sorry }

/-- Theorem stating that the scientific notation correctly represents the population -/
theorem hubei_population_scientific_notation :
  (hubei_scientific.coefficient * (10 : ℝ) ^ hubei_scientific.exponent) = hubei_population := by
  sorry

end NUMINAMATH_CALUDE_hubei_population_scientific_notation_l1077_107745


namespace NUMINAMATH_CALUDE_function_minimum_value_l1077_107758

theorem function_minimum_value (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  ((x^2) / (y - 2) + (y^2) / (x - 2)) ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_function_minimum_value_l1077_107758


namespace NUMINAMATH_CALUDE_pages_per_comic_l1077_107775

theorem pages_per_comic (total_pages : ℕ) (initial_comics : ℕ) (final_comics : ℕ)
  (h1 : total_pages = 150)
  (h2 : initial_comics = 5)
  (h3 : final_comics = 11) :
  total_pages / (final_comics - initial_comics) = 25 := by
  sorry

end NUMINAMATH_CALUDE_pages_per_comic_l1077_107775


namespace NUMINAMATH_CALUDE_distance_to_focus_l1077_107717

/-- The distance from a point on a parabola to its focus -/
theorem distance_to_focus (y : ℝ) : 
  y^2 = 8 * 2 →  -- Point M(2, y) is on the parabola y^2 = 8x
  4 = (2 - (-2)) -- Distance from M to the directrix (x = -2)
    + (2 - 0)    -- Distance from M to the x-coordinate of the focus (which is at x = 0 for this parabola)
  := by sorry

end NUMINAMATH_CALUDE_distance_to_focus_l1077_107717


namespace NUMINAMATH_CALUDE_cards_per_box_l1077_107751

theorem cards_per_box (total_cards : ℕ) (unboxed_cards : ℕ) (boxes_given : ℕ) (boxes_left : ℕ) :
  total_cards = 75 →
  unboxed_cards = 5 →
  boxes_given = 2 →
  boxes_left = 5 →
  (total_cards - unboxed_cards) % (boxes_given + boxes_left) = 0 →
  (total_cards - unboxed_cards) / (boxes_given + boxes_left) = 10 := by
  sorry

end NUMINAMATH_CALUDE_cards_per_box_l1077_107751


namespace NUMINAMATH_CALUDE_monomial_satisfies_conditions_l1077_107725

-- Define a structure for monomials
structure Monomial (α : Type) [CommRing α] where
  coeff : α
  vars : List (Nat × Nat)

-- Define the monomial -2mn^2
def target_monomial : Monomial ℤ := ⟨-2, [(1, 1), (2, 2)]⟩

-- Define functions to check the conditions
def has_variables (m : Monomial ℤ) (vars : List Nat) : Prop :=
  ∀ v ∈ vars, ∃ p ∈ m.vars, v = p.1

def coefficient (m : Monomial ℤ) : ℤ := m.coeff

def degree (m : Monomial ℤ) : Nat :=
  m.vars.foldr (fun p acc => acc + p.2) 0

-- Theorem statement
theorem monomial_satisfies_conditions :
  has_variables target_monomial [1, 2] ∧
  coefficient target_monomial = -2 ∧
  degree target_monomial = 3 := by
  sorry

end NUMINAMATH_CALUDE_monomial_satisfies_conditions_l1077_107725


namespace NUMINAMATH_CALUDE_max_value_implies_A_l1077_107715

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sin x, 1)

noncomputable def n (A x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * A * Real.cos x, A / 2 * Real.cos (2 * x))

noncomputable def f (A x : ℝ) : ℝ := (m x).1 * (n A x).1 + (m x).2 * (n A x).2

theorem max_value_implies_A (A : ℝ) (h1 : A > 0) (h2 : ∀ x, f A x ≤ 6) (h3 : ∃ x, f A x = 6) : A = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_A_l1077_107715


namespace NUMINAMATH_CALUDE_four_card_selection_ways_l1077_107796

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of suits in a standard deck -/
def num_suits : ℕ := 4

/-- The number of cards in each suit -/
def cards_per_suit : ℕ := deck_size / num_suits

/-- The number of cards to choose -/
def cards_to_choose : ℕ := 4

/-- Theorem stating the number of ways to choose 4 cards from a standard deck
    with exactly two of the same suit and the other two of different suits -/
theorem four_card_selection_ways :
  (num_suits.choose 1) *
  ((num_suits - 1).choose 2) *
  (cards_per_suit.choose 2) *
  (cards_per_suit ^ 2) = 158004 := by
  sorry

end NUMINAMATH_CALUDE_four_card_selection_ways_l1077_107796


namespace NUMINAMATH_CALUDE_percentage_problem_l1077_107772

theorem percentage_problem (x : ℝ) : 
  (x / 100) * 130 = 65 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1077_107772


namespace NUMINAMATH_CALUDE_not_always_true_false_and_implies_true_or_l1077_107784

theorem not_always_true_false_and_implies_true_or : 
  ¬ ∀ (p q : Prop), (¬(p ∧ q)) → (p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_not_always_true_false_and_implies_true_or_l1077_107784


namespace NUMINAMATH_CALUDE_race_time_theorem_l1077_107737

/-- The time taken by this year's winner to complete the race around the town square. -/
def this_year_time (laps : ℕ) (square_length : ℚ) (last_year_time : ℚ) (time_improvement : ℚ) : ℚ :=
  let total_distance := laps * square_length
  let last_year_pace := last_year_time / total_distance
  let this_year_pace := last_year_pace - time_improvement
  this_year_pace * total_distance

/-- Theorem stating that this year's winner completed the race in 42 minutes. -/
theorem race_time_theorem :
  this_year_time 7 (3/4) 47.25 1 = 42 := by
  sorry

end NUMINAMATH_CALUDE_race_time_theorem_l1077_107737


namespace NUMINAMATH_CALUDE_purple_chip_count_l1077_107756

theorem purple_chip_count (blue green purple red : ℕ) (x : ℕ) :
  blue > 0 → green > 0 → purple > 0 → red > 0 →
  5 < x → x < 11 →
  1^blue * 5^green * x^purple * 11^red = 140800 →
  purple = 1 ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_purple_chip_count_l1077_107756


namespace NUMINAMATH_CALUDE_automobile_finance_credit_l1077_107785

/-- The problem statement as a theorem --/
theorem automobile_finance_credit (total_credit : ℝ) (auto_credit_percentage : ℝ) (finance_company_fraction : ℝ) : 
  total_credit = 416.6666666666667 →
  auto_credit_percentage = 0.36 →
  finance_company_fraction = 0.5 →
  finance_company_fraction * (auto_credit_percentage * total_credit) = 75 := by
sorry

end NUMINAMATH_CALUDE_automobile_finance_credit_l1077_107785


namespace NUMINAMATH_CALUDE_man_son_age_ratio_l1077_107708

/-- 
Given a man who is 20 years older than his son, and the son's present age is 18,
prove that the ratio of the man's age to his son's age in two years will be 2:1.
-/
theorem man_son_age_ratio : 
  ∀ (son_age man_age : ℕ),
  son_age = 18 →
  man_age = son_age + 20 →
  (man_age + 2) / (son_age + 2) = 2 := by
sorry

end NUMINAMATH_CALUDE_man_son_age_ratio_l1077_107708


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1077_107786

theorem complex_modulus_problem (z : ℂ) : (1 + Complex.I) * z = 2 * Complex.I → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1077_107786


namespace NUMINAMATH_CALUDE_chocolate_bar_count_l1077_107712

/-- Given a very large box containing small boxes of chocolate bars, 
    calculate the total number of chocolate bars. -/
theorem chocolate_bar_count 
  (num_small_boxes : ℕ) 
  (bars_per_small_box : ℕ) 
  (h1 : num_small_boxes = 150) 
  (h2 : bars_per_small_box = 37) : 
  num_small_boxes * bars_per_small_box = 5550 := by
  sorry

#check chocolate_bar_count

end NUMINAMATH_CALUDE_chocolate_bar_count_l1077_107712
