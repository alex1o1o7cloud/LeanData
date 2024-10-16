import Mathlib

namespace NUMINAMATH_CALUDE_consecutive_product_not_25k_plus_1_l2003_200300

theorem consecutive_product_not_25k_plus_1 (k n : ℕ) : n * (n + 1) ≠ 25 * k + 1 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_not_25k_plus_1_l2003_200300


namespace NUMINAMATH_CALUDE_product_evaluation_l2003_200381

def product_term (n : ℕ) : ℚ := (n * (n + 2) + n) / ((n + 1)^2 : ℚ)

def product_series : ℕ → ℚ
  | 0 => 1
  | n + 1 => product_series n * product_term (n + 1)

theorem product_evaluation : 
  product_series 98 = 9800 / 9801 := by sorry

end NUMINAMATH_CALUDE_product_evaluation_l2003_200381


namespace NUMINAMATH_CALUDE_reflection_of_A_wrt_BC_l2003_200349

/-- Reflection of a point with respect to a horizontal line -/
def reflect_point (p : ℝ × ℝ) (y : ℝ) : ℝ × ℝ :=
  (p.1, 2 * y - p.2)

theorem reflection_of_A_wrt_BC :
  let A : ℝ × ℝ := (2, 3)
  let B : ℝ × ℝ := (0, 1)
  let C : ℝ × ℝ := (3, 1)
  reflect_point A B.2 = (2, -1) := by
sorry

end NUMINAMATH_CALUDE_reflection_of_A_wrt_BC_l2003_200349


namespace NUMINAMATH_CALUDE_convex_polygon_25_sides_l2003_200307

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : n ≥ 3

/-- Number of diagonals in a polygon with n sides -/
def numDiagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Sum of interior angles in a polygon with n sides (in degrees) -/
def sumInteriorAngles (n : ℕ) : ℕ := (n - 2) * 180

theorem convex_polygon_25_sides :
  let p : ConvexPolygon 25 := ⟨by norm_num⟩
  numDiagonals 25 = 275 ∧ sumInteriorAngles 25 = 4140 := by sorry

end NUMINAMATH_CALUDE_convex_polygon_25_sides_l2003_200307


namespace NUMINAMATH_CALUDE_inequality_solution_function_comparison_l2003_200391

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 - 2*x + 1)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := -Real.sqrt (x^2 + 6*x + 9) + a

theorem inequality_solution (a : ℝ) :
  (∀ x : ℝ, g a x > 6 ↔ x ∈ Set.Ioo (3 - a) (a - 9)) ↔ a > 6 := by sorry

theorem function_comparison (a : ℝ) :
  (∀ x : ℝ, 2 * f x > g a x) ↔ a < 4 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_function_comparison_l2003_200391


namespace NUMINAMATH_CALUDE_imaginary_part_of_product_l2003_200353

theorem imaginary_part_of_product : Complex.im ((1 - Complex.I) * (3 + Complex.I)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_product_l2003_200353


namespace NUMINAMATH_CALUDE_vivian_daily_songs_l2003_200315

/-- The number of songs Vivian plays each day -/
def vivian_songs : ℕ := sorry

/-- The number of songs Clara plays each day -/
def clara_songs : ℕ := sorry

/-- The number of days in June -/
def june_days : ℕ := 30

/-- The number of weekend days in June -/
def weekend_days : ℕ := 8

/-- The total number of songs both listened to in June -/
def total_songs : ℕ := 396

theorem vivian_daily_songs :
  (vivian_songs = 10) ∧
  (clara_songs = vivian_songs - 2) ∧
  (total_songs = (june_days - weekend_days) * (vivian_songs + clara_songs)) := by
  sorry

end NUMINAMATH_CALUDE_vivian_daily_songs_l2003_200315


namespace NUMINAMATH_CALUDE_cost_difference_l2003_200344

def rental_initial_cost : ℕ := 20
def rental_monthly_increase : ℕ := 5
def rental_insurance : ℕ := 15
def rental_maintenance : ℕ := 10

def new_car_monthly_payment : ℕ := 30
def new_car_down_payment : ℕ := 1500
def new_car_insurance : ℕ := 20
def new_car_maintenance_first_half : ℕ := 5
def new_car_maintenance_second_half : ℕ := 10

def months : ℕ := 12

def rental_total_cost : ℕ := 
  rental_initial_cost + rental_insurance + rental_maintenance + 
  (rental_initial_cost + rental_monthly_increase + rental_insurance + rental_maintenance) * (months - 1)

def new_car_total_cost : ℕ := 
  new_car_down_payment + 
  (new_car_monthly_payment + new_car_insurance + new_car_maintenance_first_half) * (months / 2) +
  (new_car_monthly_payment + new_car_insurance + new_car_maintenance_second_half) * (months / 2)

theorem cost_difference : new_car_total_cost - rental_total_cost = 1595 := by
  sorry

end NUMINAMATH_CALUDE_cost_difference_l2003_200344


namespace NUMINAMATH_CALUDE_joey_exam_in_six_weeks_l2003_200372

/-- Joey's SAT exam preparation schedule --/
structure SATPrep where
  weekday_hours : ℕ  -- Hours studied per weekday night
  weekday_nights : ℕ  -- Number of weekday nights studied per week
  weekend_hours : ℕ  -- Hours studied per weekend day
  total_hours : ℕ  -- Total hours to be studied

/-- Calculate the number of weeks until Joey's SAT exam --/
def weeks_until_exam (prep : SATPrep) : ℚ :=
  prep.total_hours / (prep.weekday_hours * prep.weekday_nights + prep.weekend_hours * 2)

/-- Theorem: Joey's SAT exam is 6 weeks away --/
theorem joey_exam_in_six_weeks (prep : SATPrep) 
  (h1 : prep.weekday_hours = 2)
  (h2 : prep.weekday_nights = 5)
  (h3 : prep.weekend_hours = 3)
  (h4 : prep.total_hours = 96) : 
  weeks_until_exam prep = 6 := by
  sorry

end NUMINAMATH_CALUDE_joey_exam_in_six_weeks_l2003_200372


namespace NUMINAMATH_CALUDE_equation_solution_l2003_200306

theorem equation_solution : 
  ∃! y : ℚ, (5 * y - 2) / (6 * y - 6) = 3 / 4 ∧ y = -5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2003_200306


namespace NUMINAMATH_CALUDE_eight_people_seating_theorem_l2003_200387

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def seating_arrangements (total_people : ℕ) (restricted_people : ℕ) : ℕ :=
  factorial total_people - factorial (total_people - restricted_people + 1) * factorial restricted_people

theorem eight_people_seating_theorem :
  seating_arrangements 8 3 = 36000 :=
by sorry

end NUMINAMATH_CALUDE_eight_people_seating_theorem_l2003_200387


namespace NUMINAMATH_CALUDE_triangle_inequality_l2003_200373

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2003_200373


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2003_200339

theorem polynomial_evaluation : 
  let x : ℝ := 6
  (3 * x^2 + 15 * x + 7) + (4 * x^3 + 8 * x^2 - 5 * x + 10) = 1337 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2003_200339


namespace NUMINAMATH_CALUDE_purely_imaginary_sufficient_not_necessary_l2003_200369

theorem purely_imaginary_sufficient_not_necessary (m : ℝ) :
  (∃ (z : ℂ), z = (m^2 - 1 : ℂ) + (m - 1 : ℂ) * Complex.I ∧ z.re = 0 ∧ z.im ≠ 0) →
  (m = 1 ∨ m = -1) ∧
  ¬(∀ m : ℝ, (m = 1 ∨ m = -1) → 
    (∃ (z : ℂ), z = (m^2 - 1 : ℂ) + (m - 1 : ℂ) * Complex.I ∧ z.re = 0 ∧ z.im ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_sufficient_not_necessary_l2003_200369


namespace NUMINAMATH_CALUDE_quadratic_shift_properties_l2003_200394

/-- Represents a quadratic function of the form y = ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Shifts a quadratic function up by a given amount -/
def shift_up (f : QuadraticFunction) (shift : ℝ) : QuadraticFunction :=
  { f with c := f.c + shift }

theorem quadratic_shift_properties (f : QuadraticFunction) :
  let f_shifted := shift_up f 3
  (f.a = f_shifted.a) ∧ 
  (-f.b / (2 * f.a) = -f_shifted.b / (2 * f_shifted.a)) ∧
  (f.c ≠ f_shifted.c) := by sorry

end NUMINAMATH_CALUDE_quadratic_shift_properties_l2003_200394


namespace NUMINAMATH_CALUDE_routes_on_grid_l2003_200324

/-- The number of routes on a 3x3 grid from top-left to bottom-right -/
def num_routes : ℕ := 20

/-- The size of the grid -/
def grid_size : ℕ := 3

/-- The total number of moves required -/
def total_moves : ℕ := 2 * grid_size

/-- The number of moves in each direction -/
def moves_per_direction : ℕ := grid_size

theorem routes_on_grid : 
  num_routes = Nat.choose total_moves moves_per_direction :=
sorry

end NUMINAMATH_CALUDE_routes_on_grid_l2003_200324


namespace NUMINAMATH_CALUDE_base4_10201_to_decimal_l2003_200393

def base4_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

theorem base4_10201_to_decimal :
  base4_to_decimal [1, 0, 2, 0, 1] = 289 := by
  sorry

end NUMINAMATH_CALUDE_base4_10201_to_decimal_l2003_200393


namespace NUMINAMATH_CALUDE_sector_radius_l2003_200367

/-- The radius of a circle given the area of a sector and its central angle -/
theorem sector_radius (area : ℝ) (angle : ℝ) (pi : ℝ) (h1 : area = 52.8) (h2 : angle = 42) 
  (h3 : pi = Real.pi) (h4 : area = (angle / 360) * pi * (radius ^ 2)) : radius = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_sector_radius_l2003_200367


namespace NUMINAMATH_CALUDE_pencil_pen_cost_l2003_200355

/-- Given the costs of different combinations of pencils, pens, and markers,
    prove that the cost of one pencil and one pen combined is $1.12 -/
theorem pencil_pen_cost (pencil_cost pen_cost marker_cost : ℚ) : 
  (3 * pencil_cost + 2 * pen_cost + marker_cost = 21/5) →
  (pencil_cost + 3 * pen_cost + 2 * marker_cost = 19/4) →
  (2 * marker_cost = 3) →
  (pencil_cost + pen_cost = 28/25) :=
by sorry

end NUMINAMATH_CALUDE_pencil_pen_cost_l2003_200355


namespace NUMINAMATH_CALUDE_teds_chocolates_l2003_200385

theorem teds_chocolates : ∃ (x : ℚ), 
  x > 0 ∧ 
  (3/16 * x - 3/4 - 5 = 10) ∧ 
  x = 84 := by
  sorry

end NUMINAMATH_CALUDE_teds_chocolates_l2003_200385


namespace NUMINAMATH_CALUDE_max_value_a_l2003_200365

theorem max_value_a (a b c d : ℕ+) 
  (h1 : a < 3 * b)
  (h2 : b < 2 * c)
  (h3 : c < 5 * d)
  (h4 : d < 50) :
  a ≤ 1460 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 1460 ∧ 
    a' < 3 * b' ∧ 
    b' < 2 * c' ∧ 
    c' < 5 * d' ∧ 
    d' < 50 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_a_l2003_200365


namespace NUMINAMATH_CALUDE_jake_weight_loss_l2003_200317

def jake_weight : ℝ := 93
def total_weight : ℝ := 132

theorem jake_weight_loss : ∃ (x : ℝ), 
  x ≥ 0 ∧ 
  jake_weight - x = 2 * (total_weight - jake_weight) ∧ 
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_jake_weight_loss_l2003_200317


namespace NUMINAMATH_CALUDE_min_value_fourth_power_l2003_200351

theorem min_value_fourth_power (x : ℝ) : 
  x ∈ Set.Icc 0 1 → (x^4 + (1-x)^4 : ℝ) ≥ 1/8 ∧ ∃ y ∈ Set.Icc 0 1, y^4 + (1-y)^4 = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fourth_power_l2003_200351


namespace NUMINAMATH_CALUDE_sin_360_degrees_l2003_200330

theorem sin_360_degrees : Real.sin (2 * Real.pi) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_360_degrees_l2003_200330


namespace NUMINAMATH_CALUDE_john_car_profit_l2003_200327

/-- Calculates the profit from fixing and racing a car given the following parameters:
    * original_cost: The original cost to fix the car
    * discount_percentage: The discount percentage on the repair cost
    * prize_money: The total prize money won
    * kept_percentage: The percentage of prize money kept by the racer
-/
def calculate_car_profit (original_cost discount_percentage prize_money kept_percentage : ℚ) : ℚ :=
  let discounted_cost := original_cost * (1 - discount_percentage / 100)
  let kept_prize := prize_money * (kept_percentage / 100)
  kept_prize - discounted_cost

/-- Theorem stating that given the specific conditions of John's car repair and race,
    his profit is $47,000 -/
theorem john_car_profit :
  calculate_car_profit 20000 20 70000 90 = 47000 := by
  sorry

end NUMINAMATH_CALUDE_john_car_profit_l2003_200327


namespace NUMINAMATH_CALUDE_isabella_escalator_time_l2003_200310

/-- Represents the time it takes Isabella to ride an escalator under different conditions -/
def EscalatorTime (walk_time_stopped : ℝ) (walk_time_moving : ℝ) : Prop :=
  ∃ (escalator_speed : ℝ) (isabella_speed : ℝ),
    escalator_speed > 0 ∧
    isabella_speed > 0 ∧
    walk_time_stopped * isabella_speed = walk_time_moving * (isabella_speed + escalator_speed) ∧
    walk_time_stopped / escalator_speed = 45

theorem isabella_escalator_time :
  EscalatorTime 90 30 :=
sorry

end NUMINAMATH_CALUDE_isabella_escalator_time_l2003_200310


namespace NUMINAMATH_CALUDE_right_triangle_sets_l2003_200374

def is_right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

theorem right_triangle_sets :
  (is_right_triangle 1 (Real.sqrt 2) (Real.sqrt 3)) ∧
  ¬(is_right_triangle 4 5 6) ∧
  (is_right_triangle 3 4 5) ∧
  (is_right_triangle 9 12 15) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l2003_200374


namespace NUMINAMATH_CALUDE_distance_is_60_l2003_200368

/-- The distance between a boy's house and school. -/
def distance : ℝ := sorry

/-- The time it takes for the boy to reach school when arriving on time. -/
def on_time : ℝ := sorry

/-- Assertion that when traveling at 10 km/hr, the boy arrives 2 hours late. -/
axiom late_arrival : on_time + 2 = distance / 10

/-- Assertion that when traveling at 20 km/hr, the boy arrives 1 hour early. -/
axiom early_arrival : on_time - 1 = distance / 20

/-- Theorem stating that the distance between the boy's house and school is 60 kilometers. -/
theorem distance_is_60 : distance = 60 := by sorry

end NUMINAMATH_CALUDE_distance_is_60_l2003_200368


namespace NUMINAMATH_CALUDE_count_four_digit_divisible_by_5_ending_0_is_900_l2003_200354

/-- A function that counts the number of positive four-digit integers divisible by 5 and ending in 0 -/
def count_four_digit_divisible_by_5_ending_0 : ℕ :=
  let first_digit := Finset.range 9  -- 1 to 9
  let second_digit := Finset.range 10  -- 0 to 9
  let third_digit := Finset.range 10  -- 0 to 9
  (first_digit.card * second_digit.card * third_digit.card : ℕ)

/-- Theorem stating that the count of positive four-digit integers divisible by 5 and ending in 0 is 900 -/
theorem count_four_digit_divisible_by_5_ending_0_is_900 :
  count_four_digit_divisible_by_5_ending_0 = 900 := by
  sorry

end NUMINAMATH_CALUDE_count_four_digit_divisible_by_5_ending_0_is_900_l2003_200354


namespace NUMINAMATH_CALUDE_f_value_at_7_6_l2003_200311

def periodic_function (f : ℝ → ℝ) (period : ℝ) : Prop :=
  ∀ x, f (x + period) = f x

theorem f_value_at_7_6 (f : ℝ → ℝ) (h1 : periodic_function f 4) 
  (h2 : ∀ x ∈ Set.Icc (-2) 2, f x = x + 1) : 
  f 7.6 = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_7_6_l2003_200311


namespace NUMINAMATH_CALUDE_reciprocal_abs_eq_neg_self_l2003_200322

theorem reciprocal_abs_eq_neg_self :
  ∃! (a : ℝ), |1 / a| = -a :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_reciprocal_abs_eq_neg_self_l2003_200322


namespace NUMINAMATH_CALUDE_parabola_focus_to_directrix_distance_l2003_200356

/-- Represents a parabola y^2 = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

theorem parabola_focus_to_directrix_distance
  (par : Parabola)
  (h_dist : Real.sqrt ((1 - par.p / 2)^2 + (Real.sqrt (2 * par.p) - 0)^2) = 4) :
  par.p = 6 :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_to_directrix_distance_l2003_200356


namespace NUMINAMATH_CALUDE_A_eq_zero_two_l2003_200358

/-- The set of real numbers a for which the equation ax^2 - 4x + 2 = 0 has exactly one solution -/
def A : Set ℝ :=
  {a : ℝ | ∃! x : ℝ, a * x^2 - 4 * x + 2 = 0}

/-- Theorem stating that A is equal to the set {0, 2} -/
theorem A_eq_zero_two : A = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_A_eq_zero_two_l2003_200358


namespace NUMINAMATH_CALUDE_jasons_initial_money_l2003_200305

theorem jasons_initial_money (initial_money : ℝ) : 
  let remaining_after_books := (3/4 : ℝ) * initial_money - 10
  let remaining_after_dvds := remaining_after_books - (2/5 : ℝ) * remaining_after_books - 8
  remaining_after_dvds = 130 → initial_money = 320 := by
sorry

end NUMINAMATH_CALUDE_jasons_initial_money_l2003_200305


namespace NUMINAMATH_CALUDE_triangle_area_l2003_200377

/-- The area of a triangle ABC with given side lengths and angle -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  b = 2 →
  c = 2 * Real.sqrt 2 →
  C = π / 4 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 + 1 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l2003_200377


namespace NUMINAMATH_CALUDE_elderly_arrangement_count_l2003_200331

/-- The number of ways to arrange n distinct objects. -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to choose k objects from n distinct objects, where order matters. -/
def arrangements (n k : ℕ) : ℕ := 
  if k ≤ n then
    Nat.factorial n / Nat.factorial (n - k)
  else
    0

/-- The number of ways to arrange 5 volunteers and 2 elderly people in a line,
    where the elderly people must be adjacent and not at the ends. -/
def elderly_arrangement : ℕ := 
  arrangements 5 2 * permutations 4 * permutations 2

theorem elderly_arrangement_count : elderly_arrangement = 960 := by
  sorry

end NUMINAMATH_CALUDE_elderly_arrangement_count_l2003_200331


namespace NUMINAMATH_CALUDE_power_of_four_three_halves_l2003_200303

theorem power_of_four_three_halves : (4 : ℝ) ^ (3/2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_of_four_three_halves_l2003_200303


namespace NUMINAMATH_CALUDE_radical_simplification_l2003_200388

theorem radical_simplification (y : ℝ) (h : y ≥ 0) :
  Real.sqrt (45 * y) * Real.sqrt (18 * y) * Real.sqrt (22 * y) = 18 * y * Real.sqrt (55 * y) := by
  sorry

end NUMINAMATH_CALUDE_radical_simplification_l2003_200388


namespace NUMINAMATH_CALUDE_mysterious_quadratic_polynomial_value_at_zero_l2003_200348

/-- A quadratic polynomial with real coefficients -/
def QuadraticPolynomial (b c : ℝ) : ℝ → ℝ := fun x ↦ x^2 + b*x + c

/-- A polynomial is mysterious if p(p(x))=0 has exactly four real roots, including multiplicities -/
def IsMysteri6ous (p : ℝ → ℝ) : Prop :=
  ∃ (roots : Finset ℝ), (∀ x, x ∈ roots ↔ p (p x) = 0) ∧ roots.card = 4

/-- The sum of roots of a quadratic polynomial -/
def SumOfRoots (b c : ℝ) : ℝ := -b

theorem mysterious_quadratic_polynomial_value_at_zero
  (b c : ℝ)
  (h_mysterious : IsMysteri6ous (QuadraticPolynomial b c))
  (h_minimal_sum : ∀ b' c', IsMysteri6ous (QuadraticPolynomial b' c') → SumOfRoots b c ≤ SumOfRoots b' c') :
  QuadraticPolynomial b c 0 = 4 := by
  sorry

end NUMINAMATH_CALUDE_mysterious_quadratic_polynomial_value_at_zero_l2003_200348


namespace NUMINAMATH_CALUDE_algebraic_expression_simplification_l2003_200357

theorem algebraic_expression_simplification (x : ℝ) (h : x = -4) :
  (x^2 - 2*x) / (x - 3) / ((1 / (x + 3) + 1 / (x - 3))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_simplification_l2003_200357


namespace NUMINAMATH_CALUDE_min_distance_square_l2003_200390

theorem min_distance_square (a b c d : ℝ) 
  (h1 : Real.log (b + 1) + a - 3 * b = 0) 
  (h2 : 2 * d - c + Real.sqrt 5 = 0) : 
  ∃ (min_val : ℝ), min_val = 1 ∧ 
  ∀ (x y : ℝ), Real.log (x + 1) + y - 3 * x = 0 → 
  ∀ (u v : ℝ), 2 * u - v + Real.sqrt 5 = 0 → 
  (y - v)^2 + (x - u)^2 ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_distance_square_l2003_200390


namespace NUMINAMATH_CALUDE_range_of_m_l2003_200395

def p (a : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 - 2 * a * x + 2 * a + 5 = 0 → x ∈ ({x | 4 * x^2 - 2 * a * x + 2 * a + 5 = 0} : Set ℝ)

def q (m : ℝ) : Prop := ∀ x : ℝ, 1 - m ≤ x ∧ x ≤ 1 + m

theorem range_of_m :
  (∀ a : ℝ, (¬(p a) → ∃ m : ℝ, m > 0 ∧ ¬(q m)) ∧
   (∃ m : ℝ, m > 0 ∧ ¬(q m) ∧ p a)) →
  {m : ℝ | m ≥ 9} = {m : ℝ | m > 0 ∧ (∀ a : ℝ, p a → q m)} :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2003_200395


namespace NUMINAMATH_CALUDE_alphabet_size_is_40_l2003_200302

/-- Represents the number of letters in an alphabet with specific characteristics -/
def alphabet_size (dot_and_line : ℕ) (line_no_dot : ℕ) (dot_no_line : ℕ) : ℕ :=
  dot_and_line + line_no_dot + dot_no_line

/-- Theorem stating the size of the alphabet given specific counts of letter types -/
theorem alphabet_size_is_40 : 
  alphabet_size 9 24 7 = 40 :=
by sorry

end NUMINAMATH_CALUDE_alphabet_size_is_40_l2003_200302


namespace NUMINAMATH_CALUDE_circle_intersection_area_l2003_200375

noncomputable def circleIntersection (r : ℝ) (bd ed : ℝ) : ℝ :=
  let ad := 2 * r + bd
  let ea := Real.sqrt (ad^2 + ed^2)
  let ec := ed^2 / ea
  let ac := ea - ec
  let bc := Real.sqrt ((2*r)^2 - ac^2)
  1/2 * bc * ac

theorem circle_intersection_area (r bd ed : ℝ) (hr : r = 4) (hbd : bd = 6) (hed : ed = 5) :
  circleIntersection r bd ed = 11627.6 / 221 :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_area_l2003_200375


namespace NUMINAMATH_CALUDE_historicalFictionNewReleasesFractionIs12_47_l2003_200336

/-- Represents a bookstore inventory --/
structure Inventory where
  total : ℕ
  historicalFiction : ℕ
  historicalFictionNewReleases : ℕ
  otherNewReleases : ℕ

/-- Conditions of the bookstore inventory --/
def validInventory (i : Inventory) : Prop :=
  i.historicalFiction = (30 * i.total) / 100 ∧
  i.historicalFictionNewReleases = (40 * i.historicalFiction) / 100 ∧
  i.otherNewReleases = (50 * (i.total - i.historicalFiction)) / 100

/-- The fraction of new releases that are historical fiction --/
def historicalFictionNewReleasesFraction (i : Inventory) : ℚ :=
  i.historicalFictionNewReleases / (i.historicalFictionNewReleases + i.otherNewReleases)

/-- Theorem stating the fraction of new releases that are historical fiction --/
theorem historicalFictionNewReleasesFractionIs12_47 (i : Inventory) 
  (h : validInventory i) : historicalFictionNewReleasesFraction i = 12 / 47 := by
  sorry

end NUMINAMATH_CALUDE_historicalFictionNewReleasesFractionIs12_47_l2003_200336


namespace NUMINAMATH_CALUDE_product_of_integers_l2003_200399

theorem product_of_integers (p q r : ℤ) : 
  p ≠ 0 → q ≠ 0 → r ≠ 0 →
  p + q + r = 27 →
  1 / p + 1 / q + 1 / r + 300 / (p * q * r) = 1 →
  p * q * r = 984 := by
sorry

end NUMINAMATH_CALUDE_product_of_integers_l2003_200399


namespace NUMINAMATH_CALUDE_estimate_fish_population_l2003_200321

/-- Estimates the total number of fish in a pond using the capture-recapture method. -/
theorem estimate_fish_population
  (initial_catch : ℕ)
  (second_catch : ℕ)
  (marked_recaught : ℕ)
  (h1 : initial_catch = 100)
  (h2 : second_catch = 200)
  (h3 : marked_recaught = 5) :
  (initial_catch * second_catch) / marked_recaught = 4000 :=
sorry

end NUMINAMATH_CALUDE_estimate_fish_population_l2003_200321


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_symmetry_l2003_200382

-- Define the points
variable (A B C D A₁ B₁ C₁ D₁ P : Point)

-- Define the property of being cyclic
def is_cyclic (A B C D : Point) : Prop := sorry

-- Define symmetry with respect to a point
def symmetrical_wrt (A B : Point) (P : Point) : Prop := sorry

-- State the theorem
theorem cyclic_quadrilateral_symmetry 
  (h1 : symmetrical_wrt A A₁ P) 
  (h2 : symmetrical_wrt B B₁ P) 
  (h3 : symmetrical_wrt C C₁ P) 
  (h4 : symmetrical_wrt D D₁ P)
  (h5 : is_cyclic A₁ B C D)
  (h6 : is_cyclic A B₁ C D)
  (h7 : is_cyclic A B C₁ D) :
  is_cyclic A B C D₁ := by sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_symmetry_l2003_200382


namespace NUMINAMATH_CALUDE_max_value_of_circle_l2003_200361

theorem max_value_of_circle (x y : ℝ) :
  x^2 + y^2 + 4*x - 2*y - 4 = 0 →
  x^2 + y^2 ≤ 14 + 6 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_circle_l2003_200361


namespace NUMINAMATH_CALUDE_choir_members_count_l2003_200386

theorem choir_members_count :
  ∃! n : ℕ, 150 ≤ n ∧ n ≤ 300 ∧ n % 10 = 6 ∧ n % 11 = 6 ∧ n = 226 := by
  sorry

end NUMINAMATH_CALUDE_choir_members_count_l2003_200386


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2003_200301

-- Problem 1
theorem problem_1 :
  2⁻¹ + |Real.sqrt 6 - 3| + 2 * Real.sqrt 3 * Real.sin (45 * π / 180) - (-2)^2023 * (1/2)^2023 = 9/2 := by
  sorry

-- Problem 2
theorem problem_2 (a : ℝ) (h : a = 3) :
  (3 / (a + 1) - a + 1) / ((a^2 - 4) / (a^2 + 2*a + 1)) = -4 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2003_200301


namespace NUMINAMATH_CALUDE_circle_ray_no_intersection_l2003_200346

/-- Given a circle (x-a)^2 + y^2 = 4 and a ray y = √3x (x ≥ 0) with no common points,
    the range of values for the real number a is {a | a < -2 or a > (4/3)√3}. -/
theorem circle_ray_no_intersection (a : ℝ) : 
  (∀ x y : ℝ, (x - a)^2 + y^2 = 4 → y ≠ Real.sqrt 3 * x ∨ x < 0) ↔ 
  (a < -2 ∨ a > (4/3) * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_circle_ray_no_intersection_l2003_200346


namespace NUMINAMATH_CALUDE_absolute_value_fraction_sum_l2003_200362

theorem absolute_value_fraction_sum (x y : ℝ) (h1 : x < y) (h2 : y < 0) :
  |x| / x + |x * y| / (x * y) = 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_fraction_sum_l2003_200362


namespace NUMINAMATH_CALUDE_limit_of_sequence_l2003_200392

def a (n : ℕ) : ℚ := (2 - 3 * n^2) / (4 + 5 * n^2)

theorem limit_of_sequence :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - (-3/5)| < ε := by sorry

end NUMINAMATH_CALUDE_limit_of_sequence_l2003_200392


namespace NUMINAMATH_CALUDE_vitamin_c_in_two_juices_l2003_200334

/-- Amount of vitamin C (in mg) in one 8-oz glass of apple juice -/
def apple_juice_vc : ℝ := 103

/-- Amount of vitamin C (in mg) in one 8-oz glass of orange juice -/
def orange_juice_vc : ℝ := 82

/-- Total amount of vitamin C (in mg) in two glasses of apple juice and three glasses of orange juice -/
def total_vc_five_glasses : ℝ := 452

/-- Theorem stating that one glass each of apple and orange juice contain 185 mg of vitamin C -/
theorem vitamin_c_in_two_juices :
  apple_juice_vc + orange_juice_vc = 185 ∧
  2 * apple_juice_vc + 3 * orange_juice_vc = total_vc_five_glasses :=
sorry

end NUMINAMATH_CALUDE_vitamin_c_in_two_juices_l2003_200334


namespace NUMINAMATH_CALUDE_abs_one_minus_sqrt_three_l2003_200328

theorem abs_one_minus_sqrt_three (h : Real.sqrt 3 > 1) :
  |1 - Real.sqrt 3| = Real.sqrt 3 - 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_one_minus_sqrt_three_l2003_200328


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2003_200309

theorem inequality_system_solution (x : ℝ) :
  (3 * (x - 2) ≤ x - 4 ∧ (1 + 2 * x) / 3 > x - 1) ↔ x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2003_200309


namespace NUMINAMATH_CALUDE_bullseye_mean_hits_l2003_200343

/-- The mean number of hits in a series of independent Bernoulli trials -/
def meanHits (p : ℝ) (n : ℕ) : ℝ := n * p

/-- The probability of hitting the bullseye -/
def bullseyeProbability : ℝ := 0.9

/-- The number of consecutive shots -/
def numShots : ℕ := 10

theorem bullseye_mean_hits :
  meanHits bullseyeProbability numShots = 9 := by
  sorry

end NUMINAMATH_CALUDE_bullseye_mean_hits_l2003_200343


namespace NUMINAMATH_CALUDE_choose_from_two_bags_l2003_200380

/-- The number of ways to choose one item from each of two sets -/
def choose_one_from_each (m n : ℕ) : ℕ := m * n

/-- The number of balls in the red bag -/
def red_balls : ℕ := 3

/-- The number of balls in the blue bag -/
def blue_balls : ℕ := 5

/-- Theorem: The number of ways to choose one ball from the red bag and one from the blue bag is 15 -/
theorem choose_from_two_bags : choose_one_from_each red_balls blue_balls = 15 := by
  sorry

end NUMINAMATH_CALUDE_choose_from_two_bags_l2003_200380


namespace NUMINAMATH_CALUDE_x_less_than_y_l2003_200360

theorem x_less_than_y (a : ℝ) : (a + 3) * (a - 5) < (a + 2) * (a - 4) := by
  sorry

end NUMINAMATH_CALUDE_x_less_than_y_l2003_200360


namespace NUMINAMATH_CALUDE_find_k_l2003_200366

theorem find_k (k : ℝ) (h : 16 / k = 4) : k = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l2003_200366


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2003_200397

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, 2 * x^3 + x * y - 7 = 0 ↔ 
    (x = -7 ∧ y = -99) ∨ (x = -1 ∧ y = -9) ∨ (x = 1 ∧ y = 5) ∨ (x = 7 ∧ y = -97) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2003_200397


namespace NUMINAMATH_CALUDE_oldest_to_rick_age_ratio_l2003_200347

/-- Proves that the ratio of the oldest brother's age to Rick's age is 2:1 --/
theorem oldest_to_rick_age_ratio :
  ∀ (rick_age oldest_age middle_age smallest_age youngest_age : ℕ),
    rick_age = 15 →
    ∃ (k : ℕ), oldest_age = k * rick_age →
    middle_age = oldest_age / 3 →
    smallest_age = middle_age / 2 →
    youngest_age = smallest_age - 2 →
    youngest_age = 3 →
    oldest_age / rick_age = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_oldest_to_rick_age_ratio_l2003_200347


namespace NUMINAMATH_CALUDE_intersection_coordinate_product_prove_intersection_coordinate_product_l2003_200341

/-- The product of coordinates of intersection points of two specific circles is 8 -/
theorem intersection_coordinate_product : ℝ → Prop := fun r =>
  ∀ x y : ℝ,
  (x^2 - 4*x + y^2 - 8*y + 20 = 0 ∧ x^2 - 6*x + y^2 - 8*y + 25 = 0) →
  r = x * y ∧ r = 8

/-- Proof of the theorem -/
theorem prove_intersection_coordinate_product :
  ∃ r : ℝ, intersection_coordinate_product r :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_coordinate_product_prove_intersection_coordinate_product_l2003_200341


namespace NUMINAMATH_CALUDE_simplify_expression_l2003_200329

theorem simplify_expression : (256 : ℝ) ^ (1/4) * (125 : ℝ) ^ (1/2) = 20 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2003_200329


namespace NUMINAMATH_CALUDE_bucket_fills_theorem_l2003_200320

/-- Calculates the number of times a bucket is filled to reach the top of a bathtub. -/
def bucket_fills_to_top (bucket_capacity : ℕ) (buckets_removed : ℕ) (weekly_usage : ℕ) (days_per_week : ℕ) : ℕ :=
  let daily_usage := weekly_usage / days_per_week
  let removed_water := buckets_removed * bucket_capacity
  let full_tub_water := daily_usage + removed_water
  full_tub_water / bucket_capacity

/-- Theorem stating that under given conditions, the bucket is filled 14 times to reach the top. -/
theorem bucket_fills_theorem :
  bucket_fills_to_top 120 3 9240 7 = 14 := by
  sorry

end NUMINAMATH_CALUDE_bucket_fills_theorem_l2003_200320


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l2003_200376

theorem perfect_square_trinomial (b : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, 16 * x^2 - b * x + 9 = (a * x + 3)^2) ↔ b = 24 ∨ b = -24 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l2003_200376


namespace NUMINAMATH_CALUDE_notification_completeness_l2003_200383

/-- Represents a point in the kingdom --/
structure Point where
  x : Real
  y : Real

/-- Represents the kingdom --/
structure Kingdom where
  side_length : Real
  residents : Set Point

/-- Represents the notification process --/
def NotificationProcess (k : Kingdom) (speed : Real) (start_time : Real) (end_time : Real) :=
  ∀ p ∈ k.residents, ∃ t : Real, start_time ≤ t ∧ t ≤ end_time ∧
    ∃ q : Point, q ∈ k.residents ∧ 
    Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2) ≤ speed * (t - start_time)

theorem notification_completeness 
  (k : Kingdom) 
  (h1 : k.side_length = 2) 
  (h2 : ∀ p ∈ k.residents, 0 ≤ p.x ∧ p.x ≤ k.side_length ∧ 0 ≤ p.y ∧ p.y ≤ k.side_length)
  (speed : Real) 
  (h3 : speed = 3) 
  (start_time end_time : Real) 
  (h4 : start_time = 12) 
  (h5 : end_time = 18) :
  NotificationProcess k speed start_time end_time :=
sorry

end NUMINAMATH_CALUDE_notification_completeness_l2003_200383


namespace NUMINAMATH_CALUDE_crayon_boxes_l2003_200342

theorem crayon_boxes (total_crayons : ℕ) (full_boxes : ℕ) (loose_crayons : ℕ) (friend_crayons : ℕ) :
  total_crayons = 85 →
  full_boxes = 5 →
  loose_crayons = 5 →
  friend_crayons = 27 →
  (total_crayons - loose_crayons) / full_boxes = 16 →
  ((loose_crayons + friend_crayons) + ((total_crayons - loose_crayons) / full_boxes - 1)) / ((total_crayons - loose_crayons) / full_boxes) = 2 := by
  sorry

#check crayon_boxes

end NUMINAMATH_CALUDE_crayon_boxes_l2003_200342


namespace NUMINAMATH_CALUDE_transistors_in_2010_l2003_200308

/-- Moore's law doubling period in years -/
def moores_law_period : ℕ := 2

/-- Initial year for calculation -/
def initial_year : ℕ := 1992

/-- Target year for calculation -/
def target_year : ℕ := 2010

/-- Initial number of transistors in 1992 -/
def initial_transistors : ℕ := 500000

/-- Calculate the number of transistors in a given year according to Moore's law -/
def transistors_in_year (year : ℕ) : ℕ :=
  initial_transistors * 2^((year - initial_year) / moores_law_period)

/-- Theorem stating the number of transistors in 2010 -/
theorem transistors_in_2010 :
  transistors_in_year target_year = 256000000 := by
  sorry

end NUMINAMATH_CALUDE_transistors_in_2010_l2003_200308


namespace NUMINAMATH_CALUDE_largest_circle_radius_l2003_200316

/-- Represents a standard chessboard --/
structure Chessboard :=
  (size : ℕ)
  (is_standard : size = 8)

/-- Represents a circle on the chessboard --/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- Checks if a circle intersects any white square on the chessboard --/
def intersects_white_square (c : Circle) (b : Chessboard) : Prop :=
  sorry

/-- The largest circle that doesn't intersect any white square --/
def largest_circle (b : Chessboard) : Circle :=
  sorry

theorem largest_circle_radius (b : Chessboard) :
  (largest_circle b).radius = (Real.sqrt 10) / 2 :=
sorry

end NUMINAMATH_CALUDE_largest_circle_radius_l2003_200316


namespace NUMINAMATH_CALUDE_monkey_climbing_time_l2003_200378

/-- Monkey climbing problem -/
theorem monkey_climbing_time (tree_height : ℕ) (climb_rate : ℕ) (slip_rate : ℕ) : 
  tree_height = 22 ∧ climb_rate = 3 ∧ slip_rate = 2 → 
  (tree_height - 1) / (climb_rate - slip_rate) + 1 = 22 := by
sorry

end NUMINAMATH_CALUDE_monkey_climbing_time_l2003_200378


namespace NUMINAMATH_CALUDE_logarithm_sum_simplification_l2003_200323

theorem logarithm_sum_simplification :
  1 / (Real.log 3 / Real.log 20 + 1) +
  1 / (Real.log 4 / Real.log 15 + 1) +
  1 / (Real.log 7 / Real.log 12 + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_simplification_l2003_200323


namespace NUMINAMATH_CALUDE_inequality_proof_l2003_200313

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≤ 1) : x^6 - y^6 + 2*y^3 < π/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2003_200313


namespace NUMINAMATH_CALUDE_min_X_value_l2003_200340

def F (X : ℤ) : List ℤ := [-4, -1, 0, 6, X]

def F_new (X : ℤ) : List ℤ := [2, 3, 0, 6, X]

def mean (l : List ℤ) : ℚ := (l.sum : ℚ) / l.length

theorem min_X_value : 
  ∀ X : ℤ, (mean (F_new X) ≥ 2 * mean (F X)) → X ≥ 9 ∧
  ∀ Y : ℤ, Y < 9 → mean (F_new Y) < 2 * mean (F Y) :=
sorry

end NUMINAMATH_CALUDE_min_X_value_l2003_200340


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l2003_200314

theorem log_sum_equals_two : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l2003_200314


namespace NUMINAMATH_CALUDE_fish_problem_l2003_200325

theorem fish_problem (west north left : ℕ) (E : ℕ) : 
  west = 1800 →
  north = 500 →
  left = 2870 →
  (3 * E) / 5 + west / 4 + north = left →
  E = 3200 :=
by
  sorry

end NUMINAMATH_CALUDE_fish_problem_l2003_200325


namespace NUMINAMATH_CALUDE_mrs_susnas_grade_distribution_l2003_200332

/-- Represents the fraction of students getting each grade in Mrs. Susna's class -/
structure GradeDistribution where
  a : ℚ
  b : ℚ
  c : ℚ
  d : ℚ
  f : ℚ
  passingGrade : ℚ

/-- The actual grade distribution in Mrs. Susna's class -/
def mrsSusnasClass : GradeDistribution where
  b := 1/2
  c := 1/8
  d := 1/12
  f := 1/24
  passingGrade := 7/8
  a := 0  -- We'll prove this value

theorem mrs_susnas_grade_distribution :
  let g := mrsSusnasClass
  g.a + g.b + g.c + g.d + g.f = 1 ∧
  g.a + g.b + g.c = g.passingGrade ∧
  g.a = 1/8 :=
by sorry

end NUMINAMATH_CALUDE_mrs_susnas_grade_distribution_l2003_200332


namespace NUMINAMATH_CALUDE_tangent_line_intercept_l2003_200364

/-- Given a curve y = ax + ln x with a tangent line y = 2x + b at the point (1, a), prove that b = -1 -/
theorem tangent_line_intercept (a : ℝ) : 
  (∃ (f : ℝ → ℝ), f = λ x => a * x + Real.log x) →  -- Curve definition
  (∃ (g : ℝ → ℝ), g = λ x => 2 * x + b) →           -- Tangent line definition
  (∃ (x₀ : ℝ), x₀ = 1 ∧ f x₀ = a) →                 -- Point of tangency
  (∀ x, (deriv f) x = a + 1 / x) →                  -- Derivative of f
  (deriv f) 1 = 2 →                                 -- Slope at x = 1 equals 2
  b = -1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_intercept_l2003_200364


namespace NUMINAMATH_CALUDE_square_side_length_l2003_200359

theorem square_side_length (rectangle_length : ℝ) (rectangle_width : ℝ) (square_side : ℝ) :
  rectangle_length = 9 →
  rectangle_width = 16 →
  rectangle_length * rectangle_width = square_side * square_side →
  square_side = 12 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l2003_200359


namespace NUMINAMATH_CALUDE_minutkin_bedtime_l2003_200338

/-- Represents the number of minutes in a day -/
def minutes_per_day : ℕ := 24 * 60

/-- Represents the time Minutkin winds his watch in the morning (8:30 AM) -/
def morning_wind_time : ℕ := 8 * 60 + 30

/-- Represents the number of full turns Minutkin makes in the morning -/
def morning_turns : ℕ := 9

/-- Represents the number of full turns Minutkin makes at night -/
def night_turns : ℕ := 11

/-- Represents the total number of turns in a day -/
def total_turns : ℕ := morning_turns + night_turns

/-- Theorem stating that Minutkin goes to bed at 9:42 PM -/
theorem minutkin_bedtime :
  ∃ (bedtime : ℕ),
    bedtime = (minutes_per_day + morning_wind_time - (morning_turns * minutes_per_day / total_turns)) % minutes_per_day ∧
    bedtime = 21 * 60 + 42 :=
by sorry

end NUMINAMATH_CALUDE_minutkin_bedtime_l2003_200338


namespace NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l2003_200335

/-- The radius of a sphere tangent to a truncated cone -/
theorem sphere_radius_in_truncated_cone (r₁ r₂ : ℝ) (hr₁ : r₁ = 25) (hr₂ : r₂ = 5) :
  let h := Real.sqrt ((r₁ - r₂)^2 + (r₁ + r₂)^2)
  (h / 2 : ℝ) = 5 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l2003_200335


namespace NUMINAMATH_CALUDE_floor_sqrt_50_squared_l2003_200363

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_squared_l2003_200363


namespace NUMINAMATH_CALUDE_garden_dimensions_l2003_200337

/-- Represents a rectangular garden with a surrounding walkway -/
structure GardenWithWalkway where
  garden_width : ℝ
  walkway_width : ℝ

/-- The combined area of the garden and walkway is 432 square meters -/
axiom total_area (g : GardenWithWalkway) : 
  (g.garden_width + 2 * g.walkway_width) * (3 * g.garden_width + 2 * g.walkway_width) = 432

/-- The area of the walkway alone is 108 square meters -/
axiom walkway_area (g : GardenWithWalkway) : 
  (g.garden_width + 2 * g.walkway_width) * (3 * g.garden_width + 2 * g.walkway_width) - 
  3 * g.garden_width * g.garden_width = 108

/-- The dimensions of the garden are 6√3 and 18√3 meters -/
theorem garden_dimensions (g : GardenWithWalkway) : 
  g.garden_width = 6 * Real.sqrt 3 ∧ 3 * g.garden_width = 18 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_garden_dimensions_l2003_200337


namespace NUMINAMATH_CALUDE_rational_square_difference_l2003_200312

theorem rational_square_difference (x y : ℚ) (h : x^5 + y^5 = 2*x^2*y^2) :
  ∃ z : ℚ, 1 - x*y = z^2 := by
  sorry

end NUMINAMATH_CALUDE_rational_square_difference_l2003_200312


namespace NUMINAMATH_CALUDE_amount_calculation_l2003_200333

/-- Given three people with a total amount of money, where one person has a specific fraction of the others' total, calculate that person's amount. -/
theorem amount_calculation (total : ℚ) (p q r : ℚ) : 
  total = 5000 →
  p + q + r = total →
  r = (2/3) * (p + q) →
  r = 2000 := by
  sorry

end NUMINAMATH_CALUDE_amount_calculation_l2003_200333


namespace NUMINAMATH_CALUDE_quadratic_properties_l2003_200304

def f (x : ℝ) := (x - 1)^2 + 3

theorem quadratic_properties :
  (∀ x y : ℝ, x < y → f x > f y → f ((x + y) / 2) < f x) ∧ 
  (∀ x : ℝ, f (x + 1) = f (1 - x)) ∧
  (f 1 = 3 ∧ ∀ x : ℝ, f x ≥ 3) ∧
  f 0 ≠ 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2003_200304


namespace NUMINAMATH_CALUDE_distinct_cube_configurations_l2003_200379

-- Define the cube configuration
structure CubeConfig where
  white : Fin 8 → Fin 3
  blue : Fin 8 → Fin 3
  red : Fin 8 → Fin 2

-- Define the rotation group
def RotationGroup : Type := Unit

-- Define the action of the rotation group on cube configurations
def rotate : RotationGroup → CubeConfig → CubeConfig := sorry

-- Define the orbit of a cube configuration under rotations
def orbit (c : CubeConfig) : Set CubeConfig := sorry

-- Define the set of all valid cube configurations
def AllConfigs : Set CubeConfig := sorry

-- Count the number of distinct orbits
def countDistinctOrbits : ℕ := sorry

-- The main theorem
theorem distinct_cube_configurations :
  countDistinctOrbits = 25 := by sorry

end NUMINAMATH_CALUDE_distinct_cube_configurations_l2003_200379


namespace NUMINAMATH_CALUDE_alices_number_l2003_200345

def possible_numbers : List ℕ := [1080, 1440, 1800, 2160, 2520, 2880]

theorem alices_number (n : ℕ) :
  (40 ∣ n) →
  (72 ∣ n) →
  1000 < n →
  n < 3000 →
  n ∈ possible_numbers := by
sorry

end NUMINAMATH_CALUDE_alices_number_l2003_200345


namespace NUMINAMATH_CALUDE_different_terminal_sides_l2003_200350

-- Define a function to check if two angles have the same terminal side
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, β = α + k * 360

-- Theorem statement
theorem different_terminal_sides :
  ¬ same_terminal_side 1050 (-300) :=
by
  sorry

end NUMINAMATH_CALUDE_different_terminal_sides_l2003_200350


namespace NUMINAMATH_CALUDE_river_depth_problem_l2003_200389

/-- River depth problem -/
theorem river_depth_problem (may_depth june_depth july_depth : ℕ) : 
  may_depth = 5 →
  june_depth = may_depth + 10 →
  july_depth = june_depth * 3 →
  july_depth = 45 := by
  sorry

end NUMINAMATH_CALUDE_river_depth_problem_l2003_200389


namespace NUMINAMATH_CALUDE_five_in_range_of_f_l2003_200352

/-- The function f(x) = x^2 + bx - 3 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x - 3

/-- Theorem stating that 5 is always in the range of f(x) for all real b -/
theorem five_in_range_of_f (b : ℝ) : ∃ x : ℝ, f b x = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_in_range_of_f_l2003_200352


namespace NUMINAMATH_CALUDE_attendees_count_l2003_200371

/-- The number of people attending the family reunion --/
def attendees : ℕ := sorry

/-- The number of cans in each box of soda --/
def cans_per_box : ℕ := 10

/-- The cost of each box of soda in dollars --/
def cost_per_box : ℕ := 2

/-- The number of cans each person consumes --/
def cans_per_person : ℕ := 2

/-- The number of family members paying for the soda --/
def paying_family_members : ℕ := 6

/-- The amount each family member pays in dollars --/
def payment_per_member : ℕ := 4

/-- Theorem stating that the number of attendees is 60 --/
theorem attendees_count : attendees = 60 := by sorry

end NUMINAMATH_CALUDE_attendees_count_l2003_200371


namespace NUMINAMATH_CALUDE_quadratic_root_range_l2003_200396

theorem quadratic_root_range (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    a * x₁^2 - 2*(a+1)*x₁ + (a-1) = 0 ∧
    a * x₂^2 - 2*(a+1)*x₂ + (a-1) = 0 ∧
    x₁ > 2 ∧ x₂ < 2) →
  (0 < a ∧ a < 5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l2003_200396


namespace NUMINAMATH_CALUDE_thomas_total_bill_l2003_200370

-- Define the shipping rates
def flat_rate : ℝ := 5.00
def clothes_rate : ℝ := 0.20
def accessories_rate : ℝ := 0.10
def price_threshold : ℝ := 50.00

-- Define the prices of items
def shirt_price : ℝ := 12.00
def socks_price : ℝ := 5.00
def shorts_price : ℝ := 15.00
def swim_trunks_price : ℝ := 14.00
def hat_price : ℝ := 6.00
def sunglasses_price : ℝ := 30.00

-- Define the quantities of items
def shirt_quantity : ℕ := 3
def shorts_quantity : ℕ := 2

-- Calculate the total cost of clothes and accessories
def clothes_cost : ℝ := shirt_price * shirt_quantity + socks_price + shorts_price * shorts_quantity + swim_trunks_price
def accessories_cost : ℝ := hat_price + sunglasses_price

-- Calculate the shipping costs
def clothes_shipping : ℝ := clothes_rate * clothes_cost
def accessories_shipping : ℝ := accessories_rate * accessories_cost

-- Calculate the total bill
def total_bill : ℝ := clothes_cost + accessories_cost + clothes_shipping + accessories_shipping

-- Theorem to prove
theorem thomas_total_bill : total_bill = 141.60 := by sorry

end NUMINAMATH_CALUDE_thomas_total_bill_l2003_200370


namespace NUMINAMATH_CALUDE_factors_360_divisible_by_3_not_5_l2003_200326

def factors_divisible_by_3_not_5 (n : ℕ) : ℕ :=
  (Finset.filter (λ x => x ∣ n ∧ 3 ∣ x ∧ ¬(5 ∣ x)) (Finset.range (n + 1))).card

theorem factors_360_divisible_by_3_not_5 :
  factors_divisible_by_3_not_5 360 = 8 := by
  sorry

end NUMINAMATH_CALUDE_factors_360_divisible_by_3_not_5_l2003_200326


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l2003_200398

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  Real.sqrt ((c1.1 - c2.1)^2 + (c1.2 - c2.2)^2) = r1 + r2

/-- The equation of the first circle: x^2 + y^2 = 4 -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The equation of the second circle: x^2 + y^2 - 10x + 16 = 0 -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 10*x + 16 = 0

theorem circles_externally_tangent :
  externally_tangent (0, 0) (5, 0) 2 3 :=
by sorry

#check circles_externally_tangent

end NUMINAMATH_CALUDE_circles_externally_tangent_l2003_200398


namespace NUMINAMATH_CALUDE_food_bank_remaining_lyanna_food_bank_remaining_l2003_200319

/-- Given a food bank with donations over two weeks and a distribution in the third week,
    calculate the remaining food. -/
theorem food_bank_remaining (first_week : ℝ) (second_week_multiplier : ℝ) (distribution_percentage : ℝ) : ℝ :=
  let second_week := first_week * second_week_multiplier
  let total_donated := first_week + second_week
  let distributed := total_donated * distribution_percentage
  let remaining := total_donated - distributed
  remaining

/-- The amount of food remaining in Lyanna's food bank after two weeks of donations
    and a distribution in the third week. -/
theorem lyanna_food_bank_remaining : food_bank_remaining 40 2 0.7 = 36 := by
  sorry

end NUMINAMATH_CALUDE_food_bank_remaining_lyanna_food_bank_remaining_l2003_200319


namespace NUMINAMATH_CALUDE_principal_determination_l2003_200384

/-- Given a principal amount and an unknown interest rate, if increasing the
    interest rate by 6 percentage points results in Rs. 30 more interest over 1 year,
    then the principal must be Rs. 500. -/
theorem principal_determination (P R : ℝ) (h : P * (R + 6) / 100 - P * R / 100 = 30) :
  P = 500 := by
  sorry

end NUMINAMATH_CALUDE_principal_determination_l2003_200384


namespace NUMINAMATH_CALUDE_min_value_on_transformed_curve_l2003_200318

-- Define the original curve C
def curve_C (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line l
def line_l (x y t : ℝ) : Prop := x = 1 + t/2 ∧ y = 2 + (Real.sqrt 3)/2 * t

-- Define the transformation
def transform (x y x' y' : ℝ) : Prop := x' = 3*x ∧ y' = y

-- Define the transformed curve C'
def curve_C' (x y : ℝ) : Prop := x^2/9 + y^2 = 1

-- State the theorem
theorem min_value_on_transformed_curve :
  ∀ (x y : ℝ), curve_C' x y → (x + 2 * Real.sqrt 3 * y ≥ -Real.sqrt 21) :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_transformed_curve_l2003_200318
