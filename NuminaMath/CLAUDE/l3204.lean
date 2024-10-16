import Mathlib

namespace NUMINAMATH_CALUDE_halloween_jelly_beans_l3204_320438

/-- Given the conditions of a Halloween jelly bean distribution, 
    prove that the total number of children at the celebration is 40. -/
theorem halloween_jelly_beans 
  (initial_jelly_beans : ℕ)
  (remaining_jelly_beans : ℕ)
  (allowed_percentage : ℚ)
  (jelly_beans_per_child : ℕ)
  (h1 : initial_jelly_beans = 100)
  (h2 : remaining_jelly_beans = 36)
  (h3 : allowed_percentage = 4/5)
  (h4 : jelly_beans_per_child = 2) :
  (initial_jelly_beans - remaining_jelly_beans) / jelly_beans_per_child / allowed_percentage = 40 := by
  sorry

end NUMINAMATH_CALUDE_halloween_jelly_beans_l3204_320438


namespace NUMINAMATH_CALUDE_author_earnings_calculation_l3204_320489

def author_earnings (paperback_copies : Nat) (paperback_price : Real)
                    (hardcover_copies : Nat) (hardcover_price : Real)
                    (ebook_copies : Nat) (ebook_price : Real)
                    (audiobook_copies : Nat) (audiobook_price : Real) : Real :=
  let paperback_sales := paperback_copies * paperback_price
  let hardcover_sales := hardcover_copies * hardcover_price
  let ebook_sales := ebook_copies * ebook_price
  let audiobook_sales := audiobook_copies * audiobook_price
  0.06 * paperback_sales + 0.12 * hardcover_sales + 0.08 * ebook_sales + 0.10 * audiobook_sales

theorem author_earnings_calculation :
  author_earnings 32000 0.20 15000 0.40 10000 0.15 5000 0.50 = 1474 :=
by sorry

end NUMINAMATH_CALUDE_author_earnings_calculation_l3204_320489


namespace NUMINAMATH_CALUDE_not_necessarily_congruent_with_two_sides_one_angle_l3204_320491

/-- Triangle represented by three points in a 2D plane -/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- Predicate for triangle congruence -/
def IsCongruent (t1 t2 : Triangle) : Prop :=
  sorry

/-- Predicate for two sides and one angle being equal -/
def HasTwoSidesOneAngleEqual (t1 t2 : Triangle) : Prop :=
  sorry

/-- Theorem stating that triangles with two corresponding sides and one corresponding angle equal
    are not necessarily congruent -/
theorem not_necessarily_congruent_with_two_sides_one_angle :
  ∃ t1 t2 : Triangle, HasTwoSidesOneAngleEqual t1 t2 ∧ ¬IsCongruent t1 t2 :=
sorry

end NUMINAMATH_CALUDE_not_necessarily_congruent_with_two_sides_one_angle_l3204_320491


namespace NUMINAMATH_CALUDE_ben_picking_peas_l3204_320400

/-- Given that Ben can pick 56 sugar snap peas in 7 minutes, 
    prove that it takes 9 minutes to pick 72 sugar snap peas. -/
theorem ben_picking_peas (rate : ℚ) (h1 : rate = 56 / 7) : 72 / rate = 9 := by
  sorry

end NUMINAMATH_CALUDE_ben_picking_peas_l3204_320400


namespace NUMINAMATH_CALUDE_no_real_solutions_l3204_320404

theorem no_real_solutions :
  ∀ x : ℝ, ((x - 4*x + 15)^2 + 3)^2 + 1 ≠ -(abs x)^2 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3204_320404


namespace NUMINAMATH_CALUDE_f_odd_and_periodic_l3204_320481

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_odd_and_periodic (f : ℝ → ℝ) 
  (h1 : ∀ x, f (10 + x) = f (10 - x))
  (h2 : ∀ x, f (20 - x) = -f (20 + x)) :
  is_odd f ∧ is_periodic f 40 := by
  sorry

end NUMINAMATH_CALUDE_f_odd_and_periodic_l3204_320481


namespace NUMINAMATH_CALUDE_cube_root_of_27_l3204_320495

theorem cube_root_of_27 : ∃ x : ℝ, x^3 = 27 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_27_l3204_320495


namespace NUMINAMATH_CALUDE_water_needed_for_bread_dough_bakery_recipe_water_needed_l3204_320477

theorem water_needed_for_bread_dough (water_per_portion : ℕ) (flour_per_portion : ℕ) (total_flour : ℕ) : ℕ :=
  let portions := total_flour / flour_per_portion
  portions * water_per_portion

theorem bakery_recipe_water_needed : water_needed_for_bread_dough 75 300 900 = 225 := by
  sorry

end NUMINAMATH_CALUDE_water_needed_for_bread_dough_bakery_recipe_water_needed_l3204_320477


namespace NUMINAMATH_CALUDE_point_on_curve_l3204_320493

def curve (x y : ℝ) : Prop := x^2 - x*y + 2*y + 1 = 0

theorem point_on_curve :
  curve 0 (-1/2) ∧
  ¬ curve 0 0 ∧
  ¬ curve 1 (-1) ∧
  ¬ curve 1 1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_curve_l3204_320493


namespace NUMINAMATH_CALUDE_min_stamps_for_40_cents_l3204_320428

theorem min_stamps_for_40_cents :
  let stamp_values : List Nat := [5, 7]
  let target_value : Nat := 40
  ∃ (c f : Nat),
    c * stamp_values[0]! + f * stamp_values[1]! = target_value ∧
    ∀ (c' f' : Nat),
      c' * stamp_values[0]! + f' * stamp_values[1]! = target_value →
      c + f ≤ c' + f' ∧
    c + f = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_stamps_for_40_cents_l3204_320428


namespace NUMINAMATH_CALUDE_radical_simplification_l3204_320499

theorem radical_simplification (p : ℝ) (hp : p > 0) :
  Real.sqrt (12 * p) * Real.sqrt (7 * p^3) * Real.sqrt (15 * p^5) = 6 * p^4 * Real.sqrt (35 * p) := by
  sorry

end NUMINAMATH_CALUDE_radical_simplification_l3204_320499


namespace NUMINAMATH_CALUDE_minimum_of_x_squared_l3204_320472

theorem minimum_of_x_squared :
  ∃ (m : ℝ), m = 0 ∧ ∀ x : ℝ, x^2 ≥ m := by sorry

end NUMINAMATH_CALUDE_minimum_of_x_squared_l3204_320472


namespace NUMINAMATH_CALUDE_min_distance_line_circle_l3204_320440

/-- The minimum distance between a point on the line y = m + √3x and the circle (x-√3)² + (y-1)² = 2² is 1 if and only if m = 2 or m = -6. -/
theorem min_distance_line_circle (m : ℝ) : 
  (∃ (x y : ℝ), y = m + Real.sqrt 3 * x ∧ 
   (∀ (x' y' : ℝ), y' = m + Real.sqrt 3 * x' → 
     ((x' - Real.sqrt 3)^2 + (y' - 1)^2 ≥ ((x - Real.sqrt 3)^2 + (y - 1)^2))) ∧
   (x - Real.sqrt 3)^2 + (y - 1)^2 = 5) ↔ 
  (m = 2 ∨ m = -6) :=
sorry

end NUMINAMATH_CALUDE_min_distance_line_circle_l3204_320440


namespace NUMINAMATH_CALUDE_garrison_problem_l3204_320487

/-- Represents the initial number of men in the garrison -/
def initial_men : ℕ := 2000

/-- Represents the number of reinforcement men -/
def reinforcement : ℕ := 1600

/-- Represents the initial number of days the provisions would last -/
def initial_days : ℕ := 54

/-- Represents the number of days passed before reinforcement -/
def days_before_reinforcement : ℕ := 18

/-- Represents the number of days the provisions last after reinforcement -/
def remaining_days : ℕ := 20

theorem garrison_problem :
  initial_men * initial_days = 
  (initial_men + reinforcement) * remaining_days + 
  initial_men * days_before_reinforcement :=
by sorry

end NUMINAMATH_CALUDE_garrison_problem_l3204_320487


namespace NUMINAMATH_CALUDE_base_eight_subtraction_l3204_320421

/-- Represents a number in base 8 --/
def BaseEight : Type := ℕ

/-- Convert a base 8 number to decimal --/
def to_decimal (n : BaseEight) : ℕ := sorry

/-- Convert a decimal number to base 8 --/
def to_base_eight (n : ℕ) : BaseEight := sorry

/-- Subtraction in base 8 --/
def base_eight_sub (a b : BaseEight) : BaseEight := 
  to_base_eight (to_decimal a - to_decimal b)

theorem base_eight_subtraction : 
  base_eight_sub (to_base_eight 42) (to_base_eight 25) = to_base_eight 17 := by sorry

end NUMINAMATH_CALUDE_base_eight_subtraction_l3204_320421


namespace NUMINAMATH_CALUDE_largest_n_for_factorization_l3204_320446

/-- 
Theorem: The largest value of n for which 5x^2 + nx + 100 can be factored 
as the product of two linear factors with integer coefficients is 105.
-/
theorem largest_n_for_factorization : 
  (∃ (n : ℤ), ∀ (m : ℤ), 
    (∃ (a b : ℤ), ∀ (x : ℝ), 5 * x^2 + n * x + 100 = (5 * x + a) * (x + b)) ∧ 
    (∃ (a b : ℤ), ∀ (x : ℝ), 5 * x^2 + m * x + 100 = (5 * x + a) * (x + b) → m ≤ n)) ∧ 
  (∃ (a b : ℤ), ∀ (x : ℝ), 5 * x^2 + 105 * x + 100 = (5 * x + a) * (x + b)) :=
by sorry

#check largest_n_for_factorization

end NUMINAMATH_CALUDE_largest_n_for_factorization_l3204_320446


namespace NUMINAMATH_CALUDE_intersection_locus_is_circle_l3204_320407

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hand of the watch -/
structure WatchHand where
  axis : Point
  angularVelocity : ℝ
  initialAngle : ℝ

/-- Represents the watch configuration -/
structure Watch where
  secondHand : WatchHand
  stopwatchHand : WatchHand

/-- The locus of intersection points between extended watch hands -/
def intersectionLocus (w : Watch) (t : ℝ) : Point :=
  sorry

/-- Theorem stating that the intersection locus forms a circle -/
theorem intersection_locus_is_circle (w : Watch) : 
  ∃ (center : Point) (radius : ℝ), 
    ∀ t, ∃ θ, intersectionLocus w t = Point.mk (center.x + radius * Real.cos θ) (center.y + radius * Real.sin θ) :=
  sorry

end NUMINAMATH_CALUDE_intersection_locus_is_circle_l3204_320407


namespace NUMINAMATH_CALUDE_faye_necklace_sales_l3204_320454

theorem faye_necklace_sales (bead_necklaces : ℕ) (gem_necklaces : ℕ) 
  (necklace_price : ℕ) (total_earnings : ℕ) : 
  gem_necklaces = 7 → 
  necklace_price = 7 → 
  total_earnings = 70 → 
  total_earnings = necklace_price * (bead_necklaces + gem_necklaces) → 
  bead_necklaces = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_faye_necklace_sales_l3204_320454


namespace NUMINAMATH_CALUDE_students_without_A_l3204_320492

def class_size : ℕ := 35
def history_A : ℕ := 10
def math_A : ℕ := 15
def both_A : ℕ := 5

theorem students_without_A : 
  class_size - (history_A + math_A - both_A) = 15 := by sorry

end NUMINAMATH_CALUDE_students_without_A_l3204_320492


namespace NUMINAMATH_CALUDE_relation_implications_l3204_320486

-- Define the propositions
variable (A B C D : Prop)

-- Define the relationships between propositions
def sufficient_not_necessary (P Q : Prop) : Prop :=
  (P → Q) ∧ ¬(Q → P)

def necessary (P Q : Prop) : Prop :=
  Q → P

-- State the theorem
theorem relation_implications :
  sufficient_not_necessary A B →
  necessary B C →
  sufficient_not_necessary C D →
  (sufficient_not_necessary A C ∧ 
   ¬(sufficient_not_necessary D A ∨ necessary D A)) :=
by sorry

end NUMINAMATH_CALUDE_relation_implications_l3204_320486


namespace NUMINAMATH_CALUDE_product_eleven_cubed_sum_l3204_320494

theorem product_eleven_cubed_sum (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 11^3 →
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 133 := by
sorry

end NUMINAMATH_CALUDE_product_eleven_cubed_sum_l3204_320494


namespace NUMINAMATH_CALUDE_train_length_l3204_320426

/-- Given a train that crosses a platform and a signal pole, prove its length. -/
theorem train_length (platform_length : ℝ) (platform_time : ℝ) (pole_time : ℝ)
  (h1 : platform_length = 550)
  (h2 : platform_time = 51)
  (h3 : pole_time = 18) :
  ∃ (train_length : ℝ), train_length = 300 ∧ 
    train_length + platform_length = (train_length / pole_time) * platform_time :=
by sorry

end NUMINAMATH_CALUDE_train_length_l3204_320426


namespace NUMINAMATH_CALUDE_age_difference_l3204_320402

theorem age_difference (sachin_age rahul_age : ℕ) : 
  sachin_age = 63 →
  sachin_age * 9 = rahul_age * 7 →
  rahul_age - sachin_age = 18 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l3204_320402


namespace NUMINAMATH_CALUDE_football_players_percentage_l3204_320409

theorem football_players_percentage
  (total_population : ℕ)
  (football_likers_ratio : ℚ)
  (players_in_sample : ℕ)
  (sample_size : ℕ)
  (h1 : football_likers_ratio = 24 / 60)
  (h2 : players_in_sample = 50)
  (h3 : sample_size = 250) :
  (players_in_sample : ℚ) / ((football_likers_ratio * sample_size) : ℚ) = 1/2 :=
sorry

end NUMINAMATH_CALUDE_football_players_percentage_l3204_320409


namespace NUMINAMATH_CALUDE_even_function_sum_l3204_320405

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + b * x

-- State the theorem
theorem even_function_sum (a b : ℝ) :
  (∀ x ∈ Set.Icc (a - 1) (2 * a), f a b x = f a b (-x)) →
  a + b = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_even_function_sum_l3204_320405


namespace NUMINAMATH_CALUDE_fraction_simplification_l3204_320422

theorem fraction_simplification : (3 : ℚ) / (2 - 3 / 4) = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3204_320422


namespace NUMINAMATH_CALUDE_max_value_of_f_l3204_320482

/-- Given that f(x) = 2sin(x) - cos(x) reaches its maximum value when x = θ, prove that sin(θ) = 2√5/5 -/
theorem max_value_of_f (θ : ℝ) : 
  (∀ x, 2 * Real.sin x - Real.cos x ≤ 2 * Real.sin θ - Real.cos θ) →
  Real.sin θ = 2 * Real.sqrt 5 / 5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3204_320482


namespace NUMINAMATH_CALUDE_keiths_cds_l3204_320429

/-- Calculates the number of CDs Keith wanted to buy based on his total spending and the price per CD -/
theorem keiths_cds (speakers_cost cd_player_cost tires_cost total_spent cd_price : ℝ) :
  speakers_cost = 136.01 →
  cd_player_cost = 139.38 →
  tires_cost = 112.46 →
  total_spent = 387.85 →
  cd_price = 6.16 →
  speakers_cost + cd_player_cost + tires_cost = total_spent →
  ⌊total_spent / cd_price⌋ = 62 :=
by sorry

end NUMINAMATH_CALUDE_keiths_cds_l3204_320429


namespace NUMINAMATH_CALUDE_quadratic_factoring_l3204_320488

/-- A quadratic equation is an equation of the form ax² + bx + c = 0, where a ≠ 0 -/
structure QuadraticEquation (α : Type*) [Field α] where
  a : α
  b : α
  c : α
  a_nonzero : a ≠ 0

/-- A factored form of a quadratic equation is a product of linear factors -/
structure FactoredForm (α : Type*) [Field α] where
  factor1 : α → α
  factor2 : α → α

/-- 
Given a quadratic equation that can be factored, 
it can be expressed as a multiplication of factors.
-/
theorem quadratic_factoring 
  {α : Type*} [Field α]
  (eq : QuadraticEquation α)
  (h_factorable : ∃ (f : FactoredForm α), 
    ∀ x, eq.a * x^2 + eq.b * x + eq.c = f.factor1 x * f.factor2 x) :
  ∃ (f : FactoredForm α), 
    ∀ x, eq.a * x^2 + eq.b * x + eq.c = f.factor1 x * f.factor2 x :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factoring_l3204_320488


namespace NUMINAMATH_CALUDE_smallest_X_value_l3204_320430

/-- A function that checks if a natural number consists only of 0s and 1s in its decimal representation -/
def onlyZerosAndOnes (n : ℕ) : Prop := sorry

/-- The smallest positive integer T that satisfies the given conditions -/
def T : ℕ := sorry

theorem smallest_X_value :
  T > 0 ∧
  onlyZerosAndOnes T ∧
  T % 15 = 0 ∧
  (∀ t : ℕ, t > 0 → onlyZerosAndOnes t → t % 15 = 0 → t ≥ T) →
  T / 15 = 74 := by sorry

end NUMINAMATH_CALUDE_smallest_X_value_l3204_320430


namespace NUMINAMATH_CALUDE_student_contribution_l3204_320408

theorem student_contribution
  (total_contribution : ℕ)
  (class_funds : ℕ)
  (num_students : ℕ)
  (h1 : total_contribution = 90)
  (h2 : class_funds = 14)
  (h3 : num_students = 19)
  : (total_contribution - class_funds) / num_students = 4 := by
  sorry

end NUMINAMATH_CALUDE_student_contribution_l3204_320408


namespace NUMINAMATH_CALUDE_clock_sum_after_duration_l3204_320410

/-- Represents time on a 12-hour digital clock -/
structure Time12 where
  hour : Nat
  minute : Nat
  second : Nat
  deriving Repr

/-- Adds a duration to a given time and returns the new time on a 12-hour clock -/
def addDuration (initial : Time12) (hours minutes seconds : Nat) : Time12 :=
  sorry

/-- Calculates the sum of hour, minute, and second digits for a given time -/
def sumDigits (t : Time12) : Nat :=
  t.hour + t.minute + t.second

theorem clock_sum_after_duration :
  let initial := Time12.mk 3 0 0  -- 3:00:00 PM
  let final := addDuration initial 137 58 59
  sumDigits final = 125 := by
  sorry

end NUMINAMATH_CALUDE_clock_sum_after_duration_l3204_320410


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3204_320415

theorem triangle_perimeter : ∀ (a b c : ℝ),
  a = 3 ∧ b = 5 ∧ c^2 - 3*c = c - 3 ∧ 
  a + b > c ∧ b + c > a ∧ c + a > b →
  a + b + c = 11 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3204_320415


namespace NUMINAMATH_CALUDE_max_value_of_derived_function_l3204_320436

/-- Given a function f(x) = a * sin(x) + b with max value 1 and min value -7,
    prove that the max value of b * sin²(x) - a * cos²(x) is either 4 or -3 -/
theorem max_value_of_derived_function 
  (a b : ℝ) 
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = a * Real.sin x + b)
  (h_max : ∀ x, f x ≤ 1)
  (h_min : ∀ x, f x ≥ -7)
  : (∃ x, b * Real.sin x ^ 2 - a * Real.cos x ^ 2 = 4) ∨ 
    (∃ x, b * Real.sin x ^ 2 - a * Real.cos x ^ 2 = -3) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_derived_function_l3204_320436


namespace NUMINAMATH_CALUDE_ellipse_m_value_l3204_320476

/-- An ellipse with equation x^2 + my^2 = 1, where m is a positive real number -/
structure Ellipse (m : ℝ) : Type :=
  (eq : ∀ (x y : ℝ), x^2 + m*y^2 = 1)

/-- The foci of the ellipse are on the y-axis -/
def foci_on_y_axis (e : Ellipse m) : Prop :=
  ∃ (c : ℝ), c^2 = 1/m - 1

/-- The length of the major axis is twice the length of the minor axis -/
def major_axis_twice_minor (e : Ellipse m) : Prop :=
  2 * Real.sqrt 1 = Real.sqrt (1/m)

/-- The theorem stating that m = 1/4 for the given conditions -/
theorem ellipse_m_value (m : ℝ) (e : Ellipse m)
  (h1 : m > 0)
  (h2 : foci_on_y_axis e)
  (h3 : major_axis_twice_minor e) :
  m = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l3204_320476


namespace NUMINAMATH_CALUDE_pen_pencil_difference_l3204_320414

theorem pen_pencil_difference (ratio_pens : ℕ) (ratio_pencils : ℕ) (total_pencils : ℕ) : 
  ratio_pens = 5 → ratio_pencils = 6 → total_pencils = 54 → 
  total_pencils - (total_pencils / ratio_pencils * ratio_pens) = 9 := by
sorry

end NUMINAMATH_CALUDE_pen_pencil_difference_l3204_320414


namespace NUMINAMATH_CALUDE_original_paint_intensity_l3204_320448

/-- Given a paint mixture where a fraction of 1.5 times the original amount is replaced
    with a 25% solution of red paint, and the resulting mixture has a red paint
    intensity of 30%, prove that the original intensity of the red paint was 15%. -/
theorem original_paint_intensity
  (replacement_fraction : ℝ)
  (replacement_solution_intensity : ℝ)
  (final_mixture_intensity : ℝ)
  (h1 : replacement_fraction = 1.5)
  (h2 : replacement_solution_intensity = 0.25)
  (h3 : final_mixture_intensity = 0.30)
  : ∃ (original_intensity : ℝ),
    original_intensity * (1 - replacement_fraction) +
    replacement_solution_intensity * replacement_fraction = final_mixture_intensity ∧
    original_intensity = 0.15 :=
by sorry

end NUMINAMATH_CALUDE_original_paint_intensity_l3204_320448


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3204_320435

def A : Set ℝ := {-2, -1, 2, 3}
def B : Set ℝ := {x : ℝ | x^2 - x - 6 < 0}

theorem intersection_of_A_and_B : A ∩ B = {-1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3204_320435


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l3204_320463

theorem rectangular_plot_breadth (length width area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 432 →
  width = 12 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l3204_320463


namespace NUMINAMATH_CALUDE_greatest_integer_fraction_l3204_320450

theorem greatest_integer_fraction (x : ℤ) : 
  (∃ k : ℤ, (x^2 + 4*x + 9) / (x - 4) = k) → x ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_fraction_l3204_320450


namespace NUMINAMATH_CALUDE_train_length_l3204_320412

/-- Proves that a train traveling at 45 km/hr crossing a 215-meter bridge in 30 seconds has a length of 160 meters -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_speed = 45 * 1000 / 3600 →
  bridge_length = 215 →
  crossing_time = 30 →
  train_speed * crossing_time - bridge_length = 160 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3204_320412


namespace NUMINAMATH_CALUDE_shortest_distance_on_specific_cone_l3204_320437

/-- Represents a right circular cone --/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a point on the surface of a cone --/
structure ConePoint where
  distanceFromVertex : ℝ

/-- Calculate the shortest distance between two points on the surface of a cone --/
def shortestDistanceOnCone (c : Cone) (p1 p2 : ConePoint) : ℝ :=
  sorry

theorem shortest_distance_on_specific_cone :
  let c : Cone := { baseRadius := 500, height := 150 * Real.sqrt 7 }
  let p1 : ConePoint := { distanceFromVertex := 100 }
  let p2 : ConePoint := { distanceFromVertex := 300 * Real.sqrt 2 }
  shortestDistanceOnCone c p1 p2 = Real.sqrt (460000 + 60000 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_on_specific_cone_l3204_320437


namespace NUMINAMATH_CALUDE_set_operations_and_subset_l3204_320498

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 10}
def B : Set ℝ := {x | 2 < x ∧ x ≤ 7}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2*a + 6}

-- State the theorem
theorem set_operations_and_subset :
  (A ∩ B = {x : ℝ | 3 ≤ x ∧ x ≤ 7}) ∧
  (A ∪ B = {x : ℝ | 2 < x ∧ x < 10}) ∧
  (∀ a : ℝ, A ⊆ C a ↔ 2 ≤ a ∧ a < 3) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_and_subset_l3204_320498


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l3204_320431

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15 → 
  ∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 → 
  (x : ℤ) + y ≤ (a : ℤ) + b :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l3204_320431


namespace NUMINAMATH_CALUDE_T_increasing_T_binary_ones_M_properties_l3204_320465

/-- A sequence of positive integers with exactly 9 ones in binary representation -/
def T : ℕ → ℕ := sorry

/-- The 1500th term in the sequence T -/
def M : ℕ := T 1500

/-- The remainder when M is divided by 500 -/
def r : ℕ := M % 500

theorem T_increasing (n : ℕ) : T n < T (n + 1) := sorry

theorem T_binary_ones (n : ℕ) : 
  (Nat.digits 2 (T n)).count 1 = 9 := sorry

theorem M_properties : 
  ∃ (M : ℕ), 
    M = T 1500 ∧ 
    (∀ n < 1500, T n < M) ∧
    (Nat.digits 2 M).count 1 = 9 ∧
    M % 500 = r := by sorry

end NUMINAMATH_CALUDE_T_increasing_T_binary_ones_M_properties_l3204_320465


namespace NUMINAMATH_CALUDE_four_spheres_block_light_l3204_320462

-- Define a point in 3D space
def Point := ℝ × ℝ × ℝ

-- Define a sphere in 3D space
structure Sphere where
  center : Point
  radius : ℝ
  radius_pos : radius > 0

-- Define the property of a sphere being opaque
def isOpaque (s : Sphere) : Prop := sorry

-- Define the property of two spheres being non-intersecting
def nonIntersecting (s1 s2 : Sphere) : Prop := sorry

-- Define the property of a set of spheres blocking light from a point source
def blocksLight (source : Point) (spheres : List Sphere) : Prop := sorry

-- The main theorem
theorem four_spheres_block_light :
  ∃ (s1 s2 s3 s4 : Sphere) (source : Point),
    isOpaque s1 ∧ isOpaque s2 ∧ isOpaque s3 ∧ isOpaque s4 ∧
    nonIntersecting s1 s2 ∧ nonIntersecting s1 s3 ∧ nonIntersecting s1 s4 ∧
    nonIntersecting s2 s3 ∧ nonIntersecting s2 s4 ∧ nonIntersecting s3 s4 ∧
    blocksLight source [s1, s2, s3, s4] := by
  sorry

end NUMINAMATH_CALUDE_four_spheres_block_light_l3204_320462


namespace NUMINAMATH_CALUDE_third_month_sale_l3204_320423

def average_sale : ℕ := 5500
def number_of_months : ℕ := 6
def sales : List ℕ := [5435, 5927, 6230, 5562, 3991]

theorem third_month_sale :
  (average_sale * number_of_months - sales.sum) = 5855 := by
  sorry

end NUMINAMATH_CALUDE_third_month_sale_l3204_320423


namespace NUMINAMATH_CALUDE_shaded_area_is_42_l3204_320464

/-- Square ABCD with shaded regions -/
structure ShadedSquare where
  /-- Side length of the square ABCD -/
  side_length : ℕ
  /-- Side length of the first small square -/
  small_square_side : ℕ
  /-- Width of the second shaded region -/
  second_width : ℕ
  /-- Height of the second shaded region -/
  second_height : ℕ
  /-- Width of the third shaded region -/
  third_width : ℕ
  /-- Height of the third shaded region -/
  third_height : ℕ
  /-- Assumption that the side length is 7 -/
  h_side_length : side_length = 7
  /-- Assumption that the small square side length is 1 -/
  h_small_square : small_square_side = 1
  /-- Assumption about the second shaded region dimensions -/
  h_second : second_width = 2 ∧ second_height = 4
  /-- Assumption about the third shaded region dimensions -/
  h_third : third_width = 3 ∧ third_height = 6

/-- The shaded area of the square ABCD is 42 square units -/
theorem shaded_area_is_42 (sq : ShadedSquare) : 
  sq.small_square_side ^ 2 +
  (sq.small_square_side + sq.second_width) * (sq.small_square_side + sq.second_height) - sq.small_square_side ^ 2 +
  (sq.small_square_side + sq.second_width + sq.third_width) * (sq.small_square_side + sq.second_height + sq.third_height) -
  (sq.small_square_side + sq.second_width) * (sq.small_square_side + sq.second_height) = 42 :=
sorry

end NUMINAMATH_CALUDE_shaded_area_is_42_l3204_320464


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l3204_320473

open Real

noncomputable def f (x : ℝ) : ℝ := x - 2 * sin x

theorem f_increasing_on_interval :
  ∀ x ∈ Set.Ioo (π/3) (5*π/3), 
    x ∈ Set.Ioo 0 (2*π) → 
    ∀ y ∈ Set.Ioo (π/3) (5*π/3), 
      x < y → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l3204_320473


namespace NUMINAMATH_CALUDE_cube_sum_equality_l3204_320403

theorem cube_sum_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / a - 1 / b - 1 / (a + b) = 0) : 
  (b / a)^3 + (a / b)^3 = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_equality_l3204_320403


namespace NUMINAMATH_CALUDE_matching_probability_theorem_l3204_320459

/-- Represents the distribution of shoe pairs by color -/
structure ShoeDistribution where
  black : Nat
  brown : Nat
  gray : Nat
  red : Nat

/-- Calculates the total number of individual shoes -/
def totalShoes (d : ShoeDistribution) : Nat :=
  2 * (d.black + d.brown + d.gray + d.red)

/-- Calculates the probability of selecting a matching pair -/
def matchingProbability (d : ShoeDistribution) : Rat :=
  let total := totalShoes d
  let numerator := 
    d.black * (d.black - 1) + 
    d.brown * (d.brown - 1) + 
    d.gray * (d.gray - 1) + 
    d.red * (d.red - 1)
  (numerator : Rat) / (total * (total - 1))

/-- John's shoe distribution -/
def johnsShoes : ShoeDistribution :=
  { black := 8, brown := 4, gray := 3, red := 1 }

theorem matching_probability_theorem : 
  matchingProbability johnsShoes = 45 / 248 := by
  sorry

end NUMINAMATH_CALUDE_matching_probability_theorem_l3204_320459


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l3204_320479

theorem sum_of_two_numbers (A B : ℝ) : 
  A - B = 8 → (A + B) / 4 = 6 → A = 16 → A + B = 24 := by sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l3204_320479


namespace NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l3204_320433

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (dims : BoxDimensions) : ℕ :=
  dims.length * dims.width * dims.height

/-- Represents the carton dimensions -/
def cartonDims : BoxDimensions :=
  { length := 25, width := 35, height := 50 }

/-- Represents the soap box dimensions -/
def soapBoxDims : BoxDimensions :=
  { length := 8, width := 7, height := 6 }

/-- Theorem stating the maximum number of soap boxes that can fit in the carton -/
theorem max_soap_boxes_in_carton :
  (boxVolume cartonDims) / (boxVolume soapBoxDims) = 130 := by
  sorry

end NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l3204_320433


namespace NUMINAMATH_CALUDE_garden_flowers_count_l3204_320490

/-- Represents a rectangular garden with a rose planted in it. -/
structure Garden where
  columns : ℕ
  rows : ℕ
  rose_col_left : ℕ
  rose_col_right : ℕ
  rose_row_front : ℕ
  rose_row_back : ℕ

/-- The total number of flowers in the garden. -/
def total_flowers (g : Garden) : ℕ := g.columns * g.rows

/-- Theorem stating the total number of flowers in the specific garden configuration. -/
theorem garden_flowers_count :
  ∀ g : Garden,
  g.rose_col_left = 9 →
  g.rose_col_right = 13 →
  g.rose_row_front = 7 →
  g.rose_row_back = 16 →
  g.columns = g.rose_col_left + g.rose_col_right - 1 →
  g.rows = g.rose_row_front + g.rose_row_back - 1 →
  total_flowers g = 462 := by
  sorry

#check garden_flowers_count

end NUMINAMATH_CALUDE_garden_flowers_count_l3204_320490


namespace NUMINAMATH_CALUDE_bellas_dancer_friends_l3204_320467

theorem bellas_dancer_friends (total_roses : ℕ) (parent_roses : ℕ) (roses_per_friend : ℕ) 
  (h1 : total_roses = 44)
  (h2 : parent_roses = 2 * 12)
  (h3 : roses_per_friend = 2) :
  (total_roses - parent_roses) / roses_per_friend = 10 := by
  sorry

end NUMINAMATH_CALUDE_bellas_dancer_friends_l3204_320467


namespace NUMINAMATH_CALUDE_multiply_5915581_7907_l3204_320483

theorem multiply_5915581_7907 : 5915581 * 7907 = 46757653387 := by
  sorry

end NUMINAMATH_CALUDE_multiply_5915581_7907_l3204_320483


namespace NUMINAMATH_CALUDE_water_left_over_l3204_320458

theorem water_left_over (total_water : ℕ) (num_players : ℕ) (water_per_player : ℕ) (water_spilled : ℕ) : 
  total_water = 8 →
  num_players = 30 →
  water_per_player = 200 →
  water_spilled = 250 →
  total_water * 1000 - (num_players * water_per_player + water_spilled) = 1750 :=
by
  sorry

end NUMINAMATH_CALUDE_water_left_over_l3204_320458


namespace NUMINAMATH_CALUDE_sum_of_solutions_squared_equation_l3204_320418

theorem sum_of_solutions_squared_equation : 
  ∃ (x₁ x₂ : ℝ), (x₁ + 6)^2 = 49 ∧ (x₂ + 6)^2 = 49 ∧ x₁ + x₂ = -12 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_squared_equation_l3204_320418


namespace NUMINAMATH_CALUDE_coefficient_a_is_zero_l3204_320434

-- Define the quadratic equation
def quadratic_equation (a b c p : ℝ) (x : ℝ) : Prop :=
  a * x^2 + b * x + c + p = 0

-- Define the condition that all roots are real and positive
def all_roots_real_positive (a b c : ℝ) : Prop :=
  ∀ p > 0, ∀ x, quadratic_equation a b c p x → x > 0

-- Theorem statement
theorem coefficient_a_is_zero (a b c : ℝ) :
  all_roots_real_positive a b c → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_a_is_zero_l3204_320434


namespace NUMINAMATH_CALUDE_x_less_than_zero_sufficient_not_necessary_for_x_not_equal_three_l3204_320470

theorem x_less_than_zero_sufficient_not_necessary_for_x_not_equal_three :
  (∀ x : ℝ, x < 0 → x ≠ 3) ∧
  (∃ x : ℝ, x ≠ 3 ∧ x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_x_less_than_zero_sufficient_not_necessary_for_x_not_equal_three_l3204_320470


namespace NUMINAMATH_CALUDE_willie_stickers_l3204_320474

/-- The number of stickers Willie ends up with after giving some away -/
def stickers_left (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem: Willie ends up with 29 stickers -/
theorem willie_stickers : stickers_left 36 7 = 29 := by
  sorry

end NUMINAMATH_CALUDE_willie_stickers_l3204_320474


namespace NUMINAMATH_CALUDE_total_monthly_cost_l3204_320475

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ := d.length * d.width * d.height

/-- Represents the storage details -/
structure StorageDetails where
  boxDim : BoxDimensions
  totalVolume : ℝ
  costPerBox : ℝ

/-- Theorem stating that the total monthly cost for record storage is $480 -/
theorem total_monthly_cost (s : StorageDetails)
  (h1 : s.boxDim = ⟨15, 12, 10⟩)
  (h2 : s.totalVolume = 1080000)
  (h3 : s.costPerBox = 0.8) :
  (s.totalVolume / boxVolume s.boxDim) * s.costPerBox = 480 := by
  sorry


end NUMINAMATH_CALUDE_total_monthly_cost_l3204_320475


namespace NUMINAMATH_CALUDE_olaf_total_cars_l3204_320420

/-- The number of toy cars in Olaf's collection --/
def total_cars (initial : ℕ) (grandpa uncle dad mum auntie : ℕ) : ℕ :=
  initial + grandpa + uncle + dad + mum + auntie

/-- The conditions of Olaf's toy car collection problem --/
def olaf_problem (initial grandpa uncle dad mum auntie : ℕ) : Prop :=
  initial = 150 ∧
  grandpa = 2 * uncle ∧
  dad = 10 ∧
  mum = dad + 5 ∧
  auntie = uncle + 1 ∧
  auntie = 6

/-- Theorem stating that Olaf's total number of cars is 196 --/
theorem olaf_total_cars :
  ∀ initial grandpa uncle dad mum auntie : ℕ,
  olaf_problem initial grandpa uncle dad mum auntie →
  total_cars initial grandpa uncle dad mum auntie = 196 :=
by
  sorry


end NUMINAMATH_CALUDE_olaf_total_cars_l3204_320420


namespace NUMINAMATH_CALUDE_modulo_seven_equivalence_l3204_320447

theorem modulo_seven_equivalence : 
  ∃! n : ℤ, 0 ≤ n ∧ n < 7 ∧ -1234 ≡ n [ZMOD 7] ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulo_seven_equivalence_l3204_320447


namespace NUMINAMATH_CALUDE_batch_size_l3204_320417

/-- The number of parts A can complete in one day -/
def a_rate : ℚ := 1 / 10

/-- The number of parts B can complete in one day -/
def b_rate : ℚ := 1 / 15

/-- The number of additional parts A completes compared to B in one day -/
def additional_parts : ℕ := 50

/-- The total number of parts in the batch -/
def total_parts : ℕ := 1500

theorem batch_size :
  (a_rate - b_rate) * total_parts = additional_parts := by sorry

end NUMINAMATH_CALUDE_batch_size_l3204_320417


namespace NUMINAMATH_CALUDE_miranda_stuffs_six_pillows_l3204_320419

/-- The number of pillows Miranda can stuff given the conditions -/
def miranda_pillows : ℕ :=
  let feathers_per_pound : ℕ := 300
  let goose_feathers : ℕ := 3600
  let pounds_per_pillow : ℕ := 2
  let total_pounds : ℕ := goose_feathers / feathers_per_pound
  total_pounds / pounds_per_pillow

/-- Proof that Miranda can stuff 6 pillows -/
theorem miranda_stuffs_six_pillows : miranda_pillows = 6 := by
  sorry

end NUMINAMATH_CALUDE_miranda_stuffs_six_pillows_l3204_320419


namespace NUMINAMATH_CALUDE_bike_ride_distance_l3204_320413

/-- Calculates the total distance traveled given a constant speed and time -/
def total_distance (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

theorem bike_ride_distance :
  let rate := 1.5 / 10  -- miles per minute
  let time := 40        -- minutes
  total_distance rate time = 6 := by
  sorry

end NUMINAMATH_CALUDE_bike_ride_distance_l3204_320413


namespace NUMINAMATH_CALUDE_min_value_quadratic_min_value_achieved_l3204_320442

theorem min_value_quadratic (x y : ℝ) : 
  y = 5 * x^2 - 8 * x + 20 → x ≥ 1 → y ≥ 13 := by
  sorry

theorem min_value_achieved (x : ℝ) : 
  x ≥ 1 → ∃ y : ℝ, y = 5 * x^2 - 8 * x + 20 ∧ y = 13 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_min_value_achieved_l3204_320442


namespace NUMINAMATH_CALUDE_base_conversion_problem_l3204_320456

theorem base_conversion_problem : 
  (∃ (S : Finset ℕ), 
    (∀ b ∈ S, b ≥ 2 ∧ b^3 ≤ 250 ∧ 250 < b^4) ∧ 
    (∀ b : ℕ, b ≥ 2 → b^3 ≤ 250 → 250 < b^4 → b ∈ S) ∧
    Finset.card S = 2) := by sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l3204_320456


namespace NUMINAMATH_CALUDE_product_of_solutions_l3204_320411

theorem product_of_solutions : ∃ (y₁ y₂ : ℝ), 
  (abs y₁ = 3 * (abs y₁ - 2)) ∧ 
  (abs y₂ = 3 * (abs y₂ - 2)) ∧ 
  (y₁ ≠ y₂) ∧ 
  (y₁ * y₂ = -9) := by
  sorry

end NUMINAMATH_CALUDE_product_of_solutions_l3204_320411


namespace NUMINAMATH_CALUDE_construct_incenter_l3204_320457

-- Define the basic constructions available
class Constructible (α : Type*) where
  draw_line : α → α → Prop
  draw_circle : α → α → Prop
  mark_intersection : Prop

-- Define a triangle
structure Triangle (α : Type*) where
  A : α
  B : α
  C : α

-- Define the incenter
def Incenter (α : Type*) (t : Triangle α) : α := sorry

-- Theorem statement
theorem construct_incenter 
  {α : Type*} [Constructible α] (t : Triangle α) :
  ∃ (I : α), I = Incenter α t := by sorry

end NUMINAMATH_CALUDE_construct_incenter_l3204_320457


namespace NUMINAMATH_CALUDE_jovana_shell_weight_l3204_320439

/-- Proves that the total weight of shells in Jovana's bucket is approximately 11.29 pounds -/
theorem jovana_shell_weight (initial_weight : ℝ) (large_shell_weight : ℝ) (additional_weight : ℝ) 
  (conversion_rate : ℝ) (h1 : initial_weight = 5.25) (h2 : large_shell_weight = 700) 
  (h3 : additional_weight = 4.5) (h4 : conversion_rate = 453.592) : 
  ∃ (total_weight : ℝ), abs (total_weight - 11.29) < 0.01 ∧ 
  total_weight = initial_weight + (large_shell_weight / conversion_rate) + additional_weight :=
sorry

end NUMINAMATH_CALUDE_jovana_shell_weight_l3204_320439


namespace NUMINAMATH_CALUDE_condition_A_sufficient_not_necessary_for_B_l3204_320480

theorem condition_A_sufficient_not_necessary_for_B :
  (∀ a b : ℝ, a > b ∧ b > 0 → 1 / a < 1 / b) ∧
  (∃ a b : ℝ, 1 / a < 1 / b ∧ ¬(a > b ∧ b > 0)) := by
  sorry

end NUMINAMATH_CALUDE_condition_A_sufficient_not_necessary_for_B_l3204_320480


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3204_320401

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_parallel : (1 : ℝ) / 2 = (x - 2) / (-6 * y)) :
  (3 / x + 1 / y) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3204_320401


namespace NUMINAMATH_CALUDE_vector_collinearity_l3204_320471

theorem vector_collinearity (m : ℝ) : 
  let a : Fin 2 → ℝ := ![2, 3]
  let b : Fin 2 → ℝ := ![-1, 2]
  (∃ (k : ℝ), k ≠ 0 ∧ (m • a + 4 • b) = k • (a - 2 • b)) →
  m = -2 := by
sorry

end NUMINAMATH_CALUDE_vector_collinearity_l3204_320471


namespace NUMINAMATH_CALUDE_parallelogram_with_half_circle_angles_is_rectangle_l3204_320443

-- Define a parallelogram
structure Parallelogram where
  angles : Fin 4 → ℝ
  is_parallelogram : True

-- Define the property of all angles being equal to half of the total degrees in a circle
def all_angles_half_circle (p : Parallelogram) : Prop :=
  ∀ i : Fin 4, p.angles i = 180

-- Define a rectangle
def is_rectangle (p : Parallelogram) : Prop :=
  ∀ i : Fin 4, p.angles i = 90

-- Theorem statement
theorem parallelogram_with_half_circle_angles_is_rectangle (p : Parallelogram) :
  all_angles_half_circle p → is_rectangle p := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_with_half_circle_angles_is_rectangle_l3204_320443


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l3204_320425

theorem quadratic_equation_result (x : ℝ) (h : x^2 - x - 1 = 0) :
  1995 + 2*x - x^3 = 1994 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l3204_320425


namespace NUMINAMATH_CALUDE_parabola_vertex_l3204_320478

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := -3 * x^2 + 6 * x + 1

/-- The x-coordinate of the vertex -/
def vertex_x : ℝ := 1

/-- The y-coordinate of the vertex -/
def vertex_y : ℝ := 4

/-- Theorem: The vertex of the parabola y = -3x^2 + 6x + 1 is (1, 4) -/
theorem parabola_vertex :
  (∀ x : ℝ, parabola x ≤ vertex_y) ∧
  parabola vertex_x = vertex_y :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3204_320478


namespace NUMINAMATH_CALUDE_linear_function_m_value_l3204_320453

/-- Given a linear function y = (m^2 + 2m)x + m^2 + m - 1 + (2m - 3), prove that m = 1 -/
theorem linear_function_m_value (m : ℝ) : 
  (∃ k b, ∀ x, (m^2 + 2*m)*x + (m^2 + m - 1 + (2*m - 3)) = k*x + b) → 
  (m^2 + 2*m ≠ 0) → 
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_linear_function_m_value_l3204_320453


namespace NUMINAMATH_CALUDE_extended_box_with_hemispheres_volume_l3204_320451

/-- The volume of a region formed by extending a rectangular parallelepiped and adding hemispheres at its vertices -/
theorem extended_box_with_hemispheres_volume 
  (l w h : ℝ) 
  (hl : l = 5) 
  (hw : w = 6) 
  (hh : h = 7) 
  (extension : ℝ) 
  (hemisphere_radius : ℝ) 
  (he : extension = 2) 
  (hr : hemisphere_radius = 2) : 
  (l + 2 * extension) * (w + 2 * extension) * (h + 2 * extension) + 
  8 * ((2 / 3) * π * hemisphere_radius^3) = 
  990 + (128 / 3) * π :=
sorry

end NUMINAMATH_CALUDE_extended_box_with_hemispheres_volume_l3204_320451


namespace NUMINAMATH_CALUDE_absolute_value_reciprocal_intersection_l3204_320484

/-- The equation |x + a| = 1/x has exactly two solutions if and only if a = -2 -/
theorem absolute_value_reciprocal_intersection (a : ℝ) : 
  (∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ |x₁ + a| = 1/x₁ ∧ |x₂ + a| = 1/x₂) ↔ a = -2 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_reciprocal_intersection_l3204_320484


namespace NUMINAMATH_CALUDE_original_equals_scientific_l3204_320497

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ significand ∧ significand < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 4370000

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation :=
  { significand := 4.37
    exponent := 6
    is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_form.significand * (10 : ℝ) ^ scientific_form.exponent :=
by sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l3204_320497


namespace NUMINAMATH_CALUDE_garden_snake_length_l3204_320496

-- Define the lengths of the snakes
def boa_length : Float := 1.428571429
def garden_snake_ratio : Float := 7.0

-- Theorem statement
theorem garden_snake_length : 
  boa_length * garden_snake_ratio = 10.0 := by sorry

end NUMINAMATH_CALUDE_garden_snake_length_l3204_320496


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l3204_320424

theorem fraction_zero_implies_x_equals_one (x : ℝ) :
  (x - 1) / (x + 1) = 0 → x = 1 := by
sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l3204_320424


namespace NUMINAMATH_CALUDE_inequality_B_is_linear_one_var_inequality_A_is_not_linear_one_var_inequality_C_is_not_linear_one_var_only_B_is_linear_one_var_l3204_320427

-- Define the inequalities
def inequality_A (x : ℝ) := 3 * x^2 > 45 - 9 * x
def inequality_B (x : ℝ) := 3 * x - 2 < 4
def inequality_C (x : ℝ) := 1 / x < 2
def inequality_D (x y : ℝ) := 4 * x - 3 < 2 * y - 7

-- Define what it means for an inequality to be linear with one variable
def is_linear_one_var (f : ℝ → Prop) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x ↔ a * x < b ∨ a * x > b

-- Theorem stating that inequality_B is linear with one variable
theorem inequality_B_is_linear_one_var :
  is_linear_one_var inequality_B :=
sorry

-- Theorems stating that the other inequalities are not linear with one variable
theorem inequality_A_is_not_linear_one_var :
  ¬ is_linear_one_var inequality_A :=
sorry

theorem inequality_C_is_not_linear_one_var :
  ¬ is_linear_one_var inequality_C :=
sorry

-- Note: inequality_D is not included as it has two variables

-- Main theorem
theorem only_B_is_linear_one_var :
  is_linear_one_var inequality_B ∧
  ¬ is_linear_one_var inequality_A ∧
  ¬ is_linear_one_var inequality_C :=
sorry

end NUMINAMATH_CALUDE_inequality_B_is_linear_one_var_inequality_A_is_not_linear_one_var_inequality_C_is_not_linear_one_var_only_B_is_linear_one_var_l3204_320427


namespace NUMINAMATH_CALUDE_novel_reading_difference_novel_reading_difference_proof_l3204_320416

theorem novel_reading_difference : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun jordan alexandre camille maxime =>
    jordan = 130 ∧
    alexandre = jordan / 10 ∧
    camille = 2 * alexandre ∧
    maxime = (jordan + alexandre + camille) / 2 - 5 →
    jordan - maxime = 51

-- Proof
theorem novel_reading_difference_proof :
  ∃ (jordan alexandre camille maxime : ℕ),
    novel_reading_difference jordan alexandre camille maxime :=
by
  sorry

end NUMINAMATH_CALUDE_novel_reading_difference_novel_reading_difference_proof_l3204_320416


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3204_320461

theorem sufficient_not_necessary (x : ℝ) : 
  (x = 0 → x^2 - 2*x = 0) ∧ (∃ y : ℝ, y ≠ 0 ∧ y^2 - 2*y = 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3204_320461


namespace NUMINAMATH_CALUDE_ribbon_cuts_l3204_320406

/-- The number of cuts needed to divide ribbon rolls into smaller pieces -/
def cuts_needed (num_rolls : ℕ) (roll_length : ℕ) (piece_length : ℕ) : ℕ :=
  num_rolls * ((roll_length / piece_length) - 1)

/-- Theorem: The number of cuts needed to divide 5 rolls of 50-meter ribbon into 2-meter pieces is 120 -/
theorem ribbon_cuts : cuts_needed 5 50 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_cuts_l3204_320406


namespace NUMINAMATH_CALUDE_tan_and_trig_identity_l3204_320449

open Real

theorem tan_and_trig_identity (α : ℝ) (h : tan (α + π/4) = 1/3) : 
  tan α = -1/2 ∧ 
  2 * sin α ^ 2 - sin (π - α) * sin (π/2 - α) + sin (3*π/2 + α) ^ 2 = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_and_trig_identity_l3204_320449


namespace NUMINAMATH_CALUDE_toms_age_ratio_l3204_320452

/-- Tom's age problem -/
theorem toms_age_ratio (T N : ℝ) : T > 0 → N > 0 → 
  (T = T - 4*N + T - 4*N + T - 4*N + T - 4*N) → -- Sum of children's ages
  (T - N = 3 * (T - 4*N)) →                     -- Relation N years ago
  T / N = 11 / 2 := by
sorry

end NUMINAMATH_CALUDE_toms_age_ratio_l3204_320452


namespace NUMINAMATH_CALUDE_cake_cutting_theorem_l3204_320460

/-- Represents a rectangular cake -/
structure Cake where
  length : ℕ
  width : ℕ
  pieces : ℕ

/-- The maximum number of pieces obtainable with one straight cut -/
def max_pieces_one_cut (c : Cake) : ℕ := sorry

/-- The minimum number of cuts required to ensure each piece is cut -/
def min_cuts_all_pieces (c : Cake) : ℕ := sorry

/-- Theorem for the cake cutting problem -/
theorem cake_cutting_theorem (c : Cake) 
  (h1 : c.length = 5 ∧ c.width = 2) 
  (h2 : c.pieces = 10) : 
  max_pieces_one_cut c = 16 ∧ min_cuts_all_pieces c = 2 := by sorry

end NUMINAMATH_CALUDE_cake_cutting_theorem_l3204_320460


namespace NUMINAMATH_CALUDE_chord_length_l3204_320444

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the focus of the ellipse
def focus : ℝ × ℝ := (1, 0)

-- Define a chord through the focus and perpendicular to the major axis
def chord (y : ℝ) : Prop := ellipse 1 y

-- Theorem statement
theorem chord_length : 
  ∃ y₁ y₂ : ℝ, 
    chord y₁ ∧ 
    chord y₂ ∧ 
    y₁ ≠ y₂ ∧ 
    |y₁ - y₂| = 3 :=
sorry

end NUMINAMATH_CALUDE_chord_length_l3204_320444


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l3204_320455

/-- Represents a conic section --/
inductive ConicSection
  | Parabola
  | Circle
  | Ellipse
  | Hyperbola
  | Point
  | Line
  | TwoLines
  | Empty

/-- Determines the type of conic section from the coefficients of the general equation --/
def determineConicSection (a b c d e f : ℝ) : ConicSection :=
  sorry

/-- The equation x^2 - 36y^2 - 12x + 36 = 0 represents a hyperbola --/
theorem equation_represents_hyperbola :
  determineConicSection 1 (-36) 0 (-12) 0 36 = ConicSection.Hyperbola :=
sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l3204_320455


namespace NUMINAMATH_CALUDE_iguana_feed_cost_l3204_320432

/-- The monthly cost to feed each iguana, given the number of pets, 
    the cost to feed geckos and snakes, and the total annual cost for all pets. -/
theorem iguana_feed_cost 
  (num_geckos num_iguanas num_snakes : ℕ)
  (gecko_cost snake_cost : ℚ)
  (total_annual_cost : ℚ)
  (h1 : num_geckos = 3)
  (h2 : num_iguanas = 2)
  (h3 : num_snakes = 4)
  (h4 : gecko_cost = 15)
  (h5 : snake_cost = 10)
  (h6 : total_annual_cost = 1140)
  : ∃ (iguana_cost : ℚ), 
    iguana_cost = 5 ∧
    (num_geckos : ℚ) * gecko_cost + 
    (num_iguanas : ℚ) * iguana_cost + 
    (num_snakes : ℚ) * snake_cost = 
    total_annual_cost / 12 :=
sorry

end NUMINAMATH_CALUDE_iguana_feed_cost_l3204_320432


namespace NUMINAMATH_CALUDE_count_ordered_pairs_l3204_320445

def prime_factorization : List (Nat × Nat) := [(2, 2), (3, 2), (7, 2)]

def n : Nat := 1764

theorem count_ordered_pairs : 
  (Finset.filter (fun p : Nat × Nat => p.1 * p.2 = n) (Finset.product (Finset.range (n + 1)) (Finset.range (n + 1)))).card = 27 := by
  sorry

end NUMINAMATH_CALUDE_count_ordered_pairs_l3204_320445


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_l3204_320485

theorem least_three_digit_multiple : ∃ n : ℕ, 
  (100 ≤ n ∧ n ≤ 999) ∧ 
  (3 ∣ n) ∧ (4 ∣ n) ∧ (7 ∣ n) ∧ 
  (∀ m : ℕ, (100 ≤ m ∧ m < n) → ¬((3 ∣ m) ∧ (4 ∣ m) ∧ (7 ∣ m))) ∧
  n = 168 := by
  sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_l3204_320485


namespace NUMINAMATH_CALUDE_bus_speeds_l3204_320469

theorem bus_speeds (distance : ℝ) (time_difference : ℝ) (speed_difference : ℝ)
  (h1 : distance = 48)
  (h2 : time_difference = 1/6)
  (h3 : speed_difference = 4) :
  ∃ (speed1 speed2 : ℝ),
    speed1 = 36 ∧
    speed2 = 32 ∧
    distance / speed1 + time_difference = distance / speed2 ∧
    speed1 = speed2 + speed_difference :=
by sorry

end NUMINAMATH_CALUDE_bus_speeds_l3204_320469


namespace NUMINAMATH_CALUDE_poem_line_addition_l3204_320466

theorem poem_line_addition (initial_lines : ℕ) (months : ℕ) (final_lines : ℕ) 
  (h1 : initial_lines = 24)
  (h2 : months = 22)
  (h3 : final_lines = 90) :
  (final_lines - initial_lines) / months = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_poem_line_addition_l3204_320466


namespace NUMINAMATH_CALUDE_emilys_coin_collection_value_l3204_320468

/-- Proves that given the conditions of Emily's coin collection, the total value is $128 -/
theorem emilys_coin_collection_value :
  ∀ (total_coins : ℕ) 
    (first_type_count : ℕ) 
    (first_type_total_value : ℝ) 
    (second_type_count : ℕ),
  total_coins = 20 →
  first_type_count = 8 →
  first_type_total_value = 32 →
  second_type_count = total_coins - first_type_count →
  (second_type_count * (first_type_total_value / first_type_count) * 2 + first_type_total_value = 128) :=
by
  sorry


end NUMINAMATH_CALUDE_emilys_coin_collection_value_l3204_320468


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3204_320441

theorem complex_equation_solution (z : ℂ) : (3 - I) * z = 1 - I → z = 2/5 - 1/5 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3204_320441
