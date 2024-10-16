import Mathlib

namespace NUMINAMATH_CALUDE_no_x3_term_condition_l2758_275860

/-- The coefficient of x^3 in the expansion of ((x+a)^2(2x-1/x)^5) -/
def coeff_x3 (a : ℝ) : ℝ := 80 - 80 * a^2

/-- The theorem stating that the value of 'a' for which the expansion of 
    ((x+a)^2(2x-1/x)^5) does not contain the x^3 term is ±1 -/
theorem no_x3_term_condition (a : ℝ) : 
  coeff_x3 a = 0 ↔ a = 1 ∨ a = -1 := by
  sorry

#check no_x3_term_condition

end NUMINAMATH_CALUDE_no_x3_term_condition_l2758_275860


namespace NUMINAMATH_CALUDE_sum_and_count_theorem_l2758_275899

def sum_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_and_count_theorem :
  let x := sum_integers 40 60
  let y := count_even_integers 40 60
  x + y = 1061 := by sorry

end NUMINAMATH_CALUDE_sum_and_count_theorem_l2758_275899


namespace NUMINAMATH_CALUDE_janes_number_l2758_275851

/-- A function that returns the number of positive divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- A function that returns the sum of positive divisors of a natural number -/
def sum_divisors (n : ℕ) : ℕ := sorry

/-- A function that returns the sum of prime divisors of a natural number -/
def sum_prime_divisors (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is uniquely determined by its sum of divisors -/
def is_unique_by_sum_divisors (n : ℕ) : Prop := 
  ∀ m : ℕ, sum_divisors m = sum_divisors n → m = n

/-- A function that checks if a number is uniquely determined by its sum of prime divisors -/
def is_unique_by_sum_prime_divisors (n : ℕ) : Prop := 
  ∀ m : ℕ, sum_prime_divisors m = sum_prime_divisors n → m = n

theorem janes_number : 
  ∃! n : ℕ, 
    500 < n ∧ 
    n < 1000 ∧ 
    num_divisors n = 20 ∧ 
    ¬ is_unique_by_sum_divisors n ∧ 
    ¬ is_unique_by_sum_prime_divisors n ∧ 
    n = 880 := by sorry

end NUMINAMATH_CALUDE_janes_number_l2758_275851


namespace NUMINAMATH_CALUDE_cubic_equation_coefficient_l2758_275823

theorem cubic_equation_coefficient (a b : ℝ) : 
  (∀ x : ℝ, a * x^3 + b * x^2 + 1 = (a * x - 1) * (x^2 - x - 1)) → 
  b = -2 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_coefficient_l2758_275823


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l2758_275847

/-- Two vectors in R² -/
def Vector2 := ℝ × ℝ

/-- Dot product of two vectors in R² -/
def dot_product (v w : Vector2) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Perpendicularity of two vectors in R² -/
def perpendicular (v w : Vector2) : Prop :=
  dot_product v w = 0

theorem perpendicular_vectors_m_value :
  ∀ m : ℝ,
  let a : Vector2 := (1, 2)
  let b : Vector2 := (m, 1)
  perpendicular a b → m = -2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l2758_275847


namespace NUMINAMATH_CALUDE_pistachio_count_l2758_275803

theorem pistachio_count (total : ℝ) 
  (h1 : 0.95 * total * 0.75 = 57) : total = 80 := by
  sorry

end NUMINAMATH_CALUDE_pistachio_count_l2758_275803


namespace NUMINAMATH_CALUDE_price_adjustment_l2758_275811

theorem price_adjustment (original_price : ℝ) (original_price_positive : 0 < original_price) : 
  let reduced_price := 0.8 * original_price
  let final_price := reduced_price * 1.375
  final_price = 1.1 * original_price :=
by sorry

end NUMINAMATH_CALUDE_price_adjustment_l2758_275811


namespace NUMINAMATH_CALUDE_sequence_ratio_theorem_l2758_275877

theorem sequence_ratio_theorem (d : ℝ) (q : ℚ) :
  d ≠ 0 →
  q > 0 →
  let a : ℕ → ℝ := λ n => d * n
  let b : ℕ → ℝ := λ n => d^2 * q^(n-1)
  ∃ k : ℕ+, (a 1)^2 + (a 2)^2 + (a 3)^2 = k * ((b 1) + (b 2) + (b 3)) →
  q = 2 ∨ q = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_ratio_theorem_l2758_275877


namespace NUMINAMATH_CALUDE_isosceles_triangle_30_angle_diff_l2758_275819

-- Define an isosceles triangle with one angle of 30 degrees
structure IsoscelesTriangle30 where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180
  isosceles : (angles 0 = angles 1) ∨ (angles 1 = angles 2) ∨ (angles 0 = angles 2)
  has_30 : angles 0 = 30 ∨ angles 1 = 30 ∨ angles 2 = 30

-- State the theorem
theorem isosceles_triangle_30_angle_diff 
  (t : IsoscelesTriangle30) : 
  ∃ (i j : Fin 3), i ≠ j ∧ (t.angles i - t.angles j = 90 ∨ t.angles i - t.angles j = 0) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_30_angle_diff_l2758_275819


namespace NUMINAMATH_CALUDE_robot_gloves_rings_arrangements_l2758_275864

/-- Represents the number of arms of the robot -/
def num_arms : ℕ := 6

/-- Represents the total number of items (gloves and rings) -/
def total_items : ℕ := 2 * num_arms

/-- Represents the number of valid arrangements for putting on gloves and rings -/
def valid_arrangements : ℕ := (Nat.factorial total_items) / (2^num_arms)

/-- Theorem stating the number of valid arrangements for the robot to put on gloves and rings -/
theorem robot_gloves_rings_arrangements :
  valid_arrangements = (Nat.factorial total_items) / (2^num_arms) :=
by sorry

end NUMINAMATH_CALUDE_robot_gloves_rings_arrangements_l2758_275864


namespace NUMINAMATH_CALUDE_seashells_per_day_l2758_275836

/-- 
Given a 5-day beach trip where 35 seashells were found in total, 
and assuming an equal number of seashells were found each day, 
prove that the number of seashells found per day is 7.
-/
theorem seashells_per_day 
  (days : ℕ) 
  (total_seashells : ℕ) 
  (seashells_per_day : ℕ) 
  (h1 : days = 5) 
  (h2 : total_seashells = 35) 
  (h3 : seashells_per_day * days = total_seashells) : 
  seashells_per_day = 7 := by
sorry

end NUMINAMATH_CALUDE_seashells_per_day_l2758_275836


namespace NUMINAMATH_CALUDE_find_n_l2758_275846

theorem find_n (x y n : ℝ) (h1 : (7 * x + 2 * y) / (x - n * y) = 23) (h2 : x / (2 * y) = 3 / 2) : n = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l2758_275846


namespace NUMINAMATH_CALUDE_village_population_percentage_l2758_275871

theorem village_population_percentage : 
  let part : ℕ := 23040
  let total : ℕ := 38400
  let percentage : ℚ := (part : ℚ) / (total : ℚ) * 100
  percentage = 60 := by sorry

end NUMINAMATH_CALUDE_village_population_percentage_l2758_275871


namespace NUMINAMATH_CALUDE_democrat_ratio_l2758_275874

theorem democrat_ratio (total_participants : ℕ) 
  (female_participants male_participants : ℕ)
  (female_democrats male_democrats : ℕ) :
  total_participants = 720 →
  female_participants + male_participants = total_participants →
  female_democrats = female_participants / 2 →
  male_democrats = male_participants / 4 →
  female_democrats = 120 →
  (female_democrats + male_democrats) * 3 = total_participants :=
by sorry

end NUMINAMATH_CALUDE_democrat_ratio_l2758_275874


namespace NUMINAMATH_CALUDE_x_minus_y_value_l2758_275880

theorem x_minus_y_value (x y : ℤ) 
  (sum_eq : x + y = 290) 
  (y_eq : y = 245) : 
  x - y = -200 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l2758_275880


namespace NUMINAMATH_CALUDE_angle_460_in_second_quadrant_l2758_275885

/-- An angle is in the second quadrant if it's between 90° and 180° in its standard position -/
def in_second_quadrant (angle : ℝ) : Prop :=
  let standard_angle := angle % 360
  90 < standard_angle ∧ standard_angle ≤ 180

/-- 460° is in the second quadrant -/
theorem angle_460_in_second_quadrant : in_second_quadrant 460 := by
  sorry

end NUMINAMATH_CALUDE_angle_460_in_second_quadrant_l2758_275885


namespace NUMINAMATH_CALUDE_area_of_circle_with_diameter_6_l2758_275832

-- Define the circle
def circle_diameter : ℝ := 6

-- Theorem statement
theorem area_of_circle_with_diameter_6 :
  (π * (circle_diameter / 2)^2) = 9 * π := by sorry

end NUMINAMATH_CALUDE_area_of_circle_with_diameter_6_l2758_275832


namespace NUMINAMATH_CALUDE_factor_x_squared_minus_81_l2758_275813

theorem factor_x_squared_minus_81 (x : ℝ) : x^2 - 81 = (x - 9) * (x + 9) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_squared_minus_81_l2758_275813


namespace NUMINAMATH_CALUDE_nut_mixture_price_l2758_275862

/-- Calculates the selling price per pound of a nut mixture --/
theorem nut_mixture_price
  (cashew_price : ℝ)
  (brazil_price : ℝ)
  (total_weight : ℝ)
  (cashew_weight : ℝ)
  (h1 : cashew_price = 6.75)
  (h2 : brazil_price = 5.00)
  (h3 : total_weight = 50)
  (h4 : cashew_weight = 20)
  : (cashew_weight * cashew_price + (total_weight - cashew_weight) * brazil_price) / total_weight = 5.70 := by
  sorry

end NUMINAMATH_CALUDE_nut_mixture_price_l2758_275862


namespace NUMINAMATH_CALUDE_symmetric_point_x_axis_l2758_275861

/-- Given two points P and Q in a 2D plane, where Q is symmetric to P with respect to the x-axis,
    this theorem proves that the x-coordinate of Q is the same as P, and the y-coordinate of Q
    is the negative of P's y-coordinate. -/
theorem symmetric_point_x_axis 
  (P Q : ℝ × ℝ) 
  (h_symmetric : Q.1 = P.1 ∧ Q.2 = -P.2) 
  (h_P : P = (-3, 1)) : 
  Q = (-3, -1) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_x_axis_l2758_275861


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2758_275835

theorem necessary_but_not_sufficient :
  (∀ x y : ℝ, x > 1 ∧ y > 1 → x + y > 2) ∧
  (∃ x y : ℝ, x + y > 2 ∧ ¬(x > 1 ∧ y > 1)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2758_275835


namespace NUMINAMATH_CALUDE_craftsman_production_theorem_l2758_275802

/-- The number of parts manufactured by a master craftsman during a shift -/
def parts_manufactured : ℕ → ℕ → ℕ → ℕ
  | initial_rate, rate_increase, additional_parts =>
    initial_rate + additional_parts

/-- The time needed to manufacture parts at a given rate -/
def time_needed : ℕ → ℕ → ℚ
  | parts, rate => (parts : ℚ) / (rate : ℚ)

theorem craftsman_production_theorem 
  (initial_rate : ℕ) 
  (rate_increase : ℕ) 
  (additional_parts : ℕ) :
  initial_rate = 35 →
  rate_increase = 15 →
  time_needed additional_parts initial_rate - 
    time_needed additional_parts (initial_rate + rate_increase) = (3 : ℚ) / 2 →
  parts_manufactured initial_rate rate_increase additional_parts = 210 :=
by sorry

end NUMINAMATH_CALUDE_craftsman_production_theorem_l2758_275802


namespace NUMINAMATH_CALUDE_negative_one_to_2002_is_smallest_positive_integer_l2758_275867

theorem negative_one_to_2002_is_smallest_positive_integer :
  (-1 : ℤ) ^ 2002 = 1 ∧ ∀ n : ℤ, n > 0 → n ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_negative_one_to_2002_is_smallest_positive_integer_l2758_275867


namespace NUMINAMATH_CALUDE_triangle_similarity_side_length_l2758_275858

/-- Two triangles are similar -/
def similar (t1 t2 : Set (ℝ × ℝ)) : Prop := sorry

/-- The length of a side in a triangle -/
def side_length (t : Set (ℝ × ℝ)) (s : ℝ × ℝ → ℝ × ℝ → Prop) : ℝ := sorry

theorem triangle_similarity_side_length 
  (PQR XYZ : Set (ℝ × ℝ))
  (PQ QR XY YZ : ℝ × ℝ → ℝ × ℝ → Prop) :
  similar PQR XYZ →
  side_length PQR PQ = 12 →
  side_length PQR QR = 7 →
  side_length XYZ XY = 4 →
  abs (side_length XYZ YZ - 2.3) ≤ 0.05 := by
  sorry

end NUMINAMATH_CALUDE_triangle_similarity_side_length_l2758_275858


namespace NUMINAMATH_CALUDE_lincoln_county_houses_l2758_275841

def original_houses : ℕ := 20817
def new_houses : ℕ := 97741

theorem lincoln_county_houses : original_houses + new_houses = 118558 := by
  sorry

end NUMINAMATH_CALUDE_lincoln_county_houses_l2758_275841


namespace NUMINAMATH_CALUDE_antons_number_is_729_l2758_275820

/-- Represents a three-digit number -/
def ThreeDigitNumber := { n : ℕ // 100 ≤ n ∧ n < 1000 }

/-- Returns true if exactly one digit matches between two three-digit numbers -/
def exactlyOneDigitMatches (a b : ThreeDigitNumber) : Prop :=
  (a.val / 100 = b.val / 100 ∧ a.val % 100 ≠ b.val % 100) ∨
  (a.val % 100 / 10 = b.val % 100 / 10 ∧ a.val / 100 ≠ b.val / 100 ∧ a.val % 10 ≠ b.val % 10) ∨
  (a.val % 10 = b.val % 10 ∧ a.val / 10 ≠ b.val / 10)

theorem antons_number_is_729 (x : ThreeDigitNumber) 
  (h1 : exactlyOneDigitMatches x ⟨109, by norm_num⟩)
  (h2 : exactlyOneDigitMatches x ⟨704, by norm_num⟩)
  (h3 : exactlyOneDigitMatches x ⟨124, by norm_num⟩) :
  x = ⟨729, by norm_num⟩ :=
by sorry

end NUMINAMATH_CALUDE_antons_number_is_729_l2758_275820


namespace NUMINAMATH_CALUDE_contractor_problem_l2758_275807

/-- Represents the efficiency of a worker --/
structure WorkerEfficiency where
  value : ℝ
  pos : value > 0

/-- Represents a group of workers with the same efficiency --/
structure WorkerGroup where
  count : ℕ
  efficiency : WorkerEfficiency

/-- Calculates the total work done by a group of workers in a day --/
def dailyWork (group : WorkerGroup) : ℝ :=
  group.count * group.efficiency.value

/-- Calculates the total work done by multiple groups of workers in a day --/
def totalDailyWork (groups : List WorkerGroup) : ℝ :=
  groups.map dailyWork |>.sum

/-- The contractor problem --/
theorem contractor_problem 
  (initialGroups : List WorkerGroup)
  (initialDays : ℕ)
  (totalDays : ℕ)
  (firedLessEfficient : ℕ)
  (firedMoreEfficient : ℕ)
  (h_initial_groups : initialGroups = [
    { count := 15, efficiency := { value := 1, pos := by sorry } },
    { count := 10, efficiency := { value := 1.5, pos := by sorry } }
  ])
  (h_initial_days : initialDays = 40)
  (h_total_days : totalDays = 150)
  (h_fired_less : firedLessEfficient = 4)
  (h_fired_more : firedMoreEfficient = 3)
  (h_one_third_complete : totalDailyWork initialGroups * initialDays = (1/3) * (totalDailyWork initialGroups * totalDays))
  : ∃ (remainingDays : ℕ), remainingDays = 112 ∧ 
    (totalDailyWork initialGroups * totalDays) = 
    (totalDailyWork initialGroups * initialDays + 
     totalDailyWork [
       { count := 15 - firedLessEfficient, efficiency := { value := 1, pos := by sorry } },
       { count := 10 - firedMoreEfficient, efficiency := { value := 1.5, pos := by sorry } }
     ] * remainingDays) := by
  sorry


end NUMINAMATH_CALUDE_contractor_problem_l2758_275807


namespace NUMINAMATH_CALUDE_triangle_area_bounds_l2758_275898

def triangle_area (s : ℝ) : ℝ := (s + 2)^(4/3)

theorem triangle_area_bounds :
  ∀ s : ℝ, (2^(1/2) * 3^(1/4) ≤ s ∧ s ≤ 3^(2/3) * 2^(1/3)) ↔
    (12 ≤ triangle_area s ∧ triangle_area s ≤ 72) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_bounds_l2758_275898


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_zero_l2758_275839

theorem quadratic_solution_difference_squared_zero :
  ∀ a b : ℝ,
  (5 * a^2 - 30 * a + 45 = 0) →
  (5 * b^2 - 30 * b + 45 = 0) →
  (a - b)^2 = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_zero_l2758_275839


namespace NUMINAMATH_CALUDE_total_counts_for_week_l2758_275833

/-- Represents the number of times Carla counts each item on a given day -/
structure DailyCounts where
  tiles : Nat
  books : Nat
  chairs : Nat

/-- The week's counting activities -/
def week : List DailyCounts := [
  ⟨1, 1, 0⟩,  -- Monday
  ⟨2, 3, 0⟩,  -- Tuesday
  ⟨0, 0, 4⟩,  -- Wednesday
  ⟨3, 0, 2⟩,  -- Thursday
  ⟨1, 2, 3⟩   -- Friday
]

/-- Calculates the total number of counts for a day -/
def totalCountsForDay (day : DailyCounts) : Nat :=
  day.tiles + day.books + day.chairs

/-- Theorem stating that the total number of counts for the week is 22 -/
theorem total_counts_for_week : (week.map totalCountsForDay).sum = 22 := by
  sorry

end NUMINAMATH_CALUDE_total_counts_for_week_l2758_275833


namespace NUMINAMATH_CALUDE_eight_last_to_appear_l2758_275863

def tribonacci : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | 2 => 4
  | n + 3 => tribonacci n + tribonacci (n + 1) + tribonacci (n + 2)

def lastDigit (n : ℕ) : ℕ := n % 10

def digitAppears (d : ℕ) (n : ℕ) : Prop :=
  ∃ k, k ≤ n ∧ lastDigit (tribonacci k) = d

theorem eight_last_to_appear :
  ∃ N, ∀ n, n ≥ N → 
    (∀ d, d ≠ 8 → digitAppears d n) ∧
    ¬(digitAppears 8 n) ∧
    digitAppears 8 (n + 1) := by sorry

end NUMINAMATH_CALUDE_eight_last_to_appear_l2758_275863


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2758_275812

/-- Given an ellipse and a hyperbola with specific properties, prove the equation of the hyperbola -/
theorem hyperbola_equation (x y : ℝ) :
  -- Given ellipse equation
  (x^2 / 144 + y^2 / 169 = 1) →
  -- Hyperbola passes through (0, 2)
  (∃ (a b : ℝ), y^2 / a^2 - x^2 / b^2 = 1 ∧ 2^2 / a^2 - 0^2 / b^2 = 1) →
  -- Hyperbola shares a common focus with the ellipse
  (∃ (c : ℝ), c^2 = 169 - 144 ∧ c^2 = a^2 + b^2) →
  -- Prove the equation of the hyperbola
  (y^2 / 4 - x^2 / 21 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2758_275812


namespace NUMINAMATH_CALUDE_bankers_gain_example_l2758_275849

/-- Calculates the banker's gain given present worth, interest rate, and time period. -/
def bankers_gain (present_worth : ℝ) (interest_rate : ℝ) (years : ℕ) : ℝ :=
  present_worth * (1 + interest_rate) ^ years - present_worth

/-- Theorem stating that the banker's gain is 126 given the specific conditions. -/
theorem bankers_gain_example : bankers_gain 600 0.1 2 = 126 := by
  sorry

end NUMINAMATH_CALUDE_bankers_gain_example_l2758_275849


namespace NUMINAMATH_CALUDE_coloring_scheme_exists_l2758_275855

-- Define the color type
inductive Color
| White
| Red
| Black

-- Define the coloring function type
def ColoringFunction := ℤ × ℤ → Color

-- Statement of the theorem
theorem coloring_scheme_exists : ∃ (f : ColoringFunction),
  (∀ c : Color, ∃ (S : Set ℤ), Set.Infinite S ∧ 
    ∀ y ∈ S, Set.Infinite {x : ℤ | f (x, y) = c}) ∧
  (∀ (x₁ y₁ x₂ y₂ x₃ y₃ : ℤ),
    f (x₁, y₁) = Color.White →
    f (x₂, y₂) = Color.Black →
    f (x₃, y₃) = Color.Red →
    f (x₁ + x₂ - x₃, y₁ + y₂ - y₃) = Color.Red) :=
by sorry


end NUMINAMATH_CALUDE_coloring_scheme_exists_l2758_275855


namespace NUMINAMATH_CALUDE_prove_initial_stock_l2758_275878

-- Define the total number of books sold
def books_sold : ℕ := 272

-- Define the percentage of books sold as a rational number
def percentage_sold : ℚ := 19.42857142857143 / 100

-- Define the initial stock of books
def initial_stock : ℕ := 1400

-- Theorem statement
theorem prove_initial_stock : 
  (books_sold : ℚ) / initial_stock = percentage_sold :=
by sorry

end NUMINAMATH_CALUDE_prove_initial_stock_l2758_275878


namespace NUMINAMATH_CALUDE_product_of_distinct_nonzero_reals_l2758_275845

theorem product_of_distinct_nonzero_reals (x y : ℝ) : 
  x ≠ 0 → y ≠ 0 → x ≠ y → x + 3 / x = y + 3 / y → x * y = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_distinct_nonzero_reals_l2758_275845


namespace NUMINAMATH_CALUDE_monotonicity_and_slope_conditions_l2758_275854

-- Define the function f
def f (a b x : ℝ) : ℝ := -x^3 + x^2 + a*x + b

-- Define the derivative of f
def f' (a x : ℝ) : ℝ := -3*x^2 + 2*x + a

theorem monotonicity_and_slope_conditions (a b : ℝ) :
  -- Part 1: Monotonicity when a = 3
  (∀ x ∈ Set.Ioo (-1 : ℝ) 3, (f' 3 x > 0)) ∧
  (∀ x ∈ Set.Iic (-1 : ℝ), (f' 3 x < 0)) ∧
  (∀ x ∈ Set.Ici 3, (f' 3 x < 0)) ∧
  -- Part 2: Condition on a based on slope
  ((∀ x : ℝ, f' a x < 2*a^2) → (a > 1 ∨ a < -1/2)) :=
sorry

end NUMINAMATH_CALUDE_monotonicity_and_slope_conditions_l2758_275854


namespace NUMINAMATH_CALUDE_balls_in_boxes_theorem_l2758_275830

def number_of_ways (n m k : ℕ) : ℕ :=
  Nat.choose n m * Nat.choose m k * Nat.factorial k

theorem balls_in_boxes_theorem : number_of_ways 5 4 2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_balls_in_boxes_theorem_l2758_275830


namespace NUMINAMATH_CALUDE_remaining_macaroons_weight_is_103_l2758_275881

/-- Calculates the total weight of remaining macaroons after Steve's snack --/
def remaining_macaroons_weight (
  coconut_count : ℕ)
  (coconut_weight : ℕ)
  (coconut_bags : ℕ)
  (almond_count : ℕ)
  (almond_weight : ℕ)
  (almond_bags : ℕ)
  (white_count : ℕ)
  (white_weight : ℕ) : ℕ :=
  let remaining_coconut := (coconut_count / coconut_bags) * (coconut_bags - 1) * coconut_weight
  let remaining_almond := (almond_count - almond_count / almond_bags / 2) * almond_weight
  let remaining_white := (white_count - 1) * white_weight
  remaining_coconut + remaining_almond + remaining_white

theorem remaining_macaroons_weight_is_103 :
  remaining_macaroons_weight 12 5 4 8 8 2 2 10 = 103 := by
  sorry

#eval remaining_macaroons_weight 12 5 4 8 8 2 2 10

end NUMINAMATH_CALUDE_remaining_macaroons_weight_is_103_l2758_275881


namespace NUMINAMATH_CALUDE_marbles_given_to_juan_l2758_275886

theorem marbles_given_to_juan (initial_marbles : ℕ) (remaining_marbles : ℕ) 
  (h1 : initial_marbles = 143)
  (h2 : remaining_marbles = 70) :
  initial_marbles - remaining_marbles = 73 := by
  sorry

end NUMINAMATH_CALUDE_marbles_given_to_juan_l2758_275886


namespace NUMINAMATH_CALUDE_sequence_theorem_l2758_275829

def sequence_condition (a : ℕ → Fin 2) : Prop :=
  (∀ n : ℕ, n > 0 → a n + a (n + 1) ≠ a (n + 2) + a (n + 3)) ∧
  (∀ n : ℕ, n > 0 → a n + a (n + 1) + a (n + 2) ≠ a (n + 3) + a (n + 4) + a (n + 5))

theorem sequence_theorem (a : ℕ → Fin 2) (h : sequence_condition a) (h₁ : a 1 = 0) :
  a 2020 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_theorem_l2758_275829


namespace NUMINAMATH_CALUDE_linear_equation_implies_k_equals_one_l2758_275879

/-- A function that represents the linearity condition of an equation -/
def is_linear_equation (k : ℝ) : Prop :=
  (k + 1 ≠ 0) ∧ (|k| = 1)

/-- Theorem stating that if (k+1)x + 8y^|k| + 3 = 0 is a linear equation in x and y, then k = 1 -/
theorem linear_equation_implies_k_equals_one :
  is_linear_equation k → k = 1 := by sorry

end NUMINAMATH_CALUDE_linear_equation_implies_k_equals_one_l2758_275879


namespace NUMINAMATH_CALUDE_circles_and_tangent_line_l2758_275891

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 4
def circle_O2_center : ℝ × ℝ := (3, 3)

-- Define the external tangency condition
def externally_tangent (O1 O2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (O1.1 - O2.1)^2 + (O1.2 - O2.2)^2 = (r1 + r2)^2

-- Theorem statement
theorem circles_and_tangent_line :
  ∃ (r2 : ℝ),
    -- Circle O₂ equation
    (∀ x y : ℝ, (x - 3)^2 + (y - 3)^2 = r2^2) ∧
    -- External tangency condition
    externally_tangent (0, -1) circle_O2_center 2 r2 ∧
    -- Common internal tangent line equation
    (∀ x y : ℝ, circle_O1 x y ∧ (x - 3)^2 + (y - 3)^2 = r2^2 →
      3*x + 4*y = 6) :=
by
  sorry

end NUMINAMATH_CALUDE_circles_and_tangent_line_l2758_275891


namespace NUMINAMATH_CALUDE_x_less_than_two_necessary_not_sufficient_l2758_275837

theorem x_less_than_two_necessary_not_sufficient :
  (∀ x : ℝ, |x - 1| < 1 → x < 2) ∧
  (∃ x : ℝ, x < 2 ∧ |x - 1| ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_x_less_than_two_necessary_not_sufficient_l2758_275837


namespace NUMINAMATH_CALUDE_committee_formation_count_l2758_275884

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of people in the club -/
def total_people : ℕ := 12

/-- The number of board members -/
def board_members : ℕ := 3

/-- The size of the committee -/
def committee_size : ℕ := 5

/-- The number of regular members (non-board members) -/
def regular_members : ℕ := total_people - board_members

theorem committee_formation_count :
  choose total_people committee_size - choose regular_members committee_size = 666 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_count_l2758_275884


namespace NUMINAMATH_CALUDE_set_A_properties_l2758_275887

def A : Set ℤ := {x | ∃ k : ℤ, x = 3 * k - 1}

theorem set_A_properties :
  (-11 ∉ A) ∧
  (∀ k : ℤ, 3 * k ^ 2 - 1 ∈ A) ∧
  (-34 ∈ A) := by
  sorry

end NUMINAMATH_CALUDE_set_A_properties_l2758_275887


namespace NUMINAMATH_CALUDE_bulls_and_heat_wins_l2758_275897

/-- The number of games won by the Chicago Bulls and Miami Heat combined in 2010 -/
theorem bulls_and_heat_wins (bulls_wins : ℕ) (heat_wins : ℕ) : 
  bulls_wins = 70 →
  heat_wins = bulls_wins + 5 →
  bulls_wins + heat_wins = 145 := by
  sorry

end NUMINAMATH_CALUDE_bulls_and_heat_wins_l2758_275897


namespace NUMINAMATH_CALUDE_positive_correlation_implies_positive_slope_l2758_275844

/-- Represents a simple linear regression model --/
structure LinearRegression where
  b : ℝ  -- slope
  a : ℝ  -- y-intercept
  r : ℝ  -- correlation coefficient

/-- Theorem stating that a positive correlation coefficient implies a positive slope --/
theorem positive_correlation_implies_positive_slope (model : LinearRegression) :
  model.r > 0 → model.b > 0 := by
  sorry


end NUMINAMATH_CALUDE_positive_correlation_implies_positive_slope_l2758_275844


namespace NUMINAMATH_CALUDE_methane_moles_in_reaction_l2758_275869

/-- 
Proves that the number of moles of Methane combined is equal to 1, given the conditions of the chemical reaction.
-/
theorem methane_moles_in_reaction (x : ℝ) : 
  (x > 0) →  -- Assuming positive number of moles
  (∃ y : ℝ, y > 0 ∧ x + 4 = y + 4) →  -- Mass balance equation
  (1 : ℝ) / x = (1 : ℝ) / 1 →  -- Stoichiometric ratio
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_methane_moles_in_reaction_l2758_275869


namespace NUMINAMATH_CALUDE_measure_six_pints_l2758_275804

/-- Represents the state of wine distribution -/
structure WineState :=
  (total : ℕ)
  (container8 : ℕ)
  (container5 : ℕ)

/-- Represents a pouring action -/
inductive PourAction
  | FillFrom8To5
  | FillFrom5To8
  | EmptyTo8
  | EmptyTo5
  | Empty8
  | Empty5

/-- Applies a pouring action to a wine state -/
def applyAction (state : WineState) (action : PourAction) : WineState :=
  match action with
  | PourAction.FillFrom8To5 => sorry
  | PourAction.FillFrom5To8 => sorry
  | PourAction.EmptyTo8 => sorry
  | PourAction.EmptyTo5 => sorry
  | PourAction.Empty8 => sorry
  | PourAction.Empty5 => sorry

/-- Checks if the goal state is reached -/
def isGoalState (state : WineState) : Prop :=
  state.container8 = 6

/-- Theorem: It is possible to measure 6 pints into the 8-pint container -/
theorem measure_six_pints 
  (initialState : WineState)
  (h_total : initialState.total = 12)
  (h_containers : initialState.container8 = 0 ∧ initialState.container5 = 0) :
  ∃ (actions : List PourAction), 
    isGoalState (actions.foldl applyAction initialState) :=
sorry

end NUMINAMATH_CALUDE_measure_six_pints_l2758_275804


namespace NUMINAMATH_CALUDE_quadratic_function_proof_l2758_275818

theorem quadratic_function_proof (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = -2 ∨ x = 4) →
  (∃ x, ∀ y, a * y^2 + b * y + c ≤ a * x^2 + b * x + c) →
  (∃ x, a * x^2 + b * x + c = 9) →
  (∀ x, a * x^2 + b * x + c = -x^2 + 2*x + 8) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_proof_l2758_275818


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2758_275826

theorem trigonometric_equation_solution (k : ℤ) :
  let x₁ := π / 60 + k * π / 10
  let x₂ := -π / 24 - k * π / 4
  (∀ x, x = x₁ ∨ x = x₂ → (Real.sin (3 * x) + Real.sqrt 3 * Real.cos (3 * x))^2 - 2 * Real.cos (14 * x) = 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2758_275826


namespace NUMINAMATH_CALUDE_base9_addition_l2758_275827

/-- Converts a base 9 number to base 10 --/
def base9To10 (x : ℕ) : ℕ := 
  (x / 100) * 81 + ((x / 10) % 10) * 9 + (x % 10)

/-- Converts a base 10 number to base 9 --/
def base10To9 (x : ℕ) : ℕ := 
  (x / 81) * 100 + ((x / 9) % 9) * 10 + (x % 9)

theorem base9_addition : 
  base10To9 (base9To10 236 + base9To10 327 + base9To10 284) = 858 := by
  sorry

end NUMINAMATH_CALUDE_base9_addition_l2758_275827


namespace NUMINAMATH_CALUDE_root_in_interval_l2758_275805

def f (x : ℝ) := x^3 - 2*x - 5

theorem root_in_interval :
  (∃ x ∈ Set.Icc 2 2.5, f x = 0) :=
by
  have h1 : f 2 < 0 := by sorry
  have h2 : f 2.5 > 0 := by sorry
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l2758_275805


namespace NUMINAMATH_CALUDE_line_equation_coordinate_form_l2758_275843

/-- Represents a 3D vector -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a line passing through the origin -/
structure Line where
  direction : Vector3D

/-- The direction vector of a line is a unit vector -/
def Line.isUnitVector (l : Line) : Prop :=
  l.direction.x^2 + l.direction.y^2 + l.direction.z^2 = 1

/-- The components of the direction vector are cosines of angles with coordinate axes -/
def Line.directionCosines (l : Line) (α β γ : ℝ) : Prop :=
  l.direction.x = Real.cos α ∧
  l.direction.y = Real.cos β ∧
  l.direction.z = Real.cos γ

/-- A point on the line -/
def Line.pointOnLine (l : Line) (t : ℝ) : Vector3D :=
  { x := t * l.direction.x,
    y := t * l.direction.y,
    z := t * l.direction.z }

/-- The coordinate form of the line equation -/
def Line.coordinateForm (l : Line) (α β γ : ℝ) : Prop :=
  ∀ (p : Vector3D), p ∈ Set.range (l.pointOnLine) →
    p.x / Real.cos α = p.y / Real.cos β ∧
    p.y / Real.cos β = p.z / Real.cos γ

/-- The main theorem: proving the coordinate form of the line equation -/
theorem line_equation_coordinate_form (l : Line) (α β γ : ℝ) :
  l.isUnitVector →
  l.directionCosines α β γ →
  l.coordinateForm α β γ := by
  sorry


end NUMINAMATH_CALUDE_line_equation_coordinate_form_l2758_275843


namespace NUMINAMATH_CALUDE_cayley_competition_certificates_l2758_275838

theorem cayley_competition_certificates (boys girls : ℕ) 
  (boys_percent girls_percent : ℚ) (h1 : boys = 30) (h2 : girls = 20) 
  (h3 : boys_percent = 1/10) (h4 : girls_percent = 1/5) : 
  (boys_percent * boys + girls_percent * girls) / (boys + girls) = 7/50 := by
  sorry

end NUMINAMATH_CALUDE_cayley_competition_certificates_l2758_275838


namespace NUMINAMATH_CALUDE_perpendicular_bisector_c_value_l2758_275810

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The perpendicular bisector of a line segment -/
def isPerpBisector (c : ℝ) (p1 p2 : Point) : Prop :=
  let midpoint : Point := ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2⟩
  (midpoint.x + midpoint.y = c) ∧ 
  (c - p1.x - p1.y = p2.x + p2.y - c)

/-- The theorem statement -/
theorem perpendicular_bisector_c_value :
  ∀ c : ℝ, isPerpBisector c ⟨2, 5⟩ ⟨8, 11⟩ → c = 13 := by
  sorry


end NUMINAMATH_CALUDE_perpendicular_bisector_c_value_l2758_275810


namespace NUMINAMATH_CALUDE_max_a_cubic_function_l2758_275831

/-- Given a cubic function f(x) = ax^3 + bx^2 + cx + d with a ≠ 0,
    and |f'(x)| ≤ 1 for 0 ≤ x ≤ 1, the maximum value of a is 8/3. -/
theorem max_a_cubic_function (a b c d : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, 0 ≤ x → x ≤ 1 → |3 * a * x^2 + 2 * b * x + c| ≤ 1) →
  a ≤ 8/3 :=
by sorry

end NUMINAMATH_CALUDE_max_a_cubic_function_l2758_275831


namespace NUMINAMATH_CALUDE_eighteen_letter_arrangements_l2758_275809

theorem eighteen_letter_arrangements :
  let n : ℕ := 6
  let total_letters : ℕ := 3 * n
  let arrangement_count : ℕ := (Finset.range (n + 1)).sum (fun k => (Nat.choose n k)^3)
  ∀ (arrangements : Finset (Fin total_letters → Fin 3)),
    (∀ i : Fin total_letters, 
      (arrangements.card = arrangement_count) ∧
      (arrangements.card = (Finset.filter (fun arr => 
        (∀ j : Fin n, arr (j) ≠ 0) ∧
        (∀ j : Fin n, arr (j + n) ≠ 1) ∧
        (∀ j : Fin n, arr (j + 2*n) ≠ 2) ∧
        (arrangements.filter (fun arr => arr i = 0)).card = n ∧
        (arrangements.filter (fun arr => arr i = 1)).card = n ∧
        (arrangements.filter (fun arr => arr i = 2)).card = n
      ) arrangements).card)) := by
  sorry

#check eighteen_letter_arrangements

end NUMINAMATH_CALUDE_eighteen_letter_arrangements_l2758_275809


namespace NUMINAMATH_CALUDE_left_focus_coordinates_l2758_275840

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 2 - y^2 / 2 = 1

/-- The left focus of the hyperbola -/
def left_focus : ℝ × ℝ := (-2, 0)

/-- Theorem: The coordinates of the left focus of the given hyperbola are (-2,0) -/
theorem left_focus_coordinates :
  ∀ (x y : ℝ), hyperbola_equation x y → left_focus = (-2, 0) := by
  sorry

end NUMINAMATH_CALUDE_left_focus_coordinates_l2758_275840


namespace NUMINAMATH_CALUDE_average_roots_quadratic_l2758_275808

theorem average_roots_quadratic (x₁ x₂ : ℝ) : 
  (3 * x₁^2 + 4 * x₁ - 5 = 0) → 
  (3 * x₂^2 + 4 * x₂ - 5 = 0) → 
  x₁ ≠ x₂ → 
  (x₁ + x₂) / 2 = -2/3 := by
sorry

end NUMINAMATH_CALUDE_average_roots_quadratic_l2758_275808


namespace NUMINAMATH_CALUDE_airplane_seats_total_l2758_275875

theorem airplane_seats_total (first_class : ℕ) (coach : ℕ) : 
  first_class = 77 → 
  coach = 4 * first_class + 2 → 
  first_class + coach = 387 := by
sorry

end NUMINAMATH_CALUDE_airplane_seats_total_l2758_275875


namespace NUMINAMATH_CALUDE_gcd_8421_4312_l2758_275852

theorem gcd_8421_4312 : Nat.gcd 8421 4312 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8421_4312_l2758_275852


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l2758_275892

/-- Given a point M with coordinates (3, -5), prove that its symmetric point
    with respect to the origin has coordinates (-3, 5). -/
theorem symmetric_point_wrt_origin :
  let M : ℝ × ℝ := (3, -5)
  let symmetric_point : ℝ × ℝ → ℝ × ℝ := λ (x, y) => (-x, -y)
  symmetric_point M = (-3, 5) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l2758_275892


namespace NUMINAMATH_CALUDE_count_eight_digit_numbers_product_7000_l2758_275873

/-- The number of eight-digit numbers whose digits multiply to 7000 -/
def eight_digit_numbers_with_product_7000 : ℕ := 5600

/-- The prime factorization of 7000 -/
def prime_factorization_7000 : List ℕ := [7, 2, 2, 2, 5, 5, 5]

theorem count_eight_digit_numbers_product_7000 :
  eight_digit_numbers_with_product_7000 = 5600 := by
  sorry

end NUMINAMATH_CALUDE_count_eight_digit_numbers_product_7000_l2758_275873


namespace NUMINAMATH_CALUDE_faye_flowers_proof_l2758_275890

theorem faye_flowers_proof (flowers_per_bouquet : ℕ) (wilted_flowers : ℕ) (remaining_bouquets : ℕ) 
  (h1 : flowers_per_bouquet = 5)
  (h2 : wilted_flowers = 48)
  (h3 : remaining_bouquets = 8) :
  flowers_per_bouquet * remaining_bouquets + wilted_flowers = 88 :=
by sorry

end NUMINAMATH_CALUDE_faye_flowers_proof_l2758_275890


namespace NUMINAMATH_CALUDE_inequality_proof_l2758_275872

theorem inequality_proof (a b c : ℝ) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a + b + c = 3) : 
  a^b + b^c + c^a ≤ a^2 + b^2 + c^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2758_275872


namespace NUMINAMATH_CALUDE_b_time_approx_l2758_275848

/-- The time it takes for A to complete the work alone -/
def a_time : ℝ := 20

/-- The time it takes for A and B to complete the work together -/
def ab_time : ℝ := 12.727272727272728

/-- The time it takes for B to complete the work alone -/
noncomputable def b_time : ℝ := (a_time * ab_time) / (a_time - ab_time)

/-- Theorem stating that B can complete the work in approximately 34.90909090909091 days -/
theorem b_time_approx : 
  ∃ ε > 0, |b_time - 34.90909090909091| < ε :=
sorry

end NUMINAMATH_CALUDE_b_time_approx_l2758_275848


namespace NUMINAMATH_CALUDE_correct_seating_arrangements_l2758_275806

/-- Represents a seating arrangement in an examination room --/
structure ExamRoom :=
  (rows : Nat)
  (columns : Nat)

/-- Calculates the number of seating arrangements for two students
    who cannot be seated adjacent to each other --/
def countSeatingArrangements (room : ExamRoom) : Nat :=
  sorry

/-- Theorem stating the correct number of seating arrangements --/
theorem correct_seating_arrangements :
  let room : ExamRoom := { rows := 5, columns := 6 }
  countSeatingArrangements room = 772 := by
  sorry

end NUMINAMATH_CALUDE_correct_seating_arrangements_l2758_275806


namespace NUMINAMATH_CALUDE_birthday_cake_problem_l2758_275857

/-- Represents a cube cake with icing -/
structure CakeCube where
  size : Nat
  has_icing : Bool

/-- Counts the number of small cubes with icing on exactly two sides -/
def count_two_sided_icing (cake : CakeCube) : Nat :=
  sorry

/-- The main theorem about the birthday cake problem -/
theorem birthday_cake_problem (cake : CakeCube) :
  cake.size = 5 ∧ cake.has_icing = true → count_two_sided_icing cake = 96 := by
  sorry

end NUMINAMATH_CALUDE_birthday_cake_problem_l2758_275857


namespace NUMINAMATH_CALUDE_range_of_a_l2758_275821

theorem range_of_a (a : ℝ) : 
  (∃ x₀ : ℝ, 2 * x₀^2 + (a - 1) * x₀ + 1/2 ≤ 0) ↔ 
  a ≤ -1 ∨ a ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2758_275821


namespace NUMINAMATH_CALUDE_inverse_variation_cube_l2758_275876

/-- Given that x and y are positive real numbers, x^3 and y vary inversely,
    and y = 8 when x = 2, prove that x = 1 when y = 64. -/
theorem inverse_variation_cube (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h_inverse : ∃ k : ℝ, ∀ x y, x^3 * y = k)
  (h_initial : 2^3 * 8 = (Classical.choose h_inverse))
  (h_y : y = 64) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_cube_l2758_275876


namespace NUMINAMATH_CALUDE_unique_magnitude_of_quadratic_root_l2758_275817

theorem unique_magnitude_of_quadratic_root : ∃! m : ℝ, ∃ z : ℂ, z^2 - 6*z + 25 = 0 ∧ Complex.abs z = m := by
  sorry

end NUMINAMATH_CALUDE_unique_magnitude_of_quadratic_root_l2758_275817


namespace NUMINAMATH_CALUDE_watch_sale_loss_percentage_l2758_275842

/-- Proves that the loss percentage is 10% for a watch sale scenario --/
theorem watch_sale_loss_percentage 
  (cost_price : ℝ)
  (selling_price : ℝ)
  (h1 : cost_price = 1200)
  (h2 : selling_price < cost_price)
  (h3 : selling_price + 180 = cost_price * 1.05) :
  (cost_price - selling_price) / cost_price * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_watch_sale_loss_percentage_l2758_275842


namespace NUMINAMATH_CALUDE_equation_solutions_l2758_275834

theorem equation_solutions : 
  let equation := fun x : ℝ => x^2 * (x + 1)^2 + x^2 - 3 * (x + 1)^2
  ∀ x : ℝ, equation x = 0 ↔ x = (1 + Real.sqrt 5) / 2 ∨ x = (1 - Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2758_275834


namespace NUMINAMATH_CALUDE_inlet_pipe_rate_l2758_275896

/-- Prove that the inlet pipe rate is 3 cubic inches/min given the tank and pipe conditions -/
theorem inlet_pipe_rate (tank_volume : ℝ) (outlet_rate1 outlet_rate2 : ℝ) (empty_time : ℝ) :
  tank_volume = 51840 ∧ 
  outlet_rate1 = 9 ∧ 
  outlet_rate2 = 6 ∧ 
  empty_time = 4320 →
  ∃ inlet_rate : ℝ, 
    inlet_rate = 3 ∧ 
    (outlet_rate1 + outlet_rate2 - inlet_rate) * empty_time = tank_volume :=
by
  sorry

end NUMINAMATH_CALUDE_inlet_pipe_rate_l2758_275896


namespace NUMINAMATH_CALUDE_add_12345_seconds_to_1045am_l2758_275893

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- The initial time: 10:45:00 -/
def initialTime : Time :=
  { hours := 10, minutes := 45, seconds := 0 }

/-- The number of seconds to add -/
def secondsToAdd : Nat := 12345

/-- The expected final time: 13:45:45 -/
def expectedFinalTime : Time :=
  { hours := 13, minutes := 45, seconds := 45 }

theorem add_12345_seconds_to_1045am :
  addSeconds initialTime secondsToAdd = expectedFinalTime := by
  sorry

end NUMINAMATH_CALUDE_add_12345_seconds_to_1045am_l2758_275893


namespace NUMINAMATH_CALUDE_yangzhou_construction_area_scientific_notation_l2758_275870

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem yangzhou_construction_area_scientific_notation :
  toScientificNotation 330100000 = ScientificNotation.mk 3.301 8 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_yangzhou_construction_area_scientific_notation_l2758_275870


namespace NUMINAMATH_CALUDE_yogurt_combinations_l2758_275814

def yogurt_types : ℕ := 2
def yogurt_flavors : ℕ := 5
def topping_count : ℕ := 8

def combination_count : ℕ := yogurt_types * yogurt_flavors * (topping_count.choose 2)

theorem yogurt_combinations :
  combination_count = 280 :=
by sorry

end NUMINAMATH_CALUDE_yogurt_combinations_l2758_275814


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2758_275866

/-- Represents the speed and travel time of a train -/
structure Train where
  speed : ℝ
  time_after_meeting : ℝ

/-- Theorem stating the relationship between two trains meeting and their speeds -/
theorem train_speed_calculation (train_a train_b : Train) 
  (h1 : train_a.speed = 60)
  (h2 : train_a.time_after_meeting = 9)
  (h3 : train_b.time_after_meeting = 4) :
  train_b.speed = 135 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l2758_275866


namespace NUMINAMATH_CALUDE_triangle_on_parabola_bc_length_l2758_275800

/-- Parabola function -/
def parabola (x : ℝ) : ℝ := x^2

/-- Triangle ABC -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Check if a point lies on the parabola -/
def onParabola (p : ℝ × ℝ) : Prop :=
  p.2 = parabola p.1

/-- Check if two points have the same y-coordinate (i.e., line is parallel to x-axis) -/
def parallelToXAxis (p q : ℝ × ℝ) : Prop :=
  p.2 = q.2

/-- Calculate the area of a triangle given its vertices -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  sorry

/-- Calculate the length of a line segment -/
noncomputable def segmentLength (p q : ℝ × ℝ) : ℝ :=
  sorry

/-- Main theorem -/
theorem triangle_on_parabola_bc_length (t : Triangle) :
  onParabola t.A ∧ onParabola t.B ∧ onParabola t.C ∧
  t.A = (1, 1) ∧
  parallelToXAxis t.B t.C ∧
  triangleArea t = 50 →
  ∃ ε > 0, |segmentLength t.B t.C - 5.8| < ε :=
sorry

end NUMINAMATH_CALUDE_triangle_on_parabola_bc_length_l2758_275800


namespace NUMINAMATH_CALUDE_range_of_sum_l2758_275828

theorem range_of_sum (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_eq : 2 * x + y + 4 * x * y = 15 / 2) : 2 * x + y ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sum_l2758_275828


namespace NUMINAMATH_CALUDE_smallest_consecutive_integer_l2758_275824

theorem smallest_consecutive_integer (a b c d : ℕ) : 
  a > 0 ∧ b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ a * b * c * d = 1680 → a = 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_consecutive_integer_l2758_275824


namespace NUMINAMATH_CALUDE_largest_common_number_l2758_275801

/-- First sequence with initial term 5 and common difference 9 -/
def sequence1 (n : ℕ) : ℕ := 5 + 9 * n

/-- Second sequence with initial term 3 and common difference 8 -/
def sequence2 (m : ℕ) : ℕ := 3 + 8 * m

/-- Theorem stating that 167 is the largest common number in both sequences within the range 1 to 200 -/
theorem largest_common_number :
  ∃ (n m : ℕ),
    sequence1 n = sequence2 m ∧
    sequence1 n = 167 ∧
    sequence1 n ≤ 200 ∧
    ∀ (k l : ℕ), sequence1 k = sequence2 l → sequence1 k ≤ 200 → sequence1 k ≤ 167 :=
by sorry

end NUMINAMATH_CALUDE_largest_common_number_l2758_275801


namespace NUMINAMATH_CALUDE_cube_root_of_square_64_l2758_275889

theorem cube_root_of_square_64 (x : ℝ) (h : x^2 = 64) :
  ∃ y, y^3 = x ∧ (y = 2 ∨ y = -2) := by sorry

end NUMINAMATH_CALUDE_cube_root_of_square_64_l2758_275889


namespace NUMINAMATH_CALUDE_survey_questions_l2758_275825

-- Define the number of questions per survey
def questionsPerSurvey : ℕ := sorry

-- Define the payment per question
def paymentPerQuestion : ℚ := 1/5

-- Define the number of surveys completed on Monday
def mondaySurveys : ℕ := 3

-- Define the number of surveys completed on Tuesday
def tuesdaySurveys : ℕ := 4

-- Define the total earnings
def totalEarnings : ℚ := 14

-- Theorem statement
theorem survey_questions :
  questionsPerSurvey * (mondaySurveys + tuesdaySurveys : ℚ) * paymentPerQuestion = totalEarnings ∧
  questionsPerSurvey = 10 := by
  sorry

end NUMINAMATH_CALUDE_survey_questions_l2758_275825


namespace NUMINAMATH_CALUDE_circle_equation_and_max_ratio_l2758_275856

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x + 1)^2 + y^2 = 2

-- Define the given line equations
def line1 (x y : ℝ) : Prop := x - y + 1 = 0
def line2 (x y : ℝ) : Prop := x + y + 3 = 0

-- Define the second circle
def circle2 (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*y + 3 = 0

theorem circle_equation_and_max_ratio :
  (∃ (x₀ y₀ : ℝ), line1 x₀ y₀ ∧ y₀ = 0 ∧
    ∀ (x y : ℝ), circle_C x y ↔ (x - x₀)^2 + (y - y₀)^2 = 2 ∧
    ∃ (x₁ y₁ : ℝ), line2 x₁ y₁ ∧ (x₁ - x₀)^2 + (y₁ - y₀)^2 = 2) ∧
  (∀ (x y : ℝ), circle2 x y → y / x ≤ Real.sqrt 3 / 3) ∧
  (∃ (x y : ℝ), circle2 x y ∧ y / x = Real.sqrt 3 / 3) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_and_max_ratio_l2758_275856


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2758_275882

theorem inequality_solution_set (x : ℝ) :
  x ≠ -2 ∧ x ≠ 9/2 →
  ((x + 1) / (x + 2) > (3 * x + 4) / (2 * x + 9)) ↔
  (-9/2 ≤ x ∧ x ≤ -2) ∨ ((1 - Real.sqrt 5) / 2 < x ∧ x < (1 + Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2758_275882


namespace NUMINAMATH_CALUDE_thirty_divides_p_squared_minus_one_l2758_275815

theorem thirty_divides_p_squared_minus_one (p : ℕ) (hp : p.Prime) (hp_ge_5 : p ≥ 5) :
  30 ∣ (p^2 - 1) ↔ p = 5 := by
  sorry

end NUMINAMATH_CALUDE_thirty_divides_p_squared_minus_one_l2758_275815


namespace NUMINAMATH_CALUDE_sureshs_speed_l2758_275816

/-- Suresh's walking speed problem -/
theorem sureshs_speed (track_circumference : ℝ) (meeting_time : ℝ) (wife_speed : ℝ) 
  (h1 : track_circumference = 726) 
  (h2 : meeting_time = 5.28)
  (h3 : wife_speed = 3.75) : 
  ∃ (suresh_speed : ℝ), abs (suresh_speed - 4.5054) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_sureshs_speed_l2758_275816


namespace NUMINAMATH_CALUDE_andy_late_time_l2758_275883

def school_start_time : Nat := 8 * 60  -- 8:00 AM in minutes
def normal_travel_time : Nat := 30
def red_light_delay : Nat := 3
def num_red_lights : Nat := 4
def construction_delay : Nat := 10
def departure_time : Nat := 7 * 60 + 15  -- 7:15 AM in minutes

def total_delay : Nat := red_light_delay * num_red_lights + construction_delay

def actual_travel_time : Nat := normal_travel_time + total_delay

def arrival_time : Nat := departure_time + actual_travel_time

theorem andy_late_time : arrival_time - school_start_time = 7 := by
  sorry

end NUMINAMATH_CALUDE_andy_late_time_l2758_275883


namespace NUMINAMATH_CALUDE_stamp_collection_problem_l2758_275853

/-- Represents the number of stamps Simon received from each friend -/
structure FriendStamps where
  x1 : ℕ
  x2 : ℕ
  x3 : ℕ
  x4 : ℕ
  x5 : ℕ

/-- Theorem representing the stamp collection problem -/
theorem stamp_collection_problem 
  (initial_stamps final_stamps : ℕ) 
  (friend_stamps : FriendStamps) : 
  initial_stamps = 34 →
  final_stamps = 61 →
  friend_stamps.x1 = 12 →
  friend_stamps.x3 = 21 →
  friend_stamps.x5 = 10 →
  friend_stamps.x1 + friend_stamps.x2 + friend_stamps.x3 + 
  friend_stamps.x4 + friend_stamps.x5 = final_stamps - initial_stamps :=
by
  sorry

#check stamp_collection_problem

end NUMINAMATH_CALUDE_stamp_collection_problem_l2758_275853


namespace NUMINAMATH_CALUDE_batsman_average_l2758_275859

theorem batsman_average (previous_total : ℕ) (previous_average : ℚ) : 
  previous_total = (16 : ℕ) * previous_average ∧ 
  (previous_total + 56) / 17 = previous_average + 3 →
  (previous_total + 56) / 17 = 8 := by
sorry

end NUMINAMATH_CALUDE_batsman_average_l2758_275859


namespace NUMINAMATH_CALUDE_negative_rational_and_fraction_l2758_275850

-- Define the number -0.3
def num : ℚ := -3/10

-- Theorem statement
theorem negative_rational_and_fraction (n : ℚ) (h : n = -3/10) :
  n < 0 ∧ ∃ (a b : ℤ), b ≠ 0 ∧ n = a / b :=
sorry

end NUMINAMATH_CALUDE_negative_rational_and_fraction_l2758_275850


namespace NUMINAMATH_CALUDE_prime_pairs_satisfying_equation_l2758_275894

theorem prime_pairs_satisfying_equation :
  ∀ (p q : ℕ), Prime p → Prime q →
  (∃ (x y z : ℕ), p^(2*x) + q^(2*y) = z^2) ↔ ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) :=
by sorry

end NUMINAMATH_CALUDE_prime_pairs_satisfying_equation_l2758_275894


namespace NUMINAMATH_CALUDE_no_rational_roots_l2758_275895

def polynomial (x : ℚ) : ℚ := 3 * x^4 - 7 * x^3 - 4 * x^2 + 8 * x + 3

theorem no_rational_roots :
  ∀ x : ℚ, polynomial x ≠ 0 := by sorry

end NUMINAMATH_CALUDE_no_rational_roots_l2758_275895


namespace NUMINAMATH_CALUDE_transformation_composition_dilation_property_rotation_property_transformation_is_dilation_then_rotation_l2758_275888

def dilation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 2]
def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]
def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, -2; 2, 0]

theorem transformation_composition :
  rotation_matrix * dilation_matrix = transformation_matrix :=
by sorry

theorem dilation_property (v : Fin 2 → ℝ) :
  dilation_matrix.mulVec v = 2 • v :=
by sorry

theorem rotation_property (v : Fin 2 → ℝ) :
  rotation_matrix.mulVec v = ![- v 1, v 0] :=
by sorry

theorem transformation_is_dilation_then_rotation :
  ∀ v : Fin 2 → ℝ,
  transformation_matrix.mulVec v = rotation_matrix.mulVec (dilation_matrix.mulVec v) :=
by sorry

end NUMINAMATH_CALUDE_transformation_composition_dilation_property_rotation_property_transformation_is_dilation_then_rotation_l2758_275888


namespace NUMINAMATH_CALUDE_distribute_researchers_count_l2758_275865

/-- The number of ways to distribute 4 researchers to 3 schools -/
def distribute_researchers : ℕ :=
  -- Number of ways to divide 4 researchers into 3 groups (one group of 2, two groups of 1)
  (Nat.choose 4 2) *
  -- Number of ways to assign 3 groups to 3 schools
  (Nat.factorial 3)

/-- Theorem stating that the number of distribution schemes is 36 -/
theorem distribute_researchers_count :
  distribute_researchers = 36 := by
  sorry

end NUMINAMATH_CALUDE_distribute_researchers_count_l2758_275865


namespace NUMINAMATH_CALUDE_subset_condition_l2758_275868

theorem subset_condition (a b : ℝ) : 
  let A : Set ℝ := {x | x^2 - 1 = 0}
  let B : Set ℝ := {y | y^2 - 2*a*y + b = 0}
  (B ⊆ A) ∧ (B ≠ ∅) → 
  ((a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = 1) ∨ (a = 0 ∧ b = -1)) := by
sorry

end NUMINAMATH_CALUDE_subset_condition_l2758_275868


namespace NUMINAMATH_CALUDE_max_leftover_candy_l2758_275822

theorem max_leftover_candy (x : ℕ) : ∃ (q r : ℕ), x = 10 * q + r ∧ r < 10 ∧ r ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_max_leftover_candy_l2758_275822
