import Mathlib

namespace NUMINAMATH_CALUDE_a_equals_2_sufficient_not_necessary_l2478_247895

def A : Set ℝ := {0, 4}
def B (a : ℝ) : Set ℝ := {2, a^2}

theorem a_equals_2_sufficient_not_necessary :
  (∀ a : ℝ, a = 2 → A ∩ B a = {4}) ∧
  (∃ a : ℝ, a ≠ 2 ∧ A ∩ B a = {4}) :=
by sorry

end NUMINAMATH_CALUDE_a_equals_2_sufficient_not_necessary_l2478_247895


namespace NUMINAMATH_CALUDE_students_looking_up_fraction_l2478_247816

theorem students_looking_up_fraction : 
  ∀ (total_students : ℕ) (eyes_saw_plane : ℕ) (eyes_per_student : ℕ),
    total_students = 200 →
    eyes_saw_plane = 300 →
    eyes_per_student = 2 →
    (eyes_saw_plane / eyes_per_student : ℚ) / total_students = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_students_looking_up_fraction_l2478_247816


namespace NUMINAMATH_CALUDE_symmetry_coordinates_l2478_247854

/-- Two points are symmetrical about the y-axis if their x-coordinates are negatives of each other
    and their y-coordinates are the same. -/
def symmetrical_about_y_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = q.2

theorem symmetry_coordinates :
  let p : ℝ × ℝ := (4, -5)
  let q : ℝ × ℝ := (a, b)
  symmetrical_about_y_axis p q → a = -4 ∧ b = -5 := by
sorry

end NUMINAMATH_CALUDE_symmetry_coordinates_l2478_247854


namespace NUMINAMATH_CALUDE_anna_phone_chargers_l2478_247855

/-- The number of phone chargers Anna has -/
def phone_chargers : ℕ := sorry

/-- The number of laptop chargers Anna has -/
def laptop_chargers : ℕ := sorry

/-- The total number of chargers Anna has -/
def total_chargers : ℕ := 24

theorem anna_phone_chargers :
  (laptop_chargers = 5 * phone_chargers) →
  (phone_chargers + laptop_chargers = total_chargers) →
  phone_chargers = 4 := by
sorry

end NUMINAMATH_CALUDE_anna_phone_chargers_l2478_247855


namespace NUMINAMATH_CALUDE_characterization_of_a_l2478_247813

/-- Define function composition n times -/
def iterateN (f : ℕ → ℕ) (n : ℕ) : ℕ → ℕ :=
  match n with
  | 0 => id
  | n + 1 => f ∘ (iterateN f n)

/-- The main theorem -/
theorem characterization_of_a (a : ℕ) : 
  (∃ (f g : ℕ → ℕ), Function.Bijective g ∧ 
    ∀ x, iterateN f 2009 x = g x + a) ↔ 
  ∃ k, a = 2009 * k :=
sorry

end NUMINAMATH_CALUDE_characterization_of_a_l2478_247813


namespace NUMINAMATH_CALUDE_z_pure_imaginary_iff_z_in_fourth_quadrant_iff_l2478_247838

def z (m : ℝ) : ℂ := (1 + Complex.I) * m^2 - 3 * Complex.I * m + 2 * Complex.I - 1

theorem z_pure_imaginary_iff (m : ℝ) : 
  z m = Complex.I * Complex.im (z m) ↔ m = -1 :=
sorry

theorem z_in_fourth_quadrant_iff (m : ℝ) :
  Complex.re (z m) > 0 ∧ Complex.im (z m) < 0 ↔ 1 < m ∧ m < 2 :=
sorry

end NUMINAMATH_CALUDE_z_pure_imaginary_iff_z_in_fourth_quadrant_iff_l2478_247838


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2478_247815

theorem arithmetic_mean_of_fractions : 
  (1 / 2 : ℚ) * ((3 / 8 : ℚ) + (5 / 9 : ℚ)) = (67 / 144 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2478_247815


namespace NUMINAMATH_CALUDE_base_6_addition_l2478_247889

/-- Addition of two numbers in base 6 -/
def add_base_6 (a b : ℕ) : ℕ :=
  sorry

/-- Conversion from base 10 to base 6 -/
def to_base_6 (n : ℕ) : ℕ :=
  sorry

/-- Conversion from base 6 to base 10 -/
def from_base_6 (n : ℕ) : ℕ :=
  sorry

theorem base_6_addition :
  add_base_6 (from_base_6 52301) (from_base_6 34122) = from_base_6 105032 :=
sorry

end NUMINAMATH_CALUDE_base_6_addition_l2478_247889


namespace NUMINAMATH_CALUDE_lcm_12_21_30_l2478_247844

theorem lcm_12_21_30 : Nat.lcm (Nat.lcm 12 21) 30 = 420 := by
  sorry

end NUMINAMATH_CALUDE_lcm_12_21_30_l2478_247844


namespace NUMINAMATH_CALUDE_inequality_properties_l2478_247818

theorem inequality_properties (a b : ℝ) (h : 1/a < 1/b ∧ 1/b < 0) : 
  (a + b < a * b) ∧ (a * b < b^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_properties_l2478_247818


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2478_247865

theorem inequality_system_solution (a b : ℝ) :
  (∀ x : ℝ, (2 * x - a < 1 ∧ x - 2 * b > 3) ↔ (-1 < x ∧ x < 1)) →
  (a + 1) * (b - 1) = -6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2478_247865


namespace NUMINAMATH_CALUDE_three_white_marbles_probability_l2478_247847

def total_marbles : ℕ := 5 + 7 + 15

def probability_three_white (red green white : ℕ) : ℚ :=
  (white / total_marbles) * 
  ((white - 1) / (total_marbles - 1)) * 
  ((white - 2) / (total_marbles - 2))

theorem three_white_marbles_probability :
  probability_three_white 5 7 15 = 2 / 13 := by
  sorry

end NUMINAMATH_CALUDE_three_white_marbles_probability_l2478_247847


namespace NUMINAMATH_CALUDE_range_of_m_l2478_247800

def A (m : ℝ) : Set ℝ := {x : ℝ | x^2 - m*x + 3 = 0}

theorem range_of_m :
  ∀ m : ℝ, (A m ∩ {1, 3} = A m) ↔ ((-2 * Real.sqrt 3 < m ∧ m < 2 * Real.sqrt 3) ∨ m = 4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2478_247800


namespace NUMINAMATH_CALUDE_jason_oranges_l2478_247851

/-- 
Given that Mary picked 122 oranges and the total number of oranges picked by Mary and Jason is 227,
prove that Jason picked 105 oranges.
-/
theorem jason_oranges :
  let mary_oranges : ℕ := 122
  let total_oranges : ℕ := 227
  let jason_oranges : ℕ := total_oranges - mary_oranges
  jason_oranges = 105 := by sorry

end NUMINAMATH_CALUDE_jason_oranges_l2478_247851


namespace NUMINAMATH_CALUDE_meetings_percentage_of_workday_l2478_247817

def workday_hours : ℕ := 10
def first_meeting_minutes : ℕ := 40

def second_meeting_minutes : ℕ := 2 * first_meeting_minutes
def third_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes

def total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes + third_meeting_minutes
def workday_minutes : ℕ := workday_hours * 60

theorem meetings_percentage_of_workday :
  (total_meeting_minutes : ℚ) / (workday_minutes : ℚ) = 2 / 5 :=
sorry

end NUMINAMATH_CALUDE_meetings_percentage_of_workday_l2478_247817


namespace NUMINAMATH_CALUDE_tangential_quadrilateral_theorem_l2478_247804

/-- A circle in a 2D plane -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- Check if four points are concyclic -/
def are_concyclic (A B C D : Point) : Prop := sorry

/-- Check if a point lies on a line segment -/
def point_on_segment (P A B : Point) : Prop := sorry

/-- Check if a circle is tangent to a line segment -/
def circle_tangent_to_segment (circle : Circle) (A B : Point) : Prop := sorry

/-- Distance between two points -/
def distance (A B : Point) : ℝ := sorry

theorem tangential_quadrilateral_theorem 
  (A B C D : Point) 
  (circle1 circle2 : Circle) :
  are_concyclic A B C D →
  point_on_segment circle2.center A B →
  circle_tangent_to_segment circle2 B C →
  circle_tangent_to_segment circle2 C D →
  circle_tangent_to_segment circle2 D A →
  distance A D + distance B C = distance A B := by
  sorry

end NUMINAMATH_CALUDE_tangential_quadrilateral_theorem_l2478_247804


namespace NUMINAMATH_CALUDE_f_at_zero_equals_four_l2478_247860

/-- Given a function f(x) = a * sin(x) + b * (x^(1/3)) + 4 where a and b are real numbers,
    prove that f(0) = 4 -/
theorem f_at_zero_equals_four (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin x + b * Real.rpow x (1/3) + 4
  f 0 = 4 := by sorry

end NUMINAMATH_CALUDE_f_at_zero_equals_four_l2478_247860


namespace NUMINAMATH_CALUDE_camp_food_ratio_l2478_247843

/-- The ratio of food eaten by a dog to a puppy -/
def food_ratio (num_puppies num_dogs : ℕ) 
               (puppy_meal_frequency dog_meal_frequency : ℕ) 
               (dog_food_per_meal : ℝ) 
               (total_food_per_day : ℝ) : ℚ := by
  -- Define the ratio of food eaten by a dog to a puppy
  sorry

/-- Theorem stating the food ratio given the problem conditions -/
theorem camp_food_ratio : 
  food_ratio 4 3 9 3 4 108 = 2 := by
  sorry

end NUMINAMATH_CALUDE_camp_food_ratio_l2478_247843


namespace NUMINAMATH_CALUDE_cannon_firing_time_l2478_247899

/-- Represents a cannon with a specified firing rate and number of shots -/
structure Cannon where
  firing_rate : ℕ  -- shots per minute
  total_shots : ℕ

/-- Calculates the time taken to fire all shots for a given cannon -/
def time_to_fire (c : Cannon) : ℕ :=
  c.total_shots - 1

/-- The cannon from the problem -/
def test_cannon : Cannon :=
  { firing_rate := 1, total_shots := 60 }

/-- Theorem stating that the time to fire all shots is 59 minutes -/
theorem cannon_firing_time :
  time_to_fire test_cannon = 59 := by sorry

end NUMINAMATH_CALUDE_cannon_firing_time_l2478_247899


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2478_247852

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 3 + a 8 = 22)
  (h_a6 : a 6 = 7) :
  a 5 = 15 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2478_247852


namespace NUMINAMATH_CALUDE_square_difference_plus_six_b_l2478_247822

theorem square_difference_plus_six_b (a b : ℝ) (h : a + b = 3) : 
  a^2 - b^2 + 6*b = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_plus_six_b_l2478_247822


namespace NUMINAMATH_CALUDE_min_value_quadratic_min_value_quadratic_achievable_l2478_247882

theorem min_value_quadratic (x : ℝ) : x^2 + x + 1 ≥ 3/4 :=
sorry

theorem min_value_quadratic_achievable : ∃ x : ℝ, x^2 + x + 1 = 3/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_min_value_quadratic_achievable_l2478_247882


namespace NUMINAMATH_CALUDE_pies_sold_per_day_l2478_247873

theorem pies_sold_per_day (total_pies : ℕ) (days_in_week : ℕ) 
  (h1 : total_pies = 56) 
  (h2 : days_in_week = 7) : 
  total_pies / days_in_week = 8 := by
  sorry

end NUMINAMATH_CALUDE_pies_sold_per_day_l2478_247873


namespace NUMINAMATH_CALUDE_product_of_specific_integers_l2478_247879

theorem product_of_specific_integers : 
  ∀ (a b : ℤ), a = 32 ∧ b = 32 ∧ a % 10 ≠ 0 ∧ b % 10 ≠ 0 → a * b = 1024 := by
  sorry

end NUMINAMATH_CALUDE_product_of_specific_integers_l2478_247879


namespace NUMINAMATH_CALUDE_distance_P_to_AB_l2478_247867

-- Define the rectangle ABCD
def rectangle_ABCD (A B C D : ℝ × ℝ) : Prop :=
  A.1 = 0 ∧ A.2 = 8 ∧
  B.1 = 6 ∧ B.2 = 8 ∧
  C.1 = 6 ∧ C.2 = 0 ∧
  D.1 = 0 ∧ D.2 = 0

-- Define point M as the midpoint of CD
def point_M (C D M : ℝ × ℝ) : Prop :=
  M.1 = (C.1 + D.1) / 2 ∧ M.2 = (C.2 + D.2) / 2

-- Define the circle with center M and radius 3
def circle_M (M P : ℝ × ℝ) : Prop :=
  (P.1 - M.1)^2 + (P.2 - M.2)^2 = 3^2

-- Define the circle with center B and radius 5
def circle_B (B P : ℝ × ℝ) : Prop :=
  (P.1 - B.1)^2 + (P.2 - B.2)^2 = 5^2

-- Theorem statement
theorem distance_P_to_AB (A B C D M P : ℝ × ℝ) :
  rectangle_ABCD A B C D →
  point_M C D M →
  circle_M M P →
  circle_B B P →
  P.1 = 18/5 := by sorry

end NUMINAMATH_CALUDE_distance_P_to_AB_l2478_247867


namespace NUMINAMATH_CALUDE_min_distance_squared_l2478_247890

theorem min_distance_squared (a b c d : ℝ) : 
  b = a - 2 * Real.exp a → 
  c + d = 4 → 
  ∃ (min : ℝ), min = 18 ∧ ∀ (x y : ℝ), (a - x)^2 + (b - y)^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_distance_squared_l2478_247890


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2478_247829

theorem arithmetic_expression_equality : (11 * 24 - 23 * 9) / 3 + 3 = 22 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2478_247829


namespace NUMINAMATH_CALUDE_diamond_inequality_exists_l2478_247825

/-- Definition of the diamond operation -/
def diamond (f : ℝ → ℝ) (x y : ℝ) : ℝ := |f x - f y|

/-- The function f(x) = 3x -/
def f (x : ℝ) : ℝ := 3 * x

/-- Theorem stating that 3(x ◊ y) ≠ (3x) ◊ (3y) for some x and y -/
theorem diamond_inequality_exists : ∃ x y : ℝ, 3 * (diamond f x y) ≠ diamond f (3 * x) (3 * y) := by
  sorry

end NUMINAMATH_CALUDE_diamond_inequality_exists_l2478_247825


namespace NUMINAMATH_CALUDE_calculate_expression_l2478_247814

theorem calculate_expression : (-1 : ℤ) ^ 53 + 2 ^ (4^3 + 5^2 - 7^2) = 1099511627775 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2478_247814


namespace NUMINAMATH_CALUDE_diagonal_contains_all_numbers_l2478_247837

/-- Represents a 25x25 table with integers from 1 to 25 -/
def Table := Fin 25 → Fin 25 → Fin 25

/-- The table is symmetric with respect to the main diagonal -/
def isSymmetric (t : Table) : Prop :=
  ∀ i j : Fin 25, t i j = t j i

/-- Each row contains all numbers from 1 to 25 -/
def hasAllNumbersInRow (t : Table) : Prop :=
  ∀ i : Fin 25, ∀ k : Fin 25, ∃ j : Fin 25, t i j = k

/-- The main diagonal contains all numbers from 1 to 25 -/
def allNumbersOnDiagonal (t : Table) : Prop :=
  ∀ k : Fin 25, ∃ i : Fin 25, t i i = k

theorem diagonal_contains_all_numbers (t : Table) 
  (h_sym : isSymmetric t) (h_row : hasAllNumbersInRow t) : 
  allNumbersOnDiagonal t := by
  sorry

end NUMINAMATH_CALUDE_diagonal_contains_all_numbers_l2478_247837


namespace NUMINAMATH_CALUDE_log_properties_l2478_247827

-- Define the logarithm function for base b
noncomputable def log_b (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- State the theorem
theorem log_properties (b : ℝ) (h : 0 < b ∧ b < 1) :
  (log_b b 1 = 0) ∧ 
  (log_b b b = 1) ∧ 
  (∀ x : ℝ, 1 < x → x < b → log_b b x > 0) ∧
  (∀ x y : ℝ, 1 < x → x < y → y < b → log_b b x > log_b b y) :=
by sorry

end NUMINAMATH_CALUDE_log_properties_l2478_247827


namespace NUMINAMATH_CALUDE_min_groups_and_people_is_16_l2478_247859

/-- Represents the seating arrangement in a cafe -/
structure CafeSeating where
  tables : Nat
  counter_seats : Nat
  min_group_size : Nat
  max_group_size : Nat

/-- Represents the final seating state of the cafe -/
structure SeatingState where
  groups : Nat
  total_people : Nat

/-- The minimum possible value of groups + total people given the cafe seating conditions -/
def min_groups_and_people (cafe : CafeSeating) : Nat :=
  16

/-- Theorem stating that the minimum possible value of M + N is 16 -/
theorem min_groups_and_people_is_16 (cafe : CafeSeating) 
  (h1 : cafe.tables = 3)
  (h2 : cafe.counter_seats = 5)
  (h3 : cafe.min_group_size = 1)
  (h4 : cafe.max_group_size = 4)
  (state : SeatingState)
  (h5 : state.groups + state.total_people ≥ min_groups_and_people cafe) :
  min_groups_and_people cafe = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_groups_and_people_is_16_l2478_247859


namespace NUMINAMATH_CALUDE_cashier_problem_l2478_247812

/-- Represents the denominations of banknotes available in Forints -/
inductive Denomination
  | Ft50 : Denomination
  | Ft100 : Denomination
  | Ft500 : Denomination
  | Ft1000 : Denomination
  | Ft5000 : Denomination

/-- The value of a denomination in Forints -/
def denominationValue : Denomination → Nat
  | Denomination.Ft50 => 50
  | Denomination.Ft100 => 100
  | Denomination.Ft500 => 500
  | Denomination.Ft1000 => 1000
  | Denomination.Ft5000 => 5000

/-- Represents a distribution of banknotes -/
def BankNoteDistribution := Denomination → Nat

/-- Calculates the total value of a banknote distribution in Forints -/
def totalValue (distribution : BankNoteDistribution) : Nat :=
  (distribution Denomination.Ft50) * 50 +
  (distribution Denomination.Ft100) * 100 +
  (distribution Denomination.Ft500) * 500 +
  (distribution Denomination.Ft1000) * 1000 +
  (distribution Denomination.Ft5000) * 5000

/-- Calculates the total number of banknotes in a distribution -/
def totalNotes (distribution : BankNoteDistribution) : Nat :=
  (distribution Denomination.Ft50) +
  (distribution Denomination.Ft100) +
  (distribution Denomination.Ft500) +
  (distribution Denomination.Ft1000) +
  (distribution Denomination.Ft5000)

/-- The theorem to be proved -/
theorem cashier_problem (initialDistribution finalDistribution : BankNoteDistribution) :
  (totalValue initialDistribution % 50 = 0) →
  (totalNotes initialDistribution ≥ 15) →
  (totalNotes finalDistribution ≥ 35) →
  (finalDistribution Denomination.Ft5000 = 0) →
  (totalValue initialDistribution = totalValue finalDistribution) →
  (totalValue finalDistribution = 29950) :=
by sorry


end NUMINAMATH_CALUDE_cashier_problem_l2478_247812


namespace NUMINAMATH_CALUDE_x_value_theorem_l2478_247811

theorem x_value_theorem (x y : ℝ) :
  (x / (x + 2) = (y^2 + 3*y - 2) / (y^2 + 3*y + 1)) →
  x = (2*y^2 + 6*y - 4) / 3 := by
sorry

end NUMINAMATH_CALUDE_x_value_theorem_l2478_247811


namespace NUMINAMATH_CALUDE_square_perimeter_product_l2478_247841

theorem square_perimeter_product (x y : ℝ) (h1 : x^2 + y^2 = 130) (h2 : x^2 - y^2 = 58) :
  (4*x) * (4*y) = 96 * Real.sqrt 94 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_product_l2478_247841


namespace NUMINAMATH_CALUDE_school_teachers_count_l2478_247826

theorem school_teachers_count (total : ℕ) (sample_size : ℕ) (sample_students : ℕ) : 
  total = 2400 →
  sample_size = 320 →
  sample_students = 280 →
  ∃ (teachers students : ℕ),
    teachers + students = total ∧
    teachers * sample_students = students * (sample_size - sample_students) ∧
    teachers = 300 := by
  sorry

end NUMINAMATH_CALUDE_school_teachers_count_l2478_247826


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l2478_247809

theorem simultaneous_equations_solution :
  ∃ (x y : ℚ), 
    (3 * x - 2 * y = 12) ∧ 
    (9 * y - 6 * x = -18) ∧ 
    (x = 24/5) ∧ 
    (y = 6/5) := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l2478_247809


namespace NUMINAMATH_CALUDE_man_tshirt_count_l2478_247856

/-- Given a man with pants and t-shirts, calculates the number of ways he can dress --/
def dressing_combinations (num_tshirts : ℕ) (num_pants : ℕ) : ℕ :=
  num_tshirts * num_pants

theorem man_tshirt_count :
  ∀ (num_pants : ℕ) (total_combinations : ℕ),
    num_pants = 9 →
    total_combinations = 72 →
    ∃ (num_tshirts : ℕ),
      dressing_combinations num_tshirts num_pants = total_combinations ∧
      num_tshirts = 8 :=
by
  sorry


end NUMINAMATH_CALUDE_man_tshirt_count_l2478_247856


namespace NUMINAMATH_CALUDE_range_of_a_l2478_247896

/-- The range of non-negative real number a that satisfies the given conditions -/
theorem range_of_a (a : ℝ) : 
  (∀ n : ℕ, n ≥ 1 → ∃ (an bn : ℝ), bn = an^3) →  -- Points are on y = x^3
  (∃ (a1 : ℝ), a1 = a ∧ a ≥ 0) →  -- a1 = a and a ≥ 0
  (∀ n : ℕ, n ≥ 1 → ∃ (cn : ℝ), cn = an + an+1) →  -- cn = an + an+1
  (∀ n : ℕ, n ≥ 1 → ∃ (cn an : ℝ), cn = 1/2 * an + 3/2) →  -- cn = 1/2*an + 3/2
  (∀ n : ℕ, an ≠ 1) →  -- All terms of {an} are not equal to 1
  (∀ n : ℕ, n ≥ 1 → ∃ (kn : ℝ), kn = (bn - 1) / (an - 1)) →  -- kn = (bn - 1) / (an - 1)
  (∃ (k0 : ℝ), ∀ n : ℕ, n ≥ 1 → (kn - k0) * (kn+1 - k0) < 0) →  -- Existence of k0
  (0 ≤ a ∧ a < 7 ∧ a ≠ 1) :=
by
  sorry


end NUMINAMATH_CALUDE_range_of_a_l2478_247896


namespace NUMINAMATH_CALUDE_hostel_provisions_l2478_247885

theorem hostel_provisions (initial_men : ℕ) (left_men : ℕ) (remaining_days : ℕ) :
  initial_men = 250 →
  left_men = 50 →
  remaining_days = 45 →
  (initial_men : ℚ) * (initial_men - left_men : ℚ)⁻¹ * remaining_days = 36 :=
by sorry

end NUMINAMATH_CALUDE_hostel_provisions_l2478_247885


namespace NUMINAMATH_CALUDE_simplify_expression_l2478_247872

theorem simplify_expression (x y : ℝ) : 8*y + 15 - 3*y + 20 + 2*x = 5*y + 2*x + 35 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2478_247872


namespace NUMINAMATH_CALUDE_inequality_solution_l2478_247830

theorem inequality_solution (m n : ℤ) : 
  (∀ x : ℝ, x > 0 → (m * x + 5) * (x^2 - n) ≤ 0) →
  (m + n ∈ ({-4, 24} : Set ℤ)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2478_247830


namespace NUMINAMATH_CALUDE_range_of_m_l2478_247803

theorem range_of_m (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x < 0 ∧ y < 0 ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) ∨ 
  (∀ x : ℝ, 4*x^2 + 4*(m - 2)*x + 1 ≠ 0) →
  ¬(∃ x y : ℝ, x ≠ y ∧ x < 0 ∧ y < 0 ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) →
  (∃ x : ℝ, 4*x^2 + 4*(m - 2)*x + 1 = 0) →
  (m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2478_247803


namespace NUMINAMATH_CALUDE_multiples_17_sums_l2478_247802

/-- The sum of the first n positive integers -/
def sum_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of the squares of the first n positive integers -/
def sum_squares_n (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- The sum of the first twelve positive multiples of 17 -/
def sum_multiples_17 : ℕ := 17 * sum_n 12

/-- The sum of the squares of the first twelve positive multiples of 17 -/
def sum_squares_multiples_17 : ℕ := 17^2 * sum_squares_n 12

theorem multiples_17_sums :
  sum_multiples_17 = 1326 ∧ sum_squares_multiples_17 = 187850 := by
  sorry

end NUMINAMATH_CALUDE_multiples_17_sums_l2478_247802


namespace NUMINAMATH_CALUDE_arithmetic_sequence_with_geometric_subset_l2478_247892

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (a b c : ℝ) : Prop :=
  b ^ 2 = a * c

theorem arithmetic_sequence_with_geometric_subset (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 = 1 →
  is_geometric_sequence (a 1) (a 3) (a 9) →
  (∀ n : ℕ, a n = n) ∨ (∀ n : ℕ, a n = 1) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_with_geometric_subset_l2478_247892


namespace NUMINAMATH_CALUDE_square_area_to_perimeter_ratio_l2478_247861

theorem square_area_to_perimeter_ratio (s₁ s₂ : ℝ) (h : s₁^2 / s₂^2 = 16 / 81) :
  (4 * s₁) / (4 * s₂) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_square_area_to_perimeter_ratio_l2478_247861


namespace NUMINAMATH_CALUDE_zoo_animals_ratio_l2478_247810

theorem zoo_animals_ratio (snakes monkeys lions pandas dogs : ℕ) : 
  snakes = 15 →
  monkeys = 2 * snakes →
  lions = monkeys - 5 →
  pandas = lions + 8 →
  snakes + monkeys + lions + pandas + dogs = 114 →
  dogs * 3 = pandas := by
sorry

end NUMINAMATH_CALUDE_zoo_animals_ratio_l2478_247810


namespace NUMINAMATH_CALUDE_shekar_science_marks_l2478_247868

/-- Calculates the marks in science given other subject marks and the average -/
def calculate_science_marks (math social english biology average : ℕ) : ℕ :=
  5 * average - (math + social + english + biology)

/-- Proves that Shekar's science marks are 65 given his other marks and average -/
theorem shekar_science_marks :
  let math := 76
  let social := 82
  let english := 62
  let biology := 85
  let average := 74
  calculate_science_marks math social english biology average = 65 := by
  sorry

#eval calculate_science_marks 76 82 62 85 74

end NUMINAMATH_CALUDE_shekar_science_marks_l2478_247868


namespace NUMINAMATH_CALUDE_x_range_l2478_247875

/-- An acute triangle with side lengths 2, 3, and x, where x is a positive real number -/
structure AcuteTriangle where
  x : ℝ
  x_pos : x > 0
  acute : x^2 < 2^2 + 3^2  -- Condition for the triangle to be acute

/-- The range of x in an acute triangle with side lengths 2, 3, and x -/
theorem x_range (t : AcuteTriangle) : Real.sqrt 5 < t.x ∧ t.x < Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l2478_247875


namespace NUMINAMATH_CALUDE_star_value_proof_l2478_247823

theorem star_value_proof (star : ℝ) : 
  45 - (28 - (37 - (15 - star^2))) = 59 → star = 2 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_star_value_proof_l2478_247823


namespace NUMINAMATH_CALUDE_triangle_altitude_segment_l2478_247870

/-- Given a triangle with sides 35, 85, and 90 units, prove that when an altitude is dropped on the side of length 90, the length of the larger segment cut off by the altitude is 78.33 units. -/
theorem triangle_altitude_segment (a b c : ℝ) (h1 : a = 35) (h2 : b = 85) (h3 : c = 90) :
  let x := (c^2 + a^2 - b^2) / (2 * c)
  c - x = 78.33 := by sorry

end NUMINAMATH_CALUDE_triangle_altitude_segment_l2478_247870


namespace NUMINAMATH_CALUDE_grace_mowing_hours_l2478_247894

/-- Represents the rates and hours worked by Grace in her landscaping business -/
structure LandscapingWork where
  mowing_rate : ℕ
  weeding_rate : ℕ
  mulching_rate : ℕ
  weeding_hours : ℕ
  mulching_hours : ℕ
  total_earnings : ℕ

/-- Calculates the number of hours spent mowing lawns given the landscaping work details -/
def mowing_hours (work : LandscapingWork) : ℕ :=
  (work.total_earnings - (work.weeding_rate * work.weeding_hours + work.mulching_rate * work.mulching_hours)) / work.mowing_rate

/-- Theorem stating that Grace spent 63 hours mowing lawns in September -/
theorem grace_mowing_hours :
  let work : LandscapingWork := {
    mowing_rate := 6,
    weeding_rate := 11,
    mulching_rate := 9,
    weeding_hours := 9,
    mulching_hours := 10,
    total_earnings := 567
  }
  mowing_hours work = 63 := by sorry

end NUMINAMATH_CALUDE_grace_mowing_hours_l2478_247894


namespace NUMINAMATH_CALUDE_long_distance_bill_calculation_l2478_247835

-- Define the constants
def monthly_fee : ℚ := 2
def per_minute_rate : ℚ := 12 / 100
def minutes_used : ℕ := 178

-- Define the theorem
theorem long_distance_bill_calculation :
  monthly_fee + per_minute_rate * minutes_used = 23.36 := by
  sorry

end NUMINAMATH_CALUDE_long_distance_bill_calculation_l2478_247835


namespace NUMINAMATH_CALUDE_fraction_simplification_l2478_247824

theorem fraction_simplification (x : ℝ) : (2*x - 3) / 4 + (4*x + 5) / 3 = (22*x + 11) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2478_247824


namespace NUMINAMATH_CALUDE_vasya_number_digits_l2478_247871

theorem vasya_number_digits (x : ℝ) (h_pos : x > 0) 
  (h_kolya : 10^8 ≤ x^3 ∧ x^3 < 10^9) (h_petya : 10^10 ≤ x^4 ∧ x^4 < 10^11) :
  10^32 ≤ x^12 ∧ x^12 < 10^33 := by
  sorry

end NUMINAMATH_CALUDE_vasya_number_digits_l2478_247871


namespace NUMINAMATH_CALUDE_total_subjects_l2478_247832

theorem total_subjects (avg_all : ℝ) (avg_first_five : ℝ) (last_subject : ℝ) 
  (h1 : avg_all = 78)
  (h2 : avg_first_five = 74)
  (h3 : last_subject = 98) :
  ∃ n : ℕ, n = 6 ∧ 
    n * avg_all = (n - 1) * avg_first_five + last_subject :=
by sorry

end NUMINAMATH_CALUDE_total_subjects_l2478_247832


namespace NUMINAMATH_CALUDE_probability_of_king_is_one_thirteenth_l2478_247876

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)
  (kings : ℕ)

/-- The probability of drawing a specific card type from a deck -/
def probability_of_draw (deck : Deck) (favorable_outcomes : ℕ) : ℚ :=
  favorable_outcomes / deck.total_cards

/-- Theorem: The probability of drawing a King from a standard deck is 1/13 -/
theorem probability_of_king_is_one_thirteenth (deck : Deck) 
  (h1 : deck.total_cards = 52)
  (h2 : deck.ranks = 13)
  (h3 : deck.suits = 4)
  (h4 : deck.kings = 4) :
  probability_of_draw deck deck.kings = 1 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_king_is_one_thirteenth_l2478_247876


namespace NUMINAMATH_CALUDE_sum_of_squares_l2478_247808

theorem sum_of_squares (x y z : ℤ) 
  (sum_eq : x + y + z = 3) 
  (sum_cubes_eq : x^3 + y^3 + z^3 = 3) : 
  x^2 + y^2 + z^2 = 3 ∨ x^2 + y^2 + z^2 = 57 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2478_247808


namespace NUMINAMATH_CALUDE_andrew_eggs_l2478_247891

/-- The number of eggs Andrew ends up with after buying more -/
def total_eggs (initial : ℕ) (bought : ℕ) : ℕ := initial + bought

/-- Theorem: Andrew ends up with 70 eggs when starting with 8 and buying 62 more -/
theorem andrew_eggs : total_eggs 8 62 = 70 := by
  sorry

end NUMINAMATH_CALUDE_andrew_eggs_l2478_247891


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2478_247846

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^4 = (x^2 + 7*x + 2) * q + (-315*x - 94) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2478_247846


namespace NUMINAMATH_CALUDE_simplify_polynomial_l2478_247897

theorem simplify_polynomial (x : ℝ) : 
  x * (4 * x^3 - 3) - 6 * (x^2 - 3*x + 9) = 4 * x^4 - 6 * x^2 + 15 * x - 54 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l2478_247897


namespace NUMINAMATH_CALUDE_a_4_equals_28_l2478_247821

def S (n : ℕ) : ℕ := 4 * n^2

def a (n : ℕ) : ℤ := S n - S (n - 1)

theorem a_4_equals_28 : a 4 = 28 := by
  sorry

end NUMINAMATH_CALUDE_a_4_equals_28_l2478_247821


namespace NUMINAMATH_CALUDE_victors_lives_l2478_247893

theorem victors_lives (lost : ℕ) (diff : ℕ) (current : ℕ) : 
  lost = 14 → diff = 12 → lost - current = diff → current = 2 := by
  sorry

end NUMINAMATH_CALUDE_victors_lives_l2478_247893


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2478_247836

/-- Given a triangle with inradius 2.5 cm and area 45 cm², its perimeter is 36 cm. -/
theorem triangle_perimeter (inradius : ℝ) (area : ℝ) (perimeter : ℝ) : 
  inradius = 2.5 → area = 45 → perimeter = 36 := by
  sorry

#check triangle_perimeter

end NUMINAMATH_CALUDE_triangle_perimeter_l2478_247836


namespace NUMINAMATH_CALUDE_max_combined_power_l2478_247858

theorem max_combined_power (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ < 1) (h₂ : x₂ < 1) (h₃ : x₃ < 1)
  (h : 2 * (x₁ + x₂ + x₃) + 4 * x₁ * x₂ * x₃ = 3 * (x₁ * x₂ + x₁ * x₃ + x₂ * x₃) + 1) :
  x₁ + x₂ + x₃ ≤ 3/4 := by
sorry

end NUMINAMATH_CALUDE_max_combined_power_l2478_247858


namespace NUMINAMATH_CALUDE_students_with_b_in_dawsons_class_l2478_247849

theorem students_with_b_in_dawsons_class 
  (charles_total : ℕ) 
  (charles_b : ℕ) 
  (dawson_total : ℕ) 
  (h1 : charles_total = 20)
  (h2 : charles_b = 12)
  (h3 : dawson_total = 30)
  (h4 : charles_b * dawson_total = charles_total * dawson_b) :
  dawson_b = 18 := by
    sorry

#check students_with_b_in_dawsons_class

end NUMINAMATH_CALUDE_students_with_b_in_dawsons_class_l2478_247849


namespace NUMINAMATH_CALUDE_old_clock_slow_12_minutes_l2478_247866

/-- Represents the time interval between hand overlaps on the old clock -/
def old_clock_overlap_interval : ℚ := 66

/-- Represents the standard time interval between hand overlaps -/
def standard_overlap_interval : ℚ := 720 / 11

/-- Represents the number of minutes in a standard day -/
def standard_day_minutes : ℕ := 24 * 60

/-- Theorem stating that the old clock is 12 minutes slow over a 24-hour period -/
theorem old_clock_slow_12_minutes :
  (standard_day_minutes : ℚ) / standard_overlap_interval * old_clock_overlap_interval
  - standard_day_minutes = 12 := by sorry

end NUMINAMATH_CALUDE_old_clock_slow_12_minutes_l2478_247866


namespace NUMINAMATH_CALUDE_log_46328_between_consecutive_integers_l2478_247898

theorem log_46328_between_consecutive_integers :
  ∃ (a b : ℤ), a + 1 = b ∧ (a : ℝ) < Real.log 46328 / Real.log 10 ∧ (Real.log 46328 / Real.log 10 < b) ∧ a + b = 9 := by
  sorry

end NUMINAMATH_CALUDE_log_46328_between_consecutive_integers_l2478_247898


namespace NUMINAMATH_CALUDE_B_highest_score_l2478_247857

-- Define the structure for an applicant
structure Applicant where
  name : String
  knowledge : ℕ
  experience : ℕ
  language : ℕ

-- Define the weighting function
def weightedScore (a : Applicant) : ℚ :=
  (5 * a.knowledge + 2 * a.experience + 3 * a.language) / 10

-- Define the applicants
def A : Applicant := ⟨"A", 75, 80, 80⟩
def B : Applicant := ⟨"B", 85, 80, 70⟩
def C : Applicant := ⟨"C", 70, 78, 70⟩

-- Theorem stating that B has the highest weighted score
theorem B_highest_score :
  weightedScore B > weightedScore A ∧ weightedScore B > weightedScore C :=
by sorry

end NUMINAMATH_CALUDE_B_highest_score_l2478_247857


namespace NUMINAMATH_CALUDE_line_perp_plane_parallel_plane_implies_planes_perp_planes_perp_parallel_implies_perp_l2478_247834

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (linePerpendicular : Line → Plane → Prop)
variable (lineParallel : Line → Plane → Prop)

-- Theorem 1
theorem line_perp_plane_parallel_plane_implies_planes_perp
  (α β : Plane) (l : Line)
  (h1 : linePerpendicular l α)
  (h2 : lineParallel l β) :
  perpendicular α β :=
sorry

-- Theorem 2
theorem planes_perp_parallel_implies_perp
  (α β γ : Plane)
  (h1 : perpendicular α β)
  (h2 : parallel α γ) :
  perpendicular γ β :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_parallel_plane_implies_planes_perp_planes_perp_parallel_implies_perp_l2478_247834


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2478_247848

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y + f (x + y) = x * y

/-- The theorem stating that the only functions satisfying the equation are x - 1 or -x - 1 -/
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesEquation f → (∀ x : ℝ, f x = x - 1 ∨ f x = -x - 1) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2478_247848


namespace NUMINAMATH_CALUDE_triangle_agw_area_l2478_247845

/-- Right triangle ABC with squares on legs and intersecting lines -/
structure RightTriangleWithSquares where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  W : ℝ × ℝ
  -- Conditions
  right_angle : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0
  ac_length : (A.1 - C.1)^2 + (A.2 - C.2)^2 = 14^2
  bc_length : (B.1 - C.1)^2 + (B.2 - C.2)^2 = 28^2
  square_acde : D = (A.1 + C.2 - C.1, A.2 - C.2 + C.1) ∧ E = (C.1, A.2)
  square_cbfg : F = (C.1, B.2) ∧ G = (B.1, B.2 + B.1 - C.1)
  w_on_bc : ∃ t : ℝ, W = (t * B.1 + (1 - t) * C.1, t * B.2 + (1 - t) * C.2)
  w_on_af : ∃ s : ℝ, W = (s * A.1 + (1 - s) * F.1, s * A.2 + (1 - s) * F.2)

/-- The area of triangle AGW is 196 -/
theorem triangle_agw_area (t : RightTriangleWithSquares) : 
  abs ((t.A.1 * (t.G.2 - t.W.2) + t.G.1 * (t.W.2 - t.A.2) + t.W.1 * (t.A.2 - t.G.2)) / 2) = 196 := by
  sorry

end NUMINAMATH_CALUDE_triangle_agw_area_l2478_247845


namespace NUMINAMATH_CALUDE_ted_age_l2478_247853

/-- Given that Ted's age is 20 years less than three times Sally's age,
    and the sum of their ages is 70, prove that Ted is 47.5 years old. -/
theorem ted_age (sally_age : ℝ) (ted_age : ℝ) 
  (h1 : ted_age = 3 * sally_age - 20)
  (h2 : ted_age + sally_age = 70) : 
  ted_age = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_ted_age_l2478_247853


namespace NUMINAMATH_CALUDE_tournament_has_cycle_of_length_3_l2478_247828

/-- A tournament is a complete directed graph where each edge represents a match outcome. -/
def Tournament (n : ℕ) := Fin n → Fin n → Prop

/-- In a valid tournament, every pair of distinct players has exactly one match outcome. -/
def is_valid_tournament (t : Tournament n) : Prop :=
  ∀ i j : Fin n, i ≠ j → (t i j ∧ ¬t j i) ∨ (t j i ∧ ¬t i j)

/-- A player wins at least one match if there exists another player they defeated. -/
def player_wins_at_least_one (t : Tournament n) (i : Fin n) : Prop :=
  ∃ j : Fin n, t i j

/-- A cycle of length 3 in a tournament. -/
def has_cycle_of_length_3 (t : Tournament n) : Prop :=
  ∃ a b c : Fin n, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ t a b ∧ t b c ∧ t c a

theorem tournament_has_cycle_of_length_3 :
  ∀ (t : Tournament 12),
    is_valid_tournament t →
    (∀ i : Fin 12, player_wins_at_least_one t i) →
    has_cycle_of_length_3 t :=
by sorry


end NUMINAMATH_CALUDE_tournament_has_cycle_of_length_3_l2478_247828


namespace NUMINAMATH_CALUDE_unique_square_divisible_by_three_in_range_l2478_247801

theorem unique_square_divisible_by_three_in_range : ∃! y : ℕ,
  50 < y ∧ y < 120 ∧ ∃ x : ℕ, y = x^2 ∧ y % 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_square_divisible_by_three_in_range_l2478_247801


namespace NUMINAMATH_CALUDE_thick_line_segments_length_l2478_247880

theorem thick_line_segments_length
  (perimeter_quadrilaterals : ℝ)
  (perimeter_triangles : ℝ)
  (perimeter_large_triangle : ℝ)
  (h1 : perimeter_quadrilaterals = 25)
  (h2 : perimeter_triangles = 20)
  (h3 : perimeter_large_triangle = 19) :
  (perimeter_quadrilaterals + perimeter_triangles - perimeter_large_triangle) / 2 = 13 :=
sorry

end NUMINAMATH_CALUDE_thick_line_segments_length_l2478_247880


namespace NUMINAMATH_CALUDE_total_earnings_is_1025_l2478_247883

/-- Represents the sales data for a single day -/
structure DailySales where
  size8 : Nat
  size12 : Nat
  size16 : Nat
  size20 : Nat
  size24 : Nat

/-- Calculates the earnings for a single day -/
def dailyEarnings (sales : DailySales) (basePrice : Rat) : Rat :=
  sales.size8 * basePrice +
  sales.size12 * (2.5 * basePrice) +
  sales.size16 * (3 * basePrice) +
  sales.size20 * (4 * basePrice) +
  sales.size24 * (5.5 * basePrice)

/-- Represents the sales data for a week -/
def weeklySales : List DailySales :=
  [
    { size8 := 3, size12 := 2, size16 := 1, size20 := 2, size24 := 1 },
    { size8 := 5, size12 := 1, size16 := 4, size20 := 0, size24 := 2 },
    { size8 := 4, size12 := 3, size16 := 3, size20 := 1, size24 := 0 },
    { size8 := 2, size12 := 2, size16 := 2, size20 := 1, size24 := 3 },
    { size8 := 6, size12 := 4, size16 := 2, size20 := 2, size24 := 0 },
    { size8 := 1, size12 := 3, size16 := 3, size20 := 4, size24 := 2 },
    { size8 := 3, size12 := 2, size16 := 4, size20 := 3, size24 := 1 }
  ]

theorem total_earnings_is_1025 :
  (weeklySales.map (fun sales => dailyEarnings sales 5)).sum = 1025 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_is_1025_l2478_247883


namespace NUMINAMATH_CALUDE_magnitude_of_z_l2478_247833

open Complex

theorem magnitude_of_z : ∃ z : ℂ, z = 1 + 2*I + I^3 ∧ abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l2478_247833


namespace NUMINAMATH_CALUDE_perfect_square_with_three_or_fewer_swaps_l2478_247886

/-- Represents a permutation of digits --/
def Permutation := List Nat

/-- Checks if a permutation represents a perfect square --/
def is_perfect_square (p : Permutation) : Prop :=
  ∃ n : Nat, n * n = p.foldl (fun acc d => acc * 10 + d) 0

/-- Counts the number of swaps needed to transform one permutation into another --/
def swap_count (p1 p2 : Permutation) : Nat :=
  sorry

/-- The initial permutation of digits from 1 to 9 --/
def initial_permutation : Permutation := [1, 2, 3, 4, 5, 6, 7, 8, 9]

/-- Theorem: There exists a permutation of digits 1-9 that forms a perfect square 
    and can be achieved with 3 or fewer swaps from the initial permutation --/
theorem perfect_square_with_three_or_fewer_swaps :
  ∃ (final_perm : Permutation), 
    is_perfect_square final_perm ∧ 
    swap_count initial_permutation final_perm ≤ 3 :=
  sorry

end NUMINAMATH_CALUDE_perfect_square_with_three_or_fewer_swaps_l2478_247886


namespace NUMINAMATH_CALUDE_student_venue_arrangements_l2478_247864

theorem student_venue_arrangements (n : Nat) (a b c : Nat) 
  (h1 : n = 6)
  (h2 : a = 3)
  (h3 : b = 1)
  (h4 : c = 2)
  (h5 : a + b + c = n) :
  Nat.choose n a * Nat.choose (n - a) b * Nat.choose (n - a - b) c = 60 :=
by sorry

end NUMINAMATH_CALUDE_student_venue_arrangements_l2478_247864


namespace NUMINAMATH_CALUDE_constant_sum_of_powers_l2478_247887

theorem constant_sum_of_powers (n : ℕ+) :
  (∀ x y z : ℝ, x + y + z = 0 → x * y * z = 1 → 
    ∃ c : ℝ, ∀ a b d : ℝ, a + b + d = 0 → a * b * d = 1 → 
      a^(n : ℕ) + b^(n : ℕ) + d^(n : ℕ) = c) ↔ n = 1 ∨ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_constant_sum_of_powers_l2478_247887


namespace NUMINAMATH_CALUDE_hcf_of_ratio_and_lcm_l2478_247869

/-- 
Given three positive integers a, b, and c that are in the ratio 3:4:5 and 
have a least common multiple of 2400, prove that their highest common factor is 20.
-/
theorem hcf_of_ratio_and_lcm (a b c : ℕ+) 
  (h_ratio : ∃ (k : ℕ+), a = 3 * k ∧ b = 4 * k ∧ c = 5 * k)
  (h_lcm : Nat.lcm a (Nat.lcm b c) = 2400) :
  Nat.gcd a (Nat.gcd b c) = 20 := by
  sorry

end NUMINAMATH_CALUDE_hcf_of_ratio_and_lcm_l2478_247869


namespace NUMINAMATH_CALUDE_quadratic_inverse_sum_l2478_247819

/-- A quadratic function with real coefficients -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The inverse of a quadratic function -/
def InverseQuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ c * x^2 + b * x + a

theorem quadratic_inverse_sum (a b c : ℝ) :
  (∀ x, QuadraticFunction a b c (InverseQuadraticFunction a b c x) = x) →
  a + c = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inverse_sum_l2478_247819


namespace NUMINAMATH_CALUDE_petrol_price_increase_l2478_247878

theorem petrol_price_increase (original_price : ℝ) (original_consumption : ℝ) : 
  let consumption_reduction : ℝ := 0.2857142857142857
  let new_consumption : ℝ := original_consumption * (1 - consumption_reduction)
  let new_price : ℝ := original_price * original_consumption / new_consumption
  new_price / original_price - 1 = 0.4 := by sorry

end NUMINAMATH_CALUDE_petrol_price_increase_l2478_247878


namespace NUMINAMATH_CALUDE_pufferfish_count_swordfish_pufferfish_relation_total_fish_count_l2478_247806

/-- The number of pufferfish in an aquarium exhibit -/
def num_pufferfish : ℕ := 15

/-- The number of swordfish in an aquarium exhibit -/
def num_swordfish : ℕ := 5 * num_pufferfish

/-- The total number of fish in the aquarium exhibit -/
def total_fish : ℕ := 90

/-- Theorem stating that the number of pufferfish is 15 -/
theorem pufferfish_count : num_pufferfish = 15 := by sorry

/-- Theorem verifying the relationship between swordfish and pufferfish -/
theorem swordfish_pufferfish_relation : num_swordfish = 5 * num_pufferfish := by sorry

/-- Theorem verifying the total number of fish -/
theorem total_fish_count : num_swordfish + num_pufferfish = total_fish := by sorry

end NUMINAMATH_CALUDE_pufferfish_count_swordfish_pufferfish_relation_total_fish_count_l2478_247806


namespace NUMINAMATH_CALUDE_adjacent_rectangles_area_l2478_247805

/-- The total area of two adjacent rectangles -/
theorem adjacent_rectangles_area 
  (u v w z : Real) 
  (hu : u > 0) 
  (hv : v > 0) 
  (hw : w > 0) 
  (hz : z > w) : 
  let first_rectangle := (u + v) * w
  let second_rectangle := (u + v) * (z - w)
  first_rectangle + second_rectangle = (u + v) * z :=
by sorry

end NUMINAMATH_CALUDE_adjacent_rectangles_area_l2478_247805


namespace NUMINAMATH_CALUDE_fourth_vertex_of_rectangle_l2478_247807

/-- Given a circle and three points forming part of a rectangle, 
    this theorem proves the coordinates of the fourth vertex. -/
theorem fourth_vertex_of_rectangle 
  (O : ℝ × ℝ) (R : ℝ) 
  (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) 
  (h_circle : (x₁ - O.1)^2 + (y₁ - O.2)^2 = R^2 ∧ (x₂ - O.1)^2 + (y₂ - O.2)^2 = R^2)
  (h_inside : (x₀ - O.1)^2 + (y₀ - O.2)^2 < R^2) :
  ∃ (x₄ y₄ : ℝ), 
    (x₄ = x₁ + x₂ - x₀ ∧ y₄ = y₁ + y₂ - y₀) ∧
    ((x₄ - O.1)^2 + (y₄ - O.2)^2 = R^2) ∧
    ((x₄ - x₀)^2 + (y₄ - y₀)^2 = (x₁ - x₂)^2 + (y₁ - y₂)^2) :=
by sorry

end NUMINAMATH_CALUDE_fourth_vertex_of_rectangle_l2478_247807


namespace NUMINAMATH_CALUDE_number_of_observations_l2478_247863

/-- Given a set of observations with an initial mean, a correction to one observation,
    and a new mean, prove the number of observations. -/
theorem number_of_observations
  (initial_mean : ℝ)
  (wrong_value : ℝ)
  (correct_value : ℝ)
  (new_mean : ℝ)
  (h1 : initial_mean = 36)
  (h2 : wrong_value = 23)
  (h3 : correct_value = 45)
  (h4 : new_mean = 36.5) :
  ∃ n : ℕ, n * initial_mean + (correct_value - wrong_value) = n * new_mean ∧ n = 44 := by
  sorry

end NUMINAMATH_CALUDE_number_of_observations_l2478_247863


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l2478_247877

theorem complex_power_magnitude (z : ℂ) :
  z = (1 / Real.sqrt 2 : ℂ) + (Complex.I / Real.sqrt 2) →
  Complex.abs (z^8) = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l2478_247877


namespace NUMINAMATH_CALUDE_congruence_condition_l2478_247840

/-- A triangle specified by two sides and an angle --/
structure Triangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (angle : ℝ)

/-- Predicate to check if a triangle specification guarantees congruence --/
def guarantees_congruence (t : Triangle) : Prop :=
  t.side1 > 0 ∧ t.side2 > 0 ∧ t.angle > 0 ∧ t.angle < 180 ∧
  (t.angle > 90 ∨ (t.angle = 90 ∧ t.side1 ≠ t.side2))

/-- The triangles from the problem options --/
def triangle_A : Triangle := { side1 := 2, side2 := 0, angle := 60 }
def triangle_B : Triangle := { side1 := 2, side2 := 3, angle := 0 }
def triangle_C : Triangle := { side1 := 3, side2 := 5, angle := 150 }
def triangle_D : Triangle := { side1 := 3, side2 := 2, angle := 30 }

theorem congruence_condition :
  guarantees_congruence triangle_C ∧
  ¬guarantees_congruence triangle_A ∧
  ¬guarantees_congruence triangle_B ∧
  ¬guarantees_congruence triangle_D :=
sorry

end NUMINAMATH_CALUDE_congruence_condition_l2478_247840


namespace NUMINAMATH_CALUDE_apollonius_circle_exists_l2478_247850

-- Define a circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a tangency relation between two circles
def is_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2 ∨
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius - c2.radius)^2

-- Theorem statement
theorem apollonius_circle_exists (S1 S2 S3 : Circle) :
  ∃ S : Circle, is_tangent S S1 ∧ is_tangent S S2 ∧ is_tangent S S3 :=
sorry

end NUMINAMATH_CALUDE_apollonius_circle_exists_l2478_247850


namespace NUMINAMATH_CALUDE_peter_investment_duration_l2478_247831

/-- Calculates the final amount after simple interest -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

theorem peter_investment_duration :
  let principal : ℝ := 710
  let peterFinalAmount : ℝ := 815
  let davidFinalAmount : ℝ := 850
  let davidTime : ℝ := 4
  ∃ (rate : ℝ), 
    (simpleInterest principal rate davidTime = davidFinalAmount) ∧
    (simpleInterest principal rate 3 = peterFinalAmount) := by
  sorry

end NUMINAMATH_CALUDE_peter_investment_duration_l2478_247831


namespace NUMINAMATH_CALUDE_solve_equation_l2478_247881

theorem solve_equation (m n : ℕ) (h1 : ((1^m) / (5^m)) * ((1^n) / (4^n)) = 1 / (2 * (10^31))) (h2 : m = 31) : n = 16 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2478_247881


namespace NUMINAMATH_CALUDE_prism_15_edges_has_7_faces_l2478_247874

/-- A prism is a three-dimensional geometric shape with two identical ends (bases) and flat sides. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism given its number of edges. -/
def num_faces (p : Prism) : ℕ :=
  (p.edges / 3) + 2

/-- Theorem: A prism with 15 edges has 7 faces. -/
theorem prism_15_edges_has_7_faces :
  ∀ (p : Prism), p.edges = 15 → num_faces p = 7 := by
  sorry

#check prism_15_edges_has_7_faces

end NUMINAMATH_CALUDE_prism_15_edges_has_7_faces_l2478_247874


namespace NUMINAMATH_CALUDE_modulus_of_complex_product_l2478_247842

theorem modulus_of_complex_product : Complex.abs ((3 - 4 * Complex.I) * Complex.I) = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_product_l2478_247842


namespace NUMINAMATH_CALUDE_saline_mixture_proof_l2478_247820

def initial_volume : ℝ := 50
def initial_concentration : ℝ := 0.4
def added_concentration : ℝ := 0.1
def final_concentration : ℝ := 0.25
def added_volume : ℝ := 50

theorem saline_mixture_proof :
  (initial_volume * initial_concentration + added_volume * added_concentration) / (initial_volume + added_volume) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_saline_mixture_proof_l2478_247820


namespace NUMINAMATH_CALUDE_assignments_for_thirty_points_l2478_247888

/-- Calculates the number of assignments needed for a given number of points -/
def assignments_needed (points : ℕ) : ℕ :=
  if points ≤ 10 then points
  else if points ≤ 20 then 10 + 2 * (points - 10)
  else 30 + 3 * (points - 20)

/-- Theorem stating that 60 assignments are needed for 30 points -/
theorem assignments_for_thirty_points :
  assignments_needed 30 = 60 := by sorry

end NUMINAMATH_CALUDE_assignments_for_thirty_points_l2478_247888


namespace NUMINAMATH_CALUDE_max_abs_z_value_l2478_247839

theorem max_abs_z_value (a b c z : ℂ) (d : ℝ) 
  (h1 : Complex.abs a = Complex.abs b)
  (h2 : Complex.abs a = (1 / 2) * Complex.abs c)
  (h3 : Complex.abs a > 0)
  (h4 : a * z^2 + b * z + d * c = 0)
  (h5 : d = 1) :
  ∃ (M : ℝ), M = 2 ∧ ∀ z', a * z'^2 + b * z' + d * c = 0 → Complex.abs z' ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_abs_z_value_l2478_247839


namespace NUMINAMATH_CALUDE_triangle_line_equations_l2478_247884

/-- Triangle with vertices A(4,0), B(6,7), and C(0,3) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Definition of the specific triangle -/
def specificTriangle : Triangle :=
  { A := (4, 0),
    B := (6, 7),
    C := (0, 3) }

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem stating the equations of line AC and altitude from B to AB -/
theorem triangle_line_equations (t : Triangle) (t_eq : t = specificTriangle) :
  ∃ (lineAC altitudeB : LineEquation),
    lineAC = { a := 3, b := 4, c := -12 } ∧
    altitudeB = { a := 2, b := 7, c := -21 } := by
  sorry

end NUMINAMATH_CALUDE_triangle_line_equations_l2478_247884


namespace NUMINAMATH_CALUDE_tony_saturday_sandwiches_l2478_247862

/-- The number of sandwiches Tony made on Saturday -/
def sandwiches_on_saturday (
  slices_per_sandwich : ℕ)
  (days_in_week : ℕ)
  (initial_slices : ℕ)
  (remaining_slices : ℕ)
  (sandwiches_per_day : ℕ) : ℕ :=
  ((initial_slices - remaining_slices) - (days_in_week - 1) * sandwiches_per_day * slices_per_sandwich) / slices_per_sandwich

theorem tony_saturday_sandwiches :
  sandwiches_on_saturday 2 6 22 6 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tony_saturday_sandwiches_l2478_247862
