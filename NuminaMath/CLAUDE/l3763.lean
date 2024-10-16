import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_roots_existence_l3763_376358

theorem quadratic_roots_existence :
  (∀ x : ℝ, x^2 - x + 1 ≠ 0) ∧
  (∃ x : ℝ, x*(x-1) = 0) ∧
  (∃ x : ℝ, x^2 + 12*x = 0) ∧
  (∃ x : ℝ, x^2 + x = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_existence_l3763_376358


namespace NUMINAMATH_CALUDE_range_of_a_l3763_376387

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + 1 > 0) ∧ 
  (∀ x : ℝ, a*x^2 + x - 1 ≤ 0) → 
  -2 < a ∧ a ≤ -1/4 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3763_376387


namespace NUMINAMATH_CALUDE_bert_kangaroo_count_l3763_376397

/-- The number of kangaroos Kameron has -/
def kameron_kangaroos : ℕ := 100

/-- The number of kangaroos Bert buys per day -/
def bert_daily_increase : ℕ := 2

/-- The number of days until Bert has the same number of kangaroos as Kameron -/
def days_until_equal : ℕ := 40

/-- The number of kangaroos Bert currently has -/
def bert_current_kangaroos : ℕ := 20

theorem bert_kangaroo_count :
  bert_current_kangaroos + bert_daily_increase * days_until_equal = kameron_kangaroos :=
sorry

end NUMINAMATH_CALUDE_bert_kangaroo_count_l3763_376397


namespace NUMINAMATH_CALUDE_purely_imaginary_fraction_l3763_376332

theorem purely_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (Complex.I * b : ℂ) = (2 * Complex.I - 1) / (1 + a * Complex.I)) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_fraction_l3763_376332


namespace NUMINAMATH_CALUDE_apples_to_eat_raw_l3763_376324

theorem apples_to_eat_raw (total : ℕ) (wormy : ℕ) (bruised : ℕ) : 
  total = 85 →
  wormy = total / 5 →
  bruised = wormy + 9 →
  total - bruised - wormy = 42 :=
by
  sorry

end NUMINAMATH_CALUDE_apples_to_eat_raw_l3763_376324


namespace NUMINAMATH_CALUDE_blue_eyed_students_l3763_376328

theorem blue_eyed_students (total : ℕ) (both : ℕ) (neither : ℕ) : 
  total = 40 →
  both = 8 →
  neither = 5 →
  ∃ (blue : ℕ), 
    blue + (3 * blue - both) + both + neither = total ∧
    blue = 10 := by
  sorry

end NUMINAMATH_CALUDE_blue_eyed_students_l3763_376328


namespace NUMINAMATH_CALUDE_company_average_salary_l3763_376333

/-- Calculates the average salary for a company given the number of managers,
    number of associates, average salary of managers, and average salary of associates. -/
def average_company_salary (num_managers : ℕ) (num_associates : ℕ) (avg_salary_managers : ℚ) (avg_salary_associates : ℚ) : ℚ :=
  let total_employees := num_managers + num_associates
  let total_salary := num_managers * avg_salary_managers + num_associates * avg_salary_associates
  total_salary / total_employees

/-- Theorem stating that the average salary for the company is $40,000 -/
theorem company_average_salary :
  average_company_salary 15 75 90000 30000 = 40000 := by
  sorry

end NUMINAMATH_CALUDE_company_average_salary_l3763_376333


namespace NUMINAMATH_CALUDE_dog_park_ratio_l3763_376301

theorem dog_park_ratio (total_dogs : ℕ) (spotted_dogs : ℕ) (pointy_ear_dogs : ℕ) :
  spotted_dogs = total_dogs / 2 →
  spotted_dogs = 15 →
  pointy_ear_dogs = 6 →
  (pointy_ear_dogs : ℚ) / total_dogs = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_dog_park_ratio_l3763_376301


namespace NUMINAMATH_CALUDE_square_difference_of_integers_l3763_376398

theorem square_difference_of_integers (a b : ℕ) 
  (h1 : a + b = 60) 
  (h2 : a - b = 16) : 
  a^2 - b^2 = 960 := by
sorry

end NUMINAMATH_CALUDE_square_difference_of_integers_l3763_376398


namespace NUMINAMATH_CALUDE_equal_cost_at_150_miles_unique_equal_cost_mileage_l3763_376374

-- Define the cost functions for both rental companies
def safety_cost (m : ℝ) : ℝ := 41.95 + 0.29 * m
def city_cost (m : ℝ) : ℝ := 38.95 + 0.31 * m

-- Theorem stating that the costs are equal at 150 miles
theorem equal_cost_at_150_miles : 
  safety_cost 150 = city_cost 150 := by
  sorry

-- Theorem stating that 150 miles is the unique solution
theorem unique_equal_cost_mileage :
  ∀ m : ℝ, safety_cost m = city_cost m ↔ m = 150 := by
  sorry

end NUMINAMATH_CALUDE_equal_cost_at_150_miles_unique_equal_cost_mileage_l3763_376374


namespace NUMINAMATH_CALUDE_ellipse_axes_l3763_376311

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  2 * x^2 + y^2 - 12 = 2 * x + 4 * y

-- Define the standard form of the ellipse
def standard_form (a b h k : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

-- Theorem stating the semi-major and semi-minor axes of the ellipse
theorem ellipse_axes :
  ∃ h k : ℝ, 
    (∀ x y : ℝ, ellipse_equation x y ↔ standard_form 17 8.5 h k x y) ∧
    (17 > 8.5) :=
sorry

end NUMINAMATH_CALUDE_ellipse_axes_l3763_376311


namespace NUMINAMATH_CALUDE_cost_price_calculation_l3763_376343

def selling_price : ℝ := 24000

def discount_rate : ℝ := 0.1

def profit_rate : ℝ := 0.08

theorem cost_price_calculation (cp : ℝ) : 
  cp = 20000 ↔ 
  (selling_price * (1 - discount_rate) = cp * (1 + profit_rate)) ∧
  (selling_price > 0) ∧ 
  (cp > 0) :=
sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l3763_376343


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3763_376317

/-- Given a boat that travels 8 km downstream and 2 km upstream in one hour,
    its speed in still water is 5 km/hr. -/
theorem boat_speed_in_still_water : ∀ (b s : ℝ),
  b + s = 8 →  -- Speed downstream
  b - s = 2 →  -- Speed upstream
  b = 5 :=     -- Speed in still water
by
  sorry

#check boat_speed_in_still_water

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3763_376317


namespace NUMINAMATH_CALUDE_income_ratio_proof_l3763_376384

def uma_income : ℕ := 20000
def bala_income : ℕ := 15000
def uma_savings : ℕ := 5000
def bala_savings : ℕ := 5000
def expenditure_ratio : Rat := 3 / 2

theorem income_ratio_proof :
  let uma_expenditure := uma_income - uma_savings
  let bala_expenditure := bala_income - bala_savings
  (uma_expenditure : Rat) / bala_expenditure = expenditure_ratio →
  (uma_income : Rat) / bala_income = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_income_ratio_proof_l3763_376384


namespace NUMINAMATH_CALUDE_correct_calculation_l3763_376373

theorem correct_calculation (x : ℝ) : 14 * x = 70 → x - 6 = -1 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3763_376373


namespace NUMINAMATH_CALUDE_faye_pencils_count_l3763_376304

theorem faye_pencils_count (rows : ℕ) (pencils_per_row : ℕ) 
  (h1 : rows = 14) (h2 : pencils_per_row = 11) : 
  rows * pencils_per_row = 154 := by
  sorry

end NUMINAMATH_CALUDE_faye_pencils_count_l3763_376304


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l3763_376395

theorem smallest_lcm_with_gcd_5 :
  ∃ (k l : ℕ), 
    1000 ≤ k ∧ k < 10000 ∧
    1000 ≤ l ∧ l < 10000 ∧
    Nat.gcd k l = 5 ∧
    Nat.lcm k l = 203010 ∧
    ∀ (m n : ℕ), 
      1000 ≤ m ∧ m < 10000 ∧
      1000 ≤ n ∧ n < 10000 ∧
      Nat.gcd m n = 5 →
      Nat.lcm m n ≥ 203010 :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l3763_376395


namespace NUMINAMATH_CALUDE_min_value_of_function_l3763_376320

theorem min_value_of_function (x : ℝ) : 
  (x^2 + 5) / Real.sqrt (x^2 + 4) ≥ 5/2 ∧ 
  ∀ ε > 0, ∃ x : ℝ, (x^2 + 5) / Real.sqrt (x^2 + 4) < 5/2 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3763_376320


namespace NUMINAMATH_CALUDE_solution_to_system_of_equations_l3763_376344

theorem solution_to_system_of_equations :
  ∃ (x y : ℚ), 3 * x - 18 * y = 2 ∧ 4 * y - x = 6 ∧ x = -58/3 ∧ y = -10/3 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_system_of_equations_l3763_376344


namespace NUMINAMATH_CALUDE_apple_sharing_l3763_376382

theorem apple_sharing (total_apples : ℕ) (num_friends : ℕ) (apples_per_friend : ℕ) :
  total_apples = 9 →
  num_friends = 3 →
  total_apples = num_friends * apples_per_friend →
  apples_per_friend = 3 := by
  sorry

end NUMINAMATH_CALUDE_apple_sharing_l3763_376382


namespace NUMINAMATH_CALUDE_sports_club_members_l3763_376367

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  badminton : ℕ
  tennis : ℕ
  both : ℕ
  neither : ℕ

/-- Calculates the total number of members in the sports club -/
def total_members (club : SportsClub) : ℕ :=
  club.badminton + club.tennis - club.both + club.neither

/-- Theorem stating that the total number of members in the given sports club is 30 -/
theorem sports_club_members :
  ∃ (club : SportsClub),
    club.badminton = 17 ∧
    club.tennis = 19 ∧
    club.both = 9 ∧
    club.neither = 3 ∧
    total_members club = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_sports_club_members_l3763_376367


namespace NUMINAMATH_CALUDE_function_property_l3763_376339

/-- Given two functions f and g defined on ℝ satisfying certain properties, 
    prove that g(1) + g(-1) = 1 -/
theorem function_property (f g : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f (x - y) = f x * g y - g x * f y)
  (h2 : f 1 = f 2)
  (h3 : f 1 ≠ 0) : 
  g 1 + g (-1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3763_376339


namespace NUMINAMATH_CALUDE_committee_problem_l3763_376399

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem committee_problem :
  let total_students : ℕ := 10
  let committee_size : ℕ := 5
  let shared_members : ℕ := 3

  -- Number of different 5-student committees from 10 students
  (choose total_students committee_size = 252) ∧ 
  
  -- Number of ways to choose two 5-student committees with exactly 3 overlapping members
  ((choose total_students committee_size * 
    choose committee_size shared_members * 
    choose (total_students - committee_size) (committee_size - shared_members)) / 2 = 12600) :=
by sorry

end NUMINAMATH_CALUDE_committee_problem_l3763_376399


namespace NUMINAMATH_CALUDE_okeydokey_receives_25_earthworms_l3763_376308

/-- The number of apples Okeydokey paid -/
def okeydokey_apples : ℕ := 5

/-- The number of apples Artichokey paid -/
def artichokey_apples : ℕ := 7

/-- The total number of earthworms in the box -/
def total_earthworms : ℕ := 60

/-- Calculate the number of earthworms Okeydokey should receive -/
def okeydokey_earthworms : ℕ :=
  (okeydokey_apples * total_earthworms) / (okeydokey_apples + artichokey_apples)

/-- Theorem stating that Okeydokey should receive 25 earthworms -/
theorem okeydokey_receives_25_earthworms :
  okeydokey_earthworms = 25 := by
  sorry

end NUMINAMATH_CALUDE_okeydokey_receives_25_earthworms_l3763_376308


namespace NUMINAMATH_CALUDE_sqrt_expression_simplification_l3763_376309

theorem sqrt_expression_simplification :
  2 * Real.sqrt 3 - (3 * Real.sqrt 2 + Real.sqrt 3) = Real.sqrt 3 - 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_simplification_l3763_376309


namespace NUMINAMATH_CALUDE_age_difference_l3763_376303

theorem age_difference (albert_age mary_age betty_age : ℕ) : 
  albert_age = 2 * mary_age →
  albert_age = 4 * betty_age →
  betty_age = 7 →
  albert_age - mary_age = 14 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3763_376303


namespace NUMINAMATH_CALUDE_disobedient_pair_implies_ultra_disobedient_l3763_376322

/-- A function from natural numbers to positive real numbers -/
def IncreasingPositiveFunction : Type := 
  {f : ℕ → ℝ // (∀ m n, m < n → f m < f n) ∧ (∀ n, f n > 0)}

/-- Definition of a disobedient pair -/
def IsDisobedientPair (f : IncreasingPositiveFunction) (m n : ℕ) : Prop :=
  f.val (m * n) ≠ f.val m * f.val n

/-- Definition of an ultra-disobedient number -/
def IsUltraDisobedient (f : IncreasingPositiveFunction) (m : ℕ) : Prop :=
  ∀ N : ℕ, ∀ k : ℕ, ∃ n : ℕ, n > k ∧ 
    ∀ i : ℕ, i ≤ N → IsDisobedientPair f m (n + i)

/-- Main theorem: existence of a disobedient pair implies existence of an ultra-disobedient number -/
theorem disobedient_pair_implies_ultra_disobedient
  (f : IncreasingPositiveFunction)
  (h : ∃ m n : ℕ, IsDisobedientPair f m n) :
  ∃ m : ℕ, IsUltraDisobedient f m :=
sorry

end NUMINAMATH_CALUDE_disobedient_pair_implies_ultra_disobedient_l3763_376322


namespace NUMINAMATH_CALUDE_converse_square_sum_zero_contrapositive_subset_intersection_l3763_376375

-- Define the propositions
def P (x y : ℝ) : Prop := x^2 + y^2 = 0
def Q (x y : ℝ) : Prop := x = 0 ∧ y = 0

def R (A B : Set α) : Prop := A ∩ B = A
def S (A B : Set α) : Prop := A ⊆ B

-- Theorem for the converse of statement ①
theorem converse_square_sum_zero :
  ∀ x y : ℝ, Q x y → P x y :=
sorry

-- Theorem for the contrapositive of statement ③
theorem contrapositive_subset_intersection :
  ∀ A B : Set α, ¬(S A B) → ¬(R A B) :=
sorry

end NUMINAMATH_CALUDE_converse_square_sum_zero_contrapositive_subset_intersection_l3763_376375


namespace NUMINAMATH_CALUDE_base_b_is_eight_l3763_376300

/-- Given that in base b, the square of 13_b is 211_b, prove that b = 8 -/
theorem base_b_is_eight (b : ℕ) (h : b > 1) :
  (1 * b + 3)^2 = 2 * b^2 + 1 * b + 1 → b = 8 := by
  sorry

end NUMINAMATH_CALUDE_base_b_is_eight_l3763_376300


namespace NUMINAMATH_CALUDE_least_n_with_gcd_conditions_l3763_376329

theorem least_n_with_gcd_conditions : 
  ∃ (n : ℕ), n > 1000 ∧ 
  Nat.gcd 63 (n + 120) = 21 ∧ 
  Nat.gcd (n + 63) 120 = 60 ∧
  (∀ m : ℕ, m > 1000 ∧ m < n → 
    Nat.gcd 63 (m + 120) ≠ 21 ∨ 
    Nat.gcd (m + 63) 120 ≠ 60) ∧
  n = 1917 :=
by sorry

end NUMINAMATH_CALUDE_least_n_with_gcd_conditions_l3763_376329


namespace NUMINAMATH_CALUDE_largest_odd_sum_288_largest_odd_sum_288_is_43_l3763_376369

/-- Sum of first n consecutive odd integers -/
def sum_n_odd (n : ℕ) : ℕ := n^2

/-- Sum of odd integers from a to b inclusive -/
def sum_odd_range (a b : ℕ) : ℕ := 
  (sum_n_odd ((b - a) / 2 + 1)) - (sum_n_odd ((a - 1) / 2))

/-- The largest odd integer x such that the sum of all odd integers 
    from 13 to x inclusive is 288 -/
theorem largest_odd_sum_288 : 
  ∃ x : ℕ, x % 2 = 1 ∧ sum_odd_range 13 x = 288 ∧ 
  ∀ y : ℕ, y > x → y % 2 = 1 → sum_odd_range 13 y > 288 :=
sorry
 
/-- The largest odd integer x such that the sum of all odd integers 
    from 13 to x inclusive is 288 is equal to 43 -/
theorem largest_odd_sum_288_is_43 : 
  ∃! x : ℕ, x % 2 = 1 ∧ sum_odd_range 13 x = 288 ∧ 
  ∀ y : ℕ, y > x → y % 2 = 1 → sum_odd_range 13 y > 288 ∧ x = 43 :=
sorry

end NUMINAMATH_CALUDE_largest_odd_sum_288_largest_odd_sum_288_is_43_l3763_376369


namespace NUMINAMATH_CALUDE_fraction_subtraction_l3763_376334

theorem fraction_subtraction : (8 : ℚ) / 24 - (5 : ℚ) / 40 = (5 : ℚ) / 24 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l3763_376334


namespace NUMINAMATH_CALUDE_sum_equality_l3763_376318

theorem sum_equality (a b c d : ℝ) 
  (hab : a + b = 4)
  (hbc : b + c = 5)
  (had : a + d = 2) :
  c + d = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_equality_l3763_376318


namespace NUMINAMATH_CALUDE_dot_product_range_l3763_376348

/-- The range of the dot product OP · BA -/
theorem dot_product_range (O A B P : ℝ × ℝ) : 
  O = (0, 0) →
  A = (2, 0) →
  B = (1, -2 * Real.sqrt 3) →
  (∃ (x : ℝ), P.1 = x ∧ P.2 = Real.sqrt (1 - x^2 / 4)) →
  -2 ≤ (P.1 - O.1) * (B.1 - A.1) + (P.2 - O.2) * (B.2 - A.2) ∧
  (P.1 - O.1) * (B.1 - A.1) + (P.2 - O.2) * (B.2 - A.2) ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_dot_product_range_l3763_376348


namespace NUMINAMATH_CALUDE_quadratic_root_property_l3763_376336

theorem quadratic_root_property (m : ℝ) : 
  m^2 + 2*m - 1 = 0 → 2*m^2 + 4*m = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l3763_376336


namespace NUMINAMATH_CALUDE_expand_expression_l3763_376335

theorem expand_expression (x : ℝ) : (7 * x - 3) * (3 * x^2) = 21 * x^3 - 9 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3763_376335


namespace NUMINAMATH_CALUDE_line_perpendicular_plane_parallel_l3763_376379

structure Space where
  Line : Type
  Plane : Type
  perpendicular : Line → Plane → Prop
  parallel : Line → Line → Prop

variable (S : Space)

theorem line_perpendicular_plane_parallel
  (l m : S.Line) (α : S.Plane)
  (h1 : l ≠ m)
  (h2 : S.perpendicular l α)
  (h3 : S.parallel l m) :
  S.perpendicular m α :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_plane_parallel_l3763_376379


namespace NUMINAMATH_CALUDE_lawrence_marbles_l3763_376342

theorem lawrence_marbles (total_marbles : ℕ) (marbles_per_friend : ℕ) 
  (h1 : total_marbles = 5504) 
  (h2 : marbles_per_friend = 86) : 
  total_marbles / marbles_per_friend = 64 := by
  sorry

end NUMINAMATH_CALUDE_lawrence_marbles_l3763_376342


namespace NUMINAMATH_CALUDE_reflect_P_across_x_axis_l3763_376316

/-- Reflects a point across the x-axis in a 2D Cartesian coordinate system. -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The original point P in the Cartesian coordinate system. -/
def P : ℝ × ℝ := (-2, -3)

/-- Theorem: Reflecting the point P(-2,-3) across the x-axis results in the coordinates (-2, 3). -/
theorem reflect_P_across_x_axis :
  reflect_x P = (-2, 3) := by sorry

end NUMINAMATH_CALUDE_reflect_P_across_x_axis_l3763_376316


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3763_376383

/-- An arithmetic sequence with given conditions -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  a_4 : a 4 = 70
  a_21 : a 21 = -100

/-- Main theorem about the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (seq.a 1 = 100) ∧ 
  (∀ n, seq.a (n + 1) - seq.a n = -10) ∧
  (∀ n, seq.a n = -10 * n + 110) ∧
  (Finset.filter (fun n => -18 ≤ seq.a n ∧ seq.a n ≤ 18) (Finset.range 100)).card = 3 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3763_376383


namespace NUMINAMATH_CALUDE_complex_1_2i_in_first_quadrant_l3763_376380

/-- A complex number is in the first quadrant if its real part is positive and its imaginary part is positive -/
def in_first_quadrant (z : ℂ) : Prop := 0 < z.re ∧ 0 < z.im

/-- The theorem states that the complex number 1+2i is in the first quadrant -/
theorem complex_1_2i_in_first_quadrant : in_first_quadrant (1 + 2*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_1_2i_in_first_quadrant_l3763_376380


namespace NUMINAMATH_CALUDE_no_valid_class_composition_l3763_376381

theorem no_valid_class_composition : ¬ ∃ (n b g : ℕ+), 
  32 < n ∧ n < 40 ∧ 
  n = b + g ∧
  3 * b = 5 * g :=
by sorry

end NUMINAMATH_CALUDE_no_valid_class_composition_l3763_376381


namespace NUMINAMATH_CALUDE_triangle_side_angle_relation_l3763_376356

/-- Given a triangle ABC where a, b, c are sides opposite to angles A, B, C respectively,
    if a² = b² + ¼c², then (a cos B) / c = 5/8 -/
theorem triangle_side_angle_relation (a b c : ℝ) (h : a^2 = b^2 + (1/4)*c^2) :
  (a * Real.cos (Real.arccos ((b^2 + c^2 - a^2) / (2*b*c)))) / c = 5/8 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_angle_relation_l3763_376356


namespace NUMINAMATH_CALUDE_cheryl_material_usage_l3763_376354

theorem cheryl_material_usage (bought_type1 bought_type2 leftover : ℚ) :
  bought_type1 = 5/9 →
  bought_type2 = 1/3 →
  leftover = 8/24 →
  bought_type1 + bought_type2 - leftover = 5/9 := by
sorry

end NUMINAMATH_CALUDE_cheryl_material_usage_l3763_376354


namespace NUMINAMATH_CALUDE_exists_binary_sequence_with_geometric_partial_sums_l3763_376307

/-- A sequence where each term is either 0 or 1 -/
def BinarySequence := ℕ → Fin 2

/-- The partial sum of the first n terms of a BinarySequence -/
def PartialSum (a : BinarySequence) (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun i => a i)

/-- A sequence of partial sums forms a geometric sequence -/
def IsGeometricSequence (S : ℕ → ℕ) : Prop :=
  ∃ (r : ℚ), ∀ n : ℕ, S (n + 1) = (r : ℚ) * S n

/-- There exists a BinarySequence whose partial sums form a geometric sequence -/
theorem exists_binary_sequence_with_geometric_partial_sums :
  ∃ (a : BinarySequence), IsGeometricSequence (PartialSum a) := by
  sorry

end NUMINAMATH_CALUDE_exists_binary_sequence_with_geometric_partial_sums_l3763_376307


namespace NUMINAMATH_CALUDE_custom_op_theorem_l3763_376327

/-- Definition of the custom operation ⊕ -/
def custom_op (m n : ℝ) : ℝ := m * n * (m - n)

/-- Theorem stating that (a + b) ⊕ a = a^2 * b + a * b^2 -/
theorem custom_op_theorem (a b : ℝ) : custom_op (a + b) a = a^2 * b + a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_theorem_l3763_376327


namespace NUMINAMATH_CALUDE_night_temperature_l3763_376361

/-- Given the temperature changes throughout a day, prove the night temperature. -/
theorem night_temperature (morning_temp : ℝ) (noon_rise : ℝ) (night_drop : ℝ) :
  morning_temp = 22 →
  noon_rise = 6 →
  night_drop = 10 →
  morning_temp + noon_rise - night_drop = 18 := by
  sorry

end NUMINAMATH_CALUDE_night_temperature_l3763_376361


namespace NUMINAMATH_CALUDE_max_chain_length_theorem_l3763_376357

/-- Represents an equilateral triangle divided into smaller triangles -/
structure DividedTriangle where
  n : ℕ  -- number of parts each side is divided into
  total_triangles : ℕ  -- total number of smaller triangles

/-- Represents a chain of triangles within a divided triangle -/
structure TriangleChain (dt : DividedTriangle) where
  length : ℕ  -- number of triangles in the chain

/-- The maximum possible length of a triangle chain in a divided triangle -/
def max_chain_length (dt : DividedTriangle) : ℕ :=
  dt.n^2 - dt.n + 1

/-- Theorem stating the maximum chain length in a divided triangle -/
theorem max_chain_length_theorem (dt : DividedTriangle) :
  dt.total_triangles = dt.n^2 →
  ∃ (c : TriangleChain dt), ∀ (c' : TriangleChain dt), c.length ≥ c'.length ∧ c.length = max_chain_length dt :=
sorry

end NUMINAMATH_CALUDE_max_chain_length_theorem_l3763_376357


namespace NUMINAMATH_CALUDE_purple_shells_count_l3763_376346

/-- Represents the number of shells of each color --/
structure ShellCounts where
  total : Nat
  pink : Nat
  yellow : Nat
  blue : Nat
  orange : Nat

/-- Theorem stating that the number of purple shells is 13 --/
theorem purple_shells_count (s : ShellCounts) 
  (h1 : s.total = 65)
  (h2 : s.pink = 8)
  (h3 : s.yellow = 18)
  (h4 : s.blue = 12)
  (h5 : s.orange = 14) :
  s.total - (s.pink + s.yellow + s.blue + s.orange) = 13 := by
  sorry

#check purple_shells_count

end NUMINAMATH_CALUDE_purple_shells_count_l3763_376346


namespace NUMINAMATH_CALUDE_prime_fraction_characterization_l3763_376323

theorem prime_fraction_characterization (k x y : ℕ+) :
  (∃ p : ℕ, Nat.Prime p ∧ (x : ℝ)^(k : ℕ) * y / ((x : ℝ)^2 + (y : ℝ)^2) = p) ↔ k = 2 ∨ k = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_fraction_characterization_l3763_376323


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_l3763_376377

theorem quadratic_solution_difference : 
  let f : ℝ → ℝ := λ x => x^2 - 5*x + 7 - (x + 35)
  let s₁ := (6 + Real.sqrt 148) / 2
  let s₂ := (6 - Real.sqrt 148) / 2
  f s₁ = 0 ∧ f s₂ = 0 ∧ s₁ - s₂ = 2 * Real.sqrt 37 := by sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_l3763_376377


namespace NUMINAMATH_CALUDE_sample_first_year_300_l3763_376340

/-- Represents the ratio of students in each grade -/
structure GradeRatio :=
  (first second third fourth : ℕ)

/-- Calculates the number of first-year students to be sampled given the total sample size and grade ratio -/
def sampleFirstYear (totalSample : ℕ) (ratio : GradeRatio) : ℕ :=
  let totalRatio := ratio.first + ratio.second + ratio.third + ratio.fourth
  (totalSample * ratio.first) / totalRatio

/-- Theorem stating that for a sample size of 300 and ratio 4:5:5:6, the number of first-year students sampled is 60 -/
theorem sample_first_year_300 :
  sampleFirstYear 300 ⟨4, 5, 5, 6⟩ = 60 := by
  sorry

#eval sampleFirstYear 300 ⟨4, 5, 5, 6⟩

end NUMINAMATH_CALUDE_sample_first_year_300_l3763_376340


namespace NUMINAMATH_CALUDE_f_max_value_when_a_eq_one_unique_root_f_eq_g_l3763_376396

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x
def g (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - (2*a + 1) * x + (a + 1) * Real.log x

-- Theorem for the maximum value of f when a = 1
theorem f_max_value_when_a_eq_one :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f 1 y ≤ f 1 x ∧ f 1 x = -1 := by sorry

-- Theorem for the unique root of f(x) = g(x) when a ≥ 1
theorem unique_root_f_eq_g (a : ℝ) (h : a ≥ 1) :
  ∃! (x : ℝ), x > 0 ∧ f a x = g a x := by sorry

end

end NUMINAMATH_CALUDE_f_max_value_when_a_eq_one_unique_root_f_eq_g_l3763_376396


namespace NUMINAMATH_CALUDE_first_divisor_problem_l3763_376350

theorem first_divisor_problem (m d : ℕ) : 
  (∃ q : ℕ, m = d * q + 47) →
  (∃ p : ℕ, m = 24 * p + 23) →
  (∀ x < d, ¬(∃ q : ℕ, m = x * q + 47)) →
  d = 72 := by
sorry

end NUMINAMATH_CALUDE_first_divisor_problem_l3763_376350


namespace NUMINAMATH_CALUDE_xiao_jun_travel_box_probability_l3763_376338

-- Define the number of digits in the password
def password_length : ℕ := 6

-- Define the number of possible digits (0-9)
def possible_digits : ℕ := 10

-- Define the probability of guessing the correct last digit
def probability_of_success : ℚ := 1 / possible_digits

-- Theorem statement
theorem xiao_jun_travel_box_probability :
  probability_of_success = 1 / 10 :=
sorry

end NUMINAMATH_CALUDE_xiao_jun_travel_box_probability_l3763_376338


namespace NUMINAMATH_CALUDE_ellipse_k_range_l3763_376390

/-- An ellipse with foci on the y-axis is represented by the equation (x^2)/(15-k) + (y^2)/(k-9) = 1,
    where k is a real number. This theorem states that the range of k is (12, 15). -/
theorem ellipse_k_range (k : ℝ) :
  (∀ x y : ℝ, x^2 / (15 - k) + y^2 / (k - 9) = 1) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) →
  k ∈ Set.Ioo 12 15 :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l3763_376390


namespace NUMINAMATH_CALUDE_cylinder_cut_area_l3763_376365

/-- The area of the newly exposed circular segment face when cutting a cylinder -/
theorem cylinder_cut_area (r h : ℝ) (h_r : r = 8) (h_h : h = 10) :
  let base_area := π * r^2
  let sector_area := (1/4) * base_area
  sector_area = 16 * π := by sorry

end NUMINAMATH_CALUDE_cylinder_cut_area_l3763_376365


namespace NUMINAMATH_CALUDE_set_intersection_complement_l3763_376376

def U : Set Int := Set.univ
def M : Set Int := {1, 2}
def P : Set Int := {-2, -1, 0, 1, 2}

theorem set_intersection_complement : P ∩ (U \ M) = {-2, -1, 0} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_complement_l3763_376376


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l3763_376302

theorem quadratic_one_solution (n : ℝ) : 
  (∃! x : ℝ, 4 * x^2 + n * x + 4 = 0) ↔ n = 8 := by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l3763_376302


namespace NUMINAMATH_CALUDE_genuine_items_count_l3763_376347

theorem genuine_items_count (total_purses total_handbags : ℕ) 
  (h1 : total_purses = 26)
  (h2 : total_handbags = 24)
  (h3 : total_purses / 2 + total_handbags / 4 = (total_purses + total_handbags) - 31) :
  31 = total_purses + total_handbags - (total_purses / 2 + total_handbags / 4) :=
by sorry

end NUMINAMATH_CALUDE_genuine_items_count_l3763_376347


namespace NUMINAMATH_CALUDE_sandcastle_height_difference_l3763_376372

/-- The height difference between Miki's sandcastle and her sister's sandcastle -/
theorem sandcastle_height_difference 
  (miki_height : ℝ) 
  (sister_height : ℝ) 
  (h1 : miki_height = 0.8333333333333334) 
  (h2 : sister_height = 0.5) : 
  miki_height - sister_height = 0.3333333333333334 := by
  sorry

end NUMINAMATH_CALUDE_sandcastle_height_difference_l3763_376372


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3763_376368

theorem simplify_and_rationalize :
  (Real.sqrt 6 / Real.sqrt 7) * (Real.sqrt 6 / Real.sqrt 14) * (Real.sqrt 9 / Real.sqrt 21) = Real.sqrt 2058 / 114 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3763_376368


namespace NUMINAMATH_CALUDE_zoo_feeding_theorem_l3763_376352

/-- Represents the number of animal pairs in the zoo -/
def num_pairs : ℕ := 6

/-- Represents the number of ways to feed the animals -/
def feeding_ways : ℕ := 14400

/-- Theorem stating the number of ways to feed the animals in the specified pattern -/
theorem zoo_feeding_theorem :
  (num_pairs = 6) →
  (∃ (male_choices female_choices : ℕ → ℕ),
    (∀ i, i ∈ Finset.range (num_pairs - 1) → male_choices i = num_pairs - 1 - i) ∧
    (∀ i, i ∈ Finset.range num_pairs → female_choices i = num_pairs - 1 - i) ∧
    (feeding_ways = (Finset.prod (Finset.range (num_pairs - 1)) male_choices) *
                    (Finset.prod (Finset.range num_pairs) female_choices))) :=
by sorry

#check zoo_feeding_theorem

end NUMINAMATH_CALUDE_zoo_feeding_theorem_l3763_376352


namespace NUMINAMATH_CALUDE_min_value_xyz_l3763_376386

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 8) :
  x + 2 * y + 4 * z ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_xyz_l3763_376386


namespace NUMINAMATH_CALUDE_ellipse_equation_l3763_376330

theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (2 * a * b = 4) →
  (a^2 - b^2 = 3) →
  (a = 2 ∧ b = 1) := by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3763_376330


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3763_376359

/-- Simple interest calculation -/
def simple_interest (principal time rate : ℚ) : ℚ :=
  principal * time * rate / 100

theorem interest_rate_calculation (principal interest time : ℚ) 
  (h_principal : principal = 800)
  (h_interest : interest = 200)
  (h_time : time = 4) :
  ∃ (rate : ℚ), simple_interest principal time rate = interest ∧ rate = 25/4 :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3763_376359


namespace NUMINAMATH_CALUDE_max_prob_at_one_l3763_376306

def binomial_prob (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem max_prob_at_one :
  let n : ℕ := 5
  let p : ℝ := 1/4
  ∀ k : ℕ, k ≠ 1 → k ≤ n → binomial_prob n 1 p > binomial_prob n k p :=
by sorry

end NUMINAMATH_CALUDE_max_prob_at_one_l3763_376306


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3763_376385

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def is_in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The theorem to prove -/
theorem point_in_fourth_quadrant :
  let P : Point := ⟨3, -3⟩
  is_in_fourth_quadrant P := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3763_376385


namespace NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l3763_376345

theorem smallest_addition_for_divisibility : ∃! x : ℕ, 
  (x ≤ 751 * 503 - 1) ∧ 
  ((956734 + x) % (751 * 503) = 0) ∧
  ∀ y : ℕ, y < x → ((956734 + y) % (751 * 503) ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l3763_376345


namespace NUMINAMATH_CALUDE_problem_2011_l3763_376337

theorem problem_2011 : (2011^2 + 2011) / 2011 = 2012 := by
  sorry

end NUMINAMATH_CALUDE_problem_2011_l3763_376337


namespace NUMINAMATH_CALUDE_piano_lesson_discount_percentage_l3763_376360

/-- Calculates the discount percentage on piano lessons given the piano cost, number of lessons,
    cost per lesson, and total cost after discount. -/
theorem piano_lesson_discount_percentage
  (piano_cost : ℝ)
  (num_lessons : ℕ)
  (cost_per_lesson : ℝ)
  (total_cost_after_discount : ℝ)
  (h1 : piano_cost = 500)
  (h2 : num_lessons = 20)
  (h3 : cost_per_lesson = 40)
  (h4 : total_cost_after_discount = 1100) :
  (1 - (total_cost_after_discount - piano_cost) / (num_lessons * cost_per_lesson)) * 100 = 25 :=
by sorry


end NUMINAMATH_CALUDE_piano_lesson_discount_percentage_l3763_376360


namespace NUMINAMATH_CALUDE_matrix_power_equality_l3763_376378

def A : Matrix (Fin 2) (Fin 2) ℕ := !![1, 1; 2, 1]

def B : Matrix (Fin 2) (Fin 2) ℕ := !![17, 12; 24, 17]

theorem matrix_power_equality :
  A^10 = B^5 := by sorry

end NUMINAMATH_CALUDE_matrix_power_equality_l3763_376378


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l3763_376362

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (((x^2 + y^2) * (4*x^2 + y^2)).sqrt) / (x*y) ≥ 3 :=
by sorry

theorem equality_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (((x^2 + y^2) * (4*x^2 + y^2)).sqrt) / (x*y) = 3 ↔ y = x * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l3763_376362


namespace NUMINAMATH_CALUDE_quadratic_inequality_holds_for_all_x_l3763_376370

theorem quadratic_inequality_holds_for_all_x (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k - 4)*x - k + 8 > 0) ↔ -2 < k ∧ k < 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_holds_for_all_x_l3763_376370


namespace NUMINAMATH_CALUDE_integer_squared_less_than_triple_l3763_376313

theorem integer_squared_less_than_triple :
  ∀ x : ℤ, x^2 < 3*x ↔ x = 1 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_integer_squared_less_than_triple_l3763_376313


namespace NUMINAMATH_CALUDE_sum_cube_inequality_l3763_376394

theorem sum_cube_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000/9 := by
  sorry

end NUMINAMATH_CALUDE_sum_cube_inequality_l3763_376394


namespace NUMINAMATH_CALUDE_chen_trigonometric_problem_l3763_376349

theorem chen_trigonometric_problem :
  ∃ (N : ℕ) (α β γ θ : ℝ),
    0.1 = Real.sin γ * Real.cos θ * Real.sin α ∧
    0.2 = Real.sin γ * Real.sin θ * Real.cos α ∧
    0.3 = Real.cos γ * Real.cos θ * Real.sin β ∧
    0.4 = Real.cos γ * Real.sin θ * Real.cos β ∧
    0.5 ≥ |N - 100 * Real.cos (2 * θ)| ∧
    N = 79 := by
  sorry

end NUMINAMATH_CALUDE_chen_trigonometric_problem_l3763_376349


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l3763_376321

-- Define the cost of one t-shirt
def cost_per_shirt : ℚ := 9.95

-- Define the number of t-shirts bought
def num_shirts : ℕ := 25

-- Define the total cost
def total_cost : ℚ := cost_per_shirt * num_shirts

-- Theorem to prove
theorem total_cost_is_correct : total_cost = 248.75 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l3763_376321


namespace NUMINAMATH_CALUDE_units_digit_of_7_power_2050_l3763_376314

theorem units_digit_of_7_power_2050 : (7^2050 : ℕ) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_power_2050_l3763_376314


namespace NUMINAMATH_CALUDE_circle_ratio_theorem_l3763_376331

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


end NUMINAMATH_CALUDE_circle_ratio_theorem_l3763_376331


namespace NUMINAMATH_CALUDE_jodi_walked_3_miles_week3_l3763_376389

/-- Represents the walking schedule of Jodi over 4 weeks -/
structure WalkingSchedule where
  weeks : Nat
  days_per_week : Nat
  miles_week1 : Nat
  miles_week2 : Nat
  miles_week4 : Nat
  total_miles : Nat

/-- Calculates the miles walked per day in the third week -/
def miles_per_day_week3 (schedule : WalkingSchedule) : Nat :=
  let miles_weeks_124 := schedule.miles_week1 * schedule.days_per_week +
                         schedule.miles_week2 * schedule.days_per_week +
                         schedule.miles_week4 * schedule.days_per_week
  let miles_week3 := schedule.total_miles - miles_weeks_124
  miles_week3 / schedule.days_per_week

/-- Theorem stating that Jodi walked 3 miles per day in the third week -/
theorem jodi_walked_3_miles_week3 (schedule : WalkingSchedule) 
  (h1 : schedule.weeks = 4)
  (h2 : schedule.days_per_week = 6)
  (h3 : schedule.miles_week1 = 1)
  (h4 : schedule.miles_week2 = 2)
  (h5 : schedule.miles_week4 = 4)
  (h6 : schedule.total_miles = 60) :
  miles_per_day_week3 schedule = 3 := by
  sorry

end NUMINAMATH_CALUDE_jodi_walked_3_miles_week3_l3763_376389


namespace NUMINAMATH_CALUDE_average_children_in_families_with_children_l3763_376388

/-- Given 15 families with an average of 3 children per family, 
    and exactly 3 of these families being childless, 
    prove that the average number of children in the families 
    that have children is 45/12. -/
theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (average_children : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 15)
  (h2 : average_children = 3)
  (h3 : childless_families = 3) :
  (total_families : ℚ) * average_children / 
  ((total_families : ℚ) - childless_families) = 45 / 12 := by
sorry

end NUMINAMATH_CALUDE_average_children_in_families_with_children_l3763_376388


namespace NUMINAMATH_CALUDE_square_of_real_not_always_positive_l3763_376341

theorem square_of_real_not_always_positive : 
  ¬(∀ (a : ℝ), a^2 > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_square_of_real_not_always_positive_l3763_376341


namespace NUMINAMATH_CALUDE_ice_cream_to_after_lunch_ratio_l3763_376392

def initial_money : ℚ := 30
def lunch_cost : ℚ := 10
def remaining_money : ℚ := 15

def money_after_lunch : ℚ := initial_money - lunch_cost
def ice_cream_cost : ℚ := money_after_lunch - remaining_money

theorem ice_cream_to_after_lunch_ratio :
  ice_cream_cost / money_after_lunch = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_ice_cream_to_after_lunch_ratio_l3763_376392


namespace NUMINAMATH_CALUDE_cheese_bread_solution_l3763_376363

/-- Represents the problem of buying cheese bread for a group of people. -/
structure CheeseBreadProblem where
  cost_per_100g : ℚ  -- Cost in R$ per 100g of cheese bread
  pieces_per_100g : ℕ  -- Number of pieces in 100g of cheese bread
  pieces_per_person : ℕ  -- Average number of pieces eaten per person
  total_people : ℕ  -- Total number of people
  scale_precision : ℕ  -- Precision of the bakery's scale in grams

/-- Calculates the amount to buy, cost, and leftover pieces for a given CheeseBreadProblem. -/
def solve_cheese_bread_problem (p : CheeseBreadProblem) :
  (ℕ × ℚ × ℕ) :=
  sorry

/-- Theorem stating the correct solution for the given problem. -/
theorem cheese_bread_solution :
  let problem := CheeseBreadProblem.mk 3.2 10 5 23 100
  let (amount, cost, leftover) := solve_cheese_bread_problem problem
  amount = 1200 ∧ cost = 38.4 ∧ leftover = 5 :=
sorry

end NUMINAMATH_CALUDE_cheese_bread_solution_l3763_376363


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l3763_376391

theorem factorial_sum_equality : 7 * Nat.factorial 6 + 6 * Nat.factorial 5 + 2 * Nat.factorial 5 = 6000 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l3763_376391


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_g_l3763_376366

-- Define the function g(x)
noncomputable def g : ℝ → ℤ
| x => if x > -1 then Int.ceil (1 / (x + 1))
       else if x < -1 then Int.floor (1 / (x + 1))
       else 0  -- arbitrary value for x = -1, as g is not defined there

-- Theorem statement
theorem zero_not_in_range_of_g : ∀ x : ℝ, g x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_g_l3763_376366


namespace NUMINAMATH_CALUDE_divisibility_of_squares_sum_l3763_376315

theorem divisibility_of_squares_sum (p x y z : ℕ) : 
  Prime p → 
  0 < x → x < y → y < z → z < p → 
  x^3 % p = y^3 % p → y^3 % p = z^3 % p →
  (x^2 + y^2 + z^2) % (x + y + z) = 0 :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_squares_sum_l3763_376315


namespace NUMINAMATH_CALUDE_vertical_dominoes_even_l3763_376326

/-- A grid with even rows colored white and odd rows colored black -/
structure ColoredGrid where
  rows : ℕ
  cols : ℕ

/-- A domino placement on a colored grid -/
structure DominoPlacement (grid : ColoredGrid) where
  horizontal : Finset (ℕ × ℕ)  -- Set of starting positions for horizontal dominoes
  vertical : Finset (ℕ × ℕ)    -- Set of starting positions for vertical dominoes

/-- Predicate to check if a domino placement is valid -/
def is_valid_placement (grid : ColoredGrid) (placement : DominoPlacement grid) : Prop :=
  ∀ (i j : ℕ), i < grid.rows ∧ j < grid.cols →
    ((i, j) ∈ placement.horizontal → j + 1 < grid.cols) ∧
    ((i, j) ∈ placement.vertical → i + 1 < grid.rows)

/-- The main theorem: The number of vertically placed dominoes is even -/
theorem vertical_dominoes_even (grid : ColoredGrid) (placement : DominoPlacement grid)
  (h_valid : is_valid_placement grid placement) :
  Even placement.vertical.card :=
sorry

end NUMINAMATH_CALUDE_vertical_dominoes_even_l3763_376326


namespace NUMINAMATH_CALUDE_initial_marbles_proof_l3763_376364

/-- The number of marbles Connie gave to Juan -/
def marbles_to_juan : ℕ := 73

/-- The number of marbles Connie gave to Maria -/
def marbles_to_maria : ℕ := 45

/-- The number of marbles Connie has left -/
def marbles_left : ℕ := 70

/-- The initial number of marbles Connie had -/
def initial_marbles : ℕ := marbles_to_juan + marbles_to_maria + marbles_left

theorem initial_marbles_proof : initial_marbles = 188 := by
  sorry

end NUMINAMATH_CALUDE_initial_marbles_proof_l3763_376364


namespace NUMINAMATH_CALUDE_associate_prof_charts_l3763_376325

theorem associate_prof_charts (total_people : ℕ) (total_pencils : ℕ) (total_charts : ℕ)
  (h1 : total_people = 8)
  (h2 : total_pencils = 10)
  (h3 : total_charts = 14) :
  ∃ (assoc_prof : ℕ) (asst_prof : ℕ) (charts_per_assoc : ℕ),
    assoc_prof + asst_prof = total_people ∧
    2 * assoc_prof + asst_prof = total_pencils ∧
    charts_per_assoc * assoc_prof + 2 * asst_prof = total_charts ∧
    charts_per_assoc = 1 :=
by sorry

end NUMINAMATH_CALUDE_associate_prof_charts_l3763_376325


namespace NUMINAMATH_CALUDE_proj_scale_proj_add_l3763_376371

-- Define the 2D vector type
def Vector2D := ℝ × ℝ

-- Define the projection operation on x-axis
def proj_x (v : Vector2D) : ℝ := v.1

-- Define the projection operation on y-axis
def proj_y (v : Vector2D) : ℝ := v.2

-- Define vector addition
def add (u v : Vector2D) : Vector2D := (u.1 + v.1, u.2 + v.2)

-- Define scalar multiplication
def scale (k : ℝ) (v : Vector2D) : Vector2D := (k * v.1, k * v.2)

-- Theorem for property 1 (scalar multiplication)
theorem proj_scale (k : ℝ) (v : Vector2D) :
  proj_x (scale k v) = k * proj_x v ∧ proj_y (scale k v) = k * proj_y v := by
  sorry

-- Theorem for property 2 (vector addition)
theorem proj_add (u v : Vector2D) :
  proj_x (add u v) = proj_x u + proj_x v ∧ proj_y (add u v) = proj_y u + proj_y v := by
  sorry

end NUMINAMATH_CALUDE_proj_scale_proj_add_l3763_376371


namespace NUMINAMATH_CALUDE_min_cos_sum_acute_angles_l3763_376310

theorem min_cos_sum_acute_angles (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan α + Real.tan β = 4 * Real.sin (α + β)) :
  Real.cos (α + β) ≥ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_min_cos_sum_acute_angles_l3763_376310


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_characterization_l3763_376355

/-- Two lines in 3D space -/
structure Line3D where
  m : ℝ
  n : ℝ
  p : ℝ

/-- Condition for two lines to be parallel -/
def parallel (l₁ l₂ : Line3D) : Prop :=
  l₁.m / l₂.m = l₁.n / l₂.n ∧ l₁.n / l₂.n = l₁.p / l₂.p

/-- Condition for two lines to be perpendicular -/
def perpendicular (l₁ l₂ : Line3D) : Prop :=
  l₁.m * l₂.m + l₁.n * l₂.n + l₁.p * l₂.p = 0

/-- Theorem: Characterization of parallel and perpendicular lines in 3D space -/
theorem line_parallel_perpendicular_characterization (l₁ l₂ : Line3D) :
  (parallel l₁ l₂ ↔ l₁.m / l₂.m = l₁.n / l₂.n ∧ l₁.n / l₂.n = l₁.p / l₂.p) ∧
  (perpendicular l₁ l₂ ↔ l₁.m * l₂.m + l₁.n * l₂.n + l₁.p * l₂.p = 0) := by
  sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_characterization_l3763_376355


namespace NUMINAMATH_CALUDE_two_digit_cube_diff_reverse_l3763_376312

/-- A function that reverses a two-digit number -/
def reverse (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- A predicate that checks if a number is a positive perfect cube -/
def is_positive_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ k^3 = n

/-- The main theorem -/
theorem two_digit_cube_diff_reverse :
  ∀ M : ℕ,
    10 ≤ M ∧ M < 100 ∧  -- M is a two-digit number
    (M % 10 ≠ 0) ∧      -- M's unit digit is non-zero
    is_positive_perfect_cube (M - reverse M) →
    M = 81 ∨ M = 92 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_cube_diff_reverse_l3763_376312


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3763_376353

theorem quadratic_inequality_solution_set (x : ℝ) :
  x^2 - 2*x - 3 > 0 ↔ x < -1 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3763_376353


namespace NUMINAMATH_CALUDE_gcd_lcm_problem_l3763_376393

theorem gcd_lcm_problem : 
  (Nat.gcd 60 75 * Nat.lcm 48 18 + 5 = 2165) := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_problem_l3763_376393


namespace NUMINAMATH_CALUDE_willie_stickers_l3763_376319

/-- The number of stickers Willie started with -/
def initial_stickers : ℕ := 36

/-- The number of stickers Willie ended up with -/
def final_stickers : ℕ := 29

/-- The number of stickers Willie gave to Emily -/
def stickers_given : ℕ := initial_stickers - final_stickers

theorem willie_stickers :
  stickers_given = initial_stickers - final_stickers :=
by sorry

end NUMINAMATH_CALUDE_willie_stickers_l3763_376319


namespace NUMINAMATH_CALUDE_line_properties_l3763_376305

/-- A parameterized line in 2D space -/
structure ParameterizedLine where
  point : ℝ → ℝ × ℝ

/-- The theorem stating the properties of the given parameterized line -/
theorem line_properties (L : ParameterizedLine) : 
  L.point 1 = (2, 5) ∧ L.point 4 = (5, -7) → L.point 0 = (1, 9) := by
  sorry


end NUMINAMATH_CALUDE_line_properties_l3763_376305


namespace NUMINAMATH_CALUDE_vanessa_score_l3763_376351

/-- Vanessa's basketball score record problem -/
theorem vanessa_score (total_score : ℕ) (num_players : ℕ) (other_players_avg : ℚ) :
  total_score = 68 →
  num_players = 9 →
  other_players_avg = 4.5 →
  ∃ vanessa_score : ℕ,
    vanessa_score = 32 ∧
    vanessa_score = total_score - (num_players - 1) * (other_players_avg.num / other_players_avg.den) :=
by sorry

end NUMINAMATH_CALUDE_vanessa_score_l3763_376351
