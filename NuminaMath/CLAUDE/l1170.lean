import Mathlib

namespace NUMINAMATH_CALUDE_last_two_nonzero_digits_of_100_factorial_l1170_117031

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_nonzero_digits (n : ℕ) : ℕ :=
  n % 100

theorem last_two_nonzero_digits_of_100_factorial :
  last_two_nonzero_digits (factorial 100) = 76 := by
  sorry

end NUMINAMATH_CALUDE_last_two_nonzero_digits_of_100_factorial_l1170_117031


namespace NUMINAMATH_CALUDE_du_chin_meat_pie_business_l1170_117043

/-- Du Chin's meat pie business theorem -/
theorem du_chin_meat_pie_business 
  (pies_baked : ℕ) 
  (price_per_pie : ℚ) 
  (ingredient_cost_ratio : ℚ) 
  (h1 : pies_baked = 200)
  (h2 : price_per_pie = 20)
  (h3 : ingredient_cost_ratio = 3/5) :
  pies_baked * price_per_pie - pies_baked * price_per_pie * ingredient_cost_ratio = 1600 :=
by sorry

end NUMINAMATH_CALUDE_du_chin_meat_pie_business_l1170_117043


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1170_117097

-- Define the universal set U
def U : Set (ℝ × ℝ) := Set.univ

-- Define set M
def M : Set (ℝ × ℝ) := {p | (p.2 + 2) / (p.1 - 2) = 1}

-- Define set N
def N : Set (ℝ × ℝ) := {p | p.2 ≠ p.1 - 4}

-- Statement to prove
theorem complement_intersection_theorem :
  (U \ M) ∩ (U \ N) = {(2, -2)} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1170_117097


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_l1170_117058

def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 5

theorem monotonic_decreasing_interval :
  {x : ℝ | ∀ y : ℝ, -1 < x → x < y → y < 3 → f x > f y} = Set.Ioo (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_l1170_117058


namespace NUMINAMATH_CALUDE_floor_of_5_7_l1170_117055

theorem floor_of_5_7 : ⌊(5.7 : ℝ)⌋ = 5 := by
  sorry

end NUMINAMATH_CALUDE_floor_of_5_7_l1170_117055


namespace NUMINAMATH_CALUDE_shaded_area_in_square_with_semicircle_l1170_117072

theorem shaded_area_in_square_with_semicircle (d : ℝ) (h : d > 0) :
  let s := d / Real.sqrt 2
  let square_area := s^2
  let semicircle_area := π * (d/2)^2 / 2
  square_area - semicircle_area = s^2 - (π/8) * d^2 := by sorry

end NUMINAMATH_CALUDE_shaded_area_in_square_with_semicircle_l1170_117072


namespace NUMINAMATH_CALUDE_village_population_l1170_117085

theorem village_population (P : ℕ) : 
  (P : ℝ) * 0.9 * 0.85 = 2907 → P = 3801 := by
sorry

end NUMINAMATH_CALUDE_village_population_l1170_117085


namespace NUMINAMATH_CALUDE_three_digit_divisible_by_11_l1170_117075

theorem three_digit_divisible_by_11 (x y z : ℕ) (A : ℕ) : 
  (100 ≤ A) ∧ (A < 1000) ∧ 
  (A = 100 * x + 10 * y + z) ∧ 
  (x + z = y) → 
  ∃ k : ℕ, A = 11 * k := by
sorry

end NUMINAMATH_CALUDE_three_digit_divisible_by_11_l1170_117075


namespace NUMINAMATH_CALUDE_marks_radiator_cost_l1170_117056

/-- The total cost for Mark's car radiator replacement -/
def total_cost (labor_hours : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) : ℕ :=
  labor_hours * hourly_rate + part_cost

/-- Proof that Mark's total cost for car radiator replacement is $300 -/
theorem marks_radiator_cost :
  total_cost 2 75 150 = 300 := by
  sorry

end NUMINAMATH_CALUDE_marks_radiator_cost_l1170_117056


namespace NUMINAMATH_CALUDE_quadratic_completion_l1170_117099

theorem quadratic_completion (p : ℝ) (n : ℝ) : 
  (∀ x, x^2 + p*x + 1/4 = (x+n)^2 - 1/16) → 
  p < 0 → 
  p = -Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completion_l1170_117099


namespace NUMINAMATH_CALUDE_minibuses_needed_l1170_117053

theorem minibuses_needed (students : ℕ) (teacher : ℕ) (capacity : ℕ) : 
  students = 48 → teacher = 1 → capacity = 8 → 
  (students + teacher + capacity - 1) / capacity = 7 := by
  sorry

end NUMINAMATH_CALUDE_minibuses_needed_l1170_117053


namespace NUMINAMATH_CALUDE_shopping_expenditure_l1170_117074

theorem shopping_expenditure (initial_amount : ℝ) : 
  initial_amount * (1 - 0.2) * (1 - 0.15) * (1 - 0.25) = 217 →
  initial_amount = 425.49 := by
sorry

end NUMINAMATH_CALUDE_shopping_expenditure_l1170_117074


namespace NUMINAMATH_CALUDE_hour_hand_angle_for_9_to_1_ratio_l1170_117003

/-- Represents a toy clock with a specific ratio between hour and minute hand rotations -/
structure ToyClock where
  /-- The number of full circles the minute hand makes for each full circle of the hour hand -/
  minuteToHourRatio : ℕ
  /-- Assumption that the ratio is greater than 1 -/
  ratioGtOne : minuteToHourRatio > 1

/-- Calculates the angle turned by the hour hand when it next coincides with the minute hand -/
def hourHandAngleAtNextCoincidence (clock : ToyClock) : ℚ :=
  360 / (clock.minuteToHourRatio - 1)

/-- Theorem stating that for a toy clock where the minute hand makes 9 circles 
    for each full circle of the hour hand, the hour hand turns 45° at the next coincidence -/
theorem hour_hand_angle_for_9_to_1_ratio :
  let clock : ToyClock := ⟨9, by norm_num⟩
  hourHandAngleAtNextCoincidence clock = 45 := by
  sorry

end NUMINAMATH_CALUDE_hour_hand_angle_for_9_to_1_ratio_l1170_117003


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1170_117036

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  sum : ℕ → ℝ -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, sum n = n * (2 * a 1 + (n - 1) * d) / 2

/-- Theorem: For an arithmetic sequence, if S₄ = 6 and 2a₃ - a₂ = 6, then a₁ = -3 -/
theorem arithmetic_sequence_first_term
  (seq : ArithmeticSequence)
  (sum_4 : seq.sum 4 = 6)
  (term_relation : 2 * seq.a 3 - seq.a 2 = 6) :
  seq.a 1 = -3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1170_117036


namespace NUMINAMATH_CALUDE_smallest_integer_proof_l1170_117096

def club_size : ℕ := 30

def smallest_integer : ℕ := 2329089562800

theorem smallest_integer_proof :
  (∀ i ∈ Finset.range 28, smallest_integer % i = 0) ∧
  (smallest_integer % 31 = 0) ∧
  (∀ i ∈ Finset.range 3, smallest_integer % (28 + i) ≠ 0) ∧
  (∀ n : ℕ, n < smallest_integer →
    ¬((∀ i ∈ Finset.range 28, n % i = 0) ∧
      (n % 31 = 0) ∧
      (∀ i ∈ Finset.range 3, n % (28 + i) ≠ 0))) :=
by sorry

#check smallest_integer_proof

end NUMINAMATH_CALUDE_smallest_integer_proof_l1170_117096


namespace NUMINAMATH_CALUDE_intersection_of_ellipses_l1170_117001

theorem intersection_of_ellipses :
  ∃! (points : Finset (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ points ↔ (x^2 + 9*y^2 = 9 ∧ 9*x^2 + y^2 = 1)) ∧
    points.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_ellipses_l1170_117001


namespace NUMINAMATH_CALUDE_factorization_sum_l1170_117061

theorem factorization_sum (A B C D E F G H J K : ℤ) (x y : ℝ) :
  (125 * x^8 - 2401 * y^8 = (A * x + B * y) * (C * x^4 + D * x * y + E * y^4) * 
                            (F * x + G * y) * (H * x^4 + J * x * y + K * y^4)) →
  A + B + C + D + E + F + G + H + J + K = 102 := by
sorry

end NUMINAMATH_CALUDE_factorization_sum_l1170_117061


namespace NUMINAMATH_CALUDE_high_season_packs_correct_l1170_117018

/-- Represents the number of tuna packs sold per hour during the high season -/
def high_season_packs : ℕ := 6

/-- Represents the number of tuna packs sold per hour during the low season -/
def low_season_packs : ℕ := 4

/-- Represents the price of each tuna pack in dollars -/
def price_per_pack : ℕ := 60

/-- Represents the number of hours fish are sold per day -/
def hours_per_day : ℕ := 15

/-- Represents the additional revenue in dollars during the high season compared to the low season -/
def additional_revenue : ℕ := 1800

theorem high_season_packs_correct :
  high_season_packs * hours_per_day * price_per_pack =
  low_season_packs * hours_per_day * price_per_pack + additional_revenue :=
by sorry

end NUMINAMATH_CALUDE_high_season_packs_correct_l1170_117018


namespace NUMINAMATH_CALUDE_sum_lower_bound_l1170_117039

theorem sum_lower_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  a + b ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_lower_bound_l1170_117039


namespace NUMINAMATH_CALUDE_pattern_equality_l1170_117076

theorem pattern_equality (n : ℕ) : n * (n + 2) + 1 = (n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_pattern_equality_l1170_117076


namespace NUMINAMATH_CALUDE_expand_product_l1170_117040

theorem expand_product (x : ℝ) : (x + 3) * (x^2 + x + 1) = x^3 + 4*x^2 + 4*x + 3 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1170_117040


namespace NUMINAMATH_CALUDE_employee_new_salary_is_35000_l1170_117020

/-- Calculates the new salary of employees after a salary redistribution --/
def new_employee_salary (emily_original_salary emily_new_salary num_employees employee_original_salary : ℕ) : ℕ :=
  let salary_reduction := emily_original_salary - emily_new_salary
  let additional_per_employee := salary_reduction / num_employees
  employee_original_salary + additional_per_employee

/-- Proves that the new employee salary is $35,000 given the problem conditions --/
theorem employee_new_salary_is_35000 :
  new_employee_salary 1000000 850000 10 20000 = 35000 := by
  sorry

end NUMINAMATH_CALUDE_employee_new_salary_is_35000_l1170_117020


namespace NUMINAMATH_CALUDE_inequality_proof_l1170_117094

theorem inequality_proof (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  a^2 + a*b + b^2 ≤ 3*(a - Real.sqrt (a*b) + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1170_117094


namespace NUMINAMATH_CALUDE_fourTangentCircles_l1170_117000

-- Define the circles C₁ and C₂
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the plane containing the circles
def Plane : Type := ℝ × ℝ

-- Define tangency between circles
def areTangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

-- Define the given conditions
def givenConditions (c1 c2 : Circle) : Prop :=
  c1.radius = 2 ∧ c2.radius = 2 ∧ areTangent c1 c2

-- Define a function to count tangent circles
def countTangentCircles (c1 c2 : Circle) : ℕ :=
  sorry

-- Theorem statement
theorem fourTangentCircles (c1 c2 : Circle) :
  givenConditions c1 c2 → countTangentCircles c1 c2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fourTangentCircles_l1170_117000


namespace NUMINAMATH_CALUDE_chocolate_cookie_price_l1170_117060

/-- Given the sale of chocolate and vanilla cookies, prove the price of a chocolate cookie. -/
theorem chocolate_cookie_price
  (chocolate_count : ℕ)
  (vanilla_count : ℕ)
  (vanilla_price : ℚ)
  (total_revenue : ℚ)
  (h1 : chocolate_count = 220)
  (h2 : vanilla_count = 70)
  (h3 : vanilla_price = 2)
  (h4 : total_revenue = 360)
  (h5 : total_revenue = chocolate_count * chocolate_price + vanilla_count * vanilla_price)
  : chocolate_price = 1 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_cookie_price_l1170_117060


namespace NUMINAMATH_CALUDE_smallest_n_for_powers_l1170_117052

theorem smallest_n_for_powers : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (a : ℕ), 3^n = a^4) ∧ 
  (∃ (b : ℕ), 2^n = b^6) ∧ 
  (∀ (m : ℕ), m > 0 → (∃ (c : ℕ), 3^m = c^4) → (∃ (d : ℕ), 2^m = d^6) → m ≥ n) ∧
  n = 12 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_powers_l1170_117052


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l1170_117026

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l1170_117026


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1170_117006

theorem simplify_and_rationalize :
  ∃ (x : ℝ), (x = (Real.sqrt 2 / Real.sqrt 5) * (Real.sqrt 3 / Real.sqrt 7) * (Real.rpow 4 (1/3) / Real.sqrt 6)) ∧
             (x = (Real.rpow 4 (1/3) * Real.sqrt 35) / 35) := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1170_117006


namespace NUMINAMATH_CALUDE_cube_equation_solution_l1170_117046

theorem cube_equation_solution (a n : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * n * 45 * 49) : n = 25 := by
  sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l1170_117046


namespace NUMINAMATH_CALUDE_electronic_product_pricing_l1170_117021

/-- The marked price of an electronic product -/
def marked_price : ℝ := 28

/-- The cost price of the electronic product -/
def cost_price : ℝ := 21

/-- The selling price ratio (90% of marked price) -/
def selling_price_ratio : ℝ := 0.9

/-- The profit ratio (20%) -/
def profit_ratio : ℝ := 0.2

theorem electronic_product_pricing :
  selling_price_ratio * marked_price - cost_price = profit_ratio * cost_price :=
by sorry

end NUMINAMATH_CALUDE_electronic_product_pricing_l1170_117021


namespace NUMINAMATH_CALUDE_average_equation_solution_l1170_117017

theorem average_equation_solution (x : ℝ) : 
  (1/3 : ℝ) * ((3*x + 4) + (7*x - 5) + (4*x + 9)) = 5*x - 3 → x = 17 := by
sorry

end NUMINAMATH_CALUDE_average_equation_solution_l1170_117017


namespace NUMINAMATH_CALUDE_sierra_age_l1170_117041

/-- Given that 40 less than 10 times Diaz's age is 20 more than 10 times Sierra's age,
    and Diaz will be 56 years old 20 years from now, prove that Sierra is currently 30 years old. -/
theorem sierra_age (diaz_age sierra_age : ℕ) 
  (h1 : 10 * diaz_age - 40 = 10 * sierra_age + 20)
  (h2 : diaz_age + 20 = 56) : 
  sierra_age = 30 := by sorry

end NUMINAMATH_CALUDE_sierra_age_l1170_117041


namespace NUMINAMATH_CALUDE_empty_solution_set_iff_a_in_range_l1170_117005

/-- The quadratic function f(x) = (a²-4)x² + (a+2)x - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := (a^2 - 4) * x^2 + (a + 2) * x - 1

/-- The set of x that satisfy the inequality f(x) ≥ 0 -/
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | f a x ≥ 0}

/-- The theorem stating the range of a for which the solution set is empty -/
theorem empty_solution_set_iff_a_in_range :
  ∀ a : ℝ, solution_set a = ∅ ↔ a ∈ Set.Icc (-2) (6/5) := by sorry

end NUMINAMATH_CALUDE_empty_solution_set_iff_a_in_range_l1170_117005


namespace NUMINAMATH_CALUDE_pear_price_l1170_117051

/-- Proves that the price of a pear is $60 given the conditions from the problem -/
theorem pear_price (orange pear banana : ℚ) 
  (h1 : orange - pear = banana)
  (h2 : orange + pear = 120)
  (h3 : 200 * banana + 400 * orange = 24000) : 
  pear = 60 := by
  sorry

end NUMINAMATH_CALUDE_pear_price_l1170_117051


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1170_117030

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1170_117030


namespace NUMINAMATH_CALUDE_point_P_and_min_value_l1170_117035

-- Define the points
def A : ℝ × ℝ := (3, 4)
def C : ℝ × ℝ := (0, 0)
def M : ℝ × ℝ := (0, 1)
def N : ℝ × ℝ := (1, 0)

-- Define vectors
def AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
def AM : ℝ × ℝ := (M.1 - A.1, M.2 - A.2)
def AN : ℝ × ℝ := (N.1 - A.1, N.2 - A.2)

-- Define the vector equation
def vector_equation (x y : ℝ) : Prop :=
  AC = (x * AM.1, x * AM.2) + (y * AN.1, y * AN.2)

-- Theorem statement
theorem point_P_and_min_value :
  ∃ (x y : ℝ), vector_equation x y ∧ 
  x = 2/3 ∧ y = 1/2 ∧ 
  ∀ (a b : ℝ), vector_equation a b → 9*x^2 + 16*y^2 ≤ 9*a^2 + 16*b^2 :=
sorry

end NUMINAMATH_CALUDE_point_P_and_min_value_l1170_117035


namespace NUMINAMATH_CALUDE_identity_proof_l1170_117083

theorem identity_proof (x : ℝ) : (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x := by
  sorry

end NUMINAMATH_CALUDE_identity_proof_l1170_117083


namespace NUMINAMATH_CALUDE_game_show_boxes_l1170_117034

theorem game_show_boxes (n : ℕ) (h1 : n > 0) : 
  (((n - 1 : ℝ) / n) ^ 3 = 0.2962962962962963) → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_game_show_boxes_l1170_117034


namespace NUMINAMATH_CALUDE_number_sum_theorem_l1170_117015

theorem number_sum_theorem :
  (∀ n : ℕ, n ≥ 100 → n ≥ smallest_three_digit) ∧
  (∀ n : ℕ, n < 100 → n ≤ largest_two_digit) ∧
  (∀ n : ℕ, n < 10 ∧ n % 2 = 1 → n ≥ smallest_odd_one_digit) ∧
  (∀ n : ℕ, n < 100 ∧ n % 2 = 0 → n ≤ largest_even_two_digit) →
  smallest_three_digit + largest_two_digit = 199 ∧
  smallest_odd_one_digit + largest_even_two_digit = 99 :=
by sorry

def smallest_three_digit : ℕ := 100
def largest_two_digit : ℕ := 99
def smallest_odd_one_digit : ℕ := 1
def largest_even_two_digit : ℕ := 98

end NUMINAMATH_CALUDE_number_sum_theorem_l1170_117015


namespace NUMINAMATH_CALUDE_degree_of_x2y_l1170_117014

/-- The degree of a monomial is the sum of the exponents of its variables -/
def degree_of_monomial (exponents : List ℕ) : ℕ :=
  exponents.sum

/-- The monomial x^2y has exponents [2, 1] -/
def monomial_x2y_exponents : List ℕ := [2, 1]

theorem degree_of_x2y :
  degree_of_monomial monomial_x2y_exponents = 3 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_x2y_l1170_117014


namespace NUMINAMATH_CALUDE_beach_house_rent_total_l1170_117071

theorem beach_house_rent_total (num_people : ℕ) (rent_per_person : ℚ) : 
  num_people = 7 → rent_per_person = 70 → num_people * rent_per_person = 490 := by
  sorry

end NUMINAMATH_CALUDE_beach_house_rent_total_l1170_117071


namespace NUMINAMATH_CALUDE_sum_of_two_smallest_numbers_l1170_117079

theorem sum_of_two_smallest_numbers (a b c d : ℝ) : 
  a / b = 3 / 5 ∧ 
  b / c = 5 / 7 ∧ 
  c / d = 7 / 9 ∧ 
  (a + b + c + d) / 4 = 30 →
  a + b = 40 := by
sorry

end NUMINAMATH_CALUDE_sum_of_two_smallest_numbers_l1170_117079


namespace NUMINAMATH_CALUDE_two_tails_in_seven_flips_l1170_117098

def unfair_coin_flip (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

theorem two_tails_in_seven_flips :
  unfair_coin_flip 7 2 (3/4) = 189/16384 := by
  sorry

end NUMINAMATH_CALUDE_two_tails_in_seven_flips_l1170_117098


namespace NUMINAMATH_CALUDE_root_in_interval_l1170_117004

noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem root_in_interval :
  ∃! x : ℝ, x ∈ Set.Ioo 5 6 ∧ log10 x = x - 5 :=
sorry

end NUMINAMATH_CALUDE_root_in_interval_l1170_117004


namespace NUMINAMATH_CALUDE_cricket_team_left_handed_fraction_l1170_117095

theorem cricket_team_left_handed_fraction :
  ∀ (total_players throwers right_handed : ℕ),
    total_players = 55 →
    throwers = 37 →
    right_handed = 49 →
    (total_players - throwers : ℚ) ≠ 0 →
    (left_handed_non_throwers : ℚ) / (total_players - throwers) = 1 / 3 :=
  λ total_players throwers right_handed
    h_total h_throwers h_right_handed h_non_zero ↦ by
  sorry

end NUMINAMATH_CALUDE_cricket_team_left_handed_fraction_l1170_117095


namespace NUMINAMATH_CALUDE_expand_binomial_product_simplify_algebraic_expression_expand_cubic_difference_l1170_117010

-- 1. Expansion of (2m-3)(5-3m)
theorem expand_binomial_product (m : ℝ) : 
  (2*m - 3) * (5 - 3*m) = -6*m^2 + 19*m - 15 := by sorry

-- 2. Simplification of (3a^3)^2⋅(2b^2)^3÷(6ab)^2
theorem simplify_algebraic_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (3*a^3)^2 * (2*b^2)^3 / (6*a*b)^2 = 2*a^4*b^4 := by sorry

-- 3. Expansion of (a-b)(a^2+ab+b^2)
theorem expand_cubic_difference (a b : ℝ) : 
  (a - b) * (a^2 + a*b + b^2) = a^3 - b^3 := by sorry

end NUMINAMATH_CALUDE_expand_binomial_product_simplify_algebraic_expression_expand_cubic_difference_l1170_117010


namespace NUMINAMATH_CALUDE_no_integer_solution_l1170_117077

theorem no_integer_solution (a b c d : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) :
  ¬∃ x : ℤ, x^4 - a*x^3 - b*x^2 - c*x - d = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1170_117077


namespace NUMINAMATH_CALUDE_parabola_focus_l1170_117050

/-- The parabola defined by the equation y^2 + 4x = 0 -/
def parabola (x y : ℝ) : Prop := y^2 + 4*x = 0

/-- The focus of a parabola -/
def focus (p : ℝ × ℝ) (parabola : ℝ → ℝ → Prop) : Prop := 
  ∃ (a b : ℝ), p = (a, b) ∧ 
  ∀ (x y : ℝ), parabola x y → (x - a)^2 + (y - b)^2 = (x + a)^2

theorem parabola_focus :
  focus (-1, 0) parabola := by sorry

end NUMINAMATH_CALUDE_parabola_focus_l1170_117050


namespace NUMINAMATH_CALUDE_comparison_theorem_l1170_117080

theorem comparison_theorem :
  (-7/8 : ℚ) < (-6/7 : ℚ) ∧ |(-0.1 : ℝ)| > (-0.2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_comparison_theorem_l1170_117080


namespace NUMINAMATH_CALUDE_locus_is_hyperbola_branch_l1170_117091

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 12 = 0

-- Define a circle tangent to both circles
def tangentCircle (cx cy r : ℝ) : Prop :=
  ∀ (x y : ℝ), circle1 x y → (x - cx)^2 + (y - cy)^2 = r^2 ∨
                circle2 x y → (x - cx)^2 + (y - cy)^2 = r^2

-- Define the locus of centers
def locusOfCenters (x y : ℝ) : Prop :=
  ∃ (r : ℝ), tangentCircle x y r

-- Theorem statement
theorem locus_is_hyperbola_branch :
  ∃ (a b : ℝ), ∀ (x y : ℝ), locusOfCenters x y ↔ (x^2 / a^2) - (y^2 / b^2) = 1 :=
sorry

end NUMINAMATH_CALUDE_locus_is_hyperbola_branch_l1170_117091


namespace NUMINAMATH_CALUDE_ratio_problem_l1170_117059

theorem ratio_problem (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (2 * A + 3 * B) / (A + 5 * C) = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1170_117059


namespace NUMINAMATH_CALUDE_trivia_team_groups_l1170_117086

theorem trivia_team_groups 
  (total_students : ℕ) 
  (not_picked : ℕ) 
  (students_per_group : ℕ) 
  (h1 : total_students = 36) 
  (h2 : not_picked = 9) 
  (h3 : students_per_group = 9) : 
  (total_students - not_picked) / students_per_group = 3 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_groups_l1170_117086


namespace NUMINAMATH_CALUDE_max_daily_sales_revenue_l1170_117022

def f (t : ℕ) : ℚ :=
  if t < 15 then (1/3) * t + 8 else -(1/3) * t + 18

def g (t : ℕ) : ℚ := -t + 30

def W (t : ℕ) : ℚ := (f t) * (g t)

theorem max_daily_sales_revenue (t : ℕ) (h : 0 < t ∧ t ≤ 30) : 
  W t ≤ 243 ∧ ∃ t₀ : ℕ, 0 < t₀ ∧ t₀ ≤ 30 ∧ W t₀ = 243 :=
sorry

end NUMINAMATH_CALUDE_max_daily_sales_revenue_l1170_117022


namespace NUMINAMATH_CALUDE_equation_solutions_l1170_117067

theorem equation_solutions : 
  ∀ x : ℝ, 2*x - 6 = 3*x*(x - 3) ↔ x = 3 ∨ x = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1170_117067


namespace NUMINAMATH_CALUDE_three_numbers_sum_l1170_117081

theorem three_numbers_sum (x y z : ℤ) 
  (sum_xy : x + y = 40)
  (sum_yz : y + z = 50)
  (sum_zx : z + x = 70) :
  x = 30 ∧ y = 10 ∧ z = 40 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l1170_117081


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_values_l1170_117023

theorem perpendicular_lines_m_values (m : ℝ) : 
  (∃ x y : ℝ, mx - y + 1 = 0 ∧ x + m^2*y - 2 = 0) →  -- lines exist
  (∀ x₁ y₁ x₂ y₂ : ℝ, mx₁ - y₁ + 1 = 0 ∧ x₁ + m^2*y₁ - 2 = 0 ∧ 
                      mx₂ - y₂ + 1 = 0 ∧ x₂ + m^2*y₂ - 2 = 0 →
    ((x₂ - x₁) * (mx₁ - y₁) + (y₂ - y₁) * (1 + m^2*y₁) = 0)) →  -- perpendicularity condition
  m = 0 ∨ m = 1 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_values_l1170_117023


namespace NUMINAMATH_CALUDE_reciprocal_sum_theorem_l1170_117033

theorem reciprocal_sum_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 5 * x * y) : 1 / x + 1 / y = 5 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_theorem_l1170_117033


namespace NUMINAMATH_CALUDE_opposite_numbers_linear_equation_l1170_117027

theorem opposite_numbers_linear_equation :
  ∀ x y : ℝ,
  (2 * x - 3 * y = 10) →
  (y = -x) →
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_opposite_numbers_linear_equation_l1170_117027


namespace NUMINAMATH_CALUDE_hcf_problem_l1170_117002

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 84942) (h2 : Nat.lcm a b = 2574) :
  Nat.gcd a b = 33 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l1170_117002


namespace NUMINAMATH_CALUDE_exchange_theorem_l1170_117028

/-- Represents the number of exchanges between Xiao Zhang and Xiao Li -/
def num_exchanges : ℕ := 4

/-- Xiao Zhang's initial number of pencils -/
def zhang_initial_pencils : ℕ := 200

/-- Xiao Li's initial number of fountain pens -/
def li_initial_pens : ℕ := 20

/-- Number of pencils Xiao Zhang gives in each exchange -/
def pencils_per_exchange : ℕ := 6

/-- Number of fountain pens Xiao Li gives in each exchange -/
def pens_per_exchange : ℕ := 1

/-- Xiao Zhang's final number of pencils -/
def zhang_final_pencils : ℕ := zhang_initial_pencils - num_exchanges * pencils_per_exchange

/-- Xiao Li's final number of fountain pens -/
def li_final_pens : ℕ := li_initial_pens - num_exchanges * pens_per_exchange

theorem exchange_theorem :
  zhang_final_pencils = 11 * li_final_pens :=
by sorry

end NUMINAMATH_CALUDE_exchange_theorem_l1170_117028


namespace NUMINAMATH_CALUDE_relationship_a_x_l1170_117090

theorem relationship_a_x (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 + b^3 = 14*x^3) 
  (h3 : a + b = x) : 
  a = (Real.sqrt 165 - 3) / 6 * x ∨ a = -(Real.sqrt 165 + 3) / 6 * x := by
  sorry

end NUMINAMATH_CALUDE_relationship_a_x_l1170_117090


namespace NUMINAMATH_CALUDE_mixture_ratio_theorem_l1170_117016

/-- Represents the ratio of alcohol to water in a mixture -/
structure AlcoholRatio where
  alcohol : ℝ
  water : ℝ

/-- Calculates the ratio of alcohol to water when mixing two solutions -/
def mixSolutions (v1 v2 : ℝ) (r1 r2 : AlcoholRatio) : AlcoholRatio :=
  { alcohol := v1 * r1.alcohol + v2 * r2.alcohol,
    water := v1 * r1.water + v2 * r2.water }

theorem mixture_ratio_theorem (V p q : ℝ) (hp : p > 0) (hq : q > 0) :
  let jar1 := AlcoholRatio.mk (p / (p + 2)) (2 / (p + 2))
  let jar2 := AlcoholRatio.mk (q / (q + 1)) (1 / (q + 1))
  let mixture := mixSolutions V (2 * V) jar1 jar2
  mixture.alcohol / mixture.water = (p * (q + 1) + 4 * q * (p + 2)) / (2 * (q + 1) + 4 * (p + 2)) := by
  sorry

#check mixture_ratio_theorem

end NUMINAMATH_CALUDE_mixture_ratio_theorem_l1170_117016


namespace NUMINAMATH_CALUDE_quadratic_discriminant_positive_l1170_117049

theorem quadratic_discriminant_positive 
  (a b c : ℝ) 
  (h : (a + b + c) * c < 0) : 
  b^2 - 4*a*c > 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_positive_l1170_117049


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_values_l1170_117070

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := (a - 2) * x + 3 * y + a = 0
def l₂ (a x y : ℝ) : Prop := a * x + (a - 2) * y - 1 = 0

-- Define perpendicularity condition
def perpendicular (a : ℝ) : Prop := (a - 2) * a + 3 * (a - 2) = 0

-- Theorem statement
theorem perpendicular_lines_a_values :
  ∀ a : ℝ, perpendicular a → a = 2 ∨ a = -3 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_values_l1170_117070


namespace NUMINAMATH_CALUDE_janines_earnings_l1170_117011

/-- Represents the day of the week --/
inductive Day
| Monday
| Tuesday
| Thursday
| Saturday

/-- Calculates the pay rate for a given day --/
def payRate (d : Day) : ℚ :=
  match d with
  | Day.Monday => 4
  | Day.Tuesday => 4
  | Day.Thursday => 4
  | Day.Saturday => 5

/-- Calculates the bonus rate for a given day and hours worked --/
def bonusRate (hours : ℚ) : ℚ :=
  if hours > 2 then 1 else 0

/-- Calculates the earnings for a single day --/
def dailyEarnings (d : Day) (hours : ℚ) : ℚ :=
  hours * (payRate d + bonusRate hours)

/-- Janine's work schedule --/
def schedule : List (Day × ℚ) :=
  [(Day.Monday, 2), (Day.Tuesday, 3/2), (Day.Thursday, 7/2), (Day.Saturday, 5/2)]

/-- Theorem: Janine's total earnings for the week equal $46.5 --/
theorem janines_earnings :
  (schedule.map (fun (d, h) => dailyEarnings d h)).sum = 93/2 := by
  sorry

end NUMINAMATH_CALUDE_janines_earnings_l1170_117011


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l1170_117073

/-- The parabola P with equation y = x^2 -/
def P : Set (ℝ × ℝ) := {p | p.2 = p.1^2}

/-- The point Q -/
def Q : ℝ × ℝ := (10, 6)

/-- The line through Q with slope n -/
def line (n : ℝ) : Set (ℝ × ℝ) := {p | p.2 - Q.2 = n * (p.1 - Q.1)}

/-- The condition for non-intersection -/
def non_intersect (n : ℝ) : Prop := line n ∩ P = ∅

/-- The theorem to be proved -/
theorem parabola_line_intersection :
  ∃ (a b : ℝ), (∀ n, non_intersect n ↔ a < n ∧ n < b) → a + b = 40 := by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l1170_117073


namespace NUMINAMATH_CALUDE_b_investment_is_200_l1170_117038

/-- Represents the investment scenario with two investors A and B --/
structure Investment where
  a_amount : ℝ  -- A's investment amount
  b_amount : ℝ  -- B's investment amount
  a_months : ℝ  -- Months A's money was invested
  b_months : ℝ  -- Months B's money was invested
  total_profit : ℝ  -- Total profit at the end of the year
  a_profit : ℝ  -- A's share of the profit

/-- The theorem stating that B's investment is $200 given the conditions --/
theorem b_investment_is_200 (inv : Investment) 
  (h1 : inv.a_amount = 150)
  (h2 : inv.a_months = 12)
  (h3 : inv.b_months = 6)
  (h4 : inv.total_profit = 100)
  (h5 : inv.a_profit = 60)
  (h6 : inv.a_profit / inv.total_profit = 
        (inv.a_amount * inv.a_months) / 
        (inv.a_amount * inv.a_months + inv.b_amount * inv.b_months)) :
  inv.b_amount = 200 := by
  sorry


end NUMINAMATH_CALUDE_b_investment_is_200_l1170_117038


namespace NUMINAMATH_CALUDE_alien_artifact_age_conversion_l1170_117012

/-- Converts a number from base 8 to base 10 -/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 16 -/
def decimal_to_hex (n : ℕ) : String := sorry

/-- Represents the age in octal -/
def age_octal : ℕ := 7231

/-- The expected result in hexadecimal -/
def expected_hex : String := "E99"

theorem alien_artifact_age_conversion :
  decimal_to_hex (octal_to_decimal age_octal) = expected_hex := by sorry

end NUMINAMATH_CALUDE_alien_artifact_age_conversion_l1170_117012


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l1170_117062

theorem decimal_to_fraction : 
  (0.32 : ℚ) = 8 / 25 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l1170_117062


namespace NUMINAMATH_CALUDE_sum_of_digits_l1170_117093

/-- Given a three-digit number of the form 4a4, where 'a' is a single digit,
    we add 258 to it to get a three-digit number of the form 7b2,
    where 'b' is also a single digit. If 7b2 is divisible by 3,
    then a + b = 4. -/
theorem sum_of_digits (a b : Nat) : 
  (a ≥ 0 ∧ a ≤ 9) →  -- 'a' is a single digit
  (b ≥ 0 ∧ b ≤ 9) →  -- 'b' is a single digit
  (400 + 10 * a + 4) + 258 = 700 + 10 * b + 2 →  -- 4a4 + 258 = 7b2
  (700 + 10 * b + 2) % 3 = 0 →  -- 7b2 is divisible by 3
  a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_l1170_117093


namespace NUMINAMATH_CALUDE_ceiling_neg_seven_fourths_squared_l1170_117044

theorem ceiling_neg_seven_fourths_squared : ⌈(-7/4)^2⌉ = 4 := by sorry

end NUMINAMATH_CALUDE_ceiling_neg_seven_fourths_squared_l1170_117044


namespace NUMINAMATH_CALUDE_vinay_position_from_right_l1170_117054

/-- Represents the position of a boy in a row. -/
structure Position where
  fromLeft : Nat
  fromRight : Nat
  total : Nat
  valid : fromLeft + fromRight = total + 1

/-- Given the conditions of the problem, calculate Vinay's position. -/
def vinayPosition (totalBoys : Nat) (rajanFromLeft : Nat) (betweenRajanAndVinay : Nat) : Position :=
  { fromLeft := rajanFromLeft + betweenRajanAndVinay + 1,
    fromRight := totalBoys - (rajanFromLeft + betweenRajanAndVinay),
    total := totalBoys,
    valid := by sorry }

/-- The main theorem to be proved. -/
theorem vinay_position_from_right :
  let p := vinayPosition 24 6 8
  p.fromRight = 9 := by sorry

end NUMINAMATH_CALUDE_vinay_position_from_right_l1170_117054


namespace NUMINAMATH_CALUDE_augmented_matrix_proof_l1170_117032

def system_of_equations : List (List ℝ) := [[1, -2, 5], [3, 1, 8]]

theorem augmented_matrix_proof :
  let eq1 := λ x y : ℝ => x - 2*y = 5
  let eq2 := λ x y : ℝ => 3*x + y = 8
  system_of_equations = 
    (λ (f g : ℝ → ℝ → ℝ) => 
      [[f 1 (-2), f (-2) 1, 5],
       [g 3 1, g 1 3, 8]])
    (λ a b => a)
    (λ a b => b) := by sorry

end NUMINAMATH_CALUDE_augmented_matrix_proof_l1170_117032


namespace NUMINAMATH_CALUDE_turtle_conservation_count_l1170_117078

theorem turtle_conservation_count :
  let green_turtles : ℕ := 800
  let hawksbill_turtles : ℕ := 2 * green_turtles
  let total_turtles : ℕ := green_turtles + hawksbill_turtles
  total_turtles = 2400 :=
by sorry

end NUMINAMATH_CALUDE_turtle_conservation_count_l1170_117078


namespace NUMINAMATH_CALUDE_frank_saturday_bags_l1170_117029

def total_cans : ℕ := 40
def cans_per_bag : ℕ := 5
def bags_filled_sunday : ℕ := 3

def total_bags : ℕ := total_cans / cans_per_bag

def bags_filled_saturday : ℕ := total_bags - bags_filled_sunday

theorem frank_saturday_bags :
  bags_filled_saturday = 5 :=
by sorry

end NUMINAMATH_CALUDE_frank_saturday_bags_l1170_117029


namespace NUMINAMATH_CALUDE_teachers_not_adjacent_arrangements_l1170_117047

/-- The number of arrangements of n distinct objects taken r at a time -/
def A (n : ℕ) (r : ℕ) : ℕ := sorry

/-- The number of students -/
def num_students : ℕ := 8

/-- The number of teachers -/
def num_teachers : ℕ := 2

/-- The number of gaps between and around students -/
def num_gaps : ℕ := num_students + 1

theorem teachers_not_adjacent_arrangements :
  (A num_students num_students) * (A num_gaps num_teachers) =
  (A num_students num_students) * (A 9 2) := by sorry

end NUMINAMATH_CALUDE_teachers_not_adjacent_arrangements_l1170_117047


namespace NUMINAMATH_CALUDE_multiples_count_multiples_of_4_or_5_not_20_l1170_117092

theorem multiples_count : Nat → Nat :=
  fun n => (n / 4 + n / 5 - n / 20)

theorem multiples_of_4_or_5_not_20 (upper_bound : Nat) 
  (h : upper_bound = 1500) : 
  multiples_count upper_bound = 600 := by
  sorry

end NUMINAMATH_CALUDE_multiples_count_multiples_of_4_or_5_not_20_l1170_117092


namespace NUMINAMATH_CALUDE_angle_ratio_not_determine_right_angle_l1170_117024

/-- Triangle ABC with angles A, B, and C -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Definition of a right-angled triangle -/
def IsRightAngled (t : Triangle) : Prop :=
  t.A = 90 ∨ t.B = 90 ∨ t.C = 90

/-- Theorem: Given the angle ratio A:B:C = 3:4:5, it cannot be determined that the triangle is right-angled -/
theorem angle_ratio_not_determine_right_angle (t : Triangle) 
  (h : t.A / 3 = t.B / 4 ∧ t.B / 4 = t.C / 5) : 
  ¬(IsRightAngled t ↔ True) := by
  sorry

#check angle_ratio_not_determine_right_angle

end NUMINAMATH_CALUDE_angle_ratio_not_determine_right_angle_l1170_117024


namespace NUMINAMATH_CALUDE_triangle_radius_inequality_l1170_117025

/-- For any triangle with circumradius R, inradius r, and semiperimeter p,
    the inequality 27Rr ≤ 2p² ≤ 27R²/2 holds. -/
theorem triangle_radius_inequality (R r p : ℝ) (h_positive : R > 0 ∧ r > 0 ∧ p > 0) 
  (h_triangle : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    p = (a + b + c) / 2 ∧ 
    R = (a * b * c) / (4 * (p * (p - a) * (p - b) * (p - c))^(1/2)) ∧
    r = (p * (p - a) * (p - b) * (p - c))^(1/2) / p) :
  27 * R * r ≤ 2 * p^2 ∧ 2 * p^2 ≤ 27 * R^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_radius_inequality_l1170_117025


namespace NUMINAMATH_CALUDE_two_zeros_implies_m_values_l1170_117082

/-- A cubic function with a parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x + m

/-- The statement that f has exactly two zeros -/
def has_two_zeros (m : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ f m x = 0 ∧ f m y = 0 ∧ ∀ z : ℝ, f m z = 0 → z = x ∨ z = y

/-- The theorem stating that if f has exactly two zeros, then m = -2 or m = 2 -/
theorem two_zeros_implies_m_values (m : ℝ) :
  has_two_zeros m → m = -2 ∨ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_zeros_implies_m_values_l1170_117082


namespace NUMINAMATH_CALUDE_quadratic_root_implies_u_l1170_117063

theorem quadratic_root_implies_u (u : ℝ) : 
  (4 * (((-15 + Real.sqrt 205) / 8) ^ 2) + 15 * ((-15 + Real.sqrt 205) / 8) + u = 0) → 
  u = 5/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_u_l1170_117063


namespace NUMINAMATH_CALUDE_no_integer_solution_quadratic_prime_l1170_117064

theorem no_integer_solution_quadratic_prime : 
  ¬ ∃ (x : ℤ), Nat.Prime (Int.natAbs (4 * x^2 - 39 * x + 35)) := by
sorry

end NUMINAMATH_CALUDE_no_integer_solution_quadratic_prime_l1170_117064


namespace NUMINAMATH_CALUDE_quadricycles_count_l1170_117066

/-- Given a total number of children and wheels, calculate the number of quadricycles -/
def count_quadricycles (total_children : ℕ) (total_wheels : ℕ) : ℕ :=
  let scooter_wheels := 2
  let quadricycle_wheels := 4
  let quadricycles := (total_wheels - scooter_wheels * total_children) / (quadricycle_wheels - scooter_wheels)
  quadricycles

/-- Theorem stating that given 9 children and 30 wheels, there are 6 quadricycles -/
theorem quadricycles_count : count_quadricycles 9 30 = 6 := by
  sorry

#eval count_quadricycles 9 30

end NUMINAMATH_CALUDE_quadricycles_count_l1170_117066


namespace NUMINAMATH_CALUDE_area_enclosed_circles_l1170_117069

/-- The area enclosed between the circumferences of four equal circles described about the corners of a square -/
theorem area_enclosed_circles (s : ℝ) (h : s = 14) :
  let r : ℝ := s / 2
  let square_area : ℝ := s ^ 2
  let circle_segment_area : ℝ := π * r ^ 2
  square_area - circle_segment_area = 196 - 49 * π :=
by sorry

end NUMINAMATH_CALUDE_area_enclosed_circles_l1170_117069


namespace NUMINAMATH_CALUDE_large_duck_cost_large_duck_cost_proof_l1170_117088

/-- The cost of a large size duck given the following conditions:
  * Regular size ducks cost $3.00 each
  * 221 regular size ducks were sold
  * 185 large size ducks were sold
  * Total amount raised is $1588
-/
theorem large_duck_cost : ℝ → Prop :=
  λ large_cost : ℝ =>
    let regular_cost : ℝ := 3
    let regular_sold : ℕ := 221
    let large_sold : ℕ := 185
    let total_raised : ℝ := 1588
    (regular_cost * regular_sold + large_cost * large_sold = total_raised) →
    large_cost = 5

/-- Proof of the large duck cost theorem -/
theorem large_duck_cost_proof : large_duck_cost 5 := by
  sorry

end NUMINAMATH_CALUDE_large_duck_cost_large_duck_cost_proof_l1170_117088


namespace NUMINAMATH_CALUDE_allison_win_probability_l1170_117084

-- Define the faces of each cube
def allison_cube : Finset Nat := {4, 4, 4, 4, 4, 4}
def charlie_cube : Finset Nat := {1, 1, 2, 3, 4, 5}
def dani_cube : Finset Nat := {3, 3, 3, 3, 5, 5}

-- Define the probability of rolling each number for each person
def prob_roll (cube : Finset Nat) (n : Nat) : Rat :=
  (cube.filter (· = n)).card / cube.card

-- Define the event of Allison winning
def allison_wins (c : Nat) (d : Nat) : Prop :=
  4 > c ∧ 4 > d

-- Theorem statement
theorem allison_win_probability :
  (prob_roll charlie_cube 1 + prob_roll charlie_cube 2) *
  (prob_roll dani_cube 3) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_allison_win_probability_l1170_117084


namespace NUMINAMATH_CALUDE_sibling_ages_sum_l1170_117037

theorem sibling_ages_sum (a b c : ℕ+) : 
  a = b ∧ a < c ∧ a * b * c = 72 → a + b + c = 14 :=
by sorry

end NUMINAMATH_CALUDE_sibling_ages_sum_l1170_117037


namespace NUMINAMATH_CALUDE_not_corner_2010_l1170_117007

/-- Represents the sequence of corner numbers in the spiral -/
def corner_sequence : ℕ → ℕ
| 0 => 2
| 1 => 4
| n + 2 => corner_sequence (n + 1) + 8 * (n + 1)

/-- Checks if a number is a corner number in the spiral -/
def is_corner_number (n : ℕ) : Prop :=
  ∃ k : ℕ, corner_sequence k = n

/-- The main theorem stating that 2010 is not a corner number -/
theorem not_corner_2010 : ¬ is_corner_number 2010 := by
  sorry

#eval corner_sequence 0  -- Expected: 2
#eval corner_sequence 1  -- Expected: 4
#eval corner_sequence 2  -- Expected: 6
#eval corner_sequence 3  -- Expected: 10

end NUMINAMATH_CALUDE_not_corner_2010_l1170_117007


namespace NUMINAMATH_CALUDE_solve_equation_l1170_117089

theorem solve_equation (C D : ℚ) 
  (eq1 : 5 * C + 3 * D - 4 = 47) 
  (eq2 : C = D + 2) : 
  C = 57 / 8 ∧ D = 41 / 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1170_117089


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1170_117008

def A : Set ℕ := {2, 4, 6, 16, 29}
def B : Set ℕ := {4, 16, 20, 27, 29, 32}

theorem intersection_of_A_and_B : A ∩ B = {4, 16, 29} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1170_117008


namespace NUMINAMATH_CALUDE_incorrect_calculation_l1170_117045

theorem incorrect_calculation : ¬(3 + 2 * Real.sqrt 2 = 5 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_calculation_l1170_117045


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_less_than_hundred_l1170_117048

theorem greatest_multiple_of_four_less_than_hundred : 
  ∀ n : ℕ, n % 4 = 0 ∧ n < 100 → n ≤ 96 :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_less_than_hundred_l1170_117048


namespace NUMINAMATH_CALUDE_work_completion_problem_l1170_117019

theorem work_completion_problem (first_group_days : ℕ) (second_group_men : ℕ) (second_group_days : ℕ) :
  first_group_days = 18 →
  second_group_men = 108 →
  second_group_days = 6 →
  ∃ (first_group_men : ℕ), first_group_men * first_group_days = second_group_men * second_group_days ∧ first_group_men = 36 :=
by
  sorry


end NUMINAMATH_CALUDE_work_completion_problem_l1170_117019


namespace NUMINAMATH_CALUDE_ivan_bought_ten_cards_l1170_117057

/-- The number of Uno Giant Family Cards Ivan bought -/
def num_cards : ℕ := 10

/-- The original price of each card in dollars -/
def original_price : ℚ := 12

/-- The discount per card in dollars -/
def discount : ℚ := 2

/-- The total amount Ivan paid in dollars -/
def total_paid : ℚ := 100

/-- Theorem stating that Ivan bought 10 Uno Giant Family Cards -/
theorem ivan_bought_ten_cards :
  (original_price - discount) * num_cards = total_paid :=
by sorry

end NUMINAMATH_CALUDE_ivan_bought_ten_cards_l1170_117057


namespace NUMINAMATH_CALUDE_umbrella_cost_l1170_117042

theorem umbrella_cost (house_umbrellas car_umbrellas total_cost : ℕ) 
  (h1 : house_umbrellas = 2)
  (h2 : car_umbrellas = 1)
  (h3 : total_cost = 24) :
  total_cost / (house_umbrellas + car_umbrellas) = 8 := by
  sorry

end NUMINAMATH_CALUDE_umbrella_cost_l1170_117042


namespace NUMINAMATH_CALUDE_lemonade_cups_calculation_l1170_117068

theorem lemonade_cups_calculation (sugar_cups : ℕ) (ratio : ℚ) : 
  sugar_cups = 28 → ratio = 1/2 → sugar_cups + (sugar_cups / ratio) = 84 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_cups_calculation_l1170_117068


namespace NUMINAMATH_CALUDE_sector_area_l1170_117013

theorem sector_area (θ : Real) (r : Real) (h1 : θ = π / 3) (h2 : r = 2) :
  (1 / 2) * θ * r^2 = (2 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1170_117013


namespace NUMINAMATH_CALUDE_diamond_square_counts_l1170_117087

/-- Represents a diamond-shaped arrangement of colored squares -/
structure DiamondArrangement where
  sideLength : ℕ
  totalSquares : ℕ
  greenSquares : ℕ
  whiteSquares : ℕ

/-- Properties of the diamond arrangement -/
def validDiamondArrangement (d : DiamondArrangement) : Prop :=
  d.sideLength = 4 ∧
  d.totalSquares = (2 * d.sideLength + 1)^2 ∧
  d.greenSquares = (d.totalSquares + 1) / 2 ∧
  d.whiteSquares = (d.totalSquares - 1) / 2

theorem diamond_square_counts (d : DiamondArrangement) 
  (h : validDiamondArrangement d) : 
  d.whiteSquares = 40 ∧ 
  d.greenSquares = 41 ∧ 
  100 * d.whiteSquares + d.greenSquares = 4041 := by
  sorry

end NUMINAMATH_CALUDE_diamond_square_counts_l1170_117087


namespace NUMINAMATH_CALUDE_sum_geq_five_x_squared_l1170_117065

theorem sum_geq_five_x_squared (x : ℝ) (hx : x > 0) :
  1 + x + x^2 + x^3 + x^4 ≥ 5 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_geq_five_x_squared_l1170_117065


namespace NUMINAMATH_CALUDE_decimal_sum_and_multiply_l1170_117009

theorem decimal_sum_and_multiply : 
  let a : ℚ := 0.0034
  let b : ℚ := 0.125
  let c : ℚ := 0.00678
  2 * (a + b + c) = 0.27036 := by
sorry

end NUMINAMATH_CALUDE_decimal_sum_and_multiply_l1170_117009
