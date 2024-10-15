import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_powers_of_three_l3793_379334

theorem sum_of_powers_of_three : (-3)^4 + (-3)^2 + (-3)^0 + 3^0 + 3^2 + 3^4 = 182 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_three_l3793_379334


namespace NUMINAMATH_CALUDE_max_rock_value_is_58_l3793_379308

/-- Represents a type of rock with its weight and value -/
structure Rock where
  weight : ℕ
  value : ℕ

/-- Calculates the maximum value of rocks that can be carried given the constraints -/
def maxRockValue (rocks : List Rock) (maxWeight : ℕ) (maxSixPoundRocks : ℕ) : ℕ :=
  sorry

/-- The main theorem stating the maximum value of rocks Carl can carry -/
theorem max_rock_value_is_58 :
  let rocks : List Rock := [
    { weight := 3, value := 9 },
    { weight := 6, value := 20 },
    { weight := 2, value := 5 }
  ]
  let maxWeight : ℕ := 20
  let maxSixPoundRocks : ℕ := 2
  maxRockValue rocks maxWeight maxSixPoundRocks = 58 := by
  sorry

end NUMINAMATH_CALUDE_max_rock_value_is_58_l3793_379308


namespace NUMINAMATH_CALUDE_first_friend_shells_l3793_379374

/-- Proves the amount of shells added by the first friend given initial conditions --/
theorem first_friend_shells (initial_shells : ℕ) (second_friend_shells : ℕ) (total_shells : ℕ)
  (h1 : initial_shells = 5)
  (h2 : second_friend_shells = 17)
  (h3 : total_shells = 37)
  : total_shells - initial_shells - second_friend_shells = 15 := by
  sorry

end NUMINAMATH_CALUDE_first_friend_shells_l3793_379374


namespace NUMINAMATH_CALUDE_octopus_family_total_l3793_379346

/-- Represents the number of children of each color in the octopus family -/
structure OctopusFamily :=
  (white : ℕ)
  (blue : ℕ)
  (striped : ℕ)

/-- The conditions of the octopus family problem -/
def octopusFamilyConditions (initial final : OctopusFamily) : Prop :=
  -- Initially, there were equal numbers of white, blue, and striped octopus children
  initial.white = initial.blue ∧ initial.blue = initial.striped
  -- Some blue octopus children became striped
  ∧ final.blue < initial.blue
  ∧ final.striped > initial.striped
  -- After the change, the total number of blue and white octopus children was 10
  ∧ final.blue + final.white = 10
  -- After the change, the total number of white and striped octopus children was 18
  ∧ final.white + final.striped = 18
  -- The total number of children remains constant
  ∧ initial.white + initial.blue + initial.striped = final.white + final.blue + final.striped

/-- The theorem stating that under the given conditions, the total number of children is 21 -/
theorem octopus_family_total (initial final : OctopusFamily) :
  octopusFamilyConditions initial final →
  final.white + final.blue + final.striped = 21 :=
by
  sorry


end NUMINAMATH_CALUDE_octopus_family_total_l3793_379346


namespace NUMINAMATH_CALUDE_choir_members_count_l3793_379351

theorem choir_members_count : ∃! n : ℕ, 
  150 ≤ n ∧ n ≤ 300 ∧ 
  n % 10 = 6 ∧ 
  n % 11 = 6 := by
sorry

end NUMINAMATH_CALUDE_choir_members_count_l3793_379351


namespace NUMINAMATH_CALUDE_digit_table_size_l3793_379333

/-- A table with digits -/
structure DigitTable where
  rows : ℕ
  cols : ℕ
  digits : Fin rows → Fin cols → Fin 10

/-- The property that for any row and any two columns, there exists another row
    that differs only in those two columns -/
def hasTwoColumnDifference (t : DigitTable) : Prop :=
  ∀ (r : Fin t.rows) (c₁ c₂ : Fin t.cols),
    c₁ ≠ c₂ →
    ∃ (r' : Fin t.rows),
      r' ≠ r ∧
      (∀ (c : Fin t.cols), c ≠ c₁ ∧ c ≠ c₂ → t.digits r c = t.digits r' c) ∧
      (t.digits r c₁ ≠ t.digits r' c₁ ∨ t.digits r c₂ ≠ t.digits r' c₂)

/-- The main theorem -/
theorem digit_table_size (t : DigitTable) (h : t.cols = 10) (p : hasTwoColumnDifference t) :
  t.rows ≥ 512 := by
  sorry

end NUMINAMATH_CALUDE_digit_table_size_l3793_379333


namespace NUMINAMATH_CALUDE_curve_length_l3793_379379

/-- The length of a curve defined by the intersection of a plane and a sphere --/
theorem curve_length (x y z : ℝ) : 
  x + y + z = 8 → 
  x * y + y * z + x * z = -18 → 
  (∃ (l : ℝ), l = 4 * Real.pi * Real.sqrt (59 / 3) ∧ 
    l = 2 * Real.pi * Real.sqrt (100 - (8 * 8) / 3)) := by
  sorry

end NUMINAMATH_CALUDE_curve_length_l3793_379379


namespace NUMINAMATH_CALUDE_gcd_of_36_and_54_l3793_379388

theorem gcd_of_36_and_54 : Nat.gcd 36 54 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_36_and_54_l3793_379388


namespace NUMINAMATH_CALUDE_circle_polygons_l3793_379313

/-- The number of points marked on the circle -/
def n : ℕ := 12

/-- The number of distinct convex polygons with 3 or more sides -/
def num_polygons : ℕ := 2^n - (n.choose 0 + n.choose 1 + n.choose 2)

theorem circle_polygons :
  num_polygons = 4017 :=
sorry

end NUMINAMATH_CALUDE_circle_polygons_l3793_379313


namespace NUMINAMATH_CALUDE_environmental_legislation_support_l3793_379357

theorem environmental_legislation_support (men : ℕ) (women : ℕ) 
  (men_support_rate : ℚ) (women_support_rate : ℚ) :
  men = 200 →
  women = 1200 →
  men_support_rate = 70 / 100 →
  women_support_rate = 75 / 100 →
  let total_surveyed := men + women
  let total_supporters := men * men_support_rate + women * women_support_rate
  let overall_support_rate := total_supporters / total_surveyed
  ‖overall_support_rate - 74 / 100‖ < 1 / 100 :=
by sorry

end NUMINAMATH_CALUDE_environmental_legislation_support_l3793_379357


namespace NUMINAMATH_CALUDE_modulus_of_z_is_sqrt_2_l3793_379306

theorem modulus_of_z_is_sqrt_2 :
  let z : ℂ := 1 - 1 / Complex.I
  Complex.abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_z_is_sqrt_2_l3793_379306


namespace NUMINAMATH_CALUDE_salary_after_four_months_l3793_379316

def salary_calculation (initial_salary : ℝ) (initial_increase_rate : ℝ) (initial_bonus : ℝ) (bonus_increase_rate : ℝ) (months : ℕ) : ℝ :=
  let rec helper (current_salary : ℝ) (current_bonus : ℝ) (current_increase_rate : ℝ) (month : ℕ) : ℝ :=
    if month = 0 then
      current_salary + current_bonus
    else
      let new_salary := current_salary * (1 + current_increase_rate)
      let new_bonus := current_bonus * (1 + bonus_increase_rate)
      let new_increase_rate := current_increase_rate * 2
      helper new_salary new_bonus new_increase_rate (month - 1)
  helper initial_salary initial_bonus initial_increase_rate months

theorem salary_after_four_months :
  salary_calculation 2000 0.05 150 0.1 4 = 4080.45 := by
  sorry

end NUMINAMATH_CALUDE_salary_after_four_months_l3793_379316


namespace NUMINAMATH_CALUDE_system_inequalities_solution_l3793_379381

theorem system_inequalities_solution : 
  {x : ℕ | 4 * (x + 1) ≤ 7 * x + 10 ∧ x - 5 < (x - 8) / 3} = {0, 1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_system_inequalities_solution_l3793_379381


namespace NUMINAMATH_CALUDE_calculate_expression_solve_system_of_equations_l3793_379361

-- Part 1
theorem calculate_expression : 
  (6 - 2 * Real.sqrt 3) * Real.sqrt 3 - Real.sqrt ((2 - Real.sqrt 2)^2) + 1 / Real.sqrt 2 = 
  6 * Real.sqrt 3 - 8 + 3 * Real.sqrt 2 / 2 := by sorry

-- Part 2
theorem solve_system_of_equations (x y : ℝ) :
  5 * x - y = -9 ∧ 3 * x + y = 1 → x = -1 ∧ y = 4 := by sorry

end NUMINAMATH_CALUDE_calculate_expression_solve_system_of_equations_l3793_379361


namespace NUMINAMATH_CALUDE_divisor_problem_l3793_379371

theorem divisor_problem (x d : ℤ) (h1 : x % d = 7) (h2 : (x + 11) % 31 = 18) (h3 : d > 7) : d = 31 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l3793_379371


namespace NUMINAMATH_CALUDE_horses_oats_meals_per_day_l3793_379363

/-- The number of horses Peter has -/
def num_horses : ℕ := 4

/-- The amount of oats each horse eats per meal (in pounds) -/
def oats_per_meal : ℕ := 4

/-- The amount of grain each horse eats per day (in pounds) -/
def grain_per_day : ℕ := 3

/-- The total amount of food needed for all horses for 3 days (in pounds) -/
def total_food_3days : ℕ := 132

/-- The number of days food is needed for -/
def num_days : ℕ := 3

/-- The number of times horses eat oats per day -/
def oats_meals_per_day : ℕ := 2

theorem horses_oats_meals_per_day : 
  num_days * num_horses * (oats_per_meal * oats_meals_per_day + grain_per_day) = total_food_3days :=
by sorry

end NUMINAMATH_CALUDE_horses_oats_meals_per_day_l3793_379363


namespace NUMINAMATH_CALUDE_tom_coins_value_l3793_379386

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a penny in dollars -/
def penny_value : ℚ := 0.01

/-- The number of quarters Tom found -/
def num_quarters : ℕ := 10

/-- The number of dimes Tom found -/
def num_dimes : ℕ := 3

/-- The number of nickels Tom found -/
def num_nickels : ℕ := 4

/-- The number of pennies Tom found -/
def num_pennies : ℕ := 200

/-- The total value of the coins Tom found in dollars -/
def total_value : ℚ := num_quarters * quarter_value + num_dimes * dime_value + 
                       num_nickels * nickel_value + num_pennies * penny_value

theorem tom_coins_value : total_value = 5 := by
  sorry

end NUMINAMATH_CALUDE_tom_coins_value_l3793_379386


namespace NUMINAMATH_CALUDE_dot_product_FA_AB_is_zero_l3793_379397

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 16 * x

-- Define the focus F
def focus : ℝ × ℝ := (4, 0)

-- Define point A on y-axis with |OA| = |OF|
def point_A : ℝ × ℝ := (0, 4)  -- We choose the positive y-coordinate

-- Define point B as intersection of directrix and x-axis
def point_B : ℝ × ℝ := (-4, 0)

-- Define vector FA
def vector_FA : ℝ × ℝ := (point_A.1 - focus.1, point_A.2 - focus.2)

-- Define vector AB
def vector_AB : ℝ × ℝ := (point_B.1 - point_A.1, point_B.2 - point_A.2)

-- Theorem statement
theorem dot_product_FA_AB_is_zero :
  vector_FA.1 * vector_AB.1 + vector_FA.2 * vector_AB.2 = 0 :=
sorry

end NUMINAMATH_CALUDE_dot_product_FA_AB_is_zero_l3793_379397


namespace NUMINAMATH_CALUDE_alyssas_total_spent_l3793_379330

/-- The amount Alyssa paid for grapes in dollars -/
def grapes_cost : ℚ := 12.08

/-- The amount Alyssa was refunded for cherries in dollars -/
def cherries_refund : ℚ := 9.85

/-- The total amount Alyssa spent in dollars -/
def total_spent : ℚ := grapes_cost - cherries_refund

/-- Theorem stating that the total amount Alyssa spent is $2.23 -/
theorem alyssas_total_spent : total_spent = 2.23 := by
  sorry

end NUMINAMATH_CALUDE_alyssas_total_spent_l3793_379330


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3793_379337

/-- Given an arithmetic sequence {a_n} where a_2 + a_3 + a_10 + a_11 = 48, prove that a_6 + a_7 = 24 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h2 : a 2 + a 3 + a 10 + a 11 = 48) : a 6 + a 7 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3793_379337


namespace NUMINAMATH_CALUDE_odd_even_intersection_empty_l3793_379352

def odd_integers : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}
def even_integers : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}

theorem odd_even_intersection_empty :
  odd_integers ∩ even_integers = ∅ := by sorry

end NUMINAMATH_CALUDE_odd_even_intersection_empty_l3793_379352


namespace NUMINAMATH_CALUDE_function_domain_range_implies_interval_l3793_379396

open Real

theorem function_domain_range_implies_interval (f : ℝ → ℝ) (m n : ℝ) :
  (∀ x ∈ Set.Icc m n, f x ∈ Set.Icc (-1/2) (1/4)) →
  (∀ x, f x = sin x * sin (x + π/3) - 1/4) →
  m < n →
  π/3 ≤ n - m ∧ n - m ≤ 2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_function_domain_range_implies_interval_l3793_379396


namespace NUMINAMATH_CALUDE_parabola_shift_l3793_379325

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ := (x + 3)^2 + 2

-- Theorem stating that the shifted parabola is correct
theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x + 3) + 2 :=
by
  sorry


end NUMINAMATH_CALUDE_parabola_shift_l3793_379325


namespace NUMINAMATH_CALUDE_vector_c_satisfies_conditions_l3793_379305

/-- Given vectors a and b in ℝ², prove that vector c satisfies the required conditions -/
theorem vector_c_satisfies_conditions (a b c : ℝ × ℝ) : 
  a = (1, 2) → b = (2, -3) → c = (7/2, -7/4) → 
  (c.1 * a.1 + c.2 * a.2 = 0) ∧ 
  (∃ k : ℝ, b.1 = k * (a.1 - c.1) ∧ b.2 = k * (a.2 - c.2)) := by
sorry

end NUMINAMATH_CALUDE_vector_c_satisfies_conditions_l3793_379305


namespace NUMINAMATH_CALUDE_rectangular_prism_width_zero_l3793_379321

/-- A rectangular prism with given dimensions --/
structure RectangularPrism where
  l : ℝ  -- length
  h : ℝ  -- height
  d : ℝ  -- diagonal
  w : ℝ  -- width

/-- The theorem stating that a rectangular prism with length 6, height 8, and diagonal 10 has width 0 --/
theorem rectangular_prism_width_zero (p : RectangularPrism) 
  (hl : p.l = 6) 
  (hh : p.h = 8) 
  (hd : p.d = 10) : 
  p.w = 0 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_width_zero_l3793_379321


namespace NUMINAMATH_CALUDE_percentage_problem_l3793_379378

theorem percentage_problem : ∃ x : ℝ, (0.001 * x = 0.24) ∧ (x = 240) := by sorry

end NUMINAMATH_CALUDE_percentage_problem_l3793_379378


namespace NUMINAMATH_CALUDE_career_preference_theorem_l3793_379376

/-- Represents the ratio of boys to girls in a class -/
def boy_girl_ratio : ℚ := 2 / 3

/-- Represents the fraction of boys who prefer the career -/
def boy_preference : ℚ := 1 / 3

/-- Represents the fraction of girls who prefer the career -/
def girl_preference : ℚ := 2 / 3

/-- Calculates the degrees in a circle graph for a given career preference -/
def career_preference_degrees (ratio : ℚ) (boy_pref : ℚ) (girl_pref : ℚ) : ℚ :=
  360 * ((ratio * boy_pref + girl_pref) / (ratio + 1))

/-- Theorem stating that the career preference degrees is 192 -/
theorem career_preference_theorem :
  career_preference_degrees boy_girl_ratio boy_preference girl_preference = 192 := by
  sorry

#eval career_preference_degrees boy_girl_ratio boy_preference girl_preference

end NUMINAMATH_CALUDE_career_preference_theorem_l3793_379376


namespace NUMINAMATH_CALUDE_ideal_solution_range_l3793_379354

theorem ideal_solution_range (m n q : ℝ) : 
  m + 2*n = 6 →
  2*m + n = 3*q →
  m + n > 1 →
  q > -1 := by
sorry

end NUMINAMATH_CALUDE_ideal_solution_range_l3793_379354


namespace NUMINAMATH_CALUDE_cos_180_degrees_l3793_379362

/-- Cosine of 180 degrees is -1 -/
theorem cos_180_degrees : Real.cos (π) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_180_degrees_l3793_379362


namespace NUMINAMATH_CALUDE_pasta_preference_ratio_l3793_379318

theorem pasta_preference_ratio (total_students : ℕ) (spaghetti_pref : ℕ) (fettuccine_pref : ℕ)
  (h_total : total_students = 800)
  (h_spaghetti : spaghetti_pref = 300)
  (h_fettuccine : fettuccine_pref = 80) :
  (spaghetti_pref : ℚ) / fettuccine_pref = 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_pasta_preference_ratio_l3793_379318


namespace NUMINAMATH_CALUDE_omega_even_implies_periodic_l3793_379364

/-- Definition of an Ω function -/
def is_omega_function (f : ℝ → ℝ) : Prop :=
  ∃ T : ℝ, T ≠ 0 ∧ ∀ x : ℝ, f x = T * f (x + T)

/-- Definition of an even function -/
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- Definition of a periodic function -/
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

/-- Theorem: If f is an Ω function and even, then it is periodic -/
theorem omega_even_implies_periodic
  (f : ℝ → ℝ) (h_omega : is_omega_function f) (h_even : is_even_function f) :
  ∃ T : ℝ, T ≠ 0 ∧ is_periodic f (2 * T) :=
by sorry


end NUMINAMATH_CALUDE_omega_even_implies_periodic_l3793_379364


namespace NUMINAMATH_CALUDE_sixth_term_is_one_sixteenth_l3793_379355

/-- A geometric sequence with specific conditions -/
def GeometricSequence (a : ℕ → ℚ) : Prop :=
  (∃ q : ℚ, ∀ n : ℕ, a (n + 1) = a n * q) ∧
  a 1 + a 3 = 5/2 ∧
  a 2 + a 4 = 5/4

/-- The sixth term of the geometric sequence is 1/16 -/
theorem sixth_term_is_one_sixteenth (a : ℕ → ℚ) (h : GeometricSequence a) :
  a 6 = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_is_one_sixteenth_l3793_379355


namespace NUMINAMATH_CALUDE_quadratic_to_linear_inequality_l3793_379387

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) := x^2 + a*x + b

-- Define the linear function
def g (a b : ℝ) (x : ℝ) := a*x + b

-- State the theorem
theorem quadratic_to_linear_inequality 
  (a b : ℝ) 
  (h : ∀ x : ℝ, f a b x > 0 ↔ x < -3 ∨ x > 1) :
  ∀ x : ℝ, g a b x < 0 ↔ x < 3/2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_to_linear_inequality_l3793_379387


namespace NUMINAMATH_CALUDE_fifth_day_income_l3793_379349

def cab_driver_income (income_4_days : List ℝ) (average_income : ℝ) : ℝ :=
  5 * average_income - income_4_days.sum

theorem fifth_day_income 
  (income_4_days : List ℝ) 
  (average_income : ℝ) 
  (h1 : income_4_days.length = 4) 
  (h2 : average_income = (income_4_days.sum + cab_driver_income income_4_days average_income) / 5) :
  cab_driver_income income_4_days average_income = 
    5 * average_income - income_4_days.sum :=
by
  sorry

#eval cab_driver_income [300, 150, 750, 200] 400

end NUMINAMATH_CALUDE_fifth_day_income_l3793_379349


namespace NUMINAMATH_CALUDE_fish_food_calculation_l3793_379360

/-- The total amount of food Layla needs to give her fish -/
def total_fish_food (goldfish : ℕ) (goldfish_food : ℚ)
                    (swordtails : ℕ) (swordtails_food : ℚ)
                    (guppies : ℕ) (guppies_food : ℚ)
                    (angelfish : ℕ) (angelfish_food : ℚ)
                    (tetra : ℕ) (tetra_food : ℚ) : ℚ :=
  goldfish * goldfish_food +
  swordtails * swordtails_food +
  guppies * guppies_food +
  angelfish * angelfish_food +
  tetra * tetra_food

theorem fish_food_calculation :
  total_fish_food 4 1 5 2 10 (1/2) 3 (3/2) 6 1 = 59/2 := by
  sorry

end NUMINAMATH_CALUDE_fish_food_calculation_l3793_379360


namespace NUMINAMATH_CALUDE_three_digit_sum_magic_l3793_379312

/-- Given a three-digit number abc where a, b, and c are digits in base 10,
    if the sum of (acb), (bca), (bac), (cab), and (cba) is 3333,
    then abc = 555. -/
theorem three_digit_sum_magic (a b c : Nat) : 
  a < 10 → b < 10 → c < 10 →
  (100 * a + 10 * c + b) + 
  (100 * b + 10 * c + a) + 
  (100 * b + 10 * a + c) + 
  (100 * c + 10 * a + b) + 
  (100 * c + 10 * b + a) = 3333 →
  100 * a + 10 * b + c = 555 := by
  sorry


end NUMINAMATH_CALUDE_three_digit_sum_magic_l3793_379312


namespace NUMINAMATH_CALUDE_journey_matches_graph_l3793_379394

/-- Represents a segment of a journey --/
inductive JourneySegment
  | SlowAway
  | FastAway
  | Stationary
  | FastTowards
  | SlowTowards

/-- Represents a complete journey --/
def Journey := List JourneySegment

/-- Represents the shape of a graph segment --/
inductive GraphSegment
  | GradualIncline
  | SteepIncline
  | FlatLine
  | SteepDecline
  | GradualDecline

/-- Represents a complete graph --/
def Graph := List GraphSegment

/-- The journey we're analyzing --/
def janesJourney : Journey :=
  [JourneySegment.SlowAway, JourneySegment.FastAway, JourneySegment.Stationary,
   JourneySegment.FastTowards, JourneySegment.SlowTowards]

/-- The correct graph representation --/
def correctGraph : Graph :=
  [GraphSegment.GradualIncline, GraphSegment.SteepIncline, GraphSegment.FlatLine,
   GraphSegment.SteepDecline, GraphSegment.GradualDecline]

/-- Function to convert a journey to its graph representation --/
def journeyToGraph (j : Journey) : Graph :=
  sorry

/-- Theorem stating that the journey converts to the correct graph --/
theorem journey_matches_graph : journeyToGraph janesJourney = correctGraph := by
  sorry

end NUMINAMATH_CALUDE_journey_matches_graph_l3793_379394


namespace NUMINAMATH_CALUDE_largest_n_factorial_sum_perfect_square_l3793_379358

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem largest_n_factorial_sum_perfect_square :
  (∀ n : ℕ, n > 3 → ¬(is_perfect_square (sum_factorials n))) ∧
  (is_perfect_square (sum_factorials 3)) ∧
  (∀ n : ℕ, n > 0 → n < 3 → ¬(is_perfect_square (sum_factorials n))) :=
sorry

end NUMINAMATH_CALUDE_largest_n_factorial_sum_perfect_square_l3793_379358


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l3793_379389

/-- The function f(x) = (2x+1)^2 -/
def f (x : ℝ) : ℝ := (2*x + 1)^2

/-- The derivative of f at x = 0 is 4 -/
theorem derivative_f_at_zero : 
  deriv f 0 = 4 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l3793_379389


namespace NUMINAMATH_CALUDE_epsilon_max_ratio_l3793_379380

/-- Represents a contestant's performance in a math contest --/
structure ContestPerformance where
  day1_score : ℕ
  day1_attempted : ℕ
  day2_score : ℕ
  day2_attempted : ℕ
  day3_score : ℕ
  day3_attempted : ℕ

/-- Calculates the total score for a contestant --/
def totalScore (p : ContestPerformance) : ℕ := p.day1_score + p.day2_score + p.day3_score

/-- Calculates the total attempted points for a contestant --/
def totalAttempted (p : ContestPerformance) : ℕ := 
  p.day1_attempted + p.day2_attempted + p.day3_attempted

/-- Calculates the success ratio for a contestant --/
def successRatio (p : ContestPerformance) : ℚ := 
  (totalScore p : ℚ) / (totalAttempted p : ℚ)

/-- Delta's performance in the contest --/
def delta : ContestPerformance := {
  day1_score := 210,
  day1_attempted := 350,
  day2_score := 320, -- Assumed based on total score
  day2_attempted := 450, -- Assumed based on total attempted
  day3_score := 0, -- Placeholder
  day3_attempted := 0 -- Placeholder
}

theorem epsilon_max_ratio :
  ∀ epsilon : ContestPerformance,
  totalAttempted epsilon = 800 →
  totalAttempted delta = 800 →
  successRatio delta = 530 / 800 →
  epsilon.day1_attempted ≠ 350 →
  epsilon.day1_score > 0 →
  epsilon.day2_score > 0 →
  epsilon.day3_score > 0 →
  (epsilon.day1_score : ℚ) / (epsilon.day1_attempted : ℚ) < 210 / 350 →
  (epsilon.day2_score : ℚ) / (epsilon.day2_attempted : ℚ) < (delta.day2_score : ℚ) / (delta.day2_attempted : ℚ) →
  (epsilon.day3_score : ℚ) / (epsilon.day3_attempted : ℚ) < (delta.day3_score : ℚ) / (delta.day3_attempted : ℚ) →
  successRatio epsilon ≤ 789 / 800 :=
by sorry

end NUMINAMATH_CALUDE_epsilon_max_ratio_l3793_379380


namespace NUMINAMATH_CALUDE_inscribed_rectangle_sides_l3793_379317

theorem inscribed_rectangle_sides (a b c : ℝ) (x y : ℝ) : 
  a = 10 ∧ b = 17 ∧ c = 21 →  -- Triangle sides
  c > a ∧ c > b →  -- c is the longest side
  x + y = 12 →  -- Half of rectangle's perimeter
  y < 8 →  -- Rectangle's height is less than triangle's height
  (8 - y) / 8 = (c - x) / c →  -- Similarity of triangles
  x = 72 / 13 ∧ y = 84 / 13 := by
sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_sides_l3793_379317


namespace NUMINAMATH_CALUDE_max_abs_z_cubed_minus_3z_minus_2_l3793_379319

/-- The maximum absolute value of z³ - 3z - 2 for complex z on the unit circle -/
theorem max_abs_z_cubed_minus_3z_minus_2 (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (w : ℂ), Complex.abs w = 1 ∧ 
    ∀ (u : ℂ), Complex.abs u = 1 → 
      Complex.abs (w^3 - 3*w - 2) ≥ Complex.abs (u^3 - 3*u - 2) ∧
      Complex.abs (w^3 - 3*w - 2) = 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_max_abs_z_cubed_minus_3z_minus_2_l3793_379319


namespace NUMINAMATH_CALUDE_triangle_neg_three_four_l3793_379356

/-- The triangle operation -/
def triangle (a b : ℤ) : ℤ := a * b - a - b + 1

/-- Theorem stating that (-3) △ 4 = -12 -/
theorem triangle_neg_three_four : triangle (-3) 4 = -12 := by sorry

end NUMINAMATH_CALUDE_triangle_neg_three_four_l3793_379356


namespace NUMINAMATH_CALUDE_no_real_solutions_l3793_379314

theorem no_real_solutions :
  ∀ y : ℝ, (8 * y^2 + 155 * y + 3) / (4 * y + 45) ≠ 4 * y + 3 :=
by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3793_379314


namespace NUMINAMATH_CALUDE_smallest_number_l3793_379359

def digits : List Nat := [1, 4, 5]

def is_permutation (n : Nat) : Prop :=
  let digits_of_n := n.digits 10
  digits_of_n.length = digits.length ∧ digits_of_n.toFinset = digits.toFinset

theorem smallest_number :
  ∀ n : Nat, is_permutation n → 145 ≤ n := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l3793_379359


namespace NUMINAMATH_CALUDE_wage_difference_l3793_379385

/-- Represents the hourly wages at Joe's Steakhouse -/
structure JoesSteakhouseWages where
  manager : ℝ
  dishwasher : ℝ
  chef : ℝ
  manager_wage : manager = 8.50
  dishwasher_wage : dishwasher = manager / 2
  chef_wage : chef = dishwasher * 1.20

/-- The difference between a manager's hourly wage and a chef's hourly wage is $3.40 -/
theorem wage_difference (w : JoesSteakhouseWages) : w.manager - w.chef = 3.40 := by
  sorry

end NUMINAMATH_CALUDE_wage_difference_l3793_379385


namespace NUMINAMATH_CALUDE_decimal_to_fraction_simplest_l3793_379311

theorem decimal_to_fraction_simplest : 
  ∃ (a b : ℕ), 
    a > 0 ∧ b > 0 ∧ 
    (a : ℚ) / (b : ℚ) = 0.84375 ∧
    ∀ (c d : ℕ), c > 0 ∧ d > 0 ∧ (c : ℚ) / (d : ℚ) = 0.84375 → b ≤ d ∧
    a + b = 59 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_simplest_l3793_379311


namespace NUMINAMATH_CALUDE_evaluate_expression_l3793_379390

theorem evaluate_expression : 5 * 399 + 4 * 399 + 3 * 399 + 397 = 5185 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3793_379390


namespace NUMINAMATH_CALUDE_floor_length_proof_l3793_379335

/-- Given two rectangular floors X and Y with equal areas, 
    where X is 10 feet by 18 feet and Y is 9 feet wide, 
    prove that the length of floor Y is 20 feet. -/
theorem floor_length_proof (area_x area_y length_x width_x width_y : ℝ) : 
  area_x = area_y → 
  length_x = 10 → 
  width_x = 18 → 
  width_y = 9 → 
  area_x = length_x * width_x → 
  area_y = width_y * (area_y / width_y) → 
  area_y / width_y = 20 := by
  sorry

#check floor_length_proof

end NUMINAMATH_CALUDE_floor_length_proof_l3793_379335


namespace NUMINAMATH_CALUDE_product_of_functions_l3793_379307

theorem product_of_functions (x : ℝ) (hx : x ≠ 0) : 
  let f : ℝ → ℝ := λ x => 2 * x
  let g : ℝ → ℝ := λ x => -(3 * x - 1) / x
  (f x) * (g x) = -6 * x + 2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_functions_l3793_379307


namespace NUMINAMATH_CALUDE_football_player_goal_increase_l3793_379395

/-- The increase in average goals score after the fifth match -/
def goalScoreIncrease (totalGoals : ℕ) (fifthMatchGoals : ℕ) : ℚ :=
  let firstFourAverage := (totalGoals - fifthMatchGoals : ℚ) / 4
  let newAverage := (totalGoals : ℚ) / 5
  newAverage - firstFourAverage

/-- Theorem stating the increase in average goals score -/
theorem football_player_goal_increase :
  goalScoreIncrease 4 2 = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_football_player_goal_increase_l3793_379395


namespace NUMINAMATH_CALUDE_problem_solution_l3793_379384

def f (x : ℝ) := |2 * x - 1|

def g (x : ℝ) := f x + f (x - 1)

theorem problem_solution :
  (∀ x : ℝ, f x < 4 ↔ -3/2 < x ∧ x < 5/2) ∧
  (∃ a : ℝ, (∀ x : ℝ, g x ≥ a) ∧
    ∀ m n : ℝ, m > 0 → n > 0 → m + n = a →
      2/m + 1/n ≥ 3/2 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3793_379384


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3793_379353

/-- Two 2D vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, -3)
  let b : ℝ × ℝ := (-2, m)
  parallel a b → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3793_379353


namespace NUMINAMATH_CALUDE_negation_of_conditional_l3793_379341

theorem negation_of_conditional (x : ℝ) :
  ¬(x > 1 → x^2 > x) ↔ (x ≤ 1 → x^2 ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_conditional_l3793_379341


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3793_379324

def U : Set ℤ := {-1, 0, 1, 2}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {0, 2}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {0} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3793_379324


namespace NUMINAMATH_CALUDE_extended_euclidean_algorithm_l3793_379302

theorem extended_euclidean_algorithm (m₀ m₁ : ℤ) (h : 0 < m₁ ∧ m₁ ≤ m₀) :
  ∃ u v : ℤ, m₀ * u + m₁ * v = Int.gcd m₀ m₁ := by
  sorry

end NUMINAMATH_CALUDE_extended_euclidean_algorithm_l3793_379302


namespace NUMINAMATH_CALUDE_distance_O_to_B_l3793_379348

/-- Represents a person moving on the road -/
structure Person where
  name : String
  startPosition : ℝ
  speed : ℝ

/-- Represents the road with three locations -/
structure Road where
  a : ℝ
  o : ℝ
  b : ℝ

/-- The main theorem stating the distance between O and B -/
theorem distance_O_to_B 
  (road : Road)
  (jia yi : Person)
  (h1 : road.o = 0)  -- Set O as the origin
  (h2 : road.a = -1360)  -- A is 1360 meters to the left of O
  (h3 : jia.startPosition = road.a)
  (h4 : yi.startPosition = road.o)
  (h5 : jia.speed * 10 + road.a = yi.speed * 10)  -- Equidistant at 10 minutes
  (h6 : jia.speed * 40 + road.a = road.b)  -- Meet at B at 40 minutes
  (h7 : yi.speed * 40 = road.b)  -- Yi also reaches B at 40 minutes
  : road.b = 2040 := by
  sorry


end NUMINAMATH_CALUDE_distance_O_to_B_l3793_379348


namespace NUMINAMATH_CALUDE_intersection_nonempty_iff_l3793_379301

/-- Given sets A and B, prove the condition for their non-empty intersection -/
theorem intersection_nonempty_iff (a : ℝ) : 
  (∃ x : ℝ, x ∈ {x | 1 ≤ x ∧ x ≤ 2} ∩ {x | x ≤ a}) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_iff_l3793_379301


namespace NUMINAMATH_CALUDE_prob_gpa_at_least_3_75_l3793_379368

/-- Grade points for each letter grade -/
def gradePoints : Char → ℕ
| 'A' => 4
| 'B' => 3
| 'C' => 2
| 'D' => 1
| _ => 0

/-- Calculate GPA from total points -/
def calculateGPA (totalPoints : ℕ) : ℚ :=
  totalPoints / 4

/-- Probability of getting an A in English -/
def probAEnglish : ℚ := 1 / 3

/-- Probability of getting a B in English -/
def probBEnglish : ℚ := 1 / 2

/-- Probability of getting an A in History -/
def probAHistory : ℚ := 1 / 5

/-- Probability of getting a B in History -/
def probBHistory : ℚ := 1 / 2

theorem prob_gpa_at_least_3_75 :
  let mathPoints := gradePoints 'B'
  let sciencePoints := gradePoints 'B'
  let totalFixedPoints := mathPoints + sciencePoints
  let requiredPoints := 15
  let probBothA := probAEnglish * probAHistory
  probBothA = 1 / 15 ∧
  (∀ (englishGrade historyGrade : Char),
    calculateGPA (totalFixedPoints + gradePoints englishGrade + gradePoints historyGrade) ≥ 15 / 4 →
    (englishGrade = 'A' ∧ historyGrade = 'A')) →
  probBothA = 1 / 15 := by
sorry

end NUMINAMATH_CALUDE_prob_gpa_at_least_3_75_l3793_379368


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l3793_379391

/-- Given a line with equation y = -1/2 + 3x, prove that the sum of its x-intercept and y-intercept is -1/3. -/
theorem line_intercepts_sum (x y : ℝ) : 
  y = -1/2 + 3*x → -- Line equation
  ∃ (x_int y_int : ℝ),
    (0 = -1/2 + 3*x_int) ∧  -- x-intercept
    (y_int = -1/2 + 3*0) ∧  -- y-intercept
    (x_int + y_int = -1/3) := by
  sorry


end NUMINAMATH_CALUDE_line_intercepts_sum_l3793_379391


namespace NUMINAMATH_CALUDE_tower_height_calculation_l3793_379343

/-- Given a mountain and a tower, if the angles of depression from the top of the mountain
    to the top and bottom of the tower are as specified, then the height of the tower is 200m. -/
theorem tower_height_calculation (mountain_height : ℝ) (angle_to_top angle_to_bottom : ℝ) :
  mountain_height = 300 →
  angle_to_top = 30 * π / 180 →
  angle_to_bottom = 60 * π / 180 →
  ∃ (tower_height : ℝ), tower_height = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_tower_height_calculation_l3793_379343


namespace NUMINAMATH_CALUDE_hexagon_angle_measure_l3793_379331

theorem hexagon_angle_measure (A N G L E S : ℝ) : 
  -- ANGLES is a hexagon
  A + N + G + L + E + S = 720 →
  -- ∠A ≅ ∠G ≅ ∠E
  A = G ∧ G = E →
  -- ∠N is supplementary to ∠S
  N + S = 180 →
  -- ∠L is a right angle
  L = 90 →
  -- The measure of ∠E is 150°
  E = 150 := by sorry

end NUMINAMATH_CALUDE_hexagon_angle_measure_l3793_379331


namespace NUMINAMATH_CALUDE_max_G_ratio_is_six_fifths_l3793_379344

/-- Represents a four-digit number --/
structure FourDigitNumber where
  thousands : Nat
  hundreds : Nat
  tens : Nat
  units : Nat
  is_valid : thousands ≥ 1 ∧ thousands ≤ 9 ∧ 
             hundreds ≥ 0 ∧ hundreds ≤ 9 ∧ 
             tens ≥ 0 ∧ tens ≤ 9 ∧ 
             units ≥ 0 ∧ units ≤ 9

/-- Defines a "difference 2 multiple" --/
def isDifference2Multiple (n : FourDigitNumber) : Prop :=
  n.thousands - n.hundreds = 2 ∧ n.tens - n.units = 4

/-- Defines a "difference 3 multiple" --/
def isDifference3Multiple (n : FourDigitNumber) : Prop :=
  n.thousands - n.hundreds = 3 ∧ n.tens - n.units = 6

/-- Calculates the sum of digits --/
def G (n : FourDigitNumber) : Nat :=
  n.thousands + n.hundreds + n.tens + n.units

/-- Calculates F(p,q) --/
def F (p q : FourDigitNumber) : Int :=
  (1000 * p.thousands + 100 * p.hundreds + 10 * p.tens + p.units -
   (1000 * q.thousands + 100 * q.hundreds + 10 * q.tens + q.units)) / 10

/-- Main theorem --/
theorem max_G_ratio_is_six_fifths 
  (p q : FourDigitNumber) 
  (h1 : isDifference2Multiple p)
  (h2 : isDifference3Multiple q)
  (h3 : p.units = 3)
  (h4 : q.units = 3)
  (h5 : ∃ k : Int, F p q / (G p - G q + 3) = k) :
  ∀ (p' q' : FourDigitNumber), 
    isDifference2Multiple p' → 
    isDifference3Multiple q' → 
    p'.units = 3 → 
    q'.units = 3 → 
    (∃ k : Int, F p' q' / (G p' - G q' + 3) = k) → 
    (G p : ℚ) / (G q) ≥ (G p' : ℚ) / (G q') := by
  sorry

end NUMINAMATH_CALUDE_max_G_ratio_is_six_fifths_l3793_379344


namespace NUMINAMATH_CALUDE_tangent_line_implies_a_value_l3793_379366

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 2 * x + 1

-- State the theorem
theorem tangent_line_implies_a_value (a : ℝ) (h1 : a ≠ 0) :
  (∃ (m : ℝ), (∀ x : ℝ, x + f a x - 2 = m * (x - 1)) ∧ 
               (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 1| < δ → 
                 |(f a x - f a 1) - m * (x - 1)| < ε * |x - 1|)) →
  a = -1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_implies_a_value_l3793_379366


namespace NUMINAMATH_CALUDE_max_abs_z_l3793_379340

theorem max_abs_z (z : ℂ) (h : Complex.abs (z + 3 + 4 * I) ≤ 2) :
  ∃ (max_val : ℝ), max_val = 7 ∧ ∀ w : ℂ, Complex.abs (w + 3 + 4 * I) ≤ 2 → Complex.abs w ≤ max_val :=
by sorry

end NUMINAMATH_CALUDE_max_abs_z_l3793_379340


namespace NUMINAMATH_CALUDE_marcella_shoes_l3793_379320

theorem marcella_shoes (initial_pairs : ℕ) : 
  (initial_pairs * 2 - 9 ≥ 21 * 2) ∧ 
  (∀ n : ℕ, n > initial_pairs → n * 2 - 9 < 21 * 2) → 
  initial_pairs = 25 := by
sorry

end NUMINAMATH_CALUDE_marcella_shoes_l3793_379320


namespace NUMINAMATH_CALUDE_simplify_square_roots_l3793_379350

theorem simplify_square_roots : Real.sqrt (5 * 3) * Real.sqrt (3^3 * 5^3) = 225 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l3793_379350


namespace NUMINAMATH_CALUDE_julian_facebook_friends_l3793_379377

theorem julian_facebook_friends :
  ∀ (julian_friends : ℕ) (julian_boys julian_girls boyd_boys boyd_girls : ℝ),
    julian_boys = 0.6 * julian_friends →
    julian_girls = 0.4 * julian_friends →
    boyd_girls = 2 * julian_girls →
    boyd_boys + boyd_girls = 100 →
    boyd_boys = 0.36 * 100 →
    julian_friends = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_julian_facebook_friends_l3793_379377


namespace NUMINAMATH_CALUDE_curve_is_parabola_l3793_379322

/-- The curve defined by √X + √Y = 1 is a parabola -/
theorem curve_is_parabola :
  ∃ (a b c : ℝ) (h : a ≠ 0),
    ∀ (x y : ℝ),
      (Real.sqrt x + Real.sqrt y = 1) ↔ (y = a * x^2 + b * x + c) :=
sorry

end NUMINAMATH_CALUDE_curve_is_parabola_l3793_379322


namespace NUMINAMATH_CALUDE_sqrt_fraction_equality_l3793_379370

theorem sqrt_fraction_equality : 
  (Real.sqrt ((8:ℝ)^2 + 15^2)) / (Real.sqrt (36 + 64)) = 17 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equality_l3793_379370


namespace NUMINAMATH_CALUDE_triangle_centroid_length_l3793_379332

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  -- Right triangle condition
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 ∧
  -- BC = 6
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 36 ∧
  -- AC = 8
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 64

-- Define the centroid
def Centroid (O A B C : ℝ × ℝ) : Prop :=
  O.1 = (A.1 + B.1 + C.1) / 3 ∧ O.2 = (A.2 + B.2 + C.2) / 3

-- Define the midpoint
def Midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Main theorem
theorem triangle_centroid_length (A B C O P Q : ℝ × ℝ) :
  Triangle A B C →
  Centroid O A B C →
  Midpoint Q A B →
  Midpoint P B C →
  ((O.1 - P.1)^2 + (O.2 - P.2)^2) = (4/9) * 73 :=
by sorry

end NUMINAMATH_CALUDE_triangle_centroid_length_l3793_379332


namespace NUMINAMATH_CALUDE_log_1560_base_5_rounded_l3793_379372

theorem log_1560_base_5_rounded (ε : ℝ) (h : ε > 0) :
  ∃ (n : ℤ), n = 5 ∧ |Real.log 1560 / Real.log 5 - n| < 1/2 + ε :=
sorry

end NUMINAMATH_CALUDE_log_1560_base_5_rounded_l3793_379372


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3793_379393

theorem complex_equation_solution :
  let z : ℂ := -3 * I / 4
  2 - 3 * I * z = -4 + 5 * I * z :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3793_379393


namespace NUMINAMATH_CALUDE_equation_solution_l3793_379338

theorem equation_solution :
  ∃ (x : ℝ), x^2 - 4 ≠ 0 ∧ (x - 2) / (x + 2) + 4 / (x^2 - 4) = 1 ∧ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3793_379338


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l3793_379398

theorem triangle_third_side_length 
  (a b : ℝ) 
  (θ : Real) 
  (ha : a = 5) 
  (hb : b = 12) 
  (hθ : θ = 150 * π / 180) : 
  ∃ c : ℝ, c = Real.sqrt (169 + 60 * Real.sqrt 3) ∧ 
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos θ) :=
sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l3793_379398


namespace NUMINAMATH_CALUDE_red_and_large_toys_l3793_379300

/-- Represents the color of a toy -/
inductive Color
| Red
| Green
| Blue
| Yellow
| Orange

/-- Represents the size of a toy -/
inductive Size
| Small
| Medium
| Large
| ExtraLarge

/-- Represents the distribution of toys by color and size -/
structure ToyDistribution where
  red_small : Rat
  red_medium : Rat
  red_large : Rat
  red_extra_large : Rat
  green_small : Rat
  green_medium : Rat
  green_large : Rat
  green_extra_large : Rat
  blue_small : Rat
  blue_medium : Rat
  blue_large : Rat
  blue_extra_large : Rat
  yellow_small : Rat
  yellow_medium : Rat
  yellow_large : Rat
  yellow_extra_large : Rat
  orange_small : Rat
  orange_medium : Rat
  orange_large : Rat
  orange_extra_large : Rat

/-- The given distribution of toys -/
def given_distribution : ToyDistribution :=
  { red_small := 6/100, red_medium := 8/100, red_large := 7/100, red_extra_large := 4/100,
    green_small := 4/100, green_medium := 7/100, green_large := 5/100, green_extra_large := 4/100,
    blue_small := 6/100, blue_medium := 3/100, blue_large := 4/100, blue_extra_large := 2/100,
    yellow_small := 8/100, yellow_medium := 10/100, yellow_large := 5/100, yellow_extra_large := 2/100,
    orange_small := 9/100, orange_medium := 6/100, orange_large := 5/100, orange_extra_large := 5/100 }

/-- Theorem stating the number of red and large toys -/
theorem red_and_large_toys (total_toys : ℕ) (h : total_toys * given_distribution.green_large = 47) :
  total_toys * given_distribution.red_large = 329 := by
  sorry

end NUMINAMATH_CALUDE_red_and_large_toys_l3793_379300


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3793_379399

theorem quadratic_inequality (b c : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^2 + b*x + c) 
  (h2 : f (-1) = f 3) : 
  f 1 < c ∧ c < f (-1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3793_379399


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3793_379342

open Real

theorem trigonometric_identities (α : ℝ) (h : tan α = 2) :
  (sin α - 4 * cos α) / (5 * sin α + 2 * cos α) = -1/6 ∧
  4 * sin α ^ 2 - 3 * sin α * cos α - 5 * cos α ^ 2 = 1 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3793_379342


namespace NUMINAMATH_CALUDE_line_slope_and_intercept_l3793_379392

/-- Given a line with equation 4y = 6x - 12, prove that its slope is 3/2 and y-intercept is -3. -/
theorem line_slope_and_intercept :
  ∃ (m b : ℚ), m = 3/2 ∧ b = -3 ∧
  ∀ (x y : ℚ), 4*y = 6*x - 12 ↔ y = m*x + b :=
sorry

end NUMINAMATH_CALUDE_line_slope_and_intercept_l3793_379392


namespace NUMINAMATH_CALUDE_simplify_expression_l3793_379373

theorem simplify_expression : (((81 : ℚ) / 16) ^ (3 / 4) - (-1) ^ (0 : ℕ)) = 19 / 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3793_379373


namespace NUMINAMATH_CALUDE_seven_hash_three_l3793_379367

/-- Custom operator # defined for real numbers -/
def hash (a b : ℝ) : ℝ := 4*a + 2*b - 6

/-- Theorem stating that 7 # 3 = 28 -/
theorem seven_hash_three : hash 7 3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_seven_hash_three_l3793_379367


namespace NUMINAMATH_CALUDE_february_bill_increase_l3793_379347

def january_bill : ℝ := 179.99999999999991

theorem february_bill_increase (february_bill : ℝ) 
  (h1 : february_bill / january_bill = 3 / 2) 
  (h2 : ∃ (increased_bill : ℝ), increased_bill / january_bill = 5 / 3) : 
  ∃ (increased_bill : ℝ), increased_bill - february_bill = 30 :=
sorry

end NUMINAMATH_CALUDE_february_bill_increase_l3793_379347


namespace NUMINAMATH_CALUDE_horner_polynomial_eval_l3793_379327

def horner_eval (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

theorem horner_polynomial_eval :
  let coeffs := [7, 3, -5, 11]
  let x := 23
  horner_eval coeffs x = 86652 := by
sorry

end NUMINAMATH_CALUDE_horner_polynomial_eval_l3793_379327


namespace NUMINAMATH_CALUDE_sara_second_book_cost_l3793_379336

/-- The cost of Sara's second book -/
def second_book_cost (first_book_cost bill_given change_received : ℝ) : ℝ :=
  bill_given - change_received - first_book_cost

/-- Theorem stating the cost of Sara's second book -/
theorem sara_second_book_cost :
  second_book_cost 5.5 20 8 = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_sara_second_book_cost_l3793_379336


namespace NUMINAMATH_CALUDE_train_speed_l3793_379304

theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 450) (h2 : time = 8) :
  length / time = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3793_379304


namespace NUMINAMATH_CALUDE_initial_time_theorem_l3793_379315

/-- Given a distance of 720 km, if increasing the initial time by 3/2 results in a speed of 80 kmph,
    then the initial time taken to cover the distance was 6 hours. -/
theorem initial_time_theorem (t : ℝ) (h1 : t > 0) : 
  (720 : ℝ) / ((3/2) * t) = 80 → t = 6 := by
  sorry

end NUMINAMATH_CALUDE_initial_time_theorem_l3793_379315


namespace NUMINAMATH_CALUDE_prob_two_eggplants_germination_rate_expected_value_X_l3793_379345

-- Define the number of plots
def num_plots : ℕ := 4

-- Define the probability of planting eggplant in each plot
def prob_eggplant : ℚ := 1/3

-- Define the probability of planting cucumber in each plot
def prob_cucumber : ℚ := 2/3

-- Define the emergence rate of eggplant seeds
def emergence_rate_eggplant : ℚ := 95/100

-- Define the emergence rate of cucumber seeds
def emergence_rate_cucumber : ℚ := 98/100

-- Define the number of rows
def num_rows : ℕ := 2

-- Define the number of columns
def num_columns : ℕ := 2

-- Theorem for the probability of exactly 2 plots planting eggplants
theorem prob_two_eggplants : 
  (Nat.choose num_plots 2 : ℚ) * prob_eggplant^2 * prob_cucumber^2 = 8/27 := by sorry

-- Theorem for the germination rate of seeds for each plot
theorem germination_rate : 
  prob_eggplant * emergence_rate_eggplant + prob_cucumber * emergence_rate_cucumber = 97/100 := by sorry

-- Define the random variable X as the number of rows planting cucumbers
def X : Fin 3 → ℚ
| 0 => 1/25
| 1 => 16/25
| 2 => 8/25

-- Theorem for the expected value of X
theorem expected_value_X : 
  Finset.sum (Finset.range 3) (λ i => (i : ℚ) * X i) = 32/25 := by sorry

end NUMINAMATH_CALUDE_prob_two_eggplants_germination_rate_expected_value_X_l3793_379345


namespace NUMINAMATH_CALUDE_car_rental_cost_l3793_379326

/-- The daily rental cost of a car, given specific conditions. -/
theorem car_rental_cost (daily_rate : ℝ) (cost_per_mile : ℝ) (budget : ℝ) (miles : ℝ) : 
  cost_per_mile = 0.23 →
  budget = 76 →
  miles = 200 →
  daily_rate + cost_per_mile * miles = budget →
  daily_rate = 30 := by
sorry

end NUMINAMATH_CALUDE_car_rental_cost_l3793_379326


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l3793_379303

/-- A line passing through point (1,1) with equal intercepts on both axes -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  slope : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The line passes through point (1,1) -/
  passes_through_point : slope + y_intercept = 2
  /-- The line has equal intercepts on both axes -/
  equal_intercepts : y_intercept = slope * y_intercept ∨ y_intercept = 0

/-- The equation of an EqualInterceptLine is x + y = 2 or y = x -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.slope = 1 ∧ l.y_intercept = 1) ∨ (l.slope = 1 ∧ l.y_intercept = 0) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l3793_379303


namespace NUMINAMATH_CALUDE_consecutive_even_sum_l3793_379329

theorem consecutive_even_sum (a b c : ℤ) : 
  (∃ n : ℤ, a = 2*n ∧ b = 2*n + 2 ∧ c = 2*n + 4) →  -- a, b, c are consecutive even integers
  (a + b + c = 246) →                              -- their sum is 246
  (c = 84) :=                                      -- the third number is 84
by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_sum_l3793_379329


namespace NUMINAMATH_CALUDE_smallest_dance_class_size_l3793_379369

theorem smallest_dance_class_size :
  ∀ n : ℕ,
  n > 40 →
  (∀ m : ℕ, m > 40 ∧ 5 * m + 2 < 5 * n + 2 → m = n) →
  5 * n + 2 = 207 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_dance_class_size_l3793_379369


namespace NUMINAMATH_CALUDE_hash_composition_l3793_379375

-- Define the # operation
def hash (x : ℝ) : ℝ := 8 - x

-- Define the # operation
def hash_prefix (x : ℝ) : ℝ := x - 8

-- Theorem statement
theorem hash_composition : hash_prefix (hash 14) = -14 := by
  sorry

end NUMINAMATH_CALUDE_hash_composition_l3793_379375


namespace NUMINAMATH_CALUDE_urgent_painting_time_l3793_379383

/-- Represents the time required to paint an office -/
structure PaintingTime where
  painters : ℕ
  days : ℚ

/-- Represents the total work required to paint an office -/
def totalWork (pt : PaintingTime) : ℚ := pt.painters * pt.days

theorem urgent_painting_time 
  (first_office : PaintingTime)
  (second_office_normal : PaintingTime)
  (second_office_urgent : PaintingTime)
  (h1 : first_office.painters = 3)
  (h2 : first_office.days = 2)
  (h3 : second_office_normal.painters = 2)
  (h4 : totalWork first_office = totalWork second_office_normal)
  (h5 : second_office_urgent.painters = second_office_normal.painters)
  (h6 : second_office_urgent.days = 3/4 * second_office_normal.days) :
  second_office_urgent.days = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_urgent_painting_time_l3793_379383


namespace NUMINAMATH_CALUDE_tinks_are_falars_and_gymes_l3793_379365

-- Define the types for our entities
variable (U : Type) -- Universe type
variable (Falar Gyme Halp Tink Isoy : Set U)

-- State the given conditions
variable (h1 : Falar ⊆ Gyme)
variable (h2 : Halp ⊆ Tink)
variable (h3 : Isoy ⊆ Falar)
variable (h4 : Tink ⊆ Isoy)

-- State the theorem to be proved
theorem tinks_are_falars_and_gymes : Tink ⊆ Falar ∧ Tink ⊆ Gyme := by
  sorry

end NUMINAMATH_CALUDE_tinks_are_falars_and_gymes_l3793_379365


namespace NUMINAMATH_CALUDE_equal_representations_l3793_379339

/-- Represents the number of ways to write a positive integer as a product of powers of primes,
    where each factor is greater than or equal to the previous one. -/
def primeRepresentations (n : ℕ+) : ℕ := sorry

/-- Represents the number of ways to write a positive integer as a product of integers greater than 1,
    where each factor is divisible by all previous factors. -/
def divisibilityRepresentations (n : ℕ+) : ℕ := sorry

/-- Theorem stating that for any positive integer n, the number of prime representations
    is equal to the number of divisibility representations. -/
theorem equal_representations (n : ℕ+) : primeRepresentations n = divisibilityRepresentations n := by
  sorry

end NUMINAMATH_CALUDE_equal_representations_l3793_379339


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_l3793_379309

theorem sum_of_four_consecutive_integers (a b c d : ℤ) : 
  (a + 1 = b) ∧ (b + 1 = c) ∧ (c + 1 = d) ∧ (d = 27) → a + b + c + d = 102 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_l3793_379309


namespace NUMINAMATH_CALUDE_gcf_of_3150_and_7350_l3793_379323

theorem gcf_of_3150_and_7350 : Nat.gcd 3150 7350 = 525 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_3150_and_7350_l3793_379323


namespace NUMINAMATH_CALUDE_impossible_coloring_l3793_379310

theorem impossible_coloring (R G B : Set ℤ) : 
  (∀ (x y : ℤ), (x ∈ G ∧ y ∈ B) ∨ (x ∈ R ∧ y ∈ B) ∨ (x ∈ R ∧ y ∈ G) → x + y ∈ R) →
  (R ∪ G ∪ B = Set.univ) →
  (R ∩ G = ∅ ∧ R ∩ B = ∅ ∧ G ∩ B = ∅) →
  (R ≠ ∅ ∧ G ≠ ∅ ∧ B ≠ ∅) →
  False :=
by sorry

end NUMINAMATH_CALUDE_impossible_coloring_l3793_379310


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l3793_379382

theorem quadratic_expression_value (x y : ℝ) 
  (h1 : 3 * x + y = 7) 
  (h2 : x + 3 * y = 8) : 
  10 * x^2 + 13 * x * y + 10 * y^2 = 113 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l3793_379382


namespace NUMINAMATH_CALUDE_rectangle_longer_side_l3793_379328

theorem rectangle_longer_side (a : ℝ) (h1 : a > 0) : 
  (a * (0.8 * a) = 81/20) → a = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_l3793_379328
