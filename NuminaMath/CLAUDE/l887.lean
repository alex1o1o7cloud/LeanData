import Mathlib

namespace NUMINAMATH_CALUDE_base9_repeating_fraction_l887_88725

/-- Represents a digit in base-9 system -/
def Base9Digit := Fin 9

/-- Converts a base-10 number to its base-9 representation -/
def toBase9 (n : ℚ) : List Base9Digit :=
  sorry

/-- Checks if a list of digits is repeating -/
def isRepeating (l : List Base9Digit) : Prop :=
  sorry

/-- The main theorem -/
theorem base9_repeating_fraction :
  ∃ (n d : ℕ) (l : List Base9Digit),
    n ≠ 0 ∧ d ≠ 0 ∧
    (n : ℚ) / d < 1 / 2 ∧
    isRepeating (toBase9 ((n : ℚ) / d)) ∧
    n = 13 ∧ d = 37 :=
  sorry

end NUMINAMATH_CALUDE_base9_repeating_fraction_l887_88725


namespace NUMINAMATH_CALUDE_quotient_approx_l887_88710

theorem quotient_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.000001 ∧ |0.284973 / 29 - 0.009827| < ε :=
sorry

end NUMINAMATH_CALUDE_quotient_approx_l887_88710


namespace NUMINAMATH_CALUDE_continuous_fraction_value_l887_88737

theorem continuous_fraction_value : 
  ∃ (x : ℝ), x = 2 + 4 / (1 + 4/x) ∧ x = 4 := by
sorry

end NUMINAMATH_CALUDE_continuous_fraction_value_l887_88737


namespace NUMINAMATH_CALUDE_mixed_number_calculation_l887_88729

theorem mixed_number_calculation :
  let a := 5 + 1 / 2
  let b := 2 + 2 / 3
  let c := 1 + 1 / 5
  let d := 3 + 1 / 4
  (a - b) / (c + d) = 170 / 267 :=
by sorry

end NUMINAMATH_CALUDE_mixed_number_calculation_l887_88729


namespace NUMINAMATH_CALUDE_set_A_properties_l887_88763

def A : Set ℝ := {x | x^2 - 4 = 0}

theorem set_A_properties :
  (2 ∈ A) ∧
  (-2 ∈ A) ∧
  (A = {-2, 2}) ∧
  (∅ ⊆ A) := by
sorry

end NUMINAMATH_CALUDE_set_A_properties_l887_88763


namespace NUMINAMATH_CALUDE_grade_assignment_count_l887_88742

/-- The number of students in the class -/
def num_students : ℕ := 15

/-- The number of available grades -/
def num_grades : ℕ := 4

/-- Theorem stating that the number of ways to assign grades is 4^15 -/
theorem grade_assignment_count :
  (num_grades : ℕ) ^ num_students = 1073741824 := by sorry

end NUMINAMATH_CALUDE_grade_assignment_count_l887_88742


namespace NUMINAMATH_CALUDE_perimeter_triangle_PF₁F₂_shortest_distance_opposite_branches_l887_88716

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define a point P on the hyperbola
def P_on_C (P : ℝ × ℝ) : Prop := hyperbola_C P.1 P.2

-- Define the distance between two points
def distance (A B : ℝ × ℝ) : ℝ := sorry

-- Theorem 1: Perimeter of triangle PF₁F₂
theorem perimeter_triangle_PF₁F₂ (P : ℝ × ℝ) (h₁ : P_on_C P) (h₂ : distance P F₁ = 2 * distance P F₂) :
  distance P F₁ + distance P F₂ + distance F₁ F₂ = 28 := sorry

-- Theorem 2: Shortest distance between opposite branches
theorem shortest_distance_opposite_branches :
  ∃ (P Q : ℝ × ℝ), P_on_C P ∧ P_on_C Q ∧ 
    (∀ (R S : ℝ × ℝ), P_on_C R → P_on_C S → R.1 * S.1 < 0 → distance P Q ≤ distance R S) ∧
    distance P Q = 6 := sorry

end NUMINAMATH_CALUDE_perimeter_triangle_PF₁F₂_shortest_distance_opposite_branches_l887_88716


namespace NUMINAMATH_CALUDE_first_group_machines_correct_l887_88748

/-- The number of machines in the first group -/
def first_group_machines : ℕ := 5

/-- The production rate of the first group (units per machine-hour) -/
def first_group_rate : ℚ := 20 / (first_group_machines * 10)

/-- The production rate of the second group (units per machine-hour) -/
def second_group_rate : ℚ := 180 / (20 * 22.5)

/-- Theorem stating that the number of machines in the first group is correct -/
theorem first_group_machines_correct :
  first_group_rate = second_group_rate ∧
  first_group_machines * first_group_rate * 10 = 20 := by
  sorry

#check first_group_machines_correct

end NUMINAMATH_CALUDE_first_group_machines_correct_l887_88748


namespace NUMINAMATH_CALUDE_intersecting_planes_not_imply_intersecting_lines_l887_88708

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perpendicular : Line → Plane → Prop)

-- Define the intersection relation for lines and for planes
variable (lines_intersect : Line → Line → Prop)
variable (planes_intersect : Plane → Plane → Prop)

-- State the theorem
theorem intersecting_planes_not_imply_intersecting_lines 
  (a b : Line) (α β : Plane) 
  (h1 : a ≠ b) 
  (h2 : α ≠ β) 
  (h3 : perpendicular a α) 
  (h4 : perpendicular b β) :
  ∃ (α β : Plane), planes_intersect α β ∧ ¬ lines_intersect a b :=
sorry

end NUMINAMATH_CALUDE_intersecting_planes_not_imply_intersecting_lines_l887_88708


namespace NUMINAMATH_CALUDE_odd_monotonic_function_conditions_l887_88797

-- Define the function f
def f (a b c x : ℝ) : ℝ := x^3 - a*x^2 - b*x + c

-- State the theorem
theorem odd_monotonic_function_conditions (a b c : ℝ) :
  (∀ x, f a b c x = -f a b c (-x)) →  -- f is an odd function
  (∀ x y, x ≥ 1 → y ≥ 1 → x ≤ y → f a b c x ≤ f a b c y) →  -- f is monotonic on [1, +∞)
  (a = 0 ∧ c = 0 ∧ b ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_odd_monotonic_function_conditions_l887_88797


namespace NUMINAMATH_CALUDE_xiao_ming_age_l887_88771

theorem xiao_ming_age : ∃ (xiao_ming_age : ℕ), 
  (∃ (dad_age : ℕ), 
    dad_age - xiao_ming_age = 28 ∧ 
    dad_age = 3 * xiao_ming_age) → 
  xiao_ming_age = 14 := by
  sorry

end NUMINAMATH_CALUDE_xiao_ming_age_l887_88771


namespace NUMINAMATH_CALUDE_abs_neg_three_l887_88726

theorem abs_neg_three : |(-3 : ℤ)| = 3 := by sorry

end NUMINAMATH_CALUDE_abs_neg_three_l887_88726


namespace NUMINAMATH_CALUDE_eight_digit_numbers_divisibility_l887_88717

def first_number (A B C : ℕ) : ℕ := 84000000 + A * 100000 + 53000 + B * 100 + 10 + C
def second_number (A B C D : ℕ) : ℕ := 32700000 + A * 10000 + B * 1000 + 500 + C * 10 + D

theorem eight_digit_numbers_divisibility (A B C D : ℕ) 
  (h1 : A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10) 
  (h2 : first_number A B C % 4 = 0) 
  (h3 : second_number A B C D % 3 = 0) : 
  D = 2 := by
sorry

end NUMINAMATH_CALUDE_eight_digit_numbers_divisibility_l887_88717


namespace NUMINAMATH_CALUDE_triangle_area_equality_l887_88793

/-- Given a triangle MNH with points U on MN and C on NH, where:
  MU = s, UN = 6, NC = 20, CH = s, HM = 25,
  and the areas of triangle UNC and quadrilateral MUCH are equal,
  prove that s = 4. -/
theorem triangle_area_equality (s : ℝ) : 
  s > 0 ∧ 
  (1/2 : ℝ) * 6 * 20 = (1/2 : ℝ) * (s + 6) * (s + 20) - (1/2 : ℝ) * 6 * 20 → 
  s = 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_equality_l887_88793


namespace NUMINAMATH_CALUDE_greg_sharon_harvest_difference_l887_88722

theorem greg_sharon_harvest_difference :
  let greg_harvest : Real := 0.4
  let sharon_harvest : Real := 0.1
  greg_harvest - sharon_harvest = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_greg_sharon_harvest_difference_l887_88722


namespace NUMINAMATH_CALUDE_work_problem_solution_l887_88792

/-- Proves that given the conditions of the work problem, c worked for 4 days -/
theorem work_problem_solution :
  let a_days : ℕ := 16
  let b_days : ℕ := 9
  let c_wage : ℚ := 71.15384615384615
  let total_earning : ℚ := 1480
  let wage_ratio_a : ℚ := 3
  let wage_ratio_b : ℚ := 4
  let wage_ratio_c : ℚ := 5
  let a_wage : ℚ := (wage_ratio_a / wage_ratio_c) * c_wage
  let b_wage : ℚ := (wage_ratio_b / wage_ratio_c) * c_wage
  ∃ c_days : ℕ,
    c_days * c_wage + a_days * a_wage + b_days * b_wage = total_earning ∧
    c_days = 4 :=
by sorry

end NUMINAMATH_CALUDE_work_problem_solution_l887_88792


namespace NUMINAMATH_CALUDE_range_of_m_l887_88727

/-- The proposition p -/
def p (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) < 0

/-- The proposition q -/
def q (x : ℝ) : Prop := 1/2 < x ∧ x < 2/3

/-- q is a sufficient but not necessary condition for p -/
def q_sufficient_not_necessary (m : ℝ) : Prop :=
  (∀ x, q x → p x m) ∧ ¬(∀ x, p x m → q x)

theorem range_of_m :
  ∀ m : ℝ, q_sufficient_not_necessary m ↔ -1/3 ≤ m ∧ m ≤ 3/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l887_88727


namespace NUMINAMATH_CALUDE_joanna_initial_gumballs_l887_88712

/-- 
Given:
- Jacques had 60 gumballs initially
- They purchased 4 times their initial total
- After sharing equally, each got 250 gumballs
Prove that Joanna initially had 40 gumballs
-/
theorem joanna_initial_gumballs : 
  ∀ (j : ℕ), -- j represents Joanna's initial number of gumballs
  let jacques_initial := 60
  let total_initial := j + jacques_initial
  let purchased := 4 * total_initial
  let total_final := total_initial + purchased
  let each_after_sharing := 250
  total_final = 2 * each_after_sharing →
  j = 40 := by
sorry

end NUMINAMATH_CALUDE_joanna_initial_gumballs_l887_88712


namespace NUMINAMATH_CALUDE_product_of_primes_l887_88764

def largest_odd_one_digit_prime : ℕ := 7

def largest_two_digit_prime : ℕ := 97

def second_largest_two_digit_prime : ℕ := 89

theorem product_of_primes : 
  largest_odd_one_digit_prime * largest_two_digit_prime * second_largest_two_digit_prime = 60431 := by
  sorry

end NUMINAMATH_CALUDE_product_of_primes_l887_88764


namespace NUMINAMATH_CALUDE_diet_soda_ratio_l887_88775

theorem diet_soda_ratio (total bottles : ℕ) (regular_soda diet_soda fruit_juice sparkling_water : ℕ) :
  total = 60 →
  regular_soda = 18 →
  diet_soda = 14 →
  fruit_juice = 8 →
  sparkling_water = 10 →
  total = regular_soda + diet_soda + fruit_juice + sparkling_water + (total - regular_soda - diet_soda - fruit_juice - sparkling_water) →
  (diet_soda : ℚ) / total = 7 / 30 :=
by
  sorry

end NUMINAMATH_CALUDE_diet_soda_ratio_l887_88775


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l887_88704

theorem least_positive_integer_with_remainders : ∃! n : ℕ,
  n > 0 ∧
  n % 5 = 2 ∧
  n % 4 = 2 ∧
  n % 3 = 0 ∧
  ∀ m : ℕ, m > 0 ∧ m % 5 = 2 ∧ m % 4 = 2 ∧ m % 3 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l887_88704


namespace NUMINAMATH_CALUDE_minimum_candies_to_remove_l887_88747

/-- Represents the number of candies of each flavor in the bag -/
structure CandyBag where
  chocolate : Nat
  mint : Nat
  butterscotch : Nat

/-- The initial state of the candy bag -/
def initialBag : CandyBag := { chocolate := 4, mint := 6, butterscotch := 10 }

/-- The total number of candies in the bag -/
def totalCandies (bag : CandyBag) : Nat :=
  bag.chocolate + bag.mint + bag.butterscotch

/-- Predicate to check if at least two candies of each flavor have been eaten -/
def atLeastTwoEachFlavor (removed : Nat) (bag : CandyBag) : Prop :=
  removed ≥ bag.chocolate - 1 ∧ removed ≥ bag.mint - 1 ∧ removed ≥ bag.butterscotch - 1

theorem minimum_candies_to_remove (bag : CandyBag) :
  totalCandies bag = 20 →
  bag = initialBag →
  ∃ (n : Nat), n = 18 ∧ 
    (∀ (m : Nat), m < n → ¬(atLeastTwoEachFlavor m bag)) ∧
    (atLeastTwoEachFlavor n bag) := by
  sorry

end NUMINAMATH_CALUDE_minimum_candies_to_remove_l887_88747


namespace NUMINAMATH_CALUDE_marathon_checkpoints_l887_88753

/-- Represents a circular marathon with checkpoints -/
structure Marathon where
  total_distance : ℕ
  checkpoint_spacing : ℕ
  distance_to_first : ℕ
  distance_from_last : ℕ

/-- Calculates the number of checkpoints in a marathon -/
def num_checkpoints (m : Marathon) : ℕ :=
  (m.total_distance - m.distance_to_first - m.distance_from_last) / m.checkpoint_spacing + 1

/-- Theorem stating that a marathon with given specifications has 5 checkpoints -/
theorem marathon_checkpoints :
  ∃ (m : Marathon),
    m.total_distance = 26 ∧
    m.checkpoint_spacing = 6 ∧
    m.distance_to_first = 1 ∧
    m.distance_from_last = 1 ∧
    num_checkpoints m = 5 :=
by
  sorry


end NUMINAMATH_CALUDE_marathon_checkpoints_l887_88753


namespace NUMINAMATH_CALUDE_power_calculation_l887_88749

theorem power_calculation : (10 ^ 6 : ℕ) * (10 ^ 2 : ℕ) ^ 3 / (10 ^ 4 : ℕ) = 10 ^ 8 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l887_88749


namespace NUMINAMATH_CALUDE_juan_number_problem_l887_88780

theorem juan_number_problem (n : ℝ) : 
  (2 * ((n + 3)^2) - 2) / 3 = 14 ↔ (n = -3 + Real.sqrt 22 ∨ n = -3 - Real.sqrt 22) :=
by sorry

end NUMINAMATH_CALUDE_juan_number_problem_l887_88780


namespace NUMINAMATH_CALUDE_mystery_discount_rate_l887_88719

theorem mystery_discount_rate 
  (biography_price : ℝ) 
  (mystery_price : ℝ) 
  (biography_count : ℕ) 
  (mystery_count : ℕ) 
  (total_savings : ℝ) 
  (total_discount_rate : ℝ) 
  (h1 : biography_price = 20)
  (h2 : mystery_price = 12)
  (h3 : biography_count = 5)
  (h4 : mystery_count = 3)
  (h5 : total_savings = 19)
  (h6 : total_discount_rate = 0.43)
  : ∃ (biography_discount : ℝ) (mystery_discount : ℝ),
    biography_discount + mystery_discount = total_discount_rate ∧ 
    mystery_discount = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_mystery_discount_rate_l887_88719


namespace NUMINAMATH_CALUDE_expected_games_value_l887_88795

/-- The expected number of games in a best-of-seven basketball match -/
def expected_games : ℚ :=
  let p : ℚ := 1 / 2  -- Probability of winning each game
  let prob4 : ℚ := 2 * p^4  -- Probability of ending in 4 games
  let prob5 : ℚ := 2 * 4 * p^4 * (1 - p)  -- Probability of ending in 5 games
  let prob6 : ℚ := 2 * 5 * p^3 * (1 - p)^2  -- Probability of ending in 6 games
  let prob7 : ℚ := 20 * p^3 * (1 - p)^3  -- Probability of ending in 7 games
  4 * prob4 + 5 * prob5 + 6 * prob6 + 7 * prob7

/-- Theorem: The expected number of games in a best-of-seven basketball match
    with equal win probabilities is 93/16 -/
theorem expected_games_value : expected_games = 93 / 16 := by
  sorry

end NUMINAMATH_CALUDE_expected_games_value_l887_88795


namespace NUMINAMATH_CALUDE_expression_decrease_l887_88736

theorem expression_decrease (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let original := x^2 * y^3 * z
  let new_x := 0.8 * x
  let new_y := 0.75 * y
  let new_z := 0.9 * z
  let new_expression := new_x^2 * new_y^3 * new_z
  new_expression / original = 0.2414 :=
by sorry

end NUMINAMATH_CALUDE_expression_decrease_l887_88736


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_exists_l887_88759

theorem point_in_second_quadrant_exists : ∃ (x y : ℤ), 
  x < 0 ∧ 
  y > 0 ∧ 
  y ≤ x + 4 ∧ 
  x = -1 ∧ 
  y = 3 := by
sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_exists_l887_88759


namespace NUMINAMATH_CALUDE_intersection_distance_l887_88731

/-- The distance between intersection points of a line and circle -/
theorem intersection_distance (x y : ℝ) : 
  -- Line equation
  (y = Real.sqrt 3 * x + Real.sqrt 2 / 2) →
  -- Circle equation
  ((x - Real.sqrt 2 / 2)^2 + (y - Real.sqrt 2 / 2)^2 = 1) →
  -- Distance between intersection points
  ∃ (a b : ℝ × ℝ), 
    (a.1 - b.1)^2 + (a.2 - b.2)^2 = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_l887_88731


namespace NUMINAMATH_CALUDE_no_standard_operation_satisfies_equation_l887_88714

theorem no_standard_operation_satisfies_equation : ¬∃ (op : ℝ → ℝ → ℝ), 
  (op = (·+·) ∨ op = (·-·) ∨ op = (·*·) ∨ op = (·/·)) ∧ 
  (op 12 4) - 3 + (6 - 2) = 7 := by
sorry

end NUMINAMATH_CALUDE_no_standard_operation_satisfies_equation_l887_88714


namespace NUMINAMATH_CALUDE_product_of_reciprocals_equals_one_l887_88760

theorem product_of_reciprocals_equals_one :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by sorry

end NUMINAMATH_CALUDE_product_of_reciprocals_equals_one_l887_88760


namespace NUMINAMATH_CALUDE_n_mod_9_eq_6_l887_88785

def n : ℕ := 2 + 333 + 44444 + 555555 + 6666666 + 77777777 + 888888888 + 9999999999

theorem n_mod_9_eq_6 : n % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_n_mod_9_eq_6_l887_88785


namespace NUMINAMATH_CALUDE_derivative_log2_l887_88783

-- Define the base-2 logarithm function
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem derivative_log2 (x : ℝ) (h : x > 0) : 
  deriv log2 x = 1 / (x * Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_derivative_log2_l887_88783


namespace NUMINAMATH_CALUDE_eighteen_percent_of_700_is_126_l887_88791

theorem eighteen_percent_of_700_is_126 : (18 / 100) * 700 = 126 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_percent_of_700_is_126_l887_88791


namespace NUMINAMATH_CALUDE_rectangular_plot_poles_l887_88776

/-- The number of poles needed to enclose a rectangular plot -/
def poles_needed (length width pole_distance : ℕ) : ℕ :=
  ((2 * (length + width) + pole_distance - 1) / pole_distance : ℕ)

/-- Theorem: A 135m by 80m plot with poles 7m apart needs 62 poles -/
theorem rectangular_plot_poles :
  poles_needed 135 80 7 = 62 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_poles_l887_88776


namespace NUMINAMATH_CALUDE_min_value_fraction_l887_88762

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hab : a * b = 1) :
  (a^2 + b^2) / (a - b) ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l887_88762


namespace NUMINAMATH_CALUDE_wild_weatherman_proof_l887_88718

structure TextContent where
  content : String

structure WritingStyle where
  style : String

structure CareerAspiration where
  aspiration : String

structure WeatherForecastingTechnology where
  accuracy : String
  perfection : Bool

structure WeatherScienceStudy where
  name : String

def text_content : TextContent := ⟨"[Full text content]"⟩

theorem wild_weatherman_proof 
  (text : TextContent) 
  (writing_style : WritingStyle) 
  (sam_aspiration : CareerAspiration) 
  (weather_tech : WeatherForecastingTechnology) 
  (weather_study : WeatherScienceStudy) : 
  writing_style.style = "interview" ∧ 
  sam_aspiration.aspiration = "news reporter" ∧ 
  weather_tech.accuracy = "more exact" ∧ 
  ¬weather_tech.perfection ∧
  weather_study.name = "meteorology" := by
  sorry

#check wild_weatherman_proof text_content

end NUMINAMATH_CALUDE_wild_weatherman_proof_l887_88718


namespace NUMINAMATH_CALUDE_alternating_sum_equals_neg_151_l887_88755

/-- The sum of the alternating sequence 1-2+3-4+...+100-101 -/
def alternating_sum : ℤ := 1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9 - 10 + 11 - 12 + 13 - 14 + 15 - 16 + 17 - 18 + 19 - 20 + 21 - 22 + 23 - 24 + 25 - 26 + 27 - 28 + 29 - 30 + 31 - 32 + 33 - 34 + 35 - 36 + 37 - 38 + 39 - 40 + 41 - 42 + 43 - 44 + 45 - 46 + 47 - 48 + 49 - 50 + 51 - 52 + 53 - 54 + 55 - 56 + 57 - 58 + 59 - 60 + 61 - 62 + 63 - 64 + 65 - 66 + 67 - 68 + 69 - 70 + 71 - 72 + 73 - 74 + 75 - 76 + 77 - 78 + 79 - 80 + 81 - 82 + 83 - 84 + 85 - 86 + 87 - 88 + 89 - 90 + 91 - 92 + 93 - 94 + 95 - 96 + 97 - 98 + 99 - 100 + 101

theorem alternating_sum_equals_neg_151 : alternating_sum = -151 := by
  sorry

end NUMINAMATH_CALUDE_alternating_sum_equals_neg_151_l887_88755


namespace NUMINAMATH_CALUDE_consecutive_squares_equality_l887_88774

theorem consecutive_squares_equality : ∃ x : ℕ+, 
  (x : ℤ)^2 + (x + 1)^2 + (x + 2)^2 + (x + 3)^2 = (x + 4)^2 + (x + 5)^2 + (x + 6)^2 ∧ 
  (x : ℤ)^2 = 441 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_equality_l887_88774


namespace NUMINAMATH_CALUDE_kitchen_module_cost_is_20000_l887_88798

/-- Represents the cost of a modular home construction --/
structure ModularHomeCost where
  totalSize : Nat
  kitchenSize : Nat
  bathroomSize : Nat
  bathroomCost : Nat
  otherCost : Nat
  kitchenCount : Nat
  bathroomCount : Nat
  totalCost : Nat

/-- Calculates the cost of the kitchen module --/
def kitchenModuleCost (home : ModularHomeCost) : Nat :=
  let otherSize := home.totalSize - home.kitchenSize * home.kitchenCount - home.bathroomSize * home.bathroomCount
  let otherTotalCost := otherSize * home.otherCost
  let bathroomTotalCost := home.bathroomCost * home.bathroomCount
  home.totalCost - otherTotalCost - bathroomTotalCost

/-- Theorem: The kitchen module costs $20,000 --/
theorem kitchen_module_cost_is_20000 (home : ModularHomeCost) 
  (h1 : home.totalSize = 2000)
  (h2 : home.kitchenSize = 400)
  (h3 : home.bathroomSize = 150)
  (h4 : home.bathroomCost = 12000)
  (h5 : home.otherCost = 100)
  (h6 : home.kitchenCount = 1)
  (h7 : home.bathroomCount = 2)
  (h8 : home.totalCost = 174000) :
  kitchenModuleCost home = 20000 := by
  sorry

end NUMINAMATH_CALUDE_kitchen_module_cost_is_20000_l887_88798


namespace NUMINAMATH_CALUDE_price_change_theorem_l887_88777

theorem price_change_theorem (initial_price : ℝ) (x : ℝ) : 
  initial_price > 0 →
  let price1 := initial_price * (1 + 0.3)
  let price2 := price1 * (1 - 0.15)
  let price3 := price2 * (1 + 0.1)
  let price4 := price3 * (1 - x / 100)
  price4 = initial_price →
  x = 18 := by
sorry

end NUMINAMATH_CALUDE_price_change_theorem_l887_88777


namespace NUMINAMATH_CALUDE_singleEliminationTournament_l887_88735

/-- Calculates the number of games required in a single-elimination tournament. -/
def gamesRequired (numTeams : ℕ) : ℕ := numTeams - 1

/-- Theorem: In a single-elimination tournament with 23 teams, 22 games are required to determine the winner. -/
theorem singleEliminationTournament :
  gamesRequired 23 = 22 := by sorry

end NUMINAMATH_CALUDE_singleEliminationTournament_l887_88735


namespace NUMINAMATH_CALUDE_odd_integers_equality_l887_88706

theorem odd_integers_equality (a b c d k m : ℕ) : 
  Odd a → Odd b → Odd c → Odd d →
  0 < a → a < b → b < c → c < d →
  a * d = b * c →
  a + d = 2^k →
  b + c = 2^m →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_odd_integers_equality_l887_88706


namespace NUMINAMATH_CALUDE_board_number_is_91_l887_88796

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def does_not_contain_seven (n : ℕ) : Prop :=
  ¬ (∃ d, d ∈ n.digits 10 ∧ d = 7)

def is_prime (p : ℕ) : Prop := Nat.Prime p

theorem board_number_is_91 
  (n : ℕ) 
  (x : ℕ) 
  (h_consecutive : ∀ i < n, is_two_digit (x / 10^i % 100))
  (h_descending : ∀ i < n - 1, x / 10^i % 100 > x / 10^(i+1) % 100)
  (h_last_digit : does_not_contain_seven (x % 100))
  (h_prime_factors : ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ x = p * q ∧ q = p + 4) :
  x = 91 :=
sorry

end NUMINAMATH_CALUDE_board_number_is_91_l887_88796


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l887_88711

/-- Given a line l with equation Ax + By + C = 0, 
    a line perpendicular to l has the equation Bx - Ay + C' = 0, 
    where C' is some constant. -/
theorem perpendicular_line_equation 
  (A B C : ℝ) (x y : ℝ → ℝ) (l : ℝ → Prop) :
  (l = λ t => A * (x t) + B * (y t) + C = 0) →
  ∃ C', ∃ l_perp : ℝ → Prop,
    (l_perp = λ t => B * (x t) - A * (y t) + C' = 0) ∧
    (∀ t, l_perp t → (∀ s, l s → 
      (x t - x s) * (A * (x t - x s) + B * (y t - y s)) + 
      (y t - y s) * (B * (x t - x s) - A * (y t - y s)) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l887_88711


namespace NUMINAMATH_CALUDE_expression_evaluation_l887_88744

theorem expression_evaluation :
  let x : ℤ := -1
  (x - 1)^2 - x * (x + 3) + 2 * (x + 2) * (x - 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l887_88744


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l887_88756

theorem more_girls_than_boys (total_students : ℕ) 
  (h_total : total_students = 42) 
  (h_ratio : ∃ (x : ℕ), 3 * x + 4 * x = total_students) : 
  ∃ (boys girls : ℕ), 
    boys + girls = total_students ∧ 
    3 * girls = 4 * boys ∧ 
    girls = boys + 6 := by
  sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l887_88756


namespace NUMINAMATH_CALUDE_doubled_to_original_ratio_l887_88730

theorem doubled_to_original_ratio (x : ℝ) (h : 3 * (2 * x + 5) = 135) : 
  (2 * x) / x = 2 := by
  sorry

end NUMINAMATH_CALUDE_doubled_to_original_ratio_l887_88730


namespace NUMINAMATH_CALUDE_dress_discount_calculation_l887_88757

def shoe_discount_percent : ℚ := 40 / 100
def original_shoe_price : ℚ := 50
def number_of_shoes : ℕ := 2
def original_dress_price : ℚ := 100
def total_spent : ℚ := 140

theorem dress_discount_calculation :
  let discounted_shoe_price := original_shoe_price * (1 - shoe_discount_percent)
  let total_shoe_cost := discounted_shoe_price * number_of_shoes
  let dress_cost := total_spent - total_shoe_cost
  original_dress_price - dress_cost = 20 := by sorry

end NUMINAMATH_CALUDE_dress_discount_calculation_l887_88757


namespace NUMINAMATH_CALUDE_union_of_sets_l887_88701

theorem union_of_sets : 
  let S : Set ℕ := {3, 4, 5}
  let T : Set ℕ := {4, 7, 8}
  S ∪ T = {3, 4, 5, 7, 8} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l887_88701


namespace NUMINAMATH_CALUDE_gunny_bag_capacity_is_13_tons_l887_88790

/-- Represents the weight of a packet in pounds -/
def packet_weight : ℚ := 16 + 4 / 16

/-- Represents the number of packets -/
def num_packets : ℕ := 1840

/-- Represents the number of pounds in a ton -/
def pounds_per_ton : ℕ := 2300

/-- Represents the capacity of the gunny bag in tons -/
def gunny_bag_capacity : ℚ := (packet_weight * num_packets) / pounds_per_ton

theorem gunny_bag_capacity_is_13_tons : gunny_bag_capacity = 13 := by
  sorry

end NUMINAMATH_CALUDE_gunny_bag_capacity_is_13_tons_l887_88790


namespace NUMINAMATH_CALUDE_extended_square_counts_l887_88740

/-- Represents a square configuration with extended sides -/
structure ExtendedSquare where
  /-- Side length of the small square -/
  a : ℝ
  /-- Area of the shaded triangle -/
  S : ℝ
  /-- Condition that S is a quarter of the area of the small square -/
  h_S : S = a^2 / 4

/-- Count of triangles with area 2S in the extended square configuration -/
def count_triangles_2S (sq : ExtendedSquare) : ℕ := 20

/-- Count of squares with area 8S in the extended square configuration -/
def count_squares_8S (sq : ExtendedSquare) : ℕ := 1

/-- Main theorem stating the counts of specific triangles and squares -/
theorem extended_square_counts (sq : ExtendedSquare) :
  count_triangles_2S sq = 20 ∧ count_squares_8S sq = 1 := by
  sorry

end NUMINAMATH_CALUDE_extended_square_counts_l887_88740


namespace NUMINAMATH_CALUDE_opposite_of_negative_four_thirds_l887_88787

theorem opposite_of_negative_four_thirds :
  -(-(4/3 : ℚ)) = 4/3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_four_thirds_l887_88787


namespace NUMINAMATH_CALUDE_xy_value_l887_88741

theorem xy_value (x y : ℝ) (h : (x + y)^2 - (x - y)^2 = 20) : x * y = 5 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l887_88741


namespace NUMINAMATH_CALUDE_regression_slope_effect_l887_88713

/-- Represents a linear regression model --/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Predicted y value for a given x --/
def LinearRegression.predict (model : LinearRegression) (x : ℝ) : ℝ :=
  model.intercept + model.slope * x

theorem regression_slope_effect (model : LinearRegression) 
  (h : model.slope = -1 ∧ model.intercept = 2) :
  ∀ x : ℝ, model.predict (x + 1) = model.predict x - 1 := by
  sorry

#check regression_slope_effect

end NUMINAMATH_CALUDE_regression_slope_effect_l887_88713


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_l887_88751

theorem sum_of_fifth_powers (a b c : ℝ) 
  (sum_eq : a + b + c = 2)
  (sum_squares_eq : a^2 + b^2 + c^2 = 5)
  (sum_cubes_eq : a^3 + b^3 + c^3 = 8) :
  a^5 + b^5 + c^5 = 98/6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fifth_powers_l887_88751


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l887_88715

theorem quadratic_inequality_solution (a : ℝ) (m : ℝ) :
  (∀ x : ℝ, ax^2 - 6*x + a^2 < 0 ↔ 1 < x ∧ x < m) →
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l887_88715


namespace NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l887_88709

theorem upstream_downstream_time_ratio 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : boat_speed = 63) 
  (h2 : stream_speed = 21) : 
  (boat_speed - stream_speed) / (boat_speed + stream_speed) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l887_88709


namespace NUMINAMATH_CALUDE_total_fabric_needed_l887_88769

/-- The number of shirts Jenson makes per day -/
def jenson_shirts_per_day : ℕ := 3

/-- The number of pants Kingsley makes per day -/
def kingsley_pants_per_day : ℕ := 5

/-- The number of yards of fabric used for one shirt -/
def fabric_per_shirt : ℕ := 2

/-- The number of yards of fabric used for one pair of pants -/
def fabric_per_pants : ℕ := 5

/-- The number of days to calculate fabric for -/
def days : ℕ := 3

/-- Theorem stating the total yards of fabric needed every 3 days -/
theorem total_fabric_needed : 
  jenson_shirts_per_day * fabric_per_shirt * days + 
  kingsley_pants_per_day * fabric_per_pants * days = 93 := by
  sorry

end NUMINAMATH_CALUDE_total_fabric_needed_l887_88769


namespace NUMINAMATH_CALUDE_det_A_squared_minus_3A_l887_88720

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 4; 3, 2]

theorem det_A_squared_minus_3A : Matrix.det ((A ^ 2) - 3 • A) = 140 := by
  sorry

end NUMINAMATH_CALUDE_det_A_squared_minus_3A_l887_88720


namespace NUMINAMATH_CALUDE_min_sum_squares_l887_88738

theorem min_sum_squares (a b c t : ℝ) (h : a + b + c = t) :
  a^2 + b^2 + c^2 ≥ t^2 / 3 := by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l887_88738


namespace NUMINAMATH_CALUDE_convoy_problem_l887_88733

/-- Represents the convoy of vehicles -/
structure Convoy where
  num_vehicles : ℕ
  departure_interval : ℚ
  first_departure : ℚ
  stop_time : ℚ
  speed : ℚ

/-- Calculate the travel time of the last vehicle in the convoy -/
def last_vehicle_travel_time (c : Convoy) : ℚ :=
  c.stop_time - (c.first_departure + (c.num_vehicles - 1) * c.departure_interval)

/-- Calculate the total distance traveled by the convoy -/
def total_distance_traveled (c : Convoy) : ℚ :=
  let total_time := c.num_vehicles * (c.stop_time - c.first_departure) - 
    (c.num_vehicles * (c.num_vehicles - 1) / 2) * c.departure_interval
  total_time * c.speed

/-- The main theorem statement -/
theorem convoy_problem (c : Convoy) 
  (h1 : c.num_vehicles = 15)
  (h2 : c.departure_interval = 1/6)
  (h3 : c.first_departure = 2)
  (h4 : c.stop_time = 6)
  (h5 : c.speed = 60) : 
  last_vehicle_travel_time c = 5/3 ∧ 
  total_distance_traveled c = 2550 := by
  sorry

#eval last_vehicle_travel_time ⟨15, 1/6, 2, 6, 60⟩
#eval total_distance_traveled ⟨15, 1/6, 2, 6, 60⟩

end NUMINAMATH_CALUDE_convoy_problem_l887_88733


namespace NUMINAMATH_CALUDE_fraction_addition_simplification_l887_88784

theorem fraction_addition_simplification :
  3 / 462 + 13 / 42 = 73 / 231 := by sorry

end NUMINAMATH_CALUDE_fraction_addition_simplification_l887_88784


namespace NUMINAMATH_CALUDE_a_perpendicular_b_l887_88761

/-- Two vectors in ℝ² are perpendicular if their dot product is zero -/
def isPerpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- Vector a in ℝ² -/
def a : ℝ × ℝ := (1, -2)

/-- Vector b in ℝ² -/
def b : ℝ × ℝ := (2, 1)

/-- Theorem stating that vectors a and b are perpendicular -/
theorem a_perpendicular_b : isPerpendicular a b := by
  sorry

end NUMINAMATH_CALUDE_a_perpendicular_b_l887_88761


namespace NUMINAMATH_CALUDE_lauren_subscription_rate_l887_88766

/-- Represents Lauren's earnings from her social media channel -/
structure Earnings where
  commercialRate : ℚ  -- Rate per commercial view
  commercialViews : ℕ -- Number of commercial views
  subscriptions : ℕ   -- Number of subscriptions
  totalRevenue : ℚ    -- Total revenue
  subscriptionRate : ℚ -- Rate per subscription

/-- Theorem stating that Lauren's subscription rate is $1 -/
theorem lauren_subscription_rate 
  (e : Earnings) 
  (h1 : e.commercialRate = 1/2)      -- $0.50 per commercial view
  (h2 : e.commercialViews = 100)     -- 100 commercial views
  (h3 : e.subscriptions = 27)        -- 27 subscriptions
  (h4 : e.totalRevenue = 77)         -- Total revenue is $77
  : e.subscriptionRate = 1 := by
  sorry


end NUMINAMATH_CALUDE_lauren_subscription_rate_l887_88766


namespace NUMINAMATH_CALUDE_conference_hall_seating_l887_88745

theorem conference_hall_seating
  (chairs_per_row : ℕ)
  (initial_chairs : ℕ)
  (expected_participants : ℕ)
  (h1 : chairs_per_row = 15)
  (h2 : initial_chairs = 195)
  (h3 : expected_participants = 120)
  : ∃ (removed_chairs : ℕ),
    removed_chairs = 75 ∧
    (initial_chairs - removed_chairs) % chairs_per_row = 0 ∧
    initial_chairs - removed_chairs ≥ expected_participants ∧
    initial_chairs - removed_chairs < expected_participants + chairs_per_row :=
by
  sorry

end NUMINAMATH_CALUDE_conference_hall_seating_l887_88745


namespace NUMINAMATH_CALUDE_candy_pencils_l887_88758

/-- Proves that Candy has 9 pencils given the conditions in the problem -/
theorem candy_pencils :
  ∀ (calen_original caleb candy : ℕ),
  calen_original = caleb + 5 →
  caleb = 2 * candy - 3 →
  calen_original - 10 = 10 →
  candy = 9 := by
sorry

end NUMINAMATH_CALUDE_candy_pencils_l887_88758


namespace NUMINAMATH_CALUDE_element_uniquely_identified_l887_88721

/-- Represents a 6x6 grid of distinct elements -/
def Grid := Fin 6 → Fin 6 → Fin 36

/-- The column of an element in the original grid -/
def OriginalColumn := Fin 6

/-- The column of an element in the new grid -/
def NewColumn := Fin 6

/-- Given a grid, an original column, and a new column, 
    returns the unique position of the element in both grids -/
def findElement (g : Grid) (oc : OriginalColumn) (nc : NewColumn) : 
  (Fin 6 × Fin 6) × (Fin 6 × Fin 6) :=
sorry

theorem element_uniquely_identified (g : Grid) (oc : OriginalColumn) (nc : NewColumn) :
  ∃! (p₁ p₂ : Fin 6 × Fin 6), 
    (findElement g oc nc).1 = p₁ ∧ 
    (findElement g oc nc).2 = p₂ ∧
    g p₁.1 p₁.2 = g p₂.2 p₂.1 :=
sorry

end NUMINAMATH_CALUDE_element_uniquely_identified_l887_88721


namespace NUMINAMATH_CALUDE_friends_total_points_l887_88794

def total_points (darius_points marius_points matt_points : ℕ) : ℕ :=
  darius_points + marius_points + matt_points

theorem friends_total_points :
  ∀ (darius_points marius_points matt_points : ℕ),
    darius_points = 10 →
    marius_points = darius_points + 3 →
    matt_points = darius_points + 5 →
    total_points darius_points marius_points matt_points = 38 :=
by
  sorry

end NUMINAMATH_CALUDE_friends_total_points_l887_88794


namespace NUMINAMATH_CALUDE_eight_in_C_l887_88743

def C : Set ℕ := {x | 1 ≤ x ∧ x ≤ 10}

theorem eight_in_C : 8 ∈ C := by
  sorry

end NUMINAMATH_CALUDE_eight_in_C_l887_88743


namespace NUMINAMATH_CALUDE_no_snow_probability_l887_88782

theorem no_snow_probability (p1 p2 p3 p4 : ℚ) 
  (h1 : p1 = 1/2) (h2 : p2 = 2/3) (h3 : p3 = 3/4) (h4 : p4 = 4/5) : 
  (1 - p1) * (1 - p2) * (1 - p3) * (1 - p4) = 1/120 := by
  sorry

end NUMINAMATH_CALUDE_no_snow_probability_l887_88782


namespace NUMINAMATH_CALUDE_complex_number_problem_l887_88789

-- Define the complex number z
variable (z : ℂ)

-- Define the conditions
variable (h1 : ∃ (r : ℝ), z + 2*I = r)
variable (h2 : ∃ (t : ℝ), z - 4 = t*I)

-- Define m as a real number
variable (m : ℝ)

-- Define the fourth quadrant condition
def in_fourth_quadrant (w : ℂ) : Prop :=
  w.re > 0 ∧ w.im < 0

-- Theorem statement
theorem complex_number_problem :
  z = 4 - 2*I ∧
  (in_fourth_quadrant ((z + m*I)^2) ↔ -2 < m ∧ m < 2) :=
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l887_88789


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_eq_five_l887_88779

/-- The function f(x) = x³ + ax² + 3x - 9 has an extreme value at x = -3 -/
def has_extreme_value_at_neg_three (a : ℝ) : Prop :=
  let f := fun (x : ℝ) => x^3 + a*x^2 + 3*x - 9
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ -3 ∧ |x - (-3)| < ε → f x ≤ f (-3) ∨ f x ≥ f (-3)

/-- If f(x) = x³ + ax² + 3x - 9 has an extreme value at x = -3, then a = 5 -/
theorem extreme_value_implies_a_eq_five :
  ∀ (a : ℝ), has_extreme_value_at_neg_three a → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_eq_five_l887_88779


namespace NUMINAMATH_CALUDE_tower_remainder_l887_88786

/-- Represents the number of different towers that can be built with cubes up to size n -/
def T : ℕ → ℕ
| 0 => 1
| 1 => 1
| 2 => 2
| (n+1) => if n ≤ 9 then T n * (min n 4) else T n

/-- The main theorem stating the remainder when T(10) is divided by 1000 -/
theorem tower_remainder : T 10 % 1000 = 216 := by sorry

end NUMINAMATH_CALUDE_tower_remainder_l887_88786


namespace NUMINAMATH_CALUDE_quadratic_function_property_l887_88728

/-- Given a quadratic function f(x) = x^2 - 2ax + b where a > 1,
    if both the domain and range of f are [1, a], then b = 5. -/
theorem quadratic_function_property (a b : ℝ) (f : ℝ → ℝ) :
  a > 1 →
  (∀ x, f x = x^2 - 2*a*x + b) →
  (∀ x, x ∈ Set.Icc 1 a ↔ f x ∈ Set.Icc 1 a) →
  b = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l887_88728


namespace NUMINAMATH_CALUDE_midpoint_on_grid_l887_88734

theorem midpoint_on_grid (points : Fin 5 → ℤ × ℤ) :
  ∃ i j, i ≠ j ∧ i < 5 ∧ j < 5 ∧
  (((points i).1 + (points j).1) % 2 = 0) ∧
  (((points i).2 + (points j).2) % 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_midpoint_on_grid_l887_88734


namespace NUMINAMATH_CALUDE_second_grade_selection_l887_88754

/-- Represents a stratified sampling scenario in a school -/
structure SchoolSampling where
  first_grade : ℕ
  second_grade : ℕ
  total_selected : ℕ
  first_grade_selected : ℕ

/-- Calculates the number of students selected from the second grade -/
def second_grade_selected (s : SchoolSampling) : ℕ :=
  s.total_selected - s.first_grade_selected

/-- Theorem stating that in the given scenario, 18 students are selected from the second grade -/
theorem second_grade_selection (s : SchoolSampling) 
  (h1 : s.first_grade = 400)
  (h2 : s.second_grade = 360)
  (h3 : s.total_selected = 56)
  (h4 : s.first_grade_selected = 20) :
  second_grade_selected s = 18 := by
  sorry

end NUMINAMATH_CALUDE_second_grade_selection_l887_88754


namespace NUMINAMATH_CALUDE_airport_distance_l887_88799

theorem airport_distance (initial_speed initial_time final_speed : ℝ)
  (late_time early_time : ℝ) :
  initial_speed = 40 →
  initial_time = 1 →
  final_speed = 60 →
  late_time = 1.5 →
  early_time = 1 →
  ∃ (total_time total_distance : ℝ),
    total_distance = initial_speed * initial_time +
      final_speed * (total_time - initial_time - early_time) ∧
    total_time = (total_distance / initial_speed) - late_time ∧
    total_distance = 420 :=
by sorry

end NUMINAMATH_CALUDE_airport_distance_l887_88799


namespace NUMINAMATH_CALUDE_cricket_average_proof_l887_88700

def cricket_average (total_matches : ℕ) (first_set : ℕ) (second_set : ℕ) 
  (first_avg : ℚ) (second_avg : ℚ) : ℚ :=
  let total_first := first_avg * first_set
  let total_second := second_avg * second_set
  (total_first + total_second) / total_matches

theorem cricket_average_proof :
  cricket_average 10 6 4 41 (35.75) = 38.9 := by
  sorry

#eval cricket_average 10 6 4 41 (35.75)

end NUMINAMATH_CALUDE_cricket_average_proof_l887_88700


namespace NUMINAMATH_CALUDE_congruence_solution_l887_88788

theorem congruence_solution : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 15 ∧ n ≡ 13258 [MOD 16] := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l887_88788


namespace NUMINAMATH_CALUDE_f_g_properties_l887_88773

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Define g as f'
def g : ℝ → ℝ := f'

-- State the conditions
axiom f_even : ∀ x, f (3/2 - 2*x) = f (3/2 + 2*x)
axiom g_even : ∀ x, g (2 + x) = g (2 - x)

-- State the theorem
theorem f_g_properties : g (-1/2) = 0 ∧ f (-1) = f 4 := by sorry

end NUMINAMATH_CALUDE_f_g_properties_l887_88773


namespace NUMINAMATH_CALUDE_cloth_gain_theorem_l887_88750

/-- Represents the gain percentage as a rational number -/
def gainPercentage : ℚ := 200 / 3

/-- Represents the number of meters of cloth sold -/
def metersSold : ℕ := 25

/-- Calculates the number of meters of cloth's selling price gained -/
def metersGained (gainPercentage : ℚ) (metersSold : ℕ) : ℚ :=
  (gainPercentage / 100) * metersSold / (1 + gainPercentage / 100)

/-- Theorem stating that the number of meters of cloth's selling price gained is 10 -/
theorem cloth_gain_theorem :
  metersGained gainPercentage metersSold = 10 := by
  sorry

end NUMINAMATH_CALUDE_cloth_gain_theorem_l887_88750


namespace NUMINAMATH_CALUDE_ten_bags_of_bags_l887_88707

/-- The number of ways to create a "bag of bags" structure with n identical bags. -/
def bagsOfBags : ℕ → ℕ
| 0 => 1
| 1 => 1
| n + 2 => sorry  -- The actual recursive definition would go here

/-- The number of ways to create a "bag of bags" structure with 10 identical bags is 719. -/
theorem ten_bags_of_bags : bagsOfBags 10 = 719 := by sorry

end NUMINAMATH_CALUDE_ten_bags_of_bags_l887_88707


namespace NUMINAMATH_CALUDE_sphere_in_parabolic_glass_l887_88770

/-- The distance from the highest point of a sphere to the bottom of a parabolic wine glass --/
theorem sphere_in_parabolic_glass (x y : ℝ) (b : ℝ) : 
  (∀ y, 0 ≤ y → y < 15 → x^2 = 2*y) →  -- Parabola equation
  (x^2 + (y - b)^2 = 9) →               -- Sphere equation
  ((2 - 2*b)^2 - 4*(b^2 - 9) = 0) →     -- Tangency condition
  (b + 3 = 8) :=                        -- Distance from highest point to bottom
by sorry

end NUMINAMATH_CALUDE_sphere_in_parabolic_glass_l887_88770


namespace NUMINAMATH_CALUDE_exponent_value_l887_88768

theorem exponent_value : ∃ exponent : ℝ,
  (1/5 : ℝ)^35 * (1/4 : ℝ)^exponent = 1 / (2 * (10 : ℝ)^35) ∧ exponent = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_value_l887_88768


namespace NUMINAMATH_CALUDE_dave_performance_weeks_l887_88739

/-- Given that Dave breaks 2 guitar strings per night, performs 6 shows per week,
    and needs to replace 144 guitar strings in total, prove that he performs for 12 weeks. -/
theorem dave_performance_weeks 
  (strings_per_night : ℕ)
  (shows_per_week : ℕ)
  (total_strings : ℕ)
  (h1 : strings_per_night = 2)
  (h2 : shows_per_week = 6)
  (h3 : total_strings = 144) :
  total_strings / (strings_per_night * shows_per_week) = 12 := by
sorry

end NUMINAMATH_CALUDE_dave_performance_weeks_l887_88739


namespace NUMINAMATH_CALUDE_elegant_interval_p_values_l887_88702

theorem elegant_interval_p_values (a b : ℕ) (m : ℝ) (p : ℕ) :
  (a < m ∧ m < b) →  -- m is in the "elegant interval" (a, b)
  (b = a + 1) →  -- a and b are consecutive positive integers
  (3 < Real.sqrt a + b ∧ Real.sqrt a + b ≤ 13) →  -- satisfies the given inequality
  (∃ x y : ℕ, x = b ∧ y * y = a ∧ b * x + a * y = p) →  -- x = b, y = √a, and bx + ay = p
  (p = 33 ∨ p = 127) :=
by sorry

end NUMINAMATH_CALUDE_elegant_interval_p_values_l887_88702


namespace NUMINAMATH_CALUDE_infinitely_many_square_repetitions_l887_88765

/-- The number of digits in a natural number -/
def num_digits (a : ℕ) : ℕ := sorry

/-- The repetition of a natural number -/
def repetition (a : ℕ) : ℕ := a * (10^(num_digits a)) + a

/-- There exist infinitely many natural numbers whose repetition is a perfect square -/
theorem infinitely_many_square_repetitions :
  ∀ n : ℕ, ∃ a > n, ∃ k : ℕ, repetition a = k^2 := by sorry

end NUMINAMATH_CALUDE_infinitely_many_square_repetitions_l887_88765


namespace NUMINAMATH_CALUDE_annie_total_travel_l887_88746

def blocks_to_bus_stop : ℕ := 5
def blocks_on_bus : ℕ := 7

def one_way_trip : ℕ := blocks_to_bus_stop + blocks_on_bus

theorem annie_total_travel : one_way_trip * 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_annie_total_travel_l887_88746


namespace NUMINAMATH_CALUDE_fraction_multiplication_l887_88724

theorem fraction_multiplication : (1 : ℚ) / 3 * 4 / 7 * 9 / 13 * 2 / 5 = 72 / 1365 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l887_88724


namespace NUMINAMATH_CALUDE_sin_difference_range_l887_88778

theorem sin_difference_range (a : ℝ) : 
  (∃ x : ℝ, Real.sin (x + π/4) - Real.sin (2*x) = a) → 
  -2 ≤ a ∧ a ≤ 9/8 := by
sorry

end NUMINAMATH_CALUDE_sin_difference_range_l887_88778


namespace NUMINAMATH_CALUDE_roses_per_friend_l887_88703

/-- The number of roses in a dozen -/
def dozen : ℕ := 12

/-- Prove that each dancer friend gave Bella 2 roses -/
theorem roses_per_friend (
  parents_roses : ℕ) 
  (dancer_friends : ℕ) 
  (total_roses : ℕ) 
  (h1 : parents_roses = 2 * dozen)
  (h2 : dancer_friends = 10)
  (h3 : total_roses = 44) :
  (total_roses - parents_roses) / dancer_friends = 2 := by
  sorry

#check roses_per_friend

end NUMINAMATH_CALUDE_roses_per_friend_l887_88703


namespace NUMINAMATH_CALUDE_imaginary_part_implies_a_value_l887_88772

theorem imaginary_part_implies_a_value (a : ℝ) :
  (Complex.im ((1 - a * Complex.I) / (1 + Complex.I)) = -1) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_implies_a_value_l887_88772


namespace NUMINAMATH_CALUDE_greatest_integer_a_l887_88767

theorem greatest_integer_a : ∃ (a : ℤ), 
  (∀ (x : ℤ), (x - a) * (x - 7) + 3 ≠ 0) ∧ 
  (∃ (x : ℤ), (x - 11) * (x - 7) + 3 = 0) ∧
  (∀ (b : ℤ), b > 11 → ∀ (x : ℤ), (x - b) * (x - 7) + 3 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_a_l887_88767


namespace NUMINAMATH_CALUDE_linear_function_increasing_l887_88781

/-- Given a linear function y = 2x + 1 and two points on this function,
    if the x-coordinate of the first point is less than the x-coordinate of the second point,
    then the y-coordinate of the first point is less than the y-coordinate of the second point. -/
theorem linear_function_increasing (x₁ x₂ y₁ y₂ : ℝ) :
  y₁ = 2 * x₁ + 1 →
  y₂ = 2 * x₂ + 1 →
  x₁ < x₂ →
  y₁ < y₂ :=
by sorry

end NUMINAMATH_CALUDE_linear_function_increasing_l887_88781


namespace NUMINAMATH_CALUDE_roots_have_different_signs_l887_88723

/-- Given two quadratic polynomials with specific properties, prove that the roots of the first polynomial have different signs -/
theorem roots_have_different_signs (a b c : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + 2*b*x₁ + c = 0 ∧ a * x₂^2 + 2*b*x₂ + c = 0) →  -- First polynomial has two distinct roots
  (∀ x : ℝ, a^2 * x^2 + 2*b^2*x + c^2 ≠ 0) →                                        -- Second polynomial has no roots
  ∃ x₁ x₂ : ℝ, x₁ * x₂ < 0 ∧ a * x₁^2 + 2*b*x₁ + c = 0 ∧ a * x₂^2 + 2*b*x₂ + c = 0  -- Roots of first polynomial have different signs
:= by sorry

end NUMINAMATH_CALUDE_roots_have_different_signs_l887_88723


namespace NUMINAMATH_CALUDE_revenue_change_l887_88705

theorem revenue_change 
  (T : ℝ) -- Original tax rate
  (C : ℝ) -- Original consumption
  (h1 : T > 0) -- Assumption: tax rate is positive
  (h2 : C > 0) -- Assumption: consumption is positive
  : 
  let T_new := T * (1 - 0.15) -- New tax rate after 15% decrease
  let C_new := C * (1 + 0.10) -- New consumption after 10% increase
  let R := T * C -- Original revenue
  let R_new := T_new * C_new -- New revenue
  (R_new / R) = 0.935 -- Ratio of new revenue to original revenue
  :=
by sorry

end NUMINAMATH_CALUDE_revenue_change_l887_88705


namespace NUMINAMATH_CALUDE_cuboid_surface_area_4_8_6_l887_88752

/-- The surface area of a cuboid with given dimensions -/
def cuboid_surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem stating that the surface area of a cuboid with dimensions 4x8x6 is 208 -/
theorem cuboid_surface_area_4_8_6 :
  cuboid_surface_area 4 8 6 = 208 := by
  sorry

#eval cuboid_surface_area 4 8 6

end NUMINAMATH_CALUDE_cuboid_surface_area_4_8_6_l887_88752


namespace NUMINAMATH_CALUDE_tetrahedron_volume_lower_bound_l887_88732

/-- Theorem: The volume of a tetrahedron is at least one-third the product of its opposite edge distances. -/
theorem tetrahedron_volume_lower_bound (d₁ d₂ d₃ V : ℝ) (h₁ : d₁ > 0) (h₂ : d₂ > 0) (h₃ : d₃ > 0) (hV : V > 0) :
  V ≥ (1/3) * d₁ * d₂ * d₃ := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_lower_bound_l887_88732
