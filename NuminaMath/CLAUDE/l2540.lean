import Mathlib

namespace NUMINAMATH_CALUDE_hundreds_digit_of_binomial_12_6_times_6_factorial_l2540_254004

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the function to get the hundreds digit
def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10

-- Theorem statement
theorem hundreds_digit_of_binomial_12_6_times_6_factorial :
  hundreds_digit (binomial 12 6 * Nat.factorial 6) = 8 := by
  sorry

end NUMINAMATH_CALUDE_hundreds_digit_of_binomial_12_6_times_6_factorial_l2540_254004


namespace NUMINAMATH_CALUDE_total_problems_solved_l2540_254042

def initial_problems : ℕ := 12
def additional_problems : ℕ := 7

theorem total_problems_solved :
  initial_problems + additional_problems = 19 := by
  sorry

end NUMINAMATH_CALUDE_total_problems_solved_l2540_254042


namespace NUMINAMATH_CALUDE_a_2017_equals_16_l2540_254039

def sequence_with_property_P (a : ℕ → ℕ) : Prop :=
  ∀ p q : ℕ, a p = a q → a (p + 1) = a (q + 1)

theorem a_2017_equals_16 (a : ℕ → ℕ) 
  (h_prop : sequence_with_property_P a)
  (h1 : a 1 = 1)
  (h2 : a 2 = 2)
  (h3 : a 3 = 3)
  (h5 : a 5 = 2)
  (h678 : a 6 + a 7 + a 8 = 21) :
  a 2017 = 16 := by
  sorry

end NUMINAMATH_CALUDE_a_2017_equals_16_l2540_254039


namespace NUMINAMATH_CALUDE_fraction_equality_l2540_254057

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 15)
  (h2 : p / n = 3)
  (h3 : p / q = 1 / 10) :
  m / q = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2540_254057


namespace NUMINAMATH_CALUDE_equation_is_union_of_twisted_cubics_twisted_cubic_is_parabola_like_equation_represents_two_parabolas_l2540_254096

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the equation y^6 - 9x^6 = 3y^3 - 1 -/
def equation (p : Point3D) : Prop :=
  p.y^6 - 9*p.x^6 = 3*p.y^3 - 1

/-- Represents a twisted cubic curve -/
def twistedCubic (a b c : ℝ) (p : Point3D) : Prop :=
  p.y^3 = a*p.x^3 + b*p.x + c

/-- The equation represents the union of two twisted cubic curves -/
theorem equation_is_union_of_twisted_cubics :
  ∀ p : Point3D, equation p ↔ (twistedCubic 3 0 1 p ∨ twistedCubic (-3) 0 1 p) :=
sorry

/-- Twisted cubic curves behave like parabolas -/
theorem twisted_cubic_is_parabola_like (a b c : ℝ) :
  ∀ p : Point3D, twistedCubic a b c p → (∃ q : Point3D, twistedCubic a b c q ∧ q ≠ p) :=
sorry

/-- The equation represents two parabola-like curves -/
theorem equation_represents_two_parabolas :
  ∃ (curve1 curve2 : Point3D → Prop),
    (∀ p : Point3D, equation p ↔ (curve1 p ∨ curve2 p)) ∧
    (∀ p : Point3D, curve1 p → (∃ q : Point3D, curve1 q ∧ q ≠ p)) ∧
    (∀ p : Point3D, curve2 p → (∃ q : Point3D, curve2 q ∧ q ≠ p)) :=
sorry

end NUMINAMATH_CALUDE_equation_is_union_of_twisted_cubics_twisted_cubic_is_parabola_like_equation_represents_two_parabolas_l2540_254096


namespace NUMINAMATH_CALUDE_gcd_930_868_l2540_254006

theorem gcd_930_868 : Nat.gcd 930 868 = 62 := by
  sorry

end NUMINAMATH_CALUDE_gcd_930_868_l2540_254006


namespace NUMINAMATH_CALUDE_max_value_theorem_l2540_254047

theorem max_value_theorem (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (h_sum : x^2 + y^2 + z^2 = 1) :
  3 * x * z * Real.sqrt 3 + 9 * y * z ≤ Real.sqrt ((29 * 54) / 5) ∧
  ∃ (x₀ y₀ z₀ : ℝ), 0 ≤ x₀ ∧ 0 ≤ y₀ ∧ 0 ≤ z₀ ∧ x₀^2 + y₀^2 + z₀^2 = 1 ∧
    3 * x₀ * z₀ * Real.sqrt 3 + 9 * y₀ * z₀ = Real.sqrt ((29 * 54) / 5) :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2540_254047


namespace NUMINAMATH_CALUDE_P_symmetric_l2540_254071

-- Define the polynomial sequence P_m
def P : ℕ → ℝ → ℝ → ℝ → ℝ
  | 0, x, y, z => 1
  | m + 1, x, y, z => (x + z) * (y + z) * P m x y (z + 1) - z^2 * P m x y z

-- State the theorem
theorem P_symmetric (m : ℕ) (x y z : ℝ) : 
  P m x y z = P m x z y ∧ P m x y z = P m y x z := by
  sorry

end NUMINAMATH_CALUDE_P_symmetric_l2540_254071


namespace NUMINAMATH_CALUDE_line_passes_through_point_l2540_254097

/-- A line in the form y = k(x-1) + 2 always passes through the point (1, 2) -/
theorem line_passes_through_point (k : ℝ) : 
  let f : ℝ → ℝ := λ x => k * (x - 1) + 2
  f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l2540_254097


namespace NUMINAMATH_CALUDE_missing_roots_theorem_l2540_254068

def p (x : ℝ) : ℝ := 12 * x^5 - 8 * x^4 - 45 * x^3 + 45 * x^2 + 8 * x - 12

theorem missing_roots_theorem (h1 : p 1 = 0) (h2 : p 1.5 = 0) (h3 : p (-2) = 0) :
  p (2/3) = 0 ∧ p (-1/2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_missing_roots_theorem_l2540_254068


namespace NUMINAMATH_CALUDE_evaluate_expression_l2540_254045

theorem evaluate_expression (a : ℝ) (h : a = 2) : (7 * a^2 - 15 * a + 5) * (3 * a - 4) = 6 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2540_254045


namespace NUMINAMATH_CALUDE_unique_consecutive_sum_18_l2540_254013

/-- A function that returns the sum of n consecutive integers starting from a -/
def consecutive_sum (a n : ℕ) : ℕ := n * (2 * a + n - 1) / 2

/-- A predicate that checks if a set of consecutive integers sums to 18 -/
def is_valid_set (a n : ℕ) : Prop :=
  n ≥ 2 ∧ consecutive_sum a n = 18

theorem unique_consecutive_sum_18 :
  ∃! p : ℕ × ℕ, is_valid_set p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_unique_consecutive_sum_18_l2540_254013


namespace NUMINAMATH_CALUDE_cube_function_property_l2540_254050

theorem cube_function_property (a : ℝ) : 
  (fun x : ℝ ↦ x^3 + 1) a = 11 → (fun x : ℝ ↦ x^3 + 1) (-a) = -9 := by
sorry

end NUMINAMATH_CALUDE_cube_function_property_l2540_254050


namespace NUMINAMATH_CALUDE_street_paths_l2540_254084

theorem street_paths (P Q : ℕ) (h1 : P = 130) (h2 : Q = 65) : P - 2*Q + 2014 = 2014 := by
  sorry

end NUMINAMATH_CALUDE_street_paths_l2540_254084


namespace NUMINAMATH_CALUDE_distance_traveled_l2540_254026

/-- The speed of sound in meters per second -/
def speed_of_sound : ℝ := 330

/-- The time between the first blast and when the man hears the second blast, in minutes -/
def time_between_blasts : ℝ := 30.25

/-- The time between the first and second blasts, in minutes -/
def actual_time_between_blasts : ℝ := 30

/-- Theorem: The distance the man traveled when he heard the second blast is 4950 meters -/
theorem distance_traveled : ℝ := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l2540_254026


namespace NUMINAMATH_CALUDE_supermarket_discount_items_l2540_254094

/-- Represents the supermarket's inventory and pricing --/
structure Supermarket where
  total_cost : ℝ
  items_a : ℕ
  items_b : ℕ
  cost_a : ℝ
  cost_b : ℝ
  price_a : ℝ
  price_b : ℝ

/-- Represents the second purchase scenario --/
structure SecondPurchase where
  sm : Supermarket
  items_b_new : ℕ
  discount_price_b : ℝ
  total_profit : ℝ

/-- The main theorem to prove --/
theorem supermarket_discount_items (sm : Supermarket) (sp : SecondPurchase) :
  sm.total_cost = 6000 ∧
  sm.items_a = 2 * sm.items_b - 30 ∧
  sm.cost_a = 22 ∧
  sm.cost_b = 30 ∧
  sm.price_a = 29 ∧
  sm.price_b = 40 ∧
  sp.sm = sm ∧
  sp.items_b_new = 3 * sm.items_b ∧
  sp.discount_price_b = sm.price_b / 2 ∧
  sp.total_profit = 2350 →
  ∃ (discount_items : ℕ), 
    discount_items = 70 ∧
    (sm.price_a - sm.cost_a) * sm.items_a + 
    (sm.price_b - sm.cost_b) * (sp.items_b_new - discount_items) +
    (sp.discount_price_b - sm.cost_b) * discount_items = sp.total_profit :=
by sorry


end NUMINAMATH_CALUDE_supermarket_discount_items_l2540_254094


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l2540_254027

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_with_current - 2 * current_speed

/-- Theorem stating that given the specific speeds in the problem, 
    the man's speed against the current is 12 km/hr. -/
theorem mans_speed_against_current :
  speed_against_current 22 5 = 12 := by
  sorry

#eval speed_against_current 22 5

end NUMINAMATH_CALUDE_mans_speed_against_current_l2540_254027


namespace NUMINAMATH_CALUDE_choose_officers_count_l2540_254095

/-- Represents the club with its member composition -/
structure Club where
  total_members : Nat
  boys : Nat
  girls : Nat
  senior_boys : Nat
  senior_girls : Nat

/-- Calculates the number of ways to choose a president and vice-president -/
def choose_officers (club : Club) : Nat :=
  (club.senior_boys * (club.boys - 1)) + (club.senior_girls * (club.girls - 1))

/-- The specific club instance from the problem -/
def our_club : Club :=
  { total_members := 30
  , boys := 16
  , girls := 14
  , senior_boys := 3
  , senior_girls := 3 
  }

/-- Theorem stating that the number of ways to choose officers for our club is 84 -/
theorem choose_officers_count : choose_officers our_club = 84 := by
  sorry

#eval choose_officers our_club

end NUMINAMATH_CALUDE_choose_officers_count_l2540_254095


namespace NUMINAMATH_CALUDE_add_zero_eq_self_l2540_254032

theorem add_zero_eq_self (x : ℝ) : x + 0 = x := by
  sorry

end NUMINAMATH_CALUDE_add_zero_eq_self_l2540_254032


namespace NUMINAMATH_CALUDE_stamp_collection_value_l2540_254099

theorem stamp_collection_value 
  (total_stamps : ℕ) 
  (sample_stamps : ℕ) 
  (sample_value : ℝ) 
  (h1 : total_stamps = 20)
  (h2 : sample_stamps = 4)
  (h3 : sample_value = 16) :
  (total_stamps : ℝ) * (sample_value / sample_stamps) = 80 :=
by sorry

end NUMINAMATH_CALUDE_stamp_collection_value_l2540_254099


namespace NUMINAMATH_CALUDE_bushes_needed_for_zucchinis_l2540_254059

/-- Represents the number of containers of blueberries per bush -/
def blueberries_per_bush : ℕ := 12

/-- Represents the number of containers of blueberries that can be traded for pumpkins -/
def blueberries_for_pumpkins : ℕ := 4

/-- Represents the number of pumpkins received when trading blueberries -/
def pumpkins_from_blueberries : ℕ := 3

/-- Represents the number of pumpkins that can be traded for zucchinis -/
def pumpkins_for_zucchinis : ℕ := 6

/-- Represents the number of zucchinis received when trading pumpkins -/
def zucchinis_from_pumpkins : ℕ := 5

/-- Represents the target number of zucchinis -/
def target_zucchinis : ℕ := 60

theorem bushes_needed_for_zucchinis :
  ∃ (bushes : ℕ), 
    bushes * blueberries_per_bush * pumpkins_from_blueberries * zucchinis_from_pumpkins = 
    target_zucchinis * blueberries_for_pumpkins * pumpkins_for_zucchinis ∧ 
    bushes = 8 := by
  sorry

end NUMINAMATH_CALUDE_bushes_needed_for_zucchinis_l2540_254059


namespace NUMINAMATH_CALUDE_sum_of_digits_for_four_elevenths_l2540_254029

theorem sum_of_digits_for_four_elevenths : ∃ (x y : ℕ), 
  (x < 10 ∧ y < 10) ∧ 
  (4 : ℚ) / 11 = (x * 10 + y : ℚ) / 99 ∧
  x + y = 9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_for_four_elevenths_l2540_254029


namespace NUMINAMATH_CALUDE_tank_capacity_correct_l2540_254012

/-- The capacity of a tank in gallons -/
def tank_capacity : ℝ := 32

/-- The total amount of oil in gallons -/
def total_oil : ℝ := 728

/-- The number of tanks needed -/
def num_tanks : ℕ := 23

/-- Theorem stating that the tank capacity is approximately correct -/
theorem tank_capacity_correct : 
  ∃ ε > 0, ε < 1 ∧ |tank_capacity - total_oil / num_tanks| < ε :=
sorry

end NUMINAMATH_CALUDE_tank_capacity_correct_l2540_254012


namespace NUMINAMATH_CALUDE_expression_simplification_l2540_254043

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  ((2 * x + 1) / x - 1) / ((x^2 - 1) / x) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2540_254043


namespace NUMINAMATH_CALUDE_max_poles_with_distinct_distances_l2540_254070

/-- 
Given a natural number k, this theorem states that the maximum number of poles
that can be painted in k colors, such that all distances between pairs of 
same-colored poles (with no other same-colored pole between them) are different,
is 3k - 1.
-/
theorem max_poles_with_distinct_distances (k : ℕ) : ℕ := by
  sorry

end NUMINAMATH_CALUDE_max_poles_with_distinct_distances_l2540_254070


namespace NUMINAMATH_CALUDE_eight_divided_by_point_three_repeating_l2540_254037

theorem eight_divided_by_point_three_repeating (x : ℚ) : x = 1/3 → 8 / x = 24 := by
  sorry

end NUMINAMATH_CALUDE_eight_divided_by_point_three_repeating_l2540_254037


namespace NUMINAMATH_CALUDE_math_representative_selection_l2540_254066

theorem math_representative_selection (male_students female_students : ℕ) 
  (h1 : male_students = 26) 
  (h2 : female_students = 24) : 
  (male_students + female_students : ℕ) = 50 := by
  sorry

end NUMINAMATH_CALUDE_math_representative_selection_l2540_254066


namespace NUMINAMATH_CALUDE_potato_division_l2540_254030

theorem potato_division (total_potatoes : ℕ) (num_people : ℕ) (potatoes_per_person : ℕ) :
  total_potatoes = 24 →
  num_people = 3 →
  total_potatoes = num_people * potatoes_per_person →
  potatoes_per_person = 8 := by
  sorry

end NUMINAMATH_CALUDE_potato_division_l2540_254030


namespace NUMINAMATH_CALUDE_least_multiple_of_25_greater_than_475_l2540_254058

theorem least_multiple_of_25_greater_than_475 :
  ∀ n : ℕ, n > 0 ∧ 25 ∣ n ∧ n > 475 → n ≥ 500 :=
by
  sorry

end NUMINAMATH_CALUDE_least_multiple_of_25_greater_than_475_l2540_254058


namespace NUMINAMATH_CALUDE_university_application_options_l2540_254022

theorem university_application_options : 
  let total_universities : ℕ := 6
  let applications_needed : ℕ := 3
  let universities_with_coinciding_exams : ℕ := 2
  
  (Nat.choose (total_universities - universities_with_coinciding_exams) applications_needed) +
  (universities_with_coinciding_exams * Nat.choose (total_universities - universities_with_coinciding_exams) (applications_needed - 1)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_university_application_options_l2540_254022


namespace NUMINAMATH_CALUDE_product_ab_equals_one_l2540_254078

-- Define the variables a and b
variable (a b : ℝ)

-- State the theorem
theorem product_ab_equals_one (h1 : a - b = 4) (h2 : a^2 + b^2 = 18) : a * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_ab_equals_one_l2540_254078


namespace NUMINAMATH_CALUDE_inequality_proof_l2540_254062

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a^2 / (b + c)) + (b^2 / (c + a)) + (c^2 / (a + b)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2540_254062


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l2540_254087

/-- A function from positive reals to positive reals -/
def PositiveRealFunction := {f : ℝ → ℝ // ∀ x, 0 < x → 0 < f x}

/-- The functional equation property -/
def SatisfiesFunctionalEquation (f : PositiveRealFunction) : Prop :=
  ∀ x y, 0 < x → 0 < y → f.val (x^y) = (f.val x)^(f.val y)

/-- The main theorem -/
theorem functional_equation_solutions (f : PositiveRealFunction) 
  (h : SatisfiesFunctionalEquation f) :
  (∀ x, 0 < x → f.val x = 1) ∨ (∀ x, 0 < x → f.val x = x) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l2540_254087


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2540_254067

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := 1 / (3 + 4 * I)
  Complex.im z = -4 / 25 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2540_254067


namespace NUMINAMATH_CALUDE_not_in_sequence_l2540_254028

theorem not_in_sequence : ¬∃ (n : ℕ), 24 - 2 * n = 3 := by sorry

end NUMINAMATH_CALUDE_not_in_sequence_l2540_254028


namespace NUMINAMATH_CALUDE_rationalize_denominator_35_sqrt_35_l2540_254002

theorem rationalize_denominator_35_sqrt_35 :
  (35 : ℝ) / Real.sqrt 35 = Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_35_sqrt_35_l2540_254002


namespace NUMINAMATH_CALUDE_town_population_distribution_l2540_254003

/-- Represents a category in the pie chart --/
structure Category where
  name : String
  percentage : ℝ

/-- Represents a pie chart with three categories --/
structure PieChart where
  categories : Fin 3 → Category
  sum_to_100 : (categories 0).percentage + (categories 1).percentage + (categories 2).percentage = 100

/-- The main theorem --/
theorem town_population_distribution (chart : PieChart) 
  (h1 : (chart.categories 0).name = "less than 5,000 residents")
  (h2 : (chart.categories 1).name = "5,000 to 20,000 residents")
  (h3 : (chart.categories 2).name = "20,000 or more residents")
  (h4 : (chart.categories 1).percentage = 40) :
  (chart.categories 1).percentage = 40 := by
  sorry

end NUMINAMATH_CALUDE_town_population_distribution_l2540_254003


namespace NUMINAMATH_CALUDE_james_spent_six_l2540_254090

/-- The total amount James spent on milk, bananas, and sales tax -/
def total_spent (milk_price banana_price tax_rate : ℚ) : ℚ :=
  let subtotal := milk_price + banana_price
  let tax := subtotal * tax_rate
  subtotal + tax

/-- Theorem stating that James spent $6 given the problem conditions -/
theorem james_spent_six :
  total_spent 3 2 (1/5) = 6 := by
  sorry

end NUMINAMATH_CALUDE_james_spent_six_l2540_254090


namespace NUMINAMATH_CALUDE_saras_house_difference_l2540_254054

theorem saras_house_difference (sara_house : ℕ) (nada_house : ℕ) : 
  sara_house = 1000 → nada_house = 450 → sara_house - 2 * nada_house = 100 := by
  sorry

end NUMINAMATH_CALUDE_saras_house_difference_l2540_254054


namespace NUMINAMATH_CALUDE_mat_coverage_fraction_l2540_254007

/-- The fraction of a square tabletop covered by a circular mat -/
theorem mat_coverage_fraction (mat_diameter : ℝ) (table_side : ℝ) 
  (h1 : mat_diameter = 18) (h2 : table_side = 24) : 
  (π * (mat_diameter / 2)^2) / (table_side^2) = π / 7 := by
  sorry

end NUMINAMATH_CALUDE_mat_coverage_fraction_l2540_254007


namespace NUMINAMATH_CALUDE_cubic_expression_evaluation_l2540_254091

theorem cubic_expression_evaluation : 101^3 + 3*(101^2) - 3*101 + 9 = 1060610 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_evaluation_l2540_254091


namespace NUMINAMATH_CALUDE_largest_n_divisibility_l2540_254055

theorem largest_n_divisibility : ∃ (n : ℕ), n = 20 ∧ 
  (∀ m : ℕ, m > n → ¬((m^3 + 150) % (m + 5) = 0)) ∧ 
  ((n^3 + 150) % (n + 5) = 0) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_divisibility_l2540_254055


namespace NUMINAMATH_CALUDE_teacher_friends_count_l2540_254089

theorem teacher_friends_count (total_students : ℕ) 
  (both_friends : ℕ) (neither_friends : ℕ) (friend_difference : ℕ) :
  total_students = 50 →
  both_friends = 30 →
  neither_friends = 1 →
  friend_difference = 7 →
  ∃ (zhang_friends : ℕ),
    zhang_friends = 43 ∧
    zhang_friends + (zhang_friends - friend_difference) - both_friends + neither_friends = total_students :=
by sorry

end NUMINAMATH_CALUDE_teacher_friends_count_l2540_254089


namespace NUMINAMATH_CALUDE_quadratic_point_relationship_l2540_254064

/-- A quadratic function of the form y = -(x-1)² + k -/
def quadratic_function (k : ℝ) (x : ℝ) : ℝ := -(x - 1)^2 + k

theorem quadratic_point_relationship (k : ℝ) (y₁ y₂ y₃ : ℝ) :
  quadratic_function k (-1) = y₁ →
  quadratic_function k 2 = y₂ →
  quadratic_function k 4 = y₃ →
  y₃ < y₁ ∧ y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_point_relationship_l2540_254064


namespace NUMINAMATH_CALUDE_pascal_triangle_value_l2540_254033

/-- The number of elements in the row of Pascal's triangle we're considering -/
def row_length : Nat := 47

/-- The position of the number we're looking for in the row (1-indexed) -/
def target_position : Nat := 45

/-- The row number in Pascal's triangle (0-indexed) -/
def row_number : Nat := row_length - 1

/-- The binomial coefficient we need to calculate -/
def pascal_number : Nat := Nat.choose row_number (target_position - 1)

theorem pascal_triangle_value : pascal_number = 1035 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_value_l2540_254033


namespace NUMINAMATH_CALUDE_binomial_identities_l2540_254073

theorem binomial_identities (n k : ℕ+) :
  (Nat.choose n k + Nat.choose n (k + 1) = Nat.choose (n + 1) (k + 1)) ∧
  (Nat.choose n k = (n / k) * Nat.choose (n - 1) (k - 1)) := by
  sorry

end NUMINAMATH_CALUDE_binomial_identities_l2540_254073


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l2540_254018

theorem consecutive_odd_integers_sum (x : ℤ) : 
  (∃ y : ℤ, y = x + 2 ∧ x % 2 = 1 ∧ y % 2 = 1 ∧ y = 3 * x) → 
  x + (x + 2) = 4 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l2540_254018


namespace NUMINAMATH_CALUDE_zoo_population_is_90_l2540_254041

/-- Calculates the final number of animals in a zoo after a series of events --/
def final_zoo_population (initial_animals : ℕ) 
                         (gorillas_sent : ℕ) 
                         (hippo_adopted : ℕ) 
                         (rhinos_added : ℕ) 
                         (lion_cubs_born : ℕ) : ℕ :=
  initial_animals - gorillas_sent + hippo_adopted + rhinos_added + lion_cubs_born + 2 * lion_cubs_born

/-- Theorem stating that the final zoo population is 90 given the specific events --/
theorem zoo_population_is_90 : 
  final_zoo_population 68 6 1 3 8 = 90 := by
  sorry


end NUMINAMATH_CALUDE_zoo_population_is_90_l2540_254041


namespace NUMINAMATH_CALUDE_equivalence_condition_l2540_254038

theorem equivalence_condition (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ (a, b) ≠ (0, 0)) : 
  (1 / a < 1 / b) ↔ (a * b / (a^3 - b^3) > 0) :=
by sorry

end NUMINAMATH_CALUDE_equivalence_condition_l2540_254038


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_one_l2540_254053

theorem sin_cos_sum_equals_one : 
  Real.sin (47 * π / 180) * Real.cos (43 * π / 180) + 
  Real.sin (137 * π / 180) * Real.sin (43 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_one_l2540_254053


namespace NUMINAMATH_CALUDE_complement_of_union_l2540_254024

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 3, 5}
def N : Set ℕ := {4, 5}

theorem complement_of_union :
  (M ∪ N)ᶜ = {1, 6} :=
by sorry

end NUMINAMATH_CALUDE_complement_of_union_l2540_254024


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2540_254092

def f (x : ℕ) : ℚ := (x^4 + 625 : ℚ)

theorem complex_fraction_simplification :
  (f 20 * f 40 * f 60 * f 80) / (f 10 * f 30 * f 50 * f 70) = 7 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2540_254092


namespace NUMINAMATH_CALUDE_cube_parallel_edge_pairs_l2540_254044

/-- A cube is a three-dimensional geometric shape with 12 edges. -/
structure Cube where
  edges : Fin 12
  dimensions : Fin 3

/-- A pair of parallel edges in a cube. -/
structure ParallelEdgePair where
  edge1 : Fin 12
  edge2 : Fin 12

/-- The number of parallel edge pairs in a cube. -/
def parallel_edge_pairs (c : Cube) : ℕ := 18

/-- Theorem: A cube has 18 pairs of parallel edges. -/
theorem cube_parallel_edge_pairs (c : Cube) : 
  parallel_edge_pairs c = 18 := by sorry

end NUMINAMATH_CALUDE_cube_parallel_edge_pairs_l2540_254044


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l2540_254063

theorem min_value_x_plus_2y (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (heq : x + 2*y + 2*x*y = 8) : 
  ∀ z, z = x + 2*y → z ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l2540_254063


namespace NUMINAMATH_CALUDE_equation1_solution_equation2_no_solution_l2540_254015

-- Define the equations
def equation1 (x : ℝ) : Prop := (3 / (x^2 - 9)) + (x / (x - 3)) = 1
def equation2 (x : ℝ) : Prop := 2 - (1 / (2 - x)) = (3 - x) / (x - 2)

-- Theorem for equation 1
theorem equation1_solution : 
  ∃! x : ℝ, equation1 x ∧ x ≠ 3 ∧ x ≠ -3 := by sorry

-- Theorem for equation 2
theorem equation2_no_solution : 
  ∀ x : ℝ, ¬(equation2 x ∧ x ≠ 2) := by sorry

end NUMINAMATH_CALUDE_equation1_solution_equation2_no_solution_l2540_254015


namespace NUMINAMATH_CALUDE_two_people_walking_problem_l2540_254000

/-- Two people walking problem -/
theorem two_people_walking_problem (x y : ℝ) : 
  (∃ (distance : ℝ), distance = 18) →
  (∃ (time_meeting : ℝ), time_meeting = 2) →
  (∃ (time_catchup : ℝ), time_catchup = 4) →
  (∃ (time_headstart : ℝ), time_headstart = 1) →
  (2 * x + 2 * y = 18 ∧ 5 * x - 4 * y = 18) := by
sorry

end NUMINAMATH_CALUDE_two_people_walking_problem_l2540_254000


namespace NUMINAMATH_CALUDE_temperature_84_latest_time_l2540_254008

/-- Temperature model as a function of time -/
def temperature (t : ℝ) : ℝ := -t^2 + 14*t + 40

/-- The time when the temperature is 84 degrees -/
def temperature_84 (t : ℝ) : Prop := temperature t = 84

/-- The latest time when the temperature is 84 degrees -/
def latest_time_84 : ℝ := 11

theorem temperature_84_latest_time :
  temperature_84 latest_time_84 ∧
  ∀ t, t > latest_time_84 → ¬(temperature_84 t) :=
sorry

end NUMINAMATH_CALUDE_temperature_84_latest_time_l2540_254008


namespace NUMINAMATH_CALUDE_min_value_a_plus_b_l2540_254019

theorem min_value_a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / (a + 1) + 2 / (1 + b) = 1) : 
  ∀ x y : ℝ, x > 0 → y > 0 → 1 / (x + 1) + 2 / (1 + y) = 1 → a + b ≤ x + y ∧ a + b = 2 * Real.sqrt 2 + 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_plus_b_l2540_254019


namespace NUMINAMATH_CALUDE_julia_tag_difference_l2540_254048

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := 11

/-- The number of kids Julia played with on Tuesday -/
def tuesday_kids : ℕ := 12

/-- The difference in the number of kids Julia played with on Tuesday compared to Monday -/
def difference : ℕ := tuesday_kids - monday_kids

theorem julia_tag_difference : difference = 1 := by
  sorry

end NUMINAMATH_CALUDE_julia_tag_difference_l2540_254048


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l2540_254061

theorem trig_expression_simplification :
  let left_numerator := Real.sin (15 * π / 180) + Real.sin (30 * π / 180) + 
                        Real.sin (45 * π / 180) + Real.sin (60 * π / 180) + 
                        Real.sin (75 * π / 180)
  let denominator := Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * 
                     Real.cos (30 * π / 180) * 2
  let right_numerator := 2 * Real.sqrt 2 * Real.cos (22.5 * π / 180) * 
                         Real.cos (7.5 * π / 180)
  left_numerator / denominator = right_numerator / denominator := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_simplification_l2540_254061


namespace NUMINAMATH_CALUDE_negation_of_implication_l2540_254046

theorem negation_of_implication (x y : ℝ) : 
  ¬(x + y = 1 → x * y ≤ 1) ↔ (x + y = 1 ∧ x * y > 1) :=
sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2540_254046


namespace NUMINAMATH_CALUDE_abc_sum_l2540_254086

/-- Given prime numbers a, b, c satisfying abc + a = 851, prove a + b + c = 50 -/
theorem abc_sum (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c)
  (heq : a * b * c + a = 851) : a + b + c = 50 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_l2540_254086


namespace NUMINAMATH_CALUDE_correct_group_formations_l2540_254021

/-- The number of ways to form n groups of 2 from 2n soldiers -/
def groupFormations (n : ℕ) : ℕ × ℕ :=
  (Nat.factorial (2*n) / Nat.factorial n,
   Nat.factorial (2*n) / (2^n * Nat.factorial n))

/-- Theorem stating the correct number of group formations for both cases -/
theorem correct_group_formations (n : ℕ) :
  groupFormations n = (Nat.factorial (2*n) / Nat.factorial n,
                       Nat.factorial (2*n) / (2^n * Nat.factorial n)) :=
by sorry

end NUMINAMATH_CALUDE_correct_group_formations_l2540_254021


namespace NUMINAMATH_CALUDE_system_solution_l2540_254025

theorem system_solution (x y : ℝ) : 
  (x^x = y ∧ x^y = y^x) ↔ ((x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) ∨ (x = 2 ∧ y = 4)) :=
sorry

end NUMINAMATH_CALUDE_system_solution_l2540_254025


namespace NUMINAMATH_CALUDE_stock_price_increase_l2540_254016

theorem stock_price_increase (opening_price closing_price : ℝ) 
  (percent_increase : ℝ) : 
  opening_price = 6 → 
  percent_increase = 33.33 → 
  closing_price = opening_price * (1 + percent_increase / 100) → 
  closing_price = 8 := by
sorry

end NUMINAMATH_CALUDE_stock_price_increase_l2540_254016


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2540_254093

def original_number : ℕ := 42398
def divisor : ℕ := 15
def number_to_subtract : ℕ := 8

theorem least_subtraction_for_divisibility :
  (∀ k : ℕ, k < number_to_subtract → ¬(divisor ∣ (original_number - k))) ∧
  (divisor ∣ (original_number - number_to_subtract)) := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2540_254093


namespace NUMINAMATH_CALUDE_inequality_proof_l2540_254075

theorem inequality_proof (a b c : ℝ) : 
  ((a^2 + b^2 + a*c)^2 + (a^2 + b^2 + b*c)^2) / (a^2 + b^2) ≥ (a + b + c)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2540_254075


namespace NUMINAMATH_CALUDE_sum_9000_eq_1355_l2540_254020

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  /-- The sum of the first 3000 terms -/
  sum_3000 : ℝ
  /-- The sum of the first 6000 terms -/
  sum_6000 : ℝ
  /-- The sum of the first 3000 terms is 500 -/
  sum_3000_eq : sum_3000 = 500
  /-- The sum of the first 6000 terms is 950 -/
  sum_6000_eq : sum_6000 = 950

/-- The sum of the first 9000 terms of the geometric sequence is 1355 -/
theorem sum_9000_eq_1355 (seq : GeometricSequence) : ℝ := by
  sorry

end NUMINAMATH_CALUDE_sum_9000_eq_1355_l2540_254020


namespace NUMINAMATH_CALUDE_estimate_black_balls_l2540_254065

theorem estimate_black_balls (total_balls : Nat) (total_draws : Nat) (black_draws : Nat) :
  total_balls = 15 →
  total_draws = 100 →
  black_draws = 60 →
  (black_draws : Real) / total_draws * total_balls = 9 := by
  sorry

end NUMINAMATH_CALUDE_estimate_black_balls_l2540_254065


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_find_m_for_intersection_l2540_254081

-- Define the sets A and B
def A : Set ℝ := {x | (x - 5) / (x + 1) ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

-- Theorem for part (1)
theorem intersection_A_complement_B : 
  A ∩ (Set.univ \ B 3) = Set.Icc 3 5 := by sorry

-- Theorem for part (2)
theorem find_m_for_intersection : 
  ∃ m : ℝ, A ∩ B m = Set.Ioo (-1) 4 → m = 8 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_find_m_for_intersection_l2540_254081


namespace NUMINAMATH_CALUDE_min_operations_for_jugs_l2540_254098

/-- Represents the state of the two jugs -/
structure JugState :=
  (jug7 : ℕ)
  (jug5 : ℕ)

/-- Represents an operation on the jugs -/
inductive Operation
  | Fill7
  | Fill5
  | Empty7
  | Empty5
  | Pour7to5
  | Pour5to7

/-- Applies an operation to a JugState -/
def applyOperation (state : JugState) (op : Operation) : JugState :=
  match op with
  | Operation.Fill7 => ⟨7, state.jug5⟩
  | Operation.Fill5 => ⟨state.jug7, 5⟩
  | Operation.Empty7 => ⟨0, state.jug5⟩
  | Operation.Empty5 => ⟨state.jug7, 0⟩
  | Operation.Pour7to5 => 
      let amount := min state.jug7 (5 - state.jug5)
      ⟨state.jug7 - amount, state.jug5 + amount⟩
  | Operation.Pour5to7 => 
      let amount := min state.jug5 (7 - state.jug7)
      ⟨state.jug7 + amount, state.jug5 - amount⟩

/-- Checks if a sequence of operations results in the desired state -/
def isValidSolution (ops : List Operation) : Prop :=
  let finalState := ops.foldl applyOperation ⟨0, 0⟩
  finalState.jug7 = 1 ∧ finalState.jug5 = 1

/-- The main theorem to be proved -/
theorem min_operations_for_jugs : 
  ∃ (ops : List Operation), isValidSolution ops ∧ ops.length = 42 ∧
  (∀ (other_ops : List Operation), isValidSolution other_ops → other_ops.length ≥ 42) :=
sorry

end NUMINAMATH_CALUDE_min_operations_for_jugs_l2540_254098


namespace NUMINAMATH_CALUDE_unique_line_pair_l2540_254052

/-- Two equations represent the same line if they have the same slope and y-intercept -/
def same_line (a b : ℝ) : Prop :=
  ∃ (m c : ℝ), ∀ (x y : ℝ),
    (2 * x + a * y + 10 = 0 ↔ y = m * x + c) ∧
    (b * x - 3 * y - 15 = 0 ↔ y = m * x + c)

/-- There exists exactly one pair (a, b) such that the given equations represent the same line -/
theorem unique_line_pair : ∃! (p : ℝ × ℝ), same_line p.1 p.2 := by sorry

end NUMINAMATH_CALUDE_unique_line_pair_l2540_254052


namespace NUMINAMATH_CALUDE_cone_height_increase_l2540_254010

theorem cone_height_increase (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let V := (1/3) * Real.pi * r^2 * h
  let V' := 2.3 * V
  ∃ x : ℝ, V' = (1/3) * Real.pi * r^2 * (h * (1 + x/100)) → x = 130 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_increase_l2540_254010


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l2540_254080

theorem repeating_decimal_sum : 
  (0.2222222 : ℚ) + (0.04040404 : ℚ) + (0.00080008 : ℚ) = 878 / 9999 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l2540_254080


namespace NUMINAMATH_CALUDE_units_digit_of_product_l2540_254069

theorem units_digit_of_product (a b c : ℕ) : 
  (2^1501 * 5^1502 * 11^1503) % 10 = 0 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l2540_254069


namespace NUMINAMATH_CALUDE_right_triangle_tan_A_l2540_254031

theorem right_triangle_tan_A (A B C : Real) (sinB : Real) :
  -- ABC is a right triangle with angle C = 90°
  A + B + C = Real.pi →
  C = Real.pi / 2 →
  -- sin B = 3/5
  sinB = 3 / 5 →
  -- tan A = 4/3
  Real.tan A = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_tan_A_l2540_254031


namespace NUMINAMATH_CALUDE_supermarket_comparison_l2540_254023

/-- Represents the cost of shopping at Supermarket A -/
def costA (x : ℝ) : ℝ := 200 + 0.8 * (x - 200)

/-- Represents the cost of shopping at Supermarket B -/
def costB (x : ℝ) : ℝ := 100 + 0.85 * (x - 100)

/-- The original cost of shopping is greater than 200 yuan -/
axiom cost_gt_200 : ∀ x : ℝ, x > 200

theorem supermarket_comparison :
  (costB 300 < costA 300) ∧ (∃ x : ℝ, x > 200 ∧ costA x = costB x ∧ x = 500) := by
  sorry


end NUMINAMATH_CALUDE_supermarket_comparison_l2540_254023


namespace NUMINAMATH_CALUDE_quadratic_solution_set_implies_linear_and_inverse_quadratic_l2540_254079

/-- Given a quadratic function f(x) = ax² + bx + c, where a, b, and c are real numbers and a ≠ 0 -/
def QuadraticFunction (a b c : ℝ) := λ x : ℝ => a * x^2 + b * x + c

theorem quadratic_solution_set_implies_linear_and_inverse_quadratic
  (a b c : ℝ) (ha : a ≠ 0) :
  (∀ x, QuadraticFunction a b c x > 0 ↔ x < -2 ∨ x > 3) →
  (∀ x, b * x - c > 0 ↔ x < 6) ∧
  (∀ x, c * x^2 - b * x + a ≥ 0 ↔ -1/3 ≤ x ∧ x ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_set_implies_linear_and_inverse_quadratic_l2540_254079


namespace NUMINAMATH_CALUDE_triangle_angle_c_l2540_254011

theorem triangle_angle_c (A B C : ℝ) : 
  A - B = 10 → B = A / 2 → A + B + C = 180 → C = 150 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_c_l2540_254011


namespace NUMINAMATH_CALUDE_xy_greater_than_xz_l2540_254014

theorem xy_greater_than_xz (x y z : ℝ) 
  (h1 : x > y) (h2 : y > z) (h3 : x + y + z = 0) : x * y > x * z := by
  sorry

end NUMINAMATH_CALUDE_xy_greater_than_xz_l2540_254014


namespace NUMINAMATH_CALUDE_stone_skipping_ratio_l2540_254076

theorem stone_skipping_ratio (x y : ℕ) : 
  x > 0 → -- First throw has at least one skip
  x + 2 > 0 → -- Second throw has at least one skip
  y > 0 → -- Third throw has at least one skip
  y - 3 > 0 → -- Fourth throw has at least one skip
  y - 2 = 8 → -- Fifth throw skips 8 times
  x + (x + 2) + y + (y - 3) + (y - 2) = 33 → -- Total skips is 33
  y = x + 2 -- Ratio of third to second throw is 1:1
  := by sorry

end NUMINAMATH_CALUDE_stone_skipping_ratio_l2540_254076


namespace NUMINAMATH_CALUDE_complex_magnitude_l2540_254017

theorem complex_magnitude (z : ℂ) (h : (1 + Complex.I) * z = -4 + 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2540_254017


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formulas_l2540_254074

/-- An arithmetic sequence {a_n} -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formulas
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + a 7 = 20)
  (h_diff : a 11 - a 8 = 18) :
  (∃ (an : ℕ → ℝ), ∀ n, an n = 6 * n - 14 ∧ a n = an n) ∧
  (∃ (bn : ℕ → ℝ), ∀ n, bn n = 2 * n - 10 ∧
    (∀ k, ∃ m, a m = bn (3*k - 2) ∧ a (m+1) = bn (3*k + 1))) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formulas_l2540_254074


namespace NUMINAMATH_CALUDE_school_population_equality_l2540_254005

theorem school_population_equality (m d : ℕ) (M D : ℝ) :
  m > 0 → d > 0 →
  (M / m + D / d) / 2 = (M + D) / (m + d) →
  m = d :=
sorry

end NUMINAMATH_CALUDE_school_population_equality_l2540_254005


namespace NUMINAMATH_CALUDE_negative_slope_probability_l2540_254051

def LineSet : Set ℤ := {-3, -1, 0, 2, 7}

def ValidPair (a b : ℤ) : Prop :=
  a ∈ LineSet ∧ b ∈ LineSet ∧ a ≠ b

def NegativeSlope (a b : ℤ) : Prop :=
  ValidPair a b ∧ (a / b < 0)

def TotalPairs : ℕ := 20

def NegativeSlopePairs : ℕ := 4

theorem negative_slope_probability :
  (NegativeSlopePairs : ℚ) / TotalPairs = 1 / 5 :=
sorry

end NUMINAMATH_CALUDE_negative_slope_probability_l2540_254051


namespace NUMINAMATH_CALUDE_oatmeal_cookie_baggies_l2540_254001

def total_cookies : ℝ := 41.0
def chocolate_chip_cookies : ℝ := 13.0
def cookies_per_bag : ℝ := 9.0

theorem oatmeal_cookie_baggies :
  ⌊(total_cookies - chocolate_chip_cookies) / cookies_per_bag⌋ = 3 :=
by sorry

end NUMINAMATH_CALUDE_oatmeal_cookie_baggies_l2540_254001


namespace NUMINAMATH_CALUDE_jim_current_age_l2540_254040

/-- Represents the ages of Jim, Fred, and Sam -/
structure Ages where
  jim : ℕ
  fred : ℕ
  sam : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.jim = 2 * ages.fred ∧
  ages.fred = ages.sam + 9 ∧
  ages.jim - 6 = 5 * (ages.sam - 6)

/-- The theorem stating Jim's current age -/
theorem jim_current_age :
  ∃ ages : Ages, satisfiesConditions ages ∧ ages.jim = 46 :=
sorry

end NUMINAMATH_CALUDE_jim_current_age_l2540_254040


namespace NUMINAMATH_CALUDE_worker_b_days_l2540_254009

/-- Represents the daily wages and work days of three workers -/
structure WorkerData where
  wage_a : ℚ
  wage_b : ℚ
  wage_c : ℚ
  days_a : ℕ
  days_b : ℕ
  days_c : ℕ

/-- Calculates the total earnings of the workers -/
def totalEarnings (data : WorkerData) : ℚ :=
  data.wage_a * data.days_a + data.wage_b * data.days_b + data.wage_c * data.days_c

theorem worker_b_days (data : WorkerData) 
  (h1 : data.days_a = 6)
  (h2 : data.days_c = 4)
  (h3 : data.wage_a / data.wage_b = 3 / 4)
  (h4 : data.wage_b / data.wage_c = 4 / 5)
  (h5 : totalEarnings data = 1702)
  (h6 : data.wage_c = 115) :
  data.days_b = 9 := by
sorry

end NUMINAMATH_CALUDE_worker_b_days_l2540_254009


namespace NUMINAMATH_CALUDE_salad_dressing_vinegar_weight_l2540_254056

/-- Given a bowl of salad dressing with specified properties, prove the weight of vinegar. -/
theorem salad_dressing_vinegar_weight
  (bowl_capacity : ℝ)
  (oil_fraction : ℝ)
  (vinegar_fraction : ℝ)
  (oil_density : ℝ)
  (total_weight : ℝ)
  (h_bowl : bowl_capacity = 150)
  (h_oil_frac : oil_fraction = 2/3)
  (h_vinegar_frac : vinegar_fraction = 1/3)
  (h_oil_density : oil_density = 5)
  (h_total_weight : total_weight = 700)
  (h_fractions : oil_fraction + vinegar_fraction = 1) :
  (total_weight - oil_density * (oil_fraction * bowl_capacity)) / (vinegar_fraction * bowl_capacity) = 4 := by
  sorry


end NUMINAMATH_CALUDE_salad_dressing_vinegar_weight_l2540_254056


namespace NUMINAMATH_CALUDE_studentB_is_optimal_l2540_254035

-- Define the structure for a student
structure Student where
  name : String
  average : ℝ
  variance : ℝ

-- Define the students
def studentA : Student := { name := "A", average := 92, variance := 3.6 }
def studentB : Student := { name := "B", average := 95, variance := 3.6 }
def studentC : Student := { name := "C", average := 95, variance := 7.4 }
def studentD : Student := { name := "D", average := 95, variance := 8.1 }

-- Define the list of all students
def students : List Student := [studentA, studentB, studentC, studentD]

-- Function to determine if one student is better than another
def isBetterStudent (s1 s2 : Student) : Prop :=
  (s1.average > s2.average) ∨ (s1.average = s2.average ∧ s1.variance < s2.variance)

-- Theorem stating that student B is the optimal choice
theorem studentB_is_optimal : 
  ∀ s ∈ students, s.name ≠ "B" → isBetterStudent studentB s :=
by sorry

end NUMINAMATH_CALUDE_studentB_is_optimal_l2540_254035


namespace NUMINAMATH_CALUDE_discount_problem_l2540_254072

theorem discount_problem (list_price : ℝ) (final_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  list_price = 70 →
  final_price = 59.85 →
  discount1 = 10 →
  (list_price * (1 - discount1 / 100) * (1 - discount2 / 100) = final_price) →
  discount2 = 5 := by
sorry

end NUMINAMATH_CALUDE_discount_problem_l2540_254072


namespace NUMINAMATH_CALUDE_factorization_equality_l2540_254088

theorem factorization_equality (x : ℝ) : 9*x^3 - 18*x^2 + 9*x = 9*x*(x-1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2540_254088


namespace NUMINAMATH_CALUDE_equation_one_l2540_254034

theorem equation_one (x : ℝ) : x * |x| = 4 ↔ x = 2 := by sorry

end NUMINAMATH_CALUDE_equation_one_l2540_254034


namespace NUMINAMATH_CALUDE_tan_theta_two_implies_expression_equals_negative_two_l2540_254082

theorem tan_theta_two_implies_expression_equals_negative_two (θ : Real) 
  (h : Real.tan θ = 2) : 
  (Real.sin (π / 2 + θ) - Real.cos (π - θ)) / 
  (Real.sin (π / 2 - θ) - Real.sin (π - θ)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_two_implies_expression_equals_negative_two_l2540_254082


namespace NUMINAMATH_CALUDE_jill_watching_time_l2540_254085

/-- The length of the first show Jill watched, in minutes. -/
def first_show_length : ℝ := 30

/-- The length of the second show Jill watched, in minutes. -/
def second_show_length : ℝ := 4 * first_show_length

/-- The total time Jill spent watching shows, in minutes. -/
def total_watching_time : ℝ := 150

theorem jill_watching_time :
  first_show_length + second_show_length = total_watching_time :=
by sorry

end NUMINAMATH_CALUDE_jill_watching_time_l2540_254085


namespace NUMINAMATH_CALUDE_napkin_length_calculation_l2540_254060

/-- Given a tablecloth and napkins with specified dimensions, calculate the length of each napkin. -/
theorem napkin_length_calculation
  (tablecloth_length : ℕ)
  (tablecloth_width : ℕ)
  (num_napkins : ℕ)
  (napkin_width : ℕ)
  (total_material : ℕ)
  (h1 : tablecloth_length = 102)
  (h2 : tablecloth_width = 54)
  (h3 : num_napkins = 8)
  (h4 : napkin_width = 7)
  (h5 : total_material = 5844)
  (h6 : total_material = tablecloth_length * tablecloth_width + num_napkins * napkin_width * (total_material - tablecloth_length * tablecloth_width) / (napkin_width * num_napkins)) :
  (total_material - tablecloth_length * tablecloth_width) / (napkin_width * num_napkins) = 6 := by
  sorry

#check napkin_length_calculation

end NUMINAMATH_CALUDE_napkin_length_calculation_l2540_254060


namespace NUMINAMATH_CALUDE_purchase_problem_l2540_254083

/-- Represents the prices and quantities of small light bulbs and electric motors --/
structure PurchaseInfo where
  bulb_price : ℝ
  motor_price : ℝ
  bulb_quantity : ℕ
  motor_quantity : ℕ

/-- Calculates the total cost of a purchase --/
def total_cost (info : PurchaseInfo) : ℝ :=
  info.bulb_price * info.bulb_quantity + info.motor_price * info.motor_quantity

/-- Theorem stating the properties of the purchase problem --/
theorem purchase_problem :
  ∃ (info : PurchaseInfo),
    -- Conditions
    info.bulb_price + info.motor_price = 12 ∧
    info.bulb_price * info.bulb_quantity = 30 ∧
    info.motor_price * info.motor_quantity = 45 ∧
    info.bulb_quantity = 2 * info.motor_quantity ∧
    -- Results
    info.bulb_price = 3 ∧
    info.motor_price = 9 ∧
    -- Optimal purchase
    (∀ (alt_info : PurchaseInfo),
      alt_info.bulb_quantity + alt_info.motor_quantity = 90 ∧
      alt_info.bulb_quantity ≤ alt_info.motor_quantity / 2 →
      total_cost info ≤ total_cost alt_info) ∧
    info.bulb_quantity = 30 ∧
    info.motor_quantity = 60 ∧
    total_cost info = 630 :=
  sorry


end NUMINAMATH_CALUDE_purchase_problem_l2540_254083


namespace NUMINAMATH_CALUDE_probability_three_out_of_seven_greater_than_six_l2540_254049

/-- The probability of a single 12-sided die showing a number greater than 6 -/
def p_greater_than_6 : ℚ := 1 / 2

/-- The number of dice rolled -/
def n : ℕ := 7

/-- The number of dice we want to show a number greater than 6 -/
def k : ℕ := 3

/-- The probability of exactly k out of n dice showing a number greater than 6 -/
def probability_k_out_of_n (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

theorem probability_three_out_of_seven_greater_than_six :
  probability_k_out_of_n n k p_greater_than_6 = 35 / 128 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_out_of_seven_greater_than_six_l2540_254049


namespace NUMINAMATH_CALUDE_product_remainder_l2540_254036

theorem product_remainder (N : ℕ) : 
  (1274 * 1275 * N * 1285) % 12 = 6 → N % 12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l2540_254036


namespace NUMINAMATH_CALUDE_lego_sale_quadruple_pieces_l2540_254077

/-- Represents the number of Lego pieces sold for each type -/
structure LegoSale where
  single : ℕ
  double : ℕ
  triple : ℕ
  quadruple : ℕ

/-- Calculates the total number of circles from a LegoSale -/
def totalCircles (sale : LegoSale) : ℕ :=
  sale.single + 2 * sale.double + 3 * sale.triple + 4 * sale.quadruple

/-- The main theorem to prove -/
theorem lego_sale_quadruple_pieces (sale : LegoSale) :
  sale.single = 100 →
  sale.double = 45 →
  sale.triple = 50 →
  totalCircles sale = 1000 →
  sale.quadruple = 165 := by
  sorry

#check lego_sale_quadruple_pieces

end NUMINAMATH_CALUDE_lego_sale_quadruple_pieces_l2540_254077
