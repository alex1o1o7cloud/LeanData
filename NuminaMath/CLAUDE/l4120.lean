import Mathlib

namespace NUMINAMATH_CALUDE_triangle_angle_solution_l4120_412004

theorem triangle_angle_solution (angle1 angle2 angle3 : ℝ) (x : ℝ) : 
  angle1 = 40 ∧ 
  angle2 = 4 * x ∧ 
  angle3 = 3 * x ∧ 
  angle1 + angle2 + angle3 = 180 →
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_solution_l4120_412004


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l4120_412087

theorem greatest_divisor_with_remainders : ∃ (n : ℕ), 
  n > 0 ∧
  (∀ m : ℕ, m > 0 ∧ 
    (3815 % m = 31 ∧ 4521 % m = 33) → 
    m ≤ n) ∧
  3815 % n = 31 ∧ 
  4521 % n = 33 ∧
  n = 64 := by
sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l4120_412087


namespace NUMINAMATH_CALUDE_sum_of_squares_16_to_30_l4120_412010

theorem sum_of_squares_16_to_30 :
  let sum_squares : (n : ℕ) → ℕ := λ n => n * (n + 1) * (2 * n + 1) / 6
  let sum_1_to_15 := 1280
  let sum_1_to_30 := sum_squares 30
  sum_1_to_30 - sum_1_to_15 = 8215 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_16_to_30_l4120_412010


namespace NUMINAMATH_CALUDE_total_people_after_hour_l4120_412081

/-- Represents the number of people in each line at the fair -/
structure FairLines where
  ferrisWheel : ℕ
  bumperCars : ℕ
  rollerCoaster : ℕ

/-- Calculates the total number of people across all lines after an hour -/
def totalPeopleAfterHour (initial : FairLines) (x y Z : ℕ) : ℕ :=
  Z + initial.rollerCoaster * 2

/-- Theorem stating the total number of people after an hour -/
theorem total_people_after_hour 
  (initial : FairLines)
  (x y Z : ℕ)
  (h1 : initial.ferrisWheel = 50)
  (h2 : initial.bumperCars = 50)
  (h3 : initial.rollerCoaster = 50)
  (h4 : Z = (50 - x) + (50 + y)) :
  totalPeopleAfterHour initial x y Z = Z + 100 := by
  sorry

#check total_people_after_hour

end NUMINAMATH_CALUDE_total_people_after_hour_l4120_412081


namespace NUMINAMATH_CALUDE_jaya_rank_from_bottom_l4120_412075

theorem jaya_rank_from_bottom (total_students : ℕ) (rank_from_top : ℕ) (rank_from_bottom : ℕ) : 
  total_students = 53 → 
  rank_from_top = 5 → 
  rank_from_bottom = total_students - rank_from_top + 1 →
  rank_from_bottom = 50 := by
sorry

end NUMINAMATH_CALUDE_jaya_rank_from_bottom_l4120_412075


namespace NUMINAMATH_CALUDE_xy_range_l4120_412036

theorem xy_range (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 8/y = 1) :
  ∃ (m : ℝ), m = 64 ∧ xy ≥ m ∧ ∀ (z : ℝ), z > m → ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2/a + 8/b = 1 ∧ a * b = z :=
sorry

end NUMINAMATH_CALUDE_xy_range_l4120_412036


namespace NUMINAMATH_CALUDE_farmland_equations_correct_l4120_412007

/-- Represents the farmland purchase problem -/
structure FarmlandProblem where
  total_acres : ℕ
  good_cost_per_acre : ℚ
  bad_cost_per_seven_acres : ℚ
  total_spent : ℚ

/-- Represents the system of equations for the farmland problem -/
def farmland_equations (p : FarmlandProblem) (x y : ℚ) : Prop :=
  x + y = p.total_acres ∧
  p.good_cost_per_acre * x + (p.bad_cost_per_seven_acres / 7) * y = p.total_spent

/-- Theorem stating that the system of equations correctly represents the farmland problem -/
theorem farmland_equations_correct (p : FarmlandProblem) (x y : ℚ) :
  p.total_acres = 100 →
  p.good_cost_per_acre = 300 →
  p.bad_cost_per_seven_acres = 500 →
  p.total_spent = 10000 →
  farmland_equations p x y ↔
    (x + y = 100 ∧ 300 * x + (500 / 7) * y = 10000) :=
by sorry

end NUMINAMATH_CALUDE_farmland_equations_correct_l4120_412007


namespace NUMINAMATH_CALUDE_tangents_form_diameter_l4120_412064

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define the circle E
def circle_E (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define a point on the circle E
def point_on_E (P : ℝ × ℝ) : Prop :=
  circle_E P.1 P.2

-- Define tangent lines from P to C
def tangent_to_C (P : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
  ∃ (Q : ℝ × ℝ), ellipse_C Q.1 Q.2 ∧ l Q.1 = Q.2 ∧
  ∀ (x y : ℝ), ellipse_C x y → (y - l x) * (Q.2 - l Q.1) ≥ 0

-- Define intersection points of tangents with E
def intersect_E (P : ℝ × ℝ) (l : ℝ → ℝ) (M : ℝ × ℝ) : Prop :=
  M ≠ P ∧ circle_E M.1 M.2 ∧ l M.1 = M.2

-- Main theorem
theorem tangents_form_diameter (P M N : ℝ × ℝ) (l₁ l₂ : ℝ → ℝ) :
  point_on_E P →
  tangent_to_C P l₁ →
  tangent_to_C P l₂ →
  intersect_E P l₁ M →
  intersect_E P l₂ N →
  (M.1 + N.1 = 0 ∧ M.2 + N.2 = 0) :=
sorry

end NUMINAMATH_CALUDE_tangents_form_diameter_l4120_412064


namespace NUMINAMATH_CALUDE_quadratic_polynomial_value_l4120_412057

/-- A quadratic polynomial -/
def QuadraticPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- Divisibility condition for the polynomial -/
def DivisibilityCondition (q : ℝ → ℝ) : Prop :=
  ∃ p : ℝ → ℝ, ∀ x : ℝ, q x^3 + x = p x * (x - 2) * (x + 2) * (x - 5)

theorem quadratic_polynomial_value (a b c : ℝ) :
  let q := QuadraticPolynomial a b c
  DivisibilityCondition q → q 10 = -139 * Real.rpow 2 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_value_l4120_412057


namespace NUMINAMATH_CALUDE_complex_product_theorem_l4120_412099

theorem complex_product_theorem : 
  let z1 : ℂ := -1 + 2*I
  let z2 : ℂ := 2 + I
  z1 * z2 = -4 + 3*I := by
  sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l4120_412099


namespace NUMINAMATH_CALUDE_nine_nine_nine_squared_plus_nine_nine_nine_l4120_412092

theorem nine_nine_nine_squared_plus_nine_nine_nine (n : ℕ) : 999 * 999 + 999 = 999000 := by
  sorry

end NUMINAMATH_CALUDE_nine_nine_nine_squared_plus_nine_nine_nine_l4120_412092


namespace NUMINAMATH_CALUDE_correct_division_l4120_412042

theorem correct_division (n : ℚ) : n / 22 = 2 → n / 20 = 2.2 := by
  sorry

end NUMINAMATH_CALUDE_correct_division_l4120_412042


namespace NUMINAMATH_CALUDE_point_in_region_l4120_412014

-- Define the point P
def P (a : ℝ) : ℝ × ℝ := (a, a + 1)

-- Define the inequality that represents the region
def in_region (x y a : ℝ) : Prop := x + a * y - 3 > 0

-- Theorem statement
theorem point_in_region (a : ℝ) :
  in_region (P a).1 (P a).2 a ↔ a < -3 ∨ a > 1 :=
sorry

end NUMINAMATH_CALUDE_point_in_region_l4120_412014


namespace NUMINAMATH_CALUDE_shaded_area_proof_l4120_412053

theorem shaded_area_proof (rectangle_length rectangle_width : ℝ)
  (triangle_a_leg1 triangle_a_leg2 : ℝ)
  (triangle_b_leg1 triangle_b_leg2 : ℝ)
  (h1 : rectangle_length = 14)
  (h2 : rectangle_width = 7)
  (h3 : triangle_a_leg1 = 8)
  (h4 : triangle_a_leg2 = 5)
  (h5 : triangle_b_leg1 = 6)
  (h6 : triangle_b_leg2 = 2) :
  rectangle_length * rectangle_width - 3 * ((1/2 * triangle_a_leg1 * triangle_a_leg2) + (1/2 * triangle_b_leg1 * triangle_b_leg2)) = 20 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_proof_l4120_412053


namespace NUMINAMATH_CALUDE_solution_equation_one_solution_equation_two_l4120_412059

-- Problem 1
theorem solution_equation_one (x : ℝ) : 
  3 * (x - 2)^2 - 27 = 0 ↔ x = 5 ∨ x = -1 := by sorry

-- Problem 2
theorem solution_equation_two (x : ℝ) :
  2 * (x + 1)^3 + 54 = 0 ↔ x = -4 := by sorry

end NUMINAMATH_CALUDE_solution_equation_one_solution_equation_two_l4120_412059


namespace NUMINAMATH_CALUDE_initial_plus_bought_equals_total_l4120_412094

/-- The number of bottle caps William had initially -/
def initial_caps : ℕ := 2

/-- The number of bottle caps William bought -/
def bought_caps : ℕ := 41

/-- The total number of bottle caps William has after buying more -/
def total_caps : ℕ := 43

/-- Theorem stating that the initial number of bottle caps plus the bought ones equals the total -/
theorem initial_plus_bought_equals_total : 
  initial_caps + bought_caps = total_caps := by sorry

end NUMINAMATH_CALUDE_initial_plus_bought_equals_total_l4120_412094


namespace NUMINAMATH_CALUDE_permutations_of_eight_distinct_objects_l4120_412068

theorem permutations_of_eight_distinct_objects : Nat.factorial 8 = 40320 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_eight_distinct_objects_l4120_412068


namespace NUMINAMATH_CALUDE_door_can_be_opened_l4120_412091

/-- Represents a device with toggle switches and a display -/
structure Device where
  combinations : Fin 32 → ℕ

/-- Represents the notebook used for communication -/
structure Notebook where
  pages : Fin 1001 → Option (Fin 32)

/-- Represents the state of the operation -/
structure OperationState where
  deviceA : Device
  deviceB : Device
  notebook : Notebook
  time : ℕ

/-- Checks if a matching combination is found -/
def isMatchingCombinationFound (state : OperationState) : Prop :=
  ∃ (i : Fin 32), state.deviceA.combinations i = state.deviceB.combinations i

/-- Defines the time constraints of the operation -/
def isWithinTimeConstraint (state : OperationState) : Prop :=
  state.time ≤ 75

/-- Theorem stating that a matching combination can be found within the time constraint -/
theorem door_can_be_opened (initialState : OperationState) :
  ∃ (finalState : OperationState),
    isMatchingCombinationFound finalState ∧
    isWithinTimeConstraint finalState :=
  sorry


end NUMINAMATH_CALUDE_door_can_be_opened_l4120_412091


namespace NUMINAMATH_CALUDE_flight_duration_sum_l4120_412097

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDifferenceInMinutes (t1 t2 : Time) : ℕ :=
  (t2.hours - t1.hours) * 60 + (t2.minutes - t1.minutes)

/-- Theorem: Flight duration calculation -/
theorem flight_duration_sum (departureTime : Time) (arrivalTime : Time) 
  (h m : ℕ) (hm : 0 < m ∧ m < 60) :
  departureTime.hours = 9 ∧ departureTime.minutes = 17 →
  arrivalTime.hours = 13 ∧ arrivalTime.minutes = 53 →
  timeDifferenceInMinutes departureTime arrivalTime = h * 60 + m →
  h + m = 41 := by
  sorry

#check flight_duration_sum

end NUMINAMATH_CALUDE_flight_duration_sum_l4120_412097


namespace NUMINAMATH_CALUDE_percentage_greater_than_l4120_412019

theorem percentage_greater_than (X Y Z : ℝ) : 
  (X - Y) / (Y + Z) * 100 = 100 * (X - Y) / (Y + Z) :=
by sorry

end NUMINAMATH_CALUDE_percentage_greater_than_l4120_412019


namespace NUMINAMATH_CALUDE_blue_marbles_count_l4120_412013

theorem blue_marbles_count (total : ℕ) (red : ℕ) (prob_red_or_white : ℚ) :
  total = 20 →
  red = 9 →
  prob_red_or_white = 7/10 →
  ∃ (blue white : ℕ),
    blue + red + white = total ∧
    (red + white : ℚ) / total = prob_red_or_white ∧
    blue = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_blue_marbles_count_l4120_412013


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l4120_412026

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 8 / x + 1 / y = 1) :
  x + 2 * y ≥ 18 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 8 / x₀ + 1 / y₀ = 1 ∧ x₀ + 2 * y₀ = 18 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l4120_412026


namespace NUMINAMATH_CALUDE_card_probability_theorem_probability_after_10_shuffles_value_l4120_412020

/-- The probability that card 6 is higher than card 3 after n shuffles -/
def p (n : ℕ) : ℚ :=
  (3^n - 2^n) / (2 * 3^n)

/-- The recurrence relation for the probability -/
def recurrence (p_prev : ℚ) : ℚ :=
  (4 * p_prev + 1) / 6

theorem card_probability_theorem (n : ℕ) :
  p n = recurrence (p (n - 1)) ∧ p 0 = 0 :=
by sorry

/-- The probability that card 6 is higher than card 3 after 10 shuffles -/
def probability_after_10_shuffles : ℚ := p 10

theorem probability_after_10_shuffles_value :
  probability_after_10_shuffles = (3^10 - 2^10) / (2 * 3^10) :=
by sorry

end NUMINAMATH_CALUDE_card_probability_theorem_probability_after_10_shuffles_value_l4120_412020


namespace NUMINAMATH_CALUDE_binomial_probability_problem_l4120_412001

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- Probability of a binomial random variable being greater than or equal to k -/
def prob_ge (X : BinomialRV) (k : ℕ) : ℝ :=
  sorry

theorem binomial_probability_problem (X Y : BinomialRV) 
  (hX : X.n = 2) (hY : Y.n = 4) (hp : X.p = Y.p)
  (h_prob : prob_ge X 1 = 5/9) : 
  prob_ge Y 2 = 11/27 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_problem_l4120_412001


namespace NUMINAMATH_CALUDE_sixth_sampled_item_is_101_l4120_412015

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  totalItems : ℕ
  sampleSize : ℕ
  startNumber : ℕ

/-- Calculates the nth sampled item number in a systematic sampling -/
def nthSampledItem (s : SystematicSampling) (n : ℕ) : ℕ :=
  s.startNumber + (s.totalItems / s.sampleSize) * (n - 1)

/-- The main theorem to prove -/
theorem sixth_sampled_item_is_101 :
  let s : SystematicSampling := {
    totalItems := 1000,
    sampleSize := 50,
    startNumber := 1
  }
  nthSampledItem s 6 = 101 := by sorry

end NUMINAMATH_CALUDE_sixth_sampled_item_is_101_l4120_412015


namespace NUMINAMATH_CALUDE_sean_patch_profit_l4120_412041

/-- Sean's patch business profit calculation -/
theorem sean_patch_profit :
  let order_size : ℕ := 100
  let cost_per_patch : ℚ := 125 / 100
  let selling_price : ℚ := 12
  let total_cost : ℚ := order_size * cost_per_patch
  let total_revenue : ℚ := order_size * selling_price
  let net_profit : ℚ := total_revenue - total_cost
  net_profit = 1075 := by sorry

end NUMINAMATH_CALUDE_sean_patch_profit_l4120_412041


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l4120_412077

/-- A quadratic function f(x) with leading coefficient a -/
def f (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The condition that f(x) > -2x for x ∈ (1,3) -/
def condition_solution_set (a b c : ℝ) : Prop :=
  ∀ x, 1 < x ∧ x < 3 → f a b c x > -2 * x

/-- The condition that f(x) + 6a = 0 has two equal roots -/
def condition_equal_roots (a b c : ℝ) : Prop :=
  ∃ x : ℝ, f a b c x + 6 * a = 0 ∧
    ∀ y : ℝ, f a b c y + 6 * a = 0 → y = x

/-- The condition that the maximum value of f(x) is positive -/
def condition_positive_max (a b c : ℝ) : Prop :=
  ∃ m : ℝ, m > 0 ∧ ∀ x : ℝ, f a b c x ≤ m

theorem quadratic_function_properties
  (a b c : ℝ)
  (ha : a < 0)
  (hf : condition_solution_set a b c) :
  (condition_equal_roots a b c →
    f a b c = fun x ↦ -x^2 - x - 3/5) ∧
  (condition_positive_max a b c →
    a < -2 - Real.sqrt 5 ∨ (-2 + Real.sqrt 5 < a ∧ a < 0)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l4120_412077


namespace NUMINAMATH_CALUDE_max_log_expression_l4120_412003

theorem max_log_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_eq : 4 * a - 2 * b + 25 * c = 0) :
  (∀ x y z, x > 0 → y > 0 → z > 0 → 4 * x - 2 * y + 25 * z = 0 →
    Real.log x + Real.log z - 2 * Real.log y ≤ Real.log a + Real.log c - 2 * Real.log b) ∧
  Real.log a + Real.log c - 2 * Real.log b = -2 := by
sorry

end NUMINAMATH_CALUDE_max_log_expression_l4120_412003


namespace NUMINAMATH_CALUDE_base10_512_equals_base6_2212_l4120_412023

-- Define a function to convert a list of digits in base 6 to a natural number
def base6ToNat (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => d + 6 * acc) 0

-- Define the theorem
theorem base10_512_equals_base6_2212 :
  512 = base6ToNat [2, 1, 2, 2] := by
  sorry


end NUMINAMATH_CALUDE_base10_512_equals_base6_2212_l4120_412023


namespace NUMINAMATH_CALUDE_system_solution_l4120_412035

theorem system_solution (x y : ℝ) : 
  x^5 + y^5 = 1 ∧ x^6 + y^6 = 1 ↔ (x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l4120_412035


namespace NUMINAMATH_CALUDE_julia_tag_game_l4120_412079

theorem julia_tag_game (monday tuesday : ℕ) 
  (h1 : monday = 12) 
  (h2 : tuesday = 7) : 
  monday + tuesday = 19 := by
  sorry

end NUMINAMATH_CALUDE_julia_tag_game_l4120_412079


namespace NUMINAMATH_CALUDE_axis_of_symmetry_l4120_412045

-- Define the function f with the given property
def f : ℝ → ℝ := sorry

-- Define the property that f(x) = f(6-x) for all x
axiom f_symmetry (x : ℝ) : f x = f (6 - x)

-- State the theorem: x = 3 is the axis of symmetry
theorem axis_of_symmetry :
  ∀ (x y : ℝ), f x = y ↔ f (6 - x) = y :=
by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_l4120_412045


namespace NUMINAMATH_CALUDE_no_xyz_solution_l4120_412047

theorem no_xyz_solution : ¬∃ (x y z : ℕ), 
  0 ≤ x ∧ x ≤ 9 ∧
  0 ≤ y ∧ y ≤ 9 ∧
  0 ≤ z ∧ z ≤ 9 ∧
  100 * x + 10 * y + z = y * (10 * x + z) := by
  sorry

end NUMINAMATH_CALUDE_no_xyz_solution_l4120_412047


namespace NUMINAMATH_CALUDE_no_fast_connectivity_algorithm_l4120_412085

/-- A graph with 64 vertices -/
def Graph := Fin 64 → Fin 64 → Bool

/-- Number of queries required -/
def required_queries : ℕ := 2016

/-- An algorithm that determines graph connectivity -/
def ConnectivityAlgorithm := Graph → Bool

/-- The number of queries an algorithm makes -/
def num_queries (alg : ConnectivityAlgorithm) : ℕ := sorry

/-- A graph is connected -/
def is_connected (g : Graph) : Prop := sorry

/-- Theorem: No algorithm can determine connectivity in fewer than 2016 queries -/
theorem no_fast_connectivity_algorithm :
  ¬∃ (alg : ConnectivityAlgorithm),
    (∀ g : Graph, alg g = is_connected g) ∧
    (num_queries alg < required_queries) :=
sorry

end NUMINAMATH_CALUDE_no_fast_connectivity_algorithm_l4120_412085


namespace NUMINAMATH_CALUDE_math_textbooks_in_one_box_l4120_412032

def total_textbooks : ℕ := 15
def math_textbooks : ℕ := 4
def boxes : ℕ := 3
def books_per_box : ℕ := 5

def probability_all_math_in_one_box : ℚ := 769 / 100947

theorem math_textbooks_in_one_box :
  let total_ways := (total_textbooks.choose books_per_box) * 
                    ((total_textbooks - books_per_box).choose books_per_box) * 
                    ((total_textbooks - 2 * books_per_box).choose books_per_box)
  let favorable_ways := boxes * 
                        ((total_textbooks - math_textbooks).choose 1) * 
                        ((total_textbooks - math_textbooks - 1).choose books_per_box) * 
                        ((total_textbooks - math_textbooks - 1 - books_per_box).choose books_per_box)
  (favorable_ways : ℚ) / total_ways = probability_all_math_in_one_box := by
  sorry

end NUMINAMATH_CALUDE_math_textbooks_in_one_box_l4120_412032


namespace NUMINAMATH_CALUDE_no_prime_sum_47_l4120_412069

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem no_prime_sum_47 : ¬∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 47 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_sum_47_l4120_412069


namespace NUMINAMATH_CALUDE_smallest_three_star_number_three_star_common_divisor_with_30_l4120_412017

/-- A three-star number is a three-digit positive integer that is the product of three distinct prime numbers. -/
def IsThreeStarNumber (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ n = p * q * r

/-- The smallest three-star number is 102. -/
theorem smallest_three_star_number : 
  IsThreeStarNumber 102 ∧ ∀ n, IsThreeStarNumber n → 102 ≤ n :=
sorry

/-- Every three-star number has a common divisor with 30 greater than 1. -/
theorem three_star_common_divisor_with_30 (n : ℕ) (h : IsThreeStarNumber n) : 
  ∃ d : ℕ, d > 1 ∧ d ∣ n ∧ d ∣ 30 :=
sorry

end NUMINAMATH_CALUDE_smallest_three_star_number_three_star_common_divisor_with_30_l4120_412017


namespace NUMINAMATH_CALUDE_inequality_preservation_l4120_412002

theorem inequality_preservation (a b : ℝ) (h : a > b) : a - 1 > b - 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l4120_412002


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l4120_412095

/-- Given a quadratic inequality ax^2 - bx + c > 0 with solution set (-1/2, 2), 
    prove properties about its coefficients -/
theorem quadratic_inequality_properties (a b c : ℝ) 
  (h : ∀ x, -1/2 < x ∧ x < 2 ↔ a * x^2 - b * x + c > 0) : 
  b < 0 ∧ c > 0 ∧ a - b + c > 0 ∧ a ≤ 0 ∧ a + b + c ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l4120_412095


namespace NUMINAMATH_CALUDE_empty_quadratic_inequality_solution_set_l4120_412029

/-- Given a quadratic inequality ax² + bx + c < 0 with a ≠ 0, 
    if the solution set is empty, then a > 0 and Δ ≤ 0, where Δ = b² - 4ac -/
theorem empty_quadratic_inequality_solution_set 
  (a b c : ℝ) (h1 : a ≠ 0) 
  (h2 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) : 
  a > 0 ∧ b^2 - 4*a*c ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_empty_quadratic_inequality_solution_set_l4120_412029


namespace NUMINAMATH_CALUDE_smallest_n_with_common_factor_l4120_412011

theorem smallest_n_with_common_factor : 
  ∃ (n : ℕ), n > 0 ∧ n = 10 ∧ 
  (∀ (m : ℕ), m > 0 ∧ m < n → ¬∃ (k : ℕ), k > 1 ∧ k ∣ (8*m - 3) ∧ k ∣ (5*m + 4)) ∧
  (∃ (k : ℕ), k > 1 ∧ k ∣ (8*n - 3) ∧ k ∣ (5*n + 4)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_common_factor_l4120_412011


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l4120_412005

theorem rectangular_box_volume (x : ℕ) (h : x > 0) :
  let volume := x * (2 * x) * (5 * x)
  (volume = 80 ∨ volume = 250 ∨ volume = 500 ∨ volume = 1000 ∨ volume = 2000) →
  volume = 80 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l4120_412005


namespace NUMINAMATH_CALUDE_symmetric_point_about_origin_l4120_412063

/-- Given a point P(-1, 2) in a rectangular coordinate system,
    its symmetric point about the origin has coordinates (1, -2). -/
theorem symmetric_point_about_origin :
  let P : ℝ × ℝ := (-1, 2)
  (- P.1, - P.2) = (1, -2) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_about_origin_l4120_412063


namespace NUMINAMATH_CALUDE_combined_tax_rate_calculation_l4120_412050

def john_tax_rate : ℚ := 30 / 100
def ingrid_tax_rate : ℚ := 40 / 100
def alice_tax_rate : ℚ := 25 / 100
def ben_tax_rate : ℚ := 35 / 100

def john_income : ℕ := 56000
def ingrid_income : ℕ := 74000
def alice_income : ℕ := 62000
def ben_income : ℕ := 80000

def total_tax : ℚ := john_tax_rate * john_income + ingrid_tax_rate * ingrid_income + 
                     alice_tax_rate * alice_income + ben_tax_rate * ben_income

def total_income : ℕ := john_income + ingrid_income + alice_income + ben_income

def combined_tax_rate : ℚ := total_tax / total_income

theorem combined_tax_rate_calculation : 
  combined_tax_rate = total_tax / total_income :=
by sorry

end NUMINAMATH_CALUDE_combined_tax_rate_calculation_l4120_412050


namespace NUMINAMATH_CALUDE_leila_payment_l4120_412006

/-- The total cost of Leila's cake order --/
def total_cost (chocolate_cakes strawberry_cakes : ℕ) 
               (chocolate_price strawberry_price : ℚ) : ℚ :=
  chocolate_cakes * chocolate_price + strawberry_cakes * strawberry_price

/-- Theorem stating that Leila should pay $168 for her cake order --/
theorem leila_payment : 
  total_cost 3 6 12 22 = 168 := by sorry

end NUMINAMATH_CALUDE_leila_payment_l4120_412006


namespace NUMINAMATH_CALUDE_intersection_point_of_function_and_inverse_l4120_412052

theorem intersection_point_of_function_and_inverse (b a : ℤ) : 
  let f : ℝ → ℝ := λ x ↦ -2 * x + b
  let f_inv : ℝ → ℝ := Function.invFun f
  (∀ x, f (f_inv x) = x) ∧ (∀ x, f_inv (f x) = x) ∧ f 2 = a ∧ f_inv 2 = a
  → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_function_and_inverse_l4120_412052


namespace NUMINAMATH_CALUDE_contribution_is_180_l4120_412043

/-- Calculates the individual contribution for painting a wall --/
def calculate_contribution (paint_cost_per_gallon : ℚ) (coverage_per_gallon : ℚ) (total_area : ℚ) (num_coats : ℕ) : ℚ :=
  let total_gallons := (total_area / coverage_per_gallon) * num_coats
  let total_cost := total_gallons * paint_cost_per_gallon
  total_cost / 2

/-- Proves that each person's contribution is $180 --/
theorem contribution_is_180 :
  calculate_contribution 45 400 1600 2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_contribution_is_180_l4120_412043


namespace NUMINAMATH_CALUDE_system_solution_sum_of_squares_l4120_412012

theorem system_solution_sum_of_squares (x y : ℝ) : 
  x * y = 6 → x^2 * y + x * y^2 + x + y = 63 → x^2 + y^2 = 69 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_sum_of_squares_l4120_412012


namespace NUMINAMATH_CALUDE_urn_problem_solution_l4120_412051

/-- The number of blue balls in the second urn -/
def N : ℕ := 144

/-- The probability of drawing two balls of the same color -/
def same_color_probability : ℚ := 29/50

theorem urn_problem_solution :
  let urn1_green : ℕ := 4
  let urn1_blue : ℕ := 6
  let urn2_green : ℕ := 16
  let urn1_total : ℕ := urn1_green + urn1_blue
  let urn2_total : ℕ := urn2_green + N
  let same_green : ℕ := urn1_green * urn2_green
  let same_blue : ℕ := urn1_blue * N
  let total_outcomes : ℕ := urn1_total * urn2_total
  (same_green + same_blue : ℚ) / total_outcomes = same_color_probability :=
by sorry

end NUMINAMATH_CALUDE_urn_problem_solution_l4120_412051


namespace NUMINAMATH_CALUDE_mikey_new_leaves_l4120_412040

/-- The number of new leaves that came to Mikey -/
def new_leaves (initial final : ℝ) : ℝ := final - initial

/-- Proof that Mikey received 112 new leaves -/
theorem mikey_new_leaves :
  let initial : ℝ := 356.0
  let final : ℝ := 468
  new_leaves initial final = 112 := by sorry

end NUMINAMATH_CALUDE_mikey_new_leaves_l4120_412040


namespace NUMINAMATH_CALUDE_lucky_money_distribution_l4120_412080

/-- Represents a distribution of lucky money among three grandsons -/
structure LuckyMoneyDistribution where
  grandson1 : ℕ
  grandson2 : ℕ
  grandson3 : ℕ

/-- Checks if a distribution satisfies the given conditions -/
def isValidDistribution (d : LuckyMoneyDistribution) : Prop :=
  ∃ (x y z : ℕ),
    (d.grandson1 = 10 * x ∧ d.grandson2 = 20 * y ∧ d.grandson3 = 50 * z) ∧
    (x = y * z) ∧
    (d.grandson1 + d.grandson2 + d.grandson3 = 300)

/-- The theorem stating the only valid distributions -/
theorem lucky_money_distribution :
  ∀ d : LuckyMoneyDistribution,
    isValidDistribution d →
    (d = ⟨100, 100, 100⟩ ∨ d = ⟨90, 60, 150⟩) :=
by sorry


end NUMINAMATH_CALUDE_lucky_money_distribution_l4120_412080


namespace NUMINAMATH_CALUDE_circle_radius_given_area_and_circumference_sum_l4120_412073

theorem circle_radius_given_area_and_circumference_sum (x y : ℝ) :
  x ≥ 0 →
  y > 0 →
  x = π * (y / (2 * π))^2 →
  x + y = 90 * π →
  y / (2 * π) = 10 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_given_area_and_circumference_sum_l4120_412073


namespace NUMINAMATH_CALUDE_rectangle_width_l4120_412049

theorem rectangle_width (w : ℝ) (h1 : w > 0) (h2 : 4 * w * w = 100) : w = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l4120_412049


namespace NUMINAMATH_CALUDE_motorboat_travel_time_l4120_412054

/-- Represents the scenario of a motorboat and kayak traveling on a river -/
structure RiverTravel where
  r : ℝ  -- Speed of the river current (and kayak's speed)
  m : ℝ  -- Speed of the motorboat relative to the river
  t : ℝ  -- Time for motorboat to travel from X to Y
  total_time : ℝ  -- Total time until motorboat meets kayak

/-- The theorem representing the problem -/
theorem motorboat_travel_time (rt : RiverTravel) : 
  rt.m = rt.r ∧ rt.total_time = 8 → rt.t = 4 := by
  sorry

#check motorboat_travel_time

end NUMINAMATH_CALUDE_motorboat_travel_time_l4120_412054


namespace NUMINAMATH_CALUDE_problem_statement_l4120_412024

theorem problem_statement (a : ℝ) (h : a/2 - 2/a = 5) : 
  (a^8 - 256) / (16 * a^4) * (2 * a) / (a^2 + 4) = 81 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4120_412024


namespace NUMINAMATH_CALUDE_car_average_speed_l4120_412016

/-- The average speed of a car traveling 60 km in the first hour and 30 km in the second hour is 45 km/h. -/
theorem car_average_speed : 
  let speed1 : ℝ := 60 -- Speed in the first hour (km/h)
  let speed2 : ℝ := 30 -- Speed in the second hour (km/h)
  let time : ℝ := 2 -- Total time (hours)
  let total_distance : ℝ := speed1 + speed2 -- Total distance (km)
  let average_speed : ℝ := total_distance / time -- Average speed (km/h)
  average_speed = 45 := by
  sorry

end NUMINAMATH_CALUDE_car_average_speed_l4120_412016


namespace NUMINAMATH_CALUDE_circle_point_x_coordinate_l4120_412065

theorem circle_point_x_coordinate :
  ∀ x : ℝ,
  let circle_center : ℝ × ℝ := (7, 0)
  let circle_radius : ℝ := 14
  let point_on_circle : ℝ × ℝ := (x, 10)
  (point_on_circle.1 - circle_center.1)^2 + (point_on_circle.2 - circle_center.2)^2 = circle_radius^2 →
  x = 7 + 4 * Real.sqrt 6 ∨ x = 7 - 4 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_point_x_coordinate_l4120_412065


namespace NUMINAMATH_CALUDE_average_marks_chem_math_l4120_412025

/-- Given that the total marks in physics, chemistry, and mathematics is 140 more than
    the marks in physics, prove that the average mark in chemistry and mathematics is 70. -/
theorem average_marks_chem_math (P C M : ℕ) (h : P + C + M = P + 140) :
  (C + M) / 2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_chem_math_l4120_412025


namespace NUMINAMATH_CALUDE_quadratic_m_bounds_l4120_412048

open Complex

/-- Given a quadratic equation x^2 + z₁x + z₂ + m = 0 with complex coefficients,
    prove that under certain conditions, |m| has specific min and max values. -/
theorem quadratic_m_bounds (z₁ z₂ m : ℂ) (α β : ℂ) :
  z₁^2 - 4*z₂ = 16 + 20*I →
  α^2 + z₁*α + z₂ + m = 0 →
  β^2 + z₁*β + z₂ + m = 0 →
  abs (α - β) = 2 * Real.sqrt 7 →
  (abs m = Real.sqrt 41 - 7 ∨ abs m = Real.sqrt 41 + 7) ∧
  ∀ m' : ℂ, (∃ α' β' : ℂ, α'^2 + z₁*α' + z₂ + m' = 0 ∧
                          β'^2 + z₁*β' + z₂ + m' = 0 ∧
                          abs (α' - β') = 2 * Real.sqrt 7) →
    Real.sqrt 41 - 7 ≤ abs m' ∧ abs m' ≤ Real.sqrt 41 + 7 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_m_bounds_l4120_412048


namespace NUMINAMATH_CALUDE_second_solution_concentration_l4120_412038

/-- Represents an alcohol solution --/
structure AlcoholSolution where
  volume : ℝ
  concentration : ℝ

/-- Represents a mixture of two alcohol solutions --/
structure AlcoholMixture where
  solution1 : AlcoholSolution
  solution2 : AlcoholSolution
  final : AlcoholSolution

/-- The alcohol mixture satisfies the given conditions --/
def satisfies_conditions (mixture : AlcoholMixture) : Prop :=
  mixture.final.volume = 200 ∧
  mixture.final.concentration = 0.15 ∧
  mixture.solution1.volume = 75 ∧
  mixture.solution1.concentration = 0.20 ∧
  mixture.solution2.volume = mixture.final.volume - mixture.solution1.volume

theorem second_solution_concentration
  (mixture : AlcoholMixture)
  (h : satisfies_conditions mixture) :
  mixture.solution2.concentration = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_second_solution_concentration_l4120_412038


namespace NUMINAMATH_CALUDE_fraction_square_equals_49_l4120_412090

theorem fraction_square_equals_49 : (3072 - 2993)^2 / 121 = 49 := by sorry

end NUMINAMATH_CALUDE_fraction_square_equals_49_l4120_412090


namespace NUMINAMATH_CALUDE_triangle_determinant_zero_l4120_412018

theorem triangle_determinant_zero (A B C : ℝ) (h : A + B + C = π) : 
  let matrix : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![Real.sin A ^ 2, Real.cos A / Real.sin A, Real.sin A],
    ![Real.sin B ^ 2, Real.cos B / Real.sin B, Real.sin B],
    ![Real.sin C ^ 2, Real.cos C / Real.sin C, Real.sin C]
  ]
  Matrix.det matrix = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_determinant_zero_l4120_412018


namespace NUMINAMATH_CALUDE_expression_evaluation_l4120_412070

theorem expression_evaluation (x y : ℝ) (h : x * y ≠ 0) :
  ((x^4 + 1) / x) * ((y^4 + 1) / y) + ((x^4 - 1) / y) * ((y^4 - 1) / x) = 2 * x^3 * y^3 + 2 / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4120_412070


namespace NUMINAMATH_CALUDE_new_train_distance_l4120_412058

theorem new_train_distance (old_distance : ℝ) (increase_percentage : ℝ) (new_distance : ℝ) : 
  old_distance = 180 → 
  increase_percentage = 0.5 → 
  new_distance = old_distance * (1 + increase_percentage) → 
  new_distance = 270 := by
sorry

end NUMINAMATH_CALUDE_new_train_distance_l4120_412058


namespace NUMINAMATH_CALUDE_min_difference_theorem_l4120_412060

noncomputable def f (x : ℝ) : ℝ := Real.exp (4 * x - 1)

noncomputable def g (x : ℝ) : ℝ := 1 / 2 + Real.log (2 * x)

theorem min_difference_theorem (m n : ℝ) (h : f m = g n) :
  (∀ m' n', f m' = g n' → n' - m' ≥ (1 + Real.log 2) / 4) ∧
  (∃ m₀ n₀, f m₀ = g n₀ ∧ n₀ - m₀ = (1 + Real.log 2) / 4) := by
  sorry

end NUMINAMATH_CALUDE_min_difference_theorem_l4120_412060


namespace NUMINAMATH_CALUDE_gift_cost_increase_l4120_412021

theorem gift_cost_increase (initial_friends : ℕ) (gift_cost : ℕ) (dropouts : ℕ) : 
  initial_friends = 10 → 
  gift_cost = 120 → 
  dropouts = 4 → 
  (gift_cost / (initial_friends - dropouts) : ℚ) - (gift_cost / initial_friends : ℚ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_gift_cost_increase_l4120_412021


namespace NUMINAMATH_CALUDE_y_derivative_l4120_412037

noncomputable def y (x : ℝ) : ℝ := Real.tan (Real.sqrt (Real.cos (1/3))) + (Real.sin (31*x))^2 / (31 * Real.cos (62*x))

theorem y_derivative (x : ℝ) :
  deriv y x = (2 * (Real.sin (31*x) * Real.cos (31*x) * Real.cos (62*x) + Real.sin (31*x)^2 * Real.sin (62*x))) / Real.cos (62*x)^2 :=
sorry

end NUMINAMATH_CALUDE_y_derivative_l4120_412037


namespace NUMINAMATH_CALUDE_edward_games_from_friend_l4120_412084

/-- The number of games Edward bought from his friend -/
def games_from_friend : ℕ := sorry

/-- The number of games Edward bought at the garage sale -/
def games_from_garage_sale : ℕ := 14

/-- The number of games that didn't work -/
def non_working_games : ℕ := 31

/-- The number of good games Edward ended up with -/
def good_games : ℕ := 24

theorem edward_games_from_friend :
  games_from_friend = 41 :=
by
  have h1 : games_from_friend + games_from_garage_sale - non_working_games = good_games := by sorry
  sorry

end NUMINAMATH_CALUDE_edward_games_from_friend_l4120_412084


namespace NUMINAMATH_CALUDE_function_equation_solution_l4120_412030

/-- Given a function f : ℝ → ℝ satisfying the equation
    f(x) + f(2x+y) + 7xy = f(3x - 2y) + 3x^2 + 2
    for all real numbers x and y, prove that f(15) = 1202 -/
theorem function_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f x + f (2*x + y) + 7*x*y = f (3*x - 2*y) + 3*x^2 + 2) : 
  f 15 = 1202 := by
  sorry

end NUMINAMATH_CALUDE_function_equation_solution_l4120_412030


namespace NUMINAMATH_CALUDE_marie_sold_925_reading_materials_l4120_412062

/-- The total number of reading materials Marie sold -/
def total_reading_materials (magazines newspapers books pamphlets : ℕ) : ℕ :=
  magazines + newspapers + books + pamphlets

/-- Theorem stating that Marie sold 925 reading materials -/
theorem marie_sold_925_reading_materials :
  total_reading_materials 425 275 150 75 = 925 := by
  sorry

end NUMINAMATH_CALUDE_marie_sold_925_reading_materials_l4120_412062


namespace NUMINAMATH_CALUDE_min_value_of_f_on_interval_l4120_412072

-- Define the function f
def f (x : ℝ) : ℝ := x * (3 - x^2)

-- Define the interval
def interval : Set ℝ := { x | 0 ≤ x ∧ x ≤ Real.sqrt 2 }

-- State the theorem
theorem min_value_of_f_on_interval :
  ∃ (m : ℝ), m = 0 ∧ ∀ x ∈ interval, f x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_on_interval_l4120_412072


namespace NUMINAMATH_CALUDE_digit_2009_is_zero_l4120_412027

/-- The function that returns the nth digit in the sequence formed by 
    writing successive natural numbers without spaces -/
def nthDigit (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that the 2009th digit in the sequence is 0 -/
theorem digit_2009_is_zero : nthDigit 2009 = 0 := by
  sorry

end NUMINAMATH_CALUDE_digit_2009_is_zero_l4120_412027


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l4120_412061

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x^2 - x < 0) → (-1 < x ∧ x < 1) ∧
  ∃ y : ℝ, -1 < y ∧ y < 1 ∧ ¬(y^2 - y < 0) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l4120_412061


namespace NUMINAMATH_CALUDE_parallelogram_base_l4120_412066

/-- Given a parallelogram with area 78.88 cm² and height 8 cm, its base is 9.86 cm -/
theorem parallelogram_base (area : ℝ) (height : ℝ) (base : ℝ) : 
  area = 78.88 ∧ height = 8 ∧ area = base * height → base = 9.86 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_l4120_412066


namespace NUMINAMATH_CALUDE_line_parallel_transitive_plane_parallel_transitive_l4120_412034

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines
variable (parallelLine : Line → Line → Prop)

-- Define the parallel relation for planes
variable (parallelPlane : Plane → Plane → Prop)

-- Theorem for lines
theorem line_parallel_transitive (a b c : Line) :
  parallelLine a b → parallelLine a c → parallelLine b c := by sorry

-- Theorem for planes
theorem plane_parallel_transitive (α β γ : Plane) :
  parallelPlane α β → parallelPlane α γ → parallelPlane β γ := by sorry

end NUMINAMATH_CALUDE_line_parallel_transitive_plane_parallel_transitive_l4120_412034


namespace NUMINAMATH_CALUDE_smallest_number_of_blocks_l4120_412008

/-- Represents the dimensions of a wall --/
structure WallDimensions where
  length : Nat
  height : Nat

/-- Represents the dimensions of a block --/
structure BlockDimensions where
  height : Nat
  possibleLengths : List Nat

/-- Represents the constraints for building the wall --/
structure WallConstraints where
  noCutting : Bool
  staggeredJoints : Bool
  evenEnds : Bool

/-- Calculates the smallest number of blocks needed to build the wall --/
def minBlocksNeeded (wall : WallDimensions) (block : BlockDimensions) (constraints : WallConstraints) : Nat :=
  sorry

/-- Theorem stating the smallest number of blocks needed for the given wall --/
theorem smallest_number_of_blocks
  (wall : WallDimensions)
  (block : BlockDimensions)
  (constraints : WallConstraints)
  (h_wall_length : wall.length = 120)
  (h_wall_height : wall.height = 7)
  (h_block_height : block.height = 1)
  (h_block_lengths : block.possibleLengths = [2, 3])
  (h_no_cutting : constraints.noCutting = true)
  (h_staggered : constraints.staggeredJoints = true)
  (h_even_ends : constraints.evenEnds = true) :
  minBlocksNeeded wall block constraints = 357 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_of_blocks_l4120_412008


namespace NUMINAMATH_CALUDE_entertainment_budget_percentage_l4120_412086

/-- Proves that given a budget of $1000, with 30% spent on food, 15% on accommodation,
    $300 on coursework materials, the remaining percentage spent on entertainment is 25%. -/
theorem entertainment_budget_percentage
  (total_budget : ℝ)
  (food_percentage : ℝ)
  (accommodation_percentage : ℝ)
  (coursework_materials : ℝ)
  (h1 : total_budget = 1000)
  (h2 : food_percentage = 30)
  (h3 : accommodation_percentage = 15)
  (h4 : coursework_materials = 300) :
  (total_budget - (food_percentage / 100 * total_budget + 
   accommodation_percentage / 100 * total_budget + coursework_materials)) / 
   total_budget * 100 = 25 := by
  sorry

#check entertainment_budget_percentage

end NUMINAMATH_CALUDE_entertainment_budget_percentage_l4120_412086


namespace NUMINAMATH_CALUDE_grocery_payment_possible_l4120_412028

def soup_price : ℕ := 2
def bread_price : ℕ := 5
def cereal_price : ℕ := 3
def milk_price : ℕ := 4

def soup_quantity : ℕ := 6
def bread_quantity : ℕ := 2
def cereal_quantity : ℕ := 2
def milk_quantity : ℕ := 2

def total_cost : ℕ := 
  soup_price * soup_quantity + 
  bread_price * bread_quantity + 
  cereal_price * cereal_quantity + 
  milk_price * milk_quantity

def us_bill_denominations : List ℕ := [1, 2, 5, 10, 20, 50, 100]

theorem grocery_payment_possible :
  ∃ (a b c d : ℕ), 
    a ∈ us_bill_denominations ∧ 
    b ∈ us_bill_denominations ∧ 
    c ∈ us_bill_denominations ∧ 
    d ∈ us_bill_denominations ∧ 
    a + b + c + d = total_cost :=
sorry

end NUMINAMATH_CALUDE_grocery_payment_possible_l4120_412028


namespace NUMINAMATH_CALUDE_product_ab_equals_twelve_l4120_412067

-- Define the set A
def A (a b : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ b}

-- Define the complement of A with respect to ℝ
def complement_A : Set ℝ := {x | x < 3 ∨ x > 4}

-- Theorem statement
theorem product_ab_equals_twelve (a b : ℝ) : 
  A a b ∪ complement_A = Set.univ → a * b = 12 := by
  sorry

end NUMINAMATH_CALUDE_product_ab_equals_twelve_l4120_412067


namespace NUMINAMATH_CALUDE_triangle_inequality_and_sqrt_sides_l4120_412056

/-- Given a triangle with side lengths a, b, c, prove the existence of a triangle
    with side lengths √a, √b, √c and the inequality involving these lengths. -/
theorem triangle_inequality_and_sqrt_sides {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≤ b + c) (hbc : b ≤ a + c) (hca : c ≤ a + b) :
  (∃ (u v w : ℝ), u > 0 ∧ v > 0 ∧ w > 0 ∧ 
    a = v + w ∧ b = u + w ∧ c = u + v) ∧
  (Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) ≤ a + b + c) ∧
  (a + b + c ≤ 2 * Real.sqrt (a * b) + 2 * Real.sqrt (b * c) + 2 * Real.sqrt (c * a)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_and_sqrt_sides_l4120_412056


namespace NUMINAMATH_CALUDE_sculpture_cost_in_cny_l4120_412071

/-- Exchange rate from US dollars to Namibian dollars -/
def usd_to_nad : ℝ := 8

/-- Exchange rate from US dollars to Chinese yuan -/
def usd_to_cny : ℝ := 5

/-- Cost of the sculpture in Namibian dollars -/
def sculpture_cost_nad : ℝ := 160

/-- Theorem stating the equivalent cost of the sculpture in Chinese yuan -/
theorem sculpture_cost_in_cny :
  (sculpture_cost_nad / usd_to_nad) * usd_to_cny = 100 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_cost_in_cny_l4120_412071


namespace NUMINAMATH_CALUDE_inverse_function_problem_l4120_412083

theorem inverse_function_problem (f : ℝ → ℝ) (hf : Function.Bijective f) :
  f 6 = 5 → f 5 = 1 → f 1 = 4 →
  (Function.invFun f) ((Function.invFun f 5) * (Function.invFun f 4)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_problem_l4120_412083


namespace NUMINAMATH_CALUDE_seven_consecutive_integers_product_divisible_by_ten_l4120_412076

theorem seven_consecutive_integers_product_divisible_by_ten (n : ℕ+) :
  ∃ k : ℕ, (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6) : ℕ) = 10 * k := by
  sorry

end NUMINAMATH_CALUDE_seven_consecutive_integers_product_divisible_by_ten_l4120_412076


namespace NUMINAMATH_CALUDE_supplement_of_complement_of_36_l4120_412098

-- Define the original angle
def original_angle : ℝ := 36

-- Define the complement of an angle
def complement (angle : ℝ) : ℝ := 90 - angle

-- Define the supplement of an angle
def supplement (angle : ℝ) : ℝ := 180 - angle

-- Theorem statement
theorem supplement_of_complement_of_36 : 
  supplement (complement original_angle) = 126 := by
  sorry

end NUMINAMATH_CALUDE_supplement_of_complement_of_36_l4120_412098


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l4120_412000

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 6*y - x*y = 0) :
  ∀ a b : ℝ, a > 0 → b > 0 → 2*a + 6*b - a*b = 0 → x + y ≤ a + b ∧ x + y = 8 + 4*Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l4120_412000


namespace NUMINAMATH_CALUDE_edward_money_left_l4120_412096

/-- The amount of money Edward initially had to spend --/
def initial_amount : ℚ := 1780 / 100

/-- The cost of one toy car before discount --/
def toy_car_cost : ℚ := 95 / 100

/-- The number of toy cars Edward bought --/
def num_toy_cars : ℕ := 4

/-- The discount rate on toy cars --/
def toy_car_discount_rate : ℚ := 15 / 100

/-- The cost of the race track before tax --/
def race_track_cost : ℚ := 600 / 100

/-- The tax rate on the race track --/
def race_track_tax_rate : ℚ := 8 / 100

/-- The theorem stating how much money Edward has left --/
theorem edward_money_left : 
  initial_amount - 
  (num_toy_cars * toy_car_cost * (1 - toy_car_discount_rate) + 
   race_track_cost * (1 + race_track_tax_rate)) = 809 / 100 := by
  sorry

end NUMINAMATH_CALUDE_edward_money_left_l4120_412096


namespace NUMINAMATH_CALUDE_kaleb_sold_games_l4120_412089

theorem kaleb_sold_games (initial_games : ℕ) (games_per_box : ℕ) (boxes_used : ℕ) 
  (h1 : initial_games = 76)
  (h2 : games_per_box = 5)
  (h3 : boxes_used = 6) :
  initial_games - (games_per_box * boxes_used) = 46 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_sold_games_l4120_412089


namespace NUMINAMATH_CALUDE_puzzle_solution_l4120_412074

theorem puzzle_solution (p q r s t : ℕ+) 
  (eq1 : p * q + p + q = 322)
  (eq2 : q * r + q + r = 186)
  (eq3 : r * s + r + s = 154)
  (eq4 : s * t + s + t = 272)
  (product : p * q * r * s * t = 3628800) : -- 3628800 is 10!
  p - t = 6 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_solution_l4120_412074


namespace NUMINAMATH_CALUDE_sequence_sum_l4120_412044

/-- Given a sequence {a_n} where a_1 = 1 and S_n = n^2 * a_n for all positive integers n,
    prove that S_n = 2n / (n+1) for all positive integers n. -/
theorem sequence_sum (a : ℕ → ℚ) (S : ℕ → ℚ) :
  a 1 = 1 →
  (∀ n : ℕ, n > 0 → S n = n^2 * a n) →
  ∀ n : ℕ, n > 0 → S n = 2 * n / (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l4120_412044


namespace NUMINAMATH_CALUDE_team_win_percentage_l4120_412022

theorem team_win_percentage (games_won : ℕ) (games_lost : ℕ) 
  (h : games_won / games_lost = 13 / 7) : 
  (games_won : ℚ) / (games_won + games_lost) * 100 = 65 := by
  sorry

end NUMINAMATH_CALUDE_team_win_percentage_l4120_412022


namespace NUMINAMATH_CALUDE_taxi_fare_calculation_l4120_412082

/-- Represents the taxi fare structure -/
structure TaxiFare where
  startupFee : ℝ
  ratePerMile : ℝ

/-- Calculates the total fare for a given distance -/
def totalFare (tf : TaxiFare) (distance : ℝ) : ℝ :=
  tf.startupFee + tf.ratePerMile * distance

theorem taxi_fare_calculation (tf : TaxiFare) :
  tf.startupFee = 30 ∧ totalFare tf 60 = 150 → totalFare tf 90 = 210 := by
  sorry

end NUMINAMATH_CALUDE_taxi_fare_calculation_l4120_412082


namespace NUMINAMATH_CALUDE_scout_sunday_deliveries_l4120_412093

def base_pay : ℝ := 10
def tip_per_customer : ℝ := 5
def saturday_hours : ℝ := 4
def saturday_customers : ℝ := 5
def sunday_hours : ℝ := 5
def total_earnings : ℝ := 155

theorem scout_sunday_deliveries :
  ∃ (sunday_customers : ℝ),
    base_pay * (saturday_hours + sunday_hours) +
    tip_per_customer * (saturday_customers + sunday_customers) = total_earnings ∧
    sunday_customers = 8 := by
  sorry

end NUMINAMATH_CALUDE_scout_sunday_deliveries_l4120_412093


namespace NUMINAMATH_CALUDE_matrix_is_own_inverse_l4120_412039

/-- A matrix is its own inverse if and only if its square is the identity matrix. -/
theorem matrix_is_own_inverse (c d : ℚ) : 
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![4, -2; c, d]
  A * A = 1 ↔ c = 15/2 ∧ d = -4 := by
  sorry

end NUMINAMATH_CALUDE_matrix_is_own_inverse_l4120_412039


namespace NUMINAMATH_CALUDE_inscribed_circle_square_area_l4120_412009

theorem inscribed_circle_square_area (s : ℝ) (r : ℝ) : 
  r > 0 → s = 2 * r → r^2 * Real.pi = 9 * Real.pi → s^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_square_area_l4120_412009


namespace NUMINAMATH_CALUDE_polynomial_remainder_l4120_412033

/-- Given a polynomial q(x) = Dx^4 + Ex^2 + Fx + 6, if the remainder when q(x) is divided by (x - 2) is 14, 
    then the remainder when q(x) is divided by (x + 2) is also 14 -/
theorem polynomial_remainder (D E F : ℝ) : 
  let q : ℝ → ℝ := λ x => D * x^4 + E * x^2 + F * x + 6
  (q 2 = 14) → (q (-2) = 14) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l4120_412033


namespace NUMINAMATH_CALUDE_elevator_weight_problem_l4120_412046

/-- Given 6 people in an elevator with an average weight of 154 lbs, 
    if a 7th person enters and the new average weight becomes 151 lbs, 
    then the weight of the 7th person is 133 lbs. -/
theorem elevator_weight_problem (initial_people : ℕ) (initial_avg_weight : ℝ) 
    (final_people : ℕ) (final_avg_weight : ℝ) : 
    initial_people = 6 → 
    initial_avg_weight = 154 → 
    final_people = 7 → 
    final_avg_weight = 151 → 
    (initial_people * initial_avg_weight + 
      (final_people - initial_people) * 
      ((final_people * final_avg_weight) - (initial_people * initial_avg_weight))) / 
      (final_people - initial_people) = 133 := by
  sorry

end NUMINAMATH_CALUDE_elevator_weight_problem_l4120_412046


namespace NUMINAMATH_CALUDE_triangle_side_length_l4120_412055

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  -- Triangle ABC is oblique (implied by the other conditions)
  -- Side lengths opposite to angles A, B, C are a, b, c respectively
  A = π / 4 →
  Real.sin A + Real.sin (B - C) = 2 * Real.sqrt 2 * Real.sin (2 * C) →
  (1 / 2) * b * c * Real.sin A = 1 →
  a = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4120_412055


namespace NUMINAMATH_CALUDE_range_of_function_l4120_412031

open Real

theorem range_of_function (x : ℝ) (h : 0 < x ∧ x < π/2) :
  let y := sin x - 2 * cos x + 32 / (125 * sin x * (1 - cos x))
  ∀ z, y ≥ z → z ≥ 2/5 := by sorry

end NUMINAMATH_CALUDE_range_of_function_l4120_412031


namespace NUMINAMATH_CALUDE_marks_candies_l4120_412078

-- Define the number of people
def num_people : ℕ := 3

-- Define the number of candies each person will have after sharing
def shared_candies : ℕ := 30

-- Define Peter's candies
def peter_candies : ℕ := 25

-- Define John's candies
def john_candies : ℕ := 35

-- Theorem to prove Mark's candies
theorem marks_candies :
  shared_candies * num_people - (peter_candies + john_candies) = 30 := by
  sorry


end NUMINAMATH_CALUDE_marks_candies_l4120_412078


namespace NUMINAMATH_CALUDE_algebraic_simplification_l4120_412088

theorem algebraic_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (2 / a) * (a * b / 4) = b / 2 ∧
  -6 * a * b / ((3 * b^2) / (2 * a)) = -4 * a^2 / b := by sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l4120_412088
