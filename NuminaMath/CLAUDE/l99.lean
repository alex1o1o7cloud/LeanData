import Mathlib

namespace sqrt_difference_equality_l99_9991

theorem sqrt_difference_equality (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (1 / Real.sqrt (2011 + Real.sqrt (2011^2 - 1)) : ℝ) = Real.sqrt m - Real.sqrt n →
  m + n = 2011 := by
  sorry

end sqrt_difference_equality_l99_9991


namespace mod_equivalence_2023_l99_9945

theorem mod_equivalence_2023 : ∃! n : ℕ, n ≤ 11 ∧ n ≡ -2023 [ZMOD 12] ∧ n = 9 := by
  sorry

end mod_equivalence_2023_l99_9945


namespace min_value_of_function_l99_9946

theorem min_value_of_function (x : ℝ) (h : x > 0) : 
  8 + x/2 + 2/x ≥ 10 ∧ ∃ y > 0, 8 + y/2 + 2/y = 10 := by
  sorry

end min_value_of_function_l99_9946


namespace smallest_prime_divisor_cube_sum_l99_9948

theorem smallest_prime_divisor_cube_sum (n : ℕ) : n ≥ 2 → (∃ (a d : ℕ), Prime a ∧ d > 0 ∧ d ∣ n ∧ (∀ p : ℕ, Prime p → p ∣ n → a ≤ p) ∧ n = a^3 + d^3) → (n = 16 ∨ n = 72 ∨ n = 520) :=
by sorry

end smallest_prime_divisor_cube_sum_l99_9948


namespace floor_product_equation_l99_9914

theorem floor_product_equation : ∃! (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) * x = 70 ∧ x = (70 : ℝ) / 8 := by sorry

end floor_product_equation_l99_9914


namespace coal_piles_weights_l99_9983

theorem coal_piles_weights (pile1 pile2 : ℕ) : 
  pile1 = pile2 + 80 →
  pile1 * 80 / 100 = pile2 - 50 →
  pile1 = 650 ∧ pile2 = 570 := by
sorry

end coal_piles_weights_l99_9983


namespace clock_strike_time_l99_9900

/-- If a clock takes 42 seconds to strike 7 times, it takes 60 seconds to strike 10 times. -/
theorem clock_strike_time (strike_time : ℕ → ℝ) 
  (h : strike_time 7 = 42) : strike_time 10 = 60 := by
  sorry

end clock_strike_time_l99_9900


namespace restaurant_check_amount_l99_9941

theorem restaurant_check_amount
  (tax_rate : Real)
  (total_payment : Real)
  (tip_amount : Real)
  (h1 : tax_rate = 0.20)
  (h2 : total_payment = 20)
  (h3 : tip_amount = 2) :
  ∃ (original_amount : Real),
    original_amount * (1 + tax_rate) = total_payment - tip_amount ∧
    original_amount = 15 := by
  sorry

end restaurant_check_amount_l99_9941


namespace infinite_non_representable_l99_9996

/-- A natural number is representable if it can be written as p + n^(2k) for some prime p and natural numbers n and k. -/
def Representable (m : ℕ) : Prop :=
  ∃ (p n k : ℕ), Prime p ∧ m = p + n^(2*k)

/-- The set of non-representable natural numbers is infinite. -/
theorem infinite_non_representable :
  {m : ℕ | ¬Representable m}.Infinite :=
sorry

end infinite_non_representable_l99_9996


namespace lavinia_son_older_than_daughter_l99_9997

/-- Given information about the ages of Lavinia's and Katie's children, prove that Lavinia's son is 21 years older than Lavinia's daughter. -/
theorem lavinia_son_older_than_daughter :
  ∀ (lavinia_daughter lavinia_son katie_daughter katie_son : ℕ),
  lavinia_daughter = katie_daughter / 3 →
  lavinia_son = 2 * katie_daughter →
  lavinia_daughter + lavinia_son = 2 * katie_daughter + 5 →
  katie_daughter = 12 →
  katie_son + 3 = lavinia_son →
  lavinia_son - lavinia_daughter = 21 :=
by sorry

end lavinia_son_older_than_daughter_l99_9997


namespace unique_real_root_of_system_l99_9910

theorem unique_real_root_of_system : 
  ∃! x : ℝ, x^3 + 9 = 0 ∧ x + 3 = 0 :=
by
  -- The proof goes here
  sorry

end unique_real_root_of_system_l99_9910


namespace root_implies_h_value_l99_9992

theorem root_implies_h_value (h : ℝ) : 
  (3 : ℝ)^3 + h * 3 + 5 = 0 → h = -32/3 := by
  sorry

end root_implies_h_value_l99_9992


namespace no_four_distinct_numbers_l99_9928

theorem no_four_distinct_numbers : 
  ¬ ∃ (a b c d : ℝ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧ 
    (a^11 - a = b^11 - b) ∧ 
    (a^11 - a = c^11 - c) ∧ 
    (a^11 - a = d^11 - d) := by
  sorry

end no_four_distinct_numbers_l99_9928


namespace biff_break_even_hours_l99_9990

/-- Calculates the number of hours required to break even on a bus trip. -/
def hours_to_break_even (ticket_cost snacks_cost headphones_cost hourly_rate wifi_cost : ℚ) : ℚ :=
  let total_expenses := ticket_cost + snacks_cost + headphones_cost
  let net_hourly_rate := hourly_rate - wifi_cost
  total_expenses / net_hourly_rate

/-- Proves that given Biff's expenses and earnings, the number of hours required to break even on a bus trip is 3 hours. -/
theorem biff_break_even_hours :
  hours_to_break_even 11 3 16 12 2 = 3 := by
  sorry

end biff_break_even_hours_l99_9990


namespace quadrilateral_pyramid_ratio_l99_9995

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a quadrilateral pyramid -/
structure QuadrilateralPyramid where
  P : Point3D
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Checks if two line segments are parallel -/
def areParallel (p1 p2 p3 p4 : Point3D) : Prop := sorry

/-- Checks if two line segments are perpendicular -/
def arePerpendicular (p1 p2 p3 p4 : Point3D) : Prop := sorry

/-- Checks if a line is perpendicular to a plane -/
def isPerpendicularToPlane (p1 p2 : Point3D) (plane : Plane3D) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point3D) : ℝ := sorry

/-- Calculates the sine of the angle between a line and a plane -/
def sineAngleLinePlane (p1 p2 : Point3D) (plane : Plane3D) : ℝ := sorry

/-- Main theorem -/
theorem quadrilateral_pyramid_ratio 
  (pyramid : QuadrilateralPyramid) 
  (Q : Point3D)
  (h1 : areParallel pyramid.A pyramid.B pyramid.C pyramid.D)
  (h2 : arePerpendicular pyramid.A pyramid.B pyramid.A pyramid.D)
  (h3 : distance pyramid.A pyramid.B = 4)
  (h4 : distance pyramid.A pyramid.D = 2 * Real.sqrt 2)
  (h5 : distance pyramid.C pyramid.D = 2)
  (h6 : isPerpendicularToPlane pyramid.P pyramid.A (Plane3D.mk 0 0 1 0))  -- Assuming ABCD is on the xy-plane
  (h7 : distance pyramid.P pyramid.A = 4)
  (h8 : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q.x = pyramid.P.x + t * (pyramid.B.x - pyramid.P.x) ∧
                              Q.y = pyramid.P.y + t * (pyramid.B.y - pyramid.P.y) ∧
                              Q.z = pyramid.P.z + t * (pyramid.B.z - pyramid.P.z))
  (h9 : sineAngleLinePlane Q pyramid.C (Plane3D.mk 1 0 0 0) = Real.sqrt 3 / 3)  -- Assuming PAC is on the yz-plane
  : ∃ (t : ℝ), distance pyramid.P Q / distance pyramid.P pyramid.B = 7/12 ∧ 
               Q.x = pyramid.P.x + t * (pyramid.B.x - pyramid.P.x) ∧
               Q.y = pyramid.P.y + t * (pyramid.B.y - pyramid.P.y) ∧
               Q.z = pyramid.P.z + t * (pyramid.B.z - pyramid.P.z) := by
  sorry

end quadrilateral_pyramid_ratio_l99_9995


namespace eight_possible_values_for_d_l99_9955

def is_digit (n : ℕ) : Prop := n < 10

def distinct_digits (a b c d e : ℕ) : Prop :=
  is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧ is_digit e ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

def valid_subtraction (a b c d e : ℕ) : Prop :=
  10000 * a + 1000 * b + 100 * b + 10 * c + b -
  (10000 * b + 1000 * c + 100 * a + 10 * e + a) =
  10000 * d + 1000 * b + 100 * d + 10 * d + d

theorem eight_possible_values_for_d :
  ∃ (s : Finset ℕ), s.card = 8 ∧
  (∀ d, d ∈ s ↔ ∃ (a b c e : ℕ), distinct_digits a b c d e ∧ valid_subtraction a b c d e) :=
sorry

end eight_possible_values_for_d_l99_9955


namespace probability_yellow_marble_l99_9919

theorem probability_yellow_marble (blue red yellow : ℕ) 
  (h_blue : blue = 7)
  (h_red : red = 11)
  (h_yellow : yellow = 6) :
  (yellow : ℚ) / (blue + red + yellow) = 1 / 4 := by
  sorry

end probability_yellow_marble_l99_9919


namespace no_real_roots_implies_no_real_roots_composition_l99_9942

/-- A quadratic function f(x) = ax^2 + bx + c -/
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Theorem: If f(x) = x has no real roots, then f(f(x)) = x has no real roots -/
theorem no_real_roots_implies_no_real_roots_composition
  (a b c : ℝ) :
  (∀ x : ℝ, f a b c x ≠ x) →
  (∀ x : ℝ, f a b c (f a b c x) ≠ x) :=
by sorry

end no_real_roots_implies_no_real_roots_composition_l99_9942


namespace initial_men_count_l99_9993

theorem initial_men_count (M : ℝ) : 
  M * 17 = (M + 320) * 14.010989010989011 → M = 1500 := by
  sorry

end initial_men_count_l99_9993


namespace number_of_factors_27648_l99_9907

theorem number_of_factors_27648 : Nat.card (Nat.divisors 27648) = 44 := by
  sorry

end number_of_factors_27648_l99_9907


namespace jumping_contest_l99_9947

theorem jumping_contest (G F M K : ℤ) : 
  G = 39 ∧ 
  G = F + 19 ∧ 
  M = F - 12 ∧ 
  K = 2 * F - 5 →
  K = 35 :=
by sorry

end jumping_contest_l99_9947


namespace inequality_proof_l99_9965

theorem inequality_proof (n : ℕ) (hn : n > 1) : 
  let a : ℚ := 1 / n
  (a^2 : ℚ) < a ∧ a < (1 : ℚ) / a := by
  sorry

end inequality_proof_l99_9965


namespace min_value_of_function_equality_condition_l99_9905

theorem min_value_of_function (x : ℝ) (h : x > 0) : (x^2 + 1) / x ≥ 2 :=
  sorry

theorem equality_condition (x : ℝ) (h : x > 0) : (x^2 + 1) / x = 2 ↔ x = 1 :=
  sorry

end min_value_of_function_equality_condition_l99_9905


namespace items_deleted_l99_9918

theorem items_deleted (initial : ℕ) (remaining : ℕ) (deleted : ℕ) : 
  initial = 100 → remaining = 20 → deleted = initial - remaining → deleted = 80 :=
by sorry

end items_deleted_l99_9918


namespace regular_dinosaur_count_l99_9958

theorem regular_dinosaur_count :
  ∀ (barney_weight : ℕ) (regular_dino_weight : ℕ) (total_weight : ℕ) (num_regular_dinos : ℕ),
    regular_dino_weight = 800 →
    barney_weight = regular_dino_weight * num_regular_dinos + 1500 →
    total_weight = barney_weight + regular_dino_weight * num_regular_dinos →
    total_weight = 9500 →
    num_regular_dinos = 5 := by
sorry

end regular_dinosaur_count_l99_9958


namespace sum_within_range_l99_9915

/-- Converts a decimal number to its representation in a given base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in a given base to its decimal value -/
def fromBase (digits : List ℕ) (base : ℕ) : ℕ :=
  sorry

/-- Checks if a number is within the valid range -/
def isValidNumber (n : ℕ) : Prop :=
  n ≥ 3577 ∧ n ≤ 3583

/-- Calculates the sum of base conversions -/
def sumOfBaseConversions (n : ℕ) : ℕ :=
  fromBase (toBase n 7) 10 + fromBase (toBase n 8) 10 + fromBase (toBase n 9) 10

/-- Theorem: The sum of base conversions for valid numbers is within 0.5% of 25,000 -/
theorem sum_within_range (n : ℕ) (h : isValidNumber n) :
  (sumOfBaseConversions n : ℝ) > 24875 ∧ (sumOfBaseConversions n : ℝ) < 25125 :=
  sorry

end sum_within_range_l99_9915


namespace compare_squares_and_products_l99_9966

theorem compare_squares_and_products 
  (x a b : ℝ) 
  (h1 : x < a) 
  (h2 : a < b) 
  (h3 : b < 0) : 
  x^2 > a * x ∧ 
  a * x > b * x ∧ 
  x^2 > a^2 ∧ 
  a^2 > b^2 := by
sorry

end compare_squares_and_products_l99_9966


namespace min_expression_l99_9956

theorem min_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x * y / 2 + 18 / (x * y) ≥ 6) ∧ 
  ((x * y / 2 + 18 / (x * y) = 6) → (y / 2 + x / 3 ≥ 2)) ∧
  ((x * y / 2 + 18 / (x * y) = 6) ∧ (y / 2 + x / 3 = 2) → (x = 3 ∧ y = 2)) := by
sorry

end min_expression_l99_9956


namespace max_value_on_circle_l99_9934

theorem max_value_on_circle (x y z : ℝ) (h : x^2 + y^2 - 2*x + 2*y - 1 = 0) :
  ∃ (M : ℝ), M = 2 * Real.sqrt 2 + Real.sqrt 3 ∧ 
  ∀ (w : ℝ), w = (x + 1) * Real.sin z + (y - 1) * Real.cos z → w ≤ M :=
by sorry

end max_value_on_circle_l99_9934


namespace geometric_sequence_increasing_iff_l99_9931

/-- A geometric sequence with positive first term -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q ∧ a 1 > 0

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

theorem geometric_sequence_increasing_iff (a : ℕ → ℝ) :
  GeometricSequence a → (a 2 > a 1 ↔ IncreasingSequence a) := by sorry

end geometric_sequence_increasing_iff_l99_9931


namespace no_double_application_function_exists_l99_9904

theorem no_double_application_function_exists :
  ¬ ∃ (f : ℕ → ℕ), ∀ (n : ℕ), f (f n) = n + 1987 := by
sorry

end no_double_application_function_exists_l99_9904


namespace runs_by_running_percentage_l99_9921

def total_runs : ℕ := 120
def boundaries : ℕ := 6
def sixes : ℕ := 4
def runs_per_boundary : ℕ := 4
def runs_per_six : ℕ := 6

theorem runs_by_running_percentage :
  let runs_from_boundaries := boundaries * runs_per_boundary
  let runs_from_sixes := sixes * runs_per_six
  let runs_without_running := runs_from_boundaries + runs_from_sixes
  let runs_by_running := total_runs - runs_without_running
  (runs_by_running : ℚ) / total_runs * 100 = 60 := by sorry

end runs_by_running_percentage_l99_9921


namespace problem_solution_l99_9962

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := x^2 - t*x + 1

-- Define the predicate p
def p (t : ℝ) : Prop := ∃ x, f t x = 0

-- Define the predicate q
def q (t : ℝ) : Prop := ∀ x, |x - 1| ≥ 2 - t^2

theorem problem_solution (t : ℝ) :
  (q t → t ∈ Set.Ici (Real.sqrt 2) ∪ Set.Iic (-Real.sqrt 2)) ∧
  (¬p t ∧ ¬q t → t ∈ Set.Ioo (-Real.sqrt 2) (Real.sqrt 2)) :=
sorry

end problem_solution_l99_9962


namespace total_wheels_l99_9981

theorem total_wheels (bicycles tricycles : ℕ) 
  (bicycle_wheels tricycle_wheels : ℕ) : 
  bicycles = 24 → 
  tricycles = 14 → 
  bicycle_wheels = 2 → 
  tricycle_wheels = 3 → 
  bicycles * bicycle_wheels + tricycles * tricycle_wheels = 90 := by
  sorry

end total_wheels_l99_9981


namespace solution_correctness_l99_9922

-- First system of equations
def system1 (x y : ℝ) : Prop :=
  3 * x + 2 * y = 6 ∧ y = x - 2

-- Second system of equations
def system2 (m n : ℝ) : Prop :=
  m + 2 * n = 7 ∧ -3 * m + 5 * n = 1

theorem solution_correctness :
  (∃ x y, system1 x y) ∧ (∃ m n, system2 m n) ∧
  (∀ x y, system1 x y → x = 2 ∧ y = 0) ∧
  (∀ m n, system2 m n → m = 3 ∧ n = 2) := by
  sorry

end solution_correctness_l99_9922


namespace candidate_a_democratic_votes_l99_9960

theorem candidate_a_democratic_votes 
  (total_voters : ℝ) 
  (dem_percent : ℝ) 
  (rep_percent : ℝ) 
  (rep_for_a_percent : ℝ) 
  (total_for_a_percent : ℝ) 
  (h1 : dem_percent = 0.60)
  (h2 : rep_percent = 1 - dem_percent)
  (h3 : rep_for_a_percent = 0.20)
  (h4 : total_for_a_percent = 0.47) :
  let dem_for_a_percent := (total_for_a_percent * total_voters - rep_for_a_percent * rep_percent * total_voters) / (dem_percent * total_voters)
  dem_for_a_percent = 0.65 := by
sorry

end candidate_a_democratic_votes_l99_9960


namespace f_max_value_l99_9943

-- Define the function f(x) = x³ + 3x² - 4
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 4

-- State the theorem about the maximum value of f
theorem f_max_value :
  ∃ (M : ℝ), M = 0 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end f_max_value_l99_9943


namespace quadratic_equation_properties_l99_9988

theorem quadratic_equation_properties (a b c : ℝ) (h : a ≠ 0) :
  -- Statement ①
  (a + b + c = 0 → b^2 - 4*a*c ≥ 0) ∧
  -- Statement ②
  (∃ x y : ℝ, x ≠ y ∧ a*x^2 + c = 0 ∧ a*y^2 + c = 0 →
    ∃ u v : ℝ, u ≠ v ∧ a*u^2 + b*u + c = 0 ∧ a*v^2 + b*v + c = 0) ∧
  -- Statement ④
  (∀ x₀ : ℝ, a*x₀^2 + b*x₀ + c = 0 → b^2 - 4*a*c = (2*a*x₀ + b)^2) :=
by sorry

end quadratic_equation_properties_l99_9988


namespace solution_set_when_t_3_non_negative_for_all_x_iff_t_1_l99_9920

-- Define the function f
def f (t x : ℝ) : ℝ := x^2 - (t+1)*x + t

-- Theorem for part 1
theorem solution_set_when_t_3 :
  {x : ℝ | f 3 x > 0} = {x : ℝ | x < 1 ∨ x > 3} := by sorry

-- Theorem for part 2
theorem non_negative_for_all_x_iff_t_1 :
  (∀ x : ℝ, f t x ≥ 0) ↔ t = 1 := by sorry

end solution_set_when_t_3_non_negative_for_all_x_iff_t_1_l99_9920


namespace certain_number_proof_l99_9969

theorem certain_number_proof (x : ℝ) :
  (1.12 * x) / 4.98 = 528.0642570281125 → x = 2350 := by
  sorry

end certain_number_proof_l99_9969


namespace negative_polynomial_count_l99_9924

theorem negative_polynomial_count : 
  ∃ (S : Finset ℤ), (∀ x ∈ S, x^5 - 51*x^3 + 50*x < 0) ∧ 
                    (∀ x : ℤ, x^5 - 51*x^3 + 50*x < 0 → x ∈ S) ∧ 
                    Finset.card S = 12 :=
by sorry

end negative_polynomial_count_l99_9924


namespace curve_and_function_relation_l99_9963

-- Define a curve C as a set of points in ℝ²
def C : Set (ℝ × ℝ) := sorry

-- Define the function F
def F : ℝ → ℝ → ℝ := sorry

-- Theorem statement
theorem curve_and_function_relation :
  (∀ p : ℝ × ℝ, p ∈ C → F p.1 p.2 = 0) ∧
  (∀ p : ℝ × ℝ, F p.1 p.2 ≠ 0 → p ∉ C) :=
sorry

end curve_and_function_relation_l99_9963


namespace percentage_five_digit_numbers_with_repeated_digits_l99_9975

theorem percentage_five_digit_numbers_with_repeated_digits :
  let total_five_digit_numbers : ℕ := 90000
  let five_digit_numbers_without_repeats : ℕ := 27216
  let five_digit_numbers_with_repeats : ℕ := total_five_digit_numbers - five_digit_numbers_without_repeats
  let percentage : ℚ := (five_digit_numbers_with_repeats : ℚ) / (total_five_digit_numbers : ℚ) * 100
  ∃ (ε : ℚ), abs (percentage - 69.8) < ε ∧ ε ≤ 0.05 :=
by sorry

end percentage_five_digit_numbers_with_repeated_digits_l99_9975


namespace white_spotted_mushrooms_count_l99_9959

/-- The number of red mushrooms Bill gathered -/
def red_mushrooms : ℕ := 12

/-- The number of brown mushrooms Bill gathered -/
def brown_mushrooms : ℕ := 6

/-- The number of green mushrooms Ted gathered -/
def green_mushrooms : ℕ := 14

/-- The number of blue mushrooms Ted gathered -/
def blue_mushrooms : ℕ := 6

/-- The fraction of blue mushrooms with white spots -/
def blue_spotted_fraction : ℚ := 1/2

/-- The fraction of red mushrooms with white spots -/
def red_spotted_fraction : ℚ := 2/3

/-- The fraction of brown mushrooms with white spots -/
def brown_spotted_fraction : ℚ := 1

theorem white_spotted_mushrooms_count : 
  ⌊blue_spotted_fraction * blue_mushrooms⌋ + 
  ⌊red_spotted_fraction * red_mushrooms⌋ + 
  ⌊brown_spotted_fraction * brown_mushrooms⌋ = 17 := by
  sorry

end white_spotted_mushrooms_count_l99_9959


namespace total_points_is_238_l99_9949

/-- Represents a player's statistics in the basketball game -/
structure PlayerStats :=
  (two_pointers : ℕ)
  (three_pointers : ℕ)
  (free_throws : ℕ)
  (steals : ℕ)
  (rebounds : ℕ)
  (fouls : ℕ)

/-- Calculates the total points for a player given their stats -/
def calculate_points (stats : PlayerStats) : ℤ :=
  2 * stats.two_pointers + 3 * stats.three_pointers + stats.free_throws +
  stats.steals + 2 * stats.rebounds - 5 * stats.fouls

/-- The main theorem to prove -/
theorem total_points_is_238 :
  let sam := PlayerStats.mk 20 10 5 4 6 2
  let alex := PlayerStats.mk 15 8 5 6 3 3
  let jake := PlayerStats.mk 10 6 3 7 5 4
  let lily := PlayerStats.mk 16 4 7 3 7 1
  calculate_points sam + calculate_points alex + calculate_points jake + calculate_points lily = 238 := by
  sorry

end total_points_is_238_l99_9949


namespace product_evaluation_l99_9968

theorem product_evaluation (n : ℕ) (h : n = 3) :
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) * (n + 3) = 720 := by
  sorry

end product_evaluation_l99_9968


namespace bus_passengers_problem_l99_9961

/-- Proves that the initial number of people on a bus was 50, given the conditions of passenger changes at three stops. -/
theorem bus_passengers_problem (initial : ℕ) : 
  (((initial - 15) - (8 - 2)) - (4 - 3) = 28) → initial = 50 := by
  sorry

end bus_passengers_problem_l99_9961


namespace cubic_equation_equivalence_l99_9917

theorem cubic_equation_equivalence (x : ℝ) :
  x^3 + (x + 1)^4 + (x + 2)^3 = (x + 3)^4 ↔ 7 * (x^3 + 6 * x^2 + 13.14 * x + 10.29) = 0 := by
  sorry

end cubic_equation_equivalence_l99_9917


namespace root_sum_squares_l99_9998

theorem root_sum_squares (a b c : ℝ) : 
  (a^3 - 15*a^2 + 25*a - 10 = 0) → 
  (b^3 - 15*b^2 + 25*b - 10 = 0) → 
  (c^3 - 15*c^2 + 25*c - 10 = 0) → 
  (a-b)^2 + (b-c)^2 + (c-a)^2 = 125 := by sorry

end root_sum_squares_l99_9998


namespace complement_of_N_in_M_l99_9972

def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {2, 4}

theorem complement_of_N_in_M : M \ N = {1, 3, 5} := by sorry

end complement_of_N_in_M_l99_9972


namespace total_students_is_150_l99_9973

/-- Represents the number of students in a school with age distribution. -/
structure School where
  total : ℕ
  below_8 : ℕ
  age_8 : ℕ
  above_8 : ℕ

/-- Conditions for the school problem. -/
def school_conditions (s : School) : Prop :=
  s.below_8 = (s.total * 20) / 100 ∧
  s.above_8 = (s.age_8 * 2) / 3 ∧
  s.age_8 = 72 ∧
  s.total = s.below_8 + s.age_8 + s.above_8

/-- Theorem stating that the total number of students is 150. -/
theorem total_students_is_150 :
  ∃ s : School, school_conditions s ∧ s.total = 150 := by
  sorry

#check total_students_is_150

end total_students_is_150_l99_9973


namespace fixed_point_on_graph_l99_9944

theorem fixed_point_on_graph (k : ℝ) : 
  let f := fun (x : ℝ) => 5 * x^2 + k * x - 3 * k
  f 3 = 45 := by
  sorry

end fixed_point_on_graph_l99_9944


namespace jakes_weight_l99_9901

theorem jakes_weight (jake_weight sister_weight : ℕ) : 
  jake_weight - 33 = 2 * sister_weight →
  jake_weight + sister_weight = 153 →
  jake_weight = 113 := by
sorry

end jakes_weight_l99_9901


namespace price_difference_per_can_l99_9978

/-- Proves that the difference in price per can between the grocery store and bulk warehouse is 25 cents -/
theorem price_difference_per_can (bulk_price bulk_quantity grocery_price grocery_quantity : ℚ) : 
  bulk_price = 12 →
  bulk_quantity = 48 →
  grocery_price = 6 →
  grocery_quantity = 12 →
  (grocery_price / grocery_quantity - bulk_price / bulk_quantity) * 100 = 25 := by
  sorry

end price_difference_per_can_l99_9978


namespace max_savings_l99_9927

structure Flight where
  airline : String
  basePrice : ℕ
  discountPercentage : ℕ
  layovers : ℕ
  travelTime : ℕ

def calculateDiscountedPrice (flight : Flight) : ℚ :=
  flight.basePrice - (flight.basePrice * flight.discountPercentage / 100)

def flightOptions : List Flight := [
  ⟨"Delta Airlines", 850, 20, 1, 6⟩,
  ⟨"United Airlines", 1100, 30, 1, 7⟩,
  ⟨"American Airlines", 950, 25, 2, 9⟩,
  ⟨"Southwest Airlines", 900, 15, 1, 5⟩,
  ⟨"JetBlue Airways", 1200, 40, 0, 4⟩
]

theorem max_savings (options : List Flight := flightOptions) :
  let discountedPrices := options.map calculateDiscountedPrice
  let minPrice := discountedPrices.minimum?
  let maxPrice := discountedPrices.maximum?
  ∀ min max, minPrice = some min → maxPrice = some max →
    max - min = 90 :=
by sorry

end max_savings_l99_9927


namespace unwatered_rosebushes_l99_9952

/-- The number of unwatered rosebushes in Anna and Vitya's garden -/
theorem unwatered_rosebushes 
  (total : ℕ) 
  (vitya_watered : ℕ) 
  (anna_watered : ℕ) 
  (both_watered : ℕ)
  (h1 : total = 2006)
  (h2 : vitya_watered = total / 2)
  (h3 : anna_watered = total / 2)
  (h4 : both_watered = 3) :
  total - (vitya_watered + anna_watered - both_watered) = 3 :=
by sorry

end unwatered_rosebushes_l99_9952


namespace min_value_expression_l99_9967

theorem min_value_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (x^3 / (y - 1)) + (y^3 / (x - 1)) ≥ 16 ∧
  ∃ x y, x > 1 ∧ y > 1 ∧ (x^3 / (y - 1)) + (y^3 / (x - 1)) = 16 :=
by sorry

end min_value_expression_l99_9967


namespace whoosit_count_2_l99_9937

def worker_count_1 : ℕ := 150
def widget_count_1 : ℕ := 450
def whoosit_count_1 : ℕ := 300
def hours_1 : ℕ := 1

def worker_count_2 : ℕ := 90
def widget_count_2 : ℕ := 540
def hours_2 : ℕ := 3

def worker_count_3 : ℕ := 75
def widget_count_3 : ℕ := 300
def whoosit_count_3 : ℕ := 400
def hours_3 : ℕ := 4

def widget_production_rate_1 : ℚ := widget_count_1 / (worker_count_1 * hours_1)
def whoosit_production_rate_1 : ℚ := whoosit_count_1 / (worker_count_1 * hours_1)

def widget_production_rate_3 : ℚ := widget_count_3 / (worker_count_3 * hours_3)
def whoosit_production_rate_3 : ℚ := whoosit_count_3 / (worker_count_3 * hours_3)

theorem whoosit_count_2 (h : 2 * whoosit_production_rate_3 = widget_production_rate_3) :
  ∃ n : ℕ, n = 360 ∧ n / (worker_count_2 * hours_2) = whoosit_production_rate_3 :=
by sorry

end whoosit_count_2_l99_9937


namespace mother_triple_daughter_age_l99_9979

/-- Represents the age difference between mother and daughter -/
def age_difference : ℕ := 42 - 8

/-- Represents the current age of the mother -/
def mother_age : ℕ := 42

/-- Represents the current age of the daughter -/
def daughter_age : ℕ := 8

/-- The number of years until the mother is three times as old as her daughter -/
def years_until_triple : ℕ := 9

theorem mother_triple_daughter_age :
  mother_age + years_until_triple = 3 * (daughter_age + years_until_triple) :=
sorry

end mother_triple_daughter_age_l99_9979


namespace jacket_discount_percentage_l99_9938

/-- Proves that the discount percentage is 20% given the specified conditions --/
theorem jacket_discount_percentage (purchase_price selling_price discount_price : ℝ) 
  (h1 : purchase_price = 54)
  (h2 : selling_price = purchase_price + 0.4 * selling_price)
  (h3 : discount_price - purchase_price = 18) : 
  (selling_price - discount_price) / selling_price = 0.2 := by
  sorry

end jacket_discount_percentage_l99_9938


namespace largest_consecutive_sum_of_3_12_l99_9925

theorem largest_consecutive_sum_of_3_12 :
  (∃ (k : ℕ), k > 486 ∧ 
    (∃ (n : ℕ), 3^12 = (Finset.range k).sum (λ i => n + i + 1))) →
  False :=
sorry

end largest_consecutive_sum_of_3_12_l99_9925


namespace candy_mixture_price_l99_9906

/-- Given two types of candy mixed together, prove the price per pound of the mixture -/
theorem candy_mixture_price (X : ℝ) (price_X : ℝ) (weight_Y : ℝ) (price_Y : ℝ) 
  (total_weight : ℝ) (h1 : price_X = 3.50) (h2 : weight_Y = 6.25) (h3 : price_Y = 4.30) 
  (h4 : total_weight = 10) (h5 : X + weight_Y = total_weight) : 
  (X * price_X + weight_Y * price_Y) / total_weight = 4 := by
  sorry

end candy_mixture_price_l99_9906


namespace smallest_multiple_l99_9936

theorem smallest_multiple (n : ℕ) : n = 459 ↔ 
  (∃ k : ℕ, n = 17 * k) ∧ 
  (∃ m : ℕ, n = 76 * m + 3) ∧ 
  (∀ x : ℕ, x < n → ¬(∃ k : ℕ, x = 17 * k) ∨ ¬(∃ m : ℕ, x = 76 * m + 3)) := by
sorry

end smallest_multiple_l99_9936


namespace complex_fraction_simplification_l99_9930

theorem complex_fraction_simplification (x y : ℚ) (hx : x = 3) (hy : y = 4) :
  (1 / y) / (1 / x) = 3 / 4 := by
  sorry

end complex_fraction_simplification_l99_9930


namespace divisible_by_thirty_l99_9994

theorem divisible_by_thirty (n : ℕ+) : ∃ k : ℤ, (n : ℤ)^19 - (n : ℤ)^7 = 30 * k := by
  sorry

end divisible_by_thirty_l99_9994


namespace bank_withdrawal_bill_value_l99_9933

theorem bank_withdrawal_bill_value (x n : ℕ) (h1 : x = 300) (h2 : n = 30) :
  (2 * x) / n = 20 := by
  sorry

end bank_withdrawal_bill_value_l99_9933


namespace polygon_area_theorem_l99_9953

/-- The area of a polygon with given vertices -/
def polygonArea (vertices : List (ℤ × ℤ)) : ℚ :=
  sorry

/-- The number of integer points strictly inside a polygon -/
def interiorPoints (vertices : List (ℤ × ℤ)) : ℕ :=
  sorry

/-- The number of integer points on the boundary of a polygon -/
def boundaryPoints (vertices : List (ℤ × ℤ)) : ℕ :=
  sorry

theorem polygon_area_theorem :
  let vertices : List (ℤ × ℤ) := [(0, 1), (1, 2), (3, 2), (4, 1), (2, 0)]
  polygonArea vertices = 15/2 ∧
  interiorPoints vertices = 6 ∧
  boundaryPoints vertices = 5 :=
by sorry

end polygon_area_theorem_l99_9953


namespace trailing_zeros_of_500_power_150_l99_9913

-- Define 500 as 5 * 10^2
def five_hundred : ℕ := 5 * 10^2

-- Define the exponent
def exponent : ℕ := 150

-- Define the function to count trailing zeros
def trailing_zeros (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem trailing_zeros_of_500_power_150 :
  trailing_zeros (five_hundred ^ exponent) = 300 := by sorry

end trailing_zeros_of_500_power_150_l99_9913


namespace class_average_l99_9989

theorem class_average (total_students : ℕ) (high_scorers : ℕ) (high_score : ℕ) 
  (zero_scorers : ℕ) (rest_average : ℕ) : 
  total_students = 27 →
  high_scorers = 5 →
  high_score = 95 →
  zero_scorers = 3 →
  rest_average = 45 →
  (total_students - high_scorers - zero_scorers) * rest_average + 
    high_scorers * high_score = 1330 →
  (1330 : ℚ) / total_students = 1330 / 27 := by
sorry

end class_average_l99_9989


namespace greater_number_proof_l99_9951

theorem greater_number_proof (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 8) (h3 : x > y) : x = 19 := by
  sorry

end greater_number_proof_l99_9951


namespace specific_theater_seats_l99_9926

/-- Represents a theater with an arithmetic progression of seats per row -/
structure Theater where
  first_row_seats : ℕ
  seat_increment : ℕ
  last_row_seats : ℕ

/-- Calculates the total number of seats in the theater -/
def total_seats (t : Theater) : ℕ :=
  let n := (t.last_row_seats - t.first_row_seats) / t.seat_increment + 1
  n * (t.first_row_seats + t.last_row_seats) / 2

/-- Theorem stating that a theater with specific seat arrangement has 416 seats -/
theorem specific_theater_seats :
  let t : Theater := {
    first_row_seats := 14,
    seat_increment := 3,
    last_row_seats := 50
  }
  total_seats t = 416 := by
  sorry


end specific_theater_seats_l99_9926


namespace parallelogram_to_rhombus_l99_9911

structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

def is_convex (Q : Quadrilateral) : Prop := sorry

def is_parallelogram (Q : Quadrilateral) : Prop := sorry

def is_rhombus (Q : Quadrilateral) : Prop := sorry

def is_similar_not_congruent (Q1 Q2 : Quadrilateral) : Prop := sorry

def perpendicular_move (Q : Quadrilateral) : Quadrilateral := sorry

theorem parallelogram_to_rhombus (P : Quadrilateral) 
  (h_convex : is_convex P) 
  (h_initial : is_parallelogram P) 
  (h_final : ∃ (P_final : Quadrilateral), 
    (∃ (n : ℕ), n > 0 ∧ P_final = (perpendicular_move^[n] P)) ∧ 
    is_similar_not_congruent P P_final) :
  is_rhombus P := by sorry

end parallelogram_to_rhombus_l99_9911


namespace rationalize_denominator_l99_9970

theorem rationalize_denominator : 7 / Real.sqrt 200 = (7 * Real.sqrt 2) / 20 := by
  sorry

end rationalize_denominator_l99_9970


namespace specific_tetrahedron_volume_l99_9980

/-- Represents a tetrahedron with vertices P, Q, R, and S -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  QR : ℝ
  QS : ℝ
  PS : ℝ
  RS : ℝ

/-- Calculates the volume of a tetrahedron given its edge lengths -/
def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem: The volume of the specific tetrahedron is 24/√737 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    PQ := 6,
    PR := 4,
    QR := 5,
    QS := 5,
    PS := 4,
    RS := 15/4 * Real.sqrt 2
  }
  tetrahedronVolume t = 24 / Real.sqrt 737 := by
  sorry

end specific_tetrahedron_volume_l99_9980


namespace unique_solution_l99_9932

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x - 3

-- State the theorem
theorem unique_solution :
  ∃! x : ℝ, 2 * (f x) - 21 = f (x - 4) :=
by
  -- The proof goes here
  sorry

end unique_solution_l99_9932


namespace shaded_square_area_fraction_l99_9903

/-- The area of a square with vertices at (3,2), (5,4), (3,6), and (1,4) on a 6x6 grid is 2/9 of the total grid area. -/
theorem shaded_square_area_fraction :
  let grid_size : ℕ := 6
  let total_area : ℝ := (grid_size : ℝ) ^ 2
  let shaded_square_vertices : List (ℕ × ℕ) := [(3, 2), (5, 4), (3, 6), (1, 4)]
  let shaded_square_side : ℝ := 2 * Real.sqrt 2
  let shaded_square_area : ℝ := shaded_square_side ^ 2
  shaded_square_area / total_area = 2 / 9 := by sorry

end shaded_square_area_fraction_l99_9903


namespace donald_oranges_l99_9982

theorem donald_oranges (initial_oranges found_oranges : ℕ) 
  (h1 : initial_oranges = 4)
  (h2 : found_oranges = 5) :
  initial_oranges + found_oranges = 9 := by
  sorry

end donald_oranges_l99_9982


namespace inscribe_smaller_circles_l99_9939

-- Define a triangle type
structure Triangle where
  -- We don't need to specify the exact properties of a triangle here

-- Define a circle type
structure Circle where
  radius : ℝ

-- Define a function that checks if a circle can be inscribed in a triangle
def can_inscribe (t : Triangle) (c : Circle) : Prop :=
  sorry -- The exact definition is not important for this statement

-- Main theorem
theorem inscribe_smaller_circles 
  (t : Triangle) (r : ℝ) (n : ℕ) 
  (h : can_inscribe t (Circle.mk r)) :
  ∃ (circles : Finset Circle), 
    (circles.card = n^2) ∧ 
    (∀ c ∈ circles, c.radius = r / n) ∧
    (∀ c ∈ circles, can_inscribe t c) :=
sorry


end inscribe_smaller_circles_l99_9939


namespace exactly_one_black_ball_remains_l99_9912

/-- Represents the color of a ball -/
inductive Color
| Black
| Gray
| White

/-- Represents the state of the box -/
structure BoxState :=
  (black : Nat)
  (gray : Nat)
  (white : Nat)

/-- Simulates drawing two balls from the box -/
def drawTwoBalls (state : BoxState) : BoxState :=
  sorry

/-- Checks if the given state has exactly two balls remaining -/
def hasTwoballsRemaining (state : BoxState) : Bool :=
  state.black + state.gray + state.white = 2

/-- Represents the final state of the box after the procedure -/
def finalState (initialState : BoxState) : BoxState :=
  sorry

/-- The main theorem to be proved -/
theorem exactly_one_black_ball_remains :
  let initialState : BoxState := ⟨105, 89, 5⟩
  let finalState := finalState initialState
  hasTwoballsRemaining finalState ∧ finalState.black = 1 :=
by sorry

end exactly_one_black_ball_remains_l99_9912


namespace problem_statement_l99_9986

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Theorem statement
theorem problem_statement :
  (∃ m : ℝ, m > 0 ∧
    (Set.Icc (-2 : ℝ) 2 = { x | f (x + 1/2) ≤ 2*m + 1 }) ∧
    m = 3/2) ∧
  (∀ x y : ℝ, f x ≤ 2^y + 4/2^y + |2*x + 3|) :=
by sorry

end problem_statement_l99_9986


namespace hari_investment_is_8280_l99_9999

/-- Represents the business partnership between Praveen and Hari --/
structure Partnership where
  praveen_investment : ℕ
  praveen_months : ℕ
  hari_months : ℕ
  total_months : ℕ
  profit_ratio_praveen : ℕ
  profit_ratio_hari : ℕ

/-- Calculates Hari's investment given the partnership details --/
def calculate_hari_investment (p : Partnership) : ℕ :=
  (3 * p.praveen_investment * p.total_months) / (2 * p.hari_months)

/-- Theorem stating that Hari's investment is 8280 Rs given the specific partnership conditions --/
theorem hari_investment_is_8280 :
  let p : Partnership := {
    praveen_investment := 3220,
    praveen_months := 12,
    hari_months := 7,
    total_months := 12,
    profit_ratio_praveen := 2,
    profit_ratio_hari := 3
  }
  calculate_hari_investment p = 8280 := by
  sorry


end hari_investment_is_8280_l99_9999


namespace max_soccer_balls_l99_9916

/-- Represents the cost and quantity of soccer balls and basketballs -/
structure BallPurchase where
  soccer_cost : ℕ
  basketball_cost : ℕ
  total_balls : ℕ
  max_cost : ℕ

/-- Defines the conditions of the ball purchase problem -/
def ball_purchase_problem : BallPurchase where
  soccer_cost := 80
  basketball_cost := 60
  total_balls := 50
  max_cost := 3600

/-- Theorem stating the maximum number of soccer balls that can be purchased -/
theorem max_soccer_balls (bp : BallPurchase) : 
  bp.soccer_cost * 4 + bp.basketball_cost * 7 = 740 →
  bp.soccer_cost * 7 + bp.basketball_cost * 5 = 860 →
  ∃ (m : ℕ), m ≤ bp.total_balls ∧ 
             bp.soccer_cost * m + bp.basketball_cost * (bp.total_balls - m) ≤ bp.max_cost ∧
             ∀ (n : ℕ), n > m → 
               bp.soccer_cost * n + bp.basketball_cost * (bp.total_balls - n) > bp.max_cost :=
by sorry

#eval ball_purchase_problem.soccer_cost -- Expected output: 80
#eval ball_purchase_problem.basketball_cost -- Expected output: 60

end max_soccer_balls_l99_9916


namespace parabola_translation_l99_9976

-- Define the base parabola
def base_parabola (x : ℝ) : ℝ := x^2

-- Define the transformed parabola
def transformed_parabola (x : ℝ) : ℝ := (x + 4)^2 - 5

-- Theorem stating the translation process
theorem parabola_translation :
  ∀ x : ℝ, transformed_parabola x = base_parabola (x + 4) - 5 :=
by
  sorry

end parabola_translation_l99_9976


namespace initial_stock_calculation_l99_9985

/-- The number of toys sold in the first week -/
def toys_sold_first_week : ℕ := 38

/-- The number of toys sold in the second week -/
def toys_sold_second_week : ℕ := 26

/-- The number of toys left after two weeks -/
def toys_left : ℕ := 19

/-- The initial number of toys in stock -/
def initial_stock : ℕ := toys_sold_first_week + toys_sold_second_week + toys_left

theorem initial_stock_calculation :
  initial_stock = 83 := by sorry

end initial_stock_calculation_l99_9985


namespace room_occupancy_l99_9971

theorem room_occupancy (chairs : ℕ) (people : ℕ) : 
  (3 : ℚ) / 5 * people = (2 : ℚ) / 3 * chairs ∧ 
  chairs - (2 : ℚ) / 3 * chairs = 8 →
  people = 27 := by
sorry

end room_occupancy_l99_9971


namespace skittles_taken_away_l99_9929

def initial_skittles : ℕ := 25
def remaining_skittles : ℕ := 18

theorem skittles_taken_away : initial_skittles - remaining_skittles = 7 := by
  sorry

end skittles_taken_away_l99_9929


namespace decagon_triangle_probability_l99_9902

-- Define a regular decagon inscribed in a circle
def RegularDecagon : Type := Unit

-- Define a segment in the decagon
def Segment (d : RegularDecagon) : Type := Unit

-- Define a function to check if three segments form a triangle with positive area
def formsTriangle (d : RegularDecagon) (s1 s2 s3 : Segment d) : Prop := sorry

-- Define a function to calculate the probability
def probabilityOfTriangle (d : RegularDecagon) : ℚ := sorry

-- Theorem statement
theorem decagon_triangle_probability (d : RegularDecagon) : 
  probabilityOfTriangle d = 153 / 190 := by sorry

end decagon_triangle_probability_l99_9902


namespace subtraction_value_problem_l99_9987

theorem subtraction_value_problem (x y : ℝ) : 
  ((x - 5) / 7 = 7) → ((x - y) / 10 = 3) → y = 24 := by
  sorry

end subtraction_value_problem_l99_9987


namespace f_of_2_eq_neg_2_l99_9977

def f (x : ℝ) : ℝ := x^2 - 3*x

theorem f_of_2_eq_neg_2 : f 2 = -2 := by
  sorry

end f_of_2_eq_neg_2_l99_9977


namespace roberts_balls_theorem_l99_9984

/-- Calculates the final number of balls Robert has -/
def robertsFinalBalls (robertsInitial : ℕ) (timsTotal : ℕ) (jennysTotal : ℕ) : ℕ :=
  robertsInitial + timsTotal / 2 + jennysTotal / 3

theorem roberts_balls_theorem :
  robertsFinalBalls 25 40 60 = 65 := by
  sorry

end roberts_balls_theorem_l99_9984


namespace system_solution_proof_single_equation_solution_proof_l99_9957

-- System of equations
theorem system_solution_proof (x y : ℝ) : 
  x = 1 ∧ y = 2 → 2*x + 3*y = 8 ∧ 3*x - 5*y = -7 := by sorry

-- Single equation
theorem single_equation_solution_proof (x : ℝ) :
  x = -1 → (x-2)/(x+2) - 12/(x^2-4) = 1 := by sorry

end system_solution_proof_single_equation_solution_proof_l99_9957


namespace quadratic_roots_range_l99_9950

theorem quadratic_roots_range (m : ℝ) : 
  (∀ x, x^2 + (m-2)*x + (5-m) = 0 → x > 2) →
  m ∈ Set.Ioc (-5) (-4) :=
sorry

end quadratic_roots_range_l99_9950


namespace identity_proof_l99_9954

-- Define the necessary functions and series
def infiniteProduct (f : ℕ → ℝ) : ℝ := sorry

def infiniteSum (f : ℤ → ℝ) : ℝ := sorry

-- State the theorem
theorem identity_proof (x : ℝ) (h : |x| < 1) :
  -- First identity
  (infiniteProduct (λ m => (1 - x^(2*m - 1))^2)) =
  (1 / infiniteProduct (λ m => (1 - x^(2*m)))) *
  (infiniteSum (λ k => (-1)^k * x^(k^2)))
  ∧
  -- Second identity
  (infiniteProduct (λ m => (1 - x^m))) =
  (infiniteProduct (λ m => (1 + x^m))) *
  (infiniteSum (λ k => (-1)^k * x^(k^2))) :=
by sorry

end identity_proof_l99_9954


namespace homologous_functions_count_l99_9940

def f (x : ℝ) : ℝ := x^2

def isValidDomain (D : Set ℝ) : Prop :=
  (∀ x ∈ D, f x ∈ ({0, 1} : Set ℝ)) ∧
  (∀ y ∈ ({0, 1} : Set ℝ), ∃ x ∈ D, f x = y)

theorem homologous_functions_count :
  ∃! (domains : Finset (Set ℝ)), domains.card = 3 ∧
    ∀ D ∈ domains, isValidDomain D :=
sorry

end homologous_functions_count_l99_9940


namespace smallest_number_l99_9974

def binary_to_decimal (n : ℕ) : ℕ := n

def base_6_to_decimal (n : ℕ) : ℕ := n

def base_4_to_decimal (n : ℕ) : ℕ := n

def octal_to_decimal (n : ℕ) : ℕ := n

theorem smallest_number :
  let a := binary_to_decimal 111111
  let b := base_6_to_decimal 210
  let c := base_4_to_decimal 1000
  let d := octal_to_decimal 101
  a < b ∧ a < c ∧ a < d :=
by sorry

end smallest_number_l99_9974


namespace matrix_product_equality_l99_9935

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, -4; 6, 2]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![0, 3; -2, 1]

theorem matrix_product_equality :
  A * B = !![8, 5; -4, 20] := by sorry

end matrix_product_equality_l99_9935


namespace perfect_square_power_of_two_plus_65_l99_9923

theorem perfect_square_power_of_two_plus_65 (n : ℕ+) :
  (∃ (x : ℕ), 2^n.val + 65 = x^2) ↔ n.val = 4 ∨ n.val = 10 := by
sorry

end perfect_square_power_of_two_plus_65_l99_9923


namespace milk_cartons_calculation_l99_9908

/-- Calculates the number of 1L milk cartons needed for lasagna -/
def milk_cartons_needed (servings_per_person : ℕ) : ℕ :=
  let people : ℕ := 8
  let cup_per_serving : ℚ := 1/2
  let ml_per_cup : ℕ := 250
  let ml_per_carton : ℕ := 1000
  ⌈(people * servings_per_person : ℚ) * cup_per_serving * ml_per_cup / ml_per_carton⌉₊

theorem milk_cartons_calculation (s : ℕ) :
  milk_cartons_needed s = ⌈(8 * s : ℚ) * (1/2) * 250 / 1000⌉₊ :=
by sorry

end milk_cartons_calculation_l99_9908


namespace tangent_line_at_one_monotonicity_intervals_existence_of_positive_value_l99_9964

-- Define the function f(x) = ln x + ax
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x

-- Theorem for the tangent line equation
theorem tangent_line_at_one (a : ℝ) :
  a = 1 → ∃ (m b : ℝ), ∀ x y, y = f 1 x → (2 : ℝ) * x - y - 1 = 0 := by sorry

-- Theorem for monotonicity intervals
theorem monotonicity_intervals (a : ℝ) :
  (a ≥ 0 → ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) ∧
  (a < 0 → (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < -1/a → f a x₁ < f a x₂) ∧
           (∀ x₁ x₂, -1/a < x₁ ∧ x₁ < x₂ → f a x₁ > f a x₂)) := by sorry

-- Theorem for the range of a where f(x₀) > 0 exists
theorem existence_of_positive_value (a : ℝ) :
  (∃ x₀, 0 < x₀ ∧ f a x₀ > 0) ↔ a > -1 / Real.exp 1 := by sorry

end tangent_line_at_one_monotonicity_intervals_existence_of_positive_value_l99_9964


namespace intersection_of_A_and_B_l99_9909

-- Define the sets A and B
def A : Set ℝ := {1, 2, 1/2}
def B : Set ℝ := {y | ∃ x ∈ A, y = x^2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {1} := by
  sorry

end intersection_of_A_and_B_l99_9909
