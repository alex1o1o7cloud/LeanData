import Mathlib

namespace NUMINAMATH_CALUDE_simplify_expressions_l2127_212735

theorem simplify_expressions :
  (∀ (a b c d : ℝ), 
    a = 4 * Real.sqrt 5 ∧ 
    b = Real.sqrt 45 ∧ 
    c = Real.sqrt 8 ∧ 
    d = 4 * Real.sqrt 2 →
    a + b - c + d = 7 * Real.sqrt 5 + 2 * Real.sqrt 2) ∧
  (∀ (e f g : ℝ),
    e = 2 * Real.sqrt 48 ∧
    f = 3 * Real.sqrt 27 ∧
    g = Real.sqrt 6 →
    (e - f) / g = -(Real.sqrt 2) / 2) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expressions_l2127_212735


namespace NUMINAMATH_CALUDE_count_five_digit_integers_l2127_212709

/-- The number of different positive five-digit integers that can be formed using the digits 1, 1, 1, 2, and 2 -/
def num_five_digit_integers : ℕ := 10

/-- The multiset of digits used to form the integers -/
def digit_multiset : Multiset ℕ := {1, 1, 1, 2, 2}

/-- The theorem stating that the number of different positive five-digit integers
    that can be formed using the digits 1, 1, 1, 2, and 2 is equal to 10 -/
theorem count_five_digit_integers :
  (Multiset.card digit_multiset = 5) →
  (Multiset.card (Multiset.erase digit_multiset 1) = 2) →
  (Multiset.card (Multiset.erase digit_multiset 2) = 3) →
  num_five_digit_integers = 10 := by
  sorry


end NUMINAMATH_CALUDE_count_five_digit_integers_l2127_212709


namespace NUMINAMATH_CALUDE_light_travel_distance_l2127_212743

/-- The distance light travels in one year in kilometers. -/
def light_year_distance : ℝ := 9460800000000

/-- The number of years we want to calculate the light travel distance for. -/
def years : ℕ := 50

/-- Theorem stating the distance light travels in 50 years. -/
theorem light_travel_distance : 
  (light_year_distance * years : ℝ) = 4.7304e14 := by sorry

end NUMINAMATH_CALUDE_light_travel_distance_l2127_212743


namespace NUMINAMATH_CALUDE_expand_expression_l2127_212774

theorem expand_expression (x : ℝ) : 3 * (8 * x^2 - 2 * x + 1) = 24 * x^2 - 6 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2127_212774


namespace NUMINAMATH_CALUDE_intersection_points_inequality_l2127_212753

theorem intersection_points_inequality (a b x₁ x₂ : ℝ) 
  (h₁ : x₁ ≠ x₂) 
  (h₂ : Real.log x₁ / x₁ = a / 2 * x₁ + b) 
  (h₃ : Real.log x₂ / x₂ = a / 2 * x₂ + b) : 
  (x₁ + x₂) * (a / 2 * (x₁ + x₂) + b) > 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_inequality_l2127_212753


namespace NUMINAMATH_CALUDE_stratified_sampling_l2127_212751

theorem stratified_sampling (total_employees : ℕ) (administrators : ℕ) (sample_size : ℕ)
  (h1 : total_employees = 160)
  (h2 : administrators = 32)
  (h3 : sample_size = 20) :
  (administrators * sample_size) / total_employees = 4 := by sorry

end NUMINAMATH_CALUDE_stratified_sampling_l2127_212751


namespace NUMINAMATH_CALUDE_angle_measure_from_vector_sum_l2127_212721

/-- Given a triangle ABC with vectors m and n defined in terms of angle A, 
    prove that if the magnitude of their sum is √3, then A = π/3. -/
theorem angle_measure_from_vector_sum (A B C : ℝ) (a b c : ℝ) (m n : ℝ × ℝ) :
  0 < A ∧ A < π →
  m.1 = Real.cos (3 * A / 2) ∧ m.2 = Real.sin (3 * A / 2) →
  n.1 = Real.cos (A / 2) ∧ n.2 = Real.sin (A / 2) →
  Real.sqrt ((m.1 + n.1)^2 + (m.2 + n.2)^2) = Real.sqrt 3 →
  A = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_from_vector_sum_l2127_212721


namespace NUMINAMATH_CALUDE_waiting_room_problem_l2127_212775

theorem waiting_room_problem (initial_waiting : ℕ) (interview_room : ℕ) : 
  initial_waiting = 22 → interview_room = 5 → 
  ∃ (additional : ℕ), initial_waiting + additional = 5 * interview_room ∧ additional = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_waiting_room_problem_l2127_212775


namespace NUMINAMATH_CALUDE_inequality_condition_l2127_212749

theorem inequality_condition (a : ℝ) : 
  (∀ x > 1, (Real.exp x) / (x^3) - x - a * Real.log x ≥ 1) ↔ a ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_condition_l2127_212749


namespace NUMINAMATH_CALUDE_polar_to_rectangular_l2127_212720

theorem polar_to_rectangular (r θ : Real) (h : r = 4 ∧ θ = π / 4) :
  (r * Real.cos θ, r * Real.sin θ) = (2 * Real.sqrt 2, 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_l2127_212720


namespace NUMINAMATH_CALUDE_inequality_check_l2127_212742

theorem inequality_check : 
  (¬(0 < -5)) ∧ 
  (¬(7 < -1)) ∧ 
  (¬(10 < (1/4 : ℚ))) ∧ 
  (¬(-1 < -3)) ∧ 
  (-8 < -2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_check_l2127_212742


namespace NUMINAMATH_CALUDE_students_per_bench_l2127_212737

theorem students_per_bench (male_students : ℕ) (benches : ℕ) : 
  male_students = 29 →
  benches = 29 →
  ∃ (students_per_bench : ℕ), 
    students_per_bench ≥ 5 ∧
    students_per_bench * benches ≥ male_students + 4 * male_students :=
by sorry

end NUMINAMATH_CALUDE_students_per_bench_l2127_212737


namespace NUMINAMATH_CALUDE_expression_value_l2127_212770

-- Define the expression
def f (x : ℝ) : ℝ := 3 * x^2 + 5

-- Theorem statement
theorem expression_value : f (-1) = 8 := by sorry

end NUMINAMATH_CALUDE_expression_value_l2127_212770


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l2127_212731

theorem decimal_sum_to_fraction :
  (0.2 : ℚ) + 0.03 + 0.004 + 0.0006 = 1173 / 5000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l2127_212731


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l2127_212745

/-- Proves that 3 liters of 33% alcohol solution mixed with 1 liter of water results in 24.75% alcohol concentration -/
theorem alcohol_mixture_proof (x : ℝ) :
  (x > 0) →
  (0.33 * x = 0.2475 * (x + 1)) →
  x = 3 := by
  sorry

#check alcohol_mixture_proof

end NUMINAMATH_CALUDE_alcohol_mixture_proof_l2127_212745


namespace NUMINAMATH_CALUDE_original_number_proof_l2127_212763

theorem original_number_proof (x : ℚ) : 1 + (1 / x) = 9 / 5 → x = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2127_212763


namespace NUMINAMATH_CALUDE_simplify_nested_roots_l2127_212756

theorem simplify_nested_roots : 
  (((1 / 65536)^(1/2))^(1/3))^(1/4) = 1 / (2^(2/3)) := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_roots_l2127_212756


namespace NUMINAMATH_CALUDE_base8_4523_equals_2387_l2127_212791

def base8_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base8_4523_equals_2387 :
  base8_to_base10 [3, 2, 5, 4] = 2387 := by
  sorry

end NUMINAMATH_CALUDE_base8_4523_equals_2387_l2127_212791


namespace NUMINAMATH_CALUDE_book_count_proof_l2127_212711

/-- Proves that given a total of 144 books and a ratio of 7:5 for storybooks to science books,
    the number of storybooks is 84 and the number of science books is 60. -/
theorem book_count_proof (total : ℕ) (storybook_ratio : ℕ) (science_ratio : ℕ)
    (h_total : total = 144)
    (h_ratio : (storybook_ratio : ℚ) / (science_ratio : ℚ) = 7 / 5) :
    ∃ (storybooks science_books : ℕ),
      storybooks = 84 ∧
      science_books = 60 ∧
      storybooks + science_books = total ∧
      (storybooks : ℚ) / (science_books : ℚ) = storybook_ratio / science_ratio :=
by
  sorry

end NUMINAMATH_CALUDE_book_count_proof_l2127_212711


namespace NUMINAMATH_CALUDE_segment_length_ratio_l2127_212778

/-- Given a line segment AD with points B and C on it, prove that BC + DE = 5/8 * AD -/
theorem segment_length_ratio (A B C D E : ℝ) : 
  B ∈ Set.Icc A D → -- B is on segment AD
  C ∈ Set.Icc A D → -- C is on segment AD
  B - A = 3 * (D - B) → -- AB = 3 * BD
  C - A = 7 * (D - C) → -- AC = 7 * CD
  E - D = C - B → -- DE = BC
  E - A = D - E → -- E is midpoint of AD
  C - B + E - D = 5/8 * (D - A) := by sorry

end NUMINAMATH_CALUDE_segment_length_ratio_l2127_212778


namespace NUMINAMATH_CALUDE_divisibility_by_three_l2127_212758

theorem divisibility_by_three (B : ℕ) : 
  B < 10 ∧ (5 + 2 + B + 6) % 3 = 0 ↔ B = 2 :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_three_l2127_212758


namespace NUMINAMATH_CALUDE_probability_continuous_stripe_l2127_212762

/-- Represents a single cube with stripes on its faces -/
structure StripedCube where
  faces : Fin 6 → Bool

/-- Represents a tower of three cubes -/
structure CubeTower where
  top : StripedCube
  middle : StripedCube
  bottom : StripedCube

/-- Checks if there's a continuous stripe through a vertical face pair -/
def has_continuous_stripe (face_pair : Fin 4) (tower : CubeTower) : Bool :=
  sorry

/-- Counts the number of cube towers with a continuous vertical stripe -/
def count_continuous_stripes (towers : List CubeTower) : Nat :=
  sorry

/-- The total number of possible cube tower configurations -/
def total_configurations : Nat := 2^18

/-- The number of cube tower configurations with a continuous vertical stripe -/
def favorable_configurations : Nat := 64

theorem probability_continuous_stripe :
  (favorable_configurations : ℚ) / total_configurations = 1 / 4096 :=
sorry

end NUMINAMATH_CALUDE_probability_continuous_stripe_l2127_212762


namespace NUMINAMATH_CALUDE_cubic_function_property_l2127_212798

/-- A cubic function g(x) = Ax³ + Bx² - Cx + D -/
def g (A B C D : ℝ) (x : ℝ) : ℝ := A * x^3 + B * x^2 - C * x + D

theorem cubic_function_property (A B C D : ℝ) :
  g A B C D 2 = 5 ∧ g A B C D (-1) = -8 ∧ g A B C D 0 = 2 →
  -12*A + 6*B - 3*C + D = 27.5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_property_l2127_212798


namespace NUMINAMATH_CALUDE_max_angle_on_circle_l2127_212727

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define the angle between three points
def angle (p1 p2 p3 : Point) : ℝ := sorry

-- Define a function to check if a point is on a circle
def isOnCircle (c : Circle) (p : Point) : Prop := sorry

-- Theorem statement
theorem max_angle_on_circle (c : Circle) (A : Point) :
  ∃ M : Point, isOnCircle c M ∧
  (∀ N : Point, isOnCircle c N → angle c.center M A ≥ angle c.center N A) ∧
  angle c.center A M = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_angle_on_circle_l2127_212727


namespace NUMINAMATH_CALUDE_food_drive_cans_l2127_212724

theorem food_drive_cans (mark jaydon rachel : ℕ) : 
  mark = 100 ∧ 
  mark = 4 * jaydon ∧ 
  jaydon > 2 * rachel ∧ 
  mark + jaydon + rachel = 135 → 
  jaydon = 2 * rachel + 5 :=
by sorry

end NUMINAMATH_CALUDE_food_drive_cans_l2127_212724


namespace NUMINAMATH_CALUDE_square_root_divided_by_18_equals_4_l2127_212747

theorem square_root_divided_by_18_equals_4 (x : ℝ) : 
  (Real.sqrt x) / 18 = 4 → x = 5184 := by sorry

end NUMINAMATH_CALUDE_square_root_divided_by_18_equals_4_l2127_212747


namespace NUMINAMATH_CALUDE_exists_double_application_negation_l2127_212707

theorem exists_double_application_negation :
  ∃ f : ℝ → ℝ, ∀ x : ℝ, f (f x) = -x := by
  sorry

end NUMINAMATH_CALUDE_exists_double_application_negation_l2127_212707


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2127_212739

theorem rationalize_denominator : 
  (7 / Real.sqrt 98) = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2127_212739


namespace NUMINAMATH_CALUDE_monotonic_increasing_interval_l2127_212768

open Real

noncomputable def f (ω x : ℝ) : ℝ := Real.sqrt 3 * sin (ω * x) + cos (ω * x)

theorem monotonic_increasing_interval
  (ω : ℝ)
  (h_ω_pos : ω > 0)
  (α β : ℝ)
  (h_f_α : f ω α = 2)
  (h_f_β : f ω β = 0)
  (h_min_diff : |α - β| = π / 2) :
  ∃ k : ℤ, StrictMonoOn f (Set.Icc (2 * k * π - 2 * π / 3) (2 * k * π + π / 3)) :=
by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_interval_l2127_212768


namespace NUMINAMATH_CALUDE_t_value_l2127_212796

theorem t_value (p j t : ℝ) 
  (hj_p : j = p * (1 - 0.25))
  (hj_t : j = t * (1 - 0.20))
  (ht_p : t = p * (1 - t / 100)) :
  t = 6.25 := by sorry

end NUMINAMATH_CALUDE_t_value_l2127_212796


namespace NUMINAMATH_CALUDE_triplet_solution_l2127_212765

def is_valid_triplet (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≤ b ∧ b ≤ c ∧ a^2 + b^2 + c^2 = 2005

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(24,30,23), (12,30,31), (18,40,9), (15,22,36), (12,30,31)}

theorem triplet_solution :
  ∀ (a b c : ℕ), is_valid_triplet a b c ↔ (a, b, c) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_triplet_solution_l2127_212765


namespace NUMINAMATH_CALUDE_digit_difference_in_base_d_l2127_212718

/-- 
Given a base d > 8 and digits A and C in base d,
if AC_d + CC_d = 232_d, then A_d - C_d = 1_d.
-/
theorem digit_difference_in_base_d (d : ℕ) (A C : ℕ) 
  (h_base : d > 8) 
  (h_digits : A < d ∧ C < d) 
  (h_sum : A * d + C + C * d + C = 2 * d^2 + 3 * d + 2) : 
  A - C = 1 :=
sorry

end NUMINAMATH_CALUDE_digit_difference_in_base_d_l2127_212718


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_6_l2127_212705

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

theorem largest_four_digit_divisible_by_6 :
  ∀ n : ℕ, is_four_digit n → divisible_by_6 n → n ≤ 9996 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_6_l2127_212705


namespace NUMINAMATH_CALUDE_profit_percentage_unchanged_l2127_212736

/-- Represents a retailer's sales and profit information -/
structure RetailerInfo where
  monthly_sales : ℝ
  profit_percentage : ℝ
  discount_percentage : ℝ
  break_even_sales : ℝ

/-- The retailer's original sales information -/
def original_info : RetailerInfo :=
  { monthly_sales := 100
  , profit_percentage := 0.10
  , discount_percentage := 0
  , break_even_sales := 100 }

/-- The retailer's sales information with discount -/
def discounted_info : RetailerInfo :=
  { monthly_sales := 222.22
  , profit_percentage := 0.10
  , discount_percentage := 0.05
  , break_even_sales := 222.22 }

/-- Calculates the total profit for a given RetailerInfo -/
def total_profit (info : RetailerInfo) (price : ℝ) : ℝ :=
  info.monthly_sales * (info.profit_percentage - info.discount_percentage) * price

/-- Theorem stating that the profit percentage remains the same
    regardless of the discount, given the break-even sales volume -/
theorem profit_percentage_unchanged
  (price : ℝ)
  (h_price_pos : price > 0) :
  original_info.profit_percentage = discounted_info.profit_percentage :=
by
  sorry


end NUMINAMATH_CALUDE_profit_percentage_unchanged_l2127_212736


namespace NUMINAMATH_CALUDE_tennis_to_soccer_ratio_l2127_212773

/-- Represents the number of balls of each type -/
structure BallCounts where
  total : ℕ
  soccer : ℕ
  basketball : ℕ
  baseball : ℕ
  volleyball : ℕ
  tennis : ℕ

/-- The ratio of two natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Theorem stating the ratio of tennis balls to soccer balls -/
theorem tennis_to_soccer_ratio (counts : BallCounts) : 
  counts.total = 145 →
  counts.soccer = 20 →
  counts.basketball = counts.soccer + 5 →
  counts.baseball = counts.soccer + 10 →
  counts.volleyball = 30 →
  counts.total = counts.soccer + counts.basketball + counts.baseball + counts.volleyball + counts.tennis →
  (Ratio.mk counts.tennis counts.soccer) = (Ratio.mk 2 1) := by
  sorry

end NUMINAMATH_CALUDE_tennis_to_soccer_ratio_l2127_212773


namespace NUMINAMATH_CALUDE_R_sufficient_not_necessary_for_Q_l2127_212715

-- Define the propositions
variable (R P Q S : Prop)

-- Define the given conditions
axiom S_necessary_for_R : R → S
axiom S_sufficient_for_P : S → P
axiom Q_necessary_for_P : P → Q
axiom Q_sufficient_for_S : Q → S

-- Theorem to prove
theorem R_sufficient_not_necessary_for_Q :
  (R → Q) ∧ ¬(Q → R) :=
sorry

end NUMINAMATH_CALUDE_R_sufficient_not_necessary_for_Q_l2127_212715


namespace NUMINAMATH_CALUDE_sqrt_seven_minus_fraction_inequality_l2127_212783

theorem sqrt_seven_minus_fraction_inequality (m n : ℕ) (h : Real.sqrt 7 - (m : ℝ) / n > 0) :
  Real.sqrt 7 - (m : ℝ) / n > 1 / (m * n) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_minus_fraction_inequality_l2127_212783


namespace NUMINAMATH_CALUDE_vector_properties_l2127_212761

/-- Prove properties of vectors a, b, and c in a plane -/
theorem vector_properties (a b c : ℝ × ℝ) (θ : ℝ) : 
  a = (1, -2) →
  ‖c‖ = 2 * Real.sqrt 5 →
  ∃ (k : ℝ), c = k • a →
  ‖b‖ = 1 →
  (a + b) • (a - 2 • b) = 0 →
  (c = (-2, 4) ∨ c = (2, -4)) ∧ 
  Real.cos θ = (3 * Real.sqrt 5) / 5 :=
by sorry


end NUMINAMATH_CALUDE_vector_properties_l2127_212761


namespace NUMINAMATH_CALUDE_metal_waste_l2127_212777

/-- Given a rectangle with length l and breadth b (where l > b), from which a maximum-sized
    circular piece is cut and then a maximum-sized square piece is cut from that circle,
    the total amount of metal wasted is equal to l × b - b²/2. -/
theorem metal_waste (l b : ℝ) (h : l > b) (b_pos : b > 0) :
  let circle_area := π * (b/2)^2
  let square_side := b / Real.sqrt 2
  let square_area := square_side^2
  l * b - square_area = l * b - b^2/2 :=
by sorry

end NUMINAMATH_CALUDE_metal_waste_l2127_212777


namespace NUMINAMATH_CALUDE_sqrt_nested_equality_l2127_212760

theorem sqrt_nested_equality (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x * Real.sqrt (x * Real.sqrt x)) = (x^7)^(1/8) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nested_equality_l2127_212760


namespace NUMINAMATH_CALUDE_a_cube_gt_b_cube_l2127_212793

theorem a_cube_gt_b_cube (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a * abs a > b * abs b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_a_cube_gt_b_cube_l2127_212793


namespace NUMINAMATH_CALUDE_difference_in_balls_l2127_212703

/-- The number of packs of red bouncy balls Jill bought -/
def red_packs : ℕ := 5

/-- The number of packs of yellow bouncy balls Jill bought -/
def yellow_packs : ℕ := 4

/-- The number of bouncy balls in each package -/
def balls_per_pack : ℕ := 18

/-- The total number of red bouncy balls Jill bought -/
def total_red_balls : ℕ := red_packs * balls_per_pack

/-- The total number of yellow bouncy balls Jill bought -/
def total_yellow_balls : ℕ := yellow_packs * balls_per_pack

theorem difference_in_balls : total_red_balls - total_yellow_balls = 18 := by
  sorry

end NUMINAMATH_CALUDE_difference_in_balls_l2127_212703


namespace NUMINAMATH_CALUDE_negative_fractions_in_list_l2127_212738

def given_numbers : List ℚ := [5, -1, 0, -6, 125.73, 0.3, -3.5, -0.72, 5.25]

def is_negative_fraction (x : ℚ) : Prop := x < 0 ∧ x ≠ ⌊x⌋

theorem negative_fractions_in_list :
  ∀ x ∈ given_numbers, is_negative_fraction x ↔ x = -3.5 ∨ x = -0.72 := by
  sorry

end NUMINAMATH_CALUDE_negative_fractions_in_list_l2127_212738


namespace NUMINAMATH_CALUDE_polynomial_sum_equality_l2127_212772

/-- Given polynomials p, q, and r, prove their sum is equal to the specified polynomial -/
theorem polynomial_sum_equality (x : ℝ) : 
  let p := fun x : ℝ => -4*x^2 + 2*x - 5
  let q := fun x : ℝ => -6*x^2 + 4*x - 9
  let r := fun x : ℝ => 6*x^2 + 6*x + 2
  p x + q x + r x = -4*x^2 + 12*x - 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_equality_l2127_212772


namespace NUMINAMATH_CALUDE_range_of_m_for_real_solutions_l2127_212716

theorem range_of_m_for_real_solutions (m : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, 4 * Real.cos y + Real.sin y ^ 2 + m - 4 = 0) →
  0 ≤ m ∧ m ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_for_real_solutions_l2127_212716


namespace NUMINAMATH_CALUDE_tan_alpha_beta_eq_three_tan_alpha_l2127_212713

/-- Given that 2 sin β = sin(2α + β), prove that tan(α + β) = 3 tan α -/
theorem tan_alpha_beta_eq_three_tan_alpha (α β : ℝ) 
  (h : 2 * Real.sin β = Real.sin (2 * α + β)) :
  Real.tan (α + β) = 3 * Real.tan α := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_beta_eq_three_tan_alpha_l2127_212713


namespace NUMINAMATH_CALUDE_circumscribing_sphere_surface_area_l2127_212797

/-- A right triangular rectangular pyramid with side length a -/
structure RightTriangularRectangularPyramid where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- A sphere containing the vertices of a right triangular rectangular pyramid -/
structure CircumscribingSphere (p : RightTriangularRectangularPyramid) where
  radius : ℝ
  radius_pos : radius > 0

/-- The theorem stating the surface area of the circumscribing sphere -/
theorem circumscribing_sphere_surface_area
  (p : RightTriangularRectangularPyramid)
  (s : CircumscribingSphere p) :
  4 * Real.pi * s.radius ^ 2 = 3 * Real.pi * p.side_length ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_circumscribing_sphere_surface_area_l2127_212797


namespace NUMINAMATH_CALUDE_total_games_is_140_l2127_212752

/-- The number of teams in the "High School Ten" basketball conference -/
def num_teams : ℕ := 10

/-- The number of times each team plays every other team in the conference -/
def games_against_each_team : ℕ := 2

/-- The number of games each team plays against non-conference opponents -/
def non_conference_games_per_team : ℕ := 5

/-- The total number of games in a season involving the "High School Ten" teams -/
def total_games : ℕ := (num_teams * (num_teams - 1) / 2) * games_against_each_team + num_teams * non_conference_games_per_team

/-- Theorem stating that the total number of games in a season is 140 -/
theorem total_games_is_140 : total_games = 140 := by
  sorry

end NUMINAMATH_CALUDE_total_games_is_140_l2127_212752


namespace NUMINAMATH_CALUDE_power_product_equality_l2127_212700

theorem power_product_equality : 2000 * (2000 ^ 2000) = 2000 ^ 2001 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l2127_212700


namespace NUMINAMATH_CALUDE_three_times_m_minus_n_squared_l2127_212729

/-- Expresses "3 times m minus n squared" in algebraic notation -/
theorem three_times_m_minus_n_squared (m n : ℝ) : 
  (3 * m - n^2 : ℝ) = (3*m - n)^2 := by sorry

end NUMINAMATH_CALUDE_three_times_m_minus_n_squared_l2127_212729


namespace NUMINAMATH_CALUDE_equation_solution_l2127_212788

theorem equation_solution : ∃ x : ℚ, (5 * x + 12 * x = 540 - 12 * (x - 5)) ∧ (x = 600 / 29) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2127_212788


namespace NUMINAMATH_CALUDE_notebooks_given_to_tom_l2127_212734

def bernard_notebooks (red blue white remaining : ℕ) : Prop :=
  red + blue + white - remaining = 46

theorem notebooks_given_to_tom :
  bernard_notebooks 15 17 19 5 := by
  sorry

end NUMINAMATH_CALUDE_notebooks_given_to_tom_l2127_212734


namespace NUMINAMATH_CALUDE_a_beats_b_by_seven_seconds_l2127_212755

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  distance : ℝ
  time : ℝ

/-- The race scenario -/
def race_scenario (a b : Runner) : Prop :=
  a.distance = 280 ∧
  b.distance = 224 ∧
  a.time = 28 ∧
  a.speed = a.distance / a.time ∧
  b.speed = b.distance / a.time

/-- Theorem stating that A beats B by 7 seconds -/
theorem a_beats_b_by_seven_seconds (a b : Runner) (h : race_scenario a b) :
  b.distance / b.speed - a.time = 7 := by
  sorry


end NUMINAMATH_CALUDE_a_beats_b_by_seven_seconds_l2127_212755


namespace NUMINAMATH_CALUDE_joes_bath_shop_soap_sales_l2127_212708

theorem joes_bath_shop_soap_sales : ∃ n : ℕ, n > 0 ∧ n % 7 = 0 ∧ n % 23 = 0 ∧ ∀ m : ℕ, m > 0 → m % 7 = 0 → m % 23 = 0 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_joes_bath_shop_soap_sales_l2127_212708


namespace NUMINAMATH_CALUDE_units_digit_of_150_factorial_l2127_212781

theorem units_digit_of_150_factorial (n : ℕ) : n = 150 → n.factorial % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_150_factorial_l2127_212781


namespace NUMINAMATH_CALUDE_selection_ways_equal_210_l2127_212714

/-- The number of ways to select at least one boy from a group of 6 boys and G girls is 210 if and only if G = 1 -/
theorem selection_ways_equal_210 (G : ℕ) : (63 * 2^G = 210) ↔ G = 1 := by sorry

end NUMINAMATH_CALUDE_selection_ways_equal_210_l2127_212714


namespace NUMINAMATH_CALUDE_can_form_123_l2127_212732

/-- A type representing the allowed arithmetic operations -/
inductive Operation
| Add
| Subtract
| Multiply

/-- A type representing an arithmetic expression -/
inductive Expr
| Num (n : ℕ)
| Op (op : Operation) (e1 e2 : Expr)

/-- Evaluates an arithmetic expression -/
def eval : Expr → ℤ
| Expr.Num n => n
| Expr.Op Operation.Add e1 e2 => eval e1 + eval e2
| Expr.Op Operation.Subtract e1 e2 => eval e1 - eval e2
| Expr.Op Operation.Multiply e1 e2 => eval e1 * eval e2

/-- Checks if an expression uses each of the numbers 1, 2, 3, 4, 5 exactly once -/
def usesAllNumbers : Expr → Bool := sorry

/-- The main theorem stating that 123 can be formed using the given rules -/
theorem can_form_123 : ∃ e : Expr, usesAllNumbers e ∧ eval e = 123 := by sorry

end NUMINAMATH_CALUDE_can_form_123_l2127_212732


namespace NUMINAMATH_CALUDE_range_of_a_l2127_212722

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x + 3|
def g (x : ℝ) : ℝ := |x - 1| + 2

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, f a x₁ = g x₂) → 
  a ≥ 5 ∨ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2127_212722


namespace NUMINAMATH_CALUDE_range_of_a_inequality_proof_l2127_212795

-- Question 1
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (1/2 : ℝ) 1, |x - a| + |2*x - 1| ≤ |2*x + 1|) →
  a ∈ Set.Icc (-1 : ℝ) (5/2) :=
sorry

-- Question 2
theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^2 * b^2 + a^2 + b^2 ≥ a * b * (a + b + 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_inequality_proof_l2127_212795


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2127_212704

theorem inequality_solution_set (x : ℝ) : 
  (x - 1) / (x + 3) > (2 * x + 5) / (3 * x + 8) ↔ 
  (x > -3 ∧ x < -8/3) ∨ (x > (3 - Real.sqrt 69) / 2 ∧ x < (3 + Real.sqrt 69) / 2) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2127_212704


namespace NUMINAMATH_CALUDE_max_product_sum_max_product_sum_achieved_l2127_212710

theorem max_product_sum (A M C : ℕ) (h : A + M + C = 15) :
  (A * M * C + A * M + M * C + C * A) ≤ 200 :=
by sorry

theorem max_product_sum_achieved :
  ∃ A M C : ℕ, A + M + C = 15 ∧ A * M * C + A * M + M * C + C * A = 200 :=
by sorry

end NUMINAMATH_CALUDE_max_product_sum_max_product_sum_achieved_l2127_212710


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2127_212701

/-- The function f(x) = 1 + 2a^(x-1) has a fixed point at (1, 3), where a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 1 + 2 * a^(x - 1)
  f 1 = 3 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2127_212701


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_l2127_212785

theorem sqrt_sum_fractions : Real.sqrt (1/25 + 1/36) = Real.sqrt 61 / 30 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_l2127_212785


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2127_212733

-- Problem 1
theorem simplify_expression_1 (x : ℝ) : (x + 2) * (x - 1) - 3 * x * (x + 3) = -2 * x^2 - 8 * x - 2 := by
  sorry

-- Problem 2
theorem simplify_expression_2 (a : ℝ) : (a + 3) * (a^2 + 9) * (a - 3) = a^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2127_212733


namespace NUMINAMATH_CALUDE_extended_inequality_l2127_212779

theorem extended_inequality (n k : ℕ) (h1 : n ≥ 3) (h2 : 1 ≤ k) (h3 : k ≤ n) :
  2^n + 5^n > 2^(n-k) * 5^k + 2^k * 5^(n-k) := by
  sorry

end NUMINAMATH_CALUDE_extended_inequality_l2127_212779


namespace NUMINAMATH_CALUDE_max_value_product_sum_l2127_212787

theorem max_value_product_sum (X Y Z : ℕ) (sum_constraint : X + Y + Z = 15) :
  (∀ a b c : ℕ, a + b + c = 15 → X * Y * Z + X * Y + Y * Z + Z * X ≥ a * b * c + a * b + b * c + c * a) →
  X * Y * Z + X * Y + Y * Z + Z * X = 200 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_sum_l2127_212787


namespace NUMINAMATH_CALUDE_contrapositive_example_l2127_212719

theorem contrapositive_example :
  (∀ x : ℝ, x > 2 → x > 0) ↔ (∀ x : ℝ, x ≤ 0 → x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_example_l2127_212719


namespace NUMINAMATH_CALUDE_number_of_subsets_complement_union_l2127_212740

universe u

def U : Finset ℕ := {1, 3, 5, 7, 9}
def A : Finset ℕ := {1, 5, 9}
def B : Finset ℕ := {3, 5, 9}

theorem number_of_subsets_complement_union : Finset.card (Finset.powerset (U \ (A ∪ B))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_of_subsets_complement_union_l2127_212740


namespace NUMINAMATH_CALUDE_darla_electricity_payment_l2127_212794

/-- The number of watts of electricity Darla needs to pay for -/
def watts : ℝ := 300

/-- The cost per watt of electricity in dollars -/
def cost_per_watt : ℝ := 4

/-- The late fee in dollars -/
def late_fee : ℝ := 150

/-- The total payment in dollars -/
def total_payment : ℝ := 1350

theorem darla_electricity_payment :
  cost_per_watt * watts + late_fee = total_payment := by
  sorry

end NUMINAMATH_CALUDE_darla_electricity_payment_l2127_212794


namespace NUMINAMATH_CALUDE_solution_is_negative_two_l2127_212723

/-- The equation we want to solve -/
def equation (x : ℝ) : Prop := 2 / x = 1 / (x + 1)

/-- The theorem stating that -2 is the solution to the equation -/
theorem solution_is_negative_two : ∃ x : ℝ, equation x ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_solution_is_negative_two_l2127_212723


namespace NUMINAMATH_CALUDE_simplified_fraction_sum_l2127_212786

theorem simplified_fraction_sum (a b : ℕ) (h : a = 75 ∧ b = 100) :
  ∃ (c d : ℕ), (c.gcd d = 1) ∧ (a * d = b * c) ∧ (c + d = 7) := by
  sorry

end NUMINAMATH_CALUDE_simplified_fraction_sum_l2127_212786


namespace NUMINAMATH_CALUDE_jerry_age_l2127_212782

/-- Given that Mickey's age is 16 and Mickey's age is 6 years less than 200% of Jerry's age,
    prove that Jerry's age is 11. -/
theorem jerry_age (mickey_age jerry_age : ℕ) 
  (h1 : mickey_age = 16) 
  (h2 : mickey_age = 2 * jerry_age - 6) : 
  jerry_age = 11 := by
  sorry

end NUMINAMATH_CALUDE_jerry_age_l2127_212782


namespace NUMINAMATH_CALUDE_negation_of_universal_quadratic_inequality_l2127_212741

theorem negation_of_universal_quadratic_inequality :
  ¬(∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) ↔ ∃ x : ℝ, x^2 - 2*x + 1 < 0 :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_quadratic_inequality_l2127_212741


namespace NUMINAMATH_CALUDE_fraction_equality_l2127_212766

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (3 * x + y) / (x - 3 * y) = -2) : 
  (x + 3 * y) / (3 * x - y) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2127_212766


namespace NUMINAMATH_CALUDE_euler_formula_squared_l2127_212754

theorem euler_formula_squared (x : ℝ) : (Complex.cos x + Complex.I * Complex.sin x)^2 = Complex.cos (2*x) + Complex.I * Complex.sin (2*x) :=
by
  sorry

-- Euler's formula as an axiom
axiom euler_formula (x : ℝ) : Complex.exp (Complex.I * x) = Complex.cos x + Complex.I * Complex.sin x

end NUMINAMATH_CALUDE_euler_formula_squared_l2127_212754


namespace NUMINAMATH_CALUDE_seven_power_plus_one_prime_factors_l2127_212759

theorem seven_power_plus_one_prime_factors (n : ℕ) :
  ∃ (primes : Finset ℕ), 
    (∀ p ∈ primes, Nat.Prime p) ∧ 
    primes.card = 2 * n + 3 ∧
    (primes.prod id = 7^(7^(7^(7^2))) + 1) := by
  sorry

end NUMINAMATH_CALUDE_seven_power_plus_one_prime_factors_l2127_212759


namespace NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l2127_212746

theorem unique_solution_trigonometric_equation :
  ∃! x : ℝ, 2 * Real.sin (π * x / 2) - 2 * Real.cos (π * x / 2) = x^5 + 10*x - 54 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l2127_212746


namespace NUMINAMATH_CALUDE_future_age_relationship_l2127_212744

/-- Represents the current ages and future relationship between Rehana, Jacob, and Phoebe -/
theorem future_age_relationship (x : ℕ) : 
  let rehana_current_age : ℕ := 25
  let jacob_current_age : ℕ := 3
  let phoebe_current_age : ℕ := jacob_current_age * 5 / 3
  x = 5 ↔ 
    rehana_current_age + x = 3 * (phoebe_current_age + x) ∧
    x ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_future_age_relationship_l2127_212744


namespace NUMINAMATH_CALUDE_three_incorrect_statements_l2127_212790

theorem three_incorrect_statements (a b c : ℕ+) 
  (h1 : Nat.Coprime a.val b.val) 
  (h2 : Nat.Coprime b.val c.val) : 
  ∃ (a b c : ℕ+), 
    (¬(¬(b.val ∣ (a.val + c.val)^2))) ∧ 
    (¬(¬(b.val ∣ a.val^2 + c.val^2))) ∧ 
    (¬(¬(c.val ∣ (a.val + b.val)^2))) :=
sorry

end NUMINAMATH_CALUDE_three_incorrect_statements_l2127_212790


namespace NUMINAMATH_CALUDE_triangle_expression_bounds_l2127_212799

theorem triangle_expression_bounds (A B C : Real) (a b c : Real) :
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (c * Real.sin A = -a * Real.cos C) →
  ∃ (x : Real), (1 < x) ∧ 
  (x < (Real.sqrt 6 + Real.sqrt 2) / 2) ∧
  (x = Real.sqrt 3 * Real.sin A - Real.cos (B + 3 * π / 4)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_expression_bounds_l2127_212799


namespace NUMINAMATH_CALUDE_quadratic_range_l2127_212757

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 2*x + 5

-- Theorem statement
theorem quadratic_range :
  Set.range f = {y : ℝ | y ≥ 4} :=
sorry

end NUMINAMATH_CALUDE_quadratic_range_l2127_212757


namespace NUMINAMATH_CALUDE_highest_probability_high_speed_rail_l2127_212764

theorem highest_probability_high_speed_rail (beidou tianyan high_speed_rail : ℕ) :
  beidou = 3 →
  tianyan = 2 →
  high_speed_rail = 5 →
  let total := beidou + tianyan + high_speed_rail
  (high_speed_rail : ℚ) / total > (beidou : ℚ) / total ∧
  (high_speed_rail : ℚ) / total > (tianyan : ℚ) / total :=
by sorry

end NUMINAMATH_CALUDE_highest_probability_high_speed_rail_l2127_212764


namespace NUMINAMATH_CALUDE_smallest_value_l2127_212702

theorem smallest_value : 
  54 * Real.sqrt 3 < 144 ∧ 54 * Real.sqrt 3 < 108 * Real.sqrt 6 - 108 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_l2127_212702


namespace NUMINAMATH_CALUDE_cookie_distribution_l2127_212771

theorem cookie_distribution (total_cookies : ℚ) (blue_green_fraction : ℚ) (green_ratio : ℚ) :
  blue_green_fraction = 2/3 ∧ 
  green_ratio = 5/9 → 
  ∃ blue_fraction : ℚ, blue_fraction = 8/27 ∧ 
    blue_fraction + (blue_green_fraction - blue_fraction) + (1 - blue_green_fraction) = 1 ∧
    (blue_green_fraction - blue_fraction) / blue_green_fraction = green_ratio :=
by sorry

end NUMINAMATH_CALUDE_cookie_distribution_l2127_212771


namespace NUMINAMATH_CALUDE_refrigerator_price_l2127_212725

/-- The price Ramesh paid for the refrigerator --/
def price_paid (P : ℝ) : ℝ := 0.80 * P + 375

/-- The theorem stating the price Ramesh paid for the refrigerator --/
theorem refrigerator_price :
  ∃ P : ℝ,
    (1.12 * P = 17920) ∧
    (price_paid P = 13175) := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_price_l2127_212725


namespace NUMINAMATH_CALUDE_current_average_is_53_l2127_212717

/-- Represents a cricket player's batting statistics -/
structure CricketStats where
  matchesPlayed : ℕ
  totalRuns : ℕ

/-- Calculates the batting average -/
def battingAverage (stats : CricketStats) : ℚ :=
  stats.totalRuns / stats.matchesPlayed

/-- Theorem: If a player's average becomes 58 after scoring 78 in the 5th match,
    then their current average after 4 matches is 53 -/
theorem current_average_is_53
  (player : CricketStats)
  (h1 : player.matchesPlayed = 4)
  (h2 : battingAverage ⟨5, player.totalRuns + 78⟩ = 58) :
  battingAverage player = 53 := by
  sorry

end NUMINAMATH_CALUDE_current_average_is_53_l2127_212717


namespace NUMINAMATH_CALUDE_product_mod_25_l2127_212750

theorem product_mod_25 (n : ℕ) : 
  77 * 88 * 99 ≡ n [ZMOD 25] → 0 ≤ n → n < 25 → n = 24 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_25_l2127_212750


namespace NUMINAMATH_CALUDE_john_marble_weight_l2127_212776

/-- Represents a rectangular prism -/
structure RectangularPrism where
  height : ℝ
  baseLength : ℝ
  baseWidth : ℝ
  density : ℝ

/-- Calculates the volume of a rectangular prism -/
def volume (prism : RectangularPrism) : ℝ :=
  prism.height * prism.baseLength * prism.baseWidth

/-- Calculates the weight of a rectangular prism -/
def weight (prism : RectangularPrism) : ℝ :=
  prism.density * volume prism

/-- The main theorem stating the weight of John's marble prism -/
theorem john_marble_weight :
  let prism : RectangularPrism := {
    height := 8,
    baseLength := 2,
    baseWidth := 2,
    density := 2700
  }
  weight prism = 86400 := by
  sorry


end NUMINAMATH_CALUDE_john_marble_weight_l2127_212776


namespace NUMINAMATH_CALUDE_x_power_2023_l2127_212767

theorem x_power_2023 (x : ℝ) (h : (x - 1) * (x^4 + x^3 + x^2 + x + 1) = -2) : x^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_power_2023_l2127_212767


namespace NUMINAMATH_CALUDE_power_inequality_l2127_212784

theorem power_inequality (a b : ℝ) (ha : a > 0) (ha1 : a ≠ 1) (hab : a^b > 1) : a * b > b := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l2127_212784


namespace NUMINAMATH_CALUDE_jungkook_smallest_l2127_212712

def yoongi_collection : ℕ := 4
def jungkook_collection : ℚ := 6 / 3
def yuna_collection : ℕ := 5

theorem jungkook_smallest :
  jungkook_collection < yoongi_collection ∧ jungkook_collection < yuna_collection :=
sorry

end NUMINAMATH_CALUDE_jungkook_smallest_l2127_212712


namespace NUMINAMATH_CALUDE_x_142_equals_1995_and_unique_l2127_212792

def p (x : ℕ) : ℕ := sorry

def q (x : ℕ) : ℕ := sorry

def x : ℕ → ℕ
  | 0 => 1
  | n + 1 => (x n * p (x n)) / q (x n)

theorem x_142_equals_1995_and_unique :
  x 142 = 1995 ∧ ∀ n : ℕ, n ≠ 142 → x n ≠ 1995 := by sorry

end NUMINAMATH_CALUDE_x_142_equals_1995_and_unique_l2127_212792


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l2127_212780

theorem min_value_of_sum_of_squares (a b : ℝ) : 
  (a > 0) → 
  (b > 0) → 
  ((a - 1)^3 + (b - 1)^3 ≥ 3 * (2 - a - b)) → 
  (a^2 + b^2 ≥ 2) := by
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l2127_212780


namespace NUMINAMATH_CALUDE_min_value_expression_l2127_212748

theorem min_value_expression (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^3 / (y - 2)) + (y^3 / (x - 2)) ≥ 64 ∧
  ∃ x y, x > 2 ∧ y > 2 ∧ (x^3 / (y - 2)) + (y^3 / (x - 2)) = 64 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2127_212748


namespace NUMINAMATH_CALUDE_marys_remaining_cards_l2127_212769

theorem marys_remaining_cards (initial_cards promised_cards bought_cards : ℝ) :
  initial_cards + bought_cards - promised_cards =
  initial_cards + bought_cards - promised_cards :=
by sorry

end NUMINAMATH_CALUDE_marys_remaining_cards_l2127_212769


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2127_212789

theorem smallest_n_congruence (n : ℕ) : 
  (∀ m : ℕ, 0 < m → m < 15 → ¬(890 * m ≡ 1426 * m [ZMOD 30])) ∧ 
  (890 * 15 ≡ 1426 * 15 [ZMOD 30]) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2127_212789


namespace NUMINAMATH_CALUDE_centroid_distance_theorem_l2127_212706

/-- Represents the possible distances from the centroid of a triangle to a plane -/
inductive CentroidDistance : Type
  | six : CentroidDistance
  | two : CentroidDistance
  | eight_thirds : CentroidDistance
  | four_thirds : CentroidDistance

/-- Given a triangle with vertices at distances 5, 6, and 7 from a plane,
    the distance from the centroid to the same plane is one of the defined values -/
theorem centroid_distance_theorem (d1 d2 d3 : ℝ) (h1 : d1 = 5) (h2 : d2 = 6) (h3 : d3 = 7) :
  ∃ (cd : CentroidDistance), true :=
sorry

end NUMINAMATH_CALUDE_centroid_distance_theorem_l2127_212706


namespace NUMINAMATH_CALUDE_hector_siblings_product_l2127_212730

/-- A family where one member has 4 sisters and 7 brothers -/
structure Family :=
  (sisters_of_helen : ℕ)
  (brothers_of_helen : ℕ)
  (helen_is_female : Bool)
  (hector_is_male : Bool)

/-- The number of sisters Hector has in the family -/
def sisters_of_hector (f : Family) : ℕ :=
  f.sisters_of_helen + (if f.helen_is_female then 1 else 0)

/-- The number of brothers Hector has in the family -/
def brothers_of_hector (f : Family) : ℕ :=
  f.brothers_of_helen - 1

theorem hector_siblings_product (f : Family) 
  (h1 : f.sisters_of_helen = 4)
  (h2 : f.brothers_of_helen = 7)
  (h3 : f.helen_is_female = true)
  (h4 : f.hector_is_male = true) :
  (sisters_of_hector f) * (brothers_of_hector f) = 30 :=
sorry

end NUMINAMATH_CALUDE_hector_siblings_product_l2127_212730


namespace NUMINAMATH_CALUDE_unique_prime_in_set_l2127_212728

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def six_digit_number (A : ℕ) : ℕ := 303100 + A

theorem unique_prime_in_set : 
  ∃! A : ℕ, A < 10 ∧ is_prime (six_digit_number A) ∧ six_digit_number A = 303103 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_in_set_l2127_212728


namespace NUMINAMATH_CALUDE_player_a_strategy_wins_l2127_212726

-- Define the grid as a 3x3 matrix of real numbers
def Grid := Matrix (Fin 3) (Fin 3) ℝ

-- Define a function to calculate the sum of first and third rows
def sumRows (g : Grid) : ℝ := 
  (g 0 0 + g 0 1 + g 0 2) + (g 2 0 + g 2 1 + g 2 2)

-- Define a function to calculate the sum of first and third columns
def sumCols (g : Grid) : ℝ := 
  (g 0 0 + g 1 0 + g 2 0) + (g 0 2 + g 1 2 + g 2 2)

-- Theorem statement
theorem player_a_strategy_wins 
  (cards : Finset ℝ) 
  (h_card_count : cards.card = 9) : 
  ∃ (g : Grid), (∀ i j, g i j ∈ cards) ∧ sumRows g ≥ sumCols g := by
  sorry


end NUMINAMATH_CALUDE_player_a_strategy_wins_l2127_212726
