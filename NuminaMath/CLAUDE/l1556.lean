import Mathlib

namespace NUMINAMATH_CALUDE_justin_reading_theorem_l1556_155637

/-- Calculates the total number of pages Justin reads in a week -/
def totalPagesRead (firstDayPages : ℕ) (remainingDays : ℕ) : ℕ :=
  firstDayPages + remainingDays * (2 * firstDayPages)

/-- Proves that Justin reads 130 pages in a week -/
theorem justin_reading_theorem :
  totalPagesRead 10 6 = 130 :=
by sorry

end NUMINAMATH_CALUDE_justin_reading_theorem_l1556_155637


namespace NUMINAMATH_CALUDE_equation_has_root_in_interval_l1556_155618

theorem equation_has_root_in_interval (t : ℝ) (h : t ∈ ({6, 7, 8, 9} : Set ℝ)) :
  ∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^4 - t*x + 1/t = 0 := by
  sorry

#check equation_has_root_in_interval

end NUMINAMATH_CALUDE_equation_has_root_in_interval_l1556_155618


namespace NUMINAMATH_CALUDE_at_least_one_nonzero_l1556_155640

theorem at_least_one_nonzero (a b : ℝ) : (a ≠ 0 ∨ b ≠ 0) ↔ a^2 + b^2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_nonzero_l1556_155640


namespace NUMINAMATH_CALUDE_lactate_bicarbonate_reaction_in_extracellular_fluid_l1556_155638

-- Define the extracellular fluid
structure ExtracellularFluid where
  is_liquid_environment : Bool

-- Define a biochemical reaction
structure BiochemicalReaction where
  occurs_in_extracellular_fluid : Bool

-- Define the specific reaction
def lactate_bicarbonate_reaction : BiochemicalReaction where
  occurs_in_extracellular_fluid := true

-- Theorem statement
theorem lactate_bicarbonate_reaction_in_extracellular_fluid 
  (ecf : ExtracellularFluid) 
  (h : ecf.is_liquid_environment = true) : 
  lactate_bicarbonate_reaction.occurs_in_extracellular_fluid = true := by
  sorry

end NUMINAMATH_CALUDE_lactate_bicarbonate_reaction_in_extracellular_fluid_l1556_155638


namespace NUMINAMATH_CALUDE_trapezoid_intersection_distances_l1556_155639

/-- Given a trapezoid ABCD with legs AB and CD, and bases AD and BC where AD > BC,
    this theorem proves the distances from the intersection point M of the extended legs
    to the vertices of the trapezoid. -/
theorem trapezoid_intersection_distances
  (AB CD AD BC : ℝ) -- Lengths of sides
  (h_AD_gt_BC : AD > BC) -- Condition: AD > BC
  : ∃ (BM AM CM DM : ℝ),
    BM = (AB * BC) / (AD - BC) ∧
    AM = (AB * AD) / (AD - BC) ∧
    CM = (CD * BC) / (AD - BC) ∧
    DM = (CD * AD) / (AD - BC) := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_intersection_distances_l1556_155639


namespace NUMINAMATH_CALUDE_discount_rate_proof_l1556_155683

theorem discount_rate_proof (initial_price final_price : ℝ) 
  (h1 : initial_price = 7200)
  (h2 : final_price = 3528)
  (h3 : ∃ x : ℝ, initial_price * (1 - x)^2 = final_price) :
  ∃ x : ℝ, x = 0.3 ∧ initial_price * (1 - x)^2 = final_price := by
sorry

end NUMINAMATH_CALUDE_discount_rate_proof_l1556_155683


namespace NUMINAMATH_CALUDE_condition_relationship_l1556_155677

theorem condition_relationship (x : ℝ) :
  (∀ x, -1 < x ∧ x < 3 → x < 3) ∧ 
  ¬(∀ x, x < 3 → -1 < x ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l1556_155677


namespace NUMINAMATH_CALUDE_red_ball_probability_l1556_155696

theorem red_ball_probability (w r : ℕ) : 
  r > w ∧ r < 2 * w ∧ 2 * w + 3 * r = 60 → 
  (r : ℚ) / (w + r : ℚ) = 14 / 23 := by
sorry

end NUMINAMATH_CALUDE_red_ball_probability_l1556_155696


namespace NUMINAMATH_CALUDE_ratio_fraction_equality_l1556_155619

theorem ratio_fraction_equality (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 6) :
  (5 * A + 3 * B) / (5 * C - 2 * A) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_fraction_equality_l1556_155619


namespace NUMINAMATH_CALUDE_sqrt_12_sqrt_2_div_sqrt_3_minus_2sin45_equals_1_l1556_155633

theorem sqrt_12_sqrt_2_div_sqrt_3_minus_2sin45_equals_1 :
  let sqrt_12 := 2 * Real.sqrt 3
  let sin_45 := Real.sqrt 2 / 2
  (sqrt_12 * Real.sqrt 2) / Real.sqrt 3 - 2 * sin_45 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_12_sqrt_2_div_sqrt_3_minus_2sin45_equals_1_l1556_155633


namespace NUMINAMATH_CALUDE_factorization_x_squared_minus_nine_l1556_155659

theorem factorization_x_squared_minus_nine (x : ℝ) : x^2 - 9 = (x - 3) * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x_squared_minus_nine_l1556_155659


namespace NUMINAMATH_CALUDE_mountain_valley_trail_length_l1556_155649

/-- Represents the length of the Mountain Valley Trail hike --/
def MountainValleyTrail : Type := { trail : ℕ // trail > 0 }

/-- Represents the daily hike distances --/
def DailyHikes : Type := Fin 5 → ℕ

theorem mountain_valley_trail_length 
  (hikes : DailyHikes) 
  (day1_2 : hikes 0 + hikes 1 = 30)
  (day2_4_avg : (hikes 1 + hikes 3) / 2 = 15)
  (day3_4_5 : hikes 2 + hikes 3 + hikes 4 = 45)
  (day1_3 : hikes 0 + hikes 2 = 33) :
  ∃ (trail : MountainValleyTrail), (hikes 0 + hikes 1 + hikes 2 + hikes 3 + hikes 4 : ℕ) = trail.val ∧ trail.val = 75 := by
  sorry

end NUMINAMATH_CALUDE_mountain_valley_trail_length_l1556_155649


namespace NUMINAMATH_CALUDE_ellipse_max_value_l1556_155600

theorem ellipse_max_value (x y : ℝ) : 
  x^2 + 4*y^2 = 4 → 
  ∃ (M : ℝ), M = 7 ∧ ∀ (a b : ℝ), a^2 + 4*b^2 = 4 → (3/4)*a^2 + 2*a - b^2 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_ellipse_max_value_l1556_155600


namespace NUMINAMATH_CALUDE_student_distribution_l1556_155621

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of permutations of n items -/
def factorial (n : ℕ) : ℕ := sorry

theorem student_distribution (total : ℕ) (male : ℕ) (schemes : ℕ) :
  total = 8 →
  choose male 2 * choose (total - male) 1 * factorial 3 = schemes →
  schemes = 90 →
  male = 3 ∧ total - male = 5 := by sorry

end NUMINAMATH_CALUDE_student_distribution_l1556_155621


namespace NUMINAMATH_CALUDE_zeros_not_adjacent_probability_probability_calculation_main_theorem_l1556_155657

/-- The probability of 2 zeros not being adjacent when 4 ones and 2 zeros are randomly arranged in a row -/
theorem zeros_not_adjacent_probability : ℚ :=
  2/3

/-- The total number of ways to arrange 4 ones and 2 zeros in a row -/
def total_arrangements : ℕ :=
  Nat.choose 6 2

/-- The number of arrangements where the 2 zeros are not adjacent -/
def non_adjacent_arrangements : ℕ :=
  Nat.choose 5 2

/-- The probability is the ratio of non-adjacent arrangements to total arrangements -/
theorem probability_calculation (h : zeros_not_adjacent_probability = non_adjacent_arrangements / total_arrangements) :
  zeros_not_adjacent_probability = 2/3 := by
  sorry

/-- The main theorem stating that the probability of 2 zeros not being adjacent is 2/3 -/
theorem main_theorem :
  zeros_not_adjacent_probability = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_zeros_not_adjacent_probability_probability_calculation_main_theorem_l1556_155657


namespace NUMINAMATH_CALUDE_least_integer_in_ratio_l1556_155687

theorem least_integer_in_ratio (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  b = (3 * a) / 2 →
  c = (5 * a) / 2 →
  a + b + c = 60 →
  a = 12 :=
sorry

end NUMINAMATH_CALUDE_least_integer_in_ratio_l1556_155687


namespace NUMINAMATH_CALUDE_thomson_savings_l1556_155614

def incentive : ℚ := 240

def food_fraction : ℚ := 1/3
def clothes_fraction : ℚ := 1/5
def savings_fraction : ℚ := 3/4

def food_expense : ℚ := food_fraction * incentive
def clothes_expense : ℚ := clothes_fraction * incentive
def total_expense : ℚ := food_expense + clothes_expense
def remaining : ℚ := incentive - total_expense
def savings : ℚ := savings_fraction * remaining

theorem thomson_savings : savings = 84 := by
  sorry

end NUMINAMATH_CALUDE_thomson_savings_l1556_155614


namespace NUMINAMATH_CALUDE_largest_of_four_consecutive_odd_integers_l1556_155643

theorem largest_of_four_consecutive_odd_integers (a b c d : ℤ) : 
  (∀ n : ℤ, a = 2*n + 1 ∧ b = 2*n + 3 ∧ c = 2*n + 5 ∧ d = 2*n + 7) → 
  (a + b + c + d = 200) → 
  d = 53 := by
sorry

end NUMINAMATH_CALUDE_largest_of_four_consecutive_odd_integers_l1556_155643


namespace NUMINAMATH_CALUDE_jay_change_calculation_l1556_155655

/-- Calculates the change Jay received after purchasing items with a discount --/
theorem jay_change_calculation (book pen ruler notebook pencil_case : ℚ)
  (h_book : book = 25)
  (h_pen : pen = 4)
  (h_ruler : ruler = 1)
  (h_notebook : notebook = 8)
  (h_pencil_case : pencil_case = 6)
  (discount_rate : ℚ)
  (h_discount : discount_rate = 0.1)
  (paid_amount : ℚ)
  (h_paid : paid_amount = 100) :
  let total_before_discount := book + pen + ruler + notebook + pencil_case
  let discount_amount := discount_rate * total_before_discount
  let total_after_discount := total_before_discount - discount_amount
  paid_amount - total_after_discount = 60.4 := by
sorry

end NUMINAMATH_CALUDE_jay_change_calculation_l1556_155655


namespace NUMINAMATH_CALUDE_part_a_part_b_l1556_155671

def solution_set_a : Set (ℤ × ℤ) := {(6, -21), (-13, -2), (4, 15), (23, -4), (7, -12), (-4, -1), (3, 6), (14, -5), (8, -9), (-1, 0), (2, 3), (11, -6)}

def equation_set_a : Set (ℤ × ℤ) := {(x, y) | x * y + 3 * x - 5 * y = -3}

theorem part_a : equation_set_a = solution_set_a := by sorry

def solution_set_b : Set (ℤ × ℤ) := {(4, 2)}

def equation_set_b : Set (ℤ × ℤ) := {(x, y) | x - y = x / y}

theorem part_b : equation_set_b = solution_set_b := by sorry

end NUMINAMATH_CALUDE_part_a_part_b_l1556_155671


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l1556_155601

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The function f(x) = x(x+1)(x+2)...(x+n) -/
def f (n : ℕ) (x : ℝ) : ℝ := (List.range (n + 1)).foldl (fun acc i => acc * (x + i)) x

/-- Theorem: The derivative of f(x) at x = 0 is equal to n! -/
theorem derivative_f_at_zero (n : ℕ) : 
  deriv (f n) 0 = factorial n := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l1556_155601


namespace NUMINAMATH_CALUDE_range_of_a_l1556_155653

def f (a : ℝ) (x : ℝ) : ℝ := (x - 2)^2 * |x - a|

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 2 4, x * (deriv (f a) x) ≥ 0) ↔ a ∈ Set.Iic 2 ∪ Set.Ici 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1556_155653


namespace NUMINAMATH_CALUDE_ron_siblings_product_l1556_155697

/-- Represents a family structure --/
structure Family :=
  (sisters : ℕ)
  (brothers : ℕ)

/-- The problem setup --/
def problem_setup (harry_family : Family) (harriet : Family) (ron : Family) : Prop :=
  harry_family.sisters = 4 ∧
  harry_family.brothers = 6 ∧
  harriet.sisters = harry_family.sisters - 1 ∧
  harriet.brothers = harry_family.brothers ∧
  ron.sisters = harriet.sisters ∧
  ron.brothers = harriet.brothers + 2

/-- The main theorem --/
theorem ron_siblings_product (harry_family : Family) (harriet : Family) (ron : Family) 
  (h : problem_setup harry_family harriet ron) : 
  ron.sisters * ron.brothers = 32 := by
  sorry


end NUMINAMATH_CALUDE_ron_siblings_product_l1556_155697


namespace NUMINAMATH_CALUDE_star_equation_solution_l1556_155615

/-- Define the ⋆ operation -/
def star (a b : ℝ) : ℝ := 3 * a - 2 * b^2

/-- Theorem: If a ⋆ 4 = 17, then a = 49/3 -/
theorem star_equation_solution (a : ℝ) (h : star a 4 = 17) : a = 49/3 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l1556_155615


namespace NUMINAMATH_CALUDE_unique_modulo_representation_l1556_155647

theorem unique_modulo_representation :
  ∃! n : ℤ, 0 ≤ n ∧ n < 7 ∧ -2222 ≡ n [ZMOD 7] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_modulo_representation_l1556_155647


namespace NUMINAMATH_CALUDE_letter_sum_equals_fifteen_l1556_155685

/-- Given a mapping of letters to numbers where A = 0, B = 1, C = 2, ..., Z = 25,
    prove that the sum of A + B + M + C equals 15. -/
theorem letter_sum_equals_fifteen :
  let letter_to_num : Char → ℕ := fun c => c.toNat - 65
  letter_to_num 'A' + letter_to_num 'B' + letter_to_num 'M' + letter_to_num 'C' = 15 := by
  sorry

end NUMINAMATH_CALUDE_letter_sum_equals_fifteen_l1556_155685


namespace NUMINAMATH_CALUDE_coordinate_system_proofs_l1556_155675

-- Define the points A, B, and C
def A (a : ℝ) : ℝ × ℝ := (-2, a + 1)
def B (a : ℝ) : ℝ × ℝ := (a - 1, 4)
def C (b : ℝ) : ℝ × ℝ := (b - 2, b)

-- Define the conditions and prove the statements
theorem coordinate_system_proofs :
  -- 1. When C is on the x-axis, its coordinates are (-2,0)
  (∃ b : ℝ, C b = (-2, 0) ∧ (C b).2 = 0) ∧
  -- 2. When C is on the y-axis, its coordinates are (0,2)
  (∃ b : ℝ, C b = (0, 2) ∧ (C b).1 = 0) ∧
  -- 3. When AB is parallel to the x-axis, the distance between A and B is 4
  (∃ a : ℝ, (A a).2 = (B a).2 ∧ Real.sqrt ((A a).1 - (B a).1)^2 = 4) ∧
  -- 4. When CD is perpendicular to the x-axis at point D and CD=1, 
  --    the coordinates of C are either (-1,1) or (-3,-1)
  (∃ b d : ℝ, (C b).1 = d ∧ Real.sqrt ((C b).1 - d)^2 + (C b).2^2 = 1 ∧
    ((C b = (-1, 1)) ∨ (C b = (-3, -1)))) :=
by sorry

end NUMINAMATH_CALUDE_coordinate_system_proofs_l1556_155675


namespace NUMINAMATH_CALUDE_difference_of_squares_l1556_155680

theorem difference_of_squares : 65^2 - 55^2 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1556_155680


namespace NUMINAMATH_CALUDE_time_to_go_up_mountain_l1556_155646

/-- Represents the hiking trip with given parameters. -/
structure HikingTrip where
  rate_up : ℝ
  rate_down : ℝ
  distance_down : ℝ
  time_up : ℝ
  time_down : ℝ

/-- The hiking trip satisfies the given conditions. -/
def satisfies_conditions (trip : HikingTrip) : Prop :=
  trip.time_up = trip.time_down ∧
  trip.rate_down = 1.5 * trip.rate_up ∧
  trip.rate_up = 5 ∧
  trip.distance_down = 15

/-- Theorem stating that for a trip satisfying the conditions, 
    the time to go up the mountain is 2 days. -/
theorem time_to_go_up_mountain (trip : HikingTrip) 
  (h : satisfies_conditions trip) : trip.time_up = 2 := by
  sorry


end NUMINAMATH_CALUDE_time_to_go_up_mountain_l1556_155646


namespace NUMINAMATH_CALUDE_order_relation_abc_l1556_155648

/-- Prove that given a = (4 - ln 4) / e^2, b = ln 2 / 2, and c = 1/e, we have b < a < c -/
theorem order_relation_abc :
  let a : ℝ := (4 - Real.log 4) / Real.exp 2
  let b : ℝ := Real.log 2 / 2
  let c : ℝ := 1 / Real.exp 1
  b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_order_relation_abc_l1556_155648


namespace NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_l1556_155613

theorem tan_theta_in_terms_of_x (θ : Real) (x : Real) 
  (h_acute : 0 < θ ∧ θ < π / 2)
  (h_x : x > 1)
  (h_sin : Real.sin (θ / 2) = Real.sqrt ((x + 1) / (2 * x))) :
  Real.tan θ = -Real.sqrt (x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_l1556_155613


namespace NUMINAMATH_CALUDE_sequence_properties_l1556_155678

-- Define the sequence type
def Sequence := ℕ → ℝ

-- Define arithmetic progression
def is_arithmetic_progression (a : Sequence) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define geometric progression
def is_geometric_progression (a : Sequence) :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem sequence_properties (a : Sequence) 
  (h1 : a 4 + a 7 = 2) 
  (h2 : a 5 * a 6 = -8) :
  (is_arithmetic_progression a → a 1 * a 10 = -728) ∧
  (is_geometric_progression a → a 1 + a 10 = -7) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l1556_155678


namespace NUMINAMATH_CALUDE_complex_expression_equality_l1556_155634

theorem complex_expression_equality : 
  let M := (Real.sqrt (Real.sqrt 7 + 3) + Real.sqrt (Real.sqrt 7 - 3)) / Real.sqrt (Real.sqrt 7 + 2) - Real.sqrt (5 - 2 * Real.sqrt 6)
  M = (1 + Real.sqrt 3 + Real.sqrt 5 + 3 * Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l1556_155634


namespace NUMINAMATH_CALUDE_magic_box_solution_l1556_155631

def magic_box (a b : ℝ) : ℝ := a^2 + 2*b - 3

theorem magic_box_solution (m : ℝ) : 
  magic_box m (-3*m) = 4 ↔ m = 7 ∨ m = -1 := by sorry

end NUMINAMATH_CALUDE_magic_box_solution_l1556_155631


namespace NUMINAMATH_CALUDE_quadratic_one_root_l1556_155682

theorem quadratic_one_root (m : ℝ) : 
  (∃! x : ℝ, x^2 + 6*m*x + m = 0) ↔ m = 1/9 ∧ m > 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l1556_155682


namespace NUMINAMATH_CALUDE_sam_yellow_marbles_l1556_155699

theorem sam_yellow_marbles (initial_yellow : ℝ) (received_yellow : ℝ) 
  (h1 : initial_yellow = 86.0) (h2 : received_yellow = 25.0) :
  initial_yellow + received_yellow = 111.0 := by
  sorry

end NUMINAMATH_CALUDE_sam_yellow_marbles_l1556_155699


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l1556_155609

def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_parallel_points :
  ∀ x y : ℝ, (f x = y ∧ (3 * x^2 + 1 = 4)) ↔ ((x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_l1556_155609


namespace NUMINAMATH_CALUDE_sum_of_divisors_450_prime_factors_and_gcd_l1556_155603

def sumOfDivisors (n : ℕ) : ℕ := sorry

def numberOfDistinctPrimeFactors (n : ℕ) : ℕ := sorry

theorem sum_of_divisors_450_prime_factors_and_gcd :
  let s := sumOfDivisors 450
  numberOfDistinctPrimeFactors s = 3 ∧ Nat.gcd s 450 = 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_450_prime_factors_and_gcd_l1556_155603


namespace NUMINAMATH_CALUDE_solution_set_of_equation_l1556_155668

theorem solution_set_of_equation : 
  ∃ (S : Set ℂ), S = {6, 2, 4 + 2*I, 4 - 2*I} ∧ 
  ∀ x : ℂ, (x - 2)^4 + (x - 6)^4 = 272 ↔ x ∈ S :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_equation_l1556_155668


namespace NUMINAMATH_CALUDE_point_on_line_implies_a_equals_negative_eight_l1556_155664

/-- A point (a, 0) lies on the line y = x + 8 -/
def point_on_line (a : ℝ) : Prop :=
  0 = a + 8

/-- Theorem: If (a, 0) lies on the line y = x + 8, then a = -8 -/
theorem point_on_line_implies_a_equals_negative_eight (a : ℝ) :
  point_on_line a → a = -8 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_implies_a_equals_negative_eight_l1556_155664


namespace NUMINAMATH_CALUDE_amanda_savings_l1556_155666

/-- The cost of a single lighter at the gas station in dollars -/
def gas_station_price : ℚ := 175 / 100

/-- The cost of a pack of 12 lighters online in dollars -/
def online_pack_price : ℚ := 5

/-- The number of lighters in each online pack -/
def lighters_per_pack : ℕ := 12

/-- The number of lighters Amanda wants to buy -/
def lighters_to_buy : ℕ := 24

/-- The savings Amanda would have by buying online instead of at the gas station -/
theorem amanda_savings : 
  (lighters_to_buy : ℚ) * gas_station_price - 
  (lighters_to_buy / lighters_per_pack : ℚ) * online_pack_price = 32 := by
  sorry

end NUMINAMATH_CALUDE_amanda_savings_l1556_155666


namespace NUMINAMATH_CALUDE_symmetric_line_ratio_l1556_155684

-- Define the triangle ABC and points M, N
structure Triangle :=
  (A B C M N : ℝ × ℝ)

-- Define the property of AM and AN being symmetric with respect to angle bisector of A
def isSymmetric (t : Triangle) : Prop :=
  -- This is a placeholder for the actual geometric condition
  sorry

-- Define the lengths of sides and segments
def length (p q : ℝ × ℝ) : ℝ :=
  sorry

-- State the theorem
theorem symmetric_line_ratio (t : Triangle) :
  isSymmetric t →
  (length t.B t.M * length t.B t.N) / (length t.C t.M * length t.C t.N) =
  (length t.A t.C)^2 / (length t.A t.B)^2 :=
by
  sorry

end NUMINAMATH_CALUDE_symmetric_line_ratio_l1556_155684


namespace NUMINAMATH_CALUDE_xiao_ying_final_grade_l1556_155628

/-- Represents a student's physical education grade components and scores -/
structure PhysEdGrade where
  regular_activity_weight : Real
  theory_test_weight : Real
  skills_test_weight : Real
  regular_activity_score : Real
  theory_test_score : Real
  skills_test_score : Real

/-- Calculates the final physical education grade -/
def calculate_final_grade (grade : PhysEdGrade) : Real :=
  grade.regular_activity_weight * grade.regular_activity_score +
  grade.theory_test_weight * grade.theory_test_score +
  grade.skills_test_weight * grade.skills_test_score

/-- Xiao Ying's physical education grade components and scores -/
def xiao_ying_grade : PhysEdGrade :=
  { regular_activity_weight := 0.3
    theory_test_weight := 0.2
    skills_test_weight := 0.5
    regular_activity_score := 90
    theory_test_score := 80
    skills_test_score := 94 }

/-- Theorem: Xiao Ying's final physical education grade is 90 points -/
theorem xiao_ying_final_grade :
  calculate_final_grade xiao_ying_grade = 90 := by
  sorry

end NUMINAMATH_CALUDE_xiao_ying_final_grade_l1556_155628


namespace NUMINAMATH_CALUDE_preimage_of_two_three_l1556_155667

/-- Given a mapping f : ℝ × ℝ → ℝ × ℝ defined by f(x, y) = (x+y, x-y),
    prove that f(5/2, -1/2) = (2, 3) -/
theorem preimage_of_two_three (f : ℝ × ℝ → ℝ × ℝ) 
    (h : ∀ x y : ℝ, f (x, y) = (x + y, x - y)) : 
    f (5/2, -1/2) = (2, 3) := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_two_three_l1556_155667


namespace NUMINAMATH_CALUDE_triangle_geometric_sequence_cosine_l1556_155658

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a, b, c form a geometric sequence and c = 2a, then cos B = 1/√2 -/
theorem triangle_geometric_sequence_cosine (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Ensure positive side lengths
  A > 0 ∧ B > 0 ∧ C > 0 →  -- Ensure positive angles
  A + B + C = π →  -- Sum of angles in a triangle
  c^2 = a^2 + b^2 - 2*a*b*Real.cos C →  -- Law of cosines
  (∃ r : ℝ, b = a*r ∧ c = b*r) →  -- Geometric sequence condition
  c = 2*a →  -- Given condition
  Real.cos B = 1 / Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_geometric_sequence_cosine_l1556_155658


namespace NUMINAMATH_CALUDE_power_sum_equality_l1556_155654

theorem power_sum_equality : (-1)^53 + 3^(2^3 + 5^2 - 7^2) = -43046720 / 43046721 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l1556_155654


namespace NUMINAMATH_CALUDE_prob_odd_sum_coin_dice_prob_odd_sum_coin_dice_is_seven_sixteenths_l1556_155604

def coin_toss : Type := Bool
def die_roll : Type := Fin 6

def is_head (c : coin_toss) : Prop := c = true
def is_tail (c : coin_toss) : Prop := c = false

def sum_is_odd (rolls : List ℕ) : Prop := (rolls.sum % 2 = 1)

def prob_head : ℚ := 1/2
def prob_tail : ℚ := 1/2

def prob_odd_sum_two_dice : ℚ := 1/2

theorem prob_odd_sum_coin_dice : ℚ :=
  let p_0_head := prob_tail^3
  let p_1_head := 3 * prob_head * prob_tail^2
  let p_2_head := 3 * prob_head^2 * prob_tail
  let p_3_head := prob_head^3

  let p_odd_0_dice := 0
  let p_odd_2_dice := prob_odd_sum_two_dice
  let p_odd_4_dice := 1/2
  let p_odd_6_dice := 1/2

  p_0_head * p_odd_0_dice +
  p_1_head * p_odd_2_dice +
  p_2_head * p_odd_4_dice +
  p_3_head * p_odd_6_dice

theorem prob_odd_sum_coin_dice_is_seven_sixteenths :
  prob_odd_sum_coin_dice = 7/16 := by sorry

end NUMINAMATH_CALUDE_prob_odd_sum_coin_dice_prob_odd_sum_coin_dice_is_seven_sixteenths_l1556_155604


namespace NUMINAMATH_CALUDE_triangle_with_angle_ratio_1_2_3_is_right_triangle_l1556_155622

theorem triangle_with_angle_ratio_1_2_3_is_right_triangle (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b + c = 180 →
  b = 2 * a →
  c = 3 * a →
  c = 90 :=
sorry

end NUMINAMATH_CALUDE_triangle_with_angle_ratio_1_2_3_is_right_triangle_l1556_155622


namespace NUMINAMATH_CALUDE_max_cos_sin_sum_l1556_155688

open Real

theorem max_cos_sin_sum (α β γ : ℝ) (h1 : 0 < α ∧ α < π)
                                   (h2 : 0 < β ∧ β < π)
                                   (h3 : 0 < γ ∧ γ < π)
                                   (h4 : α + β + 2 * γ = π) :
  (∀ a b c, 0 < a ∧ a < π ∧ 0 < b ∧ b < π ∧ 0 < c ∧ c < π ∧ a + b + 2 * c = π →
    cos α + cos β + sin (2 * γ) ≥ cos a + cos b + sin (2 * c)) ∧
  cos α + cos β + sin (2 * γ) = 3 * sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_max_cos_sin_sum_l1556_155688


namespace NUMINAMATH_CALUDE_naval_formation_arrangements_l1556_155605

/-- The number of ways to arrange 2 submarines one in front of the other -/
def submarine_arrangements : ℕ := 2

/-- The number of ways to arrange 6 ships in two groups of 3 -/
def ship_arrangements : ℕ := 720

/-- The number of invalid arrangements where all ships on one side are of the same type -/
def invalid_arrangements : ℕ := 2 * 2

/-- The total number of valid arrangements -/
def total_arrangements : ℕ := submarine_arrangements * (ship_arrangements - invalid_arrangements)

theorem naval_formation_arrangements : total_arrangements = 1296 := by
  sorry

end NUMINAMATH_CALUDE_naval_formation_arrangements_l1556_155605


namespace NUMINAMATH_CALUDE_double_reflection_result_l1556_155662

/-- Reflects a point about the line y = x -/
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

/-- Reflects a point about the line y = -x -/
def reflect_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

/-- The initial point -/
def initial_point : ℝ × ℝ := (3, -8)

theorem double_reflection_result :
  (reflect_y_eq_neg_x ∘ reflect_y_eq_x) initial_point = (-3, 8) := by
sorry

end NUMINAMATH_CALUDE_double_reflection_result_l1556_155662


namespace NUMINAMATH_CALUDE_chord_length_l1556_155608

/-- The length of the chord formed by the intersection of a line and a circle -/
theorem chord_length (l : Real → Real × Real) (C₁ : Real → Real × Real) : 
  (∀ t, l t = (1 + 3/5 * t, 4/5 * t)) →
  (∀ θ, C₁ θ = (Real.cos θ, Real.sin θ)) →
  (∃ A B, A ≠ B ∧ (∃ t₁ t₂, l t₁ = A ∧ l t₂ = B) ∧ (∃ θ₁ θ₂, C₁ θ₁ = A ∧ C₁ θ₂ = B)) →
  ∃ A B, A ≠ B ∧ (∃ t₁ t₂, l t₁ = A ∧ l t₂ = B) ∧ (∃ θ₁ θ₂, C₁ θ₁ = A ∧ C₁ θ₂ = B) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 6/5 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_l1556_155608


namespace NUMINAMATH_CALUDE_johns_candy_cost_l1556_155617

/-- The amount John pays for candy bars after sharing the cost with Dave -/
def johnsPay (totalBars : ℕ) (daveBars : ℕ) (originalPrice : ℚ) (discountRate : ℚ) : ℚ :=
  let discountedPrice := originalPrice * (1 - discountRate)
  let totalCost := totalBars * discountedPrice
  let johnBars := totalBars - daveBars
  johnBars * discountedPrice

/-- Theorem stating that John pays $11.20 for his share of the candy bars -/
theorem johns_candy_cost :
  johnsPay 20 6 1 (20 / 100) = 11.2 := by
  sorry

end NUMINAMATH_CALUDE_johns_candy_cost_l1556_155617


namespace NUMINAMATH_CALUDE_open_box_volume_calculation_l1556_155625

/-- Given a rectangular sheet and squares cut from corners, calculates the volume of the resulting open box. -/
def openBoxVolume (sheetLength sheetWidth squareSide : ℝ) : ℝ :=
  (sheetLength - 2 * squareSide) * (sheetWidth - 2 * squareSide) * squareSide

/-- Theorem: The volume of the open box formed from a 48m x 36m sheet with 5m squares cut from corners is 9880 m³. -/
theorem open_box_volume_calculation :
  openBoxVolume 48 36 5 = 9880 := by
  sorry

#eval openBoxVolume 48 36 5

end NUMINAMATH_CALUDE_open_box_volume_calculation_l1556_155625


namespace NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l1556_155665

def f (x : ℝ) : ℝ := x^5 - 8*x^4 + 12*x^3 + 20*x^2 - 19*x - 24

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - a) * q x + f a := sorry

theorem polynomial_remainder :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - 5) * q x + 1012 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l1556_155665


namespace NUMINAMATH_CALUDE_male_salmon_count_l1556_155672

def total_salmon : ℕ := 971639
def female_salmon : ℕ := 259378

theorem male_salmon_count : total_salmon - female_salmon = 712261 := by
  sorry

end NUMINAMATH_CALUDE_male_salmon_count_l1556_155672


namespace NUMINAMATH_CALUDE_sons_age_next_year_l1556_155645

/-- Given a father who is 35 years old and whose age is five times that of his son,
    prove that the son's age next year will be 8 years. -/
theorem sons_age_next_year (father_age : ℕ) (son_age : ℕ) : 
  father_age = 35 → father_age = 5 * son_age → son_age + 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sons_age_next_year_l1556_155645


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_500_l1556_155673

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_sum_factorials_500 :
  units_digit (sum_factorials 500) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_500_l1556_155673


namespace NUMINAMATH_CALUDE_find_a_and_b_l1556_155611

-- Define the sets A and B
def A : Set ℝ := {x | x^3 + 3*x^2 + 2*x > 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

-- State the theorem
theorem find_a_and_b :
  ∃ (a b : ℝ),
    (A ∩ B a b = {x | 0 < x ∧ x ≤ 2}) ∧
    (A ∪ B a b = {x | x > -2}) ∧
    a = -1 ∧
    b = -2 :=
by sorry

end NUMINAMATH_CALUDE_find_a_and_b_l1556_155611


namespace NUMINAMATH_CALUDE_number_problem_l1556_155674

theorem number_problem : 
  ∃ x : ℝ, x - (3/5) * x = 50 → x = 125 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1556_155674


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_iff_all_zero_l1556_155624

theorem sum_of_squares_zero_iff_all_zero (a b c : ℝ) :
  a^2 + b^2 + c^2 = 0 ↔ a = 0 ∧ b = 0 ∧ c = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_iff_all_zero_l1556_155624


namespace NUMINAMATH_CALUDE_arithmetic_computation_l1556_155660

theorem arithmetic_computation : 1325 + 572 / 52 - 225 + 2^3 = 1119 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l1556_155660


namespace NUMINAMATH_CALUDE_crows_left_on_branch_l1556_155676

/-- The number of crows remaining on a tree branch after some birds flew away -/
def remaining_crows (initial_parrots initial_total initial_crows remaining_parrots : ℕ) : ℕ :=
  initial_crows - (initial_parrots - remaining_parrots)

/-- Theorem stating the number of crows remaining on the branch -/
theorem crows_left_on_branch :
  ∀ (initial_parrots initial_total initial_crows remaining_parrots : ℕ),
    initial_parrots = 7 →
    initial_total = 13 →
    initial_crows = initial_total - initial_parrots →
    remaining_parrots = 2 →
    remaining_crows initial_parrots initial_total initial_crows remaining_parrots = 1 := by
  sorry

#eval remaining_crows 7 13 6 2

end NUMINAMATH_CALUDE_crows_left_on_branch_l1556_155676


namespace NUMINAMATH_CALUDE_monotonic_increase_interval_l1556_155641

theorem monotonic_increase_interval
  (f : ℝ → ℝ)
  (φ : ℝ)
  (h1 : ∀ x, f x = Real.sin (2 * x + φ))
  (h2 : ∀ x, f x ≤ |f (π / 6)|)
  (h3 : f (π / 2) > f π) :
  ∀ k : ℤ, StrictMonoOn f (Set.Icc (k * π + π / 6) (k * π + 2 * π / 3)) :=
by sorry

end NUMINAMATH_CALUDE_monotonic_increase_interval_l1556_155641


namespace NUMINAMATH_CALUDE_f_uniquely_determined_l1556_155656

/-- A function from ℝ² to ℝ² defined as f(x, y) = (kx, y + b) -/
def f (k b : ℝ) : ℝ × ℝ → ℝ × ℝ := fun (x, y) ↦ (k * x, y + b)

/-- Theorem: If f(3, 1) = (6, 2), then k = 2 and b = 1 -/
theorem f_uniquely_determined (k b : ℝ) : 
  f k b (3, 1) = (6, 2) → k = 2 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_uniquely_determined_l1556_155656


namespace NUMINAMATH_CALUDE_min_points_to_guarantee_win_no_smaller_guarantee_l1556_155681

/-- Represents the points earned in a single race -/
inductive RaceResult
| First  : RaceResult
| Second : RaceResult
| Third  : RaceResult

/-- Converts a race result to points -/
def points (result : RaceResult) : Nat :=
  match result with
  | RaceResult.First  => 6
  | RaceResult.Second => 4
  | RaceResult.Third  => 2

/-- Calculates the total points from a list of race results -/
def totalPoints (results : List RaceResult) : Nat :=
  results.map points |>.sum

/-- Represents the results of three races -/
def ThreeRaces := (RaceResult × RaceResult × RaceResult)

/-- Theorem: 16 points is the minimum to guarantee winning -/
theorem min_points_to_guarantee_win :
  ∀ (other : ThreeRaces),
  ∃ (winner : ThreeRaces),
    totalPoints (winner.1 :: winner.2.1 :: [winner.2.2]) = 16 ∧
    totalPoints (winner.1 :: winner.2.1 :: [winner.2.2]) >
    totalPoints (other.1 :: other.2.1 :: [other.2.2]) :=
  sorry

/-- Theorem: No smaller number of points can guarantee winning -/
theorem no_smaller_guarantee :
  ∀ (n : Nat),
  n < 16 →
  ∃ (player1 player2 : ThreeRaces),
    totalPoints (player1.1 :: player1.2.1 :: [player1.2.2]) = n ∧
    totalPoints (player2.1 :: player2.2.1 :: [player2.2.2]) ≥ n :=
  sorry

end NUMINAMATH_CALUDE_min_points_to_guarantee_win_no_smaller_guarantee_l1556_155681


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1556_155635

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y : ℝ, x > 1 ∧ y > 1 → x + y > 2) ∧
  (∃ x y : ℝ, x + y > 2 ∧ ¬(x > 1 ∧ y > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1556_155635


namespace NUMINAMATH_CALUDE_mrs_thompson_potatoes_cost_l1556_155692

/-- Calculates the cost of potatoes given the number of chickens, cost per chicken, and total amount paid. -/
def cost_of_potatoes (num_chickens : ℕ) (cost_per_chicken : ℕ) (total_paid : ℕ) : ℕ :=
  total_paid - (num_chickens * cost_per_chicken)

/-- Proves that the cost of potatoes is 6 given the specific conditions of Mrs. Thompson's purchase. -/
theorem mrs_thompson_potatoes_cost :
  cost_of_potatoes 3 3 15 = 6 := by
  sorry

end NUMINAMATH_CALUDE_mrs_thompson_potatoes_cost_l1556_155692


namespace NUMINAMATH_CALUDE_geometric_sequence_quadratic_roots_l1556_155632

/-- Given real numbers 2, b, and a form a geometric sequence, 
    the equation ax^2 + bx + 1/3 = 0 has exactly 2 real roots -/
theorem geometric_sequence_quadratic_roots 
  (b a : ℝ) 
  (h_geometric : ∃ (q : ℝ), b = 2 * q ∧ a = 2 * q^2) :
  (∃! (x y : ℝ), x ≠ y ∧ 
    (∀ (z : ℝ), a * z^2 + b * z + 1/3 = 0 ↔ z = x ∨ z = y)) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_quadratic_roots_l1556_155632


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1556_155698

theorem absolute_value_inequality (x : ℝ) :
  (|x - 1| + |x + 2| ≤ 5) ↔ (x ∈ Set.Icc (-3) 2) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1556_155698


namespace NUMINAMATH_CALUDE_temporary_employee_percentage_l1556_155695

theorem temporary_employee_percentage 
  (total_workers : ℕ) 
  (technicians : ℕ) 
  (non_technicians : ℕ) 
  (permanent_technicians : ℕ) 
  (permanent_non_technicians : ℕ) 
  (h1 : technicians = total_workers / 2) 
  (h2 : non_technicians = total_workers / 2) 
  (h3 : permanent_technicians = technicians / 2) 
  (h4 : permanent_non_technicians = non_technicians / 2) :
  (total_workers - (permanent_technicians + permanent_non_technicians)) * 100 / total_workers = 50 := by
  sorry

end NUMINAMATH_CALUDE_temporary_employee_percentage_l1556_155695


namespace NUMINAMATH_CALUDE_incenter_distance_l1556_155610

/-- Represents a triangle with vertices P, Q, and R -/
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  isosceles : dist P Q = dist P R
  pq_length : dist P Q = 17
  qr_length : dist Q R = 16

/-- Represents the incenter of a triangle -/
def incenter (t : Triangle P Q R) : ℝ × ℝ := sorry

/-- Represents the incircle of a triangle -/
def incircle (t : Triangle P Q R) : Set (ℝ × ℝ) := sorry

/-- Represents a point where the incircle touches a side of the triangle -/
def touchPoint (t : Triangle P Q R) (side : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ × ℝ := sorry

theorem incenter_distance (P Q R : ℝ × ℝ) (t : Triangle P Q R) :
  let J := incenter t
  let C := touchPoint t (Q, R)
  dist C J = Real.sqrt 87.04 := by sorry

end NUMINAMATH_CALUDE_incenter_distance_l1556_155610


namespace NUMINAMATH_CALUDE_games_last_year_l1556_155630

/-- The number of basketball games Fred attended this year -/
def games_this_year : ℕ := 25

/-- The difference in games attended between last year and this year -/
def games_difference : ℕ := 11

/-- Theorem stating the number of games Fred attended last year -/
theorem games_last_year : games_this_year + games_difference = 36 := by
  sorry

end NUMINAMATH_CALUDE_games_last_year_l1556_155630


namespace NUMINAMATH_CALUDE_correct_polynomial_result_l1556_155602

/-- Given a polynomial P, prove that if subtracting P from a^2 - 5a + 7 results in 2a^2 - 3a + 5,
    then adding P to 2a^2 - 3a + 5 yields 5a^2 - 11a + 17. -/
theorem correct_polynomial_result (P : Polynomial ℚ) : 
  (a^2 - 5*a + 7 : Polynomial ℚ) - P = 2*a^2 - 3*a + 5 →
  P + (2*a^2 - 3*a + 5 : Polynomial ℚ) = 5*a^2 - 11*a + 17 := by
  sorry

end NUMINAMATH_CALUDE_correct_polynomial_result_l1556_155602


namespace NUMINAMATH_CALUDE_train_passing_time_l1556_155670

/-- The time taken for a train to pass a stationary point -/
theorem train_passing_time (length : ℝ) (speed_kmh : ℝ) : 
  length = 280 → speed_kmh = 36 → 
  (length / (speed_kmh * 1000 / 3600)) = 28 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l1556_155670


namespace NUMINAMATH_CALUDE_initial_plums_count_l1556_155651

/-- The number of plums Melanie picked initially -/
def initial_plums : ℕ := sorry

/-- The number of plums Melanie gave to Sam -/
def plums_given : ℕ := 3

/-- The number of plums Melanie has left -/
def plums_left : ℕ := 4

/-- Theorem stating that the initial number of plums equals 7 -/
theorem initial_plums_count : initial_plums = 7 := by sorry

end NUMINAMATH_CALUDE_initial_plums_count_l1556_155651


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1556_155661

theorem trigonometric_identities (α : Real) 
  (h1 : (Real.tan α) / (Real.tan α - 1) = -1)
  (h2 : α ∈ Set.Icc (Real.pi) (3 * Real.pi / 2)) :
  (((Real.sin α - 3 * Real.cos α) / (Real.sin α + Real.cos α)) = -5/3) ∧
  ((Real.cos (-Real.pi + α) + Real.cos (Real.pi/2 + α)) = 3 * Real.sqrt 5 / 5) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1556_155661


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l1556_155686

/-- Given 2x + 3y + 5z = 29, the maximum value of √(2x+1) + √(3y+4) + √(5z+6) is 2√30 -/
theorem max_value_sqrt_sum (x y z : ℝ) (h : 2*x + 3*y + 5*z = 29) :
  (∀ a b c : ℝ, 2*a + 3*b + 5*c = 29 →
    Real.sqrt (2*a + 1) + Real.sqrt (3*b + 4) + Real.sqrt (5*c + 6) ≤
    Real.sqrt (2*x + 1) + Real.sqrt (3*y + 4) + Real.sqrt (5*z + 6)) →
  Real.sqrt (2*x + 1) + Real.sqrt (3*y + 4) + Real.sqrt (5*z + 6) = 2 * Real.sqrt 30 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l1556_155686


namespace NUMINAMATH_CALUDE_log_inequality_l1556_155652

theorem log_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  Real.log (1 + x + y) < x + y := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l1556_155652


namespace NUMINAMATH_CALUDE_isabel_weekly_distance_l1556_155690

/-- Calculates the total distance run in a week given a circuit length, 
    number of morning and afternoon runs, and number of days in a week. -/
def total_weekly_distance (circuit_length : ℕ) (morning_runs : ℕ) (afternoon_runs : ℕ) (days_in_week : ℕ) : ℕ :=
  (circuit_length * (morning_runs + afternoon_runs) * days_in_week)

/-- Theorem stating that running a 365-meter circuit 7 times in the morning and 3 times in the afternoon
    for 7 days results in a total distance of 25550 meters. -/
theorem isabel_weekly_distance :
  total_weekly_distance 365 7 3 7 = 25550 := by
  sorry

end NUMINAMATH_CALUDE_isabel_weekly_distance_l1556_155690


namespace NUMINAMATH_CALUDE_count_satisfying_pairs_l1556_155629

def satisfies_inequalities (a b : ℤ) : Prop :=
  (a^2 + b^2 < 25) ∧ (a^2 + b^2 < 10*a) ∧ (a^2 + b^2 < 10*b)

theorem count_satisfying_pairs :
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (p : ℤ × ℤ), p ∈ s ↔ satisfies_inequalities p.1 p.2) ∧
    s.card = 8 := by
  sorry

end NUMINAMATH_CALUDE_count_satisfying_pairs_l1556_155629


namespace NUMINAMATH_CALUDE_product_xyz_l1556_155642

theorem product_xyz (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x * (y + z) = 198)
  (eq2 : y * (z + x) = 216)
  (eq3 : z * (x + y) = 234) :
  x * y * z = 1080 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_l1556_155642


namespace NUMINAMATH_CALUDE_min_value_theorem_l1556_155691

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 4 ∧
  (∀ (c d : ℝ), c > 0 → d > 0 → c + d = 4 → 1 / (c + 1) + 1 / (d + 3) ≥ 1 / (x + 1) + 1 / (y + 3)) ∧
  1 / (x + 1) + 1 / (y + 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1556_155691


namespace NUMINAMATH_CALUDE_correct_probability_distribution_l1556_155669

/-- Represents the number of students -/
def num_students : ℕ := 4

/-- Represents the number of cookie types -/
def num_cookie_types : ℕ := 3

/-- Represents the total number of cookies -/
def total_cookies : ℕ := num_students * num_cookie_types

/-- Represents the number of cookies of each type -/
def cookies_per_type : ℕ := num_students

/-- Calculates the probability of each student receiving one cookie of each type -/
def probability_all_students_correct_distribution : ℚ :=
  144 / 3850

/-- Theorem stating that the calculated probability is correct -/
theorem correct_probability_distribution :
  probability_all_students_correct_distribution = 144 / 3850 := by
  sorry

end NUMINAMATH_CALUDE_correct_probability_distribution_l1556_155669


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1556_155612

theorem trigonometric_identity (x y : ℝ) : 
  Real.sin (x + y) * Real.sin x + Real.cos (x + y) * Real.cos x = Real.cos y := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1556_155612


namespace NUMINAMATH_CALUDE_unit_price_sum_l1556_155679

theorem unit_price_sum (x y z : ℝ) 
  (eq1 : 3 * x + 7 * y + z = 24)
  (eq2 : 4 * x + 10 * y + z = 33) : 
  x + y + z = 6 := by
sorry

end NUMINAMATH_CALUDE_unit_price_sum_l1556_155679


namespace NUMINAMATH_CALUDE_apples_in_baskets_l1556_155626

theorem apples_in_baskets (total_apples : ℕ) (num_baskets : ℕ) (removed_apples : ℕ) 
  (h1 : total_apples = 64)
  (h2 : num_baskets = 4)
  (h3 : removed_apples = 3)
  : (total_apples / num_baskets) - removed_apples = 13 := by
  sorry

end NUMINAMATH_CALUDE_apples_in_baskets_l1556_155626


namespace NUMINAMATH_CALUDE_infinite_series_sum_l1556_155616

/-- The sum of the infinite series ∑(n=1 to ∞) (3n - 2) / (n(n + 1)(n + 3)) is equal to 7/12. -/
theorem infinite_series_sum : 
  ∑' (n : ℕ), (3 * n - 2 : ℝ) / (n * (n + 1) * (n + 3)) = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l1556_155616


namespace NUMINAMATH_CALUDE_honey_tax_calculation_l1556_155606

/-- Represents the tax per pound of honey -/
def tax_per_pound : ℝ := 1

theorem honey_tax_calculation 
  (bulk_price : ℝ) 
  (minimum_spend : ℝ) 
  (total_paid : ℝ) 
  (excess_pounds : ℝ) 
  (h1 : bulk_price = 5)
  (h2 : minimum_spend = 40)
  (h3 : total_paid = 240)
  (h4 : excess_pounds = 32)
  : tax_per_pound = 1 := by
  sorry

#check honey_tax_calculation

end NUMINAMATH_CALUDE_honey_tax_calculation_l1556_155606


namespace NUMINAMATH_CALUDE_complex_triplet_theorem_l1556_155689

theorem complex_triplet_theorem (a b c : ℂ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → 
  Complex.abs a = Complex.abs b → 
  Complex.abs b = Complex.abs c → 
  a / b + b / c + c / a = -1 → 
  ((a = b ∧ c = -a) ∨ (b = c ∧ a = -b) ∨ (c = a ∧ b = -c)) := by
sorry

end NUMINAMATH_CALUDE_complex_triplet_theorem_l1556_155689


namespace NUMINAMATH_CALUDE_no_common_solution_l1556_155607

theorem no_common_solution : ¬∃ (x y : ℝ), x^2 + y^2 = 25 ∧ x^2 + 3*y = 45 := by
  sorry

end NUMINAMATH_CALUDE_no_common_solution_l1556_155607


namespace NUMINAMATH_CALUDE_rectangle_problem_l1556_155650

/-- Given three rectangles with equal areas and integer sides, where one side is 31,
    the length of a side perpendicular to the side of length 31 is 992. -/
theorem rectangle_problem (a b : ℕ) (h1 : a > 0) (h2 : b > 0) : 
  (a * 31 = b * (992 : ℕ)) ∧ (∃ k l : ℕ, k * l = 31 * (k + l) ∧ k = 992) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_problem_l1556_155650


namespace NUMINAMATH_CALUDE_exists_field_trip_with_frequent_participants_l1556_155623

/-- Represents a field trip -/
structure FieldTrip where
  participants : Finset (Fin 20)
  at_least_four : participants.card ≥ 4

/-- Represents the collection of all field trips -/
structure FieldTrips where
  trips : Finset FieldTrip
  nonempty : trips.Nonempty

theorem exists_field_trip_with_frequent_participants (ft : FieldTrips) :
  ∃ (trip : FieldTrip), trip ∈ ft.trips ∧
    ∀ (student : Fin 20), student ∈ trip.participants →
      (ft.trips.filter (λ t : FieldTrip => student ∈ t.participants)).card ≥ ft.trips.card / 17 :=
sorry

end NUMINAMATH_CALUDE_exists_field_trip_with_frequent_participants_l1556_155623


namespace NUMINAMATH_CALUDE_half_month_days_l1556_155694

/-- Proves that given a 30-day month with specified mean profits, 
    the number of days in each half of the month is 15. -/
theorem half_month_days (total_days : ℕ) (mean_profit : ℚ) 
    (first_half_mean : ℚ) (second_half_mean : ℚ) : 
    total_days = 30 ∧ 
    mean_profit = 350 ∧ 
    first_half_mean = 225 ∧ 
    second_half_mean = 475 → 
    ∃ (first_half_days second_half_days : ℕ), 
      first_half_days = 15 ∧ 
      second_half_days = 15 ∧ 
      first_half_days + second_half_days = total_days ∧
      (first_half_mean * first_half_days + second_half_mean * second_half_days) / total_days = mean_profit :=
by sorry

end NUMINAMATH_CALUDE_half_month_days_l1556_155694


namespace NUMINAMATH_CALUDE_tech_students_count_l1556_155693

/-- Number of students in subject elective courses -/
def subject_students : ℕ → ℕ := fun m ↦ m

/-- Number of students in physical education and arts elective courses -/
def pe_arts_students : ℕ → ℕ := fun m ↦ m + 9

/-- Number of students in technology elective courses -/
def tech_students : ℕ → ℕ := fun m ↦ (pe_arts_students m) / 3 + 5

theorem tech_students_count (m : ℕ) : 
  tech_students m = m / 3 + 8 := by sorry

end NUMINAMATH_CALUDE_tech_students_count_l1556_155693


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1556_155636

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- arithmetic sequence condition
  a 5 + a 8 = 5 →                                   -- given condition
  a 2 + a 11 = 5 :=                                 -- conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1556_155636


namespace NUMINAMATH_CALUDE_back_lot_filled_fraction_l1556_155663

/-- Proves that the fraction of the back parking lot filled is 1/2 -/
theorem back_lot_filled_fraction :
  let front_spaces : ℕ := 52
  let back_spaces : ℕ := 38
  let total_spaces : ℕ := front_spaces + back_spaces
  let parked_cars : ℕ := 39
  let available_spaces : ℕ := 32
  let filled_back_spaces : ℕ := total_spaces - parked_cars - available_spaces
  (filled_back_spaces : ℚ) / back_spaces = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_back_lot_filled_fraction_l1556_155663


namespace NUMINAMATH_CALUDE_at_most_one_perfect_square_l1556_155644

theorem at_most_one_perfect_square (a : ℕ → ℤ) 
  (h : ∀ n, a (n + 1) = a n ^ 3 + 1999) : 
  ∃! k, ∃ m : ℤ, a k = m ^ 2 :=
sorry

end NUMINAMATH_CALUDE_at_most_one_perfect_square_l1556_155644


namespace NUMINAMATH_CALUDE_fraction_ratio_l1556_155627

theorem fraction_ratio (N : ℝ) (h1 : (1/3) * (2/5) * N = 14) (h2 : 0.4 * N = 168) :
  14 / ((1/3) * (2/5) * N) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ratio_l1556_155627


namespace NUMINAMATH_CALUDE_parade_average_l1556_155620

theorem parade_average (boys girls rows : ℕ) (h1 : boys = 24) (h2 : girls = 24) (h3 : rows = 6) :
  (boys + girls) / rows = 8 :=
sorry

end NUMINAMATH_CALUDE_parade_average_l1556_155620
