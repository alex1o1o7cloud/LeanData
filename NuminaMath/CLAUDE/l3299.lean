import Mathlib

namespace NUMINAMATH_CALUDE_abc_sum_l3299_329927

/-- Given prime numbers a, b, c satisfying abc + a = 851, prove a + b + c = 50 -/
theorem abc_sum (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c)
  (heq : a * b * c + a = 851) : a + b + c = 50 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_l3299_329927


namespace NUMINAMATH_CALUDE_shirt_cost_is_15_l3299_329944

/-- The cost of one pair of jeans -/
def jeans_cost : ℝ := sorry

/-- The cost of one shirt -/
def shirt_cost : ℝ := sorry

/-- The first condition: 3 pairs of jeans and 2 shirts cost $69 -/
axiom condition1 : 3 * jeans_cost + 2 * shirt_cost = 69

/-- The second condition: 2 pairs of jeans and 3 shirts cost $71 -/
axiom condition2 : 2 * jeans_cost + 3 * shirt_cost = 71

/-- Theorem: The cost of one shirt is $15 -/
theorem shirt_cost_is_15 : shirt_cost = 15 := by sorry

end NUMINAMATH_CALUDE_shirt_cost_is_15_l3299_329944


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3299_329928

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := 1 / (3 + 4 * I)
  Complex.im z = -4 / 25 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3299_329928


namespace NUMINAMATH_CALUDE_total_problems_solved_l3299_329907

def initial_problems : ℕ := 12
def additional_problems : ℕ := 7

theorem total_problems_solved :
  initial_problems + additional_problems = 19 := by
  sorry

end NUMINAMATH_CALUDE_total_problems_solved_l3299_329907


namespace NUMINAMATH_CALUDE_totalLives_eq_110_l3299_329957

/-- The total number of lives for remaining players after some quit and bonus lives are added -/
def totalLives : ℕ :=
  let initialPlayers : ℕ := 16
  let quitPlayers : ℕ := 7
  let remainingPlayers : ℕ := initialPlayers - quitPlayers
  let playersWithTenLives : ℕ := 3
  let playersWithEightLives : ℕ := 4
  let playersWithSixLives : ℕ := 2
  let bonusLives : ℕ := 4
  
  let livesBeforeBonus : ℕ := 
    playersWithTenLives * 10 + 
    playersWithEightLives * 8 + 
    playersWithSixLives * 6
  
  let totalBonusLives : ℕ := remainingPlayers * bonusLives
  
  livesBeforeBonus + totalBonusLives

theorem totalLives_eq_110 : totalLives = 110 := by
  sorry

end NUMINAMATH_CALUDE_totalLives_eq_110_l3299_329957


namespace NUMINAMATH_CALUDE_f_equals_g_l3299_329972

theorem f_equals_g (f g : ℝ → ℝ) 
  (hf_cont : Continuous f)
  (hg_mono : Monotone g)
  (h_seq : ∀ a b c : ℝ, a < b → b < c → 
    ∃ (x : ℕ → ℝ), (∀ ε > 0, ∃ N, ∀ n ≥ N, |x n - b| < ε) ∧ 
    (∃ L : ℝ, ∀ ε > 0, ∃ N, ∀ n ≥ N, |g (x n) - L| < ε) ∧
    f a < L ∧ L < f c) :
  f = g := by
sorry

end NUMINAMATH_CALUDE_f_equals_g_l3299_329972


namespace NUMINAMATH_CALUDE_napkin_length_calculation_l3299_329932

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

end NUMINAMATH_CALUDE_napkin_length_calculation_l3299_329932


namespace NUMINAMATH_CALUDE_common_number_in_overlapping_lists_l3299_329911

theorem common_number_in_overlapping_lists (numbers : List ℝ) : 
  numbers.length = 8 →
  (numbers.take 5).sum / 5 = 7 →
  (numbers.drop 5).sum / 3 = 10 →
  numbers.sum / 8 = 8 →
  ∃ x ∈ numbers.take 5 ∩ numbers.drop 5, x = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_common_number_in_overlapping_lists_l3299_329911


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3299_329969

theorem cube_volume_from_surface_area :
  ∀ s : ℝ, 6 * s^2 = 150 → s^3 = 125 := by sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3299_329969


namespace NUMINAMATH_CALUDE_square_side_length_l3299_329942

/-- Given a rectangle with length 400 feet and width 300 feet, prove that a square with perimeter
    twice that of the rectangle has a side length of 700 feet. -/
theorem square_side_length (rectangle_length : ℝ) (rectangle_width : ℝ)
    (h1 : rectangle_length = 400)
    (h2 : rectangle_width = 300)
    (square_perimeter : ℝ)
    (h3 : square_perimeter = 2 * (2 * (rectangle_length + rectangle_width)))
    (square_side : ℝ)
    (h4 : square_perimeter = 4 * square_side) :
  square_side = 700 :=
by sorry

end NUMINAMATH_CALUDE_square_side_length_l3299_329942


namespace NUMINAMATH_CALUDE_device_improvement_l3299_329951

/-- Represents the sample mean and variance of a device's measurements -/
structure DeviceStats where
  mean : ℝ
  variance : ℝ

/-- Determines if there's a significant improvement between two devices -/
def significantImprovement (old new : DeviceStats) : Prop :=
  new.mean - old.mean ≥ 2 * Real.sqrt ((old.variance + new.variance) / 10)

theorem device_improvement (old new : DeviceStats) 
  (h_old : old = ⟨10, 0.036⟩) 
  (h_new : new = ⟨10.3, 0.04⟩) : 
  significantImprovement old new := by
  sorry

#check device_improvement

end NUMINAMATH_CALUDE_device_improvement_l3299_329951


namespace NUMINAMATH_CALUDE_main_triangle_area_l3299_329913

/-- A triangle with a point inside it -/
structure TriangleWithInnerPoint where
  /-- The triangle -/
  triangle : Set (ℝ × ℝ)
  /-- The point inside the triangle -/
  inner_point : ℝ × ℝ
  /-- The point is inside the triangle -/
  point_inside : inner_point ∈ triangle

/-- The areas of smaller triangles formed by lines parallel to the sides -/
structure SmallerTriangleAreas where
  /-- The first smaller triangle area -/
  area1 : ℝ
  /-- The second smaller triangle area -/
  area2 : ℝ
  /-- The third smaller triangle area -/
  area3 : ℝ

/-- Calculate the area of the main triangle given the areas of smaller triangles -/
def calculateMainTriangleArea (smaller_areas : SmallerTriangleAreas) : ℝ :=
  sorry

/-- The theorem stating the relationship between smaller triangle areas and the main triangle area -/
theorem main_triangle_area 
  (t : TriangleWithInnerPoint) 
  (areas : SmallerTriangleAreas)
  (h1 : areas.area1 = 16)
  (h2 : areas.area2 = 25)
  (h3 : areas.area3 = 36) :
  calculateMainTriangleArea areas = 225 :=
sorry

end NUMINAMATH_CALUDE_main_triangle_area_l3299_329913


namespace NUMINAMATH_CALUDE_infinite_primes_satisfying_conditions_l3299_329934

/-- The set of odd prime numbers -/
def OddPrimes : Set Nat := {p | Nat.Prime p ∧ p % 2 = 1}

/-- The remainder of the Euclidean division of n by p -/
def d_p (p n : Nat) : Nat := n % p

/-- A p-sequence is a sequence where a_{n+1} = a_n + d_p(a_n) -/
def IsPSequence (p : Nat) (a : Nat → Nat) : Prop :=
  ∀ n, a (n + 1) = a n + d_p p (a n)

/-- The set of primes satisfying the first condition -/
def PrimesCondition1 : Set Nat :=
  {p ∈ OddPrimes | ∃ a b : Nat → Nat,
    IsPSequence p a ∧ IsPSequence p b ∧
    (∃ S1 S2 : Set Nat, S1.Infinite ∧ S2.Infinite ∧
      (∀ n ∈ S1, a n > b n) ∧ (∀ n ∈ S2, a n < b n))}

/-- The set of primes satisfying the second condition -/
def PrimesCondition2 : Set Nat :=
  {p ∈ OddPrimes | ∃ a b : Nat → Nat,
    IsPSequence p a ∧ IsPSequence p b ∧
    a 0 < b 0 ∧ (∀ n ≥ 1, a n > b n)}

theorem infinite_primes_satisfying_conditions :
  PrimesCondition1.Infinite ∧ PrimesCondition2.Infinite := by
  sorry

end NUMINAMATH_CALUDE_infinite_primes_satisfying_conditions_l3299_329934


namespace NUMINAMATH_CALUDE_cube_sum_implies_sum_l3299_329996

theorem cube_sum_implies_sum (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_implies_sum_l3299_329996


namespace NUMINAMATH_CALUDE_fraction_equality_l3299_329963

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 15)
  (h2 : p / n = 3)
  (h3 : p / q = 1 / 10) :
  m / q = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3299_329963


namespace NUMINAMATH_CALUDE_tangent_parallel_implies_a_equals_one_l3299_329984

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1

theorem tangent_parallel_implies_a_equals_one (a : ℝ) :
  (f' a 1 = 4) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_implies_a_equals_one_l3299_329984


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l3299_329971

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 5) : x^2 + 1/x^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l3299_329971


namespace NUMINAMATH_CALUDE_josie_cart_wait_time_l3299_329910

/-- Represents the shopping trip details -/
structure ShoppingTrip where
  total_time : ℕ
  shopping_time : ℕ
  wait_cabinet : ℕ
  wait_restock : ℕ
  wait_checkout : ℕ

/-- Calculates the time waited for a cart given a shopping trip -/
def time_waited_for_cart (trip : ShoppingTrip) : ℕ :=
  trip.total_time - trip.shopping_time - (trip.wait_cabinet + trip.wait_restock + trip.wait_checkout)

/-- Theorem stating that Josie waited 3 minutes for a cart -/
theorem josie_cart_wait_time :
  ∃ (trip : ShoppingTrip),
    trip.total_time = 90 ∧
    trip.shopping_time = 42 ∧
    trip.wait_cabinet = 13 ∧
    trip.wait_restock = 14 ∧
    trip.wait_checkout = 18 ∧
    time_waited_for_cart trip = 3 := by
  sorry

end NUMINAMATH_CALUDE_josie_cart_wait_time_l3299_329910


namespace NUMINAMATH_CALUDE_youngest_child_age_l3299_329935

def restaurant_problem (father_charge : ℝ) (child_charge_per_year : ℝ) (total_bill : ℝ) : Prop :=
  ∃ (twin_age youngest_age : ℕ),
    father_charge = 4.95 ∧
    child_charge_per_year = 0.45 ∧
    total_bill = 9.45 ∧
    twin_age > youngest_age ∧
    total_bill = father_charge + child_charge_per_year * (2 * twin_age + youngest_age) ∧
    youngest_age = 2

theorem youngest_child_age :
  restaurant_problem 4.95 0.45 9.45
  := by sorry

end NUMINAMATH_CALUDE_youngest_child_age_l3299_329935


namespace NUMINAMATH_CALUDE_cos_double_angle_special_case_l3299_329954

theorem cos_double_angle_special_case (α : Real) 
  (h : Real.sin (α + Real.pi / 2) = 1 / 2) : 
  Real.cos (2 * α) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_special_case_l3299_329954


namespace NUMINAMATH_CALUDE_park_area_l3299_329922

/-- The area of a rectangular park with perimeter 80 meters and length three times the width is 300 square meters. -/
theorem park_area (width length : ℝ) (h_perimeter : 2 * (width + length) = 80) (h_length : length = 3 * width) :
  width * length = 300 :=
sorry

end NUMINAMATH_CALUDE_park_area_l3299_329922


namespace NUMINAMATH_CALUDE_quadratic_roots_complex_l3299_329975

theorem quadratic_roots_complex (x : ℂ) :
  x^2 - 6*x + 25 = 0 ↔ x = 3 + 4*I ∨ x = 3 - 4*I :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_complex_l3299_329975


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3299_329995

theorem polynomial_factorization (m x : ℝ) : m * x^2 - 6 * m * x + 9 * m = m * (x - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3299_329995


namespace NUMINAMATH_CALUDE_candidate_A_votes_l3299_329926

def total_votes : ℕ := 560000
def invalid_percentage : ℚ := 15 / 100
def candidate_A_percentage : ℚ := 85 / 100

theorem candidate_A_votes : 
  ⌊(1 - invalid_percentage) * candidate_A_percentage * total_votes⌋ = 404600 := by
  sorry

end NUMINAMATH_CALUDE_candidate_A_votes_l3299_329926


namespace NUMINAMATH_CALUDE_tyler_age_l3299_329914

/-- Represents the ages of Tyler and Clay -/
structure Ages where
  tyler : ℕ
  clay : ℕ

/-- The conditions of the problem -/
def validAges (ages : Ages) : Prop :=
  ages.tyler = 3 * ages.clay + 1 ∧ ages.tyler + ages.clay = 21

/-- The theorem to prove -/
theorem tyler_age (ages : Ages) (h : validAges ages) : ages.tyler = 16 := by
  sorry

end NUMINAMATH_CALUDE_tyler_age_l3299_329914


namespace NUMINAMATH_CALUDE_expression_simplification_l3299_329908

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  ((2 * x + 1) / x - 1) / ((x^2 - 1) / x) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3299_329908


namespace NUMINAMATH_CALUDE_gcd_930_868_l3299_329920

theorem gcd_930_868 : Nat.gcd 930 868 = 62 := by
  sorry

end NUMINAMATH_CALUDE_gcd_930_868_l3299_329920


namespace NUMINAMATH_CALUDE_percentage_problem_l3299_329987

theorem percentage_problem (x : ℝ) (h1 : 0.2 * x = 400) : 
  (2400 / x) * 100 = 120 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3299_329987


namespace NUMINAMATH_CALUDE_one_minus_repeating_decimal_l3299_329947

/-- The value of the repeating decimal 0.123123... -/
def repeating_decimal : ℚ := 41 / 333

/-- Theorem: 1 - 0.123123... = 292/333 -/
theorem one_minus_repeating_decimal :
  1 - repeating_decimal = 292 / 333 := by
  sorry

end NUMINAMATH_CALUDE_one_minus_repeating_decimal_l3299_329947


namespace NUMINAMATH_CALUDE_triangle_formation_l3299_329903

/-- Determines if three lengths can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The given groups of numbers --/
def group_A : (ℝ × ℝ × ℝ) := (5, 7, 12)
def group_B : (ℝ × ℝ × ℝ) := (7, 7, 15)
def group_C : (ℝ × ℝ × ℝ) := (6, 9, 16)
def group_D : (ℝ × ℝ × ℝ) := (6, 8, 12)

theorem triangle_formation :
  ¬(can_form_triangle group_A.1 group_A.2.1 group_A.2.2) ∧
  ¬(can_form_triangle group_B.1 group_B.2.1 group_B.2.2) ∧
  ¬(can_form_triangle group_C.1 group_C.2.1 group_C.2.2) ∧
  can_form_triangle group_D.1 group_D.2.1 group_D.2.2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l3299_329903


namespace NUMINAMATH_CALUDE_oula_deliveries_l3299_329955

/-- Proves that Oula made 96 deliveries given the problem conditions -/
theorem oula_deliveries :
  ∀ (oula_deliveries tona_deliveries : ℕ) 
    (pay_per_delivery : ℕ) 
    (pay_difference : ℕ),
  pay_per_delivery = 100 →
  tona_deliveries = 3 * oula_deliveries / 4 →
  pay_difference = 2400 →
  pay_per_delivery * oula_deliveries - pay_per_delivery * tona_deliveries = pay_difference →
  oula_deliveries = 96 := by
sorry

end NUMINAMATH_CALUDE_oula_deliveries_l3299_329955


namespace NUMINAMATH_CALUDE_cards_in_hospital_l3299_329993

/-- Proves that the number of get well cards Mariela received while in the hospital is 403 -/
theorem cards_in_hospital (total_cards : ℕ) (cards_after_home : ℕ) 
  (h1 : total_cards = 690) 
  (h2 : cards_after_home = 287) : 
  total_cards - cards_after_home = 403 := by
  sorry

end NUMINAMATH_CALUDE_cards_in_hospital_l3299_329993


namespace NUMINAMATH_CALUDE_solve_equation_l3299_329938

theorem solve_equation : ∃ x : ℝ, (10 - x = 15) ∧ (x = -5) := by sorry

end NUMINAMATH_CALUDE_solve_equation_l3299_329938


namespace NUMINAMATH_CALUDE_monthly_income_A_l3299_329937

/-- Given the average monthly incomes of pairs of individuals, prove the monthly income of A. -/
theorem monthly_income_A (income_AB income_BC income_AC : ℚ) 
  (h1 : (income_A + income_B) / 2 = 5050)
  (h2 : (income_B + income_C) / 2 = 6250)
  (h3 : (income_A + income_C) / 2 = 5200)
  : income_A = 4000 := by
  sorry

where
  income_A : ℚ := sorry
  income_B : ℚ := sorry
  income_C : ℚ := sorry

end NUMINAMATH_CALUDE_monthly_income_A_l3299_329937


namespace NUMINAMATH_CALUDE_rectangular_garden_area_l3299_329988

/-- The area of a rectangular garden with length 2.5 meters and width 0.48 meters is 1.2 square meters. -/
theorem rectangular_garden_area : 
  let length : ℝ := 2.5
  let width : ℝ := 0.48
  length * width = 1.2 := by sorry

end NUMINAMATH_CALUDE_rectangular_garden_area_l3299_329988


namespace NUMINAMATH_CALUDE_water_flow_restrictor_l3299_329992

/-- Calculates the reduced flow rate given the original flow rate. -/
def reducedFlowRate (originalRate : ℝ) : ℝ :=
  0.6 * originalRate - 1

theorem water_flow_restrictor (originalRate : ℝ) 
    (h : originalRate = 5.0) : 
    reducedFlowRate originalRate = 2.0 := by
  sorry

end NUMINAMATH_CALUDE_water_flow_restrictor_l3299_329992


namespace NUMINAMATH_CALUDE_amanda_keeps_22_candy_bars_l3299_329960

/-- The number of candy bars Amanda keeps for herself given the initial amount, 
    the amount given to her sister initially, the amount bought later, 
    and the multiplier for the second giving. -/
def amanda_candy_bars (initial : ℕ) (first_given : ℕ) (bought : ℕ) (multiplier : ℕ) : ℕ :=
  initial - first_given + bought - (multiplier * first_given)

/-- Theorem stating that Amanda keeps 22 candy bars for herself 
    given the specific conditions in the problem. -/
theorem amanda_keeps_22_candy_bars : 
  amanda_candy_bars 7 3 30 4 = 22 := by sorry

end NUMINAMATH_CALUDE_amanda_keeps_22_candy_bars_l3299_329960


namespace NUMINAMATH_CALUDE_milk_remaining_l3299_329990

theorem milk_remaining (initial : ℚ) (given_away : ℚ) (remaining : ℚ) :
  initial = 5 →
  given_away = 17 / 4 →
  remaining = initial - given_away →
  remaining = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_milk_remaining_l3299_329990


namespace NUMINAMATH_CALUDE_logarithm_properties_l3299_329901

-- Define the theorem
theorem logarithm_properties (a b m n : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hm : 0 < m) (hm1 : m ≠ 1) (hn : 0 < n) : 
  (Real.log a / Real.log b) * (Real.log b / Real.log a) = 1 ∧ 
  (Real.log n / Real.log a) / (Real.log n / Real.log (m * a)) = 1 + (Real.log m / Real.log a) :=
by sorry

end NUMINAMATH_CALUDE_logarithm_properties_l3299_329901


namespace NUMINAMATH_CALUDE_equivalence_condition_l3299_329916

theorem equivalence_condition (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ (a, b) ≠ (0, 0)) : 
  (1 / a < 1 / b) ↔ (a * b / (a^3 - b^3) > 0) :=
by sorry

end NUMINAMATH_CALUDE_equivalence_condition_l3299_329916


namespace NUMINAMATH_CALUDE_divisible_by_six_percentage_l3299_329997

theorem divisible_by_six_percentage (n : ℕ) : n = 150 →
  (((Finset.filter (fun x => x % 6 = 0) (Finset.range (n + 1))).card : ℚ) / (n : ℚ)) * 100 = 50/3 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_six_percentage_l3299_329997


namespace NUMINAMATH_CALUDE_equation_solution_l3299_329904

theorem equation_solution :
  let f : ℝ → ℝ := λ x => (2*x + 1)*(3*x + 1)*(5*x + 1)*(30*x + 1)
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    f x₁ = 10 ∧ f x₂ = 10 ∧
    x₁ = (-4 + Real.sqrt 31) / 15 ∧
    x₂ = (-4 - Real.sqrt 31) / 15 ∧
    ∀ x : ℝ, f x = 10 → (x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3299_329904


namespace NUMINAMATH_CALUDE_total_rainfall_is_23_inches_l3299_329902

/-- Calculates the total rainfall over three days given specific conditions --/
def totalRainfall (mondayHours : ℝ) (mondayRate : ℝ) 
                  (tuesdayHours : ℝ) (tuesdayRate : ℝ)
                  (wednesdayHours : ℝ) : ℝ :=
  mondayHours * mondayRate + 
  tuesdayHours * tuesdayRate + 
  wednesdayHours * (2 * tuesdayRate)

/-- Proves that the total rainfall over the three days is 23 inches --/
theorem total_rainfall_is_23_inches : 
  totalRainfall 7 1 4 2 2 = 23 := by
  sorry


end NUMINAMATH_CALUDE_total_rainfall_is_23_inches_l3299_329902


namespace NUMINAMATH_CALUDE_sum_of_ages_l3299_329921

/-- Represents the ages of Alex, Chris, and Bella -/
structure Ages where
  alex : ℕ
  chris : ℕ
  bella : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.alex = ages.chris + 8 ∧
  ages.alex + 10 = 3 * (ages.chris - 6) ∧
  ages.bella = 2 * ages.chris

/-- The theorem to prove -/
theorem sum_of_ages (ages : Ages) :
  satisfiesConditions ages →
  ages.alex + ages.chris + ages.bella = 80 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l3299_329921


namespace NUMINAMATH_CALUDE_expression_evaluation_l3299_329968

theorem expression_evaluation : (47 + 21)^2 - (47^2 + 21^2) - 7 * 47 = 1645 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3299_329968


namespace NUMINAMATH_CALUDE_probability_no_adjacent_birch_is_two_forty_fifths_l3299_329936

def total_trees : ℕ := 15
def birch_trees : ℕ := 6
def non_birch_trees : ℕ := 9

def probability_no_adjacent_birch : ℚ :=
  (Nat.choose (non_birch_trees + 1) birch_trees) / (Nat.choose total_trees birch_trees)

theorem probability_no_adjacent_birch_is_two_forty_fifths :
  probability_no_adjacent_birch = 2 / 45 := by
  sorry

end NUMINAMATH_CALUDE_probability_no_adjacent_birch_is_two_forty_fifths_l3299_329936


namespace NUMINAMATH_CALUDE_correct_equation_transformation_l3299_329915

theorem correct_equation_transformation (x : ℝ) : 
  (x / 3 = 7) → (x = 21) :=
by sorry

end NUMINAMATH_CALUDE_correct_equation_transformation_l3299_329915


namespace NUMINAMATH_CALUDE_friendship_distribution_impossibility_l3299_329983

theorem friendship_distribution_impossibility :
  ∀ (students : Finset Nat) (f : Nat → Nat),
    Finset.card students = 25 →
    (∃ s₁ s₂ s₃ : Finset Nat, 
      Finset.card s₁ = 6 ∧ 
      Finset.card s₂ = 10 ∧ 
      Finset.card s₃ = 9 ∧
      s₁ ∪ s₂ ∪ s₃ = students ∧
      Disjoint s₁ s₂ ∧ Disjoint s₁ s₃ ∧ Disjoint s₂ s₃ ∧
      (∀ i ∈ s₁, f i = 3) ∧
      (∀ i ∈ s₂, f i = 4) ∧
      (∀ i ∈ s₃, f i = 5)) →
    False := by
  sorry


end NUMINAMATH_CALUDE_friendship_distribution_impossibility_l3299_329983


namespace NUMINAMATH_CALUDE_f_min_value_l3299_329967

/-- The function f(x) = x^2 + 8x + 15 -/
def f (x : ℝ) : ℝ := x^2 + 8*x + 15

/-- Theorem: The minimum value of f(x) = x^2 + 8x + 15 is -1 -/
theorem f_min_value : ∃ (a : ℝ), f a = -1 ∧ ∀ (x : ℝ), f x ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_f_min_value_l3299_329967


namespace NUMINAMATH_CALUDE_a_2017_equals_16_l3299_329917

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

end NUMINAMATH_CALUDE_a_2017_equals_16_l3299_329917


namespace NUMINAMATH_CALUDE_point_B_left_of_A_l3299_329985

theorem point_B_left_of_A : 8/13 < 5/8 := by sorry

end NUMINAMATH_CALUDE_point_B_left_of_A_l3299_329985


namespace NUMINAMATH_CALUDE_new_to_original_student_ratio_l3299_329948

theorem new_to_original_student_ratio 
  (original_avg : ℝ) 
  (new_student_avg : ℝ) 
  (avg_decrease : ℝ) 
  (h1 : original_avg = 40)
  (h2 : new_student_avg = 34)
  (h3 : avg_decrease = 4)
  (h4 : original_avg = (original_avg - avg_decrease) + 6) :
  ∃ (O N : ℕ), N = 2 * O ∧ N > 0 ∧ O > 0 := by
  sorry

end NUMINAMATH_CALUDE_new_to_original_student_ratio_l3299_329948


namespace NUMINAMATH_CALUDE_second_group_size_l3299_329925

theorem second_group_size (total : ℕ) (group1 group3 group4 : ℕ) 
  (h1 : total = 24)
  (h2 : group1 = 5)
  (h3 : group3 = 7)
  (h4 : group4 = 4) :
  total - (group1 + group3 + group4) = 8 := by
sorry

end NUMINAMATH_CALUDE_second_group_size_l3299_329925


namespace NUMINAMATH_CALUDE_function_equality_l3299_329998

theorem function_equality (f : ℝ → ℝ) (h : ∀ x, f x = 3 * x - 5) :
  2 * (f 3) - 10 = f (3 - 2) := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l3299_329998


namespace NUMINAMATH_CALUDE_multiplication_properties_l3299_329965

theorem multiplication_properties (m n : ℕ) :
  let a := 6 * m + 1
  let b := 6 * n + 1
  let c := 6 * m + 5
  let d := 6 * n + 5
  (∃ k : ℕ, a * b = 6 * k + 1) ∧
  (∃ k : ℕ, c * d = 6 * k + 1) ∧
  (∃ k : ℕ, a * d = 6 * k + 5) :=
by sorry

end NUMINAMATH_CALUDE_multiplication_properties_l3299_329965


namespace NUMINAMATH_CALUDE_exam_correct_answers_l3299_329958

/-- Proves the number of correct answers in an exam with given conditions -/
theorem exam_correct_answers 
  (total_questions : ℕ) 
  (correct_score : ℤ) 
  (wrong_score : ℤ) 
  (total_score : ℤ) 
  (h1 : total_questions = 70)
  (h2 : correct_score = 3)
  (h3 : wrong_score = -1)
  (h4 : total_score = 38) :
  ∃ (correct wrong : ℕ),
    correct + wrong = total_questions ∧
    correct_score * correct + wrong_score * wrong = total_score ∧
    correct = 27 := by
  sorry

end NUMINAMATH_CALUDE_exam_correct_answers_l3299_329958


namespace NUMINAMATH_CALUDE_james_spent_six_l3299_329918

/-- The total amount James spent on milk, bananas, and sales tax -/
def total_spent (milk_price banana_price tax_rate : ℚ) : ℚ :=
  let subtotal := milk_price + banana_price
  let tax := subtotal * tax_rate
  subtotal + tax

/-- Theorem stating that James spent $6 given the problem conditions -/
theorem james_spent_six :
  total_spent 3 2 (1/5) = 6 := by
  sorry

end NUMINAMATH_CALUDE_james_spent_six_l3299_329918


namespace NUMINAMATH_CALUDE_magnitude_of_complex_power_l3299_329912

theorem magnitude_of_complex_power : 
  Complex.abs ((2 : ℂ) + (2 : ℂ) * Complex.I) ^ 8 = (4096 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_power_l3299_329912


namespace NUMINAMATH_CALUDE_unique_line_pair_l3299_329940

/-- Two equations represent the same line if they have the same slope and y-intercept -/
def same_line (a b : ℝ) : Prop :=
  ∃ (m c : ℝ), ∀ (x y : ℝ),
    (2 * x + a * y + 10 = 0 ↔ y = m * x + c) ∧
    (b * x - 3 * y - 15 = 0 ↔ y = m * x + c)

/-- There exists exactly one pair (a, b) such that the given equations represent the same line -/
theorem unique_line_pair : ∃! (p : ℝ × ℝ), same_line p.1 p.2 := by sorry

end NUMINAMATH_CALUDE_unique_line_pair_l3299_329940


namespace NUMINAMATH_CALUDE_marble_175_is_white_l3299_329900

/-- Represents the color of a marble -/
inductive MarbleColor
| Gray
| White
| Black

/-- Returns the color of the nth marble in the sequence -/
def marbleColor (n : ℕ) : MarbleColor :=
  match n % 12 with
  | 0 | 1 | 2 | 3 | 4 => MarbleColor.Gray
  | 5 | 6 | 7 | 8 => MarbleColor.White
  | _ => MarbleColor.Black

/-- Theorem stating that the 175th marble is white -/
theorem marble_175_is_white : marbleColor 175 = MarbleColor.White := by
  sorry

end NUMINAMATH_CALUDE_marble_175_is_white_l3299_329900


namespace NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l3299_329946

theorem multiplication_table_odd_fraction :
  let table_size : ℕ := 16
  let total_products : ℕ := table_size * table_size
  let odd_numbers : ℕ := (table_size + 1) / 2
  let odd_products : ℕ := odd_numbers * odd_numbers
  (odd_products : ℚ) / total_products = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l3299_329946


namespace NUMINAMATH_CALUDE_greatest_power_of_two_congruence_l3299_329943

theorem greatest_power_of_two_congruence (m : ℕ) : 
  (∀ n : ℤ, Odd n → (n^2 * (1 + n^2 - n^4)) ≡ 1 [ZMOD 2^m]) ↔ m ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_congruence_l3299_329943


namespace NUMINAMATH_CALUDE_game_show_probability_l3299_329906

/-- Represents the amount of money in each box -/
def box_values : Fin 3 → ℕ
  | 0 => 4
  | 1 => 400
  | 2 => 4000

/-- The total number of ways to assign 3 keys to 3 boxes -/
def total_assignments : ℕ := 6

/-- The number of assignments that result in winning more than $4000 -/
def winning_assignments : ℕ := 1

/-- The probability of winning more than $4000 -/
def win_probability : ℚ := winning_assignments / total_assignments

theorem game_show_probability :
  win_probability = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_game_show_probability_l3299_329906


namespace NUMINAMATH_CALUDE_line_circle_orthogonality_l3299_329964

/-- Given a line and a circle, prove that a specific value of 'a' ensures orthogonality of OA and OB -/
theorem line_circle_orthogonality (a : ℝ) (A B : ℝ × ℝ) :
  (∀ (x y : ℝ), x - 2*y + a = 0 → x^2 + y^2 = 2) →  -- Line intersects circle
  (A.1 - 2*A.2 + a = 0 ∧ A.1^2 + A.2^2 = 2) →        -- A satisfies both equations
  (B.1 - 2*B.2 + a = 0 ∧ B.1^2 + B.2^2 = 2) →        -- B satisfies both equations
  a = Real.sqrt 5 →                                  -- Specific value of a
  A.1 * B.1 + A.2 * B.2 = 0                          -- OA · OB = 0
  := by sorry

end NUMINAMATH_CALUDE_line_circle_orthogonality_l3299_329964


namespace NUMINAMATH_CALUDE_molecular_weight_4_moles_BaI2_value_l3299_329956

/-- The molecular weight of 4 moles of Barium iodide (BaI2) -/
def molecular_weight_4_moles_BaI2 : ℝ :=
  let atomic_weight_Ba : ℝ := 137.33
  let atomic_weight_I : ℝ := 126.90
  let molecular_weight_BaI2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_I
  4 * molecular_weight_BaI2

/-- Theorem stating that the molecular weight of 4 moles of Barium iodide is 1564.52 grams -/
theorem molecular_weight_4_moles_BaI2_value : 
  molecular_weight_4_moles_BaI2 = 1564.52 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_4_moles_BaI2_value_l3299_329956


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l3299_329952

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 8) :
  Real.sqrt (3 * x + 2) + Real.sqrt (3 * y + 2) + Real.sqrt (3 * z + 2) ≤ 3 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l3299_329952


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3299_329933

def l1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y - 1 = 0
def l2 (a : ℝ) (x y : ℝ) : Prop := x + (a + 1) * y + 4 = 0

def parallel (a : ℝ) : Prop :=
  ∀ x1 y1 x2 y2 : ℝ, l1 a x1 y1 ∧ l2 a x2 y2 → (y1 - y2) * (a + 1) = (x1 - x2) * 2

theorem sufficient_not_necessary :
  (parallel (-2)) ∧ (∃ a : ℝ, a ≠ -2 ∧ parallel a) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3299_329933


namespace NUMINAMATH_CALUDE_simplify_2A_minus_B_value_2A_minus_B_special_case_l3299_329977

/-- Definition of A in terms of a and b -/
def A (a b : ℝ) : ℝ := b^2 - a^2 + 5*a*b

/-- Definition of B in terms of a and b -/
def B (a b : ℝ) : ℝ := 3*a*b + 2*b^2 - a^2

/-- Theorem stating the simplified form of 2A - B -/
theorem simplify_2A_minus_B (a b : ℝ) : 2 * A a b - B a b = -a^2 + 7*a*b := by sorry

/-- Theorem stating the value of 2A - B when a = 1 and b = 2 -/
theorem value_2A_minus_B_special_case : 2 * A 1 2 - B 1 2 = 13 := by sorry

end NUMINAMATH_CALUDE_simplify_2A_minus_B_value_2A_minus_B_special_case_l3299_329977


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l3299_329966

theorem solution_set_equivalence (x : ℝ) : 
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := by
sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l3299_329966


namespace NUMINAMATH_CALUDE_missing_roots_theorem_l3299_329929

def p (x : ℝ) : ℝ := 12 * x^5 - 8 * x^4 - 45 * x^3 + 45 * x^2 + 8 * x - 12

theorem missing_roots_theorem (h1 : p 1 = 0) (h2 : p 1.5 = 0) (h3 : p (-2) = 0) :
  p (2/3) = 0 ∧ p (-1/2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_missing_roots_theorem_l3299_329929


namespace NUMINAMATH_CALUDE_muffin_sale_total_l3299_329959

theorem muffin_sale_total (boys : ℕ) (girls : ℕ) (boys_muffins : ℕ) (girls_muffins : ℕ) : 
  boys = 3 → 
  girls = 2 → 
  boys_muffins = 12 → 
  girls_muffins = 20 → 
  boys * boys_muffins + girls * girls_muffins = 76 := by
sorry

end NUMINAMATH_CALUDE_muffin_sale_total_l3299_329959


namespace NUMINAMATH_CALUDE_smallest_n_repeating_decimal_l3299_329924

/-- A number is a repeating decimal with period k if it can be expressed as m/(10^k - 1) for some integer m -/
def is_repeating_decimal (x : ℚ) (k : ℕ) : Prop :=
  ∃ m : ℤ, x = m / (10^k - 1)

/-- The smallest positive integer n < 1000 such that 1/n is a repeating decimal with period 3
    and 1/(n+6) is a repeating decimal with period 2 is 27 -/
theorem smallest_n_repeating_decimal : 
  ∃ n : ℕ, n < 1000 ∧ 
           is_repeating_decimal (1 / n) 3 ∧ 
           is_repeating_decimal (1 / (n + 6)) 2 ∧
           ∀ m : ℕ, m < n → ¬(is_repeating_decimal (1 / m) 3 ∧ is_repeating_decimal (1 / (m + 6)) 2) ∧
           n = 27 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_repeating_decimal_l3299_329924


namespace NUMINAMATH_CALUDE_employee_hourly_rate_l3299_329905

/-- Proves that the hourly rate for the first 40 hours is $11.25 given the conditions -/
theorem employee_hourly_rate 
  (x : ℝ) -- hourly rate for the first 40 hours
  (overtime_hours : ℝ) -- number of overtime hours
  (overtime_rate : ℝ) -- overtime hourly rate
  (gross_pay : ℝ) -- total gross pay
  (h1 : overtime_hours = 10.75)
  (h2 : overtime_rate = 16)
  (h3 : gross_pay = 622)
  (h4 : 40 * x + overtime_hours * overtime_rate = gross_pay) :
  x = 11.25 :=
by sorry

end NUMINAMATH_CALUDE_employee_hourly_rate_l3299_329905


namespace NUMINAMATH_CALUDE_cos_alpha_plus_pi_sixth_l3299_329939

theorem cos_alpha_plus_pi_sixth (α : Real) (h : Real.sin (α - π/3) = 1/3) : 
  Real.cos (α + π/6) = -1/3 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_pi_sixth_l3299_329939


namespace NUMINAMATH_CALUDE_yard_length_ratio_l3299_329970

theorem yard_length_ratio : 
  ∀ (alex_length brianne_length derrick_length : ℝ),
  brianne_length = 6 * alex_length →
  brianne_length = 30 →
  derrick_length = 10 →
  alex_length / derrick_length = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_ratio_l3299_329970


namespace NUMINAMATH_CALUDE_average_tickets_sold_l3299_329974

/-- Proves that the average number of tickets sold per member is 66 given the conditions -/
theorem average_tickets_sold (male_count : ℕ) (female_count : ℕ) 
  (male_female_ratio : female_count = 2 * male_count)
  (female_avg : ℝ) (male_avg : ℝ)
  (h_female_avg : female_avg = 70)
  (h_male_avg : male_avg = 58) :
  let total_tickets := female_count * female_avg + male_count * male_avg
  let total_members := male_count + female_count
  total_tickets / total_members = 66 := by
sorry

end NUMINAMATH_CALUDE_average_tickets_sold_l3299_329974


namespace NUMINAMATH_CALUDE_potato_price_correct_l3299_329930

/-- The price of potatoes per kilo -/
def potato_price : ℝ := 2

theorem potato_price_correct (
  initial_amount : ℝ)
  (potato_kilos : ℝ)
  (tomato_kilos : ℝ)
  (cucumber_kilos : ℝ)
  (banana_kilos : ℝ)
  (tomato_price : ℝ)
  (cucumber_price : ℝ)
  (banana_price : ℝ)
  (remaining_amount : ℝ)
  (h1 : initial_amount = 500)
  (h2 : potato_kilos = 6)
  (h3 : tomato_kilos = 9)
  (h4 : cucumber_kilos = 5)
  (h5 : banana_kilos = 3)
  (h6 : tomato_price = 3)
  (h7 : cucumber_price = 4)
  (h8 : banana_price = 5)
  (h9 : remaining_amount = 426)
  (h10 : initial_amount - (potato_kilos * potato_price + tomato_kilos * tomato_price + 
         cucumber_kilos * cucumber_price + banana_kilos * banana_price) = remaining_amount) :
  potato_price = 2 := by
  sorry

end NUMINAMATH_CALUDE_potato_price_correct_l3299_329930


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3299_329979

/-- Given vectors a and b in ℝ², where a = (-1, 1) and b = (3, m),
    and a is parallel to (a + b), prove that m = -3. -/
theorem parallel_vectors_m_value (m : ℝ) : 
  let a : Fin 2 → ℝ := ![-1, 1]
  let b : Fin 2 → ℝ := ![3, m]
  (∃ (k : ℝ), k ≠ 0 ∧ (λ i => a i + b i) = λ i => k * a i) →
  m = -3 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3299_329979


namespace NUMINAMATH_CALUDE_white_dandelions_on_saturday_l3299_329980

/-- Represents the day of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents the state of a dandelion -/
inductive DandelionState
  | Yellow
  | White
  | Dispersed

/-- Represents the count of dandelions in different states -/
structure DandelionCount where
  yellow : ℕ
  white : ℕ

/-- The life cycle of a dandelion -/
def dandelionLifeCycle (day : ℕ) : DandelionState :=
  match day with
  | 0 | 1 | 2 => DandelionState.Yellow
  | 3 | 4 => DandelionState.White
  | _ => DandelionState.Dispersed

/-- Count of dandelions on a given day -/
def dandelionCountOnDay (day : Day) : DandelionCount :=
  match day with
  | Day.Monday => { yellow := 20, white := 14 }
  | Day.Wednesday => { yellow := 15, white := 11 }
  | _ => { yellow := 0, white := 0 }  -- We don't have information for other days

/-- Days between two given days -/
def daysBetween (start finish : Day) : ℕ :=
  match start, finish with
  | Day.Monday, Day.Wednesday => 2
  | Day.Wednesday, Day.Saturday => 3
  | _, _ => 0  -- We don't need other cases for this problem

/-- The main theorem -/
theorem white_dandelions_on_saturday :
  ∃ (new_dandelions : ℕ),
    new_dandelions = (dandelionCountOnDay Day.Wednesday).yellow + (dandelionCountOnDay Day.Wednesday).white
                   - (dandelionCountOnDay Day.Monday).yellow
    ∧ new_dandelions = 6
    ∧ (dandelionLifeCycle (daysBetween Day.Tuesday Day.Saturday) = DandelionState.White
    ∧ dandelionLifeCycle (daysBetween Day.Wednesday Day.Saturday) = DandelionState.White)
    → new_dandelions = 6 := by sorry


end NUMINAMATH_CALUDE_white_dandelions_on_saturday_l3299_329980


namespace NUMINAMATH_CALUDE_school_population_equality_l3299_329919

theorem school_population_equality (m d : ℕ) (M D : ℝ) :
  m > 0 → d > 0 →
  (M / m + D / d) / 2 = (M + D) / (m + d) →
  m = d :=
sorry

end NUMINAMATH_CALUDE_school_population_equality_l3299_329919


namespace NUMINAMATH_CALUDE_badminton_tournament_matches_l3299_329989

/-- Represents a single elimination tournament -/
structure Tournament :=
  (total_participants : ℕ)
  (auto_progressed : ℕ)
  (first_round_players : ℕ)
  (h_participants : total_participants = auto_progressed + first_round_players)

/-- Calculates the total number of matches in the tournament -/
def total_matches (t : Tournament) : ℕ := t.total_participants - 1

theorem badminton_tournament_matches :
  ∀ t : Tournament,
  t.total_participants = 120 →
  t.auto_progressed = 16 →
  t.first_round_players = 104 →
  total_matches t = 119 :=
by sorry

end NUMINAMATH_CALUDE_badminton_tournament_matches_l3299_329989


namespace NUMINAMATH_CALUDE_intersection_one_element_l3299_329961

theorem intersection_one_element (a : ℝ) : 
  let A : Set ℝ := {1, a, 5}
  let B : Set ℝ := {2, a^2 + 1}
  (∃! x, x ∈ A ∩ B) → (a = 0 ∨ a = -2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_one_element_l3299_329961


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3299_329986

theorem complex_modulus_problem (z : ℂ) (h : (1 + 2*I)*z = (1 - I)) : 
  Complex.abs z = Real.sqrt 10 / 5 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3299_329986


namespace NUMINAMATH_CALUDE_fraction_equality_l3299_329991

theorem fraction_equality : (2015 : ℚ) / (2015^2 - 2016 * 2014) = 2015 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l3299_329991


namespace NUMINAMATH_CALUDE_sin_cos_equation_solution_set_l3299_329976

theorem sin_cos_equation_solution_set (x : ℝ) : 
  Real.sin (x / 2) - Real.cos (x / 2) = 1 ↔ 
  (∃ k : ℤ, x = π * (1 + 4 * k) ∨ x = 2 * π * (1 + 2 * k)) :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_equation_solution_set_l3299_329976


namespace NUMINAMATH_CALUDE_binomial_fraction_zero_l3299_329945

theorem binomial_fraction_zero : (Nat.choose 2 5 * 3^5) / Nat.choose 10 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_binomial_fraction_zero_l3299_329945


namespace NUMINAMATH_CALUDE_square_roots_equality_l3299_329978

theorem square_roots_equality (m : ℝ) :
  (∃ (x : ℝ), x > 0 ∧ (m + 1)^2 = x ∧ (3*m - 1)^2 = x) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_equality_l3299_329978


namespace NUMINAMATH_CALUDE_four_tangent_lines_l3299_329953

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Counts the number of common tangent lines to two circles -/
def countCommonTangents (c1 c2 : Circle) : ℕ := sorry

/-- The main theorem -/
theorem four_tangent_lines (c1 c2 : Circle) 
  (h1 : c1.radius = 5)
  (h2 : c2.radius = 2)
  (h3 : Real.sqrt ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2) = 10) :
  countCommonTangents c1 c2 = 4 := by sorry

end NUMINAMATH_CALUDE_four_tangent_lines_l3299_329953


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_measure_l3299_329973

/-- The measure of an interior angle of a regular octagon in degrees -/
def regular_octagon_interior_angle : ℝ := 135

/-- A regular octagon has 8 sides -/
def regular_octagon_sides : ℕ := 8

theorem regular_octagon_interior_angle_measure :
  regular_octagon_interior_angle = (((regular_octagon_sides - 2) * 180) : ℝ) / regular_octagon_sides :=
sorry

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_measure_l3299_329973


namespace NUMINAMATH_CALUDE_salad_dressing_vinegar_weight_l3299_329962

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


end NUMINAMATH_CALUDE_salad_dressing_vinegar_weight_l3299_329962


namespace NUMINAMATH_CALUDE_longest_boat_through_bend_l3299_329923

theorem longest_boat_through_bend (a : ℝ) (h : a > 0) :
  ∃ c : ℝ, c = 2 * a * Real.sqrt 2 ∧
  ∀ l : ℝ, l > c → ¬ (∃ θ : ℝ, 
    l * Real.cos θ ≤ a ∧ l * Real.sin θ ≤ a) := by
  sorry

end NUMINAMATH_CALUDE_longest_boat_through_bend_l3299_329923


namespace NUMINAMATH_CALUDE_only_first_is_prime_one_prime_in_sequence_l3299_329950

/-- Generates the nth number in the sequence starting with 47 and repeating 47 sequentially -/
def sequenceNumber (n : ℕ) : ℕ :=
  if n = 0 then 47 else
  (sequenceNumber (n - 1)) * 100 + 47

/-- Theorem stating that only the first number in the sequence is prime -/
theorem only_first_is_prime :
  ∀ n : ℕ, n > 0 → ¬ Nat.Prime (sequenceNumber n) :=
by
  sorry

/-- Corollary stating that there is exactly one prime number in the sequence -/
theorem one_prime_in_sequence :
  (∃! k : ℕ, Nat.Prime (sequenceNumber k)) :=
by
  sorry

end NUMINAMATH_CALUDE_only_first_is_prime_one_prime_in_sequence_l3299_329950


namespace NUMINAMATH_CALUDE_division_problem_l3299_329994

theorem division_problem (x : ℝ) : 
  (1.5 * 1265) / x = 271.07142857142856 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3299_329994


namespace NUMINAMATH_CALUDE_a_2022_eq_674_l3299_329909

def a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | 2 => 1
  | n+3 => (n+3) / (a n * a (n+1) * a (n+2))

theorem a_2022_eq_674 : a 2022 = 674 := by
  sorry

end NUMINAMATH_CALUDE_a_2022_eq_674_l3299_329909


namespace NUMINAMATH_CALUDE_solution_exists_l3299_329982

theorem solution_exists (x y : ℝ) : (2*x - 3*y + 5)^2 + |x - y + 2| = 0 → x = -1 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_exists_l3299_329982


namespace NUMINAMATH_CALUDE_fabulous_iff_not_power_of_two_l3299_329949

def is_fabulous (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ a : ℕ, 2 ≤ a ∧ a ≤ n - 1 ∧ (n ∣ a^n - a)

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

theorem fabulous_iff_not_power_of_two (n : ℕ) :
  is_fabulous n ↔ (¬ is_power_of_two n) :=
sorry

end NUMINAMATH_CALUDE_fabulous_iff_not_power_of_two_l3299_329949


namespace NUMINAMATH_CALUDE_unique_solution_fourth_root_equation_l3299_329999

theorem unique_solution_fourth_root_equation :
  ∃! x : ℝ, (((4 - x) ^ (1/4) : ℝ) + ((x - 2) ^ (1/2) : ℝ) = 2) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_fourth_root_equation_l3299_329999


namespace NUMINAMATH_CALUDE_probability_one_defective_l3299_329981

def total_products : ℕ := 10
def quality_products : ℕ := 7
def defective_products : ℕ := 3
def selected_products : ℕ := 4

theorem probability_one_defective :
  (Nat.choose quality_products (selected_products - 1) * Nat.choose defective_products 1) /
  Nat.choose total_products selected_products = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_probability_one_defective_l3299_329981


namespace NUMINAMATH_CALUDE_vector_inequality_l3299_329931

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]
variable (hd : FiniteDimensional.finrank ℝ V = 2)

/-- Given four vectors a, b, c, d in a 2D real vector space such that their sum is zero,
    prove that the sum of their norms is greater than or equal to the sum of the norms
    of their pairwise sums with d. -/
theorem vector_inequality (a b c d : V) (h : a + b + c + d = 0) :
  ‖a‖ + ‖b‖ + ‖c‖ + ‖d‖ ≥ ‖a + d‖ + ‖b + d‖ + ‖c + d‖ :=
sorry

end NUMINAMATH_CALUDE_vector_inequality_l3299_329931


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l3299_329941

theorem hemisphere_surface_area (r : ℝ) (h : r > 0) : 
  π * r^2 = 3 → 2 * π * r^2 + π * r^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l3299_329941
