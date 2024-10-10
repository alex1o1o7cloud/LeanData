import Mathlib

namespace trajectory_is_circle_l1217_121731

/-- The ellipse with equation x²/7 + y²/3 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 7 + p.2^2 / 3 = 1}

/-- The left focus of the ellipse -/
def F₁ : ℝ × ℝ := (-2, 0)

/-- The right focus of the ellipse -/
def F₂ : ℝ × ℝ := (2, 0)

/-- The set of all points Q obtained by extending F₁P to Q such that |PQ| = |PF₂| for all P on the ellipse -/
def TrajectoryQ (P : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {Q : ℝ × ℝ | P ∈ Ellipse ∧ ∃ t : ℝ, t > 1 ∧ Q = (t • (P - F₁) + F₁) ∧ 
    ‖Q - P‖ = ‖P - F₂‖}

/-- The theorem stating that the trajectory of Q is a circle -/
theorem trajectory_is_circle : 
  ∀ Q : ℝ × ℝ, (∃ P : ℝ × ℝ, Q ∈ TrajectoryQ P) ↔ (Q.1 + 2)^2 + Q.2^2 = 28 :=
sorry

end trajectory_is_circle_l1217_121731


namespace number_of_cows_l1217_121706

/-- Represents the number of legs for each animal type -/
def legs_per_animal : (Fin 2) → ℕ
| 0 => 2  -- chickens
| 1 => 4  -- cows

/-- Represents the total number of animals -/
def total_animals : ℕ := 160

/-- Represents the total number of legs -/
def total_legs : ℕ := 400

/-- Proves that the number of cows is 40 given the conditions -/
theorem number_of_cows : 
  ∃ (chickens cows : ℕ), 
    chickens + cows = total_animals ∧ 
    chickens * legs_per_animal 0 + cows * legs_per_animal 1 = total_legs ∧
    cows = 40 := by
  sorry

end number_of_cows_l1217_121706


namespace function_identity_l1217_121755

theorem function_identity (f : ℕ → ℕ) (h : ∀ x y : ℕ, f (f x + f y) = x + y) :
  ∀ n : ℕ, f n = n := by
  sorry

end function_identity_l1217_121755


namespace milk_dilution_l1217_121796

/-- Represents the milk dilution problem -/
theorem milk_dilution (initial_capacity : ℝ) (removal_amount : ℝ) : 
  initial_capacity = 45 →
  removal_amount = 9 →
  let first_milk_remaining := initial_capacity - removal_amount
  let first_mixture_milk_ratio := first_milk_remaining / initial_capacity
  let second_milk_remaining := first_milk_remaining - (first_mixture_milk_ratio * removal_amount)
  second_milk_remaining = 28.8 := by
  sorry

end milk_dilution_l1217_121796


namespace min_sum_of_squares_l1217_121729

theorem min_sum_of_squares (x₁ x₂ x₃ : ℝ) 
  (pos₁ : x₁ > 0) (pos₂ : x₂ > 0) (pos₃ : x₃ > 0)
  (sum_constraint : x₁ + 3*x₂ + 5*x₃ = 100) : 
  x₁^2 + x₂^2 + x₃^2 ≥ 2000/7 := by
  sorry

end min_sum_of_squares_l1217_121729


namespace coat_price_calculation_l1217_121771

/-- Calculates the final price of a coat after discounts and tax -/
def finalPrice (originalPrice : ℝ) (initialDiscount : ℝ) (additionalDiscount : ℝ) (salesTax : ℝ) : ℝ :=
  let priceAfterInitialDiscount := originalPrice * (1 - initialDiscount)
  let priceAfterAdditionalDiscount := priceAfterInitialDiscount - additionalDiscount
  priceAfterAdditionalDiscount * (1 + salesTax)

/-- Theorem stating that the final price of the coat is $112.75 -/
theorem coat_price_calculation :
  finalPrice 150 0.25 10 0.1 = 112.75 := by
  sorry

end coat_price_calculation_l1217_121771


namespace pizza_consumption_order_l1217_121760

/-- Represents the amount of pizza eaten by each sibling -/
structure PizzaConsumption where
  alex : ℚ
  beth : ℚ
  cyril : ℚ
  dan : ℚ
  eve : ℚ

/-- Calculates the pizza consumption for each sibling based on the given conditions -/
def calculate_consumption : PizzaConsumption := {
  alex := 1/6,
  beth := 2/7,
  cyril := 1/3,
  dan := 1 - (1/6 + 2/7 + 1/3 + 1/8) - 2/168,
  eve := 1/8 + 2/168
}

/-- Represents the correct order of siblings based on pizza consumption -/
def correct_order := ["Cyril", "Beth", "Eve", "Alex", "Dan"]

/-- Theorem stating that the calculated consumption leads to the correct order -/
theorem pizza_consumption_order : 
  let c := calculate_consumption
  (c.cyril > c.beth) ∧ (c.beth > c.eve) ∧ (c.eve > c.alex) ∧ (c.alex > c.dan) := by
  sorry

#check pizza_consumption_order

end pizza_consumption_order_l1217_121760


namespace bird_nests_calculation_l1217_121711

/-- Calculates the total number of nests required for birds in a park --/
theorem bird_nests_calculation (total_birds : Nat) 
  (sparrows pigeons starlings : Nat)
  (sparrow_nests pigeon_nests starling_nests : Nat)
  (h1 : total_birds = sparrows + pigeons + starlings)
  (h2 : total_birds = 10)
  (h3 : sparrows = 4)
  (h4 : pigeons = 3)
  (h5 : starlings = 3)
  (h6 : sparrow_nests = 1)
  (h7 : pigeon_nests = 2)
  (h8 : starling_nests = 3) :
  sparrows * sparrow_nests + pigeons * pigeon_nests + starlings * starling_nests = 19 := by
  sorry

end bird_nests_calculation_l1217_121711


namespace sum_of_coefficients_l1217_121712

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ) :
  (∀ x : ℚ, (3 * x - 1)^7 = a₇ * x^7 + a₆ * x^6 + a₅ * x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a) →
  a₇ + a₆ + a₅ + a₄ + a₃ + a₂ + a₁ + a = 128 := by
sorry

end sum_of_coefficients_l1217_121712


namespace fish_population_estimate_l1217_121707

/-- Estimates the number of fish in a pond based on a capture-recapture method. -/
def estimate_fish_population (tagged_fish : ℕ) (second_catch : ℕ) (recaptured : ℕ) : ℕ :=
  (tagged_fish * second_catch) / recaptured

/-- Theorem stating that given the specific conditions of the problem, 
    the estimated fish population is 600. -/
theorem fish_population_estimate :
  let tagged_fish : ℕ := 30
  let second_catch : ℕ := 40
  let recaptured : ℕ := 2
  estimate_fish_population tagged_fish second_catch recaptured = 600 := by
  sorry

#eval estimate_fish_population 30 40 2

end fish_population_estimate_l1217_121707


namespace duck_cow_problem_l1217_121750

/-- Proves that in a group of ducks and cows, if the total number of legs is 32 more than twice the number of heads, then the number of cows is 16 -/
theorem duck_cow_problem (ducks cows : ℕ) : 
  2 * (ducks + cows) + 32 = 2 * ducks + 4 * cows → cows = 16 := by
  sorry

end duck_cow_problem_l1217_121750


namespace cone_sphere_equal_volume_l1217_121795

theorem cone_sphere_equal_volume (r : ℝ) (h : ℝ) :
  r = 1 →
  (1/3 * π * r^2 * h) = (4/3 * π) →
  Real.sqrt (r^2 + h^2) = Real.sqrt 17 :=
by sorry

end cone_sphere_equal_volume_l1217_121795


namespace abs_neg_three_equals_three_l1217_121744

theorem abs_neg_three_equals_three : abs (-3 : ℝ) = 3 := by sorry

end abs_neg_three_equals_three_l1217_121744


namespace garden_ratio_l1217_121759

theorem garden_ratio (area width length : ℝ) : 
  area = 588 → width = 14 → length * width = area → (length / width = 3) := by
  sorry

end garden_ratio_l1217_121759


namespace sum_of_factors_30_l1217_121749

def factors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ x => n % x = 0) (Finset.range (n + 1))

theorem sum_of_factors_30 : (factors 30).sum id = 72 := by
  sorry

end sum_of_factors_30_l1217_121749


namespace cookie_chips_count_l1217_121700

/-- Calculates the number of chocolate chips per cookie given the total chips,
    number of batches, and cookies per batch. -/
def chips_per_cookie (total_chips : ℕ) (num_batches : ℕ) (cookies_per_batch : ℕ) : ℕ :=
  total_chips / (num_batches * cookies_per_batch)

/-- Proves that there are 9 chocolate chips per cookie given the problem conditions. -/
theorem cookie_chips_count :
  let total_chips : ℕ := 81
  let num_batches : ℕ := 3
  let cookies_per_batch : ℕ := 3
  chips_per_cookie total_chips num_batches cookies_per_batch = 9 := by
  sorry

end cookie_chips_count_l1217_121700


namespace prime_power_plus_three_l1217_121756

theorem prime_power_plus_three (p : ℕ) : 
  Prime p → Prime (p^4 + 3) → p^5 + 3 = 35 := by sorry

end prime_power_plus_three_l1217_121756


namespace combined_salaries_of_abce_l1217_121713

def average_salary : ℕ := 8800
def number_of_people : ℕ := 5
def d_salary : ℕ := 7000

theorem combined_salaries_of_abce :
  (average_salary * number_of_people) - d_salary = 37000 := by
  sorry

end combined_salaries_of_abce_l1217_121713


namespace system_solution_l1217_121788

theorem system_solution :
  ∃ (x y : ℚ), (7 * x = -5 - 3 * y) ∧ (4 * x = 5 * y - 34) :=
by
  use (-127/47), (218/47)
  sorry

end system_solution_l1217_121788


namespace teresas_pencils_l1217_121725

/-- Teresa's pencil distribution problem -/
theorem teresas_pencils (colored_pencils black_pencils : ℕ) 
  (num_siblings pencils_per_sibling : ℕ) : 
  colored_pencils = 14 →
  black_pencils = 35 →
  num_siblings = 3 →
  pencils_per_sibling = 13 →
  colored_pencils + black_pencils - num_siblings * pencils_per_sibling = 10 :=
by sorry

end teresas_pencils_l1217_121725


namespace smallest_k_for_f_iteration_zero_l1217_121753

def f (a b M : ℕ) (n : ℤ) : ℤ :=
  if n < M then n + a else n - b

def iterateF (a b M : ℕ) : ℕ → ℤ → ℤ
  | 0, n => n
  | k+1, n => f a b M (iterateF a b M k n)

theorem smallest_k_for_f_iteration_zero (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ b) :
  let M := (a + b) / 2
  ∃ k : ℕ, k = (a + b) / Nat.gcd a b ∧ 
    iterateF a b M k 0 = 0 ∧ 
    ∀ j : ℕ, j < k → iterateF a b M j 0 ≠ 0 :=
sorry

end smallest_k_for_f_iteration_zero_l1217_121753


namespace other_endpoint_of_diameter_l1217_121721

/-- A circle in a 2D coordinate plane --/
structure Circle where
  center : ℝ × ℝ

/-- A diameter of a circle --/
structure Diameter where
  circle : Circle
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ

/-- The given circle P --/
def circleP : Circle :=
  { center := (3, 4) }

/-- The diameter of circle P --/
def diameterP : Diameter :=
  { circle := circleP
    endpoint1 := (0, 0)
    endpoint2 := (-3, -4) }

/-- Theorem: The other endpoint of the diameter is at (-3, -4) --/
theorem other_endpoint_of_diameter :
  diameterP.endpoint2 = (-3, -4) := by
  sorry

#check other_endpoint_of_diameter

end other_endpoint_of_diameter_l1217_121721


namespace negative_number_identification_l1217_121798

theorem negative_number_identification :
  (0 ≥ 0) ∧ ((1/2 : ℝ) > 0) ∧ (-(-5) > 0) ∧ (-Real.sqrt 5 < 0) := by
  sorry

end negative_number_identification_l1217_121798


namespace complex_number_magnitude_l1217_121777

theorem complex_number_magnitude (z : ℂ) :
  (1 - z) / (1 + z) = Complex.I ^ 2018 + Complex.I ^ 2019 →
  Complex.abs (2 + z) = 5 * Real.sqrt 2 / 2 := by
sorry

end complex_number_magnitude_l1217_121777


namespace folded_hexagon_result_verify_interior_angle_sum_l1217_121705

/-- Represents the possible polygons resulting from folding a regular hexagon in half -/
inductive FoldedHexagonShape
  | Quadrilateral
  | Pentagon

/-- Calculates the sum of interior angles for a polygon with n sides -/
def sumOfInteriorAngles (n : ℕ) : ℕ := (n - 2) * 180

/-- Represents the result of folding a regular hexagon in half -/
structure FoldedHexagonResult where
  shape : FoldedHexagonShape
  interiorAngleSum : ℕ

/-- Theorem stating the possible results of folding a regular hexagon in half -/
theorem folded_hexagon_result :
  ∃ (result : FoldedHexagonResult),
    (result.shape = FoldedHexagonShape.Quadrilateral ∧ result.interiorAngleSum = 360) ∨
    (result.shape = FoldedHexagonShape.Pentagon ∧ result.interiorAngleSum = 540) :=
by
  sorry

/-- Verification that the sum of interior angles is correct for each shape -/
theorem verify_interior_angle_sum :
  ∀ (result : FoldedHexagonResult),
    (result.shape = FoldedHexagonShape.Quadrilateral → result.interiorAngleSum = sumOfInteriorAngles 4) ∧
    (result.shape = FoldedHexagonShape.Pentagon → result.interiorAngleSum = sumOfInteriorAngles 5) :=
by
  sorry

end folded_hexagon_result_verify_interior_angle_sum_l1217_121705


namespace inequality_implies_not_equal_l1217_121765

theorem inequality_implies_not_equal (a b : ℝ) :
  (a / b + b / a > 2) → (a ≠ b) ∧ ¬(∀ a b : ℝ, a ≠ b → a / b + b / a > 2) :=
sorry

end inequality_implies_not_equal_l1217_121765


namespace function_inequality_l1217_121764

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x > 1, (x - 1) * (deriv f x) - f x > 0) :
  (1 / (Real.sqrt 2 - 1)) * f (Real.sqrt 2) < f 2 ∧ f 2 < (1 / 2) * f 3 := by
  sorry

end function_inequality_l1217_121764


namespace election_votes_l1217_121758

theorem election_votes (winning_percentage : ℝ) (majority : ℕ) (total_votes : ℕ) : 
  winning_percentage = 0.6 →
  majority = 1380 →
  total_votes * winning_percentage - total_votes * (1 - winning_percentage) = majority →
  total_votes = 6900 := by
sorry

end election_votes_l1217_121758


namespace max_triangles_is_28_l1217_121766

/-- The number of points on the hypotenuse of a right triangle with legs of length 7 -/
def hypotenuse_points : ℕ := 8

/-- The maximum number of triangles that can be formed within the right triangle -/
def max_triangles : ℕ := Nat.choose hypotenuse_points 2

/-- Theorem stating the maximum number of triangles is 28 -/
theorem max_triangles_is_28 : max_triangles = 28 := by sorry

end max_triangles_is_28_l1217_121766


namespace fraction_of_decimals_cubed_and_squared_l1217_121743

theorem fraction_of_decimals_cubed_and_squared :
  (0.3 ^ 3) / (0.03 ^ 2) = 30 := by
  sorry

end fraction_of_decimals_cubed_and_squared_l1217_121743


namespace expression_simplification_l1217_121704

theorem expression_simplification :
  3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (Real.sqrt 3 - 3)) = 3 + (2 * Real.sqrt 3) / 3 := by
  sorry

end expression_simplification_l1217_121704


namespace inequality_solution_l1217_121776

theorem inequality_solution (x : ℝ) : 
  (8 * x^2 + 16 * x - 51) / ((2 * x - 3) * (x + 4)) < 3 ↔ 
  (x > -4 ∧ x < -3) ∨ (x > 3/2 ∧ x < 5/2) := by sorry

end inequality_solution_l1217_121776


namespace triangle_angle_cosine_inequality_l1217_121722

theorem triangle_angle_cosine_inequality (α β γ : Real) 
  (h : α + β + γ = Real.pi) : 
  Real.cos (α + Real.pi / 3) + Real.cos (β + Real.pi / 3) + Real.cos (γ + Real.pi / 3) + 3 / 2 ≥ 0 := by
  sorry

end triangle_angle_cosine_inequality_l1217_121722


namespace valid_numbers_l1217_121726

def is_valid_number (n : ℕ) : Prop :=
  n % 2 = 0 ∧ (Nat.divisors n).card = n / 2

theorem valid_numbers : {n : ℕ | is_valid_number n} = {8, 12} := by sorry

end valid_numbers_l1217_121726


namespace remainder_after_division_l1217_121767

theorem remainder_after_division (n : ℕ) : 
  (n / 7 = 12 ∧ n % 7 = 5) → n % 8 = 1 := by
sorry

end remainder_after_division_l1217_121767


namespace shortest_side_is_15_l1217_121740

/-- Represents a triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  perimeter_eq : a + b + c = 72
  side_eq : a = 30

/-- Calculates the semiperimeter of a triangle -/
def semiperimeter (t : IntTriangle) : ℚ :=
  (t.a + t.b + t.c) / 2

/-- Calculates the area of a triangle using Heron's formula -/
def area (t : IntTriangle) : ℚ :=
  let s := semiperimeter t
  (s * (s - t.a) * (s - t.b) * (s - t.c)).sqrt

/-- Main theorem: The shortest side of the triangle is 15 -/
theorem shortest_side_is_15 (t : IntTriangle) (area_int : ∃ n : ℕ, area t = n) :
  min t.a (min t.b t.c) = 15 := by
  sorry

#check shortest_side_is_15

end shortest_side_is_15_l1217_121740


namespace other_endpoint_of_line_segment_l1217_121778

/-- Given a line segment with midpoint (2, 3) and one endpoint (-1, 7),
    prove that the other endpoint is (5, -1). -/
theorem other_endpoint_of_line_segment (A B M : ℝ × ℝ) : 
  M = (2, 3) → A = (-1, 7) → M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → B = (5, -1) := by
  sorry

end other_endpoint_of_line_segment_l1217_121778


namespace square_land_perimeter_l1217_121790

theorem square_land_perimeter (a : ℝ) (h : 5 * a = 10 * (4 * Real.sqrt a) + 45) :
  4 * Real.sqrt a = 36 := by
  sorry

end square_land_perimeter_l1217_121790


namespace fish_in_tank_l1217_121763

theorem fish_in_tank (total : ℕ) (blue : ℕ) (spotted : ℕ) : 
  3 * blue = total → 
  2 * spotted = blue → 
  spotted = 10 → 
  total = 60 := by
sorry

end fish_in_tank_l1217_121763


namespace complex_sum_on_real_axis_l1217_121734

theorem complex_sum_on_real_axis (a : ℝ) : 
  let z₁ : ℂ := 2 + I
  let z₂ : ℂ := 3 + a * I
  (z₁ + z₂).im = 0 → a = -1 := by
sorry

end complex_sum_on_real_axis_l1217_121734


namespace quadratic_form_value_l1217_121724

theorem quadratic_form_value (x y : ℝ) 
  (eq1 : 4 * x + y = 12) 
  (eq2 : x + 4 * y = 18) : 
  20 * x^2 + 24 * x * y + 20 * y^2 = 468 := by
  sorry

end quadratic_form_value_l1217_121724


namespace ball_box_probabilities_l1217_121754

/-- The number of different balls -/
def num_balls : ℕ := 4

/-- The number of different boxes -/
def num_boxes : ℕ := 4

/-- The total number of possible outcomes when placing balls into boxes -/
def total_outcomes : ℕ := num_boxes ^ num_balls

/-- The probability of no empty boxes when placing balls into boxes -/
def prob_no_empty_boxes : ℚ := 3 / 32

/-- The probability of exactly one empty box when placing balls into boxes -/
def prob_one_empty_box : ℚ := 9 / 16

/-- Theorem stating the probabilities for different scenarios when placing balls into boxes -/
theorem ball_box_probabilities :
  (prob_no_empty_boxes = 3 / 32) ∧ (prob_one_empty_box = 9 / 16) := by
  sorry

end ball_box_probabilities_l1217_121754


namespace circle_center_and_radius_l1217_121748

/-- Given a circle with equation x^2 + y^2 - 4x + 2y - 4 = 0, 
    prove that its center is at (2, -1) and its radius is 3. -/
theorem circle_center_and_radius : 
  ∃ (C : ℝ × ℝ) (r : ℝ), 
    (C = (2, -1) ∧ r = 3) ∧ 
    ∀ (x y : ℝ), x^2 + y^2 - 4*x + 2*y - 4 = 0 ↔ (x - C.1)^2 + (y - C.2)^2 = r^2 :=
by sorry

end circle_center_and_radius_l1217_121748


namespace common_terms_count_l1217_121768

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  start : ℝ
  diff : ℝ
  length : ℕ

/-- Returns the nth term of an arithmetic sequence -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.start + (n - 1 : ℝ) * seq.diff

/-- Counts the number of common terms between two arithmetic sequences -/
def countCommonTerms (seq1 seq2 : ArithmeticSequence) : ℕ :=
  (seq1.length).min seq2.length

theorem common_terms_count (seq1 seq2 : ArithmeticSequence) 
  (h1 : seq1.start = 5 ∧ seq1.diff = 3 ∧ seq1.length = 100)
  (h2 : seq2.start = 3 ∧ seq2.diff = 5 ∧ seq2.length = 100) :
  countCommonTerms seq1 seq2 = 20 := by
  sorry

#check common_terms_count

end common_terms_count_l1217_121768


namespace bicycle_spokes_l1217_121714

/-- Represents a bicycle with front and back wheels -/
structure Bicycle where
  front_spokes : ℕ
  back_spokes : ℕ

/-- Calculates the total number of spokes on a bicycle -/
def total_spokes (b : Bicycle) : ℕ :=
  b.front_spokes + b.back_spokes

/-- Theorem: A bicycle with 20 front spokes and twice as many back spokes has 60 spokes in total -/
theorem bicycle_spokes :
  ∀ b : Bicycle, b.front_spokes = 20 ∧ b.back_spokes = 2 * b.front_spokes →
  total_spokes b = 60 := by
  sorry

end bicycle_spokes_l1217_121714


namespace peters_walked_distance_l1217_121784

/-- Calculates the distance Peter has already walked given the total distance,
    his walking speed, and the remaining time to reach the store. -/
theorem peters_walked_distance
  (total_distance : ℝ)
  (walking_speed : ℝ)
  (remaining_time : ℝ)
  (h1 : total_distance = 2.5)
  (h2 : walking_speed = 1 / 20)
  (h3 : remaining_time = 30) :
  total_distance - (walking_speed * remaining_time) = 1 := by
  sorry

#check peters_walked_distance

end peters_walked_distance_l1217_121784


namespace chromium_percentage_in_cast_iron_l1217_121761

theorem chromium_percentage_in_cast_iron 
  (x y : ℝ) 
  (h1 : 5 * x + y = 6 * min x y) 
  (h2 : x + y = 0.16) : 
  (x = 0.11 ∧ y = 0.05) ∨ (x = 0.05 ∧ y = 0.11) :=
sorry

end chromium_percentage_in_cast_iron_l1217_121761


namespace toy_store_optimization_l1217_121741

/-- Toy store profit optimization problem --/
theorem toy_store_optimization :
  let initial_price : ℝ := 120
  let initial_cost : ℝ := 80
  let initial_sales : ℝ := 20
  let price_reduction (x : ℝ) := x
  let sales_increase (x : ℝ) := 2 * x
  let new_price (x : ℝ) := initial_price - price_reduction x
  let new_sales (x : ℝ) := initial_sales + sales_increase x
  let profit (x : ℝ) := (new_price x - initial_cost) * new_sales x

  -- Daily sales function
  ∀ x, new_sales x = 20 + 2*x ∧

  -- Profit function and domain
  (∀ x, profit x = -2*x^2 + 60*x + 800) ∧
  (∀ x, 0 < x → x ≤ 40 → new_price x ≥ initial_cost) ∧

  -- Maximum profit
  ∃ x, 0 < x ∧ x ≤ 40 ∧ 
    profit x = 1250 ∧
    (∀ y, 0 < y → y ≤ 40 → profit y ≤ profit x) ∧
    new_price x = 105 :=
by sorry

end toy_store_optimization_l1217_121741


namespace min_value_theorem_l1217_121792

theorem min_value_theorem (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x + y ≤ 2) :
  ∃ (min_val : ℝ), min_val = (3 + 2 * Real.sqrt 2) / 4 ∧
  ∀ (z : ℝ), z = 2 / (x + 3 * y) + 1 / (x - y) → z ≥ min_val := by
  sorry

end min_value_theorem_l1217_121792


namespace anna_basketball_score_product_l1217_121742

def first_10_games : List ℕ := [5, 7, 9, 2, 6, 10, 5, 7, 8, 4]

theorem anna_basketball_score_product :
  ∀ (game11 game12 : ℕ),
  game11 < 15 ∧ game12 < 15 →
  (List.sum first_10_games + game11) % 11 = 0 →
  (List.sum first_10_games + game11 + game12) % 12 = 0 →
  game11 * game12 = 18 := by
  sorry

end anna_basketball_score_product_l1217_121742


namespace sticker_difference_l1217_121751

theorem sticker_difference (karl_stickers : ℕ) (ryan_more_than_karl : ℕ) (total_stickers : ℕ)
  (h1 : karl_stickers = 25)
  (h2 : ryan_more_than_karl = 20)
  (h3 : total_stickers = 105) :
  let ryan_stickers := karl_stickers + ryan_more_than_karl
  let ben_stickers := total_stickers - karl_stickers - ryan_stickers
  ryan_stickers - ben_stickers = 10 := by
sorry

end sticker_difference_l1217_121751


namespace cube_volume_problem_l1217_121745

theorem cube_volume_problem (a : ℝ) : 
  (a > 0) → 
  (a^3 - ((a - 2) * a * (a + 2)) = 16) → 
  (a^3 = 64) :=
by sorry

end cube_volume_problem_l1217_121745


namespace domain_of_f_l1217_121701

def f (x : ℝ) : ℝ := (2 * x - 3) ^ (1/3) + (5 - 2 * x) ^ (1/3)

theorem domain_of_f : Set.univ = {x : ℝ | ∃ y, f x = y} := by
  sorry

end domain_of_f_l1217_121701


namespace angle_D_value_l1217_121719

-- Define the angles as real numbers
variable (A B C D : ℝ)

-- State the given conditions
axiom angle_sum : A + B = 180
axiom angle_relation : C = D + 10
axiom angle_A : A = 50
axiom triangle_sum : B + C + D = 180

-- State the theorem to be proved
theorem angle_D_value : D = 20 := by
  sorry

end angle_D_value_l1217_121719


namespace max_value_of_function_l1217_121746

theorem max_value_of_function (x : ℝ) : 
  1 / (x^2 + 2) ≤ 1 / 2 ∧ ∃ y : ℝ, 1 / (y^2 + 2) = 1 / 2 :=
by sorry

end max_value_of_function_l1217_121746


namespace distance_p_to_y_axis_l1217_121717

/-- The distance from a point to the y-axis is the absolute value of its x-coordinate. -/
def distance_to_y_axis (x y : ℝ) : ℝ := |x|

/-- Given point P(-3, 5), prove that its distance to the y-axis is 3. -/
theorem distance_p_to_y_axis :
  let P : ℝ × ℝ := (-3, 5)
  distance_to_y_axis P.1 P.2 = 3 := by
  sorry

end distance_p_to_y_axis_l1217_121717


namespace tangent_line_condition_l1217_121772

/-- Given a function f(x) = e^x + a*cos(x), if its tangent line at x = 0 passes through (1, 6), then a = 4 -/
theorem tangent_line_condition (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ Real.exp x + a * Real.cos x
  let f' : ℝ → ℝ := λ x ↦ Real.exp x - a * Real.sin x
  (f 0 = 6) ∧ (f' 0 = 1) → a = 4 := by sorry

end tangent_line_condition_l1217_121772


namespace poison_frog_count_l1217_121782

theorem poison_frog_count (total : ℕ) (tree : ℕ) (wood : ℕ) (poison : ℕ) :
  total = 78 →
  tree = 55 →
  wood = 13 →
  poison = total - (tree + wood) →
  poison = 10 := by
  sorry

end poison_frog_count_l1217_121782


namespace ellipse_and_line_properties_l1217_121774

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 2 + y^2 = 1

/-- Definition of the line l -/
def line_l (x y m : ℝ) : Prop :=
  y = x + m

/-- Theorem stating the properties of the ellipse and the intersecting line -/
theorem ellipse_and_line_properties :
  ∃ (a b : ℝ),
    -- Foci conditions
    ((-1 : ℝ)^2 + 0^2 = a^2 - b^2) ∧
    ((1 : ℝ)^2 + 0^2 = a^2 - b^2) ∧
    -- Point P on the ellipse
    ellipse_C 1 (Real.sqrt 2 / 2) ∧
    -- Standard equation of the ellipse
    (∀ x y, ellipse_C x y ↔ x^2 / 2 + y^2 = 1) ∧
    -- Maximum intersection distance occurs when m = 0
    (∀ m, ∃ x₁ y₁ x₂ y₂,
      line_l x₁ y₁ m ∧ line_l x₂ y₂ m ∧
      ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
      (x₂ - x₁)^2 + (y₂ - y₁)^2 ≤ (2 : ℝ)^2 + (2 : ℝ)^2) ∧
    -- The line y = x achieves this maximum
    (∃ x₁ y₁ x₂ y₂,
      line_l x₁ y₁ 0 ∧ line_l x₂ y₂ 0 ∧
      ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
      (x₂ - x₁)^2 + (y₂ - y₁)^2 = (2 : ℝ)^2 + (2 : ℝ)^2) :=
by sorry

end ellipse_and_line_properties_l1217_121774


namespace distance_difference_l1217_121736

theorem distance_difference (john_distance nina_distance : ℝ) 
  (h1 : john_distance = 0.7)
  (h2 : nina_distance = 0.4) :
  john_distance - nina_distance = 0.3 := by
sorry

end distance_difference_l1217_121736


namespace inequality_solution_set_l1217_121720

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | (x + 1) * (x - a) < 0}
  if a > -1 then
    S = {x : ℝ | -1 < x ∧ x < a}
  else if a = -1 then
    S = ∅
  else
    S = {x : ℝ | a < x ∧ x < -1} :=
by
  sorry

end inequality_solution_set_l1217_121720


namespace floor_painting_rate_l1217_121783

/-- Proves that the painting rate is 3 Rs. per square meter for a rectangular floor with given conditions --/
theorem floor_painting_rate (length breadth area cost : ℝ) : 
  length = 3 * breadth →
  length = 15.491933384829668 →
  area = length * breadth →
  cost = 240 →
  cost / area = 3 := by sorry

end floor_painting_rate_l1217_121783


namespace middle_number_between_52_and_certain_number_l1217_121723

theorem middle_number_between_52_and_certain_number 
  (certain_number : ℕ) 
  (h1 : certain_number > 52) 
  (h2 : ∃ (n : ℕ), n ≥ 52 ∧ n < certain_number ∧ certain_number - 52 - 1 = 15) :
  (52 + certain_number) / 2 = 60 :=
sorry

end middle_number_between_52_and_certain_number_l1217_121723


namespace solutions_difference_squared_l1217_121708

theorem solutions_difference_squared (α β : ℝ) : 
  α ≠ β ∧ 
  α^2 - 3*α + 1 = 0 ∧ 
  β^2 - 3*β + 1 = 0 → 
  (α - β)^2 = 5 :=
by sorry

end solutions_difference_squared_l1217_121708


namespace min_distance_to_point_l1217_121716

/-- The line equation ax + by + 1 = 0 -/
def line_equation (a b x y : ℝ) : Prop := a * x + b * y + 1 = 0

/-- The circle equation x^2 + y^2 + 4x + 2y + 1 = 0 -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 2*y + 1 = 0

/-- The line always bisects the circumference of the circle -/
def line_bisects_circle (a b : ℝ) : Prop :=
  ∀ x y : ℝ, line_equation a b x y → circle_equation x y

/-- The theorem to be proved -/
theorem min_distance_to_point (a b : ℝ) 
  (h : line_bisects_circle a b) : 
  (∀ a' b' : ℝ, line_bisects_circle a' b' → (a-2)^2 + (b-2)^2 ≤ (a'-2)^2 + (b'-2)^2) ∧
  (a-2)^2 + (b-2)^2 = 5 :=
sorry

end min_distance_to_point_l1217_121716


namespace area_of_triangle_ABF_l1217_121794

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- A square defined by four points -/
structure Square where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Check if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Check if a point is inside a square -/
def isInside (p : Point) (s : Square) : Prop := sorry

/-- Find the intersection point of two line segments -/
def intersectionPoint (p1 p2 p3 p4 : Point) : Point := sorry

/-- Calculate the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

theorem area_of_triangle_ABF 
  (A B C D E F : Point)
  (square : Square)
  (triangle : Triangle)
  (h1 : square = Square.mk A B C D)
  (h2 : triangle = Triangle.mk A B E)
  (h3 : isEquilateral triangle)
  (h4 : isInside E square)
  (h5 : F = intersectionPoint B D A E)
  (h6 : (B.x - A.x)^2 + (B.y - A.y)^2 = 1 + Real.sqrt 3) :
  triangleArea (Triangle.mk A B F) = Real.sqrt 3 / 2 := by
  sorry

end area_of_triangle_ABF_l1217_121794


namespace magic_square_property_l1217_121710

def magic_square : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![7.5, 5, 2.5],
    ![0, 5, 10],
    ![7.5, 5, 2.5]]

def row_sum (m : Matrix (Fin 3) (Fin 3) ℚ) (i : Fin 3) : ℚ :=
  m i 0 + m i 1 + m i 2

def col_sum (m : Matrix (Fin 3) (Fin 3) ℚ) (j : Fin 3) : ℚ :=
  m 0 j + m 1 j + m 2 j

def diag_sum (m : Matrix (Fin 3) (Fin 3) ℚ) : ℚ :=
  m 0 0 + m 1 1 + m 2 2

def anti_diag_sum (m : Matrix (Fin 3) (Fin 3) ℚ) : ℚ :=
  m 0 2 + m 1 1 + m 2 0

theorem magic_square_property :
  (∀ i : Fin 3, row_sum magic_square i = 15) ∧
  (∀ j : Fin 3, col_sum magic_square j = 15) ∧
  diag_sum magic_square = 15 ∧
  anti_diag_sum magic_square = 15 := by
  sorry

end magic_square_property_l1217_121710


namespace solution_set_inequality_l1217_121769

theorem solution_set_inequality (x : ℝ) :
  x * (x - 1) > 0 ↔ x < 0 ∨ x > 1 := by sorry

end solution_set_inequality_l1217_121769


namespace maya_books_last_week_l1217_121752

/-- The number of pages in each book Maya reads. -/
def pages_per_book : ℕ := 300

/-- The total number of pages Maya read over two weeks. -/
def total_pages : ℕ := 4500

/-- The ratio of pages read this week compared to last week. -/
def week_ratio : ℕ := 2

/-- The number of books Maya read last week. -/
def books_last_week : ℕ := 5

theorem maya_books_last_week :
  books_last_week * pages_per_book * (week_ratio + 1) = total_pages :=
sorry

end maya_books_last_week_l1217_121752


namespace product_equals_29_l1217_121773

theorem product_equals_29 (a : ℝ) (h : a^2 + a + 1 = 2) : (5 - a) * (6 + a) = 29 := by
  sorry

end product_equals_29_l1217_121773


namespace complex_modulus_problem_l1217_121789

theorem complex_modulus_problem (a b : ℝ) :
  (Complex.mk a 1) * (Complex.mk 1 (-1)) = Complex.mk 3 b →
  Complex.abs (Complex.mk a b) = Real.sqrt 5 := by
  sorry

end complex_modulus_problem_l1217_121789


namespace pentagon_largest_angle_l1217_121757

theorem pentagon_largest_angle (a b c d e : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 → -- All angles are positive
  b / a = 3 / 2 ∧ c / a = 2 ∧ d / a = 5 / 2 ∧ e / a = 3 → -- Angles are in ratio 2:3:4:5:6
  a + b + c + d + e = 540 → -- Sum of angles in a pentagon
  e = 162 := by
sorry

end pentagon_largest_angle_l1217_121757


namespace angle_C_value_max_area_l1217_121786

open Real

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions of the problem
def satisfiesCondition (t : Triangle) : Prop :=
  (2 * t.a + t.b) / t.c = cos (t.A + t.C) / cos t.C

-- Theorem 1: If the condition is satisfied, then C = 2π/3
theorem angle_C_value (t : Triangle) (h : satisfiesCondition t) : t.C = 2 * π / 3 := by
  sorry

-- Theorem 2: Maximum area when c = 2 and C = 2π/3
theorem max_area (t : Triangle) (h1 : t.c = 2) (h2 : t.C = 2 * π / 3) :
  ∃ (maxArea : ℝ), maxArea = Real.sqrt 3 / 3 ∧
  ∀ (s : ℝ), s = (1 / 2) * t.a * t.b * sin t.C → s ≤ maxArea := by
  sorry

end angle_C_value_max_area_l1217_121786


namespace equation_solution_l1217_121709

theorem equation_solution : 
  ∀ x : ℝ, (x^2 + x)^2 - 4*(x^2 + x) - 12 = 0 ↔ x = -3 ∨ x = 2 :=
by sorry

end equation_solution_l1217_121709


namespace x_fourth_power_zero_l1217_121793

theorem x_fourth_power_zero (x : ℝ) (h_pos : x > 0) 
  (h_eq : Real.sqrt (1 - x^2) + Real.sqrt (1 + x^2) = 2) : x^4 = 0 := by
  sorry

end x_fourth_power_zero_l1217_121793


namespace max_value_sqrt_xy_1_minus_x_2y_l1217_121781

theorem max_value_sqrt_xy_1_minus_x_2y :
  ∀ x y : ℝ, x > 0 → y > 0 →
  Real.sqrt (x * y) * (1 - x - 2 * y) ≤ Real.sqrt 2 / 16 ∧
  (Real.sqrt (x * y) * (1 - x - 2 * y) = Real.sqrt 2 / 16 ↔ x = 1/4) :=
by sorry

end max_value_sqrt_xy_1_minus_x_2y_l1217_121781


namespace classroom_ratio_problem_l1217_121732

theorem classroom_ratio_problem (total_students : ℕ) (girl_ratio boy_ratio : ℕ) 
  (h1 : total_students = 30)
  (h2 : girl_ratio = 1)
  (h3 : boy_ratio = 2) : 
  (total_students * boy_ratio) / (girl_ratio + boy_ratio) = 20 := by
  sorry

end classroom_ratio_problem_l1217_121732


namespace similarity_criteria_l1217_121702

/-- A structure representing a triangle -/
structure Triangle where
  -- We'll assume triangles are defined by their side lengths and angles
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

/-- Two triangles are similar if they have the same shape but not necessarily the same size -/
def similar (t1 t2 : Triangle) : Prop :=
  sorry

/-- SSS (Side-Side-Side) Similarity: Two triangles are similar if their corresponding sides are proportional -/
def SSS_similarity (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ 
    t1.side1 / t2.side1 = k ∧
    t1.side2 / t2.side2 = k ∧
    t1.side3 / t2.side3 = k

/-- SAS (Side-Angle-Side) Similarity: Two triangles are similar if two pairs of corresponding sides are proportional and the included angles are equal -/
def SAS_similarity (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ 
    t1.side1 / t2.side1 = k ∧
    t1.side2 / t2.side2 = k ∧
    t1.angle3 = t2.angle3

/-- Theorem: Two triangles are similar if and only if they satisfy either SSS or SAS similarity criteria -/
theorem similarity_criteria (t1 t2 : Triangle) :
  similar t1 t2 ↔ SSS_similarity t1 t2 ∨ SAS_similarity t1 t2 :=
sorry

end similarity_criteria_l1217_121702


namespace inverse_variation_problem_l1217_121785

/-- Given that x² varies inversely with y⁴, prove that if x = 10 when y = 2, then x = 5 when y = √8 -/
theorem inverse_variation_problem (x y : ℝ) (k : ℝ) (h1 : x^2 * y^4 = k) (h2 : x = 10 ∧ y = 2) :
  x = 5 ∧ y = Real.sqrt 8 :=
by sorry

end inverse_variation_problem_l1217_121785


namespace initial_orchids_l1217_121703

theorem initial_orchids (initial_roses : ℕ) (final_orchids : ℕ) (final_roses : ℕ) :
  initial_roses = 9 →
  final_orchids = 13 →
  final_roses = 3 →
  final_orchids - final_roses = 10 →
  ∃ initial_orchids : ℕ, initial_orchids = 3 :=
by sorry

end initial_orchids_l1217_121703


namespace teresa_total_score_l1217_121775

def teresa_scores (science music social_studies : ℕ) : Prop :=
  ∃ (physics total : ℕ),
    physics = music / 2 ∧
    total = science + music + social_studies + physics

theorem teresa_total_score :
  teresa_scores 70 80 85 → ∃ total : ℕ, total = 275 :=
by sorry

end teresa_total_score_l1217_121775


namespace modulus_of_Z_l1217_121738

theorem modulus_of_Z (Z : ℂ) (h : (1 + Complex.I) * Z = Complex.I) : 
  Complex.abs Z = Real.sqrt 2 / 2 := by sorry

end modulus_of_Z_l1217_121738


namespace unique_sums_count_l1217_121762

def bag_A : Finset ℕ := {1, 4, 9}
def bag_B : Finset ℕ := {16, 25, 36}

def possible_sums : Finset ℕ := (bag_A.product bag_B).image (fun p => p.1 + p.2)

theorem unique_sums_count : possible_sums.card = 9 := by
  sorry

end unique_sums_count_l1217_121762


namespace tetrahedra_arrangement_exists_l1217_121728

/-- A type representing a regular tetrahedron -/
structure Tetrahedron where
  -- Add necessary fields

/-- A type representing the arrangement of tetrahedra -/
structure Arrangement where
  tetrahedra : Set Tetrahedron
  lower_plane : Set (ℝ × ℝ × ℝ)
  upper_plane : Set (ℝ × ℝ × ℝ)

/-- Predicate to check if two planes are parallel -/
def are_parallel (plane1 plane2 : Set (ℝ × ℝ × ℝ)) : Prop :=
  sorry

/-- Predicate to check if a tetrahedron is between two planes -/
def is_between_planes (t : Tetrahedron) (lower upper : Set (ℝ × ℝ × ℝ)) : Prop :=
  sorry

/-- Predicate to check if a tetrahedron can be removed without moving others -/
def can_be_removed (t : Tetrahedron) (arr : Arrangement) : Prop :=
  sorry

/-- The main theorem statement -/
theorem tetrahedra_arrangement_exists :
  ∃ (arr : Arrangement),
    (∀ t ∈ arr.tetrahedra, is_between_planes t arr.lower_plane arr.upper_plane) ∧
    (are_parallel arr.lower_plane arr.upper_plane) ∧
    (Set.Infinite arr.tetrahedra) ∧
    (∀ t ∈ arr.tetrahedra, ¬can_be_removed t arr) :=
  sorry

end tetrahedra_arrangement_exists_l1217_121728


namespace least_integer_with_deletion_property_l1217_121730

theorem least_integer_with_deletion_property : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (n = 17) ∧ 
  (∀ m : ℕ, m > 0 → m < n → 
    (m / 10 : ℚ) ≠ m / 17) ∧
  ((n / 10 : ℚ) = n / 17) := by
sorry

end least_integer_with_deletion_property_l1217_121730


namespace painting_time_calculation_l1217_121747

/-- Given an artist's weekly painting hours and production rate over four weeks,
    calculate the time needed to complete one painting. -/
theorem painting_time_calculation (weekly_hours : ℕ) (paintings_in_four_weeks : ℕ) :
  weekly_hours = 30 →
  paintings_in_four_weeks = 40 →
  (4 * weekly_hours) / paintings_in_four_weeks = 3 :=
by
  sorry

end painting_time_calculation_l1217_121747


namespace number_of_papers_l1217_121799

/-- Represents the marks obtained in each paper -/
structure PaperMarks where
  fullMarks : ℝ
  proportions : List ℝ
  totalPapers : ℕ

/-- Checks if the given PaperMarks satisfies the problem conditions -/
def satisfiesConditions (pm : PaperMarks) : Prop :=
  pm.proportions = [5, 6, 7, 8, 9] ∧
  pm.totalPapers = pm.proportions.length ∧
  (pm.proportions.sum * pm.fullMarks * 0.6 = pm.proportions.sum * pm.fullMarks) ∧
  (List.filter (fun p => p * pm.fullMarks > 0.5 * pm.fullMarks) pm.proportions).length = 5

/-- Theorem stating that if the conditions are satisfied, the number of papers is 5 -/
theorem number_of_papers (pm : PaperMarks) (h : satisfiesConditions pm) : pm.totalPapers = 5 := by
  sorry

end number_of_papers_l1217_121799


namespace article_cost_l1217_121737

theorem article_cost (selling_price_high : ℝ) (selling_price_low : ℝ) (cost : ℝ) :
  selling_price_high = 600 →
  selling_price_low = 580 →
  selling_price_high - cost = 1.05 * (selling_price_low - cost) →
  cost = 180 := by
  sorry

end article_cost_l1217_121737


namespace total_guests_calculation_l1217_121733

/-- Given the number of guests in different age groups, calculate the total number of guests served. -/
theorem total_guests_calculation (adults : ℕ) (h1 : adults = 58) : ∃ (children seniors teenagers toddlers : ℕ),
  children = adults - 35 ∧
  seniors = 2 * children ∧
  teenagers = seniors - 15 ∧
  toddlers = teenagers / 2 ∧
  adults + children + seniors + teenagers + toddlers = 173 := by
  sorry


end total_guests_calculation_l1217_121733


namespace newspaper_conference_max_overlap_l1217_121715

theorem newspaper_conference_max_overlap (total : ℕ) (writers : ℕ) (editors : ℕ) (x : ℕ) :
  total = 100 →
  writers = 40 →
  editors > 38 →
  (total = writers + editors - x + 2 * x) →
  (∀ y : ℕ, y > x → ¬(total = writers + editors - y + 2 * y)) →
  x = 21 :=
by sorry

end newspaper_conference_max_overlap_l1217_121715


namespace average_weight_increase_l1217_121735

/-- Proves that replacing a 70 kg person with a 110 kg person in a group of 10 increases the average weight by 4 kg -/
theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total := 10 * initial_average
  let new_total := initial_total - 70 + 110
  let new_average := new_total / 10
  new_average - initial_average = 4 := by
sorry

end average_weight_increase_l1217_121735


namespace no_solution_l1217_121727

theorem no_solution : ¬ ∃ (n : ℕ), (823435^15 % n = 0) ∧ (n^5 - n^n = 1) := by
  sorry

end no_solution_l1217_121727


namespace fallen_sheets_l1217_121770

def is_permutation (a b : Nat) : Prop :=
  (a.digits 10).sum = (b.digits 10).sum ∧
  (a.digits 10).prod = (b.digits 10).prod

theorem fallen_sheets (n : Nat) 
  (h1 : is_permutation 387 n)
  (h2 : n > 387)
  (h3 : Even n) :
  (n - 387 + 1) / 2 = 176 :=
sorry

end fallen_sheets_l1217_121770


namespace dog_care_time_ratio_l1217_121739

/-- Proves the ratio of blow-drying time to bathing time for Marcus and his dog --/
theorem dog_care_time_ratio 
  (total_time : ℕ) 
  (bath_time : ℕ) 
  (walk_speed : ℚ) 
  (walk_distance : ℚ) 
  (h1 : total_time = 60) 
  (h2 : bath_time = 20) 
  (h3 : walk_speed = 6) 
  (h4 : walk_distance = 3) : 
  (total_time - bath_time - (walk_distance / walk_speed * 60).floor) * 2 = bath_time := by
sorry


end dog_care_time_ratio_l1217_121739


namespace systematic_sampling_l1217_121797

/-- Systematic sampling problem -/
theorem systematic_sampling
  (population_size : ℕ)
  (sample_size : ℕ)
  (last_sample : ℕ)
  (h1 : population_size = 2000)
  (h2 : sample_size = 100)
  (h3 : last_sample = 1994)
  : ∃ (first_sample : ℕ), first_sample = 14 ∧
    last_sample = (sample_size - 1) * (population_size / sample_size) + first_sample :=
sorry

end systematic_sampling_l1217_121797


namespace harold_finances_theorem_l1217_121779

/-- Harold's monthly finances --/
def harold_finances (income rent car_payment groceries : ℚ) : Prop :=
  let utilities := car_payment / 2
  let total_expenses := rent + car_payment + utilities + groceries
  let remaining := income - total_expenses
  let retirement_savings := remaining / 2
  let final_remaining := remaining - retirement_savings
  income = 2500 ∧ 
  rent = 700 ∧ 
  car_payment = 300 ∧ 
  groceries = 50 ∧ 
  final_remaining = 650

theorem harold_finances_theorem :
  ∀ income rent car_payment groceries : ℚ,
  harold_finances income rent car_payment groceries :=
by
  sorry

end harold_finances_theorem_l1217_121779


namespace probability_multiple_2_3_or_5_l1217_121780

def is_multiple (n m : ℕ) : Prop := ∃ k, n = m * k

def count_multiples (max : ℕ) (divisor : ℕ) : ℕ :=
  (max / divisor : ℕ)

def count_multiples_of_2_3_or_5 (max : ℕ) : ℕ :=
  count_multiples max 2 + count_multiples max 3 + count_multiples max 5 -
  (count_multiples max 6 + count_multiples max 10 + count_multiples max 15) +
  count_multiples max 30

theorem probability_multiple_2_3_or_5 :
  (count_multiples_of_2_3_or_5 120 : ℚ) / 120 = 11 / 15 := by
  sorry

#eval count_multiples_of_2_3_or_5 120

end probability_multiple_2_3_or_5_l1217_121780


namespace equation_solution_l1217_121787

theorem equation_solution : ∃ x : ℝ, 2*x + 5 = 3*x - 2 ∧ x = 7 := by
  sorry

end equation_solution_l1217_121787


namespace sphere_surface_area_ratio_l1217_121718

theorem sphere_surface_area_ratio (V₁ V₂ A₁ A₂ : ℝ) (h : V₁ / V₂ = 8 / 27) :
  A₁ / A₂ = 4 / 9 :=
by sorry

end sphere_surface_area_ratio_l1217_121718


namespace freshman_class_size_l1217_121791

theorem freshman_class_size :
  ∃! n : ℕ, n < 700 ∧
    n % 20 = 19 ∧
    n % 25 = 24 ∧
    n % 9 = 3 ∧
    n = 399 := by sorry

end freshman_class_size_l1217_121791
