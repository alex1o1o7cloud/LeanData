import Mathlib

namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l3610_361067

/-- The number of ways to put n distinguishable balls into k distinguishable boxes -/
def ways_to_distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 243 ways to put 5 distinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes : ways_to_distribute 5 3 = 243 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l3610_361067


namespace NUMINAMATH_CALUDE_divisible_sequence_eventually_periodic_l3610_361027

/-- A sequence of positive integers satisfying the given divisibility property -/
def DivisibleSequence (a : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, (a (n + 2*m)) ∣ (a n + a (n + m))

/-- The property of eventual periodicity for a sequence -/
def EventuallyPeriodic (a : ℕ → ℕ) : Prop :=
  ∃ N d : ℕ, d > 0 ∧ ∀ n : ℕ, n > N → a n = a (n + d)

/-- The main theorem: A divisible sequence is eventually periodic -/
theorem divisible_sequence_eventually_periodic (a : ℕ → ℕ) 
  (h : DivisibleSequence a) : EventuallyPeriodic a := by
  sorry

end NUMINAMATH_CALUDE_divisible_sequence_eventually_periodic_l3610_361027


namespace NUMINAMATH_CALUDE_prob_one_female_is_half_l3610_361047

/-- Represents the composition of the extracurricular interest group -/
structure InterestGroup :=
  (male_count : Nat)
  (female_count : Nat)

/-- Calculates the probability of selecting exactly one female student
    from two selections in the interest group -/
def prob_one_female (group : InterestGroup) : Real :=
  let total := group.male_count + group.female_count
  let prob_first_female := group.female_count / total
  let prob_second_male := group.male_count / (total - 1)
  let prob_first_male := group.male_count / total
  let prob_second_female := group.female_count / (total - 1)
  prob_first_female * prob_second_male + prob_first_male * prob_second_female

/-- Theorem: The probability of selecting exactly one female student
    from two selections in a group of 3 males and 1 female is 0.5 -/
theorem prob_one_female_is_half :
  let group := InterestGroup.mk 3 1
  prob_one_female group = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_female_is_half_l3610_361047


namespace NUMINAMATH_CALUDE_probability_three_specified_coins_heads_l3610_361043

/-- The probability of exactly three specified coins out of five coming up heads -/
theorem probability_three_specified_coins_heads (n : ℕ) (k : ℕ) : 
  n = 5 → k = 3 → (2^(n-k) : ℚ) / 2^n = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_specified_coins_heads_l3610_361043


namespace NUMINAMATH_CALUDE_reflect_F_coordinates_l3610_361099

/-- Reflects a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflects a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The original point F -/
def F : ℝ × ℝ := (3, 3)

theorem reflect_F_coordinates :
  (reflect_x (reflect_y F)) = (-3, -3) := by sorry

end NUMINAMATH_CALUDE_reflect_F_coordinates_l3610_361099


namespace NUMINAMATH_CALUDE_randolph_sydney_age_difference_l3610_361033

/-- The age difference between Randolph and Sydney -/
def ageDifference (randolphAge sydneyAge : ℕ) : ℕ := randolphAge - sydneyAge

/-- Theorem stating the age difference between Randolph and Sydney -/
theorem randolph_sydney_age_difference :
  ∀ (sherryAge : ℕ),
    sherryAge = 25 →
    ∀ (sydneyAge : ℕ),
      sydneyAge = 2 * sherryAge →
      ∀ (randolphAge : ℕ),
        randolphAge = 55 →
        ageDifference randolphAge sydneyAge = 5 := by
  sorry

end NUMINAMATH_CALUDE_randolph_sydney_age_difference_l3610_361033


namespace NUMINAMATH_CALUDE_two_liters_to_milliliters_nine_thousand_milliliters_to_liters_eight_liters_to_milliliters_l3610_361094

-- Define the conversion factor
def liter_to_milliliter : ℚ := 1000

-- Theorem for the first conversion
theorem two_liters_to_milliliters :
  2 * liter_to_milliliter = 2000 := by sorry

-- Theorem for the second conversion
theorem nine_thousand_milliliters_to_liters :
  9000 / liter_to_milliliter = 9 := by sorry

-- Theorem for the third conversion
theorem eight_liters_to_milliliters :
  8 * liter_to_milliliter = 8000 := by sorry

end NUMINAMATH_CALUDE_two_liters_to_milliliters_nine_thousand_milliliters_to_liters_eight_liters_to_milliliters_l3610_361094


namespace NUMINAMATH_CALUDE_alina_twist_result_l3610_361038

/-- Alina's twisting method for periodic decimal fractions -/
def twist (n : ℚ) : ℚ :=
  sorry

/-- The period length of the decimal representation of 503/2022 -/
def period_length : ℕ := 336

theorem alina_twist_result :
  twist (503 / 2022) = 9248267898383371824480369515011881956675900099900099900099 / (10^period_length - 1) :=
sorry

end NUMINAMATH_CALUDE_alina_twist_result_l3610_361038


namespace NUMINAMATH_CALUDE_parsley_sprig_count_l3610_361006

/-- The number of parsley sprigs Carmen started with -/
def initial_sprigs : ℕ := 25

/-- The number of plates decorated with whole sprigs -/
def whole_sprig_plates : ℕ := 8

/-- The number of plates decorated with half sprigs -/
def half_sprig_plates : ℕ := 12

/-- The number of sprigs left after decorating -/
def remaining_sprigs : ℕ := 11

theorem parsley_sprig_count : 
  initial_sprigs = whole_sprig_plates + (half_sprig_plates / 2) + remaining_sprigs :=
by sorry

end NUMINAMATH_CALUDE_parsley_sprig_count_l3610_361006


namespace NUMINAMATH_CALUDE_fraction_operations_l3610_361024

/-- Define the † operation for fractions -/
def dagger (a b c d : ℚ) : ℚ := a * c * (d / b)

/-- Define the * operation for fractions -/
def star (a b c d : ℚ) : ℚ := a * c * (b / d)

/-- Theorem stating that (5/6)†(7/9)*(2/3) = 140 -/
theorem fraction_operations : 
  star (dagger (5/6) (7/9)) (2/3) = 140 := by sorry

end NUMINAMATH_CALUDE_fraction_operations_l3610_361024


namespace NUMINAMATH_CALUDE_george_sock_order_l3610_361005

/-- The ratio of black to blue socks in George's original order -/
def sock_ratio : ℚ := 2 / 11

theorem george_sock_order :
  ∀ (black_price blue_price : ℝ) (blue_count : ℝ),
    black_price = 2 * blue_price →
    3 * black_price + blue_count * blue_price = 
      (blue_count * black_price + 3 * blue_price) * (1 - 0.6) →
    sock_ratio = 3 / blue_count :=
by
  sorry

end NUMINAMATH_CALUDE_george_sock_order_l3610_361005


namespace NUMINAMATH_CALUDE_base3_to_base10_conversion_l3610_361063

/-- Converts a list of digits in base 3 to a natural number in base 10 -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 3^i) 0

/-- The base 3 representation of the number -/
def base3Digits : List Nat := [1, 2, 0, 2, 1]

theorem base3_to_base10_conversion :
  base3ToBase10 base3Digits = 142 := by
  sorry

end NUMINAMATH_CALUDE_base3_to_base10_conversion_l3610_361063


namespace NUMINAMATH_CALUDE_pythons_for_fifteen_alligators_l3610_361039

/-- The number of Burmese pythons required to eat a given number of alligators in a specified time period. -/
def pythons_required (alligators : ℕ) (weeks : ℕ) : ℕ :=
  (alligators + weeks - 1) / weeks

/-- The theorem stating that 5 Burmese pythons are required to eat 15 alligators in 3 weeks. -/
theorem pythons_for_fifteen_alligators : pythons_required 15 3 = 5 := by
  sorry

#eval pythons_required 15 3

end NUMINAMATH_CALUDE_pythons_for_fifteen_alligators_l3610_361039


namespace NUMINAMATH_CALUDE_parabola_line_intersection_slope_product_l3610_361032

/-- Given a parabola y^2 = 2px (p > 0) and a line y = x - p intersecting the parabola at points A and B,
    the product of the slopes of lines OA and OB is -2, where O is the coordinate origin. -/
theorem parabola_line_intersection_slope_product (p : ℝ) (h : p > 0) : 
  ∃ (A B : ℝ × ℝ),
    (A.2^2 = 2*p*A.1) ∧ 
    (B.2^2 = 2*p*B.1) ∧
    (A.2 = A.1 - p) ∧ 
    (B.2 = B.1 - p) ∧
    ((A.2 / A.1) * (B.2 / B.1) = -2) :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_slope_product_l3610_361032


namespace NUMINAMATH_CALUDE_real_part_of_complex_product_l3610_361000

theorem real_part_of_complex_product : ∃ (z : ℂ), z = (1 - Complex.I) * (2 + Complex.I) ∧ z.re = 3 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_product_l3610_361000


namespace NUMINAMATH_CALUDE_max_y_value_l3610_361050

theorem max_y_value (x y : ℤ) (h : x * y + 6 * x + 5 * y = -6) : 
  y ≤ 24 ∧ ∃ (x₀ : ℤ), x₀ * 24 + 6 * x₀ + 5 * 24 = -6 :=
sorry

end NUMINAMATH_CALUDE_max_y_value_l3610_361050


namespace NUMINAMATH_CALUDE_decimal_expansion_2023rd_digit_l3610_361085

/-- The decimal expansion of 7/26 -/
def decimal_expansion : ℚ := 7 / 26

/-- The length of the repeating block in the decimal expansion of 7/26 -/
def repeating_block_length : ℕ := 9

/-- The position of the 2023rd digit within the repeating block -/
def position_in_block : ℕ := 2023 % repeating_block_length

/-- The 2023rd digit past the decimal point in the decimal expansion of 7/26 -/
def digit_2023 : ℕ := 3

theorem decimal_expansion_2023rd_digit :
  digit_2023 = 3 :=
sorry

end NUMINAMATH_CALUDE_decimal_expansion_2023rd_digit_l3610_361085


namespace NUMINAMATH_CALUDE_sum_of_cubes_l3610_361055

theorem sum_of_cubes (a b c : ℝ) 
  (sum_eq : a + b + c = 8)
  (sum_products_eq : a * b + a * c + b * c = 10)
  (product_eq : a * b * c = -15) :
  a^3 + b^3 + c^3 = 227 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l3610_361055


namespace NUMINAMATH_CALUDE_smallest_number_with_hcf_twelve_l3610_361083

/-- The highest common factor of two natural numbers -/
def hcf (a b : ℕ) : ℕ := Nat.gcd a b

/-- Theorem: 48 is the smallest number greater than 36 that has a highest common factor of 12 with 36 -/
theorem smallest_number_with_hcf_twelve : 
  ∀ n : ℕ, n > 36 → hcf 36 n = 12 → n ≥ 48 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_with_hcf_twelve_l3610_361083


namespace NUMINAMATH_CALUDE_f_properties_l3610_361022

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

-- Define the derivative of f(x)
noncomputable def f' (x : ℝ) : ℝ := -3*x^2 + 6*x + 9

theorem f_properties (a : ℝ) :
  -- Part 1: Monotonically decreasing intervals
  (∀ x < -1, (f' x) < 0) ∧
  (∀ x > 3, (f' x) < 0) ∧
  -- Part 2: Maximum and minimum values
  (∃ x ∈ Set.Icc (-2) 2, f a x = 20) →
  (∃ y ∈ Set.Icc (-2) 2, f a y = -7 ∧ ∀ z ∈ Set.Icc (-2) 2, f a z ≥ f a y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3610_361022


namespace NUMINAMATH_CALUDE_plane_division_l3610_361076

/-- The maximum number of parts that n planes can divide 3D space into --/
def max_parts (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => 2^(n+1)

theorem plane_division :
  (max_parts 1 = 2) ∧
  (max_parts 2 ≤ 4) ∧
  (max_parts 3 ≤ 8) := by
  sorry

end NUMINAMATH_CALUDE_plane_division_l3610_361076


namespace NUMINAMATH_CALUDE_elevation_view_area_bounds_not_possible_area_l3610_361098

/-- The area of the elevation view of a unit cube is between 1 and √2 (inclusive) -/
theorem elevation_view_area_bounds (area : ℝ) : 
  (∃ (angle : ℝ), area = Real.cos angle + Real.sin angle) →
  1 ≤ area ∧ area ≤ Real.sqrt 2 := by
  sorry

/-- (√2 - 1) / 2 is not a possible area for the elevation view of a unit cube -/
theorem not_possible_area : 
  ¬ (∃ (angle : ℝ), (Real.sqrt 2 - 1) / 2 = Real.cos angle + Real.sin angle) := by
  sorry

end NUMINAMATH_CALUDE_elevation_view_area_bounds_not_possible_area_l3610_361098


namespace NUMINAMATH_CALUDE_total_letters_received_l3610_361017

theorem total_letters_received (brother_letters : ℕ) 
  (h1 : brother_letters = 40) 
  (h2 : ∃ greta_letters : ℕ, greta_letters = brother_letters + 10) 
  (h3 : ∃ mother_letters : ℕ, mother_letters = 2 * (brother_letters + (brother_letters + 10))) :
  ∃ total_letters : ℕ, total_letters = brother_letters + (brother_letters + 10) + 2 * (brother_letters + (brother_letters + 10)) ∧ total_letters = 270 := by
sorry


end NUMINAMATH_CALUDE_total_letters_received_l3610_361017


namespace NUMINAMATH_CALUDE_kira_breakfast_time_l3610_361081

/-- Calculates the total time Kira spent making breakfast -/
def breakfast_time (num_sausages : ℕ) (num_eggs : ℕ) (time_per_sausage : ℕ) (time_per_egg : ℕ) : ℕ :=
  num_sausages * time_per_sausage + num_eggs * time_per_egg

/-- Proves that Kira's breakfast preparation time is 39 minutes -/
theorem kira_breakfast_time : 
  breakfast_time 3 6 5 4 = 39 := by
  sorry

end NUMINAMATH_CALUDE_kira_breakfast_time_l3610_361081


namespace NUMINAMATH_CALUDE_marbles_selection_theorem_l3610_361097

def total_marbles : ℕ := 9
def marbles_to_choose : ℕ := 4
def blue_marbles : ℕ := 2

theorem marbles_selection_theorem :
  (Nat.choose total_marbles marbles_to_choose) -
  (Nat.choose (total_marbles - blue_marbles) marbles_to_choose) = 91 := by
  sorry

end NUMINAMATH_CALUDE_marbles_selection_theorem_l3610_361097


namespace NUMINAMATH_CALUDE_reduced_rate_weekend_l3610_361096

/-- Represents a day of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents the electric company's rate plan -/
structure RatePlan where
  reducedRateFraction : ℝ
  weekdayReducedHours : ℕ
  fullDayReducedRate : List Day

/-- Assertion that the given rate plan is valid and consistent with the problem statement -/
def isValidPlan (plan : RatePlan) : Prop :=
  plan.reducedRateFraction = 0.6428571428571429 ∧
  plan.weekdayReducedHours = 12 ∧
  plan.fullDayReducedRate.length = 2

/-- Theorem stating that for a valid plan, the full day reduced rate must apply on Saturday and Sunday -/
theorem reduced_rate_weekend (plan : RatePlan) (h : isValidPlan plan) :
  plan.fullDayReducedRate = [Day.Saturday, Day.Sunday] :=
sorry

end NUMINAMATH_CALUDE_reduced_rate_weekend_l3610_361096


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3610_361090

/-- Given a parabola and an ellipse with the following properties:
  1) The parabola has the equation x^2 = 2py where p > 0
  2) The ellipse has the equation x^2/3 + y^2/4 = 1
  3) The focus of the parabola coincides with one of the vertices of the ellipse
This theorem states that the distance from the focus of the parabola to its directrix is 4. -/
theorem parabola_focus_directrix_distance (p : ℝ) 
  (h_p_pos : p > 0)
  (h_focus_coincides : ∃ (x y : ℝ), x^2/3 + y^2/4 = 1 ∧ x^2 = 2*p*y ∧ (x = 0 ∨ y = 2 ∨ y = -2)) :
  p = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3610_361090


namespace NUMINAMATH_CALUDE_valid_permutations_count_l3610_361080

/-- Given integers 1 to n, where n ≥ 2, this function returns the number of permutations
    satisfying the condition that for all k = 1 to n, the kth element is ≥ k-2 -/
def countValidPermutations (n : ℕ) : ℕ :=
  2 * 3^(n-2)

/-- Theorem stating that for n ≥ 2, the number of permutations of integers 1 to n
    satisfying the condition that for all k = 1 to n, the kth element is ≥ k-2,
    is equal to 2 * 3^(n-2) -/
theorem valid_permutations_count (n : ℕ) (h : n ≥ 2) :
  (Finset.univ.filter (fun p : Fin n → Fin n =>
    ∀ k : Fin n, p k ≥ ⟨k - 2, by sorry⟩)).card = countValidPermutations n := by
  sorry

end NUMINAMATH_CALUDE_valid_permutations_count_l3610_361080


namespace NUMINAMATH_CALUDE_second_snake_length_l3610_361008

/-- Proves that the length of the second snake is 16 inches -/
theorem second_snake_length (total_snakes : Nat) (first_snake_feet : Nat) (third_snake_inches : Nat) (total_length_inches : Nat) (inches_per_foot : Nat) :
  total_snakes = 3 →
  first_snake_feet = 2 →
  third_snake_inches = 10 →
  total_length_inches = 50 →
  inches_per_foot = 12 →
  total_length_inches - (first_snake_feet * inches_per_foot + third_snake_inches) = 16 := by
  sorry

end NUMINAMATH_CALUDE_second_snake_length_l3610_361008


namespace NUMINAMATH_CALUDE_water_usage_per_person_l3610_361045

/-- Given a family's water usage, prove the amount of water needed per person per day. -/
theorem water_usage_per_person
  (cost_per_gallon : ℝ)
  (family_size : ℕ)
  (daily_cost : ℝ)
  (h1 : cost_per_gallon = 1)
  (h2 : family_size = 6)
  (h3 : daily_cost = 3) :
  daily_cost / (cost_per_gallon * family_size) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_water_usage_per_person_l3610_361045


namespace NUMINAMATH_CALUDE_simple_interest_time_calculation_l3610_361042

/-- Simple interest calculation -/
theorem simple_interest_time_calculation
  (principal : ℝ)
  (simple_interest : ℝ)
  (rate : ℝ)
  (h1 : principal = 400)
  (h2 : simple_interest = 140)
  (h3 : rate = 17.5) :
  (simple_interest * 100) / (principal * rate) = 2 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_time_calculation_l3610_361042


namespace NUMINAMATH_CALUDE_power_two_plus_two_gt_square_l3610_361002

theorem power_two_plus_two_gt_square (n : ℕ) (h : n > 0) : 2^n + 2 > n^2 := by
  sorry

end NUMINAMATH_CALUDE_power_two_plus_two_gt_square_l3610_361002


namespace NUMINAMATH_CALUDE_piece_length_in_cm_l3610_361057

-- Define the length of the rod in meters
def rod_length : ℝ := 25.5

-- Define the number of pieces that can be cut from the rod
def num_pieces : ℕ := 30

-- Define the conversion factor from meters to centimeters
def meters_to_cm : ℝ := 100

-- Theorem statement
theorem piece_length_in_cm : 
  (rod_length / num_pieces) * meters_to_cm = 85 := by
  sorry

end NUMINAMATH_CALUDE_piece_length_in_cm_l3610_361057


namespace NUMINAMATH_CALUDE_square_starts_with_sequence_l3610_361068

theorem square_starts_with_sequence (S : ℕ) : 
  ∃ (N k : ℕ), S * 10^k ≤ N^2 ∧ N^2 < (S + 1) * 10^k :=
by sorry

end NUMINAMATH_CALUDE_square_starts_with_sequence_l3610_361068


namespace NUMINAMATH_CALUDE_largest_class_size_l3610_361048

/-- Proves that in a school with 5 classes, where each class has 2 students less than the previous class,
    and the total number of students is 120, the largest class has 28 students. -/
theorem largest_class_size (n : ℕ) (h1 : n = 5) (total : ℕ) (h2 : total = 120) :
  ∃ x : ℕ, x = 28 ∧ 
    x + (x - 2) + (x - 4) + (x - 6) + (x - 8) = total :=
by sorry

end NUMINAMATH_CALUDE_largest_class_size_l3610_361048


namespace NUMINAMATH_CALUDE_geometric_propositions_l3610_361052

-- Define the four propositions
def vertical_angles_equal : Prop := sorry
def alternate_interior_angles_equal : Prop := sorry
def parallel_transitivity : Prop := sorry
def parallel_sides_equal_angles : Prop := sorry

-- Theorem stating which propositions are true
theorem geometric_propositions :
  vertical_angles_equal ∧ 
  parallel_transitivity ∧ 
  ¬alternate_interior_angles_equal ∧ 
  ¬parallel_sides_equal_angles := by
  sorry

end NUMINAMATH_CALUDE_geometric_propositions_l3610_361052


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l3610_361040

theorem hyperbola_focal_length :
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2 / 10 - y^2 / 2 = 1
  ∃ (f : ℝ), f = 4 * Real.sqrt 3 ∧ 
    ∀ (x y : ℝ), h x y → 
      f = 2 * Real.sqrt ((Real.sqrt 10)^2 + (Real.sqrt 2)^2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l3610_361040


namespace NUMINAMATH_CALUDE_greatest_x_value_l3610_361025

theorem greatest_x_value (x : ℤ) (h : (6.1 : ℝ) * (10 : ℝ) ^ (x : ℝ) < 620) :
  x ≤ 2 ∧ ∃ y : ℤ, y > 2 → (6.1 : ℝ) * (10 : ℝ) ^ (y : ℝ) ≥ 620 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l3610_361025


namespace NUMINAMATH_CALUDE_evaluate_expression_l3610_361079

theorem evaluate_expression : (((3^2 : ℚ) - 2^3 + 7^1 - 1 + 4^2)⁻¹ * (5/6)) = 5/138 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3610_361079


namespace NUMINAMATH_CALUDE_sunglasses_and_hat_probability_l3610_361037

/-- The probability that a randomly selected person wearing sunglasses is also wearing a hat -/
theorem sunglasses_and_hat_probability
  (total_sunglasses : ℕ)
  (total_hats : ℕ)
  (prob_sunglasses_given_hat : ℚ)
  (h1 : total_sunglasses = 60)
  (h2 : total_hats = 45)
  (h3 : prob_sunglasses_given_hat = 3 / 5) :
  (total_hats : ℚ) * prob_sunglasses_given_hat / total_sunglasses = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_sunglasses_and_hat_probability_l3610_361037


namespace NUMINAMATH_CALUDE_power_approximations_l3610_361070

theorem power_approximations : 
  (|((1.02 : ℝ)^30 - 1.8114)| < 0.00005) ∧ 
  (|((0.996 : ℝ)^13 - 0.9492)| < 0.00005) := by
  sorry

end NUMINAMATH_CALUDE_power_approximations_l3610_361070


namespace NUMINAMATH_CALUDE_parabola_point_value_l3610_361078

/-- 
Given a parabola y = -x^2 + bx + c that passes through the point (-2, 3),
prove that 2c - 4b - 9 = 5
-/
theorem parabola_point_value (b c : ℝ) 
  (h : 3 = -(-2)^2 + b*(-2) + c) : 2*c - 4*b - 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_value_l3610_361078


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l3610_361053

theorem sum_of_roots_quadratic (a b c d : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  let roots := {x : ℝ | f x = d}
  (∃ x y, x ∈ roots ∧ y ∈ roots ∧ x ≠ y) →
  (∀ z, z ∈ roots → z = x ∨ z = y) →
  x + y = -b / a :=
by sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 6 * x + 8
  let roots := {x : ℝ | f x = 15}
  (∃ x y, x ∈ roots ∧ y ∈ roots ∧ x ≠ y) →
  (∀ z, z ∈ roots → z = x ∨ z = y) →
  x + y = 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l3610_361053


namespace NUMINAMATH_CALUDE_cube_size_is_eight_l3610_361086

/-- Represents a cube of size n --/
structure Cube (n : ℕ) where
  size : n > 0

/-- Number of small cubes with no faces painted in a cube of size n --/
def unpainted (c : Cube n) : ℕ := (n - 2)^3

/-- Number of small cubes with exactly two faces painted in a cube of size n --/
def two_faces_painted (c : Cube n) : ℕ := 12 * (n - 2)

/-- Theorem stating that for a cube where the number of unpainted small cubes
    is three times the number of small cubes with two faces painted,
    the size of the cube must be 8 --/
theorem cube_size_is_eight (c : Cube n)
  (h : unpainted c = 3 * two_faces_painted c) : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_cube_size_is_eight_l3610_361086


namespace NUMINAMATH_CALUDE_price_adjustment_l3610_361056

-- Define the original price
variable (P : ℝ)
-- Define the percentage x
variable (x : ℝ)

-- Theorem statement
theorem price_adjustment (h : P * (1 + x/100) * (1 - x/100) = 0.75 * P) : x = 50 := by
  sorry

end NUMINAMATH_CALUDE_price_adjustment_l3610_361056


namespace NUMINAMATH_CALUDE_exists_complete_gear_rotation_l3610_361084

/-- Represents a gear with a certain number of teeth and some removed teeth -/
structure Gear where
  total_teeth : Nat
  removed_teeth : Finset Nat

/-- Represents the system of two gears -/
structure GearSystem where
  gear1 : Gear
  gear2 : Gear
  rotation : Nat

/-- Checks if a given rotation results in a complete gear -/
def is_complete_gear (gs : GearSystem) : Prop :=
  ∀ i : Nat, i < gs.gear1.total_teeth →
    (i ∉ gs.gear1.removed_teeth ∨ ((i + gs.rotation) % gs.gear1.total_teeth) ∉ gs.gear2.removed_teeth)

/-- The main theorem stating that there exists a rotation forming a complete gear -/
theorem exists_complete_gear_rotation (g1 g2 : Gear)
    (h1 : g1.total_teeth = 14)
    (h2 : g2.total_teeth = 14)
    (h3 : g1.removed_teeth.card = 4)
    (h4 : g2.removed_teeth.card = 4) :
    ∃ r : Nat, is_complete_gear ⟨g1, g2, r⟩ := by
  sorry


end NUMINAMATH_CALUDE_exists_complete_gear_rotation_l3610_361084


namespace NUMINAMATH_CALUDE_largest_b_for_divisibility_by_three_l3610_361011

def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem largest_b_for_divisibility_by_three :
  ∀ b : ℕ, b ≤ 9 →
    (is_divisible_by_three (500000 + 100000 * b + 6584) ↔ is_divisible_by_three (b + 28)) ∧
    (∀ k : ℕ, k ≤ 9 ∧ k > b → ¬is_divisible_by_three (500000 + 100000 * k + 6584)) →
    b = 8 :=
by sorry

end NUMINAMATH_CALUDE_largest_b_for_divisibility_by_three_l3610_361011


namespace NUMINAMATH_CALUDE_initial_alloy_weight_l3610_361095

/-- Given an initial alloy of weight x ounces that is 50% gold,
    if adding 24 ounces of pure gold results in a new alloy that is 80% gold,
    then the initial alloy weighs 16 ounces. -/
theorem initial_alloy_weight (x : ℝ) : 
  (0.5 * x + 24) / (x + 24) = 0.8 → x = 16 := by sorry

end NUMINAMATH_CALUDE_initial_alloy_weight_l3610_361095


namespace NUMINAMATH_CALUDE_y_order_l3610_361030

/-- The quadratic function f(x) = -2x² + 4 --/
def f (x : ℝ) : ℝ := -2 * x^2 + 4

/-- Point A on the graph of f --/
def A : ℝ × ℝ := (1, f 1)

/-- Point B on the graph of f --/
def B : ℝ × ℝ := (2, f 2)

/-- Point C on the graph of f --/
def C : ℝ × ℝ := (-3, f (-3))

theorem y_order : A.2 > B.2 ∧ B.2 > C.2 := by sorry

end NUMINAMATH_CALUDE_y_order_l3610_361030


namespace NUMINAMATH_CALUDE_alpha_value_l3610_361059

-- Define complex numbers α and β
variable (α β : ℂ)

-- Define the conditions
variable (h1 : (α + β).re > 0)
variable (h2 : (Complex.I * (α - 3 * β)).re > 0)
variable (h3 : β = 4 + 3 * Complex.I)

-- Theorem to prove
theorem alpha_value : α = 12 - 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_l3610_361059


namespace NUMINAMATH_CALUDE_jamies_coins_value_l3610_361019

/-- Proves that given 30 coins of nickels and dimes, if swapping their values
    results in a 90-cent increase, then the total value is $1.80. -/
theorem jamies_coins_value :
  ∀ (n d : ℕ),
  n + d = 30 →
  (10 * n + 5 * d) - (5 * n + 10 * d) = 90 →
  5 * n + 10 * d = 180 := by
sorry

end NUMINAMATH_CALUDE_jamies_coins_value_l3610_361019


namespace NUMINAMATH_CALUDE_geometric_arithmetic_progression_l3610_361015

theorem geometric_arithmetic_progression (a b c : ℤ) : 
  (∃ (q : ℚ), b = a * q ∧ c = b * q) →  -- Geometric progression condition
  (2 * (b + 8) = a + c) →               -- Arithmetic progression condition
  ((b + 8)^2 = a * (c + 64)) →          -- Second geometric progression condition
  (a = 4 ∧ b = 12 ∧ c = 36) :=           -- Conclusion
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_progression_l3610_361015


namespace NUMINAMATH_CALUDE_product_of_numbers_l3610_361026

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 9) (h2 : x^2 + y^2 = 153) : x * y = 36 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3610_361026


namespace NUMINAMATH_CALUDE_expansion_sum_l3610_361031

-- Define the sum of coefficients of the expansion
def P (n : ℕ) : ℕ := 4^n

-- Define the sum of all binomial coefficients
def S (n : ℕ) : ℕ := 2^n

-- Theorem statement
theorem expansion_sum (n : ℕ) : P n + S n = 272 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_expansion_sum_l3610_361031


namespace NUMINAMATH_CALUDE_average_difference_l3610_361058

theorem average_difference (a b c : ℝ) : 
  (a + b) / 2 = 45 → (b + c) / 2 = 50 → c - a = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l3610_361058


namespace NUMINAMATH_CALUDE_tim_running_hours_l3610_361087

/-- Represents Tim's running schedule --/
structure RunningSchedule where
  initial_days : ℕ  -- Initial number of days Tim ran per week
  added_days : ℕ    -- Number of days Tim added to his schedule
  morning_run : ℕ   -- Hours Tim runs in the morning
  evening_run : ℕ   -- Hours Tim runs in the evening

/-- Calculates the total hours Tim runs per week --/
def total_running_hours (schedule : RunningSchedule) : ℕ :=
  (schedule.initial_days + schedule.added_days) * (schedule.morning_run + schedule.evening_run)

/-- Theorem stating that Tim's total running hours per week is 10 --/
theorem tim_running_hours :
  ∃ (schedule : RunningSchedule),
    schedule.initial_days = 3 ∧
    schedule.added_days = 2 ∧
    schedule.morning_run = 1 ∧
    schedule.evening_run = 1 ∧
    total_running_hours schedule = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_tim_running_hours_l3610_361087


namespace NUMINAMATH_CALUDE_derivative_sqrt_l3610_361072

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

theorem derivative_sqrt (x : ℝ) (hx : x > 0) :
  deriv f x = 1 / (2 * Real.sqrt x) := by sorry

end NUMINAMATH_CALUDE_derivative_sqrt_l3610_361072


namespace NUMINAMATH_CALUDE_smallest_binary_multiple_of_15_l3610_361034

def is_binary_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 1

theorem smallest_binary_multiple_of_15 :
  ∀ T : ℕ, 
    T > 0 → 
    is_binary_number T → 
    T % 15 = 0 → 
    ∀ X : ℕ, 
      X = T / 15 → 
      X ≥ 74 :=
sorry

end NUMINAMATH_CALUDE_smallest_binary_multiple_of_15_l3610_361034


namespace NUMINAMATH_CALUDE_sichuan_selected_count_l3610_361075

/-- Represents the number of students selected from Sichuan University in a stratified sampling -/
def sichuan_selected (total_students : ℕ) (sichuan_students : ℕ) (other_students : ℕ) (selected_students : ℕ) : ℕ :=
  (selected_students * sichuan_students) / (sichuan_students + other_students)

/-- Theorem stating that 10 students from Sichuan University are selected in the given scenario -/
theorem sichuan_selected_count :
  sichuan_selected 40 25 15 16 = 10 := by
  sorry

#eval sichuan_selected 40 25 15 16

end NUMINAMATH_CALUDE_sichuan_selected_count_l3610_361075


namespace NUMINAMATH_CALUDE_female_attendees_on_time_l3610_361021

theorem female_attendees_on_time (total_attendees : ℝ) :
  let male_fraction : ℝ := 3/5
  let male_on_time_fraction : ℝ := 7/8
  let not_on_time_fraction : ℝ := 0.155
  let female_fraction : ℝ := 1 - male_fraction
  let on_time_fraction : ℝ := 1 - not_on_time_fraction
  let male_on_time : ℝ := male_fraction * male_on_time_fraction * total_attendees
  let total_on_time : ℝ := on_time_fraction * total_attendees
  let female_on_time : ℝ := total_on_time - male_on_time
  let female_attendees : ℝ := female_fraction * total_attendees
  female_on_time / female_attendees = 4/5 := by sorry

end NUMINAMATH_CALUDE_female_attendees_on_time_l3610_361021


namespace NUMINAMATH_CALUDE_line_symmetry_l3610_361001

-- Define the original line
def original_line (x y : ℝ) : Prop := 2 * x - 3 * y + 1 = 0

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := y = x

-- Define the symmetric line
def symmetric_line (x y : ℝ) : Prop := 3 * x - 2 * y - 1 = 0

-- Theorem stating the symmetry relationship
theorem line_symmetry :
  ∀ (x y : ℝ), original_line x y ↔ symmetric_line y x :=
by sorry

end NUMINAMATH_CALUDE_line_symmetry_l3610_361001


namespace NUMINAMATH_CALUDE_largest_prime_factors_difference_l3610_361060

theorem largest_prime_factors_difference (n : Nat) (h : n = 178469) : 
  ∃ (p q : Nat), Nat.Prime p ∧ Nat.Prime q ∧ p > q ∧ 
  p ∣ n ∧ q ∣ n ∧
  (∀ (r : Nat), Nat.Prime r → r ∣ n → r ≤ p) ∧
  (∀ (r : Nat), Nat.Prime r → r ∣ n → r ≠ p → r ≤ q) ∧
  p - q = 2 := by
sorry

end NUMINAMATH_CALUDE_largest_prime_factors_difference_l3610_361060


namespace NUMINAMATH_CALUDE_contrapositive_proof_l3610_361044

theorem contrapositive_proof (a b : ℝ) :
  (∀ a b, a > b → a - 1 > b - 1) ↔ (∀ a b, a - 1 ≤ b - 1 → a ≤ b) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_proof_l3610_361044


namespace NUMINAMATH_CALUDE_hulk_jump_theorem_l3610_361036

def jump_distance (n : ℕ) : ℝ :=
  2 * (3 ^ (n - 1))

theorem hulk_jump_theorem :
  (∀ k < 8, jump_distance k ≤ 2000) ∧ jump_distance 8 > 2000 := by
  sorry

end NUMINAMATH_CALUDE_hulk_jump_theorem_l3610_361036


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l3610_361091

theorem modulus_of_complex_number (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := i * (2 - i)
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l3610_361091


namespace NUMINAMATH_CALUDE_monic_cubic_polynomial_value_l3610_361046

/-- A monic cubic polynomial is a polynomial of the form x^3 + ax^2 + bx + c -/
def MonicCubicPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ x^3 + a*x^2 + b*x + c

theorem monic_cubic_polynomial_value (a b c : ℝ) :
  let p := MonicCubicPolynomial a b c
  (p 2 = 3) → (p 4 = 9) → (p 6 = 19) → (p 8 = -9) := by
  sorry

end NUMINAMATH_CALUDE_monic_cubic_polynomial_value_l3610_361046


namespace NUMINAMATH_CALUDE_evaluate_expression_l3610_361013

theorem evaluate_expression : (24 ^ 40) / (72 ^ 20) = 2 ^ 60 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3610_361013


namespace NUMINAMATH_CALUDE_multiples_of_12_between_15_and_205_l3610_361062

theorem multiples_of_12_between_15_and_205 : 
  (Finset.filter (fun n => 12 ∣ n) (Finset.Ioo 15 205)).card = 16 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_12_between_15_and_205_l3610_361062


namespace NUMINAMATH_CALUDE_tangent_line_at_one_zero_l3610_361049

-- Define the curve
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 2*x - 2

-- Theorem statement
theorem tangent_line_at_one_zero :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y = m * (x - x₀) + y₀ → y = 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_zero_l3610_361049


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3610_361020

theorem complex_number_quadrant (z : ℂ) (h : z = 1 + Complex.I) :
  2 / z + z^2 = 1 + Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3610_361020


namespace NUMINAMATH_CALUDE_line_perp_to_plane_l3610_361066

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation for lines and planes
variable (perp : Line → Line → Prop)
variable (perp_plane : Plane → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- Define the parallel relation for lines
variable (parallel : Line → Line → Prop)

-- Define the intersection operation for planes
variable (intersect : Plane → Plane → Line)

-- Theorem statement
theorem line_perp_to_plane 
  (m n : Line) 
  (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) 
  (h1 : perp_plane α β) 
  (h2 : intersect α β = m) 
  (h3 : perp m n) : 
  perp_line_plane n β :=
sorry

end NUMINAMATH_CALUDE_line_perp_to_plane_l3610_361066


namespace NUMINAMATH_CALUDE_warehouse_temp_restoration_time_l3610_361003

def initial_temp : ℝ := 43
def increase_rate : ℝ := 8
def outage_duration : ℝ := 3
def decrease_rate : ℝ := 4

theorem warehouse_temp_restoration_time :
  let total_increase : ℝ := increase_rate * outage_duration
  let restoration_time : ℝ := total_increase / decrease_rate
  restoration_time = 6 := by sorry

end NUMINAMATH_CALUDE_warehouse_temp_restoration_time_l3610_361003


namespace NUMINAMATH_CALUDE_ellipse_properties_l3610_361092

/-- An ellipse with the given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0
  h3 : (3/2)^2 / a^2 + 6 / b^2 = 1  -- Point M (3/2, √6) lies on the ellipse
  h4 : 2 * (a^2 - b^2).sqrt = 2     -- Focal length is 2

/-- The standard equation of the ellipse -/
def standard_equation (e : Ellipse) : Prop :=
  e.a = 3 ∧ e.b^2 = 8

/-- The trajectory equation of point E -/
def trajectory_equation (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 8 = 1 ∧ x ≠ 3 ∧ x ≠ -3

theorem ellipse_properties (e : Ellipse) :
  standard_equation e ∧ ∀ x y, trajectory_equation x y :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3610_361092


namespace NUMINAMATH_CALUDE_monika_movies_l3610_361061

def mall_expense : ℝ := 250
def movie_cost : ℝ := 24
def bean_bags : ℕ := 20
def bean_cost : ℝ := 1.25
def total_spent : ℝ := 347

theorem monika_movies :
  (total_spent - (mall_expense + bean_bags * bean_cost)) / movie_cost = 3 := by
  sorry

end NUMINAMATH_CALUDE_monika_movies_l3610_361061


namespace NUMINAMATH_CALUDE_sum_of_specific_numbers_l3610_361012

theorem sum_of_specific_numbers : 1357 + 3571 + 5713 + 7135 = 17776 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_numbers_l3610_361012


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3610_361051

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A as the open interval (-∞, 2)
def A : Set ℝ := {x : ℝ | x < 2}

-- State the theorem
theorem complement_of_A_in_U : 
  U \ A = {x : ℝ | x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3610_361051


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l3610_361082

def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {a-3, 2*a-1, a^2+1}

theorem intersection_implies_a_value (a : ℝ) :
  A a ∩ B a = {-3} → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l3610_361082


namespace NUMINAMATH_CALUDE_average_of_abc_is_three_l3610_361088

theorem average_of_abc_is_three (A B C : ℝ) 
  (eq1 : 2003 * C - 4004 * A = 8008)
  (eq2 : 2003 * B + 6006 * A = 10010)
  (eq3 : B = 2 * A - 6) :
  (A + B + C) / 3 = 3 := by
sorry

end NUMINAMATH_CALUDE_average_of_abc_is_three_l3610_361088


namespace NUMINAMATH_CALUDE_distance_2_neg5_abs_calculations_abs_equation_solutions_min_value_expression_l3610_361069

-- Define the distance function on the number line
def distance (a b : ℝ) : ℝ := |a - b|

-- Theorem 1: Distance between 2 and -5
theorem distance_2_neg5 : distance 2 (-5) = 7 := by sorry

-- Theorem 2: Absolute value calculations
theorem abs_calculations : 
  (|-4 + 6| = 2) ∧ (|-2 - 4| = 6) := by sorry

-- Theorem 3: Solutions to |x+2| = 4
theorem abs_equation_solutions :
  ∀ x : ℝ, |x + 2| = 4 ↔ (x = -6 ∨ x = 2) := by sorry

-- Theorem 4: Minimum value of |x+1| + |x-3|
theorem min_value_expression :
  ∃ m : ℝ, (∀ x : ℝ, |x + 1| + |x - 3| ≥ m) ∧ 
  (∃ x : ℝ, |x + 1| + |x - 3| = m) ∧ 
  (m = 4) := by sorry

end NUMINAMATH_CALUDE_distance_2_neg5_abs_calculations_abs_equation_solutions_min_value_expression_l3610_361069


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3610_361007

theorem quadratic_minimum (a b c d : ℝ) (ha : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c ≥ d) ∧ 
  (∃ x, a * x^2 + b * x + c = d) →
  c = d + b^2 / (4 * a) := by
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3610_361007


namespace NUMINAMATH_CALUDE_cubic_factorization_l3610_361004

theorem cubic_factorization (x : ℝ) : 2*x^3 - 4*x^2 + 2*x = 2*x*(x-1)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3610_361004


namespace NUMINAMATH_CALUDE_surface_area_unchanged_l3610_361064

/-- Represents the dimensions of a cube -/
structure CubeDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a cube -/
def surfaceArea (c : CubeDimensions) : ℝ :=
  6 * c.length * c.width

/-- Represents the original cube -/
def originalCube : CubeDimensions :=
  { length := 4, width := 4, height := 4 }

/-- Represents the corner cube to be removed -/
def cornerCube : CubeDimensions :=
  { length := 2, width := 2, height := 2 }

/-- The number of corners in a cube -/
def numCorners : ℕ := 8

theorem surface_area_unchanged :
  surfaceArea originalCube = surfaceArea originalCube := by sorry

end NUMINAMATH_CALUDE_surface_area_unchanged_l3610_361064


namespace NUMINAMATH_CALUDE_sum_of_min_max_x_l3610_361009

theorem sum_of_min_max_x (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 8) :
  ∃ m M : ℝ, (∀ x y z : ℝ, x + y + z = 5 → x^2 + y^2 + z^2 = 8 → m ≤ x ∧ x ≤ M) ∧
            (∃ x y z : ℝ, x + y + z = 5 ∧ x^2 + y^2 + z^2 = 8 ∧ x = m) ∧
            (∃ x y z : ℝ, x + y + z = 5 ∧ x^2 + y^2 + z^2 = 8 ∧ x = M) ∧
            m + M = 4 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_min_max_x_l3610_361009


namespace NUMINAMATH_CALUDE_markup_calculation_l3610_361073

-- Define the markup percentage
def markup_percentage : ℝ := 50

-- Define the discount percentage
def discount_percentage : ℝ := 20

-- Define the profit percentage
def profit_percentage : ℝ := 20

-- Define the relationship between cost price, marked price, and selling price
def price_relationship (cost_price marked_price selling_price : ℝ) : Prop :=
  selling_price = marked_price * (1 - discount_percentage / 100) ∧
  selling_price = cost_price * (1 + profit_percentage / 100)

-- Theorem statement
theorem markup_calculation :
  ∀ (cost_price marked_price selling_price : ℝ),
  cost_price > 0 →
  price_relationship cost_price marked_price selling_price →
  (marked_price - cost_price) / cost_price * 100 = markup_percentage :=
by sorry

end NUMINAMATH_CALUDE_markup_calculation_l3610_361073


namespace NUMINAMATH_CALUDE_quadratic_points_relationship_l3610_361041

/-- A quadratic function f(x) = -2x^2 - 8x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := -2 * x^2 - 8 * x + m

theorem quadratic_points_relationship (m : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h₁ : f m (-1) = y₁)
  (h₂ : f m (-2) = y₂)
  (h₃ : f m (-4) = y₃) :
  y₃ < y₁ ∧ y₁ < y₂ := by
sorry

end NUMINAMATH_CALUDE_quadratic_points_relationship_l3610_361041


namespace NUMINAMATH_CALUDE_greatest_integer_radius_of_semicircle_l3610_361010

theorem greatest_integer_radius_of_semicircle (A : ℝ) (h : A < 45 * Real.pi) :
  ∃ (r : ℕ), r = 9 ∧ (∀ (n : ℕ), (↑n : ℝ)^2 * Real.pi / 2 ≤ A → n ≤ 9) :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_of_semicircle_l3610_361010


namespace NUMINAMATH_CALUDE_sum_of_numbers_l3610_361035

theorem sum_of_numbers (x y : ℤ) : y = 3 * x + 11 → x = 11 → x + y = 55 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l3610_361035


namespace NUMINAMATH_CALUDE_negation_of_proposition_sin_inequality_negation_l3610_361089

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬ p x) :=
by sorry

theorem sin_inequality_negation :
  (¬ ∀ x : ℝ, x > Real.sin x) ↔ (∃ x : ℝ, x ≤ Real.sin x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_sin_inequality_negation_l3610_361089


namespace NUMINAMATH_CALUDE_probability_calculation_l3610_361029

structure ClassStats where
  total_students : ℕ
  female_percentage : ℚ
  brunette_percentage : ℚ
  short_brunette_percentage : ℚ
  club_participation_percentage : ℚ
  short_club_percentage : ℚ

def probability_short_brunette_club (stats : ClassStats) : ℚ :=
  stats.female_percentage *
  stats.brunette_percentage *
  stats.club_participation_percentage *
  stats.short_club_percentage

theorem probability_calculation (stats : ClassStats) 
  (h1 : stats.total_students = 200)
  (h2 : stats.female_percentage = 3/5)
  (h3 : stats.brunette_percentage = 1/2)
  (h4 : stats.short_brunette_percentage = 1/2)
  (h5 : stats.club_participation_percentage = 2/5)
  (h6 : stats.short_club_percentage = 3/4) :
  probability_short_brunette_club stats = 9/100 := by
  sorry

end NUMINAMATH_CALUDE_probability_calculation_l3610_361029


namespace NUMINAMATH_CALUDE_cubic_polynomial_c_value_l3610_361028

/-- A cubic polynomial function with integer coefficients -/
def f (a b c : ℤ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- Theorem stating that under given conditions, c must equal 16 -/
theorem cubic_polynomial_c_value (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  f a b c a = a^3 → f a b c b = b^3 → c = 16 := by
  sorry


end NUMINAMATH_CALUDE_cubic_polynomial_c_value_l3610_361028


namespace NUMINAMATH_CALUDE_car_owners_without_others_l3610_361077

/-- Represents the number of adults owning each type of vehicle and their intersections -/
structure VehicleOwnership where
  total : ℕ
  cars : ℕ
  motorcycles : ℕ
  bicycles : ℕ
  cars_motorcycles : ℕ
  cars_bicycles : ℕ
  motorcycles_bicycles : ℕ
  all_three : ℕ

/-- The main theorem stating the number of car owners without motorcycles or bicycles -/
theorem car_owners_without_others (v : VehicleOwnership) 
  (h_total : v.total = 500)
  (h_cars : v.cars = 450)
  (h_motorcycles : v.motorcycles = 150)
  (h_bicycles : v.bicycles = 200)
  (h_pie : v.total = v.cars + v.motorcycles + v.bicycles - v.cars_motorcycles - v.cars_bicycles - v.motorcycles_bicycles + v.all_three)
  : v.cars - (v.cars_motorcycles + v.cars_bicycles - v.all_three) = 270 := by
  sorry

/-- A lemma to ensure all adults own at least one vehicle -/
lemma all_adults_own_vehicle (v : VehicleOwnership) 
  (h_total : v.total = 500)
  (h_pie : v.total = v.cars + v.motorcycles + v.bicycles - v.cars_motorcycles - v.cars_bicycles - v.motorcycles_bicycles + v.all_three)
  : v.cars + v.motorcycles + v.bicycles ≥ v.total := by
  sorry

end NUMINAMATH_CALUDE_car_owners_without_others_l3610_361077


namespace NUMINAMATH_CALUDE_integer_product_characterization_l3610_361071

theorem integer_product_characterization (a : ℝ) : 
  (∀ n : ℕ, ∃ m : ℤ, a * n * (n + 2) * (n + 3) * (n + 4) = m) ↔ 
  (∃ k : ℤ, a = k / 6) :=
sorry

end NUMINAMATH_CALUDE_integer_product_characterization_l3610_361071


namespace NUMINAMATH_CALUDE_inequality_condition_l3610_361023

theorem inequality_condition (a : ℝ) : 
  (∀ x, -2 < x ∧ x < -1 → (x + a) * (x + 1) < 0) ∧ 
  (∃ x, (x + a) * (x + 1) < 0 ∧ (x ≤ -2 ∨ x ≥ -1)) ↔ 
  a > 2 := by sorry

end NUMINAMATH_CALUDE_inequality_condition_l3610_361023


namespace NUMINAMATH_CALUDE_equation_solution_l3610_361093

theorem equation_solution : 
  let n : ℝ := 73.0434782609
  0.07 * n + 0.12 * (30 + n) + 0.04 * n = 20.4 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3610_361093


namespace NUMINAMATH_CALUDE_fifth_month_sale_l3610_361054

def sales_1 : ℕ := 5921
def sales_2 : ℕ := 5468
def sales_3 : ℕ := 5568
def sales_4 : ℕ := 6088
def sales_6 : ℕ := 5922
def average_sale : ℕ := 5900
def num_months : ℕ := 6

theorem fifth_month_sale :
  ∃ (sales_5 : ℕ),
    sales_5 = average_sale * num_months - (sales_1 + sales_2 + sales_3 + sales_4 + sales_6) ∧
    sales_5 = 6433 := by
  sorry

end NUMINAMATH_CALUDE_fifth_month_sale_l3610_361054


namespace NUMINAMATH_CALUDE_remainder_eleven_pow_thousand_mod_five_hundred_l3610_361014

theorem remainder_eleven_pow_thousand_mod_five_hundred :
  11^1000 % 500 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_eleven_pow_thousand_mod_five_hundred_l3610_361014


namespace NUMINAMATH_CALUDE_alcohol_water_ratio_l3610_361074

theorem alcohol_water_ratio (mixture : ℝ) (alcohol water : ℝ) 
  (h1 : alcohol = (1 : ℝ) / 7 * mixture) 
  (h2 : water = (2 : ℝ) / 7 * mixture) : 
  alcohol / water = (1 : ℝ) / 2 := by
sorry

end NUMINAMATH_CALUDE_alcohol_water_ratio_l3610_361074


namespace NUMINAMATH_CALUDE_relationship_abc_l3610_361016

theorem relationship_abc : 
  let a : ℝ := Real.rpow 0.8 0.7
  let b : ℝ := Real.rpow 0.8 0.9
  let c : ℝ := Real.rpow 1.1 0.6
  c > a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l3610_361016


namespace NUMINAMATH_CALUDE_pool_filling_time_l3610_361018

def tap1_time : ℝ := 3
def tap2_time : ℝ := 6
def tap3_time : ℝ := 12

theorem pool_filling_time :
  let combined_rate := 1 / tap1_time + 1 / tap2_time + 1 / tap3_time
  (1 / combined_rate) = 12 / 7 :=
by sorry

end NUMINAMATH_CALUDE_pool_filling_time_l3610_361018


namespace NUMINAMATH_CALUDE_fireworks_saved_l3610_361065

/-- The number of fireworks Henry and his friend had saved from last year -/
def fireworks_problem (henry_new : ℕ) (friend_new : ℕ) (total : ℕ) : Prop :=
  henry_new + friend_new + (total - (henry_new + friend_new)) = total

theorem fireworks_saved (henry_new friend_new total : ℕ) 
  (h1 : henry_new = 2)
  (h2 : friend_new = 3)
  (h3 : total = 11) :
  fireworks_problem henry_new friend_new total ∧ 
  (total - (henry_new + friend_new) = 6) :=
by sorry

end NUMINAMATH_CALUDE_fireworks_saved_l3610_361065
