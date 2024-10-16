import Mathlib

namespace NUMINAMATH_CALUDE_root_equation_problem_l2580_258096

/-- Given two equations with constants p and q, prove that p = 5, q = -10, and 50p + q = 240 -/
theorem root_equation_problem (p q : ℝ) : 
  (∃! x y : ℝ, x ≠ y ∧ ((x + p) * (x + q) * (x - 8) = 0 ∨ x = 5)) →
  (∃! x y : ℝ, x ≠ y ∧ ((x + 2*p) * (x - 5) * (x - 10) = 0 ∨ x = -q ∨ x = 8)) →
  p = 5 ∧ q = -10 ∧ 50*p + q = 240 := by
sorry


end NUMINAMATH_CALUDE_root_equation_problem_l2580_258096


namespace NUMINAMATH_CALUDE_factorization_sum_l2580_258026

theorem factorization_sum (a b : ℤ) : 
  (∀ x : ℝ, 25 * x^2 - 160 * x - 336 = (5 * x + a) * (5 * x + b)) → 
  a + 2 * b = 20 := by
  sorry

end NUMINAMATH_CALUDE_factorization_sum_l2580_258026


namespace NUMINAMATH_CALUDE_edward_money_problem_l2580_258001

/-- Proves that if a person spends $17, then receives $10, and ends up with $7, they must have started with $14. -/
theorem edward_money_problem (initial_amount spent received final_amount : ℤ) :
  spent = 17 →
  received = 10 →
  final_amount = 7 →
  initial_amount - spent + received = final_amount →
  initial_amount = 14 :=
by sorry

end NUMINAMATH_CALUDE_edward_money_problem_l2580_258001


namespace NUMINAMATH_CALUDE_prob_three_even_out_of_six_l2580_258076

/-- A fair 20-sided die -/
def Die : Type := Fin 20

/-- The probability of a single die showing an even number -/
def prob_even : ℚ := 1/2

/-- The number of dice rolled -/
def num_dice : ℕ := 6

/-- The number of dice we want to show even numbers -/
def target_even : ℕ := 3

/-- The probability of exactly three out of six fair 20-sided dice showing an even number -/
theorem prob_three_even_out_of_six : 
  (Nat.choose num_dice target_even : ℚ) * prob_even^target_even * (1 - prob_even)^(num_dice - target_even) = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_even_out_of_six_l2580_258076


namespace NUMINAMATH_CALUDE_mary_initial_money_l2580_258002

/-- The amount of money Mary had before buying the pie -/
def initial_money : ℕ := sorry

/-- The cost of the pie -/
def pie_cost : ℕ := 6

/-- The amount of money Mary has after buying the pie -/
def remaining_money : ℕ := 52

theorem mary_initial_money : 
  initial_money = remaining_money + pie_cost := by sorry

end NUMINAMATH_CALUDE_mary_initial_money_l2580_258002


namespace NUMINAMATH_CALUDE_tangent_line_at_1_1_l2580_258083

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 3

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Theorem statement
theorem tangent_line_at_1_1 :
  let point : ℝ × ℝ := (1, 1)
  let slope : ℝ := f' point.1
  let tangent_line (x : ℝ) : ℝ := slope * (x - point.1) + point.2
  ∀ x, tangent_line x = -3 * x + 4 := by
sorry


end NUMINAMATH_CALUDE_tangent_line_at_1_1_l2580_258083


namespace NUMINAMATH_CALUDE_relay_race_distance_l2580_258059

/-- A runner in the relay race -/
structure Runner where
  name : String
  speed : Real
  time : Real

/-- The relay race -/
def RelayRace (runners : List Runner) (totalTime : Real) : Prop :=
  (List.sum (List.map (fun r => r.speed * r.time) runners) = 17) ∧
  (List.sum (List.map (fun r => r.time) runners) = totalTime)

theorem relay_race_distance :
  let sadie : Runner := ⟨"Sadie", 3, 2⟩
  let ariana : Runner := ⟨"Ariana", 6, 0.5⟩
  let sarah : Runner := ⟨"Sarah", 4, 2⟩
  let runners : List Runner := [sadie, ariana, sarah]
  let totalTime : Real := 4.5
  RelayRace runners totalTime := by sorry

end NUMINAMATH_CALUDE_relay_race_distance_l2580_258059


namespace NUMINAMATH_CALUDE_tan_product_squared_l2580_258005

theorem tan_product_squared (a b : ℝ) :
  3 * (Real.cos a + Real.cos b) + 5 * (Real.cos a * Real.cos b - 1) = 0 →
  (Real.tan (a / 2) * Real.tan (b / 2))^2 = 6 / 13 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_squared_l2580_258005


namespace NUMINAMATH_CALUDE_hyperbola_param_sum_l2580_258079

/-- A hyperbola with given center, focus, and vertex -/
structure Hyperbola where
  center : ℝ × ℝ
  focus : ℝ × ℝ
  vertex : ℝ × ℝ

/-- Parameters of the hyperbola equation -/
structure HyperbolaParams where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- Given a hyperbola, compute its equation parameters -/
def computeParams (hyp : Hyperbola) : HyperbolaParams := sorry

theorem hyperbola_param_sum :
  let hyp : Hyperbola := {
    center := (1, -1),
    focus := (1, 5),
    vertex := (1, 1)
  }
  let params := computeParams hyp
  params.h + params.k + params.a + params.b = 2 + 4 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_param_sum_l2580_258079


namespace NUMINAMATH_CALUDE_chicken_wings_distribution_l2580_258084

theorem chicken_wings_distribution (num_friends : ℕ) (initial_wings : ℕ) (additional_wings : ℕ) :
  num_friends = 4 →
  initial_wings = 9 →
  additional_wings = 7 →
  (initial_wings + additional_wings) / num_friends = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_chicken_wings_distribution_l2580_258084


namespace NUMINAMATH_CALUDE_f_continuous_at_1_l2580_258082

def f (x : ℝ) : ℝ := -4 * x^2 - 6

theorem f_continuous_at_1 : 
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 1| < δ → |f x - f 1| < ε :=
by sorry

end NUMINAMATH_CALUDE_f_continuous_at_1_l2580_258082


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_three_l2580_258042

theorem subset_implies_a_equals_three (A B : Set ℕ) (a : ℕ) 
  (hA : A = {1, 3}) 
  (hB : B = {1, 2, a}) 
  (hSubset : A ⊆ B) : 
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_three_l2580_258042


namespace NUMINAMATH_CALUDE_train_length_l2580_258011

/-- The length of a train given its speed, platform length, and time to cross the platform. -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 55 * (1000 / 3600) →
  platform_length = 520 →
  crossing_time = 64.79481641468682 →
  (train_speed * crossing_time) - platform_length = 470 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2580_258011


namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l2580_258072

theorem hyperbolas_same_asymptotes (M : ℝ) : 
  (∀ x y : ℝ, x^2/9 - y^2/16 = 1 ↔ x^2/25 - y^2/M = 1) → M = 400/9 := by
  sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l2580_258072


namespace NUMINAMATH_CALUDE_expand_polynomial_l2580_258039

theorem expand_polynomial (x : ℝ) : 
  (x - 3) * (x + 3) * (x^2 + 2*x + 5) = x^4 + 2*x^3 - 4*x^2 - 18*x - 45 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l2580_258039


namespace NUMINAMATH_CALUDE_fifth_term_of_special_arithmetic_sequence_l2580_258031

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- The main theorem -/
theorem fifth_term_of_special_arithmetic_sequence (seq : ArithmeticSequence) 
    (h1 : seq.a 1 = 2)
    (h2 : 3 * seq.S 3 = seq.S 2 + seq.S 4) : 
  seq.a 5 = -10 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_special_arithmetic_sequence_l2580_258031


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2580_258004

theorem polynomial_remainder_theorem (c d : ℚ) : 
  let g (x : ℚ) := c * x^3 - 8 * x^2 + d * x - 7
  (g 2 = -7) → (g (-3) = -80) → (c = -47/15 ∧ d = 428/15) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2580_258004


namespace NUMINAMATH_CALUDE_correct_addition_l2580_258038

theorem correct_addition (x : ℤ) (h : x + 42 = 50) : x + 24 = 32 := by
  sorry

end NUMINAMATH_CALUDE_correct_addition_l2580_258038


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l2580_258034

/-- Given a college with 416 total students and 160 girls, the ratio of boys to girls is 8:5 -/
theorem boys_to_girls_ratio (total_students : ℕ) (girls : ℕ) : 
  total_students = 416 → girls = 160 → 
  (total_students - girls) / girls = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l2580_258034


namespace NUMINAMATH_CALUDE_special_op_is_addition_l2580_258046

/-- An operation on real numbers satisfying (a * b) * c = a + b + c for all a, b, c -/
def special_op (a b : ℝ) : ℝ := sorry

/-- The property that (a * b) * c = a + b + c for all a, b, c -/
axiom special_op_property (a b c : ℝ) : special_op (special_op a b) c = a + b + c

/-- Theorem: The special operation is equivalent to addition -/
theorem special_op_is_addition (a b : ℝ) : special_op a b = a + b := by
  sorry

end NUMINAMATH_CALUDE_special_op_is_addition_l2580_258046


namespace NUMINAMATH_CALUDE_twenty_three_to_binary_l2580_258017

-- Define a function to convert a natural number to its binary representation
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec aux (m : ℕ) : List Bool :=
      if m = 0 then []
      else (m % 2 = 1) :: aux (m / 2)
    aux n

-- Define the decimal number we want to convert
def decimal_number : ℕ := 23

-- Define the expected binary representation
def expected_binary : List Bool := [true, true, true, false, true]

-- Theorem statement
theorem twenty_three_to_binary :
  to_binary decimal_number = expected_binary := by sorry

end NUMINAMATH_CALUDE_twenty_three_to_binary_l2580_258017


namespace NUMINAMATH_CALUDE_point_B_left_of_A_l2580_258015

theorem point_B_left_of_A : 8/13 < 5/8 := by sorry

end NUMINAMATH_CALUDE_point_B_left_of_A_l2580_258015


namespace NUMINAMATH_CALUDE_f_properties_l2580_258037

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3 / 2) * Real.sin x * Real.cos x + (1 + Real.cos (2 * x)) / 4

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ T = Real.pi ∧ ∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (k : ℤ), ∀ (x : ℝ), 
    k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6 →
    ∀ (y : ℝ), k * Real.pi - Real.pi / 3 ≤ y ∧ y ≤ x → f y ≤ f x) ∧
  (∀ (A B C : ℝ) (a b c : ℝ),
    f A = 1/2 → b + c = 3 →
    a = Real.sqrt (b^2 + c^2 - 2*b*c*Real.cos A) →
    a ≥ 3/2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2580_258037


namespace NUMINAMATH_CALUDE_tabitha_current_age_l2580_258016

/-- Tabitha's hair color tradition --/
def tabitha_age : ℕ → Prop :=
  fun current_age =>
    ∃ (colors : ℕ),
      colors = current_age - 15 + 2 ∧
      colors + 3 = 8

theorem tabitha_current_age :
  ∃ (age : ℕ), tabitha_age age ∧ age = 20 := by
  sorry

end NUMINAMATH_CALUDE_tabitha_current_age_l2580_258016


namespace NUMINAMATH_CALUDE_train_count_l2580_258006

theorem train_count (carriages_per_train : ℕ) (rows_per_carriage : ℕ) (wheels_per_row : ℕ) (total_wheels : ℕ) :
  carriages_per_train = 4 →
  rows_per_carriage = 3 →
  wheels_per_row = 5 →
  total_wheels = 240 →
  total_wheels / (carriages_per_train * rows_per_carriage * wheels_per_row) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_train_count_l2580_258006


namespace NUMINAMATH_CALUDE_tangent_sum_ratio_l2580_258088

theorem tangent_sum_ratio : 
  (Real.tan (10 * π / 180) + Real.tan (50 * π / 180) + Real.tan (120 * π / 180)) / 
  (Real.tan (10 * π / 180) * Real.tan (50 * π / 180)) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_ratio_l2580_258088


namespace NUMINAMATH_CALUDE_solve_for_y_l2580_258043

theorem solve_for_y (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 8) : y = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2580_258043


namespace NUMINAMATH_CALUDE_total_rhino_weight_l2580_258074

/-- The weight of a white rhino in pounds -/
def white_rhino_weight : ℕ := 5100

/-- The weight of a black rhino in pounds -/
def black_rhino_weight : ℕ := 2000

/-- The number of white rhinos -/
def num_white_rhinos : ℕ := 7

/-- The number of black rhinos -/
def num_black_rhinos : ℕ := 8

/-- Theorem: The total weight of 7 white rhinos and 8 black rhinos is 51,700 pounds -/
theorem total_rhino_weight :
  num_white_rhinos * white_rhino_weight + num_black_rhinos * black_rhino_weight = 51700 := by
  sorry

end NUMINAMATH_CALUDE_total_rhino_weight_l2580_258074


namespace NUMINAMATH_CALUDE_power_division_l2580_258060

theorem power_division (m : ℝ) : m^10 / m^5 = m^5 := by
  sorry

end NUMINAMATH_CALUDE_power_division_l2580_258060


namespace NUMINAMATH_CALUDE_cherry_pitting_time_l2580_258075

/-- Time required to pit cherries for a pie --/
theorem cherry_pitting_time
  (pounds_needed : ℕ)
  (cherries_per_pound : ℕ)
  (pitting_time : ℕ)
  (cherries_per_batch : ℕ)
  (h1 : pounds_needed = 3)
  (h2 : cherries_per_pound = 80)
  (h3 : pitting_time = 10)
  (h4 : cherries_per_batch = 20) :
  (pounds_needed * cherries_per_pound * pitting_time) / (cherries_per_batch * 60) = 2 :=
by sorry

end NUMINAMATH_CALUDE_cherry_pitting_time_l2580_258075


namespace NUMINAMATH_CALUDE_reflection_result_l2580_258033

/-- Reflects a point (x, y) across the line x = k -/
def reflect_point (x y k : ℝ) : ℝ × ℝ := (2 * k - x, y)

/-- Reflects a line y = mx + c across x = k -/
def reflect_line (m c k : ℝ) : ℝ × ℝ := 
  let point := reflect_point k (m * k + c) k
  (-m, 2 * m * k + c - m * point.1)

theorem reflection_result : 
  let original_slope : ℝ := -2
  let original_intercept : ℝ := 7
  let reflection_line : ℝ := 3
  let (a, b) := reflect_line original_slope original_intercept reflection_line
  2 * a + b = -1 := by sorry

end NUMINAMATH_CALUDE_reflection_result_l2580_258033


namespace NUMINAMATH_CALUDE_tangent_parallel_implies_a_equals_one_l2580_258014

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1

theorem tangent_parallel_implies_a_equals_one (a : ℝ) :
  (f' a 1 = 4) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_implies_a_equals_one_l2580_258014


namespace NUMINAMATH_CALUDE_biking_difference_l2580_258062

/-- Calculates the difference in miles biked between two cyclists given their speeds, 
    total time, and break times. -/
def miles_difference (alberto_speed bjorn_speed total_time alberto_break bjorn_break : ℝ) : ℝ :=
  let alberto_distance := alberto_speed * (total_time - alberto_break)
  let bjorn_distance := bjorn_speed * (total_time - bjorn_break)
  alberto_distance - bjorn_distance

/-- The difference in miles biked between Alberto and Bjorn is 17.625 miles. -/
theorem biking_difference : 
  miles_difference 15 10.5 5 0.5 0.25 = 17.625 := by
  sorry

end NUMINAMATH_CALUDE_biking_difference_l2580_258062


namespace NUMINAMATH_CALUDE_only_cone_cannot_have_quadrilateral_cross_section_l2580_258091

-- Define the types of solids
inductive Solid
  | Cylinder
  | Cone
  | FrustumOfCone
  | Prism

-- Define a function that checks if a solid can have a quadrilateral cross-section
def canHaveQuadrilateralCrossSection (s : Solid) : Prop :=
  match s with
  | Solid.Cylinder => true
  | Solid.Cone => false
  | Solid.FrustumOfCone => true
  | Solid.Prism => true

-- Theorem stating that only a Cone cannot have a quadrilateral cross-section
theorem only_cone_cannot_have_quadrilateral_cross_section :
  ∀ s : Solid, ¬(canHaveQuadrilateralCrossSection s) ↔ s = Solid.Cone :=
by
  sorry


end NUMINAMATH_CALUDE_only_cone_cannot_have_quadrilateral_cross_section_l2580_258091


namespace NUMINAMATH_CALUDE_train_journey_time_l2580_258063

theorem train_journey_time (S : ℝ) (x : ℝ) (h1 : x > 0) (h2 : S > 0) :
  (S / (2 * x) + S / (2 * 0.75 * x)) - S / x = 0.5 →
  S / x + 0.5 = 3.5 := by
sorry

end NUMINAMATH_CALUDE_train_journey_time_l2580_258063


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l2580_258065

theorem circle_diameter_from_area :
  ∀ (r : ℝ), r > 0 → π * r^2 = 150 * π → 2 * r = 10 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l2580_258065


namespace NUMINAMATH_CALUDE_average_height_combined_l2580_258010

theorem average_height_combined (group1_count group2_count : ℕ) 
  (group1_avg group2_avg : ℝ) (total_count : ℕ) :
  group1_count = 20 →
  group2_count = 11 →
  group1_avg = 20 →
  group2_avg = 20 →
  total_count = group1_count + group2_count →
  (group1_count * group1_avg + group2_count * group2_avg) / total_count = 20 := by
  sorry

end NUMINAMATH_CALUDE_average_height_combined_l2580_258010


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_9973_l2580_258025

theorem largest_prime_factor_of_9973 :
  ∃ (p : ℕ), p.Prime ∧ p ∣ 9973 ∧ p = 103 ∧ ∀ (q : ℕ), q.Prime → q ∣ 9973 → q ≤ p :=
by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_9973_l2580_258025


namespace NUMINAMATH_CALUDE_equation_solution_l2580_258048

theorem equation_solution (x : ℝ) : (2*x - 1)^2 = 81 → x = 5 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2580_258048


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l2580_258057

/-- The equation of the asymptotes of a hyperbola -/
def asymptote_equation (a b : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | y = (b / a) * x ∨ y = -(b / a) * x}

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 - y^2 / 3 = 1

theorem hyperbola_asymptote :
  ∀ x y : ℝ, hyperbola_equation x y →
  (x, y) ∈ asymptote_equation 1 (Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l2580_258057


namespace NUMINAMATH_CALUDE_bobby_candy_problem_l2580_258041

theorem bobby_candy_problem (x : ℕ) : 
  (x - 5 - 9 = 7) → x = 21 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_problem_l2580_258041


namespace NUMINAMATH_CALUDE_reading_time_difference_l2580_258087

/-- Proves that the difference in reading time between Lee and Kai is 150 minutes -/
theorem reading_time_difference 
  (kai_speed : ℝ) 
  (lee_speed : ℝ) 
  (book_pages : ℝ) 
  (h1 : kai_speed = 120) 
  (h2 : lee_speed = 60) 
  (h3 : book_pages = 300) : 
  (book_pages / lee_speed - book_pages / kai_speed) * 60 = 150 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_difference_l2580_258087


namespace NUMINAMATH_CALUDE_tree_spacing_l2580_258009

/-- Given a yard of length 400 meters with 26 equally spaced trees, including one at each end,
    the distance between consecutive trees is 16 meters. -/
theorem tree_spacing (yard_length : ℝ) (num_trees : ℕ) (h1 : yard_length = 400) (h2 : num_trees = 26) :
  yard_length / (num_trees - 1) = 16 :=
sorry

end NUMINAMATH_CALUDE_tree_spacing_l2580_258009


namespace NUMINAMATH_CALUDE_sector_perimeter_l2580_258081

/-- Given a circular sector with central angle 2/3π and area 3π, its perimeter is 6 + 2π. -/
theorem sector_perimeter (θ : Real) (S : Real) (R : Real) (l : Real) :
  θ = (2/3) * Real.pi →
  S = 3 * Real.pi →
  S = (1/2) * θ * R^2 →
  l = θ * R →
  (l + 2 * R) = 6 + 2 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sector_perimeter_l2580_258081


namespace NUMINAMATH_CALUDE_min_value_theorem_l2580_258052

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∃ (min : ℝ), min = 6 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 1/y = 1 → 1/(x-1) + 9/(y-1) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2580_258052


namespace NUMINAMATH_CALUDE_seventh_observation_value_l2580_258061

theorem seventh_observation_value 
  (n : ℕ) 
  (initial_avg : ℝ) 
  (decrease : ℝ) 
  (h1 : n = 6) 
  (h2 : initial_avg = 12) 
  (h3 : decrease = 1) : 
  let new_avg := initial_avg - decrease
  let new_obs := (n + 1) * new_avg - n * initial_avg
  new_obs = 5 := by sorry

end NUMINAMATH_CALUDE_seventh_observation_value_l2580_258061


namespace NUMINAMATH_CALUDE_rebus_solution_l2580_258024

theorem rebus_solution : 
  ∃! (A B C : ℕ), 
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    A < 10 ∧ B < 10 ∧ C < 10 ∧
    (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ∧
    A = 4 ∧ B = 7 ∧ C = 6 := by
  sorry

end NUMINAMATH_CALUDE_rebus_solution_l2580_258024


namespace NUMINAMATH_CALUDE_max_vector_difference_is_sqrt2_l2580_258089

theorem max_vector_difference_is_sqrt2 :
  ∀ θ : ℝ,
  let a : Fin 2 → ℝ := ![1, Real.sin θ]
  let b : Fin 2 → ℝ := ![1, Real.cos θ]
  (∀ φ : ℝ, ‖a - b‖ ≤ ‖![1, Real.sin φ] - ![1, Real.cos φ]‖) →
  ‖a - b‖ = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_vector_difference_is_sqrt2_l2580_258089


namespace NUMINAMATH_CALUDE_tangent_slope_determines_point_l2580_258003

/-- Given a curve y = 2x^2 + 4x, prove that if the slope of the tangent line
    at point P is 16, then the coordinates of P are (3, 30). -/
theorem tangent_slope_determines_point :
  ∀ x y : ℝ,
  (y = 2 * x^2 + 4 * x) →  -- Curve equation
  ((4 * x + 4) = 16) →     -- Slope of tangent line is 16
  (x = 3 ∧ y = 30)         -- Coordinates of point P
  := by sorry

end NUMINAMATH_CALUDE_tangent_slope_determines_point_l2580_258003


namespace NUMINAMATH_CALUDE_mango_rate_is_75_l2580_258020

/-- The rate of mangoes per kg given the purchase details -/
def mango_rate (apple_weight : ℕ) (apple_rate : ℕ) (mango_weight : ℕ) (total_paid : ℕ) : ℕ :=
  (total_paid - apple_weight * apple_rate) / mango_weight

/-- Theorem stating that the rate of mangoes is 75 per kg -/
theorem mango_rate_is_75 :
  mango_rate 8 70 9 1235 = 75 := by
  sorry

#eval mango_rate 8 70 9 1235

end NUMINAMATH_CALUDE_mango_rate_is_75_l2580_258020


namespace NUMINAMATH_CALUDE_anns_shopping_cost_anns_shopping_proof_l2580_258044

theorem anns_shopping_cost (total_spent : ℕ) (shorts_quantity : ℕ) (shorts_price : ℕ) 
  (shoes_quantity : ℕ) (shoes_price : ℕ) (tops_quantity : ℕ) : ℕ :=
  let shorts_total := shorts_quantity * shorts_price
  let shoes_total := shoes_quantity * shoes_price
  let tops_total := total_spent - shorts_total - shoes_total
  tops_total / tops_quantity

theorem anns_shopping_proof :
  anns_shopping_cost 75 5 7 2 10 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_anns_shopping_cost_anns_shopping_proof_l2580_258044


namespace NUMINAMATH_CALUDE_part_one_part_two_l2580_258077

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem for part 1 of the problem -/
theorem part_one (t : Triangle) (h : 2 * t.a * Real.sin t.B = Real.sqrt 3 * t.b) :
  t.A = Real.pi / 3 ∨ t.A = 2 * Real.pi / 3 :=
sorry

/-- Theorem for part 2 of the problem -/
theorem part_two (t : Triangle) (h : t.a / 2 = t.b * Real.sin t.A) :
  (∀ x : Triangle, x.c / x.b + x.b / x.c ≤ 2 * Real.sqrt 2) ∧
  (∃ x : Triangle, x.c / x.b + x.b / x.c = 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2580_258077


namespace NUMINAMATH_CALUDE_paper_boat_time_l2580_258045

/-- The time it takes for a paper boat to travel along an embankment -/
theorem paper_boat_time (embankment_length : ℝ) (boat_length : ℝ) 
  (downstream_time : ℝ) (upstream_time : ℝ) 
  (h1 : embankment_length = 50) 
  (h2 : boat_length = 10)
  (h3 : downstream_time = 5)
  (h4 : upstream_time = 4) : 
  ∃ (paper_boat_time : ℝ), paper_boat_time = 40 := by
  sorry

end NUMINAMATH_CALUDE_paper_boat_time_l2580_258045


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2580_258093

theorem absolute_value_inequality (x : ℝ) : 
  ‖‖x - 2‖ - 1‖ ≤ 1 ↔ 0 ≤ x ∧ x ≤ 4 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2580_258093


namespace NUMINAMATH_CALUDE_arithmetic_geometric_intersection_l2580_258032

/-- An arithmetic sequence of integers -/
def ArithmeticSequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def GeometricSequence (b : ℕ → ℤ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = r * b n

/-- The theorem statement -/
theorem arithmetic_geometric_intersection (a : ℕ → ℤ) (d : ℤ) (n₁ : ℕ) :
  d ≠ 0 →
  ArithmeticSequence a d →
  a 5 = 6 →
  5 < n₁ →
  GeometricSequence (fun n ↦ if n = 1 then a 3 else if n = 2 then a 5 else a (n₁ + n - 3)) →
  (∃ k : ℕ, k ≤ 7 ∧
    ∀ n : ℕ, n ≤ 2015 →
      (∃ m : ℕ, m ≤ k ∧ a n = if m = 1 then a 3 else if m = 2 then a 5 else a (n₁ + m - 3))) ∧
  (∀ k : ℕ, k > 7 →
    ¬∀ n : ℕ, n ≤ 2015 →
      (∃ m : ℕ, m ≤ k ∧ a n = if m = 1 then a 3 else if m = 2 then a 5 else a (n₁ + m - 3))) :=
by
  sorry


end NUMINAMATH_CALUDE_arithmetic_geometric_intersection_l2580_258032


namespace NUMINAMATH_CALUDE_inequality_solution_l2580_258021

theorem inequality_solution (x : ℝ) : 
  (8 * x^2 + 16 * x - 51) / ((2 * x - 3) * (x + 4)) < 3 ↔ 
  (x > -4 ∧ x < -3) ∨ (x > 3/2 ∧ x < 5/2) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2580_258021


namespace NUMINAMATH_CALUDE_gas_usage_multiple_l2580_258022

theorem gas_usage_multiple (felicity_usage adhira_usage : ℕ) 
  (h1 : felicity_usage = 23)
  (h2 : adhira_usage = 7)
  (h3 : ∃ m : ℕ, felicity_usage = m * adhira_usage - 5) :
  ∃ m : ℕ, m = 4 ∧ felicity_usage = m * adhira_usage - 5 :=
by sorry

end NUMINAMATH_CALUDE_gas_usage_multiple_l2580_258022


namespace NUMINAMATH_CALUDE_a_18_value_l2580_258050

def equal_sum_sequence (a : ℕ → ℝ) :=
  ∃ k : ℝ, ∀ n : ℕ, a n + a (n + 1) = k

theorem a_18_value (a : ℕ → ℝ) (h1 : equal_sum_sequence a) (h2 : a 1 = 2) (h3 : ∃ k : ℝ, k = 5 ∧ ∀ n : ℕ, a n + a (n + 1) = k) :
  a 18 = 3 := by
  sorry

end NUMINAMATH_CALUDE_a_18_value_l2580_258050


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_min_value_a_plus_2b_exact_min_value_a_plus_2b_equality_l2580_258030

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / (a + 2) + 1 / (b + 2) = 1 / 3) : 
  ∀ x y, x > 0 → y > 0 → 1 / (x + 2) + 1 / (y + 2) = 1 / 3 → a + 2 * b ≤ x + 2 * y :=
by sorry

theorem min_value_a_plus_2b_exact (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / (a + 2) + 1 / (b + 2) = 1 / 3) : 
  a + 2 * b ≥ 3 + 6 * Real.sqrt 2 :=
by sorry

theorem min_value_a_plus_2b_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / (a + 2) + 1 / (b + 2) = 1 / 3) : 
  (a + 2 * b = 3 + 6 * Real.sqrt 2) ↔ 
  (a = 1 + 3 * Real.sqrt 2 ∧ b = 1 + 3 * Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_min_value_a_plus_2b_exact_min_value_a_plus_2b_equality_l2580_258030


namespace NUMINAMATH_CALUDE_milk_remaining_l2580_258018

theorem milk_remaining (initial : ℚ) (given_away : ℚ) (remaining : ℚ) :
  initial = 5 →
  given_away = 17 / 4 →
  remaining = initial - given_away →
  remaining = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_milk_remaining_l2580_258018


namespace NUMINAMATH_CALUDE_min_gumballs_for_four_is_eleven_l2580_258086

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine where
  red : Nat
  white : Nat
  blue : Nat

/-- Represents the minimum number of gumballs needed to guarantee four of the same color -/
def minGumballsForFour (machine : GumballMachine) : Nat :=
  11

/-- Theorem stating that for the given gumball machine, 11 is the minimum number of gumballs 
    needed to guarantee four of the same color -/
theorem min_gumballs_for_four_is_eleven (machine : GumballMachine) 
    (h1 : machine.red = 10) 
    (h2 : machine.white = 7) 
    (h3 : machine.blue = 6) : 
  minGumballsForFour machine = 11 := by
  sorry

#eval minGumballsForFour { red := 10, white := 7, blue := 6 }

end NUMINAMATH_CALUDE_min_gumballs_for_four_is_eleven_l2580_258086


namespace NUMINAMATH_CALUDE_prove_research_paper_requirement_l2580_258000

def research_paper_requirement (yvonne_words janna_extra_words removed_words added_multiplier additional_words : ℕ) : Prop :=
  let janna_words := yvonne_words + janna_extra_words
  let initial_total := yvonne_words + janna_words
  let after_removal := initial_total - removed_words
  let added_words := removed_words * added_multiplier
  let after_addition := after_removal + added_words
  let final_requirement := after_addition + additional_words
  final_requirement = 1000

theorem prove_research_paper_requirement :
  research_paper_requirement 400 150 20 2 30 := by
  sorry

end NUMINAMATH_CALUDE_prove_research_paper_requirement_l2580_258000


namespace NUMINAMATH_CALUDE_color_drawing_cost_is_240_l2580_258058

/-- The cost of a color drawing given the cost of a black and white drawing and the additional percentage for color. -/
def color_drawing_cost (bw_cost : ℝ) (color_percentage : ℝ) : ℝ :=
  bw_cost * (1 + color_percentage)

/-- Theorem stating that the cost of a color drawing is $240 given the specified conditions. -/
theorem color_drawing_cost_is_240 :
  color_drawing_cost 160 0.5 = 240 := by
  sorry

end NUMINAMATH_CALUDE_color_drawing_cost_is_240_l2580_258058


namespace NUMINAMATH_CALUDE_unique_triple_l2580_258078

theorem unique_triple : 
  ∀ a b c : ℝ,
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a + b + c = 3) →
  (a^2 - a ≥ 1 - b*c) →
  (b^2 - b ≥ 1 - a*c) →
  (c^2 - c ≥ 1 - a*b) →
  (a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_triple_l2580_258078


namespace NUMINAMATH_CALUDE_shells_calculation_l2580_258008

/-- Given an initial amount of shells and an additional amount of shells,
    calculate the total amount of shells. -/
def total_shells (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem stating that given 5 pounds of shells initially and 23 pounds added,
    the total is 28 pounds. -/
theorem shells_calculation :
  total_shells 5 23 = 28 := by
  sorry

end NUMINAMATH_CALUDE_shells_calculation_l2580_258008


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2580_258068

theorem sum_of_squares_of_roots (a b c : ℝ) : 
  (3 * a^3 + 2 * a^2 - 3 * a - 8 = 0) →
  (3 * b^3 + 2 * b^2 - 3 * b - 8 = 0) →
  (3 * c^3 + 2 * c^2 - 3 * c - 8 = 0) →
  a^2 + b^2 + c^2 = 22 / 9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2580_258068


namespace NUMINAMATH_CALUDE_shirts_not_washed_l2580_258051

theorem shirts_not_washed (short_sleeve : ℕ) (long_sleeve : ℕ) (washed : ℕ) 
  (h1 : short_sleeve = 9)
  (h2 : long_sleeve = 21)
  (h3 : washed = 29) :
  short_sleeve + long_sleeve - washed = 1 := by
  sorry

end NUMINAMATH_CALUDE_shirts_not_washed_l2580_258051


namespace NUMINAMATH_CALUDE_tshirt_price_is_8_l2580_258090

-- Define the prices and quantities
def sweater_price : ℝ := 18
def jacket_original_price : ℝ := 80
def jacket_discount : ℝ := 0.1
def sales_tax : ℝ := 0.05
def num_tshirts : ℕ := 6
def num_sweaters : ℕ := 4
def num_jackets : ℕ := 5
def total_cost : ℝ := 504

-- Define the function to calculate the total cost
def calculate_total_cost (tshirt_price : ℝ) : ℝ :=
  let jacket_price := jacket_original_price * (1 - jacket_discount)
  let subtotal := num_tshirts * tshirt_price + num_sweaters * sweater_price + num_jackets * jacket_price
  subtotal * (1 + sales_tax)

-- Theorem to prove
theorem tshirt_price_is_8 :
  ∃ (tshirt_price : ℝ), calculate_total_cost tshirt_price = total_cost ∧ tshirt_price = 8 :=
sorry

end NUMINAMATH_CALUDE_tshirt_price_is_8_l2580_258090


namespace NUMINAMATH_CALUDE_marco_marie_age_difference_l2580_258012

theorem marco_marie_age_difference (marie_age : ℕ) (total_age : ℕ) : 
  marie_age = 12 → 
  total_age = 37 → 
  ∃ (marco_age : ℕ), marco_age + marie_age = total_age ∧ marco_age = 2 * marie_age + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_marco_marie_age_difference_l2580_258012


namespace NUMINAMATH_CALUDE_money_division_l2580_258099

theorem money_division (amanda ben carlos total : ℕ) : 
  amanda + ben + carlos = total →
  amanda = 3 * (ben / 5) →
  carlos = 9 * (ben / 5) →
  ben = 50 →
  total = 170 :=
by sorry

end NUMINAMATH_CALUDE_money_division_l2580_258099


namespace NUMINAMATH_CALUDE_smallest_common_factor_thirty_three_satisfies_smallest_n_is_33_l2580_258066

theorem smallest_common_factor (n : ℕ) : n > 0 ∧ ∃ (k : ℕ), k > 1 ∧ k ∣ (8*n - 3) ∧ k ∣ (6*n + 5) → n ≥ 33 :=
sorry

theorem thirty_three_satisfies : ∃ (k : ℕ), k > 1 ∧ k ∣ (8*33 - 3) ∧ k ∣ (6*33 + 5) :=
sorry

theorem smallest_n_is_33 : (∃ (n : ℕ), n > 0 ∧ ∃ (k : ℕ), k > 1 ∧ k ∣ (8*n - 3) ∧ k ∣ (6*n + 5)) ∧
  (∀ (m : ℕ), m > 0 ∧ ∃ (k : ℕ), k > 1 ∧ k ∣ (8*m - 3) ∧ k ∣ (6*m + 5) → m ≥ 33) :=
sorry

end NUMINAMATH_CALUDE_smallest_common_factor_thirty_three_satisfies_smallest_n_is_33_l2580_258066


namespace NUMINAMATH_CALUDE_prism_with_nine_faces_has_fourteen_vertices_l2580_258029

/-- A prism is a polyhedron with two congruent polygon bases and rectangular lateral faces. -/
structure Prism where
  num_faces : ℕ
  num_base_sides : ℕ
  num_vertices : ℕ

/-- The number of faces in a prism is related to the number of sides in its base. -/
axiom prism_faces (p : Prism) : p.num_faces = p.num_base_sides + 2

/-- The number of vertices in a prism is twice the number of sides in its base. -/
axiom prism_vertices (p : Prism) : p.num_vertices = 2 * p.num_base_sides

/-- Theorem: A prism with 9 faces has 14 vertices. -/
theorem prism_with_nine_faces_has_fourteen_vertices :
  ∃ (p : Prism), p.num_faces = 9 ∧ p.num_vertices = 14 := by
  sorry


end NUMINAMATH_CALUDE_prism_with_nine_faces_has_fourteen_vertices_l2580_258029


namespace NUMINAMATH_CALUDE_smallest_square_side_length_l2580_258047

theorem smallest_square_side_length : ∃ (n : ℕ), 
  (∀ (a b c d : ℕ), a * b * c * d = n * n) ∧ 
  (∃ (x y z w : ℕ), x * 7 = n ∧ y * 8 = n ∧ z * 9 = n ∧ w * 10 = n) ∧
  (∀ (m : ℕ), 
    (∃ (a b c d : ℕ), a * b * c * d = m * m) ∧ 
    (∃ (x y z w : ℕ), x * 7 = m ∧ y * 8 = m ∧ z * 9 = m ∧ w * 10 = m) →
    m ≥ n) ∧
  n = 1008 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_side_length_l2580_258047


namespace NUMINAMATH_CALUDE_car_trip_local_road_distance_l2580_258054

theorem car_trip_local_road_distance 
  (local_speed highway_speed avg_speed : ℝ)
  (highway_distance : ℝ)
  (local_speed_pos : local_speed > 0)
  (highway_speed_pos : highway_speed > 0)
  (avg_speed_pos : avg_speed > 0)
  (highway_distance_pos : highway_distance > 0)
  (h_local_speed : local_speed = 20)
  (h_highway_speed : highway_speed = 60)
  (h_highway_distance : highway_distance = 120)
  (h_avg_speed : avg_speed = 36) :
  ∃ (local_distance : ℝ),
    local_distance > 0 ∧
    (local_distance + highway_distance) / ((local_distance / local_speed) + (highway_distance / highway_speed)) = avg_speed ∧
    local_distance = 60 := by
  sorry

end NUMINAMATH_CALUDE_car_trip_local_road_distance_l2580_258054


namespace NUMINAMATH_CALUDE_initial_alcohol_content_75_percent_l2580_258095

/-- Represents the alcohol content of a solution as a real number between 0 and 1 -/
def AlcoholContent := { x : ℝ // 0 ≤ x ∧ x ≤ 1 }

/-- Proves that the initial alcohol content was 75% given the problem conditions -/
theorem initial_alcohol_content_75_percent 
  (initial_volume : ℝ) 
  (drained_volume : ℝ) 
  (added_content : AlcoholContent) 
  (final_content : AlcoholContent) 
  (h1 : initial_volume = 1)
  (h2 : drained_volume = 0.4)
  (h3 : added_content.val = 0.5)
  (h4 : final_content.val = 0.65) :
  ∃ (initial_content : AlcoholContent), 
    initial_content.val = 0.75 ∧
    (initial_volume - drained_volume) * initial_content.val + 
    drained_volume * added_content.val = 
    initial_volume * final_content.val :=
by sorry


end NUMINAMATH_CALUDE_initial_alcohol_content_75_percent_l2580_258095


namespace NUMINAMATH_CALUDE_log_division_simplification_l2580_258067

theorem log_division_simplification :
  Real.log 27 / Real.log (1 / 27) = -1 := by
  sorry

end NUMINAMATH_CALUDE_log_division_simplification_l2580_258067


namespace NUMINAMATH_CALUDE_trigonometric_expressions_equality_l2580_258040

theorem trigonometric_expressions_equality : 
  (|1 - Real.tan (60 * π / 180)| - (-1/2)⁻¹ + Real.sin (45 * π / 180) + Real.sqrt (1/2) = Real.sqrt 3 + Real.sqrt 2 + 1) ∧ 
  (-1^2022 + Real.sqrt 12 - (π - 3)^0 - Real.cos (30 * π / 180) = -2 + 3/2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expressions_equality_l2580_258040


namespace NUMINAMATH_CALUDE_quadratic_inequality_implication_l2580_258098

theorem quadratic_inequality_implication (y : ℝ) :
  y^2 - 7*y + 12 < 0 → 42 < y^2 + 7*y + 12 ∧ y^2 + 7*y + 12 < 56 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implication_l2580_258098


namespace NUMINAMATH_CALUDE_friendship_distribution_impossibility_l2580_258013

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


end NUMINAMATH_CALUDE_friendship_distribution_impossibility_l2580_258013


namespace NUMINAMATH_CALUDE_insect_count_proof_l2580_258036

/-- Calculates the number of insects given the total number of legs and legs per insect -/
def number_of_insects (total_legs : ℕ) (legs_per_insect : ℕ) : ℕ :=
  total_legs / legs_per_insect

/-- Proves that given 48 insect legs and 6 legs per insect, the number of insects is 8 -/
theorem insect_count_proof :
  let total_legs : ℕ := 48
  let legs_per_insect : ℕ := 6
  number_of_insects total_legs legs_per_insect = 8 := by
  sorry

end NUMINAMATH_CALUDE_insect_count_proof_l2580_258036


namespace NUMINAMATH_CALUDE_watch_cost_price_l2580_258080

theorem watch_cost_price (CP : ℝ) : 
  (0.90 * CP = CP - 0.10 * CP) →
  (1.05 * CP = CP + 0.05 * CP) →
  (1.05 * CP - 0.90 * CP = 180) →
  CP = 1200 :=
by sorry

end NUMINAMATH_CALUDE_watch_cost_price_l2580_258080


namespace NUMINAMATH_CALUDE_picture_book_shelves_l2580_258085

theorem picture_book_shelves (books_per_shelf : ℕ) (mystery_shelves : ℕ) (total_books : ℕ)
  (h1 : books_per_shelf = 8)
  (h2 : mystery_shelves = 5)
  (h3 : total_books = 72) :
  (total_books - mystery_shelves * books_per_shelf) / books_per_shelf = 4 := by
  sorry

end NUMINAMATH_CALUDE_picture_book_shelves_l2580_258085


namespace NUMINAMATH_CALUDE_three_digit_numbers_from_4_and_5_l2580_258028

def is_valid_digit (d : ℕ) : Prop := d = 4 ∨ d = 5

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def is_formed_from_4_and_5 (n : ℕ) : Prop :=
  is_three_digit_number n ∧
  is_valid_digit (n / 100) ∧
  is_valid_digit ((n / 10) % 10) ∧
  is_valid_digit (n % 10)

def valid_numbers : Finset ℕ :=
  {444, 445, 454, 455, 544, 545, 554, 555}

theorem three_digit_numbers_from_4_and_5 :
  ∀ n : ℕ, is_formed_from_4_and_5 n ↔ n ∈ valid_numbers :=
by sorry

end NUMINAMATH_CALUDE_three_digit_numbers_from_4_and_5_l2580_258028


namespace NUMINAMATH_CALUDE_h_is_even_l2580_258071

-- Define k as an even function
def k_even (k : ℝ → ℝ) : Prop := ∀ x, k (-x) = k x

-- Define h in terms of k
def h (k : ℝ → ℝ) (x : ℝ) : ℝ := |k (x^4)|

-- Theorem statement
theorem h_is_even (k : ℝ → ℝ) (h_even : k_even k) : 
  ∀ x, h k (-x) = h k x :=
sorry

end NUMINAMATH_CALUDE_h_is_even_l2580_258071


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2580_258007

/-- For any positive real number a, the function f(x) = a^(x-1) + 2 always passes through the point (1, 3). -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := fun x ↦ a^(x-1) + 2
  f 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2580_258007


namespace NUMINAMATH_CALUDE_sum_reciprocal_n_n_plus_three_l2580_258073

open Real

/-- The sum of the infinite series Σ(1/(n(n+3))) from n=1 to infinity equals 11/18 -/
theorem sum_reciprocal_n_n_plus_three : 
  ∑' n : ℕ+, (1 : ℝ) / (n * (n + 3)) = 11/18 := by sorry

end NUMINAMATH_CALUDE_sum_reciprocal_n_n_plus_three_l2580_258073


namespace NUMINAMATH_CALUDE_first_term_of_geometric_series_l2580_258035

/-- Given an infinite geometric series with first term a and common ratio r -/
def InfiniteGeometricSeries (a : ℝ) (r : ℝ) : Prop :=
  |r| < 1

theorem first_term_of_geometric_series
  (a : ℝ) (r : ℝ)
  (h_series : InfiniteGeometricSeries a r)
  (h_sum : a / (1 - r) = 30)
  (h_sum_squares : a^2 / (1 - r^2) = 180) :
  a = 10 := by
sorry

end NUMINAMATH_CALUDE_first_term_of_geometric_series_l2580_258035


namespace NUMINAMATH_CALUDE_rectangle_area_change_l2580_258092

theorem rectangle_area_change (l w : ℝ) (h : l * w = 1100) :
  (1.1 * l) * (0.9 * w) = 1089 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l2580_258092


namespace NUMINAMATH_CALUDE_real_part_of_one_plus_i_squared_l2580_258055

theorem real_part_of_one_plus_i_squared (i : ℂ) : 
  Complex.re ((1 + i)^2) = 0 := by sorry

end NUMINAMATH_CALUDE_real_part_of_one_plus_i_squared_l2580_258055


namespace NUMINAMATH_CALUDE_reduced_rate_end_time_l2580_258053

/-- Represents the fraction of a week with reduced rates -/
def reduced_rate_fraction : ℚ := 0.6428571428571429

/-- Represents the number of hours in a week -/
def hours_in_week : ℕ := 7 * 24

/-- Represents the number of hours with reduced rates on weekends -/
def weekend_reduced_hours : ℕ := 2 * 24

/-- Represents the hour when reduced rates start on weekdays (24-hour format) -/
def weekday_start_hour : ℕ := 20

/-- Represents the hour when reduced rates end on weekdays (24-hour format) -/
def weekday_end_hour : ℕ := 8

theorem reduced_rate_end_time :
  (reduced_rate_fraction * hours_in_week).floor - weekend_reduced_hours = 
  5 * (24 - weekday_start_hour + weekday_end_hour) :=
sorry

end NUMINAMATH_CALUDE_reduced_rate_end_time_l2580_258053


namespace NUMINAMATH_CALUDE_range_of_m_l2580_258027

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |x - m| ≤ 2) →
  -1 ≤ m ∧ m ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2580_258027


namespace NUMINAMATH_CALUDE_total_marbles_l2580_258049

def jungkook_marbles : ℕ := 3
def marble_difference : ℕ := 4

def jimin_marbles : ℕ := jungkook_marbles + marble_difference

theorem total_marbles :
  jungkook_marbles + jimin_marbles = 10 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_l2580_258049


namespace NUMINAMATH_CALUDE_triple_hash_45_l2580_258094

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.4 * N + 3

-- State the theorem
theorem triple_hash_45 : hash (hash (hash 45)) = 7.56 := by
  sorry

end NUMINAMATH_CALUDE_triple_hash_45_l2580_258094


namespace NUMINAMATH_CALUDE_gcd_problem_l2580_258023

theorem gcd_problem (a : ℤ) (h : 2142 ∣ a) : 
  Nat.gcd (Int.natAbs ((a^2 + 11*a + 28) : ℤ)) (Int.natAbs ((a + 6) : ℤ)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l2580_258023


namespace NUMINAMATH_CALUDE_circle_bounded_area_l2580_258056

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The area of the region bound by two circles and the x-axis -/
def boundedArea (c1 c2 : Circle) : ℝ :=
  sorry

theorem circle_bounded_area :
  let c1 : Circle := { center := (5, 5), radius := 5 }
  let c2 : Circle := { center := (15, 5), radius := 5 }
  boundedArea c1 c2 = 50 - 12.5 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circle_bounded_area_l2580_258056


namespace NUMINAMATH_CALUDE_sergio_income_l2580_258097

/-- Represents the total income from fruit sales -/
def total_income (mango_production : ℕ) (price_per_kg : ℕ) : ℕ :=
  let apple_production := 2 * mango_production
  let orange_production := mango_production + 200
  (apple_production + orange_production + mango_production) * price_per_kg

/-- Proves that Mr. Sergio's total income is $90000 given the conditions -/
theorem sergio_income : total_income 400 50 = 90000 := by
  sorry

end NUMINAMATH_CALUDE_sergio_income_l2580_258097


namespace NUMINAMATH_CALUDE_charlotte_overall_score_l2580_258019

/-- Charlotte's test scores -/
def charlotte_scores : Fin 3 → ℚ
  | 0 => 60 / 100
  | 1 => 75 / 100
  | 2 => 85 / 100

/-- Number of problems in each test -/
def test_problems : Fin 3 → ℕ
  | 0 => 15
  | 1 => 20
  | 2 => 25

/-- Total number of problems in the combined test -/
def total_problems : ℕ := 60

/-- Charlotte's overall score on the combined test -/
def overall_score : ℚ := (charlotte_scores 0 * test_problems 0 +
                          charlotte_scores 1 * test_problems 1 +
                          charlotte_scores 2 * test_problems 2) / total_problems

theorem charlotte_overall_score :
  overall_score = 75 / 100 := by sorry

end NUMINAMATH_CALUDE_charlotte_overall_score_l2580_258019


namespace NUMINAMATH_CALUDE_smallest_number_l2580_258069

theorem smallest_number (a b c d : ℝ) : 
  a = -2 → b = 4 → c = -5 → d = 1 → 
  (c < -3 ∧ a > -3 ∧ b > -3 ∧ d > -3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l2580_258069


namespace NUMINAMATH_CALUDE_min_attendees_with_both_l2580_258070

theorem min_attendees_with_both (n : ℕ) (h1 : n > 0) : ∃ x : ℕ,
  x ≥ 1 ∧
  x ≤ n ∧
  x ≤ n / 3 ∧
  x ≤ n / 2 ∧
  ∀ y : ℕ, (y < x → ¬(y ≤ n / 3 ∧ y ≤ n / 2)) :=
by
  sorry

#check min_attendees_with_both

end NUMINAMATH_CALUDE_min_attendees_with_both_l2580_258070


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2580_258064

theorem polynomial_remainder_theorem (c d : ℚ) : 
  let g : ℚ → ℚ := λ x ↦ c * x^3 - 4 * x^2 + d * x - 7
  (g 2 = -7) ∧ (g (-1) = -20) → c = -1/3 ∧ d = 28/3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2580_258064
