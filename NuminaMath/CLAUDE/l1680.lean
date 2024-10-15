import Mathlib

namespace NUMINAMATH_CALUDE_cubic_meter_to_cubic_cm_l1680_168022

/-- Conversion factor from meters to centimeters -/
def meters_to_cm : ℝ := 100

/-- The number of cubic centimeters in one cubic meter -/
def cubic_cm_in_cubic_meter : ℝ := (meters_to_cm) ^ 3

theorem cubic_meter_to_cubic_cm : 
  cubic_cm_in_cubic_meter = 1000000 :=
sorry

end NUMINAMATH_CALUDE_cubic_meter_to_cubic_cm_l1680_168022


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l1680_168038

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | x^2 - 2*x - 3 > 0}

def N : Set ℝ := {x | |x| ≤ 3}

theorem complement_M_intersect_N :
  (Set.compl M ∩ N) = Set.Icc (-1 : ℝ) 3 := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l1680_168038


namespace NUMINAMATH_CALUDE_ratio_of_percentages_l1680_168000

theorem ratio_of_percentages (P Q R M N : ℝ) 
  (hM : M = 0.4 * Q)
  (hQ : Q = 0.25 * P)
  (hN : N = 0.4 * R)
  (hR : R = 0.75 * P)
  (hP : P ≠ 0) :
  M / N = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_percentages_l1680_168000


namespace NUMINAMATH_CALUDE_child_share_calculation_l1680_168045

theorem child_share_calculation (total_amount : ℚ) (ratio_a ratio_b ratio_c : ℕ) : 
  total_amount = 4500 →
  ratio_a = 2 →
  ratio_b = 3 →
  ratio_c = 4 →
  (ratio_b : ℚ) / (ratio_a + ratio_b + ratio_c : ℚ) * total_amount = 1500 := by
sorry

end NUMINAMATH_CALUDE_child_share_calculation_l1680_168045


namespace NUMINAMATH_CALUDE_min_second_longest_side_unit_area_triangle_l1680_168099

theorem min_second_longest_side_unit_area_triangle (a b c : ℝ) (h_area : (1/2) * a * b * Real.sin γ = 1) (h_order : a ≤ b ∧ b ≤ c) (γ : ℝ) :
  b ≥ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_second_longest_side_unit_area_triangle_l1680_168099


namespace NUMINAMATH_CALUDE_hammer_wrench_problem_l1680_168055

theorem hammer_wrench_problem (H W : ℝ) (x : ℕ) 
  (h1 : 2 * H + 2 * W = (1 / 3) * (x * H + 5 * W))
  (h2 : W = 2 * H) :
  x = 8 := by
  sorry

end NUMINAMATH_CALUDE_hammer_wrench_problem_l1680_168055


namespace NUMINAMATH_CALUDE_complex_product_pure_imaginary_l1680_168075

/-- A complex number is pure imaginary if its real part is zero -/
def isPureImaginary (z : ℂ) : Prop := z.re = 0

/-- The problem statement -/
theorem complex_product_pure_imaginary (b : ℝ) :
  isPureImaginary ((1 + b * Complex.I) * (2 - Complex.I)) → b = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_pure_imaginary_l1680_168075


namespace NUMINAMATH_CALUDE_no_integer_solution_l1680_168064

theorem no_integer_solution : ¬∃ (a b c d : ℤ), 
  (a * 19^3 + b * 19^2 + c * 19 + d = 1) ∧ 
  (a * 62^3 + b * 62^2 + c * 62 + d = 2) := by
sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1680_168064


namespace NUMINAMATH_CALUDE_tan_neg_3780_degrees_l1680_168079

theorem tan_neg_3780_degrees : Real.tan ((-3780 : ℝ) * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_neg_3780_degrees_l1680_168079


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l1680_168004

theorem fractional_equation_solution :
  ∃ x : ℚ, x ≠ 0 ∧ x ≠ -3 ∧ (1 / x = 6 / (x + 3)) ∧ x = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l1680_168004


namespace NUMINAMATH_CALUDE_function_inequality_solution_l1680_168056

theorem function_inequality_solution (f : ℕ → ℝ) 
  (h1 : ∀ n ≥ 2, n * f n - (n - 1) * f (n + 1) ≥ 1)
  (h2 : f 2 = 3) :
  ∃ g : ℕ → ℝ, 
    (∀ n ≥ 2, f n = 1 + (n - 1) * g n) ∧ 
    (∀ n ≥ 2, g n ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_solution_l1680_168056


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1680_168060

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ a, a > 2 → a * (a - 2) > 0) ∧ 
  (∃ a, a * (a - 2) > 0 ∧ ¬(a > 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1680_168060


namespace NUMINAMATH_CALUDE_f_monotone_increasing_on_negative_l1680_168037

-- Define the function
def f (x : ℝ) : ℝ := -x^2

-- State the theorem
theorem f_monotone_increasing_on_negative : 
  MonotoneOn f (Set.Iic 0) :=
sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_on_negative_l1680_168037


namespace NUMINAMATH_CALUDE_count_sets_with_seven_l1680_168059

def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

theorem count_sets_with_seven :
  (Finset.filter (fun s : Finset ℕ => 
    s.card = 3 ∧ 
    (∀ x ∈ s, x ∈ S) ∧ 
    (s.sum id = 21) ∧ 
    (7 ∈ s))
  (Finset.powerset S)).card = 5 :=
sorry

end NUMINAMATH_CALUDE_count_sets_with_seven_l1680_168059


namespace NUMINAMATH_CALUDE_chord_length_circle_line_l1680_168077

/-- The length of the chord intercepted by a circle on a line -/
theorem chord_length_circle_line (t : ℝ → ℝ × ℝ) (c : ℝ × ℝ → Prop) :
  (∀ r, t r = (-2 + r, 1 - r)) →  -- Line definition
  (∀ p, c p ↔ (p.1 - 3)^2 + (p.2 + 1)^2 = 25) →  -- Circle definition
  ∃ t₁ t₂, t₁ ≠ t₂ ∧ c (t t₁) ∧ c (t t₂) ∧ 
    Real.sqrt ((t t₁).1 - (t t₂).1)^2 + ((t t₁).2 - (t t₂).2)^2 = Real.sqrt 82 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_circle_line_l1680_168077


namespace NUMINAMATH_CALUDE_cauchy_functional_equation_verify_solution_l1680_168068

/-- A function satisfying the additive Cauchy equation -/
def is_additive (f : ℕ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x + f y

/-- A function satisfying f(nk) = n f(k) for all n, k ∈ ℕ -/
def satisfies_property (f : ℕ → ℝ) : Prop :=
  ∀ n k, f (n * k) = n * f k

theorem cauchy_functional_equation (f : ℕ → ℝ) 
  (h_additive : is_additive f) (h_property : satisfies_property f) :
  ∃ a : ℝ, ∀ n : ℕ, f n = a * n := by sorry

theorem verify_solution (a : ℝ) :
  let f : ℕ → ℝ := λ n ↦ a * n
  is_additive f ∧ satisfies_property f := by sorry

end NUMINAMATH_CALUDE_cauchy_functional_equation_verify_solution_l1680_168068


namespace NUMINAMATH_CALUDE_opposite_of_negative_half_l1680_168072

theorem opposite_of_negative_half : -(-(1/2 : ℚ)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_half_l1680_168072


namespace NUMINAMATH_CALUDE_barry_larry_reach_l1680_168031

/-- The maximum height Barry and Larry can reach when Barry stands on Larry's shoulders -/
def max_reach (barry_reach : ℝ) (larry_height : ℝ) (larry_shoulder_ratio : ℝ) : ℝ :=
  barry_reach + larry_height * larry_shoulder_ratio

/-- Theorem stating the maximum reach of Barry and Larry -/
theorem barry_larry_reach :
  let barry_reach : ℝ := 5
  let larry_height : ℝ := 5
  let larry_shoulder_ratio : ℝ := 0.8
  max_reach barry_reach larry_height larry_shoulder_ratio = 9 := by
  sorry

end NUMINAMATH_CALUDE_barry_larry_reach_l1680_168031


namespace NUMINAMATH_CALUDE_complex_number_opposite_parts_l1680_168012

theorem complex_number_opposite_parts (b : ℝ) : 
  let z : ℂ := (2 - b * Complex.I) / (1 + 2 * Complex.I)
  (z.re = -z.im) → b = -2/3 := by
sorry

end NUMINAMATH_CALUDE_complex_number_opposite_parts_l1680_168012


namespace NUMINAMATH_CALUDE_relay_race_first_leg_time_l1680_168044

/-- Represents a relay race with two runners -/
structure RelayRace where
  y_time : ℝ  -- Time taken by runner y for the first leg
  z_time : ℝ  -- Time taken by runner z for the second leg

/-- Theorem: In a relay race where the second runner takes 26 seconds and the average time per leg is 42 seconds, the first runner takes 58 seconds. -/
theorem relay_race_first_leg_time (race : RelayRace) 
  (h1 : race.z_time = 26)
  (h2 : (race.y_time + race.z_time) / 2 = 42) : 
  race.y_time = 58 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_first_leg_time_l1680_168044


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1680_168041

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- Theorem: In an arithmetic sequence where a_3 + a_7 = 38, the sum a_2 + a_4 + a_6 + a_8 = 76 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_sum : a 3 + a 7 = 38) : 
  a 2 + a 4 + a 6 + a 8 = 76 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1680_168041


namespace NUMINAMATH_CALUDE_third_root_of_polynomial_l1680_168035

theorem third_root_of_polynomial (a b : ℚ) :
  (∀ x : ℚ, a * x^3 + 2*(a + b) * x^2 + (b - 2*a) * x + (10 - a) = 0 ↔ x = -1 ∨ x = 4 ∨ x = 61/35) :=
by sorry

end NUMINAMATH_CALUDE_third_root_of_polynomial_l1680_168035


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l1680_168065

theorem binomial_expansion_example : 97^3 + 3*(97^2) + 3*97 + 1 = 940792 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l1680_168065


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l1680_168074

theorem sqrt_expression_equality : 
  Real.sqrt 8 - 2 * Real.sqrt (1/2) + (Real.sqrt 27 + 2 * Real.sqrt 6) / Real.sqrt 3 = 3 * Real.sqrt 2 + 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l1680_168074


namespace NUMINAMATH_CALUDE_two_point_distribution_a_value_l1680_168009

/-- A random variable following a two-point distribution -/
structure TwoPointDistribution where
  a : ℝ
  prob_zero : ℝ := 2 * a^2
  prob_one : ℝ := a

/-- The sum of probabilities in a two-point distribution equals 1 -/
axiom prob_sum_eq_one (X : TwoPointDistribution) : X.prob_zero + X.prob_one = 1

/-- Theorem: The value of 'a' in the two-point distribution is 1/2 -/
theorem two_point_distribution_a_value (X : TwoPointDistribution) : X.a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_two_point_distribution_a_value_l1680_168009


namespace NUMINAMATH_CALUDE_cakes_sold_l1680_168008

/-- Given the initial number of cakes, the remaining number of cakes,
    and the fact that some cakes were sold, prove that the number of cakes sold is 10. -/
theorem cakes_sold (initial_cakes remaining_cakes : ℕ) 
  (h1 : initial_cakes = 149)
  (h2 : remaining_cakes = 139)
  (h3 : remaining_cakes < initial_cakes) :
  initial_cakes - remaining_cakes = 10 := by
  sorry

#check cakes_sold

end NUMINAMATH_CALUDE_cakes_sold_l1680_168008


namespace NUMINAMATH_CALUDE_inverse_variation_cube_fourth_l1680_168025

/-- Given that x³ varies inversely with y⁴, and x = 2 when y = 4,
    prove that x³ = 1/2 when y = 8 -/
theorem inverse_variation_cube_fourth (k : ℝ) :
  (∀ x y : ℝ, x^3 * y^4 = k) →
  (2^3 * 4^4 = k) →
  ∃ x : ℝ, x^3 * 8^4 = k ∧ x^3 = (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_inverse_variation_cube_fourth_l1680_168025


namespace NUMINAMATH_CALUDE_three_digit_number_from_sum_l1680_168030

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  h_a : a < 10
  h_b : b < 10
  h_c : c < 10

/-- Calculates the sum of permutations of a three-digit number -/
def sumOfPermutations (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.b + n.c +
  100 * n.a + 10 * n.c + n.b +
  100 * n.b + 10 * n.a + n.c +
  100 * n.b + 10 * n.c + n.a +
  100 * n.c + 10 * n.a + n.b +
  100 * n.c + 10 * n.b + n.a

theorem three_digit_number_from_sum (N : Nat) (h_N : N = 3194) :
  ∃ (n : ThreeDigitNumber), sumOfPermutations n = N ∧ n.a = 3 ∧ n.b = 5 ∧ n.c = 8 := by
  sorry

#eval sumOfPermutations { a := 3, b := 5, c := 8, h_a := by norm_num, h_b := by norm_num, h_c := by norm_num }

end NUMINAMATH_CALUDE_three_digit_number_from_sum_l1680_168030


namespace NUMINAMATH_CALUDE_herd_division_l1680_168086

theorem herd_division (total : ℕ) (fourth_son : ℕ) : 
  (1 : ℚ) / 3 + (1 : ℚ) / 6 + (1 : ℚ) / 9 + (fourth_son : ℚ) / total = 1 →
  fourth_son = 11 →
  total = 54 := by
sorry

end NUMINAMATH_CALUDE_herd_division_l1680_168086


namespace NUMINAMATH_CALUDE_average_of_remaining_results_l1680_168050

theorem average_of_remaining_results (average_40 : ℝ) (average_all : ℝ) :
  average_40 = 30 →
  average_all = 34.285714285714285 →
  (70 * average_all - 40 * average_40) / 30 = 40 := by
sorry

end NUMINAMATH_CALUDE_average_of_remaining_results_l1680_168050


namespace NUMINAMATH_CALUDE_line_parallel_to_x_axis_l1680_168040

/-- A line through two points (x₁, y₁) and (x₂, y₂) is parallel to the x-axis if and only if y₁ = y₂ -/
def parallel_to_x_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  y₁ = y₂

/-- The problem statement -/
theorem line_parallel_to_x_axis (k : ℝ) :
  parallel_to_x_axis 3 (2*k + 1) 8 (4*k - 5) ↔ k = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_parallel_to_x_axis_l1680_168040


namespace NUMINAMATH_CALUDE_soda_cans_purchased_l1680_168085

/-- Given that S cans of soda can be purchased for Q quarters, and 1 dollar is worth 5 quarters due to a fee,
    the number of cans of soda that can be purchased for D dollars is (5 * D * S) / Q. -/
theorem soda_cans_purchased (S Q D : ℚ) (hS : S > 0) (hQ : Q > 0) (hD : D ≥ 0) :
  (S / Q) * (5 * D) = (5 * D * S) / Q :=
by sorry

end NUMINAMATH_CALUDE_soda_cans_purchased_l1680_168085


namespace NUMINAMATH_CALUDE_new_person_age_l1680_168067

theorem new_person_age (n : ℕ) (initial_avg : ℝ) (new_avg : ℝ) :
  n = 8 ∧ initial_avg = 14 ∧ new_avg = 16 →
  ∃ new_age : ℝ,
    new_age = n * new_avg + new_avg - n * initial_avg ∧
    new_age = 32 := by
  sorry

end NUMINAMATH_CALUDE_new_person_age_l1680_168067


namespace NUMINAMATH_CALUDE_dartboard_sector_angle_l1680_168095

theorem dartboard_sector_angle (total_angle : ℝ) (sector_prob : ℝ) : 
  total_angle = 360 → 
  sector_prob = 1/4 → 
  sector_prob * total_angle = 90 :=
by sorry

end NUMINAMATH_CALUDE_dartboard_sector_angle_l1680_168095


namespace NUMINAMATH_CALUDE_japanese_study_fraction_l1680_168092

theorem japanese_study_fraction (j s : ℝ) (x : ℝ) : 
  s = 3 * j →                           -- Senior class is 3 times the junior class
  ((1/3) * s + x * j) / (s + j) = 0.4375 →  -- 0.4375 fraction of all students study Japanese
  x = 3/4 :=                             -- Fraction of juniors studying Japanese
by
  sorry

end NUMINAMATH_CALUDE_japanese_study_fraction_l1680_168092


namespace NUMINAMATH_CALUDE_vector_dot_product_l1680_168097

theorem vector_dot_product (a b : ℝ × ℝ) (h1 : a = (1, -1)) (h2 : b = (-1, 2)) :
  (2 • a + b) • a = 1 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_l1680_168097


namespace NUMINAMATH_CALUDE_indefinite_stick_shortening_l1680_168081

theorem indefinite_stick_shortening :
  ∃ t : ℝ, t > 1 ∧ ∀ n : ℕ, t^(3-n) > t^(2-n) + t^(1-n) := by
  sorry

end NUMINAMATH_CALUDE_indefinite_stick_shortening_l1680_168081


namespace NUMINAMATH_CALUDE_adjacent_different_country_probability_l1680_168019

/-- Represents a country with delegates -/
structure Country where
  delegates : Nat
  deriving Repr

/-- Represents a seating arrangement -/
structure SeatingArrangement where
  total_seats : Nat
  countries : List Country
  deriving Repr

/-- Calculates the probability of each delegate sitting adjacent to at least one delegate from a different country -/
def probability_adjacent_different_country (arrangement : SeatingArrangement) : Rat :=
  sorry

/-- The specific seating arrangement from the problem -/
def problem_arrangement : SeatingArrangement :=
  { total_seats := 12
  , countries := List.replicate 4 { delegates := 2 }
  }

/-- Theorem stating the probability for the given seating arrangement -/
theorem adjacent_different_country_probability :
  probability_adjacent_different_country problem_arrangement = 4897683 / 9979200 :=
  sorry

end NUMINAMATH_CALUDE_adjacent_different_country_probability_l1680_168019


namespace NUMINAMATH_CALUDE_basketball_fall_certain_l1680_168090

-- Define the type for events
inductive Event
  | RainTomorrow
  | RollEvenDice
  | TVAdvertisement
  | BasketballFall

-- Define a predicate for certain events
def IsCertain (e : Event) : Prop :=
  match e with
  | Event.BasketballFall => True
  | _ => False

-- Define the law of gravity (simplified)
axiom law_of_gravity : ∀ (object : Type), object → object → Prop

-- Theorem statement
theorem basketball_fall_certain :
  ∀ (e : Event), IsCertain e ↔ e = Event.BasketballFall :=
sorry

end NUMINAMATH_CALUDE_basketball_fall_certain_l1680_168090


namespace NUMINAMATH_CALUDE_geometric_progression_floor_sum_l1680_168088

theorem geometric_progression_floor_sum (a b c k r : ℝ) : 
  a > 0 → b > 0 → c > 0 → k > 0 → r > 1 → 
  b = k * r → c = k * r^2 →
  ⌊(a + b) / c⌋ + ⌊(b + c) / a⌋ + ⌊(c + a) / b⌋ ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_floor_sum_l1680_168088


namespace NUMINAMATH_CALUDE_negation_existence_quadratic_inequality_l1680_168057

theorem negation_existence_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_existence_quadratic_inequality_l1680_168057


namespace NUMINAMATH_CALUDE_walking_ring_width_l1680_168039

theorem walking_ring_width (r₁ r₂ : ℝ) (h : 2 * π * r₁ - 2 * π * r₂ = 20 * π) : 
  r₁ - r₂ = 10 := by
sorry

end NUMINAMATH_CALUDE_walking_ring_width_l1680_168039


namespace NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l1680_168032

theorem arccos_one_over_sqrt_two (π : Real) :
  Real.arccos (1 / Real.sqrt 2) = π / 4 := by sorry

end NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l1680_168032


namespace NUMINAMATH_CALUDE_apartment_occupancy_theorem_l1680_168027

/-- Represents an apartment complex with identical buildings -/
structure ApartmentComplex where
  num_buildings : ℕ
  studio_per_building : ℕ
  two_person_per_building : ℕ
  four_person_per_building : ℕ
  occupancy_rate : ℚ

/-- Calculates the number of people living in the apartment complex at the given occupancy rate -/
def occupancy (complex : ApartmentComplex) : ℕ :=
  let max_per_building := 
    complex.studio_per_building + 
    2 * complex.two_person_per_building + 
    4 * complex.four_person_per_building
  let total_max := complex.num_buildings * max_per_building
  ⌊(total_max : ℚ) * complex.occupancy_rate⌋.toNat

theorem apartment_occupancy_theorem (complex : ApartmentComplex) 
  (h1 : complex.num_buildings = 4)
  (h2 : complex.studio_per_building = 10)
  (h3 : complex.two_person_per_building = 20)
  (h4 : complex.four_person_per_building = 5)
  (h5 : complex.occupancy_rate = 3/4) :
  occupancy complex = 210 := by
  sorry

end NUMINAMATH_CALUDE_apartment_occupancy_theorem_l1680_168027


namespace NUMINAMATH_CALUDE_max_value_sin_cos_l1680_168006

theorem max_value_sin_cos (θ : Real) (h : 0 < θ ∧ θ < π) :
  ∃ (max : Real), max = (4 * Real.sqrt 3) / 9 ∧
  ∀ (x : Real), 0 < x ∧ x < π →
    Real.sin (x / 2) * (1 + Real.cos x) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_sin_cos_l1680_168006


namespace NUMINAMATH_CALUDE_sheila_hourly_rate_l1680_168013

/-- Sheila's work schedule and earnings --/
structure WorkSchedule where
  monday_hours : ℕ
  tuesday_hours : ℕ
  wednesday_hours : ℕ
  thursday_hours : ℕ
  friday_hours : ℕ
  weekly_earnings : ℕ

/-- Calculates the hourly rate given a work schedule --/
def hourly_rate (schedule : WorkSchedule) : ℚ :=
  let total_hours := schedule.monday_hours + schedule.tuesday_hours + 
                     schedule.wednesday_hours + schedule.thursday_hours + 
                     schedule.friday_hours
  schedule.weekly_earnings / total_hours

/-- Theorem: Sheila's hourly rate is $8 --/
theorem sheila_hourly_rate :
  let sheila_schedule : WorkSchedule := {
    monday_hours := 8,
    tuesday_hours := 6,
    wednesday_hours := 8,
    thursday_hours := 6,
    friday_hours := 8,
    weekly_earnings := 288
  }
  hourly_rate sheila_schedule = 8 := by sorry

end NUMINAMATH_CALUDE_sheila_hourly_rate_l1680_168013


namespace NUMINAMATH_CALUDE_distance_calculation_l1680_168093

def point : ℝ × ℝ × ℝ := (2, 3, 4)
def line_point : ℝ × ℝ × ℝ := (5, 8, 6)
def line_direction : ℝ × ℝ × ℝ := (4, 3, -3)

def distance_to_line (p : ℝ × ℝ × ℝ) (l_point : ℝ × ℝ × ℝ) (l_dir : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_calculation :
  distance_to_line point line_point line_direction = Real.sqrt 10458 / 34 := by
  sorry

end NUMINAMATH_CALUDE_distance_calculation_l1680_168093


namespace NUMINAMATH_CALUDE_sum_of_xyz_l1680_168046

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y = 30) (h2 : x * z = 60) (h3 : y * z = 90) : 
  x + y + z = 11 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l1680_168046


namespace NUMINAMATH_CALUDE_function_inequality_l1680_168034

theorem function_inequality (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, 2 * a * x^2 - a * x > 3 - a) →
  a > 24/7 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l1680_168034


namespace NUMINAMATH_CALUDE_prob_even_sum_two_dice_l1680_168063

/-- Die with faces numbered 1 through 4 -/
def Die1 : Finset Nat := {1, 2, 3, 4}

/-- Die with faces numbered 1 through 8 -/
def Die2 : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8}

/-- The probability of getting an even sum when rolling two dice -/
def ProbEvenSum : ℚ := 1 / 2

/-- Theorem stating that the probability of getting an even sum when rolling
    two dice, one with faces 1-4 and another with faces 1-8, is equal to 1/2 -/
theorem prob_even_sum_two_dice :
  let outcomes := Die1.product Die2
  let even_sum := {p : Nat × Nat | (p.1 + p.2) % 2 = 0}
  (outcomes.filter (λ p => p ∈ even_sum)).card / outcomes.card = ProbEvenSum :=
sorry


end NUMINAMATH_CALUDE_prob_even_sum_two_dice_l1680_168063


namespace NUMINAMATH_CALUDE_quadrilateral_inequality_l1680_168017

-- Define the points
variable (A B C D E : EuclideanSpace ℝ (Fin 2))

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop :=
  sorry

-- Define the intersection of diagonals
def diagonals_intersect (A B C D E : EuclideanSpace ℝ (Fin 2)) : Prop :=
  sorry

-- Define the distance function
def distance (P Q : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  sorry

-- State the theorem
theorem quadrilateral_inequality 
  (h_convex : is_convex_quadrilateral A B C D)
  (h_intersect : diagonals_intersect A B C D E)
  (h_AB : distance A B = 1)
  (h_BC : distance B C = 1)
  (h_CD : distance C D = 1)
  (h_DE : distance D E = 1) :
  distance A D < 2 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_inequality_l1680_168017


namespace NUMINAMATH_CALUDE_right_triangle_area_and_perimeter_l1680_168091

-- Define the right triangle
def right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

-- Theorem statement
theorem right_triangle_area_and_perimeter :
  ∀ (a b c : ℝ),
  right_triangle a b c →
  a = 36 →
  b = 48 →
  (1/2 * a * b = 864) ∧ (a + b + c = 144) := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_area_and_perimeter_l1680_168091


namespace NUMINAMATH_CALUDE_prime_between_n_and_nfactorial_l1680_168021

theorem prime_between_n_and_nfactorial (n : ℕ) (h : n > 2) :
  ∃ p : ℕ, Prime p ∧ n < p ∧ p ≤ n! :=
by sorry

end NUMINAMATH_CALUDE_prime_between_n_and_nfactorial_l1680_168021


namespace NUMINAMATH_CALUDE_binary_101111011_equals_379_l1680_168084

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent. -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 101111011₂ (least significant bit first) -/
def binary_101111011 : List Bool := [true, true, false, true, true, true, true, false, true]

theorem binary_101111011_equals_379 :
  binary_to_decimal binary_101111011 = 379 := by
  sorry

end NUMINAMATH_CALUDE_binary_101111011_equals_379_l1680_168084


namespace NUMINAMATH_CALUDE_min_value_product_l1680_168014

theorem min_value_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (x + 1/x) * (y + 1/y) ≥ 33/4 := by
sorry

end NUMINAMATH_CALUDE_min_value_product_l1680_168014


namespace NUMINAMATH_CALUDE_sin_cos_difference_36_degrees_l1680_168071

theorem sin_cos_difference_36_degrees : 
  Real.sin (36 * π / 180) * Real.cos (36 * π / 180) - 
  Real.cos (36 * π / 180) * Real.sin (36 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_36_degrees_l1680_168071


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l1680_168011

theorem inequality_solution_sets (a : ℝ) :
  (∀ x, ax^2 + 5*x - 2 > 0 ↔ 1/2 < x ∧ x < 2) →
  (∀ x, ax^2 - 5*x + a^2 - 1 > 0 ↔ -3 < x ∧ x < 1/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l1680_168011


namespace NUMINAMATH_CALUDE_simplify_product_of_radicals_l1680_168073

theorem simplify_product_of_radicals (x : ℝ) (hx : x ≥ 0) :
  Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (32 * x) = 120 * x * Real.sqrt x :=
by sorry

end NUMINAMATH_CALUDE_simplify_product_of_radicals_l1680_168073


namespace NUMINAMATH_CALUDE_min_sin_minus_cos_half_angle_l1680_168016

theorem min_sin_minus_cos_half_angle :
  let f : ℝ → ℝ := λ A ↦ Real.sin (A / 2) - Real.cos (A / 2)
  ∃ (min : ℝ) (A : ℝ), 
    (∀ x, f x ≥ min) ∧ 
    (f A = min) ∧ 
    (min = -Real.sqrt 2) ∧ 
    (A = 7 * Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_min_sin_minus_cos_half_angle_l1680_168016


namespace NUMINAMATH_CALUDE_distribute_plumbers_count_l1680_168052

/-- The number of ways to distribute 4 plumbers to 3 residences -/
def distribute_plumbers : ℕ :=
  Nat.choose 4 2 * (3 * 2 * 1)

/-- The conditions of the problem -/
axiom plumbers : ℕ
axiom residences : ℕ
axiom plumbers_eq_four : plumbers = 4
axiom residences_eq_three : residences = 3
axiom all_plumbers_assigned : True
axiom one_residence_per_plumber : True
axiom all_residences_checked : True

/-- The theorem to be proved -/
theorem distribute_plumbers_count :
  distribute_plumbers = Nat.choose plumbers 2 * (residences * (residences - 1) * (residences - 2)) :=
sorry

end NUMINAMATH_CALUDE_distribute_plumbers_count_l1680_168052


namespace NUMINAMATH_CALUDE_smallest_positive_solution_l1680_168069

theorem smallest_positive_solution (x : ℕ) : x = 30 ↔ 
  (x > 0 ∧ 
   (51 * x + 15) % 35 = 5 ∧ 
   ∀ y : ℕ, y > 0 → (51 * y + 15) % 35 = 5 → x ≤ y) := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_l1680_168069


namespace NUMINAMATH_CALUDE_inscribed_semicircle_radius_l1680_168053

/-- An isosceles triangle with a semicircle inscribed along its base -/
structure IsoscelesTriangleWithSemicircle where
  /-- The base length of the isosceles triangle -/
  base : ℝ
  /-- The height of the isosceles triangle -/
  height : ℝ
  /-- The radius of the inscribed semicircle -/
  radius : ℝ
  /-- The base is positive -/
  base_pos : 0 < base
  /-- The height is positive -/
  height_pos : 0 < height
  /-- The radius is positive -/
  radius_pos : 0 < radius
  /-- The area of the triangle equals the area of the semicircle -/
  area_equality : (1/2) * base * height = π * radius^2

/-- The radius of the inscribed semicircle in the given isosceles triangle -/
theorem inscribed_semicircle_radius (t : IsoscelesTriangleWithSemicircle)
    (h1 : t.base = 24) (h2 : t.height = 18) : t.radius = 18 / π := by
  sorry

end NUMINAMATH_CALUDE_inscribed_semicircle_radius_l1680_168053


namespace NUMINAMATH_CALUDE_y_relationship_l1680_168096

-- Define the quadratic function
def f (x : ℝ) : ℝ := -(x + 2)^2 + 4

-- Define the points A, B, C
def A : ℝ × ℝ := (-3, f (-3))
def B : ℝ × ℝ := (0, f 0)
def C : ℝ × ℝ := (3, f 3)

-- Define y₁, y₂, y₃
def y₁ : ℝ := A.2
def y₂ : ℝ := B.2
def y₃ : ℝ := C.2

-- Theorem statement
theorem y_relationship : y₃ < y₂ ∧ y₂ < y₁ := by
  sorry

end NUMINAMATH_CALUDE_y_relationship_l1680_168096


namespace NUMINAMATH_CALUDE_sum_xyz_is_zero_l1680_168042

theorem sum_xyz_is_zero (x y z : ℝ) 
  (eq1 : x + y = 2*x + z)
  (eq2 : x - 2*y = 4*z)
  (eq3 : y = 6*z) : 
  x + y + z = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_xyz_is_zero_l1680_168042


namespace NUMINAMATH_CALUDE_crayons_left_l1680_168029

-- Define the initial number of crayons
def initial_crayons : ℕ := 62

-- Define the number of crayons eaten
def eaten_crayons : ℕ := 52

-- Theorem to prove
theorem crayons_left : initial_crayons - eaten_crayons = 10 := by
  sorry

end NUMINAMATH_CALUDE_crayons_left_l1680_168029


namespace NUMINAMATH_CALUDE_initial_pencils_theorem_l1680_168007

/-- The number of pencils initially in the drawer -/
def initial_pencils : ℕ := 34

/-- The number of pencils Dan took from the drawer -/
def pencils_taken : ℕ := 22

/-- The number of pencils remaining in the drawer -/
def pencils_remaining : ℕ := 12

/-- Theorem: The initial number of pencils equals the sum of pencils taken and pencils remaining -/
theorem initial_pencils_theorem : initial_pencils = pencils_taken + pencils_remaining := by
  sorry

end NUMINAMATH_CALUDE_initial_pencils_theorem_l1680_168007


namespace NUMINAMATH_CALUDE_number_problem_l1680_168082

theorem number_problem (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 14) : 
  (40/100) * N = 168 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1680_168082


namespace NUMINAMATH_CALUDE_min_visible_sum_is_90_l1680_168002

/-- Represents a cube with integers on each face -/
structure SmallCube where
  faces : Fin 6 → ℕ

/-- Represents the larger 3x3x3 cube -/
structure LargeCube where
  smallCubes : Fin 27 → SmallCube

/-- Calculates the sum of visible faces on the larger cube -/
def visibleSum (c : LargeCube) : ℕ := sorry

/-- The minimum possible sum of visible faces -/
def minVisibleSum : ℕ := 90

/-- Theorem stating that the minimum possible sum is 90 -/
theorem min_visible_sum_is_90 :
  ∀ c : LargeCube, visibleSum c ≥ minVisibleSum :=
sorry

end NUMINAMATH_CALUDE_min_visible_sum_is_90_l1680_168002


namespace NUMINAMATH_CALUDE_oil_leak_calculation_l1680_168010

theorem oil_leak_calculation (total_leaked : ℕ) (leaked_before : ℕ) 
  (h1 : total_leaked = 6206)
  (h2 : leaked_before = 2475) :
  total_leaked - leaked_before = 3731 :=
by sorry

end NUMINAMATH_CALUDE_oil_leak_calculation_l1680_168010


namespace NUMINAMATH_CALUDE_rectangular_prism_cutout_l1680_168036

theorem rectangular_prism_cutout (x y : ℕ) : 
  (15 * 5 * 4 - y * 5 * x = 120) → (x < 4 ∧ y < 15) → x + y = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_cutout_l1680_168036


namespace NUMINAMATH_CALUDE_rectangular_plot_width_l1680_168054

theorem rectangular_plot_width (length width area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 432 →
  width = 12 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_width_l1680_168054


namespace NUMINAMATH_CALUDE_xy_value_l1680_168033

theorem xy_value (x y : ℝ) (h : x^2 - 2*x*y + 2*y^2 + 6*y + 9 = 0) : x*y = 9 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1680_168033


namespace NUMINAMATH_CALUDE_total_fish_equation_l1680_168094

/-- The number of fish owned by four friends, given their relative quantities -/
def total_fish (x : ℝ) : ℝ :=
  let max_fish := x
  let sam_fish := 3.25 * max_fish
  let joe_fish := 9.5 * sam_fish
  let harry_fish := 5.5 * joe_fish
  max_fish + sam_fish + joe_fish + harry_fish

/-- Theorem stating that the total number of fish is 204.9375 times the number of fish Max has -/
theorem total_fish_equation (x : ℝ) : total_fish x = 204.9375 * x := by
  sorry

end NUMINAMATH_CALUDE_total_fish_equation_l1680_168094


namespace NUMINAMATH_CALUDE_rational_coefficient_sum_for_cube_root_two_plus_x_fifth_power_l1680_168058

def binomial_coefficient (n k : ℕ) : ℕ := (Nat.choose n k)

def rational_coefficient_sum (n : ℕ) : ℕ :=
  2 * (binomial_coefficient n 2) + (binomial_coefficient n n)

theorem rational_coefficient_sum_for_cube_root_two_plus_x_fifth_power :
  rational_coefficient_sum 5 = 21 := by sorry

end NUMINAMATH_CALUDE_rational_coefficient_sum_for_cube_root_two_plus_x_fifth_power_l1680_168058


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1680_168083

/-- A quadratic function symmetric about x = 1 and passing through the origin -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

/-- The function is symmetric about x = 1 -/
def symmetric_about_one (a b : ℝ) : Prop := ∀ x, f a b (1 + x) = f a b (1 - x)

/-- The function passes through the origin -/
def passes_through_origin (a b : ℝ) : Prop := f a b 0 = 0

theorem quadratic_function_properties (a b : ℝ) 
  (h1 : symmetric_about_one a b) (h2 : passes_through_origin a b) :
  (∀ x, f a b x = x^2 - 2*x) ∧ 
  Set.Icc (-1) 3 = Set.range (fun x => f a b x) ∩ Set.Ioo 0 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1680_168083


namespace NUMINAMATH_CALUDE_fence_length_l1680_168020

/-- For a rectangular yard with one side of 40 feet and an area of 480 square feet,
    the sum of the lengths of the other three sides is 64 feet. -/
theorem fence_length (length width : ℝ) : 
  width = 40 → 
  length * width = 480 → 
  2 * length + width = 64 := by
sorry

end NUMINAMATH_CALUDE_fence_length_l1680_168020


namespace NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l1680_168051

theorem consecutive_negative_integers_sum (n : ℤ) : 
  n < 0 ∧ n * (n + 1) = 2720 → n + (n + 1) = -105 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l1680_168051


namespace NUMINAMATH_CALUDE_trapezoid_bases_l1680_168028

theorem trapezoid_bases (d : ℝ) (l : ℝ) (h : d = 15 ∧ l = 17) :
  ∃ (b₁ b₂ : ℝ),
    b₁ = 9 ∧
    b₂ = 25 ∧
    b₁ + b₂ = 2 * l ∧
    b₂ - b₁ = 2 * Real.sqrt (l^2 - d^2) :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_bases_l1680_168028


namespace NUMINAMATH_CALUDE_focus_coordinates_l1680_168070

/-- The parabola defined by the equation y = (1/8)x^2 -/
def parabola (x y : ℝ) : Prop := y = (1/8) * x^2

/-- The focus of a parabola -/
structure Focus where
  x : ℝ
  y : ℝ

/-- The focus of the parabola y = (1/8)x^2 -/
def focus_of_parabola : Focus := { x := 0, y := 2 }

/-- Theorem: The focus of the parabola y = (1/8)x^2 is (0, 2) -/
theorem focus_coordinates :
  focus_of_parabola.x = 0 ∧ focus_of_parabola.y = 2 :=
sorry

end NUMINAMATH_CALUDE_focus_coordinates_l1680_168070


namespace NUMINAMATH_CALUDE_base7_divisible_by_13_l1680_168062

/-- Converts a base-7 number of the form 3dd6₇ to base 10 --/
def base7ToBase10 (d : Nat) : Nat :=
  3 * 7^3 + d * 7^2 + d * 7 + 6

/-- Checks if a number is divisible by 13 --/
def isDivisibleBy13 (n : Nat) : Prop :=
  n % 13 = 0

/-- A base-7 digit is between 0 and 6 inclusive --/
def isBase7Digit (d : Nat) : Prop :=
  d ≤ 6

theorem base7_divisible_by_13 :
  ∃ (d : Nat), isBase7Digit d ∧ isDivisibleBy13 (base7ToBase10 d) ∧ d = 5 := by
  sorry

end NUMINAMATH_CALUDE_base7_divisible_by_13_l1680_168062


namespace NUMINAMATH_CALUDE_ellipse_k_range_l1680_168098

/-- The curve equation --/
def curve_equation (x y k : ℝ) : Prop :=
  x^2 / (1 - k) + y^2 / (1 + k) = 1

/-- The curve represents an ellipse --/
def is_ellipse (k : ℝ) : Prop :=
  1 - k > 0 ∧ 1 + k > 0 ∧ 1 - k ≠ 1 + k

/-- The range of k for which the curve represents an ellipse --/
theorem ellipse_k_range :
  ∀ k : ℝ, is_ellipse k ↔ ((-1 < k ∧ k < 0) ∨ (0 < k ∧ k < 1)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l1680_168098


namespace NUMINAMATH_CALUDE_factor_81_minus_27x_cubed_l1680_168066

theorem factor_81_minus_27x_cubed (x : ℝ) : 81 - 27 * x^3 = 27 * (3 - x) * (9 + 3*x + x^2) := by
  sorry

end NUMINAMATH_CALUDE_factor_81_minus_27x_cubed_l1680_168066


namespace NUMINAMATH_CALUDE_flower_percentage_l1680_168087

theorem flower_percentage (total_flowers : ℕ) (yellow_flowers : ℕ) (purple_increase : ℚ) :
  total_flowers = 35 →
  yellow_flowers = 10 →
  purple_increase = 80 / 100 →
  let purple_flowers := yellow_flowers + (purple_increase * yellow_flowers).floor
  let green_flowers := total_flowers - yellow_flowers - purple_flowers
  let yellow_and_purple := yellow_flowers + purple_flowers
  (green_flowers : ℚ) / yellow_and_purple * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_flower_percentage_l1680_168087


namespace NUMINAMATH_CALUDE_disk_space_calculation_l1680_168043

/-- The total space on Mike's disk drive in GB. -/
def total_space : ℕ := 28

/-- The space taken by Mike's files in GB. -/
def file_space : ℕ := 26

/-- The space left over after backing up Mike's files in GB. -/
def space_left : ℕ := 2

/-- Theorem stating that the total space on Mike's disk drive is equal to
    the sum of the space taken by his files and the space left over. -/
theorem disk_space_calculation :
  total_space = file_space + space_left := by sorry

end NUMINAMATH_CALUDE_disk_space_calculation_l1680_168043


namespace NUMINAMATH_CALUDE_polynomial_value_at_2_l1680_168089

-- Define the polynomial function
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 3*x + 8

-- Theorem statement
theorem polynomial_value_at_2 : f 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_2_l1680_168089


namespace NUMINAMATH_CALUDE_hockey_league_teams_l1680_168024

/-- The number of teams in a hockey league -/
def num_teams : ℕ := 17

/-- The number of times each team faces every other team -/
def games_per_pair : ℕ := 10

/-- The total number of games played in the season -/
def total_games : ℕ := 1360

/-- Theorem stating that the number of teams is correct given the conditions -/
theorem hockey_league_teams :
  (num_teams * (num_teams - 1) * games_per_pair) / 2 = total_games :=
sorry

end NUMINAMATH_CALUDE_hockey_league_teams_l1680_168024


namespace NUMINAMATH_CALUDE_min_radius_value_l1680_168023

/-- A circle in the Cartesian plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point on the circle satisfying the given condition -/
structure PointOnCircle (c : Circle) where
  point : ℝ × ℝ
  on_circle : (point.1 - c.center.1)^2 + (point.2 - c.center.2)^2 = c.radius^2
  condition : point.2^2 ≥ 4 * point.1

/-- The theorem stating the minimum value of r -/
theorem min_radius_value (c : Circle) (p : PointOnCircle c) :
  c.center.1 = c.radius + 1 ∧ c.center.2 = 0 → c.radius ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_radius_value_l1680_168023


namespace NUMINAMATH_CALUDE_parabola_vertex_l1680_168076

/-- The parabola defined by y = -x^2 + 2x + 3 has its vertex at the point (1, 4). -/
theorem parabola_vertex (x y : ℝ) : 
  y = -x^2 + 2*x + 3 → (1, 4) = (x, y) ∨ ∃ t : ℝ, y < -t^2 + 2*t + 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1680_168076


namespace NUMINAMATH_CALUDE_trees_after_typhoon_l1680_168078

theorem trees_after_typhoon (initial_trees : ℕ) (dead_trees : ℕ) : 
  initial_trees = 20 → dead_trees = 16 → initial_trees - dead_trees = 4 := by
  sorry

end NUMINAMATH_CALUDE_trees_after_typhoon_l1680_168078


namespace NUMINAMATH_CALUDE_mustang_length_proof_l1680_168018

theorem mustang_length_proof (smallest_model : ℝ) (mid_size_model : ℝ) (full_size : ℝ)
  (h1 : smallest_model = 12)
  (h2 : smallest_model = mid_size_model / 2)
  (h3 : mid_size_model = full_size / 10) :
  full_size = 240 := by
  sorry

end NUMINAMATH_CALUDE_mustang_length_proof_l1680_168018


namespace NUMINAMATH_CALUDE_hiker_supply_per_mile_l1680_168049

/-- A hiker's supply calculation problem -/
theorem hiker_supply_per_mile
  (hiking_rate : ℝ)
  (hours_per_day : ℝ)
  (days : ℝ)
  (first_pack_weight : ℝ)
  (resupply_percentage : ℝ)
  (h1 : hiking_rate = 2.5)
  (h2 : hours_per_day = 8)
  (h3 : days = 5)
  (h4 : first_pack_weight = 40)
  (h5 : resupply_percentage = 0.25)
  : (first_pack_weight + first_pack_weight * resupply_percentage) / (hiking_rate * hours_per_day * days) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_hiker_supply_per_mile_l1680_168049


namespace NUMINAMATH_CALUDE_jordan_rectangle_width_l1680_168047

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem jordan_rectangle_width : 
  ∀ (carol_rect jordan_rect : Rectangle),
    carol_rect.length = 5 →
    carol_rect.width = 24 →
    jordan_rect.length = 12 →
    area carol_rect = area jordan_rect →
    jordan_rect.width = 10 := by
  sorry

end NUMINAMATH_CALUDE_jordan_rectangle_width_l1680_168047


namespace NUMINAMATH_CALUDE_books_not_sold_l1680_168001

theorem books_not_sold (initial_stock : ℕ) (monday_sales tuesday_sales wednesday_sales thursday_sales friday_sales : ℕ) :
  initial_stock = 800 →
  monday_sales = 60 →
  tuesday_sales = 10 →
  wednesday_sales = 20 →
  thursday_sales = 44 →
  friday_sales = 66 →
  initial_stock - (monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales) = 600 := by
  sorry

end NUMINAMATH_CALUDE_books_not_sold_l1680_168001


namespace NUMINAMATH_CALUDE_perpendicular_to_plane_implies_parallel_l1680_168015

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_to_plane_implies_parallel 
  (m n : Line) (α : Plane) 
  (h1 : perpendicular m α) 
  (h2 : perpendicular n α) 
  (h3 : m ≠ n) : 
  parallel m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_plane_implies_parallel_l1680_168015


namespace NUMINAMATH_CALUDE_product_not_always_greater_than_factors_l1680_168026

theorem product_not_always_greater_than_factors : ∃ (a b : ℝ), a * b ≤ a ∨ a * b ≤ b := by
  sorry

end NUMINAMATH_CALUDE_product_not_always_greater_than_factors_l1680_168026


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1680_168061

theorem inequality_equivalence (x : ℝ) : 
  (x - 2) / (x - 4) ≤ 3 ↔ 4 < x ∧ x ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1680_168061


namespace NUMINAMATH_CALUDE_jasons_quarters_l1680_168005

theorem jasons_quarters (initial final given : ℕ) 
  (h1 : initial = 49)
  (h2 : final = 74)
  (h3 : final = initial + given) :
  given = 25 := by
  sorry

end NUMINAMATH_CALUDE_jasons_quarters_l1680_168005


namespace NUMINAMATH_CALUDE_fraction_inequality_l1680_168048

theorem fraction_inequality (x : ℝ) : 
  -4 ≤ (x^2 - 2*x - 3) / (2*x^2 + 2*x + 1) ∧ (x^2 - 2*x - 3) / (2*x^2 + 2*x + 1) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1680_168048


namespace NUMINAMATH_CALUDE_smallest_x_for_inequality_l1680_168003

theorem smallest_x_for_inequality :
  ∃ (x : ℝ), x = 49 ∧
  (∀ (a : ℝ), a ≥ 0 → a ≥ 14 * Real.sqrt a - x) ∧
  (∀ (y : ℝ), y < x → ∃ (a : ℝ), a ≥ 0 ∧ a < 14 * Real.sqrt a - y) :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_for_inequality_l1680_168003


namespace NUMINAMATH_CALUDE_ones_digit_of_large_power_l1680_168080

theorem ones_digit_of_large_power (n : ℕ) : 
  (35^(35*(17^17)) : ℕ) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_large_power_l1680_168080
