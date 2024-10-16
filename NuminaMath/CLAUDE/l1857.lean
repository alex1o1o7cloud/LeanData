import Mathlib

namespace NUMINAMATH_CALUDE_pet_shop_dogs_l1857_185720

theorem pet_shop_dogs (ratio_dogs : ℕ) (ratio_cats : ℕ) (ratio_bunnies : ℕ) 
  (total_dogs_bunnies : ℕ) : 
  ratio_dogs = 3 → 
  ratio_cats = 5 → 
  ratio_bunnies = 9 → 
  total_dogs_bunnies = 204 → 
  ∃ (x : ℕ), x * (ratio_dogs + ratio_bunnies) = total_dogs_bunnies ∧ 
             x * ratio_dogs = 51 := by
  sorry

end NUMINAMATH_CALUDE_pet_shop_dogs_l1857_185720


namespace NUMINAMATH_CALUDE_second_polygon_sides_l1857_185733

theorem second_polygon_sides (perimeter : ℝ) (sides_first : ℕ) (length_ratio : ℝ) (sides_second : ℕ) : 
  perimeter > 0 →
  sides_first = 50 →
  length_ratio = 3 →
  (perimeter = sides_first * (length_ratio * (perimeter / (sides_second * length_ratio)))) →
  (perimeter = sides_second * (perimeter / (sides_second * length_ratio))) →
  sides_second = 150 := by
sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l1857_185733


namespace NUMINAMATH_CALUDE_function_composition_constant_l1857_185778

theorem function_composition_constant (b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 5 * x + b
  let g : ℝ → ℝ := λ x ↦ b * x + 3
  (∀ x, f (g x) = 15 * x + 18) :=
by
  sorry

end NUMINAMATH_CALUDE_function_composition_constant_l1857_185778


namespace NUMINAMATH_CALUDE_sum_of_special_numbers_l1857_185756

-- Define the smallest odd prime number
def smallest_odd_prime : ℕ := 3

-- Define the largest integer less than 150 with exactly three positive divisors
def largest_three_divisor_under_150 : ℕ := 121

-- Theorem statement
theorem sum_of_special_numbers : 
  smallest_odd_prime + largest_three_divisor_under_150 = 124 := by sorry

end NUMINAMATH_CALUDE_sum_of_special_numbers_l1857_185756


namespace NUMINAMATH_CALUDE_swallow_weight_ratio_l1857_185768

/-- The weight an American swallow can carry -/
def american_weight : ℝ := 5

/-- The total number of swallows in the flock -/
def total_swallows : ℕ := 90

/-- The ratio of American swallows to European swallows -/
def american_to_european_ratio : ℕ := 2

/-- The maximum combined weight the flock can carry -/
def total_weight : ℝ := 600

/-- The weight a European swallow can carry -/
def european_weight : ℝ := 10

theorem swallow_weight_ratio : 
  european_weight / american_weight = 2 := by sorry

end NUMINAMATH_CALUDE_swallow_weight_ratio_l1857_185768


namespace NUMINAMATH_CALUDE_football_cost_proof_l1857_185726

def shorts_cost : ℝ := 2.40
def shoes_cost : ℝ := 11.85
def zachary_has : ℝ := 10
def zachary_needs : ℝ := 8

def total_cost : ℝ := zachary_has + zachary_needs

def football_cost : ℝ := total_cost - shorts_cost - shoes_cost

theorem football_cost_proof : football_cost = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_football_cost_proof_l1857_185726


namespace NUMINAMATH_CALUDE_clock_angle_at_15_40_clock_angle_at_15_40_is_130_l1857_185722

/-- The angle between clock hands at 15:40 --/
theorem clock_angle_at_15_40 : ℝ :=
  let minutes_past_hour : ℝ := 40
  let hours_past_12 : ℝ := 3
  let minutes_per_hour : ℝ := 60
  let degrees_per_circle : ℝ := 360
  let hours_per_revolution : ℝ := 12

  let minute_hand_angle : ℝ := (minutes_past_hour / minutes_per_hour) * degrees_per_circle
  let hour_hand_angle : ℝ := (hours_past_12 / hours_per_revolution +
                              minutes_past_hour / (minutes_per_hour * hours_per_revolution)) *
                             degrees_per_circle

  let angle_between : ℝ := |minute_hand_angle - hour_hand_angle|

  130 -- The actual proof is omitted

theorem clock_angle_at_15_40_is_130 : clock_angle_at_15_40 = 130 := by
  sorry -- Proof is omitted

end NUMINAMATH_CALUDE_clock_angle_at_15_40_clock_angle_at_15_40_is_130_l1857_185722


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1857_185779

def A : Set ℝ := {x | x - 1 > 0}
def B : Set ℝ := {x | x < 0}
def C : Set ℝ := {x | x * (x - 2) > 0}

theorem necessary_but_not_sufficient :
  (∀ x, x ∈ C → x ∈ A ∪ B) ∧
  (∃ x, x ∈ A ∪ B ∧ x ∉ C) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1857_185779


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l1857_185728

theorem polynomial_identity_sum_of_squares (p q r s t u v : ℤ) :
  (∀ x : ℝ, 1728 * x^4 + 64 = (p * x^3 + q * x^2 + r * x + s) * (t * x + u) + v) →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 + v^2 = 416 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l1857_185728


namespace NUMINAMATH_CALUDE_expected_defective_meters_l1857_185701

/-- Proves that given a rejection rate of 1.5% and a sample size of 10,000 meters,
    the expected number of defective meters is 150. -/
theorem expected_defective_meters
  (rejection_rate : ℝ)
  (sample_size : ℕ)
  (h1 : rejection_rate = 0.015)
  (h2 : sample_size = 10000) :
  ↑sample_size * rejection_rate = 150 := by
  sorry

end NUMINAMATH_CALUDE_expected_defective_meters_l1857_185701


namespace NUMINAMATH_CALUDE_square_area_with_five_equal_rectangles_l1857_185735

theorem square_area_with_five_equal_rectangles (w : ℝ) (h : w = 5) :
  ∃ (s : ℝ), s > 0 ∧ s * s = 400 ∧
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
    2 * x * y = 3 * w * y ∧
    2 * x + w = s ∧
    5 * (2 * x * y) = s * s :=
by
  sorry

#check square_area_with_five_equal_rectangles

end NUMINAMATH_CALUDE_square_area_with_five_equal_rectangles_l1857_185735


namespace NUMINAMATH_CALUDE_number_problem_l1857_185705

theorem number_problem (x : ℝ) : (4/5 * x) + 16 = 0.9 * 40 → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1857_185705


namespace NUMINAMATH_CALUDE_sin_cos_relation_l1857_185773

theorem sin_cos_relation (θ : Real) (h1 : Real.sin θ + Real.cos θ = 1/2) 
  (h2 : π/2 < θ ∧ θ < π) : Real.cos θ - Real.sin θ = -Real.sqrt 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_relation_l1857_185773


namespace NUMINAMATH_CALUDE_power_division_equality_l1857_185764

theorem power_division_equality : (3 : ℕ)^12 / (27 : ℕ)^2 = 729 := by sorry

end NUMINAMATH_CALUDE_power_division_equality_l1857_185764


namespace NUMINAMATH_CALUDE_obtuse_triangle_a_range_a_range_implies_obtuse_triangle_l1857_185743

-- Define a triangle with sides a, a+1, and a+2
def triangle (a : ℝ) : Prop :=
  ∃ (A B C : ℝ × ℝ), 
    dist A B = a ∧ 
    dist B C = a + 1 ∧ 
    dist A C = a + 2

-- Define an obtuse triangle
def obtuse_triangle (a : ℝ) : Prop :=
  triangle a ∧ 
  ∃ (A B C : ℝ × ℝ), 
    dist A B = a ∧ 
    dist B C = a + 1 ∧ 
    dist A C = a + 2 ∧
    (dist A B)^2 + (dist B C)^2 < (dist A C)^2

-- Theorem stating the range of a for an obtuse triangle with sides a, a+1, and a+2
theorem obtuse_triangle_a_range : 
  ∀ a : ℝ, obtuse_triangle a → 1 < a ∧ a < 3 :=
by sorry

-- Theorem stating that if 1 < a < 3, then we can form an obtuse triangle with sides a, a+1, and a+2
theorem a_range_implies_obtuse_triangle : 
  ∀ a : ℝ, 1 < a ∧ a < 3 → obtuse_triangle a :=
by sorry

end NUMINAMATH_CALUDE_obtuse_triangle_a_range_a_range_implies_obtuse_triangle_l1857_185743


namespace NUMINAMATH_CALUDE_pencil_cost_to_selling_ratio_l1857_185702

/-- Given a purchase of 90 pencils sold at a loss equal to the selling price of 40 pencils,
    the ratio of the cost of 90 pencils to the selling price of 90 pencils is 13:1. -/
theorem pencil_cost_to_selling_ratio :
  ∀ (C S : ℝ),
  C > 0 → S > 0 →
  90 * C - 40 * S = 90 * S →
  (90 * C) / (90 * S) = 13 / 1 := by
sorry

end NUMINAMATH_CALUDE_pencil_cost_to_selling_ratio_l1857_185702


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1857_185721

theorem complex_equation_solution (Z : ℂ) : Z = Complex.I * (2 + Z) → Z = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1857_185721


namespace NUMINAMATH_CALUDE_onion_harvest_weight_l1857_185792

/-- The total weight of onions harvested by Titan's father -/
def total_weight (bags_per_trip : ℕ) (weight_per_bag : ℕ) (num_trips : ℕ) : ℕ :=
  bags_per_trip * weight_per_bag * num_trips

/-- Theorem stating the total weight of onions harvested -/
theorem onion_harvest_weight :
  total_weight 10 50 20 = 10000 := by
  sorry

#eval total_weight 10 50 20

end NUMINAMATH_CALUDE_onion_harvest_weight_l1857_185792


namespace NUMINAMATH_CALUDE_largest_n_for_factorization_l1857_185798

theorem largest_n_for_factorization : ∃ (n : ℤ),
  (∀ (m : ℤ), (∃ (a b c d : ℤ), 3 * X^2 + m * X + 108 = (a * X + b) * (c * X + d)) → m ≤ n) ∧
  (∃ (a b c d : ℤ), 3 * X^2 + n * X + 108 = (a * X + b) * (c * X + d)) ∧
  n = 325 :=
sorry

end NUMINAMATH_CALUDE_largest_n_for_factorization_l1857_185798


namespace NUMINAMATH_CALUDE_binomial_coeff_congruence_l1857_185746

-- Define the binomial coefficient
def binomial_coeff (n p : ℕ) : ℕ := Nat.choose n p

-- State the theorem
theorem binomial_coeff_congruence (p n : ℕ) 
  (hp : Nat.Prime p) 
  (hodd : Odd p) 
  (hn : n ≥ p) :
  binomial_coeff n p ≡ (n / p) [MOD p] :=
sorry

end NUMINAMATH_CALUDE_binomial_coeff_congruence_l1857_185746


namespace NUMINAMATH_CALUDE_expression_simplification_l1857_185788

theorem expression_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let expr := ((a^(3/4) - b^(3/4)) * (a^(3/4) + b^(3/4)) / (a^(1/2) - b^(1/2)) - Real.sqrt (a * b)) *
               (2 * Real.sqrt 2.5 * (a + b)⁻¹) / (Real.sqrt 1000)^(1/3)
  expr = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1857_185788


namespace NUMINAMATH_CALUDE_yogurt_combinations_l1857_185791

theorem yogurt_combinations (flavors : Nat) (toppings : Nat) : 
  flavors = 5 → toppings = 7 → flavors * (toppings.choose 3) = 175 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_combinations_l1857_185791


namespace NUMINAMATH_CALUDE_two_points_on_curve_l1857_185795

def point_on_curve (x y : ℝ) : Prop :=
  x^2 - x*y + 2*y + 1 = 0

def point_A : ℝ × ℝ := (1, -2)
def point_B : ℝ × ℝ := (2, -3)
def point_C : ℝ × ℝ := (3, 10)

theorem two_points_on_curve :
  (point_on_curve point_A.1 point_A.2 ∧
   point_on_curve point_C.1 point_C.2 ∧
   ¬point_on_curve point_B.1 point_B.2) ∨
  (point_on_curve point_A.1 point_A.2 ∧
   point_on_curve point_B.1 point_B.2 ∧
   ¬point_on_curve point_C.1 point_C.2) ∨
  (point_on_curve point_B.1 point_B.2 ∧
   point_on_curve point_C.1 point_C.2 ∧
   ¬point_on_curve point_A.1 point_A.2) :=
sorry

end NUMINAMATH_CALUDE_two_points_on_curve_l1857_185795


namespace NUMINAMATH_CALUDE_simplify_fraction_l1857_185794

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) :
  (x^2 + 1) / (x - 1) - 2 * x / (x - 1) = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1857_185794


namespace NUMINAMATH_CALUDE_linear_function_through_points_and_m_l1857_185730

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a linear function y = kx + b -/
structure LinearFunction where
  k : ℚ
  b : ℚ

/-- Check if a point lies on a linear function -/
def pointOnFunction (p : Point) (f : LinearFunction) : Prop :=
  p.y = f.k * p.x + f.b

theorem linear_function_through_points_and_m
  (A : Point)
  (B : Point)
  (C : Point)
  (h1 : A.x = 3 ∧ A.y = 5)
  (h2 : B.x = -4 ∧ B.y = -9)
  (h3 : C.y = 2) :
  ∃ (f : LinearFunction),
    pointOnFunction A f ∧
    pointOnFunction B f ∧
    pointOnFunction C f ∧
    f.k = 2 ∧
    f.b = -1 ∧
    C.x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_through_points_and_m_l1857_185730


namespace NUMINAMATH_CALUDE_triangle_inequality_l1857_185719

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) :
  Real.sqrt (a^2 + a*b + b^2) + Real.sqrt (b^2 + b*c + c^2) + Real.sqrt (c^2 + c*a + a^2)
  ≤ Real.sqrt (5*a^2 + 5*b^2 + 5*c^2 + 4*a*b + 4*b*c + 4*c*a) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1857_185719


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l1857_185715

/-- Given a circle with equation x^2 + y^2 + 4x = 0, its center is at (-2, 0) and its radius is 2 -/
theorem circle_center_and_radius :
  ∀ (x y : ℝ), x^2 + y^2 + 4*x = 0 → ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (-2, 0) ∧ radius = 2 ∧
    (x + 2)^2 + y^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l1857_185715


namespace NUMINAMATH_CALUDE_units_digit_of_product_of_first_four_composites_l1857_185759

def first_four_composite_numbers : List Nat := [4, 6, 8, 9]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (·*·) 1

def units_digit (n : Nat) : Nat :=
  n % 10

theorem units_digit_of_product_of_first_four_composites :
  units_digit (product_of_list first_four_composite_numbers) = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_of_first_four_composites_l1857_185759


namespace NUMINAMATH_CALUDE_simplify_cube_divided_by_base_l1857_185786

theorem simplify_cube_divided_by_base (x y : ℝ) : (x + y)^3 / (x + y) = (x + y)^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_cube_divided_by_base_l1857_185786


namespace NUMINAMATH_CALUDE_project_nap_duration_l1857_185754

theorem project_nap_duration 
  (project_days : ℕ) 
  (hours_per_day : ℕ) 
  (work_hours : ℕ) 
  (num_naps : ℕ) 
  (h1 : project_days = 4) 
  (h2 : hours_per_day = 24) 
  (h3 : work_hours = 54) 
  (h4 : num_naps = 6) : 
  (project_days * hours_per_day - work_hours) / num_naps = 7 := by
  sorry

end NUMINAMATH_CALUDE_project_nap_duration_l1857_185754


namespace NUMINAMATH_CALUDE_distance_z1z2_to_origin_l1857_185765

open Complex

theorem distance_z1z2_to_origin : 
  let z₁ : ℂ := I
  let z₂ : ℂ := 1 + I
  let z : ℂ := z₁ * z₂
  abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_distance_z1z2_to_origin_l1857_185765


namespace NUMINAMATH_CALUDE_complex_power_modulus_l1857_185767

theorem complex_power_modulus : Complex.abs ((2 : ℂ) + Complex.I * Real.sqrt 11) ^ 4 = 225 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_modulus_l1857_185767


namespace NUMINAMATH_CALUDE_school_trip_students_l1857_185775

/-- The number of students in a school given the number of buses and seats per bus -/
def number_of_students (buses : ℕ) (seats_per_bus : ℕ) : ℕ :=
  buses * seats_per_bus

/-- Theorem stating that the number of students in the school is 111 -/
theorem school_trip_students :
  let buses : ℕ := 37
  let seats_per_bus : ℕ := 3
  number_of_students buses seats_per_bus = 111 := by
  sorry

#eval number_of_students 37 3

end NUMINAMATH_CALUDE_school_trip_students_l1857_185775


namespace NUMINAMATH_CALUDE_log_relation_l1857_185753

theorem log_relation (c b : ℝ) (hc : c = Real.log 625 / Real.log 16) (hb : b = Real.log 25 / Real.log 2) : 
  c = b / 2 := by
sorry

end NUMINAMATH_CALUDE_log_relation_l1857_185753


namespace NUMINAMATH_CALUDE_function_composition_result_l1857_185787

theorem function_composition_result (c d : ℝ) 
  (f : ℝ → ℝ) (g : ℝ → ℝ) 
  (hf : ∀ x, f x = 5*x + c) 
  (hg : ∀ x, g x = c*x + 3) 
  (h_comp : ∀ x, f (g x) = 15*x + d) : d = 18 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_result_l1857_185787


namespace NUMINAMATH_CALUDE_tom_catch_l1857_185749

/-- The number of trout Melanie caught -/
def melanie_catch : ℕ := 8

/-- The factor by which Tom's catch exceeds Melanie's -/
def tom_factor : ℕ := 2

/-- Tom's catch is equal to the product of Melanie's catch and Tom's factor -/
theorem tom_catch : melanie_catch * tom_factor = 16 := by
  sorry

end NUMINAMATH_CALUDE_tom_catch_l1857_185749


namespace NUMINAMATH_CALUDE_complex_equality_sum_l1857_185797

theorem complex_equality_sum (a b : ℝ) : a - 3*I = 2 + b*I → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_sum_l1857_185797


namespace NUMINAMATH_CALUDE_classroom_key_probability_is_two_sevenths_l1857_185790

/-- The probability of selecting a key that opens the classroom door -/
def classroom_key_probability (total_keys : ℕ) (classroom_keys : ℕ) : ℚ :=
  classroom_keys / total_keys

/-- Theorem: The probability of randomly selecting a key that can open the classroom door lock is 2/7 -/
theorem classroom_key_probability_is_two_sevenths :
  classroom_key_probability 7 2 = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_classroom_key_probability_is_two_sevenths_l1857_185790


namespace NUMINAMATH_CALUDE_correct_sampling_methods_l1857_185713

-- Define the sampling scenarios
structure SamplingScenario where
  total : ℕ
  sample_size : ℕ
  categories : Option (List ℕ)

-- Define the sampling methods
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

-- Define the function to determine the most appropriate sampling method
def most_appropriate_sampling_method (scenario : SamplingScenario) : SamplingMethod :=
  sorry

-- Theorem to prove the correct sampling methods for given scenarios
theorem correct_sampling_methods :
  let scenario1 := SamplingScenario.mk 10 2 none
  let scenario2 := SamplingScenario.mk 1920 32 none
  let scenario3 := SamplingScenario.mk 160 20 (some [120, 16, 24])
  (most_appropriate_sampling_method scenario1 = SamplingMethod.SimpleRandom) ∧
  (most_appropriate_sampling_method scenario2 = SamplingMethod.Systematic) ∧
  (most_appropriate_sampling_method scenario3 = SamplingMethod.Stratified) :=
  sorry

end NUMINAMATH_CALUDE_correct_sampling_methods_l1857_185713


namespace NUMINAMATH_CALUDE_correct_mean_problem_l1857_185782

def correct_mean (n : ℕ) (original_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n * original_mean - incorrect_value + correct_value) / n

theorem correct_mean_problem :
  correct_mean 20 36 40 25 = 35.25 := by
  sorry

end NUMINAMATH_CALUDE_correct_mean_problem_l1857_185782


namespace NUMINAMATH_CALUDE_ann_total_blocks_l1857_185761

/-- Ann's initial number of blocks -/
def initial_blocks : ℕ := 9

/-- Number of blocks Ann finds -/
def found_blocks : ℕ := 44

/-- Theorem stating the total number of blocks Ann ends with -/
theorem ann_total_blocks : initial_blocks + found_blocks = 53 := by
  sorry

end NUMINAMATH_CALUDE_ann_total_blocks_l1857_185761


namespace NUMINAMATH_CALUDE_ratio_of_percentages_l1857_185712

theorem ratio_of_percentages (P M N R : ℝ) : 
  P > 0 ∧ P = 0.3 * R ∧ M = 0.35 * R ∧ N = 0.55 * R → M / N = 7 / 11 :=
by
  sorry

end NUMINAMATH_CALUDE_ratio_of_percentages_l1857_185712


namespace NUMINAMATH_CALUDE_distance_to_equidistant_line_in_unit_cube_l1857_185774

/-- Represents a 3D point -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a 3D line -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Represents a unit cube -/
structure UnitCube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A1 : Point3D
  B1 : Point3D
  C1 : Point3D
  D1 : Point3D

/-- Distance between a point and a line in 3D space -/
def distancePointToLine (p : Point3D) (l : Line3D) : ℝ :=
  sorry

/-- Check if two lines are parallel -/
def areLinesParallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Check if a line is equidistant from three other lines -/
def isLineEquidistantFromThreeLines (l l1 l2 l3 : Line3D) : Prop :=
  sorry

/-- The main theorem -/
theorem distance_to_equidistant_line_in_unit_cube 
  (cube : UnitCube) 
  (l : Line3D) 
  (hParallel : areLinesParallel l (Line3D.mk cube.A cube.C1))
  (hEquidistant : isLineEquidistantFromThreeLines l 
    (Line3D.mk cube.B cube.D) 
    (Line3D.mk cube.A1 cube.D1) 
    (Line3D.mk cube.C cube.B1)) :
  distancePointToLine cube.B (Line3D.mk cube.B cube.D) = Real.sqrt 2 / 6 ∧
  distancePointToLine cube.A1 (Line3D.mk cube.A1 cube.D1) = Real.sqrt 2 / 6 ∧
  distancePointToLine cube.C (Line3D.mk cube.C cube.B1) = Real.sqrt 2 / 6 :=
sorry

end NUMINAMATH_CALUDE_distance_to_equidistant_line_in_unit_cube_l1857_185774


namespace NUMINAMATH_CALUDE_smallest_number_of_students_l1857_185700

/-- Represents the number of students in each grade --/
structure Students where
  ninth : ℕ
  eighth : ℕ
  seventh : ℕ

/-- The ratio of 9th-graders to 8th-graders is 7:4 --/
def ratio_9th_8th (s : Students) : Prop :=
  7 * s.eighth = 4 * s.ninth

/-- The ratio of 9th-graders to 7th-graders is 10:3 --/
def ratio_9th_7th (s : Students) : Prop :=
  10 * s.seventh = 3 * s.ninth

/-- The total number of students --/
def total_students (s : Students) : ℕ :=
  s.ninth + s.eighth + s.seventh

/-- The main theorem stating the smallest possible number of students --/
theorem smallest_number_of_students :
  ∃ (s : Students),
    ratio_9th_8th s ∧
    ratio_9th_7th s ∧
    total_students s = 131 ∧
    (∀ (t : Students),
      ratio_9th_8th t → ratio_9th_7th t →
      total_students t ≥ total_students s) :=
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_students_l1857_185700


namespace NUMINAMATH_CALUDE_tangent_slope_at_pi_over_4_l1857_185769

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / (Real.sin x + Real.cos x) - 1/2

theorem tangent_slope_at_pi_over_4 :
  let df := deriv f
  df (π/4) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_at_pi_over_4_l1857_185769


namespace NUMINAMATH_CALUDE_division_evaluation_l1857_185780

theorem division_evaluation : 250 / (5 + 12 * 3^2) = 250 / 113 := by sorry

end NUMINAMATH_CALUDE_division_evaluation_l1857_185780


namespace NUMINAMATH_CALUDE_lap_time_improvement_l1857_185709

-- Define the initial swimming scenario
def initial_total_time : ℚ := 29
def initial_break_time : ℚ := 3
def initial_laps : ℚ := 14

-- Define the current swimming scenario
def current_total_time : ℚ := 28
def current_laps : ℚ := 16

-- Define the lap time calculation function
def lap_time (total_time : ℚ) (break_time : ℚ) (laps : ℚ) : ℚ :=
  (total_time - break_time) / laps

-- State the theorem
theorem lap_time_improvement :
  lap_time initial_total_time initial_break_time initial_laps -
  lap_time current_total_time 0 current_laps = 3 / 28 := by
  sorry

end NUMINAMATH_CALUDE_lap_time_improvement_l1857_185709


namespace NUMINAMATH_CALUDE_tate_high_school_duration_l1857_185772

theorem tate_high_school_duration (normal_hs_duration : ℕ) (total_time : ℕ) (x : ℕ) : 
  normal_hs_duration = 4 →
  total_time = 12 →
  (normal_hs_duration - x) + 3 * (normal_hs_duration - x) = total_time →
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_tate_high_school_duration_l1857_185772


namespace NUMINAMATH_CALUDE_closest_fraction_l1857_185737

def medals_won : ℚ := 23 / 150

def options : List ℚ := [1/5, 1/6, 1/7, 1/8, 1/9]

theorem closest_fraction :
  ∃ (closest : ℚ), closest ∈ options ∧
  ∀ (x : ℚ), x ∈ options → |medals_won - closest| ≤ |medals_won - x| :=
by
  sorry

end NUMINAMATH_CALUDE_closest_fraction_l1857_185737


namespace NUMINAMATH_CALUDE_teal_color_survey_l1857_185760

theorem teal_color_survey (total : ℕ) (green : ℕ) (both : ℕ) (neither : ℕ) 
  (h_total : total = 150)
  (h_green : green = 90)
  (h_both : both = 45)
  (h_neither : neither = 24) :
  ∃ blue : ℕ, blue = 81 ∧ blue = total - (green - both) - both - neither :=
by sorry

end NUMINAMATH_CALUDE_teal_color_survey_l1857_185760


namespace NUMINAMATH_CALUDE_probability_all_scissors_l1857_185738

-- Define the possible choices in the game
inductive Choice
  | Rock
  | Paper
  | Scissors

-- Define a function to calculate the probability of a specific outcome
def probability_of_outcome (num_players : ℕ) (num_choices : ℕ) (favorable_outcomes : ℕ) : ℚ :=
  (favorable_outcomes : ℚ) / (num_choices ^ num_players : ℚ)

-- Theorem statement
theorem probability_all_scissors :
  let num_players : ℕ := 3
  let num_choices : ℕ := 3
  let favorable_outcomes : ℕ := 1
  probability_of_outcome num_players num_choices favorable_outcomes = 1 / 27 := by
  sorry


end NUMINAMATH_CALUDE_probability_all_scissors_l1857_185738


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_eccentricity_l1857_185789

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A conic section (ellipse or hyperbola) -/
structure Conic where
  center : Point
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis
  isEllipse : Bool

/-- The eccentricity of a conic section -/
def eccentricity (c : Conic) : ℝ :=
  sorry

/-- The foci of a conic section -/
def foci (c : Conic) : (Point × Point) :=
  sorry

/-- The angle between three points -/
def angle (p1 p2 p3 : Point) : ℝ :=
  sorry

/-- The intersection points of two conic sections -/
def intersection (c1 c2 : Conic) : Set Point :=
  sorry

theorem ellipse_hyperbola_eccentricity 
  (C₁ : Conic) (C₂ : Conic) (F₁ F₂ P : Point) :
  C₁.isEllipse = true →
  C₂.isEllipse = false →
  foci C₁ = (F₁, F₂) →
  foci C₂ = (F₁, F₂) →
  P ∈ intersection C₁ C₂ →
  P.x > 0 ∧ P.y > 0 →
  eccentricity C₁ * eccentricity C₂ = 1 →
  angle F₁ P F₂ = π / 3 →
  eccentricity C₁ = Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_eccentricity_l1857_185789


namespace NUMINAMATH_CALUDE_max_value_sum_fractions_l1857_185770

theorem max_value_sum_fractions (a b c d : ℝ) 
  (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c) (nonneg_d : 0 ≤ d)
  (sum_eq_one : a + b + c + d = 1) :
  (a * b) / (a + b) + (a * c) / (a + c) + (a * d) / (a + d) +
  (b * c) / (b + c) + (b * d) / (b + d) + (c * d) / (c + d) ≤ 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_fractions_l1857_185770


namespace NUMINAMATH_CALUDE_acute_triangle_side_range_l1857_185744

theorem acute_triangle_side_range (x : ℝ) : 
  x > 0 → 
  (∀ α β γ : ℝ, 0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2 ∧ 0 < γ ∧ γ < π/2 ∧ 
   α + β + γ = π ∧
   x^2 = 2^2 + 3^2 - 2*2*3*Real.cos γ ∧
   2^2 = 3^2 + x^2 - 2*3*x*Real.cos α ∧
   3^2 = 2^2 + x^2 - 2*2*x*Real.cos β) →
  Real.sqrt 5 < x ∧ x < Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_acute_triangle_side_range_l1857_185744


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1857_185745

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 9) (h2 : x ≠ -7) :
  (4 * x - 6) / (x^2 - 2*x - 63) = (15 / 8) / (x - 9) + (17 / 8) / (x + 7) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1857_185745


namespace NUMINAMATH_CALUDE_course_selection_schemes_l1857_185757

theorem course_selection_schemes :
  let total_courses : ℕ := 4
  let student_a_choices : ℕ := 2
  let student_b_choices : ℕ := 3
  let student_c_choices : ℕ := 3
  
  (Nat.choose total_courses student_a_choices) *
  (Nat.choose total_courses student_b_choices) *
  (Nat.choose total_courses student_c_choices) = 96 :=
by
  sorry

end NUMINAMATH_CALUDE_course_selection_schemes_l1857_185757


namespace NUMINAMATH_CALUDE_seashell_collection_l1857_185704

theorem seashell_collection (current : ℕ) (target : ℕ) (additional : ℕ) : 
  current = 19 → target = 25 → current + additional = target → additional = 6 := by
  sorry

end NUMINAMATH_CALUDE_seashell_collection_l1857_185704


namespace NUMINAMATH_CALUDE_spinner_probability_l1857_185707

-- Define the spinner sections
def spinner_sections : List ℕ := [3, 6, 1, 4, 8, 10, 2, 7]

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop := sorry

-- Define a function to check if a number is odd
def is_odd (n : ℕ) : Prop := sorry

-- Define a function to count elements satisfying a condition
def count_if (l : List ℕ) (f : ℕ → Prop) : ℕ := sorry

-- Theorem statement
theorem spinner_probability :
  let favorable_outcomes := count_if spinner_sections (λ n => is_prime n ∨ is_odd n)
  let total_outcomes := spinner_sections.length
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_spinner_probability_l1857_185707


namespace NUMINAMATH_CALUDE_product_of_numbers_with_sum_and_difference_l1857_185762

theorem product_of_numbers_with_sum_and_difference 
  (x y : ℝ) (h1 : x + y = 70) (h2 : x - y = 10) : x * y = 1200 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_sum_and_difference_l1857_185762


namespace NUMINAMATH_CALUDE_max_distance_a_c_theorem_l1857_185751

def max_distance_a_c (a b c : ℝ × ℝ) : Prop :=
  let norm := λ v : ℝ × ℝ => Real.sqrt (v.1^2 + v.2^2)
  let dot := λ u v : ℝ × ℝ => u.1 * v.1 + u.2 * v.2
  norm a = 2 ∧ 
  norm b = 2 ∧ 
  dot a b = 2 ∧ 
  dot c (a + 2 • b - 2 • c) = 2 →
  (∀ c', dot c' (a + 2 • b - 2 • c') = 2 → 
    norm (a - c) ≤ (Real.sqrt 3 + Real.sqrt 7) / 2) ∧
  (∃ c', dot c' (a + 2 • b - 2 • c') = 2 ∧ 
    norm (a - c') = (Real.sqrt 3 + Real.sqrt 7) / 2)

theorem max_distance_a_c_theorem (a b c : ℝ × ℝ) : 
  max_distance_a_c a b c := by sorry

end NUMINAMATH_CALUDE_max_distance_a_c_theorem_l1857_185751


namespace NUMINAMATH_CALUDE_cos_pi_plus_alpha_l1857_185784

theorem cos_pi_plus_alpha (α : Real) (h : Real.sin (π / 2 - α) = 3 / 5) :
  Real.cos (π + α) = -3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_plus_alpha_l1857_185784


namespace NUMINAMATH_CALUDE_unique_element_in_A_l1857_185783

/-- The set A defined by the quadratic equation ax^2 - x + 1 = 0 -/
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 - x + 1 = 0}

/-- The theorem stating that if A contains only one element, then a = 0 or a = 1/4 -/
theorem unique_element_in_A (a : ℝ) : (∃! x, x ∈ A a) → a = 0 ∨ a = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_unique_element_in_A_l1857_185783


namespace NUMINAMATH_CALUDE_difference_of_squares_65_35_l1857_185714

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_65_35_l1857_185714


namespace NUMINAMATH_CALUDE_distinct_divisors_count_l1857_185741

def divisors (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (λ d => n % d = 0)

theorem distinct_divisors_count (M Z : ℕ) 
  (hM : (divisors M).card = 10)
  (hZ : (divisors Z).card = 9)
  (h6M : 6 ∣ M)
  (h6Z : 6 ∣ Z) :
  ((divisors M) ∪ (divisors Z)).card = 13 := by
  sorry

end NUMINAMATH_CALUDE_distinct_divisors_count_l1857_185741


namespace NUMINAMATH_CALUDE_rectangle_area_l1857_185708

theorem rectangle_area (x : ℝ) (h : x > 0) : ∃ w l : ℝ,
  w > 0 ∧ l > 0 ∧ 
  l = 3 * w ∧ 
  w ^ 2 + l ^ 2 = x ^ 2 ∧
  w * l = (3 / 10) * x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1857_185708


namespace NUMINAMATH_CALUDE_negation_of_forall_cubic_l1857_185736

theorem negation_of_forall_cubic (P : ℝ → Prop) :
  (¬ ∀ x < 0, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x < 0, x^3 - x^2 + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_cubic_l1857_185736


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1857_185716

theorem max_value_of_expression (p q r s : ℕ) : 
  p ∈ ({1, 2, 3, 4} : Finset ℕ) → 
  q ∈ ({1, 2, 3, 4} : Finset ℕ) → 
  r ∈ ({1, 2, 3, 4} : Finset ℕ) → 
  s ∈ ({1, 2, 3, 4} : Finset ℕ) → 
  p ≠ q → p ≠ r → p ≠ s → q ≠ r → q ≠ s → r ≠ s → 
  p^q + r^s ≤ 83 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1857_185716


namespace NUMINAMATH_CALUDE_factoring_quadratic_l1857_185740

theorem factoring_quadratic (x : ℝ) : 60 * x + 90 - 15 * x^2 = 15 * (-x^2 + 4 * x + 6) := by
  sorry

end NUMINAMATH_CALUDE_factoring_quadratic_l1857_185740


namespace NUMINAMATH_CALUDE_hyperbola_to_ellipse_l1857_185766

/-- Given a hyperbola with equation x^2/4 - y^2/12 = -1, 
    prove that the equation of the ellipse with its vertices at the foci of the hyperbola 
    and its foci at the vertices of the hyperbola is x^2/4 + y^2/16 = 1 -/
theorem hyperbola_to_ellipse (x y : ℝ) :
  (x^2 / 4 - y^2 / 12 = -1) →
  ∃ (x' y' : ℝ), (x'^2 / 4 + y'^2 / 16 = 1 ∧ 
    (∀ (a b c : ℝ), (a > b ∧ b > 0 ∧ c > 0) → 
      (y'^2 / a^2 + x'^2 / b^2 = 1 ↔ 
        (a = 4 ∧ b^2 = 4 ∧ c = 2 * Real.sqrt 3)))) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_to_ellipse_l1857_185766


namespace NUMINAMATH_CALUDE_max_value_implies_a_equals_three_l1857_185703

/-- Given a function y = x(1-ax) with maximum value 1/12 for 0 < x < 1/a, prove that a = 3 -/
theorem max_value_implies_a_equals_three (a : ℝ) : 
  (∃ (max_y : ℝ), max_y = 1/12 ∧ 
    (∀ x : ℝ, 0 < x → x < 1/a → x * (1 - a*x) ≤ max_y) ∧
    (∃ x : ℝ, 0 < x ∧ x < 1/a ∧ x * (1 - a*x) = max_y)) →
  a = 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_equals_three_l1857_185703


namespace NUMINAMATH_CALUDE_only_villages_comprehensive_villages_only_comprehensive_option_l1857_185750

/-- Represents a survey option -/
inductive SurveyOption
  | VillagesPollution
  | DrugQuality
  | PublicOpinion
  | RiverWaterQuality

/-- Defines what makes a survey comprehensive -/
def is_comprehensive (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.VillagesPollution => true
  | _ => false

/-- Theorem stating that only the villages pollution survey is comprehensive -/
theorem only_villages_comprehensive :
  ∀ (option : SurveyOption),
    is_comprehensive option ↔ option = SurveyOption.VillagesPollution :=
by sorry

/-- Main theorem proving that investigating five villages is the only suitable option -/
theorem villages_only_comprehensive_option :
  ∃! (option : SurveyOption), is_comprehensive option :=
by sorry

end NUMINAMATH_CALUDE_only_villages_comprehensive_villages_only_comprehensive_option_l1857_185750


namespace NUMINAMATH_CALUDE_least_prime_angle_in_right_triangle_l1857_185739

theorem least_prime_angle_in_right_triangle : ∀ a b : ℕ,
  (a > b) →
  (a + b = 90) →
  (Nat.Prime a) →
  (Nat.Prime b) →
  (∀ c : ℕ, (c < b) → (c + a ≠ 90 ∨ ¬(Nat.Prime c))) →
  b = 7 :=
by sorry

end NUMINAMATH_CALUDE_least_prime_angle_in_right_triangle_l1857_185739


namespace NUMINAMATH_CALUDE_princess_pear_cherries_l1857_185755

def jester_height (i : ℕ) : ℕ := i

def is_valid_group (group : Finset ℕ) : Prop :=
  group.card = 6 ∧ ∃ (n : ℕ), n ≤ 100 ∧
  (∃ (lower upper : Finset ℕ),
    lower.card = 3 ∧ upper.card = 3 ∧
    lower ∪ upper = group ∧
    ∀ i ∈ lower, ∀ j ∈ upper, jester_height i < jester_height j)

def number_of_cherries : ℕ := (Nat.choose 50 3) ^ 2 * 2

theorem princess_pear_cherries :
  number_of_cherries = 384160000 := by sorry

end NUMINAMATH_CALUDE_princess_pear_cherries_l1857_185755


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1857_185717

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_a2 : a 2 = 3)
  (h_a5 : a 5 = 12) :
  a 8 = 21 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1857_185717


namespace NUMINAMATH_CALUDE_expression_evaluation_l1857_185748

theorem expression_evaluation : -20 + 7 * ((8 - 2) / 3) = -6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1857_185748


namespace NUMINAMATH_CALUDE_weeks_passed_l1857_185747

/-- Prove that the number of weeks that have already passed is 4 --/
theorem weeks_passed
  (watch_cost : ℕ)
  (weekly_allowance : ℕ)
  (current_savings : ℕ)
  (weeks_left : ℕ)
  (h1 : watch_cost = 100)
  (h2 : weekly_allowance = 5)
  (h3 : current_savings = 20)
  (h4 : weeks_left = 16)
  (h5 : current_savings + weeks_left * weekly_allowance = watch_cost) :
  current_savings / weekly_allowance = 4 := by
  sorry


end NUMINAMATH_CALUDE_weeks_passed_l1857_185747


namespace NUMINAMATH_CALUDE_oranges_thrown_away_l1857_185777

theorem oranges_thrown_away (initial : ℕ) (new : ℕ) (final : ℕ) : 
  initial = 34 → new = 13 → final = 27 → initial - (initial - final + new) = 20 := by
  sorry

end NUMINAMATH_CALUDE_oranges_thrown_away_l1857_185777


namespace NUMINAMATH_CALUDE_difference_of_squares_l1857_185776

theorem difference_of_squares : (23 + 15)^2 - (23 - 15)^2 = 1380 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1857_185776


namespace NUMINAMATH_CALUDE_equal_one_and_two_digit_prob_l1857_185771

def num_sides : ℕ := 15
def num_dice : ℕ := 5

def prob_one_digit : ℚ := 3 / 5
def prob_two_digit : ℚ := 2 / 5

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem equal_one_and_two_digit_prob :
  (choose num_dice (num_dice / 2)) * (prob_two_digit ^ (num_dice / 2)) * (prob_one_digit ^ (num_dice / 2 + 1)) = 108 / 625 :=
sorry

end NUMINAMATH_CALUDE_equal_one_and_two_digit_prob_l1857_185771


namespace NUMINAMATH_CALUDE_corn_yield_theorem_l1857_185718

/-- Calculates the total corn yield for Johnson and his neighbor after 6 months -/
def total_corn_yield (johnson_yield : ℕ) (johnson_area : ℕ) (neighbor_area : ℕ) (months : ℕ) : ℕ :=
  let johnson_total := johnson_yield * (months / 2)
  let neighbor_yield := 2 * johnson_yield
  let neighbor_total := neighbor_yield * neighbor_area * (months / 2)
  johnson_total + neighbor_total

/-- Theorem stating that the total corn yield is 1200 under given conditions -/
theorem corn_yield_theorem :
  total_corn_yield 80 1 2 6 = 1200 :=
by
  sorry

end NUMINAMATH_CALUDE_corn_yield_theorem_l1857_185718


namespace NUMINAMATH_CALUDE_corn_stalk_calculation_hilary_corn_stalks_l1857_185725

theorem corn_stalk_calculation (ears_per_stalk : ℕ) 
  (kernels_low : ℕ) (kernels_high : ℕ) (total_kernels : ℕ) : ℕ :=
  let avg_kernels := (kernels_low + kernels_high) / 2
  let total_ears := total_kernels / avg_kernels
  total_ears / ears_per_stalk

theorem hilary_corn_stalks : 
  corn_stalk_calculation 4 500 600 237600 = 108 := by
  sorry

end NUMINAMATH_CALUDE_corn_stalk_calculation_hilary_corn_stalks_l1857_185725


namespace NUMINAMATH_CALUDE_parabola_translation_l1857_185796

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := -2 * x^2

-- Define the translated parabola
def translated_parabola (x : ℝ) : ℝ := -2 * (x - 1)^2 + 2

-- Theorem stating that the translated parabola is correct
theorem parabola_translation :
  ∀ x : ℝ, translated_parabola x = original_parabola (x - 1) + 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l1857_185796


namespace NUMINAMATH_CALUDE_square_garden_multiple_l1857_185724

theorem square_garden_multiple (a p : ℝ) (h1 : p = 38) (h2 : a = (p / 4)^2) (h3 : ∃ m : ℝ, a = m * p + 14.25) : 
  ∃ m : ℝ, a = m * p + 14.25 ∧ m = 2 :=
sorry

end NUMINAMATH_CALUDE_square_garden_multiple_l1857_185724


namespace NUMINAMATH_CALUDE_total_distance_rowed_l1857_185711

/-- Calculates the total distance rowed by a man given specific conditions -/
theorem total_distance_rowed (still_water_speed wind_speed river_speed : ℝ)
  (total_time : ℝ) (h1 : still_water_speed = 8)
  (h2 : wind_speed = 1.5) (h3 : river_speed = 3.5) (h4 : total_time = 2) :
  let speed_to := still_water_speed - river_speed - wind_speed
  let speed_from := still_water_speed + river_speed + wind_speed
  let distance := (speed_to * speed_from * total_time) / (speed_to + speed_from)
  2 * distance = 9.75 :=
by sorry

end NUMINAMATH_CALUDE_total_distance_rowed_l1857_185711


namespace NUMINAMATH_CALUDE_max_product_sum_300_l1857_185706

theorem max_product_sum_300 : 
  (∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500) ∧ 
  (∀ (x y : ℤ), x + y = 300 → x * y ≤ 22500) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_300_l1857_185706


namespace NUMINAMATH_CALUDE_quote_sharer_sequences_count_l1857_185742

/-- The number of different sequences of 3 students chosen from a group of 8 students, 
    where no student can be repeated. -/
def quote_sharer_sequences : ℕ := 8 * 7 * 6

/-- Theorem stating that the number of different sequences of 3 students chosen from a group of 8 students, 
    where no student can be repeated, is equal to 336. -/
theorem quote_sharer_sequences_count : quote_sharer_sequences = 336 := by
  sorry

end NUMINAMATH_CALUDE_quote_sharer_sequences_count_l1857_185742


namespace NUMINAMATH_CALUDE_expression_equality_l1857_185799

theorem expression_equality : 
  Real.sqrt (4/3) * Real.sqrt 15 + ((-8) ^ (1/3 : ℝ)) + (π - 3) ^ (0 : ℝ) = 2 * Real.sqrt 5 - 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_equality_l1857_185799


namespace NUMINAMATH_CALUDE_mary_lambs_traded_l1857_185781

def lambs_traded_for_goat (initial_lambs : ℕ) (lambs_with_babies : ℕ) (babies_per_lamb : ℕ) (extra_lambs : ℕ) (final_lambs : ℕ) : ℕ :=
  initial_lambs + lambs_with_babies * babies_per_lamb + extra_lambs - final_lambs

theorem mary_lambs_traded :
  lambs_traded_for_goat 6 2 2 7 14 = 3 := by
  sorry

end NUMINAMATH_CALUDE_mary_lambs_traded_l1857_185781


namespace NUMINAMATH_CALUDE_inverse_f_at_10_l1857_185710

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- Define the domain of f
def f_domain (x : ℝ) : Prop := x ≥ 1

-- State the theorem
theorem inverse_f_at_10 (f_inv : ℝ → ℝ) 
  (h1 : ∀ x, f_domain x → f_inv (f x) = x) 
  (h2 : ∀ y, y ≥ 1 → f (f_inv y) = y) : 
  f_inv 10 = 3 := by sorry

end NUMINAMATH_CALUDE_inverse_f_at_10_l1857_185710


namespace NUMINAMATH_CALUDE_tan_alpha_and_fraction_l1857_185734

theorem tan_alpha_and_fraction (α : Real) 
  (h : Real.tan (α + π / 4) = 2) : 
  Real.tan α = 1 / 3 ∧ 
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = -1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_and_fraction_l1857_185734


namespace NUMINAMATH_CALUDE_josh_selena_distance_ratio_l1857_185727

/-- Proves that the ratio of Josh's distance to Selena's distance is 1/2 -/
theorem josh_selena_distance_ratio :
  let total_distance : ℝ := 36
  let selena_distance : ℝ := 24
  let josh_distance : ℝ := total_distance - selena_distance
  josh_distance / selena_distance = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_josh_selena_distance_ratio_l1857_185727


namespace NUMINAMATH_CALUDE_value_of_expression_l1857_185731

theorem value_of_expression (x : ℝ) (h : x = 5) : 3 * x + 4 = 19 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l1857_185731


namespace NUMINAMATH_CALUDE_product_is_very_large_l1857_185723

theorem product_is_very_large : 
  (3 + 2) * 
  (3^2 + 2^2) * 
  (3^4 + 2^4) * 
  (3^8 + 2^8) * 
  (3^16 + 2^16) * 
  (3^32 + 2^32) * 
  (3^64 + 2^64) > 10^400 := by
sorry

end NUMINAMATH_CALUDE_product_is_very_large_l1857_185723


namespace NUMINAMATH_CALUDE_solution_set_f_less_than_2_range_of_a_for_solution_exists_l1857_185729

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 1|

-- Theorem for the solution set of f(x) < 2
theorem solution_set_f_less_than_2 :
  {x : ℝ | f x < 2} = Set.Ioo (-4 : ℝ) (2/3) := by sorry

-- Theorem for the range of a where f(x) ≤ a - a²/2 has a solution
theorem range_of_a_for_solution_exists :
  {a : ℝ | ∃ x, f x ≤ a - a^2/2} = Set.Icc (-1 : ℝ) 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_less_than_2_range_of_a_for_solution_exists_l1857_185729


namespace NUMINAMATH_CALUDE_last_digit_3_2004_l1857_185785

/-- The last digit of 3^n -/
def last_digit (n : ℕ) : ℕ :=
  (3^n) % 10

/-- The pattern of last digits repeats every 4 steps -/
axiom last_digit_pattern (n : ℕ) :
  last_digit n = last_digit (n % 4)

/-- The last digits for the first 4 powers of 3 -/
axiom last_digit_base :
  last_digit 0 = 1 ∧ 
  last_digit 1 = 3 ∧ 
  last_digit 2 = 9 ∧ 
  last_digit 3 = 7

theorem last_digit_3_2004 :
  last_digit 2004 = 1 :=
by sorry

end NUMINAMATH_CALUDE_last_digit_3_2004_l1857_185785


namespace NUMINAMATH_CALUDE_wheel_revolutions_l1857_185752

/-- The number of revolutions of a wheel with diameter 10 feet to travel half a mile -/
theorem wheel_revolutions (π : ℝ) (h : π > 0) : 
  let diameter : ℝ := 10
  let circumference : ℝ := π * diameter
  let half_mile_in_feet : ℝ := 5280 / 2
  half_mile_in_feet / circumference = 264 / π := by
  sorry

end NUMINAMATH_CALUDE_wheel_revolutions_l1857_185752


namespace NUMINAMATH_CALUDE_monkey_climb_theorem_l1857_185763

/-- The time taken for a monkey to climb a tree -/
def monkey_climb_time (tree_height : ℕ) (hop_distance : ℕ) (slip_distance : ℕ) : ℕ :=
  let effective_climb := hop_distance - slip_distance
  let full_climbs := (tree_height - 1) / effective_climb
  let remaining_distance := (tree_height - 1) % effective_climb
  full_climbs + if remaining_distance > 0 then 1 else 0

/-- Theorem: A monkey climbing a 17 ft tree, hopping 3 ft and slipping 2 ft each hour, takes 17 hours to reach the top -/
theorem monkey_climb_theorem :
  monkey_climb_time 17 3 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_monkey_climb_theorem_l1857_185763


namespace NUMINAMATH_CALUDE_problem_statement_l1857_185793

def p (x : ℝ) : Prop := x^2 - 4*x - 5 ≤ 0

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

theorem problem_statement (m : ℝ) (h : m > 0) :
  (∀ x, p x → q x m) → m ∈ Set.Ici 4 ∧
  (m = 5 → ∀ x, (p x ∨ q x m) ∧ ¬(p x ∧ q x m) → 
    x ∈ Set.Ioc (-4) (-1) ∪ Set.Ioo 5 6) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1857_185793


namespace NUMINAMATH_CALUDE_diamond_eight_three_l1857_185758

-- Define the diamond operation
def diamond (x y : ℤ) : ℤ :=
  sorry

-- State the theorem
theorem diamond_eight_three : diamond 8 3 = 39 := by
  sorry

-- Define the properties of the diamond operation
axiom diamond_zero (x : ℤ) : diamond x 0 = x

axiom diamond_comm (x y : ℤ) : diamond x y = diamond y x

axiom diamond_recursive (x y : ℤ) : diamond (x + 2) y = diamond x y + 2 * y + 3

end NUMINAMATH_CALUDE_diamond_eight_three_l1857_185758


namespace NUMINAMATH_CALUDE_largest_n_when_floor_sqrt_n_is_5_l1857_185732

/-- Floor function: largest integer not greater than x -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

theorem largest_n_when_floor_sqrt_n_is_5 :
  ∀ n : ℕ, (floor (Real.sqrt n) = 5) → (∀ m : ℕ, m ≤ n → m ≤ 35) ∧ n ≤ 35 :=
sorry

end NUMINAMATH_CALUDE_largest_n_when_floor_sqrt_n_is_5_l1857_185732
