import Mathlib

namespace NUMINAMATH_CALUDE_largest_p_value_l672_67259

theorem largest_p_value (m n p : ℕ) : 
  m ≤ n → n ≤ p → 
  2 * m * n * p = (m + 2) * (n + 2) * (p + 2) → 
  p ≤ 130 :=
by sorry

end NUMINAMATH_CALUDE_largest_p_value_l672_67259


namespace NUMINAMATH_CALUDE_cara_don_meeting_l672_67244

/-- The distance Cara walks before meeting Don -/
def distance_cara_walks : ℝ := 18

/-- The total distance between Cara's and Don's homes -/
def total_distance : ℝ := 45

/-- Cara's walking speed in km/h -/
def cara_speed : ℝ := 6

/-- Don's walking speed in km/h -/
def don_speed : ℝ := 5

/-- The time Don starts walking after Cara (in hours) -/
def don_start_delay : ℝ := 2

theorem cara_don_meeting :
  distance_cara_walks = 18 ∧
  distance_cara_walks + don_speed * (distance_cara_walks / cara_speed) =
    total_distance - cara_speed * don_start_delay :=
sorry

end NUMINAMATH_CALUDE_cara_don_meeting_l672_67244


namespace NUMINAMATH_CALUDE_remainder_not_composite_l672_67239

theorem remainder_not_composite (p : Nat) (h_prime : Nat.Prime p) (h_gt_30 : p > 30) :
  ¬(∃ (a b : Nat), a > 1 ∧ b > 1 ∧ p % 30 = a * b) := by
  sorry

end NUMINAMATH_CALUDE_remainder_not_composite_l672_67239


namespace NUMINAMATH_CALUDE_probability_one_white_one_black_is_eight_fifteenths_l672_67287

def total_balls : ℕ := 15
def white_balls : ℕ := 8
def black_balls : ℕ := 7

def probability_one_white_one_black : ℚ := (white_balls * black_balls) / (total_balls.choose 2)

theorem probability_one_white_one_black_is_eight_fifteenths :
  probability_one_white_one_black = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_white_one_black_is_eight_fifteenths_l672_67287


namespace NUMINAMATH_CALUDE_expression_evaluation_l672_67223

theorem expression_evaluation : 2 - (-3) - 4 - (-5) - 6 - (-7) * 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l672_67223


namespace NUMINAMATH_CALUDE_sum_of_legs_is_462_l672_67256

/-- A right triangle with two inscribed squares -/
structure RightTriangleWithSquares where
  -- The right triangle
  AC : ℝ
  CB : ℝ
  -- The two inscribed squares
  S1 : ℝ
  S2 : ℝ
  -- Conditions
  right_triangle : AC^2 + CB^2 = (AC + CB)^2 / 2
  area_S1 : S1^2 = 441
  area_S2 : S2^2 = 440

/-- The sum of the legs of the right triangle is 462 -/
theorem sum_of_legs_is_462 (t : RightTriangleWithSquares) : t.AC + t.CB = 462 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_legs_is_462_l672_67256


namespace NUMINAMATH_CALUDE_max_min_product_l672_67280

theorem max_min_product (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →
  a + b + c = 12 →
  a * b + b * c + c * a = 20 →
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧
             m ≤ 12 ∧
             ∀ (k : ℝ), (∃ (x y z : ℝ), 
               0 < x ∧ 0 < y ∧ 0 < z ∧
               x + y + z = 12 ∧
               x * y + y * z + z * x = 20 ∧
               k = min (x * y) (min (y * z) (z * x))) →
             k ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_max_min_product_l672_67280


namespace NUMINAMATH_CALUDE_specific_pyramid_volume_l672_67264

/-- A right pyramid with a square base -/
structure SquarePyramid where
  base_area : ℝ
  total_surface_area : ℝ
  triangular_face_area : ℝ

/-- The volume of a square pyramid -/
def volume (p : SquarePyramid) : ℝ := sorry

/-- Theorem stating the volume of the specific pyramid -/
theorem specific_pyramid_volume :
  ∀ (p : SquarePyramid),
  p.total_surface_area = 648 ∧
  p.triangular_face_area = (1/3) * p.base_area ∧
  p.total_surface_area = p.base_area + 4 * p.triangular_face_area →
  volume p = (4232 * Real.sqrt 6) / 9 := by
  sorry

end NUMINAMATH_CALUDE_specific_pyramid_volume_l672_67264


namespace NUMINAMATH_CALUDE_fishing_rod_price_l672_67238

theorem fishing_rod_price (initial_price : ℝ) (saturday_increase : ℝ) (sunday_discount : ℝ) :
  initial_price = 50 ∧ 
  saturday_increase = 0.2 ∧ 
  sunday_discount = 0.15 →
  initial_price * (1 + saturday_increase) * (1 - sunday_discount) = 51 := by
  sorry

end NUMINAMATH_CALUDE_fishing_rod_price_l672_67238


namespace NUMINAMATH_CALUDE_minimum_value_implies_m_l672_67267

noncomputable def f (x m : ℝ) : ℝ := 2 * x * Real.log (2 * x - 1) - Real.log (2 * x - 1) - m * x + Real.exp (-1)

theorem minimum_value_implies_m (h : ∀ x ∈ Set.Icc 1 (3/2), f x m ≥ -4 + Real.exp (-1)) 
  (h_min : ∃ x ∈ Set.Icc 1 (3/2), f x m = -4 + Real.exp (-1)) : 
  m = 4/3 * Real.log 2 + 8/3 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_implies_m_l672_67267


namespace NUMINAMATH_CALUDE_largest_t_value_l672_67254

theorem largest_t_value : ∃ (t_max : ℝ), 
  (∀ t : ℝ, (15 * t^2 - 40 * t + 18) / (4 * t - 3) + 3 * t = 4 * t + 2 → t ≤ t_max) ∧
  ((15 * t_max^2 - 40 * t_max + 18) / (4 * t_max - 3) + 3 * t_max = 4 * t_max + 2) ∧
  t_max = 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_t_value_l672_67254


namespace NUMINAMATH_CALUDE_wendy_running_distance_l672_67228

theorem wendy_running_distance (ran walked : ℝ) (h1 : ran = 19.833333333333332) 
  (h2 : walked = 9.166666666666666) : 
  ran - walked = 10.666666666666666 := by
  sorry

end NUMINAMATH_CALUDE_wendy_running_distance_l672_67228


namespace NUMINAMATH_CALUDE_x_equals_y_when_t_is_half_l672_67216

theorem x_equals_y_when_t_is_half (t : ℚ) : 
  let x := 1 - 4 * t
  let y := 2 * t - 2
  x = y ↔ t = 1/2 := by
sorry

end NUMINAMATH_CALUDE_x_equals_y_when_t_is_half_l672_67216


namespace NUMINAMATH_CALUDE_tangent_line_proof_l672_67268

/-- The function f(x) = x^2 -/
def f (x : ℝ) : ℝ := x^2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 2 * x

/-- The line equation: 2x - y - 1 = 0 -/
def line_equation (x y : ℝ) : Prop := 2 * x - y - 1 = 0

theorem tangent_line_proof :
  (∃ (x₀ : ℝ), f x₀ = x₀ ∧ line_equation x₀ (f x₀)) ∧  -- The line passes through a point on f(x)
  line_equation 1 1 ∧  -- The line passes through (1,1)
  (∀ (x : ℝ), f' x = (2 : ℝ)) →  -- The derivative of f(x) is 2
  ∃ (x₀ : ℝ), f x₀ = x₀ ∧ line_equation x₀ (f x₀) ∧ f' x₀ = 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_proof_l672_67268


namespace NUMINAMATH_CALUDE_union_intersection_equality_union_subset_l672_67253

-- Define sets A and B
def A : Set ℝ := {x : ℝ | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

-- Theorem for part (1)
theorem union_intersection_equality (a : ℝ) :
  A ∪ B a = A ∩ B a → a = 1 := by sorry

-- Theorem for part (2)
theorem union_subset (a : ℝ) :
  A ∪ B a = A → a ≤ -1 ∨ a = 1 := by sorry

end NUMINAMATH_CALUDE_union_intersection_equality_union_subset_l672_67253


namespace NUMINAMATH_CALUDE_area_isosceles_right_triangle_radius_circumcircle_isosceles_right_triangle_l672_67233

/-- A right triangle with two equal angles and hypotenuse 8√2 -/
structure IsoscelesRightTriangle where
  /-- The length of the hypotenuse -/
  hypotenuse : ℝ
  /-- The hypotenuse is 8√2 -/
  hypotenuse_eq : hypotenuse = 8 * Real.sqrt 2

/-- The area of an isosceles right triangle with hypotenuse 8√2 is 32 -/
theorem area_isosceles_right_triangle (t : IsoscelesRightTriangle) :
    (1 / 2 : ℝ) * t.hypotenuse^2 / 2 = 32 := by sorry

/-- The radius of the circumcircle of an isosceles right triangle with hypotenuse 8√2 is 4√2 -/
theorem radius_circumcircle_isosceles_right_triangle (t : IsoscelesRightTriangle) :
    t.hypotenuse / 2 = 4 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_area_isosceles_right_triangle_radius_circumcircle_isosceles_right_triangle_l672_67233


namespace NUMINAMATH_CALUDE_parabola_max_vertex_sum_l672_67285

theorem parabola_max_vertex_sum (a S : ℤ) (h : S ≠ 0) :
  let parabola (x y : ℚ) := ∃ b c : ℚ, y = a * x^2 + b * x + c
  let passes_through (x y : ℚ) := parabola x y
  let vertex_sum := 
    let x₀ : ℚ := (3 * S : ℚ) / 2
    let y₀ : ℚ := -((9 * S^2 : ℚ) / 4) * a
    x₀ + y₀
  (passes_through 0 0) ∧ 
  (passes_through (3 * S) 0) ∧ 
  (passes_through (3 * S - 2) 35) →
  (∀ M : ℚ, vertex_sum ≤ M → M ≤ 1485/4)
  :=
by sorry

end NUMINAMATH_CALUDE_parabola_max_vertex_sum_l672_67285


namespace NUMINAMATH_CALUDE_sodium_bisulfite_moles_required_l672_67209

/-- Represents the balanced chemical equation for the reaction --/
structure ChemicalEquation :=
  (NaHSO3 : ℕ)
  (HCl : ℕ)
  (NaCl : ℕ)
  (H2O : ℕ)
  (SO2 : ℕ)

/-- The balanced equation for the reaction --/
def balanced_equation : ChemicalEquation :=
  { NaHSO3 := 1, HCl := 1, NaCl := 1, H2O := 1, SO2 := 1 }

/-- Theorem stating the number of moles of Sodium bisulfite required --/
theorem sodium_bisulfite_moles_required 
  (NaCl_produced : ℕ) 
  (HCl_used : ℕ) 
  (h1 : NaCl_produced = 2) 
  (h2 : HCl_used = 2) 
  (h3 : balanced_equation.NaHSO3 = balanced_equation.HCl) 
  (h4 : balanced_equation.NaHSO3 = balanced_equation.NaCl) :
  NaCl_produced = 2 := by
  sorry

end NUMINAMATH_CALUDE_sodium_bisulfite_moles_required_l672_67209


namespace NUMINAMATH_CALUDE_work_speed_ratio_l672_67217

-- Define the work speeds
def A_work_speed : ℚ := 1 / 18
def B_work_speed : ℚ := 1 / 36

-- Define the combined work speed
def combined_work_speed : ℚ := 1 / 12

-- Theorem statement
theorem work_speed_ratio :
  (A_work_speed + B_work_speed = combined_work_speed) →
  (A_work_speed / B_work_speed = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_work_speed_ratio_l672_67217


namespace NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_750_l672_67277

theorem least_integer_greater_than_sqrt_750 : ∃ n : ℕ, (n : ℝ) > Real.sqrt 750 ∧ ∀ m : ℕ, (m : ℝ) > Real.sqrt 750 → m ≥ n :=
sorry

end NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_750_l672_67277


namespace NUMINAMATH_CALUDE_numbers_satisfying_conditions_l672_67208

def ends_with_196 (n : ℕ) : Prop :=
  ∃ x : ℕ, n = 1000 * x + 196

def decreases_by_integer_factor (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 1 ∧ n / k = n - 196

def satisfies_conditions (n : ℕ) : Prop :=
  ends_with_196 n ∧ decreases_by_integer_factor n

theorem numbers_satisfying_conditions :
  {n : ℕ | satisfies_conditions n} = {1196, 2196, 4196, 7196, 14196, 49196, 98196} :=
by sorry

end NUMINAMATH_CALUDE_numbers_satisfying_conditions_l672_67208


namespace NUMINAMATH_CALUDE_shirley_cases_needed_l672_67210

-- Define the number of boxes sold
def boxes_sold : ℕ := 54

-- Define the number of boxes per case
def boxes_per_case : ℕ := 6

-- Define the number of cases needed
def cases_needed : ℕ := boxes_sold / boxes_per_case

-- Theorem statement
theorem shirley_cases_needed : cases_needed = 9 := by
  sorry

end NUMINAMATH_CALUDE_shirley_cases_needed_l672_67210


namespace NUMINAMATH_CALUDE_distribute_five_into_three_l672_67290

/-- The number of ways to distribute n distinct items into k identical bags -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 distinct items into 3 identical bags -/
theorem distribute_five_into_three : distribute 5 3 = 36 := by sorry

end NUMINAMATH_CALUDE_distribute_five_into_three_l672_67290


namespace NUMINAMATH_CALUDE_farm_heads_count_l672_67203

/-- Represents a farm with hens and cows -/
structure Farm where
  hens : ℕ
  cows : ℕ

/-- The total number of feet on the farm -/
def totalFeet (f : Farm) : ℕ := 2 * f.hens + 4 * f.cows

/-- The total number of heads on the farm -/
def totalHeads (f : Farm) : ℕ := f.hens + f.cows

/-- Theorem stating that a farm with 140 feet and 22 hens has 46 heads -/
theorem farm_heads_count (f : Farm) 
  (feet_count : totalFeet f = 140) 
  (hen_count : f.hens = 22) : 
  totalHeads f = 46 := by
  sorry

end NUMINAMATH_CALUDE_farm_heads_count_l672_67203


namespace NUMINAMATH_CALUDE_tan_negative_five_pi_sixths_l672_67211

theorem tan_negative_five_pi_sixths : 
  Real.tan (-5 * π / 6) = 1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_five_pi_sixths_l672_67211


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l672_67265

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-2) 0 → x^2 - a*x + a + 3 ≥ 0) → 
  a ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l672_67265


namespace NUMINAMATH_CALUDE_fifth_root_equation_solution_l672_67213

theorem fifth_root_equation_solution :
  ∃ x : ℝ, (x^(1/2) : ℝ) = 3 ∧ x^(1/2) = (x * (x^3)^(1/2))^(1/5) := by
  sorry

end NUMINAMATH_CALUDE_fifth_root_equation_solution_l672_67213


namespace NUMINAMATH_CALUDE_four_digit_integer_problem_l672_67240

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def middle_digit_sum (n : ℕ) : ℕ :=
  ((n / 100) % 10) + ((n / 10) % 10)

def thousands_minus_units (n : ℕ) : ℤ :=
  (n / 1000 : ℤ) - (n % 10 : ℤ)

theorem four_digit_integer_problem (n : ℕ) 
  (h1 : is_four_digit n)
  (h2 : digit_sum n = 18)
  (h3 : middle_digit_sum n = 11)
  (h4 : thousands_minus_units n = 1)
  (h5 : n % 11 = 0) :
  n = 4653 := by
sorry

end NUMINAMATH_CALUDE_four_digit_integer_problem_l672_67240


namespace NUMINAMATH_CALUDE_range_of_a_when_union_is_reals_l672_67231

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | -4 + a < x ∧ x < 4 + a}
def B : Set ℝ := {x : ℝ | x < -1 ∨ x > 5}

-- Theorem statement
theorem range_of_a_when_union_is_reals :
  ∀ a : ℝ, (A a ∪ B = Set.univ) ↔ (1 < a ∧ a < 3) := by sorry

end NUMINAMATH_CALUDE_range_of_a_when_union_is_reals_l672_67231


namespace NUMINAMATH_CALUDE_tangent_line_slope_l672_67299

theorem tangent_line_slope (k : ℝ) : 
  (∃ x : ℝ, k * x = x^3 - x^2 + x ∧ 
   k = 3 * x^2 - 2 * x + 1) → 
  k = 1 ∨ k = 3/4 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l672_67299


namespace NUMINAMATH_CALUDE_overall_gain_percentage_l672_67252

def item_a_cost : ℚ := 700
def item_b_cost : ℚ := 500
def item_c_cost : ℚ := 300
def item_a_gain : ℚ := 70
def item_b_gain : ℚ := 50
def item_c_gain : ℚ := 30

def total_cost : ℚ := item_a_cost + item_b_cost + item_c_cost
def total_gain : ℚ := item_a_gain + item_b_gain + item_c_gain

theorem overall_gain_percentage :
  (total_gain / total_cost) * 100 = 10 := by sorry

end NUMINAMATH_CALUDE_overall_gain_percentage_l672_67252


namespace NUMINAMATH_CALUDE_ratio_of_arithmetic_sequences_l672_67251

def arithmetic_sequence_sum (a₁ : ℚ) (d : ℚ) (l : ℚ) : ℚ :=
  let n := (l - a₁) / d + 1
  n / 2 * (a₁ + l)

theorem ratio_of_arithmetic_sequences :
  let seq1_sum := arithmetic_sequence_sum 3 3 96
  let seq2_sum := arithmetic_sequence_sum 4 4 64
  seq1_sum / seq2_sum = 99 / 34 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_arithmetic_sequences_l672_67251


namespace NUMINAMATH_CALUDE_average_temperature_l672_67271

def temperatures : List ℝ := [60, 59, 56, 53, 49, 48, 46]

theorem average_temperature : 
  (List.sum temperatures) / temperatures.length = 53 :=
by sorry

end NUMINAMATH_CALUDE_average_temperature_l672_67271


namespace NUMINAMATH_CALUDE_max_trailing_zeros_1003_sum_l672_67274

/-- Given three natural numbers that sum to 1003, the maximum number of trailing zeros in their product is 7. -/
theorem max_trailing_zeros_1003_sum (a b c : ℕ) (h_sum : a + b + c = 1003) :
  ∃ (n : ℕ), n ≤ 7 ∧
  ∀ (m : ℕ), (a * b * c) % (10^m) = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_max_trailing_zeros_1003_sum_l672_67274


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_two_l672_67200

theorem trigonometric_expression_equals_two :
  (Real.cos (10 * π / 180) + Real.sqrt 3 * Real.sin (10 * π / 180)) /
  Real.sqrt (1 - Real.sin (50 * π / 180) ^ 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_two_l672_67200


namespace NUMINAMATH_CALUDE_seven_arithmetic_to_hundred_l672_67298

theorem seven_arithmetic_to_hundred : (777 / 7) - (77 / 7) = 100 := by sorry

end NUMINAMATH_CALUDE_seven_arithmetic_to_hundred_l672_67298


namespace NUMINAMATH_CALUDE_product_of_specific_primes_l672_67291

theorem product_of_specific_primes : 
  let largest_one_digit_prime := 7
  let smallest_two_digit_prime1 := 11
  let smallest_two_digit_prime2 := 13
  largest_one_digit_prime * smallest_two_digit_prime1 * smallest_two_digit_prime2 = 1001 := by
sorry

end NUMINAMATH_CALUDE_product_of_specific_primes_l672_67291


namespace NUMINAMATH_CALUDE_total_cost_5_and_5_l672_67226

/-- The cost of a single room in yuan -/
def single_room_cost : ℝ := sorry

/-- The cost of a double room in yuan -/
def double_room_cost : ℝ := sorry

/-- The total cost of 3 single rooms and 6 double rooms is 1020 yuan -/
axiom cost_equation_1 : 3 * single_room_cost + 6 * double_room_cost = 1020

/-- The total cost of 1 single room and 5 double rooms is 700 yuan -/
axiom cost_equation_2 : single_room_cost + 5 * double_room_cost = 700

/-- The theorem states that the total cost of 5 single rooms and 5 double rooms is 1100 yuan -/
theorem total_cost_5_and_5 : 5 * single_room_cost + 5 * double_room_cost = 1100 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_5_and_5_l672_67226


namespace NUMINAMATH_CALUDE_largest_trick_number_l672_67214

/-- The constant k representing the number 2017 -/
def k : ℕ := 2017

/-- A function that determines whether the card trick can be performed for a given number of cards -/
def canPerformTrick (n : ℕ) : Prop :=
  n ≤ k + 1 ∧ (n ≤ k → False)

/-- Theorem stating that 2018 is the largest number for which the trick can be performed -/
theorem largest_trick_number : ∀ n : ℕ, canPerformTrick n ↔ n = k + 1 :=
  sorry

end NUMINAMATH_CALUDE_largest_trick_number_l672_67214


namespace NUMINAMATH_CALUDE_min_even_integers_l672_67235

theorem min_even_integers (a b c d e f : ℤ) : 
  a + b = 28 →
  a + b + c + d = 46 →
  a + b + c + d + e + f = 64 →
  ∃ (x y z w u v : ℤ), 
    x + y = 28 ∧
    x + y + z + w = 46 ∧
    x + y + z + w + u + v = 64 ∧
    Odd x ∧ Odd y ∧ Odd z ∧ Odd w ∧ Odd u ∧ Odd v :=
by sorry

end NUMINAMATH_CALUDE_min_even_integers_l672_67235


namespace NUMINAMATH_CALUDE_total_quarters_l672_67286

def initial_quarters : ℕ := 8
def additional_quarters : ℕ := 3

theorem total_quarters : initial_quarters + additional_quarters = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_quarters_l672_67286


namespace NUMINAMATH_CALUDE_geometry_biology_overlap_l672_67234

theorem geometry_biology_overlap (total : ℕ) (geometry : ℕ) (biology : ℕ)
  (h_total : total = 232)
  (h_geometry : geometry = 144)
  (h_biology : biology = 119) :
  min geometry biology - (geometry + biology - total) = 88 :=
by sorry

end NUMINAMATH_CALUDE_geometry_biology_overlap_l672_67234


namespace NUMINAMATH_CALUDE_remainder_of_large_number_l672_67205

theorem remainder_of_large_number (p : Nat) (h_prime : Nat.Prime p) :
  123456789012 ≡ 71 [MOD p] :=
by
  sorry

end NUMINAMATH_CALUDE_remainder_of_large_number_l672_67205


namespace NUMINAMATH_CALUDE_shoe_probabilities_l672_67278

-- Define the type for shoes
inductive Shoe
| left : Shoe
| right : Shoe

-- Define a pair of shoes
structure ShoePair :=
  (left : Shoe)
  (right : Shoe)

-- Define the cabinet with 3 pairs of shoes
def cabinet : Finset ShoePair := sorry

-- Define the sample space of choosing 2 shoes
def sampleSpace : Finset (Shoe × Shoe) := sorry

-- Event A: The taken out shoes do not form a pair
def eventA : Finset (Shoe × Shoe) := sorry

-- Event B: Both taken out shoes are for the same foot
def eventB : Finset (Shoe × Shoe) := sorry

-- Event C: One shoe is for the left foot and the other is for the right foot, but they do not form a pair
def eventC : Finset (Shoe × Shoe) := sorry

theorem shoe_probabilities :
  (Finset.card eventA : ℚ) / Finset.card sampleSpace = 4 / 5 ∧
  (Finset.card eventB : ℚ) / Finset.card sampleSpace = 2 / 5 ∧
  (Finset.card eventC : ℚ) / Finset.card sampleSpace = 2 / 5 :=
sorry

end NUMINAMATH_CALUDE_shoe_probabilities_l672_67278


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l672_67249

theorem equal_roots_quadratic (p : ℝ) : 
  (∃! p, ∀ x, x^2 - (p + 1) * x + p = 0 → (∃! x, x^2 - (p + 1) * x + p = 0)) :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l672_67249


namespace NUMINAMATH_CALUDE_no_five_circle_arrangement_l672_67297

-- Define a structure for a point in a plane
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a function to check if a point is the circumcenter of a triangle
def isCircumcenter (p : Point2D) (a b c : Point2D) : Prop :=
  (p.x - a.x)^2 + (p.y - a.y)^2 = (p.x - b.x)^2 + (p.y - b.y)^2 ∧
  (p.x - b.x)^2 + (p.y - b.y)^2 = (p.x - c.x)^2 + (p.y - c.y)^2

-- Theorem statement
theorem no_five_circle_arrangement :
  ¬ ∃ (p₁ p₂ p₃ p₄ p₅ : Point2D),
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧
    p₄ ≠ p₅ ∧
    (isCircumcenter p₁ p₂ p₃ p₄ ∨ isCircumcenter p₁ p₂ p₃ p₅ ∨ isCircumcenter p₁ p₂ p₄ p₅ ∨ isCircumcenter p₁ p₃ p₄ p₅) ∧
    (isCircumcenter p₂ p₁ p₃ p₄ ∨ isCircumcenter p₂ p₁ p₃ p₅ ∨ isCircumcenter p₂ p₁ p₄ p₅ ∨ isCircumcenter p₂ p₃ p₄ p₅) ∧
    (isCircumcenter p₃ p₁ p₂ p₄ ∨ isCircumcenter p₃ p₁ p₂ p₅ ∨ isCircumcenter p₃ p₁ p₄ p₅ ∨ isCircumcenter p₃ p₂ p₄ p₅) ∧
    (isCircumcenter p₄ p₁ p₂ p₃ ∨ isCircumcenter p₄ p₁ p₂ p₅ ∨ isCircumcenter p₄ p₁ p₃ p₅ ∨ isCircumcenter p₄ p₂ p₃ p₅) ∧
    (isCircumcenter p₅ p₁ p₂ p₃ ∨ isCircumcenter p₅ p₁ p₂ p₄ ∨ isCircumcenter p₅ p₁ p₃ p₄ ∨ isCircumcenter p₅ p₂ p₃ p₄) :=
by
  sorry


end NUMINAMATH_CALUDE_no_five_circle_arrangement_l672_67297


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l672_67275

/-- An isosceles trapezoid with given base lengths and perpendicular diagonals -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  perpendicular_diagonals : Prop

/-- The area of an isosceles trapezoid -/
def area (t : IsoscelesTrapezoid) : ℝ := sorry

/-- Theorem: The area of an isosceles trapezoid with bases 40 and 24, 
    and mutually perpendicular diagonals, is 1024 -/
theorem isosceles_trapezoid_area : 
  ∀ (t : IsoscelesTrapezoid), 
  t.base1 = 40 ∧ t.base2 = 24 ∧ t.perpendicular_diagonals → 
  area t = 1024 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l672_67275


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l672_67260

theorem sum_of_x_and_y (x y : ℝ) 
  (eq1 : x^2 + y^2 = 16*x - 10*y + 14) 
  (eq2 : x - y = 6) : 
  x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l672_67260


namespace NUMINAMATH_CALUDE_parallel_implies_m_half_perpendicular_implies_m_seven_fourths_l672_67292

-- Define the vectors
def OA : ℝ × ℝ := (3, -4)
def OB : ℝ × ℝ := (6, -3)
def OC (m : ℝ) : ℝ × ℝ := (5 - m, -3 - m)

-- Define vector operations
def vector_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

def AB : ℝ × ℝ := vector_sub OB OA
def BC (m : ℝ) : ℝ × ℝ := vector_sub (OC m) OB
def AC (m : ℝ) : ℝ × ℝ := vector_sub (OC m) OA

-- Define parallel and perpendicular conditions
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- State the theorems
theorem parallel_implies_m_half :
  parallel AB (BC (1/2)) :=
sorry

theorem perpendicular_implies_m_seven_fourths :
  perpendicular AB (AC (7/4)) :=
sorry

end NUMINAMATH_CALUDE_parallel_implies_m_half_perpendicular_implies_m_seven_fourths_l672_67292


namespace NUMINAMATH_CALUDE_sum_even_implies_one_even_l672_67218

theorem sum_even_implies_one_even (a b c : ℕ) :
  Even (a + b + c) → (Even a ∨ Even b ∨ Even c) :=
sorry

end NUMINAMATH_CALUDE_sum_even_implies_one_even_l672_67218


namespace NUMINAMATH_CALUDE_gcd_and_sum_divisibility_l672_67269

theorem gcd_and_sum_divisibility : 
  (Nat.gcd 42558 29791 = 3) ∧ 
  ¬(72349 % 3 = 0) := by sorry

end NUMINAMATH_CALUDE_gcd_and_sum_divisibility_l672_67269


namespace NUMINAMATH_CALUDE_segment_length_l672_67250

/-- Given a line segment CD with points R and S on it, prove that CD has length 146.2/11 -/
theorem segment_length (C D R S : ℝ) : 
  (R > C) →  -- R is to the right of C
  (S > R) →  -- S is to the right of R
  (D > S) →  -- D is to the right of S
  (R - C) / (D - R) = 3 / 5 →  -- R divides CD in ratio 3:5
  (S - C) / (D - S) = 4 / 7 →  -- S divides CD in ratio 4:7
  S - R = 1 →  -- RS = 1
  D - C = 146.2 / 11 := by
sorry

end NUMINAMATH_CALUDE_segment_length_l672_67250


namespace NUMINAMATH_CALUDE_reflect_point_coords_l672_67206

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a function to reflect a point across the xz-plane
def reflectAcrossXZPlane (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

-- Theorem statement
theorem reflect_point_coords :
  let original := Point3D.mk (-4) 3 5
  let reflected := reflectAcrossXZPlane original
  reflected.x = 4 ∧ reflected.y = -3 ∧ reflected.z = 5 := by
  sorry


end NUMINAMATH_CALUDE_reflect_point_coords_l672_67206


namespace NUMINAMATH_CALUDE_tangent_point_coordinates_l672_67263

/-- A point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  h : y = x^2 - 2*x - 3

/-- Predicate for a circle being tangent to x-axis or y-axis -/
def is_tangent_to_axis (p : ParabolaPoint) : Prop :=
  (p.y = 2 ∨ p.y = -2) ∨ (p.x = 2 ∨ p.x = -2)

/-- The set of points where the circle is tangent to an axis -/
def tangent_points : Set ParabolaPoint :=
  { p | is_tangent_to_axis p }

/-- Theorem stating the coordinates of tangent points -/
theorem tangent_point_coordinates :
  ∀ p ∈ tangent_points,
    (p.x = 1 + Real.sqrt 6 ∧ p.y = 2) ∨
    (p.x = 1 - Real.sqrt 6 ∧ p.y = 2) ∨
    (p.x = 1 + Real.sqrt 2 ∧ p.y = -2) ∨
    (p.x = 1 - Real.sqrt 2 ∧ p.y = -2) ∨
    (p.x = 2 ∧ p.y = -3) ∨
    (p.x = -2 ∧ p.y = 5) :=
  sorry

end NUMINAMATH_CALUDE_tangent_point_coordinates_l672_67263


namespace NUMINAMATH_CALUDE_ln_ratio_monotone_l672_67222

open Real

theorem ln_ratio_monotone (a b c : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < 1) :
  (log a) / a < (log b) / b ∧ (log b) / b < (log c) / c :=
by sorry

end NUMINAMATH_CALUDE_ln_ratio_monotone_l672_67222


namespace NUMINAMATH_CALUDE_sixth_grade_boys_l672_67255

theorem sixth_grade_boys (total_students : ℕ) (boys : ℕ) : 
  total_students = 152 →
  boys * 10 = (total_students - boys - 5) * 11 →
  boys = 77 := by
sorry

end NUMINAMATH_CALUDE_sixth_grade_boys_l672_67255


namespace NUMINAMATH_CALUDE_price_changes_l672_67229

theorem price_changes (original_price : ℝ) : 
  let price_after_first_increase := original_price * 1.2
  let price_after_second_increase := price_after_first_increase + 5
  let price_after_first_decrease := price_after_second_increase * 0.8
  let final_price := price_after_first_decrease - 5
  final_price = 120 → original_price = 126.04 := by
sorry

#eval (121 / 0.96 : Float)

end NUMINAMATH_CALUDE_price_changes_l672_67229


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l672_67215

theorem complementary_angles_difference (a b : ℝ) : 
  a + b = 90 →  -- angles are complementary
  a / b = 4 / 5 →  -- ratio of angles is 4:5
  |a - b| = 10 :=  -- positive difference is 10°
by sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l672_67215


namespace NUMINAMATH_CALUDE_smallest_positive_angle_l672_67261

theorem smallest_positive_angle (α : Real) : 
  let P : Real × Real := (Real.sin (2 * Real.pi / 3), Real.cos (2 * Real.pi / 3))
  (∃ t : Real, t > 0 ∧ P.1 = t * Real.sin α ∧ P.2 = t * Real.cos α) →
  (∀ β : Real, β > 0 ∧ (∃ s : Real, s > 0 ∧ P.1 = s * Real.sin β ∧ P.2 = s * Real.cos β) → α ≤ β) →
  α = 11 * Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_l672_67261


namespace NUMINAMATH_CALUDE_dot_path_length_on_rolling_cube_l672_67288

/-- The path length traced by a dot on a rolling cube. -/
theorem dot_path_length_on_rolling_cube : 
  ∀ (cube_edge_length : ℝ) (dot_distance_from_edge : ℝ),
    cube_edge_length = 2 →
    dot_distance_from_edge = 1 →
    ∃ (path_length : ℝ),
      path_length = 2 * Real.pi * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_dot_path_length_on_rolling_cube_l672_67288


namespace NUMINAMATH_CALUDE_speed_difference_calc_l672_67270

/-- Calculates the speed difference between return and outbound trips --/
theorem speed_difference_calc (outbound_time outbound_speed return_time : ℝ) 
  (h1 : outbound_time = 6)
  (h2 : outbound_speed = 60)
  (h3 : return_time = 5)
  (h4 : outbound_time * outbound_speed = return_time * (outbound_speed + speed_diff)) :
  speed_diff = 12 := by
  sorry

#check speed_difference_calc

end NUMINAMATH_CALUDE_speed_difference_calc_l672_67270


namespace NUMINAMATH_CALUDE_combination_distinctness_and_divisor_count_l672_67246

theorem combination_distinctness_and_divisor_count (n : ℕ) (hn : n > 3) :
  -- Part (a)
  (∀ x y z : ℕ, x > n / 2 → y > n / 2 → z > n / 2 → x < y → y < z → z ≤ n →
    (let exprs := [x + y + z, x + y * z, x * y + z, y + z * x, (x + y) * z, (z + x) * y, (y + z) * x, x * y * z]
     exprs.Pairwise (·≠·))) ∧
  -- Part (b)
  (∀ p : ℕ, Nat.Prime p → p ≤ Real.sqrt n →
    (Finset.filter (fun i => i > 1 ∧ (p - 1) % i = 0) (Finset.range (p - 1))).card =
    (Finset.filter (fun pair : ℕ × ℕ =>
      let (y, z) := pair
      p < y ∧ y < z ∧ z ≤ n ∧
      ¬(let exprs := [p + y + z, p + y * z, p * y + z, y + z * p, (p + y) * z, (z + p) * y, (y + z) * p, p * y * z]
        exprs.Pairwise (·≠·)))
     (Finset.product (Finset.range (n + 1)) (Finset.range (n + 1)))).card) :=
by sorry

end NUMINAMATH_CALUDE_combination_distinctness_and_divisor_count_l672_67246


namespace NUMINAMATH_CALUDE_no_solution_when_x_is_five_l672_67289

theorem no_solution_when_x_is_five (x : ℝ) (y : ℝ) :
  x = 5 → ¬∃y, 1 / (x + 5) + y = 1 / (x - 5) := by
sorry

end NUMINAMATH_CALUDE_no_solution_when_x_is_five_l672_67289


namespace NUMINAMATH_CALUDE_cricket_average_l672_67212

theorem cricket_average (innings : ℕ) (next_runs : ℕ) (increase : ℕ) (initial_average : ℕ) : 
  innings = 20 →
  next_runs = 200 →
  increase = 8 →
  (innings * initial_average + next_runs) / (innings + 1) = initial_average + increase →
  initial_average = 32 := by
  sorry

end NUMINAMATH_CALUDE_cricket_average_l672_67212


namespace NUMINAMATH_CALUDE_equal_sprocket_production_l672_67220

/-- Represents the production rates and times of two machines manufacturing sprockets -/
structure SprocketProduction where
  machine_a_rate : ℝ  -- Sprockets per hour for Machine A
  machine_b_rate : ℝ  -- Sprockets per hour for Machine B
  machine_b_time : ℝ  -- Time taken by Machine B in hours

/-- Theorem stating that both machines produce the same number of sprockets -/
theorem equal_sprocket_production (sp : SprocketProduction) 
  (h1 : sp.machine_a_rate = 4)  -- Machine A produces 4 sprockets per hour
  (h2 : sp.machine_b_rate = sp.machine_a_rate * 1.1)  -- Machine B is 10% faster
  (h3 : sp.machine_b_time * sp.machine_b_rate = (sp.machine_b_time + 10) * sp.machine_a_rate)  -- Total production is equal
  : sp.machine_a_rate * (sp.machine_b_time + 10) = 440 ∧ sp.machine_b_rate * sp.machine_b_time = 440 :=
by sorry

end NUMINAMATH_CALUDE_equal_sprocket_production_l672_67220


namespace NUMINAMATH_CALUDE_log_equality_l672_67224

theorem log_equality : Real.log 81 / Real.log 4 = Real.log 9 / Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_log_equality_l672_67224


namespace NUMINAMATH_CALUDE_y_intercept_of_specific_line_l672_67279

/-- A line in the xy-plane is defined by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line is the y-coordinate where the line intersects the y-axis. -/
def y_intercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

/-- 
Given a line with slope 4 passing through the point (199, 800),
prove that its y-intercept is 4.
-/
theorem y_intercept_of_specific_line :
  let l : Line := { slope := 4, point := (199, 800) }
  y_intercept l = 4 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_specific_line_l672_67279


namespace NUMINAMATH_CALUDE_sqrt_6_bounds_l672_67262

theorem sqrt_6_bounds : 2 < Real.sqrt 6 ∧ Real.sqrt 6 < 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_6_bounds_l672_67262


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l672_67204

theorem quadratic_inequality_solution :
  {z : ℝ | z^2 - 40*z + 340 ≤ 4} = Set.Icc 12 28 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l672_67204


namespace NUMINAMATH_CALUDE_carpet_shaded_area_l672_67257

/-- Calculates the total shaded area of a carpet design with given ratios and square counts. -/
theorem carpet_shaded_area (carpet_side : ℝ) (ratio_12_S : ℝ) (ratio_S_T : ℝ) (ratio_T_U : ℝ)
  (count_S : ℕ) (count_T : ℕ) (count_U : ℕ) :
  carpet_side = 12 →
  ratio_12_S = 4 →
  ratio_S_T = 2 →
  ratio_T_U = 2 →
  count_S = 1 →
  count_T = 4 →
  count_U = 8 →
  let S := carpet_side / ratio_12_S
  let T := S / ratio_S_T
  let U := T / ratio_T_U
  count_S * S^2 + count_T * T^2 + count_U * U^2 = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_carpet_shaded_area_l672_67257


namespace NUMINAMATH_CALUDE_triangle_area_at_most_half_parallelogram_l672_67284

-- Define a parallelogram
structure Parallelogram where
  area : ℝ
  area_pos : area > 0

-- Define a triangle inscribed in the parallelogram
structure InscribedTriangle (p : Parallelogram) where
  area : ℝ
  area_pos : area > 0
  inscribed : True  -- This represents that the triangle is inscribed in the parallelogram

-- Theorem statement
theorem triangle_area_at_most_half_parallelogram (p : Parallelogram) (t : InscribedTriangle p) :
  t.area ≤ p.area / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_at_most_half_parallelogram_l672_67284


namespace NUMINAMATH_CALUDE_turtle_race_ratio_l672_67202

theorem turtle_race_ratio : 
  ∀ (greta_time george_time gloria_time : ℕ),
    greta_time = 6 →
    george_time = greta_time - 2 →
    gloria_time = 8 →
    (gloria_time : ℚ) / (george_time : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_turtle_race_ratio_l672_67202


namespace NUMINAMATH_CALUDE_count_triangles_including_center_l672_67243

/-- Given a regular polygon with 2n + 1 sides, this function calculates the number of triangles
    formed by its vertices that include the center of the polygon. -/
def trianglesIncludingCenter (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

/-- Theorem stating that the number of triangles including the center of a regular polygon
    with 2n + 1 sides is equal to n(n+1)(2n+1)/6 -/
theorem count_triangles_including_center (n : ℕ) :
  trianglesIncludingCenter n = n * (n + 1) * (2 * n + 1) / 6 := by
  sorry

end NUMINAMATH_CALUDE_count_triangles_including_center_l672_67243


namespace NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l672_67236

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if three natural numbers form a scalene triangle -/
def isScalene (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- A function that checks if three natural numbers satisfy the triangle inequality -/
def satisfiesTriangleInequality (a b c : ℕ) : Prop := 
  a + b > c ∧ b + c > a ∧ c + a > b

theorem smallest_prime_perimeter_scalene_triangle : 
  ∃ (a b c : ℕ), 
    a ≥ 11 ∧ b ≥ 11 ∧ c ≥ 11 ∧
    isPrime a ∧ isPrime b ∧ isPrime c ∧
    isScalene a b c ∧
    satisfiesTriangleInequality a b c ∧
    isPrime (a + b + c) ∧
    (a + b + c = 41) ∧
    (∀ (x y z : ℕ), 
      x ≥ 11 ∧ y ≥ 11 ∧ z ≥ 11 →
      isPrime x ∧ isPrime y ∧ isPrime z →
      isScalene x y z →
      satisfiesTriangleInequality x y z →
      isPrime (x + y + z) →
      x + y + z ≥ 41) := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l672_67236


namespace NUMINAMATH_CALUDE_relationship_abc_l672_67283

theorem relationship_abc : 3^(1/10) > (1/2)^(1/10) ∧ (1/2)^(1/10) > (-1/2)^3 := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l672_67283


namespace NUMINAMATH_CALUDE_circle_center_sum_l672_67273

theorem circle_center_sum (h k : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 6*x + 8*y - 15 ↔ (x - h)^2 + (y - k)^2 = 10) →
  h + k = 7 := by
sorry

end NUMINAMATH_CALUDE_circle_center_sum_l672_67273


namespace NUMINAMATH_CALUDE_cube_rect_surface_area_ratio_l672_67227

theorem cube_rect_surface_area_ratio (a b : ℝ) (h : a > 0) :
  2 * a^2 + 4 * a * b = 0.6 * (6 * a^2) → b = 0.6 * a := by
  sorry

end NUMINAMATH_CALUDE_cube_rect_surface_area_ratio_l672_67227


namespace NUMINAMATH_CALUDE_moon_speed_conversion_l672_67207

/-- Converts a speed from kilometers per second to kilometers per hour -/
def km_per_second_to_km_per_hour (speed_km_per_second : ℝ) : ℝ :=
  speed_km_per_second * 3600

/-- The moon's speed in kilometers per second -/
def moon_speed_km_per_second : ℝ := 0.9

theorem moon_speed_conversion :
  km_per_second_to_km_per_hour moon_speed_km_per_second = 3240 := by
  sorry

end NUMINAMATH_CALUDE_moon_speed_conversion_l672_67207


namespace NUMINAMATH_CALUDE_tournament_team_b_matches_l672_67281

/-- Represents a team in the tournament -/
structure Team where
  city : Fin 16
  type : Bool -- false for A, true for B

/-- The tournament setup -/
structure Tournament where
  teams : Fin 32 → Team
  matches_played : Fin 32 → Nat
  different_matches : ∀ i j, i ≠ j → matches_played i ≠ matches_played j ∨ (teams i).city = 0 ∧ (teams i).type = false
  no_self_city_matches : ∀ i j, (teams i).city = (teams j).city → matches_played i + matches_played j ≤ 30
  max_one_match : ∀ i, matches_played i ≤ 30

theorem tournament_team_b_matches (t : Tournament) : 
  ∃ i, (t.teams i).city = 0 ∧ (t.teams i).type = true ∧ t.matches_played i = 15 :=
sorry

end NUMINAMATH_CALUDE_tournament_team_b_matches_l672_67281


namespace NUMINAMATH_CALUDE_original_number_proof_l672_67247

theorem original_number_proof (x : ℝ) (h : x * 1.25 = 250) : x = 200 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l672_67247


namespace NUMINAMATH_CALUDE_loan_B_is_5000_l672_67266

/-- Represents the loan details and interest calculation --/
structure LoanDetails where
  rate : ℚ  -- Interest rate per annum
  time_B : ℕ  -- Time for loan B in years
  time_C : ℕ  -- Time for loan C in years
  amount_C : ℚ  -- Amount lent to C
  total_interest : ℚ  -- Total interest received from both loans

/-- Calculates the amount lent to B given the loan details --/
def calculate_loan_B (loan : LoanDetails) : ℚ :=
  (loan.total_interest - loan.amount_C * loan.rate * loan.time_C) / (loan.rate * loan.time_B)

/-- Theorem stating that the amount lent to B is 5000 --/
theorem loan_B_is_5000 (loan : LoanDetails) 
  (h1 : loan.rate = 9 / 100)
  (h2 : loan.time_B = 2)
  (h3 : loan.time_C = 4)
  (h4 : loan.amount_C = 3000)
  (h5 : loan.total_interest = 1980) :
  calculate_loan_B loan = 5000 := by
  sorry

#eval calculate_loan_B { rate := 9/100, time_B := 2, time_C := 4, amount_C := 3000, total_interest := 1980 }

end NUMINAMATH_CALUDE_loan_B_is_5000_l672_67266


namespace NUMINAMATH_CALUDE_g_equiv_l672_67282

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 5

-- Define the function g using f
def g (x : ℝ) : ℝ := 2 * (f x) - 19

-- Theorem stating that g(x) is equivalent to 6x - 29
theorem g_equiv : ∀ x : ℝ, g x = 6 * x - 29 := by
  sorry

end NUMINAMATH_CALUDE_g_equiv_l672_67282


namespace NUMINAMATH_CALUDE_cos_sqrt3_over_2_necessary_not_sufficient_l672_67241

theorem cos_sqrt3_over_2_necessary_not_sufficient (α : ℝ) :
  (∃ k : ℤ, α = 2 * k * π + 5 * π / 6 → Real.cos α = -Real.sqrt 3 / 2) ∧
  (∃ α : ℝ, Real.cos α = -Real.sqrt 3 / 2 ∧ ∀ k : ℤ, α ≠ 2 * k * π + 5 * π / 6) :=
by sorry

end NUMINAMATH_CALUDE_cos_sqrt3_over_2_necessary_not_sufficient_l672_67241


namespace NUMINAMATH_CALUDE_associate_prof_charts_l672_67258

theorem associate_prof_charts (
  associate_profs : ℕ) 
  (assistant_profs : ℕ) 
  (charts_per_associate : ℕ) 
  (h1 : associate_profs + assistant_profs = 7)
  (h2 : 2 * associate_profs + assistant_profs = 10)
  (h3 : charts_per_associate * associate_profs + 2 * assistant_profs = 11)
  : charts_per_associate = 1 := by
  sorry

end NUMINAMATH_CALUDE_associate_prof_charts_l672_67258


namespace NUMINAMATH_CALUDE_fedya_statement_possible_l672_67245

/-- Represents a date with year, month, and day -/
structure Date where
  year : ℕ
  month : ℕ
  day : ℕ

/-- Represents Fedya's age on a given date -/
def age (birthdate : Date) (currentDate : Date) : ℕ := sorry

/-- Returns the date one year after the given date -/
def nextYear (d : Date) : Date := sorry

/-- Returns the date two days before the given date -/
def twoDaysAgo (d : Date) : Date := sorry

/-- Theorem stating that Fedya's statement could be true -/
theorem fedya_statement_possible : ∃ (birthdate currentDate : Date),
  age birthdate (twoDaysAgo currentDate) = 10 ∧
  age birthdate (nextYear currentDate) = 13 :=
sorry

end NUMINAMATH_CALUDE_fedya_statement_possible_l672_67245


namespace NUMINAMATH_CALUDE_problem_statement_l672_67296

theorem problem_statement (x y : ℝ) (hx : x = 3) (hy : y = 2) : 3 * x - 4 * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l672_67296


namespace NUMINAMATH_CALUDE_bucket_weight_bucket_weight_proof_l672_67230

/-- Given a bucket with the following properties:
  1. When three-quarters full of water, it weighs c kilograms (including the water).
  2. When one-third full of water, it weighs d kilograms (including the water).
  This theorem states that when the bucket is completely full of water, 
  its total weight is (8/5)c - (7/5)d kilograms. -/
theorem bucket_weight (c d : ℝ) : ℝ :=
  let three_quarters_full := c
  let one_third_full := d
  let full_weight := (8/5) * c - (7/5) * d
  full_weight

/-- Proof of the bucket_weight theorem -/
theorem bucket_weight_proof (c d : ℝ) : 
  bucket_weight c d = (8/5) * c - (7/5) * d :=
by sorry

end NUMINAMATH_CALUDE_bucket_weight_bucket_weight_proof_l672_67230


namespace NUMINAMATH_CALUDE_line_perp_plane_implies_planes_perp_l672_67225

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (containedIn : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (planePerpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_plane_implies_planes_perp
  (m n : Line) (α β : Plane)
  (h1 : m ≠ n)
  (h2 : α ≠ β)
  (h3 : containedIn m α)
  (h4 : containedIn n β)
  (h5 : perpendicular n α) :
  planePerpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_implies_planes_perp_l672_67225


namespace NUMINAMATH_CALUDE_sqrt_product_equals_24_l672_67293

theorem sqrt_product_equals_24 (x : ℝ) (h_pos : x > 0) 
  (h_eq : Real.sqrt (12 * x) * Real.sqrt (18 * x) * Real.sqrt (6 * x) * Real.sqrt (24 * x) = 24) : 
  x = Real.sqrt (3 / 22) := by
sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_24_l672_67293


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_one_l672_67295

theorem at_least_one_not_less_than_one (x : ℝ) :
  let a := |Real.cos x - Real.sin x|
  let b := |Real.cos x + Real.sin x|
  max a b ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_one_l672_67295


namespace NUMINAMATH_CALUDE_inscribed_trapezoid_median_l672_67276

-- Define the trapezoid and its properties
structure InscribedTrapezoid where
  radius : ℝ
  baseAngle : ℝ
  leg : ℝ

-- Define the median (midsegment) of the trapezoid
def median (t : InscribedTrapezoid) : ℝ :=
  sorry

-- Theorem statement
theorem inscribed_trapezoid_median
  (t : InscribedTrapezoid)
  (h1 : t.radius = 13)
  (h2 : t.baseAngle = 30 * π / 180)  -- Convert degrees to radians
  (h3 : t.leg = 10) :
  median t = 12 :=
sorry

end NUMINAMATH_CALUDE_inscribed_trapezoid_median_l672_67276


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l672_67248

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y) / z + (x + z) / y + (y + z) / x + (x + y + z) / (x + y) ≥ 7.5 :=
sorry

theorem min_value_achievable :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  (x + y) / z + (x + z) / y + (y + z) / x + (x + y + z) / (x + y) = 7.5 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l672_67248


namespace NUMINAMATH_CALUDE_new_student_weight_l672_67201

theorem new_student_weight (initial_count : ℕ) (replaced_weight : ℝ) (average_decrease : ℝ) :
  initial_count = 5 →
  replaced_weight = 72 →
  average_decrease = 12 →
  let new_weight := replaced_weight - initial_count * average_decrease
  new_weight = 12 := by
  sorry

end NUMINAMATH_CALUDE_new_student_weight_l672_67201


namespace NUMINAMATH_CALUDE_greatest_common_factor_3465_10780_l672_67232

theorem greatest_common_factor_3465_10780 : Nat.gcd 3465 10780 = 385 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_factor_3465_10780_l672_67232


namespace NUMINAMATH_CALUDE_sum_equals_rounded_sum_l672_67272

def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_five (x : ℕ) : ℕ :=
  let remainder := x % 5
  if remainder < 3 then x - remainder else x + (5 - remainder)

def sum_rounded_to_five (n : ℕ) : ℕ :=
  List.range n |> List.map (λ x => round_to_nearest_five (x + 1)) |> List.sum

theorem sum_equals_rounded_sum (n : ℕ) : sum_to_n n = sum_rounded_to_five n := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_rounded_sum_l672_67272


namespace NUMINAMATH_CALUDE_stating_equation_is_quadratic_l672_67237

/-- 
Theorem stating that when a = 3, the equation 3x^(a-1) - x = 5 is quadratic in x.
-/
theorem equation_is_quadratic (x : ℝ) : 
  let a : ℝ := 3
  let f : ℝ → ℝ := λ x => 3 * x^(a - 1) - x - 5
  ∃ (p q r : ℝ), f x = p * x^2 + q * x + r := by
  sorry

end NUMINAMATH_CALUDE_stating_equation_is_quadratic_l672_67237


namespace NUMINAMATH_CALUDE_border_area_l672_67242

/-- The area of a border around a rectangular picture --/
theorem border_area (picture_height picture_width border_width : ℝ) : 
  picture_height = 12 →
  picture_width = 15 →
  border_width = 3 →
  (picture_height + 2 * border_width) * (picture_width + 2 * border_width) - picture_height * picture_width = 198 := by
  sorry

end NUMINAMATH_CALUDE_border_area_l672_67242


namespace NUMINAMATH_CALUDE_order_of_abc_l672_67294

noncomputable def a : ℝ := (1/2)^(1/3)
noncomputable def b : ℝ := Real.log 2 / Real.log (1/3)
noncomputable def c : ℝ := 1 / Real.sin 1

theorem order_of_abc : c > a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l672_67294


namespace NUMINAMATH_CALUDE_ticket_difference_l672_67221

theorem ticket_difference (initial_tickets : ℕ) (remaining_tickets : ℕ) : 
  initial_tickets = 48 → remaining_tickets = 32 → initial_tickets - remaining_tickets = 16 := by
  sorry

end NUMINAMATH_CALUDE_ticket_difference_l672_67221


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_product_l672_67219

theorem absolute_value_equation_solution_product : 
  (∀ x : ℝ, |x - 5| + 4 = 7 → x = 8 ∨ x = 2) ∧ 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ |x₁ - 5| + 4 = 7 ∧ |x₂ - 5| + 4 = 7) ∧
  (∀ x₁ x₂ : ℝ, |x₁ - 5| + 4 = 7 → |x₂ - 5| + 4 = 7 → x₁ * x₂ = 16) := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_product_l672_67219
