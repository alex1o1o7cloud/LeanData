import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l1805_180519

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ m * x^2 + 2 * x - 1 = 0 ∧ m * y^2 + 2 * y - 1 = 0) ↔ 
  (m > -1 ∧ m ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l1805_180519


namespace NUMINAMATH_CALUDE_billy_sam_money_multiple_l1805_180589

/-- Given that Sam has $75 and Billy has $25 less than a multiple of Sam's money,
    and together they have $200, prove that the multiple is 2. -/
theorem billy_sam_money_multiple : 
  ∀ (sam_money : ℕ) (total_money : ℕ) (multiple : ℚ),
    sam_money = 75 →
    total_money = 200 →
    total_money = sam_money + (multiple * sam_money - 25) →
    multiple = 2 := by
  sorry

end NUMINAMATH_CALUDE_billy_sam_money_multiple_l1805_180589


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l1805_180551

theorem lcm_hcf_problem (a b : ℕ+) : 
  Nat.lcm a b = 2310 →
  Nat.gcd a b = 30 →
  b = 330 →
  a = 210 := by
sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l1805_180551


namespace NUMINAMATH_CALUDE_xyz_inequality_l1805_180534

theorem xyz_inequality (x y z : ℝ) (n : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) (h_sum : x + y + z = 1) (h_pos_n : n > 0) :
  x^n + y^n + z^n ≥ 1 / 3^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_xyz_inequality_l1805_180534


namespace NUMINAMATH_CALUDE_consecutive_points_length_l1805_180581

/-- Given 6 consecutive points on a straight line, prove that af = 25 -/
theorem consecutive_points_length (a b c d e f : ℝ) : 
  (c - b) = 3 * (d - c) →
  (e - d) = 8 →
  (b - a) = 5 →
  (c - a) = 11 →
  (f - e) = 4 →
  (f - a) = 25 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_points_length_l1805_180581


namespace NUMINAMATH_CALUDE_mod_eight_equivalence_l1805_180557

theorem mod_eight_equivalence :
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 7 ∧ n ≡ -4850 [ZMOD 8] ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_mod_eight_equivalence_l1805_180557


namespace NUMINAMATH_CALUDE_problem_solution_l1805_180546

theorem problem_solution (a b : ℝ) (h1 : b - a = -6) (h2 : a * b = 7) :
  a^2 * b - a * b^2 = -42 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1805_180546


namespace NUMINAMATH_CALUDE_min_difference_of_roots_l1805_180535

noncomputable def f (t : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sqrt x - t else 2 * (x + 1) - t

theorem min_difference_of_roots (t : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ > x₂ ∧ f t x₁ = 0 ∧ f t x₂ = 0 →
  ∃ min_diff : ℝ, (∀ y₁ y₂ : ℝ, y₁ > y₂ → f t y₁ = 0 → f t y₂ = 0 → y₁ - y₂ ≥ min_diff) ∧
               min_diff = 15/16 :=
sorry

end NUMINAMATH_CALUDE_min_difference_of_roots_l1805_180535


namespace NUMINAMATH_CALUDE_duck_cow_problem_l1805_180547

theorem duck_cow_problem (D C : ℕ) : 
  (2 * D + 4 * C = 2 * (D + C) + 40) → C = 20 := by
  sorry

end NUMINAMATH_CALUDE_duck_cow_problem_l1805_180547


namespace NUMINAMATH_CALUDE_kennel_arrangement_count_l1805_180500

/-- The number of chickens in the kennel -/
def num_chickens : Nat := 4

/-- The number of dogs in the kennel -/
def num_dogs : Nat := 3

/-- The number of cats in the kennel -/
def num_cats : Nat := 5

/-- The total number of animals in the kennel -/
def total_animals : Nat := num_chickens + num_dogs + num_cats

/-- The number of ways to arrange animals within their groups -/
def intra_group_arrangements : Nat := (Nat.factorial num_chickens) * (Nat.factorial num_dogs) * (Nat.factorial num_cats)

/-- The number of valid group orders (chickens-dogs-cats and chickens-cats-dogs) -/
def valid_group_orders : Nat := 2

/-- The total number of ways to arrange the animals -/
def total_arrangements : Nat := valid_group_orders * intra_group_arrangements

theorem kennel_arrangement_count :
  total_arrangements = 34560 :=
sorry

end NUMINAMATH_CALUDE_kennel_arrangement_count_l1805_180500


namespace NUMINAMATH_CALUDE_max_value_quadratic_l1805_180576

theorem max_value_quadratic (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 2, x^2 - a*x - a ≤ 1) ∧ 
  (∃ x ∈ Set.Icc 0 2, x^2 - a*x - a = 1) → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l1805_180576


namespace NUMINAMATH_CALUDE_travel_options_count_l1805_180505

/-- The number of flights from A to B in one day -/
def num_flights : ℕ := 3

/-- The number of trains from A to B in one day -/
def num_trains : ℕ := 2

/-- The total number of ways to travel from A to B in one day -/
def total_ways : ℕ := num_flights + num_trains

theorem travel_options_count : total_ways = 5 := by sorry

end NUMINAMATH_CALUDE_travel_options_count_l1805_180505


namespace NUMINAMATH_CALUDE_rain_forest_animals_l1805_180530

theorem rain_forest_animals (reptile_house : ℕ) (rain_forest : ℕ) : 
  reptile_house = 16 → 
  reptile_house = 3 * rain_forest - 5 → 
  rain_forest = 7 := by
sorry

end NUMINAMATH_CALUDE_rain_forest_animals_l1805_180530


namespace NUMINAMATH_CALUDE_k_is_even_if_adjacent_to_odds_l1805_180597

/-- A circular arrangement of numbers from 1 to 1000 -/
def CircularArrangement := Fin 1000 → ℕ

/-- Property that each number is a divisor of the sum of its neighbors -/
def IsDivisorOfNeighborsSum (arr : CircularArrangement) : Prop :=
  ∀ i : Fin 1000, arr i ∣ (arr (i - 1) + arr (i + 1))

/-- Theorem: If k is adjacent to two odd numbers in a valid circular arrangement, then k is even -/
theorem k_is_even_if_adjacent_to_odds
  (arr : CircularArrangement)
  (h_valid : IsDivisorOfNeighborsSum arr)
  (k : Fin 1000)
  (h_k_adj_odd : Odd (arr (k - 1)) ∧ Odd (arr (k + 1))) :
  Even (arr k) := by
  sorry

end NUMINAMATH_CALUDE_k_is_even_if_adjacent_to_odds_l1805_180597


namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l1805_180549

theorem ceiling_neg_sqrt_64_over_9 : ⌈-Real.sqrt (64/9)⌉ = -2 := by sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l1805_180549


namespace NUMINAMATH_CALUDE_other_number_proof_l1805_180574

theorem other_number_proof (x : ℤ) (h : x + 2001 = 3016) : x = 1015 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l1805_180574


namespace NUMINAMATH_CALUDE_race_distance_l1805_180598

/-- 
Given a race with two contestants A and B, where:
- The ratio of speeds of A and B is 3:4
- A has a start of 140 meters
- A wins by 20 meters

Prove that the total distance of the race is 480 meters.
-/
theorem race_distance (speed_A speed_B : ℝ) (total_distance : ℝ) : 
  speed_A / speed_B = 3 / 4 →
  total_distance - (total_distance - 140 + 20) = speed_A / speed_B * total_distance →
  total_distance = 480 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_l1805_180598


namespace NUMINAMATH_CALUDE_colors_needed_l1805_180593

/-- The number of people coloring the planets -/
def num_people : ℕ := 3

/-- The number of planets to be colored -/
def num_planets : ℕ := 8

/-- The total number of colors needed -/
def total_colors : ℕ := num_people * num_planets

/-- Theorem stating that the total number of colors needed is 24 -/
theorem colors_needed : total_colors = 24 := by sorry

end NUMINAMATH_CALUDE_colors_needed_l1805_180593


namespace NUMINAMATH_CALUDE_only_f₂_is_saturated_l1805_180509

/-- Definition of a "saturated function of 1" -/
def is_saturated_function_of_1 (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f (x₀ + 1) = f x₀ + f 1

/-- Function f₁(x) = 1/x -/
noncomputable def f₁ (x : ℝ) : ℝ := 1 / x

/-- Function f₂(x) = 2^x -/
noncomputable def f₂ (x : ℝ) : ℝ := 2^x

/-- Function f₃(x) = log(x² + 2) -/
noncomputable def f₃ (x : ℝ) : ℝ := Real.log (x^2 + 2)

/-- Theorem stating that only f₂ is a "saturated function of 1" -/
theorem only_f₂_is_saturated :
  ¬ is_saturated_function_of_1 f₁ ∧
  is_saturated_function_of_1 f₂ ∧
  ¬ is_saturated_function_of_1 f₃ :=
sorry

end NUMINAMATH_CALUDE_only_f₂_is_saturated_l1805_180509


namespace NUMINAMATH_CALUDE_container_volume_l1805_180558

theorem container_volume (x y z : ℝ) 
  (h_order : x < y ∧ y < z)
  (h_xy : 5 * x * y = 120)
  (h_xz : 3 * x * z = 120)
  (h_yz : 2 * y * z = 120) :
  x * y * z = 240 := by
sorry

end NUMINAMATH_CALUDE_container_volume_l1805_180558


namespace NUMINAMATH_CALUDE_min_values_constraint_l1805_180592

theorem min_values_constraint (x y z : ℝ) (h : x - 2*y + z = 4) :
  (∀ a b c : ℝ, a - 2*b + c = 4 → x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2) ∧
  (∀ a b c : ℝ, a - 2*b + c = 4 → x^2 + (y - 1)^2 + z^2 ≤ a^2 + (b - 1)^2 + c^2) ∧
  (∃ a b c : ℝ, a - 2*b + c = 4 ∧ a^2 + b^2 + c^2 = 8/3) ∧
  (∃ a b c : ℝ, a - 2*b + c = 4 ∧ a^2 + (b - 1)^2 + c^2 = 6) := by
  sorry

end NUMINAMATH_CALUDE_min_values_constraint_l1805_180592


namespace NUMINAMATH_CALUDE_star_commutative_l1805_180560

/-- Binary operation ★ defined for integers -/
def star (a b : ℤ) : ℤ := a^2 + b^2

/-- Theorem stating that ★ is commutative for all integers -/
theorem star_commutative : ∀ (a b : ℤ), star a b = star b a := by
  sorry

end NUMINAMATH_CALUDE_star_commutative_l1805_180560


namespace NUMINAMATH_CALUDE_special_triangle_common_area_l1805_180507

/-- A triangle with side lengths 18, 24, and 30 -/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 18
  hb : b = 24
  hc : c = 30

/-- The common region of two overlapping triangles -/
def CommonRegion (t1 t2 : SpecialTriangle) : Set (ℝ × ℝ) := sorry

/-- The area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- Two triangles share the same circumcircle -/
def ShareCircumcircle (t1 t2 : SpecialTriangle) : Prop := sorry

/-- Two triangles share the same inscribed circle -/
def ShareInscribedCircle (t1 t2 : SpecialTriangle) : Prop := sorry

/-- Two triangles do not completely overlap -/
def NotCompletelyOverlap (t1 t2 : SpecialTriangle) : Prop := sorry

theorem special_triangle_common_area 
  (t1 t2 : SpecialTriangle) 
  (h_circ : ShareCircumcircle t1 t2) 
  (h_insc : ShareInscribedCircle t1 t2) 
  (h_overlap : NotCompletelyOverlap t1 t2) : 
  area (CommonRegion t1 t2) = 132 := by sorry

end NUMINAMATH_CALUDE_special_triangle_common_area_l1805_180507


namespace NUMINAMATH_CALUDE_kaleb_boxes_correct_l1805_180514

/-- The number of boxes Kaleb bought initially -/
def initial_boxes : ℕ := 9

/-- The number of boxes Kaleb gave to his little brother -/
def given_boxes : ℕ := 5

/-- The number of pieces in each box -/
def pieces_per_box : ℕ := 6

/-- The number of pieces Kaleb still has -/
def remaining_pieces : ℕ := 54

/-- Theorem stating that the initial number of boxes is correct -/
theorem kaleb_boxes_correct :
  initial_boxes * pieces_per_box = remaining_pieces + given_boxes * pieces_per_box :=
by sorry

end NUMINAMATH_CALUDE_kaleb_boxes_correct_l1805_180514


namespace NUMINAMATH_CALUDE_constant_ratio_l1805_180521

/-- Two arithmetic sequences with sums of first n terms S_n and T_n -/
def arithmetic_sequences (S T : ℕ → ℝ) : Prop :=
  ∃ (a₁ d_a b₁ d_b : ℝ),
    ∀ n : ℕ, 
      S n = n / 2 * (2 * a₁ + (n - 1) * d_a) ∧
      T n = n / 2 * (2 * b₁ + (n - 1) * d_b)

/-- The product of sums equals n^3 - n for all positive n -/
def product_condition (S T : ℕ → ℝ) : Prop :=
  ∀ n : ℕ+, S n * T n = (n : ℝ)^3 - n

/-- The main theorem: if the conditions are satisfied, then S_n / T_n is constant -/
theorem constant_ratio 
  (S T : ℕ → ℝ) 
  (h1 : arithmetic_sequences S T) 
  (h2 : product_condition S T) : 
  ∃ c : ℝ, ∀ n : ℕ+, S n / T n = c :=
sorry

end NUMINAMATH_CALUDE_constant_ratio_l1805_180521


namespace NUMINAMATH_CALUDE_lucy_lovely_age_problem_l1805_180541

theorem lucy_lovely_age_problem (lucy_age : ℕ) (lovely_age : ℕ) (years_until_twice : ℕ) : 
  lucy_age = 50 →
  lucy_age - 5 = 3 * (lovely_age - 5) →
  lucy_age + years_until_twice = 2 * (lovely_age + years_until_twice) →
  years_until_twice = 10 := by
sorry

end NUMINAMATH_CALUDE_lucy_lovely_age_problem_l1805_180541


namespace NUMINAMATH_CALUDE_rotate90_clockwise_correct_rotation_result_l1805_180537

/-- Rotate a point 90 degrees clockwise around the origin -/
def rotate90Clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, -p.1)

theorem rotate90_clockwise_correct (x y : ℝ) :
  rotate90Clockwise (x, y) = (y, -x) := by sorry

/-- The original point A -/
def A : ℝ × ℝ := (2, 3)

/-- The rotated point B -/
def B : ℝ × ℝ := rotate90Clockwise A

theorem rotation_result :
  B = (3, -2) := by sorry

end NUMINAMATH_CALUDE_rotate90_clockwise_correct_rotation_result_l1805_180537


namespace NUMINAMATH_CALUDE_max_y_value_l1805_180556

theorem max_y_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x * y = (x - y) / (x + 3 * y)) : 
  y ≤ 1/3 ∧ ∃ (x₀ : ℝ), x₀ > 0 ∧ x₀ * (1/3) = (x₀ - 1/3) / (x₀ + 1) := by
sorry

end NUMINAMATH_CALUDE_max_y_value_l1805_180556


namespace NUMINAMATH_CALUDE_min_sum_of_product_l1805_180524

theorem min_sum_of_product (a b : ℤ) (h : a * b = 150) : 
  ∀ x y : ℤ, x * y = 150 → a + b ≤ x + y ∧ ∃ a₀ b₀ : ℤ, a₀ * b₀ = 150 ∧ a₀ + b₀ = -151 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_product_l1805_180524


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l1805_180520

theorem quadratic_roots_properties (x₁ x₂ : ℝ) :
  x₁^2 - 2*x₁ - 1 = 0 ∧ x₂^2 - 2*x₂ - 1 = 0 →
  (x₁ + x₂) * (x₁ * x₂) = -2 ∧ (x₁ - x₂)^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l1805_180520


namespace NUMINAMATH_CALUDE_inequality_solution_l1805_180550

theorem inequality_solution (x : ℝ) : 
  (x^2 - 4*x - 45) / (x + 7) < 0 ↔ (x > -7 ∧ x < -5) ∨ (x > -5 ∧ x < 9) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1805_180550


namespace NUMINAMATH_CALUDE_square_EFGH_side_length_l1805_180510

/-- Square ABCD with side length 10 cm -/
def square_ABCD : Real := 10

/-- Distance of line p from side AB -/
def line_p_distance : Real := 6.5

/-- Area difference between the two parts divided by line p -/
def area_difference : Real := 13.8

/-- Side length of square EFGH -/
def square_EFGH_side : Real := 5.4

theorem square_EFGH_side_length :
  ∃ (square_EFGH : Real),
    square_EFGH = square_EFGH_side ∧
    square_EFGH > 0 ∧
    square_EFGH < square_ABCD ∧
    (square_ABCD - square_EFGH) * line_p_distance = area_difference / 2 ∧
    (square_ABCD - square_EFGH) * (square_ABCD - line_p_distance) = area_difference / 2 :=
by sorry

end NUMINAMATH_CALUDE_square_EFGH_side_length_l1805_180510


namespace NUMINAMATH_CALUDE_sum_of_digits_theorem_l1805_180566

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ := sorry

-- State the theorem
theorem sum_of_digits_theorem (n : ℕ) :
  sum_of_digits n = 351 → sum_of_digits (n + 1) = 352 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_theorem_l1805_180566


namespace NUMINAMATH_CALUDE_smallest_y_for_perfect_fourth_power_l1805_180548

def x : ℕ := 5 * 15 * 35

theorem smallest_y_for_perfect_fourth_power (y : ℕ) : 
  y = 46485 ↔ 
  (∀ z : ℕ, z < y → ¬∃ (n : ℕ), x * z = n^4) ∧
  ∃ (n : ℕ), x * y = n^4 :=
sorry

end NUMINAMATH_CALUDE_smallest_y_for_perfect_fourth_power_l1805_180548


namespace NUMINAMATH_CALUDE_complete_collection_probability_l1805_180562

def total_stickers : ℕ := 18
def selected_stickers : ℕ := 10
def needed_stickers : ℕ := 6
def collected_stickers : ℕ := 8

theorem complete_collection_probability :
  (Nat.choose needed_stickers needed_stickers * Nat.choose collected_stickers (selected_stickers - needed_stickers)) / 
  Nat.choose total_stickers selected_stickers = 5 / 442 := by
  sorry

end NUMINAMATH_CALUDE_complete_collection_probability_l1805_180562


namespace NUMINAMATH_CALUDE_duck_cow_problem_l1805_180568

theorem duck_cow_problem (ducks cows : ℕ) : 
  2 * ducks + 4 * cows = 2 * (ducks + cows) + 36 → cows = 18 := by
  sorry

end NUMINAMATH_CALUDE_duck_cow_problem_l1805_180568


namespace NUMINAMATH_CALUDE_six_coin_flip_probability_six_coin_flip_probability_is_one_thirtysecond_l1805_180532

theorem six_coin_flip_probability : ℝ :=
  let n : ℕ := 6  -- number of coins
  let p : ℝ := 1 / 2  -- probability of heads for a fair coin
  let total_outcomes : ℕ := 2^n
  let favorable_outcomes : ℕ := 2  -- all heads or all tails
  favorable_outcomes / total_outcomes

theorem six_coin_flip_probability_is_one_thirtysecond : 
  six_coin_flip_probability = 1 / 32 := by
  sorry

end NUMINAMATH_CALUDE_six_coin_flip_probability_six_coin_flip_probability_is_one_thirtysecond_l1805_180532


namespace NUMINAMATH_CALUDE_soap_survey_ratio_l1805_180531

/-- Represents the survey results of household soap usage -/
structure SoapSurvey where
  total : ℕ
  neither : ℕ
  onlyE : ℕ
  both : ℕ
  onlyB : ℕ

/-- The ratio of households using only brand B to those using both brands -/
def brandBRatio (survey : SoapSurvey) : ℚ :=
  survey.onlyB / survey.both

/-- The survey satisfies the given conditions -/
def validSurvey (survey : SoapSurvey) : Prop :=
  survey.total = 200 ∧
  survey.neither = 80 ∧
  survey.onlyE = 60 ∧
  survey.both = 40 ∧
  survey.total = survey.neither + survey.onlyE + survey.onlyB + survey.both

theorem soap_survey_ratio (survey : SoapSurvey) (h : validSurvey survey) :
  brandBRatio survey = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_soap_survey_ratio_l1805_180531


namespace NUMINAMATH_CALUDE_number_difference_l1805_180525

theorem number_difference (x y : ℝ) : 
  (35 + x) / 2 = 45 →
  (35 + x + y) / 3 = 40 →
  |y - 35| = 5 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l1805_180525


namespace NUMINAMATH_CALUDE_product_sum_fractions_l1805_180584

theorem product_sum_fractions : (2 * 3 * 4) * (1 / 2 + 1 / 3 + 1 / 4) = 26 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_fractions_l1805_180584


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l1805_180540

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 60 → x - y = 10 → x * y = 875 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l1805_180540


namespace NUMINAMATH_CALUDE_value_of_expression_l1805_180571

theorem value_of_expression (x y : ℝ) (h : x - 2*y = 3) : x - 2*y + 4 = 7 := by sorry

end NUMINAMATH_CALUDE_value_of_expression_l1805_180571


namespace NUMINAMATH_CALUDE_count_non_dividing_eq_29_l1805_180513

/-- g(n) is the product of proper positive integer divisors of n -/
def g (n : ℕ) : ℕ := sorry

/-- count_non_dividing counts the number of integers n between 2 and 100 (inclusive) 
    for which n does not divide g(n) -/
def count_non_dividing : ℕ := sorry

/-- Theorem stating that the count of integers n between 2 and 100 (inclusive) 
    for which n does not divide g(n) is equal to 29 -/
theorem count_non_dividing_eq_29 : count_non_dividing = 29 := by sorry

end NUMINAMATH_CALUDE_count_non_dividing_eq_29_l1805_180513


namespace NUMINAMATH_CALUDE_this_is_2345_l1805_180533

def letter_to_digit : Char → Nat
| 'M' => 0
| 'A' => 1
| 'T' => 2
| 'H' => 3
| 'I' => 4
| 'S' => 5
| 'F' => 6
| 'U' => 7
| 'N' => 8
| _ => 9  -- Default case for completeness

def code_to_number (code : List Char) : Nat :=
  code.foldl (fun acc d => acc * 10 + letter_to_digit d) 0

theorem this_is_2345 :
  code_to_number ['T', 'H', 'I', 'S'] = 2345 := by
  sorry

end NUMINAMATH_CALUDE_this_is_2345_l1805_180533


namespace NUMINAMATH_CALUDE_markup_rate_calculation_l1805_180529

/-- Represents the markup rate calculation for a product with given profit and expense percentages. -/
theorem markup_rate_calculation (profit_percent : ℝ) (expense_percent : ℝ) :
  profit_percent = 0.12 →
  expense_percent = 0.18 →
  let cost_percent := 1 - profit_percent - expense_percent
  let markup_rate := (1 / cost_percent - 1) * 100
  ∃ ε > 0, abs (markup_rate - 42.857) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_markup_rate_calculation_l1805_180529


namespace NUMINAMATH_CALUDE_conversion_equivalence_l1805_180564

/-- Conversion rates between different units --/
structure ConversionRates where
  knicks_to_knacks : ℚ  -- 5 knicks = 3 knacks
  knacks_to_knocks : ℚ  -- 2 knacks = 5 knocks
  knocks_to_kracks : ℚ  -- 4 knocks = 1 krack

/-- Calculate the equivalent number of knicks for a given number of knocks --/
def knocks_to_knicks (rates : ConversionRates) (knocks : ℚ) : ℚ :=
  knocks * rates.knacks_to_knocks * rates.knicks_to_knacks

/-- Calculate the equivalent number of kracks for a given number of knocks --/
def knocks_to_kracks (rates : ConversionRates) (knocks : ℚ) : ℚ :=
  knocks * rates.knocks_to_kracks

theorem conversion_equivalence (rates : ConversionRates) 
  (h1 : rates.knicks_to_knacks = 3 / 5)
  (h2 : rates.knacks_to_knocks = 5 / 2)
  (h3 : rates.knocks_to_kracks = 1 / 4) :
  knocks_to_knicks rates 50 = 100 / 3 ∧ knocks_to_kracks rates 50 = 25 / 3 := by
  sorry

#check conversion_equivalence

end NUMINAMATH_CALUDE_conversion_equivalence_l1805_180564


namespace NUMINAMATH_CALUDE_select_parts_with_first_class_l1805_180526

theorem select_parts_with_first_class (total : Nat) (first_class : Nat) (second_class : Nat) (select : Nat) :
  total = first_class + second_class →
  first_class = 5 →
  second_class = 3 →
  select = 3 →
  (Nat.choose total select) - (Nat.choose second_class select) = 55 := by
  sorry

end NUMINAMATH_CALUDE_select_parts_with_first_class_l1805_180526


namespace NUMINAMATH_CALUDE_equality_of_expressions_l1805_180575

theorem equality_of_expressions (x : ℝ) : 
  (x - 2)^4 + 4*(x - 2)^3 + 6*(x - 2)^2 + 4*(x - 2) + 1 = (x - 1)^4 := by
  sorry

end NUMINAMATH_CALUDE_equality_of_expressions_l1805_180575


namespace NUMINAMATH_CALUDE_bookstore_max_revenue_l1805_180567

/-- The revenue function for the bookstore -/
def revenue (p : ℝ) : ℝ := p * (150 - 6 * p)

/-- The maximum price allowed -/
def max_price : ℝ := 30

theorem bookstore_max_revenue :
  ∃ (p : ℝ), p ≤ max_price ∧
    ∀ (q : ℝ), q ≤ max_price → revenue q ≤ revenue p ∧
    p = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_max_revenue_l1805_180567


namespace NUMINAMATH_CALUDE_dog_barks_theorem_l1805_180538

/-- The number of times a single dog barks per minute -/
def single_dog_barks_per_minute : ℕ := 30

/-- The number of dogs -/
def number_of_dogs : ℕ := 2

/-- The duration of barking in minutes -/
def duration : ℕ := 10

/-- The total number of barks from all dogs -/
def total_barks : ℕ := 600

theorem dog_barks_theorem :
  single_dog_barks_per_minute * number_of_dogs * duration = total_barks :=
by sorry

end NUMINAMATH_CALUDE_dog_barks_theorem_l1805_180538


namespace NUMINAMATH_CALUDE_slower_walk_delay_l1805_180583

/-- Proves that walking at 4/5 of the usual speed results in a 6-minute delay -/
theorem slower_walk_delay (usual_time : ℝ) (h : usual_time = 24) : 
  let slower_time := usual_time / (4/5)
  slower_time - usual_time = 6 := by
  sorry

#check slower_walk_delay

end NUMINAMATH_CALUDE_slower_walk_delay_l1805_180583


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_six_l1805_180544

theorem sqrt_expression_equals_six :
  (Real.sqrt 27 - 3 * Real.sqrt (1/3)) / (1 / Real.sqrt 3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_six_l1805_180544


namespace NUMINAMATH_CALUDE_stickers_per_pack_l1805_180508

/-- Proves that the number of stickers in each pack is 30 --/
theorem stickers_per_pack (
  num_packs : ℕ)
  (cost_per_sticker : ℚ)
  (total_cost : ℚ)
  (h1 : num_packs = 4)
  (h2 : cost_per_sticker = 1/10)
  (h3 : total_cost = 12) :
  (total_cost / cost_per_sticker) / num_packs = 30 := by
  sorry

#check stickers_per_pack

end NUMINAMATH_CALUDE_stickers_per_pack_l1805_180508


namespace NUMINAMATH_CALUDE_sequence_matches_given_terms_sequence_satisfies_conditions_l1805_180563

/-- The sequence a_n is defined as 10^n + n -/
def a (n : ℕ) : ℕ := 10^n + n

/-- The first four terms of the sequence match the given values -/
theorem sequence_matches_given_terms :
  a 1 = 11 ∧ a 2 = 102 ∧ a 3 = 1003 ∧ a 4 = 10004 := by
  sorry

/-- The sequence a_n satisfies the given first four terms -/
theorem sequence_satisfies_conditions : ∃ f : ℕ → ℕ, 
  (f 1 = 11 ∧ f 2 = 102 ∧ f 3 = 1003 ∧ f 4 = 10004) ∧
  (∀ n : ℕ, f n = a n) := by
  sorry

end NUMINAMATH_CALUDE_sequence_matches_given_terms_sequence_satisfies_conditions_l1805_180563


namespace NUMINAMATH_CALUDE_squared_sum_inequality_l1805_180523

theorem squared_sum_inequality (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a^3 + b^3 = 2*a*b) :
  a^2 + b^2 ≤ 1 + a*b := by sorry

end NUMINAMATH_CALUDE_squared_sum_inequality_l1805_180523


namespace NUMINAMATH_CALUDE_point_outside_circle_l1805_180511

/-- A circle with a given diameter -/
structure Circle where
  diameter : ℝ
  diameter_pos : diameter > 0

/-- A point with a given distance from the center of a circle -/
structure Point (c : Circle) where
  distance_from_center : ℝ
  distance_pos : distance_from_center > 0

/-- Definition of a point being outside a circle -/
def is_outside (c : Circle) (p : Point c) : Prop :=
  p.distance_from_center > c.diameter / 2

theorem point_outside_circle (c : Circle) (p : Point c) 
  (h_diam : c.diameter = 10) 
  (h_dist : p.distance_from_center = 6) : 
  is_outside c p := by
  sorry

end NUMINAMATH_CALUDE_point_outside_circle_l1805_180511


namespace NUMINAMATH_CALUDE_divisibility_of_p_and_q_l1805_180504

def ones (n : ℕ) : ℕ := (10^n - 1) / 9

def p (n : ℕ) : ℕ := ones n * (10^(3*n) + 9*10^(2*n) + 8*10^n + 7)

def q (n : ℕ) : ℕ := ones (n+1) * (10^(3*(n+1)) + 9*10^(2*(n+1)) + 8*10^(n+1) + 7)

theorem divisibility_of_p_and_q (n : ℕ) (h : 1987 ∣ ones n) : 
  1987 ∣ p n ∧ 1987 ∣ q n := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_p_and_q_l1805_180504


namespace NUMINAMATH_CALUDE_k_squared_test_probability_two_males_l1805_180552

-- Define the contingency table
def contingency_table : Matrix (Fin 2) (Fin 2) ℕ :=
  ![![30, 45],
    ![15, 10]]

-- Define the total number of people surveyed
def total_surveyed : ℕ := 100

-- Define the K² formula
def k_squared (a b c d : ℕ) : ℚ :=
  (total_surveyed * (a * d - b * c)^2 : ℚ) /
  ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for 95% confidence
def critical_value : ℚ := 3841 / 1000

-- Theorem for the K² test
theorem k_squared_test :
  k_squared (contingency_table 0 0) (contingency_table 0 1)
            (contingency_table 1 0) (contingency_table 1 1) < critical_value := by
  sorry

-- Define the number of healthy living people
def healthy_living : ℕ := 45

-- Define the number of healthy living males
def healthy_males : ℕ := 30

-- Theorem for the probability of selecting two males
theorem probability_two_males :
  (Nat.choose healthy_males 2 : ℚ) / (Nat.choose healthy_living 2) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_k_squared_test_probability_two_males_l1805_180552


namespace NUMINAMATH_CALUDE_return_trip_time_l1805_180569

/-- Represents the time taken to travel between docks -/
structure TravelTime where
  forward : ℝ  -- Time taken in the forward direction
  backward : ℝ  -- Time taken in the backward direction

/-- Proves that given specific travel times between docks, the return trip takes 72 minutes -/
theorem return_trip_time (travel : TravelTime) 
  (h1 : travel.forward = 30) 
  (h2 : travel.backward = 18) 
  (h3 : travel.forward > 0) 
  (h4 : travel.backward > 0) : 
  (3 * travel.forward * travel.backward) / (travel.forward - travel.backward) = 72 := by
  sorry

#check return_trip_time

end NUMINAMATH_CALUDE_return_trip_time_l1805_180569


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1805_180555

/-- An ellipse with focal points (-2, 0) and (2, 0) that intersects the line x + y + 4 = 0 at exactly one point has a major axis of length 8. -/
theorem ellipse_major_axis_length :
  ∀ (E : Set (ℝ × ℝ)),
  (∀ (P : ℝ × ℝ), P ∈ E ↔ 
    Real.sqrt ((P.1 + 2)^2 + P.2^2) + Real.sqrt ((P.1 - 2)^2 + P.2^2) = 8) →
  (∃! (P : ℝ × ℝ), P ∈ E ∧ P.1 + P.2 + 4 = 0) →
  8 = 8 := by
sorry


end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1805_180555


namespace NUMINAMATH_CALUDE_error_clock_correct_time_l1805_180502

/-- Represents a 12-hour digital clock with a display error -/
structure ErrorClock where
  /-- The number of hours in the clock cycle -/
  total_hours : Nat
  /-- The number of minutes in an hour -/
  minutes_per_hour : Nat
  /-- The number of hours affected by the display error -/
  incorrect_hours : Nat
  /-- The number of minutes per hour affected by the display error -/
  incorrect_minutes : Nat

/-- The fraction of the day when the ErrorClock shows the correct time -/
def correct_time_fraction (clock : ErrorClock) : Rat :=
  ((clock.total_hours - clock.incorrect_hours) * (clock.minutes_per_hour - clock.incorrect_minutes)) / 
  (clock.total_hours * clock.minutes_per_hour)

/-- The specific ErrorClock instance for the problem -/
def problem_clock : ErrorClock :=
  { total_hours := 12
  , minutes_per_hour := 60
  , incorrect_hours := 4
  , incorrect_minutes := 15 }

theorem error_clock_correct_time :
  correct_time_fraction problem_clock = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_error_clock_correct_time_l1805_180502


namespace NUMINAMATH_CALUDE_solution_s_l1805_180501

theorem solution_s (s : ℝ) : 
  Real.sqrt (3 * Real.sqrt (s - 3)) = (9 - s) ^ (1/4) → s = 3.6 := by
  sorry

end NUMINAMATH_CALUDE_solution_s_l1805_180501


namespace NUMINAMATH_CALUDE_one_not_identity_for_star_l1805_180545

-- Define the set S of all non-zero real numbers
def S : Set ℝ := {x : ℝ | x ≠ 0}

-- Define the binary operation *
def star (a b : ℝ) : ℝ := 3 * a * b + 1

-- Theorem stating that 1 is not an identity element for * in S
theorem one_not_identity_for_star :
  ¬(∀ a ∈ S, (star 1 a = a ∧ star a 1 = a)) :=
sorry

end NUMINAMATH_CALUDE_one_not_identity_for_star_l1805_180545


namespace NUMINAMATH_CALUDE_exists_function_sum_one_not_exists_function_diff_one_l1805_180570

-- Part a
theorem exists_function_sum_one : 
  ∃ f : ℝ → ℝ, (∀ x : ℝ, f (Real.sin x) + f (Real.cos x) = 1) ∧ 
  (∃ a b : ℝ, a ≠ b ∧ f a ≠ f b) :=
sorry

-- Part b
theorem not_exists_function_diff_one : 
  ¬∃ f : ℝ → ℝ, (∀ x : ℝ, f (Real.sin x) - f (Real.cos x) = 1) ∧ 
  (∃ a b : ℝ, a ≠ b ∧ f a ≠ f b) :=
sorry

end NUMINAMATH_CALUDE_exists_function_sum_one_not_exists_function_diff_one_l1805_180570


namespace NUMINAMATH_CALUDE_max_value_expression_l1805_180527

theorem max_value_expression (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_squares : x^2 + y^2 + z^2 = 1) :
  4*x*y*Real.sqrt 2 + 5*y*z + 3*x*z*Real.sqrt 3 ≤ (44*Real.sqrt 2 + 110 + 9*Real.sqrt 3) / 3 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ ≥ 0 ∧ y₀ ≥ 0 ∧ z₀ ≥ 0 ∧ x₀^2 + y₀^2 + z₀^2 = 1 ∧
    4*x₀*y₀*Real.sqrt 2 + 5*y₀*z₀ + 3*x₀*z₀*Real.sqrt 3 = (44*Real.sqrt 2 + 110 + 9*Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l1805_180527


namespace NUMINAMATH_CALUDE_sum_of_positive_factors_36_l1805_180515

-- Define the sum of positive factors function
def sumOfPositiveFactors (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_positive_factors_36 : sumOfPositiveFactors 36 = 91 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_positive_factors_36_l1805_180515


namespace NUMINAMATH_CALUDE_high_school_students_l1805_180596

theorem high_school_students (total : ℕ) 
  (h1 : total / 2 = total / 2) -- Half of the students are freshmen or sophomores
  (h2 : (total / 2) / 5 = (total / 2) / 5) -- One-fifth of freshmen and sophomores own a pet
  (h3 : total / 2 - (total / 2) / 5 = 160) -- 160 freshmen and sophomores do not own a pet
  : total = 400 := by
  sorry

end NUMINAMATH_CALUDE_high_school_students_l1805_180596


namespace NUMINAMATH_CALUDE_no_integer_roots_l1805_180518

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℕ → ℤ

/-- Evaluation of a polynomial at a point -/
def eval (P : IntPolynomial) (x : ℤ) : ℤ :=
  sorry

/-- A number is odd if it's not divisible by 2 -/
def IsOdd (n : ℤ) : Prop := n % 2 ≠ 0

theorem no_integer_roots (P : IntPolynomial) 
  (h0 : IsOdd (eval P 0)) 
  (h1 : IsOdd (eval P 1)) : 
  ∀ (n : ℤ), eval P n ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_l1805_180518


namespace NUMINAMATH_CALUDE_sock_pairs_count_l1805_180572

def total_socks : ℕ := 12
def white_socks : ℕ := 5
def brown_socks : ℕ := 5
def blue_socks : ℕ := 2

def same_color_pairs : ℕ := Nat.choose white_socks 2 + Nat.choose brown_socks 2 + Nat.choose blue_socks 2

theorem sock_pairs_count : same_color_pairs = 21 := by
  sorry

end NUMINAMATH_CALUDE_sock_pairs_count_l1805_180572


namespace NUMINAMATH_CALUDE_solution_composition_l1805_180554

theorem solution_composition (solution1_percent : Real) (solution1_carbonated : Real) 
  (solution2_carbonated : Real) (mixture_carbonated : Real) :
  solution1_percent = 0.4 →
  solution2_carbonated = 0.55 →
  mixture_carbonated = 0.65 →
  solution1_percent * solution1_carbonated + (1 - solution1_percent) * solution2_carbonated = mixture_carbonated →
  solution1_carbonated = 0.8 := by
sorry

end NUMINAMATH_CALUDE_solution_composition_l1805_180554


namespace NUMINAMATH_CALUDE_mika_gave_six_stickers_l1805_180587

/-- Represents the number of stickers Mika had, bought, received, used, and gave away --/
structure StickerCount where
  initial : Nat
  bought : Nat
  birthday : Nat
  usedForCard : Nat
  leftOver : Nat

/-- Calculates the number of stickers Mika gave to her sister --/
def stickersGivenToSister (s : StickerCount) : Nat :=
  s.initial + s.bought + s.birthday - (s.usedForCard + s.leftOver)

/-- Theorem stating that Mika gave 6 stickers to her sister --/
theorem mika_gave_six_stickers (s : StickerCount) 
  (h1 : s.initial = 20)
  (h2 : s.bought = 26)
  (h3 : s.birthday = 20)
  (h4 : s.usedForCard = 58)
  (h5 : s.leftOver = 2) : 
  stickersGivenToSister s = 6 := by
  sorry

end NUMINAMATH_CALUDE_mika_gave_six_stickers_l1805_180587


namespace NUMINAMATH_CALUDE_quadratic_zero_discriminant_l1805_180512

/-- The quadratic equation 5x^2 - 10x√3 + k = 0 has zero discriminant if and only if k = 15 -/
theorem quadratic_zero_discriminant (k : ℝ) :
  (∀ x : ℝ, 5 * x^2 - 10 * x * Real.sqrt 3 + k = 0) →
  ((-10 * Real.sqrt 3)^2 - 4 * 5 * k = 0) ↔
  k = 15 := by
sorry

end NUMINAMATH_CALUDE_quadratic_zero_discriminant_l1805_180512


namespace NUMINAMATH_CALUDE_arjun_has_largest_result_l1805_180539

def initial_number : ℕ := 15

def liam_result : ℕ := ((initial_number - 2) * 3) + 3

def maya_result : ℕ := ((initial_number * 3) - 4) + 5

def arjun_result : ℕ := ((initial_number - 3) + 4) * 3

theorem arjun_has_largest_result :
  arjun_result > liam_result ∧ arjun_result > maya_result :=
by sorry

end NUMINAMATH_CALUDE_arjun_has_largest_result_l1805_180539


namespace NUMINAMATH_CALUDE_masking_tape_calculation_l1805_180595

/-- Calculates the amount of masking tape needed for an L-shaped room --/
def masking_tape_needed (main_length main_width square_side room_height : ℝ) 
  (num_windows window_width window_height : ℝ) 
  (door_width door_height : ℝ) : ℝ :=
  let main_perimeter := 2 * (main_length + main_width)
  let square_perimeter := 3 * square_side
  let total_perimeter := main_perimeter + square_perimeter
  let windows_width := num_windows * window_width
  let perimeter_without_openings := total_perimeter - windows_width - door_width
  2 * perimeter_without_openings

/-- Theorem stating that the amount of masking tape needed is 45 meters --/
theorem masking_tape_calculation : 
  masking_tape_needed 5 3 4 2.5 3 1.5 1 1 2 = 45 := by
  sorry

#eval masking_tape_needed 5 3 4 2.5 3 1.5 1 1 2

end NUMINAMATH_CALUDE_masking_tape_calculation_l1805_180595


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_in_one_third_sector_l1805_180522

/-- The radius of a circle inscribed in a sector that is one-third of a circle with radius 5 cm -/
theorem inscribed_circle_radius_in_one_third_sector :
  ∃ (r : ℝ), 
    r > 0 ∧ 
    r * (Real.sqrt 3 + 1) = 5 ∧
    r = (5 * Real.sqrt 3 - 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_in_one_third_sector_l1805_180522


namespace NUMINAMATH_CALUDE_probability_second_white_given_first_red_l1805_180580

/-- Represents a bag of balls -/
structure Bag where
  white : ℕ
  red : ℕ

/-- The probability of drawing a white ball on the second draw, given that a red ball was drawn on the first draw -/
def conditional_probability (bagA bagB : Bag) : ℚ :=
  let total_balls (bag : Bag) := bag.white + bag.red
  let p_red_first (bag : Bag) := bag.red / total_balls bag
  let p_white_second_given_red_first (bag : Bag) := bag.white / (total_balls bag - 1)
  let p_AB (bag : Bag) := p_red_first bag * p_white_second_given_red_first bag
  let p_A := (p_red_first bagA + p_red_first bagB) / 2
  (p_AB bagA + p_AB bagB) / (2 * p_A)

/-- The main theorem stating the probability is 17/32 -/
theorem probability_second_white_given_first_red :
  let bagA : Bag := { white := 3, red := 2 }
  let bagB : Bag := { white := 2, red := 4 }
  conditional_probability bagA bagB = 17 / 32 := by
  sorry

end NUMINAMATH_CALUDE_probability_second_white_given_first_red_l1805_180580


namespace NUMINAMATH_CALUDE_unique_intersection_point_l1805_180599

theorem unique_intersection_point (m : ℤ) : 
  (∃ (x : ℕ+), -3 * x + 2 = m * (x^2 - x + 1)) ↔ m = -1 :=
by sorry

end NUMINAMATH_CALUDE_unique_intersection_point_l1805_180599


namespace NUMINAMATH_CALUDE_seashells_after_month_l1805_180588

/-- Calculates the number of seashells after a given number of weeks -/
def seashells_after_weeks (initial : ℕ) (weekly_increase : ℕ) (weeks : ℕ) : ℕ :=
  initial + weekly_increase * weeks

/-- Theorem stating that starting with 50 seashells and adding 20 per week for 4 weeks results in 130 seashells -/
theorem seashells_after_month (initial : ℕ) (weekly_increase : ℕ) (weeks : ℕ) 
    (h1 : initial = 50) 
    (h2 : weekly_increase = 20) 
    (h3 : weeks = 4) : 
  seashells_after_weeks initial weekly_increase weeks = 130 := by
  sorry

#eval seashells_after_weeks 50 20 4

end NUMINAMATH_CALUDE_seashells_after_month_l1805_180588


namespace NUMINAMATH_CALUDE_cylinder_height_l1805_180577

theorem cylinder_height (perimeter : Real) (diagonal : Real) (height : Real) : 
  perimeter = 6 → diagonal = 10 → height = 8 → 
  perimeter = 2 * Real.pi * (perimeter / (2 * Real.pi)) ∧ 
  diagonal^2 = perimeter^2 + height^2 :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_l1805_180577


namespace NUMINAMATH_CALUDE_min_fencing_length_proof_l1805_180542

/-- The minimum length of bamboo fencing needed to enclose a rectangular flower bed -/
def min_fencing_length : ℝ := 20

/-- The area of the rectangular flower bed -/
def flower_bed_area : ℝ := 50

theorem min_fencing_length_proof :
  ∀ (length width : ℝ),
  length > 0 →
  width > 0 →
  length * width = flower_bed_area →
  length + 2 * width ≥ min_fencing_length :=
by
  sorry

#check min_fencing_length_proof

end NUMINAMATH_CALUDE_min_fencing_length_proof_l1805_180542


namespace NUMINAMATH_CALUDE_division_and_subtraction_l1805_180586

theorem division_and_subtraction : (12 / (1/6)) - (1/3) = 215/3 := by
  sorry

end NUMINAMATH_CALUDE_division_and_subtraction_l1805_180586


namespace NUMINAMATH_CALUDE_potato_cost_proof_l1805_180594

/-- The initial cost of one bag of potatoes in rubles -/
def initial_cost : ℝ := 250

/-- The number of bags each trader bought -/
def bags_bought : ℕ := 60

/-- Andrey's price increase factor -/
def andrey_increase : ℝ := 2

/-- Boris's first price increase factor -/
def boris_first_increase : ℝ := 1.6

/-- Boris's second price increase factor -/
def boris_second_increase : ℝ := 1.4

/-- Number of bags Boris sold at first price -/
def boris_first_sale : ℕ := 15

/-- Number of bags Boris sold at second price -/
def boris_second_sale : ℕ := 45

/-- The difference in earnings between Boris and Andrey -/
def earnings_difference : ℝ := 1200

theorem potato_cost_proof :
  (bags_bought * initial_cost * andrey_increase) +
  earnings_difference =
  (boris_first_sale * initial_cost * boris_first_increase) +
  (boris_second_sale * initial_cost * boris_first_increase * boris_second_increase) :=
by sorry

end NUMINAMATH_CALUDE_potato_cost_proof_l1805_180594


namespace NUMINAMATH_CALUDE_only_extend_line_segment_valid_l1805_180561

-- Define the geometric objects
structure StraightLine
structure LineSegment where
  endpoint1 : Point
  endpoint2 : Point
structure Ray where
  endpoint : Point

-- Define the statements
inductive GeometricStatement
  | ExtendStraightLine
  | ExtendLineSegment
  | ExtendRay
  | DrawStraightLineWithLength
  | CutOffSegmentOnRay

-- Define a predicate for valid operations
def is_valid_operation (s : GeometricStatement) : Prop :=
  match s with
  | GeometricStatement.ExtendLineSegment => true
  | _ => false

-- Theorem statement
theorem only_extend_line_segment_valid :
  ∀ s : GeometricStatement, is_valid_operation s ↔ s = GeometricStatement.ExtendLineSegment := by
  sorry

end NUMINAMATH_CALUDE_only_extend_line_segment_valid_l1805_180561


namespace NUMINAMATH_CALUDE_prob_at_least_two_white_correct_l1805_180516

/-- The probability of drawing at least two white balls in three draws from a bag 
    containing 2 red balls and 4 white balls, with replacement -/
def prob_at_least_two_white : ℚ := 20 / 27

/-- The total number of balls in the bag -/
def total_balls : ℕ := 6

/-- The number of white balls in the bag -/
def white_balls : ℕ := 4

/-- The number of draws -/
def num_draws : ℕ := 3

theorem prob_at_least_two_white_correct : 
  prob_at_least_two_white = 
    (Nat.choose num_draws 2 * (white_balls / total_balls)^2 * ((total_balls - white_balls) / total_balls)) +
    (white_balls / total_balls)^num_draws :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_two_white_correct_l1805_180516


namespace NUMINAMATH_CALUDE_switching_strategy_wins_more_than_half_l1805_180528

structure ThreeBoxGame where
  boxes : Fin 3 → Bool  -- True if box contains prize, False if empty
  prize_exists : ∃ i, boxes i = true
  two_empty : ∃ i j, i ≠ j ∧ boxes i = false ∧ boxes j = false

def initial_choice (game : ThreeBoxGame) : Fin 3 :=
  sorry

def host_opens (game : ThreeBoxGame) (choice : Fin 3) : Fin 3 :=
  sorry

def switch (initial : Fin 3) (opened : Fin 3) : Fin 3 :=
  sorry

def probability_of_winning_by_switching (game : ThreeBoxGame) : ℝ :=
  sorry

theorem switching_strategy_wins_more_than_half :
  ∀ game : ThreeBoxGame, probability_of_winning_by_switching game > 1/2 :=
sorry

end NUMINAMATH_CALUDE_switching_strategy_wins_more_than_half_l1805_180528


namespace NUMINAMATH_CALUDE_coefficient_a2_l1805_180590

/-- Given z = 1/2 + (√3/2)i and (x-z)^4 = a₀x^4 + a₁x^3 + a₂x^2 + a₃x + a₄, prove that a₂ = -3 + 3√3i. -/
theorem coefficient_a2 (z : ℂ) (a₀ a₁ a₂ a₃ a₄ : ℂ) :
  z = (1 : ℂ) / 2 + (Complex.I * Real.sqrt 3) / 2 →
  (∀ x : ℂ, (x - z)^4 = a₀*x^4 + a₁*x^3 + a₂*x^2 + a₃*x + a₄) →
  a₂ = -3 + 3 * Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_a2_l1805_180590


namespace NUMINAMATH_CALUDE_bin_game_expected_value_l1805_180579

theorem bin_game_expected_value (k : ℕ) (h1 : k > 0) : 
  (8 / (8 + k : ℝ)) * 3 + (k / (8 + k : ℝ)) * (-1) = 1 → k = 8 :=
by sorry

end NUMINAMATH_CALUDE_bin_game_expected_value_l1805_180579


namespace NUMINAMATH_CALUDE_garden_flowers_l1805_180503

theorem garden_flowers (white_flowers : ℕ) (additional_red_needed : ℕ) (current_red_flowers : ℕ) : 
  white_flowers = 555 →
  additional_red_needed = 208 →
  white_flowers = current_red_flowers + additional_red_needed →
  current_red_flowers = 347 := by
sorry

end NUMINAMATH_CALUDE_garden_flowers_l1805_180503


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1805_180573

theorem complex_equation_solution (z : ℂ) : z + Complex.abs z = 2 + I → z = 1 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1805_180573


namespace NUMINAMATH_CALUDE_equation_solution_l1805_180553

theorem equation_solution (x : ℝ) :
  x > 9 →
  (Real.sqrt (x - 6 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 6 * Real.sqrt (x - 9)) - 3) ↔
  x ≥ 18 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1805_180553


namespace NUMINAMATH_CALUDE_eighty_six_million_scientific_notation_l1805_180506

/-- Expresses 86 million in scientific notation -/
theorem eighty_six_million_scientific_notation :
  (86000000 : ℝ) = 8.6 * 10^7 := by
  sorry

end NUMINAMATH_CALUDE_eighty_six_million_scientific_notation_l1805_180506


namespace NUMINAMATH_CALUDE_horizontal_distance_on_line_l1805_180543

/-- Given two points on a line, prove that the horizontal distance between them is 3 -/
theorem horizontal_distance_on_line (m n p : ℝ) : 
  (m = n / 7 - 2 / 5) → 
  (m + p = (n + 21) / 7 - 2 / 5) → 
  p = 3 := by
sorry

end NUMINAMATH_CALUDE_horizontal_distance_on_line_l1805_180543


namespace NUMINAMATH_CALUDE_custom_op_two_five_l1805_180591

-- Define the custom operation
def custom_op (a b : ℝ) : ℝ := 4 * a + 3 * b

-- State the theorem
theorem custom_op_two_five : custom_op 2 5 = 23 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_two_five_l1805_180591


namespace NUMINAMATH_CALUDE_min_draws_for_target_color_l1805_180517

/- Define the number of balls for each color -/
def red_balls : ℕ := 34
def green_balls : ℕ := 25
def yellow_balls : ℕ := 23
def blue_balls : ℕ := 18
def white_balls : ℕ := 14
def black_balls : ℕ := 10

/- Define the target number of balls of a single color -/
def target : ℕ := 20

/- Define the total number of balls -/
def total_balls : ℕ := red_balls + green_balls + yellow_balls + blue_balls + white_balls + black_balls

/- Theorem statement -/
theorem min_draws_for_target_color :
  ∃ (n : ℕ), n = 100 ∧
  (∀ (m : ℕ), m < n → 
    ∃ (r g y b w k : ℕ), 
      r ≤ red_balls ∧ 
      g ≤ green_balls ∧ 
      y ≤ yellow_balls ∧ 
      b ≤ blue_balls ∧ 
      w ≤ white_balls ∧ 
      k ≤ black_balls ∧
      r + g + y + b + w + k = m ∧
      r < target ∧ g < target ∧ y < target ∧ b < target ∧ w < target ∧ k < target) ∧
  (∀ (r g y b w k : ℕ),
    r ≤ red_balls →
    g ≤ green_balls →
    y ≤ yellow_balls →
    b ≤ blue_balls →
    w ≤ white_balls →
    k ≤ black_balls →
    r + g + y + b + w + k = n →
    r ≥ target ∨ g ≥ target ∨ y ≥ target ∨ b ≥ target ∨ w ≥ target ∨ k ≥ target) :=
by sorry

end NUMINAMATH_CALUDE_min_draws_for_target_color_l1805_180517


namespace NUMINAMATH_CALUDE_seven_times_coefficient_polynomials_l1805_180565

theorem seven_times_coefficient_polynomials (m n : ℤ) : 
  (∃ k : ℤ, 4 * m - n = 7 * k) → (∃ l : ℤ, 2 * m + 3 * n = 7 * l) := by
  sorry

end NUMINAMATH_CALUDE_seven_times_coefficient_polynomials_l1805_180565


namespace NUMINAMATH_CALUDE_inverse_inequality_l1805_180536

theorem inverse_inequality (a b : ℝ) (ha : a < 0) (hb : b > 0) : 1 / a < 1 / b := by
  sorry

end NUMINAMATH_CALUDE_inverse_inequality_l1805_180536


namespace NUMINAMATH_CALUDE_ball_count_l1805_180585

theorem ball_count (blue_count : ℕ) (prob_blue : ℚ) (green_count : ℕ) : 
  blue_count = 8 → 
  prob_blue = 1 / 5 → 
  prob_blue = blue_count / (blue_count + green_count) →
  green_count = 32 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_l1805_180585


namespace NUMINAMATH_CALUDE_coexisting_expression_coexisting_negation_l1805_180582

/-- Definition of coexisting rational number pairs -/
def is_coexisting (a b : ℚ) : Prop := a * b = a - b - 1

/-- Theorem 1: For coexisting pairs, the given expression equals 1/2 -/
theorem coexisting_expression (a b : ℚ) (h : is_coexisting a b) :
  3 * a * b - a + (1/2) * (a + b - 5 * a * b) + 1 = 1/2 := by sorry

/-- Theorem 2: If (a,b) is coexisting, then (-b,-a) is also coexisting -/
theorem coexisting_negation (a b : ℚ) (h : is_coexisting a b) :
  is_coexisting (-b) (-a) := by sorry

end NUMINAMATH_CALUDE_coexisting_expression_coexisting_negation_l1805_180582


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1805_180559

theorem partial_fraction_decomposition :
  let C : ℚ := 81 / 16
  let D : ℚ := -49 / 16
  ∀ x : ℚ, x ≠ 12 → x ≠ -4 →
    (7 * x - 3) / (x^2 - 8*x - 48) = C / (x - 12) + D / (x + 4) :=
by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1805_180559


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l1805_180578

theorem product_of_sum_and_sum_of_cubes (a b : ℝ) 
  (h1 : a + b = 8) 
  (h2 : a^3 + b^3 = 152) : 
  a * b = 15 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l1805_180578
