import Mathlib

namespace NUMINAMATH_CALUDE_f_increasing_range_l2535_253550

/-- The function f(x) = 2x^2 + mx - 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^2 + m * x - 1

/-- The theorem stating the range of m for which f is increasing on (1, +∞) -/
theorem f_increasing_range (m : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ > 1 ∧ x₂ > 1 ∧ x₁ ≠ x₂ → (f m x₁ - f m x₂) / (x₁ - x₂) > 0) →
  m ≥ -4 :=
sorry

end NUMINAMATH_CALUDE_f_increasing_range_l2535_253550


namespace NUMINAMATH_CALUDE_subset_implies_a_values_l2535_253540

def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}

def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

theorem subset_implies_a_values (a : ℝ) :
  B a ⊆ A → a = 1/3 ∨ a = 1/5 ∨ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_values_l2535_253540


namespace NUMINAMATH_CALUDE_limeade_calories_l2535_253506

-- Define the components of limeade
def lime_juice_weight : ℝ := 150
def sugar_weight : ℝ := 200
def water_weight : ℝ := 450

-- Define calorie content per 100g
def lime_juice_calories_per_100g : ℝ := 20
def sugar_calories_per_100g : ℝ := 396
def water_calories_per_100g : ℝ := 0

-- Define the weight of limeade we want to calculate calories for
def limeade_sample_weight : ℝ := 300

-- Theorem statement
theorem limeade_calories : 
  let total_weight := lime_juice_weight + sugar_weight + water_weight
  let total_calories := (lime_juice_calories_per_100g * lime_juice_weight / 100) + 
                        (sugar_calories_per_100g * sugar_weight / 100) + 
                        (water_calories_per_100g * water_weight / 100)
  (total_calories * limeade_sample_weight / total_weight) = 308.25 := by
  sorry

end NUMINAMATH_CALUDE_limeade_calories_l2535_253506


namespace NUMINAMATH_CALUDE_tea_mixture_price_l2535_253587

/-- Given two types of tea mixed in equal proportions, this theorem proves
    the price of the second tea given the price of the first tea and the mixture. -/
theorem tea_mixture_price
  (price_tea1 : ℝ)
  (price_mixture : ℝ)
  (h1 : price_tea1 = 64)
  (h2 : price_mixture = 69) :
  ∃ (price_tea2 : ℝ),
    price_tea2 = 74 ∧
    (price_tea1 + price_tea2) / 2 = price_mixture :=
by
  sorry

end NUMINAMATH_CALUDE_tea_mixture_price_l2535_253587


namespace NUMINAMATH_CALUDE_equation_proof_l2535_253556

theorem equation_proof : (100 - 6) * 7 - 52 + 8 + 9 = 623 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l2535_253556


namespace NUMINAMATH_CALUDE_distance_to_midpoint_zero_l2535_253581

theorem distance_to_midpoint_zero (x₁ y₁ x₂ y₂ : ℝ) 
  (h1 : x₁ = 10 ∧ y₁ = 20) 
  (h2 : x₂ = -10 ∧ y₂ = -20) : 
  Real.sqrt (((x₁ + x₂) / 2)^2 + ((y₁ + y₂) / 2)^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_midpoint_zero_l2535_253581


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l2535_253547

/-- The focal length of a hyperbola with equation x²/a² - y²/b² = 1 is 2√(a² + b²) -/
theorem hyperbola_focal_length (a b : ℝ) (h : a > 0 ∧ b > 0) :
  let focal_length := 2 * Real.sqrt (a^2 + b^2)
  focal_length = 2 * Real.sqrt 7 ↔ a^2 = 4 ∧ b^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l2535_253547


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l2535_253575

def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 11*x - 6

theorem roots_of_polynomial :
  (∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3) ∧
  (∀ x : ℝ, (x - 1) * (x - 2) * (x - 3) = f x) :=
sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l2535_253575


namespace NUMINAMATH_CALUDE_max_sum_of_digits_base8_less_than_1800_l2535_253539

/-- Represents the sum of digits in base 8 for a natural number -/
def sumOfDigitsBase8 (n : ℕ) : ℕ := sorry

/-- The greatest possible sum of digits in base 8 for numbers less than 1800 -/
def maxSumOfDigitsBase8LessThan1800 : ℕ := 23

/-- Theorem stating that the maximum sum of digits in base 8 for positive integers less than 1800 is 23 -/
theorem max_sum_of_digits_base8_less_than_1800 :
  ∀ n : ℕ, 0 < n → n < 1800 → sumOfDigitsBase8 n ≤ maxSumOfDigitsBase8LessThan1800 ∧
  ∃ m : ℕ, 0 < m ∧ m < 1800 ∧ sumOfDigitsBase8 m = maxSumOfDigitsBase8LessThan1800 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_digits_base8_less_than_1800_l2535_253539


namespace NUMINAMATH_CALUDE_dianas_age_l2535_253541

theorem dianas_age (Carlos Diana Emily : ℚ) 
  (h1 : Carlos = 4 * Diana)
  (h2 : Emily = Diana + 5)
  (h3 : Carlos = Emily) : 
  Diana = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_dianas_age_l2535_253541


namespace NUMINAMATH_CALUDE_females_advanced_count_l2535_253525

/-- A company's employee distribution by gender and education level -/
structure Company where
  total_employees : ℕ
  females : ℕ
  males : ℕ
  advanced_degrees : ℕ
  college_degrees : ℕ
  vocational_training : ℕ
  males_college : ℕ
  females_vocational : ℕ

/-- The number of females with advanced degrees in the company -/
def females_advanced (c : Company) : ℕ :=
  c.advanced_degrees - (c.males - c.males_college - (c.vocational_training - c.females_vocational))

/-- Theorem stating the number of females with advanced degrees -/
theorem females_advanced_count (c : Company) 
  (h1 : c.total_employees = 360)
  (h2 : c.females = 220)
  (h3 : c.males = 140)
  (h4 : c.advanced_degrees = 140)
  (h5 : c.college_degrees = 160)
  (h6 : c.vocational_training = 60)
  (h7 : c.males_college = 55)
  (h8 : c.females_vocational = 25)
  (h9 : c.total_employees = c.females + c.males)
  (h10 : c.total_employees = c.advanced_degrees + c.college_degrees + c.vocational_training) :
  females_advanced c = 90 := by
  sorry

#eval females_advanced {
  total_employees := 360,
  females := 220,
  males := 140,
  advanced_degrees := 140,
  college_degrees := 160,
  vocational_training := 60,
  males_college := 55,
  females_vocational := 25
}

end NUMINAMATH_CALUDE_females_advanced_count_l2535_253525


namespace NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l2535_253569

def arithmetic_sequence (a d : ℚ) (n : ℕ) : ℚ := a + (n - 1 : ℚ) * d

def sum_arithmetic_sequence (a d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * a + (n - 1 : ℚ) * d)

theorem first_term_of_arithmetic_sequence :
  ∃ (a d : ℚ),
    sum_arithmetic_sequence a d 30 = 300 ∧
    sum_arithmetic_sequence (arithmetic_sequence a d 31) d 40 = 2200 ∧
    a = -121 / 14 := by
  sorry

end NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l2535_253569


namespace NUMINAMATH_CALUDE_factor_polynomial_l2535_253577

theorem factor_polynomial (x : ℝ) : 54 * x^5 - 135 * x^9 = -27 * x^5 * (5 * x^4 - 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l2535_253577


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l2535_253578

theorem simplify_sqrt_expression :
  Real.sqrt (12 + 8 * Real.sqrt 3) + Real.sqrt (12 - 8 * Real.sqrt 3) = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l2535_253578


namespace NUMINAMATH_CALUDE_cubic_congruence_solutions_l2535_253558

theorem cubic_congruence_solutions :
  ∀ (a b : ℤ),
    (a^3 ≡ b^3 [ZMOD 121] ↔ (a ≡ b [ZMOD 121] ∨ 11 ∣ a ∧ 11 ∣ b)) ∧
    (a^3 ≡ b^3 [ZMOD 169] ↔ (a ≡ b [ZMOD 169] ∨ a ≡ 22*b [ZMOD 169] ∨ a ≡ 146*b [ZMOD 169] ∨ 13 ∣ a ∧ 13 ∣ b)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_congruence_solutions_l2535_253558


namespace NUMINAMATH_CALUDE_distribute_8_balls_3_boxes_l2535_253521

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes --/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes,
    with each box containing at least one ball --/
def distribute_balls_nonempty (n : ℕ) (k : ℕ) : ℕ := distribute_balls (n - k) k

theorem distribute_8_balls_3_boxes : distribute_balls_nonempty 8 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_distribute_8_balls_3_boxes_l2535_253521


namespace NUMINAMATH_CALUDE_cube_surface_area_l2535_253599

theorem cube_surface_area (a : ℝ) : 
  let edge_length : ℝ := 5 * a
  let surface_area : ℝ := 6 * (edge_length ^ 2)
  surface_area = 150 * (a ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l2535_253599


namespace NUMINAMATH_CALUDE_division_problem_l2535_253509

theorem division_problem : (-1) / (-5) / (-1/5) = -1 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2535_253509


namespace NUMINAMATH_CALUDE_simplified_fourth_root_l2535_253574

theorem simplified_fourth_root (c d : ℕ+) :
  (3^5 * 5^3 : ℝ)^(1/4) = c * d^(1/4) → c + d = 3378 := by
  sorry

end NUMINAMATH_CALUDE_simplified_fourth_root_l2535_253574


namespace NUMINAMATH_CALUDE_improved_representation_of_100_l2535_253500

theorem improved_representation_of_100 :
  (222 / 2 : ℚ) - (22 / 2 : ℚ) = 100 := by sorry

end NUMINAMATH_CALUDE_improved_representation_of_100_l2535_253500


namespace NUMINAMATH_CALUDE_integral_2x_minus_1_l2535_253542

theorem integral_2x_minus_1 : ∫ x in (1:ℝ)..(2:ℝ), 2*x - 1 = 2 := by sorry

end NUMINAMATH_CALUDE_integral_2x_minus_1_l2535_253542


namespace NUMINAMATH_CALUDE_max_correct_answers_l2535_253576

theorem max_correct_answers (total_questions : ℕ) (correct_points : ℤ) (incorrect_points : ℤ) (total_score : ℤ) :
  total_questions = 30 →
  correct_points = 4 →
  incorrect_points = -3 →
  total_score = 54 →
  ∃ (correct incorrect blank : ℕ),
    correct + incorrect + blank = total_questions ∧
    correct * correct_points + incorrect * incorrect_points = total_score ∧
    correct ≤ 20 ∧
    ∀ (c : ℕ), c > 20 →
      ¬∃ (i b : ℕ), c + i + b = total_questions ∧
                    c * correct_points + i * incorrect_points = total_score :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_l2535_253576


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2535_253510

theorem triangle_perimeter (a b c : ℝ) (ha : a = Real.sqrt 8) (hb : b = Real.sqrt 18) (hc : c = Real.sqrt 32) :
  a + b + c = 9 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2535_253510


namespace NUMINAMATH_CALUDE_travel_ratio_l2535_253527

theorem travel_ratio (total : ℕ) (europe : ℕ) (south_america : ℕ) (asia : ℕ)
  (h1 : total = 42)
  (h2 : europe = 20)
  (h3 : south_america = 10)
  (h4 : asia = 6)
  (h5 : europe + south_america + asia ≤ total) :
  asia * 2 = total - europe - south_america :=
by sorry

end NUMINAMATH_CALUDE_travel_ratio_l2535_253527


namespace NUMINAMATH_CALUDE_quadruplet_babies_l2535_253513

theorem quadruplet_babies (total_babies : ℕ) 
  (h_total : total_babies = 1250)
  (h_twins_quintuplets : ∃ t p : ℕ, t = 4 * p)
  (h_triplets_quadruplets : ∃ r q : ℕ, r = 2 * q)
  (h_quadruplets_quintuplets : ∃ q p : ℕ, q = 2 * p)
  (h_sum : ∃ t r q p : ℕ, 2 * t + 3 * r + 4 * q + 5 * p = total_babies) :
  ∃ q : ℕ, 4 * q = 303 :=
by sorry

end NUMINAMATH_CALUDE_quadruplet_babies_l2535_253513


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_y_axis_l2535_253554

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the 2D plane -/
structure Line where
  equation : ℝ → Prop

/-- Checks if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.equation p.x

/-- Checks if a line is parallel to the y-axis -/
def Line.parallelToYAxis (l : Line) : Prop :=
  ∃ k : ℝ, ∀ x y : ℝ, l.equation x ↔ x = k

theorem line_through_point_parallel_to_y_axis 
  (A : Point) 
  (h_A : A.x = -3 ∧ A.y = 1) 
  (l : Line) 
  (h_parallel : l.parallelToYAxis) 
  (h_passes : A.liesOn l) : 
  ∀ x : ℝ, l.equation x ↔ x = -3 :=
sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_y_axis_l2535_253554


namespace NUMINAMATH_CALUDE_ceiling_sum_sqrt_l2535_253533

theorem ceiling_sum_sqrt : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 27⌉ + ⌈Real.sqrt 243⌉ = 24 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_sqrt_l2535_253533


namespace NUMINAMATH_CALUDE_total_jogging_distance_l2535_253531

/-- The total distance jogged over three days is the sum of the distances jogged each day. -/
theorem total_jogging_distance 
  (monday_distance tuesday_distance wednesday_distance : ℕ) 
  (h1 : monday_distance = 2)
  (h2 : tuesday_distance = 5)
  (h3 : wednesday_distance = 9) :
  monday_distance + tuesday_distance + wednesday_distance = 16 := by
sorry

end NUMINAMATH_CALUDE_total_jogging_distance_l2535_253531


namespace NUMINAMATH_CALUDE_sphere_volume_of_inscribed_parallelepiped_l2535_253529

/-- The volume of a sphere circumscribing a rectangular parallelepiped with edge lengths 1, √2, and 3 -/
theorem sphere_volume_of_inscribed_parallelepiped : ∃ (V : ℝ),
  let a : ℝ := 1
  let b : ℝ := Real.sqrt 2
  let c : ℝ := 3
  let r : ℝ := Real.sqrt ((a^2 + b^2 + c^2) / 4)
  V = (4 / 3) * π * r^3 ∧ V = 4 * Real.sqrt 3 * π :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_of_inscribed_parallelepiped_l2535_253529


namespace NUMINAMATH_CALUDE_stanley_distance_difference_l2535_253598

-- Define the constants
def running_distance : ℝ := 4.8
def walking_distance_meters : ℝ := 950

-- Define the conversion factor
def meters_per_kilometer : ℝ := 1000

-- Define the theorem
theorem stanley_distance_difference :
  running_distance - (walking_distance_meters / meters_per_kilometer) = 3.85 := by
  sorry

end NUMINAMATH_CALUDE_stanley_distance_difference_l2535_253598


namespace NUMINAMATH_CALUDE_S_intersect_T_eq_T_l2535_253518

def S : Set ℝ := { y | ∃ x, y = 3^x }
def T : Set ℝ := { y | ∃ x, y = x^2 + 1 }

theorem S_intersect_T_eq_T : S ∩ T = T := by sorry

end NUMINAMATH_CALUDE_S_intersect_T_eq_T_l2535_253518


namespace NUMINAMATH_CALUDE_museum_visit_orders_l2535_253515

-- Define the number of museums
def n : ℕ := 5

-- Define the factorial function
def factorial (m : ℕ) : ℕ :=
  match m with
  | 0 => 1
  | k + 1 => (k + 1) * factorial k

-- Theorem: The number of permutations of n distinct objects is n!
theorem museum_visit_orders : factorial n = 120 := by
  sorry

end NUMINAMATH_CALUDE_museum_visit_orders_l2535_253515


namespace NUMINAMATH_CALUDE_prob_more_heads_10_coins_l2535_253593

def num_coins : ℕ := 10

-- Probability of getting more heads than tails
def prob_more_heads : ℚ := 193 / 512

theorem prob_more_heads_10_coins : 
  (prob_more_heads : ℚ) = 193 / 512 := by sorry

end NUMINAMATH_CALUDE_prob_more_heads_10_coins_l2535_253593


namespace NUMINAMATH_CALUDE_line_translations_l2535_253586

/-- Represents a line in the form y = mx + b --/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Translates a line vertically --/
def translateVertical (l : Line) (dy : ℝ) : Line :=
  { m := l.m, b := l.b + dy }

/-- Translates a line horizontally --/
def translateHorizontal (l : Line) (dx : ℝ) : Line :=
  { m := l.m, b := l.b - l.m * dx }

theorem line_translations (original : Line) :
  (original.m = 2 ∧ original.b = -4) →
  (translateVertical original 3 = { m := 2, b := -1 } ∧
   translateHorizontal original 3 = { m := 2, b := -10 }) :=
by sorry

end NUMINAMATH_CALUDE_line_translations_l2535_253586


namespace NUMINAMATH_CALUDE_winnings_proof_l2535_253580

theorem winnings_proof (total : ℝ) 
  (h1 : total > 0)
  (h2 : total / 4 + total / 7 + 17 = total) : 
  total = 28 := by
sorry

end NUMINAMATH_CALUDE_winnings_proof_l2535_253580


namespace NUMINAMATH_CALUDE_largest_gold_coin_distribution_l2535_253520

theorem largest_gold_coin_distribution (n : ℕ) : 
  (∃ k : ℕ, n = 13 * k + 3) → 
  n < 110 → 
  (∀ m : ℕ, (∃ j : ℕ, m = 13 * j + 3) → m < 110 → m ≤ n) →
  n = 107 := by
  sorry

end NUMINAMATH_CALUDE_largest_gold_coin_distribution_l2535_253520


namespace NUMINAMATH_CALUDE_hip_hop_class_cost_l2535_253548

/-- The cost of one hip-hop class -/
def hip_hop_cost : ℕ := sorry

/-- The cost of one ballet class -/
def ballet_cost : ℕ := 12

/-- The cost of one jazz class -/
def jazz_cost : ℕ := 8

/-- The total number of hip-hop classes per week -/
def hip_hop_classes : ℕ := 2

/-- The total number of ballet classes per week -/
def ballet_classes : ℕ := 2

/-- The total number of jazz classes per week -/
def jazz_classes : ℕ := 1

/-- The total cost of all classes per week -/
def total_cost : ℕ := 52

theorem hip_hop_class_cost :
  hip_hop_cost * hip_hop_classes + ballet_cost * ballet_classes + jazz_cost * jazz_classes = total_cost ∧
  hip_hop_cost = 10 :=
sorry

end NUMINAMATH_CALUDE_hip_hop_class_cost_l2535_253548


namespace NUMINAMATH_CALUDE_additional_men_needed_l2535_253568

/-- Proves that given a work that can be finished by 12 men in 11 days,
    if the work is completed in 8 days (3 days earlier),
    then the number of additional men needed is 5. -/
theorem additional_men_needed
  (original_days : ℕ)
  (original_men : ℕ)
  (actual_days : ℕ)
  (h1 : original_days = 11)
  (h2 : original_men = 12)
  (h3 : actual_days = original_days - 3)
  : ∃ (additional_men : ℕ), 
    (original_men * original_days = (original_men + additional_men) * actual_days) ∧
    additional_men = 5 := by
  sorry

end NUMINAMATH_CALUDE_additional_men_needed_l2535_253568


namespace NUMINAMATH_CALUDE_toys_in_box_time_l2535_253546

/-- The time required to put all toys in the box -/
def time_to_put_toys_in_box (total_toys : ℕ) (net_increase_per_minute : ℕ) : ℕ :=
  ((total_toys - net_increase_per_minute) / net_increase_per_minute) + 1

/-- Theorem: It takes 15 minutes to put 45 toys in the box with a net increase of 3 toys per minute -/
theorem toys_in_box_time : time_to_put_toys_in_box 45 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_toys_in_box_time_l2535_253546


namespace NUMINAMATH_CALUDE_students_per_school_l2535_253589

theorem students_per_school (total_schools : ℕ) (total_students : ℕ) 
  (h1 : total_schools = 25) (h2 : total_students = 6175) : 
  total_students / total_schools = 247 := by
  sorry

end NUMINAMATH_CALUDE_students_per_school_l2535_253589


namespace NUMINAMATH_CALUDE_initial_apples_l2535_253584

theorem initial_apples (initial : ℝ) (received : ℝ) (total : ℝ) : 
  received = 7.0 → total = 27 → initial + received = total → initial = 20.0 := by
sorry

end NUMINAMATH_CALUDE_initial_apples_l2535_253584


namespace NUMINAMATH_CALUDE_tulip_count_after_addition_tulip_count_is_24_l2535_253583

/-- Given a garden with tulips and sunflowers, prove the number of tulips after an addition of sunflowers. -/
theorem tulip_count_after_addition 
  (initial_ratio : Rat) 
  (initial_sunflowers : Nat) 
  (added_sunflowers : Nat) : Nat :=
  let final_sunflowers := initial_sunflowers + added_sunflowers
  let tulip_ratio := 3
  let sunflower_ratio := 7
  (tulip_ratio * final_sunflowers) / sunflower_ratio

#check tulip_count_after_addition (3/7) 42 14 = 24

/-- Prove that the result is indeed 24 -/
theorem tulip_count_is_24 : 
  tulip_count_after_addition (3/7) 42 14 = 24 := by
  sorry


end NUMINAMATH_CALUDE_tulip_count_after_addition_tulip_count_is_24_l2535_253583


namespace NUMINAMATH_CALUDE_mom_bought_71_packages_l2535_253557

/-- The number of t-shirts in each package -/
def shirts_per_package : ℕ := 6

/-- The total number of t-shirts Mom has -/
def total_shirts : ℕ := 426

/-- The number of packages Mom bought -/
def packages_bought : ℕ := total_shirts / shirts_per_package

theorem mom_bought_71_packages : packages_bought = 71 := by
  sorry

end NUMINAMATH_CALUDE_mom_bought_71_packages_l2535_253557


namespace NUMINAMATH_CALUDE_captain_america_awakening_year_l2535_253585

theorem captain_america_awakening_year : 2019 * 0.313 + 2.019 * 687 = 2018 := by
  sorry

end NUMINAMATH_CALUDE_captain_america_awakening_year_l2535_253585


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l2535_253570

/-- A regular polygon with exterior angles of 45 degrees has interior angle sum of 1080 degrees -/
theorem regular_polygon_interior_angle_sum :
  ∀ (n : ℕ), 
  n > 2 →
  (360 : ℝ) / n = 45 →
  (n - 2) * 180 = 1080 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l2535_253570


namespace NUMINAMATH_CALUDE_inequality_proof_l2535_253507

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  1/x + 1/y ≥ 3 + 2*Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2535_253507


namespace NUMINAMATH_CALUDE_integer_root_of_polynomial_l2535_253536

-- Define the polynomial
def P (x b c : ℚ) : ℚ := x^4 + 7*x^3 + b*x + c

-- State the theorem
theorem integer_root_of_polynomial (b c : ℚ) :
  (∃ (r : ℚ), r^2 = 5 ∧ P (2 + r) b c = 0) →  -- 2 + √5 is a root
  (∃ (n : ℤ), P n b c = 0) →                  -- There exists an integer root
  P 0 b c = 0                                 -- 0 is a root
:= by sorry

end NUMINAMATH_CALUDE_integer_root_of_polynomial_l2535_253536


namespace NUMINAMATH_CALUDE_students_in_line_l2535_253579

theorem students_in_line (front : ℕ) (behind : ℕ) (taehyung : ℕ) 
  (h1 : front = 9) 
  (h2 : behind = 16) 
  (h3 : taehyung = 1) : 
  front + behind + taehyung = 26 := by
  sorry

end NUMINAMATH_CALUDE_students_in_line_l2535_253579


namespace NUMINAMATH_CALUDE_binomial_30_3_l2535_253552

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end NUMINAMATH_CALUDE_binomial_30_3_l2535_253552


namespace NUMINAMATH_CALUDE_train_length_l2535_253526

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) :
  speed = 36 →  -- speed in km/hr
  time = 9 →    -- time in seconds
  speed * (time / 3600) = 90 / 1000 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2535_253526


namespace NUMINAMATH_CALUDE_carpenter_square_problem_l2535_253544

theorem carpenter_square_problem (s : ℝ) :
  (s^2 - 4 * (0.09 * s^2) = 256) → s = 20 := by
  sorry

end NUMINAMATH_CALUDE_carpenter_square_problem_l2535_253544


namespace NUMINAMATH_CALUDE_successive_projections_l2535_253508

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Projection of a point onto the xOy plane -/
def proj_xOy (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := 0 }

/-- Projection of a point onto the yOz plane -/
def proj_yOz (p : Point3D) : Point3D :=
  { x := 0, y := p.y, z := p.z }

/-- Projection of a point onto the xOz plane -/
def proj_xOz (p : Point3D) : Point3D :=
  { x := p.x, y := 0, z := p.z }

/-- The origin (0, 0, 0) -/
def origin : Point3D :=
  { x := 0, y := 0, z := 0 }

theorem successive_projections (M : Point3D) :
  proj_xOz (proj_yOz (proj_xOy M)) = origin := by
  sorry

end NUMINAMATH_CALUDE_successive_projections_l2535_253508


namespace NUMINAMATH_CALUDE_square_pattern_l2535_253504

theorem square_pattern (n : ℕ) : (n - 1) * (n + 1) + 1 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_square_pattern_l2535_253504


namespace NUMINAMATH_CALUDE_annual_salary_is_20_l2535_253592

/-- Represents the total annual cash salary in rupees -/
def annual_salary : ℕ := sorry

/-- Represents the number of months the servant worked -/
def months_worked : ℕ := 9

/-- Represents the total amount received by the servant after 9 months in rupees -/
def amount_received : ℕ := 55

/-- Represents the price of the turban in rupees -/
def turban_price : ℕ := 50

/-- Theorem stating that the annual salary is 20 rupees -/
theorem annual_salary_is_20 :
  annual_salary = 20 :=
by sorry

end NUMINAMATH_CALUDE_annual_salary_is_20_l2535_253592


namespace NUMINAMATH_CALUDE_x_greater_than_one_l2535_253543

theorem x_greater_than_one (x : ℝ) (h : Real.log x > 0) : x > 1 := by
  sorry

end NUMINAMATH_CALUDE_x_greater_than_one_l2535_253543


namespace NUMINAMATH_CALUDE_zeros_after_decimal_point_l2535_253596

-- Define the fraction
def fraction : ℚ := 3 / (25^25)

-- Define the number of zeros after the decimal point
def num_zeros : ℕ := 18

-- Theorem statement
theorem zeros_after_decimal_point :
  (fraction * (10^num_zeros)).floor = 0 ∧
  (fraction * (10^(num_zeros + 1))).floor ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_zeros_after_decimal_point_l2535_253596


namespace NUMINAMATH_CALUDE_M_less_than_N_l2535_253545

theorem M_less_than_N (x y : ℝ) (α : ℝ) (hx : x > 0) (hy : y > 0) :
  x^(Real.sin α)^2 * y^(Real.cos α)^2 < x + y := by
  sorry

end NUMINAMATH_CALUDE_M_less_than_N_l2535_253545


namespace NUMINAMATH_CALUDE_books_read_ratio_l2535_253538

theorem books_read_ratio : 
  let william_last_month : ℕ := 6
  let brad_this_month : ℕ := 8
  let william_this_month : ℕ := 2 * brad_this_month
  let william_total : ℕ := william_last_month + william_this_month
  let brad_total : ℕ := william_total - 4
  let brad_last_month : ℕ := brad_total - brad_this_month
  ∃ (a b : ℕ), a * william_last_month = b * brad_last_month ∧ a = 3 ∧ b = 5 :=
by
  sorry


end NUMINAMATH_CALUDE_books_read_ratio_l2535_253538


namespace NUMINAMATH_CALUDE_two_distinct_solutions_l2535_253582

/-- The cubic equation in x with parameter a -/
def cubic_equation (a x : ℝ) : ℝ := x^3 - 2*a*x^2 - 3*a*x + a^2 - 2

/-- Theorem stating the condition for the cubic equation to have exactly two distinct real solutions -/
theorem two_distinct_solutions (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    cubic_equation a x = 0 ∧ 
    cubic_equation a y = 0 ∧ 
    ∀ z : ℝ, cubic_equation a z = 0 → z = x ∨ z = y) ↔ 
  a > 15/8 :=
sorry

end NUMINAMATH_CALUDE_two_distinct_solutions_l2535_253582


namespace NUMINAMATH_CALUDE_inequality_range_of_p_l2535_253519

-- Define the inequality function
def inequality (x p : ℝ) : Prop := x^2 + p*x + 1 > 2*x + p

-- Define the theorem
theorem inequality_range_of_p :
  ∀ p : ℝ, (∀ x : ℝ, 2 ≤ x ∧ x ≤ 4 → inequality x p) → p > -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_of_p_l2535_253519


namespace NUMINAMATH_CALUDE_correct_matching_probability_l2535_253563

theorem correct_matching_probability (n : ℕ) (h : n = 4) : 
  (1 : ℚ) / (Nat.factorial n) = (1 : ℚ) / 24 :=
by sorry

end NUMINAMATH_CALUDE_correct_matching_probability_l2535_253563


namespace NUMINAMATH_CALUDE_all_statements_imply_p_and_q_implies_r_l2535_253501

theorem all_statements_imply_p_and_q_implies_r (p q r : Prop) :
  ((p ∧ q ∧ r) → ((p ∧ q) → r)) ∧
  ((¬p ∧ q ∧ ¬r) → ((p ∧ q) → r)) ∧
  ((p ∧ ¬q ∧ ¬r) → ((p ∧ q) → r)) ∧
  ((¬p ∧ ¬q ∧ r) → ((p ∧ q) → r)) :=
by sorry

end NUMINAMATH_CALUDE_all_statements_imply_p_and_q_implies_r_l2535_253501


namespace NUMINAMATH_CALUDE_min_distance_points_l2535_253528

theorem min_distance_points (a b : ℝ) : 
  a = 2 → 
  (∃ (min_val : ℝ), min_val = 7 ∧ 
    ∀ (x : ℝ), |x - a| + |x - b| ≥ min_val) → 
  (b = -5 ∨ b = 9) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_points_l2535_253528


namespace NUMINAMATH_CALUDE_simplify_P_P_value_on_inverse_proportion_l2535_253560

/-- Simplification of the expression P -/
theorem simplify_P (a b : ℝ) :
  (2*a + 3*b)^2 - (2*a + b)*(2*a - b) - 2*b*(3*a + 5*b) = 6*a*b := by sorry

/-- Value of P when (a,b) lies on y = -2/x -/
theorem P_value_on_inverse_proportion (a b : ℝ) (h : a*b = -2) :
  6*a*b = -12 := by sorry

end NUMINAMATH_CALUDE_simplify_P_P_value_on_inverse_proportion_l2535_253560


namespace NUMINAMATH_CALUDE_lighter_box_weight_l2535_253566

/-- Proves that the weight of lighter boxes is 12 pounds given the conditions of the shipment. -/
theorem lighter_box_weight (total_boxes : Nat) (heavier_box_weight : Nat) (initial_average : Nat) 
  (final_average : Nat) (removed_boxes : Nat) :
  total_boxes = 20 →
  heavier_box_weight = 20 →
  initial_average = 18 →
  final_average = 12 →
  removed_boxes = 15 →
  ∃ (lighter_box_weight : Nat), 
    lighter_box_weight = 12 ∧
    lighter_box_weight * (total_boxes - removed_boxes) = 
      final_average * (total_boxes - removed_boxes) :=
by sorry

end NUMINAMATH_CALUDE_lighter_box_weight_l2535_253566


namespace NUMINAMATH_CALUDE_gas_price_increase_l2535_253505

theorem gas_price_increase (P : ℝ) (h : P > 0) : 
  let first_increase := 0.15
  let consumption_reduction := 0.20948616600790515
  let second_increase := 0.1
  let final_price := P * (1 + first_increase) * (1 + second_increase)
  let reduced_consumption := 1 - consumption_reduction
  reduced_consumption * final_price = P :=
by sorry

end NUMINAMATH_CALUDE_gas_price_increase_l2535_253505


namespace NUMINAMATH_CALUDE_fencing_cost_theorem_l2535_253549

/-- Represents a pentagon with side lengths and angles -/
structure Pentagon where
  ab : ℝ
  bc : ℝ
  cd : ℝ
  de : ℝ
  ae : ℝ
  angle_bac : ℝ
  angle_abc : ℝ
  angle_bcd : ℝ
  angle_cde : ℝ
  angle_dea : ℝ

/-- Calculate the total cost of fencing a pentagon -/
def fencingCost (p : Pentagon) (costPerMeter : ℝ) : ℝ :=
  (p.ab + p.bc + p.cd + p.de + p.ae) * costPerMeter

/-- Theorem: The total cost of fencing the given irregular pentagon is Rs. 300 -/
theorem fencing_cost_theorem (p : Pentagon) (h1 : p.ab = 20)
    (h2 : p.bc = 25) (h3 : p.cd = 30) (h4 : p.de = 35) (h5 : p.ae = 40)
    (h6 : p.angle_bac = 110) (h7 : p.angle_abc = 95) (h8 : p.angle_bcd = 100)
    (h9 : p.angle_cde = 105) (h10 : p.angle_dea = 115) :
    fencingCost p 2 = 300 := by
  sorry

#check fencing_cost_theorem

end NUMINAMATH_CALUDE_fencing_cost_theorem_l2535_253549


namespace NUMINAMATH_CALUDE_polynomial_coefficient_values_l2535_253535

theorem polynomial_coefficient_values (a₅ a₄ a₃ a₂ a₁ a₀ : ℝ) :
  (∀ x : ℝ, x^5 = a₅*(2*x+1)^5 + a₄*(2*x+1)^4 + a₃*(2*x+1)^3 + a₂*(2*x+1)^2 + a₁*(2*x+1) + a₀) →
  a₅ = 1/32 ∧ a₄ = -5/32 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_values_l2535_253535


namespace NUMINAMATH_CALUDE_f_recursive_relation_l2535_253524

/-- The smallest integer such that any permutation on n elements, repeated f(n) times, gives the identity. -/
def f (n : ℕ) : ℕ := sorry

/-- Checks if a number is a prime power -/
def isPrimePower (n : ℕ) : Prop := sorry

/-- The prime base of a prime power -/
def primeBase (n : ℕ) : ℕ := sorry

theorem f_recursive_relation (n : ℕ) :
  (isPrimePower n → f n = primeBase n * f (n - 1)) ∧
  (¬isPrimePower n → f n = f (n - 1)) := by sorry

end NUMINAMATH_CALUDE_f_recursive_relation_l2535_253524


namespace NUMINAMATH_CALUDE_whole_number_between_36_and_40_l2535_253588

theorem whole_number_between_36_and_40 (M : ℤ) :
  (9 < M / 4 ∧ M / 4 < 10) → (M = 37 ∨ M = 38 ∨ M = 39) := by
  sorry

end NUMINAMATH_CALUDE_whole_number_between_36_and_40_l2535_253588


namespace NUMINAMATH_CALUDE_aunt_wang_earnings_l2535_253595

/-- Calculate simple interest earnings -/
def simple_interest_earnings (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem aunt_wang_earnings :
  let principal : ℝ := 50000
  let rate : ℝ := 0.045
  let time : ℝ := 2
  simple_interest_earnings principal rate time = 4500 := by
sorry

end NUMINAMATH_CALUDE_aunt_wang_earnings_l2535_253595


namespace NUMINAMATH_CALUDE_ellipse_equation_l2535_253537

/-- The equation of an ellipse given its foci and the sum of distances from any point on the ellipse to the foci -/
theorem ellipse_equation (F₁ F₂ M : ℝ × ℝ) (d : ℝ) : 
  F₁ = (-4, 0) →
  F₂ = (4, 0) →
  d = 10 →
  (dist M F₁ + dist M F₂ = d) →
  (M.1^2 / 25 + M.2^2 / 9 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2535_253537


namespace NUMINAMATH_CALUDE_danny_soda_problem_l2535_253594

theorem danny_soda_problem :
  let total_bottles : ℝ := 3
  let drunk_percentage : ℝ := 0.9
  let given_away_percentage : ℝ := 0.7
  let remaining_percentage : ℝ := 
    (total_bottles - drunk_percentage * 1 - given_away_percentage * 2) / total_bottles * 100
  remaining_percentage = 70 := by
sorry

end NUMINAMATH_CALUDE_danny_soda_problem_l2535_253594


namespace NUMINAMATH_CALUDE_odd_function_power_l2535_253502

def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| - |x - a|

theorem odd_function_power (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →  -- f is an odd function
  (∃ x, f a x ≠ 0) →          -- f is not identically zero
  a^2012 = 1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_power_l2535_253502


namespace NUMINAMATH_CALUDE_max_value_of_f_l2535_253553

/-- The quadratic function f(x) = -5x^2 + 25x - 7 -/
def f (x : ℝ) : ℝ := -5 * x^2 + 25 * x - 7

/-- The maximum value of f(x) is 53/4 -/
theorem max_value_of_f :
  ∃ (M : ℝ), M = 53/4 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2535_253553


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l2535_253567

def i : ℂ := Complex.I

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem pure_imaginary_condition (a : ℝ) :
  let Z : ℂ := (a + i) / (1 + i)
  is_pure_imaginary Z → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l2535_253567


namespace NUMINAMATH_CALUDE_product_equals_442_l2535_253590

/-- Converts a list of digits in a given base to its decimal (base 10) representation -/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base^i) 0

/-- The binary representation of the first number -/
def binary_num : List Nat := [1, 0, 1, 1]

/-- The ternary representation of the second number -/
def ternary_num : List Nat := [1, 2, 0, 1]

theorem product_equals_442 :
  (to_decimal binary_num 2) * (to_decimal ternary_num 3) = 442 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_442_l2535_253590


namespace NUMINAMATH_CALUDE_orange_distribution_l2535_253516

theorem orange_distribution (x : ℚ) : 
  (x/2 + 1/2) + (1/2 * (x/2 - 1/2) + 1/2) + (1/2 * (x/4 - 3/4) + 1/2) = x → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_orange_distribution_l2535_253516


namespace NUMINAMATH_CALUDE_probability_ten_nine_eight_sequence_l2535_253530

theorem probability_ten_nine_eight_sequence (deck : Nat) (tens : Nat) (nines : Nat) (eights : Nat) :
  deck = 52 →
  tens = 4 →
  nines = 4 →
  eights = 4 →
  (tens / deck) * (nines / (deck - 1)) * (eights / (deck - 2)) = 4 / 33150 := by
  sorry

end NUMINAMATH_CALUDE_probability_ten_nine_eight_sequence_l2535_253530


namespace NUMINAMATH_CALUDE_unique_hyperdeficient_l2535_253562

/-- Sum of all divisors of n including n itself -/
def g (n : ℕ) : ℕ := sorry

/-- A number n is hyperdeficient if g(g(n)) = n + 3 -/
def is_hyperdeficient (n : ℕ) : Prop := g (g n) = n + 3

/-- There exists exactly one hyperdeficient positive integer -/
theorem unique_hyperdeficient : ∃! n : ℕ+, is_hyperdeficient n := by sorry

end NUMINAMATH_CALUDE_unique_hyperdeficient_l2535_253562


namespace NUMINAMATH_CALUDE_monochromatic_isosceles_independent_of_coloring_l2535_253555

/-- Represents a regular polygon with 6n+1 sides and a coloring of its vertices -/
structure ColoredPolygon (n : ℕ) where
  k : ℕ
  h : ℕ := 6 * n + 1 - k
  k_valid : k ≤ 6 * n + 1

/-- Counts the number of monochromatic isosceles triangles in a colored polygon -/
def monochromaticIsoscelesCount (p : ColoredPolygon n) : ℚ :=
  (1 / 2) * (p.h * (p.h - 1) + p.k * (p.k - 1) - p.k * p.h)

/-- Theorem stating that the number of monochromatic isosceles triangles is independent of coloring -/
theorem monochromatic_isosceles_independent_of_coloring (n : ℕ) :
  ∀ p q : ColoredPolygon n, monochromaticIsoscelesCount p = monochromaticIsoscelesCount q :=
sorry

end NUMINAMATH_CALUDE_monochromatic_isosceles_independent_of_coloring_l2535_253555


namespace NUMINAMATH_CALUDE_square_difference_ben_subtraction_l2535_253522

theorem square_difference (n : ℕ) : (n - 1)^2 = n^2 - (2*n - 1) := by sorry

theorem ben_subtraction : 49^2 = 50^2 - 99 := by sorry

end NUMINAMATH_CALUDE_square_difference_ben_subtraction_l2535_253522


namespace NUMINAMATH_CALUDE_melany_money_theorem_l2535_253512

/-- The amount of money Melany initially had to fence a square field --/
def melany_initial_money (field_size : ℕ) (wire_cost_per_foot : ℕ) (unfenced_length : ℕ) : ℕ :=
  (field_size - unfenced_length) * wire_cost_per_foot

/-- Theorem stating that Melany's initial money was $120,000 --/
theorem melany_money_theorem (field_size : ℕ) (wire_cost_per_foot : ℕ) (unfenced_length : ℕ) 
  (h1 : field_size = 5000)
  (h2 : wire_cost_per_foot = 30)
  (h3 : unfenced_length = 1000) :
  melany_initial_money field_size wire_cost_per_foot unfenced_length = 120000 := by
  sorry

end NUMINAMATH_CALUDE_melany_money_theorem_l2535_253512


namespace NUMINAMATH_CALUDE_fraction_equality_l2535_253565

/-- Given that (Bx-13)/(x^2-7x+10) = A/(x-2) + 5/(x-5) for all x ≠ 2 and x ≠ 5,
    prove that A = 3/5, B = 28/5, and A + B = 31/5 -/
theorem fraction_equality (A B : ℚ) : 
  (∀ x : ℚ, x ≠ 2 → x ≠ 5 → (B * x - 13) / (x^2 - 7*x + 10) = A / (x - 2) + 5 / (x - 5)) →
  A = 3/5 ∧ B = 28/5 ∧ A + B = 31/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2535_253565


namespace NUMINAMATH_CALUDE_ammonia_formed_l2535_253591

-- Define the chemical species
structure ChemicalSpecies where
  name : String
  coefficient : ℕ

-- Define the chemical equation
structure ChemicalEquation where
  reactants : List ChemicalSpecies
  products : List ChemicalSpecies

-- Define the reaction conditions
structure ReactionConditions where
  li3n_amount : ℚ
  h2o_amount : ℚ
  lioh_amount : ℚ

-- Define the balanced equation
def balanced_equation : ChemicalEquation :=
  { reactants := [
      { name := "Li3N", coefficient := 1 },
      { name := "H2O", coefficient := 3 }
    ],
    products := [
      { name := "LiOH", coefficient := 3 },
      { name := "NH3", coefficient := 1 }
    ]
  }

-- Define the reaction conditions
def reaction_conditions : ReactionConditions :=
  { li3n_amount := 1,
    h2o_amount := 54,
    lioh_amount := 3 }

-- Theorem statement
theorem ammonia_formed (eq : ChemicalEquation) (conditions : ReactionConditions) :
  eq = balanced_equation ∧
  conditions = reaction_conditions →
  ∃ (nh3_amount : ℚ), nh3_amount = 1 :=
sorry

end NUMINAMATH_CALUDE_ammonia_formed_l2535_253591


namespace NUMINAMATH_CALUDE_bananas_per_friend_l2535_253517

/-- Given Virginia has 40 bananas and shares them equally among 40 friends,
    prove that each friend receives 1 banana. -/
theorem bananas_per_friend (total_bananas : ℕ) (num_friends : ℕ) 
  (h1 : total_bananas = 40) (h2 : num_friends = 40) :
  total_bananas / num_friends = 1 := by
  sorry

end NUMINAMATH_CALUDE_bananas_per_friend_l2535_253517


namespace NUMINAMATH_CALUDE_correct_mean_problem_l2535_253572

def correct_mean (n : ℕ) (original_mean : ℚ) (wrong_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n * original_mean - wrong_value + correct_value) / n

theorem correct_mean_problem :
  correct_mean 50 41 23 48 = 41.5 := by
  sorry

end NUMINAMATH_CALUDE_correct_mean_problem_l2535_253572


namespace NUMINAMATH_CALUDE_clubsuit_calculation_l2535_253511

/-- Custom operation ⊗ for real numbers -/
def clubsuit (a b : ℝ) : ℝ := (a + b)^2 - (a - b)^2

/-- Theorem stating that 5 ⊗ (7 ⊗ 8) = 4480 -/
theorem clubsuit_calculation : clubsuit 5 (clubsuit 7 8) = 4480 := by
  sorry

end NUMINAMATH_CALUDE_clubsuit_calculation_l2535_253511


namespace NUMINAMATH_CALUDE_base_10_1234_equals_base_7_3412_l2535_253523

def base_10_to_base_7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec to_digits (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else to_digits (m / 7) ((m % 7) :: acc)
  to_digits n []

theorem base_10_1234_equals_base_7_3412 :
  base_10_to_base_7 1234 = [3, 4, 1, 2] := by sorry

end NUMINAMATH_CALUDE_base_10_1234_equals_base_7_3412_l2535_253523


namespace NUMINAMATH_CALUDE_no_roots_composition_l2535_253559

/-- A quadratic polynomial f(x) = ax^2 + bx + c -/
structure QuadraticPolynomial (α : Type*) [Ring α] where
  a : α
  b : α
  c : α

/-- The function represented by a quadratic polynomial -/
def evalQuadratic {α : Type*} [Ring α] (f : QuadraticPolynomial α) (x : α) : α :=
  f.a * x * x + f.b * x + f.c

theorem no_roots_composition {α : Type*} [LinearOrderedField α] (f : QuadraticPolynomial α) :
  (∀ x : α, evalQuadratic f x ≠ x) →
  (∀ x : α, evalQuadratic f (evalQuadratic f x) ≠ x) := by
  sorry

end NUMINAMATH_CALUDE_no_roots_composition_l2535_253559


namespace NUMINAMATH_CALUDE_min_value_complex_sum_l2535_253514

theorem min_value_complex_sum (z : ℂ) (h : Complex.abs (z - (3 - 3*I)) = 4) :
  Complex.abs (z + (2 - I))^2 + Complex.abs (z - (6 - 5*I))^2 ≥ 76 :=
by sorry

end NUMINAMATH_CALUDE_min_value_complex_sum_l2535_253514


namespace NUMINAMATH_CALUDE_probability_one_white_ball_l2535_253532

/-- The probability of drawing exactly one white ball when drawing three balls from a bag containing
    four white balls and three black balls of the same size. -/
theorem probability_one_white_ball (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) 
  (h1 : total_balls = white_balls + black_balls)
  (h2 : white_balls = 4)
  (h3 : black_balls = 3)
  (h4 : total_balls > 0) :
  (white_balls : ℚ) / total_balls * 
  (black_balls : ℚ) / (total_balls - 1) * 
  (black_balls - 1 : ℚ) / (total_balls - 2) * 3 = 12 / 35 :=
sorry

end NUMINAMATH_CALUDE_probability_one_white_ball_l2535_253532


namespace NUMINAMATH_CALUDE_min_value_fraction_l2535_253564

theorem min_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z)^3 / ((x + y)^3 * (y + z)^3) ≥ 27/8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2535_253564


namespace NUMINAMATH_CALUDE_candidates_per_state_l2535_253534

theorem candidates_per_state (total_candidates : ℕ) : 
  (total_candidates : ℝ) * 0.06 + 80 = total_candidates * 0.07 → 
  total_candidates = 8000 := by
  sorry

end NUMINAMATH_CALUDE_candidates_per_state_l2535_253534


namespace NUMINAMATH_CALUDE_factorization_equality_l2535_253503

theorem factorization_equality (a b : ℝ) : 2*a^3 - 8*a*b^2 = 2*a*(a+2*b)*(a-2*b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2535_253503


namespace NUMINAMATH_CALUDE_range_of_expression_l2535_253597

theorem range_of_expression (x y : ℝ) (h : x^2 + y^2 - 6*x = 0) :
  ∃ (min max : ℝ), min = Real.sqrt 5 ∧ max = Real.sqrt 53 ∧
  min ≤ Real.sqrt (2*x^2 + y^2 - 4*x + 5) ∧
  Real.sqrt (2*x^2 + y^2 - 4*x + 5) ≤ max :=
sorry

end NUMINAMATH_CALUDE_range_of_expression_l2535_253597


namespace NUMINAMATH_CALUDE_area_ratio_triangle_to_hexagon_l2535_253573

/-- A regular hexagon ABCDEF with vertices A, B, C, D, E, F -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_regular : sorry

/-- The area of a regular hexagon -/
def area_hexagon (h : RegularHexagon) : ℝ := sorry

/-- The area of triangle ACE in a regular hexagon -/
def area_triangle_ACE (h : RegularHexagon) : ℝ := sorry

/-- Theorem: The area of triangle ACE is 2/3 of the area of the regular hexagon -/
theorem area_ratio_triangle_to_hexagon (h : RegularHexagon) :
  area_triangle_ACE h / area_hexagon h = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_area_ratio_triangle_to_hexagon_l2535_253573


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l2535_253551

theorem simplify_trig_expression :
  Real.sqrt (1 - 2 * Real.sin (35 * π / 180) * Real.cos (35 * π / 180)) =
  Real.cos (35 * π / 180) - Real.sin (35 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l2535_253551


namespace NUMINAMATH_CALUDE_line_slope_and_inclination_l2535_253561

def line_equation (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 1 = 0

theorem line_slope_and_inclination :
  ∃ (m θ : ℝ), 
    (∀ x y, line_equation x y → y = m * x + (1 / Real.sqrt 3)) ∧
    m = -Real.sqrt 3 / 3 ∧
    θ = 5 * π / 6 ∧
    Real.tan θ = m := by
  sorry

end NUMINAMATH_CALUDE_line_slope_and_inclination_l2535_253561


namespace NUMINAMATH_CALUDE_expression_evaluation_l2535_253571

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -1
  (6 * x^2 * y * (-2 * x * y + y^3)) / (x * y^2) = -36 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2535_253571
