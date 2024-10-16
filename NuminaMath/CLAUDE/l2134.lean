import Mathlib

namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_negation_set_equivalence_l2134_213426

theorem necessary_not_sufficient_condition (a b : ℝ) :
  (a > b → a + 1 > b) ∧ ¬(a + 1 > b → a > b) := by sorry

theorem negation_set_equivalence :
  {x : ℝ | ¬(1 / (x - 2) > 0)} = {x : ℝ | x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_negation_set_equivalence_l2134_213426


namespace NUMINAMATH_CALUDE_min_time_for_eight_people_l2134_213472

/-- Represents a group of people sharing information -/
structure InformationSharingGroup where
  numPeople : Nat
  callDuration : Nat
  initialInfo : Fin numPeople → Nat

/-- Represents the minimum time needed for complete information sharing -/
def minTimeForCompleteSharing (group : InformationSharingGroup) : Nat :=
  sorry

/-- Theorem stating the minimum time for the specific problem -/
theorem min_time_for_eight_people
  (group : InformationSharingGroup)
  (h1 : group.numPeople = 8)
  (h2 : group.callDuration = 3)
  (h3 : ∀ i j : Fin group.numPeople, i ≠ j → group.initialInfo i ≠ group.initialInfo j) :
  minTimeForCompleteSharing group = 9 :=
sorry

end NUMINAMATH_CALUDE_min_time_for_eight_people_l2134_213472


namespace NUMINAMATH_CALUDE_arithmetic_sequence_specific_values_l2134_213468

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- The sum function
  sum_property : ∀ n, S n = n * (a 1 + a n) / 2  -- Property of sum of arithmetic sequence
  arithmetic_property : ∀ n, a (n + 1) - a n = a 2 - a 1  -- Property of arithmetic sequence

/-- Main theorem about specific values in an arithmetic sequence -/
theorem arithmetic_sequence_specific_values (seq : ArithmeticSequence) 
    (h1 : seq.S 9 = -36) (h2 : seq.S 13 = -104) : 
    seq.a 5 = -4 ∧ seq.S 11 = -66 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_specific_values_l2134_213468


namespace NUMINAMATH_CALUDE_train_length_proof_l2134_213458

/-- Given a train that passes a pole in 10 seconds and a 1250m long platform in 60 seconds,
    prove that the length of the train is 250 meters. -/
theorem train_length_proof (pole_time : ℝ) (platform_time : ℝ) (platform_length : ℝ) 
    (h1 : pole_time = 10)
    (h2 : platform_time = 60)
    (h3 : platform_length = 1250) : 
  ∃ (train_length : ℝ), train_length = 250 ∧ 
    train_length / pole_time = (train_length + platform_length) / platform_time :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_proof_l2134_213458


namespace NUMINAMATH_CALUDE_hex_fraction_sum_max_l2134_213403

theorem hex_fraction_sum_max (a b c : ℕ) (y : ℕ) (h1 : a ≤ 15) (h2 : b ≤ 15) (h3 : c ≤ 15)
  (h4 : (a * 256 + b * 16 + c : ℕ) = 4096 / y) (h5 : 0 < y) (h6 : y ≤ 16) :
  a + b + c ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_hex_fraction_sum_max_l2134_213403


namespace NUMINAMATH_CALUDE_compute_expression_l2134_213400

theorem compute_expression : 4 * 6 * 8 - 24 / 3 = 184 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l2134_213400


namespace NUMINAMATH_CALUDE_always_two_distinct_roots_find_m_value_l2134_213441

/-- The quadratic equation x^2 - (2m + 1)x - 2 = 0 -/
def quadratic (m : ℝ) (x : ℝ) : Prop :=
  x^2 - (2*m + 1)*x - 2 = 0

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ :=
  (2*m + 1)^2 + 8

theorem always_two_distinct_roots (m : ℝ) :
  discriminant m > 0 :=
sorry

theorem find_m_value (m : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : quadratic m x₁) 
  (h₂ : quadratic m x₂) 
  (h₃ : x₁ + x₂ + x₁*x₂ = 1) :
  m = 1 :=
sorry

end NUMINAMATH_CALUDE_always_two_distinct_roots_find_m_value_l2134_213441


namespace NUMINAMATH_CALUDE_minimum_k_value_l2134_213480

theorem minimum_k_value (m n : ℕ+) :
  (1 : ℝ) / (m + n : ℝ)^2 ≤ (1/8) * ((1 : ℝ) / m^2 + 1 / n^2) ∧
  ∀ k : ℝ, (∀ a b : ℕ+, (1 : ℝ) / (a + b : ℝ)^2 ≤ k * ((1 : ℝ) / a^2 + 1 / b^2)) →
    k ≥ 1/8 :=
by sorry

end NUMINAMATH_CALUDE_minimum_k_value_l2134_213480


namespace NUMINAMATH_CALUDE_line_point_sum_l2134_213439

/-- The line equation y = -1/2x + 8 --/
def line_equation (x y : ℝ) : Prop := y = -1/2 * x + 8

/-- Point P is where the line crosses the x-axis --/
def P : ℝ × ℝ := (16, 0)

/-- Point Q is where the line crosses the y-axis --/
def Q : ℝ × ℝ := (0, 8)

/-- Point T is on line segment PQ --/
def T_on_PQ (r s : ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ r = t * P.1 + (1 - t) * Q.1 ∧ s = t * P.2 + (1 - t) * Q.2

/-- Area of triangle POQ is twice the area of triangle TOP --/
def area_condition (r s : ℝ) : Prop :=
  abs ((P.1 - 0) * (Q.2 - 0) - (Q.1 - 0) * (P.2 - 0)) / 2 =
  2 * abs ((P.1 - 0) * (s - 0) - (r - 0) * (P.2 - 0)) / 2

theorem line_point_sum :
  ∀ r s : ℝ,
  line_equation r s →
  T_on_PQ r s →
  area_condition r s →
  r + s = 12 := by sorry

end NUMINAMATH_CALUDE_line_point_sum_l2134_213439


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l2134_213494

theorem quadratic_always_positive : ∀ x : ℝ, x^2 + x + 2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l2134_213494


namespace NUMINAMATH_CALUDE_cricketer_average_score_l2134_213489

/-- Represents a cricketer's scoring statistics -/
structure CricketerStats where
  innings : ℕ
  lastInningScore : ℕ
  averageIncrease : ℕ

/-- Calculates the new average score after the last inning -/
def newAverageScore (stats : CricketerStats) : ℕ :=
  sorry

theorem cricketer_average_score 
  (stats : CricketerStats) 
  (h1 : stats.innings = 19) 
  (h2 : stats.lastInningScore = 98) 
  (h3 : stats.averageIncrease = 4) : 
  newAverageScore stats = 26 := by
  sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l2134_213489


namespace NUMINAMATH_CALUDE_sin_graph_symmetry_l2134_213452

theorem sin_graph_symmetry (x : ℝ) :
  let f (x : ℝ) := Real.sin (2 * x)
  let g (x : ℝ) := f (x + π / 6)
  ∀ y : ℝ, g (π / 6 - x) = g (π / 6 + x) := by
sorry

end NUMINAMATH_CALUDE_sin_graph_symmetry_l2134_213452


namespace NUMINAMATH_CALUDE_system_solution_l2134_213445

theorem system_solution (x y z : ℝ) : 
  x * y = 8 - x - 4 * y →
  y * z = 12 - 3 * y - 6 * z →
  x * z = 40 - 5 * x - 2 * z →
  x > 0 →
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2134_213445


namespace NUMINAMATH_CALUDE_red_light_estimation_l2134_213425

theorem red_light_estimation (total_surveyed : ℕ) (yes_answers : ℕ) :
  total_surveyed = 600 →
  yes_answers = 180 →
  let prob_odd_id := (1 : ℚ) / 2
  let prob_yes := (yes_answers : ℚ) / total_surveyed
  let prob_red_light := 2 * prob_yes - prob_odd_id
  ⌊total_surveyed * prob_red_light⌋ = 60 := by
  sorry

end NUMINAMATH_CALUDE_red_light_estimation_l2134_213425


namespace NUMINAMATH_CALUDE_domain_all_reals_l2134_213415

theorem domain_all_reals (k : ℝ) :
  (∀ x : ℝ, (-7 * x^2 - 4 * x + k ≠ 0)) ↔ k < -4/7 := by sorry

end NUMINAMATH_CALUDE_domain_all_reals_l2134_213415


namespace NUMINAMATH_CALUDE_range_of_m_when_p_or_q_false_l2134_213414

theorem range_of_m_when_p_or_q_false (m : ℝ) : 
  (¬(∃ x : ℝ, m * x^2 + 1 ≤ 0) ∧ ¬(∀ x : ℝ, x^2 + m * x + 1 > 0)) → m ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_when_p_or_q_false_l2134_213414


namespace NUMINAMATH_CALUDE_power_of_two_with_ones_and_twos_l2134_213409

theorem power_of_two_with_ones_and_twos (N : ℕ) : 
  ∃ k : ℕ, ∃ m : ℕ, 2^k ≡ m [MOD 10^N] ∧ 
  (∀ d : ℕ, d < N → (m / 10^d % 10 = 1 ∨ m / 10^d % 10 = 2)) :=
sorry

end NUMINAMATH_CALUDE_power_of_two_with_ones_and_twos_l2134_213409


namespace NUMINAMATH_CALUDE_opposite_of_seven_l2134_213492

theorem opposite_of_seven :
  ∀ x : ℤ, (7 + x = 0) → x = -7 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_seven_l2134_213492


namespace NUMINAMATH_CALUDE_range_of_a_l2134_213442

theorem range_of_a (a : ℝ) : 
  (∀ x, |x - 1| < 1 → x ≥ a) ∧ 
  (∃ x, x ≥ a ∧ ¬(|x - 1| < 1)) →
  a ≤ 2 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2134_213442


namespace NUMINAMATH_CALUDE_loan_amount_l2134_213433

/-- Proves that the amount lent is 2000 rupees given the specified conditions -/
theorem loan_amount (P : ℚ) 
  (h1 : P * (17/100 * 4 - 15/100 * 4) = 160) : P = 2000 := by
  sorry

#check loan_amount

end NUMINAMATH_CALUDE_loan_amount_l2134_213433


namespace NUMINAMATH_CALUDE_area_between_curves_l2134_213448

/-- The upper function in the integral -/
def f (x : ℝ) : ℝ := 2 * x - x^2 + 3

/-- The lower function in the integral -/
def g (x : ℝ) : ℝ := x^2 - 4 * x + 3

/-- The theorem stating that the area between the curves is 9 -/
theorem area_between_curves : ∫ x in (0)..(3), (f x - g x) = 9 := by
  sorry

end NUMINAMATH_CALUDE_area_between_curves_l2134_213448


namespace NUMINAMATH_CALUDE_count_grid_paths_l2134_213422

/-- The number of paths from (0,0) to (m,n) in a grid with only right and up steps -/
def grid_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

/-- Theorem: The number of paths from (0,0) to (m,n) is (m+n choose m) -/
theorem count_grid_paths (m n : ℕ) :
  grid_paths m n = Nat.choose (m + n) m := by sorry

end NUMINAMATH_CALUDE_count_grid_paths_l2134_213422


namespace NUMINAMATH_CALUDE_minimum_formulas_to_memorize_l2134_213420

theorem minimum_formulas_to_memorize (total_formulas : ℕ) (min_score_percent : ℚ) : 
  total_formulas = 300 ∧ min_score_percent = 90 / 100 →
  ∃ (min_formulas : ℕ), 
    (min_formulas : ℚ) / total_formulas ≥ min_score_percent ∧
    ∀ (x : ℕ), (x : ℚ) / total_formulas ≥ min_score_percent → x ≥ min_formulas ∧
    min_formulas = 270 :=
by sorry

end NUMINAMATH_CALUDE_minimum_formulas_to_memorize_l2134_213420


namespace NUMINAMATH_CALUDE_bus_car_speed_problem_l2134_213428

theorem bus_car_speed_problem : ∀ (v_bus v_car : ℝ),
  -- Given conditions
  (1.5 * v_bus + 1.5 * v_car = 180) →
  (2.5 * v_bus + v_car = 180) →
  -- Conclusion
  (v_bus = 40 ∧ v_car = 80) :=
by
  sorry

end NUMINAMATH_CALUDE_bus_car_speed_problem_l2134_213428


namespace NUMINAMATH_CALUDE_club_membership_l2134_213499

theorem club_membership (total : Nat) (left_handed : Nat) (jazz_lovers : Nat) (right_handed_jazz_dislikers : Nat) :
  total = 25 →
  left_handed = 12 →
  jazz_lovers = 18 →
  right_handed_jazz_dislikers = 3 →
  (∃ (left_handed_jazz_lovers : Nat),
    left_handed_jazz_lovers +
    (left_handed - left_handed_jazz_lovers) +
    (jazz_lovers - left_handed_jazz_lovers) +
    right_handed_jazz_dislikers = total ∧
    left_handed_jazz_lovers = 8) := by
  sorry

#check club_membership

end NUMINAMATH_CALUDE_club_membership_l2134_213499


namespace NUMINAMATH_CALUDE_arithmetic_sequence_14th_term_l2134_213491

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_14th_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_2 : a 2 = 5)
  (h_6 : a 6 = 17) :
  a 14 = 41 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_14th_term_l2134_213491


namespace NUMINAMATH_CALUDE_ceiling_negative_sqrt_64_over_9_l2134_213404

theorem ceiling_negative_sqrt_64_over_9 : ⌈-Real.sqrt (64/9)⌉ = -2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_negative_sqrt_64_over_9_l2134_213404


namespace NUMINAMATH_CALUDE_lines_coplanar_iff_m_eq_zero_l2134_213486

/-- Definition of the first line -/
def line1 (m s : ℝ) : ℝ × ℝ × ℝ := (1 + 2*s, 2 + 2*s, 3 - m*s)

/-- Definition of the second line -/
def line2 (m v : ℝ) : ℝ × ℝ × ℝ := (m*v, 5 + 3*v, 6 + 2*v)

/-- Two vectors are coplanar if their cross product is zero -/
def coplanar (u v w : ℝ × ℝ × ℝ) : Prop :=
  let (u₁, u₂, u₃) := u
  let (v₁, v₂, v₃) := v
  let (w₁, w₂, w₃) := w
  (v₁ - u₁) * (w₂ - u₂) * (u₃ - u₃) +
  (v₂ - u₂) * (w₃ - u₃) * (u₁ - u₁) +
  (v₃ - u₃) * (w₁ - u₁) * (u₂ - u₂) -
  (v₃ - u₃) * (w₂ - u₂) * (u₁ - u₁) -
  (v₁ - u₁) * (w₃ - u₃) * (u₂ - u₂) -
  (v₂ - u₂) * (w₁ - u₁) * (u₃ - u₃) = 0

/-- Theorem: The lines are coplanar if and only if m = 0 -/
theorem lines_coplanar_iff_m_eq_zero :
  ∀ s v : ℝ, coplanar (1, 2, 3) (line1 m s) (line2 m v) ↔ m = 0 :=
sorry

end NUMINAMATH_CALUDE_lines_coplanar_iff_m_eq_zero_l2134_213486


namespace NUMINAMATH_CALUDE_family_members_count_l2134_213449

def num_birds : ℕ := 4
def num_dogs : ℕ := 3
def num_cats : ℕ := 18

def bird_feet : ℕ := 2
def dog_feet : ℕ := 4
def cat_feet : ℕ := 4

def animal_heads : ℕ := num_birds + num_dogs + num_cats

def animal_feet : ℕ := num_birds * bird_feet + num_dogs * dog_feet + num_cats * cat_feet

def human_feet : ℕ := 2
def human_head : ℕ := 1

theorem family_members_count :
  ∃ (F : ℕ), animal_feet + F * human_feet = animal_heads + F * human_head + 74 ∧ F = 7 := by
  sorry

end NUMINAMATH_CALUDE_family_members_count_l2134_213449


namespace NUMINAMATH_CALUDE_parabola_translation_l2134_213475

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2 - 1

-- Define the translation
def left_translation : ℝ := 2
def up_translation : ℝ := 1

-- Define the translated parabola
def translated_parabola (x : ℝ) : ℝ := (x + left_translation)^2

-- Theorem statement
theorem parabola_translation :
  ∀ x y : ℝ, y = original_parabola (x + left_translation) + up_translation 
  ↔ y = translated_parabola x := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l2134_213475


namespace NUMINAMATH_CALUDE_bottles_used_first_game_l2134_213406

theorem bottles_used_first_game 
  (total_bottles : ℕ)
  (bottles_left : ℕ)
  (bottles_used_second : ℕ)
  (h1 : total_bottles = 200)
  (h2 : bottles_left = 20)
  (h3 : bottles_used_second = 110) :
  total_bottles - bottles_left - bottles_used_second = 70 :=
by sorry

end NUMINAMATH_CALUDE_bottles_used_first_game_l2134_213406


namespace NUMINAMATH_CALUDE_product_inequality_l2134_213450

theorem product_inequality (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  (a + b) * (a + c) * (b + c) ≥ 8 * a * b * c :=
by sorry

end NUMINAMATH_CALUDE_product_inequality_l2134_213450


namespace NUMINAMATH_CALUDE_negative_to_even_power_l2134_213453

theorem negative_to_even_power (a : ℝ) : (-a)^4 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_negative_to_even_power_l2134_213453


namespace NUMINAMATH_CALUDE_product_of_roots_zero_l2134_213401

theorem product_of_roots_zero (x₁ x₂ : ℝ) : 
  ((-x₁^2 + 3*x₁ = 0) ∧ (-x₂^2 + 3*x₂ = 0)) → x₁ * x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_zero_l2134_213401


namespace NUMINAMATH_CALUDE_smallest_c_inequality_l2134_213456

theorem smallest_c_inequality (c : ℝ) : 
  (∀ x y : ℝ, x ≥ 0 ∧ y ≥ 0 → Real.sqrt (x * y) + c * Real.sqrt (|x - y|) ≥ (x + y) / 2) ↔ 
  c ≥ (1 / 2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_smallest_c_inequality_l2134_213456


namespace NUMINAMATH_CALUDE_discount_rate_sum_l2134_213476

-- Define the normal prices and quantities
def biography_price : ℝ := 20
def mystery_price : ℝ := 12
def biography_quantity : ℕ := 5
def mystery_quantity : ℕ := 3

-- Define the total savings and mystery discount rate
def total_savings : ℝ := 19
def mystery_discount_rate : ℝ := 0.375

-- Define the function to calculate the total discount rate
def total_discount_rate (biography_discount_rate : ℝ) : ℝ :=
  biography_discount_rate + mystery_discount_rate

-- Theorem statement
theorem discount_rate_sum :
  ∃ (biography_discount_rate : ℝ),
    biography_discount_rate > 0 ∧
    biography_discount_rate < 1 ∧
    (biography_price * biography_quantity * (1 - biography_discount_rate) +
     mystery_price * mystery_quantity * (1 - mystery_discount_rate) =
     biography_price * biography_quantity + mystery_price * mystery_quantity - total_savings) ∧
    total_discount_rate biography_discount_rate = 0.43 :=
by sorry

end NUMINAMATH_CALUDE_discount_rate_sum_l2134_213476


namespace NUMINAMATH_CALUDE_numbering_system_base_l2134_213481

theorem numbering_system_base : ∃! (n : ℕ), n > 0 ∧ n^2 = 5*n + 6 := by sorry

end NUMINAMATH_CALUDE_numbering_system_base_l2134_213481


namespace NUMINAMATH_CALUDE_triangle_trigonometry_l2134_213410

theorem triangle_trigonometry (A B C : Real) (AC BC : Real) (h1 : AC = 2) (h2 : BC = 3) (h3 : Real.cos A = -4/5) :
  Real.sin B = 2/5 ∧ Real.sin (2*B + π/6) = (12*Real.sqrt 7 + 17) / 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_trigonometry_l2134_213410


namespace NUMINAMATH_CALUDE_shopping_tax_theorem_l2134_213447

/-- Calculates the total tax percentage given spending percentages and tax rates -/
def total_tax_percentage (clothing_percent : ℝ) (food_percent : ℝ) (other_percent : ℝ)
                         (clothing_tax : ℝ) (food_tax : ℝ) (other_tax : ℝ) : ℝ :=
  clothing_percent * clothing_tax + food_percent * food_tax + other_percent * other_tax

/-- Theorem stating that the total tax percentage is 5.2% given the specific conditions -/
theorem shopping_tax_theorem :
  total_tax_percentage 0.5 0.1 0.4 0.04 0 0.08 = 0.052 := by
  sorry

#eval total_tax_percentage 0.5 0.1 0.4 0.04 0 0.08

end NUMINAMATH_CALUDE_shopping_tax_theorem_l2134_213447


namespace NUMINAMATH_CALUDE_all_solutions_are_valid_l2134_213446

/-- A quadruple of real numbers satisfying the given conditions -/
structure Quadruple where
  x : ℝ
  y : ℝ
  z : ℝ
  w : ℝ
  sum_zero : x + y + z + w = 0
  sum_seventh_power_zero : x^7 + y^7 + z^7 + w^7 = 0

/-- Definition of a valid solution -/
def is_valid_solution (q : Quadruple) : Prop :=
  (q.x = 0 ∧ q.y = 0 ∧ q.z = 0 ∧ q.w = 0) ∨
  (q.x = -q.y ∧ q.z = -q.w) ∨
  (q.x = -q.z ∧ q.y = -q.w) ∨
  (q.x = -q.w ∧ q.y = -q.z)

/-- Main theorem: All solutions are valid -/
theorem all_solutions_are_valid (q : Quadruple) : is_valid_solution q := by
  sorry

end NUMINAMATH_CALUDE_all_solutions_are_valid_l2134_213446


namespace NUMINAMATH_CALUDE_track_length_l2134_213461

/-- The length of a circular track given specific meeting conditions of two runners -/
theorem track_length : ∀ (x : ℝ), 
  (∃ (v_brenda v_sally : ℝ), v_brenda > 0 ∧ v_sally > 0 ∧
    -- First meeting condition
    80 / v_brenda = (x/2 - 80) / v_sally ∧
    -- Second meeting condition
    (x/2 - 100) / v_brenda = (x/2 + 100) / v_sally) →
  x = 520 :=
by sorry

end NUMINAMATH_CALUDE_track_length_l2134_213461


namespace NUMINAMATH_CALUDE_pear_difference_is_five_l2134_213427

/-- The number of bags of pears Austin picked fewer than Dallas -/
def pear_difference (dallas_apples dallas_pears austin_total : ℕ) (austin_apple_diff : ℕ) : ℕ :=
  dallas_pears - (austin_total - (dallas_apples + austin_apple_diff))

theorem pear_difference_is_five :
  pear_difference 14 9 24 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_pear_difference_is_five_l2134_213427


namespace NUMINAMATH_CALUDE_three_parallel_lines_theorem_l2134_213457

-- Define a type for lines in 3D space
structure Line3D where
  -- Add necessary properties for a line in 3D space
  -- This is a placeholder and may need to be adjusted based on Lean's standard library

-- Define a type for planes in 3D space
structure Plane3D where
  -- Add necessary properties for a plane in 3D space
  -- This is a placeholder and may need to be adjusted based on Lean's standard library

-- Define a function to check if two lines are parallel
def are_parallel (l1 l2 : Line3D) : Prop :=
  sorry -- Definition of parallel lines

-- Define a function to check if three lines are coplanar
def are_coplanar (l1 l2 l3 : Line3D) : Prop :=
  sorry -- Definition of coplanar lines

-- Define a function to count the number of planes determined by three lines
def count_planes (l1 l2 l3 : Line3D) : Nat :=
  sorry -- Count the number of planes

-- Define a function to count the number of parts space is divided into by these planes
def count_space_parts (planes : List Plane3D) : Nat :=
  sorry -- Count the number of parts

-- Theorem statement
theorem three_parallel_lines_theorem (a b c : Line3D) 
  (h_parallel_ab : are_parallel a b)
  (h_parallel_bc : are_parallel b c)
  (h_parallel_ac : are_parallel a c)
  (h_not_coplanar : ¬ are_coplanar a b c) :
  (count_planes a b c = 3) ∧ 
  (count_space_parts (sorry : List Plane3D) = 7) := by
  sorry


end NUMINAMATH_CALUDE_three_parallel_lines_theorem_l2134_213457


namespace NUMINAMATH_CALUDE_bicycle_race_fraction_l2134_213412

theorem bicycle_race_fraction (total_racers : ℕ) (total_wheels : ℕ) 
  (bicycle_wheels : ℕ) (tricycle_wheels : ℕ) :
  total_racers = 40 →
  total_wheels = 96 →
  bicycle_wheels = 2 →
  tricycle_wheels = 3 →
  ∃ (bicycles : ℕ) (tricycles : ℕ),
    bicycles + tricycles = total_racers ∧
    bicycles * bicycle_wheels + tricycles * tricycle_wheels = total_wheels ∧
    (bicycles : ℚ) / total_racers = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_bicycle_race_fraction_l2134_213412


namespace NUMINAMATH_CALUDE_garden_problem_l2134_213483

/-- Represents the gardening problem with eggplants and sunflowers. -/
theorem garden_problem (eggplants_per_packet : ℕ) (sunflowers_per_packet : ℕ) 
  (eggplant_packets : ℕ) (total_plants : ℕ) :
  eggplants_per_packet = 14 →
  sunflowers_per_packet = 10 →
  eggplant_packets = 4 →
  total_plants = 116 →
  ∃ sunflower_packets : ℕ, 
    sunflower_packets = 6 ∧ 
    total_plants = eggplants_per_packet * eggplant_packets + sunflowers_per_packet * sunflower_packets :=
by sorry

end NUMINAMATH_CALUDE_garden_problem_l2134_213483


namespace NUMINAMATH_CALUDE_log_comparison_l2134_213467

theorem log_comparison : Real.log 80 / Real.log 20 < Real.log 640 / Real.log 80 := by
  sorry

end NUMINAMATH_CALUDE_log_comparison_l2134_213467


namespace NUMINAMATH_CALUDE_sum_of_21st_group_l2134_213482

/-- The first number in the n-th group -/
def first_number (n : ℕ) : ℕ := n * (n - 1) / 2 + 1

/-- The last number in the n-th group -/
def last_number (n : ℕ) : ℕ := first_number n + (n - 1)

/-- The sum of numbers in the n-th group -/
def group_sum (n : ℕ) : ℕ := n * (first_number n + last_number n) / 2

/-- Theorem: The sum of numbers in the 21st group is 4641 -/
theorem sum_of_21st_group : group_sum 21 = 4641 := by sorry

end NUMINAMATH_CALUDE_sum_of_21st_group_l2134_213482


namespace NUMINAMATH_CALUDE_min_value_expression_l2134_213485

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^2 + y^2 + 16 / (x + y)^2 ≥ 8 ∧
  (x^2 + y^2 + 16 / (x + y)^2 = 8 ↔ x = y ∧ x = 2^(1/4)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2134_213485


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l2134_213478

/-- Given a geometric sequence where the fifth term is 45 and the sixth term is 60,
    prove that the first term is 1215/256. -/
theorem geometric_sequence_first_term
  (a : ℚ)  -- First term of the sequence
  (r : ℚ)  -- Common ratio of the sequence
  (h1 : a * r^4 = 45)  -- Fifth term is 45
  (h2 : a * r^5 = 60)  -- Sixth term is 60
  : a = 1215 / 256 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l2134_213478


namespace NUMINAMATH_CALUDE_state_selection_difference_l2134_213463

theorem state_selection_difference (total_candidates : ℕ) 
  (selection_rate_A selection_rate_B : ℚ) : 
  total_candidates = 8000 →
  selection_rate_A = 6 / 100 →
  selection_rate_B = 7 / 100 →
  (selection_rate_B * total_candidates : ℚ) - (selection_rate_A * total_candidates : ℚ) = 80 := by
  sorry

end NUMINAMATH_CALUDE_state_selection_difference_l2134_213463


namespace NUMINAMATH_CALUDE_unique_solution_of_equation_l2134_213423

theorem unique_solution_of_equation :
  ∃! x : ℝ, x ≠ 3 ∧ x + 36 / (x - 3) = -9 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_of_equation_l2134_213423


namespace NUMINAMATH_CALUDE_isosceles_triangles_105_similar_l2134_213462

-- Define an isosceles triangle with a specific angle
structure IsoscelesTriangle :=
  (base_angle : ℝ)
  (vertex_angle : ℝ)
  (is_isosceles : base_angle * 2 + vertex_angle = 180)

-- Define similarity for isosceles triangles
def are_similar (t1 t2 : IsoscelesTriangle) : Prop :=
  t1.base_angle = t2.base_angle ∧ t1.vertex_angle = t2.vertex_angle

-- Theorem statement
theorem isosceles_triangles_105_similar :
  ∀ (t1 t2 : IsoscelesTriangle),
  t1.vertex_angle = 105 → t2.vertex_angle = 105 →
  are_similar t1 t2 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangles_105_similar_l2134_213462


namespace NUMINAMATH_CALUDE_collinear_points_sum_l2134_213416

/-- Three points in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Definition of collinearity for three points -/
def collinear (p q r : Point3D) : Prop :=
  ∃ t s : ℝ, q.x - p.x = t * (r.x - p.x) ∧
             q.y - p.y = t * (r.y - p.y) ∧
             q.z - p.z = t * (r.z - p.z) ∧
             q.x - p.x = s * (r.x - q.x) ∧
             q.y - p.y = s * (r.y - q.y) ∧
             q.z - p.z = s * (r.z - q.z)

theorem collinear_points_sum (m n : ℝ) :
  let M : Point3D := ⟨1, 0, 1⟩
  let N : Point3D := ⟨2, m, 3⟩
  let P : Point3D := ⟨2, 2, n + 1⟩
  collinear M N P → m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l2134_213416


namespace NUMINAMATH_CALUDE_logarithm_simplification_l2134_213471

theorem logarithm_simplification (a b c d x y : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (hx : x > 0) (hy : y > 0) :
  Real.log (a^2 / b^3) + Real.log (b^2 / c) + Real.log (c^3 / d^2) - Real.log (a^2 * y^2 / (d^3 * x)) 
  = Real.log (c^2 * d * x / y^2) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_simplification_l2134_213471


namespace NUMINAMATH_CALUDE_special_function_at_1001_l2134_213477

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f x > 0) ∧
  (∀ x y, x > y ∧ y > 0 → f (x - y) = Real.sqrt (f (x * y) + 3))

/-- The main theorem stating that f(1001) = 3 for any function satisfying the conditions -/
theorem special_function_at_1001 (f : ℝ → ℝ) (h : special_function f) : f 1001 = 3 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_1001_l2134_213477


namespace NUMINAMATH_CALUDE_max_distance_P_to_D_l2134_213454

/-- A square with side length 1 in a 2D plane -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (side_length : ℝ)
  (is_square : side_length = 1)

/-- A point P in the same plane as the square -/
def P : ℝ × ℝ := sorry

/-- Distance between two points in 2D plane -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Theorem stating the maximum distance between P and D -/
theorem max_distance_P_to_D (square : Square) 
  (h1 : distance P square.A = u)
  (h2 : distance P square.B = v)
  (h3 : distance P square.C = w)
  (h4 : u^2 + w^2 = v^2) : 
  ∃ (max_dist : ℝ), max_dist = Real.sqrt 2 ∧ 
    ∀ (P' : ℝ × ℝ), 
      distance P' square.A = u → 
      distance P' square.B = v → 
      distance P' square.C = w → 
      u^2 + w^2 = v^2 → 
      distance P' square.D ≤ max_dist :=
sorry

end NUMINAMATH_CALUDE_max_distance_P_to_D_l2134_213454


namespace NUMINAMATH_CALUDE_eggs_division_l2134_213417

theorem eggs_division (total_eggs : ℕ) (num_groups : ℕ) (eggs_per_group : ℕ) :
  total_eggs = 15 →
  num_groups = 3 →
  eggs_per_group * num_groups = total_eggs →
  eggs_per_group = 5 := by
  sorry

end NUMINAMATH_CALUDE_eggs_division_l2134_213417


namespace NUMINAMATH_CALUDE_expression_value_l2134_213496

theorem expression_value (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (3 * x^4 + 4 * y^2) / 12 = 25.5833 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2134_213496


namespace NUMINAMATH_CALUDE_power_function_above_identity_l2134_213469

theorem power_function_above_identity {x α : ℝ} (hx : x ∈ Set.Ioo 0 1) :
  x^α > x ↔ α ∈ Set.Iio 1 := by sorry

end NUMINAMATH_CALUDE_power_function_above_identity_l2134_213469


namespace NUMINAMATH_CALUDE_solution_in_quadrant_I_l2134_213470

theorem solution_in_quadrant_I (c : ℝ) :
  (∃ x y : ℝ, x - y = 4 ∧ c * x + y = 7 ∧ x > 0 ∧ y > 0) ↔ -1 < c ∧ c < 7/4 :=
by sorry

end NUMINAMATH_CALUDE_solution_in_quadrant_I_l2134_213470


namespace NUMINAMATH_CALUDE_compare_log_and_sqrt_l2134_213411

theorem compare_log_and_sqrt : 2 + Real.log 6 / Real.log 2 > 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_compare_log_and_sqrt_l2134_213411


namespace NUMINAMATH_CALUDE_evaluate_M_l2134_213464

theorem evaluate_M : 
  let M := (Real.sqrt (Real.sqrt 7 + 3) + Real.sqrt (Real.sqrt 7 - 3)) / Real.sqrt (Real.sqrt 7 + 2) + Real.sqrt (4 - 2 * Real.sqrt 3)
  M = 7/4 := by
sorry

end NUMINAMATH_CALUDE_evaluate_M_l2134_213464


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_range_l2134_213407

theorem quadratic_always_nonnegative_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*a*x + 1 ≥ 0) → -1 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_range_l2134_213407


namespace NUMINAMATH_CALUDE_positive_Y_value_l2134_213424

-- Define the ∆ relation
def triangle (X Y : ℝ) : ℝ := X^2 + 3*Y^2

-- Theorem statement
theorem positive_Y_value :
  ∃ Y : ℝ, Y > 0 ∧ triangle 9 Y = 360 ∧ Y = Real.sqrt 93 := by
  sorry

end NUMINAMATH_CALUDE_positive_Y_value_l2134_213424


namespace NUMINAMATH_CALUDE_special_function_inequality_l2134_213495

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  diff : Differentiable ℝ f
  f_prime_1 : deriv f 1 = 0
  condition : ∀ x : ℝ, x ≠ 1 → (x - 1) * (deriv f x) > 0

/-- Theorem stating that for any function satisfying the SpecialFunction conditions,
    f(0) + f(2) > 2f(1) -/
theorem special_function_inequality (sf : SpecialFunction) : sf.f 0 + sf.f 2 > 2 * sf.f 1 := by
  sorry

end NUMINAMATH_CALUDE_special_function_inequality_l2134_213495


namespace NUMINAMATH_CALUDE_platform_pillar_height_l2134_213473

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a rectangular platform with pillars -/
structure Platform where
  P : Point3D
  Q : Point3D
  R : Point3D
  S : Point3D
  slopeAngle : ℝ
  pillarHeightP : ℝ
  pillarHeightQ : ℝ
  pillarHeightR : ℝ

/-- The height of the pillar at point S -/
def pillarHeightS (p : Platform) : ℝ :=
  sorry

theorem platform_pillar_height
  (p : Platform)
  (h_PQ : p.Q.x - p.P.x = 10)
  (h_PR : p.R.y - p.P.y = 15)
  (h_slope : p.slopeAngle = π / 6)
  (h_heightP : p.pillarHeightP = 7)
  (h_heightQ : p.pillarHeightQ = 10)
  (h_heightR : p.pillarHeightR = 12) :
  pillarHeightS p = 7.5 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_platform_pillar_height_l2134_213473


namespace NUMINAMATH_CALUDE_correct_initial_chips_l2134_213418

/-- The number of chips Marnie ate initially to see if she likes them -/
def initial_chips : ℕ := 5

/-- The total number of chips in the bag -/
def total_chips : ℕ := 100

/-- The number of chips Marnie eats per day starting from the second day -/
def daily_chips : ℕ := 10

/-- The total number of days it takes Marnie to finish the bag -/
def total_days : ℕ := 10

/-- Theorem stating that the initial number of chips Marnie ate is correct -/
theorem correct_initial_chips :
  2 * initial_chips + (total_days - 1) * daily_chips = total_chips :=
by sorry

end NUMINAMATH_CALUDE_correct_initial_chips_l2134_213418


namespace NUMINAMATH_CALUDE_zoo_bus_seats_l2134_213498

theorem zoo_bus_seats (total_children : ℕ) (children_per_seat : ℕ) (seats_needed : ℕ) : 
  total_children = 58 → children_per_seat = 2 → seats_needed = total_children / children_per_seat → 
  seats_needed = 29 := by
sorry

end NUMINAMATH_CALUDE_zoo_bus_seats_l2134_213498


namespace NUMINAMATH_CALUDE_cereal_box_cups_l2134_213479

/-- Calculates the total number of cups in a cereal box -/
def total_cups (servings : ℕ) (cups_per_serving : ℕ) : ℕ :=
  servings * cups_per_serving

/-- Theorem: A cereal box with 9 servings and 2 cups per serving contains 18 cups of cereal -/
theorem cereal_box_cups : total_cups 9 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_cereal_box_cups_l2134_213479


namespace NUMINAMATH_CALUDE_car_speed_problem_l2134_213431

/-- Given a car traveling for 2 hours with a speed of 40 km/h in the second hour
    and an average speed of 65 km/h, prove that the speed in the first hour is 90 km/h. -/
theorem car_speed_problem (first_hour_speed second_hour_speed average_speed : ℝ) :
  second_hour_speed = 40 →
  average_speed = 65 →
  (first_hour_speed + second_hour_speed) / 2 = average_speed →
  first_hour_speed = 90 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2134_213431


namespace NUMINAMATH_CALUDE_tangent_parallel_at_minus_one_minus_four_l2134_213413

-- Define the function
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem tangent_parallel_at_minus_one_minus_four :
  let P₀ : ℝ × ℝ := (-1, -4)
  (f' (P₀.1) = 4) ∧ (f P₀.1 = P₀.2) := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_at_minus_one_minus_four_l2134_213413


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2134_213440

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The main theorem about the sum of a_3 and a_5 in a specific geometric sequence. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  IsGeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2134_213440


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l2134_213455

/-- Arithmetic sequence with given first term, last term, and common difference has the specified number of terms -/
theorem arithmetic_sequence_length
  (a₁ : ℤ)    -- First term
  (aₙ : ℤ)    -- Last term
  (d : ℤ)     -- Common difference
  (n : ℕ)     -- Number of terms
  (h1 : a₁ = -4)
  (h2 : aₙ = 32)
  (h3 : d = 3)
  (h4 : aₙ = a₁ + (n - 1) * d)  -- Formula for the nth term of an arithmetic sequence
  : n = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l2134_213455


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2134_213488

theorem inequality_and_equality_condition (n : ℕ+) (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) : 
  (1 + a / b)^(n : ℕ) + (1 + b / a)^(n : ℕ) ≥ 2^((n : ℕ) + 1) ∧ 
  ((1 + a / b)^(n : ℕ) + (1 + b / a)^(n : ℕ) = 2^((n : ℕ) + 1) ↔ a = b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2134_213488


namespace NUMINAMATH_CALUDE_swimmer_speed_is_5_l2134_213430

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmingScenario where
  swimmer_speed : ℝ
  stream_speed : ℝ

/-- Calculates the effective speed when swimming downstream. -/
def downstream_speed (s : SwimmingScenario) : ℝ :=
  s.swimmer_speed + s.stream_speed

/-- Calculates the effective speed when swimming upstream. -/
def upstream_speed (s : SwimmingScenario) : ℝ :=
  s.swimmer_speed - s.stream_speed

/-- Theorem stating that given the conditions, the swimmer's speed in still water is 5 km/h. -/
theorem swimmer_speed_is_5 (s : SwimmingScenario) 
    (h_downstream : downstream_speed s * 6 = 54)
    (h_upstream : upstream_speed s * 6 = 6) : 
    s.swimmer_speed = 5 := by
  sorry


end NUMINAMATH_CALUDE_swimmer_speed_is_5_l2134_213430


namespace NUMINAMATH_CALUDE_clock_overlaps_in_24_hours_l2134_213436

/-- Represents a clock with an hour hand and a minute hand -/
structure Clock :=
  (hour_revolutions : ℕ)
  (minute_revolutions : ℕ)

/-- The number of overlaps between the hour and minute hands -/
def overlaps (c : Clock) : ℕ := c.minute_revolutions - c.hour_revolutions

theorem clock_overlaps_in_24_hours :
  ∃ (c : Clock), c.hour_revolutions = 2 ∧ c.minute_revolutions = 24 ∧ overlaps c = 22 :=
sorry

end NUMINAMATH_CALUDE_clock_overlaps_in_24_hours_l2134_213436


namespace NUMINAMATH_CALUDE_divisibility_of_p_l2134_213419

theorem divisibility_of_p (p q r s : ℕ+) 
  (h1 : Nat.gcd p q = 40)
  (h2 : Nat.gcd q r = 45)
  (h3 : Nat.gcd r s = 60)
  (h4 : 110 < Nat.gcd s p ∧ Nat.gcd s p < 150) :
  11 ∣ p := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_p_l2134_213419


namespace NUMINAMATH_CALUDE_discriminant_nonnegative_root_greater_than_three_implies_a_greater_than_four_l2134_213459

-- Define the quadratic equation
def quadratic (a x : ℝ) : ℝ := x^2 - a*x + a - 1

-- Define the discriminant of the quadratic equation
def discriminant (a : ℝ) : ℝ := a^2 - 4*(a - 1)

-- Theorem 1: The discriminant is always non-negative
theorem discriminant_nonnegative (a : ℝ) : discriminant a ≥ 0 := by
  sorry

-- Theorem 2: When one root is greater than 3, a > 4
theorem root_greater_than_three_implies_a_greater_than_four (a : ℝ) :
  (∃ x, quadratic a x = 0 ∧ x > 3) → a > 4 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_nonnegative_root_greater_than_three_implies_a_greater_than_four_l2134_213459


namespace NUMINAMATH_CALUDE_fundraiser_final_day_amount_l2134_213493

def fundraiser (goal : ℕ) (bronze_donation silver_donation gold_donation : ℕ) 
  (bronze_families silver_families gold_families : ℕ) : ℕ :=
  let total_raised := bronze_donation * bronze_families + 
                      silver_donation * silver_families + 
                      gold_donation * gold_families
  goal - total_raised

theorem fundraiser_final_day_amount : 
  fundraiser 750 25 50 100 10 7 1 = 50 := by
  sorry

end NUMINAMATH_CALUDE_fundraiser_final_day_amount_l2134_213493


namespace NUMINAMATH_CALUDE_shooter_probabilities_l2134_213484

def hit_probability : ℚ := 4/5

def exactly_eight_hits (n k : ℕ) (p : ℚ) : ℚ :=
  Nat.choose n k * p^k * (1-p)^(n-k)

def at_least_eight_hits (n : ℕ) (p : ℚ) : ℚ :=
  exactly_eight_hits n 8 p + exactly_eight_hits n 9 p + p^n

theorem shooter_probabilities :
  (exactly_eight_hits 10 8 hit_probability = 
    Nat.choose 10 8 * (4/5)^8 * (1/5)^2) ∧
  (at_least_eight_hits 10 hit_probability = 
    Nat.choose 10 8 * (4/5)^8 * (1/5)^2 + 
    Nat.choose 10 9 * (4/5)^9 * (1/5) + 
    (4/5)^10) := by
  sorry

end NUMINAMATH_CALUDE_shooter_probabilities_l2134_213484


namespace NUMINAMATH_CALUDE_factorial_equality_l2134_213437

theorem factorial_equality : 6 * 8 * 3 * 280 = Nat.factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equality_l2134_213437


namespace NUMINAMATH_CALUDE_opposite_of_negative_eleven_l2134_213451

theorem opposite_of_negative_eleven : 
  ∀ x : ℤ, x + (-11) = 0 → x = 11 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_eleven_l2134_213451


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l2134_213402

theorem product_of_three_numbers (x y z n : ℤ) 
  (sum_eq : x + y + z = 200)
  (x_eq : 8 * x = n)
  (y_eq : y - 5 = n)
  (z_eq : z + 5 = n) :
  x * y * z = 372462 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l2134_213402


namespace NUMINAMATH_CALUDE_intersection_M_N_l2134_213490

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 3*x - 4 ≤ 0}
def N : Set ℝ := {x | Real.log x ≥ 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | 1 ≤ x ∧ x ≤ 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2134_213490


namespace NUMINAMATH_CALUDE_binomial_12_choose_10_l2134_213497

theorem binomial_12_choose_10 : Nat.choose 12 10 = 66 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_choose_10_l2134_213497


namespace NUMINAMATH_CALUDE_intersection_points_count_l2134_213421

/-- A line in the 2D plane --/
inductive Line
  | General (a b c : ℝ) : Line  -- ax + by + c = 0
  | Vertical (x : ℝ) : Line     -- x = k
  | Horizontal (y : ℝ) : Line   -- y = k

/-- Check if a point (x, y) is on a line --/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  match l with
  | General a b c => a * x + b * y + c = 0
  | Vertical k => x = k
  | Horizontal k => y = k

/-- The set of lines given in the problem --/
def problem_lines : List Line :=
  [Line.General 3 (-1) (-1), Line.General 1 2 (-5), Line.Vertical 3, Line.Horizontal 1]

/-- A point is an intersection point if it's contained in at least two distinct lines --/
def is_intersection_point (x y : ℝ) (lines : List Line) : Prop :=
  ∃ l1 l2 : Line, l1 ∈ lines ∧ l2 ∈ lines ∧ l1 ≠ l2 ∧ l1.contains x y ∧ l2.contains x y

/-- The theorem to be proved --/
theorem intersection_points_count :
  ∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧
    (∀ x y : ℝ, is_intersection_point x y problem_lines ↔ (x, y) = p1 ∨ (x, y) = p2) :=
  sorry

end NUMINAMATH_CALUDE_intersection_points_count_l2134_213421


namespace NUMINAMATH_CALUDE_jeans_price_proof_l2134_213405

/-- The original price of one pair of jeans -/
def original_price : ℝ := 40

/-- The discounted price for two pairs of jeans -/
def discounted_price (p : ℝ) : ℝ := 2 * p * 0.9

/-- The total price for three pairs of jeans -/
def total_price (p : ℝ) : ℝ := discounted_price p + p

theorem jeans_price_proof :
  total_price original_price = 112 :=
sorry

end NUMINAMATH_CALUDE_jeans_price_proof_l2134_213405


namespace NUMINAMATH_CALUDE_f_inequality_l2134_213444

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Theorem statement
theorem f_inequality (h1 : ∀ x, HasDerivAt f (f' x) x) (h2 : ∀ x, f' x < f x) : 
  f 3 < Real.exp 3 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l2134_213444


namespace NUMINAMATH_CALUDE_solve_equation_l2134_213429

theorem solve_equation (Z : ℝ) (h : (19 + 43 / Z) * Z = 2912) : Z = 151 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2134_213429


namespace NUMINAMATH_CALUDE_arccos_negative_sqrt3_over_2_l2134_213408

theorem arccos_negative_sqrt3_over_2 : Real.arccos (-(Real.sqrt 3 / 2)) = 5 * Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_arccos_negative_sqrt3_over_2_l2134_213408


namespace NUMINAMATH_CALUDE_toms_fruit_purchase_l2134_213432

/-- The problem of Tom's fruit purchase -/
theorem toms_fruit_purchase (apple_price : ℕ) (mango_price : ℕ) (apple_quantity : ℕ) (total_cost : ℕ) :
  apple_price = 70 →
  mango_price = 65 →
  apple_quantity = 8 →
  total_cost = 1145 →
  ∃ (mango_quantity : ℕ), 
    apple_price * apple_quantity + mango_price * mango_quantity = total_cost ∧ 
    mango_quantity = 9 := by
  sorry

#check toms_fruit_purchase

end NUMINAMATH_CALUDE_toms_fruit_purchase_l2134_213432


namespace NUMINAMATH_CALUDE_cos_sin_eq_sin_cos_third_l2134_213443

theorem cos_sin_eq_sin_cos_third (x : ℝ) :
  -π ≤ x ∧ x ≤ π ∧ Real.cos (Real.sin x) = Real.sin (Real.cos (x / 3)) → x = 0 :=
by sorry

end NUMINAMATH_CALUDE_cos_sin_eq_sin_cos_third_l2134_213443


namespace NUMINAMATH_CALUDE_absolute_value_and_exponentiation_calculation_l2134_213460

theorem absolute_value_and_exponentiation_calculation : 
  |1 - 3| * ((-12) - 2^3) = -40 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_and_exponentiation_calculation_l2134_213460


namespace NUMINAMATH_CALUDE_fraction_sqrt_cube_root_equals_power_l2134_213435

theorem fraction_sqrt_cube_root_equals_power (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a^3 * b)^(1/2) / (a * b)^(1/3) = a^(7/6) * b^(1/6) := by sorry

end NUMINAMATH_CALUDE_fraction_sqrt_cube_root_equals_power_l2134_213435


namespace NUMINAMATH_CALUDE_five_letter_words_count_l2134_213487

/-- The number of letters in the alphabet --/
def alphabet_size : ℕ := 26

/-- The length of the word --/
def word_length : ℕ := 5

/-- The number of positions that can vary --/
def variable_positions : ℕ := word_length - 2

theorem five_letter_words_count :
  (alphabet_size ^ variable_positions : ℕ) = 17576 := by
  sorry

end NUMINAMATH_CALUDE_five_letter_words_count_l2134_213487


namespace NUMINAMATH_CALUDE_repair_cost_calculation_l2134_213466

def purchase_price : ℕ := 9000
def transportation_charges : ℕ := 1000
def profit_percentage : ℚ := 50 / 100
def selling_price : ℕ := 22500

theorem repair_cost_calculation :
  ∃ (repair_cost : ℕ),
    (purchase_price + repair_cost + transportation_charges) * (1 + profit_percentage) = selling_price ∧
    repair_cost = 5000 := by
  sorry

end NUMINAMATH_CALUDE_repair_cost_calculation_l2134_213466


namespace NUMINAMATH_CALUDE_mothers_age_l2134_213438

/-- Proves that the mother's age this year is 39 years old -/
theorem mothers_age (sons_age : ℕ) (mothers_age : ℕ) : mothers_age = 39 :=
  by
  -- Define the son's current age
  have h1 : sons_age = 12 := by sorry
  
  -- Define the relationship between mother's and son's ages three years ago
  have h2 : mothers_age - 3 = 4 * (sons_age - 3) := by sorry
  
  -- Prove that the mother's age is 39
  sorry


end NUMINAMATH_CALUDE_mothers_age_l2134_213438


namespace NUMINAMATH_CALUDE_product_cost_l2134_213434

theorem product_cost (x y z : ℝ) 
  (h1 : 2*x + 3*y + z = 130)
  (h2 : 3*x + 5*y + z = 205) :
  x + y + z = 55 := by sorry

end NUMINAMATH_CALUDE_product_cost_l2134_213434


namespace NUMINAMATH_CALUDE_category_selection_probability_l2134_213474

def total_items : ℕ := 8
def swimming_items : ℕ := 1
def ball_games_items : ℕ := 3
def track_field_items : ℕ := 4
def items_to_select : ℕ := 4

theorem category_selection_probability :
  (Nat.choose swimming_items 1 * Nat.choose ball_games_items 1 * Nat.choose track_field_items 2 +
   Nat.choose swimming_items 1 * Nat.choose ball_games_items 2 * Nat.choose track_field_items 1) /
  Nat.choose total_items items_to_select = 3 / 7 := by sorry

end NUMINAMATH_CALUDE_category_selection_probability_l2134_213474


namespace NUMINAMATH_CALUDE_cubic_quadratic_inequality_l2134_213465

theorem cubic_quadratic_inequality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a < b) :
  a^3 * b^2 < a^2 * b^3 := by
sorry

end NUMINAMATH_CALUDE_cubic_quadratic_inequality_l2134_213465
