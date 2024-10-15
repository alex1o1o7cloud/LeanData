import Mathlib

namespace NUMINAMATH_CALUDE_gianna_savings_l2185_218528

/-- Calculates the total savings given a daily savings amount and number of days -/
def totalSavings (dailySavings : ℕ) (days : ℕ) : ℕ :=
  dailySavings * days

/-- Proves that saving $39 every day for 365 days results in $14,235 total savings -/
theorem gianna_savings : totalSavings 39 365 = 14235 := by
  sorry

end NUMINAMATH_CALUDE_gianna_savings_l2185_218528


namespace NUMINAMATH_CALUDE_smallest_class_size_seventeen_satisfies_conditions_smallest_class_size_is_seventeen_l2185_218524

theorem smallest_class_size (n : ℕ) : 
  (n % 4 = 1) ∧ (n % 5 = 2) ∧ (n % 7 = 3) → n ≥ 17 :=
by sorry

theorem seventeen_satisfies_conditions : 
  (17 % 4 = 1) ∧ (17 % 5 = 2) ∧ (17 % 7 = 3) :=
by sorry

theorem smallest_class_size_is_seventeen : 
  ∃ (n : ℕ), n = 17 ∧ (n % 4 = 1) ∧ (n % 5 = 2) ∧ (n % 7 = 3) ∧ 
  (∀ m : ℕ, (m % 4 = 1) ∧ (m % 5 = 2) ∧ (m % 7 = 3) → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_class_size_seventeen_satisfies_conditions_smallest_class_size_is_seventeen_l2185_218524


namespace NUMINAMATH_CALUDE_isabella_total_items_l2185_218551

/-- Given that Alexis bought 3 times more pants and dresses than Isabella,
    and Alexis bought 21 pairs of pants and 18 dresses,
    prove that Isabella bought a total of 13 items (pants and dresses combined). -/
theorem isabella_total_items (alexis_pants : ℕ) (alexis_dresses : ℕ) 
    (h1 : alexis_pants = 21) 
    (h2 : alexis_dresses = 18) 
    (h3 : ∃ (isabella_pants isabella_dresses : ℕ), 
      alexis_pants = 3 * isabella_pants ∧ 
      alexis_dresses = 3 * isabella_dresses) : 
  ∃ (isabella_total : ℕ), isabella_total = 13 := by
  sorry

end NUMINAMATH_CALUDE_isabella_total_items_l2185_218551


namespace NUMINAMATH_CALUDE_bernardo_wins_l2185_218513

theorem bernardo_wins (N : ℕ) : N ≤ 999 ∧ 72 * N < 1000 ∧ 36 * N < 1000 ∧ ∀ m : ℕ, m < N → (72 * m ≥ 1000 ∨ 36 * m ≥ 1000) → N = 13 := by
  sorry

end NUMINAMATH_CALUDE_bernardo_wins_l2185_218513


namespace NUMINAMATH_CALUDE_vector_sum_as_complex_sum_l2185_218500

theorem vector_sum_as_complex_sum :
  let z₁ : ℂ := 1 + 4*I
  let z₂ : ℂ := -3 + 2*I
  z₁ + z₂ = -2 + 6*I :=
by sorry

end NUMINAMATH_CALUDE_vector_sum_as_complex_sum_l2185_218500


namespace NUMINAMATH_CALUDE_min_variance_product_l2185_218557

theorem min_variance_product (a b : ℝ) : 
  2 ≤ 3 ∧ 3 ≤ 3 ∧ 3 ≤ 7 ∧ 7 ≤ a ∧ a ≤ b ∧ b ≤ 12 ∧ 12 ≤ 13.7 ∧ 13.7 ≤ 18.3 ∧ 18.3 ≤ 21 →
  (2 + 3 + 3 + 7 + a + b + 12 + 13.7 + 18.3 + 21) / 10 = 10 →
  a + b = 20 →
  (∀ x y : ℝ, x + y = 20 → (x - 10)^2 + (y - 10)^2 ≥ (a - 10)^2 + (b - 10)^2) →
  a * b = 100 :=
by sorry


end NUMINAMATH_CALUDE_min_variance_product_l2185_218557


namespace NUMINAMATH_CALUDE_teairra_shirt_count_l2185_218592

/-- The number of shirts Teairra has in her closet -/
def num_shirts : ℕ := sorry

/-- The total number of pants Teairra has -/
def total_pants : ℕ := 24

/-- The number of plaid shirts -/
def plaid_shirts : ℕ := 3

/-- The number of purple pants -/
def purple_pants : ℕ := 5

/-- The number of items (shirts and pants) that are neither plaid nor purple -/
def neither_plaid_nor_purple : ℕ := 21

theorem teairra_shirt_count : num_shirts = 5 := by
  sorry

end NUMINAMATH_CALUDE_teairra_shirt_count_l2185_218592


namespace NUMINAMATH_CALUDE_student_language_partition_l2185_218583

/-- Represents a student and the languages they speak -/
structure Student where
  speaksEnglish : Bool
  speaksFrench : Bool
  speaksSpanish : Bool

/-- Represents a group of students -/
def StudentGroup := List Student

/-- Checks if a group satisfies the language requirements -/
def isValidGroup (group : StudentGroup) : Bool :=
  (group.filter (·.speaksEnglish)).length = 10 ∧
  (group.filter (·.speaksFrench)).length = 10 ∧
  (group.filter (·.speaksSpanish)).length = 10

/-- Main theorem -/
theorem student_language_partition 
  (students : List Student)
  (h_english : (students.filter (·.speaksEnglish)).length = 50)
  (h_french : (students.filter (·.speaksFrench)).length = 50)
  (h_spanish : (students.filter (·.speaksSpanish)).length = 50) :
  ∃ (partition : List StudentGroup), 
    partition.length = 5 ∧ 
    (∀ group ∈ partition, isValidGroup group) ∧
    (partition.join = students) :=
  sorry

end NUMINAMATH_CALUDE_student_language_partition_l2185_218583


namespace NUMINAMATH_CALUDE_common_internal_tangent_length_bound_l2185_218568

/-- Two circles touching the sides of an angle but not each other -/
structure AngleTouchingCircles where
  R : ℝ
  r : ℝ
  h1 : R > r
  h2 : R > 0
  h3 : r > 0
  PQ : ℝ
  h4 : PQ > 0

/-- The length of the common internal tangent segment is greater than twice the geometric mean of the radii -/
theorem common_internal_tangent_length_bound (c : AngleTouchingCircles) : 
  c.PQ > 2 * Real.sqrt (c.R * c.r) := by
  sorry

end NUMINAMATH_CALUDE_common_internal_tangent_length_bound_l2185_218568


namespace NUMINAMATH_CALUDE_shirt_price_reduction_l2185_218588

theorem shirt_price_reduction (original_price : ℝ) (h1 : original_price > 0) : 
  let sale_price := 0.70 * original_price
  let final_price := 0.63 * original_price
  ∃ markdown_percent : ℝ, 
    markdown_percent = 10 ∧ 
    final_price = sale_price * (1 - markdown_percent / 100) :=
by sorry

end NUMINAMATH_CALUDE_shirt_price_reduction_l2185_218588


namespace NUMINAMATH_CALUDE_shopping_mall_probabilities_l2185_218550

/-- Probability of purchasing product A -/
def prob_A : ℝ := 0.5

/-- Probability of purchasing product B -/
def prob_B : ℝ := 0.6

/-- Number of customers -/
def n : ℕ := 3

/-- Probability of purchasing at least one product -/
def p : ℝ := 0.8

theorem shopping_mall_probabilities :
  let prob_either := prob_A * (1 - prob_B) + (1 - prob_A) * prob_B
  let prob_at_least_one := 1 - (1 - prob_A) * (1 - prob_B)
  let ξ := fun k => (n.choose k : ℝ) * p^k * (1 - p)^(n - k)
  (prob_either = 0.5) ∧
  (prob_at_least_one = 0.8) ∧
  (ξ 0 = 0.008) ∧
  (ξ 1 = 0.096) ∧
  (ξ 2 = 0.384) ∧
  (ξ 3 = 0.512) := by
  sorry

end NUMINAMATH_CALUDE_shopping_mall_probabilities_l2185_218550


namespace NUMINAMATH_CALUDE_math_homework_pages_l2185_218579

theorem math_homework_pages (total_problems : ℕ) (reading_pages : ℕ) (problems_per_page : ℕ) 
  (h1 : total_problems = 30)
  (h2 : reading_pages = 4)
  (h3 : problems_per_page = 3) :
  total_problems - reading_pages * problems_per_page = 6 * problems_per_page :=
by sorry

end NUMINAMATH_CALUDE_math_homework_pages_l2185_218579


namespace NUMINAMATH_CALUDE_pirate_loot_sum_l2185_218558

def base_6_to_10 (n : List Nat) : Nat :=
  List.foldl (fun acc d => acc * 6 + d) 0 n.reverse

theorem pirate_loot_sum :
  let silver := base_6_to_10 [4, 5, 3, 2]
  let pearls := base_6_to_10 [1, 2, 5, 4]
  let spices := base_6_to_10 [6, 5, 4]
  silver + pearls + spices = 1636 := by
sorry

end NUMINAMATH_CALUDE_pirate_loot_sum_l2185_218558


namespace NUMINAMATH_CALUDE_probability_both_truth_l2185_218597

theorem probability_both_truth (prob_A prob_B : ℝ) 
  (h_A : prob_A = 0.7) 
  (h_B : prob_B = 0.6) : 
  prob_A * prob_B = 0.42 := by
sorry

end NUMINAMATH_CALUDE_probability_both_truth_l2185_218597


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l2185_218571

theorem quadratic_roots_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x - m = 0 ∧ y^2 - 2*y - m = 0) → m ≥ -1 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l2185_218571


namespace NUMINAMATH_CALUDE_four_digit_numbers_with_5_or_8_l2185_218529

/-- The set of digits excluding 0, 5, and 8 -/
def ValidFirstDigits : Finset ℕ := {1, 2, 3, 4, 6, 7, 9}

/-- The set of digits excluding 5 and 8 -/
def ValidOtherDigits : Finset ℕ := {0, 1, 2, 3, 4, 6, 7, 9}

/-- The number of four-digit numbers -/
def TotalFourDigitNumbers : ℕ := 9000

/-- The number of four-digit numbers without 5 or 8 -/
def NumbersWithout5Or8 : ℕ := Finset.card ValidFirstDigits * Finset.card ValidOtherDigits ^ 3

theorem four_digit_numbers_with_5_or_8 :
  TotalFourDigitNumbers - NumbersWithout5Or8 = 5416 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_numbers_with_5_or_8_l2185_218529


namespace NUMINAMATH_CALUDE_two_digit_numbers_count_l2185_218527

/-- The number of permutations of n elements taken r at a time -/
def permutations (n : ℕ) (r : ℕ) : ℕ := 
  if r ≤ n then Nat.factorial n / Nat.factorial (n - r) else 0

/-- The set of digits used -/
def digits : Finset ℕ := {1, 2, 3, 4, 5}

theorem two_digit_numbers_count : permutations (Finset.card digits) 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_numbers_count_l2185_218527


namespace NUMINAMATH_CALUDE_max_value_f_range_of_m_l2185_218507

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := 2 * Real.log x - (1/2) * x^2

-- Define the interval [1/e, e]
def I : Set ℝ := { x | 1/Real.exp 1 ≤ x ∧ x ≤ Real.exp 1 }

-- Statement for part (I)
theorem max_value_f : 
  ∃ (x : ℝ), x ∈ I ∧ f x = Real.log 2 - 1 ∧ ∀ y ∈ I, f y ≤ f x :=
sorry

-- Define the function g for part (II)
def g (a x : ℝ) : ℝ := a * Real.log x

-- Define the intervals for a and x in part (II)
def A : Set ℝ := { a | 0 ≤ a ∧ a ≤ 3/2 }
def X : Set ℝ := { x | 1 < x ∧ x ≤ Real.exp 2 }

-- Statement for part (II)
theorem range_of_m :
  ∀ m : ℝ, (∀ a ∈ A, ∀ x ∈ X, g a x ≥ m + x) ↔ m ≤ -(Real.exp 2) :=
sorry

end NUMINAMATH_CALUDE_max_value_f_range_of_m_l2185_218507


namespace NUMINAMATH_CALUDE_farmer_earnings_l2185_218596

/-- Calculates the total earnings from selling potatoes and carrots given the harvest quantities and pricing. -/
theorem farmer_earnings (potato_count : ℕ) (potato_bundle_size : ℕ) (potato_bundle_price : ℚ)
                        (carrot_count : ℕ) (carrot_bundle_size : ℕ) (carrot_bundle_price : ℚ) :
  potato_count = 250 →
  potato_bundle_size = 25 →
  potato_bundle_price = 190 / 100 →
  carrot_count = 320 →
  carrot_bundle_size = 20 →
  carrot_bundle_price = 2 →
  (potato_count / potato_bundle_size * potato_bundle_price +
   carrot_count / carrot_bundle_size * carrot_bundle_price : ℚ) = 51 := by
  sorry

#eval (250 / 25 * (190 / 100) + 320 / 20 * 2 : ℚ)

end NUMINAMATH_CALUDE_farmer_earnings_l2185_218596


namespace NUMINAMATH_CALUDE_point_on_segment_with_vector_relation_l2185_218548

/-- Given two points M and N in ℝ², and a point P on the line segment MN
    such that vector PN = -2 * vector PM, prove that P has coordinates (2,4) -/
theorem point_on_segment_with_vector_relation (M N P : ℝ × ℝ) :
  M = (-2, 7) →
  N = (10, -2) →
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • M + t • N →
  (N.1 - P.1, N.2 - P.2) = (-2 * (M.1 - P.1), -2 * (M.2 - P.2)) →
  P = (2, 4) := by
  sorry

end NUMINAMATH_CALUDE_point_on_segment_with_vector_relation_l2185_218548


namespace NUMINAMATH_CALUDE_chess_tournament_wins_l2185_218517

theorem chess_tournament_wins (susan_wins susan_losses mike_wins mike_losses lana_losses : ℕ) 
  (h1 : susan_wins = 5)
  (h2 : susan_losses = 1)
  (h3 : mike_wins = 2)
  (h4 : mike_losses = 4)
  (h5 : lana_losses = 5)
  (h6 : susan_wins + mike_wins + lana_losses = susan_losses + mike_losses + lana_wins)
  : lana_wins = 3 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_wins_l2185_218517


namespace NUMINAMATH_CALUDE_spheres_radius_in_cone_l2185_218559

/-- Represents a right circular cone --/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a sphere --/
structure Sphere where
  radius : ℝ

/-- Represents the configuration of three spheres in a cone --/
structure SpheresInCone where
  cone : Cone
  sphere : Sphere
  spheresTangent : Bool
  spheresTangentToBase : Bool
  spheresNotTangentToSides : Bool

/-- The theorem statement --/
theorem spheres_radius_in_cone (config : SpheresInCone) : 
  config.cone.baseRadius = 6 ∧ 
  config.cone.height = 15 ∧ 
  config.spheresTangent ∧ 
  config.spheresTangentToBase ∧ 
  config.spheresNotTangentToSides →
  config.sphere.radius = 27 - 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_spheres_radius_in_cone_l2185_218559


namespace NUMINAMATH_CALUDE_lcm_problem_l2185_218553

theorem lcm_problem (m : ℕ) (h1 : m > 0) (h2 : Nat.lcm 30 m = 90) (h3 : Nat.lcm m 45 = 180) : m = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l2185_218553


namespace NUMINAMATH_CALUDE_store_inventory_price_l2185_218515

theorem store_inventory_price (total_items : ℕ) (discount_rate : ℚ) (sold_rate : ℚ)
  (debt : ℕ) (remaining : ℕ) :
  total_items = 2000 →
  discount_rate = 80 / 100 →
  sold_rate = 90 / 100 →
  debt = 15000 →
  remaining = 3000 →
  ∃ (price : ℚ), price = 50 ∧
    (1 - discount_rate) * (sold_rate * total_items) * price = debt + remaining :=
by sorry

end NUMINAMATH_CALUDE_store_inventory_price_l2185_218515


namespace NUMINAMATH_CALUDE_fraction_simplification_l2185_218530

theorem fraction_simplification (a : ℝ) (h : a ≠ 2) :
  (a^2 / (a - 2)) - ((4*a - 4) / (a - 2)) = a - 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2185_218530


namespace NUMINAMATH_CALUDE_orange_buckets_l2185_218511

/-- Given three buckets of oranges with specific relationships between their quantities
    and a total number of oranges, prove that the first bucket contains 22 oranges. -/
theorem orange_buckets (b1 b2 b3 : ℕ) : 
  b2 = b1 + 17 →
  b3 = b2 - 11 →
  b1 + b2 + b3 = 89 →
  b1 = 22 := by
sorry

end NUMINAMATH_CALUDE_orange_buckets_l2185_218511


namespace NUMINAMATH_CALUDE_triangle_shape_l2185_218567

theorem triangle_shape (A B C : ℝ) (hA : 0 < A ∧ A < π/2) (hB : 0 < B ∧ B < π/2) 
  (hcos : Real.cos A > Real.sin B) : 
  A + B + C = π ∧ C > π/2 :=
sorry

end NUMINAMATH_CALUDE_triangle_shape_l2185_218567


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l2185_218580

theorem gcd_of_three_numbers : Nat.gcd 13926 (Nat.gcd 20031 47058) = 33 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l2185_218580


namespace NUMINAMATH_CALUDE_pqr_value_l2185_218549

theorem pqr_value (p q r : ℤ) 
  (h1 : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)
  (h2 : p + q + r = 29)
  (h3 : 1 / p + 1 / q + 1 / r + 392 / (p * q * r) = 1) :
  p * q * r = 630 := by
sorry

end NUMINAMATH_CALUDE_pqr_value_l2185_218549


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2185_218589

-- Define the polynomial expression
def f (d : ℝ) : ℝ := -(5 - d) * (d + 2 * (5 - d))

-- Define the expanded form of the polynomial
def expanded_form (d : ℝ) : ℝ := -d^2 + 15*d - 50

-- Theorem statement
theorem sum_of_coefficients :
  (∀ d, f d = expanded_form d) →
  (-1 : ℝ) + 15 + (-50) = -36 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2185_218589


namespace NUMINAMATH_CALUDE_horner_v3_value_l2185_218554

def horner_v3 (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℤ) (x : ℤ) : ℤ :=
  let v := a₆
  let v₁ := v * x + a₅
  let v₂ := v₁ * x + a₄
  v₂ * x + a₃

theorem horner_v3_value :
  horner_v3 12 35 (-8) 79 6 5 3 (-4) = -57 := by
  sorry

end NUMINAMATH_CALUDE_horner_v3_value_l2185_218554


namespace NUMINAMATH_CALUDE_final_movie_length_l2185_218503

/-- Given an original movie length of 60 minutes and a cut scene of 6 minutes,
    the final movie length is 54 minutes. -/
theorem final_movie_length (original_length cut_length : ℕ) 
  (h1 : original_length = 60)
  (h2 : cut_length = 6) :
  original_length - cut_length = 54 := by
  sorry

end NUMINAMATH_CALUDE_final_movie_length_l2185_218503


namespace NUMINAMATH_CALUDE_work_completion_time_l2185_218576

theorem work_completion_time 
  (total_work : ℝ) 
  (a_rate : ℝ) 
  (ab_rate : ℝ) 
  (h1 : a_rate = total_work / 12) 
  (h2 : 10 * ab_rate + 9 * a_rate = total_work) :
  ab_rate = total_work / 40 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2185_218576


namespace NUMINAMATH_CALUDE_jerry_recycling_time_l2185_218570

/-- The time it takes Jerry to throw away all the cans -/
def total_time (num_cans : ℕ) (cans_per_trip : ℕ) (drain_time : ℕ) (walk_time : ℕ) : ℕ :=
  let num_trips := (num_cans + cans_per_trip - 1) / cans_per_trip
  let round_trip_time := 2 * walk_time
  let time_per_trip := round_trip_time + drain_time
  num_trips * time_per_trip

/-- Theorem stating that under the given conditions, it takes Jerry 350 seconds to throw away all the cans -/
theorem jerry_recycling_time :
  total_time 28 4 30 10 = 350 := by
  sorry

end NUMINAMATH_CALUDE_jerry_recycling_time_l2185_218570


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l2185_218521

theorem simultaneous_equations_solution :
  ∀ (a x : ℝ),
    (5 * x^3 + a * x^2 + 8 = 0 ∧ 5 * x^3 + 8 * x^2 + a = 0) ↔
    ((a = -13 ∧ x = 1) ∨ (a = -3 ∧ x = -1) ∨ (a = 8 ∧ x = -2)) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l2185_218521


namespace NUMINAMATH_CALUDE_max_areas_for_n_eq_one_l2185_218561

/-- Represents a circular disk divided by radii and secant lines -/
structure DividedDisk where
  n : ℕ
  radii : Fin (3 * n)
  secant_lines : Fin 2

/-- The maximum number of non-overlapping areas in a divided disk -/
def max_areas (disk : DividedDisk) : ℕ :=
  sorry

/-- Theorem stating the maximum number of non-overlapping areas for n = 1 -/
theorem max_areas_for_n_eq_one :
  ∀ (disk : DividedDisk),
    disk.n = 1 →
    max_areas disk = 15 :=
  sorry

end NUMINAMATH_CALUDE_max_areas_for_n_eq_one_l2185_218561


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l2185_218595

theorem greatest_divisor_four_consecutive_integers :
  ∀ n : ℕ, n > 0 →
  ∃ k : ℕ, k > 0 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) ∧
  ∀ m : ℕ, m > k → ¬(∀ i : ℕ, i > 0 → m ∣ (i * (i + 1) * (i + 2) * (i + 3))) →
  k = 24 :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l2185_218595


namespace NUMINAMATH_CALUDE_adult_ticket_price_l2185_218555

theorem adult_ticket_price (total_tickets : ℕ) (senior_price : ℕ) (total_receipts : ℕ) (senior_tickets : ℕ) :
  total_tickets = 510 →
  senior_price = 15 →
  total_receipts = 8748 →
  senior_tickets = 327 →
  ∃ (adult_price : ℕ), adult_price = 21 ∧ 
    total_receipts = senior_price * senior_tickets + adult_price * (total_tickets - senior_tickets) :=
by
  sorry

end NUMINAMATH_CALUDE_adult_ticket_price_l2185_218555


namespace NUMINAMATH_CALUDE_polyhedron_exists_l2185_218537

/-- A vertex of the polyhedron -/
inductive Vertex : Type
| A | B | C | D | E | F | G | H

/-- An edge of the polyhedron -/
inductive Edge : Type
| AB | AC | AH | BC | BD | CD | DE | EF | EG | FG | FH | GH

/-- A polyhedron structure -/
structure Polyhedron :=
  (vertices : List Vertex)
  (edges : List Edge)

/-- The specific polyhedron we're interested in -/
def specificPolyhedron : Polyhedron :=
  { vertices := [Vertex.A, Vertex.B, Vertex.C, Vertex.D, Vertex.E, Vertex.F, Vertex.G, Vertex.H],
    edges := [Edge.AB, Edge.AC, Edge.AH, Edge.BC, Edge.BD, Edge.CD, Edge.DE, Edge.EF, Edge.EG, Edge.FG, Edge.FH, Edge.GH] }

/-- Theorem stating the existence of the polyhedron -/
theorem polyhedron_exists : ∃ (p : Polyhedron), p = specificPolyhedron :=
sorry

end NUMINAMATH_CALUDE_polyhedron_exists_l2185_218537


namespace NUMINAMATH_CALUDE_opposite_of_2023_l2185_218526

theorem opposite_of_2023 : 
  ∀ x : ℤ, x + 2023 = 0 ↔ x = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l2185_218526


namespace NUMINAMATH_CALUDE_light_travel_distance_l2185_218509

/-- The distance light travels in one year in miles -/
def light_year_distance : ℝ := 5870000000000

/-- The number of years we're calculating for -/
def years : ℕ := 50

/-- Theorem stating the distance light travels in 50 years -/
theorem light_travel_distance : light_year_distance * years = 2935 * (10 : ℝ)^11 := by
  sorry

end NUMINAMATH_CALUDE_light_travel_distance_l2185_218509


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2185_218539

theorem regular_polygon_sides (n : ℕ) (h : n > 2) : 
  (((n - 2) * 180 : ℕ) / n = 144) → n = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2185_218539


namespace NUMINAMATH_CALUDE_total_new_people_value_l2185_218508

/-- The number of people born in the country last year -/
def people_born : ℕ := 90171

/-- The number of people who immigrated to the country last year -/
def people_immigrated : ℕ := 16320

/-- The total number of new people who began living in the country last year -/
def total_new_people : ℕ := people_born + people_immigrated

/-- Theorem stating that the total number of new people is 106491 -/
theorem total_new_people_value : total_new_people = 106491 := by
  sorry

end NUMINAMATH_CALUDE_total_new_people_value_l2185_218508


namespace NUMINAMATH_CALUDE_ellipse_a_range_l2185_218578

/-- An ellipse with equation (x^2)/(a-5) + (y^2)/2 = 1 and foci on the x-axis -/
structure Ellipse (a : ℝ) where
  eq : ∀ (x y : ℝ), x^2 / (a - 5) + y^2 / 2 = 1
  foci_on_x : True  -- This is a placeholder for the condition that foci are on the x-axis

/-- The range of values for a in the given ellipse -/
theorem ellipse_a_range (a : ℝ) (e : Ellipse a) : a > 7 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_a_range_l2185_218578


namespace NUMINAMATH_CALUDE_large_pizza_slices_l2185_218562

/-- Proves that a large pizza has 12 slices given the problem conditions -/
theorem large_pizza_slices :
  ∀ (small_slices medium_slices large_slices : ℕ)
    (total_pizzas small_pizzas medium_pizzas large_pizzas : ℕ)
    (total_slices : ℕ),
  small_slices = 6 →
  medium_slices = 8 →
  total_pizzas = 15 →
  small_pizzas = 4 →
  medium_pizzas = 5 →
  large_pizzas = total_pizzas - small_pizzas - medium_pizzas →
  total_slices = 136 →
  total_slices = small_slices * small_pizzas + medium_slices * medium_pizzas + large_slices * large_pizzas →
  large_slices = 12 := by
sorry

end NUMINAMATH_CALUDE_large_pizza_slices_l2185_218562


namespace NUMINAMATH_CALUDE_largest_reciprocal_l2185_218533

theorem largest_reciprocal (a b c d e : ℚ) : 
  a = 2/7 → b = 3/8 → c = 1 → d = 4 → e = 2000 → 
  (1/a > 1/b ∧ 1/a > 1/c ∧ 1/a > 1/d ∧ 1/a > 1/e) := by
  sorry

end NUMINAMATH_CALUDE_largest_reciprocal_l2185_218533


namespace NUMINAMATH_CALUDE_medicine_price_reduction_l2185_218594

theorem medicine_price_reduction (x : ℝ) : 
  (25 : ℝ) * (1 - x)^2 = 16 ↔ 
  (∃ (price_after_first_reduction : ℝ),
    price_after_first_reduction = 25 * (1 - x) ∧
    16 = price_after_first_reduction * (1 - x)) :=
by sorry

end NUMINAMATH_CALUDE_medicine_price_reduction_l2185_218594


namespace NUMINAMATH_CALUDE_herring_fat_proof_l2185_218566

/-- The amount of fat in ounces for a herring -/
def herring_fat : ℝ := 40

/-- The amount of fat in ounces for an eel -/
def eel_fat : ℝ := 20

/-- The amount of fat in ounces for a pike -/
def pike_fat : ℝ := eel_fat + 10

/-- The number of each type of fish cooked -/
def fish_count : ℕ := 40

/-- The total amount of fat served in ounces -/
def total_fat : ℝ := 3600

theorem herring_fat_proof : 
  herring_fat * fish_count + eel_fat * fish_count + pike_fat * fish_count = total_fat :=
by sorry

end NUMINAMATH_CALUDE_herring_fat_proof_l2185_218566


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l2185_218501

/-- Given a geometric sequence where the second term is 18 and the fifth term is 1458,
    prove that the first term is 6. -/
theorem geometric_sequence_first_term
  (a : ℝ)  -- First term of the sequence
  (r : ℝ)  -- Common ratio of the sequence
  (h1 : a * r = 18)  -- Second term is 18
  (h2 : a * r^4 = 1458)  -- Fifth term is 1458
  : a = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l2185_218501


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implication_l2185_218563

-- Define a type for lines in 3D space
variable (Line : Type)

-- Define the perpendicular relation between two lines
variable (perpendicular : Line → Line → Prop)

-- Define the parallel relation between two lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_parallel_implication (l₁ l₂ l₃ : Line) :
  perpendicular l₁ l₂ → parallel l₂ l₃ → perpendicular l₁ l₃ :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implication_l2185_218563


namespace NUMINAMATH_CALUDE_car_speed_problem_l2185_218506

theorem car_speed_problem (distance : ℝ) (time_difference : ℝ) (speed_difference : ℝ) :
  distance = 150 ∧
  time_difference = 2 ∧
  speed_difference = 10 →
  ∃ (speed_r : ℝ),
    speed_r > 0 ∧
    distance / speed_r - time_difference = distance / (speed_r + speed_difference) ∧
    speed_r = 25 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2185_218506


namespace NUMINAMATH_CALUDE_original_price_l2185_218531

theorem original_price (p q d : ℝ) (h_d_pos : d > 0) :
  let x := d / (1 + (p - q) / 100 - p * q / 10000)
  let price_after_increase := x * (1 + p / 100)
  let final_price := price_after_increase * (1 - q / 100)
  final_price = d :=
by sorry

end NUMINAMATH_CALUDE_original_price_l2185_218531


namespace NUMINAMATH_CALUDE_megan_bought_42_songs_l2185_218585

/-- The number of songs Megan bought given the initial number of albums,
    the number of albums removed, and the number of songs per album. -/
def total_songs (initial_albums : ℕ) (removed_albums : ℕ) (songs_per_album : ℕ) : ℕ :=
  (initial_albums - removed_albums) * songs_per_album

/-- Theorem stating that Megan bought 42 songs in total. -/
theorem megan_bought_42_songs :
  total_songs 8 2 7 = 42 := by
  sorry

end NUMINAMATH_CALUDE_megan_bought_42_songs_l2185_218585


namespace NUMINAMATH_CALUDE_complex_magnitude_l2185_218516

theorem complex_magnitude (s : ℝ) (w : ℂ) (h1 : |s| < 4) (h2 : w + 4 / w = s) : Complex.abs w = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2185_218516


namespace NUMINAMATH_CALUDE_range_of_m_l2185_218575

-- Define the ellipse C
def ellipse_C (m : ℝ) (x y : ℝ) : Prop :=
  x^2 / m + y^2 / (8 - m) = 1

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  x - y + m = 0

-- Define the circle O
def circle_O (x y : ℝ) : Prop :=
  x^2 + y^2 = 9

-- Define proposition p
def prop_p (m : ℝ) : Prop :=
  ∃ c : ℝ, c > 0 ∧ m = 4 + c ∧ 8 - m = 4 - c

-- Define proposition q
def prop_q (m : ℝ) : Prop :=
  abs m ≤ 3 * Real.sqrt 2

-- Main theorem
theorem range_of_m :
  ∀ m : ℝ, (prop_p m ∧ ¬prop_q m) ∨ (¬prop_p m ∧ prop_q m) →
    (3 * Real.sqrt 2 < m ∧ m < 8) ∨ (-3 * Real.sqrt 2 ≤ m ∧ m ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2185_218575


namespace NUMINAMATH_CALUDE_recursive_sequence_solution_l2185_218519

/-- A sequence of real numbers satisfying the given recursion -/
def RecursiveSequence (b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → b n = b (n - 1) * b (n + 1)

theorem recursive_sequence_solution 
  (b : ℕ → ℝ) 
  (h_recursive : RecursiveSequence b) 
  (h_b1 : b 1 = 2 + Real.sqrt 8) 
  (h_b1980 : b 1980 = 15 + Real.sqrt 8) : 
  b 2013 = -1/6 + 13 * Real.sqrt 8 / 6 := by
  sorry

end NUMINAMATH_CALUDE_recursive_sequence_solution_l2185_218519


namespace NUMINAMATH_CALUDE_problem_statement_l2185_218536

theorem problem_statement (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -15)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 20) :
  b / (a + b) + c / (b + c) + a / (c + a) = 19 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2185_218536


namespace NUMINAMATH_CALUDE_soccer_ball_donation_l2185_218512

/-- Calculates the total number of soccer balls donated by a public official to two schools -/
def total_soccer_balls (balls_per_class : ℕ) (num_schools : ℕ) (elementary_classes : ℕ) (middle_classes : ℕ) : ℕ :=
  balls_per_class * num_schools * (elementary_classes + middle_classes)

/-- Proves that the total number of soccer balls donated is 90 -/
theorem soccer_ball_donation : total_soccer_balls 5 2 4 5 = 90 := by
  sorry

end NUMINAMATH_CALUDE_soccer_ball_donation_l2185_218512


namespace NUMINAMATH_CALUDE_oak_grove_library_books_l2185_218518

/-- The number of books in Oak Grove's public library -/
def public_library_books : ℕ := 1986

/-- The number of books in Oak Grove's school libraries -/
def school_library_books : ℕ := 5106

/-- The total number of books in Oak Grove libraries -/
def total_books : ℕ := public_library_books + school_library_books

theorem oak_grove_library_books : total_books = 7092 := by
  sorry

end NUMINAMATH_CALUDE_oak_grove_library_books_l2185_218518


namespace NUMINAMATH_CALUDE_percentage_difference_l2185_218502

theorem percentage_difference (x y : ℝ) (h : x = 8 * y) :
  (x - y) / x * 100 = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l2185_218502


namespace NUMINAMATH_CALUDE_intersection_point_inequality_l2185_218522

theorem intersection_point_inequality (a b : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ Real.exp x₀ = a * Real.sin x₀ + b * Real.sqrt x₀) →
  a^2 + b^2 > Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_inequality_l2185_218522


namespace NUMINAMATH_CALUDE_team_games_theorem_l2185_218573

theorem team_games_theorem (first_games : Nat) (win_rate_first : Real) 
  (win_rate_remaining : Real) (total_win_rate : Real) :
  first_games = 30 →
  win_rate_first = 0.4 →
  win_rate_remaining = 0.8 →
  total_win_rate = 0.6 →
  ∃ (total_games : Nat),
    total_games = 60 ∧
    (first_games : Real) * win_rate_first + 
    (total_games - first_games : Real) * win_rate_remaining = 
    (total_games : Real) * total_win_rate :=
by sorry

#check team_games_theorem

end NUMINAMATH_CALUDE_team_games_theorem_l2185_218573


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l2185_218545

/-- Given a cube with volume 8x cubic units and surface area 2x square units, prove that x = 1728 -/
theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 8*x ∧ 6*s^2 = 2*x) → x = 1728 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l2185_218545


namespace NUMINAMATH_CALUDE_digital_music_library_space_l2185_218505

/-- Calculates the average megabytes per hour of music in a digital library, rounded to the nearest whole number -/
def averageMegabytesPerHour (days : ℕ) (totalMegabytes : ℕ) : ℕ :=
  let hoursPerDay : ℕ := 24
  let totalHours : ℕ := days * hoursPerDay
  let exactAverage : ℚ := totalMegabytes / totalHours
  (exactAverage + 1/2).floor.toNat

theorem digital_music_library_space (days : ℕ) (totalMegabytes : ℕ) 
  (h1 : days = 15) (h2 : totalMegabytes = 20400) :
  averageMegabytesPerHour days totalMegabytes = 57 := by
  sorry

end NUMINAMATH_CALUDE_digital_music_library_space_l2185_218505


namespace NUMINAMATH_CALUDE_polynomial_identity_l2185_218564

theorem polynomial_identity (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l2185_218564


namespace NUMINAMATH_CALUDE_bacon_percentage_is_twenty_l2185_218593

/-- Calculates the percentage of calories from bacon in a sandwich -/
def bacon_calorie_percentage (total_calories : ℕ) (bacon_strips : ℕ) (calories_per_strip : ℕ) : ℚ :=
  (bacon_strips * calories_per_strip : ℚ) / total_calories * 100

/-- Theorem stating that the percentage of calories from bacon in the given sandwich is 20% -/
theorem bacon_percentage_is_twenty :
  bacon_calorie_percentage 1250 2 125 = 20 := by
  sorry

end NUMINAMATH_CALUDE_bacon_percentage_is_twenty_l2185_218593


namespace NUMINAMATH_CALUDE_square_plus_double_perfect_square_l2185_218598

theorem square_plus_double_perfect_square (a : ℕ) : 
  ∃ (k : ℕ), a^2 + 2*a = k^2 ↔ a = 0 :=
sorry

end NUMINAMATH_CALUDE_square_plus_double_perfect_square_l2185_218598


namespace NUMINAMATH_CALUDE_max_value_of_S_l2185_218584

def S (a b c d e f g h k : Int) : Int :=
  a*e*k - a*f*h + b*f*g - b*d*k + c*d*h - c*e*g

theorem max_value_of_S :
  ∃ (a b c d e f g h k : Int),
    (a = 1 ∨ a = -1) ∧
    (b = 1 ∨ b = -1) ∧
    (c = 1 ∨ c = -1) ∧
    (d = 1 ∨ d = -1) ∧
    (e = 1 ∨ e = -1) ∧
    (f = 1 ∨ f = -1) ∧
    (g = 1 ∨ g = -1) ∧
    (h = 1 ∨ h = -1) ∧
    (k = 1 ∨ k = -1) ∧
    S a b c d e f g h k = 4 ∧
    ∀ (a' b' c' d' e' f' g' h' k' : Int),
      (a' = 1 ∨ a' = -1) →
      (b' = 1 ∨ b' = -1) →
      (c' = 1 ∨ c' = -1) →
      (d' = 1 ∨ d' = -1) →
      (e' = 1 ∨ e' = -1) →
      (f' = 1 ∨ f' = -1) →
      (g' = 1 ∨ g' = -1) →
      (h' = 1 ∨ h' = -1) →
      (k' = 1 ∨ k' = -1) →
      S a' b' c' d' e' f' g' h' k' ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_S_l2185_218584


namespace NUMINAMATH_CALUDE_topsoil_cost_theorem_l2185_218544

/-- The cost of topsoil in dollars per cubic foot -/
def cost_per_cubic_foot : ℝ := 8

/-- The conversion factor from cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet : ℝ := 27

/-- The volume of topsoil in cubic yards -/
def volume_in_cubic_yards : ℝ := 7

/-- The cost of topsoil for a given volume in cubic yards -/
def topsoil_cost (volume : ℝ) : ℝ :=
  volume * cubic_yards_to_cubic_feet * cost_per_cubic_foot

theorem topsoil_cost_theorem :
  topsoil_cost volume_in_cubic_yards = 1512 := by
  sorry

end NUMINAMATH_CALUDE_topsoil_cost_theorem_l2185_218544


namespace NUMINAMATH_CALUDE_zinc_copper_mixture_l2185_218542

theorem zinc_copper_mixture (total_weight : ℝ) (zinc_ratio copper_ratio : ℕ) : 
  total_weight = 70 →
  zinc_ratio = 9 →
  copper_ratio = 11 →
  (zinc_ratio : ℝ) / ((zinc_ratio : ℝ) + (copper_ratio : ℝ)) * total_weight = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_zinc_copper_mixture_l2185_218542


namespace NUMINAMATH_CALUDE_ratio_of_Q_at_one_and_minus_one_l2185_218590

/-- The polynomial g(x) = x^2009 + 19x^2008 + 1 -/
def g (x : ℂ) : ℂ := x^2009 + 19*x^2008 + 1

/-- The set of distinct zeros of g(x) -/
def S : Finset ℂ := sorry

/-- The polynomial Q of degree 2009 -/
noncomputable def Q : Polynomial ℂ := sorry

theorem ratio_of_Q_at_one_and_minus_one 
  (h1 : ∀ s ∈ S, g s = 0)
  (h2 : Finset.card S = 2009)
  (h3 : ∀ s ∈ S, Q.eval (s + 1/s) = 0)
  (h4 : Polynomial.degree Q = 2009) :
  Q.eval 1 / Q.eval (-1) = 361 / 331 := by sorry

end NUMINAMATH_CALUDE_ratio_of_Q_at_one_and_minus_one_l2185_218590


namespace NUMINAMATH_CALUDE_lily_book_count_l2185_218535

/-- The number of books Lily read last month -/
def books_last_month : ℕ := 4

/-- The number of books Lily plans to read this month -/
def books_this_month : ℕ := 2 * books_last_month

/-- The total number of books Lily will read in two months -/
def total_books : ℕ := books_last_month + books_this_month

theorem lily_book_count : total_books = 12 := by
  sorry

end NUMINAMATH_CALUDE_lily_book_count_l2185_218535


namespace NUMINAMATH_CALUDE_f_at_negative_two_l2185_218534

/-- Given a function f(x) = 2x^2 - 3x + 1, prove that f(-2) = 15 -/
theorem f_at_negative_two (f : ℝ → ℝ) (h : ∀ x, f x = 2 * x^2 - 3 * x + 1) : 
  f (-2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_f_at_negative_two_l2185_218534


namespace NUMINAMATH_CALUDE_workshop_salary_problem_l2185_218504

/-- Proves that the average salary of non-technician workers is 6000, given the conditions of the workshop --/
theorem workshop_salary_problem (total_workers : ℕ) (avg_salary_all : ℕ) 
  (num_technicians : ℕ) (avg_salary_tech : ℕ) :
  total_workers = 49 →
  avg_salary_all = 8000 →
  num_technicians = 7 →
  avg_salary_tech = 20000 →
  (total_workers - num_technicians) * 
    ((total_workers * avg_salary_all - num_technicians * avg_salary_tech) / 
     (total_workers - num_technicians)) = 
  (total_workers - num_technicians) * 6000 := by
  sorry

#check workshop_salary_problem

end NUMINAMATH_CALUDE_workshop_salary_problem_l2185_218504


namespace NUMINAMATH_CALUDE_focus_after_symmetry_l2185_218546

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = -8*x

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := y = x - 1

-- Define the focus of the original parabola
def original_focus : ℝ × ℝ := (-2, 0)

-- Define the symmetric point
def symmetric_point (a b : ℝ) (p : ℝ × ℝ) : Prop :=
  (b + p.2) / 2 = (p.1 + a) / 2 - 1 ∧
  (b - p.2) / (a - p.1) = -1

-- Theorem statement
theorem focus_after_symmetry :
  ∃ (a b : ℝ), symmetric_point a b original_focus ∧ a = 1 ∧ b = -3 :=
sorry

end NUMINAMATH_CALUDE_focus_after_symmetry_l2185_218546


namespace NUMINAMATH_CALUDE_melon_amount_in_fruit_salad_l2185_218541

/-- Given a fruit salad with melon and berries, prove the amount of melon used. -/
theorem melon_amount_in_fruit_salad
  (total_fruit : ℝ)
  (berries : ℝ)
  (h_total : total_fruit = 0.63)
  (h_berries : berries = 0.38) :
  total_fruit - berries = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_melon_amount_in_fruit_salad_l2185_218541


namespace NUMINAMATH_CALUDE_liquid_ratio_after_replacement_l2185_218514

def container_capacity : ℝ := 37.5

def liquid_replaced : ℝ := 15

def final_ratio_A : ℝ := 9

def final_ratio_B : ℝ := 16

theorem liquid_ratio_after_replacement :
  let initial_A := container_capacity
  let first_step_A := initial_A - liquid_replaced
  let first_step_B := liquid_replaced
  let second_step_A := first_step_A * (1 - liquid_replaced / container_capacity)
  let second_step_B := container_capacity - second_step_A
  (second_step_A / final_ratio_A = second_step_B / final_ratio_B) ∧
  (second_step_A + second_step_B = container_capacity) := by
  sorry

end NUMINAMATH_CALUDE_liquid_ratio_after_replacement_l2185_218514


namespace NUMINAMATH_CALUDE_cubic_equation_integer_roots_l2185_218591

theorem cubic_equation_integer_roots :
  ∃! p : ℝ, 
    (∃ x y z : ℕ+, 
      (5 * (x : ℝ)^3 - 5*(p+1)*(x : ℝ)^2 + (71*p - 1)*(x : ℝ) + 1 = 66*p) ∧
      (5 * (y : ℝ)^3 - 5*(p+1)*(y : ℝ)^2 + (71*p - 1)*(y : ℝ) + 1 = 66*p) ∧
      (5 * (z : ℝ)^3 - 5*(p+1)*(z : ℝ)^2 + (71*p - 1)*(z : ℝ) + 1 = 66*p) ∧
      x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧
    p = 76 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_integer_roots_l2185_218591


namespace NUMINAMATH_CALUDE_dihedral_angle_lower_bound_l2185_218523

/-- Given a regular n-sided polygon inscribed in an arbitrary great circle of a sphere,
    with tangent planes laid at each vertex, the dihedral angle φ of the resulting
    polyhedral angle satisfies φ ≥ π(1 - 2/n). -/
theorem dihedral_angle_lower_bound (n : ℕ) (φ : ℝ) 
  (h1 : n ≥ 3)  -- n is at least 3 for a polygon
  (h2 : φ > 0)  -- dihedral angle is positive
  (h3 : φ < π)  -- dihedral angle is less than π
  : φ ≥ π * (1 - 2 / n) :=
sorry

end NUMINAMATH_CALUDE_dihedral_angle_lower_bound_l2185_218523


namespace NUMINAMATH_CALUDE_no_always_largest_l2185_218552

theorem no_always_largest (a b c d : ℝ) (h : a - 2 = b + 3 ∧ a - 2 = c * 2 ∧ a - 2 = d + 5) :
  ¬(∀ x y : ℝ, (x = a ∨ x = b ∨ x = c ∨ x = d) → (y = a ∨ y = b ∨ y = c ∨ y = d) → x ≤ y) :=
by sorry

end NUMINAMATH_CALUDE_no_always_largest_l2185_218552


namespace NUMINAMATH_CALUDE_cricketer_average_score_l2185_218543

theorem cricketer_average_score 
  (initial_average : ℝ) 
  (runs_19th_inning : ℝ) 
  (average_increase : ℝ) : 
  runs_19th_inning = 96 →
  average_increase = 4 →
  (18 * initial_average + runs_19th_inning) / 19 = initial_average + average_increase →
  (18 * initial_average + runs_19th_inning) / 19 = 24 :=
by sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l2185_218543


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2185_218520

/-- 
Given a geometric sequence {a_n} with positive terms, 
if a_1 = 3 and S_3 = 21, then a_3 + a_4 + a_5 = 84.
-/
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- all terms are positive
  (∀ n, a (n + 1) = a n * q) →  -- definition of geometric sequence
  a 1 = 3 →  -- first term
  (a 1 + a 2 + a 3 = 21) →  -- S_3 = 21
  (a 3 + a 4 + a 5 = 84) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2185_218520


namespace NUMINAMATH_CALUDE_smallest_number_with_properties_l2185_218556

def ends_with_six (n : ℕ) : Prop :=
  n % 10 = 6

def move_six_to_front (n : ℕ) : ℕ :=
  let k := (Nat.log 10 n).succ
  6 * 10^k + (n - 6) / 10

theorem smallest_number_with_properties :
  ∃ (N : ℕ), N = 153846 ∧
  ends_with_six N ∧
  move_six_to_front N = 4 * N ∧
  ∀ (m : ℕ), m < N →
    ¬(ends_with_six m ∧ move_six_to_front m = 4 * m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_properties_l2185_218556


namespace NUMINAMATH_CALUDE_second_person_work_time_l2185_218540

/-- Given two persons who can finish a job in 8 days, where the first person alone can finish the job in 24 days, prove that the second person alone will take 12 days to finish the job. -/
theorem second_person_work_time (total_time : ℝ) (first_person_time : ℝ) (second_person_time : ℝ) : 
  total_time = 8 → first_person_time = 24 → second_person_time = 12 := by
  sorry

#check second_person_work_time

end NUMINAMATH_CALUDE_second_person_work_time_l2185_218540


namespace NUMINAMATH_CALUDE_inequality_proof_l2185_218560

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (Real.sqrt (a^2 + 8*b*c))/a + (Real.sqrt (b^2 + 8*a*c))/b + (Real.sqrt (c^2 + 8*a*b))/c ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2185_218560


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2185_218538

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x + 1 < 0) → a ∈ Set.Iio (-2) ∪ Set.Ioi 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2185_218538


namespace NUMINAMATH_CALUDE_division_powers_equality_l2185_218581

theorem division_powers_equality (a : ℝ) (h : a ≠ 0) :
  a^6 / ((1/2) * a^2) = 2 * a^4 := by sorry

end NUMINAMATH_CALUDE_division_powers_equality_l2185_218581


namespace NUMINAMATH_CALUDE_student_line_count_l2185_218574

/-- The number of students in the line -/
def num_students : ℕ := 26

/-- The counting cycle -/
def cycle_length : ℕ := 4

/-- The last number called -/
def last_number : ℕ := 2

theorem student_line_count :
  num_students % cycle_length = last_number :=
by sorry

end NUMINAMATH_CALUDE_student_line_count_l2185_218574


namespace NUMINAMATH_CALUDE_dice_probability_l2185_218572

/-- The number of sides on each die -/
def n : ℕ := 4025

/-- The threshold for the first die -/
def k : ℕ := 2012

/-- The probability that the first die is less than or equal to k,
    given that it's greater than or equal to the second die -/
def prob : ℚ :=
  (k * (k + 1)) / (n * (n + 1))

theorem dice_probability :
  prob = 1006 / 4025 := by sorry

end NUMINAMATH_CALUDE_dice_probability_l2185_218572


namespace NUMINAMATH_CALUDE_cone_height_l2185_218565

/-- Given a cone whose lateral surface development is a sector with radius 2 and central angle 180°,
    the height of the cone is √3. -/
theorem cone_height (r : ℝ) (l : ℝ) (h : ℝ) :
  r = 1 →  -- radius of the base (derived from the sector's arc length)
  l = 2 →  -- slant height (radius of the sector)
  h^2 + r^2 = l^2 →  -- Pythagorean theorem
  h = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cone_height_l2185_218565


namespace NUMINAMATH_CALUDE_triangle_theorem_l2185_218587

theorem triangle_theorem (a b c A B C : ℝ) (h1 : a * Real.cos C + (1/2) * c = b)
                         (h2 : b = 4) (h3 : c = 6) : 
  A = π/3 ∧ Real.cos B = 2/Real.sqrt 7 ∧ Real.cos (A + 2*B) = -11/14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2185_218587


namespace NUMINAMATH_CALUDE_train_meet_time_l2185_218525

/-- The time (in hours after midnight) when the trains meet -/
def meet_time : ℝ := 11

/-- The time (in hours after midnight) when the train from B starts -/
def start_time_B : ℝ := 8

/-- The distance between stations A and B in kilometers -/
def distance : ℝ := 155

/-- The speed of the train from A in km/h -/
def speed_A : ℝ := 20

/-- The speed of the train from B in km/h -/
def speed_B : ℝ := 25

/-- The time (in hours after midnight) when the train from A starts -/
def start_time_A : ℝ := 7

theorem train_meet_time :
  start_time_A = meet_time - (distance - speed_B * (meet_time - start_time_B)) / speed_A :=
by sorry

end NUMINAMATH_CALUDE_train_meet_time_l2185_218525


namespace NUMINAMATH_CALUDE_probability_of_letter_in_mathematics_l2185_218547

def alphabet : Finset Char := sorry

def mathematics : String := "MATHEMATICS"

def uniqueLetters (s : String) : Finset Char :=
  s.toList.toFinset

theorem probability_of_letter_in_mathematics :
  (uniqueLetters mathematics).card / alphabet.card = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_letter_in_mathematics_l2185_218547


namespace NUMINAMATH_CALUDE_notebook_reorganization_theorem_l2185_218582

/-- Represents the notebook reorganization problem --/
structure NotebookProblem where
  initial_notebooks : ℕ
  pages_per_notebook : ℕ
  initial_drawings_per_page : ℕ
  new_drawings_per_page : ℕ
  full_notebooks_after_reorg : ℕ
  full_pages_in_last_notebook : ℕ

/-- Calculates the number of drawings on the last page after reorganization --/
def drawings_on_last_page (p : NotebookProblem) : ℕ :=
  let total_drawings := p.initial_notebooks * p.pages_per_notebook * p.initial_drawings_per_page
  let full_pages := (p.full_notebooks_after_reorg * p.pages_per_notebook) + p.full_pages_in_last_notebook
  total_drawings - (full_pages * p.new_drawings_per_page)

/-- Theorem stating that for the given problem, the number of drawings on the last page is 4 --/
theorem notebook_reorganization_theorem (p : NotebookProblem) 
  (h1 : p.initial_notebooks = 10)
  (h2 : p.pages_per_notebook = 50)
  (h3 : p.initial_drawings_per_page = 5)
  (h4 : p.new_drawings_per_page = 8)
  (h5 : p.full_notebooks_after_reorg = 6)
  (h6 : p.full_pages_in_last_notebook = 40) :
  drawings_on_last_page p = 4 := by
  sorry

end NUMINAMATH_CALUDE_notebook_reorganization_theorem_l2185_218582


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2185_218586

theorem polynomial_factorization (x y z : ℝ) :
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) =
  (x - y) * (y - z) * (z - x) * (x*y + x*z + y*z) := by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2185_218586


namespace NUMINAMATH_CALUDE_pool_width_is_twelve_l2185_218510

/-- Represents a rectangular swimming pool with a surrounding deck -/
structure PoolWithDeck where
  poolLength : ℝ
  poolWidth : ℝ
  deckWidth : ℝ
  totalArea : ℝ

/-- Theorem stating the width of the swimming pool given specific conditions -/
theorem pool_width_is_twelve (p : PoolWithDeck)
  (h1 : p.poolLength = 10)
  (h2 : p.deckWidth = 4)
  (h3 : p.totalArea = 360)
  (h4 : (p.poolLength + 2 * p.deckWidth) * (p.poolWidth + 2 * p.deckWidth) = p.totalArea) :
  p.poolWidth = 12 := by
  sorry

end NUMINAMATH_CALUDE_pool_width_is_twelve_l2185_218510


namespace NUMINAMATH_CALUDE_alcohol_solution_percentage_l2185_218532

/-- Given a solution with initial volume and alcohol percentage, proves that adding pure alcohol to reach a target percentage results in the correct initial alcohol percentage. -/
theorem alcohol_solution_percentage
  (initial_volume : ℝ)
  (pure_alcohol_added : ℝ)
  (target_percentage : ℝ)
  (h1 : initial_volume = 6)
  (h2 : pure_alcohol_added = 3)
  (h3 : target_percentage = 0.5)
  : ∃ (initial_percentage : ℝ),
    initial_percentage * initial_volume + pure_alcohol_added =
    target_percentage * (initial_volume + pure_alcohol_added) ∧
    initial_percentage = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_solution_percentage_l2185_218532


namespace NUMINAMATH_CALUDE_total_peaches_l2185_218577

theorem total_peaches (num_baskets : ℕ) (red_per_basket : ℕ) (green_per_basket : ℕ) :
  num_baskets = 11 →
  red_per_basket = 10 →
  green_per_basket = 18 →
  num_baskets * (red_per_basket + green_per_basket) = 308 :=
by
  sorry

end NUMINAMATH_CALUDE_total_peaches_l2185_218577


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2185_218569

theorem polynomial_remainder_theorem (f : ℝ → ℝ) (a b c p q r l m n : ℝ) 
  (h_abc : a * b * c ≠ 0)
  (h_rem1 : ∀ x, ∃ k, f x = k * (x - a) * (x - b) + p * x + l)
  (h_rem2 : ∀ x, ∃ k, f x = k * (x - b) * (x - c) + q * x + m)
  (h_rem3 : ∀ x, ∃ k, f x = k * (x - c) * (x - a) + r * x + n) :
  l * (1/a - 1/b) + m * (1/b - 1/c) + n * (1/c - 1/a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2185_218569


namespace NUMINAMATH_CALUDE_largest_perfect_square_product_l2185_218599

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

/-- A function that checks if a number is a one-digit positive integer -/
def is_one_digit_positive (n : ℕ) : Prop :=
  0 < n ∧ n ≤ 9

/-- The main theorem stating that 144 is the largest perfect square
    that can be written as the product of three different one-digit positive integers -/
theorem largest_perfect_square_product : 
  (∀ a b c : ℕ, 
    is_one_digit_positive a ∧ 
    is_one_digit_positive b ∧ 
    is_one_digit_positive c ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    is_perfect_square (a * b * c) →
    a * b * c ≤ 144) ∧
  (∃ a b c : ℕ,
    is_one_digit_positive a ∧
    is_one_digit_positive b ∧
    is_one_digit_positive c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a * b * c = 144 ∧
    is_perfect_square 144) :=
by sorry

end NUMINAMATH_CALUDE_largest_perfect_square_product_l2185_218599
