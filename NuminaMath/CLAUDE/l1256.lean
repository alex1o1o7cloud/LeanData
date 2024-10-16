import Mathlib

namespace NUMINAMATH_CALUDE_digit_difference_in_base_d_l1256_125659

/-- 
Given a base d > 8 and digits A and C in base d,
if AC_d + CC_d = 232_d, then A_d - C_d = 1_d.
-/
theorem digit_difference_in_base_d (d : ℕ) (A C : ℕ) 
  (h_base : d > 8) 
  (h_digits : A < d ∧ C < d) 
  (h_sum : A * d + C + C * d + C = 2 * d^2 + 3 * d + 2) : 
  A - C = 1 :=
sorry

end NUMINAMATH_CALUDE_digit_difference_in_base_d_l1256_125659


namespace NUMINAMATH_CALUDE_nancy_crayon_packs_l1256_125605

/-- The number of crayons Nancy bought -/
def total_crayons : ℕ := 615

/-- The number of crayons in each pack -/
def crayons_per_pack : ℕ := 15

/-- The number of packs Nancy bought -/
def number_of_packs : ℕ := total_crayons / crayons_per_pack

theorem nancy_crayon_packs : number_of_packs = 41 := by
  sorry

end NUMINAMATH_CALUDE_nancy_crayon_packs_l1256_125605


namespace NUMINAMATH_CALUDE_complex_square_simplification_l1256_125608

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_square_simplification :
  (5 - 3 * i)^2 = 16 - 30 * i :=
by
  -- The proof would go here, but we're skipping it as per instructions
  sorry

end NUMINAMATH_CALUDE_complex_square_simplification_l1256_125608


namespace NUMINAMATH_CALUDE_geometric_progression_fourth_term_l1256_125632

theorem geometric_progression_fourth_term : 
  ∀ (a : ℝ) (r : ℝ),
  a > 0 → r > 0 →
  a = 2^(1/3 : ℝ) →
  a * r = 2^(1/4 : ℝ) →
  a * r^2 = 2^(1/5 : ℝ) →
  a * r^3 = 2^(1/9 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_fourth_term_l1256_125632


namespace NUMINAMATH_CALUDE_gumdrop_cost_l1256_125607

/-- Given 80 cents to buy 20 gumdrops, prove that each gumdrop costs 4 cents. -/
theorem gumdrop_cost (total_money : ℕ) (num_gumdrops : ℕ) (cost_per_gumdrop : ℕ) :
  total_money = 80 ∧ num_gumdrops = 20 ∧ total_money = num_gumdrops * cost_per_gumdrop →
  cost_per_gumdrop = 4 := by
sorry

end NUMINAMATH_CALUDE_gumdrop_cost_l1256_125607


namespace NUMINAMATH_CALUDE_product_of_roots_is_root_of_sextic_l1256_125642

theorem product_of_roots_is_root_of_sextic (a b c d : ℝ) : 
  a^4 + a^3 - 1 = 0 → 
  b^4 + b^3 - 1 = 0 → 
  c^4 + c^3 - 1 = 0 → 
  d^4 + d^3 - 1 = 0 → 
  (a * b)^6 + (a * b)^4 + (a * b)^3 - (a * b)^2 - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_is_root_of_sextic_l1256_125642


namespace NUMINAMATH_CALUDE_square_roots_equality_l1256_125667

theorem square_roots_equality (x a : ℝ) (hx : x > 0) :
  (3 * a - 14) ^ 2 = x ∧ (a - 2) ^ 2 = x → a = 4 ∧ x = 4 :=
by sorry

end NUMINAMATH_CALUDE_square_roots_equality_l1256_125667


namespace NUMINAMATH_CALUDE_trapezium_k_value_l1256_125637

/-- A trapezium PQRS with specific angle relationships -/
structure Trapezium where
  /-- Angle PQR -/
  pqr : ℝ
  /-- Angle SPQ = 2 * PQR -/
  spq : ℝ
  /-- Angle RSP = 2 * SPQ -/
  rsp : ℝ
  /-- Angle QRS = k * PQR -/
  qrs : ℝ
  /-- The value of k -/
  k : ℝ
  /-- SPQ is twice PQR -/
  h_spq : spq = 2 * pqr
  /-- RSP is twice SPQ -/
  h_rsp : rsp = 2 * spq
  /-- QRS is k times PQR -/
  h_qrs : qrs = k * pqr
  /-- Sum of angles in a quadrilateral is 360° -/
  h_sum : pqr + spq + rsp + qrs = 360

/-- The value of k in the trapezium PQRS is 5 -/
theorem trapezium_k_value (t : Trapezium) : t.k = 5 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_k_value_l1256_125637


namespace NUMINAMATH_CALUDE_triangle_cosine_inequality_l1256_125676

theorem triangle_cosine_inequality (A B C : Real) (h : A + B + C = π) :
  (Real.cos A / Real.cos B)^2 + (Real.cos B / Real.cos C)^2 + (Real.cos C / Real.cos A)^2 ≥ 
  4 * (Real.cos A^2 + Real.cos B^2 + Real.cos C^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_inequality_l1256_125676


namespace NUMINAMATH_CALUDE_area_bounds_l1256_125655

/-- An acute triangle with sides a, b, c and area t, satisfying abc = a + b + c -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  t : ℝ
  acute : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b
  area_condition : t > 0
  side_condition : a * b * c = a + b + c

/-- The area of an acute triangle satisfying the given conditions is bounded -/
theorem area_bounds (triangle : AcuteTriangle) : 1 < triangle.t ∧ triangle.t ≤ (3 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_area_bounds_l1256_125655


namespace NUMINAMATH_CALUDE_rectangle_area_l1256_125695

/-- A rectangle with one side of length 4 and a diagonal of length 5 has an area of 12. -/
theorem rectangle_area (w l d : ℝ) (hw : w = 4) (hd : d = 5) (h_pythagorean : w^2 + l^2 = d^2) : w * l = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1256_125695


namespace NUMINAMATH_CALUDE_nina_money_proof_l1256_125681

/-- The amount of money Nina has -/
def nina_money : ℕ := 48

/-- The original cost of each widget -/
def original_cost : ℕ := 8

/-- The reduced cost of each widget -/
def reduced_cost : ℕ := original_cost - 2

theorem nina_money_proof :
  (nina_money = 6 * original_cost) ∧
  (nina_money = 8 * reduced_cost) :=
by sorry

end NUMINAMATH_CALUDE_nina_money_proof_l1256_125681


namespace NUMINAMATH_CALUDE_gcd_1963_1891_l1256_125683

theorem gcd_1963_1891 : Nat.gcd 1963 1891 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1963_1891_l1256_125683


namespace NUMINAMATH_CALUDE_min_distance_squared_l1256_125696

noncomputable def e : ℝ := Real.exp 1

theorem min_distance_squared (a b c d : ℝ) 
  (h1 : b = a - 2 * e^a) 
  (h2 : c + d = 4) : 
  ∃ (min : ℝ), min = 18 ∧ ∀ (x y : ℝ), (x - c)^2 + (y - d)^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_distance_squared_l1256_125696


namespace NUMINAMATH_CALUDE_average_marks_first_five_subjects_l1256_125636

theorem average_marks_first_five_subjects 
  (total_subjects : Nat) 
  (average_six_subjects : ℝ) 
  (marks_sixth_subject : ℝ) 
  (h1 : total_subjects = 6) 
  (h2 : average_six_subjects = 76) 
  (h3 : marks_sixth_subject = 86) : 
  (total_subjects - 1 : ℝ)⁻¹ * (total_subjects * average_six_subjects - marks_sixth_subject) = 74 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_first_five_subjects_l1256_125636


namespace NUMINAMATH_CALUDE_temperature_data_inconsistency_l1256_125624

theorem temperature_data_inconsistency 
  (x_bar : ℝ) 
  (m : ℝ) 
  (S_squared : ℝ) 
  (hx : x_bar = 0) 
  (hm : m = 4) 
  (hS : S_squared = 15.917) : 
  ¬(|x_bar - m| ≤ Real.sqrt S_squared) := by
sorry

end NUMINAMATH_CALUDE_temperature_data_inconsistency_l1256_125624


namespace NUMINAMATH_CALUDE_train_crossing_time_l1256_125647

/-- Calculates the time taken for a train to cross a platform -/
theorem train_crossing_time (train_length : Real) (signal_cross_time : Real) (platform_length : Real) :
  train_length = 300 ∧ 
  signal_cross_time = 16 ∧ 
  platform_length = 431.25 →
  (train_length + platform_length) / (train_length / signal_cross_time) = 39 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l1256_125647


namespace NUMINAMATH_CALUDE_consecutive_heads_probability_l1256_125664

/-- The number of coin flips -/
def n : ℕ := 12

/-- The number of desired heads -/
def k : ℕ := 9

/-- The probability of getting heads on a single flip of a fair coin -/
def p : ℚ := 1/2

/-- The number of ways to arrange k consecutive heads in n flips -/
def consecutive_arrangements : ℕ := n - k + 1

theorem consecutive_heads_probability :
  (consecutive_arrangements : ℚ) * p^k * (1-p)^(n-k) = 1/1024 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_heads_probability_l1256_125664


namespace NUMINAMATH_CALUDE_min_value_theorem_l1256_125622

theorem min_value_theorem (a : ℝ) (ha : a > 0) :
  (∃ (x_min : ℝ), x_min > 0 ∧
    ∀ (x : ℝ), x > 0 → (a^2 + x^2) / x ≥ (a^2 + x_min^2) / x_min) ∧
  (∀ (x : ℝ), x > 0 → (a^2 + x^2) / x ≥ 2*a) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1256_125622


namespace NUMINAMATH_CALUDE_unique_spicy_pair_l1256_125694

/-- A three-digit number is spicy if it equals the sum of the cubes of its digits. -/
def IsSpicy (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ 
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  n = a^3 + b^3 + c^3

/-- 370 is the unique three-digit number n such that both n and n+1 are spicy. -/
theorem unique_spicy_pair : ∀ n : ℕ, (IsSpicy n ∧ IsSpicy (n + 1)) ↔ n = 370 := by
  sorry

end NUMINAMATH_CALUDE_unique_spicy_pair_l1256_125694


namespace NUMINAMATH_CALUDE_customers_who_tipped_l1256_125698

/-- The number of customers who left a tip at 'The Gourmet Kitchen' restaurant --/
theorem customers_who_tipped (total_customers early_morning_customers priority_customers regular_evening_customers : ℕ)
  (h_total : total_customers = 215)
  (h_early : early_morning_customers = 20)
  (h_priority : priority_customers = 60)
  (h_regular : regular_evening_customers = 22)
  (h_early_no_tip : ⌊early_morning_customers * (30 : ℚ) / 100⌋ = 6)
  (h_priority_no_tip : ⌊priority_customers * (60 : ℚ) / 100⌋ = 36)
  (h_regular_no_tip : ⌊regular_evening_customers * (50 : ℚ) / 100⌋ = 11)
  (h_remaining : total_customers - early_morning_customers - priority_customers - regular_evening_customers = 113)
  (h_remaining_no_tip : ⌊113 * (25 : ℚ) / 100⌋ = 28) :
  total_customers - (6 + 36 + 11 + 28) = 134 := by
  sorry

#check customers_who_tipped

end NUMINAMATH_CALUDE_customers_who_tipped_l1256_125698


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1256_125613

-- Problem 1
theorem problem_1 : (π - 3.14) ^ 0 + (1/2) ^ (-1) + (-1) ^ 2023 = 2 := by sorry

-- Problem 2
theorem problem_2 (b : ℝ) : (-b)^2 * b + 6*b^4 / (2*b) + (-2*b)^3 = -4*b^3 := by sorry

-- Problem 3
theorem problem_3 (x : ℝ) : (x - 1)^2 - x*(x + 2) = -4*x + 1 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1256_125613


namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_36_l1256_125689

theorem ceiling_neg_sqrt_36 : ⌈-Real.sqrt 36⌉ = -6 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_36_l1256_125689


namespace NUMINAMATH_CALUDE_power_simplification_l1256_125666

theorem power_simplification : 10^6 * (10^2)^3 / 10^4 = 10^8 := by
  sorry

end NUMINAMATH_CALUDE_power_simplification_l1256_125666


namespace NUMINAMATH_CALUDE_point_A_distance_theorem_l1256_125690

-- Define the point A on the number line
def A : ℝ → ℝ := λ a ↦ 2 * a + 1

-- Define the distance function from a point to the origin
def distance_to_origin (x : ℝ) : ℝ := |x|

-- Theorem statement
theorem point_A_distance_theorem :
  ∀ a : ℝ, distance_to_origin (A a) = 3 → a = -2 ∨ a = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_point_A_distance_theorem_l1256_125690


namespace NUMINAMATH_CALUDE_hynek_problem_bounds_l1256_125619

/-- Represents a digit assignment for Hynek's problem -/
structure DigitAssignment where
  a : Fin 5
  b : Fin 5
  c : Fin 5
  d : Fin 5
  e : Fin 5
  distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

/-- Calculates the sum for a given digit assignment -/
def calculateSum (assignment : DigitAssignment) : ℕ :=
  (assignment.a + 1) +
  11 * (assignment.b + 1) +
  111 * (assignment.c + 1) +
  1111 * (assignment.d + 1) +
  11111 * (assignment.e + 1)

/-- Checks if a number is divisible by 11 -/
def isDivisibleBy11 (n : ℕ) : Prop :=
  n % 11 = 0

/-- The main theorem stating the smallest and largest possible sums -/
theorem hynek_problem_bounds :
  (∃ (assignment : DigitAssignment),
    isDivisibleBy11 (calculateSum assignment) ∧
    (∀ (other : DigitAssignment),
      isDivisibleBy11 (calculateSum other) →
      calculateSum assignment ≤ calculateSum other)) ∧
  (∃ (assignment : DigitAssignment),
    isDivisibleBy11 (calculateSum assignment) ∧
    (∀ (other : DigitAssignment),
      isDivisibleBy11 (calculateSum other) →
      calculateSum other ≤ calculateSum assignment)) ∧
  (∀ (assignment : DigitAssignment),
    isDivisibleBy11 (calculateSum assignment) →
    23815 ≤ calculateSum assignment ∧ calculateSum assignment ≤ 60589) :=
sorry

end NUMINAMATH_CALUDE_hynek_problem_bounds_l1256_125619


namespace NUMINAMATH_CALUDE_smallest_number_with_digit_sum_47_l1256_125692

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def is_valid_number (n : ℕ) : Prop :=
  sum_of_digits n = 47

theorem smallest_number_with_digit_sum_47 :
  ∀ n : ℕ, is_valid_number n → n ≥ 299999 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_with_digit_sum_47_l1256_125692


namespace NUMINAMATH_CALUDE_ab_plus_cd_equals_27_l1256_125629

theorem ab_plus_cd_equals_27
  (a b c d : ℝ)
  (eq1 : a + b + c = 5)
  (eq2 : a + b + d = -1)
  (eq3 : a + c + d = 8)
  (eq4 : b + c + d = 12) :
  a * b + c * d = 27 := by
sorry

end NUMINAMATH_CALUDE_ab_plus_cd_equals_27_l1256_125629


namespace NUMINAMATH_CALUDE_complex_number_point_l1256_125604

theorem complex_number_point (z : ℂ) : z = Complex.I * (2 + Complex.I) → z.re = -1 ∧ z.im = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_point_l1256_125604


namespace NUMINAMATH_CALUDE_worker_c_completion_time_l1256_125609

/-- Given workers a, b, and c, and their work rates, prove that c can finish the work in 18 days -/
theorem worker_c_completion_time 
  (total_work : ℝ) 
  (work_rate_a : ℝ) 
  (work_rate_b : ℝ) 
  (work_rate_c : ℝ) 
  (h1 : work_rate_a + work_rate_b + work_rate_c = total_work / 4)
  (h2 : work_rate_a = total_work / 12)
  (h3 : work_rate_b = total_work / 9) :
  work_rate_c = total_work / 18 := by
sorry


end NUMINAMATH_CALUDE_worker_c_completion_time_l1256_125609


namespace NUMINAMATH_CALUDE_largest_non_sum_of_composites_l1256_125618

def IsComposite (n : ℕ) : Prop :=
  ∃ m : ℕ, m ≠ 1 ∧ m ≠ n ∧ n % m = 0

def IsSumOfTwoComposites (n : ℕ) : Prop :=
  ∃ a b : ℕ, IsComposite a ∧ IsComposite b ∧ n = a + b

theorem largest_non_sum_of_composites :
  (∀ n : ℕ, n > 11 → IsSumOfTwoComposites n) ∧
  ¬IsSumOfTwoComposites 11 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_of_composites_l1256_125618


namespace NUMINAMATH_CALUDE_parametric_elimination_l1256_125652

theorem parametric_elimination (x y t : ℝ) 
  (hx : x = 1 + 2 * t - 2 * t^2) 
  (hy : y = 2 * (1 + t) * Real.sqrt (1 - t^2)) : 
  y^4 + 2 * y^2 * (x^2 - 12 * x + 9) + x^4 + 8 * x^3 + 18 * x^2 - 27 = 0 := by
  sorry

end NUMINAMATH_CALUDE_parametric_elimination_l1256_125652


namespace NUMINAMATH_CALUDE_platform_length_l1256_125630

/-- The length of a platform given train crossing times -/
theorem platform_length (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) :
  train_length = 420 ∧ platform_time = 60 ∧ pole_time = 30 →
  (train_length / pole_time) * platform_time - train_length = 420 := by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_platform_length_l1256_125630


namespace NUMINAMATH_CALUDE_travel_distance_ratio_l1256_125671

/-- Given a total distance traveled, with specified portions by plane and bus,
    calculate the ratio of train distance to bus distance. -/
theorem travel_distance_ratio
  (total_distance : ℝ)
  (plane_fraction : ℝ)
  (bus_distance : ℝ)
  (h1 : total_distance = 1800)
  (h2 : plane_fraction = 1 / 3)
  (h3 : bus_distance = 720)
  : (total_distance - plane_fraction * total_distance - bus_distance) / bus_distance = 2 / 3 := by
  sorry

#check travel_distance_ratio

end NUMINAMATH_CALUDE_travel_distance_ratio_l1256_125671


namespace NUMINAMATH_CALUDE_red_light_probability_l1256_125651

/-- Represents the duration of each traffic light color in seconds -/
structure TrafficLightDuration where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the probability of seeing a red light -/
def probabilityRedLight (d : TrafficLightDuration) : ℚ :=
  d.red / (d.red + d.yellow + d.green)

/-- Theorem: The probability of seeing a red light is 2/5 for the given durations -/
theorem red_light_probability (d : TrafficLightDuration) 
  (h1 : d.red = 30) 
  (h2 : d.yellow = 5) 
  (h3 : d.green = 40) : 
  probabilityRedLight d = 2/5 := by
  sorry

#eval probabilityRedLight ⟨30, 5, 40⟩

end NUMINAMATH_CALUDE_red_light_probability_l1256_125651


namespace NUMINAMATH_CALUDE_shoe_savings_l1256_125691

theorem shoe_savings (max_budget : ℝ) (original_price : ℝ) (discount_percent : ℝ) 
  (h1 : max_budget = 130)
  (h2 : original_price = 120)
  (h3 : discount_percent = 30) : 
  max_budget - (original_price * (1 - discount_percent / 100)) = 46 := by
  sorry

end NUMINAMATH_CALUDE_shoe_savings_l1256_125691


namespace NUMINAMATH_CALUDE_age_difference_l1256_125601

theorem age_difference (A B C : ℕ) (h1 : C = A - 16) : A + B - (B + C) = 16 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1256_125601


namespace NUMINAMATH_CALUDE_line_parallel_theorem_l1256_125687

/-- Represents a plane in 3D space -/
structure Plane

/-- Represents a line in 3D space -/
structure Line

/-- Defines when a line is contained in a plane -/
def Line.containedIn (l : Line) (p : Plane) : Prop :=
  sorry

/-- Defines when a line is parallel to a plane -/
def Line.parallelToPlane (l : Line) (p : Plane) : Prop :=
  sorry

/-- Defines when two lines are coplanar -/
def Line.coplanar (l1 l2 : Line) : Prop :=
  sorry

/-- Defines when two lines are parallel -/
def Line.parallel (l1 l2 : Line) : Prop :=
  sorry

/-- Theorem: If m is contained in plane a, n is parallel to plane a,
    and m and n are coplanar, then m is parallel to n -/
theorem line_parallel_theorem (a : Plane) (m n : Line) :
  m.containedIn a → n.parallelToPlane a → m.coplanar n → m.parallel n :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_theorem_l1256_125687


namespace NUMINAMATH_CALUDE_smallestSquare_is_square_largestSquare_is_square_smallestSquare_contains_all_digits_largestSquare_contains_all_digits_smallestSquare_is_smallest_largestSquare_is_largest_l1256_125672

/-- A function that checks if a natural number contains all digits from 0 to 9 exactly once -/
def containsAllDigitsOnce (n : ℕ) : Prop := sorry

/-- The smallest perfect square containing all digits from 0 to 9 exactly once -/
def smallestSquare : ℕ := 1026753849

/-- The largest perfect square containing all digits from 0 to 9 exactly once -/
def largestSquare : ℕ := 9814072356

/-- Theorem stating that smallestSquare is a perfect square -/
theorem smallestSquare_is_square : ∃ k : ℕ, k * k = smallestSquare := sorry

/-- Theorem stating that largestSquare is a perfect square -/
theorem largestSquare_is_square : ∃ k : ℕ, k * k = largestSquare := sorry

/-- Theorem stating that smallestSquare contains all digits from 0 to 9 exactly once -/
theorem smallestSquare_contains_all_digits : containsAllDigitsOnce smallestSquare := sorry

/-- Theorem stating that largestSquare contains all digits from 0 to 9 exactly once -/
theorem largestSquare_contains_all_digits : containsAllDigitsOnce largestSquare := sorry

/-- Theorem stating that smallestSquare is the smallest such square -/
theorem smallestSquare_is_smallest :
  ∀ n : ℕ, n < smallestSquare → ¬(∃ k : ℕ, k * k = n ∧ containsAllDigitsOnce n) := sorry

/-- Theorem stating that largestSquare is the largest such square -/
theorem largestSquare_is_largest :
  ∀ n : ℕ, n > largestSquare → ¬(∃ k : ℕ, k * k = n ∧ containsAllDigitsOnce n) := sorry

end NUMINAMATH_CALUDE_smallestSquare_is_square_largestSquare_is_square_smallestSquare_contains_all_digits_largestSquare_contains_all_digits_smallestSquare_is_smallest_largestSquare_is_largest_l1256_125672


namespace NUMINAMATH_CALUDE_equation_solutions_l1256_125638

theorem equation_solutions :
  (∀ x : ℝ, x^2 + 6*x - 7 = 0 ↔ x = -7 ∨ x = 1) ∧
  (∀ x : ℝ, 4*x*(2*x+1) = 3*(2*x+1) ↔ x = -1/2 ∨ x = 3/4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1256_125638


namespace NUMINAMATH_CALUDE_f_composition_equals_pi_squared_plus_one_l1256_125610

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 + 1
  else if x = 0 then Real.pi
  else 0

theorem f_composition_equals_pi_squared_plus_one :
  f (f (f (-2016))) = Real.pi^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_equals_pi_squared_plus_one_l1256_125610


namespace NUMINAMATH_CALUDE_odd_function_value_l1256_125641

def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

theorem odd_function_value (a b c : ℝ) :
  (∀ x, f a b c (-x) = -(f a b c x)) →
  (∀ x ∈ Set.Icc (2*b - 5) (2*b - 3), f a b c x ∈ Set.range (f a b c)) →
  f a b c (1/2) = 9/8 := by
sorry

end NUMINAMATH_CALUDE_odd_function_value_l1256_125641


namespace NUMINAMATH_CALUDE_unique_solution_l1256_125682

/-- Given denominations 3, n, and n+2, returns true if m cents can be formed -/
def can_form_postage (n : ℕ) (m : ℕ) : Prop :=
  ∃ (a b c : ℕ), m = 3 * a + n * b + (n + 2) * c

/-- Returns true if n satisfies the problem conditions -/
def satisfies_conditions (n : ℕ) : Prop :=
  (∀ m > 63, can_form_postage n m) ∧
  ¬(can_form_postage n 63)

theorem unique_solution :
  ∃! n : ℕ, n > 0 ∧ satisfies_conditions n ∧ n = 30 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l1256_125682


namespace NUMINAMATH_CALUDE_cooking_and_weaving_count_l1256_125678

theorem cooking_and_weaving_count (total : ℕ) (yoga cooking weaving cooking_only cooking_and_yoga all : ℕ) 
  (h1 : yoga = 25)
  (h2 : cooking = 15)
  (h3 : weaving = 8)
  (h4 : cooking_only = 2)
  (h5 : cooking_and_yoga = 7)
  (h6 : all = 3) :
  cooking - (cooking_and_yoga + cooking_only) = 6 :=
by sorry

end NUMINAMATH_CALUDE_cooking_and_weaving_count_l1256_125678


namespace NUMINAMATH_CALUDE_numerator_exceeds_denominator_l1256_125628

theorem numerator_exceeds_denominator (x : ℝ) :
  -1 ≤ x ∧ x ≤ 3 →
  (4 * x + 5 > 10 - 3 * x ↔ 5 / 7 < x ∧ x ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_numerator_exceeds_denominator_l1256_125628


namespace NUMINAMATH_CALUDE_all_equations_are_equalities_negative_two_solves_equation_one_and_negative_two_solve_equation_l1256_125656

-- Define what it means for a real number to be a solution to an equation
def IsSolution (x : ℝ) (f : ℝ → ℝ) : Prop := f x = 0

-- All equations are equalities
theorem all_equations_are_equalities : ∀ (f : ℝ → ℝ), ∃ (x : ℝ), f x = 0 → (∃ (y : ℝ), f x = f y) :=
sorry

-- -2 is a solution to 3 - 2x = 7
theorem negative_two_solves_equation : IsSolution (-2 : ℝ) (λ x => 3 - 2*x - 7) :=
sorry

-- 1 and -2 are solutions to (x - 1)(x + 2) = 0
theorem one_and_negative_two_solve_equation : 
  IsSolution (1 : ℝ) (λ x => (x - 1)*(x + 2)) ∧ 
  IsSolution (-2 : ℝ) (λ x => (x - 1)*(x + 2)) :=
sorry

end NUMINAMATH_CALUDE_all_equations_are_equalities_negative_two_solves_equation_one_and_negative_two_solve_equation_l1256_125656


namespace NUMINAMATH_CALUDE_blueberry_zucchini_trade_l1256_125649

/-- The number of containers of blueberries yielded by one bush -/
def containers_per_bush : ℕ := 10

/-- The number of containers of blueberries that can be traded for zucchinis -/
def containers_traded : ℕ := 6

/-- The number of zucchinis received in trade for containers_traded -/
def zucchinis_received : ℕ := 3

/-- The total number of zucchinis Natalie wants to obtain -/
def target_zucchinis : ℕ := 60

/-- The number of bushes needed to obtain the target number of zucchinis -/
def bushes_needed : ℕ := 12

theorem blueberry_zucchini_trade :
  bushes_needed * containers_per_bush * zucchinis_received = 
  target_zucchinis * containers_traded := by
  sorry

end NUMINAMATH_CALUDE_blueberry_zucchini_trade_l1256_125649


namespace NUMINAMATH_CALUDE_average_decrease_l1256_125670

def initial_observations : ℕ := 6
def initial_average : ℚ := 12
def new_observation : ℕ := 5

theorem average_decrease :
  let initial_sum := initial_observations * initial_average
  let new_sum := initial_sum + new_observation
  let new_average := new_sum / (initial_observations + 1)
  initial_average - new_average = 1 := by
  sorry

end NUMINAMATH_CALUDE_average_decrease_l1256_125670


namespace NUMINAMATH_CALUDE_greatest_prime_divisor_digit_sum_l1256_125615

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem greatest_prime_divisor_digit_sum :
  ∃ (p : ℕ), is_prime p ∧ (32767 % p = 0) ∧
  (∀ q : ℕ, is_prime q → (32767 % q = 0) → q ≤ p) ∧
  sum_of_digits p = 14 := by
  sorry

end NUMINAMATH_CALUDE_greatest_prime_divisor_digit_sum_l1256_125615


namespace NUMINAMATH_CALUDE_triangle_area_l1256_125668

theorem triangle_area (a b c : ℝ) (h1 : a = 7) (h2 : b = 24) (h3 : c = 25) :
  (1/2) * a * b = 84 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1256_125668


namespace NUMINAMATH_CALUDE_max_pairs_after_loss_l1256_125648

theorem max_pairs_after_loss (initial_pairs : ℕ) (lost_shoes : ℕ) (max_pairs : ℕ) : 
  initial_pairs = 24 →
  lost_shoes = 9 →
  max_pairs = initial_pairs - (lost_shoes / 2) →
  max_pairs = 20 :=
by sorry

end NUMINAMATH_CALUDE_max_pairs_after_loss_l1256_125648


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l1256_125686

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 - 2*(m-3)*x + 16 = (x - a)^2) → 
  (m = 7 ∨ m = -1) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l1256_125686


namespace NUMINAMATH_CALUDE_solve_equation_l1256_125612

theorem solve_equation : ∃ x : ℚ, 5 * (x - 10) = 3 * (3 - 3 * x) + 9 ∧ x = 34/7 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1256_125612


namespace NUMINAMATH_CALUDE_trigonometric_equation_proof_l1256_125626

theorem trigonometric_equation_proof : 
  4.74 * (Real.cos (64 * π / 180) * Real.cos (4 * π / 180) - 
          Real.cos (86 * π / 180) * Real.cos (26 * π / 180)) / 
         (Real.cos (71 * π / 180) * Real.cos (41 * π / 180) - 
          Real.cos (49 * π / 180) * Real.cos (19 * π / 180)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_proof_l1256_125626


namespace NUMINAMATH_CALUDE_xy_length_l1256_125646

/-- Represents a trapezoid WXYZ with specific properties -/
structure Trapezoid where
  -- W, X, Y, Z are points in the plane
  W : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  -- WX is parallel to ZY
  wx_parallel_zy : (X.1 - W.1) * (Y.2 - Z.2) = (X.2 - W.2) * (Y.1 - Z.1)
  -- WY is perpendicular to ZY
  wy_perp_zy : (Y.1 - W.1) * (Y.1 - Z.1) + (Y.2 - W.2) * (Y.2 - Z.2) = 0
  -- YZ = 15
  yz_length : Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2) = 15
  -- tan Z = 2
  tan_z : (W.2 - Z.2) / (Y.1 - Z.1) = 2
  -- tan X = 3/2
  tan_x : (W.2 - Y.2) / (X.1 - W.1) = 3/2

/-- The length of XY in the trapezoid is 10√13 -/
theorem xy_length (t : Trapezoid) : 
  Real.sqrt ((t.X.1 - t.Y.1)^2 + (t.X.2 - t.Y.2)^2) = 10 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_xy_length_l1256_125646


namespace NUMINAMATH_CALUDE_contrapositive_example_l1256_125660

theorem contrapositive_example :
  (∀ x : ℝ, x > 2 → x > 0) ↔ (∀ x : ℝ, x ≤ 0 → x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_example_l1256_125660


namespace NUMINAMATH_CALUDE_fourth_root_of_four_sixes_l1256_125684

theorem fourth_root_of_four_sixes : 
  (4^6 + 4^6 + 4^6 + 4^6 : ℝ)^(1/4) = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_four_sixes_l1256_125684


namespace NUMINAMATH_CALUDE_diagonal_intersections_count_l1256_125697

/-- Represents a convex polygon with n sides where no two diagonals are parallel
    and no three diagonals intersect at the same point. -/
structure ConvexPolygon (n : ℕ) where
  sides : n ≥ 3
  no_parallel_diagonals : True
  no_triple_intersections : True

/-- The number of intersection points of diagonals outside a convex polygon. -/
def diagonal_intersections_outside (n : ℕ) (p : ConvexPolygon n) : ℚ :=
  (1 / 12 : ℚ) * n * (n - 3) * (n - 4) * (n - 5)

theorem diagonal_intersections_count (n : ℕ) (p : ConvexPolygon n) :
  diagonal_intersections_outside n p = (1 / 12 : ℚ) * n * (n - 3) * (n - 4) * (n - 5) := by
  sorry

end NUMINAMATH_CALUDE_diagonal_intersections_count_l1256_125697


namespace NUMINAMATH_CALUDE_equation_solution_l1256_125640

/-- Given two equations with the same solution for x, prove the value of an expression -/
theorem equation_solution (m n x : ℝ) : 
  (m + 3) * x^(|m| - 2) + 6 * m = 0 →  -- First equation
  n * x - 5 = x * (3 - n) →            -- Second equation
  (|m| - 2 = 0) →                      -- Condition for first-degree equation
  (m + x)^2000 * (-m^2 * n + x * n^2) + 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1256_125640


namespace NUMINAMATH_CALUDE_set_equality_l1256_125627

theorem set_equality : Finset.toSet {1, 2, 3, 4, 5} = Finset.toSet {5, 4, 3, 2, 1} := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l1256_125627


namespace NUMINAMATH_CALUDE_average_of_reeyas_scores_l1256_125616

def reeyas_scores : List ℕ := [55, 67, 76, 82, 85]

theorem average_of_reeyas_scores :
  (List.sum reeyas_scores) / (List.length reeyas_scores) = 73 := by
  sorry

end NUMINAMATH_CALUDE_average_of_reeyas_scores_l1256_125616


namespace NUMINAMATH_CALUDE_f_properties_l1256_125603

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 9 * x^2 + 12 * x + 8

-- Define the interval
def interval : Set ℝ := Set.Icc (-1) 3

-- Theorem for extreme values and max/min in the interval
theorem f_properties :
  (∃ (x : ℝ), x ∈ interval ∧ ∀ (y : ℝ), y ∈ interval → f y ≤ f x) ∧
  (∃ (x : ℝ), x ∈ interval ∧ ∀ (y : ℝ), y ∈ interval → f y ≥ f x) ∧
  (∀ (x : ℝ), x ∈ interval → f x ≤ 14) ∧
  (∀ (x : ℝ), x ∈ interval → f x ≥ -15) ∧
  (f 1 = 13 ∧ f 2 = 12) ∧
  (∀ (x : ℝ), f x ≥ 12 → x = 1 ∨ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1256_125603


namespace NUMINAMATH_CALUDE_cone_slant_height_l1256_125600

/-- Given a cone with lateral surface area of 15π when unfolded and base radius of 3, 
    its slant height is 5. -/
theorem cone_slant_height (lateral_area : ℝ) (base_radius : ℝ) : 
  lateral_area = 15 * Real.pi ∧ base_radius = 3 → 
  (lateral_area / (Real.pi * base_radius) : ℝ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_cone_slant_height_l1256_125600


namespace NUMINAMATH_CALUDE_n2o3_molecular_weight_l1256_125653

/-- The atomic weight of nitrogen in g/mol -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The molecular weight of N2O3 in g/mol -/
def n2o3_weight : ℝ := 2 * nitrogen_weight + 3 * oxygen_weight

/-- Theorem stating that the molecular weight of N2O3 is 76.02 g/mol -/
theorem n2o3_molecular_weight : n2o3_weight = 76.02 := by
  sorry

end NUMINAMATH_CALUDE_n2o3_molecular_weight_l1256_125653


namespace NUMINAMATH_CALUDE_equation_solution_l1256_125685

theorem equation_solution : ∃ x : ℝ, 64 + x * 12 / (180 / 3) = 65 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1256_125685


namespace NUMINAMATH_CALUDE_no_solution_exists_l1256_125674

theorem no_solution_exists : ¬ ∃ (m n : ℤ), 5 * m^2 - 6 * m * n + 7 * n^2 = 1985 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1256_125674


namespace NUMINAMATH_CALUDE_quadratic_sum_of_coefficients_l1256_125675

theorem quadratic_sum_of_coefficients (a b : ℝ) : 
  (∀ x, a * x^2 + b * x - 2 = 0 ↔ x = -2 ∨ x = 1/3) → 
  a + b = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_coefficients_l1256_125675


namespace NUMINAMATH_CALUDE_percentage_difference_l1256_125661

theorem percentage_difference (x y : ℝ) (h : y = x * (1 + 0.53846153846153854)) :
  x = y * (1 - 0.35) := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l1256_125661


namespace NUMINAMATH_CALUDE_bridge_length_l1256_125657

/-- The length of a bridge given train parameters and crossing time -/
theorem bridge_length
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 140)
  (h2 : train_speed_kmh = 45)
  (h3 : crossing_time = 30) :
  train_speed_kmh * (1000 / 3600) * crossing_time - train_length = 235 :=
sorry

end NUMINAMATH_CALUDE_bridge_length_l1256_125657


namespace NUMINAMATH_CALUDE_mod_congruence_unique_n_l1256_125643

theorem mod_congruence_unique_n (a b : ℤ) 
  (ha : a ≡ 22 [ZMOD 50])
  (hb : b ≡ 78 [ZMOD 50]) :
  ∃! n : ℤ, 150 ≤ n ∧ n ≤ 201 ∧ (a - b) ≡ n [ZMOD 50] ∧ n = 194 :=
sorry

end NUMINAMATH_CALUDE_mod_congruence_unique_n_l1256_125643


namespace NUMINAMATH_CALUDE_parabola_chord_theorem_l1256_125645

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a point on the parabola
def point_on_parabola (s : ℝ) : Prop := parabola s 4

-- Define perpendicular lines
def perpendicular (k1 k2 : ℝ) : Prop := k1 * k2 = -1

-- Define a point (x,y) on line AB
def point_on_AB (x y y1 y2 : ℝ) : Prop := y + 4 = (4 / (y1 + y2)) * (x - 8)

theorem parabola_chord_theorem :
  ∀ y1 y2 : ℝ,
  point_on_parabola 4 →
  parabola (y1^2 / 4) y1 →
  parabola (y2^2 / 4) y2 →
  perpendicular ((y1 - 4) / ((y1^2 - 16) / 4)) ((y2 - 4) / ((y2^2 - 16) / 4)) →
  point_on_AB 8 (-4) y1 y2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_chord_theorem_l1256_125645


namespace NUMINAMATH_CALUDE_value_of_b_l1256_125699

theorem value_of_b (a b c : ℝ) 
  (h1 : a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1))
  (h2 : 6 * b * 11 = 1) : 
  b = 15 := by
sorry

end NUMINAMATH_CALUDE_value_of_b_l1256_125699


namespace NUMINAMATH_CALUDE_r_daily_earnings_l1256_125658

/-- Given the daily earnings of three individuals p, q, and r, prove that r earns 60 per day. -/
theorem r_daily_earnings (P Q R : ℚ) 
  (h1 : P + Q + R = 190) 
  (h2 : P + R = 120)
  (h3 : Q + R = 130) : 
  R = 60 := by
  sorry

end NUMINAMATH_CALUDE_r_daily_earnings_l1256_125658


namespace NUMINAMATH_CALUDE_luzhou_gdp_scientific_correct_l1256_125639

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_in_range : 1 ≤ coefficient ∧ coefficient < 10

/-- The GDP value of Luzhou City in 2022 -/
def luzhou_gdp : ℕ := 260150000000

/-- The scientific notation representation of Luzhou's GDP -/
def luzhou_gdp_scientific : ScientificNotation :=
  { coefficient := 2.6015
    exponent := 11
    coeff_in_range := by sorry }

/-- Theorem stating that the scientific notation representation is correct -/
theorem luzhou_gdp_scientific_correct :
  (luzhou_gdp_scientific.coefficient * (10 : ℝ) ^ luzhou_gdp_scientific.exponent) = luzhou_gdp := by
  sorry

end NUMINAMATH_CALUDE_luzhou_gdp_scientific_correct_l1256_125639


namespace NUMINAMATH_CALUDE_sphere_volume_area_ratio_l1256_125644

theorem sphere_volume_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (4 / 3 * Real.pi * r₁^3) / (4 / 3 * Real.pi * r₂^3) = 8 / 27 →
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 4 / 9 :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_area_ratio_l1256_125644


namespace NUMINAMATH_CALUDE_probability_white_then_red_l1256_125611

/-- Probability of drawing a white marble first and then a red marble from a bag -/
theorem probability_white_then_red (total_marbles : ℕ) (red_marbles : ℕ) (white_marbles : ℕ) :
  total_marbles = red_marbles + white_marbles →
  red_marbles = 4 →
  white_marbles = 6 →
  (white_marbles : ℚ) / (total_marbles : ℚ) * (red_marbles : ℚ) / ((total_marbles - 1) : ℚ) = 4 / 15 :=
by sorry

end NUMINAMATH_CALUDE_probability_white_then_red_l1256_125611


namespace NUMINAMATH_CALUDE_tan_graph_property_l1256_125693

theorem tan_graph_property (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, a * Real.tan (b * x) = a * Real.tan (b * (x + 2 * Real.pi / 5))) →
  a * Real.tan (b * Real.pi / 10) = 1 →
  a * b = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_graph_property_l1256_125693


namespace NUMINAMATH_CALUDE_jessica_found_41_seashells_l1256_125602

/-- The number of seashells Mary found -/
def mary_seashells : ℕ := 18

/-- The total number of seashells Mary and Jessica found together -/
def total_seashells : ℕ := 59

/-- The number of seashells Jessica found -/
def jessica_seashells : ℕ := total_seashells - mary_seashells

theorem jessica_found_41_seashells : jessica_seashells = 41 := by
  sorry

end NUMINAMATH_CALUDE_jessica_found_41_seashells_l1256_125602


namespace NUMINAMATH_CALUDE_square_perimeter_l1256_125614

theorem square_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 450 → 
  side^2 = area → 
  perimeter = 4 * side → 
  perimeter = 60 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_l1256_125614


namespace NUMINAMATH_CALUDE_complex_magnitude_sum_reciprocals_l1256_125623

theorem complex_magnitude_sum_reciprocals (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 3) :
  Complex.abs (1 / z + 1 / w) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_sum_reciprocals_l1256_125623


namespace NUMINAMATH_CALUDE_park_area_l1256_125633

/-- The area of a rectangular park with sides in ratio 3:2 and fencing cost $150 at 60 ps per meter --/
theorem park_area (length width : ℝ) (area perimeter cost_per_meter total_cost : ℝ) : 
  length / width = 3 / 2 →
  area = length * width →
  perimeter = 2 * (length + width) →
  cost_per_meter = 0.60 →
  total_cost = 150 →
  total_cost = perimeter * cost_per_meter →
  area = 3750 :=
by sorry

end NUMINAMATH_CALUDE_park_area_l1256_125633


namespace NUMINAMATH_CALUDE_banana_cost_l1256_125634

/-- The price of bananas in dollars per 8 pounds -/
def banana_price : ℝ := 6

/-- The quantity of bananas in a standard unit (in pounds) -/
def standard_quantity : ℝ := 8

/-- The discount rate for purchases above 20 pounds -/
def discount_rate : ℝ := 0.1

/-- The threshold quantity for applying the discount (in pounds) -/
def discount_threshold : ℝ := 20

/-- The quantity of bananas to be purchased (in pounds) -/
def purchase_quantity : ℝ := 24

/-- Theorem stating the total cost of bananas -/
theorem banana_cost : 
  let price_per_pound := banana_price / standard_quantity
  let total_cost_before_discount := price_per_pound * purchase_quantity
  let discount_amount := if purchase_quantity > discount_threshold
                         then total_cost_before_discount * discount_rate
                         else 0
  let final_cost := total_cost_before_discount - discount_amount
  final_cost = 16.2 := by sorry

end NUMINAMATH_CALUDE_banana_cost_l1256_125634


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1256_125673

theorem rationalize_denominator : 
  (7 / Real.sqrt 98) = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1256_125673


namespace NUMINAMATH_CALUDE_tissues_per_box_l1256_125665

theorem tissues_per_box (boxes : ℕ) (used : ℕ) (left : ℕ) : 
  boxes = 3 → used = 210 → left = 270 → (used + left) / boxes = 160 :=
by
  sorry

end NUMINAMATH_CALUDE_tissues_per_box_l1256_125665


namespace NUMINAMATH_CALUDE_base_video_card_cost_l1256_125621

/-- Proves the cost of the base video card given the costs of other components --/
theorem base_video_card_cost 
  (computer_cost : ℝ)
  (peripheral_cost : ℝ)
  (upgraded_card_cost : ℝ → ℝ)
  (total_cost : ℝ)
  (h1 : computer_cost = 1500)
  (h2 : peripheral_cost = computer_cost / 5)
  (h3 : ∀ x, upgraded_card_cost x = 2 * x)
  (h4 : total_cost = 2100)
  (h5 : ∃ x, computer_cost + peripheral_cost + upgraded_card_cost x = total_cost) :
  ∃ x, x = 150 ∧ computer_cost + peripheral_cost + upgraded_card_cost x = total_cost :=
by sorry

end NUMINAMATH_CALUDE_base_video_card_cost_l1256_125621


namespace NUMINAMATH_CALUDE_bread_distribution_problem_l1256_125688

theorem bread_distribution_problem :
  ∃! (m w c : ℕ),
    m + w + c = 12 ∧
    2 * m + (1/2) * w + (1/4) * c = 12 ∧
    m ≥ 0 ∧ w ≥ 0 ∧ c ≥ 0 ∧
    m = 5 ∧ w = 1 ∧ c = 6 :=
by sorry

end NUMINAMATH_CALUDE_bread_distribution_problem_l1256_125688


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l1256_125677

theorem rectangle_area_diagonal (length width diagonal : ℝ) (h1 : length / width = 4 / 3) 
  (h2 : length^2 + width^2 = diagonal^2) : 
  length * width = (12 / 25) * diagonal^2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l1256_125677


namespace NUMINAMATH_CALUDE_subtract_fractions_l1256_125617

theorem subtract_fractions : (7/3 * 12/5) - 3/5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_subtract_fractions_l1256_125617


namespace NUMINAMATH_CALUDE_exponent_base_problem_l1256_125620

theorem exponent_base_problem (x : ℝ) (y : ℝ) :
  4^(2*x + 2) = y^(3*x - 1) → x = 1 → y = 16 := by
  sorry

end NUMINAMATH_CALUDE_exponent_base_problem_l1256_125620


namespace NUMINAMATH_CALUDE_roof_collapse_days_l1256_125654

theorem roof_collapse_days (roof_weight_limit : ℕ) (leaves_per_day : ℕ) (leaves_per_pound : ℕ) : 
  roof_weight_limit = 500 → 
  leaves_per_day = 100 → 
  leaves_per_pound = 1000 → 
  (roof_weight_limit * leaves_per_pound) / leaves_per_day = 5000 := by
  sorry

#check roof_collapse_days

end NUMINAMATH_CALUDE_roof_collapse_days_l1256_125654


namespace NUMINAMATH_CALUDE_teresa_total_score_l1256_125635

def teresa_scores (science music social_studies : ℕ) : ℕ → Prop :=
  λ total => 
    let physics := music / 2
    total = science + music + social_studies + physics

theorem teresa_total_score : 
  teresa_scores 70 80 85 275 := by sorry

end NUMINAMATH_CALUDE_teresa_total_score_l1256_125635


namespace NUMINAMATH_CALUDE_proposition_logic_l1256_125631

theorem proposition_logic (p q : Prop) (hp : p ↔ (3 + 3 = 5)) (hq : q ↔ (6 > 3)) :
  (p ∨ q) ∧ (¬q ↔ False) := by
  sorry

end NUMINAMATH_CALUDE_proposition_logic_l1256_125631


namespace NUMINAMATH_CALUDE_equation_solution_l1256_125662

theorem equation_solution : ∃! x : ℝ, 4 * x - 3 = 5 * x + 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1256_125662


namespace NUMINAMATH_CALUDE_rachel_envelope_stuffing_l1256_125669

/-- Rachel's envelope stuffing problem -/
theorem rachel_envelope_stuffing 
  (total_time : ℕ) 
  (total_envelopes : ℕ) 
  (first_hour : ℕ) 
  (second_hour : ℕ) 
  (h1 : total_time = 8)
  (h2 : total_envelopes = 1500)
  (h3 : first_hour = 135)
  (h4 : second_hour = 141) : 
  (total_envelopes - first_hour - second_hour) / (total_time - 2) = 204 := by
sorry

end NUMINAMATH_CALUDE_rachel_envelope_stuffing_l1256_125669


namespace NUMINAMATH_CALUDE_perimeter_of_isosceles_triangle_l1256_125679

-- Define the condition for x and y
def satisfies_equation (x y : ℝ) : Prop :=
  |x - 4| + Real.sqrt (y - 10) = 0

-- Define an isosceles triangle with side lengths x, y, and y
def isosceles_triangle (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x + y > y ∧ y + y > x

-- Define the perimeter of the triangle
def triangle_perimeter (x y : ℝ) : ℝ :=
  x + y + y

-- Theorem statement
theorem perimeter_of_isosceles_triangle (x y : ℝ) :
  satisfies_equation x y → isosceles_triangle x y → triangle_perimeter x y = 24 :=
by
  sorry


end NUMINAMATH_CALUDE_perimeter_of_isosceles_triangle_l1256_125679


namespace NUMINAMATH_CALUDE_martha_blocks_l1256_125606

/-- Given Martha's initial and found blocks, prove the total number of blocks she ends with. -/
theorem martha_blocks (initial_blocks found_blocks : ℕ) 
  (h1 : initial_blocks = 4)
  (h2 : found_blocks = 80) :
  initial_blocks + found_blocks = 84 := by
  sorry

#check martha_blocks

end NUMINAMATH_CALUDE_martha_blocks_l1256_125606


namespace NUMINAMATH_CALUDE_parking_space_savings_l1256_125625

/-- The cost of renting a parking space for one week in dollars -/
def weekly_cost : ℕ := 10

/-- The cost of renting a parking space for one month in dollars -/
def monthly_cost : ℕ := 24

/-- The number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- The number of months in a year -/
def months_per_year : ℕ := 12

/-- The savings in dollars when renting a parking space by the month instead of by the week for a year -/
theorem parking_space_savings : 
  weeks_per_year * weekly_cost - months_per_year * monthly_cost = 232 := by
  sorry

end NUMINAMATH_CALUDE_parking_space_savings_l1256_125625


namespace NUMINAMATH_CALUDE_scientific_notation_pm25_l1256_125650

theorem scientific_notation_pm25 :
  ∃ (a : ℝ) (n : ℤ), 0.000042 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 4.2 ∧ n = -5 :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_pm25_l1256_125650


namespace NUMINAMATH_CALUDE_calculation_proof_l1256_125663

theorem calculation_proof :
  ((-1 : ℝ)^2023 + |(-3 : ℝ)| - (π - 7)^0 + 2^4 * (1/2 : ℝ)^4 = 2) ∧
  (∀ (a b : ℝ), 6*a^3*b^2 / (3*a^2*b^2) + (2*a*b^3)^2 / (a*b)^2 = 2*a + 4*b^4) :=
by sorry

end NUMINAMATH_CALUDE_calculation_proof_l1256_125663


namespace NUMINAMATH_CALUDE_solution_value_l1256_125680

theorem solution_value (a : ℝ) : (3 * 5 - 2 * a = 7) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l1256_125680
