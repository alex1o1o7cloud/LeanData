import Mathlib

namespace NUMINAMATH_CALUDE_f_composition_half_l495_49580

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp x else Real.log x

theorem f_composition_half : f (f (1/2)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_half_l495_49580


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_open_zero_one_l495_49525

-- Define set M
def M : Set ℝ := {x | x^2 + x - 2 < 0}

-- Define set N
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

-- Theorem statement
theorem M_intersect_N_eq_open_zero_one : M ∩ N = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_open_zero_one_l495_49525


namespace NUMINAMATH_CALUDE_quarter_equals_point_two_five_l495_49530

theorem quarter_equals_point_two_five : (1 : ℚ) / 4 = 0.250000000 := by
  sorry

end NUMINAMATH_CALUDE_quarter_equals_point_two_five_l495_49530


namespace NUMINAMATH_CALUDE_even_perfect_square_divisible_by_eight_l495_49579

theorem even_perfect_square_divisible_by_eight (b n : ℕ) : 
  b > 0 → 
  Even b → 
  n > 1 → 
  ∃ k : ℕ, (b^n - 1) / (b - 1) = k^2 → 
  8 ∣ b :=
sorry

end NUMINAMATH_CALUDE_even_perfect_square_divisible_by_eight_l495_49579


namespace NUMINAMATH_CALUDE_range_of_t_t_value_for_diameter_6_l495_49573

-- Define the equation of the circle
def circle_equation (x y t : ℝ) : Prop :=
  x^2 + y^2 + (Real.sqrt 3 * t + 1) * x + t * y + t^2 - 2 = 0

-- Theorem for the range of t
theorem range_of_t (t : ℝ) :
  (∃ x y, circle_equation x y t) → t > -3 * Real.sqrt 3 / 2 :=
sorry

-- Theorem for the value of t when diameter is 6
theorem t_value_for_diameter_6 (t : ℝ) :
  (∃ x y, circle_equation x y t) →
  (∃ x₁ y₁ x₂ y₂, circle_equation x₁ y₁ t ∧ circle_equation x₂ y₂ t ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 6) →
  t = 9 * Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_t_t_value_for_diameter_6_l495_49573


namespace NUMINAMATH_CALUDE_j_range_l495_49589

def h (x : ℝ) : ℝ := 2 * x + 1

def j (x : ℝ) : ℝ := h (h (h (h (h x))))

theorem j_range :
  ∀ y ∈ Set.range j,
  -1 ≤ y ∧ y ≤ 127 ∧
  ∃ x : ℝ, -1 ≤ x ∧ x ≤ 3 ∧ j x = y :=
by sorry

end NUMINAMATH_CALUDE_j_range_l495_49589


namespace NUMINAMATH_CALUDE_shadow_length_proportion_l495_49534

/-- Represents a pot with its height and shadow length -/
structure Pot where
  height : ℝ
  shadowLength : ℝ

/-- Theorem stating the relationship between pot heights and shadow lengths -/
theorem shadow_length_proportion (pot1 pot2 : Pot)
  (h1 : pot1.height = 20)
  (h2 : pot1.shadowLength = 10)
  (h3 : pot2.height = 40)
  (h4 : pot2.shadowLength = 20)
  (h5 : pot2.height = 2 * pot1.height)
  (h6 : pot2.shadowLength = 2 * pot1.shadowLength) :
  pot1.shadowLength = pot2.shadowLength / 2 := by
  sorry

end NUMINAMATH_CALUDE_shadow_length_proportion_l495_49534


namespace NUMINAMATH_CALUDE_log_3_bounds_l495_49583

theorem log_3_bounds :
  2/5 < Real.log 3 / Real.log 10 ∧ Real.log 3 / Real.log 10 < 1/2 := by
  have h1 : (3 : ℝ)^5 = 243 := by norm_num
  have h2 : (3 : ℝ)^6 = 729 := by norm_num
  have h3 : (2 : ℝ)^8 = 256 := by norm_num
  have h4 : (2 : ℝ)^10 = 1024 := by norm_num
  have h5 : (10 : ℝ)^2 = 100 := by norm_num
  have h6 : (10 : ℝ)^3 = 1000 := by norm_num
  sorry

end NUMINAMATH_CALUDE_log_3_bounds_l495_49583


namespace NUMINAMATH_CALUDE_water_flow_restrictor_problem_l495_49556

/-- Proves that given a reduced flow rate of 2 gallons per minute, which is 1 gallon per minute less than 0.6 times the original flow rate, the original flow rate is 5 gallons per minute. -/
theorem water_flow_restrictor_problem (original_rate : ℝ) : 
  (2 : ℝ) = 0.6 * original_rate - 1 → original_rate = 5 := by
  sorry

end NUMINAMATH_CALUDE_water_flow_restrictor_problem_l495_49556


namespace NUMINAMATH_CALUDE_unique_three_digit_divisible_by_11_l495_49585

theorem unique_three_digit_divisible_by_11 : ∃! n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧  -- three-digit number
  n % 10 = 3 ∧          -- units digit is 3
  n / 100 = 6 ∧         -- hundreds digit is 6
  n % 11 = 0 ∧          -- divisible by 11
  n = 693               -- the number is 693
  := by sorry

end NUMINAMATH_CALUDE_unique_three_digit_divisible_by_11_l495_49585


namespace NUMINAMATH_CALUDE_max_nSn_l495_49587

/-- An arithmetic sequence with sum S_n of first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  sum : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, sum n = n * (2 * a 1 + (n - 1) * d) / 2

/-- The problem statement -/
theorem max_nSn (seq : ArithmeticSequence) 
  (h1 : seq.sum 6 = 26)
  (h2 : seq.a 7 = 2) :
  ∃ m : ℚ, m = 338 ∧ ∀ n : ℕ, n * seq.sum n ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_nSn_l495_49587


namespace NUMINAMATH_CALUDE_problem_solution_l495_49504

theorem problem_solution (a b : ℚ) 
  (h1 : 2020 * a + 2024 * b = 2030) 
  (h2 : 2022 * a + 2026 * b = 2032) : 
  a - b = -4 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l495_49504


namespace NUMINAMATH_CALUDE_women_who_bought_apples_l495_49501

/-- The number of women who bought apples -/
def num_women : ℕ := 3

/-- The number of men who bought apples -/
def num_men : ℕ := 2

/-- The number of apples each man bought -/
def apples_per_man : ℕ := 30

/-- The additional number of apples each woman bought compared to each man -/
def additional_apples_per_woman : ℕ := 20

/-- The total number of apples bought -/
def total_apples : ℕ := 210

theorem women_who_bought_apples :
  num_women * (apples_per_man + additional_apples_per_woman) +
  num_men * apples_per_man = total_apples :=
by sorry

end NUMINAMATH_CALUDE_women_who_bought_apples_l495_49501


namespace NUMINAMATH_CALUDE_negation_of_exists_le_zero_is_forall_gt_zero_l495_49565

theorem negation_of_exists_le_zero_is_forall_gt_zero :
  (¬ ∃ x : ℝ, (2 : ℝ) ^ x ≤ 0) ↔ (∀ x : ℝ, (2 : ℝ) ^ x > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_exists_le_zero_is_forall_gt_zero_l495_49565


namespace NUMINAMATH_CALUDE_composite_product_properties_l495_49550

def first_five_composites : List Nat := [4, 6, 8, 9, 10]

def product_of_composites : Nat := first_five_composites.prod

theorem composite_product_properties :
  (product_of_composites % 10 = 0) ∧
  (Nat.digits 10 product_of_composites).sum = 18 := by
  sorry

end NUMINAMATH_CALUDE_composite_product_properties_l495_49550


namespace NUMINAMATH_CALUDE_quadratic_equation_identity_l495_49590

theorem quadratic_equation_identity 
  (a₀ a₁ a₂ r s x : ℝ) 
  (h₁ : a₂ ≠ 0) 
  (h₂ : a₀ ≠ 0) 
  (h₃ : a₀ + a₁ * r + a₂ * r^2 = 0) 
  (h₄ : a₀ + a₁ * s + a₂ * s^2 = 0) :
  a₀ + a₁ * x + a₂ * x^2 = a₀ * (1 - x / r) * (1 - x / s) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_identity_l495_49590


namespace NUMINAMATH_CALUDE_min_sum_squares_l495_49567

theorem min_sum_squares (a b c : ℕ+) (h : a.val^2 + b.val^2 - c.val = 2022) :
  (∀ a' b' c' : ℕ+, a'.val^2 + b'.val^2 - c'.val = 2022 →
    a.val^2 + b.val^2 + c.val^2 ≤ a'.val^2 + b'.val^2 + c'.val^2) ∧
  a.val^2 + b.val^2 + c.val^2 = 2034 ∧
  a.val = 27 ∧ b.val = 36 ∧ c.val = 3 := by
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l495_49567


namespace NUMINAMATH_CALUDE_soda_discount_percentage_l495_49533

/-- Proves that the discount percentage is 15% given the regular price and discounted price for soda cans. -/
theorem soda_discount_percentage 
  (regular_price : ℝ) 
  (discounted_price : ℝ) 
  (can_count : ℕ) :
  regular_price = 0.30 →
  discounted_price = 18.36 →
  can_count = 72 →
  (1 - discounted_price / (regular_price * can_count)) * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_soda_discount_percentage_l495_49533


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l495_49558

theorem triangle_angle_calculation (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (A + B + C = π) →
  (a * Real.cos B - b * Real.cos A = b) →
  (C = π / 5) →
  (B = 4 * π / 15) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l495_49558


namespace NUMINAMATH_CALUDE_square_side_length_l495_49532

theorem square_side_length (square_area rectangle_area : ℝ) 
  (rectangle_width rectangle_length : ℝ) (h1 : rectangle_width = 4) 
  (h2 : rectangle_length = 4) (h3 : square_area = rectangle_area) 
  (h4 : rectangle_area = rectangle_width * rectangle_length) : 
  ∃ (side_length : ℝ), side_length * side_length = square_area ∧ side_length = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l495_49532


namespace NUMINAMATH_CALUDE_range_of_m_l495_49526

-- Define the conditions
def p (x : ℝ) : Prop := -x^2 + 7*x + 8 ≥ 0
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - 4*m^2 ≤ 0

-- Define the main theorem
theorem range_of_m :
  (∀ x m : ℝ, (¬(p x) → ¬(q x m)) ∧ (∃ x : ℝ, q x m ∧ p x)) →
  (∀ m : ℝ, -1 ≤ m ∧ m ≤ 1) ∧ (∃ m : ℝ, m = -1 ∨ m = 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l495_49526


namespace NUMINAMATH_CALUDE_problem_statement_l495_49571

theorem problem_statement (a b c : ℝ) 
  (ha : a < 0) 
  (hb : b < 0) 
  (hab : abs a < abs b) 
  (hbc : abs b < abs c) : 
  (abs (a * b) < abs (b * c)) ∧ 
  (a * c < abs (b * c)) ∧ 
  (abs (a + b) < abs (b + c)) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l495_49571


namespace NUMINAMATH_CALUDE_sum_of_multiples_of_6_and_9_is_multiple_of_3_l495_49557

theorem sum_of_multiples_of_6_and_9_is_multiple_of_3 (x y : ℤ) 
  (hx : ∃ m : ℤ, x = 6 * m) 
  (hy : ∃ n : ℤ, y = 9 * n) : 
  ∃ k : ℤ, x + y = 3 * k := by
sorry

end NUMINAMATH_CALUDE_sum_of_multiples_of_6_and_9_is_multiple_of_3_l495_49557


namespace NUMINAMATH_CALUDE_library_books_count_l495_49569

theorem library_books_count (children_percentage : ℝ) (adult_count : ℕ) : 
  children_percentage = 35 →
  adult_count = 104 →
  ∃ (total : ℕ), (total : ℝ) * (1 - children_percentage / 100) = adult_count ∧ total = 160 :=
by sorry

end NUMINAMATH_CALUDE_library_books_count_l495_49569


namespace NUMINAMATH_CALUDE_fifteen_members_without_A_l495_49519

/-- Represents the number of club members who did not receive an A in either activity. -/
def members_without_A (total_members art_A science_A both_A : ℕ) : ℕ :=
  total_members - (art_A + science_A - both_A)

/-- Theorem stating that 15 club members did not receive an A in either activity. -/
theorem fifteen_members_without_A :
  members_without_A 50 20 30 15 = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_members_without_A_l495_49519


namespace NUMINAMATH_CALUDE_circle_circumference_l495_49574

/-- Given a circle with area 1800 cm² and ratio of area to circumference 15, 
    prove that its circumference is 120 cm. -/
theorem circle_circumference (A : ℝ) (r : ℝ) :
  A = 1800 →
  A / (2 * Real.pi * r) = 15 →
  2 * Real.pi * r = 120 :=
by sorry

end NUMINAMATH_CALUDE_circle_circumference_l495_49574


namespace NUMINAMATH_CALUDE_max_rectangle_area_l495_49529

/-- The maximum area of a rectangle with integer dimensions and perimeter 34 cm is 72 square cm. -/
theorem max_rectangle_area : ∀ l w : ℕ, 
  2 * l + 2 * w = 34 → 
  l * w ≤ 72 :=
by
  sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l495_49529


namespace NUMINAMATH_CALUDE_sequence_properties_l495_49570

/-- Arithmetic sequence with a₈ = 6 and a₁₀ = 0 -/
def arithmetic_sequence (n : ℕ) : ℚ :=
  30 - 3 * n

/-- Geometric sequence with a₁ = 1/2 and a₄ = 4 -/
def geometric_sequence (n : ℕ) : ℚ :=
  2^(n - 2)

/-- Sum of the first n terms of the geometric sequence -/
def geometric_sum (n : ℕ) : ℚ :=
  2^(n - 1) - 1/2

theorem sequence_properties :
  (arithmetic_sequence 8 = 6 ∧ arithmetic_sequence 10 = 0) ∧
  (geometric_sequence 1 = 1/2 ∧ geometric_sequence 4 = 4) ∧
  (∀ n : ℕ, geometric_sum n = (geometric_sequence 1) * (1 - (2^n)) / (1 - 2)) := by
  sorry

#check sequence_properties

end NUMINAMATH_CALUDE_sequence_properties_l495_49570


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l495_49578

theorem least_addition_for_divisibility :
  ∃ (n : ℕ), n = 8 ∧
  (∀ (m : ℕ), m < n → ¬((821562 + m) % 5 = 0 ∧ (821562 + m) % 13 = 0)) ∧
  (821562 + n) % 5 = 0 ∧ (821562 + n) % 13 = 0 := by
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l495_49578


namespace NUMINAMATH_CALUDE_fraction_simplification_l495_49568

theorem fraction_simplification : (1 : ℚ) / (2 + 2/3) = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l495_49568


namespace NUMINAMATH_CALUDE_magnitude_comparison_l495_49507

theorem magnitude_comparison : 7^(0.3 : ℝ) > (0.3 : ℝ)^7 ∧ (0.3 : ℝ)^7 > Real.log 0.3 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_comparison_l495_49507


namespace NUMINAMATH_CALUDE_message_spread_time_l495_49528

/-- The number of people who have received the message after n minutes -/
def people_reached (n : ℕ) : ℕ := 2^(n + 1) - 1

/-- The time required for the message to reach 2047 people -/
def time_to_reach_2047 : ℕ := 10

theorem message_spread_time :
  people_reached time_to_reach_2047 = 2047 :=
sorry

end NUMINAMATH_CALUDE_message_spread_time_l495_49528


namespace NUMINAMATH_CALUDE_regression_change_l495_49597

/-- Represents a simple linear regression equation -/
structure LinearRegression where
  intercept : ℝ
  slope : ℝ

/-- The change in the dependent variable when the independent variable increases by one unit -/
def change_in_y (reg : LinearRegression) : ℝ :=
  reg.intercept - reg.slope * (reg.intercept + 1) - (reg.intercept - reg.slope * reg.intercept)

theorem regression_change (reg : LinearRegression) 
  (h : reg.intercept = 2 ∧ reg.slope = 3) : 
  change_in_y reg = -3 := by
  sorry

#eval change_in_y { intercept := 2, slope := 3 }

end NUMINAMATH_CALUDE_regression_change_l495_49597


namespace NUMINAMATH_CALUDE_empty_set_subset_subset_transitive_l495_49560

-- Define the empty set
def emptySet : Set α := ∅

-- Define subset relation
def isSubset (A B : Set α) : Prop := ∀ x, x ∈ A → x ∈ B

-- Theorem 1: The empty set is a subset of any set
theorem empty_set_subset (S : Set α) : isSubset emptySet S := by sorry

-- Theorem 2: Transitivity of subset relation
theorem subset_transitive (A B C : Set α) 
  (h1 : isSubset A B) (h2 : isSubset B C) : isSubset A C := by sorry

end NUMINAMATH_CALUDE_empty_set_subset_subset_transitive_l495_49560


namespace NUMINAMATH_CALUDE_barry_vitamin_d3_serving_size_l495_49542

/-- Calculates the daily serving size of capsules given the total number of days,
    capsules per bottle, and number of bottles. -/
def daily_serving_size (days : ℕ) (capsules_per_bottle : ℕ) (bottles : ℕ) : ℕ :=
  (capsules_per_bottle * bottles) / days

theorem barry_vitamin_d3_serving_size :
  let days : ℕ := 180
  let capsules_per_bottle : ℕ := 60
  let bottles : ℕ := 6
  daily_serving_size days capsules_per_bottle bottles = 2 := by
  sorry

end NUMINAMATH_CALUDE_barry_vitamin_d3_serving_size_l495_49542


namespace NUMINAMATH_CALUDE_items_can_fit_in_containers_l495_49577

/-- Represents an item with a weight -/
structure Item where
  weight : ℝ
  weight_bound : weight ≤ 1/2

/-- Represents a set of items -/
def ItemSet := List Item

/-- Calculate the total weight of a set of items -/
def totalWeight (items : ItemSet) : ℝ :=
  items.foldl (fun acc item => acc + item.weight) 0

/-- Theorem: Given a set of items, each weighing at most 1/2 unit, 
    with a total weight W > 1/3, these items can be placed into 
    ⌈(3W - 1)/2⌉ or fewer containers, each with a capacity of 1 unit. -/
theorem items_can_fit_in_containers (items : ItemSet) 
    (h_total_weight : totalWeight items > 1/3) :
    ∃ (num_containers : ℕ), 
      num_containers ≤ Int.ceil ((3 * totalWeight items - 1) / 2) ∧ 
      (∃ (partition : List (List Item)), 
        partition.length = num_containers ∧
        partition.all (fun container => totalWeight container ≤ 1) ∧
        partition.join = items) := by
  sorry

end NUMINAMATH_CALUDE_items_can_fit_in_containers_l495_49577


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l495_49594

theorem arithmetic_mean_of_fractions : 
  let a := 3 / 8
  let b := 5 / 9
  (a + b) / 2 = 67 / 144 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l495_49594


namespace NUMINAMATH_CALUDE_cn_tower_height_is_553_l495_49546

/-- The height of the Space Needle in meters -/
def space_needle_height : ℕ := 184

/-- The difference in height between the CN Tower and the Space Needle in meters -/
def height_difference : ℕ := 369

/-- The height of the CN Tower in meters -/
def cn_tower_height : ℕ := space_needle_height + height_difference

theorem cn_tower_height_is_553 : cn_tower_height = 553 := by
  sorry

end NUMINAMATH_CALUDE_cn_tower_height_is_553_l495_49546


namespace NUMINAMATH_CALUDE_fourth_root_is_negative_seven_l495_49524

/-- Represents a polynomial of degree 4 with rational coefficients -/
structure QuarticPolynomial where
  d : ℚ
  e : ℚ
  f : ℚ

/-- Checks if a given number is a root of the polynomial -/
def isRoot (p : QuarticPolynomial) (x : ℝ) : Prop :=
  x^4 + p.d * x^2 + p.e * x + p.f = 0

theorem fourth_root_is_negative_seven
  (p : QuarticPolynomial)
  (h1 : isRoot p (3 - Real.sqrt 5))
  (h2 : ∃ (a b : ℤ), isRoot p a ∧ isRoot p b) :
  isRoot p (-7) :=
sorry

end NUMINAMATH_CALUDE_fourth_root_is_negative_seven_l495_49524


namespace NUMINAMATH_CALUDE_max_stamps_purchasable_l495_49513

/-- Given a stamp price of 25 cents and a budget of 5000 cents,
    the maximum number of stamps that can be purchased is 200. -/
theorem max_stamps_purchasable (stamp_price : ℕ) (budget : ℕ) (h1 : stamp_price = 25) (h2 : budget = 5000) :
  ∃ (n : ℕ), n = 200 ∧ n * stamp_price ≤ budget ∧ ∀ m : ℕ, m * stamp_price ≤ budget → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_max_stamps_purchasable_l495_49513


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l495_49539

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ r < 1 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- Define the theorem
theorem geometric_sequence_ratio
  (a : ℕ → ℝ)
  (h_geom : is_positive_geometric_sequence a)
  (h_decreasing : ∀ n : ℕ, a (n + 1) < a n)
  (h_product : a 2 * a 8 = 6)
  (h_sum : a 4 + a 6 = 5) :
  a 5 / a 7 = 3/2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l495_49539


namespace NUMINAMATH_CALUDE_part_one_part_two_l495_49545

-- Define the sets A and B
def A (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 4 < 0}

-- Define the specific set A as given in the problem
def A_specific : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}

-- Theorem for part (1)
theorem part_one (a b : ℝ) (h : A a b = A_specific) : a + b = -7 := by
  sorry

-- Theorem for part (2)
theorem part_two (m : ℝ) (h : A (-3) (-4) = A_specific) :
  (∀ x, x ∈ A (-3) (-4) → x ∉ B m) → m ≤ -3 ∨ m ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l495_49545


namespace NUMINAMATH_CALUDE_linear_function_value_l495_49538

/-- A linear function in three variables -/
def LinearFunction (f : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∀ a b c x y z : ℝ, f (a + x) (b + y) (c + z) = f a b c + f x y z

theorem linear_function_value (f : ℝ → ℝ → ℝ → ℝ) 
  (h_linear : LinearFunction f)
  (h_value_3 : f 3 3 3 = 1 / (3 * 3 * 3))
  (h_value_4 : f 4 4 4 = 1 / (4 * 4 * 4)) :
  f 5 5 5 = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_value_l495_49538


namespace NUMINAMATH_CALUDE_number_multiplication_l495_49518

theorem number_multiplication (x : ℤ) : x - 27 = 46 → x * 46 = 3358 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplication_l495_49518


namespace NUMINAMATH_CALUDE_binomial_60_3_l495_49554

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_60_3_l495_49554


namespace NUMINAMATH_CALUDE_work_completion_time_l495_49508

-- Define the efficiency of worker B
def B_efficiency : ℚ := 1 / 24

-- Define the efficiency of worker A (twice as efficient as B)
def A_efficiency : ℚ := 2 * B_efficiency

-- Define the combined efficiency of A and B
def combined_efficiency : ℚ := A_efficiency + B_efficiency

-- Theorem: A and B together can complete the work in 8 days
theorem work_completion_time : (1 : ℚ) / combined_efficiency = 8 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l495_49508


namespace NUMINAMATH_CALUDE_candidate_a_votes_l495_49535

theorem candidate_a_votes (total_votes : ℕ) (invalid_percent : ℚ) (candidate_a_percent : ℚ) : 
  total_votes = 560000 →
  invalid_percent = 15 / 100 →
  candidate_a_percent = 60 / 100 →
  ⌊(1 - invalid_percent) * candidate_a_percent * total_votes⌋ = 285600 :=
by sorry

end NUMINAMATH_CALUDE_candidate_a_votes_l495_49535


namespace NUMINAMATH_CALUDE_additional_concession_percentage_l495_49512

def original_price : ℝ := 2000
def standard_concession : ℝ := 30
def final_price : ℝ := 1120

theorem additional_concession_percentage :
  ∃ (additional_concession : ℝ),
    (original_price * (1 - standard_concession / 100) * (1 - additional_concession / 100) = final_price) ∧
    additional_concession = 20 := by
  sorry

end NUMINAMATH_CALUDE_additional_concession_percentage_l495_49512


namespace NUMINAMATH_CALUDE_fourth_number_in_sequence_l495_49514

def fibonacci_like_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n ≥ 3, a n = a (n - 1) + a (n - 2)

theorem fourth_number_in_sequence 
  (a : ℕ → ℕ) 
  (h_seq : fibonacci_like_sequence a) 
  (h_7 : a 7 = 42) 
  (h_9 : a 9 = 110) : 
  a 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_fourth_number_in_sequence_l495_49514


namespace NUMINAMATH_CALUDE_correct_problems_l495_49596

theorem correct_problems (total : ℕ) (h1 : total = 54) : ∃ (correct : ℕ), 
  correct + 2 * correct = total ∧ correct = 18 := by
  sorry

end NUMINAMATH_CALUDE_correct_problems_l495_49596


namespace NUMINAMATH_CALUDE_a_worked_days_proof_l495_49543

/-- The number of days A needs to complete the entire work alone -/
def a_complete_days : ℝ := 40

/-- The number of days B needs to complete the entire work alone -/
def b_complete_days : ℝ := 60

/-- The number of days B needs to complete the remaining work after A leaves -/
def b_remaining_days : ℝ := 45

/-- The number of days A worked before leaving -/
def a_worked_days : ℝ := 10

theorem a_worked_days_proof :
  (1 / a_complete_days * a_worked_days) + (b_remaining_days / b_complete_days) = 1 :=
sorry

end NUMINAMATH_CALUDE_a_worked_days_proof_l495_49543


namespace NUMINAMATH_CALUDE_probability_other_note_counterfeit_l495_49522

/-- Represents the total number of banknotes -/
def total_notes : ℕ := 20

/-- Represents the number of counterfeit notes -/
def counterfeit_notes : ℕ := 5

/-- Represents the number of genuine notes -/
def genuine_notes : ℕ := total_notes - counterfeit_notes

/-- Calculates the probability that both drawn notes are counterfeit -/
def prob_both_counterfeit : ℚ :=
  (counterfeit_notes.choose 2 : ℚ) / (total_notes.choose 2 : ℚ)

/-- Calculates the probability that at least one drawn note is counterfeit -/
def prob_at_least_one_counterfeit : ℚ :=
  ((counterfeit_notes.choose 2 + counterfeit_notes * genuine_notes) : ℚ) / (total_notes.choose 2 : ℚ)

/-- The main theorem to be proved -/
theorem probability_other_note_counterfeit :
  prob_both_counterfeit / prob_at_least_one_counterfeit = 2 / 17 := by
  sorry

end NUMINAMATH_CALUDE_probability_other_note_counterfeit_l495_49522


namespace NUMINAMATH_CALUDE_f_properties_l495_49588

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log x / Real.log 2

theorem f_properties :
  (∀ x > 0, f (-x) ≠ -f x ∧ f (-x) ≠ f x) ∧
  (∀ x y, 0 < x ∧ x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l495_49588


namespace NUMINAMATH_CALUDE_card_game_draw_probability_l495_49572

theorem card_game_draw_probability (ben_win : ℚ) (sara_win : ℚ) (h1 : ben_win = 5 / 12) (h2 : sara_win = 1 / 4) :
  1 - (ben_win + sara_win) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_card_game_draw_probability_l495_49572


namespace NUMINAMATH_CALUDE_unique_number_exists_l495_49592

theorem unique_number_exists : ∃! x : ℕ, (∃ k : ℕ, 3 * x = 9 * k) ∧ 4 * x = 108 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_exists_l495_49592


namespace NUMINAMATH_CALUDE_train_speed_l495_49598

/-- Given a train of length 360 m passing a platform of length 130 m in 39.2 seconds,
    prove that the speed of the train is 45 km/hr. -/
theorem train_speed (train_length platform_length time_to_pass : ℝ) 
  (h1 : train_length = 360)
  (h2 : platform_length = 130)
  (h3 : time_to_pass = 39.2) : 
  (train_length + platform_length) / time_to_pass * 3.6 = 45 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l495_49598


namespace NUMINAMATH_CALUDE_determinant_of_cubic_roots_l495_49500

/-- Given a, b, c are roots of x^3 - 2px + q = 0, 
    prove that the determinant of the matrix is 5 - 6p + q -/
theorem determinant_of_cubic_roots (p q a b c : ℝ) : 
  a^3 - 2*p*a + q = 0 → 
  b^3 - 2*p*b + q = 0 → 
  c^3 - 2*p*c + q = 0 → 
  Matrix.det !![2 + a, 1, 1; 1, 2 + b, 1; 1, 1, 2 + c] = 5 - 6*p + q := by
  sorry

end NUMINAMATH_CALUDE_determinant_of_cubic_roots_l495_49500


namespace NUMINAMATH_CALUDE_candy_original_pencils_l495_49511

/-- The number of pencils each person has -/
structure PencilCounts where
  calen : ℕ
  caleb : ℕ
  candy : ℕ
  darlene : ℕ

/-- The conditions of the problem -/
def pencil_problem (p : PencilCounts) : Prop :=
  p.calen = p.caleb + 5 ∧
  p.caleb = 2 * p.candy - 3 ∧
  p.darlene = p.calen + p.caleb + p.candy + 4 ∧
  p.calen - 10 = 10

theorem candy_original_pencils (p : PencilCounts) : 
  pencil_problem p → p.candy = 9 := by
  sorry

end NUMINAMATH_CALUDE_candy_original_pencils_l495_49511


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l495_49561

theorem ellipse_eccentricity (m : ℝ) : 
  (∀ x y : ℝ, x^2/m + y^2/4 = 1) →  -- Equation of the ellipse
  (∃ a b c : ℝ, a > b ∧ b > 0 ∧ c^2 = a^2 - b^2 ∧ 2*c = 2) →  -- Eccentricity is 2
  (m = 3 ∨ m = 5) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l495_49561


namespace NUMINAMATH_CALUDE_max_weight_proof_l495_49505

def max_weight_single_trip : ℕ := 8750

theorem max_weight_proof (crate_weight_min crate_weight_max : ℕ) 
  (weight_8_crates weight_12_crates : ℕ) :
  crate_weight_min = 150 →
  crate_weight_max = 250 →
  weight_8_crates ≤ 1300 →
  weight_12_crates ≤ 2100 →
  max_weight_single_trip = 8750 := by
  sorry

end NUMINAMATH_CALUDE_max_weight_proof_l495_49505


namespace NUMINAMATH_CALUDE_number_equation_solution_l495_49547

theorem number_equation_solution : ∃ x : ℝ, 35 + 3 * x = 50 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l495_49547


namespace NUMINAMATH_CALUDE_lea_purchases_cost_l495_49555

/-- The cost of Léa's purchases -/
def total_cost (book_price : ℕ) (binder_price : ℕ) (notebook_price : ℕ) 
  (num_binders : ℕ) (num_notebooks : ℕ) : ℕ :=
  book_price + (binder_price * num_binders) + (notebook_price * num_notebooks)

/-- Theorem stating that the total cost of Léa's purchases is $28 -/
theorem lea_purchases_cost :
  total_cost 16 2 1 3 6 = 28 := by
  sorry

end NUMINAMATH_CALUDE_lea_purchases_cost_l495_49555


namespace NUMINAMATH_CALUDE_factorization_x2_4xy_4y2_l495_49536

/-- Factorization of a polynomial x^2 - 4xy + 4y^2 --/
theorem factorization_x2_4xy_4y2 (x y : ℝ) :
  x^2 - 4*x*y + 4*y^2 = (x - 2*y)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_x2_4xy_4y2_l495_49536


namespace NUMINAMATH_CALUDE_projection_of_v_onto_u_l495_49520

def v : Fin 2 → ℚ := ![5, 7]
def u : Fin 2 → ℚ := ![1, -3]

def projection (v u : Fin 2 → ℚ) : Fin 2 → ℚ :=
  let dot_product := (v 0) * (u 0) + (v 1) * (u 1)
  let magnitude_squared := (u 0)^2 + (u 1)^2
  let scalar := dot_product / magnitude_squared
  ![scalar * (u 0), scalar * (u 1)]

theorem projection_of_v_onto_u :
  projection v u = ![-8/5, 24/5] := by sorry

end NUMINAMATH_CALUDE_projection_of_v_onto_u_l495_49520


namespace NUMINAMATH_CALUDE_b_investment_value_l495_49591

/-- Represents the investment and profit distribution in a partnership business --/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  c_profit_share : ℕ

/-- Theorem stating that given the conditions, B's investment is 32,000 --/
theorem b_investment_value (p : Partnership) 
  (h1 : p.a_investment = 24000)
  (h2 : p.c_investment = 36000)
  (h3 : p.total_profit = 92000)
  (h4 : p.c_profit_share = 36000)
  (h5 : p.c_investment / p.c_profit_share = (p.a_investment + p.b_investment + p.c_investment) / p.total_profit) : 
  p.b_investment = 32000 := by
  sorry

#check b_investment_value

end NUMINAMATH_CALUDE_b_investment_value_l495_49591


namespace NUMINAMATH_CALUDE_fraction_problem_l495_49582

theorem fraction_problem (x : ℚ) : 
  (3/4 : ℚ) * x * (2/5 : ℚ) * 5100 = 765.0000000000001 → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l495_49582


namespace NUMINAMATH_CALUDE_negation_of_proposition_l495_49599

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 - 2*x + 1 ≥ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - 2*x + 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l495_49599


namespace NUMINAMATH_CALUDE_shopkeeper_loss_percent_l495_49527

theorem shopkeeper_loss_percent 
  (initial_value : ℝ) 
  (profit_rate : ℝ) 
  (theft_rate : ℝ) 
  (profit_rate_is_10_percent : profit_rate = 0.1)
  (theft_rate_is_30_percent : theft_rate = 0.3)
  (initial_value_positive : initial_value > 0) :
  let remaining_value := initial_value * (1 - theft_rate)
  let selling_price := remaining_value * (1 + profit_rate)
  let loss := initial_value - selling_price
  let loss_percent := (loss / initial_value) * 100
  loss_percent = 23 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_loss_percent_l495_49527


namespace NUMINAMATH_CALUDE_total_dresses_l495_49576

theorem total_dresses (emily_dresses : ℕ) (melissa_dresses : ℕ) (debora_dresses : ℕ) : 
  emily_dresses = 16 →
  melissa_dresses = emily_dresses / 2 →
  debora_dresses = melissa_dresses + 12 →
  emily_dresses + melissa_dresses + debora_dresses = 44 := by
sorry

end NUMINAMATH_CALUDE_total_dresses_l495_49576


namespace NUMINAMATH_CALUDE_reeya_average_is_67_l495_49531

def reeya_scores : List ℝ := [55, 67, 76, 82, 55]

theorem reeya_average_is_67 : 
  (reeya_scores.sum / reeya_scores.length : ℝ) = 67 := by
  sorry

end NUMINAMATH_CALUDE_reeya_average_is_67_l495_49531


namespace NUMINAMATH_CALUDE_factorial_inequality_l495_49506

theorem factorial_inequality (n : ℕ) (h : n ≥ 2) :
  2 * Real.log (Nat.factorial n) > (n^2 - 2*n + 1) / n := by
  sorry

end NUMINAMATH_CALUDE_factorial_inequality_l495_49506


namespace NUMINAMATH_CALUDE_smallest_integer_greater_than_sqrt_three_l495_49586

theorem smallest_integer_greater_than_sqrt_three : 
  ∀ n : ℤ, n > Real.sqrt 3 → n ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_greater_than_sqrt_three_l495_49586


namespace NUMINAMATH_CALUDE_max_triangles_six_lines_l495_49509

/-- A configuration of lines on a plane -/
structure LineConfiguration where
  num_lines : ℕ
  on_plane : Bool

/-- Counts the number of equilateral triangles formed by line intersections -/
def count_equilateral_triangles (config : LineConfiguration) : ℕ :=
  sorry

/-- The maximum number of equilateral triangles for a given configuration -/
def max_equilateral_triangles (config : LineConfiguration) : ℕ :=
  sorry

/-- Theorem: The maximum number of equilateral triangles formed by six lines on a plane is 8 -/
theorem max_triangles_six_lines :
  ∀ (config : LineConfiguration),
    config.num_lines = 6 ∧ config.on_plane →
    max_equilateral_triangles config = 8 :=
by sorry

end NUMINAMATH_CALUDE_max_triangles_six_lines_l495_49509


namespace NUMINAMATH_CALUDE_cupcake_frosting_l495_49540

def cagney_rate : ℚ := 1 / 15
def lacey_rate : ℚ := 1 / 25
def lacey_delay : ℕ := 30
def total_time : ℕ := 600

def total_cupcakes : ℕ := 62

theorem cupcake_frosting :
  (cagney_rate * total_time).floor +
  (lacey_rate * (total_time - lacey_delay)).floor = total_cupcakes :=
sorry

end NUMINAMATH_CALUDE_cupcake_frosting_l495_49540


namespace NUMINAMATH_CALUDE_quadratic_monotonicity_l495_49584

theorem quadratic_monotonicity (a b c : ℝ) (h_a : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  (∀ x y, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x < y → f x < f y) →
  (f 1 < f 5) ∧
  ¬ ((f 1 < f 5) → (∀ x y, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x < y → f x < f y)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_monotonicity_l495_49584


namespace NUMINAMATH_CALUDE_product_of_solutions_l495_49549

theorem product_of_solutions (x : ℝ) : 
  (|18 / x + 4| = 3) → (∃ y : ℝ, (|18 / y + 4| = 3) ∧ x * y = 324 / 7) := by
sorry

end NUMINAMATH_CALUDE_product_of_solutions_l495_49549


namespace NUMINAMATH_CALUDE_smallest_integer_in_consecutive_even_set_l495_49551

theorem smallest_integer_in_consecutive_even_set (n : ℤ) : 
  n % 2 = 0 ∧ 
  (n + 8 < 3 * ((n + (n + 2) + (n + 4) + (n + 6) + (n + 8)) / 5)) →
  n = 0 ∧ ∀ m : ℤ, (m % 2 = 0 ∧ 
    m + 8 < 3 * ((m + (m + 2) + (m + 4) + (m + 6) + (m + 8)) / 5)) →
    m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_in_consecutive_even_set_l495_49551


namespace NUMINAMATH_CALUDE_time_per_bone_l495_49510

/-- Proves that analyzing 206 bones in 206 hours with equal time per bone results in 1 hour per bone -/
theorem time_per_bone (total_time : ℕ) (num_bones : ℕ) (time_per_bone : ℚ) :
  total_time = 206 →
  num_bones = 206 →
  time_per_bone = total_time / num_bones →
  time_per_bone = 1 := by
  sorry

#check time_per_bone

end NUMINAMATH_CALUDE_time_per_bone_l495_49510


namespace NUMINAMATH_CALUDE_solution_to_linear_equation_l495_49515

theorem solution_to_linear_equation (x y m : ℝ) : 
  x = 1 → y = 3 → x - 2 * y = m → m = -5 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_linear_equation_l495_49515


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l495_49575

theorem no_positive_integer_solutions 
  (p : ℕ) (hp : Nat.Prime p) (hp_mod : p % 4 = 3) (n : ℕ+) :
  ¬ ∃ (x y : ℕ+), p^(n : ℕ) = x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l495_49575


namespace NUMINAMATH_CALUDE_lcm_9_12_15_l495_49552

theorem lcm_9_12_15 : Nat.lcm 9 (Nat.lcm 12 15) = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_9_12_15_l495_49552


namespace NUMINAMATH_CALUDE_focaccia_price_is_four_l495_49564

/-- The price of a focaccia loaf given Sean's Sunday purchases -/
def focaccia_price : ℝ :=
  let almond_croissant : ℝ := 4.50
  let salami_cheese_croissant : ℝ := 4.50
  let plain_croissant : ℝ := 3.00
  let latte : ℝ := 2.50
  let total_spent : ℝ := 21.00
  total_spent - (almond_croissant + salami_cheese_croissant + plain_croissant + 2 * latte)

theorem focaccia_price_is_four : focaccia_price = 4.00 := by
  sorry

end NUMINAMATH_CALUDE_focaccia_price_is_four_l495_49564


namespace NUMINAMATH_CALUDE_expand_product_l495_49595

theorem expand_product (x : ℝ) : 5 * (x + 6) * (x^2 + 2*x + 3) = 5*x^3 + 40*x^2 + 75*x + 90 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l495_49595


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l495_49541

theorem quadratic_one_solution (a : ℝ) :
  (∃! x : ℝ, a * x^2 + 2 * x + 1 = 0) → a = 0 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l495_49541


namespace NUMINAMATH_CALUDE_range_of_f_l495_49521

-- Define the function f
def f (x : ℝ) : ℝ := (x^2 - 2)^2 + 1

-- State the theorem
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, x ≥ 0 ∧ f x = y) ↔ y ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l495_49521


namespace NUMINAMATH_CALUDE_martin_fruit_ratio_l495_49516

/-- Given that Martin has twice as many oranges as limes now, 50 oranges, and initially had 150 fruits,
    prove that the ratio of fruits eaten to initial fruits is 1/2 -/
theorem martin_fruit_ratio :
  ∀ (oranges_now limes_now fruits_initial : ℕ),
    oranges_now = 50 →
    fruits_initial = 150 →
    oranges_now = 2 * limes_now →
    (fruits_initial - (oranges_now + limes_now)) / fruits_initial = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_martin_fruit_ratio_l495_49516


namespace NUMINAMATH_CALUDE_problem_statement_l495_49544

theorem problem_statement (x y : ℝ) 
  (h1 : (4 : ℝ) ^ x = 16 ^ (y + 2))
  (h2 : (27 : ℝ) ^ y = 9 ^ (x - 8)) :
  x + y = 40 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l495_49544


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l495_49563

def point : ℝ × ℝ := (-3, -2)

def in_third_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 < 0

theorem point_in_third_quadrant : in_third_quadrant point := by
  sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l495_49563


namespace NUMINAMATH_CALUDE_some_number_value_l495_49566

theorem some_number_value (x : ℝ) : 60 + 5 * 12 / (x / 3) = 61 → x = 180 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l495_49566


namespace NUMINAMATH_CALUDE_leak_drops_per_minute_l495_49548

/-- Proves that the leak drips 3 drops per minute given the conditions -/
theorem leak_drops_per_minute 
  (drop_volume : ℝ) 
  (pot_capacity : ℝ) 
  (fill_time : ℝ) 
  (h1 : drop_volume = 20) 
  (h2 : pot_capacity = 3000) 
  (h3 : fill_time = 50) : 
  (pot_capacity / drop_volume) / fill_time = 3 := by
  sorry

#check leak_drops_per_minute

end NUMINAMATH_CALUDE_leak_drops_per_minute_l495_49548


namespace NUMINAMATH_CALUDE_unique_positive_integer_solution_l495_49559

theorem unique_positive_integer_solution :
  ∀ x y : ℕ+, 2 * x^2 + 5 * y^2 = 11 * (x * y - 11) → x = 14 ∧ y = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_positive_integer_solution_l495_49559


namespace NUMINAMATH_CALUDE_tangent_parabola_hyperbola_l495_49553

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = x^2 + 5

/-- The hyperbola equation -/
def hyperbola (m x y : ℝ) : Prop := y^2 - m * x^2 = 1

/-- Tangency condition -/
def are_tangent (m : ℝ) : Prop := ∃ (x y : ℝ), parabola x y ∧ hyperbola m x y ∧
  ∀ (x' y' : ℝ), parabola x' y' ∧ hyperbola m x' y' → (x' = x ∧ y' = y)

theorem tangent_parabola_hyperbola (m : ℝ) :
  are_tangent m ↔ (m = 10 + 2 * Real.sqrt 6 ∨ m = 10 - 2 * Real.sqrt 6) :=
sorry

end NUMINAMATH_CALUDE_tangent_parabola_hyperbola_l495_49553


namespace NUMINAMATH_CALUDE_biased_die_probability_l495_49523

theorem biased_die_probability (p : ℝ) (h1 : 0 ≤ p ∧ p ≤ 1) :
  (Nat.choose 8 6 : ℝ) * p^6 * (1-p)^2 = 125/256 → p^6 * (1-p)^2 = 125/7168 := by
  sorry

end NUMINAMATH_CALUDE_biased_die_probability_l495_49523


namespace NUMINAMATH_CALUDE_domain_of_g_l495_49593

-- Define the function f with domain [-1, 2]
def f : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

-- Define the function g(x) = f(2x+1)
def g (x : ℝ) : Prop := (2*x + 1) ∈ f

-- Theorem stating that the domain of g is [-1, 1/2]
theorem domain_of_g : 
  {x : ℝ | g x} = {x : ℝ | -1 ≤ x ∧ x ≤ 1/2} :=
by sorry

end NUMINAMATH_CALUDE_domain_of_g_l495_49593


namespace NUMINAMATH_CALUDE_quadratic_translation_l495_49562

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Applies a horizontal translation to a quadratic function -/
def horizontalTranslation (f : QuadraticFunction) (h : ℝ) : QuadraticFunction :=
  { a := f.a
  , b := 2 * f.a * h + f.b
  , c := f.a * h^2 + f.b * h + f.c }

/-- Applies a vertical translation to a quadratic function -/
def verticalTranslation (f : QuadraticFunction) (v : ℝ) : QuadraticFunction :=
  { a := f.a
  , b := f.b
  , c := f.c + v }

/-- The original quadratic function y = x^2 + 1 -/
def originalFunction : QuadraticFunction :=
  { a := 1, b := 0, c := 1 }

theorem quadratic_translation :
  let f := originalFunction
  let g := verticalTranslation (horizontalTranslation f (-2)) (-3)
  g = { a := 1, b := 4, c := 2 } := by sorry

end NUMINAMATH_CALUDE_quadratic_translation_l495_49562


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l495_49502

theorem matrix_equation_solution (A B : Matrix (Fin 2) (Fin 2) ℝ) :
  A * B = A - B →
  A * B = ![![7, -2], ![3, -1]] →
  B * A = ![![8, -2], ![3, 0]] := by sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l495_49502


namespace NUMINAMATH_CALUDE_justin_lost_flowers_l495_49581

/-- Calculates the number of lost flowers given the gathering time, average time per flower,
    number of classmates, and additional time needed. -/
def lostFlowers (gatheringTime minutes : ℕ) (avgTimePerFlower : ℕ) (classmates : ℕ) (additionalTime : ℕ) : ℕ :=
  let flowersFilled := gatheringTime / avgTimePerFlower
  let additionalFlowers := additionalTime / avgTimePerFlower
  flowersFilled + additionalFlowers - classmates

/-- Theorem stating that Justin has lost 3 flowers. -/
theorem justin_lost_flowers : 
  lostFlowers 120 10 30 210 = 3 := by
  sorry

end NUMINAMATH_CALUDE_justin_lost_flowers_l495_49581


namespace NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_plus_one_l495_49503

theorem imaginary_part_of_reciprocal_plus_one (z : ℂ) (x y : ℝ) 
  (h1 : z = x + y * I) 
  (h2 : z ≠ x) -- z is nonreal
  (h3 : Complex.abs z = 1) : 
  Complex.im (1 / (1 + z)) = -y / (2 * (1 + x)) :=
by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_plus_one_l495_49503


namespace NUMINAMATH_CALUDE_estate_value_l495_49537

-- Define the estate and its components
def estate : ℝ := sorry
def older_child_share : ℝ := sorry
def younger_child_share : ℝ := sorry
def wife_share : ℝ := sorry
def charity_share : ℝ := 800

-- Define the conditions
axiom children_share : older_child_share + younger_child_share = 0.6 * estate
axiom children_ratio : older_child_share = (3/2) * younger_child_share
axiom wife_share_relation : wife_share = 4 * older_child_share
axiom total_distribution : estate = older_child_share + younger_child_share + wife_share + charity_share

-- Theorem to prove
theorem estate_value : estate = 1923 := by sorry

end NUMINAMATH_CALUDE_estate_value_l495_49537


namespace NUMINAMATH_CALUDE_root_product_equality_l495_49517

theorem root_product_equality (p q : ℝ) (α β γ δ : ℂ) : 
  (α^2 + p*α + 1 = 0) → 
  (β^2 + p*β + 1 = 0) → 
  (γ^2 + q*γ + 1 = 0) → 
  (δ^2 + q*δ + 1 = 0) → 
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = q^2 - p^2 := by
  sorry

end NUMINAMATH_CALUDE_root_product_equality_l495_49517
