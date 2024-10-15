import Mathlib

namespace NUMINAMATH_CALUDE_function_inequality_equivalence_l1437_143743

-- Define the functions f and g with domain and range ℝ
variable (f g : ℝ → ℝ)

-- State the theorem
theorem function_inequality_equivalence :
  (∀ x, f x > g x) ↔ (∀ x, x ∉ {x | f x ≤ g x}) :=
sorry

end NUMINAMATH_CALUDE_function_inequality_equivalence_l1437_143743


namespace NUMINAMATH_CALUDE_bees_flew_in_l1437_143716

/-- Given an initial number of bees in a hive and a final total number of bees after more fly in,
    this theorem proves that the number of bees that flew in is equal to the difference between
    the final total and the initial number. -/
theorem bees_flew_in (initial_bees final_bees : ℕ) : 
  initial_bees = 16 → final_bees = 24 → final_bees - initial_bees = 8 := by
  sorry

end NUMINAMATH_CALUDE_bees_flew_in_l1437_143716


namespace NUMINAMATH_CALUDE_tims_soda_cans_l1437_143791

theorem tims_soda_cans (x : ℕ) : 
  x - 6 + (x - 6) / 2 = 24 → x = 22 := by
  sorry

end NUMINAMATH_CALUDE_tims_soda_cans_l1437_143791


namespace NUMINAMATH_CALUDE_biased_coin_heads_probability_l1437_143710

/-- The probability of getting heads on a single flip of a biased coin -/
theorem biased_coin_heads_probability (p : ℚ) (h : p = 3/4) : 1 - p = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_biased_coin_heads_probability_l1437_143710


namespace NUMINAMATH_CALUDE_point_not_on_transformed_plane_l1437_143701

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Applies a similarity transformation to a plane -/
def transformPlane (p : Plane3D) (k : ℝ) : Plane3D :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Checks if a point lies on a plane -/
def isPointOnPlane (point : Point3D) (plane : Plane3D) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- The main theorem to be proved -/
theorem point_not_on_transformed_plane :
  let A : Point3D := { x := -3, y := -2, z := 4 }
  let a : Plane3D := { a := 2, b := -3, c := 1, d := -5 }
  let k : ℝ := -4/5
  let a' : Plane3D := transformPlane a k
  ¬ isPointOnPlane A a' := by
  sorry

end NUMINAMATH_CALUDE_point_not_on_transformed_plane_l1437_143701


namespace NUMINAMATH_CALUDE_double_age_in_two_years_l1437_143700

/-- The number of years until a man's age is twice his son's age -/
def yearsUntilDoubleAge (sonAge manAge : ℕ) : ℕ :=
  if manAge ≤ sonAge then 0
  else (manAge - sonAge)

theorem double_age_in_two_years (sonAge manAge : ℕ) 
  (h1 : manAge = sonAge + 24)
  (h2 : sonAge = 22) :
  yearsUntilDoubleAge sonAge manAge = 2 := by
sorry

end NUMINAMATH_CALUDE_double_age_in_two_years_l1437_143700


namespace NUMINAMATH_CALUDE_dividend_proof_l1437_143740

theorem dividend_proof : ∃ (a b : ℕ), 
  (11 * 10^5 + a * 10^3 + 7 * 10^2 + 7 * 10 + b) / 12 = 999809 → 
  11 * 10^5 + a * 10^3 + 7 * 10^2 + 7 * 10 + b = 11997708 :=
by
  sorry

end NUMINAMATH_CALUDE_dividend_proof_l1437_143740


namespace NUMINAMATH_CALUDE_sin_plus_cos_range_l1437_143774

open Real

theorem sin_plus_cos_range :
  ∃ (f : ℝ → ℝ), (∀ x, f x = sin x + cos x) ∧
  (∀ y, y ∈ Set.range f ↔ -Real.sqrt 2 ≤ y ∧ y ≤ Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_range_l1437_143774


namespace NUMINAMATH_CALUDE_expectation_of_function_l1437_143762

-- Define the random variable ξ
variable (ξ : ℝ → ℝ)

-- Define the expectation operator
noncomputable def E (X : ℝ → ℝ) : ℝ := sorry

-- Define the variance operator
noncomputable def D (X : ℝ → ℝ) : ℝ := E (fun x => (X x - E X)^2)

theorem expectation_of_function (ξ : ℝ → ℝ) 
  (h1 : E ξ = -1) 
  (h2 : D ξ = 3) : 
  E (fun x => 3 * ((ξ x)^2 - 2)) = 6 := 
sorry

end NUMINAMATH_CALUDE_expectation_of_function_l1437_143762


namespace NUMINAMATH_CALUDE_total_apples_for_bobbing_l1437_143717

/-- The number of apples in each bucket for bobbing apples. -/
def apples_per_bucket : ℕ := 9

/-- The number of buckets Mrs. Walker needs. -/
def number_of_buckets : ℕ := 7

/-- Theorem: Mrs. Walker has 63 apples for bobbing for apples. -/
theorem total_apples_for_bobbing : 
  apples_per_bucket * number_of_buckets = 63 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_for_bobbing_l1437_143717


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l1437_143760

theorem quadratic_root_problem (m n k : ℝ) : 
  (m^2 + 2*m + k = 0) → 
  (n^2 + 2*n + k = 0) → 
  (1/m + 1/n = 6) → 
  (k = -1/3) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l1437_143760


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1437_143708

theorem necessary_but_not_sufficient : 
  (∀ x : ℝ, x^2 < x → |x - 1| < 2) ∧ 
  (∃ x : ℝ, |x - 1| < 2 ∧ x^2 ≥ x) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1437_143708


namespace NUMINAMATH_CALUDE_womens_bathing_suits_l1437_143770

theorem womens_bathing_suits (total : ℕ) (mens : ℕ) (womens : ℕ) : 
  total = 19766 → mens = 14797 → womens = total - mens → womens = 4969 := by
  sorry

end NUMINAMATH_CALUDE_womens_bathing_suits_l1437_143770


namespace NUMINAMATH_CALUDE_sum_of_possible_base_3_digits_l1437_143751

/-- The number of digits a positive integer has in a given base -/
def num_digits (n : ℕ) (base : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log base n + 1

/-- Checks if a number has exactly 4 digits in base 7 -/
def has_four_digits_base_7 (n : ℕ) : Prop :=
  num_digits n 7 = 4

/-- The smallest 4-digit number in base 7 -/
def min_four_digit_base_7 : ℕ := 7^3

/-- The largest 4-digit number in base 7 -/
def max_four_digit_base_7 : ℕ := 7^4 - 1

/-- The theorem to be proved -/
theorem sum_of_possible_base_3_digits : 
  (∀ n : ℕ, has_four_digits_base_7 n → 
    (num_digits n 3 = 6 ∨ num_digits n 3 = 7)) ∧ 
  (∃ n m : ℕ, has_four_digits_base_7 n ∧ has_four_digits_base_7 m ∧ 
    num_digits n 3 = 6 ∧ num_digits m 3 = 7) :=
sorry

end NUMINAMATH_CALUDE_sum_of_possible_base_3_digits_l1437_143751


namespace NUMINAMATH_CALUDE_sue_votes_count_l1437_143778

def total_votes : ℕ := 1000
def candidate1_percentage : ℚ := 20 / 100
def candidate2_percentage : ℚ := 45 / 100

theorem sue_votes_count :
  let sue_percentage : ℚ := 1 - (candidate1_percentage + candidate2_percentage)
  (sue_percentage * total_votes : ℚ) = 350 := by sorry

end NUMINAMATH_CALUDE_sue_votes_count_l1437_143778


namespace NUMINAMATH_CALUDE_rhombus_tiling_2n_gon_rhombus_tiling_2002_gon_l1437_143738

/-- The number of rhombuses needed to tile a regular 2n-gon -/
def num_rhombuses (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating the number of rhombuses in a tiling of a regular 2n-gon -/
theorem rhombus_tiling_2n_gon (n : ℕ) (h : n > 1) :
  num_rhombuses n = n * (n - 1) / 2 :=
by sorry

/-- Corollary for the specific case of a 2002-gon -/
theorem rhombus_tiling_2002_gon :
  num_rhombuses 1001 = 500500 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_tiling_2n_gon_rhombus_tiling_2002_gon_l1437_143738


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l1437_143747

open Set

theorem inequality_solution_sets 
  (a b c d : ℝ) 
  (h : {x : ℝ | (b / (x + a)) + ((x + d) / (x + c)) < 0} = Ioo (-1) (-1/3) ∪ Ioo (1/2) 1) :
  {x : ℝ | (b * x / (a * x - 1)) + ((d * x - 1) / (c * x - 1)) < 0} = Ioo 1 3 ∪ Ioo (-2) (-1) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l1437_143747


namespace NUMINAMATH_CALUDE_square_area_error_l1437_143786

theorem square_area_error (s : ℝ) (h : s > 0) : 
  let measured_side := 1.08 * s
  let actual_area := s ^ 2
  let calculated_area := measured_side ^ 2
  let error_percentage := (calculated_area - actual_area) / actual_area * 100
  error_percentage = 16.64 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l1437_143786


namespace NUMINAMATH_CALUDE_count_valid_arrangements_l1437_143723

/-- The number of ways to arrange the digits 1, 1, 2, 5, 0 into a five-digit multiple of 5 -/
def arrangementCount : ℕ := 21

/-- The set of digits available for arrangement -/
def availableDigits : Finset ℕ := {1, 2, 5, 0}

/-- Predicate to check if a number is a five-digit multiple of 5 -/
def isValidNumber (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧ n % 5 = 0

/-- The set of all valid arrangements -/
def validArrangements : Finset ℕ :=
  sorry

theorem count_valid_arrangements :
  Finset.card validArrangements = arrangementCount := by
  sorry

end NUMINAMATH_CALUDE_count_valid_arrangements_l1437_143723


namespace NUMINAMATH_CALUDE_rectangle_area_l1437_143795

/-- Given a rectangle with perimeter 40 feet and length-to-width ratio 3:2, its area is 96 square feet -/
theorem rectangle_area (length width : ℝ) : 
  (2 * (length + width) = 40) →  -- perimeter condition
  (length = 3/2 * width) →       -- ratio condition
  (length * width = 96) :=        -- area is 96 square feet
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1437_143795


namespace NUMINAMATH_CALUDE_set_A_equals_one_two_l1437_143720

def A : Set ℕ := {x | x^2 - 3*x < 0 ∧ x > 0}

theorem set_A_equals_one_two : A = {1, 2} := by sorry

end NUMINAMATH_CALUDE_set_A_equals_one_two_l1437_143720


namespace NUMINAMATH_CALUDE_cube_root_5488000_l1437_143757

theorem cube_root_5488000 :
  let n : ℝ := 5488000
  ∀ (x : ℝ), x^3 = n → x = 140 * Real.rpow 2 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_5488000_l1437_143757


namespace NUMINAMATH_CALUDE_common_ratio_of_sequence_l1437_143729

def geometric_sequence (a : ℤ → ℤ) (r : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * r

theorem common_ratio_of_sequence (a : ℤ → ℤ) :
  a 0 = 25 ∧ a 1 = -50 ∧ a 2 = 100 ∧ a 3 = -200 →
  ∃ r : ℤ, geometric_sequence a r ∧ r = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_common_ratio_of_sequence_l1437_143729


namespace NUMINAMATH_CALUDE_C_equals_46_l1437_143705

/-- Custom operation ⊕ -/
def circplus (a b : ℕ) : ℕ := a * b + 10

/-- Definition of C using the custom operation -/
def C : ℕ := circplus (circplus 1 2) 3

/-- Theorem stating that C equals 46 -/
theorem C_equals_46 : C = 46 := by
  sorry

end NUMINAMATH_CALUDE_C_equals_46_l1437_143705


namespace NUMINAMATH_CALUDE_fifteen_factorial_base16_zeros_l1437_143721

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define a function to count trailing zeros in base 16
def trailingZerosBase16 (n : ℕ) : ℕ :=
  -- Implementation details omitted
  sorry

-- Theorem statement
theorem fifteen_factorial_base16_zeros :
  trailingZerosBase16 (factorial 15) = 4 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_factorial_base16_zeros_l1437_143721


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_6_l1437_143776

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def append_digit (n m : ℕ) : ℕ := n * 10 + m

theorem largest_digit_divisible_by_6 :
  ∃ (M : ℕ), M ≤ 9 ∧ 
    is_divisible_by_6 (append_digit 5172 M) ∧ 
    ∀ (K : ℕ), K ≤ 9 → is_divisible_by_6 (append_digit 5172 K) → K ≤ M :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_6_l1437_143776


namespace NUMINAMATH_CALUDE_abs_negative_2023_l1437_143733

theorem abs_negative_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_2023_l1437_143733


namespace NUMINAMATH_CALUDE_max_missed_problems_correct_l1437_143782

/-- The number of problems in the test -/
def total_problems : ℕ := 50

/-- The minimum percentage required to pass the test -/
def pass_percentage : ℚ := 85 / 100

/-- The maximum number of problems a student can miss and still pass the test -/
def max_missed_problems : ℕ := 7

theorem max_missed_problems_correct :
  (max_missed_problems ≤ total_problems) ∧
  ((total_problems - max_missed_problems : ℚ) / total_problems ≥ pass_percentage) ∧
  ∀ n : ℕ, n > max_missed_problems →
    ((total_problems - n : ℚ) / total_problems < pass_percentage) :=
by sorry

end NUMINAMATH_CALUDE_max_missed_problems_correct_l1437_143782


namespace NUMINAMATH_CALUDE_halloween_candy_weight_l1437_143732

/-- The combined weight of candy Frank and Gwen received for Halloween -/
def combined_candy_weight (frank_candy : ℕ) (gwen_candy : ℕ) : ℕ :=
  frank_candy + gwen_candy

/-- Theorem: The combined weight of candy Frank and Gwen received is 17 pounds -/
theorem halloween_candy_weight :
  combined_candy_weight 10 7 = 17 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_weight_l1437_143732


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l1437_143704

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person

-- Define the set of cards
inductive Card : Type
| Red : Card
| Blue : Card
| White : Card

-- Define a distribution of cards to people
def Distribution := Person → Card

-- Define the event "Person A receives the white card"
def event_A_white (d : Distribution) : Prop := d Person.A = Card.White

-- Define the event "Person B receives the white card"
def event_B_white (d : Distribution) : Prop := d Person.B = Card.White

-- State the theorem
theorem events_mutually_exclusive_not_complementary :
  (∀ d : Distribution, ¬(event_A_white d ∧ event_B_white d)) ∧
  (∃ d : Distribution, ¬event_A_white d ∧ ¬event_B_white d) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l1437_143704


namespace NUMINAMATH_CALUDE_jill_first_bus_wait_time_l1437_143750

/-- Represents Jill's bus journey times -/
structure BusJourney where
  first_bus_wait : ℕ
  first_bus_ride : ℕ
  second_bus_ride : ℕ

/-- The conditions of Jill's bus journey -/
def journey_conditions (j : BusJourney) : Prop :=
  j.first_bus_ride = 30 ∧
  j.second_bus_ride = 21 ∧
  j.second_bus_ride * 2 = j.first_bus_wait + j.first_bus_ride

theorem jill_first_bus_wait_time (j : BusJourney) 
  (h : journey_conditions j) : j.first_bus_wait = 12 := by
  sorry

end NUMINAMATH_CALUDE_jill_first_bus_wait_time_l1437_143750


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1437_143736

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 1 + a 9 = 10) : 
  a 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1437_143736


namespace NUMINAMATH_CALUDE_ellipse_and_line_theorem_l1437_143796

-- Define the ellipse E
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the line that B and F lie on
def line_BF (x y : ℝ) : Prop :=
  x - y + 2 = 0

-- Define the midpoint condition
def is_midpoint (x₁ y₁ x₂ y₂ xₘ yₘ : ℝ) : Prop :=
  xₘ = (x₁ + x₂) / 2 ∧ yₘ = (y₁ + y₂) / 2

theorem ellipse_and_line_theorem :
  ∀ a b : ℝ,
  a > b ∧ b > 0 →
  (∃ xB yB xF yF : ℝ,
    ellipse a b xB yB ∧
    ellipse a b xF yF ∧
    line_BF xB yB ∧
    line_BF xF yF ∧
    yB > yF) →
  (∀ x y : ℝ, ellipse a b x y ↔ x^2 / 8 + y^2 / 4 = 1) ∧
  (∀ x₁ y₁ x₂ y₂ : ℝ,
    ellipse a b x₁ y₁ ∧
    ellipse a b x₂ y₂ ∧
    is_midpoint x₁ y₁ x₂ y₂ (-1) 1 →
    x₁ - 2*y₁ + 3 = 0 ∧
    x₂ - 2*y₂ + 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_theorem_l1437_143796


namespace NUMINAMATH_CALUDE_greenfield_high_lockers_l1437_143781

/-- The cost in cents for each plastic digit used in labeling lockers -/
def digit_cost : ℚ := 3

/-- The total cost in dollars for labeling all lockers -/
def total_cost : ℚ := 273.39

/-- Calculates the cost of labeling lockers from 1 to n -/
def labeling_cost (n : ℕ) : ℚ :=
  let ones := min n 9
  let tens := min (n - 9) 90
  let hundreds := min (n - 99) 900
  let thousands := max (n - 999) 0
  (ones * digit_cost + 
   tens * 2 * digit_cost + 
   hundreds * 3 * digit_cost + 
   thousands * 4 * digit_cost) / 100

/-- The number of lockers at Greenfield High -/
def num_lockers : ℕ := 2555

theorem greenfield_high_lockers : 
  labeling_cost num_lockers = total_cost :=
sorry

end NUMINAMATH_CALUDE_greenfield_high_lockers_l1437_143781


namespace NUMINAMATH_CALUDE_x_range_for_inequality_l1437_143711

theorem x_range_for_inequality (m : ℝ) (hm : m ∈ Set.Icc 0 1) :
  {x : ℝ | m * x^2 - 2 * x - m ≥ 2} ⊆ Set.Iic (-1) := by
sorry

end NUMINAMATH_CALUDE_x_range_for_inequality_l1437_143711


namespace NUMINAMATH_CALUDE_ellipse_and_line_properties_l1437_143726

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_point : (1^2 / a^2) + ((3/2)^2 / b^2) = 1
  h_ecc : (a^2 - b^2).sqrt / a = 1/2

/-- A line intersecting the ellipse -/
structure IntersectingLine (e : Ellipse) where
  k : ℝ
  m : ℝ
  h_k : k ≠ 0
  h_intersect : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧
    (x₁^2 / 4) + (y₁^2 / 3) = 1 ∧
    (x₂^2 / 4) + (y₂^2 / 3) = 1 ∧
    y₁ = k * x₁ + m ∧
    y₂ = k * x₂ + m

/-- The main theorem -/
theorem ellipse_and_line_properties (e : Ellipse) (l : IntersectingLine e) :
  (∀ (x y : ℝ), (x^2 / 4) + (y^2 / 3) = 1 ↔ (x^2 / e.a^2) + (y^2 / e.b^2) = 1) ∧
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧
    (x₁^2 / 4) + (y₁^2 / 3) = 1 ∧
    (x₂^2 / 4) + (y₂^2 / 3) = 1 ∧
    y₁ = l.k * x₁ + l.m ∧
    y₂ = l.k * x₂ + l.m ∧
    (-(x₂ - x₁) * ((y₁ + y₂)/2 - 0) = (y₂ - y₁) * ((x₁ + x₂)/2 - 1/8)) →
    l.k < -Real.sqrt 5 / 10 ∨ l.k > Real.sqrt 5 / 10) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_and_line_properties_l1437_143726


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_106_l1437_143773

theorem last_three_digits_of_7_to_106 : ∃ n : ℕ, 7^106 ≡ 321 [ZMOD 1000] :=
by sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_106_l1437_143773


namespace NUMINAMATH_CALUDE_expression_value_l1437_143799

/-- Given x, y, and z as defined, prove that the expression equals 20 -/
theorem expression_value (x y z : ℝ) 
  (hx : x = -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5)
  (hy : y = Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5)
  (hz : z = Real.sqrt 2 + Real.sqrt 3 - Real.sqrt 5) :
  (x^4 / ((x-y)*(x-z))) + (y^4 / ((y-z)*(y-x))) + (z^4 / ((z-x)*(z-y))) = 20 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1437_143799


namespace NUMINAMATH_CALUDE_problem_statement_l1437_143724

theorem problem_statement (a b c : ℝ) (h1 : a < 0) (h2 : a < b) (h3 : b < 0) (h4 : 0 < c) :
  (a * b > a * c) ∧ (a * b > b * c) ∧ (a + c < b + c) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1437_143724


namespace NUMINAMATH_CALUDE_train_length_l1437_143759

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 108 → time = 7 → speed * time * (1000 / 3600) = 210 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1437_143759


namespace NUMINAMATH_CALUDE_reaction_result_l1437_143735

-- Define the chemical equation
structure ChemicalEquation where
  nh4cl : ℕ
  naoh : ℕ
  nh3 : ℕ
  h2o : ℕ
  nacl : ℕ

-- Define the initial reactants
def initial_reactants : ChemicalEquation :=
  { nh4cl := 2, naoh := 3, nh3 := 0, h2o := 0, nacl := 0 }

-- Define the balanced equation coefficients
def balanced_equation : ChemicalEquation :=
  { nh4cl := 1, naoh := 1, nh3 := 1, h2o := 1, nacl := 1 }

-- Define the reaction function
def react (reactants : ChemicalEquation) : ChemicalEquation :=
  let limiting_reactant := min reactants.nh4cl reactants.naoh
  { nh4cl := reactants.nh4cl - limiting_reactant,
    naoh := reactants.naoh - limiting_reactant,
    nh3 := limiting_reactant,
    h2o := limiting_reactant,
    nacl := limiting_reactant }

-- Theorem statement
theorem reaction_result :
  let result := react initial_reactants
  result.h2o = 2 ∧ result.nh3 = 2 ∧ result.nacl = 2 ∧ result.naoh = 1 :=
sorry

end NUMINAMATH_CALUDE_reaction_result_l1437_143735


namespace NUMINAMATH_CALUDE_point_P_on_circle_O_l1437_143784

/-- A circle with center at the origin and radius 5 -/
def circle_O : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = 25}

/-- Point P with coordinates (4,3) -/
def point_P : ℝ × ℝ := (4, 3)

/-- Theorem stating that point P lies on circle O -/
theorem point_P_on_circle_O : point_P ∈ circle_O := by
  sorry

end NUMINAMATH_CALUDE_point_P_on_circle_O_l1437_143784


namespace NUMINAMATH_CALUDE_bug_walk_tiles_l1437_143768

/-- The number of tiles a bug visits when walking in a straight line from one corner to the opposite corner of a rectangular floor. -/
def tilesVisited (width : ℕ) (length : ℕ) : ℕ :=
  width + length - Nat.gcd width length

/-- The theorem stating that for a 15x35 foot rectangular floor, a bug walking diagonally visits 45 tiles. -/
theorem bug_walk_tiles : tilesVisited 15 35 = 45 := by
  sorry

end NUMINAMATH_CALUDE_bug_walk_tiles_l1437_143768


namespace NUMINAMATH_CALUDE_original_pencils_count_l1437_143797

/-- The number of pencils originally in the drawer -/
def original_pencils : ℕ := sorry

/-- The number of pencils Joan added to the drawer -/
def added_pencils : ℕ := 27

/-- The total number of pencils after Joan's addition -/
def total_pencils : ℕ := 60

/-- Theorem stating that the original number of pencils was 33 -/
theorem original_pencils_count : original_pencils = 33 :=
by
  sorry

#check original_pencils_count

end NUMINAMATH_CALUDE_original_pencils_count_l1437_143797


namespace NUMINAMATH_CALUDE_adam_shopping_cost_l1437_143789

/-- The total cost of Adam's shopping given the number of sandwiches, 
    price per sandwich, and price of water. -/
def total_cost (num_sandwiches : ℕ) (price_per_sandwich : ℕ) (price_of_water : ℕ) : ℕ :=
  num_sandwiches * price_per_sandwich + price_of_water

/-- Theorem stating that Adam's total shopping cost is $11 -/
theorem adam_shopping_cost : 
  total_cost 3 3 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_adam_shopping_cost_l1437_143789


namespace NUMINAMATH_CALUDE_conic_is_ellipse_l1437_143777

-- Define the equation
def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y-2)^2) + Real.sqrt ((x-6)^2 + (y+4)^2) = 12

-- Define the two fixed points
def focus1 : ℝ × ℝ := (0, 2)
def focus2 : ℝ × ℝ := (6, -4)

-- Theorem stating that the equation describes an ellipse
theorem conic_is_ellipse :
  ∃ (a b : ℝ) (center : ℝ × ℝ),
    a > 0 ∧ b > 0 ∧ a > b ∧
    ∀ (x y : ℝ),
      conic_equation x y ↔
        (x - center.1)^2 / a^2 + (y - center.2)^2 / b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_conic_is_ellipse_l1437_143777


namespace NUMINAMATH_CALUDE_sum_division_problem_l1437_143719

theorem sum_division_problem (share_ratio_a share_ratio_b share_ratio_c : ℕ) 
  (second_person_share : ℚ) (total_amount : ℚ) : 
  share_ratio_a = 100 → 
  share_ratio_b = 45 → 
  share_ratio_c = 30 → 
  second_person_share = 63 → 
  total_amount = (second_person_share / share_ratio_b) * (share_ratio_a + share_ratio_b + share_ratio_c) → 
  total_amount = 245 := by
sorry

end NUMINAMATH_CALUDE_sum_division_problem_l1437_143719


namespace NUMINAMATH_CALUDE_power_sum_difference_l1437_143758

theorem power_sum_difference : 2^(1+2+3) - (2^1 + 2^2 + 2^3) = 50 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_difference_l1437_143758


namespace NUMINAMATH_CALUDE_H_constant_l1437_143783

def H (x : ℝ) : ℝ := |x + 2| - |x - 3|

theorem H_constant : ∀ x : ℝ, H x = 5 := by sorry

end NUMINAMATH_CALUDE_H_constant_l1437_143783


namespace NUMINAMATH_CALUDE_nina_weekend_earnings_l1437_143703

-- Define the prices and quantities
def necklace_price : ℚ := 25
def bracelet_price : ℚ := 15
def earring_pair_price : ℚ := 10
def ensemble_price : ℚ := 45

def necklaces_sold : ℕ := 5
def bracelets_sold : ℕ := 10
def earring_pairs_sold : ℕ := 20
def ensembles_sold : ℕ := 2

-- Define the total earnings
def total_earnings : ℚ :=
  necklace_price * necklaces_sold +
  bracelet_price * bracelets_sold +
  earring_pair_price * earring_pairs_sold +
  ensemble_price * ensembles_sold

-- Theorem to prove
theorem nina_weekend_earnings :
  total_earnings = 565 := by
  sorry

end NUMINAMATH_CALUDE_nina_weekend_earnings_l1437_143703


namespace NUMINAMATH_CALUDE_square_binomial_coefficient_l1437_143725

theorem square_binomial_coefficient (a : ℝ) : 
  (∃ r s : ℝ, ∀ x : ℝ, a * x^2 + 8 * x + 16 = (r * x + s)^2) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_binomial_coefficient_l1437_143725


namespace NUMINAMATH_CALUDE_regression_and_variance_l1437_143752

-- Define the data points
def x : List Real := [5, 5.5, 6, 6.5, 7]
def y : List Real := [50, 48, 43, 38, 36]

-- Define the probability of "very good" experience
def p : Real := 0.5

-- Define the number of trials
def n : Nat := 5

-- Theorem statement
theorem regression_and_variance :
  let x_mean := (x.sum) / x.length
  let y_mean := (y.sum) / y.length
  let xy_sum := (List.zip x y).map (fun (a, b) => a * b) |>.sum
  let x_squared_sum := x.map (fun a => a ^ 2) |>.sum
  let slope := (xy_sum - x.length * x_mean * y_mean) / (x_squared_sum - x.length * x_mean ^ 2)
  let intercept := y_mean - slope * x_mean
  let variance := n * p * (1 - p)
  slope = -7.6 ∧ intercept = 88.6 ∧ variance = 5/4 := by
  sorry

#check regression_and_variance

end NUMINAMATH_CALUDE_regression_and_variance_l1437_143752


namespace NUMINAMATH_CALUDE_jerry_added_figures_l1437_143761

/-- Represents the shelf of action figures -/
structure ActionFigureShelf :=
  (initial_count : Nat)
  (final_count : Nat)
  (removed_count : Nat)
  (is_arithmetic_sequence : Bool)
  (first_last_preserved : Bool)
  (common_difference_preserved : Bool)

/-- Calculates the number of action figures added to the shelf -/
def added_figures (shelf : ActionFigureShelf) : Nat :=
  shelf.final_count + shelf.removed_count - shelf.initial_count

/-- Theorem stating the number of added action figures -/
theorem jerry_added_figures (shelf : ActionFigureShelf) 
  (h1 : shelf.initial_count = 7)
  (h2 : shelf.final_count = 8)
  (h3 : shelf.removed_count = 10)
  (h4 : shelf.is_arithmetic_sequence = true)
  (h5 : shelf.first_last_preserved = true)
  (h6 : shelf.common_difference_preserved = true) :
  added_figures shelf = 18 := by
  sorry

#check jerry_added_figures

end NUMINAMATH_CALUDE_jerry_added_figures_l1437_143761


namespace NUMINAMATH_CALUDE_ellipse_properties_l1437_143763

/-- Given an ellipse with equation x²/100 + y²/36 = 1, prove that its major axis length is 20 and eccentricity is 4/5 -/
theorem ellipse_properties (x y : ℝ) :
  x^2 / 100 + y^2 / 36 = 1 →
  ∃ (a b c : ℝ),
    a = 10 ∧
    b = 6 ∧
    c^2 = a^2 - b^2 ∧
    2 * a = 20 ∧
    c / a = 4 / 5 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1437_143763


namespace NUMINAMATH_CALUDE_factorization_equality_l1437_143713

theorem factorization_equality (x : ℝ) : 
  (x + 1) * (x + 2) * (x + 3) * (x + 4) - 120 = (x^2 + 5*x + 16) * (x + 6) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1437_143713


namespace NUMINAMATH_CALUDE_completing_square_sum_l1437_143767

-- Define the original quadratic equation
def original_equation (x : ℝ) : Prop := x^2 - 6*x = 1

-- Define the transformed equation
def transformed_equation (x m n : ℝ) : Prop := (x - m)^2 = n

-- Theorem statement
theorem completing_square_sum (m n : ℝ) :
  (∀ x, original_equation x ↔ transformed_equation x m n) →
  m + n = 13 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_sum_l1437_143767


namespace NUMINAMATH_CALUDE_f_seven_equals_neg_seventeen_l1437_143715

/-- Given a function f(x) = a*x^7 + b*x^3 + c*x - 5 where a, b, and c are constants,
    if f(-7) = 7, then f(7) = -17 -/
theorem f_seven_equals_neg_seventeen 
  (a b c : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^7 + b * x^3 + c * x - 5) 
  (h2 : f (-7) = 7) : 
  f 7 = -17 := by
sorry

end NUMINAMATH_CALUDE_f_seven_equals_neg_seventeen_l1437_143715


namespace NUMINAMATH_CALUDE_truth_values_of_p_and_q_l1437_143785

theorem truth_values_of_p_and_q (p q : Prop)
  (h1 : p ∨ q)
  (h2 : ¬(p ∧ q))
  (h3 : ¬p) :
  ¬p ∧ q := by sorry

end NUMINAMATH_CALUDE_truth_values_of_p_and_q_l1437_143785


namespace NUMINAMATH_CALUDE_toy_count_is_134_l1437_143718

/-- The number of toys initially in the box, given the conditions of the problem -/
def initial_toy_count : ℕ := by sorry

theorem toy_count_is_134 :
  -- Define variables for red and white toys
  ∀ (red white : ℕ),
  -- After removing 2 red toys, red is twice white
  (red - 2 = 2 * white) →
  -- After removing 2 red toys, there are 88 red toys
  (red - 2 = 88) →
  -- The initial toy count is the sum of red and white toys
  initial_toy_count = red + white →
  -- Prove that the initial toy count is 134
  initial_toy_count = 134 := by sorry

end NUMINAMATH_CALUDE_toy_count_is_134_l1437_143718


namespace NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l1437_143739

theorem unique_solution_for_exponential_equation :
  ∀ (a b : ℕ+), 1 + 5^(a : ℕ) = 6^(b : ℕ) ↔ a = 1 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l1437_143739


namespace NUMINAMATH_CALUDE_parabola_sum_is_line_l1437_143734

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola -/
def original : QuadraticFunction :=
  { a := 3, b := 4, c := -5 }

/-- Reflects a quadratic function about the x-axis -/
def reflect (f : QuadraticFunction) : QuadraticFunction :=
  { a := -f.a, b := -f.b, c := -f.c }

/-- Translates a quadratic function horizontally -/
def translate (f : QuadraticFunction) (d : ℝ) : QuadraticFunction :=
  { a := f.a
  , b := f.b - 2 * f.a * d
  , c := f.a * d^2 - f.b * d + f.c }

/-- Adds two quadratic functions -/
def add (f g : QuadraticFunction) : QuadraticFunction :=
  { a := f.a + g.a
  , b := f.b + g.b
  , c := f.c + g.c }

/-- Theorem stating that the sum of the translated original parabola and its reflected and translated version is a non-horizontal line -/
theorem parabola_sum_is_line :
  let f := translate original 4
  let g := translate (reflect original) (-6)
  let sum := add f g
  sum.a = 0 ∧ sum.b ≠ 0 := by sorry

end NUMINAMATH_CALUDE_parabola_sum_is_line_l1437_143734


namespace NUMINAMATH_CALUDE_power_function_fixed_point_l1437_143737

theorem power_function_fixed_point (α : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ x^α
  f 1 = 1 := by sorry

end NUMINAMATH_CALUDE_power_function_fixed_point_l1437_143737


namespace NUMINAMATH_CALUDE_cube_side_area_l1437_143787

theorem cube_side_area (edge_sum : ℝ) (h : edge_sum = 132) : 
  let edge_length := edge_sum / 12
  (edge_length ^ 2) = 121 := by sorry

end NUMINAMATH_CALUDE_cube_side_area_l1437_143787


namespace NUMINAMATH_CALUDE_f_prime_zero_l1437_143744

theorem f_prime_zero (f : ℝ → ℝ) (h : Differentiable ℝ f) 
  (h1 : ∀ x, f x = x^2 + 2 * (deriv f 2) * x + 3) : 
  deriv f 0 = -8 := by sorry

end NUMINAMATH_CALUDE_f_prime_zero_l1437_143744


namespace NUMINAMATH_CALUDE_vertex_distance_is_five_l1437_143765

/-- The equation of the curve -/
def curve_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + |y - 1| = 5

/-- The y-coordinate of the upper vertex -/
def upper_vertex_y : ℝ := 3

/-- The y-coordinate of the lower vertex -/
def lower_vertex_y : ℝ := -2

/-- The distance between the vertices -/
def vertex_distance : ℝ := |upper_vertex_y - lower_vertex_y|

theorem vertex_distance_is_five :
  vertex_distance = 5 :=
by sorry

end NUMINAMATH_CALUDE_vertex_distance_is_five_l1437_143765


namespace NUMINAMATH_CALUDE_isosceles_triangle_height_l1437_143771

/-- Given a positive constant s, prove that an isosceles triangle with base 2s
    and area equal to a rectangle with dimensions 2s and s has height 2s. -/
theorem isosceles_triangle_height (s : ℝ) (hs : s > 0) : 
  let rectangle_area := 2 * s * s
  let triangle_base := 2 * s
  let triangle_height := 2 * s
  rectangle_area = 1/2 * triangle_base * triangle_height := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_height_l1437_143771


namespace NUMINAMATH_CALUDE_cubic_roots_problem_l1437_143772

/-- A cubic polynomial with coefficient c and constant term d -/
def cubic (c d : ℝ) (x : ℝ) : ℝ := x^3 + c*x + d

theorem cubic_roots_problem (c d : ℝ) (u v : ℝ) :
  (∃ w, cubic c d u = 0 ∧ cubic c d v = 0 ∧ cubic c d w = 0) ∧
  (∃ w', cubic c (d + 300) (u + 5) = 0 ∧ cubic c (d + 300) (v - 4) = 0 ∧ cubic c (d + 300) w' = 0) →
  d = -616 ∨ d = 1575 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_problem_l1437_143772


namespace NUMINAMATH_CALUDE_probability_divisible_by_five_l1437_143766

/-- A three-digit positive integer -/
def ThreeDigitInteger (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

/-- An integer ending in 5 -/
def EndsInFive (n : ℕ) : Prop :=
  n % 10 = 5

/-- The probability that a three-digit positive integer ending in 5 is divisible by 5 is 1 -/
theorem probability_divisible_by_five :
  ∀ n : ℕ, ThreeDigitInteger n → EndsInFive n → n % 5 = 0 :=
sorry

end NUMINAMATH_CALUDE_probability_divisible_by_five_l1437_143766


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l1437_143753

theorem no_positive_integer_solution (m n : ℕ+) : 4 * m * (m + 1) ≠ n * (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l1437_143753


namespace NUMINAMATH_CALUDE_no_natural_solution_for_square_difference_2014_l1437_143730

theorem no_natural_solution_for_square_difference_2014 :
  ∀ (m n : ℕ), m^2 ≠ n^2 + 2014 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solution_for_square_difference_2014_l1437_143730


namespace NUMINAMATH_CALUDE_simplify_expression_l1437_143779

theorem simplify_expression (x y : ℝ) :
  let P := 2 * x + 3 * y
  let Q := x - 2 * y
  (P + Q) / (P - Q) - (P - Q) / (P + Q) = (8 * x^2 - 4 * x * y - 24 * y^2) / (3 * x^2 + 16 * x * y + 5 * y^2) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1437_143779


namespace NUMINAMATH_CALUDE_extremum_point_of_f_l1437_143794

def f (x : ℝ) := x^2 + 1

theorem extremum_point_of_f :
  ∃ (x : ℝ), ∀ (y : ℝ), f y ≥ f x :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_extremum_point_of_f_l1437_143794


namespace NUMINAMATH_CALUDE_integers_abs_lt_3_l1437_143749

theorem integers_abs_lt_3 : 
  {n : ℤ | |n| < 3} = {-2, -1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_integers_abs_lt_3_l1437_143749


namespace NUMINAMATH_CALUDE_average_speed_calculation_l1437_143707

/-- Given a journey with a distance and time, calculate the average speed -/
theorem average_speed_calculation (distance : ℝ) (time : ℝ) (h1 : distance = 210) (h2 : time = 35/6) :
  distance / time = 36 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l1437_143707


namespace NUMINAMATH_CALUDE_problem_statement_l1437_143728

theorem problem_statement (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) 
  (h3 : a^2 + 4*b^2 = 1/(a*b) + 3) : 
  (a * b ≤ 1) ∧ 
  (b > a → 1/a^3 - 1/b^3 ≥ 3*(1/a - 1/b)) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1437_143728


namespace NUMINAMATH_CALUDE_inequality_proof_l1437_143793

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≤ a*c) :
  (a*f - c*d)^2 ≥ (a*e - b*d)*(b*f - c*e) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1437_143793


namespace NUMINAMATH_CALUDE_equation_solution_l1437_143742

theorem equation_solution :
  let f (x : ℝ) := x + 2 - 4 / (x - 3)
  ∃ (x₁ x₂ : ℝ), x₁ ≠ 3 ∧ x₂ ≠ 3 ∧
    x₁ = (1 + Real.sqrt 41) / 2 ∧
    x₂ = (1 - Real.sqrt 41) / 2 ∧
    f x₁ = 0 ∧ f x₂ = 0 ∧
    ∀ (x : ℝ), x ≠ 3 → f x = 0 → (x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1437_143742


namespace NUMINAMATH_CALUDE_sector_central_angle_l1437_143798

/-- Given a sector with radius R and circumference 3R, its central angle is 1 radian -/
theorem sector_central_angle (R : ℝ) (h : R > 0) : 
  let circumference := 3 * R
  let arc_length := circumference - 2 * R
  let central_angle := arc_length / R
  central_angle = 1 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1437_143798


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1437_143709

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 and asymptotes y = ±√2x,
    prove that its eccentricity is √3 -/
theorem hyperbola_eccentricity (a b : ℝ) (h : b / a = Real.sqrt 2) :
  let c := Real.sqrt (a^2 + b^2)
  c / a = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1437_143709


namespace NUMINAMATH_CALUDE_new_person_weight_l1437_143780

/-- Given a group of 9 people where one person weighing 86 kg is replaced by a new person,
    and the average weight of the group increases by 5.5 kg,
    prove that the weight of the new person is 135.5 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 9 →
  weight_increase = 5.5 →
  replaced_weight = 86 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 135.5 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1437_143780


namespace NUMINAMATH_CALUDE_two_face_cubes_4x4x4_l1437_143755

/-- Represents a cuboid with integer dimensions -/
structure Cuboid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of unit cubes with exactly two faces on the surface of a cuboid -/
def count_two_face_cubes (c : Cuboid) : ℕ :=
  12 * (c.length - 2)

/-- Theorem: A 4x4x4 cuboid has 24 unit cubes with exactly two faces on its surface -/
theorem two_face_cubes_4x4x4 :
  let c : Cuboid := ⟨4, 4, 4⟩
  count_two_face_cubes c = 24 := by
  sorry

#eval count_two_face_cubes ⟨4, 4, 4⟩

end NUMINAMATH_CALUDE_two_face_cubes_4x4x4_l1437_143755


namespace NUMINAMATH_CALUDE_race_winner_distance_l1437_143754

theorem race_winner_distance (catrina_distance : ℝ) (catrina_time : ℝ) 
  (sedra_distance : ℝ) (sedra_time : ℝ) (race_distance : ℝ) :
  catrina_distance = 100 ∧ 
  catrina_time = 10 ∧ 
  sedra_distance = 400 ∧ 
  sedra_time = 44 ∧ 
  race_distance = 1000 →
  let catrina_speed := catrina_distance / catrina_time
  let sedra_speed := sedra_distance / sedra_time
  let catrina_race_time := race_distance / catrina_speed
  let sedra_race_distance := sedra_speed * catrina_race_time
  race_distance - sedra_race_distance = 91 :=
by sorry

end NUMINAMATH_CALUDE_race_winner_distance_l1437_143754


namespace NUMINAMATH_CALUDE_triangle_problem_l1437_143769

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  -- Given conditions
  a = 1 ∧ b = 2 ∧ Real.cos C = 1/4 →
  -- Prove
  c = 2 ∧ Real.sin A = Real.sqrt 15 / 8 := by
    sorry

end NUMINAMATH_CALUDE_triangle_problem_l1437_143769


namespace NUMINAMATH_CALUDE_circle_motion_problem_l1437_143745

/-- Given a circle with two points A and B moving along its circumference, 
    this theorem proves the speeds of the points and the circumference of the circle. -/
theorem circle_motion_problem 
  (smaller_arc : ℝ) 
  (smaller_arc_time : ℝ) 
  (larger_arc_time : ℝ) 
  (b_distance : ℝ) 
  (h1 : smaller_arc = 150)
  (h2 : smaller_arc_time = 10)
  (h3 : larger_arc_time = 14)
  (h4 : b_distance = 90) :
  ∃ (va vb l : ℝ),
    va = 12 ∧ 
    vb = 3 ∧ 
    l = 360 ∧
    smaller_arc_time * (va + vb) = smaller_arc ∧
    larger_arc_time * (va + vb) = l - smaller_arc ∧
    l / va = b_distance / vb :=
by sorry

end NUMINAMATH_CALUDE_circle_motion_problem_l1437_143745


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1437_143748

theorem quadratic_roots_property (d e : ℝ) : 
  (3 * d^2 + 4 * d - 7 = 0) → 
  (3 * e^2 + 4 * e - 7 = 0) → 
  (d - 2) * (e - 2) = 13/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1437_143748


namespace NUMINAMATH_CALUDE_solution_set_equality_l1437_143792

open Set

-- Define the solution set
def solutionSet : Set ℝ := {x | |2*x + 1| > 3}

-- State the theorem
theorem solution_set_equality : solutionSet = Iio (-2) ∪ Ioi 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l1437_143792


namespace NUMINAMATH_CALUDE_padic_square_root_solutions_l1437_143706

/-- The number of solutions to x^2 = a in p-adic numbers is either 0 or 2 -/
theorem padic_square_root_solutions (p : ℕ) [Fact (Nat.Prime p)] (a : ℚ_[p]) :
  (∃ x y : ℚ_[p], x ^ 2 = a ∧ y ^ 2 = a ∧ x ≠ y) ∨ (∀ x : ℚ_[p], x ^ 2 ≠ a) :=
sorry

end NUMINAMATH_CALUDE_padic_square_root_solutions_l1437_143706


namespace NUMINAMATH_CALUDE_divisibility_of_expression_l1437_143790

theorem divisibility_of_expression (a b : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) 
  (ha_gt_7 : a > 7) (hb_gt_7 : b > 7) : 
  290304 ∣ (a^2 - 1) * (b^2 - 1) * (a^6 - b^6) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_expression_l1437_143790


namespace NUMINAMATH_CALUDE_abs_negative_two_l1437_143756

theorem abs_negative_two : abs (-2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_two_l1437_143756


namespace NUMINAMATH_CALUDE_inequality_proof_l1437_143775

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x^2 + y^2 + z^2 = 1) :
  x / (1 - x^2) + y / (1 - y^2) + z / (1 - z^2) ≥ (3 / 2) * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1437_143775


namespace NUMINAMATH_CALUDE_age_difference_value_l1437_143741

/-- Represents the ages of three individuals and their relationships -/
structure AgeRelationship where
  /-- Age of Ramesh -/
  x : ℚ
  /-- Age of Suresh -/
  y : ℚ
  /-- Ratio of Ramesh's age to Suresh's age is 2:y -/
  age_ratio : 2 * x = y
  /-- 20 years later, ratio of Ramesh's age to Suresh's age is 8:3 -/
  future_ratio : (5 * x + 20) / (y + 20) = 8 / 3

/-- The difference between Mahesh's and Suresh's present ages -/
def age_difference (ar : AgeRelationship) : ℚ :=
  5 * ar.x - ar.y

/-- Theorem stating the difference between Mahesh's and Suresh's present ages -/
theorem age_difference_value (ar : AgeRelationship) :
  age_difference ar = 125 / 8 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_value_l1437_143741


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_240_360_l1437_143714

theorem lcm_gcf_ratio_240_360 : 
  (Nat.lcm 240 360) / (Nat.gcd 240 360) = 6 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_240_360_l1437_143714


namespace NUMINAMATH_CALUDE_ten_people_round_table_l1437_143746

-- Define the number of people
def n : ℕ := 10

-- Define the function to calculate the number of distinct arrangements
def distinct_circular_arrangements (m : ℕ) : ℕ := Nat.factorial (m - 1)

-- Theorem statement
theorem ten_people_round_table : 
  distinct_circular_arrangements n = Nat.factorial 9 :=
sorry

end NUMINAMATH_CALUDE_ten_people_round_table_l1437_143746


namespace NUMINAMATH_CALUDE_work_completion_time_l1437_143731

/-- Given that:
  * A can finish a work in 10 days
  * When A and B work together, A's share of the work is 3/5
Prove that B can finish the work alone in 15 days -/
theorem work_completion_time (a_time : ℝ) (a_share : ℝ) (b_time : ℝ) : 
  a_time = 10 →
  a_share = 3/5 →
  b_time = (a_time * a_share) / (1 - a_share) →
  b_time = 15 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l1437_143731


namespace NUMINAMATH_CALUDE_two_different_books_count_l1437_143727

/-- The number of ways to select 2 books from different subjects -/
def selectTwoDifferentBooks (chinese : ℕ) (math : ℕ) (english : ℕ) : ℕ :=
  chinese * math + chinese * english + math * english

/-- Theorem: Given 9 Chinese books, 7 math books, and 5 English books,
    there are 143 ways to select 2 books from different subjects -/
theorem two_different_books_count :
  selectTwoDifferentBooks 9 7 5 = 143 := by
  sorry

end NUMINAMATH_CALUDE_two_different_books_count_l1437_143727


namespace NUMINAMATH_CALUDE_choose_four_from_twelve_l1437_143722

theorem choose_four_from_twelve : Nat.choose 12 4 = 495 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_twelve_l1437_143722


namespace NUMINAMATH_CALUDE_airplane_faster_than_driving_l1437_143788

/-- Proves that taking an airplane is 90 minutes faster than driving for a job interview --/
theorem airplane_faster_than_driving :
  let driving_time_minutes : ℕ := 3 * 60 + 15
  let drive_to_airport : ℕ := 10
  let wait_to_board : ℕ := 20
  let flight_time : ℕ := driving_time_minutes / 3
  let get_off_plane : ℕ := 10
  let total_airplane_time : ℕ := drive_to_airport + wait_to_board + flight_time + get_off_plane
  driving_time_minutes - total_airplane_time = 90 := by
  sorry

end NUMINAMATH_CALUDE_airplane_faster_than_driving_l1437_143788


namespace NUMINAMATH_CALUDE_minimum_cost_for_all_entries_l1437_143702

/-- The cost of a single entry in yuan -/
def entry_cost : ℕ := 2

/-- The number of ways to choose 3 consecutive numbers from 01 to 17 -/
def ways_first_segment : ℕ := 15

/-- The number of ways to choose 2 consecutive numbers from 19 to 29 -/
def ways_second_segment : ℕ := 10

/-- The number of ways to choose 1 number from 30 to 36 -/
def ways_third_segment : ℕ := 7

/-- The total number of possible entries -/
def total_entries : ℕ := ways_first_segment * ways_second_segment * ways_third_segment

/-- The theorem stating the minimum amount of money needed -/
theorem minimum_cost_for_all_entries : 
  entry_cost * total_entries = 2100 := by sorry

end NUMINAMATH_CALUDE_minimum_cost_for_all_entries_l1437_143702


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1437_143712

theorem inequality_solution_range (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 2 * x + 6 * k < 0) ↔ k < Real.sqrt 6 / 6 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1437_143712


namespace NUMINAMATH_CALUDE_max_value_sum_ratios_l1437_143764

theorem max_value_sum_ratios (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a ≤ b) (h2 : b ≤ c) (h3 : c ≤ 2*a) :
  b/a + c/b + a/c ≤ 7/2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_ratios_l1437_143764
