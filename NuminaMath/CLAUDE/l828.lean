import Mathlib

namespace NUMINAMATH_CALUDE_total_fruits_in_baskets_total_fruits_proof_l828_82875

/-- Given a group of 4 fruit baskets, where the first three baskets contain 9 apples, 
    15 oranges, and 14 bananas each, and the fourth basket contains 2 less of each fruit 
    compared to the other baskets, prove that the total number of fruits is 70. -/
theorem total_fruits_in_baskets : ℕ :=
  let apples_per_basket : ℕ := 9
  let oranges_per_basket : ℕ := 15
  let bananas_per_basket : ℕ := 14
  let num_regular_baskets : ℕ := 3
  let fruits_per_regular_basket : ℕ := apples_per_basket + oranges_per_basket + bananas_per_basket
  let fruits_in_regular_baskets : ℕ := fruits_per_regular_basket * num_regular_baskets
  let reduction_in_last_basket : ℕ := 2
  let fruits_in_last_basket : ℕ := fruits_per_regular_basket - (3 * reduction_in_last_basket)
  let total_fruits : ℕ := fruits_in_regular_baskets + fruits_in_last_basket
  70

theorem total_fruits_proof : total_fruits_in_baskets = 70 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_in_baskets_total_fruits_proof_l828_82875


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l828_82863

/-- Given two vectors in R³ that satisfy certain conditions, prove that k = -3/2 --/
theorem parallel_vectors_k_value (a b : ℝ × ℝ × ℝ) (k : ℝ) :
  a = (1, 2, 1) →
  b = (1, 2, 2) →
  ∃ (t : ℝ), t ≠ 0 ∧ (k • a + b) = t • (a - 2 • b) →
  k = -3/2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l828_82863


namespace NUMINAMATH_CALUDE_palindrome_product_sum_l828_82858

/-- A number is a three-digit palindrome if it's between 100 and 999 (inclusive) and reads the same forwards and backwards. -/
def IsThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10) ∧ ((n / 10) % 10 = (n % 100) / 10)

theorem palindrome_product_sum (a b : ℕ) : 
  IsThreeDigitPalindrome a → IsThreeDigitPalindrome b → a * b = 334491 → a + b = 1324 := by
  sorry

end NUMINAMATH_CALUDE_palindrome_product_sum_l828_82858


namespace NUMINAMATH_CALUDE_range_of_m_l828_82843

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x₀ : ℝ, x₀ + (1/3) * m = Real.exp x₀

def q (m : ℝ) : Prop :=
  let a := m
  let b := 5
  let e := Real.sqrt ((a - b) / a)
  (1/2) < e ∧ e < (2/3)

-- Define the theorem
theorem range_of_m (m : ℝ) (h : p m ∧ q m) :
  (20/3 < m ∧ m < 9) ∨ (3 ≤ m ∧ m < 15/4) := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l828_82843


namespace NUMINAMATH_CALUDE_two_color_theorem_l828_82832

/-- A line in a plane --/
structure Line where
  -- We don't need to define the specifics of a line for this problem

/-- A region in a plane --/
structure Region where
  -- We don't need to define the specifics of a region for this problem

/-- A color (we only need two colors) --/
inductive Color
  | A
  | B

/-- A function that determines if two regions are adjacent --/
def adjacent (r1 r2 : Region) : Prop :=
  sorry -- The specific implementation is not important for the statement

/-- A coloring of regions --/
def Coloring := Region → Color

/-- A valid coloring ensures adjacent regions have different colors --/
def valid_coloring (c : Coloring) : Prop :=
  ∀ r1 r2, adjacent r1 r2 → c r1 ≠ c r2

/-- The main theorem --/
theorem two_color_theorem (lines : List Line) :
  ∃ (regions : List Region) (c : Coloring), valid_coloring c :=
sorry

end NUMINAMATH_CALUDE_two_color_theorem_l828_82832


namespace NUMINAMATH_CALUDE_min_value_product_quotient_l828_82870

theorem min_value_product_quotient (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 5*a + 2) * (b^2 + 5*b + 2) * (c^2 + 5*c + 2) / (a*b*c) ≥ 343 ∧
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a^2 + 5*a + 2) * (b^2 + 5*b + 2) * (c^2 + 5*c + 2) / (a*b*c) = 343 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_quotient_l828_82870


namespace NUMINAMATH_CALUDE_min_b_over_a_is_one_minus_e_l828_82877

/-- Given two real functions f and g, if f(x) ≤ g(x) for all x > 0,
    then the minimum value of b/a is 1 - e. -/
theorem min_b_over_a_is_one_minus_e (a b : ℝ)
    (f : ℝ → ℝ) (g : ℝ → ℝ)
    (hf : ∀ x, x > 0 → f x = Real.log x + a)
    (hg : ∀ x, g x = a * x + b + 1)
    (h_le : ∀ x, x > 0 → f x ≤ g x) :
    ∃ m, m = 1 - Real.exp 1 ∧ ∀ k, (b / a ≥ k → k ≥ m) :=
  sorry

end NUMINAMATH_CALUDE_min_b_over_a_is_one_minus_e_l828_82877


namespace NUMINAMATH_CALUDE_sum_cube_over_power_of_two_l828_82892

/-- The sum of the infinite series Σ(k^3 / 2^k) from k=1 to infinity is 26 -/
theorem sum_cube_over_power_of_two : 
  (∑' k : ℕ, (k : ℝ)^3 / 2^k) = 26 := by sorry

end NUMINAMATH_CALUDE_sum_cube_over_power_of_two_l828_82892


namespace NUMINAMATH_CALUDE_last_bead_is_blue_l828_82862

/-- Represents the colors of beads -/
inductive BeadColor
| Red
| Orange
| Yellow
| Green
| Blue
| Purple

/-- Represents the pattern of beads -/
def beadPattern : List BeadColor :=
  [BeadColor.Red, BeadColor.Orange, BeadColor.Yellow, BeadColor.Yellow,
   BeadColor.Green, BeadColor.Blue, BeadColor.Purple]

/-- The total number of beads in the bracelet -/
def totalBeads : Nat := 83

/-- Theorem stating that the last bead of the bracelet is blue -/
theorem last_bead_is_blue :
  (totalBeads % beadPattern.length) = 6 →
  beadPattern[(totalBeads - 1) % beadPattern.length] = BeadColor.Blue :=
by sorry

end NUMINAMATH_CALUDE_last_bead_is_blue_l828_82862


namespace NUMINAMATH_CALUDE_card_exchange_probability_l828_82899

def number_of_people : ℕ := 4

def probability_B_drew_A_given_A_drew_B : ℚ :=
  1 / 3

theorem card_exchange_probability :
  ∀ (A B : Fin number_of_people),
  A ≠ B →
  (probability_B_drew_A_given_A_drew_B : ℚ) =
    (1 : ℚ) / (number_of_people - 1 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_card_exchange_probability_l828_82899


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_specific_l828_82856

theorem gcd_lcm_sum_specific : Nat.gcd 45 4410 + Nat.lcm 45 4410 = 4455 := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_specific_l828_82856


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l828_82839

/-- An isosceles triangle with side lengths 2 and 4 has perimeter 10 -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ), 
  a = 4 → b = 4 → c = 2 →  -- Two sides are 4, one side is 2
  a = b →  -- It's an isosceles triangle
  a + b + c = 10  -- The perimeter is 10
  := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l828_82839


namespace NUMINAMATH_CALUDE_stratified_sampling_size_l828_82884

/-- Represents the sample sizes for three districts -/
structure DistrictSamples where
  d1 : ℕ
  d2 : ℕ
  d3 : ℕ

/-- Given a population divided into three districts with a ratio of 2:3:5,
    and a maximum sample size of 60 for any district,
    prove that the total sample size is 120. -/
theorem stratified_sampling_size :
  ∀ (s : DistrictSamples),
  (s.d1 : ℚ) / 2 = s.d2 / 3 ∧
  (s.d1 : ℚ) / 2 = s.d3 / 5 ∧
  s.d3 ≤ 60 ∧
  s.d3 = 60 →
  s.d1 + s.d2 + s.d3 = 120 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_size_l828_82884


namespace NUMINAMATH_CALUDE_divisors_of_m_squared_count_specific_divisors_l828_82893

def m : ℕ := 2^40 * 5^24

theorem divisors_of_m_squared (d : ℕ) : 
  (d ∣ m^2) ∧ (d < m) ∧ ¬(d ∣ m) ↔ d ∈ Finset.filter (λ x => (x ∣ m^2) ∧ (x < m) ∧ ¬(x ∣ m)) (Finset.range (m + 1)) :=
sorry

theorem count_specific_divisors : 
  Finset.card (Finset.filter (λ x => (x ∣ m^2) ∧ (x < m) ∧ ¬(x ∣ m)) (Finset.range (m + 1))) = 959 :=
sorry

end NUMINAMATH_CALUDE_divisors_of_m_squared_count_specific_divisors_l828_82893


namespace NUMINAMATH_CALUDE_banana_arrangements_l828_82828

theorem banana_arrangements : 
  let total_letters : ℕ := 6
  let repeated_letter1_count : ℕ := 3
  let repeated_letter2_count : ℕ := 2
  let unique_letter_count : ℕ := 1
  (total_letters = repeated_letter1_count + repeated_letter2_count + unique_letter_count) →
  (Nat.factorial total_letters / (Nat.factorial repeated_letter1_count * Nat.factorial repeated_letter2_count) = 60) := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l828_82828


namespace NUMINAMATH_CALUDE_average_marks_combined_classes_l828_82886

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 = 30 →
  n2 = 50 →
  avg1 = 30 →
  avg2 = 60 →
  (n1 : ℚ) * avg1 + (n2 : ℚ) * avg2 = ((n1 + n2) : ℚ) * (48.75 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_average_marks_combined_classes_l828_82886


namespace NUMINAMATH_CALUDE_pencil_and_pen_cost_l828_82826

/-- The cost of a pencil -/
def pencil_cost : ℝ := sorry

/-- The cost of a pen -/
def pen_cost : ℝ := sorry

/-- The first condition: four pencils and three pens cost $3.70 -/
axiom condition1 : 4 * pencil_cost + 3 * pen_cost = 3.70

/-- The second condition: three pencils and four pens cost $4.20 -/
axiom condition2 : 3 * pencil_cost + 4 * pen_cost = 4.20

/-- Theorem: The cost of one pencil and one pen is $1.1286 -/
theorem pencil_and_pen_cost : pencil_cost + pen_cost = 1.1286 := by
  sorry

end NUMINAMATH_CALUDE_pencil_and_pen_cost_l828_82826


namespace NUMINAMATH_CALUDE_proposition_p_or_q_is_true_l828_82805

theorem proposition_p_or_q_is_true : 
  (1 ∈ { x : ℝ | x^2 - 2*x + 1 ≤ 0 }) ∨ (∀ x ∈ (Set.Icc 0 1 : Set ℝ), x^2 - 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_proposition_p_or_q_is_true_l828_82805


namespace NUMINAMATH_CALUDE_sandwiches_needed_l828_82881

theorem sandwiches_needed (total_people children adults : ℕ) 
  (h1 : total_people = 219)
  (h2 : children = 125)
  (h3 : adults = 94)
  (h4 : total_people = children + adults)
  (h5 : children * 4 + adults * 3 = 782) : 
  children * 4 + adults * 3 = 782 := by
  sorry

end NUMINAMATH_CALUDE_sandwiches_needed_l828_82881


namespace NUMINAMATH_CALUDE_frequency_proportion_l828_82821

theorem frequency_proportion (frequency sample_size : ℕ) 
  (h1 : frequency = 80) 
  (h2 : sample_size = 100) : 
  (frequency : ℚ) / sample_size = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_frequency_proportion_l828_82821


namespace NUMINAMATH_CALUDE_total_flowers_sold_l828_82808

/-- Represents the number of flowers in a bouquet -/
def bouquet_size : ℕ := 12

/-- Represents the total number of bouquets sold -/
def total_bouquets : ℕ := 20

/-- Represents the number of rose bouquets sold -/
def rose_bouquets : ℕ := 10

/-- Represents the number of daisy bouquets sold -/
def daisy_bouquets : ℕ := 10

/-- Theorem stating that the total number of flowers sold is 240 -/
theorem total_flowers_sold : 
  bouquet_size * rose_bouquets + bouquet_size * daisy_bouquets = 240 :=
by sorry

end NUMINAMATH_CALUDE_total_flowers_sold_l828_82808


namespace NUMINAMATH_CALUDE_average_of_six_integers_l828_82802

theorem average_of_six_integers (a b c d e f : ℤ) :
  a = 22 ∧ b = 23 ∧ c = 23 ∧ d = 25 ∧ e = 26 ∧ f = 31 →
  (a + b + c + d + e + f) / 6 = 25 := by
sorry

end NUMINAMATH_CALUDE_average_of_six_integers_l828_82802


namespace NUMINAMATH_CALUDE_ellipse_equation_max_distance_max_distance_point_l828_82866

/-- Definition of an ellipse with eccentricity 1/2 passing through (0, √3) -/
def Ellipse (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1) ∧
  (a^2 - b^2) / a^2 = 1/4 ∧ b^2 = 3

/-- The equation of the ellipse is x²/4 + y²/3 = 1 -/
theorem ellipse_equation (x y : ℝ) :
  Ellipse x y ↔ x^2/4 + y^2/3 = 1 :=
sorry

/-- The maximum distance from a point on the ellipse to (0, √3) is 2√3 -/
theorem max_distance :
  ∃ (x₀ y₀ : ℝ), Ellipse x₀ y₀ ∧
  ∀ (x y : ℝ), Ellipse x y →
  (x₀^2 + (y₀ - Real.sqrt 3)^2) ≥ (x^2 + (y - Real.sqrt 3)^2) ∧
  Real.sqrt (x₀^2 + (y₀ - Real.sqrt 3)^2) = 2 * Real.sqrt 3 :=
sorry

/-- The point that maximizes the distance has coordinates (-√3, 0) -/
theorem max_distance_point :
  ∃! (x₀ y₀ : ℝ), Ellipse x₀ y₀ ∧
  ∀ (x y : ℝ), Ellipse x y →
  (x₀^2 + (y₀ - Real.sqrt 3)^2) ≥ (x^2 + (y - Real.sqrt 3)^2) ∧
  x₀ = -Real.sqrt 3 ∧ y₀ = 0 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_max_distance_max_distance_point_l828_82866


namespace NUMINAMATH_CALUDE_largest_y_value_l828_82888

theorem largest_y_value (y : ℝ) : 
  (y / 7 + 2 / (3 * y) = 3) → y ≤ (63 + Real.sqrt 3801) / 6 :=
by sorry

end NUMINAMATH_CALUDE_largest_y_value_l828_82888


namespace NUMINAMATH_CALUDE_smallest_sum_of_quadratic_roots_l828_82895

theorem smallest_sum_of_quadratic_roots (c d : ℝ) : 
  c > 0 → d > 0 → 
  (∃ x : ℝ, x^2 + c*x + 3*d = 0) → 
  (∃ y : ℝ, y^2 + 3*d*y + c = 0) → 
  c + 3*d ≥ 8 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_quadratic_roots_l828_82895


namespace NUMINAMATH_CALUDE_winning_vote_percentage_l828_82810

/-- Given an election with two candidates, prove the percentage of votes for the winning candidate -/
theorem winning_vote_percentage
  (total_votes : ℕ)
  (vote_majority : ℕ)
  (h_total : total_votes = 700)
  (h_majority : vote_majority = 280) :
  (vote_majority : ℚ) / total_votes * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_winning_vote_percentage_l828_82810


namespace NUMINAMATH_CALUDE_newspaper_coupon_free_tickets_l828_82883

/-- Represents the amusement park scenario --/
structure AmusementPark where
  ferris_wheel_cost : ℝ
  roller_coaster_cost : ℝ
  multiple_ride_discount : ℝ
  tickets_bought : ℝ

/-- Calculates the number of free tickets from the newspaper coupon --/
def free_tickets (park : AmusementPark) : ℝ :=
  park.ferris_wheel_cost + park.roller_coaster_cost - park.multiple_ride_discount - park.tickets_bought

/-- Theorem stating that the number of free tickets is 1 given the specific conditions --/
theorem newspaper_coupon_free_tickets :
  ∀ (park : AmusementPark),
    park.ferris_wheel_cost = 2 →
    park.roller_coaster_cost = 7 →
    park.multiple_ride_discount = 1 →
    park.tickets_bought = 7 →
    free_tickets park = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_newspaper_coupon_free_tickets_l828_82883


namespace NUMINAMATH_CALUDE_train_crossing_length_train_B_length_l828_82825

/-- The length of a train crossing another train in opposite direction --/
theorem train_crossing_length (length_A : ℝ) (speed_A speed_B : ℝ) (time : ℝ) : ℝ :=
  let relative_speed := (speed_A + speed_B) * (1000 / 3600)
  let total_distance := relative_speed * time
  total_distance - length_A

/-- Proof of the length of Train B --/
theorem train_B_length : 
  train_crossing_length 360 120 150 6 = 90 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_length_train_B_length_l828_82825


namespace NUMINAMATH_CALUDE_common_difference_is_two_l828_82865

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_arithmetic : ∀ n, a (n + 1) = a n + d
  h_nonzero : d ≠ 0
  h_sum : a 1 + a 2 + a 3 = 9
  h_geometric : ∃ r : ℝ, r ≠ 0 ∧ a 2 = r * a 1 ∧ a 5 = r * a 2

/-- The common difference of the arithmetic sequence is 2 -/
theorem common_difference_is_two (seq : ArithmeticSequence) : seq.d = 2 := by
  sorry

end NUMINAMATH_CALUDE_common_difference_is_two_l828_82865


namespace NUMINAMATH_CALUDE_min_weighings_is_three_l828_82852

/-- Represents a collection of coins with two adjacent lighter coins. -/
structure CoinCollection where
  n : ℕ
  light_weight : ℕ
  heavy_weight : ℕ
  
/-- Represents a weighing operation on a subset of coins. -/
def Weighing (cc : CoinCollection) (subset : Finset ℕ) : ℕ := sorry

/-- The minimum number of weighings required to identify the two lighter coins. -/
def min_weighings (cc : CoinCollection) : ℕ := sorry

/-- Theorem stating that the minimum number of weighings is 3 for any valid coin collection. -/
theorem min_weighings_is_three (cc : CoinCollection) 
  (h1 : cc.n ≥ 2) 
  (h2 : cc.light_weight = 9) 
  (h3 : cc.heavy_weight = 10) :
  min_weighings cc = 3 := by sorry

end NUMINAMATH_CALUDE_min_weighings_is_three_l828_82852


namespace NUMINAMATH_CALUDE_profit_difference_example_l828_82871

/-- Given a total profit and a ratio of division between two parties,
    calculates the difference in their profit shares. -/
def profit_difference (total_profit : ℚ) (ratio_x : ℚ) (ratio_y : ℚ) : ℚ :=
  let total_ratio := ratio_x + ratio_y
  let share_x := (ratio_x / total_ratio) * total_profit
  let share_y := (ratio_y / total_ratio) * total_profit
  share_x - share_y

/-- Theorem stating that for a total profit of 1000 and a ratio of 1/2 : 1/3,
    the difference in profit shares is 200. -/
theorem profit_difference_example :
  profit_difference 1000 (1/2) (1/3) = 200 := by
  sorry

end NUMINAMATH_CALUDE_profit_difference_example_l828_82871


namespace NUMINAMATH_CALUDE_waiter_customers_l828_82833

/-- The number of customers a waiter served before the lunch rush -/
def customers_before_rush : ℕ := 29

/-- The number of additional customers during the lunch rush -/
def additional_customers : ℕ := 20

/-- The number of customers who didn't leave a tip -/
def customers_no_tip : ℕ := 34

/-- The number of customers who left a tip -/
def customers_with_tip : ℕ := 15

theorem waiter_customers :
  customers_before_rush + additional_customers =
  customers_no_tip + customers_with_tip :=
by sorry

end NUMINAMATH_CALUDE_waiter_customers_l828_82833


namespace NUMINAMATH_CALUDE_is_solution_l828_82834

-- Define the function f(x) = x^2 + x + C
def f (C : ℝ) (x : ℝ) : ℝ := x^2 + x + C

-- State the theorem
theorem is_solution (C : ℝ) : 
  ∀ x : ℝ, deriv (f C) x = 2 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_is_solution_l828_82834


namespace NUMINAMATH_CALUDE_at_least_three_babies_speak_l828_82806

def probability_baby_speaks : ℚ := 1 / 3

def number_of_babies : ℕ := 6

def probability_at_least_three_speak (p : ℚ) (n : ℕ) : ℚ :=
  1 - (Nat.choose n 0 * (1 - p)^n + 
       Nat.choose n 1 * p * (1 - p)^(n-1) + 
       Nat.choose n 2 * p^2 * (1 - p)^(n-2))

theorem at_least_three_babies_speak : 
  probability_at_least_three_speak probability_baby_speaks number_of_babies = 353 / 729 := by
  sorry

end NUMINAMATH_CALUDE_at_least_three_babies_speak_l828_82806


namespace NUMINAMATH_CALUDE_work_completion_time_l828_82830

theorem work_completion_time (a b : ℝ) (h1 : a = 30) 
  (h2 : 1 / a + 1 / b = 0.5 / 10) : b = 60 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l828_82830


namespace NUMINAMATH_CALUDE_right_distance_is_73_l828_82846

/-- Represents a square table with a centered round plate -/
structure TableWithPlate where
  /-- Length of the square table's side -/
  table_side : ℝ
  /-- Diameter of the round plate -/
  plate_diameter : ℝ
  /-- Distance from plate edge to left table edge -/
  left_distance : ℝ
  /-- Distance from plate edge to top table edge -/
  top_distance : ℝ
  /-- Distance from plate edge to bottom table edge -/
  bottom_distance : ℝ
  /-- The plate is centered on the table -/
  centered : left_distance + plate_diameter + (table_side - left_distance - plate_diameter) = top_distance + plate_diameter + bottom_distance

/-- Theorem stating the distance from plate edge to right table edge -/
theorem right_distance_is_73 (t : TableWithPlate) (h1 : t.left_distance = 10) (h2 : t.top_distance = 63) (h3 : t.bottom_distance = 20) : 
  t.table_side - t.left_distance - t.plate_diameter = 73 := by
  sorry

end NUMINAMATH_CALUDE_right_distance_is_73_l828_82846


namespace NUMINAMATH_CALUDE_pitcher_problem_l828_82813

theorem pitcher_problem (C : ℝ) (h : C > 0) : 
  let juice_volume : ℝ := (3/4) * C
  let num_cups : ℕ := 5
  let juice_per_cup : ℝ := juice_volume / num_cups
  (juice_per_cup / C) * 100 = 15 := by
sorry

end NUMINAMATH_CALUDE_pitcher_problem_l828_82813


namespace NUMINAMATH_CALUDE_second_half_speed_l828_82838

/-- Proves that given a journey of 300 km completed in 11 hours, where the first half of the distance
    is traveled at 30 kmph, the speed for the second half of the journey is 25 kmph. -/
theorem second_half_speed 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (first_half_speed : ℝ) 
  (h1 : total_distance = 300)
  (h2 : total_time = 11)
  (h3 : first_half_speed = 30)
  : ∃ second_half_speed : ℝ, 
    second_half_speed = 25 ∧ 
    total_distance / 2 / first_half_speed + total_distance / 2 / second_half_speed = total_time :=
by sorry

end NUMINAMATH_CALUDE_second_half_speed_l828_82838


namespace NUMINAMATH_CALUDE_linear_function_value_at_negative_two_l828_82807

/-- A linear function passing through a given point -/
def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x

theorem linear_function_value_at_negative_two 
  (k : ℝ) 
  (h : linear_function k 2 = 4) : 
  linear_function k (-2) = -4 := by
sorry

end NUMINAMATH_CALUDE_linear_function_value_at_negative_two_l828_82807


namespace NUMINAMATH_CALUDE_current_velocity_is_two_l828_82818

-- Define the rowing speed in still water
def still_water_speed : ℝ := 10

-- Define the total time for the round trip
def total_time : ℝ := 15

-- Define the distance to the place
def distance : ℝ := 72

-- Define the velocity of the current as a variable
def current_velocity : ℝ → ℝ := λ v => v

-- Define the equation for the total time of the round trip
def time_equation (v : ℝ) : Prop :=
  distance / (still_water_speed - v) + distance / (still_water_speed + v) = total_time

-- Theorem statement
theorem current_velocity_is_two :
  ∃ v : ℝ, time_equation v ∧ current_velocity v = 2 :=
sorry

end NUMINAMATH_CALUDE_current_velocity_is_two_l828_82818


namespace NUMINAMATH_CALUDE_fleet_capacity_l828_82854

theorem fleet_capacity (num_vans : ℕ) (large_capacity : ℕ) 
  (h_num_vans : num_vans = 6)
  (h_large_capacity : large_capacity = 8000)
  (h_small_capacity : ∃ small_capacity : ℕ, small_capacity = large_capacity - (large_capacity * 30 / 100))
  (h_very_large_capacity : ∃ very_large_capacity : ℕ, very_large_capacity = large_capacity + (large_capacity * 50 / 100))
  (h_num_large : ∃ num_large : ℕ, num_large = 2)
  (h_num_small : ∃ num_small : ℕ, num_small = 1)
  (h_num_very_large : ∃ num_very_large : ℕ, num_very_large = num_vans - 2 - 1) :
  ∃ total_capacity : ℕ, total_capacity = 57600 ∧
    total_capacity = (2 * large_capacity) + 
                     (large_capacity - (large_capacity * 30 / 100)) + 
                     (3 * (large_capacity + (large_capacity * 50 / 100))) :=
by
  sorry

end NUMINAMATH_CALUDE_fleet_capacity_l828_82854


namespace NUMINAMATH_CALUDE_converse_of_proposition_l828_82809

/-- The original proposition -/
def original_proposition (x : ℝ) : Prop := x < 0 → x^2 > 0

/-- The converse of the original proposition -/
def converse_proposition (x : ℝ) : Prop := x^2 > 0 → x < 0

/-- Theorem stating that the converse_proposition is indeed the converse of the original_proposition -/
theorem converse_of_proposition :
  (∀ x, original_proposition x) ↔ (∀ x, converse_proposition x) :=
sorry

end NUMINAMATH_CALUDE_converse_of_proposition_l828_82809


namespace NUMINAMATH_CALUDE_process_600_parts_l828_82822

/-- The regression line equation for processing parts -/
def regression_line (x : ℝ) : ℝ := 0.01 * x + 0.5

/-- Theorem stating that processing 600 parts takes 6.5 hours -/
theorem process_600_parts : regression_line 600 = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_process_600_parts_l828_82822


namespace NUMINAMATH_CALUDE_increasing_decreasing_behavior_l828_82815

theorem increasing_decreasing_behavior 
  (f : ℝ → ℝ) (a : ℝ) (n : ℕ) 
  (h_f : ∀ x, f x = a * x ^ n) 
  (h_a : a ≠ 0) :
  (n % 2 = 0 ∧ a > 0 → ∀ x ≠ 0, deriv f x > 0) ∧
  (n % 2 = 0 ∧ a < 0 → ∀ x ≠ 0, deriv f x < 0) ∧
  (n % 2 = 1 ∧ a > 0 → (∀ x > 0, deriv f x > 0) ∧ (∀ x < 0, deriv f x < 0)) ∧
  (n % 2 = 1 ∧ a < 0 → (∀ x > 0, deriv f x < 0) ∧ (∀ x < 0, deriv f x > 0)) :=
by sorry

end NUMINAMATH_CALUDE_increasing_decreasing_behavior_l828_82815


namespace NUMINAMATH_CALUDE_specific_sampling_problem_l828_82814

/-- Systematic sampling function -/
def systematicSample (totalPopulation sampleSize firstDrawn nthGroup : ℕ) : ℕ :=
  let interval := totalPopulation / sampleSize
  firstDrawn + interval * (nthGroup - 1)

/-- Theorem for the specific sampling problem -/
theorem specific_sampling_problem :
  systematicSample 1000 50 15 21 = 415 := by
  sorry

end NUMINAMATH_CALUDE_specific_sampling_problem_l828_82814


namespace NUMINAMATH_CALUDE_fixed_points_of_f_l828_82811

/-- Definition of the function f -/
def f (x : ℝ) : ℝ := x^2 - x - 3

/-- Definition of a fixed point -/
def is_fixed_point (x : ℝ) : Prop := f x = x

/-- Theorem stating that -1 and 3 are the fixed points of f -/
theorem fixed_points_of_f :
  {x : ℝ | is_fixed_point x} = {-1, 3} := by sorry

end NUMINAMATH_CALUDE_fixed_points_of_f_l828_82811


namespace NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l828_82831

theorem sin_cos_sixth_power_sum (α : ℝ) (h : Real.sin (2 * α) = 1/2) : 
  Real.sin α ^ 6 + Real.cos α ^ 6 = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l828_82831


namespace NUMINAMATH_CALUDE_area_equals_half_radius_times_pedal_perimeter_l828_82889

open Real

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents the pedal triangle of a given triangle -/
def pedalTriangle (T : Triangle) : Triangle := sorry

/-- The area of a triangle -/
def area (T : Triangle) : ℝ := sorry

/-- The perimeter of a triangle -/
def perimeter (T : Triangle) : ℝ := sorry

/-- The circumradius of a triangle -/
def circumradius (T : Triangle) : ℝ := sorry

/-- Predicate to check if a triangle is acute-angled -/
def isAcute (T : Triangle) : Prop := sorry

theorem area_equals_half_radius_times_pedal_perimeter (T : Triangle) 
  (h : isAcute T) : 
  area T = (circumradius T / 2) * perimeter (pedalTriangle T) := by
  sorry

end NUMINAMATH_CALUDE_area_equals_half_radius_times_pedal_perimeter_l828_82889


namespace NUMINAMATH_CALUDE_three_digit_multiples_of_seven_l828_82820

theorem three_digit_multiples_of_seven (n : ℕ) : 
  (∃ k : ℕ, n = 7 * k ∧ 100 ≤ n ∧ n ≤ 999) ↔ n ∈ Finset.range 128 ∧ n ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_three_digit_multiples_of_seven_l828_82820


namespace NUMINAMATH_CALUDE_root_implies_expression_value_l828_82887

theorem root_implies_expression_value (a : ℝ) : 
  (1^2 - 5*a*1 + a^2 = 0) → (3*a^2 - 15*a - 7 = -10) := by
  sorry

end NUMINAMATH_CALUDE_root_implies_expression_value_l828_82887


namespace NUMINAMATH_CALUDE_charles_milk_amount_l828_82882

/-- The amount of chocolate milk in each glass (in ounces) -/
def glass_size : ℝ := 8

/-- The amount of milk in each glass (in ounces) -/
def milk_per_glass : ℝ := 6.5

/-- The amount of chocolate syrup in each glass (in ounces) -/
def syrup_per_glass : ℝ := 1.5

/-- The total amount of chocolate syrup Charles has (in ounces) -/
def total_syrup : ℝ := 60

/-- The total amount of chocolate milk Charles will drink (in ounces) -/
def total_milk : ℝ := 160

/-- Theorem stating that Charles has 130 ounces of milk -/
theorem charles_milk_amount : 
  ∃ (num_glasses : ℝ),
    num_glasses * glass_size = total_milk ∧
    num_glasses * syrup_per_glass ≤ total_syrup ∧
    num_glasses * milk_per_glass = 130 := by
  sorry

end NUMINAMATH_CALUDE_charles_milk_amount_l828_82882


namespace NUMINAMATH_CALUDE_multiplication_mistake_l828_82859

theorem multiplication_mistake (x : ℚ) : (43 * x - 34 * x = 1251) → x = 139 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_mistake_l828_82859


namespace NUMINAMATH_CALUDE_inequality_count_l828_82827

theorem inequality_count (x y a b : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0)
  (hxa : x^2 < a^2) (hyb : y^2 < b^2) : 
  ∃! n : ℕ, n = 2 ∧ 
  (n = (ite (x + y < a + b) 1 0 : ℕ) + 
       (ite (x + y^2 < a + b^2) 1 0 : ℕ) + 
       (ite (x * y < a * b) 1 0 : ℕ) + 
       (ite (|x / y| < |a / b|) 1 0 : ℕ)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_count_l828_82827


namespace NUMINAMATH_CALUDE_tan_fifteen_thirty_product_l828_82857

theorem tan_fifteen_thirty_product : (1 + Real.tan (15 * π / 180)) * (1 + Real.tan (30 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_fifteen_thirty_product_l828_82857


namespace NUMINAMATH_CALUDE_frog_count_frog_count_correct_l828_82890

theorem frog_count (num_crocodiles : ℕ) (total_eyes : ℕ) (frog_eyes : ℕ) (crocodile_eyes : ℕ) : ℕ :=
  let num_frogs := (total_eyes - num_crocodiles * crocodile_eyes) / frog_eyes
  num_frogs

theorem frog_count_correct :
  frog_count 6 52 2 2 = 20 := by sorry

end NUMINAMATH_CALUDE_frog_count_frog_count_correct_l828_82890


namespace NUMINAMATH_CALUDE_functional_equation_solution_l828_82849

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x, f x + 2 * f (1 - x) = 4 * x^2 + 3

/-- Theorem stating that for any function satisfying the functional equation, f(4) = 11/3 -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) : f 4 = 11/3 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l828_82849


namespace NUMINAMATH_CALUDE_meat_for_forty_burgers_l828_82898

/-- The number of pounds of meat needed to make a given number of hamburgers -/
def meatNeeded (initialPounds : ℚ) (initialBurgers : ℕ) (targetBurgers : ℕ) : ℚ :=
  (initialPounds / initialBurgers) * targetBurgers

/-- Theorem stating that 20 pounds of meat are needed for 40 hamburgers
    given that 5 pounds of meat make 10 hamburgers -/
theorem meat_for_forty_burgers :
  meatNeeded 5 10 40 = 20 := by
  sorry

#eval meatNeeded 5 10 40

end NUMINAMATH_CALUDE_meat_for_forty_burgers_l828_82898


namespace NUMINAMATH_CALUDE_binomial_square_coefficient_l828_82879

theorem binomial_square_coefficient (a : ℚ) : 
  (∃ r s : ℚ, ∀ x : ℚ, a * x^2 + 18 * x + 16 = (r * x + s)^2) → 
  a = 81 / 16 := by
sorry

end NUMINAMATH_CALUDE_binomial_square_coefficient_l828_82879


namespace NUMINAMATH_CALUDE_jo_alan_sum_equal_l828_82823

def jo_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_five (x : ℕ) : ℕ :=
  5 * ((x + 2) / 5)

def alan_sum (n : ℕ) : ℕ :=
  List.sum (List.map round_to_nearest_five (List.range n))

theorem jo_alan_sum_equal :
  jo_sum 120 = alan_sum 120 :=
sorry

end NUMINAMATH_CALUDE_jo_alan_sum_equal_l828_82823


namespace NUMINAMATH_CALUDE_alphabetic_sequences_count_l828_82874

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The length of the sequence -/
def sequence_length : ℕ := 2013

/-- The number of alphabetic sequences of given length with letters in alphabetic order -/
def alphabetic_sequences (n : ℕ) : ℕ := Nat.choose (n + alphabet_size - 1) (alphabet_size - 1)

theorem alphabetic_sequences_count : 
  alphabetic_sequences sequence_length = Nat.choose 2038 25 := by sorry

end NUMINAMATH_CALUDE_alphabetic_sequences_count_l828_82874


namespace NUMINAMATH_CALUDE_function_minimum_condition_l828_82860

def f (x a : ℝ) := x^2 - 2*a*x + a

theorem function_minimum_condition (a : ℝ) :
  (∃ x, x < 1 ∧ ∀ y < 1, f y a ≥ f x a) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_function_minimum_condition_l828_82860


namespace NUMINAMATH_CALUDE_min_cards_to_draw_l828_82894

/- Define the number of suits in a deck -/
def num_suits : ℕ := 4

/- Define the number of cards in each suit -/
def cards_per_suit : ℕ := 13

/- Define the number of cards needed in the same suit -/
def cards_needed_same_suit : ℕ := 4

/- Define the number of jokers in the deck -/
def num_jokers : ℕ := 2

/- Theorem: The minimum number of cards to draw to ensure 4 of the same suit is 15 -/
theorem min_cards_to_draw : 
  (num_suits - 1) * (cards_needed_same_suit - 1) + cards_needed_same_suit + num_jokers = 15 := by
  sorry

end NUMINAMATH_CALUDE_min_cards_to_draw_l828_82894


namespace NUMINAMATH_CALUDE_factors_of_x4_plus_16_l828_82845

theorem factors_of_x4_plus_16 (x : ℝ) : 
  (x^4 + 16 : ℝ) = (x^2 - 4*x + 4) * (x^2 + 4*x + 4) := by
  sorry

end NUMINAMATH_CALUDE_factors_of_x4_plus_16_l828_82845


namespace NUMINAMATH_CALUDE_exists_parallelogram_with_unequal_diagonals_l828_82873

-- Define a parallelogram
structure Parallelogram :=
  (points : Fin 4 → ℝ × ℝ)
  (parallel_sides : (points 0 - points 1) = (points 3 - points 2) ∧ 
                    (points 1 - points 2) = (points 0 - points 3))
  (equal_sides : ‖points 0 - points 1‖ = ‖points 2 - points 3‖ ∧ 
                 ‖points 1 - points 2‖ = ‖points 0 - points 3‖)
  (bisecting_diagonals : (points 0 + points 2) / 2 = (points 1 + points 3) / 2)

-- Theorem statement
theorem exists_parallelogram_with_unequal_diagonals :
  ∃ (p : Parallelogram), ‖p.points 0 - p.points 2‖ ≠ ‖p.points 1 - p.points 3‖ :=
sorry

end NUMINAMATH_CALUDE_exists_parallelogram_with_unequal_diagonals_l828_82873


namespace NUMINAMATH_CALUDE_triangle_inequality_from_sum_product_l828_82829

theorem triangle_inequality_from_sum_product (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) → 
  c < a + b ∧ a < b + c ∧ b < c + a := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_from_sum_product_l828_82829


namespace NUMINAMATH_CALUDE_circles_intersect_l828_82853

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := (x-2)^2 + y^2 = 9

-- Define the distance between the centers
def distance_between_centers : ℝ := 2

-- Define the radii of the circles
def radius1 : ℝ := 2
def radius2 : ℝ := 3

-- Theorem stating that the circles are intersecting
theorem circles_intersect :
  distance_between_centers > |radius1 - radius2| ∧
  distance_between_centers < radius1 + radius2 :=
sorry

end NUMINAMATH_CALUDE_circles_intersect_l828_82853


namespace NUMINAMATH_CALUDE_imaginary_sum_zero_l828_82868

theorem imaginary_sum_zero (i : ℂ) (hi : i * i = -1) :
  1 / i + 1 / (i ^ 3) + 1 / (i ^ 5) + 1 / (i ^ 7) = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_sum_zero_l828_82868


namespace NUMINAMATH_CALUDE_seating_arrangements_l828_82897

/-- The number of seating arrangements for four students and two teachers under different conditions. -/
theorem seating_arrangements (n_students : Nat) (n_teachers : Nat) : n_students = 4 ∧ n_teachers = 2 →
  (∃ (arrangements_middle : Nat), arrangements_middle = 48) ∧
  (∃ (arrangements_together : Nat), arrangements_together = 144) ∧
  (∃ (arrangements_separate : Nat), arrangements_separate = 144) := by
  sorry

#check seating_arrangements

end NUMINAMATH_CALUDE_seating_arrangements_l828_82897


namespace NUMINAMATH_CALUDE_strawberry_jelly_amount_l828_82840

theorem strawberry_jelly_amount (total_jelly : ℕ) (blueberry_jelly : ℕ) 
  (h1 : total_jelly = 6310)
  (h2 : blueberry_jelly = 4518) :
  total_jelly - blueberry_jelly = 1792 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_jelly_amount_l828_82840


namespace NUMINAMATH_CALUDE_inequality_is_linear_one_var_l828_82837

/-- A linear inequality with one variable is an inequality of the form ax + b ≤ c or ax + b ≥ c,
    where a, b, and c are constants and x is a variable. -/
def is_linear_inequality_one_var (f : ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ 
  ((∀ x, f x ↔ a * x + b ≤ c) ∨ (∀ x, f x ↔ a * x + b ≥ c))

/-- The inequality 2 - x ≤ 4 -/
def inequality (x : ℝ) : Prop := 2 - x ≤ 4

theorem inequality_is_linear_one_var : is_linear_inequality_one_var inequality := by
  sorry

end NUMINAMATH_CALUDE_inequality_is_linear_one_var_l828_82837


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l828_82880

theorem consecutive_odd_integers_sum (n : ℤ) : 
  (n + (n + 4) = 150) → (n + (n + 2) + (n + 4) = 225) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l828_82880


namespace NUMINAMATH_CALUDE_value_of_a_l828_82885

theorem value_of_a : ∃ (a : ℝ), 
  (∃ (x y : ℝ), 2*x + y = 3*a ∧ x - 2*y = 9*a ∧ x + 3*y = 24) → 
  a = -4 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l828_82885


namespace NUMINAMATH_CALUDE_prob_e_value_l828_82816

-- Define the probability measure
variable (p : Set α → ℝ)

-- Define events e and f
variable (e f : Set α)

-- Define the conditions
variable (h1 : p f = 75)
variable (h2 : p (e ∩ f) = 75)
variable (h3 : p f / p e = 3)

-- Theorem statement
theorem prob_e_value : p e = 25 := by
  sorry

end NUMINAMATH_CALUDE_prob_e_value_l828_82816


namespace NUMINAMATH_CALUDE_point_on_y_axis_l828_82801

theorem point_on_y_axis (x : ℝ) :
  (x^2 - 1 = 0) → 
  ((x^2 - 1, 2*x + 4) = (0, 6) ∨ (x^2 - 1, 2*x + 4) = (0, 2)) :=
by sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l828_82801


namespace NUMINAMATH_CALUDE_smallest_solution_sum_is_five_l828_82842

/-- The sum of divisors function for numbers of the form 2^i * 3^j * 5^k -/
def sum_of_divisors (i j k : ℕ) : ℕ :=
  (2^(i+1) - 1) * ((3^(j+1) - 1)/2) * ((5^(k+1) - 1)/4)

/-- Predicate to check if (i, j, k) is a valid solution -/
def is_valid_solution (i j k : ℕ) : Prop :=
  sum_of_divisors i j k = 360

/-- Predicate to check if (i, j, k) is the smallest valid solution -/
def is_smallest_solution (i j k : ℕ) : Prop :=
  is_valid_solution i j k ∧
  ∀ i' j' k', is_valid_solution i' j' k' → i + j + k ≤ i' + j' + k'

/-- The main theorem: the smallest solution sums to 5 -/
theorem smallest_solution_sum_is_five :
  ∃ i j k, is_smallest_solution i j k ∧ i + j + k = 5 := by sorry

#check smallest_solution_sum_is_five

end NUMINAMATH_CALUDE_smallest_solution_sum_is_five_l828_82842


namespace NUMINAMATH_CALUDE_distance_between_locations_l828_82803

theorem distance_between_locations (s : ℝ) : 
  (s > 0) →  -- Distance is positive
  ((s/2 + 12) / (s/2 - 12) = s / (s - 40)) →  -- Condition when cars meet
  (s = 120) :=  -- Distance to prove
by sorry

end NUMINAMATH_CALUDE_distance_between_locations_l828_82803


namespace NUMINAMATH_CALUDE_rosy_work_days_l828_82848

/-- Given that Mary can do a piece of work in 26 days and Rosy is 30% more efficient than Mary,
    prove that Rosy will take 20 days to do the same piece of work. -/
theorem rosy_work_days (mary_days : ℝ) (rosy_efficiency : ℝ) :
  mary_days = 26 →
  rosy_efficiency = 1.3 →
  (mary_days / rosy_efficiency : ℝ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_rosy_work_days_l828_82848


namespace NUMINAMATH_CALUDE_feathers_needed_for_wings_l828_82872

theorem feathers_needed_for_wings 
  (feathers_per_set : ℕ) 
  (num_sets : ℕ) 
  (charlie_feathers : ℕ) 
  (susan_feathers : ℕ) :
  feathers_per_set = 900 →
  num_sets = 2 →
  charlie_feathers = 387 →
  susan_feathers = 250 →
  feathers_per_set * num_sets - (charlie_feathers + susan_feathers) = 1163 :=
by sorry

end NUMINAMATH_CALUDE_feathers_needed_for_wings_l828_82872


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l828_82850

theorem unique_solution_for_equation (N : ℕ) (a b c : ℕ+) :
  N > 3 →
  Odd N →
  a ^ N = b ^ N + 2 ^ N + a * b * c →
  c ≤ 5 * 2 ^ (N - 1) →
  N = 5 ∧ a = 3 ∧ b = 1 ∧ c = 70 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l828_82850


namespace NUMINAMATH_CALUDE_temperature_change_l828_82804

/-- The temperature change problem -/
theorem temperature_change (initial temp_rise temp_drop final : Int) : 
  initial = -12 → 
  temp_rise = 8 → 
  temp_drop = 10 → 
  final = initial + temp_rise - temp_drop → 
  final = -14 := by sorry

end NUMINAMATH_CALUDE_temperature_change_l828_82804


namespace NUMINAMATH_CALUDE_equation_equality_l828_82817

theorem equation_equality : 5 + (-6) - (-7) = 5 - 6 + 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_l828_82817


namespace NUMINAMATH_CALUDE_regular_2000_pointed_stars_count_l828_82836

theorem regular_2000_pointed_stars_count : ℕ :=
  let n : ℕ := 2000
  let φ : ℕ → ℕ := fun m => Nat.totient m
  let non_similar_count : ℕ := (φ n - 2) / 2
  399

/- Proof
sorry
-/

end NUMINAMATH_CALUDE_regular_2000_pointed_stars_count_l828_82836


namespace NUMINAMATH_CALUDE_expression_value_at_five_l828_82896

theorem expression_value_at_five : 
  let x : ℝ := 5
  (x^3 - 4*x^2 + 3*x) / (x - 3) = 20 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_five_l828_82896


namespace NUMINAMATH_CALUDE_greatest_divisor_of_fourth_power_difference_l828_82878

/-- The function that reverses the digits of a positive integer -/
noncomputable def reverse_digits (n : ℕ+) : ℕ+ := sorry

/-- Theorem stating that 99 is the greatest integer that always divides n^4 - f(n)^4 -/
theorem greatest_divisor_of_fourth_power_difference (n : ℕ+) : 
  (∃ (k : ℕ), k > 99 ∧ ∀ (m : ℕ+), k ∣ (m^4 - (reverse_digits m)^4)) → False :=
sorry

end NUMINAMATH_CALUDE_greatest_divisor_of_fourth_power_difference_l828_82878


namespace NUMINAMATH_CALUDE_x_squared_plus_y_squared_equals_four_l828_82819

theorem x_squared_plus_y_squared_equals_four (x y : ℝ) :
  (x^2 + y^2 + 2) * (x^2 + y^2 - 3) = 6 → x^2 + y^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_y_squared_equals_four_l828_82819


namespace NUMINAMATH_CALUDE_contractor_male_workers_l828_82851

/-- Represents the number of male workers employed by the contractor. -/
def male_workers : ℕ := sorry

/-- Represents the number of female workers employed by the contractor. -/
def female_workers : ℕ := 15

/-- Represents the number of child workers employed by the contractor. -/
def child_workers : ℕ := 5

/-- Represents the daily wage of a male worker in Rupees. -/
def male_wage : ℕ := 35

/-- Represents the daily wage of a female worker in Rupees. -/
def female_wage : ℕ := 20

/-- Represents the daily wage of a child worker in Rupees. -/
def child_wage : ℕ := 8

/-- Represents the average daily wage paid by the contractor in Rupees. -/
def average_wage : ℕ := 26

/-- Theorem stating that the number of male workers employed by the contractor is 20. -/
theorem contractor_male_workers :
  (male_workers * male_wage + female_workers * female_wage + child_workers * child_wage) /
  (male_workers + female_workers + child_workers) = average_wage →
  male_workers = 20 := by
  sorry

end NUMINAMATH_CALUDE_contractor_male_workers_l828_82851


namespace NUMINAMATH_CALUDE_parallel_lines_condition_l828_82861

theorem parallel_lines_condition (a : ℝ) : 
  (∀ x y : ℝ, x + 2*a*y - 1 = 0 ↔ (2*a - 1)*x - a*y - 1 = 0) ↔ (a = 0 ∨ a = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l828_82861


namespace NUMINAMATH_CALUDE_quadratic_function_property_l828_82876

def f (m n x : ℝ) : ℝ := x^2 + m*x + n

theorem quadratic_function_property (m n : ℝ) :
  (∀ x ∈ Set.Icc 1 5, |f m n x| ≤ 2) →
  (f m n 1 - 2*(f m n 3) + f m n 5 = 8) ∧ (m = -6 ∧ n = 7) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l828_82876


namespace NUMINAMATH_CALUDE_lune_area_l828_82891

/-- The area of a lune formed by two semicircles -/
theorem lune_area (r₁ r₂ : ℝ) (h : r₁ = 2 * r₂) : 
  let lune_area := π * r₂^2 / 2 + r₁ * r₂ - π * r₁^2 / 4
  lune_area = 1 - π / 2 := by
  sorry

end NUMINAMATH_CALUDE_lune_area_l828_82891


namespace NUMINAMATH_CALUDE_polynomial_sum_coefficients_l828_82855

theorem polynomial_sum_coefficients : 
  ∀ A B C D E : ℚ, 
  (∀ x : ℚ, (x + 2) * (x + 3) * (3*x^2 - x + 5) = A*x^4 + B*x^3 + C*x^2 + D*x + E) →
  A + B + C + D + E = 84 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_coefficients_l828_82855


namespace NUMINAMATH_CALUDE_increasing_quadratic_parameter_range_l828_82844

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 1

theorem increasing_quadratic_parameter_range (a : ℝ) :
  (∀ x ≥ 2, ∀ y ≥ 2, x < y → f a x < f a y) →
  a ∈ Set.Ici (-2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_increasing_quadratic_parameter_range_l828_82844


namespace NUMINAMATH_CALUDE_wheel_diameter_l828_82867

theorem wheel_diameter (radius : ℝ) (h : radius = 7) : 2 * radius = 14 := by
  sorry

end NUMINAMATH_CALUDE_wheel_diameter_l828_82867


namespace NUMINAMATH_CALUDE_odd_function_value_at_one_l828_82812

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + (a-5)*x^2 + a*x

-- State the theorem
theorem odd_function_value_at_one :
  ∀ a : ℝ, (∀ x : ℝ, f a (-x) = -(f a x)) → f a 1 = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_odd_function_value_at_one_l828_82812


namespace NUMINAMATH_CALUDE_kayak_rental_cost_l828_82847

theorem kayak_rental_cost 
  (canoe_cost : ℝ) 
  (canoe_kayak_ratio : ℚ) 
  (total_revenue : ℝ) 
  (canoe_kayak_difference : ℕ) :
  canoe_cost = 12 →
  canoe_kayak_ratio = 3 / 2 →
  total_revenue = 504 →
  canoe_kayak_difference = 7 →
  ∃ (kayak_cost : ℝ) (num_canoes num_kayaks : ℕ),
    num_canoes = num_kayaks + canoe_kayak_difference ∧
    (num_canoes : ℚ) / num_kayaks = canoe_kayak_ratio ∧
    total_revenue = canoe_cost * num_canoes + kayak_cost * num_kayaks ∧
    kayak_cost = 18 :=
by sorry

end NUMINAMATH_CALUDE_kayak_rental_cost_l828_82847


namespace NUMINAMATH_CALUDE_union_equals_A_A_subset_complement_B_l828_82800

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m - 2 ≤ x ∧ x ≤ m + 2}

-- Theorem 1
theorem union_equals_A (m : ℝ) : A ∪ B m = A → m = 1 := by
  sorry

-- Theorem 2
theorem A_subset_complement_B (m : ℝ) : A ⊆ (B m)ᶜ → m > 5 ∨ m < -3 := by
  sorry

end NUMINAMATH_CALUDE_union_equals_A_A_subset_complement_B_l828_82800


namespace NUMINAMATH_CALUDE_finish_book_in_three_days_l828_82835

def pages_to_read_on_third_day (total_pages : ℕ) (pages_day1 : ℕ) (fewer_pages_day2 : ℕ) : ℕ :=
  total_pages - (pages_day1 + (pages_day1 - fewer_pages_day2))

theorem finish_book_in_three_days (total_pages : ℕ) (pages_day1 : ℕ) (fewer_pages_day2 : ℕ)
  (h1 : total_pages = 100)
  (h2 : pages_day1 = 35)
  (h3 : fewer_pages_day2 = 5) :
  pages_to_read_on_third_day total_pages pages_day1 fewer_pages_day2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_finish_book_in_three_days_l828_82835


namespace NUMINAMATH_CALUDE_parabola_vertex_l828_82864

/-- The vertex of the parabola y = x^2 - 6x + 1 has coordinates (3, -8) -/
theorem parabola_vertex (x y : ℝ) : 
  y = x^2 - 6*x + 1 → ∃ (h k : ℝ), h = 3 ∧ k = -8 ∧ ∀ x, y = (x - h)^2 + k := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l828_82864


namespace NUMINAMATH_CALUDE_sector_angle_l828_82869

/-- A circular sector with arc length and area both equal to 4 has a central angle of 2 radians -/
theorem sector_angle (R : ℝ) (α : ℝ) (h1 : α * R = 4) (h2 : (1/2) * α * R^2 = 4) : α = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l828_82869


namespace NUMINAMATH_CALUDE_mathematics_letter_probability_l828_82824

theorem mathematics_letter_probability : 
  let alphabet_size : ℕ := 26
  let unique_letters_in_mathematics : ℕ := 8
  let probability : ℚ := unique_letters_in_mathematics / alphabet_size
  probability = 4 / 13 := by
sorry

end NUMINAMATH_CALUDE_mathematics_letter_probability_l828_82824


namespace NUMINAMATH_CALUDE_notebook_distribution_l828_82841

theorem notebook_distribution (C : ℕ) (N : ℕ) 
  (h1 : N = C^2 / 8)
  (h2 : N = 8 * (C / 2))
  : N = 512 := by
  sorry

end NUMINAMATH_CALUDE_notebook_distribution_l828_82841
