import Mathlib

namespace NUMINAMATH_CALUDE_vector_b_coordinates_l90_9036

/-- Given a vector a = (-1, 2) and a vector b with magnitude 3√5,
    if the cosine of the angle between a and b is -1,
    then b = (3, -6) -/
theorem vector_b_coordinates (a b : ℝ × ℝ) (θ : ℝ) : 
  a = (-1, 2) →
  ‖b‖ = 3 * Real.sqrt 5 →
  θ = Real.arccos (-1) →
  Real.cos θ = -1 →
  b = (3, -6) := by sorry

end NUMINAMATH_CALUDE_vector_b_coordinates_l90_9036


namespace NUMINAMATH_CALUDE_four_digit_combinations_l90_9026

/-- The number of available digits for each position in a four-digit number -/
def available_digits : Fin 4 → ℕ
  | 0 => 9  -- first digit (cannot be 0)
  | 1 => 8  -- second digit
  | 2 => 6  -- third digit
  | 3 => 4  -- fourth digit

/-- The total number of different four-digit numbers that can be formed -/
def total_combinations : ℕ := (available_digits 0) * (available_digits 1) * (available_digits 2) * (available_digits 3)

theorem four_digit_combinations : total_combinations = 1728 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_combinations_l90_9026


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l90_9068

/-- Given two arithmetic sequences {a_n} and {b_n} with sums S_n and T_n respectively,
    if S_n / T_n = (2n - 3) / (n + 2) for all n, then a_5 / b_5 = 15 / 11 -/
theorem arithmetic_sequence_ratio (a b : ℕ → ℚ) (S T : ℕ → ℚ) 
    (h_arithmetic_a : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
    (h_arithmetic_b : ∀ n, b (n + 1) - b n = b (n + 2) - b (n + 1))
    (h_sum_a : ∀ n, S n = (n : ℚ) * (a 1 + a n) / 2)
    (h_sum_b : ∀ n, T n = (n : ℚ) * (b 1 + b n) / 2)
    (h_ratio : ∀ n, S n / T n = (2 * n - 3) / (n + 2)) :
  a 5 / b 5 = 15 / 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l90_9068


namespace NUMINAMATH_CALUDE_simplify_fraction_l90_9070

theorem simplify_fraction (m : ℝ) (h : m ≠ 3) :
  (m / (m - 3) + 2 / (3 - m)) / ((m - 2) / (m^2 - 6*m + 9)) = m - 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l90_9070


namespace NUMINAMATH_CALUDE_tangent_angle_at_origin_l90_9081

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem tangent_angle_at_origin :
  let x₀ : ℝ := 0
  let y₀ : ℝ := f x₀
  let slope : ℝ := f' x₀
  Real.arctan slope = π / 4 := by sorry

end NUMINAMATH_CALUDE_tangent_angle_at_origin_l90_9081


namespace NUMINAMATH_CALUDE_range_of_a_l90_9080

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 2, x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) → 
  a ≤ -2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l90_9080


namespace NUMINAMATH_CALUDE_equation_solutions_l90_9097

theorem equation_solutions : 
  let f : ℝ → ℝ := λ x => x * (x - 3) - 10
  (f 5 = 0 ∧ f (-2) = 0) ∧ ∀ x : ℝ, f x = 0 → (x = 5 ∨ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l90_9097


namespace NUMINAMATH_CALUDE_fraction_numerator_l90_9012

theorem fraction_numerator (y : ℝ) (h : y > 0) :
  ∃ x : ℝ, (x / y) * y + (3 * y) / 10 = 0.7 * y ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_numerator_l90_9012


namespace NUMINAMATH_CALUDE_chrysanthemum_arrangements_l90_9092

/-- The number of permutations of n distinct objects. -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange k objects out of n distinct objects. -/
def arrangements (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

/-- The number of ways to arrange 6 distinct objects in a row, 
    where two specific objects (A and B) are on the same side of a third specific object (C). -/
theorem chrysanthemum_arrangements : 
  2 * (permutations 5 + 
       arrangements 4 2 * arrangements 3 3 + 
       arrangements 2 2 * arrangements 3 3 + 
       arrangements 3 2 * arrangements 3 3) = 480 := by
  sorry


end NUMINAMATH_CALUDE_chrysanthemum_arrangements_l90_9092


namespace NUMINAMATH_CALUDE_range_of_a_l90_9041

theorem range_of_a (a : ℝ) : 
  (∃ x₁ x₂ x₃ : ℤ, 
    (∀ x : ℝ, (x > 2*a - 3 ∧ 2*x ≥ 3*(x-2) + 5) ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) ∧
    (∀ x : ℤ, x ≠ x₁ ∧ x ≠ x₂ ∧ x ≠ x₃ → ¬(x > 2*a - 3 ∧ 2*x ≥ 3*(x-2) + 5))) →
  (1/2 : ℝ) ≤ a ∧ a < 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l90_9041


namespace NUMINAMATH_CALUDE_prop_1_prop_4_l90_9084

-- Define the types for lines and planes
axiom Line : Type
axiom Plane : Type

-- Define the relations
axiom perpendicular : Line → Plane → Prop
axiom parallel_line_plane : Line → Plane → Prop
axiom parallel_plane : Plane → Plane → Prop
axiom perpendicular_line : Line → Line → Prop

-- Define variables
variable (m n : Line)
variable (α β γ : Plane)

-- Axiom: m and n are different lines
axiom m_neq_n : m ≠ n

-- Axiom: α, β, and γ are different planes
axiom α_neq_β : α ≠ β
axiom α_neq_γ : α ≠ γ
axiom β_neq_γ : β ≠ γ

-- Proposition 1
theorem prop_1 : perpendicular m α → parallel_line_plane n α → perpendicular_line m n :=
sorry

-- Proposition 4
theorem prop_4 : parallel_plane α β → parallel_plane β γ → perpendicular m α → perpendicular m γ :=
sorry

end NUMINAMATH_CALUDE_prop_1_prop_4_l90_9084


namespace NUMINAMATH_CALUDE_min_white_surface_fraction_problem_cube_l90_9047

/-- Represents a cube constructed from smaller cubes -/
structure LargeCube where
  edge_length : ℕ
  total_small_cubes : ℕ
  red_cubes : ℕ
  white_cubes : ℕ

/-- Calculates the minimum fraction of white surface area for a given LargeCube -/
def min_white_surface_fraction (cube : LargeCube) : ℚ :=
  sorry

/-- The specific cube described in the problem -/
def problem_cube : LargeCube :=
  { edge_length := 4
  , total_small_cubes := 64
  , red_cubes := 52
  , white_cubes := 12 }

theorem min_white_surface_fraction_problem_cube :
  min_white_surface_fraction problem_cube = 11 / 96 :=
sorry

end NUMINAMATH_CALUDE_min_white_surface_fraction_problem_cube_l90_9047


namespace NUMINAMATH_CALUDE_chris_teslas_l90_9079

theorem chris_teslas (elon sam chris : ℕ) : 
  elon = 13 →
  elon = sam + 10 →
  sam * 2 = chris →
  chris = 6 := by sorry

end NUMINAMATH_CALUDE_chris_teslas_l90_9079


namespace NUMINAMATH_CALUDE_smallest_multiple_with_divisors_l90_9065

/-- The number of positive integral divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- n is a multiple of m -/
def is_multiple (n m : ℕ) : Prop := ∃ k, n = m * k

theorem smallest_multiple_with_divisors :
  ∃ n : ℕ,
    is_multiple n 75 ∧
    num_divisors n = 36 ∧
    (∀ m : ℕ, is_multiple m 75 → num_divisors m = 36 → n ≤ m) ∧
    n / 75 = 162 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_with_divisors_l90_9065


namespace NUMINAMATH_CALUDE_flower_shop_rearrangement_l90_9085

theorem flower_shop_rearrangement (initial_bunches : ℕ) (initial_flowers_per_bunch : ℕ) (new_flowers_per_bunch : ℕ) :
  initial_bunches = 8 →
  initial_flowers_per_bunch = 9 →
  new_flowers_per_bunch = 12 →
  (initial_bunches * initial_flowers_per_bunch) / new_flowers_per_bunch = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_flower_shop_rearrangement_l90_9085


namespace NUMINAMATH_CALUDE_sharon_coffee_cost_l90_9002

/-- Calculates the total cost of coffee pods for Sharon's vacation -/
def coffee_cost (vacation_days : ℕ) (light_daily : ℕ) (medium_daily : ℕ) (decaf_daily : ℕ)
  (light_box_qty : ℕ) (medium_box_qty : ℕ) (decaf_box_qty : ℕ)
  (light_box_price : ℕ) (medium_box_price : ℕ) (decaf_box_price : ℕ) : ℕ :=
  let light_pods := vacation_days * light_daily
  let medium_pods := vacation_days * medium_daily
  let decaf_pods := vacation_days * decaf_daily
  let light_boxes := (light_pods + light_box_qty - 1) / light_box_qty
  let medium_boxes := (medium_pods + medium_box_qty - 1) / medium_box_qty
  let decaf_boxes := (decaf_pods + decaf_box_qty - 1) / decaf_box_qty
  light_boxes * light_box_price + medium_boxes * medium_box_price + decaf_boxes * decaf_box_price

/-- Theorem stating that the total cost for Sharon's vacation coffee is $80 -/
theorem sharon_coffee_cost :
  coffee_cost 40 2 1 1 20 25 30 10 12 8 = 80 :=
by sorry


end NUMINAMATH_CALUDE_sharon_coffee_cost_l90_9002


namespace NUMINAMATH_CALUDE_volleyball_team_starters_l90_9004

/-- The number of ways to choose k items from n distinguishable items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of players on the volleyball team -/
def total_players : ℕ := 16

/-- The number of starters to be chosen -/
def num_starters : ℕ := 6

/-- Theorem: The number of ways to choose 6 starters from 16 players is 8008 -/
theorem volleyball_team_starters : choose total_players num_starters = 8008 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_starters_l90_9004


namespace NUMINAMATH_CALUDE_not_negative_review_A_two_positive_reviews_out_of_four_l90_9078

-- Define the platforms
inductive Platform
| A
| B

-- Define the review types
inductive Review
| Positive
| Neutral
| Negative

-- Define the function for the number of reviews for each platform and review type
def reviewCount (p : Platform) (r : Review) : ℕ :=
  match p, r with
  | Platform.A, Review.Positive => 75
  | Platform.A, Review.Neutral => 20
  | Platform.A, Review.Negative => 5
  | Platform.B, Review.Positive => 64
  | Platform.B, Review.Neutral => 8
  | Platform.B, Review.Negative => 8

-- Define the total number of reviews for each platform
def totalReviews (p : Platform) : ℕ :=
  reviewCount p Review.Positive + reviewCount p Review.Neutral + reviewCount p Review.Negative

-- Define the probability of a review type for a given platform
def reviewProbability (p : Platform) (r : Review) : ℚ :=
  reviewCount p r / totalReviews p

-- Theorem for the probability of not receiving a negative review for platform A
theorem not_negative_review_A :
  1 - reviewProbability Platform.A Review.Negative = 19/20 := by sorry

-- Theorem for the probability of exactly 2 out of 4 randomly selected buyers giving a positive review
theorem two_positive_reviews_out_of_four :
  let pA := reviewProbability Platform.A Review.Positive
  let pB := reviewProbability Platform.B Review.Positive
  (pA^2 * (1-pB)^2) + (2 * pA * (1-pA) * pB * (1-pB)) + ((1-pA)^2 * pB^2) = 73/400 := by sorry

end NUMINAMATH_CALUDE_not_negative_review_A_two_positive_reviews_out_of_four_l90_9078


namespace NUMINAMATH_CALUDE_expression_evaluation_l90_9095

theorem expression_evaluation : 12 - 7 + 11 * 4 + 8 - 10 * 2 + 6 / 2 - 3 = 34 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l90_9095


namespace NUMINAMATH_CALUDE_opposite_of_neg_abs_two_thirds_l90_9039

theorem opposite_of_neg_abs_two_thirds (m : ℚ) : 
  m = -(-(|-(2/3)|)) → m = 2/3 := by
sorry

end NUMINAMATH_CALUDE_opposite_of_neg_abs_two_thirds_l90_9039


namespace NUMINAMATH_CALUDE_ellen_stuffing_time_l90_9038

/-- Earl's envelope stuffing rate in envelopes per minute -/
def earl_rate : ℝ := 36

/-- Time taken by Earl and Ellen together to stuff 180 envelopes in minutes -/
def combined_time : ℝ := 3

/-- Number of envelopes stuffed by Earl and Ellen together -/
def combined_envelopes : ℝ := 180

/-- Ellen's time to stuff the same number of envelopes as Earl in minutes -/
def ellen_time : ℝ := 1.5

theorem ellen_stuffing_time :
  earl_rate * ellen_time + earl_rate = combined_envelopes / combined_time :=
by sorry

end NUMINAMATH_CALUDE_ellen_stuffing_time_l90_9038


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l90_9069

theorem sufficient_but_not_necessary (m : ℝ) : 
  (m < -2 → ∀ x : ℝ, x^2 - 2*x - m ≠ 0) ∧
  ¬(∀ m : ℝ, (∀ x : ℝ, x^2 - 2*x - m ≠ 0) → m < -2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l90_9069


namespace NUMINAMATH_CALUDE_alice_savings_l90_9008

/-- Alice's savings calculation --/
theorem alice_savings (sales : ℝ) (basic_salary : ℝ) (commission_rate : ℝ) (savings_rate : ℝ) :
  sales = 2500 →
  basic_salary = 240 →
  commission_rate = 0.02 →
  savings_rate = 0.1 →
  (basic_salary + sales * commission_rate) * savings_rate = 29 := by
  sorry

end NUMINAMATH_CALUDE_alice_savings_l90_9008


namespace NUMINAMATH_CALUDE_shipping_cost_formula_l90_9022

/-- The cost function for shipping a package -/
def shippingCost (P : ℕ) : ℕ :=
  if P ≤ 5 then 5 * P + 10 else 5 * P + 5

theorem shipping_cost_formula (P : ℕ) (h : P ≥ 1) :
  (P ≤ 5 → shippingCost P = 5 * P + 10) ∧
  (P > 5 → shippingCost P = 5 * P + 5) := by
  sorry

end NUMINAMATH_CALUDE_shipping_cost_formula_l90_9022


namespace NUMINAMATH_CALUDE_rectangle_formation_count_l90_9021

theorem rectangle_formation_count (h : ℕ) (v : ℕ) : h = 6 → v = 5 → Nat.choose h 2 * Nat.choose v 2 = 150 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_formation_count_l90_9021


namespace NUMINAMATH_CALUDE_fantasy_books_per_day_l90_9053

/-- Proves that the number of fantasy books sold per day is 5 --/
theorem fantasy_books_per_day 
  (fantasy_price : ℝ)
  (literature_price : ℝ)
  (literature_per_day : ℕ)
  (total_earnings : ℝ)
  (h1 : fantasy_price = 4)
  (h2 : literature_price = fantasy_price / 2)
  (h3 : literature_per_day = 8)
  (h4 : total_earnings = 180) :
  ∃ (fantasy_per_day : ℕ), 
    fantasy_per_day * fantasy_price * 5 + 
    literature_per_day * literature_price * 5 = total_earnings ∧ 
    fantasy_per_day = 5 := by
  sorry

end NUMINAMATH_CALUDE_fantasy_books_per_day_l90_9053


namespace NUMINAMATH_CALUDE_quadratic_range_l90_9025

theorem quadratic_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 1 < 0) ↔ a ∈ Set.Iio (-2) ∪ Set.Ioi 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_range_l90_9025


namespace NUMINAMATH_CALUDE_perimeter_ratio_of_squares_l90_9009

theorem perimeter_ratio_of_squares (s1 s2 : Real) (h : s1^2 / s2^2 = 16 / 25) :
  (4 * s1) / (4 * s2) = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_perimeter_ratio_of_squares_l90_9009


namespace NUMINAMATH_CALUDE_toms_family_stay_l90_9037

/-- Calculates the number of days Tom's family stayed at his house -/
def days_at_toms_house (total_people : ℕ) (meals_per_day : ℕ) (plates_per_meal : ℕ) (total_plates_used : ℕ) : ℕ :=
  total_plates_used / (total_people * meals_per_day * plates_per_meal)

/-- Proves that Tom's family stayed for 4 days given the problem conditions -/
theorem toms_family_stay : 
  days_at_toms_house 6 3 2 144 = 4 := by
  sorry

end NUMINAMATH_CALUDE_toms_family_stay_l90_9037


namespace NUMINAMATH_CALUDE_divisor_sum_theorem_l90_9052

def sum_of_geometric_series (a r : ℕ) (n : ℕ) : ℕ := (a * (r^(n+1) - 1)) / (r - 1)

def sum_of_divisors (i j : ℕ) : ℕ := (sum_of_geometric_series 1 2 i) * (sum_of_geometric_series 1 3 j)

theorem divisor_sum_theorem (i j : ℕ) : sum_of_divisors i j = 360 → i + j = 5 := by
  sorry

end NUMINAMATH_CALUDE_divisor_sum_theorem_l90_9052


namespace NUMINAMATH_CALUDE_second_brother_tells_truth_l90_9049

-- Define the type for card suits
inductive Suit
| Hearts
| Diamonds
| Clubs
| Spades

-- Define the type for brothers
inductive Brother
| First
| Second

-- Define the statements made by the brothers
def statement (b : Brother) : Prop :=
  match b with
  | Brother.First => ∀ (s1 s2 : Suit), s1 = s2
  | Brother.Second => ∃ (s1 s2 : Suit), s1 ≠ s2

-- Define the truth-telling property
def tellsTruth (b : Brother) : Prop := statement b

-- Theorem statement
theorem second_brother_tells_truth :
  (∃! (b : Brother), tellsTruth b) →
  (∀ (b1 b2 : Brother), b1 ≠ b2 → (tellsTruth b1 ↔ ¬tellsTruth b2)) →
  tellsTruth Brother.Second :=
by sorry

end NUMINAMATH_CALUDE_second_brother_tells_truth_l90_9049


namespace NUMINAMATH_CALUDE_midpoint_ratio_range_l90_9042

/-- Given two points P and Q on different lines, with midpoint M satisfying certain conditions,
    prove that the ratio of y₀ to x₀ (coordinates of M) is between -1 and -1/3 -/
theorem midpoint_ratio_range (P Q M : ℝ × ℝ) (x₀ y₀ : ℝ) :
  (P.1 + P.2 = 1) →  -- P lies on x + y = 1
  (Q.1 + Q.2 = -3) →  -- Q lies on x + y = -3
  (M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) →  -- M is midpoint of PQ
  (M = (x₀, y₀)) →  -- M has coordinates (x₀, y₀)
  (x₀ - y₀ + 2 < 0) →  -- given condition
  (-1 < y₀ / x₀ ∧ y₀ / x₀ < -1/3) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_ratio_range_l90_9042


namespace NUMINAMATH_CALUDE_line_segment_param_sum_squares_l90_9087

/-- Given a line segment connecting (1, -3) and (6, 4), parameterized by x = pt + q and y = rt + s
    where 0 ≤ t ≤ 1, and t = 0 corresponds to (1, -3), prove that p^2 + q^2 + r^2 + s^2 = 84 -/
theorem line_segment_param_sum_squares (p q r s : ℝ) : 
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ∃ x y : ℝ, x = p * t + q ∧ y = r * t + s) →
  (q = 1 ∧ s = -3) →
  (p + q = 6 ∧ r + s = 4) →
  p^2 + q^2 + r^2 + s^2 = 84 := by
  sorry


end NUMINAMATH_CALUDE_line_segment_param_sum_squares_l90_9087


namespace NUMINAMATH_CALUDE_ratio_of_averages_l90_9015

theorem ratio_of_averages (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : (4 + 20 + x) / 3 = (y + 16) / 2) :
  x / y = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_averages_l90_9015


namespace NUMINAMATH_CALUDE_complement_of_N_in_M_l90_9046

def M : Set ℕ := {0, 1, 2, 3, 4}
def N : Set ℕ := {0, 1, 3}

theorem complement_of_N_in_M :
  M \ N = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_N_in_M_l90_9046


namespace NUMINAMATH_CALUDE_octagon_area_l90_9013

/-- The area of an octagon formed by removing equilateral triangles from the corners of a square -/
theorem octagon_area (square_side : ℝ) (triangle_side : ℝ) : 
  square_side = 1 + Real.sqrt 3 →
  triangle_side = 1 →
  let square_area := square_side ^ 2
  let triangle_area := (Real.sqrt 3 / 4) * triangle_side ^ 2
  let octagon_area := square_area - 4 * triangle_area
  octagon_area = 4 + Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_octagon_area_l90_9013


namespace NUMINAMATH_CALUDE_tax_percentage_calculation_l90_9043

theorem tax_percentage_calculation (paycheck : ℝ) (savings : ℝ) : 
  paycheck = 125 →
  savings = 20 →
  (1 - 0.2) * (1 - (20 : ℝ) / 100) * paycheck = savings →
  (20 : ℝ) / 100 * paycheck = paycheck - ((1 - (20 : ℝ) / 100) * paycheck) :=
by sorry

end NUMINAMATH_CALUDE_tax_percentage_calculation_l90_9043


namespace NUMINAMATH_CALUDE_sum_y_four_times_equals_four_y_l90_9073

theorem sum_y_four_times_equals_four_y (y : ℝ) : y + y + y + y = 4 * y := by
  sorry

end NUMINAMATH_CALUDE_sum_y_four_times_equals_four_y_l90_9073


namespace NUMINAMATH_CALUDE_loan_amount_to_B_l90_9003

/-- Calculates the simple interest given principal, rate, and time -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem loan_amount_to_B (amountToC : ℚ) (timeB timeC : ℚ) (rate : ℚ) (totalInterest : ℚ) :
  amountToC = 3000 →
  timeB = 2 →
  timeC = 4 →
  rate = 8 →
  totalInterest = 1760 →
  ∃ amountToB : ℚ, 
    simpleInterest amountToB rate timeB + simpleInterest amountToC rate timeC = totalInterest ∧
    amountToB = 5000 := by
  sorry

end NUMINAMATH_CALUDE_loan_amount_to_B_l90_9003


namespace NUMINAMATH_CALUDE_seating_arrangement_count_l90_9055

/-- Represents the seating arrangement for two rows of seats. -/
structure SeatingArrangement where
  front_row : ℕ  -- Number of seats in the front row
  back_row : ℕ   -- Number of seats in the back row
  unavailable_front : ℕ  -- Number of unavailable seats in the front row

/-- Calculates the number of seating arrangements for two people. -/
def count_seating_arrangements (s : SeatingArrangement) : ℕ :=
  sorry

/-- The main theorem stating the number of seating arrangements. -/
theorem seating_arrangement_count :
  let s : SeatingArrangement := { front_row := 11, back_row := 12, unavailable_front := 3 }
  count_seating_arrangements s = 346 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_count_l90_9055


namespace NUMINAMATH_CALUDE_distance_between_given_planes_l90_9083

-- Define the planes
def plane1 (x y z : ℝ) : Prop := 3*x - 2*y + 4*z = 12
def plane2 (x y z : ℝ) : Prop := 6*x - 4*y + 8*z = 5

-- Define the distance function between two planes
noncomputable def distance_between_planes : ℝ := sorry

-- Theorem statement
theorem distance_between_given_planes :
  distance_between_planes = 7 * Real.sqrt 29 / 29 := by sorry

end NUMINAMATH_CALUDE_distance_between_given_planes_l90_9083


namespace NUMINAMATH_CALUDE_complex_number_properties_l90_9040

theorem complex_number_properties (z : ℂ) (h : (z - 2*I) / z = 2 + I) : 
  z.im = -1 ∧ Complex.abs z = Real.sqrt 2 ∧ z^6 = -8*I := by sorry

end NUMINAMATH_CALUDE_complex_number_properties_l90_9040


namespace NUMINAMATH_CALUDE_kath_siblings_count_l90_9050

/-- The number of siblings Kath took to the movie -/
def num_siblings : ℕ := 2

/-- The number of friends Kath took to the movie -/
def num_friends : ℕ := 3

/-- The regular admission cost -/
def regular_cost : ℕ := 8

/-- The discount for movies before 6 P.M. -/
def discount : ℕ := 3

/-- The total amount Kath paid for all admissions -/
def total_paid : ℕ := 30

/-- The actual admission cost per person (after discount) -/
def actual_cost : ℕ := regular_cost - discount

theorem kath_siblings_count :
  num_siblings = (total_paid - (num_friends + 1) * actual_cost) / actual_cost :=
sorry

end NUMINAMATH_CALUDE_kath_siblings_count_l90_9050


namespace NUMINAMATH_CALUDE_solve_equation_l90_9086

theorem solve_equation (x y : ℝ) : y = 3 / (5 * x + 4) → y = 2 → x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l90_9086


namespace NUMINAMATH_CALUDE_profit_sharing_theorem_l90_9035

/-- Represents the profit share of a business partner -/
structure ProfitShare where
  ratio : Float
  amount : Float

/-- Calculates the remaining amount after a purchase -/
def remainingAmount (share : ProfitShare) (purchase : Float) : Float :=
  share.amount - purchase

theorem profit_sharing_theorem 
  (mike johnson amy : ProfitShare)
  (mike_purchase amy_purchase : Float)
  (h1 : mike.ratio = 2.5)
  (h2 : johnson.ratio = 5.2)
  (h3 : amy.ratio = 3.8)
  (h4 : johnson.amount = 3120)
  (h5 : mike_purchase = 200)
  (h6 : amy_purchase = 150)
  : remainingAmount mike mike_purchase + remainingAmount amy amy_purchase = 3430 := by
  sorry

end NUMINAMATH_CALUDE_profit_sharing_theorem_l90_9035


namespace NUMINAMATH_CALUDE_sqrt_sqrt_two_power_ten_l90_9090

theorem sqrt_sqrt_two_power_ten : (Real.sqrt ((Real.sqrt 2) ^ 4)) ^ 10 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sqrt_two_power_ten_l90_9090


namespace NUMINAMATH_CALUDE_salt_solution_mixture_l90_9066

theorem salt_solution_mixture : ∀ (x y : ℝ),
  x > 0 ∧ y > 0 ∧ x + y = 90 ∧
  0.05 * x + 0.20 * y = 0.07 * 90 →
  x = 78 ∧ y = 12 :=
by sorry

end NUMINAMATH_CALUDE_salt_solution_mixture_l90_9066


namespace NUMINAMATH_CALUDE_product_increase_by_2016_l90_9018

theorem product_increase_by_2016 : ∃ (a b c : ℕ), 
  (a - 3) * (b - 3) * (c - 3) = a * b * c + 2016 :=
sorry

end NUMINAMATH_CALUDE_product_increase_by_2016_l90_9018


namespace NUMINAMATH_CALUDE_park_group_problem_l90_9014

theorem park_group_problem (girls boys : ℕ) (groups : ℕ) (group_size : ℕ) : 
  girls = 14 → 
  boys = 11 → 
  groups = 3 → 
  group_size = 25 → 
  groups * group_size = girls + boys + (groups * group_size - (girls + boys)) →
  groups * group_size - (girls + boys) = 50 := by
  sorry

#check park_group_problem

end NUMINAMATH_CALUDE_park_group_problem_l90_9014


namespace NUMINAMATH_CALUDE_composite_polynomial_l90_9067

theorem composite_polynomial (n : ℕ) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^3 + 3*n^2 + 6*n + 8 = a * b :=
sorry

end NUMINAMATH_CALUDE_composite_polynomial_l90_9067


namespace NUMINAMATH_CALUDE_bisector_line_l90_9001

/-- Given two lines l₁ and l₂, and a point P, this theorem states that
    the line passing through P and (4, 0) bisects the line segment formed by
    its intersections with l₁ and l₂. -/
theorem bisector_line (P : ℝ × ℝ) (l₁ l₂ : Set (ℝ × ℝ)) :
  P = (0, 1) →
  l₁ = {(x, y) | 2*x + y - 8 = 0} →
  l₂ = {(x, y) | x - 3*y + 10 = 0} →
  ∃ (A B : ℝ × ℝ),
    A ∈ l₁ ∧
    B ∈ l₂ ∧
    (∃ (t : ℝ), A = (1-t) • P + t • (4, 0) ∧ B = (1-t) • (4, 0) + t • P) :=
by sorry

end NUMINAMATH_CALUDE_bisector_line_l90_9001


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l90_9024

/-- The asymptotes of a hyperbola with equation (y^2 / 16) - (x^2 / 9) = 1
    shifted 5 units down along the y-axis are y = ± (4x/3) + 5 -/
theorem hyperbola_asymptotes (x y : ℝ) :
  let shifted_hyperbola := fun y => (y^2 / 16) - (x^2 / 9) = 1
  let asymptote₁ := fun x => (4 * x) / 3 + 5
  let asymptote₂ := fun x => -(4 * x) / 3 + 5
  shifted_hyperbola (y + 5) →
  (y = asymptote₁ x ∨ y = asymptote₂ x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l90_9024


namespace NUMINAMATH_CALUDE_least_number_with_remainders_l90_9011

theorem least_number_with_remainders : ∃ n : ℕ, 
  (n > 0) ∧ 
  (n % 41 = 5) ∧ 
  (n % 23 = 5) ∧ 
  (∀ m : ℕ, m > 0 → m % 41 = 5 → m % 23 = 5 → m ≥ n) ∧
  n = 948 := by
sorry

end NUMINAMATH_CALUDE_least_number_with_remainders_l90_9011


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_four_l90_9056

theorem sqrt_difference_equals_four :
  Real.sqrt (9 + 4 * Real.sqrt 5) - Real.sqrt (9 - 4 * Real.sqrt 5) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_four_l90_9056


namespace NUMINAMATH_CALUDE_song_book_cost_l90_9034

def flute_cost : ℚ := 142.46
def tool_cost : ℚ := 8.89
def total_spent : ℚ := 158.35

theorem song_book_cost : 
  total_spent - (flute_cost + tool_cost) = 7 := by sorry

end NUMINAMATH_CALUDE_song_book_cost_l90_9034


namespace NUMINAMATH_CALUDE_box_volume_l90_9016

-- Define the set of possible volumes
def possibleVolumes : Set ℕ := {180, 240, 300, 360, 450}

-- Theorem statement
theorem box_volume (a b c : ℕ) (h1 : a > 0 ∧ b > 0 ∧ c > 0) 
  (h2 : ∃ (k : ℕ), k > 0 ∧ a = 2 * k ∧ b = 3 * k ∧ c = 5 * k) :
  (a * b * c) ∈ possibleVolumes ↔ a * b * c = 240 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_l90_9016


namespace NUMINAMATH_CALUDE_leak_drain_time_l90_9077

/-- Given a pump that can fill a tank in 2 hours without a leak,
    and takes 2 1/7 hours to fill the tank with a leak,
    the time it takes for the leak to drain the entire tank is 30 hours. -/
theorem leak_drain_time (fill_time_no_leak fill_time_with_leak : ℚ) : 
  fill_time_no_leak = 2 →
  fill_time_with_leak = 2 + 1 / 7 →
  (1 / (1 / fill_time_no_leak - 1 / fill_time_with_leak)) = 30 := by
  sorry

end NUMINAMATH_CALUDE_leak_drain_time_l90_9077


namespace NUMINAMATH_CALUDE_vanessa_music_files_l90_9010

/-- The number of music files Vanessa initially had -/
def initial_music_files : ℕ := 13

/-- The number of video files Vanessa initially had -/
def video_files : ℕ := 30

/-- The number of files deleted -/
def deleted_files : ℕ := 10

/-- The number of files remaining after deletion -/
def remaining_files : ℕ := 33

theorem vanessa_music_files :
  initial_music_files + video_files = remaining_files + deleted_files :=
by sorry

end NUMINAMATH_CALUDE_vanessa_music_files_l90_9010


namespace NUMINAMATH_CALUDE_james_future_age_l90_9064

/-- Represents the ages and relationships of Justin, Jessica, and James -/
structure FamilyAges where
  justin_age : ℕ
  jessica_age_when_justin_born : ℕ
  james_age_diff_from_jessica : ℕ
  james_age_in_five_years : ℕ

/-- Calculates James' age after a given number of years -/
def james_age_after_years (f : FamilyAges) (years : ℕ) : ℕ :=
  f.james_age_in_five_years - 5 + years

/-- Theorem stating James' age after some years -/
theorem james_future_age (f : FamilyAges) (x : ℕ) :
  f.justin_age = 26 →
  f.jessica_age_when_justin_born = 6 →
  f.james_age_diff_from_jessica = 7 →
  f.james_age_in_five_years = 44 →
  james_age_after_years f x = 39 + x :=
by
  sorry

end NUMINAMATH_CALUDE_james_future_age_l90_9064


namespace NUMINAMATH_CALUDE_max_elevation_is_650_l90_9000

/-- The elevation function of a ball thrown vertically upward -/
def s (t : ℝ) : ℝ := 100 * t - 4 * t^2 + 25

/-- Theorem: The maximum elevation reached by the ball is 650 feet -/
theorem max_elevation_is_650 : 
  ∃ t_max : ℝ, ∀ t : ℝ, s t ≤ s t_max ∧ s t_max = 650 := by
  sorry

end NUMINAMATH_CALUDE_max_elevation_is_650_l90_9000


namespace NUMINAMATH_CALUDE_double_root_values_l90_9019

theorem double_root_values (b₃ b₂ b₁ : ℤ) (r : ℤ) :
  (∀ x : ℝ, x^4 + b₃ * x^3 + b₂ * x^2 + b₁ * x + 72 = (x - r : ℝ)^2 * ((x - r)^2 + c * (x - r) + d))
  → (r = -6 ∨ r = -3 ∨ r = -2 ∨ r = -1 ∨ r = 1 ∨ r = 2 ∨ r = 3 ∨ r = 6) :=
by sorry

end NUMINAMATH_CALUDE_double_root_values_l90_9019


namespace NUMINAMATH_CALUDE_electric_bicycle_sales_l90_9060

theorem electric_bicycle_sales (sales_A_Q1 : ℝ) (sales_BC_Q1 : ℝ) (a : ℝ) :
  sales_A_Q1 = 0.56 ∧
  sales_BC_Q1 = 1 - sales_A_Q1 ∧
  sales_A_Q1 * 1.23 + sales_BC_Q1 * (1 - a / 100) = 1.12 →
  a = 2 :=
by sorry

end NUMINAMATH_CALUDE_electric_bicycle_sales_l90_9060


namespace NUMINAMATH_CALUDE_min_value_xy_expression_min_value_achievable_l90_9074

theorem min_value_xy_expression (x y : ℝ) : (x * y + 1)^2 + (x - y)^2 ≥ 1 := by sorry

theorem min_value_achievable : ∃ x y : ℝ, (x * y + 1)^2 + (x - y)^2 = 1 := by sorry

end NUMINAMATH_CALUDE_min_value_xy_expression_min_value_achievable_l90_9074


namespace NUMINAMATH_CALUDE_greatest_drop_in_april_l90_9030

/-- Represents the months from January to June -/
inductive Month
  | January
  | February
  | March
  | April
  | May
  | June

/-- The price change for each month -/
def price_change (m : Month) : ℝ :=
  match m with
  | Month.January  => -1.00
  | Month.February => 2.50
  | Month.March    => 0.00
  | Month.April    => -3.00
  | Month.May      => -1.50
  | Month.June     => 1.00

/-- A month has a price drop if its price change is negative -/
def has_price_drop (m : Month) : Prop :=
  price_change m < 0

/-- The greatest monthly drop in price occurred in April -/
theorem greatest_drop_in_april :
  ∀ m : Month, has_price_drop m → price_change Month.April ≤ price_change m :=
by sorry

end NUMINAMATH_CALUDE_greatest_drop_in_april_l90_9030


namespace NUMINAMATH_CALUDE_cube_volume_7cm_l90_9075

-- Define the edge length of the cube
def edge_length : ℝ := 7

-- Define the volume of a cube
def cube_volume (edge : ℝ) : ℝ := edge ^ 3

-- Theorem statement
theorem cube_volume_7cm :
  cube_volume edge_length = 343 := by sorry

end NUMINAMATH_CALUDE_cube_volume_7cm_l90_9075


namespace NUMINAMATH_CALUDE_toy_store_restocking_l90_9027

theorem toy_store_restocking (initial_games : ℕ) (sold_games : ℕ) (final_games : ℕ)
  (h1 : initial_games = 95)
  (h2 : sold_games = 68)
  (h3 : final_games = 74) :
  final_games - (initial_games - sold_games) = 47 := by
  sorry

end NUMINAMATH_CALUDE_toy_store_restocking_l90_9027


namespace NUMINAMATH_CALUDE_common_power_theorem_l90_9006

theorem common_power_theorem (a b x y : ℕ) : 
  a > 1 → b > 1 → x > 1 → y > 1 → 
  Nat.gcd a b = 1 → 
  x^a = y^b → 
  ∃ n : ℕ, n > 1 ∧ x = n^b ∧ y = n^a := by
sorry

end NUMINAMATH_CALUDE_common_power_theorem_l90_9006


namespace NUMINAMATH_CALUDE_apple_collection_l90_9098

theorem apple_collection (A K : ℕ) (hA : A > 0) (hK : K > 0) : 
  let T := A + K
  (A = (K * 100) / T) → (K = (A * 100) / T) → (A = 50 ∧ K = 50) :=
by sorry

end NUMINAMATH_CALUDE_apple_collection_l90_9098


namespace NUMINAMATH_CALUDE_bakery_payment_l90_9048

theorem bakery_payment (bun_price croissant_price : ℕ) 
  (h1 : bun_price = 15) (h2 : croissant_price = 12) : 
  (¬ ∃ x y : ℕ, croissant_price * x + bun_price * y = 500) ∧
  (∃ x y : ℕ, croissant_price * x + bun_price * y = 600) := by
sorry

end NUMINAMATH_CALUDE_bakery_payment_l90_9048


namespace NUMINAMATH_CALUDE_restaurant_profit_l90_9089

/-- The profit calculated with mistakes -/
def mistaken_profit : ℕ := 1320

/-- The difference in hundreds place due to the mistake -/
def hundreds_difference : ℕ := 8 - 3

/-- The difference in tens place due to the mistake -/
def tens_difference : ℕ := 8 - 5

/-- The actual profit of the restaurant -/
def actual_profit : ℕ := mistaken_profit - hundreds_difference * 100 + tens_difference * 10

theorem restaurant_profit : actual_profit = 850 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_profit_l90_9089


namespace NUMINAMATH_CALUDE_fischer_random_chess_positions_l90_9054

/-- Represents the number of squares on one row of a chessboard -/
def boardSize : Nat := 8

/-- Represents the number of dark (or light) squares on one row -/
def darkSquares : Nat := boardSize / 2

/-- Represents the number of squares available for queen and knights after placing bishops -/
def remainingSquares : Nat := boardSize - 2

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : Nat) : Nat :=
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Represents the number of ways to arrange bishops on opposite colors -/
def bishopArrangements : Nat := darkSquares * darkSquares

/-- Represents the number of ways to choose positions for queen and knights -/
def queenKnightPositions : Nat := choose remainingSquares 3

/-- Represents the number of permutations for queen and knights -/
def queenKnightPermutations : Nat := Nat.factorial 3

/-- Represents the total number of ways to arrange queen and knights -/
def queenKnightArrangements : Nat := queenKnightPositions * queenKnightPermutations

/-- Represents the number of ways to arrange king between rooks -/
def kingRookArrangements : Nat := 1

/-- The main theorem stating the number of starting positions in Fischer Random Chess -/
theorem fischer_random_chess_positions :
  bishopArrangements * queenKnightArrangements * kingRookArrangements = 1920 := by
  sorry


end NUMINAMATH_CALUDE_fischer_random_chess_positions_l90_9054


namespace NUMINAMATH_CALUDE_base_27_number_divisibility_l90_9072

theorem base_27_number_divisibility (n : ℕ) : 
  (∃ (a b c d e f g h i j k l m o p q r s t u v w x y z : ℕ),
    (∀ digit ∈ [a, b, c, d, e, f, g, h, i, j, k, l, m, o, p, q, r, s, t, u, v, w, x, y, z], 
      1 ≤ digit ∧ digit ≤ 26) ∧
    n = a * 27^25 + b * 27^24 + c * 27^23 + d * 27^22 + e * 27^21 + f * 27^20 + 
        g * 27^19 + h * 27^18 + i * 27^17 + j * 27^16 + k * 27^15 + l * 27^14 + 
        m * 27^13 + o * 27^12 + p * 27^11 + q * 27^10 + r * 27^9 + s * 27^8 + 
        t * 27^7 + u * 27^6 + v * 27^5 + w * 27^4 + x * 27^3 + y * 27^2 + z * 27^1 + 26) →
  n % 100 = 0 := by
sorry

end NUMINAMATH_CALUDE_base_27_number_divisibility_l90_9072


namespace NUMINAMATH_CALUDE_max_of_min_is_sqrt_two_l90_9076

theorem max_of_min_is_sqrt_two (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (⨅ z ∈ ({x, 1/y, y + 1/x} : Set ℝ), z) ≤ Real.sqrt 2 ∧
  ∃ x y, x > 0 ∧ y > 0 ∧ (⨅ z ∈ ({x, 1/y, y + 1/x} : Set ℝ), z) = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_max_of_min_is_sqrt_two_l90_9076


namespace NUMINAMATH_CALUDE_tan_value_for_given_sin_cos_sum_l90_9017

theorem tan_value_for_given_sin_cos_sum (θ : Real) 
  (h1 : Real.sin θ + Real.cos θ = Real.sqrt 5 / 5)
  (h2 : θ ∈ Set.Icc 0 Real.pi) : 
  Real.tan θ = -2 := by
sorry

end NUMINAMATH_CALUDE_tan_value_for_given_sin_cos_sum_l90_9017


namespace NUMINAMATH_CALUDE_arcsin_one_half_eq_pi_sixth_l90_9029

theorem arcsin_one_half_eq_pi_sixth : Real.arcsin (1/2) = π/6 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_one_half_eq_pi_sixth_l90_9029


namespace NUMINAMATH_CALUDE_divisibility_condition_l90_9082

theorem divisibility_condition (n : ℕ+) : (n + 1) ∣ (n^2 + 1) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l90_9082


namespace NUMINAMATH_CALUDE_max_value_of_a_l90_9057

theorem max_value_of_a (a b c d : ℤ) 
  (h1 : a < 2*b) 
  (h2 : b < 3*c) 
  (h3 : c < 4*d) 
  (h4 : d < 100) : 
  a ≤ 2367 ∧ ∃ (a₀ b₀ c₀ d₀ : ℤ), a₀ = 2367 ∧ a₀ < 2*b₀ ∧ b₀ < 3*c₀ ∧ c₀ < 4*d₀ ∧ d₀ < 100 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l90_9057


namespace NUMINAMATH_CALUDE_no_snow_probability_l90_9093

theorem no_snow_probability (p1 p2 p3 p4 : ℚ) 
  (h1 : p1 = 2/3) (h2 : p2 = 3/4) (h3 : p3 = 5/6) (h4 : p4 = 1/2) :
  (1 - p1) * (1 - p2) * (1 - p3) * (1 - p4) = 1/144 := by
  sorry

end NUMINAMATH_CALUDE_no_snow_probability_l90_9093


namespace NUMINAMATH_CALUDE_p_true_and_q_false_l90_9099

-- Define proposition p
def p : Prop := ∀ z : ℂ, (z - Complex.I) * (-Complex.I) = 5 → z = 6 * Complex.I

-- Define proposition q
def q : Prop := Complex.im ((1 + Complex.I) / (1 + 2 * Complex.I)) = -1/5

-- Theorem to prove
theorem p_true_and_q_false : p ∧ ¬q := by sorry

end NUMINAMATH_CALUDE_p_true_and_q_false_l90_9099


namespace NUMINAMATH_CALUDE_inequality_solution_l90_9044

theorem inequality_solution (x : ℝ) (h : x ≥ 0) :
  2021 * (x^2020)^(1/202) - 1 ≥ 2020 * x ↔ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l90_9044


namespace NUMINAMATH_CALUDE_l_shaped_area_l90_9045

/-- The area of the L-shaped region in a square arrangement --/
theorem l_shaped_area (large_square_side : ℝ) (small_square1 : ℝ) (small_square2 : ℝ) (small_square3 : ℝ) (small_square4 : ℝ)
  (h1 : large_square_side = 7)
  (h2 : small_square1 = 2)
  (h3 : small_square2 = 3)
  (h4 : small_square3 = 2)
  (h5 : small_square4 = 1) :
  large_square_side ^ 2 - (small_square1 ^ 2 + small_square2 ^ 2 + small_square3 ^ 2 + small_square4 ^ 2) = 31 := by
  sorry

end NUMINAMATH_CALUDE_l_shaped_area_l90_9045


namespace NUMINAMATH_CALUDE_reduced_rate_start_time_l90_9062

/-- The fraction of a week during which reduced rates apply -/
def reduced_rate_fraction : ℝ := 0.6428571428571429

/-- The number of hours in a week -/
def hours_per_week : ℕ := 24 * 7

/-- The number of weekend days (Saturday and Sunday) -/
def weekend_days : ℕ := 2

/-- The time (in hours) when reduced rates end on weekdays -/
def reduced_rate_end : ℕ := 8

theorem reduced_rate_start_time :
  ∃ (start_time : ℕ),
    start_time = 20 ∧  -- 8 p.m. is 20 in 24-hour format
    (1 - reduced_rate_fraction) * hours_per_week =
      (5 * (reduced_rate_end + (24 - start_time))) +
      (weekend_days * 24) :=
by sorry

end NUMINAMATH_CALUDE_reduced_rate_start_time_l90_9062


namespace NUMINAMATH_CALUDE_unique_integer_solution_l90_9063

theorem unique_integer_solution : ∃! n : ℤ, n + 15 > 16 ∧ -3*n > -9 :=
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l90_9063


namespace NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circles_l90_9059

/-- The area of a rectangle with two circles of radius 7 cm inscribed in opposite corners is 196 cm². -/
theorem rectangle_area_with_inscribed_circles (r : ℝ) (h : r = 7) :
  let diameter := 2 * r
  let length := diameter
  let width := diameter
  let area := length * width
  area = 196 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circles_l90_9059


namespace NUMINAMATH_CALUDE_impossibleTransformation_l90_9023

/-- Represents the state of a cell in the table -/
inductive CellState
  | Zero
  | One

/-- Represents the n × n table -/
def Table (n : ℕ) := Fin n → Fin n → CellState

/-- The initial table state with n-1 ones and the rest zeros -/
def initialTable (n : ℕ) : Table n := sorry

/-- The operation of choosing a cell, subtracting one from it,
    and adding one to all other numbers in the same row or column -/
def applyOperation (t : Table n) (row col : Fin n) : Table n := sorry

/-- Checks if all cells in the table have the same value -/
def allEqual (t : Table n) : Prop := sorry

/-- The main theorem stating that it's impossible to transform the initial table
    into a table with all equal numbers using the given operations -/
theorem impossibleTransformation (n : ℕ) :
  ¬ ∃ (ops : List (Fin n × Fin n)), allEqual (ops.foldl (λ t (rc : Fin n × Fin n) => applyOperation t rc.1 rc.2) (initialTable n)) :=
sorry

end NUMINAMATH_CALUDE_impossibleTransformation_l90_9023


namespace NUMINAMATH_CALUDE_x_value_proof_l90_9061

theorem x_value_proof (x : ℝ) (h : 9 / (x^2) = x / 36) : x = (324 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l90_9061


namespace NUMINAMATH_CALUDE_inverse_sum_theorem_l90_9032

-- Define the function g(x) = x³
def g (x : ℝ) : ℝ := x^3

-- State the theorem
theorem inverse_sum_theorem : g⁻¹ 8 + g⁻¹ (-64) = -2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_theorem_l90_9032


namespace NUMINAMATH_CALUDE_root_in_interval_l90_9058

def f (x : ℝ) := x^3 - x - 3

theorem root_in_interval :
  ∃ x ∈ Set.Ioo 1 2, f x = 0 := by
sorry

end NUMINAMATH_CALUDE_root_in_interval_l90_9058


namespace NUMINAMATH_CALUDE_complex_cube_roots_sum_l90_9071

theorem complex_cube_roots_sum (a b c : ℂ) 
  (sum_condition : a + b + c = 3)
  (sum_squares_condition : a^2 + b^2 + c^2 = 3)
  (sum_cubes_condition : a^3 + b^3 + c^3 = 6) :
  (a - 1)^2023 + (b - 1)^2023 + (c - 1)^2023 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_roots_sum_l90_9071


namespace NUMINAMATH_CALUDE_crayon_difference_l90_9028

theorem crayon_difference (willy_crayons lucy_crayons : ℕ) 
  (h1 : willy_crayons = 1400) 
  (h2 : lucy_crayons = 290) : 
  willy_crayons - lucy_crayons = 1110 := by
  sorry

end NUMINAMATH_CALUDE_crayon_difference_l90_9028


namespace NUMINAMATH_CALUDE_log_greater_than_reciprocal_l90_9051

theorem log_greater_than_reciprocal (x : ℝ) (h : x > 0) : Real.log (1 + x) > 1 / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_log_greater_than_reciprocal_l90_9051


namespace NUMINAMATH_CALUDE_cookies_left_l90_9007

/-- The number of cookies in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of cookies Meena starts with -/
def initial_dozens : ℕ := 5

/-- The number of dozens of cookies sold to Mr. Stone -/
def sold_to_stone : ℕ := 2

/-- The number of cookies bought by Brock -/
def bought_by_brock : ℕ := 7

/-- Katy buys twice as many cookies as Brock -/
def bought_by_katy : ℕ := 2 * bought_by_brock

/-- The theorem stating that Meena has 15 cookies left -/
theorem cookies_left : 
  initial_dozens * dozen - sold_to_stone * dozen - bought_by_brock - bought_by_katy = 15 := by
  sorry


end NUMINAMATH_CALUDE_cookies_left_l90_9007


namespace NUMINAMATH_CALUDE_sam_bought_nine_cans_l90_9096

/-- The number of coupons Sam had -/
def num_coupons : ℕ := 5

/-- The discount per coupon in cents -/
def discount_per_coupon : ℕ := 25

/-- The amount Sam paid in cents -/
def amount_paid : ℕ := 2000

/-- The change Sam received in cents -/
def change_received : ℕ := 550

/-- The cost of each can of tuna in cents -/
def cost_per_can : ℕ := 175

/-- The number of cans Sam bought -/
def num_cans : ℕ := (amount_paid - change_received + num_coupons * discount_per_coupon) / cost_per_can

theorem sam_bought_nine_cans : num_cans = 9 := by
  sorry

end NUMINAMATH_CALUDE_sam_bought_nine_cans_l90_9096


namespace NUMINAMATH_CALUDE_afternoon_flier_fraction_l90_9033

theorem afternoon_flier_fraction (total_fliers : ℕ) (morning_fraction : ℚ) (left_for_next_day : ℕ) :
  total_fliers = 3000 →
  morning_fraction = 1 / 5 →
  left_for_next_day = 1800 →
  let morning_sent := total_fliers * morning_fraction
  let remaining_after_morning := total_fliers - morning_sent
  let afternoon_sent := remaining_after_morning - left_for_next_day
  afternoon_sent / remaining_after_morning = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_afternoon_flier_fraction_l90_9033


namespace NUMINAMATH_CALUDE_book_length_l90_9088

theorem book_length (total_pages : ℕ) : 
  (2 : ℚ) / 3 * total_pages = (1 : ℚ) / 3 * total_pages + 20 → 
  total_pages = 60 := by
sorry

end NUMINAMATH_CALUDE_book_length_l90_9088


namespace NUMINAMATH_CALUDE_tank_width_l90_9094

/-- The width of a tank given its dimensions and plastering costs -/
theorem tank_width (length : ℝ) (depth : ℝ) (plaster_rate : ℝ) (total_cost : ℝ) 
  (h1 : length = 25)
  (h2 : depth = 6)
  (h3 : plaster_rate = 0.75)
  (h4 : total_cost = 558)
  (h5 : total_cost = plaster_rate * (length * width + 2 * length * depth + 2 * width * depth)) :
  width = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_tank_width_l90_9094


namespace NUMINAMATH_CALUDE_probability_no_three_consecutive_ones_l90_9020

/-- Recurrence relation for sequences without three consecutive 1s -/
def b : ℕ → ℕ
| 0 => 1
| 1 => 2
| 2 => 4
| n + 3 => b (n + 2) + b (n + 1) + b n

/-- Total number of possible sequences of length n -/
def total_sequences (n : ℕ) : ℕ := 2^n

/-- The probability of a sequence of length 12 not containing three consecutive 1s -/
theorem probability_no_three_consecutive_ones : 
  (b 12 : ℚ) / total_sequences 12 = 927 / 4096 :=
sorry

end NUMINAMATH_CALUDE_probability_no_three_consecutive_ones_l90_9020


namespace NUMINAMATH_CALUDE_max_value_expression_l90_9031

theorem max_value_expression (a b c d : Real) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) : 
  (∀ x y z w, 0 ≤ x ∧ x ≤ 1 → 0 ≤ y ∧ y ≤ 1 → 0 ≤ z ∧ z ≤ 1 → 0 ≤ w ∧ w ≤ 1 → 
    x + y + z + w - x*y - y*z - z*w - w*x ≤ a + b + c + d - a*b - b*c - c*d - d*a) → 
  a + b + c + d - a*b - b*c - c*d - d*a = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l90_9031


namespace NUMINAMATH_CALUDE_erica_pie_percentage_l90_9005

theorem erica_pie_percentage :
  ∀ (apple_fraction cherry_fraction : ℚ),
    apple_fraction = 1/5 →
    cherry_fraction = 3/4 →
    (apple_fraction + cherry_fraction) * 100 = 95 := by
  sorry

end NUMINAMATH_CALUDE_erica_pie_percentage_l90_9005


namespace NUMINAMATH_CALUDE_inverse_f_at_486_l90_9091

/-- Given a function f with the properties f(5) = 2 and f(3x) = 3f(x) for all x,
    prove that the inverse function f⁻¹ evaluated at 486 is equal to 1215. -/
theorem inverse_f_at_486 (f : ℝ → ℝ) (h1 : f 5 = 2) (h2 : ∀ x, f (3 * x) = 3 * f x) :
  Function.invFun f 486 = 1215 := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_at_486_l90_9091
