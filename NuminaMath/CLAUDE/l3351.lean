import Mathlib

namespace NUMINAMATH_CALUDE_expand_and_simplify_l3351_335196

theorem expand_and_simplify (x : ℝ) : 3 * (x - 4) * (x + 9) = 3 * x^2 + 15 * x - 108 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l3351_335196


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l3351_335102

/-- The total surface area of a right cylinder with height 8 and radius 3 is 66π -/
theorem cylinder_surface_area :
  let h : ℝ := 8
  let r : ℝ := 3
  let lateral_area := 2 * π * r * h
  let base_area := π * r^2
  let total_area := lateral_area + 2 * base_area
  total_area = 66 * π := by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l3351_335102


namespace NUMINAMATH_CALUDE_floor_of_7_9_l3351_335120

theorem floor_of_7_9 : ⌊(7.9 : ℝ)⌋ = 7 := by sorry

end NUMINAMATH_CALUDE_floor_of_7_9_l3351_335120


namespace NUMINAMATH_CALUDE_fold_triangle_crease_length_l3351_335115

theorem fold_triangle_crease_length 
  (A B C : ℝ × ℝ) 
  (h_right_triangle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0)
  (h_side_AB : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 3)
  (h_side_BC : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 4)
  (h_side_AC : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 5) :
  let D : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let E : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
  let F : ℝ × ℝ := 
    let m := (B.2 - A.2) / (B.1 - A.1)
    let b := D.2 - m * D.1
    ((E.2 - b) / m, E.2)
  Real.sqrt ((F.1 - D.1)^2 + (F.2 - D.2)^2) = 15/8 := by
sorry

end NUMINAMATH_CALUDE_fold_triangle_crease_length_l3351_335115


namespace NUMINAMATH_CALUDE_slower_train_speed_l3351_335106

/-- Proves that the speed of the slower train is 36 km/hr given the problem conditions -/
theorem slower_train_speed (faster_speed : ℝ) (passing_time : ℝ) (train_length : ℝ)
  (h1 : faster_speed = 46)
  (h2 : passing_time = 54)
  (h3 : train_length = 75) :
  ∃ (slower_speed : ℝ),
    slower_speed = 36 ∧
    (faster_speed - slower_speed) * passing_time * (1000 / 3600) = 2 * train_length :=
by
  sorry

#check slower_train_speed

end NUMINAMATH_CALUDE_slower_train_speed_l3351_335106


namespace NUMINAMATH_CALUDE_f_derivative_l3351_335190

noncomputable def f (x : ℝ) : ℝ := Real.log x + Real.cos x

theorem f_derivative (x : ℝ) (h : x > 0) : 
  deriv f x = 1 / x - Real.sin x := by sorry

end NUMINAMATH_CALUDE_f_derivative_l3351_335190


namespace NUMINAMATH_CALUDE_place_eight_among_twelve_l3351_335108

/-- The number of ways to place black balls among white balls without adjacency. -/
def place_balls (white : ℕ) (black : ℕ) : ℕ :=
  Nat.choose (white + 1) black

/-- Theorem: Placing 8 black balls among 12 white balls without adjacency. -/
theorem place_eight_among_twelve :
  place_balls 12 8 = 1287 := by
  sorry

end NUMINAMATH_CALUDE_place_eight_among_twelve_l3351_335108


namespace NUMINAMATH_CALUDE_football_purchase_problem_l3351_335113

/-- Represents the cost and quantity of footballs --/
structure FootballPurchase where
  costA : ℝ  -- Cost of one A brand football
  costB : ℝ  -- Cost of one B brand football
  quantityB : ℕ  -- Quantity of B brand footballs purchased

/-- Theorem statement for the football purchase problem --/
theorem football_purchase_problem (fp : FootballPurchase) : 
  fp.costB = fp.costA + 30 ∧ 
  2 * fp.costA + 3 * fp.costB = 340 ∧ 
  fp.quantityB ≤ 50 ∧
  54 * (50 - fp.quantityB) + 72 * fp.quantityB = 3060 →
  fp.quantityB = 20 := by
  sorry

#check football_purchase_problem

end NUMINAMATH_CALUDE_football_purchase_problem_l3351_335113


namespace NUMINAMATH_CALUDE_apples_left_is_ten_l3351_335191

/-- The number of apples left in the cafeteria -/
def apples_left : ℕ := sorry

/-- The initial number of apples -/
def initial_apples : ℕ := 50

/-- The initial number of oranges -/
def initial_oranges : ℕ := 40

/-- The cost of an apple in cents -/
def apple_cost : ℕ := 80

/-- The cost of an orange in cents -/
def orange_cost : ℕ := 50

/-- The total earnings from apples and oranges in cents -/
def total_earnings : ℕ := 4900

/-- The number of oranges left -/
def oranges_left : ℕ := 6

/-- Theorem stating that the number of apples left is 10 -/
theorem apples_left_is_ten :
  apples_left = 10 ∧
  initial_apples * apple_cost - apples_left * apple_cost +
  (initial_oranges - oranges_left) * orange_cost = total_earnings :=
sorry

end NUMINAMATH_CALUDE_apples_left_is_ten_l3351_335191


namespace NUMINAMATH_CALUDE_fruit_crate_total_l3351_335160

theorem fruit_crate_total (strawberry_count : ℕ) (kiwi_fraction : ℚ) 
  (h1 : kiwi_fraction = 1/3)
  (h2 : strawberry_count = 52) :
  ∃ (total : ℕ), total = 78 ∧ 
    strawberry_count = (1 - kiwi_fraction) * total ∧
    kiwi_fraction * total + strawberry_count = total := by
  sorry

end NUMINAMATH_CALUDE_fruit_crate_total_l3351_335160


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_greater_than_one_l3351_335133

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*x + a ≥ 0}

-- Theorem statement
theorem intersection_nonempty_implies_a_greater_than_one (a : ℝ) :
  (∃ x, x ∈ A ∩ B a) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_greater_than_one_l3351_335133


namespace NUMINAMATH_CALUDE_not_all_naturals_equal_l3351_335114

-- Define the statement we want to disprove
def all_naturals_equal (n : ℕ) : Prop :=
  ∀ (a : ℕ → ℕ), (∀ i j, i < n → j < n → a i = a j)

-- Theorem stating that the above statement is false
theorem not_all_naturals_equal : ¬ (∀ n : ℕ, all_naturals_equal n) := by
  sorry

-- Note: The proof is omitted (replaced with 'sorry') as per the instructions

end NUMINAMATH_CALUDE_not_all_naturals_equal_l3351_335114


namespace NUMINAMATH_CALUDE_teena_yoe_distance_l3351_335198

/-- Calculates the initial distance between two drivers given their speeds and future relative position --/
def initialDistance (teenaSpeed yoeSpeed : ℝ) (timeAhead : ℝ) (distanceAhead : ℝ) : ℝ :=
  (teenaSpeed - yoeSpeed) * timeAhead - distanceAhead

theorem teena_yoe_distance :
  let teenaSpeed : ℝ := 55
  let yoeSpeed : ℝ := 40
  let timeAhead : ℝ := 1.5  -- 90 minutes in hours
  let distanceAhead : ℝ := 15
  initialDistance teenaSpeed yoeSpeed timeAhead distanceAhead = 7.5 := by
  sorry

#eval initialDistance 55 40 1.5 15

end NUMINAMATH_CALUDE_teena_yoe_distance_l3351_335198


namespace NUMINAMATH_CALUDE_pipe_a_fill_time_l3351_335181

/-- Given a tank and three pipes A, B, and C, proves that pipe A alone takes 42 hours to fill the tank. -/
theorem pipe_a_fill_time (fill_rate_a fill_rate_b fill_rate_c : ℚ) : 
  (fill_rate_a + fill_rate_b + fill_rate_c = 1 / 6) →  -- Combined rate fills tank in 6 hours
  (fill_rate_c = 2 * fill_rate_b) →  -- Pipe C is twice as fast as pipe B
  (fill_rate_b = 2 * fill_rate_a) →  -- Pipe B is twice as fast as pipe A
  (1 / fill_rate_a = 42) :=  -- Pipe A alone takes 42 hours
by sorry

end NUMINAMATH_CALUDE_pipe_a_fill_time_l3351_335181


namespace NUMINAMATH_CALUDE_cubic_roots_relation_l3351_335157

/-- Given a cubic equation x^3 + px^2 + qx + r = 0 with roots α, β, γ, 
    returns a function that computes expressions involving these roots -/
def cubicRootRelations (p q r : ℝ) : 
  (ℝ → ℝ → ℝ → ℝ) → ℝ := sorry

theorem cubic_roots_relation (a b c s t : ℝ) : 
  cubicRootRelations 3 4 (-11) (fun x y z => x) = a ∧
  cubicRootRelations 3 4 (-11) (fun x y z => y) = b ∧
  cubicRootRelations 3 4 (-11) (fun x y z => z) = c ∧
  cubicRootRelations (-2) s t (fun x y z => x) = a + b ∧
  cubicRootRelations (-2) s t (fun x y z => y) = b + c ∧
  cubicRootRelations (-2) s t (fun x y z => z) = c + a →
  s = 8 ∧ t = 23 := by sorry

end NUMINAMATH_CALUDE_cubic_roots_relation_l3351_335157


namespace NUMINAMATH_CALUDE_trig_expression_value_l3351_335182

/-- The value of the trigonometric expression is approximately 1.481 -/
theorem trig_expression_value : 
  let expr := (2 * Real.sin (20 * π / 180) * Real.cos (10 * π / 180) + 
               Real.cos (160 * π / 180) * Real.cos (110 * π / 180)) /
              (Real.sin (24 * π / 180) * Real.cos (6 * π / 180) + 
               Real.cos (156 * π / 180) * Real.cos (94 * π / 180))
  ∃ ε > 0, |expr - 1.481| < ε :=
by sorry

end NUMINAMATH_CALUDE_trig_expression_value_l3351_335182


namespace NUMINAMATH_CALUDE_ice_cream_combinations_l3351_335139

theorem ice_cream_combinations (n_flavors m_toppings : ℕ) 
  (h_flavors : n_flavors = 5) 
  (h_toppings : m_toppings = 7) : 
  n_flavors * Nat.choose m_toppings 3 = 175 := by
  sorry

#check ice_cream_combinations

end NUMINAMATH_CALUDE_ice_cream_combinations_l3351_335139


namespace NUMINAMATH_CALUDE_common_point_of_circumcircles_l3351_335121

-- Define the circle S
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define a point being outside a circle
def IsOutside (p : ℝ × ℝ) (s : Set (ℝ × ℝ)) : Prop :=
  p ∉ s

-- Define a line passing through a point
def LineThroughPoint (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | ∃ (t : ℝ), q = (p.1 + t, p.2 + t)}

-- Define the intersection of a line and a circle
def Intersect (l : Set (ℝ × ℝ)) (s : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p ∈ l ∧ p ∈ s}

-- Define the circumcircle of a triangle
def Circumcircle (p1 p2 p3 : ℝ × ℝ) : Set (ℝ × ℝ) :=
  sorry -- Actual definition would be more complex

-- Main theorem
theorem common_point_of_circumcircles
  (S : Set (ℝ × ℝ)) (center : ℝ × ℝ) (radius : ℝ)
  (A B : ℝ × ℝ) :
  S = Circle center radius →
  IsOutside A S →
  IsOutside B S →
  ∃ (C : ℝ × ℝ), C ≠ B ∧
    ∀ (l : Set (ℝ × ℝ)) (M N : ℝ × ℝ),
      l = LineThroughPoint A →
      {M, N} ⊆ Intersect l S →
      C ∈ Circumcircle B M N :=
by sorry

end NUMINAMATH_CALUDE_common_point_of_circumcircles_l3351_335121


namespace NUMINAMATH_CALUDE_at_hash_calculation_l3351_335164

/-- Operation @ for positive integers -/
def at_op (a b : ℕ+) : ℚ := (a.val * b.val : ℚ) / (a.val + b.val)

/-- Operation # for rationals -/
def hash_op (c d : ℚ) : ℚ := c + d

/-- Main theorem -/
theorem at_hash_calculation :
  hash_op (at_op 3 7) 4 = 61 / 10 := by sorry

end NUMINAMATH_CALUDE_at_hash_calculation_l3351_335164


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l3351_335104

theorem concentric_circles_ratio (r R k : ℝ) (hr : r > 0) (hR : R > r) (hk : k > 0) :
  (π * R^2 - π * r^2) = k * (π * r^2) → R / r = Real.sqrt (k + 1) := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_ratio_l3351_335104


namespace NUMINAMATH_CALUDE_unique_integer_product_of_digits_l3351_335193

/-- Given a positive integer n, returns the product of its digits -/
def productOfDigits (n : ℕ+) : ℕ := sorry

/-- Theorem: The only positive integer n whose product of digits equals n^2 - 15n - 27 is 17 -/
theorem unique_integer_product_of_digits : 
  ∃! (n : ℕ+), productOfDigits n = n^2 - 15*n - 27 ∧ n = 17 := by sorry

end NUMINAMATH_CALUDE_unique_integer_product_of_digits_l3351_335193


namespace NUMINAMATH_CALUDE_min_value_function_l3351_335127

theorem min_value_function (x : ℝ) (h : x > 3) :
  1 / (x - 3) + x ≥ 5 ∧ (1 / (x - 3) + x = 5 ↔ x = 4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_function_l3351_335127


namespace NUMINAMATH_CALUDE_max_handshakes_equals_combinations_l3351_335187

/-- The number of men in the group -/
def n : ℕ := 20

/-- The number of men involved in each handshake -/
def k : ℕ := 2

/-- Calculates the number of combinations of k items from n items -/
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Theorem: The maximum number of unique pairwise handshakes among n men is equal to the number of combinations of k=2 men from n men -/
theorem max_handshakes_equals_combinations :
  combinations n k = 190 := by sorry

end NUMINAMATH_CALUDE_max_handshakes_equals_combinations_l3351_335187


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3351_335101

/-- An arithmetic sequence with positive common ratio -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  q : ℝ
  h_positive : q > 0
  h_arithmetic : ∀ n : ℕ, a (n + 1) = a n * q

/-- The problem statement -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h1 : seq.a 2 * seq.a 6 = 8 * seq.a 4)
  (h2 : seq.a 2 = 2) :
  seq.a 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3351_335101


namespace NUMINAMATH_CALUDE_optimal_price_reduction_and_profit_l3351_335140

/-- Represents the daily profit function for a flower shop -/
def profit_function (x : ℝ) : ℝ := -2 * x^2 + 60 * x + 800

/-- Represents the constraints on the price reduction -/
def valid_price_reduction (x : ℝ) : Prop := 0 ≤ x ∧ x < 40

/-- Theorem stating the optimal price reduction and maximum profit -/
theorem optimal_price_reduction_and_profit :
  ∃ (x : ℝ), valid_price_reduction x ∧ 
    (∀ y, valid_price_reduction y → profit_function y ≤ profit_function x) ∧
    x = 15 ∧ profit_function x = 1250 :=
sorry

end NUMINAMATH_CALUDE_optimal_price_reduction_and_profit_l3351_335140


namespace NUMINAMATH_CALUDE_identity_proof_special_case_proof_l3351_335166

-- Define the sequence f_n = a^n + b^n
def f (a b : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => a^(n+1) + b^(n+1)

theorem identity_proof (a b : ℝ) (n : ℕ) :
  f a b (n + 1) = (a + b) * (f a b n) - a * b * (f a b (n - 1)) :=
by sorry

theorem special_case_proof (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = -1) :
  f a b 10 = 123 :=
by sorry

end NUMINAMATH_CALUDE_identity_proof_special_case_proof_l3351_335166


namespace NUMINAMATH_CALUDE_power_difference_equals_one_l3351_335154

theorem power_difference_equals_one (a m n : ℝ) (h1 : a^m = 8) (h2 : a^n = 2) :
  a^(m - 3*n) = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_equals_one_l3351_335154


namespace NUMINAMATH_CALUDE_last_four_digits_of_5_to_2017_l3351_335118

theorem last_four_digits_of_5_to_2017 :
  ∃ n : ℕ, 5^2017 ≡ 3125 [ZMOD 10000] :=
by
  -- We define the cycle of last four digits
  let cycle := [3125, 5625, 8125, 0625]
  
  -- We state that 5^5, 5^6, and 5^7 match the first three elements of the cycle
  have h1 : 5^5 ≡ cycle[0] [ZMOD 10000] := by sorry
  have h2 : 5^6 ≡ cycle[1] [ZMOD 10000] := by sorry
  have h3 : 5^7 ≡ cycle[2] [ZMOD 10000] := by sorry
  
  -- We state that the cycle repeats every 4 terms
  have h_cycle : ∀ k : ℕ, 5^(k+4) ≡ 5^k [ZMOD 10000] := by sorry
  
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_last_four_digits_of_5_to_2017_l3351_335118


namespace NUMINAMATH_CALUDE_least_integer_greater_than_negative_eighteen_fifths_l3351_335145

theorem least_integer_greater_than_negative_eighteen_fifths :
  ∃ n : ℤ, n > -18/5 ∧ ∀ m : ℤ, m > -18/5 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_least_integer_greater_than_negative_eighteen_fifths_l3351_335145


namespace NUMINAMATH_CALUDE_square_area_ratio_l3351_335131

theorem square_area_ratio : 
  ∀ (s₂ : ℝ), s₂ > 0 →
  let s₁ := s₂ * Real.sqrt 2
  let s₃ := s₁ / 2
  let A₂ := s₂ ^ 2
  let A₃ := s₃ ^ 2
  A₃ / A₂ = 1 / 2 := by
    sorry

end NUMINAMATH_CALUDE_square_area_ratio_l3351_335131


namespace NUMINAMATH_CALUDE_simplify_sqrt_18_l3351_335167

theorem simplify_sqrt_18 : Real.sqrt 18 = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_18_l3351_335167


namespace NUMINAMATH_CALUDE_not_divides_power_minus_one_l3351_335130

theorem not_divides_power_minus_one (n : ℕ) (h : n ≥ 2) : ¬(n ∣ 2^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_not_divides_power_minus_one_l3351_335130


namespace NUMINAMATH_CALUDE_rent_expenditure_l3351_335195

theorem rent_expenditure (x : ℝ) 
  (h1 : x + 0.7 * x + 32 = 100) : x = 40 := by
  sorry

end NUMINAMATH_CALUDE_rent_expenditure_l3351_335195


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_relation_l3351_335171

theorem isosceles_triangle_angle_relation (A B C C₁ C₂ θ : Real) :
  -- Isosceles triangle condition
  A = B →
  -- Altitude divides angle C into C₁ and C₂
  A + C₁ = 90 →
  B + C₂ = 90 →
  -- External angle θ
  θ = 30 →
  θ = A + B →
  -- Conclusion
  C₁ = 75 ∧ C₂ = 75 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angle_relation_l3351_335171


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l3351_335142

def digit_sum (n : ℕ) : ℕ := 
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem smallest_prime_with_digit_sum_23 :
  ∃ (p : ℕ), is_prime p ∧ digit_sum p = 23 ∧
  ∀ (q : ℕ), is_prime q → digit_sum q = 23 → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l3351_335142


namespace NUMINAMATH_CALUDE_intersection_equals_A_l3351_335107

def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B : Set ℝ := {x | x ≥ 1}

theorem intersection_equals_A : A ∩ B = A := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_A_l3351_335107


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l3351_335158

theorem consecutive_integers_average (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 →
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 →
  (a + b + c + d + e) / 5 = 8 →
  e - a = 4 →
  (b + d) / 2 = 8 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l3351_335158


namespace NUMINAMATH_CALUDE_a7_equals_one_l3351_335165

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem a7_equals_one (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 * a 13 = 1 →
  a 1 + a 13 = 8 →
  a 7 = 1 :=
by sorry

end NUMINAMATH_CALUDE_a7_equals_one_l3351_335165


namespace NUMINAMATH_CALUDE_binomial_minimum_sum_reciprocals_l3351_335116

/-- A discrete random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 < p
  h2 : p < 1

/-- The expectation of a binomial random variable -/
def expectation (X : BinomialRV) : ℝ := X.n * X.p

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_minimum_sum_reciprocals (X : BinomialRV) (q : ℝ) 
    (h_expect : expectation X = 4)
    (h_var : variance X = q) :
    (∀ p q, p > 0 → q > 0 → 1/p + 1/q ≥ 9/4) ∧ 
    (∃ p q, p > 0 ∧ q > 0 ∧ 1/p + 1/q = 9/4) := by
  sorry

end NUMINAMATH_CALUDE_binomial_minimum_sum_reciprocals_l3351_335116


namespace NUMINAMATH_CALUDE_circle_equation_l3351_335148

-- Define the center of the circle
def center : ℝ × ℝ := (-1, 2)

-- Define the radius of the circle
def radius : ℝ := 2

-- State the theorem
theorem circle_equation (x y : ℝ) :
  ((x - center.1)^2 + (y - center.2)^2 = radius^2) ↔
  ((x + 1)^2 + (y - 2)^2 = 4) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l3351_335148


namespace NUMINAMATH_CALUDE_equilibrium_portion_above_water_l3351_335110

/-- Represents a uniform rod partially submerged in water -/
structure PartiallySubmergedRod where
  /-- Length of the rod -/
  length : ℝ
  /-- Density of the rod -/
  density : ℝ
  /-- Density of water -/
  water_density : ℝ
  /-- Portion of the rod above water -/
  above_water_portion : ℝ

/-- Theorem stating the equilibrium condition for a partially submerged rod -/
theorem equilibrium_portion_above_water (rod : PartiallySubmergedRod)
  (h_positive_length : rod.length > 0)
  (h_density_ratio : rod.density = (5 / 9) * rod.water_density)
  (h_equilibrium : rod.above_water_portion * rod.length * rod.water_density * (rod.length / 2) =
                   (1 - rod.above_water_portion) * rod.length * rod.density * (rod.length / 2)) :
  rod.above_water_portion = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equilibrium_portion_above_water_l3351_335110


namespace NUMINAMATH_CALUDE_smallest_k_sum_squares_div_180_l3351_335168

/-- Sum of squares from 1 to n -/
def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- Predicate for a number being divisible by 180 -/
def divisible_by_180 (n : ℕ) : Prop := ∃ m : ℕ, n = 180 * m

theorem smallest_k_sum_squares_div_180 :
  (∀ k < 216, ¬(divisible_by_180 (sum_of_squares k))) ∧
  (divisible_by_180 (sum_of_squares 216)) := by sorry

end NUMINAMATH_CALUDE_smallest_k_sum_squares_div_180_l3351_335168


namespace NUMINAMATH_CALUDE_shopping_mall_problem_l3351_335153

/-- Represents the price of product A in yuan -/
def price_A : ℝ := 16

/-- Represents the price of product B in yuan -/
def price_B : ℝ := 4

/-- Represents the maximum number of product A that can be purchased -/
def max_A : ℕ := 41

theorem shopping_mall_problem :
  (20 * price_A + 15 * price_B = 380) ∧
  (15 * price_A + 10 * price_B = 280) ∧
  (∀ x : ℕ, x ≤ 100 → 
    (x * price_A + (100 - x) * price_B ≤ 900 → x ≤ max_A)) ∧
  (max_A * price_A + (100 - max_A) * price_B ≤ 900) :=
by sorry

end NUMINAMATH_CALUDE_shopping_mall_problem_l3351_335153


namespace NUMINAMATH_CALUDE_student_number_factor_l3351_335141

theorem student_number_factor : ∃ f : ℚ, 122 * f - 138 = 106 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_student_number_factor_l3351_335141


namespace NUMINAMATH_CALUDE_cube_root_simplification_l3351_335152

theorem cube_root_simplification :
  (20^3 + 30^3 + 50^3 : ℝ)^(1/3) = 10 * 160^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l3351_335152


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l3351_335146

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∃ (a : ℚ), x = Real.sqrt a ∧ ∀ (b : ℚ), b ≠ a → Real.sqrt b ≠ x

theorem simplest_quadratic_radical :
  let options : List ℝ := [1 / Real.sqrt 3, Real.sqrt (5 / 6), Real.sqrt 24, Real.sqrt 21]
  ∀ y ∈ options, is_simplest_quadratic_radical (Real.sqrt 21) ∧ 
    (is_simplest_quadratic_radical y → y = Real.sqrt 21) :=
by sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l3351_335146


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_l3351_335175

-- Factorization of 4x^2 - 16
theorem factorization_1 (x : ℝ) : 4 * x^2 - 16 = 4 * (x - 2) * (x + 2) := by sorry

-- Factorization of a^2b - 4ab + 4b
theorem factorization_2 (a b : ℝ) : a^2 * b - 4 * a * b + 4 * b = b * (a - 2)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_l3351_335175


namespace NUMINAMATH_CALUDE_unique_numbers_proof_l3351_335184

theorem unique_numbers_proof (a b : ℕ) : 
  a ≠ b →                 -- The numbers are distinct
  a > 11 →                -- a is greater than 11
  b > 11 →                -- b is greater than 11
  a + b = 28 →            -- Their sum is 28
  (Even a ∨ Even b) →     -- At least one of them is even
  ((a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12)) := by
sorry

end NUMINAMATH_CALUDE_unique_numbers_proof_l3351_335184


namespace NUMINAMATH_CALUDE_house_width_calculation_l3351_335138

/-- Given a house with length 20.5 feet, a porch measuring 6 feet by 4.5 feet,
    and a total shingle area of 232 square feet, the width of the house is 10 feet. -/
theorem house_width_calculation (house_length porch_length porch_width total_shingle_area : ℝ)
    (h1 : house_length = 20.5)
    (h2 : porch_length = 6)
    (h3 : porch_width = 4.5)
    (h4 : total_shingle_area = 232) :
    (total_shingle_area - porch_length * porch_width) / house_length = 10 := by
  sorry

#check house_width_calculation

end NUMINAMATH_CALUDE_house_width_calculation_l3351_335138


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l3351_335135

/-- Given a sphere with surface area 256π cm², its volume is (2048/3)π cm³. -/
theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
  (4 * Real.pi * r^2 = 256 * Real.pi) → 
  ((4/3) * Real.pi * r^3 = (2048/3) * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l3351_335135


namespace NUMINAMATH_CALUDE_goose_eggs_laid_l3351_335163

theorem goose_eggs_laid (
  hatch_rate : ℚ)
  (first_month_survival : ℚ)
  (first_six_months_death : ℚ)
  (first_year_death : ℚ)
  (survived_first_year : ℕ)
  (h1 : hatch_rate = 3 / 7)
  (h2 : first_month_survival = 5 / 9)
  (h3 : first_six_months_death = 11 / 16)
  (h4 : first_year_death = 7 / 12)
  (h5 : survived_first_year = 84) :
  ∃ (eggs : ℕ), eggs ≥ 678 ∧
    (eggs : ℚ) * hatch_rate * first_month_survival * (1 - first_six_months_death) * (1 - first_year_death) = survived_first_year :=
by sorry

end NUMINAMATH_CALUDE_goose_eggs_laid_l3351_335163


namespace NUMINAMATH_CALUDE_savings_calculation_l3351_335123

def income_expenditure_ratio (income expenditure : ℚ) : Prop :=
  income / expenditure = 5 / 4

theorem savings_calculation (income : ℚ) (h : income_expenditure_ratio income ((4/5) * income)) :
  income - ((4/5) * income) = 3200 :=
by
  sorry

#check savings_calculation (16000 : ℚ)

end NUMINAMATH_CALUDE_savings_calculation_l3351_335123


namespace NUMINAMATH_CALUDE_equation_solution_l3351_335105

theorem equation_solution (x k : ℝ) : 
  -x^2 - (k + 9)*x - 8 = -(x - 2)*(x - 4) → k = -15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3351_335105


namespace NUMINAMATH_CALUDE_sweetsies_leftover_l3351_335100

theorem sweetsies_leftover (m : ℕ) : 
  (∃ k : ℕ, m = 11 * k + 8) →
  (∃ l : ℕ, 4 * m = 11 * l + 10) :=
by sorry

end NUMINAMATH_CALUDE_sweetsies_leftover_l3351_335100


namespace NUMINAMATH_CALUDE_canal_length_scientific_notation_l3351_335129

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- The length of the Beijing-Hangzhou Grand Canal in meters -/
def canal_length : ℕ := 1790000

/-- Converts a natural number to its scientific notation representation -/
def to_scientific_notation (n : ℕ) : ScientificNotation :=
  sorry

theorem canal_length_scientific_notation :
  to_scientific_notation canal_length = ScientificNotation.mk 1.79 6 :=
sorry

end NUMINAMATH_CALUDE_canal_length_scientific_notation_l3351_335129


namespace NUMINAMATH_CALUDE_large_box_chocolate_count_l3351_335174

/-- The number of chocolate bars in a large box -/
def total_chocolate_bars (num_small_boxes : ℕ) (bars_per_small_box : ℕ) : ℕ :=
  num_small_boxes * bars_per_small_box

/-- Theorem: The large box contains 475 chocolate bars -/
theorem large_box_chocolate_count :
  total_chocolate_bars 19 25 = 475 := by
  sorry

end NUMINAMATH_CALUDE_large_box_chocolate_count_l3351_335174


namespace NUMINAMATH_CALUDE_a_value_in_set_equality_l3351_335150

theorem a_value_in_set_equality (a b : ℝ) : 
  let A : Set ℝ := {a, b, 2}
  let B : Set ℝ := {2, b^2, 2*a}
  A ∩ B = A ∪ B → a = 0 ∨ a = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_a_value_in_set_equality_l3351_335150


namespace NUMINAMATH_CALUDE_intersection_point_m_value_l3351_335194

theorem intersection_point_m_value (m : ℝ) :
  (∃ y : ℝ, -3 * (-6) + y = m ∧ 2 * (-6) + y = 28) →
  m = 58 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_m_value_l3351_335194


namespace NUMINAMATH_CALUDE_birthday_cake_division_l3351_335176

/-- Calculates the weight of cake each of Juelz's sisters received after the birthday party -/
theorem birthday_cake_division (total_pieces : ℕ) (square_pieces : ℕ) (triangle_pieces : ℕ)
  (square_weight : ℕ) (triangle_weight : ℕ) (square_eaten_percent : ℚ) 
  (triangle_eaten_percent : ℚ) (forest_family_percent : ℚ) (friends_percent : ℚ) 
  (num_sisters : ℕ) :
  total_pieces = square_pieces + triangle_pieces →
  square_pieces = 160 →
  triangle_pieces = 80 →
  square_weight = 25 →
  triangle_weight = 20 →
  square_eaten_percent = 60 / 100 →
  triangle_eaten_percent = 40 / 100 →
  forest_family_percent = 30 / 100 →
  friends_percent = 25 / 100 →
  num_sisters = 3 →
  ∃ (sisters_share : ℕ), sisters_share = 448 ∧
    sisters_share = 
      ((1 - friends_percent) * 
       ((1 - forest_family_percent) * 
        ((square_pieces * (1 - square_eaten_percent) * square_weight) + 
         (triangle_pieces * (1 - triangle_eaten_percent) * triangle_weight)))) / num_sisters :=
by sorry

end NUMINAMATH_CALUDE_birthday_cake_division_l3351_335176


namespace NUMINAMATH_CALUDE_inequality_proof_l3351_335149

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a * b + b * c + c * a = 1) : 
  Real.sqrt (a^3 + a) + Real.sqrt (b^3 + b) + Real.sqrt (c^3 + c) ≥ 2 * Real.sqrt (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3351_335149


namespace NUMINAMATH_CALUDE_max_distance_M_to_N_l3351_335169

-- Define the circles and point
def circle_M (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2*a*x - 2*a*y + 2*a^2 - 2 = 0
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 18
def point_N : ℝ × ℝ := (1, 2)

-- Define the theorem
theorem max_distance_M_to_N :
  ∀ a : ℝ,
  (∀ x y : ℝ, ∃ x' y' : ℝ, circle_M a x' y' ∧ circle_O x' y') →
  ∃ a_max : ℝ,
    (∀ a' : ℝ, (∃ x y : ℝ, circle_M a' x y ∧ circle_O x y) →
      Real.sqrt ((a' - point_N.1)^2 + (a' - point_N.2)^2) ≤ Real.sqrt ((a_max - point_N.1)^2 + (a_max - point_N.2)^2)) ∧
    Real.sqrt ((a_max - point_N.1)^2 + (a_max - point_N.2)^2) = Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_M_to_N_l3351_335169


namespace NUMINAMATH_CALUDE_simplified_expression_equals_sqrt_two_minus_one_l3351_335132

theorem simplified_expression_equals_sqrt_two_minus_one :
  let x : ℝ := Real.sqrt 2 - 1
  (x^2 / (x^2 + 4*x + 4)) / (x / (x + 2)) - (x - 1) / (x + 2) = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_equals_sqrt_two_minus_one_l3351_335132


namespace NUMINAMATH_CALUDE_pizza_order_l3351_335172

theorem pizza_order (total_slices : ℕ) (slices_per_pizza : ℕ) (h1 : total_slices = 14) (h2 : slices_per_pizza = 2) :
  total_slices / slices_per_pizza = 7 := by
  sorry

end NUMINAMATH_CALUDE_pizza_order_l3351_335172


namespace NUMINAMATH_CALUDE_division_subtraction_equality_l3351_335162

theorem division_subtraction_equality : 144 / (12 / 3) - 5 = 31 := by
  sorry

end NUMINAMATH_CALUDE_division_subtraction_equality_l3351_335162


namespace NUMINAMATH_CALUDE_pizza_varieties_theorem_four_topping_combinations_l3351_335192

/-- Represents the number of base pizza flavors -/
def num_flavors : Nat := 4

/-- Represents the number of topping combinations -/
def num_topping_combinations : Nat := 4

/-- Represents the total number of pizza varieties -/
def total_varieties : Nat := 16

/-- Theorem stating that the number of pizza varieties is the product of 
    the number of flavors and the number of topping combinations -/
theorem pizza_varieties_theorem :
  num_flavors * num_topping_combinations = total_varieties := by
  sorry

/-- Definition of the possible topping combinations -/
inductive ToppingCombination
  | None
  | ExtraCheese
  | Mushrooms
  | ExtraCheeseAndMushrooms

/-- Theorem stating that there are exactly 4 topping combinations -/
theorem four_topping_combinations :
  (ToppingCombination.None :: ToppingCombination.ExtraCheese :: 
   ToppingCombination.Mushrooms :: ToppingCombination.ExtraCheeseAndMushrooms :: []).length = 
  num_topping_combinations := by
  sorry

end NUMINAMATH_CALUDE_pizza_varieties_theorem_four_topping_combinations_l3351_335192


namespace NUMINAMATH_CALUDE_pizzas_with_mushrooms_or_olives_l3351_335117

def num_toppings : ℕ := 8

-- Function to calculate combinations
def combinations (n k : ℕ) : ℕ :=
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Total number of pizzas with 1, 2, or 3 toppings
def total_pizzas : ℕ :=
  combinations num_toppings 1 + combinations num_toppings 2 + combinations num_toppings 3

-- Number of pizzas with mushrooms (or olives)
def pizzas_with_one_topping : ℕ :=
  1 + combinations (num_toppings - 1) 1 + combinations (num_toppings - 1) 2

-- Number of pizzas with both mushrooms and olives
def pizzas_with_both : ℕ :=
  1 + combinations (num_toppings - 2) 1 + combinations (num_toppings - 2) 2

-- Main theorem
theorem pizzas_with_mushrooms_or_olives :
  pizzas_with_one_topping * 2 - pizzas_with_both = 86 :=
sorry

end NUMINAMATH_CALUDE_pizzas_with_mushrooms_or_olives_l3351_335117


namespace NUMINAMATH_CALUDE_base_5_103_eq_28_l3351_335128

/-- Converts a list of digits in base b to its decimal representation -/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- The decimal representation of 103 in base 5 -/
def base_5_103 : Nat := to_decimal [3, 0, 1] 5

theorem base_5_103_eq_28 : base_5_103 = 28 := by sorry

end NUMINAMATH_CALUDE_base_5_103_eq_28_l3351_335128


namespace NUMINAMATH_CALUDE_unique_m_equals_three_l3351_335143

/-- A graph is k-flowing-chromatic if it satisfies certain coloring and movement conditions -/
def is_k_flowing_chromatic (G : Graph) (k : ℕ) : Prop := sorry

/-- T(G) is the least k such that G is k-flowing-chromatic, or 0 if no such k exists -/
def T (G : Graph) : ℕ := sorry

/-- χ(G) is the chromatic number of graph G -/
def chromatic_number (G : Graph) : ℕ := sorry

/-- A graph has no small cycles if all its cycles have length at least 2017 -/
def no_small_cycles (G : Graph) : Prop := sorry

/-- Main theorem: m = 3 is the only positive integer satisfying the conditions -/
theorem unique_m_equals_three :
  ∀ m : ℕ, m > 0 →
  (∃ G : Graph, chromatic_number G ≤ m ∧ T G ≥ 2^m ∧ no_small_cycles G) ↔ m = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_m_equals_three_l3351_335143


namespace NUMINAMATH_CALUDE_danny_larry_score_difference_l3351_335125

theorem danny_larry_score_difference :
  ∀ (keith larry danny : ℕ),
    keith = 3 →
    larry = 3 * keith →
    danny > larry →
    keith + larry + danny = 26 →
    danny - larry = 5 := by
  sorry

end NUMINAMATH_CALUDE_danny_larry_score_difference_l3351_335125


namespace NUMINAMATH_CALUDE_min_concerts_is_14_l3351_335197

/-- Represents a schedule of concerts --/
structure Schedule where
  numSingers : Nat
  singersPerConcert : Nat
  numConcerts : Nat
  pairsPerformTogether : Nat

/-- Checks if a schedule is valid --/
def isValidSchedule (s : Schedule) : Prop :=
  s.numSingers = 8 ∧
  s.singersPerConcert = 4 ∧
  s.numConcerts * (s.singersPerConcert.choose 2) = s.numSingers.choose 2 * s.pairsPerformTogether

/-- Theorem: The minimum number of concerts is 14 --/
theorem min_concerts_is_14 :
  ∀ s : Schedule, isValidSchedule s → s.numConcerts ≥ 14 :=
by sorry

end NUMINAMATH_CALUDE_min_concerts_is_14_l3351_335197


namespace NUMINAMATH_CALUDE_emmas_speed_last_segment_l3351_335189

def total_distance : ℝ := 150
def total_time : ℝ := 2
def speed_segment1 : ℝ := 50
def speed_segment2 : ℝ := 75
def num_segments : ℕ := 3

theorem emmas_speed_last_segment (speed_segment3 : ℝ) : 
  (speed_segment1 + speed_segment2 + speed_segment3) / num_segments = total_distance / total_time →
  speed_segment3 = 100 := by
sorry

end NUMINAMATH_CALUDE_emmas_speed_last_segment_l3351_335189


namespace NUMINAMATH_CALUDE_linda_jeans_sold_l3351_335147

/-- The number of jeans sold by Linda -/
def jeans_sold : ℕ := 4

/-- The price of a pair of jeans in dollars -/
def jeans_price : ℕ := 11

/-- The price of a tee in dollars -/
def tees_price : ℕ := 8

/-- The number of tees sold -/
def tees_sold : ℕ := 7

/-- The total revenue in dollars -/
def total_revenue : ℕ := 100

theorem linda_jeans_sold :
  jeans_sold * jeans_price + tees_sold * tees_price = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_linda_jeans_sold_l3351_335147


namespace NUMINAMATH_CALUDE_poverty_alleviation_rate_l3351_335112

theorem poverty_alleviation_rate (initial_population final_population : ℕ) 
  (years : ℕ) (decrease_rate : ℝ) : 
  initial_population = 90000 →
  final_population = 10000 →
  years = 2 →
  final_population = initial_population * (1 - decrease_rate) ^ years →
  9 * (1 - decrease_rate) ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_poverty_alleviation_rate_l3351_335112


namespace NUMINAMATH_CALUDE_number_calculation_l3351_335136

theorem number_calculation (N : ℝ) : (0.15 * 0.30 * 0.50 * N = 108) → N = 4800 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l3351_335136


namespace NUMINAMATH_CALUDE_cubic_fraction_equals_nine_l3351_335159

theorem cubic_fraction_equals_nine (x y : ℝ) (hx : x = 7) (hy : y = 2) :
  (x^3 + y^3) / (x^2 - x*y + y^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_cubic_fraction_equals_nine_l3351_335159


namespace NUMINAMATH_CALUDE_math_competition_probabilities_l3351_335155

def number_of_students : ℕ := 5
def number_of_boys : ℕ := 2
def number_of_girls : ℕ := 3
def students_to_select : ℕ := 2

-- Total number of possible selections
def total_selections : ℕ := Nat.choose number_of_students students_to_select

-- Number of ways to select exactly one boy
def exactly_one_boy_selections : ℕ := number_of_boys * number_of_girls

-- Number of ways to select at least one boy
def at_least_one_boy_selections : ℕ := total_selections - Nat.choose number_of_girls students_to_select

theorem math_competition_probabilities :
  (total_selections = 10) ∧
  (exactly_one_boy_selections / total_selections = 3 / 5) ∧
  (at_least_one_boy_selections / total_selections = 7 / 10) := by
  sorry

end NUMINAMATH_CALUDE_math_competition_probabilities_l3351_335155


namespace NUMINAMATH_CALUDE_max_value_of_ab_l3351_335199

theorem max_value_of_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  ab ≤ 1/8 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + b₀ = 1 ∧ a₀ * b₀ = 1/8 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_ab_l3351_335199


namespace NUMINAMATH_CALUDE_remainder_2017_div_89_l3351_335122

theorem remainder_2017_div_89 : 2017 % 89 = 59 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2017_div_89_l3351_335122


namespace NUMINAMATH_CALUDE_candies_distribution_l3351_335180

/-- The number of ways to partition n identical objects into at most k non-empty parts. -/
def partitions (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 30 ways to partition 10 identical objects into at most 5 non-empty parts. -/
theorem candies_distribution : partitions 10 5 = 30 := by sorry

end NUMINAMATH_CALUDE_candies_distribution_l3351_335180


namespace NUMINAMATH_CALUDE_floor_of_three_point_six_l3351_335186

theorem floor_of_three_point_six : ⌊(3.6 : ℝ)⌋ = 3 := by sorry

end NUMINAMATH_CALUDE_floor_of_three_point_six_l3351_335186


namespace NUMINAMATH_CALUDE_remainder_sum_l3351_335173

theorem remainder_sum (n : ℤ) (h : n % 12 = 5) : (n % 4 + n % 3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l3351_335173


namespace NUMINAMATH_CALUDE_sum_of_decimals_l3351_335124

theorem sum_of_decimals : 5.67 + (-3.92) = 1.75 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l3351_335124


namespace NUMINAMATH_CALUDE_product_of_roots_l3351_335185

theorem product_of_roots (x₁ x₂ : ℝ) 
  (h_distinct : x₁ ≠ x₂)
  (h₁ : x₁^2 - 2*x₁ = 1)
  (h₂ : x₂^2 - 2*x₂ = 1) : 
  x₁ * x₂ = -1 := by sorry

end NUMINAMATH_CALUDE_product_of_roots_l3351_335185


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l3351_335134

/-- Given an arithmetic sequence with the first four terms as specified,
    prove that the fifth term is x^2 - (21x)/5 -/
theorem arithmetic_sequence_fifth_term (x y : ℝ) :
  let a₁ := x^2 + 3*y
  let a₂ := (x - 2) * y
  let a₃ := x^2 - y
  let a₄ := x / (y + 1)
  -- The sequence is arithmetic
  (a₂ - a₁ = a₃ - a₂) ∧ (a₃ - a₂ = a₄ - a₃) →
  -- The fifth term
  ∃ (a₅ : ℝ), a₅ = x^2 - (21 * x) / 5 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l3351_335134


namespace NUMINAMATH_CALUDE_midpoint_coordinates_l3351_335126

/-- Given two points M and N in a 2D plane, this theorem proves that the midpoint P of the line segment MN has specific coordinates. -/
theorem midpoint_coordinates (M N : ℝ × ℝ) (hM : M = (3, -2)) (hN : N = (-1, 0)) :
  let P := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  P = (1, -1) := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinates_l3351_335126


namespace NUMINAMATH_CALUDE_zinc_weight_in_mixture_l3351_335177

/-- Given a mixture of zinc and copper in the ratio 9:11 with a total weight of 78 kg,
    the weight of zinc in the mixture is 35.1 kg. -/
theorem zinc_weight_in_mixture (zinc_ratio : ℚ) (copper_ratio : ℚ) (total_weight : ℚ) :
  zinc_ratio = 9 →
  copper_ratio = 11 →
  total_weight = 78 →
  (zinc_ratio / (zinc_ratio + copper_ratio)) * total_weight = 35.1 := by
  sorry

#check zinc_weight_in_mixture

end NUMINAMATH_CALUDE_zinc_weight_in_mixture_l3351_335177


namespace NUMINAMATH_CALUDE_smallest_b_for_g_nested_equals_g_l3351_335111

def g (x : ℤ) : ℤ :=
  if x % 15 = 0 then x / 15
  else if x % 3 = 0 then 5 * x
  else if x % 5 = 0 then 3 * x
  else x + 5

def g_nested (b : ℕ) (x : ℤ) : ℤ :=
  match b with
  | 0 => x
  | n + 1 => g (g_nested n x)

theorem smallest_b_for_g_nested_equals_g :
  ∀ b : ℕ, b > 1 → g_nested b 2 = g 2 → b ≥ 15 :=
sorry

end NUMINAMATH_CALUDE_smallest_b_for_g_nested_equals_g_l3351_335111


namespace NUMINAMATH_CALUDE_prime_power_plus_three_l3351_335137

theorem prime_power_plus_three (P : ℕ) : 
  Prime P → Prime (P^6 + 3) → P^10 + 3 = 1027 := by sorry

end NUMINAMATH_CALUDE_prime_power_plus_three_l3351_335137


namespace NUMINAMATH_CALUDE_pond_and_field_dimensions_l3351_335119

/-- Given a square field with a circular pond inside, this theorem proves
    the diameter of the pond and the side length of the field. -/
theorem pond_and_field_dimensions :
  ∀ (pond_diameter field_side : ℝ),
    pond_diameter > 0 →
    field_side > pond_diameter →
    (field_side^2 - (pond_diameter/2)^2 * 3) = 13.75 * 240 →
    field_side - pond_diameter = 40 →
    pond_diameter = 20 ∧ field_side = 60 := by
  sorry

end NUMINAMATH_CALUDE_pond_and_field_dimensions_l3351_335119


namespace NUMINAMATH_CALUDE_badminton_players_count_l3351_335109

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  total_members : ℕ
  tennis_players : ℕ
  neither_players : ℕ
  both_players : ℕ

/-- Calculates the number of badminton players in the sports club -/
def badminton_players (club : SportsClub) : ℕ :=
  club.total_members - club.neither_players - (club.tennis_players - club.both_players)

/-- Theorem stating that in a specific sports club configuration, 
    the number of badminton players is 20 -/
theorem badminton_players_count (club : SportsClub) 
  (h1 : club.total_members = 42)
  (h2 : club.tennis_players = 23)
  (h3 : club.neither_players = 6)
  (h4 : club.both_players = 7) :
  badminton_players club = 20 := by
  sorry

end NUMINAMATH_CALUDE_badminton_players_count_l3351_335109


namespace NUMINAMATH_CALUDE_num_four_digit_numbers_eq_twelve_l3351_335151

/-- The number of different four-digit numbers that can be formed using the cards "2", "0", "0", "9" (where "9" can also be used as "6") -/
def num_four_digit_numbers : ℕ :=
  (Nat.choose 3 2) * 2 * (Nat.factorial 2)

/-- Theorem stating that the number of different four-digit numbers is 12 -/
theorem num_four_digit_numbers_eq_twelve : num_four_digit_numbers = 12 := by
  sorry

#eval num_four_digit_numbers

end NUMINAMATH_CALUDE_num_four_digit_numbers_eq_twelve_l3351_335151


namespace NUMINAMATH_CALUDE_total_remaining_apples_l3351_335183

def tree_A : ℕ := 200
def tree_B : ℕ := 250
def tree_C : ℕ := 300

def picked_A : ℕ := tree_A / 5
def picked_B : ℕ := 2 * picked_A
def picked_C : ℕ := picked_A + 20

def remaining_A : ℕ := tree_A - picked_A
def remaining_B : ℕ := tree_B - picked_B
def remaining_C : ℕ := tree_C - picked_C

theorem total_remaining_apples :
  remaining_A + remaining_B + remaining_C = 570 := by
  sorry

end NUMINAMATH_CALUDE_total_remaining_apples_l3351_335183


namespace NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2016_l3351_335161

def last_four_digits (n : ℕ) : ℕ := n % 10000

def power_five_pattern : List ℕ := [3125, 5625, 8125, 0625]

theorem last_four_digits_of_5_pow_2016 :
  last_four_digits (5^2016) = 0625 :=
sorry

end NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2016_l3351_335161


namespace NUMINAMATH_CALUDE_geese_count_l3351_335179

/-- The number of geese in a flock that land on n lakes -/
def geese (n : ℕ) : ℕ := 2^n - 1

/-- 
Theorem: The number of geese in a flock is 2^n - 1, where n is the number of lakes,
given the landing pattern described.
-/
theorem geese_count (n : ℕ) : 
  (∀ k < n, (geese k + 1) / 2 + (geese k) / 2 = geese (k + 1)) → 
  geese 0 = 0 → 
  geese n = 2^n - 1 := by
sorry

end NUMINAMATH_CALUDE_geese_count_l3351_335179


namespace NUMINAMATH_CALUDE_tangent_slope_when_chord_maximized_l3351_335144

-- Define the circles
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 2
def circle_M (x y : ℝ) : Prop := (x - 1)^2 + (y - 3)^2 = 1

-- Define a point on circle M
def point_on_M (P : ℝ × ℝ) : Prop := circle_M P.1 P.2

-- Define a tangent line from a point on M to O
def is_tangent_line (P : ℝ × ℝ) (m : ℝ) : Prop :=
  point_on_M P ∧ ∃ A : ℝ × ℝ, circle_O A.1 A.2 ∧ (A.2 - P.2) = m * (A.1 - P.1)

-- Define the other intersection point Q
def other_intersection (P : ℝ × ℝ) (m : ℝ) (Q : ℝ × ℝ) : Prop :=
  point_on_M Q ∧ (Q.2 - P.2) = m * (Q.1 - P.1) ∧ P ≠ Q

-- Theorem statement
theorem tangent_slope_when_chord_maximized :
  ∃ P : ℝ × ℝ, ∃ m : ℝ, is_tangent_line P m ∧
  (∀ Q : ℝ × ℝ, other_intersection P m Q →
    ∀ P' : ℝ × ℝ, ∀ m' : ℝ, ∀ Q' : ℝ × ℝ,
      is_tangent_line P' m' ∧ other_intersection P' m' Q' →
      (P.1 - Q.1)^2 + (P.2 - Q.2)^2 ≥ (P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) →
  m = -7 ∨ m = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_slope_when_chord_maximized_l3351_335144


namespace NUMINAMATH_CALUDE_farmer_plot_allocation_l3351_335156

theorem farmer_plot_allocation (x y : ℕ) (h : x ≠ y) :
  ∃ (a b : ℕ), a^2 + b^2 = 2 * (x^2 + y^2) :=
by
  sorry

end NUMINAMATH_CALUDE_farmer_plot_allocation_l3351_335156


namespace NUMINAMATH_CALUDE_circle_area_when_six_reciprocal_circumference_equals_diameter_l3351_335170

/-- Given a circle where six times the reciprocal of its circumference equals its diameter, the area of the circle is 3/2 -/
theorem circle_area_when_six_reciprocal_circumference_equals_diameter (r : ℝ) (h : 6 * (1 / (2 * Real.pi * r)) = 2 * r) : π * r^2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_when_six_reciprocal_circumference_equals_diameter_l3351_335170


namespace NUMINAMATH_CALUDE_factorization_equality_l3351_335103

theorem factorization_equality (x y : ℝ) : 
  (x - y)^2 - (3*x^2 - 3*x*y + y^2) = x*(y - 2*x) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3351_335103


namespace NUMINAMATH_CALUDE_equation_solution_l3351_335178

theorem equation_solution (x : ℝ) (h : x ≠ 3) : (x + 6) / (x - 3) = 4 ↔ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3351_335178


namespace NUMINAMATH_CALUDE_seating_arrangement_one_between_AB_seating_arrangement_no_adjacent_empty_l3351_335188

/-- Number of students -/
def num_students : ℕ := 4

/-- Number of seats in the row -/
def num_seats : ℕ := 6

/-- Number of seating arrangements with exactly one person between A and B and no empty seats between them -/
def arrangements_with_one_between_AB : ℕ := 48

/-- Number of seating arrangements where all empty seats are not adjacent -/
def arrangements_no_adjacent_empty : ℕ := 240

/-- Theorem for the first question -/
theorem seating_arrangement_one_between_AB :
  (num_students = 4) → (num_seats = 6) →
  (arrangements_with_one_between_AB = 48) := by sorry

/-- Theorem for the second question -/
theorem seating_arrangement_no_adjacent_empty :
  (num_students = 4) → (num_seats = 6) →
  (arrangements_no_adjacent_empty = 240) := by sorry

end NUMINAMATH_CALUDE_seating_arrangement_one_between_AB_seating_arrangement_no_adjacent_empty_l3351_335188
