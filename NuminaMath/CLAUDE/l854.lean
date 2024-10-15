import Mathlib

namespace NUMINAMATH_CALUDE_ap_length_l854_85488

/-- Square with inscribed circle -/
structure SquareWithCircle where
  /-- Side length of the square -/
  side_length : ℝ
  /-- The square ABCD -/
  square : Set (ℝ × ℝ)
  /-- The inscribed circle ω -/
  circle : Set (ℝ × ℝ)
  /-- Point A of the square -/
  A : ℝ × ℝ
  /-- Point M where the circle intersects CD -/
  M : ℝ × ℝ
  /-- Point P where AM intersects the circle (different from M) -/
  P : ℝ × ℝ
  /-- The side length is 2 -/
  h_side_length : side_length = 2
  /-- A is a vertex of the square -/
  h_A_in_square : A ∈ square
  /-- M is on the circle and on the side CD -/
  h_M_on_circle_and_CD : M ∈ circle ∧ M.2 = -1
  /-- P is on the circle and on line AM -/
  h_P_on_circle_and_AM : P ∈ circle ∧ P ≠ M ∧ ∃ t : ℝ, P = (1 - t) • A + t • M

/-- The length of AP in a square with inscribed circle is √5/5 -/
theorem ap_length (swc : SquareWithCircle) : Real.sqrt 5 / 5 = ‖swc.A - swc.P‖ := by
  sorry

end NUMINAMATH_CALUDE_ap_length_l854_85488


namespace NUMINAMATH_CALUDE_triangle_construction_existence_and_uniqueness_l854_85481

-- Define the triangle structure
structure Triangle where
  sideA : ℝ
  sideB : ℝ
  angleC : ℝ
  sideA_pos : 0 < sideA
  sideB_pos : 0 < sideB
  angle_valid : 0 < angleC ∧ angleC < π

-- Theorem statement
theorem triangle_construction_existence_and_uniqueness 
  (a b : ℝ) (γ : ℝ) (ha : 0 < a) (hb : 0 < b) (hγ : 0 < γ ∧ γ < π) :
  ∃! t : Triangle, t.sideA = a ∧ t.sideB = b ∧ t.angleC = γ :=
sorry

end NUMINAMATH_CALUDE_triangle_construction_existence_and_uniqueness_l854_85481


namespace NUMINAMATH_CALUDE_max_product_decomposition_l854_85478

theorem max_product_decomposition (n k : ℕ) (h : k ≤ n) :
  ∃ (decomp : List ℕ),
    (decomp.sum = n) ∧
    (decomp.length = k) ∧
    (∀ (other_decomp : List ℕ),
      (other_decomp.sum = n) ∧ (other_decomp.length = k) →
      decomp.prod ≥ other_decomp.prod) ∧
    (decomp = List.replicate (n - n / k * k) (n / k + 1) ++ List.replicate (k - (n - n / k * k)) (n / k)) :=
  sorry

end NUMINAMATH_CALUDE_max_product_decomposition_l854_85478


namespace NUMINAMATH_CALUDE_range_of_k_l854_85400

-- Define the propositions p and q
def p (k : ℝ) : Prop := ∃ (x y : ℝ), x^2/k + y^2/(4-k) = 1 ∧ k > 0 ∧ 4 - k > 0

def q (k : ℝ) : Prop := ∃ (x y : ℝ), x^2/(k-1) + y^2/(k-3) = 1 ∧ (k-1)*(k-3) < 0

-- State the theorem
theorem range_of_k (k : ℝ) : (p k ∨ q k) → 1 < k ∧ k < 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_k_l854_85400


namespace NUMINAMATH_CALUDE_equation_solution_l854_85484

theorem equation_solution : ∃ (x : ℝ), x > 0 ∧ 5 * Real.sqrt (1 + x) + 5 * Real.sqrt (1 - x) = 7 * Real.sqrt 2 ∧ x = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l854_85484


namespace NUMINAMATH_CALUDE_min_distance_curve_to_line_l854_85448

/-- The minimum distance from any point on the curve xy = √3 to the line x + √3y = 0 is √3 -/
theorem min_distance_curve_to_line :
  let C := {P : ℝ × ℝ | P.1 * P.2 = Real.sqrt 3}
  let l := {P : ℝ × ℝ | P.1 + Real.sqrt 3 * P.2 = 0}
  ∃ d : ℝ, d = Real.sqrt 3 ∧
    ∀ P ∈ C, ∀ Q ∈ l, d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_curve_to_line_l854_85448


namespace NUMINAMATH_CALUDE_integral_x_squared_plus_one_over_x_l854_85471

open Real MeasureTheory Interval

theorem integral_x_squared_plus_one_over_x :
  ∫ x in (1 : ℝ)..2, (x^2 + 1) / x = 3/2 + Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_x_squared_plus_one_over_x_l854_85471


namespace NUMINAMATH_CALUDE_no_super_sextalternado_smallest_sextalternado_l854_85415

/-- Checks if a number has 8 digits --/
def has_eight_digits (n : ℕ) : Prop := 10000000 ≤ n ∧ n < 100000000

/-- Checks if consecutive digits have different parity --/
def has_alternating_parity (n : ℕ) : Prop :=
  ∀ i : ℕ, i < 7 → (n / 10^i % 2) ≠ (n / 10^(i+1) % 2)

/-- Defines a sextalternado number --/
def is_sextalternado (n : ℕ) : Prop :=
  has_eight_digits n ∧ n % 30 = 0 ∧ has_alternating_parity n

/-- Defines a super sextalternado number --/
def is_super_sextalternado (n : ℕ) : Prop :=
  is_sextalternado n ∧ n % 12 = 0

theorem no_super_sextalternado :
  ¬ ∃ n : ℕ, is_super_sextalternado n :=
sorry

theorem smallest_sextalternado :
  ∃ n : ℕ, is_sextalternado n ∧ ∀ m : ℕ, is_sextalternado m → n ≤ m ∧ n = 10101030 :=
sorry

end NUMINAMATH_CALUDE_no_super_sextalternado_smallest_sextalternado_l854_85415


namespace NUMINAMATH_CALUDE_negation_of_forall_geq_zero_l854_85417

theorem negation_of_forall_geq_zero (R : Type*) [OrderedRing R] :
  (¬ (∀ x : R, x^2 - 3 ≥ 0)) ↔ (∃ x₀ : R, x₀^2 - 3 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_geq_zero_l854_85417


namespace NUMINAMATH_CALUDE_transformed_area_theorem_l854_85404

-- Define the matrix
def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 1; 7, -3]

-- Define the original region's area
def S_area : ℝ := 10

-- Theorem statement
theorem transformed_area_theorem :
  let det := Matrix.det A
  let scale_factor := |det|
  scale_factor * S_area = 130 := by sorry

end NUMINAMATH_CALUDE_transformed_area_theorem_l854_85404


namespace NUMINAMATH_CALUDE_line_segment_ratio_l854_85455

/-- Given points E, F, G, and H on a line in that order, prove that EG:FH = 10:17 -/
theorem line_segment_ratio (E F G H : ℝ) : 
  (F - E = 3) → (G - F = 7) → (H - E = 20) → (G - E) / (H - F) = 10 / 17 := by
sorry

end NUMINAMATH_CALUDE_line_segment_ratio_l854_85455


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l854_85473

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℚ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 1 / a 0)) 
  (h_a3 : a 3 = 3/2) 
  (h_S3 : (a 1) + (a 2) + (a 3) = 9/2) :
  a 1 / a 0 = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l854_85473


namespace NUMINAMATH_CALUDE_breakfast_calories_l854_85485

def total_calories : ℕ := 2200
def lunch_calories : ℕ := 885
def snack_calories : ℕ := 130
def dinner_calories : ℕ := 832

theorem breakfast_calories : 
  total_calories - lunch_calories - snack_calories - dinner_calories = 353 := by
sorry

end NUMINAMATH_CALUDE_breakfast_calories_l854_85485


namespace NUMINAMATH_CALUDE_correct_marked_price_l854_85421

/-- Represents the pricing structure of a book -/
structure BookPricing where
  cost_price : ℝ
  marked_price : ℝ
  first_discount_rate : ℝ
  additional_discount_rate : ℝ
  profit_rate : ℝ
  commission_rate : ℝ

/-- Calculates the final selling price after all discounts and commissions -/
def final_selling_price (b : BookPricing) : ℝ :=
  let price_after_first_discount := b.marked_price * (1 - b.first_discount_rate)
  let price_after_additional_discount := price_after_first_discount * (1 - b.additional_discount_rate)
  let commission := price_after_first_discount * b.commission_rate
  price_after_additional_discount + commission

/-- Theorem stating the correct marked price for the given conditions -/
theorem correct_marked_price :
  ∃ (b : BookPricing),
    b.cost_price = 75 ∧
    b.first_discount_rate = 0.12 ∧
    b.additional_discount_rate = 0.05 ∧
    b.profit_rate = 0.3 ∧
    b.commission_rate = 0.1 ∧
    b.marked_price = 99.35 ∧
    final_selling_price b = b.cost_price * (1 + b.profit_rate) :=
by
  sorry


end NUMINAMATH_CALUDE_correct_marked_price_l854_85421


namespace NUMINAMATH_CALUDE_students_neither_sport_l854_85499

theorem students_neither_sport (total : ℕ) (football : ℕ) (cricket : ℕ) (both : ℕ) :
  total = 470 →
  football = 325 →
  cricket = 175 →
  both = 80 →
  total - (football + cricket - both) = 50 := by
  sorry

end NUMINAMATH_CALUDE_students_neither_sport_l854_85499


namespace NUMINAMATH_CALUDE_sum_cis_angle_sequence_l854_85433

-- Define the cis function
noncomputable def cis (θ : ℝ) : ℂ := Complex.exp (θ * Complex.I)

-- Define the arithmetic sequence of angles
def angleSequence : List ℝ := List.range 12 |>.map (λ n => 70 + 8 * n)

-- State the theorem
theorem sum_cis_angle_sequence (r : ℝ) (θ : ℝ) 
  (h_r : r > 0) (h_θ : 0 ≤ θ ∧ θ < 360) :
  (angleSequence.map (λ α => cis α)).sum = r * cis θ → θ = 114 := by
  sorry

end NUMINAMATH_CALUDE_sum_cis_angle_sequence_l854_85433


namespace NUMINAMATH_CALUDE_no_prime_sum_10001_l854_85454

/-- A function that returns the number of ways to write n as the sum of two primes -/
def countPrimePairs (n : ℕ) : ℕ :=
  (Finset.filter (fun p => Nat.Prime p ∧ Nat.Prime (n - p)) (Finset.range n)).card

/-- Theorem stating that 10001 cannot be written as the sum of two primes -/
theorem no_prime_sum_10001 : countPrimePairs 10001 = 0 := by sorry

end NUMINAMATH_CALUDE_no_prime_sum_10001_l854_85454


namespace NUMINAMATH_CALUDE_cosine_equality_l854_85469

theorem cosine_equality (n : ℤ) : 
  100 ≤ n ∧ n ≤ 280 ∧ Real.cos (n * π / 180) = Real.cos (317 * π / 180) → n = 317 :=
by sorry

end NUMINAMATH_CALUDE_cosine_equality_l854_85469


namespace NUMINAMATH_CALUDE_sum_first_150_remainder_l854_85408

theorem sum_first_150_remainder (n : Nat) (h : n = 150) : 
  (List.range n).sum % 5600 = 125 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_150_remainder_l854_85408


namespace NUMINAMATH_CALUDE_foci_coordinates_l854_85458

/-- Given that m is the geometric mean of 2 and 8, prove that the foci of x^2 + y^2/m = 1 are at (0, ±√3) -/
theorem foci_coordinates (m : ℝ) (hm_pos : m > 0) (hm_mean : m^2 = 2 * 8) :
  let equation := fun (x y : ℝ) ↦ x^2 + y^2 / m = 1
  ∃ c : ℝ, c^2 = 3 ∧ 
    (∀ x y : ℝ, equation x y ↔ equation x (-y)) ∧
    equation 0 c ∧ equation 0 (-c) :=
sorry

end NUMINAMATH_CALUDE_foci_coordinates_l854_85458


namespace NUMINAMATH_CALUDE_parking_lot_capacity_l854_85461

/-- Calculates the number of vehicles that can still park in a lot -/
def remainingParkingSpaces (totalSpaces : ℕ) (caravanSpaces : ℕ) (caravansParked : ℕ) : ℕ :=
  totalSpaces - (caravanSpaces * caravansParked)

/-- Theorem: Given the conditions, 24 vehicles can still park -/
theorem parking_lot_capacity : remainingParkingSpaces 30 2 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_capacity_l854_85461


namespace NUMINAMATH_CALUDE_a_2010_at_1_l854_85437

def a : ℕ → (ℝ → ℝ)
  | 0 => λ x => 1
  | 1 => λ x => x^2 + x + 1
  | (n+2) => λ x => (x^(n+2) + 1) * a (n+1) x - a n x

theorem a_2010_at_1 : a 2010 1 = 4021 := by
  sorry

end NUMINAMATH_CALUDE_a_2010_at_1_l854_85437


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l854_85445

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_eq : a + b + c = 13) 
  (sum_prod_eq : a * b + a * c + b * c = 30) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 1027 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l854_85445


namespace NUMINAMATH_CALUDE_chocolate_chips_per_family_member_l854_85441

/-- Represents the number of chocolate chips per cookie for each type --/
structure ChocolateChipsPerCookie :=
  (chocolate_chip : ℕ)
  (double_chocolate_chip : ℕ)
  (white_chocolate_chip : ℕ)

/-- Represents the number of cookies per batch for each type --/
structure CookiesPerBatch :=
  (chocolate_chip : ℕ)
  (double_chocolate_chip : ℕ)
  (white_chocolate_chip : ℕ)

/-- Represents the number of batches for each type of cookie --/
structure Batches :=
  (chocolate_chip : ℕ)
  (double_chocolate_chip : ℕ)
  (white_chocolate_chip : ℕ)

def total_chocolate_chips (chips_per_cookie : ChocolateChipsPerCookie) 
                          (cookies_per_batch : CookiesPerBatch) 
                          (batches : Batches) : ℕ :=
  chips_per_cookie.chocolate_chip * cookies_per_batch.chocolate_chip * batches.chocolate_chip +
  chips_per_cookie.double_chocolate_chip * cookies_per_batch.double_chocolate_chip * batches.double_chocolate_chip +
  chips_per_cookie.white_chocolate_chip * cookies_per_batch.white_chocolate_chip * batches.white_chocolate_chip

theorem chocolate_chips_per_family_member 
  (chips_per_cookie : ChocolateChipsPerCookie)
  (cookies_per_batch : CookiesPerBatch)
  (batches : Batches)
  (family_members : ℕ)
  (h1 : chips_per_cookie = ⟨2, 4, 3⟩)
  (h2 : cookies_per_batch = ⟨12, 10, 15⟩)
  (h3 : batches = ⟨3, 2, 1⟩)
  (h4 : family_members = 4)
  : (total_chocolate_chips chips_per_cookie cookies_per_batch batches) / family_members = 49 :=
by
  sorry

end NUMINAMATH_CALUDE_chocolate_chips_per_family_member_l854_85441


namespace NUMINAMATH_CALUDE_alcohol_solution_problem_l854_85477

theorem alcohol_solution_problem (initial_alcohol_percentage : ℝ) 
                                 (water_added : ℝ) 
                                 (final_alcohol_percentage : ℝ) : 
  initial_alcohol_percentage = 0.26 →
  water_added = 5 →
  final_alcohol_percentage = 0.195 →
  ∃ (initial_volume : ℝ),
    initial_volume * initial_alcohol_percentage = 
    (initial_volume + water_added) * final_alcohol_percentage ∧
    initial_volume = 15 := by
sorry

end NUMINAMATH_CALUDE_alcohol_solution_problem_l854_85477


namespace NUMINAMATH_CALUDE_number_divided_by_005_equals_900_l854_85450

theorem number_divided_by_005_equals_900 (x : ℝ) : x / 0.05 = 900 → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_005_equals_900_l854_85450


namespace NUMINAMATH_CALUDE_matrix_value_proof_l854_85452

def matrix_operation (a b c d : ℤ) : ℤ := a * c - b * d

theorem matrix_value_proof : matrix_operation 2 3 4 5 = -7 := by
  sorry

end NUMINAMATH_CALUDE_matrix_value_proof_l854_85452


namespace NUMINAMATH_CALUDE_product_of_three_consecutive_integers_divisible_by_six_l854_85423

theorem product_of_three_consecutive_integers_divisible_by_six (n : ℕ) (h : n > 0) :
  ∃ k : ℕ, n * (n + 1) * (n + 2) = 6 * k := by
sorry

end NUMINAMATH_CALUDE_product_of_three_consecutive_integers_divisible_by_six_l854_85423


namespace NUMINAMATH_CALUDE_field_width_l854_85474

/-- The width of a rectangular field given its area and length -/
theorem field_width (area : ℝ) (length : ℝ) (h1 : area = 143.2) (h2 : length = 4) :
  area / length = 35.8 := by
sorry

end NUMINAMATH_CALUDE_field_width_l854_85474


namespace NUMINAMATH_CALUDE_ladder_length_l854_85425

theorem ladder_length (angle : Real) (adjacent : Real) (hypotenuse : Real) :
  angle = Real.pi / 3 →  -- 60 degrees in radians
  adjacent = 9.493063650744542 →
  hypotenuse = adjacent / Real.cos angle →
  hypotenuse = 18.986127301489084 := by
  sorry

end NUMINAMATH_CALUDE_ladder_length_l854_85425


namespace NUMINAMATH_CALUDE_triangle_to_decagon_area_ratio_l854_85430

/-- A regular decagon -/
structure RegularDecagon where
  -- Add necessary fields here
  area : ℝ

/-- A triangle within a regular decagon formed by connecting three non-adjacent vertices -/
structure TriangleInDecagon (d : RegularDecagon) where
  -- Add necessary fields here
  area : ℝ

/-- The ratio of the area of a triangle to the area of the regular decagon it's inscribed in is 1/5 -/
theorem triangle_to_decagon_area_ratio 
  (d : RegularDecagon) 
  (t : TriangleInDecagon d) : 
  t.area / d.area = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_triangle_to_decagon_area_ratio_l854_85430


namespace NUMINAMATH_CALUDE_correct_share_distribution_l854_85427

def total_amount : ℕ := 12000
def ratio : List ℕ := [2, 4, 6, 3, 5]

def share_amount (total : ℕ) (ratios : List ℕ) : List ℕ :=
  let total_parts := ratios.sum
  let part_value := total / total_parts
  ratios.map (· * part_value)

theorem correct_share_distribution :
  share_amount total_amount ratio = [1200, 2400, 3600, 1800, 3000] := by
  sorry

end NUMINAMATH_CALUDE_correct_share_distribution_l854_85427


namespace NUMINAMATH_CALUDE_product_unit_digit_l854_85411

-- Define a function to get the unit digit of a number
def unitDigit (n : ℕ) : ℕ := n % 10

-- State the theorem
theorem product_unit_digit : 
  unitDigit (624 * 708 * 913 * 463) = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_unit_digit_l854_85411


namespace NUMINAMATH_CALUDE_additive_increasing_non_neg_implies_odd_and_increasing_l854_85434

/-- A function satisfying f(x₁ + x₂) = f(x₁) + f(x₂) for all x₁, x₂ ∈ ℝ -/
def IsAdditive (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, f (x₁ + x₂) = f x₁ + f x₂

/-- A function that is increasing on non-negative reals -/
def IsIncreasingNonNeg (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ≥ x₂ → x₂ ≥ 0 → f x₁ ≥ f x₂

/-- Main theorem: If f is additive and increasing on non-negative reals,
    then it is odd and increasing on all reals -/
theorem additive_increasing_non_neg_implies_odd_and_increasing
    (f : ℝ → ℝ) (h1 : IsAdditive f) (h2 : IsIncreasingNonNeg f) :
    (∀ x, f (-x) = -f x) ∧ (∀ x₁ x₂, x₁ ≥ x₂ → f x₁ ≥ f x₂) := by
  sorry

end NUMINAMATH_CALUDE_additive_increasing_non_neg_implies_odd_and_increasing_l854_85434


namespace NUMINAMATH_CALUDE_square_divisibility_l854_85449

theorem square_divisibility (a b : ℕ+) (h : (a * b + 1) ∣ (a ^ 2 + b ^ 2)) :
  ∃ k : ℕ, (a ^ 2 + b ^ 2) / (a * b + 1) = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_divisibility_l854_85449


namespace NUMINAMATH_CALUDE_bacteria_growth_3hours_l854_85444

/-- The number of bacteria after a given time, given an initial population and doubling time. -/
def bacteriaPopulation (initialPopulation : ℕ) (doublingTimeMinutes : ℕ) (totalTimeMinutes : ℕ) : ℕ :=
  initialPopulation * 2 ^ (totalTimeMinutes / doublingTimeMinutes)

/-- Theorem stating that after 3 hours, starting with 1 bacterium that doubles every 20 minutes, 
    the population will be 512. -/
theorem bacteria_growth_3hours :
  bacteriaPopulation 1 20 180 = 512 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_3hours_l854_85444


namespace NUMINAMATH_CALUDE_daniel_purchase_cost_l854_85470

/-- The total amount spent on a magazine and pencil after applying a coupon discount -/
def total_spent (magazine_cost pencil_cost coupon_discount : ℚ) : ℚ :=
  magazine_cost + pencil_cost - coupon_discount

/-- Theorem stating that given specific costs and discount, the total spent is $1.00 -/
theorem daniel_purchase_cost :
  total_spent 0.85 0.50 0.35 = 1.00 := by
  sorry

end NUMINAMATH_CALUDE_daniel_purchase_cost_l854_85470


namespace NUMINAMATH_CALUDE_largest_x_and_ratio_l854_85443

theorem largest_x_and_ratio (a b c d : ℤ) (x : ℝ) : 
  (7 * x / 8 + 1 = 4 / x) →
  (x = (a + b * Real.sqrt c) / d) →
  (x ≤ (-8 + 4 * Real.sqrt 15) / 7) →
  (x = (-8 + 4 * Real.sqrt 15) / 7 → a = -8 ∧ b = 4 ∧ c = 15 ∧ d = 7) →
  (a * c * d / b = -210) :=
by sorry

end NUMINAMATH_CALUDE_largest_x_and_ratio_l854_85443


namespace NUMINAMATH_CALUDE_palindrome_divisible_by_seven_probability_l854_85467

/-- A function that checks if a number is a palindrome -/
def is_palindrome (n : ℕ) : Prop := sorry

/-- A function that generates all 5-digit palindromes -/
def five_digit_palindromes : Finset ℕ := sorry

/-- A function that counts the number of elements in a finite set satisfying a predicate -/
def count_satisfying {α : Type*} (s : Finset α) (p : α → Prop) : ℕ := sorry

/-- The main theorem -/
theorem palindrome_divisible_by_seven_probability :
  ∃ k : ℕ, (k : ℚ) / 900 = (count_satisfying five_digit_palindromes 
    (λ n => (n % 7 = 0) ∧ (is_palindrome (n / 7)))) / (five_digit_palindromes.card) :=
sorry

end NUMINAMATH_CALUDE_palindrome_divisible_by_seven_probability_l854_85467


namespace NUMINAMATH_CALUDE_prob_specific_quarter_is_one_eighth_l854_85459

/-- Represents a piece of paper with two sides, each divided into four quarters -/
structure Paper :=
  (sides : Fin 2)
  (quarters : Fin 4)

/-- The total number of distinct parts (quarters) on the paper -/
def total_parts : ℕ := 8

/-- The probability of a specific quarter being on top after random folding -/
def prob_specific_quarter_on_top : ℚ := 1 / 8

/-- Theorem stating that the probability of a specific quarter being on top is 1/8 -/
theorem prob_specific_quarter_is_one_eighth :
  prob_specific_quarter_on_top = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_specific_quarter_is_one_eighth_l854_85459


namespace NUMINAMATH_CALUDE_liquid_mixture_problem_l854_85465

/-- Proves that the initial amount of liquid A is 21 litres given the conditions of the problem -/
theorem liquid_mixture_problem (initial_ratio_A : ℚ) (initial_ratio_B : ℚ) 
  (drawn_off : ℚ) (new_ratio_A : ℚ) (new_ratio_B : ℚ) :
  initial_ratio_A = 7 ∧ 
  initial_ratio_B = 5 ∧ 
  drawn_off = 9 ∧ 
  new_ratio_A = 7 ∧ 
  new_ratio_B = 9 → 
  ∃ (x : ℚ), 
    7 * x = 21 ∧ 
    (7 * x - (7 / 12) * drawn_off) / (5 * x - (5 / 12) * drawn_off + drawn_off) = new_ratio_A / new_ratio_B :=
by sorry

end NUMINAMATH_CALUDE_liquid_mixture_problem_l854_85465


namespace NUMINAMATH_CALUDE_box_width_is_15_l854_85460

/-- Given a rectangular box with length 8 cm and height 5 cm, built using 10 cubic cm cubes,
    and requiring a minimum of 60 cubes, prove that the width of the box is 15 cm. -/
theorem box_width_is_15 (length : ℝ) (height : ℝ) (cube_volume : ℝ) (min_cubes : ℕ) :
  length = 8 →
  height = 5 →
  cube_volume = 10 →
  min_cubes = 60 →
  (min_cubes : ℝ) * cube_volume / (length * height) = 15 := by
  sorry

end NUMINAMATH_CALUDE_box_width_is_15_l854_85460


namespace NUMINAMATH_CALUDE_derivative_sin_minus_exp_two_l854_85420

theorem derivative_sin_minus_exp_two (x : ℝ) :
  deriv (fun x => Real.sin x - (2 : ℝ)^x) x = Real.cos x - (2 : ℝ)^x * Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_minus_exp_two_l854_85420


namespace NUMINAMATH_CALUDE_volume_depends_on_length_l854_85482

/-- Represents a rectangular prism with variable length -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  volume : ℝ
  length_positive : length > 2
  width_is_two : width = 2
  height_is_one : height = 1
  volume_formula : volume = length * width * height

/-- The volume of a rectangular prism is dependent on its length -/
theorem volume_depends_on_length (prism : RectangularPrism) :
  ∃ f : ℝ → ℝ, prism.volume = f prism.length :=
by sorry

end NUMINAMATH_CALUDE_volume_depends_on_length_l854_85482


namespace NUMINAMATH_CALUDE_pig_to_cow_ratio_l854_85495

/-- Represents the farm scenario with cows and pigs -/
structure Farm where
  numCows : ℕ
  revenueCow : ℕ
  revenuePig : ℕ
  totalRevenue : ℕ

/-- Calculates the number of pigs based on the farm data -/
def calculatePigs (farm : Farm) : ℕ :=
  (farm.totalRevenue - farm.numCows * farm.revenueCow) / farm.revenuePig

/-- Theorem stating that the ratio of pigs to cows is 4:1 -/
theorem pig_to_cow_ratio (farm : Farm)
  (h1 : farm.numCows = 20)
  (h2 : farm.revenueCow = 800)
  (h3 : farm.revenuePig = 400)
  (h4 : farm.totalRevenue = 48000) :
  (calculatePigs farm) / farm.numCows = 4 := by
  sorry

#check pig_to_cow_ratio

end NUMINAMATH_CALUDE_pig_to_cow_ratio_l854_85495


namespace NUMINAMATH_CALUDE_cube_edge_sum_l854_85440

theorem cube_edge_sum (surface_area : ℝ) (edge_sum : ℝ) : 
  surface_area = 150 → edge_sum = 12 * (surface_area / 6).sqrt → edge_sum = 60 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_sum_l854_85440


namespace NUMINAMATH_CALUDE_bus_rental_equation_l854_85436

theorem bus_rental_equation (x : ℝ) (h : x > 2) :
  180 / x - 180 / (x + 2) = 3 :=
by sorry


end NUMINAMATH_CALUDE_bus_rental_equation_l854_85436


namespace NUMINAMATH_CALUDE_tissue_purchase_cost_l854_85442

/-- Calculate the total cost of tissues with discounts and tax -/
theorem tissue_purchase_cost
  (num_boxes : ℕ)
  (packs_per_box : ℕ)
  (tissues_per_pack : ℕ)
  (price_per_tissue : ℚ)
  (pack_discount : ℚ)
  (volume_discount : ℚ)
  (tax_rate : ℚ)
  (volume_discount_threshold : ℕ)
  (h_num_boxes : num_boxes = 25)
  (h_packs_per_box : packs_per_box = 18)
  (h_tissues_per_pack : tissues_per_pack = 150)
  (h_price_per_tissue : price_per_tissue = 6 / 100)
  (h_pack_discount : pack_discount = 10 / 100)
  (h_volume_discount : volume_discount = 8 / 100)
  (h_tax_rate : tax_rate = 5 / 100)
  (h_volume_discount_threshold : volume_discount_threshold = 10)
  : ∃ (total_cost : ℚ), total_cost = 3521.07 :=
by
  sorry

#check tissue_purchase_cost

end NUMINAMATH_CALUDE_tissue_purchase_cost_l854_85442


namespace NUMINAMATH_CALUDE_problem_statement_l854_85405

-- Define proposition p
def p : Prop := ∃ x₀ : ℝ, x₀^2 - x₀ + 1 < 0

-- Define proposition q
def q : Prop := ∀ a b : ℝ, a^2 < b^2 → a < b

-- Theorem to prove
theorem problem_statement : ¬p ∧ ¬q := by sorry

end NUMINAMATH_CALUDE_problem_statement_l854_85405


namespace NUMINAMATH_CALUDE_max_fourth_term_arithmetic_seq_l854_85410

/-- Given a sequence of six positive integers in arithmetic progression with a sum of 90,
    the maximum possible value of the fourth term is 17. -/
theorem max_fourth_term_arithmetic_seq : ∀ (a d : ℕ),
  a > 0 → d > 0 →
  a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) + (a + 5*d) = 90 →
  a + 3*d ≤ 17 :=
by sorry

end NUMINAMATH_CALUDE_max_fourth_term_arithmetic_seq_l854_85410


namespace NUMINAMATH_CALUDE_sales_tax_percentage_l854_85457

/-- Represents the problem of calculating sales tax percentage --/
theorem sales_tax_percentage
  (total_worth : ℝ)
  (tax_rate : ℝ)
  (tax_free_cost : ℝ)
  (h1 : total_worth = 40)
  (h2 : tax_rate = 0.06)
  (h3 : tax_free_cost = 34.7) :
  (total_worth - tax_free_cost) * tax_rate / total_worth = 0.0075 := by
  sorry

end NUMINAMATH_CALUDE_sales_tax_percentage_l854_85457


namespace NUMINAMATH_CALUDE_parabola_properties_l854_85438

/-- A parabola with equation y = ax^2 + (2m-6)x + 1 passing through (1, 2m-4) -/
def Parabola (a m : ℝ) : ℝ → ℝ := λ x => a * x^2 + (2*m - 6) * x + 1

/-- Points on the parabola -/
def PointsOnParabola (m : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  ((-m, Parabola 3 m (-m)), (m, Parabola 3 m m), (m+2, Parabola 3 m (m+2)))

theorem parabola_properties (m : ℝ) :
  let (y1, y2, y3) := (Parabola 3 m (-m), Parabola 3 m m, Parabola 3 m (m+2))
  Parabola 3 m 1 = 2*m - 4 ∧ 
  y2 < y3 ∧ y3 ≤ y1 →
  (3 : ℝ) = 3 ∧
  (3 - m : ℝ) = -((2*m - 6) / (2*3)) ∧
  1 < m ∧ m ≤ 2 := by sorry

#check parabola_properties

end NUMINAMATH_CALUDE_parabola_properties_l854_85438


namespace NUMINAMATH_CALUDE_circle_constant_l854_85453

theorem circle_constant (r : ℝ) (k : ℝ) (h1 : r = 36) (h2 : 2 * π * r = 72 * k) : k = π := by
  sorry

end NUMINAMATH_CALUDE_circle_constant_l854_85453


namespace NUMINAMATH_CALUDE_triangle_with_equal_angles_isosceles_l854_85429

/-- A triangle is isosceles if it has at least two equal angles. -/
def IsIsosceles (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ c = a

/-- Given a triangle ABC where ∠A = ∠B = 2∠C, prove that the triangle is isosceles. -/
theorem triangle_with_equal_angles_isosceles (a b c : ℝ) 
  (h1 : a + b + c = 180) -- Sum of angles in a triangle is 180°
  (h2 : a = b) -- ∠A = ∠B
  (h3 : a = 2 * c) -- ∠A = 2∠C
  : IsIsosceles a b c :=
sorry

end NUMINAMATH_CALUDE_triangle_with_equal_angles_isosceles_l854_85429


namespace NUMINAMATH_CALUDE_hcf_of_48_and_64_l854_85414

theorem hcf_of_48_and_64 : 
  let a := 48
  let b := 64
  let lcm := 192
  Nat.lcm a b = lcm → Nat.gcd a b = 16 := by
sorry

end NUMINAMATH_CALUDE_hcf_of_48_and_64_l854_85414


namespace NUMINAMATH_CALUDE_alice_savings_this_month_l854_85487

/-- Alice's sales and earnings calculation --/
def alice_savings (sales : ℝ) (basic_salary : ℝ) (commission_rate : ℝ) (savings_rate : ℝ) : ℝ :=
  let commission := sales * commission_rate
  let total_earnings := basic_salary + commission
  total_earnings * savings_rate

/-- Theorem: Alice's savings this month will be $29 --/
theorem alice_savings_this_month :
  alice_savings 2500 240 0.02 0.10 = 29 := by
  sorry

end NUMINAMATH_CALUDE_alice_savings_this_month_l854_85487


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l854_85496

theorem root_exists_in_interval :
  ∃ x : ℝ, 3/2 < x ∧ x < 2 ∧ 2^x = x^2 + 1/2 := by
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l854_85496


namespace NUMINAMATH_CALUDE_equal_roots_real_roots_l854_85413

/-- The quadratic equation given in the problem -/
def quadratic_equation (m x : ℝ) : Prop :=
  2 * (m + 1) * x^2 + 4 * m * x + 3 * m = 2

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ :=
  -8 * m^2 - 8 * m + 16

theorem equal_roots (m : ℝ) : 
  (∃ x : ℝ, quadratic_equation m x ∧ 
    ∀ y : ℝ, quadratic_equation m y → y = x) ↔ 
  (m = -2 ∨ m = 1) :=
sorry

theorem real_roots (m : ℝ) :
  (m = -1 → ∃ x : ℝ, quadratic_equation m x ∧ x = -5/4) ∧
  (m ≠ -1 → ∃ x : ℝ, quadratic_equation m x ∧ 
    ∃ s : ℝ, s^2 = -2*m^2 - 2*m + 4 ∧ 
      (x = (-2*m + s) / (2*(m+1)) ∨ x = (-2*m - s) / (2*(m+1)))) :=
sorry

end NUMINAMATH_CALUDE_equal_roots_real_roots_l854_85413


namespace NUMINAMATH_CALUDE_max_distance_MN_l854_85412

noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def g (x : ℝ) : ℝ := 2 * (Real.cos x)^2 - 1

theorem max_distance_MN :
  ∃ (max_dist : ℝ),
    (∀ (a : ℝ),
      let M := (a, f a)
      let N := (a, g a)
      let dist_MN := |f a - g a|
      dist_MN ≤ max_dist) ∧
    (∃ (a : ℝ),
      let M := (a, f a)
      let N := (a, g a)
      let dist_MN := |f a - g a|
      dist_MN = max_dist) ∧
    max_dist = 2 := by sorry

end NUMINAMATH_CALUDE_max_distance_MN_l854_85412


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l854_85476

def U : Set Int := {-1, 0, 1, 2, 3}
def A : Set Int := {2, 3}
def B : Set Int := {0, 1}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l854_85476


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l854_85463

theorem smallest_positive_integer_with_remainders : 
  ∃ (x : ℕ), x > 0 ∧ 
  x % 6 = 5 ∧ 
  x % 7 = 6 ∧ 
  x % 8 = 7 ∧ 
  ∀ (y : ℕ), y > 0 → 
    (y % 6 = 5 ∧ y % 7 = 6 ∧ y % 8 = 7) → 
    x ≤ y ∧
  x = 167 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l854_85463


namespace NUMINAMATH_CALUDE_expression_value_l854_85456

theorem expression_value (p q r : ℝ) 
  (hp : p ≠ 2) (hq : q ≠ 5) (hr : r ≠ 7) : 
  ((p - 2) / (7 - r)) * ((q - 5) / (2 - p)) * ((r - 7) / (5 - q)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l854_85456


namespace NUMINAMATH_CALUDE_committee_selection_l854_85466

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem committee_selection (total : ℕ) (committee_size : ℕ) (bill_karl : ℕ) (alice_jane : ℕ) :
  total = 9 ∧ committee_size = 5 ∧ bill_karl = 2 ∧ alice_jane = 2 →
  (choose (total - bill_karl) (committee_size - bill_karl) - 
   choose (total - bill_karl - alice_jane) 1) +
  (choose (total - bill_karl) committee_size - 
   choose (total - bill_karl - alice_jane) 3) = 41 :=
by sorry

end NUMINAMATH_CALUDE_committee_selection_l854_85466


namespace NUMINAMATH_CALUDE_divisor_problem_l854_85472

theorem divisor_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 15698 →
  quotient = 89 →
  remainder = 14 →
  dividend = divisor * quotient + remainder →
  divisor = 176 := by
sorry

end NUMINAMATH_CALUDE_divisor_problem_l854_85472


namespace NUMINAMATH_CALUDE_probability_of_exactly_three_primes_out_of_five_dice_l854_85464

def is_prime (n : ℕ) : Prop := sorry

def number_of_primes_up_to_20 : ℕ := 8

def probability_of_prime_on_20_sided_die : ℚ := 
  number_of_primes_up_to_20 / 20

def number_of_dice : ℕ := 5

def number_of_dice_showing_prime : ℕ := 3

theorem probability_of_exactly_three_primes_out_of_five_dice : 
  (Nat.choose number_of_dice number_of_dice_showing_prime : ℚ) * 
  (probability_of_prime_on_20_sided_die ^ number_of_dice_showing_prime) *
  ((1 - probability_of_prime_on_20_sided_die) ^ (number_of_dice - number_of_dice_showing_prime)) = 
  5 / 16 :=
sorry

end NUMINAMATH_CALUDE_probability_of_exactly_three_primes_out_of_five_dice_l854_85464


namespace NUMINAMATH_CALUDE_cross_section_area_l854_85406

/-- Regular triangular pyramid with given dimensions -/
structure RegularTriangularPyramid where
  base_side : ℝ
  height : ℝ

/-- Plane that intersects the pyramid -/
structure IntersectingPlane where
  perpendicular_to_base : Prop
  bisects_two_sides : Prop

/-- The cross-section created by the intersecting plane -/
def cross_section (p : RegularTriangularPyramid) (plane : IntersectingPlane) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- The area of a given set in 3D space -/
def area (s : Set (ℝ × ℝ × ℝ)) : ℝ :=
  sorry

/-- Theorem stating the area of the cross-section -/
theorem cross_section_area 
  (p : RegularTriangularPyramid) 
  (plane : IntersectingPlane) 
  (h1 : p.base_side = 2) 
  (h2 : p.height = 4) 
  (h3 : plane.perpendicular_to_base) 
  (h4 : plane.bisects_two_sides) : 
  area (cross_section p plane) = 1.5 :=
by sorry

end NUMINAMATH_CALUDE_cross_section_area_l854_85406


namespace NUMINAMATH_CALUDE_total_animals_l854_85492

theorem total_animals (giraffes pigs dogs : ℕ) 
  (h1 : giraffes = 6) 
  (h2 : pigs = 8) 
  (h3 : dogs = 4) : 
  giraffes + pigs + dogs = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_animals_l854_85492


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l854_85447

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (3*x - 1)^7 = a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₇ + a₆ + a₅ + a₄ + a₃ + a₂ + a₁ + a₀ = 128 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l854_85447


namespace NUMINAMATH_CALUDE_calculator_squaring_l854_85486

theorem calculator_squaring (n : ℕ) : (1 : ℝ) ^ (2^n) ≤ 100 := by
  sorry

end NUMINAMATH_CALUDE_calculator_squaring_l854_85486


namespace NUMINAMATH_CALUDE_intersection_condition_l854_85493

/-- Set M in R^2 -/
def M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 4}

/-- Set N in R^2 parameterized by r -/
def N (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 1)^2 ≤ r^2}

/-- The theorem stating the condition for M ∩ N = N -/
theorem intersection_condition (r : ℝ) : 
  (M ∩ N r = N r) ↔ (0 < r ∧ r ≤ 2 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l854_85493


namespace NUMINAMATH_CALUDE_max_area_rectangle_l854_85462

/-- Given a rectangle with perimeter 160 feet and integer side lengths, 
    the maximum possible area is 1600 square feet. -/
theorem max_area_rectangle (l w : ℕ) : 
  2 * (l + w) = 160 → l * w ≤ 1600 := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l854_85462


namespace NUMINAMATH_CALUDE_gcd_5280_12155_l854_85435

theorem gcd_5280_12155 : Nat.gcd 5280 12155 = 55 := by
  sorry

end NUMINAMATH_CALUDE_gcd_5280_12155_l854_85435


namespace NUMINAMATH_CALUDE_complex_equation_solution_l854_85409

theorem complex_equation_solution (x y : ℝ) :
  Complex.I * (x + y) = x - 1 → x = 1 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l854_85409


namespace NUMINAMATH_CALUDE_paving_rate_per_square_metre_l854_85468

/-- Proves that the rate per square metre for paving a room is Rs. 950 given the specified conditions. -/
theorem paving_rate_per_square_metre
  (length : ℝ)
  (width : ℝ)
  (total_cost : ℝ)
  (h1 : length = 5.5)
  (h2 : width = 4)
  (h3 : total_cost = 20900) :
  total_cost / (length * width) = 950 := by
  sorry

#check paving_rate_per_square_metre

end NUMINAMATH_CALUDE_paving_rate_per_square_metre_l854_85468


namespace NUMINAMATH_CALUDE_circle_diameter_theorem_l854_85490

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on a circle
def PointOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define a point inside or on a circle
def PointInOrOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 ≤ c.radius^2

-- Define a diameter of a circle
def Diameter (c : Circle) (d : ℝ × ℝ → ℝ × ℝ → Prop) : Prop :=
  ∀ p q, d p q → PointOnCircle c p ∧ PointOnCircle c q ∧
    (p.1 - q.1)^2 + (p.2 - q.2)^2 = 4 * c.radius^2

-- Define a point being on one side of a diameter
def OnOneSideOfDiameter (c : Circle) (d : ℝ × ℝ → ℝ × ℝ → Prop) (p : ℝ × ℝ) : Prop :=
  ∃ q r, Diameter c d ∧ d q r ∧
    ((p.1 - q.1) * (r.2 - q.2) - (p.2 - q.2) * (r.1 - q.1) ≥ 0 ∨
     (p.1 - q.1) * (r.2 - q.2) - (p.2 - q.2) * (r.1 - q.1) ≤ 0)

theorem circle_diameter_theorem (ω : Circle) (inner_circle : Circle) (points : Finset (ℝ × ℝ)) 
    (h1 : ∀ p ∈ points, PointOnCircle ω p)
    (h2 : inner_circle.radius < ω.radius)
    (h3 : ∀ p ∈ points, PointInOrOnCircle inner_circle p) :
  ∃ d : ℝ × ℝ → ℝ × ℝ → Prop, Diameter ω d ∧ 
    (∀ p ∈ points, OnOneSideOfDiameter ω d p) ∧
    (∀ p q, d p q → p ∉ points ∧ q ∉ points) :=
  sorry

end NUMINAMATH_CALUDE_circle_diameter_theorem_l854_85490


namespace NUMINAMATH_CALUDE_ratio_a_to_b_l854_85424

-- Define an arithmetic sequence
def is_arithmetic_sequence (s : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, s (n + 1) - s n = d

-- Define our specific sequence
def our_sequence (s : ℕ → ℝ) (y : ℝ) : Prop :=
  s 0 = s 1 - y ∧ s 1 = y ∧ s 2 = s 1 + y ∧ s 3 = 3 * y

theorem ratio_a_to_b (s : ℕ → ℝ) (y : ℝ) :
  is_arithmetic_sequence s → our_sequence s y → s 0 / s 2 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ratio_a_to_b_l854_85424


namespace NUMINAMATH_CALUDE_ball_box_arrangements_count_l854_85432

/-- The number of ways to put 4 distinguishable balls into 4 distinguishable boxes,
    where one particular ball cannot be placed in one specific box. -/
def ball_box_arrangements : ℕ :=
  let num_balls : ℕ := 4
  let num_boxes : ℕ := 4
  let restricted_ball_choices : ℕ := num_boxes - 1
  let unrestricted_ball_choices : ℕ := num_boxes
  restricted_ball_choices * (unrestricted_ball_choices ^ (num_balls - 1))

/-- Theorem stating that the number of arrangements is 192. -/
theorem ball_box_arrangements_count : ball_box_arrangements = 192 := by
  sorry

end NUMINAMATH_CALUDE_ball_box_arrangements_count_l854_85432


namespace NUMINAMATH_CALUDE_geometric_sequence_a1_l854_85451

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ (q : ℚ), ∀ (n : ℕ), a (n + 1) = a n * q

theorem geometric_sequence_a1 (a : ℕ → ℚ) :
  geometric_sequence a →
  a 2 * a 5 = 2 * a 3 →
  (a 4 + a 6) / 2 = 5/4 →
  a 1 = 16 ∨ a 1 = -16 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_a1_l854_85451


namespace NUMINAMATH_CALUDE_reverse_digits_difference_reverse_digits_difference_proof_l854_85498

def is_valid_k (k : ℕ) : Prop :=
  100 < k ∧ k < 1000

def reverse_digits (k : ℕ) : ℕ :=
  let h := k / 100
  let t := (k / 10) % 10
  let u := k % 10
  100 * u + 10 * t + h

theorem reverse_digits_difference (n : ℕ) : Prop :=
  ∃ (ks : Finset ℕ), 
    ks.card = 80 ∧ 
    (∀ k ∈ ks, is_valid_k k) ∧
    (∀ k ∈ ks, reverse_digits k = k + n) →
    n = 99

-- The proof goes here
theorem reverse_digits_difference_proof : reverse_digits_difference 99 := by
  sorry

end NUMINAMATH_CALUDE_reverse_digits_difference_reverse_digits_difference_proof_l854_85498


namespace NUMINAMATH_CALUDE_tennis_ball_ratio_l854_85479

theorem tennis_ball_ratio : 
  let total_ordered : ℕ := 64
  let extra_yellow : ℕ := 20
  let white_balls : ℕ := total_ordered / 2
  let yellow_balls : ℕ := total_ordered / 2 + extra_yellow
  let gcd : ℕ := Nat.gcd white_balls yellow_balls
  (white_balls / gcd : ℕ) = 8 ∧ (yellow_balls / gcd : ℕ) = 13 := by
sorry

end NUMINAMATH_CALUDE_tennis_ball_ratio_l854_85479


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l854_85428

theorem arithmetic_mean_of_fractions (x a : ℝ) (hx : x ≠ 0) :
  (1 / 2) * ((x^2 + a^2) / x^2 + (x^2 - a^2) / x^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l854_85428


namespace NUMINAMATH_CALUDE_quadratic_equality_l854_85407

theorem quadratic_equality (a b c x : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  (∃ p q r : Fin 6, p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    (let f : Fin 6 → ℝ := λ i =>
      match i with
      | 0 => a*x^2 + b*x + c
      | 1 => a*x^2 + c*x + b
      | 2 => b*x^2 + c*x + a
      | 3 => b*x^2 + a*x + c
      | 4 => c*x^2 + a*x + b
      | 5 => c*x^2 + b*x + a
    f p = f q ∧ f q = f r)) →
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equality_l854_85407


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l854_85489

-- Part 1
theorem problem_1 : (2 - Real.sqrt 3) ^ 0 - Real.sqrt 12 + Real.tan (π / 3) = 1 - Real.sqrt 3 := by sorry

-- Part 2
theorem problem_2 (a b : ℝ) (h : a ≠ b) : (a - b) / (a + b) / (b - a) = -1 / (a + b) := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l854_85489


namespace NUMINAMATH_CALUDE_apples_per_basket_l854_85426

theorem apples_per_basket (total_baskets : ℕ) (total_apples : ℕ) (h1 : total_baskets = 37) (h2 : total_apples = 629) :
  total_apples / total_baskets = 17 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_basket_l854_85426


namespace NUMINAMATH_CALUDE_house_legs_l854_85422

/-- The number of legs in a house with humans and various pets -/
def total_legs (humans dogs cats parrots goldfish : ℕ) : ℕ :=
  humans * 2 + dogs * 4 + cats * 4 + parrots * 2 + goldfish * 0

/-- Theorem: The total number of legs in the house is 38 -/
theorem house_legs : total_legs 5 2 3 4 5 = 38 := by
  sorry

end NUMINAMATH_CALUDE_house_legs_l854_85422


namespace NUMINAMATH_CALUDE_triangle_value_l854_85494

theorem triangle_value (q : ℝ) (h1 : 2 * triangle + q = 134) (h2 : 2 * (triangle + q) + q = 230) :
  triangle = 43 := by sorry

end NUMINAMATH_CALUDE_triangle_value_l854_85494


namespace NUMINAMATH_CALUDE_geometric_sequence_26th_term_l854_85480

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_26th_term
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequence a)
  (h_14th : a 14 = 10)
  (h_20th : a 20 = 80) :
  a 26 = 640 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_26th_term_l854_85480


namespace NUMINAMATH_CALUDE_not_prime_sum_of_squares_l854_85475

/-- The equation has exactly two positive integer roots -/
def has_two_positive_integer_roots (a b : ℝ) : Prop :=
  ∃ x y : ℤ, x > 0 ∧ y > 0 ∧ x ≠ y ∧
    (∀ z : ℤ, z > 0 → a * z * (z^2 + a * z + 1) = b * (z^2 + b + 1) ↔ z = x ∨ z = y)

/-- Main theorem -/
theorem not_prime_sum_of_squares (a b : ℝ) :
  ab < 0 →
  has_two_positive_integer_roots a b →
  ¬ Nat.Prime (Int.natAbs (Int.floor (a^2 + b^2))) :=
sorry

end NUMINAMATH_CALUDE_not_prime_sum_of_squares_l854_85475


namespace NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l854_85418

-- Define a quadratic equation
def quadratic_equation (k : ℤ) (x : ℤ) : Prop := x^2 - 65*x + k = 0

-- Define primality
def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, m > 1 → m < n → ¬(n % m = 0)

-- Theorem statement
theorem no_prime_roots_for_quadratic :
  ¬∃ k : ℤ, ∃ x y : ℤ, 
    x ≠ y ∧ 
    quadratic_equation k x ∧ 
    quadratic_equation k y ∧
    is_prime x ∧ 
    is_prime y :=
sorry

end NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l854_85418


namespace NUMINAMATH_CALUDE_intersection_points_vary_at_least_one_intersection_l854_85439

/-- The number of intersection points between y = Bx^2 and y^3 + 2 = x^2 + 4y varies with B -/
theorem intersection_points_vary (B : ℝ) (hB : B > 0) :
  ∃ (x y : ℝ), y = B * x^2 ∧ y^3 + 2 = x^2 + 4 * y ∧
  ∃ (B₁ B₂ : ℝ) (hB₁ : B₁ > 0) (hB₂ : B₂ > 0),
    (∀ (x₁ y₁ : ℝ), y₁ = B₁ * x₁^2 → y₁^3 + 2 = x₁^2 + 4 * y₁ →
      ∀ (x₂ y₂ : ℝ), y₂ = B₂ * x₂^2 → y₂^3 + 2 = x₂^2 + 4 * y₂ →
        (x₁, y₁) ≠ (x₂, y₂)) :=
by
  sorry

/-- There is at least one intersection point for any positive B -/
theorem at_least_one_intersection (B : ℝ) (hB : B > 0) :
  ∃ (x y : ℝ), y = B * x^2 ∧ y^3 + 2 = x^2 + 4 * y :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_points_vary_at_least_one_intersection_l854_85439


namespace NUMINAMATH_CALUDE_verbal_to_inequality_l854_85402

/-- The inequality that represents "twice x plus 8 is less than five times x" -/
def twice_x_plus_8_less_than_5x (x : ℝ) : Prop :=
  2 * x + 8 < 5 * x

theorem verbal_to_inequality :
  ∀ x : ℝ, twice_x_plus_8_less_than_5x x ↔ (2 * x + 8 < 5 * x) :=
by
  sorry

#check verbal_to_inequality

end NUMINAMATH_CALUDE_verbal_to_inequality_l854_85402


namespace NUMINAMATH_CALUDE_apple_percentage_is_fifty_percent_l854_85403

-- Define the initial number of apples and oranges
def initial_apples : ℕ := 10
def initial_oranges : ℕ := 5

-- Define the number of oranges added
def added_oranges : ℕ := 5

-- Define the total number of fruits after adding oranges
def total_fruits : ℕ := initial_apples + initial_oranges + added_oranges

-- Define the percentage of apples
def apple_percentage : ℚ := (initial_apples : ℚ) / (total_fruits : ℚ) * 100

-- Theorem statement
theorem apple_percentage_is_fifty_percent :
  apple_percentage = 50 := by sorry

end NUMINAMATH_CALUDE_apple_percentage_is_fifty_percent_l854_85403


namespace NUMINAMATH_CALUDE_bee_speed_solution_l854_85401

/-- The speed of a honey bee's flight between flowers -/
def bee_speed_problem (time_daisy_rose time_rose_poppy : ℝ) 
  (distance_difference speed_difference : ℝ) : Prop :=
  let speed_daisy_rose : ℝ := 6.5
  let speed_rose_poppy : ℝ := speed_daisy_rose + speed_difference
  let distance_daisy_rose : ℝ := speed_daisy_rose * time_daisy_rose
  let distance_rose_poppy : ℝ := speed_rose_poppy * time_rose_poppy
  distance_daisy_rose = distance_rose_poppy + distance_difference ∧
  speed_daisy_rose = 6.5

theorem bee_speed_solution :
  bee_speed_problem 10 6 8 3 := by
  sorry

#check bee_speed_solution

end NUMINAMATH_CALUDE_bee_speed_solution_l854_85401


namespace NUMINAMATH_CALUDE_max_fraction_value_l854_85419

theorem max_fraction_value (A B : ℝ) (h1 : A + B = 2020) (h2 : A / B < 1 / 4) :
  A / B ≤ 403 / 1617 :=
sorry

end NUMINAMATH_CALUDE_max_fraction_value_l854_85419


namespace NUMINAMATH_CALUDE_angle_rotation_l854_85446

theorem angle_rotation (initial_angle rotation : ℝ) (h1 : initial_angle = 25) (h2 : rotation = 350) :
  (initial_angle - (rotation - 360)) % 360 = 15 :=
sorry

end NUMINAMATH_CALUDE_angle_rotation_l854_85446


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_ABD_l854_85431

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube -/
structure Cube where
  sideLength : ℝ

/-- Represents a plane -/
structure Plane where
  normal : Point3D
  point : Point3D

/-- Represents a quadrilateral -/
structure Quadrilateral where
  a : Point3D
  b : Point3D
  c : Point3D
  d : Point3D

/-- Function to calculate the area of a quadrilateral -/
def areaQuadrilateral (q : Quadrilateral) : ℝ := sorry

/-- Main theorem statement -/
theorem area_of_quadrilateral_ABD (cube : Cube) (plane : Plane) (quadABD : Quadrilateral) :
  cube.sideLength = 2 →
  -- A is a vertex of the cube
  -- B and D are midpoints of edges adjacent to A
  -- C' is the midpoint of a face diagonal not including A
  -- Plane passes through A, B, D, and C'
  -- quadABD lies in the plane
  areaQuadrilateral quadABD = 2 := by sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_ABD_l854_85431


namespace NUMINAMATH_CALUDE_math_class_registration_l854_85491

theorem math_class_registration (total : ℕ) (history : ℕ) (english : ℕ) (all_three : ℕ) (exactly_two : ℕ) :
  total = 68 →
  history = 21 →
  english = 34 →
  all_three = 3 →
  exactly_two = 7 →
  ∃ (math : ℕ), math = 14 ∧ 
    total = history + math + english - (exactly_two - all_three) - all_three :=
by sorry

end NUMINAMATH_CALUDE_math_class_registration_l854_85491


namespace NUMINAMATH_CALUDE_quadratic_to_linear_solutions_l854_85497

theorem quadratic_to_linear_solutions (x : ℝ) :
  x^2 - 2*x - 1 = 0 ∧ (x - 1 = Real.sqrt 2 ∨ x - 1 = -Real.sqrt 2) →
  (x - 1 = Real.sqrt 2 → x - 1 = -Real.sqrt 2) ∧
  (x - 1 = -Real.sqrt 2 → x - 1 = Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_to_linear_solutions_l854_85497


namespace NUMINAMATH_CALUDE_johns_purchase_cost_l854_85416

/-- The total cost of John's purchase of gum and candy bars -/
def total_cost (gum_packs : ℕ) (candy_bars : ℕ) (candy_bar_price : ℚ) : ℚ :=
  let gum_pack_price := candy_bar_price / 2
  gum_packs * gum_pack_price + candy_bars * candy_bar_price

/-- Theorem stating that the total cost of John's purchase is $6 -/
theorem johns_purchase_cost :
  total_cost 2 3 (3/2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_johns_purchase_cost_l854_85416


namespace NUMINAMATH_CALUDE_right_triangle_PQR_area_l854_85483

/-- A right triangle PQR in the xy-plane with specific properties -/
structure RightTrianglePQR where
  /-- Point P of the triangle -/
  P : ℝ × ℝ
  /-- Point Q of the triangle -/
  Q : ℝ × ℝ
  /-- Point R of the triangle (right angle) -/
  R : ℝ × ℝ
  /-- The triangle has a right angle at R -/
  right_angle_at_R : (P.1 - R.1) * (Q.1 - R.1) + (P.2 - R.2) * (Q.2 - R.2) = 0
  /-- The length of hypotenuse PQ is 50 -/
  hypotenuse_length : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 50^2
  /-- The median through P lies on the line y = x + 5 -/
  median_P : ∃ t : ℝ, (P.1 + Q.1 + R.1) / 3 = t ∧ (P.2 + Q.2 + R.2) / 3 = t + 5
  /-- The median through Q lies on the line y = 3x + 6 -/
  median_Q : ∃ t : ℝ, (P.1 + Q.1 + R.1) / 3 = t ∧ (P.2 + Q.2 + R.2) / 3 = 3 * t + 6

/-- The area of the right triangle PQR is 104.1667 -/
theorem right_triangle_PQR_area (t : RightTrianglePQR) : 
  abs ((t.P.1 - t.R.1) * (t.Q.2 - t.R.2) - (t.Q.1 - t.R.1) * (t.P.2 - t.R.2)) / 2 = 104.1667 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_PQR_area_l854_85483
