import Mathlib

namespace NUMINAMATH_CALUDE_odd_increasing_function_inequality_l2347_234788

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

theorem odd_increasing_function_inequality 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_incr : is_increasing f) 
  (m : ℝ) 
  (h_ineq : ∀ θ : ℝ, f (Real.cos (2 * θ) - 5) + f (2 * m + 4 * Real.sin θ) > 0) :
  m > 5 := by
sorry

end NUMINAMATH_CALUDE_odd_increasing_function_inequality_l2347_234788


namespace NUMINAMATH_CALUDE_triangle_theorem_l2347_234747

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a * (Real.sqrt 3 * Real.tan t.B - 1) = 
        (t.b * Real.cos t.A / Real.cos t.B) + (t.c * Real.cos t.A / Real.cos t.C))
  (h2 : t.a + t.b + t.c = 20)
  (h3 : (1/2) * t.a * t.b * Real.sin t.C = 10 * Real.sqrt 3)
  (h4 : t.a > t.b) :
  t.C = Real.pi / 3 ∧ t.a = 8 ∧ t.b = 5 ∧ t.c = 7 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l2347_234747


namespace NUMINAMATH_CALUDE_parallelogram_line_theorem_l2347_234723

/-- A parallelogram with vertices at (15,35), (15,95), (27,122), and (27,62) -/
structure Parallelogram :=
  (v1 : ℝ × ℝ) (v2 : ℝ × ℝ) (v3 : ℝ × ℝ) (v4 : ℝ × ℝ)

/-- A line that passes through the origin -/
structure Line :=
  (slope : ℚ)

/-- The line cuts the parallelogram into two congruent polygons -/
def cuts_into_congruent_polygons (p : Parallelogram) (l : Line) : Prop := sorry

/-- m and n are relatively prime integers -/
def are_relatively_prime (m n : ℕ) : Prop := sorry

theorem parallelogram_line_theorem (p : Parallelogram) (l : Line) 
  (h1 : p.v1 = (15, 35)) (h2 : p.v2 = (15, 95)) (h3 : p.v3 = (27, 122)) (h4 : p.v4 = (27, 62))
  (h5 : cuts_into_congruent_polygons p l)
  (h6 : ∃ (m n : ℕ), l.slope = m / n ∧ are_relatively_prime m n) :
  ∃ (m n : ℕ), l.slope = m / n ∧ are_relatively_prime m n ∧ m + n = 71 := by sorry

end NUMINAMATH_CALUDE_parallelogram_line_theorem_l2347_234723


namespace NUMINAMATH_CALUDE_walmart_cards_requested_l2347_234735

def best_buy_card_value : ℕ := 500
def walmart_card_value : ℕ := 200
def best_buy_cards_requested : ℕ := 6
def best_buy_cards_sent : ℕ := 1
def walmart_cards_sent : ℕ := 2
def remaining_gift_card_value : ℕ := 3900

def total_best_buy_value : ℕ := best_buy_card_value * best_buy_cards_requested
def sent_gift_card_value : ℕ := best_buy_card_value * best_buy_cards_sent + walmart_card_value * walmart_cards_sent

theorem walmart_cards_requested (walmart_cards : ℕ) : 
  walmart_cards * walmart_card_value + total_best_buy_value = 
  remaining_gift_card_value + sent_gift_card_value → walmart_cards = 9 := by
  sorry

end NUMINAMATH_CALUDE_walmart_cards_requested_l2347_234735


namespace NUMINAMATH_CALUDE_prob_dime_is_25_143_l2347_234708

/-- Represents the types of coins in the jar -/
inductive Coin
  | Quarter
  | Dime
  | Penny

/-- The value of each coin type in cents -/
def coin_value : Coin → ℕ
  | Coin.Quarter => 25
  | Coin.Dime => 10
  | Coin.Penny => 1

/-- The total value of each coin type in the jar in cents -/
def total_value : Coin → ℕ
  | Coin.Quarter => 900
  | Coin.Dime => 500
  | Coin.Penny => 200

/-- The number of coins of each type in the jar -/
def coin_count (c : Coin) : ℕ := total_value c / coin_value c

/-- The total number of coins in the jar -/
def total_coins : ℕ := coin_count Coin.Quarter + coin_count Coin.Dime + coin_count Coin.Penny

/-- The probability of picking a dime from the jar -/
def prob_dime : ℚ := coin_count Coin.Dime / total_coins

theorem prob_dime_is_25_143 : prob_dime = 25 / 143 := by
  sorry


end NUMINAMATH_CALUDE_prob_dime_is_25_143_l2347_234708


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l2347_234763

theorem geometric_sequence_properties (a b c : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ -1 = -r^4 ∧ a = -r^3 ∧ b = r^2 ∧ c = -r ∧ -9 = 1) →
  b = -3 ∧ a * c = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l2347_234763


namespace NUMINAMATH_CALUDE_eighteenth_term_of_sequence_l2347_234762

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem eighteenth_term_of_sequence (a₁ a₂ : ℝ) (h : a₂ = a₁ + 6) :
  arithmetic_sequence a₁ (a₂ - a₁) 18 = 105 :=
by
  sorry

#check eighteenth_term_of_sequence 3 9 (by norm_num)

end NUMINAMATH_CALUDE_eighteenth_term_of_sequence_l2347_234762


namespace NUMINAMATH_CALUDE_at_least_n_prime_divisors_l2347_234764

theorem at_least_n_prime_divisors (n : ℕ) :
  ∃ (S : Finset Nat), (S.card ≥ n) ∧ (∀ p ∈ S, Nat.Prime p ∧ p ∣ (2^(2^n) + 2^(2^(n-1)) + 1)) :=
by sorry

end NUMINAMATH_CALUDE_at_least_n_prime_divisors_l2347_234764


namespace NUMINAMATH_CALUDE_square_root_equation_l2347_234783

theorem square_root_equation (n : ℝ) : Real.sqrt (10 + n) = 8 → n = 54 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_l2347_234783


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2347_234784

theorem fraction_evaluation : (3020 - 2890)^2 / 196 = 86 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2347_234784


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2347_234725

theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C ∧
  (c - b) / (Real.sqrt 2 * c - a) = Real.sin A / (Real.sin B + Real.sin C) →
  B = π / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2347_234725


namespace NUMINAMATH_CALUDE_impossible_transformation_l2347_234790

/-- Represents a natural number and its digits -/
structure DigitNumber where
  value : ℕ
  digits : List ℕ
  digits_valid : digits.all (· < 10)
  value_eq_digits : value = digits.foldl (fun acc d => acc * 10 + d) 0

/-- Defines the allowed operations on a DigitNumber -/
inductive Operation
  | multiply_by_two : Operation
  | rearrange_digits : Operation

/-- Applies an operation to a DigitNumber -/
def apply_operation (n : DigitNumber) (op : Operation) : DigitNumber :=
  match op with
  | Operation.multiply_by_two => sorry
  | Operation.rearrange_digits => sorry

/-- Checks if a DigitNumber is valid (non-zero first digit) -/
def is_valid (n : DigitNumber) : Prop :=
  n.digits.head? ≠ some 0

/-- Defines a sequence of operations -/
def OperationSequence := List Operation

/-- Applies a sequence of operations to a DigitNumber -/
def apply_sequence (n : DigitNumber) (seq : OperationSequence) : DigitNumber :=
  seq.foldl apply_operation n

theorem impossible_transformation :
  ¬∃ (seq : OperationSequence),
    let start : DigitNumber := ⟨1, [1], sorry, sorry⟩
    let result := apply_sequence start seq
    result.value = 811 ∧ is_valid result :=
  sorry

end NUMINAMATH_CALUDE_impossible_transformation_l2347_234790


namespace NUMINAMATH_CALUDE_opposite_solutions_imply_a_value_l2347_234779

theorem opposite_solutions_imply_a_value (a x y : ℚ) : 
  (x - y = 3 * a + 1) → 
  (x + y = 9 - 5 * a) → 
  (x = -y) → 
  (a = 9 / 5) := by
sorry

end NUMINAMATH_CALUDE_opposite_solutions_imply_a_value_l2347_234779


namespace NUMINAMATH_CALUDE_complex_absolute_value_l2347_234712

theorem complex_absolute_value (x : ℝ) (h : x > 0) :
  Complex.abs (-3 + 2*x*Complex.I) = 5 * Real.sqrt 5 ↔ x = Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l2347_234712


namespace NUMINAMATH_CALUDE_bisecting_line_slope_intercept_sum_l2347_234713

/-- Triangle XYZ with vertices X(1, 9), Y(3, 1), and Z(9, 1) -/
structure Triangle where
  X : ℝ × ℝ := (1, 9)
  Y : ℝ × ℝ := (3, 1)
  Z : ℝ × ℝ := (9, 1)

/-- A line represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The line that bisects the area of the triangle -/
def bisectingLine (t : Triangle) : Line :=
  sorry

theorem bisecting_line_slope_intercept_sum (t : Triangle) :
  (bisectingLine t).slope + (bisectingLine t).yIntercept = -3 := by
  sorry

end NUMINAMATH_CALUDE_bisecting_line_slope_intercept_sum_l2347_234713


namespace NUMINAMATH_CALUDE_mango_purchase_quantity_l2347_234758

/-- Calculates the quantity of mangoes purchased given the total payment, apple quantity, apple price, and mango price -/
def mango_quantity (total_payment : ℕ) (apple_quantity : ℕ) (apple_price : ℕ) (mango_price : ℕ) : ℕ :=
  ((total_payment - apple_quantity * apple_price) / mango_price)

/-- Theorem stating that the quantity of mangoes purchased is 9 kg -/
theorem mango_purchase_quantity :
  mango_quantity 1055 8 70 55 = 9 := by
  sorry

end NUMINAMATH_CALUDE_mango_purchase_quantity_l2347_234758


namespace NUMINAMATH_CALUDE_not_perfect_square_for_prime_l2347_234741

theorem not_perfect_square_for_prime (p : ℕ) (h_prime : Nat.Prime p) :
  ¬∃ (a : ℤ), a^2 = 7 * p + 3^p - 4 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_for_prime_l2347_234741


namespace NUMINAMATH_CALUDE_cookies_in_box_graemes_cookies_l2347_234756

/-- Given a box that can hold a certain weight of cookies and cookies of a specific weight,
    calculate the number of cookies that can fit in the box. -/
theorem cookies_in_box (box_capacity : ℕ) (cookie_weight : ℕ) (ounces_per_pound : ℕ) : ℕ :=
  let cookies_per_pound := ounces_per_pound / cookie_weight
  box_capacity * cookies_per_pound

/-- Prove that given a box that can hold 40 pounds of cookies, and each cookie weighing 2 ounces,
    the number of cookies that can fit in the box is equal to 320. -/
theorem graemes_cookies :
  cookies_in_box 40 2 16 = 320 := by
  sorry

end NUMINAMATH_CALUDE_cookies_in_box_graemes_cookies_l2347_234756


namespace NUMINAMATH_CALUDE_sandy_marbles_l2347_234786

/-- The number of marbles in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of red marbles Jessica has -/
def jessica_dozens : ℕ := 3

/-- The number of times more red marbles Sandy has compared to Jessica -/
def sandy_multiplier : ℕ := 4

/-- Theorem stating the number of red marbles Sandy has -/
theorem sandy_marbles : jessica_dozens * dozen * sandy_multiplier = 144 := by
  sorry

end NUMINAMATH_CALUDE_sandy_marbles_l2347_234786


namespace NUMINAMATH_CALUDE_number_added_to_55_l2347_234755

theorem number_added_to_55 : ∃ x : ℤ, 55 + x = 88 ∧ x = 33 := by sorry

end NUMINAMATH_CALUDE_number_added_to_55_l2347_234755


namespace NUMINAMATH_CALUDE_max_value_of_f_on_I_l2347_234771

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the interval
def I : Set ℝ := Set.Icc (-2) 2

-- State the theorem
theorem max_value_of_f_on_I :
  ∃ (m : ℝ), m = 2 ∧ ∀ x ∈ I, f x ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_I_l2347_234771


namespace NUMINAMATH_CALUDE_circle_equation_l2347_234717

/-- A circle C in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point lies on a circle -/
def Circle.contains (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Check if a line is tangent to a circle at a given point -/
def Circle.tangentAt (c : Circle) (m : ℝ) (b : ℝ) (p : ℝ × ℝ) : Prop :=
  c.contains p ∧ 
  (c.center.2 - p.2) / (c.center.1 - p.1) = -1 / m ∧
  p.2 = m * p.1 + b

/-- The main theorem -/
theorem circle_equation (C : Circle) :
  C.center = (3, 0) ∧ C.radius = 2 →
  C.contains (4, 1) ∧
  C.tangentAt 1 (-2) (2, 1) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l2347_234717


namespace NUMINAMATH_CALUDE_opposite_of_neg_nine_l2347_234797

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem stating that the opposite of -9 is 9
theorem opposite_of_neg_nine : opposite (-9) = 9 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_neg_nine_l2347_234797


namespace NUMINAMATH_CALUDE_simplify_polynomial_l2347_234753

theorem simplify_polynomial (x : ℝ) : (3 * x^2 + 9 * x - 5) - (2 * x^2 + 3 * x - 10) = x^2 + 6 * x + 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l2347_234753


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2347_234789

theorem polynomial_evaluation (x : ℝ) (h : x = 4) : x^4 + x^3 + x^2 + x + 1 = 341 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2347_234789


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l2347_234760

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 6*x + 4 = 0 ↔ (x - 3)^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l2347_234760


namespace NUMINAMATH_CALUDE_percentage_of_120_to_50_l2347_234778

theorem percentage_of_120_to_50 : 
  (120 : ℝ) / 50 * 100 = 240 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_120_to_50_l2347_234778


namespace NUMINAMATH_CALUDE_fund_raising_exceeded_goal_l2347_234722

def fund_raising (ken_amount : ℝ) : Prop :=
  let mary_amount := 5 * ken_amount
  let scott_amount := mary_amount / 3
  let total_collected := ken_amount + mary_amount + scott_amount
  let goal := 4000
  ken_amount = 600 → total_collected - goal = 600

theorem fund_raising_exceeded_goal : fund_raising 600 := by
  sorry

end NUMINAMATH_CALUDE_fund_raising_exceeded_goal_l2347_234722


namespace NUMINAMATH_CALUDE_disjunction_true_l2347_234751

theorem disjunction_true : 
  (∀ x > 0, ∃ y, y = x + 1/(2*x) ∧ y ≥ 1 ∧ ∀ z, z = x + 1/(2*x) → z ≥ y) ∨ 
  (∀ x > 1, x^2 + 2*x - 3 > 0) := by
sorry

end NUMINAMATH_CALUDE_disjunction_true_l2347_234751


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l2347_234757

theorem sum_of_absolute_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) : 
  (∀ x : ℝ, (2 - x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| = 665 := by
sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l2347_234757


namespace NUMINAMATH_CALUDE_sequence_type_l2347_234714

theorem sequence_type (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = 2 * n^2 - 2 * n) → 
  (∃ d : ℝ, d = 4 ∧ ∀ n, a (n + 1) - a n = d) := by
sorry

end NUMINAMATH_CALUDE_sequence_type_l2347_234714


namespace NUMINAMATH_CALUDE_percentage_time_in_meetings_l2347_234704

/-- Calculates the percentage of time spent in meetings during a work shift -/
theorem percentage_time_in_meetings
  (shift_hours : ℕ) -- Total hours in the shift
  (meeting1_minutes : ℕ) -- Duration of first meeting in minutes
  (meeting2_multiplier : ℕ) -- Multiplier for second meeting duration
  (meeting3_divisor : ℕ) -- Divisor for third meeting duration
  (h1 : shift_hours = 10)
  (h2 : meeting1_minutes = 30)
  (h3 : meeting2_multiplier = 2)
  (h4 : meeting3_divisor = 2)
  : (meeting1_minutes + meeting2_multiplier * meeting1_minutes + 
     (meeting2_multiplier * meeting1_minutes) / meeting3_divisor) * 100 / 
    (shift_hours * 60) = 20 := by
  sorry

#check percentage_time_in_meetings

end NUMINAMATH_CALUDE_percentage_time_in_meetings_l2347_234704


namespace NUMINAMATH_CALUDE_find_number_l2347_234754

theorem find_number : ∃ x : ℝ, x / 2 = 9 ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2347_234754


namespace NUMINAMATH_CALUDE_exists_polyhedron_no_three_same_sided_faces_l2347_234772

/-- A face of a polyhedron --/
structure Face where
  sides : ℕ

/-- A polyhedron --/
structure Polyhedron where
  faces : List Face

/-- Predicate to check if a polyhedron has no three faces with the same number of sides --/
def has_no_three_same_sided_faces (p : Polyhedron) : Prop :=
  ∀ n : ℕ, (p.faces.filter (λ f => f.sides = n)).length < 3

/-- Theorem stating the existence of a polyhedron with no three faces having the same number of sides --/
theorem exists_polyhedron_no_three_same_sided_faces :
  ∃ p : Polyhedron, has_no_three_same_sided_faces p ∧ p.faces.length = 6 :=
sorry

end NUMINAMATH_CALUDE_exists_polyhedron_no_three_same_sided_faces_l2347_234772


namespace NUMINAMATH_CALUDE_special_sequence_non_periodic_l2347_234782

/-- A sequence of 0s and 1s satisfying the given condition -/
def SpecialSequence (a : ℕ → Fin 2) : Prop :=
  ∀ n k, k < 2^n → a k ≠ a (k + 2^n)

/-- The theorem stating that such a sequence is non-periodic -/
theorem special_sequence_non_periodic (a : ℕ → Fin 2) (h : SpecialSequence a) :
  ¬ ∃ p, p > 0 ∧ ∀ n, a n = a (n + p) :=
sorry

end NUMINAMATH_CALUDE_special_sequence_non_periodic_l2347_234782


namespace NUMINAMATH_CALUDE_uncle_zhang_revenue_l2347_234726

/-- Uncle Zhang's newspaper selling problem -/
theorem uncle_zhang_revenue 
  (a b : ℕ) -- a and b are natural numbers representing the number of newspapers
  (purchase_price sell_price return_price : ℚ) -- prices are rational numbers
  (h1 : purchase_price = 0.4) -- purchase price is 0.4 yuan
  (h2 : sell_price = 0.5) -- selling price is 0.5 yuan
  (h3 : return_price = 0.2) -- return price is 0.2 yuan
  (h4 : b ≤ a) -- number of sold newspapers is not more than purchased
  : 
  sell_price * b + return_price * (a - b) - purchase_price * a = 0.3 * b - 0.2 * a :=
by sorry

end NUMINAMATH_CALUDE_uncle_zhang_revenue_l2347_234726


namespace NUMINAMATH_CALUDE_curve_length_is_pi_l2347_234780

/-- A closed convex curve in the plane -/
structure ClosedConvexCurve where
  -- Add necessary fields here
  -- This is just a placeholder definition

/-- The length of a curve -/
noncomputable def curve_length (c : ClosedConvexCurve) : ℝ :=
  sorry

/-- The length of the projection of a curve onto a line -/
noncomputable def projection_length (c : ClosedConvexCurve) (l : Line) : ℝ :=
  sorry

/-- A line in the plane -/
structure Line where
  -- Add necessary fields here
  -- This is just a placeholder definition

theorem curve_length_is_pi (c : ClosedConvexCurve) 
  (h : ∀ l : Line, projection_length c l = 1) : 
  curve_length c = π :=
sorry

end NUMINAMATH_CALUDE_curve_length_is_pi_l2347_234780


namespace NUMINAMATH_CALUDE_problem_solution_l2347_234749

theorem problem_solution (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a^2 / b = 1) (h2 : b^2 / c = 4) (h3 : c^2 / a = 4) :
  a = 64^(1/7) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2347_234749


namespace NUMINAMATH_CALUDE_bread_slices_count_bread_slices_count_proof_l2347_234781

/-- Proves that the original number of slices in a loaf of bread is 27 --/
theorem bread_slices_count : Nat → Prop :=
  fun original_slices =>
    let slices_eaten := 3 * 2
    let slices_for_toast := 2 * 10
    let remaining_slice := 1
    original_slices = slices_eaten + slices_for_toast + remaining_slice

#check bread_slices_count 27

/-- Proves that the theorem holds for 27 slices --/
theorem bread_slices_count_proof : bread_slices_count 27 := by
  sorry

end NUMINAMATH_CALUDE_bread_slices_count_bread_slices_count_proof_l2347_234781


namespace NUMINAMATH_CALUDE_lathe_probabilities_l2347_234798

-- Define the yield rates and processing percentages
def yield_rate_1 : ℝ := 0.15
def yield_rate_2 : ℝ := 0.10
def process_percent_1 : ℝ := 0.60
def process_percent_2 : ℝ := 0.40

-- Define the theorem
theorem lathe_probabilities :
  -- Probability of both lathes producing excellent parts simultaneously
  yield_rate_1 * yield_rate_2 = 0.015 ∧
  -- Probability of randomly selecting an excellent part from mixed parts
  process_percent_1 * yield_rate_1 + process_percent_2 * yield_rate_2 = 0.13 :=
by sorry

end NUMINAMATH_CALUDE_lathe_probabilities_l2347_234798


namespace NUMINAMATH_CALUDE_congruent_triangles_x_value_l2347_234745

/-- Given two congruent triangles ABC and DEF, where ABC has sides 3, 4, and 5,
    and DEF has sides 3, 3x-2, and 2x+1, prove that x = 2. -/
theorem congruent_triangles_x_value (x : ℝ) : 
  let a₁ : ℝ := 3
  let b₁ : ℝ := 4
  let c₁ : ℝ := 5
  let a₂ : ℝ := 3
  let b₂ : ℝ := 3 * x - 2
  let c₂ : ℝ := 2 * x + 1
  (a₁ + b₁ + c₁ = a₂ + b₂ + c₂) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_congruent_triangles_x_value_l2347_234745


namespace NUMINAMATH_CALUDE_sin_45_degrees_l2347_234729

theorem sin_45_degrees : Real.sin (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_45_degrees_l2347_234729


namespace NUMINAMATH_CALUDE_shopping_change_calculation_l2347_234720

def book_price : ℝ := 25
def pen_price : ℝ := 4
def ruler_price : ℝ := 1
def notebook_price : ℝ := 8
def pencil_case_price : ℝ := 6
def book_discount : ℝ := 0.1
def pen_discount : ℝ := 0.05
def sales_tax_rate : ℝ := 0.06
def payment : ℝ := 100

theorem shopping_change_calculation :
  let discounted_book_price := book_price * (1 - book_discount)
  let discounted_pen_price := pen_price * (1 - pen_discount)
  let total_before_tax := discounted_book_price + discounted_pen_price + ruler_price + notebook_price + pencil_case_price
  let tax_amount := total_before_tax * sales_tax_rate
  let total_with_tax := total_before_tax + tax_amount
  let change := payment - total_with_tax
  change = 56.22 := by sorry

end NUMINAMATH_CALUDE_shopping_change_calculation_l2347_234720


namespace NUMINAMATH_CALUDE_jennifer_initial_oranges_l2347_234766

/-- The number of fruits Jennifer has initially and after giving some away. -/
structure FruitCount where
  initial_pears : ℕ
  initial_apples : ℕ
  initial_oranges : ℕ
  pears_left : ℕ
  apples_left : ℕ
  oranges_left : ℕ
  total_left : ℕ

/-- Theorem stating the number of oranges Jennifer had initially. -/
theorem jennifer_initial_oranges (f : FruitCount) 
  (h1 : f.initial_pears = 10)
  (h2 : f.initial_apples = 2 * f.initial_pears)
  (h3 : f.pears_left = f.initial_pears - 2)
  (h4 : f.apples_left = f.initial_apples - 2)
  (h5 : f.oranges_left = f.initial_oranges - 2)
  (h6 : f.total_left = 44)
  (h7 : f.total_left = f.pears_left + f.apples_left + f.oranges_left) :
  f.initial_oranges = 20 := by
  sorry


end NUMINAMATH_CALUDE_jennifer_initial_oranges_l2347_234766


namespace NUMINAMATH_CALUDE_deepak_age_l2347_234710

/-- Proves that Deepak's present age is 21 years given the conditions -/
theorem deepak_age (rahul_future_age : ℕ) (years_difference : ℕ) (ratio_rahul : ℕ) (ratio_deepak : ℕ) :
  rahul_future_age = 34 →
  years_difference = 6 →
  ratio_rahul = 4 →
  ratio_deepak = 3 →
  (rahul_future_age - years_difference) * ratio_deepak = 21 * ratio_rahul :=
by
  sorry

#check deepak_age

end NUMINAMATH_CALUDE_deepak_age_l2347_234710


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2347_234770

theorem geometric_sequence_product (a : ℕ → ℝ) (q : ℝ) (m : ℕ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- Geometric sequence definition
  (q ≠ 1 ∧ q ≠ -1) →                -- Common ratio not equal to ±1
  (a 1 = 1) →                       -- First term is 1
  (a m = a 1 * a 2 * a 3 * a 4 * a 5) →  -- Condition given in the problem
  m = 11 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2347_234770


namespace NUMINAMATH_CALUDE_octal_to_decimal_fraction_l2347_234709

theorem octal_to_decimal_fraction (c d : ℕ) : 
  (5 * 8^2 + 4 * 8 + 7 = 300 + 10 * c + d) → 
  (0 ≤ c) → (c ≤ 9) → (0 ≤ d) → (d ≤ 9) →
  (c * d) / 12 = 5 / 4 := by sorry

end NUMINAMATH_CALUDE_octal_to_decimal_fraction_l2347_234709


namespace NUMINAMATH_CALUDE_binomial_expansion_problem_l2347_234769

theorem binomial_expansion_problem (m n : ℕ) (hm : m ≠ 0) (hn : n ≥ 2) :
  (∀ k, 0 ≤ k ∧ k ≤ n → (n.choose k) * m^k ≤ (n.choose 5) * m^5) ∧
  (n.choose 2) * m^2 = 9 * (n.choose 1) * m →
  m = 2 ∧ n = 10 ∧ (1 - 2 * 9)^10 % 6 = 1 :=
sorry

end NUMINAMATH_CALUDE_binomial_expansion_problem_l2347_234769


namespace NUMINAMATH_CALUDE_factorial_five_equals_120_l2347_234794

theorem factorial_five_equals_120 : 5 * 4 * 3 * 2 * 1 = 120 := by
  sorry

end NUMINAMATH_CALUDE_factorial_five_equals_120_l2347_234794


namespace NUMINAMATH_CALUDE_missing_number_proof_l2347_234705

theorem missing_number_proof (n : ℕ) (sum_with_missing : ℕ) : 
  (n = 63) → 
  (sum_with_missing = 2012) → 
  (n * (n + 1) / 2 - sum_with_missing = 4) :=
by sorry

end NUMINAMATH_CALUDE_missing_number_proof_l2347_234705


namespace NUMINAMATH_CALUDE_game_lives_theorem_l2347_234743

/-- Calculates the total number of lives after two levels in a game. -/
def totalLives (initial : ℕ) (firstLevelGain : ℕ) (secondLevelGain : ℕ) : ℕ :=
  initial + firstLevelGain + secondLevelGain

/-- Theorem stating that with 2 initial lives, gaining 6 in the first level
    and 11 in the second level, the total number of lives is 19. -/
theorem game_lives_theorem :
  totalLives 2 6 11 = 19 := by
  sorry

end NUMINAMATH_CALUDE_game_lives_theorem_l2347_234743


namespace NUMINAMATH_CALUDE_f_decreasing_interval_l2347_234728

-- Define the derivative of f
def f' (x : ℝ) : ℝ := x^2 + 3*x - 4

-- Define the derivative of f(x+1)
def f'_shifted (x : ℝ) : ℝ := (x + 1)^2 + 3*(x + 1) - 4

-- Theorem statement
theorem f_decreasing_interval :
  ∀ x ∈ Set.Ioo (-5 : ℝ) 0, f'_shifted x < 0 :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_l2347_234728


namespace NUMINAMATH_CALUDE_set_intersection_problem_l2347_234740

/-- Given sets M and N, prove their intersection -/
theorem set_intersection_problem (M N : Set ℝ) 
  (hM : M = {x : ℝ | -2 < x ∧ x < 2})
  (hN : N = {x : ℝ | |x - 1| ≤ 2}) :
  M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l2347_234740


namespace NUMINAMATH_CALUDE_total_trade_scientific_notation_l2347_234744

/-- Represents the total bilateral trade in goods in yuan -/
def total_trade : ℝ := 1653 * 1000000000

/-- Represents the scientific notation of the total trade -/
def scientific_notation : ℝ := 1.6553 * (10 ^ 12)

/-- Theorem stating that the total trade is equal to its scientific notation representation -/
theorem total_trade_scientific_notation : total_trade = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_total_trade_scientific_notation_l2347_234744


namespace NUMINAMATH_CALUDE_quadratic_residue_criterion_l2347_234795

theorem quadratic_residue_criterion (p a : ℕ) (hp : Prime p) (hp2 : p ≠ 2) (ha : a ≠ 0) :
  (∃ x, x^2 ≡ a [ZMOD p]) → a^((p-1)/2) ≡ 1 [ZMOD p] ∧
  (¬∃ x, x^2 ≡ a [ZMOD p]) → a^((p-1)/2) ≡ -1 [ZMOD p] := by
  sorry

end NUMINAMATH_CALUDE_quadratic_residue_criterion_l2347_234795


namespace NUMINAMATH_CALUDE_middle_digit_is_six_l2347_234711

/-- Represents a three-digit number in a given base -/
structure ThreeDigitNumber (base : ℕ) where
  hundreds : ℕ
  tens : ℕ
  ones : ℕ
  valid_digits : hundreds < base ∧ tens < base ∧ ones < base

/-- Converts a ThreeDigitNumber to its numerical value -/
def to_nat {base : ℕ} (n : ThreeDigitNumber base) : ℕ :=
  n.hundreds * base^2 + n.tens * base + n.ones

/-- Theorem: For a number M that is a three-digit number in base 8 and
    has its digits reversed in base 10, the middle digit of M in base 8 is 6 -/
theorem middle_digit_is_six :
  ∀ (M_base8 : ThreeDigitNumber 8) (M_base10 : ThreeDigitNumber 10),
    to_nat M_base8 = to_nat M_base10 →
    M_base8.hundreds = M_base10.ones →
    M_base8.tens = M_base10.tens →
    M_base8.ones = M_base10.hundreds →
    M_base8.tens = 6 := by
  sorry

end NUMINAMATH_CALUDE_middle_digit_is_six_l2347_234711


namespace NUMINAMATH_CALUDE_expedition_duration_l2347_234791

theorem expedition_duration (total_time : ℝ) (ratio : ℝ) (h1 : total_time = 10) (h2 : ratio = 3) :
  let first_expedition := total_time / (1 + ratio)
  first_expedition = 2.5 := by
sorry

end NUMINAMATH_CALUDE_expedition_duration_l2347_234791


namespace NUMINAMATH_CALUDE_dividing_line_b_range_l2347_234719

/-- Triangle ABC with vertices A(-1,0), B(1,0), and C(0,1) -/
structure Triangle where
  A : ℝ × ℝ := (-1, 0)
  B : ℝ × ℝ := (1, 0)
  C : ℝ × ℝ := (0, 1)

/-- Line y = ax + b that divides the triangle -/
structure DividingLine where
  a : ℝ
  b : ℝ
  a_pos : a > 0

/-- The line divides the triangle into two parts of equal area -/
def dividesEqualArea (t : Triangle) (l : DividingLine) : Prop := sorry

/-- The range of b values that satisfy the condition -/
def validRange : Set ℝ := Set.Ioo (1 - Real.sqrt 2 / 2) (1 / 2)

/-- Theorem stating the range of b values -/
theorem dividing_line_b_range (t : Triangle) (l : DividingLine) 
  (h : dividesEqualArea t l) : l.b ∈ validRange := by
  sorry

end NUMINAMATH_CALUDE_dividing_line_b_range_l2347_234719


namespace NUMINAMATH_CALUDE_max_followers_count_l2347_234787

/-- Represents the types of islanders --/
inductive IslanderType
  | Knight
  | Liar
  | Follower

/-- Represents an answer to the question --/
inductive Answer
  | Yes
  | No

/-- Defines the properties of the island and its inhabitants --/
structure Island where
  totalPopulation : Nat
  knightCount : Nat
  liarCount : Nat
  followerCount : Nat
  yesAnswers : Nat
  noAnswers : Nat

/-- Defines the conditions of the problem --/
def isValidIsland (i : Island) : Prop :=
  i.totalPopulation = 2018 ∧
  i.knightCount + i.liarCount + i.followerCount = i.totalPopulation ∧
  i.yesAnswers = 1009 ∧
  i.noAnswers = i.totalPopulation - i.yesAnswers

/-- The main theorem to prove --/
theorem max_followers_count (i : Island) (h : isValidIsland i) :
  i.followerCount ≤ 1009 ∧ ∃ (j : Island), isValidIsland j ∧ j.followerCount = 1009 :=
sorry

end NUMINAMATH_CALUDE_max_followers_count_l2347_234787


namespace NUMINAMATH_CALUDE_tan_alpha_3_implies_sin_2alpha_over_cos_alpha_squared_6_l2347_234765

theorem tan_alpha_3_implies_sin_2alpha_over_cos_alpha_squared_6 (α : Real) 
  (h : Real.tan α = 3) : 
  (Real.sin (2 * α)) / (Real.cos α)^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_3_implies_sin_2alpha_over_cos_alpha_squared_6_l2347_234765


namespace NUMINAMATH_CALUDE_apples_eaten_l2347_234792

theorem apples_eaten (total : ℕ) (eaten : ℕ) : 
  total = 6 → 
  eaten + 2 * eaten = total → 
  eaten = 2 := by
sorry

end NUMINAMATH_CALUDE_apples_eaten_l2347_234792


namespace NUMINAMATH_CALUDE_complex_modulus_reciprocal_l2347_234777

theorem complex_modulus_reciprocal (z : ℂ) (h : (1 + z) / (1 + Complex.I) = 2 - Complex.I) :
  Complex.abs (1 / z) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_reciprocal_l2347_234777


namespace NUMINAMATH_CALUDE_gcd_1729_587_l2347_234738

theorem gcd_1729_587 : Nat.gcd 1729 587 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1729_587_l2347_234738


namespace NUMINAMATH_CALUDE_tank_insulation_cost_l2347_234702

def tank_length : ℝ := 4
def tank_width : ℝ := 5
def tank_height : ℝ := 3
def insulation_cost_per_sqft : ℝ := 20

def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

def total_cost (sa : ℝ) (cost_per_sqft : ℝ) : ℝ := sa * cost_per_sqft

theorem tank_insulation_cost :
  total_cost (surface_area tank_length tank_width tank_height) insulation_cost_per_sqft = 1880 := by
  sorry

end NUMINAMATH_CALUDE_tank_insulation_cost_l2347_234702


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l2347_234700

theorem tan_alpha_plus_pi_fourth (α : ℝ) (M : ℝ × ℝ) :
  M.1 = 1 ∧ M.2 = Real.sqrt 3 →
  (∃ t : ℝ, t > 0 ∧ t * M.1 = 1 ∧ t * M.2 = Real.tan α) →
  Real.tan (α + π / 4) = -2 - Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l2347_234700


namespace NUMINAMATH_CALUDE_inequality_proof_l2347_234706

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : c^2 + a*b = a^2 + b^2) : 
  c^2 + a*b ≤ a*c + b*c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2347_234706


namespace NUMINAMATH_CALUDE_tina_pen_difference_l2347_234724

/-- Prove that Tina has 3 more blue pens than green pens -/
theorem tina_pen_difference : 
  ∀ (pink green blue : ℕ),
  pink = 12 →
  green = pink - 9 →
  blue > green →
  pink + green + blue = 21 →
  blue - green = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_tina_pen_difference_l2347_234724


namespace NUMINAMATH_CALUDE_cos_sin_sum_zero_l2347_234746

theorem cos_sin_sum_zero (θ a : Real) (h : Real.cos (π / 6 - θ) = a) :
  Real.cos (5 * π / 6 + θ) + Real.sin (2 * π / 3 - θ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_sum_zero_l2347_234746


namespace NUMINAMATH_CALUDE_hannah_savings_l2347_234701

theorem hannah_savings (first_week : ℝ) : first_week = 4 := by
  have total_goal : ℝ := 80
  have fifth_week : ℝ := 20
  have savings_sum : first_week + 2 * first_week + 4 * first_week + 8 * first_week + fifth_week = total_goal := by sorry
  sorry

end NUMINAMATH_CALUDE_hannah_savings_l2347_234701


namespace NUMINAMATH_CALUDE_square_1369_product_l2347_234748

theorem square_1369_product (x : ℤ) (h : x^2 = 1369) : (x + 3) * (x - 3) = 1360 := by
  sorry

end NUMINAMATH_CALUDE_square_1369_product_l2347_234748


namespace NUMINAMATH_CALUDE_unique_solution_ceiling_equation_l2347_234733

theorem unique_solution_ceiling_equation :
  ∃! b : ℝ, b + ⌈b⌉ = 17.8 ∧ b = 8.8 := by sorry

end NUMINAMATH_CALUDE_unique_solution_ceiling_equation_l2347_234733


namespace NUMINAMATH_CALUDE_parabola_intersection_l2347_234775

-- Define the two parabola functions
def f (x : ℝ) : ℝ := 3 * x^2 - 15 * x - 4
def g (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 8

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) := {(3, -22), (4, -16)}

-- Theorem statement
theorem parabola_intersection :
  ∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x, y) ∈ intersection_points :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_l2347_234775


namespace NUMINAMATH_CALUDE_comics_after_reassembly_l2347_234742

/-- The number of comics in the box after reassembly -/
def total_comics (pages_per_comic : ℕ) (extra_pages : ℕ) (total_pages : ℕ) (untorn_comics : ℕ) : ℕ :=
  untorn_comics + (total_pages - extra_pages) / pages_per_comic

/-- Theorem stating the total number of comics after reassembly -/
theorem comics_after_reassembly :
  total_comics 47 3 3256 20 = 89 := by
  sorry

end NUMINAMATH_CALUDE_comics_after_reassembly_l2347_234742


namespace NUMINAMATH_CALUDE_set_in_proportion_l2347_234774

/-- A set of four numbers (a, b, c, d) is in proportion if a:b = c:d -/
def IsInProportion (a b c d : ℚ) : Prop :=
  a * d = b * c

/-- Prove that the set (1, 2, 2, 4) is in proportion -/
theorem set_in_proportion :
  IsInProportion 1 2 2 4 := by
  sorry

end NUMINAMATH_CALUDE_set_in_proportion_l2347_234774


namespace NUMINAMATH_CALUDE_f_symmetric_f_upper_bound_f_solution_range_l2347_234739

noncomputable section

def f (x : ℝ) : ℝ := Real.log ((1 + x) / (x - 1))

theorem f_symmetric : ∀ x : ℝ, f (-x) = -f x := by sorry

theorem f_upper_bound : ∀ x : ℝ, x > 1 → f x + Real.log (0.5 * (x - 1)) < -1 := by sorry

theorem f_solution_range (k : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 2 3 ∧ f x = Real.log (0.5 * (x + k))) ↔ k ∈ Set.Icc (-1) 1 := by sorry

end

end NUMINAMATH_CALUDE_f_symmetric_f_upper_bound_f_solution_range_l2347_234739


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2347_234752

theorem purely_imaginary_complex_number (a : ℝ) : 
  let z : ℂ := Complex.mk (a^2 + 2*a - 3) (a + 3)
  (z.re = 0 ∧ z.im ≠ 0) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2347_234752


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l2347_234715

theorem roots_quadratic_equation (m n : ℝ) : 
  (m^2 - 2*m + 1 = 0) → (n^2 - 2*n + 1 = 0) → 
  (m + n) / (m^2 - 2*m) = -2 := by
  sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l2347_234715


namespace NUMINAMATH_CALUDE_solution_replacement_concentration_l2347_234716

theorem solution_replacement_concentration 
  (initial_concentration : ℝ) 
  (final_concentration : ℝ) 
  (replaced_fraction : ℝ) 
  (replacement_concentration : ℝ) : 
  initial_concentration = 45 ∧ 
  final_concentration = 35 ∧ 
  replaced_fraction = 0.5 → 
  replacement_concentration = 25 := by
  sorry

end NUMINAMATH_CALUDE_solution_replacement_concentration_l2347_234716


namespace NUMINAMATH_CALUDE_rectangle_problem_l2347_234703

theorem rectangle_problem (l b : ℝ) : 
  l = 2 * b →
  (l - 5) * (b + 5) - l * b = 75 →
  20 < l ∧ l < 50 →
  10 < b ∧ b < 30 →
  l = 40 := by
sorry

end NUMINAMATH_CALUDE_rectangle_problem_l2347_234703


namespace NUMINAMATH_CALUDE_constant_zero_arithmetic_not_geometric_l2347_234730

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

def constant_zero_sequence : ℕ → ℝ :=
  λ _ => 0

theorem constant_zero_arithmetic_not_geometric :
  is_arithmetic_sequence constant_zero_sequence ∧
  ¬ is_geometric_sequence constant_zero_sequence :=
by sorry

end NUMINAMATH_CALUDE_constant_zero_arithmetic_not_geometric_l2347_234730


namespace NUMINAMATH_CALUDE_binary_of_25_l2347_234768

/-- Represents a binary number as a list of bits (least significant bit first) -/
def BinaryRepr := List Bool

/-- Converts a natural number to its binary representation -/
def toBinary (n : ℕ) : BinaryRepr :=
  if n = 0 then [] else (n % 2 = 1) :: toBinary (n / 2)

/-- Theorem: The binary representation of 25 is 11001 -/
theorem binary_of_25 :
  toBinary 25 = [true, false, false, true, true] := by
  sorry

end NUMINAMATH_CALUDE_binary_of_25_l2347_234768


namespace NUMINAMATH_CALUDE_tangent_slope_angle_l2347_234761

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 2*x + 4

-- Define the point of interest
def point : ℝ × ℝ := (1, 3)

-- Theorem statement
theorem tangent_slope_angle :
  let slope := (deriv f) point.1
  Real.arctan slope = π/4 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_l2347_234761


namespace NUMINAMATH_CALUDE_code_transformation_correct_l2347_234721

def initial_code : List (Fin 10 × Fin 10 × Fin 10 × Fin 10) :=
  [(4, 0, 2, 2), (0, 7, 1, 0), (4, 1, 9, 9)]

def complement_to_nine (n : Fin 10) : Fin 10 :=
  9 - n

def apply_rule (segment : Fin 10 × Fin 10 × Fin 10 × Fin 10) : Fin 10 × Fin 10 × Fin 10 × Fin 10 :=
  let (a, b, c, d) := segment
  (a, complement_to_nine b, c, complement_to_nine d)

def new_code : List (Fin 10 × Fin 10 × Fin 10 × Fin 10) :=
  initial_code.map apply_rule

theorem code_transformation_correct :
  new_code = [(4, 9, 2, 7), (0, 2, 1, 9), (4, 8, 9, 0)] :=
by sorry

end NUMINAMATH_CALUDE_code_transformation_correct_l2347_234721


namespace NUMINAMATH_CALUDE_bricks_A_is_40_l2347_234793

/-- Represents the number of bricks of type A -/
def bricks_A : ℕ := sorry

/-- Represents the number of bricks of type B -/
def bricks_B : ℕ := sorry

/-- The number of bricks of type B is half the number of bricks of type A -/
axiom half_relation : bricks_B = bricks_A / 2

/-- The total number of bricks of type A and B is 60 -/
axiom total_bricks : bricks_A + bricks_B = 60

/-- Theorem stating that the number of bricks of type A is 40 -/
theorem bricks_A_is_40 : bricks_A = 40 := by sorry

end NUMINAMATH_CALUDE_bricks_A_is_40_l2347_234793


namespace NUMINAMATH_CALUDE_wade_hot_dog_truck_l2347_234750

theorem wade_hot_dog_truck (tips_per_customer : ℚ) (friday_customers : ℕ) (total_tips : ℚ) :
  tips_per_customer = 2 →
  friday_customers = 28 →
  total_tips = 296 →
  let saturday_customers := 3 * friday_customers
  let sunday_customers := (total_tips - tips_per_customer * (friday_customers + saturday_customers)) / tips_per_customer
  sunday_customers = 36 := by
sorry


end NUMINAMATH_CALUDE_wade_hot_dog_truck_l2347_234750


namespace NUMINAMATH_CALUDE_tangerine_orange_difference_l2347_234785

def initial_oranges : ℕ := 5
def initial_tangerines : ℕ := 17
def removed_oranges : ℕ := 2
def removed_tangerines : ℕ := 10
def added_oranges : ℕ := 3
def added_tangerines : ℕ := 6

theorem tangerine_orange_difference :
  (initial_tangerines - removed_tangerines + added_tangerines) -
  (initial_oranges - removed_oranges + added_oranges) = 7 := by
sorry

end NUMINAMATH_CALUDE_tangerine_orange_difference_l2347_234785


namespace NUMINAMATH_CALUDE_complex_solutions_count_l2347_234734

theorem complex_solutions_count : ∃ (S : Finset ℂ), 
  (∀ z ∈ S, (z^3 - 1) / (z^2 - z - 2) = 0) ∧ 
  (∀ z : ℂ, (z^3 - 1) / (z^2 - z - 2) = 0 → z ∈ S) ∧ 
  Finset.card S = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_solutions_count_l2347_234734


namespace NUMINAMATH_CALUDE_melissa_driving_hours_l2347_234732

/-- Calculates the total driving hours in a year given the number of trips per month,
    hours per trip, and months in a year. -/
def annual_driving_hours (trips_per_month : ℕ) (hours_per_trip : ℕ) (months_in_year : ℕ) : ℕ :=
  trips_per_month * hours_per_trip * months_in_year

/-- Proves that Melissa spends 72 hours driving in a year given the specified conditions. -/
theorem melissa_driving_hours :
  let trips_per_month : ℕ := 2
  let hours_per_trip : ℕ := 3
  let months_in_year : ℕ := 12
  annual_driving_hours trips_per_month hours_per_trip months_in_year = 72 :=
by
  sorry


end NUMINAMATH_CALUDE_melissa_driving_hours_l2347_234732


namespace NUMINAMATH_CALUDE_max_value_when_m_2_range_of_sum_when_parallel_tangents_l2347_234727

noncomputable section

def f (m : ℝ) (x : ℝ) : ℝ := (m + 1/m) * Real.log x + 1/x - x

theorem max_value_when_m_2 :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f 2 x ≥ f 2 y ∧ f 2 x = 5/2 * Real.log 2 - 3/2 := by sorry

theorem range_of_sum_when_parallel_tangents :
  ∀ (m : ℝ), m ≥ 3 →
    ∀ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ →
      (deriv (f m) x₁ = deriv (f m) x₂) →
        x₁ + x₂ > 6/5 ∧ ∀ (ε : ℝ), ε > 0 →
          ∃ (y₁ y₂ : ℝ), y₁ > 0 ∧ y₂ > 0 ∧ y₁ ≠ y₂ ∧
            (deriv (f m) y₁ = deriv (f m) y₂) ∧
            y₁ + y₂ < 6/5 + ε := by sorry

end NUMINAMATH_CALUDE_max_value_when_m_2_range_of_sum_when_parallel_tangents_l2347_234727


namespace NUMINAMATH_CALUDE_chord_bisected_by_point_one_one_l2347_234731

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Define a chord of the ellipse
def is_chord (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  is_on_ellipse x₁ y₁ ∧ is_on_ellipse x₂ y₂

-- Define the midpoint of a chord
def is_midpoint (x y x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x = (x₁ + x₂) / 2 ∧ y = (y₁ + y₂) / 2

-- Define a line equation
def line_equation (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y + c = 0

-- Theorem statement
theorem chord_bisected_by_point_one_one :
  ∀ x₁ y₁ x₂ y₂ : ℝ,
  is_chord x₁ y₁ x₂ y₂ →
  is_midpoint 1 1 x₁ y₁ x₂ y₂ →
  line_equation 4 9 (-13) x₁ y₁ ∧ line_equation 4 9 (-13) x₂ y₂ :=
sorry

end NUMINAMATH_CALUDE_chord_bisected_by_point_one_one_l2347_234731


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2347_234707

theorem perfect_square_condition (y m : ℝ) : 
  (∃ k : ℝ, y^2 - 8*y + m = k^2) → m = 16 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2347_234707


namespace NUMINAMATH_CALUDE_tim_blue_marbles_l2347_234796

theorem tim_blue_marbles (fred_marbles : ℕ) (fred_tim_ratio : ℕ) (h1 : fred_marbles = 385) (h2 : fred_tim_ratio = 35) :
  fred_marbles / fred_tim_ratio = 11 := by
  sorry

end NUMINAMATH_CALUDE_tim_blue_marbles_l2347_234796


namespace NUMINAMATH_CALUDE_custom_mul_neg_three_two_l2347_234718

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) := a * b - a^2

/-- Theorem: The custom multiplication of -3 and 2 equals -15 -/
theorem custom_mul_neg_three_two :
  custom_mul (-3) 2 = -15 := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_neg_three_two_l2347_234718


namespace NUMINAMATH_CALUDE_shaded_region_perimeter_l2347_234767

/-- The perimeter of the shaded region formed by the segments where three identical touching circles intersect is equal to the circumference of one circle. -/
theorem shaded_region_perimeter (circle_circumference : ℝ) (segment_angle : ℝ) : 
  circle_circumference > 0 →
  segment_angle = 120 →
  (3 * (segment_angle / 360) * circle_circumference) = circle_circumference :=
by sorry

end NUMINAMATH_CALUDE_shaded_region_perimeter_l2347_234767


namespace NUMINAMATH_CALUDE_assignment_statement_properties_l2347_234773

-- Define what an assignment statement is
def AssignmentStatement : Type := Unit

-- Define the properties of assignment statements
def can_provide_initial_values (a : AssignmentStatement) : Prop := sorry
def assigns_expression_value (a : AssignmentStatement) : Prop := sorry
def can_assign_multiple_times (a : AssignmentStatement) : Prop := sorry

-- Theorem stating the properties of assignment statements
theorem assignment_statement_properties (a : AssignmentStatement) :
  can_provide_initial_values a ∧
  assigns_expression_value a ∧
  can_assign_multiple_times a := by sorry

end NUMINAMATH_CALUDE_assignment_statement_properties_l2347_234773


namespace NUMINAMATH_CALUDE_sum_max_l2347_234737

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1
  sum_odd : a 1 + a 3 + a 5 = 156
  sum_even : a 2 + a 4 + a 6 = 147

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  (List.range n).map seq.a |>.sum

/-- The theorem stating that the sum reaches its maximum at n = 20 -/
theorem sum_max (seq : ArithmeticSequence) :
  ∀ k : ℕ, sum_n seq 20 ≥ sum_n seq k :=
sorry

end NUMINAMATH_CALUDE_sum_max_l2347_234737


namespace NUMINAMATH_CALUDE_unit_circle_point_x_coordinate_l2347_234736

theorem unit_circle_point_x_coordinate 
  (P : ℝ × ℝ) 
  (α : ℝ) 
  (h1 : P.1 ≥ 0 ∧ P.2 ≥ 0) -- P is in the first quadrant
  (h2 : P.1^2 + P.2^2 = 1) -- P is on the unit circle
  (h3 : P.1 = Real.cos α ∧ P.2 = Real.sin α) -- Definition of α
  (h4 : Real.cos (α + π/3) = -11/13) -- Given condition
  : P.1 = 1/26 := by
  sorry

end NUMINAMATH_CALUDE_unit_circle_point_x_coordinate_l2347_234736


namespace NUMINAMATH_CALUDE_divided_triangle_angles_l2347_234799

/-- A triangle that can be divided into several smaller triangles -/
structure DividedTriangle where
  -- The number of triangles the original triangle is divided into
  num_divisions : ℕ
  -- Assertion that there are at least two divisions
  h_at_least_two : num_divisions ≥ 2

/-- Represents the properties of the divided triangles -/
structure DivisionProperties (T : DividedTriangle) where
  -- The number of equilateral triangles in the division
  num_equilateral : ℕ
  -- The number of isosceles (non-equilateral) triangles in the division
  num_isosceles : ℕ
  -- Assertion that there is exactly one isosceles triangle
  h_one_isosceles : num_isosceles = 1
  -- Assertion that all other triangles are equilateral
  h_rest_equilateral : num_equilateral + num_isosceles = T.num_divisions

/-- The theorem stating the angles of the original triangle -/
theorem divided_triangle_angles (T : DividedTriangle) (P : DivisionProperties T) :
  ∃ (a b c : ℝ), a = 30 ∧ b = 60 ∧ c = 90 ∧ a + b + c = 180 :=
sorry

end NUMINAMATH_CALUDE_divided_triangle_angles_l2347_234799


namespace NUMINAMATH_CALUDE_optimal_import_quantity_l2347_234759

/-- Represents the annual import volume in units -/
def annual_volume : ℕ := 10000

/-- Represents the shipping cost per import in yuan -/
def shipping_cost : ℕ := 100

/-- Represents the rent cost per unit in yuan -/
def rent_cost_per_unit : ℕ := 2

/-- Calculates the number of imports per year given the quantity per import -/
def imports_per_year (quantity_per_import : ℕ) : ℕ :=
  annual_volume / quantity_per_import

/-- Calculates the total annual shipping cost -/
def annual_shipping_cost (quantity_per_import : ℕ) : ℕ :=
  shipping_cost * imports_per_year quantity_per_import

/-- Calculates the total annual rent cost -/
def annual_rent_cost (quantity_per_import : ℕ) : ℕ :=
  rent_cost_per_unit * (quantity_per_import / 2)

/-- Calculates the total annual cost (shipping + rent) -/
def total_annual_cost (quantity_per_import : ℕ) : ℕ :=
  annual_shipping_cost quantity_per_import + annual_rent_cost quantity_per_import

/-- Theorem stating that 1000 units per import minimizes the total annual cost -/
theorem optimal_import_quantity :
  ∀ q : ℕ, q > 0 → q ≤ annual_volume → total_annual_cost 1000 ≤ total_annual_cost q :=
sorry

end NUMINAMATH_CALUDE_optimal_import_quantity_l2347_234759


namespace NUMINAMATH_CALUDE_painted_cells_20210_1505_l2347_234776

/-- The number of unique cells painted by two diagonals in a rectangle --/
def painted_cells (width height : ℕ) : ℕ :=
  let gcd := width.gcd height
  let subrect_width := width / gcd
  let subrect_height := height / gcd
  let cells_per_subrect := subrect_width + subrect_height - 1
  let total_cells := 2 * gcd * cells_per_subrect
  let overlap_cells := gcd
  total_cells - overlap_cells

/-- Theorem stating the number of painted cells in a 20210 × 1505 rectangle --/
theorem painted_cells_20210_1505 :
  painted_cells 20210 1505 = 42785 := by
  sorry

end NUMINAMATH_CALUDE_painted_cells_20210_1505_l2347_234776
