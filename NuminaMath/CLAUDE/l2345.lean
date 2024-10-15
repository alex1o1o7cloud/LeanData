import Mathlib

namespace NUMINAMATH_CALUDE_fourth_root_over_seventh_root_of_seven_l2345_234579

theorem fourth_root_over_seventh_root_of_seven (x : ℝ) :
  (x ^ (1/4)) / (x ^ (1/7)) = x ^ (3/28) :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_over_seventh_root_of_seven_l2345_234579


namespace NUMINAMATH_CALUDE_power_difference_zero_l2345_234583

theorem power_difference_zero : (2^3)^2 - 4^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_zero_l2345_234583


namespace NUMINAMATH_CALUDE_nina_running_distance_l2345_234568

theorem nina_running_distance (x : ℝ) : 
  2 * x + 0.6666666666666666 = 0.8333333333333334 → 
  x = 0.08333333333333337 := by
  sorry

end NUMINAMATH_CALUDE_nina_running_distance_l2345_234568


namespace NUMINAMATH_CALUDE_min_value_of_squares_l2345_234541

theorem min_value_of_squares (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a - c = 5) :
  (∀ x y : ℝ, a > x ∧ x > y ∧ y > c → (a - x)^2 + (x - y)^2 ≥ (a - b)^2 + (b - c)^2) ∧
  (a - b)^2 + (b - c)^2 = 25/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_squares_l2345_234541


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2345_234537

theorem polynomial_factorization (x : ℝ) : 
  x^2 + 6*x + 9 - 64*x^4 = (-8*x^2 + x + 3) * (8*x^2 + x + 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2345_234537


namespace NUMINAMATH_CALUDE_circle_tangent_to_x_axis_at_origin_l2345_234576

-- Define a circle using its general equation
def Circle (D E F : ℝ) := {(x, y) : ℝ × ℝ | x^2 + y^2 + D*x + E*y + F = 0}

-- Define what it means for a circle to be tangent to the x-axis at the origin
def TangentToXAxisAtOrigin (c : Set (ℝ × ℝ)) : Prop :=
  (0, 0) ∈ c ∧ ∀ y ≠ 0, (0, y) ∉ c

-- Theorem statement
theorem circle_tangent_to_x_axis_at_origin (D E F : ℝ) :
  TangentToXAxisAtOrigin (Circle D E F) → D = 0 ∧ E ≠ 0 ∧ F ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_x_axis_at_origin_l2345_234576


namespace NUMINAMATH_CALUDE_base7_multiplication_l2345_234590

/-- Converts a number from base 7 to base 10 --/
def toBase10 (n : ℕ) : ℕ :=
  sorry

/-- Converts a number from base 10 to base 7 --/
def toBase7 (n : ℕ) : ℕ :=
  sorry

/-- Multiplies two numbers in base 7 --/
def multiplyBase7 (a b : ℕ) : ℕ :=
  toBase7 (toBase10 a * toBase10 b)

theorem base7_multiplication :
  multiplyBase7 325 6 = 2624 :=
sorry

end NUMINAMATH_CALUDE_base7_multiplication_l2345_234590


namespace NUMINAMATH_CALUDE_tax_reduction_theorem_l2345_234575

/-- Proves that if a tax rate is reduced by X%, consumption increases by 12%,
    and the resulting revenue decreases by 14.88%, then X = 24. -/
theorem tax_reduction_theorem (X : ℝ) (T C : ℝ) (hT : T > 0) (hC : C > 0) :
  let new_tax_rate := T - (X / 100) * T
  let new_consumption := C + (12 / 100) * C
  let original_revenue := T * C
  let new_revenue := new_tax_rate * new_consumption
  new_revenue = (1 - 14.88 / 100) * original_revenue →
  X = 24 := by
  sorry

end NUMINAMATH_CALUDE_tax_reduction_theorem_l2345_234575


namespace NUMINAMATH_CALUDE_percentage_problem_l2345_234581

theorem percentage_problem (y : ℝ) (h1 : y > 0) (h2 : (y / 100) * y = 9) : y = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2345_234581


namespace NUMINAMATH_CALUDE_merchant_printers_l2345_234514

/-- Calculates the number of printers bought given the total cost, cost per item, and number of keyboards --/
def calculate_printers (total_cost : ℕ) (keyboard_cost : ℕ) (printer_cost : ℕ) (num_keyboards : ℕ) : ℕ :=
  (total_cost - keyboard_cost * num_keyboards) / printer_cost

theorem merchant_printers :
  calculate_printers 2050 20 70 15 = 25 := by
  sorry

end NUMINAMATH_CALUDE_merchant_printers_l2345_234514


namespace NUMINAMATH_CALUDE_optimal_investment_l2345_234520

/-- Represents the profit function for the company's investments -/
def profit_function (x : ℝ) : ℝ :=
  let t := 3 - x
  (-x^3 + x^2 + 3*x) + (-t^2 + 5*t) - 3

/-- Theorem stating the optimal investment allocation and maximum profit -/
theorem optimal_investment :
  ∃ (x : ℝ), 
    0 ≤ x ∧ 
    x ≤ 3 ∧ 
    ∀ (y : ℝ), 0 ≤ y ∧ y ≤ 3 → profit_function x ≥ profit_function y ∧
    x = 2 ∧
    profit_function x = 25/3 := by
  sorry

#check optimal_investment

end NUMINAMATH_CALUDE_optimal_investment_l2345_234520


namespace NUMINAMATH_CALUDE_smallest_n_congruence_four_satisfies_congruence_four_is_smallest_smallest_positive_integer_congruence_l2345_234565

theorem smallest_n_congruence (n : ℕ) : n > 0 ∧ 13 * n ≡ 567 [MOD 5] → n ≥ 4 :=
by sorry

theorem four_satisfies_congruence : 13 * 4 ≡ 567 [MOD 5] :=
by sorry

theorem four_is_smallest (m : ℕ) : m > 0 ∧ m < 4 → ¬(13 * m ≡ 567 [MOD 5]) :=
by sorry

theorem smallest_positive_integer_congruence : 
  ∃ (n : ℕ), n > 0 ∧ 13 * n ≡ 567 [MOD 5] ∧ 
  ∀ (m : ℕ), m > 0 ∧ 13 * m ≡ 567 [MOD 5] → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_four_satisfies_congruence_four_is_smallest_smallest_positive_integer_congruence_l2345_234565


namespace NUMINAMATH_CALUDE_frog_jump_probability_l2345_234580

/-- Represents the probability of ending on a horizontal side -/
def probability_horizontal_end (x y : ℝ) : ℝ := sorry

/-- The rectangle's dimensions -/
def rectangle_width : ℝ := 5
def rectangle_height : ℝ := 5

/-- The frog's starting position -/
def start_x : ℝ := 2
def start_y : ℝ := 3

/-- Theorem stating the probability of ending on a horizontal side -/
theorem frog_jump_probability :
  probability_horizontal_end start_x start_y = 13 / 14 := by sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l2345_234580


namespace NUMINAMATH_CALUDE_reflection_sum_l2345_234587

/-- Given a point A with coordinates (x, y), when reflected over the y-axis to point B,
    the sum of all coordinate values of A and B equals 2y. -/
theorem reflection_sum (x y : ℝ) : 
  let A := (x, y)
  let B := (-x, y)
  x + y + (-x) + y = 2 * y := by sorry

end NUMINAMATH_CALUDE_reflection_sum_l2345_234587


namespace NUMINAMATH_CALUDE_prob_king_queen_is_16_2862_l2345_234516

/-- Represents a standard deck of cards with Jokers -/
structure Deck :=
  (total_cards : ℕ)
  (num_kings : ℕ)
  (num_queens : ℕ)
  (num_jokers : ℕ)

/-- The probability of drawing a King then a Queen from the deck -/
def prob_king_then_queen (d : Deck) : ℚ :=
  (d.num_kings * d.num_queens : ℚ) / ((d.total_cards * (d.total_cards - 1)) : ℚ)

/-- Our specific deck -/
def our_deck : Deck :=
  { total_cards := 54
  , num_kings := 4
  , num_queens := 4
  , num_jokers := 2 }

theorem prob_king_queen_is_16_2862 :
  prob_king_then_queen our_deck = 16 / 2862 := by
  sorry

end NUMINAMATH_CALUDE_prob_king_queen_is_16_2862_l2345_234516


namespace NUMINAMATH_CALUDE_binomial_product_simplification_l2345_234559

theorem binomial_product_simplification (x : ℝ) :
  (4 * x + 3) * (x - 6) = 4 * x^2 - 21 * x - 18 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_simplification_l2345_234559


namespace NUMINAMATH_CALUDE_max_value_implies_a_range_l2345_234512

def f (a x : ℝ) : ℝ := -x^2 - 2*a*x

theorem max_value_implies_a_range (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f a x ≤ a^2) →
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ f a x = a^2) →
  -1 ≤ a ∧ a ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_range_l2345_234512


namespace NUMINAMATH_CALUDE_percentage_calculation_l2345_234539

theorem percentage_calculation (number : ℝ) (h : number = 4400) : 
  0.15 * (0.30 * (0.50 * number)) = 99 := by sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2345_234539


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l2345_234562

theorem sum_of_two_numbers (A B : ℝ) (h1 : A + B = 147) (h2 : A = 0.375 * B + 4) : A + B = 147 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l2345_234562


namespace NUMINAMATH_CALUDE_flour_recipe_total_l2345_234598

/-- The amount of flour required for Mary's cake recipe -/
def flour_recipe (flour_added : ℕ) (flour_to_add : ℕ) : ℕ :=
  flour_added + flour_to_add

/-- Theorem: The total amount of flour required by the recipe is equal to 
    the sum of the flour already added and the flour still to be added -/
theorem flour_recipe_total (flour_added flour_to_add : ℕ) :
  flour_recipe flour_added flour_to_add = flour_added + flour_to_add :=
by
  sorry

#eval flour_recipe 3 6  -- Should evaluate to 9

end NUMINAMATH_CALUDE_flour_recipe_total_l2345_234598


namespace NUMINAMATH_CALUDE_additional_money_needed_l2345_234508

def water_bottles : ℕ := 5 * 12
def original_price : ℚ := 2
def reduced_price : ℚ := 185 / 100

theorem additional_money_needed :
  water_bottles * original_price - water_bottles * reduced_price = 9 := by
  sorry

end NUMINAMATH_CALUDE_additional_money_needed_l2345_234508


namespace NUMINAMATH_CALUDE_shortest_dragon_length_l2345_234542

/-- A function that calculates the sum of digits of a positive integer -/
def digitSum (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a set of k consecutive positive integers contains a number whose digit sum is divisible by 11 -/
def isDragon (start : ℕ) (k : ℕ) : Prop :=
  ∃ i : ℕ, i < k ∧ (digitSum (start + i) % 11 = 0)

/-- The theorem stating that 39 is the smallest dragon length -/
theorem shortest_dragon_length : 
  (∀ start : ℕ, isDragon start 39) ∧ 
  (∀ k : ℕ, k < 39 → ∃ start : ℕ, ¬isDragon start k) :=
sorry

end NUMINAMATH_CALUDE_shortest_dragon_length_l2345_234542


namespace NUMINAMATH_CALUDE_min_trips_required_l2345_234585

def trays_per_trip : ℕ := 9
def trays_table1 : ℕ := 17
def trays_table2 : ℕ := 55

def total_trays : ℕ := trays_table1 + trays_table2

theorem min_trips_required : (total_trays + trays_per_trip - 1) / trays_per_trip = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_trips_required_l2345_234585


namespace NUMINAMATH_CALUDE_sum_base4_equals_l2345_234544

/-- Converts a base 4 number (represented as a list of digits) to a natural number. -/
def base4ToNat (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 4 * acc + d) 0

/-- Converts a natural number to its base 4 representation (as a list of digits). -/
def natToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else go (m / 4) ((m % 4) :: acc)
    go n []

/-- The theorem to be proved -/
theorem sum_base4_equals : 
  natToBase4 (base4ToNat [3, 0, 2] + base4ToNat [2, 2, 1] + 
              base4ToNat [1, 3, 2] + base4ToNat [0, 1, 1]) = [3, 3, 2, 2] := by
  sorry


end NUMINAMATH_CALUDE_sum_base4_equals_l2345_234544


namespace NUMINAMATH_CALUDE_eighth_group_selection_l2345_234573

/-- Represents a systematic sampling of students -/
structure StudentSampling where
  totalStudents : Nat
  sampledStudents : Nat
  groupSize : Nat
  numberGroups : Nat
  selectedFromThirdGroup : Nat

/-- Calculates the number of the student selected from a given group -/
def selectedFromGroup (s : StudentSampling) (group : Nat) : Nat :=
  s.selectedFromThirdGroup + (group - 3) * s.groupSize

/-- Theorem stating the number of the student selected from the eighth group -/
theorem eighth_group_selection (s : StudentSampling) 
  (h1 : s.totalStudents = 50)
  (h2 : s.sampledStudents = 10)
  (h3 : s.groupSize = 5)
  (h4 : s.numberGroups = 10)
  (h5 : s.selectedFromThirdGroup = 12) :
  selectedFromGroup s 8 = 37 := by
  sorry

end NUMINAMATH_CALUDE_eighth_group_selection_l2345_234573


namespace NUMINAMATH_CALUDE_infinite_solutions_imply_d_equals_five_l2345_234534

theorem infinite_solutions_imply_d_equals_five :
  (∀ (d : ℝ), (∃ (S : Set ℝ), Set.Infinite S ∧ ∀ y ∈ S, 3 * (5 + d * y) = 15 * y + 15) → d = 5) :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_imply_d_equals_five_l2345_234534


namespace NUMINAMATH_CALUDE_sum_of_digits_square_n_l2345_234505

/-- The number formed by repeating the digit 7 eight times -/
def n : ℕ := 77777777

/-- Sum of digits function -/
def sum_of_digits (k : ℕ) : ℕ := sorry

/-- The theorem to be proved -/
theorem sum_of_digits_square_n : sum_of_digits (n^2) = 13 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_square_n_l2345_234505


namespace NUMINAMATH_CALUDE_angle_q_sum_of_sin_cos_l2345_234501

theorem angle_q_sum_of_sin_cos (x : ℝ) (hx : x ≠ 0) :
  let P : ℝ × ℝ := (x, -1)
  let tan_q : ℝ := -x
  let sin_q : ℝ := -1 / Real.sqrt (1 + x^2)
  let cos_q : ℝ := x / Real.sqrt (1 + x^2)
  (sin_q + cos_q = 0) ∨ (sin_q + cos_q = -Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_angle_q_sum_of_sin_cos_l2345_234501


namespace NUMINAMATH_CALUDE_df_length_l2345_234570

/-- Right triangle ABC with square ABDE and angle bisector intersection -/
structure RightTriangleWithSquare where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  -- Conditions
  right_angle : (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = 0
  ac_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 21
  bc_length : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 28
  square_abde : (D.1 - A.1) * (B.1 - A.1) + (D.2 - A.2) * (B.2 - A.2) = 0 ∧
                (E.1 - B.1) * (D.1 - B.1) + (E.2 - B.2) * (D.2 - B.2) = 0 ∧
                Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) = Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  f_on_de : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ F = (t * D.1 + (1 - t) * E.1, t * D.2 + (1 - t) * E.2)
  f_on_bisector : ∃ s : ℝ, s > 0 ∧ F = (C.1 + s * (A.1 + B.1 - 2 * C.1), C.2 + s * (A.2 + B.2 - 2 * C.2))

/-- The length of DF is 15 -/
theorem df_length (t : RightTriangleWithSquare) : 
  Real.sqrt ((t.D.1 - t.F.1)^2 + (t.D.2 - t.F.2)^2) = 15 := by
  sorry


end NUMINAMATH_CALUDE_df_length_l2345_234570


namespace NUMINAMATH_CALUDE_log_equation_solution_l2345_234515

theorem log_equation_solution (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  p * (q + 1) = q →
  (Real.log p + Real.log q = Real.log (p + q) ↔ p = q / (q + 1)) := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2345_234515


namespace NUMINAMATH_CALUDE_inequality_solution_l2345_234532

theorem inequality_solution (x : ℝ) :
  (1 - (2*x - 2)/5 < (3 - 4*x)/2) ↔ (x < 1/16) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2345_234532


namespace NUMINAMATH_CALUDE_remainder_77_pow_77_minus_15_mod_19_l2345_234554

theorem remainder_77_pow_77_minus_15_mod_19 : 77^77 - 15 ≡ 5 [MOD 19] := by
  sorry

end NUMINAMATH_CALUDE_remainder_77_pow_77_minus_15_mod_19_l2345_234554


namespace NUMINAMATH_CALUDE_max_m_plus_2n_max_fraction_min_fraction_l2345_234571

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 14*y + 45 = 0

-- Define a point M on the circle C
def M (m n : ℝ) : Prop := C m n

-- Theorem for the maximum value of m + 2n
theorem max_m_plus_2n :
  ∃ (max : ℝ), (∀ m n, M m n → m + 2*n ≤ max) ∧ (∃ m n, M m n ∧ m + 2*n = max) ∧ max = 16 + 2*Real.sqrt 10 :=
sorry

-- Theorem for the maximum value of (n-3)/(m+2)
theorem max_fraction :
  ∃ (max : ℝ), (∀ m n, M m n → (n - 3) / (m + 2) ≤ max) ∧ (∃ m n, M m n ∧ (n - 3) / (m + 2) = max) ∧ max = 2 + Real.sqrt 3 :=
sorry

-- Theorem for the minimum value of (n-3)/(m+2)
theorem min_fraction :
  ∃ (min : ℝ), (∀ m n, M m n → min ≤ (n - 3) / (m + 2)) ∧ (∃ m n, M m n ∧ (n - 3) / (m + 2) = min) ∧ min = 2 - Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_max_m_plus_2n_max_fraction_min_fraction_l2345_234571


namespace NUMINAMATH_CALUDE_even_function_implies_a_eq_neg_one_l2345_234535

def f (a x : ℝ) : ℝ := x^2 - 2*(a+1)*x + 1

theorem even_function_implies_a_eq_neg_one (a : ℝ) :
  (∀ x, f a x = f a (-x)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_eq_neg_one_l2345_234535


namespace NUMINAMATH_CALUDE_problem_solution_l2345_234588

def star (a b : ℚ) : ℚ := 2 * a - b

theorem problem_solution (x : ℚ) (h : star x (star 1 3) = 2) : x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2345_234588


namespace NUMINAMATH_CALUDE_pizza_piece_cost_l2345_234556

/-- Given that Luigi bought 4 pizzas for $80 and each pizza was cut into 5 pieces,
    prove that the cost of each pizza piece is $4. -/
theorem pizza_piece_cost (total_pizzas : ℕ) (total_cost : ℚ) (pieces_per_pizza : ℕ) :
  total_pizzas = 4 →
  total_cost = 80 →
  pieces_per_pizza = 5 →
  total_cost / (total_pizzas * pieces_per_pizza : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_pizza_piece_cost_l2345_234556


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l2345_234503

theorem gcd_factorial_eight_and_factorial_six_squared :
  Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 7200 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l2345_234503


namespace NUMINAMATH_CALUDE_f_upper_bound_l2345_234599

-- Define the set A
def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 1}

-- Define the properties of function f
def is_valid_f (f : ℝ → ℝ) : Prop :=
  f 1 = 1 ∧
  (∀ x ∈ A, f x ≥ 0) ∧
  (∀ x y, x ∈ A → y ∈ A → x + y ∈ A → f (x + y) ≥ f x + f y)

-- Theorem statement
theorem f_upper_bound 
  (f : ℝ → ℝ) 
  (hf : is_valid_f f) :
  ∀ x ∈ A, f x ≤ 2 * x :=
sorry

end NUMINAMATH_CALUDE_f_upper_bound_l2345_234599


namespace NUMINAMATH_CALUDE_remainder_theorem_l2345_234546

theorem remainder_theorem : (7 * 10^23 + 3^25) % 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2345_234546


namespace NUMINAMATH_CALUDE_cubic_odd_extremum_sum_l2345_234521

/-- A cubic function f(x) = ax³ + bx² + cx -/
def f (a b c x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x

/-- f is an odd function -/
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- f has an extremum at x=1 -/
def has_extremum_at_one (f : ℝ → ℝ) : Prop := 
  ∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ f x ≥ f 1

theorem cubic_odd_extremum_sum (a b c : ℝ) : 
  is_odd_function (f a b c) → has_extremum_at_one (f a b c) → 3*a + b + c = 0 := by
  sorry


end NUMINAMATH_CALUDE_cubic_odd_extremum_sum_l2345_234521


namespace NUMINAMATH_CALUDE_chris_age_l2345_234595

def problem (a b c : ℚ) : Prop :=
  -- The average of Amy's, Ben's, and Chris's ages is 10
  (a + b + c) / 3 = 10 ∧
  -- Five years ago, Chris was twice the age that Amy is now
  c - 5 = 2 * a ∧
  -- In 4 years, Ben's age will be 3/4 of Amy's age at that time
  b + 4 = 3 / 4 * (a + 4)

theorem chris_age (a b c : ℚ) (h : problem a b c) : c = 263 / 11 := by
  sorry

end NUMINAMATH_CALUDE_chris_age_l2345_234595


namespace NUMINAMATH_CALUDE_quadratic_function_extrema_l2345_234509

def f (x : ℝ) := 3 * x^2 + 6 * x - 5

theorem quadratic_function_extrema :
  ∃ (min max : ℝ), 
    (∀ x ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f x ≥ min ∧ f x ≤ max) ∧
    (∃ x₁ ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f x₁ = min) ∧
    (∃ x₂ ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f x₂ = max) ∧
    min = -8 ∧ max = 19 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_extrema_l2345_234509


namespace NUMINAMATH_CALUDE_complex_exponential_form_l2345_234593

theorem complex_exponential_form (z : ℂ) : z = 1 + Complex.I * Real.sqrt 3 → 
  ∃ (r : ℝ) (θ : ℝ), z = r * Complex.exp (Complex.I * θ) ∧ θ = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_form_l2345_234593


namespace NUMINAMATH_CALUDE_sugar_profit_problem_l2345_234531

/-- Proves the quantity of sugar sold at 18% profit given the conditions -/
theorem sugar_profit_problem (total_sugar : ℝ) (profit_rate_1 profit_rate_2 overall_profit : ℝ) 
  (h1 : total_sugar = 1000)
  (h2 : profit_rate_1 = 0.08)
  (h3 : profit_rate_2 = 0.18)
  (h4 : overall_profit = 0.14)
  (h5 : ∃ x : ℝ, x ≥ 0 ∧ x ≤ total_sugar ∧ 
    profit_rate_1 * x + profit_rate_2 * (total_sugar - x) = overall_profit * total_sugar) :
  ∃ y : ℝ, y = 600 ∧ y = total_sugar - 
    Classical.choose (h5 : ∃ x : ℝ, x ≥ 0 ∧ x ≤ total_sugar ∧ 
      profit_rate_1 * x + profit_rate_2 * (total_sugar - x) = overall_profit * total_sugar) :=
by sorry

end NUMINAMATH_CALUDE_sugar_profit_problem_l2345_234531


namespace NUMINAMATH_CALUDE_subcommittee_formation_count_l2345_234596

-- Define the number of Republicans and Democrats in the Senate committee
def totalRepublicans : ℕ := 10
def totalDemocrats : ℕ := 7

-- Define the number of Republicans and Democrats needed for the subcommittee
def subcommitteeRepublicans : ℕ := 4
def subcommitteeDemocrats : ℕ := 3

-- Theorem statement
theorem subcommittee_formation_count :
  (Nat.choose totalRepublicans subcommitteeRepublicans) *
  (Nat.choose totalDemocrats subcommitteeDemocrats) = 7350 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_count_l2345_234596


namespace NUMINAMATH_CALUDE_vacation_pictures_l2345_234522

/-- The number of pictures Megan took at the zoo -/
def zoo_pictures : ℕ := 15

/-- The number of pictures Megan took at the museum -/
def museum_pictures : ℕ := 18

/-- The number of pictures Megan deleted -/
def deleted_pictures : ℕ := 31

/-- The number of pictures Megan still has from her vacation -/
def remaining_pictures : ℕ := zoo_pictures + museum_pictures - deleted_pictures

theorem vacation_pictures : remaining_pictures = 2 := by
  sorry

end NUMINAMATH_CALUDE_vacation_pictures_l2345_234522


namespace NUMINAMATH_CALUDE_christopher_stroll_distance_l2345_234569

/-- Given Christopher's strolling speed and time, calculate the distance he strolled. -/
theorem christopher_stroll_distance 
  (speed : ℝ) 
  (time : ℝ) 
  (h1 : speed = 4) 
  (h2 : time = 1.25) : 
  speed * time = 5 := by
  sorry

end NUMINAMATH_CALUDE_christopher_stroll_distance_l2345_234569


namespace NUMINAMATH_CALUDE_line_parameterization_l2345_234578

/-- Given a line y = 2x - 30 parameterized by (x, y) = (f(t), 20t - 10), 
    prove that f(t) = 10t + 10 -/
theorem line_parameterization (f : ℝ → ℝ) : 
  (∀ t : ℝ, 2 * (f t) - 30 = 20 * t - 10) → 
  (∀ t : ℝ, f t = 10 * t + 10) := by
  sorry

end NUMINAMATH_CALUDE_line_parameterization_l2345_234578


namespace NUMINAMATH_CALUDE_geometric_relations_l2345_234540

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation
variable (parallel : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- Define the subset relation
variable (subset : Line → Plane → Prop)

-- Define specific planes and lines
variable (α β : Plane)
variable (a b : Line)

-- State the theorem
theorem geometric_relations :
  (subset b α ∧ ¬subset a α →
    (∀ x y, parallel_lines x y → parallel_line_plane x α) ∧
    ¬(∀ x y, parallel_line_plane x α → parallel_lines x y)) ∧
  (subset a α ∧ subset b α →
    ¬(parallel α β ↔ (parallel α β ∧ parallel_line_plane b β))) :=
sorry

end NUMINAMATH_CALUDE_geometric_relations_l2345_234540


namespace NUMINAMATH_CALUDE_extreme_value_of_f_l2345_234586

-- Define the function f
def f (x a b c : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the solution set condition
def solution_set (f : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x, f x < 0 ↔ (x < m + 1 ∧ x ≠ m)

-- Theorem statement
theorem extreme_value_of_f (a b c m : ℝ) :
  solution_set (f · a b c) m →
  ∃ x, f x a b c = -4/27 ∧ ∀ y, f y a b c ≥ -4/27 :=
sorry

end NUMINAMATH_CALUDE_extreme_value_of_f_l2345_234586


namespace NUMINAMATH_CALUDE_probability_four_blue_l2345_234506

/-- The number of blue marbles initially in the bag -/
def initial_blue : ℕ := 10

/-- The number of red marbles initially in the bag -/
def initial_red : ℕ := 5

/-- The total number of draws -/
def total_draws : ℕ := 10

/-- The number of blue marbles we want to draw -/
def target_blue : ℕ := 4

/-- The probability of drawing a blue marble, approximated as constant throughout the process -/
def p_blue : ℚ := 2/3

/-- The probability of drawing a red marble, approximated as constant throughout the process -/
def p_red : ℚ := 1/3

/-- The probability of drawing exactly 4 blue marbles out of 10 draws -/
theorem probability_four_blue : 
  (Nat.choose total_draws target_blue : ℚ) * p_blue^target_blue * p_red^(total_draws - target_blue) = 
  (210 * 16 : ℚ) / (81 * 729) := by sorry

end NUMINAMATH_CALUDE_probability_four_blue_l2345_234506


namespace NUMINAMATH_CALUDE_xyz_minimum_l2345_234528

theorem xyz_minimum (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 2) :
  ∀ (a b c : ℝ), a^3 + b^3 + c^3 - 3*a*b*c = 2 → x*y*z ≤ a*b*c := by
  sorry

end NUMINAMATH_CALUDE_xyz_minimum_l2345_234528


namespace NUMINAMATH_CALUDE_crazy_silly_school_books_to_read_l2345_234526

/-- The number of books still to read in a series -/
def books_to_read (total_books read_books : ℕ) : ℕ :=
  total_books - read_books

/-- Theorem: For the 'crazy silly school' series, the number of books still to read is 10 -/
theorem crazy_silly_school_books_to_read :
  books_to_read 22 12 = 10 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_books_to_read_l2345_234526


namespace NUMINAMATH_CALUDE_system_solution_l2345_234557

theorem system_solution :
  ∀ x y : ℂ,
  (x^2 + y^2 = x*y ∧ x + y = x*y) ↔
  ((x = 0 ∧ y = 0) ∨
   (x = (3 + Complex.I * Real.sqrt 3) / 2 ∧ y = (3 - Complex.I * Real.sqrt 3) / 2) ∨
   (x = (3 - Complex.I * Real.sqrt 3) / 2 ∧ y = (3 + Complex.I * Real.sqrt 3) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2345_234557


namespace NUMINAMATH_CALUDE_problem_solution_l2345_234574

theorem problem_solution (x y : ℝ) (h : x^2 * (y^2 + 1) = 1) :
  (x * y < 1) ∧ (x^2 * y ≥ -1/2) ∧ (x^2 + x * y ≤ 5/4) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2345_234574


namespace NUMINAMATH_CALUDE_collinear_points_p_value_l2345_234584

/-- Three points are collinear if they lie on the same straight line -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

theorem collinear_points_p_value :
  ∀ p : ℝ, collinear 1 (-2) 3 4 6 (p/3) → p = 39 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_p_value_l2345_234584


namespace NUMINAMATH_CALUDE_composite_square_area_l2345_234511

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a square composed of rectangles -/
structure CompositeSquare where
  rectangle : Rectangle
  
/-- The perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- The side length of the composite square -/
def CompositeSquare.sideLength (s : CompositeSquare) : ℝ := s.rectangle.length + s.rectangle.width

/-- The area of the composite square -/
def CompositeSquare.area (s : CompositeSquare) : ℝ := (s.sideLength) ^ 2

theorem composite_square_area (s : CompositeSquare) 
  (h : s.rectangle.perimeter = 40) : s.area = 400 := by
  sorry

end NUMINAMATH_CALUDE_composite_square_area_l2345_234511


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l2345_234519

/-- A right triangle with sides 6, 8, and 10 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  sides : a = 6 ∧ b = 8 ∧ c = 10

/-- A square inscribed in the triangle with side along leg of length 6 -/
def inscribed_square_x (t : RightTriangle) (x : ℝ) : Prop :=
  x > 0 ∧ x ≤ t.a ∧ x / t.a = (t.b - x) / t.c

/-- A square inscribed in the triangle with side along leg of length 8 -/
def inscribed_square_y (t : RightTriangle) (y : ℝ) : Prop :=
  y > 0 ∧ y ≤ t.b ∧ y / t.b = (t.a - y) / t.c

theorem inscribed_squares_ratio (t : RightTriangle) (x y : ℝ) 
  (hx : inscribed_square_x t x) (hy : inscribed_square_y t y) : 
  x / y = 9 / 8 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l2345_234519


namespace NUMINAMATH_CALUDE_rooster_weight_unit_l2345_234564

/-- Represents units of mass measurement -/
inductive MassUnit
  | Kilogram
  | Ton
  | Gram

/-- The weight of a rooster in some unit -/
def roosterWeight : ℝ := 3

/-- Predicate to determine if a unit is appropriate for measuring rooster weight -/
def isAppropriateUnit (unit : MassUnit) : Prop :=
  match unit with
  | MassUnit.Kilogram => True
  | _ => False

theorem rooster_weight_unit :
  isAppropriateUnit MassUnit.Kilogram :=
sorry

end NUMINAMATH_CALUDE_rooster_weight_unit_l2345_234564


namespace NUMINAMATH_CALUDE_difference_of_numbers_l2345_234592

theorem difference_of_numbers (x y : ℝ) 
  (sum_eq : x + y = 8) 
  (diff_squares : x^2 - y^2 = 32) : 
  |x - y| = 4 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_numbers_l2345_234592


namespace NUMINAMATH_CALUDE_dave_guitar_strings_l2345_234560

/-- The number of guitar strings Dave breaks per night -/
def strings_per_night : ℕ := 2

/-- The number of shows Dave performs per week -/
def shows_per_week : ℕ := 6

/-- The number of weeks Dave performs -/
def total_weeks : ℕ := 12

/-- The total number of guitar strings Dave needs to replace -/
def total_strings_needed : ℕ := 144

/-- Theorem stating that the total number of guitar strings Dave needs to replace is 144 -/
theorem dave_guitar_strings :
  strings_per_night * shows_per_week * total_weeks = total_strings_needed :=
by sorry

end NUMINAMATH_CALUDE_dave_guitar_strings_l2345_234560


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2345_234543

theorem expression_simplification_and_evaluation :
  let x : ℝ := Real.sqrt 3 - 1
  ((1 / (x + 1) - 1) / (x / (x^2 - 1))) = 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2345_234543


namespace NUMINAMATH_CALUDE_picture_book_shelves_l2345_234536

theorem picture_book_shelves (total_books : ℕ) (mystery_shelves : ℕ) (books_per_shelf : ℕ) :
  total_books = 32 →
  mystery_shelves = 5 →
  books_per_shelf = 4 →
  (total_books - mystery_shelves * books_per_shelf) / books_per_shelf = 3 :=
by sorry

end NUMINAMATH_CALUDE_picture_book_shelves_l2345_234536


namespace NUMINAMATH_CALUDE_worker_c_completion_time_l2345_234558

/-- Given workers a, b, and c who can complete a work in the specified times,
    prove that c can complete the work alone in 40 days. -/
theorem worker_c_completion_time
  (total_work : ℝ)
  (time_a : ℝ) (time_b : ℝ) (time_c : ℝ)
  (total_time : ℝ) (c_left_early : ℝ)
  (h_time_a : time_a = 30)
  (h_time_b : time_b = 30)
  (h_total_time : total_time = 12)
  (h_c_left_early : c_left_early = 4)
  (h_work_completed : (total_work / time_a + total_work / time_b + total_work / time_c) *
    (total_time - c_left_early) +
    (total_work / time_a + total_work / time_b) * c_left_early = total_work) :
  time_c = 40 := by
sorry

end NUMINAMATH_CALUDE_worker_c_completion_time_l2345_234558


namespace NUMINAMATH_CALUDE_shampoo_bottles_l2345_234566

theorem shampoo_bottles (small_capacity large_capacity current_amount : ℕ) 
  (h1 : small_capacity = 40)
  (h2 : large_capacity = 800)
  (h3 : current_amount = 120) : 
  (large_capacity - current_amount) / small_capacity = 17 := by
  sorry

end NUMINAMATH_CALUDE_shampoo_bottles_l2345_234566


namespace NUMINAMATH_CALUDE_probability_three_common_books_l2345_234538

def total_books : ℕ := 12
def books_selected : ℕ := 6
def books_in_common : ℕ := 3

def probability_common_books : ℚ :=
  (Nat.choose total_books books_in_common * 
   Nat.choose (total_books - books_in_common) (books_selected - books_in_common) * 
   Nat.choose (total_books - books_selected) (books_selected - books_in_common)) /
  (Nat.choose total_books books_selected * Nat.choose total_books books_selected)

theorem probability_three_common_books :
  probability_common_books = 140 / 323 := by sorry

end NUMINAMATH_CALUDE_probability_three_common_books_l2345_234538


namespace NUMINAMATH_CALUDE_simplify_expression_l2345_234597

theorem simplify_expression (b : ℝ) : 3*b*(3*b^2 + 2*b - 4) - 2*b^2 = 9*b^3 + 4*b^2 - 12*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2345_234597


namespace NUMINAMATH_CALUDE_inequality_solution_l2345_234572

theorem inequality_solution (x : ℝ) :
  (3 * (x + 2) - 1 ≥ 5 - 2 * (x - 2)) ↔ (x ≥ 4 / 5) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2345_234572


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l2345_234533

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- State the theorem
theorem composition_of_even_is_even (f : ℝ → ℝ) (h : EvenFunction f) :
  EvenFunction (f ∘ f) := by
  sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l2345_234533


namespace NUMINAMATH_CALUDE_simplify_xy_squared_l2345_234513

theorem simplify_xy_squared (x y : ℝ) : 5 * x * y^2 - 6 * x * y^2 = -x * y^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_xy_squared_l2345_234513


namespace NUMINAMATH_CALUDE_percentage_speaking_both_truth_and_lies_l2345_234500

/-- In a class with students who speak truth, lies, or both, prove the percentage
    of students speaking both truth and lies. -/
theorem percentage_speaking_both_truth_and_lies 
  (probTruth : ℝ) 
  (probLies : ℝ) 
  (probTruthOrLies : ℝ) 
  (h1 : probTruth = 0.3) 
  (h2 : probLies = 0.2) 
  (h3 : probTruthOrLies = 0.4) : 
  probTruth + probLies - probTruthOrLies = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_percentage_speaking_both_truth_and_lies_l2345_234500


namespace NUMINAMATH_CALUDE_machine_worked_two_minutes_l2345_234549

/-- Calculates the working time of a machine given its production rate and total output -/
def machine_working_time (shirts_per_minute : ℕ) (total_shirts : ℕ) : ℚ :=
  (total_shirts : ℚ) / (shirts_per_minute : ℚ)

/-- Proves that a machine making 3 shirts per minute that made 6 shirts worked for 2 minutes -/
theorem machine_worked_two_minutes :
  machine_working_time 3 6 = 2 := by sorry

end NUMINAMATH_CALUDE_machine_worked_two_minutes_l2345_234549


namespace NUMINAMATH_CALUDE_bruno_pen_units_l2345_234523

/-- Given that Bruno buys 2.5 units of pens and ends up with 30 pens in total,
    prove that the unit he is using is 12 pens per unit. -/
theorem bruno_pen_units (units : ℝ) (total_pens : ℕ) :
  units = 2.5 ∧ total_pens = 30 → (total_pens : ℝ) / units = 12 := by
  sorry

end NUMINAMATH_CALUDE_bruno_pen_units_l2345_234523


namespace NUMINAMATH_CALUDE_max_regions_1002_1000_l2345_234545

/-- The maximum number of regions formed by n lines in a plane -/
def max_regions (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- The number of new regions added by each line through A after the first -/
def new_regions_per_line_A (lines_through_B : ℕ) : ℕ := lines_through_B + 2

/-- The maximum number of regions formed by m lines through A and n lines through B -/
def max_regions_two_points (m n : ℕ) : ℕ :=
  max_regions n + (new_regions_per_line_A n) + (m - 1) * (new_regions_per_line_A n)

theorem max_regions_1002_1000 :
  max_regions_two_points 1002 1000 = 1504503 := by
  sorry

end NUMINAMATH_CALUDE_max_regions_1002_1000_l2345_234545


namespace NUMINAMATH_CALUDE_line_equation_l2345_234548

/-- A line passing through the point (-2, 5) with slope -3/4 has the equation 3x + 4y - 14 = 0. -/
theorem line_equation (x y : ℝ) : 
  (∃ (L : Set (ℝ × ℝ)), 
    ((-2, 5) ∈ L) ∧ 
    (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ L → (x₂, y₂) ∈ L → x₁ ≠ x₂ → (y₂ - y₁) / (x₂ - x₁) = -3/4) ∧
    ((x, y) ∈ L ↔ 3*x + 4*y - 14 = 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_line_equation_l2345_234548


namespace NUMINAMATH_CALUDE_bicycle_shop_period_l2345_234547

/-- Proves that the number of weeks that passed is 4, given the conditions of the bicycle shop problem. -/
theorem bicycle_shop_period (initial_stock : ℕ) (weekly_addition : ℕ) (sold : ℕ) (final_stock : ℕ)
  (h1 : initial_stock = 51)
  (h2 : weekly_addition = 3)
  (h3 : sold = 18)
  (h4 : final_stock = 45) :
  ∃ weeks : ℕ, weeks = 4 ∧ initial_stock + weekly_addition * weeks - sold = final_stock :=
by
  sorry

#check bicycle_shop_period

end NUMINAMATH_CALUDE_bicycle_shop_period_l2345_234547


namespace NUMINAMATH_CALUDE_expression_evaluation_l2345_234567

theorem expression_evaluation : (50 - (2050 - 150)) + (2050 - (150 - 50)) = 100 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2345_234567


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2345_234524

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (f x + x * y) * f (x - 3 * y) + (f y + x * y) * f (3 * x - y) = (f (x + y))^2) : 
  (∀ x : ℝ, f x = x^2) ∨ (∀ x : ℝ, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2345_234524


namespace NUMINAMATH_CALUDE_regular_polygon_with_four_to_one_angle_ratio_l2345_234591

/-- A regular polygon where the interior angle is exactly 4 times the exterior angle has 10 sides -/
theorem regular_polygon_with_four_to_one_angle_ratio (n : ℕ) : 
  n > 2 → 
  (360 / n : ℚ) * 4 = (180 - 360 / n : ℚ) → 
  n = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_four_to_one_angle_ratio_l2345_234591


namespace NUMINAMATH_CALUDE_system_solution_l2345_234510

def equation1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y = -12
def equation2 (x z : ℝ) : Prop := x^2 + z^2 - 6*x - 2*z = -5
def equation3 (y z : ℝ) : Prop := y^2 + z^2 - 8*y - 2*z = -7

def is_solution (x y z : ℝ) : Prop :=
  equation1 x y ∧ equation2 x z ∧ equation3 y z

theorem system_solution :
  (∀ x y z : ℝ, is_solution x y z ↔
    ((x = 1 ∧ y = 1 ∧ z = 0) ∨
     (x = 1 ∧ y = 1 ∧ z = 2) ∨
     (x = 1 ∧ y = 7 ∧ z = 0) ∨
     (x = 1 ∧ y = 7 ∧ z = 2) ∨
     (x = 5 ∧ y = 1 ∧ z = 0) ∨
     (x = 5 ∧ y = 1 ∧ z = 2) ∨
     (x = 5 ∧ y = 7 ∧ z = 0) ∨
     (x = 5 ∧ y = 7 ∧ z = 2))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2345_234510


namespace NUMINAMATH_CALUDE_frisbee_price_problem_l2345_234550

/-- Represents the price and quantity of frisbees sold at that price -/
structure FrisbeeGroup where
  price : ℝ
  quantity : ℕ

/-- Calculates the total revenue from a group of frisbees -/
def revenue (group : FrisbeeGroup) : ℝ :=
  group.price * group.quantity

theorem frisbee_price_problem (total_frisbees : ℕ) (total_revenue : ℝ) 
    (cheap_frisbees : FrisbeeGroup) (expensive_frisbees : FrisbeeGroup) : 
    total_frisbees = 60 →
    cheap_frisbees.price = 4 →
    cheap_frisbees.quantity ≥ 20 →
    cheap_frisbees.quantity + expensive_frisbees.quantity = total_frisbees →
    revenue cheap_frisbees + revenue expensive_frisbees = total_revenue →
    total_revenue = 200 →
    expensive_frisbees.price = 3 := by
  sorry

end NUMINAMATH_CALUDE_frisbee_price_problem_l2345_234550


namespace NUMINAMATH_CALUDE_ellipse_equation_constants_l2345_234563

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point
  passingPoint : Point
  a : ℝ
  b : ℝ
  h : ℝ
  k : ℝ

/-- Check if a point satisfies the ellipse equation -/
def satisfiesEllipseEquation (e : Ellipse) (p : Point) : Prop :=
  (p.x - e.h)^2 / e.a^2 + (p.y - e.k)^2 / e.b^2 = 1

/-- The main theorem to prove -/
theorem ellipse_equation_constants : ∃ (e : Ellipse),
  e.focus1 = ⟨2, 2⟩ ∧
  e.focus2 = ⟨2, 6⟩ ∧
  e.passingPoint = ⟨14, -3⟩ ∧
  e.a > 0 ∧
  e.b > 0 ∧
  satisfiesEllipseEquation e e.passingPoint ∧
  e.a = 8 * Real.sqrt 3 ∧
  e.b = 14 ∧
  e.h = 2 ∧
  e.k = 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_constants_l2345_234563


namespace NUMINAMATH_CALUDE_train_passing_bridge_time_l2345_234525

/-- The time it takes for a train to pass a bridge -/
theorem train_passing_bridge_time (train_length bridge_length : ℝ) (train_speed_kmh : ℝ) :
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let time := total_distance / train_speed_ms
  train_length = 250 ∧ bridge_length = 150 ∧ train_speed_kmh = 35 →
  ∃ ε > 0, |time - 41.1528| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_train_passing_bridge_time_l2345_234525


namespace NUMINAMATH_CALUDE_complement_union_theorem_l2345_234594

def U : Set ℕ := {x : ℕ | x > 0 ∧ x < 9}
def A : Set ℕ := {1, 2, 3, 4, 5, 6}
def B : Set ℕ := {4, 5, 6}

theorem complement_union_theorem :
  (U \ A) ∪ B = {4, 5, 6, 7, 8} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l2345_234594


namespace NUMINAMATH_CALUDE_two_machine_completion_time_l2345_234529

theorem two_machine_completion_time (t₁ t_combined : ℝ) (x : ℝ) 
  (h₁ : t₁ > 0) (h₂ : t_combined > 0) (h₃ : x > 0) 
  (h₄ : t₁ = 6) (h₅ : t_combined = 1.5) :
  (1 / t₁ + 1 / x = 1 / t_combined) ↔ 
  (1 / 6 + 1 / x = 1 / 1.5) :=
sorry

end NUMINAMATH_CALUDE_two_machine_completion_time_l2345_234529


namespace NUMINAMATH_CALUDE_remainder_problem_l2345_234577

theorem remainder_problem (x : ℤ) : x % 62 = 7 → (x + 11) % 31 = 18 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2345_234577


namespace NUMINAMATH_CALUDE_gcd_x_y_eq_25_l2345_234518

/-- The sum of all even integers between 13 and 63 (inclusive) -/
def x : ℕ := (14 + 62) * 25 / 2

/-- The count of even integers between 13 and 63 (inclusive) -/
def y : ℕ := 25

/-- Theorem stating that the greatest common divisor of x and y is 25 -/
theorem gcd_x_y_eq_25 : Nat.gcd x y = 25 := by sorry

end NUMINAMATH_CALUDE_gcd_x_y_eq_25_l2345_234518


namespace NUMINAMATH_CALUDE_speed_increase_time_l2345_234553

/-- Represents the journey of Xavier from point P to point Q -/
structure Journey where
  initialSpeed : ℝ
  increasedSpeed : ℝ
  totalDistance : ℝ
  totalTime : ℝ

/-- Theorem stating that Xavier increases his speed after 24 minutes -/
theorem speed_increase_time (j : Journey)
  (h1 : j.initialSpeed = 40)
  (h2 : j.increasedSpeed = 60)
  (h3 : j.totalDistance = 56)
  (h4 : j.totalTime = 0.8) : 
  ∃ t : ℝ, t * j.initialSpeed + (j.totalTime - t) * j.increasedSpeed = j.totalDistance ∧ t = 0.4 := by
  sorry

#check speed_increase_time

end NUMINAMATH_CALUDE_speed_increase_time_l2345_234553


namespace NUMINAMATH_CALUDE_last_digit_of_power_l2345_234507

theorem last_digit_of_power (a b : ℕ) (ha : a = 954950230952380948328708) (hb : b = 470128749397540235934750230) :
  (a^b) % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_power_l2345_234507


namespace NUMINAMATH_CALUDE_triangle_problem_l2345_234502

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 0 < a ∧ 0 < b ∧ 0 < c →
  A + B + C = π →
  b * Real.cos A + (1/2) * a = c →
  (B = π/3 ∧
   (c = 5 → b = 7 → a = 8 ∧ (1/2) * a * c * Real.sin B = 10 * Real.sqrt 3) ∧
   (c = 5 → C = π/4 → a = (5 * Real.sqrt 3 + 5)/2 ∧ 
    (1/2) * a * c * Real.sin B = (75 + 25 * Real.sqrt 3)/8)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2345_234502


namespace NUMINAMATH_CALUDE_eve_envelope_count_l2345_234561

def envelope_numbers : List ℕ := [1, 2, 4, 8, 16, 32, 64, 128]

theorem eve_envelope_count :
  ∀ (eve_numbers alie_numbers : List ℕ),
    eve_numbers ++ alie_numbers = envelope_numbers →
    eve_numbers.sum = alie_numbers.sum + 31 →
    eve_numbers.length = 5 := by
  sorry

end NUMINAMATH_CALUDE_eve_envelope_count_l2345_234561


namespace NUMINAMATH_CALUDE_triangle_condition_implies_isosceles_right_l2345_234527

-- Define a triangle with side lengths a, b, and c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the condition given in the problem
def satisfiesCondition (t : Triangle) : Prop :=
  |t.c^2 - t.a^2 - t.b^2| + (t.a - t.b)^2 = 0

-- Define an isosceles right triangle
def isIsoscelesRightTriangle (t : Triangle) : Prop :=
  t.a = t.b ∧ t.a^2 + t.b^2 = t.c^2

-- The theorem to be proved
theorem triangle_condition_implies_isosceles_right (t : Triangle) 
  (h : satisfiesCondition t) : isIsoscelesRightTriangle t :=
sorry

end NUMINAMATH_CALUDE_triangle_condition_implies_isosceles_right_l2345_234527


namespace NUMINAMATH_CALUDE_max_equal_quotient_remainder_l2345_234551

theorem max_equal_quotient_remainder (A B C : ℕ) (h1 : A = 7 * B + C) (h2 : B = C) :
  B ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_max_equal_quotient_remainder_l2345_234551


namespace NUMINAMATH_CALUDE_triangle_area_from_lines_l2345_234517

/-- The area of the triangle formed by the intersection of three lines -/
theorem triangle_area_from_lines :
  let line1 : ℝ → ℝ := λ x => 3 * x - 4
  let line2 : ℝ → ℝ := λ x => -2 * x + 16
  let y_axis : ℝ → ℝ := λ x => 0
  let intersection_x : ℝ := 4
  let intersection_y : ℝ := line1 intersection_x
  let y_intercept1 : ℝ := line1 0
  let y_intercept2 : ℝ := line2 0
  let base : ℝ := y_intercept2 - y_intercept1
  let height : ℝ := intersection_x
  let area : ℝ := (1 / 2) * base * height
  area = 40 := by sorry

end NUMINAMATH_CALUDE_triangle_area_from_lines_l2345_234517


namespace NUMINAMATH_CALUDE_goldbach_138_max_diff_l2345_234582

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem goldbach_138_max_diff :
  ∃ (p q : ℕ), 
    is_prime p ∧ 
    is_prime q ∧ 
    p + q = 138 ∧ 
    p ≠ q ∧
    ∀ (r s : ℕ), is_prime r → is_prime s → r + s = 138 → r ≠ s → s - r ≤ 124 :=
sorry

end NUMINAMATH_CALUDE_goldbach_138_max_diff_l2345_234582


namespace NUMINAMATH_CALUDE_room_width_to_perimeter_ratio_l2345_234589

theorem room_width_to_perimeter_ratio :
  let length : ℝ := 22
  let width : ℝ := 15
  let perimeter : ℝ := 2 * (length + width)
  (width / perimeter) = (15 / 74) := by
sorry

end NUMINAMATH_CALUDE_room_width_to_perimeter_ratio_l2345_234589


namespace NUMINAMATH_CALUDE_sum_of_primes_below_1000_l2345_234552

-- Define a function that checks if a number is prime
def isPrime (n : Nat) : Prop := sorry

-- Define a function that counts the number of primes below a given number
def countPrimesBelow (n : Nat) : Nat := sorry

-- Define a function that sums all primes below a given number
def sumPrimesBelow (n : Nat) : Nat := sorry

-- Theorem statement
theorem sum_of_primes_below_1000 :
  (countPrimesBelow 1000 = 168) → (sumPrimesBelow 1000 = 76127) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_primes_below_1000_l2345_234552


namespace NUMINAMATH_CALUDE_product_mod_23_l2345_234504

theorem product_mod_23 : (3001 * 3002 * 3003 * 3004 * 3005) % 23 = 9 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_23_l2345_234504


namespace NUMINAMATH_CALUDE_line_through_point_with_specific_intercept_ratio_l2345_234530

/-- A line passing through the point (-5,2) with an x-intercept twice its y-intercept 
    has the equation 2x + 5y = 0 or x + 2y + 1 = 0 -/
theorem line_through_point_with_specific_intercept_ratio :
  ∀ (a b c : ℝ),
    (a ≠ 0 ∨ b ≠ 0) →
    (a * (-5) + b * 2 + c = 0) →
    (∃ t : ℝ, a * (-2*t) + c = 0 ∧ b * t + c = 0) →
    ((∃ k : ℝ, a = 2*k ∧ b = 5*k ∧ c = 0) ∨ (∃ k : ℝ, a = k ∧ b = 2*k ∧ c = -k)) :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_with_specific_intercept_ratio_l2345_234530


namespace NUMINAMATH_CALUDE_sqrt_7_irrational_l2345_234555

theorem sqrt_7_irrational : ¬ ∃ (p q : ℕ), p > 0 ∧ q > 0 ∧ (p : ℚ) ^ 2 / (q : ℚ) ^ 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_7_irrational_l2345_234555
