import Mathlib

namespace NUMINAMATH_CALUDE_speedster_convertibles_l197_19732

theorem speedster_convertibles (total : ℕ) 
  (h1 : 2 * total = 3 * (total - 60))  -- 2/3 of total are Speedsters, 60 are not
  (h2 : 5 * (total - 60) = 3 * total)  -- Restating h1 in a different form
  : (4 * (total - 60)) / 5 = 96 := by  -- 4/5 of Speedsters are convertibles
  sorry

#check speedster_convertibles

end NUMINAMATH_CALUDE_speedster_convertibles_l197_19732


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l197_19761

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_a1 : a 1 = -2)
  (h_a5 : a 5 = -8) :
  a 3 = -4 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l197_19761


namespace NUMINAMATH_CALUDE_bankers_discount_example_l197_19759

/-- Calculates the banker's discount given the face value and true discount of a bill. -/
def bankers_discount (face_value : ℚ) (true_discount : ℚ) : ℚ :=
  let present_value := face_value - true_discount
  (true_discount / present_value) * face_value

/-- Theorem stating that for a bill with face value 2660 and true discount 360,
    the banker's discount is approximately 416.35. -/
theorem bankers_discount_example :
  ∃ ε > 0, |bankers_discount 2660 360 - 416.35| < ε :=
by
  sorry

#eval bankers_discount 2660 360

end NUMINAMATH_CALUDE_bankers_discount_example_l197_19759


namespace NUMINAMATH_CALUDE_equation_solution_l197_19771

theorem equation_solution : ∃ x : ℕ, 5 + x = 10 + 20 ∧ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l197_19771


namespace NUMINAMATH_CALUDE_playground_area_is_297_l197_19777

/-- Calculates the area of a rectangular playground given the specified conditions --/
def playground_area (total_posts : ℕ) (post_spacing : ℕ) : ℕ :=
  let shorter_side_posts := 4  -- Including corners
  let longer_side_posts := 3 * shorter_side_posts
  let shorter_side_length := post_spacing * (shorter_side_posts - 1)
  let longer_side_length := post_spacing * (longer_side_posts - 1)
  shorter_side_length * longer_side_length

/-- Theorem stating that the area of the playground under given conditions is 297 square yards --/
theorem playground_area_is_297 :
  playground_area 24 3 = 297 := by
  sorry

end NUMINAMATH_CALUDE_playground_area_is_297_l197_19777


namespace NUMINAMATH_CALUDE_smallest_divisor_cube_sum_l197_19765

theorem smallest_divisor_cube_sum (n : ℕ) : n ≥ 2 →
  (∃ m : ℕ, m > 0 ∧ m ∣ n ∧
    (∃ d : ℕ, d > 1 ∧ d ∣ n ∧
      (∀ k : ℕ, k > 1 ∧ k ∣ n → k ≥ d) ∧
      n = d^3 + m^3)) →
  n = 16 ∨ n = 72 ∨ n = 520 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_cube_sum_l197_19765


namespace NUMINAMATH_CALUDE_decimal_fraction_equality_l197_19748

theorem decimal_fraction_equality (b : ℕ) : 
  b > 0 ∧ (5 * b + 22 : ℚ) / (7 * b + 15) = 87 / 100 → b = 8 := by
  sorry

end NUMINAMATH_CALUDE_decimal_fraction_equality_l197_19748


namespace NUMINAMATH_CALUDE_largest_possible_a_l197_19728

theorem largest_possible_a (a b c d : ℕ+) 
  (h1 : a < 3 * b)
  (h2 : b < 4 * c)
  (h3 : c < 5 * d)
  (h4 : d < 150) :
  a ≤ 8924 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 8924 ∧
    a' < 3 * b' ∧
    b' < 4 * c' ∧
    c' < 5 * d' ∧
    d' < 150 :=
by sorry

end NUMINAMATH_CALUDE_largest_possible_a_l197_19728


namespace NUMINAMATH_CALUDE_log_inequality_solution_l197_19793

theorem log_inequality_solution (x : ℝ) : 
  (4 * (Real.log (Real.cos (2 * x)) / Real.log 16) + 
   2 * (Real.log (Real.sin x) / Real.log 4) + 
   Real.log (Real.cos x) / Real.log 2 + 3 < 0) ↔ 
  (0 < x ∧ x < Real.pi / 24) ∨ (5 * Real.pi / 24 < x ∧ x < Real.pi / 4) :=
sorry

end NUMINAMATH_CALUDE_log_inequality_solution_l197_19793


namespace NUMINAMATH_CALUDE_range_of_h_l197_19750

noncomputable def h (t : ℝ) : ℝ := (t^2 + 5/4 * t) / (t^2 + 2)

theorem range_of_h :
  Set.range h = Set.Icc 0 (128/103) := by sorry

end NUMINAMATH_CALUDE_range_of_h_l197_19750


namespace NUMINAMATH_CALUDE_student_count_equation_l197_19792

/-- Represents the number of pens per box for the first type of pen -/
def pens_per_box_1 : ℕ := 8

/-- Represents the number of pens per box for the second type of pen -/
def pens_per_box_2 : ℕ := 12

/-- Represents the number of students without pens if x boxes of type 1 are bought -/
def students_without_pens : ℕ := 3

/-- Represents the number of fewer boxes that can be bought of type 2 -/
def fewer_boxes_type_2 : ℕ := 2

/-- Represents the number of pens left in the last box of type 2 -/
def pens_left_type_2 : ℕ := 1

theorem student_count_equation (x : ℕ) : 
  pens_per_box_1 * x + students_without_pens = 
  pens_per_box_2 * (x - fewer_boxes_type_2) - pens_left_type_2 := by
  sorry

end NUMINAMATH_CALUDE_student_count_equation_l197_19792


namespace NUMINAMATH_CALUDE_rectangular_field_distance_l197_19769

/-- The distance run around a rectangular field -/
def distance_run (length width : ℕ) (laps : ℕ) : ℕ :=
  2 * (length + width) * laps

/-- Theorem: Running 3 laps around a 75m by 15m rectangular field results in a total distance of 540m -/
theorem rectangular_field_distance :
  distance_run 75 15 3 = 540 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_distance_l197_19769


namespace NUMINAMATH_CALUDE_jack_socks_purchase_l197_19798

/-- The number of pairs of socks Jack needs to buy -/
def num_socks : ℕ := 2

/-- The cost of each pair of socks in dollars -/
def sock_cost : ℚ := 9.5

/-- The cost of the shoes in dollars -/
def shoe_cost : ℕ := 92

/-- The total amount Jack needs in dollars -/
def total_amount : ℕ := 111

theorem jack_socks_purchase :
  sock_cost * num_socks + shoe_cost = total_amount :=
by sorry

end NUMINAMATH_CALUDE_jack_socks_purchase_l197_19798


namespace NUMINAMATH_CALUDE_bathing_suit_combinations_total_combinations_l197_19701

theorem bathing_suit_combinations : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ
  | men_styles, men_sizes, men_colors, women_styles, women_sizes, women_colors =>
    (men_styles * men_sizes * men_colors) + (women_styles * women_sizes * women_colors)

theorem total_combinations (men_styles men_sizes men_colors women_styles women_sizes women_colors : ℕ) :
  men_styles = 5 →
  men_sizes = 3 →
  men_colors = 4 →
  women_styles = 4 →
  women_sizes = 4 →
  women_colors = 5 →
  bathing_suit_combinations men_styles men_sizes men_colors women_styles women_sizes women_colors = 140 :=
by
  sorry

end NUMINAMATH_CALUDE_bathing_suit_combinations_total_combinations_l197_19701


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l197_19788

theorem regular_polygon_sides (central_angle : ℝ) : 
  central_angle = 20 → (360 : ℝ) / central_angle = 18 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l197_19788


namespace NUMINAMATH_CALUDE_k_values_l197_19764

theorem k_values (p q r s k : ℂ) 
  (h_nonzero : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0)
  (h_eq1 : p * k^3 + q * k^2 + r * k + s = 0)
  (h_eq2 : q * k^3 + r * k^2 + s * k + p = 0)
  (h_pqrs : p * q = r * s) :
  k = 1 ∨ k = -1 ∨ k = Complex.I ∨ k = -Complex.I :=
sorry

end NUMINAMATH_CALUDE_k_values_l197_19764


namespace NUMINAMATH_CALUDE_different_monotonicity_implies_inequality_l197_19766

/-- Given a > 1, a ≠ 2, and (a-1)^x and (1/a)^x have different monotonicities,
    prove that (a-1)^(1/3) > (1/a)^3 -/
theorem different_monotonicity_implies_inequality (a : ℝ) 
  (h1 : a > 1) 
  (h2 : a ≠ 2) 
  (h3 : ∀ x y : ℝ, (∃ ε > 0, ∀ δ ∈ Set.Ioo (x - ε) (x + ε), 
    ((a - 1) ^ δ - (a - 1) ^ x) * ((1 / a) ^ δ - (1 / a) ^ x) < 0)) :
  (a - 1) ^ (1 / 3) > (1 / a) ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_different_monotonicity_implies_inequality_l197_19766


namespace NUMINAMATH_CALUDE_fourth_number_proof_l197_19787

theorem fourth_number_proof (x : ℝ) : 
  (3.6 * 0.48 * 2.50) / (x * 0.09 * 0.5) = 800.0000000000001 → x = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_fourth_number_proof_l197_19787


namespace NUMINAMATH_CALUDE_smallest_fraction_given_inequalities_l197_19711

theorem smallest_fraction_given_inequalities :
  ∀ r s : ℤ, 3 * r ≥ 2 * s - 3 → 4 * s ≥ r + 12 → 
  (∃ r' s' : ℤ, r' * s = r * s' ∧ s' > 0 ∧ r' * 2 = s') →
  ∀ r' s' : ℤ, r' * s = r * s' ∧ s' > 0 → r' * 2 ≤ s' :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_given_inequalities_l197_19711


namespace NUMINAMATH_CALUDE_gym_signup_fee_l197_19767

theorem gym_signup_fee 
  (cheap_monthly : ℕ)
  (expensive_monthly : ℕ)
  (expensive_signup : ℕ)
  (total_cost : ℕ)
  (h1 : cheap_monthly = 10)
  (h2 : expensive_monthly = 3 * cheap_monthly)
  (h3 : expensive_signup = 4 * expensive_monthly)
  (h4 : total_cost = 650)
  (h5 : total_cost = 12 * cheap_monthly + 12 * expensive_monthly + expensive_signup + cheap_signup) :
  cheap_signup = 50 := by
  sorry

end NUMINAMATH_CALUDE_gym_signup_fee_l197_19767


namespace NUMINAMATH_CALUDE_balloon_problem_l197_19716

theorem balloon_problem (total_people : ℕ) (total_balloons : ℕ) 
  (x₁ x₂ x₃ x₄ : ℕ) 
  (h1 : total_people = 101)
  (h2 : total_balloons = 212)
  (h3 : x₁ + x₂ + x₃ + x₄ = total_people)
  (h4 : x₁ + 2*x₂ + 3*x₃ + 4*x₄ = total_balloons)
  (h5 : x₄ = x₂ + 13) :
  x₁ = 52 := by
  sorry

end NUMINAMATH_CALUDE_balloon_problem_l197_19716


namespace NUMINAMATH_CALUDE_inequality_solution_l197_19738

open Real

theorem inequality_solution (x : ℝ) : 
  (2 * x + 3) / (x + 5) > (5 * x + 7) / (3 * x + 14) ↔ 
  (x > -103.86 ∧ x < -14/3) ∨ (x > -5 ∧ x < -0.14) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l197_19738


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l197_19775

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 4/7
  let a₂ : ℚ := 20/21
  let a₃ : ℚ := 100/63
  let r : ℚ := a₂ / a₁
  (∀ n : ℕ, n ≥ 1 → a₁ * r^(n-1) = (4/7) * (5/3)^(n-1)) →
  r = 5/3 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l197_19775


namespace NUMINAMATH_CALUDE_evaluate_expression_l197_19762

theorem evaluate_expression : 16^3 + 3*(16^2) + 3*16 + 1 = 4913 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l197_19762


namespace NUMINAMATH_CALUDE_greatest_integer_mike_l197_19718

theorem greatest_integer_mike (n : ℕ) : 
  (∃ k l : ℤ, n = 9 * k - 1 ∧ n = 10 * l - 4) →
  n < 150 →
  (∀ m : ℕ, (∃ k l : ℤ, m = 9 * k - 1 ∧ m = 10 * l - 4) → m < 150 → m ≤ n) →
  n = 86 := by
sorry

end NUMINAMATH_CALUDE_greatest_integer_mike_l197_19718


namespace NUMINAMATH_CALUDE_nikolai_silver_decrease_l197_19753

/-- Represents the number of each type of coin --/
structure CoinCount where
  gold : ℕ
  silver : ℕ
  copper : ℕ

/-- Represents a transaction at the exchange point --/
inductive Transaction
  | Type1 : Transaction  -- 2 gold for 3 silver and 1 copper
  | Type2 : Transaction  -- 5 silver for 3 gold and 1 copper

/-- Applies a single transaction to a CoinCount --/
def applyTransaction (t : Transaction) (c : CoinCount) : CoinCount :=
  match t with
  | Transaction.Type1 => CoinCount.mk (c.gold - 2) (c.silver + 3) (c.copper + 1)
  | Transaction.Type2 => CoinCount.mk (c.gold + 3) (c.silver - 5) (c.copper + 1)

/-- Applies a list of transactions to an initial CoinCount --/
def applyTransactions (ts : List Transaction) (initial : CoinCount) : CoinCount :=
  ts.foldl (fun acc t => applyTransaction t acc) initial

theorem nikolai_silver_decrease (initialSilver : ℕ) :
  ∃ (ts : List Transaction),
    let final := applyTransactions ts (CoinCount.mk 0 initialSilver 0)
    final.gold = 0 ∧
    final.copper = 50 ∧
    initialSilver - final.silver = 10 := by
  sorry

end NUMINAMATH_CALUDE_nikolai_silver_decrease_l197_19753


namespace NUMINAMATH_CALUDE_triangle_is_obtuse_l197_19770

theorem triangle_is_obtuse (A : Real) (h1 : 0 < A ∧ A < π) 
  (h2 : Real.sin A + Real.cos A = 7 / 12) : π / 2 < A ∧ A < π := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_obtuse_l197_19770


namespace NUMINAMATH_CALUDE_discount_gain_percent_l197_19708

theorem discount_gain_percent (marked_price : ℝ) (cost_price : ℝ) (discount_rate : ℝ) :
  cost_price = 0.64 * marked_price →
  discount_rate = 0.12 →
  let selling_price := marked_price * (1 - discount_rate)
  let gain_percent := ((selling_price - cost_price) / cost_price) * 100
  gain_percent = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_discount_gain_percent_l197_19708


namespace NUMINAMATH_CALUDE_sum_first_150_remainder_l197_19760

theorem sum_first_150_remainder (n : Nat) (h : n = 150) :
  (n * (n + 1) / 2) % 5000 = 1275 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_150_remainder_l197_19760


namespace NUMINAMATH_CALUDE_common_factor_of_polynomials_l197_19778

theorem common_factor_of_polynomials (a b : ℝ) :
  ∃ (k₁ k₂ : ℝ), (4 * a^2 - 2 * a * b = (2 * a - b) * k₁) ∧
                 (4 * a^2 - b^2 = (2 * a - b) * k₂) := by
  sorry

end NUMINAMATH_CALUDE_common_factor_of_polynomials_l197_19778


namespace NUMINAMATH_CALUDE_stratified_sampling_pine_saplings_l197_19705

theorem stratified_sampling_pine_saplings 
  (total_saplings : ℕ) 
  (pine_saplings : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_saplings = 30000) 
  (h2 : pine_saplings = 4000) 
  (h3 : sample_size = 150) :
  (pine_saplings : ℚ) / total_saplings * sample_size = 20 := by
sorry


end NUMINAMATH_CALUDE_stratified_sampling_pine_saplings_l197_19705


namespace NUMINAMATH_CALUDE_difference_of_squares_123_23_l197_19709

theorem difference_of_squares_123_23 : 123^2 - 23^2 = 14600 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_123_23_l197_19709


namespace NUMINAMATH_CALUDE_f_properties_l197_19744

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := (Real.log x + a) / x

def g : ℝ → ℝ := λ x => 1

theorem f_properties (a : ℝ) :
  (∀ x > 0, f a x ≤ Real.exp (a - 1)) ∧
  (∃ x > 0, x ≤ Real.exp 2 ∧ f a x = g x) ↔ a ≥ 1 := by
  sorry

end

end NUMINAMATH_CALUDE_f_properties_l197_19744


namespace NUMINAMATH_CALUDE_tens_digit_of_6_to_18_l197_19749

/-- The tens digit of a natural number -/
def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

/-- The theorem stating that the tens digit of 6^18 is 1 -/
theorem tens_digit_of_6_to_18 : tens_digit (6^18) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_6_to_18_l197_19749


namespace NUMINAMATH_CALUDE_christine_savings_l197_19723

/-- Calculates the amount saved by a salesperson given their commission rate, total sales, and personal needs allocation percentage. -/
def amount_saved (commission_rate : ℚ) (total_sales : ℚ) (personal_needs_percent : ℚ) : ℚ :=
  let total_commission := commission_rate * total_sales
  let personal_needs := personal_needs_percent * total_commission
  total_commission - personal_needs

/-- Proves that given the specific conditions, the amount saved is $1152. -/
theorem christine_savings : 
  amount_saved (12/100) 24000 (60/100) = 1152 := by
  sorry

end NUMINAMATH_CALUDE_christine_savings_l197_19723


namespace NUMINAMATH_CALUDE_inequality_equivalence_l197_19724

theorem inequality_equivalence (x : ℝ) : 
  x * Real.log (x^2 + x + 1) / Real.log (1/10) > 0 ↔ x < -1 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l197_19724


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l197_19707

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + 3*b - 1 = 0) :
  (2/a + 3/b) ≥ 25 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2*a₀ + 3*b₀ - 1 = 0 ∧ 2/a₀ + 3/b₀ = 25 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l197_19707


namespace NUMINAMATH_CALUDE_jerry_initial_figures_l197_19713

/-- The number of books on Jerry's shelf -/
def num_books : ℕ := 9

/-- The number of action figures added later -/
def added_figures : ℕ := 7

/-- The difference between action figures and books after adding -/
def difference : ℕ := 3

/-- The initial number of action figures on Jerry's shelf -/
def initial_figures : ℕ := 5

theorem jerry_initial_figures :
  initial_figures + added_figures = num_books + difference := by sorry

end NUMINAMATH_CALUDE_jerry_initial_figures_l197_19713


namespace NUMINAMATH_CALUDE_initial_markup_percentage_l197_19768

theorem initial_markup_percentage (C : ℝ) (h : C > 0) : 
  ∃ M : ℝ, 
    M ≥ 0 ∧ 
    C * (1 + M) * 1.25 * 0.93 = C * (1 + 0.395) ∧ 
    M = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_initial_markup_percentage_l197_19768


namespace NUMINAMATH_CALUDE_measure_8_and_5_cm_l197_19712

-- Define the marks on the ruler
def ruler_marks : List ℕ := [0, 7, 11]

-- Define a function to check if a length can be measured
def can_measure (length : ℕ) : Prop :=
  ∃ (a b c : ℤ), a * ruler_marks[1] + b * ruler_marks[2] + c * (ruler_marks[2] - ruler_marks[1]) = length

-- Theorem statement
theorem measure_8_and_5_cm :
  can_measure 8 ∧ can_measure 5 :=
by sorry

end NUMINAMATH_CALUDE_measure_8_and_5_cm_l197_19712


namespace NUMINAMATH_CALUDE_one_fourth_in_one_eighth_l197_19776

theorem one_fourth_in_one_eighth :
  (1 / 8 : ℚ) / (1 / 4 : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_in_one_eighth_l197_19776


namespace NUMINAMATH_CALUDE_rationalize_denominator_l197_19785

theorem rationalize_denominator : 
  (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 3 + Real.sqrt 5) = (Real.sqrt 15 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l197_19785


namespace NUMINAMATH_CALUDE_equation_has_real_solution_l197_19752

theorem equation_has_real_solution (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  ∃ x : ℝ, (a * b^x)^(x + 1) = c := by
  sorry

end NUMINAMATH_CALUDE_equation_has_real_solution_l197_19752


namespace NUMINAMATH_CALUDE_g_25_l197_19745

-- Define the function g
variable (g : ℝ → ℝ)

-- State the conditions
axiom g_property : ∀ (x y : ℝ), x > 0 → y > 0 → g (x / y) = y * g x
axiom g_50 : g 50 = 10

-- State the theorem to be proved
theorem g_25 : g 25 = 20 := by sorry

end NUMINAMATH_CALUDE_g_25_l197_19745


namespace NUMINAMATH_CALUDE_increasing_equivalent_l197_19739

/-- A function is increasing on an interval if its graph always rises when viewed from left to right. -/
def IncreasingOnInterval (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ I → x₂ ∈ I → x₁ < x₂ → f x₁ < f x₂

theorem increasing_equivalent {f : ℝ → ℝ} {I : Set ℝ} :
  IncreasingOnInterval f I ↔
  (∀ x₁ x₂, x₁ ∈ I → x₂ ∈ I → x₁ < x₂ → f x₁ < f x₂) :=
by sorry

end NUMINAMATH_CALUDE_increasing_equivalent_l197_19739


namespace NUMINAMATH_CALUDE_shopkeeper_profit_margin_l197_19740

theorem shopkeeper_profit_margin
  (C : ℝ) -- Current cost
  (S : ℝ) -- Selling price
  (y : ℝ) -- Original profit margin percentage
  (h1 : S = C * (1 + 0.01 * y)) -- Current profit margin equation
  (h2 : S = 0.9 * C * (1 + 0.01 * (y + 15))) -- New profit margin equation
  : y = 35 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_margin_l197_19740


namespace NUMINAMATH_CALUDE_day2_to_day1_rain_ratio_l197_19791

/-- Represents the rainfall data and conditions for a 4-day storm --/
structure RainfallData where
  capacity : ℝ  -- Capacity in inches
  drainRate : ℝ  -- Drain rate in inches per day
  day1Rain : ℝ  -- Rainfall on day 1 in inches
  day3Increase : ℝ  -- Percentage increase of day 3 rain compared to day 2
  day4Rain : ℝ  -- Rainfall on day 4 in inches

/-- Theorem stating the ratio of day 2 rain to day 1 rain --/
theorem day2_to_day1_rain_ratio (data : RainfallData) 
  (h1 : data.capacity = 72) -- 6 feet = 72 inches
  (h2 : data.drainRate = 3)
  (h3 : data.day1Rain = 10)
  (h4 : data.day3Increase = 1.5) -- 50% more
  (h5 : data.day4Rain = 21) :
  ∃ (x : ℝ), x = 2 ∧ 
    data.day1Rain + x * data.day1Rain + data.day3Increase * x * data.day1Rain + data.day4Rain = 
    data.capacity + 3 * data.drainRate := by
  sorry

#check day2_to_day1_rain_ratio

end NUMINAMATH_CALUDE_day2_to_day1_rain_ratio_l197_19791


namespace NUMINAMATH_CALUDE_base_sum_theorem_l197_19773

theorem base_sum_theorem : ∃! (R_A R_B : ℕ), 
  (R_A > 0 ∧ R_B > 0) ∧
  ((4 * R_A + 5) * (R_B^2 - 1) = (3 * R_B + 6) * (R_A^2 - 1)) ∧
  ((5 * R_A + 4) * (R_B^2 - 1) = (6 * R_B + 3) * (R_A^2 - 1)) ∧
  (R_A + R_B = 19) := by
sorry

end NUMINAMATH_CALUDE_base_sum_theorem_l197_19773


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l197_19747

theorem sum_of_squares_of_roots (p q r : ℝ) : 
  (3 * p^3 - 2 * p^2 + 5 * p + 15 = 0) →
  (3 * q^3 - 2 * q^2 + 5 * q + 15 = 0) →
  (3 * r^3 - 2 * r^2 + 5 * r + 15 = 0) →
  p^2 + q^2 + r^2 = -26/9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l197_19747


namespace NUMINAMATH_CALUDE_salary_solution_l197_19743

def salary_problem (s : ℕ) : Prop :=
  s - s / 3 - s / 4 - s / 5 = 1760

theorem salary_solution : ∃ (s : ℕ), salary_problem s ∧ s = 812 := by
  sorry

end NUMINAMATH_CALUDE_salary_solution_l197_19743


namespace NUMINAMATH_CALUDE_teacher_count_l197_19780

theorem teacher_count (total : ℕ) (sample_size : ℕ) (students_in_sample : ℕ) :
  total = 3000 →
  sample_size = 150 →
  students_in_sample = 140 →
  (total - (total * students_in_sample / sample_size) : ℕ) = 200 := by
  sorry

end NUMINAMATH_CALUDE_teacher_count_l197_19780


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l197_19746

theorem diophantine_equation_solution (t : ℤ) : 
  ∃ (x y : ℤ), x^4 + 2*x^3 + 8*x - 35*y + 9 = 0 ∧
  (x = 35*t + 6 ∨ x = 35*t - 4 ∨ x = 35*t - 9 ∨ 
   x = 35*t - 16 ∨ x = 35*t - 1 ∨ x = 35*t - 11) ∧
  y = (x^4 + 2*x^3 + 8*x + 9) / 35 :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l197_19746


namespace NUMINAMATH_CALUDE_nine_integer_chords_l197_19774

/-- A circle with a point P inside it -/
structure CircleWithPoint where
  radius : ℝ
  distanceFromCenter : ℝ

/-- The number of integer-length chords through P -/
def integerChordCount (c : CircleWithPoint) : ℕ :=
  sorry

theorem nine_integer_chords 
  (c : CircleWithPoint) 
  (h1 : c.radius = 20) 
  (h2 : c.distanceFromCenter = 12) : 
  integerChordCount c = 9 := by sorry

end NUMINAMATH_CALUDE_nine_integer_chords_l197_19774


namespace NUMINAMATH_CALUDE_parabola_equation_l197_19727

/-- The equation of a parabola with given focus and directrix -/
theorem parabola_equation (x y : ℝ) : 
  let focus : ℝ × ℝ := (4, 4)
  let directrix : ℝ → ℝ → ℝ := λ x y => 4*x + 8*y - 32
  let parabola : ℝ → ℝ → ℝ := λ x y => 64*x^2 - 128*x*y + 64*y^2 - 512*x - 512*y + 1024
  (∀ (p : ℝ × ℝ), p ∈ {p | parabola p.1 p.2 = 0} ↔ 
    (p.1 - focus.1)^2 + (p.2 - focus.2)^2 = (directrix p.1 p.2 / (4 * Real.sqrt 5))^2) :=
by sorry

#check parabola_equation

end NUMINAMATH_CALUDE_parabola_equation_l197_19727


namespace NUMINAMATH_CALUDE_sandwiches_per_person_l197_19733

theorem sandwiches_per_person (people : ℝ) (total_sandwiches : ℕ) 
  (h1 : people = 219) 
  (h2 : total_sandwiches = 657) : 
  (total_sandwiches : ℝ) / people = 3 := by
  sorry

end NUMINAMATH_CALUDE_sandwiches_per_person_l197_19733


namespace NUMINAMATH_CALUDE_problem_solution_l197_19797

def U : Set ℤ := {x | 0 ≤ x ∧ x ≤ 6}

theorem problem_solution (A B : Set ℤ) 
  (h1 : U = A ∪ B) 
  (h2 : A ∩ (U \ B) = {1, 3, 5}) : 
  B = {0, 2, 4, 6} := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l197_19797


namespace NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_under_1000_l197_19729

theorem greatest_multiple_of_5_and_6_under_1000 : 
  ∀ n : ℕ, n < 1000 → n % 5 = 0 → n % 6 = 0 → n ≤ 990 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_under_1000_l197_19729


namespace NUMINAMATH_CALUDE_juliet_supporter_in_capulet_probability_l197_19720

-- Define the population distribution
def montague_pop : ℚ := 5/8
def capulet_pop : ℚ := 3/16
def verona_pop : ℚ := 1/8
def mercutio_pop : ℚ := 1 - (montague_pop + capulet_pop + verona_pop)

-- Define the support rates
def montague_romeo_rate : ℚ := 4/5
def capulet_juliet_rate : ℚ := 7/10
def verona_romeo_rate : ℚ := 13/20
def mercutio_juliet_rate : ℚ := 11/20

-- Define the total Juliet supporters
def total_juliet_supporters : ℚ := capulet_pop * capulet_juliet_rate + mercutio_pop * mercutio_juliet_rate

-- Define the probability
def prob_juliet_in_capulet : ℚ := (capulet_pop * capulet_juliet_rate) / total_juliet_supporters

-- Theorem statement
theorem juliet_supporter_in_capulet_probability :
  ∃ (ε : ℚ), abs (prob_juliet_in_capulet - 66/100) < ε ∧ ε < 1/100 :=
sorry

end NUMINAMATH_CALUDE_juliet_supporter_in_capulet_probability_l197_19720


namespace NUMINAMATH_CALUDE_derivative_f_at_pi_l197_19710

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / (x^2)

theorem derivative_f_at_pi : 
  deriv f π = -(1 / π^2) := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_pi_l197_19710


namespace NUMINAMATH_CALUDE_minimum_value_problem_minimum_value_achievable_l197_19758

theorem minimum_value_problem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  4 * x^3 + 8 * y^3 + 18 * z^3 + 1 / (6 * x * y * z) ≥ 4 :=
by sorry

theorem minimum_value_achievable :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  4 * x^3 + 8 * y^3 + 18 * z^3 + 1 / (6 * x * y * z) = 4 :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_problem_minimum_value_achievable_l197_19758


namespace NUMINAMATH_CALUDE_max_value_problem_l197_19717

theorem max_value_problem (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
  (h4 : x + y + z = 3) 
  (h5 : x ≥ y) (h6 : y ≥ z) : 
  (x^2 - x*y + y^2) * (x^2 - x*z + z^2) * (y^2 - y*z + z^2) ≤ 2916/729 :=
sorry

end NUMINAMATH_CALUDE_max_value_problem_l197_19717


namespace NUMINAMATH_CALUDE_smallest_k_is_three_l197_19756

/-- A coloring of positive integers with k colors -/
def Coloring (k : ℕ) := ℕ+ → Fin k

/-- Property (i): For all positive integers m, n of the same color, f(m+n) = f(m) + f(n) -/
def PropertyOne (f : ℕ+ → ℕ+) (c : Coloring k) :=
  ∀ m n : ℕ+, c m = c n → f (m + n) = f m + f n

/-- Property (ii): There exist positive integers m, n such that f(m+n) ≠ f(m) + f(n) -/
def PropertyTwo (f : ℕ+ → ℕ+) :=
  ∃ m n : ℕ+, f (m + n) ≠ f m + f n

/-- The main theorem statement -/
theorem smallest_k_is_three :
  (∃ k : ℕ+, ∃ c : Coloring k, ∃ f : ℕ+ → ℕ+, PropertyOne f c ∧ PropertyTwo f) ∧
  (∀ k : ℕ+, k < 3 → ¬∃ c : Coloring k, ∃ f : ℕ+ → ℕ+, PropertyOne f c ∧ PropertyTwo f) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_is_three_l197_19756


namespace NUMINAMATH_CALUDE_shirt_and_sweater_cost_l197_19706

theorem shirt_and_sweater_cost (shirt_price sweater_price total_cost : ℝ) : 
  shirt_price = 36.46 →
  sweater_price = shirt_price + 7.43 →
  total_cost = shirt_price + sweater_price →
  total_cost = 80.35 := by
sorry

end NUMINAMATH_CALUDE_shirt_and_sweater_cost_l197_19706


namespace NUMINAMATH_CALUDE_weight_of_K2Cr2O7_l197_19789

/-- The atomic weight of potassium in g/mol -/
def atomic_weight_K : ℝ := 39.10

/-- The atomic weight of chromium in g/mol -/
def atomic_weight_Cr : ℝ := 52.00

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of potassium atoms in K2Cr2O7 -/
def K_count : ℕ := 2

/-- The number of chromium atoms in K2Cr2O7 -/
def Cr_count : ℕ := 2

/-- The number of oxygen atoms in K2Cr2O7 -/
def O_count : ℕ := 7

/-- The number of moles of K2Cr2O7 -/
def moles : ℝ := 4

/-- The molecular weight of K2Cr2O7 in g/mol -/
def molecular_weight_K2Cr2O7 : ℝ := 
  K_count * atomic_weight_K + Cr_count * atomic_weight_Cr + O_count * atomic_weight_O

/-- The total weight of 4 moles of K2Cr2O7 in grams -/
theorem weight_of_K2Cr2O7 : moles * molecular_weight_K2Cr2O7 = 1176.80 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_K2Cr2O7_l197_19789


namespace NUMINAMATH_CALUDE_acidic_solution_concentration_l197_19734

/-- Proves that the initial volume of a 40% acidic solution is 27 liters
    when it becomes 60% acidic after removing 9 liters of water. -/
theorem acidic_solution_concentration (initial_volume : ℝ) : 
  initial_volume > 0 →
  (0.4 * initial_volume) / (initial_volume - 9) = 0.6 →
  initial_volume = 27 := by
  sorry

end NUMINAMATH_CALUDE_acidic_solution_concentration_l197_19734


namespace NUMINAMATH_CALUDE_problem_solution_l197_19799

theorem problem_solution (x y : ℝ) 
  (h1 : x + Real.cos y = 1010)
  (h2 : x + 1010 * Real.sin y = 1009)
  (h3 : π / 4 ≤ y ∧ y ≤ π / 2) :
  x + y = 1010 + π / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l197_19799


namespace NUMINAMATH_CALUDE_quadratic_factorization_l197_19719

theorem quadratic_factorization (c d : ℕ) (hc : c > d) :
  (∀ x, x^2 - 20*x + 96 = (x - c) * (x - d)) →
  4*d - c = 20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l197_19719


namespace NUMINAMATH_CALUDE_minimize_quadratic_expression_l197_19702

theorem minimize_quadratic_expression (b : ℝ) :
  let f : ℝ → ℝ := λ x => (1/3) * x^2 + 7*x - 6
  ∀ x, f b ≤ f x ↔ b = -21/2 :=
by sorry

end NUMINAMATH_CALUDE_minimize_quadratic_expression_l197_19702


namespace NUMINAMATH_CALUDE_profit_percent_calculation_l197_19735

theorem profit_percent_calculation (selling_price cost_price : ℝ) 
  (h : cost_price = 0.4 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = 150 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_calculation_l197_19735


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_5985_l197_19783

theorem largest_prime_factor_of_5985 : ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 5985 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 5985 → q ≤ p :=
sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_5985_l197_19783


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l197_19741

-- Define the necessary structures and functions
structure Point where
  x : ℝ
  y : ℝ

def distance (p q : Point) : ℝ := sorry

def is_hyperbola (trajectory : Set Point) (F₁ F₂ : Point) : Prop := sorry

def is_constant (f : Point → ℝ) : Prop := sorry

-- State the theorem
theorem necessary_but_not_sufficient 
  (M : Point) (F₁ F₂ : Point) (trajectory : Set Point) :
  (∀ M ∈ trajectory, is_hyperbola trajectory F₁ F₂) →
    is_constant (λ M => |distance M F₁ - distance M F₂|) ∧
  ∃ trajectory' : Set Point, 
    is_constant (λ M => |distance M F₁ - distance M F₂|) ∧
    ¬(is_hyperbola trajectory' F₁ F₂) :=
by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l197_19741


namespace NUMINAMATH_CALUDE_bitna_elementary_students_l197_19796

/-- The number of pencils purchased by Bitna Elementary School -/
def total_pencils : ℕ := 10395

/-- The number of pencils distributed to each student -/
def pencils_per_student : ℕ := 11

/-- The number of students in Bitna Elementary School -/
def number_of_students : ℕ := total_pencils / pencils_per_student

theorem bitna_elementary_students : number_of_students = 945 := by
  sorry

end NUMINAMATH_CALUDE_bitna_elementary_students_l197_19796


namespace NUMINAMATH_CALUDE_total_balloons_sam_dan_l197_19795

-- Define the initial quantities
def sam_initial : ℝ := 46.5
def fred_receive : ℝ := 10.2
def gaby_receive : ℝ := 3.3
def dan_balloons : ℝ := 16.4

-- Define Sam's remaining balloons after distribution
def sam_remaining : ℝ := sam_initial - fred_receive - gaby_receive

-- Theorem statement
theorem total_balloons_sam_dan : 
  sam_remaining + dan_balloons = 49.4 := by sorry

end NUMINAMATH_CALUDE_total_balloons_sam_dan_l197_19795


namespace NUMINAMATH_CALUDE_ironman_age_relation_l197_19742

/-- Represents the age relationship between superheroes -/
structure SuperheroAges where
  thor : ℝ
  captainAmerica : ℝ
  peterParker : ℝ
  ironman : ℝ

/-- The age relationships between the superheroes are valid -/
def validAgeRelationships (ages : SuperheroAges) : Prop :=
  ages.thor = 13 * ages.captainAmerica ∧
  ages.captainAmerica = 7 * ages.peterParker ∧
  ages.ironman = ages.peterParker + 32

/-- Theorem stating the relationship between Ironman's age and Thor's age -/
theorem ironman_age_relation (ages : SuperheroAges) 
  (h : validAgeRelationships ages) : 
  ages.ironman = ages.thor / 91 + 32 := by
  sorry

end NUMINAMATH_CALUDE_ironman_age_relation_l197_19742


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l197_19751

theorem fraction_equation_solution :
  ∃! x : ℚ, (x - 4) / (x + 3) = (x + 2) / (x - 1) ∧ x = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l197_19751


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l197_19726

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 11

-- Define the point of tangency
def P : ℝ × ℝ := (1, 12)

-- Theorem statement
theorem tangent_line_y_intercept :
  let slope := (3 : ℝ) -- Derivative of f at x = 1
  let tangent_line (x : ℝ) := slope * (x - P.1) + P.2
  (tangent_line 0) = 9 := by sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l197_19726


namespace NUMINAMATH_CALUDE_daisy_germination_rate_l197_19736

/-- Proves that the germination rate of daisy seeds is 60% given the problem conditions --/
theorem daisy_germination_rate :
  let daisy_seeds : ℕ := 25
  let sunflower_seeds : ℕ := 25
  let sunflower_germination_rate : ℚ := 80 / 100
  let flower_production_rate : ℚ := 80 / 100
  let total_flowering_plants : ℕ := 28
  ∃ (daisy_germination_rate : ℚ),
    daisy_germination_rate = 60 / 100 ∧
    (↑daisy_seeds * daisy_germination_rate * flower_production_rate +
     ↑sunflower_seeds * sunflower_germination_rate * flower_production_rate : ℚ) = total_flowering_plants :=
by sorry

end NUMINAMATH_CALUDE_daisy_germination_rate_l197_19736


namespace NUMINAMATH_CALUDE_inequality_proof_l197_19725

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  (a * b / Real.sqrt (c^2 + 3)) + (b * c / Real.sqrt (a^2 + 3)) + (c * a / Real.sqrt (b^2 + 3)) ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l197_19725


namespace NUMINAMATH_CALUDE_power_of_36_l197_19714

theorem power_of_36 : (36 : ℝ) ^ (5/2 : ℝ) = 7776 := by
  sorry

end NUMINAMATH_CALUDE_power_of_36_l197_19714


namespace NUMINAMATH_CALUDE_square_root_expression_l197_19763

theorem square_root_expression (m n : ℝ) : 
  Real.sqrt ((m - 2*n - 3) * (m - 2*n + 3) + 9) = 
    if m ≥ 2*n then m - 2*n else 2*n - m := by
  sorry

end NUMINAMATH_CALUDE_square_root_expression_l197_19763


namespace NUMINAMATH_CALUDE_polynomial_equality_l197_19784

/-- Given that 7x^5 + 4x^3 - 3x + p(x) = 2x^4 - 10x^3 + 5x - 2,
    prove that p(x) = -7x^5 + 2x^4 - 6x^3 + 2x - 2 -/
theorem polynomial_equality (x : ℝ) (p : ℝ → ℝ) 
  (h : ∀ x, 7 * x^5 + 4 * x^3 - 3 * x + p x = 2 * x^4 - 10 * x^3 + 5 * x - 2) : 
  p = fun x ↦ -7 * x^5 + 2 * x^4 - 6 * x^3 + 2 * x - 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l197_19784


namespace NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l197_19779

def is_arithmetic_sequence (a b c d : ℚ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def sum_is_26 (a b c d : ℚ) : Prop :=
  a + b + c + d = 26

def middle_product_is_40 (b c : ℚ) : Prop :=
  b * c = 40

theorem arithmetic_sequence_theorem (a b c d : ℚ) :
  is_arithmetic_sequence a b c d →
  sum_is_26 a b c d →
  middle_product_is_40 b c →
  ((a = 2 ∧ b = 5 ∧ c = 8 ∧ d = 11) ∨ (a = 11 ∧ b = 8 ∧ c = 5 ∧ d = 2)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l197_19779


namespace NUMINAMATH_CALUDE_final_value_calculation_l197_19731

theorem final_value_calculation : 
  let initial_number := 16
  let doubled := initial_number * 2
  let added_five := doubled + 5
  let final_value := added_five * 3
  final_value = 111 := by
sorry

end NUMINAMATH_CALUDE_final_value_calculation_l197_19731


namespace NUMINAMATH_CALUDE_p_sufficient_but_not_necessary_for_q_l197_19721

-- Define the conditions
def p (x : ℝ) : Prop := |x - 1| < 2
def q (x : ℝ) : Prop := x^2 - 5*x - 6 < 0

-- Define what it means for p to be sufficient but not necessary for q
def sufficient_but_not_necessary (p q : ℝ → Prop) : Prop :=
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x)

-- State the theorem
theorem p_sufficient_but_not_necessary_for_q :
  sufficient_but_not_necessary p q := by
  sorry

end NUMINAMATH_CALUDE_p_sufficient_but_not_necessary_for_q_l197_19721


namespace NUMINAMATH_CALUDE_correct_operation_l197_19772

theorem correct_operation (a : ℝ) : 3 * a - 2 * a = a := by sorry

end NUMINAMATH_CALUDE_correct_operation_l197_19772


namespace NUMINAMATH_CALUDE_min_complex_sum_value_l197_19755

theorem min_complex_sum_value (p q r : ℕ+) (ζ : ℂ) 
  (h_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r)
  (h_ζ_fourth : ζ^4 = 1)
  (h_ζ_neq_one : ζ ≠ 1) :
  ∃ (m : ℝ), m = Real.sqrt 7 ∧ 
    ∀ (p' q' r' : ℕ+) (h_distinct' : p' ≠ q' ∧ q' ≠ r' ∧ p' ≠ r'),
      Complex.abs (p' + q' * ζ + r' * ζ^3) ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_complex_sum_value_l197_19755


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l197_19790

theorem bowling_ball_weight :
  ∀ (bowling_ball_weight canoe_weight : ℝ),
    (10 * bowling_ball_weight = 4 * canoe_weight) →
    (canoe_weight = 35) →
    (bowling_ball_weight = 14) :=
by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l197_19790


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_l197_19715

theorem absolute_value_inequality_solution (x : ℝ) :
  (|x + 2| + |x - 2| < x + 7) ↔ (-7/3 < x ∧ x < 7) :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_l197_19715


namespace NUMINAMATH_CALUDE_inscribed_circle_triangle_sides_l197_19703

/-- A triangle with an inscribed circle of radius 3, where one side is divided into segments of 4 and 3 by the point of tangency. -/
structure InscribedCircleTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of the first segment of the divided side -/
  s₁ : ℝ
  /-- The length of the second segment of the divided side -/
  s₂ : ℝ
  /-- Condition that the radius is 3 -/
  h_r : r = 3
  /-- Condition that the first segment is 4 -/
  h_s₁ : s₁ = 4
  /-- Condition that the second segment is 3 -/
  h_s₂ : s₂ = 3

/-- The lengths of the sides of the triangle -/
def sideLengths (t : InscribedCircleTriangle) : Fin 3 → ℝ
| 0 => 24
| 1 => 25
| 2 => 7

theorem inscribed_circle_triangle_sides (t : InscribedCircleTriangle) :
  ∀ i, sideLengths t i = if i = 0 then 24 else if i = 1 then 25 else 7 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_triangle_sides_l197_19703


namespace NUMINAMATH_CALUDE_pairs_with_female_l197_19757

theorem pairs_with_female (total : Nat) (males : Nat) (females : Nat) : 
  total = males + females → males = 3 → females = 3 → 
  (Nat.choose total 2) - (Nat.choose males 2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_pairs_with_female_l197_19757


namespace NUMINAMATH_CALUDE_min_value_expression_l197_19786

theorem min_value_expression (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ (m : ℝ), (∀ c d : ℝ, c ≠ 0 → d ≠ 0 → c^2 + d^2 + 4/c^2 + 2*d/c ≥ m) ∧
  (∃ c d : ℝ, c ≠ 0 ∧ d ≠ 0 ∧ c^2 + d^2 + 4/c^2 + 2*d/c = m) ∧
  m = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l197_19786


namespace NUMINAMATH_CALUDE_parabola_directrix_l197_19737

/-- Given a parabola defined by x = -1/4 * y^2, its directrix is the line x = 1 -/
theorem parabola_directrix (x y : ℝ) : 
  (x = -(1/4) * y^2) → (∃ (k : ℝ), k = 1 ∧ k = x) := by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l197_19737


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_for_inequality_l197_19722

-- Define the function f
def f (x a : ℝ) := |3 * x + 3| + |x - a|

-- Theorem 1
theorem solution_set_when_a_is_2 :
  {x : ℝ | f x 2 > 4} = {x : ℝ | x > -1/2 ∨ x < -5/4} := by sorry

-- Theorem 2
theorem range_of_a_for_inequality :
  ∀ a : ℝ, (∀ x : ℝ, x > -1 → f x a > 3*x + 4) ↔ a ≤ -2 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_for_inequality_l197_19722


namespace NUMINAMATH_CALUDE_votes_for_both_policies_l197_19781

-- Define the total number of students
def total_students : ℕ := 185

-- Define the number of students voting for the first policy
def first_policy_votes : ℕ := 140

-- Define the number of students voting for the second policy
def second_policy_votes : ℕ := 110

-- Define the number of students voting against both policies
def against_both : ℕ := 22

-- Define the number of students abstaining from both policies
def abstained : ℕ := 15

-- Theorem stating that the number of students voting for both policies is 102
theorem votes_for_both_policies : 
  first_policy_votes + second_policy_votes - total_students + against_both + abstained = 102 :=
by sorry

end NUMINAMATH_CALUDE_votes_for_both_policies_l197_19781


namespace NUMINAMATH_CALUDE_fraction_equality_l197_19704

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x - 2 * y) / (3 * x + y) = 3) : 
  (5 * x - y) / (2 * x + 4 * y) = -3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l197_19704


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l197_19794

theorem function_inequality_implies_a_bound 
  (f g : ℝ → ℝ) 
  (h_f : ∀ x, f x = |x - a| + a) 
  (h_g : ∀ x, g x = 4 - x^2) 
  (h_exists : ∃ x, g x ≥ f x) : 
  a ≤ 17/8 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l197_19794


namespace NUMINAMATH_CALUDE_child_ticket_cost_l197_19782

theorem child_ticket_cost (num_adults num_children : ℕ) (adult_ticket_price total_bill : ℚ) :
  num_adults = 10 →
  num_children = 11 →
  adult_ticket_price = 8 →
  total_bill = 124 →
  (total_bill - num_adults * adult_ticket_price) / num_children = 4 :=
by sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l197_19782


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l197_19700

theorem largest_prime_factor_of_expression :
  (Nat.factors (18^3 + 15^4 - 10^5)).maximum? = some 98359 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l197_19700


namespace NUMINAMATH_CALUDE_fraction_difference_zero_l197_19754

theorem fraction_difference_zero (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) (h3 : x ≠ 3) :
  1 / ((x - 2) * (x - 3)) - 2 / ((x - 1) * (x - 3)) + 1 / ((x - 1) * (x - 2)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_zero_l197_19754


namespace NUMINAMATH_CALUDE_quadratic_root_range_l197_19730

theorem quadratic_root_range (m : ℝ) (α β : ℝ) : 
  (∃ x, x^2 - 2*(m-1)*x + (m-1) = 0) ∧ 
  (α^2 - 2*(m-1)*α + (m-1) = 0) ∧ 
  (β^2 - 2*(m-1)*β + (m-1) = 0) ∧ 
  (0 < α) ∧ (α < 1) ∧ (1 < β) ∧ (β < 2) →
  (2 < m) ∧ (m < 7/3) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l197_19730
