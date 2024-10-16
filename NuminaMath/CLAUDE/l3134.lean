import Mathlib

namespace NUMINAMATH_CALUDE_fewer_baseball_cards_l3134_313423

theorem fewer_baseball_cards (hockey football baseball : ℕ) 
  (h1 : baseball < football)
  (h2 : football = 4 * hockey)
  (h3 : hockey = 200)
  (h4 : baseball + football + hockey = 1750) :
  football - baseball = 50 := by
sorry

end NUMINAMATH_CALUDE_fewer_baseball_cards_l3134_313423


namespace NUMINAMATH_CALUDE_unique_modular_solution_l3134_313430

theorem unique_modular_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 31 ∧ -250 ≡ n [ZMOD 31] := by sorry

end NUMINAMATH_CALUDE_unique_modular_solution_l3134_313430


namespace NUMINAMATH_CALUDE_circle_properties_l3134_313457

/-- The circle equation --/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 4*x + y^2 - 6*y - 12 = 0

/-- The center of the circle --/
def circle_center : ℝ × ℝ := (-2, 3)

/-- The radius of the circle --/
def circle_radius : ℝ := 5

/-- Theorem stating that the given equation represents a circle with the specified center and radius --/
theorem circle_properties :
  ∀ (x y : ℝ), circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l3134_313457


namespace NUMINAMATH_CALUDE_positive_value_of_A_l3134_313412

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem positive_value_of_A : 
  ∃ A : ℝ, A > 0 ∧ hash A 7 = 194 ∧ A = Real.sqrt 145 := by sorry

end NUMINAMATH_CALUDE_positive_value_of_A_l3134_313412


namespace NUMINAMATH_CALUDE_distance_between_trees_l3134_313496

theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) :
  yard_length = 360 →
  num_trees = 31 →
  yard_length / (num_trees - 1) = 12 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_l3134_313496


namespace NUMINAMATH_CALUDE_negation_of_forall_greater_than_one_negation_of_proposition_l3134_313461

theorem negation_of_forall_greater_than_one (P : ℝ → Prop) :
  (¬ ∀ x > 1, P x) ↔ (∃ x > 1, ¬ P x) :=
by sorry

theorem negation_of_proposition :
  (¬ ∀ x > 1, x^2 - x > 0) ↔ (∃ x > 1, x^2 - x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_greater_than_one_negation_of_proposition_l3134_313461


namespace NUMINAMATH_CALUDE_perfect_square_trinomials_l3134_313421

-- Perfect square trinomial properties
theorem perfect_square_trinomials 
  (x a b : ℝ) : 
  (x^2 + 6*x + 9 = (x + 3)^2) ∧ 
  (x^2 + 8*x + 16 = (x + 4)^2) ∧ 
  (x^2 - 12*x + 36 = (x - 6)^2) ∧ 
  (a^2 + 2*a*b + b^2 = (a + b)^2) ∧ 
  (a^2 - 2*a*b + b^2 = (a - b)^2) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomials_l3134_313421


namespace NUMINAMATH_CALUDE_difference_of_squares_l3134_313439

theorem difference_of_squares (a b : ℝ) : (2*a - b) * (2*a + b) = 4*a^2 - b^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3134_313439


namespace NUMINAMATH_CALUDE_smallest_distance_between_circles_l3134_313428

theorem smallest_distance_between_circles (z w : ℂ) 
  (hz : Complex.abs (z - (2 - 4*I)) = 2)
  (hw : Complex.abs (w - (5 - 6*I)) = 4) :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 13 - 6 ∧
    ∀ (z' w' : ℂ), Complex.abs (z' - (2 - 4*I)) = 2 →
      Complex.abs (w' - (5 - 6*I)) = 4 →
        Complex.abs (z' - w') ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_smallest_distance_between_circles_l3134_313428


namespace NUMINAMATH_CALUDE_cats_meowing_time_l3134_313463

/-- The number of minutes the cats were meowing -/
def minutes : ℚ := 5

/-- The number of meows per minute for the first cat -/
def first_cat_meows : ℚ := 3

/-- The number of meows per minute for the second cat -/
def second_cat_meows : ℚ := 2 * first_cat_meows

/-- The number of meows per minute for the third cat -/
def third_cat_meows : ℚ := (1/3) * second_cat_meows

/-- The total number of meows -/
def total_meows : ℚ := 55

theorem cats_meowing_time :
  minutes * (first_cat_meows + second_cat_meows + third_cat_meows) = total_meows :=
by sorry

end NUMINAMATH_CALUDE_cats_meowing_time_l3134_313463


namespace NUMINAMATH_CALUDE_factoring_transformation_l3134_313426

-- Define the concept of factoring
def is_factored (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g x ∧ ∃ (p q : ℝ → ℝ), g x = p x * q x

-- Define the specific expression
def left_expr : ℝ → ℝ := λ x ↦ x^2 - 4
def right_expr : ℝ → ℝ := λ x ↦ (x + 2) * (x - 2)

-- Theorem statement
theorem factoring_transformation :
  is_factored left_expr right_expr :=
sorry

end NUMINAMATH_CALUDE_factoring_transformation_l3134_313426


namespace NUMINAMATH_CALUDE_correct_result_l3134_313451

/-- Represents a five-digit number -/
structure FiveDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  e : Nat
  h1 : a ≥ 1 ∧ a ≤ 9
  h2 : b ≥ 0 ∧ b ≤ 9
  h3 : c ≥ 0 ∧ c ≤ 9
  h4 : d ≥ 0 ∧ d ≤ 9
  h5 : e ≥ 0 ∧ e ≤ 9

def reverseNumber (n : FiveDigitNumber) : Nat :=
  n.e * 10000 + n.d * 1000 + n.c * 100 + n.b * 10 + n.a

def originalNumber (n : FiveDigitNumber) : Nat :=
  n.a * 10000 + n.b * 1000 + n.c * 100 + n.d * 10 + n.e

theorem correct_result (n : FiveDigitNumber) 
  (h : reverseNumber n - originalNumber n = 34056) :
  n.e > n.a ∧ 
  n.e - n.a = 3 ∧ 
  (n.a - n.e) % 10 = 6 ∧ 
  n.b > n.d :=
sorry

end NUMINAMATH_CALUDE_correct_result_l3134_313451


namespace NUMINAMATH_CALUDE_cos_odd_function_phi_l3134_313480

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem cos_odd_function_phi (φ : ℝ) 
  (h1 : 0 ≤ φ) (h2 : φ ≤ π) 
  (h3 : is_odd_function (fun x ↦ Real.cos (x + φ))) : 
  φ = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_odd_function_phi_l3134_313480


namespace NUMINAMATH_CALUDE_v2_equals_5_l3134_313482

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- Definition of V₂ in Horner's method -/
def V₂ (a₅ a₄ a₃ a₂ a₁ a₀ : ℝ) (x : ℝ) : ℝ :=
  (a₅ * x + a₄) * x - a₃

/-- Theorem: V₂ equals 5 for the given polynomial when x = 2 -/
theorem v2_equals_5 :
  let f : ℝ → ℝ := fun x => 2 * x^5 - 3 * x^3 + 2 * x^2 - x + 5
  V₂ 2 0 (-3) 2 (-1) 5 2 = 5 := by
  sorry

#eval V₂ 2 0 (-3) 2 (-1) 5 2

end NUMINAMATH_CALUDE_v2_equals_5_l3134_313482


namespace NUMINAMATH_CALUDE_digits_of_8_power_10_times_3_power_15_l3134_313497

theorem digits_of_8_power_10_times_3_power_15 : ∃ (n : ℕ), 
  (10 ^ (n - 1) ≤ 8^10 * 3^15) ∧ (8^10 * 3^15 < 10^n) ∧ (n = 12) := by
  sorry

end NUMINAMATH_CALUDE_digits_of_8_power_10_times_3_power_15_l3134_313497


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3134_313433

def A : Set ℕ := {1, 2, 3, 4}

def B : Set ℕ := {x : ℕ | ∃ n ∈ A, x = 2 * n}

theorem intersection_of_A_and_B : A ∩ B = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3134_313433


namespace NUMINAMATH_CALUDE_octagon_diagonals_l3134_313477

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

theorem octagon_diagonals :
  num_diagonals octagon_sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l3134_313477


namespace NUMINAMATH_CALUDE_max_ab_internally_tangent_circles_l3134_313498

/-- Two circles C₁ and C₂ are internally tangent if the distance between their centers
    is equal to the difference of their radii. -/
def internally_tangent (a b : ℝ) : Prop :=
  (a + b)^2 = 1

/-- The equation of circle C₁ -/
def C₁ (x y a : ℝ) : Prop :=
  (x - a)^2 + (y + 2)^2 = 4

/-- The equation of circle C₂ -/
def C₂ (x y b : ℝ) : Prop :=
  (x + b)^2 + (y + 2)^2 = 1

/-- The theorem stating that the maximum value of ab is 1/4 -/
theorem max_ab_internally_tangent_circles (a b : ℝ) :
  internally_tangent a b → a * b ≤ 1/4 ∧ ∃ a b, internally_tangent a b ∧ a * b = 1/4 :=
sorry

end NUMINAMATH_CALUDE_max_ab_internally_tangent_circles_l3134_313498


namespace NUMINAMATH_CALUDE_cos_37_cos_23_minus_sin_37_sin_23_l3134_313418

theorem cos_37_cos_23_minus_sin_37_sin_23 :
  Real.cos (37 * π / 180) * Real.cos (23 * π / 180) - 
  Real.sin (37 * π / 180) * Real.sin (23 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_37_cos_23_minus_sin_37_sin_23_l3134_313418


namespace NUMINAMATH_CALUDE_jessica_birth_year_l3134_313467

theorem jessica_birth_year (first_amc8_year : ℕ) (jessica_age : ℕ) :
  first_amc8_year = 1985 →
  jessica_age = 15 →
  (first_amc8_year + 10 - 1) - jessica_age = 1979 :=
by
  sorry

end NUMINAMATH_CALUDE_jessica_birth_year_l3134_313467


namespace NUMINAMATH_CALUDE_complex_abs_one_plus_i_over_i_l3134_313427

theorem complex_abs_one_plus_i_over_i (i : ℂ) : i * i = -1 → Complex.abs ((1 + i) / i) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_abs_one_plus_i_over_i_l3134_313427


namespace NUMINAMATH_CALUDE_focus_after_symmetry_l3134_313471

/-- The parabola y² = -8x -/
def original_parabola (x y : ℝ) : Prop := y^2 = -8*x

/-- The line of symmetry y = x - 1 -/
def symmetry_line (x y : ℝ) : Prop := y = x - 1

/-- The focus of the original parabola -/
def original_focus : ℝ × ℝ := (-2, 0)

/-- The symmetric point with respect to a line -/
def symmetric_point (p : ℝ × ℝ) (line : ℝ → ℝ) : ℝ × ℝ := sorry

/-- The theorem stating the coordinates of the focus after symmetry -/
theorem focus_after_symmetry :
  symmetric_point original_focus (λ x => x - 1) = (1, -3) := by sorry

end NUMINAMATH_CALUDE_focus_after_symmetry_l3134_313471


namespace NUMINAMATH_CALUDE_probability_two_cards_sum_19_l3134_313403

/-- Represents a standard 52-card deck --/
def StandardDeck : ℕ := 52

/-- Number of cards that can be part of the pair (9 or 10) --/
def ValidFirstCards : ℕ := 8

/-- Number of complementary cards after drawing the first card --/
def ComplementaryCards : ℕ := 4

/-- Probability of drawing two number cards totaling 19 from a standard deck --/
theorem probability_two_cards_sum_19 :
  (ValidFirstCards : ℚ) / StandardDeck * ComplementaryCards / (StandardDeck - 1) = 8 / 663 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_cards_sum_19_l3134_313403


namespace NUMINAMATH_CALUDE_race_order_l3134_313484

-- Define the participants
inductive Participant : Type
  | Jia : Participant
  | Yi : Participant
  | Bing : Participant
  | Ding : Participant
  | Wu : Participant

-- Define a relation for "finished before"
def finished_before (a b : Participant) : Prop := sorry

-- Define the conditions
axiom ding_faster_than_yi : finished_before Participant.Ding Participant.Yi
axiom wu_before_bing : finished_before Participant.Wu Participant.Bing
axiom jia_between_bing_and_ding : 
  finished_before Participant.Bing Participant.Jia ∧ 
  finished_before Participant.Jia Participant.Ding

-- State the theorem
theorem race_order : 
  finished_before Participant.Wu Participant.Bing ∧
  finished_before Participant.Bing Participant.Jia ∧
  finished_before Participant.Jia Participant.Ding ∧
  finished_before Participant.Ding Participant.Yi :=
by sorry

end NUMINAMATH_CALUDE_race_order_l3134_313484


namespace NUMINAMATH_CALUDE_mike_song_book_price_l3134_313442

/-- The amount Mike received from selling the song book, given the cost of the trumpet and the net amount spent. -/
def song_book_price (trumpet_cost net_spent : ℚ) : ℚ :=
  trumpet_cost - net_spent

/-- Theorem stating that Mike sold the song book for $5.84, given the cost of the trumpet and the net amount spent. -/
theorem mike_song_book_price :
  let trumpet_cost : ℚ := 145.16
  let net_spent : ℚ := 139.32
  song_book_price trumpet_cost net_spent = 5.84 := by
  sorry

#eval song_book_price 145.16 139.32

end NUMINAMATH_CALUDE_mike_song_book_price_l3134_313442


namespace NUMINAMATH_CALUDE_arccos_cos_eight_l3134_313472

theorem arccos_cos_eight : Real.arccos (Real.cos 8) = 8 - 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_arccos_cos_eight_l3134_313472


namespace NUMINAMATH_CALUDE_arithmetic_equality_l3134_313485

theorem arithmetic_equality : 57 * 44 + 13 * 44 = 3080 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l3134_313485


namespace NUMINAMATH_CALUDE_negation_of_existential_absolute_value_l3134_313444

theorem negation_of_existential_absolute_value (x : ℝ) :
  (¬ ∃ x : ℝ, |x| < 1) ↔ (∀ x : ℝ, x ≤ -1 ∨ x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existential_absolute_value_l3134_313444


namespace NUMINAMATH_CALUDE_inequality_holds_iff_p_in_range_l3134_313435

theorem inequality_holds_iff_p_in_range :
  ∀ p : ℝ, p ≥ 0 →
  (∀ q : ℝ, q > 0 → (4 * (p * q^2 + p^2 * q + 4 * q^2 + 4 * p * q)) / (p + q) > 3 * p^2 * q) ↔
  p ∈ Set.Ici 0 ∩ Set.Iio 4 := by
sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_p_in_range_l3134_313435


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3134_313466

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence {a_n}, if a_3 + a_4 + a_5 + a_6 + a_7 = 25, then a_2 + a_8 = 10 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  (a 3 + a 4 + a 5 + a 6 + a 7 = 25) → (a 2 + a 8 = 10) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3134_313466


namespace NUMINAMATH_CALUDE_three_digit_divisible_by_11_l3134_313404

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ units ≤ 9

/-- Converts a ThreeDigitNumber to its numeric value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- Predicate to check if the middle digit is the sum of outer digits -/
def middleDigitIsSumOfOuter (n : ThreeDigitNumber) : Prop :=
  n.tens = n.hundreds + n.units

theorem three_digit_divisible_by_11 (n : ThreeDigitNumber) 
  (h : middleDigitIsSumOfOuter n) : 
  (n.toNat % 11 = 0) := by
  sorry

#check three_digit_divisible_by_11

end NUMINAMATH_CALUDE_three_digit_divisible_by_11_l3134_313404


namespace NUMINAMATH_CALUDE_openai_robotics_competition_weight_l3134_313476

/-- The weight of the standard robot in the OpenAI robotics competition. -/
def standard_robot_weight : ℝ := 100

/-- The maximum weight allowed for a robot in the competition. -/
def max_weight : ℝ := 210

/-- The minimum weight of a robot in the competition. -/
def min_weight : ℝ := standard_robot_weight + 5

theorem openai_robotics_competition_weight :
  standard_robot_weight = 100 ∧
  max_weight = 210 ∧
  min_weight = standard_robot_weight + 5 ∧
  max_weight ≤ 2 * min_weight :=
by sorry

end NUMINAMATH_CALUDE_openai_robotics_competition_weight_l3134_313476


namespace NUMINAMATH_CALUDE_system_solution_l3134_313407

theorem system_solution (k : ℚ) : 
  (∃ x y : ℚ, x + y = 5 * k ∧ x - y = 9 * k ∧ 2 * x + 3 * y = 6) → k = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3134_313407


namespace NUMINAMATH_CALUDE_compound_interest_rate_is_ten_percent_l3134_313408

/-- Given the conditions of the problem, prove that the compound interest rate is 10% --/
theorem compound_interest_rate_is_ten_percent
  (simple_principal : ℝ)
  (simple_rate : ℝ)
  (simple_time : ℝ)
  (compound_principal : ℝ)
  (compound_time : ℝ)
  (h1 : simple_principal = 1750.0000000000018)
  (h2 : simple_rate = 8)
  (h3 : simple_time = 3)
  (h4 : compound_principal = 4000)
  (h5 : compound_time = 2)
  (h6 : simple_principal * simple_rate * simple_time / 100 = 
        compound_principal * ((1 + compound_rate / 100) ^ compound_time - 1) / 2)
  : compound_rate = 10 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_is_ten_percent_l3134_313408


namespace NUMINAMATH_CALUDE_cubic_sequence_with_two_squares_exists_l3134_313409

/-- A cubic sequence with integer coefficients -/
def cubic_sequence (b c d : ℤ) (n : ℤ) : ℤ :=
  n^3 + b*n^2 + c*n + d

/-- Predicate for perfect squares -/
def is_perfect_square (x : ℤ) : Prop :=
  ∃ k : ℤ, x = k^2

theorem cubic_sequence_with_two_squares_exists :
  ∃ (b c d : ℤ),
    (is_perfect_square (cubic_sequence b c d 2015)) ∧
    (is_perfect_square (cubic_sequence b c d 2016)) ∧
    (∀ n : ℤ, n ≠ 2015 → n ≠ 2016 → ¬(is_perfect_square (cubic_sequence b c d n))) ∧
    (cubic_sequence b c d 2015 * cubic_sequence b c d 2016 = 0) :=
sorry

end NUMINAMATH_CALUDE_cubic_sequence_with_two_squares_exists_l3134_313409


namespace NUMINAMATH_CALUDE_xy_value_l3134_313432

theorem xy_value (x y : ℝ) 
  (h1 : (16 : ℝ)^x / (4 : ℝ)^(x + y) = 16)
  (h2 : (25 : ℝ)^(x + y) / (5 : ℝ)^(6 * y) = 625) :
  x * y = 0 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3134_313432


namespace NUMINAMATH_CALUDE_parabola_parameter_l3134_313417

/-- A parabola with equation y = ax² and latus rectum y = -1/2 has a = 1/2 --/
theorem parabola_parameter (a : ℝ) : 
  (∃ (x y : ℝ), y = a * x^2) ∧  -- Parabola equation
  (∃ (y : ℝ), y = -1/2) →       -- Latus rectum equation
  a = 1/2 := by sorry

end NUMINAMATH_CALUDE_parabola_parameter_l3134_313417


namespace NUMINAMATH_CALUDE_k_condition_necessary_not_sufficient_l3134_313459

/-- Defines the condition for k -/
def k_condition (k : ℝ) : Prop := 7 < k ∧ k < 9

/-- Defines when the equation represents an ellipse -/
def is_ellipse (k : ℝ) : Prop :=
  7 < k ∧ k < 9 ∧ k ≠ 8

/-- Theorem stating that k_condition is necessary but not sufficient for is_ellipse -/
theorem k_condition_necessary_not_sufficient :
  (∀ k : ℝ, is_ellipse k → k_condition k) ∧
  ¬(∀ k : ℝ, k_condition k → is_ellipse k) := by
  sorry


end NUMINAMATH_CALUDE_k_condition_necessary_not_sufficient_l3134_313459


namespace NUMINAMATH_CALUDE_hamans_dropped_trays_l3134_313425

theorem hamans_dropped_trays 
  (initial_trays : ℕ) 
  (additional_trays : ℕ) 
  (total_eggs_sold : ℕ) 
  (eggs_per_tray : ℕ) 
  (h1 : initial_trays = 10)
  (h2 : additional_trays = 7)
  (h3 : total_eggs_sold = 540)
  (h4 : eggs_per_tray = 30) :
  initial_trays + additional_trays + 1 - (total_eggs_sold / eggs_per_tray) = 8 :=
by sorry

end NUMINAMATH_CALUDE_hamans_dropped_trays_l3134_313425


namespace NUMINAMATH_CALUDE_annie_hamburgers_l3134_313479

theorem annie_hamburgers (initial_amount : ℕ) (hamburger_cost : ℕ) (milkshake_cost : ℕ)
  (milkshakes_bought : ℕ) (amount_left : ℕ) :
  initial_amount = 120 →
  hamburger_cost = 4 →
  milkshake_cost = 3 →
  milkshakes_bought = 6 →
  amount_left = 70 →
  ∃ (hamburgers_bought : ℕ),
    hamburgers_bought = 8 ∧
    initial_amount = amount_left + hamburger_cost * hamburgers_bought + milkshake_cost * milkshakes_bought :=
by
  sorry

end NUMINAMATH_CALUDE_annie_hamburgers_l3134_313479


namespace NUMINAMATH_CALUDE_abc_product_l3134_313468

theorem abc_product (a b c : ℕ) : 
  Nat.Prime a → 
  Nat.Prime b → 
  Nat.Prime c → 
  a * b * c < 10000 → 
  2 * a + 3 * b = c → 
  4 * a + c + 1 = 4 * b → 
  a * b * c = 1118 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l3134_313468


namespace NUMINAMATH_CALUDE_no_special_numbers_l3134_313401

/-- A number is prime if it's greater than 1 and has no divisors other than 1 and itself -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- A number is composite if it has more than two factors -/
def isComposite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, 1 < d ∧ d < n ∧ d ∣ n

/-- A number is a perfect square if it's the square of an integer -/
def isPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

/-- The set of integers from 1 to 1000 -/
def numberSet : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 1000}

theorem no_special_numbers : ∀ n ∈ numberSet, isPrime n ∨ isComposite n ∨ isPerfectSquare n := by
  sorry

end NUMINAMATH_CALUDE_no_special_numbers_l3134_313401


namespace NUMINAMATH_CALUDE_minimum_value_of_fraction_l3134_313452

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem minimum_value_of_fraction (a : ℕ → ℝ) (m n : ℕ) :
  geometric_sequence a →
  (∀ k : ℕ, a k > 0) →
  a 7 = a 6 + 2 * a 5 →
  ∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1 →
  (1 : ℝ) / m + 9 / n ≥ 8 / 3 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_of_fraction_l3134_313452


namespace NUMINAMATH_CALUDE_range_of_t_l3134_313402

def M : Set ℝ := {x | -2 < x ∧ x < 5}

def N (t : ℝ) : Set ℝ := {x | 2 - t < x ∧ x < 2*t + 1}

theorem range_of_t : 
  (∀ t : ℝ, M ∩ N t = N t) ↔ (∀ t : ℝ, t ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_t_l3134_313402


namespace NUMINAMATH_CALUDE_fixed_point_on_all_lines_l3134_313424

/-- The fixed point through which all lines of a certain form pass -/
def fixed_point : ℝ × ℝ := (2, 1)

/-- The line equation parameterized by k -/
def line_equation (k : ℝ) (x y : ℝ) : Prop :=
  k * x - y + 1 - 2 * k = 0

/-- Theorem stating that the fixed point lies on all lines of the given form -/
theorem fixed_point_on_all_lines :
  ∀ k : ℝ, line_equation k (fixed_point.1) (fixed_point.2) :=
by
  sorry

#check fixed_point_on_all_lines

end NUMINAMATH_CALUDE_fixed_point_on_all_lines_l3134_313424


namespace NUMINAMATH_CALUDE_solve_for_a_l3134_313436

theorem solve_for_a : ∃ a : ℝ, 
  (∃ x y : ℝ, x = 1 ∧ y = 2 ∧ a * x - y = 3) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l3134_313436


namespace NUMINAMATH_CALUDE_at_least_one_black_probability_l3134_313465

def total_balls : ℕ := 4
def white_balls : ℕ := 2
def black_balls : ℕ := 2
def drawn_balls : ℕ := 2

def probability_at_least_one_black : ℚ := 5 / 6

theorem at_least_one_black_probability :
  probability_at_least_one_black = 
    (Nat.choose total_balls drawn_balls - Nat.choose white_balls drawn_balls) / 
    Nat.choose total_balls drawn_balls :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_black_probability_l3134_313465


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_tangency_l3134_313431

/-- The value of m that makes the ellipse x^2 + 9y^2 = 9 tangent to the hyperbola x^2 - m(y+3)^2 = 4 -/
def tangency_value : ℚ := 5/54

/-- Definition of the ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 + 9*y^2 = 9

/-- Definition of the hyperbola equation -/
def is_on_hyperbola (x y m : ℝ) : Prop := x^2 - m*(y+3)^2 = 4

/-- Theorem stating that 5/54 is the value of m that makes the ellipse tangent to the hyperbola -/
theorem ellipse_hyperbola_tangency :
  ∃! (m : ℝ), m = tangency_value ∧ 
  (∃! (x y : ℝ), is_on_ellipse x y ∧ is_on_hyperbola x y m) :=
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_tangency_l3134_313431


namespace NUMINAMATH_CALUDE_typing_speed_equation_l3134_313437

theorem typing_speed_equation (x : ℝ) : x > 0 → x + 6 > 0 →
  (Xiao_Ming_speed : ℝ) →
  (Xiao_Zhang_speed : ℝ) →
  (Xiao_Ming_speed = x) →
  (Xiao_Zhang_speed = x + 6) →
  (120 / Xiao_Ming_speed = 180 / Xiao_Zhang_speed) →
  120 / x = 180 / (x + 6) := by
sorry

end NUMINAMATH_CALUDE_typing_speed_equation_l3134_313437


namespace NUMINAMATH_CALUDE_bacteria_habitat_limits_l3134_313478

/-- Represents a bacterial colony with its growth characteristics -/
structure BacterialColony where
  growthFactor : ℕ       -- How much the colony multiplies in size
  growthPeriod : ℕ       -- Number of days between each growth
  totalDays : ℕ          -- Total number of days the colony grows

/-- Calculates the number of days it takes for a colony to reach its habitat limit -/
def daysToHabitatLimit (colony : BacterialColony) : ℕ :=
  colony.totalDays

/-- Colony A doubles every day for 22 days -/
def colonyA : BacterialColony :=
  { growthFactor := 2
  , growthPeriod := 1
  , totalDays := 22 }

/-- Colony B triples every 2 days for 30 days -/
def colonyB : BacterialColony :=
  { growthFactor := 3
  , growthPeriod := 2
  , totalDays := 30 }

theorem bacteria_habitat_limits :
  daysToHabitatLimit colonyA = 22 ∧ daysToHabitatLimit colonyB = 30 := by
  sorry

#eval daysToHabitatLimit colonyA
#eval daysToHabitatLimit colonyB

end NUMINAMATH_CALUDE_bacteria_habitat_limits_l3134_313478


namespace NUMINAMATH_CALUDE_lauren_change_l3134_313449

/-- Represents the grocery items and their prices --/
structure GroceryItems where
  meat_price : ℝ
  meat_weight : ℝ
  buns_price : ℝ
  lettuce_price : ℝ
  tomato_price : ℝ
  tomato_weight : ℝ
  pickles_price : ℝ
  pickle_coupon : ℝ

/-- Calculates the total cost of the grocery items --/
def total_cost (items : GroceryItems) : ℝ :=
  items.meat_price * items.meat_weight +
  items.buns_price +
  items.lettuce_price +
  items.tomato_price * items.tomato_weight +
  (items.pickles_price - items.pickle_coupon)

/-- Calculates the change from a given payment --/
def calculate_change (items : GroceryItems) (payment : ℝ) : ℝ :=
  payment - total_cost items

/-- Theorem stating that Lauren's change is $6.00 --/
theorem lauren_change :
  let items : GroceryItems := {
    meat_price := 3.5,
    meat_weight := 2,
    buns_price := 1.5,
    lettuce_price := 1,
    tomato_price := 2,
    tomato_weight := 1.5,
    pickles_price := 2.5,
    pickle_coupon := 1
  }
  calculate_change items 20 = 6 := by sorry

end NUMINAMATH_CALUDE_lauren_change_l3134_313449


namespace NUMINAMATH_CALUDE_louie_junior_took_seven_cookies_l3134_313438

/-- Represents the number of cookies in various states --/
structure CookieJar where
  initial : Nat
  eatenByLouSenior : Nat
  remaining : Nat

/-- Calculates the number of cookies Louie Junior took --/
def cookiesTakenByLouieJunior (jar : CookieJar) : Nat :=
  jar.initial - jar.eatenByLouSenior - jar.remaining

/-- Theorem stating that Louie Junior took 7 cookies --/
theorem louie_junior_took_seven_cookies (jar : CookieJar) 
  (h1 : jar.initial = 22)
  (h2 : jar.eatenByLouSenior = 4)
  (h3 : jar.remaining = 11) :
  cookiesTakenByLouieJunior jar = 7 := by
  sorry

#eval cookiesTakenByLouieJunior { initial := 22, eatenByLouSenior := 4, remaining := 11 }

end NUMINAMATH_CALUDE_louie_junior_took_seven_cookies_l3134_313438


namespace NUMINAMATH_CALUDE_arithmetic_squares_sequence_l3134_313492

theorem arithmetic_squares_sequence (k : ℤ) : 
  (∃! k : ℤ, 
    (∃ a : ℤ, 
      (49 + k = a^2) ∧ 
      (361 + k = (a + 2)^2) ∧ 
      (784 + k = (a + 4)^2))) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_squares_sequence_l3134_313492


namespace NUMINAMATH_CALUDE_least_non_lucky_multiple_of_7_l3134_313494

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def isLuckyInteger (n : ℕ) : Prop :=
  n > 0 ∧ n % sumOfDigits n = 0

def isMultipleOf7 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 7 * k

theorem least_non_lucky_multiple_of_7 :
  (∀ n : ℕ, n > 0 ∧ n < 14 ∧ isMultipleOf7 n → isLuckyInteger n) ∧
  isMultipleOf7 14 ∧
  ¬isLuckyInteger 14 :=
sorry

end NUMINAMATH_CALUDE_least_non_lucky_multiple_of_7_l3134_313494


namespace NUMINAMATH_CALUDE_sum_after_transformation_l3134_313405

theorem sum_after_transformation (S a b : ℝ) (h : a + b = S) :
  3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_after_transformation_l3134_313405


namespace NUMINAMATH_CALUDE_remainder_problem_l3134_313488

theorem remainder_problem (n : ℤ) (h : ∃ k : ℤ, n = 100 * k - 1) : 
  (n^3 + n^2 + 2*n + 3) % 100 = 1 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l3134_313488


namespace NUMINAMATH_CALUDE_f_has_extrema_l3134_313443

/-- The function f(x) = 2 - x^2 - x^3 -/
def f (x : ℝ) : ℝ := 2 - x^2 - x^3

/-- Theorem stating that f has both a maximum and a minimum value -/
theorem f_has_extrema : 
  (∃ a : ℝ, ∀ x : ℝ, f x ≤ f a) ∧ (∃ b : ℝ, ∀ x : ℝ, f x ≥ f b) :=
sorry

end NUMINAMATH_CALUDE_f_has_extrema_l3134_313443


namespace NUMINAMATH_CALUDE_equilibrium_constant_temperature_relation_l3134_313458

-- Define the chemical equilibrium constant
variable (K : ℝ)

-- Define temperature
variable (T : ℝ)

-- Define a relation between K and T
def related_to_temperature (K T : ℝ) : Prop := sorry

-- Theorem stating that K is related to temperature
theorem equilibrium_constant_temperature_relation :
  related_to_temperature K T :=
sorry

end NUMINAMATH_CALUDE_equilibrium_constant_temperature_relation_l3134_313458


namespace NUMINAMATH_CALUDE_three_tangent_circles_range_l3134_313411

/-- Two circles with exactly three common tangents -/
structure ThreeTangentCircles where
  a : ℝ
  b : ℝ
  c1 : (x : ℝ) → (y : ℝ) → Prop := λ x y ↦ (x - a)^2 + y^2 = 1
  c2 : (x : ℝ) → (y : ℝ) → Prop := λ x y ↦ x^2 + y^2 - 2*b*y + b^2 - 4 = 0
  three_tangents : ∃! (p : ℝ × ℝ), c1 p.1 p.2 ∧ c2 p.1 p.2

/-- The range of a² + b² - 6a - 8b for circles with three common tangents -/
theorem three_tangent_circles_range (circles : ThreeTangentCircles) :
  -21 ≤ circles.a^2 + circles.b^2 - 6*circles.a - 8*circles.b ∧
  circles.a^2 + circles.b^2 - 6*circles.a - 8*circles.b ≤ 39 := by
  sorry

end NUMINAMATH_CALUDE_three_tangent_circles_range_l3134_313411


namespace NUMINAMATH_CALUDE_dice_sum_product_l3134_313422

theorem dice_sum_product (a b c d : ℕ) : 
  1 ≤ a ∧ a ≤ 6 ∧
  1 ≤ b ∧ b ≤ 6 ∧
  1 ≤ c ∧ c ≤ 6 ∧
  1 ≤ d ∧ d ≤ 6 ∧
  a * b * c * d = 180 →
  a + b + c + d ≠ 14 ∧ a + b + c + d ≠ 17 := by
sorry

end NUMINAMATH_CALUDE_dice_sum_product_l3134_313422


namespace NUMINAMATH_CALUDE_factorization_proof_l3134_313410

theorem factorization_proof (c : ℝ) : 196 * c^2 + 42 * c - 14 = 14 * c * (14 * c + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3134_313410


namespace NUMINAMATH_CALUDE_min_value_expression_l3134_313456

theorem min_value_expression (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x * y = 1) :
  ∃ (t : ℝ), t = 25 ∧ ∀ (z : ℝ), (3 * x^3 + 125 * y^3) / (x - y) ≥ z := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3134_313456


namespace NUMINAMATH_CALUDE_vector_length_range_l3134_313413

theorem vector_length_range (a b : ℝ × ℝ) 
  (h1 : ‖a‖ = 1) 
  (h2 : ‖a + b‖ = 2) : 
  1 ≤ ‖b‖ ∧ ‖b‖ ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_length_range_l3134_313413


namespace NUMINAMATH_CALUDE_point_not_in_plane_l3134_313490

/-- Defines the plane area represented by the inequality 3x + 2y < 6 -/
def plane_area (x y : ℝ) : Prop := 3 * x + 2 * y < 6

/-- Theorem stating that the point (2, 0) is not in the plane area -/
theorem point_not_in_plane : ¬ plane_area 2 0 := by sorry

end NUMINAMATH_CALUDE_point_not_in_plane_l3134_313490


namespace NUMINAMATH_CALUDE_min_cuts_for_4x4x4_cube_l3134_313474

/-- Represents a cube with given dimensions -/
structure Cube where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a cut operation on a cube -/
inductive Cut
  | X : Cut  -- Cut parallel to YZ plane
  | Y : Cut  -- Cut parallel to XZ plane
  | Z : Cut  -- Cut parallel to XY plane

/-- Function to calculate the minimum number of cuts required -/
def min_cuts_to_unit_cubes (c : Cube) : ℕ :=
  sorry

/-- Theorem stating the minimum number of cuts required for a 4x4x4 cube -/
theorem min_cuts_for_4x4x4_cube :
  let initial_cube : Cube := { length := 4, width := 4, height := 4 }
  min_cuts_to_unit_cubes initial_cube = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_cuts_for_4x4x4_cube_l3134_313474


namespace NUMINAMATH_CALUDE_square_root_of_sqrt_16_l3134_313446

theorem square_root_of_sqrt_16 : 
  {x : ℝ | x^2 = Real.sqrt 16} = {2, -2} := by sorry

end NUMINAMATH_CALUDE_square_root_of_sqrt_16_l3134_313446


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l3134_313434

theorem quadratic_equation_properties (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*m*x₁ + m^2 - 1 = 0 ∧ x₂^2 + 2*m*x₂ + m^2 - 1 = 0) ∧
  ((-2)^2 + 2*m*(-2) + m^2 - 1 = 0 → 2023 - m^2 + 4*m = 2026) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l3134_313434


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3134_313448

/-- An equilateral triangle with three inscribed circles -/
structure TriangleWithCircles where
  -- The side length of the equilateral triangle
  side : ℝ
  -- The radius of each inscribed circle
  radius : ℝ
  -- The offset from each vertex to the nearest point on any circle
  offset : ℝ
  -- Condition: The radius is 2
  h_radius : radius = 2
  -- Condition: The offset is 1
  h_offset : offset = 1
  -- Condition: The circles touch each other and the sides of the triangle
  h_touch : side = 2 * (radius + offset) + 2 * radius * Real.sqrt 3

/-- The perimeter of the triangle is 6√3 + 12 -/
theorem triangle_perimeter (t : TriangleWithCircles) : 
  3 * t.side = 6 * Real.sqrt 3 + 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3134_313448


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l3134_313440

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then Real.arctan (x^3 - x^(3/2) * Real.sin (1 / (3*x)))
  else 0

theorem f_derivative_at_zero : 
  deriv f 0 = 0 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l3134_313440


namespace NUMINAMATH_CALUDE_locus_of_N_l3134_313415

/-- The circle on which point M moves -/
def Circle (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y = 0

/-- Point N lies on the ray OM -/
def OnRay (x y : ℝ) : Prop := ∃ (t : ℝ), t > 0 ∧ x = t * (x/((x^2 + y^2)^(1/2))) ∧ y = t * (y/((x^2 + y^2)^(1/2)))

/-- The product of distances |OM| and |ON| is 150 -/
def DistanceProduct (x y : ℝ) : Prop := (x^2 + y^2)^(1/2) * ((x^2 + y^2)^(1/2) / (x^2 + y^2)) = 150

theorem locus_of_N (x y : ℝ) :
  (∃ (mx my : ℝ), Circle mx my ∧ OnRay x y ∧ DistanceProduct x y) →
  3*x + 4*y = 75 := by sorry

end NUMINAMATH_CALUDE_locus_of_N_l3134_313415


namespace NUMINAMATH_CALUDE_left_handed_jazz_lovers_l3134_313462

/-- Represents a club with members of different handedness and music preferences -/
structure Club where
  total : ℕ
  leftHanded : ℕ
  ambidextrous : ℕ
  rightHanded : ℕ
  jazzLovers : ℕ
  rightHandedJazzDislikers : ℕ

/-- Theorem stating the number of left-handed jazz lovers in the club -/
theorem left_handed_jazz_lovers (c : Club) 
  (h1 : c.total = 30)
  (h2 : c.leftHanded = 12)
  (h3 : c.ambidextrous = 3)
  (h4 : c.rightHanded = c.total - c.leftHanded - c.ambidextrous)
  (h5 : c.jazzLovers = 20)
  (h6 : c.rightHandedJazzDislikers = 4) :
  ∃ x : ℕ, x = 6 ∧ 
    x ≤ c.leftHanded ∧ 
    x + (c.rightHanded - c.rightHandedJazzDislikers) + c.ambidextrous = c.jazzLovers :=
  sorry


end NUMINAMATH_CALUDE_left_handed_jazz_lovers_l3134_313462


namespace NUMINAMATH_CALUDE_rolling_semicircle_path_length_l3134_313499

/-- The length of the path traveled by the center of a rolling semicircular arc -/
theorem rolling_semicircle_path_length (r : ℝ) (h : r > 0) :
  let path_length := 3 * Real.pi * r
  path_length = (Real.pi * (2 * r)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_rolling_semicircle_path_length_l3134_313499


namespace NUMINAMATH_CALUDE_min_xy_value_l3134_313441

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 10 * x + 2 * y + 60 = x * y) : 
  x * y ≥ 180 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 10 * x₀ + 2 * y₀ + 60 = x₀ * y₀ ∧ x₀ * y₀ = 180 := by
  sorry

end NUMINAMATH_CALUDE_min_xy_value_l3134_313441


namespace NUMINAMATH_CALUDE_total_turnips_l3134_313469

theorem total_turnips (melanie_turnips benny_turnips : ℕ) 
  (h1 : melanie_turnips = 139)
  (h2 : benny_turnips = 113) :
  melanie_turnips + benny_turnips = 252 := by
  sorry

end NUMINAMATH_CALUDE_total_turnips_l3134_313469


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3134_313486

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℚ), 
    P = -3/5 ∧ Q = -1 ∧ R = 13/5 ∧
    ∀ (x : ℚ), x ≠ 1 → x ≠ 4 → x ≠ 6 →
      (x^2 - 10) / ((x - 1) * (x - 4) * (x - 6)) = 
      P / (x - 1) + Q / (x - 4) + R / (x - 6) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3134_313486


namespace NUMINAMATH_CALUDE_smaller_cuboid_length_l3134_313420

/-- Proves that the length of smaller cuboids is 5 meters, given the specified conditions --/
theorem smaller_cuboid_length : 
  ∀ (large_length large_width large_height : ℝ) 
    (small_width small_height : ℝ) 
    (num_small_cuboids : ℕ),
  large_length = 18 →
  large_width = 15 →
  large_height = 2 →
  small_width = 2 →
  small_height = 3 →
  num_small_cuboids = 18 →
  ∃ (small_length : ℝ),
    small_length = 5 ∧
    large_length * large_width * large_height = 
      num_small_cuboids * small_length * small_width * small_height :=
by sorry

end NUMINAMATH_CALUDE_smaller_cuboid_length_l3134_313420


namespace NUMINAMATH_CALUDE_root_product_sum_l3134_313489

theorem root_product_sum (x₁ x₂ x₃ : ℝ) : 
  x₁ < x₂ ∧ x₂ < x₃ ∧
  (∀ x, Real.sqrt 2014 * x^3 - 4029 * x^2 + 2 = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) →
  x₂ * (x₁ + x₃) = 2 := by
sorry

end NUMINAMATH_CALUDE_root_product_sum_l3134_313489


namespace NUMINAMATH_CALUDE_production_target_is_1800_l3134_313475

/-- Calculates the yearly production target for a car manufacturing company. -/
def yearly_production_target (current_monthly_production : ℕ) (monthly_increase : ℕ) : ℕ :=
  (current_monthly_production + monthly_increase) * 12

/-- Theorem: The yearly production target is 1800 cars. -/
theorem production_target_is_1800 :
  yearly_production_target 100 50 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_production_target_is_1800_l3134_313475


namespace NUMINAMATH_CALUDE_shopkeeper_profit_calculation_l3134_313400

theorem shopkeeper_profit_calculation 
  (C L S : ℝ)
  (h1 : L = C * (1 + intended_profit_percentage))
  (h2 : S = 0.9 * L)
  (h3 : S = 1.35 * C)
  : intended_profit_percentage = 0.5 :=
by sorry


end NUMINAMATH_CALUDE_shopkeeper_profit_calculation_l3134_313400


namespace NUMINAMATH_CALUDE_spinsters_to_cats_ratio_l3134_313416

/-- Given the number of spinsters and cats, prove their ratio is 2:7 -/
theorem spinsters_to_cats_ratio :
  ∀ (spinsters cats : ℕ),
    spinsters = 14 →
    cats = spinsters + 35 →
    (spinsters : ℚ) / (cats : ℚ) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_spinsters_to_cats_ratio_l3134_313416


namespace NUMINAMATH_CALUDE_coaching_charges_calculation_l3134_313483

/-- Number of days from January 1 to November 4 in a non-leap year -/
def daysOfCoaching : Nat := 308

/-- Total payment for coaching in dollars -/
def totalPayment : Int := 7038

/-- Daily coaching charges in dollars -/
def dailyCharges : ℚ := totalPayment / daysOfCoaching

theorem coaching_charges_calculation :
  dailyCharges = 7038 / 308 := by sorry

end NUMINAMATH_CALUDE_coaching_charges_calculation_l3134_313483


namespace NUMINAMATH_CALUDE_fair_attendance_l3134_313473

theorem fair_attendance (projected_increase : Real) (actual_decrease : Real) :
  projected_increase = 0.25 →
  actual_decrease = 0.20 →
  (1 - actual_decrease) / (1 + projected_increase) * 100 = 64 := by
sorry

end NUMINAMATH_CALUDE_fair_attendance_l3134_313473


namespace NUMINAMATH_CALUDE_fraction_equality_implies_x_equals_one_l3134_313470

theorem fraction_equality_implies_x_equals_one :
  ∀ x : ℚ, (5 + x) / (7 + x) = (2 + x) / (3 + x) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_x_equals_one_l3134_313470


namespace NUMINAMATH_CALUDE_workshop_duration_is_450_l3134_313414

/-- Calculates the duration of a workshop excluding breaks -/
def workshop_duration (total_hours : ℕ) (total_minutes : ℕ) (break_minutes : ℕ) : ℕ :=
  total_hours * 60 + total_minutes - break_minutes

/-- Theorem: The workshop duration excluding breaks is 450 minutes -/
theorem workshop_duration_is_450 :
  workshop_duration 8 20 50 = 450 := by
  sorry

end NUMINAMATH_CALUDE_workshop_duration_is_450_l3134_313414


namespace NUMINAMATH_CALUDE_fraction_of_complex_l3134_313454

def complex_i : ℂ := Complex.I

theorem fraction_of_complex (z : ℂ) (h : z = 1 + complex_i) : 2 / z = 1 - complex_i := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_complex_l3134_313454


namespace NUMINAMATH_CALUDE_problem_statement_l3134_313487

theorem problem_statement (a b : ℝ) 
  (h1 : |a| = 5)
  (h2 : |b| = 7)
  (h3 : |a + b| = a + b) :
  a - b = -2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3134_313487


namespace NUMINAMATH_CALUDE_f_positive_range_min_k_for_f_plus_k_positive_l3134_313406

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x - a * x

theorem f_positive_range (a : ℝ) :
  (∀ x, f a x > 0 ↔ x > 0) ∨
  (∀ x, f a x > 0 ↔ (x > 0 ∨ x < Real.log a)) ∨
  (∀ x, f a x > 0 ↔ (x > Real.log a ∨ x < 0)) :=
sorry

theorem min_k_for_f_plus_k_positive :
  ∃! k : ℕ, k > 0 ∧ ∀ x, f 2 x + k > 0 ∧ ∀ m : ℕ, m < k → ∃ y, f 2 y + m ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_f_positive_range_min_k_for_f_plus_k_positive_l3134_313406


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l3134_313455

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  arithmetic : ∀ n, a (n + 1) = a n + d
  S : ℕ → ℝ  -- Sum function
  sum_def : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2

/-- Theorem: For an arithmetic sequence, if S₁/S₄ = 1/10, then S₃/S₅ = 2/5 -/
theorem arithmetic_sequence_sum_ratio 
  (seq : ArithmeticSequence) 
  (h : seq.S 1 / seq.S 4 = 1 / 10) : 
  seq.S 3 / seq.S 5 = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l3134_313455


namespace NUMINAMATH_CALUDE_total_equivalent_pencils_is_139_9_l3134_313445

/-- Calculates the total equivalent number of pencils in three drawers after additions and removals --/
def totalEquivalentPencils (
  initialPencils1 : Float
  ) (initialPencils2 : Float
  ) (initialPens3 : Float
  ) (mikeAddedPencils1 : Float
  ) (sarahAddedPencils2 : Float
  ) (sarahAddedPens2 : Float
  ) (joeRemovedPencils1 : Float
  ) (joeRemovedPencils2 : Float
  ) (joeRemovedPens3 : Float
  ) (exchangeRate : Float
  ) : Float :=
  let finalPencils1 := initialPencils1 + mikeAddedPencils1 - joeRemovedPencils1
  let finalPencils2 := initialPencils2 + sarahAddedPencils2 - joeRemovedPencils2
  let finalPens3 := initialPens3 + sarahAddedPens2 - joeRemovedPens3
  let totalPencils := finalPencils1 + finalPencils2 + (finalPens3 * exchangeRate)
  totalPencils

theorem total_equivalent_pencils_is_139_9 :
  totalEquivalentPencils 41.5 25.2 13.6 30.7 18.5 8.4 5.3 7.1 3.8 2 = 139.9 := by
  sorry

end NUMINAMATH_CALUDE_total_equivalent_pencils_is_139_9_l3134_313445


namespace NUMINAMATH_CALUDE_circle_equation_l3134_313429

/-- A circle with center (2, -3) passing through the origin has the equation (x - 2)^2 + (y + 3)^2 = 13 -/
theorem circle_equation (x y : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ 
    ((x - 2)^2 + (y + 3)^2 = r^2) ∧ 
    (0 - 2)^2 + (0 + 3)^2 = r^2) ↔ 
  (x - 2)^2 + (y + 3)^2 = 13 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3134_313429


namespace NUMINAMATH_CALUDE_maintenance_check_increase_l3134_313495

theorem maintenance_check_increase (original_time : ℝ) (increase_percentage : ℝ) (new_time : ℝ) :
  original_time = 25 →
  increase_percentage = 20 →
  new_time = original_time * (1 + increase_percentage / 100) →
  new_time = 30 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_check_increase_l3134_313495


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_seven_thirds_l3134_313447

theorem ceiling_floor_sum_seven_thirds : ⌈(-7 : ℚ) / 3⌉ + ⌊(7 : ℚ) / 3⌋ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_seven_thirds_l3134_313447


namespace NUMINAMATH_CALUDE_square_sum_equality_l3134_313419

-- Define the problem statement
theorem square_sum_equality (x y : ℝ) 
  (h1 : y + 9 = (x - 3)^2) 
  (h2 : x + 9 = (y - 3)^2) 
  (h3 : x ≠ y) : 
  x^2 + y^2 = 49 := by
sorry

-- Additional helper lemmas if needed
lemma helper_lemma (x y : ℝ) 
  (h1 : y + 9 = (x - 3)^2) 
  (h2 : x + 9 = (y - 3)^2) 
  (h3 : x ≠ y) : 
  x + y = 7 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equality_l3134_313419


namespace NUMINAMATH_CALUDE_jeffrey_bottle_caps_l3134_313453

/-- 
Given that Jeffrey can create 6 groups of bottle caps with 2 bottle caps in each group,
prove that the total number of bottle caps is 12.
-/
theorem jeffrey_bottle_caps : 
  let groups : ℕ := 6
  let caps_per_group : ℕ := 2
  groups * caps_per_group = 12 := by sorry

end NUMINAMATH_CALUDE_jeffrey_bottle_caps_l3134_313453


namespace NUMINAMATH_CALUDE_hotel_assignment_count_l3134_313450

/-- Represents the number of rooms in the hotel -/
def num_rooms : ℕ := 4

/-- Represents the number of friends arriving -/
def num_friends : ℕ := 6

/-- Represents the maximum number of friends allowed per room -/
def max_per_room : ℕ := 3

/-- Calculates the number of ways to assign friends to rooms -/
def num_assignments : ℕ :=
  -- The actual calculation is not provided here
  1560

/-- Theorem stating that the number of assignments is 1560 -/
theorem hotel_assignment_count :
  num_assignments = 1560 :=
sorry

end NUMINAMATH_CALUDE_hotel_assignment_count_l3134_313450


namespace NUMINAMATH_CALUDE_lattice_point_proximity_probability_l3134_313493

theorem lattice_point_proximity_probability (r : ℝ) : 
  (r > 0) → 
  (π * r^2 = 1/3) → 
  (∃ (p : ℝ × ℝ), p.1 ≥ 0 ∧ p.1 ≤ 1 ∧ p.2 ≥ 0 ∧ p.2 ≤ 1 ∧ 
    ((p.1^2 + p.2^2 ≤ r^2) ∨ 
     ((1 - p.1)^2 + p.2^2 ≤ r^2) ∨ 
     (p.1^2 + (1 - p.2)^2 ≤ r^2) ∨ 
     ((1 - p.1)^2 + (1 - p.2)^2 ≤ r^2))) = 
  (r = Real.sqrt (1 / (3 * π))) :=
sorry

end NUMINAMATH_CALUDE_lattice_point_proximity_probability_l3134_313493


namespace NUMINAMATH_CALUDE_square_side_length_l3134_313481

theorem square_side_length (perimeter : ℝ) (h : perimeter = 17.8) :
  let side_length := perimeter / 4
  side_length = 4.45 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l3134_313481


namespace NUMINAMATH_CALUDE_cookies_sold_l3134_313491

def trip_cost : ℕ := 5000
def hourly_wage : ℕ := 20
def hours_worked : ℕ := 10
def cookie_price : ℕ := 4
def lottery_win : ℕ := 500
def sister_gift : ℕ := 500
def remaining_needed : ℕ := 3214

theorem cookies_sold :
  ∃ (n : ℕ), n * cookie_price = 
    trip_cost - 
    (hourly_wage * hours_worked + 
     lottery_win + 
     2 * sister_gift + 
     remaining_needed) :=
by sorry

end NUMINAMATH_CALUDE_cookies_sold_l3134_313491


namespace NUMINAMATH_CALUDE_sandys_puppies_l3134_313460

/-- Given that Sandy initially had 8 puppies and gave away 4, prove that she now has 4 puppies. -/
theorem sandys_puppies (initial_puppies : ℕ) (given_away : ℕ) (h1 : initial_puppies = 8) (h2 : given_away = 4) :
  initial_puppies - given_away = 4 := by
  sorry

end NUMINAMATH_CALUDE_sandys_puppies_l3134_313460


namespace NUMINAMATH_CALUDE_problem_solution_l3134_313464

theorem problem_solution (a b c d m n : ℕ+) 
  (h1 : a^2 + b^2 + c^2 + d^2 = 1989)
  (h2 : a + b + c + d = m^2)
  (h3 : max a (max b (max c d)) = n^2) :
  m = 9 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3134_313464
