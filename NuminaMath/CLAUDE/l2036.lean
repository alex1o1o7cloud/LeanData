import Mathlib

namespace NUMINAMATH_CALUDE_sophias_age_problem_l2036_203640

/-- Sophia's age problem -/
theorem sophias_age_problem (S M : ℝ) (h1 : S > 0) (h2 : M > 0) 
  (h3 : ∃ (x : ℝ), S = 3 * x ∧ x > 0)  -- S is thrice the sum of children's ages
  (h4 : S - M = 4 * ((S / 3) - 2 * M)) :  -- Condition about age M years ago
  S / M = 21 := by
sorry

end NUMINAMATH_CALUDE_sophias_age_problem_l2036_203640


namespace NUMINAMATH_CALUDE_vector_calculation_l2036_203607

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_calculation (a b : V) : 
  (1 / 2 : ℝ) • ((2 : ℝ) • a - (4 : ℝ) • b) + (2 : ℝ) • b = a := by
  sorry

end NUMINAMATH_CALUDE_vector_calculation_l2036_203607


namespace NUMINAMATH_CALUDE_book_cost_l2036_203658

theorem book_cost (initial_amount : ℕ) (num_books : ℕ) (remaining_amount : ℕ) 
  (h1 : initial_amount = 79)
  (h2 : num_books = 9)
  (h3 : remaining_amount = 16) :
  (initial_amount - remaining_amount) / num_books = 7 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_l2036_203658


namespace NUMINAMATH_CALUDE_gcd_power_minus_one_l2036_203618

theorem gcd_power_minus_one : Nat.gcd (2^2000 - 1) (2^1990 - 1) = 2^10 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_minus_one_l2036_203618


namespace NUMINAMATH_CALUDE_tan_alpha_minus_beta_equals_one_l2036_203632

theorem tan_alpha_minus_beta_equals_one (α β : ℝ) 
  (h : (3 / (2 + Real.sin (2 * α))) + (2021 / (2 + Real.sin β)) = 2024) : 
  Real.tan (α - β) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_minus_beta_equals_one_l2036_203632


namespace NUMINAMATH_CALUDE_two_plus_three_equals_eight_is_proposition_x_minus_one_equals_zero_is_not_proposition_do_you_speak_english_is_not_proposition_this_is_a_big_tree_is_not_proposition_only_two_plus_three_equals_eight_is_proposition_l2036_203653

-- Define what a proposition is
def is_proposition (s : String) : Prop := 
  ∃ (b : Bool), (s = "true" ∧ b = true) ∨ (s = "false" ∧ b = false)

-- Theorem stating that "2+3=8" is a proposition
theorem two_plus_three_equals_eight_is_proposition : 
  is_proposition "2+3=8" := by sorry

-- Theorem stating that "x-1=0" is not a proposition
theorem x_minus_one_equals_zero_is_not_proposition : 
  ¬ is_proposition "x-1=0" := by sorry

-- Theorem stating that "Do you speak English?" is not a proposition
theorem do_you_speak_english_is_not_proposition : 
  ¬ is_proposition "Do you speak English?" := by sorry

-- Theorem stating that "This is a big tree" is not a proposition
theorem this_is_a_big_tree_is_not_proposition : 
  ¬ is_proposition "This is a big tree" := by sorry

-- Main theorem combining all the above
theorem only_two_plus_three_equals_eight_is_proposition : 
  is_proposition "2+3=8" ∧ 
  ¬ is_proposition "x-1=0" ∧ 
  ¬ is_proposition "Do you speak English?" ∧ 
  ¬ is_proposition "This is a big tree" := by sorry

end NUMINAMATH_CALUDE_two_plus_three_equals_eight_is_proposition_x_minus_one_equals_zero_is_not_proposition_do_you_speak_english_is_not_proposition_this_is_a_big_tree_is_not_proposition_only_two_plus_three_equals_eight_is_proposition_l2036_203653


namespace NUMINAMATH_CALUDE_evaluate_expression_l2036_203679

theorem evaluate_expression : (122^2 - 115^2 + 7) / 14 = 119 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2036_203679


namespace NUMINAMATH_CALUDE_probability_top_given_not_female_is_one_eighth_l2036_203643

/-- Represents the probability of selecting a top student given the student is not female -/
def probability_top_given_not_female (total_students : ℕ) (female_students : ℕ) (top_fraction : ℚ) (top_female_fraction : ℚ) : ℚ :=
  let male_students := total_students - female_students
  let top_students := (total_students : ℚ) * top_fraction
  let male_top_students := top_students * (1 - top_female_fraction)
  male_top_students / male_students

/-- Theorem stating the probability of selecting a top student given the student is not female -/
theorem probability_top_given_not_female_is_one_eighth :
  probability_top_given_not_female 60 20 (1/6) (1/2) = 1/8 := by
  sorry


end NUMINAMATH_CALUDE_probability_top_given_not_female_is_one_eighth_l2036_203643


namespace NUMINAMATH_CALUDE_solution_ratio_l2036_203622

/-- Given a system of linear equations with a non-zero solution (x, y, z) and parameter k:
    x + k*y + 4*z = 0
    4*x + k*y + z = 0
    3*x + 5*y - 2*z = 0
    Prove that xz/y^2 = 25 -/
theorem solution_ratio (k x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (eq1 : x + k*y + 4*z = 0)
  (eq2 : 4*x + k*y + z = 0)
  (eq3 : 3*x + 5*y - 2*z = 0) :
  x*z / (y^2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_solution_ratio_l2036_203622


namespace NUMINAMATH_CALUDE_solution_set_l2036_203633

/-- An even function that is monotonically decreasing on [0,+∞) and f(1) = 0 -/
def f (x : ℝ) : ℝ := sorry

/-- f is an even function -/
axiom f_even : ∀ x, f x = f (-x)

/-- f is monotonically decreasing on [0,+∞) -/
axiom f_decreasing : ∀ x y, 0 ≤ x → x < y → f y < f x

/-- f(1) = 0 -/
axiom f_one_eq_zero : f 1 = 0

/-- The solution set of f(x-3) ≥ 0 is [2,4] -/
theorem solution_set : Set.Icc 2 4 = {x | f (x - 3) ≥ 0} := by sorry

end NUMINAMATH_CALUDE_solution_set_l2036_203633


namespace NUMINAMATH_CALUDE_midpoint_sum_invariant_l2036_203690

/-- A polygon in the Cartesian plane -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Create a new polygon by connecting midpoints of sides -/
def midpointPolygon (P : Polygon) : Polygon :=
  sorry

/-- Sum of x-coordinates of vertices -/
def sumXCoordinates (P : Polygon) : ℝ :=
  sorry

theorem midpoint_sum_invariant (Q1 : Polygon) 
  (h1 : Q1.vertices.length = 45)
  (h2 : sumXCoordinates Q1 = 135) : 
  let Q2 := midpointPolygon Q1
  let Q3 := midpointPolygon Q2
  sumXCoordinates Q3 = 135 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_sum_invariant_l2036_203690


namespace NUMINAMATH_CALUDE_perfect_square_in_base_n_l2036_203613

theorem perfect_square_in_base_n (n : ℕ) (hn : n ≥ 2) :
  ∃ m : ℕ, m^2 = n^4 + n^3 + n^2 + n + 1 ↔ n = 3 := by sorry

end NUMINAMATH_CALUDE_perfect_square_in_base_n_l2036_203613


namespace NUMINAMATH_CALUDE_det_sine_matrix_zero_l2036_203650

theorem det_sine_matrix_zero :
  let M : Matrix (Fin 3) (Fin 3) ℝ := λ i j ↦ 
    Real.sin (((i : ℕ) * 3 + (j : ℕ) + 2) : ℝ)
  Matrix.det M = 0 := by
  sorry

end NUMINAMATH_CALUDE_det_sine_matrix_zero_l2036_203650


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2036_203637

theorem cubic_root_sum (a b c : ℝ) : 
  0 < a ∧ a < 1 ∧
  0 < b ∧ b < 1 ∧
  0 < c ∧ c < 1 ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  24 * a^3 - 36 * a^2 + 14 * a - 1 = 0 ∧
  24 * b^3 - 36 * b^2 + 14 * b - 1 = 0 ∧
  24 * c^3 - 36 * c^2 + 14 * c - 1 = 0 →
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2036_203637


namespace NUMINAMATH_CALUDE_base_conversion_sum_l2036_203651

def base8_to_10 (n : ℕ) : ℕ := 2 * 8^2 + 5 * 8^1 + 4 * 8^0

def base2_to_10 (n : ℕ) : ℕ := 1 * 2^1 + 1 * 2^0

def base5_to_10 (n : ℕ) : ℕ := 1 * 5^2 + 4 * 5^1 + 4 * 5^0

def base4_to_10 (n : ℕ) : ℕ := 3 * 4^1 + 2 * 4^0

theorem base_conversion_sum :
  (base8_to_10 254 : ℚ) / (base2_to_10 11) + (base5_to_10 144 : ℚ) / (base4_to_10 32) = 57.4 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l2036_203651


namespace NUMINAMATH_CALUDE_bacteria_growth_time_l2036_203655

/-- The growth factor of bacteria population in one tripling period -/
def tripling_factor : ℕ := 3

/-- The duration in hours of one tripling period -/
def hours_per_tripling : ℕ := 3

/-- The initial number of bacteria -/
def initial_bacteria : ℕ := 300

/-- The final number of bacteria -/
def final_bacteria : ℕ := 72900

/-- The time in hours for bacteria to grow from initial to final count -/
def growth_time : ℕ := 15

theorem bacteria_growth_time :
  (tripling_factor ^ (growth_time / hours_per_tripling)) * initial_bacteria = final_bacteria :=
sorry

end NUMINAMATH_CALUDE_bacteria_growth_time_l2036_203655


namespace NUMINAMATH_CALUDE_glove_selection_theorem_l2036_203699

theorem glove_selection_theorem :
  let total_pairs : ℕ := 6
  let gloves_to_select : ℕ := 4
  let same_color_pair : ℕ := 1
  let ways_to_select_pair : ℕ := total_pairs.choose same_color_pair
  let remaining_gloves : ℕ := 2 * (total_pairs - same_color_pair)
  let ways_to_select_others : ℕ := remaining_gloves.choose (gloves_to_select - 2) - (total_pairs - same_color_pair)
  ways_to_select_pair * ways_to_select_others = 240
  := by sorry

end NUMINAMATH_CALUDE_glove_selection_theorem_l2036_203699


namespace NUMINAMATH_CALUDE_magician_earnings_l2036_203614

/-- Calculates the money earned from selling magic card decks -/
def money_earned (price_per_deck : ℕ) (initial_decks : ℕ) (final_decks : ℕ) : ℕ :=
  (initial_decks - final_decks) * price_per_deck

/-- Proves that the magician earned 56 dollars -/
theorem magician_earnings :
  let price_per_deck : ℕ := 7
  let initial_decks : ℕ := 16
  let final_decks : ℕ := 8
  money_earned price_per_deck initial_decks final_decks = 56 := by
  sorry

end NUMINAMATH_CALUDE_magician_earnings_l2036_203614


namespace NUMINAMATH_CALUDE_quadratic_function_k_value_l2036_203694

theorem quadratic_function_k_value (a b c k : ℤ) (f : ℝ → ℝ) :
  (∀ x, f x = a * x^2 + b * x + c) →
  f 1 = 0 →
  50 < f 7 ∧ f 7 < 60 →
  70 < f 8 ∧ f 8 < 80 →
  5000 * k < f 100 ∧ f 100 < 5000 * (k + 1) →
  k = 3 := by
sorry


end NUMINAMATH_CALUDE_quadratic_function_k_value_l2036_203694


namespace NUMINAMATH_CALUDE_parabola_symmetry_l2036_203665

/-- Given that M(0,5) and N(2,5) lie on the parabola y = 2(x-m)^2 + 3, prove that m = 1 -/
theorem parabola_symmetry (m : ℝ) : 
  (5 : ℝ) = 2 * (0 - m)^2 + 3 ∧ 
  (5 : ℝ) = 2 * (2 - m)^2 + 3 → 
  m = 1 := by sorry

end NUMINAMATH_CALUDE_parabola_symmetry_l2036_203665


namespace NUMINAMATH_CALUDE_age_difference_proof_l2036_203689

def zion_age : ℕ := 8

def dad_age : ℕ := 4 * zion_age + 3

def age_difference_after_10_years : ℕ :=
  (dad_age + 10) - (zion_age + 10)

theorem age_difference_proof :
  age_difference_after_10_years = 27 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l2036_203689


namespace NUMINAMATH_CALUDE_sector_circumference_l2036_203654

/-- Given a circular sector with area 2 and central angle 4 radians, 
    its circumference is 6. -/
theorem sector_circumference (area : ℝ) (angle : ℝ) (circumference : ℝ) : 
  area = 2 → angle = 4 → circumference = 6 := by
  sorry

end NUMINAMATH_CALUDE_sector_circumference_l2036_203654


namespace NUMINAMATH_CALUDE_decimal_multiplication_l2036_203601

theorem decimal_multiplication (a b : ℕ) (h : a * b = 19732) :
  (a : ℚ) / 100 * ((b : ℚ) / 100) = 1.9732 :=
by sorry

end NUMINAMATH_CALUDE_decimal_multiplication_l2036_203601


namespace NUMINAMATH_CALUDE_third_month_sale_calculation_l2036_203648

/-- Calculates the sale in the third month given the sales of other months and the average -/
def third_month_sale (first_month : ℕ) (second_month : ℕ) (fourth_month : ℕ) (average : ℕ) : ℕ :=
  4 * average - (first_month + second_month + fourth_month)

/-- Theorem stating the sale in the third month given the problem conditions -/
theorem third_month_sale_calculation :
  third_month_sale 2500 4000 1520 2890 = 3540 := by
  sorry

end NUMINAMATH_CALUDE_third_month_sale_calculation_l2036_203648


namespace NUMINAMATH_CALUDE_unique_k_for_prime_roots_l2036_203688

theorem unique_k_for_prime_roots : ∃! k : ℕ, 
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ 
  (p + q = 73) ∧ (p * q = k) ∧ 
  ∀ x : ℝ, x^2 - 73*x + k = 0 ↔ (x = p ∨ x = q) := by
  sorry

end NUMINAMATH_CALUDE_unique_k_for_prime_roots_l2036_203688


namespace NUMINAMATH_CALUDE_bb_tileable_iff_2b_divides_l2036_203668

/-- A rectangle is (b,b)-tileable if it can be covered by b×b square tiles --/
def is_bb_tileable (m n b : ℕ) : Prop :=
  ∃ (k l : ℕ), m = k * b ∧ n = l * b

/-- Main theorem: An m×n rectangle is (b,b)-tileable iff 2b divides both m and n --/
theorem bb_tileable_iff_2b_divides (m n b : ℕ) (hm : m > 0) (hn : n > 0) (hb : b > 0) :
  is_bb_tileable m n b ↔ (2 * b ∣ m) ∧ (2 * b ∣ n) :=
sorry

end NUMINAMATH_CALUDE_bb_tileable_iff_2b_divides_l2036_203668


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2036_203639

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x < 2}
def B : Set ℝ := {x | 1 < x ∧ x < Real.exp 2}

-- Define the complement of B
def C_R_B : Set ℝ := {x | x ≤ 1 ∨ Real.exp 2 ≤ x}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ C_R_B = {x | 0 < x ∧ x ≤ 1} :=
sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2036_203639


namespace NUMINAMATH_CALUDE_inscribed_square_area_l2036_203692

-- Define the parabola function
def parabola (x : ℝ) : ℝ := x^2 - 10*x + 21

-- Define the square
structure InscribedSquare where
  center : ℝ
  side_half : ℝ

-- Theorem statement
theorem inscribed_square_area :
  ∃ (s : InscribedSquare),
    s.center = 5 ∧
    parabola (s.center + s.side_half) = -2 * s.side_half ∧
    (2 * s.side_half)^2 = 24 - 8 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l2036_203692


namespace NUMINAMATH_CALUDE_cos_330_degrees_l2036_203663

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_degrees_l2036_203663


namespace NUMINAMATH_CALUDE_A_value_l2036_203617

noncomputable def A (m n : ℝ) : ℝ :=
  (((4 * m^2 * n^2) / (4 * m * n - m^2 - 4 * n^2) -
    (2 + n / m + m / n) / (4 / (m * n) - 1 / n^2 - 4 / m^2))^(1/2)) *
  (Real.sqrt (m * n) / (m - 2 * n))

theorem A_value (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  A m n = if 1 < m / n ∧ m / n < 2 then n - m else m - n := by
  sorry

end NUMINAMATH_CALUDE_A_value_l2036_203617


namespace NUMINAMATH_CALUDE_folded_square_area_l2036_203602

/-- The area of a shape formed by folding a square along its diagonal -/
theorem folded_square_area (side_length : ℝ) (h : side_length = 2) : 
  (side_length ^ 2) / 2 = 2 := by sorry

end NUMINAMATH_CALUDE_folded_square_area_l2036_203602


namespace NUMINAMATH_CALUDE_reggie_remaining_money_l2036_203642

/-- Calculates the remaining money after a purchase --/
def remaining_money (initial_amount number_of_items cost_per_item : ℕ) : ℕ :=
  initial_amount - (number_of_items * cost_per_item)

/-- Proves that Reggie has $38 left after his purchase --/
theorem reggie_remaining_money :
  remaining_money 48 5 2 = 38 := by
  sorry

end NUMINAMATH_CALUDE_reggie_remaining_money_l2036_203642


namespace NUMINAMATH_CALUDE_remainder_777_444_mod_11_l2036_203671

theorem remainder_777_444_mod_11 : 777^444 % 11 = 3 := by sorry

end NUMINAMATH_CALUDE_remainder_777_444_mod_11_l2036_203671


namespace NUMINAMATH_CALUDE_ice_cost_l2036_203611

theorem ice_cost (cost_two_bags : ℝ) (num_bags : ℕ) : 
  cost_two_bags = 1.46 → num_bags = 4 → num_bags * (cost_two_bags / 2) = 2.92 := by
  sorry

end NUMINAMATH_CALUDE_ice_cost_l2036_203611


namespace NUMINAMATH_CALUDE_euclid_schools_count_l2036_203687

theorem euclid_schools_count :
  ∀ (n : ℕ) (andrea_rank beth_rank carla_rank : ℕ),
    -- Each school sends 3 students
    -- Total number of students is 3n
    -- Andrea's score is the median
    andrea_rank = (3 * n + 1) / 2 →
    -- Andrea's score is highest on her team
    andrea_rank < beth_rank →
    andrea_rank < carla_rank →
    -- Beth and Carla's ranks
    beth_rank = 37 →
    carla_rank = 64 →
    -- Each participant received a different score
    andrea_rank ≠ beth_rank ∧ andrea_rank ≠ carla_rank ∧ beth_rank ≠ carla_rank →
    -- Prove that the number of schools is 23
    n = 23 := by
  sorry


end NUMINAMATH_CALUDE_euclid_schools_count_l2036_203687


namespace NUMINAMATH_CALUDE_rtl_eval_eq_standard_eval_l2036_203670

/-- Right-to-left grouping evaluation function -/
noncomputable def rtlEval (a b c d : ℝ) : ℝ := a * (b / (c + d^2))

/-- Standard algebraic notation evaluation function -/
noncomputable def standardEval (a b c d : ℝ) : ℝ := (a * b) / (c + d^2)

/-- Theorem stating the equivalence of right-to-left grouping and standard algebraic notation -/
theorem rtl_eval_eq_standard_eval (a b c d : ℝ) :
  rtlEval a b c d = standardEval a b c d :=
by sorry

end NUMINAMATH_CALUDE_rtl_eval_eq_standard_eval_l2036_203670


namespace NUMINAMATH_CALUDE_nine_ones_squared_l2036_203669

def nine_ones : ℕ := 111111111

theorem nine_ones_squared :
  nine_ones ^ 2 = 12345678987654321 := by sorry

end NUMINAMATH_CALUDE_nine_ones_squared_l2036_203669


namespace NUMINAMATH_CALUDE_original_number_proof_l2036_203672

theorem original_number_proof : ∃ (n : ℕ), n + 859560 ≡ 0 [MOD 456] ∧ n = 696 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2036_203672


namespace NUMINAMATH_CALUDE_fraction_equality_l2036_203646

theorem fraction_equality : (4 * 5) / 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2036_203646


namespace NUMINAMATH_CALUDE_min_value_expression_l2036_203626

theorem min_value_expression (a b c : ℕ+) (h : a + b + c = 12) :
  (4 * c : ℚ) / (a ^ 3 + b ^ 3) + (4 * a : ℚ) / (b ^ 3 + c ^ 3) + (b : ℚ) / (a ^ 3 + c ^ 3) ≥ 9 / 32 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2036_203626


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l2036_203686

/-- Time taken for a train to cross a bridge -/
theorem train_bridge_crossing_time
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (bridge_length : ℝ)
  (h1 : train_length = 110)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 265) :
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l2036_203686


namespace NUMINAMATH_CALUDE_incorrect_statement_l2036_203674

theorem incorrect_statement : ¬(∀ (p q : Prop), (¬(p ∧ q)) → (p ∨ q)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_l2036_203674


namespace NUMINAMATH_CALUDE_square_between_bounds_l2036_203604

theorem square_between_bounds (n : ℕ) (hn : n ≥ 16088121) :
  ∃ l : ℕ, n < l ^ 2 ∧ l ^ 2 < n * (1 + 1 / 2005) := by
  sorry

end NUMINAMATH_CALUDE_square_between_bounds_l2036_203604


namespace NUMINAMATH_CALUDE_other_number_l2036_203610

theorem other_number (x : ℝ) : x + 0.525 = 0.650 → x = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_other_number_l2036_203610


namespace NUMINAMATH_CALUDE_k_value_l2036_203636

/-- The length of an integer is the number of positive prime factors (not necessarily distinct) whose product equals the integer. -/
def length (k : ℕ) : ℕ := sorry

/-- k is an integer greater than 1 with a length of 4 and prime factors 2, 2, 2, and 3 -/
def k : ℕ := sorry

theorem k_value : k = 24 := by sorry

end NUMINAMATH_CALUDE_k_value_l2036_203636


namespace NUMINAMATH_CALUDE_kaylin_age_l2036_203600

-- Define variables for each person's age
variable (kaylin sarah eli freyja alfred olivia : ℝ)

-- State the conditions
axiom kaylin_sarah : kaylin = sarah - 5
axiom sarah_eli : sarah = 2 * eli
axiom eli_freyja : eli = freyja + 9
axiom freyja_alfred : freyja = 2.5 * alfred
axiom alfred_olivia : alfred = 0.75 * olivia
axiom freyja_age : freyja = 9.5

-- Theorem to prove
theorem kaylin_age : kaylin = 32 := by
  sorry

end NUMINAMATH_CALUDE_kaylin_age_l2036_203600


namespace NUMINAMATH_CALUDE_first_pipe_fill_time_l2036_203649

/-- The time it takes for the first pipe to fill the cistern -/
def T : ℝ := 10

/-- The time it takes for the second pipe to fill the cistern -/
def second_pipe_time : ℝ := 12

/-- The time it takes for the third pipe to empty the cistern -/
def third_pipe_time : ℝ := 25

/-- The time it takes to fill the cistern when all pipes are opened simultaneously -/
def combined_time : ℝ := 6.976744186046512

theorem first_pipe_fill_time :
  (1 / T + 1 / second_pipe_time - 1 / third_pipe_time) * combined_time = 1 :=
sorry

end NUMINAMATH_CALUDE_first_pipe_fill_time_l2036_203649


namespace NUMINAMATH_CALUDE_function_sum_l2036_203684

-- Define the properties of odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def IsEven (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- State the theorem
theorem function_sum (f g : ℝ → ℝ) 
  (h1 : IsOdd f) 
  (h2 : IsEven g) 
  (h3 : ∀ x, f x - g x = 2 * x - 3) : 
  ∀ x, f x + g x = 2 * x + 3 := by
sorry

end NUMINAMATH_CALUDE_function_sum_l2036_203684


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2036_203698

theorem complex_modulus_problem (z : ℂ) (h : z * Complex.I = 1 + 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2036_203698


namespace NUMINAMATH_CALUDE_thirteen_in_binary_l2036_203616

theorem thirteen_in_binary : 
  (13 : ℕ).digits 2 = [1, 0, 1, 1] :=
sorry

end NUMINAMATH_CALUDE_thirteen_in_binary_l2036_203616


namespace NUMINAMATH_CALUDE_expected_informed_after_pairing_l2036_203624

/-- Represents the scenario of scientists sharing news during a conference break -/
def ScientistNewsSharing (total : ℕ) (initial_informed : ℕ) : Prop :=
  total = 18 ∧ initial_informed = 10

/-- Calculates the expected number of scientists who know the news after pairing -/
noncomputable def expected_informed (total : ℕ) (initial_informed : ℕ) : ℝ :=
  initial_informed + (total - initial_informed) * (initial_informed / (total - 1))

/-- Theorem stating the expected number of informed scientists after pairing -/
theorem expected_informed_after_pairing {total initial_informed : ℕ} 
  (h : ScientistNewsSharing total initial_informed) :
  expected_informed total initial_informed = 14.7 := by
  sorry

end NUMINAMATH_CALUDE_expected_informed_after_pairing_l2036_203624


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l2036_203683

theorem quadratic_equation_properties (a b : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + a*x + b
  ∃ (p q : ℝ), p ≠ q ∧ 
  (f p = 0 ∧ f q = 0) ∧
  (
    (p = 3 ∧ p + q = 2 ∧ p * q < 0) ∨
    (f 1 = 0 ∧ p + q = 2 ∧ p * q < 0) ∨
    (f 1 = 0 ∧ p = 3 ∧ p * q < 0) ∨
    (f 1 = 0 ∧ p = 3 ∧ p + q = 2)
  ) →
  f 1 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l2036_203683


namespace NUMINAMATH_CALUDE_line_slope_is_one_l2036_203630

/-- The slope of a line in the xy-plane with y-intercept -2 and passing through 
    the midpoint of the line segment with endpoints (2, 8) and (8, -2) is 1. -/
theorem line_slope_is_one : 
  ∀ (m : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ m → y = x - 2) →  -- y-intercept is -2
    ((5 : ℝ), 3) ∈ m →  -- passes through midpoint (5, 3)
    (∃ (k : ℝ), ∀ (x y : ℝ), (x, y) ∈ m → y = k * x - 2) →  -- line equation
    (∀ (x y : ℝ), (x, y) ∈ m → y = x - 2) :=  -- slope is 1
by sorry

end NUMINAMATH_CALUDE_line_slope_is_one_l2036_203630


namespace NUMINAMATH_CALUDE_day_300_is_tuesday_l2036_203605

/-- If the 26th day of a 366-day year falls on a Monday, then the 300th day of that year falls on a Tuesday. -/
theorem day_300_is_tuesday (year_length : ℕ) (day_26_weekday : ℕ) :
  year_length = 366 →
  day_26_weekday = 1 →
  (300 - 26) % 7 + day_26_weekday ≡ 2 [MOD 7] :=
by sorry

end NUMINAMATH_CALUDE_day_300_is_tuesday_l2036_203605


namespace NUMINAMATH_CALUDE_m_range_theorem_l2036_203693

def is_ellipse_with_y_foci (m : ℝ) : Prop :=
  0 < m ∧ m < 3

def hyperbola_eccentricity_in_range (m : ℝ) : Prop :=
  m > 0 ∧ Real.sqrt 1.5 < (1 + m/5).sqrt ∧ (1 + m/5).sqrt < Real.sqrt 2

def p (m : ℝ) : Prop := is_ellipse_with_y_foci m
def q (m : ℝ) : Prop := hyperbola_eccentricity_in_range m

theorem m_range_theorem (m : ℝ) :
  (0 < m ∧ m < 9) →
  ((p m ∨ q m) ∧ ¬(p m ∧ q m)) →
  ((0 < m ∧ m ≤ 5/2) ∨ (3 ≤ m ∧ m < 5)) :=
by
  sorry

end NUMINAMATH_CALUDE_m_range_theorem_l2036_203693


namespace NUMINAMATH_CALUDE_tetrahedron_center_of_mass_l2036_203691

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  vertices : Fin 4 → Point3D

/-- The centroid of a tetrahedron -/
def centroid (t : Tetrahedron) : Point3D := sorry

/-- The circumcenter of a tetrahedron -/
def circumcenter (t : Tetrahedron) : Point3D := sorry

/-- The orthocenter of a tetrahedron -/
def orthocenter (t : Tetrahedron) : Point3D := sorry

/-- Checks if three points are collinear -/
def collinear (p1 p2 p3 : Point3D) : Prop := sorry

/-- Checks if a point is the midpoint of a line segment -/
def is_midpoint (m p1 p2 : Point3D) : Prop := sorry

/-- Calculates the center of mass given masses and their positions -/
def center_of_mass (masses : List ℝ) (positions : List Point3D) : Point3D := sorry

/-- Main theorem -/
theorem tetrahedron_center_of_mass (t : Tetrahedron) :
  let s := centroid t
  let o := circumcenter t
  let m := orthocenter t
  collinear s o m ∧ is_midpoint s o m →
  center_of_mass 
    [1, 1, 1, 1, -2] 
    [t.vertices 0, t.vertices 1, t.vertices 2, t.vertices 3, m] = o := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_center_of_mass_l2036_203691


namespace NUMINAMATH_CALUDE_inequality_proof_l2036_203682

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z > 0) :
  (x^2 * y)/(y+z) + (y^2 * z)/(z+x) + (z^2 * x)/(x+y) ≥ (1/2) * (x^2 + y^2 + z^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2036_203682


namespace NUMINAMATH_CALUDE_inequality_proof_l2036_203606

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) :
  |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2036_203606


namespace NUMINAMATH_CALUDE_mudits_age_l2036_203635

/-- Mudit's present age satisfies the given condition -/
theorem mudits_age : ∃ (x : ℕ), (x + 16 = 3 * (x - 4)) ∧ (x = 14) := by
  sorry

end NUMINAMATH_CALUDE_mudits_age_l2036_203635


namespace NUMINAMATH_CALUDE_double_reflection_result_l2036_203620

def reflect_over_y_equals_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def double_reflection (p : ℝ × ℝ) : ℝ × ℝ :=
  reflect_over_y_axis (reflect_over_y_equals_x p)

theorem double_reflection_result :
  double_reflection (7, -3) = (3, 7) := by sorry

end NUMINAMATH_CALUDE_double_reflection_result_l2036_203620


namespace NUMINAMATH_CALUDE_max_ballpoint_pens_l2036_203657

/-- Represents the number of pens of each type -/
structure PenCounts where
  ballpoint : ℕ
  gel : ℕ
  fountain : ℕ

/-- Checks if the given pen counts satisfy all conditions -/
def satisfiesConditions (counts : PenCounts) : Prop :=
  counts.ballpoint + counts.gel + counts.fountain = 15 ∧
  counts.ballpoint ≥ 1 ∧ counts.gel ≥ 1 ∧ counts.fountain ≥ 1 ∧
  10 * counts.ballpoint + 40 * counts.gel + 60 * counts.fountain = 500

/-- Theorem stating that the maximum number of ballpoint pens is 6 -/
theorem max_ballpoint_pens : 
  (∃ counts : PenCounts, satisfiesConditions counts) →
  (∀ counts : PenCounts, satisfiesConditions counts → counts.ballpoint ≤ 6) ∧
  (∃ counts : PenCounts, satisfiesConditions counts ∧ counts.ballpoint = 6) :=
by sorry

end NUMINAMATH_CALUDE_max_ballpoint_pens_l2036_203657


namespace NUMINAMATH_CALUDE_bookstore_sales_l2036_203656

theorem bookstore_sales (wednesday_sales : ℕ) (thursday_sales : ℕ) (friday_sales : ℕ) : 
  wednesday_sales = 15 →
  thursday_sales = 3 * wednesday_sales →
  friday_sales = thursday_sales / 5 →
  wednesday_sales + thursday_sales + friday_sales = 69 := by
sorry

end NUMINAMATH_CALUDE_bookstore_sales_l2036_203656


namespace NUMINAMATH_CALUDE_computer_factory_earnings_l2036_203641

/-- Calculates the earnings from selling computers produced in a week -/
def weekly_earnings (daily_production : ℕ) (price_per_unit : ℕ) : ℕ :=
  daily_production * 7 * price_per_unit

/-- Proves that the weekly earnings for the given conditions equal $1,575,000 -/
theorem computer_factory_earnings :
  weekly_earnings 1500 150 = 1575000 := by
  sorry

end NUMINAMATH_CALUDE_computer_factory_earnings_l2036_203641


namespace NUMINAMATH_CALUDE_sqrt_a_div_sqrt_b_equals_five_halves_l2036_203623

theorem sqrt_a_div_sqrt_b_equals_five_halves (a b : ℝ) 
  (h : (1/3)^2 + (1/4)^2 = (25*a / 61*b) * ((1/5)^2 + (1/6)^2)) : 
  Real.sqrt a / Real.sqrt b = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_a_div_sqrt_b_equals_five_halves_l2036_203623


namespace NUMINAMATH_CALUDE_regular_polygon_center_containment_l2036_203673

/-- A regular polygon with 2n sides -/
structure RegularPolygon (n : ℕ) where
  sideLength : ℝ
  center : ℝ × ℝ

/-- Predicate to check if one polygon is inside another -/
def isInside (p1 p2 : RegularPolygon n) : Prop := sorry

/-- Predicate to check if a point is inside a polygon -/
def containsPoint (p : RegularPolygon n) (point : ℝ × ℝ) : Prop := sorry

theorem regular_polygon_center_containment (n : ℕ) (a : ℝ) 
  (M₁ : RegularPolygon n) (M₂ : RegularPolygon n) 
  (h1 : M₁.sideLength = a) 
  (h2 : M₂.sideLength = 2 * a) 
  (h3 : isInside M₁ M₂) :
  containsPoint M₁ M₂.center := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_center_containment_l2036_203673


namespace NUMINAMATH_CALUDE_equation_solution_l2036_203681

theorem equation_solution : 
  {x : ℝ | (Real.sqrt (9*x - 2) + 15 / Real.sqrt (9*x - 2) = 8)} = {3, 11/9} :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2036_203681


namespace NUMINAMATH_CALUDE_photo_arrangements_l2036_203685

/-- The number of different arrangements of 5 students and 2 teachers in a row,
    where exactly two students stand between the two teachers. -/
def arrangements_count : ℕ := 960

/-- The number of students -/
def num_students : ℕ := 5

/-- The number of teachers -/
def num_teachers : ℕ := 2

/-- The number of students required to stand between teachers -/
def students_between : ℕ := 2

theorem photo_arrangements :
  arrangements_count = 960 :=
sorry

end NUMINAMATH_CALUDE_photo_arrangements_l2036_203685


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2036_203676

theorem inequality_and_equality_condition (x y z : ℝ) (h : x^2 + y^2 + z^2 = 3) :
  x^3 - (y^2 + y*z + z^2)*x + y^2*z + y*z^2 ≤ 3 * Real.sqrt 3 ∧
  (x^3 - (y^2 + y*z + z^2)*x + y^2*z + y*z^2 = 3 * Real.sqrt 3 ↔
    ((x = Real.sqrt 3 ∧ y = 0 ∧ z = 0) ∨
     (x = -Real.sqrt 3 / 3 ∧ y = 2 * Real.sqrt 3 / 3 ∧ z = 2 * Real.sqrt 3 / 3))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2036_203676


namespace NUMINAMATH_CALUDE_a_5_equals_17_l2036_203612

theorem a_5_equals_17 (a : ℕ → ℤ) (h : ∀ n, a n = 4 * n - 3) : a 5 = 17 := by
  sorry

end NUMINAMATH_CALUDE_a_5_equals_17_l2036_203612


namespace NUMINAMATH_CALUDE_probability_not_red_marble_l2036_203678

theorem probability_not_red_marble (total : ℕ) (red green yellow blue : ℕ) 
  (h1 : total = red + green + yellow + blue)
  (h2 : red = 8)
  (h3 : green = 10)
  (h4 : yellow = 12)
  (h5 : blue = 15) :
  (green + yellow + blue : ℚ) / total = 37 / 45 := by
sorry

end NUMINAMATH_CALUDE_probability_not_red_marble_l2036_203678


namespace NUMINAMATH_CALUDE_team_b_score_l2036_203666

/-- Given a trivia game where:
  * Team A scored 2 points
  * Team C scored 4 points
  * The total points scored by all teams is 15
  Prove that Team B scored 9 points -/
theorem team_b_score (team_a_score team_c_score total_score : ℕ)
  (h1 : team_a_score = 2)
  (h2 : team_c_score = 4)
  (h3 : total_score = 15) :
  total_score - (team_a_score + team_c_score) = 9 := by
  sorry

end NUMINAMATH_CALUDE_team_b_score_l2036_203666


namespace NUMINAMATH_CALUDE_absolute_value_equals_self_implies_nonnegative_l2036_203659

theorem absolute_value_equals_self_implies_nonnegative (a : ℝ) : (|a| = a) → a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equals_self_implies_nonnegative_l2036_203659


namespace NUMINAMATH_CALUDE_relative_rate_of_change_cubic_parabola_l2036_203652

/-- For a point (x, y) on the cubic parabola 12y = x^3, the relative rate of change between y and x is x^2/4 -/
theorem relative_rate_of_change_cubic_parabola (x y : ℝ) (h : 12 * y = x^3) :
  ∃ (dx dy : ℝ), dy / dx = x^2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_relative_rate_of_change_cubic_parabola_l2036_203652


namespace NUMINAMATH_CALUDE_triangle_properties_l2036_203638

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  2 * t.b * Real.cos t.A + t.a = 2 * t.c ∧
  t.c = 8 ∧
  Real.sin t.A = (3 * Real.sqrt 3) / 14

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h : triangle_conditions t) : 
  t.B = π / 3 ∧ 
  (1 / 2 * t.a * t.c * Real.sin t.B) = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2036_203638


namespace NUMINAMATH_CALUDE_triangle_properties_l2036_203675

theorem triangle_properties (a b c A B C : ℝ) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  Real.sqrt 3 * b = 2 * a * Real.sin B * Real.cos C + 2 * c * Real.sin B * Real.cos A →
  a = 3 →
  c = 4 →
  B = π/3 ∧ b = Real.sqrt 13 ∧ Real.cos (2 * A + B) = -23/26 := by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_triangle_properties_l2036_203675


namespace NUMINAMATH_CALUDE_range_of_f_l2036_203660

def f (x : ℝ) := |x + 5| - |x - 3|

theorem range_of_f :
  Set.range f = Set.Iic 14 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l2036_203660


namespace NUMINAMATH_CALUDE_even_digit_sum_pairs_count_l2036_203619

/-- Given a natural number, returns true if its digit sum is even -/
def has_even_digit_sum (n : ℕ) : Bool :=
  sorry

/-- Returns the count of natural numbers less than 10^6 where both
    the number and its successor have even digit sums -/
def count_even_digit_sum_pairs : ℕ :=
  sorry

/-- The main theorem stating that the count of natural numbers less than 10^6
    where both the number and its successor have even digit sums is 45454 -/
theorem even_digit_sum_pairs_count :
  count_even_digit_sum_pairs = 45454 := by sorry

end NUMINAMATH_CALUDE_even_digit_sum_pairs_count_l2036_203619


namespace NUMINAMATH_CALUDE_certain_number_proof_l2036_203615

theorem certain_number_proof (given_division : 7125 / 1.25 = 5700) 
  (certain_number : ℝ) (certain_division : certain_number / 12.5 = 57) : 
  certain_number = 712.5 := by
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2036_203615


namespace NUMINAMATH_CALUDE_smallest_class_size_l2036_203627

theorem smallest_class_size : ∃ (x : ℕ), 
  x > 0 ∧ 
  5 * x + 2 > 40 ∧ 
  ∀ (y : ℕ), y > 0 → 5 * y + 2 > 40 → y ≥ x →
  5 * x + 2 = 42 :=
by sorry

end NUMINAMATH_CALUDE_smallest_class_size_l2036_203627


namespace NUMINAMATH_CALUDE_sine_theorem_trihedral_angle_first_cosine_theorem_trihedral_angle_second_cosine_theorem_trihedral_angle_l2036_203662

/-- A trihedral angle with face angles α, β, γ and opposite dihedral angles A, B, C. -/
structure TrihedralAngle where
  α : Real
  β : Real
  γ : Real
  A : Real
  B : Real
  C : Real

/-- The sine theorem for a trihedral angle holds. -/
theorem sine_theorem_trihedral_angle (t : TrihedralAngle) :
  (Real.sin t.α) / (Real.sin t.A) = (Real.sin t.β) / (Real.sin t.B) ∧
  (Real.sin t.β) / (Real.sin t.B) = (Real.sin t.γ) / (Real.sin t.C) :=
sorry

/-- The first cosine theorem for a trihedral angle holds. -/
theorem first_cosine_theorem_trihedral_angle (t : TrihedralAngle) :
  Real.cos t.α = Real.cos t.β * Real.cos t.γ + Real.sin t.β * Real.sin t.γ * Real.cos t.A :=
sorry

/-- The second cosine theorem for a trihedral angle holds. -/
theorem second_cosine_theorem_trihedral_angle (t : TrihedralAngle) :
  Real.cos t.A = -Real.cos t.B * Real.cos t.C + Real.sin t.B * Real.sin t.C * Real.cos t.α :=
sorry

end NUMINAMATH_CALUDE_sine_theorem_trihedral_angle_first_cosine_theorem_trihedral_angle_second_cosine_theorem_trihedral_angle_l2036_203662


namespace NUMINAMATH_CALUDE_compare_large_exponents_l2036_203667

theorem compare_large_exponents : 1997^(1998^1999) > 1999^(1998^1997) := by
  sorry

end NUMINAMATH_CALUDE_compare_large_exponents_l2036_203667


namespace NUMINAMATH_CALUDE_bella_steps_to_meet_ella_l2036_203621

/-- The distance between Bella's and Ella's houses in feet -/
def total_distance : ℕ := 15840

/-- The number of feet Bella covers in one step -/
def feet_per_step : ℕ := 3

/-- Ella's speed relative to Bella's -/
def speed_ratio : ℕ := 3

/-- The number of steps Bella takes before meeting Ella -/
def steps_taken : ℕ := 1320

theorem bella_steps_to_meet_ella :
  total_distance * speed_ratio = steps_taken * feet_per_step * (speed_ratio + 1) :=
sorry

end NUMINAMATH_CALUDE_bella_steps_to_meet_ella_l2036_203621


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2036_203695

theorem right_triangle_hypotenuse (leg : ℝ) (h : leg = 8) :
  let hypotenuse := leg * Real.sqrt 2
  hypotenuse = 8 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2036_203695


namespace NUMINAMATH_CALUDE_range_of_cubic_sum_l2036_203645

theorem range_of_cubic_sum (a b : ℝ) (h : a^2 + b^2 = a + b) :
  0 ≤ a^3 + b^3 ∧ a^3 + b^3 ≤ 2 := by sorry

end NUMINAMATH_CALUDE_range_of_cubic_sum_l2036_203645


namespace NUMINAMATH_CALUDE_total_soap_cost_two_years_l2036_203609

/-- Represents the types of soap -/
inductive SoapType
  | Lavender
  | Lemon
  | Sandalwood

/-- Represents the price of each soap type -/
def soapPrice (t : SoapType) : ℚ :=
  match t with
  | SoapType.Lavender => 4
  | SoapType.Lemon => 5
  | SoapType.Sandalwood => 6

/-- Represents the bulk discount for each soap type and quantity -/
def bulkDiscount (t : SoapType) (quantity : ℕ) : ℚ :=
  match t with
  | SoapType.Lavender =>
    if quantity ≥ 10 then 0.2
    else if quantity ≥ 5 then 0.1
    else 0
  | SoapType.Lemon =>
    if quantity ≥ 8 then 0.15
    else if quantity ≥ 4 then 0.05
    else 0
  | SoapType.Sandalwood =>
    if quantity ≥ 9 then 0.2
    else if quantity ≥ 6 then 0.1
    else if quantity ≥ 3 then 0.05
    else 0

/-- Calculates the cost of soap for a given type and quantity with bulk discount -/
def soapCost (t : SoapType) (quantity : ℕ) : ℚ :=
  let price := soapPrice t
  let discount := bulkDiscount t quantity
  quantity * price * (1 - discount)

/-- Theorem: The total amount Elias spends on soap in two years is $112.4 -/
theorem total_soap_cost_two_years :
  soapCost SoapType.Lavender 5 + soapCost SoapType.Lavender 3 +
  soapCost SoapType.Lemon 4 + soapCost SoapType.Lemon 4 +
  soapCost SoapType.Sandalwood 6 + soapCost SoapType.Sandalwood 2 = 112.4 := by
  sorry


end NUMINAMATH_CALUDE_total_soap_cost_two_years_l2036_203609


namespace NUMINAMATH_CALUDE_interest_equality_implies_second_sum_l2036_203608

/-- Given a total sum of 2769 divided into two parts, if the interest on the first part
    for 8 years at 3% per annum equals the interest on the second part for 3 years at 5% per annum,
    then the second part is 1704. -/
theorem interest_equality_implies_second_sum (total : ℝ) (first_part second_part : ℝ) :
  total = 2769 →
  first_part + second_part = total →
  (first_part * 3 * 8) / 100 = (second_part * 5 * 3) / 100 →
  second_part = 1704 :=
by sorry

end NUMINAMATH_CALUDE_interest_equality_implies_second_sum_l2036_203608


namespace NUMINAMATH_CALUDE_value_of_N_l2036_203661

theorem value_of_N : ∃ N : ℝ, (0.25 * N = 0.55 * 5000) ∧ (N = 11000) := by
  sorry

end NUMINAMATH_CALUDE_value_of_N_l2036_203661


namespace NUMINAMATH_CALUDE_relationship_proof_l2036_203629

open Real

noncomputable def f (x : ℝ) := Real.exp x + x - 2
noncomputable def g (x : ℝ) := Real.log x + x^2 - 3

theorem relationship_proof (a b : ℝ) (ha : f a = 0) (hb : g b = 0) :
  g a < 0 ∧ 0 < f b := by sorry

end NUMINAMATH_CALUDE_relationship_proof_l2036_203629


namespace NUMINAMATH_CALUDE_new_average_production_l2036_203625

theorem new_average_production (n : ℕ) (past_average : ℝ) (today_production : ℝ) :
  n = 1 ∧ past_average = 50 ∧ today_production = 60 →
  (n * past_average + today_production) / (n + 1) = 55 := by
sorry

end NUMINAMATH_CALUDE_new_average_production_l2036_203625


namespace NUMINAMATH_CALUDE_second_largest_number_l2036_203631

theorem second_largest_number (A B C D : ℕ) : 
  A = 3 * 3 →
  C = 4 * A →
  B = C - 15 →
  D = A + 19 →
  (C > D ∧ D > B ∧ B > A) :=
by sorry

end NUMINAMATH_CALUDE_second_largest_number_l2036_203631


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2036_203634

theorem algebraic_expression_value : 
  let a : ℝ := Real.sqrt 5 + 1
  (a^2 - 2*a + 7) = 11 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2036_203634


namespace NUMINAMATH_CALUDE_cos_alpha_plus_pi_fourth_l2036_203677

theorem cos_alpha_plus_pi_fourth (α β : Real) : 
  (3 * Real.pi / 4 < α) ∧ (α < Real.pi) ∧
  (3 * Real.pi / 4 < β) ∧ (β < Real.pi) ∧
  (Real.sin (α + β) = -4/5) ∧
  (Real.sin (β - Real.pi/4) = 12/13) →
  Real.cos (α + Real.pi/4) = -63/65 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_pi_fourth_l2036_203677


namespace NUMINAMATH_CALUDE_system_solution_l2036_203696

theorem system_solution : ∃! (x y : ℝ), x - y = 2 ∧ 2*x + y = 7 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2036_203696


namespace NUMINAMATH_CALUDE_eventual_habitable_fraction_l2036_203644

-- Define the fraction of earth's surface not covered by water
def landFraction : ℚ := 1 / 3

-- Define the fraction of exposed land initially inhabitable
def initialInhabitableFraction : ℚ := 1 / 3

-- Define the additional fraction of non-inhabitable land made viable by technology
def techAdvancementFraction : ℚ := 1 / 2

-- Theorem statement
theorem eventual_habitable_fraction :
  let initialHabitableLand := landFraction * initialInhabitableFraction
  let additionalHabitableLand := landFraction * (1 - initialInhabitableFraction) * techAdvancementFraction
  initialHabitableLand + additionalHabitableLand = 2 / 9 :=
by sorry

end NUMINAMATH_CALUDE_eventual_habitable_fraction_l2036_203644


namespace NUMINAMATH_CALUDE_derived_point_relation_find_original_point_translated_derived_point_l2036_203603

/-- Definition of an a-th order derived point -/
def derived_point (a : ℝ) (P : ℝ × ℝ) : ℝ × ℝ :=
  (a * P.1 + P.2, P.1 + a * P.2)

/-- Theorem stating the relationship between a point and its a-th order derived point -/
theorem derived_point_relation (a : ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) :
  a ≠ 0 → Q = derived_point a P ↔ 
    Q.1 = a * P.1 + P.2 ∧ Q.2 = P.1 + a * P.2 := by
  sorry

/-- Theorem for finding the original point given its a-th order derived point -/
theorem find_original_point (a : ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) :
  a ≠ 0 ∧ a ≠ 1 → Q = derived_point a P →
    P = ((a * Q.2 - Q.1) / (a * a - 1), (a * Q.1 - Q.2) / (a * a - 1)) := by
  sorry

/-- Theorem for the composition of translation and derived point transformation -/
theorem translated_derived_point (a c : ℝ) :
  let P : ℝ × ℝ := (c + 1, 2 * c - 1)
  let P₁ : ℝ × ℝ := (c - 1, 2 * c)
  let P₂ : ℝ × ℝ := derived_point (-3) P₁
  (P₂.1 = 0 ∨ P₂.2 = 0) →
    (P₂ = (0, -16) ∨ P₂ = (16/5, 0)) := by
  sorry

end NUMINAMATH_CALUDE_derived_point_relation_find_original_point_translated_derived_point_l2036_203603


namespace NUMINAMATH_CALUDE_final_result_l2036_203697

def chosen_number : ℕ := 122
def multiplier : ℕ := 2
def subtractor : ℕ := 138

theorem final_result :
  chosen_number * multiplier - subtractor = 106 := by
  sorry

end NUMINAMATH_CALUDE_final_result_l2036_203697


namespace NUMINAMATH_CALUDE_square_roots_problem_l2036_203680

theorem square_roots_problem (x : ℝ) (h1 : x > 0) 
  (h2 : ∃ a : ℝ, (2 - a)^2 = x ∧ (2*a + 1)^2 = x) :
  ∃ a : ℝ, a = -3 ∧ (17 - x)^(1/3 : ℝ) = -2 := by
sorry

end NUMINAMATH_CALUDE_square_roots_problem_l2036_203680


namespace NUMINAMATH_CALUDE_parallel_transitivity_l2036_203647

-- Define a type for planes
variable (Plane : Type)

-- Define a relation for parallel planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem parallel_transitivity (p1 p2 p3 : Plane) :
  parallel p1 p3 → parallel p2 p3 → parallel p1 p2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l2036_203647


namespace NUMINAMATH_CALUDE_abc_inequalities_l2036_203664

theorem abc_inequalities (a b : Real) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a + b = 1) : 
  (2 * a^2 + b ≥ 7/8) ∧ 
  (a * b ≤ 1/4) ∧ 
  (Real.sqrt a + Real.sqrt b ≤ Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequalities_l2036_203664


namespace NUMINAMATH_CALUDE_coin_jar_theorem_l2036_203628

/-- Represents the number of coins added or removed in each hour. 
    Positive numbers represent additions, negative numbers represent removals. -/
def coin_changes : List Int := [20, 30, 30, 40, -20, 50, 60, -15, 70, -25]

/-- The total number of hours -/
def total_hours : Nat := 10

/-- Calculates the final number of coins in the jar -/
def final_coin_count (changes : List Int) : Int :=
  changes.sum

/-- Theorem stating that the final number of coins in the jar is 240 -/
theorem coin_jar_theorem : 
  final_coin_count coin_changes = 240 := by
  sorry

end NUMINAMATH_CALUDE_coin_jar_theorem_l2036_203628
