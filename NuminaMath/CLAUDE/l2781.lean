import Mathlib

namespace NUMINAMATH_CALUDE_triangle_side_length_l2781_278192

theorem triangle_side_length (D E F : ℝ) : 
  -- Triangle DEF exists
  (0 < D) → (0 < E) → (0 < F) → 
  (D + E > F) → (D + F > E) → (E + F > D) →
  -- Given conditions
  (E = 45 * π / 180) →  -- Convert 45° to radians
  (D = 100) →
  (F = 100 * Real.sqrt 2) →
  -- Conclusion
  (E = Real.sqrt (30000 + 5000 * (Real.sqrt 6 - Real.sqrt 2))) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2781_278192


namespace NUMINAMATH_CALUDE_system_solutions_l2781_278118

/-- The system of equations has only three real solutions -/
theorem system_solutions (a b c : ℝ) : 
  (2 * a - b = a^2 * b) ∧ 
  (2 * b - c = b^2 * c) ∧ 
  (2 * c - a = c^2 * a) → 
  ((a = -1 ∧ b = -1 ∧ c = -1) ∨ 
   (a = 0 ∧ b = 0 ∧ c = 0) ∨ 
   (a = 1 ∧ b = 1 ∧ c = 1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l2781_278118


namespace NUMINAMATH_CALUDE_sum_of_solutions_l2781_278151

theorem sum_of_solutions (a b : ℝ) (ha : a > 1) (hb : b > 0) :
  let f : ℝ → ℝ := λ x => Real.sqrt (a - Real.sqrt (a + b^x))
  ∃ x : ℝ, f x = x ∧
  (∀ y : ℝ, f y = y → y ≤ x) ∧
  x = (Real.sqrt (4 * a - 3 * b) - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l2781_278151


namespace NUMINAMATH_CALUDE_factor_expression_l2781_278131

theorem factor_expression (x : ℝ) : 75 * x^12 + 225 * x^24 = 75 * x^12 * (1 + 3 * x^12) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2781_278131


namespace NUMINAMATH_CALUDE_no_real_roots_l2781_278175

theorem no_real_roots (a b : ℝ) (h1 : b/a > 1/4) (h2 : a > 0) :
  ∀ x : ℝ, x/a + b/x ≠ 1 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_l2781_278175


namespace NUMINAMATH_CALUDE_complex_product_pure_imaginary_l2781_278157

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_product_pure_imaginary (b : ℝ) :
  is_pure_imaginary ((1 + b * Complex.I) * (2 + Complex.I)) → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_pure_imaginary_l2781_278157


namespace NUMINAMATH_CALUDE_eighth_term_of_sequence_l2781_278100

theorem eighth_term_of_sequence (x : ℝ) : 
  let nth_term (n : ℕ) := (-1)^(n+1) * ((n^2 + 1) : ℝ) * x^n
  nth_term 8 = -65 * x^8 := by sorry

end NUMINAMATH_CALUDE_eighth_term_of_sequence_l2781_278100


namespace NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l2781_278195

/-- Given a cube with space diagonal 5√3, prove its volume is 125 -/
theorem cube_volume_from_space_diagonal :
  ∀ s : ℝ,
  s > 0 →
  (s * s * s = 5 * 5 * 5) →
  (s * s + s * s + s * s = (5 * Real.sqrt 3) * (5 * Real.sqrt 3)) →
  s * s * s = 125 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l2781_278195


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l2781_278106

/-- The parabola P with equation y = x^2 -/
def P : Set (ℝ × ℝ) := {p | p.2 = p.1^2}

/-- The point Q -/
def Q : ℝ × ℝ := (10, 6)

/-- The line through Q with slope n -/
def line (n : ℝ) : Set (ℝ × ℝ) := {p | p.2 - Q.2 = n * (p.1 - Q.1)}

/-- The condition for non-intersection -/
def non_intersect (n : ℝ) : Prop := line n ∩ P = ∅

/-- The theorem to be proved -/
theorem parabola_line_intersection :
  ∃ (a b : ℝ), (∀ n, non_intersect n ↔ a < n ∧ n < b) → a + b = 40 := by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l2781_278106


namespace NUMINAMATH_CALUDE_suraya_kayla_difference_l2781_278136

/-- The number of apples picked by each person --/
structure ApplePickers where
  suraya : ℕ
  caleb : ℕ
  kayla : ℕ

/-- The conditions of the apple-picking scenario --/
def apple_picking_scenario (p : ApplePickers) : Prop :=
  p.suraya = p.caleb + 12 ∧
  p.caleb + 5 = p.kayla ∧
  p.kayla = 20

/-- The theorem stating the difference between Suraya's and Kayla's apple count --/
theorem suraya_kayla_difference (p : ApplePickers) 
  (h : apple_picking_scenario p) : p.suraya - p.kayla = 7 := by
  sorry

end NUMINAMATH_CALUDE_suraya_kayla_difference_l2781_278136


namespace NUMINAMATH_CALUDE_sum_of_number_and_five_is_nine_l2781_278160

theorem sum_of_number_and_five_is_nine (x : ℤ) : x + 5 = 9 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_number_and_five_is_nine_l2781_278160


namespace NUMINAMATH_CALUDE_max_min_values_l2781_278196

theorem max_min_values (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 5 * y = 20) :
  (∃ (u : ℝ), u = Real.log x / Real.log 10 + Real.log y / Real.log 10 ∧
    u ≤ 1 ∧
    ∀ (v : ℝ), v = Real.log x / Real.log 10 + Real.log y / Real.log 10 → v ≤ u) ∧
  (∃ (w : ℝ), w = 1 / x + 1 / y ∧
    w ≥ (7 + 2 * Real.sqrt 10) / 20 ∧
    ∀ (z : ℝ), z = 1 / x + 1 / y → z ≥ w) := by
  sorry

end NUMINAMATH_CALUDE_max_min_values_l2781_278196


namespace NUMINAMATH_CALUDE_right_triangle_area_l2781_278199

/-- A right triangle with hypotenuse 13 and one leg 5 has an area of 30 -/
theorem right_triangle_area (a b c : ℝ) (h1 : a = 13) (h2 : b = 5) 
  (h3 : c^2 = a^2 - b^2) (h4 : a > 0 ∧ b > 0 ∧ c > 0) : (1/2) * b * c = 30 := by
  sorry

#check right_triangle_area

end NUMINAMATH_CALUDE_right_triangle_area_l2781_278199


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l2781_278116

/-- Represents a geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_general_term
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_sum1 : a 1 + a 3 = 5/2)
  (h_sum2 : a 2 + a 4 = 5/4) :
  ∀ n : ℕ, a n = 2^(2-n) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l2781_278116


namespace NUMINAMATH_CALUDE_total_marbles_l2781_278127

theorem total_marbles (jars : ℕ) (clay_pots : ℕ) (marbles_per_jar : ℕ) :
  jars = 16 →
  jars = 2 * clay_pots →
  marbles_per_jar = 5 →
  jars * marbles_per_jar + clay_pots * (3 * marbles_per_jar) = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_total_marbles_l2781_278127


namespace NUMINAMATH_CALUDE_original_speed_correct_l2781_278170

/-- The original speed of the car traveling between two locations. -/
def original_speed : ℝ := 80

/-- The distance between location A and location B in kilometers. -/
def distance : ℝ := 160

/-- The increase in speed as a percentage. -/
def speed_increase : ℝ := 0.25

/-- The time saved due to the increased speed, in hours. -/
def time_saved : ℝ := 0.4

/-- Theorem stating that the original speed satisfies the given conditions. -/
theorem original_speed_correct :
  distance / original_speed - distance / (original_speed * (1 + speed_increase)) = time_saved := by
  sorry

end NUMINAMATH_CALUDE_original_speed_correct_l2781_278170


namespace NUMINAMATH_CALUDE_third_term_of_geometric_sequence_l2781_278108

/-- A geometric sequence with a given product of its first five terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n, a (n + 1) = a n * r

theorem third_term_of_geometric_sequence (a : ℕ → ℝ) 
  (h_geometric : geometric_sequence a) 
  (h_product : a 1 * a 2 * a 3 * a 4 * a 5 = 32) : 
  a 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_third_term_of_geometric_sequence_l2781_278108


namespace NUMINAMATH_CALUDE_event_probability_theorem_l2781_278180

/-- Given an event A with constant probability in three independent trials, 
    if the probability of A occurring at least once is 63/64, 
    then the probability of A occurring exactly once is 9/64. -/
theorem event_probability_theorem (p : ℝ) : 
  (0 ≤ p) ∧ (p ≤ 1) →
  (1 - (1 - p)^3 = 63/64) →
  (3 * p * (1 - p)^2 = 9/64) :=
by sorry

end NUMINAMATH_CALUDE_event_probability_theorem_l2781_278180


namespace NUMINAMATH_CALUDE_contractor_male_workers_l2781_278185

/-- Proves that the number of male workers is 20 given the conditions of the problem -/
theorem contractor_male_workers :
  let female_workers : ℕ := 15
  let child_workers : ℕ := 5
  let male_wage : ℚ := 25
  let female_wage : ℚ := 20
  let child_wage : ℚ := 8
  let average_wage : ℚ := 21
  ∃ male_workers : ℕ,
    (male_wage * male_workers + female_wage * female_workers + child_wage * child_workers) /
    (male_workers + female_workers + child_workers) = average_wage ∧
    male_workers = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_contractor_male_workers_l2781_278185


namespace NUMINAMATH_CALUDE_seventh_observation_seventh_observation_value_l2781_278161

theorem seventh_observation (initial_count : Nat) (initial_avg : ℝ) (new_avg : ℝ) : ℝ :=
  let total_count : Nat := initial_count + 1
  let initial_sum : ℝ := initial_count * initial_avg
  let new_sum : ℝ := total_count * new_avg
  new_sum - initial_sum

theorem seventh_observation_value :
  seventh_observation 6 12 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_seventh_observation_seventh_observation_value_l2781_278161


namespace NUMINAMATH_CALUDE_append_nine_to_two_digit_number_l2781_278193

/-- Given a two-digit number with tens digit t and units digit u,
    appending 9 to the end results in the number 100t + 10u + 9 -/
theorem append_nine_to_two_digit_number (t u : ℕ) (h : t ≤ 9 ∧ u ≤ 9) :
  (10 * t + u) * 10 + 9 = 100 * t + 10 * u + 9 := by
  sorry

end NUMINAMATH_CALUDE_append_nine_to_two_digit_number_l2781_278193


namespace NUMINAMATH_CALUDE_heartsuit_five_three_l2781_278107

def heartsuit (x y : ℝ) : ℝ := 4 * x + 6 * y

theorem heartsuit_five_three : heartsuit 5 3 = 38 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_five_three_l2781_278107


namespace NUMINAMATH_CALUDE_fraction_zero_l2781_278126

theorem fraction_zero (x : ℝ) (h : x ≠ 3) : 
  x = 0 ↔ (2 * x^2 - 6 * x) / (x - 3) = 0 := by sorry

end NUMINAMATH_CALUDE_fraction_zero_l2781_278126


namespace NUMINAMATH_CALUDE_candy_cost_proof_l2781_278145

/-- The cost of candy A per pound -/
def cost_candy_A : ℝ := 3.20

/-- The cost of candy B per pound -/
def cost_candy_B : ℝ := 1.70

/-- The total weight of the mixture in pounds -/
def total_weight : ℝ := 5

/-- The cost per pound of the mixture -/
def mixture_cost_per_pound : ℝ := 2

/-- The weight of candy A in the mixture -/
def weight_candy_A : ℝ := 1

/-- The weight of candy B in the mixture -/
def weight_candy_B : ℝ := total_weight - weight_candy_A

theorem candy_cost_proof :
  cost_candy_A * weight_candy_A + cost_candy_B * weight_candy_B = mixture_cost_per_pound * total_weight :=
by sorry

end NUMINAMATH_CALUDE_candy_cost_proof_l2781_278145


namespace NUMINAMATH_CALUDE_sales_tax_difference_l2781_278121

-- Define the original price, discount rate, and tax rates
def original_price : ℝ := 50
def discount_rate : ℝ := 0.10
def tax_rate_1 : ℝ := 0.08
def tax_rate_2 : ℝ := 0.075

-- Define the discounted price
def discounted_price : ℝ := original_price * (1 - discount_rate)

-- Define the tax difference function
def tax_difference (price : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  price * rate1 - price * rate2

-- Theorem statement
theorem sales_tax_difference :
  tax_difference discounted_price tax_rate_1 tax_rate_2 = 0.225 := by
  sorry

end NUMINAMATH_CALUDE_sales_tax_difference_l2781_278121


namespace NUMINAMATH_CALUDE_complex_modulus_l2781_278134

theorem complex_modulus (z : ℂ) : (z - 1) * I = I - 1 → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l2781_278134


namespace NUMINAMATH_CALUDE_complement_M_U_characterization_l2781_278153

-- Define the universal set U
def U : Set Int := {x | ∃ k, x = 2 * k}

-- Define the set M
def M : Set Int := {x | ∃ k, x = 4 * k}

-- Define the complement of M with respect to U
def complement_M_U : Set Int := {x ∈ U | x ∉ M}

-- Theorem statement
theorem complement_M_U_characterization :
  complement_M_U = {x | ∃ k, x = 4 * k - 2} := by sorry

end NUMINAMATH_CALUDE_complement_M_U_characterization_l2781_278153


namespace NUMINAMATH_CALUDE_min_value_2x_plus_y_l2781_278187

/-- The minimum value of 2x + y given the constraints |y| ≤ 2 - x and x ≥ -1 is -5 -/
theorem min_value_2x_plus_y (x y : ℝ) (h1 : |y| ≤ 2 - x) (h2 : x ≥ -1) : 
  ∃ (m : ℝ), m = -5 ∧ ∀ (x' y' : ℝ), |y'| ≤ 2 - x' → x' ≥ -1 → 2*x' + y' ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_2x_plus_y_l2781_278187


namespace NUMINAMATH_CALUDE_child_growth_l2781_278139

theorem child_growth (current_height previous_height : ℝ) 
  (h1 : current_height = 41.5)
  (h2 : previous_height = 38.5) : 
  current_height - previous_height = 3 := by
sorry

end NUMINAMATH_CALUDE_child_growth_l2781_278139


namespace NUMINAMATH_CALUDE_jeremy_age_l2781_278137

theorem jeremy_age (amy jeremy chris : ℕ) 
  (h1 : amy + jeremy + chris = 132)
  (h2 : amy = jeremy / 3)
  (h3 : chris = 2 * amy) :
  jeremy = 66 := by
sorry

end NUMINAMATH_CALUDE_jeremy_age_l2781_278137


namespace NUMINAMATH_CALUDE_chad_pet_food_difference_l2781_278103

theorem chad_pet_food_difference :
  let cat_packages : ℕ := 6
  let dog_packages : ℕ := 2
  let cat_cans_per_package : ℕ := 9
  let dog_cans_per_package : ℕ := 3
  let total_cat_cans := cat_packages * cat_cans_per_package
  let total_dog_cans := dog_packages * dog_cans_per_package
  total_cat_cans - total_dog_cans = 48 :=
by sorry

end NUMINAMATH_CALUDE_chad_pet_food_difference_l2781_278103


namespace NUMINAMATH_CALUDE_functional_inequality_solution_l2781_278177

/-- A function satisfying the given inequality condition -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, f (x + y + z) + f x ≥ f (x + y) + f (x + z)

/-- The main theorem statement -/
theorem functional_inequality_solution
    (f : ℝ → ℝ)
    (h_diff : Differentiable ℝ f)
    (h_ineq : SatisfiesInequality f) :
    ∃ a b : ℝ, ∀ x, f x = a * x + b :=
  sorry

end NUMINAMATH_CALUDE_functional_inequality_solution_l2781_278177


namespace NUMINAMATH_CALUDE_square_area_to_perimeter_ratio_l2781_278174

theorem square_area_to_perimeter_ratio (s₁ s₂ : ℝ) (h : s₁ ^ 2 / s₂ ^ 2 = 25 / 36) :
  (4 * s₁) / (4 * s₂) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_square_area_to_perimeter_ratio_l2781_278174


namespace NUMINAMATH_CALUDE_quadratic_root_value_l2781_278110

theorem quadratic_root_value (r s : ℝ) : 
  (∃ x : ℂ, 2 * x^2 + r * x + s = 0 ∧ x = 3 + 2*I) → s = 26 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l2781_278110


namespace NUMINAMATH_CALUDE_cards_given_to_mary_problem_l2781_278171

def cards_given_to_mary (initial_cards found_cards final_cards : ℕ) : ℕ :=
  initial_cards + found_cards - final_cards

theorem cards_given_to_mary_problem : cards_given_to_mary 26 40 48 = 18 := by
  sorry

end NUMINAMATH_CALUDE_cards_given_to_mary_problem_l2781_278171


namespace NUMINAMATH_CALUDE_rosie_account_balance_l2781_278117

/-- Represents the total amount in Rosie's account after m deposits -/
def account_balance (initial_amount : ℕ) (deposit_amount : ℕ) (num_deposits : ℕ) : ℕ :=
  initial_amount + deposit_amount * num_deposits

/-- Theorem stating that Rosie's account balance is correctly represented -/
theorem rosie_account_balance (m : ℕ) : 
  account_balance 120 30 m = 120 + 30 * m := by
  sorry

#check rosie_account_balance

end NUMINAMATH_CALUDE_rosie_account_balance_l2781_278117


namespace NUMINAMATH_CALUDE_original_price_of_discounted_dress_l2781_278129

/-- Proves that given a 30% discount on a dress that results in a final price of $35, the original price of the dress was $50. -/
theorem original_price_of_discounted_dress (discount_percentage : ℝ) (final_price : ℝ) : 
  discount_percentage = 30 →
  final_price = 35 →
  (1 - discount_percentage / 100) * 50 = final_price :=
by sorry

end NUMINAMATH_CALUDE_original_price_of_discounted_dress_l2781_278129


namespace NUMINAMATH_CALUDE_school_sample_size_l2781_278138

theorem school_sample_size (n : ℕ) : 
  (6 : ℚ) / 11 * n / 10 - (5 : ℚ) / 11 * n / 10 = 12 → n = 1320 := by
  sorry

end NUMINAMATH_CALUDE_school_sample_size_l2781_278138


namespace NUMINAMATH_CALUDE_shaded_area_in_square_with_semicircle_l2781_278105

theorem shaded_area_in_square_with_semicircle (d : ℝ) (h : d > 0) :
  let s := d / Real.sqrt 2
  let square_area := s^2
  let semicircle_area := π * (d/2)^2 / 2
  square_area - semicircle_area = s^2 - (π/8) * d^2 := by sorry

end NUMINAMATH_CALUDE_shaded_area_in_square_with_semicircle_l2781_278105


namespace NUMINAMATH_CALUDE_smallest_w_divisible_by_13_and_w_plus_3_divisible_by_11_l2781_278172

theorem smallest_w_divisible_by_13_and_w_plus_3_divisible_by_11 : 
  ∃ w : ℕ, w > 0 ∧ w % 13 = 0 ∧ (w + 3) % 11 = 0 ∧
  ∀ x : ℕ, x > 0 ∧ x % 13 = 0 ∧ (x + 3) % 11 = 0 → w ≤ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_w_divisible_by_13_and_w_plus_3_divisible_by_11_l2781_278172


namespace NUMINAMATH_CALUDE_sam_remaining_yellow_marbles_l2781_278158

/-- The number of yellow marbles Sam has after Joan took some -/
def remaining_yellow_marbles (initial : ℕ) (taken : ℕ) : ℕ :=
  initial - taken

/-- Proof that Sam has 61 yellow marbles after Joan took 25 -/
theorem sam_remaining_yellow_marbles :
  remaining_yellow_marbles 86 25 = 61 := by
  sorry

end NUMINAMATH_CALUDE_sam_remaining_yellow_marbles_l2781_278158


namespace NUMINAMATH_CALUDE_point_on_linear_graph_l2781_278173

/-- For any point (a, b) on the graph of y = 2x - 1, 2a - b + 1 = 2 -/
theorem point_on_linear_graph (a b : ℝ) (h : b = 2 * a - 1) : 2 * a - b + 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_linear_graph_l2781_278173


namespace NUMINAMATH_CALUDE_point_P_and_min_value_l2781_278147

-- Define the points
def A : ℝ × ℝ := (3, 4)
def C : ℝ × ℝ := (0, 0)
def M : ℝ × ℝ := (0, 1)
def N : ℝ × ℝ := (1, 0)

-- Define vectors
def AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
def AM : ℝ × ℝ := (M.1 - A.1, M.2 - A.2)
def AN : ℝ × ℝ := (N.1 - A.1, N.2 - A.2)

-- Define the vector equation
def vector_equation (x y : ℝ) : Prop :=
  AC = (x * AM.1, x * AM.2) + (y * AN.1, y * AN.2)

-- Theorem statement
theorem point_P_and_min_value :
  ∃ (x y : ℝ), vector_equation x y ∧ 
  x = 2/3 ∧ y = 1/2 ∧ 
  ∀ (a b : ℝ), vector_equation a b → 9*x^2 + 16*y^2 ≤ 9*a^2 + 16*b^2 :=
sorry

end NUMINAMATH_CALUDE_point_P_and_min_value_l2781_278147


namespace NUMINAMATH_CALUDE_garden_perimeter_l2781_278152

/-- The perimeter of a rectangular garden with width 4 meters and the same area as a rectangular playground of length 16 meters and width 12 meters is 104 meters. -/
theorem garden_perimeter : 
  let playground_length : ℝ := 16
  let playground_width : ℝ := 12
  let garden_width : ℝ := 4
  let playground_area := playground_length * playground_width
  let garden_length := playground_area / garden_width
  let garden_perimeter := 2 * (garden_length + garden_width)
  garden_perimeter = 104 := by sorry

end NUMINAMATH_CALUDE_garden_perimeter_l2781_278152


namespace NUMINAMATH_CALUDE_train_speed_l2781_278122

/-- The speed of a train given its length, the platform length, and the time to cross the platform -/
theorem train_speed (train_length platform_length : Real) (crossing_time : Real) :
  train_length = 110 ∧ 
  platform_length = 323.36799999999994 ∧ 
  crossing_time = 30 →
  (train_length + platform_length) / crossing_time * 3.6 = 52.00416 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2781_278122


namespace NUMINAMATH_CALUDE_inequality_proof_l2781_278140

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 1 / b ≥ 4) ∧ (a^2 + b^2 ≥ 1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2781_278140


namespace NUMINAMATH_CALUDE_same_solution_implies_a_plus_b_equals_one_l2781_278179

theorem same_solution_implies_a_plus_b_equals_one 
  (x y a b : ℝ) 
  (h1 : 2*x + 4*y = 20) 
  (h2 : a*x + b*y = 1)
  (h3 : 2*x - y = 5)
  (h4 : b*x + a*y = 6)
  (h5 : 2*x + 4*y = 20 ∧ a*x + b*y = 1 ↔ 2*x - y = 5 ∧ b*x + a*y = 6) : 
  a + b = 1 := by
sorry


end NUMINAMATH_CALUDE_same_solution_implies_a_plus_b_equals_one_l2781_278179


namespace NUMINAMATH_CALUDE_university_population_l2781_278146

/-- Represents the total number of students at the university -/
def total_students : ℕ := 5000

/-- Represents the sample size -/
def sample_size : ℕ := 500

/-- Represents the number of freshmen in the sample -/
def freshmen_sample : ℕ := 200

/-- Represents the number of sophomores in the sample -/
def sophomore_sample : ℕ := 100

/-- Represents the number of students in other grades -/
def other_grades : ℕ := 2000

/-- Theorem stating that given the sample size, freshmen sample, sophomore sample, 
    and number of students in other grades, the total number of students at the 
    university is 5000 -/
theorem university_population : 
  sample_size = freshmen_sample + sophomore_sample + (other_grades / 10) ∧
  total_students = freshmen_sample * 10 + sophomore_sample * 10 + other_grades :=
sorry

end NUMINAMATH_CALUDE_university_population_l2781_278146


namespace NUMINAMATH_CALUDE_smaller_box_size_l2781_278181

/-- Represents the size and cost of a box of macaroni and cheese -/
structure MacaroniBox where
  size : Float
  cost : Float

/-- Calculates the price per ounce of a MacaroniBox -/
def pricePerOunce (box : MacaroniBox) : Float :=
  box.cost / box.size

theorem smaller_box_size 
  (larger_box : MacaroniBox)
  (smaller_box : MacaroniBox)
  (better_value_price : Float)
  (h1 : larger_box.size = 30)
  (h2 : larger_box.cost = 4.80)
  (h3 : smaller_box.cost = 3.40)
  (h4 : better_value_price = 0.16)
  (h5 : pricePerOunce larger_box ≤ pricePerOunce smaller_box)
  (h6 : pricePerOunce larger_box = better_value_price) :
  smaller_box.size = 21.25 := by
  sorry

#check smaller_box_size

end NUMINAMATH_CALUDE_smaller_box_size_l2781_278181


namespace NUMINAMATH_CALUDE_length_AG_l2781_278164

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  -- Right-angled at A
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 ∧
  -- AB = 3
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 9 ∧
  -- AC = 3√3
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 27

-- Define the altitude AD
def Altitude (A B C D : ℝ × ℝ) : Prop :=
  (D.1 - A.1) * (B.1 - C.1) + (D.2 - A.2) * (B.2 - C.2) = 0

-- Define the median AM
def Median (A B C M : ℝ × ℝ) : Prop :=
  M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define the intersection point G
def Intersection (A D M G : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, G = (A.1 + t * (D.1 - A.1), A.2 + t * (D.2 - A.2)) ∧
             ∃ s : ℝ, G = (A.1 + s * (M.1 - A.1), A.2 + s * (M.2 - A.2))

-- Theorem statement
theorem length_AG (A B C D M G : ℝ × ℝ) :
  Triangle A B C →
  Altitude A B C D →
  Median A B C M →
  Intersection A D M G →
  (G.1 - A.1)^2 + (G.2 - A.2)^2 = 243/64 :=
by sorry

end NUMINAMATH_CALUDE_length_AG_l2781_278164


namespace NUMINAMATH_CALUDE_monthly_spending_fraction_l2781_278190

/-- If a person saves a constant fraction of their unchanging monthly salary,
    and their yearly savings are 6 times their monthly spending,
    then they spend 2/3 of their salary each month. -/
theorem monthly_spending_fraction
  (salary : ℝ)
  (savings_fraction : ℝ)
  (h_salary_positive : 0 < salary)
  (h_savings_fraction : 0 ≤ savings_fraction ∧ savings_fraction ≤ 1)
  (h_yearly_savings : 12 * savings_fraction * salary = 6 * (1 - savings_fraction) * salary) :
  1 - savings_fraction = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_monthly_spending_fraction_l2781_278190


namespace NUMINAMATH_CALUDE_quadratic_equation_and_expression_calculation_l2781_278166

theorem quadratic_equation_and_expression_calculation :
  (∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 7 ∧ x₂ = 2 - Real.sqrt 7 ∧
    x₁^2 - 4*x₁ - 3 = 0 ∧ x₂^2 - 4*x₂ - 3 = 0) ∧
  (|-3| - 4 * Real.sin (π/4) + Real.sqrt 8 + (π - 3)^0 = 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_and_expression_calculation_l2781_278166


namespace NUMINAMATH_CALUDE_at_least_one_even_digit_in_sum_l2781_278150

def is_17_digit (n : ℕ) : Prop := 10^16 ≤ n ∧ n < 10^17

def reverse_number (n : ℕ) : ℕ := 
  let digits := List.reverse (Nat.digits 10 n)
  digits.foldl (λ acc d => acc * 10 + d) 0

theorem at_least_one_even_digit_in_sum (M : ℕ) (hM : is_17_digit M) :
  ∃ d : ℕ, d < 10 ∧ d % 2 = 0 ∧ d ∈ Nat.digits 10 (M + reverse_number M) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_even_digit_in_sum_l2781_278150


namespace NUMINAMATH_CALUDE_joe_taller_than_roy_l2781_278155

/-- Given the heights of Sara and Roy, and the difference between Sara and Joe's heights,
    prove that Joe is 3 inches taller than Roy. -/
theorem joe_taller_than_roy (sara_height joe_height roy_height : ℕ)
  (h1 : sara_height = 45)
  (h2 : sara_height = joe_height + 6)
  (h3 : roy_height = 36) :
  joe_height - roy_height = 3 :=
by sorry

end NUMINAMATH_CALUDE_joe_taller_than_roy_l2781_278155


namespace NUMINAMATH_CALUDE_intersection_equality_implies_complement_union_equality_l2781_278109

universe u

theorem intersection_equality_implies_complement_union_equality
  (U : Type u) [Nonempty U]
  (A B C : Set U)
  (h_nonempty_A : A.Nonempty)
  (h_nonempty_B : B.Nonempty)
  (h_nonempty_C : C.Nonempty)
  (h_intersection : A ∩ B = A ∩ C) :
  (Aᶜ ∪ B) = (Aᶜ ∪ C) :=
by sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_complement_union_equality_l2781_278109


namespace NUMINAMATH_CALUDE_hour_hand_angle_for_9_to_1_ratio_l2781_278104

/-- Represents a toy clock with a specific ratio between hour and minute hand rotations -/
structure ToyClock where
  /-- The number of full circles the minute hand makes for each full circle of the hour hand -/
  minuteToHourRatio : ℕ
  /-- Assumption that the ratio is greater than 1 -/
  ratioGtOne : minuteToHourRatio > 1

/-- Calculates the angle turned by the hour hand when it next coincides with the minute hand -/
def hourHandAngleAtNextCoincidence (clock : ToyClock) : ℚ :=
  360 / (clock.minuteToHourRatio - 1)

/-- Theorem stating that for a toy clock where the minute hand makes 9 circles 
    for each full circle of the hour hand, the hour hand turns 45° at the next coincidence -/
theorem hour_hand_angle_for_9_to_1_ratio :
  let clock : ToyClock := ⟨9, by norm_num⟩
  hourHandAngleAtNextCoincidence clock = 45 := by
  sorry

end NUMINAMATH_CALUDE_hour_hand_angle_for_9_to_1_ratio_l2781_278104


namespace NUMINAMATH_CALUDE_seating_arrangement_with_constraint_l2781_278156

def total_arrangements (n : ℕ) : ℕ := Nat.factorial n

def arrangements_with_pair_together (n : ℕ) : ℕ :=
  Nat.factorial (n - 1) * Nat.factorial 2

theorem seating_arrangement_with_constraint :
  total_arrangements 8 - arrangements_with_pair_together 8 = 30240 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_with_constraint_l2781_278156


namespace NUMINAMATH_CALUDE_total_combinations_l2781_278197

/-- Represents the number of friends in Victoria's group. -/
def num_friends : ℕ := 35

/-- Represents the minimum shoe size. -/
def min_size : ℕ := 5

/-- Represents the maximum shoe size. -/
def max_size : ℕ := 15

/-- Represents the number of unique designs in the store. -/
def num_designs : ℕ := 20

/-- Represents the number of colors for each design. -/
def colors_per_design : ℕ := 4

/-- Represents the number of colors each friend needs to select. -/
def colors_to_select : ℕ := 3

/-- Calculates the number of ways to choose k items from n items. -/
def combination (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- Theorem stating the total number of combinations to explore. -/
theorem total_combinations : 
  num_friends * num_designs * combination colors_per_design colors_to_select = 2800 := by
  sorry

end NUMINAMATH_CALUDE_total_combinations_l2781_278197


namespace NUMINAMATH_CALUDE_lineup_selections_15_l2781_278163

/-- The number of ways to select an ordered lineup of 5 players and 1 substitute from 15 players -/
def lineup_selections (n : ℕ) : ℕ :=
  n * (n - 1) * (n - 2) * (n - 3) * (n - 4) * (n - 5)

/-- Theorem stating that the number of lineup selections from 15 players is 3,276,000 -/
theorem lineup_selections_15 :
  lineup_selections 15 = 3276000 := by
  sorry

#eval lineup_selections 15

end NUMINAMATH_CALUDE_lineup_selections_15_l2781_278163


namespace NUMINAMATH_CALUDE_sibling_ages_sum_l2781_278149

theorem sibling_ages_sum (a b c : ℕ+) : 
  a = b ∧ a < c ∧ a * b * c = 72 → a + b + c = 14 :=
by sorry

end NUMINAMATH_CALUDE_sibling_ages_sum_l2781_278149


namespace NUMINAMATH_CALUDE_purchase_price_calculation_l2781_278182

theorem purchase_price_calculation (P : ℝ) : 0.05 * P + 12 = 30 → P = 360 := by
  sorry

end NUMINAMATH_CALUDE_purchase_price_calculation_l2781_278182


namespace NUMINAMATH_CALUDE_smallest_isosceles_perimeter_square_l2781_278169

/-- A number is composite if it's greater than 1 and not prime -/
def IsComposite (n : ℕ) : Prop := n > 1 ∧ ¬ Nat.Prime n

/-- A natural number is a perfect square if it's equal to some integer squared -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- The perimeter of an isosceles triangle with two sides of length a and one side of length b -/
def IsoscelesPerimeter (a b : ℕ) : ℕ := 2 * a + b

theorem smallest_isosceles_perimeter_square : 
  ∀ a b : ℕ, 
    IsComposite a → 
    IsComposite b → 
    a ≠ b → 
    IsPerfectSquare ((2 * a + b) * (2 * a + b)) → 
    2 * a > b → 
    a + b > a →
    ∀ c d : ℕ, 
      IsComposite c → 
      IsComposite d → 
      c ≠ d → 
      IsPerfectSquare ((2 * c + d) * (2 * c + d)) → 
      2 * c > d → 
      c + d > c →
      (IsoscelesPerimeter a b) * (IsoscelesPerimeter a b) ≤ (IsoscelesPerimeter c d) * (IsoscelesPerimeter c d) → 
      (IsoscelesPerimeter a b) * (IsoscelesPerimeter a b) = 256 :=
by sorry

end NUMINAMATH_CALUDE_smallest_isosceles_perimeter_square_l2781_278169


namespace NUMINAMATH_CALUDE_margin_selling_price_relation_l2781_278113

/-- Proof of the relationship between margin, cost, and selling price -/
theorem margin_selling_price_relation (n : ℝ) (C S M : ℝ) 
  (h1 : n > 2) 
  (h2 : M = (2/n) * C) 
  (h3 : S = C + M) : 
  M = (2/(n+2)) * S := by
  sorry

end NUMINAMATH_CALUDE_margin_selling_price_relation_l2781_278113


namespace NUMINAMATH_CALUDE_range_of_f_l2781_278125

def f (x : ℝ) : ℝ := x^4 + 6*x^2 + 9

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, x ≥ 0 ∧ f x = y) ↔ y ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l2781_278125


namespace NUMINAMATH_CALUDE_total_marbles_is_72_marble_ratio_is_2_4_6_l2781_278115

/-- Represents the number of marbles of each color in a bag -/
structure MarbleBag where
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- Defines the properties of the marble bag based on the given conditions -/
def special_marble_bag : MarbleBag :=
  { red := 12,
    blue := 24,
    yellow := 36 }

/-- Theorem stating that the total number of marbles in the special bag is 72 -/
theorem total_marbles_is_72 :
  special_marble_bag.red + special_marble_bag.blue + special_marble_bag.yellow = 72 := by
  sorry

/-- Theorem stating that the ratio of marbles in the special bag is 2:4:6 -/
theorem marble_ratio_is_2_4_6 :
  2 * special_marble_bag.red = special_marble_bag.blue ∧
  3 * special_marble_bag.red = special_marble_bag.yellow := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_is_72_marble_ratio_is_2_4_6_l2781_278115


namespace NUMINAMATH_CALUDE_demand_exceeds_50k_july_august_l2781_278119

def S (n : ℕ) : ℚ := n / 27 * (21 * n - n^2 - 5)

def demand_exceeds_50k (n : ℕ) : Prop := S n - S (n-1) > 5

theorem demand_exceeds_50k_july_august :
  demand_exceeds_50k 7 ∧ demand_exceeds_50k 8 ∧
  ∀ m, m < 7 ∨ m > 8 → ¬demand_exceeds_50k m :=
sorry

end NUMINAMATH_CALUDE_demand_exceeds_50k_july_august_l2781_278119


namespace NUMINAMATH_CALUDE_intersection_point_unique_l2781_278189

/-- The intersection point of two lines in 2D space -/
def intersection_point : ℚ × ℚ := (20/7, -11/7)

/-- First line equation: 5x - 3y = 19 -/
def line1 (x y : ℚ) : Prop := 5 * x - 3 * y = 19

/-- Second line equation: 6x + 2y = 14 -/
def line2 (x y : ℚ) : Prop := 6 * x + 2 * y = 14

theorem intersection_point_unique :
  let (x, y) := intersection_point
  (line1 x y ∧ line2 x y) ∧
  ∀ x' y', line1 x' y' ∧ line2 x' y' → x' = x ∧ y' = y :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_unique_l2781_278189


namespace NUMINAMATH_CALUDE_pi_arrangement_face_dots_l2781_278186

/-- Represents a cube with dots on its faces -/
structure Cube where
  face1 : Nat
  face2 : Nat
  face3 : Nat
  face4 : Nat
  face5 : Nat
  face6 : Nat
  three_dot_face : face1 = 3 ∨ face2 = 3 ∨ face3 = 3 ∨ face4 = 3 ∨ face5 = 3 ∨ face6 = 3
  two_dot_faces : (face1 = 2 ∧ face2 = 2) ∨ (face1 = 2 ∧ face3 = 2) ∨ (face1 = 2 ∧ face4 = 2) ∨
                  (face1 = 2 ∧ face5 = 2) ∨ (face1 = 2 ∧ face6 = 2) ∨ (face2 = 2 ∧ face3 = 2) ∨
                  (face2 = 2 ∧ face4 = 2) ∨ (face2 = 2 ∧ face5 = 2) ∨ (face2 = 2 ∧ face6 = 2) ∨
                  (face3 = 2 ∧ face4 = 2) ∨ (face3 = 2 ∧ face5 = 2) ∨ (face3 = 2 ∧ face6 = 2) ∨
                  (face4 = 2 ∧ face5 = 2) ∨ (face4 = 2 ∧ face6 = 2) ∨ (face5 = 2 ∧ face6 = 2)
  one_dot_faces : face1 + face2 + face3 + face4 + face5 + face6 = 9

/-- Represents the "П" shape arrangement of cubes -/
structure PiArrangement where
  cubes : Fin 7 → Cube
  contacting_faces_same : ∀ i j, i ≠ j → (cubes i).face1 = (cubes j).face2

/-- The theorem to be proved -/
theorem pi_arrangement_face_dots (arr : PiArrangement) :
  ∃ (a b c : Cube), (a.face1 = 2 ∧ b.face1 = 2 ∧ c.face1 = 3) :=
sorry

end NUMINAMATH_CALUDE_pi_arrangement_face_dots_l2781_278186


namespace NUMINAMATH_CALUDE_prob_less_than_four_at_least_six_of_seven_l2781_278194

/-- The probability of rolling a number less than four on a fair die. -/
def p_less_than_four : ℚ := 1/2

/-- The number of times the die is rolled. -/
def num_rolls : ℕ := 7

/-- The minimum number of successful rolls (less than four) we're interested in. -/
def min_successes : ℕ := 6

/-- The probability of rolling a number less than four at least 'min_successes' times in 'num_rolls' rolls. -/
def probability_at_least_min_successes : ℚ :=
  (Finset.range (num_rolls - min_successes + 1)).sum fun k =>
    (Nat.choose num_rolls (num_rolls - k)) *
    (p_less_than_four ^ (num_rolls - k)) *
    ((1 - p_less_than_four) ^ k)

/-- The main theorem: The probability of rolling a number less than four at least six times in seven rolls of a fair die is 15/128. -/
theorem prob_less_than_four_at_least_six_of_seven :
  probability_at_least_min_successes = 15/128 := by
  sorry

end NUMINAMATH_CALUDE_prob_less_than_four_at_least_six_of_seven_l2781_278194


namespace NUMINAMATH_CALUDE_interview_pass_probability_l2781_278112

/-- Represents a job interview with three questions and three chances to answer. -/
structure JobInterview where
  num_questions : ℕ
  num_chances : ℕ
  prob_correct : ℝ

/-- The probability of passing the given job interview. -/
def pass_probability (interview : JobInterview) : ℝ :=
  interview.prob_correct +
  (1 - interview.prob_correct) * interview.prob_correct +
  (1 - interview.prob_correct) * (1 - interview.prob_correct) * interview.prob_correct

/-- Theorem stating that for the specific interview conditions, 
    the probability of passing is 0.973. -/
theorem interview_pass_probability :
  let interview : JobInterview := {
    num_questions := 3,
    num_chances := 3,
    prob_correct := 0.7
  }
  pass_probability interview = 0.973 := by
  sorry


end NUMINAMATH_CALUDE_interview_pass_probability_l2781_278112


namespace NUMINAMATH_CALUDE_pink_ratio_theorem_l2781_278144

/-- Given a class with the following properties:
  * There are 30 students in total
  * There are 18 girls in the class
  * Half of the class likes green
  * 9 students like yellow
  * The remaining students like pink (all of whom are girls)
  Then the ratio of girls who like pink to the total number of girls is 1/3 -/
theorem pink_ratio_theorem (total_students : ℕ) (total_girls : ℕ) (yellow_fans : ℕ) :
  total_students = 30 →
  total_girls = 18 →
  yellow_fans = 9 →
  (total_students / 2 + yellow_fans + (total_girls - (total_students - total_students / 2 - yellow_fans)) = total_students) →
  (total_girls - (total_students - total_students / 2 - yellow_fans)) / total_girls = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pink_ratio_theorem_l2781_278144


namespace NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l2781_278123

theorem unique_solution_absolute_value_equation :
  ∃! x : ℝ, |x - 15| + |x - 25| = |3*x - 75| :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l2781_278123


namespace NUMINAMATH_CALUDE_dress_ratio_proof_l2781_278133

/-- Proves that the ratio of Melissa's dresses to Emily's dresses is 1:2 -/
theorem dress_ratio_proof (melissa debora emily : ℕ) : 
  debora = melissa + 12 →
  emily = 16 →
  melissa + debora + emily = 44 →
  melissa = emily / 2 := by
sorry

end NUMINAMATH_CALUDE_dress_ratio_proof_l2781_278133


namespace NUMINAMATH_CALUDE_janines_earnings_l2781_278167

/-- Represents the day of the week --/
inductive Day
| Monday
| Tuesday
| Thursday
| Saturday

/-- Calculates the pay rate for a given day --/
def payRate (d : Day) : ℚ :=
  match d with
  | Day.Monday => 4
  | Day.Tuesday => 4
  | Day.Thursday => 4
  | Day.Saturday => 5

/-- Calculates the bonus rate for a given day and hours worked --/
def bonusRate (hours : ℚ) : ℚ :=
  if hours > 2 then 1 else 0

/-- Calculates the earnings for a single day --/
def dailyEarnings (d : Day) (hours : ℚ) : ℚ :=
  hours * (payRate d + bonusRate hours)

/-- Janine's work schedule --/
def schedule : List (Day × ℚ) :=
  [(Day.Monday, 2), (Day.Tuesday, 3/2), (Day.Thursday, 7/2), (Day.Saturday, 5/2)]

/-- Theorem: Janine's total earnings for the week equal $46.5 --/
theorem janines_earnings :
  (schedule.map (fun (d, h) => dailyEarnings d h)).sum = 93/2 := by
  sorry

end NUMINAMATH_CALUDE_janines_earnings_l2781_278167


namespace NUMINAMATH_CALUDE_comparison_theorem_l2781_278101

theorem comparison_theorem :
  (-7/8 : ℚ) < (-6/7 : ℚ) ∧ |(-0.1 : ℝ)| > (-0.2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_comparison_theorem_l2781_278101


namespace NUMINAMATH_CALUDE_extra_calories_burned_l2781_278162

def calories_per_hour : ℕ := 30

def calories_burned (hours : ℕ) : ℕ := hours * calories_per_hour

theorem extra_calories_burned : calories_burned 5 - calories_burned 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_extra_calories_burned_l2781_278162


namespace NUMINAMATH_CALUDE_borrowed_sheets_average_l2781_278141

/-- Represents a document with pages printed on both sides of sheets. -/
structure Document where
  totalPages : Nat
  totalSheets : Nat
  pagesPerSheet : Nat
  borrowedSheets : Nat

/-- Calculates the average page number of remaining sheets after borrowing. -/
def averagePageNumber (doc : Document) : Rat :=
  let remainingSheets := doc.totalSheets - doc.borrowedSheets
  let totalPageSum := doc.totalPages * (doc.totalPages + 1) / 2
  let borrowedPagesStart := doc.borrowedSheets * doc.pagesPerSheet - (doc.pagesPerSheet - 1)
  let borrowedPagesEnd := doc.borrowedSheets * doc.pagesPerSheet
  let borrowedPageSum := (borrowedPagesStart + borrowedPagesEnd) * doc.borrowedSheets / 2
  (totalPageSum - borrowedPageSum) / remainingSheets

theorem borrowed_sheets_average (doc : Document) :
  doc.totalPages = 50 ∧
  doc.totalSheets = 25 ∧
  doc.pagesPerSheet = 2 ∧
  doc.borrowedSheets = 13 →
  averagePageNumber doc = 19 := by
  sorry

end NUMINAMATH_CALUDE_borrowed_sheets_average_l2781_278141


namespace NUMINAMATH_CALUDE_exists_k_for_A_l2781_278188

theorem exists_k_for_A (M : ℕ) (hM : M > 2) :
  ∃ k : ℕ, ((M + Real.sqrt (M^2 - 4 : ℝ)) / 2)^5 = (k + Real.sqrt (k^2 - 4 : ℝ)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_exists_k_for_A_l2781_278188


namespace NUMINAMATH_CALUDE_right_triangle_cosine_l2781_278198

theorem right_triangle_cosine (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : a = 5) (h3 : c = 13) :
  let cos_C := a / c
  cos_C = 5 / 13 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_cosine_l2781_278198


namespace NUMINAMATH_CALUDE_beach_house_rent_total_l2781_278111

theorem beach_house_rent_total (num_people : ℕ) (rent_per_person : ℚ) : 
  num_people = 7 → rent_per_person = 70 → num_people * rent_per_person = 490 := by
  sorry

end NUMINAMATH_CALUDE_beach_house_rent_total_l2781_278111


namespace NUMINAMATH_CALUDE_midpoint_distance_midpoint_path_l2781_278184

/-- Represents a ladder sliding down a wall --/
structure SlidingLadder where
  L : ℝ  -- Length of the ladder
  x : ℝ  -- Horizontal distance from wall to bottom of ladder
  y : ℝ  -- Vertical distance from floor to top of ladder
  h_positive : L > 0  -- Ladder has positive length
  h_pythagorean : x^2 + y^2 = L^2  -- Pythagorean theorem

/-- The midpoint of a sliding ladder is always L/2 distance from the corner --/
theorem midpoint_distance (ladder : SlidingLadder) :
  (ladder.x / 2)^2 + (ladder.y / 2)^2 = (ladder.L / 2)^2 := by
  sorry

/-- The path of the midpoint forms a quarter circle --/
theorem midpoint_path (ladder : SlidingLadder) :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (0, 0) ∧ 
    radius = ladder.L / 2 ∧
    (ladder.x / 2)^2 + (ladder.y / 2)^2 = radius^2 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_distance_midpoint_path_l2781_278184


namespace NUMINAMATH_CALUDE_meetings_count_l2781_278135

/-- Represents the movement of an individual between two points -/
structure Movement where
  speed : ℝ
  journeys : ℕ

/-- Calculates the number of meetings between two individuals -/
def calculate_meetings (a b : Movement) : ℕ :=
  sorry

theorem meetings_count :
  let a : Movement := { speed := 1, journeys := 2015 }
  let b : Movement := { speed := 2, journeys := 4029 }
  (calculate_meetings a b) = 6044 := by
  sorry

end NUMINAMATH_CALUDE_meetings_count_l2781_278135


namespace NUMINAMATH_CALUDE_age_of_other_man_l2781_278120

theorem age_of_other_man (n : ℕ) (initial_avg : ℝ) (new_avg : ℝ) (man_age : ℕ) (women_avg : ℝ) :
  n = 8 ∧ 
  new_avg = initial_avg + 2 ∧ 
  man_age = 20 ∧ 
  women_avg = 30 → 
  ∃ x : ℕ, x = 24 ∧ 
    n * initial_avg - (man_age + x) + 2 * women_avg = n * new_avg :=
by sorry

end NUMINAMATH_CALUDE_age_of_other_man_l2781_278120


namespace NUMINAMATH_CALUDE_max_m_value_l2781_278132

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.exp (2*x) - x

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (x - m) * f x - (1/4) * Real.exp (2*x) + x^2 + x

theorem max_m_value (m : ℤ) :
  (∀ x > 0, Monotone (g m)) →
  m ≤ 1 ∧ ∃ m' : ℤ, m' = 1 ∧ (∀ x > 0, Monotone (g m')) :=
sorry

end NUMINAMATH_CALUDE_max_m_value_l2781_278132


namespace NUMINAMATH_CALUDE_set_operations_l2781_278191

def A : Set ℤ := {x | |x| ≤ 6}
def B : Set ℤ := {1, 2, 3}
def C : Set ℤ := {3, 4, 5, 6}

theorem set_operations :
  (A ∩ (B ∩ C) = {3}) ∧
  (A ∩ (A \ (B ∪ C)) = {-6, -5, -4, -3, -2, -1, 0}) := by sorry

end NUMINAMATH_CALUDE_set_operations_l2781_278191


namespace NUMINAMATH_CALUDE_alien_artifact_age_conversion_l2781_278168

/-- Converts a number from base 8 to base 10 -/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 16 -/
def decimal_to_hex (n : ℕ) : String := sorry

/-- Represents the age in octal -/
def age_octal : ℕ := 7231

/-- The expected result in hexadecimal -/
def expected_hex : String := "E99"

theorem alien_artifact_age_conversion :
  decimal_to_hex (octal_to_decimal age_octal) = expected_hex := by sorry

end NUMINAMATH_CALUDE_alien_artifact_age_conversion_l2781_278168


namespace NUMINAMATH_CALUDE_socks_worn_l2781_278128

/-- Given 3 pairs of socks, if the number of pairs that can be formed from worn socks
    (where no worn socks are from the same original pair) is 6,
    then the number of socks worn is 3. -/
theorem socks_worn (total_pairs : ℕ) (formed_pairs : ℕ) (worn_socks : ℕ) :
  total_pairs = 3 →
  formed_pairs = 6 →
  worn_socks ≤ total_pairs * 2 →
  (∀ (i j : ℕ), i < worn_socks → j < worn_socks → i ≠ j →
    ∃ (p q : ℕ), p < total_pairs → q < total_pairs → p ≠ q) →
  (formed_pairs = worn_socks.choose 2) →
  worn_socks = 3 := by
sorry

end NUMINAMATH_CALUDE_socks_worn_l2781_278128


namespace NUMINAMATH_CALUDE_lenny_remaining_amount_l2781_278183

def calculate_remaining_amount (initial_amount : ℝ) 
  (console_price game_price headphones_price : ℝ)
  (book1_price book2_price book3_price : ℝ)
  (tech_discount tech_tax bookstore_fee : ℝ) : ℝ :=
  let tech_total := console_price + 2 * game_price + headphones_price
  let tech_discounted := tech_total * (1 - tech_discount)
  let tech_with_tax := tech_discounted * (1 + tech_tax)
  let book_total := book1_price + book2_price
  let bookstore_total := book_total * (1 + bookstore_fee)
  let total_spent := tech_with_tax + bookstore_total
  initial_amount - total_spent

theorem lenny_remaining_amount :
  calculate_remaining_amount 500 200 50 75 25 30 15 0.2 0.1 0.02 = 113.90 := by
  sorry

end NUMINAMATH_CALUDE_lenny_remaining_amount_l2781_278183


namespace NUMINAMATH_CALUDE_three_numbers_sum_l2781_278102

theorem three_numbers_sum (x y z : ℤ) 
  (sum_xy : x + y = 40)
  (sum_yz : y + z = 50)
  (sum_zx : z + x = 70) :
  x = 30 ∧ y = 10 ∧ z = 40 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l2781_278102


namespace NUMINAMATH_CALUDE_sin_2theta_value_l2781_278143

def line1 (θ : ℝ) : ℝ → ℝ → Prop :=
  fun x y ↦ x * Real.cos θ + 2 * y = 0

def line2 (θ : ℝ) : ℝ → ℝ → Prop :=
  fun x y ↦ 3 * x + y * Real.sin θ + 3 = 0

def perpendicular (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ x₁ y₁ x₂ y₂, f x₁ y₁ ∧ g x₂ y₂ ∧ 
    (y₂ - y₁) * (x₂ - x₁) + (x₂ - x₁) * (y₂ - y₁) = 0

theorem sin_2theta_value (θ : ℝ) :
  perpendicular (line1 θ) (line2 θ) → Real.sin (2 * θ) = -12/13 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l2781_278143


namespace NUMINAMATH_CALUDE_triangle_side_length_l2781_278114

/-- Given a triangle ABC with side lengths a, b, c, prove that if a = 2, b + c = 7, and cos B = -1/4, then b = 4 -/
theorem triangle_side_length (a b c : ℝ) (B : ℝ) (h1 : a = 2) (h2 : b + c = 7) (h3 : Real.cos B = -1/4) : b = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2781_278114


namespace NUMINAMATH_CALUDE_point_on_line_l2781_278176

/-- A point represented by its x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem point_on_line : 
  let p1 : Point := ⟨0, 4⟩
  let p2 : Point := ⟨-6, 1⟩
  let p3 : Point := ⟨6, 7⟩
  collinear p1 p2 p3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l2781_278176


namespace NUMINAMATH_CALUDE_game_show_boxes_l2781_278124

theorem game_show_boxes (n : ℕ) (h1 : n > 0) : 
  (((n - 1 : ℝ) / n) ^ 3 = 0.2962962962962963) → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_game_show_boxes_l2781_278124


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l2781_278130

/-- Given that a and b are inversely proportional, their sum is 40, and their modified difference is 10, prove that b equals 75 when a equals 4. -/
theorem inverse_proportion_problem (a b : ℝ) (k : ℝ) (h1 : a * b = k) (h2 : a + b = 40) (h3 : a - 2*b = 10) : 
  a = 4 → b = 75 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l2781_278130


namespace NUMINAMATH_CALUDE_inverse_proposition_false_l2781_278154

theorem inverse_proposition_false : ¬(∀ a b : ℝ, a + b > 0 → a > 0 ∧ b > 0) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proposition_false_l2781_278154


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l2781_278148

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  sum : ℕ → ℝ -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, sum n = n * (2 * a 1 + (n - 1) * d) / 2

/-- Theorem: For an arithmetic sequence, if S₄ = 6 and 2a₃ - a₂ = 6, then a₁ = -3 -/
theorem arithmetic_sequence_first_term
  (seq : ArithmeticSequence)
  (sum_4 : seq.sum 4 = 6)
  (term_relation : 2 * seq.a 3 - seq.a 2 = 6) :
  seq.a 1 = -3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l2781_278148


namespace NUMINAMATH_CALUDE_percent_equality_l2781_278165

theorem percent_equality (x : ℝ) (h : 0.30 * 0.15 * x = 45) : 0.15 * 0.30 * x = 45 := by
  sorry

end NUMINAMATH_CALUDE_percent_equality_l2781_278165


namespace NUMINAMATH_CALUDE_angle_ratio_MBQ_ABQ_l2781_278159

-- Define the points
variable (A B C P Q M : Point)

-- Define the angles
def angle (X Y Z : Point) : ℝ := sorry

-- State the conditions
axiom BP_bisects_ABC : angle A B P = angle P B C
axiom BQ_bisects_ABC : angle A B Q = angle Q B C
axiom BM_bisects_PBQ : angle P B M = angle M B Q

-- State the theorem
theorem angle_ratio_MBQ_ABQ : 
  (angle M B Q) / (angle A B Q) = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_angle_ratio_MBQ_ABQ_l2781_278159


namespace NUMINAMATH_CALUDE_race_distance_proof_l2781_278142

/-- The distance of a dogsled race course in Wyoming --/
def race_distance : ℝ := 300

/-- The average speed of Team R in mph --/
def team_r_speed : ℝ := 20

/-- The time difference between Team A and Team R in hours --/
def time_difference : ℝ := 3

/-- The speed difference between Team A and Team R in mph --/
def speed_difference : ℝ := 5

/-- Theorem stating the race distance given the conditions --/
theorem race_distance_proof :
  ∃ (t : ℝ), 
    t > 0 ∧
    race_distance = team_r_speed * t ∧
    race_distance = (team_r_speed + speed_difference) * (t - time_difference) :=
by
  sorry

#check race_distance_proof

end NUMINAMATH_CALUDE_race_distance_proof_l2781_278142


namespace NUMINAMATH_CALUDE_geometric_sequence_a3_value_l2781_278178

def is_geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ r : ℤ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a3_value
  (a : ℕ → ℤ)
  (h_geometric : is_geometric_sequence a)
  (h_product : a 2 * a 5 = -32)
  (h_sum : a 3 + a 4 = 4)
  (h_integer_ratio : ∃ r : ℤ, ∀ n : ℕ, a (n + 1) = a n * r) :
  a 3 = -4 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a3_value_l2781_278178
