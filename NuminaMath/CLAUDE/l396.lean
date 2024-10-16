import Mathlib

namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l396_39647

/-- A line y = 3x + c is tangent to the parabola y^2 = 12x if and only if c = 3 -/
theorem line_tangent_to_parabola (c : ℝ) : 
  (∃ x y : ℝ, y = 3 * x + c ∧ y^2 = 12 * x ∧ 
   ∀ x' y' : ℝ, y' = 3 * x' + c ∧ y'^2 = 12 * x' → x' = x ∧ y' = y) ↔ 
  c = 3 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l396_39647


namespace NUMINAMATH_CALUDE_milk_level_lowering_l396_39650

/-- Proves that lowering the milk level by 6 inches in a 50 feet by 25 feet rectangular box removes 4687.5 gallons of milk, given that 1 cubic foot equals 7.5 gallons. -/
theorem milk_level_lowering (box_length : Real) (box_width : Real) (gallons_removed : Real) (cubic_foot_to_gallon : Real) (inches_lowered : Real) : 
  box_length = 50 ∧ 
  box_width = 25 ∧ 
  gallons_removed = 4687.5 ∧ 
  cubic_foot_to_gallon = 7.5 ∧
  inches_lowered = 6 → 
  gallons_removed = (box_length * box_width * (inches_lowered / 12)) * cubic_foot_to_gallon :=
by sorry

end NUMINAMATH_CALUDE_milk_level_lowering_l396_39650


namespace NUMINAMATH_CALUDE_jimmy_calorie_consumption_l396_39603

def cracker_calories : ℕ := 15
def cookie_calories : ℕ := 50
def crackers_eaten : ℕ := 10
def cookies_eaten : ℕ := 7

theorem jimmy_calorie_consumption :
  cracker_calories * crackers_eaten + cookie_calories * cookies_eaten = 500 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_calorie_consumption_l396_39603


namespace NUMINAMATH_CALUDE_final_amount_calculation_l396_39613

def monthly_salary : ℝ := 2000
def tax_rate : ℝ := 0.20
def insurance_rate : ℝ := 0.05
def utility_bill_rate : ℝ := 0.25

theorem final_amount_calculation : 
  let tax := monthly_salary * tax_rate
  let insurance := monthly_salary * insurance_rate
  let after_deductions := monthly_salary - (tax + insurance)
  let utility_bills := after_deductions * utility_bill_rate
  monthly_salary - (tax + insurance + utility_bills) = 1125 := by
sorry

end NUMINAMATH_CALUDE_final_amount_calculation_l396_39613


namespace NUMINAMATH_CALUDE_problem_solution_l396_39687

theorem problem_solution (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 3) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = 1021 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l396_39687


namespace NUMINAMATH_CALUDE_tom_trout_catch_l396_39618

/-- Proves that Tom's catch equals 48 trout given the conditions -/
theorem tom_trout_catch (melanie_catch : ℕ) (tom_multiplier : ℕ) 
  (h1 : melanie_catch = 16)
  (h2 : tom_multiplier = 3) : 
  melanie_catch * tom_multiplier = 48 := by
  sorry

end NUMINAMATH_CALUDE_tom_trout_catch_l396_39618


namespace NUMINAMATH_CALUDE_set_union_equality_l396_39619

theorem set_union_equality (a : ℝ) : 
  let A : Set ℝ := {1, a}
  let B : Set ℝ := {a^2}
  A ∪ B = A → a = -1 ∨ a = 0 := by
sorry

end NUMINAMATH_CALUDE_set_union_equality_l396_39619


namespace NUMINAMATH_CALUDE_fifth_term_equals_twelve_sum_formula_implies_general_term_fifth_term_proof_l396_39688

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℕ := n^2 + 3*n

/-- The nth term of the sequence -/
def a (n : ℕ) : ℕ := 2*n + 2

theorem fifth_term_equals_twelve :
  a 5 = 12 :=
by sorry

theorem sum_formula_implies_general_term (n : ℕ) (h : n ≥ 1) :
  S n - S (n-1) = a n :=
by sorry

theorem fifth_term_proof :
  S 5 - S 4 = 12 :=
by sorry

end NUMINAMATH_CALUDE_fifth_term_equals_twelve_sum_formula_implies_general_term_fifth_term_proof_l396_39688


namespace NUMINAMATH_CALUDE_rectangular_strip_area_l396_39660

theorem rectangular_strip_area (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a * b + a * c + a * (b - a) + a * a + a * (c - a) = 43 →
  a = 1 ∧ b + c = 22 := by
sorry

end NUMINAMATH_CALUDE_rectangular_strip_area_l396_39660


namespace NUMINAMATH_CALUDE_factor_expression_l396_39649

theorem factor_expression (c : ℝ) : 270 * c^2 + 45 * c - 15 = 15 * c * (18 * c + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l396_39649


namespace NUMINAMATH_CALUDE_square_equation_solution_l396_39678

theorem square_equation_solution : ∃! x : ℝ, 97 + x * (19 + 91 / x) = 321 := by
  sorry

end NUMINAMATH_CALUDE_square_equation_solution_l396_39678


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_shorter_base_l396_39669

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid where
  a : ℝ  -- Length of the longer base
  b : ℝ  -- Length of the shorter base
  h : ℝ  -- Height of the trapezoid
  is_isosceles : a > b -- Condition for isosceles trapezoid

/-- 
  Theorem: In an isosceles trapezoid, if the foot of the height from a vertex 
  of the shorter base divides the longer base into two segments with a 
  difference of 10 units, then the length of the shorter base is 10 units.
-/
theorem isosceles_trapezoid_shorter_base 
  (t : IsoscelesTrapezoid) 
  (h : (t.a + t.b) / 2 = (t.a - t.b) / 2 + 10) : 
  t.b = 10 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_shorter_base_l396_39669


namespace NUMINAMATH_CALUDE_not_all_six_multiples_have_prime_neighbor_l396_39631

theorem not_all_six_multiples_have_prime_neighbor :
  ∃ n : ℕ, 6 ∣ n ∧ ¬(Nat.Prime (n - 1) ∨ Nat.Prime (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_all_six_multiples_have_prime_neighbor_l396_39631


namespace NUMINAMATH_CALUDE_prob_at_least_six_heads_in_nine_flips_prob_at_least_six_heads_in_nine_flips_proof_l396_39620

/-- The probability of getting at least 6 heads in 9 fair coin flips -/
theorem prob_at_least_six_heads_in_nine_flips : ℝ :=
  130 / 512

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The probability of getting exactly k heads in n fair coin flips -/
def prob_exactly_k_heads (n k : ℕ) : ℝ := sorry

/-- The probability of getting at least k heads in n fair coin flips -/
def prob_at_least_k_heads (n k : ℕ) : ℝ := sorry

theorem prob_at_least_six_heads_in_nine_flips_proof :
  prob_at_least_k_heads 9 6 = prob_at_least_six_heads_in_nine_flips :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_six_heads_in_nine_flips_prob_at_least_six_heads_in_nine_flips_proof_l396_39620


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_range_l396_39698

-- Define the ellipse equation
def ellipse (x y m : ℝ) : Prop := x^2/3 + y^2/m = 1

-- Define the line equation
def line (x y : ℝ) : Prop := x + 2*y - 2 = 0

-- Define the intersection condition
def intersect_at_two_points (m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂, x₁ ≠ x₂ ∧ ellipse x₁ y₁ m ∧ ellipse x₂ y₂ m ∧ 
                  line x₁ y₁ ∧ line x₂ y₂

-- Theorem statement
theorem ellipse_line_intersection_range :
  ∀ m : ℝ, intersect_at_two_points m ↔ (1/4 < m ∧ m < 3) ∨ m > 3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_range_l396_39698


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l396_39680

theorem quadratic_equation_roots (k : ℝ) (h1 : k ≠ 0) (h2 : k < 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - x₁ + k = 0 ∧ x₂^2 - x₂ + k = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l396_39680


namespace NUMINAMATH_CALUDE_sarah_molly_groups_l396_39627

def chess_club_size : ℕ := 12
def group_size : ℕ := 6

theorem sarah_molly_groups (sarah molly : Fin chess_club_size) 
  (h_distinct : sarah ≠ molly) : 
  (Finset.univ.filter (λ s : Finset (Fin chess_club_size) => 
    s.card = group_size ∧ sarah ∈ s ∧ molly ∈ s)).card = 210 := by
  sorry

end NUMINAMATH_CALUDE_sarah_molly_groups_l396_39627


namespace NUMINAMATH_CALUDE_max_pairs_for_marcella_l396_39648

/-- Given an initial number of shoe pairs and a number of individual shoes lost,
    calculate the maximum number of complete pairs remaining. -/
def max_pairs_remaining (initial_pairs : ℕ) (shoes_lost : ℕ) : ℕ :=
  initial_pairs - shoes_lost

/-- Theorem: With 23 initial pairs and 9 individual shoes lost,
    the maximum number of complete pairs remaining is 14. -/
theorem max_pairs_for_marcella :
  max_pairs_remaining 23 9 = 14 := by
  sorry

end NUMINAMATH_CALUDE_max_pairs_for_marcella_l396_39648


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l396_39681

theorem smallest_n_square_and_cube : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℕ), 5 * n = k^2) ∧ 
  (∃ (m : ℕ), 3 * n = m^3) ∧
  (∀ (x : ℕ), x > 0 → (∃ (y : ℕ), 5 * x = y^2) → (∃ (z : ℕ), 3 * x = z^3) → x ≥ n) ∧
  n = 45 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l396_39681


namespace NUMINAMATH_CALUDE_gigi_initial_flour_l396_39636

/-- The amount of flour required for one batch of cookies -/
def flour_per_batch : ℕ := 2

/-- The number of batches Gigi has already baked -/
def baked_batches : ℕ := 3

/-- The number of additional batches Gigi can make with the remaining flour -/
def future_batches : ℕ := 7

/-- The total amount of flour in Gigi's bag initially -/
def initial_flour : ℕ := flour_per_batch * (baked_batches + future_batches)

theorem gigi_initial_flour :
  initial_flour = 20 := by sorry

end NUMINAMATH_CALUDE_gigi_initial_flour_l396_39636


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_leg_length_l396_39629

/-- Proves that in an isosceles right triangle with a hypotenuse of length 8.485281374238571, the length of one leg is 6. -/
theorem isosceles_right_triangle_leg_length : 
  ∀ (a : ℝ), 
    (a > 0) →  -- Ensure positive length
    (a * Real.sqrt 2 = 8.485281374238571) →  -- Hypotenuse length condition
    (a = 6) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_leg_length_l396_39629


namespace NUMINAMATH_CALUDE_dress_price_l396_39651

/-- The final price of a dress after discounts and tax -/
def final_price (d : ℝ) : ℝ :=
  let sale_price := d * (1 - 0.25)
  let staff_price := sale_price * (1 - 0.20)
  let coupon_price := staff_price * (1 - 0.10)
  coupon_price * (1 + 0.08)

/-- Theorem stating the final price of the dress -/
theorem dress_price (d : ℝ) :
  final_price d = 0.5832 * d := by
  sorry

end NUMINAMATH_CALUDE_dress_price_l396_39651


namespace NUMINAMATH_CALUDE_particle_probability_theorem_l396_39697

/-- Probability of hitting (0,0) first when starting from (x,y) -/
noncomputable def P (x y : ℕ) : ℚ :=
  if x = 0 ∧ y = 0 then 1
  else if x = 0 ∨ y = 0 then 0
  else (P (x-1) y + P x (y-1) + P (x-1) (y-1)) / 3

theorem particle_probability_theorem :
  ∃ (p : ℕ), p > 0 ∧ ¬(3 ∣ p) ∧ P 3 5 = p / 3^7 := by sorry

end NUMINAMATH_CALUDE_particle_probability_theorem_l396_39697


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l396_39642

theorem right_triangle_side_length 
  (Q R S : ℝ × ℝ) 
  (right_angle_Q : (R.1 - Q.1) * (S.1 - Q.1) + (R.2 - Q.2) * (S.2 - Q.2) = 0) 
  (cos_R : (R.1 - Q.1) / Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) = 5/13) 
  (RS_length : (R.1 - S.1)^2 + (R.2 - S.2)^2 = 13^2) : 
  (Q.1 - S.1)^2 + (Q.2 - S.2)^2 = 12^2 := by
sorry


end NUMINAMATH_CALUDE_right_triangle_side_length_l396_39642


namespace NUMINAMATH_CALUDE_certain_number_problem_l396_39659

theorem certain_number_problem (x : ℝ) (y : ℝ) (h1 : x = 3) 
  (h2 : (x + 1) / (x + y) = (x + y) / (x + 13)) : y = 5 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l396_39659


namespace NUMINAMATH_CALUDE_correct_speeds_l396_39663

/-- Two points moving uniformly along a circumference -/
structure MovingPoints where
  circumference : ℝ
  time_difference : ℝ
  coincidence_interval : ℝ

/-- The speeds of the two points -/
def speeds (mp : MovingPoints) : ℝ × ℝ :=
  sorry

/-- Theorem stating the correct speeds for the given conditions -/
theorem correct_speeds (mp : MovingPoints) 
  (h1 : mp.circumference = 60)
  (h2 : mp.time_difference = 5)
  (h3 : mp.coincidence_interval = 60) :
  speeds mp = (3, 4) :=
sorry

end NUMINAMATH_CALUDE_correct_speeds_l396_39663


namespace NUMINAMATH_CALUDE_larger_integer_value_l396_39655

theorem larger_integer_value (a b : ℕ+) (h1 : (a : ℝ) / (b : ℝ) = 7 / 3) (h2 : (a : ℕ) * b = 441) :
  max a b = ⌊7 * Real.sqrt 21⌋ := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_value_l396_39655


namespace NUMINAMATH_CALUDE_toothpicks_count_l396_39673

/-- The number of small triangles in the base row of the large triangle -/
def base_triangles : ℕ := 1001

/-- The total number of small triangles in the large triangle -/
def total_triangles : ℕ := (base_triangles * (base_triangles + 1)) / 2

/-- The number of toothpicks required to construct the large triangle -/
def toothpicks_required : ℕ := (3 * total_triangles) / 2 + 3 * base_triangles

/-- Theorem stating that the number of toothpicks required is 755255 -/
theorem toothpicks_count : toothpicks_required = 755255 := by
  sorry

end NUMINAMATH_CALUDE_toothpicks_count_l396_39673


namespace NUMINAMATH_CALUDE_evaluate_expression_l396_39656

theorem evaluate_expression : 6 - 8 * (9 - 4^2) / 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l396_39656


namespace NUMINAMATH_CALUDE_response_rate_percentage_l396_39696

def responses_needed : ℕ := 300
def questionnaires_mailed : ℕ := 600

theorem response_rate_percentage : 
  (responses_needed : ℚ) / questionnaires_mailed * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_response_rate_percentage_l396_39696


namespace NUMINAMATH_CALUDE_shift_theorem_l396_39668

/-- Represents a quadratic function of the form a(x-h)^2 + k --/
structure QuadraticFunction where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Shifts a quadratic function horizontally --/
def horizontal_shift (f : QuadraticFunction) (d : ℝ) : QuadraticFunction :=
  { a := f.a, h := f.h + d, k := f.k }

/-- Shifts a quadratic function vertically --/
def vertical_shift (f : QuadraticFunction) (d : ℝ) : QuadraticFunction :=
  { a := f.a, h := f.h, k := f.k + d }

/-- The original quadratic function y = 2(x-2)^2 - 5 --/
def original_function : QuadraticFunction :=
  { a := 2, h := 2, k := -5 }

/-- The resulting function after shifts --/
def shifted_function : QuadraticFunction :=
  { a := 2, h := 4, k := -2 }

theorem shift_theorem :
  (vertical_shift (horizontal_shift original_function 2) 3) = shifted_function := by
  sorry

end NUMINAMATH_CALUDE_shift_theorem_l396_39668


namespace NUMINAMATH_CALUDE_safe_gold_rows_l396_39612

/-- The number of gold bars per row in the safe. -/
def gold_bars_per_row : ℕ := 20

/-- The total worth of all gold bars in the safe, in dollars. -/
def total_worth : ℕ := 1600000

/-- The number of rows of gold bars in the safe. -/
def num_rows : ℕ := total_worth / (gold_bars_per_row * (total_worth / gold_bars_per_row))

theorem safe_gold_rows : num_rows = 1 := by
  sorry

end NUMINAMATH_CALUDE_safe_gold_rows_l396_39612


namespace NUMINAMATH_CALUDE_negation_of_even_sum_proposition_l396_39633

def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem negation_of_even_sum_proposition :
  ¬(∀ a b : ℤ, is_even a ∧ is_even b → is_even (a + b)) ↔
  (∃ a b : ℤ, ¬(is_even a ∧ is_even b) ∧ ¬(is_even (a + b))) :=
sorry

end NUMINAMATH_CALUDE_negation_of_even_sum_proposition_l396_39633


namespace NUMINAMATH_CALUDE_parabola_x_intercepts_l396_39682

/-- The number of x-intercepts of the parabola y = 3x^2 - 4x + 1 -/
theorem parabola_x_intercepts :
  let f (x : ℝ) := 3 * x^2 - 4 * x + 1
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ (x : ℝ), f x = 0 → x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_parabola_x_intercepts_l396_39682


namespace NUMINAMATH_CALUDE_tims_dimes_count_l396_39602

/-- Represents the number of coins of each type --/
structure CoinCount where
  nickels : ℕ
  dimes : ℕ
  halfDollars : ℕ

/-- Calculates the total value of coins in dollars --/
def coinValue (c : CoinCount) : ℚ :=
  0.05 * c.nickels + 0.10 * c.dimes + 0.50 * c.halfDollars

/-- Represents Tim's earnings from shining shoes and tips --/
structure TimsEarnings where
  shoeShining : CoinCount
  tipJar : CoinCount

/-- The main theorem to prove --/
theorem tims_dimes_count 
  (earnings : TimsEarnings)
  (h1 : earnings.shoeShining.nickels = 3)
  (h2 : earnings.tipJar.dimes = 7)
  (h3 : earnings.tipJar.halfDollars = 9)
  (h4 : coinValue earnings.shoeShining + coinValue earnings.tipJar = 6.65) :
  earnings.shoeShining.dimes = 13 :=
by sorry

end NUMINAMATH_CALUDE_tims_dimes_count_l396_39602


namespace NUMINAMATH_CALUDE_probiotic_diameter_scientific_notation_l396_39606

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem probiotic_diameter_scientific_notation :
  toScientificNotation 0.00000002 = ScientificNotation.mk 2 (-8) (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_probiotic_diameter_scientific_notation_l396_39606


namespace NUMINAMATH_CALUDE_no_base_for_256_with_4_digits_l396_39670

theorem no_base_for_256_with_4_digits :
  ¬ ∃ b : ℕ, b ≥ 2 ∧ b^3 ≤ 256 ∧ 256 < b^4 := by
  sorry

end NUMINAMATH_CALUDE_no_base_for_256_with_4_digits_l396_39670


namespace NUMINAMATH_CALUDE_simplify_fraction_l396_39608

theorem simplify_fraction : 
  (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 := by
sorry

end NUMINAMATH_CALUDE_simplify_fraction_l396_39608


namespace NUMINAMATH_CALUDE_all_odd_rolls_probability_l396_39643

def standard_die_odd_prob : ℚ := 1/2

def roll_count : ℕ := 8

theorem all_odd_rolls_probability :
  (standard_die_odd_prob ^ roll_count : ℚ) = 1/256 := by
  sorry

end NUMINAMATH_CALUDE_all_odd_rolls_probability_l396_39643


namespace NUMINAMATH_CALUDE_correct_equation_l396_39641

theorem correct_equation (a b : ℝ) : 3 * a^2 * b - 4 * b * a^2 = -a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_l396_39641


namespace NUMINAMATH_CALUDE_sum_of_powers_l396_39657

theorem sum_of_powers (a b c d : ℕ) : 
  a < 4 → b < 4 → c < 4 → d < 4 → 
  a > 0 → b > 0 → c > 0 → d > 0 →
  b / c = 1 →
  (4^a + 3^b + 2^c + 1^d = 10) ∨ 
  (4^a + 3^b + 2^c + 1^d = 22) ∨ 
  (4^a + 3^b + 2^c + 1^d = 70) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_powers_l396_39657


namespace NUMINAMATH_CALUDE_intersection_P_Q_l396_39646

def P : Set ℝ := {0, 1, 2, 3}
def Q : Set ℝ := {x : ℝ | |x| < 2}

theorem intersection_P_Q : P ∩ Q = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l396_39646


namespace NUMINAMATH_CALUDE_bridge_length_bridge_length_proof_l396_39691

/-- The length of a bridge given train parameters and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time
  let bridge_length := total_distance - train_length
  bridge_length

/-- Proof that the bridge length is approximately 131.98 meters -/
theorem bridge_length_proof :
  ∃ ε > 0, |bridge_length 110 36 24.198064154867613 - 131.98| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_bridge_length_bridge_length_proof_l396_39691


namespace NUMINAMATH_CALUDE_girls_who_left_l396_39683

theorem girls_who_left (initial_boys : ℕ) (initial_girls : ℕ) (final_students : ℕ) :
  initial_boys = 24 →
  initial_girls = 14 →
  final_students = 30 →
  ∃ (left_girls : ℕ),
    left_girls = initial_girls - (final_students - (initial_boys - left_girls)) ∧
    left_girls = 4 := by
  sorry

end NUMINAMATH_CALUDE_girls_who_left_l396_39683


namespace NUMINAMATH_CALUDE_white_marbles_in_basket_c_l396_39634

/-- Represents a basket of marbles -/
structure Basket where
  color1 : String
  count1 : ℕ
  color2 : String
  count2 : ℕ

/-- The greatest difference between marble counts in any basket -/
def greatestDifference : ℕ := 6

/-- Basket A containing red and yellow marbles -/
def basketA : Basket := ⟨"red", 4, "yellow", 2⟩

/-- Basket B containing green and yellow marbles -/
def basketB : Basket := ⟨"green", 6, "yellow", 1⟩

/-- Basket C containing white and yellow marbles -/
def basketC : Basket := ⟨"white", 15, "yellow", 9⟩

/-- Theorem stating that the number of white marbles in Basket C is 15 -/
theorem white_marbles_in_basket_c :
  basketC.color1 = "white" ∧ basketC.count1 = 15 :=
by sorry

end NUMINAMATH_CALUDE_white_marbles_in_basket_c_l396_39634


namespace NUMINAMATH_CALUDE_cubic_equation_solution_sum_l396_39622

/-- Given r, s, and t are solutions of x^3 - 6x^2 + 11x - 16 = 0, prove that (r+s)/t + (s+t)/r + (t+r)/s = 11/8 -/
theorem cubic_equation_solution_sum (r s t : ℝ) : 
  r^3 - 6*r^2 + 11*r - 16 = 0 →
  s^3 - 6*s^2 + 11*s - 16 = 0 →
  t^3 - 6*t^2 + 11*t - 16 = 0 →
  (r+s)/t + (s+t)/r + (t+r)/s = 11/8 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_sum_l396_39622


namespace NUMINAMATH_CALUDE_shopkeeper_gain_percent_shopkeeper_gain_percent_approx_l396_39616

/-- Calculates the shopkeeper's gain percent during a clearance sale -/
theorem shopkeeper_gain_percent (marked_price : ℝ) (original_gain_percent : ℝ) 
  (initial_discount : ℝ) (sales_tax : ℝ) (additional_discount : ℝ) : ℝ :=
  let cost_price := marked_price / (1 + original_gain_percent / 100)
  let discounted_price := marked_price * (1 - initial_discount / 100)
  let price_after_tax := discounted_price * (1 + sales_tax / 100)
  let final_selling_price := price_after_tax * (1 - additional_discount / 100)
  let gain := final_selling_price - cost_price
  (gain / cost_price) * 100

/-- The shopkeeper's gain percent during the clearance sale is approximately 1.07% -/
theorem shopkeeper_gain_percent_approx :
  ∃ ε > 0, |shopkeeper_gain_percent 30 15 10 5 7 - 1.07| < ε :=
sorry

end NUMINAMATH_CALUDE_shopkeeper_gain_percent_shopkeeper_gain_percent_approx_l396_39616


namespace NUMINAMATH_CALUDE_son_age_l396_39664

theorem son_age (son_age man_age : ℕ) : 
  man_age = son_age + 37 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 35 := by
sorry

end NUMINAMATH_CALUDE_son_age_l396_39664


namespace NUMINAMATH_CALUDE_hadley_total_distance_l396_39654

-- Define the distances
def distance_to_grocery : ℕ := 2
def distance_to_pet_store : ℕ := 2 - 1
def distance_to_home : ℕ := 4 - 1

-- State the theorem
theorem hadley_total_distance :
  distance_to_grocery + distance_to_pet_store + distance_to_home = 6 := by
  sorry

end NUMINAMATH_CALUDE_hadley_total_distance_l396_39654


namespace NUMINAMATH_CALUDE_min_value_on_line_l396_39694

/-- Given a point C(a,b) on the line passing through A(1,1) and B(-2,4),
    the minimum value of 1/a + 4/b is 9/2 -/
theorem min_value_on_line (a b : ℝ) (h : a + b = 2) :
  (∀ x y : ℝ, x + y = 2 → 1/a + 4/b ≤ 1/x + 4/y) ∧ (∃ x y : ℝ, x + y = 2 ∧ 1/x + 4/y = 9/2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_on_line_l396_39694


namespace NUMINAMATH_CALUDE_largest_five_digit_congruent_to_17_mod_34_l396_39693

theorem largest_five_digit_congruent_to_17_mod_34 :
  ∀ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 34 = 17 → n ≤ 99994 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_five_digit_congruent_to_17_mod_34_l396_39693


namespace NUMINAMATH_CALUDE_parabola_range_l396_39679

theorem parabola_range (a b m : ℝ) : 
  (∃ x y : ℝ, y = -x^2 + 2*a*x + b ∧ y = x^2) →
  (m*a - (a^2 + b) - 2*m + 1 = 0) →
  (m ≥ 5/2 ∨ m ≤ 3/2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_range_l396_39679


namespace NUMINAMATH_CALUDE_select_student_count_l396_39605

/-- The number of ways to select one student from a group of high school students -/
def select_student (first_year : ℕ) (second_year : ℕ) (third_year : ℕ) : ℕ :=
  first_year + second_year + third_year

/-- Theorem: Given the specified number of students in each year,
    the number of ways to select one student is 12 -/
theorem select_student_count :
  select_student 3 5 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_select_student_count_l396_39605


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_products_l396_39686

theorem root_sum_reciprocal_products (p q r s t : ℂ) : 
  p^5 - 4*p^4 + 7*p^3 - 3*p^2 + p - 1 = 0 →
  q^5 - 4*q^4 + 7*q^3 - 3*q^2 + q - 1 = 0 →
  r^5 - 4*r^4 + 7*r^3 - 3*r^2 + r - 1 = 0 →
  s^5 - 4*s^4 + 7*s^3 - 3*s^2 + s - 1 = 0 →
  t^5 - 4*t^4 + 7*t^3 - 3*t^2 + t - 1 = 0 →
  p ≠ 0 → q ≠ 0 → r ≠ 0 → s ≠ 0 → t ≠ 0 →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(p*t) + 1/(q*r) + 1/(q*s) + 1/(q*t) + 1/(r*s) + 1/(r*t) + 1/(s*t) = 7 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_products_l396_39686


namespace NUMINAMATH_CALUDE_prob_second_draw_black_l396_39604

/-- The probability of drawing a black ball on the second draw without replacement -/
def second_draw_black_prob (total : ℕ) (black : ℕ) (white : ℕ) : ℚ :=
  if total = black + white ∧ black > 0 ∧ white > 0 then
    black / (total - 1)
  else
    0

theorem prob_second_draw_black :
  second_draw_black_prob 10 3 7 = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_prob_second_draw_black_l396_39604


namespace NUMINAMATH_CALUDE_probability_at_least_one_six_all_different_l396_39685

-- Define the number of faces on a die
def num_faces : ℕ := 6

-- Define the total number of possible outcomes when rolling three dice
def total_outcomes : ℕ := num_faces ^ 3

-- Define the number of favorable outcomes (at least one 6 and all different)
def favorable_outcomes : ℕ := 60

-- Define the number of outcomes with at least one 6
def outcomes_with_six : ℕ := total_outcomes - (num_faces - 1) ^ 3

-- Theorem statement
theorem probability_at_least_one_six_all_different :
  (favorable_outcomes : ℚ) / outcomes_with_six = 60 / 91 := by
  sorry


end NUMINAMATH_CALUDE_probability_at_least_one_six_all_different_l396_39685


namespace NUMINAMATH_CALUDE_train_distance_problem_l396_39653

/-- The distance between two cities given train travel conditions -/
theorem train_distance_problem : ∃ (dist : ℝ) (speed_A speed_B : ℝ),
  -- Two trains meet after 3.3 hours
  dist = 3.3 * (speed_A + speed_B) ∧
  -- Train A departing 24 minutes earlier condition
  0.4 * speed_A + 3 * (speed_A + speed_B) + 14 = 3.3 * (speed_A + speed_B) ∧
  -- Train B departing 36 minutes earlier condition
  0.6 * speed_B + 3 * (speed_A + speed_B) + 9 = 3.3 * (speed_A + speed_B) ∧
  -- The distance between the two cities is 660 km
  dist = 660 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_problem_l396_39653


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l396_39624

theorem quadratic_root_implies_k (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*k*x + k^2 = 0 ∧ x = -1) → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l396_39624


namespace NUMINAMATH_CALUDE_garage_spokes_count_l396_39674

/-- Represents a bicycle with two wheels -/
structure Bicycle where
  front_spokes : ℕ
  back_spokes : ℕ

/-- Represents a tricycle with three wheels -/
structure Tricycle where
  front_spokes : ℕ
  middle_spokes : ℕ
  back_spokes : ℕ

/-- The total number of spokes in all bicycles and the tricycle -/
def total_spokes (bikes : List Bicycle) (trike : Tricycle) : ℕ :=
  (bikes.map (fun b => b.front_spokes + b.back_spokes)).sum +
  (trike.front_spokes + trike.middle_spokes + trike.back_spokes)

theorem garage_spokes_count :
  let bikes : List Bicycle := [
    { front_spokes := 16, back_spokes := 18 },
    { front_spokes := 20, back_spokes := 22 },
    { front_spokes := 24, back_spokes := 26 },
    { front_spokes := 28, back_spokes := 30 }
  ]
  let trike : Tricycle := { front_spokes := 32, middle_spokes := 34, back_spokes := 36 }
  total_spokes bikes trike = 286 := by
  sorry


end NUMINAMATH_CALUDE_garage_spokes_count_l396_39674


namespace NUMINAMATH_CALUDE_sum_even_integers_402_to_500_l396_39667

/-- Sum of first n positive even integers -/
def sumFirstEvenIntegers (n : ℕ) : ℕ := n * (n + 1)

/-- Sum of even integers from a to b inclusive -/
def sumEvenIntegers (a b : ℕ) : ℕ :=
  let n := (b - a) / 2 + 1
  n * (a + b) / 2

theorem sum_even_integers_402_to_500 :
  sumFirstEvenIntegers 50 = 2550 →
  sumEvenIntegers 402 500 = 22550 := by
  sorry

end NUMINAMATH_CALUDE_sum_even_integers_402_to_500_l396_39667


namespace NUMINAMATH_CALUDE_chip_drawing_probability_l396_39695

def total_chips : ℕ := 11
def red_chips : ℕ := 5
def blue_chips : ℕ := 2
def green_chips : ℕ := 3
def yellow_chips : ℕ := 1

def consecutive_color_blocks : ℕ := 3
def yellow_positions : ℕ := total_chips + 1

theorem chip_drawing_probability : 
  (consecutive_color_blocks.factorial * red_chips.factorial * blue_chips.factorial * 
   green_chips.factorial * yellow_positions) / total_chips.factorial = 1 / 385 := by
  sorry

end NUMINAMATH_CALUDE_chip_drawing_probability_l396_39695


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l396_39640

-- Define set A
def A : Set ℝ := {x | |x + 1| < 2}

-- Define set B
def B : Set ℝ := {x | -x^2 + 2*x + 3 ≥ 0}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {x | -3 < x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l396_39640


namespace NUMINAMATH_CALUDE_cos_sin_225_degrees_l396_39644

theorem cos_sin_225_degrees :
  Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 ∧
  Real.sin (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_225_degrees_l396_39644


namespace NUMINAMATH_CALUDE_proportion_solution_l396_39601

/-- Given a proportion x : 10 :: 8 : 0.6, prove that x = 400/3 -/
theorem proportion_solution (x : ℚ) : (x / 10 = 8 / (3/5)) → x = 400/3 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l396_39601


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l396_39662

theorem cyclic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / (y + z) + y / (z + x) + z / (x + y) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l396_39662


namespace NUMINAMATH_CALUDE_range_of_a_l396_39615

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x > -1, x^2 / (x + 1) ≥ a

def q (a : ℝ) : Prop := ∃ x : ℝ, a * x^2 - a * x + 1 = 0

-- State the theorem
theorem range_of_a :
  ∃ a : ℝ, (¬(p a) ∧ ¬(q a)) ∧ (p a ∨ q a) ↔ (a = 0 ∨ a ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l396_39615


namespace NUMINAMATH_CALUDE_quadruple_cylinder_volume_l396_39626

/-- Theorem: Quadrupling Cylinder Dimensions -/
theorem quadruple_cylinder_volume (V : ℝ) (V' : ℝ) :
  V > 0 →  -- Assume positive initial volume
  V' = 64 * V →  -- Definition of V' based on problem conditions
  V' = (4^3) * V  -- Conclusion to prove
  := by sorry

end NUMINAMATH_CALUDE_quadruple_cylinder_volume_l396_39626


namespace NUMINAMATH_CALUDE_player_A_winning_strategy_l396_39628

-- Define the game state
structure GameState where
  board : ℕ

-- Define the possible moves for player A
inductive MoveA where 
  | half : MoveA
  | quarter : MoveA
  | triple : MoveA

-- Define the possible moves for player B
inductive MoveB where
  | increment : MoveB
  | decrement : MoveB

-- Define the game step for player A
def stepA (state : GameState) (move : MoveA) : GameState :=
  match move with
  | MoveA.half => 
      if state.board % 2 = 0 then { board := state.board / 2 } else state
  | MoveA.quarter => 
      if state.board % 4 = 0 then { board := state.board / 4 } else state
  | MoveA.triple => { board := state.board * 3 }

-- Define the game step for player B
def stepB (state : GameState) (move : MoveB) : GameState :=
  match move with
  | MoveB.increment => { board := state.board + 1 }
  | MoveB.decrement => 
      if state.board > 1 then { board := state.board - 1 } else state

-- Define the winning condition
def isWinningState (state : GameState) : Prop :=
  state.board = 3

-- Theorem statement
theorem player_A_winning_strategy (n : ℕ) (h : n > 0) : 
  ∃ (strategy : ℕ → MoveA), 
    ∀ (player_B_moves : ℕ → MoveB),
      ∃ (k : ℕ), isWinningState (
        (stepB (stepA { board := n } (strategy 0)) (player_B_moves 0))
      ) ∨ 
      isWinningState (
        (List.foldl 
          (λ state i => stepB (stepA state (strategy i)) (player_B_moves i))
          { board := n }
          (List.range k)
        )
      ) := by
  sorry

end NUMINAMATH_CALUDE_player_A_winning_strategy_l396_39628


namespace NUMINAMATH_CALUDE_right_triangle_circle_intersection_l396_39621

-- Define the triangle and circle
structure RightTriangle :=
  (A B C : ℝ × ℝ)
  (isRight : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0)

structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the theorem
theorem right_triangle_circle_intersection
  (triangle : RightTriangle)
  (circle : Circle)
  (D : ℝ × ℝ)
  (h1 : circle.center = ((triangle.B.1 + triangle.C.1) / 2, (triangle.B.2 + triangle.C.2) / 2))
  (h2 : circle.radius = Real.sqrt ((triangle.B.1 - triangle.C.1)^2 + (triangle.B.2 - triangle.C.2)^2) / 2)
  (h3 : D.1 = triangle.A.1 + 2 * (triangle.C.1 - triangle.A.1) / (triangle.C.1 - triangle.A.1 + triangle.C.2 - triangle.A.2))
  (h4 : D.2 = triangle.A.2 + 2 * (triangle.C.2 - triangle.A.2) / (triangle.C.1 - triangle.A.1 + triangle.C.2 - triangle.A.2))
  (h5 : Real.sqrt ((D.1 - triangle.A.1)^2 + (D.2 - triangle.A.2)^2) = 2)
  (h6 : Real.sqrt ((D.1 - triangle.B.1)^2 + (D.2 - triangle.B.2)^2) = 3)
  : Real.sqrt ((D.1 - triangle.C.1)^2 + (D.2 - triangle.C.2)^2) = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_circle_intersection_l396_39621


namespace NUMINAMATH_CALUDE_multiply_powers_same_base_l396_39661

theorem multiply_powers_same_base (x : ℝ) : x * x^2 = x^3 := by
  sorry

end NUMINAMATH_CALUDE_multiply_powers_same_base_l396_39661


namespace NUMINAMATH_CALUDE_nail_triangle_impossibility_l396_39652

/-- Given a triangle ACE on a wooden wall with nails of lengths AB = 1, CD = 2, EF = 4,
    prove that the distances between nail heads BD = √2, DF = √5, FB = √13 are impossible. -/
theorem nail_triangle_impossibility (AB CD EF BD DF FB : ℝ) :
  AB = 1 → CD = 2 → EF = 4 →
  BD = Real.sqrt 2 → DF = Real.sqrt 5 → FB = Real.sqrt 13 →
  ¬ (∃ (AC CE AE : ℝ), AC > 0 ∧ CE > 0 ∧ AE > 0 ∧
    AC + CE > AE ∧ CE + AE > AC ∧ AE + AC > CE) :=
by sorry

end NUMINAMATH_CALUDE_nail_triangle_impossibility_l396_39652


namespace NUMINAMATH_CALUDE_complex_expression_equality_l396_39645

theorem complex_expression_equality : 
  Real.sqrt (4/9) - Real.sqrt ((-2)^4) + (19/27 - 1)^(1/3) - (-1)^2017 = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l396_39645


namespace NUMINAMATH_CALUDE_chess_players_lost_to_ai_castor_island_ai_losses_l396_39699

/-- The number of chess players who have lost to a computer at least once on Castor island -/
theorem chess_players_lost_to_ai (total_players : ℝ) (never_lost_fraction : ℚ) : ℝ :=
  let never_lost := total_players * (never_lost_fraction : ℝ)
  let lost_to_ai := total_players - never_lost
  ⌊lost_to_ai + 0.5⌋

/-- Given the conditions on Castor island, prove that approximately 48 players have lost to a computer -/
theorem castor_island_ai_losses : 
  ⌊chess_players_lost_to_ai 157.83 (37/53) + 0.5⌋ = 48 := by
sorry

end NUMINAMATH_CALUDE_chess_players_lost_to_ai_castor_island_ai_losses_l396_39699


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l396_39671

/-- The focal length of a hyperbola with equation x²/4 - y²/5 = 1 is 6 -/
theorem hyperbola_focal_length : ∃ (a b c : ℝ),
  (a^2 = 4 ∧ b^2 = 5) →
  (c^2 = a^2 + b^2) →
  (2 * c = 6) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l396_39671


namespace NUMINAMATH_CALUDE_pencil_cost_proof_l396_39611

/-- The cost of 4 pencils and 5 pens in dollars -/
def total_cost_1 : ℚ := 2

/-- The cost of 3 pencils and 4 pens in dollars -/
def total_cost_2 : ℚ := 79/50

/-- The cost of a pencil in dollars -/
def pencil_cost : ℚ := 1/10

theorem pencil_cost_proof :
  ∃ (pen_cost : ℚ),
    4 * pencil_cost + 5 * pen_cost = total_cost_1 ∧
    3 * pencil_cost + 4 * pen_cost = total_cost_2 :=
by sorry

end NUMINAMATH_CALUDE_pencil_cost_proof_l396_39611


namespace NUMINAMATH_CALUDE_jake_fewer_peaches_l396_39666

theorem jake_fewer_peaches (steven_peaches jill_peaches : ℕ) 
  (h1 : steven_peaches = 14)
  (h2 : jill_peaches = 5)
  (jake_peaches : ℕ)
  (h3 : jake_peaches = jill_peaches + 3)
  (h4 : jake_peaches < steven_peaches) :
  steven_peaches - jake_peaches = 6 := by
  sorry

end NUMINAMATH_CALUDE_jake_fewer_peaches_l396_39666


namespace NUMINAMATH_CALUDE_eggs_per_year_is_3320_l396_39692

/-- Represents the number of eggs used for each family member on a given day --/
structure EggUsage where
  children : Nat
  husband : Nat
  lisa : Nat

/-- Represents the egg usage for each day of the week and holidays --/
structure WeeklyEggUsage where
  monday : EggUsage
  tuesday : EggUsage
  wednesday : EggUsage
  thursday : EggUsage
  friday : EggUsage
  holiday : EggUsage

/-- Calculates the total number of eggs used in a year based on the weekly egg usage and number of holidays --/
def totalEggsPerYear (usage : WeeklyEggUsage) (numHolidays : Nat) : Nat :=
  let weekdayTotal := 
    (usage.monday.children * 3 + usage.monday.husband + usage.monday.lisa) * 52 +
    (usage.tuesday.children * 2 + usage.tuesday.husband + usage.tuesday.lisa + 2) * 52 +
    (usage.wednesday.children * 4 + usage.wednesday.husband + usage.wednesday.lisa) * 52 +
    (usage.thursday.children * 3 + usage.thursday.husband + usage.thursday.lisa) * 52 +
    (usage.friday.children * 4 + usage.friday.husband + usage.friday.lisa) * 52
  let holidayTotal := (usage.holiday.children * 4 + usage.holiday.husband + usage.holiday.lisa) * numHolidays
  weekdayTotal + holidayTotal

/-- The main theorem to prove --/
theorem eggs_per_year_is_3320 : 
  ∃ (usage : WeeklyEggUsage) (numHolidays : Nat),
    usage.monday = EggUsage.mk 2 3 2 ∧
    usage.tuesday = EggUsage.mk 2 3 2 ∧
    usage.wednesday = EggUsage.mk 3 4 3 ∧
    usage.thursday = EggUsage.mk 1 2 1 ∧
    usage.friday = EggUsage.mk 2 3 2 ∧
    usage.holiday = EggUsage.mk 2 2 2 ∧
    numHolidays = 8 ∧
    totalEggsPerYear usage numHolidays = 3320 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_year_is_3320_l396_39692


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l396_39609

/-- The area of an equilateral triangle with perimeter 3p is (√3/4) * p^2 -/
theorem equilateral_triangle_area (p : ℝ) (p_pos : p > 0) :
  let perimeter := 3 * p
  ∃ (area : ℝ), area = (Real.sqrt 3 / 4) * p^2 ∧
  ∀ (side : ℝ), side > 0 → 3 * side = perimeter →
  area = (Real.sqrt 3 / 4) * side^2 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l396_39609


namespace NUMINAMATH_CALUDE_stating_max_bulbs_on_theorem_l396_39614

/-- Represents the maximum number of bulbs that can be turned on in an n × n grid -/
def maxBulbsOn (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    n^2 / 2
  else
    (n^2 - 1) / 2

/-- 
Theorem stating the maximum number of bulbs that can be turned on in an n × n grid,
given the constraints of the problem.
-/
theorem max_bulbs_on_theorem (n : ℕ) :
  ∀ (pressed : Finset (ℕ × ℕ)),
    (∀ (i j : ℕ), (i, j) ∈ pressed → i < n ∧ j < n) →
    (∀ (i j k l : ℕ), (i, j) ∈ pressed → (k, l) ∈ pressed → (i = k ∨ j = l) → i = k ∧ j = l) →
    (∃ (final_state : Finset (ℕ × ℕ)),
      (∀ (i j : ℕ), (i, j) ∈ final_state → i < n ∧ j < n) ∧
      final_state.card ≤ maxBulbsOn n) :=
by
  sorry

#check max_bulbs_on_theorem

end NUMINAMATH_CALUDE_stating_max_bulbs_on_theorem_l396_39614


namespace NUMINAMATH_CALUDE_choose_two_from_three_l396_39658

theorem choose_two_from_three (n : ℕ) (k : ℕ) : n.choose k = 3 :=
  by
  -- Assume n = 3 and k = 2
  have h1 : n = 3 := by sorry
  have h2 : k = 2 := by sorry
  
  -- Define the number of interest groups
  let num_groups : ℕ := 3
  
  -- Define the number of groups to choose
  let groups_to_choose : ℕ := 2
  
  -- Assert that n and k match our problem
  have h3 : n = num_groups := by rw [h1]
  have h4 : k = groups_to_choose := by rw [h2]
  
  -- Prove that choosing 2 from 3 equals 3
  sorry

end NUMINAMATH_CALUDE_choose_two_from_three_l396_39658


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l396_39623

theorem fraction_to_decimal :
  (3 : ℚ) / 40 = 0.075 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l396_39623


namespace NUMINAMATH_CALUDE_numerator_smaller_than_a_l396_39607

theorem numerator_smaller_than_a (a b n : ℕ) (h1 : a ≠ 1) (h2 : b > 0) 
  (h3 : Nat.gcd a b = 1) (h4 : (n : ℚ)⁻¹ > a / b) (h5 : a / b > (n + 1 : ℚ)⁻¹) :
  ∃ (p q : ℕ), q > 0 ∧ Nat.gcd p q = 1 ∧ 
  (a : ℚ) / b - (n + 1 : ℚ)⁻¹ = (p : ℚ) / q ∧ p < a := by
  sorry

end NUMINAMATH_CALUDE_numerator_smaller_than_a_l396_39607


namespace NUMINAMATH_CALUDE_train_speed_l396_39684

theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (h1 : train_length = 500) (h2 : crossing_time = 50) :
  train_length / crossing_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l396_39684


namespace NUMINAMATH_CALUDE_plant_growth_rate_l396_39632

/-- Proves that a plant with given growth conditions has a specific daily growth rate -/
theorem plant_growth_rate (initial_length : ℝ) (growth_percentage : ℝ) : 
  initial_length = 11 →
  growth_percentage = 0.3 →
  ∃ (x : ℝ), 
    (initial_length + 9*x) - (initial_length + 3*x) = growth_percentage * (initial_length + 3*x) ∧
    x = 11 / 17 := by
  sorry

#check plant_growth_rate

end NUMINAMATH_CALUDE_plant_growth_rate_l396_39632


namespace NUMINAMATH_CALUDE_squared_sum_geq_one_l396_39672

theorem squared_sum_geq_one (a b c : ℝ) (h : a * b + b * c + c * a = 1) :
  a^2 + b^2 + c^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_squared_sum_geq_one_l396_39672


namespace NUMINAMATH_CALUDE_cube_strictly_increasing_l396_39630

theorem cube_strictly_increasing (a b : ℝ) : a < b → a^3 < b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_strictly_increasing_l396_39630


namespace NUMINAMATH_CALUDE_solution_set_equality_l396_39600

theorem solution_set_equality (x : ℝ) : 
  Set.Icc (-1 : ℝ) (7/3 : ℝ) = { x | |x - 1| + |2*x - 1| ≤ 5 } := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l396_39600


namespace NUMINAMATH_CALUDE_smallest_n_with_common_factor_l396_39637

def has_common_factor_greater_than_one (a b : ℤ) : Prop :=
  ∃ (k : ℤ), k > 1 ∧ k ∣ a ∧ k ∣ b

theorem smallest_n_with_common_factor : 
  (∀ n : ℕ, n > 0 ∧ n < 14 → ¬(has_common_factor_greater_than_one (8*n + 3) (10*n - 4))) ∧
  (has_common_factor_greater_than_one (8*14 + 3) (10*14 - 4)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_common_factor_l396_39637


namespace NUMINAMATH_CALUDE_investment_expected_profit_l396_39690

/-- The number of investment projects -/
def num_projects : ℕ := 3

/-- The probability of success for each project -/
def prob_success : ℚ := 1/2

/-- The profit for a successful project -/
def profit_success : ℚ := 200000

/-- The loss for a failed project -/
def loss_failure : ℚ := 50000

/-- The expected profit for the investment projects -/
def expected_profit : ℚ := 225000

/-- Theorem stating that the expected profit for the investment projects is 225000 yuan -/
theorem investment_expected_profit :
  (num_projects : ℚ) * prob_success * (profit_success + loss_failure) - num_projects * loss_failure = expected_profit :=
by sorry

end NUMINAMATH_CALUDE_investment_expected_profit_l396_39690


namespace NUMINAMATH_CALUDE_books_left_to_read_l396_39625

theorem books_left_to_read (total_books assigned_books : ℕ) 
  (mcgregor_finished floyd_finished : ℕ) 
  (h1 : assigned_books = 89)
  (h2 : mcgregor_finished = 34)
  (h3 : floyd_finished = 32)
  (h4 : total_books = assigned_books - (mcgregor_finished + floyd_finished)) :
  total_books = 23 := by
  sorry

end NUMINAMATH_CALUDE_books_left_to_read_l396_39625


namespace NUMINAMATH_CALUDE_circle_product_arrangement_l396_39677

theorem circle_product_arrangement : ∃ (a b c d e f : ℚ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f ∧
  a = b * f ∧
  b = a * c ∧
  c = b * d ∧
  d = c * e ∧
  e = d * f ∧
  f = e * a := by
  sorry


end NUMINAMATH_CALUDE_circle_product_arrangement_l396_39677


namespace NUMINAMATH_CALUDE_dans_remaining_marbles_l396_39639

theorem dans_remaining_marbles (initial_green : ℝ) (taken : ℝ) (remaining : ℝ) : 
  initial_green = 32.0 → 
  taken = 23.0 → 
  remaining = initial_green - taken → 
  remaining = 9.0 := by
  sorry

end NUMINAMATH_CALUDE_dans_remaining_marbles_l396_39639


namespace NUMINAMATH_CALUDE_f_inequalities_l396_39635

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (1-a)*x - a

theorem f_inequalities :
  (∀ x, f 3 x < 0 ↔ -1 < x ∧ x < 3) ∧
  (∀ x, f (-1) x > 0 ↔ x ≠ -1) ∧
  (∀ a, a > -1 → ∀ x, f a x > 0 ↔ x < -1 ∨ x > a) ∧
  (∀ a, a < -1 → ∀ x, f a x > 0 ↔ x < a ∨ x > -1) :=
by sorry

end NUMINAMATH_CALUDE_f_inequalities_l396_39635


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l396_39665

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l396_39665


namespace NUMINAMATH_CALUDE_hot_dog_purchase_l396_39675

theorem hot_dog_purchase (cost_per_hot_dog : ℕ) (total_paid : ℕ) (h1 : cost_per_hot_dog = 50) (h2 : total_paid = 300) :
  total_paid / cost_per_hot_dog = 6 := by
  sorry

end NUMINAMATH_CALUDE_hot_dog_purchase_l396_39675


namespace NUMINAMATH_CALUDE_arnold_gas_expenditure_l396_39689

def monthly_gas_expenditure (car1_mpg car2_mpg car3_mpg : ℚ) 
  (total_mileage : ℚ) (gas_price : ℚ) : ℚ :=
  let mileage_per_car := total_mileage / 3
  let gallons_car1 := mileage_per_car / car1_mpg
  let gallons_car2 := mileage_per_car / car2_mpg
  let gallons_car3 := mileage_per_car / car3_mpg
  let total_gallons := gallons_car1 + gallons_car2 + gallons_car3
  total_gallons * gas_price

theorem arnold_gas_expenditure :
  monthly_gas_expenditure 50 10 15 450 2 = 56 := by
  sorry

end NUMINAMATH_CALUDE_arnold_gas_expenditure_l396_39689


namespace NUMINAMATH_CALUDE_student_card_distribution_l396_39638

/-- Given n students (n ≥ 3) and m = (n * (n-1)) / 2 cards, prove that if m is odd
    and there exists a distribution of m distinct integers from 1 to m among n students
    such that the pairwise sums of these integers give different remainders modulo m,
    then n - 2 is a perfect square. -/
theorem student_card_distribution (n : ℕ) (h1 : n ≥ 3) :
  let m : ℕ := n * (n - 1) / 2
  ∃ (distribution : Fin n → Fin m),
    Function.Injective distribution ∧
    (∀ i j : Fin n, i ≠ j →
      ∀ k l : Fin n, k ≠ l →
        (distribution i + distribution j : ℕ) % m ≠
        (distribution k + distribution l : ℕ) % m) →
    Odd m →
    ∃ k : ℕ, n - 2 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_student_card_distribution_l396_39638


namespace NUMINAMATH_CALUDE_spinner_sectors_area_l396_39610

/-- Represents a circular spinner with win and lose sectors. -/
structure Spinner :=
  (radius : ℝ)
  (win_prob : ℝ)
  (lose_prob : ℝ)

/-- Calculates the area of a circular sector given the total area and probability. -/
def sector_area (total_area : ℝ) (probability : ℝ) : ℝ :=
  total_area * probability

theorem spinner_sectors_area (s : Spinner) 
  (h1 : s.radius = 12)
  (h2 : s.win_prob = 1/3)
  (h3 : s.lose_prob = 1/2) :
  let total_area := Real.pi * s.radius^2
  sector_area total_area s.win_prob = 48 * Real.pi ∧
  sector_area total_area s.lose_prob = 72 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_spinner_sectors_area_l396_39610


namespace NUMINAMATH_CALUDE_sector_central_angle_l396_39676

/-- Given a circular sector with radius 10 and area 100, prove that the central angle is 2 radians. -/
theorem sector_central_angle (radius : ℝ) (area : ℝ) (h1 : radius = 10) (h2 : area = 100) :
  (2 * area) / (radius ^ 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l396_39676


namespace NUMINAMATH_CALUDE_purchase_plans_theorem_l396_39617

/-- Represents a purchasing plan for items A and B -/
structure PurchasePlan where
  a : ℕ  -- number of A items
  b : ℕ  -- number of B items

/-- Checks if a purchase plan satisfies all given conditions -/
def isValidPlan (p : PurchasePlan) : Prop :=
  p.a + p.b = 40 ∧
  p.a ≥ 3 * p.b ∧
  230 ≤ 8 * p.a + 2 * p.b ∧
  8 * p.a + 2 * p.b ≤ 266

/-- Calculates the total cost of a purchase plan -/
def totalCost (p : PurchasePlan) : ℕ :=
  8 * p.a + 2 * p.b

/-- Theorem stating the properties of valid purchase plans -/
theorem purchase_plans_theorem :
  ∃ (p1 p2 : PurchasePlan),
    isValidPlan p1 ∧
    isValidPlan p2 ∧
    p1 ≠ p2 ∧
    (∀ p, isValidPlan p → p = p1 ∨ p = p2) ∧
    (p1.a < p2.a → totalCost p1 < totalCost p2) :=
  sorry

end NUMINAMATH_CALUDE_purchase_plans_theorem_l396_39617
