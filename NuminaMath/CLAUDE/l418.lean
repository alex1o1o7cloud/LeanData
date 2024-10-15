import Mathlib

namespace NUMINAMATH_CALUDE_unique_divisible_digit_l418_41835

def is_single_digit (n : ℕ) : Prop := n < 10

def number_with_A (A : ℕ) : ℕ := 26372 * 100 + A * 10 + 21

theorem unique_divisible_digit :
  ∃! A : ℕ, is_single_digit A ∧ 
    (∃ k₁ k₂ k₃ : ℕ, 
      number_with_A A = 2 * k₁ ∧
      number_with_A A = 3 * k₂ ∧
      number_with_A A = 4 * k₃) :=
sorry

end NUMINAMATH_CALUDE_unique_divisible_digit_l418_41835


namespace NUMINAMATH_CALUDE_sector_area_l418_41890

/-- The area of a sector with a central angle of 120° and a radius of 3 is 3π. -/
theorem sector_area (angle : Real) (radius : Real) : 
  angle = 120 * π / 180 → radius = 3 → (1/2) * angle * radius^2 = 3 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l418_41890


namespace NUMINAMATH_CALUDE_circle_center_distance_l418_41849

/-- The distance between the center of the circle x²+y²=4x+6y+3 and the point (8,4) is √37 -/
theorem circle_center_distance : ∃ (h k : ℝ),
  (∀ x y : ℝ, x^2 + y^2 = 4*x + 6*y + 3 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 3)) ∧
  Real.sqrt ((8 - h)^2 + (4 - k)^2) = Real.sqrt 37 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_distance_l418_41849


namespace NUMINAMATH_CALUDE_problem_statement_l418_41899

theorem problem_statement : (-24 : ℚ) * (5/6 - 4/3 + 5/8) = -3 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l418_41899


namespace NUMINAMATH_CALUDE_percentage_calculation_l418_41832

-- Define constants
def rupees_to_paise : ℝ → ℝ := (· * 100)

-- Theorem statement
theorem percentage_calculation (x : ℝ) : 
  (x / 100) * rupees_to_paise 160 = 80 → x = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l418_41832


namespace NUMINAMATH_CALUDE_sum_1_to_1000_equals_500500_sum_forward_equals_sum_backward_l418_41846

def sum_1_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_1_to_1000_equals_500500 :
  sum_1_to_n 1000 = 500500 :=
by sorry

theorem sum_forward_equals_sum_backward (n : ℕ) :
  (List.range n).sum = (List.range n).reverse.sum :=
by sorry

#check sum_1_to_1000_equals_500500
#check sum_forward_equals_sum_backward

end NUMINAMATH_CALUDE_sum_1_to_1000_equals_500500_sum_forward_equals_sum_backward_l418_41846


namespace NUMINAMATH_CALUDE_balloon_problem_l418_41898

theorem balloon_problem (x : ℝ) : x + 5.0 = 12 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_balloon_problem_l418_41898


namespace NUMINAMATH_CALUDE_greatest_value_when_x_is_negative_six_l418_41892

theorem greatest_value_when_x_is_negative_six :
  let x : ℝ := -6
  (2 - x > 2 + x) ∧
  (2 - x > x - 1) ∧
  (2 - x > x) ∧
  (2 - x > x / 2) := by
  sorry

end NUMINAMATH_CALUDE_greatest_value_when_x_is_negative_six_l418_41892


namespace NUMINAMATH_CALUDE_circle_C_and_min_chord_length_l418_41851

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 8)^2 + y^2 = 5

-- Define the lines
def line_1 (x y : ℝ) : Prop := y = 2*x - 21
def line_2 (x y : ℝ) : Prop := y = 2*x - 11
def center_line (x y : ℝ) : Prop := x + y = 8

-- Define the intersecting line
def line_l (x y a : ℝ) : Prop := 2*x + a*y + 6*a = a*x + 14

-- Theorem statement
theorem circle_C_and_min_chord_length :
  ∃ (x₀ y₀ : ℝ),
    -- Center of C lies on the center line
    center_line x₀ y₀ ∧
    -- C is tangent to line_1 and line_2
    (∃ (x₁ y₁ : ℝ), circle_C x₁ y₁ ∧ line_1 x₁ y₁) ∧
    (∃ (x₂ y₂ : ℝ), circle_C x₂ y₂ ∧ line_2 x₂ y₂) ∧
    -- The equation of circle C
    (∀ (x y : ℝ), circle_C x y ↔ (x - 8)^2 + y^2 = 5) ∧
    -- Minimum length of chord MN
    (∀ (a : ℝ),
      (∃ (x₁ y₁ x₂ y₂ : ℝ),
        x₁ ≠ x₂ ∧ circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
        line_l x₁ y₁ a ∧ line_l x₂ y₂ a) →
      ∃ (m n : ℝ), m ≤ (x₁ - x₂)^2 + (y₁ - y₂)^2 ∧ m = 12 ∧ n^2 = m) :=
sorry

end NUMINAMATH_CALUDE_circle_C_and_min_chord_length_l418_41851


namespace NUMINAMATH_CALUDE_positive_integer_solutions_count_l418_41814

theorem positive_integer_solutions_count : 
  (Finset.filter (fun p : ℕ × ℕ => 3 * p.1 + 2 * p.2 = 1001) (Finset.product (Finset.range 1002) (Finset.range 1002))).card = 167 :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_count_l418_41814


namespace NUMINAMATH_CALUDE_system_solutions_l418_41893

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 3 * |y| - 4 * |x| = 6
def equation2 (x y a : ℝ) : Prop := x^2 + y^2 - 14*y + 49 - a^2 = 0

-- Define the number of solutions
def has_n_solutions (n : ℕ) (a : ℝ) : Prop :=
  ∃ (solutions : Finset (ℝ × ℝ)), 
    solutions.card = n ∧
    ∀ (x y : ℝ), (x, y) ∈ solutions ↔ equation1 x y ∧ equation2 x y a

-- Theorem statement
theorem system_solutions (a : ℝ) :
  (has_n_solutions 3 a ↔ |a| = 5 ∨ |a| = 9) ∧
  (has_n_solutions 2 a ↔ |a| = 3 ∨ (5 < |a| ∧ |a| < 9)) :=
sorry

end NUMINAMATH_CALUDE_system_solutions_l418_41893


namespace NUMINAMATH_CALUDE_nested_expression_evaluation_l418_41888

theorem nested_expression_evaluation : (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) = 485 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_evaluation_l418_41888


namespace NUMINAMATH_CALUDE_unique_congruence_solution_l418_41863

theorem unique_congruence_solution : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 4 ∧ n ≡ -998 [ZMOD 5] ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_solution_l418_41863


namespace NUMINAMATH_CALUDE_vertex_locus_is_parabola_l418_41864

theorem vertex_locus_is_parabola (a c : ℝ) (ha : a > 0) (hc : c > 0) :
  let vertex (t : ℝ) := (-(t / (2 * a)), a * (-(t / (2 * a)))^2 + t * (-(t / (2 * a))) + c)
  ∃ f : ℝ → ℝ, (∀ x, f x = -a * x^2 + c) ∧
    (∀ t, (vertex t).2 = f (vertex t).1) :=
by sorry

end NUMINAMATH_CALUDE_vertex_locus_is_parabola_l418_41864


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l418_41852

theorem simplify_and_evaluate (a : ℝ) (h : a = -2) :
  (1 - 1 / (a + 1)) / ((a^2 - 2*a + 1) / (a^2 - 1)) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l418_41852


namespace NUMINAMATH_CALUDE_restaurant_service_charge_l418_41886

theorem restaurant_service_charge (total_paid : ℝ) (service_charge_rate : ℝ) 
  (h1 : service_charge_rate = 0.04)
  (h2 : total_paid = 468) :
  ∃ (original_amount : ℝ), 
    original_amount * (1 + service_charge_rate) = total_paid ∧ 
    original_amount = 450 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_service_charge_l418_41886


namespace NUMINAMATH_CALUDE_opposite_of_sqrt_seven_l418_41876

-- Define the opposite function
def opposite (x : ℝ) : ℝ := -x

-- State the theorem
theorem opposite_of_sqrt_seven :
  opposite (Real.sqrt 7) = -(Real.sqrt 7) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_sqrt_seven_l418_41876


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l418_41867

theorem unique_solution_for_equation : 
  ∃! (x : ℕ+), (1 : ℕ)^(x.val + 2) + 2^(x.val + 1) + 3^(x.val - 1) + 4^x.val = 1170 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l418_41867


namespace NUMINAMATH_CALUDE_quadratic_factorization_l418_41883

theorem quadratic_factorization (x : ℝ) : 2 * x^2 + 12 * x + 18 = 2 * (x + 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l418_41883


namespace NUMINAMATH_CALUDE_negative_square_cubed_l418_41821

theorem negative_square_cubed (a : ℝ) : (-a^2)^3 = -a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_cubed_l418_41821


namespace NUMINAMATH_CALUDE_most_balls_l418_41897

theorem most_balls (soccerballs basketballs : ℕ) 
  (h1 : soccerballs = 50)
  (h2 : basketballs = 26)
  (h3 : ∃ baseballs : ℕ, baseballs = basketballs + 8) :
  soccerballs > basketballs ∧ soccerballs > basketballs + 8 := by
sorry

end NUMINAMATH_CALUDE_most_balls_l418_41897


namespace NUMINAMATH_CALUDE_range_of_g_l418_41805

def g (x : ℝ) : ℝ := -x^2 + 3*x - 3

theorem range_of_g :
  ∀ y ∈ Set.range (fun (x : ℝ) => g x), -31 ≤ y ∧ y ≤ -3/4 ∧
  ∃ x₁ x₂ : ℝ, -4 ≤ x₁ ∧ x₁ ≤ 4 ∧ -4 ≤ x₂ ∧ x₂ ≤ 4 ∧ g x₁ = -31 ∧ g x₂ = -3/4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_g_l418_41805


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l418_41830

theorem exam_maximum_marks :
  ∀ (total_marks : ℕ) (passing_percentage : ℚ) (student_score : ℕ) (failing_margin : ℕ),
    passing_percentage = 1/4 →
    student_score = 185 →
    failing_margin = 25 →
    (passing_percentage * total_marks : ℚ) = (student_score + failing_margin) →
    total_marks = 840 := by
  sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l418_41830


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l418_41868

theorem quadratic_inequality_solution_set (x : ℝ) :
  {x | x^2 - 5*x - 6 > 0} = {x | x < -1 ∨ x > 6} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l418_41868


namespace NUMINAMATH_CALUDE_smallest_product_l418_41891

def number_list : List Int := [-5, -3, -1, 2, 4, 6]

def is_valid_product (p : Int) : Prop :=
  ∃ (a b c : Int), a ∈ number_list ∧ b ∈ number_list ∧ c ∈ number_list ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ p = a * b * c

theorem smallest_product :
  ∀ p, is_valid_product p → p ≥ -120 :=
by sorry

end NUMINAMATH_CALUDE_smallest_product_l418_41891


namespace NUMINAMATH_CALUDE_symmetric_point_and_line_l418_41831

-- Define the point A
def A : ℝ × ℝ := (0, 1)

-- Define the line l₁
def l₁ (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the line l₂
def l₂ (x y : ℝ) : Prop := x - 2*y + 2 = 0

-- Define the symmetric point B
def B : ℝ × ℝ := (2, -1)

-- Define the symmetric line l
def l (x y : ℝ) : Prop := 2*x - y - 5 = 0

-- Theorem statement
theorem symmetric_point_and_line :
  (∀ x y : ℝ, l₁ x y ↔ x - y - 1 = 0) ∧ 
  (∀ x y : ℝ, l₂ x y ↔ x - 2*y + 2 = 0) →
  (B = (2, -1) ∧ (∀ x y : ℝ, l x y ↔ 2*x - y - 5 = 0)) :=
sorry

end NUMINAMATH_CALUDE_symmetric_point_and_line_l418_41831


namespace NUMINAMATH_CALUDE_function_composition_identity_l418_41889

/-- Given two functions f and g defined as f(x) = Ax² + B and g(x) = Bx² + A,
    where A ≠ B, if f(g(x)) - g(f(x)) = 2(B - A) for all x, then A + B = 0 -/
theorem function_composition_identity (A B : ℝ) (h : A ≠ B) :
  (∀ x : ℝ, (A * (B * x^2 + A)^2 + B) - (B * (A * x^2 + B)^2 + A) = 2 * (B - A)) →
  A + B = 0 := by
sorry


end NUMINAMATH_CALUDE_function_composition_identity_l418_41889


namespace NUMINAMATH_CALUDE_sqrt_inequality_l418_41861

theorem sqrt_inequality (a b : ℝ) (ha : 0 < a) (hb : a < b) (hc : b < 1) :
  let f : ℝ → ℝ := fun x ↦ Real.sqrt x
  f a < f b ∧ f b < f (1/b) ∧ f (1/b) < f (1/a) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l418_41861


namespace NUMINAMATH_CALUDE_transport_cost_is_162_50_l418_41804

/-- Calculates the transport cost for a refrigerator purchase given the following conditions:
  * purchase_price: The price Ramesh paid after discount
  * discount_rate: The discount rate on the labelled price
  * installation_cost: The cost of installation
  * profit_rate: The desired profit rate if no discount was offered
  * selling_price: The price to sell at to achieve the desired profit rate
-/
def calculate_transport_cost (purchase_price : ℚ) (discount_rate : ℚ) 
  (installation_cost : ℚ) (profit_rate : ℚ) (selling_price : ℚ) : ℚ :=
  let labelled_price := purchase_price / (1 - discount_rate)
  let profit := labelled_price * profit_rate
  let calculated_selling_price := labelled_price + profit
  selling_price - calculated_selling_price - installation_cost

/-- Theorem stating that given the specific conditions of Ramesh's refrigerator purchase,
    the transport cost is 162.50 rupees. -/
theorem transport_cost_is_162_50 :
  calculate_transport_cost 12500 0.20 250 0.10 17600 = 162.50 := by
  sorry

end NUMINAMATH_CALUDE_transport_cost_is_162_50_l418_41804


namespace NUMINAMATH_CALUDE_product_of_powers_of_ten_l418_41887

theorem product_of_powers_of_ten : (10^0.6) * (10^0.4) * (10^0.3) * (10^0.2) * (10^0.5) = 100 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_of_ten_l418_41887


namespace NUMINAMATH_CALUDE_jump_rope_results_l418_41896

def passing_score : ℕ := 140

def scores : List ℤ := [-25, 17, 23, 0, -39, -11, 9, 34]

def score_difference (scores : List ℤ) : ℕ :=
  (scores.maximum?.getD 0 - scores.minimum?.getD 0).toNat

def average_score (scores : List ℤ) : ℚ :=
  passing_score + (scores.sum : ℚ) / scores.length

def calculate_points (score : ℤ) : ℤ :=
  if score > 0 then 2 * score else -score

def total_score (scores : List ℤ) : ℤ :=
  scores.map calculate_points |>.sum

theorem jump_rope_results :
  score_difference scores = 73 ∧
  average_score scores = 141 ∧
  total_score scores = 91 := by
  sorry

end NUMINAMATH_CALUDE_jump_rope_results_l418_41896


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l418_41847

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, |x| < 2 → x^2 - x - 6 < 0) ∧
  (∃ x : ℝ, x^2 - x - 6 < 0 ∧ |x| ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l418_41847


namespace NUMINAMATH_CALUDE_complex_fraction_equals_i_l418_41824

theorem complex_fraction_equals_i : (1 + 5*I) / (5 - I) = I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_i_l418_41824


namespace NUMINAMATH_CALUDE_smallest_gcd_qr_l418_41818

theorem smallest_gcd_qr (p q r : ℕ+) (h1 : Nat.gcd p q = 210) (h2 : Nat.gcd p r = 1155) :
  ∃ (m : ℕ+), (∀ (n : ℕ+), Nat.gcd q r ≥ n → n ≤ m) ∧ Nat.gcd q r ≥ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_qr_l418_41818


namespace NUMINAMATH_CALUDE_f_composition_equality_l418_41811

noncomputable def f (x : ℝ) : ℝ :=
  if x > 3 then Real.exp x else Real.log (x + 1)

theorem f_composition_equality : f (f (f 1)) = Real.log (Real.log (Real.log 2 + 1) + 1) := by
  sorry

end NUMINAMATH_CALUDE_f_composition_equality_l418_41811


namespace NUMINAMATH_CALUDE_exists_valid_divided_rectangle_area_of_valid_divided_rectangle_l418_41884

/-- Represents a rectangle divided into four smaller rectangles --/
structure DividedRectangle where
  a : ℝ  -- vertical side of the original rectangle
  b : ℝ  -- horizontal side of the original rectangle
  x : ℝ  -- side length of the square

/-- Conditions for the divided rectangle --/
def validDividedRectangle (r : DividedRectangle) : Prop :=
  r.a > 0 ∧ r.b > 0 ∧ r.x > 0 ∧
  r.a + r.x = 10 ∧  -- perimeter of adjacent rectangle is 20
  r.b + r.x = 8     -- perimeter of adjacent rectangle is 16

/-- Theorem stating the existence of a valid divided rectangle --/
theorem exists_valid_divided_rectangle :
  ∃ (r : DividedRectangle), validDividedRectangle r :=
sorry

/-- Function to calculate the area of the original rectangle --/
def area (r : DividedRectangle) : ℝ := r.a * r.b

/-- Theorem to find the area of the valid divided rectangle --/
theorem area_of_valid_divided_rectangle :
  ∃ (r : DividedRectangle), validDividedRectangle r ∧ 
  ∃ (A : ℝ), area r = A :=
sorry

end NUMINAMATH_CALUDE_exists_valid_divided_rectangle_area_of_valid_divided_rectangle_l418_41884


namespace NUMINAMATH_CALUDE_sufficiency_not_necessity_l418_41860

theorem sufficiency_not_necessity (a b : ℝ) :
  (a < b ∧ b < 0 → 1 / a > 1 / b) ∧
  ∃ a b : ℝ, 1 / a > 1 / b ∧ ¬(a < b ∧ b < 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficiency_not_necessity_l418_41860


namespace NUMINAMATH_CALUDE_carbon_atoms_in_compound_l418_41828

/-- Represents the number of atoms of each element in a compound -/
structure Compound where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Calculates the molecular weight of a compound given atomic weights -/
def molecularWeight (c : Compound) (carbonWeight hydrogenWeight oxygenWeight : ℕ) : ℕ :=
  c.carbon * carbonWeight + c.hydrogen * hydrogenWeight + c.oxygen * oxygenWeight

theorem carbon_atoms_in_compound :
  ∀ (c : Compound),
    c.hydrogen = 6 →
    c.oxygen = 1 →
    molecularWeight c 12 1 16 = 58 →
    c.carbon = 3 := by
  sorry

end NUMINAMATH_CALUDE_carbon_atoms_in_compound_l418_41828


namespace NUMINAMATH_CALUDE_cannot_generate_AC_l418_41837

/-- Represents a sequence of letters --/
inductive Sequence
| empty : Sequence
| cons : Char → Sequence → Sequence

/-- Checks if a sequence ends with the letter B --/
def endsWithB : Sequence → Bool := sorry

/-- Checks if a sequence starts with the letter A --/
def startsWithA : Sequence → Bool := sorry

/-- Counts the number of consecutive B's in a sequence --/
def countConsecutiveB : Sequence → Nat := sorry

/-- Counts the number of consecutive C's in a sequence --/
def countConsecutiveC : Sequence → Nat := sorry

/-- Applies Rule I: If a sequence ends with B, append C --/
def applyRuleI : Sequence → Sequence := sorry

/-- Applies Rule II: If a sequence starts with A, double the sequence after A --/
def applyRuleII : Sequence → Sequence := sorry

/-- Applies Rule III: Replace BBB with C anywhere in the sequence --/
def applyRuleIII : Sequence → Sequence := sorry

/-- Applies Rule IV: Remove CC anywhere in the sequence --/
def applyRuleIV : Sequence → Sequence := sorry

/-- Checks if a sequence is equal to "AC" --/
def isAC : Sequence → Bool := sorry

/-- Initial sequence "AB" --/
def initialSequence : Sequence := sorry

/-- Represents all sequences that can be generated from the initial sequence --/
inductive GeneratedSequence : Sequence → Prop
| initial : GeneratedSequence initialSequence
| rule1 {s : Sequence} : GeneratedSequence s → endsWithB s = true → GeneratedSequence (applyRuleI s)
| rule2 {s : Sequence} : GeneratedSequence s → startsWithA s = true → GeneratedSequence (applyRuleII s)
| rule3 {s : Sequence} : GeneratedSequence s → countConsecutiveB s ≥ 3 → GeneratedSequence (applyRuleIII s)
| rule4 {s : Sequence} : GeneratedSequence s → countConsecutiveC s ≥ 2 → GeneratedSequence (applyRuleIV s)

theorem cannot_generate_AC :
  ∀ s, GeneratedSequence s → isAC s = false := by sorry

end NUMINAMATH_CALUDE_cannot_generate_AC_l418_41837


namespace NUMINAMATH_CALUDE_substitution_method_simplification_l418_41834

theorem substitution_method_simplification (x y : ℝ) :
  (4 * x - 3 * y = -1) ∧ (5 * x + y = 13) →
  y = 13 - 5 * x := by
sorry

end NUMINAMATH_CALUDE_substitution_method_simplification_l418_41834


namespace NUMINAMATH_CALUDE_prob_at_least_one_junior_l418_41820

/-- The probability of selecting at least one junior when randomly choosing 4 people from a group of 8 seniors and 4 juniors -/
theorem prob_at_least_one_junior (total : ℕ) (seniors : ℕ) (juniors : ℕ) (selected : ℕ) : 
  total = seniors + juniors →
  seniors = 8 →
  juniors = 4 →
  selected = 4 →
  (1 - (seniors.choose selected : ℚ) / (total.choose selected : ℚ)) = 85 / 99 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_junior_l418_41820


namespace NUMINAMATH_CALUDE_symmetry_axis_of_sine_l418_41873

theorem symmetry_axis_of_sine (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (1/2 * x + π/3)
  f (π/3 + (x - π/3)) = f (π/3 - (x - π/3)) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_axis_of_sine_l418_41873


namespace NUMINAMATH_CALUDE_ten_gentlemen_hat_probability_l418_41866

/-- The harmonic number H_n is defined as the sum of reciprocals of the first n positive integers. -/
def harmonicNumber (n : ℕ) : ℚ :=
  (Finset.range n).sum (fun i => 1 / (i + 1 : ℚ))

/-- The probability that n gentlemen each receive their own hat when distributed randomly. -/
def hatProbability (n : ℕ) : ℚ :=
  if n = 0 then 1 else
  (Finset.range (n - 1)).prod (fun i => harmonicNumber (i + 2) / (i + 2 : ℚ))

/-- Theorem stating the probability that 10 gentlemen each receive their own hat. -/
theorem ten_gentlemen_hat_probability :
  ∃ (p : ℚ), hatProbability 10 = p ∧ 0.000515 < p ∧ p < 0.000517 := by
  sorry


end NUMINAMATH_CALUDE_ten_gentlemen_hat_probability_l418_41866


namespace NUMINAMATH_CALUDE_minimum_average_for_remaining_semesters_l418_41833

def required_average : ℝ := 85
def num_semesters : ℕ := 5
def first_three_scores : List ℝ := [84, 88, 80]

theorem minimum_average_for_remaining_semesters :
  let total_required := required_average * num_semesters
  let current_total := first_three_scores.sum
  let remaining_semesters := num_semesters - first_three_scores.length
  let remaining_required := total_required - current_total
  (remaining_required / remaining_semesters : ℝ) = 86.5 := by
sorry

end NUMINAMATH_CALUDE_minimum_average_for_remaining_semesters_l418_41833


namespace NUMINAMATH_CALUDE_spherical_coordinate_equivalence_l418_41875

/-- Given a point in spherical coordinates, find its equivalent representation in standard spherical coordinates. -/
theorem spherical_coordinate_equivalence :
  ∀ (ρ θ φ : ℝ),
  ρ > 0 →
  (∃ (k : ℤ), θ = 3 * π / 8 + 2 * π * k) →
  (∃ (m : ℤ), φ = 9 * π / 5 + 2 * π * m) →
  ∃ (θ' φ' : ℝ),
    0 ≤ θ' ∧ θ' < 2 * π ∧
    0 ≤ φ' ∧ φ' ≤ π ∧
    (ρ, θ', φ') = (4, 11 * π / 8, π / 5) :=
by sorry


end NUMINAMATH_CALUDE_spherical_coordinate_equivalence_l418_41875


namespace NUMINAMATH_CALUDE_cherry_pie_problem_l418_41845

/-- The number of single cherries in one pound of cherries -/
def cherries_per_pound (total_cherries : ℕ) (total_pounds : ℕ) : ℕ :=
  total_cherries / total_pounds

/-- The number of cherries that can be pitted in a given time -/
def cherries_pitted (time_minutes : ℕ) (cherries_per_10_min : ℕ) : ℕ :=
  (time_minutes / 10) * cherries_per_10_min

theorem cherry_pie_problem (pounds_needed : ℕ) (pitting_time_hours : ℕ) (cherries_per_10_min : ℕ) :
  pounds_needed = 3 →
  pitting_time_hours = 2 →
  cherries_per_10_min = 20 →
  cherries_per_pound (cherries_pitted (pitting_time_hours * 60) cherries_per_10_min) pounds_needed = 80 := by
  sorry

end NUMINAMATH_CALUDE_cherry_pie_problem_l418_41845


namespace NUMINAMATH_CALUDE_solve_grocery_store_problem_l418_41874

def grocery_store_problem (regular_soda : ℕ) (diet_soda : ℕ) (total_bottles : ℕ) : Prop :=
  let lite_soda : ℕ := total_bottles - (regular_soda + diet_soda)
  lite_soda = 27

theorem solve_grocery_store_problem :
  grocery_store_problem 57 26 110 := by
  sorry

end NUMINAMATH_CALUDE_solve_grocery_store_problem_l418_41874


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l418_41838

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y, m₁ * x - y = b₁ ↔ m₂ * x - y = b₂) ↔ m₁ = m₂

/-- The value of m for which mx - y - 1 = 0 is parallel to x - 2y + 3 = 0 -/
theorem parallel_lines_m_value :
  (∀ x y, m * x - y - 1 = 0 ↔ x - 2 * y + 3 = 0) → m = 1 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_parallel_lines_m_value_l418_41838


namespace NUMINAMATH_CALUDE_egyptian_pi_approximation_l418_41840

theorem egyptian_pi_approximation (d : ℝ) (h : d > 0) :
  (π * d^2 / 4 = (8 * d / 9)^2) → π = 256 / 81 := by
  sorry

end NUMINAMATH_CALUDE_egyptian_pi_approximation_l418_41840


namespace NUMINAMATH_CALUDE_ln_plus_const_increasing_l418_41836

theorem ln_plus_const_increasing (x : ℝ) (h : x > 0) :
  Monotone (fun x => Real.log x + 2) :=
sorry

end NUMINAMATH_CALUDE_ln_plus_const_increasing_l418_41836


namespace NUMINAMATH_CALUDE_cone_volume_l418_41819

/-- The volume of a cone with slant height 5 and base radius 3 is 12π -/
theorem cone_volume (s h r : ℝ) (hs : s = 5) (hr : r = 3) 
  (height_eq : h^2 + r^2 = s^2) : 
  (1/3 : ℝ) * π * r^2 * h = 12 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l418_41819


namespace NUMINAMATH_CALUDE_q_gt_one_neither_sufficient_nor_necessary_l418_41869

/-- A geometric sequence with common ratio q -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem q_gt_one_neither_sufficient_nor_necessary :
  ∃ (a₁ b₁ : ℕ → ℝ) (q₁ q₂ : ℝ),
    GeometricSequence a₁ q₁ ∧ q₁ > 1 ∧ ¬IncreasingSequence a₁ ∧
    GeometricSequence b₁ q₂ ∧ q₂ ≤ 1 ∧ IncreasingSequence b₁ :=
  sorry

end NUMINAMATH_CALUDE_q_gt_one_neither_sufficient_nor_necessary_l418_41869


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l418_41841

/-- A regular polygon with side length 6 units and exterior angle 90 degrees has a perimeter of 24 units. -/
theorem regular_polygon_perimeter (n : ℕ) (s : ℝ) (E : ℝ) : 
  n > 0 → 
  s = 6 → 
  E = 90 → 
  E = 360 / n → 
  n * s = 24 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l418_41841


namespace NUMINAMATH_CALUDE_wage_increase_percentage_l418_41872

theorem wage_increase_percentage (initial_wage : ℝ) (final_wage : ℝ) : 
  initial_wage = 10 →
  final_wage = 9 →
  final_wage = 0.75 * (initial_wage * (1 + x/100)) →
  x = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_wage_increase_percentage_l418_41872


namespace NUMINAMATH_CALUDE_water_evaporation_period_l418_41857

theorem water_evaporation_period (initial_amount : Real) (daily_rate : Real) (evaporation_percentage : Real) : 
  initial_amount > 0 → 
  daily_rate > 0 → 
  evaporation_percentage > 0 → 
  evaporation_percentage < 100 →
  initial_amount = 40 →
  daily_rate = 0.01 →
  evaporation_percentage = 0.5 →
  (initial_amount * evaporation_percentage / 100) / daily_rate = 20 := by
sorry

end NUMINAMATH_CALUDE_water_evaporation_period_l418_41857


namespace NUMINAMATH_CALUDE_doghouse_accessible_area_l418_41807

-- Define the doghouse
def doghouse_side_length : ℝ := 2

-- Define the tether length
def tether_length : ℝ := 3

-- Theorem statement
theorem doghouse_accessible_area :
  let total_sector_area := π * tether_length^2 * (240 / 360)
  let small_sector_area := 2 * (π * doghouse_side_length^2 * (60 / 360))
  total_sector_area + small_sector_area = (22 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_doghouse_accessible_area_l418_41807


namespace NUMINAMATH_CALUDE_washing_machines_removed_count_l418_41879

/-- Represents the number of washing machines removed from a shipping container --/
def washing_machines_removed (crates boxes_per_crate machines_per_box machines_removed_per_box : ℕ) : ℕ :=
  crates * boxes_per_crate * machines_removed_per_box

/-- Theorem stating the number of washing machines removed from the shipping container --/
theorem washing_machines_removed_count : 
  washing_machines_removed 10 6 4 1 = 60 := by
  sorry

#eval washing_machines_removed 10 6 4 1

end NUMINAMATH_CALUDE_washing_machines_removed_count_l418_41879


namespace NUMINAMATH_CALUDE_last_four_average_l418_41806

theorem last_four_average (numbers : Fin 7 → ℝ) 
  (h1 : (numbers 0 + numbers 1 + numbers 2 + numbers 3) / 4 = 13)
  (h2 : numbers 4 + numbers 5 + numbers 6 = 55)
  (h3 : numbers 3 ^ 2 = numbers 6)
  (h4 : numbers 6 = 25) :
  (numbers 3 + numbers 4 + numbers 5 + numbers 6) / 4 = 15 := by
sorry

end NUMINAMATH_CALUDE_last_four_average_l418_41806


namespace NUMINAMATH_CALUDE_first_player_wins_or_draws_l418_41877

/-- Represents a game where two players take turns picking bills from a sequence. -/
structure BillGame where
  n : ℕ
  bills : List ℕ
  turn : ℕ

/-- Represents a move in the game, either taking from the left or right end. -/
inductive Move
  | Left
  | Right

/-- Represents the result of the game. -/
inductive GameResult
  | FirstPlayerWins
  | SecondPlayerWins
  | Draw

/-- Defines an optimal strategy for the first player. -/
def optimalStrategy : BillGame → Move
  | _ => sorry

/-- Simulates the game with both players following the optimal strategy. -/
def playGame : BillGame → GameResult
  | _ => sorry

/-- Theorem stating that the first player can always ensure a win or draw. -/
theorem first_player_wins_or_draws (n : ℕ) :
  ∀ (game : BillGame),
    game.n = n ∧
    game.bills = List.range (2*n) ∧
    game.turn = 0 →
    playGame game ≠ GameResult.SecondPlayerWins :=
  sorry

end NUMINAMATH_CALUDE_first_player_wins_or_draws_l418_41877


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l418_41815

theorem quadratic_root_problem (b : ℝ) :
  ((-2 : ℝ)^2 + b * (-2) = 0) → (0^2 + b * 0 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l418_41815


namespace NUMINAMATH_CALUDE_four_digit_number_problem_l418_41827

theorem four_digit_number_problem :
  ∃ (n : ℕ),
    1000 ≤ n ∧ n < 10000 ∧  -- four-digit number
    (n / 1000 = 2) ∧  -- thousand's place is 2
    (((n % 1000) * 10 + 2) = 2 * n + 66) ∧  -- condition for moving 2 to unit's place
    n = 2508 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_number_problem_l418_41827


namespace NUMINAMATH_CALUDE_rates_sum_of_squares_l418_41829

/-- Represents the rates of biking, jogging, and swimming in km/h -/
structure Rates where
  biking : ℕ
  jogging : ℕ
  swimming : ℕ

/-- The sum of activities for Ed -/
def ed_sum (r : Rates) : ℕ := 3 * r.biking + 2 * r.jogging + 3 * r.swimming

/-- The sum of activities for Sue -/
def sue_sum (r : Rates) : ℕ := 5 * r.biking + 3 * r.jogging + 2 * r.swimming

/-- The sum of squares of the rates -/
def sum_of_squares (r : Rates) : ℕ := r.biking^2 + r.jogging^2 + r.swimming^2

theorem rates_sum_of_squares : 
  ∃ r : Rates, ed_sum r = 82 ∧ sue_sum r = 99 ∧ sum_of_squares r = 314 := by
  sorry

end NUMINAMATH_CALUDE_rates_sum_of_squares_l418_41829


namespace NUMINAMATH_CALUDE_alcohol_dilution_l418_41850

/-- Proves that adding 3 litres of water to a 20-litre mixture containing 20% alcohol
    results in a new mixture with 17.391304347826086% alcohol. -/
theorem alcohol_dilution (original_volume : ℝ) (original_alcohol_percentage : ℝ) 
    (added_water : ℝ) (new_alcohol_percentage : ℝ) : 
    original_volume = 20 →
    original_alcohol_percentage = 0.20 →
    added_water = 3 →
    new_alcohol_percentage = 0.17391304347826086 →
    (original_volume * original_alcohol_percentage) / (original_volume + added_water) = new_alcohol_percentage :=
by sorry

end NUMINAMATH_CALUDE_alcohol_dilution_l418_41850


namespace NUMINAMATH_CALUDE_tank_capacity_l418_41885

/-- Proves that the capacity of a tank filled by two buckets of 4 and 3 liters,
    where the 3-liter bucket is used 4 times more, is 48 liters. -/
theorem tank_capacity (x : ℕ) : 
  (4 * x = 3 * (x + 4)) → (4 * x = 48) :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_l418_41885


namespace NUMINAMATH_CALUDE_negative_integer_square_plus_self_l418_41803

theorem negative_integer_square_plus_self (N : ℤ) : 
  N < 0 → N^2 + N = 12 → N = -4 := by sorry

end NUMINAMATH_CALUDE_negative_integer_square_plus_self_l418_41803


namespace NUMINAMATH_CALUDE_student_hall_ratio_l418_41813

theorem student_hall_ratio : 
  let general_hall : ℕ := 30
  let total_students : ℕ := 144
  let math_hall (biology_hall : ℕ) : ℚ := (3/5) * (general_hall + biology_hall)
  ∃ biology_hall : ℕ, 
    (general_hall : ℚ) + biology_hall + math_hall biology_hall = total_students ∧
    biology_hall / general_hall = 2 := by
  sorry

end NUMINAMATH_CALUDE_student_hall_ratio_l418_41813


namespace NUMINAMATH_CALUDE_math_quiz_items_l418_41878

theorem math_quiz_items (score_percentage : ℝ) (mistakes : ℕ) (total_items : ℕ) : 
  score_percentage = 0.80 → 
  mistakes = 5 → 
  (total_items - mistakes : ℝ) / total_items = score_percentage → 
  total_items = 25 := by
sorry

end NUMINAMATH_CALUDE_math_quiz_items_l418_41878


namespace NUMINAMATH_CALUDE_distinct_cubes_modulo_prime_l418_41812

theorem distinct_cubes_modulo_prime (a b c p : ℤ) : 
  Prime p → 
  p = a * b + b * c + a * c → 
  a ≠ b → b ≠ c → a ≠ c → 
  (a^3 % p ≠ b^3 % p) ∧ (b^3 % p ≠ c^3 % p) ∧ (a^3 % p ≠ c^3 % p) :=
by sorry

end NUMINAMATH_CALUDE_distinct_cubes_modulo_prime_l418_41812


namespace NUMINAMATH_CALUDE_point_on_circle_l418_41825

/-- 
Given a line ax + by - 1 = 0 that is tangent to the circle x² + y² = 1,
prove that the point P(a, b) lies on the circle.
-/
theorem point_on_circle (a b : ℝ) 
  (h_tangent : ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ a*x + b*y = 1) : 
  a^2 + b^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_point_on_circle_l418_41825


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l418_41801

theorem sqrt_product_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (15 * q) * Real.sqrt (3 * q^2) * Real.sqrt (2 * q^3) = 3 * q^3 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l418_41801


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l418_41800

theorem complex_fraction_equality : (Complex.I + 3) / (Complex.I + 1) = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l418_41800


namespace NUMINAMATH_CALUDE_min_value_of_expression_l418_41823

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (4 * x / (x + 3 * y)) + (3 * y / x) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l418_41823


namespace NUMINAMATH_CALUDE_expression_equality_l418_41844

theorem expression_equality : 
  (-3^2 ≠ -2^3) ∧ 
  ((-3)^2 ≠ (-2)^3) ∧ 
  (-3^2 ≠ (-3)^2) ∧ 
  (-2^3 = (-2)^3) := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l418_41844


namespace NUMINAMATH_CALUDE_combined_blanket_thickness_l418_41862

/-- The combined thickness of 5 blankets, each with an initial thickness of 3 inches
    and folded according to their color code (1 to 5), is equal to 186 inches. -/
theorem combined_blanket_thickness :
  let initial_thickness : ℝ := 3
  let color_codes : List ℕ := [1, 2, 3, 4, 5]
  let folded_thickness (c : ℕ) : ℝ := initial_thickness * (2 ^ c)
  List.sum (List.map folded_thickness color_codes) = 186 := by
  sorry


end NUMINAMATH_CALUDE_combined_blanket_thickness_l418_41862


namespace NUMINAMATH_CALUDE_stating_price_reduction_achieves_target_profit_l418_41856

/-- Represents the price reduction problem for a product in a shopping mall. -/
structure PriceReductionProblem where
  initialSales : ℕ        -- Initial average daily sales
  initialProfit : ℕ       -- Initial profit per unit
  salesIncrease : ℕ       -- Sales increase per yuan of price reduction
  targetProfit : ℕ        -- Target daily profit
  priceReduction : ℕ      -- Price reduction per unit

/-- 
Theorem stating that the given price reduction achieves the target profit 
for the specified problem parameters.
-/
theorem price_reduction_achieves_target_profit 
  (p : PriceReductionProblem)
  (h1 : p.initialSales = 30)
  (h2 : p.initialProfit = 50)
  (h3 : p.salesIncrease = 2)
  (h4 : p.targetProfit = 2000)
  (h5 : p.priceReduction = 25) :
  (p.initialProfit - p.priceReduction) * (p.initialSales + p.salesIncrease * p.priceReduction) = p.targetProfit :=
by sorry

end NUMINAMATH_CALUDE_stating_price_reduction_achieves_target_profit_l418_41856


namespace NUMINAMATH_CALUDE_parallelogram_area_minimum_l418_41894

theorem parallelogram_area_minimum (z : ℂ) : 
  (∃ (area : ℝ), area = (36:ℝ)/(37:ℝ) ∧ 
    area = 2 * Complex.abs (z * Complex.I * (1/z - z))) →
  (Complex.re z > 0) →
  (Complex.im z < 0) →
  (∃ (d : ℝ), d = Complex.abs (z + 1/z) ∧
    ∀ (w : ℂ), (Complex.re w > 0) → (Complex.im w < 0) →
    (∃ (area : ℝ), area = (36:ℝ)/(37:ℝ) ∧ 
      area = 2 * Complex.abs (w * Complex.I * (1/w - w))) →
    d ≤ Complex.abs (w + 1/w)) →
  (Complex.abs (z + 1/z))^2 = (12:ℝ)/(37:ℝ) :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_area_minimum_l418_41894


namespace NUMINAMATH_CALUDE_brad_age_is_13_l418_41870

def shara_age : ℕ := 10

def jaymee_age : ℕ := 2 * shara_age + 2

def average_age : ℕ := (shara_age + jaymee_age) / 2

def brad_age : ℕ := average_age - 3

theorem brad_age_is_13 : brad_age = 13 := by
  sorry

end NUMINAMATH_CALUDE_brad_age_is_13_l418_41870


namespace NUMINAMATH_CALUDE_largest_four_digit_sum_19_l418_41816

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem largest_four_digit_sum_19 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 19 → n ≤ 8920 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_sum_19_l418_41816


namespace NUMINAMATH_CALUDE_casino_money_theorem_l418_41859

/-- The amount of money on table A -/
def table_a : ℕ := 40

/-- The amount of money on table C -/
def table_c : ℕ := table_a + 20

/-- The amount of money on table B -/
def table_b : ℕ := 2 * table_c

/-- The total amount of money on all tables -/
def total_money : ℕ := table_a + table_b + table_c

theorem casino_money_theorem : total_money = 220 := by
  sorry

end NUMINAMATH_CALUDE_casino_money_theorem_l418_41859


namespace NUMINAMATH_CALUDE_congruence_problem_l418_41848

theorem congruence_problem (y : ℤ) 
  (h1 : (2 + y) % (2^3) = 2^3 % (2^3))
  (h2 : (4 + y) % (4^3) = 2^3 % (4^3))
  (h3 : (6 + y) % (6^3) = 2^3 % (6^3)) :
  y % 24 = 6 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l418_41848


namespace NUMINAMATH_CALUDE_tan_difference_alpha_pi_8_l418_41808

theorem tan_difference_alpha_pi_8 (α : ℝ) (h : 2 * Real.tan α = 3 * Real.tan (π / 8)) :
  Real.tan (α - π / 8) = (1 + 5 * Real.sqrt 2) / 49 := by
  sorry

end NUMINAMATH_CALUDE_tan_difference_alpha_pi_8_l418_41808


namespace NUMINAMATH_CALUDE_decimal_density_between_half_and_seven_tenths_l418_41826

theorem decimal_density_between_half_and_seven_tenths :
  ∃ (x y : ℚ), 0.5 < x ∧ x < y ∧ y < 0.7 :=
sorry

end NUMINAMATH_CALUDE_decimal_density_between_half_and_seven_tenths_l418_41826


namespace NUMINAMATH_CALUDE_ellipse_properties_l418_41839

/-- Prove that an ellipse with given properties has specific semi-major and semi-minor axes -/
theorem ellipse_properties (m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  (∀ x y : ℝ, x^2 / m^2 + y^2 / n^2 = 1) →  -- Ellipse equation
  (∃ c : ℝ, c > 0 ∧ c^2 = m^2 - n^2) →  -- Ellipse property: c^2 = a^2 - b^2
  (2 : ℝ) = m - (m^2 - n^2).sqrt →  -- Right focus at (2, 0)
  (1 / 2 : ℝ) = (m^2 - n^2).sqrt / m →  -- Eccentricity is 1/2
  m^2 = 16 ∧ n^2 = 12 := by
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l418_41839


namespace NUMINAMATH_CALUDE_product_of_fraction_parts_l418_41853

/-- Represents a repeating decimal with a 4-digit repeating sequence -/
def RepeatingDecimal (a b c d : ℕ) : ℚ :=
  (a * 1000 + b * 100 + c * 10 + d) / 9999

/-- The fraction representation of 0.0012 (repeating) -/
def fraction : ℚ := RepeatingDecimal 0 0 1 2

theorem product_of_fraction_parts : ∃ (n d : ℕ), fraction = n / d ∧ Nat.gcd n d = 1 ∧ n * d = 13332 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fraction_parts_l418_41853


namespace NUMINAMATH_CALUDE_donnas_truck_weight_l418_41871

-- Define the given weights and quantities
def bridge_limit : ℕ := 20000
def empty_truck_weight : ℕ := 12000
def soda_crates : ℕ := 20
def soda_crate_weight : ℕ := 50
def dryers : ℕ := 3
def dryer_weight : ℕ := 3000

-- Define the theorem
theorem donnas_truck_weight :
  let soda_weight := soda_crates * soda_crate_weight
  let dryers_weight := dryers * dryer_weight
  let produce_weight := 2 * soda_weight
  empty_truck_weight + soda_weight + dryers_weight + produce_weight = 24000 := by
  sorry

end NUMINAMATH_CALUDE_donnas_truck_weight_l418_41871


namespace NUMINAMATH_CALUDE_geometric_sequence_statements_l418_41881

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

/-- Statement 1: One term of a geometric sequence can be 0. -/
def Statement1 : Prop :=
  ∃ (a : ℕ → ℝ) (n : ℕ), IsGeometricSequence a ∧ a n = 0

/-- Statement 2: The common ratio of a geometric sequence can take any real value. -/
def Statement2 : Prop :=
  ∀ r : ℝ, ∃ a : ℕ → ℝ, IsGeometricSequence a ∧ ∀ n : ℕ, a (n + 1) = r * a n

/-- Statement 3: If b² = ac, then a, b, c form a geometric sequence. -/
def Statement3 : Prop :=
  ∀ a b c : ℝ, b^2 = a * c → ∃ r : ℝ, r ≠ 0 ∧ b = r * a ∧ c = r * b

/-- Statement 4: If a constant sequence is a geometric sequence, then its common ratio is 1. -/
def Statement4 : Prop :=
  ∀ (a : ℕ → ℝ), (∀ n m : ℕ, a n = a m) → IsGeometricSequence a → ∃ r : ℝ, r = 1 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_statements :
  ¬Statement1 ∧ ¬Statement2 ∧ ¬Statement3 ∧ Statement4 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_statements_l418_41881


namespace NUMINAMATH_CALUDE_loan_principal_calculation_l418_41854

/-- Proves that given a loan with 4% annual simple interest over 8 years,
    if the interest is Rs. 306 less than the principal,
    then the principal must be Rs. 450. -/
theorem loan_principal_calculation (P : ℚ) : 
  (P * (4 : ℚ) * (8 : ℚ) / (100 : ℚ) = P - (306 : ℚ)) → P = (450 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_loan_principal_calculation_l418_41854


namespace NUMINAMATH_CALUDE_acclimation_time_is_one_year_l418_41842

/-- Represents the time spent on different phases of PhD study -/
structure PhDTime where
  acclimation : ℝ
  basics : ℝ
  research : ℝ
  dissertation : ℝ

/-- Conditions for John's PhD timeline -/
def johnPhDConditions (t : PhDTime) : Prop :=
  t.basics = 2 ∧
  t.research = 1.75 * t.basics ∧
  t.dissertation = 0.5 * t.acclimation ∧
  t.acclimation + t.basics + t.research + t.dissertation = 7

/-- Theorem stating that under the given conditions, the acclimation time is 1 year -/
theorem acclimation_time_is_one_year (t : PhDTime) 
  (h : johnPhDConditions t) : t.acclimation = 1 := by
  sorry


end NUMINAMATH_CALUDE_acclimation_time_is_one_year_l418_41842


namespace NUMINAMATH_CALUDE_investment_problem_l418_41855

theorem investment_problem (x y T : ℝ) : 
  x + y = T →
  y = 800 →
  0.1 * x - 0.08 * y = 56 →
  T = 2000 := by
sorry

end NUMINAMATH_CALUDE_investment_problem_l418_41855


namespace NUMINAMATH_CALUDE_molar_mass_not_unique_l418_41817

/-- Represents a solution with a solute -/
structure Solution :=
  (mass_fraction : ℝ)
  (mass : ℝ)

/-- Represents the result of mixing two solutions and evaporating water -/
structure MixedSolution :=
  (solution1 : Solution)
  (solution2 : Solution)
  (evaporated_water : ℝ)
  (final_molarity : ℝ)

/-- Function to calculate molar mass given additional information -/
noncomputable def calculate_molar_mass (mixed : MixedSolution) (additional_info : ℝ) : ℝ :=
  sorry

/-- Theorem stating that molar mass cannot be uniquely determined without additional information -/
theorem molar_mass_not_unique (mixed : MixedSolution) :
  ∃ (info1 info2 : ℝ), info1 ≠ info2 ∧ 
  calculate_molar_mass mixed info1 ≠ calculate_molar_mass mixed info2 :=
sorry

end NUMINAMATH_CALUDE_molar_mass_not_unique_l418_41817


namespace NUMINAMATH_CALUDE_max_profit_profit_range_l418_41802

/-- Represents the store's pricing and sales model -/
structure Store where
  costPrice : ℝ
  maxProfitPercent : ℝ
  k : ℝ
  b : ℝ

/-- Calculates the profit given a selling price -/
def profit (s : Store) (x : ℝ) : ℝ :=
  (x - s.costPrice) * (s.k * x + s.b)

/-- Theorem stating the maximum profit and optimal selling price -/
theorem max_profit (s : Store) 
    (h1 : s.costPrice = 60)
    (h2 : s.maxProfitPercent = 0.45)
    (h3 : s.k = -1)
    (h4 : s.b = 120)
    (h5 : s.k * 65 + s.b = 55)
    (h6 : s.k * 75 + s.b = 45) :
    ∃ (maxProfit sellPrice : ℝ),
      maxProfit = 891 ∧
      sellPrice = 87 ∧
      ∀ x, s.costPrice ≤ x ∧ x ≤ s.costPrice * (1 + s.maxProfitPercent) →
        profit s x ≤ maxProfit := by
  sorry

/-- Theorem stating the selling price range for profit ≥ 500 -/
theorem profit_range (s : Store)
    (h1 : s.costPrice = 60)
    (h2 : s.maxProfitPercent = 0.45)
    (h3 : s.k = -1)
    (h4 : s.b = 120)
    (h5 : s.k * 65 + s.b = 55)
    (h6 : s.k * 75 + s.b = 45) :
    ∀ x, profit s x ≥ 500 ∧ s.costPrice ≤ x ∧ x ≤ s.costPrice * (1 + s.maxProfitPercent) →
      70 ≤ x ∧ x ≤ 87 := by
  sorry

end NUMINAMATH_CALUDE_max_profit_profit_range_l418_41802


namespace NUMINAMATH_CALUDE_sqrt_calculations_l418_41809

theorem sqrt_calculations :
  (∀ (x y : ℝ), x > 0 → y > 0 → ∃ (z : ℝ), z > 0 ∧ z * z = x * y) →
  (∀ (x y : ℝ), x > 0 → y > 0 → ∃ (z : ℝ), z > 0 ∧ z * z = x / y) →
  (∃ (sqrt10 sqrt2 sqrt15 sqrt3 sqrt5 sqrt27 sqrt12 sqrt_third : ℝ),
    sqrt10 > 0 ∧ sqrt10 * sqrt10 = 10 ∧
    sqrt2 > 0 ∧ sqrt2 * sqrt2 = 2 ∧
    sqrt15 > 0 ∧ sqrt15 * sqrt15 = 15 ∧
    sqrt3 > 0 ∧ sqrt3 * sqrt3 = 3 ∧
    sqrt5 > 0 ∧ sqrt5 * sqrt5 = 5 ∧
    sqrt27 > 0 ∧ sqrt27 * sqrt27 = 27 ∧
    sqrt12 > 0 ∧ sqrt12 * sqrt12 = 12 ∧
    sqrt_third > 0 ∧ sqrt_third * sqrt_third = 1/3 ∧
    sqrt10 * sqrt2 + sqrt15 / sqrt3 = 3 * sqrt5 ∧
    sqrt27 - (sqrt12 - sqrt_third) = 4/3 * sqrt3) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_calculations_l418_41809


namespace NUMINAMATH_CALUDE_sixth_term_of_arithmetic_sequence_l418_41882

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem sixth_term_of_arithmetic_sequence :
  let a₁ := 2
  let d := 3
  arithmetic_sequence a₁ d 6 = 17 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_arithmetic_sequence_l418_41882


namespace NUMINAMATH_CALUDE_sequence_sum_formula_l418_41843

def sequence_sum (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (List.range n).map a |>.sum

theorem sequence_sum_formula (a : ℕ → ℚ) :
  (∀ n : ℕ, n > 0 → (sequence_sum a n - 1)^2 - a n * (sequence_sum a n - 1) - a n = 0) →
  (∀ n : ℕ, n > 0 → sequence_sum a n = n / (n + 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_formula_l418_41843


namespace NUMINAMATH_CALUDE_leak_empty_time_l418_41865

def tank_capacity : ℝ := 1
def fill_time_no_leak : ℝ := 3
def empty_time_leak : ℝ := 12

theorem leak_empty_time :
  let fill_rate : ℝ := tank_capacity / fill_time_no_leak
  let leak_rate : ℝ := tank_capacity / empty_time_leak
  tank_capacity / leak_rate = empty_time_leak := by
sorry

end NUMINAMATH_CALUDE_leak_empty_time_l418_41865


namespace NUMINAMATH_CALUDE_jeff_wins_three_matches_l418_41858

/-- Calculates the number of matches won given play time in hours, 
    minutes per point, and points needed per match. -/
def matches_won (play_time_hours : ℕ) (minutes_per_point : ℕ) (points_per_match : ℕ) : ℕ :=
  (play_time_hours * 60) / (minutes_per_point * points_per_match)

/-- Theorem stating that playing for 2 hours, scoring a point every 5 minutes, 
    and needing 8 points to win a match results in winning 3 matches. -/
theorem jeff_wins_three_matches : 
  matches_won 2 5 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_jeff_wins_three_matches_l418_41858


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l418_41895

theorem negation_of_existential_proposition :
  (¬ (∃ x : ℝ, 2 * x - 3 > 1)) ↔ (∀ x : ℝ, 2 * x - 3 ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l418_41895


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l418_41880

theorem fraction_equals_zero (x : ℝ) (h : (x - 3) / x = 0) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l418_41880


namespace NUMINAMATH_CALUDE_all_props_true_l418_41810

-- Define the original proposition
def original_prop (x y : ℝ) : Prop := (x * y = 0) → (x = 0 ∨ y = 0)

-- Define the inverse proposition
def inverse_prop (x y : ℝ) : Prop := (x = 0 ∨ y = 0) → (x * y = 0)

-- Define the negation proposition
def negation_prop (x y : ℝ) : Prop := (x * y ≠ 0) → (x ≠ 0 ∧ y ≠ 0)

-- Define the contrapositive proposition
def contrapositive_prop (x y : ℝ) : Prop := (x ≠ 0 ∧ y ≠ 0) → (x * y ≠ 0)

-- Theorem stating that all three derived propositions are true
theorem all_props_true : 
  (∀ x y : ℝ, inverse_prop x y) ∧ 
  (∀ x y : ℝ, negation_prop x y) ∧ 
  (∀ x y : ℝ, contrapositive_prop x y) :=
sorry

end NUMINAMATH_CALUDE_all_props_true_l418_41810


namespace NUMINAMATH_CALUDE_reciprocal_product_equals_19901_l418_41822

-- Define the sequence a_n
def a : ℕ → ℚ
  | 0 => 1
  | 1 => 1
  | (n + 2) => a n / (1 + (n + 1) * a n * a (n + 1))

-- State the theorem
theorem reciprocal_product_equals_19901 :
  1 / (a 190 * a 200) = 19901 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_product_equals_19901_l418_41822
