import Mathlib

namespace NUMINAMATH_CALUDE_linear_equation_solution_range_l61_6170

theorem linear_equation_solution_range (x k : ℝ) : 
  (2 * x - 5 * k = x + 4) → (x > 0) → (k > -4/5) := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_range_l61_6170


namespace NUMINAMATH_CALUDE_max_value_inequality_l61_6167

theorem max_value_inequality (A : ℝ) (h : A > 0) :
  let M := max (2 + A / 2) (2 * Real.sqrt A)
  ∀ x y : ℝ, x > 0 → y > 0 →
    1 / x + 1 / y + A / (x + y) ≥ M / Real.sqrt (x * y) :=
by sorry

end NUMINAMATH_CALUDE_max_value_inequality_l61_6167


namespace NUMINAMATH_CALUDE_matrix_inverse_proof_l61_6106

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 5; 1, 3]

theorem matrix_inverse_proof :
  A⁻¹ = !![3, -5; -1, 2] := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_proof_l61_6106


namespace NUMINAMATH_CALUDE_ajay_dal_gain_l61_6199

/-- Calculates the total gain from a dal transaction -/
def calculate_gain (quantity1 : ℕ) (price1 : ℚ) (quantity2 : ℕ) (price2 : ℚ) (selling_price : ℚ) : ℚ :=
  let total_cost := quantity1 * price1 + quantity2 * price2
  let total_quantity := quantity1 + quantity2
  let total_revenue := total_quantity * selling_price
  total_revenue - total_cost

/-- Proves that Ajay's total gain in the dal transaction is Rs 27.50 -/
theorem ajay_dal_gain : calculate_gain 15 (14.5) 10 13 15 = (27.5) := by
  sorry

end NUMINAMATH_CALUDE_ajay_dal_gain_l61_6199


namespace NUMINAMATH_CALUDE_determinant_transformation_l61_6157

theorem determinant_transformation (x y z w : ℝ) :
  Matrix.det ![![x, y], ![z, w]] = 6 →
  Matrix.det ![![x, 5*x + 4*y], ![z, 5*z + 4*w]] = 24 := by
  sorry

end NUMINAMATH_CALUDE_determinant_transformation_l61_6157


namespace NUMINAMATH_CALUDE_f_deriv_l61_6196

noncomputable def f (x : ℝ) : ℝ :=
  Real.log (2 * x - 3 + Real.sqrt (4 * x^2 - 12 * x + 10)) -
  Real.sqrt (4 * x^2 - 12 * x + 10) * Real.arctan (2 * x - 3)

theorem f_deriv :
  ∀ x : ℝ, DifferentiableAt ℝ f x →
    deriv f x = - Real.arctan (2 * x - 3) / Real.sqrt (4 * x^2 - 12 * x + 10) :=
by sorry

end NUMINAMATH_CALUDE_f_deriv_l61_6196


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l61_6164

theorem right_triangle_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + b^2 = c^2) : a^4 + b^4 < c^4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l61_6164


namespace NUMINAMATH_CALUDE_triangle_inequalities_l61_6115

theorem triangle_inequalities (a b c A B C : Real) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hA : A > 0) (hB : B > 0) (hC : C > 0)
  (h_triangle : A + B + C = π)
  (h_sides : a = BC ∧ b = AC ∧ c = AB) : 
  (1 / a^3 + 1 / b^3 + 1 / c^3 + a*b*c ≥ 2 * Real.sqrt 3) ∧
  (1 / A + 1 / B + 1 / C ≥ 9 / π) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l61_6115


namespace NUMINAMATH_CALUDE_ratio_problem_l61_6144

theorem ratio_problem (a b : ℝ) (h : a / b = 3 / 2) : (a + b) / a = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l61_6144


namespace NUMINAMATH_CALUDE_complex_set_property_l61_6146

def is_closed_under_multiplication (S : Set ℂ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → (x * y) ∈ S

theorem complex_set_property (a b c d : ℂ) :
  let S : Set ℂ := {a, b, c, d}
  is_closed_under_multiplication S →
  a = 1 →
  b^2 = 1 →
  c^2 = b →
  b + c + d = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_set_property_l61_6146


namespace NUMINAMATH_CALUDE_last_digit_of_189_in_ternary_l61_6108

theorem last_digit_of_189_in_ternary (n : Nat) : n = 189 → n % 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_189_in_ternary_l61_6108


namespace NUMINAMATH_CALUDE_major_axis_length_major_axis_length_is_eight_l61_6156

/-- An ellipse with foci at (3, -4 + 2√3) and (3, -4 - 2√3), tangent to both x and y axes -/
structure TangentEllipse where
  /-- The ellipse is tangent to the x-axis -/
  tangent_x : Bool
  /-- The ellipse is tangent to the y-axis -/
  tangent_y : Bool
  /-- The first focus of the ellipse -/
  focus1 : ℝ × ℝ
  /-- The second focus of the ellipse -/
  focus2 : ℝ × ℝ
  /-- Condition that the first focus is at (3, -4 + 2√3) -/
  h1 : focus1 = (3, -4 + 2 * Real.sqrt 3)
  /-- Condition that the second focus is at (3, -4 - 2√3) -/
  h2 : focus2 = (3, -4 - 2 * Real.sqrt 3)
  /-- Condition that the ellipse is tangent to the x-axis -/
  h3 : tangent_x = true
  /-- Condition that the ellipse is tangent to the y-axis -/
  h4 : tangent_y = true

/-- The length of the major axis of the ellipse is 8 -/
theorem major_axis_length (e : TangentEllipse) : ℝ :=
  8

/-- The theorem stating that the major axis length of the given ellipse is 8 -/
theorem major_axis_length_is_eight (e : TangentEllipse) : 
  major_axis_length e = 8 := by sorry

end NUMINAMATH_CALUDE_major_axis_length_major_axis_length_is_eight_l61_6156


namespace NUMINAMATH_CALUDE_project_hours_difference_l61_6166

theorem project_hours_difference (total_hours : ℕ) 
  (h1 : total_hours = 216) 
  (kate_hours : ℕ) 
  (pat_hours : ℕ) 
  (mark_hours : ℕ) 
  (h2 : pat_hours = 2 * kate_hours) 
  (h3 : pat_hours * 3 = mark_hours) 
  (h4 : kate_hours + pat_hours + mark_hours = total_hours) : 
  mark_hours - kate_hours = 120 := by
sorry

end NUMINAMATH_CALUDE_project_hours_difference_l61_6166


namespace NUMINAMATH_CALUDE_measure_eight_liters_possible_l61_6142

/-- Represents the state of the buckets -/
structure BucketState where
  b10 : ℕ  -- Amount of water in the 10-liter bucket
  b6 : ℕ   -- Amount of water in the 6-liter bucket

/-- Represents a single operation on the buckets -/
inductive BucketOperation
  | FillFromRiver (bucket : ℕ)  -- Fill a bucket from the river
  | EmptyToRiver (bucket : ℕ)   -- Empty a bucket to the river
  | PourBetweenBuckets          -- Pour from one bucket to another

/-- Applies a single operation to the current state -/
def applyOperation (state : BucketState) (op : BucketOperation) : BucketState :=
  sorry

/-- Checks if the given sequence of operations results in 8 liters in one bucket -/
def isValidSolution (operations : List BucketOperation) : Bool :=
  sorry

/-- Theorem stating that it's possible to measure 8 liters using the given buckets -/
theorem measure_eight_liters_possible :
  ∃ (operations : List BucketOperation), isValidSolution operations :=
  sorry

end NUMINAMATH_CALUDE_measure_eight_liters_possible_l61_6142


namespace NUMINAMATH_CALUDE_queen_diamond_probability_l61_6110

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (size : cards.card = 52)
  (valid : ∀ c ∈ cards, c.1 ∈ Finset.range 13 ∧ c.2 ∈ Finset.range 4)

/-- Represents the event of drawing a Queen as the first card -/
def isFirstCardQueen (d : Deck) : Finset (Nat × Nat) :=
  d.cards.filter (λ c => c.1 = 11)

/-- Represents the event of drawing a diamond as the second card -/
def isSecondCardDiamond (d : Deck) : Finset (Nat × Nat) :=
  d.cards.filter (λ c => c.2 = 1)

/-- The main theorem stating the probability of drawing a Queen first and a diamond second -/
theorem queen_diamond_probability (d : Deck) :
  (isFirstCardQueen d).card / d.cards.card *
  (isSecondCardDiamond d).card / (d.cards.card - 1) = 18 / 221 :=
sorry

end NUMINAMATH_CALUDE_queen_diamond_probability_l61_6110


namespace NUMINAMATH_CALUDE_special_arithmetic_progression_all_integer_l61_6197

/-- An arithmetic progression with the property that the product of any two distinct terms is also a term. -/
structure SpecialArithmeticProgression where
  seq : ℕ → ℤ
  is_arithmetic : ∃ d : ℤ, ∀ n : ℕ, seq (n + 1) = seq n + d
  is_increasing : ∀ n : ℕ, seq (n + 1) > seq n
  product_property : ∀ m n : ℕ, m ≠ n → ∃ k : ℕ, seq m * seq n = seq k

/-- All terms in a SpecialArithmeticProgression are integers. -/
theorem special_arithmetic_progression_all_integer (ap : SpecialArithmeticProgression) : 
  ∀ n : ℕ, ∃ k : ℤ, ap.seq n = k :=
sorry

end NUMINAMATH_CALUDE_special_arithmetic_progression_all_integer_l61_6197


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l61_6154

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 1) (h4 : x ≠ 4) (h6 : x ≠ 6) :
  (x^2 - 13) / ((x - 1) * (x - 4) * (x - 6)) =
  (-4/5) / (x - 1) + (-1/2) / (x - 4) + (23/10) / (x - 6) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l61_6154


namespace NUMINAMATH_CALUDE_french_speaking_percentage_l61_6165

theorem french_speaking_percentage (total : ℕ) (french_and_english : ℕ) (french_only : ℕ) 
  (h1 : total = 200)
  (h2 : french_and_english = 10)
  (h3 : french_only = 40) :
  (total - (french_and_english + french_only)) / total * 100 = 75 :=
by sorry

end NUMINAMATH_CALUDE_french_speaking_percentage_l61_6165


namespace NUMINAMATH_CALUDE_dogs_eat_six_cups_l61_6171

/-- Represents the amount of dog food in various units -/
structure DogFood where
  cups : ℚ
  pounds : ℚ

/-- Represents the feeding schedule and food consumption for dogs -/
structure FeedingSchedule where
  dogsCount : ℕ
  feedingsPerDay : ℕ
  daysInMonth : ℕ
  bagsPerMonth : ℕ
  poundsPerBag : ℚ
  cupWeight : ℚ

/-- Calculates the number of cups of dog food each dog eats at a time -/
def cupsPerFeeding (fs : FeedingSchedule) : ℚ :=
  let totalPoundsPerMonth := fs.bagsPerMonth * fs.poundsPerBag
  let poundsPerDogPerMonth := totalPoundsPerMonth / fs.dogsCount
  let feedingsPerMonth := fs.feedingsPerDay * fs.daysInMonth
  let poundsPerFeeding := poundsPerDogPerMonth / feedingsPerMonth
  poundsPerFeeding / fs.cupWeight

/-- Theorem stating that each dog eats 6 cups of dog food at a time -/
theorem dogs_eat_six_cups
  (fs : FeedingSchedule)
  (h1 : fs.dogsCount = 2)
  (h2 : fs.feedingsPerDay = 2)
  (h3 : fs.daysInMonth = 30)
  (h4 : fs.bagsPerMonth = 9)
  (h5 : fs.poundsPerBag = 20)
  (h6 : fs.cupWeight = 1/4) :
  cupsPerFeeding fs = 6 := by
  sorry

#eval cupsPerFeeding {
  dogsCount := 2,
  feedingsPerDay := 2,
  daysInMonth := 30,
  bagsPerMonth := 9,
  poundsPerBag := 20,
  cupWeight := 1/4
}

end NUMINAMATH_CALUDE_dogs_eat_six_cups_l61_6171


namespace NUMINAMATH_CALUDE_evening_ticket_price_l61_6121

/-- The price of a matinee ticket in dollars -/
def matinee_price : ℚ := 5

/-- The price of a 3D ticket in dollars -/
def three_d_price : ℚ := 20

/-- The number of matinee tickets sold -/
def matinee_count : ℕ := 200

/-- The number of evening tickets sold -/
def evening_count : ℕ := 300

/-- The number of 3D tickets sold -/
def three_d_count : ℕ := 100

/-- The total revenue in dollars -/
def total_revenue : ℚ := 6600

/-- The price of an evening ticket in dollars -/
def evening_price : ℚ := 12

theorem evening_ticket_price :
  matinee_price * matinee_count + evening_price * evening_count + three_d_price * three_d_count = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_evening_ticket_price_l61_6121


namespace NUMINAMATH_CALUDE_worker_task_completion_time_l61_6158

/-- Given two workers who can complete a task together in 35 days,
    and one of them can complete the task alone in 84 days,
    prove that the other worker can complete the task alone in 70 days. -/
theorem worker_task_completion_time 
  (total_time : ℝ) 
  (worker1_time : ℝ) 
  (worker2_time : ℝ) 
  (h1 : total_time = 35) 
  (h2 : worker1_time = 84) 
  (h3 : 1 / total_time = 1 / worker1_time + 1 / worker2_time) : 
  worker2_time = 70 := by
  sorry

end NUMINAMATH_CALUDE_worker_task_completion_time_l61_6158


namespace NUMINAMATH_CALUDE_book_profit_calculation_l61_6128

/-- Calculate the overall percent profit for two books with given costs, markups, and discounts -/
theorem book_profit_calculation (cost_a cost_b : ℝ) (markup_a markup_b : ℝ) (discount_a discount_b : ℝ) :
  cost_a = 50 →
  cost_b = 70 →
  markup_a = 0.4 →
  markup_b = 0.6 →
  discount_a = 0.15 →
  discount_b = 0.2 →
  let marked_price_a := cost_a * (1 + markup_a)
  let marked_price_b := cost_b * (1 + markup_b)
  let sale_price_a := marked_price_a * (1 - discount_a)
  let sale_price_b := marked_price_b * (1 - discount_b)
  let total_cost := cost_a + cost_b
  let total_sale_price := sale_price_a + sale_price_b
  let total_profit := total_sale_price - total_cost
  let percent_profit := (total_profit / total_cost) * 100
  percent_profit = 24.25 := by sorry

end NUMINAMATH_CALUDE_book_profit_calculation_l61_6128


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l61_6186

theorem fraction_equation_solution (P Q : ℤ) :
  (∀ x : ℝ, x ≠ -5 ∧ x ≠ 0 ∧ x ≠ 6 →
    (P / (x + 5) + Q / (x * (x - 6)) : ℝ) = (x^2 - 4*x + 20) / (x^3 + x^2 - 30*x)) →
  (Q : ℚ) / P = 4 := by
sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l61_6186


namespace NUMINAMATH_CALUDE_diamonds_in_G10_num_diamonds_formula_num_diamonds_induction_l61_6150

/-- Represents the number of diamonds in figure G_n -/
def num_diamonds (n : ℕ) : ℕ :=
  4 * n^2 + 5 * n - 8

theorem diamonds_in_G10 :
  num_diamonds 10 = 442 :=
by sorry

theorem num_diamonds_formula (n : ℕ) :
  n ≥ 1 →
  num_diamonds n =
    1 + -- initial diamond in G_1
    (4 * (n - 1) * n) + -- diamonds added to sides
    (8 * (n - 1)) -- diamonds added to corners
  :=
by sorry

theorem num_diamonds_induction (n : ℕ) :
  n ≥ 1 →
  num_diamonds n =
    (if n = 1 then 1
     else num_diamonds (n - 1) + 8 * (4 * (n - 1) + 1))
  :=
by sorry

end NUMINAMATH_CALUDE_diamonds_in_G10_num_diamonds_formula_num_diamonds_induction_l61_6150


namespace NUMINAMATH_CALUDE_radius_ef_is_sqrt_136_l61_6120

/-- Triangle DEF with semicircles on its sides -/
structure TriangleWithSemicircles where
  /-- Length of side DE -/
  de : ℝ
  /-- Length of side DF -/
  df : ℝ
  /-- Length of side EF -/
  ef : ℝ
  /-- DEF is a right triangle -/
  right_angle : de^2 + df^2 = ef^2
  /-- Area of semicircle on DE -/
  area_de : (1/2) * Real.pi * (de/2)^2 = 18 * Real.pi
  /-- Arc length of semicircle on DF -/
  arc_df : Real.pi * (df/2) = 10 * Real.pi

/-- The radius of the semicircle on EF is √136 -/
theorem radius_ef_is_sqrt_136 (t : TriangleWithSemicircles) : ef/2 = Real.sqrt 136 := by
  sorry

end NUMINAMATH_CALUDE_radius_ef_is_sqrt_136_l61_6120


namespace NUMINAMATH_CALUDE_divisibility_condition_l61_6188

theorem divisibility_condition (a b : ℕ) (ha : a ≥ 3) (hb : b ≥ 3) :
  (a * b^2 + b + 7 ∣ a^2 * b + a + b) →
  ∃ k : ℕ, k ≥ 1 ∧ a = 7 * k^2 ∧ b = 7 * k :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l61_6188


namespace NUMINAMATH_CALUDE_outfit_count_l61_6131

/-- Represents the number of shirts of each color -/
def shirts_per_color : ℕ := 4

/-- Represents the number of pants -/
def pants : ℕ := 6

/-- Represents the number of hats of each color -/
def hats_per_color : ℕ := 8

/-- Represents the number of colors -/
def colors : ℕ := 3

/-- Theorem: The number of outfits with one shirt, one pair of pants, and one hat,
    where the shirt and hat are not the same color, is 1152 -/
theorem outfit_count : 
  shirts_per_color * (colors - 1) * hats_per_color * pants = 1152 := by
  sorry


end NUMINAMATH_CALUDE_outfit_count_l61_6131


namespace NUMINAMATH_CALUDE_problem_statement_l61_6134

theorem problem_statement (a b : ℝ) (h1 : a - b = 1) (h2 : a * b = -2) :
  (a + 1) * (b - 1) = -4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l61_6134


namespace NUMINAMATH_CALUDE_isosceles_triangle_coordinates_l61_6119

def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (4, 2)

def is_right_angle (p q r : ℝ × ℝ) : Prop :=
  (p.1 - q.1) * (r.1 - q.1) + (p.2 - q.2) * (r.2 - q.2) = 0

def is_isosceles (p q r : ℝ × ℝ) : Prop :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2 = (p.1 - r.1)^2 + (p.2 - r.2)^2

theorem isosceles_triangle_coordinates :
  ∀ B : ℝ × ℝ,
    is_isosceles O A B →
    is_right_angle O B A →
    (B = (1, 3) ∨ B = (3, -1)) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_coordinates_l61_6119


namespace NUMINAMATH_CALUDE_sequence_relation_l61_6148

/-- Given two sequences a and b, where a_n = n^2 and b_n are distinct positive integers,
    and for all n, the a_n-th term of b equals the b_n-th term of a,
    prove that (log(b 1 * b 4 * b 9 * b 16)) / (log(b 1 * b 2 * b 3 * b 4)) = 2 -/
theorem sequence_relation (b : ℕ+ → ℕ+) 
  (h_distinct : ∀ m n : ℕ+, m ≠ n → b m ≠ b n)
  (h_relation : ∀ n : ℕ+, b (n^2) = (b n)^2) :
  (Real.log ((b 1) * (b 4) * (b 9) * (b 16))) / (Real.log ((b 1) * (b 2) * (b 3) * (b 4))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_relation_l61_6148


namespace NUMINAMATH_CALUDE_smallest_difference_l61_6138

def Digits : Finset Nat := {1, 3, 4, 6, 7, 8}

def is_valid_subtraction (a b : Nat) : Prop :=
  a ≥ 1000 ∧ a < 10000 ∧ b ≥ 100 ∧ b < 1000 ∧
  (Digits.card = 6) ∧
  (Finset.card (Finset.filter (λ d => d ∈ Digits) (Finset.range 10)) = 6) ∧
  (∀ d ∈ Digits, (d ∈ a.digits 10 ∨ d ∈ b.digits 10)) ∧
  (∀ d ∈ (a.digits 10 ∪ b.digits 10), d ∈ Digits)

theorem smallest_difference : 
  ∀ a b : Nat, is_valid_subtraction a b → a - b ≥ 473 :=
sorry

end NUMINAMATH_CALUDE_smallest_difference_l61_6138


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l61_6101

/-- If the quadratic equation x^2 + x + m = 0 has two equal real roots,
    then m = 1/4 -/
theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 + x + m = 0 ∧ 
   ∀ y : ℝ, y^2 + y + m = 0 → y = x) → 
  m = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l61_6101


namespace NUMINAMATH_CALUDE_x_minus_y_values_l61_6107

theorem x_minus_y_values (x y : ℝ) (h : y = Real.sqrt (x^2 - 9) - Real.sqrt (9 - x^2) + 4) :
  x - y = -1 ∨ x - y = -7 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_values_l61_6107


namespace NUMINAMATH_CALUDE_inequality_proof_l61_6147

theorem inequality_proof (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  (1 - a) ^ a > (1 - b) ^ b :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l61_6147


namespace NUMINAMATH_CALUDE_parabola_properties_l61_6175

-- Define the parabola function
def f (x : ℝ) : ℝ := (x - 1)^2 - 3

theorem parabola_properties :
  -- 1. The parabola opens upwards
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f ((x₁ + x₂) / 2) < (f x₁ + f x₂) / 2) ∧
  -- 2. The parabola intersects the x-axis at two distinct points
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ∧
  -- 3. The minimum value of y is -3 and occurs when x = 1
  (∀ x : ℝ, f x ≥ -3) ∧ (f 1 = -3) ∧
  -- 4. There exists an x > 1 such that y ≤ 0
  (∃ x : ℝ, x > 1 ∧ f x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l61_6175


namespace NUMINAMATH_CALUDE_fraction_equality_l61_6111

theorem fraction_equality (a b : ℚ) (h : a / 4 = b / 3) : (a - b) / b = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l61_6111


namespace NUMINAMATH_CALUDE_quadratic_max_value_l61_6195

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_max_value 
  (a b c : ℝ) 
  (h1 : f a b c 1 = -40)
  (h2 : f a b c (-1) = -8)
  (h3 : f a b c (-3) = 8)
  (h4 : -b / (2 * a) = -4)
  (h5 : ∃ x₁ x₂, x₁ = -1 ∧ x₂ = -7 ∧ f a b c x₁ = -8 ∧ f a b c x₂ = -8)
  (h6 : a + b + c = -40) :
  ∃ x_max, ∀ x, f a b c x ≤ f a b c x_max ∧ f a b c x_max = 10 :=
sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l61_6195


namespace NUMINAMATH_CALUDE_right_triangle_sine_cosine_l61_6162

theorem right_triangle_sine_cosine (D E F : ℝ) : 
  E = 90 → -- angle E is 90 degrees
  3 * Real.sin D = 4 * Real.cos D → -- given condition
  Real.sin D = 4/5 := by sorry

end NUMINAMATH_CALUDE_right_triangle_sine_cosine_l61_6162


namespace NUMINAMATH_CALUDE_first_expression_equality_second_expression_equality_l61_6185

-- First expression
theorem first_expression_equality (a : ℝ) :
  (-2 * a)^6 * (-3 * a^3) + (2 * a)^2 * 3 = -192 * a^9 + 12 * a^2 := by sorry

-- Second expression
theorem second_expression_equality :
  |(-1/8)| + π^3 + (-1/2)^3 - (1/3)^2 = π^3 - 1/9 := by sorry

end NUMINAMATH_CALUDE_first_expression_equality_second_expression_equality_l61_6185


namespace NUMINAMATH_CALUDE_l2_passes_through_point_perpendicular_implies_a_value_max_distance_to_l1_l61_6172

-- Define the lines l1 and l2
def l1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 3 * a = 0
def l2 (a : ℝ) (x y : ℝ) : Prop := 3 * x + (a - 1) * y + 3 - a = 0

-- Define the point P
def P : ℝ × ℝ := (1, 3)

-- Statement 1
theorem l2_passes_through_point : ∀ a : ℝ, l2 a (-2/3) 1 := by sorry

-- Statement 2
theorem perpendicular_implies_a_value : 
  ∀ a : ℝ, (∀ x y : ℝ, l1 a x y → l2 a x y → (a * 3 + 2 * (a - 1) = 0)) → a = 2/5 := by sorry

-- Statement 3
theorem max_distance_to_l1 : 
  ∀ a : ℝ, ∃ x y : ℝ, l1 a x y ∧ Real.sqrt ((x - P.1)^2 + (y - P.2)^2) = 5 := by sorry

end NUMINAMATH_CALUDE_l2_passes_through_point_perpendicular_implies_a_value_max_distance_to_l1_l61_6172


namespace NUMINAMATH_CALUDE_adjacent_rectangle_area_l61_6141

-- Define the structure of our rectangle
structure DividedRectangle where
  total_length : ℝ
  total_width : ℝ
  length_split : ℝ -- Point where length is split
  width_split : ℝ -- Point where width is split
  inner_split : ℝ -- Point where the largest rectangle is split

-- Define our specific rectangle
def our_rectangle : DividedRectangle where
  total_length := 5
  total_width := 13
  length_split := 3
  width_split := 9
  inner_split := 4

-- Define areas of known rectangles
def area1 : ℝ := 12
def area2 : ℝ := 15
def area3 : ℝ := 20
def area4 : ℝ := 18
def inner_area : ℝ := 8

-- Theorem to prove
theorem adjacent_rectangle_area (r : DividedRectangle) :
  r.length_split * r.inner_split = area3 - inner_area ∧
  (r.total_length - r.length_split) * r.inner_split = inner_area ∧
  (r.total_length - r.length_split) * (r.total_width - r.width_split) = area4 →
  area4 = 18 := by sorry

end NUMINAMATH_CALUDE_adjacent_rectangle_area_l61_6141


namespace NUMINAMATH_CALUDE_similar_triangles_shortest_side_l61_6189

theorem similar_triangles_shortest_side 
  (a b c : ℝ)  -- sides of the first triangle
  (d e f : ℝ)  -- sides of the second triangle
  (h1 : a^2 + b^2 = c^2)  -- first triangle is right-angled
  (h2 : d^2 + e^2 = f^2)  -- second triangle is right-angled
  (h3 : b = 15)  -- given side of first triangle
  (h4 : c = 17)  -- hypotenuse of first triangle
  (h5 : f = 51)  -- hypotenuse of second triangle
  (h6 : (a / d) = (b / e) ∧ (b / e) = (c / f))  -- triangles are similar
  : min d e = 24 := by sorry

end NUMINAMATH_CALUDE_similar_triangles_shortest_side_l61_6189


namespace NUMINAMATH_CALUDE_short_pencil_cost_proof_l61_6159

/-- The cost of a short pencil in dollars -/
def short_pencil_cost : ℚ := 0.4

/-- The cost of a pencil with eraser in dollars -/
def eraser_pencil_cost : ℚ := 0.8

/-- The cost of a regular pencil in dollars -/
def regular_pencil_cost : ℚ := 0.5

/-- The number of pencils with eraser sold -/
def eraser_pencils_sold : ℕ := 200

/-- The number of regular pencils sold -/
def regular_pencils_sold : ℕ := 40

/-- The number of short pencils sold -/
def short_pencils_sold : ℕ := 35

/-- The total revenue from all sales in dollars -/
def total_revenue : ℚ := 194

theorem short_pencil_cost_proof :
  short_pencil_cost * short_pencils_sold +
  eraser_pencil_cost * eraser_pencils_sold +
  regular_pencil_cost * regular_pencils_sold = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_short_pencil_cost_proof_l61_6159


namespace NUMINAMATH_CALUDE_min_value_theorem_l61_6114

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Define the function g
def g (x : ℝ) : ℝ := f x + f (x - 1)

-- State the theorem
theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 2) :
  (m^2 + 2) / m + (n^2 + 1) / n ≥ (7 + 2 * Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l61_6114


namespace NUMINAMATH_CALUDE_sum_coefficients_x_minus_3y_to_20_l61_6149

theorem sum_coefficients_x_minus_3y_to_20 :
  (fun x y => (x - 3 * y) ^ 20) 1 1 = 1048576 := by
  sorry

end NUMINAMATH_CALUDE_sum_coefficients_x_minus_3y_to_20_l61_6149


namespace NUMINAMATH_CALUDE_cube_iff_greater_l61_6137

theorem cube_iff_greater (a b : ℝ) : a > b ↔ a^3 > b^3 := by sorry

end NUMINAMATH_CALUDE_cube_iff_greater_l61_6137


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_l61_6103

theorem quadratic_vertex_form (x : ℝ) :
  ∃ (a h k : ℝ), x^2 - 7*x = a*(x - h)^2 + k ∧ k = -49/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_l61_6103


namespace NUMINAMATH_CALUDE_hiking_team_gloves_l61_6143

/-- The minimum number of gloves needed for a hiking team -/
theorem hiking_team_gloves (num_participants : ℕ) (gloves_per_participant : ℕ) 
  (h1 : num_participants = 63)
  (h2 : gloves_per_participant = 3) : 
  num_participants * gloves_per_participant = 189 := by
  sorry

end NUMINAMATH_CALUDE_hiking_team_gloves_l61_6143


namespace NUMINAMATH_CALUDE_max_value_and_sum_l61_6190

theorem max_value_and_sum (x y z v w : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) (pos_v : 0 < v) (pos_w : 0 < w)
  (sum_sq : x^2 + y^2 + z^2 + v^2 + w^2 = 2016) : 
  ∃ (N x_N y_N z_N v_N w_N : ℝ),
    (∀ (a b c d e : ℝ), 0 < a → 0 < b → 0 < c → 0 < d → 0 < e → 
      a^2 + b^2 + c^2 + d^2 + e^2 = 2016 → 
      4*a*c + 3*b*c + 2*c*d + 4*c*e ≤ N) ∧
    (4*x_N*z_N + 3*y_N*z_N + 2*z_N*v_N + 4*z_N*w_N = N) ∧
    (x_N^2 + y_N^2 + z_N^2 + v_N^2 + w_N^2 = 2016) ∧
    (N + x_N + y_N + z_N + v_N + w_N = 78 + 2028 * Real.sqrt 37) := by
  sorry

end NUMINAMATH_CALUDE_max_value_and_sum_l61_6190


namespace NUMINAMATH_CALUDE_parabola_vertex_l61_6127

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := y = -(x - 2)^2 + 3

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (2, 3)

/-- Theorem: The vertex of the parabola y = -(x - 2)^2 + 3 is (2, 3) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola_equation x y → (x, y) = vertex :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l61_6127


namespace NUMINAMATH_CALUDE_square_sum_lower_bound_l61_6102

theorem square_sum_lower_bound (x y : ℝ) (h : |x - 2*y| = 5) : x^2 + y^2 ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_lower_bound_l61_6102


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l61_6123

def z : ℂ := (3 - Complex.I) * (2 - Complex.I)

theorem z_in_fourth_quadrant : 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l61_6123


namespace NUMINAMATH_CALUDE_sara_marbles_count_l61_6179

/-- The number of black marbles Sara has after receiving marbles from Fred -/
def saras_final_marbles (initial : ℝ) (received : ℝ) : ℝ :=
  initial + received

/-- Theorem: Sara's final number of marbles is 1025.0 -/
theorem sara_marbles_count :
  saras_final_marbles 792.0 233.0 = 1025.0 := by
  sorry

end NUMINAMATH_CALUDE_sara_marbles_count_l61_6179


namespace NUMINAMATH_CALUDE_max_t_is_e_l61_6105

theorem max_t_is_e (t : ℝ) : 
  (∀ a b : ℝ, 0 < a → a < b → b < t → b * Real.log a < a * Real.log b) →
  t ≤ Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_max_t_is_e_l61_6105


namespace NUMINAMATH_CALUDE_equation_solutions_l61_6169

theorem equation_solutions : 
  ∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = 2/3 ∧ 
  (∀ x : ℝ, 2*x - 6 = 3*x*(x - 3) ↔ (x = x₁ ∨ x = x₂)) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l61_6169


namespace NUMINAMATH_CALUDE_existence_of_A_l61_6160

/-- An increasing sequence of positive integers -/
def IncreasingSequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

/-- The growth condition for the sequence -/
def GrowthCondition (a : ℕ → ℕ) (M : ℝ) : Prop :=
  ∀ n : ℕ, 0 < a (n + 1) - a n ∧ (a (n + 1) - a n : ℝ) < M * (a n : ℝ) ^ (5/8)

/-- The main theorem -/
theorem existence_of_A (a : ℕ → ℕ) (M : ℝ) 
    (h_inc : IncreasingSequence a) 
    (h_growth : GrowthCondition a M) :
    ∃ A : ℝ, ∀ k : ℕ, ∃ n : ℕ, ⌊A ^ (3^k)⌋ = a n := by
  sorry

end NUMINAMATH_CALUDE_existence_of_A_l61_6160


namespace NUMINAMATH_CALUDE_lap_time_six_minutes_l61_6177

/-- Represents a circular track with two photographers -/
structure CircularTrack :=
  (length : ℝ)
  (photographer1_position : ℝ)
  (photographer2_position : ℝ)

/-- Represents a runner on the circular track -/
structure Runner :=
  (speed : ℝ)
  (start_position : ℝ)

/-- Calculates the time spent closer to each photographer -/
def time_closer_to_photographer (track : CircularTrack) (runner : Runner) : ℝ × ℝ := sorry

/-- The main theorem to prove -/
theorem lap_time_six_minutes 
  (track : CircularTrack) 
  (runner : Runner) 
  (h1 : (time_closer_to_photographer track runner).1 = 2)
  (h2 : (time_closer_to_photographer track runner).2 = 3) :
  runner.speed * track.length = 6 * runner.speed := by sorry

end NUMINAMATH_CALUDE_lap_time_six_minutes_l61_6177


namespace NUMINAMATH_CALUDE_football_games_indeterminate_l61_6152

theorem football_games_indeterminate 
  (night_games : ℕ) 
  (keith_missed : ℕ) 
  (keith_attended : ℕ) 
  (h1 : night_games = 4) 
  (h2 : keith_missed = 4) 
  (h3 : keith_attended = 4) :
  ¬ ∃ (total_games : ℕ), 
    (total_games ≥ night_games) ∧ 
    (total_games = keith_missed + keith_attended) :=
by sorry

end NUMINAMATH_CALUDE_football_games_indeterminate_l61_6152


namespace NUMINAMATH_CALUDE_pigeon_problem_l61_6130

theorem pigeon_problem (x y : ℕ) : 
  (y + 1 = (1/6) * (x + y + 1)) → 
  (x - 1 = y + 1) → 
  (x = 4 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_pigeon_problem_l61_6130


namespace NUMINAMATH_CALUDE_ryan_overall_percentage_l61_6168

def total_problems : ℕ := 25 + 40 + 10

def correct_problems : ℕ := 
  (25 * 80 / 100) + (40 * 90 / 100) + (10 * 70 / 100)

theorem ryan_overall_percentage : 
  (correct_problems * 100) / total_problems = 84 := by sorry

end NUMINAMATH_CALUDE_ryan_overall_percentage_l61_6168


namespace NUMINAMATH_CALUDE_janet_friday_gym_hours_l61_6122

/-- Janet's weekly gym schedule -/
structure GymSchedule where
  total_hours : ℝ
  monday_hours : ℝ
  wednesday_hours : ℝ
  tuesday_friday_equal : Bool

/-- Theorem: Janet spends 1 hour at the gym on Friday -/
theorem janet_friday_gym_hours (schedule : GymSchedule) 
  (h1 : schedule.total_hours = 5)
  (h2 : schedule.monday_hours = 1.5)
  (h3 : schedule.wednesday_hours = 1.5)
  (h4 : schedule.tuesday_friday_equal = true) :
  ∃ friday_hours : ℝ, friday_hours = 1 ∧ 
  schedule.total_hours = schedule.monday_hours + schedule.wednesday_hours + 2 * friday_hours :=
by
  sorry

end NUMINAMATH_CALUDE_janet_friday_gym_hours_l61_6122


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_six_l61_6140

theorem sum_of_roots_equals_six : 
  let f : ℝ → ℝ := λ x => (x - 3)^2 - 16
  ∃ r₁ r₂ : ℝ, (f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ ≠ r₂) ∧ r₁ + r₂ = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_six_l61_6140


namespace NUMINAMATH_CALUDE_remainder_three_to_89_plus_5_mod_7_l61_6133

theorem remainder_three_to_89_plus_5_mod_7 : (3^89 + 5) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_three_to_89_plus_5_mod_7_l61_6133


namespace NUMINAMATH_CALUDE_phone_number_revenue_l61_6109

theorem phone_number_revenue (X Y : ℕ) : 
  125 * X - 64 * Y = 5 ∧ X < 250 ∧ Y < 250 → 
  (X = 41 ∧ Y = 80) ∨ (X = 105 ∧ Y = 205) :=
sorry

end NUMINAMATH_CALUDE_phone_number_revenue_l61_6109


namespace NUMINAMATH_CALUDE_percentage_calculation_approximation_l61_6126

theorem percentage_calculation_approximation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  abs ((0.47 * 1442 - 0.36 * 1412) + 63 - 232.42) < ε := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_approximation_l61_6126


namespace NUMINAMATH_CALUDE_elbertas_money_l61_6100

/-- Given that Granny Smith has $45, Elberta has $4 more than Anjou, and Anjou has one-fourth as much as Granny Smith, prove that Elberta has $15.25. -/
theorem elbertas_money (granny_smith : ℝ) (elberta anjou : ℝ) 
  (h1 : granny_smith = 45)
  (h2 : elberta = anjou + 4)
  (h3 : anjou = granny_smith / 4) :
  elberta = 15.25 := by
  sorry

end NUMINAMATH_CALUDE_elbertas_money_l61_6100


namespace NUMINAMATH_CALUDE_coin_flip_probability_difference_l61_6173

def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * Nat.factorial (n - k))

def prob_heads (n k : ℕ) : ℚ :=
  (binomial n k : ℚ) * (1 / 2) ^ n

theorem coin_flip_probability_difference : 
  |prob_heads 5 2 - prob_heads 5 4| = 5 / 32 := by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_difference_l61_6173


namespace NUMINAMATH_CALUDE_antonia_pill_box_l61_6116

def pill_box_problem (num_supplements : ℕ) 
                     (num_large_bottles : ℕ) 
                     (num_small_bottles : ℕ) 
                     (pills_per_large_bottle : ℕ) 
                     (pills_per_small_bottle : ℕ) 
                     (pills_left : ℕ) 
                     (num_weeks : ℕ) : Prop :=
  let total_pills := num_large_bottles * pills_per_large_bottle + 
                     num_small_bottles * pills_per_small_bottle
  let pills_used := total_pills - pills_left
  let days_filled := num_weeks * 7
  pills_used / num_supplements = days_filled

theorem antonia_pill_box : 
  pill_box_problem 5 3 2 120 30 350 2 = true :=
sorry

end NUMINAMATH_CALUDE_antonia_pill_box_l61_6116


namespace NUMINAMATH_CALUDE_cos_pi_fourth_plus_alpha_l61_6112

theorem cos_pi_fourth_plus_alpha (α : ℝ) 
  (h : Real.sin (π / 4 - α) = 1 / 2) : 
  Real.cos (π / 4 + α) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_fourth_plus_alpha_l61_6112


namespace NUMINAMATH_CALUDE_sum_of_first_40_digits_of_fraction_l61_6163

-- Define the fraction
def fraction : ℚ := 1 / 1234

-- Define a function to get the nth digit after the decimal point
def nthDigitAfterDecimal (n : ℕ) : ℕ := sorry

-- Define the sum of the first 40 digits after the decimal point
def sumOfFirst40Digits : ℕ := (List.range 40).map nthDigitAfterDecimal |>.sum

-- Theorem statement
theorem sum_of_first_40_digits_of_fraction :
  sumOfFirst40Digits = 218 := by sorry

end NUMINAMATH_CALUDE_sum_of_first_40_digits_of_fraction_l61_6163


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l61_6161

/-- Given a parabola and a hyperbola with an intersection point on the hyperbola's asymptote,
    prove that the hyperbola's eccentricity is √5 under specific conditions. -/
theorem hyperbola_eccentricity (p a b : ℝ) (h1 : p > 0) (h2 : a > 0) (h3 : b > 0) :
  let C₁ := {(x, y) : ℝ × ℝ | y^2 = 2*p*x}
  let C₂ := {(x, y) : ℝ × ℝ | x^2/a^2 - y^2/b^2 = 1}
  let asymptote := {(x, y) : ℝ × ℝ | y = (b/a)*x}
  ∃ A : ℝ × ℝ, A ∈ C₁ ∧ A ∈ C₂ ∧ A ∈ asymptote ∧ 
    (let (x, y) := A
     x - p/2 = p) →
  (Real.sqrt ((a^2 + b^2) / a^2) : ℝ) = Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l61_6161


namespace NUMINAMATH_CALUDE_vertical_distance_theorem_l61_6129

def f (x : ℝ) := |x|
def g (x : ℝ) := -x^2 - 4*x - 3

def solution_set : Set ℝ := {(-5 + Real.sqrt 29)/2, (-5 - Real.sqrt 29)/2, (-3 + Real.sqrt 13)/2, (-3 - Real.sqrt 13)/2}

theorem vertical_distance_theorem :
  ∀ x : ℝ, (f x - g x = 4 ∨ g x - f x = 4) ↔ x ∈ solution_set := by sorry

end NUMINAMATH_CALUDE_vertical_distance_theorem_l61_6129


namespace NUMINAMATH_CALUDE_vector_equality_transitivity_l61_6184

variable {V : Type*} [AddCommGroup V]

theorem vector_equality_transitivity (a b c : V) : a = b → b = c → a = c := by
  sorry

end NUMINAMATH_CALUDE_vector_equality_transitivity_l61_6184


namespace NUMINAMATH_CALUDE_find_k_l61_6135

theorem find_k : ∃ k : ℝ, ∀ x : ℝ, -x^2 - (k + 7)*x - 8 = -(x - 2)*(x - 4) → k = -13 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l61_6135


namespace NUMINAMATH_CALUDE_smallest_solution_quadratic_l61_6117

theorem smallest_solution_quadratic (x : ℝ) :
  (6 * x^2 - 37 * x + 48 = 0) → (x ≥ 13/6) :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_quadratic_l61_6117


namespace NUMINAMATH_CALUDE_f_range_l61_6180

noncomputable def f (x : ℝ) : ℝ := (x^2 + 2*x + 1) / (x + 2)

theorem f_range :
  Set.range f = {y : ℝ | y < 1 ∨ y > 1} :=
sorry

end NUMINAMATH_CALUDE_f_range_l61_6180


namespace NUMINAMATH_CALUDE_abc_inequality_l61_6155

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a * b * c * (a + b + c) ≤ a^3 * b + b^3 * c + c^3 * a := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l61_6155


namespace NUMINAMATH_CALUDE_keith_attended_games_l61_6194

theorem keith_attended_games (total_games missed_games : ℕ) 
  (h1 : total_games = 20)
  (h2 : missed_games = 9) :
  total_games - missed_games = 11 := by
sorry

end NUMINAMATH_CALUDE_keith_attended_games_l61_6194


namespace NUMINAMATH_CALUDE_angle_I_measure_l61_6182

-- Define the pentagon and its angles
structure Pentagon where
  F : ℝ
  G : ℝ
  H : ℝ
  I : ℝ
  J : ℝ

-- Define the properties of the pentagon
def is_valid_pentagon (p : Pentagon) : Prop :=
  p.F > 0 ∧ p.G > 0 ∧ p.H > 0 ∧ p.I > 0 ∧ p.J > 0 ∧
  p.F + p.G + p.H + p.I + p.J = 540

-- Define the conditions given in the problem
def satisfies_conditions (p : Pentagon) : Prop :=
  p.F = p.G ∧ p.G = p.H ∧
  p.I = p.J ∧
  p.I = p.F + 30

-- Theorem statement
theorem angle_I_measure (p : Pentagon) 
  (h1 : is_valid_pentagon p) 
  (h2 : satisfies_conditions p) : 
  p.I = 126 := by
  sorry

end NUMINAMATH_CALUDE_angle_I_measure_l61_6182


namespace NUMINAMATH_CALUDE_power_mod_prime_l61_6139

theorem power_mod_prime (p : Nat) (h : p.Prime) :
  (3 : ZMod p)^2020 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_power_mod_prime_l61_6139


namespace NUMINAMATH_CALUDE_function_equality_l61_6132

/-- Given f(x) = 3x - 5, prove that 2 * [f(1)] - 16 = f(7) -/
theorem function_equality (f : ℝ → ℝ) (h : ∀ x, f x = 3 * x - 5) :
  2 * (f 1) - 16 = f 7 := by sorry

end NUMINAMATH_CALUDE_function_equality_l61_6132


namespace NUMINAMATH_CALUDE_cherries_refund_l61_6136

def grapes_cost : ℚ := 12.08
def total_spent : ℚ := 2.23

theorem cherries_refund :
  grapes_cost - total_spent = 9.85 := by sorry

end NUMINAMATH_CALUDE_cherries_refund_l61_6136


namespace NUMINAMATH_CALUDE_gym_weights_problem_l61_6145

/-- Given the conditions of the gym weights problem, prove that each green weight is 3 pounds. -/
theorem gym_weights_problem (blue_weight : ℕ) (num_blue : ℕ) (num_green : ℕ) (bar_weight : ℕ) (total_weight : ℕ) :
  blue_weight = 2 →
  num_blue = 4 →
  num_green = 5 →
  bar_weight = 2 →
  total_weight = 25 →
  ∃ (green_weight : ℕ), green_weight = 3 ∧ total_weight = blue_weight * num_blue + green_weight * num_green + bar_weight :=
by sorry

end NUMINAMATH_CALUDE_gym_weights_problem_l61_6145


namespace NUMINAMATH_CALUDE_tyson_race_time_l61_6104

/-- Calculates the total time Tyson spent in his races given his swimming speeds and race conditions. -/
theorem tyson_race_time (lake_speed ocean_speed : ℝ) (total_races : ℕ) (race_distance : ℝ) : 
  lake_speed = 3 → 
  ocean_speed = 2.5 → 
  total_races = 10 → 
  race_distance = 3 → 
  (total_races / 2 : ℝ) * (race_distance / lake_speed) + 
  (total_races / 2 : ℝ) * (race_distance / ocean_speed) = 11 := by
  sorry

#check tyson_race_time

end NUMINAMATH_CALUDE_tyson_race_time_l61_6104


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_l61_6192

def a : Fin 2 → ℝ := ![1, 1]
def b : Fin 2 → ℝ := ![1, 2]

theorem perpendicular_vectors_k (k : ℝ) :
  (∀ i : Fin 2, (k * a i - b i) * (b i + a i) = 0) →
  k = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_l61_6192


namespace NUMINAMATH_CALUDE_tree_height_differences_l61_6176

def pine_height : ℚ := 14 + 1/4
def birch_height : ℚ := 18 + 1/2
def cedar_height : ℚ := 20 + 5/8

theorem tree_height_differences :
  (cedar_height - pine_height = 6 + 3/8) ∧
  (cedar_height - birch_height = 2 + 1/8) := by
  sorry

end NUMINAMATH_CALUDE_tree_height_differences_l61_6176


namespace NUMINAMATH_CALUDE_skew_edges_count_l61_6113

/-- Represents a cube in 3D space -/
structure Cube where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a line in 3D space -/
structure Line where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Checks if a line lies on a face of the cube -/
def lineOnFace (c : Cube) (l : Line) : Prop :=
  sorry

/-- Counts the number of edges not in the same plane as the given line -/
def countSkewEdges (c : Cube) (l : Line) : ℕ :=
  sorry

/-- Main theorem: The number of skew edges is either 4, 6, 7, or 8 -/
theorem skew_edges_count (c : Cube) (l : Line) 
  (h : lineOnFace c l) : 
  (countSkewEdges c l = 4) ∨ 
  (countSkewEdges c l = 6) ∨ 
  (countSkewEdges c l = 7) ∨ 
  (countSkewEdges c l = 8) :=
sorry

end NUMINAMATH_CALUDE_skew_edges_count_l61_6113


namespace NUMINAMATH_CALUDE_no_real_solutions_l61_6191

theorem no_real_solutions : ∀ x : ℝ, (2*x - 10*x + 24)^2 + 4 ≠ -2*|x| := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l61_6191


namespace NUMINAMATH_CALUDE_square_area_from_adjacent_points_l61_6187

/-- The area of a square with adjacent vertices at (1,3) and (4,7) is 25 square units. -/
theorem square_area_from_adjacent_points : 
  let p1 : ℝ × ℝ := (1, 3)
  let p2 : ℝ × ℝ := (4, 7)
  let distance := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := distance^2
  area = 25 := by sorry

end NUMINAMATH_CALUDE_square_area_from_adjacent_points_l61_6187


namespace NUMINAMATH_CALUDE_multiply_power_result_l61_6153

theorem multiply_power_result : 112 * (5^4) = 70000 := by
  sorry

end NUMINAMATH_CALUDE_multiply_power_result_l61_6153


namespace NUMINAMATH_CALUDE_sofa_love_seat_cost_l61_6151

/-- The cost of a love seat and sofa, where the sofa costs double the love seat -/
def total_cost (love_seat_cost : ℝ) : ℝ :=
  love_seat_cost + 2 * love_seat_cost

/-- Theorem stating that the total cost is $444 when the love seat costs $148 -/
theorem sofa_love_seat_cost : total_cost 148 = 444 := by
  sorry

end NUMINAMATH_CALUDE_sofa_love_seat_cost_l61_6151


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l61_6178

theorem smallest_dual_base_representation :
  ∃ (n : ℕ) (a b : ℕ), 
    a > 3 ∧ b > 3 ∧
    n = 2 * a + 2 ∧
    n = 3 * b + 3 ∧
    (∀ (m : ℕ) (c d : ℕ), c > 3 → d > 3 → m = 2 * c + 2 → m = 3 * d + 3 → m ≥ n) ∧
    n = 18 := by
  sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l61_6178


namespace NUMINAMATH_CALUDE_sanity_question_suffices_l61_6124

-- Define the types of beings in Transylvania
inductive Being
| Human
| Vampire

-- Define the possible responses to the question
inductive Response
| Yes
| No

-- Define the function that represents how a being responds to the question "Are you sane?"
def respond_to_sanity_question (b : Being) : Response :=
  match b with
  | Being.Human => Response.Yes
  | Being.Vampire => Response.No

-- Define the function that determines the being type based on the response
def determine_being (r : Response) : Being :=
  match r with
  | Response.Yes => Being.Human
  | Response.No => Being.Vampire

-- Theorem: Asking "Are you sane?" is sufficient to determine if a Transylvanian is a human or a vampire
theorem sanity_question_suffices :
  ∀ (b : Being), determine_being (respond_to_sanity_question b) = b :=
by sorry


end NUMINAMATH_CALUDE_sanity_question_suffices_l61_6124


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l61_6183

theorem absolute_value_equation_solution :
  {y : ℝ | |4 * y - 5| = 39} = {11, -8.5} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l61_6183


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l61_6174

theorem purely_imaginary_complex_number (x : ℝ) :
  let z : ℂ := 2 + Complex.I + (1 - Complex.I) * x
  (∃ (y : ℝ), z = Complex.I * y) → x = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l61_6174


namespace NUMINAMATH_CALUDE_power_division_rule_l61_6193

theorem power_division_rule (a : ℝ) : a^7 / a^5 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l61_6193


namespace NUMINAMATH_CALUDE_toothpick_structure_count_l61_6118

/-- Calculates the number of toothpicks in a rectangular grid --/
def rectangle_toothpicks (length width : ℕ) : ℕ :=
  (length + 1) * width + (width + 1) * length

/-- Calculates the number of toothpicks in a right-angled triangle --/
def triangle_toothpicks (base : ℕ) : ℕ :=
  base + (Int.sqrt (2 * base * base)).toNat + 1

/-- The total number of toothpicks in the structure --/
def total_toothpicks (length width : ℕ) : ℕ :=
  rectangle_toothpicks length width + triangle_toothpicks width

theorem toothpick_structure_count :
  total_toothpicks 40 20 = 1709 := by
  sorry

end NUMINAMATH_CALUDE_toothpick_structure_count_l61_6118


namespace NUMINAMATH_CALUDE_f_is_even_f_monotonicity_on_0_1_l61_6181

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 / (x^2 - 1)

-- Theorem for the even property of f
theorem f_is_even (a : ℝ) (ha : a ≠ 0) :
  ∀ x, x ≠ 1 ∧ x ≠ -1 → f a (-x) = f a x :=
sorry

-- Theorem for the monotonicity of f on (0, 1)
theorem f_monotonicity_on_0_1 (a : ℝ) (ha : a ≠ 0) :
  ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 →
    (a > 0 → f a x₁ > f a x₂) ∧
    (a < 0 → f a x₁ < f a x₂) :=
sorry

end NUMINAMATH_CALUDE_f_is_even_f_monotonicity_on_0_1_l61_6181


namespace NUMINAMATH_CALUDE_race_time_calculation_race_problem_l61_6125

theorem race_time_calculation (race_distance : ℝ) (a_time : ℝ) (beat_distance : ℝ) : ℝ :=
  let a_speed := race_distance / a_time
  let b_distance_when_a_finishes := race_distance - beat_distance
  let b_speed := b_distance_when_a_finishes / a_time
  let b_time := race_distance / b_speed
  b_time

theorem race_problem : 
  race_time_calculation 130 20 26 = 25 := by sorry

end NUMINAMATH_CALUDE_race_time_calculation_race_problem_l61_6125


namespace NUMINAMATH_CALUDE_fine_on_fifth_day_l61_6198

/-- Calculates the fine for a given day -/
def dailyFine (previousFine : ℚ) : ℚ :=
  min (previousFine * 2) (previousFine + 0.15)

/-- Calculates the total fine up to a given day -/
def totalFine (day : ℕ) : ℚ :=
  match day with
  | 0 => 0
  | 1 => 0.05
  | n + 1 => totalFine n + dailyFine (dailyFine (totalFine n))

/-- The theorem to be proved -/
theorem fine_on_fifth_day :
  totalFine 5 = 1.35 := by
  sorry

end NUMINAMATH_CALUDE_fine_on_fifth_day_l61_6198
