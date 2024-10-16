import Mathlib

namespace NUMINAMATH_CALUDE_product_equivalence_l1342_134272

theorem product_equivalence (h : 213 * 16 = 3408) : 1.6 * 2.13 = 3.408 := by
  sorry

end NUMINAMATH_CALUDE_product_equivalence_l1342_134272


namespace NUMINAMATH_CALUDE_video_game_collection_cost_l1342_134202

theorem video_game_collection_cost (total_games : ℕ) 
  (games_at_12 : ℕ) (price_12 : ℕ) (price_7 : ℕ) (price_3 : ℕ) :
  total_games = 346 →
  games_at_12 = 80 →
  price_12 = 12 →
  price_7 = 7 →
  price_3 = 3 →
  (games_at_12 * price_12 + 
   ((total_games - games_at_12) / 2) * price_7 + 
   ((total_games - games_at_12) - ((total_games - games_at_12) / 2)) * price_3) = 2290 := by
sorry

#eval 80 * 12 + ((346 - 80) / 2) * 7 + ((346 - 80) - ((346 - 80) / 2)) * 3

end NUMINAMATH_CALUDE_video_game_collection_cost_l1342_134202


namespace NUMINAMATH_CALUDE_triangle_inequality_l1342_134213

theorem triangle_inequality (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_perimeter : a + b + c = 1) : 
  a^2 + b^2 + c^2 + 4*a*b*c < 1/2 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1342_134213


namespace NUMINAMATH_CALUDE_bookstore_shoe_store_sales_coincidence_l1342_134222

def is_multiple_of_5 (n : ℕ) : Prop := ∃ k, n = 5 * k

def shoe_store_sale_day (n : ℕ) : Prop := ∃ k, n = 3 + 6 * k

def july_day (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 31

theorem bookstore_shoe_store_sales_coincidence :
  (∃! d : ℕ, july_day d ∧ is_multiple_of_5 d ∧ shoe_store_sale_day d) := by
  sorry

end NUMINAMATH_CALUDE_bookstore_shoe_store_sales_coincidence_l1342_134222


namespace NUMINAMATH_CALUDE_larger_root_of_equation_l1342_134206

theorem larger_root_of_equation (x : ℚ) : 
  (x - 2/3) * (x - 2/3) + 2 * (x - 2/3) * (x - 4/5) = 0 →
  (x = 2/3 ∨ x = 14/15) ∧ 
  (∀ y, (y - 2/3) * (y - 2/3) + 2 * (y - 2/3) * (y - 4/5) = 0 → y ≤ 14/15) :=
by sorry

#check larger_root_of_equation

end NUMINAMATH_CALUDE_larger_root_of_equation_l1342_134206


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_9_l1342_134215

theorem smallest_three_digit_multiple_of_9 :
  ∀ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 9 ∣ n → n ≥ 108 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_9_l1342_134215


namespace NUMINAMATH_CALUDE_matrix_product_50_l1342_134282

def matrix_product (n : ℕ) : Matrix (Fin 2) (Fin 2) ℕ :=
  (List.range n).foldl
    (fun acc k => acc * !![1, 2*(k+1); 0, 1])
    !![1, 0; 0, 1]

theorem matrix_product_50 :
  matrix_product 50 = !![1, 2550; 0, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_product_50_l1342_134282


namespace NUMINAMATH_CALUDE_find_A_l1342_134274

theorem find_A : ∃ A : ℝ, (12 + 3) * (12 - A) = 120 ∧ A = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l1342_134274


namespace NUMINAMATH_CALUDE_rectangle_area_l1342_134281

/-- The area of a rectangle with length 8m and width 50dm is 40 m² -/
theorem rectangle_area : 
  let length : ℝ := 8
  let width_dm : ℝ := 50
  let width_m : ℝ := width_dm / 10
  let area : ℝ := length * width_m
  area = 40 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1342_134281


namespace NUMINAMATH_CALUDE_A_minus_2B_y_value_when_independent_l1342_134239

-- Define the expressions A and B
def A (x y : ℝ) : ℝ := 4 * x^2 - x * y + 2 * y
def B (x y : ℝ) : ℝ := 2 * x^2 - x * y + x

-- Theorem 1: A - 2B = xy - 2x + 2y
theorem A_minus_2B (x y : ℝ) : A x y - 2 * B x y = x * y - 2 * x + 2 * y := by sorry

-- Theorem 2: If A - 2B is independent of x, then y = 2
theorem y_value_when_independent (y : ℝ) : 
  (∀ x : ℝ, A x y - 2 * B x y = A 0 y - 2 * B 0 y) → y = 2 := by sorry

end NUMINAMATH_CALUDE_A_minus_2B_y_value_when_independent_l1342_134239


namespace NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_3_and_6_l1342_134207

theorem greatest_four_digit_divisible_by_3_and_6 :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 3 = 0 ∧ n % 6 = 0 → n ≤ 9996 :=
by sorry

end NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_3_and_6_l1342_134207


namespace NUMINAMATH_CALUDE_prob_select_boy_is_half_prob_same_gender_is_third_l1342_134268

/-- The number of students in the class -/
def total_students : ℕ := 4

/-- The number of boys in the class -/
def num_boys : ℕ := 2

/-- The number of girls in the class -/
def num_girls : ℕ := 2

/-- The probability of selecting a boy when one student is randomly selected -/
def prob_select_boy : ℚ := num_boys / total_students

/-- The probability of selecting two students of the same gender when two students are randomly selected -/
def prob_same_gender : ℚ := 1 / 3

theorem prob_select_boy_is_half : prob_select_boy = 1 / 2 :=
sorry

theorem prob_same_gender_is_third : prob_same_gender = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_prob_select_boy_is_half_prob_same_gender_is_third_l1342_134268


namespace NUMINAMATH_CALUDE_proposition_range_l1342_134252

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + x > a
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Theorem statement
theorem proposition_range (a : ℝ) : 
  (¬(p a) ∨ q a) = false → -2 < a ∧ a < -1/4 := by
  sorry

end NUMINAMATH_CALUDE_proposition_range_l1342_134252


namespace NUMINAMATH_CALUDE_gcd_13m_plus_4_7m_plus_2_max_l1342_134275

theorem gcd_13m_plus_4_7m_plus_2_max (m : ℕ+) : 
  (Nat.gcd (13 * m.val + 4) (7 * m.val + 2) ≤ 2) ∧ 
  (∃ m : ℕ+, Nat.gcd (13 * m.val + 4) (7 * m.val + 2) = 2) :=
by sorry

end NUMINAMATH_CALUDE_gcd_13m_plus_4_7m_plus_2_max_l1342_134275


namespace NUMINAMATH_CALUDE_range_of_a_when_p_is_false_l1342_134295

theorem range_of_a_when_p_is_false :
  (∀ m : ℝ, -1 ≤ m ∧ m ≤ 1 → a^2 - 5*a + 3 ≥ m + 2) →
  a ∈ Set.Iic 0 ∪ Set.Ici 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_when_p_is_false_l1342_134295


namespace NUMINAMATH_CALUDE_path_1310_to_1315_l1342_134277

/-- Represents a point in the cyclic path --/
def CyclicPoint := ℕ

/-- The length of one cycle in the path --/
def cycleLength : ℕ := 6

/-- Converts a given point to its equivalent position within a cycle --/
def toCyclicPosition (n : ℕ) : CyclicPoint :=
  n % cycleLength

/-- Checks if two points are equivalent in the cyclic representation --/
def areEquivalentPoints (a b : ℕ) : Prop :=
  toCyclicPosition a = toCyclicPosition b

theorem path_1310_to_1315 :
  areEquivalentPoints 1310 2 ∧ 
  areEquivalentPoints 1315 3 ∧
  (1315 - 1310 = cycleLength + 3) := by
  sorry

#check path_1310_to_1315

end NUMINAMATH_CALUDE_path_1310_to_1315_l1342_134277


namespace NUMINAMATH_CALUDE_abc_inequality_l1342_134227

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b * c = 1) :
  (a - 1 + 1/b) * (b - 1 + 1/c) * (c - 1 + 1/a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1342_134227


namespace NUMINAMATH_CALUDE_range_of_a_l1342_134203

theorem range_of_a (a : ℝ) : 
  (∀ t ∈ Set.Ioo 0 2, t / (t^2 + 9) ≤ a ∧ a ≤ (t + 2) / t^2) → 
  a ∈ Set.Icc (2/13) 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1342_134203


namespace NUMINAMATH_CALUDE_mutually_exclusive_but_not_converse_l1342_134220

/-- Represents the outcome of rolling a single die -/
inductive DieOutcome
  | One
  | Two
  | Three
  | Four
  | Five
  | Six

/-- Represents the outcome of rolling two dice -/
def TwoDiceOutcome := DieOutcome × DieOutcome

/-- Checks if a die outcome is odd -/
def isOdd (outcome : DieOutcome) : Bool :=
  match outcome with
  | DieOutcome.One => true
  | DieOutcome.Three => true
  | DieOutcome.Five => true
  | _ => false

/-- Event: Exactly one odd number -/
def exactlyOneOdd (outcome : TwoDiceOutcome) : Prop :=
  (isOdd outcome.1 && !isOdd outcome.2) || (!isOdd outcome.1 && isOdd outcome.2)

/-- Event: Exactly two odd numbers -/
def exactlyTwoOdd (outcome : TwoDiceOutcome) : Prop :=
  isOdd outcome.1 && isOdd outcome.2

/-- The sample space of all possible outcomes when rolling two fair six-sided dice -/
def sampleSpace : Set TwoDiceOutcome := sorry

theorem mutually_exclusive_but_not_converse :
  (∀ (outcome : TwoDiceOutcome), ¬(exactlyOneOdd outcome ∧ exactlyTwoOdd outcome)) ∧
  (∃ (outcome : TwoDiceOutcome), ¬exactlyOneOdd outcome ∧ ¬exactlyTwoOdd outcome) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_but_not_converse_l1342_134220


namespace NUMINAMATH_CALUDE_marys_number_l1342_134217

/-- Represents the scenario described in the problem -/
structure Scenario where
  j : Nat  -- John's number
  m : Nat  -- Mary's number
  sum : Nat := j + m
  product : Nat := j * m

/-- Predicate to check if a number has multiple factorizations -/
def hasMultipleFactorizations (n : Nat) : Prop :=
  ∃ a b c d : Nat, a * b = n ∧ c * d = n ∧ a ≠ c ∧ a ≠ d ∧ a > 1 ∧ b > 1 ∧ c > 1 ∧ d > 1

/-- The main theorem representing the problem -/
theorem marys_number (s : Scenario) : 
  s.product = 2002 ∧ 
  hasMultipleFactorizations 2002 ∧
  (∀ x : Nat, x * s.m = 2002 → hasMultipleFactorizations x) →
  s.m = 1001 := by
  sorry

#eval 1001 * 2  -- Should output 2002

end NUMINAMATH_CALUDE_marys_number_l1342_134217


namespace NUMINAMATH_CALUDE_subject_score_proof_l1342_134273

theorem subject_score_proof (physics chemistry mathematics : ℕ) : 
  (physics + chemistry + mathematics) / 3 = 85 →
  (physics + mathematics) / 2 = 90 →
  (physics + chemistry) / 2 = 70 →
  physics = 65 →
  mathematics = 115 := by
sorry

end NUMINAMATH_CALUDE_subject_score_proof_l1342_134273


namespace NUMINAMATH_CALUDE_smaller_number_proof_l1342_134261

theorem smaller_number_proof (x y : ℝ) 
  (sum_eq : x + y = 16)
  (diff_eq : x - y = 4)
  (prod_eq : x * y = 60) :
  min x y = 6 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l1342_134261


namespace NUMINAMATH_CALUDE_lamp_arrangement_l1342_134244

theorem lamp_arrangement (n : ℕ) (k : ℕ) (h : n = 6 ∧ k = 2) :
  (Finset.range (n - k + 1)).card.choose k = 10 := by
  sorry

end NUMINAMATH_CALUDE_lamp_arrangement_l1342_134244


namespace NUMINAMATH_CALUDE_reflected_light_ray_equation_l1342_134266

/-- Given an incident light ray following the line y = 2x + 1 and reflecting on the line y = x,
    the equation of the reflected light ray is x - 2y - 1 = 0 -/
theorem reflected_light_ray_equation (x y : ℝ) :
  (y = 2*x + 1) →  -- Incident light ray equation
  (y = x) →        -- Reflection line equation
  (x - 2*y - 1 = 0) -- Reflected light ray equation
  := by sorry

end NUMINAMATH_CALUDE_reflected_light_ray_equation_l1342_134266


namespace NUMINAMATH_CALUDE_brandon_gecko_sales_l1342_134204

/-- The number of geckos Brandon sold in the first half of last year -/
def first_half_last_year : ℕ := 46

/-- The number of geckos Brandon sold in the second half of last year -/
def second_half_last_year : ℕ := 55

/-- The number of geckos Brandon sold in the first half two years ago -/
def first_half_two_years_ago : ℕ := 3 * first_half_last_year

/-- The number of geckos Brandon sold in the second half two years ago -/
def second_half_two_years_ago : ℕ := 117

/-- The total number of geckos Brandon sold in the last two years -/
def total_geckos : ℕ := first_half_last_year + second_half_last_year + first_half_two_years_ago + second_half_two_years_ago

theorem brandon_gecko_sales : total_geckos = 356 := by
  sorry

end NUMINAMATH_CALUDE_brandon_gecko_sales_l1342_134204


namespace NUMINAMATH_CALUDE_fourth_grade_students_l1342_134258

theorem fourth_grade_students (initial_students : ℕ) : 
  initial_students + 11 - 5 = 37 → initial_students = 31 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_students_l1342_134258


namespace NUMINAMATH_CALUDE_a_range_l1342_134216

-- Define the function f(x,a)
def f (x a : ℝ) : ℝ := a * x^3 - x^2 + 4*x + 3

-- State the theorem
theorem a_range :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-2) 1 → f x a ≥ 0) → a ∈ Set.Icc (-6) (-2) :=
by sorry

end NUMINAMATH_CALUDE_a_range_l1342_134216


namespace NUMINAMATH_CALUDE_segment_length_sum_l1342_134235

theorem segment_length_sum (a : ℝ) : 
  let point1 := (3 * a, 2 * a - 5)
  let point2 := (5, -2)
  let distance := Real.sqrt ((point1.1 - point2.1)^2 + (point1.2 - point2.2)^2)
  distance = 3 * Real.sqrt 5 →
  ∃ (a1 a2 : ℝ), a1 ≠ a2 ∧ 
    (∀ x : ℝ, Real.sqrt ((3*x - 5)^2 + (2*x - 3)^2) = 3 * Real.sqrt 5 ↔ x = a1 ∨ x = a2) ∧
    a1 + a2 = 3.231 :=
by sorry

end NUMINAMATH_CALUDE_segment_length_sum_l1342_134235


namespace NUMINAMATH_CALUDE_purchase_price_calculation_l1342_134201

/-- Given a markup that includes overhead and net profit, calculate the purchase price. -/
theorem purchase_price_calculation (markup : ℝ) (overhead_rate : ℝ) (net_profit : ℝ) : 
  markup = 35 ∧ overhead_rate = 0.1 ∧ net_profit = 12 →
  ∃ (price : ℝ), price = 230 ∧ markup = overhead_rate * price + net_profit :=
by sorry

end NUMINAMATH_CALUDE_purchase_price_calculation_l1342_134201


namespace NUMINAMATH_CALUDE_rational_function_value_l1342_134246

/-- A rational function with specific properties -/
structure RationalFunction where
  p : ℝ → ℝ
  q : ℝ → ℝ
  p_linear : ∃ a b : ℝ, ∀ x, p x = a * x + b
  q_quadratic : ∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c
  q_roots : q (-4) = 0 ∧ q 1 = 0
  point_zero : p 0 = 0 ∧ q 0 ≠ 0
  point_neg_one : p (-1) / q (-1) = -2

/-- The main theorem -/
theorem rational_function_value (f : RationalFunction) : f.p 2 / f.q 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_value_l1342_134246


namespace NUMINAMATH_CALUDE_laundry_wash_time_l1342_134232

/-- The time it takes to wash clothes in minutes -/
def clothes_time : ℕ := 30

/-- The time it takes to wash towels in minutes -/
def towels_time : ℕ := 2 * clothes_time

/-- The time it takes to wash sheets in minutes -/
def sheets_time : ℕ := towels_time - 15

/-- The total time it takes to wash all laundry in minutes -/
def total_wash_time : ℕ := clothes_time + towels_time + sheets_time

theorem laundry_wash_time : total_wash_time = 135 := by
  sorry

end NUMINAMATH_CALUDE_laundry_wash_time_l1342_134232


namespace NUMINAMATH_CALUDE_log_sum_of_zeros_gt_two_l1342_134299

open Real

/-- Given a function g(x) = ln x - bx, if it has two distinct positive zeros,
    then the sum of their natural logarithms is greater than 2. -/
theorem log_sum_of_zeros_gt_two (b : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : 0 < x₁) (h₂ : 0 < x₂) (h₃ : x₁ ≠ x₂)
  (hz₁ : log x₁ - b * x₁ = 0) (hz₂ : log x₂ - b * x₂ = 0) :
  log x₁ + log x₂ > 2 := by
sorry


end NUMINAMATH_CALUDE_log_sum_of_zeros_gt_two_l1342_134299


namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l1342_134234

theorem lcm_gcf_problem (n m : ℕ+) 
  (h1 : Nat.lcm n m = 56)
  (h2 : Nat.gcd n m = 10)
  (h3 : n = 40) :
  m = 14 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l1342_134234


namespace NUMINAMATH_CALUDE_worm_coverage_l1342_134249

/-- A continuous curve in the plane -/
def ContinuousCurve := Set (ℝ × ℝ)

/-- The length of a continuous curve -/
noncomputable def length (γ : ContinuousCurve) : ℝ := sorry

/-- A semicircle in the plane -/
def Semicircle (center : ℝ × ℝ) (diameter : ℝ) : Set (ℝ × ℝ) := sorry

/-- Whether a set covers another set -/
def covers (A B : Set (ℝ × ℝ)) : Prop := B ⊆ A

theorem worm_coverage (γ : ContinuousCurve) (h : length γ = 1) :
  ∃ (center : ℝ × ℝ), covers (Semicircle center 1) γ := by sorry

end NUMINAMATH_CALUDE_worm_coverage_l1342_134249


namespace NUMINAMATH_CALUDE_betty_beads_l1342_134270

theorem betty_beads (red blue green : ℕ) : 
  (5 * blue = 3 * red) →
  (5 * green = 2 * red) →
  (red = 50) →
  (blue + green = 50) := by
sorry

end NUMINAMATH_CALUDE_betty_beads_l1342_134270


namespace NUMINAMATH_CALUDE_jimmy_flour_amount_l1342_134223

/-- The amount of flour Jimmy bought initially -/
def initial_flour (working_hours : ℕ) (minutes_per_pizza : ℕ) (flour_per_pizza : ℚ) (leftover_pizzas : ℕ) : ℚ :=
  let pizzas_per_hour : ℕ := 60 / minutes_per_pizza
  let total_pizzas : ℕ := working_hours * pizzas_per_hour + leftover_pizzas
  total_pizzas * flour_per_pizza

/-- Theorem stating that Jimmy bought 22 kg of flour initially -/
theorem jimmy_flour_amount :
  initial_flour 7 10 (1/2) 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_flour_amount_l1342_134223


namespace NUMINAMATH_CALUDE_june_maths_books_l1342_134285

/-- The number of maths books June bought -/
def num_maths_books : ℕ := sorry

/-- The total amount June has for school supplies -/
def total_amount : ℕ := 500

/-- The cost of each maths book -/
def maths_book_cost : ℕ := 20

/-- The cost of each science book -/
def science_book_cost : ℕ := 10

/-- The cost of each art book -/
def art_book_cost : ℕ := 20

/-- The amount spent on music books -/
def music_books_cost : ℕ := 160

/-- The total cost of all books -/
def total_cost : ℕ := 
  maths_book_cost * num_maths_books + 
  science_book_cost * (num_maths_books + 6) + 
  art_book_cost * (2 * num_maths_books) + 
  music_books_cost

theorem june_maths_books : 
  num_maths_books = 4 ∧ total_cost = total_amount :=
sorry

end NUMINAMATH_CALUDE_june_maths_books_l1342_134285


namespace NUMINAMATH_CALUDE_additional_friends_for_one_rupee_less_l1342_134292

/-- The number of additional friends needed to reduce each person's share by 1 rupee -/
def additional_friends (total_amount : ℕ) (original_friends : ℕ) : ℕ :=
  let original_share := total_amount / original_friends
  let new_share := original_share - 1
  (total_amount / new_share) - original_friends

theorem additional_friends_for_one_rupee_less :
  additional_friends 100 25 = 8 := by
  sorry

end NUMINAMATH_CALUDE_additional_friends_for_one_rupee_less_l1342_134292


namespace NUMINAMATH_CALUDE_no_integer_solution_for_ten_l1342_134200

theorem no_integer_solution_for_ten :
  ¬ ∃ (x y z : ℤ), 3 * x^2 + 4 * y^2 - 5 * z^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_ten_l1342_134200


namespace NUMINAMATH_CALUDE_weight_measurement_l1342_134286

theorem weight_measurement (n : ℕ) (h : 1 ≤ n ∧ n ≤ 63) :
  ∃ (a₀ a₁ a₂ a₃ a₄ a₅ : Bool),
    n = (if a₀ then 1 else 0) +
        (if a₁ then 2 else 0) +
        (if a₂ then 4 else 0) +
        (if a₃ then 8 else 0) +
        (if a₄ then 16 else 0) +
        (if a₅ then 32 else 0) :=
by sorry

end NUMINAMATH_CALUDE_weight_measurement_l1342_134286


namespace NUMINAMATH_CALUDE_remaining_area_approx_l1342_134229

/-- Represents a circular grass plot with a straight path cutting through it. -/
structure GrassPlot where
  diameter : ℝ
  pathWidth : ℝ
  pathEdgeDistance : ℝ

/-- Calculates the remaining grass area of the plot after the path is cut through. -/
def remainingGrassArea (plot : GrassPlot) : ℝ :=
  sorry

/-- Theorem stating that for the given dimensions, the remaining grass area is approximately 56π + 17 square feet. -/
theorem remaining_area_approx (plot : GrassPlot) 
  (h1 : plot.diameter = 16)
  (h2 : plot.pathWidth = 4)
  (h3 : plot.pathEdgeDistance = 2) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |remainingGrassArea plot - (56 * Real.pi + 17)| < ε :=
sorry

end NUMINAMATH_CALUDE_remaining_area_approx_l1342_134229


namespace NUMINAMATH_CALUDE_even_digits_in_512_base_8_l1342_134284

/-- Represents a natural number in base 8 as a list of digits -/
def BaseEightRepresentation : Type := List Nat

/-- Converts a natural number to its base-8 representation -/
def toBaseEight (n : Nat) : BaseEightRepresentation :=
  sorry

/-- Counts the number of even digits in a base-8 representation -/
def countEvenDigits (rep : BaseEightRepresentation) : Nat :=
  sorry

theorem even_digits_in_512_base_8 :
  countEvenDigits (toBaseEight 512) = 3 := by
  sorry

end NUMINAMATH_CALUDE_even_digits_in_512_base_8_l1342_134284


namespace NUMINAMATH_CALUDE_sin_product_equals_one_eighth_l1342_134242

theorem sin_product_equals_one_eighth :
  Real.sin (12 * π / 180) * Real.sin (36 * π / 180) * Real.sin (54 * π / 180) * Real.sin (84 * π / 180) = 1/8 :=
by sorry

end NUMINAMATH_CALUDE_sin_product_equals_one_eighth_l1342_134242


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1342_134253

theorem min_value_of_expression (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) :
  let A : ℝ × ℝ := (-4, 0)
  let B : ℝ × ℝ := (-1, 0)
  let P : ℝ × ℝ := (a, b)
  (‖P - A‖ = 2 * ‖P - B‖) →
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → 
    let Q : ℝ × ℝ := (x, y)
    (‖Q - A‖ = 2 * ‖Q - B‖) → 
    (4 / a^2 + 1 / b^2 ≤ 4 / x^2 + 1 / y^2)) →
  4 / a^2 + 1 / b^2 = 9/4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1342_134253


namespace NUMINAMATH_CALUDE_nancy_grew_six_potatoes_l1342_134245

/-- The number of potatoes Sandy grew -/
def sandy_potatoes : ℕ := 7

/-- The total number of potatoes Nancy and Sandy grew together -/
def total_potatoes : ℕ := 13

/-- The number of potatoes Nancy grew -/
def nancy_potatoes : ℕ := total_potatoes - sandy_potatoes

theorem nancy_grew_six_potatoes : nancy_potatoes = 6 := by
  sorry

end NUMINAMATH_CALUDE_nancy_grew_six_potatoes_l1342_134245


namespace NUMINAMATH_CALUDE_simplify_expression_l1342_134257

theorem simplify_expression : (2 * 10^9) - (6 * 10^7) / (2 * 10^2) = 1999700000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1342_134257


namespace NUMINAMATH_CALUDE_trig_identity_30_degrees_l1342_134214

theorem trig_identity_30_degrees :
  let tan30 : ℝ := 1 / Real.sqrt 3
  let sin30 : ℝ := 1 / 2
  (tan30^2 - sin30^2) / (tan30^2 * sin30^2) = 1 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_30_degrees_l1342_134214


namespace NUMINAMATH_CALUDE_cube_surface_area_l1342_134298

/-- Given a cube where the sum of all edge lengths is 72 cm, prove its surface area is 216 cm² -/
theorem cube_surface_area (edge_sum : ℝ) (h : edge_sum = 72) : 
  let edge_length := edge_sum / 12
  6 * edge_length^2 = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l1342_134298


namespace NUMINAMATH_CALUDE_three_primes_sum_47_product_1705_l1342_134248

theorem three_primes_sum_47_product_1705 : ∃ p q r : ℕ, 
  Prime p ∧ Prime q ∧ Prime r ∧ 
  p + q + r = 47 ∧ 
  p * q * r = 1705 := by
sorry

end NUMINAMATH_CALUDE_three_primes_sum_47_product_1705_l1342_134248


namespace NUMINAMATH_CALUDE_lailas_test_scores_l1342_134209

theorem lailas_test_scores (first_four_score last_score : ℕ) : 
  (0 ≤ first_four_score ∧ first_four_score ≤ 100) →
  (0 ≤ last_score ∧ last_score ≤ 100) →
  (last_score > first_four_score) →
  ((4 * first_four_score + last_score) / 5 = 82) →
  (∃ possible_scores : Finset ℕ, 
    possible_scores.card = 4 ∧
    last_score ∈ possible_scores ∧
    ∀ s, s ∈ possible_scores → 
      (0 ≤ s ∧ s ≤ 100) ∧
      (∃ x : ℕ, (0 ≤ x ∧ x ≤ 100) ∧ 
                (s > x) ∧ 
                ((4 * x + s) / 5 = 82))) :=
by sorry

end NUMINAMATH_CALUDE_lailas_test_scores_l1342_134209


namespace NUMINAMATH_CALUDE_x_lt_5_necessary_not_sufficient_for_x_lt_2_l1342_134291

theorem x_lt_5_necessary_not_sufficient_for_x_lt_2 :
  (∀ x : ℝ, x < 2 → x < 5) ∧ (∃ x : ℝ, x < 5 ∧ ¬(x < 2)) :=
by sorry

end NUMINAMATH_CALUDE_x_lt_5_necessary_not_sufficient_for_x_lt_2_l1342_134291


namespace NUMINAMATH_CALUDE_f_2_3_neg1_eq_5_3_l1342_134276

-- Define the function f
def f (a b c : ℚ) : ℚ := (a + b) / (a - c)

-- State the theorem
theorem f_2_3_neg1_eq_5_3 : f 2 3 (-1) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_f_2_3_neg1_eq_5_3_l1342_134276


namespace NUMINAMATH_CALUDE_betty_oranges_l1342_134231

theorem betty_oranges (boxes : ℝ) (oranges_per_box : ℕ) :
  boxes = 3.0 → oranges_per_box = 24 → boxes * oranges_per_box = 72 := by
  sorry

end NUMINAMATH_CALUDE_betty_oranges_l1342_134231


namespace NUMINAMATH_CALUDE_system_solution_l1342_134259

theorem system_solution : ∃ (x y z : ℤ), 
  (7*x + 3*y = 2*z + 1) ∧ 
  (4*x - 5*y = 3*z - 30) ∧ 
  (x + 2*y = 5*z + 15) ∧ 
  (x = -1) ∧ (y = 2) ∧ (z = 7) := by
  sorry

#check system_solution

end NUMINAMATH_CALUDE_system_solution_l1342_134259


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l1342_134279

theorem completing_square_equivalence (x : ℝ) : x^2 + 4*x - 3 = 0 ↔ (x + 2)^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l1342_134279


namespace NUMINAMATH_CALUDE_expression_equals_one_l1342_134290

theorem expression_equals_one (x : ℝ) : 
  ((((x + 1)^2 * (x^2 - x + 1)^2) / (x^3 + 1)^2)^2) * 
  ((((x - 1)^2 * (x^2 + x + 1)^2) / (x^3 - 1)^2)^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_equals_one_l1342_134290


namespace NUMINAMATH_CALUDE_expression_evaluation_l1342_134241

theorem expression_evaluation : 
  let x : ℤ := -2
  let expr := (x^2 - 4*x + 4) / (x^2 - 1) / ((x^2 - 2*x) / (x + 1)) + 1 / (x - 1)
  expr = -1 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1342_134241


namespace NUMINAMATH_CALUDE_expand_expression_l1342_134265

theorem expand_expression (x y z : ℝ) :
  (2 * x + 15) * (3 * y + 20 * z + 25) = 6 * x * y + 40 * x * z + 50 * x + 45 * y + 300 * z + 375 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1342_134265


namespace NUMINAMATH_CALUDE_not_divisible_seven_digit_numbers_l1342_134208

def is_seven_digit (n : ℕ) : Prop := 1000000 ≤ n ∧ n ≤ 9999999

def uses_digits_1_to_7 (n : ℕ) : Prop :=
  ∀ d : ℕ, 1 ≤ d ∧ d ≤ 7 → ∃ k : ℕ, n / (10^k) % 10 = d

theorem not_divisible_seven_digit_numbers (A B : ℕ) :
  is_seven_digit A ∧ is_seven_digit B ∧
  uses_digits_1_to_7 A ∧ uses_digits_1_to_7 B ∧
  A ≠ B →
  ¬(∃ k : ℕ, A = k * B) :=
sorry

end NUMINAMATH_CALUDE_not_divisible_seven_digit_numbers_l1342_134208


namespace NUMINAMATH_CALUDE_xy_zero_necessary_not_sufficient_l1342_134293

theorem xy_zero_necessary_not_sufficient :
  (∀ x y : ℝ, x^2 + y^2 = 0 → x * y = 0) ∧
  (∃ x y : ℝ, x * y = 0 ∧ x^2 + y^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_xy_zero_necessary_not_sufficient_l1342_134293


namespace NUMINAMATH_CALUDE_hypotenuse_length_of_special_triangle_l1342_134289

/-- Given a right-angled triangle with side lengths a, b, and c (where c is the hypotenuse),
    if the sum of squares of all sides is 2000 and the perimeter is 60,
    then the hypotenuse length is 10√10. -/
theorem hypotenuse_length_of_special_triangle (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  a^2 + b^2 + c^2 = 2000 →
  a + b + c = 60 →
  c = 10 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_of_special_triangle_l1342_134289


namespace NUMINAMATH_CALUDE_principal_calculation_l1342_134212

/-- Given an interest rate, time period, and a relationship between
    the principal and interest, prove that the principal is 9200. -/
theorem principal_calculation (r t : ℝ) (P : ℝ) :
  r = 0.12 →
  t = 3 →
  P * r * t = P - 5888 →
  P = 9200 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l1342_134212


namespace NUMINAMATH_CALUDE_johnny_money_left_l1342_134255

def savings_september : ℝ := 30
def savings_october : ℝ := 49
def savings_november : ℝ := 46
def savings_december : ℝ := 55
def january_savings_increase : ℝ := 0.15
def video_game_cost : ℝ := 58
def book_cost : ℝ := 25
def birthday_present_cost : ℝ := 40

def total_savings : ℝ :=
  savings_september + savings_october + savings_november + savings_december +
  (savings_december * (1 + january_savings_increase))

def total_expenses : ℝ :=
  video_game_cost + book_cost + birthday_present_cost

theorem johnny_money_left :
  total_savings - total_expenses = 120.25 := by
  sorry

end NUMINAMATH_CALUDE_johnny_money_left_l1342_134255


namespace NUMINAMATH_CALUDE_complex_sum_zero_l1342_134237

theorem complex_sum_zero (b a : ℝ) : 
  let z₁ : ℂ := 2 + b * Complex.I
  let z₂ : ℂ := a + Complex.I
  z₁ + z₂ = 0 → a + b * Complex.I = -2 - Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_sum_zero_l1342_134237


namespace NUMINAMATH_CALUDE_x_plus_y_equals_three_l1342_134205

theorem x_plus_y_equals_three (x y : ℝ) 
  (h1 : |x| + x + 5*y = 2) 
  (h2 : |y| - y + x = 7) : 
  x + y = 3 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_three_l1342_134205


namespace NUMINAMATH_CALUDE_mildred_blocks_l1342_134280

/-- The number of blocks Mildred found -/
def blocks_found (initial final : ℕ) : ℕ := final - initial

/-- Proof that Mildred found 84 blocks -/
theorem mildred_blocks : blocks_found 2 86 = 84 := by
  sorry

end NUMINAMATH_CALUDE_mildred_blocks_l1342_134280


namespace NUMINAMATH_CALUDE_expression_simplification_l1342_134288

theorem expression_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * (a^2)^(1/3) * b^(1/2) * (-6 * a^(1/3) * b^(1/3))^2) / (-3 * (a*b^5)^(1/6)) = -24 * a^(7/6) * b^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1342_134288


namespace NUMINAMATH_CALUDE_series_sum_equality_l1342_134297

/-- Given real numbers c and d satisfying a specific equation, 
    prove that the sum of a certain series equals a specific fraction. -/
theorem series_sum_equality (c d : ℝ) 
    (h : (c / d) / (1 - 1 / d) + (1 / d) / (1 - 1 / d) = 6) :
  c / (c + 2 * d) / (1 - 1 / (c + 2 * d)) = (6 * d - 7) / (8 * (d - 1)) := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equality_l1342_134297


namespace NUMINAMATH_CALUDE_inequality_proof_l1342_134211

theorem inequality_proof (a b x₁ x₂ : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) :
  (a * x₁ + b * x₂) * (a * x₂ + b * x₁) ≥ x₁ * x₂ := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1342_134211


namespace NUMINAMATH_CALUDE_unique_parallel_line_in_plane_l1342_134294

/-- A plane in 3D space -/
structure Plane3D where
  -- (Placeholder for plane definition)

/-- A line in 3D space -/
structure Line3D where
  -- (Placeholder for line definition)

/-- A point in 3D space -/
structure Point3D where
  -- (Placeholder for point definition)

/-- Predicate for a line being parallel to a plane -/
def parallel_line_plane (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- Predicate for a point being on a plane -/
def point_on_plane (P : Point3D) (α : Plane3D) : Prop :=
  sorry

/-- Predicate for a line passing through a point -/
def line_through_point (l : Line3D) (P : Point3D) : Prop :=
  sorry

/-- Predicate for two lines being parallel -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Predicate for a line lying in a plane -/
def line_in_plane (l : Line3D) (α : Plane3D) : Prop :=
  sorry

theorem unique_parallel_line_in_plane 
  (l : Line3D) (α : Plane3D) (P : Point3D)
  (h1 : parallel_line_plane l α)
  (h2 : point_on_plane P α) :
  ∃! m : Line3D, line_through_point m P ∧ parallel_lines m l ∧ line_in_plane m α :=
sorry

end NUMINAMATH_CALUDE_unique_parallel_line_in_plane_l1342_134294


namespace NUMINAMATH_CALUDE_light_glow_duration_l1342_134240

/-- The number of times the light glowed between 1:57:58 and 3:20:47 am -/
def glow_count : ℝ := 292.29411764705884

/-- The total time in seconds between 1:57:58 am and 3:20:47 am -/
def total_time : ℕ := 4969

/-- The duration of each light glow in seconds -/
def glow_duration : ℕ := 17

theorem light_glow_duration :
  Int.floor (total_time / glow_count) = glow_duration := by sorry

end NUMINAMATH_CALUDE_light_glow_duration_l1342_134240


namespace NUMINAMATH_CALUDE_star_four_three_l1342_134233

/-- Definition of the star operation -/
def star (a b : ℤ) : ℤ := a^2 + a*b - b^3

/-- Theorem stating that 4 ⋆ 3 = 1 -/
theorem star_four_three : star 4 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_star_four_three_l1342_134233


namespace NUMINAMATH_CALUDE_nancy_album_pictures_l1342_134224

theorem nancy_album_pictures (total : ℕ) (num_albums : ℕ) (pics_per_album : ℕ) 
  (h1 : total = 51)
  (h2 : num_albums = 8)
  (h3 : pics_per_album = 5) :
  total - (num_albums * pics_per_album) = 11 := by
sorry

end NUMINAMATH_CALUDE_nancy_album_pictures_l1342_134224


namespace NUMINAMATH_CALUDE_discount_calculation_l1342_134221

/-- Given the original cost of plants and the amount actually spent, prove that the discount received is $399.00 -/
theorem discount_calculation (original_cost spent_amount : ℚ) 
  (h1 : original_cost = 467) 
  (h2 : spent_amount = 68) : 
  original_cost - spent_amount = 399 := by
  sorry

end NUMINAMATH_CALUDE_discount_calculation_l1342_134221


namespace NUMINAMATH_CALUDE_smallest_leading_coeff_quadratic_roots_existence_quadratic_roots_five_smallest_leading_coeff_is_five_l1342_134225

theorem smallest_leading_coeff_quadratic_roots (a : ℕ) : 
  (∃ (b c : ℤ) (x₁ x₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    0 < x₁ ∧ x₁ < 1 ∧ 
    0 < x₂ ∧ x₂ < 1 ∧ 
    ∀ (x : ℝ), (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ) = 0 ↔ x = x₁ ∨ x = x₂) →
  a ≥ 5 :=
by sorry

theorem existence_quadratic_roots_five :
  ∃ (b c : ℤ) (x₁ x₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    0 < x₁ ∧ x₁ < 1 ∧ 
    0 < x₂ ∧ x₂ < 1 ∧ 
    ∀ (x : ℝ), (5 : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ) = 0 ↔ x = x₁ ∨ x = x₂ :=
by sorry

theorem smallest_leading_coeff_is_five : 
  ∀ (a : ℕ), 
    (∃ (b c : ℤ) (x₁ x₂ : ℝ), 
      x₁ ≠ x₂ ∧ 
      0 < x₁ ∧ x₁ < 1 ∧ 
      0 < x₂ ∧ x₂ < 1 ∧ 
      ∀ (x : ℝ), (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ) = 0 ↔ x = x₁ ∨ x = x₂) →
    a ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_leading_coeff_quadratic_roots_existence_quadratic_roots_five_smallest_leading_coeff_is_five_l1342_134225


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1342_134243

/-- The speed of a boat in still water, given its downstream and upstream distances in one hour -/
theorem boat_speed_in_still_water 
  (downstream_distance : ℝ) 
  (upstream_distance : ℝ) 
  (h1 : downstream_distance = 7) 
  (h2 : upstream_distance = 5) : 
  ∃ (boat_speed stream_speed : ℝ), 
    boat_speed + stream_speed = downstream_distance ∧ 
    boat_speed - stream_speed = upstream_distance ∧
    boat_speed = 6 :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1342_134243


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1342_134269

theorem complex_modulus_problem : Complex.abs (Complex.I / (1 - Complex.I)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1342_134269


namespace NUMINAMATH_CALUDE_minimizes_f_l1342_134210

/-- The function f(x) defined in the problem -/
def f (a b x : ℝ) : ℝ := 3 * (x - a)^2 + 3 * (x - b)^2

/-- The statement that (a+b)/2 minimizes f(x) -/
theorem minimizes_f (a b : ℝ) :
  ∀ x : ℝ, f a b ((a + b) / 2) ≤ f a b x :=
sorry

end NUMINAMATH_CALUDE_minimizes_f_l1342_134210


namespace NUMINAMATH_CALUDE_rhombus_field_area_l1342_134251

/-- Represents the length of the long diagonal of a rhombus-shaped field in miles. -/
def long_diagonal : ℝ := 2500

/-- Represents the area of the rhombus-shaped field in square miles. -/
def field_area : ℝ := 1562500

/-- Theorem stating that the area of the rhombus-shaped field is 1562500 square miles. -/
theorem rhombus_field_area : field_area = (1 / 2) * long_diagonal * (long_diagonal / 2) := by
  sorry

#check rhombus_field_area

end NUMINAMATH_CALUDE_rhombus_field_area_l1342_134251


namespace NUMINAMATH_CALUDE_sequence_general_term_l1342_134264

theorem sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n : ℕ, S n = 2^n + 3) →
  (a 1 = 5 ∧ ∀ n : ℕ, n ≥ 2 → a n = 2^(n-1)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1342_134264


namespace NUMINAMATH_CALUDE_binomial_expansion_cube_l1342_134263

theorem binomial_expansion_cube (x y : ℝ) : 
  (x + y)^3 = x^3 + 3*x^2*y + 3*x*y^2 + y^3 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_cube_l1342_134263


namespace NUMINAMATH_CALUDE_no_less_equal_two_mo_l1342_134247

theorem no_less_equal_two_mo (N O M : ℝ) (h : N * O ≤ 2 * M * O) : N * O ≤ 2 * M * O := by
  sorry

end NUMINAMATH_CALUDE_no_less_equal_two_mo_l1342_134247


namespace NUMINAMATH_CALUDE_trig_expression_equals_one_l1342_134267

theorem trig_expression_equals_one (d : ℝ) (h : d = 2 * Real.pi / 13) :
  (Real.sin (4 * d) * Real.sin (7 * d) * Real.sin (11 * d) * Real.sin (14 * d) * Real.sin (17 * d)) /
  (Real.sin d * Real.sin (2 * d) * Real.sin (4 * d) * Real.sin (5 * d) * Real.sin (6 * d)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_one_l1342_134267


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1342_134256

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x, x > a → x > 2) ∧ (∃ x, x > 2 ∧ x ≤ a) → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1342_134256


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l1342_134238

def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → f x - f (x + y) = f (x^2 * f y + x)

theorem functional_equation_solutions :
  ∀ f : ℝ → ℝ, (∀ x, x > 0 → f x ≥ 0) →
  FunctionalEquation f →
  (∀ x, x > 0 → f x = 0) ∨ (∀ x, x > 0 → f x = 1 / x) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l1342_134238


namespace NUMINAMATH_CALUDE_triangle_area_345_l1342_134287

theorem triangle_area_345 (a b c : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) :
  (1/2 : ℝ) * a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_345_l1342_134287


namespace NUMINAMATH_CALUDE_laptop_cost_laptop_cost_proof_l1342_134219

/-- The cost of a laptop given the following conditions:
  1. The cost of a smartphone is $400.
  2. Celine buys 2 laptops and 4 smartphones.
  3. Celine pays $3000 and receives $200 in change. -/
theorem laptop_cost : ℕ → Prop :=
  fun laptop_price =>
    let smartphone_price : ℕ := 400
    let laptops_bought : ℕ := 2
    let smartphones_bought : ℕ := 4
    let total_paid : ℕ := 3000
    let change_received : ℕ := 200
    let total_spent : ℕ := total_paid - change_received
    laptop_price * laptops_bought + smartphone_price * smartphones_bought = total_spent ∧
    laptop_price = 600

/-- Proof of the laptop cost theorem -/
theorem laptop_cost_proof : ∃ (x : ℕ), laptop_cost x :=
  sorry

end NUMINAMATH_CALUDE_laptop_cost_laptop_cost_proof_l1342_134219


namespace NUMINAMATH_CALUDE_power_function_increasing_l1342_134278

theorem power_function_increasing (m : ℝ) : 
  (∀ x > 0, Monotone (fun x => (m^2 - 4*m + 1) * x^(m^2 - 2*m - 3))) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_function_increasing_l1342_134278


namespace NUMINAMATH_CALUDE_santa_candy_distribution_l1342_134230

theorem santa_candy_distribution (n : ℕ) (total_candies left_candies : ℕ) :
  3 < n ∧ n < 15 →
  total_candies = 195 →
  left_candies = 8 →
  ∃ k : ℕ, k * n = total_candies - left_candies ∧ k = 17 := by
  sorry

end NUMINAMATH_CALUDE_santa_candy_distribution_l1342_134230


namespace NUMINAMATH_CALUDE_perfect_score_is_21_l1342_134228

/-- The perfect score in a game series, given the number of games and points per round. -/
def perfect_score (num_games : ℕ) (points_per_round : ℕ) : ℕ :=
  num_games * points_per_round

/-- Theorem stating that the perfect score is 21 points when 3 games are played with 7 points per round. -/
theorem perfect_score_is_21 :
  perfect_score 3 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_perfect_score_is_21_l1342_134228


namespace NUMINAMATH_CALUDE_units_digit_of_sum_64_75_base_8_l1342_134254

/-- Represents a number in base 8 --/
def OctalNum := Nat

/-- Converts a base 10 number to its base 8 representation --/
def toOctal (n : Nat) : OctalNum := sorry

/-- Adds two numbers in base 8 --/
def octalAdd (a b : OctalNum) : OctalNum := sorry

/-- Gets the units digit of a number in base 8 --/
def unitsDigit (n : OctalNum) : Nat := sorry

theorem units_digit_of_sum_64_75_base_8 :
  unitsDigit (octalAdd (toOctal 64) (toOctal 75)) = 1 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_64_75_base_8_l1342_134254


namespace NUMINAMATH_CALUDE_problem_statement_l1342_134271

theorem problem_statement (x y n : ℝ) : 
  x = 3 → y = 0 → n = x - y^(x+y) → n = 3 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1342_134271


namespace NUMINAMATH_CALUDE_water_drainage_proof_l1342_134296

/-- Represents the fraction of water remaining after n steps of draining -/
def waterRemaining (n : ℕ) : ℚ :=
  2 / (n + 2 : ℚ)

/-- The number of steps after which one-seventh of the water remains -/
def stepsToOneSeventh : ℕ := 12

theorem water_drainage_proof :
  waterRemaining stepsToOneSeventh = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_water_drainage_proof_l1342_134296


namespace NUMINAMATH_CALUDE_wheel_distance_l1342_134250

/-- The distance covered by a wheel with given radius and number of revolutions -/
theorem wheel_distance (radius : ℝ) (revolutions : ℕ) : 
  radius = Real.sqrt 157 → revolutions = 1000 → 
  2 * Real.pi * radius * revolutions = 78740 := by
  sorry

end NUMINAMATH_CALUDE_wheel_distance_l1342_134250


namespace NUMINAMATH_CALUDE_percentage_difference_l1342_134260

theorem percentage_difference : (0.55 * 40) - (4/5 * 25) = 2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1342_134260


namespace NUMINAMATH_CALUDE_lesser_number_proof_l1342_134283

theorem lesser_number_proof (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : 
  min x y = 25 := by
sorry

end NUMINAMATH_CALUDE_lesser_number_proof_l1342_134283


namespace NUMINAMATH_CALUDE_cube_division_equality_l1342_134226

def cube_edge_lengths : List ℕ := List.range 16

def group1 : List ℕ := [1, 4, 6, 7, 10, 11, 13, 16]
def group2 : List ℕ := [2, 3, 5, 8, 9, 12, 14, 15]

def volume (a : ℕ) : ℕ := a^3
def lateral_surface_area (a : ℕ) : ℕ := 4 * a^2
def edge_length (a : ℕ) : ℕ := 12 * a

theorem cube_division_equality :
  (group1.length = group2.length) ∧
  (group1.sum = group2.sum) ∧
  ((group1.map lateral_surface_area).sum = (group2.map lateral_surface_area).sum) ∧
  ((group1.map volume).sum = (group2.map volume).sum) ∧
  ((group1.map edge_length).sum = (group2.map edge_length).sum) :=
by sorry

end NUMINAMATH_CALUDE_cube_division_equality_l1342_134226


namespace NUMINAMATH_CALUDE_prob_C_is_five_thirtysix_l1342_134236

/-- A spinner with 5 regions A, B, C, D, and E -/
structure Spinner :=
  (probA : ℚ)
  (probB : ℚ)
  (probC : ℚ)
  (probD : ℚ)
  (probE : ℚ)

/-- The properties of the spinner as given in the problem -/
def spinner_properties (s : Spinner) : Prop :=
  s.probA = 5/12 ∧
  s.probB = 1/6 ∧
  s.probC = s.probD ∧
  s.probE = s.probD ∧
  s.probA + s.probB + s.probC + s.probD + s.probE = 1

/-- The theorem stating that the probability of region C is 5/36 -/
theorem prob_C_is_five_thirtysix (s : Spinner) 
  (h : spinner_properties s) : s.probC = 5/36 := by
  sorry

end NUMINAMATH_CALUDE_prob_C_is_five_thirtysix_l1342_134236


namespace NUMINAMATH_CALUDE_math_club_team_selection_l1342_134218

theorem math_club_team_selection (total_boys : Nat) (total_girls : Nat) (team_size : Nat) :
  total_boys = 10 →
  total_girls = 12 →
  team_size = 8 →
  (team_size / 2 : Nat) = 4 →
  Nat.choose total_boys (team_size / 2) * Nat.choose total_girls (team_size / 2) = 103950 := by
  sorry

end NUMINAMATH_CALUDE_math_club_team_selection_l1342_134218


namespace NUMINAMATH_CALUDE_tournament_dominating_set_exists_l1342_134262

/-- Represents a directed graph where vertices are players and edges represent wins. -/
structure TournamentGraph where
  players : Finset ℕ
  wins : players → players → Prop

/-- A tournament graph is complete if every player has played against every other player exactly once. -/
def IsCompleteTournament (g : TournamentGraph) : Prop :=
  ∀ p q : g.players, p ≠ q → (g.wins p q ∨ g.wins q p) ∧ ¬(g.wins p q ∧ g.wins q p)

/-- A set of players dominates the rest if every other player has lost to at least one player in the set. -/
def DominatingSet (g : TournamentGraph) (s : Finset g.players) : Prop :=
  ∀ p : g.players, p ∉ s → ∃ q ∈ s, g.wins q p

theorem tournament_dominating_set_exists (g : TournamentGraph) 
  (h_complete : IsCompleteTournament g) (h_size : g.players.card = 14) :
  ∃ s : Finset g.players, s.card = 3 ∧ DominatingSet g s := by sorry

end NUMINAMATH_CALUDE_tournament_dominating_set_exists_l1342_134262
