import Mathlib

namespace NUMINAMATH_CALUDE_min_value_sum_squares_l1018_101845

theorem min_value_sum_squares (x y z : ℝ) (h : 2*x + 3*y + 4*z = 1) :
  ∃ (min : ℝ), min = (1 : ℝ) / 29 ∧ x^2 + y^2 + z^2 ≥ min ∧
  (x^2 + y^2 + z^2 = min ↔ x/2 = y/3 ∧ y/3 = z/4) :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l1018_101845


namespace NUMINAMATH_CALUDE_painted_cubes_l1018_101836

theorem painted_cubes (n : ℕ) (interior_cubes : ℕ) : 
  n = 4 → 
  interior_cubes = 23 → 
  n^3 - interior_cubes = 41 :=
by sorry

end NUMINAMATH_CALUDE_painted_cubes_l1018_101836


namespace NUMINAMATH_CALUDE_area_cyclic_quadrilateral_l1018_101815

/-- The area of a convex cyclic quadrilateral -/
theorem area_cyclic_quadrilateral 
  (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_convex : a + b + c > d ∧ b + c + d > a ∧ c + d + a > b ∧ d + a + b > c) 
  (h_cyclic : ∃ (r : ℝ), r > 0 ∧ 
    a * c = (r + (a^2 / (4*r))) * (r + (c^2 / (4*r))) ∧ 
    b * d = (r + (b^2 / (4*r))) * (r + (d^2 / (4*r)))) :
  let p := (a + b + c + d) / 2
  ∃ (area : ℝ), area = Real.sqrt ((p-a)*(p-b)*(p-c)*(p-d)) := by
  sorry


end NUMINAMATH_CALUDE_area_cyclic_quadrilateral_l1018_101815


namespace NUMINAMATH_CALUDE_harry_total_cost_l1018_101853

-- Define the conversion rate
def silver_per_gold : ℕ := 9

-- Define the costs
def spellbook_cost_gold : ℕ := 5
def potion_kit_cost_silver : ℕ := 20
def owl_cost_gold : ℕ := 28

-- Define the quantities
def num_spellbooks : ℕ := 5
def num_potion_kits : ℕ := 3
def num_owls : ℕ := 1

-- Define the total cost function
def total_cost_silver : ℕ :=
  (num_spellbooks * spellbook_cost_gold * silver_per_gold) +
  (num_potion_kits * potion_kit_cost_silver) +
  (num_owls * owl_cost_gold * silver_per_gold)

-- Theorem statement
theorem harry_total_cost :
  total_cost_silver = 537 :=
by sorry

end NUMINAMATH_CALUDE_harry_total_cost_l1018_101853


namespace NUMINAMATH_CALUDE_jen_work_hours_l1018_101847

/-- 
Given that:
- Jen works 7 hours a week more than Ben
- Jen's work in 4 weeks equals Ben's work in 6 weeks
Prove that Jen works 21 hours per week
-/
theorem jen_work_hours (ben_hours : ℕ) 
  (h1 : ben_hours + 7 = 4 * ben_hours + 28) :
  ben_hours + 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_jen_work_hours_l1018_101847


namespace NUMINAMATH_CALUDE_gcd_three_digit_palindromes_l1018_101887

def three_digit_palindrome (a b : ℕ) : ℕ := 101 * a + 10 * b

theorem gcd_three_digit_palindromes :
  ∃ (d : ℕ), d > 0 ∧
  (∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
    d ∣ three_digit_palindrome a b) ∧
  (∀ (d' : ℕ), d' > d →
    ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧
      ¬(d' ∣ three_digit_palindrome a b)) ∧
  d = 1 :=
by sorry

end NUMINAMATH_CALUDE_gcd_three_digit_palindromes_l1018_101887


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1018_101842

theorem sqrt_equation_solution (t : ℝ) : 
  Real.sqrt (3 * Real.sqrt (2 * t - 1)) = (12 - 2 * t) ^ (1/4) → t = 21/20 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1018_101842


namespace NUMINAMATH_CALUDE_probability_in_standard_deck_l1018_101838

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Nat)
  (diamonds : Nat)
  (hearts : Nat)
  (spades : Nat)

/-- The probability of drawing a diamond, then a heart, then a spade from a standard 52 card deck -/
def probability_diamond_heart_spade (d : Deck) : ℚ :=
  (d.diamonds : ℚ) / d.cards *
  (d.hearts : ℚ) / (d.cards - 1) *
  (d.spades : ℚ) / (d.cards - 2)

/-- A standard 52 card deck -/
def standard_deck : Deck :=
  { cards := 52
  , diamonds := 13
  , hearts := 13
  , spades := 13 }

theorem probability_in_standard_deck :
  probability_diamond_heart_spade standard_deck = 2197 / 132600 :=
by sorry

end NUMINAMATH_CALUDE_probability_in_standard_deck_l1018_101838


namespace NUMINAMATH_CALUDE_triple_base_quadruple_exponent_l1018_101825

theorem triple_base_quadruple_exponent 
  (a b : ℝ) (y : ℝ) (h1 : b ≠ 0) :
  let r := (3 * a) ^ (4 * b)
  r = a ^ b * y ^ b →
  y = 81 * a ^ 3 := by
sorry

end NUMINAMATH_CALUDE_triple_base_quadruple_exponent_l1018_101825


namespace NUMINAMATH_CALUDE_perfect_square_sequence_l1018_101878

theorem perfect_square_sequence (a b : ℤ) :
  (∀ n : ℕ, ∃ k : ℤ, 2^n * a + b = k^2) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sequence_l1018_101878


namespace NUMINAMATH_CALUDE_non_obtuse_triangle_perimeter_gt_four_circumradius_l1018_101808

/-- A triangle with vertices A, B, and C in the real plane. -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The perimeter of a triangle. -/
def perimeter (t : Triangle) : ℝ := sorry

/-- The radius of the circumcircle of a triangle. -/
def circumradius (t : Triangle) : ℝ := sorry

/-- Predicate to check if a triangle is non-obtuse. -/
def is_non_obtuse (t : Triangle) : Prop := sorry

/-- Theorem: For any non-obtuse triangle, its perimeter is greater than
    four times the radius of its circumcircle. -/
theorem non_obtuse_triangle_perimeter_gt_four_circumradius (t : Triangle) :
  is_non_obtuse t → perimeter t > 4 * circumradius t := by sorry

end NUMINAMATH_CALUDE_non_obtuse_triangle_perimeter_gt_four_circumradius_l1018_101808


namespace NUMINAMATH_CALUDE_continuous_piecewise_function_sum_l1018_101814

noncomputable def f (c d : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then c * x + 1
  else if x ≥ -1 then 2 * x - 7
  else 3 * x - d

theorem continuous_piecewise_function_sum (c d : ℝ) :
  Continuous (f c d) → c + d = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_continuous_piecewise_function_sum_l1018_101814


namespace NUMINAMATH_CALUDE_range_of_x_l1018_101837

def f (x : ℝ) : ℝ := 3 * x - 2

def assignment_process (x : ℝ) : ℕ → ℝ
| 0 => x
| n + 1 => f (assignment_process x n)

def process_stops (x : ℝ) (k : ℕ) : Prop :=
  assignment_process x (k - 1) ≤ 244 ∧ assignment_process x k > 244

theorem range_of_x (x : ℝ) (k : ℕ) (h : k > 0) (h_stop : process_stops x k) :
  x ∈ Set.Ioo (3^(5 - k) + 1) (3^(6 - k) + 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_l1018_101837


namespace NUMINAMATH_CALUDE_book_sale_profit_percentage_l1018_101871

/-- Calculates the profit percentage after tax for a book sale -/
theorem book_sale_profit_percentage 
  (cost_price : ℝ) 
  (selling_price : ℝ) 
  (tax_rate : ℝ) 
  (h1 : cost_price = 32) 
  (h2 : selling_price = 56) 
  (h3 : tax_rate = 0.07) : 
  (selling_price * (1 - tax_rate) - cost_price) / cost_price * 100 = 62.75 := by
sorry

end NUMINAMATH_CALUDE_book_sale_profit_percentage_l1018_101871


namespace NUMINAMATH_CALUDE_final_result_proof_l1018_101891

theorem final_result_proof (chosen_number : ℕ) (h : chosen_number = 990) : 
  (chosen_number / 9 : ℚ) - 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_final_result_proof_l1018_101891


namespace NUMINAMATH_CALUDE_cannot_equalize_sugar_l1018_101844

/-- Represents a jar with tea and sugar -/
structure Jar :=
  (volume : ℚ)
  (sugar : ℚ)

/-- Represents the state of all three jars -/
structure JarState :=
  (jar1 : Jar)
  (jar2 : Jar)
  (jar3 : Jar)

/-- Represents a single pouring operation -/
inductive PourOperation
  | pour12 : PourOperation  -- Pour from jar1 to jar2
  | pour13 : PourOperation  -- Pour from jar1 to jar3
  | pour21 : PourOperation  -- Pour from jar2 to jar1
  | pour23 : PourOperation  -- Pour from jar2 to jar3
  | pour31 : PourOperation  -- Pour from jar3 to jar1
  | pour32 : PourOperation  -- Pour from jar3 to jar2

def initialState : JarState :=
  { jar1 := { volume := 0, sugar := 0 },
    jar2 := { volume := 700/1000, sugar := 50 },
    jar3 := { volume := 800/1000, sugar := 60 } }

def measureCup : ℚ := 100/1000

/-- Applies a single pouring operation to the current state -/
def applyOperation (state : JarState) (op : PourOperation) : JarState :=
  sorry

/-- Checks if the sugar content is equal in jars 2 and 3, and jar 1 is empty -/
def isDesiredState (state : JarState) : Prop :=
  state.jar1.volume = 0 ∧ state.jar2.sugar = state.jar3.sugar

/-- The main theorem to prove -/
theorem cannot_equalize_sugar : ¬∃ (ops : List PourOperation),
  isDesiredState (ops.foldl applyOperation initialState) :=
sorry

end NUMINAMATH_CALUDE_cannot_equalize_sugar_l1018_101844


namespace NUMINAMATH_CALUDE_quadratic_equation_one_l1018_101819

theorem quadratic_equation_one (x : ℝ) : (x - 2)^2 = 9 ↔ x = 5 ∨ x = -1 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_one_l1018_101819


namespace NUMINAMATH_CALUDE_car_profit_percent_l1018_101874

/-- Calculates the profit percent from buying, repairing, and selling a car -/
theorem car_profit_percent 
  (purchase_price : ℕ) 
  (mechanical_repairs : ℕ) 
  (bodywork : ℕ) 
  (interior_refurbishment : ℕ) 
  (taxes_and_fees : ℕ) 
  (selling_price : ℕ) 
  (h1 : purchase_price = 48000)
  (h2 : mechanical_repairs = 6000)
  (h3 : bodywork = 4000)
  (h4 : interior_refurbishment = 3000)
  (h5 : taxes_and_fees = 2000)
  (h6 : selling_price = 72900) :
  ∃ (profit_percent : ℚ), 
    abs (profit_percent - 15.71) < 0.01 ∧ 
    profit_percent = (selling_price - (purchase_price + mechanical_repairs + bodywork + interior_refurbishment + taxes_and_fees)) / 
                     (purchase_price + mechanical_repairs + bodywork + interior_refurbishment + taxes_and_fees) * 100 := by
  sorry


end NUMINAMATH_CALUDE_car_profit_percent_l1018_101874


namespace NUMINAMATH_CALUDE_horizontal_chord_theorem_l1018_101882

-- Define the set of valid d values
def ValidD : Set ℝ := {d | ∃ n : ℕ+, d = 1 / n}

theorem horizontal_chord_theorem (f : ℝ → ℝ) (h_cont : Continuous f) (h_end : f 0 = f 1) :
  ∀ d : ℝ, (∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ x + d ∈ Set.Icc 0 1 ∧ f x = f (x + d)) ↔ d ∈ ValidD :=
sorry

end NUMINAMATH_CALUDE_horizontal_chord_theorem_l1018_101882


namespace NUMINAMATH_CALUDE_other_root_of_complex_equation_l1018_101858

theorem other_root_of_complex_equation (z : ℂ) :
  z ^ 2 = -72 + 27 * I →
  (-6 + 3 * I) ^ 2 = -72 + 27 * I →
  ∃ w : ℂ, w ^ 2 = -72 + 27 * I ∧ w ≠ -6 + 3 * I ∧ w = 6 - 3 * I :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_complex_equation_l1018_101858


namespace NUMINAMATH_CALUDE_problem_statement_l1018_101843

theorem problem_statement (k : ℕ) : 
  (18^k : ℕ) ∣ 624938 → 2^k - k^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1018_101843


namespace NUMINAMATH_CALUDE_heath_carrot_planting_rate_l1018_101831

/-- Proves that Heath planted an average of 3000 carrots per hour over the weekend --/
theorem heath_carrot_planting_rate :
  let total_rows : ℕ := 400
  let first_half_rows : ℕ := 200
  let second_half_rows : ℕ := 200
  let plants_per_row_first_half : ℕ := 275
  let plants_per_row_second_half : ℕ := 325
  let hours_first_half : ℕ := 15
  let hours_second_half : ℕ := 25

  let total_plants : ℕ := first_half_rows * plants_per_row_first_half + 
                          second_half_rows * plants_per_row_second_half
  let total_hours : ℕ := hours_first_half + hours_second_half

  (total_plants : ℚ) / (total_hours : ℚ) = 3000 := by
  sorry

end NUMINAMATH_CALUDE_heath_carrot_planting_rate_l1018_101831


namespace NUMINAMATH_CALUDE_linear_equation_solution_l1018_101821

theorem linear_equation_solution (a b : ℝ) : 
  (2 * a + (-1) * b = -1) → (1 + 2 * a - b = 0) := by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l1018_101821


namespace NUMINAMATH_CALUDE_repeating_decimal_subtraction_l1018_101835

/-- Represents a repeating decimal with a 4-digit repetend -/
def RepeatingDecimal (a b c d : ℕ) : ℚ :=
  (a * 1000 + b * 100 + c * 10 + d : ℚ) / 9999

/-- The problem statement -/
theorem repeating_decimal_subtraction :
  RepeatingDecimal 4 5 6 7 - RepeatingDecimal 1 2 3 4 - RepeatingDecimal 2 3 4 5 = 988 / 9999 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_subtraction_l1018_101835


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1018_101848

def U : Set ℕ := {x | x > 0 ∧ x^2 - 9*x + 8 ≤ 0}

def A : Set ℕ := {1, 2, 3}

def B : Set ℕ := {5, 6, 7}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {4, 8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1018_101848


namespace NUMINAMATH_CALUDE_new_average_is_75_l1018_101897

/-- Calculates the new average daily production after adding a new day's production. -/
def new_average_production (past_days : ℕ) (past_average : ℚ) (today_production : ℕ) : ℚ :=
  (past_average * past_days + today_production) / (past_days + 1)

/-- Theorem stating that given the conditions, the new average daily production is 75 units. -/
theorem new_average_is_75 :
  let past_days : ℕ := 3
  let past_average : ℚ := 70
  let today_production : ℕ := 90
  new_average_production past_days past_average today_production = 75 := by
sorry

end NUMINAMATH_CALUDE_new_average_is_75_l1018_101897


namespace NUMINAMATH_CALUDE_sin_plus_sqrt3_cos_l1018_101804

theorem sin_plus_sqrt3_cos (x : ℝ) (h1 : x ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.cos (x + π/12) = Real.sqrt 2 / 10) : 
  Real.sin x + Real.sqrt 3 * Real.cos x = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_sqrt3_cos_l1018_101804


namespace NUMINAMATH_CALUDE_article_original_price_l1018_101841

/-- Given an article with a 25% profit margin, where the profit is 775 rupees, 
    prove that the original price of the article is 3100 rupees. -/
theorem article_original_price (profit_percentage : ℝ) (profit : ℝ) (original_price : ℝ) : 
  profit_percentage = 0.25 →
  profit = 775 →
  profit = profit_percentage * original_price →
  original_price = 3100 :=
by
  sorry

end NUMINAMATH_CALUDE_article_original_price_l1018_101841


namespace NUMINAMATH_CALUDE_apple_orange_pricing_l1018_101800

/-- The price of an orange in dollars -/
def orange_price : ℝ := 2

/-- The price of an apple in dollars -/
def apple_price : ℝ := 3 * orange_price

theorem apple_orange_pricing :
  (4 * apple_price + 7 * orange_price = 38) →
  (orange_price = 2 ∧ 5 * apple_price = 30) := by
  sorry

end NUMINAMATH_CALUDE_apple_orange_pricing_l1018_101800


namespace NUMINAMATH_CALUDE_puzzle_solution_l1018_101810

theorem puzzle_solution : 
  ∀ (S T U K : ℕ),
  (S ≠ T ∧ S ≠ U ∧ S ≠ K ∧ T ≠ U ∧ T ≠ K ∧ U ≠ K) →
  (100 ≤ T * 100 + U * 10 + K ∧ T * 100 + U * 10 + K < 1000) →
  (1000 ≤ S * 1000 + T * 100 + U * 10 + K ∧ S * 1000 + T * 100 + U * 10 + K < 10000) →
  (5 * (T * 100 + U * 10 + K) = S * 1000 + T * 100 + U * 10 + K) →
  (T * 100 + U * 10 + K = 250 ∨ T * 100 + U * 10 + K = 750) := by
sorry

end NUMINAMATH_CALUDE_puzzle_solution_l1018_101810


namespace NUMINAMATH_CALUDE_smallest_max_sum_l1018_101849

theorem smallest_max_sum (a b c d e f : ℕ+) 
  (sum_eq : a + b + c + d + e + f = 3012) : 
  (∃ (M : ℕ), 
    M = max (a+b) (max (b+c) (max (c+d) (max (d+e) (max (e+f) (max (a+c) (b+d)))))) ∧
    (∀ (a' b' c' d' e' f' : ℕ+) (M' : ℕ), 
      a' + b' + c' + d' + e' + f' = 3012 →
      M' = max (a'+b') (max (b'+c') (max (c'+d') (max (d'+e') (max (e'+f') (max (a'+c') (b'+d')))))) →
      M ≤ M')) ∧
  M = 1004 := by
  sorry

end NUMINAMATH_CALUDE_smallest_max_sum_l1018_101849


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1018_101889

theorem complex_fraction_simplification (z : ℂ) (h : z = 1 - I) :
  (z^2 - 2*z) / (z - 1) = -2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1018_101889


namespace NUMINAMATH_CALUDE_angle_AEC_l1018_101866

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ)
  (sum_360 : A + B + C + D = 360)

-- Define the exterior angle bisector point
def exterior_angle_bisector_point (q : Quadrilateral) : ℝ :=
  sorry

-- Theorem statement
theorem angle_AEC (q : Quadrilateral) :
  let E := exterior_angle_bisector_point q
  (360 - (q.B + q.D)) / 2 = E := by
  sorry

end NUMINAMATH_CALUDE_angle_AEC_l1018_101866


namespace NUMINAMATH_CALUDE_cuboid_base_area_l1018_101870

theorem cuboid_base_area (volume : ℝ) (height : ℝ) (base_area : ℝ) :
  volume = 144 →
  height = 8 →
  volume = base_area * height →
  base_area = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_cuboid_base_area_l1018_101870


namespace NUMINAMATH_CALUDE_triangular_numbers_and_squares_l1018_101824

theorem triangular_numbers_and_squares (n a b : ℤ) :
  (n = (a^2 + a)/2 + (b^2 + b)/2) →
  (∃ x y : ℤ, 4*n + 1 = x^2 + y^2 ∧ x = a + b + 1 ∧ y = a - b) ∧
  (∀ x y : ℤ, 4*n + 1 = x^2 + y^2 →
    ∃ a' b' : ℤ, n = (a'^2 + a')/2 + (b'^2 + b')/2) :=
by sorry

end NUMINAMATH_CALUDE_triangular_numbers_and_squares_l1018_101824


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1018_101832

def A : Set ℝ := {x | ∃ y, y = Real.sqrt (3 - x)}
def B : Set ℝ := {1, 2, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1018_101832


namespace NUMINAMATH_CALUDE_max_k_for_exp_inequality_l1018_101894

theorem max_k_for_exp_inequality : 
  (∃ k : ℝ, ∀ x : ℝ, Real.exp x ≥ k * x) ∧ 
  (∀ k : ℝ, (∀ x : ℝ, Real.exp x ≥ k * x) → k ≤ Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_max_k_for_exp_inequality_l1018_101894


namespace NUMINAMATH_CALUDE_convex_number_count_l1018_101872

/-- A function that checks if a three-digit number is convex -/
def isConvex (n : Nat) : Bool :=
  let a₁ := n / 100
  let a₂ := (n / 10) % 10
  let a₃ := n % 10
  100 ≤ n ∧ n < 1000 ∧ a₁ < a₂ ∧ a₃ < a₂

/-- The count of convex numbers -/
def convexCount : Nat :=
  (List.range 1000).filter isConvex |>.length

/-- Theorem stating that the count of convex numbers is 240 -/
theorem convex_number_count : convexCount = 240 := by
  sorry

end NUMINAMATH_CALUDE_convex_number_count_l1018_101872


namespace NUMINAMATH_CALUDE_farm_animals_l1018_101856

theorem farm_animals (horses cows : ℕ) : 
  horses = 5 * cows →                           -- Initial ratio of horses to cows is 5:1
  (horses - 15) = 17 * (cows + 15) / 7 →        -- New ratio after transaction is 17:7
  horses - 15 - (cows + 15) = 50 := by          -- Difference after transaction is 50
sorry

end NUMINAMATH_CALUDE_farm_animals_l1018_101856


namespace NUMINAMATH_CALUDE_combined_capacity_is_forty_l1018_101829

/-- The combined capacity of two buses, each with 1/6 the capacity of a train that holds 120 people. -/
def combined_bus_capacity : ℕ :=
  let train_capacity : ℕ := 120
  let bus_capacity : ℕ := train_capacity / 6
  2 * bus_capacity

/-- Theorem stating that the combined capacity of the two buses is 40 people. -/
theorem combined_capacity_is_forty : combined_bus_capacity = 40 := by
  sorry

end NUMINAMATH_CALUDE_combined_capacity_is_forty_l1018_101829


namespace NUMINAMATH_CALUDE_all_roots_real_l1018_101834

/-- The polynomial x^4 - 4x^3 + 6x^2 - 4x + 1 -/
def p (x : ℝ) : ℝ := x^4 - 4*x^3 + 6*x^2 - 4*x + 1

/-- Theorem stating that all roots of the polynomial are real -/
theorem all_roots_real : ∀ x : ℂ, p x.re = 0 → x.im = 0 := by
  sorry

end NUMINAMATH_CALUDE_all_roots_real_l1018_101834


namespace NUMINAMATH_CALUDE_floor_negative_seven_fourths_l1018_101803

theorem floor_negative_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_negative_seven_fourths_l1018_101803


namespace NUMINAMATH_CALUDE_pictures_from_phone_l1018_101875

-- Define the problem parameters
def num_albums : ℕ := 3
def pics_per_album : ℕ := 2
def camera_pics : ℕ := 4

-- Define the total number of pictures
def total_pics : ℕ := num_albums * pics_per_album

-- Define the number of pictures from the phone
def phone_pics : ℕ := total_pics - camera_pics

-- Theorem statement
theorem pictures_from_phone : phone_pics = 2 := by
  sorry

end NUMINAMATH_CALUDE_pictures_from_phone_l1018_101875


namespace NUMINAMATH_CALUDE_seating_arrangements_with_restriction_l1018_101823

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def circular_permutations (n : ℕ) : ℕ := factorial (n - 1)

def adjacent_arrangements (n : ℕ) : ℕ := 2 * factorial (n - 2)

theorem seating_arrangements_with_restriction (total_people : ℕ) 
  (h1 : total_people = 8) :
  circular_permutations total_people - adjacent_arrangements total_people = 3600 :=
sorry

end NUMINAMATH_CALUDE_seating_arrangements_with_restriction_l1018_101823


namespace NUMINAMATH_CALUDE_smallest_overlap_percentage_l1018_101879

theorem smallest_overlap_percentage (smartphone_users laptop_users : ℝ) 
  (h1 : smartphone_users = 90) 
  (h2 : laptop_users = 80) : 
  ∃ (overlap : ℝ), overlap ≥ 70 ∧ 
    ∀ (x : ℝ), x < 70 → smartphone_users + laptop_users - x > 100 :=
by sorry

end NUMINAMATH_CALUDE_smallest_overlap_percentage_l1018_101879


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l1018_101813

theorem salary_increase_percentage (S : ℝ) (h1 : S + 0.16 * S = 348) (h2 : S + x * S = 375) : x = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_percentage_l1018_101813


namespace NUMINAMATH_CALUDE_football_likers_l1018_101855

theorem football_likers (total : ℕ) (likers : ℕ) (players : ℕ) : 
  (24 : ℚ) / total = (likers : ℚ) / 250 →
  (players : ℚ) / likers = 1 / 2 →
  players = 50 →
  total = 60 := by
sorry

end NUMINAMATH_CALUDE_football_likers_l1018_101855


namespace NUMINAMATH_CALUDE_solution_range_l1018_101802

theorem solution_range (b : ℝ) : 
  let f := fun x : ℝ => x^2 - b*x - 5
  (f (-2) = 5) → 
  (f (-1) = -1) → 
  (f 4 = -1) → 
  (f 5 = 5) → 
  ∀ x : ℝ, f x = 0 → ((-2 < x ∧ x < -1) ∨ (4 < x ∧ x < 5)) :=
by sorry

end NUMINAMATH_CALUDE_solution_range_l1018_101802


namespace NUMINAMATH_CALUDE_encrypted_text_is_cipher_of_problem_statement_l1018_101862

/-- Represents a character in the Russian alphabet -/
inductive RussianChar : Type
| vowel : RussianChar
| consonant : RussianChar

/-- Represents a string of Russian characters -/
def RussianString := List RussianChar

/-- The tarabar cipher function -/
def tarabarCipher : RussianString → RussianString := sorry

/-- The first sentence of the problem statement -/
def problemStatement : RussianString := sorry

/-- The given encrypted text -/
def encryptedText : RussianString := sorry

/-- Theorem stating that the encrypted text is a cipher of the problem statement -/
theorem encrypted_text_is_cipher_of_problem_statement :
  tarabarCipher problemStatement = encryptedText := by sorry

end NUMINAMATH_CALUDE_encrypted_text_is_cipher_of_problem_statement_l1018_101862


namespace NUMINAMATH_CALUDE_pastries_sold_l1018_101888

def initial_pastries : ℕ := 148
def remaining_pastries : ℕ := 45

theorem pastries_sold : initial_pastries - remaining_pastries = 103 := by
  sorry

end NUMINAMATH_CALUDE_pastries_sold_l1018_101888


namespace NUMINAMATH_CALUDE_decagon_ratio_l1018_101884

/-- Represents a decagon with specific properties -/
structure Decagon where
  unit_squares : ℕ
  triangles : ℕ
  triangle_base : ℝ
  bottom_square : ℕ
  bottom_area : ℝ

/-- Theorem statement for the decagon problem -/
theorem decagon_ratio 
  (d : Decagon)
  (h1 : d.unit_squares = 12)
  (h2 : d.triangles = 2)
  (h3 : d.triangle_base = 3)
  (h4 : d.bottom_square = 1)
  (h5 : d.bottom_area = 6)
  : ∃ (xq yq : ℝ), xq / yq = 1 ∧ xq + yq = 3 := by
  sorry

end NUMINAMATH_CALUDE_decagon_ratio_l1018_101884


namespace NUMINAMATH_CALUDE_building_floors_l1018_101895

theorem building_floors (floors_B floors_C : ℕ) : 
  (floors_C = 5 * floors_B - 6) →
  (floors_C = 59) →
  (∃ floors_A : ℕ, floors_A = floors_B - 9 ∧ floors_A = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_building_floors_l1018_101895


namespace NUMINAMATH_CALUDE_triangle_third_vertex_l1018_101817

/-- Given a triangle with vertices (4, 3), (0, 0), and (x, 0) where x < 0,
    if the area of the triangle is 24 square units, then x = -16. -/
theorem triangle_third_vertex (x : ℝ) (h1 : x < 0) :
  (1/2 : ℝ) * abs x * 3 = 24 → x = -16 := by sorry

end NUMINAMATH_CALUDE_triangle_third_vertex_l1018_101817


namespace NUMINAMATH_CALUDE_exists_perfect_square_with_digit_sum_2011_l1018_101812

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a perfect square with sum of digits 2011 -/
theorem exists_perfect_square_with_digit_sum_2011 : 
  ∃ n : ℕ, sum_of_digits (n^2) = 2011 := by
sorry

end NUMINAMATH_CALUDE_exists_perfect_square_with_digit_sum_2011_l1018_101812


namespace NUMINAMATH_CALUDE_three_player_rotation_l1018_101857

/-- Represents the number of games played by each player in a three-player table tennis rotation. -/
structure GameCount where
  player1 : ℕ
  player2 : ℕ
  player3 : ℕ

/-- 
Theorem: In a three-player table tennis rotation where the losing player is replaced by the non-participating player,
if Player 1 played 10 games and Player 2 played 21 games, then Player 3 must have played 11 games.
-/
theorem three_player_rotation (gc : GameCount) 
  (h1 : gc.player1 = 10)
  (h2 : gc.player2 = 21)
  (h_total : gc.player1 + gc.player2 + gc.player3 = 2 * gc.player2) :
  gc.player3 = 11 := by
  sorry


end NUMINAMATH_CALUDE_three_player_rotation_l1018_101857


namespace NUMINAMATH_CALUDE_elevator_weight_problem_l1018_101826

/-- Given 6 people in an elevator with an average weight of 156 lbs,
    if a 7th person enters and the new average weight becomes 151 lbs,
    then the weight of the 7th person is 121 lbs. -/
theorem elevator_weight_problem (initial_people : ℕ) (initial_avg_weight new_avg_weight : ℝ) :
  initial_people = 6 →
  initial_avg_weight = 156 →
  new_avg_weight = 151 →
  (initial_people * initial_avg_weight + (initial_people + 1) * new_avg_weight - initial_people * new_avg_weight) = 121 :=
by sorry

end NUMINAMATH_CALUDE_elevator_weight_problem_l1018_101826


namespace NUMINAMATH_CALUDE_rectangle_area_l1018_101890

-- Define the rectangle ABCD
def rectangle (AB DE : ℝ) : Prop :=
  DE - AB = 9 ∧ DE > AB ∧ AB > 0

-- Define the relationship between areas of trapezoid ABCE and triangle ADE
def area_relation (AB DE : ℝ) : Prop :=
  (AB * DE) / 2 = 5 * ((DE - AB) * AB / 2)

-- Define the relationship between perimeters
def perimeter_relation (AB : ℝ) : Prop :=
  AB * 4/3 = 68

-- Main theorem
theorem rectangle_area (AB DE : ℝ) :
  rectangle AB DE →
  area_relation AB DE →
  perimeter_relation AB →
  AB * DE = 3060 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1018_101890


namespace NUMINAMATH_CALUDE_coeff_x4_product_l1018_101820

def p (x : ℝ) : ℝ := x^5 - 2*x^4 + 4*x^3 - 5*x^2 + 2*x - 1
def q (x : ℝ) : ℝ := 3*x^4 - x^3 + 2*x^2 + 6*x - 5

theorem coeff_x4_product (x : ℝ) : 
  ∃ (a b c d e : ℝ), p x * q x = a*x^5 + 19*x^4 + b*x^3 + c*x^2 + d*x + e := by
  sorry

end NUMINAMATH_CALUDE_coeff_x4_product_l1018_101820


namespace NUMINAMATH_CALUDE_car_cost_difference_l1018_101852

/-- The cost difference between buying and renting a car for a year -/
theorem car_cost_difference (rental_cost : ℕ) (purchase_cost : ℕ) : 
  rental_cost = 20 → purchase_cost = 30 → purchase_cost * 12 - rental_cost * 12 = 120 := by
  sorry

end NUMINAMATH_CALUDE_car_cost_difference_l1018_101852


namespace NUMINAMATH_CALUDE_billys_age_l1018_101868

/-- Given the ages of Billy, Joe, and Mary, prove that Billy is 45 years old. -/
theorem billys_age (B J M : ℕ) 
  (h1 : B = 3 * J)           -- Billy's age is three times Joe's age
  (h2 : B + J = 60)          -- The sum of Billy's and Joe's ages is 60
  (h3 : B + M = 90)          -- The sum of Billy's and Mary's ages is 90
  : B = 45 := by
  sorry


end NUMINAMATH_CALUDE_billys_age_l1018_101868


namespace NUMINAMATH_CALUDE_angle_A_measure_l1018_101807

/-- Given a complex geometric figure with the following properties:
    - Angle B is 120°
    - Angle B forms a linear pair with another angle
    - A triangle adjacent to this setup contains an angle of 50°
    - A small triangle connected to one vertex of the larger triangle has an angle of 45°
    - This small triangle shares a vertex with angle A
    Prove that the measure of angle A is 65° -/
theorem angle_A_measure (B : Real) (adjacent_angle : Real) (large_triangle_angle : Real) (small_triangle_angle : Real) (A : Real) :
  B = 120 →
  B + adjacent_angle = 180 →
  large_triangle_angle = 50 →
  small_triangle_angle = 45 →
  A + small_triangle_angle + (180 - B - large_triangle_angle) = 180 →
  A = 65 := by
  sorry


end NUMINAMATH_CALUDE_angle_A_measure_l1018_101807


namespace NUMINAMATH_CALUDE_tangent_line_to_exponential_curve_l1018_101861

/-- Given that the line y = 1/m is tangent to the curve y = xe^x, prove that m = -e -/
theorem tangent_line_to_exponential_curve (m : ℝ) : 
  (∃ n : ℝ, n * Real.exp n = 1/m ∧ 
   ∀ x : ℝ, x * Real.exp x ≤ 1/m ∧ 
   (x * Real.exp x = 1/m → x = n)) → 
  m = -Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_exponential_curve_l1018_101861


namespace NUMINAMATH_CALUDE_circle_radius_from_circumference_and_area_l1018_101806

/-- Given a circle with specified circumference and area, prove its radius is approximately 4 cm. -/
theorem circle_radius_from_circumference_and_area 
  (circumference : ℝ) 
  (area : ℝ) 
  (h_circumference : circumference = 25.132741228718345)
  (h_area : area = 50.26548245743669) :
  ∃ (radius : ℝ), abs (radius - 4) < 0.0001 ∧ 
    circumference = 2 * Real.pi * radius ∧ 
    area = Real.pi * radius ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_circumference_and_area_l1018_101806


namespace NUMINAMATH_CALUDE_fifteen_sided_figure_area_main_theorem_l1018_101811

/-- The area of a fifteen-sided figure created by cutting off three right triangles
    from the corners of a 4 × 5 rectangle --/
theorem fifteen_sided_figure_area : ℝ → Prop :=
  λ area_result : ℝ =>
    let rectangle_width : ℝ := 4
    let rectangle_height : ℝ := 5
    let rectangle_area : ℝ := rectangle_width * rectangle_height
    let triangle_side : ℝ := 1
    let triangle_area : ℝ := (1 / 2) * triangle_side * triangle_side
    let num_triangles : ℕ := 3
    let total_removed_area : ℝ := (triangle_area : ℝ) * num_triangles
    let final_area : ℝ := rectangle_area - total_removed_area
    area_result = final_area ∧ area_result = 18.5

/-- The main theorem stating that the area of the fifteen-sided figure is 18.5 cm² --/
theorem main_theorem : fifteen_sided_figure_area 18.5 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_sided_figure_area_main_theorem_l1018_101811


namespace NUMINAMATH_CALUDE_canal_construction_l1018_101886

/-- Canal construction problem -/
theorem canal_construction 
  (total_length : ℝ) 
  (team_b_extra : ℝ) 
  (time_ratio : ℝ) 
  (cost_a : ℝ) 
  (cost_b : ℝ) 
  (total_days : ℕ) 
  (h1 : total_length = 1650)
  (h2 : team_b_extra = 30)
  (h3 : time_ratio = 3/2)
  (h4 : cost_a = 90000)
  (h5 : cost_b = 120000)
  (h6 : total_days = 14) :
  ∃ (rate_a rate_b total_cost : ℝ),
    rate_a = 60 ∧ 
    rate_b = 90 ∧ 
    total_cost = 2340000 ∧
    rate_b = rate_a + team_b_extra ∧
    total_length / rate_a = time_ratio * (total_length / rate_b) ∧
    ∃ (days_a_alone : ℝ),
      0 ≤ days_a_alone ∧ 
      days_a_alone ≤ total_days ∧
      rate_a * days_a_alone + (rate_a + rate_b) * (total_days - days_a_alone) = total_length ∧
      total_cost = cost_a * days_a_alone + (cost_a + cost_b) * (total_days - days_a_alone) :=
by sorry

end NUMINAMATH_CALUDE_canal_construction_l1018_101886


namespace NUMINAMATH_CALUDE_test_number_satisfies_conditions_l1018_101846

/-- The test number that satisfies the given conditions -/
def test_number : ℕ := 5

/-- The average score before the current test -/
def previous_average : ℚ := 85

/-- The desired new average score -/
def new_average : ℚ := 88

/-- The score needed on the current test -/
def current_test_score : ℕ := 100

theorem test_number_satisfies_conditions :
  (new_average * test_number : ℚ) - (previous_average * (test_number - 1) : ℚ) = current_test_score := by
  sorry

end NUMINAMATH_CALUDE_test_number_satisfies_conditions_l1018_101846


namespace NUMINAMATH_CALUDE_tan_ratio_from_sin_sum_diff_l1018_101818

theorem tan_ratio_from_sin_sum_diff (a b : ℝ) 
  (h1 : Real.sin (a + b) = 5/8) 
  (h2 : Real.sin (a - b) = 1/4) : 
  Real.tan a / Real.tan b = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_from_sin_sum_diff_l1018_101818


namespace NUMINAMATH_CALUDE_second_number_problem_l1018_101822

theorem second_number_problem (a b c : ℚ) : 
  a + b + c = 264 →
  a = 2 * b →
  c = (1 / 3) * a →
  b = 72 := by sorry

end NUMINAMATH_CALUDE_second_number_problem_l1018_101822


namespace NUMINAMATH_CALUDE_ab_range_l1018_101816

theorem ab_range (a b q : ℝ) (h1 : (1/3 : ℝ) ≤ q ∧ q ≤ 2) 
  (h2 : ∃ m : ℝ, ∃ r1 r2 r3 r4 : ℝ, 
    (r1^2 - a*r1 + 1)*(r1^2 - b*r1 + 1) = 0 ∧
    (r2^2 - a*r2 + 1)*(r2^2 - b*r2 + 1) = 0 ∧
    (r3^2 - a*r3 + 1)*(r3^2 - b*r3 + 1) = 0 ∧
    (r4^2 - a*r4 + 1)*(r4^2 - b*r4 + 1) = 0 ∧
    r1 = m ∧ r2 = m*q ∧ r3 = m*q^2 ∧ r4 = m*q^3) :
  4 ≤ a*b ∧ a*b ≤ 112/9 := by sorry

end NUMINAMATH_CALUDE_ab_range_l1018_101816


namespace NUMINAMATH_CALUDE_probability_factor_less_than_7_l1018_101876

def factors_of_72 : Finset ℕ := {1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72}

def factors_less_than_7 : Finset ℕ := {1, 2, 3, 4, 6}

theorem probability_factor_less_than_7 :
  (factors_less_than_7.card : ℚ) / (factors_of_72.card : ℚ) = 5 / 12 := by sorry

end NUMINAMATH_CALUDE_probability_factor_less_than_7_l1018_101876


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1018_101863

/-- A triangle with two sides of length 2 and 4, and the third side being a solution of x^2 - 6x + 8 = 0 has a perimeter of 10 -/
theorem triangle_perimeter : ∀ a b c : ℝ,
  a = 2 →
  b = 4 →
  c^2 - 6*c + 8 = 0 →
  c > 0 →
  a + b > c →
  b + c > a →
  c + a > b →
  a + b + c = 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1018_101863


namespace NUMINAMATH_CALUDE_children_playing_neither_sport_l1018_101867

theorem children_playing_neither_sport (total : ℕ) (tennis : ℕ) (squash : ℕ) (both : ℕ) 
  (h1 : total = 38)
  (h2 : tennis = 19)
  (h3 : squash = 21)
  (h4 : both = 12) :
  total - (tennis + squash - both) = 10 := by
  sorry

end NUMINAMATH_CALUDE_children_playing_neither_sport_l1018_101867


namespace NUMINAMATH_CALUDE_crown_composition_l1018_101860

theorem crown_composition (total_weight : ℝ) (gold copper tin iron : ℝ)
  (h1 : total_weight = 60)
  (h2 : gold + copper + tin + iron = total_weight)
  (h3 : gold + copper = 2/3 * total_weight)
  (h4 : gold + tin = 3/4 * total_weight)
  (h5 : gold + iron = 3/5 * total_weight) :
  gold = 30.5 ∧ copper = 9.5 ∧ tin = 14.5 ∧ iron = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_crown_composition_l1018_101860


namespace NUMINAMATH_CALUDE_eggs_equation_initial_eggs_count_l1018_101809

/-- The number of eggs initially in the box -/
def initial_eggs : ℕ := sorry

/-- The number of eggs Daniel adds to the box -/
def eggs_added : ℕ := 4

/-- The total number of eggs after Daniel adds more -/
def total_eggs : ℕ := 11

/-- Theorem stating that the initial number of eggs plus the added eggs equals the total eggs -/
theorem eggs_equation : initial_eggs + eggs_added = total_eggs := by sorry

/-- Theorem proving that the initial number of eggs is 7 -/
theorem initial_eggs_count : initial_eggs = 7 := by sorry

end NUMINAMATH_CALUDE_eggs_equation_initial_eggs_count_l1018_101809


namespace NUMINAMATH_CALUDE_print_shop_price_X_l1018_101828

/-- The price per color copy at print shop Y -/
def price_Y : ℝ := 1.70

/-- The number of copies in the comparison -/
def num_copies : ℕ := 70

/-- The price difference between shops Y and X for 70 copies -/
def price_difference : ℝ := 35

/-- The price per color copy at print shop X -/
def price_X : ℝ := 1.20

theorem print_shop_price_X :
  price_X = (price_Y * num_copies - price_difference) / num_copies :=
by sorry

end NUMINAMATH_CALUDE_print_shop_price_X_l1018_101828


namespace NUMINAMATH_CALUDE_three_greater_than_sqrt_seven_l1018_101873

theorem three_greater_than_sqrt_seven : 3 > Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_three_greater_than_sqrt_seven_l1018_101873


namespace NUMINAMATH_CALUDE_first_applicant_better_by_850_l1018_101833

/-- Represents an applicant for a job position -/
structure Applicant where
  salary : ℕ
  revenue : ℕ
  training_months : ℕ
  training_cost_per_month : ℕ
  hiring_bonus_percent : ℕ

/-- Calculates the net gain for the company from an applicant -/
def net_gain (a : Applicant) : ℤ :=
  a.revenue - a.salary - (a.training_months * a.training_cost_per_month) - (a.salary * a.hiring_bonus_percent / 100)

theorem first_applicant_better_by_850 :
  let first := Applicant.mk 42000 93000 3 1200 0
  let second := Applicant.mk 45000 92000 0 0 1
  net_gain first - net_gain second = 850 := by sorry

end NUMINAMATH_CALUDE_first_applicant_better_by_850_l1018_101833


namespace NUMINAMATH_CALUDE_solve_banana_cost_l1018_101859

def banana_cost_problem (initial_amount remaining_amount pears_cost asparagus_cost chicken_cost : ℕ) 
  (num_banana_packs : ℕ) : Prop :=
  let total_spent := initial_amount - remaining_amount
  let other_items_cost := pears_cost + asparagus_cost + chicken_cost
  let banana_total_cost := total_spent - other_items_cost
  banana_total_cost / num_banana_packs = 4

theorem solve_banana_cost :
  banana_cost_problem 55 28 2 6 11 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_banana_cost_l1018_101859


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l1018_101869

theorem sqrt_expression_equality : 
  2 * Real.sqrt 12 * (3 * Real.sqrt 48 - 4 * Real.sqrt (1/8) - 3 * Real.sqrt 27) = 36 - 4 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l1018_101869


namespace NUMINAMATH_CALUDE_quadratic_coefficients_l1018_101865

-- Define the original equation
def original_equation (x : ℝ) : Prop := 3 * x^2 - 2 = 4 * x

-- Define the general form of a quadratic equation
def general_form (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Theorem statement
theorem quadratic_coefficients :
  ∃ (c : ℝ), ∀ (x : ℝ), 
    (original_equation x ↔ general_form 3 (-4) c x) :=
sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_l1018_101865


namespace NUMINAMATH_CALUDE_paco_ate_18_cookies_l1018_101851

/-- The number of cookies Paco ate -/
def cookies_eaten (initial : ℕ) (given : ℕ) : ℕ := given + given

/-- Proof that Paco ate 18 cookies -/
theorem paco_ate_18_cookies (initial : ℕ) (given : ℕ) 
  (h1 : initial = 41)
  (h2 : given = 9) :
  cookies_eaten initial given = 18 := by
  sorry

end NUMINAMATH_CALUDE_paco_ate_18_cookies_l1018_101851


namespace NUMINAMATH_CALUDE_other_factor_power_of_two_l1018_101896

def w : ℕ := 144

theorem other_factor_power_of_two :
  (∃ (k : ℕ), 936 * w = k * (3^3) * (12^2)) →
  (∀ (m : ℕ), m < w → ¬(∃ (l : ℕ), 936 * m = l * (3^3) * (12^2))) →
  (∃ (x : ℕ), 2^x ∣ (936 * w) ∧ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_other_factor_power_of_two_l1018_101896


namespace NUMINAMATH_CALUDE_diana_operations_l1018_101898

theorem diana_operations (x : ℝ) : 
  (((x + 3) * 3 - 3) / 3 + 3 = 12) → x = 7 := by
sorry

end NUMINAMATH_CALUDE_diana_operations_l1018_101898


namespace NUMINAMATH_CALUDE_ring_area_l1018_101899

theorem ring_area (r : ℝ) (h : r > 0) :
  let outer_radius : ℝ := 3 * r
  let inner_radius : ℝ := r
  let width : ℝ := 3
  outer_radius - inner_radius = width →
  (π * outer_radius^2 - π * inner_radius^2) = 72 * π :=
by
  sorry

end NUMINAMATH_CALUDE_ring_area_l1018_101899


namespace NUMINAMATH_CALUDE_average_speed_calculation_l1018_101839

-- Define the variables
def distance_day1 : ℝ := 100
def distance_day2 : ℝ := 175
def time_difference : ℝ := 3

-- Define the theorem
theorem average_speed_calculation :
  ∃ (v : ℝ), v > 0 ∧
    distance_day2 / v - distance_day1 / v = time_difference ∧
    v = 25 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l1018_101839


namespace NUMINAMATH_CALUDE_work_hours_per_day_l1018_101864

/-- Proves that working 56 hours over 14 days results in 4 hours of work per day -/
theorem work_hours_per_day (total_hours : ℕ) (total_days : ℕ) (hours_per_day : ℕ) : 
  total_hours = 56 → total_days = 14 → total_hours = total_days * hours_per_day → hours_per_day = 4 := by
  sorry

#check work_hours_per_day

end NUMINAMATH_CALUDE_work_hours_per_day_l1018_101864


namespace NUMINAMATH_CALUDE_heesu_received_most_l1018_101850

/-- The number of sweets each person received -/
structure SweetDistribution where
  total : Nat
  minsu : Nat
  jaeyoung : Nat
  heesu : Nat

/-- Heesu received the most sweets -/
def heesuReceivedMost (d : SweetDistribution) : Prop :=
  d.heesu ≥ d.minsu ∧ d.heesu ≥ d.jaeyoung

/-- Theorem: Given the distribution of sweets, Heesu received the most -/
theorem heesu_received_most (d : SweetDistribution) 
  (h1 : d.total = 30)
  (h2 : d.minsu = 12)
  (h3 : d.jaeyoung = 3)
  (h4 : d.heesu = 15) : 
  heesuReceivedMost d := by
  sorry

end NUMINAMATH_CALUDE_heesu_received_most_l1018_101850


namespace NUMINAMATH_CALUDE_inequality_proof_l1018_101893

theorem inequality_proof (a b : ℝ) (h : a < b ∧ b < 0) : a^2 > a*b ∧ a*b > b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1018_101893


namespace NUMINAMATH_CALUDE_second_subject_grade_l1018_101883

/-- Represents the grade of a student in a subject as a percentage -/
def Grade := Fin 101

/-- Calculates the average of three grades -/
def average (g1 g2 g3 : Grade) : ℚ :=
  (g1.val + g2.val + g3.val) / 3

theorem second_subject_grade 
  (g1 g3 : Grade) 
  (h1 : g1.val = 60) 
  (h3 : g3.val = 80) :
  ∃ (g2 : Grade), average g1 g2 g3 = 70 ∧ g2.val = 70 := by
  sorry

end NUMINAMATH_CALUDE_second_subject_grade_l1018_101883


namespace NUMINAMATH_CALUDE_two_books_into_five_l1018_101801

/-- The number of ways to insert new books into a shelf while maintaining the order of existing books -/
def insert_books (original : ℕ) (new : ℕ) : ℕ :=
  (original + 1) * (original + 2) / 2

/-- Theorem stating that inserting 2 books into a shelf with 5 books results in 42 different arrangements -/
theorem two_books_into_five : insert_books 5 2 = 42 := by
  sorry

end NUMINAMATH_CALUDE_two_books_into_five_l1018_101801


namespace NUMINAMATH_CALUDE_breakfast_rearrangements_count_l1018_101880

/-- The number of distinguishable rearrangements of "BREAKFAST" with vowels first -/
def breakfast_rearrangements : ℕ :=
  let vowels := 3  -- Number of vowels in "BREAKFAST"
  let repeated_vowels := 2  -- Number of times 'A' appears
  let consonants := 6  -- Number of consonants in "BREAKFAST"
  (vowels.factorial / repeated_vowels.factorial) * consonants.factorial

/-- Theorem stating that the number of rearrangements is 2160 -/
theorem breakfast_rearrangements_count :
  breakfast_rearrangements = 2160 := by
  sorry

end NUMINAMATH_CALUDE_breakfast_rearrangements_count_l1018_101880


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_l1018_101840

-- Define a complex number
def complex_number (a b : ℝ) := a + b * Complex.I

-- Define what it means for a complex number to be purely imaginary
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- Define condition p
def condition_p (a b : ℝ) : Prop := is_purely_imaginary (complex_number a b)

-- Define condition q
def condition_q (a b : ℝ) : Prop := a = 0

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary :
  (∀ a b : ℝ, condition_p a b → condition_q a b) ∧
  (∃ a b : ℝ, condition_q a b ∧ ¬condition_p a b) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_l1018_101840


namespace NUMINAMATH_CALUDE_jack_afternoon_letters_l1018_101885

/-- The number of letters Jack received in the morning -/
def morning_letters : ℕ := 8

/-- The number of letters Jack received in the afternoon -/
def afternoon_letters : ℕ := morning_letters - 1

theorem jack_afternoon_letters : afternoon_letters = 7 := by
  sorry

end NUMINAMATH_CALUDE_jack_afternoon_letters_l1018_101885


namespace NUMINAMATH_CALUDE_savings_period_is_four_months_l1018_101892

-- Define the savings and stock parameters
def wife_weekly_savings : ℕ := 100
def husband_monthly_savings : ℕ := 225
def stock_price : ℕ := 50
def shares_bought : ℕ := 25

-- Define the function to calculate the number of months saved
def months_saved : ℕ :=
  let total_investment := stock_price * shares_bought
  let total_savings := total_investment * 2
  let monthly_savings := wife_weekly_savings * 4 + husband_monthly_savings
  total_savings / monthly_savings

-- Theorem statement
theorem savings_period_is_four_months :
  months_saved = 4 :=
sorry

end NUMINAMATH_CALUDE_savings_period_is_four_months_l1018_101892


namespace NUMINAMATH_CALUDE_shoe_price_calculation_l1018_101881

theorem shoe_price_calculation (initial_price : ℝ) : 
  initial_price = 50 →
  let wednesday_price := initial_price * (1 + 0.15)
  let thursday_price := wednesday_price * (1 - 0.05)
  let monday_price := thursday_price * (1 - 0.20)
  monday_price = 43.70 := by sorry

end NUMINAMATH_CALUDE_shoe_price_calculation_l1018_101881


namespace NUMINAMATH_CALUDE_donkey_mule_bags_l1018_101830

theorem donkey_mule_bags (x y : ℕ) (hx : x > 0) (hy : y > 0) : 
  (y + 1 = 2 * (x - 1) ∧ y - 1 = x + 1) ↔ 
  (∃ (d m : ℕ), d = x ∧ m = y ∧ 
    (m + 1 = 2 * (d - 1)) ∧ 
    (m - 1 = d + 1)) :=
by sorry

end NUMINAMATH_CALUDE_donkey_mule_bags_l1018_101830


namespace NUMINAMATH_CALUDE_sandra_amount_sandra_gets_100_l1018_101877

def share_money (sandra_ratio : ℕ) (amy_ratio : ℕ) (ruth_ratio : ℕ) (amy_amount : ℕ) : ℕ → ℕ → ℕ → Prop :=
  λ sandra_amount ruth_amount total_amount =>
    sandra_amount * amy_ratio = amy_amount * sandra_ratio ∧
    ruth_amount * amy_ratio = amy_amount * ruth_ratio ∧
    total_amount = sandra_amount + amy_amount + ruth_amount

theorem sandra_amount (amy_amount : ℕ) :
  share_money 2 1 3 amy_amount (2 * amy_amount) (3 * amy_amount) (6 * amy_amount) :=
by sorry

theorem sandra_gets_100 :
  share_money 2 1 3 50 100 150 300 :=
by sorry

end NUMINAMATH_CALUDE_sandra_amount_sandra_gets_100_l1018_101877


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1018_101854

/-- A line parallel to 2x - y + 1 = 0 and tangent to x^2 + y^2 = 5 has equation 2x - y ± 5 = 0 -/
theorem tangent_line_equation (x y : ℝ) :
  ∃ (k : ℝ), k = 5 ∨ k = -5 ∧
  (∀ (x y : ℝ), 2*x - y + k = 0 →
    (∀ (x₀ y₀ : ℝ), 2*x₀ - y₀ + 1 = 0 → ∃ (t : ℝ), x = x₀ + 2*t ∧ y = y₀ + t) ∧
    (∃! (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 5 ∧ 2*x₀ - y₀ + k = 0)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1018_101854


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1018_101805

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem solution_set_of_inequality :
  ∀ x : ℝ, 0 < x ∧ x < 1 ↔ f (Real.exp x) < 1 :=
by
  sorry

-- Define the properties of f
axiom f_derivative (x : ℝ) (h : x > 0) : 
  deriv f x = (x - (Real.exp 1 - 1)) / x

axiom f_value_at_1 : f 1 = 1

axiom f_value_at_e : f (Real.exp 1) = 1

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1018_101805


namespace NUMINAMATH_CALUDE_soccer_game_time_proof_l1018_101827

/-- Calculates the total time in minutes for a soccer game and post-game ceremony -/
def total_time (game_hours : ℕ) (game_minutes : ℕ) (ceremony_minutes : ℕ) : ℕ :=
  game_hours * 60 + game_minutes + ceremony_minutes

/-- Proves that the total time for a 2 hour 35 minute game and 25 minute ceremony is 180 minutes -/
theorem soccer_game_time_proof :
  total_time 2 35 25 = 180 := by
  sorry

end NUMINAMATH_CALUDE_soccer_game_time_proof_l1018_101827
