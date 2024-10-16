import Mathlib

namespace NUMINAMATH_CALUDE_complex_number_equality_l3972_397287

theorem complex_number_equality (a : ℝ) : 
  (Complex.re ((2 * Complex.I - a) / Complex.I) = Complex.im ((2 * Complex.I - a) / Complex.I)) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l3972_397287


namespace NUMINAMATH_CALUDE_eighth_term_is_25_5_l3972_397265

/-- An arithmetic sequence with 15 terms, first term 3, and last term 48 -/
structure ArithmeticSequence where
  n : ℕ
  a₁ : ℚ
  a₁₅ : ℚ
  h_n : n = 15
  h_a₁ : a₁ = 3
  h_a₁₅ : a₁₅ = 48

/-- The 8th term of the arithmetic sequence is 25.5 -/
theorem eighth_term_is_25_5 (seq : ArithmeticSequence) : 
  let d := (seq.a₁₅ - seq.a₁) / (seq.n - 1)
  seq.a₁ + 7 * d = 25.5 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_is_25_5_l3972_397265


namespace NUMINAMATH_CALUDE_unique_solution_l3972_397273

/-- The equation holds for all real x -/
def equation_holds (k : ℕ) : Prop :=
  ∀ x : ℝ, (Real.sin x)^k * Real.sin (k * x) + (Real.cos x)^k * Real.cos (k * x) = (Real.cos (2 * x))^k

/-- k = 3 is the only positive integer solution -/
theorem unique_solution :
  ∃! k : ℕ, k > 0 ∧ equation_holds k :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l3972_397273


namespace NUMINAMATH_CALUDE_intersection_A_B_l3972_397286

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x

-- Define set A
def A : Set ℝ := {x | f x < 0}

-- Define set B
def B : Set ℝ := {x | (deriv f) x > 0}

-- State the theorem
theorem intersection_A_B : A ∩ B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3972_397286


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3972_397239

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 2 * I
  let z₂ : ℂ := 4 - 2 * I
  let z₃ : ℂ := 4 - 6 * I
  let z₄ : ℂ := 4 + 6 * I
  z₁ / z₂ + z₃ / z₄ = 14 / 65 - 8 / 65 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3972_397239


namespace NUMINAMATH_CALUDE_sum_of_digits_M_l3972_397289

-- Define a function to represent the sum of digits
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Define a predicate to check if a number only uses allowed digits
def uses_allowed_digits (n : ℕ) : Prop := sorry

theorem sum_of_digits_M (M : ℕ) 
  (h_even : Even M)
  (h_digits : uses_allowed_digits M)
  (h_double : sum_of_digits (2 * M) = 31)
  (h_half : sum_of_digits (M / 2) = 28) :
  sum_of_digits M = 29 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_M_l3972_397289


namespace NUMINAMATH_CALUDE_amount_equals_scientific_notation_l3972_397279

/-- Represents the amount in yuan -/
def amount : ℝ := 2.51e6

/-- Represents the scientific notation of the amount -/
def scientific_notation : ℝ := 2.51 * (10 ^ 6)

/-- Theorem stating that the amount is equal to its scientific notation representation -/
theorem amount_equals_scientific_notation : amount = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_amount_equals_scientific_notation_l3972_397279


namespace NUMINAMATH_CALUDE_relationship_abc_l3972_397231

theorem relationship_abc : 
  let a : ℝ := (0.6 : ℝ) ^ (2/5 : ℝ)
  let b : ℝ := (0.4 : ℝ) ^ (2/5 : ℝ)
  let c : ℝ := (0.4 : ℝ) ^ (3/5 : ℝ)
  a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l3972_397231


namespace NUMINAMATH_CALUDE_village_population_l3972_397243

theorem village_population (partial_population : ℕ) (percentage : ℚ) (total_population : ℕ) : 
  percentage = 96 / 100 ∧ 
  partial_population = 23040 ∧ 
  (percentage * total_population : ℚ) = partial_population →
  total_population = 24000 := by
sorry

end NUMINAMATH_CALUDE_village_population_l3972_397243


namespace NUMINAMATH_CALUDE_C_not_necessary_nor_sufficient_for_A_l3972_397228

-- Define the propositions
variable (A B C : Prop)

-- Define the given conditions
axiom C_sufficient_for_B : C → B
axiom B_necessary_for_A : A → B

-- Theorem to prove
theorem C_not_necessary_nor_sufficient_for_A :
  ¬(∀ (h : A), C) ∧ ¬(∀ (h : C), A) :=
by sorry

end NUMINAMATH_CALUDE_C_not_necessary_nor_sufficient_for_A_l3972_397228


namespace NUMINAMATH_CALUDE_complex_square_roots_l3972_397247

theorem complex_square_roots (z : ℂ) : 
  z ^ 2 = -91 - 49 * I ↔ z = (7 * Real.sqrt 2) / 2 - 7 * Real.sqrt 2 * I ∨ 
                         z = -(7 * Real.sqrt 2) / 2 + 7 * Real.sqrt 2 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_roots_l3972_397247


namespace NUMINAMATH_CALUDE_soccer_handshakes_l3972_397217

theorem soccer_handshakes (team_size : Nat) (referee_count : Nat) (coach_count : Nat) :
  team_size = 7 ∧ referee_count = 3 ∧ coach_count = 2 →
  let player_count := 2 * team_size
  let player_player_handshakes := team_size * team_size
  let player_referee_handshakes := player_count * referee_count
  let coach_handshakes := coach_count * (player_count + referee_count)
  player_player_handshakes + player_referee_handshakes + coach_handshakes = 125 :=
by sorry


end NUMINAMATH_CALUDE_soccer_handshakes_l3972_397217


namespace NUMINAMATH_CALUDE_contrapositive_example_l3972_397272

theorem contrapositive_example (a b : ℝ) :
  (¬(a = 0 → a * b = 0) ↔ (a * b ≠ 0 → a ≠ 0)) := by sorry

end NUMINAMATH_CALUDE_contrapositive_example_l3972_397272


namespace NUMINAMATH_CALUDE_james_purchase_cost_l3972_397214

/-- The total cost of James' purchase of dirt bikes and off-road vehicles, including registration fees. -/
def total_cost (dirt_bike_count : ℕ) (dirt_bike_price : ℕ) 
                (offroad_count : ℕ) (offroad_price : ℕ) 
                (registration_fee : ℕ) : ℕ :=
  dirt_bike_count * dirt_bike_price + 
  offroad_count * offroad_price + 
  (dirt_bike_count + offroad_count) * registration_fee

/-- Theorem stating that James' total cost is $1825 -/
theorem james_purchase_cost : 
  total_cost 3 150 4 300 25 = 1825 := by
  sorry

end NUMINAMATH_CALUDE_james_purchase_cost_l3972_397214


namespace NUMINAMATH_CALUDE_saree_final_price_l3972_397240

def original_price : ℝ := 4000

def discount1 : ℝ := 0.15
def discount2 : ℝ := 0.10
def discount3 : ℝ := 0.08
def flat_discount : ℝ := 300

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def final_price : ℝ :=
  apply_discount (apply_discount (apply_discount original_price discount1) discount2) discount3 - flat_discount

theorem saree_final_price :
  final_price = 2515.20 :=
by sorry

end NUMINAMATH_CALUDE_saree_final_price_l3972_397240


namespace NUMINAMATH_CALUDE_equation_solution_l3972_397216

theorem equation_solution (x : ℚ) : 64 * (x + 1)^3 - 27 = 0 → x = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3972_397216


namespace NUMINAMATH_CALUDE_log_equation_solution_l3972_397250

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_equation_solution :
  ∃ (x : ℝ), log (2^x) (3^20) = log (2^(x+3)) (3^2020) → x = 3/100 :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3972_397250


namespace NUMINAMATH_CALUDE_fox_max_berries_l3972_397269

/-- The number of bear cubs --/
def num_cubs : ℕ := 100

/-- The initial number of berries for the n-th bear cub --/
def initial_berries (n : ℕ) : ℕ := 2^(n-1)

/-- The total number of berries initially --/
def total_berries : ℕ := 2^num_cubs - 1

/-- The maximum number of berries the fox can eat --/
def max_fox_berries : ℕ := 2^num_cubs - (num_cubs + 1)

theorem fox_max_berries :
  ∀ (redistribution : ℕ → ℕ → ℕ),
  (∀ (a b : ℕ), redistribution a b ≤ a + b) →
  (∀ (a b : ℕ), redistribution a b = redistribution b a) →
  (∃ (final_berries : ℕ), ∀ (i : ℕ), i ≤ num_cubs → redistribution (initial_berries i) final_berries = final_berries) →
  (total_berries - num_cubs * final_berries) ≤ max_fox_berries :=
sorry

end NUMINAMATH_CALUDE_fox_max_berries_l3972_397269


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3972_397212

def arithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formula
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_arith : arithmeticSequence a d)
  (h_d_neg : d < 0)
  (h_prod : a 2 * a 4 = 12)
  (h_sum : a 2 + a 4 = 8) :
  ∀ n : ℕ+, a n = -2 * n + 10 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3972_397212


namespace NUMINAMATH_CALUDE_triangle_height_equals_twice_rectangle_width_l3972_397260

/-- Given a rectangle with dimensions a and b, and an isosceles triangle with base a and height h',
    if they have the same area, then the height of the triangle is 2b. -/
theorem triangle_height_equals_twice_rectangle_width
  (a b h' : ℝ) 
  (ha : a > 0)
  (hb : b > 0)
  (hh' : h' > 0)
  (h_area_eq : (1/2) * a * h' = a * b) :
  h' = 2 * b :=
by sorry

end NUMINAMATH_CALUDE_triangle_height_equals_twice_rectangle_width_l3972_397260


namespace NUMINAMATH_CALUDE_ravish_failed_by_40_l3972_397244

/-- The number of marks Ravish failed by in his board exam -/
def marks_failed (max_marks passing_percentage ravish_marks : ℕ) : ℕ :=
  (max_marks * passing_percentage / 100) - ravish_marks

/-- Proof that Ravish failed by 40 marks -/
theorem ravish_failed_by_40 :
  marks_failed 200 40 40 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ravish_failed_by_40_l3972_397244


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3972_397211

theorem inequality_system_solution (x : ℝ) :
  (2 * x - 1 ≥ x + 2) ∧ (x + 5 < 4 * x - 1) → x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3972_397211


namespace NUMINAMATH_CALUDE_rachels_homework_difference_l3972_397202

/-- Rachel's homework problem -/
theorem rachels_homework_difference (math_pages reading_pages : ℕ) 
  (h1 : math_pages = 3) 
  (h2 : reading_pages = 4) : 
  reading_pages - math_pages = 1 := by
  sorry

end NUMINAMATH_CALUDE_rachels_homework_difference_l3972_397202


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l3972_397290

universe u

def U : Finset ℕ := {4,5,6,8,9}
def M : Finset ℕ := {5,6,8}

theorem complement_of_M_in_U :
  (U \ M) = {4,9} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l3972_397290


namespace NUMINAMATH_CALUDE_total_red_cards_l3972_397242

/-- The number of decks of playing cards --/
def num_decks : ℕ := 8

/-- The number of red cards in one standard deck --/
def red_cards_per_deck : ℕ := 26

/-- Theorem: The total number of red cards in 8 decks is 208 --/
theorem total_red_cards : num_decks * red_cards_per_deck = 208 := by
  sorry

end NUMINAMATH_CALUDE_total_red_cards_l3972_397242


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3972_397206

theorem min_value_of_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) (hsum : x + y = 3) :
  (1 / (x - 1) + 3 / (y - 1)) ≥ 4 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3972_397206


namespace NUMINAMATH_CALUDE_number_to_add_divisibility_l3972_397249

theorem number_to_add_divisibility (p q : ℕ) (n m : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  p = 563 → q = 839 → n = 1398547 → m = 18284 →
  (p * q) ∣ (n + m) :=
by sorry

end NUMINAMATH_CALUDE_number_to_add_divisibility_l3972_397249


namespace NUMINAMATH_CALUDE_max_pencils_theorem_l3972_397296

/-- Represents the discount rules for pencil purchases -/
structure DiscountRules where
  large_set : Nat
  large_discount : Rat
  small_set : Nat
  small_discount : Rat

/-- Calculates the maximum number of pencils that can be purchased given initial funds and discount rules -/
def max_pencils (initial_funds : Nat) (rules : DiscountRules) : Nat :=
  sorry

/-- The theorem stating that given the specific initial funds and discount rules, the maximum number of pencils that can be purchased is 36 -/
theorem max_pencils_theorem (initial_funds : Nat) (rules : DiscountRules) :
  initial_funds = 30 ∧
  rules.large_set = 20 ∧
  rules.large_discount = 1/4 ∧
  rules.small_set = 5 ∧
  rules.small_discount = 1/10
  → max_pencils initial_funds rules = 36 := by
  sorry

end NUMINAMATH_CALUDE_max_pencils_theorem_l3972_397296


namespace NUMINAMATH_CALUDE_crayon_selection_theorem_l3972_397215

def total_crayons : ℕ := 15
def red_crayons : ℕ := 2
def selection_size : ℕ := 6

def select_crayons_with_red : ℕ := Nat.choose total_crayons selection_size - Nat.choose (total_crayons - red_crayons) selection_size

theorem crayon_selection_theorem : select_crayons_with_red = 2860 := by
  sorry

end NUMINAMATH_CALUDE_crayon_selection_theorem_l3972_397215


namespace NUMINAMATH_CALUDE_largest_quantity_l3972_397241

theorem largest_quantity : 
  let A := (3010 : ℚ) / 3009 + 3010 / 3011
  let B := (3010 : ℚ) / 3011 + 3012 / 3011
  let C := (3011 : ℚ) / 3010 + 3011 / 3012
  A > B ∧ A > C := by
  sorry

end NUMINAMATH_CALUDE_largest_quantity_l3972_397241


namespace NUMINAMATH_CALUDE_a_3_value_l3972_397288

/-- Given a sequence {a_n} where S_n is the sum of the first n terms -/
def S (n : ℕ) : ℚ := (n + 1 : ℚ) / (n + 2 : ℚ)

/-- Definition of a_n in terms of S_n -/
def a (n : ℕ) : ℚ :=
  if n = 1 then S 1
  else S n - S (n - 1)

theorem a_3_value : a 3 = 1 / 20 := by sorry

end NUMINAMATH_CALUDE_a_3_value_l3972_397288


namespace NUMINAMATH_CALUDE_sin_alpha_eq_neg_half_l3972_397210

theorem sin_alpha_eq_neg_half (α : Real) 
  (h : Real.sin (α/2 - Real.pi/4) * Real.cos (α/2 + Real.pi/4) = -3/4) : 
  Real.sin α = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_eq_neg_half_l3972_397210


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l3972_397221

theorem cyclic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let f := fun (a b c : ℝ) => (a + b - c)^2 / ((a + b)^2 + c^2)
  f x y z + f y z x + f z x y ≥ 3/5 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l3972_397221


namespace NUMINAMATH_CALUDE_right_rectangular_prism_volume_l3972_397237

theorem right_rectangular_prism_volume 
  (side_area front_area bottom_area : ℝ) 
  (h_side : side_area = 12) 
  (h_front : front_area = 8) 
  (h_bottom : bottom_area = 4) : 
  ∃ (a b c : ℝ), 
    a * b = side_area ∧ 
    b * c = front_area ∧ 
    a * c = bottom_area ∧ 
    a * b * c = 8 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_right_rectangular_prism_volume_l3972_397237


namespace NUMINAMATH_CALUDE_sqrt_14_plus_2_bounds_l3972_397278

theorem sqrt_14_plus_2_bounds : 5 < Real.sqrt 14 + 2 ∧ Real.sqrt 14 + 2 < 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_14_plus_2_bounds_l3972_397278


namespace NUMINAMATH_CALUDE_fraction_inequality_l3972_397201

theorem fraction_inequality (x : ℝ) : x / (x + 3) ≥ 0 ↔ x ∈ Set.Ici 0 ∪ Set.Iic (-3) :=
sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3972_397201


namespace NUMINAMATH_CALUDE_complex_on_line_l3972_397268

theorem complex_on_line (z : ℂ) (a : ℝ) :
  z = (2 + a * Complex.I) / (1 + Complex.I) →
  (z.re = -z.im) →
  a = 0 := by sorry

end NUMINAMATH_CALUDE_complex_on_line_l3972_397268


namespace NUMINAMATH_CALUDE_balloon_count_l3972_397251

/-- The number of blue balloons after a series of events --/
def total_balloons (joan_initial : ℕ) (joan_popped : ℕ) (jessica_initial : ℕ) (jessica_inflated : ℕ) (peter_initial : ℕ) (peter_deflated : ℕ) : ℕ :=
  (joan_initial - joan_popped) + (jessica_initial + jessica_inflated) + (peter_initial - peter_deflated)

/-- Theorem stating the total number of balloons after the given events --/
theorem balloon_count :
  total_balloons 9 5 2 3 4 2 = 11 := by
  sorry

#eval total_balloons 9 5 2 3 4 2

end NUMINAMATH_CALUDE_balloon_count_l3972_397251


namespace NUMINAMATH_CALUDE_temperature_difference_l3972_397280

theorem temperature_difference (M L N : ℝ) : 
  M = L + N →
  (∃ (M_4 L_4 : ℝ), 
    M_4 = M - 5 ∧
    L_4 = L + 3 ∧
    abs (M_4 - L_4) = 2) →
  (N = 6 ∨ N = 10) ∧ 6 * 10 = 60 := by
sorry

end NUMINAMATH_CALUDE_temperature_difference_l3972_397280


namespace NUMINAMATH_CALUDE_contrapositive_false_proposition_l3972_397282

theorem contrapositive_false_proposition : 
  ¬(∀ x : ℝ, x ≠ 1 → x^2 ≠ 1) := by sorry

end NUMINAMATH_CALUDE_contrapositive_false_proposition_l3972_397282


namespace NUMINAMATH_CALUDE_trig_expression_equality_l3972_397248

theorem trig_expression_equality : 
  (1 - Real.cos (10 * π / 180)^2) / 
  (Real.cos (800 * π / 180) * Real.sqrt (1 - Real.cos (20 * π / 180))) = 
  Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l3972_397248


namespace NUMINAMATH_CALUDE_periodic_even_function_theorem_l3972_397291

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem periodic_even_function_theorem (f : ℝ → ℝ) 
  (h_periodic : is_periodic f 2)
  (h_even : is_even f)
  (h_defined : ∀ x ∈ Set.Icc 2 3, f x = x) :
  ∀ x ∈ Set.Icc (-2) 0, f x = 3 - |x + 1| := by
  sorry

end NUMINAMATH_CALUDE_periodic_even_function_theorem_l3972_397291


namespace NUMINAMATH_CALUDE_difference_calculation_l3972_397245

theorem difference_calculation (total : ℝ) (h : total = 6000) : 
  (1 / 10 * total) - (1 / 1000 * total) = 594 := by
  sorry

end NUMINAMATH_CALUDE_difference_calculation_l3972_397245


namespace NUMINAMATH_CALUDE_tree_leaves_theorem_l3972_397276

/-- Calculates the number of leaves remaining on a tree after 5 weeks of shedding --/
def leaves_remaining (initial_leaves : ℕ) : ℕ :=
  let week1_remaining := initial_leaves - initial_leaves / 5
  let week2_shed := (week1_remaining * 30) / 100
  let week2_remaining := week1_remaining - week2_shed
  let week3_shed := (week2_shed * 60) / 100
  let week3_remaining := week2_remaining - week3_shed
  let week4_shed := week3_remaining / 2
  let week4_remaining := week3_remaining - week4_shed
  let week5_shed := (week3_shed * 2) / 3
  week4_remaining - week5_shed

/-- Theorem stating that a tree with 5000 initial leaves will have 560 leaves remaining after 5 weeks of shedding --/
theorem tree_leaves_theorem :
  leaves_remaining 5000 = 560 := by
  sorry

end NUMINAMATH_CALUDE_tree_leaves_theorem_l3972_397276


namespace NUMINAMATH_CALUDE_division_of_fractions_l3972_397227

theorem division_of_fractions : (3 + 1/2) / 7 / (5/3) = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_division_of_fractions_l3972_397227


namespace NUMINAMATH_CALUDE_leo_money_after_settling_debts_l3972_397297

/-- The total amount of money Leo and Ryan have together -/
def total_amount : ℚ := 48

/-- The fraction of the total amount that Ryan owns -/
def ryan_fraction : ℚ := 2/3

/-- The amount Ryan owes Leo -/
def ryan_owes_leo : ℚ := 10

/-- The amount Leo owes Ryan -/
def leo_owes_ryan : ℚ := 7

/-- Leo's final amount after settling debts -/
def leo_final_amount : ℚ := 19

theorem leo_money_after_settling_debts :
  let ryan_initial := ryan_fraction * total_amount
  let leo_initial := total_amount - ryan_initial
  let net_debt := ryan_owes_leo - leo_owes_ryan
  leo_initial + net_debt = leo_final_amount := by
sorry

end NUMINAMATH_CALUDE_leo_money_after_settling_debts_l3972_397297


namespace NUMINAMATH_CALUDE_divisibility_condition_l3972_397246

/-- Represents a four-digit number MCUD -/
structure FourDigitNumber where
  M : Nat
  C : Nat
  D : Nat
  U : Nat
  h_M : M < 10
  h_C : C < 10
  h_D : D < 10
  h_U : U < 10

/-- Calculates the value of a four-digit number -/
def FourDigitNumber.value (n : FourDigitNumber) : Nat :=
  1000 * n.M + 100 * n.C + 10 * n.D + n.U

/-- Calculates the remainders r₁, r₂, and r₃ for a given divisor -/
def calculateRemainders (A : Nat) : Nat × Nat × Nat :=
  let r₁ := 10 % A
  let r₂ := (10 * r₁) % A
  let r₃ := (10 * r₂) % A
  (r₁, r₂, r₃)

/-- The main theorem stating the divisibility condition -/
theorem divisibility_condition (n : FourDigitNumber) (A : Nat) (hA : A > 0) :
  A ∣ n.value ↔ A ∣ (n.U + n.D * (calculateRemainders A).1 + n.C * (calculateRemainders A).2.1 + n.M * (calculateRemainders A).2.2) :=
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3972_397246


namespace NUMINAMATH_CALUDE_class_average_l3972_397209

theorem class_average (students1 : ℕ) (average1 : ℚ) (students2 : ℕ) (average2 : ℚ) :
  students1 = 15 →
  average1 = 73/100 →
  students2 = 10 →
  average2 = 88/100 →
  (students1 * average1 + students2 * average2) / (students1 + students2) = 79/100 := by
  sorry

end NUMINAMATH_CALUDE_class_average_l3972_397209


namespace NUMINAMATH_CALUDE_pet_insurance_cost_l3972_397236

/-- Represents the cost of a vet appointment in dollars -/
def vet_cost : ℕ := 400

/-- Represents the number of vet appointments -/
def num_appointments : ℕ := 3

/-- Represents the percentage of the cost covered by insurance -/
def insurance_coverage : ℚ := 80 / 100

/-- Represents the total amount John paid in dollars -/
def total_paid : ℕ := 660

/-- Proves that the amount John paid for pet insurance is $100 -/
theorem pet_insurance_cost :
  ∃ (insurance_cost : ℕ),
    insurance_cost = total_paid - (vet_cost + (num_appointments - 1) * vet_cost * (1 - insurance_coverage)) ∧
    insurance_cost = 100 := by
  sorry

end NUMINAMATH_CALUDE_pet_insurance_cost_l3972_397236


namespace NUMINAMATH_CALUDE_sum_zero_ratio_negative_half_l3972_397270

theorem sum_zero_ratio_negative_half 
  (w x y z : ℝ) 
  (hw : w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z) 
  (hsum : w + x + y + z = 0) :
  (w * y + x * z) / (w^2 + x^2 + y^2 + z^2) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_zero_ratio_negative_half_l3972_397270


namespace NUMINAMATH_CALUDE_family_savings_theorem_l3972_397271

/-- Represents the monthly financial data of Ivan Tsarevich's family -/
structure FamilyFinances where
  ivan_salary : ℝ
  vasilisa_salary : ℝ
  mother_salary : ℝ
  father_salary : ℝ
  son_scholarship : ℝ
  monthly_expenses : ℝ
  tax_rate : ℝ

def calculate_net_income (gross_income : ℝ) (tax_rate : ℝ) : ℝ :=
  gross_income * (1 - tax_rate)

def calculate_total_net_income (f : FamilyFinances) : ℝ :=
  calculate_net_income f.ivan_salary f.tax_rate +
  calculate_net_income f.vasilisa_salary f.tax_rate +
  calculate_net_income f.mother_salary f.tax_rate +
  calculate_net_income f.father_salary f.tax_rate +
  f.son_scholarship

def calculate_monthly_savings (f : FamilyFinances) : ℝ :=
  calculate_total_net_income f - f.monthly_expenses

theorem family_savings_theorem (f : FamilyFinances) 
  (h1 : f.ivan_salary = 55000)
  (h2 : f.vasilisa_salary = 45000)
  (h3 : f.mother_salary = 18000)
  (h4 : f.father_salary = 20000)
  (h5 : f.son_scholarship = 3000)
  (h6 : f.monthly_expenses = 74000)
  (h7 : f.tax_rate = 0.13) :
  calculate_monthly_savings f = 49060 ∧
  calculate_monthly_savings { f with 
    mother_salary := 10000,
    son_scholarship := 3000 } = 43400 ∧
  calculate_monthly_savings { f with 
    mother_salary := 10000,
    son_scholarship := 16050 } = 56450 := by
  sorry

#check family_savings_theorem

end NUMINAMATH_CALUDE_family_savings_theorem_l3972_397271


namespace NUMINAMATH_CALUDE_exists_divisible_by_sum_of_digits_l3972_397292

/-- Sum of digits of a three-digit number -/
def sumOfDigits (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

/-- Theorem: Among 18 consecutive three-digit numbers, there is at least one divisible by its sum of digits -/
theorem exists_divisible_by_sum_of_digits (n : ℕ) (h : 100 ≤ n ∧ n ≤ 982) :
  ∃ k : ℕ, n ≤ k ∧ k ≤ n + 17 ∧ k % sumOfDigits k = 0 :=
sorry

end NUMINAMATH_CALUDE_exists_divisible_by_sum_of_digits_l3972_397292


namespace NUMINAMATH_CALUDE_m_range_l3972_397204

-- Define the condition function
def condition (m : ℝ) : Set ℝ := {x | 1 - m < x ∧ x < 1 + m}

-- Define the inequality function
def inequality : Set ℝ := {x | (x - 1)^2 < 1}

-- Theorem statement
theorem m_range :
  ∀ m : ℝ, (condition m ⊆ inequality ∧ condition m ≠ inequality) → m ∈ Set.Ioo 0 1 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l3972_397204


namespace NUMINAMATH_CALUDE_intersection_problem_l3972_397232

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  equation : a * x + b * y + c = 0

/-- Checks if two lines intersect -/
def intersect (l1 l2 : Line) : Prop :=
  ¬ (l1.a * l2.b = l1.b * l2.a)

/-- The main line x + y - 1 = 0 -/
def main_line : Line :=
  { a := 1, b := 1, c := -1, equation := sorry }

/-- Line A: 2x + 2y = 6 -/
def line_a : Line :=
  { a := 2, b := 2, c := -6, equation := sorry }

/-- Line B: x + y = 0 -/
def line_b : Line :=
  { a := 1, b := 1, c := 0, equation := sorry }

/-- Line C: y = -x - 3 -/
def line_c : Line :=
  { a := 1, b := 1, c := 3, equation := sorry }

/-- Line D: y = x - 1 -/
def line_d : Line :=
  { a := 1, b := -1, c := 1, equation := sorry }

theorem intersection_problem :
  intersect main_line line_d ∧
  ¬ intersect main_line line_a ∧
  ¬ intersect main_line line_b ∧
  ¬ intersect main_line line_c :=
sorry

end NUMINAMATH_CALUDE_intersection_problem_l3972_397232


namespace NUMINAMATH_CALUDE_min_value_of_exponential_sum_l3972_397261

theorem min_value_of_exponential_sum (x y : ℝ) (h : x + 2 * y = 3) :
  ∃ (min : ℝ), min = 4 * Real.sqrt 2 ∧ ∀ (a b : ℝ), a + 2 * b = 3 → 2^a + 4^b ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_exponential_sum_l3972_397261


namespace NUMINAMATH_CALUDE_trapezoid_triangle_area_ratio_l3972_397213

/-- An inscribed acute-angled isosceles triangle in a circle -/
structure IsoscelesTriangle (α : ℝ) :=
  (angle_base : 0 < α ∧ α < π/2)

/-- An inscribed trapezoid in a circle -/
structure Trapezoid (α : ℝ) :=
  (base_is_diameter : True)
  (sides_parallel_to_triangle : True)

/-- The theorem stating that the area of the trapezoid equals the area of the triangle -/
theorem trapezoid_triangle_area_ratio 
  (α : ℝ) 
  (triangle : IsoscelesTriangle α) 
  (trapezoid : Trapezoid α) : 
  ∃ (area_trapezoid area_triangle : ℝ), 
    area_trapezoid / area_triangle = 1 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_triangle_area_ratio_l3972_397213


namespace NUMINAMATH_CALUDE_factorial_nine_mod_eleven_l3972_397259

theorem factorial_nine_mod_eleven : Nat.factorial 9 % 11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_factorial_nine_mod_eleven_l3972_397259


namespace NUMINAMATH_CALUDE_largest_four_digit_congruent_to_15_mod_22_l3972_397235

theorem largest_four_digit_congruent_to_15_mod_22 : ∃ (n : ℕ),
  n ≤ 9999 ∧
  n ≥ 1000 ∧
  n ≡ 15 [MOD 22] ∧
  (∀ m : ℕ, m ≤ 9999 ∧ m ≥ 1000 ∧ m ≡ 15 [MOD 22] → m ≤ n) ∧
  n = 9981 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_congruent_to_15_mod_22_l3972_397235


namespace NUMINAMATH_CALUDE_smartphone_charge_time_proof_l3972_397225

/-- The time in minutes to fully charge a smartphone -/
def smartphone_charge_time : ℝ := 26

/-- The time in minutes to fully charge a tablet -/
def tablet_charge_time : ℝ := 53

/-- The total time in minutes for Ana to charge her devices -/
def ana_charge_time : ℝ := 66

theorem smartphone_charge_time_proof :
  smartphone_charge_time = 26 :=
by
  have h1 : tablet_charge_time = 53 := rfl
  have h2 : tablet_charge_time + (1/2 * smartphone_charge_time) = ana_charge_time :=
    by sorry
  sorry

end NUMINAMATH_CALUDE_smartphone_charge_time_proof_l3972_397225


namespace NUMINAMATH_CALUDE_sally_buttons_theorem_l3972_397254

/-- The number of buttons needed for Sally's shirts -/
def buttons_needed (monday_shirts tuesday_shirts wednesday_shirts buttons_per_shirt : ℕ) : ℕ :=
  (monday_shirts + tuesday_shirts + wednesday_shirts) * buttons_per_shirt

/-- Theorem: Sally needs 45 buttons for all her shirts -/
theorem sally_buttons_theorem :
  buttons_needed 4 3 2 5 = 45 := by
  sorry

end NUMINAMATH_CALUDE_sally_buttons_theorem_l3972_397254


namespace NUMINAMATH_CALUDE_sams_french_bulldogs_count_l3972_397262

/-- The number of French Bulldogs Sam has -/
def sams_french_bulldogs : ℕ := 4

/-- The number of German Shepherds Sam has -/
def sams_german_shepherds : ℕ := 3

/-- The total number of dogs Peter wants -/
def peters_total_dogs : ℕ := 17

theorem sams_french_bulldogs_count :
  sams_french_bulldogs = 4 :=
by
  have h1 : peters_total_dogs = 3 * sams_german_shepherds + 2 * sams_french_bulldogs :=
    by sorry
  sorry

end NUMINAMATH_CALUDE_sams_french_bulldogs_count_l3972_397262


namespace NUMINAMATH_CALUDE_simplify_expression_l3972_397284

theorem simplify_expression (y : ℝ) : 4 * y^3 + 8 * y + 6 - (3 - 4 * y^3 - 8 * y) = 8 * y^3 + 16 * y + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3972_397284


namespace NUMINAMATH_CALUDE_catch_up_time_meeting_distance_l3972_397256

def distance_AB : ℝ := 46
def speed_A : ℝ := 15
def speed_B : ℝ := 40
def time_difference : ℝ := 1

-- Time for Person B to catch up with Person A
theorem catch_up_time : 
  ∃ t : ℝ, speed_B * t = speed_A * (t + time_difference) ∧ t = 3/5 := by sorry

-- Distance from point B where they meet on Person B's return journey
theorem meeting_distance : 
  ∃ y : ℝ, 
    (distance_AB - y) / speed_A - (distance_AB + y) / speed_B = time_difference ∧ 
    y = 10 := by sorry

end NUMINAMATH_CALUDE_catch_up_time_meeting_distance_l3972_397256


namespace NUMINAMATH_CALUDE_candy_cost_l3972_397267

theorem candy_cost (tickets_game1 tickets_game2 candies : ℕ) 
  (h1 : tickets_game1 = 33)
  (h2 : tickets_game2 = 9)
  (h3 : candies = 7) :
  (tickets_game1 + tickets_game2) / candies = 6 := by
  sorry

end NUMINAMATH_CALUDE_candy_cost_l3972_397267


namespace NUMINAMATH_CALUDE_four_machines_completion_time_l3972_397277

/-- A machine with a given work rate in jobs per hour -/
structure Machine where
  work_rate : ℚ

/-- The time taken for multiple machines to complete one job when working together -/
def time_to_complete (machines : List Machine) : ℚ :=
  1 / (machines.map (λ m => m.work_rate) |>.sum)

theorem four_machines_completion_time :
  let machine_a : Machine := ⟨1/4⟩
  let machine_b : Machine := ⟨1/2⟩
  let machine_c : Machine := ⟨1/6⟩
  let machine_d : Machine := ⟨1/3⟩
  let machines := [machine_a, machine_b, machine_c, machine_d]
  time_to_complete machines = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_four_machines_completion_time_l3972_397277


namespace NUMINAMATH_CALUDE_exam_outcomes_count_l3972_397253

/-- The number of possible outcomes for n people in a qualification exam -/
def exam_outcomes (n : ℕ) : ℕ := 2^n

/-- Theorem: The number of possible outcomes for n people in a qualification exam is 2^n -/
theorem exam_outcomes_count (n : ℕ) : exam_outcomes n = 2^n := by
  sorry

end NUMINAMATH_CALUDE_exam_outcomes_count_l3972_397253


namespace NUMINAMATH_CALUDE_t_plus_inverse_t_l3972_397264

theorem t_plus_inverse_t (t : ℝ) (h1 : t^2 - 3*t + 1 = 0) (h2 : t ≠ 0) : 
  t + 1/t = 3 := by
  sorry

end NUMINAMATH_CALUDE_t_plus_inverse_t_l3972_397264


namespace NUMINAMATH_CALUDE_haley_cider_pints_l3972_397263

/-- Represents the number of pints of cider Haley can make --/
def cider_pints (golden_apples_per_pint : ℕ) (pink_apples_per_pint : ℕ) 
  (apples_per_hour : ℕ) (farmhands : ℕ) (hours_worked : ℕ) 
  (golden_to_pink_ratio : ℚ) : ℕ :=
  let total_apples := apples_per_hour * farmhands * hours_worked
  let apples_per_pint := golden_apples_per_pint + pink_apples_per_pint
  total_apples / apples_per_pint

/-- Theorem stating the number of pints of cider Haley can make --/
theorem haley_cider_pints : 
  cider_pints 20 40 240 6 5 (1/3) = 120 := by
  sorry

end NUMINAMATH_CALUDE_haley_cider_pints_l3972_397263


namespace NUMINAMATH_CALUDE_steve_height_l3972_397230

/-- Converts feet and inches to total inches -/
def feet_to_inches (feet : ℕ) (inches : ℕ) : ℕ := feet * 12 + inches

/-- Calculates final height after growth -/
def final_height (initial_feet : ℕ) (initial_inches : ℕ) (growth : ℕ) : ℕ :=
  feet_to_inches initial_feet initial_inches + growth

theorem steve_height :
  final_height 5 6 6 = 72 := by sorry

end NUMINAMATH_CALUDE_steve_height_l3972_397230


namespace NUMINAMATH_CALUDE_natashas_average_speed_l3972_397220

/-- Natasha's hill climbing problem -/
theorem natashas_average_speed 
  (time_up : ℝ) 
  (time_down : ℝ) 
  (speed_up : ℝ) 
  (h1 : time_up = 4)
  (h2 : time_down = 2)
  (h3 : speed_up = 1.5)
  : (2 * speed_up * time_up) / (time_up + time_down) = 2 := by
  sorry

#check natashas_average_speed

end NUMINAMATH_CALUDE_natashas_average_speed_l3972_397220


namespace NUMINAMATH_CALUDE_tan_product_30_60_l3972_397255

theorem tan_product_30_60 : 
  (1 + Real.tan (30 * π / 180)) * (1 + Real.tan (60 * π / 180)) = 2 + 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_30_60_l3972_397255


namespace NUMINAMATH_CALUDE_data_properties_l3972_397238

def data : List ℝ := [3, 4, 2, 2, 4]

def median (l : List ℝ) : ℝ := sorry

def mean (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

def mode (l : List ℝ) : List ℝ := sorry

theorem data_properties :
  median data = 3 ∧
  mean data = 3 ∧
  variance data = 0.8 ∧
  mode data ≠ [4] :=
sorry

end NUMINAMATH_CALUDE_data_properties_l3972_397238


namespace NUMINAMATH_CALUDE_modulo_equivalence_in_range_l3972_397222

theorem modulo_equivalence_in_range : 
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 10 ∧ n ≡ 123456 [MOD 11] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_modulo_equivalence_in_range_l3972_397222


namespace NUMINAMATH_CALUDE_probability_both_selected_l3972_397275

theorem probability_both_selected (prob_X prob_Y prob_both : ℚ) : 
  prob_X = 1/7 → prob_Y = 2/9 → prob_both = prob_X * prob_Y → prob_both = 2/63 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_selected_l3972_397275


namespace NUMINAMATH_CALUDE_masking_tape_for_room_l3972_397285

/-- Calculates the amount of masking tape needed for a room with given dimensions --/
def masking_tape_needed (wall_width1 : ℝ) (wall_width2 : ℝ) (window_width : ℝ) (door_width : ℝ) : ℝ :=
  2 * (wall_width1 + wall_width2) - (2 * window_width + door_width)

/-- Theorem stating that the amount of masking tape needed for the given room is 15 meters --/
theorem masking_tape_for_room : masking_tape_needed 4 6 1.5 2 = 15 := by
  sorry

#check masking_tape_for_room

end NUMINAMATH_CALUDE_masking_tape_for_room_l3972_397285


namespace NUMINAMATH_CALUDE_nesbitt_inequality_l3972_397200

theorem nesbitt_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_nesbitt_inequality_l3972_397200


namespace NUMINAMATH_CALUDE_yellow_balls_percentage_l3972_397274

/-- The percentage of yellow balls in a collection of colored balls. -/
def percentage_yellow_balls (yellow brown blue green : ℕ) : ℚ :=
  (yellow : ℚ) / ((yellow + brown + blue + green : ℕ) : ℚ) * 100

/-- Theorem stating that the percentage of yellow balls is 25% given the specific numbers. -/
theorem yellow_balls_percentage :
  percentage_yellow_balls 75 120 45 60 = 25 := by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_percentage_l3972_397274


namespace NUMINAMATH_CALUDE_expressway_lengths_l3972_397295

theorem expressway_lengths (total : ℕ) (difference : ℕ) 
  (h1 : total = 519)
  (h2 : difference = 45) : 
  ∃ (new expanded : ℕ), 
    new + expanded = total ∧ 
    new = 2 * expanded - difference ∧
    new = 331 ∧ 
    expanded = 188 := by
  sorry

end NUMINAMATH_CALUDE_expressway_lengths_l3972_397295


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l3972_397266

/-- The axis of symmetry of the parabola y = 2x² is the line x = 0 -/
theorem parabola_axis_of_symmetry :
  let f : ℝ → ℝ := fun x ↦ 2 * x^2
  ∀ x y : ℝ, f (x) = f (-x) → x = 0 :=
by sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l3972_397266


namespace NUMINAMATH_CALUDE_dmv_waiting_time_l3972_397293

/-- Calculates the additional waiting time at the DMV -/
theorem dmv_waiting_time (initial_wait : ℕ) (total_wait : ℕ) : 
  initial_wait = 20 →
  total_wait = 114 →
  total_wait = initial_wait + 4 * initial_wait + (total_wait - (initial_wait + 4 * initial_wait)) →
  total_wait - (initial_wait + 4 * initial_wait) = 34 :=
by sorry

end NUMINAMATH_CALUDE_dmv_waiting_time_l3972_397293


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_inequality_system_solution_l3972_397229

-- Part 1: Quadratic equation
theorem quadratic_equation_solution :
  ∀ x : ℝ, x^2 - 6*x + 5 = 0 ↔ x = 1 ∨ x = 5 := by sorry

-- Part 2: System of inequalities
theorem inequality_system_solution :
  ∀ x : ℝ, (x + 3 > 0 ∧ 2*(x + 1) < 4) ↔ (-3 < x ∧ x < 1) := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_inequality_system_solution_l3972_397229


namespace NUMINAMATH_CALUDE_line_plane_parallelism_l3972_397226

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallelLine : Line → Line → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- State the theorem
theorem line_plane_parallelism 
  (m n : Line) (α : Plane) 
  (h1 : m ≠ n) 
  (h2 : parallelLine m n) 
  (h3 : parallelLinePlane m α) : 
  parallelLinePlane n α :=
sorry

end NUMINAMATH_CALUDE_line_plane_parallelism_l3972_397226


namespace NUMINAMATH_CALUDE_betty_herb_garden_l3972_397218

theorem betty_herb_garden (basil thyme oregano : ℕ) : 
  basil = 5 →
  thyme = 4 →
  oregano = 2 * basil + 2 →
  basil = 3 * thyme - 3 →
  basil + oregano + thyme = 21 := by
  sorry

end NUMINAMATH_CALUDE_betty_herb_garden_l3972_397218


namespace NUMINAMATH_CALUDE_similar_triangles_segment_length_l3972_397205

/-- Two triangles are similar -/
structure SimilarTriangles (T1 T2 : Type) :=
  (sim : T1 → T2 → Prop)

/-- Triangle GHI -/
structure TriangleGHI :=
  (G H I : ℝ)

/-- Triangle XYZ -/
structure TriangleXYZ :=
  (X Y Z : ℝ)

/-- The problem statement -/
theorem similar_triangles_segment_length 
  (tri_GHI : TriangleGHI) 
  (tri_XYZ : TriangleXYZ) 
  (sim : SimilarTriangles TriangleGHI TriangleXYZ) 
  (h_sim : sim.sim tri_GHI tri_XYZ)
  (h_GH : tri_GHI.H - tri_GHI.G = 8)
  (h_HI : tri_GHI.I - tri_GHI.H = 16)
  (h_YZ : tri_XYZ.Z - tri_XYZ.Y = 24) :
  tri_XYZ.Y - tri_XYZ.X = 12 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_segment_length_l3972_397205


namespace NUMINAMATH_CALUDE_cos_seven_theta_l3972_397233

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1/2) : Real.cos (7 * θ) = -37/128 := by
  sorry

end NUMINAMATH_CALUDE_cos_seven_theta_l3972_397233


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3972_397281

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
    a = 48 → 
    b = 64 → 
    c^2 = a^2 + b^2 → 
    c = 80 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3972_397281


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3972_397203

theorem rationalize_denominator : 7 / Real.sqrt 98 = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3972_397203


namespace NUMINAMATH_CALUDE_farmer_shipped_six_boxes_last_week_l3972_397257

/-- Represents the number of pomelos in a dozen -/
def dozen : ℕ := 12

/-- Represents the number of pomelos shipped last week -/
def pomelos_last_week : ℕ := 240

/-- Represents the number of boxes shipped this week -/
def boxes_this_week : ℕ := 20

/-- Represents the total number of dozens of pomelos shipped -/
def total_dozens : ℕ := 60

/-- Represents the number of boxes shipped last week -/
def boxes_last_week : ℕ := 6

/-- Proves that the farmer shipped 6 boxes last week given the conditions -/
theorem farmer_shipped_six_boxes_last_week :
  let total_pomelos := total_dozens * dozen
  let pomelos_this_week := total_pomelos - pomelos_last_week
  let pomelos_per_box := pomelos_this_week / boxes_this_week
  pomelos_last_week / pomelos_per_box = boxes_last_week :=
by sorry

end NUMINAMATH_CALUDE_farmer_shipped_six_boxes_last_week_l3972_397257


namespace NUMINAMATH_CALUDE_parabola_directrix_l3972_397224

/-- Represents a parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The directrix of a parabola -/
def directrix (p : Parabola) : ℝ := sorry

/-- Theorem: For a parabola with equation y = -1/8 x^2, its directrix has the equation y = 2 -/
theorem parabola_directrix :
  let p : Parabola := { a := -1/8, b := 0, c := 0 }
  directrix p = 2 := by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3972_397224


namespace NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l3972_397234

theorem neither_sufficient_nor_necessary (x : ℝ) : 
  ¬(((x - 1) * (x + 3) < 0 → (x + 1) * (x - 3) < 0) ∧ 
    ((x + 1) * (x - 3) < 0 → (x - 1) * (x + 3) < 0)) :=
sorry

end NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l3972_397234


namespace NUMINAMATH_CALUDE_number_wall_solution_l3972_397258

structure NumberWall :=
  (x : ℤ)
  (a b c d : ℤ)
  (e f g : ℤ)
  (h i : ℤ)
  (j : ℤ)

def NumberWall.valid (w : NumberWall) : Prop :=
  w.e = w.x + w.a ∧
  w.f = w.a + w.b ∧
  w.g = w.b + w.c ∧
  w.d = w.c + w.d ∧
  w.h = w.e + w.f ∧
  w.i = w.g + w.d ∧
  w.j = w.h + w.i ∧
  w.a = 5 ∧
  w.b = 10 ∧
  w.c = 9 ∧
  w.d = 6 ∧
  w.i = 18 ∧
  w.j = 72

theorem number_wall_solution (w : NumberWall) (h : w.valid) : w.x = -50 := by
  sorry

end NUMINAMATH_CALUDE_number_wall_solution_l3972_397258


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l3972_397252

theorem area_between_concentric_circles :
  let r₁ : ℝ := 12  -- radius of larger circle
  let r₂ : ℝ := 7   -- radius of smaller circle
  let A₁ := π * r₁^2  -- area of larger circle
  let A₂ := π * r₂^2  -- area of smaller circle
  A₁ - A₂ = 95 * π := by sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l3972_397252


namespace NUMINAMATH_CALUDE_line_passes_through_point_l3972_397207

theorem line_passes_through_point (a b : ℝ) (h : 3 * a + 2 * b = 5) :
  a * 6 + b * 4 - 10 = 0 := by
sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l3972_397207


namespace NUMINAMATH_CALUDE_logarithm_sum_simplification_l3972_397283

theorem logarithm_sum_simplification :
  let a := (1 / (Real.log 3 / Real.log 21 + 1))
  let b := (1 / (Real.log 4 / Real.log 14 + 1))
  let c := (1 / (Real.log 7 / Real.log 9 + 1))
  let d := (1 / (Real.log 11 / Real.log 8 + 1))
  a + b + c + d = 1 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_simplification_l3972_397283


namespace NUMINAMATH_CALUDE_count_valid_permutations_l3972_397219

/-- The set of digits in the number 2033 -/
def digits : Finset ℕ := {2, 0, 3, 3}

/-- A function that checks if a number is a 4-digit number -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A function that calculates the sum of digits of a number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- The set of all 4-digit permutations of the digits in 2033 -/
def valid_permutations : Finset ℕ := sorry

theorem count_valid_permutations : Finset.card valid_permutations = 15 := by sorry

end NUMINAMATH_CALUDE_count_valid_permutations_l3972_397219


namespace NUMINAMATH_CALUDE_log_expression_equals_two_l3972_397298

theorem log_expression_equals_two :
  (Real.log 2) ^ 2 + (Real.log 2) * (Real.log 50) + Real.log 25 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_two_l3972_397298


namespace NUMINAMATH_CALUDE_existence_of_critical_point_and_upper_bound_l3972_397223

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := exp x - a * x^2 - 2 * x - 1

theorem existence_of_critical_point_and_upper_bound (a : ℝ) (h : 1 < a ∧ a < 2) :
  ∃ x₀ : ℝ, x₀ ∈ Set.Ioo (-1/a) (-1/4) ∧ 
    (deriv (f a)) x₀ = 0 ∧ 
    f a x₀ < 15/16 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_critical_point_and_upper_bound_l3972_397223


namespace NUMINAMATH_CALUDE_odd_function_fixed_point_l3972_397208

/-- A function f : ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The theorem states that if f is an odd function on ℝ,
    then (-1, -2) is a point on the graph of y = f(x+1) - 2 -/
theorem odd_function_fixed_point (f : ℝ → ℝ) (h : IsOdd f) :
  f 0 - 2 = -2 := by sorry

end NUMINAMATH_CALUDE_odd_function_fixed_point_l3972_397208


namespace NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l3972_397294

/-- The volume of a cube given its space diagonal length. -/
theorem cube_volume_from_space_diagonal (d : ℝ) (h : d = 6 * Real.sqrt 3) :
  (d / Real.sqrt 3) ^ 3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l3972_397294


namespace NUMINAMATH_CALUDE_two_times_binomial_seven_choose_four_l3972_397299

theorem two_times_binomial_seven_choose_four : 2 * (Nat.choose 7 4) = 70 := by
  sorry

end NUMINAMATH_CALUDE_two_times_binomial_seven_choose_four_l3972_397299
