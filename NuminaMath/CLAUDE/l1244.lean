import Mathlib

namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l1244_124446

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d

theorem fifth_term_of_sequence (x y : ℚ) (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a)
  (h_first : a 0 = x + 2*y)
  (h_second : a 1 = x - 2*y)
  (h_third : a 2 = x + 2*y^2)
  (h_fourth : a 3 = x / (2*y))
  (h_y_nonzero : y ≠ 0) :
  a 4 = -x/6 - 12 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l1244_124446


namespace NUMINAMATH_CALUDE_correct_managers_in_sample_l1244_124432

/-- Calculates the number of managers to be drawn in a stratified sample -/
def managers_in_sample (total_employees : ℕ) (total_managers : ℕ) (sample_size : ℕ) : ℕ :=
  (total_managers * sample_size) / total_employees

theorem correct_managers_in_sample :
  managers_in_sample 160 32 20 = 4 := by
  sorry

end NUMINAMATH_CALUDE_correct_managers_in_sample_l1244_124432


namespace NUMINAMATH_CALUDE_exists_one_one_appended_one_l1244_124412

def is_valid_number (n : ℕ) (num : List ℕ) : Prop :=
  num.length = n ∧ ∀ d ∈ num, d = 1 ∨ d = 2 ∨ d = 3

def differs_in_all_positions (n : ℕ) (num1 num2 : List ℕ) : Prop :=
  is_valid_number n num1 ∧ is_valid_number n num2 ∧
  ∀ i, i < n → num1.get ⟨i, by sorry⟩ ≠ num2.get ⟨i, by sorry⟩

def appended_digit (n : ℕ) (num : List ℕ) (d : ℕ) : Prop :=
  is_valid_number n num ∧ (d = 1 ∨ d = 2 ∨ d = 3)

def valid_appending (n : ℕ) (append : List ℕ → ℕ) : Prop :=
  ∀ num1 num2 : List ℕ, differs_in_all_positions n num1 num2 →
    append num1 ≠ append num2

theorem exists_one_one_appended_one (n : ℕ) :
  ∃ (append : List ℕ → ℕ),
    valid_appending n append →
    ∃ (num : List ℕ),
      is_valid_number n num ∧
      (num.count 1 = 1) ∧
      (append num = 1) := by sorry

end NUMINAMATH_CALUDE_exists_one_one_appended_one_l1244_124412


namespace NUMINAMATH_CALUDE_store_inventory_theorem_l1244_124430

/-- Represents the inventory of a store --/
structure Inventory where
  headphones : ℕ
  mice : ℕ
  keyboards : ℕ
  keyboard_mouse_sets : ℕ
  headphone_mouse_sets : ℕ

/-- Calculates the number of ways to buy headphones, keyboard, and mouse --/
def ways_to_buy (inv : Inventory) : ℕ :=
  inv.keyboard_mouse_sets * inv.headphones +
  inv.headphone_mouse_sets * inv.keyboards +
  inv.headphones * inv.mice * inv.keyboards

/-- The theorem stating the number of ways to buy the items --/
theorem store_inventory_theorem (inv : Inventory) 
  (h1 : inv.headphones = 9)
  (h2 : inv.mice = 13)
  (h3 : inv.keyboards = 5)
  (h4 : inv.keyboard_mouse_sets = 4)
  (h5 : inv.headphone_mouse_sets = 5) :
  ways_to_buy inv = 646 := by
  sorry

#eval ways_to_buy { headphones := 9, mice := 13, keyboards := 5, keyboard_mouse_sets := 4, headphone_mouse_sets := 5 }

end NUMINAMATH_CALUDE_store_inventory_theorem_l1244_124430


namespace NUMINAMATH_CALUDE_remainder_proof_l1244_124459

theorem remainder_proof (g : ℕ) (h : g = 144) :
  (6215 % g = 23) ∧ (7373 % g = 29) ∧
  (∀ d : ℕ, d > g → (6215 % d ≠ 6215 % g ∨ 7373 % d ≠ 7373 % g)) := by
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l1244_124459


namespace NUMINAMATH_CALUDE_range_of_a_l1244_124418

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 1

-- Define set A
def A (a : ℝ) : Set ℝ := {x | f a x = x}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | f a (f a x) = x}

-- Theorem statement
theorem range_of_a (a : ℝ) (h1 : A a = B a) (h2 : (A a).Nonempty) :
  a ∈ Set.Icc (-1/4 : ℝ) (3/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1244_124418


namespace NUMINAMATH_CALUDE_alice_bob_sum_l1244_124456

/-- A number is prime if it's greater than 1 and has no positive divisors other than 1 and itself. -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- A number is a perfect square if it's the product of an integer with itself. -/
def isPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem alice_bob_sum : 
  ∀ (A B : ℕ),
  (A ≠ 1) →  -- Alice's number is not the smallest
  (B = 2) →  -- Bob's number is the smallest prime
  (isPrime B) →
  (isPerfectSquare (100 * B + A)) →
  (1 ≤ A ∧ A ≤ 40) →  -- Alice's number is between 1 and 40
  (1 ≤ B ∧ B ≤ 40) →  -- Bob's number is between 1 and 40
  (A + B = 27) := by sorry

end NUMINAMATH_CALUDE_alice_bob_sum_l1244_124456


namespace NUMINAMATH_CALUDE_pet_ownership_l1244_124483

theorem pet_ownership (total_students : ℕ) 
  (dog_owners cat_owners bird_owners fish_only_owners no_pet_owners : ℕ) : 
  total_students = 40 →
  dog_owners = (40 * 5) / 8 →
  cat_owners = 40 / 2 →
  bird_owners = 40 / 4 →
  fish_only_owners = 8 →
  no_pet_owners = 6 →
  ∃ (all_pet_owners : ℕ), all_pet_owners = 6 ∧
    all_pet_owners + fish_only_owners + no_pet_owners ≤ total_students :=
by sorry

end NUMINAMATH_CALUDE_pet_ownership_l1244_124483


namespace NUMINAMATH_CALUDE_decimal_division_to_percentage_l1244_124480

theorem decimal_division_to_percentage : (0.15 / 0.005) * 100 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_division_to_percentage_l1244_124480


namespace NUMINAMATH_CALUDE_set_problem_l1244_124499

def U : Set ℕ := {x | x ≤ 20 ∧ Nat.Prime x}

theorem set_problem (A B : Set ℕ)
  (h1 : A ∩ (U \ B) = {3, 5})
  (h2 : (U \ A) ∩ B = {7, 19})
  (h3 : U \ (A ∪ B) = {2, 17}) :
  A = {3, 5, 11, 13} ∧ B = {7, 19, 11, 13} := by
  sorry

end NUMINAMATH_CALUDE_set_problem_l1244_124499


namespace NUMINAMATH_CALUDE_correct_group_capacity_l1244_124439

/-- The capacity of each group in a systematic sampling -/
def group_capacity (total_students : ℕ) (sample_size : ℕ) : ℕ :=
  (total_students - (total_students % sample_size)) / sample_size

/-- Theorem stating the correct group capacity for the given problem -/
theorem correct_group_capacity :
  group_capacity 5008 200 = 25 := by
  sorry

end NUMINAMATH_CALUDE_correct_group_capacity_l1244_124439


namespace NUMINAMATH_CALUDE_fraction_simplification_l1244_124464

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -3) :
  (3*x^2 + x) / ((x - 1) * (x + 3)) + (5 - x) / ((x - 1) * (x + 3)) =
  (3*x^2 + 4) / ((x - 1) * (x + 3)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1244_124464


namespace NUMINAMATH_CALUDE_marty_stripes_l1244_124493

/-- The number of narrow black stripes on Marty the zebra -/
def narrow_black_stripes : ℕ := 8

/-- The number of wide black stripes on Marty the zebra -/
def wide_black_stripes : ℕ := sorry

/-- The number of white stripes on Marty the zebra -/
def white_stripes : ℕ := wide_black_stripes + 7

/-- The total number of black stripes on Marty the zebra -/
def total_black_stripes : ℕ := wide_black_stripes + narrow_black_stripes

theorem marty_stripes : 
  total_black_stripes = white_stripes + 1 → 
  narrow_black_stripes = 8 := by
  sorry

end NUMINAMATH_CALUDE_marty_stripes_l1244_124493


namespace NUMINAMATH_CALUDE_f_monotonic_implies_a_range_l1244_124409

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*(a-1)*x + 2

-- Define monotonicity on an interval
def monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y) ∨
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x)

-- Theorem statement
theorem f_monotonic_implies_a_range :
  ∀ a : ℝ, monotonic_on (f a) 2 4 → a ≤ 3 ∨ a ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_f_monotonic_implies_a_range_l1244_124409


namespace NUMINAMATH_CALUDE_river_width_calculation_l1244_124467

/-- Given a river with depth, flow rate, and volume flow per minute, calculate its width. -/
theorem river_width_calculation (depth : ℝ) (flow_rate_kmph : ℝ) (volume_flow : ℝ) :
  depth = 3 →
  flow_rate_kmph = 2 →
  volume_flow = 3600 →
  (volume_flow / (depth * (flow_rate_kmph * 1000 / 60))) = 36 := by
  sorry

end NUMINAMATH_CALUDE_river_width_calculation_l1244_124467


namespace NUMINAMATH_CALUDE_expression_evaluation_l1244_124481

theorem expression_evaluation :
  let x : ℚ := -2
  let y : ℚ := 1/2
  (4 * x^2 * y - (6 * x * y - 3 * (4 * x - 2) - x^2 * y) + 1) = -13 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1244_124481


namespace NUMINAMATH_CALUDE_total_earnings_equals_9780_l1244_124490

-- Define the earnings of each individual
def salvadore_earnings : ℕ := 1956

-- Santo's earnings are half of Salvadore's
def santo_earnings : ℕ := salvadore_earnings / 2

-- Maria's earnings are three times Santo's
def maria_earnings : ℕ := santo_earnings * 3

-- Pedro's earnings are the sum of Santo's and Maria's
def pedro_earnings : ℕ := santo_earnings + maria_earnings

-- Total earnings of all four individuals
def total_earnings : ℕ := salvadore_earnings + santo_earnings + maria_earnings + pedro_earnings

-- Theorem statement
theorem total_earnings_equals_9780 : total_earnings = 9780 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_equals_9780_l1244_124490


namespace NUMINAMATH_CALUDE_pages_revised_once_is_30_l1244_124465

/-- Represents the typing service problem --/
structure TypingService where
  totalPages : ℕ
  pagesRevisedTwice : ℕ
  totalCost : ℕ
  firstTypingRate : ℕ
  revisionRate : ℕ

/-- Calculates the number of pages revised once --/
def pagesRevisedOnce (ts : TypingService) : ℕ :=
  ((ts.totalCost - ts.firstTypingRate * ts.totalPages - 
    ts.revisionRate * ts.pagesRevisedTwice * 2) / ts.revisionRate)

/-- Theorem stating the number of pages revised once --/
theorem pages_revised_once_is_30 (ts : TypingService) 
  (h1 : ts.totalPages = 100)
  (h2 : ts.pagesRevisedTwice = 20)
  (h3 : ts.totalCost = 1350)
  (h4 : ts.firstTypingRate = 10)
  (h5 : ts.revisionRate = 5) :
  pagesRevisedOnce ts = 30 := by
  sorry

#eval pagesRevisedOnce {
  totalPages := 100,
  pagesRevisedTwice := 20,
  totalCost := 1350,
  firstTypingRate := 10,
  revisionRate := 5
}

end NUMINAMATH_CALUDE_pages_revised_once_is_30_l1244_124465


namespace NUMINAMATH_CALUDE_sum_lower_bound_l1244_124437

noncomputable def f (x : ℝ) := Real.log x + x^2 + x

theorem sum_lower_bound (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0)
  (h : f x₁ + f x₂ + x₁ * x₂ = 0) :
  x₁ + x₂ ≥ (Real.sqrt 5 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_sum_lower_bound_l1244_124437


namespace NUMINAMATH_CALUDE_exponential_plus_x_increasing_l1244_124474

open Real

theorem exponential_plus_x_increasing (x : ℝ) : exp (x + 1) + (x + 1) > (exp x + x) + 1 := by
  sorry

end NUMINAMATH_CALUDE_exponential_plus_x_increasing_l1244_124474


namespace NUMINAMATH_CALUDE_optimal_selection_l1244_124408

/-- Represents a 5x5 matrix of integers -/
def Matrix5x5 : Type := Fin 5 → Fin 5 → ℤ

/-- The given matrix -/
def givenMatrix : Matrix5x5 :=
  λ i j => match i, j with
  | ⟨0, _⟩, ⟨0, _⟩ => 11 | ⟨0, _⟩, ⟨1, _⟩ => 17 | ⟨0, _⟩, ⟨2, _⟩ => 25 | ⟨0, _⟩, ⟨3, _⟩ => 19 | ⟨0, _⟩, ⟨4, _⟩ => 16
  | ⟨1, _⟩, ⟨0, _⟩ => 24 | ⟨1, _⟩, ⟨1, _⟩ => 10 | ⟨1, _⟩, ⟨2, _⟩ => 13 | ⟨1, _⟩, ⟨3, _⟩ => 15 | ⟨1, _⟩, ⟨4, _⟩ => 3
  | ⟨2, _⟩, ⟨0, _⟩ => 12 | ⟨2, _⟩, ⟨1, _⟩ => 5  | ⟨2, _⟩, ⟨2, _⟩ => 14 | ⟨2, _⟩, ⟨3, _⟩ => 2  | ⟨2, _⟩, ⟨4, _⟩ => 18
  | ⟨3, _⟩, ⟨0, _⟩ => 23 | ⟨3, _⟩, ⟨1, _⟩ => 4  | ⟨3, _⟩, ⟨2, _⟩ => 1  | ⟨3, _⟩, ⟨3, _⟩ => 8  | ⟨3, _⟩, ⟨4, _⟩ => 22
  | ⟨4, _⟩, ⟨0, _⟩ => 6  | ⟨4, _⟩, ⟨1, _⟩ => 20 | ⟨4, _⟩, ⟨2, _⟩ => 7  | ⟨4, _⟩, ⟨3, _⟩ => 21 | ⟨4, _⟩, ⟨4, _⟩ => 9
  | _, _ => 0

/-- A selection of 5 elements from the matrix -/
def Selection : Type := Fin 5 → (Fin 5 × Fin 5)

/-- Check if a selection is valid (no two elements in same row or column) -/
def isValidSelection (s : Selection) : Prop :=
  ∀ i j, i ≠ j → (s i).1 ≠ (s j).1 ∧ (s i).2 ≠ (s j).2

/-- The claimed optimal selection -/
def claimedOptimalSelection : Selection :=
  λ i => match i with
  | ⟨0, _⟩ => (⟨0, by norm_num⟩, ⟨2, by norm_num⟩)  -- 25
  | ⟨1, _⟩ => (⟨4, by norm_num⟩, ⟨1, by norm_num⟩)  -- 20
  | ⟨2, _⟩ => (⟨3, by norm_num⟩, ⟨0, by norm_num⟩)  -- 23
  | ⟨3, _⟩ => (⟨2, by norm_num⟩, ⟨4, by norm_num⟩)  -- 18
  | ⟨4, _⟩ => (⟨1, by norm_num⟩, ⟨3, by norm_num⟩)  -- 15

/-- The theorem to prove -/
theorem optimal_selection :
  isValidSelection claimedOptimalSelection ∧
  (∀ s : Selection, isValidSelection s →
    (∃ i, givenMatrix (s i).1 (s i).2 ≤ givenMatrix (claimedOptimalSelection 4).1 (claimedOptimalSelection 4).2)) :=
by sorry

end NUMINAMATH_CALUDE_optimal_selection_l1244_124408


namespace NUMINAMATH_CALUDE_exam_average_l1244_124486

theorem exam_average (total_boys : ℕ) (passed_boys : ℕ) (avg_passed : ℕ) (avg_failed : ℕ)
  (h1 : total_boys = 120)
  (h2 : passed_boys = 100)
  (h3 : avg_passed = 39)
  (h4 : avg_failed = 15) :
  (avg_passed * passed_boys + avg_failed * (total_boys - passed_boys)) / total_boys = 35 := by
sorry

end NUMINAMATH_CALUDE_exam_average_l1244_124486


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l1244_124402

theorem fractional_equation_solution :
  ∃! x : ℝ, x ≠ 3 ∧ (1 - x) / (x - 3) = 1 / (3 - x) - 2 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l1244_124402


namespace NUMINAMATH_CALUDE_impossible_to_use_all_stock_l1244_124442

/-- Represents the number of units required for each product type -/
structure ProductRequirements where
  alpha_A : Nat
  alpha_B : Nat
  beta_B : Nat
  beta_C : Nat
  gamma_A : Nat
  gamma_C : Nat

/-- Represents the current stock levels after production -/
structure StockLevels where
  remaining_A : Nat
  remaining_B : Nat
  remaining_C : Nat

/-- Theorem stating the impossibility of using up all stocks exactly -/
theorem impossible_to_use_all_stock 
  (req : ProductRequirements)
  (stock : StockLevels)
  (h_req : req = { 
    alpha_A := 2, alpha_B := 2, 
    beta_B := 1, beta_C := 1, 
    gamma_A := 2, gamma_C := 1 
  })
  (h_stock : stock = { remaining_A := 2, remaining_B := 1, remaining_C := 0 }) :
  ∀ (p q r : Nat), ∃ (total_A total_B total_C : Nat),
    (2 * p + 2 * r + stock.remaining_A ≠ total_A) ∨
    (2 * p + q + stock.remaining_B ≠ total_B) ∨
    (q + r ≠ total_C) :=
sorry

end NUMINAMATH_CALUDE_impossible_to_use_all_stock_l1244_124442


namespace NUMINAMATH_CALUDE_lucy_fish_goal_l1244_124410

/-- The number of fish Lucy currently has -/
def current_fish : ℕ := 212

/-- The number of additional fish Lucy needs to buy -/
def additional_fish : ℕ := 68

/-- The total number of fish Lucy wants to have -/
def total_fish : ℕ := current_fish + additional_fish

theorem lucy_fish_goal : total_fish = 280 := by
  sorry

end NUMINAMATH_CALUDE_lucy_fish_goal_l1244_124410


namespace NUMINAMATH_CALUDE_jacket_trouser_combinations_l1244_124496

theorem jacket_trouser_combinations (jacket_styles : ℕ) (trouser_colors : ℕ) : 
  jacket_styles = 4 → trouser_colors = 3 → jacket_styles * trouser_colors = 12 := by
  sorry

end NUMINAMATH_CALUDE_jacket_trouser_combinations_l1244_124496


namespace NUMINAMATH_CALUDE_correct_age_ranking_l1244_124440

-- Define the set of friends
inductive Friend : Type
| David : Friend
| Emma : Friend
| Fiona : Friend
| George : Friend

-- Define the age relation
def OlderThan : Friend → Friend → Prop := sorry

-- Define the statements
def Statement1 : Prop := ∀ f : Friend, f ≠ Friend.Emma → OlderThan Friend.Emma f
def Statement2 : Prop := ∃ f : Friend, OlderThan f Friend.Fiona
def Statement3 : Prop := ∃ f : Friend, OlderThan Friend.David f
def Statement4 : Prop := ∃ f : Friend, OlderThan f Friend.George

-- Define the theorem
theorem correct_age_ranking :
  (∀ f1 f2 : Friend, f1 ≠ f2 → (OlderThan f1 f2 ∨ OlderThan f2 f1)) →
  (Statement1 ∨ Statement2 ∨ Statement3 ∨ Statement4) →
  (¬Statement1 ∨ ¬Statement2 ∨ ¬Statement3 ∨ ¬Statement4) →
  (OlderThan Friend.Fiona Friend.Emma ∧
   OlderThan Friend.Emma Friend.George ∧
   OlderThan Friend.George Friend.David) :=
by sorry

end NUMINAMATH_CALUDE_correct_age_ranking_l1244_124440


namespace NUMINAMATH_CALUDE_investment_problem_l1244_124479

/-- Proves that the total investment amount is $5,400 given the problem conditions -/
theorem investment_problem (total : ℝ) (amount_at_8_percent : ℝ) (amount_at_10_percent : ℝ)
  (h1 : amount_at_8_percent = 3000)
  (h2 : total = amount_at_8_percent + amount_at_10_percent)
  (h3 : amount_at_8_percent * 0.08 = amount_at_10_percent * 0.10) :
  total = 5400 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l1244_124479


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1244_124462

/-- The length of the major axis of the ellipse x²/9 + y²/4 = 1 is 6 -/
theorem ellipse_major_axis_length : 
  let ellipse := fun (x y : ℝ) => x^2/9 + y^2/4 = 1
  ∃ (a b : ℝ), a > b ∧ a^2 = 9 ∧ b^2 = 4 ∧ 2*a = 6 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1244_124462


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1244_124406

-- Define set A
def A : Set ℕ := {4, 5, 6, 7}

-- Define set B
def B : Set ℕ := {x | 3 ≤ x ∧ x < 6}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {4, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1244_124406


namespace NUMINAMATH_CALUDE_tangent_product_l1244_124441

theorem tangent_product (x y : ℝ) 
  (h1 : Real.tan x - Real.tan y = 7)
  (h2 : 2 * Real.sin (2*x - 2*y) = Real.sin (2*x) * Real.sin (2*y)) :
  Real.tan x * Real.tan y = -7/6 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_l1244_124441


namespace NUMINAMATH_CALUDE_profit_and_max_profit_l1244_124420

/-- Represents the average daily profit as a function of price reduction --/
def averageDailyProfit (x : ℝ) : ℝ := -2 * x^2 + 60 * x + 800

/-- The price reduction that results in $1200 average daily profit --/
def priceReductionFor1200Profit : ℝ := 20

/-- The price reduction that maximizes average daily profit --/
def priceReductionForMaxProfit : ℝ := 15

/-- The maximum average daily profit --/
def maxAverageDailyProfit : ℝ := 1250

theorem profit_and_max_profit :
  (averageDailyProfit priceReductionFor1200Profit = 1200) ∧
  (∀ x : ℝ, averageDailyProfit x ≤ maxAverageDailyProfit) ∧
  (averageDailyProfit priceReductionForMaxProfit = maxAverageDailyProfit) := by
  sorry


end NUMINAMATH_CALUDE_profit_and_max_profit_l1244_124420


namespace NUMINAMATH_CALUDE_smallest_positive_period_of_cosine_l1244_124466

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x)

theorem smallest_positive_period_of_cosine 
  (ω : ℝ) 
  (h_ω_pos : ω > 0) 
  (P : ℝ × ℝ) 
  (h_center_symmetry : ∀ x, f ω (2 * P.1 - x) = f ω x) 
  (h_min_distance : ∀ y, abs (P.2 - y) ≥ π) :
  ∃ T > 0, (∀ x, f ω (x + T) = f ω x) ∧ 
  (∀ S, S > 0 → (∀ x, f ω (x + S) = f ω x) → S ≥ T) ∧ 
  T = 2 * π := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_period_of_cosine_l1244_124466


namespace NUMINAMATH_CALUDE_two_digit_factorizations_of_2079_l1244_124421

/-- A factorization of a number into two factors -/
structure Factorization :=
  (factor1 : ℕ)
  (factor2 : ℕ)

/-- Check if a number is two-digit (between 10 and 99, inclusive) -/
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Check if a factorization is valid for 2079 with two-digit factors -/
def isValidFactorization (f : Factorization) : Prop :=
  f.factor1 * f.factor2 = 2079 ∧ isTwoDigit f.factor1 ∧ isTwoDigit f.factor2

/-- Two factorizations are considered equal if they have the same factors (in any order) -/
def factorizationEqual (f1 f2 : Factorization) : Prop :=
  (f1.factor1 = f2.factor1 ∧ f1.factor2 = f2.factor2) ∨
  (f1.factor1 = f2.factor2 ∧ f1.factor2 = f2.factor1)

/-- The main theorem: there are exactly 2 unique factorizations of 2079 into two-digit numbers -/
theorem two_digit_factorizations_of_2079 :
  ∃ (f1 f2 : Factorization),
    isValidFactorization f1 ∧
    isValidFactorization f2 ∧
    ¬factorizationEqual f1 f2 ∧
    ∀ (f : Factorization), isValidFactorization f → (factorizationEqual f f1 ∨ factorizationEqual f f2) :=
  sorry

end NUMINAMATH_CALUDE_two_digit_factorizations_of_2079_l1244_124421


namespace NUMINAMATH_CALUDE_straight_line_angle_l1244_124485

/-- 
Given a straight line segment PQ with angle measurements of 90°, x°, and 20° along it,
prove that x = 70°.
-/
theorem straight_line_angle (x : ℝ) : 
  (90 : ℝ) + x + 20 = 180 → x = 70 := by
  sorry

end NUMINAMATH_CALUDE_straight_line_angle_l1244_124485


namespace NUMINAMATH_CALUDE_max_followers_after_three_weeks_l1244_124472

def susyInitialFollowers : ℕ := 100
def sarahInitialFollowers : ℕ := 50

def susyWeek1Gain : ℕ := 40
def sarahWeek1Gain : ℕ := 90

def susyTotalFollowers : ℕ := 
  susyInitialFollowers + susyWeek1Gain + (susyWeek1Gain / 2) + (susyWeek1Gain / 4)

def sarahTotalFollowers : ℕ := 
  sarahInitialFollowers + sarahWeek1Gain + (sarahWeek1Gain / 3) + (sarahWeek1Gain / 9)

theorem max_followers_after_three_weeks :
  max susyTotalFollowers sarahTotalFollowers = 180 := by
  sorry

end NUMINAMATH_CALUDE_max_followers_after_three_weeks_l1244_124472


namespace NUMINAMATH_CALUDE_cut_tetrahedron_edge_count_l1244_124476

/-- Represents a regular tetrahedron with its vertices cut off. -/
structure CutTetrahedron where
  /-- The number of vertices in the original tetrahedron -/
  original_vertices : Nat
  /-- The number of edges in the original tetrahedron -/
  original_edges : Nat
  /-- The number of new edges created by each cut -/
  new_edges_per_cut : Nat
  /-- The cutting planes do not intersect on the solid -/
  non_intersecting_cuts : Prop

/-- The number of edges in the new figure after cutting off each vertex -/
def edge_count (t : CutTetrahedron) : Nat :=
  t.original_edges + t.original_vertices * t.new_edges_per_cut

/-- Theorem stating that a regular tetrahedron with its vertices cut off has 18 edges -/
theorem cut_tetrahedron_edge_count :
  ∀ (t : CutTetrahedron),
    t.original_vertices = 4 →
    t.original_edges = 6 →
    t.new_edges_per_cut = 3 →
    t.non_intersecting_cuts →
    edge_count t = 18 :=
  sorry

end NUMINAMATH_CALUDE_cut_tetrahedron_edge_count_l1244_124476


namespace NUMINAMATH_CALUDE_calculation_proof_l1244_124448

theorem calculation_proof : 101 * 102^2 - 101 * 98^2 = 80800 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1244_124448


namespace NUMINAMATH_CALUDE_magazine_purchase_methods_l1244_124455

theorem magazine_purchase_methods (n : ℕ) (m : ℕ) (total : ℕ) : 
  n + m = 11 → 
  n = 8 → 
  m = 3 → 
  total = 10 →
  (Nat.choose n 5 + Nat.choose n 4 * Nat.choose m 2) = 266 := by
  sorry

end NUMINAMATH_CALUDE_magazine_purchase_methods_l1244_124455


namespace NUMINAMATH_CALUDE_equal_days_count_l1244_124489

/-- Represents the days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Counts the number of Tuesdays and Thursdays in a 30-day month starting on the given day -/
def countTuesdaysAndThursdays (startDay : DayOfWeek) : Nat × Nat :=
  let rec count (currentDay : DayOfWeek) (daysLeft : Nat) (tuesdays : Nat) (thursdays : Nat) : Nat × Nat :=
    if daysLeft = 0 then
      (tuesdays, thursdays)
    else
      match currentDay with
      | DayOfWeek.Tuesday => count (nextDay currentDay) (daysLeft - 1) (tuesdays + 1) thursdays
      | DayOfWeek.Thursday => count (nextDay currentDay) (daysLeft - 1) tuesdays (thursdays + 1)
      | _ => count (nextDay currentDay) (daysLeft - 1) tuesdays thursdays
  count startDay 30 0 0

/-- Checks if the number of Tuesdays and Thursdays are equal for a given starting day -/
def hasEqualTuesdaysAndThursdays (startDay : DayOfWeek) : Bool :=
  let (tuesdays, thursdays) := countTuesdaysAndThursdays startDay
  tuesdays = thursdays

/-- Counts the number of days that result in equal Tuesdays and Thursdays -/
def countEqualDays : Nat :=
  let days := [DayOfWeek.Sunday, DayOfWeek.Monday, DayOfWeek.Tuesday, DayOfWeek.Wednesday,
               DayOfWeek.Thursday, DayOfWeek.Friday, DayOfWeek.Saturday]
  days.filter hasEqualTuesdaysAndThursdays |>.length

theorem equal_days_count :
  countEqualDays = 4 :=
sorry

end NUMINAMATH_CALUDE_equal_days_count_l1244_124489


namespace NUMINAMATH_CALUDE_largest_number_l1244_124417

theorem largest_number (a b c d : ℝ) : 
  a + (b + c + d) / 3 = 92 →
  b + (a + c + d) / 3 = 86 →
  c + (a + b + d) / 3 = 80 →
  d + (a + b + c) / 3 = 90 →
  max a (max b (max c d)) = 51 := by
sorry

end NUMINAMATH_CALUDE_largest_number_l1244_124417


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1244_124424

theorem inequality_solution_set (a : ℝ) (h : a < -1) :
  {x : ℝ | (a * x - 1) / (x + 1) < 0} = {x : ℝ | x < -1 ∨ x > 1/a} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1244_124424


namespace NUMINAMATH_CALUDE_lovers_watches_prime_sum_squares_l1244_124404

theorem lovers_watches_prime_sum_squares :
  ∃ (x y : Fin 12 → ℕ) (m : Fin 12 → ℕ),
    (∀ i : Fin 12, Nat.Prime (x i)) ∧
    (∀ i : Fin 12, Nat.Prime (y i)) ∧
    (∀ i j : Fin 12, i ≠ j → x i ≠ x j) ∧
    (∀ i j : Fin 12, i ≠ j → y i ≠ y j) ∧
    (∀ i : Fin 12, x i ≠ y i) ∧
    (∀ k : Fin 12, x k + x (k.succ) = y k + y (k.succ)) ∧
    (∀ k : Fin 12, ∃ (m_k : ℕ), x k + x (k.succ) = m_k ^ 2) :=
by sorry


end NUMINAMATH_CALUDE_lovers_watches_prime_sum_squares_l1244_124404


namespace NUMINAMATH_CALUDE_fraction_expression_l1244_124413

theorem fraction_expression : 
  (3/7 + 5/8) / (5/12 + 2/9) = 531/322 := by
  sorry

end NUMINAMATH_CALUDE_fraction_expression_l1244_124413


namespace NUMINAMATH_CALUDE_simplify_fraction_l1244_124450

theorem simplify_fraction (x y : ℝ) (hx : x = 3) (hy : y = 2) :
  15 * x^2 * y^3 / (9 * x * y^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1244_124450


namespace NUMINAMATH_CALUDE_max_value_of_a_l1244_124491

theorem max_value_of_a (a b c : ℝ) (sum_zero : a + b + c = 0) (sum_squares_one : a^2 + b^2 + c^2 = 1) :
  a ≤ Real.sqrt 6 / 3 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ + b₀ + c₀ = 0 ∧ a₀^2 + b₀^2 + c₀^2 = 1 ∧ a₀ = Real.sqrt 6 / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l1244_124491


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l1244_124426

theorem quadratic_roots_range (k : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + (k-2)*x + 2*k-1 = 0 ↔ x = x₁ ∨ x = x₂) →
  0 < x₁ → x₁ < 1 → 1 < x₂ → x₂ < 2 →
  1/2 < k ∧ k < 2/3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l1244_124426


namespace NUMINAMATH_CALUDE_bricks_for_room_floor_bricks_needed_is_340_l1244_124484

/-- Calculates the number of bricks needed for a rectangular room floor -/
theorem bricks_for_room_floor 
  (length : ℝ) 
  (breadth : ℝ) 
  (bricks_per_sqm : ℕ) 
  (h1 : length = 4) 
  (h2 : breadth = 5) 
  (h3 : bricks_per_sqm = 17) : 
  ℕ := by
  
  sorry

#check bricks_for_room_floor

/-- Proves that 340 bricks are needed for the given room dimensions -/
theorem bricks_needed_is_340 : 
  bricks_for_room_floor 4 5 17 rfl rfl rfl = 340 := by
  
  sorry

end NUMINAMATH_CALUDE_bricks_for_room_floor_bricks_needed_is_340_l1244_124484


namespace NUMINAMATH_CALUDE_sum_ten_consecutive_naturals_odd_l1244_124405

theorem sum_ten_consecutive_naturals_odd (n : ℕ) : ∃ k : ℕ, (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) + (n + 7) + (n + 8) + (n + 9)) = 2 * k + 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_ten_consecutive_naturals_odd_l1244_124405


namespace NUMINAMATH_CALUDE_supremum_of_expression_l1244_124487

theorem supremum_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  -1 / (2 * a) - 2 / b ≤ -9 / 2 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ -1 / (2 * a₀) - 2 / b₀ = -9 / 2 :=
sorry

end NUMINAMATH_CALUDE_supremum_of_expression_l1244_124487


namespace NUMINAMATH_CALUDE_sqrt_nine_factorial_over_ninety_l1244_124492

theorem sqrt_nine_factorial_over_ninety : 
  Real.sqrt (Nat.factorial 9 / 90) = 24 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nine_factorial_over_ninety_l1244_124492


namespace NUMINAMATH_CALUDE_work_completion_time_l1244_124451

theorem work_completion_time (x : ℝ) : 
  (x > 0) →  -- Ensure x is positive
  (1/x + 1/15 = 1/6) →  -- Combined work rate equals 1/6
  (x = 10) := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l1244_124451


namespace NUMINAMATH_CALUDE_can_tile_4x7_with_4x1_l1244_124454

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a tetromino with width and height -/
structure Tetromino where
  width : ℕ
  height : ℕ

/-- Checks if a rectangle can be tiled with a given tetromino -/
def can_tile (r : Rectangle) (t : Tetromino) : Prop :=
  ∃ (n : ℕ), n * (t.width * t.height) = r.width * r.height

/-- The 4x7 rectangle -/
def rectangle_4x7 : Rectangle :=
  { width := 4, height := 7 }

/-- The 4x1 tetromino -/
def tetromino_4x1 : Tetromino :=
  { width := 4, height := 1 }

/-- Theorem stating that the 4x7 rectangle can be tiled with 4x1 tetrominos -/
theorem can_tile_4x7_with_4x1 : can_tile rectangle_4x7 tetromino_4x1 :=
  sorry

end NUMINAMATH_CALUDE_can_tile_4x7_with_4x1_l1244_124454


namespace NUMINAMATH_CALUDE_pure_imaginary_implies_a_eq_two_second_or_fourth_quadrant_implies_a_range_l1244_124427

-- Define the complex number z
def z (a : ℝ) : ℂ := (a^2 - a - 2 : ℝ) + (a^2 - 3*a - 4 : ℝ)*Complex.I

-- Part 1: z is a pure imaginary number implies a = 2
theorem pure_imaginary_implies_a_eq_two :
  ∀ a : ℝ, (z a).re = 0 → (z a).im ≠ 0 → a = 2 := by sorry

-- Part 2: z in second or fourth quadrant implies 2 < a < 4
theorem second_or_fourth_quadrant_implies_a_range :
  ∀ a : ℝ, (z a).re * (z a).im < 0 → 2 < a ∧ a < 4 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_implies_a_eq_two_second_or_fourth_quadrant_implies_a_range_l1244_124427


namespace NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l1244_124498

/-- Given a line equation in vector form, prove its slope-intercept form -/
theorem line_vector_to_slope_intercept 
  (x y : ℝ) : 
  (2 : ℝ) * (x - 4) + (-1 : ℝ) * (y + 5) = 0 ↔ y = 2 * x - 13 := by
  sorry

end NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l1244_124498


namespace NUMINAMATH_CALUDE_jerome_trail_time_l1244_124434

/-- The time it takes Jerome to run the trail -/
def jerome_time : ℝ := 6

/-- The time it takes Nero to run the trail -/
def nero_time : ℝ := 3

/-- Jerome's running speed in MPH -/
def jerome_speed : ℝ := 4

/-- Nero's running speed in MPH -/
def nero_speed : ℝ := 8

/-- Theorem stating that Jerome's time to run the trail is 6 hours -/
theorem jerome_trail_time : jerome_time = 6 := by sorry

end NUMINAMATH_CALUDE_jerome_trail_time_l1244_124434


namespace NUMINAMATH_CALUDE_weeks_to_save_l1244_124475

def console_cost : ℕ := 282
def game_cost : ℕ := 75
def initial_savings : ℕ := 42
def weekly_allowance : ℕ := 24

theorem weeks_to_save : ℕ := by
  -- The minimum number of whole weeks required to save enough money
  -- for both the console and the game is 14.
  sorry

end NUMINAMATH_CALUDE_weeks_to_save_l1244_124475


namespace NUMINAMATH_CALUDE_seven_digit_divisibility_l1244_124468

theorem seven_digit_divisibility (A B C : Nat) : 
  A < 10 → B < 10 → C < 10 →
  (74 * 100000 + A * 10000 + 52 * 100 + B * 10 + 1) % 3 = 0 →
  (326 * 10000 + A * 1000 + B * 100 + 4 * 10 + C) % 3 = 0 →
  C = 1 := by
sorry

end NUMINAMATH_CALUDE_seven_digit_divisibility_l1244_124468


namespace NUMINAMATH_CALUDE_income_182400_max_income_l1244_124461

/-- Represents the income function for the large grain grower --/
def income_function (original_land : ℝ) (original_income_per_mu : ℝ) (additional_land : ℝ) : ℝ :=
  original_land * original_income_per_mu + additional_land * (original_income_per_mu - 2 * additional_land)

/-- Theorem for the total income of 182,400 yuan --/
theorem income_182400 (original_land : ℝ) (original_income_per_mu : ℝ) :
  original_land = 360 ∧ original_income_per_mu = 440 →
  (∃ x : ℝ, (income_function original_land original_income_per_mu x = 182400 ∧ (x = 100 ∨ x = 120))) :=
sorry

/-- Theorem for the maximum total income --/
theorem max_income (original_land : ℝ) (original_income_per_mu : ℝ) :
  original_land = 360 ∧ original_income_per_mu = 440 →
  (∃ x : ℝ, (∀ y : ℝ, income_function original_land original_income_per_mu x ≥ income_function original_land original_income_per_mu y) ∧
             x = 110 ∧
             income_function original_land original_income_per_mu x = 182600) :=
sorry

end NUMINAMATH_CALUDE_income_182400_max_income_l1244_124461


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1244_124445

theorem triangle_angle_measure (A B C : ℝ) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (h_sine : (7 / Real.sin A) = (8 / Real.sin B) ∧ (8 / Real.sin B) = (13 / Real.sin C)) : 
  C = (2 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1244_124445


namespace NUMINAMATH_CALUDE_max_a_for_quadratic_inequality_l1244_124473

theorem max_a_for_quadratic_inequality :
  (∃ (a : ℝ), ∀ (x : ℝ), x^2 - 2*x - a ≥ 0) ∧
  (∀ (a : ℝ), (∀ (x : ℝ), x^2 - 2*x - a ≥ 0) → a ≤ -1) ∧
  (∀ (x : ℝ), x^2 - 2*x - (-1) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_max_a_for_quadratic_inequality_l1244_124473


namespace NUMINAMATH_CALUDE_ten_workers_needed_l1244_124477

/-- Represents the project details and worker productivity --/
structure Project where
  total_days : ℕ
  days_passed : ℕ
  work_completed : ℚ
  current_workers : ℕ

/-- Calculates the minimum number of workers needed to complete the project on schedule --/
def min_workers_needed (p : Project) : ℕ :=
  p.current_workers

/-- Theorem stating that for the given project conditions, 10 workers are needed --/
theorem ten_workers_needed (p : Project)
  (h1 : p.total_days = 40)
  (h2 : p.days_passed = 10)
  (h3 : p.work_completed = 1/4)
  (h4 : p.current_workers = 10) :
  min_workers_needed p = 10 := by
  sorry

#eval min_workers_needed {
  total_days := 40,
  days_passed := 10,
  work_completed := 1/4,
  current_workers := 10
}

end NUMINAMATH_CALUDE_ten_workers_needed_l1244_124477


namespace NUMINAMATH_CALUDE_product_increase_factor_l1244_124425

theorem product_increase_factor (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ¬(∀ a b : ℝ, (10 * a) * b = 10 * (a * b)) :=
sorry

end NUMINAMATH_CALUDE_product_increase_factor_l1244_124425


namespace NUMINAMATH_CALUDE_simplification_and_exponent_sum_l1244_124431

-- Define the original expression
def original_expression (x y z : ℝ) : ℝ := (40 * x^5 * y^3 * z^8) ^ (1/3)

-- Define the simplified expression
def simplified_expression (x y z : ℝ) : ℝ := 2 * x * y * z * (5 * x^2 * z^5) ^ (1/3)

-- Theorem statement
theorem simplification_and_exponent_sum :
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
  (original_expression x y z = simplified_expression x y z) ∧
  (1 + 1 + 1 = 3) := by sorry

end NUMINAMATH_CALUDE_simplification_and_exponent_sum_l1244_124431


namespace NUMINAMATH_CALUDE_initial_caterpillars_l1244_124411

theorem initial_caterpillars (initial : ℕ) (added : ℕ) (left : ℕ) (remaining : ℕ) : 
  added = 4 → left = 8 → remaining = 10 → initial + added - left = remaining → initial = 14 := by
  sorry

end NUMINAMATH_CALUDE_initial_caterpillars_l1244_124411


namespace NUMINAMATH_CALUDE_y1_greater_y2_l1244_124453

/-- A linear function passing through the first, second, and fourth quadrants -/
structure QuadrantCrossingLine where
  m : ℝ
  n : ℝ
  first_quadrant : ∃ x > 0, m * x + n > 0
  second_quadrant : ∃ x < 0, m * x + n > 0
  fourth_quadrant : ∃ x > 0, m * x + n < 0

/-- Theorem: For a linear function y = mx + n passing through the first, second, and fourth quadrants,
    if (1, y₁) and (3, y₂) are points on the graph, then y₁ > y₂ -/
theorem y1_greater_y2 (line : QuadrantCrossingLine) (y₁ y₂ : ℝ)
    (point1 : line.m * 1 + line.n = y₁)
    (point2 : line.m * 3 + line.n = y₂) :
    y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_greater_y2_l1244_124453


namespace NUMINAMATH_CALUDE_janous_inequality_l1244_124495

theorem janous_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a^2 + b^2 = 1/2) :
  1/(1-a) + 1/(1-b) ≥ 4 ∧ (1/(1-a) + 1/(1-b) = 4 ↔ a = 1/2 ∧ b = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_janous_inequality_l1244_124495


namespace NUMINAMATH_CALUDE_jennifer_discount_is_28_l1244_124449

/-- Calculates the discount for whole milk based on the number of cans purchased -/
def whole_milk_discount (cans : ℕ) : ℕ := (cans / 10) * 4

/-- Calculates the discount for almond milk based on the number of cans purchased -/
def almond_milk_discount (cans : ℕ) : ℕ := 
  ((cans / 7) * 3) + ((cans % 7) / 3)

/-- Represents Jennifer's milk purchase and calculates her total discount -/
def jennifer_discount : ℕ :=
  let initial_whole_milk := 40
  let mark_whole_milk := 30
  let mark_skim_milk := 15
  let additional_almond_milk := (mark_whole_milk / 3) * 2
  let additional_whole_milk := (mark_skim_milk / 5) * 4
  let total_whole_milk := initial_whole_milk + additional_whole_milk
  let total_almond_milk := additional_almond_milk
  whole_milk_discount total_whole_milk + almond_milk_discount total_almond_milk

theorem jennifer_discount_is_28 : jennifer_discount = 28 := by
  sorry

#eval jennifer_discount

end NUMINAMATH_CALUDE_jennifer_discount_is_28_l1244_124449


namespace NUMINAMATH_CALUDE_line_in_quadrants_implies_positive_slope_l1244_124422

/-- A line passing through the first and third quadrants -/
structure LineInQuadrants where
  k : ℝ
  k_nonzero : k ≠ 0
  passes_first_quadrant : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ y = k * x
  passes_third_quadrant : ∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ y = k * x

/-- If a line y = kx passes through the first and third quadrants, then k > 0 -/
theorem line_in_quadrants_implies_positive_slope (l : LineInQuadrants) : l.k > 0 := by
  sorry

end NUMINAMATH_CALUDE_line_in_quadrants_implies_positive_slope_l1244_124422


namespace NUMINAMATH_CALUDE_infinite_triples_sum_of_squares_l1244_124419

/-- A number that can be expressed as the sum of one or two squares. -/
def IsSumOfTwoSquares (k : ℤ) : Prop :=
  ∃ a b : ℤ, k = a^2 + b^2

theorem infinite_triples_sum_of_squares (n : ℤ) :
  let N := 2 * n^2 * (n + 1)^2
  IsSumOfTwoSquares N ∧
  IsSumOfTwoSquares (N + 1) ∧
  IsSumOfTwoSquares (N + 2) := by
  sorry


end NUMINAMATH_CALUDE_infinite_triples_sum_of_squares_l1244_124419


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_l1244_124407

-- Define variables
variable (a b m : ℝ)

-- Theorem for the first factorization
theorem factorization_1 : a^2 * (a - b) - 4 * b^2 * (a - b) = (a - b) * (a - 2*b) * (a + 2*b) := by
  sorry

-- Theorem for the second factorization
theorem factorization_2 : m^2 - 6*m + 9 = (m - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_l1244_124407


namespace NUMINAMATH_CALUDE_roots_of_quadratic_equation_l1244_124403

theorem roots_of_quadratic_equation :
  let f : ℂ → ℂ := λ x => x^2 + 4
  ∀ x : ℂ, f x = 0 ↔ x = 2*I ∨ x = -2*I :=
by sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_equation_l1244_124403


namespace NUMINAMATH_CALUDE_brian_final_cards_l1244_124458

def initial_cards : ℕ := 76
def cards_taken : ℕ := 59
def packs_bought : ℕ := 3
def cards_per_pack : ℕ := 15

theorem brian_final_cards : 
  initial_cards - cards_taken + packs_bought * cards_per_pack = 62 := by
  sorry

end NUMINAMATH_CALUDE_brian_final_cards_l1244_124458


namespace NUMINAMATH_CALUDE_sample_data_properties_l1244_124428

theorem sample_data_properties (x : Fin 6 → ℝ) (h : ∀ i j : Fin 6, i ≤ j → x i ≤ x j) :
  (let median1 := (x 2 + x 3) / 2
   let median2 := (x 2 + x 3) / 2
   median1 = median2) ∧
  (x 4 - x 1 ≤ x 5 - x 0) :=
by sorry

end NUMINAMATH_CALUDE_sample_data_properties_l1244_124428


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l1244_124469

theorem sum_of_two_numbers (x y : ℤ) : 
  y = 2 * x - 43 →  -- First number is 43 less than twice the second
  max x y = 31 →    -- Larger number is 31
  x + y = 68 :=     -- Sum of the two numbers is 68
by sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l1244_124469


namespace NUMINAMATH_CALUDE_point_P_coordinates_l1244_124429

def P₁ : ℝ × ℝ := (2, -1)
def P₂ : ℝ × ℝ := (0, 5)

def on_extension_line (P₁ P₂ P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 1 ∧ P = (t • P₂.1 + (1 - t) • P₁.1, t • P₂.2 + (1 - t) • P₁.2)

def distance_ratio (P₁ P₂ P : ℝ × ℝ) : Prop :=
  (P.1 - P₁.1)^2 + (P.2 - P₁.2)^2 = 4 * ((P₂.1 - P.1)^2 + (P₂.2 - P.2)^2)

theorem point_P_coordinates :
  ∀ P : ℝ × ℝ, on_extension_line P₁ P₂ P → distance_ratio P₁ P₂ P → P = (-2, 11) :=
by sorry

end NUMINAMATH_CALUDE_point_P_coordinates_l1244_124429


namespace NUMINAMATH_CALUDE_tourist_group_room_capacity_l1244_124482

/-- Given a tourist group and room arrangements, calculate the capacity of small rooms -/
theorem tourist_group_room_capacity
  (total_people : ℕ)
  (large_room_capacity : ℕ)
  (large_rooms_rented : ℕ)
  (h1 : total_people = 26)
  (h2 : large_room_capacity = 3)
  (h3 : large_rooms_rented = 8)
  : ∃ (small_room_capacity : ℕ),
    small_room_capacity > 0 ∧
    small_room_capacity * (total_people - large_room_capacity * large_rooms_rented) = total_people - large_room_capacity * large_rooms_rented ∧
    small_room_capacity = 2 :=
by sorry

end NUMINAMATH_CALUDE_tourist_group_room_capacity_l1244_124482


namespace NUMINAMATH_CALUDE_subset_properties_l1244_124463

variable {α : Type*}
variable (A B : Set α)

theorem subset_properties (hAB : A ⊆ B) (hA : A.Nonempty) (hB : B.Nonempty) :
  (∀ x, x ∈ A → x ∈ B) ∧
  (∃ x, x ∈ B ∧ x ∉ A) ∧
  (∀ x, x ∉ B → x ∉ A) :=
by sorry

end NUMINAMATH_CALUDE_subset_properties_l1244_124463


namespace NUMINAMATH_CALUDE_movie_theatre_revenue_l1244_124470

/-- Calculates the total ticket revenue for a movie theatre session -/
theorem movie_theatre_revenue 
  (total_seats : ℕ) 
  (adult_price child_price : ℕ) 
  (num_children : ℕ) 
  (h_full : num_children ≤ total_seats) : 
  let num_adults := total_seats - num_children
  (num_adults * adult_price + num_children * child_price : ℕ) = 1124 :=
by
  sorry

#check movie_theatre_revenue 250 6 4 188

end NUMINAMATH_CALUDE_movie_theatre_revenue_l1244_124470


namespace NUMINAMATH_CALUDE_magazines_to_boxes_l1244_124435

theorem magazines_to_boxes (total_magazines : ℕ) (magazines_per_box : ℕ) (h1 : total_magazines = 63) (h2 : magazines_per_box = 9) :
  total_magazines / magazines_per_box = 7 := by
  sorry

end NUMINAMATH_CALUDE_magazines_to_boxes_l1244_124435


namespace NUMINAMATH_CALUDE_word_permutation_ratio_l1244_124471

theorem word_permutation_ratio : 
  let n₁ : ℕ := 6  -- number of letters in "СКАЛКА"
  let n₂ : ℕ := 7  -- number of letters in "ТЕФТЕЛЬ"
  let r : ℕ := 2   -- number of repeated letters in each word
  
  -- number of distinct permutations for each word
  let perm₁ : ℕ := n₁! / (r! * r!)
  let perm₂ : ℕ := n₂! / (r! * r!)

  perm₂ / perm₁ = 7 := by
  sorry

end NUMINAMATH_CALUDE_word_permutation_ratio_l1244_124471


namespace NUMINAMATH_CALUDE_min_value_expression_l1244_124457

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 27) :
  x^2 + 6*x*y + 9*y^2 + (3/2)*z^2 ≥ 102 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 27 ∧
    x₀^2 + 6*x₀*y₀ + 9*y₀^2 + (3/2)*z₀^2 = 102 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1244_124457


namespace NUMINAMATH_CALUDE_ice_cream_scoops_l1244_124494

def ice_cream_problem (single_cone waffle_bowl banana_split double_cone : ℕ) : Prop :=
  single_cone = 1 ∧ 
  banana_split = 3 * single_cone ∧ 
  waffle_bowl = banana_split + 1 ∧
  double_cone = 2 ∧
  single_cone + double_cone + banana_split + waffle_bowl = 10

theorem ice_cream_scoops : 
  ∃ (single_cone waffle_bowl banana_split double_cone : ℕ),
    ice_cream_problem single_cone waffle_bowl banana_split double_cone :=
by
  sorry

end NUMINAMATH_CALUDE_ice_cream_scoops_l1244_124494


namespace NUMINAMATH_CALUDE_rectangle_area_difference_l1244_124414

/-- The difference in area between two rectangles, where one rectangle's dimensions are 1 cm less
    than the other's in both length and width, is equal to the sum of the larger rectangle's
    length and width, minus 1. -/
theorem rectangle_area_difference (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x * y - (x - 1) * (y - 1) = x + y - 1 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_difference_l1244_124414


namespace NUMINAMATH_CALUDE_power_of_four_equality_l1244_124488

theorem power_of_four_equality (m n : ℕ+) (x y : ℝ) 
  (hx : 2^(m : ℕ) = x) (hy : 2^(2*n : ℕ) = y) : 
  4^((m : ℕ) + 2*(n : ℕ)) = x^2 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_four_equality_l1244_124488


namespace NUMINAMATH_CALUDE_incircle_area_of_triangle_PF1F2_l1244_124452

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 24 = 1

-- Define the foci
def F1 : ℝ × ℝ := (-5, 0)
def F2 : ℝ × ℝ := (5, 0)

-- Define a point on the hyperbola in the first quadrant
def P : ℝ × ℝ := sorry

-- Distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem incircle_area_of_triangle_PF1F2 :
  hyperbola P.1 P.2 ∧
  P.1 > 0 ∧ P.2 > 0 ∧
  distance P F1 / distance P F2 = 4 / 3 →
  ∃ (r : ℝ), r^2 * π = 4 * π ∧
  r * (distance P F1 + distance P F2 + distance F1 F2) = distance P F1 * distance P F2 :=
by sorry

end NUMINAMATH_CALUDE_incircle_area_of_triangle_PF1F2_l1244_124452


namespace NUMINAMATH_CALUDE_polynomial_division_problem_l1244_124415

theorem polynomial_division_problem (x : ℝ) :
  let quotient := 2 * x + 6
  let divisor := x - 5
  let remainder := 2
  let polynomial := 2 * x^2 - 4 * x - 28
  polynomial = quotient * divisor + remainder := by sorry

end NUMINAMATH_CALUDE_polynomial_division_problem_l1244_124415


namespace NUMINAMATH_CALUDE_boat_speed_l1244_124447

/-- The speed of a boat in still water, given its downstream and upstream speeds -/
theorem boat_speed (downstream_speed upstream_speed : ℝ) 
  (h1 : downstream_speed = 10)
  (h2 : upstream_speed = 4) :
  (downstream_speed + upstream_speed) / 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_l1244_124447


namespace NUMINAMATH_CALUDE_chickens_bought_l1244_124443

def eggCount : ℕ := 20
def eggPrice : ℕ := 2
def chickenPrice : ℕ := 8
def totalSpent : ℕ := 88

theorem chickens_bought :
  (totalSpent - eggCount * eggPrice) / chickenPrice = 6 := by sorry

end NUMINAMATH_CALUDE_chickens_bought_l1244_124443


namespace NUMINAMATH_CALUDE_customers_who_left_l1244_124444

-- Define the initial number of customers
def initial_customers : ℕ := 13

-- Define the number of new customers
def new_customers : ℕ := 4

-- Define the final number of customers
def final_customers : ℕ := 9

-- Theorem to prove the number of customers who left
theorem customers_who_left :
  ∃ (left : ℕ), initial_customers - left + new_customers = final_customers ∧ left = 8 :=
by sorry

end NUMINAMATH_CALUDE_customers_who_left_l1244_124444


namespace NUMINAMATH_CALUDE_max_volume_is_three_l1244_124423

/-- Represents a rectangular solid with given constraints -/
structure RectangularSolid where
  width : ℝ
  length : ℝ
  height : ℝ
  sum_of_edges : width * 4 + length * 4 + height * 4 = 18
  length_width_ratio : length = 2 * width

/-- The volume of a rectangular solid -/
def volume (r : RectangularSolid) : ℝ := r.width * r.length * r.height

/-- Theorem stating that the maximum volume of the rectangular solid is 3 -/
theorem max_volume_is_three :
  ∃ (r : RectangularSolid), volume r = 3 ∧ ∀ (s : RectangularSolid), volume s ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_max_volume_is_three_l1244_124423


namespace NUMINAMATH_CALUDE_cody_tickets_l1244_124433

/-- The number of tickets Cody spent on a beanie -/
def beanie_cost : ℕ := 25

/-- The number of additional tickets Cody won later -/
def additional_tickets : ℕ := 6

/-- The number of tickets Cody has now -/
def current_tickets : ℕ := 30

/-- The initial number of tickets Cody won -/
def initial_tickets : ℕ := 49

theorem cody_tickets : 
  initial_tickets = beanie_cost + (current_tickets - additional_tickets) :=
by sorry

end NUMINAMATH_CALUDE_cody_tickets_l1244_124433


namespace NUMINAMATH_CALUDE_toms_game_sale_l1244_124436

/-- Calculates the sale amount of games given initial cost, value increase factor, and sale percentage -/
def gameSaleAmount (initialCost : ℝ) (valueIncreaseFactor : ℝ) (salePercentage : ℝ) : ℝ :=
  initialCost * valueIncreaseFactor * salePercentage

/-- Proves that Tom's game sale amount is $240 given the specified conditions -/
theorem toms_game_sale : gameSaleAmount 200 3 0.4 = 240 := by
  sorry

end NUMINAMATH_CALUDE_toms_game_sale_l1244_124436


namespace NUMINAMATH_CALUDE_two_digit_number_sum_l1244_124400

theorem two_digit_number_sum (a b : ℕ) : 
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
  (10 * a + b) - (10 * b + a) = 5 * (a + b) →
  (10 * a + b) + (10 * b + a) = 99 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_sum_l1244_124400


namespace NUMINAMATH_CALUDE_apple_count_difference_l1244_124438

/-- The number of green apples initially in the store -/
def initial_green_apples : ℕ := 32

/-- The number of additional red apples compared to green apples initially -/
def red_apple_surplus : ℕ := 200

/-- The number of green apples delivered by the truck -/
def delivered_green_apples : ℕ := 340

/-- The final difference between green and red apples -/
def final_green_red_difference : ℤ := 140

theorem apple_count_difference :
  (initial_green_apples + delivered_green_apples : ℤ) - 
  (initial_green_apples + red_apple_surplus) = 
  final_green_red_difference :=
by sorry

end NUMINAMATH_CALUDE_apple_count_difference_l1244_124438


namespace NUMINAMATH_CALUDE_floor_equation_solution_l1244_124460

theorem floor_equation_solution (a b : ℝ) : 
  (∀ n : ℕ+, a * ⌊b * n⌋ = b * ⌊a * n⌋) ↔ 
  (a = 0 ∨ b = 0 ∨ (a = b ∧ ∃ m : ℤ, a = m)) := by
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l1244_124460


namespace NUMINAMATH_CALUDE_roots_relation_l1244_124401

-- Define the polynomial h(x)
def h (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x - 4

-- Define the polynomial j(x)
def j (b c d x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

-- Theorem statement
theorem roots_relation (b c d : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    h r₁ = 0 ∧ h r₂ = 0 ∧ h r₃ = 0) →
  (∀ x : ℝ, h x = 0 → ∃ y : ℝ, j b c d y = 0 ∧ y = x^3) →
  b = -8 ∧ c = 36 ∧ d = -64 := by
sorry

end NUMINAMATH_CALUDE_roots_relation_l1244_124401


namespace NUMINAMATH_CALUDE_inequality_proof_l1244_124416

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 / (b^3 * c)) - (a / b^2) ≥ (c / b) - (c^2 / a) ∧
  ((a^2 / (b^3 * c)) - (a / b^2) = (c / b) - (c^2 / a) ↔ a = b * c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1244_124416


namespace NUMINAMATH_CALUDE_expression_evaluation_l1244_124497

theorem expression_evaluation : (2^2 - 2) - (3^2 - 3) + (4^2 - 4) - (5^2 - 5) + (6^2 - 6) = 18 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1244_124497


namespace NUMINAMATH_CALUDE_van_rental_equation_l1244_124478

theorem van_rental_equation (x : ℕ+) :
  (180 : ℝ) / x - 180 / (x + 2) = 3 :=
by sorry

end NUMINAMATH_CALUDE_van_rental_equation_l1244_124478
