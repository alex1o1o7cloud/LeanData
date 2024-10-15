import Mathlib

namespace NUMINAMATH_CALUDE_flag_designs_count_l1184_118467

/-- The number of available colors for the flag stripes -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag -/
def num_stripes : ℕ := 3

/-- The total number of possible flag designs -/
def total_flag_designs : ℕ := num_colors ^ num_stripes

/-- Theorem stating that the total number of possible flag designs is 27 -/
theorem flag_designs_count : total_flag_designs = 27 := by
  sorry

end NUMINAMATH_CALUDE_flag_designs_count_l1184_118467


namespace NUMINAMATH_CALUDE_compound_interest_problem_l1184_118416

theorem compound_interest_problem (P r : ℝ) : 
  P > 0 → r > 0 →
  P * (1 + r)^2 = 7000 →
  P * (1 + r)^3 = 9261 →
  P = 4000 := by
sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l1184_118416


namespace NUMINAMATH_CALUDE_segment_ratio_l1184_118425

/-- Given two line segments with equally spaced points, prove the ratio of their lengths -/
theorem segment_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ x : ℝ, x > 0 ∧ a = 9*x ∧ b = 99*x) → b / a = 11 := by
  sorry

end NUMINAMATH_CALUDE_segment_ratio_l1184_118425


namespace NUMINAMATH_CALUDE_smallest_possible_N_l1184_118421

def is_valid_arrangement (a b c d e f : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧
  a + b + c + d + e + f = 2520 ∧
  a ≥ 5 ∧ b ≥ 5 ∧ c ≥ 5 ∧ d ≥ 5 ∧ e ≥ 5 ∧ f ≥ 5

def N (a b c d e f : ℕ) : ℕ :=
  max (a + b) (max (b + c) (max (c + d) (max (d + e) (e + f))))

theorem smallest_possible_N :
  ∀ a b c d e f : ℕ, is_valid_arrangement a b c d e f →
  N a b c d e f ≥ 506 ∧
  (∃ a' b' c' d' e' f' : ℕ, is_valid_arrangement a' b' c' d' e' f' ∧ N a' b' c' d' e' f' = 506) :=
sorry

end NUMINAMATH_CALUDE_smallest_possible_N_l1184_118421


namespace NUMINAMATH_CALUDE_problem_solution_l1184_118474

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x < -4 ∨ x > -2}
def C (m : ℝ) : Set ℝ := {x | 3 - 2*m ≤ x ∧ x ≤ 2 + m}
def D : Set ℝ := {y | y < -6 ∨ y > -5}

theorem problem_solution (m : ℝ) :
  (∀ x, x ∈ A ∩ B → x ∈ C m) →
  (B ∪ C m = Set.univ ∧ C m ⊆ D) →
  m ≥ 5/2 ∧ 7/2 ≤ m ∧ m < 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1184_118474


namespace NUMINAMATH_CALUDE_sum_of_roots_eq_one_l1184_118492

theorem sum_of_roots_eq_one : 
  ∃ (x₁ x₂ : ℝ), (x₁ + 3) * (x₁ - 4) = 22 ∧ 
                 (x₂ + 3) * (x₂ - 4) = 22 ∧ 
                 x₁ + x₂ = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_eq_one_l1184_118492


namespace NUMINAMATH_CALUDE_range_of_a_solution_set_l1184_118408

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

-- Theorem for part I
theorem range_of_a (a : ℝ) :
  (∃ x, f x < 2 * a - 1) ↔ a > 2 :=
sorry

-- Theorem for part II
theorem solution_set :
  {x : ℝ | f x ≥ x^2 - 2*x} = {x : ℝ | -1 ≤ x ∧ x ≤ 2 + Real.sqrt 3} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_solution_set_l1184_118408


namespace NUMINAMATH_CALUDE_salary_changes_l1184_118439

/-- Represents the series of salary changes and calculates the final salary --/
def final_salary (original : ℝ) : ℝ :=
  let after_first_raise := original * 1.12
  let after_reduction := after_first_raise * 0.93
  let after_bonus := after_reduction * 1.15
  let fixed_component := after_bonus * 0.7
  let variable_component := after_bonus * 0.3 * 0.9
  fixed_component + variable_component

/-- Theorem stating that an original salary of approximately 7041.77 results in a final salary of 7600.35 --/
theorem salary_changes (ε : ℝ) (hε : ε > 0) :
  ∃ (original : ℝ), abs (original - 7041.77) < ε ∧ final_salary original = 7600.35 := by
  sorry

end NUMINAMATH_CALUDE_salary_changes_l1184_118439


namespace NUMINAMATH_CALUDE_max_leftover_candies_l1184_118456

theorem max_leftover_candies (n : ℕ) : ∃ (k : ℕ), n = 8 * k + (n % 8) ∧ n % 8 ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_max_leftover_candies_l1184_118456


namespace NUMINAMATH_CALUDE_gcd_1755_1242_l1184_118449

theorem gcd_1755_1242 : Nat.gcd 1755 1242 = 27 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1755_1242_l1184_118449


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_l1184_118438

def binomial_coefficient (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem largest_two_digit_prime_factor :
  ∃ (p : ℕ), Prime p ∧ 10 ≤ p ∧ p < 100 ∧ 
  p ∣ binomial_coefficient 210 105 ∧
  ∀ (q : ℕ), Prime q → 10 ≤ q → q < 100 → q ∣ binomial_coefficient 210 105 → q ≤ p ∧
  p = 67 :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_l1184_118438


namespace NUMINAMATH_CALUDE_cos_15_cos_30_minus_sin_15_sin_150_l1184_118413

theorem cos_15_cos_30_minus_sin_15_sin_150 : 
  Real.cos (15 * π / 180) * Real.cos (30 * π / 180) - 
  Real.sin (15 * π / 180) * Real.sin (150 * π / 180) = 
  Real.sqrt 2 / 2 :=
by
  -- Assuming sin 150° = sin 30°
  have h1 : Real.sin (150 * π / 180) = Real.sin (30 * π / 180) := by sorry
  sorry

end NUMINAMATH_CALUDE_cos_15_cos_30_minus_sin_15_sin_150_l1184_118413


namespace NUMINAMATH_CALUDE_shopkeeper_cards_l1184_118485

/-- The number of cards in a complete deck of standard playing cards -/
def standard_deck : ℕ := 52

/-- The number of cards in a complete deck of Uno cards -/
def uno_deck : ℕ := 108

/-- The number of cards in a complete deck of tarot cards -/
def tarot_deck : ℕ := 78

/-- The number of complete decks of standard playing cards -/
def standard_decks : ℕ := 4

/-- The number of complete decks of Uno cards -/
def uno_decks : ℕ := 3

/-- The number of complete decks of tarot cards -/
def tarot_decks : ℕ := 5

/-- The number of additional standard playing cards -/
def extra_standard : ℕ := 12

/-- The number of additional Uno cards -/
def extra_uno : ℕ := 7

/-- The number of additional tarot cards -/
def extra_tarot : ℕ := 9

/-- The total number of cards the shopkeeper has -/
def total_cards : ℕ := 
  standard_decks * standard_deck + extra_standard +
  uno_decks * uno_deck + extra_uno +
  tarot_decks * tarot_deck + extra_tarot

theorem shopkeeper_cards : total_cards = 950 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_cards_l1184_118485


namespace NUMINAMATH_CALUDE_crayons_remaining_l1184_118415

theorem crayons_remaining (initial : ℕ) (taken : ℕ) (remaining : ℕ) : 
  initial = 7 → taken = 3 → remaining = initial - taken → remaining = 4 := by
sorry

end NUMINAMATH_CALUDE_crayons_remaining_l1184_118415


namespace NUMINAMATH_CALUDE_license_plate_count_l1184_118476

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The number of digits in a license plate -/
def digits_in_plate : ℕ := 4

/-- The number of letters in a license plate -/
def letters_in_plate : ℕ := 3

/-- The number of possible positions for the letter block -/
def letter_block_positions : ℕ := digits_in_plate + 1

/-- The total number of distinct license plates -/
def total_license_plates : ℕ := 
  letter_block_positions * num_digits^digits_in_plate * num_letters^letters_in_plate

theorem license_plate_count : total_license_plates = 878800000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l1184_118476


namespace NUMINAMATH_CALUDE_prime_power_equation_l1184_118418

theorem prime_power_equation (p : ℕ) (x y : ℕ) (h_prime : Nat.Prime p) (h_eq : x^4 - y^4 = p * (x^3 - y^3)) :
  (x = 0 ∧ y = p) ∨ (x = p ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_prime_power_equation_l1184_118418


namespace NUMINAMATH_CALUDE_green_peaches_count_l1184_118420

/-- The number of green peaches in a basket -/
def num_green_peaches : ℕ := sorry

/-- The number of red peaches in the basket -/
def num_red_peaches : ℕ := 6

/-- The total number of red and green peaches in the basket -/
def total_red_green_peaches : ℕ := 22

/-- Theorem stating that the number of green peaches is 16 -/
theorem green_peaches_count : num_green_peaches = 16 := by sorry

end NUMINAMATH_CALUDE_green_peaches_count_l1184_118420


namespace NUMINAMATH_CALUDE_mlb_game_ratio_l1184_118422

theorem mlb_game_ratio (misses : ℕ) (total : ℕ) : 
  misses = 50 → total = 200 → (misses : ℚ) / (total - misses : ℚ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_mlb_game_ratio_l1184_118422


namespace NUMINAMATH_CALUDE_intersection_A_B_range_of_m_l1184_118469

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 5*x - 6 < 0}
def B : Set ℝ := {x | 6*x^2 - 5*x + 1 ≥ 0}
def C (m : ℝ) : Set ℝ := {x | (x - m) / (x - m - 9) < 0}

-- Theorem for the intersection of A and B
theorem intersection_A_B :
  A ∩ B = {x : ℝ | (-1 < x ∧ x ≤ 1/3) ∨ (1/2 ≤ x ∧ x < 6)} := by sorry

-- Theorem for the range of m when A ∪ C = C
theorem range_of_m (m : ℝ) :
  (A ∪ C m = C m) → (-3 ≤ m ∧ m ≤ -1) := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_range_of_m_l1184_118469


namespace NUMINAMATH_CALUDE_snack_cost_per_person_l1184_118445

/-- Calculates the cost per person for a group of friends buying snacks -/
theorem snack_cost_per_person 
  (num_friends : ℕ) 
  (num_fish_cakes : ℕ) 
  (fish_cake_price : ℕ) 
  (num_tteokbokki : ℕ) 
  (tteokbokki_price : ℕ) 
  (h1 : num_friends = 4) 
  (h2 : num_fish_cakes = 5) 
  (h3 : fish_cake_price = 200) 
  (h4 : num_tteokbokki = 7) 
  (h5 : tteokbokki_price = 800) : 
  (num_fish_cakes * fish_cake_price + num_tteokbokki * tteokbokki_price) / num_friends = 1650 := by
  sorry

end NUMINAMATH_CALUDE_snack_cost_per_person_l1184_118445


namespace NUMINAMATH_CALUDE_solve_equation_l1184_118443

theorem solve_equation (x : ℝ) (h : 0.009 / x = 0.01) : x = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1184_118443


namespace NUMINAMATH_CALUDE_min_value_fraction_l1184_118410

theorem min_value_fraction (x : ℝ) (h : x > 2) : (x^2 - 4*x + 5) / (x - 2) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l1184_118410


namespace NUMINAMATH_CALUDE_cost_mms_in_snickers_l1184_118409

/-- The cost of a pack of M&M's in terms of Snickers pieces -/
theorem cost_mms_in_snickers 
  (snickers_quantity : ℕ)
  (mms_quantity : ℕ)
  (snickers_price : ℚ)
  (total_paid : ℚ)
  (change_received : ℚ)
  (h1 : snickers_quantity = 2)
  (h2 : mms_quantity = 3)
  (h3 : snickers_price = 3/2)
  (h4 : total_paid = 20)
  (h5 : change_received = 8) :
  (total_paid - change_received - snickers_quantity * snickers_price) / mms_quantity = 2 * snickers_price :=
by sorry

end NUMINAMATH_CALUDE_cost_mms_in_snickers_l1184_118409


namespace NUMINAMATH_CALUDE_triangle_inradius_l1184_118495

/-- The inradius of a triangle with side lengths 13, 84, and 85 is 6 -/
theorem triangle_inradius : ∀ (a b c r : ℝ),
  a = 13 ∧ b = 84 ∧ c = 85 →
  a^2 + b^2 = c^2 →
  (a + b + c) / 2 * r = (a * b) / 2 →
  r = 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inradius_l1184_118495


namespace NUMINAMATH_CALUDE_max_distance_MP_l1184_118412

-- Define the equilateral triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = 8 ∧ dist B C = 8 ∧ dist C A = 8

-- Define the point O satisfying the given condition
def PointO (A B C O : ℝ × ℝ) : Prop :=
  (O.1 - A.1, O.2 - A.2) = 2 • (O.1 - B.1, O.2 - B.2) + 3 • (O.1 - C.1, O.2 - C.2)

-- Define a point M on the sides of triangle ABC
def PointM (A B C M : ℝ × ℝ) : Prop :=
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)) ∨
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (t * B.1 + (1 - t) * C.1, t * B.2 + (1 - t) * C.2)) ∨
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (t * C.1 + (1 - t) * A.1, t * C.2 + (1 - t) * A.2))

-- Define a point P such that |OP| = √19
def PointP (O P : ℝ × ℝ) : Prop :=
  dist O P = Real.sqrt 19

theorem max_distance_MP (A B C O M P : ℝ × ℝ) :
  Triangle A B C →
  PointO A B C O →
  PointM A B C M →
  PointP O P →
  (∀ M' P', PointM A B C M' → PointP O P' → dist M P ≤ dist M' P') →
  dist M P = 3 * Real.sqrt 19 :=
sorry

end NUMINAMATH_CALUDE_max_distance_MP_l1184_118412


namespace NUMINAMATH_CALUDE_f_8_5_equals_1_5_l1184_118450

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_8_5_equals_1_5 (f : ℝ → ℝ) 
  (h1 : is_even f)
  (h2 : has_period f 3)
  (h3 : ∀ x ∈ Set.Icc 0 1, f x = 3 * x) :
  f 8.5 = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_f_8_5_equals_1_5_l1184_118450


namespace NUMINAMATH_CALUDE_complement_of_union_A_B_l1184_118464

def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {x ∈ U | x^2 - 5*x + 4 < 0}

theorem complement_of_union_A_B : 
  (A ∪ B)ᶜ = {0, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_A_B_l1184_118464


namespace NUMINAMATH_CALUDE_bisection_method_sign_l1184_118489

open Set

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the interval (a, b)
variable (a b : ℝ)

-- Define the sequence of intervals
variable (seq : ℕ → ℝ × ℝ)

-- State the theorem
theorem bisection_method_sign (hcont : Continuous f) 
  (hunique : ∃! x, x ∈ Ioo a b ∧ f x = 0)
  (hseq : ∀ k, Ioo (seq k).1 (seq k).2 ⊆ Ioo (seq (k+1)).1 (seq (k+1)).2)
  (hzero : ∀ k, ∃ x, x ∈ Ioo (seq k).1 (seq k).2 ∧ f x = 0)
  (hinit : seq 0 = (a, b))
  (hsign : f a < 0 ∧ f b > 0) :
  ∀ k, f (seq k).1 < 0 :=
sorry

end NUMINAMATH_CALUDE_bisection_method_sign_l1184_118489


namespace NUMINAMATH_CALUDE_green_tea_cost_july_l1184_118448

/-- Proves that the cost of green tea per pound in July is $0.30 --/
theorem green_tea_cost_july (june_cost : ℝ) : 
  (june_cost > 0) →  -- Assuming positive cost in June
  (june_cost + june_cost = 3.45 / 1.5) →  -- July mixture cost equation
  (0.3 * june_cost = 0.30) :=  -- July green tea cost
by
  sorry

#check green_tea_cost_july

end NUMINAMATH_CALUDE_green_tea_cost_july_l1184_118448


namespace NUMINAMATH_CALUDE_egypt_promotion_free_tourists_l1184_118402

/-- Represents the number of tourists who went to Egypt for free -/
def free_tourists : ℕ := 29

/-- Represents the number of tourists who came on their own -/
def self_tourists : ℕ := 13

/-- Represents the number of tourists who brought no one -/
def no_referral_tourists : ℕ := 100

theorem egypt_promotion_free_tourists :
  ∃ (total_tourists : ℕ),
    total_tourists = self_tourists + 4 * free_tourists ∧
    total_tourists = free_tourists + no_referral_tourists ∧
    free_tourists * 4 + self_tourists = free_tourists + no_referral_tourists :=
by sorry

end NUMINAMATH_CALUDE_egypt_promotion_free_tourists_l1184_118402


namespace NUMINAMATH_CALUDE_email_difference_l1184_118405

def morning_emails : ℕ := 10
def afternoon_emails : ℕ := 7
def evening_emails : ℕ := 17

theorem email_difference : morning_emails - afternoon_emails = 3 := by
  sorry

end NUMINAMATH_CALUDE_email_difference_l1184_118405


namespace NUMINAMATH_CALUDE_count_integers_with_repeated_digits_eq_168_l1184_118424

/-- The number of positive three-digit integers less than 700 with at least two identical digits -/
def count_integers_with_repeated_digits : ℕ :=
  let total_three_digit_integers := 700 - 100
  let integers_without_repeated_digits := 6 * 9 * 8
  total_three_digit_integers - integers_without_repeated_digits

theorem count_integers_with_repeated_digits_eq_168 :
  count_integers_with_repeated_digits = 168 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_with_repeated_digits_eq_168_l1184_118424


namespace NUMINAMATH_CALUDE_min_product_under_constraints_l1184_118432

theorem min_product_under_constraints (x y z : ℝ) :
  x > 0 → y > 0 → z > 0 →
  x + y + z = 2 →
  x ≤ 3*y ∧ x ≤ 3*z ∧ y ≤ 3*x ∧ y ≤ 3*z ∧ z ≤ 3*x ∧ z ≤ 3*y →
  x * y * z ≥ 32/81 :=
by sorry

end NUMINAMATH_CALUDE_min_product_under_constraints_l1184_118432


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l1184_118427

theorem diophantine_equation_solutions :
  let S : Set (ℕ × ℕ × ℕ × ℕ) := {(x, y, z, w) | 2^x * 3^y - 5^z * 7^w = 1}
  S = {(1, 0, 0, 0), (3, 0, 0, 1), (1, 1, 1, 0), (2, 2, 1, 1)} := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l1184_118427


namespace NUMINAMATH_CALUDE_function_composition_distribution_l1184_118462

-- Define real-valued functions on ℝ
variable (f g h : ℝ → ℝ)

-- Define function composition
def comp (f g : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f (g x)

-- Define pointwise multiplication of functions
def mult (f g : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f x * g x

-- Statement of the theorem
theorem function_composition_distribution :
  ∀ x : ℝ, (comp (mult f g) h) x = (mult (comp f h) (comp g h)) x :=
by sorry

end NUMINAMATH_CALUDE_function_composition_distribution_l1184_118462


namespace NUMINAMATH_CALUDE_box_negative_two_zero_negative_one_l1184_118463

-- Define the box operation
def box (a b c : ℤ) : ℚ :=
  (a ^ b : ℚ) - if b = 0 ∧ c < 0 then 0 else (b ^ c : ℚ) + (c ^ a : ℚ)

-- State the theorem
theorem box_negative_two_zero_negative_one :
  box (-2) 0 (-1) = 2 := by sorry

end NUMINAMATH_CALUDE_box_negative_two_zero_negative_one_l1184_118463


namespace NUMINAMATH_CALUDE_fraction_equality_l1184_118430

theorem fraction_equality (x : ℝ) (k : ℝ) (h1 : x > 0) (h2 : x = 1/3) 
  (h3 : (2/3) * x = k * (1/x)) : k = 2/27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1184_118430


namespace NUMINAMATH_CALUDE_cloth_cost_theorem_l1184_118403

/-- The total cost of cloth given the length and price per meter -/
def total_cost (length : ℝ) (price_per_meter : ℝ) : ℝ :=
  length * price_per_meter

/-- Theorem: The total cost of 9.25 meters of cloth at $44 per meter is $407 -/
theorem cloth_cost_theorem :
  total_cost 9.25 44 = 407 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_theorem_l1184_118403


namespace NUMINAMATH_CALUDE_expression_equals_negative_one_l1184_118466

theorem expression_equals_negative_one (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ x ∧ z ≠ -x) :
  (x / (x + z) + z / (x - z)) / (z / (x + z) - x / (x - z)) = -1 :=
sorry

end NUMINAMATH_CALUDE_expression_equals_negative_one_l1184_118466


namespace NUMINAMATH_CALUDE_rose_bush_price_is_75_l1184_118400

-- Define the given conditions
def total_rose_bushes : ℕ := 6
def friend_rose_bushes : ℕ := 2
def aloe_count : ℕ := 2
def aloe_price : ℕ := 100
def total_spent_self : ℕ := 500

-- Define the function to calculate the price of each rose bush
def rose_bush_price : ℕ :=
  let self_rose_bushes := total_rose_bushes - friend_rose_bushes
  let aloe_total := aloe_count * aloe_price
  let rose_bushes_total := total_spent_self - aloe_total
  rose_bushes_total / self_rose_bushes

-- Theorem statement
theorem rose_bush_price_is_75 : rose_bush_price = 75 := by
  sorry

end NUMINAMATH_CALUDE_rose_bush_price_is_75_l1184_118400


namespace NUMINAMATH_CALUDE_evaluate_expression_l1184_118404

/-- Given x, y, and z are variables, prove that (25x³y) · (4xy²z) · (1/(5xyz)²) = 4x²y/z -/
theorem evaluate_expression (x y z : ℝ) (h : z ≠ 0) :
  (25 * x^3 * y) * (4 * x * y^2 * z) * (1 / (5 * x * y * z)^2) = 4 * x^2 * y / z := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1184_118404


namespace NUMINAMATH_CALUDE_inequality_proof_l1184_118494

theorem inequality_proof (m n : ℕ+) : 
  |n * Real.sqrt (n^2 + 1) - m| ≥ Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1184_118494


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1184_118487

theorem sum_of_fractions : 
  7/8 + 11/12 = 43/24 ∧ 
  (∀ n d : ℤ, (n ≠ 0 ∨ d ≠ 1) → 43 * d ≠ 24 * n) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1184_118487


namespace NUMINAMATH_CALUDE_triangle_properties_l1184_118499

open Real

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

theorem triangle_properties (t : Triangle) :
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = π →
  (t.a^2 + t.c^2 = t.b^2 + t.a * t.c →
    t.B = π / 3) ∧
  (t.a^2 + t.c^2 = t.b^2 + t.a * t.c ∧
   t.A = 5 * π / 12 ∧ t.b = 2 →
    t.c = 2 * Real.sqrt 6 / 3) ∧
  (t.a^2 + t.c^2 = t.b^2 + t.a * t.c ∧
   t.a + t.c = 4 →
    ∀ x : ℝ, x > 0 → t.b ≤ x → 2 ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1184_118499


namespace NUMINAMATH_CALUDE_f_extremum_f_two_zeros_harmonic_sum_bound_l1184_118401

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 - a / x + a * Real.log (1 / x)

theorem f_extremum :
  let f₁ := f 1
  (∃ x₀ > 0, ∀ x > 0, f₁ x ≤ f₁ x₀) ∧
  f₁ 1 = 0 ∧
  (¬∃ x₀ > 0, ∀ x > 0, f₁ x ≥ f₁ x₀) := by sorry

theorem f_two_zeros (a : ℝ) :
  (∃ x y, 1 / Real.exp 1 < x ∧ x < y ∧ y < Real.exp 1 ∧ f a x = 0 ∧ f a y = 0) ↔
  (Real.exp 1 / (Real.exp 1 + 1) < a ∧ a < 1) := by sorry

theorem harmonic_sum_bound (n : ℕ) (hn : n ≥ 3) :
  Real.log ((n + 1) / 3) < (Finset.range (n - 2)).sum (λ i => 1 / (i + 3 : ℝ)) := by sorry

end NUMINAMATH_CALUDE_f_extremum_f_two_zeros_harmonic_sum_bound_l1184_118401


namespace NUMINAMATH_CALUDE_f_difference_bound_l1184_118483

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x + 15

-- State the theorem
theorem f_difference_bound (x a : ℝ) (h : |x - a| < 1) : 
  |f x - f a| < 2 * (|a| + 1) := by
  sorry

end NUMINAMATH_CALUDE_f_difference_bound_l1184_118483


namespace NUMINAMATH_CALUDE_smaller_solution_quadratic_l1184_118461

theorem smaller_solution_quadratic (x : ℝ) : 
  (x^2 - 7*x - 18 = 0) → (x = -2 ∨ x = 9) → -2 ≤ x :=
by sorry

end NUMINAMATH_CALUDE_smaller_solution_quadratic_l1184_118461


namespace NUMINAMATH_CALUDE_parabola_translation_l1184_118426

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c - v }

theorem parabola_translation (x y : ℝ) :
  let original := Parabola.mk (-4) 0 0
  let translated := translate original 2 3
  y = -4 * x^2 → y = translated.a * (x + 2)^2 + translated.b * (x + 2) + translated.c :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l1184_118426


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l1184_118459

theorem simplify_complex_fraction :
  (1 / ((1 / (Real.sqrt 5 + 2)) + (2 / (Real.sqrt 7 - 2)))) =
  ((6 * Real.sqrt 7 + 9 * Real.sqrt 5 + 6) / (118 + 12 * Real.sqrt 35)) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l1184_118459


namespace NUMINAMATH_CALUDE_cara_don_meeting_l1184_118486

/-- Cara and Don walk towards each other's houses. -/
theorem cara_don_meeting 
  (distance_between_homes : ℝ) 
  (cara_speed : ℝ) 
  (don_speed : ℝ) 
  (don_start_delay : ℝ) 
  (h1 : distance_between_homes = 45) 
  (h2 : cara_speed = 6) 
  (h3 : don_speed = 5) 
  (h4 : don_start_delay = 2) : 
  ∃ x : ℝ, x = 30 ∧ 
  x + don_speed * (x / cara_speed - don_start_delay) = distance_between_homes :=
by
  sorry


end NUMINAMATH_CALUDE_cara_don_meeting_l1184_118486


namespace NUMINAMATH_CALUDE_cube_root_plus_sqrt_minus_sqrt_l1184_118437

theorem cube_root_plus_sqrt_minus_sqrt : ∃ x y z : ℝ, x^3 = -64 ∧ y^2 = 9 ∧ z^2 = 25/16 ∧ x + y - z = -9/4 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_plus_sqrt_minus_sqrt_l1184_118437


namespace NUMINAMATH_CALUDE_quartic_equation_roots_l1184_118414

theorem quartic_equation_roots (p : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ 
    x₁^4 + p*x₁^3 + 3*x₁^2 + p*x₁ + 4 = 0 ∧
    x₂^4 + p*x₂^3 + 3*x₂^2 + p*x₂ + 4 = 0) ↔ 
  (p ≤ -2 ∨ p ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_quartic_equation_roots_l1184_118414


namespace NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l1184_118478

/-- The ratio of the volume of a sphere with radius p to the volume of a hemisphere with radius 3p is 1/13.5 -/
theorem sphere_hemisphere_volume_ratio (p : ℝ) (hp : p > 0) :
  (4 / 3 * Real.pi * p ^ 3) / (1 / 2 * 4 / 3 * Real.pi * (3 * p) ^ 3) = 1 / 13.5 :=
by sorry

end NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l1184_118478


namespace NUMINAMATH_CALUDE_not_perfect_square_l1184_118488

theorem not_perfect_square (n : ℕ) : ¬∃ (m : ℕ), m^2 = 4*n + 2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l1184_118488


namespace NUMINAMATH_CALUDE_sqrt_log_sum_equality_l1184_118447

theorem sqrt_log_sum_equality : 
  Real.sqrt (Real.log 6 / Real.log 2 + Real.log 6 / Real.log 3) = 
    Real.sqrt (Real.log 3 / Real.log 2) + Real.sqrt (Real.log 2 / Real.log 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_log_sum_equality_l1184_118447


namespace NUMINAMATH_CALUDE_lions_volleyball_games_l1184_118498

theorem lions_volleyball_games 
  (initial_win_rate : Real) 
  (initial_win_rate_value : initial_win_rate = 0.60)
  (final_win_rate : Real) 
  (final_win_rate_value : final_win_rate = 0.55)
  (tournament_wins : Nat) 
  (tournament_wins_value : tournament_wins = 8)
  (tournament_losses : Nat) 
  (tournament_losses_value : tournament_losses = 4) :
  ∃ (total_games : Nat), 
    total_games = 40 ∧ 
    (initial_win_rate * (total_games - tournament_wins - tournament_losses) + tournament_wins) / total_games = final_win_rate :=
by sorry

end NUMINAMATH_CALUDE_lions_volleyball_games_l1184_118498


namespace NUMINAMATH_CALUDE_third_pile_balls_l1184_118481

theorem third_pile_balls (a b c : ℕ) (x : ℕ) :
  a + b + c = 2012 →
  b - x = 17 →
  a - x = 2 * (c - x) →
  c = 665 := by
sorry

end NUMINAMATH_CALUDE_third_pile_balls_l1184_118481


namespace NUMINAMATH_CALUDE_open_box_volume_l1184_118441

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet -/
theorem open_box_volume (sheet_length sheet_width cut_length : ℝ) 
  (h1 : sheet_length = 48)
  (h2 : sheet_width = 36)
  (h3 : cut_length = 4) : 
  (sheet_length - 2 * cut_length) * (sheet_width - 2 * cut_length) * cut_length = 4480 := by
  sorry

end NUMINAMATH_CALUDE_open_box_volume_l1184_118441


namespace NUMINAMATH_CALUDE_order_of_a_l1184_118433

theorem order_of_a (a : ℝ) (h : a^2 + a < 0) : -a > a^2 ∧ a^2 > -a^2 ∧ -a^2 > a := by
  sorry

end NUMINAMATH_CALUDE_order_of_a_l1184_118433


namespace NUMINAMATH_CALUDE_workshop_workers_l1184_118457

/-- The total number of workers in a workshop -/
def total_workers : ℕ := 22

/-- The number of technicians in the workshop -/
def technicians : ℕ := 7

/-- The average salary of all workers -/
def avg_salary_all : ℚ := 850

/-- The average salary of technicians -/
def avg_salary_tech : ℚ := 1000

/-- The average salary of non-technician workers -/
def avg_salary_rest : ℚ := 780

/-- Theorem stating that given the conditions, the total number of workers is 22 -/
theorem workshop_workers :
  (avg_salary_all * total_workers : ℚ) =
  (avg_salary_tech * technicians : ℚ) +
  (avg_salary_rest * (total_workers - technicians) : ℚ) :=
sorry

end NUMINAMATH_CALUDE_workshop_workers_l1184_118457


namespace NUMINAMATH_CALUDE_a_greater_than_b_l1184_118472

theorem a_greater_than_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 12345 = (111 + a) * (111 - b)) : a > b := by
  sorry

end NUMINAMATH_CALUDE_a_greater_than_b_l1184_118472


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1184_118444

/-- A regular polygon with perimeter 180 cm and side length 15 cm has 12 sides. -/
theorem regular_polygon_sides (P : ℝ) (s : ℝ) (n : ℕ) : 
  P = 180 → s = 15 → P = n * s → n = 12 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1184_118444


namespace NUMINAMATH_CALUDE_parabola_intersection_slope_l1184_118479

/-- Parabola defined by y² = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Focus of the parabola -/
def focus : ℝ × ℝ := (1, 0)

/-- Point M -/
def point_M : ℝ × ℝ := (-1, 2)

/-- Line passing through focus with slope k -/
def line (k x : ℝ) : ℝ := k * (x - focus.1)

/-- Intersection points of the line and parabola -/
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | parabola p.1 p.2 ∧ p.2 = line k p.1}

/-- Angle AMB is 90 degrees -/
def right_angle (A B : ℝ × ℝ) : Prop :=
  (A.2 - point_M.2) * (B.2 - point_M.2) = -(A.1 - point_M.1) * (B.1 - point_M.1)

theorem parabola_intersection_slope :
  ∀ k : ℝ, ∃ A B : ℝ × ℝ,
    A ∈ intersection_points k ∧
    B ∈ intersection_points k ∧
    A ≠ B ∧
    right_angle A B →
    k = 1 := by sorry

end NUMINAMATH_CALUDE_parabola_intersection_slope_l1184_118479


namespace NUMINAMATH_CALUDE_tangent_points_parallel_to_line_l1184_118470

-- Define the function f(x) = x³ + x - 2
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_points_parallel_to_line :
  ∀ x y : ℝ, 
  (y = f x) ∧ 
  (f' x = 4) → 
  ((x = -1 ∧ y = -4) ∨ (x = 1 ∧ y = 0)) :=
sorry

end NUMINAMATH_CALUDE_tangent_points_parallel_to_line_l1184_118470


namespace NUMINAMATH_CALUDE_double_inequality_solution_l1184_118431

theorem double_inequality_solution (x : ℝ) : 
  -1 < (x^2 - 20*x + 21) / (x^2 - 4*x + 5) ∧ 
  (x^2 - 20*x + 21) / (x^2 - 4*x + 5) < 1 ↔ 
  (2 < x ∧ x < 1) ∨ (26 < x) :=
sorry

end NUMINAMATH_CALUDE_double_inequality_solution_l1184_118431


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1184_118468

/-- A hyperbola and a parabola sharing a common focus -/
structure HyperbolaParabola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0
  F : ℝ × ℝ
  P : ℝ × ℝ
  h_parabola : (P.2)^2 = 8 * P.1
  h_hyperbola : (P.1)^2 / a^2 - (P.2)^2 / b^2 = 1
  h_common_focus : F = (2, 0)
  h_distance : Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2) = 5

/-- The equation of the hyperbola is x^2 - y^2/3 = 1 -/
theorem hyperbola_equation (hp : HyperbolaParabola) : 
  hp.a = 1 ∧ hp.b = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1184_118468


namespace NUMINAMATH_CALUDE_distribute_seven_balls_four_boxes_l1184_118491

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 890 ways to distribute 7 distinguishable balls into 4 indistinguishable boxes -/
theorem distribute_seven_balls_four_boxes : distribute_balls 7 4 = 890 := by
  sorry

end NUMINAMATH_CALUDE_distribute_seven_balls_four_boxes_l1184_118491


namespace NUMINAMATH_CALUDE_tommy_pencil_case_items_l1184_118493

/-- The number of items in Tommy's pencil case -/
theorem tommy_pencil_case_items (pencils : ℕ) (pens : ℕ) (eraser : ℕ) 
    (h1 : pens = 2 * pencils) 
    (h2 : eraser = 1)
    (h3 : pencils = 4) : 
  pencils + pens + eraser = 13 := by
  sorry

end NUMINAMATH_CALUDE_tommy_pencil_case_items_l1184_118493


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1184_118480

theorem polynomial_factorization (x : ℝ) :
  x^6 - 4*x^4 + 6*x^2 - 4 = (x^2 - 1)*(x^4 - 2*x^2 + 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1184_118480


namespace NUMINAMATH_CALUDE_square_root_subtraction_l1184_118471

theorem square_root_subtraction : Real.sqrt 81 - Real.sqrt 144 * 3 = -27 := by
  sorry

end NUMINAMATH_CALUDE_square_root_subtraction_l1184_118471


namespace NUMINAMATH_CALUDE_linear_function_properties_l1184_118451

-- Define the linear function
def f (x : ℝ) : ℝ := -3 * x + 2

-- Define the original line before moving up
def g (x : ℝ) : ℝ := 2 * x - 4

-- Define the line after moving up by 5 units
def h (x : ℝ) : ℝ := g x + 5

theorem linear_function_properties :
  (∀ x y : ℝ, x < 0 ∧ y < 0 → f x ≠ y) ∧
  (∀ x : ℝ, h x = 2 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_properties_l1184_118451


namespace NUMINAMATH_CALUDE_simplify_expression_l1184_118442

theorem simplify_expression (a b : ℝ) (hb : b ≠ 0) (ha : a ≠ b^(1/3)) :
  (a^3 - b^3) / (a * b) - (a * b - b^2) / (a * b - a^3) = (a^2 + a * b + b^2) / b :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1184_118442


namespace NUMINAMATH_CALUDE_class_size_l1184_118453

theorem class_size : ∃ n : ℕ, 
  (20 < n ∧ n < 30) ∧ 
  (∃ x : ℕ, n = 3 * x) ∧ 
  (∃ y : ℕ, n = 4 * y - 1) ∧ 
  n = 27 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l1184_118453


namespace NUMINAMATH_CALUDE_angle_in_second_quadrant_l1184_118458

theorem angle_in_second_quadrant (α : Real) :
  (π / 2 < α) ∧ (α < π) →  -- α is in the second quadrant
  |Real.cos (α / 3)| = -Real.cos (α / 3) →  -- |cos(α/3)| = -cos(α/3)
  (π / 2 < α / 3) ∧ (α / 3 < π)  -- α/3 is in the second quadrant
:= by sorry

end NUMINAMATH_CALUDE_angle_in_second_quadrant_l1184_118458


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1184_118423

theorem polynomial_simplification (x : ℝ) :
  (12 * x^10 + 5 * x^9 + 3 * x^8) + (2 * x^12 + 9 * x^10 + 4 * x^8 + 6 * x^4 + 7 * x^2 + 10) =
  2 * x^12 + 21 * x^10 + 5 * x^9 + 7 * x^8 + 6 * x^4 + 7 * x^2 + 10 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1184_118423


namespace NUMINAMATH_CALUDE_max_bishops_on_8x8_chessboard_l1184_118473

/-- Represents a chessboard --/
structure Chessboard :=
  (size : Nat)
  (is_valid : size = 8)

/-- Represents a bishop placement on the chessboard --/
structure BishopPlacement :=
  (board : Chessboard)
  (num_bishops : Nat)
  (max_per_diagonal : Nat)
  (is_valid : max_per_diagonal = 3)

/-- The maximum number of bishops that can be placed on the chessboard --/
def max_bishops (placement : BishopPlacement) : Nat :=
  38

/-- Theorem stating the maximum number of bishops on an 8x8 chessboard --/
theorem max_bishops_on_8x8_chessboard (placement : BishopPlacement) :
  placement.board.size = 8 →
  placement.max_per_diagonal = 3 →
  max_bishops placement = 38 := by
  sorry

end NUMINAMATH_CALUDE_max_bishops_on_8x8_chessboard_l1184_118473


namespace NUMINAMATH_CALUDE_card_area_after_shortening_l1184_118440

/-- Given a rectangle with initial dimensions 5 × 7 inches, 
    prove that shortening both sides by 1 inch results in an area of 24 square inches. -/
theorem card_area_after_shortening :
  let initial_length : ℝ := 7
  let initial_width : ℝ := 5
  let shortened_length : ℝ := initial_length - 1
  let shortened_width : ℝ := initial_width - 1
  shortened_length * shortened_width = 24 := by
  sorry

end NUMINAMATH_CALUDE_card_area_after_shortening_l1184_118440


namespace NUMINAMATH_CALUDE_nuts_to_raisins_cost_ratio_l1184_118455

/-- The ratio of the cost of nuts to raisins given the mixture proportions and cost ratio -/
theorem nuts_to_raisins_cost_ratio 
  (raisin_pounds : ℝ) 
  (nuts_pounds : ℝ)
  (raisin_cost : ℝ)
  (nuts_cost : ℝ)
  (h1 : raisin_pounds = 3)
  (h2 : nuts_pounds = 4)
  (h3 : raisin_cost > 0)
  (h4 : nuts_cost > 0)
  (h5 : raisin_pounds * raisin_cost = 0.15789473684210525 * (raisin_pounds * raisin_cost + nuts_pounds * nuts_cost)) :
  nuts_cost / raisin_cost = 4 := by
sorry

end NUMINAMATH_CALUDE_nuts_to_raisins_cost_ratio_l1184_118455


namespace NUMINAMATH_CALUDE_expression_bounds_l1184_118435

theorem expression_bounds (x y z w : Real) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) (hw : 0 ≤ w ∧ w ≤ 1) :
  2 * Real.sqrt 2 ≤ Real.sqrt (x^2 + (1-y)^2) + Real.sqrt (y^2 + (1-z)^2) + 
                    Real.sqrt (z^2 + (1-w)^2) + Real.sqrt (w^2 + (1-x)^2) ∧
  Real.sqrt (x^2 + (1-y)^2) + Real.sqrt (y^2 + (1-z)^2) + 
  Real.sqrt (z^2 + (1-w)^2) + Real.sqrt (w^2 + (1-x)^2) ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_bounds_l1184_118435


namespace NUMINAMATH_CALUDE_inverse_direct_variation_l1184_118484

/-- Given that a²c varies inversely with b³ and c varies directly as b², 
    prove that a² = 25/128 when b = 4, given initial conditions. -/
theorem inverse_direct_variation (a b c : ℝ) (k k' : ℝ) : 
  (∀ a b c, a^2 * c * b^3 = k) →  -- a²c varies inversely with b³
  (∀ b c, c = k' * b^2) →         -- c varies directly as b²
  (5^2 * 12 * 2^3 = k) →          -- initial condition for k
  (12 = k' * 2^2) →               -- initial condition for k'
  (∀ a, a^2 * (k' * 4^2) * 4^3 = k) →  -- condition for b = 4
  (∃ a, a^2 = 25 / 128) :=
by sorry

end NUMINAMATH_CALUDE_inverse_direct_variation_l1184_118484


namespace NUMINAMATH_CALUDE_max_ab_value_l1184_118434

theorem max_ab_value (a b : ℝ) 
  (h : ∀ x : ℝ, Real.exp x ≥ a * (x - 1) + b) : 
  a * b ≤ (1 / 2) * Real.exp 3 := by
  sorry

end NUMINAMATH_CALUDE_max_ab_value_l1184_118434


namespace NUMINAMATH_CALUDE_exists_divisible_by_33_l1184_118452

def original_number : ℕ := 975312468

def insert_digit (n : ℕ) (d : ℕ) (pos : ℕ) : ℕ :=
  let digits := n.digits 10
  let (before, after) := digits.splitAt pos
  ((before ++ [d] ++ after).foldl (fun acc x => acc * 10 + x) 0)

theorem exists_divisible_by_33 :
  ∃ (d : ℕ) (pos : ℕ), d < 10 ∧ pos ≤ 9 ∧ 
  (insert_digit original_number d pos) % 33 = 0 :=
sorry

end NUMINAMATH_CALUDE_exists_divisible_by_33_l1184_118452


namespace NUMINAMATH_CALUDE_isabella_house_paintable_area_l1184_118419

-- Define the problem parameters
def num_bedrooms : ℕ := 4
def room_length : ℝ := 15
def room_width : ℝ := 12
def room_height : ℝ := 9
def unpaintable_area : ℝ := 80

-- Define the function to calculate the paintable area
def paintable_area : ℝ :=
  let total_wall_area := num_bedrooms * (2 * (room_length * room_height + room_width * room_height))
  total_wall_area - (num_bedrooms * unpaintable_area)

-- State the theorem
theorem isabella_house_paintable_area :
  paintable_area = 1624 := by sorry

end NUMINAMATH_CALUDE_isabella_house_paintable_area_l1184_118419


namespace NUMINAMATH_CALUDE_apple_count_theorem_l1184_118429

def is_valid_apple_count (n : ℕ) : Prop :=
  70 ≤ n ∧ n ≤ 80 ∧ (n % 6 = 0)

theorem apple_count_theorem :
  ∀ n : ℕ, is_valid_apple_count n ↔ (n = 72 ∨ n = 78) :=
by sorry

end NUMINAMATH_CALUDE_apple_count_theorem_l1184_118429


namespace NUMINAMATH_CALUDE_swimmers_return_simultaneously_l1184_118490

/-- Represents a swimmer in the river scenario -/
structure Swimmer where
  speed : ℝ  -- Speed relative to water
  direction : Int  -- 1 for downstream, -1 for upstream

/-- Represents the river scenario -/
structure RiverScenario where
  current_speed : ℝ
  swim_time : ℝ
  swimmer1 : Swimmer
  swimmer2 : Swimmer

/-- Calculates the time taken for a swimmer to return to the raft -/
def return_time (scenario : RiverScenario) (swimmer : Swimmer) : ℝ :=
  2 * scenario.swim_time

theorem swimmers_return_simultaneously (scenario : RiverScenario) :
  return_time scenario scenario.swimmer1 = return_time scenario scenario.swimmer2 :=
sorry

end NUMINAMATH_CALUDE_swimmers_return_simultaneously_l1184_118490


namespace NUMINAMATH_CALUDE_class_strength_solution_l1184_118446

/-- Represents the problem of finding the original class strength --/
def find_original_class_strength (original_avg : ℝ) (new_students : ℕ) (new_avg : ℝ) (avg_decrease : ℝ) : Prop :=
  ∃ (original_strength : ℕ),
    (original_strength : ℝ) * original_avg + (new_students : ℝ) * new_avg = 
    ((original_strength : ℝ) + new_students) * (original_avg - avg_decrease)

/-- The theorem stating the solution to the class strength problem --/
theorem class_strength_solution : 
  find_original_class_strength 40 18 32 4 → 
  ∃ (original_strength : ℕ), original_strength = 18 :=
by
  sorry

#check class_strength_solution

end NUMINAMATH_CALUDE_class_strength_solution_l1184_118446


namespace NUMINAMATH_CALUDE_minimum_discount_l1184_118497

theorem minimum_discount (n : ℕ) : n = 38 ↔ 
  (n > 0) ∧
  (∀ m : ℕ, m < n → 
    ((1 - m / 100 : ℝ) ≥ (1 - 0.20) * (1 - 0.10) ∨
     (1 - m / 100 : ℝ) ≥ (1 - 0.08)^4 ∨
     (1 - m / 100 : ℝ) ≥ (1 - 0.30) * (1 - 0.10))) ∧
  ((1 - n / 100 : ℝ) < (1 - 0.20) * (1 - 0.10) ∧
   (1 - n / 100 : ℝ) < (1 - 0.08)^4 ∧
   (1 - n / 100 : ℝ) < (1 - 0.30) * (1 - 0.10)) :=
by sorry

end NUMINAMATH_CALUDE_minimum_discount_l1184_118497


namespace NUMINAMATH_CALUDE_sqrt_comparison_l1184_118465

theorem sqrt_comparison : Real.sqrt 7 - Real.sqrt 6 < Real.sqrt 3 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_comparison_l1184_118465


namespace NUMINAMATH_CALUDE_partitions_count_l1184_118407

/-- The number of partitions of a set with n+1 elements into n subsets -/
def num_partitions (n : ℕ) : ℕ := (2^n - 1) * n + 1

/-- Theorem stating the number of partitions of a set with n+1 elements into n subsets -/
theorem partitions_count (n : ℕ) (h : n > 0) :
  num_partitions n = (2^n - 1) * n + 1 :=
by sorry

end NUMINAMATH_CALUDE_partitions_count_l1184_118407


namespace NUMINAMATH_CALUDE_sin_negative_1920_degrees_l1184_118406

theorem sin_negative_1920_degrees : 
  Real.sin ((-1920 : ℝ) * π / 180) = -Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_negative_1920_degrees_l1184_118406


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l1184_118496

theorem rhombus_diagonal (d1 : ℝ) (area : ℝ) (d2 : ℝ) : 
  d1 = 25 → area = 375 → area = (d1 * d2) / 2 → d2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l1184_118496


namespace NUMINAMATH_CALUDE_barChartMostEffective_l1184_118436

-- Define an enumeration for chart types
inductive ChartType
  | BarChart
  | LineChart
  | PieChart

-- Define a function to evaluate the effectiveness of a chart type for comparing quantities
def effectivenessForQuantityComparison (chart : ChartType) : Nat :=
  match chart with
  | ChartType.BarChart => 3
  | ChartType.LineChart => 2
  | ChartType.PieChart => 1

-- Theorem stating that BarChart is the most effective for quantity comparison
theorem barChartMostEffective :
  ∀ (chart : ChartType),
    chart ≠ ChartType.BarChart →
    effectivenessForQuantityComparison ChartType.BarChart > effectivenessForQuantityComparison chart :=
by
  sorry


end NUMINAMATH_CALUDE_barChartMostEffective_l1184_118436


namespace NUMINAMATH_CALUDE_average_discount_rate_proof_l1184_118411

theorem average_discount_rate_proof (bag_marked bag_sold shoes_marked shoes_sold jacket_marked jacket_sold : ℝ) 
  (h1 : bag_marked = 80)
  (h2 : bag_sold = 68)
  (h3 : shoes_marked = 120)
  (h4 : shoes_sold = 96)
  (h5 : jacket_marked = 150)
  (h6 : jacket_sold = 135) :
  let bag_discount := (bag_marked - bag_sold) / bag_marked
  let shoes_discount := (shoes_marked - shoes_sold) / shoes_marked
  let jacket_discount := (jacket_marked - jacket_sold) / jacket_marked
  (bag_discount + shoes_discount + jacket_discount) / 3 = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_average_discount_rate_proof_l1184_118411


namespace NUMINAMATH_CALUDE_cube_surface_division_l1184_118482

-- Define a cube
structure Cube where
  vertices : Finset (Fin 8)
  edges : Finset (Fin 12)
  faces : Finset (Fin 6)
  bodyDiagonals : Finset (Fin 4)

-- Define a plane
structure Plane where
  normal : Vector ℝ 3

-- Define the function to erect planes perpendicular to body diagonals
def erectPerpendicularPlanes (c : Cube) : Finset Plane := sorry

-- Define the function to count surface parts
def countSurfaceParts (c : Cube) (planes : Finset Plane) : ℕ := sorry

-- Theorem statement
theorem cube_surface_division (c : Cube) :
  let perpendicularPlanes := erectPerpendicularPlanes c
  countSurfaceParts c perpendicularPlanes = 14 := by sorry

end NUMINAMATH_CALUDE_cube_surface_division_l1184_118482


namespace NUMINAMATH_CALUDE_infinite_primes_dividing_sequence_l1184_118460

theorem infinite_primes_dividing_sequence (a b c : ℕ) (ha : a ≠ c) (hb : b ≠ c) :
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ∃ n : ℕ, p ∣ a^n + b^n - c^n} := by
  sorry

end NUMINAMATH_CALUDE_infinite_primes_dividing_sequence_l1184_118460


namespace NUMINAMATH_CALUDE_union_A_B_intersection_A_complement_B_l1184_118428

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x^2 - 4*x ≤ 0}

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = {x | -1 ≤ x ∧ x ≤ 4} := by sorry

-- Theorem for A ∩ (ℝ \ B)
theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {x | -1 ≤ x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_union_A_B_intersection_A_complement_B_l1184_118428


namespace NUMINAMATH_CALUDE_unique_prime_sum_l1184_118417

/-- Given seven distinct positive integers not exceeding 7, prove that 179 is the only prime expressible as abcd + efg -/
theorem unique_prime_sum (a b c d e f g : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
  e ≠ f ∧ e ≠ g ∧
  f ≠ g ∧
  0 < a ∧ a ≤ 7 ∧
  0 < b ∧ b ≤ 7 ∧
  0 < c ∧ c ≤ 7 ∧
  0 < d ∧ d ≤ 7 ∧
  0 < e ∧ e ≤ 7 ∧
  0 < f ∧ f ≤ 7 ∧
  0 < g ∧ g ≤ 7 →
  (∃ p : ℕ, Nat.Prime p ∧ p = a * b * c * d + e * f * g) ↔ (a * b * c * d + e * f * g = 179) :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_sum_l1184_118417


namespace NUMINAMATH_CALUDE_south_five_is_negative_five_l1184_118475

/-- Represents the direction of movement -/
inductive Direction
| North
| South

/-- Represents a movement with magnitude and direction -/
structure Movement where
  magnitude : ℕ
  direction : Direction

/-- Function to convert a movement to its signed representation -/
def movementToSigned (m : Movement) : ℤ :=
  match m.direction with
  | Direction.North => m.magnitude
  | Direction.South => -m.magnitude

theorem south_five_is_negative_five :
  let southFive : Movement := ⟨5, Direction.South⟩
  movementToSigned southFive = -5 := by sorry

end NUMINAMATH_CALUDE_south_five_is_negative_five_l1184_118475


namespace NUMINAMATH_CALUDE_infinite_solutions_condition_l1184_118454

theorem infinite_solutions_condition (c : ℝ) : 
  (∀ y : ℝ, y ≠ 0 → 3 * (5 + 2 * c * y) = 15 * y + 15) ↔ c = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_condition_l1184_118454


namespace NUMINAMATH_CALUDE_amy_spending_at_fair_l1184_118477

/-- Amy's spending at the fair --/
theorem amy_spending_at_fair (initial_amount final_amount : ℕ) 
  (h1 : initial_amount = 15)
  (h2 : final_amount = 11) :
  initial_amount - final_amount = 4 := by
  sorry

end NUMINAMATH_CALUDE_amy_spending_at_fair_l1184_118477
