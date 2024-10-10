import Mathlib

namespace product_of_four_consecutive_integers_l1176_117678

theorem product_of_four_consecutive_integers (n : ℤ) :
  ∃ M : ℤ, 
    Even M ∧ 
    (n - 1) * n * (n + 1) * (n + 2) = (M - 2) * M := by
  sorry

end product_of_four_consecutive_integers_l1176_117678


namespace slipper_cost_l1176_117676

/-- Calculates the total cost of a pair of embroidered slippers with shipping --/
theorem slipper_cost (original_price discount_percentage embroidery_cost_per_shoe shipping_cost : ℚ) :
  original_price = 50 →
  discount_percentage = 10 →
  embroidery_cost_per_shoe = (11/2) →
  shipping_cost = 10 →
  original_price * (1 - discount_percentage / 100) + 2 * embroidery_cost_per_shoe + shipping_cost = 66 := by
sorry


end slipper_cost_l1176_117676


namespace mean_equality_implies_y_equals_three_l1176_117673

theorem mean_equality_implies_y_equals_three :
  let mean1 := (3 + 7 + 11 + 15) / 4
  let mean2 := (10 + 14 + y) / 3
  mean1 = mean2 → y = 3 :=
by
  sorry

end mean_equality_implies_y_equals_three_l1176_117673


namespace probability_standard_weight_l1176_117693

def total_students : ℕ := 500
def standard_weight_students : ℕ := 350

theorem probability_standard_weight :
  (standard_weight_students : ℚ) / total_students = 7 / 10 := by
  sorry

end probability_standard_weight_l1176_117693


namespace square_less_than_four_implies_less_than_two_l1176_117692

theorem square_less_than_four_implies_less_than_two (x : ℝ) : x^2 < 4 → x < 2 := by
  sorry

end square_less_than_four_implies_less_than_two_l1176_117692


namespace milo_cash_reward_l1176_117644

def grades : List ℕ := [2, 2, 2, 3, 3, 3, 3, 4, 5]

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

def cashReward (avg : ℚ) : ℚ := 5 * avg

theorem milo_cash_reward :
  cashReward (average grades) = 15 := by
  sorry

end milo_cash_reward_l1176_117644


namespace three_digit_number_divisible_by_seven_l1176_117674

theorem three_digit_number_divisible_by_seven :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 
  n % 10 = 4 ∧ 
  (n / 100) % 10 = 5 ∧ 
  n % 7 = 0 ∧
  n = 534 := by
sorry

end three_digit_number_divisible_by_seven_l1176_117674


namespace sequence_difference_l1176_117664

theorem sequence_difference (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n, S n = n^2) : 
  a 3 - a 2 = 2 := by
  sorry

end sequence_difference_l1176_117664


namespace matrix_sum_equality_l1176_117675

def matrix1 : Matrix (Fin 3) (Fin 3) ℤ := !![2, -1, 3; 0, 4, -2; 5, -3, 1]
def matrix2 : Matrix (Fin 3) (Fin 3) ℤ := !![-3, 2, -4; 1, -6, 3; -2, 4, 0]
def result : Matrix (Fin 3) (Fin 3) ℤ := !![-1, 1, -1; 1, -2, 1; 3, 1, 1]

theorem matrix_sum_equality : matrix1 + matrix2 = result := by
  sorry

end matrix_sum_equality_l1176_117675


namespace wand_price_theorem_l1176_117661

theorem wand_price_theorem (purchase_price : ℝ) (original_price : ℝ) : 
  purchase_price = 8 →
  purchase_price = (1/8) * original_price →
  original_price = 64 := by
sorry

end wand_price_theorem_l1176_117661


namespace solution_set_f_greater_than_2_min_value_f_range_of_a_l1176_117691

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 4| + |x - 2|

-- Theorem 1: Solution set of f(x) > 2
theorem solution_set_f_greater_than_2 :
  {x : ℝ | f x > 2} = {x : ℝ | x < 2 ∨ x > 4} := by sorry

-- Theorem 2: Minimum value of f(x)
theorem min_value_f :
  ∃ M : ℝ, M = 2 ∧ ∀ x : ℝ, f x ≥ M := by sorry

-- Theorem 3: Range of a
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → 2^x + a ≥ 2) → a ≥ 1 := by sorry

end solution_set_f_greater_than_2_min_value_f_range_of_a_l1176_117691


namespace discount_card_problem_l1176_117600

/-- Proves that given a discount card that costs 20 yuan and provides a 20% discount,
    if a customer saves 12 yuan by using the card, then the original price of the purchase
    before the discount was 160 yuan. -/
theorem discount_card_problem (card_cost discount_rate savings original_price : ℝ)
    (h1 : card_cost = 20)
    (h2 : discount_rate = 0.2)
    (h3 : savings = 12)
    (h4 : card_cost + (1 - discount_rate) * original_price = original_price - savings) :
    original_price = 160 :=
  sorry

end discount_card_problem_l1176_117600


namespace tangent_line_to_parabola_l1176_117628

/-- Given that the line x - y - 1 = 0 is tangent to the parabola y = ax², prove that a = 1/4 -/
theorem tangent_line_to_parabola (a : ℝ) : 
  (∃ x y : ℝ, x - y - 1 = 0 ∧ y = a * x^2 ∧ 
   ∀ x' y' : ℝ, y' = a * x'^2 → (x - x') * (2 * a * x) = y - y') → 
  a = 1/4 := by
sorry

end tangent_line_to_parabola_l1176_117628


namespace prob_three_diff_suits_is_169_425_l1176_117626

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the suits in a standard deck -/
inductive Suit
  | Hearts | Diamonds | Clubs | Spades

/-- Function to get the suit of a card -/
def card_suit : Fin 52 → Suit := sorry

/-- Probability of drawing three cards of different suits -/
def prob_three_different_suits (d : Deck) : ℚ :=
  let first_draw := d.cards.card
  let second_draw := d.cards.card - 1
  let third_draw := d.cards.card - 2
  let diff_suit_second := 39
  let diff_suit_third := 26
  (diff_suit_second / second_draw) * (diff_suit_third / third_draw)

theorem prob_three_diff_suits_is_169_425 (d : Deck) :
  prob_three_different_suits d = 169 / 425 := by sorry

end prob_three_diff_suits_is_169_425_l1176_117626


namespace smallest_area_right_triangle_l1176_117631

theorem smallest_area_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  let area := (1/2) * a * b
  area = 24 ∧ ∀ (x y : ℝ), (x = a ∧ y = b) ∨ (x = a ∧ y = b) ∨ (x^2 + y^2 = a^2 + b^2) → (1/2) * x * y ≥ area :=
by sorry

end smallest_area_right_triangle_l1176_117631


namespace short_trees_after_planting_l1176_117627

theorem short_trees_after_planting 
  (initial_short_trees : ℕ) 
  (short_trees_to_plant : ℕ) 
  (h1 : initial_short_trees = 112)
  (h2 : short_trees_to_plant = 105) :
  initial_short_trees + short_trees_to_plant = 217 := by
  sorry

end short_trees_after_planting_l1176_117627


namespace oranges_taken_l1176_117618

theorem oranges_taken (initial_oranges final_oranges : ℕ) 
  (h1 : initial_oranges = 60)
  (h2 : final_oranges = 25) :
  initial_oranges - final_oranges = 35 := by
  sorry

end oranges_taken_l1176_117618


namespace minimal_sequence_is_first_l1176_117680

/-- A sequence of n natural numbers -/
def Sequence (n : ℕ) := Fin n → ℕ

/-- Property: strictly decreasing -/
def IsStrictlyDecreasing (s : Sequence n) : Prop :=
  ∀ i j, i < j → s i > s j

/-- Property: no term divides any other term -/
def NoDivisibility (s : Sequence n) : Prop :=
  ∀ i j, i ≠ j → ¬(s i ∣ s j)

/-- Ordering relation between sequences -/
def Precedes (a b : Sequence n) : Prop :=
  ∃ k, (∀ i < k, a i = b i) ∧ a k < b k

/-- The proposed minimal sequence -/
def MinimalSequence (n : ℕ) : Sequence n :=
  λ i => 2 * n - 1 - 2 * i.val

theorem minimal_sequence_is_first (n : ℕ) :
  IsStrictlyDecreasing (MinimalSequence n) ∧
  NoDivisibility (MinimalSequence n) ∧
  (∀ s : Sequence n, IsStrictlyDecreasing s → NoDivisibility s →
    s = MinimalSequence n ∨ Precedes (MinimalSequence n) s) :=
sorry

end minimal_sequence_is_first_l1176_117680


namespace waiter_tip_calculation_l1176_117668

theorem waiter_tip_calculation (total_customers : ℕ) (non_tipping_customers : ℕ) (total_tips : ℕ) :
  total_customers = 10 →
  non_tipping_customers = 5 →
  total_tips = 15 →
  (total_tips : ℚ) / (total_customers - non_tipping_customers : ℚ) = 3 := by
  sorry

end waiter_tip_calculation_l1176_117668


namespace common_divisors_45_48_l1176_117699

theorem common_divisors_45_48 : Finset.card (Finset.filter (fun d => d ∣ 48) (Nat.divisors 45)) = 4 := by
  sorry

end common_divisors_45_48_l1176_117699


namespace salty_cookies_eaten_correct_l1176_117629

/-- The number of salty cookies Paco ate -/
def salty_cookies_eaten (initial_salty initial_sweet eaten_sweet salty_left : ℕ) : ℕ :=
  initial_salty - salty_left

/-- Theorem: The number of salty cookies Paco ate is the difference between
    the initial number of salty cookies and the number of salty cookies left -/
theorem salty_cookies_eaten_correct
  (initial_salty initial_sweet eaten_sweet salty_left : ℕ)
  (h1 : initial_salty = 26)
  (h2 : initial_sweet = 17)
  (h3 : eaten_sweet = 14)
  (h4 : salty_left = 17)
  (h5 : initial_salty ≥ salty_left) :
  salty_cookies_eaten initial_salty initial_sweet eaten_sweet salty_left = initial_salty - salty_left :=
by
  sorry

end salty_cookies_eaten_correct_l1176_117629


namespace art_gallery_pieces_l1176_117669

theorem art_gallery_pieces (total : ℕ) 
  (h1 : total / 3 = total - (total * 2 / 3))  -- 1/3 of pieces are displayed
  (h2 : (total / 3) / 6 = (total / 3) - ((total / 3) * 5 / 6))  -- 1/6 of displayed pieces are sculptures
  (h3 : (total * 2 / 3) / 3 = (total * 2 / 3) - ((total * 2 / 3) * 2 / 3))  -- 1/3 of not displayed pieces are paintings
  (h4 : (total * 2 / 3) * 2 / 3 = 400)  -- 400 sculptures are not on display
  : total = 900 := by
  sorry

end art_gallery_pieces_l1176_117669


namespace pet_store_cages_l1176_117696

/-- Given a pet store scenario, calculate the number of cages used -/
theorem pet_store_cages (initial_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) : 
  initial_puppies = 120 → 
  sold_puppies = 108 → 
  puppies_per_cage = 6 → 
  (initial_puppies - sold_puppies) / puppies_per_cage = 2 := by
sorry

end pet_store_cages_l1176_117696


namespace parallel_vectors_proportional_components_l1176_117625

/-- Given two 2D vectors a and b, if they are parallel, then their components are proportional. -/
theorem parallel_vectors_proportional_components (a b : ℝ × ℝ) :
  (∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2) ↔ ∃ m : ℝ, a = (2, -1) ∧ b = (-1, m) ∧ m = 1/2 := by
  sorry


end parallel_vectors_proportional_components_l1176_117625


namespace expression_evaluation_l1176_117647

theorem expression_evaluation : 2 + 5 * 3^2 - 4 * 2 + 7 * 3 / 3 = 46 := by
  sorry

end expression_evaluation_l1176_117647


namespace min_value_abc_l1176_117662

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/b + 1/c = 9) :
  a^2 * b^3 * c^4 ≥ 1/1728 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧
    1/a₀ + 1/b₀ + 1/c₀ = 9 ∧ a₀^2 * b₀^3 * c₀^4 = 1/1728 := by
  sorry

end min_value_abc_l1176_117662


namespace lassie_bones_problem_l1176_117634

theorem lassie_bones_problem (B : ℝ) : 
  (4/5 * (3/4 * (2/3 * B + 5) + 8) + 15 = 60) → B = 89 := by
  sorry

end lassie_bones_problem_l1176_117634


namespace tan_arctan_five_twelfths_l1176_117621

theorem tan_arctan_five_twelfths : 
  Real.tan (Real.arctan (5 / 12)) = 5 / 12 := by sorry

end tan_arctan_five_twelfths_l1176_117621


namespace multiplication_formula_examples_l1176_117638

theorem multiplication_formula_examples : 
  (102 * 98 = 9996) ∧ (99^2 = 9801) := by
  sorry

end multiplication_formula_examples_l1176_117638


namespace intersection_sum_l1176_117637

-- Define the parabolas
def parabola1 (x y : ℝ) : Prop := y = (x - 2)^2
def parabola2 (x y : ℝ) : Prop := x + 6 = (y + 1)^2

-- Define the intersection points
def intersection_points (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) : Prop :=
  parabola1 x₁ y₁ ∧ parabola2 x₁ y₁ ∧
  parabola1 x₂ y₂ ∧ parabola2 x₂ y₂ ∧
  parabola1 x₃ y₃ ∧ parabola2 x₃ y₃ ∧
  parabola1 x₄ y₄ ∧ parabola2 x₄ y₄

-- Theorem statement
theorem intersection_sum (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) :
  intersection_points x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ →
  x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 6 :=
by
  sorry

end intersection_sum_l1176_117637


namespace union_and_intersection_range_of_a_l1176_117665

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x < 5}
def B : Set ℝ := {x | 2 < x ∧ x < 8}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x ≤ a + 3}

-- Theorem for part (1)
theorem union_and_intersection :
  (A ∪ B = {x | 1 ≤ x ∧ x < 8}) ∧
  ((Set.univ \ A) ∩ B = {x | 5 ≤ x ∧ x < 8}) := by sorry

-- Theorem for part (2)
theorem range_of_a :
  {a : ℝ | C a ∩ A = C a} = {a : ℝ | 1 ≤ a ∧ a < 2} := by sorry

end union_and_intersection_range_of_a_l1176_117665


namespace circle_min_radius_l1176_117639

theorem circle_min_radius (a : ℝ) : 
  let r := Real.sqrt ((5 * a^2) / 4 + 2)
  r ≥ Real.sqrt 2 ∧ ∃ a₀, Real.sqrt ((5 * a₀^2) / 4 + 2) = Real.sqrt 2 :=
by sorry

end circle_min_radius_l1176_117639


namespace complex_arithmetic_proof_l1176_117642

theorem complex_arithmetic_proof :
  let z₁ : ℂ := 5 + 6*I
  let z₂ : ℂ := -1 + 4*I
  let z₃ : ℂ := 3 - 2*I
  (z₁ + z₂) - z₃ = 1 + 12*I :=
by sorry

end complex_arithmetic_proof_l1176_117642


namespace tempo_original_value_l1176_117601

/-- The original value of a tempo given insurance and premium information -/
def original_value (insured_fraction : ℚ) (premium_rate : ℚ) (premium_amount : ℚ) : ℚ :=
  premium_amount / (premium_rate * insured_fraction)

/-- Theorem stating the original value of the tempo given the problem conditions -/
theorem tempo_original_value :
  let insured_fraction : ℚ := 4 / 5
  let premium_rate : ℚ := 13 / 1000
  let premium_amount : ℚ := 910
  original_value insured_fraction premium_rate premium_amount = 87500 := by
sorry

end tempo_original_value_l1176_117601


namespace inverse_proportion_solution_l1176_117641

/-- Two real numbers are inversely proportional if their product is constant -/
def InverselyProportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_solution (x y : ℝ) :
  InverselyProportional x y →
  x + y = 30 →
  x - y = 10 →
  (∃ y' : ℝ, InverselyProportional 4 y' ∧ y' = 50) :=
by sorry

end inverse_proportion_solution_l1176_117641


namespace magnitude_v_l1176_117698

theorem magnitude_v (u v : ℂ) (h1 : u * v = 16 - 30 * I) (h2 : Complex.abs u = 2) : 
  Complex.abs v = 17 := by
sorry

end magnitude_v_l1176_117698


namespace book_reading_days_l1176_117685

theorem book_reading_days : ∀ (total_pages : ℕ) (pages_per_day : ℕ) (fraction : ℚ),
  total_pages = 144 →
  pages_per_day = 8 →
  fraction = 2/3 →
  (fraction * total_pages : ℚ) / pages_per_day = 12 :=
by
  sorry

end book_reading_days_l1176_117685


namespace range_of_a_for_inequality_l1176_117611

theorem range_of_a_for_inequality : 
  {a : ℝ | ∀ x : ℝ, |x + 3| + |x - 1| ≥ a^2 - 3*a} = {a : ℝ | -1 ≤ a ∧ a ≤ 4} := by
  sorry

end range_of_a_for_inequality_l1176_117611


namespace problem_polygon_area_l1176_117658

/-- Polygon PQRSTU with given side lengths and properties -/
structure Polygon where
  PQ : ℝ
  QR : ℝ
  RS : ℝ
  ST : ℝ
  TU : ℝ
  PT_parallel_QR : Bool
  PU_divides : Bool

/-- Calculate the area of the polygon PQRSTU -/
def polygon_area (p : Polygon) : ℝ :=
  sorry

/-- The specific polygon from the problem -/
def problem_polygon : Polygon := {
  PQ := 4
  QR := 7
  RS := 5
  ST := 6
  TU := 3
  PT_parallel_QR := true
  PU_divides := true
}

/-- Theorem stating that the area of the problem polygon is 41.5 square units -/
theorem problem_polygon_area :
  polygon_area problem_polygon = 41.5 := by sorry

end problem_polygon_area_l1176_117658


namespace multiply_polynomials_l1176_117607

theorem multiply_polynomials (x : ℝ) : 
  (x^4 + 12*x^2 + 144) * (x^2 - 12) = x^6 + 12*x^4 - 144*x^2 - 1728 := by
  sorry

end multiply_polynomials_l1176_117607


namespace steve_height_l1176_117606

/-- Converts feet to inches -/
def feet_to_inches (feet : ℕ) : ℕ := feet * 12

/-- Converts a height given in feet and inches to total inches -/
def height_to_inches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet_to_inches feet + inches

/-- Calculates the final height after growth -/
def final_height (initial_feet : ℕ) (initial_inches : ℕ) (growth : ℕ) : ℕ :=
  height_to_inches initial_feet initial_inches + growth

theorem steve_height :
  final_height 5 6 6 = 72 := by sorry

end steve_height_l1176_117606


namespace simplest_form_fraction_other_fractions_not_simplest_l1176_117672

/-- A fraction is in simplest form if its numerator and denominator have no common factors other than 1. -/
def IsSimplestForm (n d : ℤ) : Prop :=
  ∀ k : ℤ, k ∣ n ∧ k ∣ d → k = 1 ∨ k = -1

/-- The fraction (x^2 + y^2) / (x + y) is in simplest form. -/
theorem simplest_form_fraction (x y : ℤ) :
  IsSimplestForm (x^2 + y^2) (x + y) := by
  sorry

/-- Other fractions can be simplified further. -/
theorem other_fractions_not_simplest (x y : ℤ) :
  ¬IsSimplestForm (x * y) (x^2) ∧
  ¬IsSimplestForm (y^2 + y) (x * y) ∧
  ¬IsSimplestForm (x^2 - y^2) (x + y) := by
  sorry

end simplest_form_fraction_other_fractions_not_simplest_l1176_117672


namespace sum_of_fraction_parts_l1176_117615

def repeating_decimal : ℚ := 123 / 999

theorem sum_of_fraction_parts : ∃ (n d : ℕ), 
  repeating_decimal = n / d ∧ 
  Nat.gcd n d = 1 ∧ 
  n + d = 374 := by sorry

end sum_of_fraction_parts_l1176_117615


namespace min_sum_of_coeffs_l1176_117640

/-- Given a quadratic function f(x) with real coefficients a, b, c, 
    if the range of f(x) is [0, +∞), then a + b + c ≥ √3 -/
theorem min_sum_of_coeffs (a b c : ℝ) : 
  (∀ x, (a + 2*b)*x^2 - 2*Real.sqrt 3*x + a + 2*c ≥ 0) → 
  a + b + c ≥ Real.sqrt 3 :=
sorry

end min_sum_of_coeffs_l1176_117640


namespace no_two_digit_primes_with_digit_sum_nine_l1176_117651

def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sumOfDigits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem no_two_digit_primes_with_digit_sum_nine :
  ∀ n : ℕ, isTwoDigit n → sumOfDigits n = 9 → ¬ Nat.Prime n := by
sorry

end no_two_digit_primes_with_digit_sum_nine_l1176_117651


namespace cement_calculation_l1176_117688

/-- The amount of cement originally owned -/
def original_cement : ℕ := sorry

/-- The amount of cement bought -/
def bought_cement : ℕ := 215

/-- The amount of cement brought by the son -/
def son_brought_cement : ℕ := 137

/-- The current total amount of cement -/
def current_cement : ℕ := 450

/-- Theorem stating the relationship between the amounts of cement -/
theorem cement_calculation : 
  original_cement = current_cement - (bought_cement + son_brought_cement) :=
by sorry

end cement_calculation_l1176_117688


namespace function_inequality_l1176_117656

open Real

theorem function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x ∈ (Set.Ioo 0 (π / 2)), HasDerivAt f (f' x) x) →
  (∀ x ∈ (Set.Ioo 0 (π / 2)), f x * tan x + f' x < 0) →
  Real.sqrt 3 * f (π / 3) < f (π / 6) :=
by sorry

end function_inequality_l1176_117656


namespace solution_set_inequality_proof_l1176_117616

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 3|
def g (x : ℝ) : ℝ := |x - 2|

-- Theorem for the solution set of f(x) + g(x) < 2
theorem solution_set : 
  {x : ℝ | f x + g x < 2} = {x : ℝ | 3/2 < x ∧ x < 7/2} := by sorry

-- Theorem for the inequality proof
theorem inequality_proof (x y : ℝ) (hx : f x ≤ 1) (hy : g y ≤ 1) : 
  |x - 2*y + 1| ≤ 3 := by sorry

end solution_set_inequality_proof_l1176_117616


namespace intersection_distance_l1176_117694

/-- The distance between the intersection points of y = x - 3 and x² + 2y² = 8 is 4√3/3 -/
theorem intersection_distance : 
  ∃ (A B : ℝ × ℝ), 
    (A.2 = A.1 - 3 ∧ A.1^2 + 2*A.2^2 = 8) ∧ 
    (B.2 = B.1 - 3 ∧ B.1^2 + 2*B.2^2 = 8) ∧ 
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = (4 * Real.sqrt 3) / 3 :=
by sorry

end intersection_distance_l1176_117694


namespace abs_x_minus_3_eq_1_implies_5_minus_2x_eq_neg_3_or_1_l1176_117666

theorem abs_x_minus_3_eq_1_implies_5_minus_2x_eq_neg_3_or_1 (x : ℝ) :
  |x - 3| = 1 → (5 - 2*x = -3 ∨ 5 - 2*x = 1) :=
by sorry

end abs_x_minus_3_eq_1_implies_5_minus_2x_eq_neg_3_or_1_l1176_117666


namespace cosine_sum_equals_one_l1176_117614

theorem cosine_sum_equals_one (α β γ : Real) 
  (sum_eq_pi : α + β + γ = Real.pi)
  (tan_sum_eq_one : Real.tan ((β + γ - α) / 4) + Real.tan ((γ + α - β) / 4) + Real.tan ((α + β - γ) / 4) = 1) :
  Real.cos α + Real.cos β + Real.cos γ = 1 := by
  sorry

end cosine_sum_equals_one_l1176_117614


namespace change_in_f_l1176_117645

def f (x : ℝ) : ℝ := x^2 - 5*x

theorem change_in_f (x : ℝ) :
  (f (x + 2) - f x = 4*x - 6) ∧
  (f (x - 2) - f x = -4*x + 14) :=
by sorry

end change_in_f_l1176_117645


namespace a_n_formula_l1176_117687

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem a_n_formula (a : ℕ → ℝ) (h1 : arithmetic_sequence (λ n => a (n + 1) - a n))
  (h2 : a 1 - a 0 = 1) (h3 : ∀ n : ℕ, (a (n + 2) - a (n + 1)) - (a (n + 1) - a n) = 2) :
  ∀ n : ℕ, a n = 2^n - 1 :=
sorry

end a_n_formula_l1176_117687


namespace ring_cost_l1176_117657

theorem ring_cost (total_revenue : ℕ) (necklace_count : ℕ) (ring_count : ℕ) (necklace_price : ℕ) :
  total_revenue = 80 →
  necklace_count = 4 →
  ring_count = 8 →
  necklace_price = 12 →
  ∃ (ring_price : ℕ), ring_price = 4 ∧ total_revenue = necklace_count * necklace_price + ring_count * ring_price :=
by
  sorry

end ring_cost_l1176_117657


namespace eight_stairs_climb_ways_l1176_117654

/-- Represents the number of ways to climb n stairs with the given restrictions -/
def climbWays (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | n + 3 =>
    if n % 2 = 0 then
      climbWays (n + 2) + climbWays (n + 1)
    else
      climbWays (n + 2) + climbWays (n + 1) + climbWays n

theorem eight_stairs_climb_ways :
  climbWays 8 = 54 := by
  sorry

#eval climbWays 8

end eight_stairs_climb_ways_l1176_117654


namespace arithmetic_sequence_sum_l1176_117690

/-- Given an arithmetic sequence {a_n} with sum S_n, prove that if S_6 = 36, S_n = 324, S_(n-6) = 144, and n > 0, then n = 18. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) :
  (∀ k, S k = (k / 2) * (2 * a 1 + (k - 1) * (a 2 - a 1))) →  -- Definition of S_n
  (n > 0) →                                                   -- Condition: n > 0
  (S 6 = 36) →                                                -- Condition: S_6 = 36
  (S n = 324) →                                               -- Condition: S_n = 324
  (S (n - 6) = 144) →                                         -- Condition: S_(n-6) = 144
  (n = 18) :=                                                 -- Conclusion: n = 18
by sorry


end arithmetic_sequence_sum_l1176_117690


namespace product_equals_693_over_256_l1176_117635

theorem product_equals_693_over_256 : 
  (1 + 1/2) * (1 + 1/4) * (1 + 1/6) * (1 + 1/8) * (1 + 1/10) = 693/256 := by
  sorry

end product_equals_693_over_256_l1176_117635


namespace smallest_five_digit_divisible_by_3_5_7_l1176_117643

theorem smallest_five_digit_divisible_by_3_5_7 :
  ∀ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n → n ≥ 10080 :=
by
  sorry

#check smallest_five_digit_divisible_by_3_5_7

end smallest_five_digit_divisible_by_3_5_7_l1176_117643


namespace average_of_five_integers_l1176_117630

theorem average_of_five_integers (k m r s t : ℕ) : 
  k < m → m < r → r < s → s < t → 
  t = 20 → r = 13 → k ≥ 1 → m ≥ 2 →
  (k + m + r + s + t) / 5 = 10 := by
  sorry

end average_of_five_integers_l1176_117630


namespace no_solution_system_l1176_117655

theorem no_solution_system : ¬∃ (x y : ℝ), 
  (x^3 + x + y + 1 = 0) ∧ 
  (y*x^2 + x + y = 0) ∧ 
  (y^2 + y - x^2 + 1 = 0) := by
  sorry

end no_solution_system_l1176_117655


namespace beau_current_age_l1176_117613

/-- Represents a person with an age -/
structure Person where
  age : ℕ

/-- Represents Beau and his three sons -/
structure Family where
  beau : Person
  son1 : Person
  son2 : Person
  son3 : Person

/-- The age of Beau's sons today -/
def sonAgeToday : ℕ := 16

/-- The theorem stating Beau's current age -/
theorem beau_current_age (f : Family) : 
  (f.son1.age = sonAgeToday) ∧ 
  (f.son2.age = sonAgeToday) ∧ 
  (f.son3.age = sonAgeToday) ∧ 
  (f.beau.age = f.son1.age + f.son2.age + f.son3.age + 3) → 
  f.beau.age = 42 := by
  sorry


end beau_current_age_l1176_117613


namespace plane_speed_theorem_l1176_117602

theorem plane_speed_theorem (v : ℝ) (h1 : v > 0) :
  5 * v + 5 * (3 * v) = 4800 →
  v = 240 ∧ 3 * v = 720 := by
  sorry

end plane_speed_theorem_l1176_117602


namespace new_person_weight_is_97_l1176_117609

/-- The weight of the new person given the conditions of the problem -/
def weight_of_new_person (initial_count : ℕ) (average_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + initial_count * average_increase

/-- Theorem stating that the weight of the new person is 97 kg -/
theorem new_person_weight_is_97 :
  weight_of_new_person 8 4 65 = 97 := by
  sorry

end new_person_weight_is_97_l1176_117609


namespace angie_pretzels_l1176_117681

theorem angie_pretzels (barry_pretzels : ℕ) (shelly_pretzels : ℕ) (angie_pretzels : ℕ) :
  barry_pretzels = 12 →
  shelly_pretzels = barry_pretzels / 2 →
  angie_pretzels = 3 * shelly_pretzels →
  angie_pretzels = 18 := by
sorry

end angie_pretzels_l1176_117681


namespace demand_increase_factor_l1176_117604

def demand (p : ℝ) : ℝ := 150 - p

def supply (p : ℝ) : ℝ := 3 * p - 10

def new_demand (α : ℝ) (p : ℝ) : ℝ := α * (150 - p)

theorem demand_increase_factor (α : ℝ) :
  (∃ p_initial p_new : ℝ,
    demand p_initial = supply p_initial ∧
    new_demand α p_new = supply p_new ∧
    p_new = 1.25 * p_initial) →
  α = 1.4 := by sorry

end demand_increase_factor_l1176_117604


namespace sum_of_heights_l1176_117608

theorem sum_of_heights (n : ℕ) (h1 : n = 30) (s10 s20 : ℕ) 
  (h2 : s10 = 1450) (h3 : s20 = 3030) : ∃ (s30 : ℕ), s30 = 4610 :=
by
  sorry

end sum_of_heights_l1176_117608


namespace largest_gold_coin_distribution_l1176_117659

theorem largest_gold_coin_distribution (n : ℕ) (h1 : n < 150) 
  (h2 : ∃ k : ℕ, n = 15 * k + 3) : n ≤ 138 := by
  sorry

end largest_gold_coin_distribution_l1176_117659


namespace triangle_square_count_l1176_117632

/-- Represents a geometric figure with three layers -/
structure ThreeLayerFigure where
  first_layer_triangles : Nat
  second_layer_squares : Nat
  third_layer_triangle : Nat

/-- Counts the total number of triangles in the figure -/
def count_triangles (figure : ThreeLayerFigure) : Nat :=
  figure.first_layer_triangles + figure.third_layer_triangle

/-- Counts the total number of squares in the figure -/
def count_squares (figure : ThreeLayerFigure) : Nat :=
  figure.second_layer_squares

/-- The specific figure described in the problem -/
def problem_figure : ThreeLayerFigure :=
  { first_layer_triangles := 3
  , second_layer_squares := 2
  , third_layer_triangle := 1 }

theorem triangle_square_count :
  count_triangles problem_figure = 4 ∧ count_squares problem_figure = 2 := by
  sorry

end triangle_square_count_l1176_117632


namespace tank_water_level_l1176_117686

theorem tank_water_level (tank_capacity : ℚ) (initial_fraction : ℚ) (added_water : ℚ) : 
  tank_capacity = 40 →
  initial_fraction = 3/4 →
  added_water = 5 →
  (initial_fraction * tank_capacity + added_water) / tank_capacity = 7/8 := by
  sorry

end tank_water_level_l1176_117686


namespace min_value_expression_l1176_117633

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : x^2 + y^2 = z) :
  ∃ (min : ℝ), min = -2040200 ∧
  ∀ (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^2 + b^2 = z),
    (a + 1/b) * (a + 1/b - 2020) + (b + 1/a) * (b + 1/a - 2020) ≥ min :=
by sorry

end min_value_expression_l1176_117633


namespace even_odd_sum_difference_1500_l1176_117617

/-- The sum of the first n terms of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The difference between the sum of even and odd numbers -/
def even_odd_sum_difference (n : ℕ) : ℤ :=
  arithmetic_sum 0 2 n - arithmetic_sum 1 2 n

theorem even_odd_sum_difference_1500 :
  even_odd_sum_difference 1500 = -1500 := by
  sorry

end even_odd_sum_difference_1500_l1176_117617


namespace greatest_prime_factor_factorial_sum_l1176_117650

theorem greatest_prime_factor_factorial_sum : 
  (Nat.factors (Nat.factorial 15 + Nat.factorial 18)).maximum? = some 17 := by
  sorry

end greatest_prime_factor_factorial_sum_l1176_117650


namespace rabbit_nuts_count_l1176_117660

theorem rabbit_nuts_count :
  ∀ (rabbit_holes fox_holes : ℕ),
    rabbit_holes = fox_holes + 5 →
    4 * rabbit_holes = 6 * fox_holes →
    4 * rabbit_holes = 60 :=
by
  sorry

end rabbit_nuts_count_l1176_117660


namespace cost_of_dozen_pens_l1176_117649

/-- Given the cost of 3 pens and 5 pencils, and the cost ratio of pen to pencil,
    prove the cost of one dozen pens. -/
theorem cost_of_dozen_pens (cost_3pens_5pencils : ℕ) (cost_ratio_pen_pencil : ℚ) :
  cost_3pens_5pencils = 200 →
  cost_ratio_pen_pencil = 5 / 1 →
  ∃ (cost_pen : ℚ), cost_pen * 12 = 600 := by
  sorry

end cost_of_dozen_pens_l1176_117649


namespace binary_253_ones_minus_zeros_l1176_117620

def binary_representation (n : ℕ) : List Bool :=
  sorry

def count_zeros (l : List Bool) : ℕ :=
  sorry

def count_ones (l : List Bool) : ℕ :=
  sorry

theorem binary_253_ones_minus_zeros :
  let bin_253 := binary_representation 253
  let x := count_zeros bin_253
  let y := count_ones bin_253
  y - x = 6 := by sorry

end binary_253_ones_minus_zeros_l1176_117620


namespace expression_equality_l1176_117682

theorem expression_equality (x y z u a b c d : ℝ) :
  (a*x + b*y + c*z + d*u)^2 + (b*x + c*y + d*z + a*u)^2 + 
  (c*x + d*y + a*z + b*u)^2 + (d*x + a*y + b*z + c*u)^2 =
  (d*x + c*y + b*z + a*u)^2 + (c*x + b*y + a*z + d*u)^2 + 
  (b*x + a*y + d*z + c*u)^2 + (a*x + d*y + c*z + b*u)^2 := by
  sorry

end expression_equality_l1176_117682


namespace probability_multiple_2_or_3_30_l1176_117671

def is_multiple_of_2_or_3 (n : ℕ) : Bool :=
  n % 2 = 0 || n % 3 = 0

def count_multiples (n : ℕ) : ℕ :=
  (List.range n).filter is_multiple_of_2_or_3 |>.length

theorem probability_multiple_2_or_3_30 :
  (count_multiples 30 : ℚ) / 30 = 2 / 3 := by
  sorry

end probability_multiple_2_or_3_30_l1176_117671


namespace roots_sum_of_squares_l1176_117648

theorem roots_sum_of_squares (p q : ℝ) : 
  (p^2 - 5*p + 6 = 0) → (q^2 - 5*q + 6 = 0) → p^2 + q^2 = 13 := by
  sorry

end roots_sum_of_squares_l1176_117648


namespace league_games_l1176_117663

theorem league_games (n : ℕ) (h : n = 12) : (n * (n - 1)) / 2 = 66 := by
  sorry

end league_games_l1176_117663


namespace function_minimum_implies_inequality_l1176_117684

open Real

/-- Given a function f(x) = -3ln(x) + ax² + bx, where a > 0 and b is real,
    if for any x > 0, f(x) ≥ f(3), then ln(a) < -b - 1 -/
theorem function_minimum_implies_inequality (a b : ℝ) (ha : a > 0) :
  (∀ x > 0, -3 * log x + a * x^2 + b * x ≥ -3 * log 3 + 9 * a + 3 * b) →
  log a < -b - 1 := by
  sorry

end function_minimum_implies_inequality_l1176_117684


namespace arithmetic_sequence_fourth_term_l1176_117605

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence where a₃ + a₅ = 16, a₄ = 8 -/
theorem arithmetic_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 3 + a 5 = 16) :
  a 4 = 8 := by
sorry

end arithmetic_sequence_fourth_term_l1176_117605


namespace sum_c_plus_d_l1176_117679

theorem sum_c_plus_d (a b c d : ℤ) 
  (h1 : a + b = 16) 
  (h2 : b + c = 9) 
  (h3 : a + d = 10) : 
  c + d = 3 := by
  sorry

end sum_c_plus_d_l1176_117679


namespace max_pieces_cut_l1176_117683

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- The plywood sheet -/
def plywood : Rectangle := { length := 22, width := 15 }

/-- The piece to be cut -/
def piece : Rectangle := { length := 3, width := 5 }

/-- Theorem stating the maximum number of pieces that can be cut -/
theorem max_pieces_cut : 
  (area plywood) / (area piece) = 22 := by sorry

end max_pieces_cut_l1176_117683


namespace larger_square_area_l1176_117624

theorem larger_square_area (small_side : ℝ) (small_triangles : ℕ) (large_triangles : ℕ) :
  small_side = 12 →
  small_triangles = 16 →
  large_triangles = 18 →
  (large_triangles : ℝ) / (small_triangles : ℝ) * (small_side ^ 2) = 162 := by
  sorry

end larger_square_area_l1176_117624


namespace skip_speed_relation_l1176_117689

theorem skip_speed_relation (bruce_speed : ℝ) : 
  let tony_speed := 2 * bruce_speed
  let brandon_speed := (1/3) * tony_speed
  let colin_speed := 6 * brandon_speed
  colin_speed = 4 → bruce_speed = 1 := by
  sorry

end skip_speed_relation_l1176_117689


namespace midpoint_coordinate_sum_l1176_117677

/-- Given that P(5, 9) is the midpoint of segment CD and C has coordinates (11, 5),
    prove that the sum of the coordinates of D is 12. -/
theorem midpoint_coordinate_sum (C D : ℝ × ℝ) :
  C = (11, 5) →
  (5, 9) = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) →
  D.1 + D.2 = 12 := by
  sorry

end midpoint_coordinate_sum_l1176_117677


namespace wheat_D_tallest_and_neatest_l1176_117603

-- Define the wheat types
inductive WheatType
| A
| B
| C
| D

-- Define a function for average height
def averageHeight (t : WheatType) : ℝ :=
  match t with
  | .A => 13
  | .B => 15
  | .C => 13
  | .D => 15

-- Define a function for variance
def variance (t : WheatType) : ℝ :=
  match t with
  | .A => 3.6
  | .B => 6.3
  | .C => 6.3
  | .D => 3.6

-- Define a predicate for tallness
def isTaller (t1 t2 : WheatType) : Prop :=
  averageHeight t1 > averageHeight t2

-- Define a predicate for neatness (lower variance means neater)
def isNeater (t1 t2 : WheatType) : Prop :=
  variance t1 < variance t2

-- Theorem statement
theorem wheat_D_tallest_and_neatest :
  ∀ t : WheatType, t ≠ WheatType.D →
    (isTaller WheatType.D t ∨ averageHeight WheatType.D = averageHeight t) ∧
    (isNeater WheatType.D t ∨ variance WheatType.D = variance t) :=
by sorry

end wheat_D_tallest_and_neatest_l1176_117603


namespace reward_system_l1176_117695

/-- The number of bowls a customer needs to buy to get rewarded with two bowls -/
def bowls_for_reward : ℕ := sorry

theorem reward_system (total_bowls : ℕ) (customers : ℕ) (buying_customers : ℕ) 
  (bowls_per_customer : ℕ) (remaining_bowls : ℕ) :
  total_bowls = 70 →
  customers = 20 →
  buying_customers = customers / 2 →
  bowls_per_customer = 20 →
  remaining_bowls = 30 →
  bowls_for_reward = 10 := by sorry

end reward_system_l1176_117695


namespace age_difference_l1176_117653

/-- Given four persons a, b, c, and d with ages A, B, C, and D respectively,
    where the total age of a and b is 11 years more than the total age of b and c,
    prove that c is 11 + D years younger than the sum of the ages of a and d. -/
theorem age_difference (A B C D : ℤ) (h : A + B = B + C + 11) :
  C - (A + D) = -11 - D := by
  sorry

end age_difference_l1176_117653


namespace smallest_a_inequality_l1176_117612

theorem smallest_a_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 1) :
  ∀ a : ℝ, a ≥ 2/9 →
    a * (x^2 + y^2 + z^2) + x * y * z ≥ a / 3 + 1 / 27 ∧
    ∀ b : ℝ, b < 2/9 →
      ∃ x' y' z' : ℝ, x' ≥ 0 ∧ y' ≥ 0 ∧ z' ≥ 0 ∧ x' + y' + z' = 1 ∧
        b * (x'^2 + y'^2 + z'^2) + x' * y' * z' < b / 3 + 1 / 27 :=
sorry

end smallest_a_inequality_l1176_117612


namespace geometric_sum_15_l1176_117619

def geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ r : ℤ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sum_15 (a : ℕ → ℤ) :
  geometric_sequence a →
  a 1 = 1 →
  (∀ n : ℕ, a (n + 1) = a n * (-2)) →
  a 1 + |a 2| + a 3 + |a 4| = 15 :=
by sorry

end geometric_sum_15_l1176_117619


namespace complex_fraction_evaluation_l1176_117622

theorem complex_fraction_evaluation :
  (Complex.I : ℂ) / (12 + Complex.I) = (1 : ℂ) / 145 + (12 : ℂ) / 145 * Complex.I :=
by sorry

end complex_fraction_evaluation_l1176_117622


namespace inequality_and_equality_condition_l1176_117697

theorem inequality_and_equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b)^3 / (a^2 * b) ≥ 27/4 ∧ ((a + b)^3 / (a^2 * b) = 27/4 ↔ a = 2*b) := by
  sorry

end inequality_and_equality_condition_l1176_117697


namespace sqrt_calculation_l1176_117667

theorem sqrt_calculation : Real.sqrt 24 * Real.sqrt (1/6) - (-Real.sqrt 7)^2 = -5 := by
  sorry

end sqrt_calculation_l1176_117667


namespace three_numbers_average_l1176_117652

theorem three_numbers_average (x y z : ℝ) : 
  y = 2 * x →
  z = 4 * y →
  y = 90 →
  (x + y + z) / 3 = 165 := by
sorry

end three_numbers_average_l1176_117652


namespace smallest_number_inequality_l1176_117670

theorem smallest_number_inequality (x y z m : ℝ) 
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) 
  (hm : m = min (min (min 1 (x^9)) (y^9)) (z^7)) : 
  m ≤ x * y^2 * z^3 := by
  sorry

end smallest_number_inequality_l1176_117670


namespace paper_sheet_width_l1176_117646

theorem paper_sheet_width (sheet1_length sheet1_width sheet2_length : ℝ)
  (h1 : sheet1_length = 11)
  (h2 : sheet1_width = 17)
  (h3 : sheet2_length = 11)
  (h4 : 2 * sheet1_length * sheet1_width = 2 * sheet2_length * sheet2_width + 100) :
  sheet2_width = 12.45 := by
  sorry

end paper_sheet_width_l1176_117646


namespace xy_is_zero_l1176_117623

theorem xy_is_zero (x y : ℝ) (h1 : x - y = 3) (h2 : x^3 - y^3 = 27) : x * y = 0 := by
  sorry

end xy_is_zero_l1176_117623


namespace mean_proportional_segment_l1176_117610

theorem mean_proportional_segment (a b c : ℝ) : 
  a = 3 → b = 27 → c^2 = a * b → c = 9 := by sorry

end mean_proportional_segment_l1176_117610


namespace parallel_transitive_l1176_117636

-- Define a type for lines
def Line : Type := ℝ → ℝ → Prop

-- Define parallel relation
def parallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem parallel_transitive (l1 l2 l3 : Line) :
  parallel l1 l2 → parallel l2 l3 → parallel l1 l3 := by
  sorry

end parallel_transitive_l1176_117636
