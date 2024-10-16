import Mathlib

namespace NUMINAMATH_CALUDE_find_n_value_l3691_369198

theorem find_n_value (n : ℕ) : (1/5 : ℝ)^n * (1/4 : ℝ)^18 = 1/(2*(10 : ℝ)^35) → n = 35 := by
  sorry

end NUMINAMATH_CALUDE_find_n_value_l3691_369198


namespace NUMINAMATH_CALUDE_roots_of_g_l3691_369187

/-- Given that 2 is a root of f(x) = ax + b, prove that the roots of g(x) = bx² - ax are 0 and -1/2 --/
theorem roots_of_g (a b : ℝ) (h : a * 2 + b = 0) :
  ∃ (x y : ℝ), x = 0 ∧ y = -1/2 ∧ ∀ z : ℝ, b * z^2 - a * z = 0 ↔ z = x ∨ z = y :=
by sorry

end NUMINAMATH_CALUDE_roots_of_g_l3691_369187


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_1800_l3691_369139

def largest_perfect_square_factor (n : ℕ) : ℕ :=
  sorry

theorem largest_perfect_square_factor_1800 :
  largest_perfect_square_factor 1800 = 900 := by
  sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_1800_l3691_369139


namespace NUMINAMATH_CALUDE_equal_sum_sequence_a8_l3691_369152

/-- An equal sum sequence is a sequence where the sum of each term and its next term is constant. --/
def EqualSumSequence (a : ℕ → ℝ) :=
  ∃ k : ℝ, ∀ n : ℕ, a n + a (n + 1) = k

/-- The common sum of an equal sum sequence. --/
def CommonSum (a : ℕ → ℝ) (k : ℝ) :=
  ∀ n : ℕ, a n + a (n + 1) = k

theorem equal_sum_sequence_a8 (a : ℕ → ℝ) (h1 : EqualSumSequence a) (h2 : a 1 = 2) (h3 : CommonSum a 5) :
  a 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_equal_sum_sequence_a8_l3691_369152


namespace NUMINAMATH_CALUDE_nut_weight_l3691_369162

/-- A proof that determines the weight of a nut attached to a scale -/
theorem nut_weight (wL wS : ℝ) (h1 : wL + 20 = 300) (h2 : wS + 20 = 200) (h3 : wL + wS + 20 = 480) : 20 = 20 := by
  sorry

end NUMINAMATH_CALUDE_nut_weight_l3691_369162


namespace NUMINAMATH_CALUDE_border_mass_of_28_coin_triangle_l3691_369163

/-- Represents a triangular arrangement of coins -/
structure CoinTriangle where
  total_coins : ℕ
  border_coins : ℕ
  trio_mass : ℝ

/-- The mass of all border coins in a CoinTriangle -/
def border_mass (ct : CoinTriangle) : ℝ := sorry

/-- Theorem stating the mass of border coins in the specific arrangement -/
theorem border_mass_of_28_coin_triangle (ct : CoinTriangle) 
  (h1 : ct.total_coins = 28)
  (h2 : ct.border_coins = 18)
  (h3 : ct.trio_mass = 10) :
  border_mass ct = 60 := by sorry

end NUMINAMATH_CALUDE_border_mass_of_28_coin_triangle_l3691_369163


namespace NUMINAMATH_CALUDE_line_bisected_by_M_l3691_369105

-- Define the lines and point
def l₁ (x y : ℝ) : Prop := x - 3 * y + 10 = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y - 8 = 0
def M : ℝ × ℝ := (0, 1)

-- Define the line we want to prove
def target_line (x y : ℝ) : Prop := y = -1/3 * x + 1

-- Theorem statement
theorem line_bisected_by_M :
  ∃ (A B : ℝ × ℝ),
    l₁ A.1 A.2 ∧
    l₂ B.1 B.2 ∧
    target_line A.1 A.2 ∧
    target_line B.1 B.2 ∧
    target_line M.1 M.2 ∧
    M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) :=
  sorry


end NUMINAMATH_CALUDE_line_bisected_by_M_l3691_369105


namespace NUMINAMATH_CALUDE_tian_ji_win_probability_l3691_369170

/-- Represents the tiers of horses -/
inductive Tier
| Top
| Middle
| Bottom

/-- Represents a horse with its owner and tier -/
structure Horse :=
  (owner : String)
  (tier : Tier)

/-- Determines if one horse is better than another -/
def isBetter (h1 h2 : Horse) : Prop := sorry

/-- The set of all horses in the competition -/
def allHorses : Finset Horse := sorry

/-- The set of Tian Ji's horses -/
def tianJiHorses : Finset Horse := sorry

/-- The set of King Qi's horses -/
def kingQiHorses : Finset Horse := sorry

/-- Axioms representing the given conditions -/
axiom horse_count : (tianJiHorses.card = 3) ∧ (kingQiHorses.card = 3)
axiom tian_ji_top_vs_qi_middle : 
  ∃ (ht hm : Horse), ht ∈ tianJiHorses ∧ hm ∈ kingQiHorses ∧ 
  ht.tier = Tier.Top ∧ hm.tier = Tier.Middle ∧ isBetter ht hm
axiom tian_ji_top_vs_qi_top : 
  ∃ (ht1 ht2 : Horse), ht1 ∈ tianJiHorses ∧ ht2 ∈ kingQiHorses ∧ 
  ht1.tier = Tier.Top ∧ ht2.tier = Tier.Top ∧ isBetter ht2 ht1
axiom tian_ji_middle_vs_qi_bottom : 
  ∃ (hm hb : Horse), hm ∈ tianJiHorses ∧ hb ∈ kingQiHorses ∧ 
  hm.tier = Tier.Middle ∧ hb.tier = Tier.Bottom ∧ isBetter hm hb
axiom tian_ji_middle_vs_qi_middle : 
  ∃ (hm1 hm2 : Horse), hm1 ∈ tianJiHorses ∧ hm2 ∈ kingQiHorses ∧ 
  hm1.tier = Tier.Middle ∧ hm2.tier = Tier.Middle ∧ isBetter hm2 hm1
axiom tian_ji_bottom_vs_qi_bottom : 
  ∃ (hb1 hb2 : Horse), hb1 ∈ tianJiHorses ∧ hb2 ∈ kingQiHorses ∧ 
  hb1.tier = Tier.Bottom ∧ hb2.tier = Tier.Bottom ∧ isBetter hb2 hb1

/-- The probability of Tian Ji's horse winning in a random matchup -/
def tianJiWinProbability : ℚ := sorry

/-- Main theorem: The probability of Tian Ji's horse winning is 1/3 -/
theorem tian_ji_win_probability : tianJiWinProbability = 1/3 := by sorry

end NUMINAMATH_CALUDE_tian_ji_win_probability_l3691_369170


namespace NUMINAMATH_CALUDE_point_coordinates_l3691_369191

theorem point_coordinates (M N P : ℝ × ℝ) : 
  M = (3, -2) → 
  N = (-5, -1) → 
  P - M = (1/2 : ℝ) • (N - M) → 
  P = (-1, -3/2) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l3691_369191


namespace NUMINAMATH_CALUDE_proposition_implication_l3691_369169

theorem proposition_implication (P : ℕ → Prop) 
  (h1 : ∀ k : ℕ, k > 0 → (P k → P (k + 1)))
  (h2 : ¬ P 4) : 
  ¬ P 3 := by
  sorry

end NUMINAMATH_CALUDE_proposition_implication_l3691_369169


namespace NUMINAMATH_CALUDE_solution_characterization_l3691_369134

theorem solution_characterization (x y z : ℝ) 
  (h1 : x + y + z = 1/x + 1/y + 1/z) 
  (h2 : x^2 + y^2 + z^2 = 1/x^2 + 1/y^2 + 1/z^2) :
  ∃ (e : ℝ) (t : ℝ), e = 1 ∨ e = -1 ∧ t ≠ 0 ∧ 
    ((x = e ∧ y = t ∧ z = 1/t) ∨ 
     (x = e ∧ y = 1/t ∧ z = t) ∨ 
     (x = t ∧ y = e ∧ z = 1/t) ∨ 
     (x = t ∧ y = 1/t ∧ z = e) ∨ 
     (x = 1/t ∧ y = e ∧ z = t) ∨ 
     (x = 1/t ∧ y = t ∧ z = e)) :=
by sorry

end NUMINAMATH_CALUDE_solution_characterization_l3691_369134


namespace NUMINAMATH_CALUDE_scott_cake_sales_l3691_369175

theorem scott_cake_sales (smoothie_price : ℕ) (cake_price : ℕ) (smoothies_sold : ℕ) (total_revenue : ℕ) :
  smoothie_price = 3 →
  cake_price = 2 →
  smoothies_sold = 40 →
  total_revenue = 156 →
  ∃ (cakes_sold : ℕ), smoothie_price * smoothies_sold + cake_price * cakes_sold = total_revenue ∧ cakes_sold = 18 := by
  sorry

end NUMINAMATH_CALUDE_scott_cake_sales_l3691_369175


namespace NUMINAMATH_CALUDE_initial_discount_percentage_l3691_369149

/-- Given a dress with original price d and an initial discount percentage x,
    prove that x = 65 when a staff member pays 0.14d after an additional 60% discount. -/
theorem initial_discount_percentage (d : ℝ) (x : ℝ) (h : d > 0) :
  0.40 * (1 - x / 100) * d = 0.14 * d → x = 65 := by
  sorry

end NUMINAMATH_CALUDE_initial_discount_percentage_l3691_369149


namespace NUMINAMATH_CALUDE_expression_evaluation_l3691_369123

theorem expression_evaluation :
  let x : ℚ := 2
  let y : ℚ := -1/3
  (4*x^2 - 2*x*y + y^2) - 3*(x^2 - x*y + 5*y^2) = 16/9 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3691_369123


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l3691_369115

theorem complex_fraction_evaluation (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + a*b + b^2 = 0) : 
  (a^9 + b^9) / (a + b)^9 = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l3691_369115


namespace NUMINAMATH_CALUDE_area_between_specific_lines_l3691_369110

/-- Line passing through two points -/
structure Line where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- Calculate the area between two lines from x = 0 to x = 5 -/
def areaBetweenLines (l1 l2 : Line) : ℝ :=
  sorry

/-- The main theorem -/
theorem area_between_specific_lines :
  let line1 : Line := ⟨0, 3, 6, 0⟩
  let line2 : Line := ⟨0, 5, 10, 2⟩
  areaBetweenLines line1 line2 = 10 := by sorry

end NUMINAMATH_CALUDE_area_between_specific_lines_l3691_369110


namespace NUMINAMATH_CALUDE_ice_cream_distribution_l3691_369107

theorem ice_cream_distribution (total_sandwiches : ℕ) (num_nieces : ℕ) 
  (h1 : total_sandwiches = 143)
  (h2 : num_nieces = 11) :
  total_sandwiches / num_nieces = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_ice_cream_distribution_l3691_369107


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l3691_369174

theorem complex_expression_evaluation :
  ∀ (a b : ℂ),
  a = 5 - 3*I →
  b = 2 + 4*I →
  3*a - 4*b = 7 - 25*I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l3691_369174


namespace NUMINAMATH_CALUDE_arrangement_count_is_correct_l3691_369197

/-- The number of ways to arrange 8 balls in a row, with 5 red balls and 3 white balls,
    such that exactly three consecutive balls are painted red -/
def arrangementCount : ℕ := 24

/-- The total number of balls -/
def totalBalls : ℕ := 8

/-- The number of red balls -/
def redBalls : ℕ := 5

/-- The number of white balls -/
def whiteBalls : ℕ := 3

/-- The number of consecutive red balls required -/
def consecutiveRedBalls : ℕ := 3

theorem arrangement_count_is_correct :
  arrangementCount = 24 ∧
  totalBalls = 8 ∧
  redBalls = 5 ∧
  whiteBalls = 3 ∧
  consecutiveRedBalls = 3 ∧
  redBalls + whiteBalls = totalBalls ∧
  arrangementCount = (whiteBalls + 1) * (redBalls - consecutiveRedBalls + 1) :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_is_correct_l3691_369197


namespace NUMINAMATH_CALUDE_probability_two_non_defective_pens_l3691_369172

/-- The probability of selecting two non-defective pens from a box of pens -/
theorem probability_two_non_defective_pens 
  (total_pens : ℕ) 
  (defective_pens : ℕ) 
  (h1 : total_pens = 12) 
  (h2 : defective_pens = 4) :
  (total_pens - defective_pens : ℚ) / total_pens * 
  (total_pens - defective_pens - 1) / (total_pens - 1) = 14 / 33 := by
  sorry

#check probability_two_non_defective_pens

end NUMINAMATH_CALUDE_probability_two_non_defective_pens_l3691_369172


namespace NUMINAMATH_CALUDE_pharmacy_purchase_cost_bob_pharmacy_purchase_cost_l3691_369125

/-- Calculates the total cost of a pharmacy purchase including sales tax -/
theorem pharmacy_purchase_cost (nose_spray_cost : ℚ) (nose_spray_count : ℕ) 
  (nose_spray_discount : ℚ) (cough_syrup_cost : ℚ) (cough_syrup_count : ℕ) 
  (cough_syrup_discount : ℚ) (ibuprofen_cost : ℚ) (ibuprofen_count : ℕ) 
  (sales_tax_rate : ℚ) : ℚ :=
  let nose_spray_total := (nose_spray_cost * ↑(nose_spray_count / 2)) * (1 - nose_spray_discount)
  let cough_syrup_total := (cough_syrup_cost * ↑cough_syrup_count) * (1 - cough_syrup_discount)
  let ibuprofen_total := ibuprofen_cost * ↑ibuprofen_count
  let subtotal := nose_spray_total + cough_syrup_total + ibuprofen_total
  let total_with_tax := subtotal * (1 + sales_tax_rate)
  total_with_tax

/-- The total cost of Bob's pharmacy purchase, rounded to the nearest cent, is $56.38 -/
theorem bob_pharmacy_purchase_cost : 
  ⌊pharmacy_purchase_cost 3 10 (1/5) 7 4 (1/10) 5 3 (2/25) * 100⌋ / 100 = 56381 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_pharmacy_purchase_cost_bob_pharmacy_purchase_cost_l3691_369125


namespace NUMINAMATH_CALUDE_range_of_a_l3691_369121

theorem range_of_a (a : ℝ) : 
  (a > 0) → 
  (∀ x : ℝ, ((x - 3*a) * (x - a) < 0) → 
    ¬(x^2 - 3*x ≤ 0 ∧ x^2 - x - 2 > 0)) ∧ 
  (∃ x : ℝ, ¬(x^2 - 3*x ≤ 0 ∧ x^2 - x - 2 > 0) ∧ 
    ¬((x - 3*a) * (x - a) < 0)) ↔ 
  (0 < a ∧ a ≤ 2/3) ∨ a ≥ 3 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3691_369121


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l3691_369147

-- Define the number of white and black balls
def num_white_balls : ℕ := 1
def num_black_balls : ℕ := 2

-- Define the total number of balls
def total_balls : ℕ := num_white_balls + num_black_balls

-- Define the probability of drawing a white ball
def prob_white_ball : ℚ := num_white_balls / total_balls

-- Theorem statement
theorem probability_of_white_ball :
  prob_white_ball = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l3691_369147


namespace NUMINAMATH_CALUDE_profit_share_difference_l3691_369132

theorem profit_share_difference (a b c : ℕ) (b_profit : ℕ) : 
  a = 8000 → b = 10000 → c = 12000 → b_profit = 1400 →
  ∃ (a_profit c_profit : ℕ), 
    a_profit * b = b_profit * a ∧ 
    c_profit * b = b_profit * c ∧ 
    c_profit - a_profit = 560 := by
  sorry

end NUMINAMATH_CALUDE_profit_share_difference_l3691_369132


namespace NUMINAMATH_CALUDE_basic_computer_price_l3691_369117

/-- Given the price of a basic computer and printer, prove the price of the basic computer. -/
theorem basic_computer_price (basic_price printer_price enhanced_price : ℝ) : 
  basic_price + printer_price = 2500 →
  enhanced_price = basic_price + 500 →
  enhanced_price + printer_price = 6 * printer_price →
  basic_price = 2000 := by
  sorry

end NUMINAMATH_CALUDE_basic_computer_price_l3691_369117


namespace NUMINAMATH_CALUDE_factorization_theorem_l3691_369186

theorem factorization_theorem (m n : ℝ) : m^3*n - m*n = m*n*(m-1)*(m+1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_theorem_l3691_369186


namespace NUMINAMATH_CALUDE_muffin_cost_savings_l3691_369148

/-- Represents the cost savings when choosing raspberries over blueberries for muffins -/
def cost_savings (num_batches : ℕ) (ounces_per_batch : ℕ) 
  (blueberry_price : ℚ) (blueberry_ounces : ℕ) 
  (raspberry_price : ℚ) (raspberry_ounces : ℕ) : ℚ :=
  let total_ounces := num_batches * ounces_per_batch
  let blueberry_cartons := (total_ounces + blueberry_ounces - 1) / blueberry_ounces
  let raspberry_cartons := (total_ounces + raspberry_ounces - 1) / raspberry_ounces
  blueberry_cartons * blueberry_price - raspberry_cartons * raspberry_price

/-- The cost savings when choosing raspberries over blueberries for 4 batches of muffins -/
theorem muffin_cost_savings : 
  cost_savings 4 12 (5 / 1) 6 (3 / 1) 8 = 22 := by
  sorry

end NUMINAMATH_CALUDE_muffin_cost_savings_l3691_369148


namespace NUMINAMATH_CALUDE_pool_emptying_time_l3691_369167

/-- Given three pumps with individual emptying rates, calculates the time taken to empty a pool when all pumps work together -/
theorem pool_emptying_time 
  (rate_a rate_b rate_c : ℚ) 
  (ha : rate_a = 1 / 6) 
  (hb : rate_b = 1 / 9) 
  (hc : rate_c = 1 / 12) : 
  (1 / (rate_a + rate_b + rate_c)) = 36 / 13 := by
  sorry

#eval (36 : ℚ) / 13 * 60 -- To show the result in minutes

end NUMINAMATH_CALUDE_pool_emptying_time_l3691_369167


namespace NUMINAMATH_CALUDE_children_born_in_current_marriage_l3691_369164

/-- Represents the number of children in a blended family scenario -/
structure BlendedFamily where
  x : ℕ  -- children from father's previous marriage
  y : ℕ  -- children from mother's previous marriage
  z : ℕ  -- children born in current marriage
  total_children : x + y + z = 12
  father_bio_children : x + z = 9
  mother_bio_children : y + z = 9

/-- Theorem stating that in this blended family scenario, 6 children were born in the current marriage -/
theorem children_born_in_current_marriage (family : BlendedFamily) : family.z = 6 := by
  sorry

#check children_born_in_current_marriage

end NUMINAMATH_CALUDE_children_born_in_current_marriage_l3691_369164


namespace NUMINAMATH_CALUDE_reflection_over_x_axis_of_P_l3691_369143

/-- Reflects a point over the x-axis -/
def reflect_over_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The original point -/
def P : ℝ × ℝ := (2, -3)

theorem reflection_over_x_axis_of_P :
  reflect_over_x_axis P = (2, 3) := by sorry

end NUMINAMATH_CALUDE_reflection_over_x_axis_of_P_l3691_369143


namespace NUMINAMATH_CALUDE_inequality_proof_l3691_369171

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/b + 1/c ≥ a + b + c) : a + b + c ≥ 3*a*b*c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3691_369171


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l3691_369168

theorem square_sum_reciprocal (x : ℝ) (h : x + 1/x = 8) : x^2 + 1/x^2 = 62 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l3691_369168


namespace NUMINAMATH_CALUDE_max_value_g_and_range_of_a_l3691_369159

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (a * x)

def g (a : ℝ) (x : ℝ) : ℝ := x^2 * f a x

def h (a : ℝ) (x : ℝ) : ℝ := x^2 / f a x - 1

theorem max_value_g_and_range_of_a :
  (∀ x > 0, g (-2) x ≤ Real.exp (-2)) ∧
  (∀ a : ℝ, (∃ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 16 ∧ h a x₁ = 0 ∧ h a x₂ = 0) →
    1/2 * Real.log 2 < a ∧ a < 2 / Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_max_value_g_and_range_of_a_l3691_369159


namespace NUMINAMATH_CALUDE_train_platform_crossing_time_l3691_369130

/-- Given a train of length 1200 m that crosses a tree in 120 sec,
    prove that it takes 190 sec to pass a platform of length 700 m. -/
theorem train_platform_crossing_time :
  ∀ (train_length platform_length tree_crossing_time : ℝ),
    train_length = 1200 →
    platform_length = 700 →
    tree_crossing_time = 120 →
    (train_length + platform_length) / (train_length / tree_crossing_time) = 190 :=
by sorry

end NUMINAMATH_CALUDE_train_platform_crossing_time_l3691_369130


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l3691_369154

theorem least_positive_integer_with_remainders : ∃ n : ℕ, 
  (n > 0) ∧ 
  (n % 4 = 1) ∧ 
  (n % 5 = 2) ∧ 
  (n % 6 = 3) ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 4 = 1 ∧ m % 5 = 2 ∧ m % 6 = 3 → m ≥ n) ∧
  n = 57 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l3691_369154


namespace NUMINAMATH_CALUDE_marks_birth_year_l3691_369135

theorem marks_birth_year (current_year : ℕ) (janice_age : ℕ) 
  (h1 : current_year = 2021)
  (h2 : janice_age = 21)
  (h3 : ∃ (graham_age : ℕ), graham_age = 2 * janice_age)
  (h4 : ∃ (mark_age : ℕ), mark_age = graham_age + 3) :
  ∃ (birth_year : ℕ), birth_year = current_year - (2 * janice_age + 3) := by
  sorry

end NUMINAMATH_CALUDE_marks_birth_year_l3691_369135


namespace NUMINAMATH_CALUDE_watch_payment_in_dimes_l3691_369103

/-- The number of dimes in one dollar -/
def dimes_per_dollar : ℕ := 10

/-- The cost of the watch in dollars -/
def watch_cost : ℕ := 5

/-- Theorem: If a watch costs 5 dollars and is paid for entirely in dimes, 
    the number of dimes used is 50. -/
theorem watch_payment_in_dimes : 
  watch_cost * dimes_per_dollar = 50 := by sorry

end NUMINAMATH_CALUDE_watch_payment_in_dimes_l3691_369103


namespace NUMINAMATH_CALUDE_min_blocks_for_specific_wall_l3691_369137

/-- Represents the dimensions of the wall --/
structure WallDimensions where
  length : ℕ
  height : ℕ

/-- Represents the dimensions of a block --/
structure BlockDimensions where
  height : ℕ
  length : ℕ

/-- Calculates the minimum number of blocks required to build the wall --/
def minBlocksRequired (wall : WallDimensions) (block1 : BlockDimensions) (block2 : BlockDimensions) : ℕ :=
  sorry

/-- Theorem stating the minimum number of blocks required for the specific wall --/
theorem min_blocks_for_specific_wall :
  let wall := WallDimensions.mk 120 10
  let block1 := BlockDimensions.mk 1 3
  let block2 := BlockDimensions.mk 1 1
  minBlocksRequired wall block1 block2 = 415 :=
by sorry

end NUMINAMATH_CALUDE_min_blocks_for_specific_wall_l3691_369137


namespace NUMINAMATH_CALUDE_seating_arrangements_l3691_369193

theorem seating_arrangements (n : ℕ) (h : n = 6) : Nat.factorial n = 720 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_l3691_369193


namespace NUMINAMATH_CALUDE_complex_equality_condition_l3691_369192

theorem complex_equality_condition :
  ∃ (x y : ℂ), x + y * Complex.I = 1 + Complex.I ∧ (x ≠ 1 ∨ y ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_complex_equality_condition_l3691_369192


namespace NUMINAMATH_CALUDE_polar_midpoint_specific_case_l3691_369140

/-- The midpoint of a line segment in polar coordinates --/
def polar_midpoint (r₁ : ℝ) (θ₁ : ℝ) (r₂ : ℝ) (θ₂ : ℝ) : ℝ × ℝ :=
  sorry

/-- Theorem: The midpoint of a line segment with endpoints (10, π/3) and (10, 2π/3) in polar coordinates is (5√3, π/2) --/
theorem polar_midpoint_specific_case :
  let (r, θ) := polar_midpoint 10 (π/3) 10 (2*π/3)
  r = 5 * Real.sqrt 3 ∧ θ = π/2 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2*π :=
by sorry

end NUMINAMATH_CALUDE_polar_midpoint_specific_case_l3691_369140


namespace NUMINAMATH_CALUDE_art_show_ratio_l3691_369160

theorem art_show_ratio (total_painted : ℚ) (sold : ℚ) 
  (h1 : total_painted = 180.5)
  (h2 : sold = 76.3) :
  (total_painted - sold) / sold = 1042 / 763 := by
  sorry

end NUMINAMATH_CALUDE_art_show_ratio_l3691_369160


namespace NUMINAMATH_CALUDE_pens_for_friends_l3691_369141

/-- The number of friends who will receive pens from Kendra and Tony -/
def friends_receiving_pens (kendra_packs tony_packs pens_per_pack pens_kept_each : ℕ) : ℕ :=
  (kendra_packs + tony_packs) * pens_per_pack - 2 * pens_kept_each

/-- Theorem stating that Kendra and Tony will give pens to 14 friends -/
theorem pens_for_friends : 
  friends_receiving_pens 4 2 3 2 = 14 := by
  sorry

#eval friends_receiving_pens 4 2 3 2

end NUMINAMATH_CALUDE_pens_for_friends_l3691_369141


namespace NUMINAMATH_CALUDE_lcm_of_5_8_10_27_l3691_369161

theorem lcm_of_5_8_10_27 : Nat.lcm 5 (Nat.lcm 8 (Nat.lcm 10 27)) = 1080 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_5_8_10_27_l3691_369161


namespace NUMINAMATH_CALUDE_average_age_combined_l3691_369184

theorem average_age_combined (num_students : ℕ) (num_guardians : ℕ) 
  (avg_age_students : ℚ) (avg_age_guardians : ℚ) :
  num_students = 40 →
  num_guardians = 60 →
  avg_age_students = 10 →
  avg_age_guardians = 35 →
  (num_students * avg_age_students + num_guardians * avg_age_guardians) / (num_students + num_guardians) = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_average_age_combined_l3691_369184


namespace NUMINAMATH_CALUDE_cars_meeting_time_l3691_369106

/-- Two cars meeting on a highway -/
theorem cars_meeting_time (highway_length : ℝ) (speed1 speed2 : ℝ) (h1 : highway_length = 105)
    (h2 : speed1 = 15) (h3 : speed2 = 20) : 
  (highway_length / (speed1 + speed2)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cars_meeting_time_l3691_369106


namespace NUMINAMATH_CALUDE_unique_solution_system_l3691_369178

theorem unique_solution_system (x y : ℝ) :
  (y = (x + 2)^2 ∧ x * y + y = 2) ↔ (x = 2^(1/3) - 2 ∧ y = 2^(2/3)) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3691_369178


namespace NUMINAMATH_CALUDE_school_sections_l3691_369173

theorem school_sections (boys girls : ℕ) (h1 : boys = 408) (h2 : girls = 264) :
  let gcd := Nat.gcd boys girls
  let boys_sections := boys / gcd
  let girls_sections := girls / gcd
  boys_sections + girls_sections = 28 := by
sorry

end NUMINAMATH_CALUDE_school_sections_l3691_369173


namespace NUMINAMATH_CALUDE_sphere_radius_change_factor_l3691_369190

theorem sphere_radius_change_factor (initial_area new_area : ℝ) 
  (h1 : initial_area = 2464)
  (h2 : new_area = 9856) : 
  let factor := (new_area / initial_area).sqrt
  factor = 2 := by sorry

end NUMINAMATH_CALUDE_sphere_radius_change_factor_l3691_369190


namespace NUMINAMATH_CALUDE_apple_grape_equivalence_l3691_369150

/-- Given that 3/4 of 12 apples are worth 9 grapes, 
    prove that 1/2 of 6 apples are worth 3 grapes -/
theorem apple_grape_equivalence : 
  (3/4 : ℚ) * 12 * (1 : ℚ) = 9 * (1 : ℚ) → 
  (1/2 : ℚ) * 6 * (1 : ℚ) = 3 * (1 : ℚ) :=
by
  sorry

end NUMINAMATH_CALUDE_apple_grape_equivalence_l3691_369150


namespace NUMINAMATH_CALUDE_inheritance_tax_problem_l3691_369128

theorem inheritance_tax_problem (x : ℝ) : 
  0.25 * x + 0.12 * (0.75 * x) = 13600 → x = 40000 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_tax_problem_l3691_369128


namespace NUMINAMATH_CALUDE_divisor_ratio_of_M_l3691_369195

def M : ℕ := 126 * 36 * 187

/-- Sum of odd divisors of a natural number -/
def sum_odd_divisors (n : ℕ) : ℕ := sorry

/-- Sum of even divisors of a natural number -/
def sum_even_divisors (n : ℕ) : ℕ := sorry

/-- The ratio of sum of even divisors to sum of odd divisors -/
def divisor_ratio (n : ℕ) : ℚ :=
  (sum_even_divisors n : ℚ) / (sum_odd_divisors n : ℚ)

theorem divisor_ratio_of_M :
  divisor_ratio M = 14 := by sorry

end NUMINAMATH_CALUDE_divisor_ratio_of_M_l3691_369195


namespace NUMINAMATH_CALUDE_f_inequality_solution_comparison_theorem_l3691_369199

def f (x : ℝ) : ℝ := -abs x - abs (x + 2)

theorem f_inequality_solution (x : ℝ) : f x < -4 ↔ x < -3 ∨ x > 1 := by sorry

theorem comparison_theorem (x a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = Real.sqrt 5) :
  a^2 + b^2/4 ≥ f x + 3 := by sorry

end NUMINAMATH_CALUDE_f_inequality_solution_comparison_theorem_l3691_369199


namespace NUMINAMATH_CALUDE_staircase_cutting_count_l3691_369146

/-- Represents a staircase with a given number of steps -/
structure Staircase :=
  (steps : ℕ)

/-- Represents a cutting of the staircase into rectangles and a square -/
structure StaircaseCutting :=
  (staircase : Staircase)
  (rectangles : ℕ)
  (squares : ℕ)

/-- Counts the number of ways to cut a staircase -/
def countCuttings (s : Staircase) (r : ℕ) (sq : ℕ) : ℕ :=
  sorry

/-- The main theorem: there are 32 ways to cut a 6-step staircase into 5 rectangles and one square -/
theorem staircase_cutting_count :
  countCuttings (Staircase.mk 6) 5 1 = 32 := by
  sorry

end NUMINAMATH_CALUDE_staircase_cutting_count_l3691_369146


namespace NUMINAMATH_CALUDE_sum_of_factors_72_l3691_369118

theorem sum_of_factors_72 : (Finset.filter (· ∣ 72) (Finset.range 73)).sum id = 195 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_72_l3691_369118


namespace NUMINAMATH_CALUDE_animals_left_after_sale_l3691_369144

/-- Calculates the number of animals left in a pet store after a sale --/
theorem animals_left_after_sale (siamese_cats house_cats dogs birds cats_sold dogs_sold birds_sold : ℕ) :
  siamese_cats = 25 →
  house_cats = 55 →
  dogs = 30 →
  birds = 20 →
  cats_sold = 45 →
  dogs_sold = 25 →
  birds_sold = 10 →
  (siamese_cats + house_cats - cats_sold) + (dogs - dogs_sold) + (birds - birds_sold) = 50 := by
sorry

end NUMINAMATH_CALUDE_animals_left_after_sale_l3691_369144


namespace NUMINAMATH_CALUDE_solve_wardrobe_problem_l3691_369145

def wardrobe_problem (socks shoes tshirts new_socks : ℕ) : Prop :=
  ∃ pants : ℕ,
    let current_items := 2 * socks + 2 * shoes + tshirts + pants
    current_items + 2 * new_socks = 2 * current_items ∧
    pants = 5

theorem solve_wardrobe_problem :
  wardrobe_problem 20 5 10 35 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_wardrobe_problem_l3691_369145


namespace NUMINAMATH_CALUDE_optimal_orange_purchase_l3691_369165

-- Define the pricing options
def price_option_1 : ℕ × ℕ := (4, 15)  -- 4 oranges for 15 cents
def price_option_2 : ℕ × ℕ := (7, 25)  -- 7 oranges for 25 cents

-- Define the number of oranges to purchase
def total_oranges : ℕ := 28

-- Theorem statement
theorem optimal_orange_purchase :
  ∃ (n m : ℕ),
    n * price_option_1.1 + m * price_option_2.1 = total_oranges ∧
    n * price_option_1.2 + m * price_option_2.2 = 100 ∧
    (n * price_option_1.2 + m * price_option_2.2) / total_oranges = 25 / 7 :=
sorry

end NUMINAMATH_CALUDE_optimal_orange_purchase_l3691_369165


namespace NUMINAMATH_CALUDE_percentage_silver_cars_l3691_369131

/-- Calculates the percentage of silver cars after a new shipment -/
theorem percentage_silver_cars (initial_cars : ℕ) (initial_silver_percentage : ℚ) 
  (new_cars : ℕ) (new_non_silver_percentage : ℚ) :
  initial_cars = 40 →
  initial_silver_percentage = 1/5 →
  new_cars = 80 →
  new_non_silver_percentage = 1/2 →
  let initial_silver := initial_cars * initial_silver_percentage
  let new_silver := new_cars * (1 - new_non_silver_percentage)
  let total_silver := initial_silver + new_silver
  let total_cars := initial_cars + new_cars
  (total_silver / total_cars : ℚ) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_silver_cars_l3691_369131


namespace NUMINAMATH_CALUDE_series_sum_equals_half_l3691_369113

/-- The sum of the series defined by the nth term 1/((n+1)(n+2)) - 1/((n+2)(n+3)) for n ≥ 1 is equal to 1/2. -/
theorem series_sum_equals_half :
  (∑' n : ℕ, (1 : ℝ) / ((n + 1) * (n + 2)) - 1 / ((n + 2) * (n + 3))) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_half_l3691_369113


namespace NUMINAMATH_CALUDE_max_angle_MPN_at_x_equals_one_l3691_369153

structure Point where
  x : ℝ
  y : ℝ

def M : Point := ⟨-1, 2⟩
def N : Point := ⟨1, 4⟩

def angle_MPN (P : Point) : ℝ :=
  sorry  -- Definition of angle MPN

theorem max_angle_MPN_at_x_equals_one :
  ∃ (P : Point), P.y = 0 ∧ 
    (∀ (Q : Point), Q.y = 0 → angle_MPN P ≥ angle_MPN Q) ∧
    P.x = 1 := by
  sorry

#check max_angle_MPN_at_x_equals_one

end NUMINAMATH_CALUDE_max_angle_MPN_at_x_equals_one_l3691_369153


namespace NUMINAMATH_CALUDE_ship_round_trip_tickets_l3691_369182

/-- Given a ship with passengers, prove that 80% of passengers have round-trip tickets -/
theorem ship_round_trip_tickets (total_passengers : ℝ) 
  (h1 : total_passengers > 0) 
  (h2 : (0.4 : ℝ) * total_passengers = round_trip_with_car)
  (h3 : (0.5 : ℝ) * round_trip_tickets = round_trip_without_car)
  (h4 : round_trip_tickets = round_trip_with_car + round_trip_without_car) :
  round_trip_tickets = (0.8 : ℝ) * total_passengers :=
by
  sorry

end NUMINAMATH_CALUDE_ship_round_trip_tickets_l3691_369182


namespace NUMINAMATH_CALUDE_phillips_remaining_money_l3691_369102

/-- Calculates the remaining money after Phillip's shopping trip --/
def remaining_money (initial_amount : ℚ) 
  (orange_price : ℚ) (orange_quantity : ℚ)
  (apple_price : ℚ) (apple_quantity : ℚ)
  (candy_price : ℚ)
  (egg_price : ℚ) (egg_quantity : ℚ)
  (milk_price : ℚ)
  (sales_tax_rate : ℚ)
  (apple_discount_rate : ℚ) : ℚ :=
  sorry

/-- Theorem stating that Phillip's remaining money is $51.91 --/
theorem phillips_remaining_money :
  remaining_money 95 3 2 3.5 4 6 6 2 4 0.08 0.15 = 51.91 :=
  sorry

end NUMINAMATH_CALUDE_phillips_remaining_money_l3691_369102


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_expansion_l3691_369156

theorem coefficient_x_cubed_expansion : 
  let expansion := (fun x => (x^2 + 1)^2 * (x - 1)^6)
  ∃ (a b c d e f g h : ℤ), 
    (∀ x, expansion x = a*x^8 + b*x^7 + c*x^6 + d*x^5 + e*x^4 + (-32)*x^3 + f*x^2 + g*x + h) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_expansion_l3691_369156


namespace NUMINAMATH_CALUDE_density_of_S_l3691_369122

def S : Set ℚ := {q : ℚ | ∃ (m n : ℕ), q = (m * n : ℚ) / ((m^2 + n^2) : ℚ)}

theorem density_of_S (x y : ℚ) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y) :
  ∃ z : ℚ, z ∈ S ∧ x < z ∧ z < y := by
  sorry

end NUMINAMATH_CALUDE_density_of_S_l3691_369122


namespace NUMINAMATH_CALUDE_second_mission_duration_l3691_369177

def planned_duration : ℕ := 5
def actual_duration_increase : ℚ := 60 / 100
def total_mission_time : ℕ := 11

theorem second_mission_duration :
  let actual_first_mission := planned_duration + (planned_duration * actual_duration_increase).floor
  let second_mission := total_mission_time - actual_first_mission
  second_mission = 3 := by
  sorry

end NUMINAMATH_CALUDE_second_mission_duration_l3691_369177


namespace NUMINAMATH_CALUDE_digits_divisible_by_3_in_base_4_of_375_l3691_369114

def base_4_representation (n : ℕ) : List ℕ :=
  sorry

def count_divisible_by_3 (digits : List ℕ) : ℕ :=
  sorry

theorem digits_divisible_by_3_in_base_4_of_375 :
  count_divisible_by_3 (base_4_representation 375) = 2 :=
sorry

end NUMINAMATH_CALUDE_digits_divisible_by_3_in_base_4_of_375_l3691_369114


namespace NUMINAMATH_CALUDE_gcd_n_cube_plus_25_and_n_plus_3_l3691_369100

theorem gcd_n_cube_plus_25_and_n_plus_3 (n : ℕ) (h : n > 3^2) :
  Nat.gcd (n^3 + 5^2) (n + 3) = if (n + 3) % 2 = 0 then 2 else 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_n_cube_plus_25_and_n_plus_3_l3691_369100


namespace NUMINAMATH_CALUDE_set_A_proof_l3691_369166

def U : Set ℕ := {1, 3, 5, 7, 9}

theorem set_A_proof (A B : Set ℕ) 
  (h1 : A ⊆ U)
  (h2 : B ⊆ U)
  (h3 : A ∩ B = {3, 5})
  (h4 : A ∩ (U \ B) = {9}) :
  A = {3, 5, 9} := by
  sorry


end NUMINAMATH_CALUDE_set_A_proof_l3691_369166


namespace NUMINAMATH_CALUDE_extended_segment_endpoint_l3691_369136

-- Define points in 2D space
def Point := ℝ × ℝ

-- Define the given points
def A : Point := (-3, 5)
def B : Point := (9, -1)

-- Define vector addition
def vadd (p q : Point) : Point := (p.1 + q.1, p.2 + q.2)

-- Define scalar multiplication
def smul (k : ℝ) (p : Point) : Point := (k * p.1, k * p.2)

-- Define vector from two points
def vec (p q : Point) : Point := (q.1 - p.1, q.2 - p.2)

-- Theorem statement
theorem extended_segment_endpoint (C : Point) :
  vec A B = smul 3 (vec B C) → C = (15, -4) := by
  sorry

end NUMINAMATH_CALUDE_extended_segment_endpoint_l3691_369136


namespace NUMINAMATH_CALUDE_son_shoveling_time_l3691_369119

/-- Given a driveway shoveling scenario with three people, this theorem proves
    the time it takes for the son to shovel the entire driveway alone. -/
theorem son_shoveling_time (wayne_rate son_rate neighbor_rate : ℝ) 
  (h1 : wayne_rate = 6 * son_rate) 
  (h2 : neighbor_rate = 2 * wayne_rate) 
  (h3 : son_rate + wayne_rate + neighbor_rate = 1 / 2) : 
  1 / son_rate = 38 := by
  sorry

end NUMINAMATH_CALUDE_son_shoveling_time_l3691_369119


namespace NUMINAMATH_CALUDE_farmer_picked_thirty_today_l3691_369129

/-- Represents the number of tomatoes picked today by a farmer -/
def tomatoes_picked_today (initial : ℕ) (picked_yesterday : ℕ) (left_after_today : ℕ) : ℕ :=
  initial - picked_yesterday - left_after_today

/-- Theorem stating that the farmer picked 30 tomatoes today -/
theorem farmer_picked_thirty_today :
  tomatoes_picked_today 171 134 7 = 30 := by
  sorry

end NUMINAMATH_CALUDE_farmer_picked_thirty_today_l3691_369129


namespace NUMINAMATH_CALUDE_max_abs_x2_l3691_369196

theorem max_abs_x2 (x₁ x₂ x₃ : ℝ) (h : x₁^2 + x₂^2 + x₃^2 + x₁*x₂ + x₂*x₃ = 2) : 
  ∃ (M : ℝ), M = 2 ∧ |x₂| ≤ M ∧ ∃ (y₁ y₂ y₃ : ℝ), y₁^2 + y₂^2 + y₃^2 + y₁*y₂ + y₂*y₃ = 2 ∧ |y₂| = M :=
sorry

end NUMINAMATH_CALUDE_max_abs_x2_l3691_369196


namespace NUMINAMATH_CALUDE_total_boxes_theorem_l3691_369158

/-- Calculates the total number of boxes sold over three days given the conditions --/
def total_boxes_sold (friday_boxes : ℕ) : ℕ :=
  let saturday_boxes := friday_boxes + (friday_boxes * 50 / 100)
  let sunday_boxes := saturday_boxes - (saturday_boxes * 30 / 100)
  friday_boxes + saturday_boxes + sunday_boxes

/-- Proves that the total number of boxes sold over three days is 213 --/
theorem total_boxes_theorem : total_boxes_sold 60 = 213 := by
  sorry

#eval total_boxes_sold 60

end NUMINAMATH_CALUDE_total_boxes_theorem_l3691_369158


namespace NUMINAMATH_CALUDE_lineup_count_l3691_369189

/-- The number of ways to choose a starting lineup for a football team -/
def choose_lineup (total_members : ℕ) (offensive_linemen : ℕ) : ℕ :=
  offensive_linemen * (total_members - 1) * (total_members - 2) * (total_members - 3) * (total_members - 4)

/-- Theorem: The number of ways to choose a starting lineup for a team of 15 members
    with 5 offensive linemen is 109200 -/
theorem lineup_count :
  choose_lineup 15 5 = 109200 := by
  sorry

end NUMINAMATH_CALUDE_lineup_count_l3691_369189


namespace NUMINAMATH_CALUDE_largest_five_digit_congruent_to_31_mod_26_l3691_369194

theorem largest_five_digit_congruent_to_31_mod_26 :
  ∀ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n ≡ 31 [MOD 26] → n ≤ 99975 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_congruent_to_31_mod_26_l3691_369194


namespace NUMINAMATH_CALUDE_infinite_prime_factors_and_non_factors_l3691_369116

def sequence_a : ℕ → ℕ
  | 0 => 4
  | n + 1 => sequence_a n * (sequence_a n - 1)

def prime_factors (n : ℕ) : Set ℕ :=
  {p : ℕ | Nat.Prime p ∧ p ∣ n}

def prime_factors_of_sequence : Set ℕ :=
  ⋃ n, prime_factors (sequence_a n)

def primes_not_dividing_sequence : Set ℕ :=
  {p : ℕ | Nat.Prime p ∧ ∀ n, ¬(p ∣ sequence_a n)}

theorem infinite_prime_factors_and_non_factors :
  (Set.Infinite prime_factors_of_sequence) ∧
  (Set.Infinite primes_not_dividing_sequence) :=
sorry

end NUMINAMATH_CALUDE_infinite_prime_factors_and_non_factors_l3691_369116


namespace NUMINAMATH_CALUDE_max_value_abc_l3691_369101

theorem max_value_abc (a b : Real) (c : Fin 2) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  ∃ (a₀ b₀ : Real) (c₀ : Fin 2) (ha₀ : 0 ≤ a₀ ∧ a₀ ≤ 1) (hb₀ : 0 ≤ b₀ ∧ b₀ ≤ 1),
    Real.sqrt (a * b * c.val) + Real.sqrt ((1 - a) * (1 - b) * (1 - c.val)) ≤ 1 ∧
    Real.sqrt (a₀ * b₀ * c₀.val) + Real.sqrt ((1 - a₀) * (1 - b₀) * (1 - c₀.val)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_abc_l3691_369101


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3691_369109

theorem arithmetic_mean_problem (a b c d : ℝ) :
  (a + b) / 2 = 115 →
  (b + c) / 2 = 160 →
  (b + d) / 2 = 175 →
  a - d = -120 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3691_369109


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3691_369124

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = 2 - Complex.I) :
  z.im = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3691_369124


namespace NUMINAMATH_CALUDE_production_period_is_seven_days_l3691_369188

def computers_per_day : ℕ := 1500
def price_per_computer : ℕ := 150
def total_revenue : ℕ := 1575000

theorem production_period_is_seven_days :
  (total_revenue / price_per_computer) / computers_per_day = 7 := by
  sorry

end NUMINAMATH_CALUDE_production_period_is_seven_days_l3691_369188


namespace NUMINAMATH_CALUDE_bitangent_proof_l3691_369155

/-- The curve equation -/
def f (x : ℝ) : ℝ := x^4 + 2*x^3 - 11*x^2 - 13*x + 35

/-- The proposed bitangent line equation -/
def g (x : ℝ) : ℝ := -x - 1

/-- Theorem stating that g is a bitangent to f -/
theorem bitangent_proof :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
  f x₁ = g x₁ ∧ f x₂ = g x₂ ∧
  (deriv f) x₁ = (deriv g) x₁ ∧ (deriv f) x₂ = (deriv g) x₂ :=
sorry

end NUMINAMATH_CALUDE_bitangent_proof_l3691_369155


namespace NUMINAMATH_CALUDE_house_wall_nails_l3691_369142

/-- The number of large planks used for the house wall. -/
def large_planks : ℕ := 13

/-- The number of nails needed for each large plank. -/
def nails_per_plank : ℕ := 17

/-- The number of additional nails needed for smaller planks. -/
def additional_nails : ℕ := 8

/-- The total number of nails needed for the house wall. -/
def total_nails : ℕ := large_planks * nails_per_plank + additional_nails

theorem house_wall_nails : total_nails = 229 := by
  sorry

end NUMINAMATH_CALUDE_house_wall_nails_l3691_369142


namespace NUMINAMATH_CALUDE_product_of_fractions_l3691_369108

theorem product_of_fractions : (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l3691_369108


namespace NUMINAMATH_CALUDE_no_seven_edge_polyhedron_exists_polyhedron_with_2n_and_2n_plus_3_edges_l3691_369112

-- Define a convex polyhedron
structure ConvexPolyhedron where
  edges : ℕ
  is_convex : Bool

-- Theorem 1: A convex polyhedron cannot have exactly 7 edges
theorem no_seven_edge_polyhedron :
  ¬∃ (p : ConvexPolyhedron), p.edges = 7 ∧ p.is_convex = true :=
sorry

-- Theorem 2: For any integer n ≥ 3, there exists a convex polyhedron 
-- with 2n edges and another with 2n + 3 edges
theorem exists_polyhedron_with_2n_and_2n_plus_3_edges (n : ℕ) (h : n ≥ 3) :
  (∃ (p : ConvexPolyhedron), p.edges = 2 * n ∧ p.is_convex = true) ∧
  (∃ (q : ConvexPolyhedron), q.edges = 2 * n + 3 ∧ q.is_convex = true) :=
sorry

end NUMINAMATH_CALUDE_no_seven_edge_polyhedron_exists_polyhedron_with_2n_and_2n_plus_3_edges_l3691_369112


namespace NUMINAMATH_CALUDE_expression_evaluation_l3691_369157

theorem expression_evaluation : 3^2 / 3 - 4 * 2 + 2^3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3691_369157


namespace NUMINAMATH_CALUDE_intersection_S_T_l3691_369151

def S : Set ℝ := {x | (x - 3) / (x - 6) ≤ 0 ∧ x ≠ 6}
def T : Set ℝ := {2, 3, 4, 5, 6}

theorem intersection_S_T : S ∩ T = {3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_S_T_l3691_369151


namespace NUMINAMATH_CALUDE_clock_hands_angle_at_7_l3691_369181

/-- The number of hour marks on a clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees in a full circle -/
def full_circle_degrees : ℕ := 360

/-- The hour we're interested in -/
def target_hour : ℕ := 7

/-- The angle between each hour mark on the clock -/
def hour_angle : ℕ := full_circle_degrees / clock_hours

/-- The smaller angle formed by the clock hands at 7 o'clock -/
def smaller_angle_at_7 : ℕ := target_hour * hour_angle

theorem clock_hands_angle_at_7 :
  smaller_angle_at_7 = 150 := by sorry

end NUMINAMATH_CALUDE_clock_hands_angle_at_7_l3691_369181


namespace NUMINAMATH_CALUDE_product_of_roots_l3691_369138

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 4) = 18 → 
  ∃ y : ℝ, (y + 3) * (y - 4) = 18 ∧ x * y = -30 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l3691_369138


namespace NUMINAMATH_CALUDE_watch_price_proof_l3691_369185

/-- Represents the original cost price of the watch in Rupees. -/
def original_price : ℝ := 1800

/-- The selling price after discounts and loss. -/
def selling_price (price : ℝ) : ℝ := price * (1 - 0.05) * (1 - 0.03) * (1 - 0.10)

/-- The selling price for an 8% gain with 12% tax. -/
def selling_price_with_gain_and_tax (price : ℝ) : ℝ := price * (1 + 0.08) + price * 0.12

theorem watch_price_proof :
  selling_price original_price = original_price * 0.90 ∧
  selling_price_with_gain_and_tax original_price = selling_price original_price + 540 :=
by sorry

end NUMINAMATH_CALUDE_watch_price_proof_l3691_369185


namespace NUMINAMATH_CALUDE_sum_m_n_equals_negative_one_l3691_369183

theorem sum_m_n_equals_negative_one (m n : ℝ) 
  (h : Real.sqrt (m - 2) + (n + 3)^2 = 0) : m + n = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_m_n_equals_negative_one_l3691_369183


namespace NUMINAMATH_CALUDE_max_value_sum_fractions_l3691_369127

theorem max_value_sum_fractions (a b c : ℝ) 
  (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c)
  (h_sum : a + b + c = 2) :
  (a * b / (a + b) + a * c / (a + c) + b * c / (b + c)) ≤ 1 ∧
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ a' + b' + c' = 2 ∧
    a' * b' / (a' + b') + a' * c' / (a' + c') + b' * c' / (b' + c') = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_fractions_l3691_369127


namespace NUMINAMATH_CALUDE_cube_side_ratio_l3691_369111

/-- Given two cubes of the same material, if one cube weighs 5 pounds and the other weighs 40 pounds,
    then the ratio of the side length of the heavier cube to the side length of the lighter cube is 2:1. -/
theorem cube_side_ratio (s S : ℝ) (h1 : s > 0) (h2 : S > 0) : 
  (5 : ℝ) / s^3 = (40 : ℝ) / S^3 → S / s = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_side_ratio_l3691_369111


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l3691_369179

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  6 * Real.sqrt (a * b) + 3 / a + 3 / b ≥ 12 :=
sorry

theorem min_value_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 6 * Real.sqrt (a * b) + 3 / a + 3 / b < 12 + ε :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l3691_369179


namespace NUMINAMATH_CALUDE_soaps_in_package_l3691_369120

/-- Given a number of boxes, packages per box, and total soaps, calculates soaps per package -/
def soaps_per_package (num_boxes : ℕ) (packages_per_box : ℕ) (total_soaps : ℕ) : ℕ :=
  total_soaps / (num_boxes * packages_per_box)

/-- Theorem: There are 192 soaps in one package -/
theorem soaps_in_package :
  soaps_per_package 2 6 2304 = 192 := by sorry

end NUMINAMATH_CALUDE_soaps_in_package_l3691_369120


namespace NUMINAMATH_CALUDE_b_spending_percentage_l3691_369133

/-- Proves that B spends 85% of his salary given the specified conditions -/
theorem b_spending_percentage (total_salary : ℝ) (a_salary : ℝ) (a_spending_rate : ℝ) 
  (h1 : total_salary = 2000)
  (h2 : a_salary = 1500)
  (h3 : a_spending_rate = 0.95)
  (h4 : a_salary * (1 - a_spending_rate) = (total_salary - a_salary) * (1 - b_spending_rate)) :
  b_spending_rate = 0.85 := by
  sorry

#check b_spending_percentage

end NUMINAMATH_CALUDE_b_spending_percentage_l3691_369133


namespace NUMINAMATH_CALUDE_scout_hourly_rate_l3691_369180

/-- Represents Scout's weekend earnings --/
def weekend_earnings (hourly_rate : ℚ) : ℚ :=
  -- Saturday earnings
  (4 * hourly_rate + 5 * 5) +
  -- Sunday earnings
  (5 * hourly_rate + 8 * 5)

/-- Theorem stating that Scout's hourly rate is $10.00 --/
theorem scout_hourly_rate :
  ∃ (rate : ℚ), weekend_earnings rate = 155 ∧ rate = 10 := by
  sorry

end NUMINAMATH_CALUDE_scout_hourly_rate_l3691_369180


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_300_l3691_369104

/-- Given a natural number n, returns the sum of digits in its binary representation -/
def sumOfBinaryDigits (n : ℕ) : ℕ :=
  -- Implementation details omitted
  sorry

/-- Theorem: The sum of digits in the binary representation of 300 is 4 -/
theorem sum_of_binary_digits_300 : sumOfBinaryDigits 300 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_binary_digits_300_l3691_369104


namespace NUMINAMATH_CALUDE_range_of_a_l3691_369176

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x > a}
def B : Set ℝ := {-1, 0, 1}

-- Theorem statement
theorem range_of_a (a : ℝ) : A a ∩ B = {0, 1} → a ∈ Set.Icc (-1) 0 ∧ a ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3691_369176


namespace NUMINAMATH_CALUDE_roses_in_vase_l3691_369126

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := 7

/-- The number of orchids initially in the vase -/
def initial_orchids : ℕ := 12

/-- The number of orchids in the vase now -/
def current_orchids : ℕ := 20

/-- The difference between the number of orchids and roses in the vase now -/
def orchid_rose_difference : ℕ := 9

/-- The number of roses in the vase now -/
def current_roses : ℕ := 11

theorem roses_in_vase :
  current_orchids = current_roses + orchid_rose_difference :=
by sorry

end NUMINAMATH_CALUDE_roses_in_vase_l3691_369126
