import Mathlib

namespace NUMINAMATH_CALUDE_function_range_contained_in_unit_interval_l1639_163999

theorem function_range_contained_in_unit_interval 
  (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, x > y → (f x)^2 ≤ f y) : 
  ∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_range_contained_in_unit_interval_l1639_163999


namespace NUMINAMATH_CALUDE_people_in_line_l1639_163933

theorem people_in_line (initial_people total_people : ℕ) 
  (h1 : initial_people = 61)
  (h2 : total_people = 83)
  (h3 : total_people > initial_people) :
  total_people - initial_people = 22 := by
sorry

end NUMINAMATH_CALUDE_people_in_line_l1639_163933


namespace NUMINAMATH_CALUDE_paper_area_proof_l1639_163963

/-- The side length of each square piece of paper in centimeters -/
def side_length : ℝ := 8.5

/-- The number of pieces of square paper -/
def num_pieces : ℝ := 3.2

/-- The total area when gluing the pieces together without any gap -/
def total_area : ℝ := 231.2

/-- Theorem stating that the total area of the glued pieces is 231.2 cm² -/
theorem paper_area_proof : 
  side_length * side_length * num_pieces = total_area := by
  sorry

end NUMINAMATH_CALUDE_paper_area_proof_l1639_163963


namespace NUMINAMATH_CALUDE_petya_wins_l1639_163953

/-- Represents the game between Petya and Vasya -/
structure CandyGame where
  total_candies : ℕ
  prob_two_caramels : ℝ

/-- Defines the game with the given conditions -/
def game : CandyGame :=
  { total_candies := 25,
    prob_two_caramels := 0.54 }

/-- Theorem: Petya has a higher chance of winning -/
theorem petya_wins (g : CandyGame) 
  (h1 : g.total_candies = 25)
  (h2 : g.prob_two_caramels = 0.54) :
  g.prob_two_caramels > 1 - g.prob_two_caramels := by
  sorry

#check petya_wins game

end NUMINAMATH_CALUDE_petya_wins_l1639_163953


namespace NUMINAMATH_CALUDE_min_tokens_99x99_grid_l1639_163992

/-- Represents a grid of cells -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a square subgrid -/
structure Subgrid :=
  (size : ℕ)

/-- Calculates the minimum number of tokens required for a grid -/
def min_tokens (g : Grid) (sg : Subgrid) (tokens_per_subgrid : ℕ) : ℕ :=
  g.rows * g.cols - (g.rows / sg.size) * (g.cols / sg.size) * tokens_per_subgrid

/-- The main theorem stating the minimum number of tokens required -/
theorem min_tokens_99x99_grid : 
  let g : Grid := ⟨99, 99⟩
  let sg : Subgrid := ⟨4⟩
  let tokens_per_subgrid : ℕ := 8
  min_tokens g sg tokens_per_subgrid = 4801 := by
  sorry

#check min_tokens_99x99_grid

end NUMINAMATH_CALUDE_min_tokens_99x99_grid_l1639_163992


namespace NUMINAMATH_CALUDE_yans_distance_ratio_l1639_163995

theorem yans_distance_ratio :
  ∀ (w x y : ℝ),
    w > 0 →  -- walking speed is positive
    x > 0 →  -- distance from Yan to home is positive
    y > 0 →  -- distance from Yan to stadium is positive
    y / w = x / w + (x + y) / (10 * w) →  -- time equality condition
    x / y = 9 / 11 := by
  sorry

end NUMINAMATH_CALUDE_yans_distance_ratio_l1639_163995


namespace NUMINAMATH_CALUDE_remainder_sum_equals_27_l1639_163921

theorem remainder_sum_equals_27 (a : ℕ) (h : a > 0) : 
  (50 % a + 72 % a + 157 % a = 27) → a = 21 :=
by sorry

end NUMINAMATH_CALUDE_remainder_sum_equals_27_l1639_163921


namespace NUMINAMATH_CALUDE_modulo_five_power_difference_l1639_163906

theorem modulo_five_power_difference : (27^1235 - 19^1235) % 5 = 2 := by sorry

end NUMINAMATH_CALUDE_modulo_five_power_difference_l1639_163906


namespace NUMINAMATH_CALUDE_triangle_theorem_l1639_163968

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a * Real.sin t.C = Real.sqrt (3 * t.c) * Real.cos t.A) :
  t.A = π / 3 ∧ 
  (t.c = 4 ∧ t.a = 5 * Real.sqrt 3 → 
    Real.cos (2 * t.C - t.A) = (17 + 12 * Real.sqrt 7) / 50) := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l1639_163968


namespace NUMINAMATH_CALUDE_m_value_l1639_163954

theorem m_value (a b : ℝ) (m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : 1/a + 1/b = 2) : m = 10 := by
  sorry

end NUMINAMATH_CALUDE_m_value_l1639_163954


namespace NUMINAMATH_CALUDE_tens_digit_of_13_pow_2023_l1639_163977

theorem tens_digit_of_13_pow_2023 :
  ∃ k : ℕ, 13^2023 = 100 * k + 97 :=
by
  -- We assume 13^20 ≡ 1 (mod 100) as a hypothesis
  have h1 : ∃ m : ℕ, 13^20 = 100 * m + 1 := sorry
  
  -- We use the division algorithm to write 2023 = 20q + r
  have h2 : ∃ q r : ℕ, 2023 = 20 * q + r ∧ r < 20 := sorry
  
  -- We prove that r = 3
  have h3 : ∃ q : ℕ, 2023 = 20 * q + 3 := sorry
  
  -- We prove that 13^3 ≡ 97 (mod 100)
  have h4 : ∃ n : ℕ, 13^3 = 100 * n + 97 := sorry
  
  -- Main proof
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_13_pow_2023_l1639_163977


namespace NUMINAMATH_CALUDE_crayons_per_pack_l1639_163938

theorem crayons_per_pack (num_packs : ℕ) (extra_crayons : ℕ) (total_crayons : ℕ) 
  (h1 : num_packs = 4)
  (h2 : extra_crayons = 6)
  (h3 : total_crayons = 46) :
  ∃ (crayons_per_pack : ℕ), 
    crayons_per_pack * num_packs + extra_crayons = total_crayons ∧ 
    crayons_per_pack = 10 := by
  sorry

end NUMINAMATH_CALUDE_crayons_per_pack_l1639_163938


namespace NUMINAMATH_CALUDE_sqrt_two_irrational_in_set_l1639_163994

-- Define the set of numbers
def number_set : Set ℝ := {0, 1.414, Real.sqrt 2, 1/3}

-- Define irrationality
def is_irrational (x : ℝ) : Prop := ∀ p q : ℤ, q ≠ 0 → x ≠ (p : ℝ) / (q : ℝ)

-- Theorem statement
theorem sqrt_two_irrational_in_set : 
  ∃ x ∈ number_set, is_irrational x ∧ x = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_two_irrational_in_set_l1639_163994


namespace NUMINAMATH_CALUDE_circle_with_chords_theorem_l1639_163925

/-- Represents a circle with two intersecting chords --/
structure CircleWithChords where
  radius : ℝ
  chord_length : ℝ
  intersection_distance : ℝ

/-- Represents the area of a region in the form mπ - n√d --/
structure RegionArea where
  m : ℕ
  n : ℕ
  d : ℕ

/-- Checks if a number is square-free (not divisible by the square of any prime) --/
def is_square_free (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p * p ∣ n) → p = 1

/-- Main theorem about the circle with intersecting chords --/
theorem circle_with_chords_theorem (circle : CircleWithChords) 
  (h1 : circle.radius = 36)
  (h2 : circle.chord_length = 66)
  (h3 : circle.intersection_distance = 12) :
  ∃ (area : RegionArea), 
    (area.m : ℝ) * Real.pi - (area.n : ℝ) * Real.sqrt (area.d : ℝ) > 0 ∧
    is_square_free area.d ∧
    area.m + area.n + area.d = 378 :=
  sorry

end NUMINAMATH_CALUDE_circle_with_chords_theorem_l1639_163925


namespace NUMINAMATH_CALUDE_f_symmetric_l1639_163950

noncomputable def f (x : ℝ) : ℝ :=
  (6 * Real.cos (Real.pi + x) + 5 * (Real.sin (Real.pi - x))^2 - 4) / Real.cos (2 * Real.pi - x)

theorem f_symmetric (m : ℝ) (h : f m = 2) : f (-m) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetric_l1639_163950


namespace NUMINAMATH_CALUDE_problem_solution_l1639_163964

theorem problem_solution (x : ℝ) (h : x + Real.sqrt (x^2 - 4) + 1 / (x - Real.sqrt (x^2 - 4)) = 10) :
  x^2 + Real.sqrt (x^4 - 4) + 1 / (x^2 + Real.sqrt (x^4 - 4)) = 841/100 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1639_163964


namespace NUMINAMATH_CALUDE_ordering_of_powers_l1639_163905

theorem ordering_of_powers : 6^8 < 3^15 ∧ 3^15 < 8^10 := by
  sorry

end NUMINAMATH_CALUDE_ordering_of_powers_l1639_163905


namespace NUMINAMATH_CALUDE_twenty_sided_polygon_selection_l1639_163969

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : Fin n → Set (Fin n × Fin n)
  convex : sorry -- Additional property to ensure convexity

/-- The condition that two sides have at least k sides between them -/
def HasKSidesBetween (n : ℕ) (k : ℕ) (s₁ s₂ : Fin n × Fin n) : Prop :=
  sorry

/-- The number of ways to choose m sides from an n-sided polygon with k sides between each pair -/
def CountValidSelections (n m k : ℕ) : ℕ :=
  sorry

theorem twenty_sided_polygon_selection :
  CountValidSelections 20 3 2 = 520 :=
sorry

end NUMINAMATH_CALUDE_twenty_sided_polygon_selection_l1639_163969


namespace NUMINAMATH_CALUDE_chocolate_savings_bernie_savings_l1639_163941

/-- Calculates the savings when buying chocolates at a lower price over a given period -/
theorem chocolate_savings 
  (weeks : ℕ) 
  (chocolates_per_week : ℕ) 
  (price_local : ℚ) 
  (price_discount : ℚ) :
  weeks * chocolates_per_week * (price_local - price_discount) = 
  weeks * chocolates_per_week * price_local - weeks * chocolates_per_week * price_discount :=
by sorry

/-- Proves that Bernie saves $6 over three weeks by buying chocolates at the discounted store -/
theorem bernie_savings :
  let weeks : ℕ := 3
  let chocolates_per_week : ℕ := 2
  let price_local : ℚ := 3
  let price_discount : ℚ := 2
  weeks * chocolates_per_week * (price_local - price_discount) = 6 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_savings_bernie_savings_l1639_163941


namespace NUMINAMATH_CALUDE_game_probability_l1639_163918

theorem game_probability (lose_prob : ℚ) (h1 : lose_prob = 5/8) (h2 : lose_prob + win_prob = 1) : win_prob = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_game_probability_l1639_163918


namespace NUMINAMATH_CALUDE_quarter_power_inequality_l1639_163952

theorem quarter_power_inequality (x y : ℝ) (h1 : 0 < x) (h2 : x < y) (h3 : y < 1) :
  (1/4 : ℝ)^x > (1/4 : ℝ)^y := by
  sorry

end NUMINAMATH_CALUDE_quarter_power_inequality_l1639_163952


namespace NUMINAMATH_CALUDE_midpoint_coord_sum_l1639_163916

/-- The sum of the coordinates of the midpoint of a line segment with endpoints (10, 3) and (-4, -7) is 1 -/
theorem midpoint_coord_sum : 
  let x1 : ℝ := 10
  let y1 : ℝ := 3
  let x2 : ℝ := -4
  let y2 : ℝ := -7
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x + midpoint_y = 1 := by sorry

end NUMINAMATH_CALUDE_midpoint_coord_sum_l1639_163916


namespace NUMINAMATH_CALUDE_new_average_after_changes_l1639_163928

def initial_count : ℕ := 60
def initial_average : ℚ := 40
def removed_number1 : ℕ := 50
def removed_number2 : ℕ := 60
def added_number : ℕ := 35

theorem new_average_after_changes :
  let initial_sum := initial_count * initial_average
  let sum_after_removal := initial_sum - (removed_number1 + removed_number2)
  let final_sum := sum_after_removal + added_number
  let final_count := initial_count - 1
  final_sum / final_count = 39.41 := by sorry

end NUMINAMATH_CALUDE_new_average_after_changes_l1639_163928


namespace NUMINAMATH_CALUDE_ones_digit_of_34_power_power_4_cycle_seventeen_power_seventeen_odd_main_theorem_l1639_163943

theorem ones_digit_of_34_power (n : ℕ) : n > 0 → (34^n) % 10 = (4^n) % 10 := by sorry

theorem power_4_cycle (n : ℕ) : (4^n) % 10 = if n % 2 = 0 then 6 else 4 := by sorry

theorem seventeen_power_seventeen_odd : 17^17 % 2 = 1 := by sorry

theorem main_theorem : (34^(34*(17^17))) % 10 = 4 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_34_power_power_4_cycle_seventeen_power_seventeen_odd_main_theorem_l1639_163943


namespace NUMINAMATH_CALUDE_fraction_transformation_l1639_163990

theorem fraction_transformation (a b : ℕ) (h : a < b) :
  (∃ x : ℕ, (a + x : ℚ) / (b + x) = 1 / 2) ∧
  (¬ ∃ y z : ℕ, ((a + y : ℚ) * z) / ((b + y) * z) = 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_transformation_l1639_163990


namespace NUMINAMATH_CALUDE_burritos_per_box_burritos_problem_l1639_163904

theorem burritos_per_box (total_boxes : ℕ) (fraction_given_away : ℚ) 
  (burritos_eaten_per_day : ℕ) (days_eaten : ℕ) (burritos_left : ℕ) : ℕ :=
let burritos_per_box := 
  (burritos_left + burritos_eaten_per_day * days_eaten) / 
  (total_boxes * (1 - fraction_given_away))
20

theorem burritos_problem : 
  burritos_per_box 3 (1/3) 3 10 10 = 20 := by
sorry

end NUMINAMATH_CALUDE_burritos_per_box_burritos_problem_l1639_163904


namespace NUMINAMATH_CALUDE_discrete_probability_distribution_l1639_163903

theorem discrete_probability_distribution (p₁ p₃ : ℝ) : 
  p₃ = 4 * p₁ →
  p₁ + 0.15 + p₃ + 0.25 + 0.35 = 1 →
  p₁ = 0.05 ∧ p₃ = 0.20 := by
sorry

end NUMINAMATH_CALUDE_discrete_probability_distribution_l1639_163903


namespace NUMINAMATH_CALUDE_horner_v2_value_l1639_163915

/-- Horner's method for a polynomial --/
def horner_step (x : ℝ) (a b : ℝ) : ℝ := a * x + b

/-- The polynomial f(x) = x^6 + 6x^4 + 9x^2 + 208 --/
def f (x : ℝ) : ℝ := x^6 + 6*x^4 + 9*x^2 + 208

/-- Theorem: The value of v₂ in Horner's method for f(x) at x = -4 is 22 --/
theorem horner_v2_value :
  let x : ℝ := -4
  let v0 : ℝ := 1
  let v1 : ℝ := horner_step x v0 0
  let v2 : ℝ := horner_step x v1 6
  v2 = 22 := by sorry

end NUMINAMATH_CALUDE_horner_v2_value_l1639_163915


namespace NUMINAMATH_CALUDE_perpendicular_line_l1639_163958

/-- Given a line L1 with equation 3x - 6y = 9 and a point P (2, -3),
    prove that the line L2 with equation y = -2x + 1 is perpendicular to L1 and passes through P. -/
theorem perpendicular_line (x y : ℝ) : 
  (∃ (m b : ℝ), (3 * x - 6 * y = 9) ∧ (y = m * x + b)) →  -- L1 equation
  (y = -2 * x + 1) →                                     -- L2 equation
  ((-2) * (1/2) = -1) →                                  -- Perpendicularity condition
  ((-3) = -2 * 2 + 1) →                                  -- Point P satisfies L2
  (∃ (x₀ y₀ : ℝ), x₀ = 2 ∧ y₀ = -3 ∧ y₀ = -2 * x₀ + 1) -- L2 passes through P
  := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_l1639_163958


namespace NUMINAMATH_CALUDE_magnitude_AC_l1639_163967

def vector_AB : Fin 2 → ℝ := ![1, 2]
def vector_BC : Fin 2 → ℝ := ![3, 4]

theorem magnitude_AC : 
  let vector_AC := (vector_BC 0 - (-vector_AB 0), vector_BC 1 - (-vector_AB 1))
  Real.sqrt ((vector_AC.1)^2 + (vector_AC.2)^2) = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_AC_l1639_163967


namespace NUMINAMATH_CALUDE_remainder_is_zero_l1639_163993

-- Define the given binary number
def binary_num : Nat := 857  -- 1101011001₂ in decimal

-- Theorem statement
theorem remainder_is_zero : (binary_num + 3) % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_is_zero_l1639_163993


namespace NUMINAMATH_CALUDE_probability_two_white_balls_is_one_fifth_l1639_163978

/-- The probability of drawing two white balls from a box containing 7 white balls
    and 8 black balls, when drawing two balls at random without replacement. -/
def probability_two_white_balls : ℚ :=
  let total_balls : ℕ := 7 + 8
  let white_balls : ℕ := 7
  (Nat.choose white_balls 2 : ℚ) / (Nat.choose total_balls 2 : ℚ)

/-- Theorem stating that the probability of drawing two white balls is 1/5. -/
theorem probability_two_white_balls_is_one_fifth :
  probability_two_white_balls = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_white_balls_is_one_fifth_l1639_163978


namespace NUMINAMATH_CALUDE_special_sum_eq_1010_l1639_163966

/-- Double factorial of a natural number -/
def doubleFac : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * doubleFac n

/-- The sum from i=1 to 1010 of ((2i)!! / (2i+1)!!) * ((2i+1)! / (2i)!) -/
def specialSum : ℚ :=
  (Finset.range 1010).sum (fun i =>
    let i' := i + 1
    (doubleFac (2 * i') : ℚ) / (doubleFac (2 * i' + 1)) *
    (Nat.factorial (2 * i' + 1) : ℚ) / (Nat.factorial (2 * i')))

/-- The sum is equal to 1010 -/
theorem special_sum_eq_1010 : specialSum = 1010 := by sorry

end NUMINAMATH_CALUDE_special_sum_eq_1010_l1639_163966


namespace NUMINAMATH_CALUDE_auction_result_l1639_163957

def auction_total (tv_initial : ℝ) (tv_increase : ℝ) (phone_initial : ℝ) (phone_increase : ℝ) 
                  (laptop_initial : ℝ) (laptop_decrease : ℝ) (auction_fee_rate : ℝ) : ℝ :=
  let tv_final := tv_initial * (1 + tv_increase)
  let phone_final := phone_initial * (1 + phone_increase)
  let laptop_final := laptop_initial * (1 - laptop_decrease)
  let total_before_fee := tv_final + phone_final + laptop_final
  let fee := total_before_fee * auction_fee_rate
  total_before_fee - fee

theorem auction_result : 
  auction_total 500 (2/5) 400 0.4 800 0.15 0.05 = 1843 := by
  sorry

end NUMINAMATH_CALUDE_auction_result_l1639_163957


namespace NUMINAMATH_CALUDE_power_of_power_l1639_163948

theorem power_of_power (x : ℝ) : (x^2)^3 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l1639_163948


namespace NUMINAMATH_CALUDE_investor_c_profit_share_l1639_163965

/-- Represents the share of profit for an investor -/
def profit_share (investment : ℚ) (total_investment : ℚ) (total_profit : ℚ) : ℚ :=
  (investment / total_investment) * total_profit

/-- The problem statement -/
theorem investor_c_profit_share :
  let a_investment : ℚ := 800
  let b_investment : ℚ := 1000
  let c_investment : ℚ := 1200
  let total_profit : ℚ := 1000
  let total_investment : ℚ := a_investment + b_investment + c_investment
  profit_share c_investment total_investment total_profit = 400 := by
sorry

end NUMINAMATH_CALUDE_investor_c_profit_share_l1639_163965


namespace NUMINAMATH_CALUDE_oil_price_reduction_l1639_163942

/-- Represents the price reduction problem for oil -/
def OilPriceReduction (original_price reduced_price : ℝ) : Prop :=
  reduced_price = 0.8 * original_price

/-- Represents the relationship between price and quantity before and after reduction -/
def QuantityIncrease (original_price reduced_price : ℝ) : Prop :=
  ∃ (original_quantity : ℝ),
    800 = original_quantity * original_price ∧
    800 = (original_quantity + 5) * reduced_price

theorem oil_price_reduction (original_price reduced_price : ℝ) 
  (h1 : OilPriceReduction original_price reduced_price)
  (h2 : QuantityIncrease original_price reduced_price) :
  reduced_price = 32 := by
  sorry

end NUMINAMATH_CALUDE_oil_price_reduction_l1639_163942


namespace NUMINAMATH_CALUDE_cubic_root_sum_product_l1639_163931

theorem cubic_root_sum_product (p q r : ℂ) : 
  (6 * p ^ 3 - 5 * p ^ 2 + 13 * p - 10 = 0) →
  (6 * q ^ 3 - 5 * q ^ 2 + 13 * q - 10 = 0) →
  (6 * r ^ 3 - 5 * r ^ 2 + 13 * r - 10 = 0) →
  p * q + q * r + r * p = 13 / 6 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_product_l1639_163931


namespace NUMINAMATH_CALUDE_ratio_to_percent_l1639_163920

theorem ratio_to_percent (a b : ℕ) (h : a = 2 ∧ b = 3) : (a : ℚ) / (a + b : ℚ) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_percent_l1639_163920


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l1639_163910

theorem opposite_of_negative_two :
  ∀ x : ℤ, x + (-2) = 0 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l1639_163910


namespace NUMINAMATH_CALUDE_product_of_special_set_l1639_163919

theorem product_of_special_set (n : ℕ) (M : Finset ℝ) : 
  Odd n → 
  n > 1 → 
  Finset.card M = n →
  (∀ x ∈ M, (M.sum id - x) + x = M.sum id) →
  M.prod id = 0 := by
sorry

end NUMINAMATH_CALUDE_product_of_special_set_l1639_163919


namespace NUMINAMATH_CALUDE_intersection_M_N_l1639_163973

def M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def N : Set ℝ := {x | |x| < 2}

theorem intersection_M_N : M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1639_163973


namespace NUMINAMATH_CALUDE_easter_egg_arrangement_l1639_163923

theorem easter_egg_arrangement (yellow_eggs : Nat) (blue_eggs : Nat) 
  (min_eggs_per_basket : Nat) (min_baskets : Nat) :
  yellow_eggs = 30 →
  blue_eggs = 42 →
  min_eggs_per_basket = 6 →
  min_baskets = 3 →
  ∃ (eggs_per_basket : Nat),
    eggs_per_basket ≥ min_eggs_per_basket ∧
    eggs_per_basket ∣ yellow_eggs ∧
    eggs_per_basket ∣ blue_eggs ∧
    yellow_eggs / eggs_per_basket ≥ min_baskets ∧
    blue_eggs / eggs_per_basket ≥ min_baskets ∧
    ∀ (n : Nat),
      n > eggs_per_basket →
      ¬(n ≥ min_eggs_per_basket ∧
        n ∣ yellow_eggs ∧
        n ∣ blue_eggs ∧
        yellow_eggs / n ≥ min_baskets ∧
        blue_eggs / n ≥ min_baskets) :=
by
  sorry

end NUMINAMATH_CALUDE_easter_egg_arrangement_l1639_163923


namespace NUMINAMATH_CALUDE_scientific_notation_of_number_l1639_163935

def number : ℝ := 308000000

theorem scientific_notation_of_number :
  ∃ (a : ℝ) (n : ℤ), number = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3.08 ∧ n = 8 :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_number_l1639_163935


namespace NUMINAMATH_CALUDE_abs_is_even_and_increasing_l1639_163929

-- Define the absolute value function
def f (x : ℝ) := abs x

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define what it means for a function to be increasing on an interval
def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

theorem abs_is_even_and_increasing :
  is_even f ∧ is_increasing_on f 0 1 :=
sorry

end NUMINAMATH_CALUDE_abs_is_even_and_increasing_l1639_163929


namespace NUMINAMATH_CALUDE_skipping_odometer_conversion_l1639_163949

/-- Represents an odometer that skips digits 3 and 4 --/
def SkippingOdometer := Nat → Nat

/-- Converts a regular number to its representation on the skipping odometer --/
def toSkippingOdometer : Nat → Nat :=
  sorry

/-- Converts a number from the skipping odometer to its actual value --/
def fromSkippingOdometer : Nat → Nat :=
  sorry

theorem skipping_odometer_conversion :
  ∃ (odo : SkippingOdometer),
    (toSkippingOdometer 1029 = 002006) ∧
    (fromSkippingOdometer 002006 = 1029) := by
  sorry

end NUMINAMATH_CALUDE_skipping_odometer_conversion_l1639_163949


namespace NUMINAMATH_CALUDE_laticia_socks_l1639_163991

def sock_problem (nephew_socks week1_socks week2_extra week3_fraction week4_decrease : ℕ) : Prop :=
  let week2_socks := week1_socks + week2_extra
  let week3_socks := (week1_socks + week2_socks) / 2
  let week4_socks := week3_socks - week4_decrease
  nephew_socks + week1_socks + week2_socks + week3_socks + week4_socks = 57

theorem laticia_socks : 
  sock_problem 4 12 4 2 3 := by sorry

end NUMINAMATH_CALUDE_laticia_socks_l1639_163991


namespace NUMINAMATH_CALUDE_inequality_solution_l1639_163900

theorem inequality_solution (x : ℝ) :
  x ≠ -3 ∧ x ≠ 4 →
  ((x - 3) / (x + 3) > (2 * x - 1) / (x - 4) ↔
   (x > -6 - 3 * Real.sqrt 17 ∧ x < -6 + 3 * Real.sqrt 17) ∨
   (x > -3 ∧ x < 4)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1639_163900


namespace NUMINAMATH_CALUDE_tony_paint_area_l1639_163909

/-- The area Tony needs to paint on the wall -/
def area_to_paint (wall_height wall_length door_height door_width window_height window_width : ℝ) : ℝ :=
  wall_height * wall_length - (door_height * door_width + window_height * window_width)

/-- Theorem stating the area Tony needs to paint -/
theorem tony_paint_area :
  area_to_paint 10 15 3 5 2 3 = 129 := by
  sorry

end NUMINAMATH_CALUDE_tony_paint_area_l1639_163909


namespace NUMINAMATH_CALUDE_football_practice_kicks_l1639_163982

/-- The number of penalty kicks in a football practice session. -/
def penalty_kicks (total_players : ℕ) (goalkeepers : ℕ) : ℕ :=
  goalkeepers * (total_players - 1)

/-- Theorem: In a football club with 22 players including 4 goalkeepers,
    where each outfield player shoots once against each goalkeeper,
    the total number of penalty kicks is 84. -/
theorem football_practice_kicks :
  penalty_kicks 22 4 = 84 := by
  sorry

end NUMINAMATH_CALUDE_football_practice_kicks_l1639_163982


namespace NUMINAMATH_CALUDE_geometric_series_problem_l1639_163971

/-- Given two infinite geometric series with the specified conditions, prove that m = 8 -/
theorem geometric_series_problem (m : ℝ) : 
  let a₁ : ℝ := 18
  let b₁ : ℝ := 6
  let a₂ : ℝ := 18
  let b₂ : ℝ := 6 + m
  let r₁ := b₁ / a₁
  let r₂ := b₂ / a₂
  let S₁ := a₁ / (1 - r₁)
  let S₂ := a₂ / (1 - r₂)
  S₂ = 3 * S₁ → m = 8 := by
sorry


end NUMINAMATH_CALUDE_geometric_series_problem_l1639_163971


namespace NUMINAMATH_CALUDE_present_age_of_B_l1639_163927

/-- Given three people A, B, and C, whose ages satisfy certain conditions,
    prove that the present age of B is 30 years. -/
theorem present_age_of_B (A B C : ℕ) : 
  A + B + C = 90 →  -- Total present age is 90
  (A - 10) = 1 * x ∧ (B - 10) = 2 * x ∧ (C - 10) = 3 * x →  -- Age ratio 10 years ago
  B = 30 := by
sorry


end NUMINAMATH_CALUDE_present_age_of_B_l1639_163927


namespace NUMINAMATH_CALUDE_cube_face_perimeter_l1639_163981

-- Define the volume of the cube
def cube_volume : ℝ := 343

-- Theorem statement
theorem cube_face_perimeter :
  let side_length := (cube_volume ^ (1/3 : ℝ))
  (4 : ℝ) * side_length = 28 := by
  sorry

end NUMINAMATH_CALUDE_cube_face_perimeter_l1639_163981


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1639_163986

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1639_163986


namespace NUMINAMATH_CALUDE_simplify_fraction_l1639_163907

theorem simplify_fraction : (270 / 5400) * 30 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1639_163907


namespace NUMINAMATH_CALUDE_fifth_pattern_white_tiles_l1639_163913

/-- The number of white tiles in the n-th pattern of a hexagonal tile sequence -/
def white_tiles (n : ℕ) : ℕ := 4 * n + 2

/-- Theorem: The number of white tiles in the fifth pattern is 22 -/
theorem fifth_pattern_white_tiles : white_tiles 5 = 22 := by
  sorry

end NUMINAMATH_CALUDE_fifth_pattern_white_tiles_l1639_163913


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1639_163959

/-- Given a > 0 and f(x) = ax² + bx + c, with x₀ satisfying 2ax + b = 0,
    prove that f(x) ≥ f(x₀) for all x ∈ ℝ -/
theorem quadratic_minimum (a b c : ℝ) (ha : a > 0) :
  let f := fun x => a * x^2 + b * x + c
  let x₀ := -b / (2 * a)
  ∀ x, f x ≥ f x₀ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1639_163959


namespace NUMINAMATH_CALUDE_seventh_term_of_arithmetic_sequence_l1639_163911

def arithmetic_sequence (a : ℚ) (d : ℚ) : ℕ → ℚ
  | 0 => a
  | n + 1 => arithmetic_sequence a d n + d

theorem seventh_term_of_arithmetic_sequence 
  (a d : ℚ) 
  (h1 : (arithmetic_sequence a d 0) + 
        (arithmetic_sequence a d 1) + 
        (arithmetic_sequence a d 2) + 
        (arithmetic_sequence a d 3) + 
        (arithmetic_sequence a d 4) = 20)
  (h2 : arithmetic_sequence a d 5 = 8) :
  arithmetic_sequence a d 6 = 28 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_seventh_term_of_arithmetic_sequence_l1639_163911


namespace NUMINAMATH_CALUDE_equation_roots_l1639_163946

theorem equation_roots : ∀ x : ℝ, 3 * (x - 3) = (x - 3)^2 ↔ x = 3 ∨ x = 6 := by sorry

end NUMINAMATH_CALUDE_equation_roots_l1639_163946


namespace NUMINAMATH_CALUDE_smallest_integers_difference_difference_is_27720_l1639_163983

theorem smallest_integers_difference : ℕ → Prop :=
  fun d =>
    ∃ n₁ n₂ : ℕ,
      n₁ > 1 ∧ n₂ > 1 ∧
      n₂ > n₁ ∧
      (∀ k : ℕ, 2 ≤ k → k ≤ 11 → n₁ % k = 1) ∧
      (∀ k : ℕ, 2 ≤ k → k ≤ 11 → n₂ % k = 1) ∧
      (∀ m : ℕ, m > 1 → (∀ k : ℕ, 2 ≤ k → k ≤ 11 → m % k = 1) → m ≥ n₁) ∧
      d = n₂ - n₁

theorem difference_is_27720 : smallest_integers_difference 27720 := by sorry

end NUMINAMATH_CALUDE_smallest_integers_difference_difference_is_27720_l1639_163983


namespace NUMINAMATH_CALUDE_batting_average_is_62_l1639_163932

/-- Calculates the batting average given the total innings, highest score, score difference, and average excluding extremes. -/
def battingAverage (totalInnings : ℕ) (highestScore : ℕ) (scoreDifference : ℕ) (averageExcludingExtremes : ℕ) : ℚ :=
  let lowestScore := highestScore - scoreDifference
  let totalScoreExcludingExtremes := (totalInnings - 2) * averageExcludingExtremes
  let totalScore := totalScoreExcludingExtremes + highestScore + lowestScore
  totalScore / totalInnings

/-- Theorem stating that under the given conditions, the batting average is 62 runs. -/
theorem batting_average_is_62 :
  battingAverage 46 225 150 58 = 62 := by sorry

end NUMINAMATH_CALUDE_batting_average_is_62_l1639_163932


namespace NUMINAMATH_CALUDE_transformed_roots_l1639_163939

theorem transformed_roots (p : ℝ) (α β : ℝ) : 
  (3 * α^2 + 4 * α + p = 0) → 
  (3 * β^2 + 4 * β + p = 0) → 
  ((α / 3 - 2)^2 + 16 * (α / 3 - 2) + (60 + 3 * p) = 0) ∧
  ((β / 3 - 2)^2 + 16 * (β / 3 - 2) + (60 + 3 * p) = 0) := by
  sorry

end NUMINAMATH_CALUDE_transformed_roots_l1639_163939


namespace NUMINAMATH_CALUDE_leap_year_classification_l1639_163940

def isLeapYear (year : Nat) : Bool :=
  (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)

theorem leap_year_classification :
  (isLeapYear 2036 = true) ∧
  (isLeapYear 1996 = true) ∧
  (isLeapYear 1998 = false) ∧
  (isLeapYear 1700 = false) := by
  sorry

end NUMINAMATH_CALUDE_leap_year_classification_l1639_163940


namespace NUMINAMATH_CALUDE_ABCDE_binary_digits_l1639_163988

-- Define the base-16 number ABCDE₁₆
def ABCDE : ℕ := 10 * 16^4 + 11 * 16^3 + 12 * 16^2 + 13 * 16^1 + 14

-- Theorem stating that ABCDE₁₆ has 20 binary digits
theorem ABCDE_binary_digits : 
  2^19 ≤ ABCDE ∧ ABCDE < 2^20 :=
by sorry

end NUMINAMATH_CALUDE_ABCDE_binary_digits_l1639_163988


namespace NUMINAMATH_CALUDE_number_percentage_problem_l1639_163998

theorem number_percentage_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 14 → (40/100 : ℝ) * N = 168 := by
  sorry

end NUMINAMATH_CALUDE_number_percentage_problem_l1639_163998


namespace NUMINAMATH_CALUDE_no_valid_arrangement_l1639_163945

/-- Represents a card with a digit -/
structure Card where
  digit : Fin 10

/-- Represents an arrangement of cards -/
def Arrangement := List Card

/-- Checks if an arrangement satisfies the problem conditions -/
def satisfiesConditions (arr : Arrangement) : Prop :=
  ∀ i : Fin 10, ∃ pos1 pos2 : Nat,
    pos1 < pos2 ∧
    pos2 < arr.length ∧
    (arr.get ⟨pos1, by sorry⟩).digit = i ∧
    (arr.get ⟨pos2, by sorry⟩).digit = i ∧
    pos2 - pos1 - 1 = i.val

theorem no_valid_arrangement :
  ¬∃ (arr : Arrangement),
    arr.length = 20 ∧
    (∀ i : Fin 10, (arr.filter (λ c => c.digit = i)).length = 2) ∧
    satisfiesConditions arr := by
  sorry


end NUMINAMATH_CALUDE_no_valid_arrangement_l1639_163945


namespace NUMINAMATH_CALUDE_jessicas_allowance_l1639_163902

/-- Jessica's weekly allowance problem -/
theorem jessicas_allowance (allowance : ℝ) : 
  (allowance / 2 + 6 = 11) → allowance = 10 := by
  sorry

end NUMINAMATH_CALUDE_jessicas_allowance_l1639_163902


namespace NUMINAMATH_CALUDE_roots_sum_and_product_l1639_163930

theorem roots_sum_and_product (a b : ℝ) : 
  a^4 - 6*a^3 + 11*a^2 - 6*a - 1 = 0 →
  b^4 - 6*b^3 + 11*b^2 - 6*b - 1 = 0 →
  a + b + a*b = 4 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_and_product_l1639_163930


namespace NUMINAMATH_CALUDE_complement_of_M_l1639_163960

def U : Set ℕ := {1,2,3,4,5,6}
def M : Set ℕ := {1,2,4}

theorem complement_of_M : Mᶜ = {3,5,6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l1639_163960


namespace NUMINAMATH_CALUDE_circle_symmetry_l1639_163922

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y + 7 = 0

-- Define a line in the plane
def line (a b c : ℝ) (x y : ℝ) : Prop := a*x + b*y + c = 0

-- Define symmetry with respect to a line
def symmetric_wrt_line (C1 C2 : (ℝ → ℝ → Prop)) (l : ℝ → ℝ → Prop) : Prop :=
  ∀ (x1 y1 x2 y2 : ℝ), C1 x1 y1 ∧ C2 x2 y2 → 
    ∃ (x y : ℝ), l x y ∧ 
      (x - x1)^2 + (y - y1)^2 = (x - x2)^2 + (y - y2)^2

-- Theorem statement
theorem circle_symmetry :
  symmetric_wrt_line circle1 circle2 (line 1 (-1) 2) := by sorry

end NUMINAMATH_CALUDE_circle_symmetry_l1639_163922


namespace NUMINAMATH_CALUDE_xyz_sum_bounds_l1639_163936

theorem xyz_sum_bounds (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) :
  let m := x*y + y*z + z*x
  (∃ (k : ℝ), ∀ (a b c : ℝ), a^2 + b^2 + c^2 = 1 → x*y + y*z + z*x ≤ k) ∧
  (∃ (l : ℝ), ∀ (a b c : ℝ), a^2 + b^2 + c^2 = 1 → l ≤ x*y + y*z + z*x) ∧
  (∃ (a b c : ℝ), a^2 + b^2 + c^2 = 1 ∧ x*y + y*z + z*x = 1) ∧
  (∃ (a b c : ℝ), a^2 + b^2 + c^2 = 1 ∧ x*y + y*z + z*x = -1/2) :=
sorry

end NUMINAMATH_CALUDE_xyz_sum_bounds_l1639_163936


namespace NUMINAMATH_CALUDE_base_r_transaction_l1639_163976

/-- Converts a number from base r to base 10 -/
def to_base_10 (digits : List Nat) (r : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * r ^ i) 0

/-- The problem statement -/
theorem base_r_transaction (r : Nat) : r > 1 →
  (to_base_10 [0, 6, 5] r) + (to_base_10 [0, 2, 4] r) = (to_base_10 [0, 0, 1, 1] r) ↔ r = 8 := by
  sorry

end NUMINAMATH_CALUDE_base_r_transaction_l1639_163976


namespace NUMINAMATH_CALUDE_number_of_cats_l1639_163908

/-- Represents the number of cats on the ship. -/
def cats : ℕ := sorry

/-- Represents the number of sailors on the ship. -/
def sailors : ℕ := sorry

/-- Represents the number of cooks on the ship. -/
def cooks : ℕ := 1

/-- Represents the number of captains on the ship. -/
def captains : ℕ := 1

/-- The total number of heads on the ship. -/
def total_heads : ℕ := 16

/-- The total number of legs on the ship. -/
def total_legs : ℕ := 41

/-- Theorem stating that the number of cats on the ship is 5. -/
theorem number_of_cats : cats = 5 := by
  have head_count : cats + sailors + cooks + captains = total_heads := sorry
  have leg_count : 4 * cats + 2 * sailors + 2 * cooks + captains = total_legs := sorry
  sorry

end NUMINAMATH_CALUDE_number_of_cats_l1639_163908


namespace NUMINAMATH_CALUDE_gcd_of_45_75_105_l1639_163944

theorem gcd_of_45_75_105 : Nat.gcd 45 (Nat.gcd 75 105) = 15 := by sorry

end NUMINAMATH_CALUDE_gcd_of_45_75_105_l1639_163944


namespace NUMINAMATH_CALUDE_kabadi_player_count_l1639_163934

/-- The number of people who play kabadi -/
def kabadi_players : ℕ := 15

/-- The total number of players -/
def total_players : ℕ := 50

/-- The number of players who play kho kho only -/
def kho_kho_only : ℕ := 40

/-- The number of players who play both games -/
def both_games : ℕ := 5

theorem kabadi_player_count :
  kabadi_players = total_players - kho_kho_only + both_games :=
by sorry

end NUMINAMATH_CALUDE_kabadi_player_count_l1639_163934


namespace NUMINAMATH_CALUDE_circle_equation_simplified_fixed_point_satisfies_line_main_theorem_l1639_163989

/-- The fixed point P through which all lines pass -/
def P : ℝ × ℝ := (2, -1)

/-- The radius of the circle -/
def r : ℝ := 2

/-- The line equation that passes through P for all a ∈ ℝ -/
def line_equation (a x y : ℝ) : Prop :=
  (1 - a) * x + y + 2 * a - 1 = 0

/-- The circle equation with center P and radius r -/
def circle_equation (x y : ℝ) : Prop :=
  (x - P.1)^2 + (y - P.2)^2 = r^2

theorem circle_equation_simplified :
  ∀ x y : ℝ, circle_equation x y ↔ x^2 + y^2 - 4*x + 2*y + 1 = 0 :=
by sorry

theorem fixed_point_satisfies_line :
  ∀ a : ℝ, line_equation a P.1 P.2 :=
by sorry

theorem main_theorem :
  ∀ x y : ℝ, circle_equation x y ↔ x^2 + y^2 - 4*x + 2*y + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_simplified_fixed_point_satisfies_line_main_theorem_l1639_163989


namespace NUMINAMATH_CALUDE_factorize_difference_of_squares_l1639_163972

variable (R : Type*) [CommRing R]
variable (a x y : R)

theorem factorize_difference_of_squares : a * x^2 - a * y^2 = a * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorize_difference_of_squares_l1639_163972


namespace NUMINAMATH_CALUDE_polar_equation_is_circle_l1639_163955

-- Define the polar equation
def polar_equation (r θ : ℝ) : Prop := r = 3 * Real.sin θ * Real.cos θ

-- Define the Cartesian equation of a circle
def is_circle (x y : ℝ) : Prop := ∃ (h k r : ℝ), (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement
theorem polar_equation_is_circle :
  ∀ (x y : ℝ), (∃ (r θ : ℝ), polar_equation r θ ∧ x = r * Real.cos θ ∧ y = r * Real.sin θ) →
  is_circle x y :=
sorry

end NUMINAMATH_CALUDE_polar_equation_is_circle_l1639_163955


namespace NUMINAMATH_CALUDE_fraction_equality_l1639_163979

theorem fraction_equality (a b : ℚ) (h : a / b = 2 / 5) : (a - b) / b = -3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1639_163979


namespace NUMINAMATH_CALUDE_polar_to_circle_l1639_163926

/-- The polar equation r = 1 / (1 - sin θ) represents a circle. -/
theorem polar_to_circle : ∃ (h k R : ℝ), ∀ (x y : ℝ),
  (∃ (r θ : ℝ), r = 1 / (1 - Real.sin θ) ∧ x = r * Real.cos θ ∧ y = r * Real.sin θ) →
  (x - h)^2 + (y - k)^2 = R^2 :=
by sorry

end NUMINAMATH_CALUDE_polar_to_circle_l1639_163926


namespace NUMINAMATH_CALUDE_apples_pears_equivalence_l1639_163987

-- Define the relationship between apples and pears
def apples_to_pears (apples : ℚ) : ℚ :=
  (10 / 6) * apples

-- Theorem statement
theorem apples_pears_equivalence :
  apples_to_pears (3/4 * 6) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_apples_pears_equivalence_l1639_163987


namespace NUMINAMATH_CALUDE_savings_percentage_increase_l1639_163961

theorem savings_percentage_increase (initial_salary : ℝ) : 
  let last_year_savings := 0.10 * initial_salary
  let this_year_salary := 1.10 * initial_salary
  let this_year_savings := 0.15 * this_year_salary
  (this_year_savings / last_year_savings) * 100 = 165 := by
sorry

end NUMINAMATH_CALUDE_savings_percentage_increase_l1639_163961


namespace NUMINAMATH_CALUDE_kevins_watermelons_l1639_163985

/-- The weight of the first watermelon in pounds -/
def first_watermelon : ℝ := 9.91

/-- The weight of the second watermelon in pounds -/
def second_watermelon : ℝ := 4.11

/-- The total weight of watermelons Kevin bought -/
def total_weight : ℝ := first_watermelon + second_watermelon

/-- Theorem stating that the total weight of watermelons Kevin bought is 14.02 pounds -/
theorem kevins_watermelons : total_weight = 14.02 := by sorry

end NUMINAMATH_CALUDE_kevins_watermelons_l1639_163985


namespace NUMINAMATH_CALUDE_unique_solution_l1639_163970

-- Define the function g
def g (x : ℝ) : ℝ := (x - 1)^5 + (x - 1) - 34

-- State the theorem
theorem unique_solution :
  ∃! x : ℝ, g x = 0 ∧ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l1639_163970


namespace NUMINAMATH_CALUDE_office_age_problem_l1639_163951

/-- Given information about the ages of people in an office, prove that the average age of a specific group is 14 years. -/
theorem office_age_problem (total_people : ℕ) (avg_age_all : ℕ) (group1_size : ℕ) (group1_avg_age : ℕ) (group2_size : ℕ) (person15_age : ℕ) :
  total_people = 17 →
  avg_age_all = 15 →
  group1_size = 9 →
  group1_avg_age = 16 →
  group2_size = 5 →
  person15_age = 41 →
  (total_people * avg_age_all - group1_size * group1_avg_age - person15_age) / group2_size = 14 :=
by sorry

end NUMINAMATH_CALUDE_office_age_problem_l1639_163951


namespace NUMINAMATH_CALUDE_grid_coloring_probability_l1639_163962

/-- The number of squares in a row or column of the grid -/
def gridSize : ℕ := 4

/-- The total number of possible colorings for the grid -/
def totalColorings : ℕ := 2^(gridSize^2)

/-- The number of colorings with at least one 3-by-3 yellow square -/
def coloringsWithYellowSquare : ℕ := 510

/-- The probability of obtaining a grid without a 3-by-3 yellow square -/
def probabilityNoYellowSquare : ℚ := (totalColorings - coloringsWithYellowSquare) / totalColorings

theorem grid_coloring_probability :
  probabilityNoYellowSquare = 65026 / 65536 :=
sorry

end NUMINAMATH_CALUDE_grid_coloring_probability_l1639_163962


namespace NUMINAMATH_CALUDE_nabla_example_l1639_163912

-- Define the ∇ operation
def nabla (a b c d : ℝ) : ℝ := a * c + b * d

-- Theorem statement
theorem nabla_example : nabla 3 1 4 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_nabla_example_l1639_163912


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1639_163956

/-- Definition of the ellipse -/
def is_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

/-- The length of the major axis of the ellipse -/
def major_axis_length : ℝ := 6

/-- Theorem: The length of the major axis of the ellipse defined by x^2/9 + y^2/4 = 1 is 6 -/
theorem ellipse_major_axis_length :
  ∀ x y : ℝ, is_ellipse x y → major_axis_length = 6 := by
  sorry

#check ellipse_major_axis_length

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1639_163956


namespace NUMINAMATH_CALUDE_circle_common_chord_l1639_163997

/-- Given two circles with equations x^2 + y^2 = a^2 and x^2 + y^2 + ay - 6 = 0,
    where the common chord length is 2√3, prove that a = ±2 -/
theorem circle_common_chord (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = a^2) ∧ 
  (∃ (x y : ℝ), x^2 + y^2 + a*y - 6 = 0) ∧
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 + y₁^2 = a^2 ∧ 
    x₁^2 + y₁^2 + a*y₁ - 6 = 0 ∧
    x₂^2 + y₂^2 = a^2 ∧ 
    x₂^2 + y₂^2 + a*y₂ - 6 = 0 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 12) →
  a = 2 ∨ a = -2 :=
by sorry

end NUMINAMATH_CALUDE_circle_common_chord_l1639_163997


namespace NUMINAMATH_CALUDE_square_park_fencing_cost_l1639_163917

/-- Given a square park with a total fencing cost, calculate the cost per side -/
theorem square_park_fencing_cost (total_cost : ℝ) (h_total_cost : total_cost = 172) :
  total_cost / 4 = 43 := by
  sorry

end NUMINAMATH_CALUDE_square_park_fencing_cost_l1639_163917


namespace NUMINAMATH_CALUDE_place_left_representation_l1639_163984

/-- Represents a three-digit number -/
def ThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- Represents a two-digit number -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Represents the operation of placing a three-digit number to the left of a two-digit number -/
def PlaceLeft (x y : ℕ) : ℕ := 100 * x + y

theorem place_left_representation (x y : ℕ) 
  (hx : ThreeDigitNumber x) (hy : TwoDigitNumber y) :
  PlaceLeft x y = 100 * x + y :=
by sorry

end NUMINAMATH_CALUDE_place_left_representation_l1639_163984


namespace NUMINAMATH_CALUDE_greatest_integer_inequality_unique_greatest_integer_l1639_163974

theorem greatest_integer_inequality (x : ℤ) : (7 : ℚ) / 9 > (x : ℚ) / 15 ↔ x ≤ 11 := by sorry

theorem unique_greatest_integer : ∃! x : ℤ, x = (Nat.floor ((7 : ℚ) / 9 * 15) : ℤ) ∧ x = 11 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_inequality_unique_greatest_integer_l1639_163974


namespace NUMINAMATH_CALUDE_square_perimeter_l1639_163924

theorem square_perimeter (area : ℝ) (side : ℝ) (h1 : area = 36) (h2 : side^2 = area) :
  4 * side = 24 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l1639_163924


namespace NUMINAMATH_CALUDE_alices_preferred_number_l1639_163901

def is_between (n : ℕ) (a b : ℕ) : Prop := a ≤ n ∧ n ≤ b

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem alices_preferred_number :
  ∃! n : ℕ,
    is_between n 100 200 ∧
    11 ∣ n ∧
    ¬(2 ∣ n) ∧
    3 ∣ sum_of_digits n ∧
    n = 165 :=
by sorry

end NUMINAMATH_CALUDE_alices_preferred_number_l1639_163901


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1639_163996

/-- Given a geometric sequence {a_n} with common ratio q,
    if a_1 + a_3 = 10 and a_4 + a_6 = 5/4, then q = 1/2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = q * a n) →  -- geometric sequence condition
  a 1 + a 3 = 10 →                  -- first given condition
  a 4 + a 6 = 5/4 →                 -- second given condition
  q = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1639_163996


namespace NUMINAMATH_CALUDE_wolf_hunger_theorem_l1639_163975

/-- Represents the satiety value of a food item -/
structure SatietyValue (α : Type) where
  value : ℝ

/-- Represents the satiety state of the wolf -/
inductive SatietyState
  | Hunger
  | Satisfied
  | Overeating

/-- The satiety value of a piglet -/
def piglet_satiety : SatietyValue ℝ := ⟨1⟩

/-- The satiety value of a kid -/
def kid_satiety : SatietyValue ℝ := ⟨1⟩

/-- Calculates the total satiety value of a meal -/
def meal_satiety (piglets kids : ℕ) : ℝ :=
  (piglets : ℝ) * piglet_satiety.value + (kids : ℝ) * kid_satiety.value

/-- Determines the satiety state based on the meal satiety -/
def get_satiety_state (meal : ℝ) : SatietyState := sorry

/-- The theorem to be proved -/
theorem wolf_hunger_theorem :
  (get_satiety_state (meal_satiety 3 7) = SatietyState.Hunger) →
  (get_satiety_state (meal_satiety 7 1) = SatietyState.Overeating) →
  (get_satiety_state (meal_satiety 0 11) = SatietyState.Hunger) :=
by sorry

end NUMINAMATH_CALUDE_wolf_hunger_theorem_l1639_163975


namespace NUMINAMATH_CALUDE_trees_to_plant_l1639_163980

/-- The number of trees chopped down in the first half of the year -/
def first_half_trees : ℕ := 200

/-- The number of trees chopped down in the second half of the year -/
def second_half_trees : ℕ := 300

/-- The number of trees to be planted for each tree chopped down -/
def trees_to_plant_ratio : ℕ := 3

/-- Theorem stating the number of trees the company needs to plant -/
theorem trees_to_plant : 
  (first_half_trees + second_half_trees) * trees_to_plant_ratio = 1500 := by
  sorry

end NUMINAMATH_CALUDE_trees_to_plant_l1639_163980


namespace NUMINAMATH_CALUDE_largest_prime_divisor_check_l1639_163914

theorem largest_prime_divisor_check (n : ℕ) (h1 : 1000 ≤ n) (h2 : n ≤ 1100) :
  Nat.Prime n → ∀ p, Nat.Prime p ∧ p ≤ 31 → ¬(p ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_check_l1639_163914


namespace NUMINAMATH_CALUDE_cubic_sum_equals_linear_sum_l1639_163947

theorem cubic_sum_equals_linear_sum (k : ℝ) : 
  (∀ r s : ℝ, 3 * r^2 + 6 * r + k = 0 ∧ 3 * s^2 + 6 * s + k = 0 → r^3 + s^3 = r + s) ↔ 
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_equals_linear_sum_l1639_163947


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l1639_163937

theorem arithmetic_evaluation : 8 / 2 - 3 * 2 + 5^2 / 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l1639_163937
