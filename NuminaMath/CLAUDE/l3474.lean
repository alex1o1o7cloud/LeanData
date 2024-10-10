import Mathlib

namespace paper_sheet_length_l3474_347430

/-- The length of the second sheet of paper satisfies the area equation -/
theorem paper_sheet_length : ∃ L : ℝ, 2 * (11 * 17) = 2 * (8.5 * L) + 100 := by
  sorry

end paper_sheet_length_l3474_347430


namespace price_increase_theorem_l3474_347495

theorem price_increase_theorem (original_price : ℝ) (original_price_pos : original_price > 0) :
  let first_increase := original_price * 1.2
  let second_increase := first_increase * 1.15
  let total_increase := second_increase - original_price
  (total_increase / original_price) * 100 = 38 := by
sorry

end price_increase_theorem_l3474_347495


namespace empty_boxes_count_l3474_347492

theorem empty_boxes_count (total boxes_with_markers boxes_with_crayons boxes_with_both : ℕ) 
  (h1 : total = 15)
  (h2 : boxes_with_markers = 9)
  (h3 : boxes_with_crayons = 4)
  (h4 : boxes_with_both = 5) :
  total - (boxes_with_markers + boxes_with_crayons - boxes_with_both) = 7 := by
  sorry

end empty_boxes_count_l3474_347492


namespace janet_movie_cost_l3474_347446

/-- The cost per minute to film Janet's previous movie -/
def previous_cost_per_minute : ℝ := 5

/-- The length of Janet's previous movie in minutes -/
def previous_movie_length : ℝ := 120

/-- The length of Janet's new movie in minutes -/
def new_movie_length : ℝ := previous_movie_length * 1.6

/-- The cost per minute to film Janet's new movie -/
def new_cost_per_minute : ℝ := 2 * previous_cost_per_minute

/-- The total cost to film Janet's new movie -/
def new_movie_total_cost : ℝ := 1920

theorem janet_movie_cost : 
  new_movie_length * new_cost_per_minute = new_movie_total_cost :=
by sorry

end janet_movie_cost_l3474_347446


namespace product_of_repeating_decimals_division_of_repeating_decimals_l3474_347490

-- Define the repeating decimals
def repeating_decimal_18 : ℚ := 2 / 11
def repeating_decimal_36 : ℚ := 4 / 11

-- Theorem for the product
theorem product_of_repeating_decimals :
  repeating_decimal_18 * repeating_decimal_36 = 8 / 121 := by
  sorry

-- Theorem for the division
theorem division_of_repeating_decimals :
  repeating_decimal_18 / repeating_decimal_36 = 1 / 2 := by
  sorry

end product_of_repeating_decimals_division_of_repeating_decimals_l3474_347490


namespace carls_cupcake_goal_l3474_347450

/-- Carl's cupcake selling problem -/
theorem carls_cupcake_goal (goal : ℕ) (days : ℕ) (payment : ℕ) (cupcakes_per_day : ℕ) : 
  goal = 96 → days = 2 → payment = 24 → cupcakes_per_day * days = goal + payment → cupcakes_per_day = 60 := by
  sorry

end carls_cupcake_goal_l3474_347450


namespace alpha_beta_sum_l3474_347487

theorem alpha_beta_sum (α β : ℝ) 
  (h1 : α^3 - 3*α^2 + 5*α = 1) 
  (h2 : β^3 - 3*β^2 + 5*β = 5) : 
  α + β = 2 := by
sorry

end alpha_beta_sum_l3474_347487


namespace simple_interest_problem_l3474_347460

/-- Given a principal amount P and an interest rate R (as a percentage),
    if the amount after 2 years is 780 and after 7 years is 1020,
    then the principal amount P is 684. -/
theorem simple_interest_problem (P R : ℚ) : 
  P + (P * R * 2) / 100 = 780 →
  P + (P * R * 7) / 100 = 1020 →
  P = 684 := by
  sorry

end simple_interest_problem_l3474_347460


namespace vector_magnitude_l3474_347412

theorem vector_magnitude (a b : ℝ × ℝ × ℝ) : 
  (‖a‖ = 2) → (‖b‖ = 3) → (‖a + b‖ = 3) → ‖a + 2 • b‖ = 4 * Real.sqrt 2 := by
  sorry

end vector_magnitude_l3474_347412


namespace thirteen_to_six_mod_eight_l3474_347409

theorem thirteen_to_six_mod_eight (m : ℕ) : 
  13^6 ≡ m [ZMOD 8] → 0 ≤ m → m < 8 → m = 1 := by
  sorry

end thirteen_to_six_mod_eight_l3474_347409


namespace collinear_probability_5x4_l3474_347480

/-- A rectangular array of dots -/
structure DotArray :=
  (rows : ℕ)
  (cols : ℕ)

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of collinear sets of 4 dots in a DotArray -/
def collinearSets (arr : DotArray) : ℕ := sorry

/-- The probability of choosing 4 collinear dots from a DotArray -/
def collinearProbability (arr : DotArray) : ℚ :=
  (collinearSets arr : ℚ) / choose (arr.rows * arr.cols) 4

/-- The main theorem -/
theorem collinear_probability_5x4 :
  collinearProbability ⟨5, 4⟩ = 2 / 4845 := by sorry

end collinear_probability_5x4_l3474_347480


namespace binomial_prob_l3474_347497

/-- A random variable following a binomial distribution B(2,p) -/
def X (p : ℝ) : Type := Unit

/-- The probability that X is greater than or equal to 1 -/
def prob_X_geq_1 (p : ℝ) : ℝ := 1 - (1 - p)^2

/-- The theorem stating that if P(X ≥ 1) = 5/9 for X ~ B(2,p), then p = 1/3 -/
theorem binomial_prob (p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 1) :
  prob_X_geq_1 p = 5/9 → p = 1/3 := by
  sorry

end binomial_prob_l3474_347497


namespace number_of_violas_proof_l3474_347415

/-- The number of violas in a music store, given the following conditions:
  * There are 800 cellos in the store
  * There are 70 cello-viola pairs made from the same tree
  * The probability of randomly choosing a cello-viola pair from the same tree is 0.00014583333333333335
-/
def number_of_violas : ℕ :=
  let total_cellos : ℕ := 800
  let same_tree_pairs : ℕ := 70
  let probability : ℚ := 70 / (800 * 600)
  600

theorem number_of_violas_proof :
  let total_cellos : ℕ := 800
  let same_tree_pairs : ℕ := 70
  let probability : ℚ := 70 / (800 * 600)
  number_of_violas = 600 := by
  sorry

end number_of_violas_proof_l3474_347415


namespace merchant_profit_l3474_347493

theorem merchant_profit (C S : ℝ) (h : 17 * C = 16 * S) :
  (S - C) / C * 100 = 6.25 := by
  sorry

end merchant_profit_l3474_347493


namespace smallest_factor_smallest_factor_exists_l3474_347453

theorem smallest_factor (n : ℕ) : n > 0 ∧ 936 * n % 2^5 = 0 ∧ 936 * n % 3^3 = 0 ∧ 936 * n % 13^2 = 0 → n ≥ 468 := by
  sorry

theorem smallest_factor_exists : ∃ n : ℕ, n > 0 ∧ 936 * n % 2^5 = 0 ∧ 936 * n % 3^3 = 0 ∧ 936 * n % 13^2 = 0 ∧ n = 468 := by
  sorry

end smallest_factor_smallest_factor_exists_l3474_347453


namespace solve_equation_l3474_347458

theorem solve_equation : ∀ x : ℝ, 2 * 3 * 4 = 6 * x → x = 4 := by
  sorry

end solve_equation_l3474_347458


namespace solution_y_percent_a_l3474_347425

/-- Represents a chemical solution with a given percentage of chemical A -/
structure Solution where
  percent_a : ℝ
  h_percent_range : 0 ≤ percent_a ∧ percent_a ≤ 1

/-- Represents a mixture of two solutions -/
structure Mixture where
  solution_x : Solution
  solution_y : Solution
  proportion_x : ℝ
  h_proportion_range : 0 ≤ proportion_x ∧ proportion_x ≤ 1

/-- Calculates the percentage of chemical A in a mixture -/
def mixture_percent_a (m : Mixture) : ℝ :=
  m.proportion_x * m.solution_x.percent_a + (1 - m.proportion_x) * m.solution_y.percent_a

theorem solution_y_percent_a (x : Solution) (y : Solution) (m : Mixture) 
  (h_x : x.percent_a = 0.3)
  (h_m : m.solution_x = x ∧ m.solution_y = y ∧ m.proportion_x = 0.8)
  (h_mixture : mixture_percent_a m = 0.32) :
  y.percent_a = 0.4 := by
  sorry


end solution_y_percent_a_l3474_347425


namespace lucy_flour_purchase_l3474_347496

/-- Calculates the amount of flour needed to replenish stock --/
def flour_to_buy (initial : ℕ) (used : ℕ) (full_bag : ℕ) : ℕ :=
  let remaining := initial - used
  let after_spill := remaining / 2
  full_bag - after_spill

/-- Theorem: Given the initial conditions, Lucy needs to buy 370g of flour --/
theorem lucy_flour_purchase :
  flour_to_buy 500 240 500 = 370 := by
  sorry

end lucy_flour_purchase_l3474_347496


namespace playground_to_landscape_ratio_l3474_347437

/-- A rectangular landscape with a playground -/
structure Landscape where
  length : ℝ
  breadth : ℝ
  playground_area : ℝ
  length_breadth_relation : length = 4 * breadth
  length_value : length = 120
  playground_size : playground_area = 1200

/-- The ratio of playground area to total landscape area is 1:3 -/
theorem playground_to_landscape_ratio (L : Landscape) :
  L.playground_area / (L.length * L.breadth) = 1 / 3 := by
  sorry

end playground_to_landscape_ratio_l3474_347437


namespace triangle_area_l3474_347438

theorem triangle_area (a b : ℝ) (cos_theta : ℝ) : 
  a = 3 → 
  b = 5 → 
  5 * cos_theta^2 - 7 * cos_theta - 6 = 0 → 
  (1/2) * a * b * Real.sqrt (1 - cos_theta^2) = 6 := by
  sorry

end triangle_area_l3474_347438


namespace john_uber_profit_l3474_347469

def uber_earnings : ℕ := 30000
def car_purchase_price : ℕ := 18000
def car_trade_in_value : ℕ := 6000

theorem john_uber_profit :
  uber_earnings - (car_purchase_price - car_trade_in_value) = 18000 :=
by sorry

end john_uber_profit_l3474_347469


namespace stating_min_swaps_equals_num_pairs_l3474_347429

/-- Represents the number of volumes in the encyclopedia --/
def n : ℕ := 30

/-- Represents a swap operation between adjacent volumes --/
def Swap : Type := Unit

/-- 
Represents the minimum number of swap operations required to guarantee 
correct ordering of n volumes from any initial arrangement 
--/
def minSwaps (n : ℕ) : ℕ := sorry

/-- Calculates the number of possible pairs from n elements --/
def numPairs (n : ℕ) : ℕ := n * (n - 1) / 2

/-- 
Theorem stating that the minimum number of swaps required is equal to 
the number of possible pairs of volumes
--/
theorem min_swaps_equals_num_pairs : 
  minSwaps n = numPairs n := by sorry

end stating_min_swaps_equals_num_pairs_l3474_347429


namespace students_without_A_l3474_347464

theorem students_without_A (total : ℕ) (history : ℕ) (math : ℕ) (science : ℕ)
  (history_math : ℕ) (history_science : ℕ) (math_science : ℕ) (all_three : ℕ) :
  total = 45 →
  history = 11 →
  math = 16 →
  science = 9 →
  history_math = 5 →
  history_science = 3 →
  math_science = 4 →
  all_three = 2 →
  total - (history + math + science - history_math - history_science - math_science + all_three) = 19 :=
by sorry

end students_without_A_l3474_347464


namespace investment_gain_percentage_l3474_347489

/-- Calculate the overall gain percentage for an investment portfolio --/
theorem investment_gain_percentage
  (stock_initial : ℝ)
  (artwork_initial : ℝ)
  (crypto_initial : ℝ)
  (stock_return : ℝ)
  (artwork_return : ℝ)
  (crypto_return_rub : ℝ)
  (rub_to_rs_rate : ℝ)
  (artwork_tax_rate : ℝ)
  (crypto_fee_rate : ℝ)
  (h1 : stock_initial = 5000)
  (h2 : artwork_initial = 10000)
  (h3 : crypto_initial = 15000)
  (h4 : stock_return = 6000)
  (h5 : artwork_return = 12000)
  (h6 : crypto_return_rub = 17000)
  (h7 : rub_to_rs_rate = 1.03)
  (h8 : artwork_tax_rate = 0.05)
  (h9 : crypto_fee_rate = 0.02) :
  let total_initial := stock_initial + artwork_initial + crypto_initial
  let artwork_net_return := artwork_return * (1 - artwork_tax_rate)
  let crypto_return_rs := crypto_return_rub * rub_to_rs_rate
  let crypto_net_return := crypto_return_rs * (1 - crypto_fee_rate)
  let total_return := stock_return + artwork_net_return + crypto_net_return
  let gain_percentage := (total_return - total_initial) / total_initial * 100
  ∃ ε > 0, |gain_percentage - 15.20| < ε :=
by
  sorry

end investment_gain_percentage_l3474_347489


namespace transportation_theorem_l3474_347400

/-- Represents a type of transportation with its quantity and number of wheels -/
structure Transportation where
  name : String
  quantity : Nat
  wheels : Nat

/-- Calculates the total number of wheels for a given transportation -/
def totalWheels (t : Transportation) : Nat :=
  t.quantity * t.wheels

/-- Calculates the total number of wheels for a list of transportations -/
def sumWheels (ts : List Transportation) : Nat :=
  ts.foldl (fun acc t => acc + totalWheels t) 0

/-- Calculates the total quantity of all transportations -/
def totalQuantity (ts : List Transportation) : Nat :=
  ts.foldl (fun acc t => acc + t.quantity) 0

/-- Calculates the quantity of bicycles and tricycles -/
def bikeAndTricycleCount (ts : List Transportation) : Nat :=
  ts.filter (fun t => t.name = "bicycle" || t.name = "tricycle")
    |>.foldl (fun acc t => acc + t.quantity) 0

theorem transportation_theorem (observations : List Transportation) 
  (h1 : observations = [
    ⟨"car", 15, 4⟩, 
    ⟨"bicycle", 3, 2⟩, 
    ⟨"pickup truck", 8, 4⟩, 
    ⟨"tricycle", 1, 3⟩, 
    ⟨"motorcycle", 4, 2⟩, 
    ⟨"skateboard", 2, 4⟩, 
    ⟨"unicycle", 1, 1⟩
  ]) : 
  sumWheels observations = 118 ∧ 
  (bikeAndTricycleCount observations : Rat) / (totalQuantity observations : Rat) = 4/34 := by
  sorry

end transportation_theorem_l3474_347400


namespace mountain_climb_time_l3474_347452

/-- Proves that the time to go up the mountain is 2 hours given the specified conditions -/
theorem mountain_climb_time 
  (total_time : ℝ) 
  (uphill_speed downhill_speed : ℝ) 
  (route_difference : ℝ) :
  total_time = 4 →
  uphill_speed = 3 →
  downhill_speed = 4 →
  route_difference = 2 →
  ∃ (uphill_time : ℝ),
    uphill_time = 2 ∧
    ∃ (downhill_time uphill_distance downhill_distance : ℝ),
      uphill_time + downhill_time = total_time ∧
      uphill_distance / uphill_speed = uphill_time ∧
      downhill_distance / downhill_speed = downhill_time ∧
      downhill_distance = uphill_distance + route_difference :=
by sorry

end mountain_climb_time_l3474_347452


namespace compound_composition_l3474_347491

/-- Represents the number of atoms of each element in the compound -/
structure Compound where
  h : ℕ
  c : ℕ
  o : ℕ

/-- Atomic weights of elements in g/mol -/
def atomic_weight (element : String) : ℝ :=
  match element with
  | "H" => 1
  | "C" => 12
  | "O" => 16
  | _ => 0

/-- Calculate the molecular weight of a compound -/
def molecular_weight (comp : Compound) : ℝ :=
  comp.h * atomic_weight "H" + comp.c * atomic_weight "C" + comp.o * atomic_weight "O"

/-- The main theorem to prove -/
theorem compound_composition :
  ∃ (comp : Compound), comp.h = 2 ∧ comp.o = 3 ∧ molecular_weight comp = 62 ∧ comp.c = 1 :=
by sorry

end compound_composition_l3474_347491


namespace inequality_system_solution_l3474_347439

theorem inequality_system_solution (a b : ℝ) :
  (∀ x : ℝ, -1 < x ∧ x < 1 ↔ x - a > 2 ∧ 2*x - b < 0) →
  a^(-b) = 1/9 := by
sorry

end inequality_system_solution_l3474_347439


namespace square_sum_equals_nine_billion_four_million_l3474_347433

theorem square_sum_equals_nine_billion_four_million : (300000 : ℕ)^2 + (20000 : ℕ)^2 = 9004000000 := by
  sorry

end square_sum_equals_nine_billion_four_million_l3474_347433


namespace baseball_card_ratio_l3474_347448

/-- Proves the ratio of cards Maria took to initial cards is 8:5 --/
theorem baseball_card_ratio : 
  ∀ (initial final maria_taken peter_given : ℕ),
  initial = 15 →
  peter_given = 1 →
  final = 18 →
  maria_taken = 3 * (initial - peter_given) - final →
  (maria_taken : ℚ) / initial = 8 / 5 := by
    sorry

end baseball_card_ratio_l3474_347448


namespace mixed_number_properties_l3474_347449

/-- Represents a mixed number as a pair of integers (whole, numerator, denominator) -/
structure MixedNumber where
  whole : ℤ
  numerator : ℕ
  denominator : ℕ
  h_pos : denominator > 0
  h_proper : numerator < denominator

/-- The smallest composite number -/
def smallest_composite : ℕ := 4

/-- Converts a mixed number to a rational number -/
def mixed_to_rational (m : MixedNumber) : ℚ :=
  m.whole + (m.numerator : ℚ) / m.denominator

theorem mixed_number_properties (m : MixedNumber) 
  (h_m : m = ⟨3, 2, 7, by norm_num, by norm_num⟩) : 
  ∃ (fractional_unit : ℚ) (num_units : ℕ) (units_to_add : ℕ),
    fractional_unit = 1 / 7 ∧ 
    num_units = 23 ∧
    units_to_add = 5 ∧
    mixed_to_rational m = num_units * fractional_unit ∧
    mixed_to_rational m + units_to_add * fractional_unit = smallest_composite := by
  sorry

end mixed_number_properties_l3474_347449


namespace weight_of_three_moles_CaI2_l3474_347457

/-- The atomic weight of Calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of Iodine in g/mol -/
def atomic_weight_I : ℝ := 126.90

/-- The molecular weight of CaI2 in g/mol -/
def molecular_weight_CaI2 : ℝ := atomic_weight_Ca + 2 * atomic_weight_I

/-- The weight of n moles of CaI2 in grams -/
def weight_CaI2 (n : ℝ) : ℝ := n * molecular_weight_CaI2

theorem weight_of_three_moles_CaI2 : 
  weight_CaI2 3 = 881.64 := by sorry

end weight_of_three_moles_CaI2_l3474_347457


namespace problem_statement_l3474_347435

theorem problem_statement (x : ℝ) : 
  let a := x^2 - 1
  let b := 2*x + 2
  (a + b ≥ 0) ∧ (max a b ≥ 0) := by
  sorry

end problem_statement_l3474_347435


namespace triangle_rectangle_ratio_l3474_347482

theorem triangle_rectangle_ratio : 
  ∀ (t w l : ℝ),
  (3 * t = 24) →  -- Perimeter of equilateral triangle
  (2 * l + 2 * w = 24) →  -- Perimeter of rectangle
  (l = 2 * w) →  -- Length is twice the width
  (t / w = 2) :=
by sorry

end triangle_rectangle_ratio_l3474_347482


namespace smallest_even_sum_fourteen_is_achievable_l3474_347459

def S : Finset Int := {8, -4, 3, 27, 10}

def isValidSum (x y z : Int) : Prop :=
  x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ Even (x + y + z)

theorem smallest_even_sum :
  ∀ x y z, isValidSum x y z → x + y + z ≥ 14 :=
by sorry

theorem fourteen_is_achievable :
  ∃ x y z, isValidSum x y z ∧ x + y + z = 14 :=
by sorry

end smallest_even_sum_fourteen_is_achievable_l3474_347459


namespace final_salt_concentration_l3474_347465

/-- Represents the volume of salt solution in arbitrary units -/
def initialVolume : ℝ := 30

/-- Represents the initial concentration of salt in the solution -/
def initialConcentration : ℝ := 0.15

/-- Represents the volume ratio of the large ball -/
def largeBallRatio : ℝ := 10

/-- Represents the volume ratio of the medium ball -/
def mediumBallRatio : ℝ := 5

/-- Represents the volume ratio of the small ball -/
def smallBallRatio : ℝ := 3

/-- Represents the overflow percentage caused by the small ball -/
def overflowPercentage : ℝ := 0.1

/-- Theorem stating that the final salt concentration is 10% -/
theorem final_salt_concentration :
  let totalOverflow := smallBallRatio + mediumBallRatio + largeBallRatio
  let remainingVolume := initialVolume - totalOverflow
  let initialSaltAmount := initialVolume * initialConcentration
  (initialSaltAmount / initialVolume) * 100 = 10 := by
  sorry


end final_salt_concentration_l3474_347465


namespace prob_not_shaded_is_500_1001_l3474_347440

/-- Represents a 2 by 1001 rectangle with middle squares shaded -/
structure ShadedRectangle where
  width : ℕ := 2
  length : ℕ := 1001
  middle_shaded : ℕ := (length + 1) / 2

/-- Calculates the total number of rectangles in the figure -/
def total_rectangles (r : ShadedRectangle) : ℕ :=
  r.width * (r.length * (r.length + 1)) / 2

/-- Calculates the number of rectangles that include a shaded square -/
def shaded_rectangles (r : ShadedRectangle) : ℕ :=
  r.width * r.middle_shaded * (r.length - r.middle_shaded + 1)

/-- The probability of choosing a rectangle that doesn't include a shaded square -/
def prob_not_shaded (r : ShadedRectangle) : ℚ :=
  1 - (shaded_rectangles r : ℚ) / (total_rectangles r : ℚ)

theorem prob_not_shaded_is_500_1001 (r : ShadedRectangle) :
  prob_not_shaded r = 500 / 1001 := by
  sorry

end prob_not_shaded_is_500_1001_l3474_347440


namespace alice_next_birthday_age_l3474_347406

theorem alice_next_birthday_age :
  ∀ (a b c : ℝ),
  a = 1.25 * b →                -- Alice is 25% older than Bob
  b = 0.7 * c →                 -- Bob is 30% younger than Carlos
  a + b + c = 30 →              -- Sum of their ages is 30 years
  ⌊a⌋ + 1 = 11 :=               -- Alice's age on her next birthday
by
  sorry


end alice_next_birthday_age_l3474_347406


namespace sin_arccos_tan_arcsin_product_one_l3474_347475

theorem sin_arccos_tan_arcsin_product_one :
  ∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧
  x₁ ≠ x₂ ∧
  (∀ (x : ℝ), x > 0 → Real.sin (Real.arccos (Real.tan (Real.arcsin x))) = x → (x = x₁ ∨ x = x₂)) ∧
  x₁ * x₂ = 1 :=
sorry

end sin_arccos_tan_arcsin_product_one_l3474_347475


namespace green_shirt_pairs_l3474_347498

theorem green_shirt_pairs (red_students green_students total_students total_pairs red_red_pairs : ℕ) 
  (h1 : red_students = 63)
  (h2 : green_students = 69)
  (h3 : total_students = red_students + green_students)
  (h4 : total_pairs = 66)
  (h5 : red_red_pairs = 26)
  (h6 : total_students = 2 * total_pairs) :
  green_students - (total_pairs - red_red_pairs - (red_students - 2 * red_red_pairs)) = 29 := by
  sorry

end green_shirt_pairs_l3474_347498


namespace solve_for_A_l3474_347478

theorem solve_for_A (x₁ x₂ A : ℂ) : 
  x₁ ≠ x₂ →
  x₁ * (x₁ + 1) = A →
  x₂ * (x₂ + 1) = A →
  x₁^4 + 3*x₁^3 + 5*x₁ = x₂^4 + 3*x₂^3 + 5*x₂ →
  A = -7 := by sorry

end solve_for_A_l3474_347478


namespace team_loss_percentage_l3474_347404

/-- Represents the ratio of games won to games lost -/
def winLossRatio : Rat := 7 / 3

/-- The total number of games played -/
def totalGames : ℕ := 50

/-- Calculates the percentage of games lost -/
def percentLost : ℚ :=
  let gamesLost := totalGames / (1 + winLossRatio)
  (gamesLost / totalGames) * 100

theorem team_loss_percentage :
  ⌊percentLost⌋ = 30 :=
sorry

end team_loss_percentage_l3474_347404


namespace max_x_squared_minus_y_squared_l3474_347488

theorem max_x_squared_minus_y_squared (x y : ℝ) 
  (h : 2 * (x^3 + y^3) = x^2 + y^2) : 
  ∀ a b : ℝ, 2 * (a^3 + b^3) = a^2 + b^2 → x^2 - y^2 ≤ a^2 - b^2 := by
sorry

end max_x_squared_minus_y_squared_l3474_347488


namespace covered_number_is_eight_l3474_347424

/-- A circular arrangement of 15 numbers -/
def CircularArrangement := Fin 15 → ℕ

/-- The property that the sum of any six consecutive numbers is 50 -/
def SumProperty (arr : CircularArrangement) : Prop :=
  ∀ i : Fin 15, (arr i + arr (i + 1) + arr (i + 2) + arr (i + 3) + arr (i + 4) + arr (i + 5)) = 50

/-- The property that two adjacent numbers are 7 and 10 with a number between them -/
def AdjacentProperty (arr : CircularArrangement) : Prop :=
  ∃ i : Fin 15, arr i = 7 ∧ arr (i + 2) = 10

theorem covered_number_is_eight (arr : CircularArrangement) 
  (h1 : SumProperty arr) (h2 : AdjacentProperty arr) : 
  ∃ i : Fin 15, arr i = 7 ∧ arr (i + 1) = 8 ∧ arr (i + 2) = 10 := by
  sorry

end covered_number_is_eight_l3474_347424


namespace arithmetic_sequence_count_l3474_347418

theorem arithmetic_sequence_count (n : ℕ) (m : ℕ) (k : ℕ) (h1 : n = 2014) (h2 : m = 315) (h3 : k = 5490) :
  (∃ (sequences : Finset (Finset ℕ)),
    sequences.card = k ∧
    (∀ seq ∈ sequences,
      seq.card = m ∧
      (∃ d : ℕ, d > 0 ∧ d ≤ 6 ∧
        (∀ i j : ℕ, i < j → i ∈ seq → j ∈ seq →
          ∃ k : ℕ, j - i = k * d)) ∧
      1 ∈ seq ∧
      (∀ x ∈ seq, 1 ≤ x ∧ x ≤ n))) :=
by sorry

end arithmetic_sequence_count_l3474_347418


namespace line_slope_intercept_product_l3474_347466

theorem line_slope_intercept_product (m b : ℚ) : 
  m > 0 → b < 0 → m = 3/4 → b = -2/3 → -1 < m * b ∧ m * b < 0 := by
  sorry

end line_slope_intercept_product_l3474_347466


namespace amount_of_c_l3474_347455

theorem amount_of_c (A B C : ℝ) 
  (h1 : A + B + C = 600) 
  (h2 : A + C = 250) 
  (h3 : B + C = 450) : 
  C = 100 := by
sorry

end amount_of_c_l3474_347455


namespace circle_y_axis_intersection_sum_l3474_347441

theorem circle_y_axis_intersection_sum : 
  ∀ (x y : ℝ), 
  ((x + 8)^2 + (y - 5)^2 = 13^2) →  -- Circle equation
  (x = 0) →                        -- Points on y-axis
  ∃ (y1 y2 : ℝ),
    ((0 + 8)^2 + (y1 - 5)^2 = 13^2) ∧
    ((0 + 8)^2 + (y2 - 5)^2 = 13^2) ∧
    y1 + y2 = 10 :=
by sorry

end circle_y_axis_intersection_sum_l3474_347441


namespace speeding_ticket_percentage_l3474_347456

/-- The percentage of motorists who exceed the speed limit -/
def exceed_limit_percent : ℝ := 20

/-- The percentage of speeding motorists who do not receive tickets -/
def no_ticket_percent : ℝ := 50

/-- The percentage of motorists who receive speeding tickets -/
def receive_ticket_percent : ℝ := 10

theorem speeding_ticket_percentage :
  receive_ticket_percent = exceed_limit_percent * (100 - no_ticket_percent) / 100 := by
  sorry

end speeding_ticket_percentage_l3474_347456


namespace union_equals_A_l3474_347431

-- Define the sets A and B
def A : Set ℝ := {x | x * (x - 1) ≤ 0}
def B (a : ℝ) : Set ℝ := {x | Real.log x ≤ a}

-- State the theorem
theorem union_equals_A (a : ℝ) : A ∪ B a = A ↔ a = 0 := by sorry

end union_equals_A_l3474_347431


namespace quadratic_sum_l3474_347410

/-- A quadratic function with specified properties -/
structure QuadraticFunction where
  d : ℝ
  e : ℝ
  f : ℝ
  vertex_x : ℝ
  vertex_y : ℝ
  point_x : ℝ
  point_y : ℝ
  vertex_condition : vertex_y = d * vertex_x^2 + e * vertex_x + f
  point_condition : point_y = d * point_x^2 + e * point_x + f
  is_vertex : ∀ x : ℝ, d * x^2 + e * x + f ≥ vertex_y

/-- Theorem: For a quadratic function with given properties, d + e + 2f = 19 -/
theorem quadratic_sum (g : QuadraticFunction) 
  (h1 : g.vertex_x = -2) 
  (h2 : g.vertex_y = 3) 
  (h3 : g.point_x = 0) 
  (h4 : g.point_y = 7) : 
  g.d + g.e + 2 * g.f = 19 := by
  sorry

end quadratic_sum_l3474_347410


namespace inequality_system_solution_l3474_347407

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x : ℝ, (0 ≤ x ∧ x < 1) ↔ (x/2 + a ≥ 2 ∧ 2*x - b < 3)) → 
  a + b = 1 := by
sorry

end inequality_system_solution_l3474_347407


namespace problem_solution_l3474_347462

/-- The base-74 representation of the number in the problem -/
def base_74_num : ℕ := 235935623

/-- Converts the base-74 number to its decimal equivalent modulo 15 -/
def decimal_mod_15 : ℕ := base_74_num % 15

theorem problem_solution (a : ℤ) (h1 : 0 ≤ a) (h2 : a ≤ 14) 
  (h3 : (decimal_mod_15 - a) % 15 = 0) : a = 0 := by
  sorry

end problem_solution_l3474_347462


namespace f_g_derivatives_neg_l3474_347494

-- Define f and g as functions from ℝ to ℝ
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
variable (hf : ∀ x, f (-x) = -f x)
variable (hg : ∀ x, g (-x) = g x)

-- Define the derivative properties for x > 0
variable (hf_deriv_pos : ∀ x, x > 0 → deriv f x > 0)
variable (hg_deriv_pos : ∀ x, x > 0 → deriv g x > 0)

-- State the theorem
theorem f_g_derivatives_neg (x : ℝ) (hx : x < 0) : 
  deriv f x > 0 ∧ deriv g x < 0 :=
sorry

end f_g_derivatives_neg_l3474_347494


namespace line_through_point_and_trisection_l3474_347468

/-- The line passing through (2,3) and one of the trisection points of the line segment
    joining (1,2) and (7,-4) has the equation 4x - 9y + 15 = 0 -/
theorem line_through_point_and_trisection :
  ∃ (t : ℝ) (x y : ℝ),
    -- Define the trisection point
    x = 1 + 2 * t * (7 - 1) ∧
    y = 2 + 2 * t * (-4 - 2) ∧
    0 ≤ t ∧ t ≤ 1 ∧
    -- The trisection point is on the line
    4 * x - 9 * y + 15 = 0 ∧
    -- The point (2,3) is on the line
    4 * 2 - 9 * 3 + 15 = 0 :=
sorry

end line_through_point_and_trisection_l3474_347468


namespace statement_1_false_statement_2_false_statement_3_false_statement_4_true_l3474_347463

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Line → Prop)
variable (parallelLP : Line → Plane → Prop)
variable (parallelPP : Plane → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

variable (m n : Line)
variable (α β : Plane)

-- Statement ①
theorem statement_1_false :
  ¬(parallelLP m α → parallelLP n α → parallel m n) :=
sorry

-- Statement ②
theorem statement_2_false :
  ¬(subset m α → subset n α → parallelLP m β → parallelLP n β → parallelPP α β) :=
sorry

-- Statement ③
theorem statement_3_false :
  ¬(perpendicular α β → subset m α → perpendicularLP m β) :=
sorry

-- Statement ④
theorem statement_4_true :
  perpendicular α β → perpendicularLP m β → ¬(subset m α) → parallelLP m α :=
sorry

end statement_1_false_statement_2_false_statement_3_false_statement_4_true_l3474_347463


namespace fraction_sum_simplification_l3474_347432

theorem fraction_sum_simplification :
  18 / 462 + 35 / 77 = 38 / 77 := by
sorry

end fraction_sum_simplification_l3474_347432


namespace quadratic_inequality_solution_set_l3474_347421

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo 3 4 = {x | x^2 + a*x + b < 0}) : 
  {x : ℝ | b*x^2 + a*x + 1 > 0} = Set.Iic (1/4) ∪ Set.Ici (1/3) :=
sorry

end quadratic_inequality_solution_set_l3474_347421


namespace equal_commission_l3474_347481

/-- The list price of the item -/
def list_price : ℝ := 34

/-- Alice's selling price -/
def alice_price (x : ℝ) : ℝ := x - 15

/-- Bob's selling price -/
def bob_price (x : ℝ) : ℝ := x - 25

/-- Alice's commission rate -/
def alice_rate : ℝ := 0.12

/-- Bob's commission rate -/
def bob_rate : ℝ := 0.25

/-- Alice's commission -/
def alice_commission (x : ℝ) : ℝ := alice_rate * alice_price x

/-- Bob's commission -/
def bob_commission (x : ℝ) : ℝ := bob_rate * bob_price x

theorem equal_commission :
  alice_commission list_price = bob_commission list_price :=
sorry

end equal_commission_l3474_347481


namespace infinitely_many_non_prime_n4_plus_k_l3474_347479

theorem infinitely_many_non_prime_n4_plus_k :
  ∀ N : ℕ, ∃ k : ℕ, k > N ∧ ∀ n : ℕ, ¬ Prime (n^4 + k) := by
  sorry

end infinitely_many_non_prime_n4_plus_k_l3474_347479


namespace total_participants_is_260_l3474_347499

/-- Represents the voting scenario for a school dance -/
structure VotingScenario where
  initial_oct22_percent : ℝ
  initial_oct29_percent : ℝ
  additional_votes : ℕ
  final_oct29_percent : ℝ

/-- Calculates the total number of participants in the voting -/
def total_participants (scenario : VotingScenario) : ℕ :=
  sorry

/-- Theorem stating that the total number of participants is 260 -/
theorem total_participants_is_260 (scenario : VotingScenario) 
  (h1 : scenario.initial_oct22_percent = 0.35)
  (h2 : scenario.initial_oct29_percent = 0.65)
  (h3 : scenario.additional_votes = 80)
  (h4 : scenario.final_oct29_percent = 0.45) :
  total_participants scenario = 260 := by
  sorry

end total_participants_is_260_l3474_347499


namespace negative_difference_equality_l3474_347473

theorem negative_difference_equality (a b : ℝ) : -(a - b) = -a + b := by
  sorry

end negative_difference_equality_l3474_347473


namespace linear_coefficient_is_zero_l3474_347411

/-- The coefficient of the linear term in the standard form of (2 - x)(3x + 4) = 2x - 1 is 0 -/
theorem linear_coefficient_is_zero : 
  let f : ℝ → ℝ := λ x => (2 - x) * (3 * x + 4) - (2 * x - 1)
  ∃ a c : ℝ, ∀ x, f x = -3 * x^2 + 0 * x + c :=
sorry

end linear_coefficient_is_zero_l3474_347411


namespace isosceles_right_triangle_area_and_perimeter_l3474_347428

-- Define an isosceles right triangle with rational hypotenuse
structure IsoscelesRightTriangle where
  hypotenuse : ℚ
  hypotenuse_positive : hypotenuse > 0

-- Define the area of the triangle
def area (t : IsoscelesRightTriangle) : ℚ :=
  t.hypotenuse ^ 2 / 4

-- Define the perimeter of the triangle
noncomputable def perimeter (t : IsoscelesRightTriangle) : ℝ :=
  t.hypotenuse * (2 + Real.sqrt 2)

-- Theorem statement
theorem isosceles_right_triangle_area_and_perimeter (t : IsoscelesRightTriangle) :
  (∃ q : ℚ, area t = q) ∧ (∀ q : ℚ, perimeter t ≠ q) := by
  sorry

end isosceles_right_triangle_area_and_perimeter_l3474_347428


namespace room_painting_time_l3474_347461

theorem room_painting_time 
  (alice_rate : ℝ) 
  (bob_rate : ℝ) 
  (carla_rate : ℝ) 
  (t : ℝ) 
  (h_alice : alice_rate = 1 / 6) 
  (h_bob : bob_rate = 1 / 8) 
  (h_carla : carla_rate = 1 / 12) 
  (h_combined_work : (alice_rate + bob_rate + carla_rate) * t = 1) : 
  (1 / 6 + 1 / 8 + 1 / 12) * t = 1 := by
  sorry

end room_painting_time_l3474_347461


namespace arithmetic_sequence_max_sum_l3474_347401

/-- An arithmetic sequence with positive first term and a_3/a_4 = 7/5 reaches maximum sum at n = 6 -/
theorem arithmetic_sequence_max_sum (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence
  a 1 > 0 →  -- positive first term
  a 3 / a 4 = 7 / 5 →  -- given ratio
  ∃ S : ℕ → ℝ, ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2 ∧  -- sum formula
  (∀ m, S m ≤ S 6) :=  -- maximum sum at n = 6
by sorry

end arithmetic_sequence_max_sum_l3474_347401


namespace product_121_54_l3474_347467

theorem product_121_54 : 121 * 54 = 6534 := by
  sorry

end product_121_54_l3474_347467


namespace unique_digit_subtraction_l3474_347476

theorem unique_digit_subtraction (A B C D : Nat) :
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 →
  1000 * A + 100 * B + 10 * C + D - 989 = 109 →
  1000 * A + 100 * B + 10 * C + D = 1908 := by
sorry

end unique_digit_subtraction_l3474_347476


namespace product_mod_25_l3474_347445

theorem product_mod_25 : (43 * 67 * 92) % 25 = 2 := by
  sorry

#check product_mod_25

end product_mod_25_l3474_347445


namespace isosceles_triangle_areas_sum_l3474_347402

/-- Given a 6-8-10 right triangle, prove that the sum of the areas of right isosceles triangles
    constructed on the two shorter sides is equal to the area of the right isosceles triangle
    constructed on the hypotenuse. -/
theorem isosceles_triangle_areas_sum (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10)
  (h4 : a^2 + b^2 = c^2) : (1/2 * a^2) + (1/2 * b^2) = 1/2 * c^2 := by
  sorry

#check isosceles_triangle_areas_sum

end isosceles_triangle_areas_sum_l3474_347402


namespace place_value_ratio_l3474_347477

/-- The number we're analyzing -/
def number : ℚ := 25684.2057

/-- The place value of the digit 6 in the number -/
def place_value_6 : ℚ := 1000

/-- The place value of the digit 2 in the number -/
def place_value_2 : ℚ := 0.1

/-- Theorem stating the relationship between the place values -/
theorem place_value_ratio : place_value_6 / place_value_2 = 10000 := by
  sorry

end place_value_ratio_l3474_347477


namespace combined_savings_equals_individual_savings_l3474_347470

-- Define the regular price of a window
def regular_price : ℕ := 120

-- Define the offer: for every 5 windows purchased, 2 are free
def offer (n : ℕ) : ℕ := (n / 5) * 2

-- Calculate the cost for a given number of windows with the offer
def cost_with_offer (n : ℕ) : ℕ :=
  regular_price * (n - offer n)

-- Calculate savings for a given number of windows
def savings (n : ℕ) : ℕ :=
  regular_price * n - cost_with_offer n

-- Dave's required windows
def dave_windows : ℕ := 9

-- Doug's required windows
def doug_windows : ℕ := 10

-- Combined windows
def combined_windows : ℕ := dave_windows + doug_windows

-- Theorem: Combined savings equals sum of individual savings
theorem combined_savings_equals_individual_savings :
  savings combined_windows = savings dave_windows + savings doug_windows :=
sorry

end combined_savings_equals_individual_savings_l3474_347470


namespace speed_difference_calculation_l3474_347416

/-- Calculates the difference in average speed between no traffic and heavy traffic conditions --/
theorem speed_difference_calculation (distance : ℝ) (heavy_traffic_time : ℝ) (no_traffic_time : ℝ)
  (construction_zones : ℕ) (construction_delay : ℝ) (heavy_traffic_rest_stops : ℕ)
  (no_traffic_rest_stops : ℕ) (rest_stop_duration : ℝ) :
  distance = 200 →
  heavy_traffic_time = 5 →
  no_traffic_time = 4 →
  construction_zones = 2 →
  construction_delay = 0.25 →
  heavy_traffic_rest_stops = 3 →
  no_traffic_rest_stops = 2 →
  rest_stop_duration = 1/6 →
  let heavy_traffic_driving_time := heavy_traffic_time - (construction_zones * construction_delay + heavy_traffic_rest_stops * rest_stop_duration)
  let no_traffic_driving_time := no_traffic_time - (no_traffic_rest_stops * rest_stop_duration)
  let heavy_traffic_speed := distance / heavy_traffic_driving_time
  let no_traffic_speed := distance / no_traffic_driving_time
  ∃ ε > 0, |no_traffic_speed - heavy_traffic_speed - 4.5| < ε :=
by sorry

end speed_difference_calculation_l3474_347416


namespace linear_correlation_classification_l3474_347423

-- Define the relationships
def parent_child_height : ℝ → ℝ := sorry
def cylinder_volume_radius : ℝ → ℝ := sorry
def car_weight_fuel_efficiency : ℝ → ℝ := sorry
def household_income_expenditure : ℝ → ℝ := sorry

-- Define linear correlation
def is_linearly_correlated (f : ℝ → ℝ) : Prop := 
  ∃ (a b : ℝ), ∀ x, f x = a * x + b

-- Theorem statement
theorem linear_correlation_classification :
  is_linearly_correlated parent_child_height ∧
  is_linearly_correlated car_weight_fuel_efficiency ∧
  is_linearly_correlated household_income_expenditure ∧
  ¬ is_linearly_correlated cylinder_volume_radius :=
sorry

end linear_correlation_classification_l3474_347423


namespace equal_area_rectangles_l3474_347403

/-- Given two rectangles with equal area, where one rectangle has dimensions 5 by 24 inches
    and the other has a length of 2 inches, prove that the width of the second rectangle is 60 inches. -/
theorem equal_area_rectangles (carol_length carol_width jordan_length : ℕ)
    (jordan_width : ℕ) (h1 : carol_length = 5) (h2 : carol_width = 24) (h3 : jordan_length = 2)
    (h4 : carol_length * carol_width = jordan_length * jordan_width) :
    jordan_width = 60 := by
  sorry

end equal_area_rectangles_l3474_347403


namespace f_minimum_value_range_l3474_347451

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * (x + 1)

theorem f_minimum_value_range (a : ℝ) :
  (∃ x₀ : ℝ, ∀ x : ℝ, f a x ≥ f a x₀ ∧ f a x₀ > a^2 + a) ↔ -1 < a ∧ a < 0 :=
sorry

end f_minimum_value_range_l3474_347451


namespace robot_walk_distance_l3474_347486

/-- Represents a rectangular field with robot paths -/
structure RobotField where
  length : ℕ
  width : ℕ
  path_width : ℕ
  b_distance : ℕ

/-- Calculates the total distance walked by the robot -/
def total_distance (field : RobotField) : ℕ :=
  let outer_loop := 2 * (field.length + field.width) - 1
  let second_loop := 2 * (field.length - 2 + field.width - 2) - 1
  let third_loop := 2 * (field.length - 4 + field.width - 4) - 1
  let fourth_loop := 2 * (field.length - 6 + field.width - 6) - 1
  let final_segment := field.length - field.path_width - field.b_distance
  outer_loop + second_loop + third_loop + fourth_loop + final_segment

/-- Theorem stating the total distance walked by the robot -/
theorem robot_walk_distance (field : RobotField) 
    (h1 : field.length = 16)
    (h2 : field.width = 8)
    (h3 : field.path_width = 1)
    (h4 : field.b_distance = 1) : 
  total_distance field = 154 := by
  sorry

end robot_walk_distance_l3474_347486


namespace constant_e_value_l3474_347485

theorem constant_e_value (x y e : ℝ) 
  (h1 : x / (2 * y) = 5 / e)
  (h2 : (7 * x + 4 * y) / (x - 2 * y) = 13) :
  e = 2 := by
  sorry

end constant_e_value_l3474_347485


namespace intersecting_chords_theorem_l3474_347471

theorem intersecting_chords_theorem (chord1_segment1 chord1_segment2 chord2_ratio1 chord2_ratio2 : ℝ) :
  chord1_segment1 = 12 →
  chord1_segment2 = 18 →
  chord2_ratio1 = 3 →
  chord2_ratio2 = 8 →
  ∃ (chord2_length : ℝ),
    chord2_length = chord2_ratio1 / (chord2_ratio1 + chord2_ratio2) * chord2_length +
                    chord2_ratio2 / (chord2_ratio1 + chord2_ratio2) * chord2_length ∧
    chord1_segment1 * chord1_segment2 = (chord2_ratio1 / (chord2_ratio1 + chord2_ratio2) * chord2_length) *
                                        (chord2_ratio2 / (chord2_ratio1 + chord2_ratio2) * chord2_length) →
    chord2_length = 33 := by
  sorry

end intersecting_chords_theorem_l3474_347471


namespace sqrt_equation_solution_l3474_347419

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt x + Real.sqrt (x + 6) = 12 → x = 529 / 16 := by
  sorry

end sqrt_equation_solution_l3474_347419


namespace tangent_lines_max_value_min_value_l3474_347454

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 9*x - 3

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x - 9

-- Theorem for tangent lines
theorem tangent_lines :
  ∃ (x₀ y₀ : ℝ), 
    (f' x₀ = -9 ∧ y₀ = f x₀ ∧ (y₀ = -9*x₀ - 3 ∨ y₀ = -9*x₀ + 19)) :=
sorry

-- Theorem for maximum value
theorem max_value :
  ∃ (x : ℝ), f x = 24 ∧ ∀ y, f y ≤ f x :=
sorry

-- Theorem for minimum value
theorem min_value :
  ∃ (x : ℝ), f x = -8 ∧ ∀ y, f y ≥ f x :=
sorry

end tangent_lines_max_value_min_value_l3474_347454


namespace correct_expression_l3474_347420

theorem correct_expression : 
  (-((-8 : ℝ) ^ (1/3 : ℝ)) = 2) ∧ 
  (Real.sqrt 9 ≠ 3 ∧ Real.sqrt 9 ≠ -3) ∧
  (-Real.sqrt 16 ≠ 4) ∧
  (Real.sqrt ((-2)^2) ≠ -2) := by
  sorry

end correct_expression_l3474_347420


namespace flower_count_l3474_347434

theorem flower_count (vase_capacity : ℝ) (carnation_count : ℝ) (vases_needed : ℝ) :
  vase_capacity = 6.0 →
  carnation_count = 7.0 →
  vases_needed = 6.666666667 →
  (vases_needed * vase_capacity + carnation_count : ℝ) = 47.0 := by
  sorry

end flower_count_l3474_347434


namespace line_slope_l3474_347436

/-- A line in the xy-plane with y-intercept 20 and passing through (150, 600) has slope 580/150 -/
theorem line_slope (line : Set (ℝ × ℝ)) : 
  (∀ (x y : ℝ), (x, y) ∈ line ↔ y = (580/150) * x + 20) →
  (0, 20) ∈ line →
  (150, 600) ∈ line →
  ∃ (m : ℝ), ∀ (x y : ℝ), (x, y) ∈ line ↔ y = m * x + 20 :=
by sorry

end line_slope_l3474_347436


namespace johns_weekly_water_intake_l3474_347426

/-- Proves that John drinks 42 quarts of water in a week -/
theorem johns_weekly_water_intake :
  let daily_intake_gallons : ℚ := 3/2
  let gallons_to_quarts : ℚ := 4
  let days_in_week : ℕ := 7
  let weekly_intake_quarts : ℚ := daily_intake_gallons * gallons_to_quarts * days_in_week
  weekly_intake_quarts = 42 := by
  sorry

end johns_weekly_water_intake_l3474_347426


namespace f_difference_l3474_347405

/-- The function f(x) = 3x^2 + 5x + 4 -/
def f (x : ℝ) : ℝ := 3 * x^2 + 5 * x + 4

/-- Theorem stating that f(x+h) - f(x) = h(6x + 3h + 5) for all real x and h -/
theorem f_difference (x h : ℝ) : f (x + h) - f x = h * (6 * x + 3 * h + 5) := by
  sorry

end f_difference_l3474_347405


namespace twentieth_term_is_96_l3474_347472

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- The 20th term of the specific arithmetic sequence -/
theorem twentieth_term_is_96 :
  arithmeticSequenceTerm 1 5 20 = 96 := by
  sorry

end twentieth_term_is_96_l3474_347472


namespace disrespectful_quadratic_max_sum_at_one_l3474_347483

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  t : ℝ
  k : ℝ

/-- The value of the polynomial at x -/
def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  x^2 - p.t * x + p.k

/-- The composition of the polynomial with itself -/
def QuadraticPolynomial.compose (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.eval (p.eval x)

/-- A quadratic polynomial is disrespectful if p(p(x)) = 0 has exactly four real solutions -/
def QuadraticPolynomial.isDisrespectful (p : QuadraticPolynomial) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), (∀ x : ℝ, p.compose x = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)

/-- The sum of coefficients of a quadratic polynomial -/
def QuadraticPolynomial.sumCoefficients (p : QuadraticPolynomial) : ℝ :=
  1 - p.t + p.k

/-- The theorem to be proved -/
theorem disrespectful_quadratic_max_sum_at_one :
  ∃ (p : QuadraticPolynomial),
    p.isDisrespectful ∧
    (∀ q : QuadraticPolynomial, q.isDisrespectful → p.sumCoefficients ≥ q.sumCoefficients) ∧
    p.eval 1 = 3 := by
  sorry

end disrespectful_quadratic_max_sum_at_one_l3474_347483


namespace inverse_proportion_k_value_l3474_347427

/-- Given an inverse proportion function y = k/x passing through (-2, 3), prove k = -6 -/
theorem inverse_proportion_k_value : ∀ k : ℝ, 
  (∃ f : ℝ → ℝ, (∀ x : ℝ, x ≠ 0 → f x = k / x) ∧ f (-2) = 3) → 
  k = -6 := by
  sorry

end inverse_proportion_k_value_l3474_347427


namespace polynomial_value_relation_l3474_347422

theorem polynomial_value_relation (x y : ℝ) (h : 2 * x^2 + 3 * y + 3 = 8) :
  6 * x^2 + 9 * y + 8 = 23 := by
  sorry

end polynomial_value_relation_l3474_347422


namespace sqrt_two_irrationality_proof_assumption_l3474_347413

-- Define what it means for a real number to be rational
def IsRational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

-- Define what it means for a real number to be irrational
def IsIrrational (x : ℝ) : Prop := ¬(IsRational x)

-- State the theorem
theorem sqrt_two_irrationality_proof_assumption :
  (IsIrrational (Real.sqrt 2)) ↔ 
  (¬IsRational (Real.sqrt 2)) :=
by sorry

end sqrt_two_irrationality_proof_assumption_l3474_347413


namespace diophantine_equation_prime_divisor_l3474_347484

theorem diophantine_equation_prime_divisor (x y n : ℕ) 
  (h1 : x ≥ 3) (h2 : n ≥ 2) (h3 : x^2 + 5 = y^n) :
  ∀ p : ℕ, Nat.Prime p → p ∣ n → p ≡ 1 [MOD 4] := by
  sorry

end diophantine_equation_prime_divisor_l3474_347484


namespace thirteen_times_fifty_in_tens_l3474_347414

theorem thirteen_times_fifty_in_tens : 13 * 50 = 65 * 10 := by
  sorry

end thirteen_times_fifty_in_tens_l3474_347414


namespace negative_y_positive_l3474_347443

theorem negative_y_positive (y : ℝ) (h : y < 0) : -y > 0 := by
  sorry

end negative_y_positive_l3474_347443


namespace opposite_of_2023_l3474_347474

-- Define the concept of opposite for real numbers
def opposite (x : ℝ) : ℝ := -x

-- State the theorem
theorem opposite_of_2023 : opposite 2023 = -2023 := by
  -- The proof would go here, but we're skipping it as requested
  sorry

end opposite_of_2023_l3474_347474


namespace incorrect_statement_C_l3474_347417

theorem incorrect_statement_C :
  (∀ a b c : ℚ, a / 4 = c / 5 → (a - 4) / 4 = (c - 5) / 5) ∧
  (∀ a b : ℚ, (a - b) / b = 1 / 7 → a / b = 8 / 7) ∧
  (∃ a b : ℚ, a / b = 2 / 5 ∧ (a ≠ 2 ∨ b ≠ 5)) ∧
  (∀ a b c d : ℚ, a / b = c / d ∧ a / b = 2 / 3 ∧ b - d ≠ 0 → (a - c) / (b - d) = 2 / 3) :=
by sorry


end incorrect_statement_C_l3474_347417


namespace union_of_A_and_B_l3474_347444

-- Define set A
def A : Set ℝ := {x | x^2 + x - 2 = 0}

-- Define set B
def B : Set ℝ := {x | x ≥ 0 ∧ x < 1}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {x | x = -2 ∨ (0 ≤ x ∧ x < 1)} := by sorry

end union_of_A_and_B_l3474_347444


namespace subtract_three_five_l3474_347442

theorem subtract_three_five : 3 - 5 = -2 := by sorry

end subtract_three_five_l3474_347442


namespace real_roots_quadratic_complex_coeff_l3474_347408

theorem real_roots_quadratic_complex_coeff (i : ℂ) (m : ℝ) :
  (∃ x : ℝ, x^2 - (2*i - 1)*x + 3*m - i = 0) → m = 1/12 := by
  sorry

end real_roots_quadratic_complex_coeff_l3474_347408


namespace min_guests_banquet_l3474_347447

def total_food : ℝ := 337
def max_food_per_guest : ℝ := 2

theorem min_guests_banquet :
  ∃ (min_guests : ℕ), 
    (min_guests : ℝ) * max_food_per_guest ≥ total_food ∧
    ∀ (n : ℕ), (n : ℝ) * max_food_per_guest ≥ total_food → n ≥ min_guests ∧
    min_guests = 169 :=
by sorry

end min_guests_banquet_l3474_347447
