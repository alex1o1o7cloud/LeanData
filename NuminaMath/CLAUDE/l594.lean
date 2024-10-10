import Mathlib

namespace arithmetic_sequence_second_term_l594_59485

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_second_term 
  (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_sum : a 1 + a 3 = 2) : 
  a 2 = 1 := by
sorry

end arithmetic_sequence_second_term_l594_59485


namespace inequality_equivalence_l594_59400

theorem inequality_equivalence (x : ℝ) : 
  (x - 3) / 3 < (2 * x + 1) / 2 - 1 ↔ 2 * (x - 3) < 3 * (2 * x + 1) - 6 :=
by sorry

end inequality_equivalence_l594_59400


namespace kylie_piggy_bank_coins_kylie_piggy_bank_coins_value_l594_59486

/-- The number of coins Kylie got from her piggy bank -/
def coins_from_piggy_bank : ℕ := sorry

/-- The number of coins Kylie got from her brother -/
def coins_from_brother : ℕ := 13

/-- The number of coins Kylie got from her father -/
def coins_from_father : ℕ := 8

/-- The number of coins Kylie gave to Laura -/
def coins_given_to_laura : ℕ := 21

/-- The number of coins Kylie had left -/
def coins_left : ℕ := 15

theorem kylie_piggy_bank_coins :
  coins_from_piggy_bank + coins_from_brother + coins_from_father - coins_given_to_laura = coins_left :=
by sorry

theorem kylie_piggy_bank_coins_value : coins_from_piggy_bank = 15 :=
by sorry

end kylie_piggy_bank_coins_kylie_piggy_bank_coins_value_l594_59486


namespace parallelepiped_coverage_l594_59455

/-- A parallelepiped with dimensions a, b, and c can have three faces sharing a common vertex
    covered by five-cell strips without overlaps or gaps if and only if at least two of a, b,
    and c are divisible by 5. -/
theorem parallelepiped_coverage (a b c : ℕ) :
  (∃ (faces : Fin 3 → ℕ × ℕ), 
    (faces 0 = (a, b) ∨ faces 0 = (b, c) ∨ faces 0 = (c, a)) ∧
    (faces 1 = (a, b) ∨ faces 1 = (b, c) ∨ faces 1 = (c, a)) ∧
    (faces 2 = (a, b) ∨ faces 2 = (b, c) ∨ faces 2 = (c, a)) ∧
    faces 0 ≠ faces 1 ∧ faces 1 ≠ faces 2 ∧ faces 0 ≠ faces 2 ∧
    ∀ i : Fin 3, ∃ k : ℕ, (faces i).1 * (faces i).2 = 5 * k) ↔
  (a % 5 = 0 ∧ b % 5 = 0) ∨ (b % 5 = 0 ∧ c % 5 = 0) ∨ (c % 5 = 0 ∧ a % 5 = 0) :=
by sorry

end parallelepiped_coverage_l594_59455


namespace eight_digit_increasing_count_l594_59489

theorem eight_digit_increasing_count : ∃ M : ℕ, 
  (M = Nat.choose 7 5) ∧ 
  (M % 1000 = 21) := by sorry

end eight_digit_increasing_count_l594_59489


namespace cherry_popsicles_count_l594_59433

theorem cherry_popsicles_count (total : ℕ) (grape : ℕ) (banana : ℕ) (cherry : ℕ) :
  total = 17 → grape = 2 → banana = 2 → cherry = total - (grape + banana) → cherry = 13 := by
  sorry

end cherry_popsicles_count_l594_59433


namespace train_passing_jogger_time_l594_59408

/-- The time it takes for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time (jogger_speed train_speed : ℝ) 
  (initial_distance train_length : ℝ) (h1 : jogger_speed > 0) 
  (h2 : train_speed > jogger_speed) (h3 : initial_distance > 0) 
  (h4 : train_length > 0) : 
  (initial_distance + train_length) / (train_speed - jogger_speed) = 40 := by
  sorry

#check train_passing_jogger_time

end train_passing_jogger_time_l594_59408


namespace tree_age_conversion_l594_59475

/-- Converts a base 7 number to base 10 -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The given number in base 7 -/
def treeAgeBase7 : List Nat := [7, 4, 5, 2]

theorem tree_age_conversion :
  base7ToBase10 treeAgeBase7 = 966 := by
  sorry

end tree_age_conversion_l594_59475


namespace lower_price_proof_l594_59416

/-- Given a book with cost C and two selling prices P and H, where H yields 5% more gain than P, 
    this function calculates the lower selling price P. -/
def calculate_lower_price (C H : ℚ) : ℚ :=
  H / (1 + 0.05)

theorem lower_price_proof (C H : ℚ) (hC : C = 200) (hH : H = 350) :
  let P := calculate_lower_price C H
  ∃ ε > 0, |P - 368.42| < ε := by sorry

end lower_price_proof_l594_59416


namespace good_numbers_up_to_17_and_18_not_good_l594_59482

/-- The number of positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- A number m is "good" if there exists a positive integer n such that m = n / d(n) -/
def is_good (m : ℕ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ m = n / num_divisors n

theorem good_numbers_up_to_17_and_18_not_good :
  (∀ m : ℕ, m > 0 ∧ m ≤ 17 → is_good m) ∧ ¬ is_good 18 := by
  sorry

end good_numbers_up_to_17_and_18_not_good_l594_59482


namespace exercise_book_distribution_l594_59476

theorem exercise_book_distribution (m n : ℕ) : 
  (3 * n + 8 = m) →  -- If each student receives 3 books, there will be 8 books left over
  (0 < m - 5 * (n - 1)) →  -- The last student receives some books
  (m - 5 * (n - 1) < 5) →  -- The last student receives less than 5 books
  (n = 5 ∨ n = 6) := by
sorry

end exercise_book_distribution_l594_59476


namespace extremum_values_l594_59443

theorem extremum_values (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → Real.sqrt x + Real.sqrt y ≤ Real.sqrt 2) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 1/(2*y + 1) ≥ (3 + 2*Real.sqrt 2)/3) := by
  sorry

end extremum_values_l594_59443


namespace solution_set_equivalence_l594_59419

-- Define the solution set type
def SolutionSet := Set ℝ

-- Define the inequalities
def Inequality1 (k a b c : ℝ) (x : ℝ) : Prop :=
  k / (x + a) + (x + b) / (x + c) < 0

def Inequality2 (k a b c : ℝ) (x : ℝ) : Prop :=
  (k * x) / (a * x + 1) + (b * x + 1) / (c * x + 1) < 0

-- State the theorem
theorem solution_set_equivalence 
  (k a b c : ℝ) 
  (S1 : SolutionSet) 
  (h1 : S1 = {x | x ∈ (Set.Ioo (-3) (-1)) ∪ (Set.Ioo 1 2) ∧ Inequality1 k a b c x}) :
  {x | Inequality2 k a b c x} = 
    {x | x ∈ (Set.Ioo (-1) (-1/3)) ∪ (Set.Ioo (1/2) 1)} :=
by sorry

end solution_set_equivalence_l594_59419


namespace circle_rolling_inside_square_l594_59417

/-- The distance traveled by the center of a circle rolling inside a square -/
theorem circle_rolling_inside_square
  (circle_radius : ℝ)
  (square_side : ℝ)
  (h1 : circle_radius = 1)
  (h2 : square_side = 5) :
  (square_side - 2 * circle_radius) * 4 = 12 :=
by sorry

end circle_rolling_inside_square_l594_59417


namespace longest_common_length_l594_59451

theorem longest_common_length (wood_lengths : List Nat) : 
  wood_lengths = [90, 72, 120, 150, 108] → 
  Nat.gcd 90 (Nat.gcd 72 (Nat.gcd 120 (Nat.gcd 150 108))) = 6 := by
  sorry

end longest_common_length_l594_59451


namespace sqrt_equation_solution_l594_59492

theorem sqrt_equation_solution :
  ∃! x : ℝ, Real.sqrt (x^2 + 4*x) = 9 ∧ x^2 + 4*x ≥ 0 :=
by
  -- The unique solution is x = -2 + √85
  use -2 + Real.sqrt 85
  sorry

end sqrt_equation_solution_l594_59492


namespace base7_perfect_square_last_digit_l594_59409

def is_base7_perfect_square (x y z : ℕ) : Prop :=
  x ≠ 0 ∧ z < 7 ∧ ∃ k : ℕ, k^2 = x * 7^3 + y * 7^2 + 5 * 7 + z

theorem base7_perfect_square_last_digit 
  (x y z : ℕ) (h : is_base7_perfect_square x y z) : z = 1 ∨ z = 6 := by
  sorry

end base7_perfect_square_last_digit_l594_59409


namespace tori_height_l594_59483

/-- Tori's initial height in feet -/
def initial_height : ℝ := 4.4

/-- The amount Tori grew in feet -/
def growth : ℝ := 2.86

/-- Tori's current height in feet -/
def current_height : ℝ := initial_height + growth

theorem tori_height : current_height = 7.26 := by
  sorry

end tori_height_l594_59483


namespace sum_of_E_3_and_4_l594_59465

/-- Given a function E: ℝ → ℝ where E(3) = 5 and E(4) = 5, prove that E(3) + E(4) = 10 -/
theorem sum_of_E_3_and_4 (E : ℝ → ℝ) (h1 : E 3 = 5) (h2 : E 4 = 5) : E 3 + E 4 = 10 := by
  sorry

end sum_of_E_3_and_4_l594_59465


namespace function_composition_l594_59430

/-- Given a function f(x) = (x(x-2))/2, prove that f(x+2) = ((x+2)x)/2 -/
theorem function_composition (x : ℝ) : 
  let f : ℝ → ℝ := λ x => (x * (x - 2)) / 2
  f (x + 2) = ((x + 2) * x) / 2 := by
sorry

end function_composition_l594_59430


namespace rhombus_perimeter_l594_59459

/-- The perimeter of a rhombus with given diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 16) (h2 : d2 = 30) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 68 := by
  sorry

end rhombus_perimeter_l594_59459


namespace least_number_with_remainder_one_l594_59402

theorem least_number_with_remainder_one (n : ℕ) : 
  (∀ m : ℕ, m > 0 → m < 386 → (m % 35 ≠ 1 ∨ m % 11 ≠ 1)) ∧ 
  386 % 35 = 1 ∧ 
  386 % 11 = 1 := by
sorry

end least_number_with_remainder_one_l594_59402


namespace mistaken_calculation_correction_l594_59436

theorem mistaken_calculation_correction (x : ℝ) :
  5.46 - x = 3.97 → 5.46 + x = 6.95 := by
  sorry

end mistaken_calculation_correction_l594_59436


namespace hospital_bill_breakdown_l594_59470

theorem hospital_bill_breakdown (total_bill : ℝ) (medication_percentage : ℝ) 
  (overnight_percentage : ℝ) (food_cost : ℝ) (h1 : total_bill = 5000) 
  (h2 : medication_percentage = 0.5) (h3 : overnight_percentage = 0.25) 
  (h4 : food_cost = 175) : 
  total_bill * (1 - medication_percentage) * (1 - overnight_percentage) - food_cost = 1700 := by
  sorry

end hospital_bill_breakdown_l594_59470


namespace unique_solution_implies_m_half_l594_59463

/-- Given m > 0, if the equation m ln x - (1/2)x^2 + mx = 0 has a unique real solution, then m = 1/2 -/
theorem unique_solution_implies_m_half (m : ℝ) (hm : m > 0) :
  (∃! x : ℝ, m * Real.log x - (1/2) * x^2 + m * x = 0) → m = 1/2 := by
  sorry


end unique_solution_implies_m_half_l594_59463


namespace tangent_line_and_decreasing_condition_l594_59431

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 + (1-2*a)*x + a

-- State the theorem
theorem tangent_line_and_decreasing_condition (a : ℝ) :
  -- The tangent line at x = 1 has equation 2x + y - 2 = 0
  (∃ m b : ℝ, ∀ x y : ℝ, y = f a x → (x = 1 → y = m*x + b) ∧ m = -2 ∧ b = 2) ∧
  -- f is strictly decreasing on ℝ iff a ∈ (3-√6, 3+√6)
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ > f a x₂) ↔ (a > 3 - Real.sqrt 6 ∧ a < 3 + Real.sqrt 6) :=
sorry

end tangent_line_and_decreasing_condition_l594_59431


namespace cal_anthony_transaction_ratio_l594_59497

theorem cal_anthony_transaction_ratio :
  ∀ (mabel_transactions anthony_transactions cal_transactions jade_transactions : ℕ),
    mabel_transactions = 90 →
    anthony_transactions = mabel_transactions + mabel_transactions / 10 →
    jade_transactions = 85 →
    jade_transactions = cal_transactions + 19 →
    cal_transactions * 3 = anthony_transactions * 2 :=
by sorry

end cal_anthony_transaction_ratio_l594_59497


namespace janet_fertilizer_time_l594_59410

-- Define the constants from the problem
def gallons_per_horse_per_day : ℕ := 5
def number_of_horses : ℕ := 80
def total_acres : ℕ := 20
def gallons_per_acre : ℕ := 400
def acres_spread_per_day : ℕ := 4

-- Define the theorem
theorem janet_fertilizer_time : 
  (total_acres * gallons_per_acre) / (number_of_horses * gallons_per_horse_per_day) +
  total_acres / acres_spread_per_day = 25 := by
  sorry

end janet_fertilizer_time_l594_59410


namespace amy_baskets_l594_59481

/-- The number of baskets Amy will fill with candies -/
def num_baskets : ℕ :=
  let chocolate_bars := 5
  let mms := 7 * chocolate_bars
  let marshmallows := 6 * mms
  let total_candies := chocolate_bars + mms + marshmallows
  let candies_per_basket := 10
  total_candies / candies_per_basket

theorem amy_baskets : num_baskets = 25 := by
  sorry

end amy_baskets_l594_59481


namespace smallest_prime_digit_sum_20_l594_59477

def digit_sum (n : Nat) : Nat :=
  Nat.rec 0 (fun n sum => sum + n % 10) n

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

theorem smallest_prime_digit_sum_20 :
  ∀ n : Nat, n < 389 → ¬(is_prime n ∧ digit_sum n = 20) ∧
  is_prime 389 ∧ digit_sum 389 = 20 :=
sorry

end smallest_prime_digit_sum_20_l594_59477


namespace intersection_A_B_l594_59437

def A : Set ℕ := {0,1,2,3,4,5,6}

def B : Set ℕ := {x | ∃ n ∈ A, x = 2 * n}

theorem intersection_A_B : A ∩ B = {0,2,4,6} := by sorry

end intersection_A_B_l594_59437


namespace ali_seashells_left_l594_59461

/-- The number of seashells Ali has left after all transactions -/
def seashells_left (initial : ℝ) (given_friends : ℝ) (given_brothers : ℝ) (sold_fraction : ℝ) (traded_fraction : ℝ) : ℝ :=
  let remaining_after_giving := initial - (given_friends + given_brothers)
  let remaining_after_selling := remaining_after_giving * (1 - sold_fraction)
  remaining_after_selling * (1 - traded_fraction)

/-- Theorem stating that Ali has 76.375 seashells left after all transactions -/
theorem ali_seashells_left : 
  seashells_left 385.5 45.75 34.25 (2/3) (1/4) = 76.375 := by sorry

end ali_seashells_left_l594_59461


namespace tim_sugar_cookies_l594_59414

/-- Represents the number of sugar cookies Tim baked given the total number of cookies and the ratio of cookie types. -/
def sugar_cookies (total : ℕ) (choc_ratio sugar_ratio pb_ratio : ℕ) : ℕ :=
  (sugar_ratio * total) / (choc_ratio + sugar_ratio + pb_ratio)

/-- Theorem stating that Tim baked 15 sugar cookies given the problem conditions. -/
theorem tim_sugar_cookies :
  sugar_cookies 30 2 5 3 = 15 := by
sorry

end tim_sugar_cookies_l594_59414


namespace degree_of_sum_polynomials_l594_59428

-- Define the polynomials f and g
def f (z : ℂ) (c₃ c₂ c₁ c₀ : ℂ) : ℂ := c₃ * z^3 + c₂ * z^2 + c₁ * z + c₀
def g (z : ℂ) (d₂ d₁ d₀ : ℂ) : ℂ := d₂ * z^2 + d₁ * z + d₀

-- Define the degree of a polynomial
def degree (p : ℂ → ℂ) : ℕ := sorry

-- Theorem statement
theorem degree_of_sum_polynomials 
  (c₃ c₂ c₁ c₀ d₂ d₁ d₀ : ℂ) 
  (h₁ : c₃ ≠ 0) 
  (h₂ : d₂ ≠ 0) 
  (h₃ : c₃ + d₂ ≠ 0) : 
  degree (fun z ↦ f z c₃ c₂ c₁ c₀ + g z d₂ d₁ d₀) = 3 := by
  sorry

end degree_of_sum_polynomials_l594_59428


namespace second_division_count_correct_l594_59442

/-- Represents the number of people in the second division of money -/
def second_division_count : ℕ → Prop := λ x =>
  x > 6 ∧ (90 : ℚ) / (x - 6 : ℚ) = 120 / x

/-- The theorem stating the condition for the correct number of people in the second division -/
theorem second_division_count_correct (x : ℕ) : 
  second_division_count x ↔ 
    (∃ (y : ℕ), y > 0 ∧ 
      (90 : ℚ) / y = (120 : ℚ) / (y + 6) ∧
      x = y + 6) :=
sorry

end second_division_count_correct_l594_59442


namespace smallest_equal_packages_l594_59434

theorem smallest_equal_packages (n m : ℕ) : 
  (∀ k l : ℕ, k > 0 ∧ l > 0 ∧ 9 * k = 12 * l → n ≤ k) ∧ 
  (∃ m : ℕ, m > 0 ∧ 9 * n = 12 * m) → 
  n = 4 := by
sorry

end smallest_equal_packages_l594_59434


namespace inequality_solutions_count_l594_59458

theorem inequality_solutions_count : 
  (Finset.filter (fun p : ℕ × ℕ => 
    (p.1 : ℚ) / 76 + (p.2 : ℚ) / 71 < 1) 
    (Finset.product (Finset.range 76) (Finset.range 71))).card = 2625 :=
by sorry

end inequality_solutions_count_l594_59458


namespace chucks_team_score_final_score_proof_l594_59471

theorem chucks_team_score (red_team_score : ℕ) (score_difference : ℕ) : ℕ :=
  red_team_score + score_difference

theorem final_score_proof (red_team_score : ℕ) (score_difference : ℕ) 
  (h1 : red_team_score = 76)
  (h2 : score_difference = 19) :
  chucks_team_score red_team_score score_difference = 95 := by
  sorry

end chucks_team_score_final_score_proof_l594_59471


namespace min_value_reciprocal_sum_l594_59401

theorem min_value_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → a * 1 - b * (-1) = 1 → (1 / a + 1 / b ≥ 4 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x * 1 - y * (-1) = 1 ∧ 1 / x + 1 / y = 4) :=
by sorry

end min_value_reciprocal_sum_l594_59401


namespace greatest_BAABC_div_11_l594_59439

def is_valid_BAABC (n : ℕ) : Prop :=
  ∃ (A B C : ℕ),
    A < 10 ∧ B < 10 ∧ C < 10 ∧
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
    n = 10000 * B + 1000 * A + 100 * A + 10 * B + C

theorem greatest_BAABC_div_11 :
  ∀ n : ℕ,
    is_valid_BAABC n →
    n ≤ 96619 ∧
    is_valid_BAABC 96619 ∧
    96619 % 11 = 0 ∧
    (n % 11 = 0 → n ≤ 96619) :=
by sorry

end greatest_BAABC_div_11_l594_59439


namespace geometric_sequence_sum_l594_59457

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 4 + a 7 = 2) →
  (a 5 * a 8 = -8) →
  (a 1 + a 10 = -7) :=
by
  sorry

end geometric_sequence_sum_l594_59457


namespace extreme_perimeter_rectangles_l594_59498

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- Represents a rectangle with width w and height h -/
structure Rectangle where
  w : ℝ
  h : ℝ
  h_pos_w : 0 < w
  h_pos_h : 0 < h

/-- Predicate to check if a rectangle touches the given ellipse -/
def touches (e : Ellipse) (r : Rectangle) : Prop :=
  ∃ (x y : ℝ), (x^2 / e.a^2) + (y^2 / e.b^2) = 1 ∧
    (x = r.w / 2 ∨ x = -r.w / 2 ∨ y = r.h / 2 ∨ y = -r.h / 2)

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.w + r.h)

/-- Theorem stating the properties of rectangles with extreme perimeters touching an ellipse -/
theorem extreme_perimeter_rectangles (e : Ellipse) :
  ∃ (r_min r_max : Rectangle),
    touches e r_min ∧ touches e r_max ∧
    (∀ r : Rectangle, touches e r → perimeter r_min ≤ perimeter r) ∧
    (∀ r : Rectangle, touches e r → perimeter r ≤ perimeter r_max) ∧
    r_min.w = 2 * e.b ∧ r_min.h = 2 * Real.sqrt (e.a^2 - e.b^2) ∧
    r_max.w = r_max.h ∧ r_max.w = 2 * Real.sqrt ((e.a^2 + e.b^2) / 2) := by
  sorry

end extreme_perimeter_rectangles_l594_59498


namespace quadratic_two_distinct_roots_l594_59469

/-- A quadratic equation x^2 + mx + 9 has two distinct real roots if and only if m < -6 or m > 6 -/
theorem quadratic_two_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ m < -6 ∨ m > 6 :=
sorry

end quadratic_two_distinct_roots_l594_59469


namespace max_distance_on_circle_l594_59422

open Complex

theorem max_distance_on_circle (z : ℂ) :
  Complex.abs (z - I) = 1 →
  (∀ w : ℂ, Complex.abs (w - I) = 1 → Complex.abs (z + 2 + I) ≥ Complex.abs (w + 2 + I)) →
  Complex.abs (z + 2 + I) = Real.sqrt 2 * 2 + 1 := by
  sorry

end max_distance_on_circle_l594_59422


namespace tree_planting_variance_l594_59462

def tree_planting_data : List (Nat × Nat) := [(5, 3), (6, 4), (7, 3)]

def total_groups : Nat := tree_planting_data.map (·.2) |>.sum

theorem tree_planting_variance (h : total_groups = 10) :
  let mean := (tree_planting_data.map (fun (x, y) => x * y) |>.sum) / total_groups
  let variance := (1 : ℝ) / total_groups *
    (tree_planting_data.map (fun (x, y) => y * ((x : ℝ) - mean)^2) |>.sum)
  variance = 0.6 := by
  sorry

end tree_planting_variance_l594_59462


namespace f_2019_l594_59454

/-- The function f(n) represents the original number of the last person to leave the line. -/
def f (n : ℕ) : ℕ :=
  let m := Nat.sqrt n
  if n ≤ m * m + m then m * m + 1
  else m * m + m + 1

/-- Theorem stating that f(2019) = 1981 -/
theorem f_2019 : f 2019 = 1981 := by
  sorry

end f_2019_l594_59454


namespace isosceles_triangle_parallel_cut_l594_59453

/-- An isosceles triangle with given area and altitude --/
structure IsoscelesTriangle :=
  (area : ℝ)
  (altitude : ℝ)

/-- A line segment parallel to the base of the triangle --/
structure ParallelLine :=
  (length : ℝ)
  (trapezoidArea : ℝ)

/-- The theorem statement --/
theorem isosceles_triangle_parallel_cut (t : IsoscelesTriangle) (l : ParallelLine) :
  t.area = 150 ∧ t.altitude = 30 ∧ l.trapezoidArea = 100 →
  l.length = 10 * Real.sqrt 3 / 3 :=
sorry

end isosceles_triangle_parallel_cut_l594_59453


namespace number_multiplied_by_four_twice_l594_59479

theorem number_multiplied_by_four_twice : ∃ x : ℝ, (4 * (4 * x) = 32) ∧ (x = 2) := by
  sorry

end number_multiplied_by_four_twice_l594_59479


namespace school_students_problem_l594_59418

theorem school_students_problem (total : ℕ) (boys : ℕ) (girls : ℕ) :
  total = 150 →
  total = boys + girls →
  girls = (boys : ℚ) / 100 * total →
  boys = 60 :=
by
  sorry

end school_students_problem_l594_59418


namespace sqrt_sum_power_equality_l594_59491

theorem sqrt_sum_power_equality (m n : ℕ) : 
  ∃ k : ℕ, (Real.sqrt m + Real.sqrt (m - 1)) ^ n = Real.sqrt k + Real.sqrt (k - 1) := by
  sorry

end sqrt_sum_power_equality_l594_59491


namespace remainder_theorem_l594_59449

-- Define the polynomial P
variable (P : ℝ → ℝ)

-- Define the conditions
axiom P_19 : P 19 = 99
axiom P_99 : P 99 = 19

-- Define the remainder function
def remainder (P : ℝ → ℝ) : ℝ → ℝ :=
  λ x => -x + 118

-- Theorem statement
theorem remainder_theorem :
  ∃ Q : ℝ → ℝ, ∀ x : ℝ, P x = (x - 19) * (x - 99) * Q x + remainder P x :=
sorry

end remainder_theorem_l594_59449


namespace broken_line_path_length_l594_59466

/-- Given a circle with diameter 12 units and points C, D each 3 units from the endpoints of the diameter,
    the length of the path CPD is 6√5 units for any point P on the circle forming a right angle CPD. -/
theorem broken_line_path_length (O : ℝ × ℝ) (A B C D P : ℝ × ℝ) : 
  let r : ℝ := 6 -- radius of the circle
  dist A B = 12 ∧ -- diameter is 12 units
  dist A C = 3 ∧ -- C is 3 units from A
  dist B D = 3 ∧ -- D is 3 units from B
  dist O P = r ∧ -- P is on the circle
  (C.1 - P.1) * (D.1 - P.1) + (C.2 - P.2) * (D.2 - P.2) = 0 -- angle CPD is right angle
  →
  dist C P + dist P D = 6 * Real.sqrt 5 := by
  sorry

where
  dist : ℝ × ℝ → ℝ × ℝ → ℝ := λ (x, y) (a, b) ↦ Real.sqrt ((x - a)^2 + (y - b)^2)

end broken_line_path_length_l594_59466


namespace charles_earnings_proof_l594_59473

/-- Calculates Charles's earnings after tax from pet care activities -/
def charles_earnings (housesitting_rate : ℝ) (lab_walk_rate : ℝ) (gr_walk_rate : ℝ) (gs_walk_rate : ℝ)
                     (lab_groom_rate : ℝ) (gr_groom_rate : ℝ) (gs_groom_rate : ℝ)
                     (housesitting_time : ℝ) (lab_walk_time : ℝ) (gr_walk_time : ℝ) (gs_walk_time : ℝ)
                     (tax_rate : ℝ) : ℝ :=
  let total_before_tax := housesitting_rate * housesitting_time +
                          lab_walk_rate * lab_walk_time +
                          gr_walk_rate * gr_walk_time +
                          gs_walk_rate * gs_walk_time +
                          lab_groom_rate + gr_groom_rate + gs_groom_rate
  let tax_deduction := tax_rate * total_before_tax
  total_before_tax - tax_deduction

/-- Theorem stating Charles's earnings after tax -/
theorem charles_earnings_proof :
  charles_earnings 15 22 25 30 10 15 20 10 3 2 1.5 0.12 = 313.28 := by
  sorry

end charles_earnings_proof_l594_59473


namespace original_price_calculation_l594_59438

theorem original_price_calculation (current_price : ℝ) (reduction_percentage : ℝ) 
  (h1 : current_price = 56.10)
  (h2 : reduction_percentage = 0.15)
  : (current_price / (1 - reduction_percentage)) = 66 := by
  sorry

end original_price_calculation_l594_59438


namespace perimeter_after_adding_tiles_l594_59406

/-- Represents a rectangular figure composed of square tiles -/
structure TiledRectangle where
  length : ℕ
  width : ℕ
  extra_tiles : ℕ

/-- Calculates the perimeter of a TiledRectangle -/
def perimeter (rect : TiledRectangle) : ℕ :=
  2 * (rect.length + rect.width)

/-- The initial rectangular figure -/
def initial_rectangle : TiledRectangle :=
  { length := 5, width := 2, extra_tiles := 1 }

theorem perimeter_after_adding_tiles :
  ∃ (final_rect : TiledRectangle),
    perimeter initial_rectangle = 16 ∧
    final_rect.length + final_rect.width = initial_rectangle.length + initial_rectangle.width + 2 ∧
    final_rect.extra_tiles = initial_rectangle.extra_tiles + 2 ∧
    perimeter final_rect = 18 := by
  sorry

end perimeter_after_adding_tiles_l594_59406


namespace sum_in_base6_l594_59413

/-- Converts a number from base 6 to base 10 -/
def toBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a number from base 10 to base 6 -/
def toBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec go (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else go (m / 6) ((m % 6) :: acc)
  go n []

/-- The main theorem to prove -/
theorem sum_in_base6 :
  let a := toBase10 [4, 4, 4]
  let b := toBase10 [6, 6]
  let c := toBase10 [4]
  toBase6 (a + b + c) = [6, 0, 2] := by sorry

end sum_in_base6_l594_59413


namespace bruce_grape_purchase_l594_59403

/-- The price of grapes per kg -/
def grape_price : ℕ := 70

/-- The quantity of mangoes purchased in kg -/
def mango_quantity : ℕ := 9

/-- The price of mangoes per kg -/
def mango_price : ℕ := 55

/-- The total amount paid -/
def total_paid : ℕ := 985

/-- The quantity of grapes purchased in kg -/
def grape_quantity : ℕ := (total_paid - mango_quantity * mango_price) / grape_price

theorem bruce_grape_purchase :
  grape_quantity * grape_price + mango_quantity * mango_price = total_paid ∧ grape_quantity = 7 := by
  sorry

end bruce_grape_purchase_l594_59403


namespace quadratic_polynomial_conditions_l594_59468

-- Define the quadratic polynomial
def q (x : ℚ) : ℚ := (6/5) * x^2 - (18/5) * x - (108/5)

-- Theorem stating the conditions
theorem quadratic_polynomial_conditions :
  q (-3) = 0 ∧ q 6 = 0 ∧ q 2 = -24 := by sorry

end quadratic_polynomial_conditions_l594_59468


namespace simple_interest_example_l594_59460

/-- Calculate simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Proof that the simple interest on $10000 at 8% per annum for 12 months is $800 -/
theorem simple_interest_example : 
  simple_interest 10000 0.08 1 = 800 := by
  sorry

end simple_interest_example_l594_59460


namespace books_left_unpacked_l594_59447

theorem books_left_unpacked (initial_boxes : Nat) (books_per_initial_box : Nat) (books_per_new_box : Nat) :
  initial_boxes = 1485 →
  books_per_initial_box = 42 →
  books_per_new_box = 45 →
  (initial_boxes * books_per_initial_box) % books_per_new_box = 30 := by
  sorry

end books_left_unpacked_l594_59447


namespace new_lamp_taller_by_exact_amount_l594_59464

/-- The height difference between two lamps -/
def lamp_height_difference (old_height new_height : ℝ) : ℝ :=
  new_height - old_height

/-- Theorem stating the height difference between the new and old lamps -/
theorem new_lamp_taller_by_exact_amount : 
  lamp_height_difference 1 2.3333333333333335 = 1.3333333333333335 := by
  sorry

end new_lamp_taller_by_exact_amount_l594_59464


namespace common_roots_cubic_polynomials_l594_59499

/-- Given two cubic polynomials with two distinct common roots, prove that a = 7 and b = 8 -/
theorem common_roots_cubic_polynomials (a b : ℝ) :
  (∃ r s : ℝ, r ≠ s ∧
    (r^3 + a*r^2 + 13*r + 10 = 0) ∧
    (r^3 + b*r^2 + 16*r + 12 = 0) ∧
    (s^3 + a*s^2 + 13*s + 10 = 0) ∧
    (s^3 + b*s^2 + 16*s + 12 = 0)) →
  a = 7 ∧ b = 8 := by
sorry

end common_roots_cubic_polynomials_l594_59499


namespace smallest_n_value_l594_59427

def is_not_divisible_by_ten (m : ℕ) : Prop := ∀ k : ℕ, m ≠ 10 * k

theorem smallest_n_value (a b c m n : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  a ≥ b → b ≥ c →
  a + b + c = 2010 →
  a * b * c = m * (10 ^ n) →
  is_not_divisible_by_ten m →
  (∀ k : ℕ, k < n → ∃ (m' : ℕ), a * b * c = m' * (10 ^ k) → ¬(is_not_divisible_by_ten m')) →
  n = 500 := by
sorry

end smallest_n_value_l594_59427


namespace normal_price_after_discounts_l594_59488

theorem normal_price_after_discounts (final_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (normal_price : ℝ) : 
  final_price = 36 ∧ 
  discount1 = 0.1 ∧ 
  discount2 = 0.2 ∧ 
  final_price = normal_price * (1 - discount1) * (1 - discount2) →
  normal_price = 50 := by
sorry

end normal_price_after_discounts_l594_59488


namespace part1_part2_l594_59405

-- Define the points
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (-2, 3)
def C : ℝ × ℝ := (8, -5)

-- Define vectors
def OA : ℝ × ℝ := A
def OB : ℝ × ℝ := B
def OC : ℝ × ℝ := C
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Part 1
theorem part1 (x y : ℝ) : 
  OC = (x * OA.1 + y * OB.1, x * OA.2 + y * OB.2) → x = 2 ∧ y = -3 := by sorry

-- Part 2
theorem part2 (m : ℝ) :
  ∃ (k : ℝ), k ≠ 0 ∧ AB = (k * (m * OA.1 + OC.1), k * (m * OA.2 + OC.2)) → m = 1 := by sorry

end part1_part2_l594_59405


namespace inequality_solution_l594_59472

def satisfies_inequality (x : ℤ) : Prop :=
  8.58 * (Real.log x / Real.log 4) + Real.log (Real.sqrt x - 1) / Real.log 2 < 
  Real.log (Real.log 5 / Real.log (Real.sqrt 5)) / Real.log 2

theorem inequality_solution :
  ∀ x : ℤ, satisfies_inequality x ↔ (x = 2 ∨ x = 3) :=
sorry

end inequality_solution_l594_59472


namespace imaginary_part_of_complex_fraction_l594_59420

theorem imaginary_part_of_complex_fraction (i : ℂ) : 
  i^2 = -1 → Complex.im (i^5 / (1 - i)) = 1/2 := by sorry

end imaginary_part_of_complex_fraction_l594_59420


namespace playground_area_ratio_l594_59446

theorem playground_area_ratio (r : ℝ) (h : r > 0) :
  (π * r^2) / (π * (3*r)^2) = 1 / 9 := by
  sorry

end playground_area_ratio_l594_59446


namespace train_length_calculation_train_B_length_l594_59404

/-- Given two trains running in opposite directions, calculate the length of the second train. -/
theorem train_length_calculation (length_A : ℝ) (speed_A : ℝ) (speed_B : ℝ) (time : ℝ) : ℝ :=
  let relative_speed := (speed_A + speed_B) * 1000 / 3600
  let total_distance := relative_speed * time
  total_distance - length_A

/-- Prove that the length of Train B is approximately 219.95 meters. -/
theorem train_B_length :
  ∃ (length_B : ℝ), abs (length_B - train_length_calculation 280 120 80 9) < 0.01 := by
  sorry

end train_length_calculation_train_B_length_l594_59404


namespace math_books_in_same_box_probability_l594_59496

/-- Represents a box that can hold textbooks -/
structure Box where
  capacity : Nat

/-- Represents the collection of textbooks -/
structure Textbooks where
  total : Nat
  math : Nat

/-- Represents the problem setup -/
structure TextbookProblem where
  boxes : List Box
  books : Textbooks

/-- The probability of all math textbooks being in the same box -/
def mathBooksInSameBoxProbability (problem : TextbookProblem) : Rat :=
  18/1173

/-- The main theorem stating the probability of all math textbooks being in the same box -/
theorem math_books_in_same_box_probability 
  (problem : TextbookProblem)
  (h1 : problem.boxes.length = 3)
  (h2 : problem.books.total = 15)
  (h3 : problem.books.math = 4)
  (h4 : problem.boxes.map Box.capacity = [4, 5, 6]) :
  mathBooksInSameBoxProbability problem = 18/1173 := by
  sorry

end math_books_in_same_box_probability_l594_59496


namespace perfect_square_trinomial_l594_59429

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 2*(m-1)*x + 4 = (x + a)^2) → 
  (m = 3 ∨ m = -1) :=
by sorry

end perfect_square_trinomial_l594_59429


namespace energy_usage_is_219_l594_59426

/-- Calculates the total energy usage given the energy consumption and duration of each light. -/
def total_energy_usage (bedroom_watts_per_hour : ℝ) 
                       (bedroom_hours : ℝ)
                       (office_multiplier : ℝ)
                       (office_hours : ℝ)
                       (living_room_multiplier : ℝ)
                       (living_room_hours : ℝ)
                       (kitchen_multiplier : ℝ)
                       (kitchen_hours : ℝ)
                       (bathroom_multiplier : ℝ)
                       (bathroom_hours : ℝ) : ℝ :=
  bedroom_watts_per_hour * bedroom_hours +
  (office_multiplier * bedroom_watts_per_hour) * office_hours +
  (living_room_multiplier * bedroom_watts_per_hour) * living_room_hours +
  (kitchen_multiplier * bedroom_watts_per_hour) * kitchen_hours +
  (bathroom_multiplier * bedroom_watts_per_hour) * bathroom_hours

/-- Theorem stating that the total energy usage is 219 watts given the specified conditions. -/
theorem energy_usage_is_219 :
  total_energy_usage 6 2 3 3 4 4 2 1 5 1.5 = 219 := by
  sorry

end energy_usage_is_219_l594_59426


namespace jihoon_calculation_mistake_l594_59493

theorem jihoon_calculation_mistake (x : ℝ) : 
  x - 7 = 0.45 → x * 7 = 52.15 := by
sorry

end jihoon_calculation_mistake_l594_59493


namespace bus_journey_speed_l594_59474

/-- Calculates the average speed for the remaining distance of a bus journey -/
theorem bus_journey_speed 
  (total_distance : ℝ) 
  (partial_distance : ℝ) 
  (partial_speed : ℝ) 
  (total_time : ℝ) 
  (h1 : total_distance = 250)
  (h2 : partial_distance = 100)
  (h3 : partial_speed = 40)
  (h4 : total_time = 5)
  : (total_distance - partial_distance) / (total_time - partial_distance / partial_speed) = 60 := by
  sorry

end bus_journey_speed_l594_59474


namespace triangle_expression_range_l594_59423

theorem triangle_expression_range (A B C a b c : ℝ) : 
  0 < A → A < 3 * π / 4 →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  c * Real.sin A = a * Real.cos C →
  1 < Real.sqrt 3 * Real.sin A - Real.cos (B + π / 4) ∧ 
  Real.sqrt 3 * Real.sin A - Real.cos (B + π / 4) ≤ 2 := by
  sorry

end triangle_expression_range_l594_59423


namespace sum_of_cubes_l594_59494

theorem sum_of_cubes (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : (x + y)^2 = 2500) (h2 : x * y = 500) : x^3 + y^3 = 50000 := by
  sorry

end sum_of_cubes_l594_59494


namespace constant_k_equality_l594_59412

theorem constant_k_equality (k : ℝ) : 
  (∀ x : ℝ, -x^2 - (k + 9)*x - 8 = -(x - 2)*(x - 4)) → k = -15 := by
  sorry

end constant_k_equality_l594_59412


namespace parallel_transitive_parallel_common_not_parallel_to_common_not_parallel_no_common_l594_59478

-- Define a type for lines
def Line : Type := ℝ → ℝ → Prop

-- Define parallelism between two lines
def parallel (l1 l2 : Line) : Prop := sorry

theorem parallel_transitive (l1 l2 l3 : Line) :
  (parallel l1 l3 ∧ parallel l2 l3) → parallel l1 l2 :=
sorry

theorem parallel_common (l1 l2 : Line) :
  parallel l1 l2 → ∃ l3 : Line, parallel l1 l3 ∧ parallel l2 l3 :=
sorry

theorem not_parallel_to_common (l1 l2 l3 : Line) :
  (¬ parallel l1 l3 ∨ ¬ parallel l2 l3) → ¬ parallel l1 l2 :=
sorry

theorem not_parallel_no_common (l1 l2 : Line) :
  ¬ parallel l1 l2 → ¬ ∃ l3 : Line, parallel l1 l3 ∧ parallel l2 l3 :=
sorry

end parallel_transitive_parallel_common_not_parallel_to_common_not_parallel_no_common_l594_59478


namespace sin_20_cos_40_plus_cos_20_sin_40_l594_59441

theorem sin_20_cos_40_plus_cos_20_sin_40 : 
  Real.sin (20 * π / 180) * Real.cos (40 * π / 180) + 
  Real.cos (20 * π / 180) * Real.sin (40 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end sin_20_cos_40_plus_cos_20_sin_40_l594_59441


namespace race_course_length_correct_l594_59490

/-- The length of a race course where two runners finish at the same time -/
def race_course_length : ℝ :=
  let speed_ratio : ℝ := 7
  let head_start : ℝ := 120
  140

theorem race_course_length_correct :
  let speed_ratio : ℝ := 7  -- A is 7 times faster than B
  let head_start : ℝ := 120 -- B starts 120 meters ahead
  let course_length := race_course_length
  course_length / speed_ratio = (course_length - head_start) / 1 :=
by sorry

end race_course_length_correct_l594_59490


namespace least_subtraction_for_divisibility_l594_59425

theorem least_subtraction_for_divisibility : ∃ (n : ℕ), n = 15 ∧ 
  (∀ (m : ℕ), m < n → ¬(23 ∣ (78721 - m))) ∧ (23 ∣ (78721 - n)) := by
  sorry

end least_subtraction_for_divisibility_l594_59425


namespace red_permutations_l594_59435

theorem red_permutations : 
  let n : ℕ := 1
  let total_letters : ℕ := 3 * n
  let permutations : ℕ := Nat.factorial total_letters / (Nat.factorial n)^3
  permutations = 6 := by sorry

end red_permutations_l594_59435


namespace sum_M_l594_59480

def M : ℕ → ℕ
  | 0 => 0
  | 1 => 4^2 - 2^2
  | (n+2) => (2*n+5)^2 + (2*n+4)^2 - (2*n+3)^2 - (2*n+2)^2 + M n

theorem sum_M : M 50 = 5304 := by
  sorry

end sum_M_l594_59480


namespace rectangular_box_problem_l594_59484

theorem rectangular_box_problem :
  ∃! (a b c : ℕ+),
    (a ≤ b ∧ b ≤ c) ∧
    (a * b * c = 2 * (2 * (a * b + b * c + c * a))) ∧
    (4 * a = c) := by
  sorry

end rectangular_box_problem_l594_59484


namespace leftmost_digit_of_12_to_37_l594_59421

def log_2_lower : ℝ := 0.3010
def log_2_upper : ℝ := 0.3011
def log_3_lower : ℝ := 0.4771
def log_3_upper : ℝ := 0.4772

theorem leftmost_digit_of_12_to_37 
  (h1 : log_2_lower < Real.log 2)
  (h2 : Real.log 2 < log_2_upper)
  (h3 : log_3_lower < Real.log 3)
  (h4 : Real.log 3 < log_3_upper) :
  (12^37 : ℝ) ≥ 8 * 10^39 ∧ (12^37 : ℝ) < 9 * 10^39 :=
sorry

end leftmost_digit_of_12_to_37_l594_59421


namespace equation_solutions_l594_59456

theorem equation_solutions :
  (∃ x : ℚ, 5 * x - 2 * (x - 1) = 3 ∧ x = 1/3) ∧
  (∃ x : ℚ, (3 * x - 2) / 6 = 1 + (x - 1) / 3 ∧ x = 6) :=
by sorry

end equation_solutions_l594_59456


namespace min_throws_for_repeated_sum_l594_59415

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The number of dice thrown -/
def numDice : ℕ := 4

/-- The minimum possible sum when throwing the dice -/
def minSum : ℕ := numDice

/-- The maximum possible sum when throwing the dice -/
def maxSum : ℕ := numDice * numFaces

/-- The number of possible distinct sums -/
def numDistinctSums : ℕ := maxSum - minSum + 1

/-- The minimum number of throws required to guarantee a repeated sum -/
def minThrows : ℕ := numDistinctSums + 1

theorem min_throws_for_repeated_sum :
  minThrows = 22 := by sorry

end min_throws_for_repeated_sum_l594_59415


namespace star_emilio_sum_difference_l594_59487

def star_list : List Nat := List.range 40

def replace_digit (n : Nat) : Nat :=
  let s := toString n
  let replaced := s.map (fun c => if c == '3' then '2' else c)
  replaced.toNat!

def emilio_list : List Nat := star_list.map replace_digit

theorem star_emilio_sum_difference :
  star_list.sum - emilio_list.sum = 104 := by sorry

end star_emilio_sum_difference_l594_59487


namespace ellipse_intersection_product_range_l594_59450

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2/16 + y^2/12 = 1

-- Define point M
def M : ℝ × ℝ := (0, 2)

-- Define the dot product of two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1) + (v1.2 * v2.2)

-- Define the vector from origin to a point
def vector_from_origin (p : ℝ × ℝ) : ℝ × ℝ := p

-- Define the vector from M to a point
def vector_from_M (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - M.1, p.2 - M.2)

-- Statement of the theorem
theorem ellipse_intersection_product_range :
  ∀ P Q : ℝ × ℝ,
  C P.1 P.2 →
  C Q.1 Q.2 →
  (∃ k : ℝ, Q.2 - M.2 = k * (Q.1 - M.1) ∧ P.2 - M.2 = k * (P.1 - M.1)) →
  -20 ≤ (dot_product (vector_from_origin P) (vector_from_origin Q) +
         dot_product (vector_from_M P) (vector_from_M Q)) ∧
  (dot_product (vector_from_origin P) (vector_from_origin Q) +
   dot_product (vector_from_M P) (vector_from_M Q)) ≤ -52/3 :=
by sorry


end ellipse_intersection_product_range_l594_59450


namespace complex_magnitude_problem_l594_59440

theorem complex_magnitude_problem (z : ℂ) :
  Complex.abs z * (3 * z + 2 * Complex.I) = 2 * (Complex.I * z - 6) →
  Complex.abs z = 2 := by
sorry

end complex_magnitude_problem_l594_59440


namespace possible_integer_roots_l594_59424

def polynomial (x b₂ b₁ : ℤ) : ℤ := x^3 + b₂ * x^2 + b₁ * x - 30

def is_root (x b₂ b₁ : ℤ) : Prop := polynomial x b₂ b₁ = 0

def divisors_of_30 : Set ℤ := {-30, -15, -10, -6, -5, -3, -2, -1, 1, 2, 3, 5, 6, 10, 15, 30}

theorem possible_integer_roots (b₂ b₁ : ℤ) :
  {x : ℤ | ∃ (b₂ b₁ : ℤ), is_root x b₂ b₁} = divisors_of_30 := by sorry

end possible_integer_roots_l594_59424


namespace jeans_pricing_markup_l594_59448

theorem jeans_pricing_markup (manufacturing_cost : ℝ) (customer_price : ℝ) (retailer_price : ℝ)
  (h1 : customer_price = manufacturing_cost * 1.54)
  (h2 : customer_price = retailer_price * 1.1) :
  (retailer_price - manufacturing_cost) / manufacturing_cost * 100 = 40 := by
sorry

end jeans_pricing_markup_l594_59448


namespace seventh_term_is_13_4_l594_59411

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- The first term of the sequence
  a : ℝ
  -- The common difference of the sequence
  d : ℝ
  -- The sum of the first four terms is 14
  sum_first_four : a + (a + d) + (a + 2*d) + (a + 3*d) = 14
  -- The fifth term is 9
  fifth_term : a + 4*d = 9

/-- The seventh term of the arithmetic sequence is 13.4 -/
theorem seventh_term_is_13_4 (seq : ArithmeticSequence) :
  seq.a + 6*seq.d = 13.4 := by
  sorry


end seventh_term_is_13_4_l594_59411


namespace rachel_total_problems_l594_59452

/-- The number of math problems Rachel solved in total -/
def total_problems (problems_per_minute : ℕ) (minutes : ℕ) (problems_next_day : ℕ) : ℕ :=
  problems_per_minute * minutes + problems_next_day

/-- Theorem stating that Rachel solved 151 math problems in total -/
theorem rachel_total_problems :
  total_problems 7 18 25 = 151 := by
  sorry

end rachel_total_problems_l594_59452


namespace total_eyes_is_92_l594_59407

/-- Represents a monster family in the portrait --/
structure MonsterFamily where
  totalEyes : ℕ

/-- The main monster family --/
def mainFamily : MonsterFamily :=
  { totalEyes := 1 + 3 + 3 * 4 + 5 + 6 + 2 + 1 + 7 + 8 }

/-- The first neighboring monster family --/
def neighborFamily1 : MonsterFamily :=
  { totalEyes := 9 + 3 + 7 + 3 }

/-- The second neighboring monster family --/
def neighborFamily2 : MonsterFamily :=
  { totalEyes := 4 + 2 * 8 + 5 }

/-- The total number of eyes in the monster family portrait --/
def totalEyesInPortrait : ℕ :=
  mainFamily.totalEyes + neighborFamily1.totalEyes + neighborFamily2.totalEyes

/-- Theorem stating that the total number of eyes in the portrait is 92 --/
theorem total_eyes_is_92 : totalEyesInPortrait = 92 := by
  sorry

end total_eyes_is_92_l594_59407


namespace plot_length_is_80_l594_59467

/-- A rectangular plot with specific dimensions and fencing cost. -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  fencing_cost_per_meter : ℝ
  total_fencing_cost : ℝ
  length_breadth_difference : ℝ
  h_length : length = breadth + length_breadth_difference
  h_perimeter : 2 * (length + breadth) = total_fencing_cost / fencing_cost_per_meter

/-- The length of the rectangular plot is 80 meters given the specified conditions. -/
theorem plot_length_is_80 (plot : RectangularPlot)
  (h_length_diff : plot.length_breadth_difference = 60)
  (h_fencing_cost : plot.fencing_cost_per_meter = 26.5)
  (h_total_cost : plot.total_fencing_cost = 5300) :
  plot.length = 80 := by
  sorry

end plot_length_is_80_l594_59467


namespace minimize_expression_l594_59444

theorem minimize_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 2) (h3 : a + b = 3) :
  ∃ (min_a : ℝ), min_a = 2/3 ∧ 
    ∀ (x : ℝ), x > 0 → x + b = 3 → (4/x + 1/(b-2) ≥ 4/min_a + 1/(b-2)) :=
by sorry

end minimize_expression_l594_59444


namespace basketball_shot_minimum_l594_59495

theorem basketball_shot_minimum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a < 1) (hbb : b < 1) 
  (h_expected : 3 * a + 2 * b = 2) : 
  (2 / a + 1 / (3 * b)) ≥ 16 / 3 := by
sorry

end basketball_shot_minimum_l594_59495


namespace total_earnings_theorem_l594_59445

/-- Represents the earnings from the aqua park --/
def aqua_park_earnings 
  (admission_cost tour_cost meal_cost souvenir_cost : ℕ) 
  (group1_size group2_size group3_size : ℕ) : ℕ :=
  let group1_total := group1_size * (admission_cost + tour_cost + meal_cost + souvenir_cost)
  let group2_total := group2_size * (admission_cost + meal_cost)
  let group3_total := group3_size * (admission_cost + tour_cost + souvenir_cost)
  group1_total + group2_total + group3_total

/-- Theorem stating the total earnings from all groups --/
theorem total_earnings_theorem : 
  aqua_park_earnings 12 6 10 8 10 15 8 = 898 := by
  sorry

end total_earnings_theorem_l594_59445


namespace thirty_day_month_equal_sundays_tuesdays_l594_59432

/-- Represents the days of the week -/
inductive DayOfWeek
| sunday
| monday
| tuesday
| wednesday
| thursday
| friday
| saturday

/-- Counts the occurrences of a specific day in a 30-day month starting from a given day -/
def countDay (startDay : DayOfWeek) (targetDay : DayOfWeek) : Nat :=
  sorry

/-- Checks if Sundays and Tuesdays are equal in a 30-day month starting from a given day -/
def hasSameSundaysAndTuesdays (startDay : DayOfWeek) : Bool :=
  countDay startDay DayOfWeek.sunday = countDay startDay DayOfWeek.tuesday

/-- Counts the number of possible starting days for a 30-day month with equal Sundays and Tuesdays -/
def countValidStartDays : Nat :=
  sorry

theorem thirty_day_month_equal_sundays_tuesdays :
  countValidStartDays = 3 :=
sorry

end thirty_day_month_equal_sundays_tuesdays_l594_59432
