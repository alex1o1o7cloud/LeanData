import Mathlib

namespace square_formation_for_12_and_15_l2149_214919

/-- Given n sticks with lengths 1, 2, ..., n, determine if a square can be formed
    or the minimum number of sticks to be broken in half to form a square. -/
def minSticksToBreak (n : ℕ) : ℕ :=
  let totalLength := n * (n + 1) / 2
  if totalLength % 4 = 0 then 0
  else
    let targetLength := (totalLength / 4 + 1) * 4
    (targetLength - totalLength + 1) / 2

theorem square_formation_for_12_and_15 :
  minSticksToBreak 12 = 2 ∧ minSticksToBreak 15 = 0 := by
  sorry


end square_formation_for_12_and_15_l2149_214919


namespace complex_expression_equality_l2149_214948

theorem complex_expression_equality (x y : ℝ) (hx : x = 3) (hy : y = 2) :
  4 * (x^y * (7^y * 24^x)) / (x*y) + 5 * (x * (13^y * 15^x)) - 2 * (y * (6^x * 28^y)) + 7 * (x*y * (3^x * 19^y)) / (x+y) = 11948716.8 := by
  sorry

end complex_expression_equality_l2149_214948


namespace right_triangle_perfect_square_l2149_214928

theorem right_triangle_perfect_square (a b c : ℕ) : 
  Prime a →
  a^2 + b^2 = c^2 →
  ∃ (n : ℕ), 2 * (a + b + 1) = n^2 := by
sorry

end right_triangle_perfect_square_l2149_214928


namespace june_songs_total_l2149_214911

def songs_in_june (vivian_daily : ℕ) (clara_difference : ℕ) (total_days : ℕ) (weekend_days : ℕ) : ℕ :=
  let weekdays := total_days - weekend_days
  let vivian_total := vivian_daily * weekdays
  let clara_daily := vivian_daily - clara_difference
  let clara_total := clara_daily * weekdays
  vivian_total + clara_total

theorem june_songs_total :
  songs_in_june 10 2 30 8 = 396 := by
  sorry

end june_songs_total_l2149_214911


namespace no_integer_solution_l2149_214950

theorem no_integer_solution : ¬ ∃ (x : ℤ), x^2 * 7 = 2^14 := by
  sorry

end no_integer_solution_l2149_214950


namespace greatest_four_digit_divisible_by_3_and_6_l2149_214909

theorem greatest_four_digit_divisible_by_3_and_6 :
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 3 = 0 ∧ n % 6 = 0 → n ≤ 9996 := by
  sorry

end greatest_four_digit_divisible_by_3_and_6_l2149_214909


namespace parabola_equation_l2149_214933

/-- Definition of the hyperbola -/
def hyperbola (x y : ℝ) : Prop := 16 * x^2 - 9 * y^2 = 144

/-- Definition of the left vertex of the hyperbola -/
def left_vertex (x y : ℝ) : Prop := hyperbola x y ∧ x < 0 ∧ y = 0

/-- Definition of a parabola passing through a point -/
def parabola_through_point (eq : ℝ → ℝ → Prop) (px py : ℝ) : Prop :=
  eq px py

/-- Theorem stating the standard equation of the parabola -/
theorem parabola_equation (f : ℝ → ℝ → Prop) (fx fy : ℝ) :
  left_vertex fx fy →
  parabola_through_point f 2 (-4) →
  (∀ x y, f x y ↔ y^2 = 8*x) ∨ (∀ x y, f x y ↔ x^2 = -y) :=
sorry

end parabola_equation_l2149_214933


namespace binomial_10_2_l2149_214989

theorem binomial_10_2 : Nat.choose 10 2 = 45 := by
  sorry

end binomial_10_2_l2149_214989


namespace max_points_is_four_l2149_214931

/-- A configuration of points and associated real numbers satisfying the distance property -/
structure PointConfiguration where
  n : ℕ
  points : Fin n → ℝ × ℝ
  radii : Fin n → ℝ
  distance_property : ∀ (i j : Fin n), i ≠ j →
    Real.sqrt ((points i).1 - (points j).1)^2 + ((points i).2 - (points j).2)^2 = radii i + radii j

/-- The maximal number of points in a valid configuration is 4 -/
theorem max_points_is_four :
  (∃ (config : PointConfiguration), config.n = 4) ∧
  (∀ (config : PointConfiguration), config.n ≤ 4) :=
sorry

end max_points_is_four_l2149_214931


namespace chores_per_week_l2149_214901

theorem chores_per_week 
  (cookie_price : ℕ) 
  (cookies_per_pack : ℕ) 
  (budget : ℕ) 
  (cookies_per_chore : ℕ) 
  (weeks : ℕ) 
  (h1 : cookie_price = 3)
  (h2 : cookies_per_pack = 24)
  (h3 : budget = 15)
  (h4 : cookies_per_chore = 3)
  (h5 : weeks = 10)
  : (budget / cookie_price) * cookies_per_pack / weeks / cookies_per_chore = 4 := by
  sorry

end chores_per_week_l2149_214901


namespace time_period_is_three_years_l2149_214942

/-- Represents the simple interest calculation and conditions --/
def simple_interest_problem (t : ℝ) : Prop :=
  let initial_deposit : ℝ := 9000
  let final_amount : ℝ := 10200
  let higher_rate_amount : ℝ := 10740
  ∃ r : ℝ,
    -- Condition for the original interest rate
    initial_deposit * (1 + r * t / 100) = final_amount ∧
    -- Condition for the interest rate 2% higher
    initial_deposit * (1 + (r + 2) * t / 100) = higher_rate_amount

/-- The theorem stating that the time period is 3 years --/
theorem time_period_is_three_years :
  simple_interest_problem 3 := by sorry

end time_period_is_three_years_l2149_214942


namespace circle_intersection_theorem_l2149_214903

/-- Circle C₁ in the Cartesian plane -/
def C₁ (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*m*x - (4*m+6)*y - 4 = 0

/-- Circle C₂ in the Cartesian plane -/
def C₂ (x y : ℝ) : Prop :=
  (x + 2)^2 + (y - 3)^2 = (x + 2)^2 + (y - 3)^2

/-- The theorem stating the value of m for the given conditions -/
theorem circle_intersection_theorem (m : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  C₁ m x₁ y₁ ∧ C₁ m x₂ y₂ ∧ C₂ x₁ y₁ ∧ C₂ x₂ y₂ ∧ 
  x₁^2 - x₂^2 = y₂^2 - y₁^2 →
  m = -6 := by
  sorry

end circle_intersection_theorem_l2149_214903


namespace class_size_from_ratio_and_red_hair_count_l2149_214940

/-- Represents the number of children with each hair color in the ratio --/
structure HairColorRatio :=
  (red : ℕ)
  (blonde : ℕ)
  (black : ℕ)

/-- Calculates the total parts in the ratio --/
def totalParts (ratio : HairColorRatio) : ℕ :=
  ratio.red + ratio.blonde + ratio.black

/-- Theorem: Given the hair color ratio and number of red-haired children, 
    prove the total number of children in the class --/
theorem class_size_from_ratio_and_red_hair_count 
  (ratio : HairColorRatio) 
  (red_hair_count : ℕ) 
  (h1 : ratio.red = 3) 
  (h2 : ratio.blonde = 6) 
  (h3 : ratio.black = 7) 
  (h4 : red_hair_count = 9) : 
  (red_hair_count * totalParts ratio) / ratio.red = 48 := by
  sorry

end class_size_from_ratio_and_red_hair_count_l2149_214940


namespace playground_width_l2149_214958

theorem playground_width (area : ℝ) (length : ℝ) (h1 : area = 143.2) (h2 : length = 4) :
  area / length = 35.8 := by
  sorry

end playground_width_l2149_214958


namespace geometric_sequence_first_term_l2149_214994

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_first_term
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequence a)
  (h_sixth : a 6 = Nat.factorial 9)
  (h_ninth : a 9 = Nat.factorial 10) :
  a 1 = (Nat.factorial 9) / (10 * Real.rpow 10 (1/3)) := by
  sorry

#check geometric_sequence_first_term

end geometric_sequence_first_term_l2149_214994


namespace max_area_rectangular_pen_max_area_divided_pen_l2149_214977

/-- The maximum area of a rectangular pen given a fixed perimeter --/
theorem max_area_rectangular_pen (perimeter : ℝ) (area : ℝ) : 
  perimeter = 60 →
  area ≤ 225 ∧ 
  (∃ width height : ℝ, width > 0 ∧ height > 0 ∧ 2 * (width + height) = perimeter ∧ width * height = area) →
  (∀ width height : ℝ, width > 0 → height > 0 → 2 * (width + height) = perimeter → width * height ≤ 225) :=
by sorry

/-- The maximum area remains the same when divided into two equal sections --/
theorem max_area_divided_pen (perimeter : ℝ) (area : ℝ) (width height : ℝ) :
  perimeter = 60 →
  width > 0 →
  height > 0 →
  2 * (width + height) = perimeter →
  width * height = 225 →
  ∃ new_height : ℝ, new_height > 0 ∧ 2 * (width + new_height) = perimeter ∧ width * new_height = 225 / 2 :=
by sorry

end max_area_rectangular_pen_max_area_divided_pen_l2149_214977


namespace roden_fish_purchase_l2149_214982

/-- Represents the number of fish bought in a single visit -/
structure FishPurchase where
  goldfish : ℕ
  bluefish : ℕ
  greenfish : ℕ

/-- Calculates the total number of fish bought during three visits -/
def totalFish (visit1 visit2 visit3 : FishPurchase) : ℕ :=
  visit1.goldfish + visit1.bluefish + visit1.greenfish +
  visit2.goldfish + visit2.bluefish + visit2.greenfish +
  visit3.goldfish + visit3.bluefish + visit3.greenfish

theorem roden_fish_purchase :
  let visit1 : FishPurchase := { goldfish := 15, bluefish := 7, greenfish := 0 }
  let visit2 : FishPurchase := { goldfish := 10, bluefish := 12, greenfish := 5 }
  let visit3 : FishPurchase := { goldfish := 3, bluefish := 7, greenfish := 9 }
  totalFish visit1 visit2 visit3 = 68 := by
  sorry

end roden_fish_purchase_l2149_214982


namespace train_distance_problem_l2149_214936

theorem train_distance_problem (v1 v2 d : ℝ) (h1 : v1 = 20) (h2 : v2 = 25) (h3 : d = 65) :
  let t := d / (v1 + v2)
  let d1 := v1 * t
  let d2 := v2 * t
  d1 + d2 = 585 := by
sorry

end train_distance_problem_l2149_214936


namespace ice_cream_line_count_l2149_214937

theorem ice_cream_line_count (between : ℕ) (h : between = 5) : 
  between + 2 = 7 := by
  sorry

end ice_cream_line_count_l2149_214937


namespace pentagon_side_length_l2149_214974

/-- Given a triangle with all sides of length 20/9 cm and a pentagon with the same perimeter
    and all sides of equal length, the length of one side of the pentagon is 4/3 cm. -/
theorem pentagon_side_length (triangle_side : ℚ) (pentagon_side : ℚ) :
  triangle_side = 20 / 9 →
  3 * triangle_side = 5 * pentagon_side →
  pentagon_side = 4 / 3 := by
  sorry

#eval (4 : ℚ) / 3  -- Expected output: 4/3

end pentagon_side_length_l2149_214974


namespace unique_function_existence_l2149_214995

-- Define the type of real-valued functions
def RealFunction := ℝ → ℝ

-- State the theorem
theorem unique_function_existence :
  ∃! f : RealFunction, ∀ x y : ℝ, f (x + f y) = x + y + 1 :=
by
  -- The proof would go here
  sorry

end unique_function_existence_l2149_214995


namespace percentage_of_defective_meters_l2149_214983

theorem percentage_of_defective_meters
  (total_meters : ℕ)
  (rejected_meters : ℕ)
  (h1 : total_meters = 150)
  (h2 : rejected_meters = 15) :
  (rejected_meters : ℝ) / (total_meters : ℝ) * 100 = 10 := by
  sorry

end percentage_of_defective_meters_l2149_214983


namespace final_symbol_invariant_l2149_214925

/-- Represents the state of the blackboard -/
structure BlackboardState where
  minus_count : Nat
  total_count : Nat

/-- Represents a single operation on the blackboard -/
inductive Operation
  | erase_same_plus
  | erase_same_minus
  | erase_different

/-- Applies an operation to the blackboard state -/
def apply_operation (state : BlackboardState) (op : Operation) : BlackboardState :=
  match op with
  | Operation.erase_same_plus => ⟨state.minus_count, state.total_count - 1⟩
  | Operation.erase_same_minus => ⟨state.minus_count - 2, state.total_count - 1⟩
  | Operation.erase_different => ⟨state.minus_count, state.total_count - 1⟩

/-- The main theorem stating that the final symbol is determined by the initial parity of minus signs -/
theorem final_symbol_invariant (initial_state : BlackboardState)
  (h_initial : initial_state.total_count = 1967)
  (h_valid : initial_state.minus_count ≤ initial_state.total_count) :
  ∃ (final_symbol : Bool),
    ∀ (ops : List Operation),
      (ops.foldl apply_operation initial_state).total_count = 1 →
      final_symbol = ((ops.foldl apply_operation initial_state).minus_count % 2 = 1) :=
by sorry

end final_symbol_invariant_l2149_214925


namespace kerosene_cost_l2149_214971

/-- The cost of a pound of rice in dollars -/
def rice_cost : ℚ := 33 / 100

/-- The number of eggs in a dozen -/
def dozen : ℕ := 12

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

theorem kerosene_cost :
  ∀ (egg_cost : ℚ) (kerosene_half_liter_cost : ℚ),
    egg_cost * dozen = rice_cost →  -- A dozen eggs cost as much as a pound of rice
    kerosene_half_liter_cost = egg_cost * 8 →  -- A half-liter of kerosene costs as much as 8 eggs
    (2 * kerosene_half_liter_cost * cents_per_dollar : ℚ) = 44 :=
by sorry

end kerosene_cost_l2149_214971


namespace function_zero_at_seven_fifths_l2149_214981

theorem function_zero_at_seven_fifths :
  let f : ℝ → ℝ := λ x ↦ 5 * x - 7
  f (7/5) = 0 := by
  sorry

end function_zero_at_seven_fifths_l2149_214981


namespace quadratic_function_inequality_l2149_214912

theorem quadratic_function_inequality (a b c : ℝ) (h1 : c > b) (h2 : b > a) 
  (h3 : a * 1^2 + 2 * b * 1 + c = 0) 
  (h4 : ∃ x, a * x^2 + 2 * b * x + c = -a) : 
  0 ≤ b / a ∧ b / a < 1 := by sorry

end quadratic_function_inequality_l2149_214912


namespace gcd_lcm_45_150_l2149_214993

theorem gcd_lcm_45_150 : Nat.gcd 45 150 = 15 ∧ Nat.lcm 45 150 = 450 := by
  sorry

end gcd_lcm_45_150_l2149_214993


namespace max_value_at_neg_two_l2149_214932

-- Define the function f(x)
def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

-- State the theorem
theorem max_value_at_neg_two (c : ℝ) :
  (∀ x : ℝ, f c (-2) ≥ f c x) → c = -2 :=
by sorry

end max_value_at_neg_two_l2149_214932


namespace fraction_simplification_l2149_214910

theorem fraction_simplification (d : ℝ) : (5 + 4 * d) / 7 + 3 = (26 + 4 * d) / 7 := by
  sorry

end fraction_simplification_l2149_214910


namespace unique_solution_x_three_halves_l2149_214961

theorem unique_solution_x_three_halves :
  ∃! x : ℝ, ∀ y : ℝ, (y = (x^2 - 9) / (x - 3) ∧ y = 3*x) → x = 3/2 :=
by sorry

end unique_solution_x_three_halves_l2149_214961


namespace cookie_cost_is_18_l2149_214959

/-- The cost of each cookie Cora buys in April -/
def cookie_cost (cookies_per_day : ℕ) (days_in_april : ℕ) (total_spent : ℕ) : ℚ :=
  total_spent / (cookies_per_day * days_in_april)

/-- Theorem stating that each cookie costs 18 dollars -/
theorem cookie_cost_is_18 :
  cookie_cost 3 30 1620 = 18 := by sorry

end cookie_cost_is_18_l2149_214959


namespace area_of_quadrilateral_KLMN_l2149_214943

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)

-- Define points K, L, N
structure Points (t : Triangle) :=
  (bk : ℝ)
  (bl : ℝ)
  (an : ℝ)

-- Define the quadrilateral KLMN
def quadrilateral_area (t : Triangle) (p : Points t) : ℝ := sorry

-- Theorem statement
theorem area_of_quadrilateral_KLMN :
  let t : Triangle := { a := 13, b := 14, c := 15 }
  let p : Points t := { bk := 14/13, bl := 1, an := 10 }
  quadrilateral_area t p = 36503/1183 := by sorry

end area_of_quadrilateral_KLMN_l2149_214943


namespace complement_of_A_union_B_l2149_214900

-- Define the sets A and B
def A : Set ℝ := {x | x < 1}
def B : Set ℝ := {x | x < 0}

-- State the theorem
theorem complement_of_A_union_B :
  (A ∪ B)ᶜ = {x : ℝ | x ≥ 1} := by sorry

end complement_of_A_union_B_l2149_214900


namespace actual_sampling_method_is_other_l2149_214938

/-- Represents the sampling method used in the survey --/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic
  | Other

/-- Represents the characteristics of the sampling process --/
structure SamplingProcess where
  location : String
  selection : String
  endCondition : String

/-- The actual sampling process used in the survey --/
def actualSamplingProcess : SamplingProcess :=
  { location := "shopping mall entrance",
    selection := "randomly selected individuals",
    endCondition := "until predetermined number of respondents reached" }

/-- Theorem stating that the actual sampling method is not one of the three standard methods --/
theorem actual_sampling_method_is_other (sm : SamplingMethod) 
  (h : sm = SamplingMethod.SimpleRandom ∨ 
       sm = SamplingMethod.Stratified ∨ 
       sm = SamplingMethod.Systematic) : 
  sm ≠ SamplingMethod.Other → False := by
  sorry

end actual_sampling_method_is_other_l2149_214938


namespace number_equation_solution_l2149_214949

theorem number_equation_solution : 
  ∃ x : ℝ, (3034 - (x / 20.04) = 2984) ∧ (x = 1002) := by
  sorry

end number_equation_solution_l2149_214949


namespace exists_close_to_integer_l2149_214906

theorem exists_close_to_integer (a : ℝ) (n : ℕ) (ha : a > 0) (hn : n > 1) :
  ∃ k : ℕ, 1 ≤ k ∧ k < n ∧ ∃ m : ℤ, |k * a - m| ≤ 1 / n := by
  sorry

end exists_close_to_integer_l2149_214906


namespace no_perfect_squares_l2149_214988

-- Define the factorial function
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

-- State the theorem
theorem no_perfect_squares : 
  ¬(is_perfect_square (factorial 100 * factorial 101)) ∧
  ¬(is_perfect_square (factorial 100 * factorial 102)) ∧
  ¬(is_perfect_square (factorial 101 * factorial 102)) ∧
  ¬(is_perfect_square (factorial 101 * factorial 103)) ∧
  ¬(is_perfect_square (factorial 102 * factorial 103)) := by
  sorry

end no_perfect_squares_l2149_214988


namespace palindrome_product_sum_l2149_214956

/-- A positive three-digit palindrome is a number between 100 and 999 (inclusive) that reads the same forwards and backwards. -/
def IsPositiveThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10) ∧ ((n / 10) % 10 = (n % 100) / 10)

/-- The theorem stating that if there exist two positive three-digit palindromes whose product is 445,545, then their sum is 1436. -/
theorem palindrome_product_sum : 
  ∃ (a b : ℕ), IsPositiveThreeDigitPalindrome a ∧ 
                IsPositiveThreeDigitPalindrome b ∧ 
                a * b = 445545 → 
                a + b = 1436 := by
  sorry

end palindrome_product_sum_l2149_214956


namespace alpha_beta_values_l2149_214929

theorem alpha_beta_values (n k : ℤ) :
  let α : ℝ := π / 4 + 2 * π * (n : ℝ)
  let β : ℝ := π / 3 + 2 * π * (k : ℝ)
  (∃ m : ℤ, α = π / 4 + 2 * π * (m : ℝ)) ∧
  (∃ l : ℤ, β = π / 3 + 2 * π * (l : ℝ)) :=
by sorry

end alpha_beta_values_l2149_214929


namespace part_one_part_two_l2149_214985

-- Part 1
theorem part_one (x : ℝ) (a b : ℝ × ℝ) :
  a = (Real.sqrt 3 * Real.sin x, -1) →
  b = (Real.cos x, Real.sqrt 3) →
  ∃ (k : ℝ), a = k • b →
  (3 * Real.sin x - Real.cos x) / (Real.sin x + Real.cos x) = -3 :=
sorry

-- Part 2
def f (x m : ℝ) (a b : ℝ × ℝ) : ℝ :=
  2 * ((a.1 + b.1) * b.1 + (a.2 + b.2) * b.2) - 2 * m^2 - 1

theorem part_two (m : ℝ) :
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧
    f x m ((Real.sqrt 3 * Real.sin x, -1)) ((Real.cos x, m)) = 0) →
  m ∈ Set.Icc (-1/2) 1 :=
sorry

end part_one_part_two_l2149_214985


namespace negation_equivalence_l2149_214934

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) := by
  sorry

end negation_equivalence_l2149_214934


namespace f_property_l2149_214939

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 2

-- State the theorem
theorem f_property (a b : ℝ) : f a b (-2) = -7 → f a b 2 = 11 := by
  sorry

end f_property_l2149_214939


namespace profit_calculation_l2149_214908

/-- Represents the profit distribution in a partnership --/
structure ProfitDistribution where
  mary_investment : ℝ
  mike_investment : ℝ
  total_profit : ℝ
  effort_share : ℝ
  investment_share : ℝ
  mary_extra : ℝ

/-- Theorem stating the profit calculation based on given conditions --/
theorem profit_calculation (pd : ProfitDistribution) 
  (h1 : pd.mary_investment = 600)
  (h2 : pd.mike_investment = 400)
  (h3 : pd.effort_share = 1/3)
  (h4 : pd.investment_share = 2/3)
  (h5 : pd.mary_extra = 1000)
  (h6 : pd.effort_share + pd.investment_share = 1) :
  pd.total_profit = 15000 := by
  sorry

#check profit_calculation

end profit_calculation_l2149_214908


namespace truncation_result_l2149_214963

/-- Represents a convex polyhedron -/
structure ConvexPolyhedron where
  edges : ℕ
  convex : Bool

/-- Represents a truncated convex polyhedron -/
structure TruncatedPolyhedron where
  original : ConvexPolyhedron
  vertices : ℕ
  edges : ℕ
  truncated : Bool

/-- Function that performs truncation on a convex polyhedron -/
def truncate (p : ConvexPolyhedron) : TruncatedPolyhedron :=
  { original := p
  , vertices := 2 * p.edges
  , edges := 3 * p.edges
  , truncated := true }

/-- Theorem stating the result of truncating a specific convex polyhedron -/
theorem truncation_result :
  ∀ (p : ConvexPolyhedron),
  p.edges = 100 →
  p.convex = true →
  let tp := truncate p
  tp.vertices = 200 ∧ tp.edges = 300 := by
  sorry

end truncation_result_l2149_214963


namespace parallel_transitivity_parallel_planes_imply_parallel_line_perpendicular_implies_parallel_perpendicular_planes_imply_parallel_line_l2149_214930

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (in_plane : Line → Plane → Prop)

-- Define the lines and planes
variable (m n : Line) (α β : Plane)

-- State the conditions
variable (h1 : ¬in_plane m α)
variable (h2 : ¬in_plane m β)
variable (h3 : ¬in_plane n α)
variable (h4 : ¬in_plane n β)

-- State the theorems to be proved
theorem parallel_transitivity 
  (h5 : parallel m n) (h6 : parallel_plane n α) : 
  parallel_plane m α := by sorry

theorem parallel_planes_imply_parallel_line 
  (h5 : parallel_plane m β) (h6 : parallel_planes α β) : 
  parallel_plane m α := by sorry

theorem perpendicular_implies_parallel 
  (h5 : perpendicular m n) (h6 : perpendicular_plane n α) : 
  parallel_plane m α := by sorry

theorem perpendicular_planes_imply_parallel_line 
  (h5 : perpendicular_plane m β) (h6 : perpendicular_planes α β) : 
  parallel_plane m α := by sorry

end parallel_transitivity_parallel_planes_imply_parallel_line_perpendicular_implies_parallel_perpendicular_planes_imply_parallel_line_l2149_214930


namespace zoom_download_time_l2149_214975

theorem zoom_download_time (total_time audio_glitch_time video_glitch_time : ℕ)
  (h_total : total_time = 82)
  (h_audio : audio_glitch_time = 2 * 4)
  (h_video : video_glitch_time = 6)
  (h_glitch_ratio : 2 * (audio_glitch_time + video_glitch_time) = total_time - (audio_glitch_time + video_glitch_time) - 40) :
  let mac_download_time := (total_time - (audio_glitch_time + video_glitch_time) - 2 * (audio_glitch_time + video_glitch_time)) / 4
  mac_download_time = 10 := by sorry

end zoom_download_time_l2149_214975


namespace dress_price_after_discounts_l2149_214984

theorem dress_price_after_discounts (d : ℝ) : 
  let initial_discount_rate : ℝ := 0.65
  let staff_discount_rate : ℝ := 0.60
  let price_after_initial_discount : ℝ := d * (1 - initial_discount_rate)
  let final_price : ℝ := price_after_initial_discount * (1 - staff_discount_rate)
  final_price = d * 0.14 :=
by sorry

end dress_price_after_discounts_l2149_214984


namespace min_value_theorem_l2149_214962

theorem min_value_theorem (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) :
  2/x + 9/(1-2*x) ≥ 25 ∧ ∃ y, 0 < y ∧ y < 1/2 ∧ 2/y + 9/(1-2*y) = 25 := by
  sorry

end min_value_theorem_l2149_214962


namespace age_sum_theorem_l2149_214904

theorem age_sum_theorem (a b c : ℕ) : 
  a = b + c + 20 → 
  a^2 = (b + c)^2 + 2050 → 
  a + b + c = 80 :=
by
  sorry

end age_sum_theorem_l2149_214904


namespace prob_at_least_one_white_l2149_214902

/-- Given a bag with 5 balls where the probability of drawing 2 white balls out of 2 draws is 3/10,
    prove that the probability of getting at least 1 white ball in 2 draws is 9/10. -/
theorem prob_at_least_one_white (total_balls : ℕ) (prob_two_white : ℚ) :
  total_balls = 5 →
  prob_two_white = 3 / 10 →
  (∃ white_balls : ℕ, white_balls ≤ total_balls ∧
    prob_two_white = (white_balls.choose 2 : ℚ) / (total_balls.choose 2 : ℚ)) →
  (1 : ℚ) - ((total_balls - white_balls).choose 2 : ℚ) / (total_balls.choose 2 : ℚ) = 9 / 10 :=
by sorry

end prob_at_least_one_white_l2149_214902


namespace unique_solution_abc_l2149_214978

/-- Represents a base-7 number with two digits --/
def Base7TwoDigit (a b : ℕ) : ℕ := 7 * a + b

/-- Represents a base-7 number with one digit --/
def Base7OneDigit (c : ℕ) : ℕ := c

/-- Represents a base-7 number with two digits, where the first digit is 'c' and the second is 0 --/
def Base7TwoDigitWithZero (c : ℕ) : ℕ := 7 * c

theorem unique_solution_abc (A B C : ℕ) :
  (0 < A ∧ A < 7) →
  (0 < B ∧ B < 7) →
  (0 < C ∧ C < 7) →
  Base7TwoDigit A B + Base7OneDigit C = Base7TwoDigitWithZero C →
  Base7TwoDigit A B + Base7TwoDigit B A = Base7TwoDigit C C →
  A = 3 ∧ B = 2 ∧ C = 5 := by
  sorry

end unique_solution_abc_l2149_214978


namespace horseshoe_production_theorem_l2149_214967

/-- Represents the manufacturing and sales scenario for horseshoes --/
structure HorseshoeScenario where
  initialOutlay : ℕ
  costPerSet : ℕ
  sellingPricePerSet : ℕ
  profit : ℕ

/-- Calculates the number of sets of horseshoes produced and sold --/
def setsProducedAndSold (scenario : HorseshoeScenario) : ℕ :=
  (scenario.profit + scenario.initialOutlay) / (scenario.sellingPricePerSet - scenario.costPerSet)

/-- Theorem stating that the number of sets produced and sold is 500 --/
theorem horseshoe_production_theorem (scenario : HorseshoeScenario) 
  (h1 : scenario.initialOutlay = 10000)
  (h2 : scenario.costPerSet = 20)
  (h3 : scenario.sellingPricePerSet = 50)
  (h4 : scenario.profit = 5000) :
  setsProducedAndSold scenario = 500 := by
  sorry

#eval setsProducedAndSold { initialOutlay := 10000, costPerSet := 20, sellingPricePerSet := 50, profit := 5000 }

end horseshoe_production_theorem_l2149_214967


namespace lock_code_attempts_l2149_214960

theorem lock_code_attempts (num_digits : ℕ) (code_length : ℕ) : 
  num_digits = 5 → code_length = 3 → num_digits ^ code_length - 1 = 124 := by
  sorry

#eval 5^3 - 1  -- This should output 124

end lock_code_attempts_l2149_214960


namespace min_value_expression_l2149_214913

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 27) :
  ∃ (min : ℝ), min = 60 ∧ 
  (∀ a' b' c' : ℝ, a' > 0 → b' > 0 → c' > 0 → a' * b' * c' = 27 → 
    a'^2 + 6*a'*b' + 9*b'^2 + 3*c'^2 ≥ min) ∧
  (a^2 + 6*a*b + 9*b^2 + 3*c^2 = min) :=
sorry

end min_value_expression_l2149_214913


namespace solve_equation_l2149_214935

-- Define a custom pair type for real numbers
structure RealPair :=
  (fst : ℝ)
  (snd : ℝ)

-- Define equality for RealPair
def realPairEq (a b : RealPair) : Prop :=
  a.fst = b.fst ∧ a.snd = b.snd

-- Define the ⊕ operation
def oplus (a b : RealPair) : RealPair :=
  ⟨a.fst * b.fst - a.snd * b.snd, a.fst * b.snd + a.snd * b.fst⟩

-- Theorem statement
theorem solve_equation (p q : ℝ) :
  oplus ⟨1, 2⟩ ⟨p, q⟩ = ⟨5, 0⟩ → realPairEq ⟨p, q⟩ ⟨1, -2⟩ := by
  sorry

end solve_equation_l2149_214935


namespace solve_equation_l2149_214917

theorem solve_equation (x y : ℝ) (h1 : x^2 - x + 6 = y - 6) (h2 : x = -6) : y = 54 := by
  sorry

end solve_equation_l2149_214917


namespace coefficient_of_x_plus_two_to_ten_l2149_214926

theorem coefficient_of_x_plus_two_to_ten (x : ℝ) :
  ∃ (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ),
    (x + 1)^2 + (x + 1)^11 = a + a₁*(x + 2) + a₂*(x + 2)^2 + a₃*(x + 2)^3 + 
      a₄*(x + 2)^4 + a₅*(x + 2)^5 + a₆*(x + 2)^6 + a₇*(x + 2)^7 + 
      a₈*(x + 2)^8 + a₉*(x + 2)^9 + a₁₀*(x + 2)^10 + a₁₁*(x + 2)^11 ∧
    a₁₀ = -11 :=
by sorry

end coefficient_of_x_plus_two_to_ten_l2149_214926


namespace expression_evaluation_l2149_214927

theorem expression_evaluation :
  ∃ (m : ℕ+), (3^1002 + 7^1003)^2 - (3^1002 - 7^1003)^2 = m * 10^1003 ∧ m = 56 := by
  sorry

end expression_evaluation_l2149_214927


namespace percentage_problem_l2149_214997

theorem percentage_problem : 
  ∀ x : ℝ, (120 : ℝ) = 1.5 * x → x = 80 := by
  sorry

end percentage_problem_l2149_214997


namespace max_arrangement_length_l2149_214969

def student_height : ℝ → Prop := λ h => h = 1.60 ∨ h = 1.22

def valid_arrangement (arrangement : List ℝ) : Prop :=
  (∀ i, i + 3 < arrangement.length → 
    (arrangement.take (i + 4)).sum / 4 > 1.50) ∧
  (∀ i, i + 6 < arrangement.length → 
    (arrangement.take (i + 7)).sum / 7 < 1.50)

theorem max_arrangement_length :
  ∃ (arrangement : List ℝ),
    arrangement.length = 9 ∧
    (∀ h ∈ arrangement, student_height h) ∧
    valid_arrangement arrangement ∧
    ∀ (longer_arrangement : List ℝ),
      longer_arrangement.length > 9 →
      (∀ h ∈ longer_arrangement, student_height h) →
      ¬(valid_arrangement longer_arrangement) :=
sorry

end max_arrangement_length_l2149_214969


namespace seating_arrangement_count_l2149_214952

/-- Represents the seating arrangement problem --/
structure SeatingArrangement where
  front_seats : Nat
  back_seats : Nat
  people : Nat
  blocked_front : Nat

/-- Calculates the number of valid seating arrangements --/
def count_arrangements (s : SeatingArrangement) : Nat :=
  sorry

/-- Theorem stating the correct number of arrangements for the given problem --/
theorem seating_arrangement_count :
  let s : SeatingArrangement := {
    front_seats := 11,
    back_seats := 12,
    people := 2,
    blocked_front := 3
  }
  count_arrangements s = 346 := by
  sorry

end seating_arrangement_count_l2149_214952


namespace exist_ten_special_integers_l2149_214991

theorem exist_ten_special_integers : 
  ∃ (a : Fin 10 → ℕ+), 
    (∀ i j, i ≠ j → ¬(a i ∣ a j)) ∧ 
    (∀ i j, (a i)^2 ∣ a j) := by
  sorry

end exist_ten_special_integers_l2149_214991


namespace line_slope_45_degrees_l2149_214957

theorem line_slope_45_degrees (m : ℝ) : 
  let P : ℝ × ℝ := (-2, m)
  let Q : ℝ × ℝ := (m, 4)
  (4 - m) / (m - (-2)) = 1 → m = 1 := by
sorry

end line_slope_45_degrees_l2149_214957


namespace lucas_chocolate_candy_l2149_214947

/-- The number of pieces of chocolate candy Lucas makes for each student on Monday -/
def pieces_per_student : ℕ := 4

/-- The number of students not coming to class this upcoming Monday -/
def absent_students : ℕ := 3

/-- The number of pieces of chocolate candy Lucas will make this upcoming Monday -/
def upcoming_monday_pieces : ℕ := 28

/-- The number of pieces of chocolate candy Lucas made last Monday -/
def last_monday_pieces : ℕ := pieces_per_student * (upcoming_monday_pieces / pieces_per_student + absent_students)

theorem lucas_chocolate_candy : last_monday_pieces = 40 := by sorry

end lucas_chocolate_candy_l2149_214947


namespace N_divisible_by_7_and_9_l2149_214979

def N : ℕ := 1234567765432  -- This is the octal representation as a decimal number

theorem N_divisible_by_7_and_9 : 
  7 ∣ N ∧ 9 ∣ N :=
sorry

end N_divisible_by_7_and_9_l2149_214979


namespace beta_max_success_ratio_l2149_214920

/-- Represents a contestant's scores in a two-day math contest -/
structure ContestScores where
  day1_score : ℕ
  day1_total : ℕ
  day2_score : ℕ
  day2_total : ℕ

/-- The maximum possible two-day success ratio for Beta -/
def beta_max_ratio : ℚ := 407 / 600

theorem beta_max_success_ratio 
  (alpha : ContestScores)
  (beta : ContestScores)
  (h1 : alpha.day1_score = 180 ∧ alpha.day1_total = 350)
  (h2 : alpha.day2_score = 170 ∧ alpha.day2_total = 250)
  (h3 : beta.day1_score > 0 ∧ beta.day2_score > 0)
  (h4 : beta.day1_total + beta.day2_total = 600)
  (h5 : (beta.day1_score : ℚ) / beta.day1_total < (alpha.day1_score : ℚ) / alpha.day1_total)
  (h6 : (beta.day2_score : ℚ) / beta.day2_total < (alpha.day2_score : ℚ) / alpha.day2_total)
  (h7 : (alpha.day1_score + alpha.day2_score : ℚ) / (alpha.day1_total + alpha.day2_total) = 7 / 12) :
  (∀ b : ContestScores, 
    b.day1_score > 0 ∧ b.day2_score > 0 →
    b.day1_total + b.day2_total = 600 →
    (b.day1_score : ℚ) / b.day1_total < (alpha.day1_score : ℚ) / alpha.day1_total →
    (b.day2_score : ℚ) / b.day2_total < (alpha.day2_score : ℚ) / alpha.day2_total →
    (b.day1_score + b.day2_score : ℚ) / (b.day1_total + b.day2_total) ≤ beta_max_ratio) :=
by
  sorry

end beta_max_success_ratio_l2149_214920


namespace ninety_nine_times_one_hundred_one_l2149_214951

theorem ninety_nine_times_one_hundred_one : 99 * 101 = 9999 := by
  sorry

end ninety_nine_times_one_hundred_one_l2149_214951


namespace intersection_of_A_and_B_l2149_214954

def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end intersection_of_A_and_B_l2149_214954


namespace root_difference_quadratic_l2149_214965

theorem root_difference_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  2 * r₁^2 + 5 * r₁ = 12 ∧
  2 * r₂^2 + 5 * r₂ = 12 ∧
  abs (r₁ - r₂) = 5.5 :=
by sorry

end root_difference_quadratic_l2149_214965


namespace fixed_point_exponential_function_l2149_214990

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := fun x ↦ 4 + 2 * a^(x - 1)
  f 1 = 6 := by
  sorry

end fixed_point_exponential_function_l2149_214990


namespace lattice_right_triangles_with_specific_incenter_l2149_214955

/-- A point with integer coordinates -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A right triangle with vertices O, A, and B, where O is the origin and the right angle -/
structure LatticeRightTriangle where
  A : LatticePoint
  B : LatticePoint

/-- The incenter of a right triangle -/
def incenter (t : LatticeRightTriangle) : ℚ × ℚ :=
  let a : ℚ := t.A.x
  let b : ℚ := t.B.y
  let c : ℚ := (a^2 + b^2).sqrt
  ((a + b - c) / 2, (a + b - c) / 2)

theorem lattice_right_triangles_with_specific_incenter :
  ∃ (n : ℕ), n > 0 ∧
  ∃ (triangles : Finset LatticeRightTriangle),
    triangles.card = n ∧
    ∀ t ∈ triangles, incenter t = (2015, 14105) := by
  sorry

end lattice_right_triangles_with_specific_incenter_l2149_214955


namespace least_divisible_by_10_to_15_divided_by_26_l2149_214987

theorem least_divisible_by_10_to_15_divided_by_26 :
  let j := Nat.lcm 10 (Nat.lcm 11 (Nat.lcm 12 (Nat.lcm 13 (Nat.lcm 14 15))))
  ∀ k : ℕ, (∀ i ∈ Finset.range 6, k % (i + 10) = 0) → k ≥ j
  → j / 26 = 2310 := by
  sorry

end least_divisible_by_10_to_15_divided_by_26_l2149_214987


namespace sum_multiple_of_three_l2149_214921

theorem sum_multiple_of_three (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 6 * k) 
  (hb : ∃ m : ℤ, b = 9 * m) : 
  ∃ n : ℤ, a + b = 3 * n := by
  sorry

end sum_multiple_of_three_l2149_214921


namespace combine_squared_binomial_simplify_given_equation_solve_system_of_equations_l2149_214916

-- Problem 1
theorem combine_squared_binomial (m n : ℝ) :
  3 * (m - n)^2 - 4 * (m - n)^2 + 3 * (m - n)^2 = 2 * (m - n)^2 :=
sorry

-- Problem 2
theorem simplify_given_equation (x y : ℝ) (h : x^2 + 2*y = 4) :
  3*x^2 + 6*y - 2 = 10 :=
sorry

-- Problem 3
theorem solve_system_of_equations (x y : ℝ) 
  (h1 : x^2 + x*y = 2) (h2 : 2*y^2 + 3*x*y = 5) :
  2*x^2 + 11*x*y + 6*y^2 = 19 :=
sorry

end combine_squared_binomial_simplify_given_equation_solve_system_of_equations_l2149_214916


namespace quadratic_inequality_solution_l2149_214976

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, (x < -1/3 ∨ x > 1/2) ↔ a*x^2 + b*x + 2 < 0) → 
  a - b = -14 := by sorry

end quadratic_inequality_solution_l2149_214976


namespace opposite_sides_of_y_axis_l2149_214945

/-- Given points A and B on opposite sides of the y-axis, with B on the right side,
    prove that the x-coordinate of A is negative. -/
theorem opposite_sides_of_y_axis (a : ℝ) : 
  (∃ A B : ℝ × ℝ, A = (a, 1) ∧ B = (2, a) ∧ 
   (A.1 < 0 ∧ B.1 > 0) ∧ -- A and B are on opposite sides of the y-axis
   B.1 > 0) →              -- B is on the right side of the y-axis
  a < 0 := by
sorry

end opposite_sides_of_y_axis_l2149_214945


namespace log_cube_l2149_214953

theorem log_cube (x : ℝ) (h : Real.log x / Real.log 3 = 5) : 
  Real.log (x^3) / Real.log 3 = 15 := by
sorry

end log_cube_l2149_214953


namespace greatest_three_digit_multiple_of_17_l2149_214946

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l2149_214946


namespace team_leader_selection_l2149_214944

theorem team_leader_selection (n : ℕ) (h : n = 5) : n * (n - 1) = 20 := by
  sorry

end team_leader_selection_l2149_214944


namespace inequality_proof_l2149_214923

theorem inequality_proof (a b c : ℝ) (n : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hn : n ≥ 2) (habc : a * b * c = 1) :
  (a / (b + c)^(1 / n : ℝ)) + (b / (c + a)^(1 / n : ℝ)) + (c / (a + b)^(1 / n : ℝ)) ≥ 3 / (2^(1 / n : ℝ)) := by
  sorry

end inequality_proof_l2149_214923


namespace wilsons_theorem_l2149_214996

theorem wilsons_theorem (p : ℕ) (h : p > 1) : 
  Nat.Prime p ↔ (Nat.factorial (p - 1) : ℤ) % p = p - 1 := by
  sorry

end wilsons_theorem_l2149_214996


namespace container_volume_comparison_l2149_214907

theorem container_volume_comparison (x y : ℝ) (h : x ≠ y) :
  x^3 + y^3 > x^2*y + x*y^2 := by
  sorry

#check container_volume_comparison

end container_volume_comparison_l2149_214907


namespace juan_birth_year_l2149_214941

def first_btc_year : ℕ := 1990
def btc_frequency : ℕ := 2
def juan_age_at_fifth_btc : ℕ := 14

def btc_year (n : ℕ) : ℕ := first_btc_year + (n - 1) * btc_frequency

theorem juan_birth_year :
  first_btc_year = 1990 →
  btc_frequency = 2 →
  juan_age_at_fifth_btc = 14 →
  btc_year 5 - juan_age_at_fifth_btc = 1984 :=
by
  sorry

end juan_birth_year_l2149_214941


namespace log_3125_base_5_between_consecutive_integers_l2149_214992

theorem log_3125_base_5_between_consecutive_integers :
  ∃ (c d : ℤ), c + 1 = d ∧ (c : ℝ) < Real.log 3125 / Real.log 5 ∧ Real.log 3125 / Real.log 5 < (d : ℝ) ∧ c + d = 10 := by
  sorry

end log_3125_base_5_between_consecutive_integers_l2149_214992


namespace total_garbage_accumulation_l2149_214924

/-- Represents the garbage accumulation problem in Daniel's neighborhood --/
def garbage_accumulation (collection_days_per_week : ℕ) (kg_per_collection : ℝ) (weeks : ℕ) (reduction_factor : ℝ) : ℝ :=
  let week1_accumulation := collection_days_per_week * kg_per_collection
  let week2_accumulation := week1_accumulation * reduction_factor
  week1_accumulation + week2_accumulation

/-- Theorem stating the total garbage accumulated over two weeks --/
theorem total_garbage_accumulation :
  garbage_accumulation 3 200 2 0.5 = 900 := by
  sorry

#eval garbage_accumulation 3 200 2 0.5

end total_garbage_accumulation_l2149_214924


namespace pie_eating_contest_ratio_l2149_214914

theorem pie_eating_contest_ratio (bill_pies sierra_pies adam_pies : ℕ) :
  adam_pies = bill_pies + 3 →
  sierra_pies = 12 →
  bill_pies + adam_pies + sierra_pies = 27 →
  sierra_pies / bill_pies = 2 := by
  sorry

end pie_eating_contest_ratio_l2149_214914


namespace square_perimeter_l2149_214986

theorem square_perimeter (area : ℝ) (side : ℝ) (h1 : area = 900) (h2 : side * side = area) :
  4 * side = 120 := by
  sorry

end square_perimeter_l2149_214986


namespace probability_four_twos_l2149_214972

def num_dice : ℕ := 12
def num_sides : ℕ := 8
def target_number : ℕ := 2
def num_success : ℕ := 4

theorem probability_four_twos : 
  (Nat.choose num_dice num_success : ℚ) * (1 / num_sides : ℚ)^num_success * ((num_sides - 1) / num_sides : ℚ)^(num_dice - num_success) = 
  495 * (1 / 4096 : ℚ) * (5764801 / 16777216 : ℚ) := by
sorry

end probability_four_twos_l2149_214972


namespace arithmetic_sequence_sum_l2149_214968

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 + a 3 + a 5 = 9 →
  a 2 + a 4 + a 6 = 15 →
  a 3 + a 4 = 8 := by
sorry

end arithmetic_sequence_sum_l2149_214968


namespace gloin_tells_truth_l2149_214922

/-- Represents the type of dwarf: either a knight or a liar -/
inductive DwarfType
  | Knight
  | Liar

/-- Represents a dwarf with their position and type -/
structure Dwarf :=
  (position : Nat)
  (type : DwarfType)

/-- The statement made by a dwarf -/
def statement (d : Dwarf) (line : List Dwarf) : Prop :=
  match d.position with
  | 10 => ∃ (right : Dwarf), right.position > d.position ∧ right.type = DwarfType.Knight
  | _ => ∃ (left : Dwarf), left.position < d.position ∧ left.type = DwarfType.Knight

/-- The main theorem -/
theorem gloin_tells_truth 
  (line : List Dwarf) 
  (h_count : line.length = 10)
  (h_knight : ∃ d ∈ line, d.type = DwarfType.Knight)
  (h_statements : ∀ d ∈ line, d.position ≠ 10 → 
    (d.type = DwarfType.Knight ↔ statement d line))
  (h_gloin : ∃ gloin ∈ line, gloin.position = 10)
  : ∃ gloin ∈ line, gloin.position = 10 ∧ gloin.type = DwarfType.Knight :=
sorry

end gloin_tells_truth_l2149_214922


namespace f_properties_l2149_214973

noncomputable def f (x : ℝ) := 2 * abs (Real.sin x + Real.cos x) - Real.sin (2 * x)

theorem f_properties :
  (∀ x, f (π / 2 - x) = f x) ∧
  (∀ x, f x ≥ 1) ∧
  (∀ x y, π / 4 ≤ x ∧ x ≤ y ∧ y ≤ π / 2 → f x ≤ f y) :=
by sorry

end f_properties_l2149_214973


namespace two_digit_multiple_plus_two_l2149_214905

theorem two_digit_multiple_plus_two : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ ∃ k : ℕ, n = 3 * 4 * 5 * k + 2 :=
by
  -- The proof would go here
  sorry

end two_digit_multiple_plus_two_l2149_214905


namespace parallel_planes_condition_l2149_214964

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation
variable (parallel : Plane → Plane → Prop)
variable (lineParallel : Line → Line → Prop)

-- Define the "in plane" relation
variable (inPlane : Line → Plane → Prop)

-- Define the intersection relation
variable (intersect : Line → Line → Prop)

-- Define our specific planes and lines
variable (α β : Plane)
variable (m n l₁ l₂ : Line)

-- State the theorem
theorem parallel_planes_condition
  (h1 : m ≠ n)
  (h2 : inPlane m α)
  (h3 : inPlane n α)
  (h4 : inPlane l₁ β)
  (h5 : inPlane l₂ β)
  (h6 : intersect l₁ l₂) :
  (lineParallel m l₁ ∧ lineParallel n l₂ → parallel α β) ∧
  ¬(parallel α β → lineParallel m l₁ ∧ lineParallel n l₂) :=
sorry

end parallel_planes_condition_l2149_214964


namespace intersection_M_complement_N_l2149_214998

-- Define the set M
def M : Set ℝ := {0, 1, 2}

-- Define the set N
def N : Set ℝ := {x | x^2 - 3*x + 2 > 0}

-- Theorem statement
theorem intersection_M_complement_N : M ∩ (Set.univ \ N) = {1, 2} := by
  sorry

end intersection_M_complement_N_l2149_214998


namespace pumpkin_price_theorem_l2149_214970

-- Define the prices of seeds
def tomato_price : ℚ := 1.5
def chili_price : ℚ := 0.9

-- Define the total spent and the number of packets bought
def total_spent : ℚ := 18
def pumpkin_packets : ℕ := 3
def tomato_packets : ℕ := 4
def chili_packets : ℕ := 5

-- Define the theorem
theorem pumpkin_price_theorem :
  ∃ (pumpkin_price : ℚ),
    pumpkin_price * pumpkin_packets +
    tomato_price * tomato_packets +
    chili_price * chili_packets = total_spent ∧
    pumpkin_price = 2.5 := by
  sorry

end pumpkin_price_theorem_l2149_214970


namespace ball_distribution_after_four_rounds_l2149_214980

/-- Represents the state of the game at any point -/
structure GameState :=
  (a b c d e : ℕ)

/-- Represents a single round of the game -/
def gameRound (s : GameState) : GameState :=
  let a' := if s.e < s.a then s.a - 2 else s.a
  let b' := if s.a < s.b then s.b - 2 else s.b
  let c' := if s.b < s.c then s.c - 2 else s.c
  let d' := if s.c < s.d then s.d - 2 else s.d
  let e' := if s.d < s.e then s.e - 2 else s.e
  ⟨a', b', c', d', e'⟩

/-- Represents the initial state of the game -/
def initialState : GameState := ⟨2, 4, 6, 8, 10⟩

/-- Represents the state after 4 rounds -/
def finalState : GameState := (gameRound ∘ gameRound ∘ gameRound ∘ gameRound) initialState

/-- The main theorem to be proved -/
theorem ball_distribution_after_four_rounds :
  finalState = ⟨6, 6, 6, 6, 6⟩ := by sorry

end ball_distribution_after_four_rounds_l2149_214980


namespace cubic_root_sum_cubes_l2149_214966

theorem cubic_root_sum_cubes (r s t : ℝ) : 
  (6 * r^3 + 504 * r + 1008 = 0) →
  (6 * s^3 + 504 * s + 1008 = 0) →
  (6 * t^3 + 504 * t + 1008 = 0) →
  (r + s)^3 + (s + t)^3 + (t + r)^3 = 504 := by
  sorry

end cubic_root_sum_cubes_l2149_214966


namespace bruce_payment_l2149_214915

/-- The amount Bruce paid to the shopkeeper for grapes and mangoes -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Bruce paid 1000 to the shopkeeper -/
theorem bruce_payment : total_amount 8 70 8 55 = 1000 := by
  sorry

end bruce_payment_l2149_214915


namespace time_to_return_is_45_minutes_l2149_214918

/-- Represents a hiker's journey on a trail --/
structure HikerJourney where
  rate : Real  -- Minutes per kilometer
  initialDistance : Real  -- Kilometers hiked east initially
  totalDistance : Real  -- Total kilometers hiked east before turning back
  
/-- Calculates the time required for a hiker to return to the start of the trail --/
def timeToReturn (journey : HikerJourney) : Real :=
  sorry

/-- Theorem stating that for the given conditions, the time to return is 45 minutes --/
theorem time_to_return_is_45_minutes (journey : HikerJourney) 
  (h1 : journey.rate = 10)
  (h2 : journey.initialDistance = 2.5)
  (h3 : journey.totalDistance = 3.5) :
  timeToReturn journey = 45 := by
  sorry

end time_to_return_is_45_minutes_l2149_214918


namespace orthocenter_of_specific_triangle_l2149_214999

/-- The orthocenter of a triangle is the point where all three altitudes intersect. -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Given three points A, B, and C in 3D space, this theorem states that
    the orthocenter of the triangle formed by these points is (2, 3, 4). -/
theorem orthocenter_of_specific_triangle :
  let A : ℝ × ℝ × ℝ := (2, 3, 4)
  let B : ℝ × ℝ × ℝ := (6, -1, 2)
  let C : ℝ × ℝ × ℝ := (1, 6, 5)
  orthocenter A B C = (2, 3, 4) := by sorry

end orthocenter_of_specific_triangle_l2149_214999
