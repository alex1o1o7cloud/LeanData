import Mathlib

namespace apples_problem_l2539_253904

/-- The number of apples Adam and Jackie have together -/
def total_apples (adam : ℕ) (jackie : ℕ) : ℕ := adam + jackie

/-- Adam has 9 more apples than the total -/
def adam_more_than_total (adam : ℕ) (jackie : ℕ) : Prop :=
  adam = total_apples adam jackie + 9

/-- Adam has 8 more apples than Jackie -/
def adam_more_than_jackie (adam : ℕ) (jackie : ℕ) : Prop :=
  adam = jackie + 8

theorem apples_problem (adam jackie : ℕ) 
  (h1 : adam_more_than_total adam jackie)
  (h2 : adam_more_than_jackie adam jackie)
  (h3 : adam = 21) : 
  total_apples adam jackie = 34 := by
  sorry

end apples_problem_l2539_253904


namespace solve_quadratic_equation_l2539_253911

theorem solve_quadratic_equation :
  ∃ x : ℚ, (10 - x)^2 = x^2 + 4 ∧ x = 24/5 := by
  sorry

end solve_quadratic_equation_l2539_253911


namespace innocent_knight_convincing_l2539_253971

-- Define the types of people
inductive PersonType
| Normal
| Knight
| Liar

-- Define the properties of a person
structure Person where
  type : PersonType
  guilty : Bool

-- Define the criminal
def criminal : Person := { type := PersonType.Liar, guilty := true }

-- Define the statement made by the person
def statement (p : Person) : Prop := p.type = PersonType.Knight ∧ ¬p.guilty

-- Theorem to prove
theorem innocent_knight_convincing (p : Person) 
  (h1 : p.type ≠ PersonType.Normal) 
  (h2 : ¬p.guilty) 
  (h3 : p.type ≠ PersonType.Liar) :
  statement p → (¬p.guilty ∧ p.type ≠ PersonType.Liar) :=
by sorry

end innocent_knight_convincing_l2539_253971


namespace sequence_general_term_l2539_253987

theorem sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, a (n + 1) = a n + 2) :
  ∀ n : ℕ, a n = 2 * n - 1 := by
  sorry

end sequence_general_term_l2539_253987


namespace divisible_by_eight_last_digits_l2539_253903

theorem divisible_by_eight_last_digits : 
  ∃! (S : Finset Nat), 
    (∀ n ∈ S, n < 10) ∧ 
    (∀ m : Nat, m % 8 = 0 → m % 10 ∈ S) ∧
    Finset.card S = 5 := by
  sorry

end divisible_by_eight_last_digits_l2539_253903


namespace arrangements_count_l2539_253946

/-- The number of arrangements for 7 students with specific conditions -/
def num_arrangements : ℕ :=
  let total_students : ℕ := 7
  let middle_student : ℕ := 1
  let together_students : ℕ := 2
  let remaining_students : ℕ := total_students - middle_student - together_students
  let ways_to_place_together : ℕ := 2  -- left or right of middle
  let arrangements_within_together : ℕ := 2  -- B-C or C-B
  let permutations_of_remaining : ℕ := Nat.factorial remaining_students
  ways_to_place_together * arrangements_within_together * permutations_of_remaining

theorem arrangements_count : num_arrangements = 192 := by
  sorry

end arrangements_count_l2539_253946


namespace inequality_proof_l2539_253919

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a * b + b * c + c * a = 1) :
  Real.sqrt (a^3 + a) + Real.sqrt (b^3 + b) + Real.sqrt (c^3 + c) ≥ 2 * Real.sqrt (a + b + c) := by
  sorry

end inequality_proof_l2539_253919


namespace max_value_of_f_l2539_253901

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 + 3 * x^2 + 5 * x + 2

theorem max_value_of_f :
  ∃ (M : ℝ), M = 31/3 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end max_value_of_f_l2539_253901


namespace geometric_arithmetic_sequence_ratio_l2539_253962

theorem geometric_arithmetic_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  (2 * a 3 = a 1 + a 2) →       -- arithmetic sequence condition
  (q = 1 ∨ q = -1/2) :=
by sorry

end geometric_arithmetic_sequence_ratio_l2539_253962


namespace hyperbola_equation_l2539_253978

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ c : ℝ, c = Real.sqrt 5 ∧ c^2 = a^2 + b^2) → 
  (b / a = 1 / 2) → 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 4 - y^2 = 1) :=
by sorry

end hyperbola_equation_l2539_253978


namespace translation_result_l2539_253917

/-- Translates a point in the 2D plane along the y-axis. -/
def translate_y (x y dy : ℝ) : ℝ × ℝ := (x, y + dy)

/-- The original point M. -/
def M : ℝ × ℝ := (-10, 1)

/-- The translation distance in the y-direction. -/
def dy : ℝ := 4

/-- The resulting point M₁ after translation. -/
def M₁ : ℝ × ℝ := translate_y M.1 M.2 dy

theorem translation_result :
  M₁ = (-10, 5) := by sorry

end translation_result_l2539_253917


namespace units_digit_7_pow_million_l2539_253985

def units_digit_cycle_7 : List Nat := [7, 9, 3, 1]

theorem units_digit_7_pow_million :
  ∃ (n : Nat), n < 10 ∧ (7^(10^6 : Nat)) % 10 = n ∧ n = 1 :=
by
  sorry

#check units_digit_7_pow_million

end units_digit_7_pow_million_l2539_253985


namespace indeterminate_magnitude_l2539_253958

/-- Given two approximate numbers A and B, prove that their relative magnitude cannot be determined. -/
theorem indeterminate_magnitude (A B : ℝ) (hA : 3.55 ≤ A ∧ A < 3.65) (hB : 3.595 ≤ B ∧ B < 3.605) :
  ¬(A > B ∨ A = B ∨ A < B) := by
  sorry

#check indeterminate_magnitude

end indeterminate_magnitude_l2539_253958


namespace carmen_cookie_sales_l2539_253922

/-- Represents the number of boxes of each type of cookie sold --/
structure CookieSales where
  samoas : ℕ
  thinMints : ℕ
  fudgeDelights : ℕ
  sugarCookies : ℕ

/-- Represents the price of each type of cookie --/
structure CookiePrices where
  samoas : ℚ
  thinMints : ℚ
  fudgeDelights : ℚ
  sugarCookies : ℚ

/-- Calculates the total revenue from cookie sales --/
def totalRevenue (sales : CookieSales) (prices : CookiePrices) : ℚ :=
  sales.samoas * prices.samoas +
  sales.thinMints * prices.thinMints +
  sales.fudgeDelights * prices.fudgeDelights +
  sales.sugarCookies * prices.sugarCookies

/-- The main theorem representing Carmen's cookie sales --/
theorem carmen_cookie_sales 
  (sales : CookieSales)
  (prices : CookiePrices)
  (h1 : sales.samoas = 3)
  (h2 : sales.thinMints = 2)
  (h3 : sales.fudgeDelights = 1)
  (h4 : prices.samoas = 4)
  (h5 : prices.thinMints = 7/2)
  (h6 : prices.fudgeDelights = 5)
  (h7 : prices.sugarCookies = 2)
  (h8 : totalRevenue sales prices = 42) :
  sales.sugarCookies = 9 := by
  sorry

end carmen_cookie_sales_l2539_253922


namespace arithmetic_sequence_special_condition_l2539_253929

/-- An arithmetic sequence with positive terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- Theorem stating that if 3a_6 - a_7^2 + 3a_8 = 0 in an arithmetic sequence with positive terms, then a_7 = 6 -/
theorem arithmetic_sequence_special_condition
  (seq : ArithmeticSequence)
  (h : 3 * seq.a 6 - (seq.a 7)^2 + 3 * seq.a 8 = 0) :
  seq.a 7 = 6 := by
  sorry

end arithmetic_sequence_special_condition_l2539_253929


namespace function_difference_l2539_253907

/-- Given two functions f and g, prove that if f(3) - g(3) = 1, then the parameter m in g equals 113/3 -/
theorem function_difference (f g : ℝ → ℝ) (m : ℝ) 
  (hf : f = fun x ↦ 4 * x^2 + 2 / x + 2)
  (hg : g = fun x ↦ x^2 - 3 * x + m)
  (h : f 3 - g 3 = 1) : 
  m = 113 / 3 := by
  sorry

end function_difference_l2539_253907


namespace greatest_c_value_l2539_253990

theorem greatest_c_value (c : ℝ) : 
  (∀ x : ℝ, -x^2 + 9*x - 20 ≥ 0 → x ≤ 5) ∧ 
  (-5^2 + 9*5 - 20 ≥ 0) :=
sorry

end greatest_c_value_l2539_253990


namespace smallest_gcd_for_integer_solution_l2539_253918

theorem smallest_gcd_for_integer_solution : ∃ (n : ℕ), n > 0 ∧
  (∀ (a b c : ℤ), Int.gcd a (Int.gcd b c) = n →
    ∃ (x y z : ℤ), x + 2*y + 3*z = a ∧ 2*x + y - 2*z = b ∧ 3*x + y + 5*z = c) ∧
  (∀ (m : ℕ), 0 < m → m < n →
    ∃ (a b c : ℤ), Int.gcd a (Int.gcd b c) = m ∧
      ¬∃ (x y z : ℤ), x + 2*y + 3*z = a ∧ 2*x + y - 2*z = b ∧ 3*x + y + 5*z = c) ∧
  n = 28 :=
sorry

end smallest_gcd_for_integer_solution_l2539_253918


namespace extreme_value_implies_a_eq_five_l2539_253974

/-- The function f(x) = x^3 + ax^2 + 3x - 9 has an extreme value at x = -3 -/
def has_extreme_value_at_neg_three (a : ℝ) : Prop :=
  let f := fun x : ℝ => x^3 + a*x^2 + 3*x - 9
  ∃ ε > 0, ∀ x ∈ Set.Ioo (-3-ε) (-3+ε), f x ≤ f (-3) ∨ f x ≥ f (-3)

/-- If f(x) = x^3 + ax^2 + 3x - 9 has an extreme value at x = -3, then a = 5 -/
theorem extreme_value_implies_a_eq_five :
  ∀ a : ℝ, has_extreme_value_at_neg_three a → a = 5 := by sorry

end extreme_value_implies_a_eq_five_l2539_253974


namespace min_value_of_linear_function_l2539_253977

/-- Given a system of linear inequalities, prove that the minimum value of a linear function is -6. -/
theorem min_value_of_linear_function (x y : ℝ) :
  x - 2*y + 2 ≥ 0 →
  2*x - y - 2 ≤ 0 →
  y ≥ 0 →
  ∀ z : ℝ, z = 3*x + y → z ≥ -6 :=
by sorry

end min_value_of_linear_function_l2539_253977


namespace perfect_square_conditions_l2539_253949

theorem perfect_square_conditions (a b c d e f : ℝ) :
  (∃ (p q r : ℝ), ∀ (x y z : ℝ),
    a * x^2 + b * y^2 + c * z^2 + 2 * d * x * y + 2 * e * y * z + 2 * f * z * x = (p * x + q * y + r * z)^2)
  ↔
  (a * b = d^2 ∧ b * c = e^2 ∧ c * a = f^2 ∧ a * e = d * f ∧ b * f = d * e ∧ c * d = e * f) :=
by sorry

end perfect_square_conditions_l2539_253949


namespace probability_of_selecting_specific_animals_l2539_253909

theorem probability_of_selecting_specific_animals :
  let total_animals : ℕ := 7
  let animals_to_select : ℕ := 2
  let specific_animals : ℕ := 2

  let total_combinations := Nat.choose total_animals animals_to_select
  let favorable_combinations := total_combinations - Nat.choose (total_animals - specific_animals) animals_to_select

  (favorable_combinations : ℚ) / total_combinations = 11 / 21 :=
by sorry

end probability_of_selecting_specific_animals_l2539_253909


namespace equation_equivalence_l2539_253955

theorem equation_equivalence (x y : ℝ) :
  (2*x - 3*y)^2 = 4*x^2 + 9*y^2 ↔ x*y = 0 :=
by sorry

end equation_equivalence_l2539_253955


namespace candy_bar_multiple_l2539_253931

theorem candy_bar_multiple (max_sales seth_sales : ℕ) (m : ℚ) 
  (h1 : max_sales = 24)
  (h2 : seth_sales = 78)
  (h3 : seth_sales = m * max_sales + 6) :
  m = 3 := by
  sorry

end candy_bar_multiple_l2539_253931


namespace smallest_square_sum_20_consecutive_l2539_253916

/-- The sum of 20 consecutive positive integers starting from n -/
def sum_20_consecutive (n : ℕ) : ℕ := 10 * (2 * n + 19)

/-- A number is a perfect square -/
def is_perfect_square (m : ℕ) : Prop := ∃ k : ℕ, m = k * k

theorem smallest_square_sum_20_consecutive :
  (∃ n : ℕ, sum_20_consecutive n = 250) ∧
  (∀ m : ℕ, m < 250 → ¬∃ n : ℕ, sum_20_consecutive n = m ∧ is_perfect_square m) :=
sorry

end smallest_square_sum_20_consecutive_l2539_253916


namespace existence_of_square_between_l2539_253983

theorem existence_of_square_between (a b c d : ℕ) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : a * d = b * c) : 
  ∃ m : ℤ, (↑a : ℝ) < m^2 ∧ (m^2 : ℝ) < ↑d :=
by sorry

end existence_of_square_between_l2539_253983


namespace min_value_theorem_l2539_253942

theorem min_value_theorem (x y : ℝ) (h1 : x * y + 3 * x = 3) (h2 : 0 < x) (h3 : x < 1/2) :
  ∀ z, z = (3 / x) + (1 / (y - 3)) → z ≥ 8 :=
by sorry

end min_value_theorem_l2539_253942


namespace slope_of_line_l2539_253963

theorem slope_of_line (x y : ℝ) :
  (4 * y = -5 * x + 8) → (y = (-5/4) * x + 2) :=
by sorry

end slope_of_line_l2539_253963


namespace smallest_product_l2539_253940

def digits : List Nat := [3, 4, 5, 6]

def valid_arrangement (a b c d : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def product (a b c d : Nat) : Nat :=
  (10 * a + b) * (10 * c + d)

theorem smallest_product :
  ∀ a b c d : Nat,
    valid_arrangement a b c d →
    product a b c d ≥ 1610 :=
by sorry

end smallest_product_l2539_253940


namespace first_boy_speed_l2539_253926

/-- The speed of the second boy in km/h -/
def second_boy_speed : ℝ := 7.5

/-- The time the boys walk in hours -/
def walking_time : ℝ := 16

/-- The distance between the boys after walking in km -/
def final_distance : ℝ := 32

/-- Theorem stating the speed of the first boy -/
theorem first_boy_speed (x : ℝ) : 
  (x - second_boy_speed) * walking_time = final_distance → x = 9.5 := by
  sorry

end first_boy_speed_l2539_253926


namespace total_tickets_sold_l2539_253932

/-- Represents the price of an adult ticket in dollars -/
def adult_price : ℕ := 15

/-- Represents the price of a child ticket in dollars -/
def child_price : ℕ := 8

/-- Represents the total receipts for the day in dollars -/
def total_receipts : ℕ := 5086

/-- Represents the number of adult tickets sold -/
def adult_tickets : ℕ := 130

/-- Theorem stating that the total number of tickets sold is 522 -/
theorem total_tickets_sold : 
  ∃ (child_tickets : ℕ), 
    adult_tickets * adult_price + child_tickets * child_price = total_receipts ∧
    adult_tickets + child_tickets = 522 :=
by sorry

end total_tickets_sold_l2539_253932


namespace fraction_equality_l2539_253945

theorem fraction_equality (m n : ℚ) (h : 2/3 * m = 5/6 * n) : (m - n) / n = 1/4 := by
  sorry

end fraction_equality_l2539_253945


namespace parabola_c_value_l2539_253969

/-- A parabola with vertex (h, k) passing through point (x₀, y₀) has c = 12.5 -/
theorem parabola_c_value (a b c h k x₀ y₀ : ℝ) : 
  (∀ x y, y = a * x^2 + b * x + c) →  -- parabola equation
  (h = 3 ∧ k = -1) →                  -- vertex at (3, -1)
  (x₀ = 1 ∧ y₀ = 5) →                 -- point (1, 5) on parabola
  (∀ x, a * (x - h)^2 + k = a * x^2 + b * x + c) →  -- vertex form equals general form
  (y₀ = a * x₀^2 + b * x₀ + c) →      -- point (1, 5) satisfies equation
  c = 12.5 := by
sorry

end parabola_c_value_l2539_253969


namespace gold_bars_per_row_l2539_253923

/-- Represents the arrangement of gold bars in a safe -/
structure GoldSafe where
  rows : Nat
  totalWorth : Nat
  barValue : Nat

/-- Calculates the number of gold bars per row in a safe -/
def barsPerRow (safe : GoldSafe) : Nat :=
  (safe.totalWorth / safe.barValue) / safe.rows

/-- Theorem: If a safe has 4 rows, total worth of $1,600,000, and each bar is worth $40,000,
    then there are 10 gold bars in each row -/
theorem gold_bars_per_row :
  let safe : GoldSafe := { rows := 4, totalWorth := 1600000, barValue := 40000 }
  barsPerRow safe = 10 := by
  sorry


end gold_bars_per_row_l2539_253923


namespace gwen_birthday_money_l2539_253973

/-- The amount of money Gwen received from her mom -/
def money_from_mom : ℕ := 8

/-- The amount of money Gwen received from her dad -/
def money_from_dad : ℕ := 5

/-- The amount of money Gwen spent -/
def money_spent : ℕ := 4

/-- The difference between the amount Gwen received from her mom and her dad -/
def difference : ℕ := money_from_mom - money_from_dad

theorem gwen_birthday_money : difference = 3 := by
  sorry

end gwen_birthday_money_l2539_253973


namespace max_absolute_value_quadratic_l2539_253936

theorem max_absolute_value_quadratic (a b : ℝ) :
  (∃ m : ℝ, ∀ t ∈ Set.Icc 0 4, ∃ t' ∈ Set.Icc 0 4, |t'^2 + a*t' + b| ≥ m) ∧
  (∀ m : ℝ, (∀ t ∈ Set.Icc 0 4, ∃ t' ∈ Set.Icc 0 4, |t'^2 + a*t' + b| ≥ m) → m ≤ 2) :=
by sorry

end max_absolute_value_quadratic_l2539_253936


namespace line_through_point_l2539_253982

/-- Given a line equation -3/4 - 3kx = 7y and a point (1/3, -8) on this line,
    prove that k = 55.25 is the unique value satisfying these conditions. -/
theorem line_through_point (k : ℝ) : 
  (-3/4 - 3*k*(1/3) = 7*(-8)) ↔ k = 55.25 := by sorry

end line_through_point_l2539_253982


namespace power_of_square_l2539_253902

theorem power_of_square (a : ℝ) : (a^2)^2 = a^4 := by
  sorry

end power_of_square_l2539_253902


namespace lionel_graham_crackers_left_l2539_253954

/-- Represents the ingredients for making Oreo cheesecakes -/
structure Ingredients where
  graham_crackers : ℕ
  oreos : ℕ
  cream_cheese : ℕ

/-- Represents the recipe requirements for one Oreo cheesecake -/
structure Recipe where
  graham_crackers : ℕ
  oreos : ℕ
  cream_cheese : ℕ

/-- Calculates the maximum number of cheesecakes that can be made given the ingredients and recipe -/
def max_cheesecakes (ingredients : Ingredients) (recipe : Recipe) : ℕ :=
  min (ingredients.graham_crackers / recipe.graham_crackers)
      (min (ingredients.oreos / recipe.oreos)
           (ingredients.cream_cheese / recipe.cream_cheese))

/-- Calculates the number of Graham cracker boxes left over after making the maximum number of cheesecakes -/
def graham_crackers_left (ingredients : Ingredients) (recipe : Recipe) : ℕ :=
  ingredients.graham_crackers - (max_cheesecakes ingredients recipe * recipe.graham_crackers)

/-- Theorem stating that Lionel will have 4 boxes of Graham crackers left over -/
theorem lionel_graham_crackers_left :
  let ingredients := Ingredients.mk 14 15 36
  let recipe := Recipe.mk 2 3 4
  graham_crackers_left ingredients recipe = 4 := by
  sorry

end lionel_graham_crackers_left_l2539_253954


namespace scientific_notation_of_1_59_million_l2539_253976

/-- Expresses 1.59 million in scientific notation -/
theorem scientific_notation_of_1_59_million :
  (1.59 : ℝ) * 1000000 = 1.59 * (10 : ℝ) ^ 6 := by sorry

end scientific_notation_of_1_59_million_l2539_253976


namespace parallelogram_base_length_l2539_253939

theorem parallelogram_base_length 
  (area : ℝ) (height : ℝ) (base : ℝ) 
  (h1 : area = 72) 
  (h2 : height = 6) 
  (h3 : area = base * height) : 
  base = 12 := by
  sorry

end parallelogram_base_length_l2539_253939


namespace family_reunion_attendance_l2539_253921

/-- The number of male adults at the family reunion -/
def male_adults : ℕ := 100

/-- The number of female adults at the family reunion -/
def female_adults : ℕ := male_adults + 50

/-- The total number of adults at the family reunion -/
def total_adults : ℕ := male_adults + female_adults

/-- The number of children at the family reunion -/
def children : ℕ := 2 * total_adults

/-- The total number of attendees at the family reunion -/
def total_attendees : ℕ := total_adults + children

theorem family_reunion_attendance : 
  female_adults = male_adults + 50 ∧ 
  children = 2 * total_adults ∧ 
  total_attendees = 750 → 
  male_adults = 100 := by
  sorry

end family_reunion_attendance_l2539_253921


namespace triangle_inequality_l2539_253957

theorem triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) := by
  sorry

end triangle_inequality_l2539_253957


namespace spike_cricket_count_l2539_253900

/-- The number of crickets Spike hunts in the morning -/
def morning_crickets : ℕ := 5

/-- The number of crickets Spike hunts in the afternoon and evening -/
def afternoon_evening_crickets : ℕ := 3 * morning_crickets

/-- The total number of crickets Spike hunts per day -/
def total_crickets : ℕ := morning_crickets + afternoon_evening_crickets

theorem spike_cricket_count : total_crickets = 20 := by
  sorry

end spike_cricket_count_l2539_253900


namespace d_share_is_300_l2539_253952

/-- Calculates the share of profit for an investor given the investments and total profit -/
def calculate_share (investment_c : ℚ) (investment_d : ℚ) (total_profit : ℚ) : ℚ :=
  (investment_d / (investment_c + investment_d)) * total_profit

/-- Theorem stating that D's share of the profit is 300 given the specified investments and total profit -/
theorem d_share_is_300 
  (investment_c : ℚ) 
  (investment_d : ℚ) 
  (total_profit : ℚ) 
  (h1 : investment_c = 1000)
  (h2 : investment_d = 1500)
  (h3 : total_profit = 500) :
  calculate_share investment_c investment_d total_profit = 300 := by
  sorry

#eval calculate_share 1000 1500 500

end d_share_is_300_l2539_253952


namespace sugar_price_increase_l2539_253912

theorem sugar_price_increase (initial_price : ℝ) (consumption_reduction : ℝ) (new_price : ℝ) : 
  initial_price = 6 →
  consumption_reduction = 19.999999999999996 →
  (1 - consumption_reduction / 100) * new_price = initial_price →
  new_price = 7.5 := by
sorry

end sugar_price_increase_l2539_253912


namespace pie_chart_most_appropriate_for_milk_powder_l2539_253924

/-- Represents different types of statistical charts -/
inductive ChartType
  | Line
  | Bar
  | Pie

/-- Represents a substance in milk powder -/
structure Substance where
  name : String
  percentage : Float

/-- Represents the composition of milk powder -/
def MilkPowderComposition := List Substance

/-- Determines if a chart type is appropriate for displaying percentage composition -/
def is_appropriate_for_percentage_composition (chart : ChartType) (composition : MilkPowderComposition) : Prop :=
  chart = ChartType.Pie

/-- Theorem stating that a pie chart is the most appropriate for displaying milk powder composition -/
theorem pie_chart_most_appropriate_for_milk_powder (composition : MilkPowderComposition) :
  is_appropriate_for_percentage_composition ChartType.Pie composition :=
by sorry

end pie_chart_most_appropriate_for_milk_powder_l2539_253924


namespace sprint_medal_awarding_ways_l2539_253933

/-- The number of ways to award medals in an international sprint final --/
def medalAwardingWays (totalSprinters : ℕ) (americanSprinters : ℕ) (medals : ℕ) : ℕ :=
  -- We'll define this function without implementation
  sorry

/-- Theorem stating the number of ways to award medals under given conditions --/
theorem sprint_medal_awarding_ways :
  medalAwardingWays 10 4 3 = 696 :=
by
  sorry

end sprint_medal_awarding_ways_l2539_253933


namespace benson_ticket_cost_l2539_253960

/-- Calculates the total cost of concert tickets for Mr. Benson -/
def concert_ticket_cost (base_price : ℝ) (general_count : ℕ) (vip_count : ℕ) (premium_count : ℕ) 
  (vip_markup : ℝ) (premium_markup : ℝ) (discount_threshold : ℕ) (discount_rate : ℝ) : ℝ :=
  let total_count := general_count + vip_count + premium_count
  let vip_price := base_price * (1 + vip_markup)
  let premium_price := base_price * (1 + premium_markup)
  let discounted_count := max (total_count - discount_threshold) 0
  let general_cost := base_price * general_count
  let vip_cost := if vip_count ≤ discounted_count
                  then vip_price * vip_count * (1 - discount_rate)
                  else vip_price * (vip_count - discounted_count) + 
                       vip_price * discounted_count * (1 - discount_rate)
  let premium_cost := if premium_count ≤ (discounted_count - vip_count)
                      then premium_price * premium_count * (1 - discount_rate)
                      else premium_price * (premium_count - (discounted_count - vip_count)) +
                           premium_price * (discounted_count - vip_count) * (1 - discount_rate)
  general_cost + vip_cost + premium_cost

/-- Theorem stating that the total cost for Mr. Benson's tickets is $650.80 -/
theorem benson_ticket_cost : 
  concert_ticket_cost 40 10 3 2 0.2 0.5 10 0.05 = 650.80 := by
  sorry


end benson_ticket_cost_l2539_253960


namespace combined_average_score_l2539_253975

/-- Given three classes with average scores and student ratios, prove the combined average score -/
theorem combined_average_score 
  (score_U score_B score_C : ℝ)
  (ratio_U ratio_B ratio_C : ℕ)
  (h1 : score_U = 65)
  (h2 : score_B = 80)
  (h3 : score_C = 77)
  (h4 : ratio_U = 4)
  (h5 : ratio_B = 6)
  (h6 : ratio_C = 5) :
  (score_U * ratio_U + score_B * ratio_B + score_C * ratio_C) / (ratio_U + ratio_B + ratio_C) = 75 := by
  sorry

end combined_average_score_l2539_253975


namespace new_total_cucumber_weight_l2539_253943

/-- Calculates the new weight of cucumbers after evaporation -/
def new_cucumber_weight (initial_weight : ℝ) (water_percentage : ℝ) (evaporation_rate : ℝ) : ℝ :=
  let water_weight := initial_weight * water_percentage
  let dry_weight := initial_weight * (1 - water_percentage)
  let evaporated_water := water_weight * evaporation_rate
  (water_weight - evaporated_water) + dry_weight

/-- Theorem stating the new total weight of cucumbers after evaporation -/
theorem new_total_cucumber_weight :
  let batch1 := new_cucumber_weight 50 0.99 0.01
  let batch2 := new_cucumber_weight 30 0.98 0.02
  let batch3 := new_cucumber_weight 20 0.97 0.03
  batch1 + batch2 + batch3 = 98.335 := by
  sorry

#eval new_cucumber_weight 50 0.99 0.01 +
      new_cucumber_weight 30 0.98 0.02 +
      new_cucumber_weight 20 0.97 0.03

end new_total_cucumber_weight_l2539_253943


namespace soccer_committee_combinations_l2539_253935

theorem soccer_committee_combinations : Nat.choose 6 4 = 15 := by
  sorry

end soccer_committee_combinations_l2539_253935


namespace pi_digits_difference_l2539_253988

theorem pi_digits_difference (mina_digits : ℕ) (mina_carlos_ratio : ℕ) (sam_digits : ℕ)
  (h1 : mina_digits = 24)
  (h2 : mina_digits = mina_carlos_ratio * (sam_digits - 6))
  (h3 : sam_digits = 10) :
  sam_digits - (mina_digits / mina_carlos_ratio) = 6 := by
sorry

end pi_digits_difference_l2539_253988


namespace cyclist_speed_problem_l2539_253959

/-- Theorem: Given two cyclists on a 45-mile course, starting from opposite ends at the same time,
    where one cyclist rides at 14 mph and they meet after 1.5 hours, the speed of the second cyclist is 16 mph. -/
theorem cyclist_speed_problem (course_length : ℝ) (first_speed : ℝ) (meeting_time : ℝ) :
  course_length = 45 ∧ first_speed = 14 ∧ meeting_time = 1.5 →
  ∃ second_speed : ℝ, second_speed = 16 ∧ course_length = (first_speed + second_speed) * meeting_time :=
by
  sorry


end cyclist_speed_problem_l2539_253959


namespace percentage_of_girls_l2539_253928

/-- The percentage of girls in a school, given the total number of students and the number of boys. -/
theorem percentage_of_girls (total : ℕ) (boys : ℕ) (h1 : total = 100) (h2 : boys = 50) :
  (total - boys : ℚ) / total * 100 = 50 := by
  sorry

end percentage_of_girls_l2539_253928


namespace remainder_problem_l2539_253947

theorem remainder_problem (n : ℕ) : 
  n % 44 = 0 ∧ n / 44 = 432 → n % 39 = 15 := by
  sorry

end remainder_problem_l2539_253947


namespace find_tap_a_turnoff_time_l2539_253972

/-- Represents the time it takes for a tap to fill the cistern -/
structure TapFillTime where
  minutes : ℝ
  positive : minutes > 0

/-- Represents the state of the cistern filling process -/
structure CisternFilling where
  tapA : TapFillTime
  tapB : TapFillTime
  remainingTime : ℝ
  positive : remainingTime > 0

/-- The main theorem statement -/
theorem find_tap_a_turnoff_time (c : CisternFilling) 
    (h1 : c.tapA.minutes = 12)
    (h2 : c.tapB.minutes = 18)
    (h3 : c.remainingTime = 8) : 
  ∃ t : ℝ, t > 0 ∧ t = 4 ∧
    (t * (1 / c.tapA.minutes + 1 / c.tapB.minutes) + 
     c.remainingTime * (1 / c.tapB.minutes) = 1) := by
  sorry

#check find_tap_a_turnoff_time

end find_tap_a_turnoff_time_l2539_253972


namespace forest_trees_count_l2539_253984

/-- The side length of the square-shaped street in meters -/
def street_side_length : ℝ := 100

/-- The area of the square-shaped street in square meters -/
def street_area : ℝ := street_side_length ^ 2

/-- The area of the forest in square meters -/
def forest_area : ℝ := 3 * street_area

/-- The number of trees per square meter in the forest -/
def trees_per_square_meter : ℝ := 4

/-- The total number of trees in the forest -/
def total_trees : ℝ := forest_area * trees_per_square_meter

theorem forest_trees_count : total_trees = 120000 := by
  sorry

end forest_trees_count_l2539_253984


namespace lassis_from_nine_mangoes_l2539_253993

/-- The number of lassis that can be made from a given number of mangoes -/
def lassis_from_mangoes (mangoes : ℕ) : ℕ :=
  5 * mangoes

/-- The cost of a given number of mangoes -/
def mango_cost (mangoes : ℕ) : ℕ :=
  2 * mangoes

theorem lassis_from_nine_mangoes :
  lassis_from_mangoes 9 = 45 :=
by sorry

end lassis_from_nine_mangoes_l2539_253993


namespace nowhere_negative_polynomial_is_sum_of_squares_l2539_253941

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- A polynomial is nowhere negative if it's non-negative for all real inputs -/
def NowhereNegative (p : RealPolynomial) : Prop :=
  ∀ x : ℝ, p x ≥ 0

/-- Theorem: Any nowhere negative real polynomial can be expressed as a sum of squares -/
theorem nowhere_negative_polynomial_is_sum_of_squares :
  ∀ p : RealPolynomial, NowhereNegative p →
  ∃ q r s : RealPolynomial, ∀ x : ℝ, p x = (q x)^2 * ((r x)^2 + (s x)^2) :=
sorry

end nowhere_negative_polynomial_is_sum_of_squares_l2539_253941


namespace parallel_transitivity_l2539_253965

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and between a line and a plane
variable (parallel_line : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the "in plane" relation for a line
variable (in_plane : Line → Plane → Prop)

-- State the theorem
theorem parallel_transitivity 
  (a b : Line) (α : Plane) :
  parallel_line a b →
  ¬ in_plane a α →
  ¬ in_plane b α →
  parallel_line_plane a α →
  parallel_line_plane b α :=
sorry

end parallel_transitivity_l2539_253965


namespace roots_sum_reciprocals_l2539_253937

theorem roots_sum_reciprocals (α β : ℝ) : 
  3 * α^2 + α - 1 = 0 →
  3 * β^2 + β - 1 = 0 →
  α > β →
  (α / β) + (β / α) = -7/3 := by
sorry

end roots_sum_reciprocals_l2539_253937


namespace polynomial_multiplication_l2539_253914

theorem polynomial_multiplication (x : ℝ) :
  (2 + 3 * x^3) * (1 - 2 * x^2 + x^4) = 2 - 4 * x^2 + 3 * x^3 + 2 * x^4 - 6 * x^5 + 3 * x^7 := by
  sorry

end polynomial_multiplication_l2539_253914


namespace calcium_oxide_molecular_weight_l2539_253996

/-- The atomic weight of calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of calcium atoms in a molecule of calcium oxide -/
def num_Ca_atoms : ℕ := 1

/-- The number of oxygen atoms in a molecule of calcium oxide -/
def num_O_atoms : ℕ := 1

/-- The molecular weight of calcium oxide in g/mol -/
def molecular_weight_CaO : ℝ := atomic_weight_Ca * num_Ca_atoms + atomic_weight_O * num_O_atoms

theorem calcium_oxide_molecular_weight :
  molecular_weight_CaO = 56.08 := by sorry

end calcium_oxide_molecular_weight_l2539_253996


namespace spelling_bee_initial_students_l2539_253964

theorem spelling_bee_initial_students :
  ∀ (initial_students : ℕ),
    (initial_students : ℝ) * 0.3 * 0.5 = 18 →
    initial_students = 120 :=
by
  sorry

end spelling_bee_initial_students_l2539_253964


namespace train_length_proof_l2539_253905

/-- Proves that a train with given speed passing a platform of known length in a certain time has a specific length -/
theorem train_length_proof (train_speed : ℝ) (platform_length : ℝ) (passing_time : ℝ) :
  train_speed = 45 * 1000 / 3600 →
  platform_length = 240 →
  passing_time = 48 →
  train_speed * passing_time - platform_length = 360 :=
by sorry

end train_length_proof_l2539_253905


namespace carol_cupcakes_theorem_l2539_253992

/-- Calculates the number of cupcakes made after selling the first batch -/
def cupcakes_made_after (initial : ℕ) (sold : ℕ) (final_total : ℕ) : ℕ :=
  final_total - (initial - sold)

/-- Proves that Carol made 28 cupcakes after selling the first batch -/
theorem carol_cupcakes_theorem (initial : ℕ) (sold : ℕ) (final_total : ℕ)
    (h1 : initial = 30)
    (h2 : sold = 9)
    (h3 : final_total = 49) :
    cupcakes_made_after initial sold final_total = 28 := by
  sorry

end carol_cupcakes_theorem_l2539_253992


namespace integer_solutions_of_equation_l2539_253920

theorem integer_solutions_of_equation :
  ∀ x y : ℤ, x^2 + x*y - y = 2 ↔ (x = 2 ∧ y = -2) ∨ (x = 0 ∧ y = -2) := by
  sorry

end integer_solutions_of_equation_l2539_253920


namespace cone_cylinder_volume_ratio_l2539_253999

/-- The ratio of the volume of a cone to the volume of a cylinder with the same base radius,
    where the cone's height is one-third of the cylinder's height, is 1/9. -/
theorem cone_cylinder_volume_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  (1 / 3 * π * r^2 * (h / 3)) / (π * r^2 * h) = 1 / 9 := by
  sorry


end cone_cylinder_volume_ratio_l2539_253999


namespace abs_plus_one_nonzero_l2539_253915

theorem abs_plus_one_nonzero (a : ℚ) : |a| + 1 ≠ 0 := by
  sorry

end abs_plus_one_nonzero_l2539_253915


namespace three_equal_differences_l2539_253956

theorem three_equal_differences (n : ℕ) (a : Fin (2 * n) → ℕ) 
  (h1 : n > 2)
  (h2 : ∀ i j, i ≠ j → a i ≠ a j)
  (h3 : ∀ i, a i ≤ n^2)
  (h4 : ∀ i, a i > 0) :
  ∃ (i1 j1 i2 j2 i3 j3 : Fin (2 * n)), 
    (i1 > j1 ∧ i2 > j2 ∧ i3 > j3) ∧ 
    (i1 ≠ i2 ∨ j1 ≠ j2) ∧ 
    (i1 ≠ i3 ∨ j1 ≠ j3) ∧ 
    (i2 ≠ i3 ∨ j2 ≠ j3) ∧
    (a i1 - a j1 = a i2 - a j2) ∧ 
    (a i1 - a j1 = a i3 - a j3) :=
by sorry

end three_equal_differences_l2539_253956


namespace alyssa_cookies_l2539_253938

-- Define the number of cookies Aiyanna has
def aiyanna_cookies : ℕ := 140

-- Define the difference between Alyssa's and Aiyanna's cookies
def cookie_difference : ℕ := 11

-- Theorem stating Alyssa's number of cookies
theorem alyssa_cookies : 
  ∃ (a : ℕ), a = aiyanna_cookies + cookie_difference :=
by
  sorry

end alyssa_cookies_l2539_253938


namespace light_travel_distance_l2539_253906

/-- The distance light travels in one year (in miles) -/
def light_year_distance : ℝ := 6000000000000

/-- The number of years we're calculating for -/
def years : ℕ := 50

/-- The distance light travels in the given number of years -/
def total_distance : ℝ := light_year_distance * years

theorem light_travel_distance : total_distance = 3 * (10 ^ 14) := by
  sorry

end light_travel_distance_l2539_253906


namespace triangle_property_l2539_253925

theorem triangle_property (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →
  (A + B + C = π) →
  -- Given condition
  (2 * Real.cos B * Real.cos C + 1 = 2 * Real.sin B * Real.sin C) →
  (b + c = 4) →
  -- Conclusions
  (A = π / 3) ∧
  (∀ (area : Real), area = 1/2 * b * c * Real.sin A → area ≤ Real.sqrt 3) ∧
  (∃ (area : Real), area = 1/2 * b * c * Real.sin A ∧ area = Real.sqrt 3) :=
by sorry

end triangle_property_l2539_253925


namespace y_intercept_of_line_l2539_253968

-- Define the line equation
def line_equation (x y a b : ℝ) : Prop := x / a - y / b = 1

-- Define y-intercept
def y_intercept (f : ℝ → ℝ) : ℝ := f 0

-- Theorem statement
theorem y_intercept_of_line (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  ∃ f : ℝ → ℝ, (∀ x, line_equation x (f x) a b) ∧ y_intercept f = -b :=
sorry

end y_intercept_of_line_l2539_253968


namespace right_triangle_medians_semiperimeter_l2539_253913

theorem right_triangle_medians_semiperimeter (a b : ℝ) (h1 : a = 6) (h2 : b = 4) :
  let c := Real.sqrt (a^2 + b^2)
  let s := (a + b + c) / 2
  let m1 := c / 2
  let m2 := Real.sqrt ((2 * a^2 + 2 * b^2 - c^2) / 4)
  m1 + m2 = s := by sorry

end right_triangle_medians_semiperimeter_l2539_253913


namespace rectangular_field_fence_l2539_253953

theorem rectangular_field_fence (area : ℝ) (fence_length : ℝ) (uncovered_side : ℝ) :
  area = 600 →
  fence_length = 130 →
  uncovered_side * (fence_length - uncovered_side) / 2 = area →
  uncovered_side = 120 :=
by
  sorry

end rectangular_field_fence_l2539_253953


namespace unique_4digit_number_l2539_253986

def is_3digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_4digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem unique_4digit_number :
  ∃! n : ℕ, 
    is_4digit n ∧ 
    (∃ a : ℕ, is_3digit (400 + 10*a + 3) ∧ n = (400 + 10*a + 3) + 984) ∧
    n % 11 = 0 ∧
    (∃ h : ℕ, 10 ≤ h ∧ h ≤ 19 ∧ a + (h - 10) = 10 ∧ n = 1000*h + (n % 1000)) ∧
    n = 1397 :=
sorry

end unique_4digit_number_l2539_253986


namespace ducks_in_lake_l2539_253961

theorem ducks_in_lake (initial_ducks joining_ducks : ℕ) 
  (h1 : initial_ducks = 13)
  (h2 : joining_ducks = 20) : 
  initial_ducks + joining_ducks = 33 := by
sorry

end ducks_in_lake_l2539_253961


namespace francine_work_weeks_francine_work_weeks_solution_l2539_253966

theorem francine_work_weeks 
  (daily_distance : ℕ) 
  (workdays_per_week : ℕ) 
  (total_distance : ℕ) : ℕ :=
  let weekly_distance := daily_distance * workdays_per_week
  total_distance / weekly_distance

#check francine_work_weeks 140 4 2240

theorem francine_work_weeks_solution :
  francine_work_weeks 140 4 2240 = 4 := by
  sorry

end francine_work_weeks_francine_work_weeks_solution_l2539_253966


namespace sum_smallest_largest_primes_10_to_50_l2539_253908

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def primes_between (a b : ℕ) : Set ℕ :=
  {n : ℕ | a < n ∧ n < b ∧ is_prime n}

theorem sum_smallest_largest_primes_10_to_50 :
  let P := primes_between 10 50
  ∃ (p q : ℕ), p ∈ P ∧ q ∈ P ∧
    (∀ x ∈ P, p ≤ x) ∧
    (∀ x ∈ P, x ≤ q) ∧
    p + q = 58 :=
sorry

end sum_smallest_largest_primes_10_to_50_l2539_253908


namespace volume_third_number_l2539_253994

/-- Given a volume that is the product of three numbers, where two numbers are 12 and 18,
    and 48 cubes of edge 3 can be inserted into it, the third number in the product is 6. -/
theorem volume_third_number (volume : ℕ) (x : ℕ) : 
  volume = 12 * 18 * x →
  volume = 48 * 3^3 →
  x = 6 := by
  sorry

end volume_third_number_l2539_253994


namespace scientific_notation_of_175_billion_l2539_253995

theorem scientific_notation_of_175_billion : ∃ (a : ℝ) (n : ℤ), 
  175000000000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.75 ∧ n = 11 :=
sorry

end scientific_notation_of_175_billion_l2539_253995


namespace randy_blocks_left_l2539_253997

/-- The number of blocks Randy has left after constructions -/
def blocks_left (initial : ℕ) (tower : ℕ) (house : ℕ) : ℕ :=
  let remaining_after_tower := initial - tower
  let bridge := remaining_after_tower / 2
  let remaining_after_bridge := remaining_after_tower - bridge
  remaining_after_bridge - house

/-- Theorem stating that Randy has 19 blocks left after constructions -/
theorem randy_blocks_left :
  blocks_left 78 19 11 = 19 := by sorry

end randy_blocks_left_l2539_253997


namespace garden_width_l2539_253910

/-- Proves that the width of a rectangular garden with given conditions is 120 feet -/
theorem garden_width :
  ∀ (width : ℝ),
  (width > 0) →
  (220 * width > 0) →
  (220 * width / 2 > 0) →
  (220 * width / 2 * 2 / 3 > 0) →
  (220 * width / 2 * 2 / 3 = 8800) →
  (width = 120) := by
sorry

end garden_width_l2539_253910


namespace all_a_equal_one_l2539_253980

def cyclic_index (i : ℕ) : ℕ :=
  match i % 100 with
  | 0 => 100
  | n => n

theorem all_a_equal_one (a : ℕ → ℝ) 
  (h_ineq : ∀ i, a (cyclic_index i) - 4 * a (cyclic_index (i + 1)) + 3 * a (cyclic_index (i + 2)) ≥ 0)
  (h_a1 : a 1 = 1) :
  ∀ i, a i = 1 := by
sorry

end all_a_equal_one_l2539_253980


namespace min_distinct_prime_factors_l2539_253998

theorem min_distinct_prime_factors (m n : ℕ) :
  ∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧
  p ∣ (m * (n + 9) * (m + 2 * n^2 + 3)) ∧
  q ∣ (m * (n + 9) * (m + 2 * n^2 + 3)) :=
sorry

end min_distinct_prime_factors_l2539_253998


namespace min_distance_to_origin_l2539_253979

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y + 1 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y + 9 = 0

-- Define the condition |PA| = |PB|
def equal_chords (x y : ℝ) : Prop := 
  ∃ (xa ya xb yb : ℝ), C₁ xa ya ∧ C₂ xb yb ∧ 
  (x - xa)^2 + (y - ya)^2 = (x - xb)^2 + (y - yb)^2

-- Theorem statement
theorem min_distance_to_origin : 
  ∀ (x y : ℝ), equal_chords x y → 
  ∃ (x' y' : ℝ), equal_chords x' y' ∧ 
  ∀ (x'' y'' : ℝ), equal_chords x'' y'' → 
  (x'^2 + y'^2 : ℝ) ≤ x''^2 + y''^2 ∧
  (x'^2 + y'^2 : ℝ) = (4/5)^2 := by
  sorry

end min_distance_to_origin_l2539_253979


namespace fountain_area_l2539_253948

theorem fountain_area (AB DC : ℝ) (h1 : AB = 20) (h2 : DC = 12) : ∃ (area : ℝ), area = 244 * Real.pi := by
  sorry

end fountain_area_l2539_253948


namespace sequence_max_value_l2539_253991

def a (n : ℕ+) : ℚ := n / (n^2 + 156)

theorem sequence_max_value :
  (∃ (k : ℕ+), a k = 1/25 ∧ 
   ∀ (n : ℕ+), a n ≤ 1/25) ∧
  (∀ (n : ℕ+), a n = 1/25 → (n = 12 ∨ n = 13)) :=
sorry

end sequence_max_value_l2539_253991


namespace factorial_ratio_equals_504_l2539_253970

theorem factorial_ratio_equals_504 : ∃! n : ℕ, n > 0 ∧ n.factorial / (n - 3).factorial = 504 := by
  sorry

end factorial_ratio_equals_504_l2539_253970


namespace star_op_specific_value_l2539_253981

-- Define the * operation for non-zero integers
def star_op (a b : ℤ) : ℚ := 1 / a + 1 / b

-- Theorem statement
theorem star_op_specific_value :
  ∀ a b : ℕ+, 
  (a : ℤ) + (b : ℤ) = 15 → 
  (a : ℤ) * (b : ℤ) = 36 → 
  star_op a b = 5 / 12 := by
  sorry

end star_op_specific_value_l2539_253981


namespace iphone_case_cost_percentage_l2539_253989

/-- Proves that the percentage of the case cost relative to the phone cost is 20% --/
theorem iphone_case_cost_percentage :
  let phone_cost : ℝ := 1000
  let monthly_contract_cost : ℝ := 200
  let case_cost_percentage : ℝ → ℝ := λ x => x / 100 * phone_cost
  let headphones_cost : ℝ → ℝ := λ x => (1 / 2) * case_cost_percentage x
  let total_yearly_cost : ℝ → ℝ := λ x => 
    phone_cost + 12 * monthly_contract_cost + case_cost_percentage x + headphones_cost x
  ∃ x : ℝ, total_yearly_cost x = 3700 ∧ x = 20 :=
by
  sorry


end iphone_case_cost_percentage_l2539_253989


namespace pyramid_volume_theorem_l2539_253930

/-- A regular hexagon -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ

/-- A right pyramid with a regular hexagon base -/
structure RightPyramid where
  base : RegularHexagon
  apex : ℝ × ℝ × ℝ

/-- An equilateral triangle -/
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Calculate the volume of a right pyramid -/
def pyramidVolume (p : RightPyramid) : ℝ := sorry

/-- Check if a triangle is equilateral with given side length -/
def isEquilateralWithSideLength (t : EquilateralTriangle) (s : ℝ) : Prop := sorry

theorem pyramid_volume_theorem (p : RightPyramid) (t : EquilateralTriangle) :
  isEquilateralWithSideLength t 10 →
  pyramidVolume p = 187.5 := by
  sorry

end pyramid_volume_theorem_l2539_253930


namespace cube_edge_length_l2539_253951

/-- Represents a cube with a given total edge length. -/
structure Cube where
  total_edge_length : ℝ
  total_edge_length_positive : 0 < total_edge_length

/-- The number of edges in a cube. -/
def num_edges : ℕ := 12

/-- Theorem: In a cube where the sum of all edge lengths is 108 cm, each edge is 9 cm long. -/
theorem cube_edge_length (c : Cube) (h : c.total_edge_length = 108) :
  c.total_edge_length / num_edges = 9 := by
  sorry

end cube_edge_length_l2539_253951


namespace asha_win_probability_l2539_253927

theorem asha_win_probability (p_lose p_tie : ℚ) : 
  p_lose = 3/8 → p_tie = 1/4 → 1 - p_lose - p_tie = 3/8 := by
  sorry

end asha_win_probability_l2539_253927


namespace necklace_price_calculation_l2539_253934

def polo_shirt_price : ℕ := 26
def polo_shirt_quantity : ℕ := 3
def necklace_quantity : ℕ := 2
def computer_game_price : ℕ := 90
def computer_game_quantity : ℕ := 1
def rebate : ℕ := 12
def total_cost_after_rebate : ℕ := 322

theorem necklace_price_calculation (necklace_price : ℕ) : 
  polo_shirt_price * polo_shirt_quantity + 
  necklace_price * necklace_quantity + 
  computer_game_price * computer_game_quantity - 
  rebate = total_cost_after_rebate → 
  necklace_price = 83 := by sorry

end necklace_price_calculation_l2539_253934


namespace three_stamps_cost_l2539_253944

/-- The cost of a single stamp in dollars -/
def stamp_cost : ℚ := 34 / 100

/-- The cost of two stamps in dollars -/
def two_stamps_cost : ℚ := 68 / 100

/-- Theorem: The cost of three stamps is $1.02 -/
theorem three_stamps_cost : stamp_cost * 3 = 102 / 100 := by
  sorry

end three_stamps_cost_l2539_253944


namespace yellow_cards_per_player_l2539_253967

theorem yellow_cards_per_player (total_players : ℕ) (uncautioned_players : ℕ) (red_cards : ℕ) 
  (h1 : total_players = 11)
  (h2 : uncautioned_players = 5)
  (h3 : red_cards = 3) :
  (total_players - uncautioned_players) * ((red_cards * 2) / (total_players - uncautioned_players)) = 1 := by
  sorry

end yellow_cards_per_player_l2539_253967


namespace concentric_circles_radii_difference_l2539_253950

theorem concentric_circles_radii_difference
  (r R : ℝ) -- radii of the smaller and larger circles
  (h : r > 0) -- radius is positive
  (area_ratio : π * R^2 = 4 * (π * r^2)) -- area ratio is 1:4
  : R - r = r := by
  sorry

end concentric_circles_radii_difference_l2539_253950
