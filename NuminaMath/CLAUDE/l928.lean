import Mathlib

namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l928_92834

theorem ceiling_neg_sqrt_64_over_9 : 
  ⌈-Real.sqrt (64/9)⌉ = -2 := by sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l928_92834


namespace NUMINAMATH_CALUDE_largest_integer_problem_l928_92892

theorem largest_integer_problem (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d →  -- four different integers
  (a + b + c + d) / 4 = 74 →  -- average is 74
  a ≥ 29 →  -- smallest integer is at least 29
  d ≤ 206 :=  -- largest integer is at most 206
by sorry

end NUMINAMATH_CALUDE_largest_integer_problem_l928_92892


namespace NUMINAMATH_CALUDE_sum_of_digits_of_M_l928_92811

-- Define M as a positive integer
def M : ℕ+ := sorry

-- Define the condition M^2 = 36^49 * 49^36
axiom M_squared : (M : ℕ)^2 = 36^49 * 49^36

-- Define a function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_digits_of_M : sum_of_digits (M : ℕ) = 21 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_M_l928_92811


namespace NUMINAMATH_CALUDE_inequality_for_increasing_function_l928_92871

theorem inequality_for_increasing_function (f : ℝ → ℝ) (a b : ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_sum : a + b ≤ 0) : 
  f a + f b ≤ f (-a) + f (-b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_for_increasing_function_l928_92871


namespace NUMINAMATH_CALUDE_norma_laundry_ratio_l928_92896

/-- Proves the ratio of sweaters to T-shirts Norma left in the washer -/
theorem norma_laundry_ratio : 
  ∀ (S : ℕ), -- S is the number of sweaters Norma left
  -- Given conditions:
  (9 : ℕ) + S = 3 + 3 * 9 + 15 → -- Total items left = Total items found + Missing items
  (S : ℚ) / 9 = 2 / 1 := by
    sorry

end NUMINAMATH_CALUDE_norma_laundry_ratio_l928_92896


namespace NUMINAMATH_CALUDE_date_book_cost_date_book_cost_value_l928_92851

/-- Given the conditions of a real estate salesperson's promotional item purchase,
    prove that the cost of each date book is $0.375. -/
theorem date_book_cost (total_items : ℕ) (calendars : ℕ) (date_books : ℕ) 
                       (calendar_cost : ℚ) (total_spent : ℚ) : ℚ :=
  let date_book_cost := (total_spent - (calendar_cost * calendars)) / date_books
  by
    have h1 : total_items = calendars + date_books := by sorry
    have h2 : total_items = 500 := by sorry
    have h3 : calendars = 300 := by sorry
    have h4 : date_books = 200 := by sorry
    have h5 : calendar_cost = 3/4 := by sorry
    have h6 : total_spent = 300 := by sorry
    
    -- Prove that date_book_cost = 3/8
    sorry

#eval (300 : ℚ) - (3/4 * 300)
#eval ((300 : ℚ) - (3/4 * 300)) / 200

/-- The cost of each date book is $0.375. -/
theorem date_book_cost_value : 
  date_book_cost 500 300 200 (3/4) 300 = 3/8 := by sorry

end NUMINAMATH_CALUDE_date_book_cost_date_book_cost_value_l928_92851


namespace NUMINAMATH_CALUDE_surface_area_ratio_l928_92810

/-- The ratio of the surface area of a cube to the surface area of a rectangular solid
    with dimensions 2L, 3W, and 4H, where L, W, H are the cube's dimensions. -/
theorem surface_area_ratio (s : ℝ) (h : s > 0) : 
  (6 * s^2) / (2 * (2*s) * (3*s) + 2 * (2*s) * (4*s) + 2 * (3*s) * (4*s)) = 3 / 26 := by
  sorry

#check surface_area_ratio

end NUMINAMATH_CALUDE_surface_area_ratio_l928_92810


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l928_92816

-- Define the triangle operation
def triangle (a b : ℚ) : ℚ :=
  if a ≥ b then b^2 else 2*a - b

-- Theorem statements
theorem problem_1 : triangle (-4) (-5) = 25 := by sorry

theorem problem_2 : triangle (triangle (-3) 2) (-9) = 81 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l928_92816


namespace NUMINAMATH_CALUDE_ball_probabilities_l928_92865

/-- Represents the color of a ball -/
inductive BallColor
| Black
| White

/-- Represents the bag of balls -/
structure Bag :=
  (black : ℕ)
  (white : ℕ)

/-- The probability of drawing a white ball on the third draw with replacement -/
def prob_white_third_with_replacement (bag : Bag) : ℚ :=
  bag.white / (bag.black + bag.white)

/-- The probability of drawing a white ball only on the third draw with replacement -/
def prob_white_only_third_with_replacement (bag : Bag) : ℚ :=
  (bag.black / (bag.black + bag.white))^2 * (bag.white / (bag.black + bag.white))

/-- The probability of drawing a white ball on the third draw without replacement -/
def prob_white_third_without_replacement (bag : Bag) : ℚ :=
  (bag.white * (bag.black * (bag.black - 1) + 2 * bag.black * bag.white + bag.white * (bag.white - 1))) /
  ((bag.black + bag.white) * (bag.black + bag.white - 1) * (bag.black + bag.white - 2))

/-- The probability of drawing a white ball only on the third draw without replacement -/
def prob_white_only_third_without_replacement (bag : Bag) : ℚ :=
  (bag.black * (bag.black - 1) * bag.white) /
  ((bag.black + bag.white) * (bag.black + bag.white - 1) * (bag.black + bag.white - 2))

theorem ball_probabilities (bag : Bag) (h : bag.black = 3 ∧ bag.white = 2) :
  prob_white_third_with_replacement bag = 2/5 ∧
  prob_white_only_third_with_replacement bag = 18/125 ∧
  prob_white_third_without_replacement bag = 2/5 ∧
  prob_white_only_third_without_replacement bag = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ball_probabilities_l928_92865


namespace NUMINAMATH_CALUDE_munchausen_palindrome_exists_l928_92869

/-- A type representing a multi-digit number --/
def MultiDigitNumber := List Nat

/-- Check if a number is a palindrome --/
def isPalindrome (n : MultiDigitNumber) : Prop :=
  n = n.reverse

/-- Check if a list of numbers contains all numbers from 1 to N exactly once --/
def containsOneToN (l : List Nat) (N : Nat) : Prop :=
  l.toFinset = Finset.range N

/-- A function that represents cutting a number between digits --/
def cutBetweenDigits (n : MultiDigitNumber) : List Nat := sorry

/-- The main theorem --/
theorem munchausen_palindrome_exists :
  ∃ (n : MultiDigitNumber),
    isPalindrome n ∧
    containsOneToN (cutBetweenDigits n) 19 := by
  sorry

end NUMINAMATH_CALUDE_munchausen_palindrome_exists_l928_92869


namespace NUMINAMATH_CALUDE_base_6_arithmetic_l928_92838

/-- Convert a base 6 number to base 10 --/
def to_base_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (λ acc (i, d) => acc + d * (6 ^ i)) 0

/-- Convert a base 10 number to base 6 --/
def to_base_6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 6) ((m % 6) :: acc)
  aux n []

/-- Theorem: 1254₆ - 432₆ + 221₆ = 1043₆ in base 6 --/
theorem base_6_arithmetic :
  to_base_6 (to_base_10 [4, 5, 2, 1] - to_base_10 [2, 3, 4] + to_base_10 [1, 2, 2]) = [3, 4, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_base_6_arithmetic_l928_92838


namespace NUMINAMATH_CALUDE_ten_spheres_melted_l928_92808

/-- The radius of a sphere formed by melting multiple smaller spheres -/
def large_sphere_radius (n : ℕ) (r : ℝ) : ℝ :=
  (n * r ^ 3) ^ (1/3)

/-- Theorem: The radius of a sphere formed by melting 10 smaller spheres 
    with radius 3 inches is equal to the cube root of 270 inches -/
theorem ten_spheres_melted (n : ℕ) (r : ℝ) : 
  n = 10 ∧ r = 3 → large_sphere_radius n r = 270 ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_ten_spheres_melted_l928_92808


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l928_92849

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m : ℝ) : Prop :=
  2 * (m + 1) * (m - 3) + 2 * (m - 3) = 0

/-- m = 3 is a sufficient condition for the lines to be perpendicular -/
theorem sufficient_condition : perpendicular 3 := by sorry

/-- m = 3 is not a necessary condition for the lines to be perpendicular -/
theorem not_necessary_condition : ∃ m ≠ 3, perpendicular m := by sorry

/-- m = 3 is a sufficient but not necessary condition for the lines to be perpendicular -/
theorem sufficient_but_not_necessary :
  (perpendicular 3) ∧ (∃ m ≠ 3, perpendicular m) := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l928_92849


namespace NUMINAMATH_CALUDE_divisors_of_2013_power_13_l928_92853

theorem divisors_of_2013_power_13 : 
  let n : ℕ := 2013^13
  ∀ (p : ℕ → Prop), 
    (∀ k, p k ↔ k ∣ n ∧ k > 0) →
    (2013 = 3 * 11 * 61) →
    (∃! (s : Finset ℕ), ∀ k, k ∈ s ↔ p k) →
    Finset.card s = 2744 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_2013_power_13_l928_92853


namespace NUMINAMATH_CALUDE_darnells_average_yards_is_11_l928_92860

/-- Calculates Darnell's average yards rushed per game given the total yards and other players' yards. -/
def darnells_average_yards (total_yards : ℕ) (malik_yards_per_game : ℕ) (josiah_yards_per_game : ℕ) (num_games : ℕ) : ℕ := 
  (total_yards - (malik_yards_per_game * num_games + josiah_yards_per_game * num_games)) / num_games

/-- Proves that Darnell's average yards rushed per game is 11 yards given the problem conditions. -/
theorem darnells_average_yards_is_11 : 
  darnells_average_yards 204 18 22 4 = 11 := by
  sorry

#eval darnells_average_yards 204 18 22 4

end NUMINAMATH_CALUDE_darnells_average_yards_is_11_l928_92860


namespace NUMINAMATH_CALUDE_johnny_runs_four_times_l928_92855

theorem johnny_runs_four_times (block_length : ℝ) (average_distance : ℝ) :
  block_length = 200 →
  average_distance = 600 →
  ∃ (johnny_runs : ℕ),
    (average_distance = (block_length * johnny_runs + block_length * (johnny_runs / 2)) / 2) ∧
    johnny_runs = 4 :=
by sorry

end NUMINAMATH_CALUDE_johnny_runs_four_times_l928_92855


namespace NUMINAMATH_CALUDE_problem_solution_l928_92884

theorem problem_solution (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : 6 * x^3 + 12 * x * y = 2 * x^4 + 3 * x^3 * y) (h4 : y = x^2) : 
  x = (-1 + Real.sqrt 55) / 3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l928_92884


namespace NUMINAMATH_CALUDE_fraction_equality_l928_92822

theorem fraction_equality : 
  let f (x : ℕ) := x^4 + 324
  (∀ x, f x = (x^2 - 6*x + 18) * (x^2 + 6*x + 18)) →
  (f 64 * f 52 * f 40 * f 28 * f 16) / (f 58 * f 46 * f 34 * f 22 * f 10) = 137 / 1513 :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_l928_92822


namespace NUMINAMATH_CALUDE_absolute_value_expression_l928_92836

theorem absolute_value_expression : 
  |(-2)| * (|(-Real.sqrt 25)| - |Real.sin (5 * Real.pi / 2)|) = 8 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_expression_l928_92836


namespace NUMINAMATH_CALUDE_exam_score_calculation_l928_92802

theorem exam_score_calculation (total_questions : ℕ) (correct_score : ℕ) (total_score : ℕ) (correct_answers : ℕ) :
  total_questions = 75 →
  correct_score = 4 →
  total_score = 125 →
  correct_answers = 40 →
  (total_questions - correct_answers) * (correct_score - (correct_score * correct_answers - total_score) / (total_questions - correct_answers)) = total_score :=
by sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l928_92802


namespace NUMINAMATH_CALUDE_harmonic_mean_pairs_l928_92829

theorem harmonic_mean_pairs : 
  let count := Finset.filter (fun p : ℕ × ℕ => 
    p.1 < p.2 ∧ 
    (2 * p.1 * p.2 : ℚ) / (p.1 + p.2) = 4^30
  ) (Finset.range (2^61 + 1) ×ˢ Finset.range (2^61 + 1))
  
  count.card = 61 := by
  sorry

end NUMINAMATH_CALUDE_harmonic_mean_pairs_l928_92829


namespace NUMINAMATH_CALUDE_five_letter_words_with_consonant_l928_92809

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def consonants : Finset Char := {'B', 'C', 'D', 'F'}
def vowels : Finset Char := {'A', 'E'}

def word_length : Nat := 5

theorem five_letter_words_with_consonant :
  (alphabet.card ^ word_length) - (vowels.card ^ word_length) = 7744 :=
by sorry

end NUMINAMATH_CALUDE_five_letter_words_with_consonant_l928_92809


namespace NUMINAMATH_CALUDE_gel_pen_price_ratio_l928_92891

/-- Represents the price ratio of gel pens to ballpoint pens -/
def price_ratio (x y : ℕ) (b g : ℝ) : Prop :=
  let total := x * b + y * g
  (x + y) * g = 4 * total ∧ (x + y) * b = (1 / 2) * total ∧ g = 8 * b

/-- Theorem stating that under the given conditions, a gel pen costs 8 times as much as a ballpoint pen -/
theorem gel_pen_price_ratio {x y : ℕ} {b g : ℝ} (h : price_ratio x y b g) :
  g = 8 * b := by
  sorry

end NUMINAMATH_CALUDE_gel_pen_price_ratio_l928_92891


namespace NUMINAMATH_CALUDE_constant_remainder_iff_a_eq_neg_35_l928_92850

/-- The dividend polynomial -/
def dividend (a : ℚ) (x : ℚ) : ℚ := 10 * x^3 - 7 * x^2 + a * x + 10

/-- The divisor polynomial -/
def divisor (x : ℚ) : ℚ := 2 * x^2 - 5 * x + 2

/-- The remainder when dividend is divided by divisor -/
def remainder (a : ℚ) (x : ℚ) : ℚ := dividend a x - divisor x * (5 * x + 15/2)

theorem constant_remainder_iff_a_eq_neg_35 :
  (∃ (c : ℚ), ∀ (x : ℚ), remainder a x = c) ↔ a = -35 := by sorry

end NUMINAMATH_CALUDE_constant_remainder_iff_a_eq_neg_35_l928_92850


namespace NUMINAMATH_CALUDE_circus_tent_sections_l928_92821

theorem circus_tent_sections (section_capacity : ℕ) (total_capacity : ℕ) : 
  section_capacity = 246 → total_capacity = 984 → total_capacity / section_capacity = 4 := by
  sorry

end NUMINAMATH_CALUDE_circus_tent_sections_l928_92821


namespace NUMINAMATH_CALUDE_triangle_proof_l928_92872

theorem triangle_proof (A B C : Real) (a b c : Real) (D : Real) :
  a = Real.sqrt 19 →
  (Real.sin B + Real.sin C) / (Real.cos B + Real.cos A) = (Real.cos B - Real.cos A) / Real.sin C →
  ∃ (D : Real), D ∈ Set.Icc 0 1 ∧ 
    (3 * (1 - D)) / (4 * D) = 1 ∧
    ((1 - D) * b + D * c) * (c * Real.cos A) = 0 →
  A = 2 * Real.pi / 3 ∧
  1/2 * b * c * Real.sin A = 3 * Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_proof_l928_92872


namespace NUMINAMATH_CALUDE_tan_alpha_2_implies_fraction_eq_two_thirds_l928_92877

theorem tan_alpha_2_implies_fraction_eq_two_thirds (α : Real) 
  (h : Real.tan α = 2) : 
  (4 * Real.sin α - 2 * Real.cos α) / (3 * Real.cos α + 3 * Real.sin α) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_2_implies_fraction_eq_two_thirds_l928_92877


namespace NUMINAMATH_CALUDE_cos_75_degrees_l928_92881

theorem cos_75_degrees :
  Real.cos (75 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_75_degrees_l928_92881


namespace NUMINAMATH_CALUDE_bond_paper_cost_l928_92825

/-- Represents the cost of bond paper for an office. -/
structure BondPaperCost where
  sheets_per_ream : ℕ
  sheets_needed : ℕ
  total_cost : ℚ

/-- Calculates the cost of one ream of bond paper. -/
def cost_per_ream (bpc : BondPaperCost) : ℚ :=
  bpc.total_cost / (bpc.sheets_needed / bpc.sheets_per_ream)

/-- Theorem stating that the cost of one ream of bond paper is $27. -/
theorem bond_paper_cost (bpc : BondPaperCost)
  (h1 : bpc.sheets_per_ream = 500)
  (h2 : bpc.sheets_needed = 5000)
  (h3 : bpc.total_cost = 270) :
  cost_per_ream bpc = 27 := by
  sorry

end NUMINAMATH_CALUDE_bond_paper_cost_l928_92825


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l928_92858

/-- Given a circle with radius 5 and an isosceles trapezoid circumscribed around it,
    where the distance between the points of tangency of its lateral sides is 8,
    prove that the area of the trapezoid is 125. -/
theorem isosceles_trapezoid_area (r : ℝ) (d : ℝ) (A : ℝ) :
  r = 5 →
  d = 8 →
  A = (5 * d) * 2.5 →
  A = 125 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l928_92858


namespace NUMINAMATH_CALUDE_sqrt_x_plus_one_meaningful_l928_92839

theorem sqrt_x_plus_one_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x + 1) ↔ x ≥ -1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_one_meaningful_l928_92839


namespace NUMINAMATH_CALUDE_sector_arc_length_l928_92848

/-- Given a circular sector with perimeter 12 and central angle 4 radians,
    the length of its arc is 8. -/
theorem sector_arc_length (p : ℝ) (θ : ℝ) (l : ℝ) (r : ℝ) :
  p = 12 →  -- perimeter of the sector
  θ = 4 →   -- central angle in radians
  p = l + 2 * r →  -- perimeter formula for a sector
  l = θ * r →  -- arc length formula
  l = 8 :=
by sorry

end NUMINAMATH_CALUDE_sector_arc_length_l928_92848


namespace NUMINAMATH_CALUDE_limeade_calories_l928_92875

-- Define the components of limeade
def lime_juice_weight : ℝ := 150
def sugar_weight : ℝ := 200
def water_weight : ℝ := 450

-- Define calorie content per 100g
def lime_juice_calories_per_100g : ℝ := 20
def sugar_calories_per_100g : ℝ := 396
def water_calories_per_100g : ℝ := 0

-- Define the weight of limeade we want to calculate calories for
def limeade_sample_weight : ℝ := 300

-- Theorem statement
theorem limeade_calories : 
  let total_weight := lime_juice_weight + sugar_weight + water_weight
  let total_calories := (lime_juice_calories_per_100g * lime_juice_weight / 100) + 
                        (sugar_calories_per_100g * sugar_weight / 100) + 
                        (water_calories_per_100g * water_weight / 100)
  (total_calories * limeade_sample_weight / total_weight) = 308.25 := by
  sorry

end NUMINAMATH_CALUDE_limeade_calories_l928_92875


namespace NUMINAMATH_CALUDE_andrea_lauren_bike_problem_l928_92820

/-- The problem of Andrea and Lauren biking towards each other --/
theorem andrea_lauren_bike_problem 
  (initial_distance : ℝ) 
  (andrea_speed_ratio : ℝ) 
  (initial_closing_rate : ℝ) 
  (lauren_stop_time : ℝ) 
  (h1 : initial_distance = 30) 
  (h2 : andrea_speed_ratio = 2) 
  (h3 : initial_closing_rate = 2) 
  (h4 : lauren_stop_time = 10) :
  ∃ (total_time : ℝ), 
    total_time = 17.5 ∧ 
    (∃ (lauren_speed : ℝ),
      lauren_speed > 0 ∧
      andrea_speed_ratio * lauren_speed + lauren_speed = initial_closing_rate ∧
      total_time = lauren_stop_time + (initial_distance - lauren_stop_time * initial_closing_rate) / (andrea_speed_ratio * lauren_speed)) :=
by sorry

end NUMINAMATH_CALUDE_andrea_lauren_bike_problem_l928_92820


namespace NUMINAMATH_CALUDE_johnny_savings_l928_92806

/-- The amount Johnny saved in September -/
def september_savings : ℕ := 30

/-- The amount Johnny saved in October -/
def october_savings : ℕ := 49

/-- The amount Johnny saved in November -/
def november_savings : ℕ := 46

/-- The amount Johnny spent on a video game -/
def video_game_cost : ℕ := 58

/-- The amount Johnny has left after all transactions -/
def remaining_money : ℕ := 67

theorem johnny_savings : 
  september_savings + october_savings + november_savings - video_game_cost = remaining_money := by
  sorry

end NUMINAMATH_CALUDE_johnny_savings_l928_92806


namespace NUMINAMATH_CALUDE_gas_price_increase_l928_92874

theorem gas_price_increase (P : ℝ) (h : P > 0) : 
  let first_increase := 0.15
  let consumption_reduction := 0.20948616600790515
  let second_increase := 0.1
  let final_price := P * (1 + first_increase) * (1 + second_increase)
  let reduced_consumption := 1 - consumption_reduction
  reduced_consumption * final_price = P :=
by sorry

end NUMINAMATH_CALUDE_gas_price_increase_l928_92874


namespace NUMINAMATH_CALUDE_inequality_theorem_l928_92870

theorem inequality_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / c + c / b ≥ 4 * a / (a + b) ∧
  (a / c + c / b = 4 * a / (a + b) ↔ a = b ∧ b = c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l928_92870


namespace NUMINAMATH_CALUDE_inequality_proof_l928_92800

theorem inequality_proof (a b c : ℝ) 
  (h1 : 0 < a ∧ a < 1) 
  (h2 : 0 < b ∧ b < 1) 
  (h3 : 0 < c ∧ c < 1) 
  (h4 : a * b * c = Real.sqrt 3 / 9) : 
  a / (1 - a^2) + b / (1 - b^2) + c / (1 - c^2) ≥ 3 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l928_92800


namespace NUMINAMATH_CALUDE_inverse_g_sum_l928_92830

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 - x else x^2 - 4*x + 5

theorem inverse_g_sum : ∃ y₁ y₂ y₃ : ℝ,
  g y₁ = -1 ∧ g y₂ = 1 ∧ g y₃ = 4 ∧ y₁ + y₂ + y₃ = 4 :=
sorry

end NUMINAMATH_CALUDE_inverse_g_sum_l928_92830


namespace NUMINAMATH_CALUDE_same_group_probability_l928_92831

/-- The probability that two randomly selected people from a group of 16 divided into two equal subgroups are from the same subgroup is 7/15. -/
theorem same_group_probability (n : ℕ) (h1 : n = 16) (h2 : n % 2 = 0) : 
  (Nat.choose (n / 2) 2 * 2) / Nat.choose n 2 = 7 / 15 := by
  sorry

end NUMINAMATH_CALUDE_same_group_probability_l928_92831


namespace NUMINAMATH_CALUDE_sum_of_relatively_prime_integers_l928_92887

theorem sum_of_relatively_prime_integers (n : ℤ) (h : n ≥ 7) :
  ∃ a b : ℤ, n = a + b ∧ a > 1 ∧ b > 1 ∧ Int.gcd a b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_relatively_prime_integers_l928_92887


namespace NUMINAMATH_CALUDE_fruit_brought_to_school_l928_92898

/-- 
Given:
- Mark had an initial number of fruit pieces for the week
- Mark ate a certain number of fruit pieces in the first four days
- Mark decided to keep some pieces for next week

Prove that the number of fruit pieces Mark brought to school on Friday
is equal to the initial number minus the number eaten minus the number kept for next week
-/
theorem fruit_brought_to_school (initial_fruit pieces_eaten pieces_kept : ℕ) :
  initial_fruit - pieces_eaten - pieces_kept = initial_fruit - (pieces_eaten + pieces_kept) :=
by sorry

end NUMINAMATH_CALUDE_fruit_brought_to_school_l928_92898


namespace NUMINAMATH_CALUDE_toy_cars_in_first_box_l928_92846

theorem toy_cars_in_first_box 
  (total_boxes : Nat)
  (total_cars : Nat)
  (cars_in_second : Nat)
  (cars_in_third : Nat)
  (h1 : total_boxes = 3)
  (h2 : total_cars = 71)
  (h3 : cars_in_second = 31)
  (h4 : cars_in_third = 19) :
  total_cars - cars_in_second - cars_in_third = 21 :=
by sorry

end NUMINAMATH_CALUDE_toy_cars_in_first_box_l928_92846


namespace NUMINAMATH_CALUDE_seashell_fraction_proof_l928_92832

def dozen : ℕ := 12

theorem seashell_fraction_proof 
  (mimi_shells : ℕ) 
  (kyle_shells : ℕ) 
  (leigh_shells : ℕ) :
  mimi_shells = 2 * dozen →
  kyle_shells = 2 * mimi_shells →
  leigh_shells = 16 →
  (leigh_shells : ℚ) / (kyle_shells : ℚ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_seashell_fraction_proof_l928_92832


namespace NUMINAMATH_CALUDE_bird_families_flew_away_l928_92890

/-- The number of bird families that flew away is equal to the difference between
    the total number of bird families and the number of bird families left. -/
theorem bird_families_flew_away (total : ℕ) (left : ℕ) (flew_away : ℕ) 
    (h1 : total = 67) (h2 : left = 35) (h3 : flew_away = total - left) : 
    flew_away = 32 := by
  sorry

end NUMINAMATH_CALUDE_bird_families_flew_away_l928_92890


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l928_92813

/-- Given a hyperbola with specified asymptotes and a point it passes through,
    prove that the distance between its foci is 2√(13.5). -/
theorem hyperbola_foci_distance (x₀ y₀ : ℝ) :
  let asymptote1 : ℝ → ℝ := λ x => 2 * x + 2
  let asymptote2 : ℝ → ℝ := λ x => -2 * x + 4
  let point : ℝ × ℝ := (2, 6)
  (∀ x, y₀ = asymptote1 x ∨ y₀ = asymptote2 x → x₀ = x) →
  (y₀ = asymptote1 x₀ ∨ y₀ = asymptote2 x₀) →
  point.1 = 2 ∧ point.2 = 6 →
  ∃ (center : ℝ × ℝ) (a b : ℝ),
    (∀ x y, ((y - center.2)^2 / a^2) - ((x - center.1)^2 / b^2) = 1 →
      y = asymptote1 x ∨ y = asymptote2 x) ∧
    ((point.2 - center.2)^2 / a^2) - ((point.1 - center.1)^2 / b^2) = 1 ∧
    2 * Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 13.5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l928_92813


namespace NUMINAMATH_CALUDE_no_integer_square_root_l928_92886

theorem no_integer_square_root : ¬ ∃ (y : ℤ) (b : ℤ), y^4 + 8*y^3 + 18*y^2 + 10*y + 41 = b^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_square_root_l928_92886


namespace NUMINAMATH_CALUDE_trigonometric_identity_l928_92856

theorem trigonometric_identity : 
  (Real.cos (28 * π / 180) * Real.cos (56 * π / 180)) / Real.sin (2 * π / 180) + 
  (Real.cos (2 * π / 180) * Real.cos (4 * π / 180)) / Real.sin (28 * π / 180) = 
  (Real.sqrt 3 * Real.sin (38 * π / 180)) / (4 * Real.sin (2 * π / 180) * Real.sin (28 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l928_92856


namespace NUMINAMATH_CALUDE_perfect_square_sum_l928_92857

theorem perfect_square_sum : ∃ k : ℕ, 2^8 + 2^11 + 2^12 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sum_l928_92857


namespace NUMINAMATH_CALUDE_smallest_three_digit_palindrome_times_103_not_six_digit_palindrome_l928_92824

/-- Checks if a number is a three-digit palindrome -/
def isThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10)

/-- Checks if a number is a six-digit palindrome -/
def isSixDigitPalindrome (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999 ∧ 
  (n / 100000 = n % 10) ∧ 
  ((n / 10000) % 10 = (n / 10) % 10) ∧
  ((n / 1000) % 10 = (n / 100) % 10)

/-- The main theorem -/
theorem smallest_three_digit_palindrome_times_103_not_six_digit_palindrome :
  isThreeDigitPalindrome 131 ∧
  ¬(isSixDigitPalindrome (131 * 103)) ∧
  ∀ n : ℕ, isThreeDigitPalindrome n ∧ n < 131 → isSixDigitPalindrome (n * 103) :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_palindrome_times_103_not_six_digit_palindrome_l928_92824


namespace NUMINAMATH_CALUDE_increase_by_percentage_increase_800_by_110_percent_l928_92817

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial * (1 + percentage / 100) = initial + initial * (percentage / 100) := by
  sorry

theorem increase_800_by_110_percent :
  800 * (1 + 110 / 100) = 1680 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_increase_800_by_110_percent_l928_92817


namespace NUMINAMATH_CALUDE_greatest_x_given_lcm_l928_92880

theorem greatest_x_given_lcm (x : ℕ) : 
  Nat.lcm x (Nat.lcm 15 21) = 105 → x ≤ 105 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_given_lcm_l928_92880


namespace NUMINAMATH_CALUDE_smallest_n_for_probability_l928_92807

theorem smallest_n_for_probability (n : ℕ) : n ≥ 11 ↔ (n - 4 : ℝ)^3 / (n - 2 : ℝ)^3 > 1/3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_probability_l928_92807


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l928_92897

theorem max_value_sqrt_sum (x y z : ℝ) 
  (sum_eq_one : x + y + z = 1)
  (x_ge : x ≥ -1/3)
  (y_ge : y ≥ -1)
  (z_ge : z ≥ -5/3) :
  ∃ (max : ℝ), max = 6 ∧ 
    ∀ (a b c : ℝ), a + b + c = 1 → a ≥ -1/3 → b ≥ -1 → c ≥ -5/3 →
      Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 3) + Real.sqrt (3 * c + 5) ≤ max ∧
      Real.sqrt (3 * x + 1) + Real.sqrt (3 * y + 3) + Real.sqrt (3 * z + 5) = max :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l928_92897


namespace NUMINAMATH_CALUDE_polynomial_factorization_l928_92888

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 9) * (x^2 + 6*x + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l928_92888


namespace NUMINAMATH_CALUDE_area_of_specific_triangle_l928_92859

/-- Configuration of hexagons with a central hexagon of side length 2,
    surrounded by hexagons of side length 2 and 1 -/
structure HexagonConfiguration where
  centralHexagonSide : ℝ
  firstLevelSide : ℝ
  secondLevelSide : ℝ

/-- The triangle formed by connecting centers of three specific hexagons
    at the second surrounding level -/
def TriangleAtSecondLevel (config : HexagonConfiguration) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a triangle -/
def triangleArea (triangle : Set (ℝ × ℝ)) : ℝ :=
  sorry

theorem area_of_specific_triangle (config : HexagonConfiguration) 
  (h1 : config.centralHexagonSide = 2)
  (h2 : config.firstLevelSide = 2)
  (h3 : config.secondLevelSide = 1) :
  triangleArea (TriangleAtSecondLevel config) = 48 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_area_of_specific_triangle_l928_92859


namespace NUMINAMATH_CALUDE_parabola_focus_l928_92895

/-- A parabola is defined by the equation x^2 = 4y -/
structure Parabola where
  eq : ∀ x y : ℝ, x^2 = 4*y

/-- The focus of a parabola is a point (h, k) where h and k are real numbers -/
structure Focus where
  h : ℝ
  k : ℝ

/-- Theorem: The focus of the parabola x^2 = 4y has coordinates (0, 1) -/
theorem parabola_focus (p : Parabola) : ∃ f : Focus, f.h = 0 ∧ f.k = 1 :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l928_92895


namespace NUMINAMATH_CALUDE_june_1_2014_is_sunday_l928_92826

def is_leap_year (year : ℕ) : Bool :=
  year % 4 = 0 && (year % 100 ≠ 0 || year % 400 = 0)

def days_in_month (year : ℕ) (month : ℕ) : ℕ :=
  if month = 2 then
    if is_leap_year year then 29 else 28
  else if month ∈ [4, 6, 9, 11] then 30
  else 31

def days_between (start_year start_month start_day : ℕ) (end_year end_month end_day : ℕ) : ℕ :=
  sorry

theorem june_1_2014_is_sunday :
  let start_date := (2013, 12, 31)
  let end_date := (2014, 6, 1)
  let start_day_of_week := 2  -- Tuesday
  let days_passed := days_between start_date.1 start_date.2.1 start_date.2.2 end_date.1 end_date.2.1 end_date.2.2
  (start_day_of_week + days_passed) % 7 = 0 :=
by sorry

end NUMINAMATH_CALUDE_june_1_2014_is_sunday_l928_92826


namespace NUMINAMATH_CALUDE_junior_girls_count_l928_92847

theorem junior_girls_count (total_players : ℕ) (boy_percentage : ℚ) : 
  total_players = 50 → 
  boy_percentage = 60 / 100 → 
  (total_players : ℚ) * (1 - boy_percentage) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_junior_girls_count_l928_92847


namespace NUMINAMATH_CALUDE_max_min_values_of_f_l928_92842

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_min_values_of_f :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc 0 2, f x ≤ max) ∧
    (∃ x ∈ Set.Icc 0 2, f x = max) ∧
    (∀ x ∈ Set.Icc 0 2, min ≤ f x) ∧
    (∃ x ∈ Set.Icc 0 2, f x = min) ∧
    max = 5 ∧ min = -15 := by
  sorry

end NUMINAMATH_CALUDE_max_min_values_of_f_l928_92842


namespace NUMINAMATH_CALUDE_parallelogram_area_from_complex_equations_sum_pqrs_equals_102_l928_92868

theorem parallelogram_area_from_complex_equations : ℂ → Prop :=
  fun i =>
  i * i = -1 →
  let eq1 := fun z : ℂ => z * z = 9 + 9 * Real.sqrt 7 * i
  let eq2 := fun z : ℂ => z * z = 5 + 5 * Real.sqrt 2 * i
  let solutions := {z : ℂ | eq1 z ∨ eq2 z}
  let parallelogram_area := Real.sqrt 96 * 2 - Real.sqrt 2 * 2
  (∃ (v1 v2 v3 v4 : ℂ), v1 ∈ solutions ∧ v2 ∈ solutions ∧ v3 ∈ solutions ∧ v4 ∈ solutions ∧
    (v1 - v2).im * (v3 - v4).re - (v1 - v2).re * (v3 - v4).im = parallelogram_area)

/-- The sum of p, q, r, and s is 102 -/
theorem sum_pqrs_equals_102 : 2 + 96 + 2 + 2 = 102 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_from_complex_equations_sum_pqrs_equals_102_l928_92868


namespace NUMINAMATH_CALUDE_clock_rings_eight_times_l928_92845

/-- A clock that rings every 3 hours, starting at 1 A.M. -/
structure Clock :=
  (ring_interval : ℕ := 3)
  (first_ring : ℕ := 1)

/-- The number of times the clock rings in a 24-hour period -/
def rings_per_day (c : Clock) : ℕ :=
  ((24 - c.first_ring) / c.ring_interval) + 1

theorem clock_rings_eight_times (c : Clock) : rings_per_day c = 8 := by
  sorry

end NUMINAMATH_CALUDE_clock_rings_eight_times_l928_92845


namespace NUMINAMATH_CALUDE_no_solution_iff_k_eq_seven_l928_92899

theorem no_solution_iff_k_eq_seven :
  ∀ k : ℝ, (∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → (x - 3) / (x - 4) ≠ (x - k) / (x - 8)) ↔ k = 7 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_k_eq_seven_l928_92899


namespace NUMINAMATH_CALUDE_simplify_expression_square_root_of_expression_l928_92893

-- Part 1
theorem simplify_expression (x : ℝ) (h : 1 < x ∧ x < 4) :
  Real.sqrt ((1 - x)^2) - abs (x - 5) = 2 * x - 6 := by sorry

-- Part 2
theorem square_root_of_expression (x y : ℝ) (h : y = 1 + Real.sqrt (2*x - 1) + Real.sqrt (1 - 2*x)) :
  Real.sqrt (2*x + 3*y) = 2 ∨ Real.sqrt (2*x + 3*y) = -2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_square_root_of_expression_l928_92893


namespace NUMINAMATH_CALUDE_tetrahedron_properties_l928_92878

/-- Represents a tetrahedron SABC with mutually perpendicular edges SA, SB, SC -/
structure Tetrahedron where
  SA : ℝ
  SB : ℝ
  SC : ℝ
  perpendicular : True -- Represents that SA, SB, SC are mutually perpendicular

/-- The radius of the circumscribed sphere of the tetrahedron -/
def circumscribedSphereRadius (t : Tetrahedron) : ℝ :=
  sorry

/-- Determines if there exists a sphere with radius smaller than R that contains the tetrahedron -/
def existsSmallerSphere (t : Tetrahedron) (R : ℝ) : Prop :=
  sorry

theorem tetrahedron_properties (t : Tetrahedron) 
    (h1 : t.SA = 2) (h2 : t.SB = 3) (h3 : t.SC = 6) : 
    circumscribedSphereRadius t = 7/2 ∧ existsSmallerSphere t (7/2) :=
  sorry

end NUMINAMATH_CALUDE_tetrahedron_properties_l928_92878


namespace NUMINAMATH_CALUDE_ln_inequality_range_l928_92833

theorem ln_inequality_range (x : ℝ) (m : ℝ) (h : x > 0) :
  (∀ x > 0, Real.log x ≤ x * Real.exp (m^2 - m - 1)) ↔ (m ≤ 0 ∨ m ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_ln_inequality_range_l928_92833


namespace NUMINAMATH_CALUDE_circle_ellipse_tangent_l928_92814

-- Define the circle M
def circle_M (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*m*x - 3 = 0

-- Define the ellipse C
def ellipse_C (a : ℝ) (x y : ℝ) : Prop :=
  x^2/a^2 + y^2/3 = 1

-- Define the line l
def line_l (c : ℝ) (x y : ℝ) : Prop :=
  x = -c

theorem circle_ellipse_tangent (m c a : ℝ) :
  m < 0 →  -- m is negative
  (∀ x y, circle_M m x y → (x + m)^2 + y^2 = 4) →  -- radius of M is 2
  (∃ x y, ellipse_C a x y ∧ x = -c ∧ y = 0) →  -- left focus of C is F(-c, 0)
  (∀ x y, line_l c x y → (x - 1)^2 = 4) →  -- l is tangent to M
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_ellipse_tangent_l928_92814


namespace NUMINAMATH_CALUDE_pyramid_volume_l928_92818

/-- The volume of a pyramid with a square base of side length 10 and edges of length 15 from apex to base corners is 500√7 / 3 -/
theorem pyramid_volume : 
  ∀ (base_side edge_length : ℝ) (volume : ℝ),
  base_side = 10 →
  edge_length = 15 →
  volume = (1/3) * base_side^2 * (edge_length^2 - (base_side^2/2))^(1/2) →
  volume = 500 * Real.sqrt 7 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_l928_92818


namespace NUMINAMATH_CALUDE_no_integer_solutions_l928_92841

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), x ≠ 1 ∧ (x^7 - 1) / (x - 1) = y^5 - 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l928_92841


namespace NUMINAMATH_CALUDE_pencil_distribution_l928_92867

theorem pencil_distribution (total_pencils : ℕ) (num_people : ℕ) 
  (h1 : total_pencils = 24) 
  (h2 : num_people = 3) : 
  total_pencils / num_people = 8 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l928_92867


namespace NUMINAMATH_CALUDE_A_is_largest_l928_92861

/-- The value of expression A -/
def A : ℚ := 3009 / 3008 + 3009 / 3010

/-- The value of expression B -/
def B : ℚ := 3011 / 3010 + 3011 / 3012

/-- The value of expression C -/
def C : ℚ := 3010 / 3009 + 3010 / 3011

/-- Theorem stating that A is the largest among A, B, and C -/
theorem A_is_largest : A > B ∧ A > C := by
  sorry

end NUMINAMATH_CALUDE_A_is_largest_l928_92861


namespace NUMINAMATH_CALUDE_exists_k_divisible_by_power_of_three_l928_92840

theorem exists_k_divisible_by_power_of_three : 
  ∃ k : ℤ, (3 : ℤ)^2008 ∣ (k^3 - 36*k^2 + 51*k - 97) := by
  sorry

end NUMINAMATH_CALUDE_exists_k_divisible_by_power_of_three_l928_92840


namespace NUMINAMATH_CALUDE_sons_age_l928_92837

theorem sons_age (son father : ℕ) 
  (h1 : son = (father / 4) - 1)
  (h2 : father = 5 * son - 5) : 
  son = 9 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l928_92837


namespace NUMINAMATH_CALUDE_oak_trees_in_park_l928_92885

/-- The number of oak trees remaining in a park after some are cut down -/
def remaining_oak_trees (initial : ℕ) (cut_down : ℕ) : ℕ :=
  initial - cut_down

/-- Theorem stating that 7 oak trees remain after cutting down 2 from an initial 9 -/
theorem oak_trees_in_park : remaining_oak_trees 9 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_oak_trees_in_park_l928_92885


namespace NUMINAMATH_CALUDE_paintings_per_room_l928_92882

theorem paintings_per_room (total_paintings : ℕ) (num_rooms : ℕ) 
  (h1 : total_paintings = 32) 
  (h2 : num_rooms = 4) 
  (h3 : total_paintings % num_rooms = 0) : 
  total_paintings / num_rooms = 8 := by
  sorry

end NUMINAMATH_CALUDE_paintings_per_room_l928_92882


namespace NUMINAMATH_CALUDE_rectangle_square_division_l928_92823

theorem rectangle_square_division (n : ℕ) : 
  (∃ (a b c d : ℕ), 
    a * b = n ∧ 
    c * d = n + 76 ∧ 
    a * d = b * c) → 
  n = 324 := by sorry

end NUMINAMATH_CALUDE_rectangle_square_division_l928_92823


namespace NUMINAMATH_CALUDE_gallon_paint_cost_l928_92843

def pints_needed : ℕ := 8
def pint_cost : ℚ := 8
def gallon_equivalent_pints : ℕ := 8
def savings : ℚ := 9

def total_pint_cost : ℚ := pints_needed * pint_cost

theorem gallon_paint_cost : 
  total_pint_cost - savings = 55 := by sorry

end NUMINAMATH_CALUDE_gallon_paint_cost_l928_92843


namespace NUMINAMATH_CALUDE_zero_lite_soda_bottles_l928_92835

/-- The number of bottles of lite soda in a grocery store -/
def lite_soda_bottles (regular_soda diet_soda total_regular_and_diet : ℕ) : ℕ :=
  total_regular_and_diet - (regular_soda + diet_soda)

/-- Theorem: The number of lite soda bottles is 0 -/
theorem zero_lite_soda_bottles :
  lite_soda_bottles 49 40 89 = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_lite_soda_bottles_l928_92835


namespace NUMINAMATH_CALUDE_square_of_square_plus_eight_l928_92879

theorem square_of_square_plus_eight : (4^2 + 8)^2 = 576 := by
  sorry

end NUMINAMATH_CALUDE_square_of_square_plus_eight_l928_92879


namespace NUMINAMATH_CALUDE_t_shape_perimeter_l928_92889

/-- A T-shaped figure composed of squares -/
structure TShape where
  side_length : ℝ
  is_t_shaped : Bool
  horizontal_squares : ℕ
  vertical_squares : ℕ

/-- Calculate the perimeter of a T-shaped figure -/
def perimeter (t : TShape) : ℝ :=
  sorry

/-- Theorem: The perimeter of the specific T-shaped figure is 18 -/
theorem t_shape_perimeter :
  ∃ (t : TShape),
    t.side_length = 2 ∧
    t.is_t_shaped = true ∧
    t.horizontal_squares = 3 ∧
    t.vertical_squares = 1 ∧
    perimeter t = 18 :=
  sorry

end NUMINAMATH_CALUDE_t_shape_perimeter_l928_92889


namespace NUMINAMATH_CALUDE_problem_sum_value_l928_92854

/-- The sum of the first n terms of a geometric series with first term a and common ratio r -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The sum of the geometric series (3/4)^k from k=1 to 12 -/
def problem_sum : ℚ := geometric_sum (3/4) (3/4) 12

theorem problem_sum_value : problem_sum = 48738225 / 16777216 := by
  sorry

end NUMINAMATH_CALUDE_problem_sum_value_l928_92854


namespace NUMINAMATH_CALUDE_area_between_curves_l928_92883

-- Define the two functions
def f (y : ℝ) : ℝ := 4 - (y - 1)^2
def g (y : ℝ) : ℝ := y^2 - 4*y + 3

-- Define the bounds of integration
def lower_bound : ℝ := 0
def upper_bound : ℝ := 3

-- State the theorem
theorem area_between_curves : 
  (∫ y in lower_bound..upper_bound, f y - g y) = 9 := by sorry

end NUMINAMATH_CALUDE_area_between_curves_l928_92883


namespace NUMINAMATH_CALUDE_inequality_proof_l928_92873

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) * (a^2 + b^2 + c^2) ≥ 9 * a * b * c :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l928_92873


namespace NUMINAMATH_CALUDE_jerome_solution_l928_92827

def jerome_problem (initial_money : ℕ) (remaining_money : ℕ) (meg_amount : ℕ) : Prop :=
  initial_money = 2 * 43 ∧
  remaining_money = 54 ∧
  initial_money = remaining_money + meg_amount + 3 * meg_amount ∧
  meg_amount = 8

theorem jerome_solution : ∃ initial_money remaining_money meg_amount, jerome_problem initial_money remaining_money meg_amount :=
  sorry

end NUMINAMATH_CALUDE_jerome_solution_l928_92827


namespace NUMINAMATH_CALUDE_course_size_l928_92864

theorem course_size (a b c d : ℕ) (h1 : a + b + c + d = 800) 
  (h2 : a = 800 / 5) (h3 : b = 800 / 4) (h4 : c = 800 / 2) (h5 : d = 40) : 
  800 = 800 := by sorry

end NUMINAMATH_CALUDE_course_size_l928_92864


namespace NUMINAMATH_CALUDE_unique_corresponding_point_l928_92863

-- Define a square as a structure with a side length and a position
structure Square where
  sideLength : ℝ
  position : ℝ × ℝ

-- Define the problem setup
axiom larger_square : Square
axiom smaller_square : Square

-- The smaller square is entirely within the larger square
axiom smaller_inside_larger :
  smaller_square.position.1 ≥ larger_square.position.1 ∧
  smaller_square.position.1 + smaller_square.sideLength ≤ larger_square.position.1 + larger_square.sideLength ∧
  smaller_square.position.2 ≥ larger_square.position.2 ∧
  smaller_square.position.2 + smaller_square.sideLength ≤ larger_square.position.2 + larger_square.sideLength

-- The squares have the same area
axiom same_area : larger_square.sideLength^2 = smaller_square.sideLength^2

-- Define a point as a pair of real numbers
def Point := ℝ × ℝ

-- The theorem to be proved
theorem unique_corresponding_point :
  ∃! p : Point,
    (p.1 - larger_square.position.1) / larger_square.sideLength =
    (p.1 - smaller_square.position.1) / smaller_square.sideLength ∧
    (p.2 - larger_square.position.2) / larger_square.sideLength =
    (p.2 - smaller_square.position.2) / smaller_square.sideLength :=
  sorry

end NUMINAMATH_CALUDE_unique_corresponding_point_l928_92863


namespace NUMINAMATH_CALUDE_playground_area_l928_92804

theorem playground_area (perimeter width length : ℝ) : 
  perimeter = 120 →
  length = 3 * width →
  2 * length + 2 * width = perimeter →
  length * width = 675 :=
by
  sorry

end NUMINAMATH_CALUDE_playground_area_l928_92804


namespace NUMINAMATH_CALUDE_bus_speed_problem_l928_92803

theorem bus_speed_problem (distance : ℝ) (speed_increase : ℝ) (time_reduction : ℝ) :
  distance = 660 ∧ 
  speed_increase = 5 ∧ 
  time_reduction = 1 →
  ∃ (v : ℝ), 
    v > 0 ∧
    distance / v - time_reduction = distance / (v + speed_increase) ∧
    v = 55 := by
  sorry

end NUMINAMATH_CALUDE_bus_speed_problem_l928_92803


namespace NUMINAMATH_CALUDE_strawberry_picking_problem_l928_92819

/-- The number of times Kimberly picked more strawberries than her brother -/
def kimberlyMultiplier : ℕ → ℕ → ℕ → ℕ → ℕ
| brother_baskets, strawberries_per_basket, parents_difference, equal_share =>
  let brother_strawberries := brother_baskets * strawberries_per_basket
  let total_strawberries := equal_share * 4
  2 * total_strawberries / brother_strawberries - 2

theorem strawberry_picking_problem :
  kimberlyMultiplier 3 15 93 168 = 8 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_picking_problem_l928_92819


namespace NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l928_92801

def angle : Int := -1120

def is_coterminal (a b : Int) : Prop :=
  ∃ k : Int, a = b + k * 360

def in_fourth_quadrant (a : Int) : Prop :=
  ∃ b : Int, is_coterminal a b ∧ 270 ≤ b ∧ b < 360

theorem angle_in_fourth_quadrant : in_fourth_quadrant angle := by
  sorry

end NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l928_92801


namespace NUMINAMATH_CALUDE_planet_coloring_l928_92805

/-- Given 3 people coloring planets with 24 total colors, prove each person uses 8 colors --/
theorem planet_coloring (total_colors : ℕ) (num_people : ℕ) (h1 : total_colors = 24) (h2 : num_people = 3) :
  total_colors / num_people = 8 := by
  sorry

end NUMINAMATH_CALUDE_planet_coloring_l928_92805


namespace NUMINAMATH_CALUDE_perimeter_is_24_l928_92862

/-- A right triangle ABC with specific properties -/
structure RightTriangleABC where
  -- Points A, B, and C
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- ABC is a right triangle with right angle at B
  is_right_triangle : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0
  -- Angle BAC equals angle BCA
  angle_equality : (A.1 - B.1) * (A.1 - C.1) + (A.2 - B.2) * (A.2 - C.2) = 
                   (C.1 - B.1) * (C.1 - A.1) + (C.2 - B.2) * (C.2 - A.2)
  -- Length of AB is 9
  AB_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 9
  -- Length of BC is 6
  BC_length : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 6

/-- The perimeter of the right triangle ABC is 24 -/
theorem perimeter_is_24 (t : RightTriangleABC) : 
  Real.sqrt ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2) +
  Real.sqrt ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2) +
  Real.sqrt ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2) = 24 := by
  sorry


end NUMINAMATH_CALUDE_perimeter_is_24_l928_92862


namespace NUMINAMATH_CALUDE_correct_calculation_l928_92815

theorem correct_calculation (x : ℤ) (h : x + 5 = 43) : 5 * x = 190 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l928_92815


namespace NUMINAMATH_CALUDE_angle_420_equals_60_l928_92852

/-- The angle (in degrees) that represents a full rotation in a standard coordinate system -/
def full_rotation : ℝ := 360

/-- Two angles have the same terminal side if their difference is a multiple of a full rotation -/
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α - β = k * full_rotation

/-- Theorem: The angle 420° has the same terminal side as 60° -/
theorem angle_420_equals_60 : same_terminal_side 420 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_420_equals_60_l928_92852


namespace NUMINAMATH_CALUDE_problem_statement_l928_92894

theorem problem_statement (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (sum_prod_nonzero : a * b + a * c + b * c ≠ 0) :
  (a^7 + b^7 + c^7) / (a * b * c * (a * b + a * c + b * c)) = -7 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l928_92894


namespace NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l928_92812

/-- The volume of a cylinder minus the volume of two congruent cones --/
theorem cylinder_minus_cones_volume 
  (r : ℝ) 
  (h_cylinder : ℝ) 
  (h_cone : ℝ) 
  (h_cylinder_eq : h_cylinder = 30) 
  (h_cone_eq : h_cone = 15) 
  (r_eq : r = 10) : 
  π * r^2 * h_cylinder - 2 * (1/3 * π * r^2 * h_cone) = 2000 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l928_92812


namespace NUMINAMATH_CALUDE_investment_ratio_from_profit_ratio_and_time_l928_92844

/-- Given two partners with investments and profits, prove their investment ratio -/
theorem investment_ratio_from_profit_ratio_and_time (p q : ℝ) (h1 : p > 0) (h2 : q > 0) :
  (p * 20) / (q * 40) = 7 / 10 → p / q = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_investment_ratio_from_profit_ratio_and_time_l928_92844


namespace NUMINAMATH_CALUDE_probability_king_of_diamonds_l928_92866

/-- Represents a standard playing card suit -/
inductive Suit
| Spades
| Hearts
| Diamonds
| Clubs

/-- Represents a standard playing card rank -/
inductive Rank
| Ace
| Two
| Three
| Four
| Five
| Six
| Seven
| Eight
| Nine
| Ten
| Jack
| Queen
| King

/-- Represents a standard playing card -/
structure Card where
  rank : Rank
  suit : Suit

def standardDeck : Finset Card := sorry

/-- The probability of drawing a specific card from a standard deck -/
def probabilityOfCard (c : Card) : ℚ :=
  1 / (Finset.card standardDeck)

theorem probability_king_of_diamonds :
  probabilityOfCard ⟨Rank.King, Suit.Diamonds⟩ = 1 / 52 := by
  sorry

end NUMINAMATH_CALUDE_probability_king_of_diamonds_l928_92866


namespace NUMINAMATH_CALUDE_inequality_proof_l928_92876

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  1/x + 1/y ≥ 3 + 2*Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l928_92876


namespace NUMINAMATH_CALUDE_ryanne_hezekiah_age_difference_l928_92828

/-- Given that Ryanne and Hezekiah's combined age is 15 and Hezekiah is 4 years old,
    prove that Ryanne is 7 years older than Hezekiah. -/
theorem ryanne_hezekiah_age_difference :
  ∀ (ryanne_age hezekiah_age : ℕ),
    ryanne_age + hezekiah_age = 15 →
    hezekiah_age = 4 →
    ryanne_age - hezekiah_age = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_ryanne_hezekiah_age_difference_l928_92828
