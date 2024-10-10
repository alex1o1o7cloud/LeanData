import Mathlib

namespace team_not_losing_probability_l1656_165643

/-- Represents the positions Player A can play -/
inductive Position
| CenterForward
| Winger
| AttackingMidfielder

/-- The appearance rate for each position -/
def appearanceRate (pos : Position) : ℝ :=
  match pos with
  | .CenterForward => 0.3
  | .Winger => 0.5
  | .AttackingMidfielder => 0.2

/-- The probability of the team losing when Player A plays in each position -/
def losingProbability (pos : Position) : ℝ :=
  match pos with
  | .CenterForward => 0.3
  | .Winger => 0.2
  | .AttackingMidfielder => 0.2

/-- The probability of the team not losing when Player A participates -/
def teamNotLosingProbability : ℝ :=
  (appearanceRate Position.CenterForward * (1 - losingProbability Position.CenterForward)) +
  (appearanceRate Position.Winger * (1 - losingProbability Position.Winger)) +
  (appearanceRate Position.AttackingMidfielder * (1 - losingProbability Position.AttackingMidfielder))

theorem team_not_losing_probability :
  teamNotLosingProbability = 0.77 := by
  sorry

end team_not_losing_probability_l1656_165643


namespace median_and_mode_of_S_l1656_165620

/-- The set of data --/
def S : Finset ℕ := {6, 7, 4, 7, 5, 2}

/-- Definition of median for a finite set of natural numbers --/
def median (s : Finset ℕ) : ℚ := sorry

/-- Definition of mode for a finite set of natural numbers --/
def mode (s : Finset ℕ) : ℕ := sorry

theorem median_and_mode_of_S :
  median S = 5.5 ∧ mode S = 7 := by sorry

end median_and_mode_of_S_l1656_165620


namespace coin_and_die_probability_l1656_165653

def fair_coin_probability : ℚ := 1 / 2
def fair_die_probability : ℚ := 1 / 6

theorem coin_and_die_probability :
  let p_tails := fair_coin_probability
  let p_one_or_two := 2 * fair_die_probability
  p_tails * p_one_or_two = 1 / 6 :=
by sorry

end coin_and_die_probability_l1656_165653


namespace solve_for_a_l1656_165628

theorem solve_for_a (a b d : ℝ) (h1 : a + b = d) (h2 : b + d = 7) (h3 : d = 4) : a = 1 := by
  sorry

end solve_for_a_l1656_165628


namespace square_root_problem_l1656_165650

theorem square_root_problem (m : ℝ) (x : ℝ) 
  (h1 : m > 0) 
  (h2 : Real.sqrt m = x + 1) 
  (h3 : Real.sqrt m = x - 3) : 
  m = 4 := by sorry

end square_root_problem_l1656_165650


namespace positive_number_square_sum_l1656_165669

theorem positive_number_square_sum (n : ℝ) : n > 0 ∧ n^2 + n = 210 → n = 14 := by
  sorry

end positive_number_square_sum_l1656_165669


namespace polynomial_factorization_l1656_165633

theorem polynomial_factorization (x : ℝ) : x^4 + 16 = (x^2 - 2*x + 2) * (x^2 + 2*x + 2) := by
  sorry

end polynomial_factorization_l1656_165633


namespace firefighter_water_delivery_time_l1656_165618

/-- Proves that 5 firefighters can deliver 4000 gallons of water in 40 minutes -/
theorem firefighter_water_delivery_time :
  let water_needed : ℕ := 4000
  let firefighters : ℕ := 5
  let water_per_minute_per_hose : ℕ := 20
  let total_water_per_minute : ℕ := firefighters * water_per_minute_per_hose
  water_needed / total_water_per_minute = 40 := by
  sorry

end firefighter_water_delivery_time_l1656_165618


namespace value_of_M_l1656_165678

theorem value_of_M : ∃ M : ℝ, (0.2 * M = 0.5 * 1000) ∧ (M = 2500) := by
  sorry

end value_of_M_l1656_165678


namespace curve_tangent_theorem_l1656_165608

/-- A curve defined by y = x² + ax + b -/
def curve (a b : ℝ) : ℝ → ℝ := λ x ↦ x^2 + a*x + b

/-- The derivative of the curve -/
def curve_derivative (a : ℝ) : ℝ → ℝ := λ x ↦ 2*x + a

/-- The tangent line at x = 0 -/
def tangent_at_zero (a b : ℝ) : ℝ → ℝ := λ x ↦ 3*x - b + 1

theorem curve_tangent_theorem (a b : ℝ) :
  (∀ x, tangent_at_zero a b x = 3*x - (curve a b 0) + 1) →
  curve_derivative a 0 = 3 →
  a = 3 ∧ b = 1 := by
  sorry

end curve_tangent_theorem_l1656_165608


namespace solution_set_l1656_165611

theorem solution_set (x : ℝ) : 
  33 * 32 ≤ x ∧ 
  Int.floor x + Int.ceil x = 5 → 
  2 < x ∧ x < 3 :=
by sorry

end solution_set_l1656_165611


namespace triangle_properties_l1656_165674

/-- Properties of a triangle ABC -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the specific triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.a = t.b * Real.cos t.C + Real.sqrt 3 * t.c * Real.sin t.B)
  (h2 : t.b = 2)
  (h3 : t.a = Real.sqrt 3 * t.c) : 
  t.B = π / 6 ∧ t.a * t.c * Real.sin t.B / 2 = Real.sqrt 3 := by
  sorry

end triangle_properties_l1656_165674


namespace business_investment_l1656_165665

theorem business_investment (p q : ℕ) (h1 : q = 15000) (h2 : p / q = 4) : p = 60000 := by
  sorry

end business_investment_l1656_165665


namespace fraction_equivalence_l1656_165615

theorem fraction_equivalence (b c : ℝ) (hb : b ≠ 0) (hc : c ≠ 0) :
  (∀ x : ℝ, x ≠ -c → x ≠ -3*c → (x + 2*b) / (x + 3*c) = (x + b) / (x + c)) ↔ b = 2*c :=
sorry

end fraction_equivalence_l1656_165615


namespace x_value_from_fraction_equality_l1656_165642

theorem x_value_from_fraction_equality (x y : ℝ) :
  x ≠ 1 →
  y^2 + 3*y - 3 ≠ 0 →
  (x / (x - 1) = (y^2 + 3*y - 2) / (y^2 + 3*y - 3)) →
  x = (y^2 + 3*y - 2) / 2 := by
  sorry

end x_value_from_fraction_equality_l1656_165642


namespace power_of_product_l1656_165625

theorem power_of_product (a b : ℝ) : (-2 * a^2 * b)^3 = -8 * a^6 * b^3 := by sorry

end power_of_product_l1656_165625


namespace negative_four_squared_times_negative_one_power_2022_l1656_165673

theorem negative_four_squared_times_negative_one_power_2022 :
  -4^2 * (-1)^2022 = -16 := by
  sorry

end negative_four_squared_times_negative_one_power_2022_l1656_165673


namespace equation_solutions_l1656_165605

theorem equation_solutions : 
  {(x, y) : ℕ × ℕ | x > 0 ∧ y > 0 ∧ 2*x^2 + 5*x*y + 2*y^2 = 2006} = 
  {(28, 3), (3, 28)} :=
sorry

end equation_solutions_l1656_165605


namespace distance_difference_l1656_165687

/-- The rate at which Bjorn bikes in miles per hour -/
def bjorn_rate : ℝ := 12

/-- The rate at which Alberto bikes in miles per hour -/
def alberto_rate : ℝ := 15

/-- The duration of the biking trip in hours -/
def trip_duration : ℝ := 6

/-- The theorem stating the difference in distance traveled between Alberto and Bjorn -/
theorem distance_difference : 
  alberto_rate * trip_duration - bjorn_rate * trip_duration = 18 := by
  sorry

end distance_difference_l1656_165687


namespace zeros_count_theorem_l1656_165662

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def has_unique_zero_in_interval (f : ℝ → ℝ) (a b c : ℝ) : Prop :=
  f c = 0 ∧ ∀ x, x ∈ Set.Icc a b → f x = 0 → x = c

def count_zeros_in_interval (f : ℝ → ℝ) (a b : ℝ) : ℕ :=
  sorry

theorem zeros_count_theorem (f : ℝ → ℝ) :
  is_even_function f →
  (∀ x, f (5 + x) = f (5 - x)) →
  has_unique_zero_in_interval f 0 5 1 →
  count_zeros_in_interval f (-2012) 2012 = 806 :=
sorry

end zeros_count_theorem_l1656_165662


namespace video_game_rounds_l1656_165690

/-- The number of points earned per win in the video game competition. -/
def points_per_win : ℕ := 5

/-- The number of points Vlad scored. -/
def vlad_points : ℕ := 64

/-- The total points scored by both players. -/
def total_points : ℕ := 150

/-- Taro's points in terms of the total points. -/
def taro_points (P : ℕ) : ℤ := (3 * P) / 5 - 4

theorem video_game_rounds :
  (total_points = taro_points total_points + vlad_points) →
  (total_points / points_per_win = 30) := by
sorry

end video_game_rounds_l1656_165690


namespace sin_X_in_right_triangle_l1656_165635

-- Define the right triangle XYZ
def RightTriangle (X Y Z : ℝ) : Prop :=
  0 < X ∧ 0 < Y ∧ 0 < Z ∧ X^2 + Y^2 = Z^2

-- State the theorem
theorem sin_X_in_right_triangle :
  ∀ X Y Z : ℝ,
  RightTriangle X Y Z →
  X = 8 →
  Z = 17 →
  Real.sin (Real.arcsin (X / Z)) = 8 / 17 :=
by sorry

end sin_X_in_right_triangle_l1656_165635


namespace ceiling_floor_expression_l1656_165660

theorem ceiling_floor_expression : ⌈(7 : ℝ) / 3⌉ + ⌊-(7 : ℝ) / 3⌋ - 3 = -3 := by
  sorry

end ceiling_floor_expression_l1656_165660


namespace mary_received_more_than_mike_l1656_165626

/-- Represents the profit distribution in a partnership --/
def profit_distribution (mary_investment mike_investment total_profit : ℚ) : ℚ :=
  let equal_share := (1/3) * total_profit / 2
  let ratio_share := (2/3) * total_profit
  let mary_ratio := mary_investment / (mary_investment + mike_investment)
  let mike_ratio := mike_investment / (mary_investment + mike_investment)
  let mary_total := equal_share + mary_ratio * ratio_share
  let mike_total := equal_share + mike_ratio * ratio_share
  mary_total - mike_total

/-- Theorem stating that Mary received $800 more than Mike --/
theorem mary_received_more_than_mike :
  profit_distribution 700 300 3000 = 800 := by
  sorry


end mary_received_more_than_mike_l1656_165626


namespace rectangle_arrangement_possible_l1656_165691

/-- Represents a small 1×2 rectangle with 2 stars -/
structure SmallRectangle :=
  (width : Nat) (height : Nat) (stars : Nat)

/-- Represents the large 5×200 rectangle -/
structure LargeRectangle :=
  (width : Nat) (height : Nat)
  (smallRectangles : List SmallRectangle)

/-- Checks if a number is even -/
def isEven (n : Nat) : Prop := ∃ k, n = 2 * k

/-- Calculates the total number of stars in a list of small rectangles -/
def totalStars (rectangles : List SmallRectangle) : Nat :=
  rectangles.foldl (fun acc rect => acc + rect.stars) 0

/-- Theorem: It's possible to arrange 500 1×2 rectangles into a 5×200 rectangle
    with an even number of stars in each row and column -/
theorem rectangle_arrangement_possible :
  ∃ (largeRect : LargeRectangle),
    largeRect.width = 200 ∧
    largeRect.height = 5 ∧
    largeRect.smallRectangles.length = 500 ∧
    (∀ smallRect ∈ largeRect.smallRectangles, smallRect.width = 1 ∧ smallRect.height = 2 ∧ smallRect.stars = 2) ∧
    (∀ row ∈ List.range 5, isEven (totalStars (largeRect.smallRectangles.filter (fun _ => true)))) ∧
    (∀ col ∈ List.range 200, isEven (totalStars (largeRect.smallRectangles.filter (fun _ => true)))) :=
by sorry

end rectangle_arrangement_possible_l1656_165691


namespace sufficient_not_necessary_condition_l1656_165661

theorem sufficient_not_necessary_condition : 
  (∀ a b : ℝ, a > 1 ∧ b > 1 → a * b > 1) ∧ 
  (∃ a b : ℝ, a * b > 1 ∧ (a ≤ 1 ∨ b ≤ 1)) := by
  sorry

end sufficient_not_necessary_condition_l1656_165661


namespace cookie_sales_problem_l1656_165600

/-- Represents the number of boxes of cookies sold -/
structure CookieSales where
  chocolate : ℕ
  plain : ℕ

/-- Represents the price of cookies in cents -/
def CookiePrice : ℕ × ℕ := (125, 75)

theorem cookie_sales_problem (sales : CookieSales) : 
  sales.chocolate + sales.plain = 1585 →
  125 * sales.chocolate + 75 * sales.plain = 158675 →
  sales.plain = 789 := by
  sorry

end cookie_sales_problem_l1656_165600


namespace valid_numbers_characterization_l1656_165641

/-- A function that moves the last digit of a number to the beginning -/
def moveLastDigitToFront (n : ℕ) : ℕ :=
  let lastDigit := n % 10
  let remainingDigits := n / 10
  lastDigit * 10^5 + remainingDigits

/-- A predicate that checks if a number becomes an integer multiple when its last digit is moved to the front -/
def isValidNumber (n : ℕ) : Prop :=
  ∃ k : ℕ, moveLastDigitToFront n = k * n

/-- The set of all valid six-digit numbers -/
def validNumbers : Finset ℕ :=
  {142857, 102564, 128205, 153846, 179487, 205128, 230769}

/-- The main theorem stating that validNumbers contains all and only the six-digit numbers
    that become an integer multiple when the last digit is moved to the beginning -/
theorem valid_numbers_characterization :
  ∀ n : ℕ, 100000 ≤ n ∧ n < 1000000 →
    (n ∈ validNumbers ↔ isValidNumber n) := by
  sorry

end valid_numbers_characterization_l1656_165641


namespace arithmetic_operations_l1656_165694

theorem arithmetic_operations :
  (8 + (-1/4) - 5 - (-0.25) = 3) ∧
  (-36 * (-2/3 + 5/6 - 7/12 - 8/9) = 47) ∧
  (-2 + 2 / (-1/2) * 2 = -10) ∧
  (-3.5 * (1/6 - 0.5) * 3/7 / (1/2) = 1) := by
  sorry

end arithmetic_operations_l1656_165694


namespace parabola_translation_l1656_165668

/-- A parabola defined by a quadratic function -/
def Parabola (a b c : ℝ) := fun (x : ℝ) => a * x^2 + b * x + c

/-- Translation of a function -/
def translate (f : ℝ → ℝ) (dx dy : ℝ) := fun (x : ℝ) => f (x - dx) + dy

theorem parabola_translation (x : ℝ) :
  translate (Parabola 2 4 1) 1 3 x = Parabola 2 0 0 x := by
  sorry

end parabola_translation_l1656_165668


namespace min_value_sum_fractions_l1656_165679

theorem min_value_sum_fractions (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 9) : 
  (x^2 + y^2)/(x + y) + (x^2 + z^2)/(x + z) + (y^2 + z^2)/(y + z) ≥ 9 := by
  sorry

end min_value_sum_fractions_l1656_165679


namespace least_divisible_by_3_4_5_6_8_divisible_by_3_4_5_6_8_120_least_number_120_l1656_165614

theorem least_divisible_by_3_4_5_6_8 : ∀ n : ℕ, n > 0 → (3 ∣ n) ∧ (4 ∣ n) ∧ (5 ∣ n) ∧ (6 ∣ n) ∧ (8 ∣ n) → n ≥ 120 :=
by
  sorry

theorem divisible_by_3_4_5_6_8_120 : (3 ∣ 120) ∧ (4 ∣ 120) ∧ (5 ∣ 120) ∧ (6 ∣ 120) ∧ (8 ∣ 120) :=
by
  sorry

theorem least_number_120 : ∀ n : ℕ, n > 0 → (3 ∣ n) ∧ (4 ∣ n) ∧ (5 ∣ n) ∧ (6 ∣ n) ∧ (8 ∣ n) → n = 120 :=
by
  sorry

end least_divisible_by_3_4_5_6_8_divisible_by_3_4_5_6_8_120_least_number_120_l1656_165614


namespace calculation_result_quadratic_solution_l1656_165606

-- Problem 1
theorem calculation_result : Real.sqrt 9 + |1 - Real.sqrt 2| + ((-8 : ℝ) ^ (1/3)) - Real.sqrt 2 = 0 := by
  sorry

-- Problem 2
theorem quadratic_solution (x : ℝ) (h : 4 * x^2 - 16 = 0) : x = 2 ∨ x = -2 := by
  sorry

end calculation_result_quadratic_solution_l1656_165606


namespace T_property_M_remainder_l1656_165622

/-- A sequence of positive integers where each number has exactly 9 ones in its binary representation -/
def T : ℕ → ℕ := sorry

/-- The 500th number in the sequence T -/
def M : ℕ := T 500

/-- Predicate to check if a natural number has exactly 9 ones in its binary representation -/
def has_nine_ones (n : ℕ) : Prop := sorry

theorem T_property (n : ℕ) : has_nine_ones (T n) := sorry

theorem M_remainder : M % 500 = 191 := sorry

end T_property_M_remainder_l1656_165622


namespace quadratic_transformation_l1656_165677

theorem quadratic_transformation (a b c : ℝ) (ha : a ≠ 0) :
  ∃ (h k r : ℝ) (hr : r ≠ 0), ∀ x : ℝ,
    a * x^2 + b * x + c = a * ((x - h)^2 / r^2) + k :=
by sorry

end quadratic_transformation_l1656_165677


namespace initial_milk_water_ratio_l1656_165651

/-- Proves that the initial ratio of milk to water is 3:2 given the conditions -/
theorem initial_milk_water_ratio 
  (total_volume : ℝ) 
  (added_water : ℝ) 
  (new_ratio_milk : ℝ) 
  (new_ratio_water : ℝ) 
  (h1 : total_volume = 155) 
  (h2 : added_water = 62) 
  (h3 : new_ratio_milk = 3) 
  (h4 : new_ratio_water = 4) : 
  ∃ (initial_milk initial_water : ℝ), 
    initial_milk + initial_water = total_volume ∧ 
    initial_milk / (initial_water + added_water) = new_ratio_milk / new_ratio_water ∧
    initial_milk / initial_water = 3 / 2 :=
by sorry

end initial_milk_water_ratio_l1656_165651


namespace blue_pick_fraction_l1656_165607

def guitar_pick_collection (total : ℕ) (red : ℕ) (blue : ℕ) (yellow : ℕ) : Prop :=
  red + blue + yellow = total ∧ red = total / 2 ∧ blue = 12 ∧ yellow = 6

theorem blue_pick_fraction 
  (total : ℕ) (red : ℕ) (blue : ℕ) (yellow : ℕ) 
  (h : guitar_pick_collection total red blue yellow) : 
  blue = total / 3 := by
sorry

end blue_pick_fraction_l1656_165607


namespace group_size_l1656_165649

/-- 
Given a group of people with men, women, and children, where:
- The number of men is twice the number of women
- The number of women is 3 times the number of children
- The number of children is 30

Prove that the total number of people in the group is 300.
-/
theorem group_size (children women men : ℕ) 
  (h1 : men = 2 * women) 
  (h2 : women = 3 * children) 
  (h3 : children = 30) : 
  children + women + men = 300 := by
  sorry

end group_size_l1656_165649


namespace shaded_fraction_is_five_thirty_sixths_l1656_165613

/-- Represents a square quilt with a 3x3 grid of unit squares -/
structure Quilt :=
  (size : ℕ := 3)
  (total_area : ℚ := 9)

/-- Calculates the shaded area of the quilt -/
def shaded_area (q : Quilt) : ℚ :=
  let triangle_area : ℚ := 1/2
  let small_square_area : ℚ := 1/4
  let full_square_area : ℚ := 1
  2 * triangle_area + small_square_area + full_square_area

/-- Theorem stating that the shaded area is 5/36 of the total area -/
theorem shaded_fraction_is_five_thirty_sixths (q : Quilt) :
  shaded_area q / q.total_area = 5/36 := by
  sorry

end shaded_fraction_is_five_thirty_sixths_l1656_165613


namespace max_value_of_y_l1656_165637

def y (x : ℝ) : ℝ := |x + 1| - 2 * |x| + |x - 2|

theorem max_value_of_y :
  ∃ (α : ℝ), α = 3 ∧ ∀ x, -1 ≤ x ∧ x ≤ 2 → y x ≤ α :=
by sorry

end max_value_of_y_l1656_165637


namespace brown_family_seating_l1656_165644

/-- The number of ways to arrange boys and girls in a row --/
def seating_arrangements (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  Nat.factorial (num_boys + num_girls) - (Nat.factorial num_boys * Nat.factorial num_girls)

/-- Theorem stating the number of valid seating arrangements for 6 boys and 5 girls --/
theorem brown_family_seating :
  seating_arrangements 6 5 = 39830400 := by
  sorry

end brown_family_seating_l1656_165644


namespace max_y_value_l1656_165658

theorem max_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -4) : 
  ∀ (z : ℤ), z * x + 3 * x + 2 * z ≠ -4 ∨ z ≤ -1 :=
by sorry

end max_y_value_l1656_165658


namespace value_of_a_l1656_165681

theorem value_of_a : ∃ (a : ℝ), 
  (∃ (x y : ℝ), 2*x + y = 3*a ∧ x - 2*y = 9*a ∧ x + 3*y = 24) → 
  a = -4 := by
sorry

end value_of_a_l1656_165681


namespace prism_surface_area_l1656_165617

/-- A right rectangular prism with integer dimensions -/
structure RectPrism where
  l : ℕ
  w : ℕ
  h : ℕ
  l_ne_w : l ≠ w
  w_ne_h : w ≠ h
  h_ne_l : h ≠ l

/-- The processing fee calculation function -/
def processingFee (p : RectPrism) : ℚ :=
  0.3 * p.l + 0.4 * p.w + 0.5 * p.h

/-- The surface area calculation function -/
def surfaceArea (p : RectPrism) : ℕ :=
  2 * (p.l * p.w + p.l * p.h + p.w * p.h)

/-- The main theorem -/
theorem prism_surface_area (p : RectPrism) :
  (∃ (σ₁ σ₂ σ₃ σ₄ : Equiv.Perm (Fin 3)),
    3 * (σ₁.toFun 0 : ℕ) + 4 * (σ₁.toFun 1 : ℕ) + 5 * (σ₁.toFun 2 : ℕ) = 81 ∧
    3 * (σ₂.toFun 0 : ℕ) + 4 * (σ₂.toFun 1 : ℕ) + 5 * (σ₂.toFun 2 : ℕ) = 81 ∧
    3 * (σ₃.toFun 0 : ℕ) + 4 * (σ₃.toFun 1 : ℕ) + 5 * (σ₃.toFun 2 : ℕ) = 87 ∧
    3 * (σ₄.toFun 0 : ℕ) + 4 * (σ₄.toFun 1 : ℕ) + 5 * (σ₄.toFun 2 : ℕ) = 87) →
  surfaceArea p = 276 := by
  sorry


end prism_surface_area_l1656_165617


namespace fraction_equality_l1656_165654

theorem fraction_equality (a b : ℝ) (h : (1/a + 1/b)/(1/a - 1/b) = 1009) :
  (a + b)/(a - b) = -1009 := by sorry

end fraction_equality_l1656_165654


namespace raspberry_pie_degrees_is_45_l1656_165630

/-- The number of degrees in a full circle -/
def full_circle : ℕ := 360

/-- The total number of students in Mandy's class -/
def total_students : ℕ := 48

/-- The number of students preferring chocolate pie -/
def chocolate_preference : ℕ := 18

/-- The number of students preferring apple pie -/
def apple_preference : ℕ := 10

/-- The number of students preferring blueberry pie -/
def blueberry_preference : ℕ := 8

/-- Calculate the number of degrees for raspberry pie in the pie chart -/
def raspberry_pie_degrees : ℚ :=
  let remaining_students := total_students - (chocolate_preference + apple_preference + blueberry_preference)
  let raspberry_preference := remaining_students / 2
  (raspberry_preference : ℚ) / total_students * full_circle

/-- Theorem stating that the number of degrees for raspberry pie is 45° -/
theorem raspberry_pie_degrees_is_45 : raspberry_pie_degrees = 45 := by
  sorry

end raspberry_pie_degrees_is_45_l1656_165630


namespace min_value_expression_l1656_165683

theorem min_value_expression (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) :
  (a + 3*c) / (a + 2*b + c) + 4*b / (a + b + 2*c) - 8*c / (a + b + 3*c) ≥ -17 + 12 * Real.sqrt 2 :=
by sorry

end min_value_expression_l1656_165683


namespace equation_solution_l1656_165670

theorem equation_solution : ∃! x : ℚ, (x - 30) / 3 = (3 * x + 4) / 8 ∧ x = -252 := by sorry

end equation_solution_l1656_165670


namespace expand_expression_l1656_165632

theorem expand_expression (y : ℝ) : 5 * (y + 3) * (y - 2) * (y + 1) = 5 * y^3 + 10 * y^2 - 25 * y - 30 := by
  sorry

end expand_expression_l1656_165632


namespace decimal_to_fraction_l1656_165686

theorem decimal_to_fraction : (3.75 : ℚ) = 15 / 4 := by sorry

end decimal_to_fraction_l1656_165686


namespace inventory_problem_l1656_165601

/-- The inventory problem -/
theorem inventory_problem
  (ties : ℕ) (belts : ℕ) (black_shirts : ℕ) (white_shirts : ℕ) (hats : ℕ) (socks : ℕ)
  (h_ties : ties = 34)
  (h_belts : belts = 40)
  (h_black_shirts : black_shirts = 63)
  (h_white_shirts : white_shirts = 42)
  (h_hats : hats = 25)
  (h_socks : socks = 80)
  : let jeans := (2 * (black_shirts + white_shirts)) / 3
    let scarves := (ties + belts) / 2
    let jackets := hats + hats / 5
    jeans - (scarves + jackets) = 3 := by
  sorry

end inventory_problem_l1656_165601


namespace circle_radius_l1656_165621

theorem circle_radius (x y : ℝ) : 
  x^2 + y^2 - 2*x + 4*y = 0 → ∃ (h k r : ℝ), r = Real.sqrt 5 ∧ (x - h)^2 + (y - k)^2 = r^2 :=
sorry

end circle_radius_l1656_165621


namespace find_t_l1656_165672

-- Define variables
variable (t : ℝ)

-- Define functions for hours worked and hourly rates
def my_hours : ℝ := t + 2
def my_rate : ℝ := 4*t - 4
def bob_hours : ℝ := 4*t - 7
def bob_rate : ℝ := t + 3

-- State the theorem
theorem find_t : 
  my_hours * my_rate = bob_hours * bob_rate + 3 → t = 10 := by
  sorry

end find_t_l1656_165672


namespace root_implies_expression_value_l1656_165680

theorem root_implies_expression_value (a : ℝ) : 
  (1^2 - 5*a*1 + a^2 = 0) → (3*a^2 - 15*a - 7 = -10) := by
  sorry

end root_implies_expression_value_l1656_165680


namespace quadratic_no_real_roots_l1656_165619

theorem quadratic_no_real_roots (k : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - k ≠ 0) → k < -1 := by
  sorry

end quadratic_no_real_roots_l1656_165619


namespace project_workers_needed_l1656_165640

/-- Represents a construction project with workers -/
structure Project where
  totalDays : ℕ
  elapsedDays : ℕ
  initialWorkers : ℕ
  completionRatio : ℚ
  
/-- Calculates the minimum number of workers needed to complete the project on schedule -/
def minWorkersNeeded (p : Project) : ℕ :=
  sorry

/-- The theorem stating the minimum number of workers needed for the specific project -/
theorem project_workers_needed :
  let p : Project := {
    totalDays := 40,
    elapsedDays := 10,
    initialWorkers := 10,
    completionRatio := 2/5
  }
  minWorkersNeeded p = 5 := by sorry

end project_workers_needed_l1656_165640


namespace factorization_equality_l1656_165604

theorem factorization_equality (x y : ℝ) :
  -12 * x * y^2 * (x + y) + 18 * x^2 * y * (x + y) = 6 * x * y * (x + y) * (3 * x - 2 * y) := by
  sorry

end factorization_equality_l1656_165604


namespace no_arithmetic_progression_roots_l1656_165688

theorem no_arithmetic_progression_roots (a : ℝ) : 
  ¬ ∃ (x d : ℝ), 
    (∀ k : Fin 4, 16 * (x + k * d)^4 - a * (x + k * d)^3 + (2*a + 17) * (x + k * d)^2 - a * (x + k * d) + 16 = 0) ∧
    (d ≠ 0) :=
by sorry

end no_arithmetic_progression_roots_l1656_165688


namespace point_coordinates_l1656_165697

/-- A point in the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance from a point to the x-axis -/
def distanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def distanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- The main theorem -/
theorem point_coordinates 
    (p : Point) 
    (h1 : isInSecondQuadrant p) 
    (h2 : distanceToXAxis p = 4) 
    (h3 : distanceToYAxis p = 5) : 
  p = Point.mk (-5) 4 := by
  sorry

end point_coordinates_l1656_165697


namespace geometric_sequence_fourth_term_l1656_165612

theorem geometric_sequence_fourth_term :
  ∀ (a : ℕ → ℝ),
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 3 →                            -- first term
  a 8 = 3888 →                         -- last term
  a 4 = 648 :=                         -- fourth term
by
  sorry

end geometric_sequence_fourth_term_l1656_165612


namespace total_arrangements_l1656_165667

/-- The number of ways to arrange 3 events in 4 venues with at most 2 events per venue -/
def arrangeEvents : ℕ := sorry

/-- The total number of arrangements is 60 -/
theorem total_arrangements : arrangeEvents = 60 := by sorry

end total_arrangements_l1656_165667


namespace florist_fertilizer_l1656_165634

def fertilizer_problem (daily_amount : ℕ) (regular_days : ℕ) (extra_amount : ℕ) : Prop :=
  let regular_total := daily_amount * regular_days
  let final_day_amount := daily_amount + extra_amount
  let total_amount := regular_total + final_day_amount
  total_amount = 45

theorem florist_fertilizer :
  fertilizer_problem 3 12 6 := by
  sorry

end florist_fertilizer_l1656_165634


namespace remaining_grass_area_l1656_165629

-- Define the plot and path characteristics
def plot_diameter : ℝ := 20
def path_width : ℝ := 4

-- Define the theorem
theorem remaining_grass_area :
  let plot_radius : ℝ := plot_diameter / 2
  let effective_radius : ℝ := plot_radius - path_width / 2
  (π * effective_radius^2 : ℝ) = 64 * π := by sorry

end remaining_grass_area_l1656_165629


namespace projection_magnitude_l1656_165636

def a : Fin 2 → ℝ := ![1, -1]
def b : Fin 2 → ℝ := ![2, -1]

theorem projection_magnitude :
  ‖((a + b) • a / (a • a)) • a‖ = (5 * Real.sqrt 2) / 2 := by
  sorry

end projection_magnitude_l1656_165636


namespace eraser_ratio_is_two_to_one_l1656_165616

-- Define the number of erasers for each person
def tanya_total : ℕ := 20
def hanna_total : ℕ := 4

-- Define the number of red erasers Tanya has
def tanya_red : ℕ := tanya_total / 2

-- Define Rachel's erasers in terms of Tanya's red erasers
def rachel_total : ℕ := tanya_red / 2 - 3

-- Define the ratio of Hanna's erasers to Rachel's erasers
def eraser_ratio : ℚ := hanna_total / rachel_total

-- Theorem to prove
theorem eraser_ratio_is_two_to_one :
  eraser_ratio = 2 := by sorry

end eraser_ratio_is_two_to_one_l1656_165616


namespace congruence_from_power_difference_l1656_165645

theorem congruence_from_power_difference (a b : ℕ+) (h : a^b.val - b^a.val = 1008) :
  a ≡ b [ZMOD 1008] := by
  sorry

end congruence_from_power_difference_l1656_165645


namespace light_reflection_l1656_165696

-- Define the points A and B
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (2, -1)

-- Define the y-axis
def y_axis (x : ℝ) : Prop := x = 0

-- Define a line passing through two points
def line_through_points (p q : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - p.2) * (q.1 - p.1) = (x - p.1) * (q.2 - p.2)

-- Define reflection of a point across the y-axis
def reflect_across_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- State the theorem
theorem light_reflection :
  ∃ (C : ℝ × ℝ),
    y_axis C.1 ∧
    line_through_points A C 1 1 ∧
    line_through_points (reflect_across_y_axis A) B C.1 C.2 ∧
    (∀ x y, line_through_points A C x y ↔ x - y + 1 = 0) ∧
    (∀ x y, line_through_points (reflect_across_y_axis A) B x y ↔ x + y - 1 = 0) :=
by sorry

end light_reflection_l1656_165696


namespace possible_lists_count_l1656_165685

/-- The number of balls in the bin -/
def num_balls : ℕ := 15

/-- The number of draws Joe makes -/
def num_draws : ℕ := 4

/-- The number of possible lists when drawing with replacement -/
def num_possible_lists : ℕ := num_balls ^ num_draws

/-- Theorem: The number of possible lists is 50625 -/
theorem possible_lists_count : num_possible_lists = 50625 := by
  sorry

end possible_lists_count_l1656_165685


namespace vector_b_magnitude_l1656_165682

def a : ℝ × ℝ := (1, -2)

theorem vector_b_magnitude (b : ℝ × ℝ) (h : 2 • a - b = (-1, 0)) : 
  ‖b‖ = 5 := by sorry

end vector_b_magnitude_l1656_165682


namespace contractor_male_workers_l1656_165602

/-- Represents the number of male workers employed by the contractor. -/
def male_workers : ℕ := sorry

/-- Represents the number of female workers employed by the contractor. -/
def female_workers : ℕ := 15

/-- Represents the number of child workers employed by the contractor. -/
def child_workers : ℕ := 5

/-- Represents the daily wage of a male worker in Rupees. -/
def male_wage : ℕ := 35

/-- Represents the daily wage of a female worker in Rupees. -/
def female_wage : ℕ := 20

/-- Represents the daily wage of a child worker in Rupees. -/
def child_wage : ℕ := 8

/-- Represents the average daily wage paid by the contractor in Rupees. -/
def average_wage : ℕ := 26

/-- Theorem stating that the number of male workers employed by the contractor is 20. -/
theorem contractor_male_workers :
  (male_workers * male_wage + female_workers * female_wage + child_workers * child_wage) /
  (male_workers + female_workers + child_workers) = average_wage →
  male_workers = 20 := by
  sorry

end contractor_male_workers_l1656_165602


namespace tree_height_calculation_l1656_165664

/-- Given a flagpole and a tree, calculate the height of the tree using similar triangles -/
theorem tree_height_calculation (flagpole_height flagpole_shadow tree_shadow : ℝ) 
  (h1 : flagpole_height = 4)
  (h2 : flagpole_shadow = 6)
  (h3 : tree_shadow = 12) :
  (flagpole_height / flagpole_shadow) * tree_shadow = 8 :=
by sorry

end tree_height_calculation_l1656_165664


namespace suit_price_calculation_l1656_165624

theorem suit_price_calculation (original_price : ℝ) : 
  (original_price * 1.2 * 0.8 = 144) → original_price = 150 := by
  sorry

end suit_price_calculation_l1656_165624


namespace power_sum_equality_l1656_165609

theorem power_sum_equality : (-2)^2004 + 3 * (-2)^2003 = -2^2003 := by sorry

end power_sum_equality_l1656_165609


namespace last_three_digits_of_7_to_210_l1656_165655

theorem last_three_digits_of_7_to_210 : 7^210 % 1000 = 599 := by
  sorry

end last_three_digits_of_7_to_210_l1656_165655


namespace min_value_reciprocal_sum_l1656_165639

theorem min_value_reciprocal_sum (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (1 / a + 9 / b) ≥ 8 :=
by sorry

end min_value_reciprocal_sum_l1656_165639


namespace rectangle_perimeter_l1656_165627

theorem rectangle_perimeter (z w : ℝ) (hz : z > 0) (hw : w > 0) (h : w < z) :
  let l := z - w
  2 * (l + w) = 2 * z :=
by sorry

end rectangle_perimeter_l1656_165627


namespace stickers_on_last_page_l1656_165666

def total_books : Nat := 10
def pages_per_book : Nat := 30
def initial_stickers_per_page : Nat := 5
def new_stickers_per_page : Nat := 8
def full_books_after_rearrange : Nat := 6
def full_pages_in_seventh_book : Nat := 25

theorem stickers_on_last_page :
  let total_stickers := total_books * pages_per_book * initial_stickers_per_page
  let stickers_in_full_books := full_books_after_rearrange * pages_per_book * new_stickers_per_page
  let remaining_stickers := total_stickers - stickers_in_full_books
  let stickers_in_full_pages_of_seventh_book := (remaining_stickers / new_stickers_per_page) * new_stickers_per_page
  remaining_stickers - stickers_in_full_pages_of_seventh_book = 4 := by
  sorry

end stickers_on_last_page_l1656_165666


namespace quadratic_equation_root_l1656_165699

theorem quadratic_equation_root (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + 3 * x - 119 = 0) → 
  (a * 7^2 + 3 * 7 - 119 = 0) → 
  a = 2 := by
sorry

end quadratic_equation_root_l1656_165699


namespace m_range_l1656_165638

-- Define propositions p and q
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 1 > m

def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (2 - m) * x + 1 < (2 - m) * y + 1

-- Define the theorem
theorem m_range (m : ℝ) (h1 : p m ∨ q m) (h2 : ¬(p m ∧ q m)) : 
  1 < m ∧ m < 2 := by
  sorry


end m_range_l1656_165638


namespace largest_multiple_of_12_answer_is_valid_l1656_165631

def is_valid_number (n : ℕ) : Prop :=
  ∃ (digits : List ℕ), 
    digits.length = 10 ∧ 
    digits.toFinset = Finset.range 10 ∧
    n = digits.foldl (λ acc d => acc * 10 + d) 0

def is_multiple_of_12 (n : ℕ) : Prop :=
  n % 12 = 0

theorem largest_multiple_of_12 :
  ∀ n : ℕ, 
    is_valid_number n → 
    is_multiple_of_12 n → 
    n ≤ 9876543120 :=
by sorry

theorem answer_is_valid :
  is_valid_number 9876543120 ∧ is_multiple_of_12 9876543120 :=
by sorry

end largest_multiple_of_12_answer_is_valid_l1656_165631


namespace min_weighings_is_three_l1656_165603

/-- Represents a collection of coins with two adjacent lighter coins. -/
structure CoinCollection where
  n : ℕ
  light_weight : ℕ
  heavy_weight : ℕ
  
/-- Represents a weighing operation on a subset of coins. -/
def Weighing (cc : CoinCollection) (subset : Finset ℕ) : ℕ := sorry

/-- The minimum number of weighings required to identify the two lighter coins. -/
def min_weighings (cc : CoinCollection) : ℕ := sorry

/-- Theorem stating that the minimum number of weighings is 3 for any valid coin collection. -/
theorem min_weighings_is_three (cc : CoinCollection) 
  (h1 : cc.n ≥ 2) 
  (h2 : cc.light_weight = 9) 
  (h3 : cc.heavy_weight = 10) :
  min_weighings cc = 3 := by sorry

end min_weighings_is_three_l1656_165603


namespace tan_double_angle_l1656_165656

theorem tan_double_angle (α : Real) (h : 3 * Real.cos α + Real.sin α = 0) :
  Real.tan (2 * α) = 3 / 4 := by
  sorry

end tan_double_angle_l1656_165656


namespace pet_store_problem_l1656_165684

def puppies_sold (initial_puppies : ℕ) (puppies_per_cage : ℕ) (cages_used : ℕ) : ℕ :=
  initial_puppies - (puppies_per_cage * cages_used)

theorem pet_store_problem :
  puppies_sold 45 2 3 = 39 := by
  sorry

end pet_store_problem_l1656_165684


namespace irrational_sum_product_theorem_l1656_165623

theorem irrational_sum_product_theorem (a : ℝ) (h : Irrational a) :
  ∃ (b b' : ℝ), Irrational b ∧ Irrational b' ∧
    (¬ Irrational (a + b)) ∧
    (¬ Irrational (a * b')) ∧
    (Irrational (a * b)) ∧
    (Irrational (a + b')) :=
by sorry

end irrational_sum_product_theorem_l1656_165623


namespace grass_cutting_cost_l1656_165652

/-- The cost of cutting grass once, given specific growth and cost conditions --/
theorem grass_cutting_cost
  (initial_height : ℝ)
  (growth_rate : ℝ)
  (cut_threshold : ℝ)
  (annual_cost : ℝ)
  (h1 : initial_height = 2)
  (h2 : growth_rate = 0.5)
  (h3 : cut_threshold = 4)
  (h4 : annual_cost = 300)
  : (annual_cost / (12 / ((cut_threshold - initial_height) / growth_rate))) = 100 := by
  sorry

end grass_cutting_cost_l1656_165652


namespace stratified_sampling_size_l1656_165647

/-- Represents the sample sizes for three districts -/
structure DistrictSamples where
  d1 : ℕ
  d2 : ℕ
  d3 : ℕ

/-- Given a population divided into three districts with a ratio of 2:3:5,
    and a maximum sample size of 60 for any district,
    prove that the total sample size is 120. -/
theorem stratified_sampling_size :
  ∀ (s : DistrictSamples),
  (s.d1 : ℚ) / 2 = s.d2 / 3 ∧
  (s.d1 : ℚ) / 2 = s.d3 / 5 ∧
  s.d3 ≤ 60 ∧
  s.d3 = 60 →
  s.d1 + s.d2 + s.d3 = 120 := by
  sorry


end stratified_sampling_size_l1656_165647


namespace greatest_common_remainder_l1656_165663

theorem greatest_common_remainder (a b c : ℕ) (h1 : a = 41) (h2 : b = 71) (h3 : c = 113) :
  ∃ (d : ℕ), d > 0 ∧ 
  (∃ (r : ℕ), a % d = r ∧ b % d = r ∧ c % d = r) ∧
  (∀ (k : ℕ), k > 0 → (∃ (s : ℕ), a % k = s ∧ b % k = s ∧ c % k = s) → k ≤ d) ∧
  d = Nat.gcd (b - a) (Nat.gcd (c - b) (c - a)) :=
sorry

end greatest_common_remainder_l1656_165663


namespace combinatorial_equations_solutions_l1656_165698

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Falling factorial -/
def falling_factorial (n k : ℕ) : ℕ := sorry

theorem combinatorial_equations_solutions :
  (∃ x : ℕ, (binomial 9 x = binomial 9 (2*x - 3)) ∧ (x = 3 ∨ x = 4)) ∧
  (∃ x : ℕ, x ≤ 8 ∧ falling_factorial 8 x = 6 * falling_factorial 8 (x - 2) ∧ x = 7) :=
sorry

end combinatorial_equations_solutions_l1656_165698


namespace sqrt_of_one_plus_three_l1656_165676

theorem sqrt_of_one_plus_three : Real.sqrt (1 + 3) = 2 := by
  sorry

end sqrt_of_one_plus_three_l1656_165676


namespace newspaper_coupon_free_tickets_l1656_165646

/-- Represents the amusement park scenario --/
structure AmusementPark where
  ferris_wheel_cost : ℝ
  roller_coaster_cost : ℝ
  multiple_ride_discount : ℝ
  tickets_bought : ℝ

/-- Calculates the number of free tickets from the newspaper coupon --/
def free_tickets (park : AmusementPark) : ℝ :=
  park.ferris_wheel_cost + park.roller_coaster_cost - park.multiple_ride_discount - park.tickets_bought

/-- Theorem stating that the number of free tickets is 1 given the specific conditions --/
theorem newspaper_coupon_free_tickets :
  ∀ (park : AmusementPark),
    park.ferris_wheel_cost = 2 →
    park.roller_coaster_cost = 7 →
    park.multiple_ride_discount = 1 →
    park.tickets_bought = 7 →
    free_tickets park = 1 :=
by
  sorry


end newspaper_coupon_free_tickets_l1656_165646


namespace milkman_profit_l1656_165693

/-- Calculates the profit of a milkman selling three mixtures of milk and water -/
theorem milkman_profit (total_milk : ℝ) (total_water : ℝ) 
  (milk1 : ℝ) (water1 : ℝ) (price1 : ℝ)
  (milk2 : ℝ) (water2 : ℝ) (price2 : ℝ)
  (water3 : ℝ) (price3 : ℝ)
  (milk_cost : ℝ) :
  total_milk = 80 ∧
  total_water = 20 ∧
  milk1 = 40 ∧
  water1 = 5 ∧
  price1 = 19 ∧
  milk2 = 25 ∧
  water2 = 10 ∧
  price2 = 18 ∧
  water3 = 5 ∧
  price3 = 21 ∧
  milk_cost = 22 →
  let milk3 := total_milk - milk1 - milk2
  let revenue1 := (milk1 + water1) * price1
  let revenue2 := (milk2 + water2) * price2
  let revenue3 := (milk3 + water3) * price3
  let total_revenue := revenue1 + revenue2 + revenue3
  let total_cost := total_milk * milk_cost
  let profit := total_revenue - total_cost
  profit = 50 := by
sorry

end milkman_profit_l1656_165693


namespace salary_problem_l1656_165671

/-- The average monthly salary of employees in an organization -/
def average_salary (num_employees : ℕ) (total_salary : ℕ) : ℚ :=
  total_salary / num_employees

/-- The problem statement -/
theorem salary_problem (initial_total_salary : ℕ) :
  let num_employees : ℕ := 20
  let manager_salary : ℕ := 3300
  let new_average : ℚ := average_salary (num_employees + 1) (initial_total_salary + manager_salary)
  let initial_average : ℚ := average_salary num_employees initial_total_salary
  new_average = initial_average + 100 →
  initial_average = 1200 := by
  sorry

end salary_problem_l1656_165671


namespace equation_solution_l1656_165692

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), (x₁ = 4 + 3 * Real.sqrt 19) ∧ (x₂ = 4 - 3 * Real.sqrt 19) ∧
  (∀ x : ℝ, (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) = 1 / 8) ↔ 
  (x = x₁ ∨ x = x₂)) := by
  sorry

end equation_solution_l1656_165692


namespace junior_score_l1656_165675

theorem junior_score (n : ℝ) (junior_ratio : ℝ) (senior_ratio : ℝ) 
  (class_avg : ℝ) (senior_avg : ℝ) (junior_score : ℝ) :
  junior_ratio = 0.2 →
  senior_ratio = 0.8 →
  junior_ratio + senior_ratio = 1 →
  class_avg = 75 →
  senior_avg = 72 →
  class_avg * n = senior_avg * (senior_ratio * n) + junior_score * (junior_ratio * n) →
  junior_score = 87 := by
sorry

end junior_score_l1656_165675


namespace unmeasurable_weights_theorem_l1656_165659

def available_weights : List Nat := [1, 2, 3, 8, 16, 32]

def is_measurable (n : Nat) (weights : List Nat) : Prop :=
  ∃ (subset : List Nat), subset.Sublist weights ∧ subset.sum = n

def unmeasurable_weights : Set Nat :=
  {n | n ≤ 60 ∧ ¬(is_measurable n available_weights)}

theorem unmeasurable_weights_theorem :
  unmeasurable_weights = {7, 15, 23, 31, 39, 47, 55} := by
  sorry

end unmeasurable_weights_theorem_l1656_165659


namespace max_difference_of_constrained_integers_l1656_165610

theorem max_difference_of_constrained_integers : 
  ∃ (P Q : ℤ), 
    (∃ (x : ℤ), x^2 ≤ 729 ∧ 729 ≤ -x^3 ∧ (x = P ∨ x = Q)) ∧
    (∀ (R S : ℤ), (∃ (y : ℤ), y^2 ≤ 729 ∧ 729 ≤ -y^3 ∧ (y = R ∨ y = S)) → 
      10 * (P - Q) ≥ 10 * (R - S)) ∧
    10 * (P - Q) = 180 :=
by sorry

end max_difference_of_constrained_integers_l1656_165610


namespace right_triangle_acute_angles_l1656_165648

/-- Represents a right triangle with acute angles in the ratio 5:4 -/
structure RightTriangle where
  /-- First acute angle in degrees -/
  angle1 : ℝ
  /-- Second acute angle in degrees -/
  angle2 : ℝ
  /-- The triangle is a right triangle -/
  is_right_triangle : angle1 + angle2 = 90
  /-- The ratio of acute angles is 5:4 -/
  angle_ratio : angle1 / angle2 = 5 / 4

/-- Theorem: In a right triangle where the ratio of acute angles is 5:4,
    the measures of these angles are 50° and 40° -/
theorem right_triangle_acute_angles (t : RightTriangle) : 
  t.angle1 = 50 ∧ t.angle2 = 40 := by
  sorry


end right_triangle_acute_angles_l1656_165648


namespace smallest_solution_sum_is_five_l1656_165657

/-- The sum of divisors function for numbers of the form 2^i * 3^j * 5^k -/
def sum_of_divisors (i j k : ℕ) : ℕ :=
  (2^(i+1) - 1) * ((3^(j+1) - 1)/2) * ((5^(k+1) - 1)/4)

/-- Predicate to check if (i, j, k) is a valid solution -/
def is_valid_solution (i j k : ℕ) : Prop :=
  sum_of_divisors i j k = 360

/-- Predicate to check if (i, j, k) is the smallest valid solution -/
def is_smallest_solution (i j k : ℕ) : Prop :=
  is_valid_solution i j k ∧
  ∀ i' j' k', is_valid_solution i' j' k' → i + j + k ≤ i' + j' + k'

/-- The main theorem: the smallest solution sums to 5 -/
theorem smallest_solution_sum_is_five :
  ∃ i j k, is_smallest_solution i j k ∧ i + j + k = 5 := by sorry

#check smallest_solution_sum_is_five

end smallest_solution_sum_is_five_l1656_165657


namespace rod_division_theorem_l1656_165689

/-- Represents a rod divided into equal parts -/
structure DividedRod where
  length : ℕ
  divisions : List ℕ

/-- Calculates the total number of segments in a divided rod -/
def totalSegments (rod : DividedRod) : ℕ := sorry

/-- Calculates the length of the shortest segment in a divided rod -/
def shortestSegment (rod : DividedRod) : ℚ := sorry

/-- Theorem about a specific rod division -/
theorem rod_division_theorem (k : ℕ) :
  let rod := DividedRod.mk (72 * k) [8, 12, 18]
  totalSegments rod = 28 ∧ shortestSegment rod = 1 / 72 := by sorry

end rod_division_theorem_l1656_165689


namespace tan_300_degrees_l1656_165695

theorem tan_300_degrees : Real.tan (300 * π / 180) = -Real.sqrt 3 := by
  sorry

end tan_300_degrees_l1656_165695
