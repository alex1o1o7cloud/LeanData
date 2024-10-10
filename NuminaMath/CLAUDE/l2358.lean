import Mathlib

namespace sqrt_of_sqrt_plus_one_over_two_is_integer_l2358_235814

theorem sqrt_of_sqrt_plus_one_over_two_is_integer (n : ℕ+) 
  (h : ∃ (x : ℕ+), x^2 = 12 * n^2 + 1) :
  ∃ (q : ℕ+), q^2 = (Nat.sqrt (12 * n^2 + 1) + 1) / 2 := by
sorry

end sqrt_of_sqrt_plus_one_over_two_is_integer_l2358_235814


namespace abs_greater_than_negative_l2358_235871

theorem abs_greater_than_negative (a b : ℝ) (h : a < b ∧ b < 0) : |a| > -b := by
  sorry

end abs_greater_than_negative_l2358_235871


namespace f_expression_sum_f_expression_l2358_235846

/-- A linear function f satisfying specific conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The condition that f(8) = 15 -/
axiom f_8 : f 8 = 15

/-- The condition that f(2), f(5), f(4) form a geometric sequence -/
axiom f_geometric : ∃ (r : ℝ), f 5 = r * f 2 ∧ f 4 = r * f 5

/-- Theorem stating that f(x) = 4x - 17 -/
theorem f_expression : ∀ x, f x = 4 * x - 17 := by sorry

/-- Function to calculate the sum of f(2) + f(4) + ... + f(2n) -/
def sum_f (n : ℕ) : ℝ := sorry

/-- Theorem stating the sum of f(2) + f(4) + ... + f(2n) = 4n^2 - 13n -/
theorem sum_f_expression : ∀ n, sum_f n = 4 * n^2 - 13 * n := by sorry

end f_expression_sum_f_expression_l2358_235846


namespace forward_journey_time_l2358_235818

/-- Represents the journey of a car -/
structure Journey where
  distance : ℝ
  forwardTime : ℝ
  returnTime : ℝ
  speedIncrease : ℝ

/-- Theorem stating the conditions and the result to be proved -/
theorem forward_journey_time (j : Journey)
  (h1 : j.distance = 210)
  (h2 : j.returnTime = 5)
  (h3 : j.speedIncrease = 12)
  (h4 : j.distance = j.distance / j.forwardTime * j.returnTime + j.speedIncrease * j.returnTime) :
  j.forwardTime = 7 := by
  sorry

end forward_journey_time_l2358_235818


namespace no_prime_roots_for_specific_quadratic_l2358_235879

theorem no_prime_roots_for_specific_quadratic :
  ¬∃ (k : ℤ), ∃ (p q : ℕ), 
    Prime p ∧ Prime q ∧ 
    (p : ℤ) + q = 71 ∧
    (p : ℤ) * q = k ∧
    p ≠ q :=
sorry

end no_prime_roots_for_specific_quadratic_l2358_235879


namespace division_equation_problem_l2358_235898

theorem division_equation_problem (A B C : ℕ) : 
  (∃ (q : ℕ), A = B * q + 8) → -- A ÷ B = C with remainder 8
  (A + B + C = 2994) →         -- Sum condition
  (A = 8 ∨ A = 2864) :=        -- Conclusion
by
  sorry

end division_equation_problem_l2358_235898


namespace solve_jewelry_problem_l2358_235838

/-- Represents the jewelry store inventory problem -/
def jewelry_problem (necklace_capacity : ℕ) (current_necklaces : ℕ) 
  (ring_capacity : ℕ) (current_rings : ℕ) (bracelet_capacity : ℕ) 
  (necklace_cost : ℕ) (ring_cost : ℕ) (bracelet_cost : ℕ) (total_cost : ℕ) : Prop :=
  ∃ (current_bracelets : ℕ),
    necklace_capacity = 12 ∧
    current_necklaces = 5 ∧
    ring_capacity = 30 ∧
    current_rings = 18 ∧
    bracelet_capacity = 15 ∧
    necklace_cost = 4 ∧
    ring_cost = 10 ∧
    bracelet_cost = 5 ∧
    total_cost = 183 ∧
    (necklace_capacity - current_necklaces) * necklace_cost + 
    (ring_capacity - current_rings) * ring_cost + 
    (bracelet_capacity - current_bracelets) * bracelet_cost = total_cost ∧
    current_bracelets = 8

theorem solve_jewelry_problem :
  jewelry_problem 12 5 30 18 15 4 10 5 183 :=
sorry

end solve_jewelry_problem_l2358_235838


namespace valid_outfit_choices_eq_239_l2358_235840

/-- Represents the number of valid outfit choices given the specified conditions -/
def valid_outfit_choices : ℕ := by
  -- Define the number of shirts, pants, and hats
  let num_shirts : ℕ := 6
  let num_pants : ℕ := 7
  let num_hats : ℕ := 6
  
  -- Define the number of colors
  let num_colors : ℕ := 6
  
  -- Calculate total number of outfits without restrictions
  let total_outfits : ℕ := num_shirts * num_pants * num_hats
  
  -- Calculate number of outfits with all items the same color
  let all_same_color : ℕ := num_colors
  
  -- Calculate number of outfits with shirt and pants the same color
  let shirt_pants_same : ℕ := num_colors + 1
  
  -- Calculate the number of valid outfits
  exact total_outfits - all_same_color - shirt_pants_same

/-- Theorem stating that the number of valid outfit choices is 239 -/
theorem valid_outfit_choices_eq_239 : valid_outfit_choices = 239 := by
  sorry

end valid_outfit_choices_eq_239_l2358_235840


namespace money_redistribution_l2358_235873

/-- Represents the amount of money each person has -/
structure Money where
  amy : ℝ
  jan : ℝ
  toy : ℝ
  kim : ℝ

/-- Represents the redistribution rules -/
def redistribute (m : Money) : Money :=
  let step1 := Money.mk m.amy m.jan m.toy m.kim -- Kim equalizes others
  let step2 := Money.mk m.amy m.jan m.toy m.kim -- Amy doubles Jan and Toy
  let step3 := Money.mk m.amy m.jan m.toy m.kim -- Jan doubles Amy and Toy
  let step4 := Money.mk m.amy m.jan m.toy m.kim -- Toy doubles others
  step4

theorem money_redistribution (initial final : Money) :
  initial.toy = 48 →
  final.toy = 48 →
  final = redistribute initial →
  initial.amy + initial.jan + initial.toy + initial.kim = 192 :=
by
  sorry

#check money_redistribution

end money_redistribution_l2358_235873


namespace greenToBlueRatioIs2To3_l2358_235824

/-- Represents a box of crayons with different colors -/
structure CrayonBox where
  total : ℕ
  red : ℕ
  blue : ℕ
  pink : ℕ
  green : ℕ
  h1 : total = red + blue + pink + green

/-- Calculates the ratio of green crayons to blue crayons -/
def greenToBlueRatio (box : CrayonBox) : Rat :=
  box.green / box.blue

/-- Theorem stating that for the given crayon box, the ratio of green to blue crayons is 2:3 -/
theorem greenToBlueRatioIs2To3 (box : CrayonBox) 
    (h2 : box.total = 24)
    (h3 : box.red = 8)
    (h4 : box.blue = 6)
    (h5 : box.pink = 6) :
    greenToBlueRatio box = 2 / 3 := by
  sorry

#eval greenToBlueRatio { total := 24, red := 8, blue := 6, pink := 6, green := 4, h1 := rfl }

end greenToBlueRatioIs2To3_l2358_235824


namespace quadratic_roots_sum_of_squares_l2358_235829

theorem quadratic_roots_sum_of_squares (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 - m*x + (2*m - 1) = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁^2 + x₂^2 = 7 →
  m = -1 := by
sorry

end quadratic_roots_sum_of_squares_l2358_235829


namespace assignment_result_l2358_235861

def assignment_sequence (initial_a : ℕ) : ℕ :=
  let a₁ := initial_a
  let a₂ := a₁ + 1
  a₂

theorem assignment_result : assignment_sequence 3 = 4 := by
  sorry

end assignment_result_l2358_235861


namespace negative_one_power_equality_l2358_235881

theorem negative_one_power_equality : (-1 : ℤ)^3 = (-1 : ℤ)^2023 := by
  sorry

end negative_one_power_equality_l2358_235881


namespace cube_distance_to_plane_l2358_235816

theorem cube_distance_to_plane (cube_side : ℝ) (h1 h2 h3 : ℝ) :
  cube_side = 10 →
  h1 = 10 ∧ h2 = 11 ∧ h3 = 12 →
  ∃ (d : ℝ), d = (33 - Real.sqrt 294) / 3 ∧
    d = min h1 (min h2 h3) - (cube_side - Real.sqrt (cube_side^2 + (h2 - h1)^2 + (h3 - h1)^2)) :=
by sorry

end cube_distance_to_plane_l2358_235816


namespace inequality_proof_l2358_235849

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a^2 + a*b + b^2) + Real.sqrt (a^2 + a*c + c^2) ≥ 
  4 * Real.sqrt ((a*b / (a+b))^2 + (a*b / (a+b)) * (a*c / (a+c)) + (a*c / (a+c))^2) := by
  sorry

end inequality_proof_l2358_235849


namespace f_neg_l2358_235831

-- Define an even function f
def f : ℝ → ℝ := sorry

-- Define the property of an even function
axiom f_even : ∀ x : ℝ, f x = f (-x)

-- Define f for positive x
axiom f_pos : ∀ x : ℝ, x > 0 → f x = x^2 - 2*x

-- Theorem to prove
theorem f_neg : ∀ x : ℝ, x < 0 → f x = x^2 + 2*x := by sorry

end f_neg_l2358_235831


namespace ezekiel_shoe_pairs_l2358_235859

/-- The number of shoes in a pair -/
def shoes_per_pair : ℕ := 2

/-- The total number of new shoes Ezekiel has -/
def total_shoes : ℕ := 6

/-- The number of pairs of shoes Ezekiel bought -/
def pairs_bought : ℕ := total_shoes / shoes_per_pair

theorem ezekiel_shoe_pairs : pairs_bought = 3 := by
  sorry

end ezekiel_shoe_pairs_l2358_235859


namespace sam_initial_pennies_l2358_235825

/-- The number of pennies Sam found -/
def pennies_found : ℕ := 93

/-- The total number of pennies Sam has now -/
def total_pennies : ℕ := 191

/-- The initial number of pennies Sam had -/
def initial_pennies : ℕ := total_pennies - pennies_found

theorem sam_initial_pennies : initial_pennies = 98 := by
  sorry

end sam_initial_pennies_l2358_235825


namespace equation_solution_l2358_235877

theorem equation_solution : ∃ n : ℝ, 0.03 * n + 0.08 * (20 + n) = 12.6 ∧ n = 100 := by
  sorry

end equation_solution_l2358_235877


namespace max_ab_value_l2358_235813

/-- A function f with a parameter a and b -/
def f (a b x : ℝ) : ℝ := 4 * x^3 - a * x^2 - 2 * b * x + 2

/-- The derivative of f with respect to x -/
def f' (a b x : ℝ) : ℝ := 12 * x^2 - 2 * a * x - 2 * b

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_extremum : f' a b 1 = 0) : 
  ab ≤ 9 ∧ ∃ a b, a > 0 ∧ b > 0 ∧ f' a b 1 = 0 ∧ a * b = 9 := by
  sorry

end max_ab_value_l2358_235813


namespace millet_exceeds_half_on_day_three_l2358_235842

/-- Represents the proportion of millet in the feeder on a given day -/
def milletProportion (day : ℕ) : ℝ :=
  0.4 * (1 - (0.5 ^ day))

/-- The day when millet first exceeds half of the seeds -/
def milletExceedsHalfDay : ℕ :=
  3

theorem millet_exceeds_half_on_day_three :
  milletProportion milletExceedsHalfDay > 0.5 ∧
  ∀ d : ℕ, d < milletExceedsHalfDay → milletProportion d ≤ 0.5 :=
by sorry

end millet_exceeds_half_on_day_three_l2358_235842


namespace odd_function_symmetric_behavior_l2358_235882

def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def IsIncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def HasMinimumOn (f : ℝ → ℝ) (a b m : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → m ≤ f x) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = m)

def HasMaximumOn (f : ℝ → ℝ) (a b M : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → f x ≤ M) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = M)

theorem odd_function_symmetric_behavior (f : ℝ → ℝ) :
  IsOdd f →
  IsIncreasingOn f 3 7 →
  HasMinimumOn f 3 7 5 →
  IsIncreasingOn f (-7) (-3) ∧ HasMaximumOn f (-7) (-3) (-5) := by
  sorry

end odd_function_symmetric_behavior_l2358_235882


namespace mary_warm_hours_l2358_235874

/-- The number of sticks of wood produced by chopping up a chair -/
def sticks_per_chair : ℕ := 6

/-- The number of sticks of wood produced by chopping up a table -/
def sticks_per_table : ℕ := 9

/-- The number of sticks of wood produced by chopping up a stool -/
def sticks_per_stool : ℕ := 2

/-- The number of sticks of wood Mary needs to burn per hour to stay warm -/
def sticks_per_hour : ℕ := 5

/-- The number of chairs Mary chops up -/
def chairs_chopped : ℕ := 18

/-- The number of tables Mary chops up -/
def tables_chopped : ℕ := 6

/-- The number of stools Mary chops up -/
def stools_chopped : ℕ := 4

/-- Theorem stating how many hours Mary can keep warm -/
theorem mary_warm_hours : 
  (chairs_chopped * sticks_per_chair + 
   tables_chopped * sticks_per_table + 
   stools_chopped * sticks_per_stool) / sticks_per_hour = 34 := by
  sorry

end mary_warm_hours_l2358_235874


namespace team_selection_count_l2358_235866

/-- The number of ways to select a team of 3 people from 3 male and 3 female teachers,
    with both genders included -/
def select_team (male_teachers female_teachers team_size : ℕ) : ℕ :=
  (male_teachers.choose 2 * female_teachers.choose 1) +
  (male_teachers.choose 1 * female_teachers.choose 2)

/-- Theorem: There are 18 ways to select a team of 3 from 3 male and 3 female teachers,
    with both genders included -/
theorem team_selection_count :
  select_team 3 3 3 = 18 := by
  sorry

end team_selection_count_l2358_235866


namespace largest_three_digit_multiple_of_8_with_digit_sum_24_l2358_235807

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  sorry

/-- A function that checks if a number is a three-digit number -/
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem largest_three_digit_multiple_of_8_with_digit_sum_24 :
  ∀ n : ℕ, is_three_digit n → n.mod 8 = 0 → digit_sum n = 24 → n ≤ 888 :=
sorry

end largest_three_digit_multiple_of_8_with_digit_sum_24_l2358_235807


namespace min_translation_value_l2358_235835

theorem min_translation_value (f g : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = Real.sin (2 * x) + Real.cos (2 * x)) →
  (∀ x, g x = f (x - m)) →
  (m > 0) →
  (∀ x, g (-π/3) ≤ g x) →
  ∃ k : ℤ, m = k * π + π/24 ∧ 
  (∀ m' : ℝ, m' > 0 → (∀ x, g (-π/3) ≤ g x) → m' ≥ π/24) :=
sorry

end min_translation_value_l2358_235835


namespace group_size_theorem_l2358_235863

theorem group_size_theorem (n : ℕ) (k : ℕ) : 
  (k * (n - 1) * n = 440 ∧ n > 0 ∧ k > 0) → (n = 5 ∨ n = 11) :=
sorry

end group_size_theorem_l2358_235863


namespace cube_face_perimeter_l2358_235811

theorem cube_face_perimeter (volume : ℝ) (perimeter : ℝ) : 
  volume = 125 → perimeter = 20 := by
  sorry

end cube_face_perimeter_l2358_235811


namespace min_value_quadratic_l2358_235823

theorem min_value_quadratic :
  (∀ x : ℝ, x^2 + 6*x ≥ -9) ∧ (∃ x : ℝ, x^2 + 6*x = -9) := by
  sorry

end min_value_quadratic_l2358_235823


namespace value_of_C_l2358_235885

theorem value_of_C : ∃ C : ℝ, (4 * C + 3 = 25) ∧ (C = 5.5) := by sorry

end value_of_C_l2358_235885


namespace difference_divisible_by_nine_l2358_235834

def reverse (n : ℕ) : ℕ := sorry

theorem difference_divisible_by_nine (n : ℕ) : 
  ∃ k : ℤ, n - reverse n = 9 * k := by sorry

end difference_divisible_by_nine_l2358_235834


namespace jill_phone_time_l2358_235855

def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem jill_phone_time : geometric_sum 5 2 5 = 155 := by
  sorry

end jill_phone_time_l2358_235855


namespace greatest_multiple_of_four_cubed_less_than_5000_l2358_235851

theorem greatest_multiple_of_four_cubed_less_than_5000 :
  ∀ x : ℕ, x > 0 → x % 4 = 0 → x^3 < 5000 → x ≤ 16 :=
by
  sorry

end greatest_multiple_of_four_cubed_less_than_5000_l2358_235851


namespace moe_has_least_money_l2358_235828

-- Define the set of people
inductive Person : Type
  | Bo | Coe | Flo | Jo | Moe | Zoe

-- Define the "has more money than" relation
def has_more_money (p1 p2 : Person) : Prop := sorry

-- Define the conditions
axiom different_amounts : ∀ (p1 p2 : Person), p1 ≠ p2 → has_more_money p1 p2 ∨ has_more_money p2 p1
axiom flo_bo_zoe : has_more_money Person.Flo Person.Bo ∧ has_more_money Person.Zoe Person.Flo
axiom zoe_coe : has_more_money Person.Zoe Person.Coe
axiom bo_coe_moe : has_more_money Person.Bo Person.Moe ∧ has_more_money Person.Coe Person.Moe
axiom jo_moe_zoe : has_more_money Person.Jo Person.Moe ∧ has_more_money Person.Zoe Person.Jo

-- Define the "has least money" property
def has_least_money (p : Person) : Prop :=
  ∀ (other : Person), other ≠ p → has_more_money other p

-- Theorem statement
theorem moe_has_least_money : has_least_money Person.Moe := by
  sorry

end moe_has_least_money_l2358_235828


namespace roses_recipients_l2358_235869

/-- Given Ricky's initial number of roses, the number of roses stolen, and the number of roses
    per person, calculate the number of people who will receive roses. -/
def number_of_recipients (initial_roses : ℕ) (stolen_roses : ℕ) (roses_per_person : ℕ) : ℕ :=
  (initial_roses - stolen_roses) / roses_per_person

/-- Theorem stating that given the specific values in the problem, 
    the number of people who will receive roses is 9. -/
theorem roses_recipients : 
  number_of_recipients 40 4 4 = 9 := by
  sorry

end roses_recipients_l2358_235869


namespace prob_at_least_four_girls_l2358_235847

-- Define the number of children
def num_children : ℕ := 6

-- Define the probability of a child being a girl
def prob_girl : ℚ := 1/2

-- Define the function to calculate the probability of at least k girls out of n children
def prob_at_least_k_girls (n k : ℕ) : ℚ :=
  sorry

theorem prob_at_least_four_girls :
  prob_at_least_k_girls num_children 4 = 11/32 :=
sorry

end prob_at_least_four_girls_l2358_235847


namespace quadratic_equation_distinct_roots_l2358_235858

theorem quadratic_equation_distinct_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    (k - 1) * x₁^2 - 2 * x₁ + 3 = 0 ∧
    (k - 1) * x₂^2 - 2 * x₂ + 3 = 0) ↔
  (k < 4/3 ∧ k ≠ 1) :=
by sorry

end quadratic_equation_distinct_roots_l2358_235858


namespace room_length_proof_l2358_235839

/-- Given a room with dimensions L * 15 * 12 feet, prove that L = 25 feet
    based on the whitewashing cost and room features. -/
theorem room_length_proof (L : ℝ) : 
  L * 15 * 12 > 0 →  -- room has positive volume
  (3 : ℝ) * (2 * (L * 12 + 15 * 12) - (6 * 3 + 3 * 4 * 3)) = 2718 →
  L = 25 := by sorry

end room_length_proof_l2358_235839


namespace point_inside_circle_l2358_235893

theorem point_inside_circle (a b : ℝ) : 
  a ≠ b → 
  a^2 - a - Real.sqrt 2 = 0 → 
  b^2 - b - Real.sqrt 2 = 0 → 
  a^2 + b^2 < 8 := by
sorry

end point_inside_circle_l2358_235893


namespace coffee_shop_weekly_production_l2358_235848

/-- A coffee shop that brews a certain number of coffee cups per day -/
structure CoffeeShop where
  weekday_cups_per_hour : ℕ
  weekend_total_cups : ℕ
  hours_open_per_day : ℕ

/-- Calculate the total number of coffee cups brewed in one week -/
def weekly_coffee_cups (shop : CoffeeShop) : ℕ :=
  let weekday_cups := shop.weekday_cups_per_hour * shop.hours_open_per_day * 5
  let weekend_cups := shop.weekend_total_cups
  weekday_cups + weekend_cups

/-- Theorem stating that a coffee shop with given parameters brews 370 cups in a week -/
theorem coffee_shop_weekly_production :
  ∀ (shop : CoffeeShop),
    shop.weekday_cups_per_hour = 10 →
    shop.weekend_total_cups = 120 →
    shop.hours_open_per_day = 5 →
    weekly_coffee_cups shop = 370 :=
by
  sorry

end coffee_shop_weekly_production_l2358_235848


namespace traffic_light_problem_l2358_235890

/-- A sequence of independent events with a fixed probability -/
structure EventSequence where
  n : ℕ  -- number of events
  p : ℝ  -- probability of each event occurring
  indep : Bool  -- events are independent

/-- The probability mass function for a binomial distribution -/
def binomial_pmf (k : ℕ) (es : EventSequence) : ℝ :=
  (es.n.choose k) * (es.p ^ k) * ((1 - es.p) ^ (es.n - k))

/-- The expected value of a binomial distribution -/
def binomial_expectation (es : EventSequence) : ℝ :=
  es.n * es.p

/-- The variance of a binomial distribution -/
def binomial_variance (es : EventSequence) : ℝ :=
  es.n * es.p * (1 - es.p)

theorem traffic_light_problem (es : EventSequence) 
  (h1 : es.n = 6) (h2 : es.p = 1/3) (h3 : es.indep = true) :
  (binomial_pmf 1 {n := 3, p := 1/3, indep := true} = 4/27) ∧
  (binomial_expectation es = 2) ∧
  (binomial_variance es = 4/3) := by
  sorry

end traffic_light_problem_l2358_235890


namespace peregrine_falcon_problem_l2358_235833

/-- The percentage of pigeons eaten by peregrines -/
def percentage_eaten (initial_pigeons : ℕ) (chicks_per_pigeon : ℕ) (remaining_pigeons : ℕ) : ℚ :=
  let total_pigeons := initial_pigeons + initial_pigeons * chicks_per_pigeon
  let eaten_pigeons := total_pigeons - remaining_pigeons
  (eaten_pigeons : ℚ) / (total_pigeons : ℚ) * 100

theorem peregrine_falcon_problem :
  percentage_eaten 40 6 196 = 30 := by
  sorry

end peregrine_falcon_problem_l2358_235833


namespace max_regions_four_lines_l2358_235867

/-- The maximum number of regions into which a plane can be divided using n straight lines -/
def L (n : ℕ) : ℕ :=
  n * (n + 1) / 2 + 1

/-- The theorem stating that 4 straight lines can divide a plane into at most 11 regions -/
theorem max_regions_four_lines : L 4 = 11 := by
  sorry

end max_regions_four_lines_l2358_235867


namespace pi_is_max_l2358_235802

theorem pi_is_max : ∀ (π : ℝ), π > 0 → (1 / 2023 : ℝ) > 0 → -2 * π < 0 →
  max (max (max 0 π) (1 / 2023)) (-2 * π) = π :=
by sorry

end pi_is_max_l2358_235802


namespace dagger_example_l2358_235827

def dagger (m n p q : ℚ) : ℚ := (m + n) * (p + q) * (q / n)

theorem dagger_example : dagger (5/9) (7/4) = 616/9 := by
  sorry

end dagger_example_l2358_235827


namespace photo_arrangements_l2358_235820

/-- The number of male students -/
def num_male : ℕ := 5

/-- The number of female students -/
def num_female : ℕ := 2

/-- The total number of students -/
def total_students : ℕ := num_male + num_female

/-- Calculates the number of arrangements with female student A at one end -/
def arrangements_a_at_end : ℕ := sorry

/-- Calculates the number of arrangements with both female students not at the ends -/
def arrangements_females_not_at_ends : ℕ := sorry

/-- Calculates the number of arrangements with the two female students not adjacent -/
def arrangements_females_not_adjacent : ℕ := sorry

/-- Calculates the number of arrangements with female student A on the right side of female student B -/
def arrangements_a_right_of_b : ℕ := sorry

theorem photo_arrangements :
  arrangements_a_at_end = 1440 ∧
  arrangements_females_not_at_ends = 2400 ∧
  arrangements_females_not_adjacent = 3600 ∧
  arrangements_a_right_of_b = 2520 := by sorry

end photo_arrangements_l2358_235820


namespace q_polynomial_expression_l2358_235895

theorem q_polynomial_expression (q : ℝ → ℝ) : 
  (∀ x, q x + (2*x^6 + 4*x^4 + 6*x^2 + 2) = 8*x^4 + 27*x^3 + 30*x^2 + 10*x + 3) →
  (∀ x, q x = -2*x^6 + 4*x^4 + 27*x^3 + 24*x^2 + 10*x + 1) := by
sorry

end q_polynomial_expression_l2358_235895


namespace ant_position_2024_l2358_235876

-- Define the ant's movement pattern
def antMove (n : ℕ) : ℤ × ℤ :=
  sorry

-- Theorem statement
theorem ant_position_2024 : antMove 2024 = (13, 0) := by
  sorry

end ant_position_2024_l2358_235876


namespace integer_solutions_of_equation_l2358_235832

theorem integer_solutions_of_equation : 
  {(a, b) : ℤ × ℤ | a^2 + b = b^2022} = {(0, 0), (0, 1)} := by
sorry

end integer_solutions_of_equation_l2358_235832


namespace quadratic_equation_roots_squared_equation_solutions_l2358_235862

-- Problem 1
theorem quadratic_equation_roots (x : ℝ) : 
  x^2 - 7*x + 6 = 0 ↔ x = 1 ∨ x = 6 := by sorry

-- Problem 2
theorem squared_equation_solutions (x : ℝ) :
  (2*x + 3)^2 = (x - 3)^2 ↔ x = 0 ∨ x = -6 := by sorry

end quadratic_equation_roots_squared_equation_solutions_l2358_235862


namespace smallest_dividend_l2358_235864

theorem smallest_dividend (A B : ℕ) (h1 : A = B * 28 + 4) (h2 : B > 0) : A ≥ 144 := by
  sorry

end smallest_dividend_l2358_235864


namespace even_expressions_l2358_235843

theorem even_expressions (m n : ℕ) (hm : m % 2 = 1) (hn : n % 2 = 1) :
  ∃ (k₁ k₂ k₃ : ℕ),
    (m - n)^2 = 2 * k₁ ∧
    (m - n - 4)^2 = 2 * k₂ ∧
    2 * m * n + 4 = 2 * k₃ := by
  sorry

end even_expressions_l2358_235843


namespace system_solution_l2358_235844

theorem system_solution (x y z : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (eq1 : x * y = 4 * z)
  (eq2 : x / y = 81)
  (eq3 : x * z = 36) :
  x = 36 ∧ y = 4/9 ∧ z = 1 := by
  sorry

end system_solution_l2358_235844


namespace pure_imaginary_complex_number_l2358_235856

theorem pure_imaginary_complex_number (m : ℝ) : 
  let i : ℂ := Complex.I
  let Z : ℂ := (1 + i) / (1 - i) + m * (1 - i)
  (Z.re = 0) → m = 0 := by
  sorry

end pure_imaginary_complex_number_l2358_235856


namespace angle_trigonometry_l2358_235810

open Real

theorem angle_trigonometry (x : ℝ) (h1 : π/2 < x) (h2 : x < π) 
  (h3 : cos x = tan x) (h4 : sin x ≠ cos x) : 
  sin x = (-1 + Real.sqrt 5) / 2 := by
sorry

end angle_trigonometry_l2358_235810


namespace f_equation_solution_l2358_235837

def f (x : ℝ) : ℝ := 3 * x - 5

theorem f_equation_solution :
  ∃ x : ℝ, 1 = f (x - 6) ∧ x = 8 := by
  sorry

end f_equation_solution_l2358_235837


namespace congruence_problem_l2358_235854

theorem congruence_problem (x : ℤ) : 
  x ≡ 1 [ZMOD 27] ∧ x ≡ 6 [ZMOD 37] → x ≡ 110 [ZMOD 999] := by
  sorry

end congruence_problem_l2358_235854


namespace soap_lasts_two_months_l2358_235891

def soap_problem (cost_per_bar : ℚ) (yearly_cost : ℚ) (months_per_year : ℕ) : ℚ :=
  (months_per_year : ℚ) / (yearly_cost / cost_per_bar)

theorem soap_lasts_two_months :
  soap_problem 8 48 12 = 2 := by
  sorry

end soap_lasts_two_months_l2358_235891


namespace negation_equivalence_l2358_235812

theorem negation_equivalence (x y : ℝ) : 
  ¬(x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔ (x^2 + y^2 ≠ 0 → ¬(x = 0 ∧ y = 0)) :=
sorry

end negation_equivalence_l2358_235812


namespace point_P_in_fourth_quadrant_l2358_235865

def point_P : ℝ × ℝ := (8, -3)

def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem point_P_in_fourth_quadrant :
  in_fourth_quadrant point_P := by
  sorry

end point_P_in_fourth_quadrant_l2358_235865


namespace sum_of_roots_l2358_235889

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 2028 :=
by sorry

end sum_of_roots_l2358_235889


namespace probability_two_blue_balls_l2358_235821

/-- The probability of drawing two blue balls consecutively from an urn -/
theorem probability_two_blue_balls (total_balls : Nat) (blue_balls : Nat) (red_balls : Nat) :
  total_balls = blue_balls + red_balls →
  blue_balls = 6 →
  red_balls = 4 →
  (blue_balls : ℚ) / total_balls * (blue_balls - 1) / (total_balls - 1) = 1 / 3 := by
  sorry

end probability_two_blue_balls_l2358_235821


namespace vector_properties_l2358_235899

/-- Given points in a 2D Cartesian coordinate system -/
def O : Fin 2 → ℝ := ![0, 0]
def A : Fin 2 → ℝ := ![1, 2]
def B : Fin 2 → ℝ := ![-3, 4]

/-- Vector AB -/
def vecAB : Fin 2 → ℝ := ![B 0 - A 0, B 1 - A 1]

/-- Theorem stating properties of vectors and angles in the given problem -/
theorem vector_properties :
  (vecAB 0 = -4 ∧ vecAB 1 = 2) ∧
  Real.sqrt ((vecAB 0)^2 + (vecAB 1)^2) = 2 * Real.sqrt 5 ∧
  ((A 0 * B 0 + A 1 * B 1) / (Real.sqrt (A 0^2 + A 1^2) * Real.sqrt (B 0^2 + B 1^2))) = Real.sqrt 5 / 5 := by
  sorry

end vector_properties_l2358_235899


namespace geometric_sequence_sum_l2358_235857

/-- Given a geometric sequence {a_n} with first term a₁ and common ratio q,
    prove that if a₂ * a₃ = 2a₁ and the arithmetic mean of (1/2)a₄ and a₇ is 5/8,
    then the sum of the first 4 terms (S₄) is 30. -/
theorem geometric_sequence_sum (a₁ q : ℝ) (h1 : a₁ * q * (a₁ * q^2) = 2 * a₁)
    (h2 : (1/2 * a₁ * q^3 + a₁ * q^6) / 2 = 5/8) :
  a₁ * (1 - q^4) / (1 - q) = 30 := by
  sorry


end geometric_sequence_sum_l2358_235857


namespace equation_solutions_l2358_235875

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = -2 + Real.sqrt 5 ∧ x₂ = -2 - Real.sqrt 5 ∧
    x₁^2 + 4*x₁ - 1 = 0 ∧ x₂^2 + 4*x₂ - 1 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 3 ∧ y₂ = 1 ∧
    (y₁ - 3)^2 + 2*y₁*(y₁ - 3) = 0 ∧ (y₂ - 3)^2 + 2*y₂*(y₂ - 3) = 0) :=
by sorry

end equation_solutions_l2358_235875


namespace unique_solutions_l2358_235896

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def valid_sequence (seq : List ℕ) : Prop :=
  seq.length = 16 ∧
  (∀ n, n ∈ seq → 1 ≤ n ∧ n ≤ 16) ∧
  (∀ n, 1 ≤ n ∧ n ≤ 16 → n ∈ seq) ∧
  (∀ i, i < 15 → is_perfect_square (seq[i]! + seq[i+1]!))

def solution1 : List ℕ := [16, 9, 7, 2, 14, 11, 5, 4, 12, 13, 3, 6, 10, 15, 1, 8]
def solution2 : List ℕ := [8, 1, 15, 10, 6, 3, 13, 12, 4, 5, 11, 14, 2, 7, 9, 16]

theorem unique_solutions :
  (∀ seq : List ℕ, valid_sequence seq → seq = solution1 ∨ seq = solution2) ∧
  valid_sequence solution1 ∧
  valid_sequence solution2 :=
sorry

end unique_solutions_l2358_235896


namespace solution_satisfies_equations_l2358_235883

theorem solution_satisfies_equations :
  let x : ℚ := -256 / 29
  let y : ℚ := -37 / 29
  (7 * x - 50 * y = 2) ∧ (3 * y - x = 5) := by
sorry

end solution_satisfies_equations_l2358_235883


namespace team_discount_saving_l2358_235803

/-- Represents the prices for a brand's uniform items -/
structure BrandPrices where
  shirt : ℝ
  pants : ℝ
  socks : ℝ

/-- Represents the prices for customization -/
structure CustomizationPrices where
  name : ℝ
  number : ℝ

def teamSize : ℕ := 12

def brandA : BrandPrices := ⟨7.5, 15, 4.5⟩
def brandB : BrandPrices := ⟨10, 18, 6⟩

def discountedBrandA : BrandPrices := ⟨6.75, 13.5, 3.75⟩
def discountedBrandB : BrandPrices := ⟨9, 16.5, 5.5⟩

def customization : CustomizationPrices := ⟨5, 3⟩

def playersWithFullCustomization : ℕ := 11

theorem team_discount_saving :
  let regularCost := 
    teamSize * (brandA.shirt + customization.name + customization.number) +
    teamSize * brandB.pants +
    teamSize * brandA.socks
  let discountedCost := 
    playersWithFullCustomization * (discountedBrandA.shirt + customization.name + customization.number) +
    (discountedBrandA.shirt + customization.name) +
    teamSize * discountedBrandB.pants +
    teamSize * brandA.socks
  regularCost - discountedCost = 31 := by sorry

end team_discount_saving_l2358_235803


namespace square_roots_problem_l2358_235830

theorem square_roots_problem (n : ℝ) (x : ℝ) (hn : n > 0) 
  (h1 : x + 1 = Real.sqrt n) (h2 : x - 5 = Real.sqrt n) : n = 9 := by
  sorry

end square_roots_problem_l2358_235830


namespace odd_function_negative_x_l2358_235870

/-- A function f is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_x
  (f : ℝ → ℝ)
  (odd : OddFunction f)
  (pos : ∀ x > 0, f x = x * (1 - x)) :
  ∀ x < 0, f x = x * (1 + x) := by
sorry

end odd_function_negative_x_l2358_235870


namespace max_spheres_in_frustum_l2358_235887

/-- Represents a frustum with given height and spheres inside it -/
structure Frustum :=
  (height : ℝ)
  (O₁_radius : ℝ)
  (O₂_radius : ℝ)

/-- Calculates the maximum number of additional spheres that can fit in the frustum -/
def max_additional_spheres (f : Frustum) : ℕ :=
  -- Implementation details are omitted
  sorry

/-- The main theorem stating the maximum number of additional spheres -/
theorem max_spheres_in_frustum (f : Frustum) 
  (h₁ : f.height = 8)
  (h₂ : f.O₁_radius = 2)
  (h₃ : f.O₂_radius = 3) :
  max_additional_spheres f = 2 :=
sorry

end max_spheres_in_frustum_l2358_235887


namespace distance_to_y_axis_l2358_235801

/-- Given a point P(x, -9) where the distance from the x-axis to P is half the distance
    from the y-axis to P, prove that the distance from P to the y-axis is 18 units. -/
theorem distance_to_y_axis (x : ℝ) :
  let P : ℝ × ℝ := (x, -9)
  (abs (P.2) = (1/2 : ℝ) * abs P.1) →
  abs P.1 = 18 := by
sorry

end distance_to_y_axis_l2358_235801


namespace cos_135_degrees_l2358_235836

theorem cos_135_degrees : Real.cos (135 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_135_degrees_l2358_235836


namespace curve_symmetry_l2358_235805

/-- The curve represented by the equation xy(x+y)=1 is symmetric about the line y=x -/
theorem curve_symmetry (x y : ℝ) : x * y * (x + y) = 1 ↔ y * x * (y + x) = 1 := by
  sorry

end curve_symmetry_l2358_235805


namespace cos_sin_fifteen_degrees_l2358_235868

theorem cos_sin_fifteen_degrees : 
  Real.cos (15 * π / 180) ^ 4 - Real.sin (15 * π / 180) ^ 4 = Real.sqrt 3 / 2 := by
  sorry

end cos_sin_fifteen_degrees_l2358_235868


namespace congruence_solution_l2358_235806

theorem congruence_solution :
  ∃ n : ℕ, 0 ≤ n ∧ n < 53 ∧ (14 * n) % 53 = 9 % 53 ∧ n = 36 := by
  sorry

end congruence_solution_l2358_235806


namespace sum_and_reciprocal_inequality_l2358_235878

theorem sum_and_reciprocal_inequality (x : ℝ) (hx : x > 0) : 
  x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1) := by
  sorry

end sum_and_reciprocal_inequality_l2358_235878


namespace difference_of_squares_l2358_235884

theorem difference_of_squares (m : ℝ) : (m + 1) * (m - 1) = m^2 - 1 := by
  sorry

end difference_of_squares_l2358_235884


namespace characterization_of_k_set_l2358_235888

-- Define h as 2^r where r is a non-negative integer
def h (r : ℕ) : ℕ := 2^r

-- Define the set of k that satisfy the conditions
def k_set (h : ℕ) : Set ℕ := {k : ℕ | ∃ (m n : ℕ), m > n ∧ k ∣ (m^h - 1) ∧ n^((m^h - 1) / k) ≡ -1 [ZMOD m]}

-- The theorem to prove
theorem characterization_of_k_set (r : ℕ) : 
  k_set (h r) = {k : ℕ | ∃ (s t : ℕ), k = 2^(r+s) * t ∧ Odd t} :=
sorry

end characterization_of_k_set_l2358_235888


namespace total_pencils_l2358_235841

/-- Given that each child has 4 pencils and there are 8 children, 
    prove that the total number of pencils is 32. -/
theorem total_pencils (pencils_per_child : ℕ) (num_children : ℕ) 
  (h1 : pencils_per_child = 4) (h2 : num_children = 8) : 
  pencils_per_child * num_children = 32 := by
  sorry

end total_pencils_l2358_235841


namespace crayons_per_box_l2358_235860

theorem crayons_per_box 
  (total_boxes : ℕ) 
  (total_crayons : ℕ) 
  (h1 : total_boxes = 7)
  (h2 : total_crayons = 35)
  (h3 : total_crayons = total_boxes * (total_crayons / total_boxes)) :
  total_crayons / total_boxes = 5 := by
sorry

end crayons_per_box_l2358_235860


namespace relay_arrangement_count_l2358_235808

def relay_arrangements (n : ℕ) (k : ℕ) (a b : ℕ) : ℕ :=
  sorry

theorem relay_arrangement_count : relay_arrangements 6 4 1 4 = 252 := by
  sorry

end relay_arrangement_count_l2358_235808


namespace negation_of_universal_proposition_l2358_235804

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 2 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 2 > 0) := by sorry

end negation_of_universal_proposition_l2358_235804


namespace flower_city_theorem_l2358_235850

/-- A bipartite graph representing the relationship between short men and little girls -/
structure FlowerCityGraph where
  A : Type -- Set of short men
  B : Type -- Set of little girls
  edge : A → B → Prop -- Edge relation

/-- The property that each short man knows exactly 6 little girls -/
def each_man_knows_six_girls (G : FlowerCityGraph) : Prop :=
  ∀ a : G.A, (∃! (b1 b2 b3 b4 b5 b6 : G.B), 
    G.edge a b1 ∧ G.edge a b2 ∧ G.edge a b3 ∧ G.edge a b4 ∧ G.edge a b5 ∧ G.edge a b6 ∧
    (∀ b : G.B, G.edge a b → (b = b1 ∨ b = b2 ∨ b = b3 ∨ b = b4 ∨ b = b5 ∨ b = b6)))

/-- The property that each little girl knows exactly 6 short men -/
def each_girl_knows_six_men (G : FlowerCityGraph) : Prop :=
  ∀ b : G.B, (∃! (a1 a2 a3 a4 a5 a6 : G.A), 
    G.edge a1 b ∧ G.edge a2 b ∧ G.edge a3 b ∧ G.edge a4 b ∧ G.edge a5 b ∧ G.edge a6 b ∧
    (∀ a : G.A, G.edge a b → (a = a1 ∨ a = a2 ∨ a = a3 ∨ a = a4 ∨ a = a5 ∨ a = a6)))

/-- The theorem stating that the number of short men equals the number of little girls -/
theorem flower_city_theorem (G : FlowerCityGraph) 
  (h1 : each_man_knows_six_girls G) 
  (h2 : each_girl_knows_six_men G) : 
  Nonempty (Equiv G.A G.B) :=
sorry

end flower_city_theorem_l2358_235850


namespace parallelogram_base_length_l2358_235892

theorem parallelogram_base_length 
  (area : ℝ) 
  (altitude_base_relation : ℝ → ℝ) 
  (base : ℝ) :
  area = 128 ∧ 
  altitude_base_relation = (λ x => 2 * x) ∧ 
  area = base * (altitude_base_relation base) →
  base = 8 := by
sorry

end parallelogram_base_length_l2358_235892


namespace complement_A_relative_to_I_l2358_235886

def I : Set Int := {-2, -1, 0, 1, 2}
def A : Set Int := {x : Int | x^2 < 3}

theorem complement_A_relative_to_I :
  {x ∈ I | x ∉ A} = {-2, 2} := by
  sorry

end complement_A_relative_to_I_l2358_235886


namespace negative_east_equals_positive_west_l2358_235826

-- Define the direction type
inductive Direction
| East
| West

-- Define a function to represent movement
def move (distance : Int) (direction : Direction) : Int :=
  match direction with
  | Direction.East => distance
  | Direction.West => -distance

-- Theorem statement
theorem negative_east_equals_positive_west :
  move (-8) Direction.East = move 8 Direction.West :=
by sorry

end negative_east_equals_positive_west_l2358_235826


namespace tan_theta_two_implies_expression_l2358_235817

theorem tan_theta_two_implies_expression (θ : Real) (h : Real.tan θ = 2) :
  (Real.sin θ + Real.cos θ) * Real.cos (2 * θ) / Real.sin θ = -9/10 := by
  sorry

end tan_theta_two_implies_expression_l2358_235817


namespace v_2023_equals_1_l2358_235880

-- Define the function g
def g : ℕ → ℕ
| 1 => 3
| 2 => 4
| 3 => 2
| 4 => 1
| 5 => 5
| _ => 0  -- For completeness, though not used in the problem

-- Define the sequence v
def v : ℕ → ℕ
| 0 => 3
| (n + 1) => g (v n)

-- Theorem statement
theorem v_2023_equals_1 : v 2023 = 1 := by
  sorry

end v_2023_equals_1_l2358_235880


namespace landscaping_equation_l2358_235853

-- Define the variables and constants
def total_area : ℝ := 180
def original_workers : ℕ := 6
def additional_workers : ℕ := 2
def time_saved : ℝ := 3

-- Define the theorem
theorem landscaping_equation (x : ℝ) :
  (total_area / (original_workers * x)) - (total_area / ((original_workers + additional_workers) * x)) = time_saved :=
by sorry

end landscaping_equation_l2358_235853


namespace air_conditioner_energy_savings_l2358_235822

/-- Represents the monthly energy savings in kWh for an air conditioner type -/
structure EnergySavings where
  savings : ℝ

/-- Represents the two types of air conditioners -/
inductive AirConditionerType
  | A
  | B

/-- The energy savings after raising temperature and cleaning for both air conditioner types -/
def energy_savings_after_measures (x y : EnergySavings) : ℝ :=
  x.savings + 1.1 * y.savings

/-- The theorem to be proved -/
theorem air_conditioner_energy_savings 
  (savings_A savings_B : EnergySavings) :
  savings_A.savings - savings_B.savings = 27 ∧
  energy_savings_after_measures savings_A savings_B = 405 →
  savings_A.savings = 207 ∧ savings_B.savings = 180 := by
  sorry

end air_conditioner_energy_savings_l2358_235822


namespace sasha_plucked_leaves_l2358_235800

/-- The number of leaves Sasha plucked -/
def leaves_plucked (apple_trees poplar_trees masha_last_apple sasha_start_apple unphotographed : ℕ) : ℕ :=
  (apple_trees + poplar_trees) - (sasha_start_apple - 1) - unphotographed

/-- Theorem stating the number of leaves Sasha plucked -/
theorem sasha_plucked_leaves :
  leaves_plucked 17 18 10 8 13 = 22 := by
  sorry

#eval leaves_plucked 17 18 10 8 13

end sasha_plucked_leaves_l2358_235800


namespace intersection_P_Q_l2358_235872

def P : Set ℕ := {0, 2, 4, 6}
def Q : Set ℕ := {x | x ≤ 3}

theorem intersection_P_Q : P ∩ Q = {0, 2} := by
  sorry

end intersection_P_Q_l2358_235872


namespace diana_apollo_dice_probability_l2358_235894

theorem diana_apollo_dice_probability :
  let diana_die := Finset.range 10
  let apollo_die := Finset.range 6
  let total_outcomes := diana_die.card * apollo_die.card
  let favorable_outcomes := (apollo_die.sum fun a => 
    (diana_die.filter (fun d => d > a)).card)
  (favorable_outcomes : ℚ) / total_outcomes = 13 / 20 := by
sorry

end diana_apollo_dice_probability_l2358_235894


namespace geometric_sequence_product_l2358_235815

theorem geometric_sequence_product (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 → -- Ensuring a, b, c are positive
  (1 : ℝ) < a ∧ a < b ∧ b < c ∧ c < 256 → -- Ensuring the order of the sequence
  (b / a = a / 1) ∧ (c / b = b / a) ∧ (256 / c = c / b) → -- Geometric sequence condition
  1 * a * b * c * 256 = 2^20 := by
sorry


end geometric_sequence_product_l2358_235815


namespace divide_3x8_rectangle_into_trominoes_l2358_235897

/-- Represents an L-shaped tromino -/
structure LTromino :=
  (cells : Nat)

/-- Represents a rectangle -/
structure Rectangle :=
  (width : Nat)
  (height : Nat)

/-- Number of ways to divide a rectangle into L-shaped trominoes -/
def divideRectangle (r : Rectangle) (t : LTromino) : Nat :=
  sorry

/-- Theorem: The number of ways to divide a 3 × 8 rectangle into L-shaped trominoes is 16 -/
theorem divide_3x8_rectangle_into_trominoes :
  let r := Rectangle.mk 8 3
  let t := LTromino.mk 3
  divideRectangle r t = 16 := by
  sorry

end divide_3x8_rectangle_into_trominoes_l2358_235897


namespace score_ordering_l2358_235852

structure Participant where
  score : ℕ

def Leonard : Participant := sorry
def Nina : Participant := sorry
def Oscar : Participant := sorry
def Paula : Participant := sorry

theorem score_ordering :
  (Oscar.score = Leonard.score) →
  (Nina.score < max Oscar.score Paula.score) →
  (Paula.score > Leonard.score) →
  (Oscar.score < Nina.score) ∧ (Nina.score < Paula.score) := by
  sorry

end score_ordering_l2358_235852


namespace inequality_solution_set_l2358_235809

theorem inequality_solution_set (x : ℝ) : 
  (x^2 - 2*x - 3) * (x^2 - 4*x + 4) < 0 ↔ -1 < x ∧ x < 3 ∧ x ≠ 2 :=
by sorry

end inequality_solution_set_l2358_235809


namespace expression_evaluation_l2358_235845

theorem expression_evaluation (c : ℕ) (h : c = 4) :
  (2 * c^c - (c + 1) * (c - 1)^c)^c = 131044201 := by
  sorry

end expression_evaluation_l2358_235845


namespace factorization_of_quadratic_l2358_235819

theorem factorization_of_quadratic (a : ℝ) : a^2 - 2*a - 15 = (a + 3) * (a - 5) := by
  sorry

end factorization_of_quadratic_l2358_235819
