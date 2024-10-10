import Mathlib

namespace sqrt_two_squared_inverse_half_l2210_221079

theorem sqrt_two_squared_inverse_half : 
  (((-Real.sqrt 2)^2)^(-1/2 : ℝ)) = Real.sqrt 2 / 2 := by sorry

end sqrt_two_squared_inverse_half_l2210_221079


namespace partial_fraction_decomposition_l2210_221028

theorem partial_fraction_decomposition :
  ∃ (C D : ℚ), C = 16/3 ∧ D = 5/3 ∧
  ∀ x : ℚ, x ≠ 12 ∧ x ≠ -3 →
    (7*x - 4) / (x^2 - 9*x - 36) = C / (x - 12) + D / (x + 3) :=
by sorry

end partial_fraction_decomposition_l2210_221028


namespace cosine_function_properties_l2210_221056

/-- Given a cosine function with specific properties, prove the value of ω and cos(α+β) -/
theorem cosine_function_properties (f : ℝ → ℝ) (ω α β : ℝ) :
  (∀ x, f x = 2 * Real.cos (ω * x + π / 6)) →
  ω > 0 →
  (∀ x, f (x + 10 * π) = f x) →
  (∀ y, y > 0 → y < 10 * π → ∀ x, f (x + y) ≠ f x) →
  α ∈ Set.Icc 0 (π / 2) →
  β ∈ Set.Icc 0 (π / 2) →
  f (5 * α + 5 * π / 3) = -6 / 5 →
  f (5 * β - 5 * π / 6) = 16 / 17 →
  ω = 1 / 5 ∧ Real.cos (α + β) = -13 / 85 := by
  sorry


end cosine_function_properties_l2210_221056


namespace geometric_sequence_minimum_sum_l2210_221083

theorem geometric_sequence_minimum_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- all terms are positive
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence
  a 6 = 3 →  -- a₆ = 3
  ∃ m : ℝ, m = 6 ∧ ∀ q, q > 0 → a 4 + a 8 ≥ m :=
by sorry

end geometric_sequence_minimum_sum_l2210_221083


namespace yellow_tint_percentage_l2210_221073

/-- Calculates the percentage of yellow tint in a new mixture after adding more yellow tint --/
theorem yellow_tint_percentage 
  (original_volume : ℝ) 
  (original_yellow_percentage : ℝ) 
  (added_yellow : ℝ) : 
  original_volume = 50 → 
  original_yellow_percentage = 0.5 → 
  added_yellow = 10 → 
  (((original_volume * original_yellow_percentage + added_yellow) / (original_volume + added_yellow)) * 100 : ℝ) = 58 := by
  sorry

end yellow_tint_percentage_l2210_221073


namespace tan_105_degrees_l2210_221016

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_degrees_l2210_221016


namespace always_composite_l2210_221002

theorem always_composite (n : ℕ) : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^4 + 64 = a * b := by
  sorry

end always_composite_l2210_221002


namespace age_difference_l2210_221051

/-- Proves that the age difference between a man and his son is 26 years, given the specified conditions. -/
theorem age_difference (son_age man_age : ℕ) : 
  son_age = 24 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 26 := by
  sorry

#check age_difference

end age_difference_l2210_221051


namespace yellow_preference_l2210_221023

theorem yellow_preference (total_students : ℕ) (total_girls : ℕ) 
  (h_total : total_students = 30)
  (h_girls : total_girls = 18)
  (h_green : total_students / 2 = total_students - (total_students / 2))
  (h_pink : total_girls / 3 = total_girls - (2 * (total_girls / 3))) :
  total_students - (total_students / 2 + total_girls / 3) = 9 := by
  sorry

end yellow_preference_l2210_221023


namespace function_inequality_implies_a_range_l2210_221035

open Real

theorem function_inequality_implies_a_range (a : ℝ) :
  (∀ x ≥ 0, 2 * (exp x) - 2 * a * x - a^2 + 3 - x^2 ≥ 0) →
  a ∈ Set.Icc (-Real.sqrt 5) (3 - Real.log 3) :=
by sorry

end function_inequality_implies_a_range_l2210_221035


namespace binomial_sum_expectation_variance_l2210_221019

/-- A random variable X following a binomial distribution B(n, p) -/
structure BinomialRV (n : ℕ) (p : ℝ) where
  X : ℝ

/-- The expectation of a binomial random variable -/
def expectation (X : BinomialRV n p) : ℝ := n * p

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV n p) : ℝ := n * p * (1 - p)

theorem binomial_sum_expectation_variance
  (X : BinomialRV 10 0.6) (Y : ℝ) 
  (h₁ : X.X + Y = 10) :
  expectation X = 6 ∧ 
  variance X = 2.4 ∧ 
  expectation X + Y = 10 → 
  Y = 4 ∧ variance X = 2.4 :=
sorry

end binomial_sum_expectation_variance_l2210_221019


namespace exactly_two_win_probability_l2210_221039

/-- The probability that exactly two out of three players win a game, given their individual probabilities of success. -/
theorem exactly_two_win_probability 
  (p_alice : ℚ) 
  (p_benjamin : ℚ) 
  (p_carol : ℚ) 
  (h_alice : p_alice = 1/5) 
  (h_benjamin : p_benjamin = 3/8) 
  (h_carol : p_carol = 2/7) : 
  (p_alice * p_benjamin * (1 - p_carol) + 
   p_alice * p_carol * (1 - p_benjamin) + 
   p_benjamin * p_carol * (1 - p_alice)) = 49/280 := by
sorry


end exactly_two_win_probability_l2210_221039


namespace river_trip_longer_than_lake_l2210_221068

/-- Proves that a round trip on a river takes longer than traveling the same distance on a lake -/
theorem river_trip_longer_than_lake (v w : ℝ) (h : v > w) (h_pos : v > 0) :
  (20 * v) / (v^2 - w^2) > 20 / v := by
  sorry

end river_trip_longer_than_lake_l2210_221068


namespace michael_sarah_games_l2210_221057

def total_players : ℕ := 12
def players_per_game : ℕ := 6

theorem michael_sarah_games (michael sarah : Fin total_players) 
  (h_distinct : michael ≠ sarah) :
  (Finset.univ.filter (λ game : Finset (Fin total_players) => 
    game.card = players_per_game ∧ 
    michael ∈ game ∧ 
    sarah ∈ game)).card = Nat.choose (total_players - 2) (players_per_game - 2) := by
  sorry

end michael_sarah_games_l2210_221057


namespace tangent_slope_at_x_one_l2210_221084

noncomputable def f (x : ℝ) := x^2 / 4 - Real.log x + 1

theorem tangent_slope_at_x_one (x : ℝ) (h : x > 0) :
  (deriv f x = -1/2) → x = 1 :=
by sorry

end tangent_slope_at_x_one_l2210_221084


namespace complement_of_A_l2210_221026

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | x > 1}

-- State the theorem
theorem complement_of_A : 
  Set.compl A = {x : ℝ | x ≤ 1} := by sorry

end complement_of_A_l2210_221026


namespace faster_speed_calculation_l2210_221008

/-- Proves that the faster speed is 20 km/hr given the conditions of the problem -/
theorem faster_speed_calculation (actual_distance : ℝ) (actual_speed : ℝ) (additional_distance : ℝ)
  (h1 : actual_distance = 20)
  (h2 : actual_speed = 10)
  (h3 : additional_distance = 20) :
  let time := actual_distance / actual_speed
  let total_distance := actual_distance + additional_distance
  let faster_speed := total_distance / time
  faster_speed = 20 := by sorry

end faster_speed_calculation_l2210_221008


namespace garden_width_is_ten_l2210_221017

/-- Represents a rectangular garden with specific properties. -/
structure RectangularGarden where
  width : ℝ
  length : ℝ
  perimeter_eq : width * 2 + length * 2 = 60
  area_eq : width * length = 200
  length_twice_width : length = 2 * width

/-- Theorem stating that a rectangular garden with the given properties has a width of 10 meters. -/
theorem garden_width_is_ten (garden : RectangularGarden) : garden.width = 10 := by
  sorry

end garden_width_is_ten_l2210_221017


namespace max_distance_circle_C_to_line_L_l2210_221078

/-- Circle C with equation x^2 + y^2 - 4x + m = 0 -/
def circle_C (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 + m = 0}

/-- Circle D with equation (x-3)^2 + (y+2√2)^2 = 4 -/
def circle_D : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1-3)^2 + (p.2+2*Real.sqrt 2)^2 = 4}

/-- Line L with equation 3x - 4y + 4 = 0 -/
def line_L : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3*p.1 - 4*p.2 + 4 = 0}

/-- The distance from a point to a line -/
def distance_point_to_line (p : ℝ × ℝ) (L : Set (ℝ × ℝ)) : ℝ := sorry

/-- The maximum distance from any point on a set to a line -/
def max_distance_set_to_line (S : Set (ℝ × ℝ)) (L : Set (ℝ × ℝ)) : ℝ := sorry

theorem max_distance_circle_C_to_line_L :
  ∃ m : ℝ,
    (circle_C m).Nonempty ∧
    (∃ p : ℝ × ℝ, p ∈ circle_C m ∧ p ∈ circle_D) ∧
    max_distance_set_to_line (circle_C m) line_L = 3 := by sorry

end max_distance_circle_C_to_line_L_l2210_221078


namespace count_special_numbers_eq_3776_l2210_221074

/-- A function that counts the number of five-digit numbers beginning with 2 
    that have exactly two identical digits -/
def count_special_numbers : ℕ :=
  let case1 := 4 * 8 * 8 * 8  -- Case where the two identical digits are 2s
  let case2 := 3 * 9 * 8 * 8  -- Case where the two identical digits are not 2s
  case1 + case2

/-- Theorem stating that there are exactly 3776 five-digit numbers 
    beginning with 2 that have exactly two identical digits -/
theorem count_special_numbers_eq_3776 :
  count_special_numbers = 3776 := by
  sorry

#eval count_special_numbers  -- Should output 3776

end count_special_numbers_eq_3776_l2210_221074


namespace heroes_total_l2210_221089

theorem heroes_total (front back : ℕ) (h1 : front = 2) (h2 : back = 7) :
  front + back = 9 := by sorry

end heroes_total_l2210_221089


namespace triangle_max_perimeter_l2210_221097

theorem triangle_max_perimeter :
  ∀ (x y : ℕ),
    x > 0 →
    y > 0 →
    y = 2 * x →
    (x + y > 20 ∧ x + 20 > y ∧ y + 20 > x) →
    x + y + 20 ≤ 77 :=
by
  sorry

end triangle_max_perimeter_l2210_221097


namespace nancy_physical_education_marks_l2210_221000

def american_literature : ℕ := 66
def history : ℕ := 75
def home_economics : ℕ := 52
def art : ℕ := 89
def average_marks : ℕ := 70
def total_subjects : ℕ := 5

theorem nancy_physical_education_marks :
  let known_subjects_total := american_literature + history + home_economics + art
  let total_marks := average_marks * total_subjects
  total_marks - known_subjects_total = 68 := by
sorry

end nancy_physical_education_marks_l2210_221000


namespace interval_length_theorem_l2210_221061

theorem interval_length_theorem (c d : ℝ) : 
  (∃ (x : ℝ), c ≤ 3 * x + 5 ∧ 3 * x + 5 ≤ d) → 
  ((d - 5) / 3 - (c - 5) / 3 = 15) → 
  d - c = 45 := by
sorry

end interval_length_theorem_l2210_221061


namespace injury_healing_ratio_l2210_221050

/-- The number of days it takes for the pain to subside -/
def pain_subsided : ℕ := 3

/-- The number of days James waits after full healing before working out -/
def wait_before_workout : ℕ := 3

/-- The number of days James waits before lifting heavy -/
def wait_before_heavy : ℕ := 21

/-- The total number of days until James can lift heavy again -/
def total_days : ℕ := 39

/-- The number of days it takes for the injury to fully heal -/
def healing_time : ℕ := total_days - pain_subsided - wait_before_workout - wait_before_heavy

/-- The ratio of healing time to pain subsided time -/
def healing_ratio : ℚ := healing_time / pain_subsided

theorem injury_healing_ratio : healing_ratio = 4 / 1 := by
  sorry

end injury_healing_ratio_l2210_221050


namespace triangle_six_nine_equals_eleven_l2210_221054

-- Define the ▽ operation
def triangle (m n : ℚ) (x y : ℚ) : ℚ := m^2 * x + n * y - 1

-- Theorem statement
theorem triangle_six_nine_equals_eleven 
  (m n : ℚ) 
  (h : triangle m n 2 3 = 3) : 
  triangle m n 6 9 = 11 := by
sorry

end triangle_six_nine_equals_eleven_l2210_221054


namespace fraction_simplification_l2210_221003

theorem fraction_simplification : 
  (45 : ℚ) / 28 * 49 / 75 * 100 / 63 = 5 / 3 := by
  sorry

end fraction_simplification_l2210_221003


namespace fruit_basket_total_l2210_221081

/-- Calculates the total number of fruits in a basket given the number of oranges and relationships between fruit quantities. -/
def totalFruits (oranges : ℕ) : ℕ :=
  let apples := oranges - 2
  let bananas := 3 * apples
  let peaches := bananas / 2
  oranges + apples + bananas + peaches

/-- Theorem stating that the total number of fruits in the basket is 28. -/
theorem fruit_basket_total : totalFruits 6 = 28 := by
  sorry

end fruit_basket_total_l2210_221081


namespace drug_price_reduction_l2210_221032

theorem drug_price_reduction (initial_price final_price : ℝ) (x : ℝ) :
  initial_price = 63800 →
  final_price = 3900 →
  final_price = initial_price * (1 - x)^2 :=
by sorry

end drug_price_reduction_l2210_221032


namespace smallest_seem_l2210_221018

/-- Represents a digit mapping for the puzzle MY + ROZH = SEEM -/
structure DigitMapping where
  m : Nat
  y : Nat
  r : Nat
  o : Nat
  z : Nat
  h : Nat
  s : Nat
  e : Nat
  unique : m ≠ y ∧ m ≠ r ∧ m ≠ o ∧ m ≠ z ∧ m ≠ h ∧ m ≠ s ∧ m ≠ e ∧
           y ≠ r ∧ y ≠ o ∧ y ≠ z ∧ y ≠ h ∧ y ≠ s ∧ y ≠ e ∧
           r ≠ o ∧ r ≠ z ∧ r ≠ h ∧ r ≠ s ∧ r ≠ e ∧
           o ≠ z ∧ o ≠ h ∧ o ≠ s ∧ o ≠ e ∧
           z ≠ h ∧ z ≠ s ∧ z ≠ e ∧
           h ≠ s ∧ h ≠ e ∧
           s ≠ e
  valid_digits : m < 10 ∧ y < 10 ∧ r < 10 ∧ o < 10 ∧ z < 10 ∧ h < 10 ∧ s < 10 ∧ e < 10
  s_greater_than_one : s > 1

/-- The equation MY + ROZH = SEEM holds for the given digit mapping -/
def equation_holds (d : DigitMapping) : Prop :=
  10 * d.m + d.y + 1000 * d.r + 100 * d.o + 10 * d.z + d.h = 1000 * d.s + 100 * d.e + 10 * d.e + d.m

/-- There exists a valid digit mapping for which the equation holds -/
def exists_valid_mapping : Prop :=
  ∃ d : DigitMapping, equation_holds d

/-- 2003 is the smallest four-digit number SEEM for which there exists a valid mapping -/
theorem smallest_seem : (∃ d : DigitMapping, d.s = 2 ∧ d.e = 0 ∧ d.m = 3 ∧ equation_holds d) ∧
  (∀ n : Nat, n < 2003 → ¬∃ d : DigitMapping, 1000 * d.s + 100 * d.e + 10 * d.e + d.m = n ∧ equation_holds d) :=
sorry

end smallest_seem_l2210_221018


namespace divides_fk_iff_divides_f_l2210_221041

theorem divides_fk_iff_divides_f (k : ℕ) (f : ℕ → ℕ) (x : ℕ) :
  (∀ n : ℕ, ∃ m : ℕ, f n = m * n) →
  (x ∣ f^[k] x ↔ x ∣ f x) :=
sorry

end divides_fk_iff_divides_f_l2210_221041


namespace max_coeff_seventh_term_l2210_221006

theorem max_coeff_seventh_term (n : ℕ) : 
  (∃ k, (Nat.choose n k = Nat.choose n 6) ∧ 
        (∀ j, 0 ≤ j ∧ j ≤ n → Nat.choose n j ≤ Nat.choose n 6)) →
  n ∈ ({11, 12, 13} : Set ℕ) := by
sorry

end max_coeff_seventh_term_l2210_221006


namespace f_inequality_iff_x_gt_one_l2210_221048

noncomputable def f (x : ℝ) : ℝ := Real.sin (x - 1) + Real.exp (x - 1) - Real.exp (1 - x) - x + 1

theorem f_inequality_iff_x_gt_one :
  ∀ x : ℝ, f x + f (3 - 2*x) < 0 ↔ x > 1 := by
  sorry

end f_inequality_iff_x_gt_one_l2210_221048


namespace min_value_sum_product_l2210_221004

theorem min_value_sum_product (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → x * y = 1 → a + 2 * b ≤ x + 2 * y ∧
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y = 1 ∧ x + 2 * y = 2 * Real.sqrt 2 := by
sorry

end min_value_sum_product_l2210_221004


namespace inequality_solution_l2210_221088

noncomputable def f (x : ℝ) : ℝ := x^2 / ((x - 2)^2 * (x + 1))

theorem inequality_solution :
  ∀ x : ℝ, x ≠ 2 ∧ x ≠ -1 →
  (f x ≥ 0 ↔ x ∈ Set.Iio (-1) ∪ {0} ∪ Set.Ioi 2) :=
by sorry

end inequality_solution_l2210_221088


namespace S_is_three_rays_with_common_point_l2210_221071

-- Define the set S
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let x := p.1; let y := p.2;
    (3 = x + 2 ∧ y - 4 ≤ 3) ∨
    (3 = y - 4 ∧ x + 2 ≤ 3) ∨
    (x + 2 = y - 4 ∧ 3 ≤ x + 2)}

-- Define a ray
def Ray (start : ℝ × ℝ) (direction : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, t ≥ 0 ∧ p = (start.1 + t * direction.1, start.2 + t * direction.2)}

-- Theorem statement
theorem S_is_three_rays_with_common_point :
  ∃ (common_point : ℝ × ℝ) (ray1 ray2 ray3 : Set (ℝ × ℝ)),
    S = ray1 ∪ ray2 ∪ ray3 ∧
    (∃ (d1 d2 d3 : ℝ × ℝ),
      ray1 = Ray common_point d1 ∧
      ray2 = Ray common_point d2 ∧
      ray3 = Ray common_point d3) ∧
    ray1 ∩ ray2 = {common_point} ∧
    ray2 ∩ ray3 = {common_point} ∧
    ray3 ∩ ray1 = {common_point} :=
  sorry


end S_is_three_rays_with_common_point_l2210_221071


namespace hat_cost_theorem_l2210_221062

def josh_shopping (initial_money : ℝ) (pencil_cost : ℝ) (cookie_cost : ℝ) (cookie_count : ℕ) (money_left : ℝ) : ℝ :=
  initial_money - (pencil_cost + cookie_cost * cookie_count) - money_left

theorem hat_cost_theorem (initial_money : ℝ) (pencil_cost : ℝ) (cookie_cost : ℝ) (cookie_count : ℕ) (money_left : ℝ) 
  (h1 : initial_money = 20)
  (h2 : pencil_cost = 2)
  (h3 : cookie_cost = 1.25)
  (h4 : cookie_count = 4)
  (h5 : money_left = 3) :
  josh_shopping initial_money pencil_cost cookie_cost cookie_count money_left = 10 := by
  sorry

#eval josh_shopping 20 2 1.25 4 3

end hat_cost_theorem_l2210_221062


namespace toy_donation_difference_l2210_221092

def leila_bags : ℕ := 2
def leila_toys_per_bag : ℕ := 25
def mohamed_bags : ℕ := 3
def mohamed_toys_per_bag : ℕ := 19

theorem toy_donation_difference : 
  mohamed_bags * mohamed_toys_per_bag - leila_bags * leila_toys_per_bag = 7 := by
  sorry

end toy_donation_difference_l2210_221092


namespace problem_solution_l2210_221053

-- Define the line y = ax - 2a + 4
def line_a (a : ℝ) (x : ℝ) : ℝ := a * x - 2 * a + 4

-- Define the point (2, 4)
def point_2_4 : ℝ × ℝ := (2, 4)

-- Define the line y + 1 = 3x
def line_3x (x : ℝ) : ℝ := 3 * x - 1

-- Define the line x + √3y + 1 = 0
def line_sqrt3 (x y : ℝ) : Prop := x + Real.sqrt 3 * y + 1 = 0

-- Define the point (-2, 3)
def point_neg2_3 : ℝ × ℝ := (-2, 3)

-- Define the line x - 2y + 3 = 0
def line_1 (x y : ℝ) : Prop := x - 2 * y + 3 = 0

-- Define the line 2x + y + 1 = 0
def line_2 (x y : ℝ) : Prop := 2 * x + y + 1 = 0

theorem problem_solution :
  (∀ a : ℝ, line_a a (point_2_4.1) = point_2_4.2) ∧
  (line_2 point_neg2_3.1 point_neg2_3.2 ∧
   ∀ x y : ℝ, line_1 x y → line_2 x y → x = y) :=
sorry

end problem_solution_l2210_221053


namespace unique_solution_l2210_221022

theorem unique_solution : 
  ∃! (x y z t : ℕ), 31 * (x * y * z * t + x * y + x * t + z * t + 1) = 40 * (y * z * t + y + t) ∧
  x = 1 ∧ y = 3 ∧ z = 2 ∧ t = 4 :=
by sorry

end unique_solution_l2210_221022


namespace lily_newspaper_collection_l2210_221043

/-- Given that Chris collected 42 newspapers and the total number of newspapers
    collected by Chris and Lily is 65, prove that Lily collected 23 newspapers. -/
theorem lily_newspaper_collection (chris_newspapers lily_newspapers total_newspapers : ℕ) :
  chris_newspapers = 42 →
  total_newspapers = 65 →
  total_newspapers = chris_newspapers + lily_newspapers →
  lily_newspapers = 23 := by
sorry

end lily_newspaper_collection_l2210_221043


namespace fraction_equation_solution_l2210_221064

theorem fraction_equation_solution (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) :
  (1 / (x - 2) = 3 / x) ↔ x = 3 := by
  sorry

end fraction_equation_solution_l2210_221064


namespace tangent_line_intersects_ellipse_perpendicularly_l2210_221082

/-- An ellipse with semi-major axis 2 and semi-minor axis 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1}

/-- A circle with radius 2√5/5 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = (2 * Real.sqrt 5 / 5)^2}

/-- A line tangent to the circle at point (m, n) -/
def TangentLine (m n : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | m * p.1 + n * p.2 = (2 * Real.sqrt 5 / 5)^2}

/-- The origin of the coordinate system -/
def Origin : ℝ × ℝ := (0, 0)

/-- Two points are perpendicular with respect to the origin -/
def Perpendicular (p q : ℝ × ℝ) : Prop :=
  p.1 * q.1 + p.2 * q.2 = 0

theorem tangent_line_intersects_ellipse_perpendicularly 
  (m n : ℝ) (h : (m, n) ∈ Circle) :
  ∃ (A B : ℝ × ℝ), A ∈ Ellipse ∧ B ∈ Ellipse ∧ 
    A ∈ TangentLine m n ∧ B ∈ TangentLine m n ∧
    Perpendicular (A.1 - Origin.1, A.2 - Origin.2) (B.1 - Origin.1, B.2 - Origin.2) :=
sorry

end tangent_line_intersects_ellipse_perpendicularly_l2210_221082


namespace units_digit_153_base3_l2210_221060

/-- Converts a natural number to its base 3 representation -/
def toBase3 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 3) ((m % 3) :: acc)
    aux n []

/-- The units digit is the last digit in the list representation -/
def unitsDigit (digits : List ℕ) : ℕ :=
  match digits.reverse with
  | [] => 0
  | d :: _ => d

theorem units_digit_153_base3 :
  unitsDigit (toBase3 153) = 2 := by
  sorry

end units_digit_153_base3_l2210_221060


namespace quadrilateral_perimeter_l2210_221094

-- Define the quadrilateral ABCD and point P
structure Quadrilateral :=
  (A B C D P : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_convex (q : Quadrilateral) : Prop := sorry

def area (q : Quadrilateral) : ℝ := sorry

def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

def diagonals_orthogonal (q : Quadrilateral) : Prop := sorry

def perimeter (q : Quadrilateral) : ℝ := sorry

-- State the theorem
theorem quadrilateral_perimeter 
  (q : Quadrilateral)
  (h_convex : is_convex q)
  (h_area : area q = 2601)
  (h_PA : distance q.P q.A = 25)
  (h_PB : distance q.P q.B = 35)
  (h_PC : distance q.P q.C = 30)
  (h_PD : distance q.P q.D = 50)
  (h_ortho : diagonals_orthogonal q) :
  perimeter q = Real.sqrt 1850 + Real.sqrt 2125 + Real.sqrt 3400 + Real.sqrt 3125 :=
sorry

end quadrilateral_perimeter_l2210_221094


namespace complex_fraction_simplification_l2210_221005

/-- Given that i^2 = -1, prove that (3-2i)/(4+5i) = 2/41 - (23/41)i -/
theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (3 - 2*i) / (4 + 5*i) = 2/41 - (23/41)*i :=
by sorry

end complex_fraction_simplification_l2210_221005


namespace wings_count_l2210_221001

/-- Calculates the total number of wings for birds bought with money from grandparents -/
def total_wings (num_grandparents : ℕ) (money_per_grandparent : ℕ) (cost_per_bird : ℕ) (wings_per_bird : ℕ) : ℕ :=
  let total_money := num_grandparents * money_per_grandparent
  let num_birds := total_money / cost_per_bird
  num_birds * wings_per_bird

/-- Theorem: Given the problem conditions, the total number of wings is 20 -/
theorem wings_count :
  total_wings 4 50 20 2 = 20 := by
  sorry

end wings_count_l2210_221001


namespace sum_of_products_l2210_221009

theorem sum_of_products (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 941) 
  (h2 : a + b + c = 31) : 
  a*b + b*c + c*a = 10 := by
sorry

end sum_of_products_l2210_221009


namespace triangle_inequality_from_inequality_l2210_221034

theorem triangle_inequality_from_inequality (a b c : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (ineq : 6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) :
  c < a + b ∧ a < b + c ∧ b < c + a := by
  sorry

end triangle_inequality_from_inequality_l2210_221034


namespace infinitely_many_special_prisms_l2210_221014

/-- A rectangular prism with two equal edges and the third differing by 1 -/
structure SpecialPrism where
  a : ℕ
  b : ℕ
  h_b : b = a + 1 ∨ b = a - 1

/-- The body diagonal of a rectangular prism is an integer -/
def has_integer_diagonal (p : SpecialPrism) : Prop :=
  ∃ d : ℕ, d^2 = 2 * p.a^2 + p.b^2

/-- There are infinitely many rectangular prisms with integer edges and diagonal,
    where two edges are equal and the third differs by 1 -/
theorem infinitely_many_special_prisms :
  ∀ n : ℕ, ∃ (prisms : Finset SpecialPrism),
    prisms.card > n ∧ ∀ p ∈ prisms, has_integer_diagonal p :=
sorry

end infinitely_many_special_prisms_l2210_221014


namespace simplify_trig_expression_l2210_221087

theorem simplify_trig_expression (x : ℝ) :
  (Real.sin x + Real.sin (3 * x)) / (1 + Real.cos x + Real.cos (3 * x)) =
  4 * (Real.sin x - Real.sin x ^ 3) / (1 - 2 * Real.cos x + 4 * Real.cos x ^ 3) := by
  sorry

end simplify_trig_expression_l2210_221087


namespace polynomial_coefficient_equality_l2210_221021

theorem polynomial_coefficient_equality (m n : ℤ) : 
  (∀ x : ℝ, (x - 1) * (x + m) = x^2 - n*x - 6) → 
  (m = 6 ∧ n = -5) :=
by sorry

end polynomial_coefficient_equality_l2210_221021


namespace hot_dogs_leftover_l2210_221031

theorem hot_dogs_leftover : 36159782 % 6 = 2 := by
  sorry

end hot_dogs_leftover_l2210_221031


namespace absolute_value_sqrt_problem_l2210_221025

theorem absolute_value_sqrt_problem : |-2 * Real.sqrt 2| - Real.sqrt 4 * Real.sqrt 2 + (π - 5)^0 = 1 := by
  sorry

end absolute_value_sqrt_problem_l2210_221025


namespace inequality_proof_l2210_221059

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_prod : a * b * c = 1) : 
  (a + 1/b)^2 + (b + 1/c)^2 + (c + 1/a)^2 ≥ 3 * (a + b + c + 1) := by
  sorry

end inequality_proof_l2210_221059


namespace intersection_point_solution_and_b_value_l2210_221063

/-- The intersection point of two lines -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- A linear function of the form y = mx + c -/
structure LinearFunction where
  m : ℝ
  c : ℝ

/-- Given two linear functions and their intersection point, prove the solution and b value -/
theorem intersection_point_solution_and_b_value 
  (f1 : LinearFunction)
  (f2 : LinearFunction)
  (P : IntersectionPoint)
  (h1 : f1.m = 2 ∧ f1.c = -5)
  (h2 : f2.m = 3)
  (h3 : P.x = 1 ∧ P.y = -3)
  (h4 : P.y = f1.m * P.x + f1.c)
  (h5 : P.y = f2.m * P.x + f2.c) :
  (∃ (x y : ℝ), x = 1 ∧ y = -3 ∧ 
    y = f1.m * x + f1.c ∧
    y = f2.m * x + f2.c) ∧
  f2.c = -6 := by
  sorry

end intersection_point_solution_and_b_value_l2210_221063


namespace modulo_nine_sum_l2210_221012

theorem modulo_nine_sum (n : ℕ) : n = 2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 → 0 ≤ n ∧ n < 9 → n % 9 = 6 := by
  sorry

end modulo_nine_sum_l2210_221012


namespace test_average_l2210_221091

theorem test_average (male_count : ℕ) (female_count : ℕ) 
  (male_avg : ℝ) (female_avg : ℝ) : 
  male_count = 8 → 
  female_count = 32 → 
  male_avg = 82 → 
  female_avg = 92 → 
  (male_count * male_avg + female_count * female_avg) / (male_count + female_count) = 90 := by
  sorry

end test_average_l2210_221091


namespace largest_d_for_negative_five_in_range_l2210_221069

/-- The function g(x) = x^2 + 2x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + d

/-- The largest value of d such that -5 is in the range of g(x) -/
theorem largest_d_for_negative_five_in_range : 
  (∃ (d : ℝ), ∀ (e : ℝ), (∃ (x : ℝ), g e x = -5) → e ≤ d) ∧ 
  (∃ (x : ℝ), g (-4) x = -5) :=
sorry

end largest_d_for_negative_five_in_range_l2210_221069


namespace numbers_solution_l2210_221015

def find_numbers (x y z : ℤ) : Prop :=
  (y = 2*x - 3) ∧ 
  (x + y = 51) ∧ 
  (z = 4*x - y)

theorem numbers_solution : 
  ∃ (x y z : ℤ), find_numbers x y z ∧ x = 18 ∧ y = 33 ∧ z = 39 :=
sorry

end numbers_solution_l2210_221015


namespace conference_games_l2210_221024

/-- Calculates the number of games within a division -/
def games_within_division (n : ℕ) : ℕ := n * (n - 1)

/-- Calculates the number of games between two divisions -/
def games_between_divisions (n m : ℕ) : ℕ := 2 * n * m

/-- The total number of games in the conference -/
def total_games : ℕ :=
  let div_a := 6
  let div_b := 7
  let div_c := 5
  let within_a := games_within_division div_a
  let within_b := games_within_division div_b
  let within_c := games_within_division div_c
  let between_ab := games_between_divisions div_a div_b
  let between_ac := games_between_divisions div_a div_c
  let between_bc := games_between_divisions div_b div_c
  within_a + within_b + within_c + between_ab + between_ac + between_bc

theorem conference_games : total_games = 306 := by sorry

end conference_games_l2210_221024


namespace trig_identity_l2210_221011

theorem trig_identity : 
  1 / Real.cos (70 * π / 180) + Real.sqrt 2 / Real.sin (70 * π / 180) = 
  4 * Real.sin (65 * π / 180) / Real.sin (40 * π / 180) := by
  sorry

end trig_identity_l2210_221011


namespace savings_multiple_l2210_221066

theorem savings_multiple (monthly_pay : ℝ) (savings_fraction : ℝ) : 
  savings_fraction = 0.29411764705882354 →
  monthly_pay > 0 →
  let monthly_savings := monthly_pay * savings_fraction
  let monthly_non_savings := monthly_pay - monthly_savings
  let total_savings := monthly_savings * 12
  total_savings = 5 * monthly_non_savings :=
by sorry

end savings_multiple_l2210_221066


namespace turtle_time_to_watering_hole_l2210_221042

/-- Represents the scenario of two lion cubs and a turtle moving towards a watering hole --/
structure WateringHoleScenario where
  /-- Speed of the first lion cub (in distance units per minute) --/
  speed_lion1 : ℝ
  /-- Distance of the first lion cub from the watering hole (in minutes) --/
  distance_lion1 : ℝ
  /-- Speed multiplier of the second lion cub relative to the first --/
  speed_multiplier_lion2 : ℝ
  /-- Distance of the turtle from the watering hole (in minutes) --/
  distance_turtle : ℝ

/-- Theorem stating the time it takes for the turtle to reach the watering hole after meeting the lion cubs --/
theorem turtle_time_to_watering_hole (scenario : WateringHoleScenario)
  (h1 : scenario.distance_lion1 = 5)
  (h2 : scenario.speed_multiplier_lion2 = 1.5)
  (h3 : scenario.distance_turtle = 30)
  (h4 : scenario.speed_lion1 > 0) :
  let meeting_time := 2
  let turtle_speed := 1 / scenario.distance_turtle
  let remaining_distance := 1 - meeting_time * turtle_speed
  remaining_distance * scenario.distance_turtle = 28 := by
  sorry

end turtle_time_to_watering_hole_l2210_221042


namespace unique_determination_l2210_221072

/-- Two-digit number type -/
def TwoDigitNum := {n : ℕ // n ≥ 0 ∧ n ≤ 99}

/-- The sum function as defined in the problem -/
def sum (a b c : TwoDigitNum) (X Y Z : ℕ) : ℕ :=
  a.val * X + b.val * Y + c.val * Z

/-- Function to extract a from the sum -/
def extract_a (S : ℕ) : ℕ := S % 100

/-- Function to extract b from the sum -/
def extract_b (S : ℕ) : ℕ := (S / 100) % 100

/-- Function to extract c from the sum -/
def extract_c (S : ℕ) : ℕ := S / 10000

/-- Theorem stating that a, b, and c can be uniquely determined from the sum -/
theorem unique_determination (a b c : TwoDigitNum) :
  let X : ℕ := 1
  let Y : ℕ := 100
  let Z : ℕ := 10000
  let S := sum a b c X Y Z
  (extract_a S = a.val) ∧ (extract_b S = b.val) ∧ (extract_c S = c.val) := by
  sorry

end unique_determination_l2210_221072


namespace exam_score_problem_l2210_221055

theorem exam_score_problem (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ) :
  total_questions = 60 →
  correct_score = 4 →
  wrong_score = -1 →
  total_score = 130 →
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score ∧
    correct_answers = 38 := by
  sorry

end exam_score_problem_l2210_221055


namespace slower_painter_time_l2210_221027

-- Define the start time of the slower painter (2:00 PM)
def slower_start : ℝ := 14

-- Define the finish time (0.6 past midnight, which is 24.6)
def finish_time : ℝ := 24.6

-- Theorem to prove
theorem slower_painter_time :
  finish_time - slower_start = 10.6 := by
  sorry

end slower_painter_time_l2210_221027


namespace symmetric_point_y_axis_P_l2210_221093

/-- Given a point P in 3D space, this function returns its symmetric point with respect to the y-axis -/
def symmetricPointYAxis (P : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := P
  (-x, y, z)

/-- Theorem stating that the symmetric point of P(1,2,-1) with respect to the y-axis is (-1,2,1) -/
theorem symmetric_point_y_axis_P :
  symmetricPointYAxis (1, 2, -1) = (-1, 2, -1) := by
  sorry

end symmetric_point_y_axis_P_l2210_221093


namespace odd_even_sum_difference_problem_statement_l2210_221036

theorem odd_even_sum_difference : ℕ → Prop :=
  fun n =>
    let odd_sum := (n^2 + 2*n + 1)^2
    let even_sum := n * (n + 1) * (2*n + 2)
    odd_sum - even_sum = 3057

theorem problem_statement : odd_even_sum_difference 1012 := by
  sorry

end odd_even_sum_difference_problem_statement_l2210_221036


namespace least_valid_number_l2210_221077

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧  -- Four-digit positive integer
  ∃ (a b c d : ℕ), 
    n = 1000 * a + 100 * b + 10 * c + d ∧  -- Digit representation
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧  -- All digits are different
    (a = 5 ∨ b = 5 ∨ c = 5 ∨ d = 5) ∧  -- One of the digits is 5
    n % a = 0 ∧ n % b = 0 ∧ n % c = 0 ∧ n % d = 0  -- Divisible by each of its digits

theorem least_valid_number : 
  is_valid_number 1524 ∧ ∀ m : ℕ, is_valid_number m → m ≥ 1524 :=
sorry

end least_valid_number_l2210_221077


namespace sin_2alpha_value_l2210_221030

theorem sin_2alpha_value (α : ℝ) (h : Real.cos (π / 4 - α) = -4 / 5) : 
  Real.sin (2 * α) = 7 / 25 := by
  sorry

end sin_2alpha_value_l2210_221030


namespace total_fish_caught_l2210_221029

-- Define the types of fish
inductive FishType
| Trout
| Salmon
| Tuna

-- Define a function to calculate the pounds of fish caught for each type
def poundsCaught (fishType : FishType) : ℕ :=
  match fishType with
  | .Trout => 200
  | .Salmon => 200 + 200 / 2
  | .Tuna => 2 * (200 + 200 / 2)

-- Theorem statement
theorem total_fish_caught :
  (poundsCaught FishType.Trout) +
  (poundsCaught FishType.Salmon) +
  (poundsCaught FishType.Tuna) = 1100 := by
  sorry


end total_fish_caught_l2210_221029


namespace hay_from_grass_l2210_221065

/-- The amount of hay obtained from freshly cut grass -/
theorem hay_from_grass (initial_mass : ℝ) (grass_moisture : ℝ) (hay_moisture : ℝ) : 
  initial_mass = 1000 →
  grass_moisture = 0.6 →
  hay_moisture = 0.15 →
  (initial_mass * (1 - grass_moisture)) / (1 - hay_moisture) = 470^10 / 17 := by
  sorry

#eval (470^10 : ℚ) / 17

end hay_from_grass_l2210_221065


namespace remainder_theorem_l2210_221085

theorem remainder_theorem (n : ℤ) : n % 7 = 3 → (5 * n - 12) % 7 = 3 := by
  sorry

end remainder_theorem_l2210_221085


namespace rectangle_perimeter_l2210_221086

theorem rectangle_perimeter (l w : ℝ) : 
  l + w = 7 → 
  2 * l + w = 9.5 → 
  2 * (l + w) = 14 := by
sorry

end rectangle_perimeter_l2210_221086


namespace s_equality_l2210_221058

theorem s_equality (x : ℝ) : 
  (x - 2)^4 + 4*(x - 2)^3 + 6*(x - 2)^2 + 4*(x - 2) + 1 = (x - 1)^4 := by
  sorry

end s_equality_l2210_221058


namespace pen_distribution_l2210_221040

theorem pen_distribution (num_pencils : ℕ) (num_students : ℕ) (num_pens : ℕ) : 
  num_pencils = 910 →
  num_students = 91 →
  num_pencils % num_students = 0 →
  num_pens % num_students = 0 →
  ∃ k : ℕ, num_pens = k * num_students :=
by sorry

end pen_distribution_l2210_221040


namespace linear_function_condition_l2210_221075

/-- Given a linear function f(x) = ax - x - a where a > 0 and a ≠ 1, 
    prove that a > 1. -/
theorem linear_function_condition (a : ℝ) 
  (h1 : a > 0) (h2 : a ≠ 1) : a > 1 := by
  sorry

end linear_function_condition_l2210_221075


namespace problem_statement_l2210_221037

theorem problem_statement : ((18^10 / 18^9)^3 * 16^3) / 8^6 = 91.125 := by sorry

end problem_statement_l2210_221037


namespace zeros_of_specific_f_graph_above_line_implies_b_gt_2_solution_set_when_b_eq_2_l2210_221099

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^2 + (2*a + 1) * x + b

-- Statement 1
theorem zeros_of_specific_f :
  ∃ (x₁ x₂ : ℝ), x₁ = -4 ∧ x₂ = 1 ∧
  ∀ (x : ℝ), f 1 (-4) x = 0 ↔ (x = x₁ ∨ x = x₂) :=
sorry

-- Statement 2
theorem graph_above_line_implies_b_gt_2 (a b : ℝ) :
  (∀ x : ℝ, f a b x > x + 2) → b > 2 :=
sorry

-- Statement 3
theorem solution_set_when_b_eq_2 (a : ℝ) :
  let S := {x : ℝ | f a 2 x < 0}
  if a < 0 then
    S = {x : ℝ | x < -2 ∨ x > -1/a}
  else if a = 0 then
    S = {x : ℝ | x < -2}
  else if 0 < a ∧ a < 1/2 then
    S = {x : ℝ | -1/a < x ∧ x < -2}
  else if a = 1/2 then
    S = ∅
  else -- a > 1/2
    S = {x : ℝ | -2 < x ∧ x < -1/a} :=
sorry

end zeros_of_specific_f_graph_above_line_implies_b_gt_2_solution_set_when_b_eq_2_l2210_221099


namespace expansion_properties_l2210_221049

theorem expansion_properties (x : ℝ) (n : ℕ) :
  (∃ k : ℝ, 2 * (n.choose 2) = (n.choose 1) + (n.choose 3) ∧ k ≠ 0) →
  (n = 7 ∧ ∀ r : ℕ, r ≤ n → (7 - 2*r ≠ 0)) :=
by sorry

end expansion_properties_l2210_221049


namespace video_game_points_l2210_221095

/-- The number of points earned for defeating one enemy in a video game -/
def points_per_enemy (total_enemies : ℕ) (enemies_defeated : ℕ) (total_points : ℕ) : ℚ :=
  total_points / enemies_defeated

theorem video_game_points :
  let total_enemies : ℕ := 7
  let enemies_defeated : ℕ := total_enemies - 2
  let total_points : ℕ := 40
  points_per_enemy total_enemies enemies_defeated total_points = 8 := by
  sorry

end video_game_points_l2210_221095


namespace cubic_is_odd_rhombus_diagonals_bisect_l2210_221080

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- Define the property of being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the property of diagonals bisecting each other
def diagonals_bisect (shape : Type) : Prop := 
  ∀ d1 d2 : shape → ℝ × ℝ, d1 ≠ d2 → ∃ p : ℝ × ℝ, 
    (∃ a b : shape, d1 a = p ∧ d2 b = p) ∧
    (∀ q : shape, d1 q = p ∨ d2 q = p)

-- Define shapes
class Parallelogram (shape : Type)
class Rhombus (shape : Type) extends Parallelogram shape

-- Theorem statements
theorem cubic_is_odd : is_odd_function f := sorry

theorem rhombus_diagonals_bisect (shape : Type) [Rhombus shape] : 
  diagonals_bisect shape := sorry

end cubic_is_odd_rhombus_diagonals_bisect_l2210_221080


namespace share_a_plus_c_equals_6952_l2210_221046

def total_money : ℕ := 15800
def ratio_a : ℕ := 5
def ratio_b : ℕ := 9
def ratio_c : ℕ := 6
def ratio_d : ℕ := 5

def total_ratio : ℕ := ratio_a + ratio_b + ratio_c + ratio_d

theorem share_a_plus_c_equals_6952 :
  (ratio_a + ratio_c) * (total_money / total_ratio) = 6952 := by
  sorry

end share_a_plus_c_equals_6952_l2210_221046


namespace not_monotone_decreasing_periodic_function_l2210_221007

-- Define a real-valued function on ℝ
variable (f : ℝ → ℝ)

-- Theorem 1: If f(1) > f(-1), then f is not monotonically decreasing on ℝ
theorem not_monotone_decreasing (h : f 1 > f (-1)) : 
  ¬ (∀ x y : ℝ, x ≤ y → f x ≥ f y) := by sorry

-- Theorem 2: If f(1+x) = f(x-1) for all x ∈ ℝ, then f is periodic
theorem periodic_function (h : ∀ x : ℝ, f (1 + x) = f (x - 1)) : 
  ∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x := by sorry

end not_monotone_decreasing_periodic_function_l2210_221007


namespace circle_intersection_condition_l2210_221045

/-- Circle B with equation x^2 + y^2 + b = 0 -/
def circle_B (b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + b = 0}

/-- Circle C with equation x^2 + y^2 - 6x + 8y + 16 = 0 -/
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 6*p.1 + 8*p.2 + 16 = 0}

/-- Two circles do not intersect -/
def no_intersection (A B : Set (ℝ × ℝ)) : Prop :=
  A ∩ B = ∅

/-- The main theorem -/
theorem circle_intersection_condition (b : ℝ) :
  no_intersection (circle_B b) circle_C →
  (-4 < b ∧ b < 0) ∨ b < -25 := by
  sorry

end circle_intersection_condition_l2210_221045


namespace sandwiches_prepared_correct_l2210_221047

/-- The number of sandwiches Ruth prepared -/
def sandwiches_prepared : ℕ := 10

/-- The number of sandwiches Ruth ate -/
def sandwiches_ruth_ate : ℕ := 1

/-- The number of sandwiches Ruth gave to her brother -/
def sandwiches_given_to_brother : ℕ := 2

/-- The number of sandwiches eaten by the first cousin -/
def sandwiches_first_cousin : ℕ := 2

/-- The number of sandwiches eaten by each of the other two cousins -/
def sandwiches_per_other_cousin : ℕ := 1

/-- The number of other cousins who ate sandwiches -/
def number_of_other_cousins : ℕ := 2

/-- The number of sandwiches left at the end -/
def sandwiches_left : ℕ := 3

/-- Theorem stating that the number of sandwiches Ruth prepared is correct -/
theorem sandwiches_prepared_correct : 
  sandwiches_prepared = 
    sandwiches_ruth_ate + 
    sandwiches_given_to_brother + 
    sandwiches_first_cousin + 
    (sandwiches_per_other_cousin * number_of_other_cousins) + 
    sandwiches_left :=
by sorry

end sandwiches_prepared_correct_l2210_221047


namespace negative_integer_sum_square_l2210_221044

theorem negative_integer_sum_square (N : ℤ) : 
  N < 0 → N^2 + N = -12 → N = -4 := by
  sorry

end negative_integer_sum_square_l2210_221044


namespace juans_number_problem_l2210_221033

theorem juans_number_problem (n : ℝ) : 
  (((n + 3) * 2 - 2) / 2 = 8) → n = 6 := by
  sorry

end juans_number_problem_l2210_221033


namespace sum_of_n_values_l2210_221038

theorem sum_of_n_values (m n : ℕ+) : 
  (1 : ℚ) / m + (1 : ℚ) / n = (1 : ℚ) / 5 →
  ∃ (n₁ n₂ n₃ : ℕ+), 
    (∀ k : ℕ+, ((1 : ℚ) / m + (1 : ℚ) / k = (1 : ℚ) / 5) → (k = n₁ ∨ k = n₂ ∨ k = n₃)) ∧
    n₁.val + n₂.val + n₃.val = 46 :=
by sorry

end sum_of_n_values_l2210_221038


namespace lewis_earnings_l2210_221013

theorem lewis_earnings (weeks : ℕ) (weekly_rent : ℚ) (total_after_rent : ℚ) 
  (h1 : weeks = 233)
  (h2 : weekly_rent = 49)
  (h3 : total_after_rent = 93899) :
  (total_after_rent + weeks * weekly_rent) / weeks = 451.99 := by
sorry

end lewis_earnings_l2210_221013


namespace no_real_roots_l2210_221070

-- Define the operation ⊕
def oplus (m n : ℝ) : ℝ := n^2 - m*n + 1

-- Theorem statement
theorem no_real_roots :
  ∀ x : ℝ, oplus 1 x ≠ 0 := by
  sorry

end no_real_roots_l2210_221070


namespace sin_double_angle_l2210_221067

theorem sin_double_angle (x : Real) (h : Real.sin (x - π/4) = 2/3) : 
  Real.sin (2*x) = 1/9 := by
  sorry

end sin_double_angle_l2210_221067


namespace library_book_count_l2210_221020

/-- The number of books the library had before the grant -/
def initial_books : ℕ := 5935

/-- The number of books purchased with the grant -/
def purchased_books : ℕ := 2647

/-- The total number of books after the grant -/
def total_books : ℕ := initial_books + purchased_books

theorem library_book_count : total_books = 8582 := by
  sorry

end library_book_count_l2210_221020


namespace alcohol_volume_bound_l2210_221010

/-- Represents the volume of pure alcohol in container B after n operations -/
def alcohol_volume (x y z : ℝ) (n : ℕ+) : ℝ :=
  sorry

/-- Theorem stating that the volume of pure alcohol in container B 
    is always less than or equal to xy/(x+y) -/
theorem alcohol_volume_bound (x y z : ℝ) (n : ℕ+) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hxy : x < z) (hyz : y < z) :
  alcohol_volume x y z n ≤ (x * y) / (x + y) :=
sorry

end alcohol_volume_bound_l2210_221010


namespace complex_magnitude_product_l2210_221090

theorem complex_magnitude_product : 
  Complex.abs ((3 * Real.sqrt 3 - 3 * Complex.I) * (2 * Real.sqrt 2 + 2 * Complex.I)) = 12 * Real.sqrt 3 := by
  sorry

end complex_magnitude_product_l2210_221090


namespace triangle_area_l2210_221076

/-- Given a triangle with sides 6, 8, and 10, its area is 24 -/
theorem triangle_area (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  (1/2) * a * b = 24 := by
  sorry

end triangle_area_l2210_221076


namespace perpendicular_line_equation_l2210_221098

/-- Given line passing through (1,2) and perpendicular to 2x - 6y + 1 = 0 -/
def given_line (x y : ℝ) : Prop := 2 * x - 6 * y + 1 = 0

/-- Point that the perpendicular line passes through -/
def point : ℝ × ℝ := (1, 2)

/-- Equation of the perpendicular line -/
def perpendicular_line (x y : ℝ) : Prop := 3 * x + y - 5 = 0

/-- Theorem stating that the perpendicular line passing through (1,2) 
    has the equation 3x + y - 5 = 0 -/
theorem perpendicular_line_equation : 
  ∀ x y : ℝ, given_line x y → 
  (perpendicular_line x y ↔ 
   (perpendicular_line point.1 point.2 ∧ 
    (x - point.1) * (x - point.1) + (y - point.2) * (y - point.2) = 
    ((x - 1) * 2 + (y - 2) * (-6))^2 / (2^2 + (-6)^2))) :=
sorry

end perpendicular_line_equation_l2210_221098


namespace euclid_wrote_elements_l2210_221052

/-- The author of "Elements" -/
def author_of_elements : String := "Euclid"

/-- Theorem stating that Euclid is the author of "Elements" -/
theorem euclid_wrote_elements : author_of_elements = "Euclid" := by sorry

end euclid_wrote_elements_l2210_221052


namespace opposite_pairs_l2210_221096

theorem opposite_pairs :
  (∀ x : ℝ, -|x| = -x ∧ -(-x) = x) ∧
  (-|-3| = -(-(-3))) ∧
  (3 ≠ -|-3|) ∧
  (-3 ≠ -(-1/3)) ∧
  (-3 ≠ -(1/3)) :=
by sorry

end opposite_pairs_l2210_221096
