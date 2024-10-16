import Mathlib

namespace NUMINAMATH_CALUDE_not_proper_subset_of_itself_l1673_167313

def main_set : Set ℕ := {1, 2, 3}

theorem not_proper_subset_of_itself : ¬(main_set ⊂ main_set) := by
  sorry

end NUMINAMATH_CALUDE_not_proper_subset_of_itself_l1673_167313


namespace NUMINAMATH_CALUDE_max_digit_sum_2016_l1673_167316

/-- A function that sums the digits of a natural number -/
def sumDigits (n : ℕ) : ℕ := sorry

/-- A function that repeatedly sums the digits until a single digit is obtained -/
def repeatSumDigits (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a natural number has exactly 2016 digits -/
def has2016Digits (n : ℕ) : Prop := sorry

theorem max_digit_sum_2016 :
  ∀ n : ℕ, has2016Digits n → repeatSumDigits n ≤ 9 ∧ ∃ m : ℕ, has2016Digits m ∧ repeatSumDigits m = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_digit_sum_2016_l1673_167316


namespace NUMINAMATH_CALUDE_carton_height_theorem_l1673_167323

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

theorem carton_height_theorem (carton : BoxDimensions) (soap : BoxDimensions) 
    (h : carton.height > 0)
    (carton_dim : carton.length = 25 ∧ carton.width = 48)
    (soap_dim : soap.length = 8 ∧ soap.width = 6 ∧ soap.height = 5)
    (max_boxes : ℕ)
    (max_boxes_def : max_boxes = 300)
    (carton_capacity : boxVolume carton = max_boxes * boxVolume soap) :
  carton.height = 60 := by
sorry

end NUMINAMATH_CALUDE_carton_height_theorem_l1673_167323


namespace NUMINAMATH_CALUDE_hundred_days_after_wednesday_is_friday_l1673_167331

/-- Enumeration of days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to get the day of the week after a given number of days -/
def dayAfter (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match days with
  | 0 => start
  | n + 1 => nextDay (dayAfter start n)

/-- Theorem stating that 100 days after Wednesday is Friday -/
theorem hundred_days_after_wednesday_is_friday :
  dayAfter DayOfWeek.Wednesday 100 = DayOfWeek.Friday := by
  sorry


end NUMINAMATH_CALUDE_hundred_days_after_wednesday_is_friday_l1673_167331


namespace NUMINAMATH_CALUDE_cost_to_feed_chickens_is_60_l1673_167356

/-- Calculates the cost to feed chickens given the total number of birds and the ratio of bird types -/
def cost_to_feed_chickens (total_birds : ℕ) (duck_ratio parrot_ratio chicken_ratio : ℕ) (chicken_feed_cost : ℚ) : ℚ :=
  let total_ratio := duck_ratio + parrot_ratio + chicken_ratio
  let birds_per_ratio := total_birds / total_ratio
  let num_chickens := birds_per_ratio * chicken_ratio
  (num_chickens : ℚ) * chicken_feed_cost

/-- Theorem stating that with given conditions, the cost to feed chickens is $60 -/
theorem cost_to_feed_chickens_is_60 :
  cost_to_feed_chickens 60 2 3 5 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_cost_to_feed_chickens_is_60_l1673_167356


namespace NUMINAMATH_CALUDE_min_omega_value_l1673_167354

open Real

theorem min_omega_value (f : ℝ → ℝ) (ω φ : ℝ) (x₀ : ℝ) :
  (ω > 0) →
  (∀ x, f x = sin (ω * x + φ)) →
  (∀ x, f x₀ ≤ f x ∧ f x ≤ f (x₀ + 2016 * π)) →
  (∀ ω' > 0, (∀ x, f x₀ ≤ sin (ω' * x + φ) ∧ sin (ω' * x + φ) ≤ f (x₀ + 2016 * π)) → ω ≤ ω') →
  ω = 1 / 2016 :=
sorry

end NUMINAMATH_CALUDE_min_omega_value_l1673_167354


namespace NUMINAMATH_CALUDE_meal_center_allocation_l1673_167398

/-- Represents the meal center's soup can allocation problem -/
theorem meal_center_allocation (total_cans : ℕ) (adults_per_can children_per_can : ℕ) 
  (children_to_feed : ℕ) (adults_fed : ℕ) :
  total_cans = 10 →
  adults_per_can = 4 →
  children_per_can = 7 →
  children_to_feed = 21 →
  adults_fed = (total_cans - (children_to_feed / children_per_can)) * adults_per_can →
  adults_fed = 28 := by
sorry

end NUMINAMATH_CALUDE_meal_center_allocation_l1673_167398


namespace NUMINAMATH_CALUDE_binary_to_decimal_101101_l1673_167325

theorem binary_to_decimal_101101 : 
  (1 * 2^5 + 0 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 45 := by
  sorry

end NUMINAMATH_CALUDE_binary_to_decimal_101101_l1673_167325


namespace NUMINAMATH_CALUDE_soap_box_height_is_five_l1673_167350

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Theorem: Given the carton and soap box dimensions, and the maximum number of soap boxes,
    the height of the soap box must be 5 inches -/
theorem soap_box_height_is_five
  (carton : BoxDimensions)
  (soap : BoxDimensions)
  (max_boxes : ℕ)
  (h_carton_length : carton.length = 25)
  (h_carton_width : carton.width = 48)
  (h_carton_height : carton.height = 60)
  (h_soap_length : soap.length = 8)
  (h_soap_width : soap.width = 6)
  (h_max_boxes : max_boxes = 300)
  (h_fit : max_boxes * boxVolume soap = boxVolume carton) :
  soap.height = 5 := by
  sorry

end NUMINAMATH_CALUDE_soap_box_height_is_five_l1673_167350


namespace NUMINAMATH_CALUDE_days_before_reinforcement_l1673_167332

/-- Proves that the number of days before reinforcement arrived is 12 --/
theorem days_before_reinforcement 
  (initial_garrison : ℕ) 
  (initial_provision_days : ℕ) 
  (reinforcement : ℕ) 
  (remaining_provision_days : ℕ) 
  (h1 : initial_garrison = 1850)
  (h2 : initial_provision_days = 28)
  (h3 : reinforcement = 1110)
  (h4 : remaining_provision_days = 10) :
  (initial_garrison * initial_provision_days - 
   (initial_garrison + reinforcement) * remaining_provision_days) / initial_garrison = 12 :=
by sorry

end NUMINAMATH_CALUDE_days_before_reinforcement_l1673_167332


namespace NUMINAMATH_CALUDE_mn_equation_solutions_l1673_167315

theorem mn_equation_solutions (m n : ℤ) : 
  m^2 * n^2 + m^2 + n^2 + 10*m*n + 16 = 0 ↔ (m = 2 ∧ n = -2) ∨ (m = -2 ∧ n = 2) :=
by sorry

end NUMINAMATH_CALUDE_mn_equation_solutions_l1673_167315


namespace NUMINAMATH_CALUDE_candy_problem_l1673_167399

theorem candy_problem (initial_candies : ℕ) : 
  let day1_remaining := initial_candies / 2
  let day2_remaining := day1_remaining / 3 * 2
  let day3_remaining := day2_remaining / 4 * 3
  let day4_remaining := day3_remaining / 5 * 4
  let day5_remaining := day4_remaining / 6 * 5
  day5_remaining = 1 → initial_candies = 720 :=
by sorry

end NUMINAMATH_CALUDE_candy_problem_l1673_167399


namespace NUMINAMATH_CALUDE_complex_division_problem_l1673_167347

theorem complex_division_problem (i : ℂ) (h : i^2 = -1) :
  2 / (1 + i) = 1 - i := by sorry

end NUMINAMATH_CALUDE_complex_division_problem_l1673_167347


namespace NUMINAMATH_CALUDE_valid_outfit_count_l1673_167317

/-- The number of types of each item (shirt, pants, hat, shoe) -/
def item_types : ℕ := 6

/-- The number of colors available -/
def colors : ℕ := 6

/-- The number of items in an outfit -/
def outfit_items : ℕ := 4

/-- The total number of possible outfits -/
def total_outfits : ℕ := item_types ^ outfit_items

/-- The number of outfits with all items of the same color -/
def same_color_outfits : ℕ := colors

/-- The number of valid outfit combinations -/
def valid_outfits : ℕ := total_outfits - same_color_outfits

theorem valid_outfit_count : valid_outfits = 1290 := by
  sorry

end NUMINAMATH_CALUDE_valid_outfit_count_l1673_167317


namespace NUMINAMATH_CALUDE_polynomial_equality_l1673_167314

/-- Given that 4x^4 + 8x^3 + g(x) = 2x^4 - 5x^3 + 7x + 4,
    prove that g(x) = -2x^4 - 13x^3 + 7x + 4 -/
theorem polynomial_equality (x : ℝ) (g : ℝ → ℝ) 
    (h : ∀ x, 4 * x^4 + 8 * x^3 + g x = 2 * x^4 - 5 * x^3 + 7 * x + 4) :
  g x = -2 * x^4 - 13 * x^3 + 7 * x + 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1673_167314


namespace NUMINAMATH_CALUDE_davids_remaining_money_l1673_167322

theorem davids_remaining_money (initial_amount spent_amount remaining_amount : ℕ) :
  initial_amount = 1800 →
  remaining_amount = spent_amount - 800 →
  initial_amount - spent_amount = remaining_amount →
  remaining_amount = 500 := by
sorry

end NUMINAMATH_CALUDE_davids_remaining_money_l1673_167322


namespace NUMINAMATH_CALUDE_range_of_a_l1673_167378

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x a : ℝ) : Prop := x > a

-- Define the property that ¬p is sufficient but not necessary for ¬q
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, ¬(p x) → ¬(q x a)) ∧ (∃ x, ¬(q x a) ∧ p x)

-- Theorem statement
theorem range_of_a :
  (∀ a : ℝ, sufficient_not_necessary a) → (∀ a : ℝ, a ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1673_167378


namespace NUMINAMATH_CALUDE_larry_cards_remaining_l1673_167393

/-- Given that Larry initially has 67 cards and Dennis takes 9 cards away,
    prove that Larry now has 58 cards. -/
theorem larry_cards_remaining (initial_cards : ℕ) (cards_taken : ℕ) : 
  initial_cards = 67 → cards_taken = 9 → initial_cards - cards_taken = 58 := by
  sorry

end NUMINAMATH_CALUDE_larry_cards_remaining_l1673_167393


namespace NUMINAMATH_CALUDE_square_difference_pattern_l1673_167338

theorem square_difference_pattern (n : ℕ) :
  (2*n + 2)^2 - (2*n)^2 = 4*(2*n + 1) := by
  sorry

end NUMINAMATH_CALUDE_square_difference_pattern_l1673_167338


namespace NUMINAMATH_CALUDE_range_of_m_l1673_167321

/-- Given the conditions:
    1. p: |4-x| ≤ 6
    2. q: x^2 - 2x + 1 ≤ 0 (m > 0)
    3. p is not a necessary but not sufficient condition for q
    
    Prove that the range of values for the real number m is m ≥ 9. -/
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, |4 - x| ≤ 6 → (x^2 - 2*x + 1 ≤ 0 ∧ m > 0)) →
  (∃ x : ℝ, |4 - x| ≤ 6 ∧ (x^2 - 2*x + 1 > 0 ∨ m ≤ 0)) →
  (∀ x : ℝ, (x^2 - 2*x + 1 ≤ 0 ∧ m > 0) → |4 - x| ≤ 6) →
  m ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1673_167321


namespace NUMINAMATH_CALUDE_total_daily_salary_l1673_167311

def grocery_store_salaries (manager_salary clerk_salary : ℕ) (num_managers num_clerks : ℕ) : ℕ :=
  manager_salary * num_managers + clerk_salary * num_clerks

theorem total_daily_salary :
  grocery_store_salaries 5 2 2 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_daily_salary_l1673_167311


namespace NUMINAMATH_CALUDE_jessica_roses_thrown_away_l1673_167308

/-- The number of roses Jessica threw away -/
def roses_thrown_away (initial : ℕ) (added : ℕ) (final : ℕ) : ℕ :=
  initial + added - final

/-- Proof that Jessica threw away 4 roses -/
theorem jessica_roses_thrown_away :
  roses_thrown_away 2 25 23 = 4 := by
  sorry

end NUMINAMATH_CALUDE_jessica_roses_thrown_away_l1673_167308


namespace NUMINAMATH_CALUDE_arithmetic_sequence_and_equation_l1673_167330

-- Define arithmetic sequence
def is_arithmetic_sequence (a b c : ℝ) : Prop :=
  b - a = c - b

-- Define the equation from proposition B
def satisfies_equation (a b c : ℝ) : Prop :=
  b ≠ 0 ∧ a / b + c / b = 2

-- Theorem statement
theorem arithmetic_sequence_and_equation :
  (∀ a b c : ℝ, satisfies_equation a b c → is_arithmetic_sequence a b c) ∧
  (∃ a b c : ℝ, is_arithmetic_sequence a b c ∧ ¬satisfies_equation a b c) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_and_equation_l1673_167330


namespace NUMINAMATH_CALUDE_number_solution_l1673_167383

theorem number_solution : ∃ x : ℝ, (45 - 3 * x = 18) ∧ (x = 9) := by sorry

end NUMINAMATH_CALUDE_number_solution_l1673_167383


namespace NUMINAMATH_CALUDE_rock_sale_price_per_pound_l1673_167368

theorem rock_sale_price_per_pound 
  (average_weight : ℝ) 
  (num_rocks : ℕ) 
  (total_sale : ℝ) 
  (h1 : average_weight = 1.5)
  (h2 : num_rocks = 10)
  (h3 : total_sale = 60) :
  total_sale / (average_weight * num_rocks) = 4 := by
sorry

end NUMINAMATH_CALUDE_rock_sale_price_per_pound_l1673_167368


namespace NUMINAMATH_CALUDE_train_length_l1673_167359

/-- Calculates the length of a train given its speed and the time it takes to cross a platform of known length. -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 72 * 1000 / 3600 →
  platform_length = 250 →
  crossing_time = 15 →
  (train_speed * crossing_time) - platform_length = 50 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l1673_167359


namespace NUMINAMATH_CALUDE_squirrel_nut_division_l1673_167362

theorem squirrel_nut_division (n : ℕ) : ¬(5 ∣ (2022 + n * (n + 1))) := by
  sorry

end NUMINAMATH_CALUDE_squirrel_nut_division_l1673_167362


namespace NUMINAMATH_CALUDE_polygon_interior_angle_sum_l1673_167305

/-- A polygon where each exterior angle is 36° has a sum of interior angles equal to 1440°. -/
theorem polygon_interior_angle_sum (n : ℕ) (h : n * 36 = 360) : 
  (n - 2) * 180 = 1440 :=
sorry

end NUMINAMATH_CALUDE_polygon_interior_angle_sum_l1673_167305


namespace NUMINAMATH_CALUDE_inequality_proof_l1673_167372

-- Define the set M
def M : Set ℝ := {x | 0 < |x + 2| - |1 - x| ∧ |x + 2| - |1 - x| < 2}

-- State the theorem
theorem inequality_proof (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (|a + 1/2 * b| < 3/4) ∧ (|4 * a * b - 1| > 2 * |b - a|) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1673_167372


namespace NUMINAMATH_CALUDE_inequality_implies_range_l1673_167391

theorem inequality_implies_range (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → 4^x - 2^(x+1) - a ≤ 0) →
  a ≥ 8 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_range_l1673_167391


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1673_167351

/-- Arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ) (d : ℝ) (m : ℕ)
  (h_arith : arithmetic_sequence a d)
  (h_d_neq_0 : d ≠ 0)
  (h_sum : a 3 + a 6 + a 10 + a 13 = 32)
  (h_am : a m = 8) :
  m = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1673_167351


namespace NUMINAMATH_CALUDE_distance_between_homes_correct_l1673_167373

/-- The distance between Maxwell's and Brad's homes -/
def distance_between_homes : ℝ := 34

/-- Maxwell's walking speed in km/h -/
def maxwell_speed : ℝ := 4

/-- Brad's running speed in km/h -/
def brad_speed : ℝ := 6

/-- Time difference between Maxwell's start and Brad's start in hours -/
def time_difference : ℝ := 1

/-- Total time until Maxwell and Brad meet in hours -/
def total_time : ℝ := 4

/-- Theorem stating that the distance between homes is correct given the conditions -/
theorem distance_between_homes_correct :
  distance_between_homes = 
    maxwell_speed * total_time + 
    brad_speed * (total_time - time_difference) := by
  sorry

#check distance_between_homes_correct

end NUMINAMATH_CALUDE_distance_between_homes_correct_l1673_167373


namespace NUMINAMATH_CALUDE_function_decreasing_and_inequality_l1673_167346

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / log x - a * x

theorem function_decreasing_and_inequality (e : ℝ) (h_e : exp 1 = e) :
  (∀ a : ℝ, (∀ x : ℝ, x > 1 → (deriv (f a)) x ≤ 0) → a ≥ 1/4) ∧
  (∀ a : ℝ, (∃ x₁ x₂ : ℝ, e ≤ x₁ ∧ x₁ ≤ e^2 ∧ e ≤ x₂ ∧ x₂ ≤ e^2 ∧
    f a x₁ - (deriv (f a)) x₂ ≤ a) → a ≥ 1/2 - 1/(4*e^2)) :=
by sorry

end NUMINAMATH_CALUDE_function_decreasing_and_inequality_l1673_167346


namespace NUMINAMATH_CALUDE_product_mod_23_l1673_167343

theorem product_mod_23 : (191 * 193 * 197) % 23 = 14 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_23_l1673_167343


namespace NUMINAMATH_CALUDE_winnie_yesterday_repetitions_l1673_167335

/-- The number of repetitions Winnie completed yesterday -/
def yesterday_repetitions : ℕ := 86

/-- The number of repetitions Winnie completed today -/
def today_repetitions : ℕ := 73

/-- The number of repetitions Winnie fell behind by today -/
def difference : ℕ := 13

/-- Theorem: Winnie completed 86 repetitions yesterday -/
theorem winnie_yesterday_repetitions :
  yesterday_repetitions = today_repetitions + difference :=
by sorry

end NUMINAMATH_CALUDE_winnie_yesterday_repetitions_l1673_167335


namespace NUMINAMATH_CALUDE_multiply_powers_of_ten_l1673_167365

theorem multiply_powers_of_ten : (-2 * 10^4) * (4 * 10^5) = -8 * 10^9 := by
  sorry

end NUMINAMATH_CALUDE_multiply_powers_of_ten_l1673_167365


namespace NUMINAMATH_CALUDE_longest_side_of_triangle_with_incircle_l1673_167326

/-- A triangle with an incircle -/
structure TriangleWithIncircle where
  /-- The radius of the incircle -/
  r : ℝ
  /-- The length of the segment of the side divided by the tangent point -/
  a : ℝ
  /-- The length of the other segment of the side divided by the tangent point -/
  b : ℝ
  /-- The length of the longest side of the triangle -/
  longest_side : ℝ

/-- Theorem: In a triangle with an incircle of radius 5 units, where the incircle is tangent
    to one side at a point dividing it into segments of 9 and 5 units, the length of the
    longest side is 18 units. -/
theorem longest_side_of_triangle_with_incircle
  (t : TriangleWithIncircle)
  (h1 : t.r = 5)
  (h2 : t.a = 9)
  (h3 : t.b = 5) :
  t.longest_side = 18 := by
  sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_with_incircle_l1673_167326


namespace NUMINAMATH_CALUDE_coconut_yield_for_six_trees_l1673_167329

/-- The yield of x trees in a coconut grove --/
def coconut_grove_yield (x : ℕ) (Y : ℕ) : Prop :=
  let total_trees := 3 * x
  let total_yield := (x + 3) * 60 + x * Y + (x - 3) * 180
  (total_yield : ℚ) / total_trees = 100

theorem coconut_yield_for_six_trees :
  coconut_grove_yield 6 120 :=
sorry

end NUMINAMATH_CALUDE_coconut_yield_for_six_trees_l1673_167329


namespace NUMINAMATH_CALUDE_shirts_washed_l1673_167371

theorem shirts_washed (short_sleeve : ℕ) (long_sleeve : ℕ) (unwashed : ℕ) :
  short_sleeve = 9 →
  long_sleeve = 21 →
  unwashed = 1 →
  short_sleeve + long_sleeve - unwashed = 29 := by
sorry

end NUMINAMATH_CALUDE_shirts_washed_l1673_167371


namespace NUMINAMATH_CALUDE_contractor_wage_l1673_167333

/-- Contractor's wage problem -/
theorem contractor_wage
  (total_days : ℕ)
  (absent_days : ℕ)
  (daily_fine : ℚ)
  (total_amount : ℚ)
  (h1 : total_days = 30)
  (h2 : absent_days = 10)
  (h3 : daily_fine = 7.5)
  (h4 : total_amount = 425)
  : ∃ (daily_wage : ℚ),
    daily_wage * (total_days - absent_days : ℚ) - daily_fine * absent_days = total_amount ∧
    daily_wage = 25 := by
  sorry

end NUMINAMATH_CALUDE_contractor_wage_l1673_167333


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1673_167320

theorem fraction_to_decimal (n d : ℕ) (h : d ≠ 0) :
  (n : ℚ) / d = 0.35625 ↔ n = 57 ∧ d = 160 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1673_167320


namespace NUMINAMATH_CALUDE_greatest_common_remainder_l1673_167301

theorem greatest_common_remainder (a b c : ℕ) (h : a = 25 ∧ b = 57 ∧ c = 105) :
  ∃ (k : ℕ), k > 0 ∧ 
    (∃ (r : ℕ), a % k = r ∧ b % k = r ∧ c % k = r) ∧
    (∀ (m : ℕ), m > k → ¬(∃ (s : ℕ), a % m = s ∧ b % m = s ∧ c % m = s)) ∧
  k = 16 := by
sorry

end NUMINAMATH_CALUDE_greatest_common_remainder_l1673_167301


namespace NUMINAMATH_CALUDE_weeks_to_afford_bicycle_l1673_167389

def bicycle_cost : ℕ := 600
def birthday_money : ℕ := 165
def weekly_earnings : ℕ := 20

theorem weeks_to_afford_bicycle :
  let total_money : ℕ → ℕ := λ weeks => birthday_money + weekly_earnings * weeks
  ∀ weeks : ℕ, total_money weeks ≥ bicycle_cost → weeks ≥ 22 :=
by
  sorry

end NUMINAMATH_CALUDE_weeks_to_afford_bicycle_l1673_167389


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1673_167304

-- Define the sets A and B
def A : Set ℝ := {1, 2, 3, 4}
def B : Set ℝ := {x | -1 < x ∧ x < 3}

-- State the theorem
theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | -1 < x ∧ x ≤ 3 ∨ x = 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1673_167304


namespace NUMINAMATH_CALUDE_fraction_simplification_l1673_167381

theorem fraction_simplification :
  (1 / 2 + 1 / 3) / (3 / 7 - 1 / 5) = 175 / 48 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1673_167381


namespace NUMINAMATH_CALUDE_modified_cube_vertices_l1673_167366

/-- Calculates the number of vertices in a modified cube -/
def modifiedCubeVertices (initialSideLength : ℕ) (removedSideLength : ℕ) : ℕ :=
  8 * (3 * 4 - 3)

/-- Theorem stating that a cube of side length 5 with smaller cubes of side length 2 
    removed from each corner has 64 vertices -/
theorem modified_cube_vertices :
  modifiedCubeVertices 5 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_modified_cube_vertices_l1673_167366


namespace NUMINAMATH_CALUDE_find_set_B_l1673_167367

universe u

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}

theorem find_set_B (A B : Set Nat) 
  (h1 : (U \ (A ∪ B)) = {1, 3})
  (h2 : (A ∩ (U \ B)) = {2, 5}) :
  B = {4, 6, 7} := by
  sorry

end NUMINAMATH_CALUDE_find_set_B_l1673_167367


namespace NUMINAMATH_CALUDE_missy_yells_84_times_l1673_167337

/-- The number of times Missy yells at her obedient dog -/
def obedient_yells : ℕ := 12

/-- The ratio of yells at the stubborn dog compared to the obedient dog -/
def stubborn_ratio : ℕ := 4

/-- The ratio of yells at the mischievous dog compared to the obedient dog -/
def mischievous_ratio : ℕ := 2

/-- The total number of times Missy yells at all three dogs -/
def total_yells : ℕ := obedient_yells + stubborn_ratio * obedient_yells + mischievous_ratio * obedient_yells

theorem missy_yells_84_times : total_yells = 84 := by
  sorry

end NUMINAMATH_CALUDE_missy_yells_84_times_l1673_167337


namespace NUMINAMATH_CALUDE_correct_calculation_l1673_167355

theorem correct_calculation (x : ℝ) : 
  (x + 2.95 = 9.28) → (x - 2.95 = 3.38) :=
by sorry

end NUMINAMATH_CALUDE_correct_calculation_l1673_167355


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l1673_167370

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be represented in scientific notation -/
def number : ℕ := 12500

/-- The scientific notation representation of the number -/
def scientificForm : ScientificNotation := {
  coefficient := 1.25,
  exponent := 4,
  coeff_range := by sorry
}

/-- Theorem stating that the scientific notation form is equal to the original number -/
theorem scientific_notation_correct : 
  (scientificForm.coefficient * (10 : ℝ) ^ scientificForm.exponent) = number := by sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l1673_167370


namespace NUMINAMATH_CALUDE_isosceles_triangle_from_equation_l1673_167377

/-- Given a triangle ABC with sides a and b, and angles A and B,
    if the equation x^2 - (b cos A)x + a cos B = 0 has roots whose
    product equals their sum, then the triangle is isosceles. -/
theorem isosceles_triangle_from_equation (a b : ℝ) (A B : ℝ) :
  (∃ (x y : ℝ), x^2 - (b * Real.cos A) * x + a * Real.cos B = 0 ∧
                 x * y = x + y) →
  (a > 0 ∧ b > 0 ∧ 0 < A ∧ A < π ∧ 0 < B ∧ B < π) →
  a = b ∨ A = B :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_from_equation_l1673_167377


namespace NUMINAMATH_CALUDE_millet_majority_day_l1673_167353

def seed_mix : ℝ := 0.25
def millet_eaten_daily : ℝ := 0.25
def total_seeds_daily : ℝ := 1

def millet_proportion (n : ℕ) : ℝ := 1 - (1 - seed_mix)^n

theorem millet_majority_day :
  ∀ k : ℕ, k < 5 → millet_proportion k ≤ 0.5 ∧
  millet_proportion 5 > 0.5 := by sorry

end NUMINAMATH_CALUDE_millet_majority_day_l1673_167353


namespace NUMINAMATH_CALUDE_derivative_at_alpha_l1673_167382

open Real

theorem derivative_at_alpha (α : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 2 * cos α - sin x
  HasDerivAt f (-cos α) α := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_alpha_l1673_167382


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_180_l1673_167328

def sum_of_divisors (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_sum_of_divisors_180 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ sum_of_divisors 180 ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ sum_of_divisors 180 → q ≤ p ∧ p = 13 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_180_l1673_167328


namespace NUMINAMATH_CALUDE_geometry_class_size_l1673_167388

theorem geometry_class_size :
  ∀ (total_students : ℕ),
  (2 : ℚ) / 3 * total_students = total_boys →
  (3 : ℚ) / 4 * total_boys = boys_under_6_feet →
  boys_under_6_feet = 19 →
  total_students = 38 :=
by
  sorry

end NUMINAMATH_CALUDE_geometry_class_size_l1673_167388


namespace NUMINAMATH_CALUDE_range_of_m_for_negative_f_solution_sets_for_inequality_l1673_167344

-- Define the function f(x) = mx^2 - mx - 1
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - m * x - 1

-- Part 1: Range of m for which f(x) < 0 for all x ∈ ℝ
theorem range_of_m_for_negative_f :
  ∀ m : ℝ, (∀ x : ℝ, f m x < 0) ↔ m ∈ Set.Ioc (-4) 0 :=
sorry

-- Part 2: Solution sets for the inequality f(x) < (1-m)x - 1
theorem solution_sets_for_inequality :
  ∀ m : ℝ,
    (m = 0 → {x : ℝ | f m x < (1 - m) * x - 1} = {x : ℝ | x > 0}) ∧
    (m > 0 → {x : ℝ | f m x < (1 - m) * x - 1} = {x : ℝ | 0 < x ∧ x < 1 / m}) ∧
    (m < 0 → {x : ℝ | f m x < (1 - m) * x - 1} = {x : ℝ | x < 1 / m ∨ x > 0}) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_for_negative_f_solution_sets_for_inequality_l1673_167344


namespace NUMINAMATH_CALUDE_sector_to_cone_sector_forms_cone_l1673_167363

theorem sector_to_cone (sector_angle : Real) (sector_radius : Real) 
  (base_radius : Real) (slant_height : Real) : Prop :=
  sector_angle = 240 ∧ 
  sector_radius = 12 ∧
  base_radius = 8 ∧
  slant_height = 12 ∧
  sector_angle / 360 * (2 * Real.pi * sector_radius) = 2 * Real.pi * base_radius ∧
  slant_height = sector_radius

theorem sector_forms_cone : 
  ∃ (sector_angle : Real) (sector_radius : Real) 
     (base_radius : Real) (slant_height : Real),
  sector_to_cone sector_angle sector_radius base_radius slant_height := by
  sorry

end NUMINAMATH_CALUDE_sector_to_cone_sector_forms_cone_l1673_167363


namespace NUMINAMATH_CALUDE_total_jeans_purchased_l1673_167306

-- Define the regular prices
def fox_price : ℝ := 15
def pony_price : ℝ := 18

-- Define the number of pairs purchased
def fox_pairs : ℕ := 3
def pony_pairs : ℕ := 2

-- Define the total savings
def total_savings : ℝ := 8.64

-- Define the sum of discount rates
def total_discount_rate : ℝ := 0.22

-- Define the Pony jeans discount rate
def pony_discount_rate : ℝ := 0.13999999999999993

-- Theorem statement
theorem total_jeans_purchased :
  fox_pairs + pony_pairs = 5 := by sorry

end NUMINAMATH_CALUDE_total_jeans_purchased_l1673_167306


namespace NUMINAMATH_CALUDE_possible_values_of_p_l1673_167349

theorem possible_values_of_p (a b c p : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_eq : a + 1/b = p ∧ b + 1/c = p ∧ c + 1/a = p) :
  p = 1 ∨ p = -1 := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_p_l1673_167349


namespace NUMINAMATH_CALUDE_chemical_mixture_percentage_l1673_167342

theorem chemical_mixture_percentage (initial_volume : ℝ) (initial_percentage : ℝ) (added_volume : ℝ) :
  initial_volume = 80 →
  initial_percentage = 0.3 →
  added_volume = 20 →
  let final_volume := initial_volume + added_volume
  let initial_x_volume := initial_volume * initial_percentage
  let final_x_volume := initial_x_volume + added_volume
  final_x_volume / final_volume = 0.44 := by
  sorry

end NUMINAMATH_CALUDE_chemical_mixture_percentage_l1673_167342


namespace NUMINAMATH_CALUDE_brand_d_highest_sales_l1673_167392

/-- Represents the sales volume of a brand -/
structure BrandSales where
  name : String
  sales : ℕ

/-- Theorem: Brand D has the highest sales volume -/
theorem brand_d_highest_sales (total : ℕ) (a b c d : BrandSales) :
  total = 100 ∧
  a.name = "A" ∧ a.sales = 15 ∧
  b.name = "B" ∧ b.sales = 30 ∧
  c.name = "C" ∧ c.sales = 12 ∧
  d.name = "D" ∧ d.sales = 43 →
  d.sales ≥ a.sales ∧ d.sales ≥ b.sales ∧ d.sales ≥ c.sales :=
by sorry

end NUMINAMATH_CALUDE_brand_d_highest_sales_l1673_167392


namespace NUMINAMATH_CALUDE_rectangle_y_value_l1673_167303

/-- Rectangle EFGH with vertices E(0, 0), F(0, 5), G(y, 5), and H(y, 0) -/
structure Rectangle where
  y : ℝ
  h_positive : y > 0

/-- The area of rectangle EFGH is 40 square units -/
def area (r : Rectangle) : ℝ := 5 * r.y

theorem rectangle_y_value (r : Rectangle) (h_area : area r = 40) : r.y = 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l1673_167303


namespace NUMINAMATH_CALUDE_expression_equivalence_l1673_167309

theorem expression_equivalence (a b c m n p : ℝ) 
  (h : a / m + (b * c + n * p) / (b * p + c * n) = 0) :
  b / n + (a * c + m * p) / (a * p + c * m) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_equivalence_l1673_167309


namespace NUMINAMATH_CALUDE_savings_increase_percentage_l1673_167360

/-- Represents the financial situation of a man over two years --/
structure FinancialSituation where
  /-- Income in the first year --/
  income : ℝ
  /-- Savings rate in the first year (as a decimal) --/
  savingsRate : ℝ
  /-- Income increase rate in the second year (as a decimal) --/
  incomeIncreaseRate : ℝ

/-- Theorem stating the increase in savings percentage --/
theorem savings_increase_percentage (fs : FinancialSituation)
    (h1 : fs.savingsRate = 0.2)
    (h2 : fs.incomeIncreaseRate = 0.2)
    (h3 : fs.income > 0)
    (h4 : fs.income * (2 - fs.savingsRate) = 
          fs.income * (1 + fs.incomeIncreaseRate) * (1 - fs.savingsRate) + 
          fs.income * (1 - fs.savingsRate)) :
    (fs.income * (1 + fs.incomeIncreaseRate) * fs.savingsRate - 
     fs.income * fs.savingsRate) / 
    (fs.income * fs.savingsRate) = 1 := by
  sorry

#check savings_increase_percentage

end NUMINAMATH_CALUDE_savings_increase_percentage_l1673_167360


namespace NUMINAMATH_CALUDE_cos_2alpha_on_unit_circle_l1673_167358

theorem cos_2alpha_on_unit_circle (α : Real) :
  (Real.cos α = -Real.sqrt 5 / 5 ∧ Real.sin α = 2 * Real.sqrt 5 / 5) →
  Real.cos (2 * α) = -3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_on_unit_circle_l1673_167358


namespace NUMINAMATH_CALUDE_shopkeeper_loss_percent_l1673_167341

/-- Calculates the loss percent for a shopkeeper given profit margin and theft percentage -/
theorem shopkeeper_loss_percent 
  (profit_margin : ℝ) 
  (theft_percent : ℝ) 
  (hprofit : profit_margin = 0.1) 
  (htheft : theft_percent = 0.4) : 
  (1 - (1 - theft_percent) * (1 + profit_margin)) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_loss_percent_l1673_167341


namespace NUMINAMATH_CALUDE_matching_instrument_probability_l1673_167361

/-- The probability of selecting a matching cello-viola pair -/
theorem matching_instrument_probability
  (total_cellos : ℕ)
  (total_violas : ℕ)
  (matching_pairs : ℕ)
  (h1 : total_cellos = 800)
  (h2 : total_violas = 600)
  (h3 : matching_pairs = 100) :
  (matching_pairs : ℚ) / (total_cellos * total_violas) = 1 / 4800 :=
by sorry

end NUMINAMATH_CALUDE_matching_instrument_probability_l1673_167361


namespace NUMINAMATH_CALUDE_extreme_value_implies_zero_derivative_converse_not_always_true_l1673_167379

-- Define a function that has an extreme value at a point
def has_extreme_value (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x ∈ Set.Ioo (x₀ - ε) (x₀ + ε), f x ≤ f x₀ ∨ f x ≥ f x₀

-- Theorem statement
theorem extreme_value_implies_zero_derivative
  (f : ℝ → ℝ) (x₀ : ℝ) (hf : Differentiable ℝ f) :
  has_extreme_value f x₀ → deriv f x₀ = 0 :=
sorry

-- Counter-example to show the converse is not always true
theorem converse_not_always_true :
  ∃ f : ℝ → ℝ, Differentiable ℝ f ∧ deriv f 0 = 0 ∧ ¬(has_extreme_value f 0) :=
sorry

end NUMINAMATH_CALUDE_extreme_value_implies_zero_derivative_converse_not_always_true_l1673_167379


namespace NUMINAMATH_CALUDE_real_part_of_squared_reciprocal_l1673_167364

theorem real_part_of_squared_reciprocal (z : ℂ) (x : ℝ) (h1 : z.im ≠ 0) (h2 : Complex.abs z = 2) (h3 : z.re = x) :
  Complex.re ((1 / (2 - z)) ^ 2) = x / (4 * (4 - 4*x + x^2)) := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_squared_reciprocal_l1673_167364


namespace NUMINAMATH_CALUDE_bike_ride_time_l1673_167352

theorem bike_ride_time (distance1 distance2 time1 : ℝ) (distance1_pos : 0 < distance1) (time1_pos : 0 < time1) :
  distance1 = 2 ∧ time1 = 6 ∧ distance2 = 5 →
  distance2 / (distance1 / time1) = 15 := by sorry

end NUMINAMATH_CALUDE_bike_ride_time_l1673_167352


namespace NUMINAMATH_CALUDE_fixed_point_coordinates_l1673_167327

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line equation y - 2 = k(x + 1) -/
def lineEquation (k : ℝ) (p : Point) : Prop :=
  p.y - 2 = k * (p.x + 1)

/-- The fixed point M satisfies the line equation for all k -/
def isFixedPoint (M : Point) : Prop :=
  ∀ k : ℝ, lineEquation k M

theorem fixed_point_coordinates :
  ∀ M : Point, isFixedPoint M → M = Point.mk (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_coordinates_l1673_167327


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1673_167380

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 3 + a^(x - 1)
  f 1 = 4 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1673_167380


namespace NUMINAMATH_CALUDE_cubic_root_identity_l1673_167376

theorem cubic_root_identity (a b c t : ℝ) : 
  (∀ x, x^3 - 7*x^2 + 8*x - 1 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  t = Real.sqrt a + Real.sqrt b + Real.sqrt c →
  t^6 - 21*t^3 - 9*t = 24*t - 41 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_identity_l1673_167376


namespace NUMINAMATH_CALUDE_work_time_difference_l1673_167369

def monday_minutes : ℕ := 450
def tuesday_minutes : ℕ := monday_minutes / 2
def wednesday_minutes : ℕ := 300

theorem work_time_difference :
  wednesday_minutes - tuesday_minutes = 75 := by
  sorry

end NUMINAMATH_CALUDE_work_time_difference_l1673_167369


namespace NUMINAMATH_CALUDE_inequality_proof_l1673_167396

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1673_167396


namespace NUMINAMATH_CALUDE_inverse_proportion_through_point_l1673_167375

/-- An inverse proportion function passing through (2, -3) has m = -6 -/
theorem inverse_proportion_through_point (m : ℝ) : 
  (∀ x, x ≠ 0 → (m / x = -3 ↔ x = 2)) → m = -6 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_through_point_l1673_167375


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l1673_167394

theorem jelly_bean_probability (p_green p_blue p_red p_yellow : ℝ) :
  p_green = 0.25 →
  p_blue = 0.35 →
  p_green + p_blue + p_red + p_yellow = 1 →
  p_red + p_yellow = 0.40 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l1673_167394


namespace NUMINAMATH_CALUDE_shirt_price_calculation_l1673_167324

/-- Calculates the final price of a shirt given its original cost, profit margin, and discount percentage. -/
def final_price (original_cost : ℝ) (profit_margin : ℝ) (discount : ℝ) : ℝ :=
  let selling_price := original_cost * (1 + profit_margin)
  selling_price * (1 - discount)

/-- Theorem stating that a shirt with an original cost of $20, a 30% profit margin, and a 50% discount has a final price of $13. -/
theorem shirt_price_calculation :
  final_price 20 0.3 0.5 = 13 := by
  sorry

#eval final_price 20 0.3 0.5

end NUMINAMATH_CALUDE_shirt_price_calculation_l1673_167324


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1673_167397

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 2 + a 3 = 1) →
  (a 10 + a 11 = 9) →
  (a 5 + a 6 = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1673_167397


namespace NUMINAMATH_CALUDE_tangent_line_circle_range_l1673_167307

theorem tangent_line_circle_range (m n : ℝ) : 
  (∃ (x y : ℝ), (m + 1) * x + (n + 1) * y - 2 = 0 ∧ (x - 1)^2 + (y - 1)^2 = 1) →
  ((m + n ≤ 2 - 2 * Real.sqrt 2) ∨ (m + n ≥ 2 + 2 * Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_circle_range_l1673_167307


namespace NUMINAMATH_CALUDE_last_operation_end_time_l1673_167387

def minutes_since_8am (hour : Nat) (minute : Nat) : Nat :=
  (hour - 8) * 60 + minute

def operation_end_time (start_time : Nat) (duration : Nat) : Nat :=
  start_time + duration

theorem last_operation_end_time :
  let num_operations : Nat := 10
  let operation_duration : Nat := 45
  let interval_between_starts : Nat := 15
  let first_operation_start : Nat := minutes_since_8am 8 0
  let last_operation_start : Nat := first_operation_start + (num_operations - 1) * interval_between_starts
  let last_operation_end : Nat := operation_end_time last_operation_start operation_duration
  last_operation_end = minutes_since_8am 11 0 :=
by sorry

end NUMINAMATH_CALUDE_last_operation_end_time_l1673_167387


namespace NUMINAMATH_CALUDE_andrea_lauren_bike_problem_l1673_167310

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

end NUMINAMATH_CALUDE_andrea_lauren_bike_problem_l1673_167310


namespace NUMINAMATH_CALUDE_married_men_fraction_l1673_167312

theorem married_men_fraction (total_women : ℕ) (single_women : ℕ) :
  single_women = (3 : ℕ) * total_women / 7 →
  (total_women - single_women) / (total_women + (total_women - single_women)) = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_married_men_fraction_l1673_167312


namespace NUMINAMATH_CALUDE_equilateral_triangle_polyhedron_vertices_l1673_167395

/-- A polyhedron with equilateral triangular faces -/
structure EquilateralTrianglePolyhedron where
  /-- Number of faces -/
  f : ℕ
  /-- Number of edges -/
  e : ℕ
  /-- Number of vertices -/
  v : ℕ
  /-- Each face is an equilateral triangle -/
  faces_are_equilateral_triangles : f = 8
  /-- Euler's formula for polyhedra -/
  euler_formula : v - e + f = 2
  /-- Each edge is shared by exactly two faces -/
  edges_shared : e = (3 * f) / 2

/-- Theorem: A polyhedron with 8 equilateral triangular faces has 6 vertices -/
theorem equilateral_triangle_polyhedron_vertices 
  (p : EquilateralTrianglePolyhedron) : p.v = 6 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_polyhedron_vertices_l1673_167395


namespace NUMINAMATH_CALUDE_function_value_at_2010_l1673_167385

def positive_reals : Set ℝ := {x : ℝ | x > 0}

def function_property (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > y ∧ y > 0 → f (x - y) = Real.sqrt (f (x * y) + 3)

theorem function_value_at_2010 (f : ℝ → ℝ) 
  (h1 : ∀ x ∈ positive_reals, f x > 0)
  (h2 : function_property f) :
  f 2010 = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_2010_l1673_167385


namespace NUMINAMATH_CALUDE_chord_length_unit_circle_specific_chord_length_l1673_167336

/-- The length of the chord cut by a line on a unit circle -/
theorem chord_length_unit_circle (a b c : ℝ) (h : a^2 + b^2 ≠ 0) :
  let line := {(x, y) : ℝ × ℝ | a * x + b * y + c = 0}
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let d := |c| / Real.sqrt (a^2 + b^2)
  2 * Real.sqrt (1 - d^2) = 8/5 :=
by sorry

/-- The specific case for the given problem -/
theorem specific_chord_length :
  let line := {(x, y) : ℝ × ℝ | 3 * x - 4 * y + 3 = 0}
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let d := 3 / 5
  2 * Real.sqrt (1 - d^2) = 8/5 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_unit_circle_specific_chord_length_l1673_167336


namespace NUMINAMATH_CALUDE_area_difference_l1673_167374

/-- The difference in area between a square and a rectangle -/
theorem area_difference (square_side : ℝ) (rect_length rect_width : ℝ) : 
  square_side = 5 → rect_length = 3 → rect_width = 6 → 
  square_side * square_side - rect_length * rect_width = 7 := by
  sorry

#check area_difference

end NUMINAMATH_CALUDE_area_difference_l1673_167374


namespace NUMINAMATH_CALUDE_largest_prime_mersenne_under_500_l1673_167318

def mersenne_number (n : ℕ) : ℕ := 2^n - 1

def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

theorem largest_prime_mersenne_under_500 :
  ∀ n : ℕ, is_power_of_two n → 
    mersenne_number n < 500 → 
    Nat.Prime (mersenne_number n) → 
    mersenne_number n ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_mersenne_under_500_l1673_167318


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l1673_167357

theorem rectangle_area_diagonal (l w d : ℝ) (h1 : l / w = 5 / 2) (h2 : l^2 + w^2 = d^2) :
  l * w = (10 / 29) * d^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l1673_167357


namespace NUMINAMATH_CALUDE_inequality_proof_l1673_167300

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1 / b - c) * (b + 1 / c - a) + 
  (b + 1 / c - a) * (c + 1 / a - b) + 
  (c + 1 / a - b) * (a + 1 / b - c) ≥ 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1673_167300


namespace NUMINAMATH_CALUDE_eight_couples_handshakes_l1673_167302

/-- The number of handshakes in a gathering of couples -/
def count_handshakes (n : ℕ) : ℕ :=
  let total_people := 2 * n
  let handshakes_per_person := total_people - 3
  (total_people * handshakes_per_person) / 2

/-- Theorem: In a gathering of 8 couples, where each person shakes hands with
    everyone except their spouse and one other person, the total number of
    handshakes is 104. -/
theorem eight_couples_handshakes :
  count_handshakes 8 = 104 := by
  sorry

#eval count_handshakes 8  -- Should output 104

end NUMINAMATH_CALUDE_eight_couples_handshakes_l1673_167302


namespace NUMINAMATH_CALUDE_tourists_distribution_eight_l1673_167386

/-- The number of ways to distribute n tourists between 2 guides,
    where each guide must have at least one tourist -/
def distribute_tourists (n : ℕ) : ℕ :=
  2^n - 2

theorem tourists_distribution_eight :
  distribute_tourists 8 = 254 :=
sorry

end NUMINAMATH_CALUDE_tourists_distribution_eight_l1673_167386


namespace NUMINAMATH_CALUDE_salary_approximation_l1673_167334

/-- The salary of a man who spends specific fractions on expenses and has a remainder --/
def salary (food_fraction : ℚ) (rent_fraction : ℚ) (clothes_fraction : ℚ) (remainder : ℚ) : ℚ :=
  remainder / (1 - food_fraction - rent_fraction - clothes_fraction)

/-- Theorem stating the approximate salary of a man with given expenses and remainder --/
theorem salary_approximation :
  let s := salary (1/3) (1/4) (1/5) 1760
  ⌊s⌋ = 8123 := by sorry

end NUMINAMATH_CALUDE_salary_approximation_l1673_167334


namespace NUMINAMATH_CALUDE_additional_flowers_grown_l1673_167319

theorem additional_flowers_grown 
  (initial_flowers : ℕ) 
  (dead_flowers : ℕ) 
  (final_flowers : ℕ) : 
  final_flowers > initial_flowers → 
  final_flowers - initial_flowers = 
    final_flowers - initial_flowers + dead_flowers - dead_flowers :=
by
  sorry

#check additional_flowers_grown

end NUMINAMATH_CALUDE_additional_flowers_grown_l1673_167319


namespace NUMINAMATH_CALUDE_projection_of_a_on_b_l1673_167345

/-- Given two vectors a and b in a real inner product space, 
    with |a| = 3, |b| = 2, and |a - b| = √19,
    prove that the projection of a onto b is -3/2 -/
theorem projection_of_a_on_b 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (a b : V)
  (ha : ‖a‖ = 3)
  (hb : ‖b‖ = 2)
  (hab : ‖a - b‖ = Real.sqrt 19) :
  inner a b / ‖b‖ = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_projection_of_a_on_b_l1673_167345


namespace NUMINAMATH_CALUDE_sheets_from_jane_l1673_167340

theorem sheets_from_jane (initial_sheets final_sheets given_sheets : ℕ) 
  (h1 : initial_sheets = 212)
  (h2 : given_sheets = 156)
  (h3 : final_sheets = 363) :
  initial_sheets + (final_sheets + given_sheets - initial_sheets) - given_sheets = final_sheets := by
  sorry

#check sheets_from_jane

end NUMINAMATH_CALUDE_sheets_from_jane_l1673_167340


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_and_intersection_l1673_167339

theorem quadratic_equation_roots_and_intersection :
  ∀ a : ℚ,
  (∃ x : ℚ, x^2 + a*x + a - 2 = 0) →
  (1^2 + a*1 + a - 2 = 0) →
  (a = 1/2) ∧
  (∃ x : ℚ, x ≠ 1 ∧ x^2 + a*x + a - 2 = 0) ∧
  (∃ x y : ℚ, x ≠ y ∧ x^2 + a*x + a - 2 = 0 ∧ y^2 + a*y + a - 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_and_intersection_l1673_167339


namespace NUMINAMATH_CALUDE_power_sum_reciprocal_integer_l1673_167390

/-- For a non-zero real number x where x + 1/x is an integer, x^n + 1/x^n is an integer for all natural numbers n. -/
theorem power_sum_reciprocal_integer (x : ℝ) (h1 : x ≠ 0) (h2 : ∃ k : ℤ, x + 1/x = k) :
  ∀ n : ℕ, ∃ m : ℤ, x^n + 1/(x^n) = m := by
  sorry

end NUMINAMATH_CALUDE_power_sum_reciprocal_integer_l1673_167390


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l1673_167384

/-- Proves that a rectangle with perimeter 150 cm and length 15 cm greater than width has width 30 cm and length 45 cm -/
theorem rectangle_dimensions (w l : ℝ) 
  (h_perimeter : 2 * w + 2 * l = 150)
  (h_length_width : l = w + 15) :
  w = 30 ∧ l = 45 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l1673_167384


namespace NUMINAMATH_CALUDE_tommy_crates_count_l1673_167348

/-- Proves that Tommy has 3 crates given the problem conditions -/
theorem tommy_crates_count :
  ∀ (c : ℕ),
  (∀ (crate : ℕ), crate = 20) →  -- Each crate holds 20 kg
  (330 : ℝ) = c * (330 : ℝ) / c →  -- Cost of crates is $330
  (∀ (price : ℝ), price = 6) →  -- Selling price is $6 per kg
  (∀ (rotten : ℕ), rotten = 3) →  -- 3 kg of tomatoes are rotten
  (12 : ℝ) = (c * 20 - 3) * 6 - 330 →  -- Profit is $12
  c = 3 := by
sorry

end NUMINAMATH_CALUDE_tommy_crates_count_l1673_167348
