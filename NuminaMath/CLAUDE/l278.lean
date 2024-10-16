import Mathlib

namespace NUMINAMATH_CALUDE_chapters_undetermined_l278_27871

/-- Represents a book with a number of pages and chapters -/
structure Book where
  pages : ℕ
  chapters : ℕ

/-- Represents Jake's reading progress -/
structure ReadingProgress where
  initialRead : ℕ
  laterRead : ℕ
  totalRead : ℕ

/-- Given the conditions of Jake's reading and the book, 
    prove that the number of chapters cannot be determined -/
theorem chapters_undetermined (book : Book) (progress : ReadingProgress) : 
  book.pages = 95 ∧ 
  progress.initialRead = 37 ∧ 
  progress.laterRead = 25 ∧ 
  progress.totalRead = 62 →
  ¬ ∃ (n : ℕ), ∀ (b : Book), 
    b.pages = book.pages ∧ 
    b.chapters = n :=
by sorry

end NUMINAMATH_CALUDE_chapters_undetermined_l278_27871


namespace NUMINAMATH_CALUDE_ways_to_buy_three_items_l278_27814

/-- Represents the inventory of a store selling computer peripherals -/
structure StoreInventory where
  headphones : ℕ
  mice : ℕ
  keyboards : ℕ
  keyboard_mouse_sets : ℕ
  headphone_mouse_sets : ℕ

/-- Calculates the number of ways to buy a headphone, a keyboard, and a mouse -/
def waysToButThreeItems (inventory : StoreInventory) : ℕ :=
  inventory.headphones * inventory.mice * inventory.keyboards +
  inventory.keyboard_mouse_sets * inventory.headphones +
  inventory.headphone_mouse_sets * inventory.keyboards

/-- The store's actual inventory -/
def actualInventory : StoreInventory := {
  headphones := 9
  mice := 13
  keyboards := 5
  keyboard_mouse_sets := 4
  headphone_mouse_sets := 5
}

theorem ways_to_buy_three_items :
  waysToButThreeItems actualInventory = 646 := by
  sorry

end NUMINAMATH_CALUDE_ways_to_buy_three_items_l278_27814


namespace NUMINAMATH_CALUDE_balls_in_boxes_l278_27836

def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  if k = 2 then
    (n - 1).choose (k - 1) + (n - 2).choose (k - 1)
  else
    0

theorem balls_in_boxes :
  distribute_balls 6 2 = 21 :=
sorry

end NUMINAMATH_CALUDE_balls_in_boxes_l278_27836


namespace NUMINAMATH_CALUDE_angle_relationship_l278_27888

theorem angle_relationship (α β : Real) 
  (h1 : 0 < α) 
  (h2 : α < 2 * β) 
  (h3 : 2 * β ≤ π / 2) 
  (h4 : 2 * Real.cos (α + β) * Real.cos β = -1 + 2 * Real.sin (α + β) * Real.sin β) : 
  α + 2 * β = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_relationship_l278_27888


namespace NUMINAMATH_CALUDE_negation_equivalence_l278_27835

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l278_27835


namespace NUMINAMATH_CALUDE_bernoulli_inequality_l278_27802

theorem bernoulli_inequality (h : ℝ) (hgt : h > -1) :
  (∀ x > 1, (1 + h)^x > 1 + h*x) ∧
  (∀ x < 0, (1 + h)^x > 1 + h*x) ∧
  (∀ x, 0 < x → x < 1 → (1 + h)^x < 1 + h*x) := by
  sorry

end NUMINAMATH_CALUDE_bernoulli_inequality_l278_27802


namespace NUMINAMATH_CALUDE_correct_calculation_l278_27850

theorem correct_calculation (x : ℚ) (h : x / 6 = 12) : x * 7 = 504 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l278_27850


namespace NUMINAMATH_CALUDE_rectangle_to_square_l278_27897

/-- Given a rectangle with perimeter 50 cm, prove that decreasing its length by 4 cm
    and increasing its width by 3 cm results in a square with side 12 cm and equal area. -/
theorem rectangle_to_square (L W : ℝ) : 
  L > 0 ∧ W > 0 ∧                    -- Length and width are positive
  2 * L + 2 * W = 50 ∧               -- Perimeter of original rectangle is 50 cm
  L * W = (L - 4) * (W + 3) →        -- Area remains constant after transformation
  L = 16 ∧ W = 9 ∧                   -- Original rectangle dimensions
  L - 4 = 12 ∧ W + 3 = 12            -- New shape is a square with side 12 cm
  := by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_l278_27897


namespace NUMINAMATH_CALUDE_total_days_is_30_l278_27846

/-- The number of days being considered -/
def total_days : ℕ := sorry

/-- The mean daily profit for all days in rupees -/
def mean_profit : ℕ := 350

/-- The mean profit for the first 15 days in rupees -/
def mean_profit_first_15 : ℕ := 275

/-- The mean profit for the last 15 days in rupees -/
def mean_profit_last_15 : ℕ := 425

/-- Theorem stating that the total number of days is 30 -/
theorem total_days_is_30 :
  total_days = 30 ∧
  total_days * mean_profit = 15 * mean_profit_first_15 + 15 * mean_profit_last_15 :=
by sorry

end NUMINAMATH_CALUDE_total_days_is_30_l278_27846


namespace NUMINAMATH_CALUDE_system_equation_solution_l278_27849

theorem system_equation_solution (x y some_number : ℝ) : 
  (2 * x + y = 7) → 
  (x + 2 * y = 5) → 
  (2 * x * y / some_number = 2) →
  some_number = 3 := by
  sorry

end NUMINAMATH_CALUDE_system_equation_solution_l278_27849


namespace NUMINAMATH_CALUDE_division_problem_l278_27898

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 1565 → 
  quotient = 65 → 
  remainder = 5 → 
  dividend = divisor * quotient + remainder → 
  divisor = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_division_problem_l278_27898


namespace NUMINAMATH_CALUDE_sum_of_max_min_is_10_l278_27872

-- Define the function
def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x + 5

-- Define the interval
def I : Set ℝ := Set.Icc (-2) 2

-- State the theorem
theorem sum_of_max_min_is_10 :
  ∃ (a b : ℝ), a ∈ I ∧ b ∈ I ∧
  (∀ x ∈ I, f x ≤ f a) ∧
  (∀ x ∈ I, f x ≥ f b) ∧
  f a + f b = 10 :=
sorry

end NUMINAMATH_CALUDE_sum_of_max_min_is_10_l278_27872


namespace NUMINAMATH_CALUDE_emails_left_l278_27852

def process_emails (initial : ℕ) : ℕ :=
  let after_trash := initial / 2
  let after_work := after_trash - (after_trash * 2 / 5)
  let after_personal := after_work - (after_work / 4)
  let after_misc := after_personal - (after_personal / 10)
  let after_subfolder := after_misc - (after_misc * 3 / 10)
  after_subfolder - (after_subfolder / 5)

theorem emails_left (initial : ℕ) (h : initial = 600) : process_emails initial = 69 := by
  sorry

end NUMINAMATH_CALUDE_emails_left_l278_27852


namespace NUMINAMATH_CALUDE_total_pay_is_880_l278_27864

/-- The total amount paid to two employees, where one is paid 120% of the other's wage -/
def total_pay (y_pay : ℝ) : ℝ :=
  y_pay + 1.2 * y_pay

/-- Theorem stating that the total pay for two employees is 880 when one is paid 400 and the other 120% of that -/
theorem total_pay_is_880 :
  total_pay 400 = 880 := by
  sorry

end NUMINAMATH_CALUDE_total_pay_is_880_l278_27864


namespace NUMINAMATH_CALUDE_theresa_kayla_ratio_l278_27896

/-- The number of chocolate bars Theresa bought -/
def theresa_chocolate : ℕ := 12

/-- The number of soda cans Theresa bought -/
def theresa_soda : ℕ := 18

/-- The total number of items Kayla bought -/
def kayla_total : ℕ := 15

/-- The ratio of items Theresa bought to items Kayla bought -/
def item_ratio : ℚ := (theresa_chocolate + theresa_soda : ℚ) / kayla_total

theorem theresa_kayla_ratio : item_ratio = 2 := by sorry

end NUMINAMATH_CALUDE_theresa_kayla_ratio_l278_27896


namespace NUMINAMATH_CALUDE_class_size_from_mark_change_l278_27891

/-- Given a class where one pupil's mark was increased by 20 points, 
    causing the class average to rise by 1/2, prove that there are 40 pupils in the class. -/
theorem class_size_from_mark_change (mark_increase : ℕ) (average_increase : ℚ) : 
  mark_increase = 20 → average_increase = 1/2 → (mark_increase : ℚ) / average_increase = 40 := by
  sorry

end NUMINAMATH_CALUDE_class_size_from_mark_change_l278_27891


namespace NUMINAMATH_CALUDE_inverse_inequality_l278_27882

theorem inverse_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 1 / a < 1 / b := by
  sorry

end NUMINAMATH_CALUDE_inverse_inequality_l278_27882


namespace NUMINAMATH_CALUDE_average_age_increase_proof_l278_27879

/-- The initial number of men in a group where replacing two men with two women increases the average age by 2 years -/
def initial_men_count : ℕ := 8

theorem average_age_increase_proof :
  let men_removed_age_sum := 20 + 28
  let women_added_age_sum := 32 + 32
  let age_difference := women_added_age_sum - men_removed_age_sum
  let average_age_increase := 2
  initial_men_count * average_age_increase = age_difference := by
  sorry

#check average_age_increase_proof

end NUMINAMATH_CALUDE_average_age_increase_proof_l278_27879


namespace NUMINAMATH_CALUDE_range_of_a_l278_27855

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - a| + |x + 1| > 2) → a ∈ Set.Iio (-3) ∪ Set.Ioi 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l278_27855


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_l278_27832

def f (x : ℝ) := x^3 - 3*x^2

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, (∀ y : ℝ, x < y → f x > f y) ↔ x ∈ Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_l278_27832


namespace NUMINAMATH_CALUDE_circle_area_ratio_l278_27889

theorem circle_area_ratio (r R : ℝ) (h : r = R / 3) :
  (π * r^2) / (π * R^2) = 1 / 9 := by
sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l278_27889


namespace NUMINAMATH_CALUDE_triangle_problem_l278_27899

-- Define the triangle ABC
variable (A B C : Real) -- Angles
variable (a b c : Real) -- Sides
variable (S : Real) -- Area

-- State the theorem
theorem triangle_problem 
  (h1 : 2 * Real.cos (2 * B) = 4 * Real.cos B - 3)
  (h2 : S = Real.sqrt 3)
  (h3 : a * Real.sin A + c * Real.sin C = 5 * Real.sin B) :
  B = π / 3 ∧ b = (5 + Real.sqrt 21) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l278_27899


namespace NUMINAMATH_CALUDE_unique_intersection_l278_27801

/-- The value of m for which the line x = m intersects the parabola x = -3y² - 4y + 7 at exactly one point -/
def intersection_point : ℚ := 25 / 3

/-- The parabola equation -/
def parabola (y : ℝ) : ℝ := -3 * y^2 - 4 * y + 7

theorem unique_intersection :
  ∀ m : ℝ, (∃! y : ℝ, parabola y = m) ↔ m = intersection_point := by sorry

end NUMINAMATH_CALUDE_unique_intersection_l278_27801


namespace NUMINAMATH_CALUDE_total_pencils_l278_27834

/-- Given an initial number of pencils and a number of pencils added,
    the total number of pencils is equal to the sum of the initial number and the added number. -/
theorem total_pencils (initial : ℕ) (added : ℕ) : 
  initial + added = initial + added :=
by sorry

end NUMINAMATH_CALUDE_total_pencils_l278_27834


namespace NUMINAMATH_CALUDE_max_real_part_of_roots_l278_27803

open Complex

-- Define the polynomial
def p (z : ℂ) : ℂ := z^6 - z^4 + z^2 - 1

-- Theorem statement
theorem max_real_part_of_roots :
  ∃ (z : ℂ), p z = 0 ∧ 
  ∀ (w : ℂ), p w = 0 → z.re ≥ w.re ∧
  z.re = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_real_part_of_roots_l278_27803


namespace NUMINAMATH_CALUDE_time_for_600_parts_l278_27857

/-- Linear regression equation relating parts processed to time spent -/
def linear_regression (x : ℝ) : ℝ := 0.01 * x + 0.5

/-- Theorem stating that processing 600 parts takes 6.5 hours -/
theorem time_for_600_parts : linear_regression 600 = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_time_for_600_parts_l278_27857


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_integers_l278_27848

theorem sum_of_three_consecutive_integers (a b c : ℤ) : 
  (a + 1 = b ∧ b + 1 = c) → c = 14 → a + b + c = 39 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_integers_l278_27848


namespace NUMINAMATH_CALUDE_line_through_point_with_slope_l278_27812

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if a point (x, y) lies on the line -/
def Line.containsPoint (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.yIntercept

/-- The equation of the line in the form ax + by + c = 0 -/
def Line.equation (l : Line) (x y : ℝ) : Prop :=
  l.slope * x - y + l.yIntercept = 0

theorem line_through_point_with_slope (x₀ y₀ m : ℝ) :
  ∃ (l : Line), l.slope = m ∧ l.containsPoint x₀ y₀ ∧
  ∀ (x y : ℝ), l.equation x y ↔ (2 : ℝ) * x - y + 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_line_through_point_with_slope_l278_27812


namespace NUMINAMATH_CALUDE_lottery_probability_l278_27862

def lottery_size : ℕ := 90
def draw_size : ℕ := 5

def valid_outcomes : ℕ := 3 * (Nat.choose 86 3)

def total_outcomes : ℕ := Nat.choose lottery_size draw_size

theorem lottery_probability : 
  (valid_outcomes : ℚ) / total_outcomes = 258192 / 43949268 := by sorry

end NUMINAMATH_CALUDE_lottery_probability_l278_27862


namespace NUMINAMATH_CALUDE_absolute_difference_x_y_l278_27810

theorem absolute_difference_x_y (x y : ℝ) 
  (h1 : ⌊x⌋ + (y - ⌊y⌋) = 2.4)
  (h2 : (x - ⌊x⌋) + ⌊y⌋ = 5.1) : 
  |x - y| = 3.3 := by
sorry

end NUMINAMATH_CALUDE_absolute_difference_x_y_l278_27810


namespace NUMINAMATH_CALUDE_parallelogram_area_18_10_l278_27875

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 18 cm and height 10 cm is 180 cm² -/
theorem parallelogram_area_18_10 : 
  parallelogram_area 18 10 = 180 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_18_10_l278_27875


namespace NUMINAMATH_CALUDE_amoeba_growth_30_minutes_l278_27807

/-- The number of amoebas after a given time interval, given an initial population and growth rate. -/
def amoeba_population (initial : ℕ) (growth_factor : ℕ) (intervals : ℕ) : ℕ :=
  initial * growth_factor ^ intervals

/-- Theorem stating that given the initial conditions, the final amoeba population after 30 minutes is 36450. -/
theorem amoeba_growth_30_minutes :
  let initial_population : ℕ := 50
  let growth_factor : ℕ := 3
  let interval_duration : ℕ := 5
  let total_duration : ℕ := 30
  let num_intervals : ℕ := total_duration / interval_duration
  amoeba_population initial_population growth_factor num_intervals = 36450 := by
  sorry

#eval amoeba_population 50 3 6

end NUMINAMATH_CALUDE_amoeba_growth_30_minutes_l278_27807


namespace NUMINAMATH_CALUDE_arithmetic_sequence_decreasing_l278_27841

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_decreasing
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a2 : (a 2 - 1)^3 + 2012 * (a 2 - 1) = 1)
  (h_a2011 : (a 2011 - 1)^3 + 2012 * (a 2011 - 1) = -1) :
  a 2011 < a 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_decreasing_l278_27841


namespace NUMINAMATH_CALUDE_darks_drying_time_l278_27833

/-- Represents the time in minutes for washing and drying a load of laundry -/
structure LaundryTime where
  wash : Nat
  dry : Nat

/-- Calculates the total time for a load of laundry -/
def totalTime (lt : LaundryTime) : Nat :=
  lt.wash + lt.dry

theorem darks_drying_time (whites : LaundryTime) (darks_wash : Nat) (colors : LaundryTime) 
    (total_time : Nat) (h1 : whites.wash = 72) (h2 : whites.dry = 50)
    (h3 : darks_wash = 58) (h4 : colors.wash = 45) (h5 : colors.dry = 54)
    (h6 : total_time = 344) :
    ∃ (darks_dry : Nat), darks_dry = 65 ∧ 
    total_time = totalTime whites + totalTime colors + darks_wash + darks_dry := by
  sorry

#check darks_drying_time

end NUMINAMATH_CALUDE_darks_drying_time_l278_27833


namespace NUMINAMATH_CALUDE_coin_problem_l278_27853

/-- Represents the types of coins --/
inductive CoinType
  | Penny
  | Nickel
  | Dime
  | Quarter
  | HalfDollar

/-- Returns the value of a coin in cents --/
def coinValue (c : CoinType) : ℕ :=
  match c with
  | .Penny => 1
  | .Nickel => 5
  | .Dime => 10
  | .Quarter => 25
  | .HalfDollar => 50

/-- Represents a collection of coins --/
structure CoinCollection where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ
  halfDollars : ℕ

/-- Calculates the total number of coins in a collection --/
def totalCoins (c : CoinCollection) : ℕ :=
  c.pennies + c.nickels + c.dimes + c.quarters + c.halfDollars

/-- Calculates the total value of coins in a collection in cents --/
def totalValue (c : CoinCollection) : ℕ :=
  c.pennies * coinValue CoinType.Penny +
  c.nickels * coinValue CoinType.Nickel +
  c.dimes * coinValue CoinType.Dime +
  c.quarters * coinValue CoinType.Quarter +
  c.halfDollars * coinValue CoinType.HalfDollar

/-- The main theorem --/
theorem coin_problem (c : CoinCollection) 
  (h1 : totalCoins c = 11)
  (h2 : totalValue c = 143)
  (h3 : c.pennies ≥ 1)
  (h4 : c.nickels ≥ 1)
  (h5 : c.dimes ≥ 1)
  (h6 : c.quarters ≥ 1)
  (h7 : c.halfDollars ≥ 1) :
  c.dimes = 4 := by
  sorry

end NUMINAMATH_CALUDE_coin_problem_l278_27853


namespace NUMINAMATH_CALUDE_total_weight_sold_l278_27876

/-- Calculates the total weight of bags sold in a day given the sales data and weight per bag -/
theorem total_weight_sold (morning_potatoes afternoon_potatoes morning_onions afternoon_onions
  morning_carrots afternoon_carrots potato_weight onion_weight carrot_weight : ℕ) :
  morning_potatoes = 29 →
  afternoon_potatoes = 17 →
  morning_onions = 15 →
  afternoon_onions = 22 →
  morning_carrots = 12 →
  afternoon_carrots = 9 →
  potato_weight = 7 →
  onion_weight = 5 →
  carrot_weight = 4 →
  (morning_potatoes + afternoon_potatoes) * potato_weight +
  (morning_onions + afternoon_onions) * onion_weight +
  (morning_carrots + afternoon_carrots) * carrot_weight = 591 :=
by
  sorry

end NUMINAMATH_CALUDE_total_weight_sold_l278_27876


namespace NUMINAMATH_CALUDE_abs_fraction_inequality_l278_27800

theorem abs_fraction_inequality (x : ℝ) : 
  |((3 * x - 2) / (x - 2))| > 3 ↔ x ∈ Set.Ioo (4/3) 2 ∪ Set.Ioi 2 :=
sorry

end NUMINAMATH_CALUDE_abs_fraction_inequality_l278_27800


namespace NUMINAMATH_CALUDE_magic_box_pennies_l278_27809

def double_daily (initial : ℕ) (days : ℕ) : ℕ :=
  initial * (2 ^ days)

theorem magic_box_pennies :
  ∃ (initial : ℕ), double_daily initial 4 = 48 ∧ initial = 3 :=
sorry

end NUMINAMATH_CALUDE_magic_box_pennies_l278_27809


namespace NUMINAMATH_CALUDE_trumpet_cost_l278_27828

/-- The cost of Mike's trumpet, given his total spending and the cost of a song book. -/
theorem trumpet_cost (total_spent song_book_cost : ℚ) 
  (h1 : total_spent = 151)
  (h2 : song_book_cost = 584 / 100) : 
  total_spent - song_book_cost = 14516 / 100 := by
  sorry

end NUMINAMATH_CALUDE_trumpet_cost_l278_27828


namespace NUMINAMATH_CALUDE_tom_total_weight_l278_27823

/-- Calculates the total weight Tom is moving with given his body weight, the weight he holds in each hand, and the weight of his vest. -/
def total_weight (tom_weight : ℝ) (hand_multiplier : ℝ) (vest_multiplier : ℝ) : ℝ :=
  tom_weight * hand_multiplier * 2 + tom_weight * vest_multiplier

/-- Theorem stating that Tom's total weight moved is 525 kg given the problem conditions. -/
theorem tom_total_weight :
  let tom_weight : ℝ := 150
  let hand_multiplier : ℝ := 1.5
  let vest_multiplier : ℝ := 0.5
  total_weight tom_weight hand_multiplier vest_multiplier = 525 := by
  sorry

end NUMINAMATH_CALUDE_tom_total_weight_l278_27823


namespace NUMINAMATH_CALUDE_parallelogram_area_l278_27868

/-- The area of a parallelogram with given dimensions -/
theorem parallelogram_area (base slant_height horiz_diff : ℝ) 
  (h_base : base = 20)
  (h_slant : slant_height = 6)
  (h_diff : horiz_diff = 5) :
  base * Real.sqrt (slant_height^2 - horiz_diff^2) = 20 * Real.sqrt 11 := by
  sorry

#check parallelogram_area

end NUMINAMATH_CALUDE_parallelogram_area_l278_27868


namespace NUMINAMATH_CALUDE_calculation_proof_l278_27885

theorem calculation_proof : (300000 * 200000) / 100000 = 600000 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l278_27885


namespace NUMINAMATH_CALUDE_randy_initial_money_l278_27867

theorem randy_initial_money :
  ∀ M : ℝ,
  (M - 10 - (M - 10) / 4 = 15) →
  M = 30 := by
sorry

end NUMINAMATH_CALUDE_randy_initial_money_l278_27867


namespace NUMINAMATH_CALUDE_triangle_inequalities_l278_27818

theorem triangle_inequalities (a b c A B C : Real) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hA : A > 0) (hB : B > 0) (hC : C > 0)
  (h_triangle : A + B + C = π)
  (h_sides : a = BC ∧ b = AC ∧ c = AB) : 
  (1 / a^3 + 1 / b^3 + 1 / c^3 + a*b*c ≥ 2 * Real.sqrt 3) ∧
  (1 / A + 1 / B + 1 / C ≥ 9 / π) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l278_27818


namespace NUMINAMATH_CALUDE_plot_length_is_52_l278_27883

/-- Represents a rectangular plot with specific fencing conditions -/
structure Plot where
  breadth : ℝ
  length : ℝ
  flatCost : ℝ
  risePercent : ℝ
  totalRise : ℝ
  totalCost : ℝ

/-- Calculates the length of the plot given the conditions -/
def calculateLength (p : Plot) : ℝ :=
  p.breadth + 20

/-- Theorem stating the length of the plot under given conditions -/
theorem plot_length_is_52 (p : Plot) 
  (h1 : p.length = p.breadth + 20)
  (h2 : p.flatCost = 26.5)
  (h3 : p.risePercent = 0.1)
  (h4 : p.totalRise = 5)
  (h5 : p.totalCost = 5300)
  (h6 : p.totalCost = 2 * (p.breadth + 20) * p.flatCost + 
        2 * p.breadth * (p.flatCost * (1 + p.risePercent * p.totalRise))) :
  calculateLength p = 52 := by
  sorry

#eval calculateLength { breadth := 32, length := 52, flatCost := 26.5, 
                        risePercent := 0.1, totalRise := 5, totalCost := 5300 }

end NUMINAMATH_CALUDE_plot_length_is_52_l278_27883


namespace NUMINAMATH_CALUDE_area_of_union_rectangle_circle_l278_27874

def rectangle_width : ℝ := 8
def rectangle_height : ℝ := 12
def circle_radius : ℝ := 10

theorem area_of_union_rectangle_circle :
  let rectangle_area := rectangle_width * rectangle_height
  let circle_area := π * circle_radius^2
  let overlap_area := (π * circle_radius^2) / 4
  rectangle_area + circle_area - overlap_area = 96 + 75 * π := by
sorry

end NUMINAMATH_CALUDE_area_of_union_rectangle_circle_l278_27874


namespace NUMINAMATH_CALUDE_range_of_a_l278_27865

theorem range_of_a (a : ℝ) : 
  (¬ ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l278_27865


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l278_27877

theorem ratio_x_to_y (x y : ℝ) (h : 0.8 * x = 0.2 * y) : x / y = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l278_27877


namespace NUMINAMATH_CALUDE_mountain_bike_price_l278_27869

theorem mountain_bike_price (upfront_payment : ℝ) (upfront_percentage : ℝ) (total_price : ℝ) : 
  upfront_payment = 240 → 
  upfront_percentage = 20 → 
  upfront_payment = (upfront_percentage / 100) * total_price →
  total_price = 1200 := by
sorry

end NUMINAMATH_CALUDE_mountain_bike_price_l278_27869


namespace NUMINAMATH_CALUDE_exactly_one_greater_than_one_l278_27838

theorem exactly_one_greater_than_one 
  (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (product_one : a * b * c = 1)
  (sum_greater : a + b + c > 1/a + 1/b + 1/c) :
  (a > 1 ∧ b ≤ 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b > 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b ≤ 1 ∧ c > 1) :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_greater_than_one_l278_27838


namespace NUMINAMATH_CALUDE_library_shelf_count_l278_27827

theorem library_shelf_count (notebooks : ℕ) (pen_difference : ℕ) : 
  notebooks = 30 → pen_difference = 50 → notebooks + (notebooks + pen_difference) = 110 :=
by sorry

end NUMINAMATH_CALUDE_library_shelf_count_l278_27827


namespace NUMINAMATH_CALUDE_cubic_factorization_l278_27851

theorem cubic_factorization (x : ℝ) : x^3 - 4*x^2 + 4*x = x*(x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l278_27851


namespace NUMINAMATH_CALUDE_elizabeth_borrowed_53_cents_l278_27886

/-- The amount Elizabeth borrowed from her neighbor -/
def amount_borrowed : ℕ := by sorry

theorem elizabeth_borrowed_53_cents :
  let pencil_cost : ℕ := 600  -- in cents
  let elizabeth_has : ℕ := 500  -- in cents
  let needs_more : ℕ := 47  -- in cents
  amount_borrowed = pencil_cost - elizabeth_has - needs_more :=
by sorry

end NUMINAMATH_CALUDE_elizabeth_borrowed_53_cents_l278_27886


namespace NUMINAMATH_CALUDE_intersection_distance_proof_l278_27829

/-- The distance between intersection points of y = 5 and y = 3x^2 + 2x - 2 -/
def intersection_distance : ℝ := sorry

/-- The equation y = 3x^2 + 2x - 2 -/
def parabola (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 2

theorem intersection_distance_proof :
  let p : ℕ := 88
  let q : ℕ := 3
  (∃ (x₁ x₂ : ℝ), 
    parabola x₁ = 5 ∧ 
    parabola x₂ = 5 ∧ 
    x₁ ≠ x₂ ∧
    intersection_distance = |x₁ - x₂|) ∧
  intersection_distance = Real.sqrt p / q ∧
  p - q^2 = 79 := by sorry

end NUMINAMATH_CALUDE_intersection_distance_proof_l278_27829


namespace NUMINAMATH_CALUDE_sqrt_sum_simplification_l278_27895

theorem sqrt_sum_simplification : 
  Real.sqrt 75 - 9 * Real.sqrt (1/3) + Real.sqrt 48 = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_simplification_l278_27895


namespace NUMINAMATH_CALUDE_nickel_probability_is_5_24_l278_27859

/-- Represents the types of coins in the jar -/
inductive Coin
  | Dime
  | Nickel
  | Penny

/-- The value of each coin type in cents -/
def coinValue : Coin → ℕ
  | Coin.Dime => 10
  | Coin.Nickel => 5
  | Coin.Penny => 1

/-- The total value of each coin type in the jar in cents -/
def totalValue : Coin → ℕ
  | Coin.Dime => 800
  | Coin.Nickel => 500
  | Coin.Penny => 300

/-- The number of coins of each type in the jar -/
def coinCount (c : Coin) : ℕ := totalValue c / coinValue c

/-- The total number of coins in the jar -/
def totalCoins : ℕ := coinCount Coin.Dime + coinCount Coin.Nickel + coinCount Coin.Penny

/-- The probability of randomly selecting a nickel from the jar -/
def nickelProbability : ℚ := coinCount Coin.Nickel / totalCoins

theorem nickel_probability_is_5_24 : nickelProbability = 5 / 24 := by
  sorry

end NUMINAMATH_CALUDE_nickel_probability_is_5_24_l278_27859


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l278_27845

theorem roots_sum_of_squares (a b : ℝ) : 
  (a^2 - 9*a + 9 = 0) → (b^2 - 9*b + 9 = 0) → a^2 + b^2 = 63 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l278_27845


namespace NUMINAMATH_CALUDE_unique_valid_square_l278_27840

/-- A number is a square with exactly two non-zero digits, one of which is 3 -/
def is_valid_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 ∧ 
  (∃ a b : ℕ, a ≠ 0 ∧ b ≠ 0 ∧ (a = 3 ∨ b = 3) ∧ n = 10 * a + b) ∧
  (∀ c d e : ℕ, n ≠ 100 * c + 10 * d + e ∨ c = 0 ∨ d = 0 ∨ e = 0)

theorem unique_valid_square : 
  ∀ n : ℕ, is_valid_square n ↔ n = 36 :=
by sorry

end NUMINAMATH_CALUDE_unique_valid_square_l278_27840


namespace NUMINAMATH_CALUDE_f_max_value_l278_27893

/-- The quadratic function f(x) = -3x^2 + 15x + 9 -/
def f (x : ℝ) : ℝ := -3 * x^2 + 15 * x + 9

/-- The maximum value of f(x) is 111/4 -/
theorem f_max_value : ∃ (M : ℝ), M = 111/4 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry

end NUMINAMATH_CALUDE_f_max_value_l278_27893


namespace NUMINAMATH_CALUDE_percent_equality_l278_27866

theorem percent_equality (y : ℝ) : (18 / 100) * y = (30 / 100) * ((60 / 100) * y) := by
  sorry

end NUMINAMATH_CALUDE_percent_equality_l278_27866


namespace NUMINAMATH_CALUDE_milk_price_theorem_l278_27873

theorem milk_price_theorem (n₁ n₂ : ℕ) (p₁ p₂ : ℚ) (h₁ : n₁ = 2) (h₂ : n₂ = 3) 
  (h₃ : p₁ = 32) (h₄ : p₂ = 12) : 
  (n₁ * p₁ + n₂ * p₂) / (n₁ + n₂ : ℚ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_milk_price_theorem_l278_27873


namespace NUMINAMATH_CALUDE_max_cables_cut_specific_case_l278_27815

/-- Represents a computer network -/
structure ComputerNetwork where
  total_computers : ℕ
  initial_cables : ℕ
  initial_clusters : ℕ
  final_clusters : ℕ

/-- Calculates the maximum number of cables that can be cut in a computer network -/
def max_cables_cut (network : ComputerNetwork) : ℕ :=
  network.initial_cables - (network.total_computers - network.final_clusters)

/-- Theorem stating the maximum number of cables that can be cut in the given scenario -/
theorem max_cables_cut_specific_case :
  let network := ComputerNetwork.mk 200 345 1 8
  max_cables_cut network = 153 := by
  sorry

#eval max_cables_cut (ComputerNetwork.mk 200 345 1 8)

end NUMINAMATH_CALUDE_max_cables_cut_specific_case_l278_27815


namespace NUMINAMATH_CALUDE_equation_solutions_l278_27817

theorem equation_solutions : 
  ∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = 2/3 ∧ 
  (∀ x : ℝ, 2*x - 6 = 3*x*(x - 3) ↔ (x = x₁ ∨ x = x₂)) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l278_27817


namespace NUMINAMATH_CALUDE_inequality_proof_l278_27837

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 1/a + 1/b = 1) :
  ∀ n : ℕ, (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l278_27837


namespace NUMINAMATH_CALUDE_solution_set_f_less_than_5_range_of_a_for_f_greater_than_abs_1_minus_a_l278_27856

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| + |2*x + 3|

-- Theorem for part I
theorem solution_set_f_less_than_5 :
  {x : ℝ | f x < 5} = Set.Ioo (-7/4) (3/4) :=
sorry

-- Theorem for part II
theorem range_of_a_for_f_greater_than_abs_1_minus_a :
  {a : ℝ | ∀ x, f x > |1 - a|} = Set.Ioo (-3) 5 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_less_than_5_range_of_a_for_f_greater_than_abs_1_minus_a_l278_27856


namespace NUMINAMATH_CALUDE_balance_scale_theorem_l278_27811

/-- Represents a weight on the balance scale -/
structure Weight where
  pan : Bool  -- true for left pan, false for right pan
  value : ℝ
  number : ℕ

/-- Represents the state of the balance scale -/
structure BalanceScale where
  k : ℕ  -- number of weights on each pan
  weights : List Weight

/-- Checks if the left pan is heavier -/
def leftPanHeavier (scale : BalanceScale) : Prop :=
  let leftSum := (scale.weights.filter (fun w => w.pan)).map (fun w => w.value) |>.sum
  let rightSum := (scale.weights.filter (fun w => !w.pan)).map (fun w => w.value) |>.sum
  leftSum > rightSum

/-- Checks if swapping weights with the same number makes the right pan heavier or balances the pans -/
def swapMakesRightHeavierOrBalance (scale : BalanceScale) : Prop :=
  ∀ i, i ≤ scale.k →
    let swappedWeights := scale.weights.map (fun w => if w.number = i then { w with pan := !w.pan } else w)
    let swappedLeftSum := (swappedWeights.filter (fun w => w.pan)).map (fun w => w.value) |>.sum
    let swappedRightSum := (swappedWeights.filter (fun w => !w.pan)).map (fun w => w.value) |>.sum
    swappedRightSum ≥ swappedLeftSum

/-- The main theorem stating that k can only be 1 or 2 -/
theorem balance_scale_theorem (scale : BalanceScale) :
  leftPanHeavier scale →
  swapMakesRightHeavierOrBalance scale →
  scale.k = 1 ∨ scale.k = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_balance_scale_theorem_l278_27811


namespace NUMINAMATH_CALUDE_number_multiplied_by_six_l278_27805

theorem number_multiplied_by_six (n : ℚ) : n / 11 = 2 → n * 6 = 132 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplied_by_six_l278_27805


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l278_27894

theorem smallest_integer_with_remainders : ∃ b : ℕ, 
  b > 0 ∧ 
  b % 9 = 5 ∧ 
  b % 11 = 7 ∧
  ∀ c : ℕ, c > 0 ∧ c % 9 = 5 ∧ c % 11 = 7 → b ≤ c :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l278_27894


namespace NUMINAMATH_CALUDE_nonzero_terms_count_l278_27830

/-- The number of nonzero terms in the expansion of (x+4)(3x^3 + 2x^2 + 3x + 9) - 4(x^4 - 3x^3 + 2x^2 + 7x) -/
theorem nonzero_terms_count (x : ℝ) : 
  let expansion := (x + 4) * (3*x^3 + 2*x^2 + 3*x + 9) - 4*(x^4 - 3*x^3 + 2*x^2 + 7*x)
  ∃ (a b c d e : ℝ), 
    expansion = a*x^4 + b*x^3 + c*x^2 + d*x + e ∧ 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_nonzero_terms_count_l278_27830


namespace NUMINAMATH_CALUDE_fraction_simplification_l278_27881

theorem fraction_simplification (a b x : ℝ) :
  (Real.sqrt (a^2 + x^2) - (x^2 - b*a^2) / Real.sqrt (a^2 + x^2) + b) / (a^2 + x^2 + b^2) =
  (1 + b) / Real.sqrt ((a^2 + x^2) * (a^2 + x^2 + b^2)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l278_27881


namespace NUMINAMATH_CALUDE_zero_in_interval_l278_27844

-- Define the function f
def f (x : ℝ) : ℝ := -|x - 5| + 2*x - 1

-- State the theorem
theorem zero_in_interval : 
  ∃ x ∈ Set.Ioo 2 3, f x = 0 :=
sorry

end NUMINAMATH_CALUDE_zero_in_interval_l278_27844


namespace NUMINAMATH_CALUDE_infinite_sum_not_diff_powers_l278_27822

theorem infinite_sum_not_diff_powers (n : ℕ) (hn : n > 1) :
  ∃ S : Set ℕ, (Set.Infinite S) ∧
    (∀ k ∈ S, ∃ a b : ℕ, k = a^n + b^n) ∧
    (∀ k ∈ S, ∀ c d : ℕ, k ≠ c^n - d^n) :=
sorry

end NUMINAMATH_CALUDE_infinite_sum_not_diff_powers_l278_27822


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_sevenths_l278_27806

theorem opposite_of_negative_three_sevenths :
  let x : ℚ := -3/7
  let y : ℚ := 3/7
  (∀ a b : ℚ, (a + b = 0 ↔ b = -a)) →
  y = -x :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_sevenths_l278_27806


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l278_27804

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum :
  let a : ℚ := 1/4
  let r : ℚ := 1/4
  let n : ℕ := 6
  geometric_sum a r n = 1365/4096 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l278_27804


namespace NUMINAMATH_CALUDE_power_difference_l278_27826

theorem power_difference (a : ℝ) (m n : ℤ) (h1 : a^m = 9) (h2 : a^n = 3) : a^(m-n) = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_l278_27826


namespace NUMINAMATH_CALUDE_ant_return_probability_60_l278_27843

/-- The probability of an ant returning to its starting vertex on a tetrahedron after n random edge traversals -/
def ant_return_probability (n : ℕ) : ℚ :=
  (3^(n-1) + 1) / (4 * 3^(n-1))

/-- The theorem stating the probability of an ant returning to its starting vertex on a tetrahedron after 60 random edge traversals -/
theorem ant_return_probability_60 :
  ant_return_probability 60 = (3^59 + 1) / (4 * 3^59) := by
  sorry

end NUMINAMATH_CALUDE_ant_return_probability_60_l278_27843


namespace NUMINAMATH_CALUDE_simplify_expression_simplify_and_evaluate_l278_27887

-- Problem 1
theorem simplify_expression (a b : ℝ) : a + 2*b + 3*a - 2*b = 4*a := by sorry

-- Problem 2
theorem simplify_and_evaluate : (2*(2^2) - 3*2*1 + 8) - (5*2*1 - 4*(2^2) + 8) = 8 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_simplify_and_evaluate_l278_27887


namespace NUMINAMATH_CALUDE_min_value_expression_l278_27880

theorem min_value_expression (a b : ℝ) (h1 : ab - 4*a - b + 1 = 0) (h2 : a > 1) :
  ∀ x y : ℝ, x * y - 4*x - y + 1 = 0 → x > 1 → (a + 1) * (b + 2) ≤ (x + 1) * (y + 2) ∧
  ∃ a₀ b₀ : ℝ, a₀ * b₀ - 4*a₀ - b₀ + 1 = 0 ∧ a₀ > 1 ∧ (a₀ + 1) * (b₀ + 2) = 27 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l278_27880


namespace NUMINAMATH_CALUDE_find_B_l278_27870

theorem find_B (A B C : ℕ) (h1 : A = 520) (h2 : C = A + 204) (h3 : C = B + 179) : B = 545 := by
  sorry

end NUMINAMATH_CALUDE_find_B_l278_27870


namespace NUMINAMATH_CALUDE_f_derivative_l278_27854

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)

theorem f_derivative :
  deriv f = λ x => -2 * Real.sin (2 * x) := by sorry

end NUMINAMATH_CALUDE_f_derivative_l278_27854


namespace NUMINAMATH_CALUDE_jordans_initial_weight_jordans_weight_proof_l278_27861

/-- Calculates Jordan's initial weight based on his weight loss program and current weight -/
theorem jordans_initial_weight (initial_loss_rate : ℕ) (initial_weeks : ℕ) 
  (subsequent_loss_rate : ℕ) (subsequent_weeks : ℕ) (current_weight : ℕ) : ℕ :=
  let total_loss := initial_loss_rate * initial_weeks + subsequent_loss_rate * subsequent_weeks
  current_weight + total_loss

/-- Proves that Jordan's initial weight was 250 pounds -/
theorem jordans_weight_proof : 
  jordans_initial_weight 3 4 2 8 222 = 250 := by
  sorry

end NUMINAMATH_CALUDE_jordans_initial_weight_jordans_weight_proof_l278_27861


namespace NUMINAMATH_CALUDE_matchstick_sequence_l278_27884

/-- 
Given a sequence where:
- The first term is 4
- Each subsequent term increases by 3
This theorem proves that the 20th term of the sequence is 61.
-/
theorem matchstick_sequence (n : ℕ) : 
  let sequence : ℕ → ℕ := λ k => 4 + 3 * (k - 1)
  sequence 20 = 61 := by
  sorry

end NUMINAMATH_CALUDE_matchstick_sequence_l278_27884


namespace NUMINAMATH_CALUDE_cinnamon_nutmeg_difference_l278_27831

/-- The amount of cinnamon used in tablespoons -/
def cinnamon : ℚ := 0.6666666666666666

/-- The amount of nutmeg used in tablespoons -/
def nutmeg : ℚ := 0.5

/-- The difference between cinnamon and nutmeg amounts -/
def difference : ℚ := cinnamon - nutmeg

theorem cinnamon_nutmeg_difference : difference = 0.1666666666666666 := by sorry

end NUMINAMATH_CALUDE_cinnamon_nutmeg_difference_l278_27831


namespace NUMINAMATH_CALUDE_total_time_is_twelve_years_l278_27892

/-- Represents the time taken for each activity in months -/
structure ActivityTime where
  shape : ℕ
  climb_learn : ℕ
  climb_each : ℕ
  dive_learn : ℕ
  dive_caves : ℕ

/-- Calculates the total time taken for all activities -/
def total_time (t : ActivityTime) (num_summits : ℕ) : ℕ :=
  t.shape + t.climb_learn + (num_summits * t.climb_each) + t.dive_learn + t.dive_caves

/-- Theorem stating that the total time to complete all goals is 12 years -/
theorem total_time_is_twelve_years (t : ActivityTime) (num_summits : ℕ) :
  t.shape = 24 ∧ 
  t.climb_learn = 2 * t.shape ∧ 
  num_summits = 7 ∧ 
  t.climb_each = 5 ∧ 
  t.dive_learn = 13 ∧ 
  t.dive_caves = 24 →
  total_time t num_summits = 12 * 12 := by
  sorry

#check total_time_is_twelve_years

end NUMINAMATH_CALUDE_total_time_is_twelve_years_l278_27892


namespace NUMINAMATH_CALUDE_gamma_bank_lowest_savings_l278_27820

def initial_funds : ℝ := 150000
def total_cost : ℝ := 201200

def rebs_bank_interest : ℝ := 2720.33
def gamma_bank_interest : ℝ := 3375.00
def tisi_bank_interest : ℝ := 2349.13
def btv_bank_interest : ℝ := 2264.11

def amount_to_save (interest : ℝ) : ℝ :=
  total_cost - initial_funds - interest

theorem gamma_bank_lowest_savings :
  let rebs_savings := amount_to_save rebs_bank_interest
  let gamma_savings := amount_to_save gamma_bank_interest
  let tisi_savings := amount_to_save tisi_bank_interest
  let btv_savings := amount_to_save btv_bank_interest
  (gamma_savings ≤ rebs_savings) ∧
  (gamma_savings ≤ tisi_savings) ∧
  (gamma_savings ≤ btv_savings) :=
by sorry

end NUMINAMATH_CALUDE_gamma_bank_lowest_savings_l278_27820


namespace NUMINAMATH_CALUDE_solve_equations_l278_27858

theorem solve_equations :
  (∃ x : ℝ, 4 * x - 3 * (20 - x) + 4 = 0 ∧ x = 8) ∧
  (∃ x : ℝ, (2 * x + 1) / 3 = 1 - (x - 1) / 5 ∧ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_solve_equations_l278_27858


namespace NUMINAMATH_CALUDE_rational_inequality_solution_l278_27808

theorem rational_inequality_solution (x : ℝ) :
  (x ≠ -1 ∧ x ≠ 2) →
  ((x^2 + 3*x - 4) / (x^2 - x - 2) > 0 ↔ x > 2 ∨ x < -4) :=
by sorry

end NUMINAMATH_CALUDE_rational_inequality_solution_l278_27808


namespace NUMINAMATH_CALUDE_find_n_l278_27890

theorem find_n (d Q r m n : ℝ) (hr : r > 0) (hm : m < (1 + r)^n) 
  (hQ : Q = d / ((1 + r)^n - m)) :
  n = Real.log (d / Q + m) / Real.log (1 + r) := by
  sorry

end NUMINAMATH_CALUDE_find_n_l278_27890


namespace NUMINAMATH_CALUDE_perimeter_inequality_l278_27813

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculates the perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := sorry

/-- Checks if a triangle is acute -/
def isAcute (t : Triangle) : Prop := sorry

/-- Calculates the foot of the perpendicular from a point to a line -/
def perpendicularFoot (p : Point) (l : Point × Point) : Point := sorry

/-- Main theorem -/
theorem perimeter_inequality 
  (ABC : Triangle) 
  (h_acute : isAcute ABC) 
  (D : Point) (E : Point) (F : Point)
  (P : Point) (Q : Point) (R : Point)
  (h_D : D = perpendicularFoot ABC.A (ABC.B, ABC.C))
  (h_E : E = perpendicularFoot ABC.B (ABC.C, ABC.A))
  (h_F : F = perpendicularFoot ABC.C (ABC.A, ABC.B))
  (h_P : P = perpendicularFoot ABC.A (E, F))
  (h_Q : Q = perpendicularFoot ABC.B (F, D))
  (h_R : R = perpendicularFoot ABC.C (D, E))
  : perimeter ABC * perimeter {A := P, B := Q, C := R} ≥ (perimeter {A := D, B := E, C := F})^2 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_inequality_l278_27813


namespace NUMINAMATH_CALUDE_antonia_pill_box_l278_27819

def pill_box_problem (num_supplements : ℕ) 
                     (num_large_bottles : ℕ) 
                     (num_small_bottles : ℕ) 
                     (pills_per_large_bottle : ℕ) 
                     (pills_per_small_bottle : ℕ) 
                     (pills_left : ℕ) 
                     (num_weeks : ℕ) : Prop :=
  let total_pills := num_large_bottles * pills_per_large_bottle + 
                     num_small_bottles * pills_per_small_bottle
  let pills_used := total_pills - pills_left
  let days_filled := num_weeks * 7
  pills_used / num_supplements = days_filled

theorem antonia_pill_box : 
  pill_box_problem 5 3 2 120 30 350 2 = true :=
sorry

end NUMINAMATH_CALUDE_antonia_pill_box_l278_27819


namespace NUMINAMATH_CALUDE_half_of_half_equals_half_l278_27847

theorem half_of_half_equals_half (x : ℝ) : (1/2 * (1/2 * x) = 1/2) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_half_of_half_equals_half_l278_27847


namespace NUMINAMATH_CALUDE_jackies_break_duration_l278_27816

/-- Represents Jackie's push-up performance --/
structure PushupPerformance where
  pushups_per_10sec : ℕ
  pushups_per_minute_with_breaks : ℕ
  num_breaks : ℕ

/-- Calculates the duration of each break in seconds --/
def break_duration (perf : PushupPerformance) : ℕ :=
  let pushups_per_minute := perf.pushups_per_10sec * 6
  let total_break_time := (pushups_per_minute - perf.pushups_per_minute_with_breaks) * (10 / perf.pushups_per_10sec)
  total_break_time / perf.num_breaks

/-- Theorem: Jackie's break duration is 8 seconds --/
theorem jackies_break_duration :
  let jackie : PushupPerformance := ⟨5, 22, 2⟩
  break_duration jackie = 8 := by
  sorry

end NUMINAMATH_CALUDE_jackies_break_duration_l278_27816


namespace NUMINAMATH_CALUDE_family_reunion_attendance_l278_27824

/-- Calculates the number of people served given the amount of pasta used,
    based on a recipe where 2 pounds of pasta serves 7 people. -/
def people_served (pasta_pounds : ℚ) : ℚ :=
  (pasta_pounds / 2) * 7

/-- Theorem stating that 10 pounds of pasta will serve 35 people,
    given a recipe where 2 pounds of pasta serves 7 people. -/
theorem family_reunion_attendance :
  people_served 10 = 35 := by
  sorry

end NUMINAMATH_CALUDE_family_reunion_attendance_l278_27824


namespace NUMINAMATH_CALUDE_delaware_cell_phones_l278_27825

/-- The number of cell phones in Delaware -/
def cell_phones_in_delaware (population : ℕ) (phones_per_thousand : ℕ) : ℕ :=
  (population / 1000) * phones_per_thousand

/-- Theorem stating the number of cell phones in Delaware -/
theorem delaware_cell_phones :
  cell_phones_in_delaware 974000 673 = 655502 := by
  sorry

end NUMINAMATH_CALUDE_delaware_cell_phones_l278_27825


namespace NUMINAMATH_CALUDE_quadratic_sum_zero_l278_27842

def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_sum_zero
  (a b c : ℝ)
  (h1 : ∃ (x : ℝ), ∀ (y : ℝ), quadratic a b c y ≥ quadratic a b c x)
  (h2 : quadratic a b c 1 = 0)
  (h3 : quadratic a b c (-3) = 0)
  (h4 : ∃ (x : ℝ), quadratic a b c x = 45)
  : a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_zero_l278_27842


namespace NUMINAMATH_CALUDE_unique_number_l278_27821

def is_valid_number (n : ℕ) : Prop :=
  ∃ (x y z : ℕ),
    n = 100 * x + 10 * y + z ∧
    x ≥ 1 ∧ x ≤ 9 ∧
    y ≥ 0 ∧ y ≤ 9 ∧
    z ≥ 0 ∧ z ≤ 9 ∧
    100 * z + 10 * y + x = n + 198 ∧
    100 * x + 10 * z + y = n + 9 ∧
    x^2 + y^2 + z^2 = 4 * (x + y + z) + 2

theorem unique_number : ∃! n : ℕ, is_valid_number n ∧ n = 345 :=
sorry

end NUMINAMATH_CALUDE_unique_number_l278_27821


namespace NUMINAMATH_CALUDE_farmer_trees_problem_l278_27839

theorem farmer_trees_problem : ∃ x : ℕ, 
  (∃ n : ℕ, x + 20 = n^2) ∧ 
  (∃ m : ℕ, x - 39 = m^2) ∧ 
  x = 880 := by
  sorry

end NUMINAMATH_CALUDE_farmer_trees_problem_l278_27839


namespace NUMINAMATH_CALUDE_max_missable_problems_l278_27860

theorem max_missable_problems (total_problems : ℕ) (passing_percentage : ℚ) 
  (hp : passing_percentage = 85 / 100) (ht : total_problems = 40) :
  ⌊total_problems * (1 - passing_percentage)⌋ = 6 :=
sorry

end NUMINAMATH_CALUDE_max_missable_problems_l278_27860


namespace NUMINAMATH_CALUDE_marble_drawing_probability_l278_27863

theorem marble_drawing_probability : 
  let total_marbles : ℕ := 10
  let blue_marbles : ℕ := 4
  let green_marbles : ℕ := 6
  let prob_blue : ℚ := blue_marbles / total_marbles
  let prob_green_after_blue : ℚ := green_marbles / (total_marbles - 1)
  let prob_green_after_blue_green : ℚ := (green_marbles - 1) / (total_marbles - 2)
  prob_blue * prob_green_after_blue * prob_green_after_blue_green = 1 / 6 :=
by sorry

end NUMINAMATH_CALUDE_marble_drawing_probability_l278_27863


namespace NUMINAMATH_CALUDE_train_journey_time_l278_27878

theorem train_journey_time 
  (distance : ℝ) 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (time1 : ℝ) 
  (time2 : ℝ) :
  speed1 = 48 →
  speed2 = 60 →
  time2 = 2/3 →
  distance = speed1 * time1 →
  distance = speed2 * time2 →
  time1 = 5/6 :=
by sorry

end NUMINAMATH_CALUDE_train_journey_time_l278_27878
