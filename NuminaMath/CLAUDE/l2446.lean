import Mathlib

namespace NUMINAMATH_CALUDE_lindas_coins_l2446_244640

theorem lindas_coins (total_coins : ℕ) (nickel_value dime_value : ℚ) 
  (swap_increase : ℚ) (h1 : total_coins = 30) 
  (h2 : nickel_value = 5/100) (h3 : dime_value = 10/100)
  (h4 : swap_increase = 90/100) : ∃ (nickels : ℕ), 
  nickels * nickel_value + (total_coins - nickels) * dime_value = 180/100 := by
  sorry

end NUMINAMATH_CALUDE_lindas_coins_l2446_244640


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l2446_244679

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  min_value : ℝ
  min_x : ℝ
  point_1 : ℝ × ℝ
  point_2 : ℝ × ℝ

/-- The theorem statement -/
theorem quadratic_function_theorem (f : QuadraticFunction) 
  (h1 : f.min_value = -3)
  (h2 : f.min_x = -2)
  (h3 : f.point_1 = (1, 10))
  (h4 : f.point_2.1 = 3) :
  f.point_2.2 = 298 / 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l2446_244679


namespace NUMINAMATH_CALUDE_domain_of_f_l2446_244660

def f (x : ℝ) : ℝ := (x - 3) ^ (1/3) + (5 - x) ^ (1/3) + (x + 1) ^ (1/2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≥ -1} :=
sorry

end NUMINAMATH_CALUDE_domain_of_f_l2446_244660


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2446_244676

theorem quadratic_minimum (b : ℝ) : 
  ∃ (min : ℝ), (∀ x : ℝ, (1/2) * x^2 + 5*x - 3 ≥ (1/2) * min^2 + 5*min - 3) ∧ min = -5 :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2446_244676


namespace NUMINAMATH_CALUDE_number_exceeding_fraction_l2446_244632

theorem number_exceeding_fraction : ∃ x : ℚ, x = (3/8) * x + 15 ∧ x = 24 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_fraction_l2446_244632


namespace NUMINAMATH_CALUDE_parabola_point_distance_l2446_244628

/-- A point on a parabola with a specific distance to the focus has a specific distance to the y-axis -/
theorem parabola_point_distance (P : ℝ × ℝ) : 
  (P.2)^2 = 8 * P.1 →  -- P is on the parabola y^2 = 8x
  ((P.1 - 2)^2 + P.2^2)^(1/2 : ℝ) = 6 →  -- Distance from P to focus (2, 0) is 6
  P.1 = 4 :=  -- Distance from P to y-axis is 4
by sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l2446_244628


namespace NUMINAMATH_CALUDE_pair_farm_animals_l2446_244642

/-- Represents the number of ways to pair animals of different species -/
def pairAnimals (cows pigs horses : ℕ) : ℕ :=
  let cowPigPairs := cows * pigs
  let remainingPairs := Nat.factorial horses
  cowPigPairs * remainingPairs

/-- Theorem stating the number of ways to pair 5 cows, 4 pigs, and 7 horses -/
theorem pair_farm_animals :
  pairAnimals 5 4 7 = 100800 := by
  sorry

#eval pairAnimals 5 4 7

end NUMINAMATH_CALUDE_pair_farm_animals_l2446_244642


namespace NUMINAMATH_CALUDE_bus_trip_speed_l2446_244694

/-- The average speed of a bus trip, given the conditions of the problem -/
def average_speed : ℝ → Prop :=
  fun v =>
    v > 0 ∧
    560 / v - 560 / (v + 10) = 2

theorem bus_trip_speed : ∃ v : ℝ, average_speed v ∧ v = 50 :=
  sorry

end NUMINAMATH_CALUDE_bus_trip_speed_l2446_244694


namespace NUMINAMATH_CALUDE_min_upper_bound_fraction_l2446_244623

theorem min_upper_bound_fraction (a₁ a₂ a₃ : ℝ) (h : a₁ ≠ 0 ∨ a₂ ≠ 0 ∨ a₃ ≠ 0) :
  ∃ M : ℝ, M = Real.sqrt 2 / 2 ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x^2 + y^2 = 2 →
    (x * a₁ * a₂ + y * a₂ * a₃) / (a₁^2 + a₂^2 + a₃^2) ≤ M) ∧
  ∀ ε > 0, ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x^2 + y^2 = 2 ∧
    (x * a₁ * a₂ + y * a₂ * a₃) / (a₁^2 + a₂^2 + a₃^2) > M - ε :=
by sorry

end NUMINAMATH_CALUDE_min_upper_bound_fraction_l2446_244623


namespace NUMINAMATH_CALUDE_correct_number_of_pupils_l2446_244611

/-- The number of pupils in a class where an error in one pupil's marks
    caused the class average to increase by half. -/
def number_of_pupils : ℕ :=
  let mark_increase : ℕ := 85 - 45
  let average_increase : ℚ := 1/2
  (2 * mark_increase : ℕ)

theorem correct_number_of_pupils :
  number_of_pupils = 80 :=
sorry

end NUMINAMATH_CALUDE_correct_number_of_pupils_l2446_244611


namespace NUMINAMATH_CALUDE_inverse_36_mod_101_l2446_244661

theorem inverse_36_mod_101 : ∃ x : ℤ, 36 * x ≡ 1 [ZMOD 101] :=
by
  use 87
  sorry

end NUMINAMATH_CALUDE_inverse_36_mod_101_l2446_244661


namespace NUMINAMATH_CALUDE_garrison_provisions_duration_l2446_244658

/-- The number of days provisions last for a garrison given reinforcements --/
theorem garrison_provisions_duration 
  (initial_men : ℕ) 
  (reinforcement_men : ℕ) 
  (days_before_reinforcement : ℕ) 
  (days_after_reinforcement : ℕ) 
  (h1 : initial_men = 2000)
  (h2 : reinforcement_men = 1900)
  (h3 : days_before_reinforcement = 15)
  (h4 : days_after_reinforcement = 20) :
  ∃ (initial_days : ℕ), 
    initial_days * initial_men = 
      (initial_men + reinforcement_men) * days_after_reinforcement + 
      initial_men * days_before_reinforcement ∧
    initial_days = 54 := by
  sorry

end NUMINAMATH_CALUDE_garrison_provisions_duration_l2446_244658


namespace NUMINAMATH_CALUDE_inverse_function_parameter_l2446_244629

/-- Given a function f and its inverse, find the value of b -/
theorem inverse_function_parameter (f : ℝ → ℝ) (b : ℝ) : 
  (∀ x, f x = 1 / (2 * x + b)) →
  (∀ x, f⁻¹ x = (2 - 3 * x) / (3 * x)) →
  b = -2 := by
sorry

end NUMINAMATH_CALUDE_inverse_function_parameter_l2446_244629


namespace NUMINAMATH_CALUDE_binomial_coefficient_times_n_l2446_244616

theorem binomial_coefficient_times_n (n : ℕ+) : n * Nat.choose 4 3 = 4 * n := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_times_n_l2446_244616


namespace NUMINAMATH_CALUDE_parabola_through_point_l2446_244689

/-- A parabola passing through the point (4, -2) has either the equation y² = x or x² = -8y -/
theorem parabola_through_point (P : ℝ × ℝ) (h : P = (4, -2)) :
  (∃ (x y : ℝ), y^2 = x ∧ P = (x, y)) ∨ (∃ (x y : ℝ), x^2 = -8*y ∧ P = (x, y)) :=
sorry

end NUMINAMATH_CALUDE_parabola_through_point_l2446_244689


namespace NUMINAMATH_CALUDE_largest_after_removal_l2446_244646

/-- Represents the initial number as a string -/
def initial_number : String := "123456789101112131415...99100"

/-- Represents the final number after digit removal as a string -/
def final_number : String := "9999978596061...99100"

/-- Function to remove digits from a string -/
def remove_digits (s : String) (n : Nat) : String := sorry

/-- Function to compare two strings as numbers -/
def compare_as_numbers (s1 s2 : String) : Bool := sorry

/-- Theorem stating that the final_number is the largest possible after removing 100 digits -/
theorem largest_after_removal :
  ∀ (s : String),
    s.length = initial_number.length - 100 →
    s = remove_digits initial_number 100 →
    compare_as_numbers final_number s = true :=
sorry

end NUMINAMATH_CALUDE_largest_after_removal_l2446_244646


namespace NUMINAMATH_CALUDE_sin_alpha_for_point_on_terminal_side_l2446_244618

/-- 
If the terminal side of angle α passes through point P(m, 2m) where m > 0, 
then sin(α) = 2√5/5.
-/
theorem sin_alpha_for_point_on_terminal_side (m : ℝ) (α : ℝ) 
  (h1 : m > 0) 
  (h2 : ∃ (x y : ℝ), x = m ∧ y = 2*m ∧ 
       x = Real.cos α * Real.sqrt (m^2 + (2*m)^2) ∧ 
       y = Real.sin α * Real.sqrt (m^2 + (2*m)^2)) : 
  Real.sin α = 2 * Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_for_point_on_terminal_side_l2446_244618


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l2446_244687

/-- Calculate the profit percentage given the sale price including tax, tax rate, and cost price -/
theorem profit_percentage_calculation
  (sale_price_with_tax : ℝ)
  (tax_rate : ℝ)
  (cost_price : ℝ)
  (h1 : sale_price_with_tax = 616)
  (h2 : tax_rate = 0.1)
  (h3 : cost_price = 545.13) :
  ∃ (profit_percentage : ℝ),
    abs (profit_percentage - 2.73) < 0.01 ∧
    profit_percentage = ((sale_price_with_tax / (1 + tax_rate) - cost_price) / cost_price) * 100 :=
by sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l2446_244687


namespace NUMINAMATH_CALUDE_song_duration_l2446_244654

theorem song_duration (initial_songs : ℕ) (added_songs : ℕ) (total_time : ℕ) :
  initial_songs = 25 →
  added_songs = 10 →
  total_time = 105 →
  (initial_songs + added_songs) * (total_time / (initial_songs + added_songs)) = total_time →
  total_time / (initial_songs + added_songs) = 3 :=
by sorry

end NUMINAMATH_CALUDE_song_duration_l2446_244654


namespace NUMINAMATH_CALUDE_least_divisible_by_240_sixty_cube_divisible_by_240_least_positive_integer_cube_divisible_by_240_l2446_244675

theorem least_divisible_by_240 (a : ℕ) : a > 0 ∧ a^3 % 240 = 0 → a ≥ 60 := by
  sorry

theorem sixty_cube_divisible_by_240 : (60 : ℕ)^3 % 240 = 0 := by
  sorry

theorem least_positive_integer_cube_divisible_by_240 :
  ∃ (a : ℕ), a > 0 ∧ a^3 % 240 = 0 ∧ ∀ (b : ℕ), b > 0 ∧ b^3 % 240 = 0 → b ≥ a :=
by
  sorry

end NUMINAMATH_CALUDE_least_divisible_by_240_sixty_cube_divisible_by_240_least_positive_integer_cube_divisible_by_240_l2446_244675


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_l2446_244650

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallel_plane_line : Plane → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (intersects : Plane → Plane → Line → Prop)
variable (in_plane : Line → Plane → Prop)

-- Define the theorem
theorem line_parallel_to_plane 
  (m n : Line) (α β : Plane) 
  (h1 : ¬ (m = n)) -- m and n are non-overlapping
  (h2 : ¬ (α = β)) -- α and β are non-overlapping
  (h3 : intersects α β n) -- α intersects β at n
  (h4 : ¬ in_plane m α) -- m is not in α
  (h5 : parallel m n) -- m is parallel to n
  : parallel_plane_line α m := by sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_l2446_244650


namespace NUMINAMATH_CALUDE_javier_speech_time_l2446_244656

/-- Represents the time spent on different activities of speech preparation --/
structure SpeechTime where
  outline : ℕ
  write : ℕ
  practice : ℕ

/-- Calculates the total time spent on speech preparation --/
def totalTime (st : SpeechTime) : ℕ :=
  st.outline + st.write + st.practice

/-- Theorem stating the total time Javier spends on his speech --/
theorem javier_speech_time :
  ∃ (st : SpeechTime),
    st.outline = 30 ∧
    st.write = st.outline + 28 ∧
    st.practice = st.write / 2 ∧
    totalTime st = 117 := by
  sorry


end NUMINAMATH_CALUDE_javier_speech_time_l2446_244656


namespace NUMINAMATH_CALUDE_fraction_equality_l2446_244655

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (5 * x + 2 * y) / (2 * x - 5 * y) = 3) : 
  (x + 3 * y) / (3 * x - y) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2446_244655


namespace NUMINAMATH_CALUDE_unique_intersection_point_f_bijective_f_inv_is_inverse_l2446_244663

/-- The cubic function f(x) = x^3 + 6x^2 + 9x + 15 -/
def f (x : ℝ) : ℝ := x^3 + 6*x^2 + 9*x + 15

/-- The theorem stating that the unique intersection point of f and its inverse is (-3, -3) -/
theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, p.1 = f p.2 ∧ p.2 = f p.1 ∧ p = (-3, -3) := by
  sorry

/-- The function f is bijective -/
theorem f_bijective : Function.Bijective f := by
  sorry

/-- The inverse function of f exists -/
noncomputable def f_inv : ℝ → ℝ := Function.invFun f

/-- The theorem stating that f_inv is indeed the inverse of f -/
theorem f_inv_is_inverse :
  Function.LeftInverse f_inv f ∧ Function.RightInverse f_inv f := by
  sorry

end NUMINAMATH_CALUDE_unique_intersection_point_f_bijective_f_inv_is_inverse_l2446_244663


namespace NUMINAMATH_CALUDE_homer_candy_crush_score_l2446_244677

theorem homer_candy_crush_score (first_try : ℕ) (second_try : ℕ) (third_try : ℕ) 
  (h1 : first_try = 400)
  (h2 : second_try < first_try)
  (h3 : third_try = 2 * second_try)
  (h4 : first_try + second_try + third_try = 1390) :
  second_try = 330 := by
  sorry

end NUMINAMATH_CALUDE_homer_candy_crush_score_l2446_244677


namespace NUMINAMATH_CALUDE_charles_journey_l2446_244696

/-- Represents the distance traveled by Charles -/
def total_distance : ℝ := 1800

/-- Represents the speed for the first half of the journey -/
def speed1 : ℝ := 90

/-- Represents the speed for the second half of the journey -/
def speed2 : ℝ := 180

/-- Represents the total time of the journey -/
def total_time : ℝ := 30

/-- Theorem stating that given the conditions of Charles' journey, the total distance is 1800 miles -/
theorem charles_journey :
  (total_distance / 2 / speed1 + total_distance / 2 / speed2 = total_time) →
  total_distance = 1800 :=
by sorry

end NUMINAMATH_CALUDE_charles_journey_l2446_244696


namespace NUMINAMATH_CALUDE_peach_basket_ratios_and_percentages_l2446_244636

/-- Represents the number of peaches of each color in the basket -/
structure PeachBasket where
  red : ℕ
  yellow : ℕ
  green : ℕ
  orange : ℕ

/-- Calculates the total number of peaches in the basket -/
def totalPeaches (basket : PeachBasket) : ℕ :=
  basket.red + basket.yellow + basket.green + basket.orange

/-- Represents a ratio as a pair of natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Calculates the ratio of a specific color to the total -/
def colorRatio (count : ℕ) (total : ℕ) : Ratio :=
  let gcd := Nat.gcd count total
  { numerator := count / gcd, denominator := total / gcd }

/-- Calculates the percentage of a specific color -/
def colorPercentage (count : ℕ) (total : ℕ) : Float :=
  (count.toFloat / total.toFloat) * 100

theorem peach_basket_ratios_and_percentages
  (basket : PeachBasket)
  (h_red : basket.red = 8)
  (h_yellow : basket.yellow = 14)
  (h_green : basket.green = 6)
  (h_orange : basket.orange = 4) :
  let total := totalPeaches basket
  (colorRatio basket.green total = Ratio.mk 3 16) ∧
  (colorRatio basket.yellow total = Ratio.mk 7 16) ∧
  (colorPercentage basket.green total = 18.75) ∧
  (colorPercentage basket.yellow total = 43.75) := by
  sorry


end NUMINAMATH_CALUDE_peach_basket_ratios_and_percentages_l2446_244636


namespace NUMINAMATH_CALUDE_book_arrangement_count_book_arrangement_theorem_l2446_244601

theorem book_arrangement_count : ℕ :=
  let total_books : ℕ := 11
  let arabic_books : ℕ := 3
  let german_books : ℕ := 3
  let spanish_books : ℕ := 5
  let arabic_group : ℕ := 1  -- Treat Arabic books as one unit
  let spanish_group : ℕ := 1  -- Treat Spanish books as one unit
  let german_group : ℕ := 1  -- Treat German books as one ordered unit
  let total_groups : ℕ := arabic_group + spanish_group + german_group
  let group_arrangements : ℕ := Nat.factorial total_groups
  let arabic_arrangements : ℕ := Nat.factorial arabic_books
  let spanish_arrangements : ℕ := Nat.factorial spanish_books

  group_arrangements * arabic_arrangements * spanish_arrangements

-- Prove that book_arrangement_count equals 4320
theorem book_arrangement_theorem : book_arrangement_count = 4320 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_book_arrangement_theorem_l2446_244601


namespace NUMINAMATH_CALUDE_orange_juice_fraction_l2446_244610

theorem orange_juice_fraction (pitcher_capacity : ℚ) 
  (pitcher1_orange : ℚ) (pitcher1_apple : ℚ)
  (pitcher2_orange : ℚ) (pitcher2_apple : ℚ) :
  pitcher_capacity = 800 →
  pitcher1_orange = 1/4 →
  pitcher1_apple = 1/8 →
  pitcher2_orange = 1/5 →
  pitcher2_apple = 1/10 →
  (pitcher_capacity * pitcher1_orange + pitcher_capacity * pitcher2_orange) / (2 * pitcher_capacity) = 9/40 := by
  sorry

#check orange_juice_fraction

end NUMINAMATH_CALUDE_orange_juice_fraction_l2446_244610


namespace NUMINAMATH_CALUDE_parabola_focus_l2446_244657

/-- The parabola equation -/
def parabola_eq (x y : ℝ) : Prop := y = -1/8 * x^2

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (0, 2)

/-- Theorem: The focus of the parabola y = -1/8 * x^2 is (0, 2) -/
theorem parabola_focus :
  ∀ (x y : ℝ), parabola_eq x y →
  ∃ (d : ℝ), 
    (x - focus.1)^2 + (y - focus.2)^2 = (y - d)^2 ∧
    (∀ (x' y' : ℝ), parabola_eq x' y' → 
      (x' - focus.1)^2 + (y' - focus.2)^2 = (y' - d)^2) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l2446_244657


namespace NUMINAMATH_CALUDE_f_triple_3_l2446_244609

def f (x : ℝ) : ℝ := 3 * x + 2

theorem f_triple_3 : f (f (f 3)) = 107 := by
  sorry

end NUMINAMATH_CALUDE_f_triple_3_l2446_244609


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2446_244635

theorem arithmetic_calculation : 4 * 6 * 8 - 24 / 3 = 184 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2446_244635


namespace NUMINAMATH_CALUDE_min_product_of_three_numbers_l2446_244627

theorem min_product_of_three_numbers (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  x + y + z = 2 → 
  x ≤ 3*y ∧ x ≤ 3*z ∧ y ≤ 3*x ∧ y ≤ 3*z ∧ z ≤ 3*x ∧ z ≤ 3*y → 
  x * y * z ≥ 1/9 := by
sorry

end NUMINAMATH_CALUDE_min_product_of_three_numbers_l2446_244627


namespace NUMINAMATH_CALUDE_gcf_of_45_and_75_l2446_244671

theorem gcf_of_45_and_75 : Nat.gcd 45 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_45_and_75_l2446_244671


namespace NUMINAMATH_CALUDE_employment_after_growth_and_new_category_l2446_244673

/-- Represents the employment data for town X -/
structure TownEmployment where
  initial_rate : ℝ
  annual_growth : ℝ
  years : ℕ
  male_percentage : ℝ
  tourism_percentage : ℝ
  female_edu_percentage : ℝ

/-- Theorem about employment percentages after growth and new category introduction -/
theorem employment_after_growth_and_new_category 
  (town : TownEmployment)
  (h_initial : town.initial_rate = 0.64)
  (h_growth : town.annual_growth = 0.02)
  (h_years : town.years = 5)
  (h_male : town.male_percentage = 0.55)
  (h_tourism : town.tourism_percentage = 0.1)
  (h_female_edu : town.female_edu_percentage = 0.6) :
  let final_rate := town.initial_rate + town.annual_growth * town.years
  let female_percentage := 1 - town.male_percentage
  (female_percentage = 0.45) ∧ 
  (town.female_edu_percentage > 0.5) := by
  sorry

#check employment_after_growth_and_new_category

end NUMINAMATH_CALUDE_employment_after_growth_and_new_category_l2446_244673


namespace NUMINAMATH_CALUDE_nested_bracket_equals_two_l2446_244692

-- Define the bracket operation
def bracket (x y z : ℚ) : ℚ := (x + y) / z

-- State the theorem
theorem nested_bracket_equals_two :
  bracket (bracket 45 15 60) (bracket 3 3 6) (bracket 20 10 30) = 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_bracket_equals_two_l2446_244692


namespace NUMINAMATH_CALUDE_sequence_relation_l2446_244664

-- Define the sequence u
def u (n : ℕ) : ℝ := 17^n * (n + 2)

-- State the theorem
theorem sequence_relation (a b : ℝ) :
  (∀ n : ℕ, u (n + 2) = a * u (n + 1) + b * u n) →
  a^2 - b = 144.5 :=
by sorry

end NUMINAMATH_CALUDE_sequence_relation_l2446_244664


namespace NUMINAMATH_CALUDE_section_B_students_l2446_244678

def section_A_students : ℕ := 50
def section_A_avg_weight : ℝ := 50
def section_B_avg_weight : ℝ := 70
def total_avg_weight : ℝ := 58.89

theorem section_B_students :
  ∃ x : ℕ, 
    (section_A_students * section_A_avg_weight + x * section_B_avg_weight) / (section_A_students + x) = total_avg_weight ∧
    x = 40 :=
by sorry

end NUMINAMATH_CALUDE_section_B_students_l2446_244678


namespace NUMINAMATH_CALUDE_rectangular_plot_width_l2446_244680

theorem rectangular_plot_width
  (length : ℝ)
  (num_poles : ℕ)
  (pole_distance : ℝ)
  (width : ℝ)
  (h1 : length = 60)
  (h2 : num_poles = 44)
  (h3 : pole_distance = 5)
  (h4 : 2 * (length + width) = pole_distance * num_poles) :
  width = 50 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_width_l2446_244680


namespace NUMINAMATH_CALUDE_identity_function_theorem_l2446_244667

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

theorem identity_function_theorem (f : ℕ+ → ℕ+) : 
  (∀ x y : ℕ+, is_perfect_square (x * f x + 2 * x * f y + (f y) ^ 2)) → 
  (∀ x : ℕ+, f x = x) :=
sorry

end NUMINAMATH_CALUDE_identity_function_theorem_l2446_244667


namespace NUMINAMATH_CALUDE_smallest_number_for_2_and_4_l2446_244625

def smallest_number (a b : ℕ) : ℕ := 
  if a ≤ b then 10 * a + b else 10 * b + a

theorem smallest_number_for_2_and_4 : 
  smallest_number 2 4 = 24 := by sorry

end NUMINAMATH_CALUDE_smallest_number_for_2_and_4_l2446_244625


namespace NUMINAMATH_CALUDE_min_median_length_l2446_244644

/-- In a right triangle with height h dropped onto the hypotenuse,
    the minimum length of the median that bisects the longer leg is (3/2) * h. -/
theorem min_median_length (h : ℝ) (h_pos : h > 0) :
  ∃ (m : ℝ), m ≥ (3/2) * h ∧
  ∀ (x y : ℝ), x > 0 → y > 0 → x * y = h^2 →
    ((x/2 + y)^2 + (h/2)^2).sqrt ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_median_length_l2446_244644


namespace NUMINAMATH_CALUDE_sum_of_roots_equation_l2446_244668

theorem sum_of_roots_equation (x : ℝ) : 
  let eq := (3*x + 4)*(x - 3) + (3*x + 4)*(x - 5) = 0
  ∃ (r₁ r₂ : ℝ), (3*r₁ + 4)*(r₁ - 3) + (3*r₁ + 4)*(r₁ - 5) = 0 ∧
                 (3*r₂ + 4)*(r₂ - 3) + (3*r₂ + 4)*(r₂ - 5) = 0 ∧
                 r₁ + r₂ = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equation_l2446_244668


namespace NUMINAMATH_CALUDE_correct_linear_regression_l2446_244608

-- Define the variables and constants
variable (x y : ℝ)
def x_mean : ℝ := 2.5
def y_mean : ℝ := 3.5

-- Define the linear regression equation
def linear_regression (x : ℝ) : ℝ := 0.4 * x + 2.5

-- State the theorem
theorem correct_linear_regression :
  (∃ r : ℝ, r > 0 ∧ (∀ x y : ℝ, y - y_mean = r * (x - x_mean))) →  -- Positive correlation
  (linear_regression x_mean = y_mean) →                           -- Passes through (x̄, ȳ)
  (∀ x : ℝ, linear_regression x = 0.4 * x + 2.5) :=               -- The equation is correct
by sorry

end NUMINAMATH_CALUDE_correct_linear_regression_l2446_244608


namespace NUMINAMATH_CALUDE_fingernail_growth_rate_l2446_244645

/-- Proves that the rate of fingernail growth is 0.1 inch per month given the specified conditions. -/
theorem fingernail_growth_rate 
  (current_age : ℕ) 
  (record_age : ℕ) 
  (current_length : ℚ) 
  (record_length : ℚ) 
  (h1 : current_age = 12) 
  (h2 : record_age = 32) 
  (h3 : current_length = 2) 
  (h4 : record_length = 26) : 
  (record_length - current_length) / ((record_age - current_age) * 12 : ℚ) = 1/10 := by
  sorry

#eval (26 - 2 : ℚ) / ((32 - 12) * 12 : ℚ)

end NUMINAMATH_CALUDE_fingernail_growth_rate_l2446_244645


namespace NUMINAMATH_CALUDE_selena_bashar_passes_l2446_244691

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ  -- speed in m/min
  radius : ℝ  -- radius of the lane in meters
  direction : ℤ  -- 1 for clockwise, -1 for counterclockwise

/-- Calculates the number of times two runners pass each other on a circular track -/
def number_of_passes (runner1 runner2 : Runner) (total_time : ℝ) (delay : ℝ) : ℕ :=
  sorry

theorem selena_bashar_passes : 
  let selena : Runner := ⟨200, 70, 1⟩
  let bashar : Runner := ⟨240, 80, -1⟩
  let total_time : ℝ := 35
  let delay : ℝ := 5
  number_of_passes selena bashar total_time delay = 21 := by
  sorry

end NUMINAMATH_CALUDE_selena_bashar_passes_l2446_244691


namespace NUMINAMATH_CALUDE_fish_brought_home_l2446_244669

/-- The number of fish Kendra caught -/
def kendras_catch : ℕ := 30

/-- The number of fish Ken released -/
def ken_released : ℕ := 3

/-- The number of fish Ken caught -/
def kens_catch : ℕ := 2 * kendras_catch

/-- The number of fish Ken brought home -/
def ken_brought_home : ℕ := kens_catch - ken_released

/-- The total number of fish brought home by Ken and Kendra -/
def total_brought_home : ℕ := ken_brought_home + kendras_catch

theorem fish_brought_home :
  total_brought_home = 87 :=
by sorry

end NUMINAMATH_CALUDE_fish_brought_home_l2446_244669


namespace NUMINAMATH_CALUDE_crackers_distribution_l2446_244605

theorem crackers_distribution
  (initial_crackers : ℕ)
  (num_friends : ℕ)
  (remaining_crackers : ℕ)
  (h1 : initial_crackers = 15)
  (h2 : num_friends = 5)
  (h3 : remaining_crackers = 10)
  (h4 : num_friends > 0) :
  (initial_crackers - remaining_crackers) / num_friends = 1 :=
by sorry

end NUMINAMATH_CALUDE_crackers_distribution_l2446_244605


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2446_244684

theorem complex_equation_solution (x : ℝ) : 
  (1 - 2*Complex.I) * (x + Complex.I) = 4 - 3*Complex.I → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2446_244684


namespace NUMINAMATH_CALUDE_apple_pyramid_theorem_l2446_244614

/-- Calculates the number of apples in a layer of the pyramid --/
def apples_in_layer (base_width : ℕ) (base_length : ℕ) (layer : ℕ) : ℕ :=
  (base_width - layer + 1) * (base_length - layer + 1)

/-- Calculates the total number of apples in the pyramid --/
def total_apples (base_width : ℕ) (base_length : ℕ) : ℕ :=
  let max_layers := min base_width base_length
  (List.range max_layers).foldl (fun acc layer => acc + apples_in_layer base_width base_length layer) 0

/-- The theorem stating that a pyramid with a 6x9 base contains 154 apples --/
theorem apple_pyramid_theorem :
  total_apples 6 9 = 154 := by
  sorry

end NUMINAMATH_CALUDE_apple_pyramid_theorem_l2446_244614


namespace NUMINAMATH_CALUDE_couplet_distribution_ways_l2446_244659

def num_widows : ℕ := 4
def num_long_couplets : ℕ := 4
def num_short_couplets : ℕ := 7

def long_couplets_per_widow : ℕ := 1
def short_couplets_for_one_widow : ℕ := 1
def short_couplets_for_three_widows : ℕ := 2

theorem couplet_distribution_ways :
  (Nat.choose num_long_couplets long_couplets_per_widow) *
  (Nat.choose num_short_couplets short_couplets_for_three_widows) *
  (Nat.choose (num_long_couplets - long_couplets_per_widow) long_couplets_per_widow) *
  (Nat.choose (num_short_couplets - short_couplets_for_three_widows) short_couplets_for_one_widow) *
  (Nat.choose (num_long_couplets - 2 * long_couplets_per_widow) long_couplets_per_widow) *
  (Nat.choose (num_short_couplets - short_couplets_for_three_widows - short_couplets_for_one_widow) short_couplets_for_three_widows) *
  (Nat.choose (num_long_couplets - 3 * long_couplets_per_widow) long_couplets_per_widow) *
  (Nat.choose (num_short_couplets - 2 * short_couplets_for_three_widows - short_couplets_for_one_widow) short_couplets_for_three_widows) = 15120 := by
  sorry

end NUMINAMATH_CALUDE_couplet_distribution_ways_l2446_244659


namespace NUMINAMATH_CALUDE_square_sum_ge_double_product_l2446_244638

theorem square_sum_ge_double_product :
  (∀ x y : ℝ, x^2 + y^2 ≥ 2*x*y) ↔ (x^2 + y^2 ≥ 2*x*y) := by sorry

end NUMINAMATH_CALUDE_square_sum_ge_double_product_l2446_244638


namespace NUMINAMATH_CALUDE_license_plate_equality_l2446_244693

def florida_plates : ℕ := 26^2 * 10^3 * 26^1
def north_dakota_plates : ℕ := 26^3 * 10^3

theorem license_plate_equality :
  florida_plates = north_dakota_plates :=
by sorry

end NUMINAMATH_CALUDE_license_plate_equality_l2446_244693


namespace NUMINAMATH_CALUDE_unique_solution_l2446_244653

def equation (y : ℝ) : Prop :=
  y ≠ 0 ∧ y ≠ 3 ∧ (3 * y^2 - 15 * y) / (y^2 - 3 * y) = y + 1

theorem unique_solution :
  ∃! y : ℝ, equation y :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2446_244653


namespace NUMINAMATH_CALUDE_train_catch_up_time_l2446_244690

/-- The problem of finding the time difference between two trains --/
theorem train_catch_up_time (goods_speed express_speed catch_up_time : ℝ) 
  (h1 : goods_speed = 36)
  (h2 : express_speed = 90)
  (h3 : catch_up_time = 4) :
  ∃ t : ℝ, t > 0 ∧ goods_speed * (t + catch_up_time) = express_speed * catch_up_time ∧ t = 6 := by
  sorry


end NUMINAMATH_CALUDE_train_catch_up_time_l2446_244690


namespace NUMINAMATH_CALUDE_total_candies_l2446_244648

/-- The total number of candies Linda and Chloe have together is 62, 
    given that Linda has 34 candies and Chloe has 28 candies. -/
theorem total_candies (linda_candies chloe_candies : ℕ) 
  (h1 : linda_candies = 34) 
  (h2 : chloe_candies = 28) : 
  linda_candies + chloe_candies = 62 := by
  sorry

end NUMINAMATH_CALUDE_total_candies_l2446_244648


namespace NUMINAMATH_CALUDE_eggs_left_for_breakfast_l2446_244620

def total_eggs : ℕ := 36

def eggs_for_crepes : ℕ := (2 * total_eggs) / 5

def eggs_after_crepes : ℕ := total_eggs - eggs_for_crepes

def eggs_for_cupcakes : ℕ := (3 * eggs_after_crepes) / 7

def eggs_after_cupcakes : ℕ := eggs_after_crepes - eggs_for_cupcakes

def eggs_for_quiche : ℕ := eggs_after_cupcakes / 2

def eggs_left : ℕ := eggs_after_cupcakes - eggs_for_quiche

theorem eggs_left_for_breakfast : eggs_left = 7 := by
  sorry

end NUMINAMATH_CALUDE_eggs_left_for_breakfast_l2446_244620


namespace NUMINAMATH_CALUDE_max_incorrect_answers_is_correct_l2446_244617

/-- The passing threshold for the exam as a percentage -/
def pass_threshold : ℝ := 85

/-- The total number of questions in the exam -/
def total_questions : ℕ := 50

/-- The maximum number of questions that can be answered incorrectly while still passing -/
def max_incorrect_answers : ℕ := 7

/-- Theorem stating that max_incorrect_answers is the maximum number of questions
    that can be answered incorrectly while still passing the exam -/
theorem max_incorrect_answers_is_correct :
  ∀ n : ℕ, 
    (n ≤ max_incorrect_answers ↔ 
      (total_questions - n : ℝ) / total_questions * 100 ≥ pass_threshold) :=
by sorry

end NUMINAMATH_CALUDE_max_incorrect_answers_is_correct_l2446_244617


namespace NUMINAMATH_CALUDE_hall_volume_l2446_244637

theorem hall_volume (length width : ℝ) (h : ℝ) : 
  length = 6 ∧ width = 6 ∧ 2 * (length * width) = 2 * (length * h) + 2 * (width * h) → 
  length * width * h = 108 := by
sorry

end NUMINAMATH_CALUDE_hall_volume_l2446_244637


namespace NUMINAMATH_CALUDE_range_of_y_l2446_244672

theorem range_of_y (y : ℝ) (h1 : 1 / y < 3) (h2 : 1 / y > -4) : y > 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_y_l2446_244672


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l2446_244699

theorem infinite_geometric_series_first_term 
  (r : ℝ) (S : ℝ) (a : ℝ) 
  (h1 : r = -1/3) 
  (h2 : S = 12) 
  (h3 : S = a / (1 - r)) : 
  a = 16 := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l2446_244699


namespace NUMINAMATH_CALUDE_count_distinct_values_l2446_244682

def is_pythagorean_triple (a b c : ℕ) : Prop := a^2 + b^2 = c^2

def satisfies_conditions (f : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, f n ∣ n^2016) ∧
  (∀ a b c : ℕ, is_pythagorean_triple a b c → f a * f b = f c)

theorem count_distinct_values :
  ∃ (S : Finset ℕ),
    (∀ f : ℕ → ℕ, satisfies_conditions f →
      (f 2014 + f 2 - f 2016) ∈ S) ∧
    S.card = 2^2017 - 1 :=
sorry

end NUMINAMATH_CALUDE_count_distinct_values_l2446_244682


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2446_244686

def A : Set ℝ := {x | x = Real.log 1 ∨ x = 1}
def B : Set ℝ := {x | x = -1 ∨ x = 0}

theorem union_of_A_and_B : A ∪ B = {x | x = -1 ∨ x = 0 ∨ x = 1} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2446_244686


namespace NUMINAMATH_CALUDE_cosine_sine_identity_l2446_244666

theorem cosine_sine_identity (θ : ℝ) 
  (h : Real.cos (π / 6 - θ) = 1 / 3) : 
  Real.cos (5 * π / 6 + θ) - Real.sin (θ - π / 6) ^ 2 = -11 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_identity_l2446_244666


namespace NUMINAMATH_CALUDE_percentage_equality_l2446_244600

theorem percentage_equality (x : ℝ) : 
  (60 / 100 : ℝ) * 500 = (50 / 100 : ℝ) * x → x = 600 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l2446_244600


namespace NUMINAMATH_CALUDE_kristin_bell_peppers_count_l2446_244621

/-- The number of bell peppers Kristin has -/
def kristin_bell_peppers : ℕ := 2

/-- The number of carrots Jaylen has -/
def jaylen_carrots : ℕ := 5

/-- The number of cucumbers Jaylen has -/
def jaylen_cucumbers : ℕ := 2

/-- The number of green beans Kristin has -/
def kristin_green_beans : ℕ := 20

/-- The total number of vegetables Jaylen has -/
def jaylen_total_vegetables : ℕ := 18

theorem kristin_bell_peppers_count :
  (jaylen_carrots + jaylen_cucumbers + 
   (kristin_green_beans / 2 - 3) + 
   (2 * kristin_bell_peppers) = jaylen_total_vegetables) →
  kristin_bell_peppers = 2 := by
  sorry

end NUMINAMATH_CALUDE_kristin_bell_peppers_count_l2446_244621


namespace NUMINAMATH_CALUDE_angle_twice_complement_l2446_244604

theorem angle_twice_complement (x : ℝ) : 
  (x = 2 * (90 - x)) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_twice_complement_l2446_244604


namespace NUMINAMATH_CALUDE_prob_same_length_hexagon_l2446_244649

/-- The set of all sides and diagonals of a regular hexagon -/
def T : Finset ℝ := sorry

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The total number of elements in T -/
def total_elements : ℕ := num_sides + num_diagonals

/-- The probability of selecting two segments of the same length -/
def prob_same_length : ℚ := 17 / 35

theorem prob_same_length_hexagon :
  (num_sides * (num_sides - 1) + num_diagonals * (num_diagonals - 1)) /
  (total_elements * (total_elements - 1)) = prob_same_length :=
sorry

end NUMINAMATH_CALUDE_prob_same_length_hexagon_l2446_244649


namespace NUMINAMATH_CALUDE_sticker_distribution_l2446_244670

/-- The number of ways to distribute n identical objects into k distinct containers -/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of stickers -/
def num_stickers : ℕ := 10

/-- The number of sheets of paper -/
def num_sheets : ℕ := 5

theorem sticker_distribution :
  distribute num_stickers num_sheets = 29 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l2446_244670


namespace NUMINAMATH_CALUDE_circle_area_ratio_after_tripling_radius_l2446_244624

theorem circle_area_ratio_after_tripling_radius (r : ℝ) (h : r > 0) :
  (π * r^2) / (π * (3*r)^2) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_after_tripling_radius_l2446_244624


namespace NUMINAMATH_CALUDE_monotonic_increasing_f_implies_m_leq_neg_three_l2446_244603

def f (m : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 - m*x + 1

theorem monotonic_increasing_f_implies_m_leq_neg_three :
  ∀ m : ℝ, (∀ x y : ℝ, -2 ≤ x ∧ x < y ∧ y ≤ 2 → f m x < f m y) →
  m ≤ -3 :=
by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_f_implies_m_leq_neg_three_l2446_244603


namespace NUMINAMATH_CALUDE_trig_identity_l2446_244685

theorem trig_identity (α : Real) (h : Real.tan α = 2) :
  Real.sin α ^ 2 + 2 * Real.sin α * Real.cos α + 3 * Real.cos α ^ 2 = 11 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2446_244685


namespace NUMINAMATH_CALUDE_odd_periodic_function_value_l2446_244665

-- Define the properties of the function f
def is_odd_and_periodic (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 2) = -f x)

-- State the theorem
theorem odd_periodic_function_value (f : ℝ → ℝ) (h : is_odd_and_periodic f) : 
  f 2008 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_periodic_function_value_l2446_244665


namespace NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l2446_244633

theorem quadratic_is_square_of_binomial :
  ∃ (r s : ℚ), (81/16 : ℚ) * x^2 + 18 * x + 16 = (r * x + s)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l2446_244633


namespace NUMINAMATH_CALUDE_profit_calculation_l2446_244634

def number_of_bags : ℕ := 100
def selling_price : ℚ := 10
def buying_price : ℚ := 7

theorem profit_calculation :
  (number_of_bags : ℚ) * (selling_price - buying_price) = 300 := by sorry

end NUMINAMATH_CALUDE_profit_calculation_l2446_244634


namespace NUMINAMATH_CALUDE_sum_20_225_base7_l2446_244631

/-- Represents a number in base 7 --/
def Base7 : Type := ℕ

/-- Converts a natural number to its base 7 representation --/
def toBase7 (n : ℕ) : Base7 := sorry

/-- Adds two numbers in base 7 --/
def addBase7 (a b : Base7) : Base7 := sorry

/-- Theorem: The sum of 20₇ and 225₇ in base 7 is 245₇ --/
theorem sum_20_225_base7 :
  addBase7 (toBase7 20) (toBase7 225) = toBase7 245 := by sorry

end NUMINAMATH_CALUDE_sum_20_225_base7_l2446_244631


namespace NUMINAMATH_CALUDE_robie_second_purchase_l2446_244641

/-- The number of bags of chocolates Robie bought the second time -/
def second_purchase (initial : ℕ) (given_away : ℕ) (final : ℕ) : ℕ :=
  final - (initial - given_away)

/-- Theorem: Robie bought 3 bags of chocolates the second time -/
theorem robie_second_purchase :
  second_purchase 3 2 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_robie_second_purchase_l2446_244641


namespace NUMINAMATH_CALUDE_geometric_sequence_a7_l2446_244647

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a7 (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 3 = 4 →
  a 7 - 2 * a 5 = 32 →
  a 7 = 64 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a7_l2446_244647


namespace NUMINAMATH_CALUDE_triangle_area_and_minimum_ratio_l2446_244674

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition 2sin²A + sin²B = sin²C -/
def satisfiesCondition (t : Triangle) : Prop :=
  2 * (Real.sin t.A)^2 + (Real.sin t.B)^2 = (Real.sin t.C)^2

theorem triangle_area_and_minimum_ratio (t : Triangle) 
  (h1 : satisfiesCondition t) 
  (h2 : t.b = 2 * t.a) 
  (h3 : t.b = 4) :
  -- Part 1: Area of triangle ABC is √15
  (1/2 * t.a * t.b * Real.sin t.C = Real.sqrt 15) ∧
  -- Part 2: Minimum value of c²/(ab) is 2√2, and c/a = 2 at this minimum
  (∀ t' : Triangle, satisfiesCondition t' → 
    t'.c^2 / (t'.a * t'.b) ≥ 2 * Real.sqrt 2) ∧
  (∃ t' : Triangle, satisfiesCondition t' ∧ 
    t'.c^2 / (t'.a * t'.b) = 2 * Real.sqrt 2 ∧ t'.c / t'.a = 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_and_minimum_ratio_l2446_244674


namespace NUMINAMATH_CALUDE_committee_formation_count_l2446_244630

/-- Represents a department in the division of science -/
inductive Department
| physics
| chemistry
| biology
| mathematics

/-- The number of departments in the division -/
def num_departments : Nat := 4

/-- The number of male professors in each department -/
def male_professors_per_dept : Nat := 3

/-- The number of female professors in each department -/
def female_professors_per_dept : Nat := 3

/-- The total number of professors in the committee -/
def committee_size : Nat := 8

/-- The number of male professors in the committee -/
def male_committee_members : Nat := 4

/-- The number of female professors in the committee -/
def female_committee_members : Nat := 4

/-- The number of departments contributing exactly two professors -/
def depts_with_two_profs : Nat := 2

/-- The number of departments contributing one male and one female professor -/
def depts_with_one_each : Nat := 2

/-- The number of ways to form the committee under the given conditions -/
def committee_formation_ways : Nat := 48114

theorem committee_formation_count :
  (num_departments = 4) →
  (male_professors_per_dept = 3) →
  (female_professors_per_dept = 3) →
  (committee_size = 8) →
  (male_committee_members = 4) →
  (female_committee_members = 4) →
  (depts_with_two_profs = 2) →
  (depts_with_one_each = 2) →
  (committee_formation_ways = 48114) := by
  sorry


end NUMINAMATH_CALUDE_committee_formation_count_l2446_244630


namespace NUMINAMATH_CALUDE_samson_activity_solution_l2446_244615

/-- Represents the utility function for Samson's activities -/
def utility (math : ℝ) (frisbee : ℝ) : ℝ := math * frisbee

/-- Represents the total hours spent on activities -/
def totalHours (math : ℝ) (frisbee : ℝ) : ℝ := math + frisbee

theorem samson_activity_solution :
  ∃ (t : ℝ),
    (utility (10 - t) t = utility (t + 5) (4 - t)) ∧
    (totalHours (10 - t) t ≥ 8) ∧
    (totalHours (t + 5) (4 - t) ≥ 8) ∧
    (t ≥ 0) ∧
    (∀ (s : ℝ),
      (utility (10 - s) s = utility (s + 5) (4 - s)) ∧
      (totalHours (10 - s) s ≥ 8) ∧
      (totalHours (s + 5) (4 - s) ≥ 8) ∧
      (s ≥ 0) →
      s = t) ∧
    t = 0 :=
by sorry

end NUMINAMATH_CALUDE_samson_activity_solution_l2446_244615


namespace NUMINAMATH_CALUDE_inequality_solution_l2446_244613

theorem inequality_solution (x : ℝ) : 
  (x + 1) / 2 > 1 - (2 * x - 1) / 3 ↔ x > 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2446_244613


namespace NUMINAMATH_CALUDE_dirt_bike_cost_l2446_244612

/-- Proves that the cost of each dirt bike is $150 given the problem conditions -/
theorem dirt_bike_cost :
  ∀ (dirt_bike_cost : ℕ),
  (3 * dirt_bike_cost + 4 * 300 + 7 * 25 = 1825) →
  dirt_bike_cost = 150 :=
by
  sorry

#check dirt_bike_cost

end NUMINAMATH_CALUDE_dirt_bike_cost_l2446_244612


namespace NUMINAMATH_CALUDE_simplify_expression_l2446_244697

/-- For all real numbers z, (2-3z) - (3+4z) = -1-7z -/
theorem simplify_expression (z : ℝ) : (2 - 3*z) - (3 + 4*z) = -1 - 7*z := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2446_244697


namespace NUMINAMATH_CALUDE_belts_count_l2446_244607

/-- The number of ties in the store -/
def ties : ℕ := 34

/-- The number of black shirts in the store -/
def black_shirts : ℕ := 63

/-- The number of white shirts in the store -/
def white_shirts : ℕ := 42

/-- The number of jeans in the store -/
def jeans : ℕ := (2 * (black_shirts + white_shirts)) / 3

/-- The number of scarves in the store -/
def scarves (belts : ℕ) : ℕ := (ties + belts) / 2

/-- The relationship between jeans and scarves -/
def jeans_scarves_relation (belts : ℕ) : Prop :=
  jeans = scarves belts + 33

theorem belts_count : ∃ (belts : ℕ), jeans_scarves_relation belts ∧ belts = 40 :=
sorry

end NUMINAMATH_CALUDE_belts_count_l2446_244607


namespace NUMINAMATH_CALUDE_salem_poem_word_count_l2446_244606

/-- Represents a poem with a specific structure -/
structure Poem where
  stanzas : Nat
  lines_per_stanza : Nat
  words_per_line : Nat

/-- Calculates the total number of words in a poem -/
def total_words (p : Poem) : Nat :=
  p.stanzas * p.lines_per_stanza * p.words_per_line

/-- Theorem: A poem with 35 stanzas, 15 lines per stanza, and 12 words per line has 6300 words -/
theorem salem_poem_word_count :
  let p : Poem := { stanzas := 35, lines_per_stanza := 15, words_per_line := 12 }
  total_words p = 6300 := by
  sorry

#eval total_words { stanzas := 35, lines_per_stanza := 15, words_per_line := 12 }

end NUMINAMATH_CALUDE_salem_poem_word_count_l2446_244606


namespace NUMINAMATH_CALUDE_max_value_fraction_max_value_achievable_l2446_244622

theorem max_value_fraction (x y : ℝ) : 
  (2 * x + Real.sqrt 2 * y) / (2 * x^4 + 4 * y^4 + 9) ≤ 1/4 :=
by sorry

theorem max_value_achievable : 
  ∃ x y : ℝ, (2 * x + Real.sqrt 2 * y) / (2 * x^4 + 4 * y^4 + 9) = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_max_value_achievable_l2446_244622


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2446_244681

theorem absolute_value_inequality (x : ℝ) :
  |x^2 - 5| < 9 ↔ -Real.sqrt 14 < x ∧ x < Real.sqrt 14 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2446_244681


namespace NUMINAMATH_CALUDE_cylinder_radius_determination_l2446_244651

theorem cylinder_radius_determination (z : ℝ) : 
  let original_height : ℝ := 3
  let volume_increase (r : ℝ) : ℝ → ℝ := λ h => π * (r^2 * h - r^2 * original_height)
  ∀ r : ℝ, 
    (volume_increase r (original_height + 4) = z ∧ 
     volume_increase (r + 4) original_height = z) → 
    r = 8 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_radius_determination_l2446_244651


namespace NUMINAMATH_CALUDE_rectangle_square_area_ratio_l2446_244698

/-- Given a square S and a rectangle R, where the longer side of R is 20% more than
    the side of S, the shorter side of R is 20% less than the side of S, and the
    diagonal of R is 10% longer than the diagonal of S, prove that the ratio of
    the area of R to the area of S is 24/25. -/
theorem rectangle_square_area_ratio 
  (S : Real) -- Side length of square S
  (R_long : Real) -- Longer side of rectangle R
  (R_short : Real) -- Shorter side of rectangle R
  (R_diag : Real) -- Diagonal of rectangle R
  (h1 : R_long = 1.2 * S) -- Longer side of R is 20% more than side of S
  (h2 : R_short = 0.8 * S) -- Shorter side of R is 20% less than side of S
  (h3 : R_diag = 1.1 * S * Real.sqrt 2) -- Diagonal of R is 10% longer than diagonal of S
  : (R_long * R_short) / (S * S) = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_square_area_ratio_l2446_244698


namespace NUMINAMATH_CALUDE_substitution_remainder_l2446_244695

/-- Represents the number of players on the roster. -/
def totalPlayers : ℕ := 15

/-- Represents the number of players in the starting lineup. -/
def startingLineup : ℕ := 10

/-- Represents the number of substitute players. -/
def substitutes : ℕ := 5

/-- Represents the maximum number of substitutions allowed. -/
def maxSubstitutions : ℕ := 2

/-- Calculates the number of ways to make substitutions given the number of substitutions. -/
def substitutionWays (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => startingLineup * substitutes
  | 2 => startingLineup * substitutes * (startingLineup - 1) * (substitutes - 1)
  | _ => 0

/-- Calculates the total number of possible substitution scenarios. -/
def totalScenarios : ℕ :=
  (List.range (maxSubstitutions + 1)).map substitutionWays |>.sum

/-- The main theorem stating that the remainder of totalScenarios divided by 500 is 351. -/
theorem substitution_remainder :
  totalScenarios % 500 = 351 := by
  sorry

end NUMINAMATH_CALUDE_substitution_remainder_l2446_244695


namespace NUMINAMATH_CALUDE_even_quadratic_function_l2446_244626

/-- A function f is even on an interval if f(-x) = f(x) for all x in the interval -/
def IsEvenOn (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x, x ∈ [min (-b) (-a), max a b] → f (-x) = f x

theorem even_quadratic_function (a : ℝ) :
  let f := fun x ↦ a * x^2 + 1
  IsEvenOn f (3 - a) 5 → a = 8 := by
sorry

end NUMINAMATH_CALUDE_even_quadratic_function_l2446_244626


namespace NUMINAMATH_CALUDE_square_sum_identity_l2446_244662

theorem square_sum_identity (a b : ℝ) : a^2 + b^2 = (a + b)^2 + (-2 * a * b) := by
  sorry

end NUMINAMATH_CALUDE_square_sum_identity_l2446_244662


namespace NUMINAMATH_CALUDE_rosie_lou_speed_ratio_l2446_244683

/-- The ratio of Rosie's speed to Lou's speed on a circular track -/
theorem rosie_lou_speed_ratio :
  let track_length : ℚ := 1/4  -- Length of the track in miles
  let lou_distance : ℚ := 3    -- Lou's total distance in miles
  let rosie_laps : ℕ := 24     -- Number of laps Rosie completes
  let rosie_distance : ℚ := rosie_laps * track_length  -- Rosie's total distance in miles
  ∀ (lou_speed rosie_speed : ℚ),
    lou_speed > 0 →  -- Lou's speed is positive
    rosie_speed > 0 →  -- Rosie's speed is positive
    lou_speed * lou_distance = rosie_speed * rosie_distance →  -- They run for the same duration
    rosie_speed / lou_speed = 2/1 :=
by sorry

end NUMINAMATH_CALUDE_rosie_lou_speed_ratio_l2446_244683


namespace NUMINAMATH_CALUDE_exists_constant_function_l2446_244652

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 5

-- State the theorem
theorem exists_constant_function (x : ℝ) : ∃ k : ℝ, 2 * f 3 - 10 = f k ∧ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_exists_constant_function_l2446_244652


namespace NUMINAMATH_CALUDE_divisors_of_180_l2446_244619

def sum_of_divisors (n : ℕ) : ℕ := sorry

def largest_prime_factor (n : ℕ) : ℕ := sorry

def count_divisors (n : ℕ) : ℕ := sorry

theorem divisors_of_180 :
  (largest_prime_factor (sum_of_divisors 180) = 13) ∧
  (count_divisors 180 = 18) := by sorry

end NUMINAMATH_CALUDE_divisors_of_180_l2446_244619


namespace NUMINAMATH_CALUDE_max_gcd_of_eight_numbers_sum_595_l2446_244639

/-- The maximum possible GCD of eight natural numbers summing to 595 -/
theorem max_gcd_of_eight_numbers_sum_595 :
  ∃ (a b c d e f g h : ℕ),
    a + b + c + d + e + f + g + h = 595 ∧
    ∀ (k : ℕ),
      k ∣ a ∧ k ∣ b ∧ k ∣ c ∧ k ∣ d ∧ k ∣ e ∧ k ∣ f ∧ k ∣ g ∧ k ∣ h →
      k ≤ 35 :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_of_eight_numbers_sum_595_l2446_244639


namespace NUMINAMATH_CALUDE_simplify_expression_l2446_244688

theorem simplify_expression (x : ℝ) (h : x ≠ 0) :
  (x^2)⁻¹ - 2 = (1 - 2*x^2) / x^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2446_244688


namespace NUMINAMATH_CALUDE_student_age_problem_l2446_244643

theorem student_age_problem (total_students : Nat) (avg_age : Nat) 
  (group1_count : Nat) (group1_avg : Nat)
  (group2_count : Nat) (group2_avg : Nat)
  (group3_count : Nat) (group3_avg : Nat) :
  total_students = 25 →
  avg_age = 16 →
  group1_count = 7 →
  group1_avg = 15 →
  group2_count = 12 →
  group2_avg = 16 →
  group3_count = 5 →
  group3_avg = 18 →
  group1_count + group2_count + group3_count = total_students - 1 →
  (total_students * avg_age) - (group1_count * group1_avg + group2_count * group2_avg + group3_count * group3_avg) = 13 := by
  sorry

end NUMINAMATH_CALUDE_student_age_problem_l2446_244643


namespace NUMINAMATH_CALUDE_zoo_trip_bus_capacity_l2446_244602

/-- Given a school trip to the zoo with the following conditions:
  * total_students: The total number of students on the trip
  * num_buses: The number of buses used for transportation
  * students_in_cars: The number of students who traveled in cars
  * students_per_bus: The number of students in each bus

  This theorem proves that when total_students = 396, num_buses = 7, and students_in_cars = 4,
  the number of students in each bus (students_per_bus) is equal to 56. -/
theorem zoo_trip_bus_capacity 
  (total_students : ℕ) 
  (num_buses : ℕ) 
  (students_in_cars : ℕ) 
  (students_per_bus : ℕ) 
  (h1 : total_students = 396) 
  (h2 : num_buses = 7) 
  (h3 : students_in_cars = 4) 
  (h4 : students_per_bus * num_buses + students_in_cars = total_students) :
  students_per_bus = 56 := by
  sorry

end NUMINAMATH_CALUDE_zoo_trip_bus_capacity_l2446_244602
