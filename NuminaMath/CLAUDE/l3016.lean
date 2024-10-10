import Mathlib

namespace greatest_value_is_product_of_zeros_l3016_301658

def Q (x : ℝ) : ℝ := x^4 + 2*x^3 - x^2 - 4*x + 4

theorem greatest_value_is_product_of_zeros :
  let product_of_zeros : ℝ := 4
  let q_of_one : ℝ := Q 1
  let sum_of_coefficients : ℝ := 1 + 2 - 1 - 4 + 4
  let sum_of_real_zeros : ℝ := 0  -- Assumption based on estimated real zeros
  product_of_zeros > q_of_one ∧
  product_of_zeros > sum_of_coefficients ∧
  product_of_zeros > sum_of_real_zeros :=
by sorry

end greatest_value_is_product_of_zeros_l3016_301658


namespace certain_number_proof_l3016_301655

theorem certain_number_proof : ∃ x : ℝ, 0.45 * 60 = 0.35 * x + 13 ∧ x = 40 := by
  sorry

end certain_number_proof_l3016_301655


namespace parabola_properties_l3016_301697

-- Define the parabola C
def C (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the line on which the focus lies
def focus_line (x y : ℝ) : Prop := x + 2*y - 4 = 0

-- Theorem statement
theorem parabola_properties :
  ∃ (p : ℝ), 
    (∃ (x y : ℝ), C p x y ∧ focus_line x y) →
    (p = 8 ∧ 
     ∀ (x : ℝ), (x = -4) ↔ (∃ (y : ℝ), C p x y ∧ ∀ (x' y' : ℝ), C p x' y' → (x - x')^2 + (y - y')^2 ≥ (x + 4)^2)) :=
sorry

end parabola_properties_l3016_301697


namespace sphere_volume_fraction_l3016_301698

theorem sphere_volume_fraction (R : ℝ) (h : R > 0) :
  let sphereVolume := (4 / 3) * Real.pi * R^3
  let capVolume := Real.pi * R^3 * ((2 / 3) - (5 * Real.sqrt 2) / 12)
  capVolume / sphereVolume = (8 - 5 * Real.sqrt 2) / 16 := by
  sorry

end sphere_volume_fraction_l3016_301698


namespace particle_movement_l3016_301686

/-- Represents a particle in a 2D grid -/
structure Particle where
  x : ℚ
  y : ℚ

/-- Represents the probabilities of movement for Particle A -/
structure ProbA where
  left : ℚ
  right : ℚ
  up : ℚ
  down : ℚ

/-- Represents the probability of movement for Particle B -/
def ProbB : ℚ → Prop := λ y ↦ ∀ (direction : Fin 4), y = 1/4

/-- The theorem statement -/
theorem particle_movement 
  (A : Particle) 
  (B : Particle) 
  (probA : ProbA) 
  (probB : ℚ → Prop) :
  A.x = 0 ∧ A.y = 0 ∧
  B.x = 1 ∧ B.y = 1 ∧
  probA.left = 1/4 ∧ probA.right = 1/4 ∧ probA.up = 1/3 ∧
  ProbB probA.down ∧
  (∃ (x : ℚ), probA.down = x ∧ x + 1/4 + 1/4 + 1/3 = 1) →
  probA.down = 1/6 ∧
  ProbB (1/4) ∧
  (∃ (t : ℕ), t = 3 ∧ 
    (∀ (t' : ℕ), (∃ (A' B' : Particle), A'.x = 2 ∧ A'.y = 1 ∧ B'.x = 2 ∧ B'.y = 1) → t' ≥ t)) ∧
  (9 : ℚ)/1024 = 
    (3 * (1/4)^2 * 1/3) * -- Probability for A
    (1/4 * 3 * (1/4)^2)   -- Probability for B
  := by sorry

end particle_movement_l3016_301686


namespace solution_value_l3016_301617

/-- The function F as defined in the problem -/
def F (a b c : ℝ) : ℝ := a * b^2 + c

/-- Theorem stating that -1/8 is the solution to the equation F(a,3,8) = F(a,5,10) -/
theorem solution_value :
  ∃ a : ℝ, F a 3 8 = F a 5 10 ∧ a = -1/8 := by
  sorry

end solution_value_l3016_301617


namespace graduating_class_boys_count_l3016_301614

theorem graduating_class_boys_count (total : ℕ) (diff : ℕ) (boys : ℕ) : 
  total = 466 → diff = 212 → boys + (boys + diff) = total → boys = 127 := by
  sorry

end graduating_class_boys_count_l3016_301614


namespace parking_arrangement_count_l3016_301667

/-- The number of parking spaces -/
def n : ℕ := 50

/-- The number of cars to be arranged -/
def k : ℕ := 2

/-- The number of ways to arrange k distinct cars in n parking spaces -/
def total_arrangements (n k : ℕ) : ℕ := n * (n - 1)

/-- The number of ways to arrange k distinct cars adjacently in n parking spaces -/
def adjacent_arrangements (n : ℕ) : ℕ := 2 * (n - 1)

/-- The number of ways to arrange k distinct cars in n parking spaces with at least one empty space between them -/
def valid_arrangements (n k : ℕ) : ℕ := total_arrangements n k - adjacent_arrangements n

theorem parking_arrangement_count :
  valid_arrangements n k = 2352 :=
by sorry

end parking_arrangement_count_l3016_301667


namespace geometric_series_sum_l3016_301633

theorem geometric_series_sum (c d : ℝ) (h : ∑' n, c / d^n = 3) :
  ∑' n, c / (c + 2*d)^n = (3*d - 3) / (5*d - 4) := by
  sorry

end geometric_series_sum_l3016_301633


namespace pure_imaginary_m_equals_four_l3016_301691

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := (2*m - 8) + (m - 2)*Complex.I

-- Define what it means for a complex number to be pure imaginary
def isPureImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- Theorem statement
theorem pure_imaginary_m_equals_four :
  ∃ m : ℝ, isPureImaginary (z m) → m = 4 := by
sorry

end pure_imaginary_m_equals_four_l3016_301691


namespace pages_to_read_tomorrow_l3016_301662

/-- Given a book and Julie's reading progress, calculate the number of pages to read tomorrow --/
theorem pages_to_read_tomorrow (total_pages yesterday_pages : ℕ) : 
  total_pages = 120 →
  yesterday_pages = 12 →
  (total_pages - (yesterday_pages + 2 * yesterday_pages)) / 2 = 42 := by
sorry

end pages_to_read_tomorrow_l3016_301662


namespace tens_digit_of_6_pow_2047_l3016_301629

/-- The cycle of the last two digits of powers of 6 -/
def last_two_digits_cycle : List ℕ := [16, 96, 76, 56]

/-- The length of the cycle -/
def cycle_length : ℕ := 4

theorem tens_digit_of_6_pow_2047 (h : last_two_digits_cycle = [16, 96, 76, 56]) :
  (6^2047 / 10) % 10 = 7 := by
  sorry

end tens_digit_of_6_pow_2047_l3016_301629


namespace hat_price_after_discounts_l3016_301649

def initial_price : ℝ := 15
def first_discount : ℝ := 0.25
def second_discount : ℝ := 0.50

theorem hat_price_after_discounts :
  let price_after_first_discount := initial_price * (1 - first_discount)
  let final_price := price_after_first_discount * (1 - second_discount)
  final_price = 5.625 := by sorry

end hat_price_after_discounts_l3016_301649


namespace a_less_than_two_thirds_l3016_301666

-- Define a decreasing function
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem a_less_than_two_thirds
  (f : ℝ → ℝ) (a : ℝ)
  (h_decreasing : DecreasingFunction f)
  (h_inequality : f (1 - a) < f (2 * a - 1)) :
  a < 2 / 3 := by
  sorry

end a_less_than_two_thirds_l3016_301666


namespace sole_mart_meals_l3016_301694

theorem sole_mart_meals (initial_meals : ℕ) (given_away : ℕ) (left : ℕ) 
  (h1 : initial_meals = 113)
  (h2 : given_away = 85)
  (h3 : left = 78) :
  initial_meals + (given_away + left) - initial_meals = 50 :=
by
  sorry

end sole_mart_meals_l3016_301694


namespace shaded_area_in_square_l3016_301682

/-- Given a square with side length a, the area bounded by a semicircle on one side
    and two quarter-circle arcs on the adjacent sides is equal to a²/2 -/
theorem shaded_area_in_square (a : ℝ) (h : a > 0) :
  (π * a^2 / 8) + (π * a^2 / 8) = a^2 / 2 := by
  sorry

#check shaded_area_in_square

end shaded_area_in_square_l3016_301682


namespace min_value_of_f_l3016_301601

/-- The quadratic function f(x) = (x-1)^2 - 3 -/
def f (x : ℝ) : ℝ := (x - 1)^2 - 3

/-- The minimum value of f(x) is -3 -/
theorem min_value_of_f :
  ∃ (m : ℝ), m = -3 ∧ ∀ (x : ℝ), f x ≥ m :=
sorry

end min_value_of_f_l3016_301601


namespace quadratic_roots_l3016_301612

theorem quadratic_roots (a b c : ℝ) (ha : a ≠ 0) (h1 : a + b + c = 0) (h2 : a - b + c = 0) :
  ∃ (x y : ℝ), x = 1 ∧ y = -1 ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 :=
by sorry

end quadratic_roots_l3016_301612


namespace defective_units_shipped_l3016_301622

theorem defective_units_shipped (total_units : ℝ) (defective_rate : ℝ) (shipped_rate : ℝ) :
  defective_rate = 0.05 →
  shipped_rate = 0.04 →
  (defective_rate * shipped_rate * 100) = 0.2 := by
  sorry

end defective_units_shipped_l3016_301622


namespace function_existence_and_properties_l3016_301609

/-- A function satisfying the given equation -/
def SatisfiesEquation (f : ℤ → ℤ → ℤ) : Prop :=
  ∀ n m : ℤ, f n m = (1/4) * (f (n-1) m + f (n+1) m + f n (m-1) + f n (m+1))

/-- The function is non-constant -/
def IsNonConstant (f : ℤ → ℤ → ℤ) : Prop :=
  ∃ n₁ m₁ n₂ m₂ : ℤ, f n₁ m₁ ≠ f n₂ m₂

/-- The function takes values both greater and less than any integer -/
def SpansAllIntegers (f : ℤ → ℤ → ℤ) : Prop :=
  ∀ k : ℤ, (∃ n₁ m₁ : ℤ, f n₁ m₁ > k) ∧ (∃ n₂ m₂ : ℤ, f n₂ m₂ < k)

/-- The main theorem -/
theorem function_existence_and_properties :
  ∃ f : ℤ → ℤ → ℤ, SatisfiesEquation f ∧ IsNonConstant f ∧ SpansAllIntegers f := by
  sorry

end function_existence_and_properties_l3016_301609


namespace edward_lost_lives_l3016_301672

theorem edward_lost_lives (initial_lives : ℕ) (remaining_lives : ℕ) (lost_lives : ℕ) : 
  initial_lives = 15 → remaining_lives = 7 → lost_lives = initial_lives - remaining_lives → lost_lives = 8 := by
  sorry

end edward_lost_lives_l3016_301672


namespace simplify_expression_l3016_301683

theorem simplify_expression (x : ℝ) : x^2 * x^4 + x * x^2 * x^3 = 2 * x^6 := by
  sorry

end simplify_expression_l3016_301683


namespace maximum_garden_area_l3016_301643

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
def gardenArea (d : GardenDimensions) : ℝ := d.length * d.width

/-- Calculates the fence length required for three sides of a rectangular garden -/
def fenceLength (d : GardenDimensions) : ℝ := d.length + 2 * d.width

/-- The total available fencing -/
def totalFence : ℝ := 400

theorem maximum_garden_area :
  ∃ (d : GardenDimensions),
    fenceLength d = totalFence ∧
    ∀ (d' : GardenDimensions), fenceLength d' = totalFence → gardenArea d' ≤ gardenArea d ∧
    gardenArea d = 20000 := by
  sorry

end maximum_garden_area_l3016_301643


namespace cubic_equation_real_root_l3016_301618

/-- The cubic equation 5z^3 - 4iz^2 + z - k = 0 has at least one real root for all positive real k -/
theorem cubic_equation_real_root (k : ℝ) (hk : k > 0) : 
  ∃ (z : ℂ), z.im = 0 ∧ 5 * z^3 - 4 * Complex.I * z^2 + z - k = 0 := by sorry

end cubic_equation_real_root_l3016_301618


namespace survey_min_overlap_l3016_301627

/-- Given a survey of 120 people, where 95 like Mozart, 80 like Bach, and 75 like Beethoven,
    the minimum number of people who like both Mozart and Bach but not Beethoven is 45. -/
theorem survey_min_overlap (total : ℕ) (mozart : ℕ) (bach : ℕ) (beethoven : ℕ)
  (h_total : total = 120)
  (h_mozart : mozart = 95)
  (h_bach : bach = 80)
  (h_beethoven : beethoven = 75)
  (h_mozart_le : mozart ≤ total)
  (h_bach_le : bach ≤ total)
  (h_beethoven_le : beethoven ≤ total) :
  ∃ (overlap : ℕ), overlap ≥ 45 ∧
    overlap ≤ mozart ∧
    overlap ≤ bach ∧
    overlap ≤ total - beethoven ∧
    ∀ (x : ℕ), x < overlap →
      ¬(x ≤ mozart ∧ x ≤ bach ∧ x ≤ total - beethoven) :=
by sorry

end survey_min_overlap_l3016_301627


namespace population_growth_rate_l3016_301656

/-- The time it takes for one person to be added to the population, given the rate of population increase. -/
def time_per_person (persons_per_hour : ℕ) : ℚ :=
  (60 * 60) / persons_per_hour

/-- Theorem stating that the time it takes for one person to be added to the population is 15 seconds, 
    given that the population increases by 240 persons in 60 minutes. -/
theorem population_growth_rate : time_per_person 240 = 15 := by
  sorry

end population_growth_rate_l3016_301656


namespace estimate_fish_population_l3016_301670

/-- Estimates the total number of fish in a pond using the mark-recapture method. -/
theorem estimate_fish_population (tagged_fish : ℕ) (second_sample : ℕ) (tagged_in_sample : ℕ) :
  tagged_fish = 100 →
  second_sample = 200 →
  tagged_in_sample = 10 →
  (tagged_fish * second_sample) / tagged_in_sample = 2000 :=
by
  sorry

#check estimate_fish_population

end estimate_fish_population_l3016_301670


namespace license_plate_count_l3016_301673

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The number of digits in a license plate -/
def digits_in_plate : ℕ := 5

/-- The number of letters in a license plate -/
def letters_in_plate : ℕ := 2

/-- The number of positions where the letter block can be placed -/
def letter_block_positions : ℕ := digits_in_plate + 1

/-- The number of distinct license plates possible -/
def num_license_plates : ℕ := 
  letter_block_positions * num_digits^digits_in_plate * num_letters^letters_in_plate

theorem license_plate_count : num_license_plates = 40560000 := by
  sorry

end license_plate_count_l3016_301673


namespace double_price_profit_percentage_l3016_301619

theorem double_price_profit_percentage (cost : ℝ) (initial_profit_percentage : ℝ) 
  (initial_selling_price : ℝ) (new_selling_price : ℝ) (new_profit_percentage : ℝ) :
  initial_profit_percentage = 20 →
  initial_selling_price = cost * (1 + initial_profit_percentage / 100) →
  new_selling_price = 2 * initial_selling_price →
  new_profit_percentage = ((new_selling_price - cost) / cost) * 100 →
  new_profit_percentage = 140 :=
by sorry

end double_price_profit_percentage_l3016_301619


namespace probability_of_correct_distribution_l3016_301616

/-- Represents the types of rolls -/
inductive RollType
  | Nut
  | Cheese
  | Fruit
  | Chocolate

/-- Represents a guest's set of rolls -/
def GuestRolls := Finset RollType

/-- The number of guests -/
def num_guests : Nat := 3

/-- The number of roll types -/
def num_roll_types : Nat := 4

/-- The total number of rolls -/
def total_rolls : Nat := num_guests * num_roll_types

/-- A function to calculate the probability of a specific distribution of rolls -/
noncomputable def probability_of_distribution (distribution : Finset GuestRolls) : ℚ := sorry

/-- The correct distribution where each guest has one of each roll type -/
def correct_distribution : Finset GuestRolls := sorry

/-- Theorem stating that the probability of the correct distribution is 24/1925 -/
theorem probability_of_correct_distribution :
  probability_of_distribution correct_distribution = 24 / 1925 := by sorry

end probability_of_correct_distribution_l3016_301616


namespace arcsin_equation_solutions_l3016_301687

theorem arcsin_equation_solutions :
  let f (x : ℝ) := Real.arcsin (2 * x / Real.sqrt 15) + Real.arcsin (3 * x / Real.sqrt 15) = Real.arcsin (4 * x / Real.sqrt 15)
  let valid (x : ℝ) := abs (2 * x / Real.sqrt 15) ≤ 1 ∧ abs (3 * x / Real.sqrt 15) ≤ 1 ∧ abs (4 * x / Real.sqrt 15) ≤ 1
  ∀ x : ℝ, valid x → (f x ↔ x = 0 ∨ x = 15 / 16 ∨ x = -15 / 16) :=
by sorry

end arcsin_equation_solutions_l3016_301687


namespace jonahs_calorie_burn_l3016_301693

/-- Calculates the difference in calories burned between two running durations -/
def calorie_difference (rate : ℕ) (duration1 duration2 : ℕ) : ℕ :=
  rate * duration2 - rate * duration1

/-- The problem statement -/
theorem jonahs_calorie_burn :
  let rate : ℕ := 30
  let short_duration : ℕ := 2
  let long_duration : ℕ := 5
  calorie_difference rate short_duration long_duration = 90 := by
  sorry

end jonahs_calorie_burn_l3016_301693


namespace pages_read_today_l3016_301647

theorem pages_read_today (pages_yesterday pages_total : ℕ) 
  (h1 : pages_yesterday = 21)
  (h2 : pages_total = 38) :
  pages_total - pages_yesterday = 17 := by
sorry

end pages_read_today_l3016_301647


namespace green_upgrade_area_l3016_301625

/-- Proves that the actual average annual area of green upgrade is 90 million square meters --/
theorem green_upgrade_area (total_area : ℝ) (planned_years original_plan actual_plan : ℝ) :
  total_area = 180 →
  actual_plan = 2 * original_plan →
  planned_years - (total_area / actual_plan) = 2 →
  actual_plan = 90 := by
  sorry

end green_upgrade_area_l3016_301625


namespace parallel_vectors_x_value_l3016_301642

/-- Two vectors in ℝ² are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (-2, 4)
  let b : ℝ × ℝ := (x, -2)
  are_parallel a b → x = 1 := by
  sorry

end parallel_vectors_x_value_l3016_301642


namespace pure_imaginary_modulus_l3016_301611

theorem pure_imaginary_modulus (a : ℝ) : 
  let z : ℂ := (a + Complex.I) / (1 - 2 * Complex.I)
  (z.re = 0 ∧ z.im ≠ 0) → Complex.abs (a + 2 * Complex.I) = 2 * Real.sqrt 2 := by
  sorry

end pure_imaginary_modulus_l3016_301611


namespace complex_fraction_simplification_l3016_301653

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (3 - 2 * i) / (1 + 4 * i) = -5 / 17 - 14 / 17 * i :=
by
  sorry

end complex_fraction_simplification_l3016_301653


namespace positive_sum_inequality_l3016_301688

theorem positive_sum_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := by
  sorry

end positive_sum_inequality_l3016_301688


namespace problem_solution_l3016_301637

/-- The function f(x) = x^2 - 1 --/
def f (x : ℝ) : ℝ := x^2 - 1

/-- The function g(x) = a|x-1| --/
def g (a x : ℝ) : ℝ := a * |x - 1|

/-- The function h(x) = |f(x)| + g(x) --/
def h (a x : ℝ) : ℝ := |f x| + g a x

theorem problem_solution (a : ℝ) :
  (∀ x : ℝ, f x ≥ g a x) →
  (a ≤ -2 ∧
   (∀ x ∈ Set.Icc 0 1,
      (a ≥ -3 → h a x ≤ a + 3) ∧
      (a < -3 → h a x ≤ 0))) :=
by sorry

end problem_solution_l3016_301637


namespace geometric_sequence_product_constant_geometric_sequence_product_l3016_301636

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The product of terms equidistant from the beginning and end is constant -/
theorem geometric_sequence_product_constant {a : ℕ → ℝ} (h : geometric_sequence a) :
  ∀ n k : ℕ, a n * a (k + 1 - n) = a 1 * a k :=
sorry

/-- Main theorem: If a₃a₄ = 2 in a geometric sequence, then a₁a₂a₃a₄a₅a₆ = 8 -/
theorem geometric_sequence_product (a : ℕ → ℝ) (h : geometric_sequence a) 
  (h2 : a 3 * a 4 = 2) : a 1 * a 2 * a 3 * a 4 * a 5 * a 6 = 8 :=
sorry

end geometric_sequence_product_constant_geometric_sequence_product_l3016_301636


namespace ellipse_symmetry_l3016_301677

-- Define the original ellipse
def original_ellipse (x y : ℝ) : Prop :=
  (x - 3)^2 / 9 + (y - 2)^2 / 4 = 1

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop :=
  x + y = 0

-- Define the reflection transformation
def reflect (x y : ℝ) : ℝ × ℝ :=
  (-y, -x)

-- Define the resulting ellipse C
def ellipse_c (x y : ℝ) : Prop :=
  (x + 2)^2 / 9 + (y + 3)^2 / 4 = 1

-- Theorem statement
theorem ellipse_symmetry :
  ∀ (x y : ℝ),
    original_ellipse x y →
    let (x', y') := reflect x y
    ellipse_c x' y' :=
by sorry

end ellipse_symmetry_l3016_301677


namespace tom_remaining_candy_l3016_301681

/-- The number of candy pieces Tom still has after giving some away to his brother -/
def remaining_candy_pieces : ℕ :=
  let initial_chocolate_boxes : ℕ := 14
  let initial_fruit_boxes : ℕ := 10
  let initial_caramel_boxes : ℕ := 8
  let given_chocolate_boxes : ℕ := 8
  let given_fruit_boxes : ℕ := 5
  let pieces_per_chocolate_box : ℕ := 3
  let pieces_per_fruit_box : ℕ := 4
  let pieces_per_caramel_box : ℕ := 5

  let initial_total_pieces : ℕ := 
    initial_chocolate_boxes * pieces_per_chocolate_box +
    initial_fruit_boxes * pieces_per_fruit_box +
    initial_caramel_boxes * pieces_per_caramel_box

  let given_away_pieces : ℕ := 
    given_chocolate_boxes * pieces_per_chocolate_box +
    given_fruit_boxes * pieces_per_fruit_box

  initial_total_pieces - given_away_pieces

theorem tom_remaining_candy : remaining_candy_pieces = 78 := by
  sorry

end tom_remaining_candy_l3016_301681


namespace blue_eyed_percentage_l3016_301626

/-- Represents the number of kittens with blue eyes for a cat -/
def blue_eyed_kittens (cat : Nat) : Nat :=
  if cat = 1 then 3 else 4

/-- Represents the number of kittens with brown eyes for a cat -/
def brown_eyed_kittens (cat : Nat) : Nat :=
  if cat = 1 then 7 else 6

/-- The total number of kittens -/
def total_kittens : Nat :=
  (blue_eyed_kittens 1 + brown_eyed_kittens 1) + (blue_eyed_kittens 2 + brown_eyed_kittens 2)

/-- The total number of blue-eyed kittens -/
def total_blue_eyed : Nat :=
  blue_eyed_kittens 1 + blue_eyed_kittens 2

/-- Theorem stating that 35% of all kittens have blue eyes -/
theorem blue_eyed_percentage :
  (total_blue_eyed : ℚ) / (total_kittens : ℚ) * 100 = 35 := by
  sorry

end blue_eyed_percentage_l3016_301626


namespace new_consumption_per_soldier_l3016_301669

/-- Calculates the new daily consumption per soldier after additional soldiers join a fort, given the initial conditions and the number of new soldiers. -/
theorem new_consumption_per_soldier
  (initial_soldiers : ℕ)
  (initial_consumption : ℚ)
  (initial_duration : ℕ)
  (new_duration : ℕ)
  (new_soldiers : ℕ)
  (h_initial_soldiers : initial_soldiers = 1200)
  (h_initial_consumption : initial_consumption = 3)
  (h_initial_duration : initial_duration = 30)
  (h_new_duration : new_duration = 25)
  (h_new_soldiers : new_soldiers = 528) :
  let total_provisions := initial_soldiers * initial_consumption * initial_duration
  let total_soldiers := initial_soldiers + new_soldiers
  total_provisions / (total_soldiers * new_duration) = 2.5 := by
  sorry

end new_consumption_per_soldier_l3016_301669


namespace mutter_lagaan_payment_l3016_301663

theorem mutter_lagaan_payment (total_lagaan : ℝ) (mutter_percentage : ℝ) :
  total_lagaan = 344000 →
  mutter_percentage = 0.23255813953488372 →
  mutter_percentage / 100 * total_lagaan = 800 := by
sorry

end mutter_lagaan_payment_l3016_301663


namespace hyperbola_asymptotes_l3016_301664

def hyperbola (m n : ℝ) (x y : ℝ) : Prop :=
  x^2 / m - y^2 / n = 1

def tangent_line (m n : ℝ) (x y : ℝ) : Prop :=
  2 * m * x - n * y + 2 = 0

def asymptote (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x ∨ y = -k * x

theorem hyperbola_asymptotes (m n : ℝ) :
  (∀ x y, hyperbola m n x y) →
  (∀ x y, tangent_line m n x y) →
  (∀ x y, asymptote (Real.sqrt 2) x y) :=
sorry

end hyperbola_asymptotes_l3016_301664


namespace solution_difference_l3016_301699

theorem solution_difference (p q : ℝ) : 
  p ≠ q →
  (p - 5) * (p + 3) = 24 * p - 72 →
  (q - 5) * (q + 3) = 24 * q - 72 →
  p > q →
  p - q = 20 := by
  sorry

end solution_difference_l3016_301699


namespace min_value_expression_l3016_301684

theorem min_value_expression (x y : ℝ) : 
  ∃ (a b : ℝ), (x * y + 1)^2 + (x + y + 1)^2 ≥ 0 ∧ (a * b + 1)^2 + (a + b + 1)^2 = 0 := by
  sorry

end min_value_expression_l3016_301684


namespace opposite_numbers_and_unit_absolute_value_l3016_301661

theorem opposite_numbers_and_unit_absolute_value 
  (a b c : ℝ) 
  (h1 : a + b = 0) 
  (h2 : abs c = 1) : 
  a + b - c = 1 ∨ a + b - c = -1 := by
sorry

end opposite_numbers_and_unit_absolute_value_l3016_301661


namespace median_of_consecutive_integers_l3016_301615

theorem median_of_consecutive_integers (n : ℕ) (a : ℤ) (h : n > 0) :
  (∀ i, 0 ≤ i ∧ i < n → (a + i) + (a + (n - 1) - i) = 120) →
  (n % 2 = 1 → (a + (n - 1) / 2) = 60) ∧
  (n % 2 = 0 → (2 * a + n - 1) / 2 = 60) :=
sorry

end median_of_consecutive_integers_l3016_301615


namespace N2O3_molecular_weight_l3016_301639

/-- The atomic weight of nitrogen in g/mol -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of nitrogen atoms in N2O3 -/
def nitrogen_count : ℕ := 2

/-- The number of oxygen atoms in N2O3 -/
def oxygen_count : ℕ := 3

/-- The molecular weight of N2O3 in g/mol -/
def N2O3_weight : ℝ := nitrogen_count * nitrogen_weight + oxygen_count * oxygen_weight

theorem N2O3_molecular_weight : N2O3_weight = 76.02 := by
  sorry

end N2O3_molecular_weight_l3016_301639


namespace banquet_solution_l3016_301610

def banquet_problem (total_attendees : ℕ) (resident_price : ℚ) (non_resident_price : ℚ) (total_revenue : ℚ) : Prop :=
  ∃ (residents : ℕ),
    residents ≤ total_attendees ∧
    residents * resident_price + (total_attendees - residents) * non_resident_price = total_revenue

theorem banquet_solution :
  banquet_problem 586 (12.95 : ℚ) (17.95 : ℚ) (9423.70 : ℚ) →
  ∃ (residents : ℕ), residents = 220 ∧ banquet_problem 586 (12.95 : ℚ) (17.95 : ℚ) (9423.70 : ℚ) :=
by
  sorry

#check banquet_solution

end banquet_solution_l3016_301610


namespace simplify_nested_roots_l3016_301604

theorem simplify_nested_roots (a : ℝ) : 
  (((a^9)^(1/6))^(1/3))^4 * (((a^9)^(1/3))^(1/6))^4 = a^4 := by sorry

end simplify_nested_roots_l3016_301604


namespace experiment_sequences_l3016_301603

/-- The number of procedures in the experiment -/
def num_procedures : ℕ := 6

/-- The number of possible positions for procedure A -/
def a_positions : ℕ := 2

/-- The number of procedures excluding A -/
def remaining_procedures : ℕ := num_procedures - 1

/-- The number of arrangements of B and C -/
def bc_arrangements : ℕ := 2

theorem experiment_sequences :
  (a_positions * remaining_procedures.factorial * bc_arrangements) = 96 := by
  sorry

end experiment_sequences_l3016_301603


namespace number_of_boys_in_class_l3016_301648

/-- Given the conditions of a class weight calculation, prove the number of boys in the class -/
theorem number_of_boys_in_class 
  (incorrect_avg : ℝ) 
  (misread_weight : ℝ) 
  (correct_weight : ℝ) 
  (correct_avg : ℝ) 
  (h1 : incorrect_avg = 58.4)
  (h2 : misread_weight = 56)
  (h3 : correct_weight = 60)
  (h4 : correct_avg = 58.6) :
  ∃ n : ℕ, n * incorrect_avg + (correct_weight - misread_weight) = n * correct_avg ∧ n = 20 :=
by sorry

end number_of_boys_in_class_l3016_301648


namespace total_cost_calculation_l3016_301600

def shirt_cost : ℕ := 5
def hat_cost : ℕ := 4
def jeans_cost : ℕ := 10

def num_shirts : ℕ := 3
def num_hats : ℕ := 4
def num_jeans : ℕ := 2

theorem total_cost_calculation :
  shirt_cost * num_shirts + hat_cost * num_hats + jeans_cost * num_jeans = 51 := by
  sorry

end total_cost_calculation_l3016_301600


namespace rectangular_hyperbola_equation_l3016_301640

/-- A rectangular hyperbola with coordinate axes as its axes of symmetry
    passing through the point (2, √2) has the equation x² - y² = 2 -/
theorem rectangular_hyperbola_equation :
  ∀ (f : ℝ → ℝ → Prop),
    (∀ x y, f x y ↔ x^2 - y^2 = 2) →  -- Definition of the hyperbola equation
    (∀ x, f x 0 ↔ f 0 x) →            -- Symmetry about y = x
    (∀ x, f x 0 ↔ f (-x) 0) →         -- Symmetry about y-axis
    (∀ y, f 0 y ↔ f 0 (-y)) →         -- Symmetry about x-axis
    f 2 (Real.sqrt 2) →               -- Point (2, √2) lies on the hyperbola
    ∀ x y, f x y ↔ x^2 - y^2 = 2 :=
by sorry

end rectangular_hyperbola_equation_l3016_301640


namespace solve_triangle_problem_l3016_301678

/-- Represents a right-angled isosceles triangle --/
structure RightIsoscelesTriangle where
  side : ℝ
  area : ℝ
  area_eq : area = side^2 / 2

/-- The problem setup --/
def triangle_problem (k : ℝ) : Prop :=
  let t1 := RightIsoscelesTriangle.mk k (k^2 / 2) (by rfl)
  let t2 := RightIsoscelesTriangle.mk (k * Real.sqrt 2) (k^2) (by sorry)
  let t3 := RightIsoscelesTriangle.mk (2 * k) (2 * k^2) (by sorry)
  t1.area + t2.area + t3.area = 56

/-- The theorem to prove --/
theorem solve_triangle_problem : 
  ∃ k : ℝ, triangle_problem k ∧ k = 4 := by sorry

end solve_triangle_problem_l3016_301678


namespace imaginary_part_of_complex_power_l3016_301608

theorem imaginary_part_of_complex_power (i : ℂ) (h : i * i = -1) :
  let z := (1 + i) / (1 - i)
  Complex.im (z ^ 2023) = -Complex.im i :=
by
  sorry

end imaginary_part_of_complex_power_l3016_301608


namespace box_surface_area_l3016_301613

def sheet_length : ℕ := 25
def sheet_width : ℕ := 40
def corner_size : ℕ := 8

def surface_area : ℕ :=
  sheet_length * sheet_width - 4 * (corner_size * corner_size)

theorem box_surface_area :
  surface_area = 744 := by sorry

end box_surface_area_l3016_301613


namespace jennifer_sweets_sharing_l3016_301644

theorem jennifer_sweets_sharing (total_sweets : ℕ) (sweets_per_person : ℕ) (h1 : total_sweets = 1024) (h2 : sweets_per_person = 256) :
  (total_sweets / sweets_per_person) - 1 = 3 := by
  sorry

end jennifer_sweets_sharing_l3016_301644


namespace five_balls_four_boxes_l3016_301654

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 56 := by
  sorry

end five_balls_four_boxes_l3016_301654


namespace cow_characteristic_difference_l3016_301675

def total_cows : ℕ := 600
def male_ratio : ℕ := 5
def female_ratio : ℕ := 3
def transgender_ratio : ℕ := 2

def male_horned_percentage : ℚ := 50 / 100
def male_spotted_percentage : ℚ := 40 / 100
def male_brown_percentage : ℚ := 20 / 100

def female_spotted_percentage : ℚ := 35 / 100
def female_horned_percentage : ℚ := 25 / 100
def female_white_percentage : ℚ := 60 / 100

def transgender_unique_pattern_percentage : ℚ := 45 / 100
def transgender_spotted_horned_percentage : ℚ := 30 / 100
def transgender_black_percentage : ℚ := 50 / 100

theorem cow_characteristic_difference :
  let total_ratio := male_ratio + female_ratio + transgender_ratio
  let male_count := (male_ratio : ℚ) / total_ratio * total_cows
  let female_count := (female_ratio : ℚ) / total_ratio * total_cows
  let transgender_count := (transgender_ratio : ℚ) / total_ratio * total_cows
  let spotted_females := female_spotted_percentage * female_count
  let horned_males := male_horned_percentage * male_count
  let brown_males := male_brown_percentage * male_count
  let unique_pattern_transgender := transgender_unique_pattern_percentage * transgender_count
  let white_horned_females := female_horned_percentage * female_white_percentage * female_count
  let characteristic_sum := horned_males + brown_males + unique_pattern_transgender + white_horned_females
  spotted_females - characteristic_sum = -291 := by sorry

end cow_characteristic_difference_l3016_301675


namespace cosine_sum_l3016_301650

theorem cosine_sum (α β : Real) : 
  α ∈ Set.Ioo 0 (π/3) →
  β ∈ Set.Ioo (π/6) (π/2) →
  5 * Real.sqrt 3 * Real.sin α + 5 * Real.cos α = 8 →
  Real.sqrt 2 * Real.sin β + Real.sqrt 6 * Real.cos β = 2 →
  Real.cos (α + β) = -(Real.sqrt 2) / 10 := by
sorry

end cosine_sum_l3016_301650


namespace triangle_perimeter_in_square_l3016_301602

/-- Given a square with side length 70√2 cm, divided into four congruent 45-45-90 triangles
    by its diagonals, the perimeter of one of these triangles is 140√2 + 140 cm. -/
theorem triangle_perimeter_in_square (side_length : ℝ) (h : side_length = 70 * Real.sqrt 2) :
  let diagonal := side_length * Real.sqrt 2
  let triangle_perimeter := 2 * side_length + diagonal
  triangle_perimeter = 140 * Real.sqrt 2 + 140 := by
  sorry

end triangle_perimeter_in_square_l3016_301602


namespace park_track_area_increase_l3016_301605

def small_diameter : ℝ := 15
def large_diameter : ℝ := 20

theorem park_track_area_increase :
  let small_area := π * (small_diameter / 2)^2
  let large_area := π * (large_diameter / 2)^2
  (large_area - small_area) / small_area = 7 / 9 := by
  sorry

end park_track_area_increase_l3016_301605


namespace total_gain_percentage_five_articles_l3016_301680

/-- Calculate the total gain percentage for five articles --/
theorem total_gain_percentage_five_articles 
  (cp1 cp2 cp3 cp4 cp5 : ℝ)
  (sp1 sp2 sp3 sp4 sp5 : ℝ)
  (h1 : cp1 = 18.50)
  (h2 : cp2 = 25.75)
  (h3 : cp3 = 42.60)
  (h4 : cp4 = 29.90)
  (h5 : cp5 = 56.20)
  (h6 : sp1 = 22.50)
  (h7 : sp2 = 32.25)
  (h8 : sp3 = 49.60)
  (h9 : sp4 = 36.40)
  (h10 : sp5 = 65.80) :
  let total_cp := cp1 + cp2 + cp3 + cp4 + cp5
  let total_sp := sp1 + sp2 + sp3 + sp4 + sp5
  let total_gain := total_sp - total_cp
  let gain_percentage := (total_gain / total_cp) * 100
  gain_percentage = 19.35 :=
by
  sorry


end total_gain_percentage_five_articles_l3016_301680


namespace expression_simplification_and_evaluation_l3016_301689

theorem expression_simplification_and_evaluation :
  let x : ℤ := -3
  let y : ℤ := -2
  3 * x^2 * y - (2 * x^2 * y - (2 * x * y - x^2 * y) - 4 * x^2 * y) - x * y = -66 :=
by sorry

end expression_simplification_and_evaluation_l3016_301689


namespace square_root_divided_by_six_l3016_301679

theorem square_root_divided_by_six : Real.sqrt 144 / 6 = 2 := by sorry

end square_root_divided_by_six_l3016_301679


namespace fraction_evaluation_l3016_301690

theorem fraction_evaluation : (3 : ℚ) / (2 - 5 / 4) = 4 := by
  sorry

end fraction_evaluation_l3016_301690


namespace pizza_order_proof_l3016_301695

/-- The number of slices in each pizza -/
def slices_per_pizza : ℕ := 12

/-- The number of slices Dean ate -/
def dean_slices : ℕ := slices_per_pizza / 2

/-- The number of slices Frank ate -/
def frank_slices : ℕ := 3

/-- The number of slices Sammy ate -/
def sammy_slices : ℕ := slices_per_pizza / 3

/-- The number of slices left over -/
def leftover_slices : ℕ := 11

/-- The total number of pizzas Dean ordered -/
def total_pizzas : ℕ := 2

theorem pizza_order_proof :
  (dean_slices + frank_slices + sammy_slices + leftover_slices) / slices_per_pizza = total_pizzas :=
by sorry

end pizza_order_proof_l3016_301695


namespace problem_solution_l3016_301668

def f (n : ℕ) : ℚ := (n^2 - 5*n + 4) / (n - 4)

theorem problem_solution :
  (f 1 = 0) ∧
  (∀ n : ℕ, n ≠ 4 → (f n = 5 ↔ n = 6)) ∧
  (∀ n : ℕ, n ≠ 4 → f n ≠ 3) := by sorry

end problem_solution_l3016_301668


namespace diophantine_equation_solution_l3016_301660

theorem diophantine_equation_solution (k ℓ : ℤ) :
  5 * k + 3 * ℓ = 32 ↔ ∃ x : ℤ, k = -32 + 3 * x ∧ ℓ = 64 - 5 * x :=
by sorry

end diophantine_equation_solution_l3016_301660


namespace chocolate_pieces_per_box_l3016_301696

theorem chocolate_pieces_per_box 
  (total_boxes : ℕ) 
  (given_away : ℕ) 
  (remaining_pieces : ℕ) 
  (h1 : total_boxes = 12) 
  (h2 : given_away = 7) 
  (h3 : remaining_pieces = 30) : 
  remaining_pieces / (total_boxes - given_away) = 6 :=
by sorry

end chocolate_pieces_per_box_l3016_301696


namespace contrapositive_correct_l3016_301657

-- Define a triangle
structure Triangle :=
  (A B C : Point)

-- Define the property of being an isosceles triangle
def isIsosceles (t : Triangle) : Prop := sorry

-- Define the property of having two equal interior angles
def hasTwoEqualAngles (t : Triangle) : Prop := sorry

-- The original statement
def originalStatement (t : Triangle) : Prop :=
  ¬(isIsosceles t) → ¬(hasTwoEqualAngles t)

-- The contrapositive of the original statement
def contrapositive (t : Triangle) : Prop :=
  hasTwoEqualAngles t → isIsosceles t

-- Theorem stating that the contrapositive is correct
theorem contrapositive_correct :
  ∀ t : Triangle, originalStatement t ↔ contrapositive t :=
sorry

end contrapositive_correct_l3016_301657


namespace inscribed_octagon_area_l3016_301631

/-- The area of a regular octagon inscribed in a circle -/
theorem inscribed_octagon_area (r : ℝ) (h : r^2 = 256) :
  2 * (1 + Real.sqrt 2) * (r * Real.sqrt (2 - Real.sqrt 2))^2 = 512 * Real.sqrt 2 := by
  sorry

end inscribed_octagon_area_l3016_301631


namespace executive_committee_selection_l3016_301606

theorem executive_committee_selection (total_members : ℕ) (committee_size : ℕ) (ineligible_members : ℕ) :
  total_members = 30 →
  committee_size = 5 →
  ineligible_members = 4 →
  Nat.choose (total_members - ineligible_members) committee_size = 60770 := by
sorry

end executive_committee_selection_l3016_301606


namespace sum_of_reciprocals_equals_two_l3016_301628

theorem sum_of_reciprocals_equals_two (a b c : ℝ) 
  (ha : a^3 - 2020*a + 1010 = 0)
  (hb : b^3 - 2020*b + 1010 = 0)
  (hc : c^3 - 2020*c + 1010 = 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  1/a + 1/b + 1/c = 2 := by
  sorry

end sum_of_reciprocals_equals_two_l3016_301628


namespace function_proof_l3016_301685

theorem function_proof (f : ℤ → ℤ) (h1 : f 0 = 1) (h2 : f 2012 = 2013) :
  ∀ n : ℤ, f n = n + 1 := by
  sorry

end function_proof_l3016_301685


namespace A_intersect_B_eq_open_zero_one_l3016_301624

def A : Set ℝ := {x : ℝ | x^2 - 1 < 0}
def B : Set ℝ := {x : ℝ | x > 0}

theorem A_intersect_B_eq_open_zero_one : A ∩ B = Set.Ioo 0 1 := by sorry

end A_intersect_B_eq_open_zero_one_l3016_301624


namespace alkaline_probability_is_two_fifths_l3016_301652

/-- Represents the total number of solutions -/
def total_solutions : ℕ := 5

/-- Represents the number of alkaline solutions -/
def alkaline_solutions : ℕ := 2

/-- Represents the probability of selecting an alkaline solution -/
def alkaline_probability : ℚ := alkaline_solutions / total_solutions

/-- Theorem stating that the probability of selecting an alkaline solution is 2/5 -/
theorem alkaline_probability_is_two_fifths : 
  alkaline_probability = 2 / 5 := by sorry

end alkaline_probability_is_two_fifths_l3016_301652


namespace simplify_fraction_l3016_301671

theorem simplify_fraction : (216 : ℚ) / 4536 = 1 / 21 := by
  sorry

end simplify_fraction_l3016_301671


namespace daves_tiling_area_l3016_301665

theorem daves_tiling_area (total_area : ℝ) (clara_ratio : ℕ) (dave_ratio : ℕ) 
  (h1 : total_area = 330)
  (h2 : clara_ratio = 4)
  (h3 : dave_ratio = 7) : 
  (dave_ratio : ℝ) / ((clara_ratio : ℝ) + (dave_ratio : ℝ)) * total_area = 210 :=
by sorry

end daves_tiling_area_l3016_301665


namespace max_sum_of_pairwise_sums_l3016_301623

/-- Given a set of four numbers with six pairwise sums, find the maximum value of x + y -/
theorem max_sum_of_pairwise_sums (a b c d : ℝ) : 
  let sums : Finset ℝ := {a + b, a + c, a + d, b + c, b + d, c + d}
  ∃ (x y : ℝ), x ∈ sums ∧ y ∈ sums ∧ 
    sums = {210, 345, 275, 255, x, y} →
    (∀ (u v : ℝ), u ∈ sums ∧ v ∈ sums → u + v ≤ 775) ∧
    (∃ (u v : ℝ), u ∈ sums ∧ v ∈ sums ∧ u + v = 775) :=
by sorry

end max_sum_of_pairwise_sums_l3016_301623


namespace polynomial_coefficient_difference_l3016_301638

theorem polynomial_coefficient_difference (m n : ℝ) : 
  (∀ x : ℝ, 3 * x * (x - 1) = m * x^2 + n * x) → m - n = 6 := by
sorry

end polynomial_coefficient_difference_l3016_301638


namespace f_neg_one_eq_zero_iff_r_eq_neg_eight_l3016_301620

/-- A polynomial function f(x) with a parameter r -/
def f (r : ℝ) (x : ℝ) : ℝ := 3 * x^4 + x^3 + 2 * x^2 - 4 * x + r

/-- Theorem stating that f(-1) = 0 if and only if r = -8 -/
theorem f_neg_one_eq_zero_iff_r_eq_neg_eight :
  ∀ r : ℝ, f r (-1) = 0 ↔ r = -8 := by sorry

end f_neg_one_eq_zero_iff_r_eq_neg_eight_l3016_301620


namespace cos_210_degrees_l3016_301692

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_degrees_l3016_301692


namespace tan_60_plus_inv_sqrt_3_l3016_301659

theorem tan_60_plus_inv_sqrt_3 :
  Real.tan (π / 3) + (Real.sqrt 3)⁻¹ = (4 * Real.sqrt 3) / 3 := by
  sorry

end tan_60_plus_inv_sqrt_3_l3016_301659


namespace cookie_difference_l3016_301621

theorem cookie_difference (initial_sweet initial_salty eaten_sweet eaten_salty : ℕ) :
  initial_sweet = 39 →
  initial_salty = 6 →
  eaten_sweet = 32 →
  eaten_salty = 23 →
  eaten_sweet - eaten_salty = 9 := by
  sorry

end cookie_difference_l3016_301621


namespace exists_axisymmetric_capital_letter_l3016_301630

-- Define a type for capital letters
inductive CapitalLetter
  | A | B | C | D | E | F | G | H | I | J | K | L | M
  | N | O | P | Q | R | S | T | U | V | W | X | Y | Z

-- Define a predicate for axisymmetric figures
def isAxisymmetric (letter : CapitalLetter) : Prop :=
  sorry  -- The actual implementation would depend on how we define axisymmetry

-- Theorem statement
theorem exists_axisymmetric_capital_letter :
  ∃ (letter : CapitalLetter), 
    (letter = CapitalLetter.A ∨ 
     letter = CapitalLetter.B ∨ 
     letter = CapitalLetter.D ∨ 
     letter = CapitalLetter.E) ∧ 
    isAxisymmetric letter :=
by
  sorry


end exists_axisymmetric_capital_letter_l3016_301630


namespace sqrt_product_equality_l3016_301651

theorem sqrt_product_equality (x y : ℝ) (hx : x ≥ 0) :
  Real.sqrt (3 * x) * Real.sqrt ((1 / 3) * x * y) = x * Real.sqrt y := by
  sorry

end sqrt_product_equality_l3016_301651


namespace ryegrass_percentage_in_mixture_l3016_301674

-- Define the compositions of mixtures X and Y
def x_ryegrass : ℝ := 0.4
def x_bluegrass : ℝ := 0.6
def y_ryegrass : ℝ := 0.25
def y_fescue : ℝ := 0.75

-- Define the proportion of X in the final mixture
def x_proportion : ℝ := 0.3333333333333333

-- Define the proportion of Y in the final mixture
def y_proportion : ℝ := 1 - x_proportion

-- Theorem statement
theorem ryegrass_percentage_in_mixture :
  x_ryegrass * x_proportion + y_ryegrass * y_proportion = 0.3 := by sorry

end ryegrass_percentage_in_mixture_l3016_301674


namespace solve_quadratic_l3016_301641

-- Define the universal set U
def U : Set ℕ := {2, 3, 5}

-- Define the set A
def A (b c : ℤ) : Set ℕ := {x ∈ U | x^2 + b*x + c = 0}

-- Define the complement of A with respect to U
def complement_A (b c : ℤ) : Set ℕ := U \ A b c

-- Theorem statement
theorem solve_quadratic (b c : ℤ) : complement_A b c = {2} → b = -8 ∧ c = 15 := by
  sorry


end solve_quadratic_l3016_301641


namespace roots_between_values_l3016_301676

theorem roots_between_values (a b c : ℝ) (h : a < b ∧ b < c) :
  ∃ x₁ x₂ : ℝ,
    (1 / (x₁ - a) + 1 / (x₁ - b) + 1 / (x₁ - c) = 0) ∧
    (1 / (x₂ - a) + 1 / (x₂ - b) + 1 / (x₂ - c) = 0) ∧
    (a < x₁ ∧ x₁ < b ∧ b < x₂ ∧ x₂ < c) := by
  sorry

end roots_between_values_l3016_301676


namespace sum_of_common_ratios_is_three_l3016_301632

/-- Given two nonconstant geometric sequences with different common ratios,
    prove that the sum of their common ratios is 3 under certain conditions. -/
theorem sum_of_common_ratios_is_three
  (k : ℝ) (p r : ℝ) (hp : p ≠ 1) (hr : r ≠ 1) (hpr : p ≠ r) (hk : k ≠ 0)
  (h : k * p^2 - k * r^2 = 3 * (k * p - k * r)) :
  p + r = 3 := by
  sorry

end sum_of_common_ratios_is_three_l3016_301632


namespace neg_one_is_square_sum_of_three_squares_zero_not_sum_of_three_nonzero_squares_l3016_301646

/-- A field K of characteristic p where p ≡ 1 (mod 4) -/
class CharacteristicP (K : Type) [Field K] where
  char_p : Nat
  char_p_prime : Prime char_p
  char_p_mod_4 : char_p % 4 = 1

variable {K : Type} [Field K] [CharacteristicP K]

/-- -1 is a square in K -/
theorem neg_one_is_square : ∃ x : K, x^2 = -1 := by sorry

/-- Any nonzero element in K can be written as the sum of three nonzero squares -/
theorem sum_of_three_squares (a : K) (ha : a ≠ 0) : 
  ∃ x y z : K, x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x^2 + y^2 + z^2 = a := by sorry

/-- 0 cannot be written as the sum of three nonzero squares -/
theorem zero_not_sum_of_three_nonzero_squares :
  ¬∃ x y z : K, x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x^2 + y^2 + z^2 = 0 := by sorry

end neg_one_is_square_sum_of_three_squares_zero_not_sum_of_three_nonzero_squares_l3016_301646


namespace last_digits_divisible_by_4_l3016_301635

-- Define a function to check if a number is divisible by 4
def divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

-- Define a function to get the last digit of a number
def last_digit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem last_digits_divisible_by_4 :
  ∃! (s : Finset ℕ), (∀ n ∈ s, ∃ m : ℕ, divisible_by_4 m ∧ last_digit m = n) ∧ s.card = 3 :=
sorry

end last_digits_divisible_by_4_l3016_301635


namespace range_of_m_l3016_301645

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 5

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) m, f x ≤ 10) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) m, f x ≥ 1) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) m, f x = 10) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) m, f x = 1) →
  m ∈ Set.Icc 2 5 :=
by sorry

end range_of_m_l3016_301645


namespace train_length_train_length_approximation_l3016_301607

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_sec : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  speed_ms * time_sec

/-- Prove that a train traveling at 50 km/h and crossing a pole in 18 seconds has a length of approximately 250 meters -/
theorem train_length_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ |train_length 50 18 - 250| < ε :=
sorry

end train_length_train_length_approximation_l3016_301607


namespace inverse_iff_horizontal_line_test_l3016_301634

-- Define a type for our functions
def Function2D := ℝ → ℝ

-- Define the horizontal line test
def passes_horizontal_line_test (f : Function2D) : Prop :=
  ∀ y : ℝ, ∀ x₁ x₂ : ℝ, f x₁ = y ∧ f x₂ = y → x₁ = x₂

-- Define what it means for a function to have an inverse
def has_inverse (f : Function2D) : Prop :=
  ∃ g : Function2D, (∀ x : ℝ, g (f x) = x) ∧ (∀ y : ℝ, f (g y) = y)

-- Theorem statement
theorem inverse_iff_horizontal_line_test (f : Function2D) :
  has_inverse f ↔ passes_horizontal_line_test f :=
sorry

end inverse_iff_horizontal_line_test_l3016_301634
