import Mathlib

namespace NUMINAMATH_CALUDE_cos_2alpha_2beta_l3687_368793

theorem cos_2alpha_2beta (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1/3) 
  (h2 : Real.cos α * Real.sin β = 1/6) : 
  Real.cos (2*α + 2*β) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_2beta_l3687_368793


namespace NUMINAMATH_CALUDE_choose_three_from_nine_l3687_368795

theorem choose_three_from_nine : Nat.choose 9 3 = 84 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_nine_l3687_368795


namespace NUMINAMATH_CALUDE_necessary_condition_equality_l3687_368741

theorem necessary_condition_equality (a b c : ℝ) (h : c ≠ 0) :
  a = b → a * c = b * c :=
by sorry

end NUMINAMATH_CALUDE_necessary_condition_equality_l3687_368741


namespace NUMINAMATH_CALUDE_duck_cow_problem_l3687_368751

theorem duck_cow_problem (ducks cows : ℕ) : 
  2 * ducks + 4 * cows = 2 * (ducks + cows) + 34 → cows = 17 := by
  sorry

end NUMINAMATH_CALUDE_duck_cow_problem_l3687_368751


namespace NUMINAMATH_CALUDE_total_non_basalt_rocks_l3687_368728

/-- Given two boxes of rocks with some being basalt, calculate the total number of non-basalt rocks --/
theorem total_non_basalt_rocks (total_A total_B basalt_A basalt_B : ℕ) 
  (h1 : total_A = 57)
  (h2 : basalt_A = 25)
  (h3 : total_B = 49)
  (h4 : basalt_B = 19) :
  (total_A - basalt_A) + (total_B - basalt_B) = 62 := by
  sorry

#check total_non_basalt_rocks

end NUMINAMATH_CALUDE_total_non_basalt_rocks_l3687_368728


namespace NUMINAMATH_CALUDE_solve_for_x_l3687_368788

-- Define the variables
variable (x y : ℝ)

-- State the theorem
theorem solve_for_x (eq1 : x + 2 * y = 12) (eq2 : y = 3) : x = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l3687_368788


namespace NUMINAMATH_CALUDE_rachel_money_theorem_l3687_368758

def rachel_money_problem (initial_earnings : ℝ) 
  (lunch_fraction : ℝ) (clothes_percent : ℝ) (dvd_cost : ℝ) (supplies_percent : ℝ) : Prop :=
  let lunch_cost := initial_earnings * lunch_fraction
  let clothes_cost := initial_earnings * (clothes_percent / 100)
  let supplies_cost := initial_earnings * (supplies_percent / 100)
  let total_expenses := lunch_cost + clothes_cost + dvd_cost + supplies_cost
  let money_left := initial_earnings - total_expenses
  money_left = 74.50

theorem rachel_money_theorem :
  rachel_money_problem 200 0.25 15 24.50 10.5 := by
  sorry

end NUMINAMATH_CALUDE_rachel_money_theorem_l3687_368758


namespace NUMINAMATH_CALUDE_min_value_of_function_l3687_368763

theorem min_value_of_function (x : ℝ) (h : x ≥ 0) :
  x + 1 / (x + 1) ≥ 1 ∧ ∃ y : ℝ, y ≥ 0 ∧ y + 1 / (y + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3687_368763


namespace NUMINAMATH_CALUDE_expected_other_marbles_l3687_368717

/-- Represents the distribution of marble colors in Percius's collection -/
structure MarbleCollection where
  clear_percent : ℝ
  black_percent : ℝ
  other_percent : ℝ
  sum_to_one : clear_percent + black_percent + other_percent = 1

/-- Percius's marble collection -/
def percius_marbles : MarbleCollection where
  clear_percent := 0.4
  black_percent := 0.2
  other_percent := 0.4
  sum_to_one := by norm_num

/-- The number of marbles selected by the friend -/
def selected_marbles : ℕ := 5

/-- Theorem: The expected number of marbles of other colors when selecting 5 marbles is 2 -/
theorem expected_other_marbles :
  (selected_marbles : ℝ) * percius_marbles.other_percent = 2 := by sorry

end NUMINAMATH_CALUDE_expected_other_marbles_l3687_368717


namespace NUMINAMATH_CALUDE_amy_remaining_money_l3687_368742

theorem amy_remaining_money (initial_amount : ℝ) 
  (doll_cost doll_quantity : ℝ)
  (board_game_cost board_game_quantity : ℝ)
  (comic_book_cost comic_book_quantity : ℝ) :
  initial_amount = 100 ∧
  doll_cost = 1.25 ∧
  doll_quantity = 3 ∧
  board_game_cost = 12.75 ∧
  board_game_quantity = 2 ∧
  comic_book_cost = 3.50 ∧
  comic_book_quantity = 4 →
  initial_amount - (doll_cost * doll_quantity + board_game_cost * board_game_quantity + comic_book_cost * comic_book_quantity) = 56.75 := by
sorry

end NUMINAMATH_CALUDE_amy_remaining_money_l3687_368742


namespace NUMINAMATH_CALUDE_planet_surface_area_unchanged_l3687_368745

theorem planet_surface_area_unchanged 
  (planet_diameter : ℝ) 
  (explosion_radius : ℝ) 
  (h1 : planet_diameter = 10000) 
  (h2 : explosion_radius = 5000) :
  let planet_radius : ℝ := planet_diameter / 2
  let initial_surface_area : ℝ := 4 * Real.pi * planet_radius ^ 2
  let new_surface_area : ℝ := initial_surface_area
  new_surface_area = 100000000 * Real.pi := by sorry

end NUMINAMATH_CALUDE_planet_surface_area_unchanged_l3687_368745


namespace NUMINAMATH_CALUDE_initial_interest_rate_l3687_368723

/-- Proves that the initial interest rate is 5% given the problem conditions --/
theorem initial_interest_rate 
  (initial_investment : ℝ) 
  (additional_investment : ℝ) 
  (additional_rate : ℝ) 
  (total_rate : ℝ) 
  (h1 : initial_investment = 8000)
  (h2 : additional_investment = 4000)
  (h3 : additional_rate = 8)
  (h4 : total_rate = 6)
  : (initial_investment * (100 * total_rate - additional_investment * additional_rate) / 
    (100 * (initial_investment + additional_investment))) = 5 := by
  sorry

end NUMINAMATH_CALUDE_initial_interest_rate_l3687_368723


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l3687_368756

def determinant (a b c d : ℂ) : ℂ := a * d - b * c

theorem z_in_fourth_quadrant (z : ℂ) 
  (h : determinant z 1 Complex.I Complex.I = 2 + Complex.I) : 
  0 < z.re ∧ z.im < 0 := by sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l3687_368756


namespace NUMINAMATH_CALUDE_mary_sugar_already_added_l3687_368794

/-- Given a recipe that requires a total amount of sugar and the amount still needed to be added,
    calculate the amount of sugar already put in. -/
def sugar_already_added (total_required : ℕ) (still_needed : ℕ) : ℕ :=
  total_required - still_needed

/-- Theorem stating that given the specific values from the problem,
    the amount of sugar already added is 2 cups. -/
theorem mary_sugar_already_added :
  sugar_already_added 13 11 = 2 := by sorry

end NUMINAMATH_CALUDE_mary_sugar_already_added_l3687_368794


namespace NUMINAMATH_CALUDE_simplify_expression_l3687_368781

theorem simplify_expression (a b : ℝ) : 
  (-2 * a^2 * b^3) * (-a * b^2)^2 + (-1/2 * a^2 * b^3)^2 * 4 * b = -a^4 * b^7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3687_368781


namespace NUMINAMATH_CALUDE_sqrt_62_plus_24_sqrt_11_l3687_368719

theorem sqrt_62_plus_24_sqrt_11 :
  ∃ (a b c : ℤ), 
    (∀ (n : ℕ), n > 1 → ¬(∃ (k : ℕ), c = n^2 * k)) →
    Real.sqrt (62 + 24 * Real.sqrt 11) = a + b * Real.sqrt c ∧
    a = 6 ∧ b = 2 ∧ c = 11 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_62_plus_24_sqrt_11_l3687_368719


namespace NUMINAMATH_CALUDE_fourth_six_probability_l3687_368782

/-- Represents a six-sided die --/
structure Die :=
  (prob_six : ℚ)
  (prob_other : ℚ)
  (valid_probs : prob_six + 5 * prob_other = 1)

/-- The fair die --/
def fair_die : Die :=
  { prob_six := 1/6,
    prob_other := 1/6,
    valid_probs := by norm_num }

/-- The biased die --/
def biased_die : Die :=
  { prob_six := 3/4,
    prob_other := 1/20,
    valid_probs := by norm_num }

/-- The probability of rolling three sixes with a given die --/
def prob_three_sixes (d : Die) : ℚ := d.prob_six^3

/-- The probability of the fourth roll being a six given the first three were sixes --/
def prob_fourth_six (fair : Die) (biased : Die) : ℚ :=
  let p_fair := prob_three_sixes fair
  let p_biased := prob_three_sixes biased
  let total := p_fair + p_biased
  (p_fair / total) * fair.prob_six + (p_biased / total) * biased.prob_six

theorem fourth_six_probability :
  prob_fourth_six fair_die biased_die = 685 / 922 :=
sorry

end NUMINAMATH_CALUDE_fourth_six_probability_l3687_368782


namespace NUMINAMATH_CALUDE_congruence_theorem_l3687_368797

theorem congruence_theorem (n : ℕ+) :
  (122 ^ n.val - 102 ^ n.val - 21 ^ n.val) % 2020 = 2019 := by
  sorry

end NUMINAMATH_CALUDE_congruence_theorem_l3687_368797


namespace NUMINAMATH_CALUDE_least_common_multiple_5_to_15_l3687_368727

theorem least_common_multiple_5_to_15 : ∃ n : ℕ, 
  (∀ k : ℕ, 5 ≤ k → k ≤ 15 → k ∣ n) ∧ 
  (∀ m : ℕ, m > 0 → (∀ k : ℕ, 5 ≤ k → k ≤ 15 → k ∣ m) → n ≤ m) ∧
  n = 360360 := by
sorry

end NUMINAMATH_CALUDE_least_common_multiple_5_to_15_l3687_368727


namespace NUMINAMATH_CALUDE_jar_water_problem_l3687_368726

theorem jar_water_problem (s l : ℝ) (hs : s > 0) (hl : l > 0) : 
  (1/8 : ℝ) * s = (1/6 : ℝ) * l → (1/6 : ℝ) * l + (1/8 : ℝ) * s = (1/3 : ℝ) * l :=
by sorry

end NUMINAMATH_CALUDE_jar_water_problem_l3687_368726


namespace NUMINAMATH_CALUDE_whale_consumption_increase_l3687_368765

/-- Represents the whale's plankton consumption pattern over 9 hours -/
structure WhaleConsumption where
  initial : ℝ  -- Initial consumption in the first hour
  increase : ℝ  -- Increase in consumption each hour
  total : ℝ     -- Total consumption over 9 hours
  sixth_hour : ℝ -- Consumption in the sixth hour

/-- The whale's consumption satisfies the given conditions -/
def satisfies_conditions (w : WhaleConsumption) : Prop :=
  w.total = 270 ∧ 
  w.sixth_hour = 33 ∧ 
  w.total = (9 * w.initial + 36 * w.increase) ∧
  w.sixth_hour = w.initial + 5 * w.increase

/-- The theorem stating that the increase in consumption is 3 kilos per hour -/
theorem whale_consumption_increase (w : WhaleConsumption) 
  (h : satisfies_conditions w) : w.increase = 3 := by
  sorry

end NUMINAMATH_CALUDE_whale_consumption_increase_l3687_368765


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l3687_368724

theorem rectangle_diagonal (side1 : ℝ) (area : ℝ) (diagonal : ℝ) : 
  side1 = 6 → area = 48 → diagonal = 10 → 
  ∃ (side2 : ℝ), 
    side1 * side2 = area ∧ 
    diagonal^2 = side1^2 + side2^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l3687_368724


namespace NUMINAMATH_CALUDE_odd_sum_of_squares_implies_odd_sum_l3687_368722

theorem odd_sum_of_squares_implies_odd_sum (n m : ℤ) 
  (h : Odd (n^2 + m^2)) : Odd (n + m) := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_of_squares_implies_odd_sum_l3687_368722


namespace NUMINAMATH_CALUDE_sum_of_cubic_roots_l3687_368774

theorem sum_of_cubic_roots (a b c d : ℝ) (h : ∀ x : ℝ, x^3 + x^2 - 6*x - 20 = 4*x + 24 ↔ a*x^3 + b*x^2 + c*x + d = 0) :
  a ≠ 0 → (sum_of_roots : ℝ) = -b / a ∧ sum_of_roots = -1 :=
by sorry


end NUMINAMATH_CALUDE_sum_of_cubic_roots_l3687_368774


namespace NUMINAMATH_CALUDE_no_solutions_absolute_value_equation_l3687_368752

theorem no_solutions_absolute_value_equation :
  ¬ ∃ x : ℝ, |x - 2| = |x - 1| + |x - 4| := by
sorry

end NUMINAMATH_CALUDE_no_solutions_absolute_value_equation_l3687_368752


namespace NUMINAMATH_CALUDE_R_value_at_S_5_l3687_368787

/-- Given R = gS^2 - 4S, and R = 11 when S = 3, prove that R = 395/9 when S = 5 -/
theorem R_value_at_S_5 (g : ℚ) :
  (∀ S : ℚ, g * S^2 - 4 * S = 11 → S = 3) →
  g * 5^2 - 4 * 5 = 395 / 9 := by
sorry

end NUMINAMATH_CALUDE_R_value_at_S_5_l3687_368787


namespace NUMINAMATH_CALUDE_parallelogram_cross_section_exists_l3687_368778

/-- A cuboid in 3D space -/
structure Cuboid where
  -- Define the cuboid structure (you may need to add more fields)
  dummy : Unit

/-- A plane in 3D space -/
structure Plane where
  -- Define the plane structure (you may need to add more fields)
  dummy : Unit

/-- The cross-section resulting from a plane intersecting a cuboid -/
def crossSection (c : Cuboid) (p : Plane) : Set (ℝ × ℝ × ℝ) :=
  sorry -- Define the cross-section

/-- A predicate to check if a set of points forms a parallelogram -/
def isParallelogram (s : Set (ℝ × ℝ × ℝ)) : Prop :=
  sorry -- Define the conditions for a parallelogram

/-- Theorem stating that there exists a plane that intersects a cuboid to form a parallelogram cross-section -/
theorem parallelogram_cross_section_exists :
  ∃ (c : Cuboid) (p : Plane), isParallelogram (crossSection c p) :=
sorry

end NUMINAMATH_CALUDE_parallelogram_cross_section_exists_l3687_368778


namespace NUMINAMATH_CALUDE_complex_modulus_theorem_l3687_368714

theorem complex_modulus_theorem (r : ℝ) (z : ℂ) 
  (h1 : |r| < 3) 
  (h2 : r ≠ 2) 
  (h3 : z + r * z⁻¹ = 2) : 
  Complex.abs z = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_theorem_l3687_368714


namespace NUMINAMATH_CALUDE_g_zeros_l3687_368792

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - 2 * x + 1

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + x * Real.log x

theorem g_zeros (a : ℝ) (h : a > 0) :
  (∃! x, g a x = 0 ∧ x > 0) ∧ a = Real.exp (-1) ∨
  (∀ x > 0, g a x ≠ 0) ∧ a > Real.exp (-1) ∨
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ g a x₁ = 0 ∧ g a x₂ = 0 ∧
    ∀ x, x > 0 → g a x = 0 → x = x₁ ∨ x = x₂) ∧ 0 < a ∧ a < Real.exp (-1) :=
sorry

end NUMINAMATH_CALUDE_g_zeros_l3687_368792


namespace NUMINAMATH_CALUDE_specific_bulb_probability_l3687_368747

/-- The number of light bulbs -/
def num_bulbs : ℕ := 4

/-- The number of bulbs to be installed -/
def num_installed : ℕ := 3

/-- The number of ways to arrange n items taken k at a time -/
def permutations (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

/-- The probability of installing a specific bulb at a specific vertex -/
def probability : ℚ := (permutations (num_bulbs - 1) (num_installed - 1)) / (permutations num_bulbs num_installed)

theorem specific_bulb_probability : probability = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_specific_bulb_probability_l3687_368747


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_even_integers_l3687_368739

def is_sum_of_four_consecutive_even (n : ℤ) : Prop :=
  ∃ m : ℤ, 4 * m + 12 = n ∧ m % 2 = 0

theorem sum_of_four_consecutive_even_integers :
  (is_sum_of_four_consecutive_even 12) ∧
  (¬ is_sum_of_four_consecutive_even 40) ∧
  (¬ is_sum_of_four_consecutive_even 80) ∧
  (is_sum_of_four_consecutive_even 100) ∧
  (is_sum_of_four_consecutive_even 180) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_even_integers_l3687_368739


namespace NUMINAMATH_CALUDE_count_non_multiples_is_675_l3687_368790

/-- The count of three-digit numbers that are not multiples of 6 or 8 -/
def count_non_multiples : ℕ :=
  let total_three_digit_numbers := 999 - 100 + 1
  let multiples_of_6 := (999 / 6) - (99 / 6)
  let multiples_of_8 := (999 / 8) - (99 / 8)
  let multiples_of_24 := (999 / 24) - (99 / 24)
  total_three_digit_numbers - (multiples_of_6 + multiples_of_8 - multiples_of_24)

theorem count_non_multiples_is_675 : count_non_multiples = 675 := by
  sorry

end NUMINAMATH_CALUDE_count_non_multiples_is_675_l3687_368790


namespace NUMINAMATH_CALUDE_cloth_cost_price_l3687_368773

theorem cloth_cost_price 
  (selling_price : ℕ) 
  (cloth_length : ℕ) 
  (loss_per_meter : ℕ) 
  (h1 : selling_price = 18000) 
  (h2 : cloth_length = 600) 
  (h3 : loss_per_meter = 5) : 
  (selling_price + cloth_length * loss_per_meter) / cloth_length = 35 := by
sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l3687_368773


namespace NUMINAMATH_CALUDE_triangle_max_side_sum_l3687_368753

theorem triangle_max_side_sum (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  B = π / 3 ∧
  b = Real.sqrt 3 ∧
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C →
  (2 * a + c) ≤ 2 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_side_sum_l3687_368753


namespace NUMINAMATH_CALUDE_equation_solution_l3687_368798

theorem equation_solution :
  let f (x : ℝ) := x^4 / (2*x + 1) + x^2 - 6*(2*x + 1)
  ∀ x : ℝ, f x = 0 ↔ x = -3 - Real.sqrt 6 ∨ x = -3 + Real.sqrt 6 ∨ x = 2 - Real.sqrt 6 ∨ x = 2 + Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3687_368798


namespace NUMINAMATH_CALUDE_six_point_five_minutes_in_seconds_l3687_368737

/-- Converts minutes to seconds -/
def minutes_to_seconds (minutes : ℝ) : ℝ := minutes * 60

/-- Theorem stating that 6.5 minutes equals 390 seconds -/
theorem six_point_five_minutes_in_seconds : 
  minutes_to_seconds 6.5 = 390 := by sorry

end NUMINAMATH_CALUDE_six_point_five_minutes_in_seconds_l3687_368737


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l3687_368703

theorem arithmetic_sequence_first_term
  (a d : ℝ)
  (sum_100 : (100 : ℝ) / 2 * (2 * a + 99 * d) = 1800)
  (sum_51_to_150 : (100 : ℝ) / 2 * (2 * a + 199 * d) = 6300) :
  a = -26.55 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l3687_368703


namespace NUMINAMATH_CALUDE_max_sum_of_four_digit_integers_l3687_368721

/-- A function that returns true if a number is a 4-digit integer -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A function that returns the set of digits in a number -/
def digits (n : ℕ) : Finset ℕ :=
  Finset.filter (fun d => d < 10) (Finset.range (n + 1))

/-- The theorem statement -/
theorem max_sum_of_four_digit_integers (a c : ℕ) :
  is_four_digit a ∧ is_four_digit c ∧
  (digits a ∪ digits c = Finset.range 10) →
  a + c ≤ 18395 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_four_digit_integers_l3687_368721


namespace NUMINAMATH_CALUDE_time_to_paint_remaining_rooms_l3687_368702

/-- Given a painting job with the following conditions:
  - There are 10 rooms in total to be painted
  - Each room takes 8 hours to paint
  - 8 rooms have already been painted
This theorem proves that it will take 16 hours to paint the remaining rooms. -/
theorem time_to_paint_remaining_rooms :
  let total_rooms : ℕ := 10
  let painted_rooms : ℕ := 8
  let time_per_room : ℕ := 8
  let remaining_rooms := total_rooms - painted_rooms
  let time_for_remaining := remaining_rooms * time_per_room
  time_for_remaining = 16 := by sorry

end NUMINAMATH_CALUDE_time_to_paint_remaining_rooms_l3687_368702


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3687_368700

theorem complex_equation_solution (z : ℂ) : (1 + 2*I)*z = 5 → z = 1 - 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3687_368700


namespace NUMINAMATH_CALUDE_sector_angle_measure_l3687_368766

theorem sector_angle_measure (r : ℝ) (α : ℝ) 
  (h1 : α * r = 2)  -- arc length = 2
  (h2 : (1/2) * α * r^2 = 2)  -- area = 2
  : α = 1 := by
sorry

end NUMINAMATH_CALUDE_sector_angle_measure_l3687_368766


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3687_368785

-- Define sets A and B
def A : Set ℝ := {x : ℝ | x < 1}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3687_368785


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l3687_368754

theorem pizza_toppings_combinations :
  Nat.choose 9 3 = 84 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l3687_368754


namespace NUMINAMATH_CALUDE_race_result_l3687_368733

/-- Represents the state of the race between Alex and Max -/
structure RaceState where
  alex_lead : Int
  distance_covered : Int

/-- Calculates the remaining distance for Max to catch up to Alex -/
def remaining_distance (total_length : Int) (final_state : RaceState) : Int :=
  total_length - final_state.distance_covered - final_state.alex_lead

/-- Updates the race state after a change in lead -/
def update_state (state : RaceState) (lead_change : Int) : RaceState :=
  { alex_lead := state.alex_lead + lead_change,
    distance_covered := state.distance_covered }

theorem race_result (total_length : Int) (initial_even : Int) (alex_lead1 : Int) 
                     (max_lead : Int) (alex_lead2 : Int) : 
  total_length = 5000 →
  initial_even = 200 →
  alex_lead1 = 300 →
  max_lead = 170 →
  alex_lead2 = 440 →
  let initial_state : RaceState := { alex_lead := 0, distance_covered := initial_even }
  let state1 := update_state initial_state alex_lead1
  let state2 := update_state state1 (-max_lead)
  let final_state := update_state state2 alex_lead2
  remaining_distance total_length final_state = 4430 := by
  sorry

#check race_result

end NUMINAMATH_CALUDE_race_result_l3687_368733


namespace NUMINAMATH_CALUDE_shoe_price_problem_l3687_368783

theorem shoe_price_problem (first_pair_price : ℝ) (total_paid : ℝ) :
  first_pair_price = 40 →
  total_paid = 60 →
  ∃ (second_pair_price : ℝ),
    second_pair_price ≥ first_pair_price ∧
    total_paid = (3/4) * (first_pair_price + (1/2) * second_pair_price) ∧
    second_pair_price = 80 :=
by
  sorry

#check shoe_price_problem

end NUMINAMATH_CALUDE_shoe_price_problem_l3687_368783


namespace NUMINAMATH_CALUDE_gondor_wednesday_laptops_l3687_368799

/-- Represents the earnings and repair data for Gondor --/
structure RepairData where
  phone_repair_fee : ℕ
  laptop_repair_fee : ℕ
  monday_phones : ℕ
  tuesday_phones : ℕ
  thursday_laptops : ℕ
  total_earnings : ℕ

/-- Calculates the number of laptops repaired on Wednesday --/
def laptops_repaired_wednesday (data : RepairData) : ℕ :=
  let phone_earnings := data.phone_repair_fee * (data.monday_phones + data.tuesday_phones)
  let thursday_laptop_earnings := data.laptop_repair_fee * data.thursday_laptops
  let wednesday_laptop_earnings := data.total_earnings - phone_earnings - thursday_laptop_earnings
  wednesday_laptop_earnings / data.laptop_repair_fee

/-- Theorem stating that Gondor repaired 2 laptops on Wednesday --/
theorem gondor_wednesday_laptops :
  laptops_repaired_wednesday {
    phone_repair_fee := 10,
    laptop_repair_fee := 20,
    monday_phones := 3,
    tuesday_phones := 5,
    thursday_laptops := 4,
    total_earnings := 200
  } = 2 := by
  sorry

end NUMINAMATH_CALUDE_gondor_wednesday_laptops_l3687_368799


namespace NUMINAMATH_CALUDE_blackboard_numbers_l3687_368735

def blackboard_rule (a b : ℕ) : ℕ := a * b + a + b

def is_generable (n : ℕ) : Prop :=
  ∃ k m : ℕ, n = 2^k * 3^m - 1

theorem blackboard_numbers :
  (is_generable 13121) ∧ (¬ is_generable 12131) := by sorry

end NUMINAMATH_CALUDE_blackboard_numbers_l3687_368735


namespace NUMINAMATH_CALUDE_max_M_inequality_l3687_368761

theorem max_M_inequality (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  (∃ (M : ℝ), ∀ (a b c : ℝ), a ≥ 0 → b ≥ 0 → c ≥ 0 → 
    a^3 + b^3 + c^3 - 3*a*b*c ≥ M*(a-b)*(b-c)*(c-a)) ↔ 
  (M ≤ Real.sqrt (9 + 6 * Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_max_M_inequality_l3687_368761


namespace NUMINAMATH_CALUDE_stratified_sampling_group_D_l3687_368748

/-- Represents the number of districts in each group -/
structure GroupSizes :=
  (A : ℕ)
  (B : ℕ)
  (C : ℕ)
  (D : ℕ)

/-- Calculates the total number of districts -/
def total_districts (g : GroupSizes) : ℕ := g.A + g.B + g.C + g.D

/-- Calculates the number of districts to be selected from a group in stratified sampling -/
def stratified_sample (group_size : ℕ) (total : ℕ) (sample_size : ℕ) : ℚ :=
  (group_size : ℚ) / (total : ℚ) * (sample_size : ℚ)

theorem stratified_sampling_group_D :
  let groups : GroupSizes := ⟨4, 10, 16, 8⟩
  let total := total_districts groups
  let sample_size := 9
  stratified_sample groups.D total sample_size = 2 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_group_D_l3687_368748


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3687_368775

/-- The function f(x) = x³ + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a

/-- f is monotonically increasing on ℝ -/
def is_monotone_increasing (a : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f a x < f a y

/-- The statement "if a > 1, then f(x) = x³ + a is monotonically increasing on ℝ" 
    is a sufficient but not necessary condition -/
theorem sufficient_not_necessary : 
  (∀ a : ℝ, a > 1 → is_monotone_increasing a) ∧ 
  (∃ a : ℝ, a ≤ 1 ∧ is_monotone_increasing a) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3687_368775


namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l3687_368746

/-- Given two hyperbolas with equations x^2/16 - y^2/25 = 1 and y^2/50 - x^2/M = 1,
    if they have the same asymptotes, then M = 32 -/
theorem hyperbolas_same_asymptotes (M : ℝ) :
  (∀ x y : ℝ, x^2/16 - y^2/25 = 1 ↔ y^2/50 - x^2/M = 1) →
  M = 32 := by
  sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l3687_368746


namespace NUMINAMATH_CALUDE_parabola_properties_l3687_368720

/-- A parabola with the given properties -/
def Parabola : Set (ℝ × ℝ) :=
  {(x, y) | y^2 = 8*x}

theorem parabola_properties :
  -- The parabola is symmetric about the x-axis
  (∀ x y, (x, y) ∈ Parabola ↔ (x, -y) ∈ Parabola) ∧
  -- The vertex of the parabola is at the origin
  (0, 0) ∈ Parabola ∧
  -- The parabola passes through point (2, 4)
  (2, 4) ∈ Parabola :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l3687_368720


namespace NUMINAMATH_CALUDE_simplify_expression_l3687_368791

theorem simplify_expression (x : ℝ) (h : x < 0) :
  (2 * abs x + (x^6)^(1/6) + (x^5)^(1/5)) / x = -2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3687_368791


namespace NUMINAMATH_CALUDE_cookout_buns_per_pack_alex_cookout_buns_per_pack_l3687_368755

/-- Calculates the number of buns in each pack given the cookout conditions -/
theorem cookout_buns_per_pack (total_guests : ℕ) (burgers_per_guest : ℕ) 
  (non_meat_guests : ℕ) (non_bread_guests : ℕ) (bun_packs : ℕ) : ℕ :=
  let guests_eating_meat := total_guests - non_meat_guests
  let guests_eating_bread := guests_eating_meat - non_bread_guests
  let total_buns_needed := guests_eating_bread * burgers_per_guest
  total_buns_needed / bun_packs

/-- Proves that the number of buns in each pack for Alex's cookout is 8 -/
theorem alex_cookout_buns_per_pack : 
  cookout_buns_per_pack 10 3 1 1 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cookout_buns_per_pack_alex_cookout_buns_per_pack_l3687_368755


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l3687_368707

/-- Two planar vectors are perpendicular if and only if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- The theorem states that for given planar vectors a = (-6, 2) and b = (3, m),
    if they are perpendicular, then m = 9 -/
theorem perpendicular_vectors_m_value :
  let a : ℝ × ℝ := (-6, 2)
  let b : ℝ × ℝ := (3, m)
  perpendicular a b → m = 9 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l3687_368707


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3687_368769

theorem purely_imaginary_complex_number (m : ℝ) :
  let z : ℂ := (m - 3 * Complex.I) / (2 + Complex.I)
  (∃ (y : ℝ), z = Complex.I * y) → m = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3687_368769


namespace NUMINAMATH_CALUDE_oldest_sibling_age_l3687_368725

/-- Represents the ages and relationships in Kay's family --/
structure KayFamily where
  kay_age : ℕ
  num_siblings : ℕ
  youngest_sibling_age : ℕ
  oldest_sibling_age : ℕ

/-- The conditions given in the problem --/
def kay_family_conditions (f : KayFamily) : Prop :=
  f.kay_age = 32 ∧
  f.num_siblings = 14 ∧
  f.youngest_sibling_age = f.kay_age / 2 - 5 ∧
  f.oldest_sibling_age = 4 * f.youngest_sibling_age

/-- Theorem stating that the oldest sibling's age is 44 given the conditions --/
theorem oldest_sibling_age (f : KayFamily) 
  (h : kay_family_conditions f) : f.oldest_sibling_age = 44 := by
  sorry


end NUMINAMATH_CALUDE_oldest_sibling_age_l3687_368725


namespace NUMINAMATH_CALUDE_correct_monthly_repayment_l3687_368777

/-- Calculates the monthly repayment amount for a loan -/
def calculate_monthly_repayment (loan_amount : ℝ) (monthly_interest_rate : ℝ) (loan_term_months : ℕ) : ℝ :=
  sorry

/-- Theorem stating the correct monthly repayment amount -/
theorem correct_monthly_repayment :
  let loan_amount : ℝ := 500000
  let monthly_interest_rate : ℝ := 0.005
  let loan_term_months : ℕ := 360
  abs (calculate_monthly_repayment loan_amount monthly_interest_rate loan_term_months - 2997.75) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_correct_monthly_repayment_l3687_368777


namespace NUMINAMATH_CALUDE_no_solution_double_inequality_l3687_368767

theorem no_solution_double_inequality :
  ¬ ∃ y : ℝ, (3 * y^2 - 4 * y - 5 < (y + 1)^2) ∧ ((y + 1)^2 < 4 * y^2 - y - 1) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_double_inequality_l3687_368767


namespace NUMINAMATH_CALUDE_days_with_parrot_l3687_368734

-- Define the given conditions
def total_phrases : ℕ := 17
def phrases_per_week : ℕ := 2
def initial_phrases : ℕ := 3
def days_per_week : ℕ := 7

-- Define the theorem
theorem days_with_parrot : 
  (total_phrases - initial_phrases) / phrases_per_week * days_per_week = 49 := by
  sorry

end NUMINAMATH_CALUDE_days_with_parrot_l3687_368734


namespace NUMINAMATH_CALUDE_josies_remaining_money_l3687_368744

/-- Given an initial amount of money and the costs of items,
    calculate the remaining amount after purchasing the items. -/
def remaining_money (initial_amount : ℕ) (item1_cost : ℕ) (item1_quantity : ℕ) (item2_cost : ℕ) : ℕ :=
  initial_amount - (item1_cost * item1_quantity + item2_cost)

/-- Prove that given an initial amount of $50, after spending $9 each on two items
    and $25 on another item, the remaining amount is $7. -/
theorem josies_remaining_money :
  remaining_money 50 9 2 25 = 7 := by
  sorry

end NUMINAMATH_CALUDE_josies_remaining_money_l3687_368744


namespace NUMINAMATH_CALUDE_quadratic_properties_l3687_368743

-- Define the quadratic function
def quadratic (a b x : ℝ) : ℝ := a * x^2 - b * x

-- State the theorem
theorem quadratic_properties
  (a b m n : ℝ)
  (h_a : a ≠ 0)
  (h_point : quadratic a b m = 2)
  (h_range : ∀ x, quadratic a b x ≥ -2/3 → x ≤ n - 1 ∨ x ≥ -3 - n) :
  (∃ x, ∀ y, quadratic a b y = quadratic a b x → y = x ∨ y = -4 - x) ∧
  (quadratic a b 1 = 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l3687_368743


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3687_368708

theorem imaginary_part_of_z (z : ℂ) : z - Complex.I = (4 - 2 * Complex.I) / (1 + 2 * Complex.I) → z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3687_368708


namespace NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l3687_368740

def k : ℕ := 2015^2 + 2^2015

theorem units_digit_of_k_squared_plus_two_to_k (k : ℕ) : k = 2015^2 + 2^2015 → (k^2 + 2^k) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l3687_368740


namespace NUMINAMATH_CALUDE_coffee_stock_percentage_l3687_368731

theorem coffee_stock_percentage (initial_stock : ℝ) (initial_decaf_percent : ℝ)
  (additional_stock : ℝ) (additional_decaf_percent : ℝ)
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 30)
  (h3 : additional_stock = 100)
  (h4 : additional_decaf_percent = 60) :
  let total_stock := initial_stock + additional_stock
  let total_decaf := (initial_stock * initial_decaf_percent / 100) +
                     (additional_stock * additional_decaf_percent / 100)
  total_decaf / total_stock * 100 = 36 := by
sorry

end NUMINAMATH_CALUDE_coffee_stock_percentage_l3687_368731


namespace NUMINAMATH_CALUDE_amusement_park_spending_l3687_368730

theorem amusement_park_spending (admission_cost food_cost total_cost : ℕ) : 
  food_cost = admission_cost - 13 →
  total_cost = admission_cost + food_cost →
  total_cost = 77 →
  admission_cost = 45 := by
sorry

end NUMINAMATH_CALUDE_amusement_park_spending_l3687_368730


namespace NUMINAMATH_CALUDE_system_solution_l3687_368713

theorem system_solution (x y z : ℚ) : 
  (x * y = 5 * (x + y) ∧ 
   x * z = 4 * (x + z) ∧ 
   y * z = 2 * (y + z)) → 
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ 
   (x = -40 ∧ y = 40/9 ∧ z = 40/11)) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3687_368713


namespace NUMINAMATH_CALUDE_harriet_miles_run_l3687_368736

/-- Proves that given four runners who ran a combined total of 195 miles,
    with one runner running 51 miles and the other three runners running equal distances,
    each of the other three runners ran 48 miles. -/
theorem harriet_miles_run (total_miles : ℕ) (katarina_miles : ℕ) (other_runners : ℕ) :
  total_miles = 195 →
  katarina_miles = 51 →
  other_runners = 3 →
  ∃ (harriet_miles : ℕ),
    harriet_miles * other_runners = total_miles - katarina_miles ∧
    harriet_miles = 48 := by
  sorry

end NUMINAMATH_CALUDE_harriet_miles_run_l3687_368736


namespace NUMINAMATH_CALUDE_min_omega_l3687_368784

theorem min_omega (f : ℝ → ℝ) (ω : ℝ) :
  (∀ x, f x = 2 * Real.sin (ω * x)) →
  ω > 0 →
  (∀ x ∈ Set.Icc (-π/3) (π/4), f x ≥ -2) →
  (∃ x ∈ Set.Icc (-π/3) (π/4), f x = -2) →
  ω ≥ 3/2 ∧ ∀ ω' > 0, (∀ x ∈ Set.Icc (-π/3) (π/4), 2 * Real.sin (ω' * x) ≥ -2) →
    (∃ x ∈ Set.Icc (-π/3) (π/4), 2 * Real.sin (ω' * x) = -2) → ω' ≥ 3/2 :=
by sorry

end NUMINAMATH_CALUDE_min_omega_l3687_368784


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3687_368770

theorem quadratic_real_roots_condition 
  (a b c : ℝ) : 
  (∃ x : ℝ, a * x^2 + b * x + c = 0) ↔ (a ≠ 0 ∧ b^2 - 4*a*c ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3687_368770


namespace NUMINAMATH_CALUDE_cricket_time_calculation_l3687_368757

/-- The total time Sean and Indira played cricket together -/
def total_cricket_time (sean_daily_time : ℕ) (sean_days : ℕ) (indira_time : ℕ) : ℕ :=
  sean_daily_time * sean_days + indira_time

/-- Theorem stating the total time Sean and Indira played cricket -/
theorem cricket_time_calculation :
  total_cricket_time 50 14 812 = 1512 := by
  sorry

end NUMINAMATH_CALUDE_cricket_time_calculation_l3687_368757


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l3687_368796

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem geometric_sequence_increasing_condition
  (a : ℕ → ℝ) (h_geometric : is_geometric_sequence a) :
  (is_increasing_sequence a → a 1 < a 2 ∧ a 2 < a 3) ∧
  ¬(a 1 < a 2 ∧ a 2 < a 3 → is_increasing_sequence a) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l3687_368796


namespace NUMINAMATH_CALUDE_vector_properties_l3687_368712

def e₁ : ℝ × ℝ := (1, 0)
def e₂ : ℝ × ℝ := (0, 1)
def a : ℝ × ℝ := (3 * e₁.1 - 2 * e₂.1, 3 * e₁.2 - 2 * e₂.2)
def b : ℝ × ℝ := (4 * e₁.1 + e₂.1, 4 * e₁.2 + e₂.2)

theorem vector_properties :
  (a.1 * b.1 + a.2 * b.2 = 10) ∧
  ((a.1 + b.1)^2 + (a.2 + b.2)^2 = 50) ∧
  ((a.1 * b.1 + a.2 * b.2)^2 = 100 * ((a.1^2 + a.2^2) * (b.1^2 + b.2^2)) / 221) := by
  sorry

end NUMINAMATH_CALUDE_vector_properties_l3687_368712


namespace NUMINAMATH_CALUDE_nested_fourth_root_solution_l3687_368704

/-- The positive solution to the nested fourth root equation --/
noncomputable def x : ℝ := 3.1412

/-- The left-hand side of the equation --/
noncomputable def lhs (x : ℝ) : ℝ := Real.sqrt (x + Real.sqrt (x + Real.sqrt (x + Real.sqrt x)))

/-- The right-hand side of the equation --/
noncomputable def rhs (x : ℝ) : ℝ := Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x)))

/-- Theorem stating that x is the positive solution to the equation --/
theorem nested_fourth_root_solution :
  lhs x = rhs x ∧ x > 0 := by sorry

end NUMINAMATH_CALUDE_nested_fourth_root_solution_l3687_368704


namespace NUMINAMATH_CALUDE_quadratic_coefficients_l3687_368760

/-- A quadratic function with vertex (-2, 5) passing through (0, 3) -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_coefficients :
  ∃ (a b c : ℝ),
    (∀ x, f a b c x = a * x^2 + b * x + c) ∧
    (f a b c (-2) = 5) ∧
    (∀ x, f a b c (x) = f a b c (-x - 4)) ∧
    (f a b c 0 = 3) ∧
    (a = -1/2 ∧ b = -2 ∧ c = 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_l3687_368760


namespace NUMINAMATH_CALUDE_square_area_error_l3687_368732

theorem square_area_error (s : ℝ) (h : s > 0) :
  let measured_side := s * (1 + 0.02)
  let actual_area := s^2
  let calculated_area := measured_side^2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.0404 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l3687_368732


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3687_368776

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 16) = 12 → x = 128 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3687_368776


namespace NUMINAMATH_CALUDE_abs_neg_six_l3687_368762

theorem abs_neg_six : |(-6 : ℝ)| = 6 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_six_l3687_368762


namespace NUMINAMATH_CALUDE_shaded_area_is_four_thirds_l3687_368779

/-- Rectangle with specific dimensions and lines forming a shaded region --/
structure ShadedRectangle where
  J : ℝ × ℝ
  K : ℝ × ℝ
  L : ℝ × ℝ
  M : ℝ × ℝ
  h_rectangle : J.1 = 0 ∧ J.2 = 0 ∧ K.1 = 4 ∧ K.2 = 0 ∧ L.1 = 4 ∧ L.2 = 5 ∧ M.1 = 0 ∧ M.2 = 5
  h_mj : M.2 - J.2 = 2
  h_jk : K.1 - J.1 = 1
  h_kl : L.2 - K.2 = 1
  h_lm : M.1 - L.1 = 1

/-- The area of the shaded region in the rectangle --/
def shadedArea (r : ShadedRectangle) : ℝ := sorry

/-- Theorem stating that the shaded area is 4/3 --/
theorem shaded_area_is_four_thirds (r : ShadedRectangle) : shadedArea r = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_four_thirds_l3687_368779


namespace NUMINAMATH_CALUDE_correct_average_l3687_368716

theorem correct_average (n : ℕ) (initial_avg wrong_num correct_num : ℚ) :
  n = 10 →
  initial_avg = 15 →
  wrong_num = 26 →
  correct_num = 36 →
  (n : ℚ) * initial_avg + (correct_num - wrong_num) = n * 16 :=
by sorry

end NUMINAMATH_CALUDE_correct_average_l3687_368716


namespace NUMINAMATH_CALUDE_chocolate_difference_l3687_368718

theorem chocolate_difference (robert_chocolates nickel_chocolates : ℕ) 
  (h1 : robert_chocolates = 7)
  (h2 : nickel_chocolates = 3) : 
  robert_chocolates - nickel_chocolates = 4 := by
sorry

end NUMINAMATH_CALUDE_chocolate_difference_l3687_368718


namespace NUMINAMATH_CALUDE_megan_pop_albums_l3687_368705

/-- The number of songs on each album -/
def songs_per_album : ℕ := 7

/-- The number of country albums bought -/
def country_albums : ℕ := 2

/-- The total number of songs bought -/
def total_songs : ℕ := 70

/-- The number of pop albums bought -/
def pop_albums : ℕ := (total_songs - country_albums * songs_per_album) / songs_per_album

theorem megan_pop_albums : pop_albums = 8 := by sorry

end NUMINAMATH_CALUDE_megan_pop_albums_l3687_368705


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3687_368772

theorem quadratic_inequality_solution (a : ℝ) (h : a > 0) :
  let f := fun x => a * x^2 - (a^2 + 1) * x + a
  (∀ x, f x > 0 ↔
    (a > 1 ∧ (x < 1/a ∨ x > a)) ∨
    (a = 1 ∧ x ≠ 1) ∨
    (0 < a ∧ a < 1 ∧ (x < a ∨ x > 1/a))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3687_368772


namespace NUMINAMATH_CALUDE_unique_solution_l3687_368701

/-- The functional equation that f must satisfy for all real x and y -/
def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 2 + f x * f y ≤ x * y + 2 * f (x + y + 1)

/-- The theorem stating that the only function satisfying the equation is f(x) = x + 2 -/
theorem unique_solution (f : ℝ → ℝ) (h : functional_equation f) : 
  ∀ x : ℝ, f x = x + 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3687_368701


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3687_368764

theorem inequality_system_solution :
  let S := {x : ℝ | (x - 1 < 2) ∧ (2*x + 3 ≥ x - 1)}
  S = {x : ℝ | -4 ≤ x ∧ x < 3} :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3687_368764


namespace NUMINAMATH_CALUDE_steve_wood_needed_l3687_368709

/-- The amount of wood Steve needs to buy for his bench project -/
def total_wood_needed (long_pieces : ℕ) (long_length : ℕ) (short_pieces : ℕ) (short_length : ℕ) : ℕ :=
  long_pieces * long_length + short_pieces * short_length

/-- Proof that Steve needs to buy 28 feet of wood -/
theorem steve_wood_needed : total_wood_needed 6 4 2 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_steve_wood_needed_l3687_368709


namespace NUMINAMATH_CALUDE_handshake_count_l3687_368715

/-- Represents the number of women in each age group -/
def women_per_group : ℕ := 5

/-- Represents the number of age groups -/
def num_groups : ℕ := 3

/-- Calculates the number of inter-group handshakes -/
def inter_group_handshakes : ℕ := women_per_group * women_per_group * (num_groups.choose 2)

/-- Calculates the number of intra-group handshakes for a single group -/
def intra_group_handshakes : ℕ := women_per_group.choose 2

/-- Calculates the total number of handshakes -/
def total_handshakes : ℕ := inter_group_handshakes + num_groups * intra_group_handshakes

/-- Theorem stating that the total number of handshakes is 105 -/
theorem handshake_count : total_handshakes = 105 := by
  sorry

end NUMINAMATH_CALUDE_handshake_count_l3687_368715


namespace NUMINAMATH_CALUDE_roots_in_intervals_l3687_368738

/-- The quadratic function f(x) = 7x^2 - (k+13)x + k^2 - k - 2 -/
def f (k : ℝ) (x : ℝ) : ℝ := 7 * x^2 - (k + 13) * x + k^2 - k - 2

/-- Theorem stating the range of k for which f(x) has roots in (0,1) and (1,2) -/
theorem roots_in_intervals (k : ℝ) : 
  (∃ x y, 0 < x ∧ x < 1 ∧ 1 < y ∧ y < 2 ∧ f k x = 0 ∧ f k y = 0) ↔ 
  ((3 < k ∧ k < 4) ∨ (-2 < k ∧ k < -1)) :=
sorry

end NUMINAMATH_CALUDE_roots_in_intervals_l3687_368738


namespace NUMINAMATH_CALUDE_data_set_average_l3687_368711

theorem data_set_average (a : ℝ) : 
  let data_set := [4, 2*a, 3-a, 5, 6]
  (data_set.sum / data_set.length = 4) → a = 2 := by
sorry

end NUMINAMATH_CALUDE_data_set_average_l3687_368711


namespace NUMINAMATH_CALUDE_sum_difference_3010_l3687_368786

/-- The sum of the first n odd counting numbers -/
def sum_odd (n : ℕ) : ℕ := n * (2 * n - 1)

/-- The sum of the first n even counting numbers -/
def sum_even (n : ℕ) : ℕ := n * (2 * n + 2)

/-- The difference between the sum of the first n even counting numbers
    and the sum of the first n odd counting numbers -/
def sum_difference (n : ℕ) : ℕ := sum_even n - sum_odd n

theorem sum_difference_3010 :
  sum_difference 3010 = 3010 := by sorry

end NUMINAMATH_CALUDE_sum_difference_3010_l3687_368786


namespace NUMINAMATH_CALUDE_matrix_inverse_proof_l3687_368759

theorem matrix_inverse_proof :
  let A : Matrix (Fin 4) (Fin 4) ℝ := !![2, -3, 0, 0;
                                       -4, 6, 0, 0;
                                        0, 0, 3, -5;
                                        0, 0, 1, -2]
  let M : Matrix (Fin 4) (Fin 4) ℝ := !![0, 0, 0.5, -0.5;
                                        0, 0, 0.5, -0.5;
                                        0, 0, 0.5, -0.5;
                                        0, 0, 0.5, -0.5]
  M * A = (1 : Matrix (Fin 4) (Fin 4) ℝ) := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_proof_l3687_368759


namespace NUMINAMATH_CALUDE_simplify_expression_l3687_368729

theorem simplify_expression (x : ℝ) : (3*x - 4)*(x + 9) - (x + 6)*(3*x - 2) = 7*x - 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3687_368729


namespace NUMINAMATH_CALUDE_fir_trees_count_l3687_368706

theorem fir_trees_count : ∃ (n : ℕ), 
  (n ≠ 15) ∧ 
  (n % 11 = 0) ∧ 
  (n < 25) ∧ 
  (n % 22 ≠ 0) ∧
  (n = 11) := by
  sorry

end NUMINAMATH_CALUDE_fir_trees_count_l3687_368706


namespace NUMINAMATH_CALUDE_min_max_abs_cubic_linear_l3687_368750

theorem min_max_abs_cubic_linear (y : ℝ) : 
  (∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧ |x^3 - x*y| ≥ 1) ∧
  (∃ (y₀ : ℝ), ∀ (x : ℝ), x ∈ Set.Icc 0 1 → |x^3 - x*y₀| ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_min_max_abs_cubic_linear_l3687_368750


namespace NUMINAMATH_CALUDE_carrots_in_second_bed_l3687_368771

/-- Given Kelly's carrot harvest information, prove the number of carrots in the second bed --/
theorem carrots_in_second_bed 
  (total_pounds : ℕ)
  (carrots_per_pound : ℕ)
  (first_bed : ℕ)
  (third_bed : ℕ)
  (h1 : total_pounds = 39)
  (h2 : carrots_per_pound = 6)
  (h3 : first_bed = 55)
  (h4 : third_bed = 78) :
  total_pounds * carrots_per_pound - first_bed - third_bed = 101 := by
  sorry

#check carrots_in_second_bed

end NUMINAMATH_CALUDE_carrots_in_second_bed_l3687_368771


namespace NUMINAMATH_CALUDE_height_comparison_l3687_368768

theorem height_comparison (height_a height_b : ℝ) (h : height_a = 0.75 * height_b) :
  (height_b - height_a) / height_a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_height_comparison_l3687_368768


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l3687_368789

/-- A quadratic radical is considered simpler if it cannot be further simplified to a non-radical form or a simpler radical form. -/
def IsSimplestQuadraticRadical (x : ℝ) : Prop :=
  ∀ y : ℝ, y ≠ x → (∃ z : ℝ, y = z ^ 2) → ¬(∃ w : ℝ, x = w ^ 2)

/-- The given options for quadratic radicals -/
def QuadraticRadicals (a b : ℝ) : Set ℝ :=
  {Real.sqrt 9, Real.sqrt (a^2 + b^2), Real.sqrt 0.7, Real.sqrt (a^3)}

/-- Theorem stating that √(a² + b²) is the simplest quadratic radical among the given options -/
theorem simplest_quadratic_radical (a b : ℝ) :
  ∀ x ∈ QuadraticRadicals a b, x = Real.sqrt (a^2 + b^2) ∨ ¬(IsSimplestQuadraticRadical x) :=
sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l3687_368789


namespace NUMINAMATH_CALUDE_function_inequality_l3687_368710

theorem function_inequality (a : ℝ) : 
  (∀ x : ℝ, x ≤ 1 → a + 2 * 2^x + 4^x < 0) → a < -8 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3687_368710


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l3687_368780

theorem angle_sum_is_pi_over_two (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : (Real.cos α / Real.sin β) + (Real.cos β / Real.sin α) = 2) : α + β = π/2 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l3687_368780


namespace NUMINAMATH_CALUDE_circledTimes_calculation_l3687_368749

-- Define the ⊗ operation
def circledTimes (a b : ℚ) : ℚ := (a + b) / (a - b)

-- State the theorem
theorem circledTimes_calculation :
  circledTimes (circledTimes 5 7) (circledTimes 4 2) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_circledTimes_calculation_l3687_368749
