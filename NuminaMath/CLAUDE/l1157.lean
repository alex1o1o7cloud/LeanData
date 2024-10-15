import Mathlib

namespace NUMINAMATH_CALUDE_area_circle_outside_square_l1157_115787

/-- The area inside a circle but outside a square with shared center -/
theorem area_circle_outside_square (r : ℝ) (d : ℝ) :
  r = 1 →  -- radius of circle is 1
  d = 2 →  -- diagonal of square is 2
  π - d^2 / 2 = π - 2 :=
by
  sorry

#check area_circle_outside_square

end NUMINAMATH_CALUDE_area_circle_outside_square_l1157_115787


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_inequality_l1157_115780

/-- A geometric sequence with positive terms and common ratio not equal to 1 -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (∀ n, a (n + 1) = q * a n) ∧ 
  q ≠ 1

theorem geometric_sequence_sum_inequality 
  (a : ℕ → ℝ) (q : ℝ) (h : GeometricSequence a q) : 
  a 1 + a 4 > a 2 + a 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_inequality_l1157_115780


namespace NUMINAMATH_CALUDE_gcd_lcm_problem_l1157_115762

theorem gcd_lcm_problem : 
  (Nat.gcd 60 75 * Nat.lcm 48 18 + 5 = 2165) := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_problem_l1157_115762


namespace NUMINAMATH_CALUDE_jump_height_ratio_l1157_115736

/-- The jump heights of four people and their ratios -/
theorem jump_height_ratio :
  let mark_height := 6
  let lisa_height := 2 * mark_height
  let jacob_height := 2 * lisa_height
  let james_height := 16
  (james_height : ℚ) / jacob_height = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_jump_height_ratio_l1157_115736


namespace NUMINAMATH_CALUDE_initial_ratio_is_four_to_one_l1157_115724

-- Define the total initial volume of the mixture
def total_volume : ℚ := 60

-- Define the volume of water added
def added_water : ℚ := 60

-- Define the ratio of milk to water after adding water
def new_ratio : ℚ := 1 / 2

-- Theorem statement
theorem initial_ratio_is_four_to_one :
  ∀ (initial_milk initial_water : ℚ),
    initial_milk + initial_water = total_volume →
    initial_milk / (initial_water + added_water) = new_ratio →
    initial_milk / initial_water = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_ratio_is_four_to_one_l1157_115724


namespace NUMINAMATH_CALUDE_people_owning_only_dogs_l1157_115783

theorem people_owning_only_dogs :
  let total_pet_owners : ℕ := 79
  let only_cats : ℕ := 10
  let cats_and_dogs : ℕ := 5
  let cats_dogs_snakes : ℕ := 3
  let total_snakes : ℕ := 49
  let only_dogs : ℕ := total_pet_owners - only_cats - cats_and_dogs - cats_dogs_snakes - (total_snakes - cats_dogs_snakes)
  only_dogs = 15 :=
by sorry

end NUMINAMATH_CALUDE_people_owning_only_dogs_l1157_115783


namespace NUMINAMATH_CALUDE_x_fourth_minus_reciprocal_l1157_115775

theorem x_fourth_minus_reciprocal (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/x^4 = 727 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_minus_reciprocal_l1157_115775


namespace NUMINAMATH_CALUDE_order_of_abc_l1157_115702

open Real

theorem order_of_abc (a b c : ℝ) (h1 : a = 24/7) (h2 : b * exp b = 7 * log 7) (h3 : 3^(c-1) = 7/exp 1) : a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_order_of_abc_l1157_115702


namespace NUMINAMATH_CALUDE_jack_vacation_budget_l1157_115758

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Represents the amount of money Jack has saved in base 8 -/
def jack_savings : ℕ := 3777

/-- Represents the cost of the airline ticket in base 10 -/
def ticket_cost : ℕ := 1200

/-- Calculates the remaining money after buying the ticket -/
def remaining_money : ℕ := base8_to_base10 jack_savings - ticket_cost

theorem jack_vacation_budget :
  remaining_money = 847 := by sorry

end NUMINAMATH_CALUDE_jack_vacation_budget_l1157_115758


namespace NUMINAMATH_CALUDE_larger_integer_proof_l1157_115723

theorem larger_integer_proof (x y : ℕ+) : 
  (y = x + 8) → (x * y = 272) → y = 21 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_proof_l1157_115723


namespace NUMINAMATH_CALUDE_unique_solution_l1157_115707

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) * f (x - y) = (f x ^ 3 + f y ^ 3 + 3 * x * y) - 3 * x^2 * y^2 * f x

/-- The theorem stating that there is a unique function satisfying the equation -/
theorem unique_solution :
  ∃! f : ℝ → ℝ, SatisfiesEquation f ∧ ∀ x : ℝ, f x = 0 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l1157_115707


namespace NUMINAMATH_CALUDE_triangle_semiperimeter_inequality_l1157_115754

/-- 
For any triangle with semiperimeter p, incircle radius r, and circumcircle radius R,
the inequality p ≥ (3/2) * sqrt(6 * R * r) holds.
-/
theorem triangle_semiperimeter_inequality (p r R : ℝ) 
  (hp : p > 0) (hr : r > 0) (hR : R > 0) : p ≥ (3/2) * Real.sqrt (6 * R * r) := by
  sorry

end NUMINAMATH_CALUDE_triangle_semiperimeter_inequality_l1157_115754


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1157_115738

theorem triangle_perimeter : ∀ x : ℝ,
  x^2 - 13*x + 40 = 0 →
  3 + 4 + x > x ∧ 3 + x > 4 ∧ 4 + x > 3 →
  3 + 4 + x = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1157_115738


namespace NUMINAMATH_CALUDE_tan_equality_proof_l1157_115748

theorem tan_equality_proof (n : ℤ) : 
  -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (345 * π / 180) → n = -15 := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_proof_l1157_115748


namespace NUMINAMATH_CALUDE_intersection_nonempty_l1157_115766

theorem intersection_nonempty (a : ℕ) (ha : a ≥ 2) :
  ∃ b : ℕ, 1 ≤ b ∧ b ≤ a ∧
  (∃ y : ℕ, (∃ x : ℕ, y = a^x) ∧ (∃ x : ℕ, y = (a+1)^x + b)) := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_l1157_115766


namespace NUMINAMATH_CALUDE_gary_earnings_l1157_115796

def total_flour : ℚ := 6
def flour_for_cakes : ℚ := 4
def flour_per_cake : ℚ := 1/2
def flour_for_cupcakes : ℚ := 2
def flour_per_cupcake : ℚ := 1/5
def price_per_cake : ℚ := 5/2
def price_per_cupcake : ℚ := 1

def num_cakes : ℚ := flour_for_cakes / flour_per_cake
def num_cupcakes : ℚ := flour_for_cupcakes / flour_per_cupcake

def earnings_from_cakes : ℚ := num_cakes * price_per_cake
def earnings_from_cupcakes : ℚ := num_cupcakes * price_per_cupcake

theorem gary_earnings :
  earnings_from_cakes + earnings_from_cupcakes = 30 :=
by sorry

end NUMINAMATH_CALUDE_gary_earnings_l1157_115796


namespace NUMINAMATH_CALUDE_zoo_feeding_theorem_l1157_115794

/-- Represents the number of animal pairs in the zoo -/
def num_pairs : ℕ := 6

/-- Represents the number of ways to feed the animals -/
def feeding_ways : ℕ := 14400

/-- Theorem stating the number of ways to feed the animals in the specified pattern -/
theorem zoo_feeding_theorem :
  (num_pairs = 6) →
  (∃ (male_choices female_choices : ℕ → ℕ),
    (∀ i, i ∈ Finset.range (num_pairs - 1) → male_choices i = num_pairs - 1 - i) ∧
    (∀ i, i ∈ Finset.range num_pairs → female_choices i = num_pairs - 1 - i) ∧
    (feeding_ways = (Finset.prod (Finset.range (num_pairs - 1)) male_choices) *
                    (Finset.prod (Finset.range num_pairs) female_choices))) :=
by sorry

#check zoo_feeding_theorem

end NUMINAMATH_CALUDE_zoo_feeding_theorem_l1157_115794


namespace NUMINAMATH_CALUDE_prob_both_white_l1157_115713

def box_A_white : ℕ := 3
def box_A_black : ℕ := 2
def box_B_white : ℕ := 2
def box_B_black : ℕ := 3

def prob_white_from_A : ℚ := box_A_white / (box_A_white + box_A_black)
def prob_white_from_B : ℚ := box_B_white / (box_B_white + box_B_black)

theorem prob_both_white :
  prob_white_from_A * prob_white_from_B = 6 / 25 := by
  sorry

end NUMINAMATH_CALUDE_prob_both_white_l1157_115713


namespace NUMINAMATH_CALUDE_isabel_finished_problems_l1157_115708

/-- Calculates the number of finished homework problems given the initial total,
    remaining pages, and problems per page. -/
def finished_problems (initial : ℕ) (remaining_pages : ℕ) (problems_per_page : ℕ) : ℕ :=
  initial - (remaining_pages * problems_per_page)

/-- Proves that Isabel finished 32 problems given the initial conditions. -/
theorem isabel_finished_problems :
  finished_problems 72 5 8 = 32 := by
  sorry


end NUMINAMATH_CALUDE_isabel_finished_problems_l1157_115708


namespace NUMINAMATH_CALUDE_coprime_20172019_l1157_115717

theorem coprime_20172019 : 
  (Nat.gcd 20172019 20172017 = 1) ∧ 
  (Nat.gcd 20172019 20172018 = 1) ∧ 
  (Nat.gcd 20172019 20172020 = 1) ∧ 
  (Nat.gcd 20172019 20172021 = 1) := by
  sorry

end NUMINAMATH_CALUDE_coprime_20172019_l1157_115717


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_distance_line_equations_l1157_115711

-- Define the lines
def l₁ (x y : ℝ) : Prop := y = 2 * x
def l₂ (x y : ℝ) : Prop := x + y = 6
def l₀ (x y : ℝ) : Prop := x - 2 * y = 0

-- Define the intersection point P
def P : ℝ × ℝ := (2, 4)

-- Theorem for part (1)
theorem perpendicular_line_equation :
  ∀ x y : ℝ, (x - P.1) = -2 * (y - P.2) ↔ 2 * x + y - 8 = 0 :=
sorry

-- Theorem for part (2)
theorem distance_line_equations :
  ∀ x y : ℝ, 
    (x = P.1 ∨ 3 * x - 4 * y + 10 = 0) ↔
    (∃ k : ℝ, y - P.2 = k * (x - P.1) ∧ 
      |k * P.1 - P.2| / Real.sqrt (k^2 + 1) = 2) ∨
    (x = P.1 ∧ |x| = 2) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_distance_line_equations_l1157_115711


namespace NUMINAMATH_CALUDE_sum_of_powers_and_mersenne_is_sum_of_squares_l1157_115765

/-- A Mersenne prime is a prime number of the form 2^k - 1 for some positive integer k. -/
def is_mersenne_prime (p : ℕ) : Prop :=
  Nat.Prime p ∧ ∃ k : ℕ, k > 0 ∧ p = 2^k - 1

/-- An integer n that is both the sum of two different powers of 2 
    and the sum of two different Mersenne primes is the sum of two different square numbers. -/
theorem sum_of_powers_and_mersenne_is_sum_of_squares (n : ℕ) 
  (h1 : ∃ a b : ℕ, a ≠ b ∧ n = 2^a + 2^b)
  (h2 : ∃ p q : ℕ, p ≠ q ∧ is_mersenne_prime p ∧ is_mersenne_prime q ∧ n = p + q) :
  ∃ x y : ℕ, x ≠ y ∧ n = x^2 + y^2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_powers_and_mersenne_is_sum_of_squares_l1157_115765


namespace NUMINAMATH_CALUDE_unique_number_l1157_115755

theorem unique_number : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  (n + 3) % 3 = 0 ∧ 
  (n + 4) % 4 = 0 ∧ 
  (n + 5) % 5 = 0 ∧ 
  n = 60 := by sorry

end NUMINAMATH_CALUDE_unique_number_l1157_115755


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1023_l1157_115705

theorem largest_prime_factor_of_1023 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 1023 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 1023 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1023_l1157_115705


namespace NUMINAMATH_CALUDE_problem_statement_l1157_115771

theorem problem_statement 
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (heq : a^2 + 2*b^2 + 3*c^2 = 4) : 
  (a = c → a*b ≤ Real.sqrt 2 / 2) ∧ 
  (a + 2*b + 3*c ≤ 2 * Real.sqrt 6) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1157_115771


namespace NUMINAMATH_CALUDE_cloth_sale_calculation_l1157_115795

theorem cloth_sale_calculation (total_selling_price : ℝ) (profit_per_meter : ℝ) (cost_price_per_meter : ℝ)
  (h1 : total_selling_price = 9890)
  (h2 : profit_per_meter = 24)
  (h3 : cost_price_per_meter = 83.5) :
  (total_selling_price / (cost_price_per_meter + profit_per_meter)) = 92 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_calculation_l1157_115795


namespace NUMINAMATH_CALUDE_right_triangle_area_l1157_115740

theorem right_triangle_area (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 13) (h3 : a = 5) : (1/2) * a * b = 30 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1157_115740


namespace NUMINAMATH_CALUDE_marilyn_shared_bottlecaps_l1157_115734

/-- 
Given that Marilyn starts with 51 bottle caps and ends up with 15 bottle caps,
prove that she shared 36 bottle caps with Nancy.
-/
theorem marilyn_shared_bottlecaps : 
  let initial_caps : ℕ := 51
  let remaining_caps : ℕ := 15
  let shared_caps : ℕ := initial_caps - remaining_caps
  shared_caps = 36 := by sorry

end NUMINAMATH_CALUDE_marilyn_shared_bottlecaps_l1157_115734


namespace NUMINAMATH_CALUDE_equation_satisfies_condition_l1157_115772

theorem equation_satisfies_condition (x y z : ℤ) : 
  x = z - 2 ∧ y = x + 1 → x * (x - y) + y * (y - z) + z * (z - x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_satisfies_condition_l1157_115772


namespace NUMINAMATH_CALUDE_cat_grooming_time_l1157_115746

/-- Calculates the total grooming time for a cat given specific grooming tasks and the cat's characteristics. -/
theorem cat_grooming_time :
  let clip_time_per_claw : ℕ := 10
  let clean_time_per_ear : ℕ := 90
  let shampoo_time_minutes : ℕ := 5
  let claws_per_foot : ℕ := 4
  let feet : ℕ := 4
  let ears : ℕ := 2
  clip_time_per_claw * claws_per_foot * feet + 
  clean_time_per_ear * ears + 
  shampoo_time_minutes * 60 = 640 := by
sorry


end NUMINAMATH_CALUDE_cat_grooming_time_l1157_115746


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l1157_115773

theorem inequality_system_solution_set :
  {x : ℝ | x - 1 < 0 ∧ x + 1 > 0} = {x : ℝ | -1 < x ∧ x < 1} :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l1157_115773


namespace NUMINAMATH_CALUDE_clown_balloon_count_l1157_115744

/-- The number of balloons a clown has after a series of actions -/
def final_balloon_count (initial : ℕ) (additional : ℕ) (given_away : ℕ) : ℕ :=
  initial + additional - given_away

/-- Theorem stating that the clown has 149 balloons at the end -/
theorem clown_balloon_count :
  final_balloon_count 123 53 27 = 149 := by
  sorry

end NUMINAMATH_CALUDE_clown_balloon_count_l1157_115744


namespace NUMINAMATH_CALUDE_bridge_length_l1157_115756

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 156 →
  train_speed_kmh = 45 →
  crossing_time = 40 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 344 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l1157_115756


namespace NUMINAMATH_CALUDE_books_from_second_shop_l1157_115776

theorem books_from_second_shop
  (books_first_shop : ℕ)
  (cost_first_shop : ℕ)
  (cost_second_shop : ℕ)
  (average_price : ℕ)
  (h1 : books_first_shop = 42)
  (h2 : cost_first_shop = 520)
  (h3 : cost_second_shop = 248)
  (h4 : average_price = 12)
  : ∃ (books_second_shop : ℕ),
    (cost_first_shop + cost_second_shop) / (books_first_shop + books_second_shop) = average_price ∧
    books_second_shop = 22 := by
  sorry

#check books_from_second_shop

end NUMINAMATH_CALUDE_books_from_second_shop_l1157_115776


namespace NUMINAMATH_CALUDE_even_odd_property_l1157_115719

theorem even_odd_property (a b : ℤ) : 
  (Even (a - b) ∧ Odd (a + b + 1)) ∨ (Odd (a - b) ∧ Even (a + b + 1)) := by
sorry

end NUMINAMATH_CALUDE_even_odd_property_l1157_115719


namespace NUMINAMATH_CALUDE_speed_calculation_l1157_115741

/-- The speed of the first person traveling from A to B -/
def speed_person1 : ℝ := 70

/-- The speed of the second person traveling from B to A -/
def speed_person2 : ℝ := 80

/-- The total distance between A and B in km -/
def total_distance : ℝ := 600

/-- The time in hours it takes for the two people to meet -/
def meeting_time : ℝ := 4

theorem speed_calculation :
  speed_person1 * meeting_time + speed_person2 * meeting_time = total_distance ∧
  speed_person1 * meeting_time = total_distance - speed_person2 * meeting_time :=
by sorry

end NUMINAMATH_CALUDE_speed_calculation_l1157_115741


namespace NUMINAMATH_CALUDE_turquoise_color_perception_l1157_115799

theorem turquoise_color_perception (total : ℕ) (more_blue : ℕ) (both : ℕ) (neither : ℕ) :
  total = 150 →
  more_blue = 90 →
  both = 40 →
  neither = 20 →
  ∃ (more_green : ℕ), more_green = 80 ∧ 
    more_green + more_blue - both + neither = total :=
by sorry

end NUMINAMATH_CALUDE_turquoise_color_perception_l1157_115799


namespace NUMINAMATH_CALUDE_johnny_rate_is_four_l1157_115743

/-- The walking problem scenario -/
structure WalkingScenario where
  total_distance : ℝ
  matthew_rate : ℝ
  johnny_distance : ℝ
  matthew_head_start : ℝ

/-- Calculate Johnny's walking rate given a WalkingScenario -/
def calculate_johnny_rate (scenario : WalkingScenario) : ℝ :=
  sorry

/-- Theorem stating that Johnny's walking rate is 4 km/h given the specific scenario -/
theorem johnny_rate_is_four (scenario : WalkingScenario) 
  (h1 : scenario.total_distance = 45)
  (h2 : scenario.matthew_rate = 3)
  (h3 : scenario.johnny_distance = 24)
  (h4 : scenario.matthew_head_start = 1) :
  calculate_johnny_rate scenario = 4 := by
  sorry

end NUMINAMATH_CALUDE_johnny_rate_is_four_l1157_115743


namespace NUMINAMATH_CALUDE_paper_fold_crease_length_l1157_115733

/-- Given a rectangular paper of width 8 inches, when folded so that the bottom right corner 
    touches the left edge dividing it in a 1:2 ratio, the length of the crease L is equal to 
    16/3 csc θ, where θ is the angle between the crease and the bottom edge. -/
theorem paper_fold_crease_length (width : ℝ) (θ : ℝ) (L : ℝ) :
  width = 8 →
  0 < θ → θ < π / 2 →
  L = (16 / 3) * (1 / Real.sin θ) := by
  sorry

end NUMINAMATH_CALUDE_paper_fold_crease_length_l1157_115733


namespace NUMINAMATH_CALUDE_square_difference_l1157_115712

theorem square_difference (x y : ℝ) 
  (h1 : (x + y)^2 = 49) 
  (h2 : x * y = 6) : 
  (x - y)^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1157_115712


namespace NUMINAMATH_CALUDE_quadratic_equation_form_l1157_115747

/-- 
Given a quadratic equation ax^2 + bx + c = 0,
if a = 3 and c = 1, then the equation is equivalent to 3x^2 + 1 = 0.
-/
theorem quadratic_equation_form (a b c : ℝ) : 
  a = 3 → c = 1 → (∃ x, a * x^2 + b * x + c = 0) ↔ (∃ x, 3 * x^2 + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_form_l1157_115747


namespace NUMINAMATH_CALUDE_forty_percent_of_number_l1157_115727

theorem forty_percent_of_number (n : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * n = 16 → 
  (40/100 : ℝ) * n = 192 := by
sorry

end NUMINAMATH_CALUDE_forty_percent_of_number_l1157_115727


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_one_l1157_115701

variable (n : ℕ+)

/-- The sum of the coefficients of the terms in the expansion of (4-3x+2y)^n that do not contain y -/
def sum_of_coefficients (n : ℕ+) : ℝ :=
  (4 - 3)^(n : ℕ)

/-- Theorem stating that the sum of coefficients is always 1 -/
theorem sum_of_coefficients_is_one (n : ℕ+) : 
  sum_of_coefficients n = 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_one_l1157_115701


namespace NUMINAMATH_CALUDE_graphs_intersect_once_l1157_115786

/-- The value of b for which the graphs of y = bx^2 + 5x + 2 and y = -2x - 3 intersect at exactly one point -/
def b : ℚ := 49 / 20

/-- The quadratic function representing the first graph -/
def f (x : ℝ) : ℝ := b * x^2 + 5 * x + 2

/-- The linear function representing the second graph -/
def g (x : ℝ) : ℝ := -2 * x - 3

/-- Theorem stating that the graphs intersect at exactly one point when b = 49/20 -/
theorem graphs_intersect_once :
  ∃! x : ℝ, f x = g x :=
sorry

end NUMINAMATH_CALUDE_graphs_intersect_once_l1157_115786


namespace NUMINAMATH_CALUDE_units_digit_17_power_2024_l1157_115704

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The sequence of units digits for powers of a number -/
def unitsDigitSequence (base : ℕ) : ℕ → ℕ
  | 0 => unitsDigit base
  | n + 1 => unitsDigit (base * unitsDigitSequence base n)

theorem units_digit_17_power_2024 :
  unitsDigit (17^2024) = 1 :=
sorry

end NUMINAMATH_CALUDE_units_digit_17_power_2024_l1157_115704


namespace NUMINAMATH_CALUDE_parabola_intersection_points_l1157_115777

/-- The intersection points of two parabolas that also lie on a given line -/
theorem parabola_intersection_points (x y : ℝ) :
  (y = 3 * x^2 - 9 * x + 4) ∧ 
  (y = -x^2 + 3 * x + 6) ∧ 
  (y = x + 3) →
  ((x = (3 + Real.sqrt 11) / 2 ∧ y = (9 + Real.sqrt 11) / 2) ∨
   (x = (3 - Real.sqrt 11) / 2 ∧ y = (9 - Real.sqrt 11) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_points_l1157_115777


namespace NUMINAMATH_CALUDE_min_dials_for_lighting_l1157_115774

/-- A regular 12-sided polygon dial with numbers from 1 to 12 -/
structure Dial :=
  (numbers : Fin 12 → Fin 12)

/-- A stack of dials -/
def DialStack := List Dial

/-- The sum of numbers in a column of the dial stack -/
def columnSum (stack : DialStack) (column : Fin 12) : ℕ :=
  stack.foldr (λ dial acc => acc + dial.numbers column) 0

/-- Predicate for when the Christmas tree lights up -/
def lightsUp (stack : DialStack) : Prop :=
  ∀ i j : Fin 12, columnSum stack i % 12 = columnSum stack j % 12

/-- The theorem stating the minimum number of dials required -/
theorem min_dials_for_lighting : 
  ∀ n : ℕ, (∃ stack : DialStack, stack.length = n ∧ lightsUp stack) → n ≥ 12 :=
sorry

end NUMINAMATH_CALUDE_min_dials_for_lighting_l1157_115774


namespace NUMINAMATH_CALUDE_twenty_cows_twenty_days_l1157_115725

/-- The number of bags of husk eaten by a group of cows over a period of days -/
def bags_eaten (num_cows : ℕ) (num_days : ℕ) : ℚ :=
  (num_cows : ℚ) * (num_days : ℚ) * (1 / 20 : ℚ)

/-- Theorem stating that 20 cows eat 20 bags of husk in 20 days -/
theorem twenty_cows_twenty_days : bags_eaten 20 20 = 20 := by
  sorry

end NUMINAMATH_CALUDE_twenty_cows_twenty_days_l1157_115725


namespace NUMINAMATH_CALUDE_p_shape_point_count_l1157_115797

/-- Calculates the number of distinct points on a "P" shape derived from a square --/
def count_points_on_p_shape (side_length : ℕ) (point_interval : ℕ) : ℕ :=
  let points_per_side := side_length / point_interval + 1
  let total_points := points_per_side * 3
  total_points - 2

theorem p_shape_point_count :
  count_points_on_p_shape 10 1 = 31 := by
  sorry

#eval count_points_on_p_shape 10 1

end NUMINAMATH_CALUDE_p_shape_point_count_l1157_115797


namespace NUMINAMATH_CALUDE_bryans_deposit_l1157_115781

theorem bryans_deposit (mark_deposit : ℕ) (bryan_deposit : ℕ) : 
  mark_deposit = 88 →
  bryan_deposit = 5 * mark_deposit - 40 →
  bryan_deposit = 400 := by
sorry

end NUMINAMATH_CALUDE_bryans_deposit_l1157_115781


namespace NUMINAMATH_CALUDE_minimum_score_for_target_average_l1157_115721

def test_count : ℕ := 6
def max_score : ℕ := 100
def target_average : ℕ := 85
def scores : List ℕ := [82, 70, 88]

theorem minimum_score_for_target_average :
  ∃ (x y z : ℕ), 
    x ≤ max_score ∧ y ≤ max_score ∧ z ≤ max_score ∧
    (scores.sum + x + y + z) / test_count = target_average ∧
    (∀ w, w < 70 → (scores.sum + w + max_score + max_score) / test_count < target_average) := by
  sorry

end NUMINAMATH_CALUDE_minimum_score_for_target_average_l1157_115721


namespace NUMINAMATH_CALUDE_perpendicular_tangents_ratio_l1157_115722

theorem perpendicular_tangents_ratio (a b : ℝ) : 
  (∃ (x y : ℝ), a*x + b*y - 5 = 0 ∧ y = x^3) →  -- Line and curve equations
  (∃ (x y : ℝ), x = 1 ∧ y = 1 ∧ a*x + b*y - 5 = 0 ∧ y = x^3) →  -- Point P(1, 1) satisfies both equations
  (∀ (m₁ m₂ : ℝ), (m₁ * m₂ = -1) → 
    (m₁ = -a/b ∧ m₂ = 3 * 1^2)) →  -- Perpendicular tangent lines condition
  a/b = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_ratio_l1157_115722


namespace NUMINAMATH_CALUDE_box_of_books_l1157_115757

theorem box_of_books (box_weight : ℕ) (book_weight : ℕ) (h1 : box_weight = 42) (h2 : book_weight = 3) :
  box_weight / book_weight = 14 := by
  sorry

end NUMINAMATH_CALUDE_box_of_books_l1157_115757


namespace NUMINAMATH_CALUDE_f_range_l1157_115710

-- Define the function
def f (x : ℝ) := x^2 - 6*x + 7

-- State the theorem
theorem f_range :
  {y : ℝ | ∃ x ≥ 4, f x = y} = {y : ℝ | y ≥ -1} :=
by sorry

end NUMINAMATH_CALUDE_f_range_l1157_115710


namespace NUMINAMATH_CALUDE_shirt_sweater_cost_l1157_115752

/-- The total cost of a shirt and a sweater given their price relationship -/
theorem shirt_sweater_cost (shirt_price sweater_price total_cost : ℝ) : 
  shirt_price = 36.46 →
  shirt_price = sweater_price - 7.43 →
  total_cost = shirt_price + sweater_price →
  total_cost = 80.35 := by
sorry

end NUMINAMATH_CALUDE_shirt_sweater_cost_l1157_115752


namespace NUMINAMATH_CALUDE_paul_crayons_l1157_115760

/-- The number of crayons Paul had initially -/
def initial_crayons : ℕ := 253

/-- The number of crayons Paul lost or gave away -/
def lost_crayons : ℕ := 70

/-- The number of crayons Paul had left -/
def remaining_crayons : ℕ := initial_crayons - lost_crayons

theorem paul_crayons : remaining_crayons = 183 := by sorry

end NUMINAMATH_CALUDE_paul_crayons_l1157_115760


namespace NUMINAMATH_CALUDE_sales_volume_decrease_and_may_prediction_l1157_115770

/-- Represents the monthly sales volume decrease rate -/
def monthly_decrease_rate : ℝ := 0.05

/-- Calculates the sales volume after n months given an initial volume and monthly decrease rate -/
def sales_volume (initial_volume : ℝ) (n : ℕ) : ℝ :=
  initial_volume * (1 - monthly_decrease_rate) ^ n

theorem sales_volume_decrease_and_may_prediction
  (january_volume : ℝ)
  (march_volume : ℝ)
  (h1 : january_volume = 6000)
  (h2 : march_volume = 5400)
  (h3 : sales_volume january_volume 2 = march_volume)
  : monthly_decrease_rate = 0.05 ∧ sales_volume january_volume 4 > 4500 := by
  sorry

#eval sales_volume 6000 4

end NUMINAMATH_CALUDE_sales_volume_decrease_and_may_prediction_l1157_115770


namespace NUMINAMATH_CALUDE_ice_cream_to_after_lunch_ratio_l1157_115761

def initial_money : ℚ := 30
def lunch_cost : ℚ := 10
def remaining_money : ℚ := 15

def money_after_lunch : ℚ := initial_money - lunch_cost
def ice_cream_cost : ℚ := money_after_lunch - remaining_money

theorem ice_cream_to_after_lunch_ratio :
  ice_cream_cost / money_after_lunch = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_ice_cream_to_after_lunch_ratio_l1157_115761


namespace NUMINAMATH_CALUDE_smallest_ccd_is_227_l1157_115791

/-- Represents a two-digit number -/
def TwoDigitNumber (c d : ℕ) : Prop :=
  c ≠ 0 ∧ c ≤ 9 ∧ d ≤ 9

/-- Represents a three-digit number -/
def ThreeDigitNumber (c d : ℕ) : Prop :=
  TwoDigitNumber c d ∧ c * 100 + c * 10 + d ≥ 100

/-- The main theorem -/
theorem smallest_ccd_is_227 :
  ∃ (c d : ℕ),
    TwoDigitNumber c d ∧
    ThreeDigitNumber c d ∧
    c ≠ d ∧
    (c * 10 + d : ℚ) = (1 / 7) * (c * 100 + c * 10 + d) ∧
    c * 100 + c * 10 + d = 227 ∧
    ∀ (c' d' : ℕ),
      TwoDigitNumber c' d' →
      ThreeDigitNumber c' d' →
      c' ≠ d' →
      (c' * 10 + d' : ℚ) = (1 / 7) * (c' * 100 + c' * 10 + d') →
      c' * 100 + c' * 10 + d' ≥ 227 :=
by sorry

end NUMINAMATH_CALUDE_smallest_ccd_is_227_l1157_115791


namespace NUMINAMATH_CALUDE_find_n_l1157_115790

def vector_AB : Fin 2 → ℝ := ![2, 4]
def vector_BC (n : ℝ) : Fin 2 → ℝ := ![-2, 2*n]
def vector_AC : Fin 2 → ℝ := ![0, 2]

theorem find_n : ∃ n : ℝ, 
  (∀ i : Fin 2, vector_AB i + vector_BC n i = vector_AC i) ∧ n = -1 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l1157_115790


namespace NUMINAMATH_CALUDE_train_length_l1157_115729

/-- Proves the length of a train given its speed, time to pass a platform, and platform length -/
theorem train_length (train_speed : ℝ) (time_to_pass : ℝ) (platform_length : ℝ) :
  train_speed = 45 * (1000 / 3600) →
  time_to_pass = 39.2 →
  platform_length = 130 →
  train_speed * time_to_pass - platform_length = 360 := by
sorry

end NUMINAMATH_CALUDE_train_length_l1157_115729


namespace NUMINAMATH_CALUDE_horner_method_properties_l1157_115720

def f (x : ℝ) : ℝ := 12 + 35*x + 9*x^3 + 5*x^5 + 3*x^6

def horner_v3 (a : List ℝ) (x : ℝ) : ℝ :=
  match a with
  | [] => 0
  | a₀ :: as => List.foldl (fun acc a_i => acc * x + a_i) a₀ as

theorem horner_method_properties :
  let a := [3, 5, 0, 9, 0, 35, 12]
  let x := -1
  ∃ (multiplications additions : ℕ) (v3 : ℝ),
    multiplications = 6 ∧
    additions = 6 ∧
    v3 = horner_v3 (List.take 4 a) x ∧
    v3 = 11 ∧
    f x = horner_v3 a x :=
by sorry

end NUMINAMATH_CALUDE_horner_method_properties_l1157_115720


namespace NUMINAMATH_CALUDE_trig_expression_equals_five_fourths_l1157_115715

theorem trig_expression_equals_five_fourths :
  2 * (Real.cos (5 * π / 16))^6 + 2 * (Real.sin (11 * π / 16))^6 + (3 * Real.sqrt 2) / 8 = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_five_fourths_l1157_115715


namespace NUMINAMATH_CALUDE_sum_cube_inequality_l1157_115763

theorem sum_cube_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000/9 := by
  sorry

end NUMINAMATH_CALUDE_sum_cube_inequality_l1157_115763


namespace NUMINAMATH_CALUDE_unique_solution_l1157_115779

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def xyz_to_decimal (x y z : ℕ) : ℚ := (100 * x + 10 * y + z : ℚ) / 1000

theorem unique_solution (x y z : ℕ) :
  is_digit x ∧ is_digit y ∧ is_digit z →
  (1 : ℚ) / (x + y + z : ℚ) = xyz_to_decimal x y z →
  x = 1 ∧ y = 2 ∧ z = 5 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l1157_115779


namespace NUMINAMATH_CALUDE_hyperbola_sum_l1157_115706

-- Define the hyperbola
def Hyperbola (center focus vertex : ℝ × ℝ) : Prop :=
  let (h, k) := center
  let (_, f_y) := focus
  let (_, v_y) := vertex
  let a : ℝ := |k - v_y|
  let c : ℝ := |f_y - k|
  let b : ℝ := Real.sqrt (c^2 - a^2)
  ∀ x y : ℝ, ((y - k)^2 / a^2) - ((x - h)^2 / b^2) = 1

-- State the theorem
theorem hyperbola_sum (center focus vertex : ℝ × ℝ) 
  (h : Hyperbola center focus vertex) 
  (hc : center = (3, 1)) 
  (hf : focus = (3, 9)) 
  (hv : vertex = (3, -2)) : 
  let (h, k) := center
  let (_, f_y) := focus
  let (_, v_y) := vertex
  let a : ℝ := |k - v_y|
  let c : ℝ := |f_y - k|
  let b : ℝ := Real.sqrt (c^2 - a^2)
  h + k + a + b = 7 + Real.sqrt 55 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l1157_115706


namespace NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l1157_115726

theorem factorization_xy_squared_minus_x (x y : ℝ) : x * y^2 - x = x * (y - 1) * (y + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l1157_115726


namespace NUMINAMATH_CALUDE_solve_for_n_l1157_115718

variable (n : ℝ)

def f (x : ℝ) : ℝ := x^2 - 3*x + n
def g (x : ℝ) : ℝ := x^2 - 3*x + 5*n

theorem solve_for_n : 3 * f 3 = 2 * g 3 → n = 0 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_n_l1157_115718


namespace NUMINAMATH_CALUDE_flower_visitation_l1157_115728

theorem flower_visitation (total_flowers : ℕ) (num_bees : ℕ) (flowers_per_bee : ℕ)
  (h_total : total_flowers = 88)
  (h_bees : num_bees = 3)
  (h_flowers_per_bee : flowers_per_bee = 54)
  : ∃ (sweet bitter : ℕ), 
    sweet + bitter ≤ total_flowers ∧ 
    num_bees * flowers_per_bee = 3 * sweet + 2 * (total_flowers - sweet - bitter) + bitter ∧
    bitter = sweet + 14 := by
  sorry

end NUMINAMATH_CALUDE_flower_visitation_l1157_115728


namespace NUMINAMATH_CALUDE_meaningful_square_root_l1157_115750

theorem meaningful_square_root (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2023) ↔ x ≥ 2023 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_square_root_l1157_115750


namespace NUMINAMATH_CALUDE_luigi_pizza_count_l1157_115793

/-- The number of pizzas Luigi bought -/
def num_pizzas : ℕ := 4

/-- The total cost of pizzas in dollars -/
def total_cost : ℕ := 80

/-- The number of pieces each pizza is cut into -/
def pieces_per_pizza : ℕ := 5

/-- The cost of each piece of pizza in dollars -/
def cost_per_piece : ℕ := 4

/-- Theorem stating that the number of pizzas Luigi bought is 4 -/
theorem luigi_pizza_count :
  num_pizzas = 4 ∧
  total_cost = 80 ∧
  pieces_per_pizza = 5 ∧
  cost_per_piece = 4 ∧
  total_cost = num_pizzas * pieces_per_pizza * cost_per_piece :=
by sorry

end NUMINAMATH_CALUDE_luigi_pizza_count_l1157_115793


namespace NUMINAMATH_CALUDE_remaining_books_and_games_l1157_115778

/-- The number of remaining items to experience in a category -/
def remaining (total : ℕ) (experienced : ℕ) : ℕ := total - experienced

/-- The total number of remaining items to experience across categories -/
def total_remaining (remaining1 : ℕ) (remaining2 : ℕ) : ℕ := remaining1 + remaining2

/-- Proof that the number of remaining books and games to experience is 109 -/
theorem remaining_books_and_games :
  let total_books : ℕ := 150
  let total_games : ℕ := 50
  let books_read : ℕ := 74
  let games_played : ℕ := 17
  let remaining_books := remaining total_books books_read
  let remaining_games := remaining total_games games_played
  total_remaining remaining_books remaining_games = 109 := by
  sorry

end NUMINAMATH_CALUDE_remaining_books_and_games_l1157_115778


namespace NUMINAMATH_CALUDE_symmetry_theorem_l1157_115731

/-- The line about which the points are symmetrical -/
def symmetry_line (x y : ℝ) : Prop := x - y - 1 = 0

/-- Defines a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines symmetry between two points about a line -/
def symmetric_about_line (P Q : Point) : Prop :=
  -- The midpoint of PQ lies on the symmetry line
  symmetry_line ((P.x + Q.x) / 2) ((P.y + Q.y) / 2) ∧
  -- The slope of PQ is perpendicular to the slope of the symmetry line
  (Q.y - P.y) / (Q.x - P.x) = -1

/-- The theorem to be proved -/
theorem symmetry_theorem (a b : ℝ) :
  let P : Point := ⟨3, 4⟩
  let Q : Point := ⟨a, b⟩
  symmetric_about_line P Q → a = 5 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_theorem_l1157_115731


namespace NUMINAMATH_CALUDE_xyz_product_l1157_115788

theorem xyz_product (x y z : ℂ) 
  (eq1 : x * y + 6 * y = -24)
  (eq2 : y * z + 6 * z = -24)
  (eq3 : z * x + 6 * x = -24) :
  x * y * z = 144 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_l1157_115788


namespace NUMINAMATH_CALUDE_equation_solution_l1157_115798

theorem equation_solution (x : ℝ) : 
  (x / 5) / 3 = 15 / (x / 3) → x = 15 * Real.sqrt 3 ∨ x = -15 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1157_115798


namespace NUMINAMATH_CALUDE_day_after_53_from_friday_l1157_115789

/-- Days of the week -/
inductive DayOfWeek
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday
  | sunday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.monday => DayOfWeek.tuesday
  | DayOfWeek.tuesday => DayOfWeek.wednesday
  | DayOfWeek.wednesday => DayOfWeek.thursday
  | DayOfWeek.thursday => DayOfWeek.friday
  | DayOfWeek.friday => DayOfWeek.saturday
  | DayOfWeek.saturday => DayOfWeek.sunday
  | DayOfWeek.sunday => DayOfWeek.monday

/-- Function to get the day of the week after a given number of days -/
def dayAfter (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => start
  | Nat.succ m => nextDay (dayAfter start m)

/-- Theorem stating that 53 days after Friday is Tuesday -/
theorem day_after_53_from_friday :
  dayAfter DayOfWeek.friday 53 = DayOfWeek.tuesday := by
  sorry


end NUMINAMATH_CALUDE_day_after_53_from_friday_l1157_115789


namespace NUMINAMATH_CALUDE_perimeter_ratio_not_integer_l1157_115785

theorem perimeter_ratio_not_integer (a k l : ℕ+) (h : a ^ 2 = k * l) :
  ¬ ∃ (n : ℕ), (2 * (k + l) : ℚ) / (4 * a) = n := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ratio_not_integer_l1157_115785


namespace NUMINAMATH_CALUDE_parallel_planes_line_relations_l1157_115700

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the contained relation for lines in planes
variable (contained_in : Line → Plane → Prop)

-- Define the intersect relation for lines
variable (intersect : Line → Line → Prop)

-- Define the coplanar relation for lines
variable (coplanar : Line → Line → Prop)

-- Define the skew relation for lines
variable (skew : Line → Line → Prop)

-- Theorem statement
theorem parallel_planes_line_relations 
  (α β : Plane) (a b : Line)
  (h1 : parallel_planes α β)
  (h2 : contained_in a α)
  (h3 : contained_in b β) :
  (¬ intersect a b) ∧ (coplanar a b ∨ skew a b) :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_line_relations_l1157_115700


namespace NUMINAMATH_CALUDE_clownfish_display_count_l1157_115767

/-- Represents the number of fish in the aquarium -/
def total_fish : ℕ := 100

/-- Represents the number of blowfish that stay in their own tank -/
def blowfish_in_own_tank : ℕ := 26

/-- Calculates the number of clownfish in the display tank -/
def clownfish_in_display (total_fish : ℕ) (blowfish_in_own_tank : ℕ) : ℕ :=
  let total_per_species := total_fish / 2
  let blowfish_in_display := total_per_species - blowfish_in_own_tank
  let initial_clownfish_in_display := blowfish_in_display
  initial_clownfish_in_display - (initial_clownfish_in_display / 3)

theorem clownfish_display_count :
  clownfish_in_display total_fish blowfish_in_own_tank = 16 := by
  sorry

end NUMINAMATH_CALUDE_clownfish_display_count_l1157_115767


namespace NUMINAMATH_CALUDE_pencils_per_student_l1157_115769

/-- Represents the distribution of pencils to students -/
def pencil_distribution (total_pencils : ℕ) (max_students : ℕ) : ℕ :=
  total_pencils / max_students

/-- Theorem stating that given 910 pencils and 91 students, each student receives 10 pencils -/
theorem pencils_per_student :
  pencil_distribution 910 91 = 10 := by
  sorry

#check pencils_per_student

end NUMINAMATH_CALUDE_pencils_per_student_l1157_115769


namespace NUMINAMATH_CALUDE_part1_part2_l1157_115753

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x - a| + a

-- Part 1
theorem part1 (a : ℝ) : 
  (∀ x, f a x ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) → a = 1 := by sorry

-- Part 2
theorem part2 (m : ℝ) : 
  (∃ n : ℝ, f 1 n ≤ m - f 1 (-n)) → m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l1157_115753


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1157_115716

/-- Two planar vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given planar vectors a and b, if they are parallel, then x = -3/2 -/
theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (1, x)
  let b : ℝ × ℝ := (-2, 3)
  are_parallel a b → x = -3/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1157_115716


namespace NUMINAMATH_CALUDE_custom_mul_four_three_l1157_115751

/-- Custom multiplication operation -/
def customMul (a b : ℕ) : ℕ := a^2 + a * Nat.factorial b - b^2

/-- Theorem stating that 4 * 3 = 31 under the custom multiplication -/
theorem custom_mul_four_three : customMul 4 3 = 31 := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_four_three_l1157_115751


namespace NUMINAMATH_CALUDE_binary_110_equals_6_l1157_115732

def binary_to_decimal (b₂ b₁ b₀ : Nat) : Nat :=
  b₂ * 2^2 + b₁ * 2^1 + b₀ * 2^0

theorem binary_110_equals_6 : binary_to_decimal 1 1 0 = 6 := by
  sorry

end NUMINAMATH_CALUDE_binary_110_equals_6_l1157_115732


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1157_115749

theorem inequality_solution_set (x : ℝ) : 
  (1/2 - x) * (x - 1/3) > 0 ↔ 1/3 < x ∧ x < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1157_115749


namespace NUMINAMATH_CALUDE_square_plus_twice_a_equals_three_l1157_115742

theorem square_plus_twice_a_equals_three (a : ℝ) : 
  (∃ x : ℝ, x = -5 ∧ 2 * x + 8 = x / 5 - a) → a^2 + 2*a = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_twice_a_equals_three_l1157_115742


namespace NUMINAMATH_CALUDE_baking_time_l1157_115714

/-- 
Given:
- It takes 7 minutes to bake 1 pan of cookies
- The total time to bake 4 pans is 28 minutes

Prove that the time to bake 4 pans of cookies is 28 minutes.
-/
theorem baking_time (time_for_one_pan : ℕ) (total_time : ℕ) (num_pans : ℕ) :
  time_for_one_pan = 7 →
  total_time = 28 →
  num_pans = 4 →
  total_time = 28 := by
sorry

end NUMINAMATH_CALUDE_baking_time_l1157_115714


namespace NUMINAMATH_CALUDE_fraction_product_equals_one_fourteenth_l1157_115735

def product_fraction (n : ℕ) : ℚ := (n^2 - 1) / (n^2 + 1)

theorem fraction_product_equals_one_fourteenth :
  (product_fraction 2) * (product_fraction 3) * (product_fraction 4) * 
  (product_fraction 5) * (product_fraction 6) = 1 / 14 := by
sorry

end NUMINAMATH_CALUDE_fraction_product_equals_one_fourteenth_l1157_115735


namespace NUMINAMATH_CALUDE_mikes_ride_length_l1157_115768

/-- Proves that Mike's ride was 36 miles long given the taxi fare conditions -/
theorem mikes_ride_length :
  let mike_base_fare : ℚ := 2.5
  let mike_per_mile : ℚ := 0.25
  let annie_base_fare : ℚ := 2.5
  let annie_toll : ℚ := 5
  let annie_per_mile : ℚ := 0.25
  let annie_miles : ℚ := 16
  ∀ m : ℚ,
    mike_base_fare + mike_per_mile * m = 
    annie_base_fare + annie_toll + annie_per_mile * annie_miles →
    m = 36 :=
by
  sorry


end NUMINAMATH_CALUDE_mikes_ride_length_l1157_115768


namespace NUMINAMATH_CALUDE_x_13_plus_inv_x_13_l1157_115730

theorem x_13_plus_inv_x_13 (x : ℝ) (hx : x ≠ 0) :
  let y := x + 1/x
  x^13 + 1/x^13 = y^13 - 13*y^11 + 65*y^9 - 156*y^7 + 182*y^5 - 91*y^3 + 13*y :=
by
  sorry

end NUMINAMATH_CALUDE_x_13_plus_inv_x_13_l1157_115730


namespace NUMINAMATH_CALUDE_quadratic_function_value_bound_l1157_115745

theorem quadratic_function_value_bound (p q : ℝ) : 
  ¬(∀ x ∈ ({1, 2, 3} : Set ℝ), |x^2 + p*x + q| < (1/2 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_bound_l1157_115745


namespace NUMINAMATH_CALUDE_correct_number_misread_l1157_115759

theorem correct_number_misread (n : ℕ) (initial_avg correct_avg wrong_num : ℚ) : 
  n = 10 → 
  initial_avg = 15 → 
  correct_avg = 16 → 
  wrong_num = 26 → 
  ∃ (correct_num : ℚ), 
    (n : ℚ) * initial_avg - wrong_num + correct_num = (n : ℚ) * correct_avg ∧ 
    correct_num = 36 :=
by sorry

end NUMINAMATH_CALUDE_correct_number_misread_l1157_115759


namespace NUMINAMATH_CALUDE_optimal_strategy_l1157_115792

/-- Represents the clothing types --/
inductive ClothingType
| A
| B

/-- Represents the cost and selling price of each clothing type --/
def clothingInfo : ClothingType → (ℕ × ℕ)
| ClothingType.A => (80, 120)
| ClothingType.B => (60, 90)

/-- The total number of clothing items --/
def totalClothing : ℕ := 100

/-- The maximum total cost allowed --/
def maxTotalCost : ℕ := 7500

/-- The minimum number of type A clothing --/
def minTypeA : ℕ := 65

/-- The maximum number of type A clothing --/
def maxTypeA : ℕ := 75

/-- Calculates the total profit given the number of type A clothing and the discount --/
def totalProfit (x : ℕ) (a : ℚ) : ℚ :=
  (10 - a) * x + 3000

/-- Represents the optimal purchase strategy --/
structure OptimalStrategy where
  typeACount : ℕ
  typeBCount : ℕ

/-- Theorem stating the optimal purchase strategy based on the discount --/
theorem optimal_strategy (a : ℚ) (h1 : 0 < a) (h2 : a < 20) :
  (∃ (strategy : OptimalStrategy),
    (0 < a ∧ a < 10 → strategy.typeACount = maxTypeA ∧ strategy.typeBCount = totalClothing - maxTypeA) ∧
    (a = 10 → strategy.typeACount ≥ minTypeA ∧ strategy.typeACount ≤ maxTypeA) ∧
    (10 < a ∧ a < 20 → strategy.typeACount = minTypeA ∧ strategy.typeBCount = totalClothing - minTypeA) ∧
    (∀ (x : ℕ), minTypeA ≤ x → x ≤ maxTypeA → totalProfit strategy.typeACount a ≥ totalProfit x a)) :=
sorry

end NUMINAMATH_CALUDE_optimal_strategy_l1157_115792


namespace NUMINAMATH_CALUDE_larger_solution_of_quadratic_l1157_115709

theorem larger_solution_of_quadratic (x : ℝ) :
  x^2 - 9*x - 22 = 0 →
  (∃ y : ℝ, y ≠ x ∧ y^2 - 9*y - 22 = 0) →
  (x = 11 ∨ x < 11) :=
sorry

end NUMINAMATH_CALUDE_larger_solution_of_quadratic_l1157_115709


namespace NUMINAMATH_CALUDE_largest_integer_in_interval_l1157_115739

theorem largest_integer_in_interval : 
  ∃ (y : ℤ), (1/4 : ℚ) < (y : ℚ)/7 ∧ (y : ℚ)/7 < 3/5 ∧ 
  ∀ (z : ℤ), ((1/4 : ℚ) < (z : ℚ)/7 ∧ (z : ℚ)/7 < 3/5) → z ≤ y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_integer_in_interval_l1157_115739


namespace NUMINAMATH_CALUDE_jose_investment_is_225000_l1157_115782

/-- Calculates Jose's investment given the problem conditions -/
def calculate_jose_investment (tom_investment : ℕ) (tom_duration : ℕ) (jose_duration : ℕ) (total_profit : ℕ) (jose_profit : ℕ) : ℕ :=
  (tom_investment * tom_duration * (total_profit - jose_profit)) / (jose_profit * jose_duration)

/-- Proves that Jose's investment is 225000 given the problem conditions -/
theorem jose_investment_is_225000 :
  calculate_jose_investment 30000 12 10 27000 15000 = 225000 := by
  sorry

end NUMINAMATH_CALUDE_jose_investment_is_225000_l1157_115782


namespace NUMINAMATH_CALUDE_statement_a_is_correct_l1157_115784

theorem statement_a_is_correct (x y : ℝ) : x + y < 0 → x^2 - y > x := by
  sorry

end NUMINAMATH_CALUDE_statement_a_is_correct_l1157_115784


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1157_115703

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 216 →
  volume = (((surface_area / 6) ^ (1/2 : ℝ)) ^ 3) →
  volume = 216 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1157_115703


namespace NUMINAMATH_CALUDE_jack_second_half_time_l1157_115764

/-- Jack and Jill's race up the hill -/
def hill_race (jack_first_half jack_total jill_total : ℕ) : Prop :=
  jack_first_half = 19 ∧
  jill_total = 32 ∧
  jack_total + 7 = jill_total

theorem jack_second_half_time 
  (jack_first_half jack_total jill_total : ℕ) 
  (h : hill_race jack_first_half jack_total jill_total) : 
  jack_total - jack_first_half = 6 := by
  sorry

end NUMINAMATH_CALUDE_jack_second_half_time_l1157_115764


namespace NUMINAMATH_CALUDE_line_canonical_form_l1157_115737

/-- Given two planes that intersect to form a line, prove that the line can be represented in canonical form. -/
theorem line_canonical_form (x y z : ℝ) : 
  (2*x - y + 3*z = 1) ∧ (5*x + 4*y - z = 7) →
  ∃ (t : ℝ), x = -11*t ∧ y = 17*t + 2 ∧ z = 13*t + 1 :=
by sorry

end NUMINAMATH_CALUDE_line_canonical_form_l1157_115737
