import Mathlib

namespace problem_1_problem_2_problem_3_l735_73514

-- Problem 1
theorem problem_1 : -3^2 + (-1/2)^2 + (2023 - Real.pi)^0 - |-2| = -47/4 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) : (-2*a^2)^3 * a^2 + a^8 = -7*a^8 := by sorry

-- Problem 3
theorem problem_3 : 2023^2 - 2024 * 2022 = 1 := by sorry

end problem_1_problem_2_problem_3_l735_73514


namespace museum_ticket_cost_l735_73557

theorem museum_ticket_cost : 
  ∀ (num_students num_teachers : ℕ) 
    (student_ticket_cost teacher_ticket_cost : ℕ),
  num_students = 12 →
  num_teachers = 4 →
  student_ticket_cost = 1 →
  teacher_ticket_cost = 3 →
  num_students * student_ticket_cost + num_teachers * teacher_ticket_cost = 24 :=
by
  sorry

end museum_ticket_cost_l735_73557


namespace four_number_sequence_l735_73569

def is_arithmetic_sequence (a b c : ℝ) : Prop := b - a = c - b

def is_geometric_sequence (b c d : ℝ) : Prop := c * c = b * d

theorem four_number_sequence (a b c d : ℝ) 
  (h1 : is_arithmetic_sequence a b c)
  (h2 : is_geometric_sequence b c d)
  (h3 : a + d = 16)
  (h4 : b + c = 12) :
  ((a, b, c, d) = (0, 4, 8, 16)) ∨ ((a, b, c, d) = (15, 9, 3, 1)) := by
  sorry

end four_number_sequence_l735_73569


namespace consecutive_integers_product_l735_73530

theorem consecutive_integers_product (n : ℕ) : 
  n > 0 ∧ (n + (n + 1) < 150) → n * (n + 1) ≤ 5550 := by
  sorry

end consecutive_integers_product_l735_73530


namespace equation_rewrite_l735_73580

/-- Given an equation with roots α and β, prove that a related equation can be rewritten in terms of α, β, and a constant k. -/
theorem equation_rewrite (a b c d α β : ℝ) (hα : α = (a * α + b) / (c * α + d)) (hβ : β = (a * β + b) / (c * β + d)) :
  ∃ k : ℝ, ∀ y z : ℝ, y = (a * z + b) / (c * z + d) →
    (y - α) / (y - β) = k * (z - α) / (z - β) ∧ k = (c * β + d) / (c * α + d) := by
  sorry

end equation_rewrite_l735_73580


namespace number_percentage_problem_l735_73523

theorem number_percentage_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 20 → (40/100 : ℝ) * N = 240 := by
  sorry

end number_percentage_problem_l735_73523


namespace part_one_part_two_l735_73520

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x - a < 0}
def B : Set ℝ := {x | x^2 - 2*x - 8 < 0}

-- Part 1
theorem part_one (a : ℝ) (h : a = 3) :
  let U := A a ∪ B
  B ∪ (U \ A a) = {x | x > -2} := by sorry

-- Part 2
theorem part_two :
  {a : ℝ | A a ∩ B = B} = {a : ℝ | a ≥ 4} := by sorry

end part_one_part_two_l735_73520


namespace slips_with_two_l735_73568

theorem slips_with_two (total : ℕ) (expected_value : ℚ) : 
  total = 15 → expected_value = 46/10 → ∃ x y z : ℕ, 
    x + y + z = total ∧ 
    (2 * x + 5 * y + 8 * z : ℚ) / total = expected_value ∧ 
    x = 8 ∧ y + z = 7 := by
  sorry

end slips_with_two_l735_73568


namespace geometric_sequence_sixth_term_l735_73586

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_ratio : (a 2 + a 3) / (a 1 + a 2) = 2)
  (h_fourth : a 4 = 8) :
  a 6 = 32 := by
  sorry

end geometric_sequence_sixth_term_l735_73586


namespace matrix_power_2018_l735_73560

def A : Matrix (Fin 2) (Fin 2) ℕ := !![1, 1; 1, 1]

theorem matrix_power_2018 :
  A ^ 2018 = !![2^2017, 2^2017; 2^2017, 2^2017] := by sorry

end matrix_power_2018_l735_73560


namespace arrangements_count_l735_73597

/-- The number of ways to arrange 2 objects out of 2 positions -/
def A_2_2 : ℕ := 2

/-- The number of ways to arrange 2 objects out of 3 positions -/
def A_3_2 : ℕ := 6

/-- The number of ways to bind A and B together -/
def bind_AB : ℕ := 2

/-- The total number of people -/
def total_people : ℕ := 5

/-- Theorem: The number of arrangements of 5 people where A and B must stand next to each other,
    and C and D cannot stand next to each other, is 24. -/
theorem arrangements_count : 
  bind_AB * A_2_2 * A_3_2 = 24 :=
by sorry

end arrangements_count_l735_73597


namespace boat_downstream_time_l735_73579

/-- Proves that a boat traveling downstream takes 1 hour to cover 45 km,
    given its speed in still water and the stream's speed. -/
theorem boat_downstream_time 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (distance : ℝ) 
  (h1 : boat_speed = 40)
  (h2 : stream_speed = 5)
  (h3 : distance = 45) :
  distance / (boat_speed + stream_speed) = 1 :=
by sorry

end boat_downstream_time_l735_73579


namespace strategy_game_cost_l735_73561

def total_spent : ℝ := 35.52
def football_cost : ℝ := 14.02
def batman_cost : ℝ := 12.04

theorem strategy_game_cost :
  total_spent - football_cost - batman_cost = 9.46 := by
  sorry

end strategy_game_cost_l735_73561


namespace smallest_divisible_by_1_to_10_l735_73501

/-- The smallest positive integer divisible by all integers from 1 to 10 -/
def smallestDivisibleBy1To10 : ℕ := 2520

/-- Checks if a number is divisible by all integers from 1 to 10 -/
def isDivisibleBy1To10 (n : ℕ) : Prop :=
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ 10 → n % i = 0

theorem smallest_divisible_by_1_to_10 :
  isDivisibleBy1To10 smallestDivisibleBy1To10 ∧
  ∀ n : ℕ, 0 < n ∧ n < smallestDivisibleBy1To10 → ¬isDivisibleBy1To10 n := by
  sorry

end smallest_divisible_by_1_to_10_l735_73501


namespace simplify_expression_l735_73581

theorem simplify_expression (x y : ℝ) : 
  (2 * x + 20) + (150 * x + 30) + y = 152 * x + 50 + y := by
sorry

end simplify_expression_l735_73581


namespace number_problem_l735_73543

theorem number_problem : 
  ∃ (number : ℝ), number * 11 = 165 ∧ number = 15 := by
sorry

end number_problem_l735_73543


namespace functions_continuous_and_equal_l735_73533

/-- Darboux property (intermediate value property) -/
def has_darboux_property (f : ℝ → ℝ) : Prop :=
  ∀ a b y, a < b → f a < y → y < f b → ∃ c, a < c ∧ c < b ∧ f c = y

/-- The problem statement -/
theorem functions_continuous_and_equal
  (f g : ℝ → ℝ)
  (h1 : ∀ a, ⨅ (x > a), f x = g a)
  (h2 : ∀ a, ⨆ (x < a), g x = f a)
  (h3 : has_darboux_property f) :
  Continuous f ∧ Continuous g ∧ f = g := by
  sorry

end functions_continuous_and_equal_l735_73533


namespace petya_prize_probability_at_least_one_prize_probability_l735_73524

-- Define the number of players
def num_players : ℕ := 10

-- Define the number of sides on the die
def die_sides : ℕ := 6

-- Theorem for Petya's probability of winning a prize
theorem petya_prize_probability :
  (5 / 6 : ℚ) ^ (num_players - 1) = (5 / 6 : ℚ) ^ 9 := by sorry

-- Theorem for the probability of at least one player winning a prize
theorem at_least_one_prize_probability :
  1 - (1 / die_sides : ℚ) ^ (num_players - 1) = 1 - (1 / 6 : ℚ) ^ 9 := by sorry

end petya_prize_probability_at_least_one_prize_probability_l735_73524


namespace inequality_solution_f_less_than_one_l735_73540

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x + 1|

-- Theorem 1
theorem inequality_solution (x : ℝ) : f x > x + 5 ↔ x > 4 ∨ x < -2 := by sorry

-- Theorem 2
theorem f_less_than_one (x y : ℝ) (h1 : |x - 3*y - 1| < 1/4) (h2 : |2*y + 1| < 1/6) : f x < 1 := by sorry

end inequality_solution_f_less_than_one_l735_73540


namespace log_expression_simplification_l735_73594

theorem log_expression_simplification 
  (p q r s x y : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (hx : x > 0) (hy : y > 0) : 
  Real.log (p^2 / q) + Real.log (q^3 / r) + Real.log (r^2 / s) - Real.log (p^2 * y / (s^3 * x)) 
  = Real.log (q^2 * r * x * s^2 / y) := by
  sorry

end log_expression_simplification_l735_73594


namespace angle_sum_equality_l735_73531

theorem angle_sum_equality (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan α = 1/7) (h4 : Real.tan β = 3/79) : 5*α + 2*β = π/4 := by
  sorry

end angle_sum_equality_l735_73531


namespace pyramid_circumscribed_sphere_area_l735_73503

theorem pyramid_circumscribed_sphere_area :
  ∀ (a b c : ℝ),
    a = 1 →
    b = Real.sqrt 6 →
    c = 3 →
    (∃ (r : ℝ), r * r = (a * a + b * b + c * c) / 4 ∧
      4 * Real.pi * r * r = 16 * Real.pi) :=
by sorry

end pyramid_circumscribed_sphere_area_l735_73503


namespace exists_permutation_multiple_of_seven_l735_73521

/-- A function that generates all permutations of a list -/
def permutations (l : List ℕ) : List (List ℕ) :=
  sorry

/-- A function that converts a list of digits to a natural number -/
def list_to_number (l : List ℕ) : ℕ :=
  sorry

/-- The theorem stating that there exists a permutation of digits 1, 3, 7, 9 that forms a multiple of 7 -/
theorem exists_permutation_multiple_of_seven :
  ∃ (perm : List ℕ), perm ∈ permutations [1, 3, 7, 9] ∧ (list_to_number perm) % 7 = 0 :=
by
  sorry

end exists_permutation_multiple_of_seven_l735_73521


namespace valid_outfit_count_l735_73574

/-- The number of colors available for each item -/
def num_colors : ℕ := 6

/-- The number of different types of clothing items -/
def num_items : ℕ := 4

/-- Calculates the total number of outfit combinations without restrictions -/
def total_combinations : ℕ := num_colors ^ num_items

/-- Calculates the number of outfits where all items are the same color -/
def same_color_outfits : ℕ := num_colors

/-- Calculates the number of outfits where shoes don't match any other item color -/
def valid_shoe_combinations : ℕ := num_colors * num_colors * num_colors * (num_colors - 1)

/-- Calculates the number of outfits where shirt, pants, and hat are the same color, but shoes are different -/
def same_color_except_shoes : ℕ := num_colors * (num_colors - 1) - num_colors

/-- The main theorem stating the number of valid outfit combinations -/
theorem valid_outfit_count : 
  total_combinations - same_color_outfits - valid_shoe_combinations - same_color_except_shoes = 1104 := by
  sorry

end valid_outfit_count_l735_73574


namespace renovation_material_sum_l735_73593

/-- The amount of sand required for the renovation project in truck-loads -/
def sand : ℚ := 0.16666666666666666

/-- The amount of dirt required for the renovation project in truck-loads -/
def dirt : ℚ := 0.3333333333333333

/-- The amount of cement required for the renovation project in truck-loads -/
def cement : ℚ := 0.16666666666666666

/-- The total amount of material required for the renovation project in truck-loads -/
def total_material : ℚ := 0.6666666666666666

/-- Theorem stating that the sum of sand, dirt, and cement equals the total material required -/
theorem renovation_material_sum :
  sand + dirt + cement = total_material := by sorry

end renovation_material_sum_l735_73593


namespace train_platform_length_equality_l735_73591

/-- Given a train and a platform with specific conditions, prove that the length of the platform equals the length of the train. -/
theorem train_platform_length_equality
  (train_speed : Real) -- Speed of the train in km/hr
  (crossing_time : Real) -- Time to cross the platform in minutes
  (train_length : Real) -- Length of the train in meters
  (h1 : train_speed = 180) -- Train speed is 180 km/hr
  (h2 : crossing_time = 1) -- Time to cross the platform is 1 minute
  (h3 : train_length = 1500) -- Length of the train is 1500 meters
  : Real := -- Length of the platform in meters
by
  sorry

#check train_platform_length_equality

end train_platform_length_equality_l735_73591


namespace opposite_of_negative_2016_l735_73534

theorem opposite_of_negative_2016 : Int.neg (-2016) = 2016 := by
  sorry

end opposite_of_negative_2016_l735_73534


namespace interval_for_quadratic_function_l735_73566

/-- The function f(x) = -x^2 -/
def f (x : ℝ) : ℝ := -x^2

theorem interval_for_quadratic_function (a b : ℝ) :
  (∀ x ∈ Set.Icc a b, f x ≥ 2*a) ∧  -- minimum value condition
  (∀ x ∈ Set.Icc a b, f x ≤ 2*b) ∧  -- maximum value condition
  (∃ x ∈ Set.Icc a b, f x = 2*a) ∧  -- minimum value is achieved
  (∃ x ∈ Set.Icc a b, f x = 2*b) →  -- maximum value is achieved
  a = 1 ∧ b = 3 :=
by sorry

end interval_for_quadratic_function_l735_73566


namespace twins_age_problem_l735_73519

theorem twins_age_problem (age : ℕ) : 
  (age + 1) * (age + 1) = age * age + 5 → age = 2 := by
sorry

end twins_age_problem_l735_73519


namespace inequality_proof_l735_73572

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a*b + b*c + c*a) := by
  sorry

end inequality_proof_l735_73572


namespace right_triangle_median_l735_73550

/-- Given a right triangle XYZ with ∠XYZ as the right angle, XY = 6, YZ = 8, and N as the midpoint of XZ, prove that YN = 5 -/
theorem right_triangle_median (X Y Z N : ℝ × ℝ) : 
  (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = 6^2 →
  (Y.1 - Z.1)^2 + (Y.2 - Z.2)^2 = 8^2 →
  (X.1 - Z.1)^2 + (X.2 - Z.2)^2 = (X.1 - Y.1)^2 + (X.2 - Y.2)^2 + (Y.1 - Z.1)^2 + (Y.2 - Z.2)^2 →
  N = ((X.1 + Z.1) / 2, (X.2 + Z.2) / 2) →
  (Y.1 - N.1)^2 + (Y.2 - N.2)^2 = 5^2 := by
sorry

end right_triangle_median_l735_73550


namespace coinciding_rest_days_theorem_l735_73548

def charlie_cycle : ℕ := 6
def dana_cycle : ℕ := 7
def total_days : ℕ := 1000

def coinciding_rest_days (c_cycle d_cycle total : ℕ) : ℕ :=
  let lcm := Nat.lcm c_cycle d_cycle
  let full_cycles := total / lcm
  let c_rest_days := 2
  let d_rest_days := 2
  let coinciding_days_per_cycle := 4  -- This should be proven, not assumed
  full_cycles * coinciding_days_per_cycle

theorem coinciding_rest_days_theorem :
  coinciding_rest_days charlie_cycle dana_cycle total_days = 92 := by
  sorry

#eval coinciding_rest_days charlie_cycle dana_cycle total_days

end coinciding_rest_days_theorem_l735_73548


namespace sophie_donuts_result_l735_73582

def sophie_donuts (budget : ℝ) (box_cost : ℝ) (discount_rate : ℝ) 
                   (boxes_bought : ℕ) (donuts_per_box : ℕ) 
                   (boxes_to_mom : ℕ) (donuts_to_sister : ℕ) : ℝ × ℕ :=
  let total_cost := box_cost * boxes_bought
  let discounted_cost := total_cost * (1 - discount_rate)
  let total_donuts := boxes_bought * donuts_per_box
  let donuts_given := boxes_to_mom * donuts_per_box + donuts_to_sister
  let donuts_left := total_donuts - donuts_given
  (discounted_cost, donuts_left)

theorem sophie_donuts_result : 
  sophie_donuts 50 12 0.1 4 12 1 6 = (43.2, 30) :=
by sorry

end sophie_donuts_result_l735_73582


namespace june_design_purple_tiles_l735_73518

/-- Represents the number of tiles of each color in June's design -/
structure TileDesign where
  total : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  purple : Nat

/-- Theorem stating the number of purple tiles in June's design -/
theorem june_design_purple_tiles (d : TileDesign) : 
  d.total = 20 ∧ 
  d.yellow = 3 ∧ 
  d.blue = d.yellow + 1 ∧ 
  d.white = 7 → 
  d.purple = 6 := by
  sorry

#check june_design_purple_tiles

end june_design_purple_tiles_l735_73518


namespace triangle_existence_l735_73564

theorem triangle_existence (n : ℕ) (points : Finset (ℝ × ℝ × ℝ)) (segments : Finset ((ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ))) :
  points.card = 2 * n →
  segments.card = n^2 + 1 →
  ∃ (a b c : ℝ × ℝ × ℝ), 
    a ∈ points ∧ b ∈ points ∧ c ∈ points ∧
    (a, b) ∈ segments ∧ (b, c) ∈ segments ∧ (a, c) ∈ segments :=
by sorry

end triangle_existence_l735_73564


namespace house_painting_time_l735_73544

/-- Given that 12 women can paint a house in 6 days, prove that 18 women 
    working at the same rate can paint the same house in 4 days. -/
theorem house_painting_time 
  (women_rate : ℝ → ℝ → ℝ) -- Function that takes number of women and days, returns houses painted
  (h1 : women_rate 12 6 = 1) -- 12 women paint 1 house in 6 days
  (h2 : ∀ w d, women_rate w d = w * d * (women_rate 1 1)) -- Linear relationship
  : women_rate 18 4 = 1 := by
  sorry

end house_painting_time_l735_73544


namespace exists_distinct_singleton_solutions_l735_73552

/-- Solution set of x^2 + 4x - 2a ≤ 0 -/
def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 + 4*x - 2*a ≤ 0}

/-- Solution set of x^2 - ax + a + 3 ≤ 0 -/
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - a*x + a + 3 ≤ 0}

/-- Theorem stating that there exists an 'a' for which A and B are singleton sets with different elements -/
theorem exists_distinct_singleton_solutions :
  ∃ (a : ℝ), (∃! x, x ∈ A a) ∧ (∃! y, y ∈ B a) ∧ (∀ x y, x ∈ A a → y ∈ B a → x ≠ y) :=
sorry

end exists_distinct_singleton_solutions_l735_73552


namespace geometric_sequence_ratio_l735_73507

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

-- Define the arithmetic sequence condition
def arithmetic_sequence_condition (a : ℕ → ℝ) (q : ℝ) : Prop :=
  3 * a 1 + 2 * a 3 = 2 * ((1/2) * a 5)

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  arithmetic_sequence_condition a q →
  (a 9 + a 10) / (a 7 + a 8) = 3 := by
sorry

end geometric_sequence_ratio_l735_73507


namespace rays_grocery_bill_l735_73573

def hamburger_price : ℝ := 5.00
def crackers_price : ℝ := 3.50
def vegetable_price : ℝ := 2.00
def vegetable_bags : ℕ := 4
def cheese_price : ℝ := 3.50
def discount_rate : ℝ := 0.10

def total_before_discount : ℝ :=
  hamburger_price + crackers_price + (vegetable_price * vegetable_bags) + cheese_price

def discount_amount : ℝ := total_before_discount * discount_rate

def total_after_discount : ℝ := total_before_discount - discount_amount

theorem rays_grocery_bill :
  total_after_discount = 18.00 := by
  sorry

end rays_grocery_bill_l735_73573


namespace school_pencils_l735_73598

theorem school_pencils (num_pens : ℕ) (pencil_cost pen_cost total_cost : ℚ) :
  num_pens = 56 ∧
  pencil_cost = 5/2 ∧
  pen_cost = 7/2 ∧
  total_cost = 291 →
  ∃ num_pencils : ℕ, num_pencils * pencil_cost + num_pens * pen_cost = total_cost ∧ num_pencils = 38 :=
by sorry

end school_pencils_l735_73598


namespace saturday_zoo_visitors_l735_73517

/-- The number of people who visited the zoo on Friday -/
def friday_visitors : ℕ := 1250

/-- The number of people who visited the zoo on Saturday -/
def saturday_visitors : ℕ := 3 * friday_visitors

/-- Theorem stating the number of visitors on Saturday -/
theorem saturday_zoo_visitors : saturday_visitors = 3750 := by sorry

end saturday_zoo_visitors_l735_73517


namespace sugar_consumption_reduction_l735_73526

theorem sugar_consumption_reduction (initial_price new_price : ℝ) 
  (h1 : initial_price = 6)
  (h2 : new_price = 7.50) :
  let reduction_percentage := (1 - initial_price / new_price) * 100
  reduction_percentage = 20 := by sorry

end sugar_consumption_reduction_l735_73526


namespace sine_graph_translation_l735_73535

theorem sine_graph_translation (x : ℝ) :
  5 * Real.sin (2 * (x + π/12) + π/6) = 5 * Real.sin (2 * x) := by
  sorry

end sine_graph_translation_l735_73535


namespace workshop_average_salary_l735_73508

/-- Proves that the average salary of all workers is 750, given the specified conditions. -/
theorem workshop_average_salary
  (total_workers : ℕ)
  (technicians : ℕ)
  (technician_salary : ℕ)
  (non_technician_salary : ℕ)
  (h1 : total_workers = 20)
  (h2 : technicians = 5)
  (h3 : technician_salary = 900)
  (h4 : non_technician_salary = 700) :
  (technicians * technician_salary + (total_workers - technicians) * non_technician_salary) / total_workers = 750 :=
by sorry

end workshop_average_salary_l735_73508


namespace geometric_sequence_sum_l735_73587

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 1 > 0 →
  a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 16 →
  a 3 + a 5 = 4 := by
  sorry

end geometric_sequence_sum_l735_73587


namespace product_digit_sum_l735_73590

def repeat_digits (d : ℕ) (n : ℕ) : ℕ :=
  d * (10^(3*n) - 1) / 999

def number1 : ℕ := repeat_digits 400 333
def number2 : ℕ := repeat_digits 606 333

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def units_digit (n : ℕ) : ℕ := n % 10

theorem product_digit_sum :
  tens_digit (number1 * number2) + units_digit (number1 * number2) = 0 := by
  sorry

end product_digit_sum_l735_73590


namespace sqrt_of_16_l735_73595

theorem sqrt_of_16 : Real.sqrt 16 = 4 := by
  sorry

end sqrt_of_16_l735_73595


namespace fraction_sum_theorem_l735_73583

theorem fraction_sum_theorem : 
  (1 : ℚ) / 15 + 2 / 15 + 3 / 15 + 4 / 15 + 5 / 15 + 
  6 / 15 + 7 / 15 + 8 / 15 + 9 / 15 + 46 / 15 = 91 / 15 := by
  sorry

end fraction_sum_theorem_l735_73583


namespace egg_collection_difference_l735_73510

/-- Egg collection problem -/
theorem egg_collection_difference :
  ∀ (benjamin carla trisha : ℕ),
  benjamin = 6 →
  carla = 3 * benjamin →
  benjamin + carla + trisha = 26 →
  benjamin - trisha = 4 :=
by sorry

end egg_collection_difference_l735_73510


namespace set_union_problem_l735_73515

theorem set_union_problem (A B : Set ℕ) (a : ℕ) :
  A = {1, 2} →
  B = {3, a} →
  A ∩ B = {1} →
  A ∪ B = {1, 2, 3} := by
sorry

end set_union_problem_l735_73515


namespace units_digit_of_product_l735_73562

theorem units_digit_of_product (a b c : ℕ) : (4^1001 * 8^1002 * 12^1003) % 10 = 8 := by
  sorry

end units_digit_of_product_l735_73562


namespace five_digit_sum_contains_zero_l735_73584

def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n < 100000

def digits_differ_by_two (a b : ℕ) : Prop :=
  ∃ (d1 d2 d3 d4 d5 e1 e2 e3 e4 e5 : ℕ),
    a = d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5 ∧
    b = e1 * 10000 + e2 * 1000 + e3 * 100 + e4 * 10 + e5 ∧
    ({d1, d2, d3, d4, d5} : Finset ℕ) = {e1, e2, e3, e4, e5} ∧
    (d1 = e1 ∧ d2 = e2 ∧ d4 = e4 ∧ d5 = e5) ∨
    (d1 = e1 ∧ d2 = e2 ∧ d3 = e3 ∧ d5 = e5) ∨
    (d1 = e1 ∧ d2 = e2 ∧ d3 = e3 ∧ d4 = e4)

def contains_zero (n : ℕ) : Prop :=
  ∃ (d1 d2 d3 d4 d5 : ℕ),
    n = d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5 ∧
    (d1 = 0 ∨ d2 = 0 ∨ d3 = 0 ∨ d4 = 0 ∨ d5 = 0)

theorem five_digit_sum_contains_zero (a b : ℕ) :
  is_five_digit a → is_five_digit b → digits_differ_by_two a b → a + b = 111111 →
  contains_zero a ∨ contains_zero b :=
sorry

end five_digit_sum_contains_zero_l735_73584


namespace four_balls_three_boxes_l735_73522

/-- The number of ways to put distinguishable balls into distinguishable boxes -/
def ways_to_distribute (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 81 ways to put 4 distinguishable balls into 3 distinguishable boxes -/
theorem four_balls_three_boxes : ways_to_distribute 4 3 = 81 := by
  sorry

end four_balls_three_boxes_l735_73522


namespace gravel_density_l735_73589

/-- Proves that the density of gravel is approximately 267 kg/m³ given the conditions of the bucket problem. -/
theorem gravel_density (bucket_volume : ℝ) (additional_water : ℝ) (full_bucket_weight : ℝ) (empty_bucket_weight : ℝ) 
  (h1 : bucket_volume = 12)
  (h2 : additional_water = 3)
  (h3 : full_bucket_weight = 28)
  (h4 : empty_bucket_weight = 1)
  (h5 : ∀ x, x > 0 → x * 1 = x) -- 1 liter of water weighs 1 kg
  : ∃ (density : ℝ), abs (density - 267) < 1 ∧ 
    density = (full_bucket_weight - empty_bucket_weight - additional_water) / 
              (bucket_volume - additional_water) * 1000 := by
  sorry

end gravel_density_l735_73589


namespace prime_ratio_natural_numbers_l735_73528

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem prime_ratio_natural_numbers :
  ∀ x y : ℕ, x > 0 ∧ y > 0 →
    (is_prime ((x * y^3) / (x + y)) ↔ x = 14 ∧ y = 2) :=
by sorry


end prime_ratio_natural_numbers_l735_73528


namespace triangle_sin_a_l735_73571

theorem triangle_sin_a (A B C : ℝ) (a b c : ℝ) (h : ℝ) : 
  B = π / 4 →
  h = c / 3 →
  (1/2) * a * h = (1/2) * a * c * Real.sin B →
  b^2 = a^2 + c^2 - 2 * a * c * Real.cos B →
  Real.sin A = a * Real.sin B / b →
  Real.sin A = 3 * Real.sqrt 10 / 10 :=
sorry

end triangle_sin_a_l735_73571


namespace min_value_theorem_l735_73576

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 1) (hab : a + b = 3/2) :
  (∀ x y : ℝ, x > 0 → y > 1 → x + y = 3/2 → 2/x + 1/(y-1) ≥ 2/a + 1/(b-1)) ∧
  2/a + 1/(b-1) = 6 + 4 * Real.sqrt 2 :=
sorry

end min_value_theorem_l735_73576


namespace teacher_budget_shortfall_l735_73539

def euro_to_usd_rate : ℝ := 1.2
def last_year_budget : ℝ := 6
def this_year_allocation : ℝ := 50
def charity_grant : ℝ := 20
def gift_card : ℝ := 10

def textbooks_price : ℝ := 45
def textbooks_discount : ℝ := 0.15
def textbooks_tax : ℝ := 0.08

def notebooks_price : ℝ := 18
def notebooks_discount : ℝ := 0.10
def notebooks_tax : ℝ := 0.05

def pens_price : ℝ := 27
def pens_discount : ℝ := 0.05
def pens_tax : ℝ := 0.06

def art_supplies_price : ℝ := 35
def art_supplies_tax : ℝ := 0.07

def folders_price : ℝ := 15
def folders_voucher : ℝ := 5
def folders_tax : ℝ := 0.04

theorem teacher_budget_shortfall :
  let converted_budget := last_year_budget * euro_to_usd_rate
  let total_budget := converted_budget + this_year_allocation + charity_grant + gift_card
  
  let textbooks_cost := textbooks_price * (1 - textbooks_discount) * (1 + textbooks_tax)
  let notebooks_cost := notebooks_price * (1 - notebooks_discount) * (1 + notebooks_tax)
  let pens_cost := pens_price * (1 - pens_discount) * (1 + pens_tax)
  let art_supplies_cost := art_supplies_price * (1 + art_supplies_tax)
  let folders_cost := (folders_price - folders_voucher) * (1 + folders_tax)
  
  let total_cost := textbooks_cost + notebooks_cost + pens_cost + art_supplies_cost + folders_cost - gift_card
  
  total_budget - total_cost = -36.16 :=
by sorry

end teacher_budget_shortfall_l735_73539


namespace jessie_weight_calculation_l735_73599

/-- Calculates the initial weight given the current weight and weight lost -/
def initial_weight (current_weight weight_lost : ℝ) : ℝ :=
  current_weight + weight_lost

/-- Theorem: If Jessie's current weight is 27 kg and she lost 10 kg, her initial weight was 37 kg -/
theorem jessie_weight_calculation :
  let current_weight : ℝ := 27
  let weight_lost : ℝ := 10
  initial_weight current_weight weight_lost = 37 := by
  sorry

end jessie_weight_calculation_l735_73599


namespace andy_remaining_demerits_l735_73563

/-- The maximum number of demerits Andy can get in a month before getting fired -/
def max_demerits : ℕ := 50

/-- The number of times Andy showed up late in the first week -/
def late_instances : ℕ := 6

/-- The number of demerits Andy gets for each late instance -/
def late_demerits : ℕ := 2

/-- The number of demerits Andy got for making an inappropriate joke in the second week -/
def joke_demerits : ℕ := 15

/-- The number of times Andy used his phone during work hours in the third week -/
def phone_instances : ℕ := 4

/-- The number of demerits Andy gets for each phone use instance -/
def phone_demerits : ℕ := 3

/-- The number of days Andy didn't tidy up his work area in the fourth week -/
def untidy_days : ℕ := 5

/-- The number of demerits Andy gets for each day of not tidying up -/
def untidy_demerits : ℕ := 1

/-- The total number of demerits Andy has accumulated so far -/
def total_demerits : ℕ := 
  late_instances * late_demerits + 
  joke_demerits + 
  phone_instances * phone_demerits + 
  untidy_days * untidy_demerits

/-- The number of additional demerits Andy can receive before getting fired -/
def additional_demerits : ℕ := max_demerits - total_demerits

theorem andy_remaining_demerits : additional_demerits = 6 := by
  sorry

end andy_remaining_demerits_l735_73563


namespace field_trip_van_capacity_l735_73536

theorem field_trip_van_capacity (students : ℕ) (adults : ℕ) (vans : ℕ) :
  students = 25 → adults = 5 → vans = 6 →
  (students + adults) / vans = 5 := by
  sorry

end field_trip_van_capacity_l735_73536


namespace aquarium_illness_percentage_l735_73529

theorem aquarium_illness_percentage (total_visitors : ℕ) (healthy_visitors : ℕ) : 
  total_visitors = 500 → 
  healthy_visitors = 300 → 
  (total_visitors - healthy_visitors : ℚ) / total_visitors * 100 = 40 := by
  sorry

end aquarium_illness_percentage_l735_73529


namespace polynomial_division_theorem_l735_73546

/-- The polynomial to be divided -/
def P (x : ℝ) : ℝ := 9*x^3 - 5*x^2 + 8*x + 15

/-- The divisor polynomial -/
def D (x : ℝ) : ℝ := x - 3

/-- The quotient polynomial -/
def Q (x : ℝ) : ℝ := 9*x^2 + 22*x + 74

/-- The remainder -/
def R : ℝ := 237

theorem polynomial_division_theorem :
  ∀ x : ℝ, P x = D x * Q x + R :=
by sorry

end polynomial_division_theorem_l735_73546


namespace complex_magnitude_problem_l735_73551

theorem complex_magnitude_problem (z : ℂ) (i : ℂ) (h : z = (1 + i) / i) : Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_magnitude_problem_l735_73551


namespace least_integer_with_specific_divisibility_l735_73588

theorem least_integer_with_specific_divisibility : ∃ (n : ℕ), 
  (∀ (k : ℕ), k ≤ 28 → k ∣ n) ∧ 
  (30 ∣ n) ∧ 
  ¬(29 ∣ n) ∧
  (∀ (m : ℕ), m < n → ¬((∀ (k : ℕ), k ≤ 28 → k ∣ m) ∧ (30 ∣ m) ∧ ¬(29 ∣ m))) ∧
  n = 232792560 := by
sorry

end least_integer_with_specific_divisibility_l735_73588


namespace complex_powers_sum_l735_73585

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_powers_sum : i^245 + i^246 + i^247 + i^248 + i^249 = i := by sorry

end complex_powers_sum_l735_73585


namespace smallest_prime_perimeter_scalene_triangle_l735_73577

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

/-- A function that checks if three numbers form a valid triangle -/
def isValidTriangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- The theorem stating the smallest possible perimeter of a scalene triangle
    with prime side lengths greater than 3 and prime perimeter -/
theorem smallest_prime_perimeter_scalene_triangle :
  ∃ (a b c : ℕ),
    a < b ∧ b < c ∧
    isPrime a ∧ isPrime b ∧ isPrime c ∧
    a > 3 ∧ b > 3 ∧ c > 3 ∧
    isValidTriangle a b c ∧
    isPrime (a + b + c) ∧
    (∀ (x y z : ℕ),
      x < y ∧ y < z ∧
      isPrime x ∧ isPrime y ∧ isPrime z ∧
      x > 3 ∧ y > 3 ∧ z > 3 ∧
      isValidTriangle x y z ∧
      isPrime (x + y + z) →
      a + b + c ≤ x + y + z) ∧
    a + b + c = 23 :=
sorry

end smallest_prime_perimeter_scalene_triangle_l735_73577


namespace negative_sixty_four_to_seven_thirds_l735_73596

theorem negative_sixty_four_to_seven_thirds : (-64 : ℝ) ^ (7/3) = -16384 := by
  sorry

end negative_sixty_four_to_seven_thirds_l735_73596


namespace final_selling_price_l735_73555

/-- Given an original price and a first discount, calculate the final selling price after an additional 20% discount -/
theorem final_selling_price (m n : ℝ) : 
  let original_price := m
  let first_discount := n
  let price_after_first_discount := original_price - first_discount
  let second_discount_rate := (20 : ℝ) / 100
  let final_price := price_after_first_discount * (1 - second_discount_rate)
  final_price = (4/5) * (m - n) := by
sorry

end final_selling_price_l735_73555


namespace equilateral_is_cute_specific_triangle_is_cute_right_angled_cute_triangle_side_length_l735_73509

/-- A triangle is cute if the sum of the squares of two sides is equal to twice the square of the third side -/
def IsCuteTriangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = 2 * c^2 ∨ a^2 + c^2 = 2 * b^2 ∨ b^2 + c^2 = 2 * a^2

theorem equilateral_is_cute (a : ℝ) (ha : a > 0) : IsCuteTriangle a a a :=
  sorry

theorem specific_triangle_is_cute : IsCuteTriangle 4 (2 * Real.sqrt 6) (2 * Real.sqrt 5) :=
  sorry

theorem right_angled_cute_triangle_side_length 
  (a b c : ℝ) 
  (h_right : a^2 + b^2 = c^2) 
  (h_cute : IsCuteTriangle a b c) 
  (h_ac : b = 2 * Real.sqrt 2) : 
  c = 2 * Real.sqrt 6 ∨ c = 2 * Real.sqrt 3 :=
  sorry

end equilateral_is_cute_specific_triangle_is_cute_right_angled_cute_triangle_side_length_l735_73509


namespace simplify_and_rationalize_l735_73537

theorem simplify_and_rationalize (x y z : ℝ) :
  x > 0 → y > 0 → z > 0 →
  (x / Real.sqrt 7) * (Real.sqrt 9 / Real.sqrt 11) * (z / Real.sqrt 13) =
  15 * Real.sqrt 1001 / 1001 →
  x = Real.sqrt 5 ∧ y = Real.sqrt 9 ∧ z = Real.sqrt 15 :=
by sorry

end simplify_and_rationalize_l735_73537


namespace arccos_cos_eight_l735_73592

theorem arccos_cos_eight (h : 0 ≤ 8 - 2 * Real.pi ∧ 8 - 2 * Real.pi < Real.pi) :
  Real.arccos (Real.cos 8) = 8 - 2 * Real.pi := by
  sorry

end arccos_cos_eight_l735_73592


namespace intercept_sum_l735_73506

/-- The line equation 2x - y + 4 = 0 -/
def line_equation (x y : ℝ) : Prop := 2 * x - y + 4 = 0

/-- The x-intercept of the line -/
def x_intercept : ℝ := -2

/-- The y-intercept of the line -/
def y_intercept : ℝ := 4

/-- Theorem: The sum of the x-intercept and y-intercept of the line 2x - y + 4 = 0 is equal to 2 -/
theorem intercept_sum : x_intercept + y_intercept = 2 := by
  sorry

end intercept_sum_l735_73506


namespace price_after_discounts_l735_73541

-- Define the discount rates
def discount1 : ℚ := 20 / 100
def discount2 : ℚ := 10 / 100
def discount3 : ℚ := 5 / 100

-- Define the original and final prices
def originalPrice : ℚ := 10000
def finalPrice : ℚ := 6800

-- Theorem statement
theorem price_after_discounts :
  originalPrice * (1 - discount1) * (1 - discount2) * (1 - discount3) = finalPrice := by
  sorry

end price_after_discounts_l735_73541


namespace identify_genuine_coin_l735_73556

/-- Represents the result of a weighing -/
inductive WeighingResult
  | Equal : WeighingResult
  | Unequal : WeighingResult

/-- Represents a coin -/
inductive Coin
  | Genuine : Coin
  | Counterfeit : Coin

/-- Represents a weighing operation -/
def weighing (a b : Coin) : WeighingResult :=
  match a, b with
  | Coin.Genuine, Coin.Genuine => WeighingResult.Equal
  | Coin.Counterfeit, Coin.Counterfeit => WeighingResult.Equal
  | _, _ => WeighingResult.Unequal

/-- Theorem stating that at least one genuine coin can be identified in at most 2 weighings -/
theorem identify_genuine_coin
  (coins : Fin 5 → Coin)
  (h_genuine : ∃ i j k, i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ coins i = Coin.Genuine ∧ coins j = Coin.Genuine ∧ coins k = Coin.Genuine)
  (h_counterfeit : ∃ i j, i ≠ j ∧ coins i = Coin.Counterfeit ∧ coins j = Coin.Counterfeit) :
  ∃ (w₁ w₂ : Fin 5 × Fin 5), ∃ (i : Fin 5), coins i = Coin.Genuine :=
sorry

end identify_genuine_coin_l735_73556


namespace imaginary_part_of_complex_division_l735_73500

theorem imaginary_part_of_complex_division (Z : ℂ) (h : Z = 1 - 2*I) :
  (Complex.im ((1 : ℂ) + 3*I) / Z) = 1 := by
  sorry

end imaginary_part_of_complex_division_l735_73500


namespace fraction_equality_l735_73547

theorem fraction_equality : (1721^2 - 1714^2) / (1728^2 - 1707^2) = 1/3 := by
  sorry

end fraction_equality_l735_73547


namespace chocolate_division_l735_73502

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (num_friends : ℕ) :
  total_chocolate = 75 / 7 →
  num_piles = 5 →
  num_friends = 4 →
  (total_chocolate / num_piles) * (num_piles - 1) / num_friends = 15 / 7 :=
by sorry

end chocolate_division_l735_73502


namespace parallel_vectors_m_value_l735_73512

/-- Two vectors in ℝ² are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (2, -3)
  let b : ℝ × ℝ := (m, 6)
  parallel a b → m = -4 :=
by
  sorry

end parallel_vectors_m_value_l735_73512


namespace find_number_l735_73504

theorem find_number (G N : ℕ) (h1 : G = 4) (h2 : N % G = 6) (h3 : 1856 % G = 4) : N = 1862 := by
  sorry

end find_number_l735_73504


namespace smallest_hypotenuse_right_triangle_isosceles_right_triangle_minimizes_hypotenuse_l735_73513

theorem smallest_hypotenuse_right_triangle (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  a + b + c = 8 →
  c ≥ 4 * Real.sqrt 2 :=
by sorry

theorem isosceles_right_triangle_minimizes_hypotenuse :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a^2 + b^2 = c^2 ∧
    a + b + c = 8 ∧
    c = 4 * Real.sqrt 2 ∧
    a = b :=
by sorry

end smallest_hypotenuse_right_triangle_isosceles_right_triangle_minimizes_hypotenuse_l735_73513


namespace driving_time_is_55_minutes_l735_73570

/-- Calculates the driving time per trip given total moving time, number of trips, and car filling time -/
def driving_time_per_trip (total_time_hours : ℕ) (num_trips : ℕ) (filling_time_minutes : ℕ) : ℕ :=
  let total_time_minutes := total_time_hours * 60
  let total_filling_time := num_trips * filling_time_minutes
  let total_driving_time := total_time_minutes - total_filling_time
  total_driving_time / num_trips

/-- Theorem stating that given the problem conditions, the driving time per trip is 55 minutes -/
theorem driving_time_is_55_minutes :
  driving_time_per_trip 7 6 15 = 55 := by
  sorry

#eval driving_time_per_trip 7 6 15

end driving_time_is_55_minutes_l735_73570


namespace remaining_payment_prove_remaining_payment_l735_73567

/-- Given a product with a deposit, sales tax, and processing fee, calculate the remaining amount to be paid -/
theorem remaining_payment (deposit_percentage : ℝ) (deposit_amount : ℝ) (sales_tax_percentage : ℝ) (processing_fee : ℝ) : ℝ :=
  let full_price := deposit_amount / deposit_percentage
  let sales_tax := sales_tax_percentage * full_price
  let total_additional_expenses := sales_tax + processing_fee
  full_price - deposit_amount + total_additional_expenses

/-- Prove that the remaining payment for the given conditions is $1520 -/
theorem prove_remaining_payment :
  remaining_payment 0.1 140 0.15 50 = 1520 := by
  sorry

end remaining_payment_prove_remaining_payment_l735_73567


namespace new_ratio_is_7_to_5_l735_73578

/-- Represents the ratio of toddlers to infants -/
structure Ratio :=
  (toddlers : ℕ)
  (infants : ℕ)

def initial_ratio : Ratio := ⟨7, 3⟩
def toddler_count : ℕ := 42
def new_infants : ℕ := 12

def calculate_new_ratio (r : Ratio) (t : ℕ) (n : ℕ) : Ratio :=
  let initial_infants := t * r.infants / r.toddlers
  ⟨t, initial_infants + n⟩

theorem new_ratio_is_7_to_5 :
  let new_ratio := calculate_new_ratio initial_ratio toddler_count new_infants
  ∃ (k : ℕ), k > 0 ∧ new_ratio.toddlers = 7 * k ∧ new_ratio.infants = 5 * k :=
sorry

end new_ratio_is_7_to_5_l735_73578


namespace number_of_tippers_l735_73558

def lawn_price : ℕ := 33
def lawns_mowed : ℕ := 16
def tip_amount : ℕ := 10
def total_earnings : ℕ := 558

theorem number_of_tippers : ℕ :=
  by
    sorry

end number_of_tippers_l735_73558


namespace right_triangle_hypotenuse_l735_73554

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ),
  -- Conditions
  (a > 0) → (b > 0) → (c > 0) →  -- Positive sides
  (a^2 + b^2 = c^2) →            -- Right triangle (Pythagorean theorem)
  (a + b = 7) →                  -- Sum of legs
  (a * b / 2 = 6) →              -- Area
  (c = 5) :=                     -- Conclusion: hypotenuse length
by
  sorry

#check right_triangle_hypotenuse

end right_triangle_hypotenuse_l735_73554


namespace green_hat_cost_l735_73545

/-- Proves the cost of green hats given the total number of hats, cost of blue hats, 
    total price, and number of green hats. -/
theorem green_hat_cost 
  (total_hats : ℕ) 
  (blue_hat_cost : ℕ) 
  (total_price : ℕ) 
  (green_hats : ℕ) 
  (h1 : total_hats = 85) 
  (h2 : blue_hat_cost = 6) 
  (h3 : total_price = 540) 
  (h4 : green_hats = 30) : 
  (total_price - blue_hat_cost * (total_hats - green_hats)) / green_hats = 7 := by
  sorry

end green_hat_cost_l735_73545


namespace max_students_equal_distribution_l735_73575

theorem max_students_equal_distribution (pens pencils erasers notebooks : ℕ) 
  (h1 : pens = 1802)
  (h2 : pencils = 1203)
  (h3 : erasers = 1508)
  (h4 : notebooks = 2400) :
  Nat.gcd (Nat.gcd (Nat.gcd pens pencils) erasers) notebooks = 1 := by
  sorry

end max_students_equal_distribution_l735_73575


namespace parallel_line_plane_intersection_not_always_parallel_l735_73525

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation between lines and planes
variable (parallelLP : Line → Plane → Prop)

-- Define the parallel relation between lines
variable (parallelLL : Line → Line → Prop)

-- Define the intersection of two planes
variable (intersectionPP : Plane → Plane → Line)

-- Define the "contained in" relation for a line in a plane
variable (containedIn : Line → Plane → Prop)

-- Theorem statement
theorem parallel_line_plane_intersection_not_always_parallel 
  (α β : Plane) (m n : Line) : 
  ∃ (α β : Plane) (m n : Line), 
    α ≠ β ∧ m ≠ n ∧ 
    parallelLP m α ∧ 
    intersectionPP α β = n ∧ 
    ¬(parallelLL m n) := by sorry

end parallel_line_plane_intersection_not_always_parallel_l735_73525


namespace inequality_proof_l735_73549

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c < d) : a - c > b - d := by
  sorry

end inequality_proof_l735_73549


namespace max_value_f_neg_one_range_of_a_l735_73511

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := x * f a x

-- Define the function h
def h (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 - (2*a - 1) * x + a - 1

-- Theorem for the maximum value of f when a = -1
theorem max_value_f_neg_one :
  ∃ (max : ℝ), max = -1 ∧ ∀ x > 0, f (-1) x ≤ max :=
sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∀ x ≥ 1, g a x ≤ h a x) → a ≥ 1 :=
sorry

end max_value_f_neg_one_range_of_a_l735_73511


namespace mathematics_letter_probability_l735_73505

theorem mathematics_letter_probability : 
  let alphabet_size : ℕ := 26
  let unique_letters : ℕ := 8
  let probability : ℚ := unique_letters / alphabet_size
  probability = 4 / 13 := by
sorry

end mathematics_letter_probability_l735_73505


namespace mark_work_hours_l735_73553

/-- Calculates the number of hours Mark needs to work per week to earn a target amount --/
def hours_per_week (spring_hours_per_week : ℚ) (spring_weeks : ℚ) (spring_earnings : ℚ) 
  (target_weeks : ℚ) (target_earnings : ℚ) : ℚ :=
  let hourly_wage := spring_earnings / (spring_hours_per_week * spring_weeks)
  let total_hours_needed := target_earnings / hourly_wage
  total_hours_needed / target_weeks

theorem mark_work_hours 
  (spring_hours_per_week : ℚ) (spring_weeks : ℚ) (spring_earnings : ℚ) 
  (target_weeks : ℚ) (target_earnings : ℚ) :
  spring_hours_per_week = 35 ∧ 
  spring_weeks = 15 ∧ 
  spring_earnings = 4200 ∧ 
  target_weeks = 50 ∧ 
  target_earnings = 21000 →
  hours_per_week spring_hours_per_week spring_weeks spring_earnings target_weeks target_earnings = 52.5 := by
  sorry

end mark_work_hours_l735_73553


namespace largest_digit_change_l735_73527

/-- The original incorrect sum -/
def incorrect_sum : ℕ := 1742

/-- The first addend in the original problem -/
def addend1 : ℕ := 789

/-- The second addend in the original problem -/
def addend2 : ℕ := 436

/-- The third addend in the original problem -/
def addend3 : ℕ := 527

/-- The corrected first addend after changing a digit -/
def corrected_addend1 : ℕ := 779

theorem largest_digit_change :
  (∃ (d : ℕ), d ≤ 9 ∧
    corrected_addend1 + addend2 + addend3 = incorrect_sum ∧
    d = (addend1 / 10) % 10 - (corrected_addend1 / 10) % 10 ∧
    ∀ (x y z : ℕ), x ≤ addend1 ∧ y ≤ addend2 ∧ z ≤ addend3 →
      x + y + z = incorrect_sum →
      (∃ (d' : ℕ), d' ≤ 9 ∧ d' = (addend1 / 10) % 10 - (x / 10) % 10) →
      d' ≤ d) :=
sorry

end largest_digit_change_l735_73527


namespace geometric_sequence_sum_l735_73532

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 6 * a 10 + a 3 * a 5 = 26 →
  a 5 * a 7 = 5 →
  a 4 + a 8 = 6 := by
sorry

end geometric_sequence_sum_l735_73532


namespace intersection_of_M_and_N_l735_73516

def M : Set Nat := {1, 2, 3}
def N : Set Nat := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by sorry

end intersection_of_M_and_N_l735_73516


namespace batsman_average_after_12th_inning_l735_73542

/-- Represents a batsman's performance --/
structure BatsmanPerformance where
  innings : ℕ
  runsInLastInning : ℕ
  averageIncrease : ℕ
  boundaries : ℕ
  strikeRate : ℚ

/-- Calculates the average after the last inning --/
def averageAfterLastInning (b : BatsmanPerformance) : ℚ :=
  (b.innings * b.averageIncrease + b.runsInLastInning) / b.innings

/-- Theorem stating the batsman's average after the 12th inning --/
theorem batsman_average_after_12th_inning (b : BatsmanPerformance)
  (h1 : b.innings = 12)
  (h2 : b.runsInLastInning = 60)
  (h3 : b.averageIncrease = 4)
  (h4 : b.boundaries ≥ 8)
  (h5 : b.strikeRate ≥ 130) :
  averageAfterLastInning b = 16 := by
  sorry

end batsman_average_after_12th_inning_l735_73542


namespace cylinder_sphere_volume_ratio_l735_73538

/-- Given a cylinder and a sphere with equal radii, if the ratio of their surface areas is m:n,
    then the ratio of their volumes is (6m - 3n) : 4n. -/
theorem cylinder_sphere_volume_ratio (R : ℝ) (H : ℝ) (m n : ℝ) (h_positive : R > 0 ∧ H > 0 ∧ m > 0 ∧ n > 0) :
  (2 * π * R^2 + 2 * π * R * H) / (4 * π * R^2) = m / n →
  (π * R^2 * H) / ((4/3) * π * R^3) = (6 * m - 3 * n) / (4 * n) := by
  sorry

end cylinder_sphere_volume_ratio_l735_73538


namespace function_determination_l735_73559

/-- Given a function f: ℝ → ℝ satisfying f(1/x) = 1/(x+1) for x ≠ 0 and x ≠ -1,
    prove that f(x) = x/(x+1) for x ≠ 0 and x ≠ -1 -/
theorem function_determination (f : ℝ → ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 → f (1/x) = 1/(x+1)) →
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 → f x = x/(x+1)) :=
by sorry

end function_determination_l735_73559


namespace fraction_sum_division_specific_fraction_sum_division_l735_73565

theorem fraction_sum_division (a b c d : ℚ) :
  (a / b + c / d) / 4 = (a * d + b * c) / (4 * b * d) :=
by sorry

theorem specific_fraction_sum_division :
  (2 / 5 + 1 / 3) / 4 = 11 / 60 :=
by sorry

end fraction_sum_division_specific_fraction_sum_division_l735_73565
