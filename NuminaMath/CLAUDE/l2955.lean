import Mathlib

namespace equation_solution_l2955_295581

theorem equation_solution : 
  ∃ (x : ℝ), x ≠ -2 ∧ (4*x^2 + 5*x + 2) / (x + 2) = 4*x - 2 ↔ x = 6 :=
by sorry

end equation_solution_l2955_295581


namespace g_composition_value_l2955_295519

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x^2 - 6

-- State the theorem
theorem g_composition_value : g (g 2) = 394 := by
  sorry

end g_composition_value_l2955_295519


namespace prime_solution_existence_l2955_295533

theorem prime_solution_existence (p : ℕ) : 
  Prime p ↔ (p = 2 ∨ p = 3 ∨ p = 7) ∧ 
  (∃ x y : ℕ+, x * (y^2 - p) + y * (x^2 - p) = 5 * p) :=
sorry

end prime_solution_existence_l2955_295533


namespace juans_number_l2955_295590

theorem juans_number (j k : ℕ) (h1 : j > 0) (h2 : k > 0) 
  (h3 : 10^(k+1) + 10*j + 1 - j = 14789) : j = 532 := by
  sorry

end juans_number_l2955_295590


namespace intersection_range_l2955_295543

def M : Set (ℝ × ℝ) := {p | p.2 = Real.sqrt (9 - p.1^2) ∧ p.2 ≠ 0}

def N (b : ℝ) : Set (ℝ × ℝ) := {p | p.2 = p.1 + b}

theorem intersection_range (b : ℝ) (h : (M ∩ N b).Nonempty) : b ∈ Set.Ioo (-3) (3 * Real.sqrt 2) := by
  sorry

end intersection_range_l2955_295543


namespace salad_dressing_total_l2955_295553

/-- Given a ratio of ingredients and the amount of one ingredient, 
    calculate the total amount of all ingredients. -/
def total_ingredients (ratio : List Nat) (water_amount : Nat) : Nat :=
  let total_parts := ratio.sum
  let part_size := water_amount / ratio.head!
  part_size * total_parts

/-- Theorem: For a salad dressing with a water:olive oil:salt ratio of 3:2:1 
    and 15 cups of water, the total amount of ingredients is 30 cups. -/
theorem salad_dressing_total : 
  total_ingredients [3, 2, 1] 15 = 30 := by
  sorry

#eval total_ingredients [3, 2, 1] 15

end salad_dressing_total_l2955_295553


namespace remainder_of_7_350_mod_43_l2955_295562

theorem remainder_of_7_350_mod_43 : 7^350 % 43 = 6 := by
  sorry

end remainder_of_7_350_mod_43_l2955_295562


namespace geometric_sequence_ratio_l2955_295540

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_geom : ∃ r, ∀ n, a (n + 1) = r * a n)
  (h_arith : a 1 + 2 * a 2 = a 3) :
  (a 9 + a 10) / (a 9 + a 8) = 1 + Real.sqrt 2 := by
sorry

end geometric_sequence_ratio_l2955_295540


namespace laura_savings_l2955_295501

def original_price : ℝ := 3.00
def discount_rate : ℝ := 0.30
def num_notebooks : ℕ := 7

theorem laura_savings : 
  (num_notebooks : ℝ) * original_price * discount_rate = 6.30 := by
  sorry

end laura_savings_l2955_295501


namespace no_valid_solution_l2955_295580

theorem no_valid_solution (total_days : ℕ) (present_pay absent_pay total_pay : ℚ) : 
  total_days = 60 ∧ 
  present_pay = 7 ∧ 
  absent_pay = 3 ∧ 
  total_pay = 170 → 
  ¬∃ (days_present : ℕ), 
    days_present ≤ total_days ∧ 
    (days_present : ℚ) * present_pay + (total_days - days_present : ℚ) * absent_pay = total_pay := by
  sorry

#check no_valid_solution

end no_valid_solution_l2955_295580


namespace solve_equation_l2955_295517

theorem solve_equation : ∃ x : ℝ, (0.5^3 - 0.1^3 / 0.5^2 + x + 0.1^2 = 0.4) ∧ (x = 0.269) := by
  sorry

end solve_equation_l2955_295517


namespace no_simple_condition_for_equality_l2955_295525

/-- There is no simple general condition for when a + b + c² = (a+b)(a+c) for all real numbers a, b, and c. -/
theorem no_simple_condition_for_equality (a b c : ℝ) : 
  ¬ ∃ (simple_condition : Prop), simple_condition ↔ (a + b + c^2 = (a+b)*(a+c)) :=
sorry

end no_simple_condition_for_equality_l2955_295525


namespace six_digit_palindrome_divisibility_l2955_295538

theorem six_digit_palindrome_divisibility (a b : Nat) (h1 : a ≥ 1) (h2 : a ≤ 9) (h3 : b ≤ 9) :
  let ab := 10 * a + b
  let ababab := 100000 * ab + 1000 * ab + ab
  ∃ (k1 k2 k3 k4 : Nat), ababab = 101 * k1 ∧ ababab = 7 * k2 ∧ ababab = 11 * k3 ∧ ababab = 13 * k4 := by
  sorry

end six_digit_palindrome_divisibility_l2955_295538


namespace fraction_sum_equals_decimal_l2955_295599

theorem fraction_sum_equals_decimal : 
  (3 : ℚ) / 30 + (5 : ℚ) / 300 + (7 : ℚ) / 3000 = 0.119 := by sorry

end fraction_sum_equals_decimal_l2955_295599


namespace solution_set_l2955_295536

theorem solution_set (m : ℤ) 
  (h1 : ∃! (x : ℤ), |2*x - m| ≤ 1 ∧ x = 2) :
  {x : ℝ | |x - 1| + |x - 3| ≥ m} = 
    {x : ℝ | x ≤ 0} ∪ {x : ℝ | x ≥ 4} := by
  sorry

end solution_set_l2955_295536


namespace hotel_reunion_attendees_l2955_295565

theorem hotel_reunion_attendees (total_guests dates_attendees hall_attendees : ℕ) 
  (h1 : total_guests = 50)
  (h2 : dates_attendees = 50)
  (h3 : hall_attendees = 60)
  (h4 : ∀ g, g ≤ total_guests → (g ≤ dates_attendees ∨ g ≤ hall_attendees)) :
  dates_attendees + hall_attendees - total_guests = 60 := by
  sorry

end hotel_reunion_attendees_l2955_295565


namespace base_16_to_binary_bits_l2955_295546

/-- The base-16 number represented as 66666 --/
def base_16_num : ℕ := 6 * 16^4 + 6 * 16^3 + 6 * 16^2 + 6 * 16 + 6

/-- The number of bits in the binary representation of a natural number --/
def num_bits (n : ℕ) : ℕ :=
  if n = 0 then 0 else Nat.log2 n + 1

theorem base_16_to_binary_bits :
  num_bits base_16_num = 19 := by
  sorry

end base_16_to_binary_bits_l2955_295546


namespace degree_of_g_is_two_l2955_295594

/-- The degree of a polynomial -/
noncomputable def degree (p : Polynomial ℝ) : ℕ := sorry

/-- Composition of polynomials -/
def compose (f g : Polynomial ℝ) : Polynomial ℝ := sorry

theorem degree_of_g_is_two
  (f g : Polynomial ℝ)
  (h : Polynomial ℝ)
  (h_def : h = compose f g + g)
  (deg_h : degree h = 8)
  (deg_f : degree f = 3) :
  degree g = 2 := by sorry

end degree_of_g_is_two_l2955_295594


namespace circle_line_intersection_l2955_295527

theorem circle_line_intersection (k : ℝ) : 
  k ≤ -2 * Real.sqrt 2 → 
  ∃ x y : ℝ, x^2 + y^2 = 1 ∧ y = k * x - 3 := by
  sorry

end circle_line_intersection_l2955_295527


namespace max_profit_at_90_l2955_295545

noncomputable section

-- Define the cost function
def cost (x : ℝ) : ℝ :=
  if x < 90 then 0.5 * x^2 + 60 * x + 5
  else 121 * x + 8100 / x - 2180 + 5

-- Define the revenue function
def revenue (x : ℝ) : ℝ := 1.2 * x

-- Define the profit function
def profit (x : ℝ) : ℝ := revenue x - cost x

-- Theorem statement
theorem max_profit_at_90 :
  ∀ x > 0, profit x ≤ profit 90 ∧ profit 90 = 1500 := by sorry

end

end max_profit_at_90_l2955_295545


namespace polynomial_factorization_sum_l2955_295566

theorem polynomial_factorization_sum (a₁ a₂ c₁ b₂ c₂ : ℝ) 
  (h : ∀ x : ℝ, x^5 - x^4 + x^3 - x^2 + x - 1 = (x^3 + a₁*x^2 + a₂*x + c₁)*(x^2 + b₂*x + c₂)) :
  a₁*c₁ + b₂*c₂ = -1 := by
  sorry

end polynomial_factorization_sum_l2955_295566


namespace discount_percentage_proof_l2955_295567

theorem discount_percentage_proof (num_toys : ℕ) (cost_per_toy : ℚ) (total_paid : ℚ) :
  num_toys = 5 →
  cost_per_toy = 3 →
  total_paid = 12 →
  (1 - total_paid / (num_toys * cost_per_toy)) * 100 = 20 := by
  sorry

end discount_percentage_proof_l2955_295567


namespace inequality_solution_l2955_295586

theorem inequality_solution (x : ℝ) :
  3 * x^2 - 2 * x ≥ 9 ↔ x ≤ -1 ∨ x ≥ 3 := by sorry

end inequality_solution_l2955_295586


namespace round_repeating_decimal_to_hundredth_l2955_295579

-- Define the repeating decimal
def repeating_decimal : ℚ := 82 + 367 / 999

-- Define the rounding function to the nearest hundredth
def round_to_hundredth (x : ℚ) : ℚ := 
  (⌊x * 100 + 0.5⌋ : ℚ) / 100

-- Theorem statement
theorem round_repeating_decimal_to_hundredth :
  round_to_hundredth repeating_decimal = 82.37 := by sorry

end round_repeating_decimal_to_hundredth_l2955_295579


namespace product_of_sums_l2955_295578

theorem product_of_sums : (-1-2-3-4-5-6-7-8-9-10) * (1-2+3-4+5-6+7-8+9-10) = 275 := by
  sorry

end product_of_sums_l2955_295578


namespace evaluate_expression_l2955_295560

theorem evaluate_expression : (-1 : ℤ) ^ (6 ^ 2) + (1 : ℤ) ^ (3 ^ 4) = 2 := by
  sorry

end evaluate_expression_l2955_295560


namespace number_of_cut_cubes_l2955_295518

/-- Represents a cube with integer edge length -/
structure Cube where
  edge : ℕ

/-- Represents a set of cubes resulting from cutting a larger cube -/
structure CutCube where
  original : Cube
  pieces : List Cube
  all_same_size : Bool

/-- The volume of a cube -/
def volume (c : Cube) : ℕ := c.edge ^ 3

/-- The total volume of a list of cubes -/
def total_volume (cubes : List Cube) : ℕ :=
  cubes.map volume |>.sum

/-- Theorem: The number of smaller cubes obtained by cutting a 4cm cube is 57 -/
theorem number_of_cut_cubes : ∃ (cut : CutCube), 
  cut.original.edge = 4 ∧ 
  cut.all_same_size = false ∧
  (∀ c ∈ cut.pieces, c.edge > 0) ∧
  total_volume cut.pieces = volume cut.original ∧
  cut.pieces.length = 57 := by
  sorry

end number_of_cut_cubes_l2955_295518


namespace minimum_occupied_seats_l2955_295523

theorem minimum_occupied_seats (total_seats : ℕ) (h : total_seats = 120) :
  let min_occupied := (total_seats + 2) / 3
  min_occupied = 40 ∧
  ∀ n : ℕ, n < min_occupied → ∃ i : ℕ, i < total_seats ∧ 
    (∀ j : ℕ, j < total_seats → (j = i ∨ j = i + 1) → n ≤ j) :=
by sorry

end minimum_occupied_seats_l2955_295523


namespace arithmetic_sequence_11th_term_l2955_295535

/-- An arithmetic sequence {aₙ} where a₃ = 4 and a₅ = 8 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a 3 = 4 ∧ a 5 = 8

/-- The 11th term of the arithmetic sequence is 20 -/
theorem arithmetic_sequence_11th_term (a : ℕ → ℝ) 
  (h : arithmetic_sequence a) : a 11 = 20 := by
  sorry

end arithmetic_sequence_11th_term_l2955_295535


namespace diophantine_equation_solutions_l2955_295596

def is_solution (x y w : ℕ) : Prop :=
  2^x * 3^y - 5^x * 7^w = 1

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(1, 0, 0), (3, 0, 1), (1, 1, 0), (2, 2, 1)}

theorem diophantine_equation_solutions :
  ∀ x y w : ℕ, is_solution x y w ↔ (x, y, w) ∈ solution_set := by
  sorry

end diophantine_equation_solutions_l2955_295596


namespace museum_trip_cost_l2955_295551

/-- The total cost of entrance tickets for a group of students and teachers -/
def total_cost (num_students : ℕ) (num_teachers : ℕ) (ticket_price : ℕ) : ℕ :=
  (num_students + num_teachers) * ticket_price

/-- Theorem: The total cost for 20 students and 3 teachers with $5 tickets is $115 -/
theorem museum_trip_cost : total_cost 20 3 5 = 115 := by
  sorry

end museum_trip_cost_l2955_295551


namespace stream_speed_l2955_295532

/-- Prove that the speed of the stream is 4 km/hr given the conditions of the boat problem -/
theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (time : ℝ) (h1 : boat_speed = 24)
  (h2 : distance = 56) (h3 : time = 2) : 
  (distance / time - boat_speed) = 4 := by
  sorry

end stream_speed_l2955_295532


namespace f_inequality_l2955_295571

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem f_inequality (hf : Differentiable ℝ f) 
  (h : ∀ x, f x > deriv f x) : 
  f 2013 < Real.exp 2013 * f 0 := by
  sorry

end f_inequality_l2955_295571


namespace tank_fill_time_main_theorem_l2955_295570

-- Define the fill rates of the pipes
def fill_rate_1 : ℚ := 1 / 18
def fill_rate_2 : ℚ := 1 / 60
def empty_rate : ℚ := 1 / 45

-- Define the combined rate
def combined_rate : ℚ := fill_rate_1 + fill_rate_2 - empty_rate

-- Theorem statement
theorem tank_fill_time :
  combined_rate = 1 / 20 := by sorry

-- Time to fill the tank
def fill_time : ℚ := 1 / combined_rate

-- Main theorem
theorem main_theorem :
  fill_time = 20 := by sorry

end tank_fill_time_main_theorem_l2955_295570


namespace repeating_decimal_value_l2955_295542

/-- The decimal representation 0.7overline{23}15 as a rational number -/
def repeating_decimal : ℚ := 62519 / 66000

/-- Theorem stating that 0.7overline{23}15 is equal to 62519/66000 -/
theorem repeating_decimal_value : repeating_decimal = 0.7 + 0.015 + (23 : ℚ) / 990 := by
  sorry

#eval repeating_decimal

end repeating_decimal_value_l2955_295542


namespace max_crates_first_trip_solution_l2955_295512

/-- The maximum number of crates that can be carried in the first part of the trip -/
def max_crates_first_trip (total_crates : ℕ) (min_crate_weight : ℕ) (max_trip_weight : ℕ) : ℕ :=
  min (total_crates) (max_trip_weight / min_crate_weight)

theorem max_crates_first_trip_solution :
  max_crates_first_trip 12 120 600 = 5 := by
  sorry

end max_crates_first_trip_solution_l2955_295512


namespace speed_increase_ratio_l2955_295544

theorem speed_increase_ratio (v : ℝ) (h : (v + 2) / v = 2.5) :
  (v + 4) / v = 4 := by
  sorry

end speed_increase_ratio_l2955_295544


namespace choose_three_from_nine_l2955_295503

theorem choose_three_from_nine : Nat.choose 9 3 = 84 := by
  sorry

end choose_three_from_nine_l2955_295503


namespace roque_bike_trips_l2955_295514

/-- Represents the number of times Roque rides his bike to and from work per week -/
def bike_trips : ℕ := 2

/-- Represents the time it takes Roque to walk to work one way (in hours) -/
def walk_time : ℕ := 2

/-- Represents the time it takes Roque to bike to work one way (in hours) -/
def bike_time : ℕ := 1

/-- Represents the number of times Roque walks to and from work per week -/
def walk_trips : ℕ := 3

/-- Represents the total time Roque spends commuting per week (in hours) -/
def total_commute_time : ℕ := 16

theorem roque_bike_trips :
  walk_trips * (2 * walk_time) + bike_trips * (2 * bike_time) = total_commute_time :=
by sorry

end roque_bike_trips_l2955_295514


namespace hydropump_volume_l2955_295573

/-- Represents the rate of water pumping in gallons per hour -/
def pump_rate : ℝ := 600

/-- Represents the time in hours -/
def pump_time : ℝ := 1.5

/-- Represents the volume of water pumped in gallons -/
def water_volume : ℝ := pump_rate * pump_time

theorem hydropump_volume : water_volume = 900 := by
  sorry

end hydropump_volume_l2955_295573


namespace tangent_y_intercept_l2955_295515

-- Define the circles
def circle1_center : ℝ × ℝ := (3, 0)
def circle1_radius : ℝ := 3
def circle2_center : ℝ × ℝ := (8, 0)
def circle2_radius : ℝ := 2

-- Define the tangent line
def tangent_line : Set (ℝ × ℝ) := sorry

-- Define the property of being tangent to a circle in the fourth quadrant
def is_tangent_in_fourth_quadrant (line : Set (ℝ × ℝ)) (center : ℝ × ℝ) (radius : ℝ) : Prop := sorry

-- Theorem statement
theorem tangent_y_intercept :
  is_tangent_in_fourth_quadrant tangent_line circle1_center circle1_radius ∧
  is_tangent_in_fourth_quadrant tangent_line circle2_center circle2_radius →
  ∃ (y : ℝ), y = 6/5 ∧ (0, y) ∈ tangent_line :=
sorry

end tangent_y_intercept_l2955_295515


namespace jade_stone_volume_sum_l2955_295575

-- Define the weights per cubic inch
def jade_weight_per_cubic_inch : ℝ := 7
def stone_weight_per_cubic_inch : ℝ := 6

-- Define the edge length of the cubic stone
def edge_length : ℝ := 3

-- Define the total weight in taels
def total_weight : ℝ := 176

-- Theorem statement
theorem jade_stone_volume_sum (x y : ℝ) 
  (h1 : x + y = total_weight) 
  (h2 : x ≥ 0) 
  (h3 : y ≥ 0) : 
  x / jade_weight_per_cubic_inch + y / stone_weight_per_cubic_inch = edge_length ^ 3 := by
  sorry

end jade_stone_volume_sum_l2955_295575


namespace product_97_103_l2955_295506

theorem product_97_103 : 97 * 103 = 9991 := by
  sorry

end product_97_103_l2955_295506


namespace polynomial_division_theorem_l2955_295522

theorem polynomial_division_theorem (x : ℝ) : 
  (4 * x^3 + x^2 + 2 * x + 3) * (3 * x - 2) + 11 = 
  12 * x^4 - 9 * x^3 + 6 * x^2 + 11 * x - 3 := by sorry

end polynomial_division_theorem_l2955_295522


namespace sqrt_product_equality_l2955_295554

theorem sqrt_product_equality : Real.sqrt 72 * Real.sqrt 18 * Real.sqrt 8 = 72 * Real.sqrt 2 := by
  sorry

end sqrt_product_equality_l2955_295554


namespace solve_quadratic_equation_l2955_295559

theorem solve_quadratic_equation (B : ℝ) : 3 * B^2 + 3 * B + 2 = 29 →
  B = (-1 + Real.sqrt 37) / 2 ∨ B = (-1 - Real.sqrt 37) / 2 := by
  sorry

end solve_quadratic_equation_l2955_295559


namespace count_pairs_theorem_l2955_295555

def X : Finset Nat := Finset.range 10

def intersection_set : Finset Nat := {5, 7, 8}

def count_valid_pairs (X : Finset Nat) (intersection_set : Finset Nat) : Nat :=
  let remaining_elements := X \ intersection_set
  3^(remaining_elements.card) - 1

theorem count_pairs_theorem (X : Finset Nat) (intersection_set : Finset Nat) :
  X = Finset.range 10 →
  intersection_set = {5, 7, 8} →
  count_valid_pairs X intersection_set = 2186 := by
  sorry

end count_pairs_theorem_l2955_295555


namespace custom_op_example_l2955_295552

-- Define the custom operation
def custom_op (a b : ℝ) : ℝ := a - 2 * b

-- State the theorem
theorem custom_op_example : custom_op 2 (-3) = 8 := by
  sorry

end custom_op_example_l2955_295552


namespace sequence_problem_l2955_295549

/-- Given a sequence {aₙ} that satisfies the recurrence relation
    aₙ₊₁/(n+1) = aₙ/n for all n, and a₅ = 15, prove that a₈ = 24. -/
theorem sequence_problem (a : ℕ → ℚ)
    (h1 : ∀ n, a (n + 1) / (n + 1) = a n / n)
    (h2 : a 5 = 15) :
    a 8 = 24 := by
  sorry

end sequence_problem_l2955_295549


namespace power_congruence_l2955_295593

theorem power_congruence (h : 2^200 ≡ 1 [MOD 800]) : 2^6000 ≡ 1 [MOD 800] := by
  sorry

end power_congruence_l2955_295593


namespace license_plate_theorem_l2955_295548

def vowels : Nat := 5
def consonants : Nat := 21
def odd_digits : Nat := 5
def even_digits : Nat := 5

def license_plate_count : Nat :=
  (vowels^2 + consonants^2) * odd_digits * even_digits^2

theorem license_plate_theorem :
  license_plate_count = 58250 := by
  sorry

end license_plate_theorem_l2955_295548


namespace temp_difference_l2955_295587

-- Define the temperatures
def southern_temp : Int := -7
def northern_temp : Int := -15

-- State the theorem
theorem temp_difference : southern_temp - northern_temp = 8 := by
  sorry

end temp_difference_l2955_295587


namespace equal_even_odd_probability_l2955_295583

/-- The number of dice being rolled -/
def n : ℕ := 8

/-- The probability of rolling an even number on a single die -/
def p_even : ℚ := 1/2

/-- The probability of rolling an odd number on a single die -/
def p_odd : ℚ := 1/2

/-- The number of ways to choose half of the dice -/
def ways_to_choose : ℕ := n.choose (n/2)

theorem equal_even_odd_probability :
  (ways_to_choose : ℚ) * p_even^(n/2) * p_odd^(n/2) = 35/128 := by sorry

end equal_even_odd_probability_l2955_295583


namespace max_value_expression_l2955_295513

theorem max_value_expression (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) 
  (h_sum : a^2 + b^2 + c^2 = 3) : 
  3 * a * b * Real.sqrt 2 + 9 * b * c ≤ 3 * Real.sqrt 11 := by
  sorry

end max_value_expression_l2955_295513


namespace inscribed_quadrilateral_exists_l2955_295589

/-- A quadrilateral inscribed in a circle -/
structure InscribedQuadrilateral where
  /-- Side lengths of the quadrilateral -/
  sides : Fin 4 → ℕ+
  /-- Diagonal lengths of the quadrilateral -/
  diagonals : Fin 2 → ℕ+
  /-- Area of the quadrilateral -/
  area : ℕ+
  /-- Radius of the circumcircle -/
  radius : ℕ+
  /-- The quadrilateral is inscribed in a circle -/
  inscribed : True
  /-- The side lengths are pairwise distinct -/
  distinct_sides : ∀ i j, i ≠ j → sides i ≠ sides j

/-- There exists an inscribed quadrilateral with integer parameters -/
theorem inscribed_quadrilateral_exists : 
  ∃ q : InscribedQuadrilateral, True :=
sorry

end inscribed_quadrilateral_exists_l2955_295589


namespace cos_75_minus_cos_15_l2955_295563

theorem cos_75_minus_cos_15 : Real.cos (75 * π / 180) - Real.cos (15 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_75_minus_cos_15_l2955_295563


namespace smallest_a_in_special_progression_l2955_295521

theorem smallest_a_in_special_progression (a b c : ℤ) 
  (h1 : a < c ∧ c < b)
  (h2 : 2 * c = a + b)  -- arithmetic progression condition
  (h3 : b * b = a * c)  -- geometric progression condition
  : a ≥ -4 ∧ ∃ (a₀ b₀ c₀ : ℤ), a₀ = -4 ∧ b₀ = 2 ∧ c₀ = -1 ∧ 
    a₀ < c₀ ∧ c₀ < b₀ ∧ 
    2 * c₀ = a₀ + b₀ ∧ 
    b₀ * b₀ = a₀ * c₀ :=
by sorry

end smallest_a_in_special_progression_l2955_295521


namespace abs_x_minus_sqrt_x_minus_one_squared_l2955_295558

theorem abs_x_minus_sqrt_x_minus_one_squared (x : ℝ) (h : x < 0) :
  |x - Real.sqrt ((x - 1)^2)| = 1 - 2*x := by
  sorry

end abs_x_minus_sqrt_x_minus_one_squared_l2955_295558


namespace parabola_intersection_point_l2955_295576

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluates the y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.evaluate (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_intersection_point (p : Parabola) (h1 : p.a = 1 ∧ p.b = -2 ∧ p.c = -3)
    (h2 : p.evaluate (-1) = 0) :
    p.evaluate 3 = 0 := by
  sorry

end parabola_intersection_point_l2955_295576


namespace bulls_win_in_seven_games_l2955_295591

def probability_heat_wins : ℚ := 3/4

def games_to_win : ℕ := 4

def total_games : ℕ := 7

theorem bulls_win_in_seven_games :
  let probability_bulls_wins := 1 - probability_heat_wins
  let combinations := Nat.choose (total_games - 1) (games_to_win - 1)
  let probability_tied_after_six := combinations * (probability_bulls_wins ^ (games_to_win - 1)) * (probability_heat_wins ^ (games_to_win - 1))
  let probability_bulls_win_last := probability_bulls_wins
  probability_tied_after_six * probability_bulls_win_last = 540/16384 := by
  sorry

end bulls_win_in_seven_games_l2955_295591


namespace tan_alpha_plus_pi_third_l2955_295561

theorem tan_alpha_plus_pi_third (α β : ℝ) 
  (h1 : Real.tan (α + β) = 3/5)
  (h2 : Real.tan (β - π/3) = 1/4) :
  Real.tan (α + π/3) = 7/23 := by
  sorry

end tan_alpha_plus_pi_third_l2955_295561


namespace f_properties_l2955_295557

def f (x : ℝ) : ℝ := (x - 2)^2

theorem f_properties :
  (∀ x, f (x + 2) = f (-x + 2)) ∧ 
  (∀ x y, x < y → x < 2 → f x > f y) ∧
  (∀ x y, x < y → y > 2 → f x < f y) ∧
  (∀ x y, x < y → f (x + 2) - f x < f (y + 2) - f y) := by
  sorry

end f_properties_l2955_295557


namespace director_sphere_theorem_l2955_295585

/-- The surface S: ax^2 + by^2 + cz^2 = 1 -/
def S (a b c : ℝ) (x y z : ℝ) : Prop :=
  a * x^2 + b * y^2 + c * z^2 = 1

/-- The director sphere K: x^2 + y^2 + z^2 = 1/a + 1/b + 1/c -/
def K (a b c : ℝ) (x y z : ℝ) : Prop :=
  x^2 + y^2 + z^2 = 1/a + 1/b + 1/c

/-- A plane tangent to S at point (u, v, w) -/
def tangent_plane (a b c : ℝ) (u v w x y z : ℝ) : Prop :=
  a * u * x + b * v * y + c * w * z = 1

/-- Three mutually perpendicular planes -/
def perpendicular_planes (p₁ q₁ r₁ p₂ q₂ r₂ p₃ q₃ r₃ : ℝ) : Prop :=
  p₁ * p₂ + q₁ * q₂ + r₁ * r₂ = 0 ∧
  p₁ * p₃ + q₁ * q₃ + r₁ * r₃ = 0 ∧
  p₂ * p₃ + q₂ * q₃ + r₂ * r₃ = 0

theorem director_sphere_theorem (a b c : ℝ) (x₀ y₀ z₀ : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (∃ u₁ v₁ w₁ u₂ v₂ w₂ u₃ v₃ w₃ : ℝ,
    S a b c u₁ v₁ w₁ ∧ S a b c u₂ v₂ w₂ ∧ S a b c u₃ v₃ w₃ ∧
    tangent_plane a b c u₁ v₁ w₁ x₀ y₀ z₀ ∧
    tangent_plane a b c u₂ v₂ w₂ x₀ y₀ z₀ ∧
    tangent_plane a b c u₃ v₃ w₃ x₀ y₀ z₀ ∧
    perpendicular_planes 
      (a * u₁) (b * v₁) (c * w₁)
      (a * u₂) (b * v₂) (c * w₂)
      (a * u₃) (b * v₃) (c * w₃)) →
  K a b c x₀ y₀ z₀ := by
  sorry

end director_sphere_theorem_l2955_295585


namespace cube_triangle_area_sum_l2955_295577

/-- Represents a 2x2x2 cube -/
structure Cube :=
  (side : ℝ)
  (is_two_by_two_by_two : side = 2)

/-- Represents a triangle with vertices on the cube -/
structure CubeTriangle :=
  (cube : Cube)
  (vertices : Fin 3 → Fin 8)

/-- The area of a triangle on the cube -/
noncomputable def triangleArea (t : CubeTriangle) : ℝ := sorry

/-- The sum of areas of all triangles on the cube -/
noncomputable def totalArea (c : Cube) : ℝ := sorry

/-- The representation of the total area in the form m + √n + √p -/
structure AreaRepresentation (c : Cube) :=
  (m n p : ℕ)
  (total_area_eq : totalArea c = m + Real.sqrt n + Real.sqrt p)

theorem cube_triangle_area_sum (c : Cube) (rep : AreaRepresentation c) :
  rep.m + rep.n + rep.p = 11584 := by sorry

end cube_triangle_area_sum_l2955_295577


namespace min_odd_in_A_P_l2955_295529

/-- A polynomial of degree 8 -/
def Polynomial8 : Type := ℝ → ℝ

/-- The set A_P for a polynomial P -/
def A_P (P : Polynomial8) (c : ℝ) : Set ℝ :=
  {x : ℝ | P x = c}

/-- Theorem: For any polynomial P of degree 8, if 8 is in A_P, then A_P contains at least one odd number -/
theorem min_odd_in_A_P (P : Polynomial8) (h : 8 ∈ A_P P (P 8)) :
  ∃ (x : ℝ), x ∈ A_P P (P 8) ∧ ∃ (n : ℤ), x = 2 * n + 1 :=
sorry

end min_odd_in_A_P_l2955_295529


namespace unique_function_satisfying_inequality_l2955_295511

theorem unique_function_satisfying_inequality (a c d : ℝ) :
  ∃! f : ℝ → ℝ, ∀ x : ℝ, f (a * x + c) + d ≤ x ∧ x ≤ f (x + d) + c :=
by
  -- The proof goes here
  sorry

end unique_function_satisfying_inequality_l2955_295511


namespace min_value_reciprocal_sum_l2955_295528

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 4 := by
sorry

end min_value_reciprocal_sum_l2955_295528


namespace jennie_drive_time_l2955_295504

def drive_time_proof (distance : ℝ) (time_with_traffic : ℝ) (speed_difference : ℝ) : Prop :=
  let speed_with_traffic := distance / time_with_traffic
  let speed_no_traffic := speed_with_traffic + speed_difference
  let time_no_traffic := distance / speed_no_traffic
  distance = 200 ∧ time_with_traffic = 5 ∧ speed_difference = 10 →
  time_no_traffic = 4

theorem jennie_drive_time : drive_time_proof 200 5 10 := by
  sorry

end jennie_drive_time_l2955_295504


namespace quadratic_equal_roots_l2955_295556

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + m = 1 ∧ 
   ∀ y : ℝ, y^2 - 4*y + m = 1 → y = x) ↔ 
  m = 5 := by
sorry

end quadratic_equal_roots_l2955_295556


namespace angle_bisector_length_l2955_295509

/-- The length of an angle bisector in a triangle -/
theorem angle_bisector_length (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (htri : a < b + c ∧ b < a + c ∧ c < a + b) :
  let p := (a + b + c) / 2
  ∃ l_a : ℝ, l_a = (2 / (b + c)) * Real.sqrt (b * c * p * (p - a)) := by
  sorry

end angle_bisector_length_l2955_295509


namespace hot_dog_stand_sales_time_l2955_295598

/-- 
Given a hot dog stand that sells 10 hot dogs per hour at $2 each,
prove that it takes 10 hours to reach $200 in sales.
-/
theorem hot_dog_stand_sales_time : 
  let hot_dogs_per_hour : ℕ := 10
  let price_per_hot_dog : ℚ := 2
  let sales_goal : ℚ := 200
  let sales_per_hour : ℚ := hot_dogs_per_hour * price_per_hot_dog
  let hours_needed : ℚ := sales_goal / sales_per_hour
  hours_needed = 10 := by sorry

end hot_dog_stand_sales_time_l2955_295598


namespace triangle_area_ratio_bounds_l2955_295588

theorem triangle_area_ratio_bounds (a b c r R : ℝ) (S S₁ : ℝ) :
  a > 0 → b > 0 → c > 0 → r > 0 → R > 0 → S > 0 →
  6 * (a + b + c) * r^2 = a * b * c →
  R = 3 * r →
  S = (r * (a + b + c)) / 2 →
  ∃ (M : ℝ × ℝ), 
    (5 - 2 * Real.sqrt 3) / 36 ≤ S₁ / S ∧ 
    S₁ / S ≤ (5 + 2 * Real.sqrt 3) / 36 :=
by sorry

end triangle_area_ratio_bounds_l2955_295588


namespace factorization_problem_1_factorization_problem_2_l2955_295526

-- Problem 1
theorem factorization_problem_1 (x y : ℝ) :
  -10 * x * y^2 + y^3 + 25 * x^2 * y = y * (5 * x - y)^2 := by sorry

-- Problem 2
theorem factorization_problem_2 (a b : ℝ) :
  a^3 + a^2 * b - a * b^2 - b^3 = (a + b)^2 * (a - b) := by sorry

end factorization_problem_1_factorization_problem_2_l2955_295526


namespace integer_roots_of_polynomial_l2955_295572

def polynomial (b₂ b₁ : ℤ) (x : ℤ) : ℤ := x^3 + b₂ * x^2 + b₁ * x - 30

def divisors_of_30 : Set ℤ := {-30, -15, -10, -6, -5, -3, -2, -1, 1, 2, 3, 5, 6, 10, 15, 30}

theorem integer_roots_of_polynomial (b₂ b₁ : ℤ) :
  {x : ℤ | polynomial b₂ b₁ x = 0} = divisors_of_30 := by sorry

end integer_roots_of_polynomial_l2955_295572


namespace second_candidate_votes_l2955_295510

theorem second_candidate_votes (total_votes : ℕ) (first_candidate_percentage : ℚ) : 
  total_votes = 600 → 
  first_candidate_percentage = 60 / 100 → 
  (total_votes : ℚ) * (1 - first_candidate_percentage) = 240 := by
  sorry

end second_candidate_votes_l2955_295510


namespace h_equation_l2955_295584

/-- Given the equation 4x^4 + 2x^2 - 5x + 1 + h(x) = x^3 - 3x^2 + 2x - 4,
    prove that h(x) = -4x^4 + x^3 - 5x^2 + 7x - 5 -/
theorem h_equation (x : ℝ) (h : ℝ → ℝ) 
    (eq : 4 * x^4 + 2 * x^2 - 5 * x + 1 + h x = x^3 - 3 * x^2 + 2 * x - 4) : 
  h x = -4 * x^4 + x^3 - 5 * x^2 + 7 * x - 5 := by
  sorry

end h_equation_l2955_295584


namespace moles_of_HCN_l2955_295537

-- Define the reaction components
structure Reaction where
  CuSO4 : ℝ
  HCN : ℝ
  Cu_CN_2 : ℝ
  H2SO4 : ℝ

-- Define the balanced equation
def balanced_equation (r : Reaction) : Prop :=
  r.CuSO4 = r.Cu_CN_2 ∧ r.HCN = 4 * r.CuSO4 ∧ r.H2SO4 = r.CuSO4

-- Define the given conditions
def given_conditions (r : Reaction) : Prop :=
  r.CuSO4 = 1 ∧ r.Cu_CN_2 = 1

-- Theorem to prove
theorem moles_of_HCN (r : Reaction) 
  (h1 : balanced_equation r) 
  (h2 : given_conditions r) : 
  r.HCN = 4 :=
sorry

end moles_of_HCN_l2955_295537


namespace union_of_A_and_complement_of_B_l2955_295539

open Set

def U : Set ℝ := univ

def A : Set ℝ := {x | |x - 1| < 1}

def B : Set ℝ := {x | x < 1 ∨ x ≥ 4}

theorem union_of_A_and_complement_of_B :
  A ∪ (U \ B) = {x : ℝ | 0 < x ∧ x < 4} :=
by sorry

end union_of_A_and_complement_of_B_l2955_295539


namespace prime_gap_2015_l2955_295500

theorem prime_gap_2015 : ∃ p q : ℕ, 
  Prime p ∧ Prime q ∧ p < q ∧ q - p > 2015 ∧ 
  ∀ k : ℕ, p < k ∧ k < q → ¬(Prime k) :=
sorry

end prime_gap_2015_l2955_295500


namespace min_value_of_a_l2955_295541

-- Define the set of x values
def X : Set ℝ := { x | 0 < x ∧ x ≤ 1/2 }

-- Define the inequality condition
def inequality_holds (a : ℝ) : Prop :=
  ∀ x ∈ X, x^2 + a*x + 1 ≥ 0

-- State the theorem
theorem min_value_of_a :
  (∃ a_min : ℝ, inequality_holds a_min ∧
    ∀ a : ℝ, inequality_holds a → a ≥ a_min) ∧
  (∀ a_min : ℝ, (inequality_holds a_min ∧
    ∀ a : ℝ, inequality_holds a → a ≥ a_min) →
    a_min = -5/2) :=
sorry

end min_value_of_a_l2955_295541


namespace quadratic_function_difference_l2955_295520

/-- A quadratic function with the property g(x+1) - g(x) = 2x + 3 for all real x -/
def g_property (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (x + 1) - g x = 2 * x + 3

theorem quadratic_function_difference (g : ℝ → ℝ) (h : g_property g) : 
  g 2 - g 6 = -40 := by
  sorry

end quadratic_function_difference_l2955_295520


namespace curve_sum_invariant_under_translation_l2955_295507

-- Define a type for points in a plane
variable (P : Type) [AddCommGroup P]

-- Define a type for convex curves in the plane
variable (Curve : Type) [AddCommGroup Curve]

-- Define a parallel translation operation
variable (T : P → P)

-- Define an operation to apply translation to a curve
variable (applyTranslation : (P → P) → Curve → Curve)

-- Define a sum operation for curves
variable (curveSum : Curve → Curve → Curve)

-- Define a congruence relation for curves
variable (congruent : Curve → Curve → Prop)

-- Statement of the theorem
theorem curve_sum_invariant_under_translation 
  (K₁ K₂ : Curve) :
  congruent (curveSum K₁ K₂) (curveSum (applyTranslation T K₁) (applyTranslation T K₂)) :=
sorry

end curve_sum_invariant_under_translation_l2955_295507


namespace largest_even_three_digit_number_with_conditions_l2955_295568

theorem largest_even_three_digit_number_with_conditions :
  ∃ (x : ℕ), 
    x = 972 ∧
    x % 2 = 0 ∧
    100 ≤ x ∧ x < 1000 ∧
    x % 5 = 2 ∧
    Nat.gcd 30 (Nat.gcd x 15) = 3 ∧
    ∀ (y : ℕ), 
      y % 2 = 0 → 
      100 ≤ y → y < 1000 → 
      y % 5 = 2 → 
      Nat.gcd 30 (Nat.gcd y 15) = 3 → 
      y ≤ x :=
by sorry

end largest_even_three_digit_number_with_conditions_l2955_295568


namespace negative_number_identification_l2955_295574

theorem negative_number_identification (a b c d : ℝ) 
  (ha : a = -6) (hb : b = 0) (hc : c = 0.2) (hd : d = 3) :
  a < 0 ∧ b ≥ 0 ∧ c > 0 ∧ d > 0 :=
by sorry

end negative_number_identification_l2955_295574


namespace division_problem_l2955_295592

theorem division_problem (n : ℕ) : n / 4 = 12 → n / 3 = 16 := by
  sorry

end division_problem_l2955_295592


namespace distance_from_origin_l2955_295569

theorem distance_from_origin (x y n : ℝ) : 
  x > 1 →
  y = 8 →
  (x - 1)^2 + (y - 6)^2 = 12^2 →
  n^2 = x^2 + y^2 →
  n = Real.sqrt (205 + 2 * Real.sqrt 140) :=
by sorry

end distance_from_origin_l2955_295569


namespace milk_chocolate_caramel_percentage_l2955_295530

/-- The percentage of milk chocolate with caramel bars in a box of chocolates -/
theorem milk_chocolate_caramel_percentage
  (milk : ℕ)
  (dark : ℕ)
  (milk_almond : ℕ)
  (white : ℕ)
  (milk_caramel : ℕ)
  (h_milk : milk = 36)
  (h_dark : dark = 21)
  (h_milk_almond : milk_almond = 40)
  (h_white : white = 15)
  (h_milk_caramel : milk_caramel = 28) :
  (milk_caramel : ℚ) / (milk + dark + milk_almond + white + milk_caramel) = 1/5 := by
  sorry

end milk_chocolate_caramel_percentage_l2955_295530


namespace circle_area_circumscribed_square_l2955_295547

theorem circle_area_circumscribed_square (s : ℝ) (h : s = 12) :
  let r := s * Real.sqrt 2 / 2
  π * r^2 = 72 * π := by sorry

end circle_area_circumscribed_square_l2955_295547


namespace intersection_of_A_and_B_l2955_295597

def A : Set Int := {-1, 1, 2, 4}
def B : Set Int := {-1, 0, 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 2} := by sorry

end intersection_of_A_and_B_l2955_295597


namespace binomial_expansion_sum_l2955_295516

theorem binomial_expansion_sum (a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (a - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₂ = 80 →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ = 1 :=
by sorry

end binomial_expansion_sum_l2955_295516


namespace gcd_n_power_7_minus_n_l2955_295505

theorem gcd_n_power_7_minus_n (n : ℤ) : 42 ∣ (n^7 - n) := by
  sorry

end gcd_n_power_7_minus_n_l2955_295505


namespace sqrt_x_minus_5_real_implies_x_geq_5_l2955_295508

theorem sqrt_x_minus_5_real_implies_x_geq_5 (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 5) → x ≥ 5 := by sorry

end sqrt_x_minus_5_real_implies_x_geq_5_l2955_295508


namespace alligator_population_after_one_year_l2955_295582

/-- The number of alligators after a given number of doubling periods -/
def alligator_population (initial_population : ℕ) (doubling_periods : ℕ) : ℕ :=
  initial_population * 2^doubling_periods

/-- Theorem: After one year (two doubling periods), 
    the alligator population will be 16 given an initial population of 4 -/
theorem alligator_population_after_one_year :
  alligator_population 4 2 = 16 := by
  sorry

end alligator_population_after_one_year_l2955_295582


namespace range_of_f_l2955_295531

def f (x : ℝ) : ℝ := x^2 - 4*x

theorem range_of_f :
  {y : ℝ | ∃ x : ℝ, x ∈ Set.Icc (-3) 3 ∧ f x = y} = Set.Icc (-4) 21 :=
by sorry

end range_of_f_l2955_295531


namespace min_value_expression_equality_condition_l2955_295550

theorem min_value_expression (x : ℝ) : 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) ≥ 2 * Real.sqrt 5 :=
by sorry

theorem equality_condition : 
  ∃ x : ℝ, Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) = 2 * Real.sqrt 5 :=
by sorry

end min_value_expression_equality_condition_l2955_295550


namespace largest_b_value_l2955_295534

theorem largest_b_value (b : ℚ) (h : (3*b + 7) * (b - 2) = 9*b) : b ≤ 2 := by
  sorry

end largest_b_value_l2955_295534


namespace max_xy_value_l2955_295502

theorem max_xy_value (x y : ℕ+) (h : 7 * x + 4 * y = 150) : x * y ≤ 200 := by
  sorry

end max_xy_value_l2955_295502


namespace expo_min_rental_fee_l2955_295595

/-- Represents a bus type with its seat capacity and rental fee -/
structure BusType where
  seats : ℕ
  fee : ℕ

/-- Calculates the minimum rental fee for transporting people using two types of buses -/
def minRentalFee (people : ℕ) (typeA typeB : BusType) : ℕ :=
  sorry

/-- Theorem stating the minimum rental fee for the given problem -/
theorem expo_min_rental_fee :
  let typeA : BusType := ⟨40, 400⟩
  let typeB : BusType := ⟨50, 480⟩
  minRentalFee 360 typeA typeB = 3520 := by
  sorry

end expo_min_rental_fee_l2955_295595


namespace min_value_reciprocal_sum_l2955_295524

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 2) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → 2 * x' + y' = 2 → 1 / x' + 1 / y' ≥ 1 / x + 1 / y) →
  1 / x + 1 / y = 3 / 2 + Real.sqrt 2 :=
by sorry

end min_value_reciprocal_sum_l2955_295524


namespace percentage_problem_l2955_295564

theorem percentage_problem (x : ℝ) (h : 0.4 * x = 160) : 0.1 * x = 40 := by
  sorry

end percentage_problem_l2955_295564
