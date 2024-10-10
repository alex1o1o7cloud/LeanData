import Mathlib

namespace isosceles_triangle_circumradius_l2445_244550

/-- Given an isosceles triangle with base angle α, if the altitude to the base exceeds
    the radius of the inscribed circle by m, then the radius of the circumscribed circle
    is m / (4 * sin²(α/2)). -/
theorem isosceles_triangle_circumradius (α m : ℝ) (h_α : 0 < α ∧ α < π) (h_m : m > 0) :
  let altitude := inradius + m
  circumradius = m / (4 * Real.sin (α / 2) ^ 2) :=
by sorry

end isosceles_triangle_circumradius_l2445_244550


namespace parabola_minimum_distance_sum_l2445_244586

/-- The minimum distance sum from a point on the parabola y² = 4x to two fixed points -/
theorem parabola_minimum_distance_sum :
  ∀ x y : ℝ,
  y^2 = 4*x →
  (∀ x' y' : ℝ, y'^2 = 4*x' →
    Real.sqrt ((x - 2)^2 + (y - 1)^2) + Real.sqrt ((x - 1)^2 + y^2) ≤
    Real.sqrt ((x' - 2)^2 + (y' - 1)^2) + Real.sqrt ((x' - 1)^2 + y'^2)) →
  Real.sqrt ((x - 2)^2 + (y - 1)^2) + Real.sqrt ((x - 1)^2 + y^2) = 3 :=
by sorry

end parabola_minimum_distance_sum_l2445_244586


namespace circle_properties_l2445_244501

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 2*y - 4 = 0

-- Theorem statement
theorem circle_properties :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, circle_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    center_x = -2 ∧
    center_y = 1 ∧
    radius = 3 := by
  sorry

end circle_properties_l2445_244501


namespace prob_and_arrangements_correct_l2445_244593

/-- The number of class officers -/
def total_officers : Nat := 6

/-- The number of boys among the officers -/
def num_boys : Nat := 3

/-- The number of girls among the officers -/
def num_girls : Nat := 3

/-- The number of people selected for voluntary labor -/
def num_selected : Nat := 3

/-- The probability of selecting at least 2 girls out of 3 people from a group of 3 boys and 3 girls -/
def prob_at_least_two_girls : ℚ := 1/2

/-- The number of ways to arrange 6 people (3 boys and 3 girls) in a row, 
    where one boy must be at an end and two specific girls must be together -/
def num_arrangements : Nat := 96

theorem prob_and_arrangements_correct : 
  (total_officers = num_boys + num_girls) →
  (prob_at_least_two_girls = 1/2) ∧ 
  (num_arrangements = 96) := by sorry

end prob_and_arrangements_correct_l2445_244593


namespace max_divisor_of_f_l2445_244570

def f (n : ℕ) : ℕ := (2 * n + 7) * 3^n + 9

theorem max_divisor_of_f :
  ∃ (m : ℕ), (∀ (n : ℕ), m ∣ f n) ∧ 
  (∀ (k : ℕ), (∀ (n : ℕ), k ∣ f n) → k ≤ 36) ∧
  (∀ (n : ℕ), 36 ∣ f n) :=
sorry

end max_divisor_of_f_l2445_244570


namespace inequality_proof_l2445_244529

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b * c = 1) :
  (a - 1 + 1/b) * (b - 1 + 1/c) * (c - 1 + 1/a) ≤ 1 := by
sorry

end inequality_proof_l2445_244529


namespace expression_evaluation_l2445_244530

theorem expression_evaluation :
  ∃ k : ℝ, k > 0 ∧ (3^512 + 7^513)^2 - (3^512 - 7^513)^2 = k * 10^513 ∧ k = 28 * 2.1^512 := by
  sorry

end expression_evaluation_l2445_244530


namespace inequality_proof_l2445_244515

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  (a^2 + b^2 + c^2) * (a / (b + c) + b / (a + c) + c / (a + b)) ≥ 1 / 2 := by
  sorry

end inequality_proof_l2445_244515


namespace simplify_complex_expression_l2445_244567

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- Theorem stating that the simplification of 4(2-i) + 2i(3-2i) is 12 + 2i -/
theorem simplify_complex_expression : 4 * (2 - i) + 2 * i * (3 - 2 * i) = 12 + 2 * i :=
by sorry

end simplify_complex_expression_l2445_244567


namespace sum_of_fifth_powers_l2445_244596

theorem sum_of_fifth_powers (α β γ : ℂ) 
  (h1 : α + β + γ = 2)
  (h2 : α^2 + β^2 + γ^2 = 5)
  (h3 : α^3 + β^3 + γ^3 = 10) :
  α^5 + β^5 + γ^5 = 47.2 := by
  sorry

end sum_of_fifth_powers_l2445_244596


namespace parabola_equation_from_ellipse_focus_l2445_244573

/-- The standard equation of a parabola with its focus at the right focus of the ellipse x^2/3 + y^2 = 1 -/
theorem parabola_equation_from_ellipse_focus : 
  ∃ (a : ℝ), a > 0 ∧ 
  (∀ (x y : ℝ), (x^2 / 3 + y^2 = 1 ∧ x > 0) → 
    (∀ (u v : ℝ), v^2 = 4 * Real.sqrt 2 * u ↔ 
      (u - x)^2 + v^2 = (u - a)^2 + v^2)) :=
sorry

end parabola_equation_from_ellipse_focus_l2445_244573


namespace zoo_total_animals_l2445_244577

def zoo_animals (num_penguins : ℕ) (num_polar_bears : ℕ) : ℕ :=
  num_penguins + num_polar_bears

theorem zoo_total_animals :
  let num_penguins : ℕ := 21
  let num_polar_bears : ℕ := 2 * num_penguins
  zoo_animals num_penguins num_polar_bears = 63 := by
  sorry

end zoo_total_animals_l2445_244577


namespace second_number_less_than_twice_first_l2445_244592

theorem second_number_less_than_twice_first (x y : ℤ) : 
  x + y = 57 → y = 37 → 2 * x - y = 3 := by
  sorry

end second_number_less_than_twice_first_l2445_244592


namespace car_a_speed_calculation_l2445_244585

-- Define the problem parameters
def initial_distance : ℝ := 40
def overtake_distance : ℝ := 8
def car_b_speed : ℝ := 50
def overtake_time : ℝ := 6

-- Define the theorem
theorem car_a_speed_calculation :
  ∃ (speed_a : ℝ),
    speed_a * overtake_time = car_b_speed * overtake_time + initial_distance + overtake_distance ∧
    speed_a = 58 :=
by sorry

end car_a_speed_calculation_l2445_244585


namespace triangle_inequality_range_l2445_244578

theorem triangle_inequality_range (x : ℝ) : 
  (3 : ℝ) > 0 ∧ (1 + 2*x) > 0 ∧ 8 > 0 ∧
  3 + (1 + 2*x) > 8 ∧
  3 + 8 > (1 + 2*x) ∧
  (1 + 2*x) + 8 > 3 ↔
  2 < x ∧ x < 5 := by sorry

end triangle_inequality_range_l2445_244578


namespace correct_number_of_elements_l2445_244511

theorem correct_number_of_elements 
  (n : ℕ) 
  (S : ℝ) 
  (initial_average : ℝ) 
  (correct_average : ℝ) 
  (wrong_number : ℝ) 
  (correct_number : ℝ) 
  (h1 : initial_average = 15) 
  (h2 : correct_average = 16) 
  (h3 : wrong_number = 26) 
  (h4 : correct_number = 36) 
  (h5 : (S + wrong_number) / n = initial_average) 
  (h6 : (S + correct_number) / n = correct_average) : 
  n = 10 := by
sorry

end correct_number_of_elements_l2445_244511


namespace decimal_operations_l2445_244560

theorem decimal_operations (x y : ℝ) : 
  (x / 10 = 0.09 → x = 0.9) ∧ 
  (3.24 * y = 3240 → y = 1000) := by
  sorry

end decimal_operations_l2445_244560


namespace min_n_for_inequality_l2445_244557

theorem min_n_for_inequality : 
  ∃ (n : ℕ), (∀ (x y z : ℝ), x^2 + y^2 + z^2 ≤ n * (x^4 + y^4 + z^4)) ∧ 
  (∀ (m : ℕ), (∀ (x y z : ℝ), x^2 + y^2 + z^2 ≤ m * (x^4 + y^4 + z^4)) → n ≤ m) ∧
  n = 3 := by
  sorry

end min_n_for_inequality_l2445_244557


namespace xyz_sum_root_l2445_244554

theorem xyz_sum_root (x y z : ℝ) 
  (eq1 : y + z = 14)
  (eq2 : z + x = 15)
  (eq3 : x + y = 16) :
  Real.sqrt (x * y * z * (x + y + z)) = Real.sqrt 134.24375 := by
  sorry

end xyz_sum_root_l2445_244554


namespace easter_egg_distribution_l2445_244526

theorem easter_egg_distribution (red_eggs orange_eggs min_eggs : ℕ) 
  (h1 : red_eggs = 30)
  (h2 : orange_eggs = 45)
  (h3 : min_eggs = 5) :
  ∃ (eggs_per_basket : ℕ), 
    eggs_per_basket ≥ min_eggs ∧ 
    eggs_per_basket ∣ red_eggs ∧ 
    eggs_per_basket ∣ orange_eggs ∧
    ∀ (n : ℕ), n > eggs_per_basket → ¬(n ∣ red_eggs ∧ n ∣ orange_eggs) :=
by
  sorry

end easter_egg_distribution_l2445_244526


namespace every_three_connected_graph_without_K5_K33_is_planar_l2445_244512

-- Define a graph type
structure Graph (V : Type) where
  edges : V → V → Prop

-- Define 3-connectivity
def isThreeConnected (G : Graph V) : Prop := sorry

-- Define subgraph relation
def isSubgraph (H G : Graph V) : Prop := sorry

-- Define K^5 graph
def K5 (V : Type) : Graph V := sorry

-- Define K_{3,3} graph
def K33 (V : Type) : Graph V := sorry

-- Define planarity
def isPlanar (G : Graph V) : Prop := sorry

-- The main theorem
theorem every_three_connected_graph_without_K5_K33_is_planar 
  (G : Graph V) 
  (h1 : isThreeConnected G) 
  (h2 : ¬ isSubgraph (K5 V) G) 
  (h3 : ¬ isSubgraph (K33 V) G) : 
  isPlanar G := by sorry

end every_three_connected_graph_without_K5_K33_is_planar_l2445_244512


namespace eleventh_number_with_digit_sum_14_l2445_244566

/-- A function that returns the sum of digits of a positive integer -/
def sumOfDigits (n : ℕ+) : ℕ :=
  sorry

/-- A function that returns the nth positive integer whose digits sum to 14 -/
def nthNumberWithDigitSum14 (n : ℕ+) : ℕ+ :=
  sorry

/-- The theorem stating that the 11th number with digit sum 14 is 194 -/
theorem eleventh_number_with_digit_sum_14 : 
  nthNumberWithDigitSum14 11 = 194 := by sorry

end eleventh_number_with_digit_sum_14_l2445_244566


namespace divisible_by_perfect_cube_l2445_244556

theorem divisible_by_perfect_cube (a b : ℕ) (ha : a > 0) (hb : b > 0)
  (h_divides : (a^2 + 3*a*b + 3*b^2 - 1) ∣ (a + b^3)) :
  ∃ (n : ℕ), n > 1 ∧ (n^3 : ℕ) ∣ (a^2 + 3*a*b + 3*b^2 - 1) := by
  sorry

end divisible_by_perfect_cube_l2445_244556


namespace ball_distribution_ways_l2445_244531

/-- Represents the number of ways to distribute balls of a given color --/
def distribute_balls (remaining : ℕ) : ℕ := remaining + 1

/-- Represents the total number of ways to distribute balls between two boys --/
def total_distributions (white : ℕ) (black : ℕ) (red : ℕ) : ℕ :=
  distribute_balls (white - 4) * distribute_balls (red - 4)

theorem ball_distribution_ways :
  let white := 6
  let black := 4
  let red := 8
  total_distributions white black red = 15 := by
  sorry

end ball_distribution_ways_l2445_244531


namespace broken_line_rectangle_ratio_l2445_244507

/-- A rectangle with a broken line inside it -/
structure BrokenLineRectangle where
  /-- The shorter side of the rectangle -/
  short_side : ℝ
  /-- The longer side of the rectangle -/
  long_side : ℝ
  /-- The broken line consists of segments equal to the shorter side -/
  segment_length : ℝ
  /-- The short side is positive -/
  short_positive : 0 < short_side
  /-- The long side is longer than the short side -/
  long_longer : short_side < long_side
  /-- The segment length is equal to the shorter side -/
  segment_eq_short : segment_length = short_side
  /-- Adjacent segments of the broken line are perpendicular -/
  segments_perpendicular : True

/-- The ratio of the shorter side to the longer side is 1:2 -/
theorem broken_line_rectangle_ratio (r : BrokenLineRectangle) :
  r.short_side / r.long_side = 1 / 2 := by
  sorry

end broken_line_rectangle_ratio_l2445_244507


namespace second_denomination_value_l2445_244575

theorem second_denomination_value (total_amount : ℕ) (total_notes : ℕ) : 
  total_amount = 400 →
  total_notes = 75 →
  ∃ (x : ℕ), 
    x > 1 ∧ 
    x < 10 ∧
    (total_notes / 3) * (1 + x + 10) = total_amount →
    x = 5 := by
  sorry

end second_denomination_value_l2445_244575


namespace impossible_all_defective_l2445_244587

theorem impossible_all_defective (total : ℕ) (defective : ℕ) (selected : ℕ)
  (h1 : total = 25)
  (h2 : defective = 2)
  (h3 : selected = 3)
  (h4 : defective < total)
  (h5 : selected ≤ total) :
  Nat.choose defective selected / Nat.choose total selected = 0 :=
by sorry

end impossible_all_defective_l2445_244587


namespace jade_savings_l2445_244502

def monthly_income : ℝ := 1600

def living_expenses_ratio : ℝ := 0.75
def insurance_ratio : ℝ := 0.2

def savings (income : ℝ) (living_ratio : ℝ) (insurance_ratio : ℝ) : ℝ :=
  income - (income * living_ratio) - (income * insurance_ratio)

theorem jade_savings : savings monthly_income living_expenses_ratio insurance_ratio = 80 := by
  sorry

end jade_savings_l2445_244502


namespace painting_height_l2445_244546

theorem painting_height (wall_height wall_width painting_width : ℝ) 
  (wall_area painting_area : ℝ) (painting_percentage : ℝ) :
  wall_height = 5 →
  wall_width = 10 →
  painting_width = 4 →
  painting_percentage = 0.16 →
  wall_area = wall_height * wall_width →
  painting_area = painting_percentage * wall_area →
  painting_area = painting_width * 2 :=
by
  sorry

end painting_height_l2445_244546


namespace odd_increasing_function_inequality_l2445_244509

-- Define an odd function that is increasing on [0,+∞)
def OddIncreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x y, 0 ≤ x → x < y → f x < f y)

-- State the theorem
theorem odd_increasing_function_inequality (f : ℝ → ℝ) (h : OddIncreasingFunction f) :
  ∀ x : ℝ, f (Real.log x) < 0 → 0 < x ∧ x < 1 := by
  sorry

end odd_increasing_function_inequality_l2445_244509


namespace cube_volume_from_surface_area_l2445_244521

/-- Given a cube with surface area 864 square units, its volume is 1728 cubic units. -/
theorem cube_volume_from_surface_area :
  ∀ s : ℝ,
  (6 * s^2 = 864) →
  s^3 = 1728 :=
by
  sorry

end cube_volume_from_surface_area_l2445_244521


namespace absolute_value_inequality_l2445_244538

theorem absolute_value_inequality (x : ℝ) : |x - 3| ≥ |x| ↔ x ≤ 3/2 := by sorry

end absolute_value_inequality_l2445_244538


namespace max_value_reciprocal_sum_l2445_244528

theorem max_value_reciprocal_sum (x y a b : ℝ) 
  (ha : a > 1) (hb : b > 1) 
  (hax : a^x = 3) (hby : b^y = 3) 
  (hab : a + b = 2 * Real.sqrt 3) : 
  ∃ (max : ℝ), max = 1 ∧ ∀ (x' y' : ℝ), 
    (∃ (a' b' : ℝ), a' > 1 ∧ b' > 1 ∧ a'^x' = 3 ∧ b'^y' = 3 ∧ a' + b' = 2 * Real.sqrt 3) →
    1/x' + 1/y' ≤ max :=
by
  sorry

end max_value_reciprocal_sum_l2445_244528


namespace quadratic_perfect_square_l2445_244525

theorem quadratic_perfect_square (k : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, x^2 - 20*x + k = (x + b)^2) ↔ k = 100 :=
by sorry

end quadratic_perfect_square_l2445_244525


namespace diamond_equation_solution_l2445_244582

-- Define the diamond operation
def diamond (a b : ℝ) : ℝ := 3 * a * b - a + b

-- Theorem statement
theorem diamond_equation_solution :
  ∃ x : ℝ, diamond 3 x = 24 ∧ x = 2.7 := by
  sorry

end diamond_equation_solution_l2445_244582


namespace sphere_volume_sum_l2445_244589

theorem sphere_volume_sum (π : ℝ) (h : π > 0) : 
  let sphere_volume (r : ℝ) := (4/3) * π * r^3
  sphere_volume 1 + sphere_volume 4 + sphere_volume 6 + sphere_volume 3 = (1232/3) * π :=
by sorry

end sphere_volume_sum_l2445_244589


namespace circle_equation_correct_l2445_244581

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the equation of a circle
def circleEquation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

-- Theorem statement
theorem circle_equation_correct :
  let c : Circle := { center := (-1, 3), radius := 2 }
  ∀ x y : ℝ, circleEquation c x y ↔ (x + 1)^2 + (y - 3)^2 = 4 := by
  sorry

end circle_equation_correct_l2445_244581


namespace andrews_piggy_bank_donation_l2445_244574

/-- Calculates the amount Andrew donated from his piggy bank to the homeless shelter --/
theorem andrews_piggy_bank_donation
  (total_earnings : ℕ)
  (ingredient_cost : ℕ)
  (total_shelter_donation : ℕ)
  (h1 : total_earnings = 400)
  (h2 : ingredient_cost = 100)
  (h3 : total_shelter_donation = 160) :
  total_shelter_donation - ((total_earnings - ingredient_cost) / 2) = 10 := by
  sorry

#check andrews_piggy_bank_donation

end andrews_piggy_bank_donation_l2445_244574


namespace phoebe_peanut_butter_l2445_244505

/-- The number of jars of peanut butter needed for Phoebe and her dog for 30 days -/
def jars_needed (
  phoebe_servings : ℕ)  -- Phoebe's daily servings
  (dog_servings : ℕ)    -- Dog's daily servings
  (days : ℕ)            -- Number of days
  (servings_per_jar : ℕ) -- Servings per jar
  : ℕ :=
  ((phoebe_servings + dog_servings) * days + servings_per_jar - 1) / servings_per_jar

/-- Theorem stating the number of jars needed for Phoebe and her dog for 30 days -/
theorem phoebe_peanut_butter :
  jars_needed 1 1 30 15 = 4 := by
  sorry

end phoebe_peanut_butter_l2445_244505


namespace school_election_votes_l2445_244579

/-- Represents the total number of votes in a school election --/
def total_votes : ℕ := 180

/-- Represents Brenda's share of the total votes --/
def brenda_fraction : ℚ := 4 / 15

/-- Represents the number of votes Brenda received --/
def brenda_votes : ℕ := 48

/-- Represents the number of votes Colby received --/
def colby_votes : ℕ := 35

/-- Theorem stating that given the conditions, the total number of votes is 180 --/
theorem school_election_votes : 
  (brenda_fraction * total_votes = brenda_votes) ∧ 
  (colby_votes < total_votes) ∧ 
  (brenda_votes + colby_votes < total_votes) :=
sorry


end school_election_votes_l2445_244579


namespace root_equation_problem_l2445_244565

theorem root_equation_problem (a b m p r : ℝ) : 
  (a^2 - m*a + 3 = 0) →
  (b^2 - m*b + 3 = 0) →
  ((a + 1/b)^2 - p*(a + 1/b) + r = 0) →
  ((b + 1/a)^2 - p*(b + 1/a) + r = 0) →
  r = 16/3 := by
sorry

end root_equation_problem_l2445_244565


namespace sum_mod_twelve_l2445_244543

theorem sum_mod_twelve : (2150 + 2151 + 2152 + 2153 + 2154 + 2155) % 12 = 3 := by
  sorry

end sum_mod_twelve_l2445_244543


namespace tan_240_degrees_l2445_244559

theorem tan_240_degrees : Real.tan (240 * π / 180) = Real.sqrt 3 := by
  sorry

end tan_240_degrees_l2445_244559


namespace calculate_expression_l2445_244508

theorem calculate_expression : 2 * Real.cos (45 * π / 180) - (π - 2023) ^ 0 + |3 - Real.sqrt 2| = 2 := by
  sorry

end calculate_expression_l2445_244508


namespace G_is_odd_and_f_neg_b_value_l2445_244534

noncomputable def f (x : ℝ) : ℝ := 2 * Real.exp x / (Real.exp x + 1)

noncomputable def G (x : ℝ) : ℝ := f x - 1

theorem G_is_odd_and_f_neg_b_value (b : ℝ) (h : f b = 3/2) :
  (∀ x, G (-x) = -G x) ∧ f (-b) = 1/2 := by sorry

end G_is_odd_and_f_neg_b_value_l2445_244534


namespace train_bridge_crossing_time_l2445_244568

/-- Proves that a train of given length and speed takes 30 seconds to cross a bridge of given length -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (bridge_length : ℝ) 
  (h1 : train_length = 120)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 255) : 
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

end train_bridge_crossing_time_l2445_244568


namespace vehicle_value_last_year_l2445_244524

/-- If a vehicle's value this year is 16000 dollars and is 0.8 times its value last year,
    then its value last year was 20000 dollars. -/
theorem vehicle_value_last_year 
  (value_this_year : ℝ) 
  (value_ratio : ℝ) 
  (h1 : value_this_year = 16000)
  (h2 : value_ratio = 0.8)
  (h3 : value_this_year = value_ratio * value_last_year) : 
  value_last_year = 20000 := by
  sorry


end vehicle_value_last_year_l2445_244524


namespace abcd_congruence_l2445_244535

theorem abcd_congruence (a b c d : ℕ) 
  (h1 : a < 7) (h2 : b < 7) (h3 : c < 7) (h4 : d < 7)
  (c1 : (a + 2*b + 3*c + 4*d) % 7 = 1)
  (c2 : (2*a + 3*b + c + 2*d) % 7 = 5)
  (c3 : (3*a + b + 2*c + 3*d) % 7 = 3)
  (c4 : (4*a + 2*b + d + c) % 7 = 2) :
  (a * b * c * d) % 7 = 0 := by
sorry

end abcd_congruence_l2445_244535


namespace triangle_side_length_l2445_244598

/-- Triangle DEF with given properties -/
structure Triangle where
  D : ℝ  -- Angle D
  E : ℝ  -- Angle E
  F : ℝ  -- Angle F
  d : ℝ  -- Side length opposite to angle D
  e : ℝ  -- Side length opposite to angle E
  f : ℝ  -- Side length opposite to angle F

/-- The theorem stating the properties of the triangle and the value of f -/
theorem triangle_side_length 
  (t : Triangle)
  (h1 : t.d = 7)
  (h2 : t.e = 3)
  (h3 : Real.cos (t.D - t.E) = 7/8) :
  t.f = 6.5 := by
  sorry

end triangle_side_length_l2445_244598


namespace smallest_c_value_l2445_244541

theorem smallest_c_value (a b c : ℤ) 
  (h1 : a < b) (h2 : b < c)
  (h3 : b - a = c - b)  -- arithmetic progression
  (h4 : a * a = c * b)  -- geometric progression
  : c ≥ 4 ∧ ∃ (a' b' : ℤ), a' < b' ∧ b' < 4 ∧ b' - a' = 4 - b' ∧ a' * a' = 4 * b' := by
  sorry

end smallest_c_value_l2445_244541


namespace line_moved_down_three_units_l2445_244555

/-- Represents a linear function of the form y = mx + b -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- Moves a linear function vertically by a given amount -/
def moveVertically (f : LinearFunction) (amount : ℝ) : LinearFunction :=
  { slope := f.slope, intercept := f.intercept - amount }

theorem line_moved_down_three_units :
  let original := LinearFunction.mk 2 5
  let moved := moveVertically original 3
  moved = LinearFunction.mk 2 2 := by
  sorry

end line_moved_down_three_units_l2445_244555


namespace parking_lot_capacity_l2445_244542

/-- Represents a multi-story parking lot -/
structure ParkingLot where
  levels : Nat
  capacity_per_level : Nat

/-- Calculates the total capacity of a parking lot -/
def total_capacity (p : ParkingLot) : Nat :=
  p.levels * p.capacity_per_level

/-- Theorem stating the total capacity of the specific parking lot -/
theorem parking_lot_capacity :
  ∃ (p : ParkingLot), p.levels = 5 ∧ p.capacity_per_level = 23 + 62 ∧ total_capacity p = 425 :=
by
  sorry

end parking_lot_capacity_l2445_244542


namespace initial_paint_amount_l2445_244527

theorem initial_paint_amount (total_needed : ℕ) (bought : ℕ) (still_needed : ℕ) 
  (h1 : total_needed = 70)
  (h2 : bought = 23)
  (h3 : still_needed = 11) :
  total_needed - still_needed - bought = 36 :=
by sorry

end initial_paint_amount_l2445_244527


namespace range_of_set_A_l2445_244591

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def set_A : Set ℕ := {p | 15 < p ∧ p < 36 ∧ is_prime p}

theorem range_of_set_A : 
  ∃ (min max : ℕ), min ∈ set_A ∧ max ∈ set_A ∧ 
  (∀ x ∈ set_A, min ≤ x ∧ x ≤ max) ∧
  max - min = 14 :=
sorry

end range_of_set_A_l2445_244591


namespace triangle_radii_relation_l2445_244551

/-- Given a triangle with sides a, b, c, semi-perimeter p, inradius r, and excircle radii r_a, r_b, r_c,
    prove that 1/((p-a)(p-b)) + 1/((p-b)(p-c)) + 1/((p-c)(p-a)) = 1/r^2 -/
theorem triangle_radii_relation (a b c p r r_a r_b r_c : ℝ) 
  (h_p : p = (a + b + c) / 2)
  (h_r : r > 0)
  (h_ra : r_a > 0)
  (h_rb : r_b > 0)
  (h_rc : r_c > 0)
  (h_pbc : 1 / ((p - b) * (p - c)) = 1 / (r * r_a))
  (h_pca : 1 / ((p - c) * (p - a)) = 1 / (r * r_b))
  (h_pab : 1 / ((p - a) * (p - b)) = 1 / (r * r_c))
  (h_sum : 1 / r_a + 1 / r_b + 1 / r_c = 1 / r) :
  1 / ((p - a) * (p - b)) + 1 / ((p - b) * (p - c)) + 1 / ((p - c) * (p - a)) = 1 / r^2 := by
  sorry

end triangle_radii_relation_l2445_244551


namespace z_in_fourth_quadrant_l2445_244597

/-- Given a complex number z satisfying zi = 2 + i, prove that the real part of z is positive
    and the imaginary part of z is negative. -/
theorem z_in_fourth_quadrant (z : ℂ) (h : z * Complex.I = 2 + Complex.I) :
  0 < z.re ∧ z.im < 0 := by
  sorry

end z_in_fourth_quadrant_l2445_244597


namespace least_integer_greater_than_sqrt_300_l2445_244580

theorem least_integer_greater_than_sqrt_300 : ∃ n : ℕ, n > ⌊Real.sqrt 300⌋ ∧ ∀ m : ℕ, m > ⌊Real.sqrt 300⌋ → m ≥ n :=
by
  sorry

end least_integer_greater_than_sqrt_300_l2445_244580


namespace unique_solution_for_squared_geometric_sum_l2445_244563

theorem unique_solution_for_squared_geometric_sum : 
  ∃! (n m : ℕ), n > 0 ∧ m > 0 ∧ n^2 = (m^5 - 1) / (m - 1) := by
  sorry

end unique_solution_for_squared_geometric_sum_l2445_244563


namespace total_interest_calculation_total_interest_is_805_l2445_244561

/-- Calculate the total interest after 10 years given the conditions -/
theorem total_interest_calculation (P R : ℝ) 
  (h1 : P * R * 10 / 100 = 700) 
  (h2 : P * R = 700) : 
  P * R * 5 / 100 + 3 * P * R * 5 / 100 = 805 := by
  sorry

/-- Prove that the total interest is 805 -/
theorem total_interest_is_805 (P R : ℝ) 
  (h1 : P * R * 10 / 100 = 700) 
  (h2 : P * R = 700) : 
  ∃ (total_interest : ℝ), total_interest = 805 := by
  sorry

end total_interest_calculation_total_interest_is_805_l2445_244561


namespace arithmetic_sequence_max_terms_l2445_244548

theorem arithmetic_sequence_max_terms 
  (a : ℝ) (n : ℕ) 
  (h1 : a^2 + (n - 1) * (a + 2 * (n - 1)) ≤ 100) : n ≤ 8 := by
  sorry

#check arithmetic_sequence_max_terms

end arithmetic_sequence_max_terms_l2445_244548


namespace absolute_value_equals_negative_l2445_244549

theorem absolute_value_equals_negative (a : ℝ) : 
  (abs a = -a) → a ≤ 0 := by
sorry

end absolute_value_equals_negative_l2445_244549


namespace cubic_equation_game_strategy_l2445_244562

theorem cubic_equation_game_strategy (second_player_choice : ℤ) : 
  ∃ (a b c : ℤ), ∃ (x y z : ℤ),
    (x^3 + a*x^2 + b*x + c = 0) ∧
    (y^3 + a*y^2 + b*y + c = 0) ∧
    (z^3 + a*z^2 + b*z + c = 0) ∧
    (a = second_player_choice ∨ b = second_player_choice) :=
by sorry

end cubic_equation_game_strategy_l2445_244562


namespace distance_traveled_l2445_244510

/-- 
Given a speed of 20 km/hr and a time of 8 hr, prove that the distance traveled is 160 km.
-/
theorem distance_traveled (speed : ℝ) (time : ℝ) (h1 : speed = 20) (h2 : time = 8) :
  speed * time = 160 := by
  sorry

end distance_traveled_l2445_244510


namespace quadratic_rational_root_contradiction_l2445_244540

theorem quadratic_rational_root_contradiction (a b c : ℤ) (h_a_nonzero : a ≠ 0) 
  (h_rational_root : ∃ (p q : ℤ), q ≠ 0 ∧ a * (p / q)^2 + b * (p / q) + c = 0) 
  (h_all_odd : Odd a ∧ Odd b ∧ Odd c) : False :=
sorry

end quadratic_rational_root_contradiction_l2445_244540


namespace rachel_coloring_books_l2445_244564

/-- The number of pictures remaining to be colored given the number of pictures in three coloring books and the number of pictures already colored. -/
def remaining_pictures (book1 book2 book3 colored : ℕ) : ℕ :=
  book1 + book2 + book3 - colored

/-- Theorem stating that given the specific numbers from the problem, the remaining pictures to be colored is 56. -/
theorem rachel_coloring_books : remaining_pictures 23 32 45 44 = 56 := by
  sorry

end rachel_coloring_books_l2445_244564


namespace philips_farm_animals_l2445_244537

/-- The number of animals on Philip's farm --/
def total_animals (cows ducks pigs : ℕ) : ℕ := cows + ducks + pigs

/-- Theorem stating the total number of animals on Philip's farm --/
theorem philips_farm_animals :
  ∀ (cows ducks pigs : ℕ),
  cows = 20 →
  ducks = cows + cows / 2 →
  pigs = (cows + ducks) / 5 →
  total_animals cows ducks pigs = 60 := by
sorry

end philips_farm_animals_l2445_244537


namespace interval_partition_existence_l2445_244503

theorem interval_partition_existence : ∃ (x : Fin 10 → ℝ), 
  (∀ i, x i ∈ Set.Icc (0 : ℝ) 1) ∧ 
  (∀ k : Fin 9, k.val + 2 ≤ 10 → 
    ∀ i j : Fin (k.val + 2), i ≠ j → 
      ⌊(k.val + 2 : ℝ) * x i⌋ ≠ ⌊(k.val + 2 : ℝ) * x j⌋) :=
sorry

end interval_partition_existence_l2445_244503


namespace matrix_power_equals_fibonacci_l2445_244553

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℕ := !![1, 2; 1, 1]

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

-- State the theorem
theorem matrix_power_equals_fibonacci (n : ℕ) :
  A^n = !![fib (2*n + 1), fib (2*n + 2); fib (2*n), fib (2*n + 1)] := by
  sorry

end matrix_power_equals_fibonacci_l2445_244553


namespace unique_quadratic_solution_l2445_244594

theorem unique_quadratic_solution (m : ℝ) : 
  (∃! x : ℝ, m * x^2 + 2 * x - 1 = 0) → (m = 0 ∨ m = -1) :=
by sorry

end unique_quadratic_solution_l2445_244594


namespace ninth_term_of_geometric_sequence_l2445_244536

/-- A geometric sequence of positive real numbers -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem ninth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_fifth : a 5 = 32)
  (h_eleventh : a 11 = 2) :
  a 9 = 2 := by
  sorry

end ninth_term_of_geometric_sequence_l2445_244536


namespace expression_simplification_l2445_244532

theorem expression_simplification (x : ℝ) (h : x^2 - 3*x - 4 = 0) :
  (x / (x + 1) - 2 / (x - 1)) / (1 / (x^2 - 1)) = 2 := by
  sorry

end expression_simplification_l2445_244532


namespace salary_adjustment_proof_l2445_244506

def initial_salary : ℝ := 2500
def june_raise_percentage : ℝ := 0.15
def june_bonus : ℝ := 300
def july_cut_percentage : ℝ := 0.25

def final_salary : ℝ :=
  (initial_salary * (1 + june_raise_percentage) + june_bonus) * (1 - july_cut_percentage)

theorem salary_adjustment_proof :
  final_salary = 2381.25 := by sorry

end salary_adjustment_proof_l2445_244506


namespace first_number_proof_l2445_244533

theorem first_number_proof (x y : ℕ) (h1 : x + y = 20) (h2 : y = 15) : x = 5 := by
  sorry

end first_number_proof_l2445_244533


namespace perpendicular_segments_s_value_l2445_244544

/-- Given two perpendicular line segments PQ and PR, where P(4, 2), R(0, 1), and Q(2, s),
    prove that s = 10 -/
theorem perpendicular_segments_s_value (P Q R : ℝ × ℝ) (s : ℝ) : 
  P = (4, 2) →
  R = (0, 1) →
  Q = (2, s) →
  (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0 →
  s = 10 := by
  sorry

end perpendicular_segments_s_value_l2445_244544


namespace evaluate_expression_l2445_244588

theorem evaluate_expression (x y z : ℤ) (hx : x = -1) (hy : y = 4) (hz : z = 2) :
  z * (2 * y - 3 * x) = 22 := by
  sorry

end evaluate_expression_l2445_244588


namespace reflection_line_sum_l2445_244590

/-- Given a line y = mx + b, if the point (-2, 0) is reflected to (6, 4) across this line, then m + b = 4 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), x = 6 ∧ y = 4 ∧ 
    (x - (-2))^2 + (y - 0)^2 = ((x + 2)/2 - (m * ((x + (-2))/2) + b))^2 + 
    ((y + 0)/2 - ((x + (-2))/(2*m) + b))^2) → 
  m + b = 4 :=
by sorry

end reflection_line_sum_l2445_244590


namespace coconut_grove_problem_l2445_244516

theorem coconut_grove_problem (x : ℕ) : 
  (∃ (t₄₀ t₁₂₀ t₁₈₀ : ℕ),
    t₄₀ = x + 2 ∧
    t₁₂₀ = x ∧
    t₁₈₀ = x - 2 ∧
    (40 * t₄₀ + 120 * t₁₂₀ + 180 * t₁₈₀) / (t₄₀ + t₁₂₀ + t₁₈₀) = 100) →
  x = 7 := by
sorry

end coconut_grove_problem_l2445_244516


namespace f_properties_l2445_244500

open Real

/-- The function f defined by the given conditions -/
noncomputable def f (x : ℝ) : ℝ := x / (1 + 2 * x^2)

/-- The theorem stating the properties of f -/
theorem f_properties :
  ∀ α β x y : ℝ,
  (sin (2 * α + β) = 3 * sin β) →
  (tan α = x) →
  (tan β = y) →
  (y = f x) →
  (0 < α) →
  (α < π / 3) →
  (∀ z : ℝ, 0 < z → z < f x → z < sqrt 2 / 4) ∧
  (f x ≤ sqrt 2 / 4) ∧
  (∃ z : ℝ, 0 < z ∧ z < sqrt 2 / 4 ∧ z = f x) :=
by sorry

#check f_properties

end f_properties_l2445_244500


namespace heart_value_is_three_l2445_244520

/-- Represents a digit in base 9 and base 10 notation -/
def Heart : ℕ → Prop :=
  λ n => 0 ≤ n ∧ n ≤ 9

theorem heart_value_is_three :
  ∀ h : ℕ,
  Heart h →
  (h * 9 + 6 = h * 10 + 3) →
  h = 3 := by
sorry

end heart_value_is_three_l2445_244520


namespace binomial_n_minus_two_l2445_244595

theorem binomial_n_minus_two (n : ℕ+) : 
  Nat.choose n (n - 2) = n * (n - 1) / 2 := by
  sorry

end binomial_n_minus_two_l2445_244595


namespace binomial_18_10_l2445_244518

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 11440) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 32318 := by
  sorry

end binomial_18_10_l2445_244518


namespace new_sailor_weight_l2445_244513

theorem new_sailor_weight (n : ℕ) (original_weight replaced_weight : ℝ) 
  (h1 : n = 8)
  (h2 : replaced_weight = 56)
  (h3 : ∀ (total_weight new_weight : ℝ), 
    (total_weight + new_weight - replaced_weight) / n = total_weight / n + 1) :
  ∃ (new_weight : ℝ), new_weight = 64 := by
sorry

end new_sailor_weight_l2445_244513


namespace absolute_value_inequality_l2445_244522

theorem absolute_value_inequality (x : ℝ) : 
  |x - x^2 - 2| > x^2 - 3*x - 4 ↔ x > -3 := by sorry

end absolute_value_inequality_l2445_244522


namespace table_tennis_probabilities_l2445_244558

-- Define the probability of scoring on a serve
def p_score : ℝ := 0.6

-- Define events
def A_i (i : Nat) : ℝ := 
  if i = 0 then (1 - p_score)^2
  else if i = 1 then 2 * p_score * (1 - p_score)
  else p_score^2

def B_i (i : Nat) : ℝ := 
  if i = 0 then p_score^2
  else if i = 1 then 2 * (1 - p_score) * p_score
  else (1 - p_score)^2

def A : ℝ := 1 - p_score

-- Define the probabilities we want to prove
def p_B : ℝ := A_i 0 * A + A_i 1 * (1 - A)
def p_C : ℝ := A_i 1 * B_i 2 + A_i 2 * B_i 1 + A_i 2 * B_i 2

theorem table_tennis_probabilities : 
  p_B = 0.352 ∧ p_C = 0.3072 := by sorry

end table_tennis_probabilities_l2445_244558


namespace soccer_team_beverage_consumption_l2445_244519

theorem soccer_team_beverage_consumption 
  (team_size : ℕ) 
  (total_beverage : ℕ) 
  (h1 : team_size = 36) 
  (h2 : total_beverage = 252) :
  total_beverage / team_size = 7 := by
sorry

end soccer_team_beverage_consumption_l2445_244519


namespace min_pieces_to_control_l2445_244504

/-- Represents a rhombus game board -/
structure GameBoard where
  angle : ℝ
  side_divisions : ℕ

/-- Represents a piece on the game board -/
structure Piece where
  position : ℕ × ℕ

/-- Checks if a piece controls a given position -/
def controls (p : Piece) (pos : ℕ × ℕ) : Prop := sorry

/-- Checks if a set of pieces controls all positions on the board -/
def controls_all (pieces : Finset Piece) (board : GameBoard) : Prop := sorry

/-- The main theorem stating the minimum number of pieces required -/
theorem min_pieces_to_control (board : GameBoard) :
  board.angle = 60 ∧ board.side_divisions = 9 →
  ∃ (pieces : Finset Piece),
    pieces.card = 6 ∧
    controls_all pieces board ∧
    ∀ (other_pieces : Finset Piece),
      controls_all other_pieces board →
      other_pieces.card ≥ 6 :=
sorry

end min_pieces_to_control_l2445_244504


namespace intersection_probability_odd_polygon_l2445_244523

/-- The probability that two randomly chosen diagonals intersect inside a convex polygon with 2n+1 vertices -/
theorem intersection_probability_odd_polygon (n : ℕ) :
  let vertices := 2 * n + 1
  let diagonals := n * (2 * n + 1) - (2 * n + 1)
  let ways_to_choose_diagonals := (diagonals.choose 2 : ℚ)
  let ways_to_choose_vertices := ((2 * n + 1).choose 4 : ℚ)
  ways_to_choose_vertices / ways_to_choose_diagonals = n * (2 * n - 1) / (3 * (2 * n^2 - n - 2)) :=
by sorry

end intersection_probability_odd_polygon_l2445_244523


namespace theater_company_max_members_l2445_244545

/-- The number of columns in the rectangular formation -/
def n : ℕ := 15

/-- The total number of members in the theater company -/
def total_members : ℕ := n * (n + 9)

/-- Theorem stating that the maximum number of members satisfying the given conditions is 360 -/
theorem theater_company_max_members :
  (∃ k : ℕ, total_members = k^2 + 3) ∧
  (total_members = n * (n + 9)) ∧
  (∀ m > total_members, ¬(∃ j : ℕ, m = j^2 + 3) ∨ ¬(∃ p : ℕ, m = p * (p + 9))) ∧
  total_members = 360 := by
  sorry

end theater_company_max_members_l2445_244545


namespace club_members_count_l2445_244584

theorem club_members_count : ∃ (M : ℕ), 
  M > 0 ∧ 
  (2 : ℚ) / 5 * M = (M : ℚ) - (3 : ℚ) / 5 * M ∧ 
  (1 : ℚ) / 3 * ((3 : ℚ) / 5 * M) = (1 : ℚ) / 5 * M ∧ 
  (2 : ℚ) / 5 * M = 6 ∧ 
  M = 15 :=
by sorry

end club_members_count_l2445_244584


namespace fraction_calculation_l2445_244576

theorem fraction_calculation : (1/4 + 3/8 - 7/12) / (1/24) = 1 := by
  sorry

end fraction_calculation_l2445_244576


namespace smallest_n_congruence_l2445_244517

theorem smallest_n_congruence (n : ℕ) : 
  (∀ m : ℕ, m > 0 → m < n → ¬(629 * m ≡ 1181 * m [ZMOD 35])) ∧ 
  (629 * n ≡ 1181 * n [ZMOD 35]) → 
  n = 35 := by sorry

end smallest_n_congruence_l2445_244517


namespace cube_sum_inequality_l2445_244572

theorem cube_sum_inequality (x y z : Real) 
  (hx : 0 ≤ x ∧ x ≤ 1) 
  (hy : 0 ≤ y ∧ y ≤ 1) 
  (hz : 0 ≤ z ∧ z ≤ 1) : 
  2 * (x^3 + y^3 + z^3) - (x^2 * y + y^2 * z + z^2 * x) ≤ 3 := by
sorry

end cube_sum_inequality_l2445_244572


namespace initial_ratio_of_liquids_l2445_244547

/-- Given a mixture of two liquids p and q with total volume 40 liters,
    if adding 15 liters of q results in a ratio of 5:6 for p:q,
    then the initial ratio of p:q was 5:3. -/
theorem initial_ratio_of_liquids (p q : ℝ) : 
  p + q = 40 →
  p / (q + 15) = 5 / 6 →
  p / q = 5 / 3 := by
  sorry

end initial_ratio_of_liquids_l2445_244547


namespace no_valid_tournament_l2445_244552

/-- Represents a round-robin chess tournament -/
structure ChessTournament where
  num_players : ℕ
  wins : Fin num_players → ℕ
  draws : Fin num_players → ℕ
  losses : Fin num_players → ℕ

/-- Definition of a valid round-robin tournament -/
def is_valid_tournament (t : ChessTournament) : Prop :=
  t.num_players = 20 ∧
  ∀ i : Fin t.num_players, 
    t.wins i + t.draws i + t.losses i = t.num_players - 1 ∧
    t.wins i = t.draws i

/-- Theorem stating that a valid tournament as described is impossible -/
theorem no_valid_tournament : ¬∃ t : ChessTournament, is_valid_tournament t := by
  sorry


end no_valid_tournament_l2445_244552


namespace vectors_form_basis_l2445_244539

def e₁ : ℝ × ℝ := (-1, 2)
def e₂ : ℝ × ℝ := (5, 7)

theorem vectors_form_basis : 
  LinearIndependent ℝ ![e₁, e₂] ∧ Submodule.span ℝ {e₁, e₂} = ⊤ :=
by sorry

end vectors_form_basis_l2445_244539


namespace complex_number_in_first_quadrant_l2445_244583

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (2 - I) / (1 - 3*I)
  (z.re > 0) ∧ (z.im > 0) :=
by
  sorry

end complex_number_in_first_quadrant_l2445_244583


namespace retired_staff_samples_calculation_l2445_244599

/-- Calculates the number of samples for retired staff given total samples and ratio -/
def retired_staff_samples (total_samples : ℕ) (retired_ratio current_ratio student_ratio : ℕ) : ℕ :=
  let total_ratio := retired_ratio + current_ratio + student_ratio
  let unit_value := total_samples / total_ratio
  retired_ratio * unit_value

/-- Theorem stating that given 300 total samples and a ratio of 3:7:40, 
    the number of samples from retired staff is 18 -/
theorem retired_staff_samples_calculation :
  retired_staff_samples 300 3 7 40 = 18 := by
  sorry

end retired_staff_samples_calculation_l2445_244599


namespace max_planes_from_parallel_lines_max_planes_is_six_l2445_244514

/-- Given four parallel lines, the maximum number of unique planes formed by selecting two lines -/
theorem max_planes_from_parallel_lines : ℕ :=
  -- Define the number of lines
  let num_lines : ℕ := 4

  -- Define the function to calculate combinations
  let combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

  -- Calculate the number of ways to select 2 lines out of 4
  combinations num_lines 2

/-- Proof that the maximum number of planes is 6 -/
theorem max_planes_is_six : max_planes_from_parallel_lines = 6 := by
  sorry

end max_planes_from_parallel_lines_max_planes_is_six_l2445_244514


namespace cube_sum_and_reciprocal_l2445_244569

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = -7) : x^3 + 1/x^3 = -322 := by
  sorry

end cube_sum_and_reciprocal_l2445_244569


namespace last_digit_of_4139_power_467_l2445_244571

theorem last_digit_of_4139_power_467 : (4139^467) % 10 = 9 := by
  sorry

end last_digit_of_4139_power_467_l2445_244571
