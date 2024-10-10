import Mathlib

namespace equation_equality_l3276_327615

theorem equation_equality (a b : ℝ) : 1 - a^2 + 2*a*b - b^2 = 1 - (a^2 - 2*a*b + b^2) := by
  sorry

end equation_equality_l3276_327615


namespace equation_solutions_l3276_327675

/-- The set of solutions to the equation (3x+6)/(x^2+5x-14) = (3-x)/(x-2) -/
def solutions : Set ℝ := {x | x = 3 ∨ x = -5}

/-- The original equation -/
def equation (x : ℝ) : Prop :=
  x ≠ 2 ∧ x ≠ -7 ∧ (3*x + 6) / (x^2 + 5*x - 14) = (3 - x) / (x - 2)

theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ x ∈ solutions := by sorry

end equation_solutions_l3276_327675


namespace normal_distribution_probability_l3276_327613

/-- A random variable following a normal distribution -/
structure NormalRandomVariable where
  μ : ℝ
  σ : ℝ
  hσ : σ > 0

/-- The cumulative distribution function (CDF) of a normal random variable -/
noncomputable def normalCDF (X : NormalRandomVariable) (x : ℝ) : ℝ := sorry

/-- The probability that a random variable falls within an interval -/
noncomputable def probInterval (X : NormalRandomVariable) (a b : ℝ) : ℝ := 
  normalCDF X b - normalCDF X a

theorem normal_distribution_probability 
  (X : NormalRandomVariable) 
  (h1 : X.μ = 3) 
  (h2 : normalCDF X 4 = 0.84) : 
  probInterval X 2 4 = 0.68 := by sorry

end normal_distribution_probability_l3276_327613


namespace broken_line_circle_cover_l3276_327643

/-- A closed broken line on a plane -/
structure ClosedBrokenLine :=
  (points : Set (ℝ × ℝ))
  (is_closed : sorry)
  (length : ℝ)

/-- Theorem: Any closed broken line of length 1 on a plane can be covered by a circle of radius 1/4 -/
theorem broken_line_circle_cover (L : ClosedBrokenLine) (h : L.length = 1) :
  ∃ (center : ℝ × ℝ), ∀ p ∈ L.points, dist p center ≤ (1/4 : ℝ) := by
  sorry

end broken_line_circle_cover_l3276_327643


namespace min_value_of_squared_ratios_l3276_327651

theorem min_value_of_squared_ratios (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / b)^2 + (b / c)^2 + (c / a)^2 ≥ 3 ∧
  ((a / b)^2 + (b / c)^2 + (c / a)^2 = 3 ↔ a = b ∧ b = c) :=
sorry

end min_value_of_squared_ratios_l3276_327651


namespace remainder_equality_l3276_327642

theorem remainder_equality (P P' D : ℕ) (hP : P > P') : 
  let R := P % D
  let R' := P' % D
  let r := (P * P') % D
  let r' := (R * R') % D
  r = r' := by
sorry

end remainder_equality_l3276_327642


namespace school_colors_percentage_l3276_327617

theorem school_colors_percentage (N : ℝ) (h_pos : N > 0) : 
  let girls := 0.45 * N
  let boys := N - girls
  let girls_in_colors := 0.60 * girls
  let boys_in_colors := 0.80 * boys
  let total_in_colors := girls_in_colors + boys_in_colors
  (total_in_colors / N) = 0.71 := by
  sorry

end school_colors_percentage_l3276_327617


namespace daniels_purchase_worth_l3276_327649

/-- The total worth of Daniel's purchases -/
def total_worth (taxable_purchase : ℝ) (tax_free_items : ℝ) : ℝ :=
  taxable_purchase + tax_free_items

/-- The amount of sales tax paid on taxable purchases -/
def sales_tax (taxable_purchase : ℝ) (tax_rate : ℝ) : ℝ :=
  taxable_purchase * tax_rate

theorem daniels_purchase_worth :
  ∃ (taxable_purchase : ℝ),
    sales_tax taxable_purchase 0.05 = 0.30 ∧
    total_worth taxable_purchase 18.7 = 24.7 := by
  sorry

end daniels_purchase_worth_l3276_327649


namespace intersection_line_and_chord_length_l3276_327680

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y + 1 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y - 11 = 0

-- Define the line equation
def lineAB (x y : ℝ) : Prop := 3*x - 4*y + 6 = 0

-- Theorem statement
theorem intersection_line_and_chord_length :
  ∃ (A B : ℝ × ℝ),
    C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧
    C₂ A.1 A.2 ∧ C₂ B.1 B.2 ∧
    (∀ (x y : ℝ), C₁ x y ∧ C₂ x y → lineAB x y) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 24/5 :=
by sorry

end intersection_line_and_chord_length_l3276_327680


namespace origin_outside_circle_l3276_327662

theorem origin_outside_circle (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 2*y + a - 2 = 0 → (x^2 + y^2 > 0)) ↔ (2 < a ∧ a < 3) :=
by sorry

end origin_outside_circle_l3276_327662


namespace bookshelf_average_l3276_327612

theorem bookshelf_average (initial_books : ℕ) (new_books : ℕ) (shelves : ℕ) (leftover : ℕ) 
  (h1 : initial_books = 56)
  (h2 : new_books = 26)
  (h3 : shelves = 4)
  (h4 : leftover = 2) :
  (initial_books + new_books - leftover) / shelves = 20 := by
  sorry

end bookshelf_average_l3276_327612


namespace prob_not_adjacent_ten_chairs_l3276_327602

-- Define the number of chairs
def n : ℕ := 10

-- Define the probability function
def prob_not_adjacent (n : ℕ) : ℚ :=
  1 - (n - 1 : ℚ) / (n.choose 2 : ℚ)

-- Theorem statement
theorem prob_not_adjacent_ten_chairs :
  prob_not_adjacent n = 4/5 := by
  sorry

end prob_not_adjacent_ten_chairs_l3276_327602


namespace cube_minus_cylinder_volume_l3276_327646

/-- The remaining volume of a cube after removing a cylindrical section -/
theorem cube_minus_cylinder_volume (cube_side : ℝ) (cylinder_radius : ℝ) (cylinder_height : ℝ)
  (h1 : cube_side = 5)
  (h2 : cylinder_radius = 1)
  (h3 : cylinder_height = 5) :
  cube_side ^ 3 - π * cylinder_radius ^ 2 * cylinder_height = 125 - 5 * π := by
  sorry

#check cube_minus_cylinder_volume

end cube_minus_cylinder_volume_l3276_327646


namespace harry_travel_time_l3276_327610

/-- Calculates the total travel time for Harry's journey --/
def total_travel_time (initial_bus_time remaining_bus_time : ℕ) : ℕ :=
  let bus_time := initial_bus_time + remaining_bus_time
  let walk_time := bus_time / 2
  bus_time + walk_time

/-- Proves that Harry's total travel time is 60 minutes --/
theorem harry_travel_time :
  total_travel_time 15 25 = 60 := by
  sorry

end harry_travel_time_l3276_327610


namespace count_numbers_with_4_or_6_eq_1105_l3276_327628

-- Define the range of numbers we're considering
def range_end : Nat := 2401

-- Define a function to check if a number in base 8 contains 4 or 6
def contains_4_or_6 (n : Nat) : Bool :=
  sorry

-- Define the count of numbers containing 4 or 6
def count_numbers_with_4_or_6 : Nat :=
  (List.range range_end).filter contains_4_or_6 |>.length

-- Theorem to prove
theorem count_numbers_with_4_or_6_eq_1105 :
  count_numbers_with_4_or_6 = 1105 := by
  sorry

end count_numbers_with_4_or_6_eq_1105_l3276_327628


namespace total_balls_is_seven_l3276_327661

/-- The number of balls in the first box -/
def box1_balls : ℕ := 3

/-- The number of balls in the second box -/
def box2_balls : ℕ := 4

/-- The total number of balls in both boxes -/
def total_balls : ℕ := box1_balls + box2_balls

/-- Theorem stating that the total number of balls is 7 -/
theorem total_balls_is_seven : total_balls = 7 := by
  sorry

end total_balls_is_seven_l3276_327661


namespace functional_equation_solution_l3276_327677

theorem functional_equation_solution (f : ℝ → ℝ) (a : ℝ) 
  (h : ∀ x y : ℝ, f x * f y - a * f (x * y) = x + y) : 
  a = 1 ∨ a = -1 := by
sorry

end functional_equation_solution_l3276_327677


namespace ice_cream_profit_l3276_327699

/-- Proves the number of ice cream cones needed to be sold for a specific profit -/
theorem ice_cream_profit (cone_price : ℚ) (expense_ratio : ℚ) (target_profit : ℚ) :
  cone_price = 5 →
  expense_ratio = 4/5 →
  target_profit = 200 →
  (target_profit / (1 - expense_ratio)) / cone_price = 200 := by
  sorry

end ice_cream_profit_l3276_327699


namespace linear_function_point_values_l3276_327691

theorem linear_function_point_values (a m n b : ℝ) :
  (∃ (m n : ℝ), n = 2 * m + b ∧ a = 2 * (1/2) + b) →
  (∀ (m n : ℝ), n = 2 * m + b → m * n ≥ -8) →
  (∃ (m n : ℝ), n = 2 * m + b ∧ m * n = -8) →
  (a = -7 ∨ a = 9) :=
by sorry

end linear_function_point_values_l3276_327691


namespace isosceles_triangle_base_length_l3276_327694

/-- 
Given an isosceles triangle with two sides of length 15 and a perimeter of 40,
prove that the length of the third side (base) is 10.
-/
theorem isosceles_triangle_base_length 
  (a b c : ℝ) 
  (h1 : a = 15) 
  (h2 : b = 15) 
  (h3 : a + b + c = 40) : 
  c = 10 := by
  sorry

end isosceles_triangle_base_length_l3276_327694


namespace circle_radius_is_six_l3276_327654

theorem circle_radius_is_six (r : ℝ) : r > 0 → 3 * (2 * Real.pi * r) = Real.pi * r^2 → r = 6 := by
  sorry

end circle_radius_is_six_l3276_327654


namespace train_distance_problem_l3276_327687

/-- The distance between two points A and B, given train speeds and time difference --/
theorem train_distance_problem (v_ab v_ba : ℝ) (time_diff : ℝ) : 
  v_ab = 160 → v_ba = 120 → time_diff = 1 → 
  ∃ D : ℝ, D / v_ba = D / v_ab + time_diff ∧ D = 480 := by
  sorry

#check train_distance_problem

end train_distance_problem_l3276_327687


namespace rhombus_perimeter_l3276_327659

/-- Given a rhombus with diagonals of 10 inches and 24 inches, its perimeter is 52 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
  let side := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  4 * side = 52 := by
  sorry

end rhombus_perimeter_l3276_327659


namespace income_left_percentage_l3276_327692

/-- Given a man's spending habits, calculate the percentage of income left --/
theorem income_left_percentage (total_income : ℝ) (food_percent : ℝ) (education_percent : ℝ) (rent_percent : ℝ)
  (h1 : food_percent = 50)
  (h2 : education_percent = 15)
  (h3 : rent_percent = 50)
  (h4 : total_income > 0) :
  let remaining_after_food := total_income * (1 - food_percent / 100)
  let remaining_after_education := remaining_after_food - (total_income * education_percent / 100)
  let remaining_after_rent := remaining_after_education * (1 - rent_percent / 100)
  remaining_after_rent / total_income * 100 = 17.5 := by
  sorry

end income_left_percentage_l3276_327692


namespace geometric_sequence_ak_l3276_327647

/-- Given a geometric sequence {a_n} with sum S_n = k * 2^n - 3, prove a_k = 12 -/
theorem geometric_sequence_ak (k : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = k * 2^n - 3) →
  (∀ n, S (n + 1) - S n = a (n + 1)) →
  (∀ n, a (n + 1) = 2 * a n) →
  a k = 12 := by
  sorry


end geometric_sequence_ak_l3276_327647


namespace grape_juice_solution_l3276_327644

/-- Represents the problem of adding grape juice to a mixture --/
def GrapeJuiceProblem (initial_volume : ℝ) (initial_concentration : ℝ) (final_concentration : ℝ) (added_juice : ℝ) : Prop :=
  let final_volume := initial_volume + added_juice
  let initial_juice := initial_volume * initial_concentration
  let final_juice := final_volume * final_concentration
  final_juice = initial_juice + added_juice

/-- Theorem stating the solution to the grape juice problem --/
theorem grape_juice_solution :
  GrapeJuiceProblem 30 0.1 0.325 10 := by
  sorry

end grape_juice_solution_l3276_327644


namespace max_consecutive_new_numbers_l3276_327671

def is_new (n : Nat) : Prop :=
  n > 5 ∧ ∃ m : Nat, (∀ k < n, m % k = 0) ∧ m % n ≠ 0

theorem max_consecutive_new_numbers :
  ∃ a : Nat, a > 5 ∧
    is_new a ∧ is_new (a + 1) ∧ is_new (a + 2) ∧
    ¬(is_new (a - 1) ∧ is_new a ∧ is_new (a + 1) ∧ is_new (a + 2)) ∧
    ¬(is_new a ∧ is_new (a + 1) ∧ is_new (a + 2) ∧ is_new (a + 3)) :=
  sorry

end max_consecutive_new_numbers_l3276_327671


namespace janice_stairs_walked_l3276_327609

/-- The number of flights of stairs to Janice's office -/
def flights_to_office : ℕ := 3

/-- The number of times Janice goes up the stairs in a day -/
def times_up : ℕ := 5

/-- The number of times Janice goes down the stairs in a day -/
def times_down : ℕ := 3

/-- The total number of flights Janice walks in a day -/
def total_flights : ℕ := flights_to_office * times_up + flights_to_office * times_down

theorem janice_stairs_walked : total_flights = 24 := by
  sorry

end janice_stairs_walked_l3276_327609


namespace tom_flashlight_batteries_l3276_327676

/-- The number of batteries Tom used on his flashlights -/
def batteries_on_flashlights : ℕ := 28

/-- The number of batteries Tom used in his toys -/
def batteries_in_toys : ℕ := 15

/-- The number of batteries Tom used in his controllers -/
def batteries_in_controllers : ℕ := 2

/-- The difference between the number of batteries on flashlights and in toys -/
def battery_difference : ℕ := 13

theorem tom_flashlight_batteries :
  batteries_on_flashlights = batteries_in_toys + battery_difference := by
  sorry

end tom_flashlight_batteries_l3276_327676


namespace initial_fraction_is_half_l3276_327689

/-- Represents a journey with two parts at different speeds -/
structure Journey where
  totalDistance : ℝ
  totalTime : ℝ
  initialSpeed : ℝ
  remainingSpeed : ℝ
  initialFraction : ℝ

/-- The conditions of the journey -/
def journeyConditions (j : Journey) : Prop :=
  j.initialSpeed = 40 ∧
  j.remainingSpeed = 20 ∧
  j.initialFraction * j.totalDistance = j.initialSpeed * (j.totalTime / 3) ∧
  (1 - j.initialFraction) * j.totalDistance = j.remainingSpeed * (2 * j.totalTime / 3) ∧
  j.totalDistance > 0 ∧
  j.totalTime > 0

/-- The theorem stating that under the given conditions, the initial fraction is 1/2 -/
theorem initial_fraction_is_half (j : Journey) :
  journeyConditions j → j.initialFraction = 1/2 := by
  sorry

end initial_fraction_is_half_l3276_327689


namespace vector_calculation_l3276_327619

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_calculation (a b : V) :
  (1 / 3 : ℝ) • (a - 2 • b) + b = (1 / 3 : ℝ) • a + (1 / 3 : ℝ) • b :=
by sorry

end vector_calculation_l3276_327619


namespace jump_rope_median_and_mode_l3276_327606

def jump_rope_scores : List ℕ := [129, 130, 130, 130, 132, 132, 135, 135, 137, 137]

def median (scores : List ℕ) : ℚ := sorry

def mode (scores : List ℕ) : ℕ := sorry

theorem jump_rope_median_and_mode :
  median jump_rope_scores = 132 ∧ mode jump_rope_scores = 130 := by sorry

end jump_rope_median_and_mode_l3276_327606


namespace systematic_sampling_60_5_l3276_327673

/-- Systematic sampling function that returns a list of sample numbers -/
def systematicSample (totalPopulation : ℕ) (sampleSize : ℕ) : List ℕ :=
  let interval := totalPopulation / sampleSize
  List.range sampleSize |>.map (fun i => i * interval + interval)

/-- Theorem: The systematic sampling of 5 students from a class of 60 yields [6, 18, 30, 42, 54] -/
theorem systematic_sampling_60_5 :
  systematicSample 60 5 = [6, 18, 30, 42, 54] := by
  sorry

#eval systematicSample 60 5

end systematic_sampling_60_5_l3276_327673


namespace cost_780_candies_l3276_327625

/-- The cost of buying a given number of chocolate candies -/
def chocolateCost (candies : ℕ) : ℚ :=
  let boxSize := 30
  let boxCost := 8
  let discountThreshold := 500
  let discountRate := 0.1
  let boxes := (candies + boxSize - 1) / boxSize  -- Ceiling division
  let totalCost := boxes * boxCost
  if candies > discountThreshold then
    totalCost * (1 - discountRate)
  else
    totalCost

/-- Theorem: The cost of buying 780 chocolate candies is $187.2 -/
theorem cost_780_candies :
  chocolateCost 780 = 187.2 := by
  sorry

end cost_780_candies_l3276_327625


namespace power_function_property_l3276_327633

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ a

-- State the theorem
theorem power_function_property (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 2 = Real.sqrt 2) : 
  f 8 = 2 * Real.sqrt 2 := by
sorry

end power_function_property_l3276_327633


namespace most_tickets_have_four_hits_l3276_327672

/-- Number of matches in a lottery ticket -/
def num_matches : ℕ := 13

/-- Number of possible outcomes for each match -/
def outcomes_per_match : ℕ := 3

/-- Number of tickets with k correct predictions -/
def tickets_with_k_hits (k : ℕ) : ℕ :=
  (num_matches.choose k) * (outcomes_per_match - 1)^(num_matches - k)

/-- The number of correct predictions that maximizes the number of tickets -/
def max_hits : ℕ := 4

theorem most_tickets_have_four_hits :
  ∀ k : ℕ, k ≤ num_matches → k ≠ max_hits →
    tickets_with_k_hits k ≤ tickets_with_k_hits max_hits :=
by sorry

end most_tickets_have_four_hits_l3276_327672


namespace smallest_sum_is_84_l3276_327626

/-- Represents a rectangular prism made of dice -/
structure DicePrism where
  length : Nat
  width : Nat
  height : Nat
  total_dice : Nat
  dice_opposite_sum : Nat

/-- Calculates the smallest possible sum of visible values on the prism faces -/
def smallest_visible_sum (prism : DicePrism) : Nat :=
  sorry

/-- Theorem stating the smallest possible sum for the given prism configuration -/
theorem smallest_sum_is_84 (prism : DicePrism) 
  (h1 : prism.length = 4)
  (h2 : prism.width = 3)
  (h3 : prism.height = 2)
  (h4 : prism.total_dice = 24)
  (h5 : prism.dice_opposite_sum = 7) :
  smallest_visible_sum prism = 84 := by
  sorry

end smallest_sum_is_84_l3276_327626


namespace jellybean_problem_l3276_327623

/-- The number of jellybeans initially in the jar -/
def initial_jellybeans : ℕ := 90

/-- The number of jellybeans Samantha took -/
def samantha_took : ℕ := 24

/-- The number of jellybeans Shelby ate -/
def shelby_ate : ℕ := 12

/-- The final number of jellybeans in the jar -/
def final_jellybeans : ℕ := 72

theorem jellybean_problem :
  initial_jellybeans - samantha_took - shelby_ate +
  ((samantha_took + shelby_ate) / 2) = final_jellybeans :=
by sorry

end jellybean_problem_l3276_327623


namespace polygon_sides_l3276_327650

theorem polygon_sides (d : ℕ) (v : ℕ) : d = 77 ∧ v = 1 → ∃ n : ℕ, n * (n - 3) / 2 = d ∧ n + v = 15 := by
  sorry

end polygon_sides_l3276_327650


namespace negative_sixty_four_to_four_thirds_l3276_327683

theorem negative_sixty_four_to_four_thirds : (-64 : ℝ) ^ (4/3) = 256 := by
  sorry

end negative_sixty_four_to_four_thirds_l3276_327683


namespace and_or_relationship_l3276_327657

theorem and_or_relationship (p q : Prop) :
  (∀ (p q : Prop), p ∧ q → p ∨ q) ∧
  (∃ (p q : Prop), p ∨ q ∧ ¬(p ∧ q)) :=
by sorry

end and_or_relationship_l3276_327657


namespace smallest_natural_solution_l3276_327630

theorem smallest_natural_solution (n : ℕ) : 
  (2023 / 2022 : ℝ) ^ (36 * (1 - (2/3 : ℝ)^(n+1)) / (1 - 2/3)) > (2023 / 2022 : ℝ) ^ 96 ↔ n ≥ 5 :=
by sorry

end smallest_natural_solution_l3276_327630


namespace favorite_fruit_strawberries_l3276_327684

theorem favorite_fruit_strawberries (total : ℕ) (oranges pears apples bananas grapes : ℕ)
  (h_total : total = 900)
  (h_oranges : oranges = 130)
  (h_pears : pears = 210)
  (h_apples : apples = 275)
  (h_bananas : bananas = 93)
  (h_grapes : grapes = 119) :
  total - (oranges + pears + apples + bananas + grapes) = 73 := by
  sorry

end favorite_fruit_strawberries_l3276_327684


namespace stock_exchange_problem_l3276_327608

theorem stock_exchange_problem (h l : ℕ) : 
  h = l + l / 5 →  -- 20% more stocks closed higher
  h = 1080 →      -- 1080 stocks closed higher
  h + l = 1980    -- Total number of stocks
  := by sorry

end stock_exchange_problem_l3276_327608


namespace swimming_difference_l3276_327655

theorem swimming_difference (camden_total : ℕ) (susannah_total : ℕ) (weeks : ℕ) : 
  camden_total = 16 → susannah_total = 24 → weeks = 4 →
  (susannah_total / weeks) - (camden_total / weeks) = 2 := by
  sorry

end swimming_difference_l3276_327655


namespace sauce_per_burger_is_quarter_cup_l3276_327679

/-- The amount of barbecue sauce per burger -/
def sauce_per_burger (total_sauce : ℚ) (sauce_per_sandwich : ℚ) (num_sandwiches : ℕ) (num_burgers : ℕ) : ℚ :=
  (total_sauce - sauce_per_sandwich * num_sandwiches) / num_burgers

/-- Theorem stating that the amount of sauce per burger is 1/4 cup -/
theorem sauce_per_burger_is_quarter_cup :
  sauce_per_burger 5 (1/6) 18 8 = 1/4 := by sorry

end sauce_per_burger_is_quarter_cup_l3276_327679


namespace survey_total_students_l3276_327635

theorem survey_total_students : 
  let mac_preference : ℕ := 60
  let both_preference : ℕ := mac_preference / 3
  let no_preference : ℕ := 90
  let windows_preference : ℕ := 40
  mac_preference + both_preference + no_preference + windows_preference = 210 := by
sorry

end survey_total_students_l3276_327635


namespace diamond_expression_result_l3276_327697

/-- The diamond operation defined as a ⋄ b = a - 1/b -/
def diamond (a b : ℚ) : ℚ := a - 1 / b

/-- Theorem stating the result of the given expression -/
theorem diamond_expression_result :
  (diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4)) = -29/132 := by
  sorry

end diamond_expression_result_l3276_327697


namespace cylinder_height_l3276_327621

/-- The height of a cylindrical tin given its diameter and volume -/
theorem cylinder_height (diameter : ℝ) (volume : ℝ) (h_diameter : diameter = 14) (h_volume : volume = 245) :
  (volume / (π * (diameter / 2)^2)) = 245 / (49 * π) := by
  sorry

end cylinder_height_l3276_327621


namespace m_minus_n_values_l3276_327678

theorem m_minus_n_values (m n : ℤ) 
  (hm : |m| = 14)
  (hn : |n| = 23)
  (hmn_pos : m + n > 0) :
  m - n = -9 ∨ m - n = -37 := by
sorry

end m_minus_n_values_l3276_327678


namespace solve_bowtie_equation_l3276_327670

-- Define the operation ⊗
noncomputable def bowtie (a b : ℝ) : ℝ := a + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- Theorem statement
theorem solve_bowtie_equation (y : ℝ) : bowtie 5 y = 15 → y = 90 := by
  sorry

end solve_bowtie_equation_l3276_327670


namespace coffee_stock_problem_l3276_327667

/-- Represents the coffee stock problem --/
theorem coffee_stock_problem 
  (initial_stock : ℝ) 
  (initial_decaf_percent : ℝ) 
  (second_batch_decaf_percent : ℝ) 
  (final_decaf_percent : ℝ) 
  (second_batch : ℝ)
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 0.20)
  (h3 : second_batch_decaf_percent = 0.60)
  (h4 : final_decaf_percent = 0.28000000000000004)
  (h5 : (initial_stock * initial_decaf_percent + second_batch * second_batch_decaf_percent) / 
        (initial_stock + second_batch) = final_decaf_percent) : 
  second_batch = 100 := by
  sorry

end coffee_stock_problem_l3276_327667


namespace inequality_proof_l3276_327622

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) : 
  (a * b + b * c + a * c ≤ 1/3) ∧ 
  (a^2 / b + b^2 / c + c^2 / a ≥ 1) := by
  sorry

end inequality_proof_l3276_327622


namespace scientific_notation_of_nanometer_l3276_327636

def nanometer : ℝ := 0.000000001

theorem scientific_notation_of_nanometer :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ nanometer = a * (10 : ℝ) ^ n ∧ a = 1 ∧ n = -9 :=
by sorry

end scientific_notation_of_nanometer_l3276_327636


namespace square_difference_l3276_327604

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 12) : 
  (x - y)^2 = 16 := by
  sorry

end square_difference_l3276_327604


namespace addition_problem_solution_l3276_327669

/-- Represents a digit in the addition problem -/
structure Digit :=
  (value : Nat)
  (property : value < 10)

/-- Represents the addition problem -/
structure AdditionProblem :=
  (E : Digit)
  (I : Digit)
  (G : Digit)
  (H : Digit)
  (T : Digit)
  (F : Digit)
  (V : Digit)
  (R : Digit)
  (N : Digit)
  (all_different : ∀ d1 d2 : Digit, d1.value = d2.value → d1 = d2)
  (E_is_nine : E.value = 9)
  (G_is_odd : G.value % 2 = 1)
  (equation_holds : 
    10000 * E.value + 1000 * I.value + 100 * G.value + 10 * H.value + T.value +
    10000 * F.value + 1000 * I.value + 100 * V.value + 10 * E.value =
    10000000 * T.value + 1000000 * H.value + 100000 * I.value + 10000 * R.value +
    1000 * T.value + 100 * E.value + 10 * E.value + N.value)

theorem addition_problem_solution (problem : AdditionProblem) : problem.I.value = 4 := by
  sorry

end addition_problem_solution_l3276_327669


namespace sum_of_cubes_l3276_327620

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 3) (h2 : x * y = 2) : x^3 + y^3 = 9 := by
  sorry

end sum_of_cubes_l3276_327620


namespace arithmetic_progression_polynomial_j_eq_neg_40_l3276_327686

/-- A polynomial of degree 4 with four distinct real zeros in arithmetic progression -/
structure ArithmeticProgressionPolynomial where
  j : ℝ
  k : ℝ
  roots : Fin 4 → ℝ
  distinct : ∀ i j, i ≠ j → roots i ≠ roots j
  arithmetic_progression : ∃ (b d : ℝ), ∀ i, roots i = b + i * d

/-- The coefficient j of an ArithmeticProgressionPolynomial is -40 -/
theorem arithmetic_progression_polynomial_j_eq_neg_40 (p : ArithmeticProgressionPolynomial) :
  p.j = -40 := by sorry

end arithmetic_progression_polynomial_j_eq_neg_40_l3276_327686


namespace diary_ratio_proof_l3276_327664

theorem diary_ratio_proof (initial_diaries : ℕ) (final_diaries : ℕ) (bought_diaries : ℕ) :
  initial_diaries = 8 →
  final_diaries = 18 →
  final_diaries = (initial_diaries + bought_diaries) * 3 / 4 →
  bought_diaries / initial_diaries = 2 := by
  sorry

#check diary_ratio_proof

end diary_ratio_proof_l3276_327664


namespace probability_same_color_is_19_39_l3276_327637

def num_green_balls : ℕ := 5
def num_white_balls : ℕ := 8

def total_balls : ℕ := num_green_balls + num_white_balls

def probability_same_color : ℚ :=
  (Nat.choose num_green_balls 2 + Nat.choose num_white_balls 2) / Nat.choose total_balls 2

theorem probability_same_color_is_19_39 :
  probability_same_color = 19 / 39 := by
  sorry

end probability_same_color_is_19_39_l3276_327637


namespace davids_math_marks_l3276_327632

theorem davids_math_marks (english : ℕ) (physics : ℕ) (chemistry : ℕ) (biology : ℕ) (average : ℕ) (total_subjects : ℕ) :
  english = 96 →
  physics = 82 →
  chemistry = 97 →
  biology = 95 →
  average = 93 →
  total_subjects = 5 →
  (english + physics + chemistry + biology + (average * total_subjects - (english + physics + chemistry + biology))) / total_subjects = average :=
by sorry

end davids_math_marks_l3276_327632


namespace sum_reciprocals_and_powers_l3276_327645

theorem sum_reciprocals_and_powers (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 1 / b ≥ 4) ∧ (1 / a^2016 + 1 / b^2016 ≥ 2^2017) := by
  sorry

end sum_reciprocals_and_powers_l3276_327645


namespace division_simplification_l3276_327665

theorem division_simplification (x y : ℝ) : 6 * x^3 * y^2 / (3 * x) = 2 * x^2 * y^2 := by
  sorry

end division_simplification_l3276_327665


namespace perpendicular_line_through_point_l3276_327607

/-- Given a point M and two lines, this theorem proves that the second line passes through M and is perpendicular to the first line. -/
theorem perpendicular_line_through_point (x₀ y₀ a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) :
  (a₁ * x₀ + b₁ * y₀ + c₁ ≠ 0) →  -- M is not on the first line
  (a₁ * b₂ = -a₂ * b₁) →          -- Lines are perpendicular
  (a₂ * x₀ + b₂ * y₀ + c₂ = 0) →  -- Second line passes through M
  ∃ (k : ℝ), k ≠ 0 ∧ a₂ = k * 4 ∧ b₂ = k * 3 ∧ c₂ = k * (-13) ∧
             a₁ = k * 3 ∧ b₁ = k * (-4) ∧ c₁ = k * 6 ∧
             x₀ = 4 ∧ y₀ = -1 :=
by sorry

end perpendicular_line_through_point_l3276_327607


namespace john_total_paint_l3276_327681

/-- The number of primary colors John has -/
def num_colors : ℕ := 3

/-- The amount of paint John has for each color (in liters) -/
def paint_per_color : ℕ := 5

/-- The total amount of paint John has (in liters) -/
def total_paint : ℕ := num_colors * paint_per_color

theorem john_total_paint : total_paint = 15 := by
  sorry

end john_total_paint_l3276_327681


namespace order_of_fractions_l3276_327629

theorem order_of_fractions (x y : ℝ) (hx : x < 0) (hy : 0 < y ∧ y < 1) :
  (1 / x) < (y / x) ∧ (y / x) < (y^2 / x) := by
  sorry

end order_of_fractions_l3276_327629


namespace x_value_l3276_327653

theorem x_value (x y : ℚ) (h1 : x / y = 12 / 3) (h2 : y = 27) : x = 108 := by
  sorry

end x_value_l3276_327653


namespace meaningful_expression_range_l3276_327616

-- Define the set of real numbers where the expression is meaningful
def meaningfulSet : Set ℝ :=
  {x : ℝ | 3 - x ≥ 0 ∧ x + 1 ≠ 0}

-- Theorem statement
theorem meaningful_expression_range :
  meaningfulSet = {x : ℝ | x ≤ 3 ∧ x ≠ -1} := by
  sorry

end meaningful_expression_range_l3276_327616


namespace gcf_of_lcms_l3276_327638

theorem gcf_of_lcms : Nat.gcd (Nat.lcm 9 21) (Nat.lcm 10 22) = 1 := by sorry

end gcf_of_lcms_l3276_327638


namespace exam_problem_solution_l3276_327688

theorem exam_problem_solution (pA pB pC : ℝ) 
  (hA : pA = 1/3) 
  (hB : pB = 1/4) 
  (hC : pC = 1/5) 
  (h_independent : True) -- Assumption of independence
  : 1 - (1 - pA) * (1 - pB) * (1 - pC) = 3/5 := by
  sorry

end exam_problem_solution_l3276_327688


namespace unique_pairs_theorem_l3276_327627

theorem unique_pairs_theorem (x y : ℕ) : 
  x ≥ 2 → y ≥ 2 → 
  (3 * x) % y = 1 → 
  (3 * y) % x = 1 → 
  (x * y) % 3 = 1 → 
  ((x = 2 ∧ y = 5) ∨ (x = 5 ∧ y = 2)) := by
  sorry

#check unique_pairs_theorem

end unique_pairs_theorem_l3276_327627


namespace reading_assignment_valid_l3276_327698

/-- Represents the reading assignment for Alice, Bob, and Chandra -/
structure ReadingAssignment where
  alice_pages : ℕ
  bob_pages : ℕ
  chandra_pages : ℕ
  alice_speed : ℕ
  bob_speed : ℕ
  chandra_speed : ℕ
  total_pages : ℕ

/-- Calculates the time spent reading for a given number of pages and reading speed -/
def reading_time (pages : ℕ) (speed : ℕ) : ℕ := pages * speed

/-- Proves that the given reading assignment satisfies the conditions -/
theorem reading_assignment_valid (ra : ReadingAssignment) 
  (h_alice : ra.alice_pages = 416)
  (h_bob : ra.bob_pages = 208)
  (h_chandra : ra.chandra_pages = 276)
  (h_alice_speed : ra.alice_speed = 18)
  (h_bob_speed : ra.bob_speed = 36)
  (h_chandra_speed : ra.chandra_speed = 27)
  (h_total : ra.total_pages = 900) : 
  ra.alice_pages + ra.bob_pages + ra.chandra_pages = ra.total_pages ∧
  reading_time ra.alice_pages ra.alice_speed = reading_time ra.bob_pages ra.bob_speed ∧
  reading_time ra.bob_pages ra.bob_speed = reading_time ra.chandra_pages ra.chandra_speed :=
by sorry


end reading_assignment_valid_l3276_327698


namespace angle_value_l3276_327660

theorem angle_value (PQR PQS QRS : ℝ) (x : ℝ) : 
  PQR = 120 → PQS = 2*x → QRS = x → PQR = PQS + QRS → x = 40 := by
  sorry

end angle_value_l3276_327660


namespace max_six_yuan_items_proof_l3276_327603

/-- The maximum number of 6-yuan items that can be bought given the conditions -/
def max_six_yuan_items : ℕ := 7

theorem max_six_yuan_items_proof :
  ∀ (x y z : ℕ),
    6 * x + 4 * y + 2 * z = 60 →
    x + y + z = 16 →
    x ≤ max_six_yuan_items :=
by
  sorry

#check max_six_yuan_items_proof

end max_six_yuan_items_proof_l3276_327603


namespace ladder_problem_l3276_327648

theorem ladder_problem (ladder_length height : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ (base : ℝ), base^2 + height^2 = ladder_length^2 ∧ base = 5 := by
  sorry

end ladder_problem_l3276_327648


namespace shaded_portion_is_four_ninths_l3276_327618

-- Define the square ABCD
def square_side : ℝ := 6

-- Define the shaded areas
def shaded_area_1 : ℝ := 2 * 2
def shaded_area_2 : ℝ := 4 * 4 - 2 * 2
def shaded_area_3 : ℝ := 6 * 6

-- Total square area
def total_area : ℝ := square_side * square_side

-- Total shaded area
def total_shaded_area : ℝ := shaded_area_1 + shaded_area_2

-- Theorem to prove
theorem shaded_portion_is_four_ninths :
  total_shaded_area / total_area = 4 / 9 := by sorry

end shaded_portion_is_four_ninths_l3276_327618


namespace equation_solution_l3276_327666

theorem equation_solution : 
  ∃! x : ℚ, (x - 20) / 3 = (4 - 3 * x) / 4 ∧ x = 92 / 13 := by sorry

end equation_solution_l3276_327666


namespace kitchen_planks_l3276_327668

/-- Represents the number of wooden planks used in Andrew's house flooring project. -/
structure FlooringProject where
  bedroom : ℕ
  livingRoom : ℕ
  guestBedroom : ℕ
  hallway : ℕ
  kitchen : ℕ
  leftover : ℕ
  replacedBedroom : ℕ
  replacedGuestBedroom : ℕ

/-- Theorem stating the number of planks used for the kitchen in Andrew's flooring project. -/
theorem kitchen_planks (project : FlooringProject) 
    (h1 : project.bedroom = 8)
    (h2 : project.livingRoom = 20)
    (h3 : project.guestBedroom = project.bedroom - 2)
    (h4 : project.hallway = 4 * 2)
    (h5 : project.leftover = 6)
    (h6 : project.replacedBedroom = 3)
    (h7 : project.replacedGuestBedroom = 3)
    : project.kitchen = 6 := by
  sorry


end kitchen_planks_l3276_327668


namespace tokyo_tech_1956_entrance_exam_l3276_327685

theorem tokyo_tech_1956_entrance_exam
  (a b c : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (ha1 : a < 1) (hb1 : b < 1) (hc1 : c < 1) :
  a + b + c - a * b * c < 2 :=
sorry

end tokyo_tech_1956_entrance_exam_l3276_327685


namespace car_distance_theorem_l3276_327631

/-- The distance covered by a car given initial time and adjusted speed -/
theorem car_distance_theorem (initial_time : ℝ) (adjusted_speed : ℝ) : 
  initial_time = 6 →
  adjusted_speed = 60 →
  (initial_time * 3 / 2) * adjusted_speed = 540 := by
sorry

end car_distance_theorem_l3276_327631


namespace exists_equidistant_point_l3276_327658

/-- A line in a plane --/
structure Line where
  -- Add necessary fields for a line

/-- A point in a plane --/
structure Point where
  -- Add necessary fields for a point

/-- Three lines in a plane --/
def three_lines : Fin 3 → Line := sorry

/-- Condition that at most two lines are parallel --/
def at_most_two_parallel (lines : Fin 3 → Line) : Prop := sorry

/-- A point is equidistant from three lines --/
def equidistant_from_lines (p : Point) (lines : Fin 3 → Line) : Prop := sorry

/-- Main theorem: There always exists a point equidistant from three lines
    given that at most two of them are parallel --/
theorem exists_equidistant_point (lines : Fin 3 → Line) 
  (h : at_most_two_parallel lines) : 
  ∃ (p : Point), equidistant_from_lines p lines := by
  sorry

end exists_equidistant_point_l3276_327658


namespace tangent_line_to_circle_l3276_327614

theorem tangent_line_to_circle (m : ℝ) : 
  (∃ x y : ℝ, y = m * x ∧ x^2 + y^2 - 4*x + 2 = 0 ∧
   ∀ x' y' : ℝ, y' = m * x' → x'^2 + y'^2 - 4*x' + 2 ≥ 0) →
  m = 1 ∨ m = -1 :=
by sorry

end tangent_line_to_circle_l3276_327614


namespace work_done_by_force_l3276_327639

theorem work_done_by_force (F : ℝ → ℝ) (x₁ x₂ : ℝ) :
  (∀ x, F x = 1 + Real.exp x) →
  x₁ = 0 →
  x₂ = 1 →
  ∫ x in x₁..x₂, F x = Real.exp 1 := by
  sorry

end work_done_by_force_l3276_327639


namespace regular_hexagon_dimensions_l3276_327634

/-- Regular hexagon with given area and side lengths -/
structure RegularHexagon where
  area : ℝ
  x : ℝ
  y : ℝ
  area_eq : area = 54 * Real.sqrt 3
  side_length : x > 0
  diagonal_length : y > 0

/-- Theorem: For a regular hexagon with area 54√3 cm², if AB = x cm and AC = y√3 cm, then x = 6 and y = 6 -/
theorem regular_hexagon_dimensions (h : RegularHexagon) : h.x = 6 ∧ h.y = 6 := by
  sorry

end regular_hexagon_dimensions_l3276_327634


namespace simplify_expressions_l3276_327641

variable (x y : ℝ)

theorem simplify_expressions :
  (3 * x^2 - 2*x*y + y^2 - 3*x^2 + 3*x*y = x*y + y^2) ∧
  ((7*x^2 - 3*x*y) - 6*(x^2 - 1/3*x*y) = x^2 - x*y) := by sorry

end simplify_expressions_l3276_327641


namespace not_prime_257_pow_1092_plus_1092_l3276_327663

theorem not_prime_257_pow_1092_plus_1092 : ¬ Nat.Prime (257^1092 + 1092) := by
  sorry

end not_prime_257_pow_1092_plus_1092_l3276_327663


namespace subset_union_equality_l3276_327600

theorem subset_union_equality (n : ℕ+) (A : Fin (n + 1) → Set (Fin n)) 
  (h : ∀ i, (A i).Nonempty) :
  ∃ (I J : Set (Fin (n + 1))), I.Nonempty ∧ J.Nonempty ∧ I ∩ J = ∅ ∧
    (⋃ i ∈ I, A i) = (⋃ j ∈ J, A j) := by
  sorry

end subset_union_equality_l3276_327600


namespace point_not_in_second_quadrant_l3276_327693

theorem point_not_in_second_quadrant : ¬ ((-Real.sqrt 2 < 0) ∧ (-Real.sqrt 3 > 0)) := by
  sorry

end point_not_in_second_quadrant_l3276_327693


namespace systematic_sampling_interval_l3276_327601

theorem systematic_sampling_interval 
  (population : ℕ) 
  (sample_size : ℕ) 
  (h1 : population = 800) 
  (h2 : sample_size = 40) 
  (h3 : population > 0) 
  (h4 : sample_size > 0) :
  population / sample_size = 20 := by
  sorry

end systematic_sampling_interval_l3276_327601


namespace intercepted_arc_is_60_degrees_l3276_327611

/-- An equilateral triangle with a circle rolling along its side -/
structure RollingCircleTriangle where
  -- The side length of the equilateral triangle
  side : ℝ
  -- The radius of the rolling circle
  radius : ℝ
  -- The radius equals the height of the triangle
  height_eq_radius : radius = (side * Real.sqrt 3) / 2

/-- The angular measure of the arc intercepted on the circle by the sides of the triangle -/
def intercepted_arc_measure (t : RollingCircleTriangle) : ℝ := 
  -- Definition to be proved
  60

/-- Theorem: The angular measure of the arc intercepted on the circle 
    by the sides of the triangle is always 60° -/
theorem intercepted_arc_is_60_degrees (t : RollingCircleTriangle) : 
  intercepted_arc_measure t = 60 := by
  sorry

end intercepted_arc_is_60_degrees_l3276_327611


namespace gcd_54000_36000_l3276_327624

theorem gcd_54000_36000 : Nat.gcd 54000 36000 = 18000 := by
  sorry

end gcd_54000_36000_l3276_327624


namespace stratified_sample_elderly_count_l3276_327696

/-- Represents the number of teachers in a sample -/
structure TeacherSample where
  young : ℕ
  elderly : ℕ

/-- Represents the ratio of young to elderly teachers -/
structure TeacherRatio where
  young : ℕ
  elderly : ℕ

/-- 
Given a stratified sample of teachers where:
- The ratio of young to elderly teachers is 16:9
- There are 320 young teachers in the sample
Prove that there are 180 elderly teachers in the sample
-/
theorem stratified_sample_elderly_count 
  (ratio : TeacherRatio) 
  (sample : TeacherSample) :
  ratio.young = 16 →
  ratio.elderly = 9 →
  sample.young = 320 →
  (ratio.young : ℚ) / ratio.elderly = sample.young / sample.elderly →
  sample.elderly = 180 := by
sorry

end stratified_sample_elderly_count_l3276_327696


namespace attendance_scientific_notation_l3276_327656

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  mantissa : ℝ
  exponent : ℤ
  mantissa_range : 1 ≤ mantissa ∧ mantissa < 10

/-- Convert a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem attendance_scientific_notation :
  toScientificNotation 204000 = ScientificNotation.mk 2.04 5 (by norm_num) :=
sorry

end attendance_scientific_notation_l3276_327656


namespace exam_score_proof_l3276_327690

theorem exam_score_proof (total_questions : ℕ) 
                         (correct_score wrong_score : ℤ) 
                         (total_score : ℤ) 
                         (h1 : total_questions = 100)
                         (h2 : correct_score = 5)
                         (h3 : wrong_score = -2)
                         (h4 : total_score = 150) :
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_score * correct_answers + 
    wrong_score * (total_questions - correct_answers) = total_score ∧
    correct_answers = 50 :=
by sorry

end exam_score_proof_l3276_327690


namespace milk_cisterns_l3276_327605

theorem milk_cisterns (x y z : ℝ) (h1 : x + y + z = 780) 
  (h2 : (3/4) * x = (4/5) * y) (h3 : (3/4) * x = (4/7) * z) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  x = 240 ∧ y = 225 ∧ z = 315 := by
  sorry

end milk_cisterns_l3276_327605


namespace game_cost_l3276_327695

/-- 
Given:
- Frank's initial money: 11 dollars
- Frank's allowance: 14 dollars
- Frank's final money: 22 dollars

Prove that the cost of the new game is 3 dollars.
-/
theorem game_cost (initial_money : ℕ) (allowance : ℕ) (final_money : ℕ)
  (h1 : initial_money = 11)
  (h2 : allowance = 14)
  (h3 : final_money = 22) :
  initial_money - (final_money - allowance) = 3 :=
by sorry

end game_cost_l3276_327695


namespace angle_in_fourth_quadrant_l3276_327652

theorem angle_in_fourth_quadrant (α : Real) 
  (h1 : Real.sin α < 0) (h2 : Real.tan α < 0) : 
  ∃ (x y : Real), x > 0 ∧ y < 0 ∧ Real.cos α = x ∧ Real.sin α = y :=
sorry

end angle_in_fourth_quadrant_l3276_327652


namespace quadratic_roots_and_triangle_perimeter_l3276_327674

/-- The quadratic equation in terms of x and k -/
def quadratic (x k : ℝ) : Prop :=
  x^2 - (3*k + 1)*x + 2*k^2 + 2*k = 0

/-- An isosceles triangle with side lengths a, b, and c -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : (a = b ∧ a ≠ c) ∨ (a = c ∧ a ≠ b) ∨ (b = c ∧ b ≠ a)

/-- The theorem to be proved -/
theorem quadratic_roots_and_triangle_perimeter :
  (∀ k : ℝ, ∃ x : ℝ, quadratic x k) ∧
  (∃ t : IsoscelesTriangle, 
    t.a = 6 ∧
    quadratic t.b (t.b/2) ∧
    quadratic t.c ((t.c - 1)/2) ∧
    (t.a + t.b + t.c = 16 ∨ t.a + t.b + t.c = 22)) := by
  sorry

end quadratic_roots_and_triangle_perimeter_l3276_327674


namespace plane_line_parallel_l3276_327682

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)  -- Line is subset of plane
variable (parallel : Line → Line → Prop) -- Lines are parallel
variable (parallel_plane : Line → Plane → Prop) -- Line is parallel to plane
variable (intersect : Plane → Plane → Line → Prop) -- Planes intersect in a line

-- State the theorem
theorem plane_line_parallel 
  (α β : Plane) (m n : Line) 
  (h1 : intersect α β m) 
  (h2 : parallel n m) 
  (h3 : ¬ subset n α) 
  (h4 : ¬ subset n β) : 
  parallel_plane n α ∧ parallel_plane n β :=
sorry

end plane_line_parallel_l3276_327682


namespace sqrt_meaningfulness_l3276_327640

theorem sqrt_meaningfulness (x : ℝ) : x = 5 → (2*x - 4 ≥ 0) ∧ 
  (x = -1 → ¬(2*x - 4 ≥ 0)) ∧ 
  (x = 0 → ¬(2*x - 4 ≥ 0)) ∧ 
  (x = 1 → ¬(2*x - 4 ≥ 0)) := by
  sorry

end sqrt_meaningfulness_l3276_327640
