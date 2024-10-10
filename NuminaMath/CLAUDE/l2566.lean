import Mathlib

namespace imaginary_part_of_z_l2566_256610

def i : ℂ := Complex.I

theorem imaginary_part_of_z (z : ℂ) (h : (i - 1) * z = i) : 
  z.im = -1/2 := by sorry

end imaginary_part_of_z_l2566_256610


namespace cloth_trimming_l2566_256664

theorem cloth_trimming (x : ℝ) : 
  x > 0 → 
  (x - 4) * (x - 3) = 120 → 
  x = 12 :=
by sorry

end cloth_trimming_l2566_256664


namespace ben_walking_time_l2566_256645

/-- Given that Ben walks at a constant speed and covers 3 km in 2 hours,
    prove that the time required to walk 12 km is 480 minutes. -/
theorem ben_walking_time (speed : ℝ) (h1 : speed > 0) : 
  (3 : ℝ) / speed = 2 → (12 : ℝ) / speed * 60 = 480 := by
sorry

end ben_walking_time_l2566_256645


namespace tobias_driveways_shoveled_l2566_256681

/-- Calculates the number of driveways Tobias shoveled given his earnings and expenses. -/
theorem tobias_driveways_shoveled (
  shoe_cost : ℕ)
  (saving_months : ℕ)
  (monthly_allowance : ℕ)
  (lawn_mowing_fee : ℕ)
  (driveway_shoveling_fee : ℕ)
  (change_after_purchase : ℕ)
  (lawns_mowed : ℕ)
  (h1 : shoe_cost = 95)
  (h2 : saving_months = 3)
  (h3 : monthly_allowance = 5)
  (h4 : lawn_mowing_fee = 15)
  (h5 : driveway_shoveling_fee = 7)
  (h6 : change_after_purchase = 15)
  (h7 : lawns_mowed = 4) :
  (shoe_cost + change_after_purchase
    - saving_months * monthly_allowance
    - lawns_mowed * lawn_mowing_fee) / driveway_shoveling_fee = 5 :=
by sorry


end tobias_driveways_shoveled_l2566_256681


namespace homework_problems_per_page_l2566_256639

theorem homework_problems_per_page 
  (total_problems : ℕ) 
  (finished_problems : ℕ) 
  (remaining_pages : ℕ) 
  (h1 : total_problems = 101)
  (h2 : finished_problems = 47)
  (h3 : remaining_pages = 6)
  (h4 : remaining_pages > 0)
  : (total_problems - finished_problems) / remaining_pages = 9 := by
  sorry

end homework_problems_per_page_l2566_256639


namespace systematic_sampling_first_two_numbers_l2566_256684

/-- Systematic sampling function that returns the nth sample number -/
def systematicSample (populationSize sampleSize n : ℕ) : ℕ :=
  (n - 1) * (populationSize / sampleSize)

/-- Theorem stating the first two sample numbers in the given systematic sampling scenario -/
theorem systematic_sampling_first_two_numbers
  (populationSize : ℕ)
  (sampleSize : ℕ)
  (lastSampleNumber : ℕ)
  (h1 : populationSize = 8000)
  (h2 : sampleSize = 50)
  (h3 : lastSampleNumber = 7900) :
  systematicSample populationSize sampleSize 1 = 159 ∧
  systematicSample populationSize sampleSize 2 = 319 := by
  sorry

#eval systematicSample 8000 50 1  -- Expected: 159
#eval systematicSample 8000 50 2  -- Expected: 319

end systematic_sampling_first_two_numbers_l2566_256684


namespace total_books_is_24_l2566_256676

/-- The number of boxes Victor bought -/
def num_boxes : ℕ := 8

/-- The number of books in each box -/
def books_per_box : ℕ := 3

/-- Theorem: The total number of books Victor bought is 24 -/
theorem total_books_is_24 : num_boxes * books_per_box = 24 := by
  sorry

end total_books_is_24_l2566_256676


namespace units_digit_of_sum_of_powers_l2566_256632

theorem units_digit_of_sum_of_powers : (47^4 + 28^4) % 10 = 7 := by
  sorry

end units_digit_of_sum_of_powers_l2566_256632


namespace orange_harvest_duration_l2566_256644

/-- The number of sacks of oranges harvested per day -/
def sacks_per_day : ℕ := 14

/-- The total number of sacks of oranges harvested -/
def total_sacks : ℕ := 56

/-- The number of days the harvest lasts -/
def harvest_days : ℕ := total_sacks / sacks_per_day

theorem orange_harvest_duration :
  harvest_days = 4 := by
  sorry

end orange_harvest_duration_l2566_256644


namespace equation_solution_l2566_256603

theorem equation_solution : ∃ x : ℝ, 0.05 * x + 0.07 * (30 + x) = 14.7 ∧ x = 105 := by
  sorry

end equation_solution_l2566_256603


namespace shortest_distance_between_circles_l2566_256688

/-- The shortest distance between two circles -/
theorem shortest_distance_between_circles : 
  let circle1 := {(x, y) : ℝ × ℝ | x^2 - 8*x + y^2 + 6*y + 9 = 0}
  let circle2 := {(x, y) : ℝ × ℝ | x^2 + 10*x + y^2 - 2*y + 25 = 0}
  ∃ d : ℝ, d = Real.sqrt 97 - 5 ∧ 
    ∀ p1 ∈ circle1, ∀ p2 ∈ circle2, d ≤ dist p1 p2 :=
by sorry

end shortest_distance_between_circles_l2566_256688


namespace min_vertical_distance_is_zero_l2566_256621

-- Define the functions
def f (x : ℝ) := abs x
def g (x : ℝ) := -x^2 - 5*x - 4

-- Define the vertical distance between the graphs
def vertical_distance (x : ℝ) := abs (f x - g x)

-- Theorem statement
theorem min_vertical_distance_is_zero :
  ∃ x : ℝ, vertical_distance x = 0 ∧ ∀ y : ℝ, vertical_distance y ≥ 0 :=
sorry

end min_vertical_distance_is_zero_l2566_256621


namespace solution_difference_l2566_256689

-- Define the equation
def equation (x : ℝ) : Prop :=
  (6 * x - 18) / (x^2 + 4 * x - 21) = x + 3

-- Define the theorem
theorem solution_difference (r s : ℝ) 
  (hr : equation r) 
  (hs : equation s) 
  (hdistinct : r ≠ s) 
  (horder : r > s) : 
  r - s = 2 := by
  sorry

end solution_difference_l2566_256689


namespace train_passing_jogger_time_train_passes_jogger_in_36_seconds_l2566_256600

/-- Time for a train to pass a jogger given their speeds, train length, and initial distance -/
theorem train_passing_jogger_time (jogger_speed train_speed : Real) 
  (train_length initial_distance : Real) : Real :=
  let jogger_speed_ms := jogger_speed * (1000 / 3600)
  let train_speed_ms := train_speed * (1000 / 3600)
  let relative_speed := train_speed_ms - jogger_speed_ms
  let total_distance := initial_distance + train_length
  total_distance / relative_speed

/-- Proof that the time for the train to pass the jogger is 36 seconds -/
theorem train_passes_jogger_in_36_seconds :
  train_passing_jogger_time 9 45 120 240 = 36 := by
  sorry

end train_passing_jogger_time_train_passes_jogger_in_36_seconds_l2566_256600


namespace part_one_part_two_l2566_256648

-- Define the quadratic function
def f (a b x : ℝ) := x^2 - a*x + b

-- Define the solution set condition
def solution_set (a b : ℝ) : Prop :=
  ∀ x, f a b x < 0 ↔ 2 < x ∧ x < 3

-- Part I
theorem part_one (a b : ℝ) (h : solution_set a b) : a + b = 11 := by
  sorry

-- Part II
def g (b c x : ℝ) := -x^2 + b*x + c

theorem part_two (b c : ℝ) (h1 : b = 6) 
  (h2 : ∀ x, g b c x ≤ 0) : c ≤ -9 := by
  sorry

end part_one_part_two_l2566_256648


namespace fifth_number_is_one_l2566_256637

def random_table : List (List Nat) := [
  [7816, 6572, 0802, 6314, 0702, 4369, 9728, 0198],
  [3204, 9234, 4935, 8200, 3623, 4869, 6938, 7481]
]

def is_valid_number (n : Nat) : Bool :=
  n ≥ 1 ∧ n ≤ 20

def extract_valid_numbers (lst : List Nat) : List Nat :=
  lst.filter (λ n => is_valid_number n)

def select_numbers (table : List (List Nat)) : List Nat :=
  let flattened := table.join
  let valid_numbers := extract_valid_numbers flattened
  valid_numbers.take 5

theorem fifth_number_is_one :
  (select_numbers random_table).get? 4 = some 1 :=
sorry

end fifth_number_is_one_l2566_256637


namespace lunch_cakes_count_cakes_sum_equals_total_l2566_256605

/-- The number of cakes served during lunch today -/
def lunch_cakes : ℕ := sorry

/-- The number of cakes served during dinner today -/
def dinner_cakes : ℕ := 6

/-- The number of cakes served yesterday -/
def yesterday_cakes : ℕ := 3

/-- The total number of cakes served -/
def total_cakes : ℕ := 14

/-- Theorem stating that the number of cakes served during lunch today is 5 -/
theorem lunch_cakes_count : lunch_cakes = 5 := by
  sorry

/-- Theorem proving that the sum of cakes served equals the total -/
theorem cakes_sum_equals_total : lunch_cakes + dinner_cakes + yesterday_cakes = total_cakes := by
  sorry

end lunch_cakes_count_cakes_sum_equals_total_l2566_256605


namespace smallest_a_value_l2566_256611

theorem smallest_a_value (a : ℝ) (h_a_pos : a > 0) : 
  (∀ x > 0, x + a / x ≥ 4) ↔ a ≥ 4 :=
by sorry

end smallest_a_value_l2566_256611


namespace function_zero_between_consecutive_integers_l2566_256674

theorem function_zero_between_consecutive_integers :
  ∃ (a b : ℤ), 
    (∀ x ∈ Set.Ioo a b, (Real.log x + x - 3 : ℝ) ≠ 0) ∧
    b = a + 1 ∧
    a + b = 5 := by
  sorry

end function_zero_between_consecutive_integers_l2566_256674


namespace construction_time_correct_l2566_256627

/-- Represents the construction project with given parameters -/
structure ConstructionProject where
  initialWorkers : ℕ
  additionalWorkers : ℕ
  additionalWorkersStartDay : ℕ
  delayWithoutAdditionalWorkers : ℕ

/-- The planned construction time in days -/
def plannedConstructionTime (project : ConstructionProject) : ℕ := 110

theorem construction_time_correct (project : ConstructionProject) 
  (h1 : project.initialWorkers = 100)
  (h2 : project.additionalWorkers = 100)
  (h3 : project.additionalWorkersStartDay = 10)
  (h4 : project.delayWithoutAdditionalWorkers = 90) :
  plannedConstructionTime project = 110 := by
  sorry

#check construction_time_correct

end construction_time_correct_l2566_256627


namespace total_metal_needed_l2566_256641

/-- Given that Charlie has 276 lbs of metal in storage and needs to buy an additional 359 lbs,
    prove that the total amount of metal he needs for the wings is 635 lbs. -/
theorem total_metal_needed (storage : ℕ) (additional : ℕ) (total : ℕ) 
    (h1 : storage = 276)
    (h2 : additional = 359)
    (h3 : total = storage + additional) : 
  total = 635 := by
  sorry

end total_metal_needed_l2566_256641


namespace increase_by_percentage_l2566_256685

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 70 ∧ percentage = 50 → final = initial * (1 + percentage / 100) → final = 105 := by
  sorry

end increase_by_percentage_l2566_256685


namespace modulus_of_complex_square_l2566_256668

theorem modulus_of_complex_square : ∃ (z : ℂ), z = (3 - Complex.I)^2 ∧ Complex.abs z = 10 := by
  sorry

end modulus_of_complex_square_l2566_256668


namespace integral_sin_cos_sin_l2566_256693

open Real

theorem integral_sin_cos_sin (x : ℝ) :
  ∃ C : ℝ, ∫ t, sin t * cos (2*t) * sin (5*t) = 
    (1/24) * sin (6*x) - (1/32) * sin (8*x) - (1/8) * sin (2*x) + (1/16) * sin (4*x) + C :=
by
  sorry

end integral_sin_cos_sin_l2566_256693


namespace sqrt_200_equals_10_l2566_256696

theorem sqrt_200_equals_10 : Real.sqrt 200 = 10 := by
  sorry

end sqrt_200_equals_10_l2566_256696


namespace us_apples_sold_fresh_l2566_256656

/-- Calculates the amount of apples sold fresh given total production and mixing percentage -/
def apples_sold_fresh (total_production : ℝ) (mixing_percentage : ℝ) : ℝ :=
  let remaining := total_production * (1 - mixing_percentage)
  remaining * 0.4

/-- Theorem stating that given the U.S. apple production conditions, 
    the amount of apples sold fresh is 2.24 million tons -/
theorem us_apples_sold_fresh :
  apples_sold_fresh 8 0.3 = 2.24 := by
  sorry

#eval apples_sold_fresh 8 0.3

end us_apples_sold_fresh_l2566_256656


namespace no_nonzero_triple_sum_equals_third_l2566_256617

theorem no_nonzero_triple_sum_equals_third : 
  ¬∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
    (a + b = c) ∧ (b + c = a) ∧ (c + a = b) := by
  sorry

end no_nonzero_triple_sum_equals_third_l2566_256617


namespace triangle_side_length_l2566_256640

/-- Triangle ABC with sides a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- Theorem: In triangle ABC, if AB = 2, BC = 5, and the perimeter is even, then AC = 5 -/
theorem triangle_side_length (t : Triangle) 
  (h1 : t.a = 2)
  (h2 : t.b = 5)
  (h3 : ∃ n : ℕ, t.perimeter = 2 * n) :
  t.c = 5 := by
  sorry

end triangle_side_length_l2566_256640


namespace circle_line_slope_range_l2566_256690

/-- Given a circle and a line, if there are at least three distinct points on the circle
    with a specific distance from the line, then the slope of the line is within a certain range. -/
theorem circle_line_slope_range (a b : ℝ) :
  let circle := fun (x y : ℝ) => x^2 + y^2 - 4*x - 4*y - 10 = 0
  let line := fun (x y : ℝ) => a*x + b*y = 0
  let k := -a/b  -- slope of the line
  let distance_point_to_line := fun (x y : ℝ) => |a*x + b*y| / Real.sqrt (a^2 + b^2)
  (∃ (p q r : ℝ × ℝ), p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    circle p.1 p.2 ∧ circle q.1 q.2 ∧ circle r.1 r.2 ∧
    distance_point_to_line p.1 p.2 = 2 * Real.sqrt 2 ∧
    distance_point_to_line q.1 q.2 = 2 * Real.sqrt 2 ∧
    distance_point_to_line r.1 r.2 = 2 * Real.sqrt 2) →
  2 - Real.sqrt 3 ≤ k ∧ k ≤ 2 + Real.sqrt 3 :=
by sorry

end circle_line_slope_range_l2566_256690


namespace vectors_not_basis_l2566_256679

/-- Two vectors are non-collinear if they are not scalar multiples of each other -/
def NonCollinear (v w : ℝ × ℝ) : Prop :=
  ∀ (c : ℝ), v ≠ c • w

/-- Two vectors are linearly dependent if one is a scalar multiple of the other -/
def LinearlyDependent (v w : ℝ × ℝ) : Prop :=
  ∃ (c : ℝ), v = c • w

theorem vectors_not_basis (e₁ e₂ : ℝ × ℝ) (h : NonCollinear e₁ e₂) :
  LinearlyDependent (e₁ + 3 • e₂) (6 • e₂ + 2 • e₁) :=
sorry

end vectors_not_basis_l2566_256679


namespace house_sale_revenue_distribution_l2566_256678

theorem house_sale_revenue_distribution (market_value : ℝ) (selling_price_percentage : ℝ) 
  (num_people : ℕ) (tax_rate : ℝ) (individual_share : ℝ) : 
  market_value = 500000 →
  selling_price_percentage = 1.20 →
  num_people = 4 →
  tax_rate = 0.10 →
  individual_share = (market_value * selling_price_percentage * (1 - tax_rate)) / num_people →
  individual_share = 135000 := by
  sorry

end house_sale_revenue_distribution_l2566_256678


namespace milk_fraction_is_two_thirds_l2566_256667

/-- Represents the content of a cup --/
structure CupContent where
  milk : ℚ
  honey : ℚ

/-- Performs the transfers between cups as described in the problem --/
def performTransfers (initial1 initial2 : CupContent) : CupContent × CupContent :=
  let afterFirstTransfer1 := CupContent.mk (initial1.milk / 2) 0
  let afterFirstTransfer2 := CupContent.mk (initial1.milk / 2) initial2.honey
  
  let totalSecond := afterFirstTransfer2.milk + afterFirstTransfer2.honey
  let secondToFirst := totalSecond / 2
  let milkRatio := afterFirstTransfer2.milk / totalSecond
  
  let afterSecondTransfer1 := CupContent.mk 
    (afterFirstTransfer1.milk + secondToFirst * milkRatio)
    (secondToFirst * (1 - milkRatio))
  let afterSecondTransfer2 := CupContent.mk 
    (afterFirstTransfer2.milk - secondToFirst * milkRatio)
    (afterFirstTransfer2.honey - secondToFirst * (1 - milkRatio))
  
  let thirdTransferAmount := (afterSecondTransfer1.milk + afterSecondTransfer1.honey) / 3
  let finalFirst := CupContent.mk 
    (afterSecondTransfer1.milk - thirdTransferAmount)
    afterSecondTransfer1.honey
  let finalSecond := CupContent.mk 
    (afterSecondTransfer2.milk + thirdTransferAmount)
    afterSecondTransfer2.honey
  
  (finalFirst, finalSecond)

/-- The main theorem stating that the fraction of milk in the second cup is 2/3 after transfers --/
theorem milk_fraction_is_two_thirds :
  let initial1 := CupContent.mk 8 0
  let initial2 := CupContent.mk 0 6
  let (_, finalSecond) := performTransfers initial1 initial2
  finalSecond.milk / (finalSecond.milk + finalSecond.honey) = 2/3 := by
  sorry


end milk_fraction_is_two_thirds_l2566_256667


namespace problem_solution_l2566_256638

variable {S : Type*} [Inhabited S] [Nontrivial S]
variable (mul : S → S → S)

axiom mul_def : ∀ (a b : S), mul a (mul b a) = b

theorem problem_solution :
  (∀ (b : S), mul b (mul b b) = b) ∧
  (∀ (a b : S), mul (mul a b) (mul b (mul a b)) = b) :=
by sorry

end problem_solution_l2566_256638


namespace correct_selection_methods_l2566_256620

/-- The number of ways to select students for health checks -/
def select_students (total_students : ℕ) (num_classes : ℕ) (min_per_class : ℕ) : ℕ :=
  -- Definition to be implemented
  sorry

/-- Theorem stating the correct number of selection methods -/
theorem correct_selection_methods :
  select_students 23 10 2 = 220 := by
  sorry

end correct_selection_methods_l2566_256620


namespace correct_stratified_sample_l2566_256695

/-- Represents the population sizes for each age group -/
structure PopulationSizes where
  young : ℕ
  middleAged : ℕ
  elderly : ℕ

/-- Represents the sample sizes for each age group -/
structure SampleSizes where
  young : ℕ
  middleAged : ℕ
  elderly : ℕ

/-- Calculates the stratified sample sizes given population sizes and total sample size -/
def stratifiedSampleSizes (pop : PopulationSizes) (totalSample : ℕ) : SampleSizes :=
  { young := (pop.young * totalSample) / (pop.young + pop.middleAged + pop.elderly),
    middleAged := (pop.middleAged * totalSample) / (pop.young + pop.middleAged + pop.elderly),
    elderly := (pop.elderly * totalSample) / (pop.young + pop.middleAged + pop.elderly) }

theorem correct_stratified_sample (pop : PopulationSizes) (totalSample : ℕ) :
  pop.young = 45 ∧ pop.middleAged = 25 ∧ pop.elderly = 30 ∧ totalSample = 20 →
  let sample := stratifiedSampleSizes pop totalSample
  sample.young = 9 ∧ sample.middleAged = 5 ∧ sample.elderly = 6 := by
  sorry

end correct_stratified_sample_l2566_256695


namespace average_age_of_three_l2566_256612

/-- The average age of three people given the average age of two of them and the age of the third -/
theorem average_age_of_three (age_a : ℝ) (age_b : ℝ) (age_c : ℝ) 
  (h1 : (age_a + age_c) / 2 = 29) 
  (h2 : age_b = 23) : 
  (age_a + age_b + age_c) / 3 = 27 := by
  sorry


end average_age_of_three_l2566_256612


namespace algebra_test_average_l2566_256651

theorem algebra_test_average (total_students : ℕ) (male_students : ℕ) (female_students : ℕ)
  (total_average : ℚ) (male_average : ℚ) :
  total_students = male_students + female_students →
  total_students = 36 →
  male_students = 8 →
  female_students = 28 →
  total_average = 90 →
  male_average = 83 →
  (total_students : ℚ) * total_average = 
    (male_students : ℚ) * male_average + (female_students : ℚ) * ((3240 - 664 : ℚ) / 28) :=
by sorry

end algebra_test_average_l2566_256651


namespace edward_total_spent_l2566_256625

/-- The total amount Edward spent on a board game and action figures -/
def total_spent (board_game_cost : ℕ) (num_figures : ℕ) (figure_cost : ℕ) : ℕ :=
  board_game_cost + num_figures * figure_cost

/-- Theorem stating that Edward spent $30 in total -/
theorem edward_total_spent :
  total_spent 2 4 7 = 30 := by
  sorry

end edward_total_spent_l2566_256625


namespace equal_area_division_l2566_256670

-- Define a type for points in a plane
variable (Point : Type)

-- Define a type for lines in a plane
variable (Line : Type)

-- Define a type for figures in a plane
variable (Figure : Type)

-- Function to check if two lines are parallel
variable (parallel : Line → Line → Prop)

-- Function to measure the area of a figure
variable (area : Figure → ℝ)

-- Function to get the part of a figure on one side of a line
variable (figurePart : Figure → Line → Figure)

-- Theorem statement
theorem equal_area_division 
  (Φ : Figure) (l₀ : Line) : 
  ∃ (l : Line), 
    parallel l l₀ ∧ 
    area (figurePart Φ l) = (area Φ) / 2 := by
  sorry

end equal_area_division_l2566_256670


namespace cistern_wet_surface_area_l2566_256631

/-- Calculates the total wet surface area of a rectangular cistern. -/
def totalWetSurfaceArea (length width depth : ℝ) : ℝ :=
  length * width + 2 * length * depth + 2 * width * depth

/-- Theorem: The total wet surface area of a cistern with given dimensions is 62 square meters. -/
theorem cistern_wet_surface_area :
  totalWetSurfaceArea 8 4 1.25 = 62 := by
  sorry

end cistern_wet_surface_area_l2566_256631


namespace music_stand_cost_l2566_256669

/-- The cost of Jason's music stand purchase --/
theorem music_stand_cost (flute_cost song_book_cost total_spent : ℝ) 
  (h1 : flute_cost = 142.46)
  (h2 : song_book_cost = 7)
  (h3 : total_spent = 158.35) :
  total_spent - (flute_cost + song_book_cost) = 8.89 := by
  sorry

end music_stand_cost_l2566_256669


namespace pentagon_diagonals_through_vertex_l2566_256686

/-- The number of diagonals passing through a vertex in a pentagon -/
def diagonals_through_vertex_in_pentagon : ℕ :=
  (5 : ℕ) - 3

theorem pentagon_diagonals_through_vertex :
  diagonals_through_vertex_in_pentagon = 2 :=
by sorry

end pentagon_diagonals_through_vertex_l2566_256686


namespace one_ounce_bottle_caps_count_l2566_256680

/-- The number of one-ounce bottle caps in a collection -/
def oneOunceBottleCaps (totalWeight : ℕ) (totalCaps : ℕ) : ℕ :=
  totalWeight * 16

/-- Theorem: The number of one-ounce bottle caps is equal to the total weight in ounces -/
theorem one_ounce_bottle_caps_count 
  (totalWeight : ℕ) 
  (totalCaps : ℕ) 
  (h1 : totalWeight = 18) 
  (h2 : totalCaps = 2016) : 
  oneOunceBottleCaps totalWeight totalCaps = totalWeight * 16 :=
by sorry

end one_ounce_bottle_caps_count_l2566_256680


namespace quadrilateral_area_at_least_30_l2566_256607

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)

-- Define the conditions
def is_valid_quadrilateral (q : Quadrilateral) : Prop :=
  let (ex, ey) := q.E
  let (fx, fy) := q.F
  let (gx, gy) := q.G
  let (hx, hy) := q.H
  -- EF = 5
  (ex - fx)^2 + (ey - fy)^2 = 25 ∧
  -- FG = 12
  (fx - gx)^2 + (fy - gy)^2 = 144 ∧
  -- GH = 5
  (gx - hx)^2 + (gy - hy)^2 = 25 ∧
  -- HE = 13
  (hx - ex)^2 + (hy - ey)^2 = 169 ∧
  -- Angle EFG is a right angle
  (ex - fx) * (gx - fx) + (ey - fy) * (gy - fy) = 0

-- Define the area function
def area (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_area_at_least_30 (q : Quadrilateral) :
  is_valid_quadrilateral q → area q ≥ 30 := by sorry

end quadrilateral_area_at_least_30_l2566_256607


namespace percentage_problem_l2566_256692

theorem percentage_problem (p : ℝ) : p * 50 / 100 = 200 → p = 400 := by
  sorry

end percentage_problem_l2566_256692


namespace count_integers_in_range_l2566_256672

theorem count_integers_in_range : 
  ∃! n : ℕ, n = (Finset.filter (fun x : ℕ => 
    50 < x^2 + 6*x + 9 ∧ x^2 + 6*x + 9 < 100) (Finset.range 100)).card ∧ n = 3 := by
  sorry

end count_integers_in_range_l2566_256672


namespace unique_cyclic_number_l2566_256657

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n < 1000000

def same_digits (a b : ℕ) : Prop :=
  ∀ d : ℕ, d < 10 → (∃ k, a / 10^k % 10 = d) ↔ (∃ k, b / 10^k % 10 = d)

theorem unique_cyclic_number : 
  ∃! N : ℕ, is_six_digit N ∧ 
    (∀ k : ℕ, 2 ≤ k ∧ k ≤ 6 → 
      is_six_digit (k * N) ∧ 
      same_digits N (k * N) ∧ 
      N ≠ k * N) ∧
    N = 142857 :=
sorry

end unique_cyclic_number_l2566_256657


namespace length_of_chord_AB_equation_of_line_PQ_l2566_256677

-- Define the circles and line
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4
def line_l (x y : ℝ) : Prop := x - y + 2 = 0
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 2*x + 2*Real.sqrt 3*y

-- Theorem for the length of chord AB
theorem length_of_chord_AB :
  ∃ (A B : ℝ × ℝ),
    circle_O A.1 A.2 ∧ circle_O B.1 B.2 ∧
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 :=
sorry

-- Theorem for the equation of line PQ
theorem equation_of_line_PQ :
  ∃ (P Q : ℝ × ℝ),
    circle_O P.1 P.2 ∧ circle_O Q.1 Q.2 ∧
    circle_C P.1 P.2 ∧ circle_C Q.1 Q.2 ∧
    ∀ (x y : ℝ), (x - P.1) * (Q.2 - P.2) = (y - P.2) * (Q.1 - P.1) ↔
      x + Real.sqrt 3 * y - 2 = 0 :=
sorry

end length_of_chord_AB_equation_of_line_PQ_l2566_256677


namespace six_double_prime_value_l2566_256665

-- Define the prime operation
def prime (q : ℝ) : ℝ := 3 * q - 3

-- State the theorem
theorem six_double_prime_value : prime (prime 6) = 42 := by
  sorry

end six_double_prime_value_l2566_256665


namespace quilt_shaded_fraction_l2566_256615

/-- Represents a quilt block as described in the problem -/
structure QuiltBlock where
  size : Nat
  fully_shaded : Nat
  half_shaded : Nat
  quarter_shaded : Nat

/-- The fraction of the quilt that is shaded -/
def shaded_fraction (q : QuiltBlock) : Rat :=
  (q.fully_shaded + q.half_shaded / 2 + q.quarter_shaded / 2) / (q.size * q.size)

/-- The specific quilt block described in the problem -/
def problem_quilt : QuiltBlock :=
  { size := 4
    fully_shaded := 4
    half_shaded := 8
    quarter_shaded := 4 }

theorem quilt_shaded_fraction :
  shaded_fraction problem_quilt = 5 / 8 := by sorry

end quilt_shaded_fraction_l2566_256615


namespace find_t_l2566_256675

theorem find_t : ∃ t : ℝ, 
  (∀ x : ℝ, |2*x + t| - t ≤ 8 ↔ -5 ≤ x ∧ x ≤ 4) → 
  t = 1 := by sorry

end find_t_l2566_256675


namespace finger_2004_is_index_l2566_256636

def finger_sequence : ℕ → String
| 0 => "pinky"
| 1 => "ring"
| 2 => "middle"
| 3 => "index"
| 4 => "thumb"
| 5 => "index"
| 6 => "middle"
| 7 => "ring"
| n + 8 => finger_sequence n

theorem finger_2004_is_index : finger_sequence 2003 = "index" := by
  sorry

end finger_2004_is_index_l2566_256636


namespace unique_magnitude_for_complex_roots_l2566_256699

theorem unique_magnitude_for_complex_roots (x : ℂ) : 
  x^2 - 4*x + 29 = 0 → ∃! m : ℝ, ∃ y : ℂ, y^2 - 4*y + 29 = 0 ∧ Complex.abs y = m :=
by sorry

end unique_magnitude_for_complex_roots_l2566_256699


namespace linda_original_amount_l2566_256687

/-- The amount of money Lucy originally had -/
def lucy_original : ℕ := 20

/-- The amount of money Linda originally had -/
def linda_original : ℕ := 10

/-- The amount of money Lucy would give to Linda -/
def transfer_amount : ℕ := 5

theorem linda_original_amount : 
  (lucy_original - transfer_amount = linda_original + transfer_amount) →
  linda_original = 10 := by
sorry

end linda_original_amount_l2566_256687


namespace square_difference_equals_sixteen_l2566_256691

theorem square_difference_equals_sixteen
  (x y : ℝ)
  (sum_eq : x + y = 6)
  (product_eq : x * y = 5) :
  (x - y)^2 = 16 := by
  sorry

end square_difference_equals_sixteen_l2566_256691


namespace expression_evaluation_l2566_256604

theorem expression_evaluation : (3^(2+3+4) - (3^2 * 3^3 + 3^4)) = 19359 := by
  sorry

end expression_evaluation_l2566_256604


namespace basic_computer_price_l2566_256654

theorem basic_computer_price 
  (total_price : ℝ) 
  (enhanced_computer_diff : ℝ) 
  (printer_ratio : ℝ) :
  total_price = 2500 →
  enhanced_computer_diff = 500 →
  printer_ratio = 1/4 →
  ∃ (basic_computer : ℝ) (printer : ℝ),
    basic_computer + printer = total_price ∧
    printer = printer_ratio * (basic_computer + enhanced_computer_diff + printer) ∧
    basic_computer = 1750 :=
by sorry

end basic_computer_price_l2566_256654


namespace range_of_r_l2566_256606

theorem range_of_r (a b c r : ℝ) 
  (h1 : b + c ≤ 4 * a)
  (h2 : c - b ≥ 0)
  (h3 : b ≥ a)
  (h4 : a > 0)
  (h5 : r > 0)
  (h6 : (a + b)^2 + (a + c)^2 ≠ (a * r)^2) :
  r ∈ Set.Ioo 0 (2 * Real.sqrt 2) ∪ Set.Ioi (3 * Real.sqrt 2) :=
sorry

end range_of_r_l2566_256606


namespace range_of_shifted_f_l2566_256694

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h1 : ∀ x, f x ∈ Set.Icc 1 2)
variable (h2 : Set.range f = Set.Icc 1 2)

-- State the theorem
theorem range_of_shifted_f :
  Set.range (fun x ↦ f (x + 1)) = Set.Icc 1 2 := by
  sorry

end range_of_shifted_f_l2566_256694


namespace product_of_reals_l2566_256629

theorem product_of_reals (a b : ℝ) 
  (sum_eq : a + b = 7)
  (sum_cubes_eq : a^3 + b^3 = 91) : 
  a * b = 12 := by sorry

end product_of_reals_l2566_256629


namespace digital_earth_correct_application_l2566_256601

/-- Represents the capabilities of the digital Earth -/
structure DigitalEarthCapabilities where
  resourceOptimization : Bool
  informationAccess : Bool

/-- Represents possible applications of the digital Earth -/
inductive DigitalEarthApplication
  | crimeControl
  | projectDecisionSupport
  | precipitationControl
  | disasterControl

/-- Determines if an application is correct given the capabilities of the digital Earth -/
def isCorrectApplication (capabilities : DigitalEarthCapabilities) (application : DigitalEarthApplication) : Prop :=
  capabilities.resourceOptimization ∧ 
  capabilities.informationAccess ∧ 
  application = DigitalEarthApplication.projectDecisionSupport

theorem digital_earth_correct_application (capabilities : DigitalEarthCapabilities) 
  (h1 : capabilities.resourceOptimization = true) 
  (h2 : capabilities.informationAccess = true) :
  isCorrectApplication capabilities DigitalEarthApplication.projectDecisionSupport :=
sorry

end digital_earth_correct_application_l2566_256601


namespace extreme_values_and_monotonicity_l2566_256646

-- Define the function f
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- State the theorem
theorem extreme_values_and_monotonicity 
  (a b c : ℝ) 
  (h1 : ∃ (y : ℝ), y = f a b c (-2) ∧ (∀ x, f a b c x ≤ y))
  (h2 : ∃ (y : ℝ), y = f a b c 1 ∧ (∀ x, f a b c x ≤ y))
  (h3 : ∀ x ∈ Set.Icc (-1) 2, f a b c x < c^2) :
  (a = 3/2 ∧ b = -6) ∧ 
  (∀ x < -2, ∀ y ∈ Set.Ioo x (-2), f a b c x < f a b c y) ∧
  (∀ x ∈ Set.Ioo (-2) 1, ∀ y ∈ Set.Ioo x 1, f a b c x > f a b c y) ∧
  (∀ x > 1, ∀ y ∈ Set.Ioo 1 x, f a b c x > f a b c y) ∧
  (c > 2 ∨ c < -1) := by
  sorry


end extreme_values_and_monotonicity_l2566_256646


namespace well_diameter_l2566_256634

theorem well_diameter (depth : ℝ) (volume : ℝ) (diameter : ℝ) : 
  depth = 14 →
  volume = 43.982297150257104 →
  volume = Real.pi * (diameter / 2)^2 * depth →
  diameter = 2 := by
sorry

end well_diameter_l2566_256634


namespace arccos_sqrt3_over_2_l2566_256642

theorem arccos_sqrt3_over_2 : Real.arccos (Real.sqrt 3 / 2) = π / 6 := by
  sorry

end arccos_sqrt3_over_2_l2566_256642


namespace product_sum_theorem_l2566_256652

theorem product_sum_theorem (a b c d e : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e →
  (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 120 →
  a + b + c + d + e = 33 := by
sorry

end product_sum_theorem_l2566_256652


namespace three_digit_automorphic_numbers_l2566_256613

theorem three_digit_automorphic_numbers : 
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ n^2 % 1000 = n} = {625, 376} := by sorry

end three_digit_automorphic_numbers_l2566_256613


namespace faulty_clock_correct_time_fraction_l2566_256623

/-- Represents a 12-hour digital clock with a faulty display of '2' as '5' -/
structure FaultyClock where
  /-- The number of hours that display correctly -/
  correct_hours : ℕ
  /-- The number of minutes per hour that display correctly -/
  correct_minutes : ℕ

/-- The fraction of the day during which the faulty clock displays the correct time -/
def correct_time_fraction (clock : FaultyClock) : ℚ :=
  (clock.correct_hours : ℚ) / 12 * (clock.correct_minutes : ℚ) / 60

/-- The specific faulty clock described in the problem -/
def problem_clock : FaultyClock := {
  correct_hours := 10,
  correct_minutes := 44
}

theorem faulty_clock_correct_time_fraction :
  correct_time_fraction problem_clock = 11 / 18 := by
  sorry

end faulty_clock_correct_time_fraction_l2566_256623


namespace sequence_sum_property_l2566_256697

/-- A sequence of positive terms satisfying a specific equation. -/
def sequence_a (n : ℕ) : ℝ :=
  sorry

/-- The sum of the first n terms of the sequence. -/
def S (n : ℕ) : ℝ :=
  sorry

/-- The main theorem stating the property of the sequence sum. -/
theorem sequence_sum_property :
  ∀ (n : ℕ), n ≥ 1 →
  (n * (n + 1) * (sequence_a n)^2 + (n^2 + n - 1) * sequence_a n - 1 = 0) →
  sequence_a n > 0 →
  2019 * S 2018 = 2018 :=
sorry

end sequence_sum_property_l2566_256697


namespace quadratic_vertex_property_l2566_256618

/-- Given a quadratic function y = x^2 - 2x + n with vertex (m, 1), prove that m - n = -1 -/
theorem quadratic_vertex_property (n m : ℝ) : 
  (∀ x, x^2 - 2*x + n = (x - m)^2 + 1) → m - n = -1 := by
  sorry

end quadratic_vertex_property_l2566_256618


namespace triangle_construction_cases_l2566_256650

/-- A triangle with side lengths a, b, c and angles A, B, C. --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The height of a triangle from vertex A to side BC. --/
def height_A (t : Triangle) : ℝ := sorry

/-- The height of a triangle from vertex C to side AB. --/
def height_C (t : Triangle) : ℝ := sorry

/-- Constructs triangles given side AB, height CC₁, and angle A. --/
def construct_ABC_CC1_A (c : ℝ) (h : ℝ) (α : ℝ) : Set Triangle := sorry

/-- Constructs triangles given side AB, height CC₁, and angle C. --/
def construct_ABC_CC1_C (c : ℝ) (h : ℝ) (γ : ℝ) : Set Triangle := sorry

/-- Constructs triangles given side AB, height AA₁, and angle A. --/
def construct_ABC_AA1_A (c : ℝ) (h : ℝ) (α : ℝ) : Set Triangle := sorry

/-- Constructs triangles given side AB, height AA₁, and angle B. --/
def construct_ABC_AA1_B (c : ℝ) (h : ℝ) (β : ℝ) : Set Triangle := sorry

/-- Constructs triangles given side AB, height AA₁, and angle C. --/
def construct_ABC_AA1_C (c : ℝ) (h : ℝ) (γ : ℝ) : Set Triangle := sorry

/-- The total number of distinct triangles that can be constructed from all cases. --/
def total_distinct_triangles : ℕ := sorry

theorem triangle_construction_cases :
  ∀ (c h α β γ : ℝ),
    c > 0 → h > 0 → 0 < α < π → 0 < β < π → 0 < γ < π →
    total_distinct_triangles = 11 := by sorry

end triangle_construction_cases_l2566_256650


namespace regular_polygon_angle_sum_l2566_256649

/-- For a regular polygon with n sides, if the sum of its interior angles
    is 4 times the sum of its exterior angles, then n = 10 -/
theorem regular_polygon_angle_sum (n : ℕ) : n ≥ 3 →
  (n - 2) * 180 = 4 * 360 → n = 10 := by
  sorry

end regular_polygon_angle_sum_l2566_256649


namespace stratified_sample_grad_count_l2566_256647

/-- Represents the number of students to be sampled from each stratum in a stratified sampling -/
structure StratifiedSample where
  total : ℕ
  junior : ℕ
  undergrad : ℕ
  grad : ℕ

/-- Calculates the stratified sample given the total population and sample size -/
def calculateStratifiedSample (totalPopulation : ℕ) (juniorCount : ℕ) (undergradCount : ℕ) (sampleSize : ℕ) : StratifiedSample :=
  let gradCount := totalPopulation - juniorCount - undergradCount
  let sampleRatio := sampleSize / totalPopulation
  { total := sampleSize,
    junior := juniorCount * sampleRatio,
    undergrad := undergradCount * sampleRatio,
    grad := sampleSize - (juniorCount * sampleRatio) - (undergradCount * sampleRatio) }

theorem stratified_sample_grad_count 
  (totalPopulation : ℕ) (juniorCount : ℕ) (undergradCount : ℕ) (sampleSize : ℕ)
  (h1 : totalPopulation = 5600)
  (h2 : juniorCount = 1300)
  (h3 : undergradCount = 3000)
  (h4 : sampleSize = 280) :
  (calculateStratifiedSample totalPopulation juniorCount undergradCount sampleSize).grad = 65 := by
  sorry

#eval (calculateStratifiedSample 5600 1300 3000 280).grad

end stratified_sample_grad_count_l2566_256647


namespace m_range_l2566_256653

theorem m_range (m : ℝ) : 
  (∀ x : ℝ, (1 < x ∧ x < 2) → (m - 1 < x ∧ x < m + 1)) → 
  (1 ≤ m ∧ m ≤ 2) := by
sorry

end m_range_l2566_256653


namespace book_price_changes_l2566_256614

theorem book_price_changes (initial_price : ℝ) (decrease_percent : ℝ) (increase_percent : ℝ) (final_price : ℝ) : 
  initial_price = 400 →
  decrease_percent = 15 →
  increase_percent = 40 →
  final_price = 476 →
  initial_price * (1 - decrease_percent / 100) * (1 + increase_percent / 100) = final_price := by
sorry

end book_price_changes_l2566_256614


namespace total_material_ordered_l2566_256619

def concrete : ℝ := 0.17
def bricks : ℝ := 0.17
def stone : ℝ := 0.5

theorem total_material_ordered : concrete + bricks + stone = 0.84 := by
  sorry

end total_material_ordered_l2566_256619


namespace randy_pictures_l2566_256666

theorem randy_pictures (peter_pictures quincy_pictures randy_pictures total_pictures : ℕ) :
  peter_pictures = 8 →
  quincy_pictures = peter_pictures + 20 →
  total_pictures = 41 →
  total_pictures = peter_pictures + quincy_pictures + randy_pictures →
  randy_pictures = 5 := by
sorry

end randy_pictures_l2566_256666


namespace rectangle_area_bounds_l2566_256635

/-- Represents the reported dimension of a rectangular tile -/
structure ReportedDimension where
  value : ℝ
  min : ℝ := value - 1.0
  max : ℝ := value + 1.0

/-- Represents a rectangular tile with reported dimensions -/
structure ReportedRectangle where
  length : ReportedDimension
  width : ReportedDimension

/-- Calculates the minimum area of a reported rectangle -/
def minArea (rect : ReportedRectangle) : ℝ :=
  rect.length.min * rect.width.min

/-- Calculates the maximum area of a reported rectangle -/
def maxArea (rect : ReportedRectangle) : ℝ :=
  rect.length.max * rect.width.max

theorem rectangle_area_bounds :
  let rect : ReportedRectangle := {
    length := { value := 4 },
    width := { value := 6 }
  }
  minArea rect = 15.0 ∧ maxArea rect = 35.0 := by
  sorry

end rectangle_area_bounds_l2566_256635


namespace p_necessary_not_sufficient_l2566_256671

def p (x : ℝ) : Prop := (x^2 + 6*x + 8) * Real.sqrt (x + 3) ≥ 0

def q (x : ℝ) : Prop := x = -3

theorem p_necessary_not_sufficient :
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x) := by sorry

end p_necessary_not_sufficient_l2566_256671


namespace yoongi_has_fewest_apples_l2566_256658

def yoongi_apples : ℕ := 4
def yuna_apples : ℕ := 5
def jungkook_apples : ℕ := 6 * 3

theorem yoongi_has_fewest_apples :
  yoongi_apples < yuna_apples ∧ yoongi_apples < jungkook_apples := by
  sorry

end yoongi_has_fewest_apples_l2566_256658


namespace undecided_voters_percentage_l2566_256602

theorem undecided_voters_percentage
  (total_polled : ℕ)
  (biff_percentage : ℚ)
  (marty_voters : ℕ)
  (h1 : total_polled = 200)
  (h2 : biff_percentage = 45 / 100)
  (h3 : marty_voters = 94) :
  (total_polled - (marty_voters + (biff_percentage * total_polled).floor)) / total_polled = 8 / 100 := by
  sorry

end undecided_voters_percentage_l2566_256602


namespace min_value_problem_l2566_256616

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∃ (min : ℝ), min = 4 ∧ ∀ (x y : ℝ), x > 0 ∧ y > 0 ∧ 1/x + 1/y = 1 → 1/(x-1) + 4/(y-1) ≥ min := by
sorry

end min_value_problem_l2566_256616


namespace win_sector_area_l2566_256682

theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 8) (h2 : p = 1/4) :
  p * π * r^2 = 16 * π := by
  sorry

end win_sector_area_l2566_256682


namespace food_preferences_l2566_256683

theorem food_preferences (total students : ℕ)
  (french_fries burgers pizza tacos : ℕ)
  (fries_burgers fries_pizza fries_tacos : ℕ)
  (burgers_pizza burgers_tacos pizza_tacos : ℕ)
  (all_four : ℕ)
  (h_total : total = 30)
  (h_fries : french_fries = 18)
  (h_burgers : burgers = 12)
  (h_pizza : pizza = 14)
  (h_tacos : tacos = 10)
  (h_fries_burgers : fries_burgers = 8)
  (h_fries_pizza : fries_pizza = 6)
  (h_fries_tacos : fries_tacos = 4)
  (h_burgers_pizza : burgers_pizza = 5)
  (h_burgers_tacos : burgers_tacos = 3)
  (h_pizza_tacos : pizza_tacos = 7)
  (h_all_four : all_four = 2) :
  total - (french_fries + burgers + pizza + tacos
           - fries_burgers - fries_pizza - fries_tacos
           - burgers_pizza - burgers_tacos - pizza_tacos
           + all_four) = 11 := by
  sorry

end food_preferences_l2566_256683


namespace interest_difference_approximately_128_l2566_256660

-- Define the initial deposit
def initial_deposit : ℝ := 14000

-- Define the interest rates
def compound_rate : ℝ := 0.06
def simple_rate : ℝ := 0.08

-- Define the time period
def years : ℕ := 10

-- Define the compound interest function
def compound_interest (p r : ℝ) (n : ℕ) : ℝ := p * (1 + r) ^ n

-- Define the simple interest function
def simple_interest (p r : ℝ) (t : ℕ) : ℝ := p * (1 + r * t)

-- State the theorem
theorem interest_difference_approximately_128 :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧
  abs (simple_interest initial_deposit simple_rate years - 
       compound_interest initial_deposit compound_rate years - 128) < ε :=
sorry

end interest_difference_approximately_128_l2566_256660


namespace village_assistants_selection_l2566_256661

theorem village_assistants_selection (n : ℕ) (k : ℕ) (a b : ℕ) :
  n = 10 → k = 3 → a ≠ b → a ≤ n → b ≤ n →
  (Nat.choose n k - Nat.choose (n - 2) k) = 64 := by
  sorry

end village_assistants_selection_l2566_256661


namespace photo_arrangements_count_l2566_256659

/-- The number of ways to rearrange a group photo with the given conditions -/
def photo_arrangements : ℕ :=
  Nat.choose 8 2 * (5 * 4)

/-- Theorem stating that the number of photo arrangements is 560 -/
theorem photo_arrangements_count : photo_arrangements = 560 := by
  sorry

end photo_arrangements_count_l2566_256659


namespace minimum_guests_l2566_256608

theorem minimum_guests (total_food : ℝ) (max_per_guest : ℝ) (h1 : total_food = 406) (h2 : max_per_guest = 2.5) :
  ∃ n : ℕ, n * max_per_guest ≥ total_food ∧ ∀ m : ℕ, m * max_per_guest ≥ total_food → m ≥ n ∧ n = 163 :=
by sorry

end minimum_guests_l2566_256608


namespace symmetric_difference_of_M_and_N_l2566_256624

-- Define the symmetric difference operation
def symmetricDifference (A B : Set ℝ) : Set ℝ :=
  (A ∪ B) \ (A ∩ B)

-- Define sets M and N
def M : Set ℝ := {y | ∃ x, 0 < x ∧ x < 2 ∧ y = -x^2 + 2*x}
def N : Set ℝ := {y | ∃ x, x > 0 ∧ y = 2^(x-1)}

-- State the theorem
theorem symmetric_difference_of_M_and_N :
  symmetricDifference M N = {y | (0 < y ∧ y ≤ 1/2) ∨ (1 < y)} := by
  sorry

end symmetric_difference_of_M_and_N_l2566_256624


namespace sufficient_not_necessary_l2566_256609

theorem sufficient_not_necessary :
  (∀ x : ℝ, x - 1 > 0 → x^2 - 1 > 0) ∧
  (∃ x : ℝ, x^2 - 1 > 0 ∧ x - 1 ≤ 0) := by
  sorry

end sufficient_not_necessary_l2566_256609


namespace triangle_angle_sum_l2566_256673

theorem triangle_angle_sum (A B C : ℝ) (h1 : A + B + C = 180) (h2 : 180 - C = 130) :
  A + B = 130 := by sorry

end triangle_angle_sum_l2566_256673


namespace cubic_sum_theorem_l2566_256643

theorem cubic_sum_theorem (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : (a^3 - 12) / a = (b^3 - 12) / b ∧ (b^3 - 12) / b = (c^3 - 12) / c) : 
  a^3 + b^3 + c^3 = 36 := by
sorry

end cubic_sum_theorem_l2566_256643


namespace largest_number_in_ratio_l2566_256663

theorem largest_number_in_ratio (a b c : ℕ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 →
  (b = (5 * a) / 3) →
  (c = (7 * a) / 3) →
  (c - a = 32) →
  c = 56 := by
sorry

end largest_number_in_ratio_l2566_256663


namespace second_account_interest_rate_l2566_256628

/-- Proves that the interest rate of the second account is 0.1 given the problem conditions -/
theorem second_account_interest_rate 
  (total_investment : ℝ)
  (first_account_rate : ℝ)
  (first_account_investment : ℝ)
  (h_total : total_investment = 7200)
  (h_first_rate : first_account_rate = 0.08)
  (h_first_inv : first_account_investment = 4000)
  (h_equal_interest : first_account_rate * first_account_investment = 
    (total_investment - first_account_investment) * (10 / 100)) :
  (10 : ℝ) / 100 = (first_account_rate * first_account_investment) / (total_investment - first_account_investment) :=
by sorry

end second_account_interest_rate_l2566_256628


namespace total_volume_of_stacked_boxes_l2566_256655

/-- The volume of a single box in cubic centimeters -/
def single_box_volume : ℝ := 30

/-- The number of horizontal rows of boxes -/
def horizontal_rows : ℕ := 7

/-- The number of vertical rows of boxes -/
def vertical_rows : ℕ := 5

/-- The number of floors of boxes -/
def floors : ℕ := 3

/-- The total number of boxes -/
def total_boxes : ℕ := horizontal_rows * vertical_rows * floors

/-- The theorem stating the total volume of stacked boxes -/
theorem total_volume_of_stacked_boxes :
  (single_box_volume * total_boxes : ℝ) = 3150 := by sorry

end total_volume_of_stacked_boxes_l2566_256655


namespace soccer_team_selection_l2566_256698

theorem soccer_team_selection (total_players : ℕ) (quadruplets : ℕ) (starters : ℕ) (quad_starters : ℕ) :
  total_players = 16 →
  quadruplets = 4 →
  starters = 7 →
  quad_starters = 2 →
  (Nat.choose quadruplets quad_starters) * (Nat.choose (total_players - quadruplets) (starters - quad_starters)) = 4752 := by
  sorry

end soccer_team_selection_l2566_256698


namespace smallest_perfect_square_divisible_by_2_3_5_l2566_256626

theorem smallest_perfect_square_divisible_by_2_3_5 : 
  ∀ n : ℕ, n > 0 → n^2 < 900 → ¬(2 ∣ n^2 ∧ 3 ∣ n^2 ∧ 5 ∣ n^2) :=
by sorry

end smallest_perfect_square_divisible_by_2_3_5_l2566_256626


namespace rotation_of_negative_six_minus_three_i_l2566_256622

def rotate90Clockwise (z : ℂ) : ℂ := -Complex.I * z

theorem rotation_of_negative_six_minus_three_i :
  rotate90Clockwise (-6 - 3*Complex.I) = -3 + 6*Complex.I :=
by sorry

end rotation_of_negative_six_minus_three_i_l2566_256622


namespace result_line_properties_l2566_256662

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

/-- The given line equation -/
def given_line_eq (x y : ℝ) : Prop := 2*x + 3*y = 0

/-- The resulting line equation -/
def result_line_eq (x y : ℝ) : Prop := 3*x - 2*y + 7 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-1, 2)

/-- Theorem stating that the resulting line passes through the center of the circle
    and is perpendicular to the given line -/
theorem result_line_properties :
  result_line_eq (circle_center.1) (circle_center.2) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), given_line_eq x₁ y₁ → given_line_eq x₂ y₂ →
    result_line_eq x₁ y₁ → result_line_eq x₂ y₂ →
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((x₂ - x₁) * 2 + (y₂ - y₁) * 3) * ((x₂ - x₁) * 3 + (y₂ - y₁) * (-2)) = 0) :=
sorry

end result_line_properties_l2566_256662


namespace orthogonal_vectors_solution_l2566_256633

theorem orthogonal_vectors_solution :
  ∃! y : ℝ, (2 : ℝ) * (-1 : ℝ) + (-1 : ℝ) * y + (3 : ℝ) * (0 : ℝ) + (1 : ℝ) * (-4 : ℝ) = 0 :=
by sorry

end orthogonal_vectors_solution_l2566_256633


namespace bill_denomination_l2566_256630

theorem bill_denomination (num_bills : ℕ) (total_value : ℕ) (denomination : ℕ) :
  num_bills = 10 →
  total_value = 50 →
  num_bills * denomination = total_value →
  denomination = 5 := by
  sorry

end bill_denomination_l2566_256630
