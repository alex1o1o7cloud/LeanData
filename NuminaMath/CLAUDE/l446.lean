import Mathlib

namespace walts_interest_l446_44646

/-- Calculates the total interest earned from two investments with different rates -/
def total_interest (total_amount : ℝ) (amount_at_lower_rate : ℝ) (lower_rate : ℝ) (higher_rate : ℝ) : ℝ :=
  (amount_at_lower_rate * lower_rate) + ((total_amount - amount_at_lower_rate) * higher_rate)

/-- Proves that Walt's total interest is $770 given the problem conditions -/
theorem walts_interest :
  let total_amount : ℝ := 9000
  let amount_at_lower_rate : ℝ := 4000
  let lower_rate : ℝ := 0.08
  let higher_rate : ℝ := 0.09
  total_interest total_amount amount_at_lower_rate lower_rate higher_rate = 770 := by
  sorry

#eval total_interest 9000 4000 0.08 0.09

end walts_interest_l446_44646


namespace lynne_total_spent_l446_44631

/-- The total amount Lynne spent on books and magazines -/
def total_spent (cat_books : ℕ) (solar_books : ℕ) (magazines : ℕ) (book_cost : ℕ) (magazine_cost : ℕ) : ℕ :=
  (cat_books + solar_books) * book_cost + magazines * magazine_cost

/-- Theorem stating that Lynne spent $75 in total -/
theorem lynne_total_spent :
  total_spent 7 2 3 7 4 = 75 := by
  sorry

#eval total_spent 7 2 3 7 4

end lynne_total_spent_l446_44631


namespace painted_cubes_l446_44667

theorem painted_cubes (n : ℕ) (h : n = 4) :
  let total_cubes := n^3
  let unpainted_cubes := (n - 2)^3
  let painted_cubes := total_cubes - unpainted_cubes
  painted_cubes = 42 := by
  sorry

end painted_cubes_l446_44667


namespace bird_cage_problem_l446_44650

theorem bird_cage_problem (B : ℚ) : 
  (B > 0) →                         -- Ensure positive number of birds
  (B * (2/3) * (3/5) * (1/3) = 60)  -- Remaining birds after three stages equal 60
  ↔ 
  (B = 450) :=                      -- Total initial number of birds is 450
by sorry

end bird_cage_problem_l446_44650


namespace geometric_sequence_ratio_l446_44673

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_geom : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q) 
  (h_arith : 2 * a 1 - a 2 = a 2 - a 3 / 2) :
  (a 2017 + a 2016) / (a 2015 + a 2014) = 4 := by
  sorry

end geometric_sequence_ratio_l446_44673


namespace plywood_cut_perimeter_difference_l446_44619

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

theorem plywood_cut_perimeter_difference :
  let original := Rectangle.mk 9 6
  let pieces : List Rectangle := [
    Rectangle.mk 9 2,  -- Configuration 1
    Rectangle.mk 6 3   -- Configuration 2
  ]
  let perimeters := pieces.map perimeter
  let max_perimeter := perimeters.maximum?
  let min_perimeter := perimeters.minimum?
  ∀ (max min : ℝ), max_perimeter = some max → min_perimeter = some min →
    max - min = 6 :=
by sorry

end plywood_cut_perimeter_difference_l446_44619


namespace max_value_expression_l446_44671

theorem max_value_expression (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_squares : x^2 + y^2 + z^2 = 1) :
  3 * x * y * Real.sqrt 5 + 9 * y * z ≤ (3/2) * Real.sqrt 409 := by
sorry

end max_value_expression_l446_44671


namespace decimal_2009_to_octal_l446_44612

def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

theorem decimal_2009_to_octal :
  decimal_to_octal 2009 = [3, 7, 3, 1] :=
by sorry

end decimal_2009_to_octal_l446_44612


namespace initial_amount_of_liquid_A_solution_is_correct_l446_44614

/-- Given a mixture of liquids A and B, this theorem proves the initial amount of liquid A. -/
theorem initial_amount_of_liquid_A
  (initial_ratio : ℚ) -- Initial ratio of A to B
  (replacement_volume : ℚ) -- Volume of mixture replaced with B
  (final_ratio : ℚ) -- Final ratio of A to B
  (h1 : initial_ratio = 4 / 1)
  (h2 : replacement_volume = 40)
  (h3 : final_ratio = 2 / 3)
  : ℚ :=
by
  sorry

#check initial_amount_of_liquid_A

/-- The solution to the problem -/
def solution : ℚ := 32

/-- Proof that the solution is correct -/
theorem solution_is_correct :
  initial_amount_of_liquid_A (4 / 1) 40 (2 / 3) rfl rfl rfl = solution :=
by
  sorry

end initial_amount_of_liquid_A_solution_is_correct_l446_44614


namespace fourth_power_subset_exists_l446_44623

/-- The set of prime numbers less than or equal to 26 -/
def primes_le_26 : Finset ℕ := sorry

/-- A function that represents a number as a tuple of exponents of primes <= 26 -/
def exponent_tuple (n : ℕ) : Fin 9 → ℕ := sorry

/-- The set M of 1985 different positive integers with prime factors <= 26 -/
def M : Finset ℕ := sorry

/-- The cardinality of M is 1985 -/
axiom M_card : Finset.card M = 1985

/-- All elements in M have prime factors <= 26 -/
axiom M_primes (n : ℕ) : n ∈ M → ∀ p : ℕ, p.Prime → p ∣ n → p ≤ 26

/-- All elements in M are different -/
axiom M_distinct : ∀ a b : ℕ, a ∈ M → b ∈ M → a ≠ b

/-- Main theorem: There exists a subset of 4 elements from M whose product is a fourth power -/
theorem fourth_power_subset_exists : 
  ∃ (a b c d : ℕ) (k : ℕ), a ∈ M ∧ b ∈ M ∧ c ∈ M ∧ d ∈ M ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a * b * c * d = k^4 := by sorry

end fourth_power_subset_exists_l446_44623


namespace james_car_value_l446_44672

/-- The value of James' old car -/
def old_car_value : ℝ := 20000

/-- The percentage of the old car's value James received when selling it -/
def old_car_sell_percentage : ℝ := 0.8

/-- The sticker price of the new car -/
def new_car_sticker_price : ℝ := 30000

/-- The percentage of the new car's sticker price James paid after haggling -/
def new_car_buy_percentage : ℝ := 0.9

/-- The out-of-pocket amount James paid -/
def out_of_pocket : ℝ := 11000

theorem james_car_value :
  new_car_buy_percentage * new_car_sticker_price - old_car_sell_percentage * old_car_value = out_of_pocket :=
by sorry

end james_car_value_l446_44672


namespace meetings_percentage_of_workday_l446_44676

/-- Represents the duration of a work day in minutes -/
def work_day_minutes : ℕ := 10 * 60

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_minutes : ℕ := 35

/-- Calculates the total duration of all meetings in minutes -/
def total_meeting_minutes : ℕ := 
  first_meeting_minutes + 2 * first_meeting_minutes + (first_meeting_minutes + 2 * first_meeting_minutes)

/-- Theorem stating that the percentage of work day spent in meetings is 35% -/
theorem meetings_percentage_of_workday : 
  (total_meeting_minutes : ℚ) / (work_day_minutes : ℚ) * 100 = 35 := by
  sorry

end meetings_percentage_of_workday_l446_44676


namespace inequality_solution_set_l446_44653

theorem inequality_solution_set (x : ℝ) : x + 8 < 4*x - 1 ↔ x > 3 := by
  sorry

end inequality_solution_set_l446_44653


namespace sum_of_absolute_roots_l446_44600

/-- Given a cubic polynomial x^3 - 2023x + m with three integer roots,
    prove that the sum of the absolute values of the roots is 80. -/
theorem sum_of_absolute_roots (m : ℤ) (a b c : ℤ) : 
  (∀ x : ℤ, x^3 - 2023*x + m = 0 ↔ x = a ∨ x = b ∨ x = c) →
  |a| + |b| + |c| = 80 := by
  sorry

end sum_of_absolute_roots_l446_44600


namespace no_zeros_in_2_16_l446_44686

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of f having a unique zero point in (0, 2)
def has_unique_zero_in_0_2 (f : ℝ → ℝ) : Prop :=
  ∃! x, x ∈ (Set.Ioo 0 2) ∧ f x = 0

-- Theorem statement
theorem no_zeros_in_2_16 (h : has_unique_zero_in_0_2 f) :
  ∀ x ∈ Set.Ico 2 16, f x ≠ 0 := by
  sorry

end no_zeros_in_2_16_l446_44686


namespace average_of_numbers_l446_44621

def numbers : List ℝ := [10, 4, 8, 7, 6]

theorem average_of_numbers : (numbers.sum / numbers.length : ℝ) = 7 := by
  sorry

end average_of_numbers_l446_44621


namespace f_is_even_l446_44654

-- Define g as an even function
def g_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

-- Define f in terms of g
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := |g (x^4)|

-- Theorem statement
theorem f_is_even (g : ℝ → ℝ) (h : g_even g) : ∀ x, f g x = f g (-x) := by
  sorry

end f_is_even_l446_44654


namespace job_completion_time_l446_44696

/-- Given two workers A and B who can complete a job in 15 and 30 days respectively,
    prove that they worked together for 4 days if 0.6 of the job is left unfinished. -/
theorem job_completion_time 
  (rate_A : ℝ) (rate_B : ℝ) (days_worked : ℝ) (fraction_left : ℝ) :
  rate_A = 1 / 15 →
  rate_B = 1 / 30 →
  fraction_left = 0.6 →
  (rate_A + rate_B) * days_worked = 1 - fraction_left →
  days_worked = 4 := by
sorry

end job_completion_time_l446_44696


namespace road_trip_distances_l446_44661

theorem road_trip_distances (total_distance : ℕ) 
  (tracy_distance michelle_distance katie_distance : ℕ) : 
  total_distance = 1000 →
  tracy_distance = 2 * michelle_distance + 20 →
  michelle_distance = 3 * katie_distance →
  tracy_distance + michelle_distance + katie_distance = total_distance →
  michelle_distance = 294 := by
  sorry

end road_trip_distances_l446_44661


namespace circle_through_intersections_and_tangent_to_line_l446_44635

/-- Given two circles and a line, prove that a specific circle passes through 
    the intersection points of the given circles and is tangent to the given line. -/
theorem circle_through_intersections_and_tangent_to_line :
  let C₁ : ℝ × ℝ → Prop := λ (x, y) ↦ x^2 + y^2 = 4
  let C₂ : ℝ × ℝ → Prop := λ (x, y) ↦ x^2 + y^2 - 2*x - 4*y + 4 = 0
  let l : ℝ × ℝ → Prop := λ (x, y) ↦ x + 2*y = 0
  let result_circle : ℝ × ℝ → Prop := λ (x, y) ↦ (x - 1/2)^2 + (y - 1)^2 = 5/4
  
  (∀ p, C₁ p ∧ C₂ p → result_circle p) ∧ 
  (∃ unique_p, l unique_p ∧ result_circle unique_p ∧ 
    ∀ q, l q ∧ result_circle q → q = unique_p) :=
by
  sorry

end circle_through_intersections_and_tangent_to_line_l446_44635


namespace fourth_term_of_specific_gp_l446_44668

def geometric_progression (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem fourth_term_of_specific_gp :
  let a := Real.sqrt 2
  let r := (Real.sqrt 2) ^ (1/4)
  let third_term := geometric_progression a r 3
  third_term = 2 ^ (1/8) →
  geometric_progression a r 4 = 1 / (Real.sqrt 2) ^ (1/4) := by
sorry

end fourth_term_of_specific_gp_l446_44668


namespace final_cell_count_l446_44615

def initial_cells : ℕ := 5
def split_ratio : ℕ := 3
def split_interval : ℕ := 3
def total_days : ℕ := 12

def geometric_sequence (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * r^(n - 1)

theorem final_cell_count :
  geometric_sequence initial_cells split_ratio (total_days / split_interval) = 135 := by
  sorry

end final_cell_count_l446_44615


namespace circle_area_difference_l446_44688

theorem circle_area_difference : 
  let r1 : ℝ := 30
  let d2 : ℝ := 15
  let r2 : ℝ := d2 / 2
  let area1 : ℝ := π * r1^2
  let area2 : ℝ := π * r2^2
  area1 - area2 = 843.75 * π := by sorry

end circle_area_difference_l446_44688


namespace proportion_equality_l446_44628

theorem proportion_equality (x : ℝ) : (0.75 / x = 3 / 8) → x = 2 := by
  sorry

end proportion_equality_l446_44628


namespace circle_equation_from_diameter_line_l446_44647

/-- Given a line 3x - 4y + 12 = 0 intersecting the x-axis and y-axis, 
    the circle with the line segment between these intersections as its diameter 
    has the equation x^2 + 4x + y^2 - 3y = 0 -/
theorem circle_equation_from_diameter_line (x y : ℝ) : 
  (∃ (t : ℝ), 3*t - 4*0 + 12 = 0 ∧ 3*0 - 4*t + 12 = 0) →  -- Line intersects both axes
  (x^2 + 4*x + y^2 - 3*y = 0 ↔ 
   ∃ (p : ℝ × ℝ), (3*(p.1) - 4*(p.2) + 12 = 0) ∧ 
                  ((x - p.1)^2 + (y - p.2)^2 = 
                   ((3*0 - 4*0 + 12)/3 - (3*0 - 4*0 + 12)/(-4))^2/4 + 
                   ((3*0 - 4*0 + 12)/(-4))^2/4)) :=
by sorry

end circle_equation_from_diameter_line_l446_44647


namespace max_store_visits_l446_44687

theorem max_store_visits (total_stores : ℕ) (total_visits : ℕ) (total_shoppers : ℕ) 
  (double_visitors : ℕ) (h1 : total_stores = 8) (h2 : total_visits = 23) 
  (h3 : total_shoppers = 12) (h4 : double_visitors = 8) 
  (h5 : double_visitors ≤ total_shoppers) 
  (h6 : double_visitors * 2 + (total_shoppers - double_visitors) ≤ total_visits) :
  ∃ (max_visits : ℕ), max_visits ≤ 4 ∧ 
  ∀ (individual_visits : ℕ), individual_visits ≤ max_visits :=
by sorry

end max_store_visits_l446_44687


namespace repair_cost_percentage_l446_44658

def apple_price : ℚ := 5/4
def bike_cost : ℚ := 80
def apples_sold : ℕ := 20
def remaining_fraction : ℚ := 1/5

theorem repair_cost_percentage :
  let total_earned : ℚ := apple_price * apples_sold
  let repair_cost : ℚ := total_earned * (1 - remaining_fraction)
  repair_cost / bike_cost = 1/4
:= by sorry

end repair_cost_percentage_l446_44658


namespace steve_bike_time_l446_44629

/-- Given that Steve biked 5 miles in the same time Jordan biked 3 miles,
    and Jordan took 18 minutes to bike 3 miles,
    prove that Steve will take 126/5 minutes to bike 7 miles. -/
theorem steve_bike_time (steve_distance : ℝ) (jordan_distance : ℝ) (jordan_time : ℝ) (steve_new_distance : ℝ) :
  steve_distance = 5 →
  jordan_distance = 3 →
  jordan_time = 18 →
  steve_new_distance = 7 →
  (steve_new_distance / (steve_distance / jordan_time)) = 126 / 5 := by
  sorry

end steve_bike_time_l446_44629


namespace ella_jasper_passing_count_l446_44618

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ  -- speed in m/min
  radius : ℝ  -- radius of the track in meters
  direction : ℝ  -- 1 for clockwise, -1 for counterclockwise

/-- Calculates the number of times two runners pass each other on a circular track -/
def passingCount (runner1 runner2 : Runner) (duration : ℝ) : ℕ :=
  sorry

/-- Theorem: Ella and Jasper pass each other 93 times during their 40-minute jog -/
theorem ella_jasper_passing_count : 
  let ella : Runner := { speed := 300, radius := 40, direction := 1 }
  let jasper : Runner := { speed := 360, radius := 50, direction := -1 }
  passingCount ella jasper 40 = 93 := by
  sorry

end ella_jasper_passing_count_l446_44618


namespace role_assignment_count_l446_44601

def num_men : ℕ := 6
def num_women : ℕ := 7
def num_male_roles : ℕ := 3
def num_female_roles : ℕ := 3
def num_neutral_roles : ℕ := 2

def total_roles : ℕ := num_male_roles + num_female_roles + num_neutral_roles

theorem role_assignment_count : 
  (num_men.factorial / (num_men - num_male_roles).factorial) *
  (num_women.factorial / (num_women - num_female_roles).factorial) *
  ((num_men + num_women - num_male_roles - num_female_roles).factorial / 
   (num_men + num_women - total_roles).factorial) = 1058400 := by
  sorry

end role_assignment_count_l446_44601


namespace company_problem_solution_l446_44674

def company_problem (total_employees : ℕ) 
                    (clerical_fraction technical_fraction managerial_fraction : ℚ)
                    (clerical_reduction technical_reduction managerial_reduction : ℚ) : ℚ :=
  let initial_clerical := (clerical_fraction * total_employees : ℚ)
  let initial_technical := (technical_fraction * total_employees : ℚ)
  let initial_managerial := (managerial_fraction * total_employees : ℚ)
  
  let remaining_clerical := initial_clerical * (1 - clerical_reduction)
  let remaining_technical := initial_technical * (1 - technical_reduction)
  let remaining_managerial := initial_managerial * (1 - managerial_reduction)
  
  let total_remaining := remaining_clerical + remaining_technical + remaining_managerial
  
  remaining_clerical / total_remaining

theorem company_problem_solution :
  let result := company_problem 5000 (1/5) (2/5) (2/5) (1/3) (1/4) (1/5)
  ∃ (ε : ℚ), abs (result - 177/1000) < ε ∧ ε < 1/1000 :=
by
  sorry

end company_problem_solution_l446_44674


namespace curve_self_intersection_l446_44643

theorem curve_self_intersection :
  ∃ (a b : ℝ), a ≠ b ∧
    a^2 - 4 = b^2 - 4 ∧
    a^3 - 6*a + 4 = b^3 - 6*b + 4 ∧
    (a^2 - 4 = 2 ∧ a^3 - 6*a + 4 = 4) :=
sorry

end curve_self_intersection_l446_44643


namespace incorrect_statement_is_false_l446_44603

/-- Represents the method used for separation and counting of bacteria. -/
inductive SeparationMethod
| DilutionPlating
| StreakPlate

/-- Represents a biotechnology practice. -/
structure BiotechPractice where
  soil_bacteria_method : SeparationMethod
  fruit_vinegar_air : Bool
  nitrite_detection : Bool
  dna_extraction : Bool

/-- The correct biotechnology practices. -/
def correct_practices : BiotechPractice := {
  soil_bacteria_method := SeparationMethod.DilutionPlating,
  fruit_vinegar_air := true,
  nitrite_detection := true,
  dna_extraction := true
}

/-- The statement to be proven false. -/
def incorrect_statement : BiotechPractice := {
  soil_bacteria_method := SeparationMethod.StreakPlate,
  fruit_vinegar_air := true,
  nitrite_detection := true,
  dna_extraction := true
}

/-- Theorem stating that the incorrect statement is indeed incorrect. -/
theorem incorrect_statement_is_false : incorrect_statement ≠ correct_practices := by
  sorry

end incorrect_statement_is_false_l446_44603


namespace odot_two_four_l446_44699

def odot (a b : ℝ) : ℝ := 5 * a + 2 * b

theorem odot_two_four : odot 2 4 = 18 := by
  sorry

end odot_two_four_l446_44699


namespace exists_question_with_different_answers_l446_44684

/-- A type representing questions that can be asked -/
inductive Question
| NumberOfQuestions : Question
| CurrentTime : Question

/-- A type representing the state of the world at a given moment -/
structure WorldState where
  questionsAsked : Nat
  currentTime : Nat

/-- A function that gives the truthful answer to a question given the world state -/
def truthfulAnswer (q : Question) (w : WorldState) : Nat :=
  match q with
  | Question.NumberOfQuestions => w.questionsAsked
  | Question.CurrentTime => w.currentTime

/-- Theorem stating that there exists a question that can have different truthful answers at different times -/
theorem exists_question_with_different_answers :
  ∃ (q : Question) (w1 w2 : WorldState), w1 ≠ w2 → truthfulAnswer q w1 ≠ truthfulAnswer q w2 := by
  sorry


end exists_question_with_different_answers_l446_44684


namespace prime_equation_solutions_l446_44659

theorem prime_equation_solutions :
  ∃ (S : Finset ℕ), (∀ p ∈ S, Nat.Prime p ∧ p^3 + p^2 - 18*p + 26 = 0) ∧ S.card = 2 :=
by sorry

end prime_equation_solutions_l446_44659


namespace union_of_A_and_B_l446_44606

def A : Set ℝ := {x | x^2 < 4}
def B : Set ℝ := {y | ∃ x ∈ A, y = x^2 - 2*x - 1}

theorem union_of_A_and_B : A ∪ B = {x | -2 ≤ x ∧ x < 7} := by sorry

end union_of_A_and_B_l446_44606


namespace max_value_of_s_l446_44694

theorem max_value_of_s (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let s := min x (min (y + 1/x) (1/y))
  s ≤ Real.sqrt 2 ∧ 
  (s = Real.sqrt 2 ↔ x = Real.sqrt 2 ∧ y = 1 / Real.sqrt 2) :=
by sorry

end max_value_of_s_l446_44694


namespace largest_number_l446_44689

theorem largest_number (a b c d : ℝ) 
  (sum1 : a + b + c = 180)
  (sum2 : a + b + d = 197)
  (sum3 : a + c + d = 208)
  (sum4 : b + c + d = 222) :
  max a (max b (max c d)) = 89 := by
sorry

end largest_number_l446_44689


namespace equation_solutions_l446_44640

theorem equation_solutions :
  (∀ x : ℚ, 2 * x * (x + 1) = x + 1 ↔ x = -1 ∨ x = 1/2) ∧
  (∀ x : ℚ, 2 * x^2 + 3 * x - 5 = 0 ↔ x = -5/2 ∨ x = 1) := by
  sorry

end equation_solutions_l446_44640


namespace evaluate_expression_l446_44691

theorem evaluate_expression (x : ℝ) (y : ℝ) (h1 : x = 5) (h2 : y = 2 * x) : 
  y * (y - 2 * x) = 0 := by
  sorry

end evaluate_expression_l446_44691


namespace middle_card_number_l446_44678

theorem middle_card_number (a b c : ℕ) : 
  a < b → b < c → 
  a + b + c = 15 → 
  a + b < 10 → 
  (∀ x y z : ℕ, x < y → y < z → x + y + z = 15 → x + y < 10 → (x = a ∧ z = c) → y ≠ b) →
  b = 5 := by
  sorry

end middle_card_number_l446_44678


namespace abs_neg_two_and_half_l446_44692

theorem abs_neg_two_and_half : |(-5/2 : ℚ)| = 5/2 := by
  sorry

end abs_neg_two_and_half_l446_44692


namespace stating_tournament_winners_l446_44679

/-- Represents the number of participants in a tournament round -/
def participants : ℕ := 512

/-- Represents the number of wins we're interested in -/
def target_wins : ℕ := 6

/-- 
Represents the number of participants who finish with exactly k wins 
in a single-elimination tournament with n rounds
-/
def participants_with_wins (n k : ℕ) : ℕ := Nat.choose n k

/-- 
Theorem stating that in a single-elimination tournament with 512 participants,
exactly 84 participants will finish with 6 wins
-/
theorem tournament_winners : 
  participants_with_wins (Nat.log 2 participants) target_wins = 84 := by
  sorry

end stating_tournament_winners_l446_44679


namespace base_difference_theorem_l446_44670

/-- Convert a number from base 16 to base 10 -/
def base16_to_base10 (n : String) : ℕ :=
  match n with
  | "1A3" => 419
  | _ => 0

/-- Convert a number from base 7 to base 10 -/
def base7_to_base10 (n : String) : ℕ :=
  match n with
  | "142" => 79
  | _ => 0

/-- The main theorem stating that the difference between 1A3 (base 16) and 142 (base 7) in base 10 is 340 -/
theorem base_difference_theorem :
  base16_to_base10 "1A3" - base7_to_base10 "142" = 340 := by
  sorry

end base_difference_theorem_l446_44670


namespace probability_between_R_and_S_l446_44636

/-- Given points P, Q, R, and S on a line segment PQ, where PQ = 4PR and PQ = 8RS,
    the probability that a randomly selected point on PQ is between R and S is 5/8 -/
theorem probability_between_R_and_S (P Q R S : ℝ) (h1 : P < R) (h2 : R < S) (h3 : S < Q)
    (h4 : Q - P = 4 * (R - P)) (h5 : Q - P = 8 * (S - R)) :
    (S - R) / (Q - P) = 5 / 8 := by sorry

end probability_between_R_and_S_l446_44636


namespace triangle_1234_l446_44669

/-- Define the operation △ -/
def triangle (n m : ℕ) : ℕ := sorry

/-- Axiom for the first condition -/
axiom triangle_1 {a b c d : ℕ} (h1 : 0 < a ∧ a < 10) (h2 : 0 < b ∧ b < 10) 
  (h3 : 0 < c ∧ c < 10) (h4 : 0 < d ∧ d < 10) : 
  triangle (a * 1000 + b * 100 + c * 10 + d) 1 = b * 1000 + c * 100 + a * 10 + d

/-- Axiom for the second condition -/
axiom triangle_2 {a b c d : ℕ} (h1 : 0 < a ∧ a < 10) (h2 : 0 < b ∧ b < 10) 
  (h3 : 0 < c ∧ c < 10) (h4 : 0 < d ∧ d < 10) : 
  triangle (a * 1000 + b * 100 + c * 10 + d) 2 = c * 1000 + d * 100 + a * 10 + b

/-- The main theorem to prove -/
theorem triangle_1234 : triangle (triangle 1234 1) 2 = 3412 := by sorry

end triangle_1234_l446_44669


namespace max_m_value_l446_44622

theorem max_m_value (x y m : ℝ) : 
  (4 * x + 3 * y = 4 * m + 5) →
  (3 * x - y = m - 1) →
  (x + 4 * y ≤ 3) →
  (∀ m' : ℝ, m' > m → ¬(∃ x' y' : ℝ, 
    (4 * x' + 3 * y' = 4 * m' + 5) ∧
    (3 * x' - y' = m' - 1) ∧
    (x' + 4 * y' ≤ 3))) →
  m = -1 := by
sorry

end max_m_value_l446_44622


namespace remaining_flowers_l446_44660

/-- Represents the flower arrangement along the path --/
structure FlowerPath :=
  (peonies : Nat)
  (tulips : Nat)
  (watered : Nat)
  (unwatered : Nat)
  (picked_tulips : Nat)

/-- Theorem stating the number of remaining flowers after Neznayka's picking --/
theorem remaining_flowers (path : FlowerPath) 
  (h1 : path.peonies = 15)
  (h2 : path.tulips = 15)
  (h3 : path.unwatered = 10)
  (h4 : path.watered + path.unwatered = path.peonies + path.tulips)
  (h5 : path.picked_tulips = 6) :
  path.watered - path.picked_tulips = 19 := by
  sorry

#check remaining_flowers

end remaining_flowers_l446_44660


namespace sample_capacity_l446_44685

theorem sample_capacity (f : ℕ) (fr : ℚ) (h1 : f = 36) (h2 : fr = 1/4) :
  ∃ n : ℕ, f / n = fr ∧ n = 144 := by
  sorry

end sample_capacity_l446_44685


namespace students_count_l446_44605

/-- The number of seats on each school bus -/
def seats_per_bus : ℕ := 2

/-- The number of buses needed for the trip -/
def buses_needed : ℕ := 7

/-- The number of students going on the field trip -/
def students_on_trip : ℕ := seats_per_bus * buses_needed

theorem students_count : students_on_trip = 14 := by
  sorry

end students_count_l446_44605


namespace range_of_a_l446_44641

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 3, a ≥ -x^2 + 2*x - 2/3) ∧ 
  (∃ x : ℝ, x^2 + 4*x + a = 0) ↔ 
  a ∈ Set.Icc (1/3) 4 := by
  sorry

end range_of_a_l446_44641


namespace even_function_implies_a_plus_b_eq_four_l446_44681

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + (b - 3) * x + 3

-- Define the property of being an even function on an interval
def is_even_on (f : ℝ → ℝ) (l r : ℝ) : Prop :=
  ∀ x, l ≤ x ∧ x ≤ r → f x = f (-x)

-- Theorem statement
theorem even_function_implies_a_plus_b_eq_four (a b : ℝ) :
  is_even_on (f a b) (a^2 - 2) a →
  a^2 - 2 ≤ a →
  a + b = 4 := by
  sorry


end even_function_implies_a_plus_b_eq_four_l446_44681


namespace power_of_product_l446_44637

theorem power_of_product (a : ℝ) : (2 * a^2)^3 = 8 * a^6 := by
  sorry

end power_of_product_l446_44637


namespace pencil_pen_cost_l446_44642

theorem pencil_pen_cost (p q : ℝ) 
  (h1 : 4 * p + 3 * q = 4.20)
  (h2 : 3 * p + 4 * q = 4.55) :
  p + q = 1.25 := by
sorry

end pencil_pen_cost_l446_44642


namespace count_common_divisors_36_90_l446_44645

def divisors_of_both (a b : ℕ) : Finset ℕ :=
  (Finset.range a).filter (fun x => x > 0 ∧ a % x = 0 ∧ b % x = 0)

theorem count_common_divisors_36_90 : (divisors_of_both 36 90).card = 6 := by
  sorry

end count_common_divisors_36_90_l446_44645


namespace unique_solution_iff_k_zero_l446_44664

/-- 
Theorem: The pair of equations y = x^2 and y = 2x^2 + k have exactly one solution 
if and only if k = 0.
-/
theorem unique_solution_iff_k_zero (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = p.1^2 ∧ p.2 = 2*p.1^2 + k) ↔ k = 0 :=
by sorry

end unique_solution_iff_k_zero_l446_44664


namespace fraction_of_fraction_of_fraction_one_fifth_of_one_third_of_one_sixth_of_ninety_l446_44677

theorem fraction_of_fraction_of_fraction (a b c d : ℚ) :
  a * (b * (c * d)) = (a * b * c) * d :=
by sorry

theorem one_fifth_of_one_third_of_one_sixth_of_ninety :
  (1 / 5 : ℚ) * ((1 / 3 : ℚ) * ((1 / 6 : ℚ) * 90)) = 1 :=
by sorry

end fraction_of_fraction_of_fraction_one_fifth_of_one_third_of_one_sixth_of_ninety_l446_44677


namespace cyclic_sum_inequality_l446_44611

theorem cyclic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z ≥ 1) :
  (x^5 - x^2) / (x^5 + y^2 + z^2) + (y^5 - y^2) / (y^5 + z^2 + x^2) + (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 := by
  sorry

end cyclic_sum_inequality_l446_44611


namespace problem_statement_l446_44697

/-- Given real numbers a, b, and c satisfying certain conditions, 
    prove that c^2 * (a + b) = 2008 -/
theorem problem_statement (a b c : ℝ) 
    (h1 : a^2 * (b + c) = 2008)
    (h2 : b^2 * (a + c) = 2008)
    (h3 : a ≠ b) :
  c^2 * (a + b) = 2008 := by
  sorry

end problem_statement_l446_44697


namespace school_commute_theorem_l446_44639

/-- Represents the number of students in different commute categories -/
structure SchoolCommute where
  localStudents : ℕ
  publicTransport : ℕ
  privateTransport : ℕ
  train : ℕ
  bus : ℕ
  cycle : ℕ
  drivenByParents : ℕ

/-- Given the commute ratios and public transport users, proves the number of train commuters
    minus parent-driven students and the total number of students -/
theorem school_commute_theorem (sc : SchoolCommute) 
  (h1 : sc.localStudents = 3 * (sc.publicTransport + sc.privateTransport))
  (h2 : 3 * sc.privateTransport = 2 * sc.publicTransport)
  (h3 : 7 * sc.bus = 5 * sc.train)
  (h4 : 5 * sc.drivenByParents = 3 * sc.cycle)
  (h5 : sc.publicTransport = 24)
  (h6 : sc.publicTransport = sc.train + sc.bus)
  (h7 : sc.privateTransport = sc.cycle + sc.drivenByParents) :
  sc.train - sc.drivenByParents = 8 ∧ 
  sc.localStudents + sc.publicTransport + sc.privateTransport = 160 := by
  sorry


end school_commute_theorem_l446_44639


namespace remainder_sum_l446_44624

theorem remainder_sum (a b : ℤ) (h1 : a % 60 = 49) (h2 : b % 40 = 29) : (a + b) % 20 = 18 := by
  sorry

end remainder_sum_l446_44624


namespace pure_imaginary_roots_of_f_l446_44617

/-- The polynomial function we're analyzing -/
def f (x : ℂ) : ℂ := x^5 - 2*x^4 + 4*x^3 - 8*x^2 + 16*x - 32

/-- A complex number is pure imaginary if its real part is zero -/
def isPureImaginary (z : ℂ) : Prop := z.re = 0

theorem pure_imaginary_roots_of_f :
  ∃! (z : ℂ), f z = 0 ∧ isPureImaginary z ∧ z = 0 := by
  sorry

end pure_imaginary_roots_of_f_l446_44617


namespace hyperbola_ellipse_range_l446_44680

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (1 - m) + y^2 / (m + 2) = 1
def q (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (2 * m) + y^2 / (2 - m) = 1

-- Define the condition that q represents an ellipse with foci on the x-axis
def q_ellipse (m : ℝ) : Prop := 2 * m > 0 ∧ 2 - m > 0

-- Define the theorem
theorem hyperbola_ellipse_range (m : ℝ) : 
  (¬(p m ∧ q m) ∧ q_ellipse m) ↔ (m ≤ 1 ∨ m ≥ 2) :=
sorry

end hyperbola_ellipse_range_l446_44680


namespace special_function_sqrt_5753_l446_44607

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x * y) = x * f y + y * f x) ∧
  (∀ x y : ℝ, f (x + y) = f (x * 1993) + f (y * 1993))

/-- The main theorem -/
theorem special_function_sqrt_5753 (f : ℝ → ℝ) (h : special_function f) :
  f (Real.sqrt 5753) = 0 := by sorry

end special_function_sqrt_5753_l446_44607


namespace factor_congruence_l446_44620

theorem factor_congruence (n : ℕ+) (k : ℕ) :
  k ∣ (2 * n.val)^(2^n.val) + 1 → k ≡ 1 [MOD 2^(n.val + 1)] := by
  sorry

end factor_congruence_l446_44620


namespace largest_triangle_area_l446_44610

/-- The largest area of a triangle ABC, where A = (2,1), B = (5,3), and C = (p,q) 
    lie on the parabola y = -x^2 + 7x - 10, with 2 ≤ p ≤ 5 -/
theorem largest_triangle_area : 
  let A : ℝ × ℝ := (2, 1)
  let B : ℝ × ℝ := (5, 3)
  let C : ℝ → ℝ × ℝ := λ p => (p, -p^2 + 7*p - 10)
  let triangle_area : ℝ → ℝ := λ p => 
    (1/2) * abs (A.1 * B.2 + B.1 * (C p).2 + (C p).1 * A.2 - 
                 A.2 * B.1 - B.2 * (C p).1 - (C p).2 * A.1)
  ∃ (max_area : ℝ), max_area = 13/8 ∧ 
    ∀ p : ℝ, 2 ≤ p ∧ p ≤ 5 → triangle_area p ≤ max_area :=
by sorry

end largest_triangle_area_l446_44610


namespace quadratic_roots_always_positive_implies_a_zero_l446_44682

theorem quadratic_roots_always_positive_implies_a_zero 
  (a b c : ℝ) 
  (h : ∀ (p : ℝ), p > 0 → ∀ (x : ℝ), a * x^2 + b * x + c + p = 0 → x > 0) : 
  a = 0 := by
sorry

end quadratic_roots_always_positive_implies_a_zero_l446_44682


namespace fifth_term_is_32_l446_44648

/-- A geometric sequence where all terms are positive and satisfy a_n * a_(n+1) = 2^(2n+1) -/
def special_geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (∀ n, a n * a (n + 1) = 2^(2*n + 1)) ∧
  (∀ n, ∃ q > 0, a (n + 1) = q * a n)

/-- The 5th term of the special geometric sequence is 32 -/
theorem fifth_term_is_32 (a : ℕ → ℝ) (h : special_geometric_sequence a) : 
  a 5 = 32 := by
sorry

end fifth_term_is_32_l446_44648


namespace two_digit_reverse_sqrt_l446_44698

theorem two_digit_reverse_sqrt (n x y : ℕ) : 
  (x > y) →
  (2 * n = x + y) →
  (10 ≤ n ∧ n < 100) →
  (∃ (a b : ℕ), n = 10 * a + b ∧ a < 10 ∧ b < 10) →
  (∃ (k : ℕ), k * k = x * y) →
  (∃ (a b : ℕ), n = 10 * a + b ∧ k = 10 * b + a) →
  (x - y = 66) :=
by sorry

end two_digit_reverse_sqrt_l446_44698


namespace passing_percentage_is_45_percent_l446_44613

/-- Represents an examination with a passing percentage. -/
structure Examination where
  max_marks : ℕ
  passing_percentage : ℚ

/-- Calculates the passing marks for an examination. -/
def passing_marks (exam : Examination) : ℚ :=
  (exam.passing_percentage / 100) * exam.max_marks

/-- Theorem: The passing percentage is 45% given the conditions. -/
theorem passing_percentage_is_45_percent 
  (max_marks : ℕ) 
  (failing_score : ℕ) 
  (deficit : ℕ) 
  (h1 : max_marks = 500)
  (h2 : failing_score = 180)
  (h3 : deficit = 45)
  : ∃ (exam : Examination), 
    exam.max_marks = max_marks ∧ 
    exam.passing_percentage = 45 ∧
    passing_marks exam = failing_score + deficit :=
  sorry


end passing_percentage_is_45_percent_l446_44613


namespace range_of_a_l446_44632

-- Define the propositions p and q
def p (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 1 2, (1/2) * x^2 - Real.log (x - a) ≥ 0

def q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*a*x - 8 - 6*a = 0

-- Define the range of a
def range_a : Set ℝ := Set.Ici (-4) ∪ Set.Icc (-2) (1/2)

-- Theorem statement
theorem range_of_a (a : ℝ) : p a ∧ q a → a ∈ range_a := by
  sorry


end range_of_a_l446_44632


namespace line_through_point_with_equal_intercepts_l446_44625

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

def has_equal_intercepts (l : Line2D) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c / l.a = l.c / l.b

def point_on_line (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem line_through_point_with_equal_intercepts :
  ∃ (l1 l2 : Line2D),
    (point_on_line ⟨1, 2⟩ l1 ∧ has_equal_intercepts l1) ∧
    (point_on_line ⟨1, 2⟩ l2 ∧ has_equal_intercepts l2) ∧
    ((l1.a = 1 ∧ l1.b = 1 ∧ l1.c = -3) ∨ (l2.a = 2 ∧ l2.b = -1 ∧ l2.c = 0)) :=
by sorry

end line_through_point_with_equal_intercepts_l446_44625


namespace consecutive_integers_sum_of_squares_l446_44675

theorem consecutive_integers_sum_of_squares (b : ℤ) : 
  (b - 1) * b * (b + 1) = 12 * (3 * b) + b^2 → 
  (b - 1)^2 + b^2 + (b + 1)^2 = 149 := by
sorry

end consecutive_integers_sum_of_squares_l446_44675


namespace pythagorean_triple_check_l446_44651

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

theorem pythagorean_triple_check :
  ¬ is_pythagorean_triple 1 1 2 ∧
  ¬ is_pythagorean_triple 3 4 5 ∧
  is_pythagorean_triple 6 8 10 ∧
  ¬ is_pythagorean_triple 1 1 1 :=
sorry

end pythagorean_triple_check_l446_44651


namespace paper_supply_duration_l446_44626

/-- The number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- The number of stories John writes per week -/
def stories_per_week : ℕ := 3

/-- The number of pages in each short story -/
def pages_per_story : ℕ := 50

/-- The number of pages in John's yearly novel -/
def novel_pages_per_year : ℕ := 1200

/-- The number of pages that can fit on one sheet of paper -/
def pages_per_sheet : ℕ := 2

/-- The number of reams of paper John buys -/
def reams_bought : ℕ := 3

/-- The number of sheets in each ream of paper -/
def sheets_per_ream : ℕ := 500

/-- The number of weeks John is buying paper for -/
def weeks_of_paper_supply : ℕ := 18

theorem paper_supply_duration :
  let total_pages_per_week := stories_per_week * pages_per_story + novel_pages_per_year / weeks_per_year
  let sheets_per_week := (total_pages_per_week + pages_per_sheet - 1) / pages_per_sheet
  let total_sheets := reams_bought * sheets_per_ream
  (total_sheets + sheets_per_week - 1) / sheets_per_week = weeks_of_paper_supply :=
by sorry

end paper_supply_duration_l446_44626


namespace cake_eaten_after_six_trips_l446_44652

/-- The fraction of cake eaten after n trips, given that 1/3 is eaten on the first trip
    and half of the remaining cake is eaten on each subsequent trip -/
def cakeEaten (n : ℕ) : ℚ :=
  if n = 0 then 0
  else if n = 1 then 1/3
  else 1/3 + (1 - 1/3) * (1 - (1/2)^(n-1))

/-- The theorem stating that after 6 trips, 47/48 of the cake is eaten -/
theorem cake_eaten_after_six_trips :
  cakeEaten 6 = 47/48 := by sorry

end cake_eaten_after_six_trips_l446_44652


namespace number_of_elements_in_set_l446_44638

theorem number_of_elements_in_set (initial_average : ℚ) (incorrect_number : ℚ) (correct_number : ℚ) (correct_average : ℚ) (n : ℕ) : 
  initial_average = 21 →
  incorrect_number = 26 →
  correct_number = 36 →
  correct_average = 22 →
  (n : ℚ) * initial_average + (correct_number - incorrect_number) = (n : ℚ) * correct_average →
  n = 10 := by
sorry

end number_of_elements_in_set_l446_44638


namespace nates_run_ratio_l446_44665

theorem nates_run_ratio (total_distance field_length rest_distance : ℕ) 
  (h1 : total_distance = 1172)
  (h2 : field_length = 168)
  (h3 : rest_distance = 500)
  (h4 : ∃ k : ℕ, total_distance - rest_distance = k * field_length) :
  (total_distance - rest_distance) / field_length = 4 := by
sorry

end nates_run_ratio_l446_44665


namespace polynomial_simplification_l446_44690

theorem polynomial_simplification (x : ℝ) :
  5 - 3*x - 7*x^2 + 11 - 5*x + 9*x^2 - 13 + 7*x - 4*x^3 + 7*x^2 + 2*x^3 =
  3 - x + 9*x^2 - 2*x^3 := by
  sorry

end polynomial_simplification_l446_44690


namespace cone_rolling_ratio_l446_44634

theorem cone_rolling_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  (2 * Real.pi * Real.sqrt (r^2 + h^2) = 30 * Real.pi * r) → h / r = 4 * Real.sqrt 14 := by
  sorry

end cone_rolling_ratio_l446_44634


namespace range_of_trigonometric_function_l446_44609

theorem range_of_trigonometric_function :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 →
  (π / 2 - Real.arctan 2) ≤ Real.arcsin x + Real.arccos x + Real.arctan (2 * x) ∧
  Real.arcsin x + Real.arccos x + Real.arctan (2 * x) ≤ (π / 2 + Real.arctan 2) :=
by sorry

end range_of_trigonometric_function_l446_44609


namespace geometry_theorem_l446_44649

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (planePerp : Plane → Plane → Prop)

-- State the theorem
theorem geometry_theorem 
  (α β : Plane) (m n : Line) 
  (h_distinct_planes : α ≠ β) 
  (h_distinct_lines : m ≠ n) :
  (∀ (m n : Line) (α : Plane), perpendicular m α → perpendicular n α → parallel m n) ∧
  (∀ (m n : Line) (α β : Plane), perpendicular m α → parallel m n → contains β n → planePerp α β) :=
sorry

end geometry_theorem_l446_44649


namespace absolute_value_and_square_l446_44630

theorem absolute_value_and_square (x : ℝ) : 
  (x < 0 → abs x > x) ∧ (x > 2 → x^2 > 4) := by
  sorry

end absolute_value_and_square_l446_44630


namespace area_ACE_is_60_l446_44662

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : Point)

-- Define the intersection point O of diagonals AC and BD
def O (q : Quadrilateral) : Point := sorry

-- Define the height DE of triangle DBC
def DE (q : Quadrilateral) : Real := 15

-- Define the length of DC
def DC (q : Quadrilateral) : Real := 17

-- Define the areas of triangles
def area_ABO (q : Quadrilateral) : Real := sorry
def area_DCO (q : Quadrilateral) : Real := sorry
def area_ACE (q : Quadrilateral) : Real := sorry

-- State the theorem
theorem area_ACE_is_60 (q : Quadrilateral) :
  area_ABO q = area_DCO q → area_ACE q = 60 := by
  sorry

end area_ACE_is_60_l446_44662


namespace extremum_implies_a_and_monotonicity_l446_44693

/-- The function f(x) = ax^3 - x^2 with a ∈ ℝ -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x^2

/-- The derivative of f(x) -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 2 * x

theorem extremum_implies_a_and_monotonicity (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ 1/3 ∧ |x - 1/3| < ε → |f a x| ≤ |f a (1/3)|) →
  (a = 6 ∧
   (∀ (x y : ℝ), (x < y ∧ y < 0) → f a x < f a y) ∧
   (∀ (x y : ℝ), (0 < x ∧ x < y ∧ y < 1/3) → f a x > f a y) ∧
   (∀ (x y : ℝ), (1/3 < x ∧ x < y) → f a x < f a y)) :=
by sorry

end extremum_implies_a_and_monotonicity_l446_44693


namespace eight_n_even_when_n_seven_l446_44608

theorem eight_n_even_when_n_seven :
  ∃ k : ℤ, 8 * 7 = 2 * k := by
  sorry

end eight_n_even_when_n_seven_l446_44608


namespace divisibility_of_cubic_difference_l446_44633

theorem divisibility_of_cubic_difference (x a b : ℝ) :
  ∃ P : ℝ → ℝ, (x + a + b)^3 - x^3 - a^3 - b^3 = P x * ((x + a) * (x + b)) := by
  sorry

end divisibility_of_cubic_difference_l446_44633


namespace sum_A_B_equals_negative_five_halves_l446_44655

theorem sum_A_B_equals_negative_five_halves (A B : ℚ) :
  (∀ x : ℚ, x ≠ 4 ∧ x ≠ 5 →
    (B * x - 15) / (x^2 - 9*x + 20) = A / (x - 4) + 4 / (x - 5)) →
  A + B = -5/2 := by
sorry

end sum_A_B_equals_negative_five_halves_l446_44655


namespace nine_students_in_front_of_hoseok_l446_44695

/-- The number of students standing in front of Hoseok in a line of 20 students, 
    where 11 students are behind Yoongi and Hoseok is right behind Yoongi. -/
def studentsInFrontOfHoseok (totalStudents : Nat) (studentsBehinYoongi : Nat) : Nat :=
  totalStudents - studentsBehinYoongi

/-- Theorem stating that 9 students are in front of Hoseok given the conditions -/
theorem nine_students_in_front_of_hoseok :
  studentsInFrontOfHoseok 20 11 = 9 := by
  sorry

end nine_students_in_front_of_hoseok_l446_44695


namespace paving_stone_width_l446_44627

/-- Given a rectangular courtyard and paving stones with specific dimensions,
    prove that the width of each paving stone is 2 meters. -/
theorem paving_stone_width
  (courtyard_length : ℝ)
  (courtyard_width : ℝ)
  (stone_length : ℝ)
  (total_stones : ℕ)
  (h1 : courtyard_length = 60)
  (h2 : courtyard_width = 16.5)
  (h3 : stone_length = 2.5)
  (h4 : total_stones = 198) :
  ∃ (stone_width : ℝ), stone_width = 2 ∧
    courtyard_length * courtyard_width = stone_length * stone_width * total_stones :=
by sorry

end paving_stone_width_l446_44627


namespace inequality_condition_l446_44616

theorem inequality_condition (x : ℝ) : (x - Real.pi) * (x - Real.exp 1) ≤ 0 ↔ Real.exp 1 < x ∧ x < Real.pi := by
  sorry

end inequality_condition_l446_44616


namespace f_is_fraction_l446_44666

-- Define what a fraction is
def is_fraction (f : ℚ → ℚ) : Prop :=
  ∃ (a b : ℚ), ∀ x, b ≠ 0 → f x = a / b

-- Define the specific function we're proving is a fraction
def f (x : ℚ) : ℚ := x / (x + 2)

-- Theorem statement
theorem f_is_fraction : is_fraction f := by sorry

end f_is_fraction_l446_44666


namespace circle_equation_correct_l446_44663

def circle_equation (x y : ℝ) : Prop :=
  (x - 4)^2 + (y + 6)^2 = 16

def is_on_circle (x y : ℝ) : Prop :=
  ((x - 4)^2 + (y + 6)^2) = 16

theorem circle_equation_correct :
  ∀ x y : ℝ, is_on_circle x y ↔ 
    ((x - 4)^2 + (y + 6)^2 = 16 ∧ 
     (x - 4)^2 + (y - (-6))^2 = 4^2) :=
by sorry

end circle_equation_correct_l446_44663


namespace apples_to_sell_for_target_profit_l446_44644

/-- Represents the number of apples bought in one transaction -/
def apples_bought : ℕ := 4

/-- Represents the cost in cents for buying apples_bought apples -/
def buying_cost : ℕ := 15

/-- Represents the number of apples sold in one transaction -/
def apples_sold : ℕ := 7

/-- Represents the revenue in cents from selling apples_sold apples -/
def selling_revenue : ℕ := 35

/-- Represents the target profit in cents -/
def target_profit : ℕ := 140

/-- Theorem stating that 112 apples need to be sold to achieve the target profit -/
theorem apples_to_sell_for_target_profit :
  (selling_revenue * 112 / apples_sold) - (buying_cost * 112 / apples_bought) = target_profit := by
  sorry

end apples_to_sell_for_target_profit_l446_44644


namespace charlie_crayon_count_l446_44656

/-- The number of crayons each person has -/
structure CrayonCounts where
  billie : ℕ
  bobbie : ℕ
  lizzie : ℕ
  charlie : ℕ

/-- The conditions of the crayon problem -/
def crayon_problem (c : CrayonCounts) : Prop :=
  c.billie = 18 ∧
  c.bobbie = 3 * c.billie ∧
  c.lizzie = c.bobbie / 2 ∧
  c.charlie = 2 * c.lizzie

theorem charlie_crayon_count (c : CrayonCounts) (h : crayon_problem c) : c.charlie = 54 := by
  sorry

end charlie_crayon_count_l446_44656


namespace box_width_is_ten_inches_l446_44657

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  height : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℝ := d.height * d.width * d.length

theorem box_width_is_ten_inches (box : Dimensions) (block : Dimensions) :
  box.height = 8 →
  box.length = 12 →
  block.height = 3 →
  block.width = 2 →
  block.length = 4 →
  volume box = 40 * volume block →
  box.width = 10 := by
  sorry

end box_width_is_ten_inches_l446_44657


namespace correct_price_reduction_equation_l446_44604

/-- Represents the price reduction scenario for a shirt -/
def price_reduction_scenario (original_price final_price : ℝ) (x : ℝ) : Prop :=
  original_price * (1 - x)^2 = final_price

/-- The theorem stating that the given equation correctly represents the price reduction scenario -/
theorem correct_price_reduction_equation :
  price_reduction_scenario 400 200 x ↔ 400 * (1 - x)^2 = 200 :=
by sorry

end correct_price_reduction_equation_l446_44604


namespace neg_neg_two_eq_two_neg_six_plus_six_eq_zero_neg_three_times_five_eq_neg_fifteen_two_x_minus_three_x_eq_neg_x_l446_44683

-- Problem 1
theorem neg_neg_two_eq_two : -(-2) = 2 := by sorry

-- Problem 2
theorem neg_six_plus_six_eq_zero : -6 + 6 = 0 := by sorry

-- Problem 3
theorem neg_three_times_five_eq_neg_fifteen : (-3) * 5 = -15 := by sorry

-- Problem 4
theorem two_x_minus_three_x_eq_neg_x (x : ℤ) : 2*x - 3*x = -x := by sorry

end neg_neg_two_eq_two_neg_six_plus_six_eq_zero_neg_three_times_five_eq_neg_fifteen_two_x_minus_three_x_eq_neg_x_l446_44683


namespace tan_45_degrees_equals_one_l446_44602

theorem tan_45_degrees_equals_one : Real.tan (π / 4) = 1 := by
  sorry

end tan_45_degrees_equals_one_l446_44602
