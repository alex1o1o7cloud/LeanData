import Mathlib

namespace NUMINAMATH_CALUDE_pokemon_cards_cost_l1987_198780

/-- The cost of a pack of Pokemon cards -/
def pokemon_cost (football_pack_cost baseball_deck_cost total_cost : ℚ) : ℚ :=
  total_cost - (2 * football_pack_cost + baseball_deck_cost)

/-- Theorem: The cost of the Pokemon cards is $4.01 -/
theorem pokemon_cards_cost : 
  pokemon_cost 2.73 8.95 18.42 = 4.01 := by
  sorry

end NUMINAMATH_CALUDE_pokemon_cards_cost_l1987_198780


namespace NUMINAMATH_CALUDE_student_selection_plans_l1987_198708

/-- The number of students in the group -/
def total_students : ℕ := 5

/-- The number of students to be selected -/
def selected_students : ℕ := 4

/-- The number of competitions -/
def num_competitions : ℕ := 4

/-- The number of competitions student A cannot participate in -/
def restricted_competitions : ℕ := 2

/-- The number of different plans to select students for competitions -/
def num_plans : ℕ := 72

theorem student_selection_plans :
  (Nat.choose total_students selected_students * Nat.factorial selected_students) +
  (Nat.choose (total_students - 1) (selected_students - 1) *
   Nat.choose (num_competitions - restricted_competitions) 1 *
   Nat.factorial (selected_students - 1)) = num_plans :=
sorry

end NUMINAMATH_CALUDE_student_selection_plans_l1987_198708


namespace NUMINAMATH_CALUDE_geometric_with_arithmetic_subsequence_l1987_198705

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Define an arithmetic subsequence
def arithmetic_subsequence (a : ℕ → ℝ) (sub : ℕ → ℕ) (d : ℝ) : Prop :=
  ∀ k : ℕ, a (sub (k + 1)) = a (sub k) + d

-- Main theorem
theorem geometric_with_arithmetic_subsequence
  (a : ℕ → ℝ) (q : ℝ) (sub : ℕ → ℕ) (d : ℝ) :
  geometric_sequence a q →
  q ≠ 1 →
  arithmetic_subsequence a sub d →
  (∀ k : ℕ, sub (k + 1) > sub k) →
  q = -1 :=
sorry

end NUMINAMATH_CALUDE_geometric_with_arithmetic_subsequence_l1987_198705


namespace NUMINAMATH_CALUDE_integer_expression_multiple_of_three_l1987_198717

theorem integer_expression_multiple_of_three (n k : ℕ) (h1 : 1 ≤ k) (h2 : k < n) (h3 : ∃ m : ℕ, n = 3 * m) :
  ∃ z : ℤ, (2 * n - 3 * k - 2) * (n.choose k) = (k + 2) * z := by
  sorry

end NUMINAMATH_CALUDE_integer_expression_multiple_of_three_l1987_198717


namespace NUMINAMATH_CALUDE_permutations_of_eight_distinct_objects_l1987_198707

theorem permutations_of_eight_distinct_objects : 
  Nat.factorial 8 = 40320 := by sorry

end NUMINAMATH_CALUDE_permutations_of_eight_distinct_objects_l1987_198707


namespace NUMINAMATH_CALUDE_tan_two_fifths_pi_plus_theta_l1987_198714

theorem tan_two_fifths_pi_plus_theta (θ : ℝ) 
  (h : Real.sin ((12 / 5) * Real.pi + θ) + 2 * Real.sin ((11 / 10) * Real.pi - θ) = 0) : 
  Real.tan ((2 / 5) * Real.pi + θ) = 2 := by
sorry

end NUMINAMATH_CALUDE_tan_two_fifths_pi_plus_theta_l1987_198714


namespace NUMINAMATH_CALUDE_paige_folders_l1987_198784

def initial_files : ℕ := 135
def deleted_files : ℕ := 27
def files_per_folder : ℚ := 8.5

theorem paige_folders : 
  ∃ (folders : ℕ), 
    folders = (initial_files - deleted_files : ℚ) / files_per_folder
    ∧ folders = 13 := by sorry

end NUMINAMATH_CALUDE_paige_folders_l1987_198784


namespace NUMINAMATH_CALUDE_incorrect_division_l1987_198793

theorem incorrect_division (D : ℕ) (h : D / 36 = 48) : D / 72 = 24 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_division_l1987_198793


namespace NUMINAMATH_CALUDE_bryans_book_collection_l1987_198740

theorem bryans_book_collection (total_books : ℕ) (num_continents : ℕ) 
  (h1 : total_books = 488) 
  (h2 : num_continents = 4) 
  (h3 : total_books % num_continents = 0) : 
  total_books / num_continents = 122 := by
sorry

end NUMINAMATH_CALUDE_bryans_book_collection_l1987_198740


namespace NUMINAMATH_CALUDE_min_value_theorem_l1987_198729

def f (x : ℝ) := |2*x + 1| + |x - 1|

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : (1/2)*a + b + 2*c = 3/2) : 
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (a^2 + b^2 + c^2 ≥ 3/7) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1987_198729


namespace NUMINAMATH_CALUDE_find_k_l1987_198791

/-- The function f(x) -/
def f (k a x : ℝ) : ℝ := 2*k + (k^3)*a - x

/-- The function g(x) -/
def g (k a x : ℝ) : ℝ := x^2 + f k a x

theorem find_k (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ k : ℝ,
    (∀ x : ℝ, f k a x = -f k a (-x)) ∧  -- f is odd
    (∃ x : ℝ, f k a x = 3) ∧  -- f = 3 for some x
    (∀ x : ℝ, x ≥ 2 → g k a x ≥ -2) ∧  -- g has minimum -2 on [2, +∞)
    (∃ x : ℝ, x ≥ 2 ∧ g k a x = -2) ∧  -- g achieves minimum -2 on [2, +∞)
    k = 1 :=
  sorry

end NUMINAMATH_CALUDE_find_k_l1987_198791


namespace NUMINAMATH_CALUDE_vector_at_t_4_l1987_198731

/-- A line in 3D space parameterized by t -/
structure ParametricLine where
  point : ℝ → (ℝ × ℝ × ℝ)

/-- The given line satisfying the conditions -/
def given_line : ParametricLine :=
  { point := sorry }

theorem vector_at_t_4 :
  given_line.point 1 = (4, 5, 9) →
  given_line.point 3 = (1, 0, -2) →
  given_line.point 4 = (-1, 0, -15) :=
by sorry

end NUMINAMATH_CALUDE_vector_at_t_4_l1987_198731


namespace NUMINAMATH_CALUDE_distance_difference_l1987_198732

/-- The walking speed of Taehyung in meters per minute -/
def taehyung_speed : ℕ := 114

/-- The walking speed of Minyoung in meters per minute -/
def minyoung_speed : ℕ := 79

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Theorem stating the difference in distance walked by Taehyung and Minyoung in an hour -/
theorem distance_difference : 
  taehyung_speed * minutes_per_hour - minyoung_speed * minutes_per_hour = 2100 := by
  sorry


end NUMINAMATH_CALUDE_distance_difference_l1987_198732


namespace NUMINAMATH_CALUDE_rex_driving_lessons_l1987_198795

/-- The number of hour-long lessons Rex wants to take before his test -/
def total_lessons : ℕ := 40

/-- The number of hours of lessons Rex takes per week -/
def hours_per_week : ℕ := 4

/-- The number of weeks Rex has already completed -/
def completed_weeks : ℕ := 6

/-- The number of additional weeks Rex needs to reach his goal -/
def additional_weeks : ℕ := 4

/-- Theorem stating that the total number of hour-long lessons Rex wants to take is 40 -/
theorem rex_driving_lessons :
  total_lessons = hours_per_week * (completed_weeks + additional_weeks) :=
by sorry

end NUMINAMATH_CALUDE_rex_driving_lessons_l1987_198795


namespace NUMINAMATH_CALUDE_midpoint_sum_equals_vertex_sum_l1987_198728

theorem midpoint_sum_equals_vertex_sum (a b c : ℝ) :
  let vertex_sum := a + b + c
  let midpoint_sum := (a + b) / 2 + (a + c) / 2 + (b + c) / 2
  vertex_sum = midpoint_sum := by
  sorry

end NUMINAMATH_CALUDE_midpoint_sum_equals_vertex_sum_l1987_198728


namespace NUMINAMATH_CALUDE_three_true_propositions_l1987_198790

theorem three_true_propositions (a b c d : ℝ) : 
  (∃ (p q r : Prop), 
    (p ∧ q → r) ∧ 
    (p ∧ r → q) ∧ 
    (q ∧ r → p) ∧
    (p = (a * b > 0)) ∧ 
    (q = (-c / a < -d / b)) ∧ 
    (r = (b * c > a * d))) :=
by sorry

end NUMINAMATH_CALUDE_three_true_propositions_l1987_198790


namespace NUMINAMATH_CALUDE_ramesh_investment_l1987_198737

def suresh_investment : ℕ := 24000
def total_profit : ℕ := 19000
def ramesh_profit_share : ℕ := 11875

theorem ramesh_investment :
  ∃ (ramesh_investment : ℕ),
    (ramesh_investment * suresh_investment * ramesh_profit_share
     = (total_profit - ramesh_profit_share) * suresh_investment * total_profit)
    ∧ ramesh_investment = 42000 :=
by sorry

end NUMINAMATH_CALUDE_ramesh_investment_l1987_198737


namespace NUMINAMATH_CALUDE_final_game_score_l1987_198797

/-- Represents the points scored by each player in the basketball game -/
structure TeamPoints where
  bailey : ℕ
  michiko : ℕ
  akiko : ℕ
  chandra : ℕ

/-- Calculates the total points scored by the team -/
def total_points (t : TeamPoints) : ℕ :=
  t.bailey + t.michiko + t.akiko + t.chandra

/-- Theorem stating the total points scored by the team under given conditions -/
theorem final_game_score (t : TeamPoints) 
  (h1 : t.bailey = 14)
  (h2 : t.michiko = t.bailey / 2)
  (h3 : t.akiko = t.michiko + 4)
  (h4 : t.chandra = 2 * t.akiko) :
  total_points t = 54 := by
  sorry

end NUMINAMATH_CALUDE_final_game_score_l1987_198797


namespace NUMINAMATH_CALUDE_parallelogram_area_l1987_198701

/-- The area of a parallelogram with one angle of 150 degrees and two consecutive sides of lengths 12 inches and 20 inches is equal to 120 square inches. -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) : 
  a = 12 → b = 20 → θ = 150 * π / 180 → 
  a * b * Real.sin θ = 120 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1987_198701


namespace NUMINAMATH_CALUDE_function_value_at_point_l1987_198756

theorem function_value_at_point (h : ℝ → ℝ) (b : ℝ) :
  (∀ x, h x = 4 * x - 5) →
  h b = 1 ↔ b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_point_l1987_198756


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_five_elevenths_l1987_198771

theorem sum_of_fractions_equals_five_elevenths :
  (1 / (2^2 - 1) + 1 / (4^2 - 1) + 1 / (6^2 - 1) + 1 / (8^2 - 1) + 1 / (10^2 - 1) : ℚ) = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_five_elevenths_l1987_198771


namespace NUMINAMATH_CALUDE_marley_has_31_fruits_l1987_198725

-- Define the number of fruits for Louis and Samantha
def louis_oranges : ℕ := 5
def louis_apples : ℕ := 3
def samantha_oranges : ℕ := 8
def samantha_apples : ℕ := 7

-- Define Marley's fruits based on the conditions
def marley_oranges : ℕ := 2 * louis_oranges
def marley_apples : ℕ := 3 * samantha_apples

-- Define Marley's total fruits
def marley_total_fruits : ℕ := marley_oranges + marley_apples

-- Theorem statement
theorem marley_has_31_fruits : marley_total_fruits = 31 := by
  sorry

end NUMINAMATH_CALUDE_marley_has_31_fruits_l1987_198725


namespace NUMINAMATH_CALUDE_chess_probabilities_l1987_198778

theorem chess_probabilities (p_draw p_b_win : ℝ) 
  (h_draw : p_draw = 1/2)
  (h_b_win : p_b_win = 1/3) :
  let p_a_win := 1 - p_draw - p_b_win
  let p_a_not_lose := p_draw + p_a_win
  (p_a_win = 1/6) ∧ (p_a_not_lose = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_chess_probabilities_l1987_198778


namespace NUMINAMATH_CALUDE_cube_root_equation_l1987_198718

theorem cube_root_equation (x : ℝ) : 
  (x * (x^5)^(1/4))^(1/3) = 5 ↔ x = 5 * 5^(1/3) :=
sorry

end NUMINAMATH_CALUDE_cube_root_equation_l1987_198718


namespace NUMINAMATH_CALUDE_range_of_expression_l1987_198700

theorem range_of_expression (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_one : x + y + z = 1) :
  0 < x * y + y * z + z * x - 2 * x * y * z ∧ 
  x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
by sorry

end NUMINAMATH_CALUDE_range_of_expression_l1987_198700


namespace NUMINAMATH_CALUDE_green_marbles_in_basket_b_l1987_198759

/-- Represents a basket with two types of marbles -/
structure Basket :=
  (color1 : Nat)
  (color2 : Nat)

/-- Calculates the absolute difference between two numbers -/
def absDiff (a b : Nat) : Nat :=
  max a b - min a b

/-- Finds the maximum difference among a list of baskets -/
def maxDiff (baskets : List Basket) : Nat :=
  baskets.map (λ b => absDiff b.color1 b.color2) |>.maximum?
    |>.getD 0

theorem green_marbles_in_basket_b :
  let basketA : Basket := ⟨4, 2⟩
  let basketC : Basket := ⟨3, 9⟩
  let basketB : Basket := ⟨x, 1⟩
  let allBaskets : List Basket := [basketA, basketB, basketC]
  maxDiff allBaskets = 6 →
  x = 7 :=
by sorry

end NUMINAMATH_CALUDE_green_marbles_in_basket_b_l1987_198759


namespace NUMINAMATH_CALUDE_table_function_proof_l1987_198768

def f (x : ℝ) : ℝ := x^2 - x + 2

theorem table_function_proof :
  (f 2 = 3) ∧ (f 3 = 8) ∧ (f 4 = 15) ∧ (f 5 = 24) ∧ (f 6 = 35) := by
  sorry

end NUMINAMATH_CALUDE_table_function_proof_l1987_198768


namespace NUMINAMATH_CALUDE_math_majors_consecutive_probability_l1987_198730

/-- The number of people sitting at the table -/
def total_people : ℕ := 12

/-- The number of math majors -/
def math_majors : ℕ := 5

/-- The number of physics majors -/
def physics_majors : ℕ := 4

/-- The number of chemistry majors -/
def chemistry_majors : ℕ := 3

/-- The probability of all math majors sitting consecutively -/
def prob_consecutive_math : ℚ := 1 / 66

theorem math_majors_consecutive_probability :
  (total_people : ℚ) / (total_people.choose math_majors) = prob_consecutive_math := by
  sorry

end NUMINAMATH_CALUDE_math_majors_consecutive_probability_l1987_198730


namespace NUMINAMATH_CALUDE_isosceles_triangle_l1987_198723

theorem isosceles_triangle (A B C : ℝ) (h₁ : A + B + C = π) 
  (h₂ : (Real.sin A + Real.sin B) * (Real.cos A + Real.cos B) = 2 * Real.sin C) : 
  A = B :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l1987_198723


namespace NUMINAMATH_CALUDE_cyclist_pedestrian_meeting_point_l1987_198798

/-- The meeting point of a cyclist and a pedestrian on a straight path --/
theorem cyclist_pedestrian_meeting_point (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let total_distance := a + b
  let cyclist_speed := total_distance
  let pedestrian_speed := a
  let meeting_point := a * (a + b) / (2 * a + b)
  meeting_point / cyclist_speed = (a - meeting_point) / pedestrian_speed ∧
  meeting_point < a :=
by sorry

end NUMINAMATH_CALUDE_cyclist_pedestrian_meeting_point_l1987_198798


namespace NUMINAMATH_CALUDE_shop_width_calculation_l1987_198764

/-- Calculates the width of a shop given its monthly rent, length, and annual rent per square foot. -/
theorem shop_width_calculation (monthly_rent : ℕ) (length : ℕ) (annual_rent_per_sqft : ℕ) : 
  monthly_rent = 1440 → length = 18 → annual_rent_per_sqft = 48 → 
  (monthly_rent * 12) / (annual_rent_per_sqft * length) = 20 := by
  sorry

end NUMINAMATH_CALUDE_shop_width_calculation_l1987_198764


namespace NUMINAMATH_CALUDE_monotonic_increasing_interval_f_l1987_198750

noncomputable def f (x : ℝ) := (x - 2) * Real.exp x

theorem monotonic_increasing_interval_f :
  {x : ℝ | ∀ y, x < y → f x < f y} = {x : ℝ | x > 1} := by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_interval_f_l1987_198750


namespace NUMINAMATH_CALUDE_proof_by_contradiction_assumption_l1987_198739

theorem proof_by_contradiction_assumption 
  (P : ℝ → ℝ → Prop) 
  (Q : ℝ → Prop) 
  (R : ℝ → Prop) 
  (h : ∀ x y, P x y → (Q x ∨ R y)) :
  (∀ x y, P x y → (Q x ∨ R y)) ↔ 
  (∀ x y, P x y ∧ ¬Q x ∧ ¬R y → False) := by
sorry

end NUMINAMATH_CALUDE_proof_by_contradiction_assumption_l1987_198739


namespace NUMINAMATH_CALUDE_school_boys_count_l1987_198749

theorem school_boys_count (boys girls : ℕ) : 
  (boys : ℚ) / girls = 5 / 13 →
  girls = boys + 128 →
  boys = 80 := by
sorry

end NUMINAMATH_CALUDE_school_boys_count_l1987_198749


namespace NUMINAMATH_CALUDE_infinitely_many_odd_floor_squares_l1987_198782

theorem infinitely_many_odd_floor_squares (α : ℝ) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, Odd ⌊n^2 * α⌋ :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_odd_floor_squares_l1987_198782


namespace NUMINAMATH_CALUDE_time_to_fill_tank_l1987_198738

/-- Represents the tank and pipe system -/
structure TankSystem where
  capacity : ℝ
  pipeA_rate : ℝ
  pipeB_rate : ℝ
  pipeC_rate : ℝ
  pipeA_time : ℝ
  pipeB_time : ℝ
  pipeC_time : ℝ

/-- Calculates the net volume filled in one cycle -/
def netVolumeFilled (system : TankSystem) : ℝ :=
  system.pipeA_rate * system.pipeA_time +
  system.pipeB_rate * system.pipeB_time -
  system.pipeC_rate * system.pipeC_time

/-- Calculates the time for one cycle -/
def cycleTime (system : TankSystem) : ℝ :=
  system.pipeA_time + system.pipeB_time + system.pipeC_time

/-- Theorem stating the time to fill the tank -/
theorem time_to_fill_tank (system : TankSystem)
  (h1 : system.capacity = 2000)
  (h2 : system.pipeA_rate = 200)
  (h3 : system.pipeB_rate = 50)
  (h4 : system.pipeC_rate = 25)
  (h5 : system.pipeA_time = 1)
  (h6 : system.pipeB_time = 2)
  (h7 : system.pipeC_time = 2) :
  (system.capacity / netVolumeFilled system) * cycleTime system = 40 := by
  sorry

end NUMINAMATH_CALUDE_time_to_fill_tank_l1987_198738


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_times_three_l1987_198703

theorem arithmetic_sequence_sum_times_three (a₁ l n : ℕ) (h1 : n = 11) (h2 : a₁ = 101) (h3 : l = 121) :
  3 * (a₁ + (a₁ + 2) + (a₁ + 4) + (a₁ + 6) + (a₁ + 8) + (a₁ + 10) + (a₁ + 12) + (a₁ + 14) + (a₁ + 16) + (a₁ + 18) + l) = 3663 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_times_three_l1987_198703


namespace NUMINAMATH_CALUDE_fiftieth_ring_squares_l1987_198760

/-- The number of unit squares in the nth ring of a square array with a center square,
    where each ring increases by 3 on each side. -/
def ring_squares (n : ℕ) : ℕ := 8 * n + 8

/-- The 50th ring contains 408 unit squares -/
theorem fiftieth_ring_squares : ring_squares 50 = 408 := by
  sorry

#eval ring_squares 50  -- This will evaluate to 408

end NUMINAMATH_CALUDE_fiftieth_ring_squares_l1987_198760


namespace NUMINAMATH_CALUDE_arithmetic_correctness_l1987_198787

theorem arithmetic_correctness : 
  ((-4) + (-5) = -9) ∧ 
  (4 / (-2) = -2) ∧ 
  (-5 - (-6) ≠ 11) ∧ 
  (-2 * (-10) ≠ -20) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_correctness_l1987_198787


namespace NUMINAMATH_CALUDE_certain_number_minus_two_l1987_198766

theorem certain_number_minus_two (x : ℝ) (h : 6 - x = 2) : x - 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_minus_two_l1987_198766


namespace NUMINAMATH_CALUDE_new_boy_age_l1987_198742

theorem new_boy_age (initial_size : Nat) (initial_avg : Nat) (time_passed : Nat) (new_size : Nat) :
  initial_size = 6 →
  initial_avg = 19 →
  time_passed = 3 →
  new_size = 7 →
  (initial_size * initial_avg + initial_size * time_passed + 1) / new_size = initial_avg →
  1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_new_boy_age_l1987_198742


namespace NUMINAMATH_CALUDE_sqrt_49_times_sqrt_25_l1987_198773

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_49_times_sqrt_25_l1987_198773


namespace NUMINAMATH_CALUDE_ratio_of_percentages_l1987_198757

theorem ratio_of_percentages (A B C D : ℝ) 
  (hA : A = 0.4 * B) 
  (hB : B = 0.25 * C) 
  (hD : D = 0.6 * C) 
  (hC : C ≠ 0) : A / D = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_percentages_l1987_198757


namespace NUMINAMATH_CALUDE_odd_factors_of_360_l1987_198785

/-- The number of odd factors of 360 -/
def num_odd_factors_360 : ℕ := sorry

/-- 360 is the number we're considering -/
def n : ℕ := 360

theorem odd_factors_of_360 : num_odd_factors_360 = 6 := by sorry

end NUMINAMATH_CALUDE_odd_factors_of_360_l1987_198785


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l1987_198727

theorem inequality_system_solution_range (m : ℝ) : 
  (∃! (a b : ℤ), (a ≠ b) ∧ 
    ((a : ℝ) > -2) ∧ ((a : ℝ) ≤ (m + 2) / 3) ∧
    ((b : ℝ) > -2) ∧ ((b : ℝ) ≤ (m + 2) / 3) ∧
    (∀ (x : ℤ), (x ≠ a ∧ x ≠ b) → 
      ¬((x : ℝ) > -2 ∧ (x : ℝ) ≤ (m + 2) / 3))) →
  (-2 : ℝ) ≤ m ∧ m < 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l1987_198727


namespace NUMINAMATH_CALUDE_distribute_5_3_l1987_198753

/-- The number of ways to distribute n distinct objects into k identical containers,
    allowing empty containers. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 51 ways to distribute 5 distinct objects into 3 identical containers,
    allowing empty containers. -/
theorem distribute_5_3 : distribute 5 3 = 51 := by sorry

end NUMINAMATH_CALUDE_distribute_5_3_l1987_198753


namespace NUMINAMATH_CALUDE_marathon_volunteer_assignment_l1987_198775

def number_of_students : ℕ := 5
def number_of_tasks : ℕ := 4
def number_of_students_who_can_drive : ℕ := 3

theorem marathon_volunteer_assignment :
  let total_arrangements := 
    (Nat.choose number_of_students_who_can_drive 1 * 
     Nat.choose (number_of_students - 1) 2 * 
     Nat.factorial 3) +
    (Nat.choose number_of_students_who_can_drive 2 * 
     Nat.factorial 3)
  total_arrangements = 
    Nat.choose number_of_students_who_can_drive 1 * 
    Nat.choose number_of_students 2 * 
    Nat.factorial 3 +
    Nat.choose number_of_students_who_can_drive 2 * 
    Nat.factorial 3 := by
  sorry

end NUMINAMATH_CALUDE_marathon_volunteer_assignment_l1987_198775


namespace NUMINAMATH_CALUDE_employee_pay_calculation_l1987_198746

/-- Given two employees with a total weekly pay of 560, where one employee's pay is 150% of the other's, prove that the employee with the lower pay receives 224 per week. -/
theorem employee_pay_calculation (total_pay : ℝ) (a_pay b_pay : ℝ) : 
  total_pay = 560 →
  a_pay = 1.5 * b_pay →
  a_pay + b_pay = total_pay →
  b_pay = 224 := by sorry

end NUMINAMATH_CALUDE_employee_pay_calculation_l1987_198746


namespace NUMINAMATH_CALUDE_braking_distance_at_120_less_than_33_braking_distance_at_40_equals_10_braking_distance_linear_and_nonnegative_l1987_198748

/-- Represents the braking distance function for a car -/
def brakingDistance (speed : ℝ) : ℝ :=
  0.25 * speed

/-- Theorem: The braking distance at 120 km/h is less than 33m -/
theorem braking_distance_at_120_less_than_33 :
  brakingDistance 120 < 33 := by
  sorry

/-- Theorem: The braking distance at 40 km/h is 10m -/
theorem braking_distance_at_40_equals_10 :
  brakingDistance 40 = 10 := by
  sorry

/-- Theorem: The braking distance function is linear and non-negative for non-negative speeds -/
theorem braking_distance_linear_and_nonnegative :
  ∀ (speed : ℝ), speed ≥ 0 → brakingDistance speed ≥ 0 ∧ 
  ∀ (speed1 speed2 : ℝ), brakingDistance (speed1 + speed2) = brakingDistance speed1 + brakingDistance speed2 := by
  sorry

end NUMINAMATH_CALUDE_braking_distance_at_120_less_than_33_braking_distance_at_40_equals_10_braking_distance_linear_and_nonnegative_l1987_198748


namespace NUMINAMATH_CALUDE_fraction_irreducible_l1987_198734

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l1987_198734


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l1987_198724

theorem geometric_sequence_seventh_term 
  (a : ℝ) -- first term
  (r : ℝ) -- common ratio
  (h1 : a = 2) -- first term is 2
  (h2 : a * r^8 = 32) -- last term (9th term) is 32
  : a * r^6 = 128 := by -- seventh term is 128
sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l1987_198724


namespace NUMINAMATH_CALUDE_f_maximum_properties_l1987_198779

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (1 + x) - Real.log x

theorem f_maximum_properties (x₀ : ℝ) 
  (h₁ : ∀ x > 0, f x ≤ f x₀) 
  (h₂ : x₀ > 0) : 
  f x₀ = x₀ ∧ f x₀ < (1/2) := by
  sorry

end NUMINAMATH_CALUDE_f_maximum_properties_l1987_198779


namespace NUMINAMATH_CALUDE_dot_product_range_l1987_198702

theorem dot_product_range (a b : EuclideanSpace ℝ (Fin n)) :
  norm a = 2 →
  norm b = 2 →
  (∀ x : ℝ, norm (a + x • b) ≥ 1) →
  -2 * Real.sqrt 3 ≤ inner a b ∧ inner a b ≤ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_range_l1987_198702


namespace NUMINAMATH_CALUDE_g_equals_inverse_at_three_point_five_l1987_198715

def g (x : ℝ) : ℝ := 3 * x - 7

theorem g_equals_inverse_at_three_point_five :
  g (3.5) = (Function.invFun g) (3.5) := by sorry

end NUMINAMATH_CALUDE_g_equals_inverse_at_three_point_five_l1987_198715


namespace NUMINAMATH_CALUDE_jason_clothing_expenses_l1987_198735

/-- The cost of Jason's shorts in dollars -/
def shorts_cost : ℝ := 14.28

/-- The cost of Jason's jacket in dollars -/
def jacket_cost : ℝ := 4.74

/-- The total amount Jason spent on clothing -/
def total_spent : ℝ := shorts_cost + jacket_cost

/-- Theorem stating that the total amount Jason spent on clothing is $19.02 -/
theorem jason_clothing_expenses : total_spent = 19.02 := by
  sorry

end NUMINAMATH_CALUDE_jason_clothing_expenses_l1987_198735


namespace NUMINAMATH_CALUDE_ellipse_triangle_area_l1987_198743

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Define the foci
def are_foci (F₁ F₂ : ℝ × ℝ) : Prop := 
  let (x₁, y₁) := F₁
  let (x₂, y₂) := F₂
  x₁^2 + y₁^2 = 5 ∧ x₂^2 + y₂^2 = 5 ∧ x₁ = -x₂ ∧ y₁ = -y₂

-- Define the distance ratio condition
def distance_ratio (P F₁ F₂ : ℝ × ℝ) : Prop :=
  let d₁ := Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2)
  let d₂ := Real.sqrt ((P.2 - F₂.1)^2 + (P.2 - F₂.2)^2)
  d₁ / d₂ = 2

-- Theorem statement
theorem ellipse_triangle_area 
  (P F₁ F₂ : ℝ × ℝ) 
  (h₁ : is_on_ellipse P.1 P.2) 
  (h₂ : are_foci F₁ F₂) 
  (h₃ : distance_ratio P F₁ F₂) : 
  let area := Real.sqrt (
    (F₁.1 - P.1)^2 + (F₁.2 - P.2)^2 +
    (F₂.1 - P.1)^2 + (F₂.2 - P.2)^2 +
    (F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2
  ) / 4
  area = 4 := by sorry

end NUMINAMATH_CALUDE_ellipse_triangle_area_l1987_198743


namespace NUMINAMATH_CALUDE_sum_of_abs_first_six_terms_l1987_198755

def sequence_a (n : ℕ) : ℤ :=
  -5 + 2 * (n - 1)

theorem sum_of_abs_first_six_terms :
  (∀ n, sequence_a (n + 1) - sequence_a n = 2) →
  sequence_a 1 = -5 →
  (Finset.range 6).sum (fun i => |sequence_a (i + 1)|) = 18 := by
sorry

end NUMINAMATH_CALUDE_sum_of_abs_first_six_terms_l1987_198755


namespace NUMINAMATH_CALUDE_billion_scientific_notation_l1987_198726

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ coefficient
  h2 : coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem billion_scientific_notation :
  toScientificNotation 1075000000 = ScientificNotation.mk 1.075 9 sorry sorry := by
  sorry

end NUMINAMATH_CALUDE_billion_scientific_notation_l1987_198726


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1987_198762

theorem inequality_system_solution (x : ℝ) :
  x > -6 - 2*x ∧ x ≤ (3 + x) / 4 → -2 < x ∧ x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1987_198762


namespace NUMINAMATH_CALUDE_orange_basket_problem_l1987_198720

theorem orange_basket_problem (N : ℕ) : 
  N % 10 = 2 → N % 12 = 0 → N = 72 := by
  sorry

end NUMINAMATH_CALUDE_orange_basket_problem_l1987_198720


namespace NUMINAMATH_CALUDE_sine_in_triangle_l1987_198769

theorem sine_in_triangle (a b : ℝ) (A B : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : Real.sin A = 3/5) :
  Real.sin B = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_sine_in_triangle_l1987_198769


namespace NUMINAMATH_CALUDE_square_circumcenter_segment_length_l1987_198721

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square with side length 1 -/
structure UnitSquare where
  A : Point
  B : Point
  C : Point
  D : Point

/-- The circumcenter of a triangle -/
def circumcenter (A B C : Point) : Point :=
  sorry

/-- The length of a segment between two points -/
def segmentLength (P Q : Point) : ℝ :=
  sorry

/-- The main theorem -/
theorem square_circumcenter_segment_length 
  (ABCD : UnitSquare) 
  (P Q : Point) 
  (h1 : Q = circumcenter B P C) 
  (h2 : D = circumcenter P Q ABCD.A) : 
  segmentLength P Q = Real.sqrt (2 - Real.sqrt 3) ∨ 
  segmentLength P Q = Real.sqrt (2 + Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_square_circumcenter_segment_length_l1987_198721


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1987_198706

/-- A right triangle with perimeter 40 and area 24 has a hypotenuse of length 18.8 -/
theorem right_triangle_hypotenuse : ∃ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Positive side lengths
  a + b + c = 40 ∧  -- Perimeter condition
  (1/2) * a * b = 24 ∧  -- Area condition
  a^2 + b^2 = c^2 ∧  -- Pythagorean theorem (right triangle condition)
  c = 18.8 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1987_198706


namespace NUMINAMATH_CALUDE_consecutive_integers_square_difference_l1987_198722

theorem consecutive_integers_square_difference (n : ℕ) : 
  n > 0 ∧ 
  n + (n + 1) < 150 ∧ 
  (n + 1)^2 - n^2 = 149 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_difference_l1987_198722


namespace NUMINAMATH_CALUDE_rate_percent_is_twelve_l1987_198754

/-- Calculates the rate percent on simple interest given principal, amount, and time. -/
def calculate_rate_percent (principal amount : ℚ) (time : ℕ) : ℚ :=
  let simple_interest := amount - principal
  (simple_interest * 100) / (principal * time)

/-- Theorem stating that the rate percent on simple interest is 12% for the given conditions. -/
theorem rate_percent_is_twelve :
  let principal : ℚ := 750
  let amount : ℚ := 1200
  let time : ℕ := 5
  calculate_rate_percent principal amount time = 12 := by
  sorry

#eval calculate_rate_percent 750 1200 5

end NUMINAMATH_CALUDE_rate_percent_is_twelve_l1987_198754


namespace NUMINAMATH_CALUDE_button_up_shirt_cost_l1987_198792

def total_budget : ℕ := 200
def suit_pants : ℕ := 46
def suit_coat : ℕ := 38
def socks : ℕ := 11
def belt : ℕ := 18
def shoes : ℕ := 41
def amount_left : ℕ := 16

theorem button_up_shirt_cost : 
  total_budget - (suit_pants + suit_coat + socks + belt + shoes + amount_left) = 30 := by
  sorry

end NUMINAMATH_CALUDE_button_up_shirt_cost_l1987_198792


namespace NUMINAMATH_CALUDE_max_value_is_eight_l1987_198786

/-- The feasible region defined by the given constraints -/
def FeasibleRegion (x y : ℝ) : Prop :=
  x + y - 7 ≤ 0 ∧ x - 3*y + 1 ≤ 0 ∧ 3*x - y - 5 ≥ 0

/-- The objective function to be maximized -/
def ObjectiveFunction (x y : ℝ) : ℝ :=
  2*x - y

/-- Theorem stating that the maximum value of the objective function is 8 -/
theorem max_value_is_eight :
  ∃ (x y : ℝ), FeasibleRegion x y ∧
    ∀ (x' y' : ℝ), FeasibleRegion x' y' →
      ObjectiveFunction x y ≥ ObjectiveFunction x' y' ∧
      ObjectiveFunction x y = 8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_is_eight_l1987_198786


namespace NUMINAMATH_CALUDE_ratio_problem_l1987_198712

theorem ratio_problem (first_part : ℝ) (ratio_percent : ℝ) (second_part : ℝ) :
  first_part = 25 →
  ratio_percent = 50 →
  first_part / (first_part + second_part) * 100 = ratio_percent →
  second_part = 25 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1987_198712


namespace NUMINAMATH_CALUDE_milk_container_problem_l1987_198761

/-- The initial quantity of milk in container A --/
def initial_quantity : ℝ := 1264

/-- The fraction of milk in container B relative to A's capacity --/
def fraction_in_B : ℝ := 0.375

/-- The amount transferred from C to B --/
def transfer_amount : ℝ := 158

theorem milk_container_problem :
  (fraction_in_B * initial_quantity + transfer_amount = 
   (1 - fraction_in_B) * initial_quantity - transfer_amount) ∧
  (initial_quantity > 0) := by
  sorry

end NUMINAMATH_CALUDE_milk_container_problem_l1987_198761


namespace NUMINAMATH_CALUDE_ship_total_distance_l1987_198758

/-- Represents the daily travel of a ship --/
structure DailyTravel where
  distance : ℝ
  direction : String

/-- Calculates the total distance traveled by a ship over 4 days --/
def totalDistance (day1 day2 day3 day4 : DailyTravel) : ℝ :=
  day1.distance + day2.distance + day3.distance + day4.distance

/-- Theorem: The ship's total travel distance over 4 days is 960 miles --/
theorem ship_total_distance :
  let day1 := DailyTravel.mk 100 "north"
  let day2 := DailyTravel.mk (3 * 100) "east"
  let day3 := DailyTravel.mk (3 * 100 + 110) "east"
  let day4 := DailyTravel.mk 150 "30-degree angle with north"
  totalDistance day1 day2 day3 day4 = 960 := by
  sorry

#check ship_total_distance

end NUMINAMATH_CALUDE_ship_total_distance_l1987_198758


namespace NUMINAMATH_CALUDE_perfect_cube_implies_one_l1987_198719

theorem perfect_cube_implies_one (a : ℕ) 
  (h : ∀ n : ℕ, ∃ k : ℕ, 4 * (a^n + 1) = k^3) : 
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_perfect_cube_implies_one_l1987_198719


namespace NUMINAMATH_CALUDE_log_343_equation_solution_l1987_198704

theorem log_343_equation_solution (x : ℝ) : 
  (Real.log 343 / Real.log (3 * x) = x) → 
  (∃ (a b : ℤ), x = a / b ∧ b ≠ 0 ∧ ¬∃ (n : ℤ), x = n ∧ ¬∃ (m : ℚ), x = m ^ 2 ∧ ¬∃ (k : ℚ), x = k ^ 3) := by
  sorry

end NUMINAMATH_CALUDE_log_343_equation_solution_l1987_198704


namespace NUMINAMATH_CALUDE_line_not_in_fourth_quadrant_l1987_198736

/-- The line l: ax + by + c = 0 does not pass through the fourth quadrant when ab < 0 and bc < 0 -/
theorem line_not_in_fourth_quadrant (a b c : ℝ) (h1 : a * b < 0) (h2 : b * c < 0) :
  ∃ (x y : ℝ), x > 0 ∧ y < 0 ∧ a * x + b * y + c ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_line_not_in_fourth_quadrant_l1987_198736


namespace NUMINAMATH_CALUDE_perfect_square_quadratic_l1987_198796

theorem perfect_square_quadratic (k : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 - 12*x + k = (x + a)^2) ↔ k = 36 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_quadratic_l1987_198796


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l1987_198765

/-- Given a parabola with equation 7x + 4y² = 0, its focus has coordinates (-7/16, 0) -/
theorem parabola_focus_coordinates :
  ∀ (x y : ℝ),
  (7 * x + 4 * y^2 = 0) →
  ∃ (f : ℝ × ℝ),
  f = (-7/16, 0) ∧
  f.1 = -1/(4 * (4/7)) ∧
  f.2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l1987_198765


namespace NUMINAMATH_CALUDE_certain_number_is_36_l1987_198788

theorem certain_number_is_36 : ∃ x : ℝ, 
  ((((x + 10) * 2) / 2) - 2) = 88 / 2 ∧ x = 36 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_36_l1987_198788


namespace NUMINAMATH_CALUDE_complex_number_properties_l1987_198733

theorem complex_number_properties (z : ℂ) (h : z - 2*I = z*I + 4) :
  z = 1 + 3*I ∧ Complex.abs z = Real.sqrt 10 ∧ ((z - 1) / 3)^2023 = -I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l1987_198733


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l1987_198783

theorem quadratic_equation_condition (m : ℝ) : 
  (abs m + 1 = 2 ∧ m + 1 ≠ 0) ↔ m = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l1987_198783


namespace NUMINAMATH_CALUDE_three_digit_numbers_with_eight_or_nine_l1987_198774

theorem three_digit_numbers_with_eight_or_nine (total_three_digit : ℕ) (without_eight_or_nine : ℕ) :
  total_three_digit = 900 →
  without_eight_or_nine = 448 →
  total_three_digit - without_eight_or_nine = 452 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_numbers_with_eight_or_nine_l1987_198774


namespace NUMINAMATH_CALUDE_complex_cube_simplification_l1987_198710

theorem complex_cube_simplification :
  let i : ℂ := Complex.I
  (1 + Real.sqrt 3 * i)^3 = -8 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_simplification_l1987_198710


namespace NUMINAMATH_CALUDE_curve_properties_l1987_198752

-- Define the curve
def curve (x y : ℝ) : Prop := abs x + y^2 - 3*y = 0

-- Theorem for the axis of symmetry and range of y
theorem curve_properties :
  (∀ x y : ℝ, curve x y ↔ curve (-x) y) ∧
  (∀ y : ℝ, (∃ x : ℝ, curve x y) → 0 ≤ y ∧ y ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_curve_properties_l1987_198752


namespace NUMINAMATH_CALUDE_initial_shirts_count_l1987_198794

/-- The number of shirts Haley returned -/
def returned_shirts : ℕ := 6

/-- The number of shirts Haley ended up with -/
def final_shirts : ℕ := 5

/-- The initial number of shirts Haley bought -/
def initial_shirts : ℕ := returned_shirts + final_shirts

/-- Theorem stating that the initial number of shirts is 11 -/
theorem initial_shirts_count : initial_shirts = 11 := by
  sorry

end NUMINAMATH_CALUDE_initial_shirts_count_l1987_198794


namespace NUMINAMATH_CALUDE_side_BC_equation_l1987_198751

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a line equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def altitude_from_AC : Line := { a := 2, b := -3, c := 1 }
def altitude_from_AB : Line := { a := 1, b := 1, c := -1 }

def vertex_A : ℝ × ℝ := (1, 2)

theorem side_BC_equation (t : Triangle) 
  (h1 : t.A = vertex_A)
  (h2 : altitude_from_AC.a * t.B.1 + altitude_from_AC.b * t.B.2 + altitude_from_AC.c = 0)
  (h3 : altitude_from_AC.a * t.C.1 + altitude_from_AC.b * t.C.2 + altitude_from_AC.c = 0)
  (h4 : altitude_from_AB.a * t.B.1 + altitude_from_AB.b * t.B.2 + altitude_from_AB.c = 0)
  (h5 : altitude_from_AB.a * t.C.1 + altitude_from_AB.b * t.C.2 + altitude_from_AB.c = 0) :
  ∃ (l : Line), l.a * t.B.1 + l.b * t.B.2 + l.c = 0 ∧
                l.a * t.C.1 + l.b * t.C.2 + l.c = 0 ∧
                l = { a := 2, b := 3, c := 7 } := by
  sorry

end NUMINAMATH_CALUDE_side_BC_equation_l1987_198751


namespace NUMINAMATH_CALUDE_percentage_difference_l1987_198744

theorem percentage_difference (x y : ℝ) (h : y = 1.8 * x) : 
  (x - y) / y * 100 = -(4 / 9) * 100 := by sorry

end NUMINAMATH_CALUDE_percentage_difference_l1987_198744


namespace NUMINAMATH_CALUDE_smallest_possible_b_l1987_198711

theorem smallest_possible_b (a b : ℝ) 
  (h1 : 2 < a ∧ a < b) 
  (h2 : 2 + a ≤ b) 
  (h3 : 1/a + 1/b ≤ 1/2) : 
  b ≥ 3 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_possible_b_l1987_198711


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_l1987_198716

/-- Given a triangle ABC where b = a * sin(C) and c = a * cos(B), prove that ABC is an isosceles right triangle -/
theorem isosceles_right_triangle (a b c : ℝ) (A B C : ℝ) 
  (h1 : b = a * Real.sin C) 
  (h2 : c = a * Real.cos B) 
  (h3 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h4 : 0 < A ∧ 0 < B ∧ 0 < C) 
  (h5 : A + B + C = π) : 
  A = π / 2 ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_l1987_198716


namespace NUMINAMATH_CALUDE_calculation_proof_l1987_198781

theorem calculation_proof : (1 / 6 : ℚ) * (-6 : ℚ) / (-1/6 : ℚ) * (6 : ℚ) = 36 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1987_198781


namespace NUMINAMATH_CALUDE_initial_blue_balls_l1987_198763

theorem initial_blue_balls (total : ℕ) (removed : ℕ) (prob : ℚ) : 
  total = 18 → removed = 3 → prob = 1/5 → 
  ∃ (initial_blue : ℕ), 
    initial_blue = 6 ∧ 
    (initial_blue - removed : ℚ) / (total - removed) = prob :=
by sorry

end NUMINAMATH_CALUDE_initial_blue_balls_l1987_198763


namespace NUMINAMATH_CALUDE_line_through_vectors_l1987_198799

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem line_through_vectors (a b : V) (k : ℝ) (h : a ≠ b) :
  (∃ t : ℝ, k • a + (2/3 : ℝ) • b = a + t • (b - a)) →
  k = 1/3 := by
sorry

end NUMINAMATH_CALUDE_line_through_vectors_l1987_198799


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1987_198741

/-- The radius of the inscribed circle of a triangle with side lengths 5, 12, and 13 is 2 -/
theorem inscribed_circle_radius (a b c r : ℝ) (h1 : a = 5) (h2 : b = 12) (h3 : c = 13) :
  r = (a + b - c) / 2 → r = 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1987_198741


namespace NUMINAMATH_CALUDE_min_distance_circle_line_l1987_198777

theorem min_distance_circle_line : 
  let circle := {p : ℝ × ℝ | (p.1 + 2)^2 + (p.2 - 1)^2 = 4}
  let line := {p : ℝ × ℝ | 4 * p.1 - p.2 - 1 = 0}
  ∃ (min_dist : ℝ), 
    (∀ (p q : ℝ × ℝ), p ∈ circle → q ∈ line → min_dist ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) ∧
    (∃ (p q : ℝ × ℝ), p ∈ circle ∧ q ∈ line ∧ min_dist = Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) ∧
    min_dist = (10 * Real.sqrt 17) / 17 - 2 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_circle_line_l1987_198777


namespace NUMINAMATH_CALUDE_quadratic_root_sum_squares_l1987_198789

theorem quadratic_root_sum_squares (h : ℝ) : 
  (∃ r s : ℝ, r^2 + 2*h*r + 2 = 0 ∧ s^2 + 2*h*s + 2 = 0 ∧ r^2 + s^2 = 8) → 
  |h| = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_squares_l1987_198789


namespace NUMINAMATH_CALUDE_x1_value_l1987_198709

theorem x1_value (x₁ x₂ x₃ : Real) 
  (h1 : 0 ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 1)
  (h2 : (1 - x₁^2) + (x₁ - x₂)^2 + (x₂ - x₃)^2 + x₃^2 = 1/4) :
  x₁ = Real.sqrt 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_x1_value_l1987_198709


namespace NUMINAMATH_CALUDE_venus_hall_rental_cost_prove_venus_hall_rental_cost_l1987_198745

/-- The rental cost of Venus Hall, given the conditions of the prom venue problem -/
theorem venus_hall_rental_cost : ℝ → Prop :=
  fun v =>
    let caesars_total : ℝ := 800 + 60 * 30
    let venus_total : ℝ := v + 60 * 35
    caesars_total = venus_total →
    v = 500

/-- Proof of the venus_hall_rental_cost theorem -/
theorem prove_venus_hall_rental_cost : ∃ v, venus_hall_rental_cost v :=
  sorry

end NUMINAMATH_CALUDE_venus_hall_rental_cost_prove_venus_hall_rental_cost_l1987_198745


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1987_198747

-- Define the arithmetic sequence a_n
def a (n : ℕ+) : ℚ :=
  sorry

-- Define the sum S_n of the first n terms of a_n
def S (n : ℕ+) : ℚ :=
  sorry

-- Define the sequence b_n
def b (n : ℕ+) : ℚ :=
  1 / (a n ^ 2 - 1)

-- Define the sum T_n of the first n terms of b_n
def T (n : ℕ+) : ℚ :=
  sorry

theorem arithmetic_sequence_properties :
  (a 3 = 6) ∧
  (a 5 + a 7 = 24) ∧
  (∀ n : ℕ+, a n = 2 * n) ∧
  (∀ n : ℕ+, S n = n^2 + n) ∧
  (∀ n : ℕ+, T n = n / (2 * n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1987_198747


namespace NUMINAMATH_CALUDE_polygon_exterior_angles_l1987_198772

theorem polygon_exterior_angles (n : ℕ) (exterior_angle : ℝ) : 
  (n > 2) → (exterior_angle = 36) → (n * exterior_angle = 360) → n = 10 :=
by sorry

end NUMINAMATH_CALUDE_polygon_exterior_angles_l1987_198772


namespace NUMINAMATH_CALUDE_age_difference_l1987_198776

theorem age_difference (jack_age bill_age : ℕ) : 
  jack_age = 2 * bill_age → 
  (jack_age + 8) = 3 * (bill_age + 8) → 
  jack_age - bill_age = 16 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l1987_198776


namespace NUMINAMATH_CALUDE_x_range_for_inequality_l1987_198770

theorem x_range_for_inequality (x : ℝ) :
  (∀ m ∈ Set.Icc (1/2 : ℝ) 3, x^2 + m*x + 4 > 2*m + 4*x) →
  x > 2 ∨ x < -1 :=
by sorry

end NUMINAMATH_CALUDE_x_range_for_inequality_l1987_198770


namespace NUMINAMATH_CALUDE_coronavirus_cases_day3_l1987_198767

/-- Represents the number of Coronavirus cases over three days -/
structure CoronavirusCases where
  initial_cases : ℕ
  day2_increase : ℕ
  day2_recoveries : ℕ
  day3_recoveries : ℕ
  final_total : ℕ

/-- Calculates the number of new cases on day 3 -/
def new_cases_day3 (c : CoronavirusCases) : ℕ :=
  c.final_total - (c.initial_cases + c.day2_increase - c.day2_recoveries - c.day3_recoveries)

/-- Theorem stating that given the conditions, the number of new cases on day 3 is 1500 -/
theorem coronavirus_cases_day3 (c : CoronavirusCases) 
  (h1 : c.initial_cases = 2000)
  (h2 : c.day2_increase = 500)
  (h3 : c.day2_recoveries = 50)
  (h4 : c.day3_recoveries = 200)
  (h5 : c.final_total = 3750) :
  new_cases_day3 c = 1500 := by
  sorry

end NUMINAMATH_CALUDE_coronavirus_cases_day3_l1987_198767


namespace NUMINAMATH_CALUDE_salary_change_l1987_198713

/-- Proves that when a salary is increased by 10% and then reduced by 10%, 
    the net change is a decrease of 1% of the original salary. -/
theorem salary_change (S : ℝ) : 
  (S + S * (10 / 100)) * (1 - 10 / 100) = S * 0.99 := by
  sorry

end NUMINAMATH_CALUDE_salary_change_l1987_198713
