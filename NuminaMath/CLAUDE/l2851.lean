import Mathlib

namespace right_rectangular_prism_volume_l2851_285190

/-- The volume of a right rectangular prism with face areas 6, 8, and 12 square inches is 24 cubic inches. -/
theorem right_rectangular_prism_volume (l w h : ℝ) 
  (area1 : l * w = 6)
  (area2 : w * h = 8)
  (area3 : l * h = 12) :
  l * w * h = 24 := by
  sorry

end right_rectangular_prism_volume_l2851_285190


namespace approximate_cost_price_of_toy_l2851_285142

/-- The cost price of a toy given the selling conditions --/
def cost_price_of_toy (num_toys : ℕ) (total_selling_price : ℚ) (gain_in_toys : ℕ) : ℚ :=
  let selling_price_per_toy := total_selling_price / num_toys
  let x := selling_price_per_toy * num_toys / (num_toys + gain_in_toys)
  x

/-- Theorem stating the approximate cost price of a toy under given conditions --/
theorem approximate_cost_price_of_toy :
  let calculated_price := cost_price_of_toy 18 27300 3
  ⌊calculated_price⌋ = 1300 := by sorry

end approximate_cost_price_of_toy_l2851_285142


namespace chord_longer_than_arc_l2851_285141

theorem chord_longer_than_arc (R : ℝ) (h : R > 0) :
  let angle := 60 * π / 180
  let arc_length := angle * R
  let new_radius := 1.05 * R
  let chord_length := 2 * new_radius * Real.sin (angle / 2)
  chord_length > arc_length := by sorry

end chord_longer_than_arc_l2851_285141


namespace abc_divisibility_theorem_l2851_285100

theorem abc_divisibility_theorem (a b c : ℕ+) 
  (h1 : a * b ∣ c * (c^2 - c + 1)) 
  (h2 : (c^2 + 1) ∣ (a + b)) :
  (a = c ∧ b = c^2 - c + 1) ∨ (a = c^2 - c + 1 ∧ b = c) := by
  sorry

end abc_divisibility_theorem_l2851_285100


namespace inequality_equivalence_l2851_285167

theorem inequality_equivalence (x : ℝ) : 
  (-1/3 ≤ (5-x)/2 ∧ (5-x)/2 < 1/3) ↔ (13/3 < x ∧ x ≤ 17/3) := by sorry

end inequality_equivalence_l2851_285167


namespace bhanu_house_rent_expenditure_l2851_285166

/-- Calculates Bhanu's expenditure on house rent given his spending pattern and petrol expenditure -/
theorem bhanu_house_rent_expenditure (income : ℝ) (petrol_expenditure : ℝ) :
  petrol_expenditure = 0.3 * income →
  0.3 * (0.7 * income) = 210 := by
  sorry

end bhanu_house_rent_expenditure_l2851_285166


namespace carters_additional_cakes_l2851_285158

/-- The number of additional cakes Carter bakes in a week when tripling his usual production. -/
theorem carters_additional_cakes 
  (cheesecakes muffins red_velvet : ℕ) 
  (h1 : cheesecakes = 6)
  (h2 : muffins = 5)
  (h3 : red_velvet = 8) :
  3 * (cheesecakes + muffins + red_velvet) - (cheesecakes + muffins + red_velvet) = 38 :=
by sorry


end carters_additional_cakes_l2851_285158


namespace prime_dates_february_2024_l2851_285180

/-- A natural number is prime if it's greater than 1 and has no positive divisors other than 1 and itself. -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- The number of days in February during a leap year. -/
def februaryDaysInLeapYear : ℕ := 29

/-- The month number for February. -/
def februaryMonth : ℕ := 2

/-- A prime date occurs when both the month and day are prime numbers. -/
def isPrimeDate (month day : ℕ) : Prop := isPrime month ∧ isPrime day

/-- The number of prime dates in February of a leap year. -/
def primeDatesInFebruaryLeapYear : ℕ := 10

/-- Theorem stating that the number of prime dates in February 2024 is 10. -/
theorem prime_dates_february_2024 :
  isPrime februaryMonth →
  (∀ d : ℕ, d ≤ februaryDaysInLeapYear → isPrimeDate februaryMonth d ↔ isPrime d) →
  (∃ dates : Finset ℕ, dates.card = primeDatesInFebruaryLeapYear ∧
    ∀ d ∈ dates, d ≤ februaryDaysInLeapYear ∧ isPrime d) :=
by sorry

end prime_dates_february_2024_l2851_285180


namespace train_passing_time_train_passing_time_proof_l2851_285136

/-- Calculates the time taken for two trains to pass each other --/
theorem train_passing_time (train_length : ℝ) (speed_fast : ℝ) (speed_slow : ℝ) : ℝ :=
  let speed_fast_ms := speed_fast * 1000 / 3600
  let speed_slow_ms := speed_slow * 1000 / 3600
  let relative_speed := speed_fast_ms + speed_slow_ms
  train_length / relative_speed

/-- Proves that the time taken for the slower train to pass the driver of the faster train is approximately 18 seconds --/
theorem train_passing_time_proof :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |train_passing_time 475 55 40 - 18| < ε :=
sorry

end train_passing_time_train_passing_time_proof_l2851_285136


namespace total_legs_in_javiers_household_l2851_285106

/-- The number of legs in Javier's household -/
def total_legs : ℕ :=
  let num_humans := 5 -- Javier, his wife, and 3 children
  let num_dogs := 2
  let num_cats := 1
  let legs_per_human := 2
  let legs_per_dog := 4
  let legs_per_cat := 4
  num_humans * legs_per_human + num_dogs * legs_per_dog + num_cats * legs_per_cat

theorem total_legs_in_javiers_household :
  total_legs = 22 := by sorry

end total_legs_in_javiers_household_l2851_285106


namespace range_of_m_l2851_285170

-- Define the sets A and B
def A : Set ℝ := {a | a < -1}
def B (m : ℝ) : Set ℝ := {x | 3*m < x ∧ x < m + 2}

-- Define the proposition P
def P (a : ℝ) : Prop := ∃ x : ℝ, a*x^2 + 2*x - 1 = 0

-- State the theorem
theorem range_of_m :
  (∀ a : ℝ, ¬(P a)) →
  (∀ m : ℝ, ∀ x : ℝ, x ∈ B m → x ∉ A) →
  {m : ℝ | -1/3 ≤ m} = {m : ℝ | ∃ x : ℝ, x ∈ B m} := by
  sorry

end range_of_m_l2851_285170


namespace specific_group_size_l2851_285157

/-- Represents a group of people with language skills -/
structure LanguageGroup where
  latin : ℕ     -- Number of people who can speak Latin
  french : ℕ    -- Number of people who can speak French
  neither : ℕ   -- Number of people who can't speak either Latin or French
  both : ℕ      -- Number of people who can speak both Latin and French

/-- Calculates the total number of people in the group -/
def totalPeople (group : LanguageGroup) : ℕ :=
  (group.latin + group.french - group.both) + group.neither

/-- Theorem: The specific group has 25 people -/
theorem specific_group_size :
  let group : LanguageGroup := {
    latin := 13,
    french := 15,
    neither := 6,
    both := 9
  }
  totalPeople group = 25 := by sorry

end specific_group_size_l2851_285157


namespace cistern_emptying_time_l2851_285176

theorem cistern_emptying_time (fill_time : ℝ) (combined_time : ℝ) : 
  fill_time = 7 → combined_time = 31.5 → 
  (fill_time⁻¹ - (fill_time⁻¹ - combined_time⁻¹)⁻¹) = 9⁻¹ :=
by
  sorry

end cistern_emptying_time_l2851_285176


namespace quadratic_max_l2851_285164

theorem quadratic_max (x : ℝ) : 
  ∃ (m : ℝ), (∀ y : ℝ, -y^2 - 8*y + 16 ≤ m) ∧ (-x^2 - 8*x + 16 = m) → x = -4 :=
by sorry

end quadratic_max_l2851_285164


namespace surface_area_unchanged_l2851_285145

/-- Represents a rectangular solid with length, width, and height -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (solid : RectangularSolid) : ℝ :=
  2 * (solid.length * solid.width + solid.length * solid.height + solid.width * solid.height)

/-- Represents the removal of a smaller prism from a larger solid -/
structure PrismRemoval where
  original : RectangularSolid
  removed : RectangularSolid
  flushFaces : ℕ

/-- Theorem stating that the surface area remains unchanged after removal -/
theorem surface_area_unchanged (removal : PrismRemoval) :
  removal.original = RectangularSolid.mk 4 3 2 →
  removal.removed = RectangularSolid.mk 1 1 2 →
  removal.flushFaces = 2 →
  surfaceArea removal.original = surfaceArea removal.original - surfaceArea removal.removed + 2 * removal.removed.length * removal.removed.width :=
by sorry

end surface_area_unchanged_l2851_285145


namespace no_polynomial_transform_l2851_285132

theorem no_polynomial_transform : ¬∃ (P : ℝ → ℝ), 
  (∀ x : ℝ, ∃ (a b c d : ℝ), P x = a * x^3 + b * x^2 + c * x + d) ∧
  P (-3) = -3 ∧ P (-1) = -1 ∧ P 1 = -3 ∧ P 3 = 3 := by
  sorry

end no_polynomial_transform_l2851_285132


namespace part1_selection_count_part2_selection_count_l2851_285144

def num_male : ℕ := 4
def num_female : ℕ := 5
def total_selected : ℕ := 4

def combinations (n k : ℕ) : ℕ := 
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem part1_selection_count : 
  combinations num_male 2 * combinations num_female 2 = 60 := by sorry

theorem part2_selection_count :
  let total_selections := combinations num_male 1 * combinations num_female 3 +
                          combinations num_male 2 * combinations num_female 2 +
                          combinations num_male 3 * combinations num_female 1
  let invalid_selections := combinations (num_male - 1) 2 +
                            combinations (num_female - 1) 1 * combinations (num_male - 1) 1 +
                            combinations (num_female - 1) 2
  total_selections - invalid_selections = 99 := by sorry

end part1_selection_count_part2_selection_count_l2851_285144


namespace turquoise_tile_cost_l2851_285160

/-- Proves that the cost of each turquoise tile is $13 given the problem conditions -/
theorem turquoise_tile_cost :
  ∀ (total_area : ℝ) (tiles_per_sqft : ℝ) (purple_cost : ℝ) (savings : ℝ),
    total_area = 96 →
    tiles_per_sqft = 4 →
    purple_cost = 11 →
    savings = 768 →
    ∃ (turquoise_cost : ℝ),
      turquoise_cost = 13 ∧
      (total_area * tiles_per_sqft) * turquoise_cost - (total_area * tiles_per_sqft) * purple_cost = savings :=
by
  sorry


end turquoise_tile_cost_l2851_285160


namespace rowing_speed_problem_l2851_285102

/-- The rowing speed problem -/
theorem rowing_speed_problem (v c : ℝ) (h1 : c = 1.1)
  (h2 : (v + c) * t = (v - c) * (2 * t) → t ≠ 0) : v = 3.3 := by
  sorry

end rowing_speed_problem_l2851_285102


namespace graduating_class_size_l2851_285179

theorem graduating_class_size (boys : ℕ) (girls : ℕ) (h1 : boys = 127) (h2 : girls = boys + 212) :
  boys + girls = 466 := by
  sorry

end graduating_class_size_l2851_285179


namespace negation_of_universal_proposition_l2851_285103

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 0 → x^2 + 4*x + 3 > 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^2 + 4*x + 3 ≤ 0) :=
by sorry

end negation_of_universal_proposition_l2851_285103


namespace circle_diameter_endpoint_l2851_285104

/-- Given a circle with center (3.5, -2) and one endpoint of a diameter at (1, -6),
    prove that the other endpoint of the diameter is at (6, 2). -/
theorem circle_diameter_endpoint (center : ℝ × ℝ) (endpoint1 : ℝ × ℝ) (endpoint2 : ℝ × ℝ) : 
  center = (3.5, -2) →
  endpoint1 = (1, -6) →
  endpoint2 = (6, 2) →
  (center.1 - endpoint1.1 = endpoint2.1 - center.1) ∧
  (center.2 - endpoint1.2 = endpoint2.2 - center.2) := by
  sorry


end circle_diameter_endpoint_l2851_285104


namespace no_valid_partition_l2851_285195

theorem no_valid_partition : ¬∃ (A B C : Set ℕ), 
  (A ≠ ∅) ∧ (B ≠ ∅) ∧ (C ≠ ∅) ∧
  (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (C ∩ A = ∅) ∧
  (A ∪ B ∪ C = Set.univ) ∧
  (∀ a b, a ∈ A → b ∈ B → a + b + 1 ∈ C) ∧
  (∀ b c, b ∈ B → c ∈ C → b + c + 1 ∈ A) ∧
  (∀ c a, c ∈ C → a ∈ A → c + a + 1 ∈ B) :=
by sorry

end no_valid_partition_l2851_285195


namespace last_group_markers_theorem_l2851_285183

/-- Calculates the number of markers each student in the last group receives --/
def markers_per_last_student (total_students : ℕ) (marker_boxes : ℕ) (markers_per_box : ℕ)
  (first_group_students : ℕ) (first_group_markers_per_student : ℕ)
  (second_group_students : ℕ) (second_group_markers_per_student : ℕ) : ℕ :=
  let total_markers := marker_boxes * markers_per_box
  let first_group_markers := first_group_students * first_group_markers_per_student
  let second_group_markers := second_group_students * second_group_markers_per_student
  let remaining_markers := total_markers - first_group_markers - second_group_markers
  let last_group_students := total_students - first_group_students - second_group_students
  remaining_markers / last_group_students

/-- Theorem stating that under the given conditions, each student in the last group receives 6 markers --/
theorem last_group_markers_theorem :
  markers_per_last_student 30 22 5 10 2 15 4 = 6 := by
  sorry

end last_group_markers_theorem_l2851_285183


namespace maddie_thursday_viewing_l2851_285178

/-- Represents the viewing schedule for a TV show --/
structure ViewingSchedule where
  totalEpisodes : ℕ
  episodeLength : ℕ
  mondayMinutes : ℕ
  fridayEpisodes : ℕ
  weekendMinutes : ℕ

/-- Calculates the number of minutes watched on Thursday --/
def thursdayMinutes (schedule : ViewingSchedule) : ℕ :=
  schedule.totalEpisodes * schedule.episodeLength -
  (schedule.mondayMinutes + schedule.fridayEpisodes * schedule.episodeLength + schedule.weekendMinutes)

/-- Theorem stating that Maddie watched 21 minutes on Thursday --/
theorem maddie_thursday_viewing : 
  let schedule : ViewingSchedule := {
    totalEpisodes := 8,
    episodeLength := 44,
    mondayMinutes := 138,
    fridayEpisodes := 2,
    weekendMinutes := 105
  }
  thursdayMinutes schedule = 21 := by
  sorry

end maddie_thursday_viewing_l2851_285178


namespace sequence_property_l2851_285113

theorem sequence_property (a : ℕ → ℕ) 
  (h1 : ∀ n, 1 < a n)
  (h2 : ∀ n, a n < a (n + 1))
  (h3 : ∀ n, a (n + a n) = 2 * a n) :
  ∃ c : ℕ, ∀ n, a n = n + c := by
sorry

end sequence_property_l2851_285113


namespace integral_equation_solution_l2851_285156

theorem integral_equation_solution (k : ℝ) : (∫ x in (0:ℝ)..1, (3 * x^2 + k)) = 10 ↔ k = 9 := by
  sorry

end integral_equation_solution_l2851_285156


namespace sin_two_alpha_value_l2851_285110

theorem sin_two_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π/2) π) 
  (h2 : 2 * Real.cos (2*α) = Real.sin (π/4 - α)) : 
  Real.sin (2*α) = -7/8 := by
  sorry

end sin_two_alpha_value_l2851_285110


namespace symmetric_point_x_axis_l2851_285161

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines symmetry with respect to the x-axis -/
def symmetricXAxis (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- Theorem: The point symmetric to P(1, -2) with respect to the x-axis is (1, 2) -/
theorem symmetric_point_x_axis :
  let P : Point := { x := 1, y := -2 }
  symmetricXAxis P = { x := 1, y := 2 } := by
  sorry

end symmetric_point_x_axis_l2851_285161


namespace pizza_toppings_l2851_285120

theorem pizza_toppings (total_slices cheese_slices olive_slices : ℕ) 
  (h1 : total_slices = 24)
  (h2 : cheese_slices = 16)
  (h3 : olive_slices = 18)
  (h4 : ∀ slice, slice ≤ total_slices → (slice ≤ cheese_slices ∨ slice ≤ olive_slices)) :
  ∃ both_toppings : ℕ, 
    both_toppings = 10 ∧ 
    cheese_slices + olive_slices - both_toppings = total_slices :=
by
  sorry

end pizza_toppings_l2851_285120


namespace line_equation_correct_l2851_285111

-- Define the line passing through two points
def line_through_points (x1 y1 x2 y2 : ℝ) (x y : ℝ) : Prop :=
  (y - y1) * (x2 - x1) = (x - x1) * (y2 - y1)

-- Define the equation of the line we want to prove
def line_equation (x y : ℝ) : Prop :=
  -x + y - 2 = 0

-- Theorem statement
theorem line_equation_correct :
  ∀ (x y : ℝ), line_through_points 3 2 1 4 x y ↔ line_equation x y :=
by sorry

end line_equation_correct_l2851_285111


namespace trigonometric_expression_equals_one_f_at_specific_angle_l2851_285191

-- Problem 1
theorem trigonometric_expression_equals_one :
  Real.sin (-120 * Real.pi / 180) * Real.cos (210 * Real.pi / 180) +
  Real.cos (-300 * Real.pi / 180) * Real.sin (-330 * Real.pi / 180) = 1 := by
sorry

-- Problem 2
noncomputable def f (α : Real) : Real :=
  (2 * Real.sin (Real.pi + α) * Real.cos (Real.pi - α) - Real.cos (Real.pi + α)) /
  (1 + Real.sin α ^ 2 + Real.cos ((3 * Real.pi) / 2 + α) - Real.sin ((Real.pi / 2 + α) ^ 2))

theorem f_at_specific_angle (h : 1 + 2 * Real.sin (-23 * Real.pi / 6) ≠ 0) :
  f (-23 * Real.pi / 6) = Real.sqrt 3 := by
sorry

end trigonometric_expression_equals_one_f_at_specific_angle_l2851_285191


namespace shower_tiles_width_l2851_285153

/-- Given a 3-walled shower with 20 tiles running the height of each wall and 480 tiles in total,
    the number of tiles running the width of each wall is 8. -/
theorem shower_tiles_width (num_walls : Nat) (height_tiles : Nat) (total_tiles : Nat) :
  num_walls = 3 → height_tiles = 20 → total_tiles = 480 →
  ∃ width_tiles : Nat, width_tiles = 8 ∧ num_walls * height_tiles * width_tiles = total_tiles :=
by sorry

end shower_tiles_width_l2851_285153


namespace quadratic_inequality_solution_l2851_285197

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, x ∈ Set.Ioo 1 3 ↔ a * x^2 + b * x + 3 < 0) →
  a + b = -3 :=
by sorry

end quadratic_inequality_solution_l2851_285197


namespace range_of_f_l2851_285175

-- Define the function f
def f (x : ℝ) : ℝ := 3 - x

-- State the theorem
theorem range_of_f :
  {y : ℝ | ∃ x ≤ 1, f x = y} = {y : ℝ | y ≥ 2} := by sorry

end range_of_f_l2851_285175


namespace other_number_proof_l2851_285148

theorem other_number_proof (a b : ℕ+) : 
  Nat.lcm a b = 2520 →
  Nat.gcd a b = 12 →
  a = 240 →
  b = 126 := by
sorry

end other_number_proof_l2851_285148


namespace cake_mix_buyers_cake_mix_buyers_is_50_l2851_285140

theorem cake_mix_buyers (total_buyers : ℕ) (muffin_buyers : ℕ) (both_buyers : ℕ) 
  (neither_prob : ℚ) (h1 : total_buyers = 100) (h2 : muffin_buyers = 40) 
  (h3 : both_buyers = 15) (h4 : neither_prob = 1/4) : ℕ :=
by
  -- The number of buyers who purchase cake mix
  sorry

#check cake_mix_buyers

-- The theorem statement proves that given the conditions,
-- the number of buyers who purchase cake mix is 50
theorem cake_mix_buyers_is_50 : 
  cake_mix_buyers 100 40 15 (1/4) rfl rfl rfl rfl = 50 :=
by
  sorry

end cake_mix_buyers_cake_mix_buyers_is_50_l2851_285140


namespace matrix_equation_implies_even_dimension_l2851_285194

theorem matrix_equation_implies_even_dimension (n : ℕ+) :
  (∃ (A B : Matrix (Fin n) (Fin n) ℝ), 
    Matrix.det A ≠ 0 ∧ 
    Matrix.det B ≠ 0 ∧ 
    A * B - B * A = B ^ 2 * A) → 
  Even n := by
sorry

end matrix_equation_implies_even_dimension_l2851_285194


namespace problem_solution_l2851_285105

theorem problem_solution : (150 * (150 - 4)) / (150 * 150 * 2 - 4) = 21900 / 44996 := by
  sorry

end problem_solution_l2851_285105


namespace sin_thirteen_pi_fourth_l2851_285174

theorem sin_thirteen_pi_fourth : Real.sin (13 * π / 4) = -Real.sqrt 2 / 2 := by
  sorry

end sin_thirteen_pi_fourth_l2851_285174


namespace jelly_bean_probability_l2851_285169

theorem jelly_bean_probability (p_red p_orange p_green p_yellow : ℝ) :
  p_red = 0.15 →
  p_orange = 0.35 →
  p_green = 0.25 →
  p_red + p_orange + p_green + p_yellow = 1 →
  p_yellow = 0.25 := by
sorry

end jelly_bean_probability_l2851_285169


namespace function_property_l2851_285196

def evenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def decreasingOnNegative (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ ∧ x₂ ≤ 0 → (f x₂ - f x₁) / (x₂ - x₁) < 0

theorem function_property (f : ℝ → ℝ) (heven : evenFunction f) (hdec : decreasingOnNegative f) :
  (∀ a : ℝ, f (1 - a) > f (2 * a - 1) ↔ 0 < a ∧ a < 2/3) :=
sorry

end function_property_l2851_285196


namespace complex_roots_theorem_l2851_285107

theorem complex_roots_theorem (a b c : ℂ) : 
  a + b + c = 1 ∧ 
  a * b + a * c + b * c = 1 ∧ 
  a * b * c = -1 → 
  ({a, b, c} : Set ℂ) = {1, Complex.I, -Complex.I} :=
sorry

end complex_roots_theorem_l2851_285107


namespace lcm_problem_l2851_285181

theorem lcm_problem (a b c : ℕ) (h1 : Nat.lcm a b = 60) (h2 : Nat.lcm a c = 270) :
  Nat.lcm b c = 540 := by
  sorry

end lcm_problem_l2851_285181


namespace inequality_no_solution_l2851_285152

theorem inequality_no_solution : {x : ℝ | x * (2 - x) > 3} = ∅ := by sorry

end inequality_no_solution_l2851_285152


namespace new_barbell_total_cost_l2851_285117

def old_barbell_cost : ℝ := 250

def new_barbell_cost_increase_percentage : ℝ := 0.3

def sales_tax_percentage : ℝ := 0.1

def new_barbell_cost_before_tax : ℝ := old_barbell_cost * (1 + new_barbell_cost_increase_percentage)

def sales_tax : ℝ := new_barbell_cost_before_tax * sales_tax_percentage

def total_cost : ℝ := new_barbell_cost_before_tax + sales_tax

theorem new_barbell_total_cost : total_cost = 357.50 := by
  sorry

end new_barbell_total_cost_l2851_285117


namespace sqrt_50_plus_fraction_minus_sqrt_half_plus_power_eq_5_sqrt_2_l2851_285184

theorem sqrt_50_plus_fraction_minus_sqrt_half_plus_power_eq_5_sqrt_2 :
  Real.sqrt 50 + 2 / (Real.sqrt 2 + 1) - 4 * Real.sqrt (1/2) + 2 * (Real.sqrt 2 - 1)^0 = 5 * Real.sqrt 2 := by
  sorry

end sqrt_50_plus_fraction_minus_sqrt_half_plus_power_eq_5_sqrt_2_l2851_285184


namespace z_is_negative_intercept_l2851_285135

/-- Represents a linear equation in two variables -/
structure LinearEquation where
  slope : ℝ
  intercept : ℝ

/-- Converts an objective function z = ax - y to a linear equation y = ax - z -/
def objectiveFunctionToLinearEquation (a : ℝ) (z : ℝ) : LinearEquation :=
  { slope := a, intercept := -z }

/-- Theorem: In the equation z = 3x - y, z represents the negative of the vertical intercept -/
theorem z_is_negative_intercept (z : ℝ) :
  let eq := objectiveFunctionToLinearEquation 3 z
  eq.intercept = -z := by sorry

end z_is_negative_intercept_l2851_285135


namespace zeros_of_quadratic_l2851_285126

/-- The quadratic function f(x) = x^2 - 2x - 3 -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

/-- Theorem stating that -1 and 3 are the zeros of the quadratic function f -/
theorem zeros_of_quadratic :
  (f (-1) = 0) ∧ (f 3 = 0) ∧ (∀ x : ℝ, f x = 0 → x = -1 ∨ x = 3) :=
by sorry

end zeros_of_quadratic_l2851_285126


namespace expression_simplification_l2851_285168

theorem expression_simplification (x y m n : ℝ) : 
  (2 * x^2 * y - 3 * x * y + 2 - x^2 * y + 3 * x * y = x^2 * y + 2) ∧
  (9 * m^2 - 4 * (2 * m^2 - 3 * m * n + n^2) + 4 * n^2 = m^2 + 12 * m * n) :=
by sorry

end expression_simplification_l2851_285168


namespace cube_sum_equals_sum_l2851_285146

theorem cube_sum_equals_sum (a b : ℝ) : 
  (a / (1 + b) + b / (1 + a) = 1) → a^3 + b^3 = a + b := by
  sorry

end cube_sum_equals_sum_l2851_285146


namespace three_eighths_divided_by_one_fourth_l2851_285151

theorem three_eighths_divided_by_one_fourth : (3 : ℚ) / 8 / ((1 : ℚ) / 4) = (3 : ℚ) / 2 := by
  sorry

end three_eighths_divided_by_one_fourth_l2851_285151


namespace line_intersects_circle_intersection_point_polar_coordinates_l2851_285119

-- Define the line l
def line_l (x y : ℝ) : Prop := y - 1 = 2 * (x + 1)

-- Define the circle C₁
def circle_C₁ (x y : ℝ) : Prop := (x - 2)^2 + (y - 4)^2 = 4

-- Define the curve C₂
def curve_C₂ (x y : ℝ) : Prop := x^2 + y^2 = 4*x

-- Theorem 1: Line l intersects circle C₁
theorem line_intersects_circle : ∃ (x y : ℝ), line_l x y ∧ circle_C₁ x y := by sorry

-- Theorem 2: The intersection point of C₁ and C₂ is (2, 2) in Cartesian coordinates
theorem intersection_point : ∃! (x y : ℝ), circle_C₁ x y ∧ curve_C₂ x y ∧ x = 2 ∧ y = 2 := by sorry

-- Theorem 3: The polar coordinates of the intersection point are (2√2, π/4)
theorem polar_coordinates : 
  let (x, y) := (2, 2)
  let ρ := Real.sqrt (x^2 + y^2)
  let θ := Real.arctan (y / x)
  ρ = 2 * Real.sqrt 2 ∧ θ = π / 4 := by sorry

end line_intersects_circle_intersection_point_polar_coordinates_l2851_285119


namespace power_inequality_l2851_285192

theorem power_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^5 + y^5 - (x^4*y + x*y^4) ≥ 0 := by
  sorry

end power_inequality_l2851_285192


namespace quadratic_two_distinct_roots_l2851_285154

theorem quadratic_two_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) := by
sorry

end quadratic_two_distinct_roots_l2851_285154


namespace rd_cost_productivity_relation_l2851_285187

/-- The R&D costs required to increase the average labor productivity by 1 million rubles per person -/
def rd_cost_per_unit_productivity : ℝ := 4576

/-- The current R&D costs in million rubles -/
def current_rd_cost : ℝ := 3157.61

/-- The change in average labor productivity in million rubles per person -/
def delta_productivity : ℝ := 0.69

/-- Theorem stating that the R&D costs required to increase the average labor productivity
    by 1 million rubles per person is equal to the ratio of current R&D costs to the change
    in average labor productivity -/
theorem rd_cost_productivity_relation :
  rd_cost_per_unit_productivity = current_rd_cost / delta_productivity := by
  sorry

end rd_cost_productivity_relation_l2851_285187


namespace probability_theorem_l2851_285159

/-- Represents the number of white balls in the bag -/
def white_balls : ℕ := 2

/-- Represents the number of black balls in the bag -/
def black_balls : ℕ := 3

/-- Represents the total number of balls in the bag -/
def total_balls : ℕ := white_balls + black_balls

/-- The probability of drawing two balls of different colors without replacement -/
def prob_different_without_replacement : ℚ :=
  (white_balls * black_balls : ℚ) / (total_balls * (total_balls - 1) / 2 : ℚ)

/-- The probability of drawing two balls of different colors with replacement -/
def prob_different_with_replacement : ℚ :=
  2 * (white_balls : ℚ) / total_balls * (black_balls : ℚ) / total_balls

theorem probability_theorem :
  prob_different_without_replacement = 3/5 ∧
  prob_different_with_replacement = 12/25 := by
  sorry

end probability_theorem_l2851_285159


namespace tan_theta_in_terms_of_x_l2851_285137

theorem tan_theta_in_terms_of_x (θ x : ℝ) (h1 : 0 < θ ∧ θ < π/2) 
  (h2 : Real.cos (θ/2) = Real.sqrt ((x - 2)/(2*x))) : 
  Real.tan θ = -1/2 * Real.sqrt (x^2 - 4) := by
  sorry

end tan_theta_in_terms_of_x_l2851_285137


namespace teacher_distribution_count_l2851_285139

def distribute_teachers (n : ℕ) (k : ℕ) (min_a : ℕ) (min_others : ℕ) : ℕ :=
  -- n: total number of teachers
  -- k: number of schools
  -- min_a: minimum number of teachers for school A
  -- min_others: minimum number of teachers for other schools
  sorry

theorem teacher_distribution_count :
  distribute_teachers 6 4 2 1 = 660 := by sorry

end teacher_distribution_count_l2851_285139


namespace no_solution_equation_l2851_285114

theorem no_solution_equation (x y : ℝ) (hx : x ≠ 0) (hxy : x + y ≠ 0) :
  (x + y) / (2 * x) ≠ y / (x + y) :=
by sorry

end no_solution_equation_l2851_285114


namespace complement_intersection_theorem_l2851_285101

-- Define the sets
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x < 3}
def N : Set ℝ := {y | y > 2}

-- State the theorem
theorem complement_intersection_theorem :
  (Set.compl M) ∩ N = {x : ℝ | x ≥ 3} := by sorry

end complement_intersection_theorem_l2851_285101


namespace older_rabbit_catch_up_steps_l2851_285186

/-- Represents the rabbits in the race -/
inductive Rabbit
| Younger
| Older

/-- Properties of the rabbit race -/
structure RaceProperties where
  initial_lead : ℕ
  younger_steps_per_time : ℕ
  older_steps_per_time : ℕ
  younger_distance_steps : ℕ
  older_distance_steps : ℕ
  younger_distance : ℕ
  older_distance : ℕ

/-- The race between the two rabbits -/
def rabbit_race (props : RaceProperties) : Prop :=
  props.initial_lead = 10 ∧
  props.younger_steps_per_time = 4 ∧
  props.older_steps_per_time = 3 ∧
  props.younger_distance_steps = 7 ∧
  props.older_distance_steps = 5 ∧
  props.younger_distance = props.older_distance

/-- Theorem stating the number of steps for the older rabbit to catch up -/
theorem older_rabbit_catch_up_steps (props : RaceProperties) 
  (h : rabbit_race props) : ∃ (steps : ℕ), steps = 150 := by
  sorry


end older_rabbit_catch_up_steps_l2851_285186


namespace ryan_recruitment_count_l2851_285177

def total_funding_required : ℕ := 1000
def ryan_initial_funds : ℕ := 200
def average_funding_per_person : ℕ := 10

theorem ryan_recruitment_count :
  (total_funding_required - ryan_initial_funds) / average_funding_per_person = 80 := by
  sorry

end ryan_recruitment_count_l2851_285177


namespace largest_four_digit_divisible_by_8_l2851_285171

theorem largest_four_digit_divisible_by_8 : 
  ∀ n : ℕ, n ≤ 9999 → n ≥ 1000 → n % 8 = 0 → n ≤ 9992 :=
by sorry

end largest_four_digit_divisible_by_8_l2851_285171


namespace concert_songs_count_l2851_285147

/-- Represents the number of songs sung by each girl -/
structure SongCount where
  mary : ℕ
  alina : ℕ
  tina : ℕ
  hanna : ℕ

/-- Calculates the total number of songs sung by the trios -/
def total_songs (sc : SongCount) : ℕ :=
  (sc.mary + sc.alina + sc.tina + sc.hanna) / 3

/-- The theorem to be proved -/
theorem concert_songs_count :
  ∀ (sc : SongCount),
    sc.mary = 3 →
    sc.alina = 5 →
    sc.hanna = 6 →
    sc.mary < sc.tina →
    sc.tina < sc.hanna →
    total_songs sc = 6 := by
  sorry


end concert_songs_count_l2851_285147


namespace range_of_m_l2851_285165

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : 4/x + 1/y = 1) (h2 : ∀ x y, x > 0 → y > 0 → 4/x + 1/y = 1 → x + y ≥ m^2 + m + 3) :
  -3 ≤ m ∧ m ≤ 2 := by
sorry

end range_of_m_l2851_285165


namespace car_distance_proof_l2851_285115

/-- Proves that the distance covered by a car is 450 km given the specified conditions -/
theorem car_distance_proof (initial_time : ℝ) (speed : ℝ) : 
  initial_time = 6 →
  speed = 50 →
  (3/2 : ℝ) * initial_time * speed = 450 := by
  sorry

end car_distance_proof_l2851_285115


namespace upstream_downstream_time_ratio_l2851_285185

/-- 
Given a boat with speed 48 kmph in still water and a stream with speed 16 kmph,
prove that the ratio of time taken to row upstream to the time taken to row downstream is 2:1.
-/
theorem upstream_downstream_time_ratio 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : boat_speed = 48) 
  (h2 : stream_speed = 16) : 
  (boat_speed - stream_speed) / (boat_speed + stream_speed) = 1 / 2 := by
sorry

end upstream_downstream_time_ratio_l2851_285185


namespace equation_solution_l2851_285131

theorem equation_solution : ∃ x : ℝ, 10.0003 * x = 10000.3 ∧ x = 1000 := by
  sorry

end equation_solution_l2851_285131


namespace sum_reciprocals_bound_l2851_285155

theorem sum_reciprocals_bound (a b c : ℕ) (h : 1 / a + 1 / b + 1 / c < 1) :
  1 / a + 1 / b + 1 / c ≤ 41 / 42 := by
  sorry

end sum_reciprocals_bound_l2851_285155


namespace white_tiger_number_count_l2851_285189

/-- A function that returns true if a number is a multiple of 6 -/
def isMultipleOf6 (n : ℕ) : Bool :=
  n % 6 = 0

/-- A function that calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- A function that returns true if a number is a "White Tiger number" -/
def isWhiteTigerNumber (n : ℕ) : Bool :=
  isMultipleOf6 n ∧ sumOfDigits n = 6

/-- The count of "White Tiger numbers" up to 2022 -/
def whiteTigerNumberCount : ℕ :=
  (List.range 2023).filter isWhiteTigerNumber |>.length

theorem white_tiger_number_count : whiteTigerNumberCount = 30 := by
  sorry

end white_tiger_number_count_l2851_285189


namespace rectangle_area_l2851_285143

/-- Proves that a rectangle with perimeter 126 and difference between sides 37 has an area of 650 -/
theorem rectangle_area (l w : ℝ) : 
  (2 * (l + w) = 126) → 
  (l - w = 37) → 
  (l * w = 650) := by
sorry

end rectangle_area_l2851_285143


namespace f_value_at_107_5_l2851_285129

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_value_at_107_5 (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_periodic : ∀ x, f (x + 3) = -1 / f x)
  (h_interval : ∀ x ∈ Set.Icc (-3) (-2), f x = 4 * x) :
  f 107.5 = 1/10 := by
  sorry

end f_value_at_107_5_l2851_285129


namespace square_sum_ge_product_sum_abs_diff_product_gt_abs_diff_l2851_285118

-- Theorem 1
theorem square_sum_ge_product_sum (a b : ℝ) : a^2 + b^2 ≥ a*b + a + b - 1 := by
  sorry

-- Theorem 2
theorem abs_diff_product_gt_abs_diff (a b : ℝ) (ha : |a| < 1) (hb : |b| < 1) :
  |1 - a*b| > |a - b| := by
  sorry

end square_sum_ge_product_sum_abs_diff_product_gt_abs_diff_l2851_285118


namespace cylinder_height_relationship_l2851_285198

/-- Given two right circular cylinders with radii r₁ and r₂, and heights h₁ and h₂,
    prove that if the volume of the second is twice the first and r₂ = 1.1 * r₁,
    then h₂ ≈ 1.65 * h₁ -/
theorem cylinder_height_relationship (r₁ r₂ h₁ h₂ : ℝ) 
  (volume_relation : π * r₂^2 * h₂ = 2 * π * r₁^2 * h₁)
  (radius_relation : r₂ = 1.1 * r₁)
  (h₁_pos : h₁ > 0) (r₁_pos : r₁ > 0) :
  ∃ ε > 0, abs (h₂ / h₁ - 200 / 121) < ε :=
by sorry

end cylinder_height_relationship_l2851_285198


namespace largest_factorial_as_product_of_four_consecutive_l2851_285134

/-- Predicate that checks if a number is expressible as the product of 4 consecutive integers -/
def is_product_of_four_consecutive (n : ℕ) : Prop :=
  ∃ x : ℕ, n = x * (x + 1) * (x + 2) * (x + 3)

/-- Theorem stating that 6 is the largest integer n such that n! can be expressed as the product of 4 consecutive integers -/
theorem largest_factorial_as_product_of_four_consecutive :
  (6 : ℕ).factorial = 6 * 7 * 8 * 9 ∧
  ∀ n : ℕ, n > 6 → ¬(is_product_of_four_consecutive n.factorial) :=
sorry

end largest_factorial_as_product_of_four_consecutive_l2851_285134


namespace abs_inequality_solution_l2851_285150

theorem abs_inequality_solution (x : ℝ) : |x + 3| > x + 3 ↔ x < -3 := by sorry

end abs_inequality_solution_l2851_285150


namespace indefinite_integral_proof_l2851_285138

open Real

theorem indefinite_integral_proof (x : ℝ) (C : ℝ) (h : x ≠ -2 ∧ x ≠ -1) :
  deriv (λ y => 2 * log (abs (y + 2)) - 1 / (2 * (y + 1)^2) + C) x =
  (2 * x^3 + 6 * x^2 + 7 * x + 4) / ((x + 2) * (x + 1)^3) := by
  sorry

end indefinite_integral_proof_l2851_285138


namespace rectangle_area_increase_l2851_285193

theorem rectangle_area_increase 
  (l w : ℝ) 
  (hl : l > 0) 
  (hw : w > 0) : 
  let new_length := 1.3 * l
  let new_width := 1.15 * w
  let original_area := l * w
  let new_area := new_length * new_width
  (new_area - original_area) / original_area = 0.495 := by
sorry

end rectangle_area_increase_l2851_285193


namespace find_other_number_l2851_285173

theorem find_other_number (x y : ℤ) : 
  3 * x + 4 * y = 161 → (x = 17 ∨ y = 17) → (x = 31 ∨ y = 31) := by
  sorry

end find_other_number_l2851_285173


namespace row_swap_matrix_l2851_285127

theorem row_swap_matrix : ∃ (N : Matrix (Fin 2) (Fin 2) ℝ),
  ∀ (A : Matrix (Fin 2) (Fin 2) ℝ),
    N * A = ![![A 1 0, A 1 1], ![A 0 0, A 0 1]] := by
  sorry

end row_swap_matrix_l2851_285127


namespace zero_mxn_table_l2851_285122

/-- Represents a move on the table -/
inductive Move
  | Row (i : Nat)
  | Column (j : Nat)
  | Diagonal (d : Int)

/-- Represents the state of the table -/
def Table (m n : Nat) := Fin m → Fin n → Int

/-- Applies a move to the table -/
def applyMove (t : Table m n) (move : Move) (delta : Int) : Table m n :=
  sorry

/-- Checks if all elements in the table are zero -/
def allZero (t : Table m n) : Prop :=
  sorry

/-- Checks if we can change all numbers to zero in a 3x3 table -/
def canZero3x3 : Prop :=
  ∀ (t : Table 3 3), ∃ (moves : List (Move × Int)), 
    allZero (moves.foldl (fun acc (m, d) => applyMove acc m d) t)

/-- Main theorem: If we can zero any 3x3 table, we can zero any mxn table -/
theorem zero_mxn_table (m n : Nat) (h : canZero3x3) : 
  ∀ (t : Table m n), ∃ (moves : List (Move × Int)), 
    allZero (moves.foldl (fun acc (m, d) => applyMove acc m d) t) :=
  sorry

end zero_mxn_table_l2851_285122


namespace arithmetic_sequence_sum_l2851_285128

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

-- Define the sum of the first n terms
def S (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a →
  a 2 * a 4 * a 6 * a 8 = 120 →
  1 / (a 4 * a 6 * a 8) + 1 / (a 2 * a 6 * a 8) + 1 / (a 2 * a 4 * a 8) + 1 / (a 2 * a 4 * a 6) = 7 / 60 →
  S a 9 = 63 / 2 := by
  sorry

end arithmetic_sequence_sum_l2851_285128


namespace fraction_less_than_sqrt_l2851_285182

theorem fraction_less_than_sqrt (x : ℝ) (h : x > 0) : x / (1 + x) < Real.sqrt x := by
  sorry

end fraction_less_than_sqrt_l2851_285182


namespace reciprocal_equals_self_l2851_285125

theorem reciprocal_equals_self (x : ℝ) : x ≠ 0 ∧ x = 1 / x → x = 1 ∨ x = -1 := by
  sorry

end reciprocal_equals_self_l2851_285125


namespace water_bottle_consumption_l2851_285124

theorem water_bottle_consumption (total : ℕ) (first_day_fraction : ℚ) (remaining : ℕ) : 
  total = 24 → 
  first_day_fraction = 1/3 → 
  remaining = 8 → 
  (total - (first_day_fraction * total).num - remaining : ℚ) / (total - (first_day_fraction * total).num) = 1/2 := by
sorry

end water_bottle_consumption_l2851_285124


namespace possible_values_of_a_l2851_285123

theorem possible_values_of_a (a b x : ℤ) (h1 : a ≠ b) (h2 : a^3 - b^3 = 27*x^3) (h3 : a - b = 2*x) :
  a = (7*x + 5*(6: ℤ).sqrt*x) / 6 ∨ a = (7*x - 5*(6: ℤ).sqrt*x) / 6 :=
sorry

end possible_values_of_a_l2851_285123


namespace locus_of_Q_l2851_285109

-- Define the polar coordinate system
structure PolarCoord where
  ρ : ℝ
  θ : ℝ

-- Define the circle C
def circle_C (p : PolarCoord) : Prop :=
  p.ρ = 2

-- Define the line l
def line_l (p : PolarCoord) : Prop :=
  p.ρ * (Real.cos p.θ + Real.sin p.θ) = 2

-- Define the relationship between points O, P, Q, and R
def point_relationship (P Q R : PolarCoord) : Prop :=
  Q.ρ * P.ρ = R.ρ^2

-- Theorem statement
theorem locus_of_Q (P Q R : PolarCoord) :
  circle_C R →
  line_l P →
  point_relationship P Q R →
  Q.ρ = 2 * (Real.cos Q.θ + Real.sin Q.θ) ∧ Q.ρ ≠ 0 :=
by sorry

end locus_of_Q_l2851_285109


namespace triangle_angle_sum_l2851_285188

theorem triangle_angle_sum (a b c : ℝ) (ha : a = 5) (hb : b = 7) (hc : c = 8) :
  let θ := Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))
  (π - θ) * (180 / π) = 120 := by sorry

end triangle_angle_sum_l2851_285188


namespace line_through_point_with_equal_intercepts_l2851_285149

-- Define a line in 2D space
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ∀ x y : ℝ, a * x + b * y + c = 0

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a line passes through a point
def passes_through (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has equal intercepts on both axes
def has_equal_intercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ (-l.c / l.a = -l.c / l.b)

-- Theorem statement
theorem line_through_point_with_equal_intercepts :
  ∃ (l1 l2 : Line),
    (passes_through l1 ⟨1, 2⟩ ∧ has_equal_intercepts l1) ∧
    (passes_through l2 ⟨1, 2⟩ ∧ has_equal_intercepts l2) ∧
    ((l1.a = 2 ∧ l1.b = -1 ∧ l1.c = 0) ∨ (l2.a = 1 ∧ l2.b = 1 ∧ l2.c = -3)) :=
sorry

end line_through_point_with_equal_intercepts_l2851_285149


namespace weekly_earnings_l2851_285112

def phone_repair_cost : ℕ := 11
def laptop_repair_cost : ℕ := 15
def computer_repair_cost : ℕ := 18
def tablet_repair_cost : ℕ := 12
def smartwatch_repair_cost : ℕ := 8

def phone_repairs : ℕ := 9
def laptop_repairs : ℕ := 5
def computer_repairs : ℕ := 4
def tablet_repairs : ℕ := 6
def smartwatch_repairs : ℕ := 8

def total_earnings : ℕ := 
  phone_repair_cost * phone_repairs +
  laptop_repair_cost * laptop_repairs +
  computer_repair_cost * computer_repairs +
  tablet_repair_cost * tablet_repairs +
  smartwatch_repair_cost * smartwatch_repairs

theorem weekly_earnings : total_earnings = 382 := by
  sorry

end weekly_earnings_l2851_285112


namespace problem_solution_l2851_285133

theorem problem_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 * b + a * b^2 = 2) :
  (a^3 + b^3 ≥ 2) ∧ ((a + b) * (a^5 + b^5) ≥ 4) := by
  sorry

end problem_solution_l2851_285133


namespace quadratic_equation_solution_l2851_285162

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 - 2*x - 3
  (f 3 = 0 ∧ f (-1) = 0) ∧
  ∀ x : ℝ, f x = 0 → x = 3 ∨ x = -1 := by
  sorry

end quadratic_equation_solution_l2851_285162


namespace five_balls_four_boxes_l2851_285121

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 68 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 68 := by sorry

end five_balls_four_boxes_l2851_285121


namespace smallest_value_theorem_l2851_285116

theorem smallest_value_theorem (a b : ℕ+) (h : a.val^2 - b.val^2 = 16) :
  (∀ (c d : ℕ+), c.val^2 - d.val^2 = 16 →
    (a.val + b.val : ℚ) / (a.val - b.val : ℚ) + (a.val - b.val : ℚ) / (a.val + b.val : ℚ) ≤
    (c.val + d.val : ℚ) / (c.val - d.val : ℚ) + (c.val - d.val : ℚ) / (c.val + d.val : ℚ)) ∧
  (a.val + b.val : ℚ) / (a.val - b.val : ℚ) + (a.val - b.val : ℚ) / (a.val + b.val : ℚ) = 9/4 :=
sorry

end smallest_value_theorem_l2851_285116


namespace marble_problem_l2851_285130

theorem marble_problem (M : ℕ) 
  (h1 : M > 0)
  (h2 : (M - M / 3) / 4 > 0)
  (h3 : M - M / 3 - (M - M / 3) / 4 - 2 * ((M - M / 3) / 4) = 7) : 
  M = 42 := by
sorry

end marble_problem_l2851_285130


namespace wolf_hunting_problem_l2851_285163

theorem wolf_hunting_problem (hunting_wolves : ℕ) (pack_wolves : ℕ) (meat_per_wolf : ℕ) 
  (hunting_days : ℕ) (meat_per_deer : ℕ) : 
  hunting_wolves = 4 → 
  pack_wolves = 16 → 
  meat_per_wolf = 8 → 
  hunting_days = 5 → 
  meat_per_deer = 200 → 
  (hunting_wolves + pack_wolves) * meat_per_wolf * hunting_days / meat_per_deer / hunting_wolves = 1 := by
  sorry

#check wolf_hunting_problem

end wolf_hunting_problem_l2851_285163


namespace biff_break_even_l2851_285172

/-- The number of hours required for Biff to break even on his bus trip -/
def break_even_hours (ticket_cost drinks_snacks_cost headphones_cost online_earnings wifi_cost : ℚ) : ℚ :=
  (ticket_cost + drinks_snacks_cost + headphones_cost) / (online_earnings - wifi_cost)

/-- Theorem stating that Biff needs 3 hours to break even on his bus trip -/
theorem biff_break_even :
  break_even_hours 11 3 16 12 2 = 3 := by
  sorry

end biff_break_even_l2851_285172


namespace smallest_valid_number_last_four_digits_l2851_285199

def is_valid_representation (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 4 ∨ d = 9

def has_at_least_two_of_each (n : ℕ) : Prop :=
  (n.digits 10).count 4 ≥ 2 ∧ (n.digits 10).count 9 ≥ 2

def last_four_digits (n : ℕ) : ℕ := n % 10000

theorem smallest_valid_number_last_four_digits :
  ∃ m : ℕ,
    m > 0 ∧
    m % 4 = 0 ∧
    m % 9 = 0 ∧
    is_valid_representation m ∧
    has_at_least_two_of_each m ∧
    (∀ k : ℕ, k > 0 ∧ k % 4 = 0 ∧ k % 9 = 0 ∧ is_valid_representation k ∧ has_at_least_two_of_each k → m ≤ k) ∧
    last_four_digits m = 9494 :=
  by sorry

end smallest_valid_number_last_four_digits_l2851_285199


namespace total_drive_distance_l2851_285108

/-- The total distance of a drive given two drivers with different speed limits and driving times -/
theorem total_drive_distance (christina_speed : ℝ) (friend_speed : ℝ) (christina_time_min : ℝ) (friend_time_hr : ℝ) : 
  christina_speed = 30 →
  friend_speed = 40 →
  christina_time_min = 180 →
  friend_time_hr = 3 →
  christina_speed * (christina_time_min / 60) + friend_speed * friend_time_hr = 210 := by
  sorry

#check total_drive_distance

end total_drive_distance_l2851_285108
