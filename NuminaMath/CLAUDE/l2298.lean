import Mathlib

namespace NUMINAMATH_CALUDE_taxi_fare_proof_l2298_229840

/-- Proves that given an initial fare of $2.00 for the first 1/5 mile and a total fare of $25.40 for an 8-mile ride, the fare for each 1/5 mile after the first 1/5 mile is $0.60. -/
theorem taxi_fare_proof (initial_fare : ℝ) (total_fare : ℝ) (ride_distance : ℝ) 
  (h1 : initial_fare = 2)
  (h2 : total_fare = 25.4)
  (h3 : ride_distance = 8) :
  let increments : ℝ := ride_distance * 5
  let remaining_fare : ℝ := total_fare - initial_fare
  let remaining_increments : ℝ := increments - 1
  remaining_fare / remaining_increments = 0.6 := by
sorry

end NUMINAMATH_CALUDE_taxi_fare_proof_l2298_229840


namespace NUMINAMATH_CALUDE_shelter_dogs_l2298_229806

theorem shelter_dogs (C : ℕ) (h1 : C > 0) (h2 : (15 : ℚ) / C = 11 / (C + 8)) : 
  (15 : ℕ) * C = 15 * 15 :=
sorry

end NUMINAMATH_CALUDE_shelter_dogs_l2298_229806


namespace NUMINAMATH_CALUDE_worker_b_completion_time_l2298_229896

/-- Given a piece of work that can be completed by three workers a, b, and c, 
    this theorem proves the time taken by worker b to complete the work alone. -/
theorem worker_b_completion_time 
  (total_time : ℝ) 
  (time_a : ℝ) 
  (time_c : ℝ) 
  (h1 : total_time = 4) 
  (h2 : time_a = 36) 
  (h3 : time_c = 6) : 
  ∃ (time_b : ℝ), time_b = 18 ∧ 
  1 / total_time = 1 / time_a + 1 / time_b + 1 / time_c :=
by sorry

end NUMINAMATH_CALUDE_worker_b_completion_time_l2298_229896


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l2298_229869

-- Define the quadratic inequality and its solution set
def quadratic_inequality (a : ℝ) (x : ℝ) : Prop := (1 - a) * x^2 - 4*x + 6 > 0
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | -3 < x ∧ x < 1}

-- State the theorem
theorem quadratic_inequality_theorem (a : ℝ) :
  (∀ x, x ∈ solution_set a ↔ quadratic_inequality a x) →
  (a = 3) ∧
  (∀ x, 2*x^2 + (2-a)*x - a > 0 ↔ x < -1 ∨ x > 1) ∧
  (∀ b, (∀ x, a*x^2 + b*x + 3 ≥ 0) ↔ -6 ≤ b ∧ b ≤ 6) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l2298_229869


namespace NUMINAMATH_CALUDE_tangent_length_to_circle_l2298_229803

def origin : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (4, 5)
def B : ℝ × ℝ := (8, 10)
def C : ℝ × ℝ := (10, 25)

theorem tangent_length_to_circle (circle : Set (ℝ × ℝ)) 
  (h1 : A ∈ circle) (h2 : B ∈ circle) (h3 : C ∈ circle) :
  ∃ T ∈ circle, ‖T - origin‖ = Real.sqrt 82 ∧ 
  ∀ P ∈ circle, P ≠ T → ‖P - origin‖ > Real.sqrt 82 := by
  sorry

end NUMINAMATH_CALUDE_tangent_length_to_circle_l2298_229803


namespace NUMINAMATH_CALUDE_book_arrangement_count_l2298_229846

/-- Represents the number of books -/
def n : ℕ := 6

/-- Represents the number of ways to arrange books A and B at the ends -/
def end_arrangements : ℕ := 2

/-- Represents the number of ways to order books C and D -/
def cd_orders : ℕ := 2

/-- Represents the number of ways to arrange the C-D pair and the other 2 books in the middle -/
def middle_arrangements : ℕ := 6

/-- The total number of valid arrangements -/
def total_arrangements : ℕ := end_arrangements * cd_orders * middle_arrangements

theorem book_arrangement_count :
  total_arrangements = 24 :=
sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l2298_229846


namespace NUMINAMATH_CALUDE_sector_max_area_l2298_229807

/-- Given a sector with perimeter 16, its maximum area is 16. -/
theorem sector_max_area (r θ : ℝ) (h : 2 * r + r * θ = 16) : 
  ∀ r' θ' : ℝ, 2 * r' + r' * θ' = 16 → (1/2) * r' * r' * θ' ≤ 16 := by
sorry

end NUMINAMATH_CALUDE_sector_max_area_l2298_229807


namespace NUMINAMATH_CALUDE_max_value_of_sqrt_sum_max_value_achievable_l2298_229826

theorem max_value_of_sqrt_sum (x y z : ℝ) :
  x ≥ 0 → y ≥ 0 → z ≥ 0 → x + y + z = 6 →
  Real.sqrt (3 * x + 2) + Real.sqrt (3 * y + 2) + Real.sqrt (3 * z + 2) ≤ 3 * Real.sqrt 20 :=
by sorry

theorem max_value_achievable (x y z : ℝ) :
  x ≥ 0 → y ≥ 0 → z ≥ 0 → x + y + z = 6 →
  ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 6 ∧
  Real.sqrt (3 * a + 2) + Real.sqrt (3 * b + 2) + Real.sqrt (3 * c + 2) = 3 * Real.sqrt 20 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sqrt_sum_max_value_achievable_l2298_229826


namespace NUMINAMATH_CALUDE_circle_packing_problem_l2298_229853

theorem circle_packing_problem (n : ℕ) :
  (n^2 = ((n + 14) * (n + 15)) / 2) → n^2 = 1225 := by
  sorry

end NUMINAMATH_CALUDE_circle_packing_problem_l2298_229853


namespace NUMINAMATH_CALUDE_dice_probability_theorem_l2298_229851

/-- Represents a 12-sided die with colored sides -/
structure ColoredDie :=
  (violet : ℕ)
  (orange : ℕ)
  (lime : ℕ)
  (total : ℕ)
  (h1 : violet + orange + lime = total)
  (h2 : total = 12)

/-- The probability of two dice showing the same color -/
def same_color_probability (d : ColoredDie) : ℚ :=
  (d.violet^2 + d.orange^2 + d.lime^2) / d.total^2

/-- Theorem statement for the probability problem -/
theorem dice_probability_theorem (d : ColoredDie) 
  (hv : d.violet = 3) (ho : d.orange = 4) (hl : d.lime = 5) : 
  same_color_probability d = 25 / 72 := by
  sorry

#eval same_color_probability ⟨3, 4, 5, 12, by norm_num, rfl⟩

end NUMINAMATH_CALUDE_dice_probability_theorem_l2298_229851


namespace NUMINAMATH_CALUDE_chickens_and_rabbits_l2298_229811

theorem chickens_and_rabbits (total_heads total_feet : ℕ) 
  (h1 : total_heads = 35) 
  (h2 : total_feet = 94) : 
  ∃ (chickens rabbits : ℕ), 
    chickens + rabbits = total_heads ∧ 
    2 * chickens + 4 * rabbits = total_feet ∧ 
    chickens = 23 ∧ 
    rabbits = 12 := by
  sorry

#check chickens_and_rabbits

end NUMINAMATH_CALUDE_chickens_and_rabbits_l2298_229811


namespace NUMINAMATH_CALUDE_product_of_recurring_decimal_and_nine_l2298_229852

theorem product_of_recurring_decimal_and_nine (x : ℚ) : 
  x = 1/3 → x * 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_recurring_decimal_and_nine_l2298_229852


namespace NUMINAMATH_CALUDE_class_vision_median_l2298_229872

/-- Represents the vision data for a class of students -/
structure VisionData where
  visions : List ℝ
  counts : List ℕ
  total_students : ℕ

/-- Calculates the median of a VisionData set -/
def median (data : VisionData) : ℝ :=
  sorry

/-- The specific vision data for the class -/
def class_vision_data : VisionData :=
  { visions := [4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0],
    counts := [1, 2, 6, 3, 3, 4, 1, 2, 5, 7, 5],
    total_students := 39 }

/-- Theorem stating that the median of the class vision data is 4.6 -/
theorem class_vision_median :
  median class_vision_data = 4.6 := by
  sorry

end NUMINAMATH_CALUDE_class_vision_median_l2298_229872


namespace NUMINAMATH_CALUDE_sara_has_108_golf_balls_l2298_229832

/-- The number of dozens of golf balls Sara has -/
def saras_dozens : ℕ := 9

/-- The number of items in one dozen -/
def items_per_dozen : ℕ := 12

/-- The total number of golf balls Sara has -/
def saras_golf_balls : ℕ := saras_dozens * items_per_dozen

theorem sara_has_108_golf_balls : saras_golf_balls = 108 := by
  sorry

end NUMINAMATH_CALUDE_sara_has_108_golf_balls_l2298_229832


namespace NUMINAMATH_CALUDE_school_bus_capacity_l2298_229888

/-- The number of rows of seats in the school bus -/
def num_rows : ℕ := 20

/-- The number of kids that can sit in each row -/
def kids_per_row : ℕ := 4

/-- The total number of kids that can ride the school bus -/
def total_capacity : ℕ := num_rows * kids_per_row

theorem school_bus_capacity : total_capacity = 80 := by
  sorry

end NUMINAMATH_CALUDE_school_bus_capacity_l2298_229888


namespace NUMINAMATH_CALUDE_reese_practice_hours_l2298_229849

/-- Calculates the total piano practice hours for Reese over a given number of months -/
def total_practice_hours (months : ℕ) : ℕ :=
  let initial_weekly_hours := 4
  let initial_months := 2
  let increased_weekly_hours := 5
  let workshop_hours := 3
  
  let initial_practice := initial_weekly_hours * 4 * min months initial_months
  let increased_practice := increased_weekly_hours * 4 * max (months - initial_months) 0
  let total_workshops := months * workshop_hours
  
  initial_practice + increased_practice + total_workshops

/-- Theorem stating that Reese's total practice hours after 5 months is 107 -/
theorem reese_practice_hours : total_practice_hours 5 = 107 := by
  sorry

end NUMINAMATH_CALUDE_reese_practice_hours_l2298_229849


namespace NUMINAMATH_CALUDE_binomial_12_choose_6_l2298_229863

theorem binomial_12_choose_6 : Nat.choose 12 6 = 924 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_choose_6_l2298_229863


namespace NUMINAMATH_CALUDE_school_students_count_l2298_229859

/-- Represents the donation and student information for a school --/
structure SchoolDonation where
  total_donation : ℕ
  average_donation_7_8 : ℕ
  grade_9_intended_donation : ℕ
  grade_9_rejection_rate : ℚ

/-- Calculates the total number of students in the school based on donation information --/
def total_students (sd : SchoolDonation) : ℕ :=
  sd.total_donation / sd.average_donation_7_8

/-- Theorem stating that the total number of students in the school is 224 --/
theorem school_students_count (sd : SchoolDonation) 
  (h1 : sd.total_donation = 13440)
  (h2 : sd.average_donation_7_8 = 60)
  (h3 : sd.grade_9_intended_donation = 100)
  (h4 : sd.grade_9_rejection_rate = 2/5) :
  total_students sd = 224 := by
  sorry

#eval total_students { 
  total_donation := 13440, 
  average_donation_7_8 := 60, 
  grade_9_intended_donation := 100, 
  grade_9_rejection_rate := 2/5 
}

end NUMINAMATH_CALUDE_school_students_count_l2298_229859


namespace NUMINAMATH_CALUDE_watson_class_second_graders_l2298_229830

/-- The number of second graders in Ms. Watson's class -/
def second_graders (kindergartners first_graders total_students : ℕ) : ℕ :=
  total_students - (kindergartners + first_graders)

/-- Theorem stating the number of second graders in Ms. Watson's class -/
theorem watson_class_second_graders :
  second_graders 14 24 42 = 4 := by
  sorry

end NUMINAMATH_CALUDE_watson_class_second_graders_l2298_229830


namespace NUMINAMATH_CALUDE_m_values_l2298_229816

def A (m : ℝ) : Set ℝ := {1, 2, 3, m}
def B (m : ℝ) : Set ℝ := {m^2, 3}

theorem m_values (m : ℝ) :
  A m ∪ B m = A m →
  m = -1 ∨ m = Real.sqrt 2 ∨ m = -Real.sqrt 2 ∨ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_m_values_l2298_229816


namespace NUMINAMATH_CALUDE_valid_lineup_count_is_14_l2298_229825

/-- Represents the four athletes in the relay race -/
inductive Athlete : Type
| A : Athlete
| B : Athlete
| C : Athlete
| D : Athlete

/-- Represents the four positions in the relay race -/
inductive Position : Type
| first : Position
| second : Position
| third : Position
| fourth : Position

/-- A valid lineup for the relay race -/
def Lineup := Position → Athlete

/-- Predicate to check if a lineup is valid according to the given conditions -/
def isValidLineup (l : Lineup) : Prop :=
  l Position.first ≠ Athlete.A ∧ l Position.fourth ≠ Athlete.B

/-- The number of valid lineups -/
def validLineupCount : ℕ := sorry

/-- Theorem stating that the number of valid lineups is 14 -/
theorem valid_lineup_count_is_14 : validLineupCount = 14 := by sorry

end NUMINAMATH_CALUDE_valid_lineup_count_is_14_l2298_229825


namespace NUMINAMATH_CALUDE_floor_negative_seven_thirds_l2298_229864

theorem floor_negative_seven_thirds : ⌊(-7 : ℚ) / 3⌋ = -3 := by
  sorry

end NUMINAMATH_CALUDE_floor_negative_seven_thirds_l2298_229864


namespace NUMINAMATH_CALUDE_crayon_distribution_l2298_229818

theorem crayon_distribution (initial_boxes : Nat) (crayons_per_box : Nat) 
  (to_mae : Nat) (to_rey : Nat) (left : Nat) :
  initial_boxes = 7 →
  crayons_per_box = 15 →
  to_mae = 12 →
  to_rey = 20 →
  left = 25 →
  (initial_boxes * crayons_per_box - to_mae - to_rey - left) - to_mae = 36 := by
  sorry

end NUMINAMATH_CALUDE_crayon_distribution_l2298_229818


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2298_229883

theorem geometric_sequence_problem (a : ℕ → ℝ) :
  (∀ n, a n > 0) →  -- all terms are positive
  (∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q) →  -- geometric sequence
  a 2 = 1 →  -- a_2 = 1
  a 8 = a 6 + 6 * a 4 →  -- a_8 = a_6 + 6a_4
  a 3 = Real.sqrt 3 :=  -- a_3 = √3
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2298_229883


namespace NUMINAMATH_CALUDE_pie_difference_l2298_229858

/-- The number of apple pies baked on Mondays and Fridays -/
def monday_friday_apple : ℕ := 16

/-- The number of apple pies baked on Wednesdays -/
def wednesday_apple : ℕ := 20

/-- The number of cherry pies baked on Tuesdays -/
def tuesday_cherry : ℕ := 14

/-- The number of cherry pies baked on Thursdays -/
def thursday_cherry : ℕ := 18

/-- The number of apple pies baked on Saturdays -/
def saturday_apple : ℕ := 10

/-- The number of cherry pies baked on Saturdays -/
def saturday_cherry : ℕ := 8

/-- The number of apple pies baked on Sundays -/
def sunday_apple : ℕ := 6

/-- The number of cherry pies baked on Sundays -/
def sunday_cherry : ℕ := 12

/-- The total number of apple pies baked in one week -/
def total_apple : ℕ := 2 * monday_friday_apple + wednesday_apple + saturday_apple + sunday_apple

/-- The total number of cherry pies baked in one week -/
def total_cherry : ℕ := tuesday_cherry + thursday_cherry + saturday_cherry + sunday_cherry

theorem pie_difference : total_apple - total_cherry = 16 := by
  sorry

end NUMINAMATH_CALUDE_pie_difference_l2298_229858


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l2298_229878

/-- The x-intercept of the line 4x + 7y = 28 is the point (7, 0). -/
theorem x_intercept_of_line (x y : ℝ) : 
  (4 * x + 7 * y = 28) → (x = 7 ∧ y = 0 → 4 * x + 7 * y = 28) := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l2298_229878


namespace NUMINAMATH_CALUDE_right_triangle_area_l2298_229819

/-- The area of a right triangle with hypotenuse 9 inches and one angle 30° --/
theorem right_triangle_area (h : ℝ) (α : ℝ) (area : ℝ) : 
  h = 9 →  -- hypotenuse is 9 inches
  α = 30 * π / 180 →  -- one angle is 30°
  area = (9^2 * Real.sin (30 * π / 180) * Real.sin (60 * π / 180)) / 4 →  -- area formula for right triangle
  area = 10.125 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2298_229819


namespace NUMINAMATH_CALUDE_show_length_l2298_229802

/-- Proves that the length of each show is 50 minutes given the conditions -/
theorem show_length (gina_choice_ratio : ℝ) (total_shows : ℕ) (gina_minutes : ℝ) 
  (h1 : gina_choice_ratio = 3)
  (h2 : total_shows = 24)
  (h3 : gina_minutes = 900) : 
  (gina_minutes / (gina_choice_ratio * total_shows / (gina_choice_ratio + 1))) = 50 := by
  sorry


end NUMINAMATH_CALUDE_show_length_l2298_229802


namespace NUMINAMATH_CALUDE_longest_rod_in_cube_l2298_229829

theorem longest_rod_in_cube (side_length : ℝ) (h : side_length = 4) :
  Real.sqrt (3 * side_length^2) = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_longest_rod_in_cube_l2298_229829


namespace NUMINAMATH_CALUDE_product_of_twelve_and_3460_l2298_229800

theorem product_of_twelve_and_3460 : ∃ x : ℕ, 12 * x = 173 * x ∧ x = 3460 → 12 * 3460 = 41520 := by
  sorry

end NUMINAMATH_CALUDE_product_of_twelve_and_3460_l2298_229800


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2298_229897

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 2) : x^2 + 1/x^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2298_229897


namespace NUMINAMATH_CALUDE_min_draw_for_red_card_l2298_229893

theorem min_draw_for_red_card (total : ℕ) (blue yellow red : ℕ) :
  total = 20 →
  blue + yellow + red = total →
  blue = yellow / 6 →
  red < yellow →
  15 = total - red + 1 :=
by sorry

end NUMINAMATH_CALUDE_min_draw_for_red_card_l2298_229893


namespace NUMINAMATH_CALUDE_count_repetitive_permutations_formula_l2298_229850

/-- The count of n-repetitive permutations formed by a₁, a₂, a₃, a₄, a₅, a₆ 
    where both a₁ and a₃ each appear an even number of times -/
def count_repetitive_permutations (n : ℕ) : ℕ :=
  (6^n - 2 * 5^n + 4^n) / 4

/-- Theorem stating that the count of n-repetitive permutations with the given conditions
    is equal to (6^n - 2 * 5^n + 4^n) / 4 -/
theorem count_repetitive_permutations_formula (n : ℕ) :
  count_repetitive_permutations n = (6^n - 2 * 5^n + 4^n) / 4 := by
  sorry

end NUMINAMATH_CALUDE_count_repetitive_permutations_formula_l2298_229850


namespace NUMINAMATH_CALUDE_ice_cream_melt_time_l2298_229898

/-- The time it takes for an ice cream cone to melt, given the distance to the beach and Jack's jogging speed -/
theorem ice_cream_melt_time 
  (blocks_to_beach : ℕ)
  (miles_per_block : ℚ)
  (jogging_speed : ℚ)
  (h1 : blocks_to_beach = 16)
  (h2 : miles_per_block = 1 / 8)
  (h3 : jogging_speed = 12) :
  (blocks_to_beach : ℚ) * miles_per_block / jogging_speed * 60 = 10 := by
  sorry

#check ice_cream_melt_time

end NUMINAMATH_CALUDE_ice_cream_melt_time_l2298_229898


namespace NUMINAMATH_CALUDE_multiple_of_twelve_l2298_229865

theorem multiple_of_twelve (x : ℤ) : 
  (∃ k : ℤ, 7 * x - 3 = 12 * k) ↔ 
  (∃ t : ℤ, x = 12 * t + 9 ∨ x = 12 * t + 1029) :=
sorry

end NUMINAMATH_CALUDE_multiple_of_twelve_l2298_229865


namespace NUMINAMATH_CALUDE_smallest_staircase_steps_l2298_229833

theorem smallest_staircase_steps : ∃ n : ℕ,
  n > 15 ∧
  n % 3 = 1 ∧
  n % 5 = 3 ∧
  n % 7 = 1 ∧
  (∀ m : ℕ, m > 15 ∧ m % 3 = 1 ∧ m % 5 = 3 ∧ m % 7 = 1 → m ≥ n) ∧
  n = 73 := by
sorry

end NUMINAMATH_CALUDE_smallest_staircase_steps_l2298_229833


namespace NUMINAMATH_CALUDE_d_squared_plus_5d_l2298_229886

theorem d_squared_plus_5d (d : ℤ) : d = 5 + 6 → d^2 + 5*d = 176 := by
  sorry

end NUMINAMATH_CALUDE_d_squared_plus_5d_l2298_229886


namespace NUMINAMATH_CALUDE_exponential_inequality_range_l2298_229808

theorem exponential_inequality_range (a : ℝ) :
  (∀ x : ℝ, Real.exp (2 * x) - (a - 3) * Real.exp x + 4 - 3 * a > 0) →
  a < 4/3 :=
by sorry

end NUMINAMATH_CALUDE_exponential_inequality_range_l2298_229808


namespace NUMINAMATH_CALUDE_remainder_sum_l2298_229891

theorem remainder_sum (c d : ℤ) 
  (hc : c % 60 = 47) 
  (hd : d % 45 = 14) : 
  (c + d) % 15 = 1 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l2298_229891


namespace NUMINAMATH_CALUDE_halfway_fraction_l2298_229827

theorem halfway_fraction (a b c d : ℚ) (h1 : a = 3/4) (h2 : b = 5/6) :
  (a + b) / 2 = 19/24 := by sorry

end NUMINAMATH_CALUDE_halfway_fraction_l2298_229827


namespace NUMINAMATH_CALUDE_sum_of_ratios_theorem_l2298_229899

theorem sum_of_ratios_theorem (a b c : ℚ) 
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : a * b * c ≠ 0) (h5 : a + b + c = 0) : 
  a / |a| + b / |b| + c / |c| + (a * b * c) / |a * b * c| = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_ratios_theorem_l2298_229899


namespace NUMINAMATH_CALUDE_wire_service_reporters_l2298_229841

theorem wire_service_reporters (total : ℝ) (local_politics : ℝ) (non_local_politics : ℝ) 
  (h1 : local_politics = 0.18 * total)
  (h2 : non_local_politics = 0.4 * (local_politics + non_local_politics)) :
  (total - (local_politics + non_local_politics)) / total = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_wire_service_reporters_l2298_229841


namespace NUMINAMATH_CALUDE_swift_stream_pump_l2298_229837

/-- The SwiftStream pump problem -/
theorem swift_stream_pump (pump_rate : ℝ) (time : ℝ) (volume : ℝ) : 
  pump_rate = 500 → time = 1/2 → volume = pump_rate * time → volume = 250 := by
  sorry

end NUMINAMATH_CALUDE_swift_stream_pump_l2298_229837


namespace NUMINAMATH_CALUDE_production_time_is_13_hours_l2298_229815

/-- The time needed to complete the remaining production task -/
def time_to_complete (total_parts : ℕ) (apprentice_rate : ℕ) (master_rate : ℕ) (parts_done : ℕ) : ℚ :=
  (total_parts - parts_done) / (apprentice_rate + master_rate)

/-- Proof that the time to complete the production task is 13 hours -/
theorem production_time_is_13_hours :
  let total_parts : ℕ := 500
  let apprentice_rate : ℕ := 15
  let master_rate : ℕ := 20
  let parts_done : ℕ := 45
  time_to_complete total_parts apprentice_rate master_rate parts_done = 13 := by
  sorry

#eval time_to_complete 500 15 20 45

end NUMINAMATH_CALUDE_production_time_is_13_hours_l2298_229815


namespace NUMINAMATH_CALUDE_negation_of_forall_greater_than_five_l2298_229860

theorem negation_of_forall_greater_than_five (S : Set ℝ) :
  (¬ ∀ x ∈ S, x > 5) ↔ (∃ x ∈ S, x ≤ 5) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_greater_than_five_l2298_229860


namespace NUMINAMATH_CALUDE_both_selected_probability_l2298_229885

theorem both_selected_probability 
  (ram_prob : ℚ) 
  (ravi_prob : ℚ) 
  (h1 : ram_prob = 6 / 7) 
  (h2 : ravi_prob = 1 / 5) : 
  ram_prob * ravi_prob = 6 / 35 := by
sorry

end NUMINAMATH_CALUDE_both_selected_probability_l2298_229885


namespace NUMINAMATH_CALUDE_complex_magnitude_l2298_229870

theorem complex_magnitude (a b : ℝ) (z : ℂ) :
  (a + Complex.I)^2 = b * Complex.I →
  z = a + b * Complex.I →
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2298_229870


namespace NUMINAMATH_CALUDE_seven_divides_special_integer_l2298_229842

/-- Represents a 7-digit positive integer with the specified structure -/
structure SevenDigitInteger where
  value : ℕ
  is_seven_digit : 1000000 ≤ value ∧ value < 10000000
  first_three_equals_middle_three : ∃ (a b c : ℕ), value = a * 1000000 + b * 100000 + c * 10000 + a * 1000 + b * 100 + c * 10 + (value % 10)
  last_digit_multiple_of_first : ∃ (k : ℕ), value % 10 = k * ((value / 1000000) % 10)

/-- Theorem stating that 7 is a factor of any SevenDigitInteger -/
theorem seven_divides_special_integer (W : SevenDigitInteger) : 7 ∣ W.value := by
  sorry

end NUMINAMATH_CALUDE_seven_divides_special_integer_l2298_229842


namespace NUMINAMATH_CALUDE_quadratic_always_real_roots_roots_ratio_three_implies_m_values_l2298_229822

/-- The quadratic equation x^2 - 4x - m(m+4) = 0 -/
def quadratic_equation (x m : ℝ) : Prop :=
  x^2 - 4*x - m*(m+4) = 0

theorem quadratic_always_real_roots :
  ∀ m : ℝ, ∃ x₁ x₂ : ℝ, quadratic_equation x₁ m ∧ quadratic_equation x₂ m ∧ x₁ ≠ x₂ :=
sorry

theorem roots_ratio_three_implies_m_values :
  ∀ m x₁ x₂ : ℝ, quadratic_equation x₁ m → quadratic_equation x₂ m → x₂ = 3*x₁ →
  m = -1 ∨ m = -3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_always_real_roots_roots_ratio_three_implies_m_values_l2298_229822


namespace NUMINAMATH_CALUDE_max_candies_eaten_l2298_229836

/-- Represents the board with numbers -/
structure Board :=
  (numbers : List Nat)

/-- Represents Karlson's candy-eating process -/
def process (b : Board) : Nat :=
  let n := b.numbers.length
  n * (n - 1) / 2

/-- The initial board with 37 ones -/
def initial_board : Board :=
  { numbers := List.replicate 37 1 }

/-- The theorem stating the maximum number of candies Karlson can eat -/
theorem max_candies_eaten (b : Board := initial_board) : 
  process b = 666 := by
  sorry

end NUMINAMATH_CALUDE_max_candies_eaten_l2298_229836


namespace NUMINAMATH_CALUDE_youngest_sibling_age_l2298_229889

/-- Given 4 siblings where 3 are 4, 5, and 7 years older than the youngest,
    and their average age is 21, prove that the age of the youngest sibling is 17. -/
theorem youngest_sibling_age (y : ℕ) : 
  (y + (y + 4) + (y + 5) + (y + 7)) / 4 = 21 → y = 17 := by
  sorry

end NUMINAMATH_CALUDE_youngest_sibling_age_l2298_229889


namespace NUMINAMATH_CALUDE_problem_solution_l2298_229880

theorem problem_solution (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a * b * c = 1) (h5 : a + 1 / c = 7) (h6 : b + 1 / a = 35) :
  c + 1 / b = 11 / 61 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2298_229880


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2298_229810

theorem sufficient_not_necessary_condition (a : ℝ) :
  (a ≥ 5) → (∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≤ 0) ∧
  ¬(∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≤ 0) → a ≥ 5) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2298_229810


namespace NUMINAMATH_CALUDE_angle_b_range_in_geometric_progression_triangle_l2298_229855

/-- In a triangle ABC, if sides a, b, c form a geometric progression,
    then angle B is in the range (0, π/3] -/
theorem angle_b_range_in_geometric_progression_triangle
  (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a + b > c → b + c > a → c + a > b →
  b^2 = a * c →
  0 < B ∧ B ≤ π/3 := by
  sorry

end NUMINAMATH_CALUDE_angle_b_range_in_geometric_progression_triangle_l2298_229855


namespace NUMINAMATH_CALUDE_recipe_ratio_l2298_229854

/-- Given a recipe with 5 cups of flour and 1 cup of shortening,
    if 2/3 cup of shortening is used, then 3 1/3 cups of flour
    should be used to maintain the same ratio. -/
theorem recipe_ratio (original_flour : ℚ) (original_shortening : ℚ)
                     (available_shortening : ℚ) (needed_flour : ℚ) :
  original_flour = 5 →
  original_shortening = 1 →
  available_shortening = 2/3 →
  needed_flour = 10/3 →
  needed_flour / available_shortening = original_flour / original_shortening :=
by sorry

end NUMINAMATH_CALUDE_recipe_ratio_l2298_229854


namespace NUMINAMATH_CALUDE_danielle_apartment_rooms_l2298_229813

theorem danielle_apartment_rooms : 
  ∀ (heidi grant danielle : ℕ),
  heidi = 3 * danielle →
  grant * 9 = heidi →
  grant = 2 →
  danielle = 6 := by
sorry

end NUMINAMATH_CALUDE_danielle_apartment_rooms_l2298_229813


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2298_229887

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I)^2 * z = 3 + 2*Complex.I → z = -1 + (3/2)*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2298_229887


namespace NUMINAMATH_CALUDE_imaginary_equation_solution_l2298_229845

theorem imaginary_equation_solution (z : ℂ) (b : ℝ) : 
  (z.re = 0) →  -- z is a pure imaginary number
  ((2 - I) * z = 4 - b * (1 + I)^2) →
  b = -4 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_equation_solution_l2298_229845


namespace NUMINAMATH_CALUDE_workers_total_earnings_l2298_229862

/-- Calculates the total earnings of three workers given their daily wages and days worked. -/
def total_earnings (daily_wage_a daily_wage_b daily_wage_c : ℕ) (days_a days_b days_c : ℕ) : ℕ :=
  daily_wage_a * days_a + daily_wage_b * days_b + daily_wage_c * days_c

/-- Theorem stating that under the given conditions, the total earnings of three workers is 1480. -/
theorem workers_total_earnings :
  ∀ (daily_wage_a daily_wage_b daily_wage_c : ℕ),
    daily_wage_a * 3 = daily_wage_b * 3 * 3/4 →
    daily_wage_b * 4 = daily_wage_c * 4 * 4/5 →
    daily_wage_c = 100 →
    total_earnings daily_wage_a daily_wage_b daily_wage_c 6 9 4 = 1480 :=
by
  sorry

#eval total_earnings 60 80 100 6 9 4

end NUMINAMATH_CALUDE_workers_total_earnings_l2298_229862


namespace NUMINAMATH_CALUDE_fifth_number_21st_row_l2298_229876

/-- Represents the triangular array of numbers -/
def TriangularArray (n : ℕ) (k : ℕ) : ℕ :=
  sorry

theorem fifth_number_21st_row :
  (∀ n : ℕ, TriangularArray n n = n^2) →
  (∀ n k : ℕ, k < n → TriangularArray n (k+1) = TriangularArray n k + 1) →
  (∀ n : ℕ, TriangularArray (n+1) 1 = TriangularArray n n + 1) →
  TriangularArray 21 5 = 405 :=
sorry

end NUMINAMATH_CALUDE_fifth_number_21st_row_l2298_229876


namespace NUMINAMATH_CALUDE_jennifer_additional_tanks_l2298_229844

/-- Represents the number of fish in each type of tank --/
structure TankCapacity where
  goldfish : Nat
  betta : Nat
  guppy : Nat
  clownfish : Nat

/-- Represents the number of tanks for each type of fish --/
structure TankCount where
  goldfish : Nat
  betta : Nat
  guppy : Nat
  clownfish : Nat

/-- Calculates the total number of fish given tank capacities and counts --/
def totalFish (capacity : TankCapacity) (count : TankCount) : Nat :=
  capacity.goldfish * count.goldfish +
  capacity.betta * count.betta +
  capacity.guppy * count.guppy +
  capacity.clownfish * count.clownfish

/-- Calculates the total number of tanks --/
def totalTanks (count : TankCount) : Nat :=
  count.goldfish + count.betta + count.guppy + count.clownfish

/-- Represents Jennifer's aquarium setup --/
def jennifer_setup : Prop :=
  ∃ (capacity : TankCapacity) (existing_count : TankCount) (new_count : TankCount),
    capacity.goldfish = 15 ∧
    capacity.betta = 1 ∧
    capacity.guppy = 5 ∧
    capacity.clownfish = 4 ∧
    existing_count.goldfish = 3 ∧
    existing_count.betta = 0 ∧
    existing_count.guppy = 0 ∧
    existing_count.clownfish = 0 ∧
    totalFish capacity (TankCount.mk
      existing_count.goldfish
      (existing_count.betta + new_count.betta)
      (existing_count.guppy + new_count.guppy)
      (existing_count.clownfish + new_count.clownfish)) = 75 ∧
    new_count.betta + new_count.guppy + new_count.clownfish = 15 ∧
    ∀ (alt_count : TankCount),
      totalFish capacity (TankCount.mk
        existing_count.goldfish
        (existing_count.betta + alt_count.betta)
        (existing_count.guppy + alt_count.guppy)
        (existing_count.clownfish + alt_count.clownfish)) = 75 →
      totalTanks alt_count ≥ totalTanks new_count

theorem jennifer_additional_tanks : jennifer_setup := by
  sorry

end NUMINAMATH_CALUDE_jennifer_additional_tanks_l2298_229844


namespace NUMINAMATH_CALUDE_complex_calculation_l2298_229890

-- Define the complex number z
def z : ℂ := 1 + Complex.I

-- Define w as a function of z
def w (z : ℂ) : ℂ := z^2 + 3 - 4

-- Theorem statement
theorem complex_calculation :
  w z = 2 * Complex.I - 1 := by sorry

end NUMINAMATH_CALUDE_complex_calculation_l2298_229890


namespace NUMINAMATH_CALUDE_simplify_nested_expression_l2298_229847

theorem simplify_nested_expression (x : ℝ) : 1 - (2 - (2 - (2 - (2 - x)))) = 1 - x := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_expression_l2298_229847


namespace NUMINAMATH_CALUDE_wine_drinking_time_is_correct_l2298_229879

/-- Represents the time taken for three assistants to drink 40 liters of wine -/
def wine_drinking_time : ℚ :=
  let rate1 := (40 : ℚ) / 12  -- Rate of the first assistant
  let rate2 := (40 : ℚ) / 10  -- Rate of the second assistant
  let rate3 := (40 : ℚ) / 8   -- Rate of the third assistant
  let total_rate := rate1 + rate2 + rate3
  (40 : ℚ) / total_rate

/-- The wine drinking time is equal to 3 9/37 hours -/
theorem wine_drinking_time_is_correct : wine_drinking_time = 3 + 9 / 37 := by
  sorry

#eval wine_drinking_time

end NUMINAMATH_CALUDE_wine_drinking_time_is_correct_l2298_229879


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l2298_229821

/-- Given two points on a line and another line equation, 
    prove that the value of k for which the lines are parallel is 14. -/
theorem parallel_lines_k_value (k : ℝ) : 
  (∃ (m b : ℝ), (∀ x y : ℝ, y = m * x + b ↔ 6 * x - 2 * y = -8) ∧
   (∃ (m' b' : ℝ), m' = m ∧ 23 = m' * k + b' ∧ -4 = m' * 5 + b')) →
  k = 14 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l2298_229821


namespace NUMINAMATH_CALUDE_distance_product_range_l2298_229814

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := y^2 = 4*x
def C₂ (x y : ℝ) : Prop := (x-4)^2 + y^2 = 8

-- Define a point P on C₁
structure PointOnC₁ where
  x : ℝ
  y : ℝ
  on_C₁ : C₁ x y

-- Define the line l with 45° inclination passing through P
def line_l (P : PointOnC₁) (x y : ℝ) : Prop :=
  y - P.y = (x - P.x)

-- Define the intersection points Q and R
structure IntersectionPoint where
  x : ℝ
  y : ℝ
  on_C₂ : C₂ x y
  on_l : line_l P x y

-- Define the product of distances |PQ| · |PR|
def distance_product (P : PointOnC₁) (Q R : IntersectionPoint) : ℝ :=
  ((Q.x - P.x)^2 + (Q.y - P.y)^2) * ((R.x - P.x)^2 + (R.y - P.y)^2)

-- State the theorem
theorem distance_product_range (P : PointOnC₁) (Q R : IntersectionPoint) 
  (h_distinct : Q ≠ R) :
  ∃ (d : ℝ), distance_product P Q R = d ∧ (d ∈ Set.Icc 4 8 ∨ d ∈ Set.Ioo 8 200) :=
sorry

end NUMINAMATH_CALUDE_distance_product_range_l2298_229814


namespace NUMINAMATH_CALUDE_guard_max_demand_l2298_229834

/-- Represents the outcome of the outsider's decision -/
inductive Outcome
| Pay
| Refuse

/-- Represents the guard's demand and the outsider's decision -/
structure Scenario where
  guardDemand : ℕ
  outsiderDecision : Outcome

/-- Calculates the outsider's loss based on the scenario -/
def outsiderLoss (s : Scenario) : ℤ :=
  match s.outsiderDecision with
  | Outcome.Pay => s.guardDemand - 100
  | Outcome.Refuse => 100

/-- Determines if the outsider will pay based on personal benefit -/
def willPay (guardDemand : ℕ) : Prop :=
  outsiderLoss { guardDemand := guardDemand, outsiderDecision := Outcome.Pay } <
  outsiderLoss { guardDemand := guardDemand, outsiderDecision := Outcome.Refuse }

/-- The maximum number of coins the guard can demand -/
def maxGuardDemand : ℕ := 199

theorem guard_max_demand :
  (∀ n : ℕ, n ≤ maxGuardDemand → willPay n) ∧
  (∀ n : ℕ, n > maxGuardDemand → ¬willPay n) :=
sorry

end NUMINAMATH_CALUDE_guard_max_demand_l2298_229834


namespace NUMINAMATH_CALUDE_not_cheap_necessary_not_sufficient_for_good_quality_l2298_229828

-- Define the universe of products
variable (Product : Type)

-- Define predicates for product qualities
variable (not_cheap : Product → Prop)
variable (good_quality : Product → Prop)

-- Define the saying "you get what you pay for" as an axiom
axiom you_get_what_you_pay_for : ∀ (p : Product), good_quality p → not_cheap p

-- Theorem to prove
theorem not_cheap_necessary_not_sufficient_for_good_quality :
  (∀ (p : Product), good_quality p → not_cheap p) ∧
  (∃ (p : Product), not_cheap p ∧ ¬good_quality p) :=
sorry

end NUMINAMATH_CALUDE_not_cheap_necessary_not_sufficient_for_good_quality_l2298_229828


namespace NUMINAMATH_CALUDE_equal_std_dev_and_range_l2298_229804

variable (n : ℕ) (c : ℝ)
variable (x y : Fin n → ℝ)

-- Define the relationship between x and y
def y_def : Prop := ∀ i : Fin n, y i = x i + c

-- Define sample standard deviation
def sample_std_dev (z : Fin n → ℝ) : ℝ := sorry

-- Define sample range
def sample_range (z : Fin n → ℝ) : ℝ := sorry

-- Theorem statement
theorem equal_std_dev_and_range (hc : c ≠ 0) (h_y_def : y_def n c x y) :
  (sample_std_dev n x = sample_std_dev n y) ∧
  (sample_range n x = sample_range n y) := by sorry

end NUMINAMATH_CALUDE_equal_std_dev_and_range_l2298_229804


namespace NUMINAMATH_CALUDE_number_of_divisors_2001_l2298_229809

theorem number_of_divisors_2001 : Finset.card (Nat.divisors 2001) = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_2001_l2298_229809


namespace NUMINAMATH_CALUDE_tangent_intersections_symmetric_l2298_229831

/-- A line intersecting two hyperbolas -/
structure IntersectingLine where
  m : ℝ  -- slope of the line
  q : ℝ  -- y-intercept of the line

/-- The intersection points of tangents to a hyperbola -/
structure TangentIntersection where
  x : ℝ
  y : ℝ

/-- Calculate the intersection point of tangents for y = 1/x hyperbola -/
noncomputable def tangentIntersection1 (line : IntersectingLine) : TangentIntersection :=
  { x := 2 * line.m / line.q
  , y := -2 / line.q }

/-- Calculate the intersection point of tangents for y = -1/x hyperbola -/
noncomputable def tangentIntersection2 (line : IntersectingLine) : TangentIntersection :=
  { x := -2 * line.m / line.q
  , y := 2 / line.q }

/-- Two points are symmetric about the origin -/
def symmetricAboutOrigin (p1 p2 : TangentIntersection) : Prop :=
  p1.x = -p2.x ∧ p1.y = -p2.y

/-- Main theorem: The intersection points of tangents are symmetric about the origin -/
theorem tangent_intersections_symmetric (line : IntersectingLine) :
  symmetricAboutOrigin (tangentIntersection1 line) (tangentIntersection2 line) := by
  sorry

end NUMINAMATH_CALUDE_tangent_intersections_symmetric_l2298_229831


namespace NUMINAMATH_CALUDE_triangular_prism_area_bound_l2298_229884

/-- Given a triangular prism P-ABC with specified side lengths, 
    the sum of squared areas of triangles ABC and PBC is bounded. -/
theorem triangular_prism_area_bound 
  (AB : ℝ) (AC : ℝ) (PB : ℝ) (PC : ℝ)
  (h_AB : AB = Real.sqrt 3)
  (h_AC : AC = 1)
  (h_PB : PB = Real.sqrt 2)
  (h_PC : PC = Real.sqrt 2) :
  ∃ (S_ABC S_PBC : ℝ),
    (1/4 : ℝ) < S_ABC^2 + S_PBC^2 ∧ 
    S_ABC^2 + S_PBC^2 ≤ (7/4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_triangular_prism_area_bound_l2298_229884


namespace NUMINAMATH_CALUDE_min_max_sum_l2298_229848

theorem min_max_sum (x y z u v : ℕ+) (h : x + y + z + u + v = 2505) :
  let N := max (x + y) (max (y + z) (max (z + u) (u + v)))
  N ≥ 1253 ∧ ∃ (a b c d e : ℕ+), a + b + c + d + e = 2505 ∧
    max (a + b) (max (b + c) (max (c + d) (d + e))) = 1253 := by
  sorry

end NUMINAMATH_CALUDE_min_max_sum_l2298_229848


namespace NUMINAMATH_CALUDE_equation_solution_inequality_system_solution_l2298_229882

-- Part 1: Equation solution
theorem equation_solution :
  ∀ x : ℝ, x ≠ 1 ∧ x ≠ 0 → (x / (x - 1) = (x - 1) / (2*x - 2) ↔ x = -1) :=
sorry

-- Part 2: Inequality system solution
theorem inequality_system_solution :
  ∀ x : ℝ, (5*x - 1 > 3*x - 4 ∧ -1/3*x ≤ 2/3 - x) ↔ (-3/2 < x ∧ x ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_inequality_system_solution_l2298_229882


namespace NUMINAMATH_CALUDE_electronic_product_failure_probability_l2298_229867

theorem electronic_product_failure_probability
  (p_working : ℝ)
  (h_working : p_working = 0.992)
  (h_probability : 0 ≤ p_working ∧ p_working ≤ 1) :
  1 - p_working = 0.008 := by
sorry

end NUMINAMATH_CALUDE_electronic_product_failure_probability_l2298_229867


namespace NUMINAMATH_CALUDE_find_set_N_l2298_229857

def U : Set ℕ := {1, 2, 3, 4, 5}

theorem find_set_N (M N : Set ℕ) 
  (h1 : U = M ∪ N) 
  (h2 : M ∩ (U \ N) = {2, 4}) : 
  N = {1, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_find_set_N_l2298_229857


namespace NUMINAMATH_CALUDE_exists_fibonacci_divisible_by_n_l2298_229892

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

-- Statement of the theorem
theorem exists_fibonacci_divisible_by_n (n : ℕ) (hn : n > 0) : 
  ∃ m : ℕ, m > 0 ∧ n ∣ fib m :=
sorry

end NUMINAMATH_CALUDE_exists_fibonacci_divisible_by_n_l2298_229892


namespace NUMINAMATH_CALUDE_functional_equation_equivalence_l2298_229868

theorem functional_equation_equivalence (f : ℝ → ℝ) :
  (∀ x y, f (x + y) = f x + f y) ↔ (∀ x y, f (x + y + x * y) = f x + f y + f (x * y)) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_equivalence_l2298_229868


namespace NUMINAMATH_CALUDE_other_number_proof_l2298_229801

theorem other_number_proof (a b : ℕ+) 
  (h1 : Nat.lcm a b = 5040)
  (h2 : Nat.gcd a b = 24)
  (h3 : a = 240) :
  b = 504 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l2298_229801


namespace NUMINAMATH_CALUDE_min_value_with_constraint_l2298_229824

theorem min_value_with_constraint (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_xyz : x * y * z = 3) :
  x^2 + 4*x*y + 12*y^2 + 8*y*z + 3*z^2 ≥ 162 ∧ 
  (x^2 + 4*x*y + 12*y^2 + 8*y*z + 3*z^2 = 162 ↔ x = 3 ∧ y = 1/2 ∧ z = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_with_constraint_l2298_229824


namespace NUMINAMATH_CALUDE_special_line_equation_l2298_229856

/-- A line passing through (-4, -1) with x-intercept twice its y-intercept -/
structure SpecialLine where
  /-- The slope of the line -/
  slope : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The line passes through (-4, -1) -/
  passes_through : slope * (-4) + y_intercept = -1
  /-- The x-intercept is twice the y-intercept -/
  intercept_relation : -y_intercept / slope = 2 * y_intercept

/-- The equation of the special line is x + 2y + 6 = 0 or y = 1/4 x -/
theorem special_line_equation (l : SpecialLine) :
  (∀ x y, y = l.slope * x + l.y_intercept ↔ x + 2*y + 6 = 0) ∨
  (l.slope = 1/4 ∧ l.y_intercept = 0) :=
sorry

end NUMINAMATH_CALUDE_special_line_equation_l2298_229856


namespace NUMINAMATH_CALUDE_division_remainder_l2298_229874

theorem division_remainder (N : ℕ) : 
  (∃ r : ℕ, N = 5 * 5 + r ∧ r < 5) ∧ 
  (∃ q : ℕ, N = 11 * q + 3) → 
  N % 5 = 0 := by sorry

end NUMINAMATH_CALUDE_division_remainder_l2298_229874


namespace NUMINAMATH_CALUDE_yard_length_with_26_trees_l2298_229820

/-- The length of a yard with equally spaced trees -/
def yard_length (num_trees : ℕ) (tree_distance : ℝ) : ℝ :=
  (num_trees - 1 : ℝ) * tree_distance

theorem yard_length_with_26_trees :
  yard_length 26 11 = 275 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_with_26_trees_l2298_229820


namespace NUMINAMATH_CALUDE_smallest_area_of_2020th_square_l2298_229881

theorem smallest_area_of_2020th_square (n : ℕ) (A : ℕ) : 
  n > 0 → 
  n^2 = 2019 + A → 
  A ≠ 1 → 
  (∀ m : ℕ, m > 0 ∧ m^2 = 2019 + A → n ≤ m) → 
  A ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_area_of_2020th_square_l2298_229881


namespace NUMINAMATH_CALUDE_expression_evaluation_l2298_229877

theorem expression_evaluation :
  let x : ℚ := -1/2
  let y : ℚ := -2
  (3*x - 2*y)^2 - (2*y + x)*(2*y - x) - 2*x*(5*x - 6*y + x*y) = 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2298_229877


namespace NUMINAMATH_CALUDE_normal_distribution_probabilities_l2298_229805

-- Define a random variable following a normal distribution
def normal_distribution (μ : ℝ) (σ : ℝ) : Type := ℝ

-- Define the cumulative distribution function (CDF) for a normal distribution
noncomputable def normal_cdf (μ σ : ℝ) (x : ℝ) : ℝ := sorry

-- State the theorem
theorem normal_distribution_probabilities 
  (ξ : normal_distribution 1.5 σ) 
  (h : normal_cdf 1.5 σ 2.5 = 0.78) : 
  normal_cdf 1.5 σ 0.5 = 0.22 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probabilities_l2298_229805


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l2298_229835

/-- Given two functions f and g defined as:
    f(x) = -2|x-a| + b
    g(x) = 2|x-c| + d
    If f(5) = g(5) = 10 and f(11) = g(11) = 6,
    then a + c = 16 -/
theorem intersection_implies_sum (a b c d : ℝ) :
  (∀ x, -2 * |x - a| + b = 2 * |x - c| + d → x = 5 ∨ x = 11) →
  -2 * |5 - a| + b = 10 →
  2 * |5 - c| + d = 10 →
  -2 * |11 - a| + b = 6 →
  2 * |11 - c| + d = 6 →
  a + c = 16 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l2298_229835


namespace NUMINAMATH_CALUDE_helmet_store_theorem_l2298_229839

/-- Represents the sales data for a single day -/
structure DailySales where
  helmetA : ℕ
  helmetB : ℕ
  totalAmount : ℕ

/-- Represents the helmet store problem -/
structure HelmetStore where
  day1 : DailySales
  day2 : DailySales
  costPriceA : ℕ
  costPriceB : ℕ
  totalHelmets : ℕ
  budget : ℕ
  profitGoal : ℕ

/-- The main theorem for the helmet store problem -/
theorem helmet_store_theorem (store : HelmetStore)
  (h1 : store.day1 = ⟨10, 15, 1150⟩)
  (h2 : store.day2 = ⟨6, 12, 810⟩)
  (h3 : store.costPriceA = 40)
  (h4 : store.costPriceB = 30)
  (h5 : store.totalHelmets = 100)
  (h6 : store.budget = 3400)
  (h7 : store.profitGoal = 1300) :
  ∃ (priceA priceB maxA : ℕ),
    priceA = 55 ∧
    priceB = 40 ∧
    maxA = 40 ∧
    ¬∃ (numA : ℕ), numA ≤ maxA ∧ 
      (priceA - store.costPriceA) * numA + 
      (priceB - store.costPriceB) * (store.totalHelmets - numA) ≥ store.profitGoal :=
sorry

end NUMINAMATH_CALUDE_helmet_store_theorem_l2298_229839


namespace NUMINAMATH_CALUDE_quadratic_sum_l2298_229894

theorem quadratic_sum (x : ℝ) : ∃ (a b c : ℝ), 
  (15 * x^2 + 150 * x + 2250 = a * (x + b)^2 + c) ∧ (a + b + c = 1895) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2298_229894


namespace NUMINAMATH_CALUDE_trigonometric_equality_l2298_229866

theorem trigonometric_equality (x y : ℝ) 
  (h : (Real.sin x ^ 2 - Real.cos x ^ 2 + Real.cos x ^ 2 * Real.cos y ^ 2 - Real.sin x ^ 2 * Real.sin y ^ 2) / Real.sin (x + y) = 1) :
  ∃ k : ℤ, x - y = 2 * k * Real.pi + Real.pi / 2 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l2298_229866


namespace NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l2298_229871

theorem cos_x_plus_2y_equals_one 
  (x y : ℝ) 
  (a : ℝ) 
  (h1 : x ∈ Set.Icc (-π/4) (π/4)) 
  (h2 : y ∈ Set.Icc (-π/4) (π/4)) 
  (h3 : x^3 + Real.sin x - 2*a = 0) 
  (h4 : 4*y^3 + Real.sin y * Real.cos y + a = 0) : 
  Real.cos (x + 2*y) = 1 := by
sorry

end NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l2298_229871


namespace NUMINAMATH_CALUDE_six_digit_numbers_with_zero_l2298_229895

/-- The total number of 6-digit numbers -/
def total_six_digit_numbers : ℕ := 900000

/-- The number of 6-digit numbers without any zero -/
def six_digit_numbers_without_zero : ℕ := 531441

/-- Theorem: The number of 6-digit numbers with at least one zero is 368,559 -/
theorem six_digit_numbers_with_zero :
  total_six_digit_numbers - six_digit_numbers_without_zero = 368559 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_numbers_with_zero_l2298_229895


namespace NUMINAMATH_CALUDE_timmy_calories_needed_l2298_229838

/-- Represents the number of calories in an orange -/
def calories_per_orange : ℕ := 80

/-- Represents the cost of an orange in cents -/
def cost_per_orange : ℕ := 120

/-- Represents Timmy's initial amount of money in cents -/
def initial_money : ℕ := 1000

/-- Represents the amount of money Timmy has left after buying oranges in cents -/
def money_left : ℕ := 400

/-- Calculates the number of calories Timmy needs to get -/
def calories_needed : ℕ := 
  ((initial_money - money_left) / cost_per_orange) * calories_per_orange

theorem timmy_calories_needed : calories_needed = 400 := by
  sorry

end NUMINAMATH_CALUDE_timmy_calories_needed_l2298_229838


namespace NUMINAMATH_CALUDE_floor_equation_solution_l2298_229875

theorem floor_equation_solution (x : ℝ) : 
  (⌊⌊3 * x⌋ - 1⌋ = ⌊x + 3⌋) ↔ (5/3 ≤ x ∧ x < 3 ∧ x ≠ 2) :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l2298_229875


namespace NUMINAMATH_CALUDE_box_surface_area_is_1600_l2298_229861

/-- Calculates the surface area of the interior of an open box formed by removing square corners from a rectangular sheet and folding the sides. -/
def boxSurfaceArea (length width cornerSize : ℕ) : ℕ :=
  length * width - 4 * (cornerSize * cornerSize)

/-- Theorem stating that the surface area of the interior of the box is 1600 square units. -/
theorem box_surface_area_is_1600 :
  boxSurfaceArea 40 50 10 = 1600 := by
  sorry

#eval boxSurfaceArea 40 50 10

end NUMINAMATH_CALUDE_box_surface_area_is_1600_l2298_229861


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l2298_229843

theorem quadratic_one_solution (p : ℚ) : 
  (∃! x, 3 * x^2 - 7 * x + p = 0) ↔ p = 49/12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l2298_229843


namespace NUMINAMATH_CALUDE_election_vote_count_l2298_229823

theorem election_vote_count 
  (candidate1_percentage : ℝ) 
  (candidate2_votes : ℕ) 
  (total_votes : ℕ) : 
  candidate1_percentage = 0.7 →
  candidate2_votes = 240 →
  (candidate2_votes : ℝ) / total_votes = 1 - candidate1_percentage →
  total_votes = 800 := by
sorry

end NUMINAMATH_CALUDE_election_vote_count_l2298_229823


namespace NUMINAMATH_CALUDE_orange_eaters_difference_l2298_229812

def family_gathering (total : ℕ) (orange_eaters : ℕ) (banana_eaters : ℕ) (apple_eaters : ℕ) : Prop :=
  total = 20 ∧
  orange_eaters = total / 2 ∧
  banana_eaters = (total - orange_eaters) / 2 ∧
  apple_eaters = total - orange_eaters - banana_eaters ∧
  orange_eaters < total

theorem orange_eaters_difference (total orange_eaters banana_eaters apple_eaters : ℕ) :
  family_gathering total orange_eaters banana_eaters apple_eaters →
  total - orange_eaters = 10 := by
  sorry

end NUMINAMATH_CALUDE_orange_eaters_difference_l2298_229812


namespace NUMINAMATH_CALUDE_complex_inequality_condition_l2298_229873

theorem complex_inequality_condition (z : ℂ) :
  (∀ z, Complex.abs z ≤ 1 → Complex.abs (Complex.re z) ≤ 1 ∧ Complex.abs (Complex.im z) ≤ 1) ∧
  (∃ z, Complex.abs (Complex.re z) ≤ 1 ∧ Complex.abs (Complex.im z) ≤ 1 ∧ Complex.abs z > 1) :=
by sorry

end NUMINAMATH_CALUDE_complex_inequality_condition_l2298_229873


namespace NUMINAMATH_CALUDE_system_solution_l2298_229817

theorem system_solution :
  ∀ x y z : ℝ,
  (x * y = z * (x + y + z) ∧
   y * z = 4 * x * (x + y + z) ∧
   z * x = 9 * y * (x + y + z)) →
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨
   ∃ t : ℝ, t ≠ 0 ∧ x = -3 * t ∧ y = -2 * t ∧ z = 6 * t) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2298_229817
