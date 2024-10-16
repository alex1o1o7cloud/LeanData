import Mathlib

namespace NUMINAMATH_CALUDE_polar_equation_defines_parabola_l118_11878

/-- The polar equation r = 1 / (1 + cos θ) defines a parabola. -/
theorem polar_equation_defines_parabola :
  ∃ (a b c : ℝ), a ≠ 0 ∧
  (∀ (x y : ℝ), (∃ (r θ : ℝ), r > 0 ∧ 
    r = 1 / (1 + Real.cos θ) ∧
    x = r * Real.cos θ ∧
    y = r * Real.sin θ) ↔
    a * y^2 + b * x + c = 0) :=
sorry

end NUMINAMATH_CALUDE_polar_equation_defines_parabola_l118_11878


namespace NUMINAMATH_CALUDE_parabola_rhombus_theorem_l118_11819

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the rhombus
def rhombus (O B F C : ℝ × ℝ) : Prop :=
  let (xo, yo) := O
  let (xb, yb) := B
  let (xf, yf) := F
  let (xc, yc) := C
  (xf - xo)^2 + (yf - yo)^2 = (xb - xc)^2 + (yb - yc)^2 ∧
  (xb - xo)^2 + (yb - yo)^2 = (xc - xo)^2 + (yc - yo)^2

-- Define the theorem
theorem parabola_rhombus_theorem (p : ℝ) (O B F C : ℝ × ℝ) :
  parabola p B.1 B.2 →
  parabola p C.1 C.2 →
  rhombus O B F C →
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = 4 →
  p = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_parabola_rhombus_theorem_l118_11819


namespace NUMINAMATH_CALUDE_system_solution_unique_l118_11865

theorem system_solution_unique (x y : ℚ) : 
  (4 * x + 7 * y = -19) ∧ (4 * x - 5 * y = 17) ↔ (x = 1/2) ∧ (y = -3) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l118_11865


namespace NUMINAMATH_CALUDE_larger_screen_diagonal_l118_11817

theorem larger_screen_diagonal (d : ℝ) : 
  d ^ 2 = 17 ^ 2 + 36 → d = 5 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_larger_screen_diagonal_l118_11817


namespace NUMINAMATH_CALUDE_profit_difference_is_183_50_l118_11847

-- Define the given quantities
def cat_food_packages : ℕ := 9
def dog_food_packages : ℕ := 7
def cans_per_cat_package : ℕ := 15
def cans_per_dog_package : ℕ := 8
def cost_per_cat_package : ℚ := 14
def cost_per_dog_package : ℚ := 10
def price_per_cat_can : ℚ := 2.5
def price_per_dog_can : ℚ := 1.75

-- Define the profit calculation function
def profit_difference : ℚ :=
  let cat_revenue := (cat_food_packages * cans_per_cat_package : ℚ) * price_per_cat_can
  let dog_revenue := (dog_food_packages * cans_per_dog_package : ℚ) * price_per_dog_can
  let cat_cost := (cat_food_packages : ℚ) * cost_per_cat_package
  let dog_cost := (dog_food_packages : ℚ) * cost_per_dog_package
  (cat_revenue - cat_cost) - (dog_revenue - dog_cost)

-- Theorem statement
theorem profit_difference_is_183_50 : profit_difference = 183.5 := by sorry

end NUMINAMATH_CALUDE_profit_difference_is_183_50_l118_11847


namespace NUMINAMATH_CALUDE_remainder_double_number_l118_11850

theorem remainder_double_number (N : ℤ) : 
  N % 398 = 255 → (2 * N) % 398 = 112 := by
sorry

end NUMINAMATH_CALUDE_remainder_double_number_l118_11850


namespace NUMINAMATH_CALUDE_ellipse_k_value_l118_11876

/-- Theorem: For an ellipse with equation 5x^2 - ky^2 = 5 and one focus at (0, 2), the value of k is -1. -/
theorem ellipse_k_value (k : ℝ) : 
  (∃ (x y : ℝ), 5 * x^2 - k * y^2 = 5) → -- Ellipse equation
  (∃ (c : ℝ), c = 2 ∧ c^2 = 5 - (-5/k)) → -- Focus at (0, 2) and standard form relation
  k = -1 := by
sorry

end NUMINAMATH_CALUDE_ellipse_k_value_l118_11876


namespace NUMINAMATH_CALUDE_minimum_travel_time_for_problem_scenario_l118_11866

/-- Represents the travel scenario between two cities -/
structure TravelScenario where
  distance : ℝ
  num_people : ℕ
  num_bicycles : ℕ
  cyclist_speed : ℝ
  pedestrian_speed : ℝ

/-- The minimum time for all people to reach the destination -/
def minimum_travel_time (scenario : TravelScenario) : ℝ :=
  sorry

/-- The specific travel scenario from the problem -/
def problem_scenario : TravelScenario :=
  { distance := 45
    num_people := 3
    num_bicycles := 2
    cyclist_speed := 15
    pedestrian_speed := 5 }

theorem minimum_travel_time_for_problem_scenario :
  minimum_travel_time problem_scenario = 3 :=
by sorry

end NUMINAMATH_CALUDE_minimum_travel_time_for_problem_scenario_l118_11866


namespace NUMINAMATH_CALUDE_restaurant_spend_l118_11851

/-- The total amount spent by a group at a restaurant -/
def total_spent (n : ℕ) (individual_spends : Fin n → ℚ) : ℚ :=
  (Finset.univ.sum fun i => individual_spends i)

/-- The average expenditure of a group -/
def average_spend (n : ℕ) (individual_spends : Fin n → ℚ) : ℚ :=
  (total_spent n individual_spends) / n

theorem restaurant_spend :
  ∀ (individual_spends : Fin 8 → ℚ),
  (∀ i : Fin 7, individual_spends i = 10) →
  (individual_spends 7 = average_spend 8 individual_spends + 7) →
  total_spent 8 individual_spends = 88 := by
sorry

end NUMINAMATH_CALUDE_restaurant_spend_l118_11851


namespace NUMINAMATH_CALUDE_ashleys_age_l118_11845

theorem ashleys_age (ashley mary : ℕ) : 
  (ashley : ℚ) / mary = 4 / 7 → 
  ashley + mary = 22 → 
  ashley = 8 :=
by sorry

end NUMINAMATH_CALUDE_ashleys_age_l118_11845


namespace NUMINAMATH_CALUDE_sean_train_track_length_l118_11896

theorem sean_train_track_length 
  (ruth_piece_length : ℕ) 
  (total_length : ℕ) 
  (ruth_pieces : ℕ) 
  (sean_pieces : ℕ) :
  ruth_piece_length = 18 →
  total_length = 72 →
  ruth_pieces * ruth_piece_length = total_length →
  sean_pieces = ruth_pieces →
  sean_pieces * (total_length / sean_pieces) = total_length →
  total_length / sean_pieces = 18 :=
by
  sorry

#check sean_train_track_length

end NUMINAMATH_CALUDE_sean_train_track_length_l118_11896


namespace NUMINAMATH_CALUDE_probability_heart_then_spade_or_club_l118_11830

-- Define the total number of cards in a standard deck
def total_cards : ℕ := 52

-- Define the number of hearts in a standard deck
def num_hearts : ℕ := 13

-- Define the number of spades and clubs combined in a standard deck
def num_spades_clubs : ℕ := 26

-- Theorem statement
theorem probability_heart_then_spade_or_club :
  (num_hearts / total_cards) * (num_spades_clubs / (total_cards - 1)) = 13 / 102 := by
  sorry

end NUMINAMATH_CALUDE_probability_heart_then_spade_or_club_l118_11830


namespace NUMINAMATH_CALUDE_selection_methods_eq_51_l118_11859

/-- The number of ways to select k elements from n elements -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of students -/
def total_students : ℕ := 9

/-- The number of students to be selected -/
def selected_students : ℕ := 4

/-- The number of specific students (A, B, C) -/
def specific_students : ℕ := 3

/-- The number of ways to select 4 students from 9, where at least two of three specific students must be selected -/
def selection_methods : ℕ :=
  choose specific_students 2 * choose (total_students - specific_students) (selected_students - 2) +
  choose specific_students 3 * choose (total_students - specific_students) (selected_students - 3)

theorem selection_methods_eq_51 : selection_methods = 51 := by sorry

end NUMINAMATH_CALUDE_selection_methods_eq_51_l118_11859


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l118_11861

theorem parallelogram_base_length 
  (area : ℝ) (height : ℝ) (base : ℝ) 
  (h1 : area = 72) 
  (h2 : height = 6) 
  (h3 : area = base * height) : 
  base = 12 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l118_11861


namespace NUMINAMATH_CALUDE_sphere_surface_area_change_l118_11842

theorem sphere_surface_area_change (r₁ r₂ : ℝ) (h : r₁ > 0) (h' : r₂ > 0) : 
  (π * r₂^2 = 4 * π * r₁^2) → (4 * π * r₂^2 = 4 * (4 * π * r₁^2)) := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_change_l118_11842


namespace NUMINAMATH_CALUDE_lg_sum_equals_lg_product_l118_11805

-- Define logarithm base 10
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem lg_sum_equals_lg_product : lg 2 + lg 5 = lg 10 := by sorry

end NUMINAMATH_CALUDE_lg_sum_equals_lg_product_l118_11805


namespace NUMINAMATH_CALUDE_sum_f_positive_l118_11888

-- Define the function f
def f (x : ℝ) : ℝ := x + x^3

-- State the theorem
theorem sum_f_positive (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ + x₂ > 0) 
  (h₂ : x₂ + x₃ > 0) 
  (h₃ : x₃ + x₁ > 0) : 
  f x₁ + f x₂ + f x₃ > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_positive_l118_11888


namespace NUMINAMATH_CALUDE_probability_bounds_l118_11823

/-- A segment of natural numbers -/
structure Segment where
  start : ℕ
  length : ℕ

/-- The probability of a number in a segment being divisible by 10 -/
def probability_divisible_by_10 (s : Segment) : ℚ :=
  (s.length.div 10) / s.length

/-- The maximum probability of a number in any segment being divisible by 10 -/
def max_probability : ℚ := 1

/-- The minimum non-zero probability of a number in any segment being divisible by 10 -/
def min_nonzero_probability : ℚ := 1 / 19

theorem probability_bounds :
  ∀ s : Segment, 
    probability_divisible_by_10 s ≤ max_probability ∧
    (probability_divisible_by_10 s ≠ 0 → probability_divisible_by_10 s ≥ min_nonzero_probability) :=
by sorry

end NUMINAMATH_CALUDE_probability_bounds_l118_11823


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l118_11889

theorem modulus_of_complex_fraction :
  let z : ℂ := (1 - 2*I) / (3 - I)
  Complex.abs z = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l118_11889


namespace NUMINAMATH_CALUDE_system_of_equations_l118_11881

theorem system_of_equations (x y m : ℝ) : 
  x - y = 5 → 
  x + 2*y = 3*m - 1 → 
  2*x + y = 13 → 
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_l118_11881


namespace NUMINAMATH_CALUDE_midpoint_after_translation_l118_11879

/-- Given points A, J, and H in a 2D coordinate system, and a translation vector,
    prove that the midpoint of A'H' after translation is as specified. -/
theorem midpoint_after_translation (A J H : ℝ × ℝ) (translation : ℝ × ℝ) :
  A = (3, 3) →
  J = (4, 8) →
  H = (7, 3) →
  translation = (-6, 3) →
  let A' := (A.1 + translation.1, A.2 + translation.2)
  let H' := (H.1 + translation.1, H.2 + translation.2)
  ((A'.1 + H'.1) / 2, (A'.2 + H'.2) / 2) = (-1, 6) := by
  sorry

end NUMINAMATH_CALUDE_midpoint_after_translation_l118_11879


namespace NUMINAMATH_CALUDE_divisibility_of_expression_l118_11890

theorem divisibility_of_expression (x : ℤ) (h : Odd x) :
  ∃ k : ℤ, (8 * x + 6) * (8 * x + 10) * (4 * x + 4) = 384 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_expression_l118_11890


namespace NUMINAMATH_CALUDE_songs_storable_jeff_l118_11860

/-- Calculates the number of songs that can be stored on a phone given the total storage, used storage, and size of each song. -/
def songs_storable (total_storage : ℕ) (used_storage : ℕ) (song_size : ℕ) : ℕ :=
  ((total_storage - used_storage) * 1000) / song_size

/-- Theorem stating that given the specific conditions, 400 songs can be stored. -/
theorem songs_storable_jeff : songs_storable 16 4 30 = 400 := by
  sorry

#eval songs_storable 16 4 30

end NUMINAMATH_CALUDE_songs_storable_jeff_l118_11860


namespace NUMINAMATH_CALUDE_sphere_cylinder_volume_difference_l118_11867

/-- The volume of the space inside a sphere and outside an inscribed right cylinder -/
theorem sphere_cylinder_volume_difference (r_sphere : ℝ) (r_cylinder : ℝ) :
  r_sphere = 6 →
  r_cylinder = 4 →
  let h_cylinder := 2 * Real.sqrt (r_sphere ^ 2 - r_cylinder ^ 2)
  let v_sphere := (4 / 3) * Real.pi * r_sphere ^ 3
  let v_cylinder := Real.pi * r_cylinder ^ 2 * h_cylinder
  v_sphere - v_cylinder = (288 - 64 * Real.sqrt 5) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_cylinder_volume_difference_l118_11867


namespace NUMINAMATH_CALUDE_xyz_sum_equals_zero_l118_11891

theorem xyz_sum_equals_zero 
  (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_eq1 : x^2 + x*y + y^2 = 48)
  (h_eq2 : y^2 + y*z + z^2 = 25)
  (h_eq3 : z^2 + x*z + x^2 = 73) :
  x*y + y*z + x*z = 0 :=
sorry

end NUMINAMATH_CALUDE_xyz_sum_equals_zero_l118_11891


namespace NUMINAMATH_CALUDE_unique_positive_solution_l118_11875

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ 3 * x^2 + 13 * x - 10 = 0 :=
by
  -- The unique positive solution is x = 2/3
  use 2/3
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l118_11875


namespace NUMINAMATH_CALUDE_highest_possible_average_after_removing_lowest_score_l118_11855

def number_of_tests : ℕ := 9
def original_average : ℚ := 68
def lowest_possible_score : ℚ := 0

theorem highest_possible_average_after_removing_lowest_score :
  let total_score : ℚ := number_of_tests * original_average
  let remaining_score : ℚ := total_score - lowest_possible_score
  let new_average : ℚ := remaining_score / (number_of_tests - 1)
  new_average = 76.5 := by
  sorry

end NUMINAMATH_CALUDE_highest_possible_average_after_removing_lowest_score_l118_11855


namespace NUMINAMATH_CALUDE_prime_8p_plus_1_square_cube_l118_11812

theorem prime_8p_plus_1_square_cube (p : ℕ) : 
  Prime p → 
  ((∃ n : ℕ, 8 * p + 1 = n^2) ↔ p = 3) ∧ 
  (¬∃ n : ℕ, 8 * p + 1 = n^3) := by
sorry

end NUMINAMATH_CALUDE_prime_8p_plus_1_square_cube_l118_11812


namespace NUMINAMATH_CALUDE_jane_start_babysitting_age_l118_11836

/-- Represents the age at which Jane started babysitting -/
def start_age : ℕ := 8

/-- Jane's current age -/
def current_age : ℕ := 32

/-- Years since Jane stopped babysitting -/
def years_since_stopped : ℕ := 10

/-- Current age of the oldest person Jane could have babysat -/
def oldest_babysat_current_age : ℕ := 24

/-- Theorem stating that Jane started babysitting at age 8 -/
theorem jane_start_babysitting_age :
  (start_age + years_since_stopped < current_age) ∧
  (∀ (jane_age : ℕ) (child_age : ℕ),
    jane_age ≥ start_age →
    jane_age ≤ current_age - years_since_stopped →
    child_age ≤ oldest_babysat_current_age - (current_age - jane_age) →
    child_age ≤ jane_age / 2) ∧
  (oldest_babysat_current_age = current_age - (start_age + 8)) :=
by sorry

#check jane_start_babysitting_age

end NUMINAMATH_CALUDE_jane_start_babysitting_age_l118_11836


namespace NUMINAMATH_CALUDE_elective_schemes_count_l118_11811

/-- The number of courses offered -/
def total_courses : ℕ := 10

/-- The number of courses that can't be chosen together -/
def conflicting_courses : ℕ := 3

/-- The number of courses each student must choose -/
def courses_to_choose : ℕ := 3

/-- The number of different elective schemes -/
def num_elective_schemes : ℕ := 98

theorem elective_schemes_count :
  (total_courses = 10) →
  (conflicting_courses = 3) →
  (courses_to_choose = 3) →
  (num_elective_schemes = Nat.choose (total_courses - conflicting_courses) courses_to_choose +
                          conflicting_courses * Nat.choose (total_courses - conflicting_courses) (courses_to_choose - 1)) :=
by sorry

end NUMINAMATH_CALUDE_elective_schemes_count_l118_11811


namespace NUMINAMATH_CALUDE_gwen_birthday_money_l118_11853

/-- The amount of money Gwen received from her mom -/
def money_from_mom : ℕ := 8

/-- The amount of money Gwen received from her dad -/
def money_from_dad : ℕ := 5

/-- The amount of money Gwen spent -/
def money_spent : ℕ := 4

/-- The difference between the amount Gwen received from her mom and her dad -/
def difference : ℕ := money_from_mom - money_from_dad

theorem gwen_birthday_money : difference = 3 := by
  sorry

end NUMINAMATH_CALUDE_gwen_birthday_money_l118_11853


namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l118_11882

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- Theorem: For a geometric sequence, a₁ < a₃ if and only if a₅ < a₇ -/
theorem geometric_sequence_inequality (a : ℕ → ℝ) (h : geometric_sequence a) :
  a 1 < a 3 ↔ a 5 < a 7 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_inequality_l118_11882


namespace NUMINAMATH_CALUDE_investment_ratio_l118_11873

/-- Given three investors A, B, and C with the following conditions:
  1. A invests the same amount as B
  2. A invests 2/3 of what C invests
  3. Total profit is 11000
  4. C's share of the profit is 3000
Prove that the ratio of A's investment to B's investment is 1:1 -/
theorem investment_ratio (a b c : ℝ) (h1 : a = b) (h2 : a = (2/3) * c)
  (total_profit : ℝ) (h3 : total_profit = 11000)
  (c_share : ℝ) (h4 : c_share = 3000) :
  a / b = 1 := by
  sorry

end NUMINAMATH_CALUDE_investment_ratio_l118_11873


namespace NUMINAMATH_CALUDE_apps_deleted_minus_added_l118_11887

theorem apps_deleted_minus_added (initial_apps added_apps final_apps : ℕ) : 
  initial_apps = 15 → added_apps = 71 → final_apps = 14 →
  (initial_apps + added_apps - final_apps) - added_apps = 1 := by
  sorry

end NUMINAMATH_CALUDE_apps_deleted_minus_added_l118_11887


namespace NUMINAMATH_CALUDE_equation_solutions_l118_11872

theorem equation_solutions :
  (∀ x : ℝ, (x - 4)^2 - 9 = 0 ↔ x = 7 ∨ x = 1) ∧
  (∀ x : ℝ, (x + 1)^3 = -27 ↔ x = -4) := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l118_11872


namespace NUMINAMATH_CALUDE_divisibility_by_13_l118_11870

theorem divisibility_by_13 (x y : ℤ) 
  (h1 : (x^2 - 3*x*y + 2*y^2 + x - y) % 13 = 0)
  (h2 : (x^2 - 2*x*y + y^2 - 5*x + 7) % 13 = 0) :
  (x*y - 12*x + 15*y) % 13 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_13_l118_11870


namespace NUMINAMATH_CALUDE_eugene_toothpick_boxes_l118_11895

/-- Represents the number of toothpicks needed for Eugene's model house --/
def toothpicks_needed (total_cards : ℕ) (unused_cards : ℕ) (wall_toothpicks : ℕ) 
  (window_count : ℕ) (door_count : ℕ) (window_door_toothpicks : ℕ) (roof_toothpicks : ℕ) : ℕ :=
  let used_cards := total_cards - unused_cards
  let wall_total := used_cards * wall_toothpicks
  let window_door_total := used_cards * (window_count + door_count) * window_door_toothpicks
  wall_total + window_door_total + roof_toothpicks

/-- Theorem stating that Eugene used at least 7 boxes of toothpicks --/
theorem eugene_toothpick_boxes : 
  ∀ (box_capacity : ℕ),
  box_capacity = 750 →
  ∃ (n : ℕ), n ≥ 7 ∧ 
  n * box_capacity ≥ toothpicks_needed 52 23 64 3 2 12 1250 :=
by sorry

end NUMINAMATH_CALUDE_eugene_toothpick_boxes_l118_11895


namespace NUMINAMATH_CALUDE_complex_number_forms_l118_11802

theorem complex_number_forms (z : ℂ) : 
  z = 4 * (Complex.cos (4 * Real.pi / 3) + Complex.I * Complex.sin (4 * Real.pi / 3)) →
  (z = -2 - 2 * Complex.I * Real.sqrt 3) ∧ 
  (z = 4 * Complex.exp (Complex.I * (4 * Real.pi / 3))) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_forms_l118_11802


namespace NUMINAMATH_CALUDE_triangle_side_length_l118_11877

theorem triangle_side_length (A B C : ℝ × ℝ) (tanB : ℝ) (AB : ℝ) :
  tanB = 4 / 3 →
  AB = 3 →
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = AB^2 →
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = (AB * tanB)^2 →
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 25 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l118_11877


namespace NUMINAMATH_CALUDE_sum_of_factors_of_30_l118_11864

def sum_of_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_factors_of_30 : sum_of_factors 30 = 72 := by sorry

end NUMINAMATH_CALUDE_sum_of_factors_of_30_l118_11864


namespace NUMINAMATH_CALUDE_dogwood_trees_in_park_l118_11828

theorem dogwood_trees_in_park (current trees_today trees_tomorrow total : ℕ) : 
  trees_today = 41 → 
  trees_tomorrow = 20 → 
  total = 100 → 
  current + trees_today + trees_tomorrow = total → 
  current = 39 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_trees_in_park_l118_11828


namespace NUMINAMATH_CALUDE_total_outfits_l118_11827

/-- Represents the number of shirts available. -/
def num_shirts : ℕ := 7

/-- Represents the number of ties available. -/
def num_ties : ℕ := 5

/-- Represents the number of pairs of pants available. -/
def num_pants : ℕ := 4

/-- Represents the number of shoe types available. -/
def num_shoe_types : ℕ := 2

/-- Calculates the number of outfit combinations with a tie. -/
def outfits_with_tie : ℕ := num_shirts * num_pants * num_ties

/-- Calculates the number of outfit combinations without a tie. -/
def outfits_without_tie : ℕ := num_shirts * num_pants

/-- Theorem stating the total number of different outfits. -/
theorem total_outfits : outfits_with_tie + outfits_without_tie = 168 := by
  sorry

end NUMINAMATH_CALUDE_total_outfits_l118_11827


namespace NUMINAMATH_CALUDE_sum_remainder_mod_nine_l118_11899

theorem sum_remainder_mod_nine (n : ℤ) : ((9 - n) + (n + 5)) % 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_nine_l118_11899


namespace NUMINAMATH_CALUDE_tank_capacity_l118_11825

/-- Represents a tank with a leak and an inlet pipe -/
structure Tank where
  capacity : ℝ
  leakRate : ℝ
  inletRate : ℝ

/-- The conditions of the problem -/
def tankProblem (t : Tank) : Prop :=
  t.leakRate = t.capacity / 6 ∧ 
  t.inletRate = 3.5 * 60 ∧ 
  t.inletRate - t.leakRate = t.capacity / 8

/-- The theorem stating that under the given conditions, the tank's capacity is 720 liters -/
theorem tank_capacity (t : Tank) : tankProblem t → t.capacity = 720 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l118_11825


namespace NUMINAMATH_CALUDE_abs_sum_leq_sum_abs_l118_11858

theorem abs_sum_leq_sum_abs (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  |a| + |b| ≤ |a + b| := by
sorry

end NUMINAMATH_CALUDE_abs_sum_leq_sum_abs_l118_11858


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_eq_five_l118_11854

/-- The function f(x) = x^3 + ax^2 + 3x - 9 has an extreme value at x = -3 -/
def has_extreme_value_at_neg_three (a : ℝ) : Prop :=
  let f := fun x : ℝ => x^3 + a*x^2 + 3*x - 9
  ∃ ε > 0, ∀ x ∈ Set.Ioo (-3-ε) (-3+ε), f x ≤ f (-3) ∨ f x ≥ f (-3)

/-- If f(x) = x^3 + ax^2 + 3x - 9 has an extreme value at x = -3, then a = 5 -/
theorem extreme_value_implies_a_eq_five :
  ∀ a : ℝ, has_extreme_value_at_neg_three a → a = 5 := by sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_eq_five_l118_11854


namespace NUMINAMATH_CALUDE_periodic_decimal_difference_l118_11837

theorem periodic_decimal_difference : (4 : ℚ) / 11 - (9 : ℚ) / 25 = (1 : ℚ) / 275 := by sorry

end NUMINAMATH_CALUDE_periodic_decimal_difference_l118_11837


namespace NUMINAMATH_CALUDE_function_inequality_condition_l118_11804

theorem function_inequality_condition (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = 3 * x + 2) →
  a > 0 →
  b > 0 →
  (∀ x, |x + 2| < b → |f x + 4| < a) ↔
  b ≤ a / 3 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_condition_l118_11804


namespace NUMINAMATH_CALUDE_find_tap_a_turnoff_time_l118_11852

/-- Represents the time it takes for a tap to fill the cistern -/
structure TapFillTime where
  minutes : ℝ
  positive : minutes > 0

/-- Represents the state of the cistern filling process -/
structure CisternFilling where
  tapA : TapFillTime
  tapB : TapFillTime
  remainingTime : ℝ
  positive : remainingTime > 0

/-- The main theorem statement -/
theorem find_tap_a_turnoff_time (c : CisternFilling) 
    (h1 : c.tapA.minutes = 12)
    (h2 : c.tapB.minutes = 18)
    (h3 : c.remainingTime = 8) : 
  ∃ t : ℝ, t > 0 ∧ t = 4 ∧
    (t * (1 / c.tapA.minutes + 1 / c.tapB.minutes) + 
     c.remainingTime * (1 / c.tapB.minutes) = 1) := by
  sorry

#check find_tap_a_turnoff_time

end NUMINAMATH_CALUDE_find_tap_a_turnoff_time_l118_11852


namespace NUMINAMATH_CALUDE_parallel_sufficient_not_necessary_l118_11869

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane3D where
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Predicate to check if a line is parallel to a plane -/
def is_parallel (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Predicate to check if a line is outside of a plane -/
def is_outside (l : Line3D) (p : Plane3D) : Prop :=
  sorry

theorem parallel_sufficient_not_necessary
  (l : Line3D) (α : Plane3D) :
  (is_parallel l α → is_outside l α) ∧
  ∃ l', is_outside l' α ∧ ¬is_parallel l' α :=
sorry

end NUMINAMATH_CALUDE_parallel_sufficient_not_necessary_l118_11869


namespace NUMINAMATH_CALUDE_gcd_1239_2829_times_15_l118_11843

theorem gcd_1239_2829_times_15 : 15 * Int.gcd 1239 2829 = 315 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1239_2829_times_15_l118_11843


namespace NUMINAMATH_CALUDE_part_a_part_b_part_b_max_exists_l118_11894

-- Part (a)
def P (k : ℝ) (x : ℝ) : ℝ := x^3 - k*x + 2

theorem part_a (k : ℝ) : P k 2 = 0 → k = 5 := by sorry

-- Part (b)
theorem part_b (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  2*a + b + 4/(a*b) = 10 → a ≤ 4 := by sorry

theorem part_b_max_exists :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2*a + b + 4/(a*b) = 10 ∧ a = 4 := by sorry

end NUMINAMATH_CALUDE_part_a_part_b_part_b_max_exists_l118_11894


namespace NUMINAMATH_CALUDE_solution_set_inequality_l118_11883

theorem solution_set_inequality (a b : ℝ) :
  (∀ x, x^2 + a*x + b < 0 ↔ 2 < x ∧ x < 3) →
  (∀ x, b*x^2 + a*x + 1 > 0 ↔ x < 1/3 ∨ x > 1/2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l118_11883


namespace NUMINAMATH_CALUDE_value_of_a_l118_11849

-- Define the function f
def f (x : ℝ) : ℝ := 2 * (2 * x + 2) - 5

-- State the theorem
theorem value_of_a : ∃ a : ℝ, f (1/2 * a - 1) = 2 * a - 5 ∧ f a = 6 → a = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l118_11849


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l118_11807

theorem min_value_theorem (x : ℝ) : 
  (x^2 + 11) / Real.sqrt (x^2 + 5) ≥ 2 * Real.sqrt 6 :=
by sorry

theorem min_value_achievable : 
  ∃ x : ℝ, (x^2 + 11) / Real.sqrt (x^2 + 5) = 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l118_11807


namespace NUMINAMATH_CALUDE_odd_function_negative_domain_l118_11829

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_domain
  (f : ℝ → ℝ)
  (h_odd : IsOdd f)
  (h_pos : ∀ x > 0, f x = 2^x + 1) :
  ∀ x < 0, f x = -2^(-x) - 1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_negative_domain_l118_11829


namespace NUMINAMATH_CALUDE_sailboat_two_sail_speed_l118_11832

/-- Represents the speed of a sailboat in knots -/
structure SailboatSpeed :=
  (speed : ℝ)

/-- Represents the travel conditions for a sailboat -/
structure TravelConditions :=
  (oneSpeedSail : SailboatSpeed)
  (twoSpeedSail : SailboatSpeed)
  (timeOneSail : ℝ)
  (timeTwoSail : ℝ)
  (totalDistance : ℝ)
  (nauticalMileToLandMile : ℝ)

/-- The main theorem stating the speed of the sailboat with two sails -/
theorem sailboat_two_sail_speed 
  (conditions : TravelConditions)
  (h1 : conditions.oneSpeedSail.speed = 25)
  (h2 : conditions.timeOneSail = 4)
  (h3 : conditions.timeTwoSail = 4)
  (h4 : conditions.totalDistance = 345)
  (h5 : conditions.nauticalMileToLandMile = 1.15)
  : conditions.twoSpeedSail.speed = 50 := by
  sorry

#check sailboat_two_sail_speed

end NUMINAMATH_CALUDE_sailboat_two_sail_speed_l118_11832


namespace NUMINAMATH_CALUDE_students_not_in_chorus_or_band_l118_11813

theorem students_not_in_chorus_or_band 
  (total : ℕ) (chorus : ℕ) (band : ℕ) (both : ℕ) 
  (h1 : total = 50) 
  (h2 : chorus = 18) 
  (h3 : band = 26) 
  (h4 : both = 2) : 
  total - (chorus + band - both) = 8 := by
  sorry

end NUMINAMATH_CALUDE_students_not_in_chorus_or_band_l118_11813


namespace NUMINAMATH_CALUDE_max_value_of_sum_of_squares_l118_11893

theorem max_value_of_sum_of_squares (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 = 10) :
  (∃ (m : ℝ), ∀ (x y z w : ℝ), x^2 + y^2 + z^2 + w^2 = 10 →
    (x - y)^2 + (x - z)^2 + (x - w)^2 + (y - z)^2 + (y - w)^2 + (z - w)^2 ≤ m) ∧
  (a - b)^2 + (a - c)^2 + (a - d)^2 + (b - c)^2 + (b - d)^2 + (c - d)^2 ≤ 40 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_of_squares_l118_11893


namespace NUMINAMATH_CALUDE_sunzi_carriage_problem_l118_11848

theorem sunzi_carriage_problem (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  (x / 3 = y + 2 ∧ x / 2 + 9 = y) ↔ (x / 3 = y - 2 ∧ (x - 9) / 2 = y) :=
by sorry

end NUMINAMATH_CALUDE_sunzi_carriage_problem_l118_11848


namespace NUMINAMATH_CALUDE_tommy_balloons_l118_11820

/-- Given that Tommy had 26 balloons initially and received 34 more from his mom,
    prove that he ended up with 60 balloons in total. -/
theorem tommy_balloons (initial_balloons : ℕ) (mom_gift : ℕ) : 
  initial_balloons = 26 → mom_gift = 34 → initial_balloons + mom_gift = 60 := by
sorry

end NUMINAMATH_CALUDE_tommy_balloons_l118_11820


namespace NUMINAMATH_CALUDE_incenter_position_l118_11856

-- Define a triangle PQR
structure Triangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ

-- Define the side lengths
def side_lengths (t : Triangle) : ℝ × ℝ × ℝ :=
  (11, 5, 8)

-- Define the incenter of a triangle
def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Theorem stating the position of the incenter
theorem incenter_position (t : Triangle) :
  let (p, q, r) := side_lengths t
  let J := incenter t
  J = (11/24 * t.P.1 + 5/24 * t.Q.1 + 8/24 * t.R.1,
       11/24 * t.P.2 + 5/24 * t.Q.2 + 8/24 * t.R.2) :=
by sorry

end NUMINAMATH_CALUDE_incenter_position_l118_11856


namespace NUMINAMATH_CALUDE_annulus_area_l118_11815

/-- The area of an annulus formed by two concentric circles. -/
theorem annulus_area (b c a : ℝ) (h1 : b > c) (h2 : b^2 = c^2 + a^2) :
  (π * b^2 - π * c^2) = π * a^2 := by sorry

end NUMINAMATH_CALUDE_annulus_area_l118_11815


namespace NUMINAMATH_CALUDE_compton_basketball_league_members_l118_11871

theorem compton_basketball_league_members : 
  let sock_cost : ℚ := 4
  let tshirt_cost : ℚ := sock_cost + 6
  let cap_cost : ℚ := tshirt_cost - 3
  let member_cost : ℚ := 2 * (sock_cost + tshirt_cost + cap_cost)
  let total_expenditure : ℚ := 3144
  (total_expenditure / member_cost : ℚ) = 75 := by
  sorry

end NUMINAMATH_CALUDE_compton_basketball_league_members_l118_11871


namespace NUMINAMATH_CALUDE_carla_initial_marbles_l118_11880

/-- The number of marbles Carla bought -/
def marbles_bought : ℕ := 134

/-- The number of marbles Carla has now -/
def marbles_now : ℕ := 187

/-- The number of marbles Carla started with -/
def marbles_start : ℕ := marbles_now - marbles_bought

theorem carla_initial_marbles : marbles_start = 53 := by
  sorry

end NUMINAMATH_CALUDE_carla_initial_marbles_l118_11880


namespace NUMINAMATH_CALUDE_amount_ratio_l118_11884

theorem amount_ratio (total : ℚ) (r_amount : ℚ) 
  (h1 : total = 9000)
  (h2 : r_amount = 3600.0000000000005) :
  r_amount / (total - r_amount) = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_amount_ratio_l118_11884


namespace NUMINAMATH_CALUDE_coefficient_is_200_l118_11810

/-- The coefficient of x^4 in the expansion of (1+x^3)(1-x)^10 -/
def coefficientOfX4 : ℕ :=
  (Nat.choose 10 4) - (Nat.choose 10 1)

/-- Theorem stating that the coefficient of x^4 in the expansion of (1+x^3)(1-x)^10 is 200 -/
theorem coefficient_is_200 : coefficientOfX4 = 200 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_is_200_l118_11810


namespace NUMINAMATH_CALUDE_smallest_product_l118_11862

def digits : List Nat := [3, 4, 5, 6]

def valid_arrangement (a b c d : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def product (a b c d : Nat) : Nat :=
  (10 * a + b) * (10 * c + d)

theorem smallest_product :
  ∀ a b c d : Nat,
    valid_arrangement a b c d →
    product a b c d ≥ 1610 :=
by sorry

end NUMINAMATH_CALUDE_smallest_product_l118_11862


namespace NUMINAMATH_CALUDE_smallest_cube_ending_576_l118_11839

theorem smallest_cube_ending_576 : 
  ∀ n : ℕ, n > 0 → n < 706 → n^3 % 1000 ≠ 576 ∧ 706^3 % 1000 = 576 := by
  sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_576_l118_11839


namespace NUMINAMATH_CALUDE_count_positive_integers_l118_11897

def given_numbers : List ℚ := [-2, -1, 0, -1/2, 2, 1/3]

theorem count_positive_integers (n : List ℚ := given_numbers) :
  (n.filter (λ x => x > 0 ∧ x.den = 1)).length = 1 :=
sorry

end NUMINAMATH_CALUDE_count_positive_integers_l118_11897


namespace NUMINAMATH_CALUDE_roots_sum_to_four_l118_11868

theorem roots_sum_to_four : ∃ (x y : ℝ), x^2 - 4*x - 1 = 0 ∧ y^2 - 4*y - 1 = 0 ∧ x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_to_four_l118_11868


namespace NUMINAMATH_CALUDE_parallelogram_area_l118_11814

/-- The area of a parallelogram with one angle of 150 degrees and two consecutive sides of lengths 10 and 12 is 60 square units. -/
theorem parallelogram_area (a b : ℝ) (angle : ℝ) (h1 : a = 10) (h2 : b = 12) (h3 : angle = 150) :
  a * b * Real.sin (angle * π / 180) = 60 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l118_11814


namespace NUMINAMATH_CALUDE_unique_angle_sin_cos_l118_11885

theorem unique_angle_sin_cos :
  ∃! x : ℝ, 0 ≤ x ∧ x < π / 2 ∧ Real.sin x = 0.6 ∧ Real.cos x = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_unique_angle_sin_cos_l118_11885


namespace NUMINAMATH_CALUDE_factor_bound_l118_11857

/-- The number of ways to factor a positive integer into a product of integers greater than 1 -/
def f (k : ℕ) : ℕ := sorry

/-- Theorem: For any positive integer n > 1 and any prime factor p of n,
    the number of ways to factor n is less than or equal to n/p -/
theorem factor_bound {n p : ℕ} (h1 : n > 1) (h2 : p.Prime) (h3 : p ∣ n) : f n ≤ n / p := by
  sorry

end NUMINAMATH_CALUDE_factor_bound_l118_11857


namespace NUMINAMATH_CALUDE_probability_of_different_homes_l118_11803

def num_volunteers : ℕ := 5
def num_homes : ℕ := 2

def probability_different_homes : ℚ := 8/15

theorem probability_of_different_homes :
  let total_arrangements := (2^num_volunteers - 2)
  let arrangements_same_home := (2^(num_volunteers - 2) - 1) * 2
  (total_arrangements - arrangements_same_home : ℚ) / total_arrangements = probability_different_homes :=
sorry

end NUMINAMATH_CALUDE_probability_of_different_homes_l118_11803


namespace NUMINAMATH_CALUDE_revenue_change_l118_11806

theorem revenue_change (R : ℝ) (p : ℝ) (h1 : R > 0) :
  (R + p / 100 * R) * (1 - p / 100) = R * (1 - 4 / 100) →
  p = 20 := by
sorry

end NUMINAMATH_CALUDE_revenue_change_l118_11806


namespace NUMINAMATH_CALUDE_franklin_gathering_handshakes_l118_11800

/-- Represents a gathering of married couples -/
structure Gathering where
  couples : Nat
  men : Nat
  women : Nat
  total_people : Nat

/-- Calculates the number of handshakes in the gathering -/
def handshakes (g : Gathering) : Nat :=
  let men_handshakes := g.men.choose 2
  let men_women_handshakes := g.men * (g.women - 1)
  men_handshakes + men_women_handshakes

theorem franklin_gathering_handshakes :
  ∀ g : Gathering,
    g.couples = 15 →
    g.men = g.couples →
    g.women = g.couples →
    g.total_people = g.men + g.women →
    handshakes g = 315 := by
  sorry

#eval handshakes { couples := 15, men := 15, women := 15, total_people := 30 }

end NUMINAMATH_CALUDE_franklin_gathering_handshakes_l118_11800


namespace NUMINAMATH_CALUDE_festival_groups_l118_11834

theorem festival_groups (n : ℕ) (h : n = 7) : 
  (Nat.choose n 4 = 35) ∧ (Nat.choose n 3 = 35) := by
  sorry

#check festival_groups

end NUMINAMATH_CALUDE_festival_groups_l118_11834


namespace NUMINAMATH_CALUDE_range_of_sum_l118_11831

theorem range_of_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + 2*x*y + 4*y^2 = 1) :
  0 < x + y ∧ x + y < 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_sum_l118_11831


namespace NUMINAMATH_CALUDE_candy_soda_price_before_increase_l118_11826

theorem candy_soda_price_before_increase 
  (candy_price_after : ℝ) 
  (soda_price_after : ℝ) 
  (candy_increase_rate : ℝ) 
  (soda_increase_rate : ℝ) 
  (h1 : candy_price_after = 15) 
  (h2 : soda_price_after = 6) 
  (h3 : candy_increase_rate = 0.25) 
  (h4 : soda_increase_rate = 0.5) : 
  candy_price_after / (1 + candy_increase_rate) + 
  soda_price_after / (1 + soda_increase_rate) = 21 := by
  sorry

#check candy_soda_price_before_increase

end NUMINAMATH_CALUDE_candy_soda_price_before_increase_l118_11826


namespace NUMINAMATH_CALUDE_eutectic_alloy_mixture_l118_11824

/-- Represents the composition of an alloy --/
structure Alloy where
  pb : Real  -- Percentage of lead
  sn : Real  -- Percentage of tin
  mass : Real  -- Mass in grams

/-- Checks if an alloy's composition is valid (sums to 100%) --/
def Alloy.isValid (a : Alloy) : Prop :=
  a.pb + a.sn = 100 ∧ a.pb ≥ 0 ∧ a.sn ≥ 0 ∧ a.mass > 0

theorem eutectic_alloy_mixture 
  (alloy1 : Alloy) 
  (alloy2 : Alloy) 
  (eutecticAlloy : Alloy) :
  alloy1.isValid ∧ 
  alloy2.isValid ∧ 
  eutecticAlloy.isValid ∧
  alloy1.pb = 25 ∧ 
  alloy2.pb = 60 ∧ 
  eutecticAlloy.pb = 36 ∧
  alloy1.mass = 685.71 ∧ 
  alloy2.mass = 314.29 ∧
  eutecticAlloy.mass = 1000 →
  (alloy1.mass * alloy1.pb + alloy2.mass * alloy2.pb) / 100 = 
    eutecticAlloy.mass * eutecticAlloy.pb / 100 ∧
  alloy1.mass + alloy2.mass = eutecticAlloy.mass :=
by sorry

end NUMINAMATH_CALUDE_eutectic_alloy_mixture_l118_11824


namespace NUMINAMATH_CALUDE_breakfast_consumption_l118_11821

/-- Represents the number of slices of bread each member consumes during breakfast -/
def breakfast_slices : ℕ := 3

/-- Represents the number of members in the household -/
def household_members : ℕ := 4

/-- Represents the number of slices each member consumes for snacks -/
def snack_slices : ℕ := 2

/-- Represents the number of slices in a loaf of bread -/
def slices_per_loaf : ℕ := 12

/-- Represents the number of loaves that last for 3 days -/
def loaves_for_three_days : ℕ := 5

/-- Represents the number of days the loaves last -/
def days_lasted : ℕ := 3

theorem breakfast_consumption :
  breakfast_slices = 3 ∧
  household_members * (breakfast_slices + snack_slices) * days_lasted = 
  loaves_for_three_days * slices_per_loaf := by
  sorry

#check breakfast_consumption

end NUMINAMATH_CALUDE_breakfast_consumption_l118_11821


namespace NUMINAMATH_CALUDE_election_votes_proof_l118_11840

/-- The total number of votes in a school election where Emily received 45 votes, 
    which accounted for 25% of the total votes. -/
def total_votes : ℕ := 180

/-- Emily's votes in the election -/
def emily_votes : ℕ := 45

/-- The percentage of total votes that Emily received -/
def emily_percentage : ℚ := 25 / 100

theorem election_votes_proof : 
  total_votes = emily_votes / emily_percentage :=
by sorry

end NUMINAMATH_CALUDE_election_votes_proof_l118_11840


namespace NUMINAMATH_CALUDE_no_positive_roots_when_m_is_three_positive_root_exists_when_m_not_three_unique_m_with_no_positive_roots_l118_11838

/-- The equation has no positive real roots when m = 3 -/
theorem no_positive_roots_when_m_is_three (x : ℝ) : 
  (x^2 + (5 - 2*3)*x + 3 - 3) / (x - 1) ≠ 2*x + 3 ∨ x ≤ 0 :=
sorry

/-- For any m ≠ 3, the equation has at least one positive real root -/
theorem positive_root_exists_when_m_not_three (m : ℝ) (hm : m ≠ 3) :
  ∃ x > 0, (x^2 + (5 - 2*m)*x + m - 3) / (x - 1) = 2*x + m :=
sorry

/-- The only value of m for which the equation has no positive real roots is 3 -/
theorem unique_m_with_no_positive_roots :
  ∃! m : ℝ, ∀ x > 0, (x^2 + (5 - 2*m)*x + m - 3) / (x - 1) ≠ 2*x + m :=
sorry

end NUMINAMATH_CALUDE_no_positive_roots_when_m_is_three_positive_root_exists_when_m_not_three_unique_m_with_no_positive_roots_l118_11838


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l118_11874

/-- An arithmetic-geometric sequence -/
def ArithGeomSeq (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem arithmetic_geometric_sequence_problem (a : ℕ → ℝ) 
    (h_seq : ArithGeomSeq a)
    (h_first : a 1 = 3)
    (h_sum : a 1 + a 3 + a 5 = 21) :
    a 2 * a 6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l118_11874


namespace NUMINAMATH_CALUDE_certain_number_problem_l118_11833

theorem certain_number_problem : 
  ∃ x : ℝ, 0.60 * x = 0.42 * 30 + 17.4 ∧ x = 50 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l118_11833


namespace NUMINAMATH_CALUDE_one_li_equals_150_zhang_l118_11898

-- Define the conversion ratios
def meter_to_chi : ℚ := 3
def zhang_to_chi : ℚ := 10
def kilometer_to_li : ℚ := 2

-- Define the relationship between li and zhang
def li_to_zhang (li : ℚ) : ℚ := 
  li * (1000 / kilometer_to_li) * meter_to_chi / zhang_to_chi

-- Theorem statement
theorem one_li_equals_150_zhang : 
  li_to_zhang 1 = 150 := by sorry

end NUMINAMATH_CALUDE_one_li_equals_150_zhang_l118_11898


namespace NUMINAMATH_CALUDE_union_equality_iff_a_in_range_l118_11841

-- Define the sets M and N
def M (a : ℝ) : Set ℝ := {x | x * (x - a - 1) < 0}
def N : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- State the theorem
theorem union_equality_iff_a_in_range (a : ℝ) :
  M a ∪ N = N ↔ a ∈ Set.Icc (-2) 2 := by sorry

end NUMINAMATH_CALUDE_union_equality_iff_a_in_range_l118_11841


namespace NUMINAMATH_CALUDE_sin_210_degrees_l118_11892

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l118_11892


namespace NUMINAMATH_CALUDE_triangle_angle_B_l118_11809

theorem triangle_angle_B (A B C : Real) (a b : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- Side lengths
  a = 4 ∧ b = 5 →
  -- Given condition
  Real.cos (B + C) + 3/5 = 0 →
  -- Conclusion: Measure of angle B
  B = Real.pi - Real.arccos (3/5) := by sorry

end NUMINAMATH_CALUDE_triangle_angle_B_l118_11809


namespace NUMINAMATH_CALUDE_guitar_picks_l118_11835

theorem guitar_picks (total : ℕ) (red blue yellow : ℕ) : 
  2 * red = total →
  3 * blue = total →
  red + blue + yellow = total →
  blue = 12 →
  yellow = 6 := by
sorry

end NUMINAMATH_CALUDE_guitar_picks_l118_11835


namespace NUMINAMATH_CALUDE_family_pizza_order_l118_11846

/-- Calculates the number of pizzas needed for a family -/
def pizzas_needed (adults : ℕ) (children : ℕ) (adult_slices : ℕ) (child_slices : ℕ) (slices_per_pizza : ℕ) : ℕ :=
  ((adults * adult_slices + children * child_slices) + slices_per_pizza - 1) / slices_per_pizza

/-- Proves that a family of 2 adults and 6 children needs 3 pizzas -/
theorem family_pizza_order : pizzas_needed 2 6 3 1 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_family_pizza_order_l118_11846


namespace NUMINAMATH_CALUDE_opposite_terminal_sides_sin_equality_l118_11886

theorem opposite_terminal_sides_sin_equality (α β : Real) : 
  (∃ k : Int, β = α + (2 * k + 1) * Real.pi) → |Real.sin α| = |Real.sin β| := by
  sorry

end NUMINAMATH_CALUDE_opposite_terminal_sides_sin_equality_l118_11886


namespace NUMINAMATH_CALUDE_A_equals_set_l118_11863

def A : Set ℝ :=
  {x | ∃ a b c : ℝ, a * b * c ≠ 0 ∧ 
       x = a / |a| + |b| / b + |c| / c + (a * b * c) / |a * b * c|}

theorem A_equals_set : A = {-4, 0, 4} := by
  sorry

end NUMINAMATH_CALUDE_A_equals_set_l118_11863


namespace NUMINAMATH_CALUDE_f_lower_bound_a_range_l118_11801

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x + 2*a + 3|

-- Theorem 1: For all real x and a, f(x) ≥ 2
theorem f_lower_bound (x a : ℝ) : f x a ≥ 2 := by
  sorry

-- Theorem 2: If f(-3/2) < 3, then -1 < a < 0
theorem a_range (a : ℝ) : f (-3/2) a < 3 → -1 < a ∧ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_f_lower_bound_a_range_l118_11801


namespace NUMINAMATH_CALUDE_sales_theorem_l118_11816

def sales_problem (sales1 sales2 sales3 sales5 sales6 : ℕ) (average : ℕ) : Prop :=
  let total_sales := average * 6
  let known_sales := sales1 + sales2 + sales3 + sales5 + sales6
  let sales4 := total_sales - known_sales
  sales4 = 11707

theorem sales_theorem :
  sales_problem 5266 5768 5922 6029 4937 5600 :=
by
  sorry

end NUMINAMATH_CALUDE_sales_theorem_l118_11816


namespace NUMINAMATH_CALUDE_intersection_point_is_solution_l118_11818

/-- The intersection point of two lines -/
def intersection_point : ℝ × ℝ := (3.5, -1.25)

/-- The first line equation -/
def line1 (x y : ℝ) : Prop := 5 * x - 2 * y = 20

/-- The second line equation -/
def line2 (x y : ℝ) : Prop := 3 * x + 2 * y = 8

theorem intersection_point_is_solution :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y ∧
  ∀ x' y', line1 x' y' ∧ line2 x' y' → x' = x ∧ y' = y :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_is_solution_l118_11818


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l118_11844

theorem cubic_equation_roots (k m : ℝ) : 
  (∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    ∀ x : ℝ, x^3 - 11*x^2 + k*x - m = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  k + m = 52 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l118_11844


namespace NUMINAMATH_CALUDE_always_odd_l118_11808

theorem always_odd (p m : ℤ) (h : Odd p) : Odd (p^2 + 2*m*p) := by
  sorry

end NUMINAMATH_CALUDE_always_odd_l118_11808


namespace NUMINAMATH_CALUDE_cost_of_second_box_l118_11822

/-- The cost of cards in the first box -/
def cost_box1 : ℚ := 1.25

/-- The number of cards bought from each box -/
def cards_bought : ℕ := 6

/-- The total amount spent -/
def total_spent : ℚ := 18

/-- The cost of cards in the second box -/
def cost_box2 : ℚ := (total_spent - cards_bought * cost_box1) / cards_bought

theorem cost_of_second_box : cost_box2 = 1.75 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_second_box_l118_11822
