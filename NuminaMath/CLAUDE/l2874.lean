import Mathlib

namespace tangent_line_to_circle_l2874_287465

theorem tangent_line_to_circle (c : ℝ) : 
  (c > 0) → 
  (∀ x y : ℝ, x^2 + y^2 = 8 → (x + y = c → (x - 0)^2 + (y - 0)^2 = 8)) → 
  c = 4 := by
sorry

end tangent_line_to_circle_l2874_287465


namespace country_club_members_l2874_287494

def initial_fee : ℕ := 4000
def monthly_cost : ℕ := 1000
def john_payment : ℕ := 32000

theorem country_club_members : 
  ∀ (F : ℕ), 
    (F + 1) * (initial_fee + 12 * monthly_cost) / 2 = john_payment → 
    F = 3 :=
by sorry

end country_club_members_l2874_287494


namespace consecutive_even_numbers_sum_l2874_287406

/-- Given four consecutive even numbers whose sum of squares is 344, their sum is 36 -/
theorem consecutive_even_numbers_sum (n : ℕ) : 
  (n^2 + (n + 2)^2 + (n + 4)^2 + (n + 6)^2 = 344) → 
  (n + (n + 2) + (n + 4) + (n + 6) = 36) :=
by
  sorry

#check consecutive_even_numbers_sum

end consecutive_even_numbers_sum_l2874_287406


namespace ceiling_product_equation_l2874_287489

theorem ceiling_product_equation : ∃ x : ℝ, ⌈x⌉ * x = 156 ∧ x = 12 := by sorry

end ceiling_product_equation_l2874_287489


namespace total_bankers_discount_l2874_287423

/-- Represents a bill with its amount, true discount, and interest rate -/
structure Bill where
  amount : ℝ
  trueDiscount : ℝ
  interestRate : ℝ

/-- Calculates the banker's discount for a given bill -/
def bankerDiscount (bill : Bill) : ℝ :=
  (bill.amount - bill.trueDiscount) * bill.interestRate

/-- The four bills given in the problem -/
def bills : List Bill := [
  { amount := 2260, trueDiscount := 360, interestRate := 0.08 },
  { amount := 3280, trueDiscount := 520, interestRate := 0.10 },
  { amount := 4510, trueDiscount := 710, interestRate := 0.12 },
  { amount := 6240, trueDiscount := 980, interestRate := 0.15 }
]

/-- Theorem: The total banker's discount for the given bills is 1673 -/
theorem total_bankers_discount :
  (bills.map bankerDiscount).sum = 1673 := by
  sorry

end total_bankers_discount_l2874_287423


namespace specific_project_time_l2874_287484

/-- A project requires workers to complete it. The number of workers and time can change during the project. -/
structure Project where
  initial_workers : ℕ
  initial_days : ℕ
  additional_workers : ℕ
  days_before_addition : ℕ

/-- Calculate the total time required to complete the project -/
def total_time (p : Project) : ℕ :=
  sorry

/-- The specific project described in the problem -/
def specific_project : Project :=
  { initial_workers := 10
  , initial_days := 15
  , additional_workers := 5
  , days_before_addition := 5 }

/-- Theorem stating that the total time for the specific project is 6 days -/
theorem specific_project_time : total_time specific_project = 6 :=
  sorry

end specific_project_time_l2874_287484


namespace abs_sum_reciprocals_ge_two_l2874_287422

theorem abs_sum_reciprocals_ge_two (a b : ℝ) (h : a * b ≠ 0) :
  |a / b + b / a| ≥ 2 := by
  sorry

end abs_sum_reciprocals_ge_two_l2874_287422


namespace set_operations_l2874_287424

def A : Set ℝ := {x | 2 < x ∧ x < 6}
def B (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2*m}

theorem set_operations (m : ℝ) :
  (m = 2 → A ∪ B m = A) ∧
  (B m ⊆ A ↔ m ≤ 3) ∧
  (B m ≠ ∅ ∧ A ∩ B m = ∅ ↔ m ≥ 5) :=
by sorry

end set_operations_l2874_287424


namespace difference_of_differences_l2874_287401

theorem difference_of_differences (a b c : ℤ) 
  (h1 : a - b = 2) 
  (h2 : b - c = -3) : 
  a - c = -1 := by
  sorry

end difference_of_differences_l2874_287401


namespace at_least_one_less_than_or_equal_to_one_l2874_287464

theorem at_least_one_less_than_or_equal_to_one
  (x y z : ℝ)
  (pos_x : 0 < x)
  (pos_y : 0 < y)
  (pos_z : 0 < z)
  (sum_eq_three : x + y + z = 3) :
  min (x * (x + y - z)) (min (y * (y + z - x)) (z * (z + x - y))) ≤ 1 := by
  sorry

end at_least_one_less_than_or_equal_to_one_l2874_287464


namespace complex_power_2018_l2874_287446

theorem complex_power_2018 : (((1 - Complex.I) / (1 + Complex.I)) ^ 2018 : ℂ) = -1 := by
  sorry

end complex_power_2018_l2874_287446


namespace emily_beads_count_l2874_287426

/-- Given that Emily can make 4 necklaces and each necklace requires 7 beads,
    prove that she has 28 beads in total. -/
theorem emily_beads_count :
  ∀ (necklaces : ℕ) (beads_per_necklace : ℕ),
    necklaces = 4 →
    beads_per_necklace = 7 →
    necklaces * beads_per_necklace = 28 :=
by
  sorry

end emily_beads_count_l2874_287426


namespace mother_age_twice_alex_age_l2874_287483

/-- The year when Alex's mother's age will be twice his age -/
def target_year : ℕ := 2025

/-- Alex's birth year -/
def alex_birth_year : ℕ := 1997

/-- The year of Alex's 7th birthday -/
def seventh_birthday_year : ℕ := 2004

/-- Alex's age on his 7th birthday -/
def alex_age_seventh_birthday : ℕ := 7

/-- Alex's mother's age on Alex's 7th birthday -/
def mother_age_seventh_birthday : ℕ := 35

theorem mother_age_twice_alex_age :
  (target_year - seventh_birthday_year) + alex_age_seventh_birthday = 
  (target_year - seventh_birthday_year + mother_age_seventh_birthday) / 2 ∧
  mother_age_seventh_birthday = 5 * alex_age_seventh_birthday ∧
  seventh_birthday_year - alex_birth_year = alex_age_seventh_birthday :=
by sorry

end mother_age_twice_alex_age_l2874_287483


namespace ellipse_focal_property_l2874_287485

/-- The equation of the ellipse -/
def ellipse_eq (x y : ℝ) : Prop := x^2 / 16 + y^2 / 9 = 1

/-- The foci of the ellipse -/
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

/-- Points A and B on the ellipse -/
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem ellipse_focal_property :
  ellipse_eq A.1 A.2 ∧ 
  ellipse_eq B.1 B.2 ∧ 
  (∃ (t : ℝ), A = F₂ + t • (B - F₂) ∨ B = F₂ + t • (A - F₂)) ∧
  distance A B = 5 →
  distance A F₁ + distance B F₁ = 11 := by sorry

end ellipse_focal_property_l2874_287485


namespace point_coordinate_sum_l2874_287416

theorem point_coordinate_sum (a b : ℝ) : 
  (2 = b - 1 ∧ -1 = a + 3) → a + b = -1 := by
sorry

end point_coordinate_sum_l2874_287416


namespace unique_four_digit_number_l2874_287445

theorem unique_four_digit_number : ∃! n : ℕ,
  (1000 ≤ n) ∧ (n < 10000) ∧  -- 4-digit number
  (∃ a : ℕ, n = a^2) ∧  -- perfect square
  (∃ b : ℕ, n % 1000 = b^3) ∧  -- removing first digit results in a perfect cube
  (∃ c : ℕ, n % 100 = c^4) ∧  -- removing first two digits results in a fourth power
  n = 9216 :=
by sorry

end unique_four_digit_number_l2874_287445


namespace degrees_to_radians_conversion_l2874_287453

theorem degrees_to_radians_conversion (deg : ℝ) (rad : ℝ) : 
  deg = 50 → rad = deg * (π / 180) → rad = 5 * π / 18 := by
  sorry

end degrees_to_radians_conversion_l2874_287453


namespace chemical_mixture_problem_l2874_287495

theorem chemical_mixture_problem (original_conc : ℝ) (final_conc : ℝ) (replaced_portion : ℝ) 
  (h1 : original_conc = 0.9)
  (h2 : final_conc = 0.4)
  (h3 : replaced_portion = 0.7142857142857143) :
  let replacement_conc := (final_conc - original_conc * (1 - replaced_portion)) / replaced_portion
  replacement_conc = 0.2 := by
sorry

end chemical_mixture_problem_l2874_287495


namespace tangent_line_value_l2874_287469

/-- The value of a when the line 2x - y + 1 = 0 is tangent to the curve y = ae^x + x -/
theorem tangent_line_value (a : ℝ) : 
  (∃ x y : ℝ, 2*x - y + 1 = 0 ∧ y = a*(Real.exp x) + x ∧ 
    (∀ h : ℝ, h ≠ 0 → (a*(Real.exp (x + h)) + (x + h) - y) / h ≠ 2)) → 
  a = 1 :=
by sorry

end tangent_line_value_l2874_287469


namespace golf_course_distance_l2874_287400

/-- Represents a golf shot with distance and wind conditions -/
structure GolfShot where
  distance : ℝ
  windSpeed : ℝ
  windDirection : String

/-- Calculates the total distance to the hole given three golf shots -/
def distanceToHole (shot1 shot2 shot3 : GolfShot) (slopeEffect : ℝ) : ℝ :=
  shot1.distance + (shot2.distance - slopeEffect)

theorem golf_course_distance :
  let shot1 : GolfShot := { distance := 180, windSpeed := 10, windDirection := "tailwind" }
  let shot2 : GolfShot := { distance := 90, windSpeed := 7, windDirection := "crosswind" }
  let shot3 : GolfShot := { distance := 0, windSpeed := 5, windDirection := "headwind" }
  let slopeEffect : ℝ := 20
  distanceToHole shot1 shot2 shot3 slopeEffect = 270 := by
  sorry

end golf_course_distance_l2874_287400


namespace negation_of_proposition_l2874_287475

theorem negation_of_proposition (p : Prop) :
  (p ↔ ∃ x, x < -1 ∧ x^2 - x + 1 < 0) →
  (¬p ↔ ∀ x, x < -1 → x^2 - x + 1 ≥ 0) :=
by sorry

end negation_of_proposition_l2874_287475


namespace basketball_team_selection_l2874_287442

def total_players : ℕ := 16
def quadruplets : ℕ := 4
def players_to_select : ℕ := 7
def quadruplets_to_select : ℕ := 3

theorem basketball_team_selection :
  (Nat.choose quadruplets quadruplets_to_select) *
  (Nat.choose (total_players - quadruplets) (players_to_select - quadruplets_to_select)) = 1980 :=
by sorry

end basketball_team_selection_l2874_287442


namespace product_real_implies_b_value_l2874_287412

/-- Given complex numbers z₁ and z₂, if their product is real, then b = -2 -/
theorem product_real_implies_b_value (z₁ z₂ : ℂ) (b : ℝ) 
  (h₁ : z₁ = 1 + I) 
  (h₂ : z₂ = 2 + b * I) 
  (h₃ : (z₁ * z₂).im = 0) : 
  b = -2 := by
  sorry

end product_real_implies_b_value_l2874_287412


namespace students_in_one_language_class_l2874_287438

theorem students_in_one_language_class 
  (french_class : ℕ) 
  (spanish_class : ℕ) 
  (both_classes : ℕ) 
  (h1 : french_class = 21) 
  (h2 : spanish_class = 21) 
  (h3 : both_classes = 6) :
  french_class + spanish_class - 2 * both_classes = 36 := by
  sorry

end students_in_one_language_class_l2874_287438


namespace least_number_divisible_by_11_with_remainder_2_l2874_287462

def is_divisible_by_11 (n : ℕ) : Prop := ∃ k : ℕ, n = 11 * k

def leaves_remainder_2 (n : ℕ) (d : ℕ) : Prop := ∃ k : ℕ, n = d * k + 2

theorem least_number_divisible_by_11_with_remainder_2 : 
  (is_divisible_by_11 1262) ∧ 
  (∀ d : ℕ, 3 ≤ d → d ≤ 7 → leaves_remainder_2 1262 d) ∧
  (∀ m : ℕ, m < 1262 → 
    ¬(is_divisible_by_11 m ∧ 
      (∀ d : ℕ, 3 ≤ d → d ≤ 7 → leaves_remainder_2 m d))) :=
by sorry

end least_number_divisible_by_11_with_remainder_2_l2874_287462


namespace digit_sum_l2874_287411

theorem digit_sum (a b : ℕ) : 
  a < 10 → b < 10 → (32 * a + 300) * (10 * b + 4) = 1486 → a + b = 5 := by
  sorry

end digit_sum_l2874_287411


namespace appropriate_speech_lengths_l2874_287402

/-- Represents the duration of a speech in minutes -/
def SpeechDuration := { d : ℝ // 20 ≤ d ∧ d ≤ 40 }

/-- The recommended speech rate in words per minute -/
def SpeechRate : ℝ := 120

/-- Calculates the number of words for a given duration -/
def wordCount (d : SpeechDuration) : ℝ := d.val * SpeechRate

/-- Checks if a word count is appropriate for the speech -/
def isAppropriateLength (w : ℝ) : Prop :=
  ∃ (d : SpeechDuration), wordCount d = w

theorem appropriate_speech_lengths :
  isAppropriateLength 2500 ∧ 
  isAppropriateLength 3800 ∧ 
  isAppropriateLength 4600 := by sorry

end appropriate_speech_lengths_l2874_287402


namespace sum_of_roots_quadratic_l2874_287478

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 + 5*x₁ - 2 = 0) → (x₂^2 + 5*x₂ - 2 = 0) → (x₁ + x₂ = -5) := by
  sorry

end sum_of_roots_quadratic_l2874_287478


namespace problem_solution_l2874_287479

theorem problem_solution (x y z p q r : ℝ) 
  (h1 : x * y / (x + y) = p)
  (h2 : x * z / (x + z) = q)
  (h3 : y * z / (y + z) = r)
  (h4 : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)
  (h5 : x ≠ -y ∧ x ≠ -z ∧ y ≠ -z)
  (h6 : p = 3 * q)
  (h7 : p = 2 * r) :
  x = 3 * p / 2 := by
  sorry

end problem_solution_l2874_287479


namespace bella_steps_l2874_287474

/-- The distance between Bella's and Ella's houses in feet -/
def distance : ℝ := 10560

/-- Bella's step length in feet -/
def step_length : ℝ := 2.5

/-- The ratio of Ella's speed to Bella's speed -/
def speed_ratio : ℝ := 5

/-- Calculates the number of steps Bella takes before meeting Ella -/
def steps_taken : ℕ := sorry

/-- Theorem stating that Bella takes 704 steps before meeting Ella -/
theorem bella_steps : steps_taken = 704 := by sorry

end bella_steps_l2874_287474


namespace integer_fraction_count_l2874_287419

theorem integer_fraction_count : 
  ∃! (S : Finset ℕ), 
    (∀ n ∈ S, 0 < n ∧ n < 50 ∧ ∃ k : ℕ, n = k * (50 - n)) ∧ 
    Finset.card S = 2 := by sorry

end integer_fraction_count_l2874_287419


namespace opposite_of_three_l2874_287433

-- Define the opposite function for real numbers
def opposite (x : ℝ) : ℝ := -x

-- State the theorem
theorem opposite_of_three : opposite 3 = -3 := by sorry

end opposite_of_three_l2874_287433


namespace solve_cubic_equation_l2874_287466

theorem solve_cubic_equation (n : ℝ) :
  (n - 5) ^ 3 = (1 / 9)⁻¹ ↔ n = 5 + 3 ^ (2 / 3) :=
by sorry

end solve_cubic_equation_l2874_287466


namespace nearest_integer_to_x_minus_y_l2874_287443

theorem nearest_integer_to_x_minus_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : |x| + y = 5) (h2 : |x| * y - x^3 = 0) : x - y = 5 := by
  sorry

end nearest_integer_to_x_minus_y_l2874_287443


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l2874_287482

/-- Returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Returns true if the number is a three-digit number -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∀ n : ℕ, is_three_digit n → n % 9 = 0 → digit_sum n = 27 → n ≤ 999 :=
by sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l2874_287482


namespace smallest_earring_collection_l2874_287427

theorem smallest_earring_collection (M : ℕ) : 
  M > 2 ∧ 
  M % 7 = 2 ∧ 
  M % 11 = 2 ∧ 
  M % 13 = 2 → 
  M ≥ 1003 :=
by sorry

end smallest_earring_collection_l2874_287427


namespace shaded_area_percentage_l2874_287460

/-- Given two congruent squares ABCD and EFGH with side length 20 units that overlap
    to form a 20 by 35 rectangle AEGD, prove that 14% of AEGD's area is shaded. -/
theorem shaded_area_percentage (side_length : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) :
  side_length = 20 →
  rectangle_width = 20 →
  rectangle_length = 35 →
  (((2 * side_length - rectangle_length) * side_length) / (rectangle_width * rectangle_length)) * 100 = 14 := by
  sorry

end shaded_area_percentage_l2874_287460


namespace total_turnips_l2874_287468

theorem total_turnips (melanie_turnips benny_turnips : ℕ) 
  (h1 : melanie_turnips = 139) 
  (h2 : benny_turnips = 113) : 
  melanie_turnips + benny_turnips = 252 := by
  sorry

end total_turnips_l2874_287468


namespace cube_surface_area_l2874_287455

/-- The surface area of a cube with edge length 6 cm is 216 square centimeters. -/
theorem cube_surface_area : 
  let edge_length : ℝ := 6
  let surface_area := 6 * edge_length^2
  surface_area = 216 :=
by
  sorry

end cube_surface_area_l2874_287455


namespace decreasing_cubic_condition_l2874_287418

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 - x + 1

-- Define what it means for a function to be decreasing on ℝ
def IsDecreasing (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → g x > g y

-- Theorem statement
theorem decreasing_cubic_condition (a : ℝ) :
  IsDecreasing (f a) → a < -3 := by
  sorry

end decreasing_cubic_condition_l2874_287418


namespace largest_stamps_per_page_l2874_287457

theorem largest_stamps_per_page : Nat.gcd (Nat.gcd 1020 1275) 1350 = 15 := by
  sorry

end largest_stamps_per_page_l2874_287457


namespace complex_subtraction_l2874_287456

theorem complex_subtraction : (4 - 3*I) - (7 - 5*I) = -3 + 2*I := by
  sorry

end complex_subtraction_l2874_287456


namespace function_satisfying_conditions_l2874_287405

theorem function_satisfying_conditions : ∃ (f : ℝ → ℝ), 
  (∀ x, f (-x) = f x) ∧ 
  (∀ x, f (2 - x) + f x = 0) ∧ 
  (∀ x, f x = Real.cos (Real.pi / 2 * x)) ∧
  (∃ x y, f x ≠ f y) := by
  sorry

end function_satisfying_conditions_l2874_287405


namespace rectangle_area_l2874_287471

theorem rectangle_area (r : ℝ) (ratio : ℝ) : 
  r = 7 → ratio = 3 → 2 * r * (1 + ratio) = 588 := by
  sorry

end rectangle_area_l2874_287471


namespace environmental_policy_survey_l2874_287428

theorem environmental_policy_survey (group_a_size : ℕ) (group_b_size : ℕ) 
  (group_a_favor_percent : ℚ) (group_b_favor_percent : ℚ) : 
  group_a_size = 200 →
  group_b_size = 800 →
  group_a_favor_percent = 70 / 100 →
  group_b_favor_percent = 75 / 100 →
  (group_a_size * group_a_favor_percent + group_b_size * group_b_favor_percent) / 
  (group_a_size + group_b_size) = 74 / 100 := by
  sorry

end environmental_policy_survey_l2874_287428


namespace regular_polygon_with_150_degree_interior_angle_has_12_sides_l2874_287440

/-- A regular polygon with an interior angle of 150° has 12 sides -/
theorem regular_polygon_with_150_degree_interior_angle_has_12_sides :
  ∀ (n : ℕ) (interior_angle : ℝ),
    n ≥ 3 →
    interior_angle = 150 →
    (n - 2) * 180 = n * interior_angle →
    n = 12 := by
  sorry

end regular_polygon_with_150_degree_interior_angle_has_12_sides_l2874_287440


namespace log_of_negative_one_not_real_l2874_287491

/-- For b > 0 and b ≠ 1, log_b(-1) is not a real number -/
theorem log_of_negative_one_not_real (b : ℝ) (hb_pos : b > 0) (hb_ne_one : b ≠ 1) :
  ¬ ∃ (y : ℝ), b^y = -1 :=
sorry

end log_of_negative_one_not_real_l2874_287491


namespace binomial_expansion_special_case_l2874_287454

theorem binomial_expansion_special_case : 
  98^3 + 3*(98^2)*2 + 3*98*(2^2) + 2^3 = 1000000 := by
  sorry

end binomial_expansion_special_case_l2874_287454


namespace document_delivery_equation_l2874_287404

theorem document_delivery_equation (x : ℝ) (h : x > 3) : 
  let distance : ℝ := 900
  let slow_horse_time : ℝ := x + 1
  let fast_horse_time : ℝ := x - 3
  let slow_horse_speed : ℝ := distance / slow_horse_time
  let fast_horse_speed : ℝ := distance / fast_horse_time
  fast_horse_speed = 2 * slow_horse_speed →
  (distance / slow_horse_time) * 2 = distance / fast_horse_time :=
by sorry


end document_delivery_equation_l2874_287404


namespace abs_eq_neg_implies_nonpositive_l2874_287414

theorem abs_eq_neg_implies_nonpositive (a : ℝ) : |a| = -a → a ≤ 0 := by
  sorry

end abs_eq_neg_implies_nonpositive_l2874_287414


namespace largest_angle_is_right_angle_l2874_287486

/-- Given a triangle with sides a, b, and c, if its area is (a+b+c)(a+b-c)/4, 
    then its largest angle is 90°. -/
theorem largest_angle_is_right_angle 
  (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_area : (a + b + c) * (a + b - c) / 4 = (a + b + c) * (a + b - c) / 4) :
  ∃ θ : ℝ, θ = Real.pi / 2 ∧ θ = max (Real.arccos ((b^2 + c^2 - a^2) / (2*b*c))) 
                                    (max (Real.arccos ((a^2 + c^2 - b^2) / (2*a*c)))
                                         (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b)))) :=
by sorry

end largest_angle_is_right_angle_l2874_287486


namespace perfect_square_binomial_l2874_287403

theorem perfect_square_binomial (k : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 - 10*x + k = (x + a)^2) ↔ k = 25 := by
sorry

end perfect_square_binomial_l2874_287403


namespace power_tower_mod_1000_l2874_287458

theorem power_tower_mod_1000 : 7^(7^(7^7)) ≡ 343 [ZMOD 1000] := by
  sorry

end power_tower_mod_1000_l2874_287458


namespace average_weight_e_f_l2874_287493

theorem average_weight_e_f (d e f : ℝ) 
  (h1 : (d + e + f) / 3 = 42)
  (h2 : (d + e) / 2 = 35)
  (h3 : e = 26) :
  (e + f) / 2 = 41 := by
sorry

end average_weight_e_f_l2874_287493


namespace first_day_is_wednesday_l2874_287429

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a day in a month -/
structure DayInMonth where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Given that the 22nd day of a month is a Wednesday, 
    prove that the 1st day of that month is also a Wednesday -/
theorem first_day_is_wednesday 
  (h : ∃ (m : List DayInMonth), 
    m.length = 31 ∧ 
    (∃ (d : DayInMonth), d ∈ m ∧ d.day = 22 ∧ d.dayOfWeek = DayOfWeek.Wednesday)) :
  ∃ (m : List DayInMonth),
    m.length = 31 ∧
    (∃ (d : DayInMonth), d ∈ m ∧ d.day = 1 ∧ d.dayOfWeek = DayOfWeek.Wednesday) :=
by sorry

end first_day_is_wednesday_l2874_287429


namespace other_asymptote_equation_l2874_287473

/-- Represents a hyperbola -/
structure Hyperbola where
  /-- One of the asymptotes of the hyperbola -/
  asymptote1 : ℝ → ℝ
  /-- x-coordinate of the foci -/
  foci_x : ℝ

/-- Theorem: Given a hyperbola with one asymptote y = 2x and foci x-coordinate 4,
    the other asymptote has equation y = -0.5x + 10 -/
theorem other_asymptote_equation (h : Hyperbola) 
    (h1 : h.asymptote1 = fun x => 2 * x) 
    (h2 : h.foci_x = 4) : 
    ∃ asymptote2 : ℝ → ℝ, asymptote2 = fun x => -0.5 * x + 10 := by
  sorry

end other_asymptote_equation_l2874_287473


namespace log_inequality_conditions_l2874_287431

/-- The set of positive real numbers excluding 1 -/
def S : Set ℝ := {x : ℝ | x > 0 ∧ x ≠ 1}

/-- The theorem stating the conditions for the logarithmic inequality -/
theorem log_inequality_conditions (a b : ℝ) :
  a > 0 → b > 0 → a ≠ 1 →
  (Real.log b / Real.log a < Real.log (b + 1) / Real.log (a + 1)) ↔
  ((b = 1 ∧ a ∈ S) ∨
   (a > b ∧ b > 1) ∨
   (b > 1 ∧ 1 > a) ∨
   (a < b ∧ b < 1) ∨
   (b < 1 ∧ 1 < a)) :=
by sorry

end log_inequality_conditions_l2874_287431


namespace balls_after_1500_steps_l2874_287409

/-- Represents the state of boxes with balls -/
def BoxState := List Nat

/-- Converts a natural number to its base-4 representation -/
def toBase4 (n : Nat) : List Nat :=
  sorry

/-- Sums the digits in a list -/
def sumDigits (digits : List Nat) : Nat :=
  sorry

/-- Simulates the ball placement process for a given number of steps -/
def simulateBallPlacement (steps : Nat) : BoxState :=
  sorry

/-- Counts the total number of balls in a BoxState -/
def countBalls (state : BoxState) : Nat :=
  sorry

/-- Theorem stating that the number of balls after 1500 steps
    is equal to the sum of digits of 1500 in base-4 -/
theorem balls_after_1500_steps :
  countBalls (simulateBallPlacement 1500) = sumDigits (toBase4 1500) :=
sorry

end balls_after_1500_steps_l2874_287409


namespace find_number_l2874_287459

theorem find_number : ∃ x : ℤ, x - 263 + 419 = 725 ∧ x = 569 := by sorry

end find_number_l2874_287459


namespace geometric_sequence_minimum_l2874_287470

theorem geometric_sequence_minimum (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_geom : ∃ r > 0, ∀ n, a (n + 1) = a n * r) (h_prod : a 5 * a 6 = 16) :
  (∀ x, a 2 + a 9 ≥ x) → x ≤ 8 :=
sorry

end geometric_sequence_minimum_l2874_287470


namespace B_coordinates_l2874_287421

/-- Represents a point in 2D space -/
structure Point where
  x : ℤ
  y : ℤ

/-- Moves a point up by a given number of units -/
def moveUp (p : Point) (units : ℤ) : Point :=
  { x := p.x, y := p.y + units }

/-- Moves a point left by a given number of units -/
def moveLeft (p : Point) (units : ℤ) : Point :=
  { x := p.x - units, y := p.y }

/-- The initial point A -/
def A : Point := { x := -3, y := -5 }

/-- The final point B after moving A -/
def B : Point := moveLeft (moveUp A 4) 3

/-- Theorem stating that B has the correct coordinates -/
theorem B_coordinates : B.x = -6 ∧ B.y = -1 := by sorry

end B_coordinates_l2874_287421


namespace domino_tiling_theorem_l2874_287490

/-- Represents a rectangular grid -/
structure Rectangle where
  m : ℕ
  n : ℕ

/-- Represents a domino tile -/
structure Domino where
  length : ℕ
  width : ℕ

/-- Represents a tiling of a rectangle with dominoes -/
def Tiling (r : Rectangle) (d : Domino) := Unit

/-- Predicate to check if a tiling has no straight cuts -/
def has_no_straight_cuts (t : Tiling r d) : Prop := sorry

theorem domino_tiling_theorem :
  /- Part a -/
  (∀ (r : Rectangle) (d : Domino), r.m = 6 ∧ r.n = 6 ∧ d.length = 1 ∧ d.width = 2 →
    ¬ ∃ (t : Tiling r d), has_no_straight_cuts t) ∧
  /- Part b -/
  (∀ (r : Rectangle) (d : Domino), r.m > 6 ∧ r.n > 6 ∧ (r.m * r.n) % 2 = 0 ∧ d.length = 1 ∧ d.width = 2 →
    ∃ (t : Tiling r d), has_no_straight_cuts t) ∧
  /- Part c -/
  (∃ (r : Rectangle) (d : Domino) (t : Tiling r d), r.m = 6 ∧ r.n = 8 ∧ d.length = 1 ∧ d.width = 2 ∧
    has_no_straight_cuts t) :=
by sorry

end domino_tiling_theorem_l2874_287490


namespace lizard_wrinkle_eye_ratio_l2874_287476

theorem lizard_wrinkle_eye_ratio :
  ∀ (W : ℕ) (S : ℕ),
    S = 7 * W →
    3 = S + W - 69 →
    (W : ℚ) / 3 = 3 :=
by
  sorry

end lizard_wrinkle_eye_ratio_l2874_287476


namespace pythagoras_students_l2874_287497

theorem pythagoras_students : ∃ n : ℕ, 
  n > 0 ∧ 
  (n / 2 : ℚ) + (n / 4 : ℚ) + (n / 7 : ℚ) + 3 = n ∧ 
  n = 28 := by
  sorry

end pythagoras_students_l2874_287497


namespace sheets_exceed_500_at_step_31_l2874_287410

def sheets_after_steps (initial_sheets : ℕ) (steps : ℕ) : ℕ :=
  initial_sheets + steps * (steps + 1) / 2

theorem sheets_exceed_500_at_step_31 :
  sheets_after_steps 10 31 > 500 ∧ sheets_after_steps 10 30 ≤ 500 := by
  sorry

end sheets_exceed_500_at_step_31_l2874_287410


namespace vacation_savings_proof_l2874_287481

/-- Calculates the amount to save per paycheck given a savings goal, time frame, and number of paychecks per month. -/
def amount_per_paycheck (savings_goal : ℚ) (months : ℕ) (paychecks_per_month : ℕ) : ℚ :=
  savings_goal / (months * paychecks_per_month)

/-- Proves that given a savings goal of $3,000.00 over 15 months with 2 paychecks per month, 
    the amount to save per paycheck is $100.00. -/
theorem vacation_savings_proof : 
  amount_per_paycheck 3000 15 2 = 100 := by
sorry

end vacation_savings_proof_l2874_287481


namespace toad_frog_percentage_increase_l2874_287452

/-- Represents the number of bugs eaten by each animal -/
structure BugsEaten where
  gecko : ℕ
  lizard : ℕ
  frog : ℕ
  toad : ℕ

/-- Conditions from the problem -/
def garden_conditions (b : BugsEaten) : Prop :=
  b.gecko = 12 ∧
  b.lizard = b.gecko / 2 ∧
  b.frog = 3 * b.lizard ∧
  b.gecko + b.lizard + b.frog + b.toad = 63

/-- Calculate percentage increase -/
def percentage_increase (old_value new_value : ℕ) : ℚ :=
  (new_value - old_value : ℚ) / old_value * 100

/-- Theorem stating the percentage increase in bugs eaten by toad compared to frog -/
theorem toad_frog_percentage_increase (b : BugsEaten) 
  (h : garden_conditions b) : 
  percentage_increase b.frog b.toad = 50 := by
  sorry

end toad_frog_percentage_increase_l2874_287452


namespace athlete_heartbeats_l2874_287437

/-- The number of heartbeats during a race -/
def heartbeats_during_race (heart_rate : ℕ) (pace : ℕ) (distance : ℕ) : ℕ :=
  heart_rate * pace * distance

/-- Proof that the athlete's heart beats 25200 times during the race -/
theorem athlete_heartbeats :
  heartbeats_during_race 140 6 30 = 25200 := by
  sorry

#eval heartbeats_during_race 140 6 30

end athlete_heartbeats_l2874_287437


namespace problem_statement_l2874_287451

theorem problem_statement (x y : ℝ) (h : x^2 + y^2 - x*y = 1) : 
  (x + y ≥ -2) ∧ (x^2 + y^2 ≤ 2) := by
  sorry

end problem_statement_l2874_287451


namespace unique_positive_solution_l2874_287436

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ (x - 5) / 12 = 5 / (x - 12) :=
by
  -- The proof goes here
  sorry

end unique_positive_solution_l2874_287436


namespace group_purchase_equations_l2874_287425

/-- Represents a group purchase scenario -/
structure GroupPurchase where
  x : ℕ  -- number of people
  y : ℕ  -- price of the item

/-- Defines the conditions of the group purchase -/
def validGroupPurchase (gp : GroupPurchase) : Prop :=
  (9 * gp.x - gp.y = 4) ∧ (gp.y - 6 * gp.x = 5)

/-- Theorem stating that the given system of equations correctly represents the group purchase scenario -/
theorem group_purchase_equations (gp : GroupPurchase) : 
  validGroupPurchase gp ↔ (9 * gp.x - gp.y = 4 ∧ gp.y - 6 * gp.x = 5) :=
sorry

end group_purchase_equations_l2874_287425


namespace max_b_rectangular_prism_l2874_287450

theorem max_b_rectangular_prism (a b c : ℕ) : 
  (a * b * c = 360) →
  (1 < c) →
  (c < b) →
  (b < a) →
  (∀ a' b' c' : ℕ, (a' * b' * c' = 360) → (1 < c') → (c' < b') → (b' < a') → b' ≤ b) →
  b = 12 :=
by sorry

end max_b_rectangular_prism_l2874_287450


namespace rational_division_equality_l2874_287420

theorem rational_division_equality : 
  (-2 / 21) / (1 / 6 - 3 / 14 + 2 / 3 - 9 / 7) = 1 / 7 := by sorry

end rational_division_equality_l2874_287420


namespace integer_solution_inequality_system_l2874_287441

theorem integer_solution_inequality_system : 
  ∃! x : ℤ, 2 * x ≤ 1 ∧ x + 2 > 1 :=
by
  -- The proof goes here
  sorry

end integer_solution_inequality_system_l2874_287441


namespace fourth_child_age_is_eight_l2874_287415

/-- The age of the first child -/
def first_child_age : ℕ := 15

/-- The age difference between the first and second child -/
def age_diff_first_second : ℕ := 1

/-- The age of the second child when the third child was born -/
def second_child_age_at_third_birth : ℕ := 4

/-- The age difference between the third and fourth child -/
def age_diff_third_fourth : ℕ := 2

/-- The age of the fourth child -/
def fourth_child_age : ℕ := first_child_age - age_diff_first_second - second_child_age_at_third_birth - age_diff_third_fourth

theorem fourth_child_age_is_eight : fourth_child_age = 8 := by
  sorry

end fourth_child_age_is_eight_l2874_287415


namespace pencil_cost_calculation_l2874_287430

/-- The original cost of a pencil before discount -/
def original_cost : ℝ := 4.00

/-- The discount applied to the pencil -/
def discount : ℝ := 0.63

/-- The final price of the pencil after discount -/
def final_price : ℝ := 3.37

/-- Theorem stating that the original cost minus the discount equals the final price -/
theorem pencil_cost_calculation : original_cost - discount = final_price := by
  sorry

end pencil_cost_calculation_l2874_287430


namespace sin_difference_l2874_287487

theorem sin_difference (A B : ℝ) : 
  Real.sin (A - B) = Real.sin A * Real.cos B - Real.cos A * Real.sin B := by
  sorry

end sin_difference_l2874_287487


namespace geometric_progression_solution_l2874_287492

theorem geometric_progression_solution :
  ∀ (a b c d : ℝ),
  (∃ (q : ℝ), b = a * q ∧ c = a * q^2 ∧ d = a * q^3) →
  a + d = -49 →
  b + c = 14 →
  ((a = 7 ∧ b = -14 ∧ c = 28 ∧ d = -56) ∨
   (a = -56 ∧ b = 28 ∧ c = -14 ∧ d = 7)) :=
by sorry

end geometric_progression_solution_l2874_287492


namespace hypotenuse_segment_ratio_l2874_287448

/-- A right triangle with leg lengths in ratio 3:4 -/
structure RightTriangle where
  a : ℝ  -- length of first leg
  b : ℝ  -- length of second leg
  h : ℝ  -- ratio of legs is 3:4
  leg_ratio : b = (4/3) * a

/-- The segments of the hypotenuse created by the altitude -/
structure HypotenuseSegments where
  x : ℝ  -- length of first segment
  y : ℝ  -- length of second segment

/-- Theorem: The ratio of hypotenuse segments is 21:16 -/
theorem hypotenuse_segment_ratio (t : RightTriangle) (s : HypotenuseSegments) :
  s.y / s.x = 21 / 16 :=
sorry

end hypotenuse_segment_ratio_l2874_287448


namespace three_a_equals_30_l2874_287488

theorem three_a_equals_30 
  (h1 : 3 * a - 2 * b - 2 * c = 30)
  (h2 : Real.sqrt (3 * a) - Real.sqrt (2 * b + 2 * c) = 4)
  (h3 : a + b + c = 10)
  : 3 * a = 30 := by
  sorry

end three_a_equals_30_l2874_287488


namespace range_of_a_l2874_287434

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + (2-a) = 0) → 
  a ≤ -2 ∨ a = 1 :=
by sorry

end range_of_a_l2874_287434


namespace sqrt_3_expression_simplification_l2874_287499

theorem sqrt_3_expression_simplification :
  Real.sqrt 3 * (Real.sqrt 3 - 2) - Real.sqrt 12 / Real.sqrt 3 + |2 - Real.sqrt 3| = 3 - 3 * Real.sqrt 3 := by
  sorry

end sqrt_3_expression_simplification_l2874_287499


namespace total_food_consumption_l2874_287444

/-- The amount of food needed per soldier per day on the first side -/
def food_per_soldier_first : ℕ := 10

/-- The amount of food needed per soldier per day on the second side -/
def food_per_soldier_second : ℕ := food_per_soldier_first - 2

/-- The number of soldiers on the first side -/
def soldiers_first : ℕ := 4000

/-- The number of soldiers on the second side -/
def soldiers_second : ℕ := soldiers_first - 500

/-- The total amount of food consumed by both sides per day -/
def total_food : ℕ := soldiers_first * food_per_soldier_first + soldiers_second * food_per_soldier_second

theorem total_food_consumption :
  total_food = 68000 := by sorry

end total_food_consumption_l2874_287444


namespace no_integer_solutions_l2874_287447

theorem no_integer_solutions : 
  ¬∃ (x y z : ℤ), 
    (x^2 - 3*x*y + 2*y^2 - z^2 = 31) ∧ 
    (-x^2 + 6*y*z + 2*z^2 = 44) ∧ 
    (x^2 + x*y + 8*z^2 = 100) := by
  sorry

end no_integer_solutions_l2874_287447


namespace aquarium_trainers_l2874_287496

/-- The number of trainers required to equally split the total training hours for all dolphins -/
def number_of_trainers (num_dolphins : ℕ) (hours_per_dolphin : ℕ) (hours_per_trainer : ℕ) : ℕ :=
  (num_dolphins * hours_per_dolphin) / hours_per_trainer

theorem aquarium_trainers :
  number_of_trainers 4 3 6 = 2 := by
  sorry

end aquarium_trainers_l2874_287496


namespace abs_one_point_five_minus_sqrt_two_l2874_287467

theorem abs_one_point_five_minus_sqrt_two : |1.5 - Real.sqrt 2| = 1.5 - Real.sqrt 2 := by
  sorry

end abs_one_point_five_minus_sqrt_two_l2874_287467


namespace sqrt_product_simplification_l2874_287449

theorem sqrt_product_simplification (p : ℝ) (hp : p ≥ 0) :
  Real.sqrt (40 * p^2) * Real.sqrt (10 * p^3) * Real.sqrt (8 * p^2) = 40 * p^3 * Real.sqrt p :=
by sorry

end sqrt_product_simplification_l2874_287449


namespace sum_of_solutions_quadratic_l2874_287413

theorem sum_of_solutions_quadratic (x : ℝ) : 
  let equation := -48 * x^2 + 108 * x - 27 = 0
  let sum_of_solutions := -108 / (-48)
  sum_of_solutions = 9/4 := by
sorry

end sum_of_solutions_quadratic_l2874_287413


namespace complex_arithmetic_equation_l2874_287461

theorem complex_arithmetic_equation : 
  (8 * 2.25 - 5 * 0.85) / 2.5 + (3/5 * 1.5 - 7/8 * 0.35) / 1.25 = 5.975 := by
  sorry

end complex_arithmetic_equation_l2874_287461


namespace triangle_properties_l2874_287417

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = Real.sqrt 3 ∧
  (t.c - 2 * t.a) * Real.cos t.B + t.b * Real.cos t.C = 0

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.B = π / 3 ∧ Real.sqrt 3 < t.a + t.c ∧ t.a + t.c ≤ 2 * Real.sqrt 3 :=
by
  sorry

end triangle_properties_l2874_287417


namespace sin_330_degrees_l2874_287408

-- Define the angle in degrees
def angle : ℝ := 330

-- State the theorem
theorem sin_330_degrees : Real.sin (angle * π / 180) = -1/2 := by
  sorry

end sin_330_degrees_l2874_287408


namespace quadratic_inequality_l2874_287477

theorem quadratic_inequality (x : ℝ) : x^2 - 36*x + 325 ≤ 9 ↔ 16 ≤ x ∧ x ≤ 20 := by
  sorry

end quadratic_inequality_l2874_287477


namespace circle_equation_proof_l2874_287407

-- Define the two given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 10*y - 24 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 8 = 0

-- Define the line on which the center of the desired circle lies
def centerLine (x y : ℝ) : Prop := x + y = 0

-- Define the desired circle
def desiredCircle (x y : ℝ) : Prop := (x + 3)^2 + (y - 3)^2 = 10

-- Theorem statement
theorem circle_equation_proof :
  ∃ (cx cy : ℝ),
    -- The center is on the line x + y = 0
    centerLine cx cy ∧
    -- The circle passes through the intersection points of circle1 and circle2
    (∀ (x y : ℝ), circle1 x y ∧ circle2 x y → desiredCircle x y) ∧
    -- The equation (x + 3)² + (y - 3)² = 10 represents the desired circle
    (∀ (x y : ℝ), desiredCircle x y ↔ (x - cx)^2 + (y - cy)^2 = 10) :=
by sorry

end circle_equation_proof_l2874_287407


namespace total_matches_is_120_l2874_287480

/-- Represents the number of factions in the game -/
def num_factions : ℕ := 3

/-- Represents the number of players in each team -/
def team_size : ℕ := 4

/-- Represents the total number of players -/
def total_players : ℕ := 8

/-- Calculates the number of ways to form a team of given size from available factions -/
def ways_to_form_team (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Calculates the total number of distinct matches possible -/
def total_distinct_matches : ℕ :=
  let ways_one_team := ways_to_form_team team_size num_factions
  ways_one_team + Nat.choose ways_one_team 2

/-- Theorem stating that the total number of distinct matches is 120 -/
theorem total_matches_is_120 : total_distinct_matches = 120 := by
  sorry

end total_matches_is_120_l2874_287480


namespace alcohol_quantity_l2874_287472

/-- Proves that the quantity of alcohol is 16 liters given the initial and final ratios -/
theorem alcohol_quantity (initial_alcohol : ℚ) (initial_water : ℚ) (final_water : ℚ) :
  initial_alcohol / initial_water = 4 / 3 →
  initial_alcohol / (initial_water + 8) = 4 / 5 →
  initial_alcohol = 16 := by
sorry


end alcohol_quantity_l2874_287472


namespace factoring_expression_l2874_287435

theorem factoring_expression (x : ℝ) : 60 * x + 45 = 15 * (4 * x + 3) := by
  sorry

end factoring_expression_l2874_287435


namespace opposite_of_2023_l2874_287463

theorem opposite_of_2023 : 
  ∃ x : ℤ, (x + 2023 = 0) ∧ (x = -2023) := by
  sorry

end opposite_of_2023_l2874_287463


namespace family_fruit_consumption_l2874_287432

/-- Represents the number of fruits in a box for each type of fruit -/
structure FruitBox where
  apples : ℕ := 14
  bananas : ℕ := 20
  oranges : ℕ := 12

/-- Represents the daily consumption of fruits for each family member -/
structure DailyConsumption where
  apples : ℕ := 2  -- Henry and his brother combined
  bananas : ℕ := 2 -- Henry's sister (on odd days)
  oranges : ℕ := 3 -- Father

/-- Represents the number of boxes for each type of fruit -/
structure FruitSupply where
  appleBoxes : ℕ := 3
  bananaBoxes : ℕ := 4
  orangeBoxes : ℕ := 5

/-- Calculates the maximum number of days the family can eat their preferred fruits together -/
def max_days_eating_fruits (box : FruitBox) (consumption : DailyConsumption) (supply : FruitSupply) : ℕ :=
  sorry

/-- Theorem stating that the family can eat their preferred fruits together for 20 days -/
theorem family_fruit_consumption 
  (box : FruitBox) 
  (consumption : DailyConsumption) 
  (supply : FruitSupply) 
  (orange_days : ℕ := 20) -- Oranges are only available for 20 days
  (h1 : box.apples = 14)
  (h2 : box.bananas = 20)
  (h3 : box.oranges = 12)
  (h4 : consumption.apples = 2)
  (h5 : consumption.bananas = 2)
  (h6 : consumption.oranges = 3)
  (h7 : supply.appleBoxes = 3)
  (h8 : supply.bananaBoxes = 4)
  (h9 : supply.orangeBoxes = 5) :
  max_days_eating_fruits box consumption supply = 20 :=
sorry

end family_fruit_consumption_l2874_287432


namespace percentage_relation_l2874_287439

theorem percentage_relation (x a b : ℝ) (h1 : a = 0.06 * x) (h2 : b = 0.18 * x) :
  a / b * 100 = 100 / 3 :=
by sorry

end percentage_relation_l2874_287439


namespace two_prime_factors_phi_tau_equality_l2874_287498

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- Number of positive divisors function -/
def tau (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number has exactly two distinct prime factors -/
def has_two_distinct_prime_factors (n : ℕ) : Prop := sorry

/-- Main theorem -/
theorem two_prime_factors_phi_tau_equality (n : ℕ) :
  has_two_distinct_prime_factors n ∧ phi (tau n) = tau (phi n) ↔
  ∃ (t r : ℕ), r.Prime ∧ t > 0 ∧ n = 2^(t-1) * 3^(r-1) :=
sorry

end two_prime_factors_phi_tau_equality_l2874_287498
