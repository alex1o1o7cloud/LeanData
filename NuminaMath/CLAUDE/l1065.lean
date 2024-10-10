import Mathlib

namespace system_equations_properties_l1065_106580

theorem system_equations_properties (x y a : ℝ) 
  (eq1 : 3 * x + 2 * y = 8 + a) 
  (eq2 : 2 * x + 3 * y = 3 * a) : 
  (x = -y → a = -2) ∧ 
  (x - y = 8 - 2 * a) ∧ 
  (7 * x + 3 * y = 24) ∧ 
  (x = -3/7 * y + 24/7) := by
sorry

end system_equations_properties_l1065_106580


namespace xy_minimum_value_l1065_106598

theorem xy_minimum_value (x y : ℝ) (hx : x > 1) (hy : y > 1) 
  (h_geom : (1/4 * Real.log x) * (Real.log y) = (1/4)^2) : x * y ≥ Real.exp 1 := by
  sorry

end xy_minimum_value_l1065_106598


namespace distance_from_origin_to_point_l1065_106502

theorem distance_from_origin_to_point : 
  let x : ℝ := 3
  let y : ℝ := -4
  Real.sqrt (x^2 + y^2) = 5 := by sorry

end distance_from_origin_to_point_l1065_106502


namespace geometric_sequence_first_term_l1065_106536

/-- Given a geometric sequence where the second term is 18 and the fifth term is 1458,
    prove that the first term is 6. -/
theorem geometric_sequence_first_term
  (a : ℝ)  -- First term of the sequence
  (r : ℝ)  -- Common ratio of the sequence
  (h1 : a * r = 18)  -- Second term is 18
  (h2 : a * r^4 = 1458)  -- Fifth term is 1458
  : a = 6 := by
sorry

end geometric_sequence_first_term_l1065_106536


namespace path_width_is_three_l1065_106564

/-- Represents a rectangular garden surrounded by a path of constant width. -/
structure GardenWithPath where
  garden_length : ℝ
  garden_width : ℝ
  path_width : ℝ

/-- Calculates the perimeter of the garden. -/
def garden_perimeter (g : GardenWithPath) : ℝ :=
  2 * (g.garden_length + g.garden_width)

/-- Calculates the perimeter of the outer edge of the path. -/
def outer_perimeter (g : GardenWithPath) : ℝ :=
  2 * ((g.garden_length + 2 * g.path_width) + (g.garden_width + 2 * g.path_width))

/-- Theorem: If the perimeter of the garden is 24 m shorter than the outer perimeter,
    then the path width is 3 m. -/
theorem path_width_is_three (g : GardenWithPath) :
  outer_perimeter g = garden_perimeter g + 24 → g.path_width = 3 := by
  sorry

#check path_width_is_three

end path_width_is_three_l1065_106564


namespace convex_ngon_non_acute_side_l1065_106546

/-- A convex n-gon is a polygon with n sides and n vertices, where all internal angles are less than 180 degrees. -/
def ConvexNGon (n : ℕ) : Type := sorry

/-- An angle is acute if it is less than 90 degrees. -/
def IsAcute (angle : ℝ) : Prop := angle < 90

/-- Given a convex n-gon and a side, returns the two angles at the endpoints of that side. -/
def EndpointAngles (polygon : ConvexNGon n) (side : Fin n) : ℝ × ℝ := sorry

theorem convex_ngon_non_acute_side (n : ℕ) (hn : n ≥ 7) :
  ∀ (polygon : ConvexNGon n), ∃ (side : Fin n),
    let (angle1, angle2) := EndpointAngles polygon side
    ¬(IsAcute angle1 ∨ IsAcute angle2) :=
sorry

end convex_ngon_non_acute_side_l1065_106546


namespace digital_music_library_space_l1065_106566

/-- Calculates the average megabytes per hour of music in a digital library, rounded to the nearest whole number -/
def averageMegabytesPerHour (days : ℕ) (totalMegabytes : ℕ) : ℕ :=
  let hoursPerDay : ℕ := 24
  let totalHours : ℕ := days * hoursPerDay
  let exactAverage : ℚ := totalMegabytes / totalHours
  (exactAverage + 1/2).floor.toNat

theorem digital_music_library_space (days : ℕ) (totalMegabytes : ℕ) 
  (h1 : days = 15) (h2 : totalMegabytes = 20400) :
  averageMegabytesPerHour days totalMegabytes = 57 := by
  sorry

end digital_music_library_space_l1065_106566


namespace problem_solution_l1065_106597

theorem problem_solution (a b c : ℝ) : 
  8 = 0.06 * a → 
  6 = 0.08 * b → 
  c = b / a → 
  c = 0.5625 := by
sorry

end problem_solution_l1065_106597


namespace john_volunteer_hours_l1065_106575

/-- Represents John's volunteering schedule for a year -/
structure VolunteerSchedule where
  first_six_months_frequency : Nat
  first_six_months_hours : Nat
  next_five_months_frequency : Nat
  next_five_months_hours : Nat
  december_days : Nat
  december_total_hours : Nat

/-- Calculates the total volunteering hours for a year given a schedule -/
def total_volunteer_hours (schedule : VolunteerSchedule) : Nat :=
  (schedule.first_six_months_frequency * schedule.first_six_months_hours * 6) +
  (schedule.next_five_months_frequency * schedule.next_five_months_hours * 4 * 5) +
  schedule.december_total_hours

/-- Theorem stating that John's volunteering schedule results in 82 hours for the year -/
theorem john_volunteer_hours :
  ∃ (schedule : VolunteerSchedule),
    schedule.first_six_months_frequency = 2 ∧
    schedule.first_six_months_hours = 3 ∧
    schedule.next_five_months_frequency = 1 ∧
    schedule.next_five_months_hours = 2 ∧
    schedule.december_days = 3 ∧
    schedule.december_total_hours = 6 ∧
    total_volunteer_hours schedule = 82 := by
  sorry

end john_volunteer_hours_l1065_106575


namespace sanitizer_theorem_l1065_106549

/-- Represents the prices and quantities of hand sanitizer and disinfectant -/
structure SanitizerProblem where
  x : ℚ  -- Price of hand sanitizer
  y : ℚ  -- Price of 84 disinfectant
  eq1 : 100 * x + 150 * y = 1500
  eq2 : 120 * x + 160 * y = 1720
  promotion : ℕ → ℕ
  promotion_def : ∀ n : ℕ, promotion n = n / 150 * 10

/-- The solution to the sanitizer problem -/
def sanitizer_solution (p : SanitizerProblem) : Prop :=
  p.x = 9 ∧ p.y = 4 ∧ 
  9 * 150 + 4 * (60 - p.promotion 150) = 1550

/-- The main theorem stating that the solution is correct -/
theorem sanitizer_theorem (p : SanitizerProblem) : sanitizer_solution p := by
  sorry

end sanitizer_theorem_l1065_106549


namespace volunteer_allocation_schemes_l1065_106558

theorem volunteer_allocation_schemes (n : ℕ) (m : ℕ) (k : ℕ) : 
  n = 5 → m = 3 → k = 2 →
  (Nat.choose n 1) * (Nat.choose (n - 1) k / 2) * Nat.factorial m = 90 := by
  sorry

end volunteer_allocation_schemes_l1065_106558


namespace obtuse_triangle_count_l1065_106579

/-- A function that determines if a triangle with sides a, b, and c is obtuse -/
def is_obtuse (a b c : ℕ) : Prop :=
  (a ^ 2 > b ^ 2 + c ^ 2) ∨ (b ^ 2 > a ^ 2 + c ^ 2) ∨ (c ^ 2 > a ^ 2 + b ^ 2)

/-- A function that determines if three lengths can form a valid triangle -/
def is_valid_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b)

/-- The main theorem stating that there are exactly 14 positive integer values of k
    for which a triangle with side lengths 13, 17, and k is obtuse -/
theorem obtuse_triangle_count :
  (∃! (s : Finset ℕ), s.card = 14 ∧ 
    (∀ k, k ∈ s ↔ (k > 0 ∧ is_valid_triangle 13 17 k ∧ is_obtuse 13 17 k))) :=
sorry

end obtuse_triangle_count_l1065_106579


namespace highest_power_of_two_dividing_difference_of_fourth_powers_l1065_106592

theorem highest_power_of_two_dividing_difference_of_fourth_powers :
  ∃ k : ℕ, k = 7 ∧ 2^k = (Nat.gcd (17^4 - 15^4) (2^64)) :=
by
  sorry

end highest_power_of_two_dividing_difference_of_fourth_powers_l1065_106592


namespace sin6_cos2_integral_l1065_106525

theorem sin6_cos2_integral : ∫ x in (0 : ℝ)..(2 * Real.pi), (Real.sin x)^6 * (Real.cos x)^2 = (5 * Real.pi) / 64 := by
  sorry

end sin6_cos2_integral_l1065_106525


namespace set_equality_l1065_106517

theorem set_equality : {x : ℕ | x - 1 ≤ 2} = {0, 1, 2, 3} := by sorry

end set_equality_l1065_106517


namespace f_one_equals_phi_l1065_106528

noncomputable section

def φ : ℝ := (1 + Real.sqrt 5) / 2

-- Define the properties of function f
def IsValidF (f : ℝ → ℝ) : Prop :=
  (∀ x y, x > 0 → y > 0 → x < y → f x < f y) ∧ 
  (∀ x, x > 0 → f x * f (f x + 1/x) = 1)

-- State the theorem
theorem f_one_equals_phi (f : ℝ → ℝ) (h : IsValidF f) : f 1 = φ := by
  sorry

end

end f_one_equals_phi_l1065_106528


namespace liquid_ratio_after_replacement_l1065_106550

def container_capacity : ℝ := 37.5

def liquid_replaced : ℝ := 15

def final_ratio_A : ℝ := 9

def final_ratio_B : ℝ := 16

theorem liquid_ratio_after_replacement :
  let initial_A := container_capacity
  let first_step_A := initial_A - liquid_replaced
  let first_step_B := liquid_replaced
  let second_step_A := first_step_A * (1 - liquid_replaced / container_capacity)
  let second_step_B := container_capacity - second_step_A
  (second_step_A / final_ratio_A = second_step_B / final_ratio_B) ∧
  (second_step_A + second_step_B = container_capacity) := by
  sorry

end liquid_ratio_after_replacement_l1065_106550


namespace train_distance_difference_l1065_106531

/-- Proves that the difference in distance traveled by two trains meeting each other is 100 km -/
theorem train_distance_difference (v1 v2 total_distance : ℝ) 
  (h1 : v1 = 50) 
  (h2 : v2 = 60)
  (h3 : total_distance = 1100) : 
  (v2 * (total_distance / (v1 + v2))) - (v1 * (total_distance / (v1 + v2))) = 100 := by
  sorry

end train_distance_difference_l1065_106531


namespace exam_combinations_l1065_106541

/-- The number of compulsory subjects -/
def compulsory_subjects : ℕ := 3

/-- The number of subjects to choose from for the "1" part -/
def choose_one_from : ℕ := 2

/-- The number of subjects to choose from for the "2" part -/
def choose_two_from : ℕ := 4

/-- The number of subjects to be chosen in the "2" part -/
def subjects_to_choose : ℕ := 2

/-- Calculates the number of ways to choose k items from n items -/
def combinations (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The maximum number of different combinations in the "3+1+2" plan -/
def max_combinations : ℕ :=
  1 * choose_one_from * combinations choose_two_from subjects_to_choose

theorem exam_combinations :
  max_combinations = 12 :=
sorry

end exam_combinations_l1065_106541


namespace f_increasing_iff_a_nonnegative_l1065_106562

/-- A function f is increasing on an interval [a, +∞) if for all x, y in the interval with x < y, f(x) < f(y) -/
def IncreasingOnInterval (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → f x < f y

/-- The function f(x) = x^2 + 2(a-1)x + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 3

theorem f_increasing_iff_a_nonnegative :
  ∀ a : ℝ, IncreasingOnInterval (f a) 1 ↔ a ∈ Set.Ici 0 := by
  sorry

end f_increasing_iff_a_nonnegative_l1065_106562


namespace work_completion_time_l1065_106588

/-- 
Given:
- A can do a work in 14 days
- A and B together can do the same work in 10 days

Prove that B can do the work alone in 35 days
-/
theorem work_completion_time (work : ℝ) (a_rate b_rate : ℝ) 
  (h1 : a_rate = work / 14)
  (h2 : a_rate + b_rate = work / 10) :
  b_rate = work / 35 :=
sorry

end work_completion_time_l1065_106588


namespace vector_sum_as_complex_sum_l1065_106535

theorem vector_sum_as_complex_sum :
  let z₁ : ℂ := 1 + 4*I
  let z₂ : ℂ := -3 + 2*I
  z₁ + z₂ = -2 + 6*I :=
by sorry

end vector_sum_as_complex_sum_l1065_106535


namespace jane_baking_time_l1065_106594

/-- Represents the time it takes Jane to bake cakes individually -/
def jane_time : ℝ := 4

/-- Represents the time it takes Roy to bake cakes individually -/
def roy_time : ℝ := 5

/-- The time Jane and Roy work together -/
def joint_work_time : ℝ := 2

/-- The time Jane works alone after Roy leaves -/
def jane_solo_time : ℝ := 0.4

/-- The theorem stating that Jane's individual baking time is 4 hours -/
theorem jane_baking_time :
  (joint_work_time * (1 / jane_time + 1 / roy_time)) + 
  (jane_solo_time * (1 / jane_time)) = 1 :=
sorry

end jane_baking_time_l1065_106594


namespace add_preserves_inequality_l1065_106574

theorem add_preserves_inequality (a b : ℝ) (h : a > b) : a + 2 > b + 2 := by
  sorry

end add_preserves_inequality_l1065_106574


namespace custom_operation_result_l1065_106557

/-- Custom operation * for non-zero integers -/
def star (a b : ℤ) : ℚ := 1 / a + 1 / b

/-- Theorem: Given the conditions, prove that a * b = 3/8 -/
theorem custom_operation_result (a b : ℤ) 
  (h1 : a + b = 12) 
  (h2 : a * b = 32) 
  (h3 : b = 8) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) : 
  star a b = 3 / 8 := by
  sorry

#check custom_operation_result

end custom_operation_result_l1065_106557


namespace fertilizer_per_acre_l1065_106563

theorem fertilizer_per_acre 
  (num_horses : ℕ) 
  (fertilizer_per_horse_per_day : ℕ) 
  (total_acres : ℕ) 
  (acres_per_day : ℕ) 
  (total_days : ℕ) 
  (h1 : num_horses = 80)
  (h2 : fertilizer_per_horse_per_day = 5)
  (h3 : total_acres = 20)
  (h4 : acres_per_day = 4)
  (h5 : total_days = 25) :
  (num_horses * fertilizer_per_horse_per_day * total_days) / total_acres = 500 := by
  sorry

#check fertilizer_per_acre

end fertilizer_per_acre_l1065_106563


namespace car_distance_theorem_l1065_106506

/-- The distance traveled by a car under specific conditions -/
theorem car_distance_theorem (actual_speed : ℝ) (speed_increase : ℝ) (time_decrease : ℝ) :
  actual_speed = 20 →
  speed_increase = 10 →
  time_decrease = 0.5 →
  ∃ (distance : ℝ),
    distance = actual_speed * (distance / actual_speed) ∧
    distance = (actual_speed + speed_increase) * (distance / actual_speed - time_decrease) ∧
    distance = 30 :=
by
  sorry

end car_distance_theorem_l1065_106506


namespace conic_parametric_to_cartesian_l1065_106516

theorem conic_parametric_to_cartesian (t : ℝ) (x y : ℝ) :
  x = t^2 + 1/t^2 - 2 ∧ y = t - 1/t → y^2 = x :=
by sorry

end conic_parametric_to_cartesian_l1065_106516


namespace complex_sum_of_powers_l1065_106585

theorem complex_sum_of_powers : 
  ((-1 + Complex.I * Real.sqrt 3) / 2) ^ 12 + ((-1 - Complex.I * Real.sqrt 3) / 2) ^ 12 = 2 := by
  sorry

end complex_sum_of_powers_l1065_106585


namespace f_5_equals_2015_l1065_106556

/-- Horner's method representation of a polynomial --/
def horner_poly (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x^5 - 2x^4 + x^3 + x^2 - x - 5 --/
def f (x : ℤ) : ℤ := horner_poly [-5, -1, 1, 1, -2, 1] x

theorem f_5_equals_2015 : f 5 = 2015 := by
  sorry

end f_5_equals_2015_l1065_106556


namespace light_travel_distance_l1065_106510

/-- The distance light travels in one year in miles -/
def light_year_distance : ℝ := 5870000000000

/-- The number of years we're calculating for -/
def years : ℕ := 50

/-- Theorem stating the distance light travels in 50 years -/
theorem light_travel_distance : light_year_distance * years = 2935 * (10 : ℝ)^11 := by
  sorry

end light_travel_distance_l1065_106510


namespace family_size_l1065_106505

/-- Given a family where one side has 10 members and the other side is 30% larger,
    the total number of family members is 23. -/
theorem family_size (fathers_side : ℕ) (mothers_side : ℕ) : 
  fathers_side = 10 →
  mothers_side = fathers_side + (fathers_side * 3 / 10) →
  fathers_side + mothers_side = 23 :=
by
  sorry

end family_size_l1065_106505


namespace maggies_income_l1065_106596

/-- Maggie's weekly income calculation -/
theorem maggies_income
  (office_rate : ℝ)
  (tractor_rate : ℝ)
  (tractor_hours : ℝ)
  (total_income : ℝ)
  (h1 : tractor_rate = 12)
  (h2 : tractor_hours = 13)
  (h3 : total_income = 416)
  (h4 : office_rate * (2 * tractor_hours) + tractor_rate * tractor_hours = total_income) :
  office_rate = 10 := by
sorry

end maggies_income_l1065_106596


namespace area_18_rectangles_l1065_106571

def rectangle_pairs : Set (ℕ × ℕ) :=
  {(1, 18), (2, 9), (3, 6), (6, 3), (9, 2), (18, 1)}

theorem area_18_rectangles :
  ∀ (w l : ℕ), w > 0 → l > 0 → w * l = 18 ↔ (w, l) ∈ rectangle_pairs := by
  sorry

end area_18_rectangles_l1065_106571


namespace orange_buckets_l1065_106500

/-- Given three buckets of oranges with specific relationships between their quantities
    and a total number of oranges, prove that the first bucket contains 22 oranges. -/
theorem orange_buckets (b1 b2 b3 : ℕ) : 
  b2 = b1 + 17 →
  b3 = b2 - 11 →
  b1 + b2 + b3 = 89 →
  b1 = 22 := by
sorry

end orange_buckets_l1065_106500


namespace oak_grove_library_books_l1065_106514

/-- The number of books in Oak Grove's public library -/
def public_library_books : ℕ := 1986

/-- The number of books in Oak Grove's school libraries -/
def school_library_books : ℕ := 5106

/-- The total number of books in Oak Grove libraries -/
def total_books : ℕ := public_library_books + school_library_books

theorem oak_grove_library_books : total_books = 7092 := by
  sorry

end oak_grove_library_books_l1065_106514


namespace salt_solution_mixture_l1065_106512

/-- Given a mixture of pure water and salt solution, prove the amount of salt solution needed. -/
theorem salt_solution_mixture (x : ℝ) : 
  (0.30 * x = 0.20 * (x + 1)) → x = 2 := by
  sorry

end salt_solution_mixture_l1065_106512


namespace base_ten_satisfies_equation_l1065_106532

/-- Given a base b, converts a number in base b to decimal --/
def toDecimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun digit acc => digit + b * acc) 0

/-- Checks if the equation 253_b + 176_b = 431_b holds for a given base b --/
def equationHolds (b : Nat) : Prop :=
  toDecimal [2, 5, 3] b + toDecimal [1, 7, 6] b = toDecimal [4, 3, 1] b

theorem base_ten_satisfies_equation :
  equationHolds 10 ∧ ∀ b : Nat, b ≠ 10 → ¬equationHolds b :=
sorry

end base_ten_satisfies_equation_l1065_106532


namespace commute_time_is_120_minutes_l1065_106568

def minutes_in_hour : ℕ := 60

def rise_time : ℕ := 6 * 60  -- 6:00 a.m. in minutes
def leave_time : ℕ := 7 * 60  -- 7:00 a.m. in minutes
def return_time : ℕ := 17 * 60 + 30  -- 5:30 p.m. in minutes

def num_lectures : ℕ := 8
def lecture_duration : ℕ := 45
def lunch_duration : ℕ := 60
def library_duration : ℕ := 90

def total_time_away : ℕ := return_time - leave_time

def total_college_time : ℕ := num_lectures * lecture_duration + lunch_duration + library_duration

theorem commute_time_is_120_minutes :
  total_time_away - total_college_time = 120 := by
  sorry

end commute_time_is_120_minutes_l1065_106568


namespace fractional_inequality_solution_set_l1065_106587

theorem fractional_inequality_solution_set (x : ℝ) : 
  (x - 1) / (2 * x + 3) > 1 ↔ -4 < x ∧ x < -3/2 :=
by sorry

end fractional_inequality_solution_set_l1065_106587


namespace total_spent_theorem_l1065_106578

/-- Calculates the total amount spent on pens by Dorothy, Julia, and Robert --/
def total_spent_on_pens (robert_pens : ℕ) (julia_factor : ℕ) (dorothy_factor : ℚ) (cost_per_pen : ℚ) : ℚ :=
  let julia_pens := julia_factor * robert_pens
  let dorothy_pens := dorothy_factor * julia_pens
  let total_pens := robert_pens + julia_pens + dorothy_pens
  total_pens * cost_per_pen

/-- Theorem stating the total amount spent on pens by Dorothy, Julia, and Robert --/
theorem total_spent_theorem :
  total_spent_on_pens 4 3 (1/2) (3/2) = 33 := by
  sorry

end total_spent_theorem_l1065_106578


namespace smallest_a_value_l1065_106540

theorem smallest_a_value (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) 
  (h : ∀ x : ℤ, Real.sin (a * ↑x + b) = Real.sin (17 * ↑x + π)) : 
  a ≥ 17 ∧ (∀ a' ≥ 0, (∀ x : ℤ, Real.sin (a' * ↑x + b) = Real.sin (17 * ↑x + π)) → a' ≥ a) :=
sorry

end smallest_a_value_l1065_106540


namespace li_shuang_walking_speed_l1065_106589

/-- The problem of finding Li Shuang's walking speed -/
theorem li_shuang_walking_speed 
  (initial_speed : ℝ) 
  (walking_time : ℝ) 
  (repair_distance : ℝ) 
  (repair_time : ℝ) 
  (speed_multiplier : ℝ) 
  (delay : ℝ)
  (h1 : initial_speed = 320)
  (h2 : walking_time = 5)
  (h3 : repair_distance = 1800)
  (h4 : repair_time = 15)
  (h5 : speed_multiplier = 1.5)
  (h6 : delay = 17) :
  ∃ (walking_speed : ℝ), walking_speed = 72 ∧ 
  (∃ (total_distance : ℝ), 
    total_distance / initial_speed + delay = 
    walking_time + repair_time + 
    (total_distance - repair_distance - walking_speed * walking_time) / (initial_speed * speed_multiplier)) := by
  sorry

end li_shuang_walking_speed_l1065_106589


namespace probability_after_removal_l1065_106542

/-- Represents a deck of cards -/
structure Deck :=
  (total : ℕ)
  (numbers : ℕ)
  (cards_per_number : ℕ)

/-- The probability of selecting a pair from a deck -/
def probability_of_pair (d : Deck) : ℚ :=
  sorry

/-- The original deck configuration -/
def original_deck : Deck :=
  { total := 52, numbers := 13, cards_per_number := 4 }

/-- The deck after removing a matching pair -/
def remaining_deck : Deck :=
  { total := 48, numbers := 12, cards_per_number := 4 }

theorem probability_after_removal :
  probability_of_pair remaining_deck = 3 / 47 := by
  sorry

end probability_after_removal_l1065_106542


namespace bus_fare_impossible_l1065_106586

/-- Represents the denominations of coins available --/
inductive Coin : Type
  | ten : Coin
  | fifteen : Coin
  | twenty : Coin

/-- The value of a coin in kopecks --/
def coin_value : Coin → Nat
  | Coin.ten => 10
  | Coin.fifteen => 15
  | Coin.twenty => 20

/-- A configuration of coins --/
def CoinConfig := List Coin

/-- The total value of a coin configuration in kopecks --/
def total_value (config : CoinConfig) : Nat :=
  config.foldl (fun acc c => acc + coin_value c) 0

/-- The number of coins in a configuration --/
def coin_count (config : CoinConfig) : Nat := config.length

theorem bus_fare_impossible : 
  ∀ (config : CoinConfig), 
    (coin_count config = 49) → 
    (total_value config = 200) → 
    False :=
sorry

end bus_fare_impossible_l1065_106586


namespace symmetric_lines_ellipse_intersection_l1065_106553

/-- Given two lines symmetric about y = x + 1 intersecting an ellipse, 
    prove properties about their slopes and intersection points. -/
theorem symmetric_lines_ellipse_intersection 
  (k : ℝ) 
  (h_k_pos : k > 0) 
  (h_k_neq_one : k ≠ 1) 
  (k₁ : ℝ) 
  (h_symmetric : ∀ x y, y = k * x + 1 ↔ y = k₁ * x + 1) 
  (E : Set (ℝ × ℝ)) 
  (h_E : E = {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1}) 
  (A M N : ℝ × ℝ) 
  (h_A : A ∈ E ∧ A.2 = k * A.1 + 1 ∧ A.2 = k₁ * A.1 + 1) 
  (h_M : M ∈ E ∧ M.2 = k * M.1 + 1) 
  (h_N : N ∈ E ∧ N.2 = k₁ * N.1 + 1) : 
  k * k₁ = 1 ∧ 
  ∃ t : ℝ, (1 - t) * M.1 + t * N.1 = 0 ∧ (1 - t) * M.2 + t * N.2 = -5/3 := by
  sorry


end symmetric_lines_ellipse_intersection_l1065_106553


namespace store_inventory_price_l1065_106533

theorem store_inventory_price (total_items : ℕ) (discount_rate : ℚ) (sold_rate : ℚ)
  (debt : ℕ) (remaining : ℕ) :
  total_items = 2000 →
  discount_rate = 80 / 100 →
  sold_rate = 90 / 100 →
  debt = 15000 →
  remaining = 3000 →
  ∃ (price : ℚ), price = 50 ∧
    (1 - discount_rate) * (sold_rate * total_items) * price = debt + remaining :=
by sorry

end store_inventory_price_l1065_106533


namespace puzzle_sum_l1065_106547

def is_valid_puzzle (a b c d e f g h i : ℕ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0 ∧ i ≠ 0 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
  g ≠ h ∧ g ≠ i ∧
  h ≠ i

theorem puzzle_sum (a b c d e f g h i : ℕ) :
  is_valid_puzzle a b c d e f g h i →
  (100 * a + 10 * b + c) + (100 * d + 10 * e + f) + (100 * g + 10 * h + i) = 1665 →
  b + e + h = 15 := by
  sorry

end puzzle_sum_l1065_106547


namespace max_min_sum_of_f_l1065_106503

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + 1) - x) + (3 * Real.exp x + 1) / (Real.exp x + 1)

def domain : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }

theorem max_min_sum_of_f :
  ∃ (M N : ℝ), (∀ x ∈ domain, f x ≤ M) ∧
               (∀ x ∈ domain, N ≤ f x) ∧
               (∃ x₁ ∈ domain, f x₁ = M) ∧
               (∃ x₂ ∈ domain, f x₂ = N) ∧
               M + N = 4 := by
  sorry

end max_min_sum_of_f_l1065_106503


namespace total_methods_is_fifteen_l1065_106577

/-- A two-stage test with options for each stage -/
structure TwoStageTest where
  first_stage_options : Nat
  second_stage_options : Nat

/-- Calculate the total number of testing methods for a two-stage test -/
def total_testing_methods (test : TwoStageTest) : Nat :=
  test.first_stage_options * test.second_stage_options

/-- The specific test configuration -/
def our_test : TwoStageTest :=
  { first_stage_options := 3
  , second_stage_options := 5 }

theorem total_methods_is_fifteen :
  total_testing_methods our_test = 15 := by
  sorry

#eval total_testing_methods our_test

end total_methods_is_fifteen_l1065_106577


namespace bookseller_display_windows_l1065_106569

/-- Given the conditions of the bookseller's display windows problem, prove that the number of non-fiction books is 2. -/
theorem bookseller_display_windows (fiction_books : ℕ) (display_fiction : ℕ) (total_configs : ℕ) :
  fiction_books = 4 →
  display_fiction = 3 →
  total_configs = 36 →
  ∃ n : ℕ, n = 2 ∧ (Nat.factorial fiction_books / Nat.factorial (fiction_books - display_fiction)) * Nat.factorial n = total_configs :=
by sorry

end bookseller_display_windows_l1065_106569


namespace triangle_properties_l1065_106543

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  -- Condition 1
  (Real.sqrt 2 * c = a * Real.sin C + c * Real.cos A →
    A = Real.pi / 4) ∧
  -- Condition 2
  (Real.sin (B + C) = Real.sqrt 2 - 1 + 2 * (Real.sin (A / 2))^2 →
    A = Real.pi / 4) ∧
  -- Condition 3
  (Real.sqrt 2 * Real.cos (Real.pi / 2 - A) = Real.sin (2 * A) →
    A = Real.pi / 4) ∧
  -- Part 2
  (A = Real.pi / 4 ∧ S = 6 ∧ b = 2 * Real.sqrt 2 →
    a = 2 * Real.sqrt 5) :=
by sorry

end triangle_properties_l1065_106543


namespace chess_tournament_wins_l1065_106513

theorem chess_tournament_wins (susan_wins susan_losses mike_wins mike_losses lana_losses : ℕ) 
  (h1 : susan_wins = 5)
  (h2 : susan_losses = 1)
  (h3 : mike_wins = 2)
  (h4 : mike_losses = 4)
  (h5 : lana_losses = 5)
  (h6 : susan_wins + mike_wins + lana_losses = susan_losses + mike_losses + lana_wins)
  : lana_wins = 3 := by
  sorry

end chess_tournament_wins_l1065_106513


namespace max_large_chips_l1065_106524

theorem max_large_chips :
  ∀ (small large : ℕ),
  small + large = 72 →
  ∃ (p : ℕ), Prime p ∧ small = large + p →
  large ≤ 35 :=
by sorry

end max_large_chips_l1065_106524


namespace r_value_when_n_is_3_l1065_106507

theorem r_value_when_n_is_3 :
  let n : ℕ := 3
  let s : ℕ := 2^(n^2) + n
  let r : ℕ := 3^s - 2*s
  r = 3^515 - 1030 := by
sorry

end r_value_when_n_is_3_l1065_106507


namespace intersection_when_m_is_one_necessary_condition_range_l1065_106529

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3/2}
def B (m : ℝ) : Set ℝ := {x | 1-m < x ∧ x ≤ 3*m+1}

theorem intersection_when_m_is_one :
  A ∩ B 1 = {x : ℝ | 0 < x ∧ x ≤ 3/2} := by sorry

theorem necessary_condition_range :
  ∀ m : ℝ, (∀ x : ℝ, x ∈ B m → x ∈ A) ↔ m ≤ 1/6 := by sorry

end intersection_when_m_is_one_necessary_condition_range_l1065_106529


namespace remainder_3_88_plus_5_mod_7_l1065_106559

theorem remainder_3_88_plus_5_mod_7 : (3^88 + 5) % 7 = 2 := by
  sorry

end remainder_3_88_plus_5_mod_7_l1065_106559


namespace similar_triangles_solution_l1065_106591

/-- Two similar right triangles with legs 15 and 12 in the first triangle, 
    and y and 9 in the second triangle. -/
def similar_triangles (y : ℝ) : Prop :=
  15 / y = 12 / 9

theorem similar_triangles_solution :
  ∃ y : ℝ, similar_triangles y ∧ y = 11.25 := by
  sorry

end similar_triangles_solution_l1065_106591


namespace car_speed_problem_l1065_106567

theorem car_speed_problem (distance : ℝ) (time_difference : ℝ) (speed_difference : ℝ) :
  distance = 150 ∧
  time_difference = 2 ∧
  speed_difference = 10 →
  ∃ (speed_r : ℝ),
    speed_r > 0 ∧
    distance / speed_r - time_difference = distance / (speed_r + speed_difference) ∧
    speed_r = 25 :=
by sorry

end car_speed_problem_l1065_106567


namespace number_2005_location_l1065_106590

/-- The sum of the first n positive integers -/
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The last number in the nth row of the pattern -/
def last_in_row (n : ℕ) : ℕ := n^2

/-- The first number in the nth row of the pattern -/
def first_in_row (n : ℕ) : ℕ := last_in_row (n - 1) + 1

/-- The number of elements in the nth row of the pattern -/
def elements_in_row (n : ℕ) : ℕ := 2 * n - 1

/-- The position of a number within its row in the pattern -/
def position_in_row (n : ℕ) (target : ℕ) : ℕ :=
  target - first_in_row n + 1

theorem number_2005_location :
  ∃ (i j : ℕ), i = 45 ∧ j = 20 ∧ 
  first_in_row i ≤ 2005 ∧
  2005 ≤ last_in_row i ∧
  position_in_row i 2005 = j :=
sorry

end number_2005_location_l1065_106590


namespace function_passes_through_point_l1065_106539

theorem function_passes_through_point (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) - 1
  f 2 = 0 := by sorry

end function_passes_through_point_l1065_106539


namespace sphere_cone_equal_volume_l1065_106595

/-- Given a cone with radius 2 inches and height 6 inches, prove that a sphere
    with radius ∛6 inches has the same volume as the cone. -/
theorem sphere_cone_equal_volume :
  let cone_radius : ℝ := 2
  let cone_height : ℝ := 6
  let sphere_radius : ℝ := (6 : ℝ) ^ (1/3)
  (1/3 : ℝ) * Real.pi * cone_radius^2 * cone_height = (4/3 : ℝ) * Real.pi * sphere_radius^3 :=
by sorry

end sphere_cone_equal_volume_l1065_106595


namespace a_seq_divisibility_l1065_106599

/-- Given a natural number a ≥ 2, define the sequence a_n recursively -/
def a_seq (a : ℕ) : ℕ → ℕ
  | 0 => a
  | n + 1 => a ^ (a_seq a n)

/-- The main theorem stating the divisibility property of the sequence -/
theorem a_seq_divisibility (a : ℕ) (h : a ≥ 2) (n : ℕ) :
  (a_seq a (n + 1) - a_seq a n) ∣ (a_seq a (n + 2) - a_seq a (n + 1)) :=
by sorry

end a_seq_divisibility_l1065_106599


namespace other_people_in_house_l1065_106515

-- Define the given conditions
def cups_per_person_per_day : ℕ := 2
def ounces_per_cup : ℚ := 1/2
def price_per_ounce : ℚ := 5/4
def weekly_spend : ℚ := 35

-- Define the theorem
theorem other_people_in_house :
  let total_ounces : ℚ := weekly_spend / price_per_ounce
  let ounces_per_person_per_week : ℚ := 7 * cups_per_person_per_day * ounces_per_cup
  let total_people : ℕ := Nat.floor (total_ounces / ounces_per_person_per_week)
  total_people - 1 = 3 := by
  sorry

end other_people_in_house_l1065_106515


namespace jonas_bookshelves_l1065_106576

/-- Calculates the maximum number of bookshelves that can fit in a room. -/
def max_bookshelves (total_space : ℕ) (reserved_space : ℕ) (shelf_space : ℕ) : ℕ :=
  (total_space - reserved_space) / shelf_space

/-- Proves that given the specific conditions, the maximum number of bookshelves is 3. -/
theorem jonas_bookshelves :
  max_bookshelves 400 160 80 = 3 := by
  sorry

end jonas_bookshelves_l1065_106576


namespace inequality_solution_l1065_106522

open Set

theorem inequality_solution (x : ℝ) : 
  (x^2 - 1) / (x^2 - 3*x + 2) ≥ 2 ↔ x ∈ Ioo 1 2 ∪ Ioo (3 - Real.sqrt 6) (3 + Real.sqrt 6) :=
by sorry

end inequality_solution_l1065_106522


namespace final_movie_length_l1065_106538

/-- Given an original movie length of 60 minutes and a cut scene of 6 minutes,
    the final movie length is 54 minutes. -/
theorem final_movie_length (original_length cut_length : ℕ) 
  (h1 : original_length = 60)
  (h2 : cut_length = 6) :
  original_length - cut_length = 54 := by
  sorry

end final_movie_length_l1065_106538


namespace expression_simplification_l1065_106520

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2) :
  (a^2 - 1) / (a^2 - a) / (2 + (a^2 + 1) / a) = Real.sqrt 2 - 1 := by
  sorry

end expression_simplification_l1065_106520


namespace johns_age_l1065_106504

/-- Proves that John's current age is 39 years old given the problem conditions -/
theorem johns_age (john_age : ℕ) (james_age : ℕ) (james_brother_age : ℕ) : 
  james_brother_age = 16 →
  james_brother_age = james_age + 4 →
  john_age - 3 = 2 * (james_age + 6) →
  john_age = 39 := by
  sorry

#check johns_age

end johns_age_l1065_106504


namespace soccer_ball_donation_l1065_106501

/-- Calculates the total number of soccer balls donated by a public official to two schools -/
def total_soccer_balls (balls_per_class : ℕ) (num_schools : ℕ) (elementary_classes : ℕ) (middle_classes : ℕ) : ℕ :=
  balls_per_class * num_schools * (elementary_classes + middle_classes)

/-- Proves that the total number of soccer balls donated is 90 -/
theorem soccer_ball_donation : total_soccer_balls 5 2 4 5 = 90 := by
  sorry

end soccer_ball_donation_l1065_106501


namespace nigel_money_problem_l1065_106572

theorem nigel_money_problem (initial_amount : ℕ) (mother_gift : ℕ) (final_amount : ℕ) : 
  initial_amount = 45 →
  mother_gift = 80 →
  final_amount = 2 * initial_amount + 10 →
  initial_amount - (final_amount - mother_gift) = 25 :=
by
  sorry

end nigel_money_problem_l1065_106572


namespace candy_bar_calories_l1065_106518

theorem candy_bar_calories
  (distance : ℕ) -- Total distance walked
  (calories_per_mile : ℕ) -- Calories burned per mile
  (net_deficit : ℕ) -- Net calorie deficit
  (h1 : distance = 3) -- Cary walks 3 miles round-trip
  (h2 : calories_per_mile = 150) -- Cary burns 150 calories per mile
  (h3 : net_deficit = 250) -- Cary's net calorie deficit is 250 calories
  : distance * calories_per_mile - net_deficit = 200 := by
  sorry

end candy_bar_calories_l1065_106518


namespace clothes_cost_l1065_106560

def total_spent : ℕ := 8000
def adidas_cost : ℕ := 600

theorem clothes_cost (nike_cost : ℕ) (skechers_cost : ℕ) :
  nike_cost = 3 * adidas_cost →
  skechers_cost = 5 * adidas_cost →
  total_spent - (adidas_cost + nike_cost + skechers_cost) = 2600 :=
by sorry

end clothes_cost_l1065_106560


namespace bush_leaves_theorem_l1065_106523

theorem bush_leaves_theorem (total_branches : ℕ) (leaves_only : ℕ) (leaves_with_flower : ℕ) : 
  total_branches = 10 →
  leaves_only = 5 →
  leaves_with_flower = 2 →
  ∀ (total_leaves : ℕ),
    (∃ (m n : ℕ), m + n = total_branches ∧ total_leaves = m * leaves_only + n * leaves_with_flower) →
    total_leaves ≠ 45 ∧ total_leaves ≠ 39 ∧ total_leaves ≠ 37 ∧ total_leaves ≠ 31 :=
by sorry

end bush_leaves_theorem_l1065_106523


namespace total_new_people_value_l1065_106509

/-- The number of people born in the country last year -/
def people_born : ℕ := 90171

/-- The number of people who immigrated to the country last year -/
def people_immigrated : ℕ := 16320

/-- The total number of new people who began living in the country last year -/
def total_new_people : ℕ := people_born + people_immigrated

/-- Theorem stating that the total number of new people is 106491 -/
theorem total_new_people_value : total_new_people = 106491 := by
  sorry

end total_new_people_value_l1065_106509


namespace complex_magnitude_power_four_l1065_106581

theorem complex_magnitude_power_four : 
  Complex.abs ((1 - Complex.I * Real.sqrt 3) ^ 4) = 16 := by sorry

end complex_magnitude_power_four_l1065_106581


namespace workshop_salary_problem_l1065_106554

/-- Proves that the average salary of non-technician workers is 6000, given the conditions of the workshop --/
theorem workshop_salary_problem (total_workers : ℕ) (avg_salary_all : ℕ) 
  (num_technicians : ℕ) (avg_salary_tech : ℕ) :
  total_workers = 49 →
  avg_salary_all = 8000 →
  num_technicians = 7 →
  avg_salary_tech = 20000 →
  (total_workers - num_technicians) * 
    ((total_workers * avg_salary_all - num_technicians * avg_salary_tech) / 
     (total_workers - num_technicians)) = 
  (total_workers - num_technicians) * 6000 := by
  sorry

#check workshop_salary_problem

end workshop_salary_problem_l1065_106554


namespace glycerin_solution_problem_l1065_106582

/-- Proves that given a solution with an initial volume of 4 gallons, 
    adding 0.8 gallons of water to achieve a 75% glycerin solution 
    implies that the initial percentage of glycerin was 90%. -/
theorem glycerin_solution_problem (initial_volume : ℝ) (water_added : ℝ) (final_percentage : ℝ) :
  initial_volume = 4 →
  water_added = 0.8 →
  final_percentage = 0.75 →
  (initial_volume * (initial_volume / (initial_volume + water_added))) / initial_volume = 0.9 :=
by sorry

end glycerin_solution_problem_l1065_106582


namespace exists_circumscribing_square_l1065_106573

/-- A type representing a bounded convex shape in a plane -/
structure BoundedConvexShape where
  -- Add necessary fields/axioms to define a bounded convex shape
  is_bounded : Bool
  is_convex : Bool

/-- A type representing a square in a plane -/
structure Square where
  -- Add necessary fields to define a square

/-- Predicate to check if a square circumscribes a bounded convex shape -/
def circumscribes (s : Square) (shape : BoundedConvexShape) : Prop :=
  sorry -- Define the circumscription condition

/-- Theorem stating that every bounded convex shape can be circumscribed by a square -/
theorem exists_circumscribing_square (shape : BoundedConvexShape) :
  shape.is_bounded ∧ shape.is_convex → ∃ s : Square, circumscribes s shape := by
  sorry


end exists_circumscribing_square_l1065_106573


namespace g_inequality_l1065_106570

/-- A quadratic function f(x) = ax^2 + a that is even on the interval [-a, a^2] -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a

/-- The function g(x) = f(x-1) -/
def g (a : ℝ) (x : ℝ) : ℝ := f a (x - 1)

/-- Theorem stating the relationship between g(3/2), g(0), and g(3) -/
theorem g_inequality (a : ℝ) (h1 : a ≠ 0) 
  (h2 : ∀ x ∈ Set.Icc (-a) (a^2), f a x = f a (-x)) : 
  g a (3/2) < g a 0 ∧ g a 0 < g a 3 := by
  sorry

end g_inequality_l1065_106570


namespace sin_decreasing_interval_l1065_106561

theorem sin_decreasing_interval :
  ∀ x ∈ Set.Icc (π / 2) (3 * π / 2),
    ∀ y ∈ Set.Icc (π / 2) (3 * π / 2),
      x ≤ y → Real.sin x ≥ Real.sin y :=
by sorry

end sin_decreasing_interval_l1065_106561


namespace simplify_expression_l1065_106545

theorem simplify_expression : (27 * (10 ^ 12)) / (9 * (10 ^ 4)) = 300000000 := by
  sorry

end simplify_expression_l1065_106545


namespace widget_production_theorem_l1065_106555

/-- Represents the widget production difference between Monday and Tuesday -/
def widget_production_difference (t : ℝ) : ℝ :=
  let w := 3 * t  -- Monday's production rate
  let monday_production := w * t
  let tuesday_production := (w + 5) * (t - 3)
  monday_production - tuesday_production

/-- Theorem stating the widget production difference -/
theorem widget_production_theorem (t : ℝ) :
  widget_production_difference t = 4 * t + 15 := by
  sorry

end widget_production_theorem_l1065_106555


namespace max_value_f_range_of_m_l1065_106508

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := 2 * Real.log x - (1/2) * x^2

-- Define the interval [1/e, e]
def I : Set ℝ := { x | 1/Real.exp 1 ≤ x ∧ x ≤ Real.exp 1 }

-- Statement for part (I)
theorem max_value_f : 
  ∃ (x : ℝ), x ∈ I ∧ f x = Real.log 2 - 1 ∧ ∀ y ∈ I, f y ≤ f x :=
sorry

-- Define the function g for part (II)
def g (a x : ℝ) : ℝ := a * Real.log x

-- Define the intervals for a and x in part (II)
def A : Set ℝ := { a | 0 ≤ a ∧ a ≤ 3/2 }
def X : Set ℝ := { x | 1 < x ∧ x ≤ Real.exp 2 }

-- Statement for part (II)
theorem range_of_m :
  ∀ m : ℝ, (∀ a ∈ A, ∀ x ∈ X, g a x ≥ m + x) ↔ m ≤ -(Real.exp 2) :=
sorry

end max_value_f_range_of_m_l1065_106508


namespace base_seven_sum_of_digits_product_l1065_106530

def to_decimal (n : ℕ) (base : ℕ) : ℕ := sorry

def from_decimal (n : ℕ) (base : ℕ) : ℕ := sorry

def add_base (a b base : ℕ) : ℕ := 
  from_decimal (to_decimal a base + to_decimal b base) base

def mult_base (a b base : ℕ) : ℕ := 
  from_decimal (to_decimal a base * to_decimal b base) base

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem base_seven_sum_of_digits_product : 
  let base := 7
  let a := 35
  let b := add_base 12 16 base
  let product := mult_base a b base
  sum_of_digits product = 7 := by sorry

end base_seven_sum_of_digits_product_l1065_106530


namespace expression_simplification_l1065_106521

theorem expression_simplification :
  let a := 16 / 2015
  let b := 17 / 2016
  (6 + a) * (9 + b) - (3 - a) * (18 - b) - 27 * a = 17 / 224 := by sorry

end expression_simplification_l1065_106521


namespace probability_no_same_color_boxes_l1065_106527

/-- Represents a person with 4 colored blocks -/
structure Person :=
  (blocks : Fin 4 → Color)

/-- The four possible colors of blocks -/
inductive Color
  | Red
  | Blue
  | Yellow
  | Black

/-- Represents a placement of blocks in boxes -/
def Placement := Fin 4 → Fin 4

/-- The probability space of all possible placements -/
def PlacementSpace := Person → Placement

/-- Checks if a box has blocks of all the same color -/
def hasSameColorBlocks (p : PlacementSpace) (box : Fin 4) : Prop :=
  ∃ c : Color, ∀ person : Person, (person.blocks ((p person) box)) = c

/-- The event where no box has blocks of all the same color -/
def NoSameColorBoxes (p : PlacementSpace) : Prop :=
  ∀ box : Fin 4, ¬(hasSameColorBlocks p box)

/-- The probability measure on the placement space -/
noncomputable def P : (PlacementSpace → Prop) → ℝ :=
  sorry

theorem probability_no_same_color_boxes :
  P NoSameColorBoxes = 14811 / 65536 :=
sorry

end probability_no_same_color_boxes_l1065_106527


namespace x_intercepts_count_l1065_106551

theorem x_intercepts_count : Nat.card { k : ℤ | 100 < k * Real.pi ∧ k * Real.pi < 1000 } = 286 := by
  sorry

end x_intercepts_count_l1065_106551


namespace average_math_chemistry_l1065_106534

-- Define the marks for each subject
variable (M P C : ℕ)

-- Define the conditions
axiom total_math_physics : M + P = 70
axiom chemistry_score : C = P + 20

-- Define the theorem to prove
theorem average_math_chemistry : (M + C) / 2 = 45 := by
  sorry

end average_math_chemistry_l1065_106534


namespace final_sum_after_operations_l1065_106519

theorem final_sum_after_operations (a b S : ℝ) (h : a + b = S) :
  3 * ((a + 5) + (b + 5)) = 3 * S + 30 := by
  sorry


end final_sum_after_operations_l1065_106519


namespace lucas_pet_beds_lucas_pet_beds_solution_l1065_106526

theorem lucas_pet_beds (initial_beds : ℕ) (beds_per_pet : ℕ) (pets_capacity : ℕ) : ℕ :=
  let total_beds_needed := pets_capacity * beds_per_pet
  let additional_beds := total_beds_needed - initial_beds
  additional_beds

theorem lucas_pet_beds_solution :
  lucas_pet_beds 12 2 10 = 8 := by
  sorry

end lucas_pet_beds_lucas_pet_beds_solution_l1065_106526


namespace abs_5x_minus_3_not_positive_l1065_106511

theorem abs_5x_minus_3_not_positive (x : ℚ) : 
  ¬(|5*x - 3| > 0) ↔ x = 3/5 := by sorry

end abs_5x_minus_3_not_positive_l1065_106511


namespace locus_of_points_l1065_106584

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  a : ℝ
  b : ℝ
  center : Point

/-- Represents an octagon -/
structure Octagon where
  vertices : Fin 8 → Point

/-- Calculate the absolute distance from a point to a line segment -/
def distToSegment (p : Point) (s1 s2 : Point) : ℝ :=
  sorry

/-- Calculate the sum of distances from a point to the sides of a rectangle -/
def sumDistToSides (p : Point) (r : Rectangle) : ℝ :=
  sorry

/-- Check if a point is inside or on the boundary of an octagon -/
def isInOctagon (p : Point) (o : Octagon) : Prop :=
  sorry

/-- Construct the octagon based on the rectangle and c value -/
def constructOctagon (r : Rectangle) (c : ℝ) : Octagon :=
  sorry

/-- The main theorem statement -/
theorem locus_of_points (r : Rectangle) (c : ℝ) :
  ∀ p : Point, sumDistToSides p r = r.a + r.b + c ↔ isInOctagon p (constructOctagon r c) :=
  sorry

end locus_of_points_l1065_106584


namespace percentage_difference_l1065_106537

theorem percentage_difference (x y : ℝ) (h : x = 8 * y) :
  (x - y) / x * 100 = 87.5 := by
  sorry

end percentage_difference_l1065_106537


namespace parabola_directrix_equation_l1065_106548

/-- Proves that for a parabola y^2 = 2px with focus (2, 0) coinciding with the right focus of the ellipse x^2/9 + y^2/5 = 1, the equation of the directrix is x = -2. -/
theorem parabola_directrix_equation (p : ℝ) : 
  (∀ x y : ℝ, y^2 = 2*p*x → (2 : ℝ) = p/2) → 
  (∀ x y : ℝ, x^2/9 + y^2/5 = 1 → (2 : ℝ) = Real.sqrt (9 - 5)) → 
  (∀ x : ℝ, x = -p/2 ↔ x = -2) := by
sorry

end parabola_directrix_equation_l1065_106548


namespace thirtieth_in_base_five_l1065_106565

def to_base_five (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem thirtieth_in_base_five :
  to_base_five 30 = [1, 1, 0] :=
sorry

end thirtieth_in_base_five_l1065_106565


namespace ratio_of_divisor_sums_l1065_106593

def M : ℕ := 36 * 36 * 75 * 224

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisor_sums :
  (sum_odd_divisors M : ℚ) / (sum_even_divisors M : ℚ) = 1 / 510 := by sorry

end ratio_of_divisor_sums_l1065_106593


namespace ice_cream_group_size_l1065_106583

/-- The number of days it takes one person to eat a gallon of ice cream -/
def days_per_person : ℕ := 5 * 16

/-- The number of days it takes the group to eat a gallon of ice cream -/
def days_for_group : ℕ := 10

/-- The number of people in the group -/
def people_in_group : ℕ := days_per_person / days_for_group

theorem ice_cream_group_size :
  people_in_group = 8 :=
by sorry

end ice_cream_group_size_l1065_106583


namespace larger_gate_width_l1065_106544

/-- Calculates the width of the larger gate for a rectangular garden. -/
theorem larger_gate_width
  (length : ℝ)
  (width : ℝ)
  (small_gate_width : ℝ)
  (total_fencing : ℝ)
  (h1 : length = 225)
  (h2 : width = 125)
  (h3 : small_gate_width = 3)
  (h4 : total_fencing = 687) :
  2 * (length + width) - (small_gate_width + total_fencing) = 10 :=
by sorry

end larger_gate_width_l1065_106544


namespace perfume_production_l1065_106552

/-- The number of rose petals required to make an ounce of perfume -/
def petals_per_ounce (petals_per_rose : ℕ) (roses_per_bush : ℕ) (bushes_harvested : ℕ) (bottles : ℕ) (ounces_per_bottle : ℕ) : ℕ :=
  (petals_per_rose * roses_per_bush * bushes_harvested) / (bottles * ounces_per_bottle)

/-- Theorem stating the number of rose petals required to make an ounce of perfume under given conditions -/
theorem perfume_production (petals_per_rose roses_per_bush bushes_harvested bottles ounces_per_bottle : ℕ) 
  (h1 : petals_per_rose = 8)
  (h2 : roses_per_bush = 12)
  (h3 : bushes_harvested = 800)
  (h4 : bottles = 20)
  (h5 : ounces_per_bottle = 12) :
  petals_per_ounce petals_per_rose roses_per_bush bushes_harvested bottles ounces_per_bottle = 320 := by
  sorry

end perfume_production_l1065_106552
