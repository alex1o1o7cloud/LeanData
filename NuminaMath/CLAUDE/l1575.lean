import Mathlib

namespace triangle_existence_and_area_l1575_157520

theorem triangle_existence_and_area 
  (a b c : ℝ) 
  (h : |a - Real.sqrt 8| + Real.sqrt (b^2 - 5) + (c - Real.sqrt 3)^2 = 0) : 
  ∃ (s : ℝ), s = (a + b + c) / 2 ∧ 
  a + b > c ∧ b + c > a ∧ c + a > b ∧
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) = Real.sqrt 15 / 2 := by
  sorry

end triangle_existence_and_area_l1575_157520


namespace power_fraction_equality_l1575_157519

theorem power_fraction_equality : 
  (3^2015 - 3^2013 + 3^2011) / (3^2015 + 3^2013 - 3^2011) = 73/89 := by
  sorry

end power_fraction_equality_l1575_157519


namespace line_circle_intersection_l1575_157572

/-- Line equation: ax + y - 2 = 0 -/
def line_equation (a x y : ℝ) : Prop :=
  a * x + y - 2 = 0

/-- Circle equation: (x - 1)^2 + (y - a)^2 = 16/3 -/
def circle_equation (a x y : ℝ) : Prop :=
  (x - 1)^2 + (y - a)^2 = 16/3

/-- Circle center: C(1, a) -/
def circle_center (a : ℝ) : ℝ × ℝ :=
  (1, a)

/-- Triangle ABC is equilateral -/
def is_equilateral_triangle (A B C : ℝ × ℝ) : Prop :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2

theorem line_circle_intersection (a : ℝ) :
  ∃ A B : ℝ × ℝ,
    line_equation a A.1 A.2 ∧
    line_equation a B.1 B.2 ∧
    circle_equation a A.1 A.2 ∧
    circle_equation a B.1 B.2 ∧
    is_equilateral_triangle A B (circle_center a) →
  a = 0 :=
sorry

end line_circle_intersection_l1575_157572


namespace not_subset_iff_exists_not_mem_l1575_157554

theorem not_subset_iff_exists_not_mem {M P : Set α} (hM : M.Nonempty) :
  ¬(M ⊆ P) ↔ ∃ x ∈ M, x ∉ P := by
  sorry

end not_subset_iff_exists_not_mem_l1575_157554


namespace rectangular_plot_length_difference_l1575_157511

/-- Proves that for a rectangular plot with given conditions, the length is 60 meters more than the breadth. -/
theorem rectangular_plot_length_difference (length breadth : ℝ) : 
  length = 80 ∧ 
  length > breadth ∧ 
  (4 * breadth + 2 * (length - breadth)) * 26.5 = 5300 →
  length - breadth = 60 := by
  sorry

end rectangular_plot_length_difference_l1575_157511


namespace henrys_earnings_per_lawn_l1575_157598

theorem henrys_earnings_per_lawn 
  (total_lawns : ℕ) 
  (unmowed_lawns : ℕ) 
  (total_earnings : ℕ) 
  (h1 : total_lawns = 12) 
  (h2 : unmowed_lawns = 7) 
  (h3 : total_earnings = 25) : 
  total_earnings / (total_lawns - unmowed_lawns) = 5 := by
  sorry

end henrys_earnings_per_lawn_l1575_157598


namespace students_at_start_l1575_157565

theorem students_at_start (students_left : ℕ) (new_students : ℕ) (final_students : ℕ) : 
  students_left = 4 → new_students = 42 → final_students = 48 → 
  final_students - (new_students - students_left) = 10 :=
by sorry

end students_at_start_l1575_157565


namespace tournament_committee_count_l1575_157546

/-- The number of teams in the frisbee league -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 7

/-- The number of members selected from the host team for the committee -/
def host_committee_size : ℕ := 4

/-- The number of members selected from each non-host team for the committee -/
def non_host_committee_size : ℕ := 2

/-- The total number of members in the tournament committee -/
def total_committee_size : ℕ := 12

/-- Theorem stating the total number of possible tournament committees -/
theorem tournament_committee_count :
  (num_teams : ℕ) * (Nat.choose team_size host_committee_size) *
  (Nat.choose team_size non_host_committee_size ^ (num_teams - 1)) = 340342925 := by
  sorry

end tournament_committee_count_l1575_157546


namespace total_investment_l1575_157592

theorem total_investment (T : ℝ) : T = 2000 :=
  let invested_at_8_percent : ℝ := 600
  let invested_at_10_percent : ℝ := T - 600
  let income_difference : ℝ := 92
  have h1 : 0.10 * invested_at_10_percent - 0.08 * invested_at_8_percent = income_difference := by sorry
  sorry

end total_investment_l1575_157592


namespace initial_boys_on_slide_l1575_157599

theorem initial_boys_on_slide (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  additional = 13 → total = 35 → initial + additional = total → initial = 22 := by
sorry

end initial_boys_on_slide_l1575_157599


namespace polynomial_value_at_negative_one_l1575_157505

/-- Given a polynomial function g(x) = px^4 + qx^3 + rx^2 + sx + t 
    where g(-1) = 1, prove that 16p - 8q + 4r - 2s + t = 1 -/
theorem polynomial_value_at_negative_one 
  (p q r s t : ℝ) 
  (g : ℝ → ℝ)
  (h1 : ∀ x, g x = p * x^4 + q * x^3 + r * x^2 + s * x + t)
  (h2 : g (-1) = 1) :
  16 * p - 8 * q + 4 * r - 2 * s + t = 1 := by
  sorry

end polynomial_value_at_negative_one_l1575_157505


namespace trigonometric_form_of_negative_3i_l1575_157549

theorem trigonometric_form_of_negative_3i :
  ∀ z : ℂ, z = -3 * Complex.I →
  z = 3 * (Complex.cos (3 * Real.pi / 2) + Complex.I * Complex.sin (3 * Real.pi / 2)) :=
by sorry

end trigonometric_form_of_negative_3i_l1575_157549


namespace cassidy_poster_collection_l1575_157564

/-- The number of posters Cassidy has now -/
def current_posters : ℕ := 22

/-- The number of posters Cassidy will add -/
def added_posters : ℕ := 6

/-- The number of posters Cassidy had two years ago -/
def posters_two_years_ago : ℕ := 14

theorem cassidy_poster_collection :
  2 * posters_two_years_ago = current_posters + added_posters :=
by sorry

end cassidy_poster_collection_l1575_157564


namespace ratio_of_first_to_third_term_l1575_157562

/-- An arithmetic sequence with first four terms a, y, b, 3y -/
def ArithmeticSequence (a y b : ℝ) : Prop :=
  ∃ d : ℝ, y - a = d ∧ b - y = d ∧ 3*y - b = d

theorem ratio_of_first_to_third_term (a y b : ℝ) 
  (h : ArithmeticSequence a y b) : a / b = 0 := by
  sorry

end ratio_of_first_to_third_term_l1575_157562


namespace marbles_problem_l1575_157518

theorem marbles_problem (fabian kyle miles : ℕ) : 
  fabian = 36 ∧ 
  fabian = 4 * kyle ∧ 
  fabian = 9 * miles → 
  kyle + miles = 13 := by
sorry

end marbles_problem_l1575_157518


namespace ab_plus_cd_equals_27_l1575_157566

theorem ab_plus_cd_equals_27
  (a b c d : ℝ)
  (eq1 : a + b + c = 5)
  (eq2 : a + b + d = -1)
  (eq3 : a + c + d = 8)
  (eq4 : b + c + d = 12) :
  a * b + c * d = 27 := by
sorry

end ab_plus_cd_equals_27_l1575_157566


namespace worker_c_completion_time_l1575_157590

/-- Given workers a, b, and c, and their work rates, prove that c can finish the work in 18 days -/
theorem worker_c_completion_time 
  (total_work : ℝ) 
  (work_rate_a : ℝ) 
  (work_rate_b : ℝ) 
  (work_rate_c : ℝ) 
  (h1 : work_rate_a + work_rate_b + work_rate_c = total_work / 4)
  (h2 : work_rate_a = total_work / 12)
  (h3 : work_rate_b = total_work / 9) :
  work_rate_c = total_work / 18 := by
sorry


end worker_c_completion_time_l1575_157590


namespace f_properties_l1575_157580

def f (x m : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + m

theorem f_properties (m : ℝ) :
  (∀ x y, x < y ∧ ((x < -1 ∧ y < -1) ∨ (x > 3 ∧ y > 3)) → f x m > f y m) ∧
  (∃ x₀ ∈ Set.Icc (-2) 2, ∀ x ∈ Set.Icc (-2) 2, f x m ≤ f x₀ m) ∧
  f x₀ m = 20 →
  ∃ x₁ ∈ Set.Icc (-2) 2, ∀ x ∈ Set.Icc (-2) 2, f x m ≥ f x₁ m ∧ f x₁ m = -7 :=
by sorry

end f_properties_l1575_157580


namespace sugar_content_per_bar_l1575_157591

/-- The sugar content of each chocolate bar -/
def sugar_per_bar (total_sugar total_bars lollipop_sugar : ℕ) : ℚ :=
  (total_sugar - lollipop_sugar) / total_bars

/-- Proof that the sugar content of each chocolate bar is 10 grams -/
theorem sugar_content_per_bar :
  sugar_per_bar 177 14 37 = 10 := by
  sorry

end sugar_content_per_bar_l1575_157591


namespace odd_function_with_period_two_negation_at_six_l1575_157581

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period_two_negation (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = -f x

theorem odd_function_with_period_two_negation_at_six
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_period : has_period_two_negation f) :
  f 6 = 0 := by
sorry

end odd_function_with_period_two_negation_at_six_l1575_157581


namespace mean_squared_sum_l1575_157585

theorem mean_squared_sum (x y z : ℝ) 
  (h_arithmetic : (x + y + z) / 3 = 10)
  (h_geometric : (x * y * z) ^ (1/3 : ℝ) = 6)
  (h_harmonic : 3 / (1/x + 1/y + 1/z) = 4) :
  x^2 + y^2 + z^2 = 576 := by
sorry

end mean_squared_sum_l1575_157585


namespace randys_trip_length_l1575_157547

theorem randys_trip_length :
  ∀ (total_length : ℚ),
  (1 / 4 : ℚ) * total_length +  -- First part (gravel road)
  30 +                          -- Second part (pavement)
  (1 / 6 : ℚ) * total_length    -- Third part (dirt road)
  = total_length                -- Sum of all parts equals total length
  →
  total_length = 360 / 7 := by
sorry

end randys_trip_length_l1575_157547


namespace complement_of_A_l1575_157512

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | (x - 1) * (x + 2) > 0}

theorem complement_of_A : Set.compl A = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end complement_of_A_l1575_157512


namespace three_numbers_proof_l1575_157502

theorem three_numbers_proof (a b c : ℕ) (h1 : a < b) (h2 : b < c) 
  (h3 : (a + b + c) / 3 = b) (h4 : c - a = 321) (h5 : a + c = 777) : 
  a = 228 ∧ b = 549 ∧ c = 870 := by
sorry

end three_numbers_proof_l1575_157502


namespace tan_inequality_l1575_157571

theorem tan_inequality (n : ℕ) (x : ℝ) (h1 : 0 < x) (h2 : x < π / (2 * n)) :
  (1/2) * (Real.tan x + Real.tan (n * x) - Real.tan ((n - 1) * x)) > (1/n) * Real.tan (n * x) := by
  sorry

end tan_inequality_l1575_157571


namespace max_xy_value_max_xy_achieved_l1575_157513

theorem max_xy_value (x y : ℕ+) (h : 7 * x + 4 * y = 140) : x * y ≤ 168 := by
  sorry

theorem max_xy_achieved : ∃ (x y : ℕ+), 7 * x + 4 * y = 140 ∧ x * y = 168 := by
  sorry

end max_xy_value_max_xy_achieved_l1575_157513


namespace tan_22_5_deg_decomposition_l1575_157560

theorem tan_22_5_deg_decomposition :
  ∃ (a b c d : ℕ+),
    (Real.tan (22.5 * π / 180) = (a : ℝ).sqrt - b + (c : ℝ).sqrt - (d : ℝ).sqrt) ∧
    (a ≥ b) ∧ (b ≥ c) ∧ (c ≥ d) ∧
    (a + b + c + d = 3) := by
  sorry

end tan_22_5_deg_decomposition_l1575_157560


namespace x_over_y_value_l1575_157558

theorem x_over_y_value (x y : ℝ) (h1 : 3 < (x - y) / (x + y)) (h2 : (x - y) / (x + y) < 6) 
  (h3 : ∃ (n : ℤ), x / y = n) : x / y = -2 := by
  sorry

end x_over_y_value_l1575_157558


namespace subset_union_of_product_zero_l1575_157506

variable {X : Type*}
variable (f g : X → ℝ)

def M (f : X → ℝ) := {x : X | f x = 0}
def N (g : X → ℝ) := {x : X | g x = 0}
def P (f g : X → ℝ) := {x : X | f x * g x = 0}

theorem subset_union_of_product_zero (hM : M f ≠ ∅) (hN : N g ≠ ∅) (hP : P f g ≠ ∅) :
  P f g ⊆ M f ∪ N g := by
  sorry

end subset_union_of_product_zero_l1575_157506


namespace modular_inverse_11_mod_1033_l1575_157507

theorem modular_inverse_11_mod_1033 : ∃ x : ℕ, x < 1033 ∧ (11 * x) % 1033 = 1 :=
by
  use 94
  sorry

end modular_inverse_11_mod_1033_l1575_157507


namespace apples_in_shop_l1575_157575

/-- Given a ratio of fruits and the number of mangoes, calculate the number of apples -/
def calculate_apples (mango_ratio : ℕ) (orange_ratio : ℕ) (apple_ratio : ℕ) (mango_count : ℕ) : ℕ :=
  (mango_count / mango_ratio) * apple_ratio

/-- Theorem: Given the ratio 10:2:3 for mangoes:oranges:apples and 120 mangoes, there are 36 apples -/
theorem apples_in_shop :
  calculate_apples 10 2 3 120 = 36 := by
  sorry

end apples_in_shop_l1575_157575


namespace triangle_property_l1575_157534

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C,
    prove that if sin A(a^2 + b^2 - c^2) = ab(2sin B - sin C),
    then A = π/3 and 3/2 < sin B + sin C ≤ √3 -/
theorem triangle_property (a b c A B C : ℝ) 
  (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
  (h_triangle : A + B + C = π)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_condition : Real.sin A * (a^2 + b^2 - c^2) = a * b * (2 * Real.sin B - Real.sin C)) :
  A = π/3 ∧ 3/2 < Real.sin B + Real.sin C ∧ Real.sin B + Real.sin C ≤ Real.sqrt 3 := by
  sorry

end triangle_property_l1575_157534


namespace bus_ride_difference_l1575_157543

/-- Given Oscar's and Charlie's bus ride lengths, prove the difference between them -/
theorem bus_ride_difference (oscar_ride : ℝ) (charlie_ride : ℝ)
  (h1 : oscar_ride = 0.75)
  (h2 : charlie_ride = 0.25) :
  oscar_ride - charlie_ride = 0.50 := by
sorry

end bus_ride_difference_l1575_157543


namespace points_in_first_quadrant_l1575_157550

theorem points_in_first_quadrant (x y : ℝ) : 
  y > -x + 3 ∧ y > 3*x - 1 → x > 0 ∧ y > 0 :=
sorry

end points_in_first_quadrant_l1575_157550


namespace student_rabbit_difference_is_95_l1575_157552

/-- The number of students in each fourth-grade classroom -/
def students_per_classroom : ℕ := 22

/-- The number of rabbits in each fourth-grade classroom -/
def rabbits_per_classroom : ℕ := 3

/-- The number of fourth-grade classrooms -/
def number_of_classrooms : ℕ := 5

/-- The difference between the total number of students and rabbits in all classrooms -/
def student_rabbit_difference : ℕ :=
  (students_per_classroom * number_of_classrooms) - (rabbits_per_classroom * number_of_classrooms)

theorem student_rabbit_difference_is_95 :
  student_rabbit_difference = 95 := by sorry

end student_rabbit_difference_is_95_l1575_157552


namespace arcade_vending_machines_total_beverages_in_arcade_l1575_157535

/-- Given the conditions of vending machines in an arcade, calculate the total number of beverages --/
theorem arcade_vending_machines (num_machines : ℕ) 
  (front_position : ℕ) (back_position : ℕ) 
  (top_position : ℕ) (bottom_position : ℕ) : ℕ :=
  let beverages_per_column := front_position + back_position - 1
  let rows_per_machine := top_position + bottom_position - 1
  let beverages_per_machine := beverages_per_column * rows_per_machine
  num_machines * beverages_per_machine

/-- Prove that the total number of beverages in the arcade is 3696 --/
theorem total_beverages_in_arcade : 
  arcade_vending_machines 28 14 20 3 2 = 3696 := by
  sorry

end arcade_vending_machines_total_beverages_in_arcade_l1575_157535


namespace volume_to_surface_area_ratio_l1575_157555

/-- A shape formed by unit cubes in a straight line -/
structure LineShape where
  num_cubes : ℕ

/-- Volume of a LineShape -/
def volume (shape : LineShape) : ℕ :=
  shape.num_cubes

/-- Surface area of a LineShape -/
def surface_area (shape : LineShape) : ℕ :=
  2 * 5 + (shape.num_cubes - 2) * 4

/-- Theorem stating the ratio of volume to surface area for a LineShape with 8 cubes -/
theorem volume_to_surface_area_ratio (shape : LineShape) (h : shape.num_cubes = 8) :
  (volume shape : ℚ) / (surface_area shape : ℚ) = 4 / 17 := by
  sorry


end volume_to_surface_area_ratio_l1575_157555


namespace circumscribed_sphere_surface_area_is_20pi_l1575_157540

/-- Represents a triangular pyramid with vertex P and base ABC. -/
structure TriangularPyramid where
  PA : ℝ
  AB : ℝ
  angleCBA : ℝ
  perpendicular : Bool

/-- Calculates the surface area of the circumscribed sphere of a triangular pyramid. -/
def circumscribedSphereSurfaceArea (pyramid : TriangularPyramid) : ℝ :=
  sorry

/-- Theorem: The surface area of the circumscribed sphere of the given triangular pyramid is 20π. -/
theorem circumscribed_sphere_surface_area_is_20pi :
  let pyramid := TriangularPyramid.mk 2 2 (π/6) true
  circumscribedSphereSurfaceArea pyramid = 20 * π :=
by sorry

end circumscribed_sphere_surface_area_is_20pi_l1575_157540


namespace sum_of_combinations_l1575_157525

theorem sum_of_combinations : Nat.choose 8 2 + Nat.choose 8 3 = 84 := by sorry

end sum_of_combinations_l1575_157525


namespace five_sundays_in_july_l1575_157504

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month -/
structure Month where
  days : ℕ
  first_day : DayOfWeek

/-- Given a day of the week, returns the next day -/
def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Counts the occurrences of a specific day in a month -/
def count_day_occurrences (m : Month) (d : DayOfWeek) : ℕ :=
  sorry

/-- Theorem: If June has five Fridays and 30 days, July (with 31 days) must have five Sundays -/
theorem five_sundays_in_july 
  (june : Month) 
  (july : Month) 
  (h1 : june.days = 30)
  (h2 : july.days = 31)
  (h3 : count_day_occurrences june DayOfWeek.Friday = 5)
  (h4 : july.first_day = next_day june.first_day) :
  count_day_occurrences july DayOfWeek.Sunday = 5 :=
sorry

end five_sundays_in_july_l1575_157504


namespace projectile_speed_proof_l1575_157522

/-- Proves that the speed of the first projectile is 445 km/h given the problem conditions -/
theorem projectile_speed_proof (v : ℝ) : 
  (v + 545) * (84 / 60) = 1386 → v = 445 := by
  sorry

end projectile_speed_proof_l1575_157522


namespace grocer_sale_problem_l1575_157517

theorem grocer_sale_problem (sale1 sale2 sale3 sale5 average_sale : ℕ) 
  (h1 : sale1 = 5700)
  (h2 : sale2 = 8550)
  (h3 : sale3 = 6855)
  (h5 : sale5 = 14045)
  (h_avg : average_sale = 7800) :
  ∃ sale4 : ℕ, 
    sale4 = 3850 ∧ 
    (sale1 + sale2 + sale3 + sale4 + sale5) / 5 = average_sale :=
by sorry

end grocer_sale_problem_l1575_157517


namespace minimize_y_l1575_157542

/-- The function y in terms of x, a, and b -/
def y (x a b : ℝ) : ℝ := (x - a)^2 + (x - b)^2

/-- The theorem stating that (a+b)/2 minimizes y -/
theorem minimize_y (a b : ℝ) :
  ∃ (x_min : ℝ), ∀ (x : ℝ), y x_min a b ≤ y x a b ∧ x_min = (a + b) / 2 := by
  sorry

end minimize_y_l1575_157542


namespace rectangular_solid_volume_l1575_157584

/-- The volume of a rectangular solid with side lengths 1 m, 20 cm, and 50 cm is 100000 cm³ -/
theorem rectangular_solid_volume : 
  let length_m : ℝ := 1
  let width_cm : ℝ := 20
  let height_cm : ℝ := 50
  let m_to_cm : ℝ := 100
  (length_m * m_to_cm * width_cm * height_cm) = 100000 := by
sorry

end rectangular_solid_volume_l1575_157584


namespace smallest_congruent_difference_l1575_157589

theorem smallest_congruent_difference : ∃ m n : ℕ,
  (m ≥ 100 ∧ m < 1000 ∧ m % 13 = 3 ∧ ∀ k, k ≥ 100 ∧ k < 1000 ∧ k % 13 = 3 → m ≤ k) ∧
  (n ≥ 1000 ∧ n < 10000 ∧ n % 13 = 3 ∧ ∀ l, l ≥ 1000 ∧ l < 10000 ∧ l % 13 = 3 → n ≤ l) ∧
  n - m = 896 :=
by sorry

end smallest_congruent_difference_l1575_157589


namespace inequality_proof_l1575_157545

theorem inequality_proof (a b x y z : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  x / (a * y + b * z) + y / (a * z + b * x) + z / (a * x + b * y) ≥ 3 / (a + b) := by
  sorry

end inequality_proof_l1575_157545


namespace airplane_passengers_survey_is_census_l1575_157537

/-- A survey type -/
inductive SurveyType
| FrozenFood
| AirplanePassengers
| RefrigeratorLifespan
| EnvironmentalAwareness

/-- Predicate for whether a survey requires examining every individual -/
def requiresExaminingAll (s : SurveyType) : Prop :=
  match s with
  | .AirplanePassengers => True
  | _ => False

/-- Definition of a census -/
def isCensus (s : SurveyType) : Prop :=
  requiresExaminingAll s

theorem airplane_passengers_survey_is_census :
  isCensus SurveyType.AirplanePassengers := by
  sorry

end airplane_passengers_survey_is_census_l1575_157537


namespace mans_speed_with_current_l1575_157515

/-- Calculates the man's speed with the current given his speed against the current and the current's speed. -/
def speed_with_current (speed_against_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_against_current + 2 * current_speed

/-- Theorem stating that given the man's speed against the current and the current's speed,
    the man's speed with the current is 12 km/hr. -/
theorem mans_speed_with_current :
  speed_with_current 8 2 = 12 := by
  sorry

#eval speed_with_current 8 2

end mans_speed_with_current_l1575_157515


namespace probability_second_science_question_l1575_157594

/-- Given a set of questions with science and humanities questions,
    prove the probability of drawing a second science question
    after drawing a science question first. -/
theorem probability_second_science_question
  (total_questions : ℕ)
  (science_questions : ℕ)
  (humanities_questions : ℕ)
  (h1 : total_questions = 6)
  (h2 : science_questions = 4)
  (h3 : humanities_questions = 2)
  (h4 : total_questions = science_questions + humanities_questions)
  (h5 : science_questions > 0) :
  (science_questions - 1 : ℚ) / (total_questions - 1) = 3/5 := by
sorry

end probability_second_science_question_l1575_157594


namespace square_area_error_l1575_157561

theorem square_area_error (s : ℝ) (s' : ℝ) (h : s' = s * (1 + 0.02)) :
  (s'^2 - s^2) / s^2 * 100 = 4.04 := by
  sorry

end square_area_error_l1575_157561


namespace line_equation_l1575_157530

/-- The ellipse in the first quadrant -/
def ellipse (x y : ℝ) : Prop := x^2/6 + y^2/3 = 1 ∧ x > 0 ∧ y > 0

/-- The line l -/
def line_l (x y : ℝ) : Prop := ∃ (k m : ℝ), y = k*x + m ∧ k < 0 ∧ m > 0

/-- Points A and B on the ellipse and line l -/
def point_A_B (xA yA xB yB : ℝ) : Prop :=
  ellipse xA yA ∧ ellipse xB yB ∧ line_l xA yA ∧ line_l xB yB

/-- Points M and N on the axes -/
def point_M_N (xM yM xN yN : ℝ) : Prop :=
  xM < 0 ∧ yM = 0 ∧ xN = 0 ∧ yN > 0 ∧ line_l xM yM ∧ line_l xN yN

/-- Equal distances |MA| = |NB| -/
def equal_distances (xA yA xB yB xM yM xN yN : ℝ) : Prop :=
  (xA - xM)^2 + yA^2 = xB^2 + (yB - yN)^2

/-- Distance |MN| = 2√3 -/
def distance_MN (xM yM xN yN : ℝ) : Prop :=
  (xM - xN)^2 + (yM - yN)^2 = 12

theorem line_equation (xA yA xB yB xM yM xN yN : ℝ) :
  point_A_B xA yA xB yB →
  point_M_N xM yM xN yN →
  equal_distances xA yA xB yB xM yM xN yN →
  distance_MN xM yM xN yN →
  ∃ (x y : ℝ), x + Real.sqrt 2 * y - 2 * Real.sqrt 2 = 0 ∧ line_l x y :=
sorry

end line_equation_l1575_157530


namespace solution_set_implies_a_b_values_f_1_negative_implies_a_conditions_l1575_157570

-- Define the function f
def f (a b x : ℝ) : ℝ := -3 * x^2 + a * (5 - a) * x + b

-- Part 1
theorem solution_set_implies_a_b_values (a b : ℝ) :
  (∀ x : ℝ, f a b x > 0 ↔ -1 < x ∧ x < 3) →
  ((a = 2 ∧ b = 9) ∨ (a = 3 ∧ b = 9)) :=
sorry

-- Part 2
theorem f_1_negative_implies_a_conditions (b : ℝ) :
  (∀ a : ℝ, f a b 1 < 0 ↔
    (b < -13/4 ∧ a ∈ Set.univ) ∨
    (b = -13/4 ∧ a ≠ 5/2) ∨
    (b > -13/4 ∧ (a > (5 + Real.sqrt (4*b + 13))/2 ∨ a < (5 - Real.sqrt (4*b + 13))/2))) :=
sorry

end solution_set_implies_a_b_values_f_1_negative_implies_a_conditions_l1575_157570


namespace triangle_8_8_15_l1575_157593

/-- Triangle Inequality Theorem: A set of three line segments can form a triangle
    if and only if the sum of the lengths of any two sides is greater than
    the length of the remaining side. -/
def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The set of line segments with lengths 8cm, 8cm, and 15cm can form a triangle. -/
theorem triangle_8_8_15 : canFormTriangle 8 8 15 := by
  sorry

end triangle_8_8_15_l1575_157593


namespace shelter_puppies_count_l1575_157563

theorem shelter_puppies_count :
  ∀ (puppies kittens : ℕ),
    kittens = 2 * puppies + 14 →
    puppies > 0 →
    kittens = 78 →
    puppies = 32 := by
  sorry

end shelter_puppies_count_l1575_157563


namespace card_drawing_certainty_l1575_157582

theorem card_drawing_certainty (total : ℕ) (hearts clubs spades drawn : ℕ) 
  (h_total : total = hearts + clubs + spades)
  (h_hearts : hearts = 5)
  (h_clubs : clubs = 4)
  (h_spades : spades = 3)
  (h_drawn : drawn = 10) :
  ∀ (draw : Finset ℕ), draw.card = drawn → 
    (∃ (h c s : ℕ), h ∈ draw ∧ c ∈ draw ∧ s ∈ draw ∧ 
      h ≤ hearts ∧ c ≤ clubs ∧ s ≤ spades) :=
sorry

end card_drawing_certainty_l1575_157582


namespace parallel_vectors_x_value_l1575_157510

/-- Given two vectors a and b in ℝ², if they are parallel and a = (4,2) and b = (x,3), then x = 6. -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![4, 2]
  let b : Fin 2 → ℝ := ![x, 3]
  (∃ (k : ℝ), b = k • a) → x = 6 := by
  sorry

end parallel_vectors_x_value_l1575_157510


namespace scissors_count_l1575_157586

/-- The total number of scissors after addition -/
def total_scissors (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem stating that the total number of scissors is 52 -/
theorem scissors_count : total_scissors 39 13 = 52 := by
  sorry

end scissors_count_l1575_157586


namespace proposition_logic_l1575_157568

theorem proposition_logic (p q : Prop) (hp : p ↔ (3 + 3 = 5)) (hq : q ↔ (6 > 3)) :
  (p ∨ q) ∧ (¬q ↔ False) := by
  sorry

end proposition_logic_l1575_157568


namespace trig_identities_l1575_157577

/-- Prove trigonometric identities -/
theorem trig_identities :
  let π : ℝ := Real.pi
  let cos_45 : ℝ := Real.cos (π / 4)
  let tan_30 : ℝ := Real.tan (π / 6)
  let cos_30 : ℝ := Real.cos (π / 6)
  let sin_60 : ℝ := Real.sin (π / 3)
  let sin_30 : ℝ := Real.sin (π / 6)
  let tan_60 : ℝ := Real.tan (π / 3)
  2 * cos_45 - (3 / 2) * tan_30 * cos_30 + sin_60 ^ 2 = Real.sqrt 2 ∧
  (sin_30)⁻¹ * (sin_60 - cos_45) - Real.sqrt ((1 - tan_60) ^ 2) = 1 - Real.sqrt 2 := by
  sorry

end trig_identities_l1575_157577


namespace additional_surcharge_l1575_157559

/-- Calculates the additional surcharge for a special project given the tax information --/
theorem additional_surcharge (tax_1995 tax_1996 : ℕ) (increase_rate : ℚ) : 
  tax_1995 = 1800 →
  increase_rate = 6 / 100 →
  tax_1996 = 2108 →
  (tax_1996 : ℚ) = (tax_1995 : ℚ) * (1 + increase_rate) + 200 := by
  sorry

end additional_surcharge_l1575_157559


namespace smallest_multiple_l1575_157551

theorem smallest_multiple : ∃ (a : ℕ), 
  (a % 3 = 0) ∧ 
  ((a - 1) % 4 = 0) ∧ 
  ((a - 2) % 5 = 0) ∧ 
  (∀ b : ℕ, b < a → ¬((b % 3 = 0) ∧ ((b - 1) % 4 = 0) ∧ ((b - 2) % 5 = 0))) ∧
  a = 57 := by
sorry

end smallest_multiple_l1575_157551


namespace geometric_sequence_b_value_l1575_157583

/-- Given a geometric sequence with first term 120, second term b, and third term 60/24,
    prove that b = 10√3 when b is positive. -/
theorem geometric_sequence_b_value (b : ℝ) (h1 : b > 0) 
    (h2 : ∃ (r : ℝ), 120 * r = b ∧ b * r = 60 / 24) : b = 10 * Real.sqrt 3 := by
  sorry

end geometric_sequence_b_value_l1575_157583


namespace stratified_sampling_size_l1575_157516

/-- Represents a workshop with its production quantity -/
structure Workshop where
  quantity : ℕ

/-- Calculates the total sample size for stratified sampling -/
def calculateSampleSize (workshops : List Workshop) (sampledUnits : ℕ) (sampledWorkshopQuantity : ℕ) : ℕ :=
  let totalQuantity := workshops.map (·.quantity) |>.sum
  (sampledUnits * totalQuantity) / sampledWorkshopQuantity

theorem stratified_sampling_size :
  let workshops := [
    { quantity := 120 },  -- Workshop A
    { quantity := 80 },   -- Workshop B
    { quantity := 60 }    -- Workshop C
  ]
  let sampledUnits := 3
  let sampledWorkshopQuantity := 60
  calculateSampleSize workshops sampledUnits sampledWorkshopQuantity = 13 := by
  sorry


end stratified_sampling_size_l1575_157516


namespace star_three_two_l1575_157529

/-- The star operation defined as a^3 + 3a^2b + 3ab^2 + b^3 -/
def star (a b : ℝ) : ℝ := a^3 + 3*a^2*b + 3*a*b^2 + b^3

/-- Theorem stating that 3 star 2 equals 125 -/
theorem star_three_two : star 3 2 = 125 := by
  sorry

end star_three_two_l1575_157529


namespace two_digit_product_less_than_five_digit_l1575_157533

theorem two_digit_product_less_than_five_digit : ∀ a b : ℕ,
  10 ≤ a ∧ a ≤ 99 →
  10 ≤ b ∧ b ≤ 99 →
  a * b < 10000 := by
sorry

end two_digit_product_less_than_five_digit_l1575_157533


namespace energy_drink_cost_l1575_157509

theorem energy_drink_cost (cupcakes : Nat) (cupcake_price : ℚ) 
  (cookies : Nat) (cookie_price : ℚ) (basketballs : Nat) 
  (basketball_price : ℚ) (energy_drinks : Nat) :
  cupcakes = 50 →
  cupcake_price = 2 →
  cookies = 40 →
  cookie_price = 1/2 →
  basketballs = 2 →
  basketball_price = 40 →
  energy_drinks = 20 →
  (cupcakes * cupcake_price + cookies * cookie_price - basketballs * basketball_price) / energy_drinks = 2 := by
sorry


end energy_drink_cost_l1575_157509


namespace perpendicular_planes_from_perpendicular_lines_l1575_157556

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation for lines and planes
variable (perpendicular_lines : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Define the property of a line being outside a plane
variable (outside : Line → Plane → Prop)

-- Theorem statement
theorem perpendicular_planes_from_perpendicular_lines
  (α β : Plane) (m n : Line)
  (different_planes : α ≠ β)
  (distinct_lines : m ≠ n)
  (m_outside_α : outside m α)
  (m_outside_β : outside m β)
  (n_outside_α : outside n α)
  (n_outside_β : outside n β)
  (h1 : perpendicular_lines m n)
  (h3 : perpendicular_line_plane n β)
  (h4 : perpendicular_line_plane m α) :
  perpendicular_planes α β :=
sorry

end perpendicular_planes_from_perpendicular_lines_l1575_157556


namespace solve_equation_l1575_157521

theorem solve_equation : ∃ x : ℝ, (2 * x + 5) / 7 = 15 ∧ x = 50 := by
  sorry

end solve_equation_l1575_157521


namespace closest_multiple_of_17_to_2502_l1575_157528

theorem closest_multiple_of_17_to_2502 :
  ∀ k : ℤ, k ≠ 147 → |2502 - 17 * 147| ≤ |2502 - 17 * k| :=
sorry

end closest_multiple_of_17_to_2502_l1575_157528


namespace negative_fraction_comparison_l1575_157553

theorem negative_fraction_comparison : -1/3 < -1/4 := by
  sorry

end negative_fraction_comparison_l1575_157553


namespace cookies_problem_l1575_157573

/-- Calculates the number of cookies taken out in four days given the initial count,
    remaining count after a week, and assuming equal daily removal. -/
def cookies_taken_in_four_days (initial : ℕ) (remaining : ℕ) : ℕ :=
  let total_taken := initial - remaining
  let daily_taken := total_taken / 7
  4 * daily_taken

/-- Proves that given 70 initial cookies and 28 remaining after a week,
    Paul took out 24 cookies in four days. -/
theorem cookies_problem :
  cookies_taken_in_four_days 70 28 = 24 := by
  sorry

end cookies_problem_l1575_157573


namespace max_log_sum_l1575_157557

theorem max_log_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 4 * y = 40) :
  ∃ (max_val : ℝ), max_val = 2 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a + 4 * b = 40 → Real.log a + Real.log b ≤ max_val := by
  sorry

end max_log_sum_l1575_157557


namespace geometric_sequence_fourth_term_l1575_157596

/-- A geometric sequence with first term x, second term 3x+3, and third term 6x+6 has fourth term -24 -/
theorem geometric_sequence_fourth_term :
  ∀ x : ℝ,
  let a₁ : ℝ := x
  let a₂ : ℝ := 3*x + 3
  let a₃ : ℝ := 6*x + 6
  let r : ℝ := a₂ / a₁
  (a₂ = r * a₁) ∧ (a₃ = r * a₂) →
  r * a₃ = -24 :=
by
  sorry

end geometric_sequence_fourth_term_l1575_157596


namespace currency_conversion_l1575_157538

-- Define the conversion rates
def cents_per_jiao : ℝ := 10
def cents_per_yuan : ℝ := 100

-- Define the theorem
theorem currency_conversion :
  (5 / cents_per_jiao = 0.5) ∧ 
  (5 / cents_per_yuan = 0.05) ∧ 
  (3.25 * cents_per_yuan = 325) := by
sorry


end currency_conversion_l1575_157538


namespace sum_divisibility_l1575_157576

theorem sum_divisibility (x y a b S : ℤ) 
  (sum_eq : x + y = S) 
  (masha_div : S ∣ (a * x + b * y)) : 
  S ∣ (b * x + a * y) := by
sorry

end sum_divisibility_l1575_157576


namespace work_completion_time_l1575_157569

/-- Given workers a and b, where:
    - a can complete the work in 20 days
    - a and b together can complete the work in 15 days when b works half-time
    Prove that a and b together can complete the work in 12 days when b works full-time -/
theorem work_completion_time (a b : ℝ) 
  (ha : a = 1 / 20)  -- a's work rate per day
  (hab_half : a + b / 2 = 1 / 15)  -- combined work rate when b works half-time
  : a + b = 1 / 12 := by  -- combined work rate when b works full-time
  sorry

end work_completion_time_l1575_157569


namespace ab_zero_necessary_not_sufficient_l1575_157514

def f (a b x : ℝ) : ℝ := x * abs (x + a) + b

theorem ab_zero_necessary_not_sufficient (a b : ℝ) :
  (∀ x, f a b x = -f a b (-x)) → a * b = 0 ∧
  ∃ a b, a * b = 0 ∧ ¬(∀ x, f a b x = -f a b (-x)) :=
sorry

end ab_zero_necessary_not_sufficient_l1575_157514


namespace tire_circumference_l1575_157523

/-- The circumference of a tire given its rotations per minute and the car's speed -/
theorem tire_circumference (rotations_per_minute : ℝ) (car_speed_kmh : ℝ) : 
  rotations_per_minute = 400 → car_speed_kmh = 24 → 
  (car_speed_kmh * 1000 / 60) / rotations_per_minute = 1 := by
  sorry

#check tire_circumference

end tire_circumference_l1575_157523


namespace equal_angles_with_vectors_l1575_157532

/-- Given two vectors a and b in ℝ², prove that the vector c satisfies the condition
    that the angle between c and a is equal to the angle between c and b. -/
theorem equal_angles_with_vectors (a b c : ℝ × ℝ) : 
  a = (1, 0) → b = (1, -Real.sqrt 3) → c = (Real.sqrt 3, -1) →
  (c.1 * a.1 + c.2 * a.2) / (Real.sqrt (c.1^2 + c.2^2) * Real.sqrt (a.1^2 + a.2^2)) =
  (c.1 * b.1 + c.2 * b.2) / (Real.sqrt (c.1^2 + c.2^2) * Real.sqrt (b.1^2 + b.2^2)) := by
  sorry

end equal_angles_with_vectors_l1575_157532


namespace square_root_of_8_l1575_157548

-- Define the square root property
def is_square_root (x : ℝ) (y : ℝ) : Prop := x * x = y

-- Theorem statement
theorem square_root_of_8 :
  ∃ (x : ℝ), is_square_root x 8 ∧ x = Real.sqrt 8 ∨ x = -Real.sqrt 8 :=
by sorry

end square_root_of_8_l1575_157548


namespace complex_number_in_fourth_quadrant_l1575_157574

/-- The complex number z = (1-3i)(2+i) is located in the fourth quadrant of the complex plane. -/
theorem complex_number_in_fourth_quadrant : 
  let z : ℂ := (1 - 3*Complex.I) * (2 + Complex.I)
  z.re > 0 ∧ z.im < 0 :=
by sorry

end complex_number_in_fourth_quadrant_l1575_157574


namespace greater_solution_quadratic_l1575_157527

theorem greater_solution_quadratic : 
  ∃ (x : ℝ), x^2 + 14*x - 88 = 0 ∧ 
  (∀ (y : ℝ), y^2 + 14*y - 88 = 0 → y ≤ x) ∧
  x = 4 := by
  sorry

end greater_solution_quadratic_l1575_157527


namespace store_discount_percentage_l1575_157579

/-- Proves that the discount percentage is 9% given the specified markups and profit -/
theorem store_discount_percentage (C : ℝ) (h : C > 0) : 
  let initial_price := 1.20 * C
  let marked_up_price := 1.25 * initial_price
  let final_profit := 0.365 * C
  ∃ (D : ℝ), 
    marked_up_price * (1 - D) - C = final_profit ∧ 
    D = 0.09 := by
  sorry

end store_discount_percentage_l1575_157579


namespace box_surface_area_and_volume_l1575_157587

/-- Represents the dimensions of a rectangular sheet and the size of square corners to be removed --/
structure BoxParameters where
  length : ℕ
  width : ℕ
  corner_size : ℕ

/-- Calculates the surface area of the interior of the box --/
def calculate_surface_area (params : BoxParameters) : ℕ :=
  params.length * params.width - 4 * params.corner_size * params.corner_size

/-- Calculates the volume of the box --/
def calculate_volume (params : BoxParameters) : ℕ :=
  (params.length - 2 * params.corner_size) * (params.width - 2 * params.corner_size) * params.corner_size

/-- Theorem stating the surface area and volume of the box --/
theorem box_surface_area_and_volume :
  let params : BoxParameters := { length := 25, width := 35, corner_size := 6 }
  calculate_surface_area params = 731 ∧ calculate_volume params = 1794 :=
by sorry

end box_surface_area_and_volume_l1575_157587


namespace det_sum_of_matrices_l1575_157500

def A : Matrix (Fin 2) (Fin 2) ℤ := !![5, -2; 3, 4]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![1, 3; -1, 2]

theorem det_sum_of_matrices : Matrix.det (A + B) = 34 := by sorry

end det_sum_of_matrices_l1575_157500


namespace angle_terminal_side_x_value_l1575_157544

theorem angle_terminal_side_x_value (x : ℝ) (θ : ℝ) :
  x < 0 →
  (∃ y : ℝ, y = 3 ∧ (x^2 + y^2).sqrt * Real.cos θ = x) →
  Real.cos θ = (Real.sqrt 10 / 10) * x →
  x = -1 :=
by sorry

end angle_terminal_side_x_value_l1575_157544


namespace largest_angle_in_pentagon_l1575_157503

/-- The measure of the largest angle in a pentagon ABCDE with specific angle conditions -/
theorem largest_angle_in_pentagon (A B C D E : ℝ) : 
  A = 108 ∧ 
  B = 72 ∧ 
  C = D ∧ 
  E = 3 * C ∧ 
  A + B + C + D + E = 540 →
  (max A (max B (max C (max D E)))) = 216 := by
  sorry

end largest_angle_in_pentagon_l1575_157503


namespace inscribed_square_area_ratio_l1575_157539

/-- Given a square ABCD with side length s, and an inscribed square A'B'C'D' where each vertex
    of A'B'C'D' is on a diagonal of ABCD and equidistant from the center of ABCD,
    the area of A'B'C'D' is 1/5 of the area of ABCD. -/
theorem inscribed_square_area_ratio (s : ℝ) (h : s > 0) :
  let abcd_area := s^2
  let apbpcpdp_side := s / Real.sqrt 5
  let apbpcpdp_area := apbpcpdp_side^2
  apbpcpdp_area / abcd_area = 1 / 5 := by
  sorry


end inscribed_square_area_ratio_l1575_157539


namespace burger_combinations_count_l1575_157531

/-- The number of condiment options available. -/
def num_condiments : ℕ := 10

/-- The number of choices for meat patties. -/
def num_patty_choices : ℕ := 3

/-- The number of choices for bun types. -/
def num_bun_choices : ℕ := 2

/-- Calculates the total number of different burger combinations. -/
def total_burger_combinations : ℕ := 2^num_condiments * num_patty_choices * num_bun_choices

/-- Theorem stating that the total number of different burger combinations is 6144. -/
theorem burger_combinations_count : total_burger_combinations = 6144 := by
  sorry

end burger_combinations_count_l1575_157531


namespace platform_length_l1575_157567

/-- The length of a platform given train crossing times -/
theorem platform_length (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) :
  train_length = 420 ∧ platform_time = 60 ∧ pole_time = 30 →
  (train_length / pole_time) * platform_time - train_length = 420 := by
  sorry

#check platform_length

end platform_length_l1575_157567


namespace tan_alpha_plus_pi_fourth_l1575_157501

theorem tan_alpha_plus_pi_fourth (α : Real) (h : Real.tan α = 2) : 
  Real.tan (α + π/4) = -3 := by sorry

end tan_alpha_plus_pi_fourth_l1575_157501


namespace purple_marble_probability_l1575_157524

structure Bag where
  red : ℕ
  green : ℕ
  orange : ℕ
  purple : ℕ

def bagX : Bag := { red := 5, green := 3, orange := 0, purple := 0 }
def bagY : Bag := { red := 0, green := 0, orange := 8, purple := 2 }
def bagZ : Bag := { red := 0, green := 0, orange := 3, purple := 7 }

def total_marbles (b : Bag) : ℕ := b.red + b.green + b.orange + b.purple

def prob_red (b : Bag) : ℚ := b.red / (total_marbles b)
def prob_green (b : Bag) : ℚ := b.green / (total_marbles b)
def prob_purple (b : Bag) : ℚ := b.purple / (total_marbles b)

theorem purple_marble_probability :
  let p_red_X := prob_red bagX
  let p_green_X := prob_green bagX
  let p_purple_Y := prob_purple bagY
  let p_purple_Z := prob_purple bagZ
  p_red_X * p_purple_Y + p_green_X * p_purple_Z = 31 / 80 := by
  sorry

end purple_marble_probability_l1575_157524


namespace perpendicular_line_through_point_l1575_157588

/-- Given a line L1 with equation x - 3y + 2 = 0 and a point P(1, 2),
    prove that the line L2 with equation 3x + y - 5 = 0 passes through P
    and is perpendicular to L1. -/
theorem perpendicular_line_through_point (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => x - 3*y + 2 = 0
  let L2 : ℝ → ℝ → Prop := λ x y => 3*x + y - 5 = 0
  let P : ℝ × ℝ := (1, 2)
  (L2 P.1 P.2) ∧                        -- L2 passes through P
  (∀ x1 y1 x2 y2 : ℝ,                   -- L2 is perpendicular to L1
    L1 x1 y1 → L1 x2 y2 → L2 x1 y1 → L2 x2 y2 →
    (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) ≠ 0 →
    ((x2 - x1) * 1 + (y2 - y1) * (-3)) * ((x2 - x1) * 3 + (y2 - y1) * 1) = 0) :=
by
  sorry


end perpendicular_line_through_point_l1575_157588


namespace milk_cartons_accepted_l1575_157508

/-- Proves that given 400 total cartons equally distributed among 4 customers,
    with each customer returning 60 damaged cartons, the total number of
    cartons accepted by all customers is 160. -/
theorem milk_cartons_accepted (total_cartons : ℕ) (num_customers : ℕ) (damaged_per_customer : ℕ)
    (h1 : total_cartons = 400)
    (h2 : num_customers = 4)
    (h3 : damaged_per_customer = 60) :
    (total_cartons / num_customers - damaged_per_customer) * num_customers = 160 :=
  by sorry

end milk_cartons_accepted_l1575_157508


namespace and_sufficient_not_necessary_for_or_l1575_157526

theorem and_sufficient_not_necessary_for_or :
  (∀ p q : Prop, p ∧ q → p ∨ q) ∧
  (∃ p q : Prop, p ∨ q ∧ ¬(p ∧ q)) :=
by sorry

end and_sufficient_not_necessary_for_or_l1575_157526


namespace bottle_caps_found_at_park_l1575_157541

/-- Represents Danny's collection of bottle caps and wrappers --/
structure Collection where
  bottleCaps : ℕ
  wrappers : ℕ

/-- Represents the items Danny found at the park --/
structure ParkFindings where
  bottleCaps : ℕ
  wrappers : ℕ

/-- Theorem stating the number of bottle caps Danny found at the park --/
theorem bottle_caps_found_at_park 
  (initialCollection : Collection)
  (parkFindings : ParkFindings)
  (finalCollection : Collection)
  (h1 : parkFindings.wrappers = 18)
  (h2 : finalCollection.wrappers = 67)
  (h3 : finalCollection.bottleCaps = 35)
  (h4 : finalCollection.wrappers = finalCollection.bottleCaps + 32)
  (h5 : finalCollection.bottleCaps = initialCollection.bottleCaps + parkFindings.bottleCaps)
  (h6 : finalCollection.wrappers = initialCollection.wrappers + parkFindings.wrappers) :
  parkFindings.bottleCaps = 18 := by
  sorry

end bottle_caps_found_at_park_l1575_157541


namespace parking_lot_cars_l1575_157578

theorem parking_lot_cars (total_wheels : ℕ) (wheels_per_car : ℕ) (h1 : total_wheels = 68) (h2 : wheels_per_car = 4) :
  total_wheels / wheels_per_car = 17 := by
  sorry

end parking_lot_cars_l1575_157578


namespace square_difference_l1575_157536

theorem square_difference : (169 * 169) - (168 * 168) = 337 := by
  sorry

end square_difference_l1575_157536


namespace evaluate_expression_l1575_157597

theorem evaluate_expression : 4 - (-3)^(-1/2 : ℂ) = 4 + (Complex.I * Real.sqrt 3) / 3 := by
  sorry

end evaluate_expression_l1575_157597


namespace tangent_line_coincidence_l1575_157595

/-- Given a differentiable function f where the tangent line of y = f(x) at (0,0) 
    coincides with the tangent line of y = f(x)/x at (2,1), prove that f'(2) = 2 -/
theorem tangent_line_coincidence (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x, x ≠ 0 → (f x) / x = ((f 0) + (deriv f 0) * x)) →
  (f 2) / 2 = 1 →
  deriv f 2 = 2 := by
sorry

end tangent_line_coincidence_l1575_157595
