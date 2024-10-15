import Mathlib

namespace NUMINAMATH_CALUDE_f_max_value_l3676_367665

open Real

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_derivative (x : ℝ) : deriv f x = (1 / x^2 - 2 * f x) / x

axiom f_initial_value : f 1 = 2

-- State the theorem
theorem f_max_value :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≤ f x ∧ f x = Real.exp 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l3676_367665


namespace NUMINAMATH_CALUDE_max_profit_on_day_6_l3676_367675

-- Define the sales price function
def p (x : ℕ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 6 then 44 + x
  else if 6 < x ∧ x ≤ 20 then 56 - x
  else 0

-- Define the sales volume function
def q (x : ℕ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 8 then 48 - x
  else if 8 < x ∧ x ≤ 20 then 32 + x
  else 0

-- Define the profit function
def profit (x : ℕ) : ℝ := (p x - 25) * q x

-- Theorem statement
theorem max_profit_on_day_6 :
  ∀ x : ℕ, 1 ≤ x ∧ x ≤ 20 → profit x ≤ profit 6 ∧ profit 6 = 1050 :=
sorry

end NUMINAMATH_CALUDE_max_profit_on_day_6_l3676_367675


namespace NUMINAMATH_CALUDE_cube_root_problem_l3676_367635

theorem cube_root_problem (a : ℝ) (h : a^3 = 21 * 25 * 15 * 147) : a = 105 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_problem_l3676_367635


namespace NUMINAMATH_CALUDE_isha_pencil_length_l3676_367617

/-- The length of a pencil after sharpening, given its original length and the length sharpened off. -/
def pencil_length_after_sharpening (original_length sharpened_off : ℕ) : ℕ :=
  original_length - sharpened_off

/-- Theorem stating that a 31-inch pencil sharpened by 17 inches results in a 14-inch pencil. -/
theorem isha_pencil_length :
  pencil_length_after_sharpening 31 17 = 14 := by
  sorry

end NUMINAMATH_CALUDE_isha_pencil_length_l3676_367617


namespace NUMINAMATH_CALUDE_number_of_boys_l3676_367629

theorem number_of_boys (total_children happy_children sad_children neutral_children girls happy_boys sad_girls : ℕ) :
  total_children = 60 →
  happy_children = 30 →
  sad_children = 10 →
  neutral_children = 20 →
  girls = 42 →
  happy_boys = 6 →
  sad_girls = 4 →
  total_children = happy_children + sad_children + neutral_children →
  ∃ boys, boys = total_children - girls ∧ boys = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_boys_l3676_367629


namespace NUMINAMATH_CALUDE_first_month_sale_l3676_367601

/-- Given the sales data for 6 months and the average sale, prove the sale amount for the first month -/
theorem first_month_sale
  (sales_2 sales_3 sales_4 sales_5 sales_6 : ℕ)
  (average_sale : ℕ)
  (h1 : sales_2 = 6500)
  (h2 : sales_3 = 9855)
  (h3 : sales_4 = 7230)
  (h4 : sales_5 = 7000)
  (h5 : sales_6 = 11915)
  (h6 : average_sale = 7500)
  : ∃ (sales_1 : ℕ), sales_1 = 2500 ∧ 
    (sales_1 + sales_2 + sales_3 + sales_4 + sales_5 + sales_6) / 6 = average_sale :=
by sorry

end NUMINAMATH_CALUDE_first_month_sale_l3676_367601


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_x_half_l3676_367622

/-- Two vectors are parallel if their cross product is zero -/
def IsParallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_imply_x_half :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (1, x)
  IsParallel (a.1 + b.1, a.2 + b.2) (a.1 - b.1, a.2 - b.2) →
  x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_x_half_l3676_367622


namespace NUMINAMATH_CALUDE_g_at_minus_one_l3676_367611

/-- The function g(x) = -2x^2 + 5x - 7 --/
def g (x : ℝ) : ℝ := -2 * x^2 + 5 * x - 7

/-- Theorem: g(-1) = -14 --/
theorem g_at_minus_one : g (-1) = -14 := by
  sorry

end NUMINAMATH_CALUDE_g_at_minus_one_l3676_367611


namespace NUMINAMATH_CALUDE_compute_expression_l3676_367616

theorem compute_expression : 16 * (125 / 2 + 25 / 4 + 9 / 16 + 1) = 1125 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3676_367616


namespace NUMINAMATH_CALUDE_employees_using_public_transportation_l3676_367699

theorem employees_using_public_transportation
  (total_employees : ℕ)
  (drive_percentage : ℚ)
  (public_transport_fraction : ℚ)
  (h1 : total_employees = 100)
  (h2 : drive_percentage = 60 / 100)
  (h3 : public_transport_fraction = 1 / 2) :
  (total_employees : ℚ) * (1 - drive_percentage) * public_transport_fraction = 20 := by
  sorry

end NUMINAMATH_CALUDE_employees_using_public_transportation_l3676_367699


namespace NUMINAMATH_CALUDE_daily_shoppers_l3676_367674

theorem daily_shoppers (tax_free_percentage : ℝ) (weekly_tax_payers : ℕ) : 
  tax_free_percentage = 0.06 →
  weekly_tax_payers = 6580 →
  ∃ (daily_shoppers : ℕ), daily_shoppers = 1000 ∧ 
    (1 - tax_free_percentage) * (daily_shoppers : ℝ) * 7 = weekly_tax_payers := by
  sorry

end NUMINAMATH_CALUDE_daily_shoppers_l3676_367674


namespace NUMINAMATH_CALUDE_similar_triangles_ratio_equality_l3676_367661

/-- Two triangles are similar if there exists a complex number k that maps one triangle to the other -/
def similar_triangles (a b c a' b' c' : ℂ) : Prop :=
  ∃ k : ℂ, k ≠ 0 ∧ b - a = k * (b' - a') ∧ c - a = k * (c' - a')

/-- Theorem: For similar triangles abc and a'b'c' on the complex plane, 
    the ratio (b-a)/(c-a) equals (b'-a')/(c'-a') -/
theorem similar_triangles_ratio_equality 
  (a b c a' b' c' : ℂ) 
  (h : similar_triangles a b c a' b' c') 
  (h1 : c ≠ a) 
  (h2 : c' ≠ a') : 
  (b - a) / (c - a) = (b' - a') / (c' - a') := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_ratio_equality_l3676_367661


namespace NUMINAMATH_CALUDE_greatest_cars_with_ac_no_stripes_l3676_367606

theorem greatest_cars_with_ac_no_stripes (total : Nat) (no_ac : Nat) (min_stripes : Nat)
  (h1 : total = 100)
  (h2 : no_ac = 37)
  (h3 : min_stripes = 51)
  (h4 : min_stripes ≤ total)
  (h5 : no_ac < total) :
  (total - no_ac) - min_stripes = 12 :=
sorry

end NUMINAMATH_CALUDE_greatest_cars_with_ac_no_stripes_l3676_367606


namespace NUMINAMATH_CALUDE_cubic_sum_inequality_l3676_367619

theorem cubic_sum_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : a^3 + b^3 = 2) :
  ((a + b) * (a^5 + b^5) ≥ 4) ∧ (a + b ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_inequality_l3676_367619


namespace NUMINAMATH_CALUDE_tribe_leadership_organization_l3676_367602

def tribe_size : ℕ := 12
def num_chiefs : ℕ := 1
def num_supporting_chiefs : ℕ := 3
def inferior_officers_per_chief : ℕ := 2

theorem tribe_leadership_organization :
  (tribe_size.choose num_chiefs) *
  ((tribe_size - num_chiefs).choose 1) *
  ((tribe_size - num_chiefs - 1).choose 1) *
  ((tribe_size - num_chiefs - 2).choose 1) *
  ((tribe_size - num_chiefs - num_supporting_chiefs).choose inferior_officers_per_chief) *
  ((tribe_size - num_chiefs - num_supporting_chiefs - inferior_officers_per_chief).choose inferior_officers_per_chief) *
  ((tribe_size - num_chiefs - num_supporting_chiefs - 2 * inferior_officers_per_chief).choose inferior_officers_per_chief) = 1069200 := by
  sorry

end NUMINAMATH_CALUDE_tribe_leadership_organization_l3676_367602


namespace NUMINAMATH_CALUDE_function_zero_in_interval_l3676_367621

theorem function_zero_in_interval (a : ℝ) : 
  (∃ x : ℝ, x ∈ (Set.Ioo 0 1) ∧ a^2 * x - 2*a + 1 = 0) ↔ 
  a ∈ (Set.Ioo (1/2) 1) ∪ (Set.Ioi 1) := by
sorry

end NUMINAMATH_CALUDE_function_zero_in_interval_l3676_367621


namespace NUMINAMATH_CALUDE_line_segment_sum_l3676_367681

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := y = -3/4 * x + 9

/-- Point P is on the x-axis -/
def P : ℝ × ℝ := (12, 0)

/-- Point Q is on the y-axis -/
def Q : ℝ × ℝ := (0, 9)

/-- Point T is on the line segment PQ -/
def T_on_PQ (r s : ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ (r, s) = (1 - t) • P + t • Q

/-- The area of triangle POQ is three times the area of triangle TOP -/
def area_condition (r s : ℝ) : Prop :=
  abs ((P.1 * Q.2 - Q.1 * P.2) / 2) = 3 * abs ((P.1 * s - r * P.2) / 2)

/-- The main theorem -/
theorem line_segment_sum (r s : ℝ) :
  line_equation r s → T_on_PQ r s → area_condition r s → r + s = 11 := by sorry

end NUMINAMATH_CALUDE_line_segment_sum_l3676_367681


namespace NUMINAMATH_CALUDE_max_value_tan_l3676_367647

/-- Given a function f(x) = 3sin(x) + 2cos(x), when f(x) reaches its maximum value, tan(x) = 3/2 -/
theorem max_value_tan (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 3 * Real.sin x + 2 * Real.cos x
  ∃ (x_max : ℝ), (∀ y, f y ≤ f x_max) → Real.tan x_max = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_tan_l3676_367647


namespace NUMINAMATH_CALUDE_role_assignment_count_l3676_367641

def number_of_role_assignments (men : ℕ) (women : ℕ) : ℕ :=
  let male_role_assignments := men
  let female_role_assignments := women
  let remaining_actors := men + women - 2
  let either_gender_role_assignments := Nat.choose remaining_actors 4 * Nat.factorial 4
  male_role_assignments * female_role_assignments * either_gender_role_assignments

theorem role_assignment_count :
  number_of_role_assignments 6 7 = 33120 :=
sorry

end NUMINAMATH_CALUDE_role_assignment_count_l3676_367641


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3676_367604

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → x * Real.sin x < 2^x - 1) ↔
  (∃ x : ℝ, x > 0 ∧ x * Real.sin x ≥ 2^x - 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3676_367604


namespace NUMINAMATH_CALUDE_shaded_squares_formula_l3676_367672

/-- Represents a row of squares in the pattern -/
structure Row :=
  (number : ℕ)  -- The row number
  (total : ℕ)   -- Total number of squares in the row
  (unshaded : ℕ) -- Number of unshaded squares
  (shaded : ℕ)   -- Number of shaded squares

/-- The properties of the sequence of rows -/
def ValidSequence (rows : ℕ → Row) : Prop :=
  (rows 1).total = 1 ∧ 
  (rows 1).unshaded = 1 ∧
  (rows 1).shaded = 0 ∧
  (∀ n : ℕ, n > 0 → (rows n).number = n) ∧
  (∀ n : ℕ, n > 1 → (rows n).total = (rows (n-1)).total + 2) ∧
  (∀ n : ℕ, n > 0 → (rows n).unshaded = (rows n).total - (rows n).shaded) ∧
  (∀ n : ℕ, n > 0 → (rows n).unshaded = n)

theorem shaded_squares_formula (rows : ℕ → Row) 
  (h : ValidSequence rows) (n : ℕ) (hn : n > 0) : 
  (rows n).shaded = n - 1 :=
sorry

end NUMINAMATH_CALUDE_shaded_squares_formula_l3676_367672


namespace NUMINAMATH_CALUDE_mary_current_books_l3676_367646

/-- Calculates the number of books Mary has checked out after a series of library transactions. -/
def marysBooks (initialBooks : ℕ) (firstReturn : ℕ) (firstCheckout : ℕ) (secondReturn : ℕ) (secondCheckout : ℕ) : ℕ :=
  (((initialBooks - firstReturn) + firstCheckout) - secondReturn) + secondCheckout

/-- Proves that Mary currently has 12 books checked out from the library. -/
theorem mary_current_books :
  marysBooks 5 3 5 2 7 = 12 := by
  sorry

end NUMINAMATH_CALUDE_mary_current_books_l3676_367646


namespace NUMINAMATH_CALUDE_problem_solution_l3676_367600

theorem problem_solution (a : ℝ) : 3 ∈ ({a, a^2 - 2*a} : Set ℝ) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3676_367600


namespace NUMINAMATH_CALUDE_f_max_min_l3676_367612

-- Define the function
def f (x : ℝ) : ℝ := |-(x)| - |x - 3|

-- State the theorem
theorem f_max_min :
  (∀ x : ℝ, f x ≤ 3) ∧
  (∃ x : ℝ, f x = 3) ∧
  (∀ x : ℝ, f x ≥ -3) ∧
  (∃ x : ℝ, f x = -3) :=
sorry

end NUMINAMATH_CALUDE_f_max_min_l3676_367612


namespace NUMINAMATH_CALUDE_x_power_minus_reciprocal_l3676_367623

theorem x_power_minus_reciprocal (θ : Real) (x : Real) (n : Nat) 
  (h1 : 0 < θ) (h2 : θ < π) (h3 : x - 1/x = 2 * Real.sin θ) (h4 : n > 0) : 
  x^n - 1/(x^n) = 2 * Real.sinh (n * θ) := by
  sorry

end NUMINAMATH_CALUDE_x_power_minus_reciprocal_l3676_367623


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l3676_367615

/-- If the cost price of 50 articles equals the selling price of 35 articles,
    then the gain percent is (3/7) * 100. -/
theorem gain_percent_calculation (C S : ℝ) (h : 50 * C = 35 * S) :
  (S - C) / C * 100 = (3 / 7) * 100 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l3676_367615


namespace NUMINAMATH_CALUDE_bug_return_probability_l3676_367692

/-- Probability of a bug returning to the starting vertex of a regular tetrahedron after n steps -/
def P (n : ℕ) : ℚ :=
  if n = 0 then 1
  else (1 - P (n - 1)) / 3

/-- The regular tetrahedron has edge length 1 and the bug starts at vertex A -/
theorem bug_return_probability :
  P 10 = 4921 / 59049 :=
sorry

end NUMINAMATH_CALUDE_bug_return_probability_l3676_367692


namespace NUMINAMATH_CALUDE_translate_sin_function_l3676_367654

/-- Translates the given trigonometric function and proves the result -/
theorem translate_sin_function :
  let f (x : ℝ) := Real.sin (2 * x + π / 6)
  let g (x : ℝ) := f (x + π / 6) + 1
  ∀ x, g x = 2 * (Real.cos x) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_translate_sin_function_l3676_367654


namespace NUMINAMATH_CALUDE_supplementary_angle_measure_l3676_367609

theorem supplementary_angle_measure (angle : ℝ) (supplementary : ℝ) (complementary : ℝ) : 
  angle = 45 →
  angle + supplementary = 180 →
  angle + complementary = 90 →
  supplementary = 3 * complementary →
  supplementary = 135 := by
sorry

end NUMINAMATH_CALUDE_supplementary_angle_measure_l3676_367609


namespace NUMINAMATH_CALUDE_stephanie_orange_spending_l3676_367650

def num_visits : Nat := 8
def oranges_per_visit : Nat := 2

def prices : List Float := [0.50, 0.60, 0.55, 0.65, 0.70, 0.55, 0.50, 0.60]

theorem stephanie_orange_spending :
  prices.length = num_visits →
  (prices.map (· * oranges_per_visit.toFloat)).sum = 9.30 := by
  sorry

end NUMINAMATH_CALUDE_stephanie_orange_spending_l3676_367650


namespace NUMINAMATH_CALUDE_parabola_intersection_l3676_367639

/-- Two parabolas with different vertices have equations y = px^2 and y = q(x-a)^2 + b, 
    where (0,0) is the vertex of the first parabola and (a,b) is the vertex of the second parabola. 
    Each vertex lies on the other parabola. -/
theorem parabola_intersection (p q a b : ℝ) (h1 : a ≠ 0) (h2 : b = p * a^2) (h3 : 0 = q * a^2 + b) : 
  p + q = 0 := by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l3676_367639


namespace NUMINAMATH_CALUDE_count_distinct_cube_colorings_l3676_367620

/-- The number of distinct colorings of a cube with six colors -/
def distinct_cube_colorings : ℕ := 30

/-- The number of faces on a cube -/
def cube_faces : ℕ := 6

/-- The number of rotational symmetries of a cube -/
def cube_rotations : ℕ := 24

/-- Theorem stating the number of distinct colorings of a cube -/
theorem count_distinct_cube_colorings :
  distinct_cube_colorings = (cube_faces * (cube_faces - 1) * (cube_faces - 2) / 2) := by
  sorry

#check count_distinct_cube_colorings

end NUMINAMATH_CALUDE_count_distinct_cube_colorings_l3676_367620


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l3676_367634

theorem tan_alpha_plus_pi_fourth (α : Real) (h1 : α ∈ Set.Ioo 0 Real.pi) (h2 : Real.cos α = -4/5) :
  Real.tan (α + Real.pi/4) = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l3676_367634


namespace NUMINAMATH_CALUDE_rachel_total_score_l3676_367618

/-- Rachel's video game scoring system -/
def video_game_score (points_per_treasure : ℕ) (treasures_level1 : ℕ) (treasures_level2 : ℕ) : ℕ :=
  points_per_treasure * (treasures_level1 + treasures_level2)

/-- Theorem: Rachel's total score is 63 points -/
theorem rachel_total_score :
  video_game_score 9 5 2 = 63 :=
by sorry

end NUMINAMATH_CALUDE_rachel_total_score_l3676_367618


namespace NUMINAMATH_CALUDE_harry_anna_pencil_ratio_l3676_367605

/-- Proves that the ratio of Harry's initial pencils to Anna's pencils is 2:1 --/
theorem harry_anna_pencil_ratio :
  ∀ (anna_pencils : ℕ) (harry_initial : ℕ) (harry_lost : ℕ) (harry_left : ℕ),
    anna_pencils = 50 →
    harry_initial = anna_pencils * harry_left / (anna_pencils - harry_lost) →
    harry_lost = 19 →
    harry_left = 81 →
    harry_initial / anna_pencils = 2 := by
  sorry

end NUMINAMATH_CALUDE_harry_anna_pencil_ratio_l3676_367605


namespace NUMINAMATH_CALUDE_zero_in_interval_l3676_367673

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3 - 8 + 2 * x

theorem zero_in_interval :
  ∃ c ∈ Set.Ioo 3 4, f c = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_zero_in_interval_l3676_367673


namespace NUMINAMATH_CALUDE_least_area_rectangle_l3676_367624

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- Theorem: The least possible area of a rectangle with perimeter 200 and area divisible by 10 is 900 -/
theorem least_area_rectangle :
  ∃ (r : Rectangle),
    perimeter r = 200 ∧
    area r % 10 = 0 ∧
    area r = 900 ∧
    ∀ (s : Rectangle),
      perimeter s = 200 →
      area s % 10 = 0 →
      area r ≤ area s :=
sorry

end NUMINAMATH_CALUDE_least_area_rectangle_l3676_367624


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3676_367677

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 2 → x^2 ≥ 4) ↔ (∃ x : ℝ, x ≥ 2 ∧ x^2 < 4) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3676_367677


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3676_367698

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {0, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3676_367698


namespace NUMINAMATH_CALUDE_students_walking_home_fraction_l3676_367632

theorem students_walking_home_fraction :
  let bus_fraction : ℚ := 1/3
  let auto_fraction : ℚ := 1/6
  let bike_fraction : ℚ := 1/15
  let total_fraction : ℚ := 1
  let other_transport_fraction : ℚ := bus_fraction + auto_fraction + bike_fraction
  let walking_fraction : ℚ := total_fraction - other_transport_fraction
  walking_fraction = 13/30 := by
sorry

end NUMINAMATH_CALUDE_students_walking_home_fraction_l3676_367632


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l3676_367614

-- Define the right triangle XYZ
def XYZ : Set (ℝ × ℝ) := sorry

-- Define the lengths of the sides
def XZ : ℝ := 15
def YZ : ℝ := 8

-- Define that Z is a right angle
def Z_is_right_angle : sorry := sorry

-- Define the inscribed circle
def inscribed_circle : Set (ℝ × ℝ) := sorry

-- State the theorem
theorem inscribed_circle_radius :
  ∃ (r : ℝ), r = 3 ∧ ∀ (p : ℝ × ℝ), p ∈ inscribed_circle → 
    ∃ (c : ℝ × ℝ), c ∈ XYZ ∧ dist p c = r :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l3676_367614


namespace NUMINAMATH_CALUDE_jessie_weight_loss_l3676_367688

/-- Jessie's weight loss journey -/
theorem jessie_weight_loss (current_weight weight_lost : ℕ) 
  (h1 : current_weight = 66)
  (h2 : weight_lost = 126) : 
  current_weight + weight_lost = 192 := by
  sorry

end NUMINAMATH_CALUDE_jessie_weight_loss_l3676_367688


namespace NUMINAMATH_CALUDE_cosine_difference_l3676_367683

theorem cosine_difference (α β : ℝ) 
  (h1 : α + β = π / 3)
  (h2 : Real.tan α + Real.tan β = 2) :
  Real.cos (α - β) = (Real.sqrt 3 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_difference_l3676_367683


namespace NUMINAMATH_CALUDE_work_completion_time_l3676_367668

/-- Given:
  * A can do a work in 20 days
  * A works for 10 days and then leaves
  * B can finish the remaining work in 15 days
Prove that B can do the entire work in 30 days -/
theorem work_completion_time (a_time b_remaining_time : ℕ) 
  (h1 : a_time = 20)
  (h2 : b_remaining_time = 15) :
  let a_work_rate : ℚ := 1 / a_time
  let a_work_done : ℚ := a_work_rate * 10
  let remaining_work : ℚ := 1 - a_work_done
  let b_rate : ℚ := remaining_work / b_remaining_time
  b_rate⁻¹ = 30 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l3676_367668


namespace NUMINAMATH_CALUDE_tax_diminished_percentage_l3676_367695

/-- Proves that a 12% increase in consumption and a 23.84% decrease in revenue
    implies a 32% decrease in tax rate. -/
theorem tax_diminished_percentage
  (original_tax : ℝ)
  (original_consumption : ℝ)
  (new_tax : ℝ)
  (new_consumption : ℝ)
  (original_revenue : ℝ)
  (new_revenue : ℝ)
  (h1 : original_tax > 0)
  (h2 : original_consumption > 0)
  (h3 : new_consumption = original_consumption * 1.12)
  (h4 : new_revenue = original_revenue * 0.7616)
  (h5 : original_revenue = original_tax * original_consumption)
  (h6 : new_revenue = new_tax * new_consumption) :
  new_tax = original_tax * 0.68 :=
sorry

end NUMINAMATH_CALUDE_tax_diminished_percentage_l3676_367695


namespace NUMINAMATH_CALUDE_power_of_product_l3676_367610

theorem power_of_product (a : ℝ) : (2 * a) ^ 3 = 8 * a ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l3676_367610


namespace NUMINAMATH_CALUDE_rectangle_shorter_side_length_l3676_367682

/-- Given a rectangle made from a rope of length 100cm with longer sides of 28cm,
    the length of each shorter side is 22cm. -/
theorem rectangle_shorter_side_length
  (total_length : ℝ)
  (longer_side : ℝ)
  (h1 : total_length = 100)
  (h2 : longer_side = 28)
  : (total_length - 2 * longer_side) / 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_shorter_side_length_l3676_367682


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l3676_367685

/-- The x-intercept of a line passing through two given points is -3/2 -/
theorem x_intercept_of_line (p1 p2 : ℝ × ℝ) : 
  p1 = (-1, 1) → p2 = (0, 3) → 
  ∃ (m b : ℝ), (∀ (x y : ℝ), y = m * x + b ↔ (x, y) ∈ ({p1, p2} : Set (ℝ × ℝ))) → 
  (0 = m * (-3/2) + b) := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l3676_367685


namespace NUMINAMATH_CALUDE_sets_equality_l3676_367645

def A : Set ℕ := {x | ∃ a : ℕ, x = a^2 + 1}
def B : Set ℕ := {y | ∃ b : ℕ, y = b^2 - 4*b + 5}

theorem sets_equality : A = B := by sorry

end NUMINAMATH_CALUDE_sets_equality_l3676_367645


namespace NUMINAMATH_CALUDE_photo_difference_l3676_367659

theorem photo_difference (initial_photos : ℕ) (final_photos : ℕ) : 
  initial_photos = 400 →
  final_photos = 920 →
  let first_day_photos := initial_photos / 2
  let total_new_photos := final_photos - initial_photos
  let second_day_photos := total_new_photos - first_day_photos
  second_day_photos - first_day_photos = 120 := by
sorry


end NUMINAMATH_CALUDE_photo_difference_l3676_367659


namespace NUMINAMATH_CALUDE_b_age_l3676_367680

-- Define variables for ages
variable (a b c : ℕ)

-- Define the conditions from the problem
axiom age_relation : a = b + 2
axiom b_twice_c : b = 2 * c
axiom total_age : a + b + c = 27

-- Theorem to prove
theorem b_age : b = 10 := by
  sorry

end NUMINAMATH_CALUDE_b_age_l3676_367680


namespace NUMINAMATH_CALUDE_eighth_odd_multiple_of_5_l3676_367636

/-- A function that generates the nth positive odd multiple of 5 -/
def nthOddMultipleOf5 (n : ℕ) : ℕ := 10 * n - 5

/-- The property of being both odd and a multiple of 5 -/
def isOddMultipleOf5 (k : ℕ) : Prop := k % 2 = 1 ∧ k % 5 = 0

theorem eighth_odd_multiple_of_5 :
  nthOddMultipleOf5 8 = 75 ∧ 
  isOddMultipleOf5 (nthOddMultipleOf5 8) ∧
  (∀ m < 8, ∃ k < nthOddMultipleOf5 8, k > 0 ∧ isOddMultipleOf5 k) :=
sorry

end NUMINAMATH_CALUDE_eighth_odd_multiple_of_5_l3676_367636


namespace NUMINAMATH_CALUDE_f_composition_of_five_l3676_367608

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 5 * x + 1

theorem f_composition_of_five : f (f (f (f (f 5)))) = 166 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_of_five_l3676_367608


namespace NUMINAMATH_CALUDE_custom_op_theorem_l3676_367625

def customOp (M N : Set ℕ) : Set ℕ := {x | x ∈ M ∨ x ∈ N ∧ x ∉ M ∩ N}

def M : Set ℕ := {0, 2, 4, 6, 8, 10}
def N : Set ℕ := {0, 3, 6, 9, 12, 15}

theorem custom_op_theorem :
  (customOp (customOp M N) M) = {2, 4, 8, 10, 3, 9, 12, 15} := by sorry

end NUMINAMATH_CALUDE_custom_op_theorem_l3676_367625


namespace NUMINAMATH_CALUDE_triangle_transformation_result_l3676_367626

/-- Represents a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Rotates a point 90 degrees clockwise around the origin -/
def rotate90Clockwise (p : Point) : Point :=
  ⟨p.y, -p.x⟩

/-- Reflects a point over the x-axis -/
def reflectOverX (p : Point) : Point :=
  ⟨p.x, -p.y⟩

/-- Translates a point vertically by a given amount -/
def translateVertical (p : Point) (dy : ℝ) : Point :=
  ⟨p.x, p.y + dy⟩

/-- Rotates a point 180 degrees around the origin -/
def rotate180 (p : Point) : Point :=
  ⟨-p.x, -p.y⟩

/-- Applies all transformations to a single point -/
def applyAllTransformations (p : Point) : Point :=
  rotate180 (translateVertical (reflectOverX (rotate90Clockwise p)) 3)

/-- The main theorem stating the result of the transformations -/
theorem triangle_transformation_result :
  let initial := Triangle.mk ⟨1, 2⟩ ⟨4, 2⟩ ⟨1, 5⟩
  let final := Triangle.mk (applyAllTransformations initial.A)
                           (applyAllTransformations initial.B)
                           (applyAllTransformations initial.C)
  final = Triangle.mk ⟨-2, -4⟩ ⟨-2, -7⟩ ⟨-5, -4⟩ := by
  sorry

end NUMINAMATH_CALUDE_triangle_transformation_result_l3676_367626


namespace NUMINAMATH_CALUDE_inverse_of_A_l3676_367660

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, 7; -1, -1]

theorem inverse_of_A :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![-1/3, -7/3; 1/3, 4/3]
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_l3676_367660


namespace NUMINAMATH_CALUDE_hypotenuse_square_of_right_triangle_from_polynomial_roots_l3676_367652

/-- Given complex numbers a, b, and c that are zeros of a polynomial P(z) = z³ + pz² + qz + r,
    if |a|² + |b|² + |c|² = 300 and they form a right triangle in the complex plane,
    then the square of the hypotenuse h² = 400. -/
theorem hypotenuse_square_of_right_triangle_from_polynomial_roots
  (a b c : ℂ) (p q r : ℂ) :
  (a^3 + p*a^2 + q*a + r = 0) →
  (b^3 + p*b^2 + q*b + r = 0) →
  (c^3 + p*c^2 + q*c + r = 0) →
  Complex.abs a ^ 2 + Complex.abs b ^ 2 + Complex.abs c ^ 2 = 300 →
  ∃ (h : ℝ), (Complex.abs (a - c))^2 + (Complex.abs (b - c))^2 = h^2 →
  h^2 = 400 :=
by sorry

end NUMINAMATH_CALUDE_hypotenuse_square_of_right_triangle_from_polynomial_roots_l3676_367652


namespace NUMINAMATH_CALUDE_lew_gumballs_correct_l3676_367628

/-- The number of gumballs Carolyn bought -/
def carolyn_gumballs : ℕ := 17

/-- The number of gumballs Lew bought -/
def lew_gumballs : ℕ := 21

/-- The minimum number of gumballs Carey could have bought -/
def carey_min_gumballs : ℕ := 19

/-- The maximum number of gumballs Carey could have bought -/
def carey_max_gumballs : ℕ := 37

/-- The difference between the maximum and minimum number of gumballs Carey could have bought -/
def carey_gumballs_diff : ℕ := 18

/-- The minimum average number of gumballs -/
def min_avg : ℕ := 19

/-- The maximum average number of gumballs -/
def max_avg : ℕ := 25

theorem lew_gumballs_correct :
  ∀ x : ℕ,
  carey_min_gumballs ≤ x ∧ x ≤ carey_max_gumballs →
  (carolyn_gumballs + lew_gumballs + x : ℚ) / 3 ≥ min_avg ∧
  (carolyn_gumballs + lew_gumballs + x : ℚ) / 3 ≤ max_avg ∧
  carey_max_gumballs - carey_min_gumballs = carey_gumballs_diff →
  lew_gumballs = 21 :=
by sorry

end NUMINAMATH_CALUDE_lew_gumballs_correct_l3676_367628


namespace NUMINAMATH_CALUDE_tom_reading_pages_l3676_367693

/-- Tom's initial reading speed in pages per hour -/
def initial_speed : ℕ := 12

/-- The factor by which Tom increases his reading speed -/
def speed_increase : ℕ := 3

/-- The number of hours Tom reads -/
def reading_time : ℕ := 2

/-- Theorem stating the number of pages Tom can read with increased speed -/
theorem tom_reading_pages : initial_speed * speed_increase * reading_time = 72 := by
  sorry

end NUMINAMATH_CALUDE_tom_reading_pages_l3676_367693


namespace NUMINAMATH_CALUDE_wizard_elixir_combinations_l3676_367658

/-- The number of enchanted herbs available to the wizard. -/
def num_herbs : ℕ := 4

/-- The number of mystical crystals available to the wizard. -/
def num_crystals : ℕ := 6

/-- The number of incompatible crystals. -/
def num_incompatible_crystals : ℕ := 2

/-- The number of herbs incompatible with the incompatible crystals. -/
def num_incompatible_herbs : ℕ := 3

/-- The number of valid combinations for the wizard's elixir. -/
def valid_combinations : ℕ := num_herbs * num_crystals - num_incompatible_crystals * num_incompatible_herbs

theorem wizard_elixir_combinations :
  valid_combinations = 18 := by
  sorry

end NUMINAMATH_CALUDE_wizard_elixir_combinations_l3676_367658


namespace NUMINAMATH_CALUDE_exists_perpendicular_plane_containing_line_l3676_367666

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Predicate to check if a line intersects a plane -/
def intersects (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- Predicate to check if a line is perpendicular to a plane -/
def perpendicular_line_plane (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- Predicate to check if a plane contains a line -/
def contains_line (β : Plane3D) (l : Line3D) : Prop :=
  sorry

/-- Predicate to check if two planes are perpendicular -/
def perpendicular_planes (α β : Plane3D) : Prop :=
  sorry

/-- Theorem: If a line intersects a plane but is not perpendicular to it,
    then there exists a plane containing the line that is perpendicular to the original plane -/
theorem exists_perpendicular_plane_containing_line
  (l : Line3D) (α : Plane3D)
  (h1 : intersects l α)
  (h2 : ¬perpendicular_line_plane l α) :
  ∃ β : Plane3D, contains_line β l ∧ perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_exists_perpendicular_plane_containing_line_l3676_367666


namespace NUMINAMATH_CALUDE_rectangle_combination_forms_square_l3676_367638

theorem rectangle_combination_forms_square (n : Nat) (h : n = 100) :
  ∃ (square : Set (ℕ × ℕ)), 
    (∀ (x y : ℕ), (x, y) ∈ square → x < n ∧ y < n) ∧ 
    (∀ (x y : ℕ), (x, y) ∈ square → (x + 1, y) ∈ square ∨ (x, y + 1) ∈ square) ∧
    (∃ (x y : ℕ), 
      (x, y) ∈ square ∧ 
      (x + 1, y) ∈ square ∧ 
      (x, y + 1) ∈ square ∧ 
      (x + 1, y + 1) ∈ square) :=
by
  sorry


end NUMINAMATH_CALUDE_rectangle_combination_forms_square_l3676_367638


namespace NUMINAMATH_CALUDE_coin_flip_probability_l3676_367642

theorem coin_flip_probability (n : ℕ) : 
  (1 + n : ℚ) / 2^n = 5/32 ↔ n = 6 :=
by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l3676_367642


namespace NUMINAMATH_CALUDE_truck_speed_through_tunnel_l3676_367686

/-- Calculates the speed of a truck passing through a tunnel -/
theorem truck_speed_through_tunnel 
  (truck_length : ℝ) 
  (tunnel_length : ℝ) 
  (exit_time : ℝ) 
  (feet_per_mile : ℝ) 
  (h1 : truck_length = 66) 
  (h2 : tunnel_length = 330) 
  (h3 : exit_time = 6) 
  (h4 : feet_per_mile = 5280) :
  (truck_length + tunnel_length) / exit_time * 3600 / feet_per_mile = 45 := by
sorry

end NUMINAMATH_CALUDE_truck_speed_through_tunnel_l3676_367686


namespace NUMINAMATH_CALUDE_rectangle_ratio_squared_l3676_367633

theorem rectangle_ratio_squared (a b : ℝ) (h : a > 0 ∧ b > 0 ∧ a ≤ b) : 
  (a / b + 1 / 2 = b / Real.sqrt (a^2 + b^2)) → (a / b)^2 = (3 - Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_squared_l3676_367633


namespace NUMINAMATH_CALUDE_square_to_acute_triangle_with_different_sides_l3676_367653

/-- A part of a square -/
structure SquarePart where
  -- Add necessary fields

/-- A triangle formed from parts of a square -/
structure TriangleFromSquare where
  parts : Finset SquarePart
  -- Add necessary fields for angles and sides

/-- Represents a square that can be cut into parts -/
structure CuttableSquare where
  side : ℝ
  -- Add other necessary fields

/-- Predicate to check if a triangle has acute angles -/
def has_acute_angles (t : TriangleFromSquare) : Prop :=
  sorry

/-- Predicate to check if a triangle has different sides -/
def has_different_sides (t : TriangleFromSquare) : Prop :=
  sorry

/-- Theorem stating that a square can be cut into 3 parts to form a specific triangle -/
theorem square_to_acute_triangle_with_different_sides :
  ∃ (s : CuttableSquare) (t : TriangleFromSquare),
    t.parts.card = 3 ∧
    has_acute_angles t ∧
    has_different_sides t :=
  sorry

end NUMINAMATH_CALUDE_square_to_acute_triangle_with_different_sides_l3676_367653


namespace NUMINAMATH_CALUDE_extra_discount_is_four_percent_l3676_367651

/-- Calculates the percentage of extra discount given initial price, first discount, and final price -/
def extra_discount_percentage (initial_price first_discount final_price : ℚ) : ℚ :=
  let price_after_first_discount := initial_price - first_discount
  let extra_discount_amount := price_after_first_discount - final_price
  (extra_discount_amount / price_after_first_discount) * 100

/-- Theorem stating that the extra discount percentage is 4% given the problem conditions -/
theorem extra_discount_is_four_percent :
  extra_discount_percentage 50 2.08 46 = 4 := by
  sorry

end NUMINAMATH_CALUDE_extra_discount_is_four_percent_l3676_367651


namespace NUMINAMATH_CALUDE_T_equality_l3676_367627

theorem T_equality (x : ℝ) : 
  (x - 2)^4 + 5*(x - 2)^3 + 10*(x - 2)^2 + 10*(x - 2) + 5 = (x - 1)^4 + 1 := by
sorry

end NUMINAMATH_CALUDE_T_equality_l3676_367627


namespace NUMINAMATH_CALUDE_homologous_pair_from_both_parents_l3676_367678

/-- Represents a parent (mother or father) -/
inductive Parent : Type
| mother : Parent
| father : Parent

/-- Represents a chromosome -/
structure Chromosome : Type :=
  (source : Parent)

/-- Represents a pair of homologous chromosomes -/
structure HomologousPair : Type :=
  (chromosome1 : Chromosome)
  (chromosome2 : Chromosome)

/-- Represents a diploid cell -/
structure DiploidCell : Type :=
  (chromosomePairs : List HomologousPair)

/-- Theorem: In a diploid organism, each pair of homologous chromosomes
    is contributed jointly by the two parents -/
theorem homologous_pair_from_both_parents (cell : DiploidCell) :
  ∀ pair ∈ cell.chromosomePairs,
    (pair.chromosome1.source = Parent.mother ∧ pair.chromosome2.source = Parent.father) ∨
    (pair.chromosome1.source = Parent.father ∧ pair.chromosome2.source = Parent.mother) :=
sorry

end NUMINAMATH_CALUDE_homologous_pair_from_both_parents_l3676_367678


namespace NUMINAMATH_CALUDE_greatest_fraction_l3676_367687

theorem greatest_fraction : 
  let f1 := 44444 / 55555
  let f2 := 5555 / 6666
  let f3 := 666 / 777
  let f4 := 77 / 88
  let f5 := 8 / 9
  (f5 > f1) ∧ (f5 > f2) ∧ (f5 > f3) ∧ (f5 > f4) := by
  sorry

end NUMINAMATH_CALUDE_greatest_fraction_l3676_367687


namespace NUMINAMATH_CALUDE_floor_sum_equality_l3676_367670

theorem floor_sum_equality (n : ℕ+) : 
  ∑' k : ℕ, ⌊(n + 2^k : ℝ) / 2^(k+1)⌋ = n := by sorry

end NUMINAMATH_CALUDE_floor_sum_equality_l3676_367670


namespace NUMINAMATH_CALUDE_lollipop_distribution_l3676_367667

/-- The number of kids in the group -/
def num_kids : ℕ := 42

/-- The initial number of lollipops available -/
def initial_lollipops : ℕ := 650

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The additional lollipops needed -/
def additional_lollipops : ℕ := sum_first_n num_kids - initial_lollipops

theorem lollipop_distribution :
  additional_lollipops = 253 ∧
  ∀ k, k ≤ num_kids → k ≤ sum_first_n num_kids ∧
  sum_first_n num_kids = initial_lollipops + additional_lollipops :=
sorry

end NUMINAMATH_CALUDE_lollipop_distribution_l3676_367667


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l3676_367649

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

theorem tenth_term_of_sequence (a₁ r : ℚ) (h₁ : a₁ = 12) (h₂ : r = 1/2) :
  geometric_sequence a₁ r 10 = 3/128 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l3676_367649


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3676_367643

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 62) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 18 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3676_367643


namespace NUMINAMATH_CALUDE_parabolas_intersection_l3676_367669

/-- First parabola equation -/
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

/-- Second parabola equation -/
def g (x : ℝ) : ℝ := 9 * x^2 + 6 * x + 2

/-- The set of intersection points of the two parabolas -/
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | f p.1 = g p.1 ∧ p.2 = f p.1}

theorem parabolas_intersection :
  intersection_points = {(0, 2), (-5/3, 17)} := by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l3676_367669


namespace NUMINAMATH_CALUDE_binary_1010101_equals_85_l3676_367640

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1010 101₍₂₎ -/
def binary_number : List Bool := [true, false, true, false, true, false, true]

/-- Theorem stating that 1010 101₍₂₎ is equal to 85 in decimal -/
theorem binary_1010101_equals_85 : binary_to_decimal binary_number = 85 := by
  sorry

end NUMINAMATH_CALUDE_binary_1010101_equals_85_l3676_367640


namespace NUMINAMATH_CALUDE_bipyramid_volume_l3676_367655

/-- A bipyramid with square bases -/
structure Bipyramid :=
  (side : ℝ)
  (apex_angle : ℝ)

/-- The volume of a bipyramid -/
noncomputable def volume (b : Bipyramid) : ℝ :=
  sorry

/-- Theorem stating the volume of a specific bipyramid -/
theorem bipyramid_volume (b : Bipyramid) (h1 : b.side = 2) (h2 : b.apex_angle = π / 3) :
  volume b = 16 * Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_bipyramid_volume_l3676_367655


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l3676_367684

theorem max_value_on_ellipse :
  ∀ x y : ℝ, x^2/4 + y^2 = 1 → 2*x + y ≤ Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l3676_367684


namespace NUMINAMATH_CALUDE_sum_f_negative_l3676_367671

def f (x : ℝ) : ℝ := x + x^3 + x^5

theorem sum_f_negative (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ + x₂ < 0) (h₂ : x₂ + x₃ < 0) (h₃ : x₃ + x₁ < 0) :
  f x₁ + f x₂ + f x₃ < 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_negative_l3676_367671


namespace NUMINAMATH_CALUDE_bargain_bin_books_l3676_367664

theorem bargain_bin_books (initial_books sold_books added_books remaining_books : ℕ) :
  initial_books - sold_books + added_books = remaining_books →
  sold_books = 33 →
  added_books = 2 →
  remaining_books = 10 →
  initial_books = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_bargain_bin_books_l3676_367664


namespace NUMINAMATH_CALUDE_system_solution_l3676_367657

theorem system_solution (x y m : ℚ) : 
  x + 3 * y = 7 ∧ 
  x - 3 * y + m * x + 3 = 0 ∧ 
  2 * x - 3 * y = 2 → 
  m = -2/3 := by sorry

end NUMINAMATH_CALUDE_system_solution_l3676_367657


namespace NUMINAMATH_CALUDE_square_perimeter_unchanged_l3676_367603

/-- The perimeter of a square with side length 5 remains unchanged after cutting out four small rectangles from its corners. -/
theorem square_perimeter_unchanged (side_length : ℝ) (h : side_length = 5) :
  let original_perimeter := 4 * side_length
  let modified_perimeter := original_perimeter
  modified_perimeter = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_unchanged_l3676_367603


namespace NUMINAMATH_CALUDE_wand_original_price_l3676_367696

theorem wand_original_price (price_paid : ℝ) (original_price : ℝ) 
  (h1 : price_paid = 8)
  (h2 : price_paid = original_price / 8) : 
  original_price = 64 := by
  sorry

end NUMINAMATH_CALUDE_wand_original_price_l3676_367696


namespace NUMINAMATH_CALUDE_hcf_of_numbers_l3676_367662

def number1 : ℕ := 210
def number2 : ℕ := 330
def lcm_value : ℕ := 2310

theorem hcf_of_numbers : Nat.gcd number1 number2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_hcf_of_numbers_l3676_367662


namespace NUMINAMATH_CALUDE_d_value_approx_l3676_367676

-- Define the equation
def equation (d : ℝ) : Prop :=
  4 * ((3.6 * 0.48 * 2.50) / (d * 0.09 * 0.5)) = 3200.0000000000005

-- Theorem statement
theorem d_value_approx :
  ∃ d : ℝ, equation d ∧ abs (d - 0.3) < 0.0000001 :=
sorry

end NUMINAMATH_CALUDE_d_value_approx_l3676_367676


namespace NUMINAMATH_CALUDE_trapezoid_area_l3676_367691

/-- Given a trapezoid with bases a and b, prove that its area is 150 -/
theorem trapezoid_area (a b : ℝ) : 
  ((a + b) / 2) * ((a - b) / 2) = 25 →
  ∃ h : ℝ, h = 3 * (a - b) →
  (1 / 2) * (a + b) * h = 150 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_l3676_367691


namespace NUMINAMATH_CALUDE_problem_solution_l3676_367690

theorem problem_solution (a : ℝ) (h : a^2 - 4*a + 3 = 0) :
  (9 - 3*a) / (2*a - 4) / (a + 2 - 5 / (a - 2)) = -3/8 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3676_367690


namespace NUMINAMATH_CALUDE_exists_divisible_by_two_not_four_l3676_367679

theorem exists_divisible_by_two_not_four : ∃ m : ℕ, (2 ∣ m) ∧ ¬(4 ∣ m) := by
  sorry

end NUMINAMATH_CALUDE_exists_divisible_by_two_not_four_l3676_367679


namespace NUMINAMATH_CALUDE_coupon1_best_at_229_95_l3676_367697

def coupon1_discount (price : ℝ) : ℝ := 0.15 * price

def coupon2_discount (price : ℝ) : ℝ := 30

def coupon3_discount (price : ℝ) : ℝ := 0.2 * (price - 150)

def price_list : List ℝ := [199.95, 229.95, 249.95, 289.95, 319.95]

theorem coupon1_best_at_229_95 :
  let p := 229.95
  (p ≥ 50) ∧
  (p ≥ 150) ∧
  (coupon1_discount p > coupon2_discount p) ∧
  (coupon1_discount p > coupon3_discount p) ∧
  (∀ q ∈ price_list, q < p → 
    coupon1_discount q ≤ coupon2_discount q ∨ 
    coupon1_discount q ≤ coupon3_discount q) :=
by sorry

end NUMINAMATH_CALUDE_coupon1_best_at_229_95_l3676_367697


namespace NUMINAMATH_CALUDE_five_twelve_thirteen_right_triangle_l3676_367631

/-- A triple of positive integers representing the sides of a triangle -/
structure TripleSides where
  a : ℕ+
  b : ℕ+
  c : ℕ+

/-- Checks if a triple of sides satisfies the Pythagorean theorem -/
def is_right_triangle (sides : TripleSides) : Prop :=
  (sides.a.val ^ 2 : ℕ) + (sides.b.val ^ 2 : ℕ) = (sides.c.val ^ 2 : ℕ)

/-- The triple (5, 12, 13) forms a right triangle -/
theorem five_twelve_thirteen_right_triangle :
  is_right_triangle ⟨5, 12, 13⟩ := by sorry

end NUMINAMATH_CALUDE_five_twelve_thirteen_right_triangle_l3676_367631


namespace NUMINAMATH_CALUDE_xyz_less_than_one_l3676_367644

theorem xyz_less_than_one (x y z : ℝ) 
  (h1 : 2 * x > y^2 + z^2)
  (h2 : 2 * y > x^2 + z^2)
  (h3 : 2 * z > y^2 + x^2) : 
  x * y * z < 1 := by
  sorry

end NUMINAMATH_CALUDE_xyz_less_than_one_l3676_367644


namespace NUMINAMATH_CALUDE_probability_theorem_l3676_367694

def club_sizes : List Nat := [6, 9, 11, 13]

def probability_select_officers (sizes : List Nat) : Rat :=
  let total_probability := sizes.map (fun n => 1 / Nat.choose n 3)
  (1 / sizes.length) * total_probability.sum

theorem probability_theorem :
  probability_select_officers club_sizes = 905 / 55440 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l3676_367694


namespace NUMINAMATH_CALUDE_quadratic_sum_l3676_367648

/-- Given a quadratic function g(x) = 2x^2 + Bx + C, 
    if g(1) = 3 and g(2) = 0, then 2 + B + C + 2C = 23 -/
theorem quadratic_sum (B C : ℝ) : 
  (2 * 1^2 + B * 1 + C = 3) → 
  (2 * 2^2 + B * 2 + C = 0) → 
  (2 + B + C + 2 * C = 23) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3676_367648


namespace NUMINAMATH_CALUDE_greatest_integer_solution_l3676_367689

theorem greatest_integer_solution : 
  ∃ (n : ℤ), (∀ (x : ℤ), 6*x^2 + 5*x - 8 < 3*x^2 - 4*x + 1 → x ≤ n) ∧ 
  (6*n^2 + 5*n - 8 < 3*n^2 - 4*n + 1) ∧ 
  n = 0 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_solution_l3676_367689


namespace NUMINAMATH_CALUDE_inverse_of_3_mod_179_l3676_367663

theorem inverse_of_3_mod_179 : ∃ x : ℕ, x < 179 ∧ (3 * x) % 179 = 1 :=
by
  use 60
  sorry

#eval (3 * 60) % 179  -- Should output 1

end NUMINAMATH_CALUDE_inverse_of_3_mod_179_l3676_367663


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3676_367613

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, |a * x + 2| < 6 ↔ -1 < x ∧ x < 2) → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3676_367613


namespace NUMINAMATH_CALUDE_perpendicular_tangents_a_value_l3676_367630

/-- The value of 'a' for which the curves y = ax³ - 6x² + 12x and y = exp(x)
    have perpendicular tangents at x = 1 -/
theorem perpendicular_tangents_a_value :
  ∀ a : ℝ,
  (∀ x : ℝ, deriv (fun x => a * x^3 - 6 * x^2 + 12 * x) 1 *
             deriv (fun x => Real.exp x) 1 = -1) →
  a = -1 / (3 * Real.exp 1) := by
sorry


end NUMINAMATH_CALUDE_perpendicular_tangents_a_value_l3676_367630


namespace NUMINAMATH_CALUDE_prime_square_minus_one_div_24_l3676_367607

theorem prime_square_minus_one_div_24 (p : Nat) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  24 ∣ (p^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_square_minus_one_div_24_l3676_367607


namespace NUMINAMATH_CALUDE_curve_is_circle_and_line_l3676_367637

/-- The polar equation of the curve -/
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ^2 * Real.cos θ - 3 * ρ * Real.cos θ + ρ - 3 = 0

/-- Definition of a circle in polar coordinates -/
def is_circle (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b r : ℝ, ∀ ρ θ : ℝ, f ρ θ ↔ (ρ * Real.cos θ - a)^2 + (ρ * Real.sin θ - b)^2 = r^2

/-- Definition of a line in polar coordinates -/
def is_line (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b : ℝ, ∀ ρ θ : ℝ, f ρ θ ↔ ρ * (a * Real.cos θ + b * Real.sin θ) = 1

/-- The theorem stating that the curve consists of a circle and a line -/
theorem curve_is_circle_and_line :
  (∃ f g : ℝ → ℝ → Prop, 
    (∀ ρ θ : ℝ, polar_equation ρ θ ↔ (f ρ θ ∨ g ρ θ)) ∧
    is_circle f ∧ is_line g) :=
sorry

end NUMINAMATH_CALUDE_curve_is_circle_and_line_l3676_367637


namespace NUMINAMATH_CALUDE_hyperbola_triangle_perimeter_l3676_367656

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define the foci
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- State the theorem
theorem hyperbola_triangle_perimeter :
  hyperbola A.1 A.2 →
  hyperbola B.1 B.2 →
  (A.1 < 0 ∧ B.1 < 0) →  -- A and B are on the left branch
  F₁ ∈ Set.Icc A B →     -- F₁ is on the line segment AB
  dist A B = 6 →
  dist A F₂ + dist B F₂ + dist A B = 28 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_triangle_perimeter_l3676_367656
