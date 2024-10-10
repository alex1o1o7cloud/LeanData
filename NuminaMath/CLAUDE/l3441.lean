import Mathlib

namespace water_after_40_days_l3441_344146

/-- Calculates the remaining water in a trough after a given number of days -/
def remaining_water (initial_amount : ℝ) (evaporation_rate : ℝ) (days : ℝ) : ℝ :=
  initial_amount - evaporation_rate * days

/-- Theorem stating that given the initial conditions, the remaining water after 40 days is 270 gallons -/
theorem water_after_40_days :
  let initial_amount : ℝ := 300
  let evaporation_rate : ℝ := 0.75
  let days : ℝ := 40
  remaining_water initial_amount evaporation_rate days = 270 := by
sorry

#eval remaining_water 300 0.75 40

end water_after_40_days_l3441_344146


namespace first_term_of_geometric_series_l3441_344179

/-- The first term of an infinite geometric series with common ratio -1/3 and sum 24 is 32 -/
theorem first_term_of_geometric_series (a : ℝ) : 
  (∑' n, a * (-1/3)^n : ℝ) = 24 → a = 32 := by
  sorry

end first_term_of_geometric_series_l3441_344179


namespace right_triangle_pq_length_l3441_344100

/-- Represents a right triangle PQR -/
structure RightTriangle where
  PQ : ℝ
  PR : ℝ
  QR : ℝ
  tanQ : ℝ

/-- Theorem: In a right triangle PQR where ∠R = 90°, tan Q = 3/4, and PR = 12, PQ = 9 -/
theorem right_triangle_pq_length 
  (t : RightTriangle) 
  (h1 : t.tanQ = 3 / 4) 
  (h2 : t.PR = 12) : 
  t.PQ = 9 := by
  sorry

end right_triangle_pq_length_l3441_344100


namespace sum_of_distance_and_reciprocal_l3441_344135

theorem sum_of_distance_and_reciprocal (a b : ℝ) : 
  (|a| = 5 ∧ b = (-1/3)⁻¹) → (a + b = 2 ∨ a + b = -8) :=
by sorry

end sum_of_distance_and_reciprocal_l3441_344135


namespace points_symmetric_wrt_origin_l3441_344171

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = -B.2

/-- Point A has coordinates (3, 4) -/
def A : ℝ × ℝ := (3, 4)

/-- Point B has coordinates (-3, -4) -/
def B : ℝ × ℝ := (-3, -4)

theorem points_symmetric_wrt_origin : symmetric_wrt_origin A B := by
  sorry

end points_symmetric_wrt_origin_l3441_344171


namespace vanessa_scoring_record_l3441_344101

/-- Vanessa's new scoring record in a basketball game -/
theorem vanessa_scoring_record (total_score : ℕ) (other_players : ℕ) (average_score : ℕ) 
  (h1 : total_score = 55)
  (h2 : other_players = 7)
  (h3 : average_score = 4) : 
  total_score - (other_players * average_score) = 27 := by
  sorry

end vanessa_scoring_record_l3441_344101


namespace impossible_configuration_l3441_344159

theorem impossible_configuration : ¬ ∃ (arrangement : List ℕ) (sum : ℕ),
  (arrangement.toFinset = {1, 4, 9, 16, 25, 36, 49}) ∧
  (∀ radial_line : List ℕ, radial_line.sum = sum) ∧
  (∀ triangle : List ℕ, triangle.sum = sum) :=
by sorry

end impossible_configuration_l3441_344159


namespace hyperbola_asymptotes_l3441_344175

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - 4*y^2 = -1

/-- The asymptote equations -/
def asymptotes (x y : ℝ) : Prop := x + 2*y = 0 ∨ x - 2*y = 0

/-- Theorem: The asymptotes of the given hyperbola are x ± 2y = 0 -/
theorem hyperbola_asymptotes : 
  ∀ x y : ℝ, hyperbola x y → asymptotes x y :=
sorry

end hyperbola_asymptotes_l3441_344175


namespace complex_exponentiation_165_deg_60_l3441_344169

theorem complex_exponentiation_165_deg_60 : 
  (Complex.exp (Complex.I * Real.pi * 165 / 180)) ^ 60 = -1 := by
  sorry

end complex_exponentiation_165_deg_60_l3441_344169


namespace time_per_question_l3441_344187

/-- Proves that given a test with 100 questions, where 40 questions are left unanswered
    and 2 hours are spent answering, the time taken for each answered question is 2 minutes. -/
theorem time_per_question (total_questions : Nat) (unanswered_questions : Nat) (time_spent : Nat) :
  total_questions = 100 →
  unanswered_questions = 40 →
  time_spent = 120 →
  (time_spent : ℚ) / ((total_questions - unanswered_questions) : ℚ) = 2 := by
  sorry

end time_per_question_l3441_344187


namespace nine_eat_both_veg_and_non_veg_l3441_344198

/-- Represents the number of people in different dietary categories in a family -/
structure FamilyDiet where
  only_veg : ℕ
  only_non_veg : ℕ
  total_veg : ℕ

/-- Calculates the number of people who eat both veg and non-veg -/
def both_veg_and_non_veg (f : FamilyDiet) : ℕ :=
  f.total_veg - f.only_veg

/-- Theorem stating that 9 people eat both veg and non-veg in the given family -/
theorem nine_eat_both_veg_and_non_veg (f : FamilyDiet)
    (h1 : f.only_veg = 11)
    (h2 : f.only_non_veg = 6)
    (h3 : f.total_veg = 20) :
    both_veg_and_non_veg f = 9 := by
  sorry

end nine_eat_both_veg_and_non_veg_l3441_344198


namespace white_balls_count_l3441_344189

theorem white_balls_count (total : ℕ) (green yellow red purple : ℕ) (prob_not_red_purple : ℚ) :
  total = 100 →
  green = 30 →
  yellow = 10 →
  red = 37 →
  purple = 3 →
  prob_not_red_purple = 3/5 →
  ∃ white : ℕ, white = 20 ∧ total = white + green + yellow + red + purple :=
by sorry

end white_balls_count_l3441_344189


namespace geometric_sequence_sum_l3441_344164

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 6 + 2 * a 4 * a 5 + a 5 ^ 2 = 25 →
  a 4 + a 5 = 5 := by
  sorry

end geometric_sequence_sum_l3441_344164


namespace set_equality_l3441_344178

def M : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def N : Set ℝ := {1, 2}

theorem set_equality : M = N := by
  sorry

end set_equality_l3441_344178


namespace concentric_circles_intersection_l3441_344155

theorem concentric_circles_intersection (r_outer r_inner : ℝ) (h_outer : r_outer * 2 * Real.pi = 24 * Real.pi) (h_inner : r_inner * 2 * Real.pi = 14 * Real.pi) : r_outer - r_inner = 5 := by
  sorry

end concentric_circles_intersection_l3441_344155


namespace complex_division_result_l3441_344176

theorem complex_division_result : (10 * Complex.I) / (1 - 2 * Complex.I) = -4 + 2 * Complex.I := by
  sorry

end complex_division_result_l3441_344176


namespace abs_four_minus_xy_gt_two_abs_x_minus_y_l3441_344165

theorem abs_four_minus_xy_gt_two_abs_x_minus_y 
  (x y : ℝ) (hx : |x| < 2) (hy : |y| < 2) : 
  |4 - x * y| > 2 * |x - y| := by
  sorry

end abs_four_minus_xy_gt_two_abs_x_minus_y_l3441_344165


namespace cosine_problem_l3441_344138

theorem cosine_problem (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.cos (α + β) = 12/13) (h4 : Real.cos (2*α + β) = 3/5) :
  Real.cos α = 56/65 := by
  sorry

end cosine_problem_l3441_344138


namespace pulley_centers_distance_l3441_344196

theorem pulley_centers_distance (r1 r2 contact_distance : ℝ) 
  (h1 : r1 = 12)
  (h2 : r2 = 6)
  (h3 : contact_distance = 30) :
  ∃ (center_distance : ℝ), center_distance = 2 * Real.sqrt 234 := by
sorry

end pulley_centers_distance_l3441_344196


namespace total_goats_l3441_344150

theorem total_goats (washington_goats : ℕ) (paddington_extra : ℕ) : 
  washington_goats = 180 → 
  paddington_extra = 70 → 
  washington_goats + (washington_goats + paddington_extra) = 430 := by
sorry

end total_goats_l3441_344150


namespace f_form_when_a_equals_b_f_max_value_with_three_zeros_l3441_344188

-- Define the function f(x) with parameters a and b
def f (a b x : ℝ) : ℝ := (x - a) * (x^2 - (b - 1) * x - b)

-- Theorem 1: When a = b = 1, f(x) = (x-1)^2(x+1)
theorem f_form_when_a_equals_b (x : ℝ) :
  f 1 1 x = (x - 1)^2 * (x + 1) := by sorry

-- Theorem 2: When f(x) = x(x-1)(x+1), the maximum value is 2√3/9
theorem f_max_value_with_three_zeros :
  let g (x : ℝ) := x * (x - 1) * (x + 1)
  ∃ (x_max : ℝ), g x_max = 2 * Real.sqrt 3 / 9 ∧ ∀ (x : ℝ), g x ≤ g x_max := by sorry

end f_form_when_a_equals_b_f_max_value_with_three_zeros_l3441_344188


namespace camping_trip_percentage_l3441_344113

theorem camping_trip_percentage (total_students : ℕ) 
  (h1 : (16 : ℝ) / 100 * total_students = (25 : ℝ) / 100 * (64 : ℝ) / 100 * total_students)
  (h2 : (75 : ℝ) / 100 * (64 : ℝ) / 100 * total_students = (64 : ℝ) / 100 * total_students - (16 : ℝ) / 100 * total_students) :
  (64 : ℝ) / 100 * total_students = (64 : ℝ) / 100 * total_students :=
by sorry

end camping_trip_percentage_l3441_344113


namespace florist_roses_problem_l3441_344122

/-- A florist problem involving roses -/
theorem florist_roses_problem (initial_roses : ℕ) (picked_roses : ℕ) (final_roses : ℕ) :
  initial_roses = 50 →
  picked_roses = 21 →
  final_roses = 56 →
  ∃ (sold_roses : ℕ), initial_roses - sold_roses + picked_roses = final_roses ∧ sold_roses = 15 :=
by sorry

end florist_roses_problem_l3441_344122


namespace soap_boxes_in_carton_l3441_344148

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the theoretical maximum number of smaller boxes that can fit in a larger box -/
def maxBoxes (large : BoxDimensions) (small : BoxDimensions) : ℕ :=
  (boxVolume large) / (boxVolume small)

/-- Theorem: The maximum number of soap boxes that can theoretically fit in the carton is 150 -/
theorem soap_boxes_in_carton :
  let carton := BoxDimensions.mk 25 42 60
  let soapBox := BoxDimensions.mk 7 12 5
  maxBoxes carton soapBox = 150 := by
  sorry

end soap_boxes_in_carton_l3441_344148


namespace simplify_fraction_l3441_344152

theorem simplify_fraction : 21 * (8 / 15) * (1 / 14) = 4 / 5 := by
  sorry

end simplify_fraction_l3441_344152


namespace candy_solution_l3441_344106

/-- Represents the candy distribution problem --/
def candy_problem (billy_initial caleb_initial andy_initial : ℕ)
                  (new_candies billy_new caleb_new : ℕ) : Prop :=
  let billy_total := billy_initial + billy_new
  let caleb_total := caleb_initial + caleb_new
  let andy_new := new_candies - billy_new - caleb_new
  let andy_total := andy_initial + andy_new
  andy_total - caleb_total = 4

/-- Theorem stating the solution to the candy problem --/
theorem candy_solution :
  candy_problem 6 11 9 36 8 11 := by
  sorry

end candy_solution_l3441_344106


namespace smallest_volume_is_180_l3441_344153

/-- Represents the dimensions and cube counts of a rectangular box. -/
structure BoxDimensions where
  a : ℕ+  -- length
  b : ℕ+  -- width
  c : ℕ+  -- height
  red_in_bc : ℕ+  -- number of red cubes in each 1×b×c layer
  green_in_bc : ℕ+  -- number of green cubes in each 1×b×c layer
  green_in_ac : ℕ+  -- number of green cubes in each a×1×c layer
  yellow_in_ac : ℕ+  -- number of yellow cubes in each a×1×c layer

/-- Checks if the given box dimensions satisfy the problem conditions. -/
def valid_box_dimensions (box : BoxDimensions) : Prop :=
  box.red_in_bc = 9 ∧
  box.green_in_bc = 12 ∧
  box.green_in_ac = 20 ∧
  box.yellow_in_ac = 25

/-- Calculates the volume of the box. -/
def box_volume (box : BoxDimensions) : ℕ :=
  box.a * box.b * box.c

/-- The main theorem stating that the smallest possible volume is 180. -/
theorem smallest_volume_is_180 :
  ∀ box : BoxDimensions, valid_box_dimensions box → box_volume box ≥ 180 :=
by sorry

end smallest_volume_is_180_l3441_344153


namespace inscribed_rectangle_delta_l3441_344154

/-- Triangle with side lengths a, b, c -/
structure Triangle (a b c : ℝ) where
  side_a : a > 0
  side_b : b > 0
  side_c : c > 0

/-- Rectangle inscribed in a triangle -/
structure InscribedRectangle (T : Triangle a b c) where
  area : ℝ → ℝ  -- Area as a function of the rectangle's width

/-- The coefficient δ in the quadratic area formula of an inscribed rectangle -/
def delta (T : Triangle 15 39 36) (R : InscribedRectangle T) : ℚ :=
  60 / 169

theorem inscribed_rectangle_delta :
  ∀ (T : Triangle 15 39 36) (R : InscribedRectangle T),
  ∃ (γ : ℝ), ∀ (ω : ℝ), R.area ω = γ * ω - (delta T R : ℝ) * ω^2 := by
  sorry

end inscribed_rectangle_delta_l3441_344154


namespace polynomial_factorization_l3441_344151

theorem polynomial_factorization (b : ℝ) : 
  (8 * b^4 - 100 * b^3 + 18) - (3 * b^4 - 11 * b^3 + 18) = b^3 * (5 * b - 89) := by
  sorry

end polynomial_factorization_l3441_344151


namespace max_value_of_trig_function_l3441_344131

theorem max_value_of_trig_function :
  let f : ℝ → ℝ := λ x ↦ Real.cos (2 * x) + 6 * Real.cos (Real.pi / 2 - x)
  ∃ M : ℝ, M = 5 ∧ ∀ x : ℝ, f x ≤ M :=
by sorry

end max_value_of_trig_function_l3441_344131


namespace complex_number_location_l3441_344192

theorem complex_number_location (z : ℂ) (h : (1 - Complex.I) * z = 2 * Complex.I) :
  z.re < 0 ∧ z.im > 0 := by
  sorry

end complex_number_location_l3441_344192


namespace sequence_sum_l3441_344136

theorem sequence_sum (n : ℕ) (y : ℕ → ℕ) (h1 : y 1 = 2) 
  (h2 : ∀ k ∈ Finset.range (n - 1), y (k + 1) = y k + k + 1) : 
  Finset.sum (Finset.range n) (λ k => y (k + 1)) = 2 * n + (n - 1) * n * (n + 1) / 6 := by
  sorry

end sequence_sum_l3441_344136


namespace total_coughs_after_20_minutes_l3441_344168

/-- The number of coughs per minute for Georgia -/
def georgia_coughs_per_minute : ℕ := 5

/-- The number of coughs per minute for Robert -/
def robert_coughs_per_minute : ℕ := 2 * georgia_coughs_per_minute

/-- The time period in minutes -/
def time_period : ℕ := 20

/-- The total number of coughs by Georgia and Robert after the given time period -/
def total_coughs : ℕ := (georgia_coughs_per_minute * time_period) + (robert_coughs_per_minute * time_period)

theorem total_coughs_after_20_minutes : total_coughs = 300 := by
  sorry

end total_coughs_after_20_minutes_l3441_344168


namespace trigonometric_equation_solutions_l3441_344115

theorem trigonometric_equation_solutions (x : ℝ) : 
  (1 + Real.sin x + Real.cos (3 * x) = Real.cos x + Real.sin (2 * x) + Real.cos (2 * x)) ↔ 
  (∃ k : ℤ, x = k * Real.pi ∨ 
            x = (-1)^(k+1) * Real.pi / 6 + k * Real.pi ∨ 
            x = Real.pi / 3 + 2 * k * Real.pi ∨ 
            x = -Real.pi / 3 + 2 * k * Real.pi) :=
by sorry

end trigonometric_equation_solutions_l3441_344115


namespace unique_solution_absolute_value_equation_l3441_344133

theorem unique_solution_absolute_value_equation :
  ∃! x : ℝ, |x - 1| = |x - 2| + |x + 3| + 1 := by
  sorry

end unique_solution_absolute_value_equation_l3441_344133


namespace base_conversion_512_to_base_7_l3441_344139

theorem base_conversion_512_to_base_7 :
  (1 * 7^3 + 3 * 7^2 + 3 * 7^1 + 1 * 7^0) = 512 := by
  sorry

end base_conversion_512_to_base_7_l3441_344139


namespace expression_simplification_l3441_344162

theorem expression_simplification :
  (((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 9) / 4)) = 12.75 := by sorry

end expression_simplification_l3441_344162


namespace cars_produced_in_north_america_l3441_344102

theorem cars_produced_in_north_america :
  ∀ (total_cars europe_cars north_america_cars : ℕ),
    total_cars = 6755 →
    europe_cars = 2871 →
    total_cars = europe_cars + north_america_cars →
    north_america_cars = 3884 :=
by
  sorry

end cars_produced_in_north_america_l3441_344102


namespace pear_banana_weight_equality_l3441_344141

/-- Given that 10 pears weigh the same as 6 bananas, 
    prove that 50 pears weigh the same as 30 bananas. -/
theorem pear_banana_weight_equality :
  ∀ (pear_weight banana_weight : ℕ → ℝ),
  (∀ n : ℕ, pear_weight (10 * n) = banana_weight (6 * n)) →
  pear_weight 50 = banana_weight 30 :=
by
  sorry

end pear_banana_weight_equality_l3441_344141


namespace value_in_scientific_notation_l3441_344172

-- Define a billion
def billion : ℝ := 10^9

-- Define the value in question
def value : ℝ := 101.49 * billion

-- Theorem statement
theorem value_in_scientific_notation : value = 1.0149 * 10^10 := by
  sorry

end value_in_scientific_notation_l3441_344172


namespace inequality_proof_l3441_344157

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_sum : a + b + c = 1) : 
  (a * b + b * c + c * a ≤ 1 / 3) ∧ 
  (a^2 / b + b^2 / c + c^2 / a ≥ 1) := by
  sorry

end inequality_proof_l3441_344157


namespace geometric_sequence_ratio_sum_l3441_344170

/-- Given three nonconstant geometric sequences with different common ratios,
    if a certain condition holds, then the sum of their common ratios is 1 + 2√2 -/
theorem geometric_sequence_ratio_sum (k a₂ a₃ b₂ b₃ c₂ c₃ m n o : ℝ) 
  (hm : m ≠ 1) (hn : n ≠ 1) (ho : o ≠ 1)  -- nonconstant sequences
  (hm_ne_n : m ≠ n) (hm_ne_o : m ≠ o) (hn_ne_o : n ≠ o)  -- different ratios
  (ha₂ : a₂ = k * m) (ha₃ : a₃ = k * m^2)  -- first sequence
  (hb₂ : b₂ = k * n) (hb₃ : b₃ = k * n^2)  -- second sequence
  (hc₂ : c₂ = k * o) (hc₃ : c₃ = k * o^2)  -- third sequence
  (heq : a₃ - b₃ + c₃ = 2 * (a₂ - b₂ + c₂))  -- given condition
  : m + n + o = 1 + 2 * Real.sqrt 2 := by
  sorry

end geometric_sequence_ratio_sum_l3441_344170


namespace two_hundred_squared_minus_399_is_composite_l3441_344107

theorem two_hundred_squared_minus_399_is_composite : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 200^2 - 399 = a * b :=
by
  sorry

end two_hundred_squared_minus_399_is_composite_l3441_344107


namespace transformation_constructible_l3441_344147

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

-- Define the points
variable (A B C A' B' C' : V)

-- Define the transformation
variable (T : V → V)

-- Define the non-collinearity condition
def NonCollinear (A B C : V) : Prop :=
  ¬ ∃ (t : ℝ), C - A = t • (B - A)

-- Define the concept of constructible with straightedge and compass
def Constructible (P P' : V) (T : V → V) : Prop :=
  ∃ (construction : V → V),
    (∀ X Y : V, ∃ line : ℝ → V, line 0 = X ∧ line 1 = Y) ∧
    (∀ X Y Z : V, ∃ W : V, ‖X - W‖ = ‖Y - Z‖) ∧
    construction P = P'

-- State the theorem
theorem transformation_constructible
  (h_non_collinear : NonCollinear A B C)
  (h_transform : T A = A' ∧ T B = B' ∧ T C = C') :
  ∀ P : V, Constructible P (T P) T :=
by sorry

end transformation_constructible_l3441_344147


namespace candy_distribution_l3441_344160

theorem candy_distribution (n : ℕ) (h1 : n > 0) : 
  (100 % n = 1) ↔ n = 11 := by sorry

end candy_distribution_l3441_344160


namespace sarah_birthday_next_monday_l3441_344111

def is_leap_year (year : ℕ) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

def days_since_reference_date (year month day : ℕ) : ℕ :=
  sorry

def day_of_week (year month day : ℕ) : ℕ :=
  (days_since_reference_date year month day) % 7

theorem sarah_birthday_next_monday (start_year : ℕ) (start_day_of_week : ℕ) :
  start_year = 2017 →
  start_day_of_week = 5 →
  day_of_week 2025 6 16 = 1 →
  ∀ y : ℕ, start_year < y → y < 2025 → day_of_week y 6 16 ≠ 1 :=
sorry

end sarah_birthday_next_monday_l3441_344111


namespace basketball_shot_probability_l3441_344127

theorem basketball_shot_probability :
  let p1 : ℚ := 2/3  -- Probability of making the first shot
  let p2_success : ℚ := 2/3  -- Probability of making the second shot if the first shot was successful
  let p2_fail : ℚ := 1/3  -- Probability of making the second shot if the first shot failed
  let p3_success : ℚ := 2/3  -- Probability of making the third shot after making the second
  let p3_fail : ℚ := 1/3  -- Probability of making the third shot after missing the second
  
  (p1 * p2_success * p3_success) +  -- Case 1: Make all three shots
  (p1 * (1 - p2_success) * p3_fail) +  -- Case 2: Make first, miss second, make third
  ((1 - p1) * p2_fail * p3_success) +  -- Case 3: Miss first, make second and third
  ((1 - p1) * (1 - p2_fail) * p3_fail) = 14/27  -- Case 4: Miss first and second, make third
  := by sorry

end basketball_shot_probability_l3441_344127


namespace intersection_point_unique_l3441_344145

/-- The line equation -/
def line_equation (x y z : ℝ) : Prop :=
  (x - 1) / 6 = (y - 3) / 1 ∧ (y - 3) / 1 = (z + 5) / 3

/-- The plane equation -/
def plane_equation (x y z : ℝ) : Prop :=
  3 * x - 2 * y + 5 * z - 3 = 0

/-- The intersection point -/
def intersection_point : ℝ × ℝ × ℝ := (7, 4, -2)

/-- Theorem stating that the intersection_point is the unique point satisfying both equations -/
theorem intersection_point_unique :
  line_equation intersection_point.1 intersection_point.2.1 intersection_point.2.2 ∧
  plane_equation intersection_point.1 intersection_point.2.1 intersection_point.2.2 ∧
  ∀ x y z : ℝ, line_equation x y z ∧ plane_equation x y z → (x, y, z) = intersection_point :=
by sorry


end intersection_point_unique_l3441_344145


namespace fraction_equals_zero_l3441_344185

theorem fraction_equals_zero (x : ℝ) :
  x = 3 → (2 * x - 6) / (5 * x + 10) = 0 :=
by
  sorry

end fraction_equals_zero_l3441_344185


namespace largest_common_divisor_of_consecutive_odds_l3441_344117

theorem largest_common_divisor_of_consecutive_odds (n : ℕ) (h : Even n) (h_pos : 0 < n) :
  ∃ (k : ℕ), k = 315 ∧ 
  (∀ (d : ℕ), d ∣ ((n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) → d ≤ k) ∧
  k ∣ ((n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) :=
sorry

end largest_common_divisor_of_consecutive_odds_l3441_344117


namespace original_number_l3441_344120

theorem original_number (x : ℝ) : 3 * (2 * x + 5) = 135 → x = 20 := by
  sorry

end original_number_l3441_344120


namespace count_five_digit_numbers_without_five_is_52488_l3441_344132

/-- The count of five-digit numbers not containing the digit 5 -/
def count_five_digit_numbers_without_five : ℕ :=
  8 * 9^4

/-- Theorem stating that the count of five-digit numbers not containing the digit 5 is 52488 -/
theorem count_five_digit_numbers_without_five_is_52488 :
  count_five_digit_numbers_without_five = 52488 := by
  sorry

end count_five_digit_numbers_without_five_is_52488_l3441_344132


namespace solve_for_m_l3441_344156

theorem solve_for_m (x y m : ℝ) : 
  x = 2 → 
  y = m → 
  3 * x + 2 * y = 10 → 
  m = 2 := by
sorry

end solve_for_m_l3441_344156


namespace shift_down_quadratic_l3441_344130

-- Define the original quadratic function
def original_function (x : ℝ) : ℝ := x^2

-- Define the transformation (shift down by 2 units)
def shift_down (y : ℝ) : ℝ := y - 2

-- Define the resulting function after transformation
def resulting_function (x : ℝ) : ℝ := x^2 - 2

-- Theorem stating that shifting the original function down by 2 units
-- results in the resulting function
theorem shift_down_quadratic :
  ∀ x : ℝ, shift_down (original_function x) = resulting_function x :=
by
  sorry


end shift_down_quadratic_l3441_344130


namespace eventually_single_digit_or_zero_l3441_344125

/-- Function to calculate the product of digits of a natural number -/
def digitProduct (n : ℕ) : ℕ :=
  if n = 0 then 0 else
  let digits := Nat.digits 10 n
  digits.foldl (·*·) 1

/-- Predicate to check if a number is single-digit or zero -/
def isSingleDigitOrZero (n : ℕ) : Prop :=
  n < 10

/-- Theorem stating that repeatedly applying digitProduct will eventually
    result in a single-digit number or zero -/
theorem eventually_single_digit_or_zero (n : ℕ) :
  ∃ k : ℕ, isSingleDigitOrZero ((digitProduct^[k]) n) :=
sorry


end eventually_single_digit_or_zero_l3441_344125


namespace random_events_count_l3441_344119

/-- Represents an event --/
inductive Event
| ClassPresident
| StrongerTeamWins
| BirthdayProblem
| SetInclusion
| PainterDeath
| JulySnow
| EvenSum
| RedLights

/-- Determines if an event is random --/
def isRandomEvent : Event → Bool
| Event.ClassPresident => true
| Event.StrongerTeamWins => true
| Event.BirthdayProblem => true
| Event.SetInclusion => false
| Event.PainterDeath => false
| Event.JulySnow => true
| Event.EvenSum => false
| Event.RedLights => true

/-- List of all events --/
def allEvents : List Event := [
  Event.ClassPresident,
  Event.StrongerTeamWins,
  Event.BirthdayProblem,
  Event.SetInclusion,
  Event.PainterDeath,
  Event.JulySnow,
  Event.EvenSum,
  Event.RedLights
]

/-- Theorem: The number of random events in the list is 5 --/
theorem random_events_count :
  (allEvents.filter isRandomEvent).length = 5 := by sorry

end random_events_count_l3441_344119


namespace cyclic_sum_inequality_l3441_344128

theorem cyclic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z ≥ 1) :
  (x^5 - x^2) / (x^5 + y^2 + z^2) + (y^5 - y^2) / (y^5 + z^2 + x^2) + (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 :=
sorry

end cyclic_sum_inequality_l3441_344128


namespace pet_store_dogs_l3441_344190

/-- Calculates the number of dogs in a pet store after a series of events --/
def final_dog_count (initial : ℕ) 
  (sunday_received sunday_sold : ℕ)
  (monday_received monday_returned : ℕ)
  (tuesday_received tuesday_sold : ℕ) : ℕ :=
  initial + sunday_received - sunday_sold + 
  monday_received + monday_returned +
  tuesday_received - tuesday_sold

/-- Theorem stating the final number of dogs in the pet store --/
theorem pet_store_dogs : 
  final_dog_count 2 5 2 3 1 4 3 = 10 := by
  sorry

end pet_store_dogs_l3441_344190


namespace x_squared_in_set_l3441_344112

theorem x_squared_in_set (x : ℝ) : x^2 ∈ ({1, 0, x} : Set ℝ) → x = -1 := by
  sorry

end x_squared_in_set_l3441_344112


namespace root_cubic_value_l3441_344140

theorem root_cubic_value (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2*m^2 + 2014 = 2015 := by
  sorry

end root_cubic_value_l3441_344140


namespace percentage_relation_l3441_344149

theorem percentage_relation (A B C x y : ℝ) 
  (h1 : A > 0) (h2 : B > 0) (h3 : C > 0)
  (h4 : A > B) (h5 : B > C)
  (h6 : A = B * (1 + x / 100))
  (h7 : B = C * (1 + y / 100)) : 
  x = 100 * (A / (C * (1 + y / 100)) - 1) := by
  sorry

end percentage_relation_l3441_344149


namespace triangle_sum_theorem_l3441_344183

def is_valid_triangle (t : Finset Nat) : Prop :=
  t.card = 3 ∧ ∀ x ∈ t, 1 ≤ x ∧ x ≤ 9

def sum_of_triangle (t : Finset Nat) : Nat :=
  t.sum id

def valid_sum (s : Nat) : Prop :=
  12 ≤ s ∧ s ≤ 27 ∧ s ≠ 14 ∧ s ≠ 25

theorem triangle_sum_theorem :
  {s : Nat | ∃ t1 t2 : Finset Nat,
    is_valid_triangle t1 ∧
    is_valid_triangle t2 ∧
    t1 ∩ t2 = ∅ ∧
    sum_of_triangle t1 = s ∧
    sum_of_triangle t2 = s ∧
    valid_sum s} =
  {12, 13, 15, 16, 17, 18, 19} :=
sorry

end triangle_sum_theorem_l3441_344183


namespace circle_area_equality_l3441_344193

theorem circle_area_equality (r₁ r₂ : ℝ) (h₁ : r₁ = 25) (h₂ : r₂ = 17) :
  ∃ r : ℝ, π * r^2 = π * r₁^2 - π * r₂^2 ∧ r = 4 * Real.sqrt 21 := by
  sorry

end circle_area_equality_l3441_344193


namespace max_area_rectangle_l3441_344191

theorem max_area_rectangle (l w : ℕ) : 
  (2 * (l + w) = 120) →  -- perimeter condition
  (∀ a b : ℕ, 2 * (a + b) = 120 → l * w ≥ a * b) →  -- maximum area condition
  l * w = 900 := by
sorry

end max_area_rectangle_l3441_344191


namespace pgcd_and_divisibility_properties_l3441_344129

/-- Given a ≥ 2 and m ≥ n ≥ 1, prove three statements about PGCD and divisibility -/
theorem pgcd_and_divisibility_properties (a m n : ℕ) (ha : a ≥ 2) (hmn : m ≥ n) (hn : n ≥ 1) :
  (gcd (a^m - 1) (a^n - 1) = gcd (a^(m-n) - 1) (a^n - 1)) ∧
  (gcd (a^m - 1) (a^n - 1) = a^(gcd m n) - 1) ∧
  ((a^m - 1) ∣ (a^n - 1) ↔ m ∣ n) := by
  sorry


end pgcd_and_divisibility_properties_l3441_344129


namespace negation_of_universal_proposition_l3441_344110

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 3 > 0) ↔ (∃ x : ℝ, x^2 - x + 3 ≤ 0) := by
  sorry

end negation_of_universal_proposition_l3441_344110


namespace mystery_number_proof_l3441_344174

theorem mystery_number_proof : ∃ x : ℝ, x * 6 = 72 ∧ x = 12 := by
  sorry

end mystery_number_proof_l3441_344174


namespace guy_has_sixty_cents_l3441_344163

/-- The amount of money each person has in cents -/
structure Money where
  lance : ℕ
  margaret : ℕ
  bill : ℕ
  guy : ℕ

/-- The total amount of money in cents -/
def total (m : Money) : ℕ := m.lance + m.margaret + m.bill + m.guy

/-- Theorem: Given the conditions, Guy has 60 cents -/
theorem guy_has_sixty_cents (m : Money) 
  (h1 : m.lance = 70)
  (h2 : m.margaret = 75)  -- Three-fourths of a dollar is 75 cents
  (h3 : m.bill = 60)      -- Six dimes is 60 cents
  (h4 : total m = 265) : 
  m.guy = 60 := by
  sorry


end guy_has_sixty_cents_l3441_344163


namespace largest_two_digit_power_ending_l3441_344194

/-- A number is a two-digit number if it's between 10 and 99, inclusive. -/
def IsTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A number satisfies the power condition if all its positive integer powers end with itself modulo 100. -/
def SatisfiesPowerCondition (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 0 → n^k % 100 = n % 100

/-- 76 is the largest two-digit number divisible by 4 that satisfies the power condition. -/
theorem largest_two_digit_power_ending : 
  IsTwoDigit 76 ∧ 
  76 % 4 = 0 ∧ 
  SatisfiesPowerCondition 76 ∧ 
  ∀ n : ℕ, IsTwoDigit n → n % 4 = 0 → SatisfiesPowerCondition n → n ≤ 76 :=
sorry

end largest_two_digit_power_ending_l3441_344194


namespace complex_expression_equality_l3441_344142

theorem complex_expression_equality : 
  let a : ℂ := 3 - 2*I
  let b : ℂ := 2 + 3*I
  3*a + 4*b = 17 + 6*I := by sorry

end complex_expression_equality_l3441_344142


namespace parallelogram_cyclic_equidistant_implies_bisector_l3441_344186

-- Define the necessary structures and functions
structure Point := (x y : ℝ)

def Line := Point → Point → Prop

def parallelogram (A B C D : Point) : Prop := sorry

def cyclic_quadrilateral (B C E D : Point) : Prop := sorry

def intersects_interior (l : Line) (A B : Point) (F : Point) : Prop := sorry

def intersects (l : Line) (A B : Point) (G : Point) : Prop := sorry

def distance (P Q : Point) : ℝ := sorry

def angle_bisector (l : Line) (A B C : Point) : Prop := sorry

-- State the theorem
theorem parallelogram_cyclic_equidistant_implies_bisector
  (A B C D E F G : Point) (ℓ : Line) :
  parallelogram A B C D →
  cyclic_quadrilateral B C E D →
  ℓ A F →
  ℓ A G →
  intersects_interior ℓ D C F →
  intersects ℓ B C G →
  distance E F = distance E G →
  distance E F = distance E C →
  angle_bisector ℓ D A B :=
sorry

end parallelogram_cyclic_equidistant_implies_bisector_l3441_344186


namespace tree_height_problem_l3441_344105

theorem tree_height_problem (h₁ h₂ : ℝ) : 
  h₂ = h₁ + 24 →  -- The taller tree is 24 feet higher
  h₁ / h₂ = 5 / 7 →  -- The ratio of heights is 5:7
  h₂ = 84 := by
sorry

end tree_height_problem_l3441_344105


namespace divisible_by_nine_l3441_344137

theorem divisible_by_nine (n : ℕ) : ∃ k : ℤ, (4 : ℤ)^n + 15*n - 1 = 9*k := by
  sorry

end divisible_by_nine_l3441_344137


namespace ellipse_and_triangle_area_l3441_344144

-- Define the ellipse C
def ellipse_C (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1 ∧ a > b ∧ b > 0

-- Define the inscribed circle
def inscribed_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 2

-- Define the parabola E
def parabola_E (p : ℝ) (x y : ℝ) : Prop :=
  y^2 = 2 * p * x ∧ p > 0

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  y = x + m ∧ 0 ≤ m ∧ m ≤ 1

-- State the theorem
theorem ellipse_and_triangle_area :
  ∀ (a b c p m : ℝ) (x y : ℝ),
  ellipse_C a b x y →
  inscribed_circle x y →
  parabola_E p x y →
  line_l m x y →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), line_l m x₁ y₁ ∧ line_l m x₂ y₂ ∧ parabola_E p x₁ y₁ ∧ parabola_E p x₂ y₂) →
  (∃ (F : ℝ × ℝ), F.1 = c ∧ F.2 = 0 ∧ c^2 = a^2 - b^2) →
  (b = c) →
  (a^2 = 8 ∧ b^2 = 4) ∧
  (∃ (S : ℝ), S = (32 * Real.sqrt 6) / 9 ∧
    ∀ (S' : ℝ), S' ≤ S) :=
sorry

end ellipse_and_triangle_area_l3441_344144


namespace watch_cost_price_l3441_344158

theorem watch_cost_price (selling_price loss_percent gain_percent additional_amount : ℝ) 
  (h1 : selling_price = (1 - loss_percent / 100) * 1500)
  (h2 : selling_price + additional_amount = (1 + gain_percent / 100) * 1500)
  (h3 : loss_percent = 10)
  (h4 : gain_percent = 5)
  (h5 : additional_amount = 225) : 
  1500 = 1500 := by sorry

end watch_cost_price_l3441_344158


namespace equation_solution_l3441_344184

theorem equation_solution :
  ∃ n : ℚ, (22 + Real.sqrt (-4 + 18 * n) = 24) ∧ n = 4/9 := by
  sorry

end equation_solution_l3441_344184


namespace negation_of_forall_inequality_l3441_344177

theorem negation_of_forall_inequality :
  (¬ ∀ x : ℝ, x^2 - x > x + 1) ↔ (∃ x : ℝ, x^2 - x ≤ x + 1) := by
  sorry

end negation_of_forall_inequality_l3441_344177


namespace range_of_a_l3441_344197

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (a < 0) →
  (∀ x, ¬(p x a) → ¬(q x)) →
  (∃ x, ¬(p x a) ∧ (q x)) →
  -2/3 ≤ a ∧ a < 0 :=
by sorry

end range_of_a_l3441_344197


namespace min_value_of_function_l3441_344124

theorem min_value_of_function (x : ℝ) (h : x < 0) :
  -x - 2/x ≥ 2 * Real.sqrt 2 ∧
  (-(-Real.sqrt 2) - 2/(-Real.sqrt 2) = 2 * Real.sqrt 2) :=
by sorry

end min_value_of_function_l3441_344124


namespace vector_problem_l3441_344134

/-- Given two vectors a and b in ℝ², prove that:
    1) a = (-3, 4) and b = (5, -12)
    2) The dot product of a and b is -63
    3) The cosine of the angle between a and b is -63/65
-/
theorem vector_problem (a b : ℝ × ℝ) :
  (a.1 + b.1 = 2 ∧ a.2 + b.2 = -8) ∧  -- a + b = (2, -8)
  (a.1 - b.1 = -8 ∧ a.2 - b.2 = 16) → -- a - b = (-8, 16)
  (a = (-3, 4) ∧ b = (5, -12)) ∧      -- Part 1
  (a.1 * b.1 + a.2 * b.2 = -63) ∧     -- Part 2
  ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = -63/65) -- Part 3
  := by sorry

end vector_problem_l3441_344134


namespace pythagorean_triple_square_l3441_344103

theorem pythagorean_triple_square (a b c : ℕ+) (h : a^2 + b^2 = c^2) :
  ∃ m n : ℤ, (1/2 : ℚ) * ((c : ℚ) - (a : ℚ)) * ((c : ℚ) - (b : ℚ)) = (n^2 * (m - n)^2 : ℚ) := by
  sorry

end pythagorean_triple_square_l3441_344103


namespace inequality_solution_set_l3441_344182

theorem inequality_solution_set (x : ℝ) :
  (2 ≤ |x - 3| ∧ |x - 3| ≤ 8 ∧ x^2 - 6*x + 8 ≥ 0) ↔ 
  (x ∈ Set.Icc (-5) 1 ∪ Set.Icc 5 11) :=
by sorry

end inequality_solution_set_l3441_344182


namespace quadratic_function_m_value_l3441_344181

/-- A function y of x is quadratic if it can be written in the form y = ax² + bx + c, where a ≠ 0 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The main theorem stating that m = -1 is the only value satisfying the given conditions -/
theorem quadratic_function_m_value :
  ∃! m : ℝ, IsQuadratic (fun x ↦ (m - 1) * x^(m^2 + 1) + 3 * x) ∧ m - 1 ≠ 0 :=
by
  sorry


end quadratic_function_m_value_l3441_344181


namespace complex_expression_simplification_l3441_344173

theorem complex_expression_simplification :
  ∀ (i : ℂ), i^2 = -1 →
  7 * (4 - i) + 4 * i * (7 - i) + 2 * (3 + i) = 38 + 23 * i := by
sorry

end complex_expression_simplification_l3441_344173


namespace function_extrema_implies_a_range_l3441_344167

/-- If f(x) = x^3 + 2ax^2 + 3(a+2)x + 1 has both a maximum and a minimum value, then a > 2 or a < -1 -/
theorem function_extrema_implies_a_range (a : ℝ) : 
  (∃ (max min : ℝ), ∀ x, (x^3 + 2*a*x^2 + 3*(a+2)*x + 1 ≤ max ∧ x^3 + 2*a*x^2 + 3*(a+2)*x + 1 ≥ min)) →
  (a > 2 ∨ a < -1) := by
  sorry


end function_extrema_implies_a_range_l3441_344167


namespace equal_intercept_line_correct_l3441_344104

/-- A line passing through point (1, 2) with equal x and y intercepts -/
def equal_intercept_line (x y : ℝ) : Prop :=
  x + y - 3 = 0

theorem equal_intercept_line_correct :
  (equal_intercept_line 1 2) ∧
  (∃ (a : ℝ), a ≠ 0 ∧ equal_intercept_line a 0 ∧ equal_intercept_line 0 a) :=
by sorry

end equal_intercept_line_correct_l3441_344104


namespace cube_surface_area_increase_l3441_344121

theorem cube_surface_area_increase (L : ℝ) (L_new : ℝ) (h : L > 0) :
  L_new = 1.3 * L →
  (6 * L_new^2 - 6 * L^2) / (6 * L^2) = 0.69 := by
sorry

end cube_surface_area_increase_l3441_344121


namespace thirty_percent_of_hundred_l3441_344195

theorem thirty_percent_of_hundred : (30 : ℝ) / 100 * 100 = 30 := by
  sorry

end thirty_percent_of_hundred_l3441_344195


namespace men_left_bus_l3441_344118

/-- Represents the state of passengers on the bus --/
structure BusState where
  men : ℕ
  women : ℕ

/-- The initial state of the bus --/
def initialState : BusState :=
  { men := 48, women := 24 }

/-- The final state of the bus after some men leave and 8 women enter --/
def finalState : BusState :=
  { men := 32, women := 32 }

/-- The number of women who entered the bus in city Y --/
def womenEntered : ℕ := 8

theorem men_left_bus (initial : BusState) (final : BusState) :
  initial.men + initial.women = 72 →
  initial.women = initial.men / 2 →
  final.men = final.women →
  final.women = initial.women + womenEntered →
  initial.men - final.men = 16 := by
  sorry

#check men_left_bus initialState finalState

end men_left_bus_l3441_344118


namespace quadratic_distinct_roots_l3441_344180

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) := by
sorry

end quadratic_distinct_roots_l3441_344180


namespace cone_central_angle_l3441_344114

/-- Given a circular piece of paper with radius 18 cm, when partially cut to form a cone
    with radius 8 cm and volume 128π cm³, the central angle of the sector used to create
    the cone is approximately 53 degrees. -/
theorem cone_central_angle (paper_radius : ℝ) (cone_radius : ℝ) (cone_volume : ℝ) :
  paper_radius = 18 →
  cone_radius = 8 →
  cone_volume = 128 * Real.pi →
  ∃ (central_angle : ℝ), 52 < central_angle ∧ central_angle < 54 := by
  sorry

end cone_central_angle_l3441_344114


namespace two_car_efficiency_l3441_344109

/-- Two-car family fuel efficiency problem -/
theorem two_car_efficiency (mpg1 : ℝ) (total_miles : ℝ) (total_gallons : ℝ) (gallons1 : ℝ) :
  mpg1 = 25 →
  total_miles = 1825 →
  total_gallons = 55 →
  gallons1 = 30 →
  (total_miles - mpg1 * gallons1) / (total_gallons - gallons1) = 43 := by
sorry

end two_car_efficiency_l3441_344109


namespace repeating_decimal_division_l3441_344166

theorem repeating_decimal_division :
  let a : ℚ := 64 / 99
  let b : ℚ := 16 / 99
  a / b = 4 := by sorry

end repeating_decimal_division_l3441_344166


namespace triangle_max_area_l3441_344199

/-- Given a triangle ABC with circumradius 1 and tan(A) / tan(B) = (2c - b) / b, 
    the maximum area of the triangle is 3√3 / 4 -/
theorem triangle_max_area (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a = 2 * Real.sin A ∧
  b = 2 * Real.sin B ∧
  c = 2 * Real.sin C ∧
  Real.tan A / Real.tan B = (2 * c - b) / b →
  (∃ (S : ℝ), S = 1/2 * b * c * Real.sin A ∧ 
    ∀ (S' : ℝ), S' = 1/2 * b * c * Real.sin A → S' ≤ 3 * Real.sqrt 3 / 4) :=
by sorry


end triangle_max_area_l3441_344199


namespace complex_fraction_simplification_l3441_344143

theorem complex_fraction_simplification :
  (2 : ℂ) / (1 + Complex.I) = 1 - Complex.I := by sorry

end complex_fraction_simplification_l3441_344143


namespace anna_bills_count_l3441_344108

theorem anna_bills_count (five_dollar_bills : ℕ) (ten_dollar_bills : ℕ) : 
  five_dollar_bills = 4 → ten_dollar_bills = 8 → five_dollar_bills + ten_dollar_bills = 12 := by
  sorry

end anna_bills_count_l3441_344108


namespace quadratic_inequality_solution_set_l3441_344161

theorem quadratic_inequality_solution_set :
  {x : ℝ | 3 * x^2 - 5 * x - 8 > 0} = Set.Iio (-4/3) ∪ Set.Ioi 2 := by
  sorry

end quadratic_inequality_solution_set_l3441_344161


namespace instantaneous_velocity_at_2_l3441_344116

def motion_equation (t : ℝ) : ℝ := 2 * t^2 + 3

theorem instantaneous_velocity_at_2 :
  (deriv motion_equation) 2 = 8 := by sorry

end instantaneous_velocity_at_2_l3441_344116


namespace units_digit_of_special_two_digit_number_l3441_344123

/-- 
Given a two-digit number M = 10a + b, where a and b are single digits,
if M = ab + (a + b) + 5, then b = 8.
-/
theorem units_digit_of_special_two_digit_number (a b : ℕ) : 
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 →
  (10 * a + b = a * b + a + b + 5) →
  b = 8 := by
sorry

end units_digit_of_special_two_digit_number_l3441_344123


namespace mark_weekly_reading_pages_l3441_344126

-- Define the initial reading time in hours
def initial_reading_time : ℝ := 2

-- Define the percentage increase in reading time
def reading_time_increase : ℝ := 150

-- Define the initial pages read per day
def initial_pages_per_day : ℝ := 100

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Theorem to prove
theorem mark_weekly_reading_pages :
  let new_reading_time := initial_reading_time * (1 + reading_time_increase / 100)
  let new_pages_per_day := initial_pages_per_day * (new_reading_time / initial_reading_time)
  let weekly_pages := new_pages_per_day * days_in_week
  weekly_pages = 1750 := by sorry

end mark_weekly_reading_pages_l3441_344126
