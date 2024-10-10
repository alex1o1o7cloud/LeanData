import Mathlib

namespace number_difference_l1230_123017

theorem number_difference (x y : ℝ) (h1 : x + y = 50) (h2 : 3 * y - 4 * x = 10) :
  |y - x| = 10 := by
  sorry

end number_difference_l1230_123017


namespace sugar_cube_weight_l1230_123062

theorem sugar_cube_weight
  (ants1 : ℕ) (cubes1 : ℕ) (weight1 : ℝ) (hours1 : ℝ)
  (ants2 : ℕ) (cubes2 : ℕ) (hours2 : ℝ)
  (h1 : ants1 = 15)
  (h2 : cubes1 = 600)
  (h3 : weight1 = 10)
  (h4 : hours1 = 5)
  (h5 : ants2 = 20)
  (h6 : cubes2 = 960)
  (h7 : hours2 = 3)
  : ∃ weight2 : ℝ,
    weight2 = 5 ∧
    (ants1 : ℝ) * (cubes1 : ℝ) * weight1 / hours1 =
    (ants2 : ℝ) * (cubes2 : ℝ) * weight2 / hours2 :=
by
  sorry

end sugar_cube_weight_l1230_123062


namespace pauls_books_l1230_123031

theorem pauls_books (books_sold : ℕ) (books_left : ℕ) : 
  books_sold = 137 → books_left = 105 → books_sold + books_left = 242 :=
by sorry

end pauls_books_l1230_123031


namespace find_number_l1230_123090

theorem find_number : ∃ x : ℕ, x * 9999 = 183868020 ∧ x = 18387 := by
  sorry

end find_number_l1230_123090


namespace complex_simplification_l1230_123030

theorem complex_simplification (i : ℂ) (h : i^2 = -1) :
  7 * (4 - 2*i) + 4*i * (7 - 2*i) = 36 + 14*i := by
  sorry

end complex_simplification_l1230_123030


namespace water_addition_changes_ratio_l1230_123052

/-- Given a mixture of alcohol and water, prove that adding 10 liters of water
    changes the ratio from 4:3 to 4:5 when the initial amount of alcohol is 20 liters. -/
theorem water_addition_changes_ratio :
  let initial_alcohol : ℝ := 20
  let initial_ratio : ℝ := 4 / 3
  let final_ratio : ℝ := 4 / 5
  let water_added : ℝ := 10
  let initial_water : ℝ := initial_alcohol / initial_ratio
  let final_water : ℝ := initial_water + water_added
  initial_alcohol / initial_water = initial_ratio ∧
  initial_alcohol / final_water = final_ratio :=
by sorry

end water_addition_changes_ratio_l1230_123052


namespace intersection_is_ellipse_l1230_123056

-- Define the plane
def plane (z : ℝ) : Prop := z = 2

-- Define the ellipsoid
def ellipsoid (x y z : ℝ) : Prop := x^2/12 + y^2/4 + z^2/16 = 1

-- Define the intersection curve
def intersection_curve (x y : ℝ) : Prop := x^2/9 + y^2/3 = 1

-- Theorem statement
theorem intersection_is_ellipse :
  ∀ x y z : ℝ,
  plane z ∧ ellipsoid x y z →
  intersection_curve x y ∧
  ∃ a b : ℝ, a = 3 ∧ b = Real.sqrt 3 :=
sorry

end intersection_is_ellipse_l1230_123056


namespace inequality_and_minimum_value_l1230_123015

theorem inequality_and_minimum_value 
  (a b m n : ℝ) (x : ℝ) 
  (ha : a > 0) (hb : b > 0) (hm : m > 0) (hn : n > 0)
  (hx : 0 < x ∧ x < 1/2) : 
  (m^2 / a + n^2 / b ≥ (m + n)^2 / (a + b)) ∧
  (2 / x + 9 / (1 - 2*x) ≥ 25) ∧
  (∀ y, 0 < y ∧ y < 1/2 → 2 / y + 9 / (1 - 2*y) ≥ 2 / x + 9 / (1 - 2*x)) ∧
  (x = 1/5) := by
sorry

end inequality_and_minimum_value_l1230_123015


namespace arctanSum_implies_powerSum_l1230_123006

theorem arctanSum_implies_powerSum (x y z : ℝ) (n : ℕ) 
  (h1 : x + y + z = 1) 
  (h2 : Real.arctan x + Real.arctan y + Real.arctan z = π / 4) 
  (h3 : n > 0) : 
  x^(2*n+1) + y^(2*n+1) + z^(2*n+1) = 1 := by
sorry

end arctanSum_implies_powerSum_l1230_123006


namespace decreasing_functions_a_range_l1230_123003

/-- Given two functions f and g, prove that if they are both decreasing on [1,2],
    then the parameter a is in the interval (0,1]. -/
theorem decreasing_functions_a_range 
  (f g : ℝ → ℝ) 
  (hf : f = fun x ↦ -x^2 + 2*a*x) 
  (hg : g = fun x ↦ a / (x + 1)) 
  (hf_decreasing : ∀ x y, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 2 → x < y → f x > f y) 
  (hg_decreasing : ∀ x y, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 2 → x < y → g x > g y) 
  : a ∈ Set.Ioo 0 1 := by
  sorry

end decreasing_functions_a_range_l1230_123003


namespace discriminant_of_2x2_minus_5x_plus_6_l1230_123009

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem: The discriminant of 2x^2 - 5x + 6 is -23 -/
theorem discriminant_of_2x2_minus_5x_plus_6 :
  discriminant 2 (-5) 6 = -23 := by
  sorry

end discriminant_of_2x2_minus_5x_plus_6_l1230_123009


namespace exactly_one_positive_integer_satisfies_condition_l1230_123054

theorem exactly_one_positive_integer_satisfies_condition :
  ∃! (n : ℕ), n > 0 ∧ 20 - 5 * n > 12 :=
by sorry

end exactly_one_positive_integer_satisfies_condition_l1230_123054


namespace perpendicular_slope_l1230_123012

/-- The slope of a line perpendicular to a line passing through (2, 3) and (7, 8) is -1 -/
theorem perpendicular_slope : 
  let p1 : ℝ × ℝ := (2, 3)
  let p2 : ℝ × ℝ := (7, 8)
  let m : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
  (-1 : ℝ) * m = -1 :=
by sorry

end perpendicular_slope_l1230_123012


namespace train_speed_l1230_123027

def train_length : Real := 250.00000000000003
def crossing_time : Real := 15

theorem train_speed : 
  (train_length / 1000) / (crossing_time / 3600) = 60 := by sorry

end train_speed_l1230_123027


namespace abs_x_minus_one_necessary_not_sufficient_l1230_123096

theorem abs_x_minus_one_necessary_not_sufficient :
  (∃ x : ℝ, |x - 1| < 2 ∧ ¬(x * (x - 3) < 0)) ∧
  (∀ x : ℝ, x * (x - 3) < 0 → |x - 1| < 2) :=
by sorry

end abs_x_minus_one_necessary_not_sufficient_l1230_123096


namespace base_number_proof_l1230_123067

theorem base_number_proof (w : ℕ) (x : ℝ) (h1 : w = 12) (h2 : 2^(2*w) = x^(w-4)) : x = 8 := by
  sorry

end base_number_proof_l1230_123067


namespace train_travel_time_l1230_123082

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  hlt24 : hours < 24
  mlt60 : minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDifference (t1 t2 : Time) : Nat :=
  let totalMinutes1 := t1.hours * 60 + t1.minutes
  let totalMinutes2 := t2.hours * 60 + t2.minutes
  totalMinutes2 - totalMinutes1

/-- The train travel time theorem -/
theorem train_travel_time :
  let departureTime : Time := ⟨7, 5, by norm_num, by norm_num⟩
  let arrivalTime : Time := ⟨7, 59, by norm_num, by norm_num⟩
  timeDifference departureTime arrivalTime = 54 := by
  sorry

end train_travel_time_l1230_123082


namespace exists_geometric_subsequence_l1230_123040

/-- A strictly increasing sequence of positive integers in arithmetic progression -/
def ArithmeticSequence : ℕ → ℕ := λ n => sorry

/-- The first term of the arithmetic progression -/
def a : ℕ := sorry

/-- The common difference of the arithmetic progression -/
def d : ℕ := sorry

/-- Condition: ArithmeticSequence is strictly increasing -/
axiom strictly_increasing : ∀ n : ℕ, ArithmeticSequence n < ArithmeticSequence (n + 1)

/-- Condition: ArithmeticSequence is an arithmetic progression -/
axiom is_arithmetic_progression : ∀ n : ℕ, ArithmeticSequence n = a + (n - 1) * d

/-- The existence of an infinite geometric sub-sequence -/
theorem exists_geometric_subsequence :
  ∃ (SubSeq : ℕ → ℕ) (r : ℚ),
    (∀ n : ℕ, ∃ k : ℕ, ArithmeticSequence k = SubSeq n) ∧
    (∀ n : ℕ, SubSeq (n + 1) = r * SubSeq n) :=
sorry

end exists_geometric_subsequence_l1230_123040


namespace no_solution_to_inequality_l1230_123013

theorem no_solution_to_inequality : ¬ ∃ x : ℝ, |x - 3| + |x + 4| < 6 := by
  sorry

end no_solution_to_inequality_l1230_123013


namespace food_weight_l1230_123020

/-- Given a bowl with 14 pieces of food, prove that each piece weighs 0.76 kg -/
theorem food_weight (total_weight : ℝ) (empty_bowl_weight : ℝ) (num_pieces : ℕ) :
  total_weight = 11.14 ∧ 
  empty_bowl_weight = 0.5 ∧ 
  num_pieces = 14 →
  (total_weight - empty_bowl_weight) / num_pieces = 0.76 := by
  sorry

end food_weight_l1230_123020


namespace six_people_arrangement_l1230_123060

/-- The number of arrangements of six people in a row,
    where A and B must be adjacent with B to the left of A -/
def arrangements_count : ℕ := 120

/-- Theorem stating that the number of arrangements is 120 -/
theorem six_people_arrangement :
  arrangements_count = 120 := by
  sorry

end six_people_arrangement_l1230_123060


namespace rectangle_area_l1230_123033

theorem rectangle_area (perimeter : ℝ) (length_ratio width_ratio : ℕ) : 
  perimeter = 280 →
  length_ratio = 5 →
  width_ratio = 2 →
  ∃ (length width : ℝ),
    length / width = length_ratio / width_ratio ∧
    2 * (length + width) = perimeter ∧
    length * width = 4000 := by
  sorry

end rectangle_area_l1230_123033


namespace ellipse_tangent_existence_l1230_123097

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is outside the ellipse -/
def isOutside (e : Ellipse) (p : Point) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) > 1

/-- Represents a line passing through two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Checks if a line is tangent to the ellipse -/
def isTangent (e : Ellipse) (l : Line) : Prop :=
  ∃ θ : ℝ, l.p2.x = e.a * Real.cos θ ∧ l.p2.y = e.b * Real.sin θ ∧
    (l.p1.x * l.p2.x / e.a^2) + (l.p1.y * l.p2.y / e.b^2) = 1

/-- Main theorem: For any ellipse and point outside it, there exist two tangent lines -/
theorem ellipse_tangent_existence (e : Ellipse) (p : Point) (h : isOutside e p) :
  ∃ l1 l2 : Line, l1 ≠ l2 ∧ l1.p1 = p ∧ l2.p1 = p ∧ isTangent e l1 ∧ isTangent e l2 := by
  sorry

end ellipse_tangent_existence_l1230_123097


namespace last_digit_sum_powers_l1230_123008

theorem last_digit_sum_powers : (2^2011 + 3^2011) % 10 = 5 := by sorry

end last_digit_sum_powers_l1230_123008


namespace enrollment_system_correct_l1230_123024

/-- Represents the enrollment plan and actual enrollment of a school -/
structure EnrollmentPlan where
  planned_total : ℕ
  actual_total : ℕ
  boys_exceed_percent : ℚ
  girls_exceed_percent : ℚ

/-- The correct system of equations for the enrollment plan -/
def correct_system (plan : EnrollmentPlan) (x y : ℚ) : Prop :=
  x + y = plan.planned_total ∧
  (1 + plan.boys_exceed_percent) * x + (1 + plan.girls_exceed_percent) * y = plan.actual_total

/-- Theorem stating that the given system of equations is correct for the enrollment plan -/
theorem enrollment_system_correct (plan : EnrollmentPlan)
  (h1 : plan.planned_total = 1000)
  (h2 : plan.actual_total = 1240)
  (h3 : plan.boys_exceed_percent = 1/5)
  (h4 : plan.girls_exceed_percent = 3/10) :
  ∀ x y : ℚ, correct_system plan x y ↔ 
    (x + y = 1000 ∧ 6/5 * x + 13/10 * y = 1240) :=
by sorry

end enrollment_system_correct_l1230_123024


namespace min_slope_tangent_line_l1230_123094

noncomputable def f (x b a : ℝ) : ℝ := Real.log x + x^2 - b*x + a

theorem min_slope_tangent_line (b a : ℝ) (hb : b > 0) :
  ∃ m : ℝ, m = 2 ∧ ∀ x, x > 0 → (1/x + x) ≥ m :=
sorry

end min_slope_tangent_line_l1230_123094


namespace inequality_range_l1230_123059

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, (Real.sin x)^2 + a * Real.cos x - a^2 ≤ 1 + Real.cos x) ↔ 
  (a ≤ -1 ∨ a ≥ 1/3) :=
sorry

end inequality_range_l1230_123059


namespace sin_cos_alpha_abs_value_l1230_123043

theorem sin_cos_alpha_abs_value (α : Real) 
  (h : Real.sin (3 * Real.pi - α) = -2 * Real.sin (Real.pi / 2 + α)) : 
  |Real.sin α * Real.cos α| = 2/5 := by sorry

end sin_cos_alpha_abs_value_l1230_123043


namespace no_eight_term_ap_with_r_ap_l1230_123016

/-- r(n) is the odd positive integer whose binary representation is the reverse of n's binary representation -/
def r (n : ℕ) : ℕ :=
  sorry

/-- Theorem: There does not exist a strictly increasing eight-term arithmetic progression of odd positive integers such that their r values form an arithmetic progression -/
theorem no_eight_term_ap_with_r_ap :
  ¬ ∃ (a : ℕ → ℕ) (d : ℕ),
    (∀ i, i ∈ Finset.range 8 → Odd (a i)) ∧
    (∀ i j, i < j → i ∈ Finset.range 8 → j ∈ Finset.range 8 → a i < a j) ∧
    (∀ i, i ∈ Finset.range 7 → a (i + 1) - a i = d) ∧
    (∃ d' : ℕ, ∀ i, i ∈ Finset.range 7 → r (a (i + 1)) - r (a i) = d') :=
  sorry

end no_eight_term_ap_with_r_ap_l1230_123016


namespace range_of_m_l1230_123011

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x | 2*m - 1 ≤ x ∧ x ≤ m + 3}

-- Theorem statement
theorem range_of_m (m : ℝ) : B m ⊆ A → m < -4 ∨ m > 2 := by
  sorry

end range_of_m_l1230_123011


namespace max_closable_companies_l1230_123023

/-- The number of planets (vertices) in the Intergalactic empire -/
def n : ℕ := 10^2015

/-- The number of travel companies (colors) -/
def m : ℕ := 2015

/-- A function that determines if a graph remains connected after removing k colors -/
def remains_connected (k : ℕ) : Prop :=
  ∀ (removed_colors : Finset (Fin m)),
    removed_colors.card = k →
    ∃ (remaining_graph : SimpleGraph (Fin n)),
      remaining_graph.Connected

/-- The theorem stating the maximum number of companies that can be closed -/
theorem max_closable_companies :
  (∀ k ≤ 1007, remains_connected k) ∧
  ¬(remains_connected 1008) :=
sorry

end max_closable_companies_l1230_123023


namespace color_combination_count_l1230_123026

/-- The number of colors available -/
def total_colors : ℕ := 9

/-- The number of colors to be chosen -/
def colors_to_choose : ℕ := 2

/-- The number of forbidden combinations (red and pink) -/
def forbidden_combinations : ℕ := 1

/-- The number of ways to choose colors, excluding forbidden combinations -/
def valid_combinations : ℕ := (total_colors.choose colors_to_choose) - forbidden_combinations

theorem color_combination_count : valid_combinations = 35 := by
  sorry

end color_combination_count_l1230_123026


namespace stratified_sampling_proportion_l1230_123084

theorem stratified_sampling_proportion (total_population : ℕ) (stratum_a : ℕ) (stratum_b : ℕ) (sample_size : ℕ) :
  total_population = stratum_a + stratum_b →
  total_population = 120 →
  stratum_a = 20 →
  stratum_b = 100 →
  sample_size = 12 →
  (sample_size * stratum_a) / total_population = 2 :=
by sorry

end stratified_sampling_proportion_l1230_123084


namespace solve_quadratic_equation_l1230_123070

theorem solve_quadratic_equation (s t : ℝ) (h1 : t = 8 * s^2) (h2 : t = 4.8) :
  s = Real.sqrt 0.6 ∨ s = -Real.sqrt 0.6 := by
  sorry

end solve_quadratic_equation_l1230_123070


namespace triangle_perimeter_l1230_123057

-- Define the triangle
def triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the equation for the third side
def third_side_equation (x : ℝ) : Prop :=
  x^2 - 8*x + 12 = 0

-- Theorem statement
theorem triangle_perimeter : 
  ∃ (x : ℝ), 
    third_side_equation x ∧ 
    triangle 4 7 x ∧ 
    4 + 7 + x = 17 := by
  sorry

end triangle_perimeter_l1230_123057


namespace koolaid_percentage_is_four_percent_l1230_123019

def koolaid_percentage (initial_powder initial_water evaporation water_multiplier : ℕ) : ℚ :=
  let remaining_water := initial_water - evaporation
  let final_water := remaining_water * water_multiplier
  let total_liquid := initial_powder + final_water
  (initial_powder : ℚ) / total_liquid * 100

theorem koolaid_percentage_is_four_percent :
  koolaid_percentage 2 16 4 4 = 4 := by
  sorry

end koolaid_percentage_is_four_percent_l1230_123019


namespace x_range_for_P_in_fourth_quadrant_l1230_123099

-- Define the point P
def P (x : ℝ) : ℝ × ℝ := (2*x - 6, x - 5)

-- Define the condition for a point to be in the fourth quadrant
def in_fourth_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

-- Theorem statement
theorem x_range_for_P_in_fourth_quadrant :
  ∀ x : ℝ, in_fourth_quadrant (P x) ↔ 3 < x ∧ x < 5 := by sorry

end x_range_for_P_in_fourth_quadrant_l1230_123099


namespace sparrows_among_non_pigeons_l1230_123045

theorem sparrows_among_non_pigeons (sparrows : ℝ) (pigeons : ℝ) (parrots : ℝ) (crows : ℝ)
  (h1 : sparrows = 0.4)
  (h2 : pigeons = 0.2)
  (h3 : parrots = 0.15)
  (h4 : crows = 0.25)
  (h5 : sparrows + pigeons + parrots + crows = 1) :
  sparrows / (1 - pigeons) = 0.5 := by
sorry

end sparrows_among_non_pigeons_l1230_123045


namespace helen_cookies_l1230_123061

/-- The number of chocolate chip cookies Helen baked yesterday -/
def cookies_yesterday : ℕ := 527

/-- The number of chocolate chip cookies Helen baked this morning -/
def cookies_today : ℕ := 554

/-- The total number of chocolate chip cookies Helen baked -/
def total_cookies : ℕ := cookies_yesterday + cookies_today

theorem helen_cookies : total_cookies = 1081 := by
  sorry

end helen_cookies_l1230_123061


namespace complex_real_condition_l1230_123002

theorem complex_real_condition (m : ℝ) : 
  let z : ℂ := (m - 2 : ℂ) + (m^2 - 3*m + 2 : ℂ) * Complex.I
  (z ≠ 0 ∧ z.im = 0) → m = 1 :=
by sorry

end complex_real_condition_l1230_123002


namespace inequality_solution_set_l1230_123078

theorem inequality_solution_set : 
  {x : ℝ | x * (2 - x) ≤ 0} = {x : ℝ | x ≤ 0 ∨ x ≥ 2} := by sorry

end inequality_solution_set_l1230_123078


namespace prob_other_is_one_given_one_is_one_l1230_123093

/-- Represents the number of balls with each label -/
def ballCounts : Fin 3 → Nat
  | 0 => 1  -- number of balls labeled 0
  | 1 => 2  -- number of balls labeled 1
  | 2 => 2  -- number of balls labeled 2

/-- The total number of balls -/
def totalBalls : Nat := (ballCounts 0) + (ballCounts 1) + (ballCounts 2)

/-- The probability of drawing two balls, one of which is labeled 1 -/
def probOneIsOne : ℚ := (ballCounts 1 * (totalBalls - ballCounts 1)) / (totalBalls.choose 2)

/-- The probability of drawing two balls, both labeled 1 -/
def probBothAreOne : ℚ := ((ballCounts 1).choose 2) / (totalBalls.choose 2)

/-- The main theorem to prove -/
theorem prob_other_is_one_given_one_is_one :
  probBothAreOne / probOneIsOne = 1 / 7 := by sorry

end prob_other_is_one_given_one_is_one_l1230_123093


namespace total_length_of_remaining_segments_l1230_123038

/-- A figure with perpendicular adjacent sides -/
structure PerpendicularFigure where
  top_segments : List ℝ
  bottom_segment : ℝ
  left_segment : ℝ
  right_segment : ℝ

/-- The remaining figure after removing six sides -/
def RemainingFigure (f : PerpendicularFigure) : PerpendicularFigure :=
  { top_segments := [1],
    bottom_segment := f.bottom_segment,
    left_segment := f.left_segment,
    right_segment := 9 }

theorem total_length_of_remaining_segments (f : PerpendicularFigure)
  (h1 : f.top_segments = [3, 1, 1])
  (h2 : f.left_segment = 10)
  (h3 : f.bottom_segment = f.top_segments.sum)
  : (RemainingFigure f).top_segments.sum + 
    (RemainingFigure f).bottom_segment + 
    (RemainingFigure f).left_segment + 
    (RemainingFigure f).right_segment = 25 := by
  sorry


end total_length_of_remaining_segments_l1230_123038


namespace quadratic_inequality_solution_set_l1230_123075

theorem quadratic_inequality_solution_set (x : ℝ) : 
  {x : ℝ | x^2 - 4*x + 3 < 0} = Set.Ioo 1 3 := by
  sorry

end quadratic_inequality_solution_set_l1230_123075


namespace binary_addition_subtraction_l1230_123041

/-- Converts a list of bits (represented as Bools) to a natural number. -/
def bitsToNat (bits : List Bool) : Nat :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- The binary number 10101₂ -/
def num1 : List Bool := [true, false, true, false, true]

/-- The binary number 1011₂ -/
def num2 : List Bool := [true, false, true, true]

/-- The binary number 1110₂ -/
def num3 : List Bool := [true, true, true, false]

/-- The binary number 110001₂ -/
def num4 : List Bool := [true, true, false, false, false, true]

/-- The binary number 1101₂ -/
def num5 : List Bool := [true, true, false, true]

/-- The binary number 101100₂ (the expected result) -/
def result : List Bool := [true, false, true, true, false, false]

theorem binary_addition_subtraction :
  bitsToNat num1 + bitsToNat num2 + bitsToNat num3 + bitsToNat num4 - bitsToNat num5 = bitsToNat result := by
  sorry

end binary_addition_subtraction_l1230_123041


namespace max_rabbits_with_traits_l1230_123036

theorem max_rabbits_with_traits (N : ℕ) : 
  (∃ (long_ears jump_far both : Finset (Fin N)),
    long_ears.card = 13 ∧ 
    jump_far.card = 17 ∧ 
    both ⊆ long_ears ∧ 
    both ⊆ jump_far ∧ 
    both.card ≥ 3) →
  N ≤ 27 := by
sorry

end max_rabbits_with_traits_l1230_123036


namespace quadratic_expression_equals_39_l1230_123091

theorem quadratic_expression_equals_39 (x : ℝ) :
  (x + 2)^2 + 2*(x + 2)*(4 - x) + (4 - x)^2 + 3 = 39 := by
  sorry

end quadratic_expression_equals_39_l1230_123091


namespace sum_of_cubes_equation_l1230_123063

theorem sum_of_cubes_equation (x y : ℝ) : 
  x^3 + 21*x*y + y^3 = 343 → (x + y = 7 ∨ x + y = -14) :=
by sorry

end sum_of_cubes_equation_l1230_123063


namespace max_min_x_plus_y_l1230_123018

noncomputable def f (x y : ℝ) : Prop :=
  1 - Real.sqrt (x - 1) = Real.sqrt (y - 1) ∧ x ≥ 1 ∧ y ≥ 1

theorem max_min_x_plus_y :
  ∀ x y : ℝ, f x y →
    (∀ a b : ℝ, f a b → x + y ≤ a + b) ∧
    (∃ a b : ℝ, f a b ∧ a + b = 3) ∧
    (∀ a b : ℝ, f a b → a + b ≥ 5/2) ∧
    (∃ a b : ℝ, f a b ∧ a + b = 5/2) :=
by sorry

end max_min_x_plus_y_l1230_123018


namespace x_one_value_l1230_123004

theorem x_one_value (x₁ x₂ x₃ : Real) 
  (h1 : 0 ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 0.8)
  (h2 : (1-x₁)^2 + (x₁-x₂)^2 + (x₂-x₃)^2 + x₃^2 = 1/3) :
  x₁ = 3/4 := by
  sorry

end x_one_value_l1230_123004


namespace square_property_l1230_123029

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def remove_last_two_digits (n : ℕ) : ℕ := n / 100

theorem square_property (n : ℕ) :
  (n > 0 ∧ is_perfect_square (remove_last_two_digits (n^2))) ↔
  (∃ k : ℕ, k > 0 ∧ n = 10 * k) ∨
  (n ∈ ({11,12,13,14,21,22,31,41,1,2,3,4,5,6,7,8,9} : Finset ℕ)) :=
sorry

end square_property_l1230_123029


namespace square_difference_formula_l1230_123007

-- Define the expressions
def expr_A (x : ℝ) := (x + 1) * (x - 1)
def expr_B (x : ℝ) := (-x + 1) * (-x - 1)
def expr_C (x : ℝ) := (x + 1) * (-x + 1)
def expr_D (x : ℝ) := (x + 1) * (1 + x)

-- Define a predicate for expressions that can be written as a difference of squares
def is_diff_of_squares (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ → ℝ), ∀ x, f x = (a x)^2 - (b x)^2

-- State the theorem
theorem square_difference_formula :
  (is_diff_of_squares expr_A) ∧
  (is_diff_of_squares expr_B) ∧
  (is_diff_of_squares expr_C) ∧
  ¬(is_diff_of_squares expr_D) := by
  sorry

end square_difference_formula_l1230_123007


namespace running_competition_sample_l1230_123035

/-- Given a school with 2000 students, where 3/5 participate in a running competition
    with grade ratios of 2:3:5, and a sample of 200 students is taken, 
    the number of 2nd grade students in the running competition sample is 36. -/
theorem running_competition_sample (total_students : ℕ) (sample_size : ℕ) 
  (running_ratio : ℚ) (grade_ratios : Fin 3 → ℚ) :
  total_students = 2000 →
  sample_size = 200 →
  running_ratio = 3/5 →
  grade_ratios 0 = 2/10 ∧ grade_ratios 1 = 3/10 ∧ grade_ratios 2 = 5/10 →
  ↑sample_size * running_ratio * grade_ratios 1 = 36 := by
  sorry

#check running_competition_sample

end running_competition_sample_l1230_123035


namespace smartphone_sample_correct_l1230_123085

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  group_size : ℕ
  first_item : ℕ

/-- Conditions for the smartphone sampling problem -/
def smartphone_sample : SystematicSample where
  population_size := 160
  sample_size := 20
  group_size := 8
  first_item := 2  -- This is what we want to prove

theorem smartphone_sample_correct :
  let s := smartphone_sample
  s.population_size = 160 ∧
  s.sample_size = 20 ∧
  s.group_size = 8 ∧
  (s.first_item + 8 * 8 + s.first_item + 9 * 8 = 140) →
  s.first_item = 2 := by sorry

end smartphone_sample_correct_l1230_123085


namespace assignment_b_is_valid_l1230_123074

-- Define what a valid assignment statement is
def is_valid_assignment (stmt : String) : Prop :=
  ∃ (var : String) (expr : String), stmt = var ++ "=" ++ expr ∧ var.length > 0

-- Define the specific statement we're checking
def statement_to_check : String := "a=a+1"

-- Theorem to prove
theorem assignment_b_is_valid : is_valid_assignment statement_to_check := by
  sorry

end assignment_b_is_valid_l1230_123074


namespace edric_hourly_rate_l1230_123032

/-- Edric's salary calculation --/
def salary_calculation (B C S P D : ℚ) (H W : ℕ) : ℚ :=
  let E := B + (C * S) + P - D
  let T := (H * W * 4 : ℚ)
  E / T

/-- Edric's hourly rate is approximately $3.86 --/
theorem edric_hourly_rate :
  let B := 576
  let C := 3 / 100
  let S := 4000
  let P := 75
  let D := 30
  let H := 8
  let W := 6
  abs (salary_calculation B C S P D H W - 386 / 100) < 1 / 100 := by
  sorry

end edric_hourly_rate_l1230_123032


namespace new_members_average_weight_l1230_123058

theorem new_members_average_weight 
  (initial_count : ℕ) 
  (initial_average : ℝ) 
  (new_count : ℕ) 
  (new_average : ℝ) 
  (double_counted_weight : ℝ) :
  initial_count = 10 →
  initial_average = 75 →
  new_count = 3 →
  new_average = 77 →
  double_counted_weight = 65 →
  let corrected_total := initial_count * initial_average - double_counted_weight
  let new_total := (initial_count + new_count - 1) * new_average
  let new_members_total := new_total - corrected_total
  (new_members_total / new_count) = 79.67 := by
sorry

end new_members_average_weight_l1230_123058


namespace larger_triangle_perimeter_l1230_123076

-- Define the original right triangle
def original_triangle (a b c : ℝ) : Prop :=
  a = 8 ∧ b = 15 ∧ c^2 = a^2 + b^2

-- Define the similarity ratio
def similarity_ratio (k : ℝ) (a : ℝ) : Prop :=
  k * a = 20 ∧ k > 0

-- Define the larger similar triangle
def larger_triangle (a b c k : ℝ) : Prop :=
  original_triangle a b c ∧ similarity_ratio k a

-- Theorem statement
theorem larger_triangle_perimeter 
  (a b c k : ℝ) 
  (h : larger_triangle a b c k) : 
  k * (a + b + c) = 100 := by
    sorry


end larger_triangle_perimeter_l1230_123076


namespace pigeonhole_on_permutation_sums_l1230_123088

theorem pigeonhole_on_permutation_sums (n : ℕ) : 
  ∀ (p : Fin (2*n) → Fin (2*n)), 
  ∃ (i j : Fin (2*n)), i ≠ j ∧ 
  ((p i).val + i.val + 1) % (2*n) = ((p j).val + j.val + 1) % (2*n) :=
sorry

end pigeonhole_on_permutation_sums_l1230_123088


namespace max_value_expression_l1230_123066

theorem max_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z)^2 / (x^2 + y^2 + z^2) ≤ 3 ∧
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ (a + b + c)^2 / (a^2 + b^2 + c^2) = 3 :=
sorry

end max_value_expression_l1230_123066


namespace sum_of_three_at_least_fifty_l1230_123000

theorem sum_of_three_at_least_fifty (S : Finset ℕ) (h1 : S.card = 7) 
  (h2 : ∀ x ∈ S, x > 0) (h3 : S.sum id = 100) :
  ∃ T ⊆ S, T.card = 3 ∧ T.sum id ≥ 50 := by
  sorry

end sum_of_three_at_least_fifty_l1230_123000


namespace max_delta_ratio_l1230_123005

/-- Represents a contestant's score in a two-day competition -/
structure ContestantScore where
  day1_score : ℕ
  day1_total : ℕ
  day2_score : ℕ
  day2_total : ℕ

/-- Calculate the two-day success ratio -/
def two_day_ratio (score : ContestantScore) : ℚ :=
  (score.day1_score + score.day2_score : ℚ) / (score.day1_total + score.day2_total)

/-- Charlie's score in the competition -/
def charlie : ContestantScore :=
  { day1_score := 210, day1_total := 400, day2_score := 150, day2_total := 200 }

theorem max_delta_ratio :
  ∀ delta : ContestantScore,
    delta.day1_score > 0 ∧ 
    delta.day2_score > 0 ∧
    delta.day1_total + delta.day2_total = 600 ∧
    (delta.day1_score : ℚ) / delta.day1_total < 210 / 400 ∧
    (delta.day2_score : ℚ) / delta.day2_total < 3 / 4 →
    two_day_ratio delta ≤ 349 / 600 :=
  sorry

end max_delta_ratio_l1230_123005


namespace line_AB_parallel_to_xOz_plane_l1230_123050

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a vector in 3D space -/
def Vector3D : Type := ℝ × ℝ × ℝ

/-- Calculate the vector from point A to point B -/
def vectorBetweenPoints (A B : Point3D) : Vector3D :=
  (B.x - A.x, B.y - A.y, B.z - A.z)

/-- Check if a vector is parallel to the xOz plane -/
def isParallelToXOZ (v : Vector3D) : Prop :=
  v.2 = 0

/-- The main theorem: Line AB is parallel to xOz plane -/
theorem line_AB_parallel_to_xOz_plane :
  let A : Point3D := ⟨1, 3, 0⟩
  let B : Point3D := ⟨0, 3, -1⟩
  let AB : Vector3D := vectorBetweenPoints A B
  isParallelToXOZ AB := by sorry

end line_AB_parallel_to_xOz_plane_l1230_123050


namespace multiples_imply_lower_bound_l1230_123087

theorem multiples_imply_lower_bound (n : ℕ) (a : ℕ) (h1 : n > 1) (h2 : a > n^2) :
  (∀ i ∈ Finset.range n, ∃ k ∈ Finset.range n, (a + k + 1) % (n^2 + i + 1) = 0) →
  a > n^4 - n^3 := by
  sorry

end multiples_imply_lower_bound_l1230_123087


namespace x_value_l1230_123081

theorem x_value (x y : ℚ) (h1 : x / y = 15 / 5) (h2 : y = 10) : x = 30 := by
  sorry

end x_value_l1230_123081


namespace m_in_open_interval_l1230_123071

def monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem m_in_open_interval
  (f : ℝ → ℝ)
  (h_decreasing : monotonically_decreasing f)
  (h_inequality : f (m^2) > f m)
  : m ∈ Set.Ioo 0 1 :=
by sorry

end m_in_open_interval_l1230_123071


namespace inequality_proof_l1230_123055

theorem inequality_proof : (-abs (abs (-20 : ℝ))) / 2 > -4.5 := by
  sorry

end inequality_proof_l1230_123055


namespace B_when_a_is_3_range_of_a_when_A_equals_B_l1230_123010

-- Define the set B
def B (a : ℝ) : Set ℝ := {x | (a - 2) * x^2 + 2 * (a - 2) * x - 3 < 0}

-- Theorem 1: When a = 3, B = (-3, 1)
theorem B_when_a_is_3 : B 3 = Set.Ioo (-3) 1 := by sorry

-- Theorem 2: When A = B = ℝ, a ∈ (-1, 2]
theorem range_of_a_when_A_equals_B :
  (∀ x, x ∈ B a) ↔ a ∈ Set.Ioc (-1) 2 := by sorry

end B_when_a_is_3_range_of_a_when_A_equals_B_l1230_123010


namespace power_four_remainder_l1230_123014

theorem power_four_remainder (a : ℕ) (h1 : a > 0) (h2 : 2 ∣ a) : 4^a % 10 = 6 := by
  sorry

end power_four_remainder_l1230_123014


namespace josh_marbles_count_l1230_123092

/-- The number of marbles Josh has after receiving some from Jack -/
def total_marbles (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem stating that Josh's final marble count is the sum of his initial count and received marbles -/
theorem josh_marbles_count (initial : ℕ) (received : ℕ) :
  total_marbles initial received = initial + received := by
  sorry

end josh_marbles_count_l1230_123092


namespace percentage_problem_l1230_123051

theorem percentage_problem (x : ℝ) (h : 0.2 * x = 60) : 0.8 * x = 240 := by
  sorry

end percentage_problem_l1230_123051


namespace solution_in_interval_implies_a_range_l1230_123039

theorem solution_in_interval_implies_a_range (a : ℝ) :
  (∃ x ∈ Set.Icc 1 5, x^2 + a*x - 2 = 0) →
  a ∈ Set.Icc (-23/5) 1 :=
by sorry

end solution_in_interval_implies_a_range_l1230_123039


namespace concrete_blocks_theorem_l1230_123034

/-- Calculates the number of concrete blocks per section in a hedge. -/
def concrete_blocks_per_section (total_sections : ℕ) (total_cost : ℕ) (cost_per_piece : ℕ) : ℕ :=
  (total_cost / cost_per_piece) / total_sections

/-- Proves that the number of concrete blocks per section is 30 given the specified conditions. -/
theorem concrete_blocks_theorem :
  concrete_blocks_per_section 8 480 2 = 30 := by
  sorry

end concrete_blocks_theorem_l1230_123034


namespace intersection_M_N_l1230_123072

def M : Set ℝ := {x | |x + 1| ≤ 1}
def N : Set ℝ := {-1, 0, 1}

theorem intersection_M_N : M ∩ N = {-1, 0} := by sorry

end intersection_M_N_l1230_123072


namespace complement_union_theorem_l1230_123098

def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x : ℝ | x < 6}

theorem complement_union_theorem :
  (Set.univ \ B) ∪ A = {x : ℝ | x ≥ 0} := by sorry

end complement_union_theorem_l1230_123098


namespace coconut_grove_theorem_l1230_123069

theorem coconut_grove_theorem (x : ℝ) : 
  ((x + 4) * 60 + x * 120 + (x - 4) * 180) / (3 * x) = 100 → x = 8 := by
  sorry

end coconut_grove_theorem_l1230_123069


namespace equation_solutions_l1230_123079

theorem equation_solutions : 
  {x : ℝ | x^2 - 3 * |x| - 4 = 0} = {4, -4} := by sorry

end equation_solutions_l1230_123079


namespace at_least_one_quadratic_has_root_l1230_123047

theorem at_least_one_quadratic_has_root (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ x : ℝ, (a * x^2 + 2 * b * x + c = 0) ∨ (b * x^2 + 2 * c * x + a = 0) ∨ (c * x^2 + 2 * a * x + b = 0) := by
  sorry

end at_least_one_quadratic_has_root_l1230_123047


namespace cos_equation_solution_l1230_123046

theorem cos_equation_solution (x : ℝ) : 
  (Real.cos (2 * x) - 3 * Real.cos (4 * x))^2 = 16 + (Real.cos (5 * x))^2 → 
  ∃ k : ℤ, x = π / 2 + k * π :=
by sorry

end cos_equation_solution_l1230_123046


namespace water_tower_capacity_l1230_123048

/-- The capacity of a water tower serving four neighborhoods --/
theorem water_tower_capacity :
  let first_neighborhood : ℕ := 150
  let second_neighborhood : ℕ := 2 * first_neighborhood
  let third_neighborhood : ℕ := second_neighborhood + 100
  let fourth_neighborhood : ℕ := 350
  first_neighborhood + second_neighborhood + third_neighborhood + fourth_neighborhood = 1200 :=
by sorry

end water_tower_capacity_l1230_123048


namespace probability_theorem_l1230_123068

def total_balls : ℕ := 6
def new_balls : ℕ := 4
def old_balls : ℕ := 2

def probability_one_new_one_old : ℚ :=
  (new_balls * old_balls) / (total_balls * (total_balls - 1) / 2)

theorem probability_theorem :
  probability_one_new_one_old = 8 / 15 := by
  sorry

end probability_theorem_l1230_123068


namespace oxygen_atoms_in_compound_l1230_123049

def atomic_weight_carbon : ℕ := 12
def atomic_weight_hydrogen : ℕ := 1
def atomic_weight_oxygen : ℕ := 16

def num_carbon_atoms : ℕ := 3
def num_hydrogen_atoms : ℕ := 6
def total_molecular_weight : ℕ := 58

theorem oxygen_atoms_in_compound :
  let weight_carbon_hydrogen := num_carbon_atoms * atomic_weight_carbon + num_hydrogen_atoms * atomic_weight_hydrogen
  let weight_oxygen := total_molecular_weight - weight_carbon_hydrogen
  weight_oxygen / atomic_weight_oxygen = 1 := by sorry

end oxygen_atoms_in_compound_l1230_123049


namespace circle_properties_l1230_123044

/-- Given a circle with diameter endpoints (2, 1) and (8, 7), prove its center and diameter length -/
theorem circle_properties :
  let p1 : ℝ × ℝ := (2, 1)
  let p2 : ℝ × ℝ := (8, 7)
  let center := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  let diameter_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  center = (5, 4) ∧ diameter_length = 6 * Real.sqrt 2 := by
  sorry

end circle_properties_l1230_123044


namespace remaining_episodes_l1230_123095

theorem remaining_episodes (series1_seasons series2_seasons episodes_per_season episodes_lost_per_season : ℕ) 
  (h1 : series1_seasons = 12)
  (h2 : series2_seasons = 14)
  (h3 : episodes_per_season = 16)
  (h4 : episodes_lost_per_season = 2) :
  (series1_seasons * episodes_per_season + series2_seasons * episodes_per_season) -
  (series1_seasons * episodes_lost_per_season + series2_seasons * episodes_lost_per_season) = 364 := by
sorry

end remaining_episodes_l1230_123095


namespace solve_for_k_l1230_123065

/-- The function f(x) = 4x³ - 3x² + 2x + 5 -/
def f (x : ℝ) : ℝ := 4 * x^3 - 3 * x^2 + 2 * x + 5

/-- The function g(x) = x³ - (k+1)x² - 7x - 8 -/
def g (k x : ℝ) : ℝ := x^3 - (k + 1) * x^2 - 7 * x - 8

/-- If f(5) - g(5) = 24, then k = -16.36 -/
theorem solve_for_k : ∃ k : ℝ, f 5 - g k 5 = 24 ∧ k = -16.36 := by sorry

end solve_for_k_l1230_123065


namespace sandy_current_fingernail_length_l1230_123080

/-- Sandy's current age in years -/
def current_age : ℕ := 12

/-- Sandy's age when she achieves the world record in years -/
def record_age : ℕ := 32

/-- The world record for longest fingernails in inches -/
def world_record : ℝ := 26

/-- Sandy's fingernail growth rate in inches per month -/
def growth_rate : ℝ := 0.1

/-- The number of months in a year -/
def months_per_year : ℕ := 12

theorem sandy_current_fingernail_length :
  world_record - (growth_rate * months_per_year * (record_age - current_age : ℝ)) = 2 := by
  sorry

end sandy_current_fingernail_length_l1230_123080


namespace exists_minimal_period_greater_than_l1230_123053

/-- Definition of the sequence family F(x) -/
def F (x : ℝ) : (ℕ → ℝ) → Prop :=
  λ a => ∀ n, a (n + 1) = x - 1 / a n

/-- Definition of periodicity for a sequence -/
def IsPeriodic (a : ℕ → ℝ) (p : ℕ) : Prop :=
  ∀ n, a (n + p) = a n

/-- Definition of minimal period for the family F(x) -/
def IsMinimalPeriod (x : ℝ) (p : ℕ) : Prop :=
  (∀ a, F x a → IsPeriodic a p) ∧
  (∀ q, 0 < q → q < p → ∃ a, F x a ∧ ¬IsPeriodic a q)

/-- Main theorem statement -/
theorem exists_minimal_period_greater_than (P : ℕ) :
  ∃ x : ℝ, ∃ p : ℕ, p > P ∧ IsMinimalPeriod x p :=
sorry

end exists_minimal_period_greater_than_l1230_123053


namespace min_orders_for_given_conditions_l1230_123028

/-- The minimum number of orders required to purchase a given number of items
    while minimizing the total cost under specific discount conditions. -/
def min_orders (original_price : ℚ) (total_items : ℕ) (discount_percent : ℚ) 
                (additional_discount_threshold : ℚ) (additional_discount : ℚ) : ℕ :=
  sorry

/-- The theorem stating that the minimum number of orders is 4 under the given conditions. -/
theorem min_orders_for_given_conditions : 
  min_orders 48 42 0.6 300 100 = 4 := by sorry

end min_orders_for_given_conditions_l1230_123028


namespace geometric_sequence_sum_l1230_123086

/-- The number of terms in a geometric sequence with first term 1 and common ratio 1/4 
    that sum to 85/64 -/
theorem geometric_sequence_sum (n : ℕ) : 
  (1 - (1/4)^n) / (1 - 1/4) = 85/64 → n = 4 :=
by sorry

end geometric_sequence_sum_l1230_123086


namespace rectangles_with_one_gray_cell_l1230_123064

/-- The number of rectangles containing exactly one gray cell in a checkered rectangle -/
theorem rectangles_with_one_gray_cell 
  (total_gray_cells : ℕ) 
  (cells_with_four_rectangles : ℕ) 
  (cells_with_eight_rectangles : ℕ) 
  (h1 : total_gray_cells = 40)
  (h2 : cells_with_four_rectangles = 36)
  (h3 : cells_with_eight_rectangles = 4)
  (h4 : total_gray_cells = cells_with_four_rectangles + cells_with_eight_rectangles) :
  cells_with_four_rectangles * 4 + cells_with_eight_rectangles * 8 = 176 := by
sorry

end rectangles_with_one_gray_cell_l1230_123064


namespace general_ticket_price_is_six_l1230_123021

/-- Represents the ticket sales and pricing scenario -/
structure TicketSale where
  student_price : ℕ
  total_tickets : ℕ
  total_revenue : ℕ
  general_tickets : ℕ

/-- Calculates the price of a general admission ticket -/
def general_price (sale : TicketSale) : ℚ :=
  (sale.total_revenue - sale.student_price * (sale.total_tickets - sale.general_tickets)) / sale.general_tickets

/-- Theorem stating that the general admission ticket price is 6 dollars -/
theorem general_ticket_price_is_six (sale : TicketSale) 
  (h1 : sale.student_price = 4)
  (h2 : sale.total_tickets = 525)
  (h3 : sale.total_revenue = 2876)
  (h4 : sale.general_tickets = 388) :
  general_price sale = 6 := by
  sorry

#eval general_price {
  student_price := 4,
  total_tickets := 525,
  total_revenue := 2876,
  general_tickets := 388
}

end general_ticket_price_is_six_l1230_123021


namespace finn_bought_12_boxes_l1230_123089

/-- The cost of one package of index cards -/
def index_card_cost : ℚ := (55.40 - 15 * 1.85) / 7

/-- The number of boxes of paper clips Finn bought -/
def finn_paper_clips : ℚ := (61.70 - 10 * index_card_cost) / 1.85

theorem finn_bought_12_boxes :
  finn_paper_clips = 12 := by sorry

end finn_bought_12_boxes_l1230_123089


namespace minimum_contribution_l1230_123077

theorem minimum_contribution 
  (n : ℕ) 
  (total : ℝ) 
  (max_individual : ℝ) 
  (h1 : n = 15) 
  (h2 : total = 30) 
  (h3 : max_individual = 16) : 
  ∃ (min_contribution : ℝ), 
    (∀ (i : ℕ), i ≤ n → min_contribution ≤ max_individual) ∧ 
    (n * min_contribution ≤ total) ∧ 
    (∀ (x : ℝ), (∀ (i : ℕ), i ≤ n → x ≤ max_individual) ∧ (n * x ≤ total) → x ≤ min_contribution) ∧
    min_contribution = 2 := by
  sorry

end minimum_contribution_l1230_123077


namespace sum_of_squares_of_roots_l1230_123022

theorem sum_of_squares_of_roots (p q r : ℝ) : 
  (3 * p^3 - 2 * p^2 + 5 * p + 15 = 0) →
  (3 * q^3 - 2 * q^2 + 5 * q + 15 = 0) →
  (3 * r^3 - 2 * r^2 + 5 * r + 15 = 0) →
  p^2 + q^2 + r^2 = -26/9 := by
sorry

end sum_of_squares_of_roots_l1230_123022


namespace cost_price_calculation_l1230_123083

theorem cost_price_calculation (C : ℝ) : 
  (0.9 * C = C - 0.1 * C) →
  (1.1 * C = C + 0.1 * C) →
  (1.1 * C - 0.9 * C = 50) →
  C = 250 := by
sorry

end cost_price_calculation_l1230_123083


namespace total_practice_time_is_307_5_l1230_123073

/-- Represents Daniel's weekly practice schedule -/
structure PracticeSchedule where
  basketball_school_day : ℝ  -- Minutes of basketball practice on school days
  basketball_weekend_day : ℝ  -- Minutes of basketball practice on weekend days
  soccer_weekday : ℝ  -- Minutes of soccer practice on weekdays
  gymnastics : ℝ  -- Minutes of gymnastics practice
  soccer_saturday : ℝ  -- Minutes of soccer practice on Saturday (averaged)
  swimming_saturday : ℝ  -- Minutes of swimming practice on Saturday (averaged)

/-- Calculates the total practice time for one week -/
def total_practice_time (schedule : PracticeSchedule) : ℝ :=
  schedule.basketball_school_day * 5 +
  schedule.basketball_weekend_day * 2 +
  schedule.soccer_weekday * 3 +
  schedule.gymnastics * 2 +
  schedule.soccer_saturday +
  schedule.swimming_saturday

/-- Daniel's actual practice schedule -/
def daniel_schedule : PracticeSchedule :=
  { basketball_school_day := 15
  , basketball_weekend_day := 30
  , soccer_weekday := 20
  , gymnastics := 30
  , soccer_saturday := 22.5
  , swimming_saturday := 30 }

theorem total_practice_time_is_307_5 :
  total_practice_time daniel_schedule = 307.5 := by
  sorry

end total_practice_time_is_307_5_l1230_123073


namespace log_inequality_l1230_123037

theorem log_inequality : ∃ (a b c : ℝ), 
  a = Real.log 2 / Real.log 5 ∧ 
  b = Real.log 3 / Real.log 8 ∧ 
  c = (1 : ℝ) / 2 ∧ 
  a < c ∧ c < b :=
sorry

end log_inequality_l1230_123037


namespace dealer_profit_theorem_l1230_123042

/-- Represents the pricing and discount strategy of a dealer -/
structure DealerStrategy where
  markup_percentage : ℝ
  discount_percentage : ℝ
  bulk_deal_articles_sold : ℕ
  bulk_deal_articles_cost : ℕ

/-- Calculates the profit percentage for a dealer given their strategy -/
def calculate_profit_percentage (strategy : DealerStrategy) : ℝ :=
  -- The actual calculation is not implemented here
  sorry

/-- Theorem stating that the dealer's profit percentage is 80% under the given conditions -/
theorem dealer_profit_theorem (strategy : DealerStrategy) 
  (h1 : strategy.markup_percentage = 100)
  (h2 : strategy.discount_percentage = 10)
  (h3 : strategy.bulk_deal_articles_sold = 20)
  (h4 : strategy.bulk_deal_articles_cost = 15) :
  calculate_profit_percentage strategy = 80 := by
  sorry

end dealer_profit_theorem_l1230_123042


namespace display_rows_l1230_123001

/-- The number of cans in the nth row of the display -/
def cans_in_row (n : ℕ) : ℕ := 2 + 3 * (n - 1)

/-- The total number of cans in the first n rows of the display -/
def total_cans (n : ℕ) : ℕ := n * (cans_in_row 1 + cans_in_row n) / 2

/-- The number of rows in the display -/
def num_rows : ℕ := 12

theorem display_rows :
  total_cans num_rows = 225 ∧
  cans_in_row 1 = 2 ∧
  ∀ n : ℕ, n > 1 → cans_in_row n = cans_in_row (n - 1) + 3 :=
sorry

end display_rows_l1230_123001


namespace quadratic_equation_1_l1230_123025

theorem quadratic_equation_1 : 
  ∀ x : ℝ, x^2 - 2*x + 1 = 0 → x = 1 := by
sorry

end quadratic_equation_1_l1230_123025
