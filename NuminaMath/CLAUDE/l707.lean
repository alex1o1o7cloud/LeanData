import Mathlib

namespace percentage_difference_l707_70737

theorem percentage_difference : (70 : ℝ) / 100 * 100 - (60 : ℝ) / 100 * 80 = 22 := by
  sorry

end percentage_difference_l707_70737


namespace intersection_values_l707_70731

/-- The function f(x) = mx² - 6x + 2 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 6 * x + 2

/-- The graph of f intersects the x-axis at only one point -/
def single_intersection (m : ℝ) : Prop :=
  ∃! x, f m x = 0

theorem intersection_values (m : ℝ) :
  single_intersection m → m = 0 ∨ m = 9/2 := by
  sorry

end intersection_values_l707_70731


namespace erica_ride_percentage_longer_l707_70783

-- Define the ride times for Dave, Chuck, and Erica
def dave_ride_time : ℕ := 10
def chuck_ride_time : ℕ := 5 * dave_ride_time
def erica_ride_time : ℕ := 65

-- Define the percentage difference
def percentage_difference : ℚ := (erica_ride_time - chuck_ride_time : ℚ) / chuck_ride_time * 100

-- Theorem statement
theorem erica_ride_percentage_longer :
  percentage_difference = 30 := by sorry

end erica_ride_percentage_longer_l707_70783


namespace circular_bed_circumference_circular_bed_specific_circumference_l707_70771

/-- The circumference of a circular bed containing a given number of plants -/
theorem circular_bed_circumference (num_plants : Real) (area_per_plant : Real) : Real :=
  let total_area := num_plants * area_per_plant
  let radius := (total_area / Real.pi).sqrt
  2 * Real.pi * radius

/-- Proof that the circular bed with given specifications has the expected circumference -/
theorem circular_bed_specific_circumference : 
  ∃ (ε : Real), ε > 0 ∧ ε < 0.000001 ∧ 
  |circular_bed_circumference 22.997889276778874 4 - 34.007194| < ε :=
sorry

end circular_bed_circumference_circular_bed_specific_circumference_l707_70771


namespace bisection_method_for_f_l707_70796

/-- The function f(x) = x^5 + 8x^3 - 1 -/
def f (x : ℝ) : ℝ := x^5 + 8*x^3 - 1

/-- Theorem stating the properties of the bisection method for f(x) -/
theorem bisection_method_for_f :
  f 0 < 0 →
  f 0.5 > 0 →
  ∃ (a b : ℝ), a = 0 ∧ b = 0.5 ∧
    (∃ (x : ℝ), a < x ∧ x < b ∧ f x = 0) ∧
    ((a + b) / 2 = 0.25) :=
sorry

end bisection_method_for_f_l707_70796


namespace inverse_proportionality_example_l707_70711

/-- Definition of inverse proportionality -/
def is_inversely_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, x ≠ 0 → f x = k / x

/-- The function y = 6/x is inversely proportional -/
theorem inverse_proportionality_example :
  is_inversely_proportional (λ x : ℝ => 6 / x) :=
by
  sorry


end inverse_proportionality_example_l707_70711


namespace initial_bananas_per_child_l707_70753

/-- Proves that the initial number of bananas per child is 2 --/
theorem initial_bananas_per_child (total_children : ℕ) (absent_children : ℕ) (extra_bananas : ℕ) : 
  total_children = 320 →
  absent_children = 160 →
  extra_bananas = 2 →
  ∃ (initial_bananas : ℕ), 
    (total_children - absent_children) * (initial_bananas + extra_bananas) = 
    total_children * initial_bananas ∧
    initial_bananas = 2 := by
  sorry

end initial_bananas_per_child_l707_70753


namespace stage_20_toothpicks_l707_70793

/-- Calculates the number of toothpicks in a given stage of the pattern -/
def toothpicks (stage : ℕ) : ℕ :=
  3 + 3 * (stage - 1)

/-- Theorem: The 20th stage of the pattern has 60 toothpicks -/
theorem stage_20_toothpicks : toothpicks 20 = 60 := by
  sorry

end stage_20_toothpicks_l707_70793


namespace supplement_of_complement_35_l707_70748

/-- The complement of an angle in degrees -/
def complement (α : ℝ) : ℝ := 90 - α

/-- The supplement of an angle in degrees -/
def supplement (β : ℝ) : ℝ := 180 - β

/-- The original angle in degrees -/
def original_angle : ℝ := 35

/-- Theorem: The degree measure of the supplement of the complement of a 35-degree angle is 125° -/
theorem supplement_of_complement_35 : 
  supplement (complement original_angle) = 125 := by
  sorry

end supplement_of_complement_35_l707_70748


namespace score_difference_l707_70749

def sammy_score : ℝ := 20

def gab_score : ℝ := 2 * sammy_score

def cher_score : ℝ := 2 * gab_score

def alex_score : ℝ := cher_score * 1.1

def team1_score : ℝ := sammy_score + gab_score + cher_score + alex_score

def opponent_initial_score : ℝ := 85

def opponent_final_score : ℝ := opponent_initial_score * 1.5

theorem score_difference :
  team1_score - opponent_final_score = 100.5 := by
  sorry

end score_difference_l707_70749


namespace trip_time_difference_l707_70732

theorem trip_time_difference (distance1 distance2 speed : ℝ) 
  (h1 : distance1 = 160)
  (h2 : distance2 = 280)
  (h3 : speed = 40)
  : distance2 / speed - distance1 / speed = 3 := by
  sorry

end trip_time_difference_l707_70732


namespace book_selection_combinations_l707_70707

/-- The number of ways to choose one book from each of three genres -/
def book_combinations (mystery_count : ℕ) (fantasy_count : ℕ) (biography_count : ℕ) : ℕ :=
  mystery_count * fantasy_count * biography_count

/-- Theorem: Given 4 mystery novels, 3 fantasy novels, and 3 biographies,
    the number of ways to choose one book from each genre is 36 -/
theorem book_selection_combinations :
  book_combinations 4 3 3 = 36 := by
  sorry

end book_selection_combinations_l707_70707


namespace kenny_lawn_mowing_l707_70746

theorem kenny_lawn_mowing (cost_per_lawn : ℕ) (cost_per_game : ℕ) (cost_per_book : ℕ)
  (num_games : ℕ) (num_books : ℕ) 
  (h1 : cost_per_lawn = 15)
  (h2 : cost_per_game = 45)
  (h3 : cost_per_book = 5)
  (h4 : num_games = 5)
  (h5 : num_books = 60) :
  cost_per_lawn * 35 = cost_per_game * num_games + cost_per_book * num_books :=
by sorry

end kenny_lawn_mowing_l707_70746


namespace fourth_year_exam_count_l707_70716

/-- Represents the number of exams taken in each year -/
structure ExamCount where
  year1 : ℕ
  year2 : ℕ
  year3 : ℕ
  year4 : ℕ
  year5 : ℕ

/-- Conditions for the exam count problem -/
def ValidExamCount (e : ExamCount) : Prop :=
  e.year1 + e.year2 + e.year3 + e.year4 + e.year5 = 31 ∧
  e.year1 < e.year2 ∧ e.year2 < e.year3 ∧ e.year3 < e.year4 ∧ e.year4 < e.year5 ∧
  e.year5 = 3 * e.year1

/-- The theorem stating that if the exam count is valid, the fourth year must have 8 exams -/
theorem fourth_year_exam_count (e : ExamCount) : ValidExamCount e → e.year4 = 8 := by
  sorry

end fourth_year_exam_count_l707_70716


namespace valid_integers_exist_l707_70794

def is_valid_integer (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  (n / 1000 + (n / 100 % 10) + (n / 10 % 10) + (n % 10) = 18) ∧
  ((n / 100 % 10) + (n / 10 % 10) = 11) ∧
  (n / 1000 - n % 10 = 3) ∧
  n % 9 = 0

theorem valid_integers_exist : ∃ n : ℕ, is_valid_integer n :=
sorry

end valid_integers_exist_l707_70794


namespace shenny_vacation_shirts_l707_70715

/-- The number of shirts Shenny needs to pack for her vacation -/
def shirts_to_pack (vacation_days : ℕ) (same_shirt_days : ℕ) (different_shirts_per_day : ℕ) : ℕ :=
  (vacation_days - same_shirt_days) * different_shirts_per_day + 1

/-- Proof that Shenny needs to pack 11 shirts for her vacation -/
theorem shenny_vacation_shirts :
  shirts_to_pack 7 2 2 = 11 := by
  sorry

end shenny_vacation_shirts_l707_70715


namespace quadratic_form_k_value_l707_70705

theorem quadratic_form_k_value :
  ∀ (a h k : ℝ), (∀ x, x^2 - 7*x = a*(x - h)^2 + k) → k = -49/4 := by
  sorry

end quadratic_form_k_value_l707_70705


namespace triangle_problem_l707_70784

theorem triangle_problem (A B C : Real) (a b c : Real) (D : Real × Real) :
  -- Triangle ABC exists
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  -- Sides a, b, c are positive
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given equation
  b * Real.sin (C + π/3) - c * Real.sin B = 0 →
  -- Area condition
  1/2 * a * b * Real.sin C = 10 * Real.sqrt 3 →
  -- D is midpoint of AC
  D.1 = (0 + a * Real.cos B) / 2 ∧ D.2 = (0 + a * Real.sin B) / 2 →
  -- Prove:
  C = π/3 ∧ 
  (∀ (BD : Real), BD^2 ≥ a^2 + b^2/4 - a*b*Real.cos C → BD ≥ 2 * Real.sqrt 5) :=
by sorry

end triangle_problem_l707_70784


namespace pirate_treasure_division_l707_70747

theorem pirate_treasure_division (N : ℕ) : 
  220 ≤ N ∧ N ≤ 300 →
  let first_take := 2 + (N - 2) / 3
  let remain_after_first := N - first_take
  let second_take := 2 + (remain_after_first - 2) / 3
  let remain_after_second := remain_after_first - second_take
  let third_take := 2 + (remain_after_second - 2) / 3
  let final_remain := remain_after_second - third_take
  final_remain % 3 = 0 →
  first_take = 84 ∧ 
  second_take = 54 ∧ 
  third_take = 54 ∧
  final_remain / 3 = 54 := by
sorry


end pirate_treasure_division_l707_70747


namespace dividend_calculation_l707_70734

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17)
  (h2 : quotient = 9)
  (h3 : remainder = 10) :
  divisor * quotient + remainder = 163 := by
  sorry

end dividend_calculation_l707_70734


namespace angle_relations_l707_70787

theorem angle_relations (θ : Real) 
  (h1 : θ ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi)) -- θ is in the fourth quadrant
  (h2 : Real.sin θ + Real.cos θ = 1/5) :
  (Real.sin θ - Real.cos θ = -7/5) ∧ (Real.tan θ = -3/4) := by
  sorry

end angle_relations_l707_70787


namespace parabola_c_value_l707_70743

/-- A parabola passing through two points -/
def Parabola (b c : ℝ) :=
  {f : ℝ → ℝ | ∃ (x : ℝ), f x = x^2 + b*x + c}

/-- The parabola passes through the point (1,4) -/
def passes_through_1_4 (b c : ℝ) : Prop :=
  1^2 + b*1 + c = 4

/-- The parabola passes through the point (5,4) -/
def passes_through_5_4 (b c : ℝ) : Prop :=
  5^2 + b*5 + c = 4

/-- Theorem: For a parabola y = x² + bx + c passing through (1,4) and (5,4), c = 9 -/
theorem parabola_c_value (b c : ℝ) 
  (h1 : passes_through_1_4 b c) 
  (h2 : passes_through_5_4 b c) : 
  c = 9 := by
  sorry

end parabola_c_value_l707_70743


namespace AM_GM_inequality_counterexample_AM_GM_inequality_l707_70712

theorem AM_GM_inequality_counterexample : ¬ ∀ x : ℝ, x + 1/x ≥ 2 * Real.sqrt (x * (1/x)) :=
by
  sorry

theorem AM_GM_inequality {a b : ℝ} (ha : a > 0) (hb : b > 0) : a + b ≥ 2 * Real.sqrt (a * b) :=
by
  sorry

end AM_GM_inequality_counterexample_AM_GM_inequality_l707_70712


namespace library_schedule_l707_70781

theorem library_schedule (sam fran mike julio : ℕ) 
  (h_sam : sam = 5)
  (h_fran : fran = 8)
  (h_mike : mike = 10)
  (h_julio : julio = 12) :
  Nat.lcm (Nat.lcm (Nat.lcm sam fran) mike) julio = 120 := by
  sorry

end library_schedule_l707_70781


namespace inequality_proof_l707_70769

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a^2 / (b + c)) + (b^2 / (c + a)) + (c^2 / (a + b)) ≥ 3/2 := by
  sorry

end inequality_proof_l707_70769


namespace archipelago_islands_l707_70761

theorem archipelago_islands (n : ℕ) : 
  (n * (n - 1)) / 2 + n = 28 →
  n + 1 = 8 :=
by
  sorry

#check archipelago_islands

end archipelago_islands_l707_70761


namespace freddys_age_l707_70700

theorem freddys_age (job_age stephanie_age freddy_age : ℕ) : 
  job_age = 5 →
  stephanie_age = 4 * job_age →
  freddy_age = stephanie_age - 2 →
  freddy_age = 18 := by
  sorry

end freddys_age_l707_70700


namespace min_k_for_triangle_inequality_l707_70722

theorem min_k_for_triangle_inequality : 
  ∃ (k : ℕ), k > 0 ∧ 
  (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → 
    k * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) → 
    (a + b > c ∧ b + c > a ∧ c + a > b)) ∧
  (∀ (k' : ℕ), k' > 0 → k' < k → 
    ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    k' * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) ∧
    ¬(a + b > c ∧ b + c > a ∧ c + a > b)) ∧
  k = 6 :=
sorry

end min_k_for_triangle_inequality_l707_70722


namespace linear_function_intersection_l707_70724

-- Define the linear function
def f (k : ℝ) (x : ℝ) : ℝ := k * x + 3

-- Define the theorem
theorem linear_function_intersection (k : ℝ) :
  (∃ t : ℝ, t > 0 ∧ f k t = 0) →  -- x-axis intersection exists and is positive
  (f k 0 = 3) →  -- y-axis intersection is (0, 3)
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f k x₁ > f k x₂) →  -- y decreases as x increases
  (∃ t : ℝ, t > 0 ∧ f k t = 0 ∧ t^2 + 3^2 = 5^2) →  -- distance between intersections is 5
  k = -3/4 := by
sorry

end linear_function_intersection_l707_70724


namespace smallest_integer_for_quadratic_inequality_l707_70736

theorem smallest_integer_for_quadratic_inequality :
  ∃ n : ℤ, (∀ m : ℤ, m^2 - 13*m + 40 ≤ 0 → n ≤ m) ∧ (n^2 - 13*n + 40 ≤ 0) ∧ n = 5 := by
  sorry

end smallest_integer_for_quadratic_inequality_l707_70736


namespace inequality_proof_l707_70735

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_xz : x * z = 1)
  (h_x_1z : x * (1 + z) > 1)
  (h_y_1x : y * (1 + x) > 1)
  (h_z_1y : z * (1 + y) > 1) :
  2 * (x + y + z) ≥ -1/x + 1/y + 1/z + 3 := by
sorry


end inequality_proof_l707_70735


namespace perpendicular_vectors_l707_70717

/-- Given planar vectors a and b, prove that m = 1 makes ma + b perpendicular to a -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h1 : a = (-1, 3)) (h2 : b = (4, -2)) :
  ∃ m : ℝ, m = 1 ∧ (m • a + b) • a = 0 :=
by sorry

end perpendicular_vectors_l707_70717


namespace valid_numbers_l707_70763

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧  -- three-digit number
  (n / 100 = n % 10) ∧  -- hundreds and units digits are the same
  n % 15 = 0            -- divisible by 15

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {525, 555, 585} := by sorry

end valid_numbers_l707_70763


namespace group_interval_equals_frequency_over_height_l707_70738

/-- Given a group [a, b] in a sampling process with frequency m and histogram height h, 
    prove that the group interval |a-b| equals m/h -/
theorem group_interval_equals_frequency_over_height 
  (a b m h : ℝ) (hm : m > 0) (hh : h > 0) : |a - b| = m / h := by
  sorry

end group_interval_equals_frequency_over_height_l707_70738


namespace coffee_mix_price_l707_70760

/-- The price of the first kind of coffee in dollars per pound -/
def price_first : ℚ := 215 / 100

/-- The price of the mixed coffee in dollars per pound -/
def price_mix : ℚ := 230 / 100

/-- The total weight of the mixed coffee in pounds -/
def total_weight : ℚ := 18

/-- The weight of each kind of coffee in the mix in pounds -/
def weight_each : ℚ := 9

/-- The price of the second kind of coffee in dollars per pound -/
def price_second : ℚ := 245 / 100

theorem coffee_mix_price :
  price_second = 
    (price_mix * total_weight - price_first * weight_each) / weight_each :=
by sorry

end coffee_mix_price_l707_70760


namespace division_problem_l707_70762

theorem division_problem (L S Q : ℕ) : 
  L - S = 1365 → 
  L = 1636 → 
  L = Q * S + 10 → 
  Q = 6 := by sorry

end division_problem_l707_70762


namespace pencil_count_l707_70750

-- Define the number of items in the pencil case
def total_items : ℕ := 13

-- Define the relationship between pens and pencils
def pen_pencil_relation (pencils : ℕ) : ℕ := 2 * pencils

-- Define the number of erasers
def erasers : ℕ := 1

-- Theorem statement
theorem pencil_count : 
  ∃ (pencils : ℕ), 
    pencils + pen_pencil_relation pencils + erasers = total_items ∧ 
    pencils = 4 := by
  sorry

end pencil_count_l707_70750


namespace radio_show_song_time_l707_70788

/-- Calculates the time spent on songs in a radio show -/
theorem radio_show_song_time (total_show_time : ℕ) (talking_segment_duration : ℕ) 
  (ad_break_duration : ℕ) (num_talking_segments : ℕ) (num_ad_breaks : ℕ) :
  total_show_time = 3 * 60 →
  talking_segment_duration = 10 →
  ad_break_duration = 5 →
  num_talking_segments = 3 →
  num_ad_breaks = 5 →
  total_show_time - (num_talking_segments * talking_segment_duration + num_ad_breaks * ad_break_duration) = 125 := by
  sorry

end radio_show_song_time_l707_70788


namespace registration_scientific_correct_l707_70766

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number of people registered for the national college entrance examination in 2023 -/
def registration_number : ℕ := 12910000

/-- The scientific notation representation of the registration number -/
def registration_scientific : ScientificNotation :=
  { coefficient := 1.291,
    exponent := 7,
    is_valid := by sorry }

/-- Theorem stating that the registration number is correctly represented in scientific notation -/
theorem registration_scientific_correct :
  (registration_scientific.coefficient * (10 : ℝ) ^ registration_scientific.exponent) = registration_number := by
  sorry

end registration_scientific_correct_l707_70766


namespace circle_translation_l707_70797

-- Define the original circle
def original_circle (x y : ℝ) : Prop := x^2 + y^2 = 16

-- Define the translation
def translation : ℝ × ℝ := (-5, -3)

-- Define the translated circle
def translated_circle (x y : ℝ) : Prop := (x+5)^2 + (y+3)^2 = 16

-- Theorem statement
theorem circle_translation :
  ∀ (x y : ℝ), original_circle (x + 5) (y + 3) ↔ translated_circle x y :=
by sorry

end circle_translation_l707_70797


namespace students_left_on_bus_l707_70777

def initial_students : ℕ := 10
def students_who_left : ℕ := 3

theorem students_left_on_bus : initial_students - students_who_left = 7 := by
  sorry

end students_left_on_bus_l707_70777


namespace number_equation_solution_l707_70778

theorem number_equation_solution : ∃ x : ℝ, x + x + 2*x + 4*x = 104 ∧ x = 13 := by
  sorry

end number_equation_solution_l707_70778


namespace paint_brush_square_ratio_l707_70713

/-- Given a square with side length s and a paint brush of width w that sweeps along both diagonals,
    if half the area of the square is painted, then the ratio of the square's diagonal length to the brush width is 2√2 + 2. -/
theorem paint_brush_square_ratio (s w : ℝ) (h_positive : s > 0 ∧ w > 0) 
  (h_half_painted : w^2 + (s - w)^2 / 2 = s^2 / 2) : 
  s * Real.sqrt 2 / w = 2 * Real.sqrt 2 + 2 := by
  sorry

end paint_brush_square_ratio_l707_70713


namespace integer_root_count_l707_70772

theorem integer_root_count : ∃! (S : Finset ℝ), 
  (∀ x ∈ S, ∃ k : ℤ, Real.sqrt (123 - Real.sqrt x) = k) ∧ 
  (∀ x : ℝ, (∃ k : ℤ, Real.sqrt (123 - Real.sqrt x) = k) → x ∈ S) ∧ 
  Finset.card S = 12 := by sorry

end integer_root_count_l707_70772


namespace eggs_taken_l707_70757

theorem eggs_taken (initial : ℕ) (remaining : ℕ) (taken : ℕ) : 
  initial = 47 → remaining = 42 → taken = initial - remaining → taken = 5 := by
  sorry

end eggs_taken_l707_70757


namespace twenty_seven_in_base_two_l707_70795

theorem twenty_seven_in_base_two : 
  27 = 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 := by
  sorry

end twenty_seven_in_base_two_l707_70795


namespace simplify_expression_l707_70741

theorem simplify_expression : 
  (18 * 10^9 - 6 * 10^9) / (6 * 10^4 + 3 * 10^4) = 400000 / 3 := by
  sorry

end simplify_expression_l707_70741


namespace product_of_sums_inequality_l707_70765

theorem product_of_sums_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c :=
sorry

end product_of_sums_inequality_l707_70765


namespace min_perimeter_of_divided_rectangle_l707_70720

/-- Represents the side lengths of the two main squares in the rectangle -/
structure MainSquares where
  a : ℕ
  b : ℕ

/-- Represents the dimensions of the rectangle -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the perimeter of a rectangle -/
def perimeter (rect : Rectangle) : ℕ :=
  2 * (rect.width + rect.height)

/-- Checks if the given main square side lengths satisfy the rectangle division conditions -/
def satisfiesConditions (squares : MainSquares) : Prop :=
  5 * squares.a + 2 * squares.b = 20 * squares.a - 3 * squares.b

/-- Calculates the rectangle dimensions from the main square side lengths -/
def calculateRectangle (squares : MainSquares) : Rectangle :=
  { width := 2 * squares.a + 2 * squares.b
  , height := 3 * squares.a + 2 * squares.b }

theorem min_perimeter_of_divided_rectangle :
  ∃ (squares : MainSquares),
    satisfiesConditions squares ∧
    ∀ (other : MainSquares),
      satisfiesConditions other →
      perimeter (calculateRectangle squares) ≤ perimeter (calculateRectangle other) ∧
      perimeter (calculateRectangle squares) = 52 :=
sorry

end min_perimeter_of_divided_rectangle_l707_70720


namespace min_value_a_l707_70725

theorem min_value_a (a x y : ℤ) (h1 : x - y^2 = a) (h2 : y - x^2 = a) (h3 : x ≠ y) (h4 : |x| ≤ 10) :
  ∃ (a_min : ℤ), a ≥ a_min ∧ a_min = -111 :=
by sorry

end min_value_a_l707_70725


namespace equality_condition_l707_70727

theorem equality_condition (a b c : ℝ) : 
  a + 2*b*c = (a + 2*b)*(a + 2*c) ↔ a + 2*b + 2*c = 0 := by
  sorry

end equality_condition_l707_70727


namespace team_selection_count_l707_70745

theorem team_selection_count (total : ℕ) (veterans : ℕ) (new : ℕ) (team_size : ℕ) (max_veterans : ℕ) :
  total = veterans + new →
  total = 10 →
  veterans = 2 →
  new = 8 →
  team_size = 3 →
  max_veterans = 1 →
  Nat.choose (new - 1) team_size + veterans * Nat.choose (new - 1) (team_size - 1) = 77 :=
by sorry

end team_selection_count_l707_70745


namespace tangent_line_y_intercept_l707_70792

open Real

noncomputable def f (x : ℝ) := exp x + exp (-x)

theorem tangent_line_y_intercept :
  let x₀ : ℝ := log (sqrt 2)
  let f' : ℝ → ℝ := λ x => exp x - exp (-x)
  let m : ℝ := f' x₀
  let b : ℝ := f x₀ - m * x₀
  (∀ x, f' (-x) = -f' x) →  -- f' is an odd function
  (m * (sqrt 2) / 2 = -1) →  -- tangent line is perpendicular to √2x + y + 1 = 0
  b = 3 * sqrt 2 / 2 - sqrt 2 / 4 * log 2 :=
by sorry

end tangent_line_y_intercept_l707_70792


namespace sqrt_equation_solution_l707_70752

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 7) + 5 = 14 → x = 74 := by
  sorry

end sqrt_equation_solution_l707_70752


namespace distance_AC_proof_l707_70764

/-- The distance between two cities A and C, given specific travel conditions. -/
def distance_AC : ℝ := 17.5

/-- The speed of the truck in km/h. -/
def truck_speed : ℝ := 50

/-- The distance traveled by delivery person A before meeting the truck, in km. -/
def distance_A_meeting : ℝ := 3

/-- The time between the meeting point and arrival at C, in hours. -/
def time_after_meeting : ℝ := 0.2  -- 12 minutes = 0.2 hours

/-- Theorem stating the distance between cities A and C under given conditions. -/
theorem distance_AC_proof :
  ∃ (speed_delivery : ℝ),
    speed_delivery > 0 ∧
    distance_AC = truck_speed * (time_after_meeting + distance_A_meeting / truck_speed) :=
by sorry


end distance_AC_proof_l707_70764


namespace largest_integer_inequality_l707_70714

theorem largest_integer_inequality :
  ∃ (n : ℕ), (∀ (m : ℕ), (1/4 : ℚ) + (m : ℚ)/8 < 7/8 → m ≤ n) ∧
             ((1/4 : ℚ) + (n : ℚ)/8 < 7/8) ∧ n = 4 := by
  sorry

end largest_integer_inequality_l707_70714


namespace ferris_wheel_seats_l707_70755

theorem ferris_wheel_seats (people_per_seat : ℕ) (total_people : ℕ) (h1 : people_per_seat = 6) (h2 : total_people = 84) :
  total_people / people_per_seat = 14 := by
  sorry

end ferris_wheel_seats_l707_70755


namespace bales_equation_initial_bales_count_l707_70790

/-- The initial number of bales in the barn -/
def initial_bales : ℕ := sorry

/-- The number of bales added to the barn -/
def added_bales : ℕ := 35

/-- The final number of bales in the barn -/
def final_bales : ℕ := 82

/-- Theorem stating that the initial number of bales plus the added bales equals the final number of bales -/
theorem bales_equation : initial_bales + added_bales = final_bales := by sorry

/-- Theorem proving that the initial number of bales was 47 -/
theorem initial_bales_count : initial_bales = 47 := by sorry

end bales_equation_initial_bales_count_l707_70790


namespace triangle_side_length_l707_70774

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_side_length (t : Triangle) : 
  t.B = π / 3 ∧ 
  t.b = 6 ∧ 
  Real.sin t.A - 2 * Real.sin t.C = 0 → 
  t.a = 4 * Real.sqrt 3 := by
  sorry


end triangle_side_length_l707_70774


namespace smallest_positive_root_l707_70739

noncomputable def α : Real := Real.arctan (2 / 9)
noncomputable def β : Real := Real.arctan (6 / 7)

def equation (x : Real) : Prop :=
  2 * Real.sin (6 * x) + 9 * Real.cos (6 * x) = 6 * Real.sin (2 * x) + 7 * Real.cos (2 * x)

theorem smallest_positive_root :
  ∃ (x : Real), x > 0 ∧ equation x ∧ ∀ (y : Real), y > 0 ∧ equation y → x ≤ y :=
by
  sorry

end smallest_positive_root_l707_70739


namespace equal_discriminants_l707_70785

theorem equal_discriminants (p1 p2 q1 q2 a1 a2 b1 b2 : ℝ) 
  (hP : ∀ x, x^2 + p1*x + q1 = (x - a1)*(x - a2))
  (hQ : ∀ x, x^2 + p2*x + q2 = (x - b1)*(x - b2))
  (ha : a1 ≠ a2)
  (hb : b1 ≠ b2)
  (h_eq : (b1^2 + p1*b1 + q1) + (b2^2 + p1*b2 + q1) = 
          (a1^2 + p2*a1 + q2) + (a2^2 + p2*a2 + q2)) :
  (a1 - a2)^2 = (b1 - b2)^2 := by
  sorry

end equal_discriminants_l707_70785


namespace like_terms_characterization_l707_70702

/-- Represents a term in an algebraic expression -/
structure Term where
  letters : List Char
  exponents : List Nat
  deriving Repr

/-- Defines when two terms are considered like terms -/
def like_terms (t1 t2 : Term) : Prop :=
  t1.letters = t2.letters ∧ t1.exponents = t2.exponents

theorem like_terms_characterization (t1 t2 : Term) :
  like_terms t1 t2 ↔ t1.letters = t2.letters ∧ t1.exponents = t2.exponents :=
by sorry

end like_terms_characterization_l707_70702


namespace unique_three_digit_even_with_digit_sum_26_l707_70718

/-- The digit sum of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Checks if a natural number is a 3-digit number -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- The set of 3-digit even numbers with digit sum 26 -/
def S : Set ℕ := {n : ℕ | is_three_digit n ∧ Even n ∧ digit_sum n = 26}

theorem unique_three_digit_even_with_digit_sum_26 : ∃! n, n ∈ S := by sorry

end unique_three_digit_even_with_digit_sum_26_l707_70718


namespace all_circles_contain_common_point_l707_70775

/-- A parabola of the form y = x² + 2px + q -/
structure Parabola where
  p : ℝ
  q : ℝ

/-- The circle passing through the intersection points of a parabola with the coordinate axes -/
def circle_through_intersections (par : Parabola) : Set (ℝ × ℝ) :=
  {(x, y) | (x + par.p)^2 + (y - par.q/2)^2 = par.p^2 + par.q^2/4}

/-- Predicate to check if a parabola intersects the coordinate axes in three distinct points -/
def has_three_distinct_intersections (par : Parabola) : Prop :=
  par.p^2 > par.q ∧ par.q ≠ 0

theorem all_circles_contain_common_point :
  ∀ (par : Parabola), has_three_distinct_intersections par →
  (0, 1) ∈ circle_through_intersections par :=
sorry

end all_circles_contain_common_point_l707_70775


namespace cistern_fill_time_l707_70751

/-- Represents a tap that can fill or empty a cistern -/
structure Tap where
  rate : ℚ  -- Rate at which the tap fills (positive) or empties (negative) the cistern per hour

/-- Calculates the time to fill a cistern given a list of taps -/
def timeTofill (taps : List Tap) : ℚ :=
  1 / (taps.map (λ t => t.rate) |>.sum)

theorem cistern_fill_time (tapA tapB tapC : Tap)
  (hA : tapA.rate = 1/3)
  (hB : tapB.rate = -1/6)
  (hC : tapC.rate = 1/2) :
  timeTofill [tapA, tapB, tapC] = 3/2 := by
  sorry

#eval timeTofill [{ rate := 1/3 }, { rate := -1/6 }, { rate := 1/2 }]

end cistern_fill_time_l707_70751


namespace farm_feet_count_l707_70701

/-- Given a farm with hens and cows, calculates the total number of feet -/
def total_feet (total_heads : ℕ) (num_hens : ℕ) : ℕ :=
  let num_cows := total_heads - num_hens
  let hen_feet := num_hens * 2
  let cow_feet := num_cows * 4
  hen_feet + cow_feet

/-- Theorem: In a farm with 46 total heads and 24 hens, there are 136 feet in total -/
theorem farm_feet_count : total_feet 46 24 = 136 := by
  sorry

end farm_feet_count_l707_70701


namespace mingis_test_pages_l707_70706

/-- The number of pages in Mingi's math test -/
def pages_in_test (first_page last_page : ℕ) : ℕ :=
  last_page - first_page + 1

/-- Theorem stating the number of pages in Mingi's math test -/
theorem mingis_test_pages : pages_in_test 8 21 = 14 := by
  sorry

end mingis_test_pages_l707_70706


namespace right_triangle_area_l707_70729

theorem right_triangle_area (hypotenuse : ℝ) (angle : ℝ) : 
  hypotenuse = 10 * Real.sqrt 2 →
  angle = 45 →
  (1 / 2) * (hypotenuse / Real.sqrt 2) * (hypotenuse / Real.sqrt 2) = 50 := by
  sorry

end right_triangle_area_l707_70729


namespace min_length_3rd_order_repeatable_last_term_value_l707_70733

/-- Definition of a kth-order repeatable sequence -/
def is_kth_order_repeatable (a : ℕ → Fin 2) (m k : ℕ) : Prop :=
  ∃ i j, 1 ≤ i ∧ i + k - 1 ≤ m ∧ 1 ≤ j ∧ j + k - 1 ≤ m ∧ i ≠ j ∧
  ∀ t, 0 ≤ t ∧ t < k → a (i + t) = a (j + t)

theorem min_length_3rd_order_repeatable :
  ∀ m : ℕ, m ≥ 3 →
  ((∀ a : ℕ → Fin 2, is_kth_order_repeatable a m 3) ↔ m ≥ 11) :=
sorry

theorem last_term_value (a : ℕ → Fin 2) (m : ℕ) :
  m ≥ 3 →
  a 4 ≠ 1 →
  (¬ is_kth_order_repeatable a m 5) →
  (∃ b : Fin 2, is_kth_order_repeatable (Function.update a (m + 1) b) (m + 1) 5) →
  a m = 0 :=
sorry

end min_length_3rd_order_repeatable_last_term_value_l707_70733


namespace isosceles_right_triangle_roots_l707_70744

/-- 
Given complex numbers a and b, this theorem states that a^2 = 2b ≠ 0 
if and only if the roots of the polynomial x^2 + ax + b form an isosceles 
right triangle on the complex plane with the right angle at the origin.
-/
theorem isosceles_right_triangle_roots 
  (a b : ℂ) : a^2 = 2*b ∧ b ≠ 0 ↔ 
  ∃ (x₁ x₂ : ℂ), x₁^2 + a*x₁ + b = 0 ∧ 
                 x₂^2 + a*x₂ + b = 0 ∧ 
                 x₁ ≠ x₂ ∧
                 (x₁ = Complex.I * x₂ ∨ x₂ = Complex.I * x₁) :=
by sorry

end isosceles_right_triangle_roots_l707_70744


namespace parabola_directrix_l707_70756

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop :=
  y = (x^2 - 6*x + 5) / 12

/-- The directrix equation -/
def directrix (y : ℝ) : Prop :=
  y = -10/3

/-- Theorem stating that the given directrix is correct for the parabola -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola x y → ∃ d : ℝ, directrix d ∧ 
  (∀ p : ℝ × ℝ, p.1 = x ∧ p.2 = y → 
    ∃ f : ℝ × ℝ, ∃ q : ℝ × ℝ, 
      q.2 = d ∧ 
      (p.1 - f.1)^2 + (p.2 - f.2)^2 = (p.1 - q.1)^2 + (p.2 - q.2)^2) :=
by sorry

end parabola_directrix_l707_70756


namespace factorization_valid_l707_70754

theorem factorization_valid (x : ℝ) : 10 * x^2 - 5 * x = 5 * x * (2 * x - 1) := by
  sorry

end factorization_valid_l707_70754


namespace set_A_equals_explicit_set_l707_70768

def set_A : Set (ℤ × ℤ) :=
  {p | p.1^2 = p.2 + 1 ∧ |p.1| < 2}

theorem set_A_equals_explicit_set : 
  set_A = {(-1, 0), (0, -1), (1, 0)} := by sorry

end set_A_equals_explicit_set_l707_70768


namespace transportation_theorem_l707_70726

/-- Represents the capacity and cost of vehicles --/
structure VehicleInfo where
  typeA_capacity : ℝ
  typeB_capacity : ℝ
  typeA_cost : ℝ
  typeB_cost : ℝ

/-- Represents the transportation problem --/
structure TransportationProblem where
  info : VehicleInfo
  total_vehicles : ℕ
  min_transport : ℝ
  max_cost : ℝ

/-- Solves the transportation problem --/
def solve_transportation (p : TransportationProblem) :
  (ℝ × ℝ) × ℕ × (ℕ × ℕ × ℝ) :=
sorry

/-- The main theorem --/
theorem transportation_theorem (p : TransportationProblem) :
  let vi := VehicleInfo.mk 50 40 3000 2000
  let tp := TransportationProblem.mk vi 20 955 58800
  let ((typeA_cap, typeB_cap), min_typeA, (opt_typeA, opt_typeB, min_cost)) := solve_transportation tp
  typeA_cap = 50 ∧ 
  typeB_cap = 40 ∧ 
  min_typeA = 16 ∧ 
  opt_typeA = 16 ∧ 
  opt_typeB = 4 ∧ 
  min_cost = 56000 ∧
  5 * typeA_cap + 3 * typeB_cap = 370 ∧
  4 * typeA_cap + 7 * typeB_cap = 480 ∧
  opt_typeA + opt_typeB = p.total_vehicles ∧
  opt_typeA * typeA_cap + opt_typeB * typeB_cap ≥ p.min_transport ∧
  opt_typeA * p.info.typeA_cost + opt_typeB * p.info.typeB_cost ≤ p.max_cost :=
by sorry


end transportation_theorem_l707_70726


namespace divisibility_of_difference_quotient_l707_70728

theorem divisibility_of_difference_quotient (a b n : ℝ) 
  (ha : a > 0) (hb : b > 0) (hn : n > 0) 
  (h_div : ∃ k : ℤ, a^n - b^n = n * k) :
  ∃ m : ℤ, (a^n - b^n) / (a - b) = n * m := by
sorry

end divisibility_of_difference_quotient_l707_70728


namespace max_value_theorem_range_of_a_l707_70703

-- Define the constraint function
def constraint (x y z : ℝ) : Prop := x^2 + y^2 + z^2 = 1

-- Define the objective function
def objective (x y z : ℝ) : ℝ := x + 2*y + 2*z

-- Theorem 1: Maximum value of the objective function
theorem max_value_theorem (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_constraint : constraint x y z) :
  objective x y z ≤ 3 :=
sorry

-- Theorem 2: Range of a
theorem range_of_a (a : ℝ) :
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → constraint x y z → |a - 3| ≥ objective x y z) ↔
  a ≤ 0 ∨ a ≥ 6 :=
sorry

end max_value_theorem_range_of_a_l707_70703


namespace gcd_12012_18018_l707_70798

theorem gcd_12012_18018 : Nat.gcd 12012 18018 = 6006 := by
  sorry

end gcd_12012_18018_l707_70798


namespace movements_correctly_classified_l707_70799

-- Define an enumeration for movement types
inductive MovementType
  | Translation
  | Rotation

-- Define a structure for a movement
structure Movement where
  description : String
  classification : MovementType

-- Define the list of movements
def movements : List Movement := [
  { description := "Xiaoming walking forward 3 meters", classification := MovementType.Translation },
  { description := "Rocket launching into the sky", classification := MovementType.Translation },
  { description := "Car wheels constantly rotating", classification := MovementType.Rotation },
  { description := "Archer shooting an arrow onto the target", classification := MovementType.Translation }
]

-- Theorem statement
theorem movements_correctly_classified :
  movements.map (λ m => m.classification) = 
    [MovementType.Translation, MovementType.Translation, MovementType.Rotation, MovementType.Translation] := by
  sorry


end movements_correctly_classified_l707_70799


namespace problem_solution_l707_70767

theorem problem_solution (a b : ℝ) (h1 : a - b = 4) (h2 : a * b = 3) :
  (a^2 + b^2 = 22) ∧ ((a - 2) * (b + 2) = 7) := by
  sorry

end problem_solution_l707_70767


namespace lily_remaining_milk_l707_70721

theorem lily_remaining_milk (initial_milk : ℚ) (james_milk : ℚ) (maria_milk : ℚ) :
  initial_milk = 5 →
  james_milk = 15 / 4 →
  maria_milk = 3 / 4 →
  initial_milk - (james_milk + maria_milk) = 1 / 2 := by
  sorry

end lily_remaining_milk_l707_70721


namespace blake_receives_four_dollars_change_l707_70759

/-- The amount of change Blake receives after purchasing lollipops and chocolate. -/
def blakes_change (lollipop_count : ℕ) (chocolate_pack_count : ℕ) (lollipop_price : ℕ) (bill_count : ℕ) (bill_value : ℕ) : ℕ :=
  let chocolate_pack_price := 4 * lollipop_price
  let total_cost := lollipop_count * lollipop_price + chocolate_pack_count * chocolate_pack_price
  let payment := bill_count * bill_value
  payment - total_cost

/-- Theorem stating that Blake's change is $4 given the problem conditions. -/
theorem blake_receives_four_dollars_change :
  blakes_change 4 6 2 6 10 = 4 := by
  sorry

end blake_receives_four_dollars_change_l707_70759


namespace sufficient_condition_range_l707_70770

theorem sufficient_condition_range (a : ℝ) : 
  (∀ x : ℝ, x = 1 → x > a) → a < 1 := by sorry

end sufficient_condition_range_l707_70770


namespace right_angled_triangle_not_axisymmetric_l707_70708

-- Define the types of geometric figures
inductive GeometricFigure
  | Angle
  | EquilateralTriangle
  | LineSegment
  | RightAngledTriangle

-- Define the property of being axisymmetric
def isAxisymmetric : GeometricFigure → Prop :=
  fun figure =>
    match figure with
    | GeometricFigure.Angle => true
    | GeometricFigure.EquilateralTriangle => true
    | GeometricFigure.LineSegment => true
    | GeometricFigure.RightAngledTriangle => false

-- Theorem statement
theorem right_angled_triangle_not_axisymmetric :
  ∀ (figure : GeometricFigure),
    ¬(isAxisymmetric figure) ↔ figure = GeometricFigure.RightAngledTriangle :=
by
  sorry

end right_angled_triangle_not_axisymmetric_l707_70708


namespace bobs_sandwich_cost_l707_70740

/-- Proves that the cost of each of Bob's sandwiches after discount and before tax is $2.412 -/
theorem bobs_sandwich_cost 
  (andy_soda : ℝ) (andy_hamburger : ℝ) (andy_chips : ℝ) (andy_tax_rate : ℝ)
  (bob_sandwich_before_discount : ℝ) (bob_sandwich_count : ℕ) (bob_water : ℝ)
  (bob_sandwich_discount_rate : ℝ) (bob_water_tax_rate : ℝ)
  (h_andy_soda : andy_soda = 1.50)
  (h_andy_hamburger : andy_hamburger = 2.75)
  (h_andy_chips : andy_chips = 1.25)
  (h_andy_tax_rate : andy_tax_rate = 0.08)
  (h_bob_sandwich_before_discount : bob_sandwich_before_discount = 2.68)
  (h_bob_sandwich_count : bob_sandwich_count = 5)
  (h_bob_water : bob_water = 1.25)
  (h_bob_sandwich_discount_rate : bob_sandwich_discount_rate = 0.10)
  (h_bob_water_tax_rate : bob_water_tax_rate = 0.07)
  (h_equal_total : 
    (andy_soda + 3 * andy_hamburger + andy_chips) * (1 + andy_tax_rate) = 
    bob_sandwich_count * bob_sandwich_before_discount * (1 - bob_sandwich_discount_rate) + 
    bob_water * (1 + bob_water_tax_rate)) :
  bob_sandwich_before_discount * (1 - bob_sandwich_discount_rate) = 2.412 := by
  sorry


end bobs_sandwich_cost_l707_70740


namespace fifteenth_student_age_l707_70742

theorem fifteenth_student_age 
  (total_students : ℕ) 
  (avg_age_all : ℝ) 
  (num_group1 : ℕ) 
  (avg_age_group1 : ℝ) 
  (num_group2 : ℕ) 
  (avg_age_group2 : ℝ) 
  (h1 : total_students = 15)
  (h2 : avg_age_all = 15)
  (h3 : num_group1 = 5)
  (h4 : avg_age_group1 = 13)
  (h5 : num_group2 = 9)
  (h6 : avg_age_group2 = 16)
  (h7 : num_group1 + num_group2 + 1 = total_students) :
  (total_students : ℝ) * avg_age_all - 
  ((num_group1 : ℝ) * avg_age_group1 + (num_group2 : ℝ) * avg_age_group2) = 16 := by
  sorry

end fifteenth_student_age_l707_70742


namespace a3_value_l707_70719

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) :=
  b ^ 2 = a * c

theorem a3_value (a : ℕ → ℝ) :
  arithmetic_sequence a 2 →
  geometric_sequence (a 1) (a 3) (a 4) →
  a 3 = -4 :=
by sorry

end a3_value_l707_70719


namespace f_odd_and_increasing_l707_70780

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * abs x

-- State the theorem
theorem f_odd_and_increasing : 
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) :=
sorry

end f_odd_and_increasing_l707_70780


namespace price_per_pack_is_one_l707_70730

/-- Represents the number of boxes in a carton -/
def boxes_per_carton : ℕ := 12

/-- Represents the number of packs of cheese cookies in a box -/
def packs_per_box : ℕ := 10

/-- Represents the cost of a dozen cartons in dollars -/
def cost_dozen_cartons : ℕ := 1440

/-- Represents the number of cartons in a dozen -/
def cartons_in_dozen : ℕ := 12

/-- Theorem stating that the price of a pack of cheese cookies is $1 -/
theorem price_per_pack_is_one :
  (cost_dozen_cartons : ℚ) / (cartons_in_dozen * boxes_per_carton * packs_per_box) = 1 := by
  sorry

end price_per_pack_is_one_l707_70730


namespace damaged_tins_percentage_l707_70710

theorem damaged_tins_percentage (cases : ℕ) (tins_per_case : ℕ) (remaining_tins : ℕ) : 
  cases = 15 → tins_per_case = 24 → remaining_tins = 342 →
  (cases * tins_per_case - remaining_tins) / (cases * tins_per_case) * 100 = 5 := by
  sorry

end damaged_tins_percentage_l707_70710


namespace bags_difference_l707_70776

/-- The number of bags Tiffany had on Monday -/
def monday_bags : ℕ := 7

/-- The number of bags Tiffany found on the next day -/
def next_day_bags : ℕ := 12

/-- Theorem: The difference between the number of bags found on the next day
    and the number of bags on Monday is equal to 5 -/
theorem bags_difference : next_day_bags - monday_bags = 5 := by
  sorry

end bags_difference_l707_70776


namespace elder_sister_savings_l707_70786

theorem elder_sister_savings (total : ℝ) (elder_donation_rate : ℝ) (younger_donation_rate : ℝ)
  (h_total : total = 108)
  (h_elder_rate : elder_donation_rate = 0.75)
  (h_younger_rate : younger_donation_rate = 0.8)
  (h_equal_remainder : ∃ (elder younger : ℝ), 
    elder + younger = total ∧ 
    elder * (1 - elder_donation_rate) = younger * (1 - younger_donation_rate)) :
  ∃ (elder : ℝ), elder = 48 ∧ 
    ∃ (younger : ℝ), younger = total - elder ∧
    elder * (1 - elder_donation_rate) = younger * (1 - younger_donation_rate) := by
  sorry

end elder_sister_savings_l707_70786


namespace tan_sum_special_angle_l707_70782

theorem tan_sum_special_angle (θ : Real) (h : Real.tan θ = 1/3) :
  Real.tan (θ + π/4) = 2 := by sorry

end tan_sum_special_angle_l707_70782


namespace smallest_factorizable_b_l707_70791

theorem smallest_factorizable_b : ∃ (b : ℕ), 
  (∀ (x : ℤ), ∃ (p q : ℤ), x^2 + b*x + 2520 = (x + p) * (x + q)) ∧
  (∀ (b' : ℕ), b' < b → ¬∃ (p q : ℤ), ∀ (x : ℤ), x^2 + b'*x + 2520 = (x + p) * (x + q)) ∧
  b = 106 :=
sorry

end smallest_factorizable_b_l707_70791


namespace unshaded_area_between_circles_l707_70758

/-- The area of the unshaded region between two concentric circles -/
theorem unshaded_area_between_circles (r₁ r₂ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 7) :
  π * r₂^2 - π * r₁^2 = 33 * π :=
by sorry

end unshaded_area_between_circles_l707_70758


namespace unique_solution_abs_equation_l707_70709

theorem unique_solution_abs_equation :
  ∃! x : ℝ, |x - 10| + |x - 14| = |3*x - 42| :=
by
  -- The proof goes here
  sorry

end unique_solution_abs_equation_l707_70709


namespace teahouse_on_tuesday_or_thursday_not_all_plays_on_tuesday_heavenly_sound_not_on_wednesday_thunderstorm_not_only_on_tuesday_l707_70779

-- Define the days of the week
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday

-- Define the plays
inductive Play
| Thunderstorm
| Teahouse
| HeavenlySound
| ShatteredHoofbeats

def Schedule := Day → Play

def valid_schedule (s : Schedule) : Prop :=
  (s Day.Monday ≠ Play.Thunderstorm) ∧
  (s Day.Thursday ≠ Play.Thunderstorm) ∧
  (s Day.Monday ≠ Play.Teahouse) ∧
  (s Day.Wednesday ≠ Play.Teahouse) ∧
  (s Day.Wednesday ≠ Play.HeavenlySound) ∧
  (s Day.Thursday ≠ Play.HeavenlySound) ∧
  (s Day.Monday ≠ Play.ShatteredHoofbeats) ∧
  (s Day.Thursday ≠ Play.ShatteredHoofbeats) ∧
  (∀ d1 d2, d1 ≠ d2 → s d1 ≠ s d2)

theorem teahouse_on_tuesday_or_thursday :
  ∃ (s : Schedule), valid_schedule s ∧
    (s Day.Tuesday = Play.Teahouse ∨ s Day.Thursday = Play.Teahouse) :=
by sorry

theorem not_all_plays_on_tuesday :
  ¬∃ (s : Schedule), valid_schedule s ∧
    (s Day.Tuesday = Play.Thunderstorm ∧
     s Day.Tuesday = Play.Teahouse ∧
     s Day.Tuesday = Play.HeavenlySound ∧
     s Day.Tuesday = Play.ShatteredHoofbeats) :=
by sorry

theorem heavenly_sound_not_on_wednesday :
  ∀ (s : Schedule), valid_schedule s →
    s Day.Wednesday ≠ Play.HeavenlySound :=
by sorry

theorem thunderstorm_not_only_on_tuesday :
  ∃ (s1 s2 : Schedule), valid_schedule s1 ∧ valid_schedule s2 ∧
    s1 Day.Tuesday = Play.Thunderstorm ∧
    s2 Day.Wednesday = Play.Thunderstorm :=
by sorry

end teahouse_on_tuesday_or_thursday_not_all_plays_on_tuesday_heavenly_sound_not_on_wednesday_thunderstorm_not_only_on_tuesday_l707_70779


namespace sum_of_squares_specific_numbers_l707_70723

theorem sum_of_squares_specific_numbers : 52^2 + 81^2 + 111^2 = 21586 := by
  sorry

end sum_of_squares_specific_numbers_l707_70723


namespace farmer_animals_count_l707_70704

/-- Represents the number of animals a farmer has -/
structure FarmAnimals where
  goats : ℕ
  cows : ℕ
  pigs : ℕ

/-- Calculates the total number of animals -/
def totalAnimals (animals : FarmAnimals) : ℕ :=
  animals.goats + animals.cows + animals.pigs

/-- Theorem stating the total number of animals given the conditions -/
theorem farmer_animals_count :
  ∀ (animals : FarmAnimals),
    animals.goats = 11 →
    animals.cows = animals.goats + 4 →
    animals.pigs = 2 * animals.cows →
    totalAnimals animals = 56 := by
  sorry

end farmer_animals_count_l707_70704


namespace point_outside_circle_l707_70789

/-- A line intersects a circle if and only if the distance from the center of the circle to the line is less than the radius of the circle. -/
axiom line_intersects_circle (a b : ℝ) : 
  (∃ x y, a * x + b * y = 1 ∧ x^2 + y^2 = 1) ↔ (1 / Real.sqrt (a^2 + b^2) < 1)

/-- A point (x, y) is outside a circle centered at the origin with radius r if and only if x^2 + y^2 > r^2. -/
def outside_circle (x y r : ℝ) : Prop := x^2 + y^2 > r^2

theorem point_outside_circle (a b : ℝ) :
  (∃ x y, a * x + b * y = 1 ∧ x^2 + y^2 = 1) → outside_circle a b 1 := by
  sorry

end point_outside_circle_l707_70789


namespace factor_implies_q_value_l707_70773

theorem factor_implies_q_value (q : ℚ) :
  (∀ m : ℚ, (m - 8) ∣ (m^2 - q*m - 24)) → q = 5 := by
  sorry

end factor_implies_q_value_l707_70773
