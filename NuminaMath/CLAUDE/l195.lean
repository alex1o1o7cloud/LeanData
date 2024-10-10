import Mathlib

namespace jennifers_money_l195_19516

theorem jennifers_money (initial_amount : ℚ) : 
  initial_amount > 0 →
  initial_amount - (initial_amount / 5 + initial_amount / 6 + initial_amount / 2) = 12 →
  initial_amount = 90 := by
sorry

end jennifers_money_l195_19516


namespace line_intersection_parameter_range_l195_19525

/-- Given two points A and B, and a line that intersects the line segment AB,
    this theorem proves the range of the parameter m in the line equation. -/
theorem line_intersection_parameter_range :
  let A : ℝ × ℝ := (-1, 2)
  let B : ℝ × ℝ := (2, -1)
  let line (m : ℝ) (x y : ℝ) := x - 2*y + m = 0
  ∀ m : ℝ, (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
    line m ((1-t)*A.1 + t*B.1) ((1-t)*A.2 + t*B.2)) ↔ 
  -4 ≤ m ∧ m ≤ 5 := by
sorry

end line_intersection_parameter_range_l195_19525


namespace balloon_count_is_22_l195_19594

/-- The number of balloons each person brought to the park -/
structure BalloonCount where
  allan : ℕ
  jake : ℕ
  maria : ℕ
  tom_initial : ℕ
  tom_lost : ℕ

/-- The total number of balloons in the park -/
def total_balloons (bc : BalloonCount) : ℕ :=
  bc.allan + bc.jake + bc.maria + (bc.tom_initial - bc.tom_lost)

/-- Theorem: The total number of balloons in the park is 22 -/
theorem balloon_count_is_22 (bc : BalloonCount) 
    (h1 : bc.allan = 5)
    (h2 : bc.jake = 7)
    (h3 : bc.maria = 3)
    (h4 : bc.tom_initial = 9)
    (h5 : bc.tom_lost = 2) : 
  total_balloons bc = 22 := by
  sorry

end balloon_count_is_22_l195_19594


namespace arithmetic_sequence_sum_congruence_l195_19504

def arithmetic_sequence_sum (a : ℕ) (l : ℕ) (d : ℕ) : ℕ :=
  let n := (l - a) / d + 1
  n * (a + l) / 2

theorem arithmetic_sequence_sum_congruence :
  let a := 2
  let l := 137
  let d := 5
  let S := arithmetic_sequence_sum a l d
  S % 20 = 6 := by sorry

end arithmetic_sequence_sum_congruence_l195_19504


namespace triangle_side_length_l195_19522

/-- Proves that in a triangle ABC with A = 60°, B = 45°, and c = 20, the length of side a is equal to 30√2 - 10√6. -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = π / 3 → -- 60° in radians
  B = π / 4 → -- 45° in radians
  c = 20 →
  a = 30 * Real.sqrt 2 - 10 * Real.sqrt 6 :=
by sorry


end triangle_side_length_l195_19522


namespace bons_winning_probability_l195_19563

/-- The probability of rolling a six -/
def prob_six : ℚ := 1/6

/-- The probability of not rolling a six -/
def prob_not_six : ℚ := 5/6

/-- The probability that B. Bons wins the game -/
def prob_bons_wins : ℚ := 5/11

/-- Theorem stating that the probability of B. Bons winning is 5/11 -/
theorem bons_winning_probability : 
  prob_bons_wins = prob_not_six * prob_six + prob_not_six * prob_not_six * prob_bons_wins :=
by sorry

end bons_winning_probability_l195_19563


namespace coffee_cost_for_three_dozen_l195_19560

/-- Calculates the cost of coffee for a given number of dozens of donuts -/
def coffee_cost (dozens : ℕ) : ℕ :=
  let donuts_per_dozen : ℕ := 12
  let coffee_per_donut : ℕ := 2
  let coffee_per_pot : ℕ := 12
  let cost_per_pot : ℕ := 3
  let total_donuts : ℕ := dozens * donuts_per_dozen
  let total_coffee : ℕ := total_donuts * coffee_per_donut
  let pots_needed : ℕ := (total_coffee + coffee_per_pot - 1) / coffee_per_pot
  pots_needed * cost_per_pot

theorem coffee_cost_for_three_dozen : coffee_cost 3 = 18 := by
  sorry

end coffee_cost_for_three_dozen_l195_19560


namespace max_value_cubic_sum_l195_19586

theorem max_value_cubic_sum (x y : ℝ) 
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) 
  (h_x_bound : x ≤ 2) (h_y_bound : y ≤ 3) 
  (h_sum : x + y = 3) : 
  (∀ a b : ℝ, 0 < a → 0 < b → a ≤ 2 → b ≤ 3 → a + b = 3 → 4*a^3 + b^3 ≤ 4*x^3 + y^3) → 
  4*x^3 + y^3 = 33 := by
sorry

end max_value_cubic_sum_l195_19586


namespace final_reflection_of_C_l195_19592

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

def C : ℝ × ℝ := (3, 2)

theorem final_reflection_of_C :
  (reflect_y_eq_x ∘ reflect_y ∘ reflect_x) C = (-2, -3) := by
  sorry

end final_reflection_of_C_l195_19592


namespace parabola_point_x_coord_l195_19503

/-- A point on a parabola with a specific distance to its focus -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4*x
  distance_to_focus : (x - 1)^2 + y^2 = 25

/-- The x-coordinate of a point on a parabola with distance 5 to its focus is 4 -/
theorem parabola_point_x_coord (M : ParabolaPoint) : M.x = 4 := by
  sorry

end parabola_point_x_coord_l195_19503


namespace lorry_weight_is_1800_l195_19575

/-- The total weight of a fully loaded lorry -/
def lorry_weight (empty_weight : ℕ) (apple_bags : ℕ) (apple_weight : ℕ) 
  (orange_bags : ℕ) (orange_weight : ℕ) (watermelon_crates : ℕ) (watermelon_weight : ℕ)
  (firewood_bundles : ℕ) (firewood_weight : ℕ) : ℕ :=
  empty_weight + 
  apple_bags * apple_weight + 
  orange_bags * orange_weight + 
  watermelon_crates * watermelon_weight + 
  firewood_bundles * firewood_weight

/-- Theorem stating the total weight of the fully loaded lorry is 1800 pounds -/
theorem lorry_weight_is_1800 : 
  lorry_weight 500 10 55 5 45 3 125 2 75 = 1800 := by
  sorry

#eval lorry_weight 500 10 55 5 45 3 125 2 75

end lorry_weight_is_1800_l195_19575


namespace brittany_brooke_money_ratio_l195_19561

/-- Given the following conditions about money possession:
  - Alison has half as much money as Brittany
  - Brooke has twice as much money as Kent
  - Kent has $1,000
  - Alison has $4,000
Prove that Brittany has 4 times as much money as Brooke -/
theorem brittany_brooke_money_ratio :
  ∀ (alison brittany brooke kent : ℝ),
  alison = brittany / 2 →
  brooke = 2 * kent →
  kent = 1000 →
  alison = 4000 →
  brittany = 4 * brooke :=
by sorry

end brittany_brooke_money_ratio_l195_19561


namespace uncool_parents_count_l195_19515

theorem uncool_parents_count (total : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (cool_both : ℕ) : 
  total = 30 → cool_dads = 12 → cool_moms = 15 → cool_both = 9 →
  total - (cool_dads + cool_moms - cool_both) = 12 :=
by sorry

end uncool_parents_count_l195_19515


namespace triangle_properties_l195_19582

/-- Triangle ABC with given side lengths and angle -/
structure Triangle where
  c : ℝ
  b : ℝ
  B : ℝ

/-- The possible values for angle C in the triangle -/
def possible_C (t : Triangle) : Set ℝ :=
  {60, 120}

/-- The possible areas of the triangle -/
def possible_areas (t : Triangle) : Set ℝ :=
  {Real.sqrt 3 / 2, Real.sqrt 3 / 4}

/-- Theorem stating the properties of the triangle -/
theorem triangle_properties (t : Triangle) 
  (h_c : t.c = Real.sqrt 3)
  (h_b : t.b = 1)
  (h_B : t.B = 30) :
  (∃ (C : ℝ), C ∈ possible_C t) ∧
  (∃ (area : ℝ), area ∈ possible_areas t) := by
  sorry

end triangle_properties_l195_19582


namespace boys_score_in_class_l195_19536

theorem boys_score_in_class (boy_percentage : ℝ) (girl_percentage : ℝ) 
  (girl_score : ℝ) (class_average : ℝ) : 
  boy_percentage = 40 →
  girl_percentage = 100 - boy_percentage →
  girl_score = 90 →
  class_average = 86 →
  (boy_percentage * boy_score + girl_percentage * girl_score) / 100 = class_average →
  boy_score = 80 :=
by
  sorry

end boys_score_in_class_l195_19536


namespace cubic_root_sum_cubes_l195_19573

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (a^3 - 3*a^2 + 4*a - 5 = 0) → 
  (b^3 - 3*b^2 + 4*b - 5 = 0) → 
  (c^3 - 3*c^2 + 4*c - 5 = 0) → 
  a^3 + b^3 + c^3 = 6 := by
  sorry

end cubic_root_sum_cubes_l195_19573


namespace fraction_equality_l195_19521

theorem fraction_equality (A B : ℤ) (x : ℝ) :
  (A / (x - 2) + B / (x^2 - 4*x + 8) = (x^2 - 4*x + 18) / (x^3 - 6*x^2 + 16*x - 16)) →
  (x ≠ 2 ∧ x ≠ 4 ∧ x^2 - 4*x + 8 ≠ 0) →
  B / A = -4 / 9 := by
sorry

end fraction_equality_l195_19521


namespace trig_expression_evaluation_l195_19585

theorem trig_expression_evaluation : 
  (Real.sqrt 3 * Real.tan (12 * π / 180) - 3) / 
  (Real.sin (12 * π / 180) * (4 * Real.cos (12 * π / 180) ^ 2 - 2)) = -2 * Real.sqrt 3 := by
  sorry

end trig_expression_evaluation_l195_19585


namespace probability_two_red_shoes_l195_19546

theorem probability_two_red_shoes :
  let total_shoes : ℕ := 9
  let red_shoes : ℕ := 5
  let green_shoes : ℕ := 4
  let draw_count : ℕ := 2
  
  total_shoes = red_shoes + green_shoes →
  (Nat.choose red_shoes draw_count : ℚ) / (Nat.choose total_shoes draw_count : ℚ) = 5 / 18 :=
by sorry

end probability_two_red_shoes_l195_19546


namespace simplified_expression_sum_l195_19539

theorem simplified_expression_sum (d : ℝ) (a b c : ℤ) : 
  d ≠ 0 → 
  (15 * d + 16 + 17 * d^2) + (3 * d + 2) = a * d + b + c * d^2 → 
  a + b + c = 53 := by
sorry

end simplified_expression_sum_l195_19539


namespace new_average_weight_l195_19583

def original_team_size : ℕ := 7
def original_average_weight : ℚ := 76
def new_player1_weight : ℚ := 110
def new_player2_weight : ℚ := 60

theorem new_average_weight :
  let original_total_weight := original_team_size * original_average_weight
  let new_total_weight := original_total_weight + new_player1_weight + new_player2_weight
  let new_team_size := original_team_size + 2
  new_total_weight / new_team_size = 78 := by
sorry

end new_average_weight_l195_19583


namespace unique_k_for_quadratic_equation_l195_19501

theorem unique_k_for_quadratic_equation : ∃! k : ℝ, k ≠ 0 ∧
  (∃! a : ℝ, a ≠ 0 ∧
    (∃! x : ℝ, x^2 - (a^3 + 1/a^3) * x + k = 0)) :=
by
  -- The proof goes here
  sorry

end unique_k_for_quadratic_equation_l195_19501


namespace girls_to_boys_ratio_l195_19552

def physics_students : ℕ := 200
def biology_students : ℕ := physics_students / 2
def boys_in_biology : ℕ := 25

def girls_in_biology : ℕ := biology_students - boys_in_biology

theorem girls_to_boys_ratio :
  girls_in_biology / boys_in_biology = 3 := by sorry

end girls_to_boys_ratio_l195_19552


namespace expansion_coefficient_zero_l195_19587

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient function for the expansion of (1 - 1/x)(1+x)^5
def coefficient (r : ℤ) : ℚ :=
  if r = 2 then binomial 5 2 - binomial 5 3
  else if r = 1 then binomial 5 1 - binomial 5 2
  else if r = 0 then 1 - binomial 5 1
  else if r = -1 then -1
  else if r = 3 then binomial 5 3 - binomial 5 4
  else if r = 4 then binomial 5 4 - binomial 5 5
  else if r = 5 then binomial 5 5
  else 0

theorem expansion_coefficient_zero :
  ∃ (r : ℤ), r ∈ Set.Icc (-1 : ℤ) 5 ∧ coefficient r = 0 ∧ r = 2 :=
by sorry

end expansion_coefficient_zero_l195_19587


namespace course_failure_implies_question_failure_l195_19520

-- Define the universe of students
variable (Student : Type)

-- Define predicates
variable (passed_course : Student → Prop)
variable (failed_no_questions : Student → Prop)

-- Ms. Johnson's statement
variable (johnsons_statement : ∀ s : Student, failed_no_questions s → passed_course s)

-- Theorem to prove
theorem course_failure_implies_question_failure :
  ∀ s : Student, ¬(passed_course s) → ¬(failed_no_questions s) :=
by sorry

-- Note: The proof is omitted as per instructions

end course_failure_implies_question_failure_l195_19520


namespace stacy_homework_problem_l195_19568

/-- Represents the number of homework problems assigned by Stacy. -/
def homework_problems : ℕ → ℕ → ℕ → ℕ 
  | true_false, free_response, multiple_choice => 
    true_false + free_response + multiple_choice

theorem stacy_homework_problem :
  ∃ (true_false free_response multiple_choice : ℕ),
    true_false = 6 ∧
    free_response = true_false + 7 ∧
    multiple_choice = 2 * free_response ∧
    homework_problems true_false free_response multiple_choice = 45 :=
by
  sorry

#check stacy_homework_problem

end stacy_homework_problem_l195_19568


namespace travel_ratio_l195_19547

/-- The ratio of distances in a specific travel scenario -/
theorem travel_ratio (d x : ℝ) (h1 : 0 < d) (h2 : 0 < x) (h3 : x < d) :
  (d - x) / 1 = x / 1 + d / 7 → x / (d - x) = 3 / 4 := by
  sorry

end travel_ratio_l195_19547


namespace min_value_sqrt_sum_squares_l195_19523

theorem min_value_sqrt_sum_squares (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  Real.sqrt (a^2 + b^2 + c^2) ≥ Real.sqrt 3 ∧ 
  (Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 3 ↔ a = 1 ∧ b = 1 ∧ c = 1) := by
  sorry

end min_value_sqrt_sum_squares_l195_19523


namespace four_digit_equal_digits_l195_19508

theorem four_digit_equal_digits (n : ℕ+) : 
  (∃ d : ℕ, d ∈ Finset.range 10 ∧ 12 * n.val^2 + 12 * n.val + 11 = d * 1111) → n = 21 := by
  sorry

end four_digit_equal_digits_l195_19508


namespace apple_sales_theorem_l195_19541

/-- Represents the sales of apples over three days in a store. -/
structure AppleSales where
  day1 : ℝ  -- Sales on day 1 in kg
  day2 : ℝ  -- Sales on day 2 in kg
  day3 : ℝ  -- Sales on day 3 in kg

/-- The conditions of the apple sales problem. -/
def appleSalesProblem (s : AppleSales) : Prop :=
  s.day2 = s.day1 / 4 + 8 ∧
  s.day3 = s.day2 / 4 + 8 ∧
  s.day3 = 18

/-- The theorem stating that if the conditions are met, 
    the sales on the first day were 128 kg. -/
theorem apple_sales_theorem (s : AppleSales) :
  appleSalesProblem s → s.day1 = 128 := by
  sorry

#check apple_sales_theorem

end apple_sales_theorem_l195_19541


namespace sixth_sum_is_189_l195_19510

/-- A sequence and its partial sums satisfying the given condition -/
def SequenceWithSum (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → S n = 2 * a n - 3

/-- The sixth partial sum of the sequence is 189 -/
theorem sixth_sum_is_189 (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : SequenceWithSum a S) : S 6 = 189 := by
  sorry

end sixth_sum_is_189_l195_19510


namespace book_pages_count_l195_19595

/-- Given a book with 24 chapters that Frank read in 6 days at a rate of 102 pages per day,
    prove that the total number of pages in the book is 612. -/
theorem book_pages_count (chapters : ℕ) (days : ℕ) (pages_per_day : ℕ) 
  (h1 : chapters = 24)
  (h2 : days = 6)
  (h3 : pages_per_day = 102) :
  chapters * (days * pages_per_day) / chapters = 612 :=
by sorry

end book_pages_count_l195_19595


namespace tangent_line_to_quartic_l195_19511

/-- The value of b for which y = x^4 is tangent to y = 4x + b is -3 -/
theorem tangent_line_to_quartic (x : ℝ) : 
  ∃ (m n : ℝ), 
    n = m^4 ∧ 
    n = 4*m + (-3) ∧ 
    (4:ℝ) = 4*m^3 := by
  sorry

end tangent_line_to_quartic_l195_19511


namespace quadratic_roots_l195_19544

theorem quadratic_roots : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 3*x
  ∃ x₁ x₂ : ℝ, x₁ = 0 ∧ x₂ = 3 ∧ 
    (∀ x : ℝ, f x = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end quadratic_roots_l195_19544


namespace david_did_58_pushups_l195_19557

/-- The number of push-ups David did -/
def davids_pushups (zachary_pushups : ℕ) (difference : ℕ) : ℕ :=
  zachary_pushups + difference

theorem david_did_58_pushups :
  davids_pushups 19 39 = 58 := by
  sorry

end david_did_58_pushups_l195_19557


namespace vacation_pictures_remaining_l195_19593

def zoo_pictures : ℕ := 15
def museum_pictures : ℕ := 18
def deleted_pictures : ℕ := 31

theorem vacation_pictures_remaining :
  zoo_pictures + museum_pictures - deleted_pictures = 2 :=
by
  sorry

end vacation_pictures_remaining_l195_19593


namespace actual_distance_calculation_l195_19551

/-- Given a map distance and scale, calculate the actual distance between two towns. -/
theorem actual_distance_calculation (map_distance : ℝ) (scale_distance : ℝ) (scale_miles : ℝ) : 
  map_distance = 20 → scale_distance = 0.5 → scale_miles = 10 → 
  (map_distance * scale_miles / scale_distance) = 400 := by
sorry

end actual_distance_calculation_l195_19551


namespace xiaolis_estimate_l195_19590

theorem xiaolis_estimate (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (2 * x - y) / 2 > x - y := by
  sorry

end xiaolis_estimate_l195_19590


namespace pascals_triangle_ratio_l195_19581

theorem pascals_triangle_ratio (n : ℕ) (r : ℕ) : n = 84 →
  ∃ r, r + 2 ≤ n ∧
    (Nat.choose n r : ℚ) / (Nat.choose n (r + 1)) = 5 / 6 ∧
    (Nat.choose n (r + 1) : ℚ) / (Nat.choose n (r + 2)) = 6 / 7 :=
by
  sorry


end pascals_triangle_ratio_l195_19581


namespace range_of_k_for_quadratic_inequality_l195_19599

theorem range_of_k_for_quadratic_inequality :
  {k : ℝ | ∀ x : ℝ, k * x^2 - k * x - 1 < 0} = {k : ℝ | -4 < k ∧ k ≤ 0} := by
  sorry

end range_of_k_for_quadratic_inequality_l195_19599


namespace y_completion_time_l195_19540

/-- Represents the time it takes for worker y to complete the job alone -/
def y_time : ℝ := 12

/-- Represents the time it takes for workers x and y to complete the job together -/
def xy_time : ℝ := 20

/-- Represents the number of days x worked alone before y joined -/
def x_solo_days : ℝ := 4

/-- Represents the total number of days the job took to complete -/
def total_days : ℝ := 10

/-- Represents the portion of work completed in one day -/
def work_unit : ℝ := 1

theorem y_completion_time :
  (x_solo_days * (work_unit / xy_time)) +
  ((total_days - x_solo_days) * (work_unit / xy_time + work_unit / y_time)) = work_unit :=
sorry

end y_completion_time_l195_19540


namespace money_distribution_l195_19571

theorem money_distribution (A B C : ℤ) 
  (total : A + B + C = 300)
  (AC_sum : A + C = 200)
  (BC_sum : B + C = 350) :
  C = 250 := by
sorry

end money_distribution_l195_19571


namespace poly_properties_l195_19526

/-- The polynomial under consideration -/
def p (x y : ℝ) : ℝ := 2*x*y - x^2*y + 3*x^3*y - 5

/-- The degree of a term in a polynomial of two variables -/
def term_degree (a b : ℕ) : ℕ := a + b

/-- The degree of the polynomial p -/
def poly_degree : ℕ := 4

/-- The number of terms in the polynomial p -/
def num_terms : ℕ := 4

theorem poly_properties :
  (∃ x y : ℝ, term_degree 3 1 = poly_degree ∧ p x y ≠ 0) ∧
  num_terms = 4 :=
sorry

end poly_properties_l195_19526


namespace students_not_playing_sports_l195_19514

theorem students_not_playing_sports (total_students football_players cricket_players both_players : ℕ) 
  (h1 : total_students = 420)
  (h2 : football_players = 325)
  (h3 : cricket_players = 175)
  (h4 : both_players = 130)
  (h5 : both_players ≤ football_players)
  (h6 : both_players ≤ cricket_players)
  (h7 : football_players ≤ total_students)
  (h8 : cricket_players ≤ total_students) :
  total_students - (football_players + cricket_players - both_players) = 50 := by
sorry

end students_not_playing_sports_l195_19514


namespace infinitely_many_fixed_points_l195_19529

def is_cyclic (f : ℕ → ℕ) : Prop :=
  ∀ n, ∃ k, k > 0 ∧ (f^[k] n = n)

theorem infinitely_many_fixed_points
  (f : ℕ → ℕ)
  (h1 : ∀ n, f n - n < 2021)
  (h2 : is_cyclic f) :
  ∀ m, ∃ n > m, f n = n :=
sorry

end infinitely_many_fixed_points_l195_19529


namespace arithmetic_sequence_sum_l195_19524

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 2 + a 3 = 2 →
  a 4 + a 5 = 6 →
  a 5 + a 6 = 8 := by
sorry

end arithmetic_sequence_sum_l195_19524


namespace carson_total_stars_l195_19596

/-- The number of gold stars Carson earned yesterday -/
def stars_yesterday : ℕ := 6

/-- The number of gold stars Carson earned today -/
def stars_today : ℕ := 9

/-- The total number of gold stars Carson earned -/
def total_stars : ℕ := stars_yesterday + stars_today

theorem carson_total_stars : total_stars = 15 := by
  sorry

end carson_total_stars_l195_19596


namespace prime_remainder_30_l195_19588

theorem prime_remainder_30 (a : ℕ) (h_prime : Nat.Prime a) :
  ∃ (q r : ℕ), a = 30 * q + r ∧ 0 ≤ r ∧ r < 30 ∧ (Nat.Prime r ∨ r = 1) := by
  sorry

end prime_remainder_30_l195_19588


namespace complex_multiplication_result_l195_19548

theorem complex_multiplication_result : (1 + 2 * Complex.I) * (1 - Complex.I) = 3 + Complex.I := by
  sorry

end complex_multiplication_result_l195_19548


namespace bruno_books_l195_19500

theorem bruno_books (initial_books : ℝ) : 
  initial_books - 4.5 + 10.25 = 39.75 → initial_books = 34 := by
sorry

end bruno_books_l195_19500


namespace remainder_1234567_div_123_l195_19517

theorem remainder_1234567_div_123 : 1234567 % 123 = 129 := by
  sorry

end remainder_1234567_div_123_l195_19517


namespace price_change_equivalence_l195_19530

theorem price_change_equivalence :
  let initial_increase := 0.40
  let subsequent_decrease := 0.15
  let equivalent_single_increase := 0.19
  ∀ (original_price : ℝ),
    original_price > 0 →
    original_price * (1 + initial_increase) * (1 - subsequent_decrease) =
    original_price * (1 + equivalent_single_increase) := by
  sorry

end price_change_equivalence_l195_19530


namespace sum_between_nine_half_and_ten_l195_19578

theorem sum_between_nine_half_and_ten : 
  let sum := (29/9 : ℚ) + (11/4 : ℚ) + (81/20 : ℚ)
  (9.5 : ℚ) < sum ∧ sum < (10 : ℚ) :=
by sorry

end sum_between_nine_half_and_ten_l195_19578


namespace arithmetic_sequence_problem_geometric_sequence_problem_l195_19591

-- Problem 1
theorem arithmetic_sequence_problem (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence
  S 2 = S 6 →  -- S₂ = S₆
  a 4 = 1 →    -- a₄ = 1
  a 5 = -1 := by sorry

-- Problem 2
theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence
  a 4 - a 2 = 24 →  -- a₄ - a₂ = 24
  a 2 + a 3 = 6 →   -- a₂ + a₃ = 6
  a 1 = 1/5 ∧ q = 5 := by sorry

end arithmetic_sequence_problem_geometric_sequence_problem_l195_19591


namespace unique_charming_number_l195_19562

theorem unique_charming_number :
  ∃! (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 10 * a + b = 2 * a + b^3 := by
  sorry

end unique_charming_number_l195_19562


namespace ratio_problem_l195_19527

/-- Given that a:b = 4:3 and a:c = 4:15, prove that b:c = 1:5 -/
theorem ratio_problem (a b c : ℚ) 
  (hab : a / b = 4 / 3) 
  (hac : a / c = 4 / 15) : 
  b / c = 1 / 5 := by sorry

end ratio_problem_l195_19527


namespace f_monotonicity_and_extrema_l195_19531

noncomputable def f (x : ℝ) := Real.exp x * (x^2 + x + 1)

theorem f_monotonicity_and_extrema :
  (∀ x y, x < y ∧ y < -2 → f x < f y) ∧
  (∀ x y, -2 < x ∧ x < y ∧ y < -1 → f x > f y) ∧
  (∀ x y, -1 < x ∧ x < y → f x < f y) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x - (-2)| < δ ∧ x ≠ -2 → f x < f (-2)) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x - (-1)| < δ ∧ x ≠ -1 → f x > f (-1)) ∧
  f (-2) = 3 / Real.exp 2 ∧
  f (-1) = 1 / Real.exp 1 :=
by sorry

end f_monotonicity_and_extrema_l195_19531


namespace income_expenditure_ratio_l195_19579

/-- Given income and savings, calculate the ratio of income to expenditure --/
theorem income_expenditure_ratio 
  (income : ℕ) 
  (savings : ℕ) 
  (expenditure : ℕ) 
  (h1 : income = 16000) 
  (h2 : savings = 3200) 
  (h3 : savings = income - expenditure) : 
  (income : ℚ) / expenditure = 5 / 4 := by
sorry

end income_expenditure_ratio_l195_19579


namespace sum_of_coefficients_l195_19576

def polynomial (x : ℝ) : ℝ := -3*(x^8 - 2*x^5 + x^3 - 6) + 5*(2*x^4 - 3*x + 1) - 2*(x^6 - 5)

theorem sum_of_coefficients : 
  (polynomial 1) = 26 := by sorry

end sum_of_coefficients_l195_19576


namespace smallest_staircase_steps_l195_19550

theorem smallest_staircase_steps (n : ℕ) : 
  n > 20 ∧ 
  n % 6 = 4 ∧ 
  n % 7 = 3 ∧ 
  (∀ m : ℕ, m > 20 ∧ m % 6 = 4 ∧ m % 7 = 3 → m ≥ n) → 
  n = 52 := by
sorry

end smallest_staircase_steps_l195_19550


namespace polynomial_evaluation_l195_19589

theorem polynomial_evaluation : 
  let x : ℕ := 2
  (x^4 + x^3 + x^2 + x + 1 : ℕ) = 31 := by
  sorry

end polynomial_evaluation_l195_19589


namespace iris_count_after_rose_addition_l195_19584

/-- Given a garden with an initial ratio of irises to roses of 3:7,
    and an initial count of 35 roses, prove that after adding 30 roses,
    the number of irises that maintains the ratio is 27. -/
theorem iris_count_after_rose_addition 
  (initial_roses : ℕ) 
  (added_roses : ℕ) 
  (iris_ratio : ℕ) 
  (rose_ratio : ℕ) : 
  initial_roses = 35 →
  added_roses = 30 →
  iris_ratio = 3 →
  rose_ratio = 7 →
  (∃ (total_irises : ℕ), 
    total_irises * rose_ratio = (initial_roses + added_roses) * iris_ratio ∧
    total_irises = 27) :=
by sorry

end iris_count_after_rose_addition_l195_19584


namespace rachel_earnings_calculation_l195_19532

/-- Rachel's earnings in one hour -/
def rachel_earnings (base_wage : ℚ) (num_customers : ℕ) (tip_per_customer : ℚ) : ℚ :=
  base_wage + num_customers * tip_per_customer

/-- Theorem: Rachel's earnings in one hour -/
theorem rachel_earnings_calculation : 
  rachel_earnings 12 20 (5/4) = 37 := by
  sorry

end rachel_earnings_calculation_l195_19532


namespace union_covers_reals_iff_a_leq_neg_two_l195_19502

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x < -2 ∨ x ≥ 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | x ≥ a}

-- State the theorem
theorem union_covers_reals_iff_a_leq_neg_two (a : ℝ) :
  A ∪ B a = Set.univ ↔ a ≤ -2 := by
  sorry

end union_covers_reals_iff_a_leq_neg_two_l195_19502


namespace road_length_probability_l195_19569

theorem road_length_probability : 
  ∀ (p_ab p_bc : ℝ),
    0 ≤ p_ab ∧ p_ab ≤ 1 →
    0 ≤ p_bc ∧ p_bc ≤ 1 →
    p_ab = 2/3 →
    p_bc = 1/2 →
    1 - (1 - p_ab) * (1 - p_bc) = 5/6 :=
by
  sorry

end road_length_probability_l195_19569


namespace circle_center_l195_19555

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Represents a circle in polar coordinates -/
structure PolarCircle where
  equation : ℝ → ℝ → Prop

theorem circle_center (C : PolarCircle) :
  C.equation = (fun ρ θ ↦ ρ = 2 * Real.cos (θ + π/4)) →
  ∃ (center : PolarPoint), center.r = 1 ∧ center.θ = -π/4 := by
  sorry

end circle_center_l195_19555


namespace product_of_fractions_l195_19565

theorem product_of_fractions : (1 : ℚ) / 5 * (3 : ℚ) / 7 = (3 : ℚ) / 35 := by
  sorry

end product_of_fractions_l195_19565


namespace sqrt_two_minus_one_zero_minus_three_inv_l195_19534

theorem sqrt_two_minus_one_zero_minus_three_inv :
  (Real.sqrt 2 - 1) ^ 0 - 3⁻¹ = 2 / 3 := by
  sorry

end sqrt_two_minus_one_zero_minus_three_inv_l195_19534


namespace valid_arrangement_probability_l195_19567

/-- Represents the color of a bead -/
inductive BeadColor
  | Green
  | Yellow
  | Purple

/-- Represents an arrangement of beads -/
def BeadArrangement := List BeadColor

/-- Checks if an arrangement is valid according to the given conditions -/
def isValidArrangement (arr : BeadArrangement) : Bool :=
  sorry

/-- Counts the number of valid arrangements -/
def countValidArrangements (green yellow purple : Nat) : Nat :=
  sorry

/-- Calculates the total number of possible arrangements -/
def totalArrangements (green yellow purple : Nat) : Nat :=
  sorry

/-- Theorem stating the probability of a valid arrangement -/
theorem valid_arrangement_probability :
  let green := 4
  let yellow := 3
  let purple := 2
  (countValidArrangements green yellow purple : Rat) / (totalArrangements green yellow purple) = 7 / 315 :=
sorry

end valid_arrangement_probability_l195_19567


namespace cover_room_with_tiles_l195_19574

/-- The width of the room -/
def room_width : ℝ := 8

/-- The length of the room -/
def room_length : ℝ := 12

/-- The width of a tile -/
def tile_width : ℝ := 1.5

/-- The length of a tile -/
def tile_length : ℝ := 2

/-- The number of tiles needed to cover the room -/
def tiles_needed : ℕ := 32

theorem cover_room_with_tiles :
  (room_width * room_length) / (tile_width * tile_length) = tiles_needed := by
  sorry

end cover_room_with_tiles_l195_19574


namespace positive_cube_sum_inequality_l195_19570

theorem positive_cube_sum_inequality (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a^3 + b^3 = 2) : 
  (a + b) * (a^5 + b^5) ≥ 4 ∧ a + b ≤ 2 := by
  sorry

end positive_cube_sum_inequality_l195_19570


namespace vip_seat_cost_l195_19513

theorem vip_seat_cost (total_tickets : ℕ) (total_revenue : ℕ) (general_price : ℕ) (ticket_difference : ℕ) :
  total_tickets = 320 →
  total_revenue = 7500 →
  general_price = 20 →
  ticket_difference = 276 →
  ∃ vip_price : ℕ,
    vip_price = 70 ∧
    (total_tickets - ticket_difference) * general_price + ticket_difference * vip_price = total_revenue :=
by
  sorry

#check vip_seat_cost

end vip_seat_cost_l195_19513


namespace album_jumps_l195_19533

/-- Calculates the number of jumps a person can do while listening to an album --/
theorem album_jumps (jumps_per_second : ℝ) (num_songs : ℕ) (song_length_minutes : ℝ) : 
  jumps_per_second = 1 →
  num_songs = 10 →
  song_length_minutes = 3.5 →
  (jumps_per_second * num_songs * song_length_minutes * 60 : ℝ) = 2100 :=
by
  sorry

end album_jumps_l195_19533


namespace simplify_and_evaluate_expression_l195_19519

theorem simplify_and_evaluate_expression (x y : ℚ) 
  (hx : x = -1) (hy : y = 1/5) : 
  2 * (x^2 * y - 2 * x * y) - 3 * (x^2 * y - 3 * x * y) + x^2 * y = -1 := by
  sorry

end simplify_and_evaluate_expression_l195_19519


namespace function_equality_implies_m_zero_l195_19507

/-- Given two functions f and g, prove that m = 0 when 3f(3) = g(3) -/
theorem function_equality_implies_m_zero (m : ℝ) : 
  let f := fun (x : ℝ) => x^2 - 3*x + m
  let g := fun (x : ℝ) => x^2 - 3*x + 5*m
  3 * f 3 = g 3 → m = 0 := by
  sorry

end function_equality_implies_m_zero_l195_19507


namespace negative_two_cubed_equality_l195_19556

theorem negative_two_cubed_equality : (-2)^3 = -2^3 := by
  sorry

end negative_two_cubed_equality_l195_19556


namespace max_plain_cookies_is_20_l195_19535

/-- Represents the number of cookies with a specific ingredient -/
structure CookieCount where
  total : ℕ
  chocolate : ℕ
  nuts : ℕ
  raisins : ℕ
  sprinkles : ℕ

/-- The conditions of the cookie problem -/
def cookieProblem : CookieCount where
  total := 60
  chocolate := 20
  nuts := 30
  raisins := 40
  sprinkles := 15

/-- The maximum number of cookies without any of the specified ingredients -/
def maxPlainCookies (c : CookieCount) : ℕ :=
  c.total - max c.chocolate (max c.nuts (max c.raisins c.sprinkles))

/-- Theorem stating the maximum number of plain cookies in the given problem -/
theorem max_plain_cookies_is_20 :
  maxPlainCookies cookieProblem = 20 := by
  sorry

end max_plain_cookies_is_20_l195_19535


namespace shortest_path_on_right_angle_polyhedron_l195_19543

/-- A polyhedron with all dihedral angles as right angles -/
structure RightAnglePolyhedron where
  -- We don't need to define the full structure, just what we need for the theorem
  edge_length : ℝ
  all_dihedral_angles_right : True  -- placeholder for the condition

/-- The shortest path between two vertices on the surface of the polyhedron -/
def shortest_surface_path (p : RightAnglePolyhedron) (X Y : ℝ × ℝ × ℝ) : ℝ :=
  sorry  -- The actual implementation would depend on how we represent the polyhedron

theorem shortest_path_on_right_angle_polyhedron 
  (p : RightAnglePolyhedron) 
  (X Y : ℝ × ℝ × ℝ) 
  (h_adjacent : True)  -- placeholder for the condition that X and Y are on adjacent faces
  (h_diagonal : True)  -- placeholder for the condition that X and Y are diagonally opposite
  (h_unit_edge : p.edge_length = 1) :
  shortest_surface_path p X Y = 2 * Real.sqrt 2 := by
  sorry

end shortest_path_on_right_angle_polyhedron_l195_19543


namespace vector_difference_magnitude_l195_19528

/-- Given vectors a and b in ℝ², prove that the magnitude of their difference is 5. -/
theorem vector_difference_magnitude (a b : ℝ × ℝ) : 
  a = (2, 1) → b = (-2, 4) → ‖a - b‖ = 5 := by sorry

end vector_difference_magnitude_l195_19528


namespace binomial_expansion_example_l195_19542

theorem binomial_expansion_example : 7^3 + 3*(7^2)*2 + 3*7*(2^2) + 2^3 = (7 + 2)^3 := by
  sorry

end binomial_expansion_example_l195_19542


namespace triangle_properties_l195_19509

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.b = Real.sqrt 13 ∧ 
  t.a + t.c = 4 ∧
  Real.cos t.B / Real.cos t.C = -t.b / (2 * t.a + t.c)

theorem triangle_properties (t : Triangle) (h : TriangleConditions t) : 
  t.B = 2 * Real.pi / 3 ∧ 
  (1/2) * t.a * t.c * Real.sin t.B = (3/4) * Real.sqrt 3 := by
  sorry

end triangle_properties_l195_19509


namespace polynomial_not_perfect_square_l195_19537

theorem polynomial_not_perfect_square (a b c d : ℤ) (n : ℕ+) :
  ∃ (S : Finset ℕ), 
    S.card ≥ n / 4 ∧ 
    ∀ m ∈ S, m ≤ n ∧ 
    ¬∃ (k : ℤ), (m^5 : ℤ) + d*m^4 + c*m^3 + b*m^2 + 2023*m + a = k^2 := by
  sorry

end polynomial_not_perfect_square_l195_19537


namespace digit_symmetrical_equation_l195_19566

theorem digit_symmetrical_equation (a b : ℤ) (h : 2 ≤ a + b ∧ a + b ≤ 9) :
  (10*a + b) * (100*b + 10*(a + b) + a) = (100*a + 10*(a + b) + b) * (10*b + a) := by
  sorry

end digit_symmetrical_equation_l195_19566


namespace semicircle_radius_l195_19506

theorem semicircle_radius (a b c : ℝ) (h_right : a^2 + b^2 = c^2)
  (h_area : π * a^2 / 8 = 12.5 * π) (h_arc : π * b / 2 = 11 * π) :
  c / 2 = Real.sqrt 584 / 2 := by
  sorry

end semicircle_radius_l195_19506


namespace trailer_homes_added_l195_19554

/-- Represents the number of trailer homes added 5 years ago -/
def added_homes : ℕ := sorry

/-- The initial number of trailer homes -/
def initial_homes : ℕ := 30

/-- The initial average age of trailer homes in years -/
def initial_avg_age : ℕ := 15

/-- The age of added homes when they were added, in years -/
def added_homes_age : ℕ := 3

/-- The number of years that have passed since new homes were added -/
def years_passed : ℕ := 5

/-- The current average age of all trailer homes in years -/
def current_avg_age : ℕ := 17

theorem trailer_homes_added :
  (initial_homes * (initial_avg_age + years_passed) + added_homes * (added_homes_age + years_passed)) /
  (initial_homes + added_homes) = current_avg_age →
  added_homes = 10 := by sorry

end trailer_homes_added_l195_19554


namespace pure_imaginary_condition_second_quadrant_condition_l195_19545

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m^2 - m - 6) (m^2 - 11*m + 24)

-- Theorem for part 1
theorem pure_imaginary_condition (m : ℝ) :
  z m = Complex.I * Complex.im (z m) ↔ m = -2 :=
sorry

-- Theorem for part 2
theorem second_quadrant_condition (m : ℝ) :
  Complex.re (z m) < 0 ∧ Complex.im (z m) > 0 ↔ -2 < m ∧ m < 3 :=
sorry

end pure_imaginary_condition_second_quadrant_condition_l195_19545


namespace largest_divisor_of_15_less_than_15_l195_19559

theorem largest_divisor_of_15_less_than_15 :
  ∃ n : ℕ, n ∣ 15 ∧ n ≠ 15 ∧ ∀ m : ℕ, m ∣ 15 ∧ m ≠ 15 → m ≤ n :=
by
  sorry

end largest_divisor_of_15_less_than_15_l195_19559


namespace unique_m_value_l195_19598

def A (m : ℝ) : Set ℝ := {1, 3, 2*m-1}
def B (m : ℝ) : Set ℝ := {3, m^2}

theorem unique_m_value : ∀ m : ℝ, B m ⊆ A m → m = -1 := by sorry

end unique_m_value_l195_19598


namespace john_drive_distance_l195_19558

-- Define the constants
def speed : ℝ := 55
def time_before_lunch : ℝ := 2
def time_after_lunch : ℝ := 3

-- Define the total distance function
def total_distance (s t1 t2 : ℝ) : ℝ := s * (t1 + t2)

-- Theorem statement
theorem john_drive_distance :
  total_distance speed time_before_lunch time_after_lunch = 275 := by
  sorry

end john_drive_distance_l195_19558


namespace square_binomial_expansion_l195_19549

theorem square_binomial_expansion (x : ℝ) : (x - 2)^2 = x^2 - 4*x + 4 := by
  sorry

end square_binomial_expansion_l195_19549


namespace board_number_remainder_l195_19577

theorem board_number_remainder (n a b c d : ℕ) : 
  n = 102 * a + b ∧ 
  n = 103 * c + d ∧ 
  a + d = 20 ∧ 
  b < 102 →
  b = 20 := by
sorry

end board_number_remainder_l195_19577


namespace unique_number_with_specific_remainders_l195_19553

theorem unique_number_with_specific_remainders :
  ∃! n : ℕ, ∀ k ∈ Finset.range 11, n % (k + 2) = k + 1 := by
  sorry

end unique_number_with_specific_remainders_l195_19553


namespace total_amount_is_468_l195_19572

/-- Calculates the total amount paid including service charge -/
def totalAmountPaid (originalAmount : ℝ) (serviceChargeRate : ℝ) : ℝ :=
  originalAmount * (1 + serviceChargeRate)

/-- Proves that the total amount paid is 468 given the conditions -/
theorem total_amount_is_468 :
  let originalAmount : ℝ := 450
  let serviceChargeRate : ℝ := 0.04
  totalAmountPaid originalAmount serviceChargeRate = 468 := by
  sorry

end total_amount_is_468_l195_19572


namespace line_passes_first_third_quadrants_l195_19564

/-- A line y = kx passes through the first and third quadrants if and only if k > 0 -/
theorem line_passes_first_third_quadrants (k : ℝ) (hk : k ≠ 0) :
  (∀ x y : ℝ, y = k * x → ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0))) ↔ k > 0 :=
sorry

end line_passes_first_third_quadrants_l195_19564


namespace min_additional_squares_for_symmetry_l195_19597

/-- Represents a point on the grid --/
structure Point :=
  (x : Nat) (y : Nat)

/-- Represents the initial configuration of shaded squares --/
def initial_shaded : List Point :=
  [⟨2, 4⟩, ⟨3, 2⟩, ⟨5, 1⟩]

/-- The dimensions of the grid --/
def grid_width : Nat := 6
def grid_height : Nat := 5

/-- Checks if a point is within the grid --/
def is_valid_point (p : Point) : Bool :=
  p.x > 0 ∧ p.x ≤ grid_width ∧ p.y > 0 ∧ p.y ≤ grid_height

/-- Reflects a point across the vertical line of symmetry --/
def reflect_vertical (p : Point) : Point :=
  ⟨grid_width + 1 - p.x, p.y⟩

/-- Reflects a point across the horizontal line of symmetry --/
def reflect_horizontal (p : Point) : Point :=
  ⟨p.x, grid_height + 1 - p.y⟩

/-- Theorem: The minimum number of additional squares to shade for symmetry is 7 --/
theorem min_additional_squares_for_symmetry :
  ∃ (additional_shaded : List Point),
    (∀ p ∈ additional_shaded, is_valid_point p) ∧
    (∀ p ∈ initial_shaded, reflect_vertical p ∈ additional_shaded ∨ reflect_vertical p ∈ initial_shaded) ∧
    (∀ p ∈ initial_shaded, reflect_horizontal p ∈ additional_shaded ∨ reflect_horizontal p ∈ initial_shaded) ∧
    (∀ p ∈ additional_shaded, reflect_vertical p ∈ additional_shaded ∨ reflect_vertical p ∈ initial_shaded) ∧
    (∀ p ∈ additional_shaded, reflect_horizontal p ∈ additional_shaded ∨ reflect_horizontal p ∈ initial_shaded) ∧
    additional_shaded.length = 7 ∧
    (∀ other_shaded : List Point,
      (∀ p ∈ other_shaded, is_valid_point p) →
      (∀ p ∈ initial_shaded, reflect_vertical p ∈ other_shaded ∨ reflect_vertical p ∈ initial_shaded) →
      (∀ p ∈ initial_shaded, reflect_horizontal p ∈ other_shaded ∨ reflect_horizontal p ∈ initial_shaded) →
      (∀ p ∈ other_shaded, reflect_vertical p ∈ other_shaded ∨ reflect_vertical p ∈ initial_shaded) →
      (∀ p ∈ other_shaded, reflect_horizontal p ∈ other_shaded ∨ reflect_horizontal p ∈ initial_shaded) →
      other_shaded.length ≥ 7) :=
by
  sorry

end min_additional_squares_for_symmetry_l195_19597


namespace rita_dress_count_l195_19538

def initial_money : ℕ := 400
def remaining_money : ℕ := 139
def pants_count : ℕ := 3
def jackets_count : ℕ := 4
def dress_price : ℕ := 20
def pants_price : ℕ := 12
def jacket_price : ℕ := 30
def transportation_cost : ℕ := 5

theorem rita_dress_count :
  let total_spent := initial_money - remaining_money
  let pants_jackets_cost := pants_count * pants_price + jackets_count * jacket_price
  let dress_total_cost := total_spent - pants_jackets_cost - transportation_cost
  dress_total_cost / dress_price = 5 := by sorry

end rita_dress_count_l195_19538


namespace transportation_problem_l195_19518

/-- Transportation problem between warehouses and factories -/
theorem transportation_problem 
  (warehouse_a warehouse_b : ℕ)
  (factory_a factory_b : ℕ)
  (cost_a_to_a cost_a_to_b cost_b_to_a cost_b_to_b : ℕ)
  (total_cost : ℕ)
  (h1 : warehouse_a = 20)
  (h2 : warehouse_b = 6)
  (h3 : factory_a = 10)
  (h4 : factory_b = 16)
  (h5 : cost_a_to_a = 400)
  (h6 : cost_a_to_b = 800)
  (h7 : cost_b_to_a = 300)
  (h8 : cost_b_to_b = 500)
  (h9 : total_cost = 16000) :
  ∃ (x y : ℕ),
    x + (warehouse_b - y) = factory_a ∧
    (warehouse_a - x) + y = factory_b ∧
    cost_a_to_a * x + cost_a_to_b * (warehouse_a - x) + 
    cost_b_to_a * (warehouse_b - y) + cost_b_to_b * y = total_cost ∧
    x = 5 ∧ y = 1 :=
by sorry

end transportation_problem_l195_19518


namespace divisibility_of_expression_l195_19505

theorem divisibility_of_expression (a b c : ℤ) (h : 4 * b = 10 - 3 * a + c) :
  ∃ k : ℤ, 3 * b + 15 - c = 1 * k :=
by sorry

end divisibility_of_expression_l195_19505


namespace correct_expression_proof_l195_19512

theorem correct_expression_proof (x a b : ℝ) : 
  ((2*x - a) * (3*x + b) = 6*x^2 - 13*x + 6) →
  ((2*x + a) * (x + b) = 2*x^2 - x - 6) →
  (a = 3 ∧ b = -2 ∧ (2*x + a) * (3*x + b) = 6*x^2 + 5*x - 6) := by
  sorry

end correct_expression_proof_l195_19512


namespace square_side_length_l195_19580

theorem square_side_length (d : ℝ) (h : d = 2) :
  ∃ s : ℝ, s * s = 2 ∧ s * Real.sqrt 2 = d :=
by sorry

end square_side_length_l195_19580
