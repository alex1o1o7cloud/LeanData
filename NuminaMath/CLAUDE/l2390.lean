import Mathlib

namespace stock_price_increase_l2390_239061

theorem stock_price_increase (x : ℝ) : 
  (1 + x / 100) * 0.75 * 1.25 = 1.125 → x = 20 := by
sorry

end stock_price_increase_l2390_239061


namespace right_triangles_with_increasing_sides_l2390_239069

theorem right_triangles_with_increasing_sides (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_pyth1 : a^2 + (b-100)^2 = (c-30)^2)
  (h_pyth2 : a^2 + b^2 = c^2)
  (h_pyth3 : a^2 + (b+100)^2 = (c+40)^2) :
  a = 819 ∧ b = 308 ∧ c = 875 := by
  sorry

end right_triangles_with_increasing_sides_l2390_239069


namespace willys_age_proof_l2390_239047

theorem willys_age_proof :
  ∃ (P : ℤ → ℤ) (A : ℤ),
    (∀ x, ∃ (a₀ a₁ a₂ a₃ : ℤ), P x = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3) ∧
    P 7 = 77 ∧
    P 8 = 85 ∧
    A > 8 ∧
    P A = 0 ∧
    A = 14 := by
  sorry

end willys_age_proof_l2390_239047


namespace minimum_k_value_l2390_239065

theorem minimum_k_value (k : ℝ) : 
  (∀ x y : ℝ, Real.sqrt x + Real.sqrt y ≤ k * Real.sqrt (x + y)) → 
  k ≥ Real.sqrt 2 := by
  sorry

end minimum_k_value_l2390_239065


namespace coordinate_problem_l2390_239053

/-- Represents a point in the coordinate system -/
structure Point where
  x : ℕ
  y : ℕ

/-- The problem statement -/
theorem coordinate_problem (A B : Point) : 
  (A.x < A.y) →  -- Angle OA > 45°
  (B.x > B.y) →  -- Angle OB < 45°
  (B.x * B.y - A.x * A.y = 67) →  -- Area difference
  (A.x * 1000 + B.x * 100 + B.y * 10 + A.y = 1985) := by
  sorry

end coordinate_problem_l2390_239053


namespace range_of_m_l2390_239093

theorem range_of_m (m : ℝ) : 
  Real.sqrt (2 * m + 1) > Real.sqrt (m^2 + m - 1) → 
  m ∈ Set.Ici ((Real.sqrt 5 - 1) / 2) ∩ Set.Iio 2 := by
sorry

end range_of_m_l2390_239093


namespace larger_number_problem_l2390_239009

theorem larger_number_problem (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 10) : 
  max x y = 25 := by
sorry

end larger_number_problem_l2390_239009


namespace binomial_100_3_l2390_239024

theorem binomial_100_3 : Nat.choose 100 3 = 161700 := by
  sorry

end binomial_100_3_l2390_239024


namespace car_wash_earnings_difference_l2390_239002

theorem car_wash_earnings_difference :
  ∀ (total : ℝ) (lisa_earnings : ℝ) (tommy_earnings : ℝ),
  total = 60 →
  lisa_earnings = total / 2 →
  tommy_earnings = lisa_earnings / 2 →
  lisa_earnings - tommy_earnings = 15 :=
by
  sorry

end car_wash_earnings_difference_l2390_239002


namespace toms_deck_cost_l2390_239067

/-- Represents the cost of a deck of cards -/
def deck_cost (rare_count : ℕ) (uncommon_count : ℕ) (common_count : ℕ) 
              (rare_price : ℚ) (uncommon_price : ℚ) (common_price : ℚ) : ℚ :=
  rare_count * rare_price + uncommon_count * uncommon_price + common_count * common_price

/-- Theorem stating that the cost of Tom's deck is $32 -/
theorem toms_deck_cost : 
  deck_cost 19 11 30 1 (1/2) (1/4) = 32 := by
  sorry

end toms_deck_cost_l2390_239067


namespace perpendicular_slope_l2390_239068

theorem perpendicular_slope (x y : ℝ) :
  let original_line := {(x, y) | 4 * x - 6 * y = 12}
  let original_slope := 2 / 3
  let perpendicular_slope := -1 / original_slope
  perpendicular_slope = -3 / 2 := by
sorry

end perpendicular_slope_l2390_239068


namespace river_depth_calculation_l2390_239010

/-- Proves that given a river with specified width, flow rate, and discharge,
    the depth of the river is as calculated. -/
theorem river_depth_calculation
  (width : ℝ)
  (flow_rate_kmph : ℝ)
  (discharge_per_minute : ℝ)
  (h1 : width = 25)
  (h2 : flow_rate_kmph = 8)
  (h3 : discharge_per_minute = 26666.666666666668) :
  let flow_rate_mpm := flow_rate_kmph * 1000 / 60
  let depth := discharge_per_minute / (width * flow_rate_mpm)
  depth = 8 := by sorry

end river_depth_calculation_l2390_239010


namespace triangle_side_length_l2390_239039

theorem triangle_side_length (a c : ℝ) (B : ℝ) (h1 : a = 5) (h2 : c = 8) (h3 : B = π / 3) :
  ∃ b : ℝ, b > 0 ∧ b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) ∧ b = 7 := by
  sorry

end triangle_side_length_l2390_239039


namespace cosine_value_problem_l2390_239044

theorem cosine_value_problem (α : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi / 6)
  (h3 : Real.sin α ^ 6 + Real.cos α ^ 6 = 7 / 12) : 
  1998 * Real.cos α = 333 * Real.sqrt 30 := by
  sorry

end cosine_value_problem_l2390_239044


namespace pizza_toppings_combinations_l2390_239048

theorem pizza_toppings_combinations : Nat.choose 9 3 = 84 := by
  sorry

end pizza_toppings_combinations_l2390_239048


namespace system_solutions_l2390_239060

def system (x y z : ℝ) : Prop :=
  x + y + z = 8 ∧ x * y * z = 8 ∧ 1/x - 1/y - 1/z = 1/8

def solution_set : Set (ℝ × ℝ × ℝ) :=
  { (1, (7 + Real.sqrt 17)/2, (7 - Real.sqrt 17)/2),
    (1, (7 - Real.sqrt 17)/2, (7 + Real.sqrt 17)/2),
    (-1, (9 + Real.sqrt 113)/2, (9 - Real.sqrt 113)/2),
    (-1, (9 - Real.sqrt 113)/2, (9 + Real.sqrt 113)/2) }

theorem system_solutions :
  ∀ x y z : ℝ, system x y z ↔ (x, y, z) ∈ solution_set :=
sorry

end system_solutions_l2390_239060


namespace stratified_sampling_sum_l2390_239098

/-- Calculates the number of items drawn from a category in stratified sampling -/
def items_drawn (category_size : ℕ) (total_size : ℕ) (sample_size : ℕ) : ℕ :=
  (category_size * sample_size) / total_size

/-- Represents the stratified sampling problem -/
theorem stratified_sampling_sum (grains : ℕ) (vegetable_oil : ℕ) (animal_products : ℕ) (fruits_vegetables : ℕ) 
  (sample_size : ℕ) (h1 : grains = 40) (h2 : vegetable_oil = 10) (h3 : animal_products = 30) 
  (h4 : fruits_vegetables = 20) (h5 : sample_size = 20) :
  items_drawn vegetable_oil (grains + vegetable_oil + animal_products + fruits_vegetables) sample_size + 
  items_drawn fruits_vegetables (grains + vegetable_oil + animal_products + fruits_vegetables) sample_size = 6 := by
  sorry


end stratified_sampling_sum_l2390_239098


namespace quadratic_transformation_l2390_239059

theorem quadratic_transformation (a b : ℝ) :
  (∀ x : ℝ, x^2 - 10*x + b = (x - a)^2 - 1) → b - a = 19 := by
  sorry

end quadratic_transformation_l2390_239059


namespace smallest_number_of_eggs_l2390_239013

theorem smallest_number_of_eggs (total_containers : ℕ) (deficient_containers : ℕ) 
  (container_capacity : ℕ) (eggs_in_deficient : ℕ) : 
  total_containers > 10 ∧ 
  deficient_containers = 3 ∧ 
  container_capacity = 15 ∧ 
  eggs_in_deficient = 13 →
  container_capacity * total_containers - 
    deficient_containers * (container_capacity - eggs_in_deficient) = 159 ∧
  159 > 150 :=
by sorry

end smallest_number_of_eggs_l2390_239013


namespace quadratic_downwards_condition_l2390_239073

/-- A quadratic function of the form y = (2a-6)x^2 + 4 -/
def quadratic_function (a : ℝ) (x : ℝ) : ℝ := (2*a - 6)*x^2 + 4

/-- The condition for a quadratic function to open downwards -/
def opens_downwards (a : ℝ) : Prop := 2*a - 6 < 0

theorem quadratic_downwards_condition (a : ℝ) :
  opens_downwards a → a < 3 := by
  sorry

end quadratic_downwards_condition_l2390_239073


namespace correct_sum_calculation_l2390_239077

theorem correct_sum_calculation (tens_digit : Nat) : 
  let original_number := tens_digit * 10 + 9
  let mistaken_number := tens_digit * 10 + 6
  mistaken_number + 57 = 123 →
  original_number + 57 = 126 := by
  sorry

end correct_sum_calculation_l2390_239077


namespace steves_commute_l2390_239020

/-- The distance from Steve's house to work -/
def distance : ℝ := sorry

/-- Steve's speed on the way to work -/
def speed_to_work : ℝ := sorry

/-- Steve's speed on the way back from work -/
def speed_from_work : ℝ := 10

/-- Total time Steve spends on the road daily -/
def total_time : ℝ := 6

theorem steves_commute :
  (speed_from_work = 2 * speed_to_work) →
  (distance / speed_to_work + distance / speed_from_work = total_time) →
  distance = 20 := by sorry

end steves_commute_l2390_239020


namespace saturday_visitors_count_l2390_239080

def friday_visitors : ℕ := 12315
def saturday_multiplier : ℕ := 7

theorem saturday_visitors_count : friday_visitors * saturday_multiplier = 86205 := by
  sorry

end saturday_visitors_count_l2390_239080


namespace unique_solution_product_l2390_239037

theorem unique_solution_product (r : ℝ) : 
  (∃! x : ℝ, (1 : ℝ) / (3 * x) = (r - 2 * x) / 10) → 
  (∃ r₁ r₂ : ℝ, r = r₁ ∨ r = r₂) ∧ r₁ * r₂ = -80 / 3 := by
sorry

end unique_solution_product_l2390_239037


namespace greatest_perimeter_of_special_triangle_l2390_239087

theorem greatest_perimeter_of_special_triangle :
  ∀ (a b c : ℕ),
    a > 0 ∧ b > 0 ∧ c > 0 →
    b = 4 * a →
    c = 20 →
    a + b > c ∧ a + c > b ∧ b + c > a →
    a + b + c ≤ 50 :=
by sorry

end greatest_perimeter_of_special_triangle_l2390_239087


namespace partnership_investment_ratio_l2390_239076

/-- Partnership investment problem -/
theorem partnership_investment_ratio :
  ∀ (n : ℚ) (c : ℚ),
  c > 0 →  -- C's investment is positive
  (2 / 3 * c) / (n * (2 / 3 * c) + (2 / 3 * c) + c) = 800 / 4400 →
  n = 3 :=
by sorry

end partnership_investment_ratio_l2390_239076


namespace minimum_race_distance_proof_l2390_239092

/-- The minimum distance a runner must travel in the race setup -/
def minimum_race_distance : ℝ := 1011

/-- Point A's vertical distance from the wall -/
def distance_A_to_wall : ℝ := 400

/-- Point B's vertical distance above the wall -/
def distance_B_above_wall : ℝ := 600

/-- Point B's horizontal distance to the right of point A -/
def horizontal_distance_A_to_B : ℝ := 150

/-- Theorem stating the minimum distance a runner must travel -/
theorem minimum_race_distance_proof :
  let total_vertical_distance := distance_A_to_wall + distance_B_above_wall
  let squared_distance := horizontal_distance_A_to_B ^ 2 + total_vertical_distance ^ 2
  Real.sqrt squared_distance = minimum_race_distance := by sorry

end minimum_race_distance_proof_l2390_239092


namespace smallest_number_divisible_l2390_239016

theorem smallest_number_divisible (h : ℕ) : 
  (∀ n : ℕ, n < 259 → ¬(((n + 5) % 8 = 0) ∧ ((n + 5) % 11 = 0) ∧ ((n + 5) % 24 = 0))) ∧
  ((259 + 5) % 8 = 0) ∧ ((259 + 5) % 11 = 0) ∧ ((259 + 5) % 24 = 0) :=
by sorry

end smallest_number_divisible_l2390_239016


namespace trig_equality_l2390_239099

theorem trig_equality (θ : ℝ) (h : Real.sin (θ + π/3) = 2/3) : 
  Real.cos (θ - π/6) = 2/3 := by
  sorry

end trig_equality_l2390_239099


namespace parabola_properties_l2390_239072

/-- A parabola is defined by its coefficients a, h, and k in the equation y = a(x-h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- A parabola opens downwards if its 'a' coefficient is negative -/
def opens_downwards (p : Parabola) : Prop := p.a < 0

/-- The axis of symmetry of a parabola is the line x = h -/
def axis_of_symmetry (p : Parabola) (x : ℝ) : Prop := x = p.h

theorem parabola_properties (p : Parabola) :
  opens_downwards p ∧ axis_of_symmetry p 3 → p.a < 0 ∧ p.h = 3 := by
  sorry

end parabola_properties_l2390_239072


namespace friday_to_thursday_ratio_l2390_239033

/-- Represents the study time for each day of the week -/
structure StudyTime where
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ
  weekend : ℝ

/-- The study time satisfies the given conditions -/
def valid_study_time (st : StudyTime) : Prop :=
  st.wednesday = 2 ∧
  st.thursday = 3 * st.wednesday ∧
  st.weekend = st.wednesday + st.thursday + st.friday ∧
  st.wednesday + st.thursday + st.friday + st.weekend = 22

/-- The theorem to be proved -/
theorem friday_to_thursday_ratio (st : StudyTime) 
  (h : valid_study_time st) : st.friday / st.thursday = 1 / 2 := by
  sorry

end friday_to_thursday_ratio_l2390_239033


namespace min_sum_of_mn_l2390_239007

theorem min_sum_of_mn (m n : ℕ+) (h : m.val * n.val - 2 * m.val - 3 * n.val - 20 = 0) :
  ∃ (m' n' : ℕ+), m'.val * n'.val - 2 * m'.val - 3 * n'.val - 20 = 0 ∧ 
  m'.val + n'.val = 20 ∧ 
  ∀ (a b : ℕ+), a.val * b.val - 2 * a.val - 3 * b.val - 20 = 0 → a.val + b.val ≥ 20 :=
sorry

end min_sum_of_mn_l2390_239007


namespace dexter_sam_same_team_l2390_239094

/-- The number of students in the dodgeball league -/
def total_students : ℕ := 12

/-- The number of players in each team -/
def team_size : ℕ := 6

/-- The number of students not including Dexter and Sam -/
def other_students : ℕ := total_students - 2

/-- The number of additional players needed to form a team with Dexter and Sam -/
def additional_players : ℕ := team_size - 2

theorem dexter_sam_same_team :
  (Nat.choose other_students additional_players) = 210 :=
sorry

end dexter_sam_same_team_l2390_239094


namespace factorial_sum_equation_l2390_239005

theorem factorial_sum_equation : ∃ n m : ℕ, n * n.factorial + m * m.factorial = 4032 ∧ n = 7 ∧ m = 6 := by
  sorry

end factorial_sum_equation_l2390_239005


namespace equation_solution_l2390_239075

theorem equation_solution (x : ℚ) :
  (1 / (x + 4) + 1 / (x - 4) = 1 / (x - 4)) → x = 1/2 := by
sorry

end equation_solution_l2390_239075


namespace marble_count_l2390_239022

theorem marble_count (blue : ℕ) (yellow : ℕ) (p_yellow : ℚ) (red : ℕ) :
  blue = 7 →
  yellow = 6 →
  p_yellow = 1/4 →
  red = blue + yellow + red →
  yellow = p_yellow * (blue + yellow + red) →
  red = 11 := by sorry

end marble_count_l2390_239022


namespace negation_of_universal_proposition_l2390_239086

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + x + 1 < 0) := by
  sorry

end negation_of_universal_proposition_l2390_239086


namespace trigonometric_evaluations_l2390_239017

open Real

theorem trigonometric_evaluations :
  (∃ (x : ℝ), x = sin (18 * π / 180) ∧ x = (Real.sqrt 5 - 1) / 4) ∧
  sin (18 * π / 180) * sin (54 * π / 180) = 1 / 4 ∧
  sin (36 * π / 180) * sin (72 * π / 180) = Real.sqrt 5 / 4 :=
by sorry

end trigonometric_evaluations_l2390_239017


namespace percentage_seniors_in_statistics_l2390_239045

theorem percentage_seniors_in_statistics :
  ∀ (total_students : ℕ) (seniors_in_statistics : ℕ),
    total_students = 120 →
    seniors_in_statistics = 54 →
    (seniors_in_statistics : ℚ) / ((total_students : ℚ) / 2) * 100 = 90 := by
  sorry

end percentage_seniors_in_statistics_l2390_239045


namespace first_field_rows_l2390_239011

/-- Represents a corn field with a certain number of rows -/
structure CornField where
  rows : ℕ

/-- Represents a farm with two corn fields -/
structure Farm where
  field1 : CornField
  field2 : CornField

def corn_cobs_per_row : ℕ := 4

def total_corn_cobs (f : Farm) : ℕ :=
  (f.field1.rows + f.field2.rows) * corn_cobs_per_row

theorem first_field_rows (f : Farm) :
  f.field2.rows = 16 → total_corn_cobs f = 116 → f.field1.rows = 13 := by
  sorry

end first_field_rows_l2390_239011


namespace supplementary_angles_ratio_l2390_239088

theorem supplementary_angles_ratio (a b : ℝ) : 
  a + b = 180 →  -- angles are supplementary
  a = 4 * b →    -- angles are in ratio 4:1
  b = 36 :=      -- smaller angle is 36°
by sorry

end supplementary_angles_ratio_l2390_239088


namespace arithmetic_progression_properties_l2390_239031

/-- An infinite arithmetic progression of natural numbers -/
structure ArithmeticProgression :=
  (a : ℕ)  -- First term
  (d : ℕ)  -- Common difference

/-- Predicate to check if a number is composite -/
def IsComposite (n : ℕ) : Prop := ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n = a * b

/-- Predicate to check if a number is a perfect square -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ (k : ℕ), n = k * k

/-- The nth term of an arithmetic progression -/
def nthTerm (ap : ArithmeticProgression) (n : ℕ) : ℕ := ap.a + n * ap.d

theorem arithmetic_progression_properties (ap : ArithmeticProgression) :
  (∀ (N : ℕ), ∃ (n : ℕ), n > N ∧ IsComposite (nthTerm ap n)) ∧
  ((∀ (n : ℕ), ¬IsPerfectSquare (nthTerm ap n)) ∨
   (∀ (N : ℕ), ∃ (n : ℕ), n > N ∧ IsPerfectSquare (nthTerm ap n))) :=
sorry

end arithmetic_progression_properties_l2390_239031


namespace student_number_problem_l2390_239058

theorem student_number_problem (x : ℝ) : 2 * x - 140 = 102 → x = 121 := by
  sorry

end student_number_problem_l2390_239058


namespace negation_equivalence_l2390_239032

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) := by
  sorry

end negation_equivalence_l2390_239032


namespace frost_90_cupcakes_l2390_239030

/-- The number of cupcakes frosted by three people working together --/
def cupcakes_frosted (rate1 rate2 rate3 time : ℚ) : ℚ :=
  time / (1 / rate1 + 1 / rate2 + 1 / rate3)

/-- Theorem stating that Cagney, Lacey, and Jamie can frost 90 cupcakes in 10 minutes --/
theorem frost_90_cupcakes :
  cupcakes_frosted (1/20) (1/30) (1/15) 600 = 90 := by
  sorry

#eval cupcakes_frosted (1/20) (1/30) (1/15) 600

end frost_90_cupcakes_l2390_239030


namespace max_player_salary_l2390_239021

theorem max_player_salary (n : ℕ) (min_salary : ℕ) (total_cap : ℕ) :
  n = 18 →
  min_salary = 20000 →
  total_cap = 600000 →
  (∃ (salaries : Fin n → ℕ), 
    (∀ i, salaries i ≥ min_salary) ∧ 
    (Finset.sum Finset.univ salaries ≤ total_cap) ∧
    (∀ i, salaries i ≤ 260000) ∧
    (∃ j, salaries j = 260000)) ∧
  ¬(∃ (salaries : Fin n → ℕ),
    (∀ i, salaries i ≥ min_salary) ∧
    (Finset.sum Finset.univ salaries ≤ total_cap) ∧
    (∃ j, salaries j > 260000)) :=
by
  sorry

end max_player_salary_l2390_239021


namespace wire_length_around_square_field_l2390_239036

/-- The length of wire required to go 15 times round a square field with area 69696 m^2 is 15840 meters. -/
theorem wire_length_around_square_field : 
  let field_area : ℝ := 69696
  let side_length : ℝ := (field_area) ^ (1/2 : ℝ)
  let perimeter : ℝ := 4 * side_length
  let wire_length : ℝ := 15 * perimeter
  wire_length = 15840 := by sorry

end wire_length_around_square_field_l2390_239036


namespace no_equal_shards_l2390_239006

theorem no_equal_shards : ¬∃ (x y : ℕ), 17 * x + 18 * (35 - y) = 17 * (25 - x) + 18 * y := by
  sorry

end no_equal_shards_l2390_239006


namespace sum_of_fractions_zero_l2390_239096

theorem sum_of_fractions_zero (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 1) :
  a / (b - c) + b / (c - a) + c / (a - b) = 0 := by
  sorry

end sum_of_fractions_zero_l2390_239096


namespace vehicles_separation_time_l2390_239083

/-- Given two vehicles moving in opposite directions, calculate the time taken to reach a specific distance apart. -/
theorem vehicles_separation_time
  (initial_distance : ℝ)
  (speed1 speed2 : ℝ)
  (final_distance : ℝ)
  (h1 : initial_distance = 5)
  (h2 : speed1 = 60)
  (h3 : speed2 = 40)
  (h4 : final_distance = 85) :
  (final_distance - initial_distance) / (speed1 + speed2) = 0.8 := by
  sorry

end vehicles_separation_time_l2390_239083


namespace puppy_cost_first_year_l2390_239040

def adoption_fee : ℝ := 150.00
def dog_food : ℝ := 40.00
def treats : ℝ := 3 * 5.00
def toys : ℝ := 2 * 25.00
def crate : ℝ := 120.00
def bed : ℝ := 80.00
def collar_leash : ℝ := 35.00
def grooming_tools : ℝ := 45.00
def training_classes : ℝ := 55.00 + 60.00 + 60.00 + 70.00 + 70.00
def discount_rate : ℝ := 0.12
def dog_license : ℝ := 25.00
def pet_insurance_first_half : ℝ := 6 * 25.00
def pet_insurance_second_half : ℝ := 6 * 30.00

def discountable_items : ℝ := dog_food + treats + toys + crate + bed + collar_leash + grooming_tools

theorem puppy_cost_first_year :
  let total_initial := adoption_fee + dog_food + treats + toys + crate + bed + collar_leash + grooming_tools + training_classes
  let discount := discount_rate * discountable_items
  let total_after_discount := total_initial - discount
  let total_insurance := pet_insurance_first_half + pet_insurance_second_half
  total_after_discount + dog_license + total_insurance = 1158.80 := by
sorry

end puppy_cost_first_year_l2390_239040


namespace solution_characterization_l2390_239000

/-- The set of ordered pairs (m, n) that satisfy the given condition -/
def SolutionSet : Set (ℕ × ℕ) :=
  {(2, 2), (2, 1), (3, 1), (1, 2), (1, 3), (5, 2), (5, 3), (2, 5), (3, 5)}

/-- Predicate to check if a pair (m, n) satisfies the condition -/
def SatisfiesCondition (p : ℕ × ℕ) : Prop :=
  let m := p.1
  let n := p.2
  m > 0 ∧ n > 0 ∧ ∃ k : ℤ, (n^3 + 1 : ℤ) = k * (m * n - 1)

theorem solution_characterization :
  ∀ p : ℕ × ℕ, p ∈ SolutionSet ↔ SatisfiesCondition p :=
sorry

end solution_characterization_l2390_239000


namespace average_height_problem_l2390_239056

/-- Given the heights of four people with specific relationships, prove their average height. -/
theorem average_height_problem (reese daisy parker giselle : ℝ) : 
  reese = 60 →
  daisy = reese + 8 →
  parker = daisy - 4 →
  giselle = parker - 2 →
  (reese + daisy + parker + giselle) / 4 = 63.5 := by
  sorry

end average_height_problem_l2390_239056


namespace largest_integer_solution_l2390_239014

theorem largest_integer_solution : 
  (∀ x : ℤ, 10 - 3*x > 25 → x ≤ -6) ∧ (10 - 3*(-6) > 25) := by sorry

end largest_integer_solution_l2390_239014


namespace rotten_tomatoes_solution_l2390_239089

/-- Represents the problem of calculating rotten tomatoes --/
def RottenTomatoesProblem (crate_capacity : ℕ) (num_crates : ℕ) (total_cost : ℕ) (selling_price : ℕ) (profit : ℕ) : Prop :=
  let total_capacity := crate_capacity * num_crates
  let revenue := total_cost + profit
  let sold_kg := revenue / selling_price
  total_capacity - sold_kg = 3

/-- Theorem stating the solution to the rotten tomatoes problem --/
theorem rotten_tomatoes_solution :
  RottenTomatoesProblem 20 3 330 6 12 := by
  sorry

#check rotten_tomatoes_solution

end rotten_tomatoes_solution_l2390_239089


namespace press_conference_arrangement_l2390_239062

/-- Number of reporters in each station -/
def n : ℕ := 5

/-- Total number of reporters to be selected -/
def k : ℕ := 4

/-- Number of ways to arrange questioning when selecting 1 from A and 3 from B -/
def case1 : ℕ := Nat.choose n 1 * Nat.choose n 3 * Nat.choose k 1 * (Nat.factorial 3)

/-- Number of ways to arrange questioning when selecting 2 from A and 2 from B -/
def case2 : ℕ := Nat.choose n 2 * Nat.choose n 2 * (2 * (Nat.factorial 2) * (Nat.factorial 2) + (Nat.factorial 2) * (Nat.factorial 2))

/-- Total number of ways to arrange the questioning -/
def total_ways : ℕ := case1 + case2

theorem press_conference_arrangement :
  total_ways = 2400 := by sorry

end press_conference_arrangement_l2390_239062


namespace product_pure_imaginary_l2390_239051

theorem product_pure_imaginary (b : ℝ) : 
  let Z1 : ℂ := 3 - 4*I
  let Z2 : ℂ := 4 + b*I
  (∃ (y : ℝ), Z1 * Z2 = y*I) → b = -3 := by
sorry

end product_pure_imaginary_l2390_239051


namespace ten_boys_handshakes_l2390_239091

/-- The number of handshakes in a group of boys with special conditions -/
def specialHandshakes (n : ℕ) : ℕ :=
  n * (n - 1) / 2 - 2

/-- Theorem: In a group of 10 boys with the given handshake conditions, 
    the total number of handshakes is 43 -/
theorem ten_boys_handshakes : specialHandshakes 10 = 43 := by
  sorry

end ten_boys_handshakes_l2390_239091


namespace yogurt_cases_l2390_239012

theorem yogurt_cases (total_cups : ℕ) (cups_per_box : ℕ) (boxes_per_case : ℕ) 
  (h1 : total_cups = 960) 
  (h2 : cups_per_box = 6) 
  (h3 : boxes_per_case = 8) : 
  (total_cups / cups_per_box) / boxes_per_case = 20 := by
  sorry

end yogurt_cases_l2390_239012


namespace tim_has_five_marbles_l2390_239026

/-- The number of blue marbles Fred has -/
def fred_marbles : ℕ := 110

/-- The ratio of Fred's marbles to Tim's marbles -/
def ratio : ℕ := 22

/-- The number of blue marbles Tim has -/
def tim_marbles : ℕ := fred_marbles / ratio

theorem tim_has_five_marbles : tim_marbles = 5 := by
  sorry

end tim_has_five_marbles_l2390_239026


namespace kendras_goal_is_sixty_l2390_239004

/-- Kendra's goal for new words to learn before her eighth birthday -/
def kendras_goal (words_learned : ℕ) (words_needed : ℕ) : ℕ :=
  words_learned + words_needed

/-- Theorem: Kendra's goal is 60 words -/
theorem kendras_goal_is_sixty :
  kendras_goal 36 24 = 60 := by
  sorry

end kendras_goal_is_sixty_l2390_239004


namespace least_addition_for_divisibility_least_addition_to_1100_for_23_divisibility_least_addition_is_4_l2390_239043

theorem least_addition_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x < d ∧ (n + x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n + y) % d ≠ 0 :=
by sorry

theorem least_addition_to_1100_for_23_divisibility :
  ∃ (x : ℕ), x < 23 ∧ (1100 + x) % 23 = 0 ∧ ∀ (y : ℕ), y < x → (1100 + y) % 23 ≠ 0 :=
by
  apply least_addition_for_divisibility 1100 23
  norm_num

#eval (1100 + 4) % 23  -- This should evaluate to 0

theorem least_addition_is_4 :
  4 < 23 ∧ (1100 + 4) % 23 = 0 ∧ ∀ (y : ℕ), y < 4 → (1100 + y) % 23 ≠ 0 :=
by sorry

end least_addition_for_divisibility_least_addition_to_1100_for_23_divisibility_least_addition_is_4_l2390_239043


namespace a_5_equals_one_l2390_239052

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n, a (n + 1) = r * a n

theorem a_5_equals_one
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a 2)
  (h_pos : ∀ n, a n > 0)
  (h_prod : a 3 * a 11 = 16) :
  a 5 = 1 := by
sorry

end a_5_equals_one_l2390_239052


namespace pizza_size_increase_l2390_239081

theorem pizza_size_increase (r : ℝ) (hr : r > 0) : 
  let medium_area := π * r^2
  let large_radius := 1.1 * r
  let large_area := π * large_radius^2
  (large_area - medium_area) / medium_area = 0.21
  := by sorry

end pizza_size_increase_l2390_239081


namespace calculate_expression_l2390_239034

theorem calculate_expression : (-2)^3 + Real.sqrt 12 + (1/3)⁻¹ = 2 * Real.sqrt 3 - 5 := by
  sorry

end calculate_expression_l2390_239034


namespace work_completion_time_l2390_239008

theorem work_completion_time (x_time y_time : ℝ) (hx : x_time = 30) (hy : y_time = 45) :
  (1 / x_time + 1 / y_time)⁻¹ = 18 := by sorry

end work_completion_time_l2390_239008


namespace compare_squares_l2390_239064

theorem compare_squares (a : ℝ) : (a + 1)^2 > a^2 + 2*a := by
  sorry

end compare_squares_l2390_239064


namespace min_stamps_for_33_cents_l2390_239090

def is_valid_combination (c f : ℕ) : Prop :=
  3 * c + 4 * f = 33

def total_stamps (c f : ℕ) : ℕ :=
  c + f

theorem min_stamps_for_33_cents :
  ∃ (c f : ℕ), is_valid_combination c f ∧
    total_stamps c f = 9 ∧
    ∀ (c' f' : ℕ), is_valid_combination c' f' →
      total_stamps c' f' ≥ 9 :=
sorry

end min_stamps_for_33_cents_l2390_239090


namespace product_sequence_sum_l2390_239001

theorem product_sequence_sum (a b : ℕ) (h1 : a / 3 = 16) (h2 : b = a - 1) : a + b = 95 := by
  sorry

end product_sequence_sum_l2390_239001


namespace fraction_reduction_l2390_239018

theorem fraction_reduction (b y : ℝ) : 
  (Real.sqrt (b^2 + y^2) - (y^2 - b^2) / Real.sqrt (b^2 + y^2)) / (b^2 + y^2) = 
  2 * b^2 / (b^2 + y^2)^(3/2) := by sorry

end fraction_reduction_l2390_239018


namespace product_of_sum_and_difference_l2390_239085

theorem product_of_sum_and_difference (a b : ℝ) : (a + b) * (a - b) = (a + b) * (a - b) := by
  sorry

end product_of_sum_and_difference_l2390_239085


namespace complex_equation_solution_l2390_239055

theorem complex_equation_solution (z : ℂ) (h : z * (1 + Complex.I) = 1 - Complex.I) : 
  z = -Complex.I := by sorry

end complex_equation_solution_l2390_239055


namespace lower_class_students_l2390_239097

/-- Proves that in a school with 120 total students, where the lower class has 36 more students than the upper class, the number of students in the lower class is 78. -/
theorem lower_class_students (total : ℕ) (upper : ℕ) (lower : ℕ) 
  (h1 : total = 120)
  (h2 : upper + lower = total)
  (h3 : lower = upper + 36) :
  lower = 78 := by
  sorry

end lower_class_students_l2390_239097


namespace polynomial_nonnegative_l2390_239038

theorem polynomial_nonnegative (a : ℝ) : a^2 * (a^2 - 1) - a^2 + 1 ≥ 0 := by
  sorry

end polynomial_nonnegative_l2390_239038


namespace angle_with_same_terminal_side_as_600_degrees_l2390_239035

theorem angle_with_same_terminal_side_as_600_degrees :
  ∀ α : ℝ, (∃ k : ℤ, α = 600 + k * 360) → (∃ k : ℤ, α = k * 360 + 240) :=
by sorry

end angle_with_same_terminal_side_as_600_degrees_l2390_239035


namespace power_sum_equals_four_sqrt_three_over_three_logarithm_equation_solution_l2390_239070

-- Define a as log_4(3)
noncomputable def a : ℝ := Real.log 3 / Real.log 4

-- Theorem 1: 2^a + 2^(-a) = (4 * sqrt(3)) / 3
theorem power_sum_equals_four_sqrt_three_over_three :
  2^a + 2^(-a) = (4 * Real.sqrt 3) / 3 := by sorry

-- Theorem 2: The solution to log_2(9^(x-1) - 5) = log_2(3^(x-1) - 2) + 2 is x = 2
theorem logarithm_equation_solution :
  ∃! x : ℝ, (x > 1 ∧ Real.log (9^(x-1) - 5) / Real.log 2 = Real.log (3^(x-1) - 2) / Real.log 2 + 2) ∧ x = 2 := by sorry

end power_sum_equals_four_sqrt_three_over_three_logarithm_equation_solution_l2390_239070


namespace min_value_quadratic_l2390_239074

theorem min_value_quadratic (x y : ℝ) : 
  3 ≤ 5 * x^2 + 4 * y^2 - 8 * x * y + 2 * x + 4 := by
  sorry

end min_value_quadratic_l2390_239074


namespace account_balance_difference_l2390_239063

/-- Calculates the balance of an account with compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Calculates the balance of an account with simple interest -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate * time)

/-- The positive difference between Angela's and Bob's account balances after 15 years -/
theorem account_balance_difference : 
  let angela_balance := compound_interest 9000 0.05 15
  let bob_balance := simple_interest 11000 0.06 15
  ⌊|bob_balance - angela_balance|⌋ = 2189 := by
sorry

end account_balance_difference_l2390_239063


namespace inequality_proof_l2390_239003

theorem inequality_proof (a b : ℝ) (n : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : 1/a + 1/b = 1) :
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) :=
by sorry

end inequality_proof_l2390_239003


namespace smallest_sum_proof_l2390_239050

noncomputable def smallest_sum (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧
  (∃ x : ℝ, x^2 + (Real.sqrt 2 * a) * x + (Real.sqrt 2 * b) = 0) ∧
  (∃ x : ℝ, x^2 + (2 * b) * x + (Real.sqrt 2 * a) = 0) ∧
  a + b = (4 * Real.sqrt 2)^(2/3) / Real.sqrt 2 + (4 * Real.sqrt 2)^(1/3)

theorem smallest_sum_proof (a b : ℝ) :
  smallest_sum a b ↔ 
  (∀ c d : ℝ, c > 0 ∧ d > 0 ∧ 
   (∃ x : ℝ, x^2 + (Real.sqrt 2 * c) * x + (Real.sqrt 2 * d) = 0) ∧
   (∃ x : ℝ, x^2 + (2 * d) * x + (Real.sqrt 2 * c) = 0) →
   c + d ≥ (4 * Real.sqrt 2)^(2/3) / Real.sqrt 2 + (4 * Real.sqrt 2)^(1/3)) :=
by sorry

end smallest_sum_proof_l2390_239050


namespace arrival_time_difference_l2390_239019

-- Define the constants
def distance : ℝ := 2
def jenna_speed : ℝ := 12
def jamie_speed : ℝ := 6

-- Define the theorem
theorem arrival_time_difference : 
  (distance / jenna_speed * 60 - distance / jamie_speed * 60) = 10 := by
  sorry

end arrival_time_difference_l2390_239019


namespace negation_of_implication_l2390_239049

theorem negation_of_implication (x : ℝ) :
  ¬(x^2 = 1 → x = 1 ∨ x = -1) ↔ (x^2 ≠ 1 ∧ x ≠ 1 ∧ x ≠ -1) :=
by sorry

end negation_of_implication_l2390_239049


namespace infinite_rational_points_in_circle_l2390_239041

theorem infinite_rational_points_in_circle : 
  ∀ ε > 0, ∃ (x y : ℚ), x > 0 ∧ y > 0 ∧ x^2 + y^2 ≤ 25 ∧ 
  ∀ (x' y' : ℚ), x' > 0 → y' > 0 → x'^2 + y'^2 ≤ 25 → (x - x')^2 + (y - y')^2 < ε^2 :=
sorry

end infinite_rational_points_in_circle_l2390_239041


namespace insufficient_blue_points_l2390_239027

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A tetrahedron formed by four points in 3D space -/
structure Tetrahedron where
  p1 : Point3D
  p2 : Point3D
  p3 : Point3D
  p4 : Point3D

/-- Checks if a point is inside a tetrahedron -/
def isInside (t : Tetrahedron) (p : Point3D) : Prop := sorry

/-- The set of all tetrahedra formed by n red points -/
def allTetrahedra (redPoints : Finset Point3D) : Finset Tetrahedron := sorry

/-- Theorem: There exists a configuration of n red points such that 3n blue points
    are not sufficient to cover all tetrahedra formed by the red points -/
theorem insufficient_blue_points (n : ℕ) :
  ∃ (redPoints : Finset Point3D),
    redPoints.card = n ∧
    ∀ (bluePoints : Finset Point3D),
      bluePoints.card = 3 * n →
      ∃ (t : Tetrahedron),
        t ∈ allTetrahedra redPoints ∧
        ∀ (p : Point3D), p ∈ bluePoints → ¬isInside t p :=
sorry

end insufficient_blue_points_l2390_239027


namespace intersection_complement_theorem_l2390_239015

def P : Set ℝ := {0, 1, 2}
def N : Set ℝ := {x | x^2 - 3*x + 2 = 0}

theorem intersection_complement_theorem : P ∩ (Set.univ \ N) = {0} := by sorry

end intersection_complement_theorem_l2390_239015


namespace multiple_choice_probabilities_l2390_239046

/-- Represents the scoring rules for multiple-choice questions -/
structure ScoringRules where
  all_correct : Nat
  some_correct : Nat
  incorrect_or_none : Nat

/-- Represents the probabilities of selecting different numbers of options -/
structure SelectionProbabilities where
  one_option : Real
  two_options : Real
  three_options : Real

/-- Represents a multiple-choice question with its correct answer -/
structure MultipleChoiceQuestion where
  correct_options : Nat

/-- Theorem stating the probabilities for the given scenario -/
theorem multiple_choice_probabilities 
  (rules : ScoringRules)
  (probs : SelectionProbabilities)
  (q11 q12 : MultipleChoiceQuestion)
  (h1 : rules.all_correct = 5 ∧ rules.some_correct = 2 ∧ rules.incorrect_or_none = 0)
  (h2 : probs.one_option = 1/3 ∧ probs.two_options = 1/3 ∧ probs.three_options = 1/3)
  (h3 : q11.correct_options = 2 ∧ q12.correct_options = 2) :
  (∃ (p1 p2 : Real),
    -- Probability of getting 2 points for question 11
    p1 = 1/6 ∧
    -- Probability of scoring a total of 7 points for questions 11 and 12
    p2 = 1/54) :=
  sorry

end multiple_choice_probabilities_l2390_239046


namespace complex_equation_sum_l2390_239057

def complex_power (z : ℂ) (n : ℕ) := z ^ n

theorem complex_equation_sum (a b : ℝ) :
  (↑a + ↑b * Complex.I : ℂ) = complex_power Complex.I 2019 →
  a + b = -1 := by
sorry

end complex_equation_sum_l2390_239057


namespace ellipse_foci_distance_l2390_239071

/-- An ellipse with axes parallel to the coordinate axes -/
structure ParallelAxesEllipse where
  /-- The point where the ellipse is tangent to the x-axis -/
  x_tangent : ℝ × ℝ
  /-- The point where the ellipse is tangent to the y-axis -/
  y_tangent : ℝ × ℝ

/-- The distance between the foci of an ellipse -/
def foci_distance (e : ParallelAxesEllipse) : ℝ := sorry

theorem ellipse_foci_distance 
  (e : ParallelAxesEllipse) 
  (h1 : e.x_tangent = (8, 0)) 
  (h2 : e.y_tangent = (0, 2)) : 
  foci_distance e = 4 * Real.sqrt 15 := by sorry

end ellipse_foci_distance_l2390_239071


namespace theater_casting_theorem_l2390_239042

/-- Represents the number of ways to fill roles in a theater company. -/
def theater_casting_combinations (
  female_roles : Nat
) (male_roles : Nat) (
  neutral_roles : Nat
) (auditioning_men : Nat) (
  auditioning_women : Nat
) (qualified_lead_actresses : Nat) : Nat :=
  auditioning_men *
  qualified_lead_actresses *
  (auditioning_women - qualified_lead_actresses) *
  (auditioning_women - qualified_lead_actresses - 1) *
  (auditioning_men + auditioning_women - female_roles - male_roles) *
  (auditioning_men + auditioning_women - female_roles - male_roles - 1) *
  (auditioning_men + auditioning_women - female_roles - male_roles - 2)

/-- Theorem stating the number of ways to fill roles in the specific theater casting scenario. -/
theorem theater_casting_theorem :
  theater_casting_combinations 3 1 3 6 7 3 = 108864 := by
  sorry

end theater_casting_theorem_l2390_239042


namespace product_binary_ternary_l2390_239029

/-- Converts a binary number represented as a list of digits to its decimal value -/
def binary_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 2^i) 0

/-- Converts a ternary number represented as a list of digits to its decimal value -/
def ternary_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 3^i) 0

/-- The main theorem stating that the product of 1011 (base 2) and 1021 (base 3) is 374 (base 10) -/
theorem product_binary_ternary :
  (binary_to_decimal [1, 1, 0, 1]) * (ternary_to_decimal [1, 2, 0, 1]) = 374 := by
  sorry

end product_binary_ternary_l2390_239029


namespace distance_between_points_l2390_239082

theorem distance_between_points :
  let p1 : ℝ × ℝ := (5, 5)
  let p2 : ℝ × ℝ := (0, 0)
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = 5 * Real.sqrt 2 := by
  sorry

end distance_between_points_l2390_239082


namespace total_pencils_l2390_239095

theorem total_pencils (initial_pencils : ℕ) (additional_pencils : ℕ) :
  initial_pencils = 37 → additional_pencils = 17 →
  initial_pencils + additional_pencils = 54 :=
by sorry

end total_pencils_l2390_239095


namespace consecutive_odd_squares_difference_l2390_239023

theorem consecutive_odd_squares_difference (n : ℤ) : 
  ∃ k : ℤ, (2*n + 1)^2 - (2*n - 1)^2 = 8*k := by sorry

end consecutive_odd_squares_difference_l2390_239023


namespace johny_total_distance_l2390_239066

def johny_journey (south_distance : ℕ) : ℕ :=
  let east_distance := south_distance + 20
  let north_distance := 2 * east_distance
  south_distance + east_distance + north_distance

theorem johny_total_distance :
  johny_journey 40 = 220 := by
  sorry

end johny_total_distance_l2390_239066


namespace charlie_coins_l2390_239079

/-- The number of coins Alice and Charlie have satisfy the given conditions -/
def satisfy_conditions (a c : ℕ) : Prop :=
  (c + 2 = 5 * (a - 2)) ∧ (c - 2 = 4 * (a + 2))

/-- Charlie has 98 coins given the conditions -/
theorem charlie_coins : ∃ a : ℕ, satisfy_conditions a 98 := by
  sorry

end charlie_coins_l2390_239079


namespace triangle_inequalities_l2390_239025

/-- For any triangle ABC with exradii r_a, r_b, r_c, inradius r, and circumradius R -/
theorem triangle_inequalities (r_a r_b r_c r R : ℝ) (h_positive : r_a > 0 ∧ r_b > 0 ∧ r_c > 0 ∧ r > 0 ∧ R > 0) :
  r_a^2 + r_b^2 + r_c^2 ≥ 27 * r^2 ∧ 4 * R < r_a + r_b + r_c ∧ r_a + r_b + r_c ≤ 9/2 * R := by
  sorry

end triangle_inequalities_l2390_239025


namespace asterisk_value_for_solution_l2390_239078

theorem asterisk_value_for_solution (x : ℝ) (asterisk : ℝ) :
  (2 * x - 7)^2 + (5 * x - asterisk)^2 = 0 → asterisk = 35/2 := by
  sorry

end asterisk_value_for_solution_l2390_239078


namespace stuffed_animals_sales_difference_l2390_239054

/-- Given the sales of stuffed animals by Jake, Thor, and Quincy, prove that Quincy sold 170 more than Jake. -/
theorem stuffed_animals_sales_difference :
  ∀ (jake_sales thor_sales quincy_sales : ℕ),
  jake_sales = thor_sales + 10 →
  quincy_sales = 10 * thor_sales →
  quincy_sales = 200 →
  quincy_sales - jake_sales = 170 := by
sorry

end stuffed_animals_sales_difference_l2390_239054


namespace power_sum_equals_two_l2390_239028

theorem power_sum_equals_two : (-1)^2 + (1/3)^0 = 2 := by
  sorry

end power_sum_equals_two_l2390_239028


namespace daisy_toys_count_l2390_239084

/-- The total number of dog toys Daisy would have if all lost toys were found -/
def total_toys (initial : ℕ) (bought_tuesday : ℕ) (bought_wednesday : ℕ) : ℕ :=
  initial + bought_tuesday + bought_wednesday

/-- Theorem stating the total number of Daisy's toys if all were found -/
theorem daisy_toys_count :
  total_toys 5 3 5 = 13 :=
by sorry

end daisy_toys_count_l2390_239084
