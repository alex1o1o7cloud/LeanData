import Mathlib

namespace train_journey_length_l4031_403155

theorem train_journey_length 
  (speed_on_time : ℝ) 
  (speed_late : ℝ) 
  (late_time : ℝ) 
  (h1 : speed_on_time = 100)
  (h2 : speed_late = 80)
  (h3 : late_time = 1/3)
  : ∃ (distance : ℝ), distance = 400/3 ∧ 
    distance / speed_on_time = distance / speed_late - late_time :=
by sorry

end train_journey_length_l4031_403155


namespace evaluate_expression_l4031_403111

theorem evaluate_expression : (900^2 : ℚ) / (200^2 - 196^2) = 511 := by sorry

end evaluate_expression_l4031_403111


namespace sin_cos_sum_equals_sqrt3_div_2_l4031_403189

theorem sin_cos_sum_equals_sqrt3_div_2 :
  Real.sin (17 * π / 180) * Real.cos (43 * π / 180) + 
  Real.sin (73 * π / 180) * Real.sin (43 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end sin_cos_sum_equals_sqrt3_div_2_l4031_403189


namespace digit_sum_problem_l4031_403132

theorem digit_sum_problem (x y z w : ℕ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ w ≠ 0 →
  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w →
  x < 10 ∧ y < 10 ∧ z < 10 ∧ w < 10 →
  100 * x + 10 * y + w + 100 * z + 10 * w + x = 1000 →
  x + y + z + w = 18 := by
sorry

end digit_sum_problem_l4031_403132


namespace function_identity_l4031_403190

theorem function_identity (f : ℕ+ → ℕ+) 
  (h : ∀ n : ℕ+, f (n + 1) > f (f n)) : 
  ∀ n : ℕ+, f n = n := by
sorry

end function_identity_l4031_403190


namespace sugar_water_dilution_l4031_403109

theorem sugar_water_dilution (initial_weight : ℝ) (initial_concentration : ℝ) 
  (final_concentration : ℝ) (water_added : ℝ) : 
  initial_weight = 300 →
  initial_concentration = 0.08 →
  final_concentration = 0.05 →
  initial_concentration * initial_weight = final_concentration * (initial_weight + water_added) →
  water_added = 180 := by
sorry

end sugar_water_dilution_l4031_403109


namespace stratified_sampling_teachers_l4031_403193

theorem stratified_sampling_teachers (total : ℕ) (sample_size : ℕ) (students_in_sample : ℕ) 
  (h1 : total = 4000)
  (h2 : sample_size = 200)
  (h3 : students_in_sample = 190) :
  (sample_size : ℚ) / total * (sample_size - students_in_sample) = 200 := by
  sorry

end stratified_sampling_teachers_l4031_403193


namespace total_athletes_l4031_403162

/-- Represents the number of athletes in each sport -/
structure Athletes :=
  (football : ℕ)
  (baseball : ℕ)
  (soccer : ℕ)
  (basketball : ℕ)

/-- The ratio of athletes in different sports -/
def ratio : Athletes := ⟨10, 7, 5, 4⟩

/-- The number of basketball players -/
def basketball_players : ℕ := 16

/-- Theorem stating the total number of athletes in the school -/
theorem total_athletes : 
  ∃ (k : ℕ), 
    k * ratio.football + 
    k * ratio.baseball + 
    k * ratio.soccer + 
    k * ratio.basketball = 104 ∧
    k * ratio.basketball = basketball_players :=
by sorry

end total_athletes_l4031_403162


namespace kelly_carrot_harvest_l4031_403172

/-- Calculates the total weight of carrots harvested given the number of carrots in each bed and the weight ratio --/
def total_carrot_weight (bed1 bed2 bed3 carrots_per_pound : ℕ) : ℕ :=
  ((bed1 + bed2 + bed3) / carrots_per_pound : ℕ)

/-- Theorem stating that Kelly harvested 39 pounds of carrots --/
theorem kelly_carrot_harvest :
  total_carrot_weight 55 101 78 6 = 39 := by
  sorry

#eval total_carrot_weight 55 101 78 6

end kelly_carrot_harvest_l4031_403172


namespace part_one_part_two_l4031_403116

-- Define polynomials A and B
def A (x y : ℝ) : ℝ := x^2 + x*y + 3*y
def B (x y : ℝ) : ℝ := x^2 - x*y

-- Part 1
theorem part_one (x y : ℝ) : (x - 2)^2 + |y + 5| = 0 → 2 * A x y - B x y = -56 := by
  sorry

-- Part 2
theorem part_two (x : ℝ) : (∀ y : ℝ, ∃ c : ℝ, 2 * A x y - B x y = c) ↔ x = -2 := by
  sorry

end part_one_part_two_l4031_403116


namespace carols_age_difference_l4031_403159

theorem carols_age_difference (bob_age carol_age : ℕ) : 
  bob_age = 16 → carol_age = 50 → carol_age - 3 * bob_age = 2 :=
by
  sorry

end carols_age_difference_l4031_403159


namespace sum_of_cyclic_equations_l4031_403198

theorem sum_of_cyclic_equations (p q r : ℝ) : 
  p ≠ q ∧ q ≠ r ∧ r ≠ p →
  q = p * (4 - p) →
  r = q * (4 - q) →
  p = r * (4 - r) →
  p + q + r = 6 := by
  sorry

end sum_of_cyclic_equations_l4031_403198


namespace shortest_combined_track_length_l4031_403105

def melanie_pieces : List Nat := [8, 12]
def martin_pieces : List Nat := [20, 30]
def area_width : Nat := 100
def area_length : Nat := 200

theorem shortest_combined_track_length :
  let melanie_gcd := melanie_pieces.foldl Nat.gcd 0
  let martin_gcd := martin_pieces.foldl Nat.gcd 0
  let common_segment := Nat.lcm melanie_gcd martin_gcd
  let length_segments := area_length / common_segment
  let width_segments := area_width / common_segment
  let total_segments := 2 * (length_segments + width_segments)
  let single_track_length := total_segments * common_segment
  single_track_length * 2 = 1200 := by
sorry


end shortest_combined_track_length_l4031_403105


namespace sum_of_reciprocals_of_roots_l4031_403112

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 + 3*x - 1 = 0

-- Define the roots of the equation
def roots_of_equation (x₁ x₂ : ℝ) : Prop :=
  quadratic_equation x₁ ∧ quadratic_equation x₂ ∧ x₁ ≠ x₂

-- Theorem statement
theorem sum_of_reciprocals_of_roots (x₁ x₂ : ℝ) :
  roots_of_equation x₁ x₂ → 1/x₁ + 1/x₂ = 3 :=
by sorry

end sum_of_reciprocals_of_roots_l4031_403112


namespace cone_height_l4031_403160

/-- Prove that a cone with base area 30 cm² and volume 60 cm³ has a height of 6 cm -/
theorem cone_height (base_area : ℝ) (volume : ℝ) (height : ℝ) : 
  base_area = 30 → volume = 60 → volume = (1/3) * base_area * height → height = 6 := by
  sorry

end cone_height_l4031_403160


namespace max_vertex_sum_l4031_403121

/-- Represents the set of numbers to be assigned to cube faces -/
def cube_numbers : Finset ℕ := {7, 8, 9, 10, 11, 12}

/-- Represents a valid assignment of numbers to cube faces -/
def valid_assignment (assignment : Fin 6 → ℕ) : Prop :=
  ∀ i : Fin 6, assignment i ∈ cube_numbers ∧ (∀ j : Fin 6, i ≠ j → assignment i ≠ assignment j)

/-- Calculates the sum of products at vertices given a face assignment -/
def vertex_sum (assignment : Fin 6 → ℕ) : ℕ :=
  let opposite_pairs := [(0, 1), (2, 3), (4, 5)]
  (assignment 0 + assignment 1) * (assignment 2 + assignment 3) * (assignment 4 + assignment 5)

/-- Theorem stating the maximum sum of vertex products -/
theorem max_vertex_sum :
  ∃ assignment : Fin 6 → ℕ,
    valid_assignment assignment ∧
    vertex_sum assignment = 6859 ∧
    ∀ other : Fin 6 → ℕ, valid_assignment other → vertex_sum other ≤ 6859 := by
  sorry

end max_vertex_sum_l4031_403121


namespace inscribed_triangle_angle_measure_l4031_403145

/-- Given a triangle PQR inscribed in a circle, if the measures of arcs PQ, QR, and RP
    are y + 60°, 2y + 40°, and 3y - 10° respectively, then the measure of interior angle Q
    is 62.5°. -/
theorem inscribed_triangle_angle_measure (y : ℝ) :
  let arc_PQ : ℝ := y + 60
  let arc_QR : ℝ := 2 * y + 40
  let arc_RP : ℝ := 3 * y - 10
  arc_PQ + arc_QR + arc_RP = 360 →
  (1 / 2 : ℝ) * arc_RP = 62.5 := by
  sorry

end inscribed_triangle_angle_measure_l4031_403145


namespace total_non_hot_peppers_l4031_403194

-- Define the types of peppers
inductive PepperType
| Hot
| Sweet
| Mild

-- Define a structure for daily pepper counts
structure DailyPeppers where
  hot : Nat
  sweet : Nat
  mild : Nat

-- Define the week's pepper counts
def weekPeppers : List DailyPeppers := [
  ⟨7, 10, 13⟩,  -- Sunday
  ⟨12, 8, 10⟩,  -- Monday
  ⟨14, 19, 7⟩,  -- Tuesday
  ⟨12, 5, 23⟩,  -- Wednesday
  ⟨5, 20, 5⟩,   -- Thursday
  ⟨18, 15, 12⟩, -- Friday
  ⟨12, 8, 30⟩   -- Saturday
]

-- Function to calculate non-hot peppers for a day
def nonHotPeppers (day : DailyPeppers) : Nat :=
  day.sweet + day.mild

-- Theorem: The sum of non-hot peppers throughout the week is 185
theorem total_non_hot_peppers :
  (weekPeppers.map nonHotPeppers).sum = 185 := by
  sorry


end total_non_hot_peppers_l4031_403194


namespace consecutive_even_numbers_product_product_equals_87526608_l4031_403122

theorem consecutive_even_numbers_product : Int → Prop :=
  fun n => (n - 2) * n * (n + 2) = 87526608

theorem product_equals_87526608 : consecutive_even_numbers_product 444 := by
  sorry

end consecutive_even_numbers_product_product_equals_87526608_l4031_403122


namespace sqrt_sqrt_equation_l4031_403183

theorem sqrt_sqrt_equation (x : ℝ) : Real.sqrt (Real.sqrt x) = 3 → x = 81 := by
  sorry

end sqrt_sqrt_equation_l4031_403183


namespace sinusoidal_function_properties_l4031_403100

/-- Proves that for a sinusoidal function y = A sin(Bx) - C with given properties, A = 2 and C = 1 -/
theorem sinusoidal_function_properties (A B C : ℝ) (hA : A > 0) (hB : B > 0) (hC : C > 0) 
  (hMax : A - C = 3) (hMin : -A - C = -1) : A = 2 ∧ C = 1 := by
  sorry

end sinusoidal_function_properties_l4031_403100


namespace rose_difference_l4031_403188

/-- Given the initial number of roses in a vase, the number of roses thrown away,
    and the final number of roses in the vase, calculate the difference between
    the number of roses thrown away and the number of roses cut from the garden. -/
theorem rose_difference (initial : ℕ) (thrown_away : ℕ) (final : ℕ) :
  initial = 21 → thrown_away = 34 → final = 15 →
  thrown_away - final = 19 := by sorry

end rose_difference_l4031_403188


namespace moving_circle_trajectory_l4031_403141

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

-- Define external tangency
def externally_tangent (x y : ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧
    ((x + 3)^2 + y^2 = (r + 1)^2) ∧
    ((x - 3)^2 + y^2 = (r + 3)^2)

-- Define the trajectory of the center of M
def trajectory (x y : ℝ) : Prop :=
  x < 0 ∧ x^2 - y^2/8 = 1

-- Theorem statement
theorem moving_circle_trajectory :
  ∀ x y : ℝ, externally_tangent x y → trajectory x y :=
sorry

end moving_circle_trajectory_l4031_403141


namespace proposition_equivalence_implies_m_range_l4031_403124

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 3*x - 10 > 0
def q (x m : ℝ) : Prop := x > m^2 - m + 3

-- Define the range of m
def m_range (m : ℝ) : Prop := m ≤ -1 ∨ m ≥ 2

-- State the theorem
theorem proposition_equivalence_implies_m_range :
  (∀ x m : ℝ, (¬(p x) ↔ ¬(q x m))) → 
  (∀ m : ℝ, m_range m) :=
sorry

end proposition_equivalence_implies_m_range_l4031_403124


namespace twenty_sixth_digit_of_N_l4031_403173

def N (d : ℕ) : ℕ := 
  (10^49 - 1) / 9 + d * 10^24 - 10^25 + 1

theorem twenty_sixth_digit_of_N (d : ℕ) : 
  d < 10 → N d % 13 = 0 → d = 9 := by
  sorry

end twenty_sixth_digit_of_N_l4031_403173


namespace probability_not_pulling_prize_l4031_403195

/-- Given odds of 3:4 for pulling a prize, the probability of not pulling the prize is 4/7 -/
theorem probability_not_pulling_prize (odds_for : ℚ) (odds_against : ℚ) 
  (h_odds : odds_for = 3 ∧ odds_against = 4) :
  (odds_against / (odds_for + odds_against)) = 4/7 := by
sorry

end probability_not_pulling_prize_l4031_403195


namespace spelling_contest_questions_spelling_contest_total_questions_l4031_403140

/-- Given a spelling contest with two competitors, Drew and Carla, prove the total number of questions asked. -/
theorem spelling_contest_questions (drew_correct : ℕ) (drew_wrong : ℕ) (carla_correct : ℕ) : ℕ :=
  let drew_total := drew_correct + drew_wrong
  let carla_wrong := 2 * drew_wrong
  let carla_total := carla_correct + carla_wrong
  drew_total + carla_total

/-- Prove that the total number of questions in the spelling contest is 52. -/
theorem spelling_contest_total_questions : spelling_contest_questions 20 6 14 = 52 := by
  sorry

end spelling_contest_questions_spelling_contest_total_questions_l4031_403140


namespace factory_output_l4031_403197

/-- Calculates the number of batteries manufactured by robots in a given time period. -/
def batteries_manufactured (gather_time min_per_battery : ℕ) (create_time min_per_battery : ℕ) 
  (num_robots : ℕ) (total_time hours : ℕ) : ℕ :=
  let total_time_minutes := total_time * 60
  let time_per_battery := gather_time + create_time
  let batteries_per_robot_per_hour := 60 / time_per_battery
  let batteries_per_hour := num_robots * batteries_per_robot_per_hour
  batteries_per_hour * total_time

/-- The number of batteries manufactured by 10 robots in 5 hours is 200. -/
theorem factory_output : batteries_manufactured 6 9 10 5 = 200 := by
  sorry

end factory_output_l4031_403197


namespace mittens_per_box_example_l4031_403179

/-- Given a number of boxes, scarves per box, and total clothing pieces,
    calculate the number of mittens per box. -/
def mittensPerBox (numBoxes : ℕ) (scarvesPerBox : ℕ) (totalClothes : ℕ) : ℕ :=
  let totalScarves := numBoxes * scarvesPerBox
  let totalMittens := totalClothes - totalScarves
  totalMittens / numBoxes

/-- Prove that given 7 boxes, 3 scarves per box, and 49 total clothing pieces,
    there are 4 mittens in each box. -/
theorem mittens_per_box_example :
  mittensPerBox 7 3 49 = 4 := by
  sorry

end mittens_per_box_example_l4031_403179


namespace arithmetic_calculation_l4031_403170

theorem arithmetic_calculation : 24 * 36 + 18 * 24 - 12 * (36 / 6) = 1224 := by
  sorry

end arithmetic_calculation_l4031_403170


namespace base_10_to_base_7_conversion_l4031_403176

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The problem statement --/
theorem base_10_to_base_7_conversion :
  base7ToBase10 [1, 5, 5, 1] = 624 := by
  sorry

end base_10_to_base_7_conversion_l4031_403176


namespace largest_B_term_l4031_403134

def B (k : ℕ) : ℝ := (Nat.choose 500 k) * (0.1 ^ k)

theorem largest_B_term : 
  ∀ k ∈ Finset.range 501, B 45 ≥ B k :=
sorry

end largest_B_term_l4031_403134


namespace modulus_of_z_l4031_403165

theorem modulus_of_z (z : ℂ) (h : z * Complex.I = 1 + Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end modulus_of_z_l4031_403165


namespace suraj_average_increase_l4031_403103

/-- Represents a cricket player's innings record -/
structure InningsRecord where
  initial_innings : ℕ
  new_innings_score : ℕ
  new_average : ℚ

/-- Calculates the increase in average for a given innings record -/
def average_increase (record : InningsRecord) : ℚ :=
  record.new_average - (record.new_average * (record.initial_innings + 1) - record.new_innings_score) / record.initial_innings

/-- Theorem stating that Suraj's average increased by 6 runs -/
theorem suraj_average_increase :
  let suraj_record : InningsRecord := {
    initial_innings := 16,
    new_innings_score := 112,
    new_average := 16
  }
  average_increase suraj_record = 6 := by sorry

end suraj_average_increase_l4031_403103


namespace waiter_customers_proof_l4031_403131

/-- Calculates the number of remaining customers for a waiter given the initial number of tables,
    number of tables that left, and number of customers per table. -/
def remaining_customers (initial_tables : ℝ) (tables_left : ℝ) (customers_per_table : ℝ) : ℝ :=
  (initial_tables - tables_left) * customers_per_table

/-- Proves that the number of remaining customers for a waiter with 44.0 initial tables,
    12.0 tables that left, and 8.0 customers per table is 256.0. -/
theorem waiter_customers_proof :
  remaining_customers 44.0 12.0 8.0 = 256.0 := by
  sorry

#eval remaining_customers 44.0 12.0 8.0

end waiter_customers_proof_l4031_403131


namespace min_quotient_four_digit_number_l4031_403118

def is_digit (n : ℕ) : Prop := 0 < n ∧ n ≤ 9

theorem min_quotient_four_digit_number :
  ∃ (a b c d : ℕ),
    is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    ∀ (w x y z : ℕ),
      is_digit w → is_digit x → is_digit y → is_digit z →
      w ≠ x → w ≠ y → w ≠ z → x ≠ y → x ≠ z → y ≠ z →
      (1000 * a + 100 * b + 10 * c + d : ℚ) / (a + b + c + d) ≤
      (1000 * w + 100 * x + 10 * y + z : ℚ) / (w + x + y + z) :=
by sorry

end min_quotient_four_digit_number_l4031_403118


namespace smallest_value_for_y_between_zero_and_one_l4031_403163

theorem smallest_value_for_y_between_zero_and_one 
  (y : ℝ) (h1 : 0 < y) (h2 : y < 1) :
  y^3 ≤ min (2*y) (min (3*y) (min (y^(1/3)) (1/y))) := by
  sorry

end smallest_value_for_y_between_zero_and_one_l4031_403163


namespace minimum_handshakes_l4031_403113

theorem minimum_handshakes (n : ℕ) (h : n = 30) :
  let handshakes_per_person := 3
  (n * handshakes_per_person) / 2 = 45 :=
by sorry

end minimum_handshakes_l4031_403113


namespace min_value_of_fraction_l4031_403123

theorem min_value_of_fraction (x : ℝ) (h : x ≥ 3/2) :
  (2*x^2 - 2*x + 1) / (x - 1) ≥ 2*Real.sqrt 2 + 2 := by
  sorry

end min_value_of_fraction_l4031_403123


namespace digit_sum_of_special_number_l4031_403168

theorem digit_sum_of_special_number : 
  ∀ (x : ℕ) (x' : ℕ) (y : ℕ),
  10000 ≤ x ∧ x < 100000 →  -- x is a five-digit number
  1000 ≤ x' ∧ x' < 10000 →  -- x' is a four-digit number
  0 ≤ y ∧ y < 10 →          -- y is a single digit
  x = 10 * x' + y →         -- x' is x with the ones digit removed
  x + x' = 52713 →          -- given condition
  (x / 10000) + ((x / 1000) % 10) + ((x / 100) % 10) + ((x / 10) % 10) + (x % 10) = 23 :=
by sorry

end digit_sum_of_special_number_l4031_403168


namespace remainder_problem_l4031_403164

theorem remainder_problem (n : ℤ) (h : n % 25 = 4) : (n + 15) % 5 = 4 := by
  sorry

end remainder_problem_l4031_403164


namespace three_digit_numbers_problem_l4031_403139

theorem three_digit_numbers_problem : ∃ (a b : ℕ), 
  (100 ≤ a ∧ a < 1000) ∧ 
  (100 ≤ b ∧ b < 1000) ∧ 
  (a / 100 = b % 10) ∧ 
  (b / 100 = a % 10) ∧ 
  (a > b → a - b = 297) ∧ 
  (b > a → b - a = 297) ∧ 
  ((a < b → (a / 100 + (a / 10) % 10 + a % 10 = 23)) ∧ 
   (b < a → (b / 100 + (b / 10) % 10 + b % 10 = 23))) ∧ 
  ((a = 986 ∧ b = 689) ∨ (a = 689 ∧ b = 986)) := by
sorry

end three_digit_numbers_problem_l4031_403139


namespace intersection_complement_equals_singleton_zero_l4031_403191

def U : Finset Int := {-1, 0, 1, 2, 3, 4}
def A : Finset Int := {-1, 1, 2, 4}
def B : Finset Int := {-1, 0, 2}

theorem intersection_complement_equals_singleton_zero :
  B ∩ (U \ A) = {0} := by sorry

end intersection_complement_equals_singleton_zero_l4031_403191


namespace work_completion_time_l4031_403153

theorem work_completion_time (T : ℝ) 
  (h1 : 100 * T = 200 * (T - 35)) 
  (h2 : T > 35) : T = 70 := by
sorry

end work_completion_time_l4031_403153


namespace sum_of_cubes_product_l4031_403102

theorem sum_of_cubes_product (x y : ℤ) : x^3 + y^3 = 189 → x * y = 20 := by
  sorry

end sum_of_cubes_product_l4031_403102


namespace money_needed_for_trip_l4031_403177

def trip_cost : ℕ := 5000
def hourly_wage : ℕ := 20
def hours_worked : ℕ := 10
def cookie_price : ℕ := 4
def cookies_sold : ℕ := 24
def lottery_ticket_cost : ℕ := 10
def lottery_winnings : ℕ := 500
def sister_gift : ℕ := 500
def num_sisters : ℕ := 2

theorem money_needed_for_trip :
  trip_cost - (hourly_wage * hours_worked + cookie_price * cookies_sold - lottery_ticket_cost + lottery_winnings + sister_gift * num_sisters) = 3214 := by
  sorry

end money_needed_for_trip_l4031_403177


namespace stating_teacher_duty_arrangements_l4031_403101

/-- Represents the number of science teachers -/
def num_science_teachers : ℕ := 6

/-- Represents the number of liberal arts teachers -/
def num_liberal_arts_teachers : ℕ := 2

/-- Represents the number of days for duty arrangement -/
def num_days : ℕ := 3

/-- Represents the number of science teachers required per day -/
def science_teachers_per_day : ℕ := 2

/-- Represents the number of liberal arts teachers required per day -/
def liberal_arts_teachers_per_day : ℕ := 1

/-- Represents the minimum number of days a teacher should be on duty -/
def min_duty_days : ℕ := 1

/-- Represents the maximum number of days a teacher can be on duty -/
def max_duty_days : ℕ := 2

/-- 
Calculates the number of different arrangements for the teacher duty roster
given the specified conditions
-/
def num_arrangements : ℕ := 540

/-- 
Theorem stating that the number of different arrangements for the teacher duty roster
is equal to 540, given the specified conditions
-/
theorem teacher_duty_arrangements :
  num_arrangements = 540 :=
by sorry

end stating_teacher_duty_arrangements_l4031_403101


namespace point_line_plane_relationship_l4031_403180

-- Define the types for point, line, and plane
variable (Point Line Plane : Type)

-- Define the relationships
variable (lies_on : Point → Line → Prop)
variable (lies_in : Line → Plane → Prop)

-- Define the subset and element relationships
variable (subset : Line → Plane → Prop)
variable (element : Point → Line → Prop)

-- State the theorem
theorem point_line_plane_relationship 
  (A : Point) (a : Line) (α : Plane) 
  (h1 : lies_on A a) 
  (h2 : lies_in a α) :
  element A a ∧ subset a α := by sorry

end point_line_plane_relationship_l4031_403180


namespace circle_max_area_l4031_403199

/-- Given a circle with equation x^2 + y^2 + kx + 2y + k^2 = 0, 
    prove that its area is maximized when its center is at (0, -1) -/
theorem circle_max_area (k : ℝ) : 
  let circle_eq (x y : ℝ) := x^2 + y^2 + k*x + 2*y + k^2 = 0
  let center := (0, -1)
  let radius_squared (k : ℝ) := 1 - (3/4) * k^2
  ∀ x y : ℝ, circle_eq x y → 
    radius_squared k ≤ radius_squared 0 ∧ 
    circle_eq (center.1) (center.2) := by
  sorry

end circle_max_area_l4031_403199


namespace lucas_age_probability_l4031_403154

def coin_sides : Finset ℕ := {5, 15}
def die_sides : Finset ℕ := {1, 2, 3, 4, 5, 6}

def sum_probability (target : ℕ) : ℚ :=
  (coin_sides.filter (λ c => ∃ d ∈ die_sides, c + d = target)).card /
    (coin_sides.card * die_sides.card)

theorem lucas_age_probability :
  sum_probability 15 = 0 := by sorry

end lucas_age_probability_l4031_403154


namespace sum_of_b_and_c_l4031_403186

theorem sum_of_b_and_c (a b c d : ℝ) 
  (h1 : a + b = 14)
  (h2 : c + d = 3)
  (h3 : a + d = 8) :
  b + c = 9 := by
  sorry

end sum_of_b_and_c_l4031_403186


namespace car_speed_time_relation_l4031_403151

/-- Represents a car with its speed and travel time -/
structure Car where
  speed : ℝ
  time : ℝ

/-- The distance traveled by a car -/
def distance (c : Car) : ℝ := c.speed * c.time

theorem car_speed_time_relation (m n : Car) 
  (h1 : n.speed = 2 * m.speed) 
  (h2 : distance n = distance m) : 
  n.time = m.time / 2 := by
  sorry

end car_speed_time_relation_l4031_403151


namespace complement_A_inter_B_l4031_403114

-- Define set A
def A : Set ℝ := {x | x^2 - x - 2 > 0}

-- Define set B
def B : Set ℝ := {x | |2*x - 3| < 3}

-- Theorem statement
theorem complement_A_inter_B :
  ∀ x : ℝ, x ∈ (A ∩ B)ᶜ ↔ x ≥ 3 ∨ x ≤ 2 := by
  sorry

end complement_A_inter_B_l4031_403114


namespace samanthas_last_name_has_seven_letters_l4031_403161

/-- The length of Jamie's last name -/
def jamies_last_name_length : ℕ := 4

/-- The length of Bobbie's last name -/
def bobbies_last_name_length : ℕ := 
  2 * jamies_last_name_length + 2

/-- The length of Samantha's last name -/
def samanthas_last_name_length : ℕ := 
  bobbies_last_name_length - 3

/-- Theorem stating that Samantha's last name has 7 letters -/
theorem samanthas_last_name_has_seven_letters : 
  samanthas_last_name_length = 7 := by
  sorry

end samanthas_last_name_has_seven_letters_l4031_403161


namespace rectangle_diagonal_l4031_403142

/-- Given a rectangle with perimeter 100 meters and length-to-width ratio 5:2,
    prove that its diagonal length is (5 * sqrt 290) / 7 meters. -/
theorem rectangle_diagonal (length width : ℝ) : 
  (2 * (length + width) = 100) →  -- Perimeter condition
  (length / width = 5 / 2) →      -- Ratio condition
  Real.sqrt (length^2 + width^2) = (5 * Real.sqrt 290) / 7 := by
  sorry

end rectangle_diagonal_l4031_403142


namespace subway_speed_problem_l4031_403106

theorem subway_speed_problem :
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 7 ∧ (t^2 + 2*t) - (3^2 + 2*3) = 20 ∧ t = 5 := by
sorry

end subway_speed_problem_l4031_403106


namespace initial_sum_proof_l4031_403133

def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

theorem initial_sum_proof (P : ℝ) (r : ℝ) : 
  simple_interest P r 5 = 1500 ∧ 
  simple_interest P (r + 0.05) 5 = 1750 → 
  P = 1000 := by
  sorry

end initial_sum_proof_l4031_403133


namespace log_inequality_range_l4031_403104

-- Define the logarithm with base 1/2
noncomputable def log_half (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

-- Define the range set
def range_set : Set ℝ := {x | x ∈ Set.Ioc 0 (1/8) ∪ Set.Ici 8}

-- State the theorem
theorem log_inequality_range :
  ∀ x > 0, Complex.abs (log_half x - (0 : ℝ) + 4*Complex.I) ≥ Complex.abs (3 + 4*Complex.I) ↔ x ∈ range_set :=
sorry

end log_inequality_range_l4031_403104


namespace fifth_toss_probability_l4031_403185

def coin_flip_probability (n : ℕ) : ℚ :=
  (1 / 2) ^ (n - 1) * (1 / 2)

theorem fifth_toss_probability :
  coin_flip_probability 5 = 1 / 4 := by
  sorry

end fifth_toss_probability_l4031_403185


namespace square_minus_three_times_l4031_403156

/-- The expression "square of a minus 3 times b" is equivalent to a^2 - 3*b -/
theorem square_minus_three_times (a b : ℝ) : (a^2 - 3*b) = (a^2 - 3*b) := by sorry

end square_minus_three_times_l4031_403156


namespace roots_of_quadratic_expression_l4031_403107

theorem roots_of_quadratic_expression (x₁ x₂ : ℝ) 
  (h₁ : x₁^2 - x₁ - 2023 = 0) 
  (h₂ : x₂^2 - x₂ - 2023 = 0) : 
  x₁^3 - 2023*x₁ + x₂^2 = 4047 := by
  sorry

end roots_of_quadratic_expression_l4031_403107


namespace multiples_of_15_between_17_and_158_l4031_403187

theorem multiples_of_15_between_17_and_158 : 
  (Finset.filter (λ x => x % 15 = 0) (Finset.range (158 - 17 + 1))).card = 9 := by
  sorry

end multiples_of_15_between_17_and_158_l4031_403187


namespace complex_square_l4031_403110

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_square : (1 + i)^2 = 2*i := by sorry

end complex_square_l4031_403110


namespace polynomial_sum_theorem_l4031_403175

theorem polynomial_sum_theorem (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (1 + x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
sorry

end polynomial_sum_theorem_l4031_403175


namespace sum_of_digits_45_times_40_l4031_403171

def product_45_40 : ℕ := 45 * 40

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_45_times_40 : sum_of_digits product_45_40 = 9 := by
  sorry

end sum_of_digits_45_times_40_l4031_403171


namespace single_displacement_equivalent_l4031_403152

-- Define a type for plane figures
structure PlaneFigure where
  -- Add necessary properties for a plane figure

-- Define a function for parallel displacement
def parallelDisplacement (F : PlaneFigure) (v : ℝ × ℝ) : PlaneFigure :=
  sorry

-- Theorem statement
theorem single_displacement_equivalent (F : PlaneFigure) (v1 v2 : ℝ × ℝ) :
  ∃ v : ℝ × ℝ, parallelDisplacement F v = parallelDisplacement (parallelDisplacement F v1) v2 :=
sorry

end single_displacement_equivalent_l4031_403152


namespace leadership_team_selection_l4031_403129

theorem leadership_team_selection (n : ℕ) (k : ℕ) (h1 : n = 20) (h2 : k = 3) : 
  Nat.choose n k = 1140 := by
  sorry

end leadership_team_selection_l4031_403129


namespace fixed_point_on_line_fixed_point_coordinates_l4031_403146

/-- The fixed point through which the line ax + y + 1 = 0 always passes -/
def fixed_point : ℝ × ℝ := sorry

/-- The line equation ax + y + 1 = 0 -/
def line_equation (a x y : ℝ) : Prop := a * x + y + 1 = 0

/-- The theorem stating that the fixed point satisfies the line equation for all values of a -/
theorem fixed_point_on_line : ∀ a : ℝ, line_equation a (fixed_point.1) (fixed_point.2) := sorry

/-- The theorem proving that the fixed point is (0, -1) -/
theorem fixed_point_coordinates : fixed_point = (0, -1) := by sorry

end fixed_point_on_line_fixed_point_coordinates_l4031_403146


namespace race_speed_ratio_l4031_403148

theorem race_speed_ratio (k : ℝ) (v_B : ℝ) : 
  v_B > 0 →
  k * v_B * (20 / v_B) = 80 →
  k = 4 := by
sorry

end race_speed_ratio_l4031_403148


namespace order_of_magnitude_l4031_403119

theorem order_of_magnitude : 70.3 > 70.2 ∧ 70.2 > Real.log 0.3 := by
  have h : 0 < 0.3 ∧ 0.3 < 1 := by sorry
  sorry

end order_of_magnitude_l4031_403119


namespace arithmetic_sequence_third_term_l4031_403178

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n = n^2 + n, a_3 = 6 -/
theorem arithmetic_sequence_third_term (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = n^2 + n) → 
  (∀ n ≥ 2, a n = S n - S (n-1)) → 
  a 3 = 6 := by
sorry

end arithmetic_sequence_third_term_l4031_403178


namespace intersection_when_m_is_two_subset_condition_l4031_403125

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x : ℝ | m - 1 ≤ x ∧ x ≤ 2*m + 1}
def B : Set ℝ := {x : ℝ | -4 ≤ x ∧ x ≤ 2}

-- Theorem 1: When m = 2, A ∩ B = [1, 2]
theorem intersection_when_m_is_two :
  A 2 ∩ B = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem 2: A ⊆ A ∩ B if and only if -2 ≤ m ≤ 1/2
theorem subset_condition (m : ℝ) :
  A m ⊆ A m ∩ B ↔ -2 ≤ m ∧ m ≤ 1/2 := by sorry

end intersection_when_m_is_two_subset_condition_l4031_403125


namespace isosceles_triangle_base_length_l4031_403149

/-- An isosceles triangle with specific properties -/
structure IsoscelesTriangle where
  /-- The length of two equal sides -/
  side_length : ℝ
  /-- The ratio of EJ to JF -/
  ej_jf_ratio : ℝ
  /-- side_length is positive -/
  side_length_pos : 0 < side_length
  /-- ej_jf_ratio is greater than 1 -/
  ej_jf_ratio_gt_one : 1 < ej_jf_ratio

/-- The theorem stating the length of the base of the isosceles triangle -/
theorem isosceles_triangle_base_length (t : IsoscelesTriangle)
    (h1 : t.side_length = 6)
    (h2 : t.ej_jf_ratio = 2) :
  ∃ (base_length : ℝ), base_length = 6 * Real.sqrt 2 := by
  sorry

end isosceles_triangle_base_length_l4031_403149


namespace group_sizes_min_group_a_size_l4031_403182

/-- Represents the ticket price based on the number of people -/
def ticket_price (m : ℕ) : ℕ :=
  if 10 ≤ m ∧ m ≤ 50 then 60
  else if 51 ≤ m ∧ m ≤ 100 then 50
  else 40

/-- The total number of people in both groups -/
def total_people : ℕ := 102

/-- The total amount paid when buying tickets separately -/
def total_amount : ℕ := 5580

/-- Theorem stating the number of people in each group -/
theorem group_sizes :
  ∃ (a b : ℕ), a < 50 ∧ b > 50 ∧ a + b = total_people ∧
  ticket_price a * a + ticket_price b * b = total_amount :=
sorry

/-- Theorem stating the minimum number of people in Group A for savings -/
theorem min_group_a_size :
  ∃ (min_a : ℕ), ∀ a : ℕ, a ≥ min_a →
  ticket_price a * a + ticket_price (total_people - a) * (total_people - a) - 
  ticket_price total_people * total_people ≥ 1200 :=
sorry

end group_sizes_min_group_a_size_l4031_403182


namespace third_smallest_four_digit_pascal_l4031_403128

/-- Pascal's triangle as a function from row and column to value -/
def pascal (n k : ℕ) : ℕ := sorry

/-- Predicate to check if a number is four digits -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- The set of all four-digit numbers in Pascal's triangle -/
def four_digit_pascal : Set ℕ :=
  {n | ∃ (i j : ℕ), pascal i j = n ∧ is_four_digit n}

/-- The third smallest element in a set of natural numbers -/
noncomputable def third_smallest (S : Set ℕ) : ℕ := sorry

theorem third_smallest_four_digit_pascal :
  third_smallest four_digit_pascal = 1002 := by sorry

end third_smallest_four_digit_pascal_l4031_403128


namespace circle_passes_through_points_l4031_403137

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point lies on a circle -/
def Circle.contains (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- The equation of our circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y = 0

theorem circle_passes_through_points :
  ∃ (c : Circle),
    (∀ (x y : ℝ), circle_equation x y ↔ c.contains (x, y)) ∧
    c.contains (0, 0) ∧
    c.contains (4, 0) ∧
    c.contains (-1, 1) := by
  sorry

end circle_passes_through_points_l4031_403137


namespace quadratic_decreasing_before_vertex_l4031_403138

-- Define the quadratic function
def f (x : ℝ) : ℝ := 5 * (x - 3)^2 + 2

-- Theorem statement
theorem quadratic_decreasing_before_vertex :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₂ < 3 → f x₁ > f x₂ := by
  sorry

end quadratic_decreasing_before_vertex_l4031_403138


namespace rectangle_area_error_percent_l4031_403130

/-- Given a rectangle with actual length L and width W, where one side is measured
    as 1.05L and the other as 0.96W, the error percent in the calculated area is 0.8%. -/
theorem rectangle_area_error_percent (L W : ℝ) (L_pos : L > 0) (W_pos : W > 0) :
  let actual_area := L * W
  let measured_area := (1.05 * L) * (0.96 * W)
  let error := measured_area - actual_area
  let error_percent := (error / actual_area) * 100
  error_percent = 0.8 := by sorry

end rectangle_area_error_percent_l4031_403130


namespace age_of_b_l4031_403147

theorem age_of_b (a b c : ℕ) 
  (avg_abc : (a + b + c) / 3 = 25)
  (avg_ac : (a + c) / 2 = 29) : 
  b = 17 := by sorry

end age_of_b_l4031_403147


namespace greatest_multiple_of_four_l4031_403150

theorem greatest_multiple_of_four (x : ℕ) : x > 0 ∧ 4 ∣ x ∧ x^3 < 4096 → x ≤ 12 :=
sorry

end greatest_multiple_of_four_l4031_403150


namespace equal_gum_distribution_l4031_403166

/-- Proves that when three people share 99 pieces of gum equally, each person receives 33 pieces. -/
theorem equal_gum_distribution (john_gum : ℕ) (cole_gum : ℕ) (aubrey_gum : ℕ) 
  (h1 : john_gum = 54)
  (h2 : cole_gum = 45)
  (h3 : aubrey_gum = 0)
  (h4 : (john_gum + cole_gum + aubrey_gum) % 3 = 0) :
  (john_gum + cole_gum + aubrey_gum) / 3 = 33 := by
  sorry

end equal_gum_distribution_l4031_403166


namespace min_total_weight_proof_l4031_403143

/-- The maximum number of crates the trailer can carry on a single trip -/
def max_crates : ℕ := 6

/-- The minimum weight of each crate in kilograms -/
def min_crate_weight : ℕ := 120

/-- The minimum total weight of crates on a single trip when carrying the maximum number of crates -/
def min_total_weight : ℕ := max_crates * min_crate_weight

theorem min_total_weight_proof :
  min_total_weight = 720 := by sorry

end min_total_weight_proof_l4031_403143


namespace count_valid_pairs_l4031_403115

def has_two_distinct_real_solutions (a b c : ℤ) : Prop :=
  b^2 - 4*a*c > 0

def valid_pair (b c : ℕ+) : Prop :=
  ¬(has_two_distinct_real_solutions 1 b c) ∧
  ¬(has_two_distinct_real_solutions 1 c b)

theorem count_valid_pairs :
  ∃ (S : Finset (ℕ+ × ℕ+)), 
    (∀ (p : ℕ+ × ℕ+), p ∈ S ↔ valid_pair p.1 p.2) ∧
    Finset.card S = 6 := by sorry

end count_valid_pairs_l4031_403115


namespace sqrt_sum_equals_seven_l4031_403108

theorem sqrt_sum_equals_seven (y : ℝ) (h : Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4) :
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
  sorry

end sqrt_sum_equals_seven_l4031_403108


namespace line_increase_l4031_403181

/-- Given a line where an x-increase of 4 corresponds to a y-increase of 10,
    prove that an x-increase of 12 results in a y-increase of 30. -/
theorem line_increase (f : ℝ → ℝ) (h : ∀ x, f (x + 4) - f x = 10) :
  ∀ x, f (x + 12) - f x = 30 := by
  sorry

end line_increase_l4031_403181


namespace line_slope_intercept_sum_l4031_403120

/-- Given a line passing through points (1,3) and (4,-2), prove that the sum of its slope and y-intercept is -1/3 -/
theorem line_slope_intercept_sum : 
  ∀ (m b : ℚ), 
  (3 : ℚ) = m * 1 + b →  -- Point (1,3) satisfies the equation
  (-2 : ℚ) = m * 4 + b → -- Point (4,-2) satisfies the equation
  m + b = -1/3 := by
sorry

end line_slope_intercept_sum_l4031_403120


namespace polar_coordinate_transformation_l4031_403192

theorem polar_coordinate_transformation (x y r θ : ℝ) :
  x = 8 ∧ y = 6 ∧ r = Real.sqrt (x^2 + y^2) ∧ 
  x = r * Real.cos θ ∧ y = r * Real.sin θ →
  ∃ (x' y' : ℝ), 
    x' = 2 * Real.sqrt 2 ∧ 
    y' = 14 * Real.sqrt 2 ∧
    x' = (2 * r) * Real.cos (θ + π/4) ∧ 
    y' = (2 * r) * Real.sin (θ + π/4) := by
  sorry

end polar_coordinate_transformation_l4031_403192


namespace complex_fraction_simplification_l4031_403196

theorem complex_fraction_simplification (a : ℝ) (h1 : a ≠ 1) (h2 : a ≠ -1) :
  (1 / (a + 1) - 1 / (a^2 - 1)) / (a / (a - 1) - a) = -1 / (a^2 + a) := by
  sorry

end complex_fraction_simplification_l4031_403196


namespace polynomial_simplification_l4031_403127

theorem polynomial_simplification (x : ℝ) :
  (2 * x^6 + 3 * x^5 + x^4 + 2 * x^2 + 15) - (x^6 + 4 * x^5 - 2 * x^4 + x^3 - 3 * x^2 + 18) =
  x^6 - x^5 + 3 * x^4 - x^3 + 5 * x^2 - 3 := by
  sorry

end polynomial_simplification_l4031_403127


namespace recipe_flour_amount_l4031_403135

def initial_flour : ℕ := 8
def additional_flour : ℕ := 2

theorem recipe_flour_amount : initial_flour + additional_flour = 10 := by sorry

end recipe_flour_amount_l4031_403135


namespace compound_molecular_weight_l4031_403144

/-- Calculates the molecular weight of a compound given the number of atoms and atomic weights -/
def molecularWeight (carbon_atoms : ℕ) (hydrogen_atoms : ℕ) (oxygen_atoms : ℕ) 
  (carbon_weight : ℝ) (hydrogen_weight : ℝ) (oxygen_weight : ℝ) : ℝ :=
  carbon_atoms * carbon_weight + hydrogen_atoms * hydrogen_weight + oxygen_atoms * oxygen_weight

/-- Theorem stating that the molecular weight of the given compound is 192.124 g/mol -/
theorem compound_molecular_weight :
  molecularWeight 6 8 7 12.01 1.008 16.00 = 192.124 := by
  sorry

end compound_molecular_weight_l4031_403144


namespace complement_of_union_l4031_403158

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 2}
def B : Set Int := {x : Int | x^2 - 4*x + 3 = 0}

theorem complement_of_union : (U \ (A ∪ B)) = {-2, 0} := by
  sorry

end complement_of_union_l4031_403158


namespace even_quadratic_function_l4031_403117

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem even_quadratic_function (a b : ℝ) :
  let f : ℝ → ℝ := fun x ↦ a * x^2 + (b - 3) * x + 3
  IsEven f ∧ (∀ x, x ∈ Set.Icc (a^2 - 2) a → f x ∈ Set.range f) →
  a + b = 4 := by
  sorry

end even_quadratic_function_l4031_403117


namespace four_digit_count_l4031_403126

/-- The count of four-digit numbers -/
def count_four_digit_numbers : ℕ := 9000

/-- The smallest four-digit number -/
def min_four_digit : ℕ := 1000

/-- The largest four-digit number -/
def max_four_digit : ℕ := 9999

/-- Theorem: The count of integers from the smallest to the largest four-digit number
    (inclusive) is equal to the count of four-digit numbers. -/
theorem four_digit_count :
  (Finset.range (max_four_digit - min_four_digit + 1)).card = count_four_digit_numbers := by
  sorry

end four_digit_count_l4031_403126


namespace supply_duration_l4031_403174

/-- Represents the number of pills in one supply -/
def supply : ℕ := 90

/-- Represents the fraction of a pill consumed in one dose -/
def dose : ℚ := 3/4

/-- Represents the number of days between doses -/
def interval : ℕ := 3

/-- Represents the number of days in a month (assumed average) -/
def days_per_month : ℕ := 30

/-- Theorem stating that the given supply lasts 12 months -/
theorem supply_duration :
  (supply : ℚ) * interval / dose / days_per_month = 12 := by
  sorry

end supply_duration_l4031_403174


namespace isosceles_base_angles_equal_l4031_403167

/-- An isosceles triangle is a triangle with two sides of equal length -/
structure IsoscelesTriangle where
  points : Fin 3 → ℝ × ℝ
  isosceles : ∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ 
    dist (points i) (points j) = dist (points i) (points k)

/-- The base angles of an isosceles triangle are the angles opposite the equal sides -/
def base_angles (t : IsoscelesTriangle) : ℝ × ℝ := sorry

/-- In an isosceles triangle, the two base angles are equal -/
theorem isosceles_base_angles_equal (t : IsoscelesTriangle) : 
  (base_angles t).1 = (base_angles t).2 := by sorry

end isosceles_base_angles_equal_l4031_403167


namespace payment_difference_l4031_403136

/-- Represents the pizza with its properties and consumption details -/
structure Pizza :=
  (total_slices : ℕ)
  (plain_cost : ℚ)
  (mushroom_cost : ℚ)
  (mushroom_slices : ℕ)
  (alex_plain_slices : ℕ)
  (ally_plain_slices : ℕ)

/-- Calculates the total cost of the pizza -/
def total_cost (p : Pizza) : ℚ :=
  p.plain_cost + p.mushroom_cost

/-- Calculates the cost per slice -/
def cost_per_slice (p : Pizza) : ℚ :=
  total_cost p / p.total_slices

/-- Calculates Alex's payment -/
def alex_payment (p : Pizza) : ℚ :=
  cost_per_slice p * (p.mushroom_slices + p.alex_plain_slices)

/-- Calculates Ally's payment -/
def ally_payment (p : Pizza) : ℚ :=
  cost_per_slice p * p.ally_plain_slices

/-- Theorem stating the difference in payment between Alex and Ally -/
theorem payment_difference (p : Pizza) 
  (h1 : p.total_slices = 12)
  (h2 : p.plain_cost = 12)
  (h3 : p.mushroom_cost = 3)
  (h4 : p.mushroom_slices = 4)
  (h5 : p.alex_plain_slices = 4)
  (h6 : p.ally_plain_slices = 4)
  : alex_payment p - ally_payment p = 5 := by
  sorry


end payment_difference_l4031_403136


namespace f_inequality_solution_f_bounded_by_mn_l4031_403184

def f (x : ℝ) : ℝ := 2 * |x| + |x - 1|

theorem f_inequality_solution (x : ℝ) :
  f x > 4 ↔ x < -1 ∨ x > 5/3 := by sorry

theorem f_bounded_by_mn (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  {x : ℝ | f x ≤ 1/m^2 + 1/n^2 + 2*n*m} = {x : ℝ | -1 ≤ x ∧ x ≤ 5/3} := by sorry

end f_inequality_solution_f_bounded_by_mn_l4031_403184


namespace largest_base5_5digit_in_base10_l4031_403169

/-- The largest five-digit number in base 5 -/
def largest_base5_5digit : ℕ := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_5digit_in_base10 : largest_base5_5digit = 3124 := by
  sorry

end largest_base5_5digit_in_base10_l4031_403169


namespace passes_through_first_and_fourth_quadrants_l4031_403157

-- Define a linear function
def f (x : ℝ) : ℝ := x - 1

-- Theorem statement
theorem passes_through_first_and_fourth_quadrants :
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ f x = y) ∧
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ f x = y) :=
sorry

end passes_through_first_and_fourth_quadrants_l4031_403157
