import Mathlib

namespace original_cube_volume_l932_93247

theorem original_cube_volume (s : ℝ) : 
  (2 * s)^3 = 2744 → s^3 = 343 := by
  sorry

end original_cube_volume_l932_93247


namespace workshop_workers_l932_93238

/-- Represents the total number of workers in the workshop -/
def total_workers : ℕ := sorry

/-- Represents the average salary of all workers -/
def average_salary : ℚ := 9500

/-- Represents the number of technicians -/
def num_technicians : ℕ := 7

/-- Represents the average salary of technicians -/
def technician_salary : ℚ := 12000

/-- Represents the average salary of non-technicians -/
def non_technician_salary : ℚ := 6000

/-- Theorem stating that the total number of workers is 12 -/
theorem workshop_workers : total_workers = 12 := by sorry

end workshop_workers_l932_93238


namespace income_comparison_l932_93280

theorem income_comparison (a b : ℝ) (h : a = 0.75 * b) : 
  (b - a) / a = 1 / 3 := by
  sorry

end income_comparison_l932_93280


namespace museum_paintings_l932_93211

theorem museum_paintings (initial : ℕ) (removed : ℕ) (remaining : ℕ) : 
  initial = 98 → removed = 3 → remaining = initial - removed → remaining = 95 := by
sorry

end museum_paintings_l932_93211


namespace apples_left_l932_93200

theorem apples_left (frank_apples susan_apples : ℕ) : 
  frank_apples = 36 →
  susan_apples = 3 * frank_apples →
  (frank_apples - frank_apples / 3) + (susan_apples - susan_apples / 2) = 78 :=
by
  sorry

end apples_left_l932_93200


namespace bacteria_growth_l932_93269

/-- Bacteria growth problem -/
theorem bacteria_growth (initial_count : ℕ) (growth_factor : ℕ) (interval_count : ℕ) : 
  initial_count = 50 → 
  growth_factor = 3 → 
  interval_count = 5 → 
  initial_count * growth_factor ^ interval_count = 12150 := by
sorry

#eval 50 * 3 ^ 5  -- Expected output: 12150

end bacteria_growth_l932_93269


namespace five_dice_probability_l932_93242

/-- The probability of rolling a number greater than 1 on a single die -/
def prob_not_one : ℚ := 5/6

/-- The number of ways to choose 2 dice out of 5 -/
def choose_two_from_five : ℕ := 10

/-- The probability of two dice summing to 10 -/
def prob_sum_ten : ℚ := 1/12

/-- The probability of rolling five dice where none show 1 and two of them sum to 10 -/
def prob_five_dice : ℚ := (prob_not_one ^ 5) * choose_two_from_five * prob_sum_ten

theorem five_dice_probability :
  prob_five_dice = 2604.1667 / 7776 :=
sorry

end five_dice_probability_l932_93242


namespace negation_of_proposition_l932_93260

theorem negation_of_proposition (p : ∀ x : ℝ, x^2 - x + 1 > 0) :
  (∃ x_0 : ℝ, x_0^2 - x_0 + 1 ≤ 0) ↔ ¬(∀ x : ℝ, x^2 - x + 1 > 0) :=
by sorry

end negation_of_proposition_l932_93260


namespace odd_factors_of_360_is_6_l932_93208

/-- The number of odd factors of 360 -/
def odd_factors_of_360 : ℕ :=
  -- Definition goes here
  sorry

/-- Theorem: The number of odd factors of 360 is 6 -/
theorem odd_factors_of_360_is_6 : odd_factors_of_360 = 6 := by
  sorry

end odd_factors_of_360_is_6_l932_93208


namespace point_reflection_y_axis_l932_93261

/-- Given a point Q with coordinates (-3,7) in the Cartesian coordinate system,
    its coordinates with respect to the y-axis are (3,7). -/
theorem point_reflection_y_axis :
  let Q : ℝ × ℝ := (-3, 7)
  let reflected_Q : ℝ × ℝ := (3, 7)
  reflected_Q = (- Q.1, Q.2) :=
by sorry

end point_reflection_y_axis_l932_93261


namespace team_sports_count_l932_93275

theorem team_sports_count (total_score : ℕ) : ∃ (n : ℕ), 
  (n > 0) ∧ 
  ((97 + total_score) / n = 90) ∧ 
  ((73 + total_score) / n = 87) → 
  n = 8 := by
sorry

end team_sports_count_l932_93275


namespace trihedral_angle_properties_l932_93290

-- Define a trihedral angle
structure TrihedralAngle where
  planeAngle1 : ℝ
  planeAngle2 : ℝ
  planeAngle3 : ℝ
  dihedralAngle1 : ℝ
  dihedralAngle2 : ℝ
  dihedralAngle3 : ℝ

-- State the theorem
theorem trihedral_angle_properties (t : TrihedralAngle) :
  t.planeAngle1 + t.planeAngle2 + t.planeAngle3 < 2 * Real.pi ∧
  t.dihedralAngle1 + t.dihedralAngle2 + t.dihedralAngle3 > Real.pi :=
by sorry

end trihedral_angle_properties_l932_93290


namespace probiotic_diameter_scientific_notation_l932_93229

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem probiotic_diameter_scientific_notation :
  toScientificNotation 0.00000002 = ScientificNotation.mk 2 (-8) (by norm_num) :=
sorry

end probiotic_diameter_scientific_notation_l932_93229


namespace park_ticket_cost_l932_93254

theorem park_ticket_cost (teacher_count student_count ticket_price total_budget : ℕ) :
  teacher_count = 3 →
  student_count = 9 →
  ticket_price = 22 →
  total_budget = 300 →
  (teacher_count + student_count) * ticket_price ≤ total_budget :=
by
  sorry

end park_ticket_cost_l932_93254


namespace f_at_negative_one_l932_93276

def f (x : ℝ) : ℝ := 5 * (2 * x^3 - 3 * x^2 + 4 * x - 1)

theorem f_at_negative_one : f (-1) = -50 := by
  sorry

end f_at_negative_one_l932_93276


namespace max_min_values_l932_93216

theorem max_min_values (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  (∀ x y, 0 < x ∧ 0 < y ∧ x + y = 1 → Real.sqrt x + Real.sqrt y ≤ Real.sqrt 2) ∧
  (a^2 + b^2 ≥ 1/2) ∧
  (1 / (a + 2*b) + 1 / (2*a + b) ≥ 4/3) := by
  sorry

end max_min_values_l932_93216


namespace remainder_problem_l932_93225

theorem remainder_problem : ∃ q : ℕ, 
  6598574241545098875458255622898854689448911257658451215825362549889 = 
  3721858987156557895464215545212524189541456658712589687354871258 * q + 8 * 23 + r ∧ 
  r < 23 := by sorry

end remainder_problem_l932_93225


namespace hyperbola_perpendicular_points_sum_l932_93235

/-- Given a hyperbola x²/a² - y²/b² = 1 with 0 < a < b, and points A and B on the hyperbola
    such that OA is perpendicular to OB, prove that 1/|OA|² + 1/|OB|² = 1/a² - 1/b² -/
theorem hyperbola_perpendicular_points_sum (a b : ℝ) (ha : 0 < a) (hb : a < b)
  (A B : ℝ × ℝ) (hA : A.1^2 / a^2 - A.2^2 / b^2 = 1) (hB : B.1^2 / a^2 - B.2^2 / b^2 = 1)
  (hperp : A.1 * B.1 + A.2 * B.2 = 0) :
  1 / (A.1^2 + A.2^2) + 1 / (B.1^2 + B.2^2) = 1 / a^2 - 1 / b^2 :=
sorry

end hyperbola_perpendicular_points_sum_l932_93235


namespace min_sum_given_log_condition_l932_93259

theorem min_sum_given_log_condition (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  Real.log m / Real.log 3 + Real.log n / Real.log 3 ≥ 4 → m + n ≥ 18 := by
  sorry


end min_sum_given_log_condition_l932_93259


namespace best_washing_effect_and_full_capacity_l932_93288

-- Define the constants
def drum_capacity : Real := 25
def current_clothes : Real := 4.92
def current_detergent_scoops : Nat := 3
def scoop_weight : Real := 0.02
def water_per_scoop : Real := 5

-- Define the variables for additional detergent and water
def additional_detergent : Real := 0.02
def additional_water : Real := 20

-- Theorem statement
theorem best_washing_effect_and_full_capacity : 
  -- The total weight equals the drum capacity
  current_clothes + (current_detergent_scoops * scoop_weight) + additional_detergent + additional_water = drum_capacity ∧
  -- The ratio of water to detergent is correct for best washing effect
  (current_detergent_scoops * scoop_weight + additional_detergent) / 
    (additional_water + water_per_scoop * current_detergent_scoops) = 1 / water_per_scoop :=
by sorry

end best_washing_effect_and_full_capacity_l932_93288


namespace nested_fraction_evaluation_l932_93289

theorem nested_fraction_evaluation : 
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / 3))) = 8 / 21 := by
  sorry

end nested_fraction_evaluation_l932_93289


namespace inequality_preservation_l932_93258

theorem inequality_preservation (a b c : ℝ) (h : a > b) : a / (c^2 + 1) > b / (c^2 + 1) := by
  sorry

end inequality_preservation_l932_93258


namespace function_composition_equality_l932_93240

theorem function_composition_equality (m n p q : ℝ) 
  (f : ℝ → ℝ) (g : ℝ → ℝ) 
  (hf : ∀ x, f x = m * x + n) 
  (hg : ∀ x, g x = p * x + q) : 
  (∃ x, f (g x) = g (f x)) ↔ n * (1 - p) = q * (1 - m) := by
sorry

end function_composition_equality_l932_93240


namespace never_equal_implies_m_range_l932_93210

theorem never_equal_implies_m_range (m : ℝ) :
  (∀ x : ℝ, 2 * x^2 + 4 * x + m ≠ 3 * x^2 - 2 * x + 6) →
  m < -3 := by
sorry

end never_equal_implies_m_range_l932_93210


namespace half_vector_MN_l932_93281

/-- Given two vectors OM and ON in ℝ², prove that half of vector MN equals (-4, 1/2) -/
theorem half_vector_MN (OM ON : ℝ × ℝ) (h1 : OM = (3, -2)) (h2 : ON = (-5, -1)) :
  (1 / 2 : ℝ) • (ON - OM) = (-4, 1/2) := by sorry

end half_vector_MN_l932_93281


namespace other_root_of_complex_equation_l932_93239

theorem other_root_of_complex_equation (z : ℂ) :
  z^2 = -100 + 75*I ∧ z = 5 + 10*I → -5 - 10*I = -z :=
by sorry

end other_root_of_complex_equation_l932_93239


namespace prime_sequence_l932_93232

def f (p : ℕ) (x : ℕ) : ℕ := x^2 + x + p

theorem prime_sequence (p : ℕ) :
  (∀ k : ℕ, k ≤ Real.sqrt (p / 3) → Nat.Prime (f p k)) →
  (∀ n : ℕ, n ≤ p - 2 → Nat.Prime (f p n)) :=
sorry

end prime_sequence_l932_93232


namespace brad_read_more_books_l932_93223

def william_last_month : ℕ := 6
def brad_this_month : ℕ := 8

def brad_last_month : ℕ := 3 * william_last_month
def william_this_month : ℕ := 2 * brad_this_month

def william_total : ℕ := william_last_month + william_this_month
def brad_total : ℕ := brad_last_month + brad_this_month

theorem brad_read_more_books : brad_total = william_total + 4 := by
  sorry

end brad_read_more_books_l932_93223


namespace parabola_properties_l932_93271

-- Define the parabola
def parabola (x : ℝ) : ℝ := -2 * (x + 1)^2 - 3

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ := -2 * (x + 3)^2 + 1

-- Theorem statement
theorem parabola_properties :
  (∀ x : ℝ, parabola x ≤ parabola (-1)) ∧
  (parabola (-1) = -3) ∧
  (∀ x : ℝ, shifted_parabola x = parabola (x + 2) + 4) :=
sorry

end parabola_properties_l932_93271


namespace product_zero_implies_factor_zero_l932_93262

theorem product_zero_implies_factor_zero (a b : ℝ) : a * b = 0 → (a = 0 ∨ b = 0) := by
  contrapose
  intro h
  push_neg
  simp
  sorry

end product_zero_implies_factor_zero_l932_93262


namespace four_hearts_probability_l932_93295

-- Define a standard deck of cards
def standard_deck : ℕ := 52

-- Define the number of hearts in a standard deck
def hearts_in_deck : ℕ := 13

-- Define the number of cards we're drawing
def cards_drawn : ℕ := 4

-- Define the probability of drawing four hearts
def prob_four_hearts : ℚ := 2 / 95

-- Theorem statement
theorem four_hearts_probability :
  (hearts_in_deck.factorial / (hearts_in_deck - cards_drawn).factorial) /
  (standard_deck.factorial / (standard_deck - cards_drawn).factorial) = prob_four_hearts := by
  sorry

end four_hearts_probability_l932_93295


namespace used_car_selection_l932_93205

theorem used_car_selection (num_cars num_clients : ℕ) (selections_per_car : ℕ) :
  num_cars = 15 →
  num_clients = 15 →
  selections_per_car = 3 →
  (num_cars * selections_per_car) / num_clients = 3 :=
by sorry

end used_car_selection_l932_93205


namespace set_operation_result_l932_93253

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 3, 6}
def B : Set Nat := {1, 2}

theorem set_operation_result (C : Set Nat) (h : C ⊆ U) : 
  (C ∪ A) ∩ B = {2} := by sorry

end set_operation_result_l932_93253


namespace circle_parabola_tangent_radius_l932_93284

-- Define the parabola Γ
def Γ : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

-- Define the circle Ω with center (1, r) and radius r
def Ω (r : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - r)^2 = r^2}

-- State the theorem
theorem circle_parabola_tangent_radius :
  ∃! r : ℝ, r > 0 ∧
  (∃! p : ℝ × ℝ, p ∈ Γ ∩ Ω r) ∧
  (1, 0) ∈ Ω r ∧
  (∀ ε > 0, ∃ q : ℝ × ℝ, q.2 = -ε ∧ q ∉ Ω r) ∧
  r = 4 * Real.sqrt 3 / 9 := by
sorry

end circle_parabola_tangent_radius_l932_93284


namespace equation_solutions_l932_93206

theorem equation_solutions :
  (∃ x : ℚ, 2 * x - 9 = 4 * x ∧ x = -9/2) ∧
  (∃ x : ℚ, (5/2) * x - (7/3) * x = (4/3) * 5 - 5 ∧ x = 10) := by
  sorry

end equation_solutions_l932_93206


namespace min_product_xyz_l932_93268

theorem min_product_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hsum : x + y + z = 1) (hz_twice_y : z = 2 * y)
  (hx_le_2y : x ≤ 2 * y) (hy_le_2x : y ≤ 2 * x) (hz_le_2x : z ≤ 2 * x) :
  ∃ (min_val : ℝ), min_val = 8 / 243 ∧ x * y * z ≥ min_val :=
sorry

end min_product_xyz_l932_93268


namespace banana_distribution_l932_93214

theorem banana_distribution (total_children : ℕ) 
  (original_bananas_per_child : ℕ) 
  (extra_bananas_per_child : ℕ) : 
  total_children = 720 →
  original_bananas_per_child = 2 →
  extra_bananas_per_child = 2 →
  (total_children - (total_children * original_bananas_per_child) / 
   (original_bananas_per_child + extra_bananas_per_child)) = 360 := by
  sorry

end banana_distribution_l932_93214


namespace test_questions_count_l932_93221

theorem test_questions_count (total_questions : ℕ) 
  (correct_answers : ℕ) (final_score : ℚ) :
  correct_answers = 104 →
  final_score = 100 →
  (correct_answers : ℚ) + ((total_questions - correct_answers : ℕ) : ℚ) * (-1/4) = final_score →
  total_questions = 120 :=
by sorry

end test_questions_count_l932_93221


namespace paving_rate_calculation_l932_93213

/-- Calculates the rate of paving per square meter given room dimensions and total cost -/
theorem paving_rate_calculation (length width total_cost : ℝ) :
  length = 5.5 ∧ width = 3.75 ∧ total_cost = 12375 →
  total_cost / (length * width) = 600 := by
  sorry

#check paving_rate_calculation

end paving_rate_calculation_l932_93213


namespace three_points_on_circle_at_distance_from_line_l932_93212

theorem three_points_on_circle_at_distance_from_line :
  ∃! (points : Finset (ℝ × ℝ)), points.card = 3 ∧
  (∀ p ∈ points, p.1^2 + p.2^2 = 4 ∧
    (|p.1 - p.2 + Real.sqrt 2|) / Real.sqrt 2 = 1) :=
by sorry

end three_points_on_circle_at_distance_from_line_l932_93212


namespace probability_of_red_ball_l932_93272

theorem probability_of_red_ball (p_red_white p_red_black : ℝ) 
  (h1 : p_red_white = 0.58)
  (h2 : p_red_black = 0.62) :
  p_red_white + p_red_black - 1 = 0.2 := by
  sorry

end probability_of_red_ball_l932_93272


namespace man_downstream_speed_l932_93249

/-- Calculates the downstream speed given upstream and still water speeds -/
def downstream_speed (upstream : ℝ) (still_water : ℝ) : ℝ :=
  2 * still_water - upstream

theorem man_downstream_speed :
  let upstream_speed : ℝ := 12
  let still_water_speed : ℝ := 7
  downstream_speed upstream_speed still_water_speed = 14 := by
  sorry

end man_downstream_speed_l932_93249


namespace smallest_screw_count_screw_packs_problem_l932_93270

theorem smallest_screw_count : ℕ → Prop :=
  fun k => (∃ x y : ℕ, x ≠ y ∧ k = 10 * x ∧ k = 12 * y) ∧
           (∀ m : ℕ, m < k → ¬(∃ a b : ℕ, a ≠ b ∧ m = 10 * a ∧ m = 12 * b))

theorem screw_packs_problem : smallest_screw_count 60 := by
  sorry

end smallest_screw_count_screw_packs_problem_l932_93270


namespace keith_total_spent_l932_93248

-- Define the amounts spent on each item
def speakers_cost : ℚ := 136.01
def cd_player_cost : ℚ := 139.38
def tires_cost : ℚ := 112.46

-- Define the total amount spent
def total_spent : ℚ := speakers_cost + cd_player_cost + tires_cost

-- Theorem to prove
theorem keith_total_spent :
  total_spent = 387.85 :=
by sorry

end keith_total_spent_l932_93248


namespace sqrt_seven_plus_one_bounds_l932_93234

theorem sqrt_seven_plus_one_bounds :
  3 < Real.sqrt 7 + 1 ∧ Real.sqrt 7 + 1 < 4 :=
by
  sorry

end sqrt_seven_plus_one_bounds_l932_93234


namespace complex_w_values_l932_93279

theorem complex_w_values (z : ℂ) (w : ℂ) 
  (h1 : ∃ (r : ℝ), (1 + 3*I) * z = r)
  (h2 : w = z / (2 + I))
  (h3 : Complex.abs w = 5 * Real.sqrt 2) :
  w = 1 + 7*I ∨ w = -1 - 7*I := by
  sorry

end complex_w_values_l932_93279


namespace intersection_point_on_line_and_plane_l932_93291

/-- The line passing through the point (5, 2, -4) in the direction <-2, 0, -1> -/
def line (t : ℝ) : ℝ × ℝ × ℝ := (5 - 2*t, 2, -4 - t)

/-- The plane 2x - 5y + 4z + 24 = 0 -/
def plane (p : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := p
  2*x - 5*y + 4*z + 24 = 0

/-- The intersection point of the line and the plane -/
def intersection_point : ℝ × ℝ × ℝ := (3, 2, -5)

theorem intersection_point_on_line_and_plane :
  ∃ t : ℝ, line t = intersection_point ∧ plane intersection_point := by
  sorry


end intersection_point_on_line_and_plane_l932_93291


namespace x_sixth_geq_2a_minus_1_l932_93285

theorem x_sixth_geq_2a_minus_1 (x a : ℝ) (h : x^5 - x^3 + x = a) : x^6 ≥ 2*a - 1 := by
  sorry

end x_sixth_geq_2a_minus_1_l932_93285


namespace subsequence_theorem_l932_93217

theorem subsequence_theorem (seq : List ℕ) (h1 : seq.length = 101) 
  (h2 : ∀ n, n ∈ seq → 1 ≤ n ∧ n ≤ 101) 
  (h3 : ∀ n, 1 ≤ n ∧ n ≤ 101 → n ∈ seq) :
  ∃ subseq : List ℕ, subseq.length = 11 ∧ 
    (∀ i j, i < j → subseq.get ⟨i, by sorry⟩ < subseq.get ⟨j, by sorry⟩) ∨
    (∀ i j, i < j → subseq.get ⟨i, by sorry⟩ > subseq.get ⟨j, by sorry⟩) :=
sorry

end subsequence_theorem_l932_93217


namespace clothespin_count_total_clothespins_l932_93203

theorem clothespin_count (handkerchiefs : ℕ) (ropes : ℕ) : ℕ :=
  let ends_per_handkerchief := 2
  let pins_for_handkerchiefs := handkerchiefs * ends_per_handkerchief
  let pins_for_ropes := ropes
  pins_for_handkerchiefs + pins_for_ropes

theorem total_clothespins : clothespin_count 40 3 = 83 := by
  sorry

end clothespin_count_total_clothespins_l932_93203


namespace lottery_win_probability_l932_93299

def megaBallCount : ℕ := 30
def winnerBallCount : ℕ := 50
def winnerBallDrawCount : ℕ := 6

def lotteryWinProbability : ℚ :=
  2 / (megaBallCount * (winnerBallCount.choose winnerBallDrawCount))

theorem lottery_win_probability :
  lotteryWinProbability = 2 / 477621000 := by sorry

end lottery_win_probability_l932_93299


namespace missing_number_proof_l932_93277

theorem missing_number_proof : ∃ x : ℚ, (3/4 * 60 - 8/5 * 60 + x = 12) ∧ (x = 63) := by
  sorry

end missing_number_proof_l932_93277


namespace monotonic_decreasing_intervals_l932_93202

/-- The function f(x) = (x + 1) / x is monotonically decreasing on (-∞, 0) and (0, +∞) -/
theorem monotonic_decreasing_intervals (f : ℝ → ℝ) :
  (∀ x ≠ 0, f x = (x + 1) / x) →
  (StrictMonoOn f (Set.Iio 0) ∧ StrictMonoOn f (Set.Ioi 0)) :=
by sorry

end monotonic_decreasing_intervals_l932_93202


namespace square_root_of_four_l932_93228

theorem square_root_of_four :
  {x : ℝ | x ^ 2 = 4} = {2, -2} := by sorry

end square_root_of_four_l932_93228


namespace simplify_and_evaluate_l932_93224

theorem simplify_and_evaluate (a : ℝ) (h : a^2 - 7 = a) :
  (a - (2*a - 1) / a) / ((a - 1) / a^2) = 7 := by
  sorry

end simplify_and_evaluate_l932_93224


namespace intersection_of_A_and_B_l932_93209

-- Define set A
def A : Set ℝ := {x : ℝ | x * Real.sqrt (x^2 - 4) ≥ 0}

-- Define set B
def B : Set ℝ := {x : ℝ | |x - 1| + |x + 1| ≥ 2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {-2} ∪ Set.Ici 2 := by sorry

end intersection_of_A_and_B_l932_93209


namespace sqrt_problem_l932_93245

theorem sqrt_problem (h : Real.sqrt 100.4004 = 10.02) : Real.sqrt 1.004004 = 1.002 := by
  sorry

end sqrt_problem_l932_93245


namespace x_intercept_ratio_l932_93274

/-- Given two lines with the same non-zero y-intercept, prove that the ratio of their x-intercepts is 1/2 -/
theorem x_intercept_ratio (b s t : ℝ) (hb : b ≠ 0) : 
  (0 = 8 * s + b) → (0 = 4 * t + b) → s / t = 1 / 2 := by
  sorry

end x_intercept_ratio_l932_93274


namespace michael_gave_two_crates_l932_93294

/-- Calculates the number of crates Michael gave to Susan -/
def crates_given_to_susan (crates_tuesday : ℕ) (crates_thursday : ℕ) (eggs_per_crate : ℕ) (eggs_remaining : ℕ) : ℕ :=
  let total_crates := crates_tuesday + crates_thursday
  let total_eggs := total_crates * eggs_per_crate
  (total_eggs - eggs_remaining) / eggs_per_crate

theorem michael_gave_two_crates :
  crates_given_to_susan 6 5 30 270 = 2 := by
  sorry

end michael_gave_two_crates_l932_93294


namespace equation_solution_l932_93236

theorem equation_solution : ∃ x : ℤ, x * (x + 2) + 1 = 36 ∧ x = 5 := by
  sorry

end equation_solution_l932_93236


namespace product_of_fractions_l932_93243

theorem product_of_fractions (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end product_of_fractions_l932_93243


namespace triangle_side_length_l932_93286

theorem triangle_side_length (a b c : ℝ) (angle_C : ℝ) : 
  a = 3 → b = 5 → angle_C = 2 * π / 3 → c = 7 := by sorry

end triangle_side_length_l932_93286


namespace consecutive_binomial_coefficient_ratio_l932_93287

theorem consecutive_binomial_coefficient_ratio (n k : ℕ) : 
  (n.choose k : ℚ) / (n.choose (k + 1) : ℚ) = 1 / 3 ∧
  (n.choose (k + 1) : ℚ) / (n.choose (k + 2) : ℚ) = 1 / 2 →
  n + k = 12 := by
  sorry

end consecutive_binomial_coefficient_ratio_l932_93287


namespace f_not_prime_l932_93293

def f (n : ℕ+) : ℤ := n.val^4 - 380 * n.val^2 + 841

theorem f_not_prime : ∀ n : ℕ+, ¬ Nat.Prime (Int.natAbs (f n)) := by
  sorry

end f_not_prime_l932_93293


namespace carson_giant_slide_rides_l932_93265

/-- Represents the number of times Carson can ride the giant slide at the carnival -/
def giant_slide_rides (total_time minutes_per_hour roller_coaster_wait tilt_a_whirl_wait giant_slide_wait : ℕ)
  (roller_coaster_rides tilt_a_whirl_rides : ℕ) : ℕ :=
  let remaining_time := total_time * minutes_per_hour -
    (roller_coaster_wait * roller_coaster_rides + tilt_a_whirl_wait * tilt_a_whirl_rides)
  remaining_time / giant_slide_wait

/-- Theorem stating the number of times Carson can ride the giant slide -/
theorem carson_giant_slide_rides :
  giant_slide_rides 4 60 30 60 15 4 1 = 4 :=
sorry

end carson_giant_slide_rides_l932_93265


namespace residue_of_11_power_1234_mod_19_l932_93215

theorem residue_of_11_power_1234_mod_19 :
  (11 : ℤ)^1234 ≡ 16 [ZMOD 19] := by sorry

end residue_of_11_power_1234_mod_19_l932_93215


namespace point_on_line_with_vector_relation_l932_93297

/-- Given points A and B, if point P is on line AB and vector AB is twice vector AP, 
    then P has specific coordinates -/
theorem point_on_line_with_vector_relation (A B P : ℝ × ℝ) : 
  A = (2, 0) → 
  B = (4, 2) → 
  (∃ t : ℝ, P = (1 - t) • A + t • B) →  -- P is on line AB
  B - A = 2 • (P - A) → 
  P = (3, 1) := by
  sorry

end point_on_line_with_vector_relation_l932_93297


namespace purchase_plans_theorem_l932_93283

/-- Represents a purchasing plan for items A and B -/
structure PurchasePlan where
  a : ℕ  -- number of A items
  b : ℕ  -- number of B items

/-- Checks if a purchase plan satisfies all given conditions -/
def isValidPlan (p : PurchasePlan) : Prop :=
  p.a + p.b = 40 ∧
  p.a ≥ 3 * p.b ∧
  230 ≤ 8 * p.a + 2 * p.b ∧
  8 * p.a + 2 * p.b ≤ 266

/-- Calculates the total cost of a purchase plan -/
def totalCost (p : PurchasePlan) : ℕ :=
  8 * p.a + 2 * p.b

/-- Theorem stating the properties of valid purchase plans -/
theorem purchase_plans_theorem :
  ∃ (p1 p2 : PurchasePlan),
    isValidPlan p1 ∧
    isValidPlan p2 ∧
    p1 ≠ p2 ∧
    (∀ p, isValidPlan p → p = p1 ∨ p = p2) ∧
    (p1.a < p2.a → totalCost p1 < totalCost p2) :=
  sorry

end purchase_plans_theorem_l932_93283


namespace fraction_simplification_l932_93246

theorem fraction_simplification (a : ℝ) (h : a ≠ 1) :
  a / (a - 1) + 1 / (1 - a) = 1 := by
  sorry

end fraction_simplification_l932_93246


namespace taehyungs_mother_age_l932_93264

/-- Given the age differences and the age of Taehyung's younger brother, 
    prove that Taehyung's mother is 43 years old. -/
theorem taehyungs_mother_age :
  ∀ (taehyung_age brother_age mother_age : ℕ),
    taehyung_age = brother_age + 5 →
    brother_age = 7 →
    mother_age = taehyung_age + 31 →
    mother_age = 43 :=
by
  sorry

end taehyungs_mother_age_l932_93264


namespace nail_triangle_impossibility_l932_93250

/-- Given a triangle ACE on a wooden wall with nails of lengths AB = 1, CD = 2, EF = 4,
    prove that the distances between nail heads BD = √2, DF = √5, FB = √13 are impossible. -/
theorem nail_triangle_impossibility (AB CD EF BD DF FB : ℝ) :
  AB = 1 → CD = 2 → EF = 4 →
  BD = Real.sqrt 2 → DF = Real.sqrt 5 → FB = Real.sqrt 13 →
  ¬ (∃ (AC CE AE : ℝ), AC > 0 ∧ CE > 0 ∧ AE > 0 ∧
    AC + CE > AE ∧ CE + AE > AC ∧ AE + AC > CE) :=
by sorry

end nail_triangle_impossibility_l932_93250


namespace min_selection_for_tenfold_l932_93204

theorem min_selection_for_tenfold (n : ℕ) (h : n = 2020) :
  ∃ k : ℕ, k = 203 ∧
  (∀ S : Finset ℕ, S.card < k → S ⊆ Finset.range (n + 1) →
    ¬∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a = 10 * b) ∧
  (∃ S : Finset ℕ, S.card = k ∧ S ⊆ Finset.range (n + 1) ∧
    ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a = 10 * b) :=
by sorry

end min_selection_for_tenfold_l932_93204


namespace probability_of_prime_on_die_l932_93237

/-- A standard die with six faces -/
def StandardDie := Finset.range 6

/-- The set of prime numbers on a standard die -/
def PrimeNumbersOnDie : Finset Nat := {2, 3, 5}

/-- The probability of rolling a prime number on a standard die -/
def ProbabilityOfPrime : ℚ := (PrimeNumbersOnDie.card : ℚ) / (StandardDie.card : ℚ)

/-- The given probability expression -/
def GivenProbability (a : ℕ) : ℚ := (a : ℚ) / 72

theorem probability_of_prime_on_die (a : ℕ) :
  GivenProbability a = ProbabilityOfPrime → a = 36 := by
  sorry

end probability_of_prime_on_die_l932_93237


namespace lemons_for_new_recipe_l932_93267

/-- Represents the number of lemons per gallon in the original recipe -/
def original_lemons_per_gallon : ℚ := 36 / 48

/-- Represents the additional lemons per gallon in the new recipe -/
def additional_lemons_per_gallon : ℚ := 2 / 6

/-- Represents the number of gallons we want to make -/
def gallons_to_make : ℚ := 18

/-- Theorem stating that 18 gallons of the new recipe requires 19.5 lemons -/
theorem lemons_for_new_recipe : 
  (original_lemons_per_gallon + additional_lemons_per_gallon) * gallons_to_make = 19.5 := by
  sorry

end lemons_for_new_recipe_l932_93267


namespace sqrt_6_simplest_l932_93222

-- Define the criteria for simplest square root form
def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y : ℚ, x ≠ y^2 ∧ (∀ z : ℕ, z > 1 → ¬ (∃ w : ℕ, x = z * w^2))

-- Define the set of square roots to compare
def sqrt_set : Set ℝ := {Real.sqrt 0.2, Real.sqrt (1/2), Real.sqrt 6, Real.sqrt 12}

-- Theorem statement
theorem sqrt_6_simplest :
  ∀ x ∈ sqrt_set, x ≠ Real.sqrt 6 → ¬(is_simplest_sqrt x) ∧ is_simplest_sqrt (Real.sqrt 6) :=
sorry

end sqrt_6_simplest_l932_93222


namespace log_exponent_simplification_l932_93207

theorem log_exponent_simplification :
  Real.log 2 + Real.log 5 - 42 * (8 ^ (1/4 : ℝ)) - (2017 ^ (0 : ℝ)) = -2 :=
by sorry

end log_exponent_simplification_l932_93207


namespace equal_quantities_after_transfer_l932_93252

def container_problem (initial_A initial_B initial_C transfer : ℝ) : Prop :=
  let final_B := initial_B + transfer
  let final_C := initial_C - transfer
  initial_A = 1184 ∧
  initial_B = 0.375 * initial_A ∧
  initial_C = initial_A - initial_B ∧
  transfer = 148 →
  final_B = final_C

theorem equal_quantities_after_transfer :
  ∃ (initial_A initial_B initial_C transfer : ℝ),
    container_problem initial_A initial_B initial_C transfer :=
  sorry

end equal_quantities_after_transfer_l932_93252


namespace jenny_lasagna_profit_l932_93244

/-- Calculate Jenny's profit from selling lasagna pans -/
def jennys_profit (cost_per_pan : ℝ) (price_per_pan : ℝ) (num_pans : ℕ) : ℝ :=
  (price_per_pan * num_pans) - (cost_per_pan * num_pans)

/-- Theorem: Jenny's profit is $300 given the problem conditions -/
theorem jenny_lasagna_profit :
  jennys_profit 10 25 20 = 300 := by
  sorry

end jenny_lasagna_profit_l932_93244


namespace absolute_value_simplification_l932_93233

theorem absolute_value_simplification : |(-4^2 + 6)| = 10 := by
  sorry

end absolute_value_simplification_l932_93233


namespace odd_function_product_nonpositive_l932_93282

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- For any odd function f: ℝ → ℝ, f(x)f(-x) ≤ 0 for all x ∈ ℝ -/
theorem odd_function_product_nonpositive (f : ℝ → ℝ) (h : IsOdd f) :
  ∀ x, f x * f (-x) ≤ 0 :=
by sorry

end odd_function_product_nonpositive_l932_93282


namespace line_equivalence_l932_93227

/-- Given a line in the form (2, -1) · ((x, y) - (1, -3)) = 0, prove it's equivalent to y = 2x - 5 --/
theorem line_equivalence :
  ∀ (x y : ℝ), (2 : ℝ) * (x - 1) + (-1 : ℝ) * (y + 3) = 0 ↔ y = 2 * x - 5 := by
  sorry

end line_equivalence_l932_93227


namespace actual_distance_traveled_l932_93220

/-- Given that if a person walks at 16 km/hr instead of 12 km/hr, they would have walked 20 km more,
    prove that the actual distance traveled is 60 km. -/
theorem actual_distance_traveled (D : ℝ) : 
  (D / 12 = (D + 20) / 16) → D = 60 := by
  sorry

end actual_distance_traveled_l932_93220


namespace max_cross_section_area_l932_93263

/-- A regular hexagonal prism with side length 8 and vertical edges parallel to the z-axis -/
structure HexagonalPrism where
  side_length : ℝ
  side_length_eq : side_length = 8

/-- The plane that cuts the prism -/
def cutting_plane (x y z : ℝ) : Prop :=
  5 * x - 8 * y + 3 * z = 40

/-- The cross-section formed by cutting the prism with the plane -/
def cross_section (p : HexagonalPrism) (x y z : ℝ) : Prop :=
  cutting_plane x y z

/-- The area of the cross-section -/
noncomputable def cross_section_area (p : HexagonalPrism) : ℝ :=
  sorry

/-- The theorem stating that the maximum area of the cross-section is 144√3 -/
theorem max_cross_section_area (p : HexagonalPrism) :
    ∃ (a : ℝ), cross_section_area p = a ∧ a ≤ 144 * Real.sqrt 3 ∧
    ∀ (b : ℝ), cross_section_area p ≤ b → b ≥ 144 * Real.sqrt 3 :=
  sorry

end max_cross_section_area_l932_93263


namespace midpoint_coordinate_sum_l932_93298

/-- Given a line segment with one endpoint at (7, 4) and midpoint at (5, -8),
    the sum of coordinates of the other endpoint is -17. -/
theorem midpoint_coordinate_sum :
  ∀ x y : ℝ,
  (5 : ℝ) = (7 + x) / 2 →
  (-8 : ℝ) = (4 + y) / 2 →
  x + y = -17 := by
sorry

end midpoint_coordinate_sum_l932_93298


namespace manufacturing_cost_of_shoe_l932_93273

/-- The manufacturing cost of a shoe given specific conditions -/
theorem manufacturing_cost_of_shoe (transportation_cost : ℚ) (selling_price : ℚ) (gain_percentage : ℚ) :
  transportation_cost = 500 / 100 →
  selling_price = 270 →
  gain_percentage = 20 / 100 →
  ∃ (manufacturing_cost : ℚ),
    manufacturing_cost = selling_price / (1 + gain_percentage) - transportation_cost ∧
    manufacturing_cost = 220 := by
  sorry

end manufacturing_cost_of_shoe_l932_93273


namespace quadratic_function_range_l932_93278

/-- Given a quadratic function f(x) = ax^2 + bx, prove that if 1 ≤ f(-1) ≤ 2 and 2 ≤ f(1) ≤ 4, then 3 ≤ f(-2) ≤ 12. -/
theorem quadratic_function_range (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^2 + b * x
  (1 ≤ f (-1) ∧ f (-1) ≤ 2) ∧ (2 ≤ f 1 ∧ f 1 ≤ 4) →
  3 ≤ f (-2) ∧ f (-2) ≤ 12 := by
  sorry

end quadratic_function_range_l932_93278


namespace alpha_value_at_negative_four_l932_93201

/-- Given that α is inversely proportional to β², prove that α = 5/4 when β = -4, 
    given that α = 5 when β = 2. -/
theorem alpha_value_at_negative_four (α β : ℝ) (k : ℝ) 
  (h1 : ∀ β, α * β^2 = k)  -- α is inversely proportional to β²
  (h2 : α = 5 ∧ β = 2 → k = 20)  -- α = 5 when β = 2
  : β = -4 → α = 5/4 := by
  sorry

end alpha_value_at_negative_four_l932_93201


namespace fourth_dog_weight_l932_93292

theorem fourth_dog_weight (y : ℝ) :
  let dog1 : ℝ := 25
  let dog2 : ℝ := 31
  let dog3 : ℝ := 35
  let dog4 : ℝ := x
  let dog5 : ℝ := y
  (dog1 + dog2 + dog3 + dog4) / 4 = (dog1 + dog2 + dog3 + dog4 + dog5) / 5 →
  x = -91 - 5 * y :=
by
  sorry

end fourth_dog_weight_l932_93292


namespace expression_equality_l932_93241

theorem expression_equality : 6 * 111 - 2 * 111 = 444 := by
  sorry

end expression_equality_l932_93241


namespace simplify_fraction_l932_93231

theorem simplify_fraction : 
  (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 := by
sorry

end simplify_fraction_l932_93231


namespace numerator_smaller_than_a_l932_93230

theorem numerator_smaller_than_a (a b n : ℕ) (h1 : a ≠ 1) (h2 : b > 0) 
  (h3 : Nat.gcd a b = 1) (h4 : (n : ℚ)⁻¹ > a / b) (h5 : a / b > (n + 1 : ℚ)⁻¹) :
  ∃ (p q : ℕ), q > 0 ∧ Nat.gcd p q = 1 ∧ 
  (a : ℚ) / b - (n + 1 : ℚ)⁻¹ = (p : ℚ) / q ∧ p < a := by
  sorry

end numerator_smaller_than_a_l932_93230


namespace can_determine_ten_gram_coins_can_determine_coin_weight_l932_93219

/-- Represents the weight of coins in grams -/
inductive CoinWeight
  | Ten
  | Eleven
  | Twelve
  | Thirteen
  | Fourteen

/-- Represents a bag of coins -/
structure Bag where
  weight : CoinWeight
  count : Nat
  h_count : count = 100

/-- Represents the result of a weighing -/
inductive WeighingResult
  | Equal
  | LeftHeavier
  | RightHeavier

/-- Represents a collection of bags -/
structure BagCollection where
  bags : Fin 5 → Bag
  h_distinct : ∀ i j, i ≠ j → (bags i).weight ≠ (bags j).weight

/-- Function to perform a weighing -/
noncomputable def weigh (left right : List Nat) : WeighingResult :=
  sorry

/-- Theorem stating that it's possible to determine if a specific bag contains 10g coins with one weighing -/
theorem can_determine_ten_gram_coins (bags : BagCollection) (pointed : Fin 5) : 
  ∃ (left right : List Nat), 
    (∀ n ∈ left ∪ right, n ≤ 100) ∧ 
    (weigh left right = WeighingResult.Equal ↔ (bags.bags pointed).weight = CoinWeight.Ten) :=
  sorry

/-- Theorem stating that it's possible to determine the weight of coins in a specific bag with at most two weighings -/
theorem can_determine_coin_weight (bags : BagCollection) (pointed : Fin 5) :
  ∃ (left1 right1 left2 right2 : List Nat),
    (∀ n ∈ left1 ∪ right1 ∪ left2 ∪ right2, n ≤ 100) ∧
    (∃ f : WeighingResult → WeighingResult → CoinWeight,
      f (weigh left1 right1) (weigh left2 right2) = (bags.bags pointed).weight) :=
  sorry

end can_determine_ten_gram_coins_can_determine_coin_weight_l932_93219


namespace remainder_of_2_pow_33_mod_9_l932_93226

theorem remainder_of_2_pow_33_mod_9 : 2^33 % 9 = 8 := by sorry

end remainder_of_2_pow_33_mod_9_l932_93226


namespace solid_figures_count_l932_93266

-- Define the list of shapes
inductive Shape
  | Circle
  | Square
  | Cone
  | Cuboid
  | LineSegment
  | Sphere
  | TriangularPrism
  | RightAngledTriangle

-- Define a function to determine if a shape is solid
def isSolid (s : Shape) : Bool :=
  match s with
  | Shape.Cone => true
  | Shape.Cuboid => true
  | Shape.Sphere => true
  | Shape.TriangularPrism => true
  | _ => false

-- Define the list of shapes
def shapeList : List Shape := [
  Shape.Circle,
  Shape.Square,
  Shape.Cone,
  Shape.Cuboid,
  Shape.LineSegment,
  Shape.Sphere,
  Shape.TriangularPrism,
  Shape.RightAngledTriangle
]

-- Theorem: The number of solid figures in the list is 4
theorem solid_figures_count :
  (shapeList.filter isSolid).length = 4 := by
  sorry

end solid_figures_count_l932_93266


namespace apple_consumption_theorem_l932_93218

/-- Represents the apple's division and consumption rates -/
structure AppleConsumption where
  above_water : ℚ
  below_water : ℚ
  fish_rate : ℚ
  bird_rate : ℚ

/-- Theorem stating the portions of apple eaten by fish and bird -/
theorem apple_consumption_theorem (a : AppleConsumption) 
  (h1 : a.above_water = 1/5)
  (h2 : a.below_water = 4/5)
  (h3 : a.fish_rate = 120)
  (h4 : a.bird_rate = 60) :
  ∃ (fish_portion bird_portion : ℚ),
    fish_portion = 2/3 ∧ 
    bird_portion = 1/3 ∧
    fish_portion + bird_portion = 1 :=
sorry

end apple_consumption_theorem_l932_93218


namespace n_value_for_specific_x_y_l932_93251

theorem n_value_for_specific_x_y : ∀ (x y n : ℝ), 
  x = 3 → y = -1 → n = x - y^(x-y) → n = 2 := by
  sorry

end n_value_for_specific_x_y_l932_93251


namespace outfit_combinations_l932_93257

theorem outfit_combinations : 
  let blue_shirts : ℕ := 6
  let green_shirts : ℕ := 4
  let pants : ℕ := 7
  let blue_hats : ℕ := 9
  let green_hats : ℕ := 7
  let blue_shirt_green_hat := blue_shirts * pants * green_hats
  let green_shirt_blue_hat := green_shirts * pants * blue_hats
  blue_shirt_green_hat + green_shirt_blue_hat = 546 := by
  sorry

end outfit_combinations_l932_93257


namespace remaining_surface_area_after_removal_l932_93256

/-- The remaining surface area of a cube after removing a smaller cube from its corner --/
theorem remaining_surface_area_after_removal (a b : ℝ) (ha : a > 0) (hb : b > 0) (hba : b < 3*a) :
  6 * (3*a)^2 - 3 * b^2 + 3 * b^2 = 54 * a^2 := by
  sorry

#check remaining_surface_area_after_removal

end remaining_surface_area_after_removal_l932_93256


namespace k_negative_sufficient_not_necessary_l932_93296

-- Define the condition for the equation to represent a hyperbola
def is_hyperbola (k : ℝ) : Prop := k * (k - 1) > 0

-- State the theorem
theorem k_negative_sufficient_not_necessary :
  (∀ k : ℝ, k < 0 → is_hyperbola k) ∧
  (∃ k : ℝ, ¬(k < 0) ∧ is_hyperbola k) :=
sorry

end k_negative_sufficient_not_necessary_l932_93296


namespace box_volume_l932_93255

/-- The volume of a box formed by cutting squares from corners of a square sheet -/
theorem box_volume (sheet_side : ℝ) (corner_cut : ℝ) : 
  sheet_side = 12 → corner_cut = 2 → 
  (sheet_side - 2 * corner_cut) * (sheet_side - 2 * corner_cut) * corner_cut = 128 :=
by
  sorry

end box_volume_l932_93255
