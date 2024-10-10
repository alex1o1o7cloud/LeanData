import Mathlib

namespace license_plate_count_l1572_157260

/-- The number of possible first letters for the license plate -/
def first_letter_choices : ℕ := 3

/-- The number of choices for each digit position -/
def digit_choices : ℕ := 10

/-- The number of digit positions after the letter -/
def num_digits : ℕ := 5

/-- The total number of possible license plates -/
def total_license_plates : ℕ := first_letter_choices * digit_choices ^ num_digits

theorem license_plate_count : total_license_plates = 300000 := by
  sorry

end license_plate_count_l1572_157260


namespace sum_specific_sequence_l1572_157222

theorem sum_specific_sequence : 
  (1000 + 20 + 1000 + 30 + 1000 + 40 + 1000 + 10) = 4100 := by
  sorry

end sum_specific_sequence_l1572_157222


namespace soccer_team_captains_l1572_157273

theorem soccer_team_captains (n : ℕ) (k : ℕ) (h1 : n = 14) (h2 : k = 3) :
  Nat.choose n k = 364 := by
  sorry

end soccer_team_captains_l1572_157273


namespace negation_of_existence_negation_of_cubic_equation_l1572_157275

theorem negation_of_existence (f : ℝ → ℝ) :
  (¬ ∃ x, f x = 0) ↔ (∀ x, f x ≠ 0) := by sorry

theorem negation_of_cubic_equation :
  (¬ ∃ x : ℝ, x^3 - 2*x + 1 = 0) ↔ (∀ x : ℝ, x^3 - 2*x + 1 ≠ 0) := by sorry

end negation_of_existence_negation_of_cubic_equation_l1572_157275


namespace necessary_but_not_sufficient_condition_l1572_157259

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (0 < x ∧ x < 4) → (x^2 - 3*x < 0 → 0 < x ∧ x < 4) ∧ ¬(0 < x ∧ x < 4 → x^2 - 3*x < 0) :=
sorry

end necessary_but_not_sufficient_condition_l1572_157259


namespace triangle_area_l1572_157247

/-- The area of a triangle with base 15 and height p is equal to 15p/2 -/
theorem triangle_area (p : ℝ) : 
  (1/2 : ℝ) * 15 * p = 15 * p / 2 := by sorry

end triangle_area_l1572_157247


namespace cement_mixture_weight_l1572_157290

theorem cement_mixture_weight :
  ∀ (total_weight : ℝ),
    (1/5 : ℝ) * total_weight +     -- Sand
    (3/4 : ℝ) * total_weight +     -- Water
    6 = total_weight →             -- Gravel
    total_weight = 120 := by
  sorry

end cement_mixture_weight_l1572_157290


namespace second_valid_number_is_068_l1572_157214

/-- Represents a random number table as a list of natural numbers. -/
def RandomNumberTable : List ℕ := [84, 42, 17, 53, 31, 57, 24, 55, 06, 88, 77, 04, 74, 47, 67, 21, 76, 33, 50, 25, 83, 92, 12, 06, 76]

/-- Represents the total number of units. -/
def TotalUnits : ℕ := 200

/-- Represents the starting column in the random number table. -/
def StartColumn : ℕ := 5

/-- Checks if a number is valid (i.e., between 1 and TotalUnits). -/
def isValidNumber (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ TotalUnits

/-- Finds the nth valid number in the random number table. -/
def nthValidNumber (n : ℕ) : ℕ := sorry

/-- The main theorem stating that the second valid number is 068. -/
theorem second_valid_number_is_068 : nthValidNumber 2 = 68 := by sorry

end second_valid_number_is_068_l1572_157214


namespace books_in_history_section_l1572_157233

/-- Calculates the number of books shelved in the history section. -/
def books_shelved_in_history (initial_books : ℕ) (fiction_books : ℕ) (children_books : ℕ) 
  (misplaced_books : ℕ) (books_left : ℕ) : ℕ :=
  initial_books - fiction_books - children_books + misplaced_books - books_left

/-- Theorem stating the number of books shelved in the history section. -/
theorem books_in_history_section :
  books_shelved_in_history 51 19 8 4 16 = 12 := by
  sorry

#eval books_shelved_in_history 51 19 8 4 16

end books_in_history_section_l1572_157233


namespace perpendicular_lines_m_values_l1572_157285

theorem perpendicular_lines_m_values (m : ℝ) : 
  (∀ x y : ℝ, (m + 2) * x + 3 * m * y + 7 = 0 ∧ 
               (m - 2) * x + (m + 2) * y - 5 = 0 → 
               ((m + 2) * (m - 2) + 3 * m * (m + 2) = 0)) → 
  m = 1/2 ∨ m = -2 := by
sorry

end perpendicular_lines_m_values_l1572_157285


namespace baseball_games_per_month_l1572_157288

theorem baseball_games_per_month 
  (total_games : ℕ) 
  (season_months : ℕ) 
  (h1 : total_games = 14) 
  (h2 : season_months = 2) : 
  total_games / season_months = 7 := by
sorry

end baseball_games_per_month_l1572_157288


namespace probability_b_greater_than_a_l1572_157207

def A : Finset ℕ := {1, 2, 3, 4, 5}
def B : Finset ℕ := {1, 2, 3}

def favorable_outcomes : Finset (ℕ × ℕ) :=
  (A.product B).filter (fun p => p.2 > p.1)

theorem probability_b_greater_than_a :
  (favorable_outcomes.card : ℚ) / ((A.card * B.card) : ℚ) = 1 / 5 := by
  sorry

end probability_b_greater_than_a_l1572_157207


namespace inverse_proportion_through_point_l1572_157297

/-- An inverse proportion function passing through (-2, 3) has the equation y = -6/x -/
theorem inverse_proportion_through_point (f : ℝ → ℝ) :
  (∀ x ≠ 0, ∃ k, f x = k / x) →  -- f is an inverse proportion function
  f (-2) = 3 →                   -- f passes through the point (-2, 3)
  ∀ x ≠ 0, f x = -6 / x :=       -- The equation of f is y = -6/x
by sorry

end inverse_proportion_through_point_l1572_157297


namespace complement_union_M_N_l1572_157255

open Set

-- Define the universal set as ℝ
universe u
variable {α : Type u}

-- Define sets M and N
def M : Set ℝ := {x | x ≤ 0}
def N : Set ℝ := {x | x > 2}

-- State the theorem
theorem complement_union_M_N :
  (M ∪ N)ᶜ = {x : ℝ | 0 < x ∧ x ≤ 2} :=
by sorry

end complement_union_M_N_l1572_157255


namespace complex_modulus_one_minus_i_l1572_157216

theorem complex_modulus_one_minus_i :
  let z : ℂ := 1 - I
  Complex.abs z = Real.sqrt 2 := by sorry

end complex_modulus_one_minus_i_l1572_157216


namespace expression_value_l1572_157263

theorem expression_value : 12 * (1 / 15) * 30 - 6 = 18 := by
  sorry

end expression_value_l1572_157263


namespace quadratic_inequality_solution_sets_l1572_157279

theorem quadratic_inequality_solution_sets (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) 
  (h₁ : a₁ ≠ 0) (h₂ : b₁ ≠ 0) (h₃ : c₁ ≠ 0) 
  (h₄ : a₂ ≠ 0) (h₅ : b₂ ≠ 0) (h₆ : c₂ ≠ 0) :
  ¬(((a₁ / a₂ = b₁ / b₂) ∧ (b₁ / b₂ = c₁ / c₂)) ↔
    ({x : ℝ | a₁ * x^2 + b₁ * x + c₁ > 0} = {x : ℝ | a₂ * x^2 + b₂ * x + c₂ > 0})) :=
by sorry

end quadratic_inequality_solution_sets_l1572_157279


namespace area_to_paint_is_15_l1572_157277

/-- The area of the wall to be painted -/
def area_to_paint (wall_length wall_width blackboard_length blackboard_width : ℝ) : ℝ :=
  wall_length * wall_width - blackboard_length * blackboard_width

/-- Theorem: The area to be painted is 15 square meters -/
theorem area_to_paint_is_15 :
  area_to_paint 6 3 3 1 = 15 := by
  sorry

end area_to_paint_is_15_l1572_157277


namespace joyces_property_size_l1572_157202

theorem joyces_property_size (new_property_size old_property_size pond_size suitable_land : ℝ) : 
  new_property_size = 10 * old_property_size →
  pond_size = 1 →
  suitable_land = 19 →
  new_property_size = suitable_land + pond_size →
  old_property_size = 2 := by
sorry

end joyces_property_size_l1572_157202


namespace sophomores_in_sample_l1572_157245

-- Define the total number of students in each grade
def freshmen : ℕ := 400
def sophomores : ℕ := 600
def juniors : ℕ := 500

-- Define the total sample size
def sample_size : ℕ := 100

-- Theorem to prove
theorem sophomores_in_sample :
  (sophomores * sample_size) / (freshmen + sophomores + juniors) = 40 := by
  sorry

end sophomores_in_sample_l1572_157245


namespace doubling_base_and_exponent_l1572_157230

theorem doubling_base_and_exponent (a b y : ℝ) (ha : a > 0) (hb : b > 0) (hy : y > 0) :
  (2*a)^(2*b) = a^b * y^b → y = 4*a :=
by sorry

end doubling_base_and_exponent_l1572_157230


namespace one_line_passes_through_trisection_point_l1572_157209

-- Define the points
def A : ℝ × ℝ := (-3, 6)
def B : ℝ × ℝ := (6, -3)
def P : ℝ × ℝ := (2, 3)

-- Define the trisection points
def T₁ : ℝ × ℝ := (0, 3)
def T₂ : ℝ × ℝ := (3, 0)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 3 * x + y - 9 = 0

-- Theorem statement
theorem one_line_passes_through_trisection_point :
  ∃ T, (T = T₁ ∨ T = T₂) ∧ 
       line_equation P.1 P.2 ∧
       line_equation T.1 T.2 :=
sorry

end one_line_passes_through_trisection_point_l1572_157209


namespace cricket_bat_cost_price_l1572_157208

theorem cricket_bat_cost_price 
  (profit_A_to_B : ℝ) 
  (profit_B_to_C : ℝ) 
  (price_C : ℝ) 
  (h1 : profit_A_to_B = 0.20)
  (h2 : profit_B_to_C = 0.25)
  (h3 : price_C = 234) :
  ∃ (cost_price_A : ℝ), cost_price_A = 156 ∧
    price_C = (1 + profit_B_to_C) * ((1 + profit_A_to_B) * cost_price_A) := by
  sorry

end cricket_bat_cost_price_l1572_157208


namespace cube_of_product_l1572_157249

theorem cube_of_product (a b : ℝ) : (-2 * a * b^2)^3 = -8 * a^3 * b^6 := by
  sorry

end cube_of_product_l1572_157249


namespace rectangular_field_area_difference_l1572_157267

theorem rectangular_field_area_difference : 
  let stan_length : ℕ := 30
  let stan_width : ℕ := 50
  let isla_length : ℕ := 35
  let isla_width : ℕ := 55
  let stan_area := stan_length * stan_width
  let isla_area := isla_length * isla_width
  isla_area - stan_area = 425 ∧ isla_area > stan_area := by
sorry

end rectangular_field_area_difference_l1572_157267


namespace quadratic_roots_sum_l1572_157299

theorem quadratic_roots_sum (u v : ℝ) : 
  (u^2 - 5*u + 6 = 0) → 
  (v^2 - 5*v + 6 = 0) → 
  u^2 + v^2 + u + v = 18 := by
sorry

end quadratic_roots_sum_l1572_157299


namespace total_pencils_l1572_157262

/-- Given 4.0 pencil boxes, each filled with 648.0 pencils, prove that the total number of pencils is 2592.0 -/
theorem total_pencils (num_boxes : Float) (pencils_per_box : Float) 
  (h1 : num_boxes = 4.0) 
  (h2 : pencils_per_box = 648.0) : 
  num_boxes * pencils_per_box = 2592.0 := by
  sorry

end total_pencils_l1572_157262


namespace total_fish_count_l1572_157200

/-- Given 261 fishbowls with 23 fish each, prove that the total number of fish is 6003. -/
theorem total_fish_count (num_fishbowls : ℕ) (fish_per_bowl : ℕ) 
  (h1 : num_fishbowls = 261) 
  (h2 : fish_per_bowl = 23) : 
  num_fishbowls * fish_per_bowl = 6003 := by
  sorry

end total_fish_count_l1572_157200


namespace sum_consecutive_odd_integers_divisible_by_16_l1572_157248

theorem sum_consecutive_odd_integers_divisible_by_16 :
  let start := 2101
  let count := 15
  let sequence := List.range count |>.map (fun i => start + 2 * i)
  sequence.sum % 16 = 0 := by
  sorry

end sum_consecutive_odd_integers_divisible_by_16_l1572_157248


namespace loaves_delivered_correct_evening_delivery_l1572_157235

/-- Given the initial number of loaves, the number of loaves sold, and the final number of loaves,
    calculate the number of loaves delivered. -/
theorem loaves_delivered (initial : ℕ) (sold : ℕ) (final : ℕ) :
  final - (initial - sold) = final - initial + sold :=
by sorry

/-- The number of loaves delivered in the evening -/
def evening_delivery : ℕ := 2215 - (2355 - 629)

theorem correct_evening_delivery : evening_delivery = 489 :=
by sorry

end loaves_delivered_correct_evening_delivery_l1572_157235


namespace rhett_salary_l1572_157298

/-- Rhett's monthly salary calculation --/
theorem rhett_salary (monthly_rent : ℝ) (tax_rate : ℝ) (late_payments : ℕ) 
  (after_tax_fraction : ℝ) (salary : ℝ) :
  monthly_rent = 1350 →
  tax_rate = 0.1 →
  late_payments = 2 →
  after_tax_fraction = 3/5 →
  after_tax_fraction * (1 - tax_rate) * salary = late_payments * monthly_rent →
  salary = 5000 := by
sorry

end rhett_salary_l1572_157298


namespace robotics_club_subjects_l1572_157228

theorem robotics_club_subjects (total : ℕ) (math : ℕ) (physics : ℕ) (cs : ℕ) 
  (math_physics : ℕ) (math_cs : ℕ) (physics_cs : ℕ) (all_three : ℕ) : 
  total = 60 ∧ 
  math = 42 ∧ 
  physics = 35 ∧ 
  cs = 15 ∧ 
  math_physics = 25 ∧ 
  math_cs = 10 ∧ 
  physics_cs = 5 ∧ 
  all_three = 4 → 
  total - (math + physics + cs - math_physics - math_cs - physics_cs + all_three) = 0 :=
by sorry

end robotics_club_subjects_l1572_157228


namespace min_nSn_value_l1572_157226

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n : ℕ, S n = n / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1))

/-- The main theorem -/
theorem min_nSn_value (seq : ArithmeticSequence) 
    (h1 : seq.S 10 = 0) 
    (h2 : seq.S 15 = 25) : 
  ∃ n : ℕ, ∀ m : ℕ, n * seq.S n ≤ m * seq.S m ∧ n * seq.S n = -49 := by
  sorry

end min_nSn_value_l1572_157226


namespace stewart_farm_ratio_l1572_157234

def stewart_farm (horse_food_per_day : ℕ) (total_horse_food : ℕ) (num_sheep : ℕ) : Prop :=
  ∃ (num_horses : ℕ),
    horse_food_per_day * num_horses = total_horse_food ∧
    (num_sheep : ℚ) / num_horses = 5 / 7

theorem stewart_farm_ratio :
  stewart_farm 230 12880 40 := by
  sorry

end stewart_farm_ratio_l1572_157234


namespace lucy_fish_count_lucy_fish_proof_l1572_157204

theorem lucy_fish_count : ℕ → Prop :=
  fun current_fish =>
    (current_fish + 68 = 280) → (current_fish = 212)

-- Proof
theorem lucy_fish_proof : lucy_fish_count 212 := by
  sorry

end lucy_fish_count_lucy_fish_proof_l1572_157204


namespace largest_divisor_of_n_l1572_157224

theorem largest_divisor_of_n (n : ℕ+) (h : 450 ∣ n^2) : 
  ∀ d : ℕ, d ∣ n → d ≤ 30 ∧ 30 ∣ n := by
  sorry

end largest_divisor_of_n_l1572_157224


namespace wire_service_reporters_l1572_157264

theorem wire_service_reporters (total : ℕ) (h_total : total > 0) :
  let local_politics := (18 : ℚ) / 100 * total
  let no_politics := (70 : ℚ) / 100 * total
  let cover_politics := total - no_politics
  let cover_not_local := cover_politics - local_politics
  (cover_not_local / cover_politics) = (2 : ℚ) / 5 := by
sorry

end wire_service_reporters_l1572_157264


namespace arithmetic_sequence_product_l1572_157227

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  (∀ n : ℕ, b (n + 1) > b n) →
  (∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d) →
  b 5 * b 6 = 14 →
  b 4 * b 7 = -324 ∨ b 4 * b 7 = -36 :=
by sorry

end arithmetic_sequence_product_l1572_157227


namespace green_percentage_is_25_l1572_157292

def amber_pieces : ℕ := 20
def green_pieces : ℕ := 35
def clear_pieces : ℕ := 85

def total_pieces : ℕ := amber_pieces + green_pieces + clear_pieces

def percentage_green : ℚ := (green_pieces : ℚ) / (total_pieces : ℚ) * 100

theorem green_percentage_is_25 : percentage_green = 25 := by
  sorry

end green_percentage_is_25_l1572_157292


namespace oldest_child_age_l1572_157212

theorem oldest_child_age (age1 age2 : ℕ) (avg : ℚ) :
  age1 = 6 →
  age2 = 9 →
  avg = 10 →
  (age1 + age2 + (3 * avg - age1 - age2 : ℚ) : ℚ) / 3 = avg →
  3 * avg - age1 - age2 = 15 :=
by sorry

end oldest_child_age_l1572_157212


namespace evaluate_expression_l1572_157253

theorem evaluate_expression (x y : ℝ) (hx : x = 2) (hy : y = 5) : 
  y * (2 * y - 5 * x) = 0 := by
  sorry

end evaluate_expression_l1572_157253


namespace target_probabilities_l1572_157211

/-- The probability of hitting the target for both A and B -/
def p : ℝ := 0.6

/-- The probability that both A and B hit the target -/
def prob_both_hit : ℝ := p * p

/-- The probability that exactly one of A and B hits the target -/
def prob_one_hit : ℝ := 2 * p * (1 - p)

/-- The probability that at least one of A and B hits the target -/
def prob_at_least_one_hit : ℝ := 1 - (1 - p) * (1 - p)

theorem target_probabilities :
  prob_both_hit = 0.36 ∧
  prob_one_hit = 0.48 ∧
  prob_at_least_one_hit = 0.84 := by
  sorry

end target_probabilities_l1572_157211


namespace round_to_nearest_whole_number_l1572_157239

theorem round_to_nearest_whole_number : 
  let x : ℝ := 6703.4999
  ‖x - 6703‖ < ‖x - 6704‖ :=
by sorry

end round_to_nearest_whole_number_l1572_157239


namespace water_distribution_l1572_157238

theorem water_distribution (total_water : ℕ) (eight_oz_glasses : ℕ) (four_oz_glasses : ℕ) 
  (h1 : total_water = 122)
  (h2 : eight_oz_glasses = 4)
  (h3 : four_oz_glasses = 15) : 
  (total_water - (8 * eight_oz_glasses + 4 * four_oz_glasses)) / 5 = 6 := by
sorry

end water_distribution_l1572_157238


namespace inequality_proof_l1572_157246

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  4 * (a^2 + b^2) > (a + b)^2 := by
  sorry

end inequality_proof_l1572_157246


namespace ellipse_m_range_l1572_157280

theorem ellipse_m_range (m : ℝ) : 
  (∃ (x y : ℝ), x^2/m + y^2/(2*m-1) = 1 ∧ 
   ∃ (a b : ℝ), a > b ∧ a^2 = m ∧ b^2 = 2*m-1) ↔ 
  (1/2 < m ∧ m < 1) :=
sorry

end ellipse_m_range_l1572_157280


namespace chord_bisected_by_M_l1572_157254

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/16 + y^2/4 = 1

-- Define the point M
def M : ℝ × ℝ := (2, 1)

-- Define a chord of the ellipse
def is_chord (A B : ℝ × ℝ) : Prop :=
  is_on_ellipse A.1 A.2 ∧ is_on_ellipse B.1 B.2

-- Define the midpoint of a chord
def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Define a line by its equation ax + by + c = 0
def line_equation (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y + c = 0

-- The main theorem
theorem chord_bisected_by_M :
  ∀ A B : ℝ × ℝ,
  is_chord A B →
  is_midpoint M A B →
  line_equation 1 2 (-4) A.1 A.2 ∧ line_equation 1 2 (-4) B.1 B.2 :=
sorry

end chord_bisected_by_M_l1572_157254


namespace train_length_l1572_157217

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmph : ℝ) (crossing_time : ℝ) : 
  speed_kmph = 72 → crossing_time = 7 → 
  (speed_kmph * 1000 / 3600) * crossing_time = 140 := by
  sorry

end train_length_l1572_157217


namespace april_roses_problem_l1572_157219

theorem april_roses_problem (rose_price : ℕ) (roses_left : ℕ) (earnings : ℕ) :
  rose_price = 7 →
  roses_left = 4 →
  earnings = 35 →
  ∃ (initial_roses : ℕ), initial_roses = (earnings / rose_price) + roses_left ∧ initial_roses = 9 :=
by sorry

end april_roses_problem_l1572_157219


namespace seashell_count_l1572_157250

/-- Given a collection of seashells with specific counts for different colors,
    calculate the number of shells that are not red, green, or blue. -/
theorem seashell_count (total : ℕ) (red green blue : ℕ) 
    (h_total : total = 501)
    (h_red : red = 123)
    (h_green : green = 97)
    (h_blue : blue = 89) :
    total - (red + green + blue) = 192 := by
  sorry

end seashell_count_l1572_157250


namespace linearly_dependent_implies_k_equals_six_l1572_157281

/-- Two vectors in ℝ² are linearly dependent if there exist non-zero scalars such that their linear combination is zero. -/
def linearlyDependent (v w : ℝ × ℝ) : Prop :=
  ∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ a • v + b • w = (0, 0)

/-- The theorem states that if the vectors (2, 3) and (4, k) are linearly dependent, then k must equal 6. -/
theorem linearly_dependent_implies_k_equals_six :
  linearlyDependent (2, 3) (4, k) → k = 6 := by
  sorry

end linearly_dependent_implies_k_equals_six_l1572_157281


namespace consecutive_integers_problem_l1572_157274

theorem consecutive_integers_problem (x y z : ℤ) : 
  (y = x - 1) → (z = y - 1) → (x > y) → (y > z) → 
  (2 * x + 3 * y + 3 * z = 5 * y + 11) → (z = 3) → 
  (2 * x = 10) := by
  sorry

end consecutive_integers_problem_l1572_157274


namespace range_of_a_l1572_157251

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |2*x - a|

/-- The theorem stating the range of a -/
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x ≥ (1/4)*a^2 + 1) → a ∈ Set.Icc (-2) 0 :=
by sorry

end range_of_a_l1572_157251


namespace rectangle_folding_cutting_perimeter_ratio_l1572_157293

theorem rectangle_folding_cutting_perimeter_ratio :
  let initial_length : ℚ := 6
  let initial_width : ℚ := 4
  let folded_length : ℚ := initial_length / 2
  let folded_width : ℚ := initial_width
  let cut_length : ℚ := folded_length
  let cut_width : ℚ := folded_width / 2
  let small_perimeter : ℚ := 2 * (cut_length + cut_width)
  let large_perimeter : ℚ := 2 * (folded_length + folded_width)
  small_perimeter / large_perimeter = 5 / 7 := by
  sorry

end rectangle_folding_cutting_perimeter_ratio_l1572_157293


namespace inscribed_square_existence_uniqueness_l1572_157296

/-- A sector in a plane --/
structure Sector where
  center : Point
  p : Point
  q : Point

/-- Angle of a sector --/
def Sector.angle (s : Sector) : ℝ := sorry

/-- A square in a plane --/
structure Square where
  a : Point
  b : Point
  c : Point
  d : Point

/-- Predicate to check if a square is inscribed in a sector according to the problem conditions --/
def isInscribed (sq : Square) (s : Sector) : Prop := sorry

/-- Theorem stating the existence and uniqueness of the inscribed square --/
theorem inscribed_square_existence_uniqueness (s : Sector) :
  (∃! sq : Square, isInscribed sq s) ↔ s.angle ≤ 180 := by sorry

end inscribed_square_existence_uniqueness_l1572_157296


namespace triangle_inequality_l1572_157206

theorem triangle_inequality (A B C m n l : ℝ) (h : A + B + C = π) : 
  (m^2 + Real.tan (A/2) * Real.tan (B/2))^(1/2) + 
  (n^2 + Real.tan (B/2) * Real.tan (C/2))^(1/2) + 
  (l^2 + Real.tan (C/2) * Real.tan (A/2))^(1/2) ≤ 
  (3 * (m^2 + n^2 + l^2 + 1))^(1/2) := by
sorry

end triangle_inequality_l1572_157206


namespace base_equation_solution_l1572_157223

/-- Converts a list of digits in base b to its decimal representation -/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldl (fun acc d => acc * b + d) 0

/-- Checks if a list of digits is valid in base b -/
def valid_digits (digits : List Nat) (b : Nat) : Prop :=
  digits.all (· < b)

theorem base_equation_solution :
  ∃! b : Nat, b > 1 ∧
    valid_digits [3, 4, 6, 4] b ∧
    valid_digits [4, 6, 2, 3] b ∧
    valid_digits [1, 0, 0, 0, 0] b ∧
    to_decimal [3, 4, 6, 4] b + to_decimal [4, 6, 2, 3] b = to_decimal [1, 0, 0, 0, 0] b :=
by
  sorry

end base_equation_solution_l1572_157223


namespace sqrt_expression_equals_seven_l1572_157210

theorem sqrt_expression_equals_seven :
  (Real.sqrt 3 - 2)^2 + Real.sqrt 12 + 6 * Real.sqrt (1/3) = 7 := by
  sorry

end sqrt_expression_equals_seven_l1572_157210


namespace james_shirts_count_l1572_157287

/-- The number of shirts James has -/
def num_shirts : ℕ := 10

/-- The number of pairs of pants James has -/
def num_pants : ℕ := 12

/-- The time it takes to fix a shirt (in hours) -/
def shirt_time : ℚ := 3/2

/-- The hourly rate charged by the tailor (in dollars) -/
def hourly_rate : ℕ := 30

/-- The total cost for fixing all shirts and pants (in dollars) -/
def total_cost : ℕ := 1530

theorem james_shirts_count :
  num_shirts = 10 ∧
  num_pants = 12 ∧
  shirt_time = 3/2 ∧
  hourly_rate = 30 ∧
  total_cost = 1530 →
  num_shirts * (shirt_time * hourly_rate) + num_pants * (2 * shirt_time * hourly_rate) = total_cost :=
by sorry

end james_shirts_count_l1572_157287


namespace proposition_analysis_l1572_157258

theorem proposition_analysis (a b : ℝ) : 
  (∃ a b, a * b > 0 ∧ (a ≤ 0 ∨ b ≤ 0)) ∧ 
  (∃ a b, (a ≤ 0 ∨ b ≤ 0) ∧ a * b > 0) ∧
  (∀ a b, a * b ≤ 0 → a ≤ 0 ∨ b ≤ 0) := by
  sorry

end proposition_analysis_l1572_157258


namespace arithmetic_sequence_ratio_l1572_157231

/-- Given an arithmetic sequence {a_n} with common difference d,
    S_n is the sum of the first n terms. -/
def arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) (S : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d ∧ S n = n * a 1 + n * (n - 1) * d / 2

theorem arithmetic_sequence_ratio
  (a : ℕ → ℚ) (d : ℚ) (S : ℕ → ℚ)
  (h : arithmetic_sequence a d S)
  (h_ratio : S 5 / S 3 = 3) :
  a 5 / a 3 = 17 / 9 := by
sorry

end arithmetic_sequence_ratio_l1572_157231


namespace expression_value_l1572_157203

theorem expression_value (a b : ℤ) (ha : a = 4) (hb : b = -3) : -a - b^3 + a*b = 11 := by
  sorry

end expression_value_l1572_157203


namespace real_part_of_z_l1572_157261

theorem real_part_of_z (z : ℂ) : z = (2 + I) / (1 + I)^2 → z.re = 1/2 := by
  sorry

end real_part_of_z_l1572_157261


namespace product_of_four_expressions_l1572_157278

theorem product_of_four_expressions (A B C D : ℝ) : 
  A = (Real.sqrt 2018 + Real.sqrt 2019 + 1) →
  B = (-Real.sqrt 2018 - Real.sqrt 2019 - 1) →
  C = (Real.sqrt 2018 - Real.sqrt 2019 + 1) →
  D = (Real.sqrt 2019 - Real.sqrt 2018 + 1) →
  A * B * C * D = 9 := by sorry

end product_of_four_expressions_l1572_157278


namespace sum_of_x_and_y_is_two_l1572_157201

theorem sum_of_x_and_y_is_two (x y : ℝ) 
  (eq1 : x^3 + y^3 = 98) 
  (eq2 : x^2*y + x*y^2 = -30) : 
  x + y = 2 := by
sorry

end sum_of_x_and_y_is_two_l1572_157201


namespace element_four_in_B_l1572_157244

def U : Set ℕ := {x | x ≤ 7}

theorem element_four_in_B (A B : Set ℕ) 
  (h1 : U = A ∪ B) 
  (h2 : A ∩ (Bᶜ) = {2, 3, 5, 7}) : 
  4 ∈ B := by
  sorry

end element_four_in_B_l1572_157244


namespace ethanol_percentage_fuel_B_l1572_157256

-- Define the constants
def tank_capacity : ℝ := 212
def fuel_A_ethanol_percentage : ℝ := 0.12
def fuel_A_volume : ℝ := 98
def total_ethanol : ℝ := 30

-- Define the theorem
theorem ethanol_percentage_fuel_B :
  let ethanol_A := fuel_A_ethanol_percentage * fuel_A_volume
  let ethanol_B := total_ethanol - ethanol_A
  let fuel_B_volume := tank_capacity - fuel_A_volume
  (ethanol_B / fuel_B_volume) * 100 = 16 := by
  sorry

end ethanol_percentage_fuel_B_l1572_157256


namespace amount_with_r_l1572_157289

/-- Given three people sharing a total amount of money, where one person has
    two-thirds of what the other two have combined, this theorem proves
    the amount held by that person. -/
theorem amount_with_r (total : ℝ) (amount_r : ℝ) : 
  total = 7000 →
  amount_r = (2/3) * (total - amount_r) →
  amount_r = 2800 := by
sorry


end amount_with_r_l1572_157289


namespace expected_different_faces_correct_l1572_157284

/-- A fair six-sided die is rolled six times. -/
def num_rolls : ℕ := 6

/-- The number of faces on the die. -/
def num_faces : ℕ := 6

/-- The expected number of different faces that will appear when rolling a fair six-sided die six times. -/
def expected_different_faces : ℚ :=
  (num_faces ^ num_rolls - (num_faces - 1) ^ num_rolls) / (num_faces ^ (num_rolls - 1))

/-- Theorem stating that the expected number of different faces is correct. -/
theorem expected_different_faces_correct :
  expected_different_faces = (6^6 - 5^6) / 6^5 := by
  sorry


end expected_different_faces_correct_l1572_157284


namespace area_ratio_small_large_triangles_l1572_157294

/-- The ratio of areas between four small equilateral triangles and one large equilateral triangle -/
theorem area_ratio_small_large_triangles : 
  let small_side : ℝ := 10
  let small_perimeter : ℝ := 3 * small_side
  let total_perimeter : ℝ := 4 * small_perimeter
  let large_side : ℝ := total_perimeter / 3
  let small_area : ℝ := (Real.sqrt 3 / 4) * small_side ^ 2
  let large_area : ℝ := (Real.sqrt 3 / 4) * large_side ^ 2
  (4 * small_area) / large_area = 1 / 4 := by
  sorry

end area_ratio_small_large_triangles_l1572_157294


namespace laylas_track_distance_l1572_157268

/-- The distance Layla rode around the running track, given her total mileage and the distance to the high school. -/
theorem laylas_track_distance (total_mileage : ℝ) (distance_to_school : ℝ) 
  (h1 : total_mileage = 10)
  (h2 : distance_to_school = 3) :
  total_mileage - 2 * distance_to_school = 4 :=
by sorry

end laylas_track_distance_l1572_157268


namespace imaginary_part_of_complex_number_l1572_157236

theorem imaginary_part_of_complex_number :
  let z : ℂ := 1 - 2 * Complex.I
  Complex.im z = -2 := by sorry

end imaginary_part_of_complex_number_l1572_157236


namespace tan_family_total_cost_l1572_157220

/-- Represents the composition of the group visiting the amusement park -/
structure GroupComposition where
  children : Nat
  adults : Nat
  seniors : Nat

/-- Represents the discount rules for the amusement park -/
structure DiscountRules where
  seniorDiscount : Rat
  childDiscount : Rat
  groupDiscountThreshold : Nat
  groupDiscount : Rat

/-- Calculates the total cost for a group visiting the amusement park -/
def calculateTotalCost (composition : GroupComposition) (rules : DiscountRules) (adultPrice : Rat) : Rat :=
  sorry

/-- Theorem stating that the total cost for the Tan family's tickets is $45 -/
theorem tan_family_total_cost :
  let composition : GroupComposition := { children := 2, adults := 2, seniors := 2 }
  let rules : DiscountRules := { seniorDiscount := 3/10, childDiscount := 1/5, groupDiscountThreshold := 5, groupDiscount := 1/10 }
  let adultPrice : Rat := 10
  calculateTotalCost composition rules adultPrice = 45 := by
  sorry

end tan_family_total_cost_l1572_157220


namespace bela_winning_strategy_l1572_157282

/-- The game state, representing the current player and the list of chosen numbers -/
inductive GameState
  | bela (choices : List ℝ)
  | jenn (choices : List ℝ)

/-- The game rules -/
def GameRules (n : ℕ) : GameState → Prop :=
  λ state =>
    n > 10 ∧
    match state with
    | GameState.bela choices => choices.all (λ x => 0 ≤ x ∧ x ≤ n)
    | GameState.jenn choices => choices.all (λ x => 0 ≤ x ∧ x ≤ n)

/-- A valid move in the game -/
def ValidMove (n : ℕ) (state : GameState) (move : ℝ) : Prop :=
  0 ≤ move ∧ move ≤ n ∧
  match state with
  | GameState.bela choices => choices.all (λ x => |x - move| > 2)
  | GameState.jenn choices => choices.all (λ x => |x - move| > 2)

/-- Bela has a winning strategy -/
theorem bela_winning_strategy (n : ℕ) :
  n > 10 →
  ∃ (strategy : GameState → ℝ),
    ∀ (state : GameState),
      GameRules n state →
      (∃ (move : ℝ), ValidMove n state move) →
      ValidMove n state (strategy state) :=
sorry

end bela_winning_strategy_l1572_157282


namespace sector_area_l1572_157271

theorem sector_area (perimeter : ℝ) (central_angle : ℝ) (h1 : perimeter = 16) (h2 : central_angle = 2) :
  let radius := perimeter / (2 + central_angle)
  (1 / 2) * radius^2 * central_angle = 16 := by sorry

end sector_area_l1572_157271


namespace not_necessarily_p_or_q_l1572_157232

theorem not_necessarily_p_or_q (p q : Prop) 
  (h1 : ¬p) 
  (h2 : ¬(p ∧ q)) : 
  ¬∀ (p q : Prop), (¬p ∧ ¬(p ∧ q)) → (p ∨ q) :=
by
  sorry

end not_necessarily_p_or_q_l1572_157232


namespace triangle_circle_areas_l1572_157215

theorem triangle_circle_areas (r s t : ℝ) : 
  r + s = 6 →
  r + t = 8 →
  s + t = 10 →
  r > 0 →
  s > 0 →
  t > 0 →
  π * r^2 + π * s^2 + π * t^2 = 36 * π :=
by sorry

end triangle_circle_areas_l1572_157215


namespace banana_distribution_l1572_157243

theorem banana_distribution (total_bananas : Nat) (num_groups : Nat) (bananas_per_group : Nat) :
  total_bananas = 407 →
  num_groups = 11 →
  bananas_per_group = total_bananas / num_groups →
  bananas_per_group = 37 := by
  sorry

end banana_distribution_l1572_157243


namespace problem_proof_l1572_157237

theorem problem_proof : (-2)^0 - 3 * Real.tan (30 * π / 180) - |Real.sqrt 3 - 2| = -1 := by
  sorry

end problem_proof_l1572_157237


namespace problem_solution_l1572_157218

theorem problem_solution (x y : ℚ) (hx : x = 3/4) (hy : y = 4/3) :
  (1/3 * x^7 * y^6) * 4 = 1 := by sorry

end problem_solution_l1572_157218


namespace trigonometric_identities_l1572_157272

theorem trigonometric_identities (θ : Real) 
  (h1 : π/2 < θ ∧ θ < π) -- θ is in the second quadrant
  (h2 : Real.tan (2*θ) = -2*Real.sqrt 2) : -- tan 2θ = -2√2
  (Real.tan θ = -Real.sqrt 2 / 2) ∧ 
  ((2 * (Real.cos (θ/2))^2 - Real.sin θ - Real.tan (5*π/4)) / 
   (Real.sqrt 2 * Real.sin (θ + π/4)) = 4 + 2*Real.sqrt 2) := by
  sorry

end trigonometric_identities_l1572_157272


namespace x_range_for_equation_l1572_157205

theorem x_range_for_equation (x y : ℝ) (h : x / y = x - y) : x ≥ 4 ∨ x ≤ 0 := by
  sorry

end x_range_for_equation_l1572_157205


namespace cube_dimension_reduction_l1572_157252

theorem cube_dimension_reduction (initial_face_area : ℝ) (reduction : ℝ) : 
  initial_face_area = 36 ∧ reduction = 1 → 
  (3 : ℝ) * (Real.sqrt initial_face_area - reduction) = 15 := by
  sorry

end cube_dimension_reduction_l1572_157252


namespace parabola_points_l1572_157241

theorem parabola_points : 
  {p : ℝ × ℝ | p.2 = p.1^2 - 1 ∧ p.2 = 3} = {(-2, 3), (2, 3)} := by
  sorry

end parabola_points_l1572_157241


namespace circle_radius_condition_l1572_157240

theorem circle_radius_condition (x y c : ℝ) : 
  (∀ x y, x^2 + 8*x + y^2 + 2*y + c = 0 → (x + 4)^2 + (y + 1)^2 = 25) → c = -8 :=
by sorry

end circle_radius_condition_l1572_157240


namespace exist_x_y_different_squares_no_x_y_different_squares_in_range_l1572_157295

-- Define the property for two numbers to be different perfect squares
def areDifferentPerfectSquares (a b : ℕ) : Prop :=
  ∃ m n : ℕ, m ≠ n ∧ a = m^2 ∧ b = n^2

-- Theorem 1: Existence of x and y satisfying the condition
theorem exist_x_y_different_squares :
  ∃ x y : ℕ, areDifferentPerfectSquares (x*y + x) (x*y + y) :=
sorry

-- Theorem 2: Non-existence of x and y between 988 and 1991 satisfying the condition
theorem no_x_y_different_squares_in_range :
  ¬∃ x y : ℕ, 988 ≤ x ∧ x ≤ 1991 ∧ 988 ≤ y ∧ y ≤ 1991 ∧
    areDifferentPerfectSquares (x*y + x) (x*y + y) :=
sorry

end exist_x_y_different_squares_no_x_y_different_squares_in_range_l1572_157295


namespace complex_sum_reciprocal_squared_l1572_157265

def complex_number (a b : ℝ) : ℂ := a + b * Complex.I

theorem complex_sum_reciprocal_squared (x : ℂ) :
  x + (1 / x) = 5 → x^2 + (1 / x)^2 = (7 : ℝ) / 2 := by
  sorry

end complex_sum_reciprocal_squared_l1572_157265


namespace inverse_function_point_and_sum_l1572_157213

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem inverse_function_point_and_sum :
  (f 2 = 9) →  -- This captures the condition that (2,3) is on y = f(x)/3
  (∃ (x : ℝ), f x = 9 ∧ f⁻¹ 9 = 2) ∧  -- This states that (9, 2/3) is on y = f^(-1)(x)/3
  (9 + 2/3 = 29/3) :=  -- This is the sum of coordinates
by sorry

end inverse_function_point_and_sum_l1572_157213


namespace perpendicular_lines_a_value_l1572_157286

theorem perpendicular_lines_a_value (a : ℝ) : 
  (∃ (x y : ℝ), ax - y + 2*a = 0 ∧ (2*a - 1)*x + a*y + a = 0) →
  (a*(2*a - 1) + (-1)*a = 0) →
  (a = 0 ∨ a = 1) := by
sorry

end perpendicular_lines_a_value_l1572_157286


namespace average_L_value_l1572_157276

/-- Represents a coin configuration with H and T sides -/
def Configuration (n : ℕ) := Fin n → Bool

/-- The number of operations before stopping for a given configuration -/
def L (n : ℕ) (c : Configuration n) : ℕ :=
  sorry  -- Definition of L would go here

/-- The average value of L(C) over all 2^n possible initial configurations -/
def averageLValue (n : ℕ) : ℚ :=
  sorry  -- Definition of average L value would go here

/-- Theorem stating that the average L value is n(n+1)/4 -/
theorem average_L_value (n : ℕ) : 
  averageLValue n = ↑n * (↑n + 1) / 4 :=
sorry

end average_L_value_l1572_157276


namespace cos_two_pi_thirds_minus_alpha_l1572_157225

theorem cos_two_pi_thirds_minus_alpha (α : ℝ) (h : Real.sin (α - π/6) = 3/5) :
  Real.cos (2*π/3 - α) = 3/5 := by sorry

end cos_two_pi_thirds_minus_alpha_l1572_157225


namespace egg_difference_is_thirteen_l1572_157266

/-- Represents the egg problem with given conditions --/
structure EggProblem where
  total_dozens : Nat
  trays : Nat
  dropped_trays : Nat
  first_tray_broken : Nat
  first_tray_cracked : Nat
  first_tray_slightly_cracked : Nat
  second_tray_shattered : Nat
  second_tray_cracked : Nat
  second_tray_slightly_cracked : Nat

/-- Calculates the difference between perfect eggs in undropped trays and cracked eggs in dropped trays --/
def egg_difference (p : EggProblem) : Nat :=
  let total_eggs := p.total_dozens * 12
  let eggs_per_tray := total_eggs / p.trays
  let undropped_trays := p.trays - p.dropped_trays
  let perfect_eggs := undropped_trays * eggs_per_tray
  let cracked_eggs := p.first_tray_cracked + p.second_tray_cracked
  perfect_eggs - cracked_eggs

/-- Theorem stating the difference is 13 for the given problem conditions --/
theorem egg_difference_is_thirteen : egg_difference {
  total_dozens := 4
  trays := 4
  dropped_trays := 2
  first_tray_broken := 3
  first_tray_cracked := 5
  first_tray_slightly_cracked := 2
  second_tray_shattered := 4
  second_tray_cracked := 6
  second_tray_slightly_cracked := 1
} = 13 := by
  sorry

end egg_difference_is_thirteen_l1572_157266


namespace tunnel_regression_theorem_prove_tunnel_regression_l1572_157269

/-- Statistical data for tunnel sinking analysis -/
structure TunnelData where
  sum_tz : Real  -- ∑(t_i - t̄)(z_i - z̄)
  sum_z2 : Real  -- ∑(z_i - z̄)^2
  mean_z : Real  -- z̄
  sum_tu : Real  -- ∑(t_i - t̄)(u_i - ū)
  sum_u2 : Real  -- ∑(u_i - ū)^2

/-- Parameters for the regression equation z = ke^(bt) -/
structure RegressionParams where
  k : Real
  b : Real

/-- Theorem stating the correctness of the regression equation and adjustment day -/
theorem tunnel_regression_theorem (data : TunnelData) 
  (params : RegressionParams) (adjust_day : Nat) : Prop :=
  data.sum_tz = 22.3 ∧
  data.sum_z2 = 27.5 ∧
  data.mean_z = 1.2 ∧
  data.sum_tu = 25.2 ∧
  data.sum_u2 = 30 ∧
  params.b = 0.9 ∧
  params.k = Real.exp (-4.8) ∧
  adjust_day = 9 ∧
  (∀ t : Real, 
    Real.exp (params.b * t - 4.8) = params.k * Real.exp (params.b * t)) ∧
  (∀ n : Real, 
    0.9 * Real.exp (0.9 * n - 4.8) > 27 → n > 9.1)

/-- Proof of the tunnel regression theorem -/
theorem prove_tunnel_regression : 
  ∃ (data : TunnelData) (params : RegressionParams) (adjust_day : Nat),
    tunnel_regression_theorem data params adjust_day :=
sorry

end tunnel_regression_theorem_prove_tunnel_regression_l1572_157269


namespace discount_percentage_calculation_l1572_157221

/-- Calculates the discount percentage given item costs and final amount spent -/
theorem discount_percentage_calculation 
  (hand_mitts_cost apron_cost utensils_cost final_amount : ℚ)
  (nieces : ℕ)
  (h1 : hand_mitts_cost = 14)
  (h2 : apron_cost = 16)
  (h3 : utensils_cost = 10)
  (h4 : nieces = 3)
  (h5 : final_amount = 135) :
  let knife_cost := 2 * utensils_cost
  let single_set_cost := hand_mitts_cost + apron_cost + utensils_cost + knife_cost
  let total_cost := nieces * single_set_cost
  let discount_amount := total_cost - final_amount
  let discount_percentage := (discount_amount / total_cost) * 100
  discount_percentage = 25 := by
sorry


end discount_percentage_calculation_l1572_157221


namespace vampire_blood_consumption_l1572_157291

/-- The amount of blood a vampire needs per week in gallons -/
def blood_needed_per_week : ℚ := 7

/-- The number of people the vampire sucks blood from each day -/
def people_per_day : ℕ := 4

/-- The number of pints in a gallon -/
def pints_per_gallon : ℕ := 8

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- Theorem: Given a vampire who needs 7 gallons of blood per week and sucks blood from 4 people each day, 
    the amount of blood sucked per person is 2 pints. -/
theorem vampire_blood_consumption :
  (blood_needed_per_week * pints_per_gallon) / (people_per_day * days_per_week) = 2 := by
  sorry

end vampire_blood_consumption_l1572_157291


namespace right_angle_implies_acute_fraction_inequality_l1572_157283

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define angle measures
def angle_measure (T : Triangle) (vertex : ℕ) : ℝ := sorry

-- Statement 1
theorem right_angle_implies_acute (T : Triangle) :
  angle_measure T 3 = π / 2 → angle_measure T 2 < π / 2 := by sorry

-- Statement 2
theorem fraction_inequality (a b m : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (hm : m > 0) :
  (b + m) / (a + m) > b / a := by sorry

end right_angle_implies_acute_fraction_inequality_l1572_157283


namespace parabola_directrix_l1572_157257

/-- The directrix of the parabola y = x^2 -/
theorem parabola_directrix : ∃ (k : ℝ), ∀ (x y : ℝ),
  y = x^2 → (4 * y + 1 = 0 ↔ y = k) := by sorry

end parabola_directrix_l1572_157257


namespace cindy_same_color_probability_l1572_157242

def total_marbles : ℕ := 8
def red_marbles : ℕ := 3
def blue_marbles : ℕ := 3
def green_marbles : ℕ := 1
def yellow_marbles : ℕ := 1

def alice_draw : ℕ := 3
def bob_draw : ℕ := 2
def cindy_draw : ℕ := 2

def probability_cindy_same_color : ℚ := 1 / 35

theorem cindy_same_color_probability :
  probability_cindy_same_color = 1 / 35 :=
by sorry

end cindy_same_color_probability_l1572_157242


namespace dvd_rental_cost_l1572_157270

def total_cost : ℝ := 4.80
def num_dvds : ℕ := 4

theorem dvd_rental_cost : total_cost / num_dvds = 1.20 := by
  sorry

end dvd_rental_cost_l1572_157270


namespace line_intercepts_sum_zero_l1572_157229

/-- Given a line l with equation 2x + (k - 3)y - 2k + 6 = 0, where k ≠ 3,
    if the sum of its x-intercept and y-intercept is 0, then k = 1. -/
theorem line_intercepts_sum_zero (k : ℝ) (h : k ≠ 3) :
  let l := {(x, y) : ℝ × ℝ | 2 * x + (k - 3) * y - 2 * k + 6 = 0}
  let x_intercept := (k - 3 : ℝ)
  let y_intercept := (2 : ℝ)
  x_intercept + y_intercept = 0 → k = 1 := by
  sorry

end line_intercepts_sum_zero_l1572_157229
