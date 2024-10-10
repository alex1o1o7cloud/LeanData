import Mathlib

namespace abs_sum_equals_sum_abs_necessary_not_sufficient_l248_24871

theorem abs_sum_equals_sum_abs_necessary_not_sufficient (a b : ℝ) :
  (∀ a b : ℝ, a * b > 0 → |a + b| = |a| + |b|) ∧
  (∃ a b : ℝ, |a + b| = |a| + |b| ∧ a * b ≤ 0) :=
by sorry

end abs_sum_equals_sum_abs_necessary_not_sufficient_l248_24871


namespace triangle_theorem_l248_24859

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)  -- Angles
  (a b c : Real)  -- Sides opposite to angles A, B, C respectively
  (S : Real)      -- Area

-- State the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : t.b + t.c = 2 * t.a * Real.cos t.B)  -- Given condition
  (h2 : t.S = t.a^2 / 4)                     -- Given area condition
  : t.A = 2 * t.B ∧ (t.A = Real.pi / 2 ∨ t.A = Real.pi / 4) :=
by sorry

end triangle_theorem_l248_24859


namespace unattainable_y_value_l248_24804

theorem unattainable_y_value (x : ℝ) (hx : x ≠ -4/3) :
  ¬∃y : ℝ, y = (2 - x) / (3 * x + 4) ↔ y = -1/3 :=
by sorry

end unattainable_y_value_l248_24804


namespace essay_writing_speed_l248_24803

/-- Represents the essay writing scenario -/
structure EssayWriting where
  total_words : ℕ
  initial_speed : ℕ
  initial_hours : ℕ
  total_hours : ℕ

/-- Calculates the words written per hour after the initial period -/
def words_per_hour_after (e : EssayWriting) : ℕ :=
  (e.total_words - e.initial_speed * e.initial_hours) / (e.total_hours - e.initial_hours)

/-- Theorem stating that under the given conditions, the writing speed after
    the first two hours is 200 words per hour -/
theorem essay_writing_speed (e : EssayWriting) 
    (h1 : e.total_words = 1200)
    (h2 : e.initial_speed = 400)
    (h3 : e.initial_hours = 2)
    (h4 : e.total_hours = 4) : 
  words_per_hour_after e = 200 := by
  sorry

#eval words_per_hour_after { total_words := 1200, initial_speed := 400, initial_hours := 2, total_hours := 4 }

end essay_writing_speed_l248_24803


namespace gathering_handshakes_l248_24875

def num_dwarves : ℕ := 25
def num_elves : ℕ := 18

def handshakes_among_dwarves (n : ℕ) : ℕ := n * (n - 1) / 2
def handshakes_between_dwarves_and_elves (d e : ℕ) : ℕ := d * e

def total_handshakes (d e : ℕ) : ℕ :=
  handshakes_among_dwarves d + handshakes_between_dwarves_and_elves d e

theorem gathering_handshakes :
  total_handshakes num_dwarves num_elves = 750 := by
  sorry

end gathering_handshakes_l248_24875


namespace lcm_36_132_l248_24887

theorem lcm_36_132 : Nat.lcm 36 132 = 396 := by
  sorry

end lcm_36_132_l248_24887


namespace min_value_x_plus_4y_min_value_is_2_plus_sqrt2_min_value_achieved_l248_24823

theorem min_value_x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 1/(2*y) = 2) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 1/a + 1/(2*b) = 2 → x + 4*y ≤ a + 4*b :=
by sorry

theorem min_value_is_2_plus_sqrt2 (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 1/(2*y) = 2) :
  x + 4*y ≥ 2 + Real.sqrt 2 :=
by sorry

theorem min_value_achieved (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 1/(2*y) = 2) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 1/a + 1/(2*b) = 2 ∧ a + 4*b = 2 + Real.sqrt 2 :=
by sorry

end min_value_x_plus_4y_min_value_is_2_plus_sqrt2_min_value_achieved_l248_24823


namespace sum_of_A_and_C_is_seven_l248_24876

theorem sum_of_A_and_C_is_seven (A B C D : ℕ) : 
  A ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
  B ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
  C ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
  D ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  (A : ℚ) / B + (C : ℚ) / D = 3 →
  A + C = 7 := by
sorry

end sum_of_A_and_C_is_seven_l248_24876


namespace carmen_daniel_difference_l248_24872

/-- Calculates the difference in miles biked between two cyclists after a given time -/
def miles_difference (carmen_rate daniel_rate time : ℝ) : ℝ :=
  carmen_rate * time - daniel_rate * time

theorem carmen_daniel_difference :
  miles_difference 15 10 3 = 15 := by sorry

end carmen_daniel_difference_l248_24872


namespace intersection_M_N_l248_24889

def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {x | x^2 - 3*x ≤ 0}

theorem intersection_M_N : N ∩ M = {x | 2 < x ∧ x ≤ 3} := by sorry

end intersection_M_N_l248_24889


namespace ellipse_equation_l248_24879

/-- An ellipse with foci and points satisfying certain conditions -/
structure Ellipse where
  a : ℝ
  b : ℝ
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  h₁ : a > b
  h₂ : b > 0
  h₃ : A.1^2 / a^2 + A.2^2 / b^2 = 1  -- A is on the ellipse
  h₄ : B.1^2 / a^2 + B.2^2 / b^2 = 1  -- B is on the ellipse
  h₅ : (A.1 - B.1) * (F₁.1 - F₂.1) + (A.2 - B.2) * (F₁.2 - F₂.2) = 0  -- AB ⟂ F₁F₂
  h₆ : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 16  -- |AB| = 4
  h₇ : (F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2 = 12  -- |F₁F₂| = 2√3

/-- The equation of the ellipse is x²/9 + y²/6 = 1 -/
theorem ellipse_equation (e : Ellipse) : e.a^2 = 9 ∧ e.b^2 = 6 := by
  sorry

end ellipse_equation_l248_24879


namespace movie_ticket_price_l248_24849

/-- The price of a 3D movie ticket --/
def price_3d : ℕ := sorry

/-- The price of a matinee ticket --/
def price_matinee : ℕ := 5

/-- The price of an evening ticket --/
def price_evening : ℕ := 12

/-- The number of matinee tickets sold --/
def num_matinee : ℕ := 200

/-- The number of evening tickets sold --/
def num_evening : ℕ := 300

/-- The number of 3D tickets sold --/
def num_3d : ℕ := 100

/-- The total revenue from all ticket sales --/
def total_revenue : ℕ := 6600

theorem movie_ticket_price :
  price_3d = 20 ∧
  price_matinee * num_matinee +
  price_evening * num_evening +
  price_3d * num_3d = total_revenue :=
by sorry

end movie_ticket_price_l248_24849


namespace max_value_expression_l248_24891

theorem max_value_expression (a b c d : ℝ) 
  (ha : a ∈ Set.Icc (-9.5) 9.5)
  (hb : b ∈ Set.Icc (-9.5) 9.5)
  (hc : c ∈ Set.Icc (-9.5) 9.5)
  (hd : d ∈ Set.Icc (-9.5) 9.5) :
  (a + 2*b + c + 2*d - a*b - b*c - c*d - d*a) ≤ 380 :=
by sorry

end max_value_expression_l248_24891


namespace relationship_abc_l248_24874

theorem relationship_abc : ∀ (a b c : ℝ), 
  a = 1/3 → b = Real.sin (1/3) → c = 1/Real.pi → a > b ∧ b > c := by
  sorry

end relationship_abc_l248_24874


namespace function_satisfies_equation_l248_24856

theorem function_satisfies_equation (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ (x - 3) / (x^2 - x + 4)
  2 * f (1 - x) + 1 = x * f x := by
  sorry

end function_satisfies_equation_l248_24856


namespace greatest_multiple_of_12_with_unique_digits_M_mod_1000_l248_24841

/-- A function that checks if a natural number has all unique digits -/
def has_unique_digits (n : ℕ) : Prop := sorry

/-- The greatest integer multiple of 12 with all unique digits -/
def M : ℕ := sorry

theorem greatest_multiple_of_12_with_unique_digits : 
  M % 12 = 0 ∧ 
  has_unique_digits M ∧ 
  ∀ k, k % 12 = 0 → has_unique_digits k → k ≤ M :=
sorry

theorem M_mod_1000 : M % 1000 = 320 := sorry

end greatest_multiple_of_12_with_unique_digits_M_mod_1000_l248_24841


namespace fq_length_l248_24818

-- Define the triangle DEF
structure RightTriangle where
  DE : ℝ
  DF : ℝ
  rightAngleAtE : True

-- Define the circle
structure TangentCircle where
  centerOnDE : True
  tangentToDF : True
  tangentToEF : True

-- Define the theorem
theorem fq_length
  (triangle : RightTriangle)
  (circle : TangentCircle)
  (h1 : triangle.DF = Real.sqrt 85)
  (h2 : triangle.DE = 7)
  : ∃ Q : ℝ × ℝ, ∃ F : ℝ × ℝ, ‖F - Q‖ = 6 :=
by
  sorry

end fq_length_l248_24818


namespace tom_batteries_in_toys_l248_24816

/-- The number of batteries Tom used in his toys -/
def batteries_in_toys (total batteries_in_flashlights batteries_in_controllers : ℕ) : ℕ :=
  total - (batteries_in_flashlights + batteries_in_controllers)

/-- Theorem stating that Tom used 15 batteries in his toys -/
theorem tom_batteries_in_toys :
  batteries_in_toys 19 2 2 = 15 := by
  sorry

end tom_batteries_in_toys_l248_24816


namespace quadratic_distinct_roots_l248_24843

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) :=
by sorry

end quadratic_distinct_roots_l248_24843


namespace triangle_problem_l248_24830

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove the values of b and cos(2B - π/3) under specific conditions. -/
theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  b * Real.sin A = 3 * c * Real.sin B →
  a = 3 →
  Real.cos B = 2/3 →
  b = Real.sqrt 6 ∧ Real.cos (2*B - π/3) = (4 * Real.sqrt 15 - 1) / 18 := by
  sorry

end triangle_problem_l248_24830


namespace arithmetic_calculation_l248_24895

theorem arithmetic_calculation : 4 * 6 * 8 - 24 / 3 = 184 := by
  sorry

end arithmetic_calculation_l248_24895


namespace cloth_sale_problem_l248_24893

/-- Given a shopkeeper's cloth sale scenario, calculate the number of metres sold. -/
theorem cloth_sale_problem (total_selling_price : ℕ) (loss_per_metre : ℕ) (cost_price_per_metre : ℕ) : 
  total_selling_price = 9000 →
  loss_per_metre = 6 →
  cost_price_per_metre = 36 →
  (total_selling_price / (cost_price_per_metre - loss_per_metre) : ℕ) = 300 := by
  sorry

end cloth_sale_problem_l248_24893


namespace greatest_missable_problems_l248_24848

theorem greatest_missable_problems (total_problems : ℕ) (passing_percentage : ℚ) 
  (h1 : total_problems = 50)
  (h2 : passing_percentage = 85 / 100) :
  ∃ (max_missable : ℕ), 
    max_missable = 7 ∧ 
    (total_problems - max_missable : ℚ) / total_problems ≥ passing_percentage ∧
    ∀ (n : ℕ), n > max_missable → (total_problems - n : ℚ) / total_problems < passing_percentage :=
by sorry

end greatest_missable_problems_l248_24848


namespace root_sequence_difference_l248_24847

theorem root_sequence_difference (m n : ℝ) : 
  (∃ a b c d : ℝ, 
    (a = 1) ∧
    (a * d = b * c) ∧
    ({a, b, c, d} = {x : ℝ | (x^2 - m*x + 27 = 0) ∨ (x^2 - n*x + 27 = 0)}) ∧
    (∃ q : ℝ, b = a*q ∧ c = b*q ∧ d = c*q)) →
  |m - n| = 16 :=
by sorry

end root_sequence_difference_l248_24847


namespace contrapositive_equivalence_l248_24883

theorem contrapositive_equivalence (x : ℝ) :
  (¬ (-2 < x ∧ x < 2) → ¬ (x^2 < 4)) ↔ ((x ≤ -2 ∨ x ≥ 2) → x^2 ≥ 4) :=
by sorry

end contrapositive_equivalence_l248_24883


namespace correct_distribution_l248_24839

/-- Represents the distribution of sampled students across three camps -/
structure CampDistribution where
  camp1 : Nat
  camp2 : Nat
  camp3 : Nat

/-- Parameters for the systematic sampling -/
structure SamplingParams where
  totalStudents : Nat
  sampleSize : Nat
  startNumber : Nat

/-- Function to perform systematic sampling and calculate camp distribution -/
def systematicSampling (params : SamplingParams) : CampDistribution :=
  sorry

/-- Theorem stating the correct distribution for the given problem -/
theorem correct_distribution :
  let params : SamplingParams := {
    totalStudents := 300,
    sampleSize := 20,
    startNumber := 3
  }
  let result : CampDistribution := systematicSampling params
  result.camp1 = 14 ∧ result.camp2 = 3 ∧ result.camp3 = 3 :=
sorry

end correct_distribution_l248_24839


namespace inverse_function_of_log_l248_24821

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem inverse_function_of_log (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : f a (2 : ℝ) = -1) :
  ∀ x, f⁻¹ a x = (1/2 : ℝ) ^ x :=
by sorry

end inverse_function_of_log_l248_24821


namespace train_speed_l248_24811

/-- Proves that a train with given length and time to cross a pole has a specific speed -/
theorem train_speed (train_length : Real) (crossing_time : Real) (speed : Real) : 
  train_length = 200 → 
  crossing_time = 12 → 
  speed = (train_length / 1000) / (crossing_time / 3600) → 
  speed = 60 := by
  sorry

#check train_speed

end train_speed_l248_24811


namespace cos_210_degrees_l248_24835

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_degrees_l248_24835


namespace small_pizza_has_eight_slices_l248_24881

/-- The number of slices in a small pizza -/
def small_pizza_slices : ℕ := sorry

/-- The number of people -/
def num_people : ℕ := 3

/-- The number of slices each person can eat -/
def slices_per_person : ℕ := 12

/-- The number of slices in a large pizza -/
def large_pizza_slices : ℕ := 14

/-- The number of small pizzas ordered -/
def num_small_pizzas : ℕ := 1

/-- The number of large pizzas ordered -/
def num_large_pizzas : ℕ := 2

theorem small_pizza_has_eight_slices :
  small_pizza_slices = 8 ∧
  num_people * slices_per_person ≤ 
    num_small_pizzas * small_pizza_slices + num_large_pizzas * large_pizza_slices :=
by sorry

end small_pizza_has_eight_slices_l248_24881


namespace committee_arrangement_count_l248_24815

def committee_size : ℕ := 10
def num_men : ℕ := 3
def num_women : ℕ := 7

theorem committee_arrangement_count :
  (committee_size.choose num_men) = 120 := by
  sorry

end committee_arrangement_count_l248_24815


namespace no_double_composition_inverse_l248_24800

-- Define the quadratic function g
def g (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem no_double_composition_inverse
  (a b c : ℝ)
  (h1 : ∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
                         g a b c (g a b c x₁) = x₁ ∧
                         g a b c (g a b c x₂) = x₂ ∧
                         g a b c (g a b c x₃) = x₃ ∧
                         g a b c (g a b c x₄) = x₄) :
  ¬∃ f : ℝ → ℝ, ∀ x : ℝ, f (f x) = g a b c x :=
by sorry

end no_double_composition_inverse_l248_24800


namespace garden_vegetable_ratio_l248_24836

theorem garden_vegetable_ratio :
  let potatoes : ℕ := 237
  let cucumbers : ℕ := potatoes - 60
  let total_vegetables : ℕ := 768
  let peppers : ℕ := total_vegetables - potatoes - cucumbers
  peppers = 2 * cucumbers :=
by sorry

end garden_vegetable_ratio_l248_24836


namespace basketball_shots_l248_24855

theorem basketball_shots (t h f : ℕ) : 
  (2 * t = 3 * h) →  -- Two-point shots scored double the points of three-point shots
  (f = h - 4) →      -- Number of free throws is four fewer than three-point shots
  (t + h + f = 40) → -- Total shots is 40
  (2 * t + 3 * h + f = 76) → -- Total score is 76
  h = 8 := by sorry

end basketball_shots_l248_24855


namespace cabbage_count_this_year_l248_24899

/-- Represents the number of cabbages in a square garden --/
def CabbageCount (side : ℕ) : ℕ := side * side

/-- Theorem stating the number of cabbages this year given the conditions --/
theorem cabbage_count_this_year :
  ∀ (last_year_side : ℕ),
  (CabbageCount (last_year_side + 1) - CabbageCount last_year_side = 197) →
  (CabbageCount (last_year_side + 1) = 9801) :=
by
  sorry

end cabbage_count_this_year_l248_24899


namespace sand_received_by_city_c_l248_24882

/-- The amount of sand received by City C given the total sand and amounts received by other cities -/
theorem sand_received_by_city_c 
  (total : ℝ) 
  (city_a : ℝ) 
  (city_b : ℝ) 
  (city_d : ℝ) 
  (h_total : total = 95) 
  (h_city_a : city_a = 16.5) 
  (h_city_b : city_b = 26) 
  (h_city_d : city_d = 28) : 
  total - (city_a + city_b + city_d) = 24.5 := by
sorry

end sand_received_by_city_c_l248_24882


namespace playlist_song_length_l248_24832

theorem playlist_song_length 
  (n_unknown : ℕ) 
  (n_known : ℕ) 
  (known_length : ℕ) 
  (total_duration : ℕ) : 
  n_unknown = 10 → 
  n_known = 15 → 
  known_length = 2 → 
  total_duration = 60 → 
  ∃ (unknown_length : ℕ), 
    unknown_length = 3 ∧ 
    n_unknown * unknown_length + n_known * known_length = total_duration :=
by sorry

end playlist_song_length_l248_24832


namespace dans_gift_l248_24898

-- Define the number of cards Sally sold
def cards_sold : ℕ := 27

-- Define the number of cards Sally bought
def cards_bought : ℕ := 20

-- Define the total number of cards Sally has now
def total_cards : ℕ := 34

-- Theorem to prove
theorem dans_gift (cards_from_dan : ℕ) : 
  cards_from_dan = total_cards - cards_bought := by
  sorry

#check dans_gift

end dans_gift_l248_24898


namespace average_of_three_numbers_l248_24807

theorem average_of_three_numbers (x : ℝ) : (15 + 19 + x) / 3 = 20 → x = 26 := by
  sorry

end average_of_three_numbers_l248_24807


namespace tank_emptied_in_two_minutes_l248_24869

/-- Represents the state and properties of a water tank system -/
structure WaterTank where
  initial_fill : ℚ
  fill_rate : ℚ
  empty_rate : ℚ

/-- Calculates the time to empty or fill the tank completely -/
def time_to_complete (tank : WaterTank) : ℚ :=
  let combined_rate := tank.fill_rate - tank.empty_rate
  let amount_to_change := 1 - tank.initial_fill
  amount_to_change / (-combined_rate)

/-- Theorem stating that the tank will be emptied in 2 minutes -/
theorem tank_emptied_in_two_minutes :
  let tank := WaterTank.mk (1/5) (1/15) (1/6)
  time_to_complete tank = 2 := by
  sorry

end tank_emptied_in_two_minutes_l248_24869


namespace fencing_required_l248_24844

/-- Calculates the fencing required for a rectangular field with given area and one uncovered side. -/
theorem fencing_required (area : ℝ) (uncovered_side : ℝ) : area = 720 ∧ uncovered_side = 20 →
  uncovered_side + 2 * (area / uncovered_side) = 92 := by
  sorry

end fencing_required_l248_24844


namespace boa_constrictors_count_l248_24878

/-- The number of boa constrictors in the park -/
def num_boa : ℕ := sorry

/-- The number of pythons in the park -/
def num_python : ℕ := sorry

/-- The number of rattlesnakes in the park -/
def num_rattlesnake : ℕ := 40

/-- The total number of snakes in the park -/
def total_snakes : ℕ := 200

theorem boa_constrictors_count :
  (num_boa + num_python + num_rattlesnake = total_snakes) →
  (num_python = 3 * num_boa) →
  (num_boa = 40) :=
by sorry

end boa_constrictors_count_l248_24878


namespace complex_equation_solution_l248_24819

theorem complex_equation_solution (a : ℝ) : (Complex.I + a) * (1 - a * Complex.I) = 2 → a = 1 := by
  sorry

end complex_equation_solution_l248_24819


namespace r_value_when_m_is_3_l248_24802

theorem r_value_when_m_is_3 (m : ℕ) (t : ℕ) (r : ℕ) : 
  m = 3 → 
  t = 3^m + 2 → 
  r = 4^t - 2*t → 
  r = 4^29 - 58 := by
sorry

end r_value_when_m_is_3_l248_24802


namespace difference_of_sum_and_difference_of_squares_l248_24805

theorem difference_of_sum_and_difference_of_squares 
  (x y : ℝ) 
  (h1 : x + y = 6) 
  (h2 : x^2 - y^2 = 24) : 
  x - y = 4 := by
sorry

end difference_of_sum_and_difference_of_squares_l248_24805


namespace inequality_system_solution_set_l248_24828

theorem inequality_system_solution_set :
  ∀ x : ℝ, (x + 1 > 2 ∧ 2*x - 4 < x) ↔ (1 < x ∧ x < 4) := by
  sorry

end inequality_system_solution_set_l248_24828


namespace soccer_ball_cost_is_6_l248_24813

/-- The cost of a soccer ball purchased by four friends -/
def soccer_ball_cost (x1 x2 x3 x4 : ℝ) : Prop :=
  x1 = 2.30 ∧
  x2 = (1/3) * (x1 + x3 + x4) ∧
  x3 = (1/4) * (x1 + x2 + x4) ∧
  x4 = (1/5) * (x1 + x2 + x3) ∧
  x1 + x2 + x3 + x4 = 6

theorem soccer_ball_cost_is_6 :
  ∃ x1 x2 x3 x4 : ℝ, soccer_ball_cost x1 x2 x3 x4 :=
sorry

end soccer_ball_cost_is_6_l248_24813


namespace marching_band_theorem_l248_24829

def marching_band_ratio (total_members brass_players : ℕ) : Prop :=
  ∃ (percussion woodwind : ℕ),
    -- Total members condition
    total_members = percussion + woodwind + brass_players ∧
    -- Woodwind is twice brass
    woodwind = 2 * brass_players ∧
    -- Percussion is a multiple of woodwind
    ∃ (k : ℕ), percussion = k * woodwind ∧
    -- Ratio of percussion to woodwind is 4:1
    percussion = 4 * woodwind

theorem marching_band_theorem :
  marching_band_ratio 110 10 := by
  sorry

end marching_band_theorem_l248_24829


namespace smaller_number_in_ratio_l248_24897

theorem smaller_number_in_ratio (a b c x y : ℝ) : 
  0 < a → a < b → x > 0 → y > 0 → x / y = a / b → x * y = c → 
  x < y ∧ x = Real.sqrt (a * c / b) := by
  sorry

end smaller_number_in_ratio_l248_24897


namespace evaluate_expression_l248_24840

theorem evaluate_expression : 
  2000 * 1995 * 0.1995 - 10 = 0.2 * 1995^2 - 10 := by
  sorry

end evaluate_expression_l248_24840


namespace series_sum_equals_81_and_two_fifths_l248_24884

def series_sum : ℚ :=
  1 + 3 * (1/6) + 5 * (1/12) + 7 * (1/20) + 9 * (1/30) + 11 * (1/42) + 
  13 * (1/56) + 15 * (1/72) + 17 * (1/90)

theorem series_sum_equals_81_and_two_fifths : 
  series_sum = 81 + 2/5 := by sorry

end series_sum_equals_81_and_two_fifths_l248_24884


namespace petya_wins_2021_petya_wins_l248_24824

/-- Represents the game state -/
structure GameState :=
  (piles : ℕ)

/-- Represents a player in the game -/
inductive Player
  | Petya
  | Vasya

/-- Defines a valid move in the game -/
def valid_move (state : GameState) : Prop :=
  state.piles ≥ 3

/-- Applies a move to the game state -/
def apply_move (state : GameState) : GameState :=
  { piles := state.piles - 2 }

/-- Determines the winner of the game -/
def winner (initial_piles : ℕ) : Player :=
  if initial_piles % 2 = 0 then Player.Vasya else Player.Petya

/-- Theorem stating that Petya wins the game with 2021 initial piles -/
theorem petya_wins_2021 : winner 2021 = Player.Petya := by
  sorry

/-- Main theorem proving Petya's victory -/
theorem petya_wins :
  ∀ (initial_state : GameState),
    initial_state.piles = 2021 →
    winner initial_state.piles = Player.Petya := by
  sorry

end petya_wins_2021_petya_wins_l248_24824


namespace determinant_cubic_roots_l248_24852

theorem determinant_cubic_roots (p q k : ℝ) (a b c : ℝ) : 
  a^3 + p*a + q = 0 → 
  b^3 + p*b + q = 0 → 
  c^3 + p*c + q = 0 → 
  Matrix.det !![k + a, 1, 1; 1, k + b, 1; 1, 1, k + c] = k^3 + k*p - q := by
  sorry

end determinant_cubic_roots_l248_24852


namespace prob_both_blue_is_25_64_l248_24834

/-- Represents the contents of a jar --/
structure JarContents where
  red : ℕ
  blue : ℕ

/-- The probability of selecting a blue button from a jar --/
def prob_blue (jar : JarContents) : ℚ :=
  jar.blue / (jar.red + jar.blue)

/-- The initial contents of Jar C --/
def initial_jar_c : JarContents :=
  { red := 6, blue := 10 }

/-- The number of buttons removed from Jar C --/
def removed : JarContents :=
  { red := 3, blue := 5 }

/-- The contents of Jar C after removal --/
def final_jar_c : JarContents :=
  { red := initial_jar_c.red - removed.red,
    blue := initial_jar_c.blue - removed.blue }

/-- The contents of Jar D after removal --/
def jar_d : JarContents := removed

theorem prob_both_blue_is_25_64 :
  (prob_blue final_jar_c * prob_blue jar_d = 25 / 64) ∧
  (final_jar_c.red + final_jar_c.blue = (initial_jar_c.red + initial_jar_c.blue) / 2) :=
sorry

end prob_both_blue_is_25_64_l248_24834


namespace abs_x_eq_3_system_solution_2_system_solution_3_l248_24890

-- Part 1
theorem abs_x_eq_3 (x : ℝ) : |x| = 3 ↔ x = 3 ∨ x = -3 := by sorry

-- Part 2
theorem system_solution_2 (x y : ℝ) : 
  y * (x - 1) = 0 ∧ 2 * x + 5 * y = 7 ↔ 
  (x = 7/2 ∧ y = 0) ∨ (x = 1 ∧ y = 1) := by sorry

-- Part 3
theorem system_solution_3 (x y : ℝ) :
  x * y - 2 * x - y + 2 = 0 ∧ x + 6 * y = 3 ∧ 3 * x + y = 8 ↔
  (x = 1 ∧ y = 5) ∨ (x = 2 ∧ y = 2) := by sorry

end abs_x_eq_3_system_solution_2_system_solution_3_l248_24890


namespace sum_fraction_equality_l248_24877

theorem sum_fraction_equality (x y z : ℝ) (h : x + y + z = 1) :
  (x*y + y*z + z*x) / (x^2 + y^2 + z^2) = (1 - (x^2 + y^2 + z^2)) / (2*(x^2 + y^2 + z^2)) := by
  sorry

end sum_fraction_equality_l248_24877


namespace infinite_sum_equals_one_tenth_l248_24873

/-- The infinite sum of n^2 / (n^6 + 5) from n = 0 to infinity equals 1/10 -/
theorem infinite_sum_equals_one_tenth :
  (∑' n : ℕ, (n^2 : ℝ) / (n^6 + 5)) = 1/10 := by sorry

end infinite_sum_equals_one_tenth_l248_24873


namespace rectangle_cut_and_rearrange_l248_24809

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℕ := r.width * r.height

/-- Represents a cut of a rectangle into two parts -/
structure Cut where
  original : Rectangle
  part1 : Rectangle
  part2 : Rectangle

/-- Checks if a cut is valid (preserves area) -/
def Cut.isValid (c : Cut) : Prop :=
  c.original.area = c.part1.area + c.part2.area

/-- Theorem: A 14x6 rectangle can be cut into two parts that form a 21x4 rectangle -/
theorem rectangle_cut_and_rearrange :
  ∃ (c : Cut),
    c.original = { width := 14, height := 6 } ∧
    c.isValid ∧
    ∃ (new : Rectangle),
      new = { width := 21, height := 4 } ∧
      new.area = c.part1.area + c.part2.area :=
sorry

end rectangle_cut_and_rearrange_l248_24809


namespace work_completion_time_l248_24880

/-- The time taken to complete a work given two workers with different rates and a partial work completion scenario -/
theorem work_completion_time
  (amit_rate : ℚ)
  (ananthu_rate : ℚ)
  (amit_days : ℕ)
  (h_amit_rate : amit_rate = 1 / 15)
  (h_ananthu_rate : ananthu_rate = 1 / 45)
  (h_amit_days : amit_days = 3)
  : ∃ (total_days : ℕ), total_days = amit_days + ⌈(1 - amit_rate * amit_days) / ananthu_rate⌉ ∧ total_days = 39 := by
  sorry

#check work_completion_time

end work_completion_time_l248_24880


namespace car_distance_proof_l248_24858

theorem car_distance_proof (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 125 → time = 3 → distance = speed * time → distance = 375 := by
  sorry

end car_distance_proof_l248_24858


namespace intersection_radius_l248_24833

/-- A sphere intersecting planes -/
structure IntersectingSphere where
  /-- Center of the circle in xz-plane -/
  xz_center : ℝ × ℝ × ℝ
  /-- Radius of the circle in xz-plane -/
  xz_radius : ℝ
  /-- Center of the circle in xy-plane -/
  xy_center : ℝ × ℝ × ℝ
  /-- Radius of the circle in xy-plane -/
  xy_radius : ℝ

/-- The theorem stating the radius of the xy-plane intersection -/
theorem intersection_radius (sphere : IntersectingSphere) 
  (h1 : sphere.xz_center = (3, 0, 3))
  (h2 : sphere.xz_radius = 2)
  (h3 : sphere.xy_center = (3, 3, 0)) :
  sphere.xy_radius = 3 := by
  sorry


end intersection_radius_l248_24833


namespace tom_vegetable_ratio_l248_24868

/-- The ratio of broccoli to carrots eaten by Tom -/
def broccoli_to_carrots_ratio : ℚ := by sorry

theorem tom_vegetable_ratio :
  let carrot_calories_per_pound : ℚ := 51
  let carrot_amount : ℚ := 1
  let broccoli_calories_per_pound : ℚ := carrot_calories_per_pound / 3
  let total_calories : ℚ := 85
  let broccoli_amount : ℚ := (total_calories - carrot_calories_per_pound * carrot_amount) / broccoli_calories_per_pound
  broccoli_to_carrots_ratio = broccoli_amount / carrot_amount := by sorry

end tom_vegetable_ratio_l248_24868


namespace crayon_selection_proof_l248_24806

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem crayon_selection_proof : choose 15 3 = 455 := by
  sorry

end crayon_selection_proof_l248_24806


namespace question_1_question_2_question_3_l248_24846

-- Define the functions f and g
def f (k : ℝ) (x : ℝ) : ℝ := 8 * x^2 + 16 * x - k
def g (x : ℝ) : ℝ := 2 * x^3 + 5 * x^2 + 4 * x

-- Define the interval [-3, 3]
def I : Set ℝ := Set.Icc (-3) 3

-- Statement for question 1
theorem question_1 (k : ℝ) : 
  (∀ x ∈ I, f k x ≤ g x) ↔ k ≥ 45 := by sorry

-- Statement for question 2
theorem question_2 (k : ℝ) : 
  (∃ x ∈ I, f k x ≤ g x) ↔ k ≥ -7 := by sorry

-- Statement for question 3
theorem question_3 (k : ℝ) : 
  (∀ x₁ ∈ I, ∀ x₂ ∈ I, f k x₁ ≤ g x₂) ↔ k ≥ 141 := by sorry

end question_1_question_2_question_3_l248_24846


namespace equation_solutions_l248_24808

theorem equation_solutions : 
  {x : ℝ | (2010 + 2*x)^2 = x^2} = {-2010, -670} := by sorry

end equation_solutions_l248_24808


namespace range_of_a_l248_24814

def proposition_p (a : ℝ) : Prop :=
  ∀ m : ℝ, m ∈ Set.Icc (-1) 1 → a^2 - 5*a + 7 ≥ m + 2

def proposition_q (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + a*x₁ = 2 ∧ x₂^2 + a*x₂ = 2

theorem range_of_a :
  ∃ S : Set ℝ, (∀ a : ℝ, (proposition_p a ∨ proposition_q a) ∧ ¬(proposition_p a ∧ proposition_q a) ↔ a ∈ S) ∧
  S = {a : ℝ | -2*Real.sqrt 2 ≤ a ∧ a ≤ 1 ∨ 2*Real.sqrt 2 < a ∧ a < 4} :=
sorry

end range_of_a_l248_24814


namespace smallest_three_digit_divisible_by_parts_l248_24826

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def first_digit (n : ℕ) : ℕ := n / 100

def second_digit (n : ℕ) : ℕ := (n / 10) % 10

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem smallest_three_digit_divisible_by_parts : 
  ∃ (n : ℕ), is_three_digit n ∧ 
  first_digit n ≠ 0 ∧
  n % (n / 10) = 0 ∧ 
  n % (last_two_digits n) = 0 ∧
  ∀ m, is_three_digit m ∧ 
       first_digit m ≠ 0 ∧ 
       m % (m / 10) = 0 ∧ 
       m % (last_two_digits m) = 0 → 
       n ≤ m ∧
  n = 110 := by
sorry

end smallest_three_digit_divisible_by_parts_l248_24826


namespace set_inclusion_implies_a_geq_two_l248_24854

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | x < a}

-- State the theorem
theorem set_inclusion_implies_a_geq_two (a : ℝ) :
  A ⊆ B a → a ≥ 2 := by
  sorry

end set_inclusion_implies_a_geq_two_l248_24854


namespace principal_is_15000_l248_24861

/-- Calculates the principal amount given simple interest, rate, and time -/
def calculate_principal (simple_interest : ℕ) (rate : ℚ) (time : ℕ) : ℚ :=
  (simple_interest : ℚ) * 100 / (rate * time)

/-- Theorem: Given the specified conditions, the principal sum is 15000 -/
theorem principal_is_15000 :
  let simple_interest : ℕ := 2700
  let rate : ℚ := 6 / 100
  let time : ℕ := 3
  calculate_principal simple_interest rate time = 15000 := by
  sorry

end principal_is_15000_l248_24861


namespace water_lost_is_eight_gallons_l248_24825

/-- Represents the water filling and leaking scenario of a pool --/
structure PoolFilling where
  hour1_rate : ℝ
  hour2_3_rate : ℝ
  hour4_rate : ℝ
  final_amount : ℝ

/-- Calculates the amount of water lost due to the leak --/
def water_lost (p : PoolFilling) : ℝ :=
  p.hour1_rate * 1 + p.hour2_3_rate * 2 + p.hour4_rate * 1 - p.final_amount

/-- Theorem stating that for the given scenario, the water lost is 8 gallons --/
theorem water_lost_is_eight_gallons : 
  ∀ (p : PoolFilling), 
  p.hour1_rate = 8 ∧ 
  p.hour2_3_rate = 10 ∧ 
  p.hour4_rate = 14 ∧ 
  p.final_amount = 34 → 
  water_lost p = 8 := by
  sorry


end water_lost_is_eight_gallons_l248_24825


namespace airline_capacity_example_l248_24838

/-- Calculates the total number of passengers an airline can accommodate daily --/
def airline_capacity (num_airplanes : ℕ) (rows_per_airplane : ℕ) (seats_per_row : ℕ) (flights_per_day : ℕ) : ℕ :=
  num_airplanes * rows_per_airplane * seats_per_row * flights_per_day

/-- Theorem: An airline with 5 airplanes, 20 rows per airplane, 7 seats per row, and 2 flights per day can accommodate 1400 passengers daily --/
theorem airline_capacity_example : airline_capacity 5 20 7 2 = 1400 := by
  sorry

end airline_capacity_example_l248_24838


namespace matrix_vector_computation_l248_24820

variable (M : Matrix (Fin 2) (Fin 2) ℝ)
variable (u v w : Fin 2 → ℝ)

theorem matrix_vector_computation
  (hu : M.mulVec u = ![(-3), 4])
  (hv : M.mulVec v = ![2, (-7)])
  (hw : M.mulVec w = ![9, 0]) :
  M.mulVec (3 • u - 4 • v + 2 • w) = ![1, 40] := by
sorry

end matrix_vector_computation_l248_24820


namespace binomial_12_choose_6_l248_24845

theorem binomial_12_choose_6 : Nat.choose 12 6 = 1848 := by
  sorry

end binomial_12_choose_6_l248_24845


namespace hcf_is_three_l248_24892

-- Define the properties of our two numbers
def number_properties (a b : ℕ) : Prop :=
  ∃ (k : ℕ), a = 3 * k ∧ b = 4 * k ∧ Nat.lcm a b = 36

-- Theorem statement
theorem hcf_is_three {a b : ℕ} (h : number_properties a b) : Nat.gcd a b = 3 := by
  sorry

end hcf_is_three_l248_24892


namespace units_digit_of_j_squared_plus_three_to_j_l248_24867

def j : ℕ := 19^2 + 3^10

theorem units_digit_of_j_squared_plus_three_to_j (j : ℕ := 19^2 + 3^10) : 
  (j^2 + 3^j) % 10 = 3 := by
  sorry

end units_digit_of_j_squared_plus_three_to_j_l248_24867


namespace percentage_problem_l248_24888

theorem percentage_problem (P : ℝ) : 25 = (P / 100) * 25 + 21 → P = 16 := by
  sorry

end percentage_problem_l248_24888


namespace tom_customers_per_hour_l248_24812

/-- The number of customers Tom served per hour -/
def customers_per_hour : ℝ := 10

/-- The number of hours Tom worked -/
def hours_worked : ℝ := 8

/-- The bonus point percentage (20% = 0.2) -/
def bonus_percentage : ℝ := 0.2

/-- The total bonus points Tom earned -/
def total_bonus_points : ℝ := 16

theorem tom_customers_per_hour :
  customers_per_hour * hours_worked * bonus_percentage = total_bonus_points :=
by sorry

end tom_customers_per_hour_l248_24812


namespace inequality_for_positive_product_l248_24885

theorem inequality_for_positive_product (a b : ℝ) (h : a * b > 0) :
  b / a + a / b ≥ 2 := by sorry

end inequality_for_positive_product_l248_24885


namespace inequality_not_always_true_l248_24865

theorem inequality_not_always_true (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ¬ (∀ a b, a > b ∧ b > 0 → a + b < 2 * Real.sqrt (a * b)) :=
sorry

end inequality_not_always_true_l248_24865


namespace evaluate_expression_l248_24857

theorem evaluate_expression (a : ℝ) (h : a = 2) : 
  8^3 + 4*a*(8^2) + 6*(a^2)*8 + a^3 = 1224 := by
  sorry

end evaluate_expression_l248_24857


namespace annual_interest_proof_l248_24863

/-- Calculates the simple interest for a loan -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Proves that the annual interest for a $9,000 loan at 9% simple interest is $810 -/
theorem annual_interest_proof :
  let principal : ℝ := 9000
  let rate : ℝ := 0.09
  let time : ℝ := 1
  simple_interest principal rate time = 810 := by
sorry


end annual_interest_proof_l248_24863


namespace min_value_of_expression_l248_24822

theorem min_value_of_expression (x y : ℝ) :
  (2 * x * y - 1)^2 + (x - y)^2 ≥ 0 ∧
  ∃ a b : ℝ, (2 * a * b - 1)^2 + (a - b)^2 = 0 := by
  sorry

end min_value_of_expression_l248_24822


namespace hot_dog_price_is_two_l248_24837

/-- Calculates the price of a single hot dog given the hourly sales rate, operating hours, and total sales. -/
def hot_dog_price (hourly_rate : ℕ) (hours : ℕ) (total_sales : ℕ) : ℚ :=
  total_sales / (hourly_rate * hours)

/-- Theorem stating that the price of each hot dog is $2 under given conditions. -/
theorem hot_dog_price_is_two :
  hot_dog_price 10 10 200 = 2 := by
  sorry

#eval hot_dog_price 10 10 200

end hot_dog_price_is_two_l248_24837


namespace quadratic_equation_solution_l248_24801

theorem quadratic_equation_solution (m : ℝ) : 
  (∃ x : ℝ, x^2 + m*x - 2 = 0 ∧ x = 1) → 
  (∃ y : ℝ, y^2 + m*y - 2 = 0 ∧ y = -2) :=
by sorry

end quadratic_equation_solution_l248_24801


namespace average_speed_calculation_l248_24810

/-- Calculates the average speed given distances and speeds for multiple segments of a ride -/
theorem average_speed_calculation 
  (d₁ d₂ d₃ : ℝ) 
  (v₁ v₂ v₃ : ℝ) 
  (h₁ : d₁ = 50)
  (h₂ : d₂ = 20)
  (h₃ : d₃ = 10)
  (h₄ : v₁ = 12)
  (h₅ : v₂ = 40)
  (h₆ : v₃ = 20) :
  (d₁ + d₂ + d₃) / ((d₁ / v₁) + (d₂ / v₂) + (d₃ / v₃)) = 480 / 31 := by
  sorry

#check average_speed_calculation

end average_speed_calculation_l248_24810


namespace geometric_progression_fourth_term_l248_24896

theorem geometric_progression_fourth_term : 
  ∀ (a : ℝ) (r : ℝ),
  a > 0 → r > 0 →
  a = 2^(1/3 : ℝ) →
  a * r = 2^(1/4 : ℝ) →
  a * r^2 = 2^(1/5 : ℝ) →
  a * r^3 = 2^(1/9 : ℝ) := by
sorry

end geometric_progression_fourth_term_l248_24896


namespace alloy_mixture_problem_l248_24860

/-- Represents the composition of an alloy --/
structure Alloy where
  component1 : ℝ
  component2 : ℝ
  ratio : ℚ

/-- Represents the mixture of two alloys --/
structure Mixture where
  alloyA : Alloy
  alloyB : Alloy
  massA : ℝ
  massB : ℝ
  tinTotal : ℝ

/-- The theorem to be proved --/
theorem alloy_mixture_problem (m : Mixture) : 
  m.alloyA.ratio = 1/3 ∧ 
  m.alloyB.ratio = 3/5 ∧ 
  m.massA = 170 ∧ 
  m.tinTotal = 221.25 → 
  m.massB = 250 := by
  sorry

end alloy_mixture_problem_l248_24860


namespace triangle_altitude_median_equations_l248_24851

/-- Triangle ABC with given coordinates -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given triangle ABC, return the equation of the altitude from C to AB -/
def altitude (t : Triangle) : LineEquation :=
  sorry

/-- Given triangle ABC, return the equation of the median from C to AB -/
def median (t : Triangle) : LineEquation :=
  sorry

theorem triangle_altitude_median_equations :
  let t : Triangle := { A := (3, 3), B := (2, -2), C := (-7, 1) }
  (altitude t = { a := 1, b := 5, c := 2 }) ∧
  (median t = { a := 1, b := 19, c := -12 }) := by
  sorry

end triangle_altitude_median_equations_l248_24851


namespace odd_prime_non_divisibility_l248_24827

theorem odd_prime_non_divisibility (p r : ℕ) : 
  Prime p → Odd p → Odd r → ¬(p * r + 1 ∣ p^p - 1) := by
  sorry

end odd_prime_non_divisibility_l248_24827


namespace complex_multiplication_l248_24817

theorem complex_multiplication (R S T : ℂ) : 
  R = 3 + 4*I ∧ S = 2*I ∧ T = 3 - 4*I → R * S * T = 50 * I :=
by
  sorry

end complex_multiplication_l248_24817


namespace investment_problem_l248_24853

theorem investment_problem (initial_investment : ℝ) (growth_rate_year1 : ℝ) (growth_rate_year2 : ℝ) (final_value : ℝ) (amount_added : ℝ) : 
  initial_investment = 80 →
  growth_rate_year1 = 0.15 →
  growth_rate_year2 = 0.10 →
  final_value = 132 →
  amount_added = 28 →
  final_value = (initial_investment * (1 + growth_rate_year1) + amount_added) * (1 + growth_rate_year2) :=
by sorry

#check investment_problem

end investment_problem_l248_24853


namespace profit_calculation_l248_24894

def number_of_bags : ℕ := 100
def selling_price : ℚ := 10
def buying_price : ℚ := 7

theorem profit_calculation :
  (number_of_bags : ℚ) * (selling_price - buying_price) = 300 := by sorry

end profit_calculation_l248_24894


namespace quadratic_decreasing_implies_a_range_l248_24870

/-- A quadratic function f(x) = x^2 + 2(a-1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

/-- The theorem states that if f(x) is decreasing on (-∞, 4], then a < -5 -/
theorem quadratic_decreasing_implies_a_range (a : ℝ) :
  (∀ x ≤ 4, ∀ y ≤ 4, x < y → f a x > f a y) →
  a < -5 :=
sorry

end quadratic_decreasing_implies_a_range_l248_24870


namespace perpendicular_lines_condition_l248_24886

-- Define the slopes of the two lines
def slope1 (a : ℝ) := a
def slope2 (a : ℝ) := -4 * a

-- Define perpendicularity condition
def isPerpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem perpendicular_lines_condition (a : ℝ) :
  (isPerpendicular (slope1 a) (slope2 a) → (a = 1/2 ∨ a = -1/2)) ∧
  ¬(a = 1/2 → isPerpendicular (slope1 a) (slope2 a)) :=
sorry

end perpendicular_lines_condition_l248_24886


namespace prob_at_least_two_heads_l248_24850

-- Define the number of coins
def n : ℕ := 5

-- Define the probability of getting heads on a single coin toss
def p : ℚ := 1/2

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the probability of getting exactly k heads in n tosses
def prob_exactly (k : ℕ) : ℚ := (binomial n k : ℚ) * p^n

-- State the theorem
theorem prob_at_least_two_heads :
  1 - (prob_exactly 0 + prob_exactly 1) = 13/16 := by sorry

end prob_at_least_two_heads_l248_24850


namespace amy_biking_distance_l248_24831

def miles_yesterday : ℕ := 12

def miles_today (y : ℕ) : ℕ := 2 * y - 3

def total_miles (y t : ℕ) : ℕ := y + t

theorem amy_biking_distance :
  total_miles miles_yesterday (miles_today miles_yesterday) = 33 :=
by sorry

end amy_biking_distance_l248_24831


namespace bicycle_sale_profit_l248_24866

/-- Profit percentage calculation for a bicycle sale chain --/
theorem bicycle_sale_profit (cost_price_A : ℝ) (profit_percent_A : ℝ) (price_C : ℝ)
  (h1 : cost_price_A = 150)
  (h2 : profit_percent_A = 20)
  (h3 : price_C = 225) :
  let price_B := cost_price_A * (1 + profit_percent_A / 100)
  let profit_B := price_C - price_B
  let profit_percent_B := (profit_B / price_B) * 100
  profit_percent_B = 25 := by
sorry

end bicycle_sale_profit_l248_24866


namespace inequality_proof_l248_24842

theorem inequality_proof (a b c d e : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0) 
  (h_prod : a * b * c * d * e = 1) : 
  (a + a*b*c) / (1 + a*b + a*b*c*d) + 
  (b + b*c*d) / (1 + b*c + b*c*d*e) + 
  (c + c*d*e) / (1 + c*d + c*d*e*a) + 
  (d + d*e*a) / (1 + d*e + d*e*a*b) + 
  (e + e*a*b) / (1 + e*a + e*a*b*c) ≥ 10/3 := by
sorry

end inequality_proof_l248_24842


namespace g_properties_l248_24864

-- Define the function g
def g : ℝ → ℝ := fun x ↦ -x

-- Theorem stating that g is an odd function and monotonically decreasing
theorem g_properties :
  (∀ x : ℝ, g (-x) = -g x) ∧ 
  (∀ x y : ℝ, x < y → g x > g y) :=
by
  sorry


end g_properties_l248_24864


namespace max_integer_solutions_l248_24862

/-- A quadratic function f(x) = ax^2 + bx + c where a > 100 -/
def QuadraticFunction (a b c : ℝ) (h : a > 100) := fun (x : ℤ) => a * x^2 + b * x + c

/-- The maximum number of integer solutions for |f(x)| ≤ 50 is at most 2 -/
theorem max_integer_solutions (a b c : ℝ) (h : a > 100) :
  ∃ (n : ℕ), n ≤ 2 ∧ 
  ∀ (S : Finset ℤ), (∀ x ∈ S, |QuadraticFunction a b c h x| ≤ 50) → S.card ≤ n :=
sorry

end max_integer_solutions_l248_24862
