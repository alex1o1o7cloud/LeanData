import Mathlib

namespace left_handed_fraction_l2174_217444

theorem left_handed_fraction (red blue : ℕ) (h_ratio : red = blue) 
  (h_red_left : red / 3 = red.div 3) 
  (h_blue_left : 2 * (blue / 3) = blue.div 3 * 2) : 
  (red.div 3 + blue.div 3 * 2) / (red + blue) = 1 / 2 := by
  sorry

end left_handed_fraction_l2174_217444


namespace tangent_line_and_minimum_l2174_217418

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x + Real.log x - 1

theorem tangent_line_and_minimum (a : ℝ) :
  (∃ y, x - 4 * y + 4 * Real.log 2 - 4 = 0 ↔ 
    y = f 1 x ∧ x = 2) ∧
  (a ≤ 0 → ∀ x ∈ Set.Ioo 0 (Real.exp 1), ∃ y ∈ Set.Ioo 0 (Real.exp 1), f a y < f a x) ∧
  (0 < a → a < Real.exp 1 → ∀ x ∈ Set.Ioo 0 (Real.exp 1), f a a ≤ f a x ∧ f a a = Real.log a) ∧
  (Real.exp 1 ≤ a → ∀ x ∈ Set.Ioo 0 (Real.exp 1), a / Real.exp 1 ≤ f a x ∧ a / Real.exp 1 = f a (Real.exp 1)) :=
by sorry

end tangent_line_and_minimum_l2174_217418


namespace square_identities_l2174_217416

theorem square_identities (a b c : ℝ) : 
  ((a + b)^2 = a^2 + 2*a*b + b^2) ∧ 
  ((a - b)^2 = a^2 - 2*a*b + b^2) ∧ 
  ((a + b + c)^2 = a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c) := by
  sorry

end square_identities_l2174_217416


namespace sum_15_terms_eq_56_25_l2174_217478

/-- An arithmetic progression with specific properties -/
structure ArithmeticProgression where
  -- The 11th term is 5.25
  a11 : ℝ
  a11_eq : a11 = 5.25
  -- The 7th term is 3.25
  a7 : ℝ
  a7_eq : a7 = 3.25

/-- The sum of the first 15 terms of the arithmetic progression -/
def sum_15_terms (ap : ArithmeticProgression) : ℝ :=
  -- Definition of the sum (to be proved)
  56.25

/-- Theorem stating that the sum of the first 15 terms is 56.25 -/
theorem sum_15_terms_eq_56_25 (ap : ArithmeticProgression) :
  sum_15_terms ap = 56.25 := by
  sorry

end sum_15_terms_eq_56_25_l2174_217478


namespace atop_difference_l2174_217476

-- Define the @ operation
def atop (x y : ℤ) : ℤ := x * y + x - y

-- Theorem statement
theorem atop_difference : (atop 7 4) - (atop 4 7) = 6 := by
  sorry

end atop_difference_l2174_217476


namespace ruby_height_l2174_217479

/-- Given the heights of various people, prove Ruby's height --/
theorem ruby_height
  (janet_height : ℕ)
  (charlene_height : ℕ)
  (pablo_height : ℕ)
  (ruby_height : ℕ)
  (h1 : janet_height = 62)
  (h2 : charlene_height = 2 * janet_height)
  (h3 : pablo_height = charlene_height + 70)
  (h4 : ruby_height = pablo_height - 2)
  : ruby_height = 192 := by
  sorry

#check ruby_height

end ruby_height_l2174_217479


namespace sports_day_theorem_l2174_217428

/-- Represents the score awarded to a class in a single event -/
structure EventScore where
  first : ℕ
  second : ℕ
  third : ℕ
  first_gt_second : first > second
  second_gt_third : second > third

/-- Represents the total scores of all classes -/
structure TotalScores where
  scores : List ℕ
  four_classes : scores.length = 4

/-- The Sports Day competition setup -/
structure SportsDay where
  event_score : EventScore
  total_scores : TotalScores
  events_count : ℕ
  events_count_eq_five : events_count = 5
  scores_sum_eq_events_total : total_scores.scores.sum = events_count * (event_score.first + event_score.second + event_score.third)

theorem sports_day_theorem (sd : SportsDay) 
  (h_scores : sd.total_scores.scores = [21, 6, 9, 4]) : 
  sd.event_score.first + sd.event_score.second + sd.event_score.third = 8 ∧ 
  sd.event_score.first = 5 := by
  sorry

end sports_day_theorem_l2174_217428


namespace correct_assembly_rates_l2174_217402

/-- Represents the assembly and disassembly rates of coffee grinders for two robots -/
structure CoffeeGrinderRates where
  hubert_assembly : ℝ     -- Hubert's assembly rate (grinders per hour)
  robert_assembly : ℝ     -- Robert's assembly rate (grinders per hour)

/-- Checks if the given rates satisfy the problem conditions -/
def satisfies_conditions (rates : CoffeeGrinderRates) : Prop :=
  -- Each assembles four times faster than the other disassembles
  rates.hubert_assembly = 4 * (rates.robert_assembly / 4) ∧
  rates.robert_assembly = 4 * (rates.hubert_assembly / 4) ∧
  -- Morning shift conditions
  (rates.hubert_assembly - rates.robert_assembly / 4) * 3 = 27 ∧
  -- Afternoon shift conditions
  (rates.robert_assembly - rates.hubert_assembly / 4) * 6 = 120

/-- The theorem stating the correct assembly rates for Hubert and Robert -/
theorem correct_assembly_rates :
  ∃ (rates : CoffeeGrinderRates),
    satisfies_conditions rates ∧
    rates.hubert_assembly = 12 ∧
    rates.robert_assembly = 80 / 3 := by
  sorry


end correct_assembly_rates_l2174_217402


namespace slower_speed_calculation_l2174_217401

theorem slower_speed_calculation (actual_distance : ℝ) (faster_speed : ℝ) (additional_distance : ℝ)
  (h1 : actual_distance = 50)
  (h2 : faster_speed = 14)
  (h3 : additional_distance = 20) :
  ∃ x : ℝ, x > 0 ∧ actual_distance / x = (actual_distance + additional_distance) / faster_speed ∧ x = 10 := by
  sorry

end slower_speed_calculation_l2174_217401


namespace solution_set_inequalities_l2174_217483

theorem solution_set_inequalities :
  {x : ℝ | x - 2 > 1 ∧ x < 4} = {x : ℝ | 3 < x ∧ x < 4} := by
  sorry

end solution_set_inequalities_l2174_217483


namespace second_train_length_calculation_l2174_217470

/-- Calculates the length of the second train given the speeds of two trains,
    the length of the first train, and the time they take to clear each other. -/
def second_train_length (speed1 speed2 : ℝ) (length1 time : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let distance := relative_speed * time
  distance - length1

theorem second_train_length_calculation :
  let speed1 := 42 * (1000 / 3600)  -- Convert 42 kmph to m/s
  let speed2 := 30 * (1000 / 3600)  -- Convert 30 kmph to m/s
  let length1 := 200
  let time := 23.998
  abs (second_train_length speed1 speed2 length1 time - 279.96) < 0.01 :=
sorry

end second_train_length_calculation_l2174_217470


namespace polynomial_simplification_l2174_217450

theorem polynomial_simplification (q : ℝ) : 
  (4 * q^3 - 7 * q^2 + 3 * q - 2) + (5 * q^2 - 9 * q + 8) = 4 * q^3 - 2 * q^2 - 6 * q + 6 := by
  sorry

end polynomial_simplification_l2174_217450


namespace rectangle_area_l2174_217458

theorem rectangle_area (x : ℝ) (h : x > 0) :
  ∃ (w : ℝ), w > 0 ∧ 
  w^2 + (3*w)^2 = x^2 ∧ 
  3*w^2 = (3/10)*x^2 :=
by sorry

end rectangle_area_l2174_217458


namespace total_rooms_to_paint_l2174_217453

theorem total_rooms_to_paint 
  (time_per_room : ℕ) 
  (rooms_painted : ℕ) 
  (time_remaining : ℕ) : 
  time_per_room = 7 → 
  rooms_painted = 2 → 
  time_remaining = 63 → 
  rooms_painted + (time_remaining / time_per_room) = 11 :=
by sorry

end total_rooms_to_paint_l2174_217453


namespace x_intercept_implies_b_value_l2174_217441

/-- 
Given a line y = 2x - b with an x-intercept of 1, prove that b = 2.
-/
theorem x_intercept_implies_b_value (b : ℝ) :
  (∃ x : ℝ, x = 1 ∧ 2 * x - b = 0) → b = 2 := by
  sorry

end x_intercept_implies_b_value_l2174_217441


namespace min_value_theorem_l2174_217412

/-- The circle C: (x-2)^2+(y+1)^2=5 is symmetric with respect to the line ax-by-1=0 -/
def symmetric_circle (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), (x - 2)^2 + (y + 1)^2 = 5 ∧ a * x - b * y - 1 = 0

/-- The theorem stating the minimum value of 3/b + 2/a -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h_sym : symmetric_circle a b) : 
    (∀ x y : ℝ, x > 0 → y > 0 → symmetric_circle x y → 3/y + 2/x ≥ 7 + 4 * Real.sqrt 3) ∧ 
    (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ symmetric_circle x y ∧ 3/y + 2/x = 7 + 4 * Real.sqrt 3) :=
  sorry

end min_value_theorem_l2174_217412


namespace average_of_combined_results_l2174_217460

theorem average_of_combined_results :
  let n₁ : ℕ := 100
  let n₂ : ℕ := 75
  let avg₁ : ℚ := 45
  let avg₂ : ℚ := 65
  let total_sum : ℚ := n₁ * avg₁ + n₂ * avg₂
  let total_count : ℕ := n₁ + n₂
  total_sum / total_count = 9375 / 175 := by
  sorry

end average_of_combined_results_l2174_217460


namespace debby_deleted_pictures_l2174_217408

theorem debby_deleted_pictures (zoo_pics museum_pics remaining_pics : ℕ) 
  (h1 : zoo_pics = 24)
  (h2 : museum_pics = 12)
  (h3 : remaining_pics = 22) :
  zoo_pics + museum_pics - remaining_pics = 14 := by
  sorry

end debby_deleted_pictures_l2174_217408


namespace print_shop_pricing_l2174_217421

/-- The price per color copy at print shop Y -/
def price_Y : ℚ := 2.75

/-- The number of color copies -/
def num_copies : ℕ := 40

/-- The additional charge at print shop Y compared to print shop X for 40 copies -/
def additional_charge : ℚ := 60

/-- The price per color copy at print shop X -/
def price_X : ℚ := 1.25

theorem print_shop_pricing :
  price_Y * num_copies = price_X * num_copies + additional_charge :=
sorry

end print_shop_pricing_l2174_217421


namespace geometric_sequence_ratio_l2174_217439

/-- A geometric sequence with sum of first n terms Sn -/
def geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n, n > 0 → a n = a 1 * r^(n-1) ∧ S n = a 1 * (1 - r^n) / (1 - r)

theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  geometric_sequence a S →
  a 1 + a 3 = 5/2 →
  a 2 + a 4 = 5/4 →
  ∀ n, n > 0 → S n / a n = 2^n - 1 :=
sorry

end geometric_sequence_ratio_l2174_217439


namespace not_necessary_nor_sufficient_condition_l2174_217452

theorem not_necessary_nor_sufficient_condition (m n : ℕ+) :
  ¬(∀ a b : ℝ, (a^m.val - b^m.val) * (a^n.val - b^n.val) > 0 → a > b) ∧
  ¬(∀ a b : ℝ, a > b → (a^m.val - b^m.val) * (a^n.val - b^n.val) > 0) :=
by sorry

end not_necessary_nor_sufficient_condition_l2174_217452


namespace farm_cows_count_l2174_217415

/-- The total number of cows on the farm -/
def total_cows : ℕ := 140

/-- The percentage of cows with a red spot -/
def red_spot_percentage : ℚ := 40 / 100

/-- The percentage of cows without a red spot that have a blue spot -/
def blue_spot_percentage : ℚ := 25 / 100

/-- The number of cows with no spot -/
def no_spot_cows : ℕ := 63

theorem farm_cows_count :
  (total_cows : ℚ) * (1 - red_spot_percentage) * (1 - blue_spot_percentage) = no_spot_cows :=
sorry

end farm_cows_count_l2174_217415


namespace final_result_calculation_l2174_217435

theorem final_result_calculation (chosen_number : ℤ) : 
  chosen_number = 120 → (chosen_number / 6 - 15 : ℚ) = 5 := by
  sorry

end final_result_calculation_l2174_217435


namespace sequence_length_correct_l2174_217405

/-- The number of terms in the arithmetic sequence from 5 to 2n-1 with a common difference of 2 -/
def sequence_length (n : ℕ) : ℕ :=
  n - 2

/-- The nth term of the sequence -/
def sequence_term (n : ℕ) : ℕ :=
  2 * n + 3

theorem sequence_length_correct (n : ℕ) :
  sequence_term (sequence_length n) = 2 * n - 1 :=
by sorry

end sequence_length_correct_l2174_217405


namespace nikita_mistaken_l2174_217485

theorem nikita_mistaken (b s : ℕ) : 
  (9 * b + 4 * s) - (4 * b + 9 * s) ≠ 49 := by
  sorry

end nikita_mistaken_l2174_217485


namespace complex_magnitude_three_fourths_plus_three_i_l2174_217481

theorem complex_magnitude_three_fourths_plus_three_i :
  Complex.abs (3 / 4 + 3 * Complex.I) = Real.sqrt 153 / 4 := by
  sorry

end complex_magnitude_three_fourths_plus_three_i_l2174_217481


namespace fish_distribution_theorem_l2174_217488

theorem fish_distribution_theorem (a b c d e f : ℕ) : 
  a + b + c + d + e + f = 100 ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
  d ≠ e ∧ d ≠ f ∧ 
  e ≠ f ∧
  (b + c + d + e + f) % 5 = 0 ∧
  (a + c + d + e + f) % 5 = 0 ∧
  (a + b + d + e + f) % 5 = 0 ∧
  (a + b + c + e + f) % 5 = 0 ∧
  (a + b + c + d + f) % 5 = 0 ∧
  (a + b + c + d + e) % 5 = 0 →
  a = 20 ∨ b = 20 ∨ c = 20 ∨ d = 20 ∨ e = 20 ∨ f = 20 :=
by sorry

end fish_distribution_theorem_l2174_217488


namespace megatek_graph_is_pie_chart_l2174_217417

-- Define the properties of the graph
structure EmployeeGraph where
  -- The graph is circular
  isCircular : Bool
  -- The angle of each sector is proportional to the quantity it represents
  isSectorProportional : Bool
  -- The manufacturing sector angle
  manufacturingAngle : ℝ
  -- The percentage of employees in manufacturing
  manufacturingPercentage : ℝ

-- Define a pie chart
def isPieChart (graph : EmployeeGraph) : Prop :=
  graph.isCircular ∧ 
  graph.isSectorProportional ∧
  graph.manufacturingAngle = 144 ∧
  graph.manufacturingPercentage = 40

-- Theorem to prove
theorem megatek_graph_is_pie_chart (graph : EmployeeGraph) 
  (h1 : graph.isCircular = true)
  (h2 : graph.isSectorProportional = true)
  (h3 : graph.manufacturingAngle = 144)
  (h4 : graph.manufacturingPercentage = 40) :
  isPieChart graph :=
sorry

end megatek_graph_is_pie_chart_l2174_217417


namespace rubber_elongation_improvement_l2174_217464

def n : ℕ := 10

def z_bar : ℝ := 11

def s_squared : ℝ := 61

def significant_improvement (z_bar s_squared : ℝ) : Prop :=
  z_bar ≥ 2 * Real.sqrt (s_squared / n)

theorem rubber_elongation_improvement :
  significant_improvement z_bar s_squared :=
sorry

end rubber_elongation_improvement_l2174_217464


namespace intersection_of_three_lines_l2174_217419

/-- A line represented by y = mx + b -/
structure Line where
  m : ℚ
  b : ℚ

/-- The point of intersection of two lines -/
def intersection (l1 l2 : Line) : ℚ × ℚ :=
  let x := (l2.b - l1.b) / (l1.m - l2.m)
  let y := l1.m * x + l1.b
  (x, y)

/-- Theorem: If three lines intersect at a single point, and two of them are
    y = 3x + 5 and y = -5x + 20, then the third line y = 4x + p must have p = 25/8 -/
theorem intersection_of_three_lines
  (l1 : Line)
  (l2 : Line)
  (l3 : Line)
  (h1 : l1 = ⟨3, 5⟩)
  (h2 : l2 = ⟨-5, 20⟩)
  (h3 : l3.m = 4)
  (h_intersect : intersection l1 l2 = intersection l2 l3) :
  l3.b = 25/8 := by
sorry

end intersection_of_three_lines_l2174_217419


namespace sin_45_degrees_l2174_217497

theorem sin_45_degrees : Real.sin (π / 4) = 1 / Real.sqrt 2 := by sorry

end sin_45_degrees_l2174_217497


namespace arithmetic_sequence_tenth_term_l2174_217455

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_tenth_term
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_third : a 3 = 23)
  (h_seventh : a 7 = 35) :
  a 10 = 44 := by
sorry

end arithmetic_sequence_tenth_term_l2174_217455


namespace opposite_numbers_theorem_l2174_217404

theorem opposite_numbers_theorem (a b c d : ℤ) : 
  (a + b = 0) → 
  (c = -1) → 
  (d = 1 ∨ d = -1) → 
  (2*a + 2*b - c*d = 1 ∨ 2*a + 2*b - c*d = -1) :=
by sorry

end opposite_numbers_theorem_l2174_217404


namespace situp_ratio_l2174_217493

theorem situp_ratio (ken_situps : ℕ) (nathan_ratio : ℚ) (bob_situps : ℕ) :
  ken_situps = 20 →
  bob_situps = (ken_situps + nathan_ratio * ken_situps) / 2 →
  bob_situps = ken_situps + 10 →
  nathan_ratio = 2 :=
by sorry

end situp_ratio_l2174_217493


namespace evaluate_expression_l2174_217477

theorem evaluate_expression : 150 * (150 - 5) - (150 * 150 + 13) = -763 := by
  sorry

end evaluate_expression_l2174_217477


namespace no_roots_implication_l2174_217449

theorem no_roots_implication (p q b c : ℝ) 
  (h1 : ∀ x : ℝ, x^2 + p*x + q ≠ 0)
  (h2 : ∀ x : ℝ, x^2 + b*x + c ≠ 0) :
  ∀ x : ℝ, 7*x^2 + (2*p + 3*b + 4)*x + 2*q + 3*c + 2 ≠ 0 := by
sorry

end no_roots_implication_l2174_217449


namespace curve_ellipse_equivalence_l2174_217424

-- Define the curve equation
def curve_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y+4)^2) + Real.sqrt (x^2 + (y-4)^2) = 10

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  y^2/25 + x^2/9 = 1

-- Theorem stating the equivalence of the two equations
theorem curve_ellipse_equivalence :
  ∀ x y : ℝ, curve_equation x y ↔ ellipse_equation x y :=
by sorry

end curve_ellipse_equivalence_l2174_217424


namespace binomial_sum_distinct_values_l2174_217425

theorem binomial_sum_distinct_values :
  ∃ (S : Finset ℕ), (∀ r : ℤ, 7 ≤ r ∧ r ≤ 9 →
    (Nat.choose 10 (r.toNat + 1) + Nat.choose 10 (17 - r.toNat)) ∈ S) ∧ 
    S.card = 2 := by
  sorry

end binomial_sum_distinct_values_l2174_217425


namespace carrie_pays_94_l2174_217451

/-- The amount Carrie pays for clothes given the quantities and prices of items, and that her mom pays half the total cost. -/
def carriePays (shirtQuantity pantQuantity jacketQuantity : ℕ) 
               (shirtPrice pantPrice jacketPrice : ℚ) : ℚ :=
  let totalCost := shirtQuantity * shirtPrice + 
                   pantQuantity * pantPrice + 
                   jacketQuantity * jacketPrice
  totalCost / 2

/-- Theorem stating that Carrie pays $94 for the clothes. -/
theorem carrie_pays_94 : 
  carriePays 4 2 2 8 18 60 = 94 := by
  sorry

end carrie_pays_94_l2174_217451


namespace percentage_watching_two_shows_l2174_217469

def total_residents : ℕ := 600
def watch_island_survival : ℕ := (35 * total_residents) / 100
def watch_lovelost_lawyers : ℕ := (40 * total_residents) / 100
def watch_medical_emergency : ℕ := (50 * total_residents) / 100
def watch_all_three : ℕ := 21

theorem percentage_watching_two_shows :
  let watch_two_shows := watch_island_survival + watch_lovelost_lawyers + watch_medical_emergency - total_residents + watch_all_three
  (watch_two_shows : ℚ) / total_residents * 100 = 285 / 10 := by
  sorry

end percentage_watching_two_shows_l2174_217469


namespace star_commutative_star_not_distributive_no_star_identity_star_not_associative_l2174_217480

-- Define the binary operation ⋆
def star (x y : ℝ) : ℝ := x^2 * y^2 + x + y

-- Commutativity
theorem star_commutative : ∀ x y : ℝ, star x y = star y x := by sorry

-- Non-distributivity
theorem star_not_distributive : ¬(∀ x y z : ℝ, star x (y + z) = star x y + star x z) := by sorry

-- Non-existence of identity element
theorem no_star_identity : ¬(∃ e : ℝ, ∀ x : ℝ, star x e = x ∧ star e x = x) := by sorry

-- Non-associativity
theorem star_not_associative : ¬(∀ x y z : ℝ, star (star x y) z = star x (star y z)) := by sorry

end star_commutative_star_not_distributive_no_star_identity_star_not_associative_l2174_217480


namespace pascal_theorem_l2174_217422

-- Define the conic section
structure ConicSection where
  -- Add necessary fields to define a conic section
  -- This is a placeholder and should be replaced with actual definition

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ -- ax + by + c = 0

-- Define the hexagon
structure Hexagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  F : Point

-- Function to check if a point lies on a conic section
def pointOnConic (p : Point) (c : ConicSection) : Prop :=
  sorry -- Define the condition for a point to lie on the conic section

-- Function to check if two lines intersect at a point
def linesIntersectAt (l1 l2 : Line) (p : Point) : Prop :=
  sorry -- Define the condition for two lines to intersect at a given point

-- Function to check if three points are collinear
def areCollinear (p1 p2 p3 : Point) : Prop :=
  sorry -- Define the condition for three points to be collinear

-- Theorem statement
theorem pascal_theorem (c : ConicSection) (h : Hexagon) 
  (hInscribed : pointOnConic h.A c ∧ pointOnConic h.B c ∧ pointOnConic h.C c ∧ 
                pointOnConic h.D c ∧ pointOnConic h.E c ∧ pointOnConic h.F c)
  (M N P : Point)
  (hM : linesIntersectAt (Line.mk 0 0 0) (Line.mk 0 0 0) M) -- AB and DE
  (hN : linesIntersectAt (Line.mk 0 0 0) (Line.mk 0 0 0) N) -- BC and EF
  (hP : linesIntersectAt (Line.mk 0 0 0) (Line.mk 0 0 0) P) -- CD and FA
  : areCollinear M N P :=
by
  sorry -- Proof goes here

end pascal_theorem_l2174_217422


namespace fourth_number_unit_digit_l2174_217443

def unit_digit (n : ℕ) : ℕ := n % 10

theorem fourth_number_unit_digit 
  (a b c : ℕ) 
  (ha : a = 7858) 
  (hb : b = 1086) 
  (hc : c = 4582) : 
  ∃ d : ℕ, unit_digit (a * b * c * d) = 8 ↔ unit_digit d = 4 := by
  sorry

end fourth_number_unit_digit_l2174_217443


namespace transformed_area_is_63_l2174_217431

/-- The transformation matrix --/
def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 1, -1]

/-- The original region's area --/
def original_area : ℝ := 9

/-- Theorem stating the area of the transformed region --/
theorem transformed_area_is_63 : 
  |A.det| * original_area = 63 := by sorry

end transformed_area_is_63_l2174_217431


namespace geometric_sequence_property_l2174_217495

/-- A sequence a : ℕ → ℝ is geometric if there exists a common ratio r such that
    aₙ₊₁ = r * aₙ for all n. -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) (h : IsGeometric a) :
  a 1 * a 2 * a 3 = -8 → a 2 = -2 := by
  sorry

end geometric_sequence_property_l2174_217495


namespace perimeter_plus_area_equals_9_sqrt_41_l2174_217434

/-- A parallelogram with integer coordinates -/
structure Parallelogram where
  a : ℤ × ℤ
  b : ℤ × ℤ
  c : ℤ × ℤ
  d : ℤ × ℤ

/-- The specific parallelogram from the problem -/
def specificParallelogram : Parallelogram :=
  { a := (0, 0),
    b := (4, 5),
    c := (11, 5),
    d := (7, 0) }

/-- Calculate the perimeter of a parallelogram -/
def perimeter (p : Parallelogram) : ℝ :=
  sorry

/-- Calculate the area of a parallelogram -/
def area (p : Parallelogram) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem perimeter_plus_area_equals_9_sqrt_41 :
  perimeter specificParallelogram + area specificParallelogram = 9 * Real.sqrt 41 :=
sorry

end perimeter_plus_area_equals_9_sqrt_41_l2174_217434


namespace uf_championship_ratio_l2174_217472

/-- The ratio of UF's points in the championship game to their average points per game -/
theorem uf_championship_ratio : 
  ∀ (total_points : ℕ) (num_games : ℕ) (opponent_points : ℕ) (win_margin : ℕ),
    total_points = 720 →
    num_games = 24 →
    opponent_points = 11 →
    win_margin = 2 →
    (opponent_points + win_margin : ℚ) / (total_points / num_games : ℚ) = 13 / 30 := by
  sorry

end uf_championship_ratio_l2174_217472


namespace triangle_area_l2174_217473

theorem triangle_area (a b c : ℝ) (ha : a^2 = 225) (hb : b^2 = 225) (hc : c^2 = 64) :
  (1/2) * a * c = 60 := by
  sorry

end triangle_area_l2174_217473


namespace max_expression_value_l2174_217411

def expression (a b c d : ℕ) : ℕ := d * (c^a - b)

theorem max_expression_value :
  ∃ (a b c d : ℕ),
    a ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    b ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    c ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    d ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    expression a b c d = 126 ∧
    ∀ (x y z w : ℕ),
      x ∈ ({1, 2, 3, 4} : Set ℕ) →
      y ∈ ({1, 2, 3, 4} : Set ℕ) →
      z ∈ ({1, 2, 3, 4} : Set ℕ) →
      w ∈ ({1, 2, 3, 4} : Set ℕ) →
      x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w →
      expression x y z w ≤ 126 :=
by
  sorry


end max_expression_value_l2174_217411


namespace gcd_378_90_l2174_217437

theorem gcd_378_90 : Nat.gcd 378 90 = 18 := by
  sorry

end gcd_378_90_l2174_217437


namespace overlap_area_is_nine_l2174_217410

/-- Regular hexagon with area 36 -/
structure RegularHexagon :=
  (area : ℝ)
  (is_regular : Bool)
  (area_eq : area = 36)

/-- Equilateral triangle formed by connecting every other vertex of the hexagon -/
structure EquilateralTriangle (hex : RegularHexagon) :=
  (vertices : Fin 3 → Fin 6)
  (is_equilateral : Bool)
  (area : ℝ)
  (area_eq : area = hex.area / 2)

/-- The overlapping region of two equilateral triangles in the hexagon -/
def overlap_area (hex : RegularHexagon) (t1 t2 : EquilateralTriangle hex) : ℝ := sorry

/-- Theorem stating that the overlap area is 9 -/
theorem overlap_area_is_nine (hex : RegularHexagon) 
  (t1 t2 : EquilateralTriangle hex) : overlap_area hex t1 t2 = 9 := by sorry

end overlap_area_is_nine_l2174_217410


namespace sine_graph_shift_l2174_217407

theorem sine_graph_shift (x : ℝ) : 
  2 * Real.sin (3 * (x - π/15) + π/5) = 2 * Real.sin (3 * x) := by
  sorry

end sine_graph_shift_l2174_217407


namespace max_perimeter_of_special_triangle_l2174_217487

/-- Represents the sides of a triangle --/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Checks if the given sides form a valid triangle --/
def isValidTriangle (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- Calculates the perimeter of a triangle --/
def perimeter (t : Triangle) : ℕ :=
  t.a + t.b + t.c

/-- Theorem stating the maximum perimeter of the triangle --/
theorem max_perimeter_of_special_triangle :
  ∃ (t : Triangle),
    t.a = 5 ∧
    t.b = 6 ∧
    isValidTriangle t ∧
    (∀ (t' : Triangle),
      t'.a = 5 →
      t'.b = 6 →
      isValidTriangle t' →
      perimeter t' ≤ perimeter t) ∧
    perimeter t = 21 :=
  sorry

end max_perimeter_of_special_triangle_l2174_217487


namespace simplify_fraction_l2174_217426

theorem simplify_fraction : 5 * (18 / 7) * (21 / -54) = -5 := by sorry

end simplify_fraction_l2174_217426


namespace ellipse_eccentricity_theorem_l2174_217423

def ellipse_eccentricity_range (a b c : ℝ) (F₁ F₂ P : ℝ × ℝ) : Prop :=
  let e := c / a
  0 < b ∧ b < a ∧
  c^2 + b^2 = a^2 ∧
  F₁ = (-c, 0) ∧
  F₂ = (c, 0) ∧
  P.1 = a^2 / c ∧
  (∃ m : ℝ, P = (a^2 / c, m) ∧
    let K := ((a^2 - c^2) / (2 * c), m / 2)
    (P.2 - F₁.2) * (K.2 - F₂.2) = -(P.1 - F₁.1) * (K.1 - F₂.1)) →
  Real.sqrt 3 / 3 ≤ e ∧ e < 1

theorem ellipse_eccentricity_theorem (a b c : ℝ) (F₁ F₂ P : ℝ × ℝ) :
  ellipse_eccentricity_range a b c F₁ F₂ P := by sorry

end ellipse_eccentricity_theorem_l2174_217423


namespace decreasing_cubic_implies_m_leq_neg_three_exists_m_leq_neg_three_not_decreasing_l2174_217482

/-- A function f : ℝ → ℝ is decreasing if for all x y, x < y implies f x > f y -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- The cubic function f(x) = mx³ + 3x² - x + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^3 + 3 * x^2 - x + 1

theorem decreasing_cubic_implies_m_leq_neg_three :
  ∀ m : ℝ, DecreasingFunction (f m) → m ≤ -3 :=
sorry

theorem exists_m_leq_neg_three_not_decreasing :
  ∃ m : ℝ, m ≤ -3 ∧ ¬(DecreasingFunction (f m)) :=
sorry

end decreasing_cubic_implies_m_leq_neg_three_exists_m_leq_neg_three_not_decreasing_l2174_217482


namespace triangle_property_l2174_217403

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem states that for a triangle satisfying certain conditions, 
    angle B is 2π/3 and the area is (3√3)/2. -/
theorem triangle_property (t : Triangle) 
  (h1 : t.a * Real.sin t.B + t.b * Real.sin t.A = t.b * Real.sin t.C - t.c * Real.sin t.B)
  (h2 : t.b = Real.sqrt 13)
  (h3 : t.a + t.c = 4) : 
  t.B = 2 * Real.pi / 3 ∧ 
  (1/2) * t.a * t.c * Real.sin t.B = (3 * Real.sqrt 3) / 2 := by
  sorry

end triangle_property_l2174_217403


namespace workshop_attendance_l2174_217463

theorem workshop_attendance : 
  ∀ (total wolf_laureates wolf_and_nobel_laureates nobel_laureates : ℕ),
    wolf_laureates = 31 →
    wolf_and_nobel_laureates = 16 →
    nobel_laureates = 27 →
    ∃ (non_wolf_nobel non_wolf_non_nobel : ℕ),
      non_wolf_nobel = nobel_laureates - wolf_and_nobel_laureates ∧
      non_wolf_nobel = non_wolf_non_nobel + 3 ∧
      total = wolf_laureates + non_wolf_nobel + non_wolf_non_nobel →
      total = 50 :=
by
  sorry

end workshop_attendance_l2174_217463


namespace ninas_pet_eyes_l2174_217456

/-- The total number of eyes among Nina's pet insects -/
theorem ninas_pet_eyes : 
  let spider_count : ℕ := 3
  let ant_count : ℕ := 50
  let eyes_per_spider : ℕ := 8
  let eyes_per_ant : ℕ := 2
  let total_eyes : ℕ := spider_count * eyes_per_spider + ant_count * eyes_per_ant
  total_eyes = 124 := by sorry

end ninas_pet_eyes_l2174_217456


namespace ellipse_m_value_l2174_217430

/-- An ellipse with equation x²/10 + y²/m = 1, foci on y-axis, and major axis length 8 has m = 16 -/
theorem ellipse_m_value (m : ℝ) : 
  (∀ x y : ℝ, x^2 / 10 + y^2 / m = 1) →  -- Ellipse equation
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 = 10 ∧ b^2 = m) →  -- Standard form of ellipse
  (∀ x : ℝ, x^2 / 10 + 0^2 / m ≠ 1) →  -- Foci on y-axis
  (2 * Real.sqrt m = 8) →  -- Major axis length
  m = 16 := by
sorry

end ellipse_m_value_l2174_217430


namespace cloth_cost_unchanged_l2174_217427

/-- Represents the scenario of a cloth purchase with changing length and price --/
structure ClothPurchase where
  originalCost : ℝ  -- Total cost in rupees
  originalLength : ℝ  -- Length in meters
  lengthIncrease : ℝ  -- Increase in length in meters
  priceDecrease : ℝ  -- Decrease in price per meter in rupees

/-- The total cost remains unchanged after increasing length and decreasing price --/
def costUnchanged (cp : ClothPurchase) : Prop :=
  cp.originalCost = (cp.originalLength + cp.lengthIncrease) * 
    ((cp.originalCost / cp.originalLength) - cp.priceDecrease)

/-- Theorem stating that for the given conditions, the cost remains unchanged when length increases by 4 meters --/
theorem cloth_cost_unchanged : 
  ∃ (cp : ClothPurchase), 
    cp.originalCost = 35 ∧ 
    cp.originalLength = 10 ∧ 
    cp.priceDecrease = 1 ∧ 
    cp.lengthIncrease = 4 ∧ 
    costUnchanged cp := by
  sorry

end cloth_cost_unchanged_l2174_217427


namespace chord_length_concentric_circles_l2174_217413

theorem chord_length_concentric_circles (area_ring : ℝ) (r_small : ℝ) (r_large : ℝ) (chord_length : ℝ) : 
  area_ring = 18.75 * Real.pi ∧ 
  r_large = 2 * r_small ∧ 
  area_ring = Real.pi * (r_large^2 - r_small^2) ∧
  chord_length^2 = 4 * (r_large^2 - r_small^2) →
  chord_length = Real.sqrt 75 := by
  sorry

end chord_length_concentric_circles_l2174_217413


namespace complex_magnitude_l2174_217496

/-- Given a complex number z = (3+i)/(1+2i), prove that its magnitude |z| is equal to √2 -/
theorem complex_magnitude (z : ℂ) : z = (3 + I) / (1 + 2*I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_magnitude_l2174_217496


namespace union_of_sets_l2174_217432

def set_A : Set ℝ := {x | x * (x + 1) ≤ 0}
def set_B : Set ℝ := {x | -1 < x ∧ x < 1}

theorem union_of_sets : set_A ∪ set_B = {x | -1 ≤ x ∧ x < 1} := by sorry

end union_of_sets_l2174_217432


namespace greatest_triangle_perimeter_l2174_217467

theorem greatest_triangle_perimeter : 
  ∀ a b c : ℕ,
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (b = 4 * a) →
  (c = 20) →
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  (∀ x y z : ℕ, 
    (x > 0 ∧ y > 0 ∧ z > 0) →
    (y = 4 * x) →
    (z = 20) →
    (x + y > z ∧ y + z > x ∧ z + x > y) →
    (a + b + c ≥ x + y + z)) →
  a + b + c = 50 :=
by sorry

end greatest_triangle_perimeter_l2174_217467


namespace polynomial_value_l2174_217445

theorem polynomial_value (x y : ℝ) (h : 2 * x^2 + 3 * y + 7 = 8) :
  -2 * x^2 - 3 * y + 10 = 9 := by
  sorry

end polynomial_value_l2174_217445


namespace min_PM_dot_PF_l2174_217489

/-- Parabola C: y^2 = 2px (p > 0) -/
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

/-- Circle M with center on positive x-axis and tangent to y-axis -/
def circle_M (center_x : ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center_x)^2 + y^2 = radius^2 ∧ center_x > 0 ∧ center_x = radius

/-- Line m passing through origin with inclination angle π/3 -/
def line_m (x y : ℝ) : Prop := y = x * Real.sqrt 3

/-- Point A on directrix l and point B on circle M, both on line m -/
def points_A_B (A B : ℝ × ℝ) : Prop :=
  line_m A.1 A.2 ∧ line_m B.1 B.2 ∧ A.1^2 + A.2^2 = 4 ∧ B.1^2 + B.2^2 = 4

/-- Theorem: Minimum value of PM⋅PF is 2 -/
theorem min_PM_dot_PF (p : ℝ) (center_x radius : ℝ) (A B : ℝ × ℝ) :
  parabola p 1 2 →
  circle_M center_x radius center_x 0 →
  points_A_B A B →
  (∀ x y : ℝ, parabola p x y → 
    (x^2 - center_x*x + (center_x^2)/4 + y^2) ≥ 2) :=
sorry

end min_PM_dot_PF_l2174_217489


namespace paintings_removed_l2174_217486

theorem paintings_removed (initial : ℕ) (final : ℕ) (h1 : initial = 98) (h2 : final = 95) :
  initial - final = 3 := by
  sorry

end paintings_removed_l2174_217486


namespace expand_quadratic_l2174_217475

theorem expand_quadratic (a : ℝ) : a * (a - 3) = a^2 - 3*a := by
  sorry

end expand_quadratic_l2174_217475


namespace problem_statement_l2174_217459

theorem problem_statement (a b c p q : ℕ) (hp : p > q) 
  (h_sum : a + b + c = 2 * p * q * (p^30 + q^30)) : 
  let k := a^3 + b^3 + c^3
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ k = x * y ∧ 
  (∀ (a' b' c' : ℕ), a' + b' + c' = 2 * p * q * (p^30 + q^30) → 
    a' * b' * c' ≤ a * b * c → 1984 ∣ k) :=
by sorry

end problem_statement_l2174_217459


namespace perfect_square_trinomial_condition_l2174_217400

theorem perfect_square_trinomial_condition (a : ℝ) :
  (∃ b c : ℝ, ∀ x y : ℝ, 4*x^2 - (a-1)*x*y + 9*y^2 = (b*x + c*y)^2) →
  (a = 13 ∨ a = -11) :=
by sorry

end perfect_square_trinomial_condition_l2174_217400


namespace janeles_cats_average_weight_is_correct_l2174_217447

/-- The combined average weight of Janele's cats -/
def janeles_cats_average_weight : ℝ := by sorry

/-- The weights of Janele's first 7 cats -/
def first_seven_cats_weights : List ℝ := [12, 12, 14.7, 9.3, 14.9, 15.6, 8.7]

/-- Lily's weights over 4 days -/
def lily_weights : List ℝ := [14, 15.3, 13.2, 14.7]

/-- The number of Janele's cats -/
def num_cats : ℕ := 8

theorem janeles_cats_average_weight_is_correct :
  janeles_cats_average_weight = 
    (List.sum first_seven_cats_weights + List.sum lily_weights / 4) / num_cats := by sorry

end janeles_cats_average_weight_is_correct_l2174_217447


namespace scientific_notation_1500_l2174_217484

theorem scientific_notation_1500 : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 1500 = a * (10 : ℝ) ^ n :=
by
  -- The proof goes here
  sorry

end scientific_notation_1500_l2174_217484


namespace min_reciprocal_sum_l2174_217498

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (1 / a + 1 / b) ≥ 2 :=
sorry

end min_reciprocal_sum_l2174_217498


namespace sin_thirteen_pi_sixths_l2174_217438

theorem sin_thirteen_pi_sixths : Real.sin (13 * π / 6) = 1 / 2 := by
  sorry

end sin_thirteen_pi_sixths_l2174_217438


namespace annas_pencils_l2174_217465

theorem annas_pencils (anna_pencils : ℕ) (harry_pencils : ℕ) : 
  (harry_pencils = 2 * anna_pencils) → -- Harry has twice Anna's pencils initially
  (harry_pencils - 19 = 81) → -- Harry lost 19 pencils and now has 81 left
  anna_pencils = 50 := by
sorry

end annas_pencils_l2174_217465


namespace loading_dock_problem_l2174_217414

/-- Proves that given the conditions of the loading dock problem, 
    the fraction of boxes loaded by each night crew worker 
    compared to each day crew worker is 5/14 -/
theorem loading_dock_problem 
  (day_crew : ℕ) 
  (night_crew : ℕ) 
  (h1 : night_crew = (4 : ℚ) / 5 * day_crew) 
  (h2 : (5 : ℚ) / 7 = day_crew_boxes / total_boxes) 
  (day_crew_boxes : ℚ) 
  (night_crew_boxes : ℚ) 
  (total_boxes : ℚ) 
  (h3 : total_boxes = day_crew_boxes + night_crew_boxes) 
  (h4 : total_boxes ≠ 0) 
  (h5 : day_crew ≠ 0) 
  (h6 : night_crew ≠ 0) :
  (night_crew_boxes / night_crew) / (day_crew_boxes / day_crew) = (5 : ℚ) / 14 := by
  sorry

end loading_dock_problem_l2174_217414


namespace right_triangle_xy_length_l2174_217499

/-- Given a right triangle XYZ where YZ = 20 and tan Z = 3 * cos Y, 
    the length of XY is (40 * √2) / 3 -/
theorem right_triangle_xy_length (X Y Z : ℝ) : 
  -- Triangle XYZ is right-angled at X
  X + Y + Z = Real.pi ∧ X = Real.pi / 2 →
  -- YZ = 20
  Real.sqrt ((Y - Z)^2 + X^2) = 20 →
  -- tan Z = 3 * cos Y
  Real.tan Z = 3 * Real.cos Y →
  -- XY = (40 * √2) / 3
  Real.sqrt (Y^2 + Z^2) = (40 * Real.sqrt 2) / 3 := by
  sorry

end right_triangle_xy_length_l2174_217499


namespace gcd_sum_lcm_eq_gcd_l2174_217494

theorem gcd_sum_lcm_eq_gcd (a b : ℤ) : Int.gcd (a + b) (Int.lcm a b) = Int.gcd a b := by
  sorry

end gcd_sum_lcm_eq_gcd_l2174_217494


namespace barium_atoms_in_compound_l2174_217433

/-- The number of Barium atoms in the compound -/
def num_barium_atoms : ℕ := 1

/-- The number of Oxygen atoms in the compound -/
def num_oxygen_atoms : ℕ := 2

/-- The number of Hydrogen atoms in the compound -/
def num_hydrogen_atoms : ℕ := 2

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 171

/-- The atomic weight of Barium in g/mol -/
def atomic_weight_barium : ℝ := 137.33

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_oxygen : ℝ := 16.00

/-- The atomic weight of Hydrogen in g/mol -/
def atomic_weight_hydrogen : ℝ := 1.01

theorem barium_atoms_in_compound :
  num_barium_atoms = 1 :=
sorry

end barium_atoms_in_compound_l2174_217433


namespace object_distances_l2174_217454

-- Define the parameters
def speed1 : ℝ := 3
def speed2 : ℝ := 4
def initial_distance : ℝ := 20
def final_distance : ℝ := 10
def time_elapsed : ℝ := 2

-- Define the theorem
theorem object_distances (x y : ℝ) :
  -- Conditions
  (x^2 + y^2 = initial_distance^2) →
  ((x - speed1 * time_elapsed)^2 + (y - speed2 * time_elapsed)^2 = final_distance^2) →
  -- Conclusion
  (x = 12 ∧ y = 16) :=
by sorry

end object_distances_l2174_217454


namespace area_of_region_t_l2174_217436

/-- A rhombus with side length 3 and one right angle -/
structure RightRhombus where
  side_length : ℝ
  angle_q : ℝ
  side_length_eq : side_length = 3
  angle_q_eq : angle_q = 90

/-- The region T inside the rhombus -/
def region_t (r : RightRhombus) : Set (ℝ × ℝ) := sorry

/-- The area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating that the area of region T is 2.25 -/
theorem area_of_region_t (r : RightRhombus) : area (region_t r) = 2.25 := by sorry

end area_of_region_t_l2174_217436


namespace equation_solutions_l2174_217492

/-- The equation we're solving -/
def equation (x : ℝ) : Prop :=
  (15*x - x^2) / (x + 2) * (x + (15 - x) / (x + 2)) = 54

/-- The set of solutions to the equation -/
def solutions : Set ℝ := {0, -1, -3, -3.5}

/-- Theorem stating that the solutions are correct -/
theorem equation_solutions :
  ∀ x : ℝ, x ∈ solutions ↔ equation x :=
by sorry

end equation_solutions_l2174_217492


namespace determinant_difference_l2174_217461

theorem determinant_difference (a b c d : ℝ) :
  Matrix.det !![a, b; c, d] = 15 →
  Matrix.det !![3*a, 3*b; 3*c, 3*d] - Matrix.det !![3*b, 3*a; 3*d, 3*c] = 270 := by
sorry

end determinant_difference_l2174_217461


namespace circle_ratio_invariant_l2174_217448

theorem circle_ratio_invariant (r : ℝ) (h : r > 2) : 
  let new_radius := r - 2
  let new_diameter := 2 * r - 4
  let new_circumference := 2 * Real.pi * new_radius
  new_circumference / new_diameter = Real.pi := by
sorry

end circle_ratio_invariant_l2174_217448


namespace parking_spots_fourth_level_l2174_217420

theorem parking_spots_fourth_level 
  (total_levels : Nat) 
  (first_level_spots : Nat) 
  (second_level_diff : Nat) 
  (third_level_diff : Nat) 
  (total_spots : Nat) :
  total_levels = 4 →
  first_level_spots = 4 →
  second_level_diff = 7 →
  third_level_diff = 6 →
  total_spots = 46 →
  let second_level_spots := first_level_spots + second_level_diff
  let third_level_spots := second_level_spots + third_level_diff
  let fourth_level_spots := total_spots - (first_level_spots + second_level_spots + third_level_spots)
  fourth_level_spots = 14 := by
sorry

end parking_spots_fourth_level_l2174_217420


namespace polynomial_real_root_l2174_217462

/-- The polynomial in question -/
def polynomial (b x : ℝ) : ℝ := x^4 + b*x^3 + x^2 + b*x + 1

/-- The theorem statement -/
theorem polynomial_real_root (b : ℝ) :
  (∃ x : ℝ, polynomial b x = 0) ↔ b ≤ -1.5 := by sorry

end polynomial_real_root_l2174_217462


namespace circle_parameter_range_l2174_217474

-- Define the equation of the circle
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 4*y + 1 + a = 0

-- Define what it means for an equation to represent a circle
def represents_circle (a : ℝ) : Prop :=
  ∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), circle_equation x y a ↔ (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement
theorem circle_parameter_range (a : ℝ) :
  represents_circle a → a < 4 := by
  sorry

end circle_parameter_range_l2174_217474


namespace bales_stored_l2174_217406

/-- Given the initial number of bales and the final number of bales,
    prove that Jason stored 23 bales in the barn. -/
theorem bales_stored (initial_bales final_bales : ℕ) 
  (h1 : initial_bales = 73)
  (h2 : final_bales = 96) :
  final_bales - initial_bales = 23 := by
  sorry

end bales_stored_l2174_217406


namespace no_solution_condition_l2174_217490

theorem no_solution_condition (k : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → x ≠ -1 → (1 / (x + 1) ≠ 3 * k / x)) ↔ (k = 0 ∨ k = 1/3) :=
sorry

end no_solution_condition_l2174_217490


namespace absolute_value_inequality_l2174_217468

theorem absolute_value_inequality (x : ℝ) : 
  abs (x - 1) + abs (x + 2) < 5 ↔ -3 < x ∧ x < 2 := by sorry

end absolute_value_inequality_l2174_217468


namespace problem_solution_l2174_217466

theorem problem_solution (x : ℝ) 
  (h : x + Real.sqrt (x^2 - 4) + 1 / (x - Real.sqrt (x^2 - 4)) = 25) : 
  x^2 + Real.sqrt (x^4 - 4) + 1 / (x^2 - Real.sqrt (x^4 - 4)) = 82.1762 := by
  sorry

end problem_solution_l2174_217466


namespace fraction_zero_implies_x_plus_minus_one_l2174_217446

theorem fraction_zero_implies_x_plus_minus_one (x : ℝ) :
  (x^2 - 1) / x = 0 → x ≠ 0 → (x = 1 ∨ x = -1) :=
by sorry

end fraction_zero_implies_x_plus_minus_one_l2174_217446


namespace triangle_perimeter_is_eight_l2174_217442

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the perimeter of a triangle
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

-- Define the triangle inequality
def is_valid_triangle (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

-- Define the quadratic equation
def is_root_of_equation (x : ℝ) : Prop :=
  x^2 - 4*x + 3 = 0

-- Theorem statement
theorem triangle_perimeter_is_eight :
  ∃ (t : Triangle), t.a = 2 ∧ t.b = 3 ∧ 
  is_root_of_equation t.c ∧ 
  is_valid_triangle t ∧
  perimeter t = 8 :=
sorry

end triangle_perimeter_is_eight_l2174_217442


namespace number_of_ways_to_choose_cards_l2174_217491

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of suits in a standard deck -/
def NumberOfSuits : ℕ := 4

/-- Number of cards per suit in a standard deck -/
def CardsPerSuit : ℕ := 13

/-- Number of cards to be chosen -/
def CardsToChoose : ℕ := 4

/-- Number of cards to be chosen from one suit -/
def CardsFromOneSuit : ℕ := 2

/-- Calculate the number of ways to choose cards according to the problem conditions -/
def calculateWays : ℕ :=
  Nat.choose NumberOfSuits 3 *  -- Choose 3 suits from 4
  3 *  -- Choose which of the 3 suits will have 2 cards
  Nat.choose CardsPerSuit 2 *  -- Choose 2 cards from the chosen suit
  CardsPerSuit * CardsPerSuit  -- Choose 1 card each from the other two suits

/-- Theorem stating that the number of ways to choose cards is 158184 -/
theorem number_of_ways_to_choose_cards :
  calculateWays = 158184 := by sorry

end number_of_ways_to_choose_cards_l2174_217491


namespace consecutive_integers_sum_l2174_217440

theorem consecutive_integers_sum (n : ℤ) : 
  (n - 1) + (n + 1) = 118 → n = 59 := by
sorry

end consecutive_integers_sum_l2174_217440


namespace max_value_expression_l2174_217457

/-- The maximum value of (x + y) / z given the conditions -/
theorem max_value_expression (x y z : ℕ) : 
  x ≥ 10 ∧ x ≤ 99 ∧ 
  y ≥ 10 ∧ y ≤ 99 ∧ 
  z ≥ 10 ∧ z ≤ 99 ∧ 
  (x + y + z) / 3 = 60 → 
  (x + y : ℚ) / z ≤ 17 :=
by sorry

end max_value_expression_l2174_217457


namespace blood_expires_same_day_l2174_217429

/- Define the number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/- Define the expiration time in seconds (8!) -/
def expiration_time : ℕ := Nat.factorial 8

/- Theorem: The blood expires in less than one day -/
theorem blood_expires_same_day : 
  (expiration_time : ℚ) / seconds_per_day < 1 := by
  sorry


end blood_expires_same_day_l2174_217429


namespace specific_group_probability_l2174_217471

-- Define the number of students in the class
def n : ℕ := 32

-- Define the number of students chosen each day
def k : ℕ := 3

-- Define the combination function
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the probability of selecting a specific group
def probability : ℚ := 1 / (combination n k)

-- Theorem statement
theorem specific_group_probability :
  probability = 1 / 4960 := by
  sorry

end specific_group_probability_l2174_217471


namespace six_digit_integers_count_l2174_217409

/-- The number of different positive, six-digit integers that can be formed
    using the digits 2, 2, 2, 5, 5, and 9 -/
def six_digit_integers : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 1)

/-- Theorem stating that the number of different positive, six-digit integers
    that can be formed using the digits 2, 2, 2, 5, 5, and 9 is equal to 60 -/
theorem six_digit_integers_count : six_digit_integers = 60 := by
  sorry

end six_digit_integers_count_l2174_217409
