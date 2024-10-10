import Mathlib

namespace first_podcast_length_l946_94696

/-- Given a 6-hour drive and podcast lengths, prove the first podcast is 0.75 hours long -/
theorem first_podcast_length (total_time : ℝ) (podcast1 : ℝ) (podcast2 : ℝ) (podcast3 : ℝ) (podcast4 : ℝ) (podcast5 : ℝ) :
  total_time = 6 →
  podcast2 = 2 * podcast1 →
  podcast3 = 1.75 →
  podcast4 = 1 →
  podcast5 = 1 →
  podcast1 + podcast2 + podcast3 + podcast4 + podcast5 = total_time →
  podcast1 = 0.75 := by
  sorry

end first_podcast_length_l946_94696


namespace not_all_observed_values_yield_significant_regression_l946_94659

/-- A set of observed values -/
structure ObservedValues where
  values : Set (ℝ × ℝ)

/-- A regression line equation -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Definition of representative significance for a regression line -/
def has_representative_significance (ov : ObservedValues) (rl : RegressionLine) : Prop :=
  sorry

/-- The theorem stating that not all sets of observed values yield a regression line with representative significance -/
theorem not_all_observed_values_yield_significant_regression :
  ¬ ∀ (ov : ObservedValues), ∃ (rl : RegressionLine), has_representative_significance ov rl :=
sorry

end not_all_observed_values_yield_significant_regression_l946_94659


namespace min_y_squared_l946_94657

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  EF : ℝ
  GH : ℝ
  y : ℝ
  is_isosceles : EF > GH
  circle_tangent : True  -- Represents the condition about the tangent circle

/-- The theorem stating the minimum value of y^2 -/
theorem min_y_squared (t : IsoscelesTrapezoid) 
  (h1 : t.EF = 72) 
  (h2 : t.GH = 45) : 
  ∃ (n : ℝ), n^2 = 486 ∧ ∀ (y : ℝ), 
    (∃ (t' : IsoscelesTrapezoid), t'.y = y ∧ t'.EF = t.EF ∧ t'.GH = t.GH) → 
    y^2 ≥ n^2 :=
sorry

end min_y_squared_l946_94657


namespace division_sum_equality_l946_94648

theorem division_sum_equality : 3752 / (39 * 2) + 5030 / (39 * 10) = 61 := by
  sorry

end division_sum_equality_l946_94648


namespace trigonometric_inequality_l946_94670

theorem trigonometric_inequality (x : ℝ) :
  (-1/4 : ℝ) ≤ 5 * (Real.cos x)^2 - 5 * (Real.cos x)^4 + 5 * Real.sin x * Real.cos x + 1 ∧
  5 * (Real.cos x)^2 - 5 * (Real.cos x)^4 + 5 * Real.sin x * Real.cos x + 1 ≤ (19/4 : ℝ) := by
sorry

end trigonometric_inequality_l946_94670


namespace figure_can_form_square_l946_94641

/-- Represents a figure drawn on a grid -/
structure GridFigure where
  cells : Set (ℤ × ℤ)

/-- Represents a cut of the figure -/
structure Cut where
  piece1 : Set (ℤ × ℤ)
  piece2 : Set (ℤ × ℤ)
  piece3 : Set (ℤ × ℤ)

/-- Checks if a set of cells forms a square -/
def isSquare (s : Set (ℤ × ℤ)) : Prop :=
  ∃ (x y w : ℤ), ∀ (i j : ℤ), (i, j) ∈ s ↔ x ≤ i ∧ i < x + w ∧ y ≤ j ∧ j < y + w

/-- Theorem stating that the figure can be cut into three parts and reassembled into a square -/
theorem figure_can_form_square (f : GridFigure) :
  ∃ (c : Cut), c.piece1 ∪ c.piece2 ∪ c.piece3 = f.cells ∧
               isSquare (c.piece1 ∪ c.piece2 ∪ c.piece3) :=
sorry

end figure_can_form_square_l946_94641


namespace password_decryption_probability_l946_94642

theorem password_decryption_probability :
  let p_A : ℚ := 1/5  -- Probability of A decrypting the password
  let p_B : ℚ := 1/4  -- Probability of B decrypting the password
  let p_either : ℚ := 1 - (1 - p_A) * (1 - p_B)  -- Probability of either A or B (or both) decrypting the password
  p_either = 2/5 := by sorry

end password_decryption_probability_l946_94642


namespace combined_degrees_theorem_l946_94685

/-- Represents the budget allocation for Megatech Corporation's research and development --/
structure BudgetAllocation where
  microphotonics : Float
  homeElectronics : Float
  foodAdditives : Float
  geneticallyModifiedMicroorganisms : Float
  industrialLubricants : Float
  nanotechnology : Float

/-- Calculates the degrees in a circle graph for a given percentage --/
def percentageToDegrees (percentage : Float) : Float :=
  percentage * 360 / 100

/-- Calculates the combined degrees for basic astrophysics and nanotechnology --/
def combinedDegrees (budget : BudgetAllocation) : Float :=
  let basicAstrophysics := 100 - (budget.microphotonics + budget.homeElectronics + 
                                  budget.foodAdditives + budget.geneticallyModifiedMicroorganisms + 
                                  budget.industrialLubricants + budget.nanotechnology)
  percentageToDegrees (basicAstrophysics + budget.nanotechnology)

/-- Theorem: The combined degrees for basic astrophysics and nanotechnology is 50.4 --/
theorem combined_degrees_theorem (budget : BudgetAllocation) 
  (h1 : budget.microphotonics = 10)
  (h2 : budget.homeElectronics = 24)
  (h3 : budget.foodAdditives = 15)
  (h4 : budget.geneticallyModifiedMicroorganisms = 29)
  (h5 : budget.industrialLubricants = 8)
  (h6 : budget.nanotechnology = 7) :
  combinedDegrees budget = 50.4 := by
  sorry


end combined_degrees_theorem_l946_94685


namespace obtuse_triangle_side_range_l946_94666

/-- A triangle with side lengths a, b, and c is obtuse if and only if a² + b² < c² for some permutation of its sides. --/
def IsObtuseTriangle (a b c : ℝ) : Prop :=
  (a^2 + b^2 < c^2) ∨ (a^2 + c^2 < b^2) ∨ (b^2 + c^2 < a^2)

/-- The range of possible values for the third side of an obtuse triangle with two sides of length 3 and 4. --/
theorem obtuse_triangle_side_range :
  ∀ x : ℝ, IsObtuseTriangle 3 4 x ↔ (5 < x ∧ x < 7) ∨ (1 < x ∧ x < Real.sqrt 7) :=
by sorry

end obtuse_triangle_side_range_l946_94666


namespace money_distribution_l946_94695

theorem money_distribution (total : ℝ) (a b c d : ℝ) :
  a + b + c + d = total →
  a = (5 / 14) * total →
  b = (2 / 14) * total →
  c = (4 / 14) * total →
  d = (3 / 14) * total →
  c = d + 500 →
  d = 1500 := by
sorry

end money_distribution_l946_94695


namespace first_number_in_set_l946_94608

theorem first_number_in_set (x : ℝ) : 
  let set1 := [10, 70, 28]
  let set2 := [x, 40, 60]
  (set2.sum / set2.length : ℝ) = (set1.sum / set1.length : ℝ) + 4 →
  x = 20 := by
sorry

end first_number_in_set_l946_94608


namespace smallest_dual_base_representation_l946_94649

theorem smallest_dual_base_representation : ∃ (a b : ℕ), 
  a > 3 ∧ b > 3 ∧ 
  13 = a + 3 ∧ 
  13 = 3 * b + 1 ∧
  (∀ (x y : ℕ), x > 3 → y > 3 → x + 3 = 3 * y + 1 → x + 3 ≥ 13) :=
by sorry

end smallest_dual_base_representation_l946_94649


namespace nick_babysitting_charge_l946_94637

/-- Nick's babysitting charge calculation -/
theorem nick_babysitting_charge (y : ℝ) : 
  let travel_cost : ℝ := 7
  let hourly_rate : ℝ := 10
  let total_charge := hourly_rate * y + travel_cost
  total_charge = 10 * y + 7 := by sorry

end nick_babysitting_charge_l946_94637


namespace hancho_height_calculation_l946_94602

/-- Hancho's height in centimeters, given Hansol's height and the ratio between their heights -/
def hanchos_height (hansols_height : ℝ) (height_ratio : ℝ) : ℝ :=
  hansols_height * height_ratio

/-- Theorem stating that Hancho's height is 142.57 cm -/
theorem hancho_height_calculation :
  let hansols_height : ℝ := 134.5
  let height_ratio : ℝ := 1.06
  hanchos_height hansols_height height_ratio = 142.57 := by sorry

end hancho_height_calculation_l946_94602


namespace picnic_watermelon_slices_l946_94673

/-- The number of watermelon slices at a family picnic -/
def total_watermelon_slices : ℕ :=
  let danny_watermelons : ℕ := 3
  let danny_slices_per_watermelon : ℕ := 10
  let sister_watermelons : ℕ := 1
  let sister_slices_per_watermelon : ℕ := 15
  (danny_watermelons * danny_slices_per_watermelon) + (sister_watermelons * sister_slices_per_watermelon)

theorem picnic_watermelon_slices : total_watermelon_slices = 45 := by
  sorry

end picnic_watermelon_slices_l946_94673


namespace unique_m_value_l946_94697

theorem unique_m_value : ∃! m : ℝ, ∀ y : ℝ, 
  (y - 2 = 1) → (m * y - 2 = 4) := by
  sorry

end unique_m_value_l946_94697


namespace xyz_product_l946_94609

theorem xyz_product (x y z : ℂ) 
  (eq1 : x * y + 3 * y = -9)
  (eq2 : y * z + 3 * z = -9)
  (eq3 : z * x + 3 * x = -9) : 
  x * y * z = -27 := by
sorry

end xyz_product_l946_94609


namespace correct_statements_count_l946_94652

/-- Represents a programming statement --/
inductive Statement
  | Output (cmd : String) (vars : List String)
  | Input (var : String) (value : String)
  | Assignment (lhs : String) (rhs : String)

/-- Checks if a statement is correct --/
def is_correct (s : Statement) : Bool :=
  match s with
  | Statement.Output cmd vars => cmd = "PRINT"
  | Statement.Input var value => true  -- Simplified for this problem
  | Statement.Assignment lhs rhs => true  -- Simplified for this problem

/-- The list of statements to evaluate --/
def statements : List Statement :=
  [ Statement.Output "INPUT" ["a", "b", "c"]
  , Statement.Input "x" "3"
  , Statement.Assignment "3" "A"
  , Statement.Assignment "A" "B=C"
  ]

/-- Counts the number of correct statements --/
def count_correct (stmts : List Statement) : Nat :=
  stmts.filter is_correct |>.length

theorem correct_statements_count :
  count_correct statements = 0 := by
  sorry

end correct_statements_count_l946_94652


namespace minimum_apples_l946_94631

theorem minimum_apples (n : ℕ) (total_apples : ℕ) : 
  (∃ k : ℕ, total_apples = 25 * k + 24) →   -- Condition 1 and 2
  total_apples > 300 →                      -- Condition 3
  total_apples ≥ 324 :=                     -- Minimum number of apples
by
  sorry

#check minimum_apples

end minimum_apples_l946_94631


namespace total_games_in_league_l946_94619

theorem total_games_in_league (n : ℕ) (h : n = 35) : 
  (n * (n - 1)) / 2 = 595 := by
  sorry

end total_games_in_league_l946_94619


namespace horner_method_v3_l946_94645

def f (x : ℝ) : ℝ := 2*x^6 + 5*x^5 + 6*x^4 + 23*x^3 - 8*x^2 + 10*x - 3

def horner_v3 (a₆ a₅ a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  ((a₆ * x + a₅) * x + a₄) * x + a₃

theorem horner_method_v3 :
  horner_v3 2 5 6 23 (-8) 10 (-3) 2 = 71 :=
by sorry

end horner_method_v3_l946_94645


namespace tangent_parallel_to_x_axis_l946_94686

noncomputable def f (x : ℝ) : ℝ := x - Real.exp x

theorem tangent_parallel_to_x_axis :
  ∃ (p : ℝ × ℝ), 
    (∀ x : ℝ, (p.2 = f p.1) ∧ 
    (HasDerivAt f 0 p.1)) →
    p = (0, -1) := by
  sorry

end tangent_parallel_to_x_axis_l946_94686


namespace patio_layout_change_l946_94606

theorem patio_layout_change (total_tiles : ℕ) (original_rows : ℕ) (added_rows : ℕ) :
  total_tiles = 96 →
  original_rows = 8 →
  added_rows = 4 →
  (total_tiles / original_rows) - (total_tiles / (original_rows + added_rows)) = 4 :=
by sorry

end patio_layout_change_l946_94606


namespace yujeong_drank_most_l946_94625

/-- Represents the amount of water drunk by each person in liters -/
structure WaterConsumption where
  yujeong : ℚ
  eunji : ℚ
  yuna : ℚ

/-- Determines who drank the most water -/
def drankMost (consumption : WaterConsumption) : String :=
  if consumption.yujeong > consumption.eunji ∧ consumption.yujeong > consumption.yuna then
    "Yujeong"
  else if consumption.eunji > consumption.yujeong ∧ consumption.eunji > consumption.yuna then
    "Eunji"
  else
    "Yuna"

theorem yujeong_drank_most (consumption : WaterConsumption) 
  (h1 : consumption.yujeong = 7/10)
  (h2 : consumption.eunji = 1/2)
  (h3 : consumption.yuna = 6/10) :
  drankMost consumption = "Yujeong" :=
by
  sorry

#eval drankMost ⟨7/10, 1/2, 6/10⟩

end yujeong_drank_most_l946_94625


namespace product_equals_32_over_9_l946_94615

/-- The repeating decimal 0.4444... --/
def repeating_four : ℚ := 4/9

/-- The product of the repeating decimal 0.4444... and 8 --/
def product : ℚ := repeating_four * 8

theorem product_equals_32_over_9 : product = 32/9 := by
  sorry

end product_equals_32_over_9_l946_94615


namespace parabola_intersection_l946_94644

/-- First parabola equation -/
def f (x : ℝ) : ℝ := 3 * x^2 - 9 * x - 5

/-- Second parabola equation -/
def g (x : ℝ) : ℝ := x^2 + 2 * x + 1

/-- Theorem stating that (-0.5, 0.25) and (6, 49) are the only intersection points -/
theorem parabola_intersection :
  (∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x = -0.5 ∧ y = 0.25) ∨ (x = 6 ∧ y = 49)) := by
  sorry

end parabola_intersection_l946_94644


namespace rectangular_prism_max_volume_l946_94676

/-- Given a rectangular prism with space diagonal 10 and orthogonal projection 8,
    its maximum volume is 192 -/
theorem rectangular_prism_max_volume :
  ∀ (a b h : ℝ),
  (a > 0) → (b > 0) → (h > 0) →
  (a^2 + b^2 + h^2 = 10^2) →
  (a^2 + b^2 = 8^2) →
  ∀ (V : ℝ), V = a * b * h →
  V ≤ 192 :=
by sorry

end rectangular_prism_max_volume_l946_94676


namespace trophy_count_proof_l946_94664

/-- The number of trophies Michael has right now -/
def michael_trophies : ℕ := 30

/-- The number of trophies Jack will have in three years -/
def jack_future_trophies : ℕ := 10 * michael_trophies

/-- The number of trophies Michael will have in three years -/
def michael_future_trophies : ℕ := michael_trophies + 100

theorem trophy_count_proof :
  michael_trophies = 30 ∧
  jack_future_trophies = 10 * michael_trophies ∧
  michael_future_trophies = michael_trophies + 100 ∧
  jack_future_trophies + michael_future_trophies = 430 :=
by sorry

end trophy_count_proof_l946_94664


namespace park_pathway_width_l946_94653

/-- Represents a rectangular park with pathways -/
structure Park where
  length : ℝ
  width : ℝ
  lawn_area : ℝ

/-- Calculates the total width of all pathways in the park -/
def total_pathway_width (p : Park) : ℝ :=
  -- Define the function here, but don't implement it
  sorry

/-- Theorem stating the total pathway width for the given park specifications -/
theorem park_pathway_width :
  let p : Park := { length := 60, width := 40, lawn_area := 2109 }
  total_pathway_width p = 2.91 := by
  sorry

end park_pathway_width_l946_94653


namespace min_distance_parabola_to_line_l946_94617

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 2px -/
structure Parabola where
  p : ℝ

/-- Theorem: Minimum distance from a point on the parabola to the line y = x + 3 -/
theorem min_distance_parabola_to_line (para : Parabola) (P : Point) :
  para.p = 4 →  -- Derived from directrix x = -2
  P.y^2 = 2 * para.p * P.x →  -- Point P is on the parabola
  ∃ (d : ℝ), d = |P.x - P.y + 3| / Real.sqrt 2 ∧  -- Distance formula
  d ≥ Real.sqrt 2 / 2 ∧  -- Minimum distance
  (∃ (Q : Point), Q.y^2 = 2 * para.p * Q.x ∧  -- Another point on parabola
    |Q.x - Q.y + 3| / Real.sqrt 2 = Real.sqrt 2 / 2) :=  -- Achieving minimum distance
by sorry


end min_distance_parabola_to_line_l946_94617


namespace real_part_of_complex_product_l946_94650

def complex_mul (a b c d : ℝ) : ℂ := Complex.mk (a*c - b*d) (a*d + b*c)

theorem real_part_of_complex_product : 
  (complex_mul 1 1 1 (-2)).re = 3 := by sorry

end real_part_of_complex_product_l946_94650


namespace apple_sharing_ways_l946_94672

/-- The number of ways to distribute n indistinguishable objects into k distinguishable boxes --/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute apples among people with a minimum requirement --/
def apple_distribution (total min_per_person people : ℕ) : ℕ :=
  stars_and_bars (total - min_per_person * people) people

theorem apple_sharing_ways :
  apple_distribution 24 2 3 = 190 := by
  sorry

end apple_sharing_ways_l946_94672


namespace medal_award_ways_l946_94604

def total_sprinters : ℕ := 8
def american_sprinters : ℕ := 3
def medals : ℕ := 3

def ways_to_award_medals (total : ℕ) (americans : ℕ) (medals : ℕ) : ℕ :=
  -- Number of ways to award medals with at most one American getting a medal
  sorry

theorem medal_award_ways :
  ways_to_award_medals total_sprinters american_sprinters medals = 240 :=
sorry

end medal_award_ways_l946_94604


namespace increase_by_percentage_increase_80_by_150_percent_l946_94605

theorem increase_by_percentage (x : ℝ) (p : ℝ) :
  x * (1 + p / 100) = x + x * (p / 100) :=
by sorry

theorem increase_80_by_150_percent :
  80 * (1 + 150 / 100) = 200 :=
by sorry

end increase_by_percentage_increase_80_by_150_percent_l946_94605


namespace f_composition_equals_226_l946_94656

def f (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 1

theorem f_composition_equals_226 : f (f (f 1)) = 226 := by
  sorry

end f_composition_equals_226_l946_94656


namespace train_journey_time_l946_94654

theorem train_journey_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) (h2 : usual_time > 0)
  (h3 : (6 / 7 * usual_speed) * (usual_time + 20) = usual_speed * usual_time) :
  usual_time = 140 := by
sorry

end train_journey_time_l946_94654


namespace abc_inequality_l946_94689

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * c ≥ (b + c - a) * (a + c - b) * (a + b - c) := by
  sorry

end abc_inequality_l946_94689


namespace billion_to_scientific_notation_l946_94626

-- Define the number in billions
def number_in_billions : ℝ := 8.36

-- Define the scientific notation
def scientific_notation : ℝ := 8.36 * (10 ^ 9)

-- Theorem statement
theorem billion_to_scientific_notation :
  (number_in_billions * 10^9) = scientific_notation := by
  sorry

end billion_to_scientific_notation_l946_94626


namespace equation_solution_l946_94630

theorem equation_solution : ∃ x : ℝ, 9 - 3 / (1/3) + x = 3 :=
by
  -- The proof goes here
  sorry

end equation_solution_l946_94630


namespace inequality_equivalence_l946_94635

theorem inequality_equivalence (a : ℝ) :
  (∀ x : ℝ, |3*x + 2*a| + |2 - 3*x| - |a + 1| > 2) ↔ (a < -1/3 ∨ a > 5) := by
sorry

end inequality_equivalence_l946_94635


namespace triangle_area_l946_94643

/-- The area of a triangle with side lengths 15, 36, and 39 is 270 -/
theorem triangle_area (a b c : ℝ) (ha : a = 15) (hb : b = 36) (hc : c = 39) :
  (1 / 2 : ℝ) * a * b = 270 := by
  sorry

end triangle_area_l946_94643


namespace no_positive_integer_solution_l946_94669

theorem no_positive_integer_solution (f : ℕ+ → ℕ+) (a b : ℕ+) : 
  (∀ x, f x = x^2 + x) → 4 * (f a) ≠ f b := by
  sorry

end no_positive_integer_solution_l946_94669


namespace cube_coloring_count_l946_94678

/-- The number of distinct colorings of a cube with 6 faces using m colors -/
def g (m : ℕ) : ℚ :=
  (1 / 24) * (m^6 + 3*m^4 + 12*m^3 + 8*m^2)

/-- Theorem: The number of distinct colorings of a cube with 6 faces,
    using m colors, where each face is painted one color,
    is equal to (1/24)(m^6 + 3m^4 + 12m^3 + 8m^2) -/
theorem cube_coloring_count (m : ℕ) :
  g m = (1 / 24) * (m^6 + 3*m^4 + 12*m^3 + 8*m^2) :=
by sorry

end cube_coloring_count_l946_94678


namespace troy_computer_savings_l946_94611

/-- The amount Troy needs to save to buy a new computer -/
theorem troy_computer_savings (new_computer_cost initial_savings old_computer_value : ℕ) 
  (h1 : new_computer_cost = 1800)
  (h2 : initial_savings = 350)
  (h3 : old_computer_value = 100) :
  new_computer_cost - (initial_savings + old_computer_value) = 1350 := by
  sorry

end troy_computer_savings_l946_94611


namespace painting_progress_l946_94665

/-- Represents the fraction of a wall painted in a given time -/
def fraction_painted (total_time minutes : ℕ) : ℚ :=
  minutes / total_time

theorem painting_progress (heidi_time karl_time minutes : ℕ) 
  (h1 : heidi_time = 60)
  (h2 : karl_time = heidi_time / 2)
  (h3 : minutes = 20) :
  (fraction_painted heidi_time minutes = 1/3) ∧ 
  (fraction_painted karl_time minutes = 2/3) := by
  sorry

#check painting_progress

end painting_progress_l946_94665


namespace min_area_rectangle_l946_94633

theorem min_area_rectangle (l w : ℕ) : 
  (2 * l + 2 * w = 60) → 
  (l * w ≥ 29) :=
sorry

end min_area_rectangle_l946_94633


namespace ellipse_equation_hyperbola_equation_l946_94699

-- Problem 1
theorem ellipse_equation (x y : ℝ) :
  let equation := x^2 / 13 + y^2 / (13/9) = 1
  let center_at_origin := ∀ (t : ℝ), t^2 / 13 + 0^2 / (13/9) ≠ 1 ∧ 0^2 / 13 + t^2 / (13/9) ≠ 1
  let foci_on_x_axis := ∃ (c : ℝ), c^2 = 13 - 13/9 ∧ (c^2 / 13 + 0^2 / (13/9) = 1 ∨ (-c)^2 / 13 + 0^2 / (13/9) = 1)
  let major_axis_triple_minor := 13 = 3 * (13/9)
  let passes_through_p := 3^2 / 13 + 2^2 / (13/9) = 1
  center_at_origin ∧ foci_on_x_axis ∧ major_axis_triple_minor ∧ passes_through_p → equation :=
by sorry

-- Problem 2
theorem hyperbola_equation (x y : ℝ) :
  let equation := x^2 / 10 - y^2 / 6 = 1
  let common_asymptote := ∃ (k : ℝ), k^2 = 10/6 ∧ k^2 = 5/3
  let focal_length_8 := ∃ (c : ℝ), c^2 = 10 + 6 ∧ 2*c = 8
  common_asymptote ∧ focal_length_8 → equation :=
by sorry

end ellipse_equation_hyperbola_equation_l946_94699


namespace initial_phone_price_l946_94691

/-- The initial price of a phone given negotiation conditions. -/
theorem initial_phone_price (negotiated_price : ℝ) (negotiation_percentage : ℝ) 
  (h1 : negotiated_price = 480)
  (h2 : negotiation_percentage = 0.20)
  (h3 : negotiated_price = negotiation_percentage * initial_price) : 
  initial_price = 2400 :=
by
  sorry

end initial_phone_price_l946_94691


namespace expression_simplification_l946_94661

theorem expression_simplification (a c x z : ℝ) :
  (c * x * (a^3 * x^3 + 3 * a^3 * z^3 + c^3 * z^3) + a * z * (a^3 * x^3 + 3 * c^3 * x^3 + c^3 * z^3)) / (c * x + a * z) = 
  a^3 * x^3 + 3 * a^3 * z^3 + c^3 * z^3 :=
by sorry

end expression_simplification_l946_94661


namespace calculator_trick_l946_94600

theorem calculator_trick (a b c : ℕ) (h1 : 100 ≤ a * 100 + b * 10 + c) (h2 : a * 100 + b * 10 + c < 1000) :
  let abc := a * 100 + b * 10 + c
  let abcabc := abc * 1000 + abc
  (((abcabc / 7) / 11) / 13) = abc :=
sorry

end calculator_trick_l946_94600


namespace thirty_percent_of_two_hundred_l946_94662

theorem thirty_percent_of_two_hundred : (30 / 100) * 200 = 60 := by
  sorry

end thirty_percent_of_two_hundred_l946_94662


namespace exists_synchronous_exp_sin_synchronous_log_square_implies_a_gt_2e_l946_94677

/-- Definition of synchronous functions -/
def Synchronous (f g : ℝ → ℝ) (m n : ℝ) : Prop :=
  f m = g m ∧ f n = g n

/-- Statement for option B -/
theorem exists_synchronous_exp_sin :
  ∃ n : ℝ, 1/2 < n ∧ n < 1 ∧
  Synchronous (fun x ↦ Real.exp x - 1) (fun x ↦ Real.sin (π * x)) 0 n :=
sorry

/-- Statement for option C -/
theorem synchronous_log_square_implies_a_gt_2e (a : ℝ) :
  (∃ m n : ℝ, Synchronous (fun x ↦ a * Real.log x) (fun x ↦ x^2) m n) →
  a > 2 * Real.exp 1 :=
sorry

end exists_synchronous_exp_sin_synchronous_log_square_implies_a_gt_2e_l946_94677


namespace number_with_given_division_properties_l946_94634

theorem number_with_given_division_properties : ∃ n : ℕ, n / 9 = 80 ∧ n % 9 = 4 ∧ n = 724 := by
  sorry

end number_with_given_division_properties_l946_94634


namespace unique_solution_l946_94698

/-- The system of equations has a unique solution at (-2, -4) -/
theorem unique_solution : ∃! (x y : ℝ), 
  (x + 3*y + 14 ≤ 0) ∧ 
  (x^4 + 2*x^2*y^2 + y^4 + 64 - 20*x^2 - 20*y^2 = 8*x*y) ∧
  (x = -2) ∧ (y = -4) := by
  sorry

end unique_solution_l946_94698


namespace symmetry_sum_l946_94624

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = -B.2

theorem symmetry_sum (a b : ℝ) : 
  symmetric_wrt_origin (a, 2) (4, b) → a + b = -6 := by
  sorry

end symmetry_sum_l946_94624


namespace chord_length_implies_a_value_l946_94628

theorem chord_length_implies_a_value (a : ℝ) :
  (∃ (x y : ℝ), (a * x + y + 1 = 0) ∧ (x^2 + y^2 - 2*a*x + a = 0)) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (a * x₁ + y₁ + 1 = 0) ∧ (x₁^2 + y₁^2 - 2*a*x₁ + a = 0) ∧
    (a * x₂ + y₂ + 1 = 0) ∧ (x₂^2 + y₂^2 - 2*a*x₂ + a = 0) ∧
    ((x₁ - x₂)^2 + (y₁ - y₂)^2 = 4)) →
  a = -2 :=
sorry


end chord_length_implies_a_value_l946_94628


namespace greatest_b_value_l946_94651

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + 7*x - 10 ≥ 0 → x ≤ 5) ∧ 
  (-5^2 + 7*5 - 10 ≥ 0) := by
  sorry

end greatest_b_value_l946_94651


namespace roots_equation_q_value_l946_94658

theorem roots_equation_q_value (a b m p q : ℝ) : 
  (a^2 - m*a + 3 = 0) →
  (b^2 - m*b + 3 = 0) →
  ((a + 1/b)^2 - p*(a + 1/b) + q = 0) →
  ((b + 1/a)^2 - p*(b + 1/a) + q = 0) →
  q = 16/3 := by
sorry

end roots_equation_q_value_l946_94658


namespace certain_number_proof_l946_94614

theorem certain_number_proof (x : ℝ) : 
  (x / 10) - (x / 2000) = 796 → x = 8000 := by
  sorry

end certain_number_proof_l946_94614


namespace unique_balance_l946_94601

def weights : List ℕ := [1, 2, 4, 8, 16, 32]
def candy_weight : ℕ := 25

def is_valid_partition (partition : List ℕ × List ℕ) : Prop :=
  partition.1.length = 3 ∧ 
  partition.2.length = 3 ∧
  (partition.1 ++ partition.2).toFinset = weights.toFinset

def is_balanced (partition : List ℕ × List ℕ) : Prop :=
  (partition.1.sum + candy_weight = partition.2.sum) ∧
  is_valid_partition partition

theorem unique_balance :
  ∃! partition : List ℕ × List ℕ, 
    is_balanced partition ∧ 
    partition.2.toFinset = {4, 8, 32} := by sorry

end unique_balance_l946_94601


namespace hollow_block_length_l946_94629

/-- Represents a hollow rectangular block made of small cubes -/
structure HollowBlock where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Calculates the number of small cubes used in a hollow rectangular block -/
def cubesUsed (block : HollowBlock) : ℕ :=
  2 * (block.length * block.width + block.width * block.depth + block.length * block.depth) -
  4 * (block.length + block.width + block.depth) + 8 -
  ((block.length - 2) * (block.width - 2) * (block.depth - 2))

/-- Theorem stating that a hollow block with given dimensions uses 114 cubes and has a length of 10 -/
theorem hollow_block_length :
  ∃ (block : HollowBlock), block.width = 9 ∧ block.depth = 5 ∧ cubesUsed block = 114 ∧ block.length = 10 :=
by sorry

end hollow_block_length_l946_94629


namespace x_minus_y_value_l946_94667

theorem x_minus_y_value (x y : ℤ) (h1 : x + y = 4) (h2 : x = 20) : x - y = 36 := by
  sorry

end x_minus_y_value_l946_94667


namespace gcd_of_specific_numbers_l946_94610

theorem gcd_of_specific_numbers (p : Nat) (h : Prime p) :
  Nat.gcd (p^10 + 1) (p^10 + p^3 + 1) = 1 :=
by sorry

end gcd_of_specific_numbers_l946_94610


namespace elevator_problem_l946_94616

theorem elevator_problem (initial_avg : ℝ) (new_avg : ℝ) (new_person_weight : ℝ) :
  initial_avg = 152 →
  new_avg = 151 →
  new_person_weight = 145 →
  ∃ n : ℕ, n > 0 ∧ 
    n * initial_avg + new_person_weight = (n + 1) * new_avg ∧
    n = 6 := by
  sorry

end elevator_problem_l946_94616


namespace expand_product_l946_94663

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 := by
  sorry

end expand_product_l946_94663


namespace sequence_properties_l946_94636

/-- The sum of the first n terms of sequence a_n -/
def S (n : ℕ) : ℝ := sorry

/-- The nth term of sequence a_n -/
def a (n : ℕ) : ℝ := sorry

/-- The nth term of arithmetic sequence b_n -/
def b (n : ℕ) : ℝ := sorry

/-- The sum of the first n terms of sequence b_n -/
def T (n : ℕ) : ℝ := sorry

theorem sequence_properties :
  (∀ n, 2 * S n = 3 * a n - 3) ∧
  (b 1 = a 1) ∧
  (b 7 = b 1 * b 2) ∧
  (∀ n m, b (n + m) - b n = m * (b 2 - b 1)) →
  (∀ n, a n = 3^n) ∧
  (∀ n, T n = n^2 + 2*n) := by
sorry

end sequence_properties_l946_94636


namespace store_paid_twenty_six_l946_94620

/-- The price the store paid for a pair of pants, given the selling price and the difference between the selling price and the store's cost. -/
def store_paid_price (selling_price : ℕ) (price_difference : ℕ) : ℕ :=
  selling_price - price_difference

/-- Theorem stating that if the selling price is $34 and the store paid $8 less, then the store paid $26. -/
theorem store_paid_twenty_six :
  store_paid_price 34 8 = 26 := by
  sorry

end store_paid_twenty_six_l946_94620


namespace min_distance_line_parabola_l946_94612

/-- The minimum distance between a point on the line y = (5/12)x - 11 and a point on the parabola y = x² is 6311/624 -/
theorem min_distance_line_parabola :
  let line := λ x : ℝ => (5/12) * x - 11
  let parabola := λ x : ℝ => x^2
  ∃ (d : ℝ), d = 6311/624 ∧
    ∀ (x₁ x₂ : ℝ),
      d ≤ Real.sqrt ((x₂ - x₁)^2 + (parabola x₂ - line x₁)^2) :=
by sorry

end min_distance_line_parabola_l946_94612


namespace exponent_calculation_correct_and_uses_operations_l946_94684

-- Define the exponent operations
inductive ExponentOperation
  | multiplication
  | division
  | exponentiation
  | productExponentiation

-- Define a function to represent the calculation
def exponentCalculation (a : ℝ) : ℝ := (a^3 * a^2)^2

-- Define a function to represent the result of the calculation
def exponentResult (a : ℝ) : ℝ := a^10

-- Define a function to check if an operation is used in the calculation
def isOperationUsed (op : ExponentOperation) : Prop :=
  match op with
  | ExponentOperation.multiplication => True
  | ExponentOperation.exponentiation => True
  | ExponentOperation.productExponentiation => True
  | _ => False

-- Theorem stating that the calculation is correct and uses the specified operations
theorem exponent_calculation_correct_and_uses_operations (a : ℝ) :
  exponentCalculation a = exponentResult a ∧
  isOperationUsed ExponentOperation.multiplication ∧
  isOperationUsed ExponentOperation.exponentiation ∧
  isOperationUsed ExponentOperation.productExponentiation :=
by sorry

end exponent_calculation_correct_and_uses_operations_l946_94684


namespace altered_detergent_theorem_l946_94694

/-- Represents the ratio of components in a cleaning solution -/
structure CleaningSolution where
  bleach : ℚ
  detergent : ℚ
  water : ℚ

/-- Calculates the amount of detergent in the altered solution -/
def altered_detergent_amount (original : CleaningSolution) (water_amount : ℚ) : ℚ :=
  let original_detergent_water_ratio := original.detergent / original.water
  let new_detergent_water_ratio := original_detergent_water_ratio / 2
  water_amount * new_detergent_water_ratio

/-- Theorem stating the amount of detergent in the altered solution -/
theorem altered_detergent_theorem (original : CleaningSolution) 
    (h1 : original.bleach = 2)
    (h2 : original.detergent = 25)
    (h3 : original.water = 100)
    (h4 : altered_detergent_amount original 300 = 37.5) : 
  altered_detergent_amount original 300 = 37.5 := by
  sorry

#check altered_detergent_theorem

end altered_detergent_theorem_l946_94694


namespace reciprocal_of_negative_three_l946_94681

theorem reciprocal_of_negative_three :
  (1 : ℚ) / (-3 : ℚ) = -1/3 := by sorry

end reciprocal_of_negative_three_l946_94681


namespace future_value_proof_l946_94639

/-- Calculates the future value of an investment with compound interest. -/
def future_value (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proves that given the specified conditions, the future value is $3600. -/
theorem future_value_proof :
  let principal : ℝ := 2500
  let rate : ℝ := 0.20
  let time : ℕ := 2
  future_value principal rate time = 3600 := by
sorry

#eval future_value 2500 0.20 2

end future_value_proof_l946_94639


namespace exactly_two_referees_match_l946_94646

-- Define the number of referees/seats
def n : ℕ := 5

-- Define the number of referees that should match their seat number
def k : ℕ := 2

-- Define the function to calculate the number of permutations
-- where exactly k out of n elements are in their original positions
def permutations_with_k_fixed (n k : ℕ) : ℕ :=
  (n.choose k) * 2

-- State the theorem
theorem exactly_two_referees_match : permutations_with_k_fixed n k = 20 := by
  sorry

end exactly_two_referees_match_l946_94646


namespace arithmetic_square_root_of_sqrt_81_l946_94623

theorem arithmetic_square_root_of_sqrt_81 : Real.sqrt (Real.sqrt 81) = 3 := by
  sorry

end arithmetic_square_root_of_sqrt_81_l946_94623


namespace roots_transformation_l946_94613

theorem roots_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 4*r₁^2 + r₁ + 6 = 0) ∧ 
  (r₂^3 - 4*r₂^2 + r₂ + 6 = 0) ∧ 
  (r₃^3 - 4*r₃^2 + r₃ + 6 = 0) →
  ((3*r₁)^3 - 12*(3*r₁)^2 + 9*(3*r₁) + 162 = 0) ∧
  ((3*r₂)^3 - 12*(3*r₂)^2 + 9*(3*r₂) + 162 = 0) ∧
  ((3*r₃)^3 - 12*(3*r₃)^2 + 9*(3*r₃) + 162 = 0) := by
  sorry

end roots_transformation_l946_94613


namespace expected_heads_is_94_l946_94680

/-- The probability of a coin landing on heads after at most four flips -/
def prob_heads : ℚ := 15 / 16

/-- The number of coins -/
def num_coins : ℕ := 100

/-- The expected number of coins landing on heads -/
def expected_heads : ℚ := num_coins * prob_heads

theorem expected_heads_is_94 :
  ⌊expected_heads⌋ = 94 := by sorry

end expected_heads_is_94_l946_94680


namespace union_of_A_and_B_l946_94640

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x - 4 ≤ 0}
def B : Set ℝ := {x | 2*x - 3 > 0}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = Set.Ici (-1) := by sorry

end union_of_A_and_B_l946_94640


namespace science_book_page_count_l946_94693

def history_book_pages : ℕ := 300

def novel_pages (history : ℕ) : ℕ := history / 2

def science_book_pages (novel : ℕ) : ℕ := 4 * novel

theorem science_book_page_count : 
  science_book_pages (novel_pages history_book_pages) = 600 := by
  sorry

end science_book_page_count_l946_94693


namespace flower_pots_total_cost_l946_94621

/-- The number of flower pots -/
def num_pots : ℕ := 6

/-- The price difference between consecutive pots -/
def price_diff : ℚ := 1/10

/-- The price of the largest pot -/
def largest_pot_price : ℚ := 13/8

/-- The total cost of all flower pots -/
def total_cost : ℚ := 33/4

theorem flower_pots_total_cost :
  let prices := List.range num_pots |>.map (fun i => largest_pot_price - i * price_diff)
  prices.sum = total_cost := by sorry

end flower_pots_total_cost_l946_94621


namespace solve_lollipops_problem_l946_94622

def lollipops_problem (alison_lollipops henry_lollipops diane_lollipops days : ℕ) : Prop :=
  alison_lollipops = 60 ∧
  henry_lollipops = alison_lollipops + 30 ∧
  diane_lollipops = 2 * alison_lollipops ∧
  days = 6 ∧
  (alison_lollipops + henry_lollipops + diane_lollipops) / days = 45

theorem solve_lollipops_problem :
  ∃ (alison_lollipops henry_lollipops diane_lollipops days : ℕ),
    lollipops_problem alison_lollipops henry_lollipops diane_lollipops days :=
by
  sorry

end solve_lollipops_problem_l946_94622


namespace zip_code_relationship_l946_94675

/-- 
Theorem: Given a sequence of five numbers A, B, C, D, and E satisfying certain conditions,
prove that the sum of the first two numbers (A + B) equals 2.
-/
theorem zip_code_relationship (A B C D E : ℕ) 
  (sum_condition : A + B + C + D + E = 10)
  (third_zero : C = 0)
  (fourth_double_first : D = 2 * A)
  (fourth_fifth_sum : D + E = 8) :
  A + B = 2 := by
  sorry

end zip_code_relationship_l946_94675


namespace function_inequality_l946_94627

theorem function_inequality (f : ℝ → ℝ) 
  (h1 : ∀ x, f (2 - x) = f x) 
  (h2 : ∀ x ≥ 1, f x = Real.log x) : 
  f (1/2) < f 2 ∧ f 2 < f (1/3) := by
  sorry

end function_inequality_l946_94627


namespace remainder_of_sum_l946_94607

def start_num : ℕ := 11085

theorem remainder_of_sum (start : ℕ) (h : start = start_num) : 
  (2 * (List.sum (List.map (λ i => start + 2 * i) (List.range 8)))) % 14 = 2 := by
  sorry

end remainder_of_sum_l946_94607


namespace lg_17_not_uniquely_calculable_l946_94682

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Given conditions
axiom lg_8 : lg 8 = 0.9031
axiom lg_9 : lg 9 = 0.9542

-- Define a proposition that lg 17 cannot be uniquely calculated
def lg_17_not_calculable : Prop :=
  ∀ f : ℝ → ℝ → ℝ, 
    (∀ x y : ℝ, f (lg x) (lg y) = lg (x + y)) → 
    ¬∃! z : ℝ, f (lg 8) (lg 9) = z ∧ z = lg 17

-- Theorem statement
theorem lg_17_not_uniquely_calculable : lg_17_not_calculable :=
sorry

end lg_17_not_uniquely_calculable_l946_94682


namespace max_blocks_fit_l946_94692

/-- The dimensions of the larger box -/
def box_dimensions : Fin 3 → ℕ
| 0 => 3  -- length
| 1 => 2  -- width
| 2 => 3  -- height
| _ => 0

/-- The dimensions of the smaller block -/
def block_dimensions : Fin 3 → ℕ
| 0 => 2  -- length
| 1 => 2  -- width
| 2 => 1  -- height
| _ => 0

/-- Calculate the volume of a rectangular object given its dimensions -/
def volume (dimensions : Fin 3 → ℕ) : ℕ :=
  dimensions 0 * dimensions 1 * dimensions 2

/-- The maximum number of blocks that can fit in the box -/
def max_blocks : ℕ := 4

/-- Theorem stating that the maximum number of blocks that can fit in the box is 4 -/
theorem max_blocks_fit :
  (volume box_dimensions ≥ max_blocks * volume block_dimensions) ∧
  (∀ n : ℕ, n > max_blocks → volume box_dimensions < n * volume block_dimensions) :=
sorry

end max_blocks_fit_l946_94692


namespace becky_new_necklaces_l946_94674

/-- The number of new necklaces Becky bought -/
def new_necklaces (initial : ℕ) (broken : ℕ) (given_away : ℕ) (final : ℕ) : ℕ :=
  final - (initial - broken - given_away)

/-- Theorem stating that Becky bought 5 new necklaces -/
theorem becky_new_necklaces :
  new_necklaces 50 3 15 37 = 5 := by
  sorry

end becky_new_necklaces_l946_94674


namespace circle_line_distances_l946_94671

/-- The maximum and minimum distances from a point on the circle x^2 + y^2 = 1 to the line x - 2y - 12 = 0 -/
theorem circle_line_distances :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let line := {(x, y) : ℝ × ℝ | x - 2*y - 12 = 0}
  ∃ (max_dist min_dist : ℝ),
    (∀ p ∈ circle, ∀ q ∈ line, dist p q ≤ max_dist) ∧
    (∃ p ∈ circle, ∃ q ∈ line, dist p q = max_dist) ∧
    (∀ p ∈ circle, ∀ q ∈ line, dist p q ≥ min_dist) ∧
    (∃ p ∈ circle, ∃ q ∈ line, dist p q = min_dist) ∧
    max_dist = (12 * Real.sqrt 5) / 5 + 1 ∧
    min_dist = (12 * Real.sqrt 5) / 5 - 1 :=
by sorry

end circle_line_distances_l946_94671


namespace fence_poles_count_l946_94687

theorem fence_poles_count (total_length bridge_length pole_spacing : ℕ) 
  (h1 : total_length = 900)
  (h2 : bridge_length = 42)
  (h3 : pole_spacing = 6) : 
  (2 * ((total_length - bridge_length) / pole_spacing)) = 286 := by
  sorry

end fence_poles_count_l946_94687


namespace nice_people_count_l946_94679

/-- Represents the number of nice people for a given name and total count -/
def nice_count (name : String) (total : ℕ) : ℕ :=
  match name with
  | "Barry" => total
  | "Kevin" => total / 2
  | "Julie" => total * 3 / 4
  | "Joe" => total / 10
  | _ => 0

/-- The total number of nice people in the crowd -/
def total_nice_people : ℕ :=
  nice_count "Barry" 24 + nice_count "Kevin" 20 + nice_count "Julie" 80 + nice_count "Joe" 50

theorem nice_people_count : total_nice_people = 99 := by
  sorry

end nice_people_count_l946_94679


namespace input_is_only_input_statement_l946_94638

-- Define the possible statement types
inductive StatementType
| Output
| Input
| Conditional
| Termination

-- Define the statements
def PRINT : StatementType := StatementType.Output
def INPUT : StatementType := StatementType.Input
def IF : StatementType := StatementType.Conditional
def END : StatementType := StatementType.Termination

-- Theorem: INPUT is the only input statement among the given options
theorem input_is_only_input_statement :
  (PRINT = StatementType.Input → False) ∧
  (INPUT = StatementType.Input) ∧
  (IF = StatementType.Input → False) ∧
  (END = StatementType.Input → False) :=
by sorry

end input_is_only_input_statement_l946_94638


namespace expected_value_of_sum_l946_94603

def marbles : Finset ℕ := {1, 2, 3, 4, 5, 6}

def sum_of_pairs (s : Finset ℕ) : ℕ :=
  (s.powerset.filter (fun t => t.card = 2)).sum (fun t => t.sum id)

def number_of_pairs (s : Finset ℕ) : ℕ :=
  (s.powerset.filter (fun t => t.card = 2)).card

theorem expected_value_of_sum (s : Finset ℕ) :
  s = marbles →
  (sum_of_pairs s : ℚ) / (number_of_pairs s : ℚ) = 7 := by
  sorry

end expected_value_of_sum_l946_94603


namespace no_solutions_to_inequality_system_l946_94655

theorem no_solutions_to_inequality_system :
  ¬ ∃ (x y : ℝ), 11 * x^2 - 10 * x * y + 3 * y^2 ≤ 3 ∧ 5 * x + y ≤ -10 := by
  sorry

end no_solutions_to_inequality_system_l946_94655


namespace min_sum_of_reciprocal_sum_l946_94647

theorem min_sum_of_reciprocal_sum (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 8) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 8 ∧ (a : ℕ) + b = 36 ∧
  ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 8 → (c : ℕ) + d ≥ 36 :=
by sorry

end min_sum_of_reciprocal_sum_l946_94647


namespace constant_term_expansion_l946_94618

theorem constant_term_expansion (x : ℝ) : 
  ∃ (f : ℝ → ℝ), 
    (∀ x, f x = (x^4 + x^2 + 3) * (2*x^5 + x^3 + 7)) ∧ 
    (f 0 = 21) := by
  sorry

end constant_term_expansion_l946_94618


namespace jerry_softball_time_l946_94688

theorem jerry_softball_time (
  num_daughters : ℕ)
  (games_per_daughter : ℕ)
  (practice_hours_per_game : ℕ)
  (game_duration : ℕ)
  (h1 : num_daughters = 4)
  (h2 : games_per_daughter = 12)
  (h3 : practice_hours_per_game = 6)
  (h4 : game_duration = 3) :
  num_daughters * games_per_daughter * (practice_hours_per_game + game_duration) = 432 :=
by sorry

end jerry_softball_time_l946_94688


namespace trader_gain_l946_94632

theorem trader_gain (cost selling_price : ℝ) (h1 : selling_price = 1.25 * cost) : 
  (80 * selling_price - 80 * cost) / cost = 20 := by
  sorry

#check trader_gain

end trader_gain_l946_94632


namespace planes_parallel_l946_94690

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (inPlane : Line → Plane → Prop)
variable (skew : Line → Line → Prop)

-- Theorem statement
theorem planes_parallel (α β : Plane) (a b : Line) :
  (α ≠ β) →
  (perpendicular a α ∧ perpendicular a β) ∨
  (inPlane a α ∧ inPlane b β ∧ parallel a β ∧ parallel b α ∧ skew a b) →
  parallel a β :=
sorry

end planes_parallel_l946_94690


namespace school_teachers_count_l946_94668

theorem school_teachers_count 
  (total : ℕ) 
  (sample_size : ℕ) 
  (sample_students : ℕ) 
  (h1 : total = 2400)
  (h2 : sample_size = 120)
  (h3 : sample_students = 110)
  (h4 : sample_size ≤ total)
  (h5 : sample_students < sample_size) :
  (sample_size - sample_students) * total / sample_size = 200 := by
sorry

end school_teachers_count_l946_94668


namespace sphere_volume_area_ratio_l946_94660

theorem sphere_volume_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (4 / 3 * Real.pi * r₁^3) / (4 / 3 * Real.pi * r₂^3) = 8 / 27 →
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 4 / 9 :=
by sorry

end sphere_volume_area_ratio_l946_94660


namespace article_price_decrease_l946_94683

theorem article_price_decrease (price_after_decrease : ℝ) (decrease_percentage : ℝ) :
  price_after_decrease = 200 ∧ decrease_percentage = 20 →
  (price_after_decrease / (1 - decrease_percentage / 100)) = 250 :=
by sorry

end article_price_decrease_l946_94683
