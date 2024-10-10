import Mathlib

namespace imaginary_sum_zero_l741_74133

theorem imaginary_sum_zero (i : ℂ) (hi : i * i = -1) :
  1 / i + 1 / (i ^ 3) + 1 / (i ^ 5) + 1 / (i ^ 7) = 0 := by
  sorry

end imaginary_sum_zero_l741_74133


namespace knowledge_competition_probability_l741_74185

/-- Represents the probability of correctly answering a single question -/
def p : ℝ := 0.8

/-- Represents the number of preset questions -/
def n : ℕ := 5

/-- Represents the probability of answering exactly 4 questions before advancing -/
def prob_4_questions : ℝ := 2 * p^3 * (1 - p)

theorem knowledge_competition_probability : prob_4_questions = 0.128 := by
  sorry


end knowledge_competition_probability_l741_74185


namespace line_through_center_perpendicular_to_axis_l741_74135

/-- The polar equation of a circle -/
def circle_equation (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ

/-- The center of the circle in Cartesian coordinates -/
def circle_center : ℝ × ℝ := (2, 0)

/-- The polar equation of the line -/
def line_equation (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 2

/-- The line passes through the center of the circle and is perpendicular to the polar axis -/
theorem line_through_center_perpendicular_to_axis :
  (∀ ρ θ : ℝ, circle_equation ρ θ → line_equation ρ θ) ∧
  (line_equation (circle_center.1) 0) ∧
  (∀ ρ : ℝ, line_equation ρ (Real.pi / 2)) :=
sorry

end line_through_center_perpendicular_to_axis_l741_74135


namespace spongebob_daily_earnings_l741_74114

/-- Calculates Spongebob's earnings for the day based on burger and fries sales -/
def spongebob_earnings (num_burgers : ℕ) (burger_price : ℚ) (num_fries : ℕ) (fries_price : ℚ) : ℚ :=
  num_burgers * burger_price + num_fries * fries_price

/-- Theorem stating Spongebob's earnings for the day -/
theorem spongebob_daily_earnings :
  spongebob_earnings 30 2 12 (3/2) = 78 := by
  sorry


end spongebob_daily_earnings_l741_74114


namespace curling_teams_l741_74123

theorem curling_teams (n : ℕ) (h : n * (n - 1) / 2 = 45) : n = 10 := by
  sorry

end curling_teams_l741_74123


namespace sequence_ratio_proof_l741_74168

theorem sequence_ratio_proof (a : ℕ → ℕ+) 
  (h1 : a 1 = 1)
  (h2 : a 2 = 2009)
  (h3 : ∀ n : ℕ, n ≥ 1 → (a (n + 2)) * (a n) = (a (n + 1))^2 + (a (n + 1)) * (a n)) :
  (a 993 : ℚ) / (100 * (a 991)) = 89970 := by
  sorry

end sequence_ratio_proof_l741_74168


namespace pentagon_y_coordinate_l741_74108

/-- Pentagon with vertices A, B, C, D, E -/
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- The area of a rectangle given its width and height -/
def rectangleArea (width height : ℝ) : ℝ := width * height

/-- The area of a triangle given its base and height -/
def triangleArea (base height : ℝ) : ℝ := 0.5 * base * height

/-- The total area of the pentagon -/
def pentagonArea (p : Pentagon) : ℝ :=
  let rectangleABDE := rectangleArea 4 3
  let triangleBCD := triangleArea 4 (p.C.2 - 3)
  rectangleABDE + triangleBCD

theorem pentagon_y_coordinate (p : Pentagon) 
  (h1 : p.A = (0, 0))
  (h2 : p.B = (0, 3))
  (h3 : p.C = (2, p.C.2))
  (h4 : p.D = (4, 3))
  (h5 : p.E = (4, 0))
  (h6 : pentagonArea p = 35) :
  p.C.2 = 14.5 := by
  sorry

end pentagon_y_coordinate_l741_74108


namespace min_sum_of_parallel_vectors_l741_74112

theorem min_sum_of_parallel_vectors (x y : ℝ) : 
  x > 0 → y > 0 → 
  (∃ (k : ℝ), k ≠ 0 ∧ (1 - x, x) = k • (1, -y)) →
  4 ≤ x + y ∧ (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    (∃ (k : ℝ), k ≠ 0 ∧ (1 - x₀, x₀) = k • (1, -y₀)) ∧ 
    x₀ + y₀ = 4) :=
by sorry

end min_sum_of_parallel_vectors_l741_74112


namespace binary_1101_is_13_l741_74155

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.enumFrom 0 b).foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1101_is_13 : 
  binary_to_decimal [true, false, true, true] = 13 := by
  sorry

end binary_1101_is_13_l741_74155


namespace cuboid_dimensions_l741_74158

/-- Represents a cuboid with side areas a, b, and c, and dimensions l, w, h -/
structure Cuboid where
  a : ℝ
  b : ℝ
  c : ℝ
  l : ℝ
  w : ℝ
  h : ℝ

/-- The theorem stating that a cuboid with side areas 5, 8, and 10 has dimensions 4, 2.5, and 2 -/
theorem cuboid_dimensions (cube : Cuboid) 
  (h1 : cube.a = 5) 
  (h2 : cube.b = 8) 
  (h3 : cube.c = 10) 
  (h4 : cube.l * cube.w = cube.a) 
  (h5 : cube.l * cube.h = cube.b) 
  (h6 : cube.w * cube.h = cube.c) :
  cube.l = 4 ∧ cube.w = 2.5 ∧ cube.h = 2 := by
  sorry


end cuboid_dimensions_l741_74158


namespace min_value_and_inequality_l741_74102

theorem min_value_and_inequality (x a b : ℝ) : x > 0 ∧ a > 0 ∧ b > 0 → 
  (∀ y : ℝ, y > 0 → x + 1/x ≤ y + 1/y) ∧ 
  (a * b ≤ ((a + b) / 2)^2) :=
sorry

end min_value_and_inequality_l741_74102


namespace sin_alpha_value_l741_74161

theorem sin_alpha_value (α : Real) : 
  α ∈ Set.Ioo (π) (3*π/2) →  -- α is in the third quadrant
  Real.tan (α + π/4) = 3 → 
  Real.sin α = -Real.sqrt 5 / 5 := by
sorry

end sin_alpha_value_l741_74161


namespace polygon_diagonals_sides_l741_74132

theorem polygon_diagonals_sides (n : ℕ) (h : n = 8) : (n * (n - 3)) / 2 = 2 * n + 7 := by
  sorry

end polygon_diagonals_sides_l741_74132


namespace difference_of_squares_special_case_l741_74127

theorem difference_of_squares_special_case : (727 : ℤ) * 727 - 726 * 728 = 1 := by
  sorry

end difference_of_squares_special_case_l741_74127


namespace money_division_l741_74147

/-- Represents the share ratios of five individuals over five weeks -/
structure ShareRatios :=
  (a b c d e : Fin 5 → ℚ)

/-- Calculates the total ratio for a given week -/
def totalRatio (sr : ShareRatios) (week : Fin 5) : ℚ :=
  sr.a week + sr.b week + sr.c week + sr.d week + sr.e week

/-- Defines the initial ratios and weekly changes -/
def initialRatios : ShareRatios :=
  { a := λ _ => 1,
    b := λ w => 75/100 - w.val * 5/100,
    c := λ w => 60/100 - w.val * 5/100,
    d := λ w => 45/100 - w.val * 5/100,
    e := λ w => 30/100 + w.val * 15/100 }

/-- Theorem statement -/
theorem money_division (sr : ShareRatios) (h1 : sr = initialRatios) 
    (h2 : sr.e 4 * (totalRatio sr 0 / sr.e 4) = 413.33) : 
  sr.e 4 = 120 → totalRatio sr 0 = 413.33 := by
  sorry


end money_division_l741_74147


namespace investment_profit_sharing_l741_74172

/-- Represents the capital contribution of an investor over a year -/
def capital_contribution (initial_investment : ℕ) (doubled_after_six_months : Bool) : ℕ :=
  if doubled_after_six_months
  then initial_investment * 6 + (initial_investment * 2) * 6
  else initial_investment * 12

/-- Represents the profit-sharing ratio between two investors -/
def profit_sharing_ratio (a_contribution : ℕ) (b_contribution : ℕ) : Prop :=
  a_contribution = b_contribution

theorem investment_profit_sharing :
  let a_initial_investment : ℕ := 3000
  let b_initial_investment : ℕ := 4500
  let a_doubles_capital : Bool := true
  let b_doubles_capital : Bool := false
  
  let a_contribution := capital_contribution a_initial_investment a_doubles_capital
  let b_contribution := capital_contribution b_initial_investment b_doubles_capital
  
  profit_sharing_ratio a_contribution b_contribution :=
by
  sorry

end investment_profit_sharing_l741_74172


namespace inequality_proof_l741_74157

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end inequality_proof_l741_74157


namespace janessas_initial_cards_l741_74104

/-- The number of cards Janessa's father gave her -/
def fathers_cards : ℕ := 13

/-- The number of cards Janessa ordered from eBay -/
def ordered_cards : ℕ := 36

/-- The number of cards Janessa threw away -/
def discarded_cards : ℕ := 4

/-- The number of cards Janessa gave to Dexter -/
def cards_given_to_dexter : ℕ := 29

/-- The number of cards Janessa kept for herself -/
def cards_kept_for_self : ℕ := 20

/-- The initial number of cards Janessa had -/
def initial_cards : ℕ := 4

theorem janessas_initial_cards : 
  initial_cards + fathers_cards + ordered_cards - discarded_cards = 
  cards_given_to_dexter + cards_kept_for_self :=
by sorry

end janessas_initial_cards_l741_74104


namespace scientific_notation_of_218000_l741_74171

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_218000 :
  toScientificNotation 218000 = ScientificNotation.mk 2.18 5 sorry := by sorry

end scientific_notation_of_218000_l741_74171


namespace systematic_sampling_theorem_l741_74121

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  totalStudents : Nat
  sampleSize : Nat
  startingNumber : Nat

/-- Generates the sequence of selected student numbers. -/
def generateSequence (s : SystematicSampling) : List Nat :=
  List.range s.sampleSize |>.map (fun i => s.startingNumber + i * (s.totalStudents / s.sampleSize))

/-- Theorem stating the properties of systematic sampling for the given problem. -/
theorem systematic_sampling_theorem (s : SystematicSampling) 
  (h1 : s.totalStudents = 50)
  (h2 : s.sampleSize = 5)
  (h3 : 1 ≤ s.startingNumber)
  (h4 : s.startingNumber ≤ 10) :
  ∃ (a : Nat), 1 ≤ a ∧ a ≤ 10 ∧ 
  generateSequence s = [a, a + 10, a + 20, a + 30, a + 40] :=
sorry

end systematic_sampling_theorem_l741_74121


namespace coin_toss_probability_l741_74136

theorem coin_toss_probability : 
  let n : ℕ := 5  -- Total number of coins
  let k : ℕ := 3  -- Number of tails (or heads, whichever is smaller)
  let p : ℚ := 1/2  -- Probability of getting tails (or heads) on a single toss
  (n.choose k) * p^n = 5/16 :=
sorry

end coin_toss_probability_l741_74136


namespace inverse_proportion_ratio_l741_74140

theorem inverse_proportion_ratio (a₁ a₂ b₁ b₂ c₁ c₂ k : ℝ) 
  (h1 : a₁ ≠ 0) (h2 : a₂ ≠ 0) (h3 : c₁ ≠ 0) (h4 : c₂ ≠ 0)
  (h5 : a₁ * b₁ * c₁ = k) (h6 : a₂ * b₂ * c₂ = k)
  (h7 : a₁ / a₂ = 3 / 4) (h8 : b₁ = 2 * b₂) : 
  c₁ / c₂ = 2 / 3 := by
  sorry

end inverse_proportion_ratio_l741_74140


namespace gcd_power_minus_one_l741_74173

theorem gcd_power_minus_one (k : ℤ) : Int.gcd (k^1024 - 1) (k^1035 - 1) = k - 1 := by
  sorry

end gcd_power_minus_one_l741_74173


namespace percentage_of_black_cats_l741_74141

theorem percentage_of_black_cats 
  (total_cats : ℕ) 
  (white_cats : ℕ) 
  (grey_cats : ℕ) 
  (h1 : total_cats = 16) 
  (h2 : white_cats = 2) 
  (h3 : grey_cats = 10) :
  (((total_cats - white_cats - grey_cats) : ℚ) / total_cats) * 100 = 25 := by
  sorry

end percentage_of_black_cats_l741_74141


namespace b_age_is_twelve_l741_74159

/-- Given three people a, b, and c, where:
  - a is two years older than b
  - b is twice as old as c
  - The total of their ages is 32
  Prove that b is 12 years old -/
theorem b_age_is_twelve (a b c : ℕ) 
  (h1 : a = b + 2) 
  (h2 : b = 2 * c) 
  (h3 : a + b + c = 32) : 
  b = 12 := by
  sorry

end b_age_is_twelve_l741_74159


namespace probability_green_is_81_160_l741_74115

structure Container where
  red : ℕ
  green : ℕ

def containerA : Container := ⟨3, 5⟩
def containerB : Container := ⟨5, 5⟩
def containerC : Container := ⟨7, 3⟩
def containerD : Container := ⟨4, 6⟩

def containers : List Container := [containerA, containerB, containerC, containerD]

def probabilityGreenFromContainer (c : Container) : ℚ :=
  c.green / (c.red + c.green)

def probabilityGreen : ℚ :=
  (1 / containers.length) * (containers.map probabilityGreenFromContainer).sum

theorem probability_green_is_81_160 : probabilityGreen = 81 / 160 := by
  sorry

end probability_green_is_81_160_l741_74115


namespace animal_shelter_count_l741_74196

/-- The number of cats received by the animal shelter -/
def num_cats : ℕ := 40

/-- The difference between the number of cats and dogs -/
def cat_dog_difference : ℕ := 20

/-- The total number of animals received by the shelter -/
def total_animals : ℕ := num_cats + (num_cats - cat_dog_difference)

theorem animal_shelter_count : total_animals = 60 := by
  sorry

end animal_shelter_count_l741_74196


namespace y2_greater_than_y1_l741_74120

-- Define the linear function
def f (x : ℝ) : ℝ := -2 * x + 1

-- Define the points A and B
def A : ℝ × ℝ := (-1, f (-1))
def B : ℝ × ℝ := (-2, f (-2))

-- Theorem statement
theorem y2_greater_than_y1 : A.2 < B.2 := by
  sorry

end y2_greater_than_y1_l741_74120


namespace box_2_neg1_3_equals_1_l741_74137

def box (a b c : ℤ) : ℚ :=
  let k : ℤ := 2
  (k : ℚ) * (a : ℚ) ^ b - (b : ℚ) ^ c + (c : ℚ) ^ (a - k)

theorem box_2_neg1_3_equals_1 :
  box 2 (-1) 3 = 1 := by sorry

end box_2_neg1_3_equals_1_l741_74137


namespace buffet_meal_combinations_l741_74175

theorem buffet_meal_combinations : ℕ := by
  -- Define the number of options for each food category
  let num_meats : ℕ := 4
  let num_vegetables : ℕ := 4
  let num_desserts : ℕ := 4
  let num_drinks : ℕ := 2

  -- Define the number of items Tyler chooses from each category
  let chosen_meats : ℕ := 2
  let chosen_vegetables : ℕ := 2
  let chosen_desserts : ℕ := 1
  let chosen_drinks : ℕ := 1

  -- Calculate the total number of meal combinations
  have h : (Nat.choose num_meats chosen_meats) * 
           (Nat.choose num_vegetables chosen_vegetables) * 
           num_desserts * num_drinks = 288 := by sorry

  exact 288

end buffet_meal_combinations_l741_74175


namespace sixty_first_term_is_201_l741_74162

/-- An arithmetic sequence with a_5 = 33 and common difference d = 3 -/
def arithmetic_sequence (n : ℕ) : ℝ :=
  33 + 3 * (n - 5)

/-- Theorem: The 61st term of the sequence is 201 -/
theorem sixty_first_term_is_201 : arithmetic_sequence 61 = 201 := by
  sorry

end sixty_first_term_is_201_l741_74162


namespace erica_saw_three_warthogs_l741_74130

/-- Represents the number of animals Erica saw on each day of her safari --/
structure SafariCount where
  saturday : Nat
  sunday : Nat
  monday_rhinos : Nat
  monday_warthogs : Nat

/-- The total number of animals seen during the safari --/
def total_animals : Nat := 20

/-- The number of animals Erica saw on Saturday --/
def saturday_count : Nat := 3 + 2

/-- The number of animals Erica saw on Sunday --/
def sunday_count : Nat := 2 + 5

/-- The number of rhinos Erica saw on Monday --/
def monday_rhinos : Nat := 5

/-- Theorem stating that Erica saw 3 warthogs on Monday --/
theorem erica_saw_three_warthogs (safari : SafariCount) :
  safari.saturday = saturday_count →
  safari.sunday = sunday_count →
  safari.monday_rhinos = monday_rhinos →
  safari.saturday + safari.sunday + safari.monday_rhinos + safari.monday_warthogs = total_animals →
  safari.monday_warthogs = 3 := by
  sorry


end erica_saw_three_warthogs_l741_74130


namespace min_value_of_expression_min_value_achieved_l741_74105

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + b = 1) :
  1 / a + 27 / b ≥ 48 := by
  sorry

theorem min_value_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 3 * a + b = 1 ∧ 1 / a + 27 / b < 48 + ε := by
  sorry

end min_value_of_expression_min_value_achieved_l741_74105


namespace absolute_value_theorem_l741_74119

theorem absolute_value_theorem (x : ℝ) (h : x < -1) :
  |x - Real.sqrt ((x + 2)^2)| = -2*x - 2 := by
  sorry

end absolute_value_theorem_l741_74119


namespace opposite_of_negative_seven_l741_74111

-- Definition of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem statement
theorem opposite_of_negative_seven :
  opposite (-7) = 7 :=
by
  -- The proof goes here
  sorry

end opposite_of_negative_seven_l741_74111


namespace roots_transformation_l741_74122

theorem roots_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 4*r₁^2 + r₁ + 6 = 0) → 
  (r₂^3 - 4*r₂^2 + r₂ + 6 = 0) → 
  (r₃^3 - 4*r₃^2 + r₃ + 6 = 0) → 
  ∀ x, (x - 3*r₁) * (x - 3*r₂) * (x - 3*r₃) = x^3 - 12*x^2 + 9*x + 162 :=
by sorry

end roots_transformation_l741_74122


namespace total_rainfall_five_days_l741_74167

/-- Represents the rainfall data for a day -/
structure RainfallData where
  hours : ℝ
  rate : ℝ

/-- Calculates the total rainfall for a given day -/
def totalRainfall (data : RainfallData) : ℝ :=
  data.hours * data.rate

theorem total_rainfall_five_days (monday tuesday wednesday thursday friday : RainfallData)
  (h_monday : monday = { hours := 5, rate := 1 })
  (h_tuesday : tuesday = { hours := 3, rate := 1.5 })
  (h_wednesday : wednesday = { hours := 4, rate := 2 * monday.rate })
  (h_thursday : thursday = { hours := 6, rate := 0.5 * tuesday.rate })
  (h_friday : friday = { hours := 2, rate := 1.5 * wednesday.rate }) :
  totalRainfall monday + totalRainfall tuesday + totalRainfall wednesday +
  totalRainfall thursday + totalRainfall friday = 28 := by
  sorry

end total_rainfall_five_days_l741_74167


namespace wendy_ribbon_calculation_l741_74151

/-- The amount of ribbon Wendy used to wrap presents, in inches. -/
def ribbon_used : ℕ := 46

/-- The amount of ribbon Wendy had left, in inches. -/
def ribbon_left : ℕ := 38

/-- The total amount of ribbon Wendy bought, in inches. -/
def total_ribbon : ℕ := ribbon_used + ribbon_left

theorem wendy_ribbon_calculation :
  total_ribbon = 84 := by
  sorry

end wendy_ribbon_calculation_l741_74151


namespace terminal_side_of_negative_400_degrees_l741_74106

/-- The quadrant of an angle in degrees -/
inductive Quadrant
  | first
  | second
  | third
  | fourth

/-- Normalizes an angle to the range [0, 360) -/
def normalizeAngle (angle : Int) : Int :=
  (angle % 360 + 360) % 360

/-- Determines the quadrant of a normalized angle -/
def quadrantOfNormalizedAngle (angle : Int) : Quadrant :=
  if 0 ≤ angle ∧ angle < 90 then Quadrant.first
  else if 90 ≤ angle ∧ angle < 180 then Quadrant.second
  else if 180 ≤ angle ∧ angle < 270 then Quadrant.third
  else Quadrant.fourth

/-- Determines the quadrant of any angle -/
def quadrantOfAngle (angle : Int) : Quadrant :=
  quadrantOfNormalizedAngle (normalizeAngle angle)

theorem terminal_side_of_negative_400_degrees :
  quadrantOfAngle (-400) = Quadrant.fourth := by
  sorry

end terminal_side_of_negative_400_degrees_l741_74106


namespace y_range_given_inequality_l741_74117

/-- Custom multiplication operation on ℝ -/
def star (x y : ℝ) : ℝ := x * (1 - y)

/-- Theorem stating the range of y given the condition -/
theorem y_range_given_inequality :
  (∀ x : ℝ, star (x - y) (x + y) < 1) →
  ∃ a b : ℝ, a = -1/2 ∧ b = 3/2 ∧ y ∈ Set.Ioo a b :=
by sorry

end y_range_given_inequality_l741_74117


namespace committee_meeting_attendance_l741_74129

theorem committee_meeting_attendance :
  ∀ (associate_profs assistant_profs : ℕ),
    2 * associate_profs + assistant_profs = 7 →
    associate_profs + 2 * assistant_profs = 11 →
    associate_profs + assistant_profs = 6 :=
by
  sorry

end committee_meeting_attendance_l741_74129


namespace P_subset_Q_l741_74191

-- Define the sets P and Q
def P : Set ℝ := {x | x^2 - 3*x + 2 < 0}
def Q : Set ℝ := {x | 1 < x ∧ x < 3}

-- Theorem statement
theorem P_subset_Q : P ⊆ Q := by
  sorry

end P_subset_Q_l741_74191


namespace parallelogram_intersection_theorem_l741_74160

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram -/
structure Parallelogram where
  E : Point
  F : Point
  G : Point
  H : Point

/-- Checks if a point is on the extension of a line segment -/
def isOnExtension (A B H : Point) : Prop := sorry

/-- Checks if two line segments intersect at a point -/
def intersectsAt (P Q R S J : Point) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (P Q : Point) : ℝ := sorry

/-- Main theorem -/
theorem parallelogram_intersection_theorem (EFGH : Parallelogram) (H J K : Point) : 
  isOnExtension EFGH.E EFGH.F H →
  intersectsAt EFGH.G H EFGH.E EFGH.F J →
  intersectsAt EFGH.G H EFGH.F EFGH.G K →
  distance J K = 40 →
  distance H K = 30 →
  distance EFGH.G J = 20 := by sorry

end parallelogram_intersection_theorem_l741_74160


namespace books_remaining_l741_74134

/-- Given Sandy has 10 books, Tim has 33 books, and Benny lost 24 of their books,
    prove that they have 19 books together now. -/
theorem books_remaining (sandy_books tim_books lost_books : ℕ) 
  (h1 : sandy_books = 10)
  (h2 : tim_books = 33)
  (h3 : lost_books = 24) : 
  sandy_books + tim_books - lost_books = 19 := by
  sorry

end books_remaining_l741_74134


namespace intercept_sum_l741_74163

/-- Given a line with equation y - 3 = -3(x - 5), prove that the sum of its x-intercept and y-intercept is 24 -/
theorem intercept_sum (x y : ℝ) : 
  (y - 3 = -3 * (x - 5)) → 
  ∃ (x_int y_int : ℝ), 
    (0 - 3 = -3 * (x_int - 5)) ∧ 
    (y_int - 3 = -3 * (0 - 5)) ∧ 
    (x_int + y_int = 24) := by
  sorry

end intercept_sum_l741_74163


namespace tennis_balls_count_l741_74188

theorem tennis_balls_count (baskets : ℕ) (soccer_balls : ℕ) (students_8 : ℕ) (students_10 : ℕ) 
  (balls_removed_8 : ℕ) (balls_removed_10 : ℕ) (balls_remaining : ℕ) :
  baskets = 5 →
  soccer_balls = 5 →
  students_8 = 3 →
  students_10 = 2 →
  balls_removed_8 = 8 →
  balls_removed_10 = 10 →
  balls_remaining = 56 →
  ∃ T : ℕ, 
    baskets * (T + soccer_balls) - (students_8 * balls_removed_8 + students_10 * balls_removed_10) = balls_remaining ∧
    T = 15 :=
by sorry

end tennis_balls_count_l741_74188


namespace find_x_and_y_l741_74187

theorem find_x_and_y :
  ∃ (x y : ℚ), 3 * (2 * x + 9 * y) = 75 ∧ x + y = 10 ∧ x = 65/7 ∧ y = 5/7 := by
  sorry

end find_x_and_y_l741_74187


namespace bakers_new_cakes_l741_74109

/-- Baker's cake problem -/
theorem bakers_new_cakes 
  (initial_cakes : ℕ) 
  (sold_cakes : ℕ) 
  (difference : ℕ) 
  (h1 : initial_cakes = 13) 
  (h2 : sold_cakes = 91) 
  (h3 : difference = 63) : 
  sold_cakes + difference = 154 := by
  sorry

end bakers_new_cakes_l741_74109


namespace decimal_118_to_base6_l741_74148

def decimal_to_base6 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 6) ((m % 6) :: acc)
    aux n []

theorem decimal_118_to_base6 :
  decimal_to_base6 118 = [3, 1, 4] :=
sorry

end decimal_118_to_base6_l741_74148


namespace fraction_sum_and_lcd_l741_74124

theorem fraction_sum_and_lcd : 
  let fractions : List ℚ := [1/2, 1/3, 1/4, 1/5, 1/6, 1/8, 1/9]
  let lcd := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 9)))))
  lcd = 360 ∧ fractions.sum = 607 / 360 := by
  sorry

end fraction_sum_and_lcd_l741_74124


namespace orchid_count_l741_74107

/-- Time in minutes to paint each type of flower or vine -/
def lily_time : ℕ := 5
def rose_time : ℕ := 7
def orchid_time : ℕ := 3
def vine_time : ℕ := 2

/-- Number of each type of flower or vine painted -/
def lily_count : ℕ := 17
def rose_count : ℕ := 10
def vine_count : ℕ := 20

/-- Total time spent painting -/
def total_time : ℕ := 213

/-- Theorem stating the number of orchids painted -/
theorem orchid_count : 
  ∃ (x : ℕ), x * orchid_time = total_time - (lily_count * lily_time + rose_count * rose_time + vine_count * vine_time) ∧ x = 6 := by
  sorry

end orchid_count_l741_74107


namespace max_radius_circle_x_value_l741_74113

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the theorem
theorem max_radius_circle_x_value 
  (C : ℝ × ℝ → ℝ → Set (ℝ × ℝ)) 
  (max_radius : ℝ) 
  (x : ℝ) :
  (∀ r : ℝ, r ≤ max_radius) →
  ((8, 0) ∈ C (0, 0) max_radius) →
  ((x, 0) ∈ C (0, 0) max_radius) →
  (max_radius = 8) →
  (x = -8) :=
by sorry

end max_radius_circle_x_value_l741_74113


namespace ellipse_and_circle_l741_74143

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  focal_length : ℝ
  short_axis_length : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_focal : focal_length = 2 * Real.sqrt 6
  h_short : short_axis_length = 2 * Real.sqrt 2

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The line that intersects the ellipse -/
def intersecting_line (x y : ℝ) : Prop :=
  y = x + 2

/-- Main theorem about the ellipse and its intersecting circle -/
theorem ellipse_and_circle (e : Ellipse) :
  (∀ x y, ellipse_equation e x y ↔ x^2 / 8 + y^2 / 2 = 1) ∧
  (∃ A B : ℝ × ℝ,
    ellipse_equation e A.1 A.2 ∧
    ellipse_equation e B.1 B.2 ∧
    intersecting_line A.1 A.2 ∧
    intersecting_line B.1 B.2 ∧
    ∀ x y, (x + 8/5)^2 + (y - 2/5)^2 = 48/25 ↔
      ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
        x = (1 - t) * A.1 + t * B.1 ∧
        y = (1 - t) * A.2 + t * B.2) :=
sorry

end ellipse_and_circle_l741_74143


namespace price_ratio_theorem_l741_74195

theorem price_ratio_theorem (cost : ℝ) (price1 price2 : ℝ) 
  (h1 : price1 = cost * (1 + 0.35))
  (h2 : price2 = cost * (1 - 0.10)) :
  price2 / price1 = 2 / 3 := by
  sorry

end price_ratio_theorem_l741_74195


namespace largest_interior_angle_of_triangle_l741_74149

theorem largest_interior_angle_of_triangle (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  (a : ℝ) / 2 = b / 3 → b / 3 = c / 4 →
  a + b + c = 360 →
  180 - min a (min b c) = 100 := by
  sorry

end largest_interior_angle_of_triangle_l741_74149


namespace prime_power_sum_implies_power_of_three_l741_74198

theorem prime_power_sum_implies_power_of_three (n : ℕ) :
  Nat.Prime (1 + 2^n + 4^n) → ∃ k : ℕ+, n = 3^(k : ℕ) := by
  sorry

end prime_power_sum_implies_power_of_three_l741_74198


namespace apple_distribution_theorem_l741_74101

def distribute_apples (total_apples : ℕ) (num_people : ℕ) (min_apples : ℕ) : ℕ :=
  Nat.choose (total_apples - num_people * min_apples + num_people - 1) (num_people - 1)

theorem apple_distribution_theorem :
  distribute_apples 30 3 3 = 253 :=
sorry

end apple_distribution_theorem_l741_74101


namespace initial_passengers_l741_74169

theorem initial_passengers (remaining : ℕ) : 
  remaining = 216 →
  ∃ initial : ℕ,
    initial > 0 ∧
    remaining = initial - 
      (initial / 10 + 
       (initial - initial / 10) / 7 + 
       (initial - initial / 10 - (initial - initial / 10) / 7) / 5) ∧
    initial = 350 := by
  sorry

end initial_passengers_l741_74169


namespace min_value_function_l741_74179

theorem min_value_function (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  (6 * x^2 + 9 * x + 2 * y^2 + 3 * y + 20) / (9 * (x + y + 2)) ≥ 4 * Real.sqrt 10 / 3 := by
  sorry

end min_value_function_l741_74179


namespace cyclist_journey_l741_74100

theorem cyclist_journey 
  (v : ℝ) -- original speed in mph
  (t : ℝ) -- original time in hours
  (d : ℝ) -- distance in miles
  (h₁ : d = v * t) -- distance = speed * time
  (h₂ : d = (v + 1/3) * (3/4 * t)) -- increased speed condition
  (h₃ : d = (v - 1/3) * (t + 3/2)) -- decreased speed condition
  : v = 1 ∧ d = 3 := by
  sorry

end cyclist_journey_l741_74100


namespace absolute_value_ab_l741_74181

-- Define the constants for the foci locations
def ellipse_focus : ℝ := 5
def hyperbola_focus : ℝ := 7

-- Define the equations for the ellipse and hyperbola
def ellipse_equation (a b : ℝ) : Prop := b^2 - a^2 = ellipse_focus^2
def hyperbola_equation (a b : ℝ) : Prop := a^2 + b^2 = hyperbola_focus^2

-- Theorem statement
theorem absolute_value_ab (a b : ℝ) 
  (h_ellipse : ellipse_equation a b) 
  (h_hyperbola : hyperbola_equation a b) : 
  |a * b| = 2 * Real.sqrt 111 := by
  sorry

end absolute_value_ab_l741_74181


namespace sunRiseOnlyCertainEvent_l741_74144

-- Define the type for events
inductive Event
  | SunRise
  | OpenBook
  | Thumbtack
  | Student

-- Define a function to check if an event is certain
def isCertain (e : Event) : Prop :=
  match e with
  | Event.SunRise => true
  | _ => false

-- Theorem stating that SunRise is the only certain event
theorem sunRiseOnlyCertainEvent : 
  ∀ (e : Event), isCertain e ↔ e = Event.SunRise :=
by
  sorry


end sunRiseOnlyCertainEvent_l741_74144


namespace distance_p_ran_l741_74183

/-- A race between two runners p and q, where p is faster but q gets a head start. -/
structure Race where
  /-- The speed of runner q in meters per minute -/
  v : ℝ
  /-- The time of the race in minutes -/
  t : ℝ
  /-- The head start distance given to runner q in meters -/
  d : ℝ
  /-- Assumption that the speeds and time are positive -/
  hv : v > 0
  ht : t > 0
  /-- Assumption that p runs 30% faster than q -/
  hp_speed : ℝ := 1.3 * v
  /-- Assumption that the race ends in a tie -/
  h_tie : d + v * t = hp_speed * t

/-- The theorem stating the distance p ran in the race -/
theorem distance_p_ran (race : Race) : ℝ := by
  sorry

end distance_p_ran_l741_74183


namespace niles_collection_l741_74194

/-- The total amount collected by Niles from the book club members -/
def total_collected (num_members : ℕ) (snack_fee : ℕ) (num_hardcover : ℕ) (hardcover_price : ℕ) (num_paperback : ℕ) (paperback_price : ℕ) : ℕ :=
  num_members * (snack_fee + num_hardcover * hardcover_price + num_paperback * paperback_price)

/-- Theorem stating the total amount collected by Niles -/
theorem niles_collection : 
  total_collected 6 150 6 30 6 12 = 2412 := by
  sorry

end niles_collection_l741_74194


namespace pushup_difference_l741_74174

theorem pushup_difference (zachary_pushups john_pushups : ℕ) 
  (h1 : zachary_pushups = 51)
  (h2 : john_pushups = 69)
  (h3 : ∃ david_pushups : ℕ, david_pushups = john_pushups + 4) :
  ∃ david_pushups : ℕ, david_pushups - zachary_pushups = 22 :=
by sorry

end pushup_difference_l741_74174


namespace geometric_sequence_fourth_term_l741_74116

theorem geometric_sequence_fourth_term 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : ∀ n, S n = a 1 * (1 - q^n) / (1 - q)) 
  (h2 : a 2 = 2) 
  (h3 : q > 0) 
  (h4 : S 4 / S 2 = 10) : 
  a 4 = 18 := by
sorry

end geometric_sequence_fourth_term_l741_74116


namespace c_younger_than_a_l741_74199

-- Define variables for the ages of A, B, and C
variable (a b c : ℕ)

-- Define the condition given in the problem
def age_difference : Prop := a + b = b + c + 11

-- Theorem to prove
theorem c_younger_than_a (h : age_difference a b c) : a - c = 11 := by
  sorry

end c_younger_than_a_l741_74199


namespace modulus_of_complex_fraction_l741_74150

open Complex

theorem modulus_of_complex_fraction : 
  let z : ℂ := exp (π / 3 * I)
  ∀ (euler_formula : ∀ x : ℝ, exp (x * I) = cos x + I * sin x),
  abs (z / (1 - I)) = Real.sqrt 2 / 2 := by
  sorry

end modulus_of_complex_fraction_l741_74150


namespace peanut_distribution_theorem_l741_74153

/-- Represents the distribution of peanuts among three people -/
structure PeanutDistribution where
  alex : ℕ
  betty : ℕ
  charlie : ℕ

/-- Checks if three numbers form a geometric progression -/
def is_geometric_progression (a b c : ℕ) : Prop :=
  ∃ r : ℚ, r > 0 ∧ b = a * r ∧ c = b * r

/-- Checks if three numbers form an arithmetic progression -/
def is_arithmetic_progression (a b c : ℕ) : Prop :=
  ∃ d : ℤ, b = a + d ∧ c = b + d

/-- The main theorem about the peanut distribution -/
theorem peanut_distribution_theorem (init : PeanutDistribution) 
  (h_total : init.alex + init.betty + init.charlie = 444)
  (h_order : init.alex < init.betty ∧ init.betty < init.charlie)
  (h_geometric : is_geometric_progression init.alex init.betty init.charlie)
  (final : PeanutDistribution)
  (h_eating : final.alex = init.alex - 5 ∧ final.betty = init.betty - 9 ∧ final.charlie = init.charlie - 25)
  (h_arithmetic : is_arithmetic_progression final.alex final.betty final.charlie) :
  init.alex = 108 := by
  sorry

end peanut_distribution_theorem_l741_74153


namespace system_solution_l741_74145

/-- The system of linear equations -/
def system (x y : ℝ) : Prop :=
  x + y = 6 ∧ x = 2*y

/-- The solution set of the system -/
def solution_set : Set (ℝ × ℝ) :=
  {(4, 2)}

/-- Theorem stating that the solution set is correct -/
theorem system_solution : 
  {(x, y) | system x y} = solution_set :=
sorry

end system_solution_l741_74145


namespace pentagon_rectangle_ratio_l741_74138

/-- The ratio of a regular pentagon's side length to a rectangle's width, 
    given that they have the same perimeter and the rectangle's length is twice its width -/
theorem pentagon_rectangle_ratio (perimeter : ℝ) : 
  perimeter > 0 → 
  (5 : ℝ) * (perimeter / 5) / (perimeter / 6) = 6 / 5 := by
  sorry

end pentagon_rectangle_ratio_l741_74138


namespace charles_milk_amount_l741_74152

/-- The amount of chocolate milk in each glass (in ounces) -/
def glass_size : ℝ := 8

/-- The amount of milk in each glass (in ounces) -/
def milk_per_glass : ℝ := 6.5

/-- The amount of chocolate syrup in each glass (in ounces) -/
def syrup_per_glass : ℝ := 1.5

/-- The total amount of chocolate syrup Charles has (in ounces) -/
def total_syrup : ℝ := 60

/-- The total amount of chocolate milk Charles will drink (in ounces) -/
def total_milk : ℝ := 160

/-- Theorem stating that Charles has 130 ounces of milk -/
theorem charles_milk_amount : 
  ∃ (num_glasses : ℝ),
    num_glasses * glass_size = total_milk ∧
    num_glasses * syrup_per_glass ≤ total_syrup ∧
    num_glasses * milk_per_glass = 130 := by
  sorry

end charles_milk_amount_l741_74152


namespace problem_solution_l741_74139

theorem problem_solution (x y : ℝ) (h1 : 3 * x + y = 6) (h2 : x + 3 * y = 6) :
  3 * x^2 + 5 * x * y + 3 * y^2 = 99 / 4 := by
  sorry

end problem_solution_l741_74139


namespace bill_take_home_salary_l741_74192

def take_home_salary (gross_salary property_taxes sales_taxes income_tax_rate : ℝ) : ℝ :=
  gross_salary - (property_taxes + sales_taxes + income_tax_rate * gross_salary)

theorem bill_take_home_salary :
  take_home_salary 50000 2000 3000 0.1 = 40000 := by
  sorry

end bill_take_home_salary_l741_74192


namespace rosy_work_days_l741_74176

/-- Given that Mary can do a piece of work in 26 days and Rosy is 30% more efficient than Mary,
    prove that Rosy will take 20 days to do the same piece of work. -/
theorem rosy_work_days (mary_days : ℝ) (rosy_efficiency : ℝ) :
  mary_days = 26 →
  rosy_efficiency = 1.3 →
  (mary_days / rosy_efficiency : ℝ) = 20 := by
  sorry

end rosy_work_days_l741_74176


namespace smallest_number_with_weight_2000_l741_74178

/-- The weight of a number is the sum of its digits -/
def weight (n : ℕ) : ℕ := sorry

/-- Construct a number with a leading digit followed by a sequence of nines -/
def constructNumber (lead : ℕ) (nines : ℕ) : ℕ := sorry

theorem smallest_number_with_weight_2000 :
  ∀ n : ℕ, weight n = 2000 → n ≥ constructNumber 2 222 := by sorry

end smallest_number_with_weight_2000_l741_74178


namespace f_derivative_at_zero_implies_a_equals_one_l741_74142

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (2 * x) / (x + a)

theorem f_derivative_at_zero_implies_a_equals_one (a : ℝ) :
  (deriv (f a)) 0 = 1 → a = 1 := by
  sorry

end f_derivative_at_zero_implies_a_equals_one_l741_74142


namespace alligators_not_hiding_l741_74184

theorem alligators_not_hiding (total_alligators hiding_alligators : ℕ) 
  (h1 : total_alligators = 75)
  (h2 : hiding_alligators = 19) : 
  total_alligators - hiding_alligators = 56 := by
  sorry

end alligators_not_hiding_l741_74184


namespace lambda_positive_infinite_lambda_negative_infinite_l741_74118

/-- Definition of Ω(n) -/
def Omega (n : ℕ) : ℕ := sorry

/-- Definition of λ(n) -/
def lambda (n : ℕ) : Int := (-1) ^ (Omega n)

/-- The set of positive integers n such that λ(n) = λ(n+1) = 1 is infinite -/
theorem lambda_positive_infinite : Set.Infinite {n : ℕ | lambda n = 1 ∧ lambda (n + 1) = 1} := by sorry

/-- The set of positive integers n such that λ(n) = λ(n+1) = -1 is infinite -/
theorem lambda_negative_infinite : Set.Infinite {n : ℕ | lambda n = -1 ∧ lambda (n + 1) = -1} := by sorry

end lambda_positive_infinite_lambda_negative_infinite_l741_74118


namespace smallest_sum_of_quadratic_roots_l741_74197

theorem smallest_sum_of_quadratic_roots (c d : ℝ) : 
  c > 0 → d > 0 → 
  (∃ x : ℝ, x^2 + c*x + 3*d = 0) → 
  (∃ y : ℝ, y^2 + 3*d*y + c = 0) → 
  c + 3*d ≥ 8 := by
sorry

end smallest_sum_of_quadratic_roots_l741_74197


namespace fraction_equality_solution_l741_74190

theorem fraction_equality_solution : ∃! x : ℝ, (4 + x) / (6 + x) = (2 + x) / (3 + x) := by
  sorry

end fraction_equality_solution_l741_74190


namespace unique_root_implies_specific_function_max_min_on_interval_l741_74189

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + b * x

-- Theorem 1
theorem unique_root_implies_specific_function (a b : ℝ) (h1 : a ≠ 0) (h2 : f a b 2 = 0) 
  (h3 : ∃! x, f a b x - x = 0) : 
  ∀ x, f (-1/2) 1 x = f a b x := by sorry

-- Theorem 2
theorem max_min_on_interval (x : ℝ) (h : x ∈ Set.Icc (-1) 2) : 
  f 1 (-2) x ≤ 3 ∧ f 1 (-2) x ≥ -1 ∧ 
  (∃ x₁ ∈ Set.Icc (-1) 2, f 1 (-2) x₁ = 3) ∧ 
  (∃ x₂ ∈ Set.Icc (-1) 2, f 1 (-2) x₂ = -1) := by sorry

end unique_root_implies_specific_function_max_min_on_interval_l741_74189


namespace optimal_fence_placement_l741_74154

/-- Represents the state of a tree (healthy or dead) --/
inductive TreeState
  | Healthy
  | Dead

/-- Represents a 6x6 grid of trees --/
def TreeGrid := Fin 6 → Fin 6 → TreeState

/-- Represents a fence placement --/
structure FencePlacement where
  vertical : Fin 3
  horizontal : Fin 3

/-- Checks if a tree is isolated by the given fence placement --/
def isIsolated (grid : TreeGrid) (fences : FencePlacement) (row col : Fin 6) : Prop :=
  sorry

/-- Counts the number of healthy trees in the grid --/
def countHealthyTrees (grid : TreeGrid) : Nat :=
  sorry

/-- The main theorem to be proved --/
theorem optimal_fence_placement
  (grid : TreeGrid)
  (healthy_count : countHealthyTrees grid = 20) :
  ∃ (fences : FencePlacement),
    ∀ (row col : Fin 6),
      grid row col = TreeState.Healthy →
      isIsolated grid fences row col :=
sorry

end optimal_fence_placement_l741_74154


namespace sqrt_sum_inverse_squares_l741_74103

theorem sqrt_sum_inverse_squares : 
  Real.sqrt (1 / 25 + 1 / 36) = Real.sqrt 61 / 30 := by
  sorry

end sqrt_sum_inverse_squares_l741_74103


namespace lines_intersect_at_point_l741_74182

/-- Represents a 2D point or vector -/
structure Vec2 where
  x : ℚ
  y : ℚ

/-- Represents a parametric line in 2D -/
structure ParamLine where
  origin : Vec2
  direction : Vec2

/-- The first line -/
def line1 : ParamLine := {
  origin := { x := 1, y := 4 },
  direction := { x := -2, y := 3 }
}

/-- The second line -/
def line2 : ParamLine := {
  origin := { x := 0, y := 5 },
  direction := { x := -1, y := 6 }
}

/-- The intersection point of the two lines -/
def intersection : Vec2 := { x := -1/9, y := 17/3 }

/-- Theorem stating that the given point is the intersection of the two lines -/
theorem lines_intersect_at_point : 
  ∃ (s v : ℚ), 
    line1.origin.x + s * line1.direction.x = intersection.x ∧
    line1.origin.y + s * line1.direction.y = intersection.y ∧
    line2.origin.x + v * line2.direction.x = intersection.x ∧
    line2.origin.y + v * line2.direction.y = intersection.y :=
by
  sorry

end lines_intersect_at_point_l741_74182


namespace quadratic_one_solution_l741_74110

theorem quadratic_one_solution (n : ℝ) : 
  (n > 0 ∧ ∃! x : ℝ, 4 * x^2 + n * x + 4 = 0) ↔ n = 8 := by
  sorry

end quadratic_one_solution_l741_74110


namespace product_xyz_equals_one_l741_74170

theorem product_xyz_equals_one 
  (x y z : ℝ) 
  (h1 : x + 1/y = 2) 
  (h2 : y + 1/z = 2) : 
  x * y * z = 1 := by
sorry

end product_xyz_equals_one_l741_74170


namespace sector_area_l741_74164

/-- The area of a circular sector with radius R and circumference 4R is R^2 -/
theorem sector_area (R : ℝ) (R_pos : R > 0) : 
  let circumference := 4 * R
  let arc_length := circumference - 2 * R
  let sector_area := (1 / 2) * arc_length * R
  sector_area = R^2 := by
sorry

end sector_area_l741_74164


namespace circles_are_tangent_l741_74146

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 16*y - 48 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 8*y - 44 = 0

-- Define what it means for two circles to be tangent
def are_tangent (c1 c2 : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), c1 x y ∧ c2 x y ∧ 
  ∀ (x' y' : ℝ), (x' ≠ x ∨ y' ≠ y) → ¬(c1 x' y' ∧ c2 x' y')

-- Theorem statement
theorem circles_are_tangent : are_tangent circle1 circle2 := by
  sorry

end circles_are_tangent_l741_74146


namespace charlie_area_is_72_l741_74156

-- Define the total area to be painted
def total_area : ℝ := 360

-- Define the ratio of work done by each person
def allen_ratio : ℝ := 3
def ben_ratio : ℝ := 5
def charlie_ratio : ℝ := 2

-- Define the total ratio
def total_ratio : ℝ := allen_ratio + ben_ratio + charlie_ratio

-- Theorem to prove
theorem charlie_area_is_72 : 
  charlie_ratio / total_ratio * total_area = 72 := by
  sorry

end charlie_area_is_72_l741_74156


namespace functional_equation_solution_l741_74177

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x, f x + 2 * f (1 - x) = 4 * x^2 + 3

/-- Theorem stating that for any function satisfying the functional equation, f(4) = 11/3 -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) : f 4 = 11/3 := by
  sorry

end functional_equation_solution_l741_74177


namespace function_upper_bound_l741_74166

theorem function_upper_bound (x : ℝ) (h : x > 0) : (1 + Real.log x) / x ≤ 1 := by
  sorry

end function_upper_bound_l741_74166


namespace earth_surface_available_for_living_l741_74193

theorem earth_surface_available_for_living : 
  let earth_surface : ℝ := 1
  let land_fraction : ℝ := 1 / 3
  let inhabitable_fraction : ℝ := 1 / 4
  let residential_fraction : ℝ := 0.6
  earth_surface * land_fraction * inhabitable_fraction * residential_fraction = 1 / 20 :=
by sorry

end earth_surface_available_for_living_l741_74193


namespace bread_calculation_l741_74128

def initial_bread : ℕ := 200

def day1_fraction : ℚ := 1/4
def day2_fraction : ℚ := 2/5
def day3_fraction : ℚ := 1/2

def remaining_bread : ℕ := 45

theorem bread_calculation :
  (initial_bread - (day1_fraction * initial_bread).floor) -
  (day2_fraction * (initial_bread - (day1_fraction * initial_bread).floor)).floor -
  (day3_fraction * ((initial_bread - (day1_fraction * initial_bread).floor) -
    (day2_fraction * (initial_bread - (day1_fraction * initial_bread).floor)).floor)).floor = remaining_bread := by
  sorry

end bread_calculation_l741_74128


namespace range_of_3a_minus_b_l741_74186

theorem range_of_3a_minus_b (a b : ℝ) 
  (h1 : -1 < a + b ∧ a + b < 3) 
  (h2 : 2 < a - b ∧ a - b < 4) : 
  (∀ x, 3*a - b ≥ x → x ≥ 3) ∧ 
  (∀ y, 3*a - b ≤ y → y ≤ 11) :=
sorry

end range_of_3a_minus_b_l741_74186


namespace skew_lines_and_planes_l741_74125

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (intersect : Line → Plane → Prop)
variable (skew : Line → Line → Prop)

-- Define the given conditions
variable (a b : Line)
variable (α : Plane)

-- Theorem statement
theorem skew_lines_and_planes 
  (h_skew : skew a b)
  (h_parallel : parallel a α) :
  (∃ β : Plane, parallel b β) ∧ 
  (∃ γ : Plane, subset b γ) ∧
  (∃ δ : Set Plane, Set.Infinite δ ∧ ∀ π ∈ δ, intersect b π) :=
sorry

end skew_lines_and_planes_l741_74125


namespace correct_solution_l741_74131

theorem correct_solution (a b : ℚ) : 
  (∀ x y, x = 13 ∧ y = 7 → b * x - 7 * y = 16) →
  (∀ x y, x = 9 ∧ y = 4 → 2 * x + a * y = 6) →
  2 * 6 + a * 2 = 6 ∧ b * 6 - 7 * 2 = 16 :=
by sorry

end correct_solution_l741_74131


namespace max_remainder_div_by_nine_l741_74180

theorem max_remainder_div_by_nine (n : ℕ) (h : n % 9 = 6) : 
  ∀ m : ℕ, m % 9 < 9 ∧ m % 9 ≤ 8 :=
by sorry

end max_remainder_div_by_nine_l741_74180


namespace three_draw_probability_l741_74126

def blue_chips : ℕ := 6
def yellow_chips : ℕ := 4
def total_chips : ℕ := blue_chips + yellow_chips

def prob_different_colors : ℚ := 72 / 625

theorem three_draw_probability :
  let prob_blue : ℚ := blue_chips / total_chips
  let prob_yellow : ℚ := yellow_chips / total_chips
  let prob_diff_first_second : ℚ := prob_blue * prob_yellow + prob_yellow * prob_blue
  prob_diff_first_second * (prob_blue * prob_yellow + prob_yellow * prob_blue) = prob_different_colors :=
by sorry

end three_draw_probability_l741_74126


namespace max_valid_config_l741_74165

/-- Represents a chessboard configuration -/
structure ChessboardConfig where
  white : Nat
  black : Nat

/-- Checks if a configuration is valid for an 8x8 chessboard -/
def is_valid_config (config : ChessboardConfig) : Prop :=
  config.white + config.black ≤ 64 ∧
  config.white = 2 * config.black ∧
  (config.white + config.black) % 8 = 0

/-- The maximum valid configuration -/
def max_config : ChessboardConfig :=
  ⟨32, 16⟩

/-- Theorem: The maximum valid configuration is (32, 16) -/
theorem max_valid_config :
  is_valid_config max_config ∧
  ∀ (c : ChessboardConfig), is_valid_config c → c.white ≤ max_config.white ∧ c.black ≤ max_config.black :=
by sorry


end max_valid_config_l741_74165
