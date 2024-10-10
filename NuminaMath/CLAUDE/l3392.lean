import Mathlib

namespace exists_m_n_for_k_l3392_339264

theorem exists_m_n_for_k (k : ℕ) : 
  (∃ m n : ℕ, m * (m + k) = n * (n + 1)) ↔ k ≠ 2 ∧ k ≠ 3 := by
  sorry

end exists_m_n_for_k_l3392_339264


namespace no_infinite_harmonic_mean_sequence_l3392_339290

theorem no_infinite_harmonic_mean_sequence :
  ¬ ∃ (a : ℕ → ℕ), 
    (∃ i j, a i ≠ a j) ∧ 
    (∀ n : ℕ, n ≥ 2 → a n = (2 * a (n-1) * a (n+1)) / (a (n-1) + a (n+1))) :=
by sorry

end no_infinite_harmonic_mean_sequence_l3392_339290


namespace relationship_abc_l3392_339243

theorem relationship_abc (x : ℝ) (h : x > 2) : (1/3)^3 < Real.log x ∧ Real.log x < x^3 := by
  sorry

end relationship_abc_l3392_339243


namespace sets_theorem_l3392_339221

-- Define the sets A and B
def A : Set ℝ := {x | 0 ≤ 2*x - 1 ∧ 2*x - 1 ≤ 5}
def B (a : ℝ) : Set ℝ := {x | x^2 + a < 0}

-- State the theorem
theorem sets_theorem :
  (∀ x : ℝ, x ∈ A ∩ B (-4) ↔ 1/2 ≤ x ∧ x < 2) ∧
  (∀ x : ℝ, x ∈ A ∪ B (-4) ↔ -2 < x ∧ x ≤ 3) ∧
  (∀ a : ℝ, (B a ∩ (Aᶜ : Set ℝ) = B a) ↔ a ≥ -1/4) :=
sorry

end sets_theorem_l3392_339221


namespace shirt_total_price_l3392_339261

/-- The total price of 25 shirts given the conditions in the problem -/
theorem shirt_total_price : 
  ∀ (shirt_price sweater_price : ℝ),
  75 * sweater_price = 1500 →
  sweater_price = shirt_price + 4 →
  25 * shirt_price = 400 := by
    sorry

end shirt_total_price_l3392_339261


namespace sheep_distribution_l3392_339204

theorem sheep_distribution (A B C D : ℕ) : 
  C = D + 10 ∧ 
  (3 * C) / 4 + A = B + C / 4 + D ∧
  (∃ (x : ℕ), x > 0 ∧ 
    (2 * A) / 3 + (B + A / 3 - (B + A / 3) / 4) + 
    (C + (B + A / 3) / 4 - (C + (B + A / 3) / 4) / 5) + 
    (D + (C + (B + A / 3) / 4) / 5 + x) = 
    4 * ((2 * A) / 3 + (B + A / 3 - (B + A / 3) / 4) + x)) →
  A = 60 ∧ B = 50 ∧ C = 40 ∧ D = 30 := by
sorry

end sheep_distribution_l3392_339204


namespace teddy_pillow_count_l3392_339272

/-- The amount of fluffy foam material used for each pillow in pounds -/
def material_per_pillow : ℝ := 5 - 3

/-- The amount of fluffy foam material Teddy has in tons -/
def total_material_tons : ℝ := 3

/-- The number of pounds in a ton -/
def pounds_per_ton : ℝ := 2000

/-- The theorem stating how many pillows Teddy can make -/
theorem teddy_pillow_count : 
  (total_material_tons * pounds_per_ton) / material_per_pillow = 3000 := by
  sorry

end teddy_pillow_count_l3392_339272


namespace distance_between_intersecting_circles_l3392_339286

/-- The distance between the centers of two intersecting circles -/
def distance_between_centers (a : ℝ) : Set ℝ :=
  {a / 6 * (3 + Real.sqrt 3), a / 6 * (3 - Real.sqrt 3)}

/-- Represents two intersecting circles with a common chord -/
structure IntersectingCircles (a : ℝ) where
  /-- The common chord length -/
  chord_length : ℝ
  /-- The chord is a side of a regular inscribed triangle in one circle -/
  is_triangle_side : Bool
  /-- The chord is a side of an inscribed square in the other circle -/
  is_square_side : Bool
  /-- The chord length is positive -/
  chord_positive : chord_length > 0
  /-- The chord length is equal to a -/
  chord_eq_a : chord_length = a
  /-- One circle has the chord as a triangle side, the other as a square side -/
  different_inscriptions : is_triangle_side ≠ is_square_side

/-- Theorem stating the distance between centers of intersecting circles -/
theorem distance_between_intersecting_circles (a : ℝ) (circles : IntersectingCircles a) :
  ∃ d ∈ distance_between_centers a,
    d = (circles.chord_length / 6) * (3 + Real.sqrt 3) ∨
    d = (circles.chord_length / 6) * (3 - Real.sqrt 3) :=
  sorry

end distance_between_intersecting_circles_l3392_339286


namespace expression_bounds_l3392_339287

theorem expression_bounds : 1 < (3 * Real.sqrt 2 - Real.sqrt 12) * Real.sqrt 3 ∧ 
                            (3 * Real.sqrt 2 - Real.sqrt 12) * Real.sqrt 3 < 2 := by
  sorry

end expression_bounds_l3392_339287


namespace common_chord_triangle_area_l3392_339236

/-- Circle type representing x^2 + y^2 + ax + by + c = 0 --/
structure Circle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Line type representing ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Function to find the common chord of two circles --/
def commonChord (c1 c2 : Circle) : Line := sorry

/-- Function to find the intersection points of a line with the coordinate axes --/
def axisIntersections (l : Line) : ℝ × ℝ := sorry

/-- Function to calculate the area of a triangle given two side lengths --/
def triangleArea (base height : ℝ) : ℝ := sorry

theorem common_chord_triangle_area :
  let c1 : Circle := { a := 0, b := 0, c := -1 }
  let c2 : Circle := { a := -2, b := 2, c := 0 }
  let commonChordLine := commonChord c1 c2
  let (xIntercept, yIntercept) := axisIntersections commonChordLine
  triangleArea xIntercept yIntercept = 1/8 := by sorry

end common_chord_triangle_area_l3392_339236


namespace complex_radical_equality_l3392_339268

theorem complex_radical_equality (a b : ℝ) (ha : a ≥ 0) (hb : b > 0) :
  2.355 * |a^(1/4) - b^(1/6)| = 
  Real.sqrt ((a - 8 * (a^3 * b^2)^(1/6) + 4 * b^(2/3)) / 
             (a^(1/2) - 2 * b^(1/3) + 2 * (a^3 * b^2)^(1/12)) + 3 * b^(1/3)) := by
  sorry

end complex_radical_equality_l3392_339268


namespace square_equation_solutions_l3392_339274

theorem square_equation_solutions (n : ℝ) :
  ∃ (x y : ℝ), x ≠ y ∧
  (n - (2 * n + 1) / 2)^2 = ((n + 1) - (2 * n + 1) / 2)^2 ∧
  (x = n - (2 * n + 1) / 2 ∧ y = (n + 1) - (2 * n + 1) / 2) ∨
  (x = n - (2 * n + 1) / 2 ∧ y = -((n + 1) - (2 * n + 1) / 2)) :=
by sorry

end square_equation_solutions_l3392_339274


namespace base_4_20312_equals_566_l3392_339294

def base_4_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

theorem base_4_20312_equals_566 :
  base_4_to_10 [2, 1, 3, 0, 2] = 566 := by
  sorry

end base_4_20312_equals_566_l3392_339294


namespace sin_shift_l3392_339256

theorem sin_shift (x : ℝ) : Real.sin (2 * x + π / 4) = Real.sin (2 * (x - π / 8)) := by
  sorry

end sin_shift_l3392_339256


namespace probability_all_colors_l3392_339242

/-- The probability of selecting 4 balls of all three colors from 11 balls (3 red, 3 black, 5 white) -/
theorem probability_all_colors (total : ℕ) (red : ℕ) (black : ℕ) (white : ℕ) (select : ℕ) : 
  total = 11 → red = 3 → black = 3 → white = 5 → select = 4 →
  (Nat.choose red 2 * Nat.choose black 1 * Nat.choose white 1 +
   Nat.choose black 2 * Nat.choose red 1 * Nat.choose white 1 +
   Nat.choose white 2 * Nat.choose red 1 * Nat.choose black 1) / 
  Nat.choose total select = 6 / 11 := by
sorry

end probability_all_colors_l3392_339242


namespace number_interval_l3392_339214

theorem number_interval (x : ℝ) (h : x = (1/x) * (-x) + 4) : 2 < x ∧ x ≤ 4 := by
  sorry

end number_interval_l3392_339214


namespace range_of_m_l3392_339250

theorem range_of_m (m : ℝ) : 
  (∃! (n : ℕ), n = 4 ∧ (∀ x : ℤ, (m < x ∧ x < 4) ↔ (0 ≤ x ∧ x < 4))) → 
  (-1 ≤ m ∧ m < 0) :=
sorry

end range_of_m_l3392_339250


namespace paint_replacement_fractions_l3392_339271

/-- Represents the fraction of paint replaced -/
def fraction_replaced (initial_intensity final_intensity new_intensity : ℚ) : ℚ :=
  (initial_intensity - final_intensity) / (initial_intensity - new_intensity)

theorem paint_replacement_fractions :
  let red_initial := (50 : ℚ) / 100
  let blue_initial := (60 : ℚ) / 100
  let red_new := (35 : ℚ) / 100
  let blue_new := (45 : ℚ) / 100
  let red_final := (45 : ℚ) / 100
  let blue_final := (55 : ℚ) / 100
  (fraction_replaced red_initial red_final red_new = 1/3) ∧
  (fraction_replaced blue_initial blue_final blue_new = 1/3) := by
  sorry

end paint_replacement_fractions_l3392_339271


namespace min_reciprocal_sum_min_reciprocal_sum_achievable_l3392_339207

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  (1 / x + 1 / y) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

theorem min_reciprocal_sum_achievable :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 ∧ 1 / x + 1 / y = 3 + 2 * Real.sqrt 2 := by
  sorry

end min_reciprocal_sum_min_reciprocal_sum_achievable_l3392_339207


namespace dist_P_F₂_eq_two_l3392_339299

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 2 = 1

-- Define the foci
variable (F₁ F₂ : ℝ × ℝ)

-- Define a point on the ellipse
variable (P : ℝ × ℝ)

-- Axiom: P is on the ellipse
axiom P_on_ellipse : is_on_ellipse P.1 P.2

-- Axiom: Distance from P to F₁ is 4
axiom dist_P_F₁ : Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) = 4

-- Theorem to prove
theorem dist_P_F₂_eq_two : Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 2 := by
  sorry

end dist_P_F₂_eq_two_l3392_339299


namespace merchant_pricing_strategy_l3392_339249

theorem merchant_pricing_strategy 
  (list_price : ℝ) 
  (purchase_discount : ℝ) 
  (sale_discount : ℝ) 
  (profit_margin : ℝ) 
  (h1 : purchase_discount = 0.3) 
  (h2 : sale_discount = 0.2) 
  (h3 : profit_margin = 0.3) 
  (h4 : list_price > 0) :
  let purchase_price := list_price * (1 - purchase_discount)
  let marked_price := list_price * 1.25
  let selling_price := marked_price * (1 - sale_discount)
  selling_price = purchase_price * (1 + profit_margin) := by
sorry

end merchant_pricing_strategy_l3392_339249


namespace congruence_problem_l3392_339202

theorem congruence_problem (x : ℤ) : 
  (4 * x + 9) % 25 = 3 → (3 * x + 14) % 25 = 22 := by
  sorry

end congruence_problem_l3392_339202


namespace sqrt_450_simplified_l3392_339279

theorem sqrt_450_simplified : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end sqrt_450_simplified_l3392_339279


namespace quadratic_coefficients_theorem_l3392_339270

/-- A quadratic function f(x) = ax^2 + bx + c satisfying specific conditions -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The set of possible coefficient triples (a, b, c) for the quadratic function -/
def PossibleCoefficients : Set (ℝ × ℝ × ℝ) :=
  {(4, -16, 14), (2, -6, 2), (2, -10, 10)}

theorem quadratic_coefficients_theorem (a b c : ℝ) :
  a > 0 ∧
  (∀ x ∈ ({1, 2, 3} : Set ℝ), |QuadraticFunction a b c x| = 2) →
  (a, b, c) ∈ PossibleCoefficients := by
  sorry

end quadratic_coefficients_theorem_l3392_339270


namespace x_seven_y_eight_l3392_339260

theorem x_seven_y_eight (x y : ℚ) (hx : x = 3/4) (hy : y = 4/3) : x^7 * y^8 = 4/3 := by
  sorry

end x_seven_y_eight_l3392_339260


namespace extreme_value_derivative_condition_l3392_339285

open Real

theorem extreme_value_derivative_condition (f : ℝ → ℝ) (x₀ : ℝ) :
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x - x₀| < δ → f x ≤ f x₀ + ε ∨ f x ≥ f x₀ - ε) →
  (deriv f) x₀ = 0 ∧
  ∃ g : ℝ → ℝ, (deriv g) 0 = 0 ∧ ¬(∀ ε > 0, ∃ δ > 0, ∀ x, |x - 0| < δ → g x ≤ g 0 + ε ∨ g x ≥ g 0 - ε) :=
by sorry

end extreme_value_derivative_condition_l3392_339285


namespace inequality_solution_set_l3392_339277

theorem inequality_solution_set (a : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 4| < a) → a > 1 := by
  sorry

end inequality_solution_set_l3392_339277


namespace rectangle_diagonal_l3392_339225

theorem rectangle_diagonal (a b : ℝ) (h_perimeter : 2 * (a + b) = 178) (h_area : a * b = 1848) :
  Real.sqrt (a^2 + b^2) = 65 := by
  sorry

end rectangle_diagonal_l3392_339225


namespace lawrence_walking_days_l3392_339253

/-- Given Lawrence's walking data, prove the number of days he walked. -/
theorem lawrence_walking_days (daily_distance : ℝ) (total_distance : ℝ) 
  (h1 : daily_distance = 4.0)
  (h2 : total_distance = 12) : 
  total_distance / daily_distance = 3 := by
  sorry

end lawrence_walking_days_l3392_339253


namespace rectangular_box_height_l3392_339267

theorem rectangular_box_height (wooden_box_length wooden_box_width wooden_box_height : ℕ)
  (box_length box_width : ℕ) (max_boxes : ℕ) :
  wooden_box_length = 800 ∧ wooden_box_width = 700 ∧ wooden_box_height = 600 ∧
  box_length = 8 ∧ box_width = 7 ∧ max_boxes = 1000000 →
  ∃ (box_height : ℕ), 
    (wooden_box_length * wooden_box_width * wooden_box_height) / max_boxes = 
    box_length * box_width * box_height ∧ box_height = 6 := by
  sorry

end rectangular_box_height_l3392_339267


namespace inequality_equivalence_l3392_339209

theorem inequality_equivalence (x : ℝ) : 
  -1 < (x^2 - 10*x + 9) / (x^2 - 4*x + 5) ∧ (x^2 - 10*x + 9) / (x^2 - 4*x + 5) < 1 ↔ x > 5.3 := by
sorry

end inequality_equivalence_l3392_339209


namespace fraction_sum_of_squares_is_integer_l3392_339295

theorem fraction_sum_of_squares_is_integer (a b : ℚ) 
  (h1 : ∃ k : ℤ, a + b = k) 
  (h2 : ∃ m : ℤ, a * b / (a + b) = m) : 
  ∃ n : ℤ, (a^2 + b^2) / (a + b) = n := by
sorry

end fraction_sum_of_squares_is_integer_l3392_339295


namespace minimum_percentage_owning_95_percent_l3392_339205

/-- Represents the distribution of wealth in a population -/
structure WealthDistribution where
  totalPeople : ℝ
  totalWealth : ℝ
  wealthFunction : ℝ → ℝ
  -- wealthFunction x represents the amount of wealth owned by the top x fraction of people
  wealthMonotone : ∀ x y, 0 ≤ x → x ≤ y → y ≤ 1 → wealthFunction x ≤ wealthFunction y
  wealthBounds : wealthFunction 0 = 0 ∧ wealthFunction 1 = totalWealth

/-- The theorem stating the minimum percentage of people owning 95% of wealth -/
theorem minimum_percentage_owning_95_percent
  (dist : WealthDistribution)
  (h_10_percent : dist.wealthFunction 0.1 ≥ 0.9 * dist.totalWealth) :
  ∃ x : ℝ, x ≤ 0.55 ∧ dist.wealthFunction x ≥ 0.95 * dist.totalWealth := by
  sorry


end minimum_percentage_owning_95_percent_l3392_339205


namespace card_drawing_combinations_l3392_339276

-- Define the number of piles and cards per pile
def num_piles : ℕ := 3
def cards_per_pile : ℕ := 3

-- Define the total number of cards
def total_cards : ℕ := num_piles * cards_per_pile

-- Define the function to calculate the number of ways to draw the cards
def ways_to_draw_cards : ℕ := (Nat.factorial total_cards) / ((Nat.factorial cards_per_pile) ^ num_piles)

-- Theorem statement
theorem card_drawing_combinations :
  ways_to_draw_cards = 1680 :=
sorry

end card_drawing_combinations_l3392_339276


namespace old_supervisor_salary_l3392_339224

/-- Proves that the old supervisor's salary was $870 given the problem conditions -/
theorem old_supervisor_salary
  (num_workers : ℕ)
  (initial_average : ℚ)
  (new_average : ℚ)
  (new_supervisor_salary : ℚ)
  (h_num_workers : num_workers = 8)
  (h_initial_average : initial_average = 430)
  (h_new_average : new_average = 390)
  (h_new_supervisor_salary : new_supervisor_salary = 510)
  : ∃ (old_supervisor_salary : ℚ),
    (num_workers + 1) * initial_average = num_workers * new_average + old_supervisor_salary
    ∧ old_supervisor_salary = 870 :=
by sorry

end old_supervisor_salary_l3392_339224


namespace min_value_on_circle_l3392_339227

theorem min_value_on_circle (x y : ℝ) :
  (x - 1)^2 + (y + 2)^2 = 4 →
  ∃ (S : ℝ), S = 3*x - y ∧ S ≥ -5 - 2*Real.sqrt 10 ∧
  ∀ (S' : ℝ), (∃ (x' y' : ℝ), (x' - 1)^2 + (y' + 2)^2 = 4 ∧ S' = 3*x' - y') →
  S' ≥ -5 - 2*Real.sqrt 10 :=
sorry

end min_value_on_circle_l3392_339227


namespace average_of_xyz_l3392_339210

theorem average_of_xyz (x y z : ℝ) (h : (5 / 4) * (x + y + z) = 20) :
  (x + y + z) / 3 = 16 / 3 := by
  sorry

end average_of_xyz_l3392_339210


namespace class_average_score_l3392_339288

theorem class_average_score (total_students : ℕ) (group1_students : ℕ) (group2_students : ℕ)
  (group1_average : ℚ) (group2_average : ℚ) :
  total_students = group1_students + group2_students →
  group1_students = 10 →
  group2_students = 10 →
  group1_average = 80 →
  group2_average = 60 →
  (group1_students * group1_average + group2_students * group2_average) / total_students = 70 := by
  sorry

end class_average_score_l3392_339288


namespace problem_solution_l3392_339206

theorem problem_solution (a b : ℝ) (h1 : a + b = 4) (h2 : a^2 - 2*a*b + b^2 + 2*a + 2*b = 17) : 
  ((a + 1) * (b + 1) - a * b = 5) ∧ ((a - b)^2 = 9) := by
  sorry

end problem_solution_l3392_339206


namespace x_value_l3392_339217

theorem x_value : ∃ x : ℝ, (2*x - 3*x + 5*x - x = 120) ∧ (x = 40) := by sorry

end x_value_l3392_339217


namespace pond_depth_l3392_339258

/-- Proves that a rectangular pond with given dimensions has a depth of 5 meters -/
theorem pond_depth (length width volume : ℝ) (h1 : length = 20) (h2 : width = 10) (h3 : volume = 1000) :
  volume / (length * width) = 5 := by
  sorry

end pond_depth_l3392_339258


namespace root_relationship_l3392_339281

def f (x : ℝ) : ℝ := x^3 - 7*x^2 + 12*x - 10
def g (x : ℝ) : ℝ := x^3 - 10*x^2 - 2*x + 20

theorem root_relationship :
  ∃ (x₀ : ℝ), f x₀ = 0 ∧ g (2*x₀) = 0 →
  f 5 = 0 ∧ g 10 = 0 := by
sorry

end root_relationship_l3392_339281


namespace floor_equation_solution_l3392_339296

theorem floor_equation_solution (x : ℝ) : 
  ⌊⌊3*x⌋ - 1/2⌋ = ⌊x + 3⌋ ↔ 5/3 ≤ x ∧ x < 7/3 :=
sorry

end floor_equation_solution_l3392_339296


namespace polar_to_rectangular_l3392_339241

/-- Conversion from polar coordinates to rectangular coordinates --/
theorem polar_to_rectangular (r θ : ℝ) :
  r = 6 ∧ θ = π / 3 →
  ∃ x y : ℝ, x = 3 ∧ y = 3 * Real.sqrt 3 ∧
  x = r * Real.cos θ ∧ y = r * Real.sin θ := by
  sorry

end polar_to_rectangular_l3392_339241


namespace average_points_is_27_l3392_339203

/-- Represents a hockey team's record --/
structure TeamRecord where
  wins : ℕ
  ties : ℕ

/-- Calculates the points for a team given their record --/
def calculatePoints (record : TeamRecord) : ℕ :=
  2 * record.wins + record.ties

/-- The number of teams in the playoffs --/
def numTeams : ℕ := 3

/-- The records of the three playoff teams --/
def team1 : TeamRecord := ⟨12, 4⟩
def team2 : TeamRecord := ⟨13, 1⟩
def team3 : TeamRecord := ⟨8, 10⟩

/-- Theorem: The average number of points for the playoff teams is 27 --/
theorem average_points_is_27 : 
  (calculatePoints team1 + calculatePoints team2 + calculatePoints team3) / numTeams = 27 := by
  sorry


end average_points_is_27_l3392_339203


namespace mass_of_X_in_BaX_l3392_339283

/-- The molar mass of barium in g/mol -/
def molar_mass_Ba : ℝ := 137.33

/-- The mass percentage of barium in the compound -/
def mass_percentage_Ba : ℝ := 66.18

/-- The mass of the compound in grams -/
def total_mass : ℝ := 100

theorem mass_of_X_in_BaX : 
  let mass_Ba := total_mass * (mass_percentage_Ba / 100)
  let mass_X := total_mass - mass_Ba
  mass_X = 33.82 := by sorry

end mass_of_X_in_BaX_l3392_339283


namespace marc_total_spending_l3392_339238

/-- The total amount spent by Marc on his purchases -/
def total_spent (model_car_price : ℕ) (paint_bottle_price : ℕ) (paintbrush_price : ℕ) 
  (model_car_quantity : ℕ) (paint_bottle_quantity : ℕ) (paintbrush_quantity : ℕ) : ℕ :=
  model_car_price * model_car_quantity + 
  paint_bottle_price * paint_bottle_quantity + 
  paintbrush_price * paintbrush_quantity

/-- Theorem stating that Marc's total spending is $160 -/
theorem marc_total_spending :
  total_spent 20 10 2 5 5 5 = 160 := by
  sorry

end marc_total_spending_l3392_339238


namespace basketball_team_selection_l3392_339273

def team_size : ℕ := 15
def captain_count : ℕ := 2
def lineup_size : ℕ := 5

theorem basketball_team_selection :
  (team_size.choose captain_count) * 
  (team_size - captain_count).factorial / (team_size - captain_count - lineup_size).factorial = 16201200 := by
  sorry

end basketball_team_selection_l3392_339273


namespace magic_square_d_plus_e_l3392_339201

/-- Represents a 3x3 magic square with some known values and variables -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  sum : ℕ
  sum_eq : sum = 30 + e + 15
         ∧ sum = 10 + c + d
         ∧ sum = a + 25 + b
         ∧ sum = 30 + 10 + a
         ∧ sum = e + c + 25
         ∧ sum = 15 + d + b
         ∧ sum = 30 + c + b
         ∧ sum = a + c + e
         ∧ sum = 15 + 25 + a

theorem magic_square_d_plus_e (sq : MagicSquare) : sq.d + sq.e = 25 := by
  sorry

end magic_square_d_plus_e_l3392_339201


namespace fedya_deposit_l3392_339232

theorem fedya_deposit (n : ℕ) (hn : 0 < n ∧ n < 30) : 
  (∃ (x : ℕ), x * (100 - n) = 847 * 100) → 
  (∃ (x : ℕ), x * (100 - n) = 847 * 100 ∧ x = 1100) :=
by sorry

end fedya_deposit_l3392_339232


namespace sixty_degrees_in_vlecs_l3392_339265

/-- Represents the number of vlecs in a full circle on Venus -/
def full_circle_vlecs : ℕ := 800

/-- Represents the number of degrees in a full circle on Earth -/
def full_circle_degrees : ℕ := 360

/-- Represents the angle in degrees we want to convert to vlecs -/
def angle_degrees : ℕ := 60

/-- Converts an angle from degrees to vlecs -/
def degrees_to_vlecs (degrees : ℕ) : ℕ :=
  (degrees * full_circle_vlecs + full_circle_degrees / 2) / full_circle_degrees

theorem sixty_degrees_in_vlecs :
  degrees_to_vlecs angle_degrees = 133 := by
  sorry

end sixty_degrees_in_vlecs_l3392_339265


namespace simplify_trig_expression_l3392_339211

theorem simplify_trig_expression (α β : ℝ) : 
  Real.sqrt ((1 - Real.sin α * Real.sin β)^2 - (Real.cos α * Real.cos β)^2) = 
  |Real.sin α - Real.sin β| := by
sorry

end simplify_trig_expression_l3392_339211


namespace function_and_inequality_problem_l3392_339218

/-- Given a function f(x) = b * a^x with the specified properties, 
    prove that f(x) = 3 * 2^x and find the maximum value of m. -/
theorem function_and_inequality_problem 
  (f : ℝ → ℝ) 
  (a b : ℝ) 
  (h1 : ∀ x, f x = b * a^x)
  (h2 : a > 0)
  (h3 : a ≠ 1)
  (h4 : f 1 = 6)
  (h5 : f 3 = 24) :
  (∀ x, f x = 3 * 2^x) ∧ 
  (∀ m, (∀ x ≤ 1, (1/a)^x + (1/b)^x - m ≥ 0) ↔ m ≤ 5/6) :=
by sorry

end function_and_inequality_problem_l3392_339218


namespace quadratic_trinomial_transformation_root_l3392_339230

/-- Given a quadratic trinomial ax^2 + bx + c, if we swap b and c, 
    add the result to the original trinomial, and the resulting 
    trinomial has a single root, then that root must be either 0 or -2. -/
theorem quadratic_trinomial_transformation_root (a b c : ℝ) :
  let original := fun x => a * x^2 + b * x + c
  let swapped := fun x => a * x^2 + c * x + b
  let result := fun x => original x + swapped x
  (∃! r, result r = 0) → (result 0 = 0 ∨ result (-2) = 0) :=
by sorry

end quadratic_trinomial_transformation_root_l3392_339230


namespace parallel_vectors_k_value_l3392_339263

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), t ≠ 0 ∧ a.1 = t * b.1 ∧ a.2 = t * b.2

/-- The theorem stating that if (1,k) is parallel to (2,1), then k = 1/2 -/
theorem parallel_vectors_k_value (k : ℝ) :
  are_parallel (1, k) (2, 1) → k = 1/2 := by
  sorry

end parallel_vectors_k_value_l3392_339263


namespace integer_part_sqrt_seven_l3392_339216

theorem integer_part_sqrt_seven : ⌊Real.sqrt 7⌋ = 2 := by sorry

end integer_part_sqrt_seven_l3392_339216


namespace harry_blue_weights_l3392_339233

/-- Represents the weight configuration on a gym bar -/
structure WeightConfig where
  blue_weight : ℕ  -- Weight of each blue weight in pounds
  green_weight : ℕ  -- Weight of each green weight in pounds
  num_green : ℕ  -- Number of green weights
  bar_weight : ℕ  -- Weight of the bar in pounds
  total_weight : ℕ  -- Total weight in pounds

/-- Calculates the number of blue weights given a weight configuration -/
def num_blue_weights (config : WeightConfig) : ℕ :=
  (config.total_weight - config.bar_weight - config.num_green * config.green_weight) / config.blue_weight

/-- Theorem stating that Harry's configuration results in 4 blue weights -/
theorem harry_blue_weights :
  let config : WeightConfig := {
    blue_weight := 2,
    green_weight := 3,
    num_green := 5,
    bar_weight := 2,
    total_weight := 25
  }
  num_blue_weights config = 4 := by sorry

end harry_blue_weights_l3392_339233


namespace rectangles_in_5x5_grid_l3392_339223

/-- The number of dots on each side of the square grid -/
def grid_size : ℕ := 5

/-- The number of different rectangles that can be formed in the grid -/
def num_rectangles : ℕ := (grid_size.choose 2) * (grid_size.choose 2)

/-- Theorem stating that the number of rectangles in a 5x5 grid is 100 -/
theorem rectangles_in_5x5_grid : num_rectangles = 100 := by
  sorry

end rectangles_in_5x5_grid_l3392_339223


namespace least_positive_integer_divisible_by_four_primes_l3392_339234

theorem least_positive_integer_divisible_by_four_primes : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (p₁ p₂ p₃ p₄ : ℕ), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    p₁ ∣ n ∧ p₂ ∣ n ∧ p₃ ∣ n ∧ p₄ ∣ n) ∧
  (∀ m : ℕ, m > 0 → 
    (∃ (q₁ q₂ q₃ q₄ : ℕ), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
      q₁ ∣ m ∧ q₂ ∣ m ∧ q₃ ∣ m ∧ q₄ ∣ m) → 
    n ≤ m) ∧
  n = 210 := by
sorry

end least_positive_integer_divisible_by_four_primes_l3392_339234


namespace triangle_properties_l3392_339246

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The vectors (cos A, cos B) and (a, 2c - b) are parallel -/
def vectors_parallel (t : Triangle) : Prop :=
  (2 * t.c - t.b) * Real.cos t.A = t.a * Real.cos t.B

/-- The theorem to be proved -/
theorem triangle_properties (t : Triangle) (h : vectors_parallel t) :
  t.A = π / 3 ∧ (t.a = 4 → ∃ (max_area : ℝ), max_area = 4 * Real.sqrt 3 ∧
    ∀ (area : ℝ), area = 1 / 2 * t.b * t.c * Real.sin t.A → area ≤ max_area) :=
sorry

end

end triangle_properties_l3392_339246


namespace max_value_2ac_minus_abc_l3392_339262

theorem max_value_2ac_minus_abc : 
  ∀ a b c : ℕ+, 
  a ≤ 7 → b ≤ 6 → c ≤ 4 → 
  (2 * a * c - a * b * c : ℤ) ≤ 28 ∧ 
  ∃ a' b' c' : ℕ+, a' ≤ 7 ∧ b' ≤ 6 ∧ c' ≤ 4 ∧ 2 * a' * c' - a' * b' * c' = 28 :=
sorry

end max_value_2ac_minus_abc_l3392_339262


namespace cistern_wet_surface_area_l3392_339247

/-- Calculates the total wet surface area of a rectangular cistern -/
def total_wet_surface_area (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth) + 2 * (width * depth)

/-- Theorem stating the total wet surface area of a specific cistern -/
theorem cistern_wet_surface_area :
  total_wet_surface_area 8 6 1.85 = 99.8 := by
  sorry

end cistern_wet_surface_area_l3392_339247


namespace complex_fraction_evaluation_l3392_339275

theorem complex_fraction_evaluation : 
  (((7 - 6.35) / 6.5 + 9.9) * (1 / 12.8)) / 
  ((1.2 / 36 + (1 + 1/5) / 0.25 - (1 + 5/6)) * (1 + 1/4)) / 0.125 = 5/3 := by
  sorry

end complex_fraction_evaluation_l3392_339275


namespace football_team_right_handed_players_l3392_339244

theorem football_team_right_handed_players 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (h1 : total_players = 70) 
  (h2 : throwers = 34) 
  (h3 : (total_players - throwers) % 3 = 0) -- Ensures non-throwers can be divided into thirds
  (h4 : throwers ≤ total_players) : -- Ensures there are not more throwers than total players
  throwers + ((total_players - throwers) - (total_players - throwers) / 3) = 58 := by
  sorry

end football_team_right_handed_players_l3392_339244


namespace percentage_problem_l3392_339266

theorem percentage_problem (x : ℝ) (h : 0.3 * x = 120) : 0.4 * x = 160 := by
  sorry

end percentage_problem_l3392_339266


namespace only_one_correct_probability_l3392_339220

theorem only_one_correct_probability (p_a p_b : ℝ) : 
  p_a = 1/5 → p_b = 1/4 → 
  p_a * (1 - p_b) + (1 - p_a) * p_b = 7/20 := by
  sorry

end only_one_correct_probability_l3392_339220


namespace equal_bills_at_80_minutes_l3392_339257

/-- United Telephone's base rate in dollars -/
def united_base : ℚ := 8

/-- United Telephone's per-minute rate in dollars -/
def united_per_minute : ℚ := 1/4

/-- Atlantic Call's base rate in dollars -/
def atlantic_base : ℚ := 12

/-- Atlantic Call's per-minute rate in dollars -/
def atlantic_per_minute : ℚ := 1/5

/-- The number of minutes at which the bills are equal -/
def equal_bill_minutes : ℚ := 80

theorem equal_bills_at_80_minutes :
  united_base + united_per_minute * equal_bill_minutes =
  atlantic_base + atlantic_per_minute * equal_bill_minutes :=
by sorry

end equal_bills_at_80_minutes_l3392_339257


namespace distinct_prime_factors_of_divisor_sum_360_l3392_339228

/-- The sum of positive divisors of a natural number n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The number of distinct prime factors of a natural number n -/
def num_distinct_prime_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The number of distinct prime factors of the sum of positive divisors of 360 is 4 -/
theorem distinct_prime_factors_of_divisor_sum_360 : 
  num_distinct_prime_factors (sum_of_divisors 360) = 4 := by sorry

end distinct_prime_factors_of_divisor_sum_360_l3392_339228


namespace tangent_slope_angle_at_one_l3392_339284

noncomputable def f (x : ℝ) : ℝ := -(Real.sqrt 3 / 3) * x^3 + 2

theorem tangent_slope_angle_at_one :
  let f' : ℝ → ℝ := λ x ↦ -(Real.sqrt 3) * x^2
  let slope : ℝ := f' 1
  let angle_with_neg_x : ℝ := Real.arctan (Real.sqrt 3)
  let angle_with_pos_x : ℝ := π - angle_with_neg_x
  angle_with_pos_x = 2 * π / 3 := by sorry

end tangent_slope_angle_at_one_l3392_339284


namespace lottery_probability_l3392_339231

theorem lottery_probability (p : ℝ) (n : ℕ) (h1 : p = 1 / 10000000) (h2 : n = 5) :
  n * p = 5 / 10000000 := by sorry

end lottery_probability_l3392_339231


namespace percentage_problem_l3392_339245

theorem percentage_problem (x : ℝ) : 
  (16 / 100) * ((40 / 100) * x) = 6 → x = 93.75 := by
  sorry

end percentage_problem_l3392_339245


namespace twenty_cent_items_count_l3392_339240

/-- Represents the number of items at each price point -/
structure ItemCounts where
  cents20 : ℕ
  dollars150 : ℕ
  dollars250 : ℕ

/-- Checks if the given item counts satisfy the problem conditions -/
def satisfiesConditions (counts : ItemCounts) : Prop :=
  counts.cents20 + counts.dollars150 + counts.dollars250 = 50 ∧
  20 * counts.cents20 + 150 * counts.dollars150 + 250 * counts.dollars250 = 5000

/-- Theorem stating that the number of 20-cent items is 31 -/
theorem twenty_cent_items_count :
  ∃ (counts : ItemCounts), satisfiesConditions counts ∧ counts.cents20 = 31 := by
  sorry

end twenty_cent_items_count_l3392_339240


namespace problem_statement_l3392_339237

theorem problem_statement (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : a^2 + b^2 + 4*c^2 = 3) :
  (a + b + 2*c ≤ 3) ∧ (b = 2*c → 1/a + 1/c ≥ 3) := by
  sorry

end problem_statement_l3392_339237


namespace tangent_line_to_ln_curve_l3392_339282

/-- The line y = kx is tangent to the curve y = ln x if and only if k = 1/e -/
theorem tangent_line_to_ln_curve (k : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ k * x = Real.log x ∧ k = 1 / x) ↔ k = 1 / Real.exp 1 := by
  sorry

end tangent_line_to_ln_curve_l3392_339282


namespace part_one_part_two_l3392_339208

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 2) / (x - (3 * a + 1)) < 0}
def B (a : ℝ) : Set ℝ := {x | (x - a^2 - 2) / (x - a) < 0}

-- Part 1
theorem part_one : 
  (Set.univ \ B (1/2)) ∩ A (1/2) = {x : ℝ | 9/4 ≤ x ∧ x < 5/2} := by sorry

-- Part 2
theorem part_two : 
  ∀ a : ℝ, (∀ x : ℝ, x ∈ A a → x ∈ B a) ↔ a ∈ Set.Icc (-1/2) ((3 - Real.sqrt 5) / 2) := by sorry

end part_one_part_two_l3392_339208


namespace binomial_1500_1_l3392_339269

theorem binomial_1500_1 : Nat.choose 1500 1 = 1500 := by
  sorry

end binomial_1500_1_l3392_339269


namespace log_expression_equality_l3392_339213

theorem log_expression_equality : 
  Real.log 3 / Real.log 2 * (Real.log 4 / Real.log 3) + 
  (Real.log 24 / Real.log 2 - Real.log 6 / Real.log 2 + 6) ^ (2/3) = 6 := by
  sorry

end log_expression_equality_l3392_339213


namespace unique_functional_equation_solution_l3392_339254

theorem unique_functional_equation_solution :
  ∃! f : ℝ → ℝ, ∀ x y : ℝ, f x * f y - f (x * y) = x^2 + y :=
by
  -- The proof goes here
  sorry

end unique_functional_equation_solution_l3392_339254


namespace golden_ratio_range_l3392_339235

theorem golden_ratio_range : 
  let φ := (Real.sqrt 5 - 1) / 2
  0.6 < φ ∧ φ < 0.7 := by sorry

end golden_ratio_range_l3392_339235


namespace carnival_sales_proof_l3392_339291

/-- Represents the daily sales of popcorn in dollars -/
def daily_popcorn_sales : ℝ := 50

/-- Represents the daily sales of cotton candy in dollars -/
def daily_cotton_candy_sales : ℝ := 3 * daily_popcorn_sales

/-- Duration of the carnival in days -/
def carnival_duration : ℕ := 5

/-- Total expenses for rent and ingredients in dollars -/
def total_expenses : ℝ := 105

/-- Net earnings after expenses in dollars -/
def net_earnings : ℝ := 895

theorem carnival_sales_proof :
  daily_popcorn_sales * carnival_duration +
  daily_cotton_candy_sales * carnival_duration -
  total_expenses = net_earnings :=
by sorry

end carnival_sales_proof_l3392_339291


namespace tan_alpha_value_l3392_339292

theorem tan_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin (2 * α) = -Real.sin α) : 
  Real.tan α = -Real.sqrt 3 := by
  sorry

end tan_alpha_value_l3392_339292


namespace marble_distribution_l3392_339226

theorem marble_distribution (total_marbles : ℕ) (additional_people : ℕ) : 
  total_marbles = 220 →
  additional_people = 2 →
  ∃ (x : ℕ), 
    (x > 0) ∧ 
    (total_marbles / x - 1 = total_marbles / (x + additional_people)) ∧
    x = 20 :=
by sorry

end marble_distribution_l3392_339226


namespace bear_path_discrepancy_l3392_339289

/-- Represents the circular path of a polar bear on an ice floe -/
structure BearPath where
  diameter_instrument : ℝ  -- Diameter measured by instruments
  diameter_footprint : ℝ   -- Diameter measured from footprints
  is_in_still_water : Prop -- The ice floe is in still water

/-- The difference in measured diameters is due to relative motion -/
theorem bear_path_discrepancy (path : BearPath) 
  (h_instrument : path.diameter_instrument = 8.5)
  (h_footprint : path.diameter_footprint = 9)
  (h_water : path.is_in_still_water) :
  ∃ (relative_motion : ℝ), 
    relative_motion > 0 ∧ 
    path.diameter_footprint - path.diameter_instrument = relative_motion :=
by sorry

end bear_path_discrepancy_l3392_339289


namespace blocks_left_l3392_339280

/-- Given that Randy has 97 blocks initially and uses 25 blocks to build a tower,
    prove that the number of blocks left is 72. -/
theorem blocks_left (initial_blocks : ℕ) (used_blocks : ℕ) (h1 : initial_blocks = 97) (h2 : used_blocks = 25) :
  initial_blocks - used_blocks = 72 := by
  sorry

end blocks_left_l3392_339280


namespace inequalities_hold_l3392_339200

theorem inequalities_hold (a b : ℝ) (h : a > b) :
  (∀ c : ℝ, c ≠ 0 → a / c^2 > b / c^2) ∧
  (∀ c : ℝ, a * |c| ≥ b * |c|) := by
  sorry

end inequalities_hold_l3392_339200


namespace number_equation_proof_l3392_339215

theorem number_equation_proof (n : ℤ) : n - 8 = 5 * 7 + 12 ↔ n = 55 := by
  sorry

end number_equation_proof_l3392_339215


namespace inverse_of_B_squared_l3392_339219

theorem inverse_of_B_squared (B : Matrix (Fin 3) (Fin 3) ℝ) 
  (h : B⁻¹ = ![![2, -3, 0], ![0, -1, 0], ![0, 0, 5]]) : 
  (B^2)⁻¹ = ![![4, -3, 0], ![0, 1, 0], ![0, 0, 25]] := by
  sorry

end inverse_of_B_squared_l3392_339219


namespace distinct_ratios_theorem_l3392_339255

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then
    1/2 - |x - 3/2|
  else
    Real.exp (x - 2) * (-x^2 + 8*x - 12)

theorem distinct_ratios_theorem (n : ℕ) (x : Fin n → ℝ) :
  n ≥ 2 →
  (∀ i : Fin n, x i > 1) →
  (∀ i j : Fin n, i ≠ j → x i ≠ x j) →
  (∀ i j : Fin n, f (x i) / (x i) = f (x j) / (x j)) →
  n ∈ ({2, 3, 4} : Set ℕ) :=
sorry

end distinct_ratios_theorem_l3392_339255


namespace fraction_equality_l3392_339248

theorem fraction_equality (a b c d : ℝ) (h1 : b ≠ c) 
  (h2 : (a * c - b^2) / (a - 2*b + c) = (b * d - c^2) / (b - 2*c + d)) : 
  (a * c - b^2) / (a - 2*b + c) = (a * d - b * c) / (a - b - c + d) ∧ 
  (b * d - c^2) / (b - 2*c + d) = (a * d - b * c) / (a - b - c + d) := by
  sorry

end fraction_equality_l3392_339248


namespace f_is_even_l3392_339212

-- Define the function f(x) = |x| + 1
def f (x : ℝ) : ℝ := |x| + 1

-- State the theorem that f is an even function
theorem f_is_even : ∀ x : ℝ, f x = f (-x) := by
  sorry

end f_is_even_l3392_339212


namespace rabbit_carrots_l3392_339293

theorem rabbit_carrots (rabbit_per_burrow deer_per_burrow : ℕ)
  (rabbit_burrows deer_burrows : ℕ) :
  rabbit_per_burrow = 4 →
  deer_per_burrow = 6 →
  rabbit_per_burrow * rabbit_burrows = deer_per_burrow * deer_burrows →
  rabbit_burrows = deer_burrows + 3 →
  rabbit_per_burrow * rabbit_burrows = 36 :=
by
  sorry

end rabbit_carrots_l3392_339293


namespace stone_order_calculation_l3392_339259

theorem stone_order_calculation (total material_ordered concrete_ordered bricks_ordered stone_ordered : ℝ) :
  total_material_ordered = 0.83 ∧
  concrete_ordered = 0.17 ∧
  bricks_ordered = 0.17 ∧
  total_material_ordered = concrete_ordered + bricks_ordered + stone_ordered →
  stone_ordered = 0.49 := by
sorry

end stone_order_calculation_l3392_339259


namespace total_dolls_count_l3392_339251

/-- The number of dolls in a big box -/
def dolls_per_big_box : ℕ := 7

/-- The number of dolls in a small box -/
def dolls_per_small_box : ℕ := 4

/-- The number of big boxes -/
def num_big_boxes : ℕ := 5

/-- The number of small boxes -/
def num_small_boxes : ℕ := 9

/-- The total number of dolls in all boxes -/
def total_dolls : ℕ := dolls_per_big_box * num_big_boxes + dolls_per_small_box * num_small_boxes

theorem total_dolls_count : total_dolls = 71 := by
  sorry

end total_dolls_count_l3392_339251


namespace dogs_with_tags_l3392_339298

theorem dogs_with_tags (total : ℕ) (with_flea_collars : ℕ) (with_both : ℕ) (with_neither : ℕ) : 
  total = 80 → 
  with_flea_collars = 40 → 
  with_both = 6 → 
  with_neither = 1 → 
  total - with_flea_collars + with_both - with_neither = 45 := by
sorry

end dogs_with_tags_l3392_339298


namespace inequality_range_l3392_339222

theorem inequality_range (a : ℚ) : 
  a^7 < a^5 ∧ a^5 < a^3 ∧ a^3 < a ∧ a < a^2 ∧ a^2 < a^4 ∧ a^4 < a^6 → a < -1 := by
  sorry

end inequality_range_l3392_339222


namespace six_digit_divisible_by_396_l3392_339252

theorem six_digit_divisible_by_396 : ∃ (x y z : ℕ), 
  x < 10 ∧ y < 10 ∧ z < 10 ∧ 
  (243000 + 100 * x + 10 * y + z) % 396 = 0 := by
  sorry

end six_digit_divisible_by_396_l3392_339252


namespace horner_V3_value_l3392_339229

-- Define the polynomial coefficients
def a : List ℤ := [12, 35, -8, 79, 6, 5, 3]

-- Define Horner's method for a single step
def horner_step (v : ℤ) (x : ℤ) (a : ℤ) : ℤ := v * x + a

-- Define the function to compute V_3 using Horner's method
def compute_V3 (coeffs : List ℤ) (x : ℤ) : ℤ :=
  let v0 := coeffs.reverse.head!
  let v1 := horner_step v0 x (coeffs.reverse.tail!.head!)
  let v2 := horner_step v1 x (coeffs.reverse.tail!.tail!.head!)
  horner_step v2 x (coeffs.reverse.tail!.tail!.tail!.head!)

-- State the theorem
theorem horner_V3_value :
  compute_V3 a (-4) = -57 := by sorry

end horner_V3_value_l3392_339229


namespace inverse_A_cubed_l3392_339239

def A_inv : Matrix (Fin 2) (Fin 2) ℤ := !![3, 8; -2, -5]

theorem inverse_A_cubed :
  let A := A_inv⁻¹
  (A^3)⁻¹ = !![5, 0; -66, -137] := by
  sorry

end inverse_A_cubed_l3392_339239


namespace quadratic_inequality_solution_range_l3392_339297

theorem quadratic_inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, x^2 - a*x - a ≤ -3) ↔ (a ≤ -6 ∨ a ≥ 2) :=
by sorry

end quadratic_inequality_solution_range_l3392_339297


namespace quadratic_always_positive_implies_a_in_open_unit_interval_l3392_339278

theorem quadratic_always_positive_implies_a_in_open_unit_interval (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) → 0 < a ∧ a < 1 := by
  sorry

end quadratic_always_positive_implies_a_in_open_unit_interval_l3392_339278
