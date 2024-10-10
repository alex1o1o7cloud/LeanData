import Mathlib

namespace company_fund_problem_l1175_117521

theorem company_fund_problem (n : ℕ) (initial_fund : ℕ) : 
  (50 * n = initial_fund + 5) →
  (45 * n + 95 = initial_fund) →
  initial_fund = 995 := by
  sorry

end company_fund_problem_l1175_117521


namespace logarithm_expression_equals_one_third_l1175_117584

-- Define the logarithm functions
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

noncomputable def log_base (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- State the theorem
theorem logarithm_expression_equals_one_third :
  (lg (1/4) - lg 25) / (2 * log_base 5 10 + log_base 5 (1/4)) + log_base 3 4 * log_base 8 9 = 1/3 := by
  sorry


end logarithm_expression_equals_one_third_l1175_117584


namespace contrapositive_equivalence_l1175_117544

theorem contrapositive_equivalence (a b : ℝ) :
  (((a ≠ 0 ∨ b ≠ 0) → a^2 + b^2 ≠ 0) ↔ 
   (a^2 + b^2 = 0 → a = 0 ∧ b = 0)) :=
by sorry

end contrapositive_equivalence_l1175_117544


namespace floor_equation_solutions_l1175_117564

-- Define the floor function
def floor (x : ℚ) : ℤ := Int.floor x

-- Define the theorem
theorem floor_equation_solutions (a : ℚ) (ha : 0 < a) :
  ∀ x : ℕ+, (floor ((3 * x.val + a) / 4) = 2) ↔ (x.val = 1 ∨ x.val = 2 ∨ x.val = 3) :=
by sorry

end floor_equation_solutions_l1175_117564


namespace max_cells_crossed_cells_crossed_achievable_l1175_117518

/-- Represents a circle on a grid --/
structure GridCircle where
  radius : ℝ
  center : ℝ × ℝ

/-- Represents a cell on a grid --/
structure GridCell where
  x : ℤ
  y : ℤ

/-- Function to count the number of cells crossed by a circle --/
def countCrossedCells (c : GridCircle) : ℕ :=
  sorry

/-- Theorem stating the maximum number of cells crossed by a circle with radius 10 --/
theorem max_cells_crossed (c : GridCircle) (h : c.radius = 10) :
  countCrossedCells c ≤ 80 :=
sorry

/-- Theorem stating that 80 cells can be crossed --/
theorem cells_crossed_achievable :
  ∃ (c : GridCircle), c.radius = 10 ∧ countCrossedCells c = 80 :=
sorry

end max_cells_crossed_cells_crossed_achievable_l1175_117518


namespace fraction_inequality_l1175_117563

theorem fraction_inequality (c x y : ℝ) (h1 : c > x) (h2 : x > y) (h3 : y > 0) :
  x / (c - x) > y / (c - y) := by
  sorry

end fraction_inequality_l1175_117563


namespace tetrahedron_vector_sum_same_sign_l1175_117562

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- A point is inside a tetrahedron if it can be expressed as a convex combination of the vertices -/
def IsInsideTetrahedron (O A B C D : V) : Prop :=
  ∃ (a b c d : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a + b + c + d = 1 ∧
    O = a • A + b • B + c • C + d • D

/-- All real numbers have the same sign if they are all positive or all negative -/
def AllSameSign (α β γ δ : ℝ) : Prop :=
  (α > 0 ∧ β > 0 ∧ γ > 0 ∧ δ > 0) ∨ (α < 0 ∧ β < 0 ∧ γ < 0 ∧ δ < 0)

theorem tetrahedron_vector_sum_same_sign
  (O A B C D : V) (α β γ δ : ℝ)
  (h_inside : IsInsideTetrahedron O A B C D)
  (h_sum : α • (A - O) + β • (B - O) + γ • (C - O) + δ • (D - O) = 0) :
  AllSameSign α β γ δ :=
sorry

end tetrahedron_vector_sum_same_sign_l1175_117562


namespace cube_volumes_from_surface_area_l1175_117520

theorem cube_volumes_from_surface_area :
  ∀ a b c : ℕ,
  (6 * (a^2 + b^2 + c^2) = 564) →
  (a^3 + b^3 + c^3 = 764 ∨ a^3 + b^3 + c^3 = 586) :=
by
  sorry

end cube_volumes_from_surface_area_l1175_117520


namespace cone_volume_from_circle_sector_l1175_117535

/-- The volume of a right circular cone formed by rolling a two-thirds sector of a circle -/
theorem cone_volume_from_circle_sector (r : ℝ) (h : r = 6) :
  let sector_fraction : ℝ := 2 / 3
  let base_radius : ℝ := r * sector_fraction
  let cone_height : ℝ := (r^2 - base_radius^2).sqrt
  let cone_volume : ℝ := (1/3) * π * base_radius^2 * cone_height
  cone_volume = (32/3) * π * Real.sqrt 5 := by
  sorry

end cone_volume_from_circle_sector_l1175_117535


namespace smallest_solution_for_floor_equation_l1175_117581

theorem smallest_solution_for_floor_equation :
  let x : ℝ := 131 / 11
  ∀ y : ℝ, y > 0 → (⌊y^2⌋ : ℝ) - y * ⌊y⌋ = 10 → y ≥ x :=
by sorry

end smallest_solution_for_floor_equation_l1175_117581


namespace rainfall_difference_l1175_117509

/-- The number of Mondays -/
def num_mondays : ℕ := 13

/-- The rainfall on each Monday in centimeters -/
def rain_per_monday : ℝ := 1.75

/-- The number of Tuesdays -/
def num_tuesdays : ℕ := 16

/-- The rainfall on each Tuesday in centimeters -/
def rain_per_tuesday : ℝ := 2.65

/-- The difference in total rainfall between Tuesdays and Mondays -/
theorem rainfall_difference : 
  (num_tuesdays : ℝ) * rain_per_tuesday - (num_mondays : ℝ) * rain_per_monday = 19.65 := by
  sorry

end rainfall_difference_l1175_117509


namespace mixture_problem_l1175_117527

/-- Represents the quantities of milk, water, and juice in a mixture --/
structure Mixture where
  milk : ℝ
  water : ℝ
  juice : ℝ

/-- Calculates the total quantity of a mixture --/
def totalQuantity (m : Mixture) : ℝ := m.milk + m.water + m.juice

/-- Checks if the given quantities form the specified ratio --/
def isRatio (m : Mixture) (r : Mixture) : Prop :=
  m.milk / r.milk = m.water / r.water ∧ m.milk / r.milk = m.juice / r.juice

/-- The main theorem to prove --/
theorem mixture_problem (initial : Mixture) (final : Mixture) : 
  isRatio initial ⟨5, 3, 4⟩ → 
  final.milk = initial.milk ∧ 
  final.water = initial.water + 12 ∧ 
  final.juice = initial.juice + 6 →
  isRatio final ⟨5, 9, 8⟩ →
  totalQuantity initial = 24 := by
  sorry

end mixture_problem_l1175_117527


namespace expansion_coefficient_implies_a_equals_one_l1175_117596

-- Define the binomial expansion coefficient function
def binomial_coefficient (n k : ℕ) : ℕ := sorry

-- Define the function for the coefficient of x^(-3) in the expansion
def coefficient_x_neg_3 (a : ℝ) : ℝ :=
  (binomial_coefficient 7 2) * (2^2) * (a^5)

-- State the theorem
theorem expansion_coefficient_implies_a_equals_one :
  coefficient_x_neg_3 1 = 84 := by sorry

end expansion_coefficient_implies_a_equals_one_l1175_117596


namespace min_value_expression_l1175_117550

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y^2 * z = 72) : 
  x^2 + 4*x*y + 4*y^2 + 2*z^2 ≥ 120 ∧ 
  (x^2 + 4*x*y + 4*y^2 + 2*z^2 = 120 ↔ x = 6 ∧ y = 3 ∧ z = 4) :=
by sorry

end min_value_expression_l1175_117550


namespace two_card_probability_l1175_117594

/-- A standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 4 × Fin 13))
  (size : cards.card = 52)

/-- The probability of selecting two cards from a standard 52-card deck
    that are neither of the same value nor the same suit is 12/17. -/
theorem two_card_probability (d : Deck) : 
  let first_card := d.cards.card
  let second_card := d.cards.card - 1
  let favorable_outcomes := 3 * 12
  (favorable_outcomes : ℚ) / second_card = 12 / 17 := by sorry

end two_card_probability_l1175_117594


namespace real_estate_investment_l1175_117530

theorem real_estate_investment 
  (total_investment : ℝ) 
  (real_estate_ratio : ℝ) 
  (h1 : total_investment = 200000)
  (h2 : real_estate_ratio = 7) : 
  let mutual_funds := total_investment / (real_estate_ratio + 1)
  let real_estate := real_estate_ratio * mutual_funds
  real_estate = 175000 := by
sorry

end real_estate_investment_l1175_117530


namespace vector_decomposition_l1175_117534

def x : Fin 3 → ℝ := ![3, -1, 2]
def p : Fin 3 → ℝ := ![2, 0, 1]
def q : Fin 3 → ℝ := ![1, -1, 1]
def r : Fin 3 → ℝ := ![1, -1, -2]

theorem vector_decomposition :
  x = p + q :=
by sorry

end vector_decomposition_l1175_117534


namespace smallest_integer_solution_l1175_117536

theorem smallest_integer_solution :
  ∀ y : ℤ, (8 - 3 * y ≤ 23) → y ≥ -5 ∧ ∀ z : ℤ, z < -5 → (8 - 3 * z > 23) :=
by sorry

end smallest_integer_solution_l1175_117536


namespace haley_balls_count_l1175_117519

theorem haley_balls_count (balls_per_bag : ℕ) (num_bags : ℕ) (h1 : balls_per_bag = 4) (h2 : num_bags = 9) :
  balls_per_bag * num_bags = 36 := by
  sorry

end haley_balls_count_l1175_117519


namespace inequalities_solution_sets_l1175_117514

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := x^2 > 1
def inequality2 (x : ℝ) : Prop := -x^2 + 2*x + 3 > 0

-- Define the solution sets
def solution_set1 : Set ℝ := {x | x < -1 ∨ x > 1}
def solution_set2 : Set ℝ := {x | -1 < x ∧ x < 3}

-- Theorem statement
theorem inequalities_solution_sets :
  (∀ x : ℝ, inequality1 x ↔ x ∈ solution_set1) ∧
  (∀ x : ℝ, inequality2 x ↔ x ∈ solution_set2) :=
sorry

end inequalities_solution_sets_l1175_117514


namespace triangle_value_l1175_117543

theorem triangle_value (triangle : ℝ) :
  (∀ x : ℝ, (x - 5) * (x + triangle) = x^2 + 2*x - 35) →
  triangle = 7 :=
by
  sorry

end triangle_value_l1175_117543


namespace sum_of_quadratic_solutions_l1175_117589

theorem sum_of_quadratic_solutions :
  let f (x : ℝ) := x^2 - 6*x - 8 - (2*x + 18)
  let solutions := {x : ℝ | f x = 0}
  (∃ x₁ x₂ : ℝ, solutions = {x₁, x₂}) →
  (∃ s : ℝ, ∀ x ∈ solutions, ∃ y ∈ solutions, x + y = s) →
  (∃ s : ℝ, ∀ x ∈ solutions, ∃ y ∈ solutions, x + y = s ∧ s = 8) :=
by sorry

end sum_of_quadratic_solutions_l1175_117589


namespace triangle_properties_l1175_117565

theorem triangle_properties (A B C : Real) (a b c : Real) (D : Real × Real) :
  -- Given conditions
  (c / Real.cos C = (a + b) / (Real.cos A + Real.cos B)) →
  (Real.cos A + Real.cos B ≠ 0) →
  (D.1 = (B + C) / 2) →
  (D.2 = 0) →
  (Real.sqrt ((A - D.1)^2 + D.2^2) = 2) →
  (Real.sqrt ((A - C)^2 + 0^2) = Real.sqrt 7) →
  -- Conclusions
  (C = Real.pi / 3) ∧
  (Real.sqrt ((B - A)^2 + 0^2) = 2 * Real.sqrt 7) :=
by sorry

end triangle_properties_l1175_117565


namespace journey_speeds_correct_l1175_117522

/-- Represents the journey details and speeds -/
structure Journey where
  uphill_distance : ℝ
  downhill_distance : ℝ
  flat_distance : ℝ
  time_ab : ℝ
  time_ba : ℝ
  flat_speed : ℝ
  uphill_speed : ℝ
  downhill_speed : ℝ

/-- Checks if the given speeds satisfy the journey conditions -/
def satisfies_conditions (j : Journey) : Prop :=
  let flat_time := j.flat_distance / j.flat_speed
  let hill_time_ab := j.time_ab - flat_time
  let hill_time_ba := j.time_ba - flat_time
  (j.uphill_distance / j.uphill_speed + j.downhill_distance / j.downhill_speed = hill_time_ab) ∧
  (j.uphill_distance / j.downhill_speed + j.downhill_distance / j.uphill_speed = hill_time_ba)

/-- Theorem stating that the given speeds satisfy the journey conditions -/
theorem journey_speeds_correct (j : Journey) 
  (h1 : j.uphill_distance = 3)
  (h2 : j.downhill_distance = 6)
  (h3 : j.flat_distance = 12)
  (h4 : j.time_ab = 67/60)
  (h5 : j.time_ba = 76/60)
  (h6 : j.flat_speed = 18)
  (h7 : j.uphill_speed = 12)
  (h8 : j.downhill_speed = 30) :
  satisfies_conditions j := by
  sorry

end journey_speeds_correct_l1175_117522


namespace probability_3_1_is_5_over_10_2_l1175_117531

def blue_balls : ℕ := 10
def red_balls : ℕ := 8
def total_balls : ℕ := blue_balls + red_balls
def drawn_balls : ℕ := 4

def probability_3_1 : ℚ :=
  let total_ways := Nat.choose total_balls drawn_balls
  let ways_3blue_1red := Nat.choose blue_balls 3 * Nat.choose red_balls 1
  let ways_1blue_3red := Nat.choose blue_balls 1 * Nat.choose red_balls 3
  (ways_3blue_1red + ways_1blue_3red : ℚ) / total_ways

theorem probability_3_1_is_5_over_10_2 :
  probability_3_1 = 5 / 10.2 := by sorry

end probability_3_1_is_5_over_10_2_l1175_117531


namespace nuts_in_tree_l1175_117577

theorem nuts_in_tree (squirrels : ℕ) (difference : ℕ) (nuts : ℕ) : 
  squirrels = 4 → 
  squirrels = nuts + difference → 
  difference = 2 → 
  nuts = 2 := by sorry

end nuts_in_tree_l1175_117577


namespace ball_drawing_probabilities_l1175_117598

/-- The number of white balls in the bag -/
def white_balls : ℕ := 5

/-- The number of black balls in the bag -/
def black_balls : ℕ := 3

/-- The total number of balls in the bag -/
def total_balls : ℕ := white_balls + black_balls

/-- The number of balls drawn -/
def drawn_balls : ℕ := 4

/-- Probability of drawing either 2 or 3 white balls -/
def prob_two_or_three_white : ℚ := 6/7

/-- Probability of drawing at least one black ball -/
def prob_at_least_one_black : ℚ := 13/14

/-- Theorem stating the probabilities of drawing specific combinations of balls -/
theorem ball_drawing_probabilities :
  (prob_two_or_three_white = 6/7) ∧ (prob_at_least_one_black = 13/14) :=
by sorry

end ball_drawing_probabilities_l1175_117598


namespace problem_solution_l1175_117569

def f (k : ℝ) (x : ℝ) := k - |x - 3|

theorem problem_solution (k : ℝ) (a b c : ℝ) 
  (h1 : Set.Icc (-1 : ℝ) 1 = {x | f k (x + 3) ≥ 0})
  (h2 : a > 0)
  (h3 : b > 0)
  (h4 : c > 0)
  (h5 : 1/a + 1/(2*b) + 1/(3*c) = 1) :
  k = 1 ∧ 1/9 * a + 2/9 * b + 3/9 * c ≥ 1 := by
  sorry

end problem_solution_l1175_117569


namespace triangle_inequality_l1175_117525

/-- Given a triangle with side lengths a, b, c, semiperimeter p, inradius r, 
    and distances from incenter to sides l_a, l_b, l_c, prove that 
    l_a * l_b * l_c ≤ r * p^2 -/
theorem triangle_inequality (a b c p r l_a l_b l_c : ℝ) 
  (h1 : l_a * l_b * l_c ≤ Real.sqrt (p^3 * (p - a) * (p - b) * (p - c)))
  (h2 : ∃ S, S = Real.sqrt (p * (p - a) * (p - b) * (p - c)))
  (h3 : ∃ S, S = r * p) :
  l_a * l_b * l_c ≤ r * p^2 := by
  sorry

end triangle_inequality_l1175_117525


namespace ellipse_perpendicular_points_sum_l1175_117547

/-- Ellipse defined by parametric equations x = 2cos(α) and y = √3sin(α) -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ α : ℝ, p.1 = 2 * Real.cos α ∧ p.2 = Real.sqrt 3 * Real.sin α}

/-- Distance squared from origin to a point -/
def distanceSquared (p : ℝ × ℝ) : ℝ := p.1^2 + p.2^2

/-- Two points are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

theorem ellipse_perpendicular_points_sum (A B : ℝ × ℝ) 
  (hA : A ∈ Ellipse) (hB : B ∈ Ellipse) (hPerp : perpendicular A B) :
  1 / distanceSquared A + 1 / distanceSquared B = 7 / 12 := by
  sorry

end ellipse_perpendicular_points_sum_l1175_117547


namespace other_number_proof_l1175_117556

theorem other_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 12) (h2 : Nat.lcm a b = 396) (h3 : a = 132) : b = 36 := by
  sorry

end other_number_proof_l1175_117556


namespace power_division_l1175_117523

theorem power_division (h : 64 = 8^2) : 8^15 / 64^7 = 8 := by
  sorry

end power_division_l1175_117523


namespace least_four_digit_solution_l1175_117532

theorem least_four_digit_solution (x : ℕ) : x = 1011 ↔ 
  (x ≥ 1000 ∧ x < 10000) ∧
  (∀ y : ℕ, y ≥ 1000 ∧ y < 10000 →
    (5 * y ≡ 15 [ZMOD 20] ∧
     3 * y + 10 ≡ 19 [ZMOD 7] ∧
     -3 * y + 4 ≡ 2 * y [ZMOD 16]) →
    x ≤ y) ∧
  (5 * x ≡ 15 [ZMOD 20]) ∧
  (3 * x + 10 ≡ 19 [ZMOD 7]) ∧
  (-3 * x + 4 ≡ 2 * x [ZMOD 16]) :=
by sorry

end least_four_digit_solution_l1175_117532


namespace nine_integer_chords_l1175_117507

/-- Represents a circle with a given radius and a point P at a distance from the center -/
structure CircleWithPoint where
  radius : ℝ
  distance_to_p : ℝ

/-- Counts the number of integer-length chords containing P in the given circle -/
def count_integer_chords (c : CircleWithPoint) : ℕ :=
  sorry

/-- The main theorem stating that for a circle with radius 20 and P at distance 12,
    there are exactly 9 integer-length chords containing P -/
theorem nine_integer_chords :
  let c := CircleWithPoint.mk 20 12
  count_integer_chords c = 9 := by
  sorry

end nine_integer_chords_l1175_117507


namespace div_decimal_equals_sixty_l1175_117595

theorem div_decimal_equals_sixty : (0.24 : ℚ) / (0.004 : ℚ) = 60 := by
  sorry

end div_decimal_equals_sixty_l1175_117595


namespace expression_simplification_l1175_117549

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3) :
  ((x - 1) / x - (x - 2) / (x + 1)) / ((2 * x - 1) / (x^2 + 2 * x + 1)) = 1 + Real.sqrt 3 / 3 := by
  sorry

end expression_simplification_l1175_117549


namespace race_time_l1175_117548

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  time : ℝ

/-- The race scenario -/
def Race (a b : Runner) : Prop :=
  -- The race is 1000 meters long
  1000 = a.speed * a.time ∧
  -- A beats B by 50 meters
  950 = b.speed * a.time ∧
  -- A beats B by 10 seconds
  b.time = a.time + 10

theorem race_time (a b : Runner) (h : Race a b) : a.time = 200 := by
  sorry

end race_time_l1175_117548


namespace modulus_of_complex_fraction_l1175_117555

theorem modulus_of_complex_fraction : Complex.abs (2 / (1 + Complex.I)) = Real.sqrt 2 := by
  sorry

end modulus_of_complex_fraction_l1175_117555


namespace white_roses_needed_l1175_117578

/-- Calculates the total number of white roses needed for wedding arrangements -/
theorem white_roses_needed (num_bouquets num_table_decorations roses_per_bouquet roses_per_table_decoration : ℕ) : 
  num_bouquets = 5 → 
  num_table_decorations = 7 → 
  roses_per_bouquet = 5 → 
  roses_per_table_decoration = 12 → 
  num_bouquets * roses_per_bouquet + num_table_decorations * roses_per_table_decoration = 109 := by
  sorry

#check white_roses_needed

end white_roses_needed_l1175_117578


namespace nell_initial_ace_cards_l1175_117585

/-- Prove that Nell had 315 Ace cards initially -/
theorem nell_initial_ace_cards 
  (initial_baseball : ℕ)
  (final_ace : ℕ)
  (final_baseball : ℕ)
  (baseball_ace_difference : ℕ)
  (h1 : initial_baseball = 438)
  (h2 : final_ace = 55)
  (h3 : final_baseball = 178)
  (h4 : final_baseball = final_ace + baseball_ace_difference)
  (h5 : baseball_ace_difference = 123) :
  ∃ (initial_ace : ℕ), initial_ace = 315 ∧ 
    initial_ace - final_ace = initial_baseball - final_baseball :=
by
  sorry

end nell_initial_ace_cards_l1175_117585


namespace intersection_points_existence_and_variability_l1175_117568

/-- The parabola equation -/
def parabola (A : ℝ) (x y : ℝ) : Prop := y = A * x^2

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := y^2 + 2 = x^2 + 6 * y

/-- The line equation -/
def line (x y : ℝ) : Prop := y = 2 * x - 1

/-- The intersection point satisfies all three equations -/
def is_intersection_point (A : ℝ) (x y : ℝ) : Prop :=
  parabola A x y ∧ hyperbola x y ∧ line x y

/-- The theorem stating that there is at least one intersection point and the number can vary -/
theorem intersection_points_existence_and_variability :
  ∀ A : ℝ, A > 0 →
  (∃ x y : ℝ, is_intersection_point A x y) ∧
  (∃ A₁ A₂ : ℝ, A₁ > 0 ∧ A₂ > 0 ∧
    (∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ,
      is_intersection_point A₁ x₁ y₁ ∧
      is_intersection_point A₁ x₂ y₂ ∧
      is_intersection_point A₂ x₃ y₃ ∧
      (x₁ ≠ x₂ ∨ y₁ ≠ y₂))) :=
sorry

end intersection_points_existence_and_variability_l1175_117568


namespace gcd_factorial_eight_and_factorial_six_squared_l1175_117500

theorem gcd_factorial_eight_and_factorial_six_squared : 
  Nat.gcd (Nat.factorial 8) ((Nat.factorial 6) ^ 2) = 5040 := by
  sorry

end gcd_factorial_eight_and_factorial_six_squared_l1175_117500


namespace length_of_AB_l1175_117553

/-- Given a line segment AB with points P and Q, prove that AB has length 48 -/
theorem length_of_AB (A B P Q : ℝ) : 
  (P - A) / (B - P) = 1 / 4 →  -- P divides AB in ratio 1:4
  (Q - A) / (B - Q) = 2 / 5 →  -- Q divides AB in ratio 2:5
  Q - P = 3 →                  -- Length of PQ is 3
  B - A = 48 := by             -- Length of AB is 48
sorry


end length_of_AB_l1175_117553


namespace factor_x_squared_minus_196_l1175_117524

theorem factor_x_squared_minus_196 (x : ℝ) : x^2 - 196 = (x - 14) * (x + 14) := by
  sorry

end factor_x_squared_minus_196_l1175_117524


namespace min_value_expression_l1175_117541

open Real

theorem min_value_expression :
  ∃ (m : ℝ), m = 2 * Real.sqrt 2 - 12 ∧
  ∀ (x y : ℝ),
    (Real.sqrt (2 * (1 + Real.cos (2 * x))) - Real.sqrt (3 - Real.sqrt 2) * Real.sin x + 1) *
    (3 + 2 * Real.sqrt (7 - Real.sqrt 2) * Real.cos y - Real.cos (2 * y)) ≥ m :=
by sorry

end min_value_expression_l1175_117541


namespace scalper_discount_l1175_117590

def discount_problem (normal_price : ℝ) (scalper_markup : ℝ) (friend_discount : ℝ) (total_payment : ℝ) : Prop :=
  let website_tickets := 2 * normal_price
  let scalper_tickets := 2 * (normal_price * scalper_markup)
  let friend_ticket := normal_price * friend_discount
  let total_before_discount := website_tickets + scalper_tickets + friend_ticket
  total_before_discount - total_payment = 10

theorem scalper_discount :
  discount_problem 50 2.4 0.6 360 := by
  sorry

end scalper_discount_l1175_117590


namespace leadership_structure_count_correct_l1175_117575

def colony_size : Nat := 35
def num_deputy_governors : Nat := 3
def lieutenants_per_deputy : Nat := 3
def subordinates_per_lieutenant : Nat := 2

def leadership_structure_count : Nat :=
  colony_size * 
  Nat.choose (colony_size - 1) num_deputy_governors *
  Nat.choose (colony_size - 1 - num_deputy_governors) lieutenants_per_deputy *
  Nat.choose (colony_size - 1 - num_deputy_governors - lieutenants_per_deputy) lieutenants_per_deputy *
  Nat.choose (colony_size - 1 - num_deputy_governors - 2 * lieutenants_per_deputy) lieutenants_per_deputy *
  Nat.choose (colony_size - 1 - num_deputy_governors - 3 * lieutenants_per_deputy) 2 *
  Nat.choose (colony_size - 1 - num_deputy_governors - 3 * lieutenants_per_deputy - 2) 2 *
  Nat.choose (colony_size - 1 - num_deputy_governors - 3 * lieutenants_per_deputy - 4) 2 *
  Nat.choose (colony_size - 1 - num_deputy_governors - 3 * lieutenants_per_deputy - 6) 2 *
  Nat.choose (colony_size - 1 - num_deputy_governors - 3 * lieutenants_per_deputy - 8) 2 *
  Nat.choose (colony_size - 1 - num_deputy_governors - 3 * lieutenants_per_deputy - 10) 2 *
  Nat.choose (colony_size - 1 - num_deputy_governors - 3 * lieutenants_per_deputy - 12) 2 *
  Nat.choose (colony_size - 1 - num_deputy_governors - 3 * lieutenants_per_deputy - 14) 2

theorem leadership_structure_count_correct : 
  leadership_structure_count = 35 * 5984 * 4495 * 3276 * 2300 * 120 * 91 * 66 * 45 * 28 * 15 * 6 * 1 :=
by sorry

end leadership_structure_count_correct_l1175_117575


namespace complex_fraction_evaluation_l1175_117528

theorem complex_fraction_evaluation (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + a*b + b^2 = 0) : 
  (a^9 + b^9) / (a + b)^9 = -2 := by
  sorry

end complex_fraction_evaluation_l1175_117528


namespace min_value_fraction_l1175_117551

theorem min_value_fraction (x : ℝ) (h : x > 6) : 
  (∀ y > 6, x^2 / (x - 6) ≤ y^2 / (y - 6)) → x^2 / (x - 6) = 24 := by
  sorry

end min_value_fraction_l1175_117551


namespace hyperbola_standard_equation_l1175_117582

/-- The standard equation of a hyperbola given specific conditions -/
theorem hyperbola_standard_equation :
  ∀ (a b m : ℝ) (P : ℝ × ℝ),
    b > a ∧ a > 0 ∧ m > 0 →
    P.1 = Real.sqrt 5 ∧ P.2 = m →
    P.1^2 / a^2 - P.2^2 / b^2 = 1 →
    P.1 = Real.sqrt (a^2 + b^2) →
    (∃ (A B : ℝ × ℝ),
      (A.2 - P.2) / (A.1 - P.1) = b / a ∧
      (B.2 - P.2) / (B.1 - P.1) = -b / a ∧
      (A.1 - P.1) * (B.2 - P.2) - (A.2 - P.2) * (B.1 - P.1) = 2) →
    ∀ (x y : ℝ), x^2 - y^2 / 4 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1 :=
by sorry

end hyperbola_standard_equation_l1175_117582


namespace divisors_4k_plus_1_ge_4k_minus_1_l1175_117561

/-- The number of divisors of n of the form 4k+1 -/
def divisors_4k_plus_1 (n : ℕ+) : ℕ := sorry

/-- The number of divisors of n of the form 4k-1 -/
def divisors_4k_minus_1 (n : ℕ+) : ℕ := sorry

/-- The difference between the number of divisors of the form 4k+1 and 4k-1 -/
def D (n : ℕ+) : ℤ := (divisors_4k_plus_1 n : ℤ) - (divisors_4k_minus_1 n : ℤ)

theorem divisors_4k_plus_1_ge_4k_minus_1 (n : ℕ+) : D n ≥ 0 := by sorry

end divisors_4k_plus_1_ge_4k_minus_1_l1175_117561


namespace intersection_and_union_of_A_and_B_l1175_117537

def A : Set ℝ := {x | x^2 - 4 = 0}
def B : Set ℝ := {y | ∃ x, y = x^2 - 4}

theorem intersection_and_union_of_A_and_B :
  (A ∩ B = A) ∧ (A ∪ B = B) := by sorry

end intersection_and_union_of_A_and_B_l1175_117537


namespace no_valid_arrangement_l1175_117599

/-- Represents a person at the table -/
structure Person :=
  (id : Nat)

/-- Represents the circular table with 40 people -/
def Table := Fin 40 → Person

/-- Defines the relationship of acquaintance between two people -/
def IsAcquainted (table : Table) (p1 p2 : Person) : Prop := sorry

/-- Counts the number of people between two given positions on the table -/
def PeopleBetween (pos1 pos2 : Fin 40) : Nat := sorry

/-- States that for any two people with an even number between them, there's a common acquaintance -/
def EvenHaveCommonAcquaintance (table : Table) : Prop :=
  ∀ (pos1 pos2 : Fin 40), Even (PeopleBetween pos1 pos2) →
    ∃ (p : Person), IsAcquainted table (table pos1) p ∧ IsAcquainted table (table pos2) p

/-- States that for any two people with an odd number between them, there's no common acquaintance -/
def OddNoCommonAcquaintance (table : Table) : Prop :=
  ∀ (pos1 pos2 : Fin 40), Odd (PeopleBetween pos1 pos2) →
    ∀ (p : Person), ¬(IsAcquainted table (table pos1) p ∧ IsAcquainted table (table pos2) p)

/-- The main theorem stating that no arrangement satisfies both conditions -/
theorem no_valid_arrangement :
  ¬∃ (table : Table), EvenHaveCommonAcquaintance table ∧ OddNoCommonAcquaintance table :=
sorry

end no_valid_arrangement_l1175_117599


namespace output_increase_percentage_l1175_117557

/-- Represents the increase in output per hour when production increases by 80% and working hours decrease by 10% --/
theorem output_increase_percentage (B : ℝ) (H : ℝ) (B_pos : B > 0) (H_pos : H > 0) :
  let original_output := B / H
  let new_output := (1.8 * B) / (0.9 * H)
  (new_output - original_output) / original_output = 1 := by
sorry

end output_increase_percentage_l1175_117557


namespace g_of_six_equals_eleven_l1175_117591

theorem g_of_six_equals_eleven (g : ℝ → ℝ) 
  (h : ∀ x : ℝ, g (4 * x - 2) = x^2 + 2 * x + 3) : 
  g 6 = 11 := by
sorry

end g_of_six_equals_eleven_l1175_117591


namespace count_pairs_eq_three_l1175_117511

/-- The number of distinct ordered pairs of positive integers (m,n) satisfying 1/m + 1/n = 1/3 -/
def count_pairs : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    p.1 > 0 ∧ p.2 > 0 ∧ (1 : ℚ) / p.1 + (1 : ℚ) / p.2 = (1 : ℚ) / 3)
    (Finset.product (Finset.range 100) (Finset.range 100))).card

theorem count_pairs_eq_three : count_pairs = 3 := by
  sorry

end count_pairs_eq_three_l1175_117511


namespace remainder_of_three_to_500_mod_17_l1175_117542

theorem remainder_of_three_to_500_mod_17 : 3^500 % 17 = 13 := by
  sorry

end remainder_of_three_to_500_mod_17_l1175_117542


namespace cyclic_inequality_l1175_117588

def cyclic_sum (f : ℝ → ℝ → ℝ → ℝ) (a b c : ℝ) : ℝ :=
  f a b c + f b c a + f c a b

def cyclic_prod (f : ℝ → ℝ → ℝ → ℝ) (a b c : ℝ) : ℝ :=
  f a b c * f b c a * f c a b

theorem cyclic_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  cyclic_sum (fun x y z => x / (y + z)) a b c ≥ 2 - 4 * cyclic_prod (fun x y z => x / (y + z)) a b c := by
  sorry

end cyclic_inequality_l1175_117588


namespace quadratic_inequality_solution_set_l1175_117510

theorem quadratic_inequality_solution_set (a : ℝ) (h : a < 1) :
  {x : ℝ | x^2 - (a + 1)*x + a < 0} = {x : ℝ | a < x ∧ x < 1} := by
  sorry

end quadratic_inequality_solution_set_l1175_117510


namespace largest_divided_by_smallest_l1175_117546

def numbers : List ℕ := [38, 114, 152, 95]

theorem largest_divided_by_smallest : 
  (List.maximum numbers).get! / (List.minimum numbers).get! = 4 := by
  sorry

end largest_divided_by_smallest_l1175_117546


namespace sally_orange_balloons_l1175_117513

/-- The number of orange balloons Sally has now, given her initial count and the number she lost. -/
def remaining_balloons (initial : ℕ) (lost : ℕ) : ℕ :=
  initial - lost

/-- Theorem stating that Sally now has 7 orange balloons -/
theorem sally_orange_balloons : remaining_balloons 9 2 = 7 := by
  sorry

end sally_orange_balloons_l1175_117513


namespace sci_fi_section_pages_per_book_l1175_117508

/-- Given a library section with a number of books and a total number of pages,
    calculate the number of pages per book. -/
def pages_per_book (num_books : ℕ) (total_pages : ℕ) : ℕ :=
  total_pages / num_books

/-- Theorem stating that in a library section with 8 books and 3824 total pages,
    each book has 478 pages. -/
theorem sci_fi_section_pages_per_book :
  pages_per_book 8 3824 = 478 := by
  sorry

end sci_fi_section_pages_per_book_l1175_117508


namespace laundry_earnings_for_three_days_l1175_117572

def laundry_earnings (charge_per_kilo : ℝ) (day1_kilos : ℝ) : ℝ :=
  let day2_kilos := day1_kilos + 5
  let day3_kilos := 2 * day2_kilos
  charge_per_kilo * (day1_kilos + day2_kilos + day3_kilos)

theorem laundry_earnings_for_three_days :
  laundry_earnings 2 5 = 70 := by
  sorry

end laundry_earnings_for_three_days_l1175_117572


namespace rearrangement_theorem_l1175_117559

/-- The number of terms in the expansion of ((x^(1/2) + x^(1/3))^12) -/
def total_terms : ℕ := 13

/-- The number of terms with positive integer powers of x in the expansion -/
def integer_power_terms : ℕ := 3

/-- The number of terms without positive integer powers of x in the expansion -/
def non_integer_power_terms : ℕ := total_terms - integer_power_terms

/-- The number of ways to rearrange the terms in the expansion of ((x^(1/2) + x^(1/3))^12)
    so that the terms containing positive integer powers of x are not adjacent to each other -/
def rearrangement_count : ℕ := (Nat.factorial non_integer_power_terms) * (Nat.factorial (non_integer_power_terms + 1) / (Nat.factorial (non_integer_power_terms - 2)))

theorem rearrangement_theorem : 
  rearrangement_count = (Nat.factorial 10) * (Nat.factorial 11 / (Nat.factorial 8)) :=
sorry

end rearrangement_theorem_l1175_117559


namespace balloon_arrangement_count_l1175_117515

/-- The number of letters in the word BALLOON -/
def n : ℕ := 7

/-- The number of times 'L' appears in BALLOON -/
def k₁ : ℕ := 2

/-- The number of times 'O' appears in BALLOON -/
def k₂ : ℕ := 2

/-- The number of unique arrangements of letters in BALLOON -/
def balloon_arrangements : ℕ := n.factorial / (k₁.factorial * k₂.factorial)

theorem balloon_arrangement_count : balloon_arrangements = 1260 := by
  sorry

end balloon_arrangement_count_l1175_117515


namespace square_cube_sum_condition_l1175_117566

theorem square_cube_sum_condition (n : ℕ) : 
  (∃ a b : ℤ, n^2 = a + b ∧ n^3 = a^2 + b^2) ↔ n = 0 ∨ n = 1 ∨ n = 2 := by
  sorry

end square_cube_sum_condition_l1175_117566


namespace rectangle_distance_l1175_117593

theorem rectangle_distance (small_perimeter : ℝ) (large_area : ℝ) : 
  small_perimeter = 6 →
  large_area = 12 →
  let small_width := small_perimeter / 6
  let small_length := 2 * small_width
  let large_width := 3 * small_width
  let large_length := 2 * small_length
  large_width * large_length = large_area →
  let horizontal_distance := large_length
  let vertical_distance := large_width - small_width
  Real.sqrt (horizontal_distance ^ 2 + vertical_distance ^ 2) = 3 * Real.sqrt 5 := by
sorry

end rectangle_distance_l1175_117593


namespace circle_tangent_to_y_axis_l1175_117516

/-- A circle in the Euclidean plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle -/
def Circle.equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- A point is on the y-axis if its x-coordinate is 0 -/
def on_y_axis (p : ℝ × ℝ) : Prop := p.1 = 0

/-- A circle is tangent to the y-axis if there exists exactly one point that is both on the circle and on the y-axis -/
def tangent_to_y_axis (c : Circle) : Prop :=
  ∃! p : ℝ × ℝ, c.equation p.1 p.2 ∧ on_y_axis p

/-- The main theorem -/
theorem circle_tangent_to_y_axis :
  let c := Circle.mk (-2, 3) 2
  c.equation x y ↔ (x + 2)^2 + (y - 3)^2 = 4 ∧ tangent_to_y_axis c :=
sorry

end circle_tangent_to_y_axis_l1175_117516


namespace negation_of_proposition_l1175_117567

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, 2 * x^2 + 2 * x + (1/2 : ℝ) < 0) ↔
  (∃ x₀ : ℝ, 2 * x₀^2 + 2 * x₀ + (1/2 : ℝ) ≥ 0) :=
by sorry

end negation_of_proposition_l1175_117567


namespace manfred_average_paycheck_l1175_117586

/-- Calculates the average paycheck amount for Manfred's year, rounded to the nearest dollar. -/
def average_paycheck (total_paychecks : ℕ) (initial_paychecks : ℕ) (initial_amount : ℚ) (increase : ℚ) : ℕ :=
  let remaining_paychecks := total_paychecks - initial_paychecks
  let total_amount := initial_paychecks * initial_amount + remaining_paychecks * (initial_amount + increase)
  let average := total_amount / total_paychecks
  (average + 1/2).floor.toNat

/-- Proves that Manfred's average paycheck for the year, rounded to the nearest dollar, is $765. -/
theorem manfred_average_paycheck :
  average_paycheck 26 6 750 20 = 765 := by
  sorry

end manfred_average_paycheck_l1175_117586


namespace book_length_l1175_117517

theorem book_length (pages_read : ℚ) (pages_remaining : ℚ) (total_pages : ℚ) : 
  pages_read = (2 : ℚ) / 3 * total_pages →
  pages_remaining = (1 : ℚ) / 3 * total_pages →
  pages_read = pages_remaining + 30 →
  total_pages = 90 := by
sorry

end book_length_l1175_117517


namespace not_necessarily_square_l1175_117502

/-- A quadrilateral with four sides and two diagonals -/
structure Quadrilateral :=
  (side1 side2 side3 side4 diagonal1 diagonal2 : ℝ)

/-- Predicate to check if a quadrilateral has 4 equal sides and 2 equal diagonals -/
def has_equal_sides_and_diagonals (q : Quadrilateral) : Prop :=
  q.side1 = q.side2 ∧ q.side2 = q.side3 ∧ q.side3 = q.side4 ∧
  q.diagonal1 = q.diagonal2 ∧
  q.side1 ≠ q.diagonal1

/-- Predicate to check if a quadrilateral is a square -/
def is_square (q : Quadrilateral) : Prop :=
  q.side1 = q.side2 ∧ q.side2 = q.side3 ∧ q.side3 = q.side4 ∧
  q.diagonal1 = q.diagonal2 ∧
  q.diagonal1 = q.side1 * Real.sqrt 2

/-- Theorem stating that a quadrilateral with 4 equal sides and 2 equal diagonals
    is not necessarily a square -/
theorem not_necessarily_square :
  ∃ q : Quadrilateral, has_equal_sides_and_diagonals q ∧ ¬is_square q :=
sorry


end not_necessarily_square_l1175_117502


namespace rent_increase_group_size_l1175_117533

theorem rent_increase_group_size 
  (initial_avg : ℝ) 
  (new_avg : ℝ) 
  (increased_rent : ℝ) 
  (increase_rate : ℝ) :
  initial_avg = 800 →
  new_avg = 870 →
  increased_rent = 1400 →
  increase_rate = 0.2 →
  ∃ n : ℕ, 
    n > 0 ∧
    n * new_avg = (n * initial_avg - increased_rent + increased_rent * (1 + increase_rate)) ∧
    n = 4 :=
by sorry

end rent_increase_group_size_l1175_117533


namespace xy_is_zero_l1175_117597

theorem xy_is_zero (x y : ℝ) (h1 : x + y = 5) (h2 : x^3 + y^3 = 125) : x * y = 0 := by
  sorry

end xy_is_zero_l1175_117597


namespace function_inequality_implies_upper_bound_l1175_117560

theorem function_inequality_implies_upper_bound (a : ℝ) : 
  (∀ x₁ ∈ Set.Icc (1/2 : ℝ) 3, ∃ x₂ ∈ Set.Icc (2 : ℝ) 3, 
    x₁ + 4/x₁ ≥ 2^x₂ + a) → a ≤ 0 := by sorry

end function_inequality_implies_upper_bound_l1175_117560


namespace f_increasing_implies_a_range_a_range_implies_f_increasing_f_increasing_iff_a_range_l1175_117501

/-- The function f(x) defined as 2x^2 - 4(1-a)x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 4 * (1 - a) * x + 1

/-- The theorem stating that if f(x) is increasing on [3,+∞), then a ≥ -2 -/
theorem f_increasing_implies_a_range (a : ℝ) :
  (∀ x y, 3 ≤ x → x < y → f a x < f a y) → a ≥ -2 :=
sorry

/-- The theorem stating that if a ≥ -2, then f(x) is increasing on [3,+∞) -/
theorem a_range_implies_f_increasing (a : ℝ) :
  a ≥ -2 → (∀ x y, 3 ≤ x → x < y → f a x < f a y) :=
sorry

/-- The main theorem stating the equivalence between f(x) being increasing on [3,+∞) and a ≥ -2 -/
theorem f_increasing_iff_a_range (a : ℝ) :
  (∀ x y, 3 ≤ x → x < y → f a x < f a y) ↔ a ≥ -2 :=
sorry

end f_increasing_implies_a_range_a_range_implies_f_increasing_f_increasing_iff_a_range_l1175_117501


namespace wendy_glasses_difference_l1175_117504

/-- The number of glasses polished by Wendy -/
def total_glasses : ℕ := 110

/-- The number of small glasses polished by Wendy -/
def small_glasses : ℕ := 50

/-- The number of large glasses polished by Wendy -/
def large_glasses : ℕ := total_glasses - small_glasses

theorem wendy_glasses_difference :
  large_glasses > small_glasses ∧ large_glasses - small_glasses = 10 := by
  sorry

end wendy_glasses_difference_l1175_117504


namespace bat_costs_60_l1175_117512

/-- The cost of a ball in pounds -/
def ball_cost : ℝ := sorry

/-- The cost of a bat in pounds -/
def bat_cost : ℝ := sorry

/-- The sum of the cost of a ball and a bat is £90 -/
axiom sum_ball_bat : ball_cost + bat_cost = 90

/-- The sum of the cost of three balls and two bats is £210 -/
axiom sum_three_balls_two_bats : 3 * ball_cost + 2 * bat_cost = 210

/-- The cost of a bat is £60 -/
theorem bat_costs_60 : bat_cost = 60 := by sorry

end bat_costs_60_l1175_117512


namespace circle_intersection_distance_ellipse_standard_form_collinearity_condition_l1175_117506

noncomputable section

-- Define the curve C
def C (t : ℝ) (x y : ℝ) : Prop := (4 - t) * x^2 + t * y^2 = 12

-- Part 1
theorem circle_intersection_distance (t : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, C t x₁ y₁ ∧ C t x₂ y₂ ∧ 
   y₁ = x₁ - 2 ∧ y₂ = x₂ - 2 ∧ 
   ∀ x y : ℝ, C t x y → ∃ r : ℝ, x^2 + y^2 = r^2) →
  ∃ A B : ℝ × ℝ, C t A.1 A.2 ∧ C t B.1 B.2 ∧ 
            A.2 = A.1 - 2 ∧ B.2 = B.1 - 2 ∧
            (A.1 - B.1)^2 + (A.2 - B.2)^2 = 16 :=
sorry

-- Part 2
theorem ellipse_standard_form (t : ℝ) :
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a ≠ b ∧
   ∀ x y : ℝ, C t x y ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
  (1 - b^2 / a^2 = 2/3 ∨ 1 - a^2 / b^2 = 2/3) →
  (∀ x y : ℝ, C t x y ↔ x^2 / 12 + y^2 / 4 = 1) ∨
  (∀ x y : ℝ, C t x y ↔ x^2 / 4 + y^2 / 12 = 1) :=
sorry

-- Part 3
theorem collinearity_condition (k m s : ℝ) :
  let P := {p : ℝ × ℝ | p.1^2 + 3 * p.2^2 = 12 ∧ p.2 = k * p.1 + m}
  let Q := {q : ℝ × ℝ | q.1^2 + 3 * q.2^2 = 12 ∧ q.2 = k * q.1 + m ∧ q ≠ (0, 2) ∧ q ≠ (0, -2)}
  let G := {g : ℝ × ℝ | g.2 = s ∧ ∃ q ∈ Q, g.2 - (-2) = (g.1 - 0) * (q.2 - (-2)) / (q.1 - 0)}
  s * m = 4 →
  ∃ p ∈ P, ∃ g ∈ G, (2 - g.2) * (p.1 - g.1) = (p.2 - g.2) * (0 - g.1) :=
sorry

end circle_intersection_distance_ellipse_standard_form_collinearity_condition_l1175_117506


namespace cubic_roots_from_conditions_l1175_117558

theorem cubic_roots_from_conditions (p q r : ℂ) :
  p + q + r = 0 →
  p * q + p * r + q * r = -1 →
  p * q * r = -1 →
  {p, q, r} = {x : ℂ | x^3 - x - 1 = 0} := by sorry

end cubic_roots_from_conditions_l1175_117558


namespace parabola_focus_coordinates_l1175_117545

/-- Given a parabola with equation x = (1/4m)y^2, its focus has coordinates (m, 0) --/
theorem parabola_focus_coordinates (m : ℝ) (h : m ≠ 0) :
  let parabola := {(x, y) : ℝ × ℝ | x = (1 / (4 * m)) * y^2}
  ∃ (focus : ℝ × ℝ), focus ∈ parabola ∧ focus = (m, 0) := by
  sorry

end parabola_focus_coordinates_l1175_117545


namespace smallest_number_with_divisible_digit_sums_l1175_117592

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number satisfies the divisibility condition -/
def satisfiesCondition (n : ℕ) : Prop :=
  17 ∣ sumOfDigits n ∧ 17 ∣ sumOfDigits (n + 1)

theorem smallest_number_with_divisible_digit_sums :
  satisfiesCondition 8899 ∧ ∀ m < 8899, ¬satisfiesCondition m := by sorry

end smallest_number_with_divisible_digit_sums_l1175_117592


namespace compound_interest_rate_l1175_117574

theorem compound_interest_rate (P : ℝ) (h1 : P * (1 + r)^6 = 6000) (h2 : P * (1 + r)^7 = 7500) : r = 0.25 := by
  sorry

end compound_interest_rate_l1175_117574


namespace A_intersect_B_l1175_117538

-- Define the universe U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Define set A
def A : Set Nat := {2, 4, 6, 8, 10}

-- Define complement of A with respect to U
def C_UA : Set Nat := {1, 3, 5, 7, 9}

-- Define complement of B with respect to U
def C_UB : Set Nat := {1, 4, 6, 8, 9}

-- Define set B (derived from its complement)
def B : Set Nat := U \ C_UB

theorem A_intersect_B : A ∩ B = {2} := by
  sorry

end A_intersect_B_l1175_117538


namespace base7_246_equals_132_l1175_117505

/-- Converts a base 7 number to base 10 -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

theorem base7_246_equals_132 :
  base7ToBase10 [6, 4, 2] = 132 := by
  sorry

end base7_246_equals_132_l1175_117505


namespace beijing_olympics_village_area_notation_l1175_117583

/-- Expresses 38.66 million in scientific notation -/
theorem beijing_olympics_village_area_notation :
  (38.66 * 1000000 : ℝ) = 3.866 * (10 ^ 5) := by
  sorry

end beijing_olympics_village_area_notation_l1175_117583


namespace circle_distance_extrema_l1175_117576

-- Define the circle C
def Circle (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

-- Define points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the distance function d
def d (x y : ℝ) : ℝ := 
  (x + 1)^2 + y^2 + (x - 1)^2 + y^2

-- Theorem statement
theorem circle_distance_extrema :
  (∃ x y : ℝ, Circle x y ∧ ∀ x' y' : ℝ, Circle x' y' → d x y ≥ d x' y') ∧
  (∃ x y : ℝ, Circle x y ∧ ∀ x' y' : ℝ, Circle x' y' → d x y ≤ d x' y') ∧
  (∀ x y : ℝ, Circle x y → d x y ≤ 14) ∧
  (∀ x y : ℝ, Circle x y → d x y ≥ 10) :=
by sorry

end circle_distance_extrema_l1175_117576


namespace inequality_solution_set_l1175_117554

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | 12 * x^2 - a * x > a^2}
  (a > 0 → S = {x : ℝ | x < -a/4 ∨ x > a/3}) ∧
  (a = 0 → S = {x : ℝ | x ≠ 0}) ∧
  (a < 0 → S = {x : ℝ | x < a/3 ∨ x > -a/4}) := by
  sorry

end inequality_solution_set_l1175_117554


namespace cube_frame_construction_l1175_117503

/-- Represents a cube -/
structure Cube where
  edge_length : ℝ

/-- Represents a wire -/
structure Wire where
  length : ℝ

/-- Represents the number of cuts needed to construct a cube frame -/
def num_cuts_needed (c : Cube) (w : Wire) : ℕ := sorry

theorem cube_frame_construction (c : Cube) (w : Wire) 
  (h1 : c.edge_length = 10)
  (h2 : w.length = 120) :
  ¬ (num_cuts_needed c w = 0) ∧ (num_cuts_needed c w = 3) := by sorry

end cube_frame_construction_l1175_117503


namespace xy_minus_10_squared_l1175_117587

theorem xy_minus_10_squared (x y : ℝ) (h : (x + 1) * (y + 2) = 8) : 
  (x * y - 10)^2 ≥ 64 ∧ 
  ((x * y - 10)^2 = 64 ↔ (x = 1 ∧ y = 2) ∨ (x = -3 ∧ y = -6)) := by
  sorry

end xy_minus_10_squared_l1175_117587


namespace circle_center_radius_sum_l1175_117573

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 14*y + 73 = -y^2 + 6*x

-- Define the center and radius of the circle
def is_center_radius (a b r : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - a)^2 + (y - b)^2 = r^2

-- Theorem statement
theorem circle_center_radius_sum :
  ∃ a b r : ℝ, is_center_radius a b r ∧ a + b + r = 10 + Real.sqrt 15 :=
sorry

end circle_center_radius_sum_l1175_117573


namespace translation_result_l1175_117571

-- Define the properties of a triangle
structure Triangle :=
  (shape : Type)
  (size : ℝ)
  (orientation : ℝ)

-- Define the translation operation
def translate (t : Triangle) : Triangle := t

-- Define the given shaded triangle
def shaded_triangle : Triangle := sorry

-- Define the options A, B, C, D, E
def option_A : Triangle := sorry
def option_B : Triangle := sorry
def option_C : Triangle := sorry
def option_D : Triangle := sorry
def option_E : Triangle := sorry

-- State the theorem
theorem translation_result :
  ∀ (t : Triangle),
    translate t = t →
    translate shaded_triangle = option_D :=
by sorry

end translation_result_l1175_117571


namespace work_time_solution_l1175_117579

def work_time (T : ℝ) : Prop :=
  let A_alone := T + 8
  let B_alone := T + 4.5
  (1 / A_alone) + (1 / B_alone) = 1 / T

theorem work_time_solution : ∃ T : ℝ, work_time T ∧ T = 6 := by sorry

end work_time_solution_l1175_117579


namespace parallel_vectors_trig_identity_l1175_117526

/-- Given vectors a and b where a is parallel to b, prove that 2sin(α)cos(α) = -4/5 -/
theorem parallel_vectors_trig_identity (α : ℝ) :
  let a : ℝ × ℝ := (Real.cos α, -2)
  let b : ℝ × ℝ := (Real.sin α, 1)
  (∃ (k : ℝ), a = k • b) →
  2 * Real.sin α * Real.cos α = -4/5 := by
sorry

end parallel_vectors_trig_identity_l1175_117526


namespace radio_cost_price_l1175_117552

/-- Proves that the cost price of a radio is 1500 Rs. given the selling price and loss percentage --/
theorem radio_cost_price (selling_price : ℝ) (loss_percentage : ℝ) : 
  selling_price = 1110 → loss_percentage = 26 → 
  ∃ (cost_price : ℝ), cost_price = 1500 ∧ selling_price = cost_price * (1 - loss_percentage / 100) :=
by
  sorry

end radio_cost_price_l1175_117552


namespace range_of_a_l1175_117539

/-- Given that the inequality x^2 + ax - 2 > 0 has a solution in the interval [1,2],
    the range of a is (-1, +∞) -/
theorem range_of_a (a : ℝ) : 
  (∃ x ∈ Set.Icc 1 2, x^2 + a*x - 2 > 0) ↔ a ∈ Set.Ioi (-1) := by
  sorry

end range_of_a_l1175_117539


namespace fred_gave_25_seashells_l1175_117529

/-- The number of seashells Fred initially had -/
def initial_seashells : ℕ := 47

/-- The number of seashells Fred has now -/
def remaining_seashells : ℕ := 22

/-- The number of seashells Fred gave to Jessica -/
def seashells_given : ℕ := initial_seashells - remaining_seashells

theorem fred_gave_25_seashells : seashells_given = 25 := by
  sorry

end fred_gave_25_seashells_l1175_117529


namespace sum_rounded_to_hundredth_l1175_117570

-- Define the repeating decimals
def repeating_decimal_37 : ℚ := 37 + 37 / 99
def repeating_decimal_15 : ℚ := 15 + 15 / 99

-- Define the sum of the repeating decimals
def sum : ℚ := repeating_decimal_37 + repeating_decimal_15

-- Define a function to round to the nearest hundredth
def round_to_hundredth (x : ℚ) : ℚ := 
  ⌊x * 100 + 1/2⌋ / 100

-- Theorem statement
theorem sum_rounded_to_hundredth : 
  round_to_hundredth sum = 52 / 100 := by sorry

end sum_rounded_to_hundredth_l1175_117570


namespace total_bus_ride_distance_l1175_117540

theorem total_bus_ride_distance :
  let vince_ride : ℚ := 5/8
  let zachary_ride : ℚ := 1/2
  let alice_ride : ℚ := 17/20
  let rebecca_ride : ℚ := 2/5
  vince_ride + zachary_ride + alice_ride + rebecca_ride = 19/8
  := by sorry

end total_bus_ride_distance_l1175_117540


namespace total_dots_is_78_l1175_117580

/-- The number of ladybugs Andre caught on Monday -/
def monday_ladybugs : ℕ := 8

/-- The number of ladybugs Andre caught on Tuesday -/
def tuesday_ladybugs : ℕ := 5

/-- The number of dots each ladybug has -/
def dots_per_ladybug : ℕ := 6

/-- The total number of dots on all ladybugs Andre caught -/
def total_dots : ℕ := (monday_ladybugs + tuesday_ladybugs) * dots_per_ladybug

theorem total_dots_is_78 : total_dots = 78 := by
  sorry

end total_dots_is_78_l1175_117580
