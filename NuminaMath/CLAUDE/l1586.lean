import Mathlib

namespace example_polygon_area_l1586_158617

/-- A polygon on a unit grid with specified vertices -/
structure GridPolygon where
  vertices : List (Int × Int)

/-- Calculate the area of a GridPolygon -/
def area (p : GridPolygon) : ℕ :=
  sorry

/-- The specific polygon from the problem -/
def examplePolygon : GridPolygon :=
  { vertices := [(0,0), (20,0), (20,20), (10,20), (10,10), (0,10)] }

/-- Theorem stating that the area of the example polygon is 250 square units -/
theorem example_polygon_area : area examplePolygon = 250 :=
  sorry

end example_polygon_area_l1586_158617


namespace probability_continuous_stripe_is_two_over_81_l1586_158660

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron :=
  (faces : Fin 4 → Face)

/-- Represents a face of the tetrahedron -/
structure Face :=
  (vertices : Fin 3 → Vertex)
  (stripe_start : Vertex)

/-- Represents a vertex of a face -/
inductive Vertex
| A | B | C

/-- Represents a stripe configuration on the tetrahedron -/
def StripeConfiguration := RegularTetrahedron

/-- Checks if a stripe configuration forms a continuous stripe around the tetrahedron -/
def is_continuous_stripe (config : StripeConfiguration) : Prop :=
  sorry

/-- The total number of possible stripe configurations -/
def total_configurations : ℕ := 81

/-- The number of stripe configurations that form a continuous stripe -/
def continuous_stripe_configurations : ℕ := 2

/-- The probability of a continuous stripe encircling the tetrahedron -/
def probability_continuous_stripe : ℚ :=
  continuous_stripe_configurations / total_configurations

theorem probability_continuous_stripe_is_two_over_81 :
  probability_continuous_stripe = 2 / 81 :=
sorry

end probability_continuous_stripe_is_two_over_81_l1586_158660


namespace tan_equality_solution_l1586_158658

open Real

theorem tan_equality_solution (x : ℝ) : 
  0 ≤ x ∧ x ≤ 180 ∧ 
  tan (150 * π / 180 - x * π / 180) = 
    (sin (150 * π / 180) - sin (x * π / 180)) / 
    (cos (150 * π / 180) - cos (x * π / 180)) →
  x = 110 := by
sorry

end tan_equality_solution_l1586_158658


namespace half_of_three_fifths_of_120_l1586_158644

theorem half_of_three_fifths_of_120 : (1/2 : ℚ) * ((3/5 : ℚ) * 120) = 36 := by
  sorry

end half_of_three_fifths_of_120_l1586_158644


namespace derivative_of_log2_l1586_158607

-- Define the base-2 logarithm
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem derivative_of_log2 (x : ℝ) (h : x > 0) :
  deriv log2 x = 1 / (x * Real.log 2) :=
sorry

end derivative_of_log2_l1586_158607


namespace new_york_to_cape_town_duration_l1586_158687

/-- Represents a time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Represents a day of the week -/
inductive Day
| Monday
| Tuesday

/-- Calculates the time difference between two times in hours -/
def timeDifference (t1 t2 : Time) (d1 d2 : Day) : ℕ :=
  sorry

/-- The departure time from London -/
def londonDeparture : Time := { hours := 6, minutes := 0, valid := by simp }

/-- The arrival time in Cape Town -/
def capeTownArrival : Time := { hours := 10, minutes := 0, valid := by simp }

/-- Theorem stating the duration of the New York to Cape Town flight -/
theorem new_york_to_cape_town_duration :
  let londonToNewYorkDuration : ℕ := 18
  let newYorkArrival : Time := 
    { hours := 0, minutes := 0, valid := by simp }
  let newYorkToCapeArrivalDay : Day := Day.Tuesday
  timeDifference newYorkArrival capeTownArrival Day.Tuesday newYorkToCapeArrivalDay = 10 :=
sorry

end new_york_to_cape_town_duration_l1586_158687


namespace divisibility_by_101_l1586_158655

theorem divisibility_by_101 (n : ℕ) : 
  (101 ∣ (10^n - 1)) ↔ (4 ∣ n) :=
sorry

end divisibility_by_101_l1586_158655


namespace triangle_abc_problem_l1586_158621

noncomputable section

open Real

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C (in radians),
    prove that if b = √3, c = 1, and B = π/3, then a = 2, A = π/2, and C = π/6 -/
theorem triangle_abc_problem (a b c A B C : ℝ) : 
  b = sqrt 3 → c = 1 → B = π/3 →
  (sin A) / a = (sin B) / b → (sin B) / b = (sin C) / c →  -- Law of sines
  A + B + C = π →  -- Angle sum in a triangle
  a = 2 ∧ A = π/2 ∧ C = π/6 := by
sorry

end

end triangle_abc_problem_l1586_158621


namespace largest_gold_coins_l1586_158645

theorem largest_gold_coins (n : ℕ) : 
  (∃ k : ℕ, n = 13 * k + 3) ∧ 
  n < 150 → 
  n ≤ 146 ∧ 
  (∃ m : ℕ, m > n ∧ (∃ j : ℕ, m = 13 * j + 3) → m ≥ 150) := by
sorry

end largest_gold_coins_l1586_158645


namespace imaginary_part_of_z_l1586_158631

theorem imaginary_part_of_z : Complex.im ((2 : ℂ) - Complex.I) ^ 2 = -4 := by sorry

end imaginary_part_of_z_l1586_158631


namespace flower_cost_is_nine_l1586_158615

/-- The cost of planting flowers -/
def flower_planting (flower_cost : ℚ) : Prop :=
  let pot_cost : ℚ := flower_cost + 20
  let soil_cost : ℚ := flower_cost - 2
  flower_cost + pot_cost + soil_cost = 45

/-- Theorem: The cost of the flower is $9 -/
theorem flower_cost_is_nine : ∃ (flower_cost : ℚ), flower_cost = 9 ∧ flower_planting flower_cost := by
  sorry

end flower_cost_is_nine_l1586_158615


namespace temp_rise_negative_equals_decrease_l1586_158603

/-- Represents a temperature change in degrees Celsius -/
structure TemperatureChange where
  value : ℝ
  unit : String

/-- Defines a temperature rise -/
def temperature_rise (t : ℝ) : TemperatureChange :=
  { value := t, unit := "°C" }

/-- Defines a temperature decrease -/
def temperature_decrease (t : ℝ) : TemperatureChange :=
  { value := t, unit := "°C" }

/-- Theorem stating that a temperature rise of -2°C is equivalent to a temperature decrease of 2°C -/
theorem temp_rise_negative_equals_decrease :
  temperature_rise (-2) = temperature_decrease 2 := by
  sorry

end temp_rise_negative_equals_decrease_l1586_158603


namespace correlation_coefficient_properties_l1586_158626

-- Define the correlation coefficient
def correlation_coefficient (x y : ℝ → ℝ) : ℝ := sorry

-- Define positive correlation
def positively_correlated (x y : ℝ → ℝ) : Prop := 
  ∀ t₁ t₂, t₁ < t₂ → x t₁ < x t₂ → y t₁ < y t₂

-- Define the strength of linear correlation
def linear_correlation_strength (x y : ℝ → ℝ) : ℝ := sorry

-- Define perfect linear relationship
def perfect_linear_relationship (x y : ℝ → ℝ) : Prop := 
  ∃ a b : ℝ, ∀ t, y t = a * x t + b

-- Theorem statement
theorem correlation_coefficient_properties 
  (x y : ℝ → ℝ) (r : ℝ) (h : r = correlation_coefficient x y) :
  (r > 0 → positively_correlated x y) ∧
  (∀ ε > 0, |r| > 1 - ε → linear_correlation_strength x y > 1 - ε) ∧
  (r = 1 ∨ r = -1 → perfect_linear_relationship x y) := by
  sorry

end correlation_coefficient_properties_l1586_158626


namespace negation_equivalence_l1586_158643

theorem negation_equivalence :
  (¬ ∃ a ∈ Set.Icc (-1 : ℝ) 2, ∃ x : ℝ, a * x^2 + 1 < 0) ↔
  (∀ a ∈ Set.Icc (-1 : ℝ) 2, ∀ x : ℝ, a * x^2 + 1 ≥ 0) :=
by sorry

end negation_equivalence_l1586_158643


namespace randy_blocks_proof_l1586_158653

theorem randy_blocks_proof (house_blocks tower_blocks : ℕ) 
  (h1 : house_blocks = 89)
  (h2 : tower_blocks = 63)
  (h3 : house_blocks - tower_blocks = 26) :
  house_blocks + tower_blocks = 152 := by
  sorry

end randy_blocks_proof_l1586_158653


namespace sum_of_angles_satisfying_equation_l1586_158692

theorem sum_of_angles_satisfying_equation (x : Real) : 
  (0 ≤ x ∧ x ≤ 2 * Real.pi) →
  (Real.sin x ^ 3 + Real.cos x ^ 3 = 1 / Real.cos x + 1 / Real.sin x) →
  ∃ (y : Real), (0 ≤ y ∧ y ≤ 2 * Real.pi) ∧
    (Real.sin y ^ 3 + Real.cos y ^ 3 = 1 / Real.cos y + 1 / Real.sin y) ∧
    (x + y = 3 * Real.pi / 2) :=
by sorry

end sum_of_angles_satisfying_equation_l1586_158692


namespace triangle_area_l1586_158691

/-- Given a triangle ABC with |AB| = 2, |AC| = 3, and AB · AC = -3, 
    prove that the area of triangle ABC is (3√3)/2 -/
theorem triangle_area (A B C : ℝ × ℝ) : 
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  (AB.1^2 + AB.2^2 = 4) →
  (AC.1^2 + AC.2^2 = 9) →
  (AB.1 * AC.1 + AB.2 * AC.2 = -3) →
  (1/2 * Real.sqrt ((AB.1 * AC.2 - AB.2 * AC.1)^2) = (3 * Real.sqrt 3) / 2) :=
by sorry

end triangle_area_l1586_158691


namespace difference_not_1998_l1586_158663

theorem difference_not_1998 (n m : ℕ) : (n^2 + 4*n) - (m^2 + 4*m) ≠ 1998 := by
  sorry

end difference_not_1998_l1586_158663


namespace perimeter_of_figure_C_l1586_158616

/-- Represents the dimensions of a rectangle in terms of small rectangles -/
structure RectDimension where
  width : ℕ
  height : ℕ

/-- Calculates the perimeter of a figure given its dimensions and the size of small rectangles -/
def perimeter (dim : RectDimension) (x y : ℝ) : ℝ :=
  2 * (dim.width * x + dim.height * y)

theorem perimeter_of_figure_C (x y : ℝ) : 
  perimeter ⟨6, 1⟩ x y = 56 →
  perimeter ⟨2, 3⟩ x y = 56 →
  perimeter ⟨1, 3⟩ x y = 40 := by
  sorry

end perimeter_of_figure_C_l1586_158616


namespace canada_population_density_l1586_158668

def population : ℕ := 38005238
def area_sq_miles : ℕ := 3855103
def sq_feet_per_sq_mile : ℕ := 5280 * 5280

def total_sq_feet : ℕ := area_sq_miles * sq_feet_per_sq_mile
def avg_sq_feet_per_person : ℚ := total_sq_feet / population

theorem canada_population_density :
  (2700000 : ℚ) < avg_sq_feet_per_person ∧ avg_sq_feet_per_person < (2900000 : ℚ) :=
sorry

end canada_population_density_l1586_158668


namespace spider_human_leg_ratio_l1586_158674

/-- The number of legs a spider has -/
def spider_legs : ℕ := 8

/-- The number of legs a human has -/
def human_legs : ℕ := 2

/-- The ratio of spider legs to human legs -/
def leg_ratio : ℚ := spider_legs / human_legs

/-- Theorem: The ratio of spider legs to human legs is 4 -/
theorem spider_human_leg_ratio : leg_ratio = 4 := by
  sorry

end spider_human_leg_ratio_l1586_158674


namespace h_equality_l1586_158699

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - x + 1
def g (x : ℝ) : ℝ := -x^2 + x + 1

-- Define the polynomial h
def h (x : ℝ) : ℝ := (x - 1)^2

-- Theorem statement
theorem h_equality (x : ℝ) : h (f x) = h (g x) := by
  sorry

end h_equality_l1586_158699


namespace reciprocal_of_five_l1586_158694

theorem reciprocal_of_five (x : ℚ) : x = 5 → (1 : ℚ) / x = 1 / 5 := by
  sorry

end reciprocal_of_five_l1586_158694


namespace nickel_count_l1586_158622

structure CoinPurse where
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ
  pennies : ℕ

def totalValue (p : CoinPurse) : ℕ :=
  25 * p.quarters + 10 * p.dimes + 5 * p.nickels + p.pennies

def totalCoins (p : CoinPurse) : ℕ :=
  p.quarters + p.dimes + p.nickels + p.pennies

theorem nickel_count (p : CoinPurse) 
  (h1 : totalValue p = 17 * totalCoins p)
  (h2 : totalValue p - 1 = 18 * (totalCoins p - 1)) :
  p.nickels = 2 := by
sorry

end nickel_count_l1586_158622


namespace square_area_proof_l1586_158646

theorem square_area_proof (x : ℚ) :
  (5 * x - 20 : ℚ) = (25 - 2 * x : ℚ) →
  ((5 * x - 20)^2 : ℚ) = 7225 / 49 := by
sorry

end square_area_proof_l1586_158646


namespace inverse_proportion_points_l1586_158624

/-- An inverse proportion function passing through (-6, 1) also passes through (2, -3) -/
theorem inverse_proportion_points :
  ∀ k : ℝ, k / (-6 : ℝ) = 1 → k / (2 : ℝ) = -3 := by
  sorry

end inverse_proportion_points_l1586_158624


namespace jake_first_test_score_l1586_158600

/-- Represents the marks Jake scored in his tests -/
structure JakeScores where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Theorem stating that Jake's first test score was 80 given the conditions -/
theorem jake_first_test_score (scores : JakeScores) :
  (scores.first + scores.second + scores.third + scores.fourth) / 4 = 75 →
  scores.second = scores.first + 10 →
  scores.third = scores.fourth →
  scores.third = 65 →
  scores.first = 80 := by
  sorry

#check jake_first_test_score

end jake_first_test_score_l1586_158600


namespace triangle_probability_l1586_158670

theorem triangle_probability (total_figures : ℕ) (triangle_count : ℕ) 
  (h1 : total_figures = 8) (h2 : triangle_count = 3) :
  (triangle_count : ℚ) / total_figures = 3 / 8 := by
sorry

end triangle_probability_l1586_158670


namespace translate_quadratic_function_l1586_158633

/-- Represents a function f(x) = (x-a)^2 + b --/
def quadratic_function (a b : ℝ) : ℝ → ℝ := λ x ↦ (x - a)^2 + b

/-- Translates a function horizontally by h units and vertically by k units --/
def translate (f : ℝ → ℝ) (h k : ℝ) : ℝ → ℝ := λ x ↦ f (x - h) + k

theorem translate_quadratic_function :
  let f := quadratic_function 2 1
  let g := translate f 1 1
  g = quadratic_function 1 2 := by sorry

end translate_quadratic_function_l1586_158633


namespace unique_triple_solution_l1586_158608

theorem unique_triple_solution : ∃! (a b c : ℕ+), 5^(a.val) + 3^(b.val) - 2^(c.val) = 32 ∧ a = 2 ∧ b = 2 ∧ c = 1 := by
  sorry

end unique_triple_solution_l1586_158608


namespace problem_solution_l1586_158679

theorem problem_solution : 
  (Real.sqrt 48 / Real.sqrt 3 - 2 * Real.sqrt (1/5) * Real.sqrt 30 + Real.sqrt 24 = 4) ∧
  ((2 * Real.sqrt 3 - 1)^2 + (Real.sqrt 3 + 2) * (Real.sqrt 3 - 2) = 12 - 4 * Real.sqrt 3) := by
  sorry

end problem_solution_l1586_158679


namespace no_perfect_square_3000_001_l1586_158602

theorem no_perfect_square_3000_001 (n : ℕ) : ¬ ∃ k : ℤ, (3 * 10^n + 1 : ℤ) = k^2 := by
  sorry

end no_perfect_square_3000_001_l1586_158602


namespace prob_red_or_blue_l1586_158618

-- Define the total number of marbles
def total_marbles : ℕ := 120

-- Define the probabilities of each color
def prob_white : ℚ := 1/5
def prob_green : ℚ := 1/10
def prob_orange : ℚ := 1/6
def prob_violet : ℚ := 1/8

-- Theorem statement
theorem prob_red_or_blue :
  let prob_others := prob_white + prob_green + prob_orange + prob_violet
  (1 - prob_others) = 49/120 := by sorry

end prob_red_or_blue_l1586_158618


namespace two_digit_squares_mod_15_l1586_158681

theorem two_digit_squares_mod_15 : ∃ (S : Finset Nat), (∀ a ∈ S, 10 ≤ a ∧ a < 100 ∧ a^2 % 15 = 1) ∧ S.card = 24 := by
  sorry

end two_digit_squares_mod_15_l1586_158681


namespace solution_set_theorem_l1586_158669

def inequality_system (x : ℝ) : Prop :=
  (4 * x + 1 > 2) ∧ (1 - 2 * x < 7)

theorem solution_set_theorem :
  {x : ℝ | inequality_system x} = {x : ℝ | x > 1/4} := by sorry

end solution_set_theorem_l1586_158669


namespace equation_linearity_implies_m_n_values_l1586_158657

/-- A linear equation in two variables has the form ax + by = c, where a, b, and c are constants -/
def is_linear_in_two_variables (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y + c

/-- The equation 3x^(2m+1) - 2y^(n-1) = 7 -/
def equation (m n : ℕ) (x y : ℝ) : ℝ :=
  3 * x^(2*m+1) - 2 * y^(n-1) - 7

theorem equation_linearity_implies_m_n_values (m n : ℕ) :
  is_linear_in_two_variables (equation m n) → m = 0 ∧ n = 2 := by
  sorry

end equation_linearity_implies_m_n_values_l1586_158657


namespace fifth_root_of_161051_l1586_158641

theorem fifth_root_of_161051 : ∃ n : ℕ, n^5 = 161051 ∧ n = 11 := by
  sorry

end fifth_root_of_161051_l1586_158641


namespace soda_price_ratio_l1586_158697

theorem soda_price_ratio (v p : ℝ) (hv : v > 0) (hp : p > 0) : 
  let brand_x_volume := 1.3 * v
  let brand_x_price := 0.85 * p
  (brand_x_price / brand_x_volume) / (p / v) = 17 / 26 := by
sorry

end soda_price_ratio_l1586_158697


namespace largest_common_term_l1586_158613

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)

def is_in_first_sequence (x : ℤ) : Prop :=
  ∃ n : ℕ, x = arithmetic_sequence 3 8 n

def is_in_second_sequence (x : ℤ) : Prop :=
  ∃ n : ℕ, x = arithmetic_sequence 5 9 n

theorem largest_common_term :
  ∀ x : ℤ, 1 ≤ x ∧ x ≤ 200 ∧ is_in_first_sequence x ∧ is_in_second_sequence x →
  x ≤ 131 ∧ is_in_first_sequence 131 ∧ is_in_second_sequence 131 :=
by sorry

end largest_common_term_l1586_158613


namespace negative_390_same_terminal_side_as_330_l1586_158630

-- Define a function to check if two angles have the same terminal side
def same_terminal_side (a b : Int) : Prop :=
  ∃ k : Int, a ≡ b + 360 * k [ZMOD 360]

-- State the theorem
theorem negative_390_same_terminal_side_as_330 :
  same_terminal_side (-390) 330 := by
  sorry

end negative_390_same_terminal_side_as_330_l1586_158630


namespace expand_and_simplify_l1586_158688

theorem expand_and_simplify (x : ℝ) : (2 * x + 6) * (x + 9) = 2 * x^2 + 24 * x + 54 := by
  sorry

end expand_and_simplify_l1586_158688


namespace ellipse_x_intercept_l1586_158690

-- Define the ellipse
def ellipse (P : ℝ × ℝ) : Prop :=
  let F₁ := (0, 3)
  let F₂ := (4, 0)
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) +
  Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 8

-- Theorem statement
theorem ellipse_x_intercept :
  ellipse (0, 0) →
  ∃ x : ℝ, x ≠ 0 ∧ ellipse (x, 0) ∧ x = 45/8 := by
  sorry

end ellipse_x_intercept_l1586_158690


namespace crayons_given_to_friends_l1586_158604

theorem crayons_given_to_friends 
  (initial_crayons : ℕ)
  (lost_crayons : ℕ)
  (extra_crayons_given : ℕ)
  (h1 : initial_crayons = 589)
  (h2 : lost_crayons = 161)
  (h3 : extra_crayons_given = 410) :
  lost_crayons + extra_crayons_given = 571 :=
by
  sorry

end crayons_given_to_friends_l1586_158604


namespace area_relationship_l1586_158652

/-- A right triangle with sides 18, 24, and 30 -/
structure RightTriangle where
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  is_right : side1^2 + side2^2 = hypotenuse^2
  side1_eq : side1 = 18
  side2_eq : side2 = 24
  hypotenuse_eq : hypotenuse = 30

/-- Areas of non-triangular regions in a circumscribed circle -/
structure CircleAreas where
  D : ℝ
  E : ℝ
  F : ℝ
  F_largest : F ≥ D ∧ F ≥ E

/-- Theorem stating the relationship between areas D, E, F, and the triangle area -/
theorem area_relationship (t : RightTriangle) (areas : CircleAreas) :
  areas.D + areas.E + 216 = areas.F := by
  sorry

end area_relationship_l1586_158652


namespace p_necessary_not_sufficient_for_q_l1586_158675

theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, x > 3 → x > 2) ∧
  (∃ x : ℝ, x > 2 ∧ x ≤ 3) := by
  sorry

end p_necessary_not_sufficient_for_q_l1586_158675


namespace function_inequality_l1586_158642

theorem function_inequality (f : ℝ → ℝ) (h1 : Differentiable ℝ f) (h2 : ∀ x, deriv f x > 1) :
  f 3 > f 1 + 2 := by
  sorry

end function_inequality_l1586_158642


namespace sector_arc_length_l1586_158605

/-- Given a circular sector with central angle 2π/3 and area 25π/3, 
    its arc length is 10π/3 -/
theorem sector_arc_length (α : Real) (S : Real) (l : Real) :
  α = 2 * π / 3 →
  S = 25 * π / 3 →
  l = 10 * π / 3 :=
by
  sorry


end sector_arc_length_l1586_158605


namespace juice_cost_proof_l1586_158629

/-- The cost of 5 cans of juice during a store's anniversary sale -/
def cost_of_five_juice_cans : ℝ := by sorry

theorem juice_cost_proof (original_ice_cream_price : ℝ) 
                         (ice_cream_discount : ℝ) 
                         (total_cost : ℝ) : 
  original_ice_cream_price = 12 →
  ice_cream_discount = 2 →
  total_cost = 24 →
  2 * (original_ice_cream_price - ice_cream_discount) + 2 * cost_of_five_juice_cans = total_cost →
  cost_of_five_juice_cans = 2 := by sorry

end juice_cost_proof_l1586_158629


namespace max_sum_on_parabola_l1586_158609

theorem max_sum_on_parabola :
  ∃ (max : ℝ), max = 13/4 ∧ 
  ∀ (m n : ℝ), n = -m^2 + 3 → m + n ≤ max :=
by sorry

end max_sum_on_parabola_l1586_158609


namespace simple_interest_principal_calculation_l1586_158664

theorem simple_interest_principal_calculation
  (rate : ℝ) (interest : ℝ) (time : ℝ) (principal : ℝ)
  (h_rate : rate = 4.166666666666667)
  (h_interest : interest = 130)
  (h_time : time = 4)
  (h_formula : interest = principal * rate * time / 100) :
  principal = 780 := by
sorry

end simple_interest_principal_calculation_l1586_158664


namespace no_valid_coloring_l1586_158611

/-- A coloring of a 5x5 board using 4 colors -/
def Coloring := Fin 5 → Fin 5 → Fin 4

/-- Predicate to check if a coloring satisfies the constraint -/
def ValidColoring (c : Coloring) : Prop :=
  ∀ (r1 r2 c1 c2 : Fin 5), r1 ≠ r2 → c1 ≠ c2 →
    (Finset.card {c r1 c1, c r1 c2, c r2 c1, c r2 c2} ≥ 3)

/-- Theorem stating that no valid coloring exists -/
theorem no_valid_coloring : ¬ ∃ (c : Coloring), ValidColoring c := by
  sorry

end no_valid_coloring_l1586_158611


namespace congruent_count_l1586_158698

theorem congruent_count : Nat.card {n : ℕ | 0 < n ∧ n < 500 ∧ n % 7 = 3} = 71 := by
  sorry

end congruent_count_l1586_158698


namespace four_part_cut_possible_five_triangular_part_cut_possible_l1586_158620

-- Define the original figure
def original_figure : Set (ℝ × ℝ) :=
  sorry

-- Define the area of the original figure
def original_area : ℝ := 64

-- Define a square with area 64
def target_square : Set (ℝ × ℝ) :=
  sorry

-- Define a function that represents cutting the figure into parts
def cut (figure : Set (ℝ × ℝ)) (n : ℕ) : List (Set (ℝ × ℝ)) :=
  sorry

-- Define a function that represents assembling parts into a new figure
def assemble (parts : List (Set (ℝ × ℝ))) : Set (ℝ × ℝ) :=
  sorry

-- Define a predicate to check if a set is triangular
def is_triangular (s : Set (ℝ × ℝ)) : Prop :=
  sorry

-- Theorem for part a
theorem four_part_cut_possible :
  ∃ (parts : List (Set (ℝ × ℝ))),
    parts.length ≤ 4 ∧
    (∀ p ∈ parts, p ⊆ original_figure) ∧
    assemble parts = target_square :=
  sorry

-- Theorem for part b
theorem five_triangular_part_cut_possible :
  ∃ (parts : List (Set (ℝ × ℝ))),
    parts.length ≤ 5 ∧
    (∀ p ∈ parts, p ⊆ original_figure ∧ is_triangular p) ∧
    assemble parts = target_square :=
  sorry

end four_part_cut_possible_five_triangular_part_cut_possible_l1586_158620


namespace student_assignment_l1586_158684

/-- The number of ways to assign n indistinguishable objects to k distinct containers,
    with each container receiving at least one object. -/
def assign_objects (n k : ℕ) : ℕ :=
  sorry

/-- The number of ways to assign 5 students to 3 towns -/
theorem student_assignment : assign_objects 5 3 = 150 := by
  sorry

end student_assignment_l1586_158684


namespace moon_speed_km_per_second_l1586_158685

-- Define the speed of the moon in kilometers per hour
def moon_speed_km_per_hour : ℝ := 3672

-- Define the number of seconds in an hour
def seconds_per_hour : ℝ := 3600

-- Theorem statement
theorem moon_speed_km_per_second :
  moon_speed_km_per_hour / seconds_per_hour = 1.02 := by
  sorry

end moon_speed_km_per_second_l1586_158685


namespace white_square_area_l1586_158696

/-- Given a cube with edge length 12 feet and 432 square feet of green paint used equally on all faces as a border, the area of the white square centered on each face is 72 square feet. -/
theorem white_square_area (cube_edge : ℝ) (green_paint_area : ℝ) (white_square_area : ℝ) : 
  cube_edge = 12 →
  green_paint_area = 432 →
  white_square_area = 72 →
  white_square_area = cube_edge^2 - green_paint_area / 6 :=
by sorry

end white_square_area_l1586_158696


namespace correct_subtraction_l1586_158659

theorem correct_subtraction (x : ℤ) : x - 32 = 25 → x - 23 = 34 := by sorry

end correct_subtraction_l1586_158659


namespace garden_dimensions_possible_longest_side_l1586_158671

/-- Represents a rectangular garden with one side along a wall -/
structure RectangularGarden where
  side1 : ℕ
  side2 : ℕ
  side3 : ℕ
  total_fence : ℕ
  h_fence : side1 + side2 + side3 = total_fence

/-- The total fence length is 140 meters -/
def total_fence : ℕ := 140

theorem garden_dimensions (g : RectangularGarden) 
  (h1 : g.side1 = 40) (h2 : g.side2 = 40) (h_total : g.total_fence = total_fence) : 
  g.side3 = 60 := by
  sorry

theorem possible_longest_side (g : RectangularGarden) (h_total : g.total_fence = total_fence) :
  (∃ (g' : RectangularGarden), g'.side1 = 65 ∨ g'.side2 = 65 ∨ g'.side3 = 65) ∧
  (¬∃ (g' : RectangularGarden), g'.side1 = 85 ∨ g'.side2 = 85 ∨ g'.side3 = 85) := by
  sorry

end garden_dimensions_possible_longest_side_l1586_158671


namespace floor_equation_solution_l1586_158648

theorem floor_equation_solution (n : ℤ) : 
  (⌊n^2 / 4⌋ - ⌊n / 2⌋^2 = 3) ↔ (n = 7) :=
by sorry

end floor_equation_solution_l1586_158648


namespace gcd_lcm_sum_l1586_158693

theorem gcd_lcm_sum : Nat.gcd 45 75 + Nat.lcm 40 10 = 55 := by
  sorry

end gcd_lcm_sum_l1586_158693


namespace ellipse_intersection_ratio_l1586_158625

/-- Given an ellipse mx^2 + ny^2 = 1 intersecting with a line y = -x + 1,
    if a line through the origin and the midpoint of the intersection points
    has slope √2/2, then n/m = √2 -/
theorem ellipse_intersection_ratio (m n : ℝ) (h_pos : m > 0 ∧ n > 0) :
  (∃ A B : ℝ × ℝ,
    (m * A.1^2 + n * A.2^2 = 1) ∧
    (m * B.1^2 + n * B.2^2 = 1) ∧
    (A.2 = -A.1 + 1) ∧
    (B.2 = -B.1 + 1) ∧
    ((A.2 + B.2) / (A.1 + B.1) = Real.sqrt 2 / 2)) →
  n / m = Real.sqrt 2 := by
sorry

end ellipse_intersection_ratio_l1586_158625


namespace partner_b_contribution_l1586_158683

/-- Represents the capital contribution of a business partner -/
structure Capital where
  amount : ℝ
  duration : ℕ

/-- Calculates the adjusted capital contribution -/
def adjustedCapital (c : Capital) : ℝ := c.amount * c.duration

/-- Represents the profit-sharing ratio between two partners -/
structure ProfitRatio where
  partner1 : ℕ
  partner2 : ℕ

theorem partner_b_contribution 
  (a : Capital) 
  (b : Capital)
  (ratio : ProfitRatio)
  (h1 : a.amount = 3500)
  (h2 : a.duration = 12)
  (h3 : b.duration = 7)
  (h4 : ratio.partner1 = 2)
  (h5 : ratio.partner2 = 3)
  (h6 : (adjustedCapital a) / (adjustedCapital b) = ratio.partner1 / ratio.partner2) :
  b.amount = 4500 := by
  sorry

end partner_b_contribution_l1586_158683


namespace olya_candies_l1586_158632

theorem olya_candies 
  (total : ℕ)
  (pasha masha tolya olya : ℕ)
  (h_total : pasha + masha + tolya + olya = total)
  (h_total_val : total = 88)
  (h_masha_tolya : masha + tolya = 57)
  (h_pasha_most : pasha > masha ∧ pasha > tolya ∧ pasha > olya)
  (h_at_least_one : pasha ≥ 1 ∧ masha ≥ 1 ∧ tolya ≥ 1 ∧ olya ≥ 1) :
  olya = 1 := by
sorry

end olya_candies_l1586_158632


namespace final_card_expectation_l1586_158635

/-- Represents a deck of cards -/
def Deck := List Nat

/-- The process of drawing two cards, discarding one, and reinserting the other -/
def drawDiscardReinsert (d : Deck) : Deck :=
  sorry

/-- The expected value of the label of the remaining card after the process -/
def expectedValue (d : Deck) : Rat :=
  sorry

/-- Theorem stating the expected value of the final card in a 100-card deck -/
theorem final_card_expectation :
  let initialDeck : Deck := List.range 100
  expectedValue initialDeck = 467 / 8 := by
  sorry

end final_card_expectation_l1586_158635


namespace luther_clothing_line_l1586_158695

/-- The number of silk pieces in Luther's clothing line -/
def silk_pieces : ℕ := 7

/-- The number of cashmere pieces in Luther's clothing line -/
def cashmere_pieces : ℕ := silk_pieces / 2

/-- The number of blended pieces using both cashmere and silk -/
def blended_pieces : ℕ := 2

/-- The total number of pieces in Luther's clothing line -/
def total_pieces : ℕ := 13

theorem luther_clothing_line :
  silk_pieces + cashmere_pieces + blended_pieces = total_pieces ∧
  cashmere_pieces = silk_pieces / 2 ∧
  silk_pieces = 7 := by sorry

end luther_clothing_line_l1586_158695


namespace minimum_packages_l1586_158665

theorem minimum_packages (p : ℕ) : p > 0 → (∃ N : ℕ, N = 19 * p ∧ N % 7 = 4 ∧ N % 11 = 1) → p ≥ 40 :=
by sorry

end minimum_packages_l1586_158665


namespace circle_equation_l1586_158639

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def passes_through (c : Circle) (p : ℝ × ℝ) : Prop :=
  (c.center.1 - p.1)^2 + (c.center.2 - p.2)^2 = c.radius^2

def center_on_line (c : Circle) : Prop :=
  c.center.2 = 2 * c.center.1

def tangent_to_line (c : Circle) : Prop :=
  c.radius = |2 * c.center.1 - c.center.2 + 5| / Real.sqrt 5

-- Theorem statement
theorem circle_equation (c : Circle) :
  passes_through c (3, 2) ∧
  center_on_line c ∧
  tangent_to_line c →
  ((λ (x y : ℝ) => (x - 2)^2 + (y - 4)^2 = 5) c.center.1 c.center.2) ∨
  ((λ (x y : ℝ) => (x - 4/5)^2 + (y - 8/5)^2 = 5) c.center.1 c.center.2) :=
by sorry

end circle_equation_l1586_158639


namespace arithmetic_mean_problem_l1586_158650

theorem arithmetic_mean_problem (x y : ℝ) : 
  (8 + x + 21 + y + 14 + 11) / 6 = 15 → x + y = 36 := by
  sorry

end arithmetic_mean_problem_l1586_158650


namespace product_congruence_l1586_158682

theorem product_congruence : 198 * 963 ≡ 24 [ZMOD 50] := by sorry

end product_congruence_l1586_158682


namespace tan_sum_diff_implies_sin_2alpha_cos_2beta_l1586_158627

theorem tan_sum_diff_implies_sin_2alpha_cos_2beta
  (α β : ℝ)
  (h1 : Real.tan (α + β) = 1)
  (h2 : Real.tan (α - β) = 2) :
  (Real.sin (2 * α)) / (Real.cos (2 * β)) = 1 := by
  sorry

end tan_sum_diff_implies_sin_2alpha_cos_2beta_l1586_158627


namespace min_value_2x_3y_l1586_158610

theorem min_value_2x_3y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 3*y + 3*x*y = 6) :
  ∀ z : ℝ, (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2*a + 3*b + 3*a*b = 6 ∧ 2*a + 3*b = z) → z ≥ 4 :=
by sorry

end min_value_2x_3y_l1586_158610


namespace intercepts_congruence_l1586_158678

/-- Proof of x-intercept and y-intercept properties for the congruence 6x ≡ 5y - 1 (mod 28) --/
theorem intercepts_congruence :
  ∃ (x₀ y₀ : ℕ),
    x₀ < 28 ∧ y₀ < 28 ∧
    (6 * x₀) % 28 = 27 ∧
    (5 * y₀) % 28 = 1 ∧
    x₀ + y₀ = 20 := by
  sorry


end intercepts_congruence_l1586_158678


namespace original_equals_scientific_l1586_158637

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 7003000

/-- The scientific notation representation of the number -/
def scientific_form : ScientificNotation :=
  { coefficient := 7.003
    exponent := 6
    is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent := by
  sorry

end original_equals_scientific_l1586_158637


namespace exists_periodic_product_l1586_158686

/-- A function f: ℝ → ℝ is periodic with period p if it's not constant and
    f(x) = f(x + p) for all x ∈ ℝ -/
def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ (∃ x y, f x ≠ f y) ∧ ∀ x, f x = f (x + p)

/-- The period of a periodic function is the smallest positive p satisfying the periodicity condition -/
def Period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  IsPeriodic f p ∧ ∀ q, 0 < q ∧ q < p → ¬ IsPeriodic f q

/-- Given any two positive real numbers a and b, there exist two periodic functions
    f₁ and f₂ with periods a and b respectively, such that their product f₁(x) · f₂(x)
    is also a periodic function -/
theorem exists_periodic_product (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (f₁ f₂ : ℝ → ℝ), Period f₁ a ∧ Period f₂ b ∧
  ∃ p, p > 0 ∧ IsPeriodic (fun x ↦ f₁ x * f₂ x) p := by
  sorry

end exists_periodic_product_l1586_158686


namespace divisible_by_24_count_l1586_158634

theorem divisible_by_24_count :
  (∃! (s : Finset ℕ), 
    (∀ a ∈ s, 0 < a ∧ a < 100 ∧ 24 ∣ (a^3 + 23)) ∧ 
    s.card = 5) :=
by sorry

end divisible_by_24_count_l1586_158634


namespace geometric_sequence_problem_l1586_158638

theorem geometric_sequence_problem (b : ℝ) : 
  b > 0 ∧ 
  (∃ r : ℝ, 180 * r = b ∧ b * r = 36 / 25) → 
  b = Real.sqrt (6480 / 25) :=
sorry

end geometric_sequence_problem_l1586_158638


namespace fraction_equality_l1586_158628

theorem fraction_equality (a : ℕ+) : (a : ℚ) / (a + 35 : ℚ) = 875 / 1000 → a = 245 := by
  sorry

end fraction_equality_l1586_158628


namespace tv_show_minor_characters_l1586_158672

/-- The problem of determining the number of minor characters in a TV show. -/
theorem tv_show_minor_characters :
  let main_characters : ℕ := 5
  let minor_character_pay : ℕ := 15000
  let main_character_pay : ℕ := 3 * minor_character_pay
  let total_pay : ℕ := 285000
  let minor_characters : ℕ := (total_pay - main_characters * main_character_pay) / minor_character_pay
  minor_characters = 4 := by sorry

end tv_show_minor_characters_l1586_158672


namespace min_max_perimeter_12_pieces_l1586_158662

/-- Represents a rectangular piece with length and width in centimeters -/
structure Piece where
  length : ℝ
  width : ℝ

/-- Represents a collection of identical rectangular pieces -/
structure PieceCollection where
  piece : Piece
  count : ℕ

/-- Calculates the area of a rectangular piece -/
def pieceArea (p : Piece) : ℝ := p.length * p.width

/-- Calculates the total area of a collection of pieces -/
def totalArea (pc : PieceCollection) : ℝ := (pieceArea pc.piece) * pc.count

/-- Calculates the perimeter of a rectangle given its length and width -/
def rectanglePerimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Theorem: Minimum and maximum perimeter of rectangle formed by 12 pieces of 4x3 cm -/
theorem min_max_perimeter_12_pieces :
  let pieces : PieceCollection := ⟨⟨4, 3⟩, 12⟩
  let area : ℝ := totalArea pieces
  ∃ (min_perim max_perim : ℝ),
    min_perim = 48 ∧
    max_perim = 102 ∧
    (∀ (l w : ℝ), l * w = area → rectanglePerimeter l w ≥ min_perim) ∧
    (∃ (l w : ℝ), l * w = area ∧ rectanglePerimeter l w = max_perim) :=
by sorry

end min_max_perimeter_12_pieces_l1586_158662


namespace tangent_slope_implies_a_over_b_l1586_158654

-- Define the function f(x) = ax^2 + b
def f (a b x : ℝ) : ℝ := a * x^2 + b

-- Define the derivative of f
def f_derivative (a : ℝ) : ℝ → ℝ := λ x ↦ 2 * a * x

theorem tangent_slope_implies_a_over_b (a b : ℝ) : 
  f a b 1 = 3 ∧ f_derivative a 1 = 2 → a / b = 1 / 2 := by
  sorry

end tangent_slope_implies_a_over_b_l1586_158654


namespace max_value_trig_expression_l1586_158623

theorem max_value_trig_expression (a b c : ℝ) :
  (∀ θ : ℝ, c * (Real.cos θ)^2 ≠ -a) →
  (∃ M : ℝ, M = Real.sqrt (a^2 + b^2 + c^2) ∧
    ∀ θ : ℝ, a * Real.cos θ + b * Real.sin θ + c * Real.tan θ ≤ M) :=
sorry

end max_value_trig_expression_l1586_158623


namespace sin_cos_inequality_l1586_158649

theorem sin_cos_inequality (x : ℝ) : (Real.sin x + 2 * Real.cos (2 * x)) * (2 * Real.sin (2 * x) - Real.cos x) < 4.5 := by
  sorry

end sin_cos_inequality_l1586_158649


namespace max_a_for_integer_solution_l1586_158619

theorem max_a_for_integer_solution : 
  (∃ (a : ℕ+), ∀ (b : ℕ+), 
    (∃ (x : ℤ), x^2 + (b : ℤ) * x = -30) → 
    (b : ℤ) ≤ (a : ℤ)) ∧ 
  (∃ (x : ℤ), x^2 + 31 * x = -30) := by
  sorry

end max_a_for_integer_solution_l1586_158619


namespace arithmetic_mean_of_special_set_l1586_158673

theorem arithmetic_mean_of_special_set (n : ℕ) (h : n > 1) : 
  let set := [1 - 1 / n, 1 + 1 / n] ++ List.replicate (n - 1) 1
  (List.sum set) / (n + 1) = 1 := by
  sorry

end arithmetic_mean_of_special_set_l1586_158673


namespace digit_at_position_l1586_158680

/-- The fraction we're examining -/
def f : ℚ := 17 / 270

/-- The length of the repeating sequence in the decimal representation of f -/
def period : ℕ := 3

/-- The repeating sequence in the decimal representation of f -/
def repeating_sequence : List ℕ := [6, 2, 9]

/-- The position we're interested in -/
def target_position : ℕ := 145

theorem digit_at_position :
  (target_position - 1) % period = 0 →
  List.get! repeating_sequence (period - 1) = 9 := by
  sorry

end digit_at_position_l1586_158680


namespace monomial_sum_implies_a_power_l1586_158606

/-- Given two monomials in x and y whose sum is a monomial, prove a^2004 - 1 = 0 --/
theorem monomial_sum_implies_a_power (m n : ℤ) (a : ℕ) :
  (∃ (x y : ℝ), (3 * m * x^a * y) + (-2 * n * x^(4*a - 3) * y) = x^k * y) →
  a^2004 - 1 = 0 := by
  sorry

end monomial_sum_implies_a_power_l1586_158606


namespace series_sum_eight_l1586_158636

def series_sum : ℕ → ℕ
  | 0 => 0
  | n + 1 => 2^(n + 1) + series_sum n

theorem series_sum_eight : series_sum 8 = 510 := by
  sorry

end series_sum_eight_l1586_158636


namespace division_problem_l1586_158689

theorem division_problem : (250 : ℝ) / (15 + 13 * 3 - 4) = 5 := by
  sorry

end division_problem_l1586_158689


namespace exam_correct_answers_l1586_158614

/-- Given an exam with the following conditions:
  * Total number of questions is 150
  * Correct answers score 4 marks
  * Wrong answers score -2 marks
  * Total score is 420 marks
  Prove that the number of correct answers is 120 -/
theorem exam_correct_answers 
  (total_questions : ℕ) 
  (correct_score wrong_score total_score : ℤ) 
  (h1 : total_questions = 150)
  (h2 : correct_score = 4)
  (h3 : wrong_score = -2)
  (h4 : total_score = 420) : 
  ∃ (correct_answers : ℕ), 
    correct_answers = 120 ∧ 
    correct_answers ≤ total_questions ∧
    correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score :=
by
  sorry

end exam_correct_answers_l1586_158614


namespace test_configuration_theorem_l1586_158661

/-- Represents the fraction of problems that are difficult and the fraction of students who perform well -/
structure TestConfiguration (α : ℚ) :=
  (difficult_problems : ℚ)
  (well_performing_students : ℚ)
  (difficult_problems_ge : difficult_problems ≥ α)
  (well_performing_students_ge : well_performing_students ≥ α)

/-- Theorem stating the existence and non-existence of certain test configurations -/
theorem test_configuration_theorem :
  (∃ (config : TestConfiguration (2/3)), True) ∧
  (¬ ∃ (config : TestConfiguration (3/4)), True) ∧
  (¬ ∃ (config : TestConfiguration (7/10^7)), True) := by
  sorry

end test_configuration_theorem_l1586_158661


namespace unique_solution_diophantine_equation_l1586_158656

theorem unique_solution_diophantine_equation :
  ∃! (x y z t : ℕ+), 1 + 5^x.val = 2^y.val + 2^z.val * 5^t.val :=
by sorry

end unique_solution_diophantine_equation_l1586_158656


namespace purple_length_is_three_l1586_158666

/-- The length of the purple part of a pencil -/
def purple_length (total black blue : ℝ) : ℝ := total - black - blue

/-- Theorem stating that the length of the purple part of the pencil is 3 cm -/
theorem purple_length_is_three :
  let total := 6
  let black := 2
  let blue := 1
  purple_length total black blue = 3 := by
  sorry

end purple_length_is_three_l1586_158666


namespace equal_wins_losses_probability_l1586_158677

/-- Represents the result of a single match -/
inductive MatchResult
  | Win
  | Loss
  | Tie

/-- Probability distribution for match results -/
def matchProbability : MatchResult → Rat
  | MatchResult.Win => 1/4
  | MatchResult.Loss => 1/4
  | MatchResult.Tie => 1/2

/-- Total number of matches played -/
def totalMatches : Nat := 10

/-- Calculates the probability of having equal wins and losses in a season -/
def probabilityEqualWinsLosses : Rat :=
  63/262144

theorem equal_wins_losses_probability :
  probabilityEqualWinsLosses = 63/262144 := by
  sorry

#check equal_wins_losses_probability

end equal_wins_losses_probability_l1586_158677


namespace equation_solutions_l1586_158612

theorem equation_solutions :
  (∃ x1 x2 : ℝ, x1 = (1 + Real.sqrt 5) / 4 ∧ x2 = (1 - Real.sqrt 5) / 4 ∧
    4 * x1^2 - 2 * x1 - 1 = 0 ∧ 4 * x2^2 - 2 * x2 - 1 = 0) ∧
  (∃ y1 y2 : ℝ, y1 = 1 ∧ y2 = 0 ∧
    (y1 + 1)^2 = (3 * y1 - 1)^2 ∧ (y2 + 1)^2 = (3 * y2 - 1)^2) :=
by sorry

end equation_solutions_l1586_158612


namespace opposite_sides_line_parameter_range_l1586_158651

/-- Given two points on opposite sides of a line, determine the range of the line's parameter --/
theorem opposite_sides_line_parameter_range :
  ∀ (m : ℝ),
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁ = 3 ∧ y₁ = 1 ∧ x₂ = -4 ∧ y₂ = 6 ∧
    (3 * x₁ - 2 * y₁ + m) * (3 * x₂ - 2 * y₂ + m) < 0) →
  7 < m ∧ m < 24 :=
by sorry


end opposite_sides_line_parameter_range_l1586_158651


namespace all_heads_possible_l1586_158667

/-- Represents the state of a coin (heads or tails) -/
inductive CoinState
| Heads
| Tails

/-- Represents the state of all coins in a row -/
def CoinRow := Vector CoinState 100

/-- An operation that flips 7 equally spaced coins -/
def FlipOperation := Fin 100 → Fin 7 → Bool

/-- Applies a flip operation to a coin row -/
def applyFlip (row : CoinRow) (op : FlipOperation) : CoinRow :=
  sorry

/-- The theorem stating that any initial coin configuration can be transformed to all heads -/
theorem all_heads_possible (initial : CoinRow) : 
  ∃ (ops : List FlipOperation), 
    let final := ops.foldl applyFlip initial
    ∀ i, final.get i = CoinState.Heads :=
  sorry

end all_heads_possible_l1586_158667


namespace jerry_shelves_problem_l1586_158640

def shelves_needed (total_books : ℕ) (books_taken : ℕ) (books_per_shelf : ℕ) : ℕ :=
  (total_books - books_taken + books_per_shelf - 1) / books_per_shelf

theorem jerry_shelves_problem :
  shelves_needed 34 7 3 = 9 :=
by sorry

end jerry_shelves_problem_l1586_158640


namespace tangent_ratio_problem_l1586_158676

theorem tangent_ratio_problem (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β - π/4) = 1/4) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3/22 := by
  sorry

end tangent_ratio_problem_l1586_158676


namespace simplified_expression_ratio_l1586_158647

theorem simplified_expression_ratio (m : ℝ) :
  let original := (6 * m + 18) / 6
  ∃ (c d : ℤ), (∃ (x : ℝ), original = c * x + d) ∧ (c : ℚ) / d = 1 / 3 :=
by sorry

end simplified_expression_ratio_l1586_158647


namespace expansive_sequence_existence_l1586_158601

def expansive (a : ℕ → ℝ) : Prop :=
  ∀ i j : ℕ, i < j → |a i - a j| ≥ 1 / j

theorem expansive_sequence_existence (C : ℝ) :
  (C > 0 ∧ ∃ a : ℕ → ℝ, expansive a ∧ ∀ n, 0 ≤ a n ∧ a n ≤ C) ↔ C ≥ 2 * Real.log 2 :=
sorry

end expansive_sequence_existence_l1586_158601
