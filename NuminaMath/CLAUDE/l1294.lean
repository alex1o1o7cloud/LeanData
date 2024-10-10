import Mathlib

namespace puppy_weight_l1294_129475

/-- Represents the weight of animals in pounds -/
structure AnimalWeights where
  puppy : ℝ
  smaller_cat : ℝ
  larger_cat : ℝ

/-- The conditions of the problem -/
def satisfies_conditions (w : AnimalWeights) : Prop :=
  w.puppy + 2 * w.smaller_cat + w.larger_cat = 38 ∧
  w.puppy + w.larger_cat = 3 * w.smaller_cat ∧
  w.puppy + 2 * w.smaller_cat = w.larger_cat

/-- The theorem stating the puppy's weight -/
theorem puppy_weight (w : AnimalWeights) (h : satisfies_conditions w) : w.puppy = 3.8 := by
  sorry

#check puppy_weight

end puppy_weight_l1294_129475


namespace polynomial_multiplication_l1294_129404

-- Define the polynomials
def p (z : ℝ) : ℝ := 3 * z^2 - 4 * z + 1
def q (z : ℝ) : ℝ := 4 * z^3 + z^2 - 5 * z + 3

-- State the theorem
theorem polynomial_multiplication :
  ∀ z : ℝ, p z * q z = 12 * z^5 + 3 * z^4 + 32 * z^3 + z^2 - 7 * z + 3 :=
by sorry

end polynomial_multiplication_l1294_129404


namespace weight_of_3_moles_HBrO3_l1294_129453

/-- The molecular weight of a single HBrO3 molecule in g/mol -/
def molecular_weight_HBrO3 : ℝ :=
  1.01 + 79.90 + 3 * 16.00

/-- The weight of 3 moles of HBrO3 in grams -/
def weight_3_moles_HBrO3 : ℝ :=
  3 * molecular_weight_HBrO3

theorem weight_of_3_moles_HBrO3 :
  weight_3_moles_HBrO3 = 386.73 := by sorry

end weight_of_3_moles_HBrO3_l1294_129453


namespace congruence_solution_l1294_129427

theorem congruence_solution :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -437 [ZMOD 10] ∧ n = 3 := by
  sorry

end congruence_solution_l1294_129427


namespace people_dislike_radio_and_music_l1294_129434

theorem people_dislike_radio_and_music
  (total_people : ℕ)
  (radio_dislike_percent : ℚ)
  (music_dislike_percent : ℚ)
  (h_total : total_people = 1500)
  (h_radio : radio_dislike_percent = 40 / 100)
  (h_music : music_dislike_percent = 15 / 100) :
  (total_people : ℚ) * radio_dislike_percent * music_dislike_percent = 90 := by
  sorry

end people_dislike_radio_and_music_l1294_129434


namespace solution_difference_l1294_129408

theorem solution_difference : ∃ (x₁ x₂ : ℝ),
  (x₁^(1/3 : ℝ) = -3 ∧ 9 - x₁^2 / 4 = (-3)^3) ∧
  (x₂^(1/3 : ℝ) = -3 ∧ 9 - x₂^2 / 4 = (-3)^3) ∧
  x₁ ≠ x₂ ∧
  |x₁ - x₂| = 24 :=
by sorry

end solution_difference_l1294_129408


namespace vacation_cost_l1294_129483

theorem vacation_cost (C : ℝ) : 
  (C / 3 - C / 5 = 50) → C = 375 := by
  sorry

end vacation_cost_l1294_129483


namespace net_change_in_cards_l1294_129426

def sold_cards : ℤ := 27
def received_cards : ℤ := 41
def bought_cards : ℤ := 20

theorem net_change_in_cards : -sold_cards + received_cards + bought_cards = 34 := by
  sorry

end net_change_in_cards_l1294_129426


namespace grandchildren_probability_l1294_129412

def num_children : ℕ := 12

theorem grandchildren_probability :
  let total_outcomes := 2^num_children
  let equal_boys_girls := Nat.choose num_children (num_children / 2)
  let all_same_gender := 2
  (total_outcomes - (equal_boys_girls + all_same_gender)) / total_outcomes = 3170 / 4096 := by
  sorry

end grandchildren_probability_l1294_129412


namespace mount_pilot_snow_amount_l1294_129477

/-- The amount of snow on Mount Pilot in centimeters -/
def mount_pilot_snow (bald_snow billy_snow : ℝ) : ℝ :=
  (billy_snow * 100 + (billy_snow * 100 + bald_snow * 100 + 326) - bald_snow * 100) - billy_snow * 100

/-- Theorem stating that Mount Pilot received 326 cm of snow -/
theorem mount_pilot_snow_amount :
  mount_pilot_snow 1.5 3.5 = 326 := by
  sorry

#eval mount_pilot_snow 1.5 3.5

end mount_pilot_snow_amount_l1294_129477


namespace cube_volume_problem_l1294_129498

theorem cube_volume_problem (a : ℝ) : 
  (a > 0) →  -- Ensure positive side length
  (a^3 - ((a - 1) * a * (a + 1)) = 5) →
  (a^3 = 125) :=
by sorry

end cube_volume_problem_l1294_129498


namespace kenya_peanuts_l1294_129400

theorem kenya_peanuts (jose_peanuts : ℕ) (kenya_difference : ℕ) 
  (h1 : jose_peanuts = 85)
  (h2 : kenya_difference = 48) :
  jose_peanuts + kenya_difference = 133 := by
  sorry

end kenya_peanuts_l1294_129400


namespace min_sum_reciprocal_one_l1294_129452

theorem min_sum_reciprocal_one (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 1) :
  x + y ≥ 4 ∧ (x + y = 4 ↔ x = 2 ∧ y = 2) :=
by sorry

end min_sum_reciprocal_one_l1294_129452


namespace sqrt_three_squared_four_to_fourth_l1294_129479

theorem sqrt_three_squared_four_to_fourth : Real.sqrt (3^2 * 4^4) = 48 := by sorry

end sqrt_three_squared_four_to_fourth_l1294_129479


namespace distance_sum_bounds_l1294_129401

/-- Given three mutually perpendicular segments with lengths a, b, and c,
    this theorem proves the bounds for the sum of distances from the endpoints
    to any line passing through the origin. -/
theorem distance_sum_bounds
  (a b c : ℝ)
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c)
  (h_order : a ≤ b ∧ b ≤ c) :
  ∀ (α β γ : ℝ),
    (α^2 + β^2 + γ^2 = 1) →
    (a * α + b * β + c * γ ≥ a + b) ∧
    (a * α + b * β + c * γ ≤ c + Real.sqrt (a^2 + b^2)) :=
by sorry

end distance_sum_bounds_l1294_129401


namespace angle_c_is_60_degrees_l1294_129403

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define the concept of an angle in a quadrilateral
def angle (q : Quadrilateral) (v : Fin 4) : ℝ := sorry

-- State the theorem
theorem angle_c_is_60_degrees (q : Quadrilateral) :
  angle q 0 + 60 = angle q 1 →  -- Angle A is 60° smaller than angle B
  angle q 2 = 60 := by  -- Angle C is 60°
  sorry

end angle_c_is_60_degrees_l1294_129403


namespace pascal_row20_sum_l1294_129490

theorem pascal_row20_sum : Nat.choose 20 4 + Nat.choose 20 5 = 20349 := by
  sorry

end pascal_row20_sum_l1294_129490


namespace smallest_divisor_l1294_129470

theorem smallest_divisor : 
  let n : ℕ := 1012
  let m : ℕ := n - 4
  let divisors : List ℕ := [16, 18, 21, 28]
  (∀ d ∈ divisors, m % d = 0) ∧ 
  (∀ d ∈ divisors, d ≥ 16) ∧
  16 ∈ divisors →
  16 = (divisors.filter (λ d => m % d = 0)).minimum?.getD 0 :=
by sorry

end smallest_divisor_l1294_129470


namespace bottles_to_buy_promotion_l1294_129496

/-- Calculates the number of bottles to buy given a promotion and total bottles needed -/
def bottlesToBuy (bottlesNeeded : ℕ) (buyQuantity : ℕ) (freeQuantity : ℕ) : ℕ :=
  bottlesNeeded - (bottlesNeeded / (buyQuantity + freeQuantity)) * freeQuantity

/-- Proves that 8 bottles need to be bought given the promotion and number of people -/
theorem bottles_to_buy_promotion (numPeople : ℕ) (buyQuantity : ℕ) (freeQuantity : ℕ) :
  numPeople = 10 → buyQuantity = 4 → freeQuantity = 1 →
  bottlesToBuy numPeople buyQuantity freeQuantity = 8 :=
by
  sorry

#eval bottlesToBuy 10 4 1  -- Should output 8

end bottles_to_buy_promotion_l1294_129496


namespace parallelogram_area_l1294_129459

theorem parallelogram_area (base height : ℝ) (h1 : base = 60) (h2 : height = 16) :
  base * height = 960 := by
  sorry

end parallelogram_area_l1294_129459


namespace smallest_divisible_by_1_to_10_is_correct_l1294_129480

/-- The smallest positive integer divisible by all integers from 1 to 10 -/
def smallest_divisible_by_1_to_10 : ℕ := 27720

/-- Proposition: smallest_divisible_by_1_to_10 is the smallest positive integer 
    divisible by all integers from 1 to 10 -/
theorem smallest_divisible_by_1_to_10_is_correct :
  (∀ n : ℕ, n > 0 ∧ n < smallest_divisible_by_1_to_10 → 
    ∃ m : ℕ, m ∈ Finset.range 10 ∧ n % (m + 1) ≠ 0) ∧
  (∀ m : ℕ, m ∈ Finset.range 10 → smallest_divisible_by_1_to_10 % (m + 1) = 0) :=
sorry

end smallest_divisible_by_1_to_10_is_correct_l1294_129480


namespace triangle_problem_l1294_129463

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  -- Given conditions
  a = Real.sqrt 7 →
  b = 2 →
  A = 60 * π / 180 →  -- Convert 60° to radians
  -- Triangle inequality
  a + b > c ∧ b + c > a ∧ c + a > b →
  -- Positive side lengths
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Sum of angles in a triangle is π
  A + B + C = π →
  -- Sine law
  a / Real.sin A = b / Real.sin B →
  -- Cosine law
  c^2 = a^2 + b^2 - 2*a*b*Real.cos C →
  -- Conclusions
  Real.sin B = Real.sqrt 21 / 7 ∧ c = 3 := by
sorry


end triangle_problem_l1294_129463


namespace ratio_of_fifth_terms_in_arithmetic_sequences_l1294_129437

/-- Given two arithmetic sequences, prove the ratio of their 5th terms -/
theorem ratio_of_fifth_terms_in_arithmetic_sequences 
  (a b : ℕ → ℚ) 
  (h_arithmetic_a : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_arithmetic_b : ∀ n, b (n + 1) - b n = b 1 - b 0)
  (h_ratio : ∀ n, (n * (a 0 + a n)) / (n * (b 0 + b n)) = (3 * n) / (2 * n + 9)) :
  a 5 / b 5 = 15 / 19 := by
sorry


end ratio_of_fifth_terms_in_arithmetic_sequences_l1294_129437


namespace no_extremum_condition_l1294_129485

/-- A cubic function f(x) = ax³ + bx² + cx + d with a > 0 -/
def cubic_function (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

/-- The derivative of the cubic function -/
def cubic_derivative (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

/-- Condition for no extremum: the derivative is always non-negative -/
def no_extremum (a b c : ℝ) : Prop :=
  ∀ x, cubic_derivative a b c x ≥ 0

theorem no_extremum_condition (a b c d : ℝ) (ha : a > 0) :
  no_extremum a b c → b^2 - 3*a*c ≤ 0 := by
  sorry

end no_extremum_condition_l1294_129485


namespace geometric_sequence_sum_of_squares_l1294_129441

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum_of_squares 
  (a : ℕ → ℝ) 
  (h_geo : GeometricSequence a) 
  (h_sum : a 3 + a 5 = 5) 
  (h_prod : a 2 * a 6 = 4) : 
  a 3 ^ 2 + a 5 ^ 2 = 17 := by
sorry

end geometric_sequence_sum_of_squares_l1294_129441


namespace randy_pictures_l1294_129435

/-- Given that Peter drew 8 pictures, Quincy drew 20 more pictures than Peter,
    and they drew 41 pictures altogether, prove that Randy drew 5 pictures. -/
theorem randy_pictures (peter_pictures : ℕ) (quincy_pictures : ℕ) (total_pictures : ℕ) :
  peter_pictures = 8 →
  quincy_pictures = peter_pictures + 20 →
  total_pictures = 41 →
  total_pictures = peter_pictures + quincy_pictures + 5 := by
  sorry

end randy_pictures_l1294_129435


namespace julia_total_food_expense_l1294_129409

/-- Represents the weekly food cost and number of weeks for an animal -/
structure AnimalExpense where
  weeklyFoodCost : ℕ
  numberOfWeeks : ℕ

/-- Calculates the total food expense for Julia's animals -/
def totalFoodExpense (animals : List AnimalExpense) : ℕ :=
  animals.map (fun a => a.weeklyFoodCost * a.numberOfWeeks) |>.sum

/-- The list of Julia's animals with their expenses -/
def juliaAnimals : List AnimalExpense := [
  ⟨15, 3⟩,  -- Parrot
  ⟨12, 5⟩,  -- Rabbit
  ⟨8, 2⟩,   -- Turtle
  ⟨5, 6⟩    -- Guinea pig
]

/-- Theorem stating that Julia's total food expense is $151 -/
theorem julia_total_food_expense :
  totalFoodExpense juliaAnimals = 151 := by
  sorry

end julia_total_food_expense_l1294_129409


namespace unique_solution_logarithmic_equation_l1294_129429

theorem unique_solution_logarithmic_equation :
  ∃! x : ℝ, x > 0 ∧ x^(Real.log 3) + x^(Real.log 4) = x^(Real.log 5) :=
by sorry

end unique_solution_logarithmic_equation_l1294_129429


namespace basketball_score_problem_l1294_129448

theorem basketball_score_problem (total_points hawks_points eagles_points : ℕ) : 
  total_points = 50 →
  hawks_points = eagles_points + 6 →
  hawks_points + eagles_points = total_points →
  eagles_points = 22 := by
sorry

end basketball_score_problem_l1294_129448


namespace marlas_grid_squares_per_row_l1294_129461

/-- Represents a grid with colored squares -/
structure ColoredGrid where
  rows : ℕ
  squaresPerRow : ℕ
  redSquares : ℕ
  blueRows : ℕ
  greenSquares : ℕ

/-- The number of squares in each row of Marla's grid -/
def marlasGridSquaresPerRow : ℕ := 15

/-- Theorem stating that Marla's grid has 15 squares per row -/
theorem marlas_grid_squares_per_row :
  ∃ (g : ColoredGrid),
    g.rows = 10 ∧
    g.redSquares = 24 ∧
    g.blueRows = 4 ∧
    g.greenSquares = 66 ∧
    g.squaresPerRow = marlasGridSquaresPerRow :=
by sorry


end marlas_grid_squares_per_row_l1294_129461


namespace min_value_of_f_l1294_129493

-- Define the function f(x)
def f (x a : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

-- State the theorem
theorem min_value_of_f (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f x a ≥ f y a) ∧ 
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x a = 20) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f x a ≤ f y a) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x a = -7) :=
by sorry

end min_value_of_f_l1294_129493


namespace remaining_nails_l1294_129411

def initial_nails : ℕ := 400

def kitchen_repair (n : ℕ) : ℕ := n - (n * 35 / 100)

def fence_repair (n : ℕ) : ℕ := n - (n * 75 / 100)

def table_repair (n : ℕ) : ℕ := n - (n * 55 / 100)

def floorboard_repair (n : ℕ) : ℕ := n - (n * 30 / 100)

theorem remaining_nails :
  floorboard_repair (table_repair (fence_repair (kitchen_repair initial_nails))) = 21 :=
by sorry

end remaining_nails_l1294_129411


namespace max_sum_of_squares_l1294_129464

/-- Given a system of equations, prove that the maximum value of a^2 + b^2 + c^2 + d^2 is 82 -/
theorem max_sum_of_squares (a b c d : ℝ) 
  (h1 : a + b = 16)
  (h2 : a * b + c + d = 81)
  (h3 : a * d + b * c = 168)
  (h4 : c * d = 100) :
  ∀ (w x y z : ℝ), 
  (w + x = 16) → 
  (w * x + y + z = 81) → 
  (w * z + x * y = 168) → 
  (y * z = 100) → 
  a^2 + b^2 + c^2 + d^2 ≥ w^2 + x^2 + y^2 + z^2 ∧
  a^2 + b^2 + c^2 + d^2 ≤ 82 :=
by
  sorry

#check max_sum_of_squares

end max_sum_of_squares_l1294_129464


namespace min_k_for_inequality_l1294_129468

theorem min_k_for_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (∃ k : ℝ, ∀ x y : ℝ, x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ k * Real.sqrt (5 * x + y)) ↔
  (∃ k : ℝ, k ≥ Real.sqrt 30 / 5 ∧ 
    ∀ x y : ℝ, x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ k * Real.sqrt (5 * x + y)) :=
by sorry

end min_k_for_inequality_l1294_129468


namespace prism_pyramid_sum_l1294_129432

/-- A shape formed by adding a pyramid to one face of a rectangular prism -/
structure PrismPyramid where
  prism_faces : ℕ
  prism_edges : ℕ
  prism_vertices : ℕ
  pyramid_faces : ℕ
  pyramid_edges : ℕ
  pyramid_vertex : ℕ

/-- The total number of exterior faces in the combined shape -/
def total_faces (pp : PrismPyramid) : ℕ := pp.prism_faces - 1 + pp.pyramid_faces

/-- The total number of edges in the combined shape -/
def total_edges (pp : PrismPyramid) : ℕ := pp.prism_edges + pp.pyramid_edges

/-- The total number of vertices in the combined shape -/
def total_vertices (pp : PrismPyramid) : ℕ := pp.prism_vertices + pp.pyramid_vertex

/-- The sum of exterior faces, edges, and vertices in the combined shape -/
def total_sum (pp : PrismPyramid) : ℕ := total_faces pp + total_edges pp + total_vertices pp

theorem prism_pyramid_sum :
  ∃ (pp : PrismPyramid), total_sum pp = 34 ∧
  ∀ (pp' : PrismPyramid), total_sum pp' ≤ total_sum pp :=
sorry

end prism_pyramid_sum_l1294_129432


namespace profit_starts_in_third_year_option1_more_cost_effective_l1294_129424

-- Define the constants
def initial_cost : ℕ := 980000
def first_year_expenses : ℕ := 120000
def yearly_expense_increase : ℕ := 40000
def annual_income : ℕ := 500000

-- Define a function to calculate expenses for a given year
def expenses (year : ℕ) : ℕ :=
  first_year_expenses + (year - 1) * yearly_expense_increase

-- Define a function to calculate cumulative profit for a given year
def cumulative_profit (year : ℕ) : ℤ :=
  year * annual_income - (initial_cost + (Finset.range year).sum (λ i => expenses (i + 1)))

-- Define a function to calculate average profit for a given year
def average_profit (year : ℕ) : ℚ :=
  (cumulative_profit year : ℚ) / year

-- Theorem 1: The company starts to make a profit in the third year
theorem profit_starts_in_third_year :
  cumulative_profit 3 > 0 ∧ ∀ y : ℕ, y < 3 → cumulative_profit y ≤ 0 := by sorry

-- Define the selling prices for the two options
def option1_price : ℕ := 260000
def option2_price : ℕ := 80000

-- Theorem 2: Option 1 is more cost-effective than Option 2
theorem option1_more_cost_effective :
  ∃ y1 y2 : ℕ,
    (∀ y : ℕ, average_profit y ≤ average_profit y1) ∧
    (∀ y : ℕ, cumulative_profit y ≤ cumulative_profit y2) ∧
    option1_price + cumulative_profit y1 > option2_price + cumulative_profit y2 := by sorry

end profit_starts_in_third_year_option1_more_cost_effective_l1294_129424


namespace rational_function_sum_l1294_129402

/-- A rational function with specific properties -/
def RationalFunction (p q : ℝ → ℝ) : Prop :=
  (∃ k a : ℝ, q = fun x ↦ k * (x + 3) * (x - 1) * (x - a)) ∧
  (∃ b : ℝ, p = fun x ↦ b * x + 2) ∧
  q 0 = -2

/-- The theorem statement -/
theorem rational_function_sum (p q : ℝ → ℝ) :
  RationalFunction p q →
  ∃! p, p + q = fun x ↦ (1/3) * x^3 - (1/3) * x^2 + (11/3) * x + 4 :=
by sorry

end rational_function_sum_l1294_129402


namespace dress_discount_percentage_l1294_129458

theorem dress_discount_percentage (d : ℝ) (x : ℝ) (h : d > 0) :
  d * ((100 - x) / 100) * 0.5 = 0.225 * d → x = 55 := by
sorry

end dress_discount_percentage_l1294_129458


namespace circle_radius_spherical_coordinates_l1294_129413

/-- The radius of the circle formed by points with spherical coordinates (1, θ, π/4) is √2/2 --/
theorem circle_radius_spherical_coordinates :
  let r := Real.sqrt ((Real.sin (π/4))^2 + (Real.cos (π/4))^2)
  r = Real.sqrt 2 / 2 := by
  sorry

end circle_radius_spherical_coordinates_l1294_129413


namespace arithmetic_mean_of_fractions_l1294_129472

theorem arithmetic_mean_of_fractions : 
  (3/8 + 5/9) / 2 = 67/144 := by sorry

end arithmetic_mean_of_fractions_l1294_129472


namespace max_product_sum_300_l1294_129431

theorem max_product_sum_300 : 
  (∃ a b : ℤ, a + b = 300 ∧ ∀ x y : ℤ, x + y = 300 → x * y ≤ a * b) ∧ 
  (∃ a b : ℤ, a + b = 300 ∧ a * b = 22500) := by
sorry

end max_product_sum_300_l1294_129431


namespace simplify_complex_radical_expression_l1294_129433

theorem simplify_complex_radical_expression :
  (3 * (Real.sqrt 5 + Real.sqrt 7)) / (4 * Real.sqrt (3 + Real.sqrt 5)) =
  Real.sqrt (414 - 98 * Real.sqrt 35) / 8 := by
  sorry

end simplify_complex_radical_expression_l1294_129433


namespace g_sum_neg_one_l1294_129415

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- State the functional equation
axiom func_eq : ∀ x y : ℝ, f (x - y) = f x * g y - g x * f y

-- State the condition f(-2) = f(1) ≠ 0
axiom f_cond : f (-2) = f 1 ∧ f 1 ≠ 0

-- Theorem to prove
theorem g_sum_neg_one : g 1 + g (-1) = -1 :=
sorry

end g_sum_neg_one_l1294_129415


namespace choose_two_from_four_with_repetition_l1294_129414

/-- The number of ways to choose r items from n items with repetition allowed -/
def combinationsWithRepetition (n : ℕ) (r : ℕ) : ℕ :=
  Nat.choose (n + r - 1) r

theorem choose_two_from_four_with_repetition :
  combinationsWithRepetition 4 2 = 10 := by
  sorry

end choose_two_from_four_with_repetition_l1294_129414


namespace nth_term_from_sum_l1294_129478

/-- Given a sequence {a_n} where S_n = 3n^2 - 2n is the sum of its first n terms,
    prove that the n-th term of the sequence is a_n = 6n - 5 for all natural numbers n. -/
theorem nth_term_from_sum (n : ℕ) (a : ℕ → ℚ) (S : ℕ → ℚ) 
    (h : ∀ k, S k = 3 * k^2 - 2 * k) : 
  a n = 6 * n - 5 := by
  sorry

end nth_term_from_sum_l1294_129478


namespace computer_room_arrangements_l1294_129418

/-- The number of different computer rooms -/
def n : ℕ := 6

/-- The minimum number of rooms that must be open -/
def k : ℕ := 2

/-- The number of arrangements for opening at least k out of n rooms -/
def num_arrangements (n k : ℕ) : ℕ := sorry

/-- Sum of combinations for opening 3 to 6 rooms, with 4 rooms counted twice -/
def sum_combinations (n : ℕ) : ℕ := 
  Nat.choose n 3 + 2 * Nat.choose n 4 + Nat.choose n 5 + Nat.choose n 6

/-- Total arrangements minus arrangements for 0 and 1 room -/
def power_minus_seven (n : ℕ) : ℕ := 2^n - 7

theorem computer_room_arrangements :
  num_arrangements n k = sum_combinations n ∧ 
  num_arrangements n k = power_minus_seven n := by sorry

end computer_room_arrangements_l1294_129418


namespace set_relationship_l1294_129473

-- Define the sets
def set1 : Set ℝ := {x | (1 : ℝ) / x ≤ 1}
def set2 : Set ℝ := {x | Real.log x ≥ 0}

-- Theorem statement
theorem set_relationship : Set.Subset set2 set1 ∧ ¬(set1 = set2) := by
  sorry

end set_relationship_l1294_129473


namespace gift_purchase_probability_is_correct_l1294_129465

/-- The probability of purchasing gifts from all three stores and still having money left -/
def gift_purchase_probability : ℚ :=
  let initial_amount : ℕ := 5000
  let num_stores : ℕ := 3
  let prices : List ℕ := [1000, 1500, 2000]
  let total_combinations : ℕ := 3^num_stores
  let favorable_cases : ℕ := 17
  favorable_cases / total_combinations

/-- Theorem stating the probability of successful gift purchases -/
theorem gift_purchase_probability_is_correct :
  gift_purchase_probability = 17 / 27 := by sorry

end gift_purchase_probability_is_correct_l1294_129465


namespace gate_width_scientific_notation_l1294_129484

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |significand| ∧ |significand| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem gate_width_scientific_notation :
  toScientificNotation 0.000000007 = ScientificNotation.mk 7 (-9) sorry := by
  sorry

end gate_width_scientific_notation_l1294_129484


namespace interest_rate_increase_l1294_129410

/-- Proves that if an interest rate increases by 10 percent to become 11 percent,
    the original interest rate was 10 percent. -/
theorem interest_rate_increase (original_rate : ℝ) : 
  (original_rate * 1.1 = 0.11) → (original_rate = 0.1) := by
  sorry

end interest_rate_increase_l1294_129410


namespace a_is_independent_variable_l1294_129488

-- Define the perimeter function for a rhombus
def rhombus_perimeter (a : ℝ) : ℝ := 4 * a

-- Statement to prove
theorem a_is_independent_variable :
  ∃ (C : ℝ → ℝ), C = rhombus_perimeter ∧ 
  (∀ (a : ℝ), C a = 4 * a) ∧
  (∀ (a₁ a₂ : ℝ), a₁ ≠ a₂ → C a₁ ≠ C a₂) :=
sorry

end a_is_independent_variable_l1294_129488


namespace courtyard_width_prove_courtyard_width_l1294_129481

/-- The width of a rectangular courtyard given specific conditions -/
theorem courtyard_width : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun (length width stone_side num_stones : ℝ) =>
    length = 30 ∧
    stone_side = 2 ∧
    num_stones = 135 ∧
    length * width = num_stones * stone_side * stone_side →
    width = 18

/-- Proof of the courtyard width theorem -/
theorem prove_courtyard_width :
  ∃ (length width stone_side num_stones : ℝ),
    courtyard_width length width stone_side num_stones :=
by
  sorry

end courtyard_width_prove_courtyard_width_l1294_129481


namespace intersection_of_complements_l1294_129451

def U : Set ℕ := {x | x ≤ 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 4}

theorem intersection_of_complements :
  (U \ A) ∩ (U \ B) = {0, 5} := by sorry

end intersection_of_complements_l1294_129451


namespace opposite_of_2023_l1294_129492

theorem opposite_of_2023 : 
  ∀ (x : ℤ), x = 2023 → -x = -2023 := by
  sorry

end opposite_of_2023_l1294_129492


namespace option_d_most_suitable_for_comprehensive_survey_l1294_129439

/-- Represents a survey option -/
inductive SurveyOption
| A : SurveyOption  -- Investigating the service life of a batch of infrared thermometers
| B : SurveyOption  -- Investigating the travel methods of the people of Henan during the Spring Festival
| C : SurveyOption  -- Investigating the viewership of the Henan TV program "Li Yuan Chun"
| D : SurveyOption  -- Investigating the heights of all classmates

/-- Characteristics of a survey -/
structure SurveyCharacteristics where
  population_size : ℕ
  precision : ℝ

/-- Defines what makes a survey suitable for a comprehensive survey -/
def is_suitable_for_comprehensive_survey (s : SurveyCharacteristics) : Prop :=
  s.population_size ≤ 1000 ∧ s.precision ≥ 0.99

/-- Associates survey options with their characteristics -/
def survey_characteristics : SurveyOption → SurveyCharacteristics
| SurveyOption.A => ⟨10000, 0.9⟩
| SurveyOption.B => ⟨20000000, 0.8⟩
| SurveyOption.C => ⟨5000000, 0.85⟩
| SurveyOption.D => ⟨50, 0.99⟩

/-- Theorem: Option D is the most suitable for a comprehensive survey -/
theorem option_d_most_suitable_for_comprehensive_survey :
  ∀ (o : SurveyOption), o ≠ SurveyOption.D →
    is_suitable_for_comprehensive_survey (survey_characteristics SurveyOption.D) ∧
    ¬is_suitable_for_comprehensive_survey (survey_characteristics o) :=
by sorry


end option_d_most_suitable_for_comprehensive_survey_l1294_129439


namespace circle_triangle_construction_l1294_129444

theorem circle_triangle_construction (R r : ℝ) (h : R > r) :
  ∃ (a b : ℝ),
    a > 0 ∧ b > 0 ∧
    a^2 = 2 * (R^2 + r^2) ∧
    b^2 = 2 * (R^2 - r^2) ∧
    (π * a^2 / 4 + π * b^2 / 4 = π * R^2) ∧
    (π * a^2 / 4 - π * b^2 / 4 = π * r^2) := by
  sorry

end circle_triangle_construction_l1294_129444


namespace compound_proposition_true_l1294_129487

theorem compound_proposition_true (a b : ℝ) :
  (a > 0 ∧ a + b < 0) → b < 0 := by
  sorry

end compound_proposition_true_l1294_129487


namespace continuous_function_image_interval_l1294_129438

open Set

theorem continuous_function_image_interval 
  (f : ℝ → ℝ) (hf : Continuous f) (a b : ℝ) (hab : a < b)
  (ha : a ∈ Set.range f) (hb : b ∈ Set.range f) :
  ∃ (I : Set ℝ), ∃ (s t : ℝ), I = Icc s t ∧ f '' I = Icc a b := by
  sorry

end continuous_function_image_interval_l1294_129438


namespace max_value_theorem_l1294_129421

theorem max_value_theorem (x : ℝ) (h : x < -3) :
  x + 2 / (x + 3) ≤ -2 * Real.sqrt 2 - 3 ∧
  ∃ y, y < -3 ∧ y + 2 / (y + 3) = -2 * Real.sqrt 2 - 3 :=
by sorry

end max_value_theorem_l1294_129421


namespace modulus_of_z_l1294_129450

-- Define the complex number z
def z : ℂ := 1 + Complex.I

-- Theorem statement
theorem modulus_of_z : Complex.abs z = Real.sqrt 2 := by
  sorry

end modulus_of_z_l1294_129450


namespace distribute_5_3_l1294_129425

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 3^5 ways to distribute 5 distinguishable balls into 3 distinguishable boxes -/
theorem distribute_5_3 : distribute 5 3 = 3^5 := by sorry

end distribute_5_3_l1294_129425


namespace complement_intersection_equals_singleton_l1294_129489

def U : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}
def A : Set ℤ := {x | -1 < x ∧ x < 3}
def B : Set ℤ := {x | x^2 - x - 2 ≤ 0}

theorem complement_intersection_equals_singleton :
  (U \ A) ∩ B = {-1} := by sorry

end complement_intersection_equals_singleton_l1294_129489


namespace extremum_condition_l1294_129422

def f (a b x : ℝ) := x^3 - a*x^2 - b*x + a^2

theorem extremum_condition (a b : ℝ) :
  (∃ (ε : ℝ), ∀ (x : ℝ), x ≠ 1 → |x - 1| < ε → f a b x ≤ f a b 1) ∧
  f a b 1 = 10 →
  a + b = 7 := by sorry

end extremum_condition_l1294_129422


namespace problem_statement_l1294_129442

theorem problem_statement (x y z : ℝ) 
  (h1 : 2 * x - y - 2 * z - 6 = 0) 
  (h2 : x^2 + y^2 + z^2 ≤ 4) : 
  2 * x + y + z = 2/3 := by
sorry

end problem_statement_l1294_129442


namespace christine_distance_l1294_129416

theorem christine_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 20 → time = 4 → distance = speed * time → distance = 80 := by
  sorry

end christine_distance_l1294_129416


namespace triangle_interior_lines_sum_bound_l1294_129482

-- Define a triangle with side lengths x, y, z
structure Triangle where
  x : ℝ
  y : ℝ
  z : ℝ
  hxy : x ≤ y
  hyz : y ≤ z

-- Define the sum s
def s (t : Triangle) (XX' YY' ZZ' : ℝ) : ℝ := XX' + YY' + ZZ'

-- Theorem statement
theorem triangle_interior_lines_sum_bound (t : Triangle) 
  (XX' YY' ZZ' : ℝ) (hXX' : XX' ≥ 0) (hYY' : YY' ≥ 0) (hZZ' : ZZ' ≥ 0) : 
  s t XX' YY' ZZ' ≤ t.x + t.y + t.z := by
  sorry

end triangle_interior_lines_sum_bound_l1294_129482


namespace solve_equation_l1294_129462

theorem solve_equation : ∃ x : ℝ, 0.5 * x + (0.3 * 0.2) = 0.26 ∧ x = 0.4 := by
  sorry

end solve_equation_l1294_129462


namespace inverse_function_sum_l1294_129467

/-- Given two real numbers a and b, and functions f and f_inv defined as follows:
    f(x) = ax + 2b
    f_inv(x) = bx + 2a
    If f and f_inv are true functional inverses, then a + 2b = -3 -/
theorem inverse_function_sum (a b : ℝ) 
  (f : ℝ → ℝ) (f_inv : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x + 2 * b)
  (h2 : ∀ x, f_inv x = b * x + 2 * a)
  (h3 : Function.LeftInverse f_inv f ∧ Function.RightInverse f_inv f) :
  a + 2 * b = -3 := by
  sorry

end inverse_function_sum_l1294_129467


namespace unique_general_term_implies_m_eq_one_third_l1294_129497

/-- Two geometric sequences satisfying given conditions -/
structure GeometricSequences (m : ℝ) :=
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (a_geom : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q)
  (b_geom : ∃ q : ℝ, ∀ n : ℕ, b (n + 1) = b n * q)
  (a_first : a 1 = m)
  (b_minus_a_1 : b 1 - a 1 = 1)
  (b_minus_a_2 : b 2 - a 2 = 2)
  (b_minus_a_3 : b 3 - a 3 = 3)
  (m_pos : m > 0)

/-- The uniqueness of the general term formula for sequence a -/
def uniqueGeneralTerm (m : ℝ) (gs : GeometricSequences m) :=
  ∃! q : ℝ, ∀ n : ℕ, gs.a (n + 1) = gs.a n * q

/-- Main theorem: If the general term formula of a_n is unique, then m = 1/3 -/
theorem unique_general_term_implies_m_eq_one_third (m : ℝ) (gs : GeometricSequences m) :
  uniqueGeneralTerm m gs → m = 1 / 3 := by
  sorry

end unique_general_term_implies_m_eq_one_third_l1294_129497


namespace train_length_l1294_129407

/-- Calculates the length of a train given its speed, time to cross a bridge, and the bridge length. -/
theorem train_length (speed : Real) (time : Real) (bridge_length : Real) :
  speed = 10 → -- 36 kmph converted to m/s
  time = 29.997600191984642 →
  bridge_length = 150 →
  speed * time - bridge_length = 149.97600191984642 := by
  sorry

end train_length_l1294_129407


namespace smallest_x_for_quadratic_inequality_l1294_129454

theorem smallest_x_for_quadratic_inequality :
  ∃ x₀ : ℝ, x₀ = 3 ∧
  (∀ x : ℝ, x^2 - 8*x + 15 ≤ 0 → x ≥ x₀) ∧
  (x₀^2 - 8*x₀ + 15 ≤ 0) := by
  sorry

end smallest_x_for_quadratic_inequality_l1294_129454


namespace sarah_homework_problem_l1294_129430

/-- The total number of problems Sarah has to complete given her homework assignments -/
def total_problems (math_pages reading_pages science_pages : ℕ) 
  (math_problems_per_page reading_problems_per_page science_problems_per_page : ℕ) : ℕ :=
  math_pages * math_problems_per_page + 
  reading_pages * reading_problems_per_page + 
  science_pages * science_problems_per_page

theorem sarah_homework_problem :
  total_problems 4 6 5 4 4 6 = 70 := by
  sorry

end sarah_homework_problem_l1294_129430


namespace benny_turnips_l1294_129476

theorem benny_turnips (melanie_turnips benny_turnips total_turnips : ℕ) : 
  melanie_turnips = 139 → total_turnips = 252 → benny_turnips = total_turnips - melanie_turnips → 
  benny_turnips = 113 := by
  sorry

end benny_turnips_l1294_129476


namespace final_passenger_count_l1294_129495

def bus_passengers (initial : ℕ) (first_stop : ℕ) (off_other : ℕ) (on_other : ℕ) : ℕ :=
  initial + first_stop - off_other + on_other

theorem final_passenger_count :
  bus_passengers 50 16 22 5 = 49 := by
  sorry

end final_passenger_count_l1294_129495


namespace smallest_product_of_factors_l1294_129466

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem smallest_product_of_factors (a b : ℕ) : 
  a ≠ b → 
  a > 0 → 
  b > 0 → 
  is_factor a 48 → 
  is_factor b 48 → 
  ¬ is_factor (a * b) 48 → 
  (∀ (x y : ℕ), x ≠ y → x > 0 → y > 0 → is_factor x 48 → is_factor y 48 → 
    ¬ is_factor (x * y) 48 → a * b ≤ x * y) → 
  a * b = 18 :=
sorry

end smallest_product_of_factors_l1294_129466


namespace money_distribution_l1294_129469

theorem money_distribution (total : ℕ) (p q r : ℕ) : 
  total = 9000 →
  p + q + r = total →
  r = 2 * (p + q) / 3 →
  r = 3600 := by
sorry

end money_distribution_l1294_129469


namespace ad_probability_is_one_third_l1294_129428

/-- The duration of advertisements per hour in minutes -/
def ad_duration : ℕ := 20

/-- The total duration of an hour in minutes -/
def hour_duration : ℕ := 60

/-- The probability of seeing an advertisement when turning on the TV -/
def ad_probability : ℚ := ad_duration / hour_duration

theorem ad_probability_is_one_third : ad_probability = 1/3 := by
  sorry

end ad_probability_is_one_third_l1294_129428


namespace max_revenue_at_22_l1294_129445

def cinema_revenue (price : ℕ) : ℤ :=
  if price ≤ 10 then
    1000 * price - 5750
  else
    -30 * price * price + 1300 * price - 5750

def valid_price (price : ℕ) : Prop :=
  (6 ≤ price) ∧ (price ≤ 38)

theorem max_revenue_at_22 :
  (∀ p, valid_price p → cinema_revenue p ≤ cinema_revenue 22) ∧
  cinema_revenue 22 = 8330 :=
sorry

end max_revenue_at_22_l1294_129445


namespace quadrilaterals_in_100gon_l1294_129436

/-- A regular polygon with n vertices -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A coloring of vertices of a regular polygon -/
def Coloring (n : ℕ) := Fin n → Bool

/-- A convex quadrilateral formed by four vertices of a regular polygon -/
structure Quadrilateral (n : ℕ) where
  v1 : Fin n
  v2 : Fin n
  v3 : Fin n
  v4 : Fin n

/-- Check if two quadrilaterals are disjoint -/
def are_disjoint (n : ℕ) (q1 q2 : Quadrilateral n) : Prop :=
  q1.v1 ≠ q2.v1 ∧ q1.v1 ≠ q2.v2 ∧ q1.v1 ≠ q2.v3 ∧ q1.v1 ≠ q2.v4 ∧
  q1.v2 ≠ q2.v1 ∧ q1.v2 ≠ q2.v2 ∧ q1.v2 ≠ q2.v3 ∧ q1.v2 ≠ q2.v4 ∧
  q1.v3 ≠ q2.v1 ∧ q1.v3 ≠ q2.v2 ∧ q1.v3 ≠ q2.v3 ∧ q1.v3 ≠ q2.v4 ∧
  q1.v4 ≠ q2.v1 ∧ q1.v4 ≠ q2.v2 ∧ q1.v4 ≠ q2.v3 ∧ q1.v4 ≠ q2.v4

/-- Check if a quadrilateral has three corners of one color and one of the other -/
def has_three_one_coloring (n : ℕ) (q : Quadrilateral n) (c : Coloring n) : Prop :=
  (c q.v1 = c q.v2 ∧ c q.v2 = c q.v3 ∧ c q.v3 ≠ c q.v4) ∨
  (c q.v1 = c q.v2 ∧ c q.v2 = c q.v4 ∧ c q.v4 ≠ c q.v3) ∨
  (c q.v1 = c q.v3 ∧ c q.v3 = c q.v4 ∧ c q.v4 ≠ c q.v2) ∨
  (c q.v2 = c q.v3 ∧ c q.v3 = c q.v4 ∧ c q.v4 ≠ c q.v1)

/-- The main theorem -/
theorem quadrilaterals_in_100gon :
  ∃ (p : RegularPolygon 100) (c : Coloring 100) (qs : Fin 24 → Quadrilateral 100),
    (∀ i : Fin 100, c i = true → (∃ j : Fin 41, true)) ∧  -- 41 black vertices
    (∀ i : Fin 100, c i = false → (∃ j : Fin 59, true)) ∧  -- 59 white vertices
    (∀ i j : Fin 24, i ≠ j → are_disjoint 100 (qs i) (qs j)) ∧
    (∀ i : Fin 24, has_three_one_coloring 100 (qs i) c) :=
by sorry

end quadrilaterals_in_100gon_l1294_129436


namespace concentric_circles_angle_l1294_129494

theorem concentric_circles_angle (r₁ r₂ : ℝ) (α : ℝ) :
  r₁ = 1 →
  r₂ = 2 →
  (((360 - α) / 360 * π * r₁^2) + (α / 360 * π * r₂^2) - (α / 360 * π * r₁^2)) = (1/3) * (π * r₂^2) →
  α = 60 := by
sorry

end concentric_circles_angle_l1294_129494


namespace order_of_magnitude_l1294_129449

theorem order_of_magnitude (a b c : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > 0)
  (x : ℝ) (hx : x = Real.sqrt (a^2 + (b+c)^2))
  (y : ℝ) (hy : y = Real.sqrt (b^2 + (c+a)^2))
  (z : ℝ) (hz : z = Real.sqrt (c^2 + (a+b)^2)) :
  z > y ∧ y > x := by
  sorry

end order_of_magnitude_l1294_129449


namespace debbys_percentage_share_l1294_129447

theorem debbys_percentage_share (total : ℝ) (maggies_share : ℝ) 
  (h1 : total = 6000)
  (h2 : maggies_share = 4500) :
  (total - maggies_share) / total * 100 = 25 := by
sorry

end debbys_percentage_share_l1294_129447


namespace purely_imaginary_condition_fourth_quadrant_condition_l1294_129491

def z (a : ℝ) : ℂ := Complex.mk (a^2 - 7*a + 6) (a^2 - 5*a - 6)

theorem purely_imaginary_condition (a : ℝ) :
  z a = Complex.I * (z a).im → a = 1 := by sorry

theorem fourth_quadrant_condition (a : ℝ) :
  (z a).re > 0 ∧ (z a).im < 0 → a > -1 ∧ a < 1 := by sorry

end purely_imaginary_condition_fourth_quadrant_condition_l1294_129491


namespace test_coincidences_l1294_129405

theorem test_coincidences (n : ℕ) (p_vasya p_misha : ℝ) 
  (hn : n = 20) 
  (hv : p_vasya = 6 / 20) 
  (hm : p_misha = 8 / 20) : 
  n * (p_vasya * p_misha + (1 - p_vasya) * (1 - p_misha)) = 10.8 := by
  sorry

end test_coincidences_l1294_129405


namespace equation_is_linear_one_var_l1294_129443

/-- Predicate to check if an expression is linear in one variable -/
def IsLinearOneVar (e : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, e x = a * x + b ∧ a ≠ 0

/-- The specific equation we're checking -/
def equation (x : ℝ) : ℝ := 3 - 2*x

/-- Theorem stating that our equation is linear in one variable -/
theorem equation_is_linear_one_var : IsLinearOneVar equation :=
sorry

end equation_is_linear_one_var_l1294_129443


namespace museum_visitors_l1294_129417

theorem museum_visitors (V : ℕ) 
  (h1 : V = (3/4 : ℚ) * V + 130)
  (h2 : ∃ E U : ℕ, E = U ∧ E = (3/4 : ℚ) * V) : 
  V = 520 := by
  sorry

end museum_visitors_l1294_129417


namespace sqrt_720_simplified_l1294_129440

theorem sqrt_720_simplified : Real.sqrt 720 = 12 * Real.sqrt 5 := by
  sorry

end sqrt_720_simplified_l1294_129440


namespace quadratic_inequality_range_l1294_129460

theorem quadratic_inequality_range (p : ℝ) (α : ℝ) (h1 : p = 4 * (Real.sin α) ^ 4)
  (h2 : α ∈ Set.Icc (π / 6) (5 * π / 6)) :
  (∀ x : ℝ, x^2 + p*x + 1 > 2*x + p) ↔ (∀ x : ℝ, x > 1 ∨ x < -3) := by sorry

end quadratic_inequality_range_l1294_129460


namespace smallest_integer_divisible_l1294_129499

theorem smallest_integer_divisible (x : ℤ) : x = 36629 ↔ 
  (∀ y : ℤ, y < x → ¬(∃ k₁ k₂ k₃ k₄ : ℤ, 
    2 * y + 2 = 33 * k₁ ∧ 
    2 * y + 2 = 44 * k₂ ∧ 
    2 * y + 2 = 55 * k₃ ∧ 
    2 * y + 2 = 666 * k₄)) ∧ 
  (∃ k₁ k₂ k₃ k₄ : ℤ, 
    2 * x + 2 = 33 * k₁ ∧ 
    2 * x + 2 = 44 * k₂ ∧ 
    2 * x + 2 = 55 * k₃ ∧ 
    2 * x + 2 = 666 * k₄) :=
by sorry

end smallest_integer_divisible_l1294_129499


namespace count_divisors_3240_multiple_of_three_l1294_129420

/-- The number of positive divisors of 3240 that are multiples of 3 -/
def num_divisors_multiple_of_three : ℕ := 32

/-- The prime factorization of 3240 -/
def factorization_3240 : List (ℕ × ℕ) := [(2, 3), (3, 4), (5, 1)]

/-- A function to count the number of positive divisors of 3240 that are multiples of 3 -/
def count_divisors_multiple_of_three (factorization : List (ℕ × ℕ)) : ℕ :=
  sorry

theorem count_divisors_3240_multiple_of_three :
  count_divisors_multiple_of_three factorization_3240 = num_divisors_multiple_of_three :=
sorry

end count_divisors_3240_multiple_of_three_l1294_129420


namespace cylinder_in_hemisphere_height_l1294_129406

theorem cylinder_in_hemisphere_height (r c h : ℝ) : 
  r > 0 ∧ c > 0 ∧ r > c ∧ r = 8 ∧ c = 3 → h = Real.sqrt 55 := by
  sorry

end cylinder_in_hemisphere_height_l1294_129406


namespace exactly_one_absent_probability_l1294_129457

theorem exactly_one_absent_probability (p_absent : ℝ) (h1 : p_absent = 1 / 20) :
  let p_present := 1 - p_absent
  2 * p_absent * p_present = 19 / 200 := by
  sorry

end exactly_one_absent_probability_l1294_129457


namespace probability_both_selected_l1294_129474

theorem probability_both_selected (prob_X prob_Y prob_both : ℚ) : 
  prob_X = 1/7 → prob_Y = 2/9 → prob_both = prob_X * prob_Y → prob_both = 2/63 := by
  sorry

end probability_both_selected_l1294_129474


namespace candidate_vote_percentage_l1294_129446

/-- Proves that given a total of 8000 votes and a loss margin of 4000 votes,
    the percentage of votes received by the losing candidate is 25%. -/
theorem candidate_vote_percentage
  (total_votes : ℕ)
  (loss_margin : ℕ)
  (h_total : total_votes = 8000)
  (h_margin : loss_margin = 4000) :
  (total_votes - loss_margin) / total_votes * 100 = 25 := by
  sorry

end candidate_vote_percentage_l1294_129446


namespace origin_outside_circle_l1294_129455

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y a : ℝ) : ℝ := x^2 + y^2 + 2*a*x + 2*y + (a-1)^2

/-- Predicate to check if a point (x, y) is outside the circle -/
def is_outside_circle (x y a : ℝ) : Prop := circle_equation x y a > 0

theorem origin_outside_circle (a : ℝ) (h : 0 < a ∧ a < 1) : 
  is_outside_circle 0 0 a :=
sorry

end origin_outside_circle_l1294_129455


namespace athlete_score_comparison_l1294_129419

theorem athlete_score_comparison 
  (p₁ p₂ p₃ : ℝ) 
  (hp₁ : p₁ > 0) 
  (hp₂ : p₂ > 0) 
  (hp₃ : p₃ > 0) : 
  (16/25) * p₁ + (9/25) * p₂ + (4/15) * p₃ > 
  (16/25) * p₁ + (1/4) * p₂ + (27/128) * p₃ :=
sorry

end athlete_score_comparison_l1294_129419


namespace modular_inverse_15_l1294_129423

theorem modular_inverse_15 :
  (¬ ∃ x : ℤ, (15 * x) % 1105 = 1) ∧
  (∃ x : ℤ, (15 * x) % 221 = 1) ∧
  ((15 * 59) % 221 = 1) := by
sorry

end modular_inverse_15_l1294_129423


namespace negative_roots_equation_reciprocal_roots_equation_l1294_129456

-- Part 1
theorem negative_roots_equation (r1 r2 : ℝ) :
  r1^2 + 3*r1 - 2 = 0 ∧ r2^2 + 3*r2 - 2 = 0 →
  (-r1)^2 - 3*(-r1) - 2 = 0 ∧ (-r2)^2 - 3*(-r2) - 2 = 0 := by sorry

-- Part 2
theorem reciprocal_roots_equation (a b c r1 r2 : ℝ) :
  a ≠ 0 ∧ r1 ≠ r2 ∧ r1 ≠ 0 ∧ r2 ≠ 0 ∧
  a*r1^2 - b*r1 + c = 0 ∧ a*r2^2 - b*r2 + c = 0 →
  c*(1/r1)^2 - b*(1/r1) + a = 0 ∧ c*(1/r2)^2 - b*(1/r2) + a = 0 := by sorry

end negative_roots_equation_reciprocal_roots_equation_l1294_129456


namespace factorization_of_2a_squared_minus_8_l1294_129471

theorem factorization_of_2a_squared_minus_8 (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) := by
  sorry

end factorization_of_2a_squared_minus_8_l1294_129471


namespace circle_center_and_radius_l1294_129486

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Theorem statement
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (3, 0) ∧ 
    radius = 3 ∧
    ∀ (x y : ℝ), circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end circle_center_and_radius_l1294_129486
