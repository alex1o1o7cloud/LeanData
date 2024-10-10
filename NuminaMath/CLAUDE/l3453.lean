import Mathlib

namespace line_through_points_equation_l3453_345343

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  slope : ℝ
  yIntercept : ℝ

-- Define the two given points
def pointA : Point2D := { x := 3, y := 0 }
def pointB : Point2D := { x := -3, y := 0 }

-- Theorem: The line passing through pointA and pointB has the equation y = 0
theorem line_through_points_equation :
  ∃ (l : Line2D), l.slope = 0 ∧ l.yIntercept = 0 ∧
  (l.slope * pointA.x + l.yIntercept = pointA.y) ∧
  (l.slope * pointB.x + l.yIntercept = pointB.y) :=
by sorry

end line_through_points_equation_l3453_345343


namespace fourth_minus_third_tiles_l3453_345306

/-- The side length of the n-th square in the sequence -/
def side_length (n : ℕ) : ℕ := n^2

/-- The number of tiles in the n-th square -/
def tiles (n : ℕ) : ℕ := (side_length n)^2

theorem fourth_minus_third_tiles : tiles 4 - tiles 3 = 175 := by
  sorry

end fourth_minus_third_tiles_l3453_345306


namespace hyperbola_center_l3453_345350

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 + 54 * x - 16 * y^2 - 128 * y - 200 = 0

/-- The center of a hyperbola -/
def is_center (h k : ℝ) : Prop :=
  ∀ x y : ℝ, hyperbola_equation x y ↔ 
    ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ((x - h)^2 / a^2) - ((y - k)^2 / b^2) = 1

/-- Theorem: The center of the given hyperbola is (-3, -4) -/
theorem hyperbola_center : is_center (-3) (-4) :=
sorry

end hyperbola_center_l3453_345350


namespace stock_price_change_l3453_345381

def total_stocks : ℕ := 8000

theorem stock_price_change (higher lower : ℕ) 
  (h1 : higher + lower = total_stocks)
  (h2 : higher = lower + lower / 2) :
  higher = 4800 := by
sorry

end stock_price_change_l3453_345381


namespace riverside_total_multiple_of_five_l3453_345382

/-- Represents the population of animals and people in Riverside --/
structure Riverside where
  people : ℕ
  horses : ℕ
  sheep : ℕ
  cows : ℕ
  ducks : ℕ

/-- The conditions given in the problem --/
def valid_riverside (r : Riverside) : Prop :=
  r.people = 5 * r.horses ∧
  r.sheep = 6 * r.cows ∧
  r.ducks = 4 * r.people ∧
  r.sheep * 2 = r.ducks

/-- The theorem states that the total population in a valid Riverside setup is always a multiple of 5 --/
theorem riverside_total_multiple_of_five (r : Riverside) (h : valid_riverside r) :
  ∃ k : ℕ, r.people + r.horses + r.sheep + r.cows + r.ducks = 5 * k :=
sorry

end riverside_total_multiple_of_five_l3453_345382


namespace intersection_with_complement_l3453_345332

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3, 4}

theorem intersection_with_complement : A ∩ (U \ B) = {1} := by
  sorry

end intersection_with_complement_l3453_345332


namespace original_data_set_properties_l3453_345373

/-- Represents a data set with its average and variance -/
structure DataSet where
  average : ℝ
  variance : ℝ

/-- The transformation applied to the original data set -/
def decrease_by_80 (d : DataSet) : DataSet :=
  { average := d.average - 80, variance := d.variance }

/-- Theorem stating the relationship between the original and transformed data sets -/
theorem original_data_set_properties (transformed : DataSet)
  (h1 : transformed = decrease_by_80 { average := 81.2, variance := 4.4 })
  (h2 : transformed.average = 1.2)
  (h3 : transformed.variance = 4.4) :
  ∃ (original : DataSet), original.average = 81.2 ∧ original.variance = 4.4 :=
sorry

end original_data_set_properties_l3453_345373


namespace baseball_cards_total_l3453_345301

def total_baseball_cards (carlos_cards matias_cards jorge_cards : ℕ) : ℕ :=
  carlos_cards + matias_cards + jorge_cards

theorem baseball_cards_total (carlos_cards matias_cards jorge_cards : ℕ) 
  (h1 : carlos_cards = 20)
  (h2 : matias_cards = carlos_cards - 6)
  (h3 : jorge_cards = matias_cards) :
  total_baseball_cards carlos_cards matias_cards jorge_cards = 48 :=
by
  sorry

end baseball_cards_total_l3453_345301


namespace kohens_apples_l3453_345333

/-- Kohen's Apple Business Theorem -/
theorem kohens_apples (boxes : ℕ) (apples_per_box : ℕ) (sold_fraction : ℚ) 
  (h1 : boxes = 10)
  (h2 : apples_per_box = 300)
  (h3 : sold_fraction = 3/4) : 
  boxes * apples_per_box - (sold_fraction * (boxes * apples_per_box)).num = 750 := by
  sorry

end kohens_apples_l3453_345333


namespace bob_corn_harvest_l3453_345325

/-- Calculates the number of whole bushels of corn harvested given the number of rows, stalks per row, and stalks per bushel. -/
def cornHarvest (rows : ℕ) (stalksPerRow : ℕ) (stalksPerBushel : ℕ) : ℕ :=
  (rows * stalksPerRow) / stalksPerBushel

theorem bob_corn_harvest :
  cornHarvest 7 92 9 = 71 := by
  sorry

end bob_corn_harvest_l3453_345325


namespace circle_equation_through_points_l3453_345398

theorem circle_equation_through_points :
  let equation (x y : ℝ) := x^2 + y^2 - 4*x - 6*y
  ∀ (x y : ℝ), equation x y = 0 →
    (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) :=
by sorry

end circle_equation_through_points_l3453_345398


namespace cylinder_surface_area_and_volume_l3453_345356

/-- Represents a right circular cylinder -/
structure RightCircularCylinder where
  radius : ℝ
  height : ℝ

/-- Properties of the cylinder -/
def CylinderProperties (c : RightCircularCylinder) : Prop :=
  let lateral_area := 2 * Real.pi * c.radius * c.height
  let base_area := Real.pi * c.radius ^ 2
  lateral_area / base_area = 5 / 3 ∧
  (4 * c.radius ^ 2 + c.height ^ 2) = 39 ^ 2

/-- Theorem statement -/
theorem cylinder_surface_area_and_volume 
  (c : RightCircularCylinder) 
  (h : CylinderProperties c) : 
  (2 * Real.pi * c.radius * c.height + 2 * Real.pi * c.radius ^ 2 = 1188 * Real.pi) ∧
  (Real.pi * c.radius ^ 2 * c.height = 4860 * Real.pi) := by
  sorry

end cylinder_surface_area_and_volume_l3453_345356


namespace sock_combinations_l3453_345310

theorem sock_combinations (n : ℕ) (k : ℕ) (h1 : n = 9) (h2 : k = 2) :
  Nat.choose n k = 36 := by
  sorry

end sock_combinations_l3453_345310


namespace wednesday_pages_proof_l3453_345302

def total_pages : ℕ := 158
def monday_pages : ℕ := 23
def tuesday_pages : ℕ := 38
def thursday_pages : ℕ := 12

def friday_pages : ℕ := 2 * thursday_pages

theorem wednesday_pages_proof :
  total_pages - (monday_pages + tuesday_pages + thursday_pages + friday_pages) = 61 := by
  sorry

end wednesday_pages_proof_l3453_345302


namespace collinear_vectors_l3453_345303

/-- Given vectors a and b in ℝ², if ma + nb is collinear with a - 2b, then m/n = -1/2 -/
theorem collinear_vectors (a b : ℝ × ℝ) (m n : ℝ) 
  (h1 : a = (2, 3))
  (h2 : b = (-1, 2))
  (h3 : ∃ (k : ℝ), k ≠ 0 ∧ (m • a + n • b) = k • (a - 2 • b)) :
  m / n = -1 / 2 := by
  sorry

end collinear_vectors_l3453_345303


namespace arithmetic_operations_l3453_345344

theorem arithmetic_operations (a b : ℝ) : 
  (a ≠ 0 → a / a = 1) ∧ 
  (b ≠ 0 → a / b = a * (1 / b)) ∧ 
  (a * 1 = a) ∧ 
  (0 / b = 0) :=
sorry

#check arithmetic_operations

end arithmetic_operations_l3453_345344


namespace concentric_circles_radii_difference_l3453_345351

theorem concentric_circles_radii_difference
  (r R : ℝ)
  (h_positive : r > 0)
  (h_ratio : π * R^2 = 4 * π * r^2) :
  R - r = r :=
sorry

end concentric_circles_radii_difference_l3453_345351


namespace family_size_problem_l3453_345376

theorem family_size_problem (avg_age_before avg_age_now baby_age : ℝ) 
  (h1 : avg_age_before = 17)
  (h2 : avg_age_now = 17)
  (h3 : baby_age = 2) : 
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℝ) * avg_age_before + (n : ℝ) * 3 + baby_age = (n + 1 : ℝ) * avg_age_now ∧
    n = 5 := by
  sorry

end family_size_problem_l3453_345376


namespace remainder_of_binary_div_four_l3453_345315

def binary_number : ℕ := 110110111101

theorem remainder_of_binary_div_four :
  binary_number % 4 = 1 := by
  sorry

end remainder_of_binary_div_four_l3453_345315


namespace square_perimeter_inequality_l3453_345311

theorem square_perimeter_inequality (t₁ t₂ t₃ t₄ k₁ k₂ k₃ k₄ : ℝ) 
  (h₁ : t₁ > 0) (h₂ : t₂ > 0) (h₃ : t₃ > 0) (h₄ : t₄ > 0)
  (hk₁ : k₁ = 4 * Real.sqrt t₁)
  (hk₂ : k₂ = 4 * Real.sqrt t₂)
  (hk₃ : k₃ = 4 * Real.sqrt t₃)
  (hk₄ : k₄ = 4 * Real.sqrt t₄)
  (ht : t₁ + t₂ + t₃ = t₄) :
  k₁ + k₂ + k₃ ≤ k₄ * Real.sqrt 3 := by
  sorry

end square_perimeter_inequality_l3453_345311


namespace gcd_lcm_equalities_l3453_345328

/-- Define * as the greatest common divisor operation -/
def gcd_op (a b : ℕ) : ℕ := Nat.gcd a b

/-- Define ∘ as the least common multiple operation -/
def lcm_op (a b : ℕ) : ℕ := Nat.lcm a b

/-- The main theorem stating the equalities for gcd and lcm operations -/
theorem gcd_lcm_equalities (a b c : ℕ) :
  (gcd_op a (lcm_op b c) = lcm_op (gcd_op a b) (gcd_op a c)) ∧
  (lcm_op a (gcd_op b c) = gcd_op (lcm_op a b) (lcm_op a c)) := by
  sorry

end gcd_lcm_equalities_l3453_345328


namespace exactly_three_numbers_l3453_345346

/-- A two-digit number -/
def TwoDigitNumber := { n : ℕ // 10 ≤ n ∧ n < 100 }

/-- The tens digit of a two-digit number -/
def tens_digit (n : TwoDigitNumber) : ℕ := n.val / 10

/-- The units digit of a two-digit number -/
def units_digit (n : TwoDigitNumber) : ℕ := n.val % 10

/-- The sum of digits of a two-digit number -/
def sum_of_digits (n : TwoDigitNumber) : ℕ := tens_digit n + units_digit n

/-- Predicate for numbers satisfying the given conditions -/
def satisfies_conditions (n : TwoDigitNumber) : Prop :=
  (n.val - sum_of_digits n) % 10 = 2 ∧ n.val % 3 = 0

/-- The main theorem stating there are exactly 3 numbers satisfying the conditions -/
theorem exactly_three_numbers :
  ∃! (s : Finset TwoDigitNumber), (∀ n ∈ s, satisfies_conditions n) ∧ s.card = 3 :=
sorry

end exactly_three_numbers_l3453_345346


namespace wire_length_proof_l3453_345379

theorem wire_length_proof (piece1 piece2 : ℝ) : 
  piece1 = 14 → 
  piece2 = 16 → 
  piece2 = piece1 + 2 → 
  piece1 + piece2 = 30 :=
by
  sorry

end wire_length_proof_l3453_345379


namespace similar_triangle_perimeter_l3453_345367

/-- Represents a triangle with side lengths -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- Checks if a triangle is isosceles -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.side1 = t.side2 ∨ t.side1 = t.side3 ∨ t.side2 = t.side3

/-- Calculates the perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℝ :=
  t.side1 + t.side2 + t.side3

/-- Checks if two triangles are similar -/
def areTrianglesSimilar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧
    t2.side1 = k * t1.side1 ∧
    t2.side2 = k * t1.side2 ∧
    t2.side3 = k * t1.side3

theorem similar_triangle_perimeter
  (small large : Triangle)
  (h_small_isosceles : small.isIsosceles)
  (h_small_sides : small.side1 = 12 ∧ small.side2 = 24)
  (h_similar : areTrianglesSimilar small large)
  (h_large_shortest : min large.side1 (min large.side2 large.side3) = 30) :
  large.perimeter = 150 := by
  sorry

end similar_triangle_perimeter_l3453_345367


namespace zhang_slower_than_li_l3453_345385

theorem zhang_slower_than_li :
  let zhang_efficiency : ℚ := 5 / 8
  let li_efficiency : ℚ := 3 / 4
  zhang_efficiency < li_efficiency :=
by
  sorry

end zhang_slower_than_li_l3453_345385


namespace cupcakes_left_over_l3453_345321

/-- The number of cupcakes Quinton brought to school -/
def total_cupcakes : ℕ := 80

/-- The number of students in Ms. Delmont's class -/
def ms_delmont_students : ℕ := 18

/-- The number of students in Mrs. Donnelly's class -/
def mrs_donnelly_students : ℕ := 16

/-- The number of school custodians -/
def custodians : ℕ := 3

/-- The number of Quinton's favorite teachers -/
def favorite_teachers : ℕ := 5

/-- The number of other classmates who received cupcakes -/
def other_classmates : ℕ := 10

/-- The number of cupcakes given to each favorite teacher -/
def cupcakes_per_favorite_teacher : ℕ := 2

/-- The total number of cupcakes given away -/
def cupcakes_given_away : ℕ :=
  ms_delmont_students + mrs_donnelly_students + 2 + 1 + 1 + custodians +
  (favorite_teachers * cupcakes_per_favorite_teacher) + other_classmates

/-- Theorem stating the number of cupcakes Quinton has left over -/
theorem cupcakes_left_over : total_cupcakes - cupcakes_given_away = 19 := by
  sorry

end cupcakes_left_over_l3453_345321


namespace percent_of_y_l3453_345354

theorem percent_of_y (y : ℝ) (h : y > 0) : ((8 * y) / 20 + (3 * y) / 10) / y = 0.7 := by
  sorry

end percent_of_y_l3453_345354


namespace arctan_equation_solution_l3453_345340

theorem arctan_equation_solution :
  ∃ x : ℝ, Real.arctan (2 / x) + Real.arctan (3 / x^3) = π / 4 := by
  sorry

end arctan_equation_solution_l3453_345340


namespace cage_cost_l3453_345345

/-- The cost of the cage given the payment and change -/
theorem cage_cost (bill : ℝ) (change : ℝ) (h1 : bill = 20) (h2 : change = 0.26) :
  bill - change = 19.74 := by
  sorry

end cage_cost_l3453_345345


namespace total_school_supplies_cost_l3453_345352

-- Define the quantities and prices
def haley_paper_reams : ℕ := 2
def haley_paper_price : ℚ := 3.5
def sister_paper_reams : ℕ := 3
def sister_paper_price : ℚ := 4.25
def haley_pens : ℕ := 5
def haley_pen_price : ℚ := 1.25
def sister_pens : ℕ := 8
def sister_pen_price : ℚ := 1.5

-- Define the theorem
theorem total_school_supplies_cost :
  (haley_paper_reams : ℚ) * haley_paper_price +
  (sister_paper_reams : ℚ) * sister_paper_price +
  (haley_pens : ℚ) * haley_pen_price +
  (sister_pens : ℚ) * sister_pen_price = 38 :=
by sorry

end total_school_supplies_cost_l3453_345352


namespace recipe_total_cups_l3453_345371

/-- Represents the ratio of ingredients in the recipe -/
structure RecipeRatio where
  butter : ℕ
  flour : ℕ
  sugar : ℕ

/-- Calculates the total cups of ingredients given a recipe ratio and the amount of flour used -/
def totalCups (ratio : RecipeRatio) (flourUsed : ℕ) : ℕ :=
  let partSize := flourUsed / ratio.flour
  partSize * (ratio.butter + ratio.flour + ratio.sugar)

/-- Theorem stating that for the given recipe ratio and flour amount, the total cups is 20 -/
theorem recipe_total_cups :
  let ratio : RecipeRatio := ⟨2, 3, 5⟩
  let flourUsed : ℕ := 6
  totalCups ratio flourUsed = 20 := by
  sorry


end recipe_total_cups_l3453_345371


namespace mean_proportional_problem_l3453_345313

theorem mean_proportional_problem (x : ℝ) :
  (Real.sqrt (x * 100) = 90.5) → x = 81.9025 := by
  sorry

end mean_proportional_problem_l3453_345313


namespace cube_volume_problem_l3453_345399

theorem cube_volume_problem (V₁ : ℝ) (V₂ : ℝ) (A₁ : ℝ) (A₂ : ℝ) (s₁ : ℝ) (s₂ : ℝ) :
  V₁ = 8 →
  s₁ ^ 3 = V₁ →
  A₁ = 6 * s₁ ^ 2 →
  A₂ = 3 * A₁ →
  A₂ = 6 * s₂ ^ 2 →
  V₂ = s₂ ^ 3 →
  V₂ = 24 * Real.sqrt 3 :=
by sorry

end cube_volume_problem_l3453_345399


namespace min_value_T_l3453_345355

/-- Given a quadratic inequality with no real solutions and constraints on its coefficients,
    prove that a certain expression T has a minimum value of 4. -/
theorem min_value_T (a b c : ℝ) : 
  (∀ x, (1/a) * x^2 + b*x + c ≥ 0) →  -- No real solutions to the inequality
  a > 0 →
  a * b > 1 → 
  (∀ T, T = 1/(2*(a*b-1)) + (a*(b+2*c))/(a*b-1) → T ≥ 4) ∧ 
  (∃ T, T = 1/(2*(a*b-1)) + (a*(b+2*c))/(a*b-1) ∧ T = 4) :=
by sorry


end min_value_T_l3453_345355


namespace tom_video_game_spending_l3453_345390

/-- The total amount Tom spent on new video games --/
def total_spent (batman_price superman_price discount_rate tax_rate : ℚ) : ℚ :=
  let discounted_batman := batman_price * (1 - discount_rate)
  let discounted_superman := superman_price * (1 - discount_rate)
  let total_before_tax := discounted_batman + discounted_superman
  total_before_tax * (1 + tax_rate)

/-- Theorem stating the total amount Tom spent on new video games --/
theorem tom_video_game_spending :
  total_spent 13.60 5.06 0.20 0.08 = 16.12 := by
  sorry

#eval total_spent 13.60 5.06 0.20 0.08

end tom_video_game_spending_l3453_345390


namespace smallest_number_in_sample_l3453_345314

/-- Systematic sampling function that returns the smallest number given the parameters -/
def systematicSampling (totalSchools : ℕ) (sampleSize : ℕ) (highestDrawn : ℕ) : ℕ :=
  let interval := totalSchools / sampleSize
  highestDrawn - (sampleSize - 1) * interval

/-- Theorem stating the smallest number drawn in the specific scenario -/
theorem smallest_number_in_sample (totalSchools : ℕ) (sampleSize : ℕ) (highestDrawn : ℕ) 
    (h1 : totalSchools = 32)
    (h2 : sampleSize = 8)
    (h3 : highestDrawn = 31) :
  systematicSampling totalSchools sampleSize highestDrawn = 3 := by
  sorry

#eval systematicSampling 32 8 31

end smallest_number_in_sample_l3453_345314


namespace equation_solution_l3453_345320

theorem equation_solution (x : ℝ) :
  (1 : ℝ) = 1 / (4 * x^2 + 2 * x + 1) →
  x = 0 ∨ x = -1/2 := by
  sorry

end equation_solution_l3453_345320


namespace rain_probability_l3453_345366

theorem rain_probability (p : ℝ) (h : p = 1 / 2) : 
  1 - (1 - p)^4 = 15 / 16 := by
  sorry

end rain_probability_l3453_345366


namespace count_tuples_divisible_sum_l3453_345357

theorem count_tuples_divisible_sum : 
  let n := 2012
  let f : Fin n → ℕ → ℕ := fun i x => (i.val + 1) * x
  (Finset.univ.filter (fun t : Fin n → Fin n => 
    (Finset.sum Finset.univ (fun i => f i (t i).val)) % n = 0)).card = n^(n-1) := by
  sorry

end count_tuples_divisible_sum_l3453_345357


namespace x_eq_one_sufficient_not_necessary_for_x_squared_eq_one_l3453_345305

theorem x_eq_one_sufficient_not_necessary_for_x_squared_eq_one :
  (∃ x : ℝ, x = 1 → x^2 = 1) ∧
  (∃ x : ℝ, x^2 = 1 ∧ x ≠ 1) := by
  sorry

end x_eq_one_sufficient_not_necessary_for_x_squared_eq_one_l3453_345305


namespace midpoint_coord_sum_l3453_345387

/-- The sum of the coordinates of the midpoint of a line segment with endpoints (10, 3) and (-4, -7) is 1 -/
theorem midpoint_coord_sum : 
  let x1 : ℝ := 10
  let y1 : ℝ := 3
  let x2 : ℝ := -4
  let y2 : ℝ := -7
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x + midpoint_y = 1 := by sorry

end midpoint_coord_sum_l3453_345387


namespace ellipse_equation_l3453_345361

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  eccentricity : ℝ
  perimeter_triangle : ℝ
  h1 : a > b
  h2 : b > 0
  h3 : eccentricity = Real.sqrt 3 / 3
  h4 : perimeter_triangle = 4 * Real.sqrt 3

/-- The theorem stating that an ellipse with the given properties has the equation x²/3 + y²/2 = 1 -/
theorem ellipse_equation (C : Ellipse) : 
  C.a = Real.sqrt 3 ∧ C.b = Real.sqrt 2 :=
sorry

end ellipse_equation_l3453_345361


namespace binomial_distribution_unique_parameters_l3453_345347

/-- A random variable following a binomial distribution B(n,p) -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (X : BinomialRV) : ℝ := X.n * X.p

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_distribution_unique_parameters
  (X : BinomialRV)
  (h_expectation : expectation X = 1.6)
  (h_variance : variance X = 1.28) :
  X.n = 8 ∧ X.p = 0.2 := by sorry

end binomial_distribution_unique_parameters_l3453_345347


namespace manuscript_typing_cost_is_1400_l3453_345360

/-- Calculates the total cost of typing a manuscript with given parameters. -/
def manuscriptTypingCost (totalPages : ℕ) (firstTypeCost : ℕ) (revisionCost : ℕ) 
  (pagesRevisedOnce : ℕ) (pagesRevisedTwice : ℕ) : ℕ :=
  totalPages * firstTypeCost + 
  pagesRevisedOnce * revisionCost + 
  pagesRevisedTwice * revisionCost * 2

/-- Proves that the total cost of typing the manuscript is $1400. -/
theorem manuscript_typing_cost_is_1400 : 
  manuscriptTypingCost 100 10 5 20 30 = 1400 := by
  sorry

#eval manuscriptTypingCost 100 10 5 20 30

end manuscript_typing_cost_is_1400_l3453_345360


namespace semicircle_area_ratio_l3453_345307

/-- Proves that for a rectangle with sides 8 meters and 12 meters, with semicircles
    drawn on each side (diameters coinciding with the sides), the ratio of the area
    of the large semicircles to the area of the small semicircles is 2.25. -/
theorem semicircle_area_ratio (π : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ)
  (h1 : rectangle_width = 8)
  (h2 : rectangle_length = 12)
  (h3 : π > 0) :
  (2 * π * (rectangle_length / 2)^2 / 2) / (2 * π * (rectangle_width / 2)^2 / 2) = 2.25 :=
by sorry

end semicircle_area_ratio_l3453_345307


namespace arithmetic_mean_of_scores_l3453_345370

def scores : List ℝ := [93, 87, 90, 96, 88, 94]

theorem arithmetic_mean_of_scores :
  (scores.sum / scores.length : ℝ) = 91.333 := by
  sorry

end arithmetic_mean_of_scores_l3453_345370


namespace intersection_sum_l3453_345369

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x + y + 1 = 0
def l₂ (b : ℝ) (x y : ℝ) : Prop := 2 * x - b * y - 1 = 0

-- Theorem statement
theorem intersection_sum (a b : ℝ) : 
  (l₁ a 1 1 ∧ l₂ b 1 1) → a + b = -1 := by
  sorry

end intersection_sum_l3453_345369


namespace locus_of_circle_center_l3453_345368

-- Define the two given circles
def circle1 (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

-- Define the condition for a circle to be tangent to both given circles
def is_tangent_to_both (cx cy r : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ),
    circle1 x1 y1 ∧ circle2 x2 y2 ∧
    (cx - x1)^2 + (cy - y1)^2 = (r + 1)^2 ∧
    (cx - x2)^2 + (cy - y2)^2 = (r + 3)^2

-- State the theorem
theorem locus_of_circle_center :
  ∀ (x y : ℝ), x < 0 →
    (∃ (r : ℝ), is_tangent_to_both x y r) ↔ x^2 - y^2/8 = 1 :=
sorry

end locus_of_circle_center_l3453_345368


namespace smallest_reciprocal_l3453_345318

theorem smallest_reciprocal (a b c d e : ℚ) : 
  a = 1/4 → b = 3/7 → c = -2 → d = 10 → e = 2023 →
  (1/c < 1/a ∧ 1/c < 1/b ∧ 1/c < 1/d ∧ 1/c < 1/e) := by
  sorry

end smallest_reciprocal_l3453_345318


namespace triangle_abc_properties_l3453_345389

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) (m n : ℝ × ℝ) :
  m = (Real.sqrt 3, 1) →
  n = (Real.cos A + 1, Real.sin A) →
  m.1 * n.1 + m.2 * n.2 = 2 + Real.sqrt 3 →
  a = Real.sqrt 3 →
  Real.cos B = Real.sqrt 3 / 3 →
  A = π / 6 ∧ 
  (1 / 2 : ℝ) * a * b * Real.sin C = Real.sqrt 2 / 2 + Real.sqrt 3 :=
by sorry

end triangle_abc_properties_l3453_345389


namespace jeff_truck_count_l3453_345353

/-- The number of trucks Jeff has -/
def num_trucks : ℕ := sorry

/-- The number of cars Jeff has -/
def num_cars : ℕ := sorry

/-- The total number of vehicles Jeff has -/
def total_vehicles : ℕ := 60

theorem jeff_truck_count :
  (num_cars = 2 * num_trucks) ∧
  (num_cars + num_trucks = total_vehicles) →
  num_trucks = 20 := by sorry

end jeff_truck_count_l3453_345353


namespace solution_uniqueness_l3453_345383

theorem solution_uniqueness (x y : ℝ) : x^2 - 2*x + y^2 + 6*y + 10 = 0 → x = 1 ∧ y = -3 := by
  sorry

end solution_uniqueness_l3453_345383


namespace orchard_difference_l3453_345380

/-- Represents the number of trees of each type in an orchard -/
structure Orchard where
  orange : ℕ
  lemon : ℕ
  apple : ℕ
  apricot : ℕ

/-- Calculates the total number of trees in an orchard -/
def totalTrees (o : Orchard) : ℕ :=
  o.orange + o.lemon + o.apple + o.apricot

theorem orchard_difference : 
  let ahmed : Orchard := { orange := 8, lemon := 6, apple := 4, apricot := 0 }
  let hassan : Orchard := { orange := 2, lemon := 5, apple := 1, apricot := 3 }
  totalTrees ahmed - totalTrees hassan = 7 := by
  sorry

end orchard_difference_l3453_345380


namespace functional_equation_solution_l3453_345304

-- Define the function type
def FunctionType (k : ℝ) := {f : ℝ → ℝ // ∀ x, x ∈ Set.Icc (-k) k → f x ∈ Set.Icc 0 k}

-- State the theorem
theorem functional_equation_solution (k : ℝ) (h_k : k > 0) :
  ∀ f : FunctionType k,
    (∀ x y, x ∈ Set.Icc (-k) k → y ∈ Set.Icc (-k) k → x + y ∈ Set.Icc (-k) k →
      (f.val x)^2 + (f.val y)^2 - 2*x*y = k^2 + (f.val (x + y))^2) →
    ∃ a c : ℝ, ∀ x ∈ Set.Icc (-k) k,
      f.val x = Real.sqrt (a * x + c - x^2) ∧
      0 ≤ a * x + c - x^2 ∧
      a * x + c - x^2 ≤ k^2 :=
by
  sorry

end functional_equation_solution_l3453_345304


namespace sqrt_difference_squared_l3453_345377

theorem sqrt_difference_squared : (Real.sqrt 169 - Real.sqrt 25)^2 = 64 := by
  sorry

end sqrt_difference_squared_l3453_345377


namespace initial_red_marbles_l3453_345338

theorem initial_red_marbles (initial_blue : ℕ) (removed_red : ℕ) (total_remaining : ℕ) : 
  initial_blue = 30 →
  removed_red = 3 →
  total_remaining = 35 →
  ∃ initial_red : ℕ, 
    initial_red = 20 ∧ 
    total_remaining = (initial_red - removed_red) + (initial_blue - 4 * removed_red) :=
by sorry

end initial_red_marbles_l3453_345338


namespace seedlings_per_packet_l3453_345365

theorem seedlings_per_packet (total_seedlings : ℕ) (num_packets : ℕ) 
  (h1 : total_seedlings = 420) (h2 : num_packets = 60) :
  total_seedlings / num_packets = 7 := by
  sorry

end seedlings_per_packet_l3453_345365


namespace clock_initial_time_l3453_345312

/-- Represents a time of day with hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : hours < 24 ∧ minutes < 60

/-- Calculates the total minutes from midnight for a given time -/
def totalMinutes (t : Time) : ℕ :=
  t.hours * 60 + t.minutes

/-- Represents the properties of the clock in the problem -/
structure Clock where
  initialTime : Time
  gainPerHour : ℕ
  totalGainBy6PM : ℕ

/-- The theorem to be proved -/
theorem clock_initial_time (c : Clock)
  (morning : c.initialTime.hours < 12)
  (gain_rate : c.gainPerHour = 5)
  (total_gain : c.totalGainBy6PM = 35) :
  c.initialTime.hours = 11 ∧ c.initialTime.minutes = 55 := by
  sorry


end clock_initial_time_l3453_345312


namespace election_winner_percentage_l3453_345339

theorem election_winner_percentage (total_votes : ℕ) (majority : ℕ) : 
  total_votes = 460 → majority = 184 → 
  (70 : ℚ) = (100 * (total_votes / 2 + majority) : ℚ) / total_votes := by
  sorry

end election_winner_percentage_l3453_345339


namespace third_year_cost_l3453_345375

def total_first_year_cost : ℝ := 10000
def tuition_percentage : ℝ := 0.40
def room_and_board_percentage : ℝ := 0.35
def tuition_increase_rate : ℝ := 0.06
def room_and_board_increase_rate : ℝ := 0.03
def initial_financial_aid_percentage : ℝ := 0.25
def financial_aid_increase_rate : ℝ := 0.02

def tuition (year : ℕ) : ℝ :=
  total_first_year_cost * tuition_percentage * (1 + tuition_increase_rate) ^ (year - 1)

def room_and_board (year : ℕ) : ℝ :=
  total_first_year_cost * room_and_board_percentage * (1 + room_and_board_increase_rate) ^ (year - 1)

def textbooks_and_transportation : ℝ :=
  total_first_year_cost * (1 - tuition_percentage - room_and_board_percentage)

def financial_aid (year : ℕ) : ℝ :=
  tuition year * (initial_financial_aid_percentage + financial_aid_increase_rate * (year - 1))

def total_cost (year : ℕ) : ℝ :=
  tuition year + room_and_board year + textbooks_and_transportation - financial_aid year

theorem third_year_cost :
  total_cost 3 = 9404.17 := by
  sorry

end third_year_cost_l3453_345375


namespace grade10_sample_size_l3453_345362

/-- Calculates the number of students to be selected from a specific grade in a stratified random sample. -/
def stratifiedSampleSize (gradeSize : ℕ) (totalSize : ℕ) (sampleSize : ℕ) : ℕ :=
  (gradeSize * sampleSize) / totalSize

/-- The number of students to be selected from grade 10 in a stratified random sample is 40. -/
theorem grade10_sample_size :
  stratifiedSampleSize 1200 3000 100 = 40 := by
  sorry

end grade10_sample_size_l3453_345362


namespace mixture_ratio_correct_l3453_345378

def initial_alcohol : ℚ := 4
def initial_water : ℚ := 4
def added_water : ℚ := 8/3

def final_alcohol : ℚ := initial_alcohol
def final_water : ℚ := initial_water + added_water
def final_total : ℚ := final_alcohol + final_water

def desired_alcohol_ratio : ℚ := 3/8
def desired_water_ratio : ℚ := 5/8

theorem mixture_ratio_correct :
  (final_alcohol / final_total = desired_alcohol_ratio) ∧
  (final_water / final_total = desired_water_ratio) :=
sorry

end mixture_ratio_correct_l3453_345378


namespace gcd_problem_l3453_345300

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 570 * k) :
  Int.gcd (5 * b^3 + 2 * b^2 + 6 * b + 95) b = 95 := by
  sorry

end gcd_problem_l3453_345300


namespace abc_sign_sum_l3453_345316

theorem abc_sign_sum (a b c : ℚ) (h : |a*b*c| / (a*b*c) = 1) :
  |a| / a + |b| / b + |c| / c = -1 ∨ |a| / a + |b| / b + |c| / c = 3 := by
  sorry

end abc_sign_sum_l3453_345316


namespace monotonically_decreasing_interval_of_f_shifted_l3453_345317

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the derivative of f
axiom f_derivative (x : ℝ) : deriv f x = 2 * x - 4

-- Theorem statement
theorem monotonically_decreasing_interval_of_f_shifted :
  ∀ x : ℝ, (∀ y : ℝ, y < 3 → deriv (fun z ↦ f (z - 1)) y < 0) ∧
           (∀ y : ℝ, y ≥ 3 → deriv (fun z ↦ f (z - 1)) y ≥ 0) :=
by sorry

end monotonically_decreasing_interval_of_f_shifted_l3453_345317


namespace routes_count_l3453_345334

/-- The number of routes from A to B with 6 horizontal and 6 vertical moves -/
def num_routes : ℕ := 924

/-- The total number of moves -/
def total_moves : ℕ := 12

/-- The number of horizontal moves -/
def horizontal_moves : ℕ := 6

/-- The number of vertical moves -/
def vertical_moves : ℕ := 6

theorem routes_count : 
  num_routes = Nat.choose total_moves horizontal_moves :=
by sorry

end routes_count_l3453_345334


namespace wire_connections_count_l3453_345322

/-- The number of wire segments --/
def n : ℕ := 5

/-- The number of possible orientations for each segment --/
def orientations : ℕ := 2

/-- The total number of ways to connect the wire segments --/
def total_connections : ℕ := n.factorial * orientations ^ n

theorem wire_connections_count : total_connections = 3840 := by
  sorry

end wire_connections_count_l3453_345322


namespace geometric_sequence_sum_l3453_345308

theorem geometric_sequence_sum (a r : ℝ) : 
  (a + a * r = 15) →
  (a * (1 - r^6) / (1 - r) = 195) →
  (a * (1 - r^4) / (1 - r) = 82) :=
by
  sorry

end geometric_sequence_sum_l3453_345308


namespace total_cost_beef_vegetables_l3453_345359

/-- The total cost of beef and vegetables given their weights and prices -/
theorem total_cost_beef_vegetables 
  (beef_weight : ℝ) 
  (vegetable_weight : ℝ) 
  (vegetable_price : ℝ) 
  (beef_price_multiplier : ℝ) : 
  beef_weight = 4 →
  vegetable_weight = 6 →
  vegetable_price = 2 →
  beef_price_multiplier = 3 →
  beef_weight * (vegetable_price * beef_price_multiplier) + vegetable_weight * vegetable_price = 36 := by
  sorry

end total_cost_beef_vegetables_l3453_345359


namespace function_characterization_l3453_345330

/-- A function from positive integers to non-negative integers -/
def PositiveToNonNegative := ℕ+ → ℕ

/-- The p-adic valuation of a positive integer -/
noncomputable def vp (p : ℕ+) (n : ℕ+) : ℕ := sorry

theorem function_characterization 
  (f : PositiveToNonNegative) 
  (h1 : ∃ n, f n ≠ 0)
  (h2 : ∀ x y, f (x * y) = f x + f y)
  (h3 : ∃ S : Set ℕ+, Set.Infinite S ∧ ∀ n ∈ S, ∀ k < n, f k = f (n - k)) :
  ∃ (N : ℕ+) (p : ℕ+), Nat.Prime p ∧ ∀ n, f n = N * vp p n :=
sorry

end function_characterization_l3453_345330


namespace fred_has_ten_balloons_l3453_345384

/-- The number of red balloons Fred has -/
def fred_balloons (total sam dan : ℕ) : ℕ := total - (sam + dan)

/-- Theorem stating that Fred has 10 red balloons -/
theorem fred_has_ten_balloons (total sam dan : ℕ) 
  (h_total : total = 72)
  (h_sam : sam = 46)
  (h_dan : dan = 16) :
  fred_balloons total sam dan = 10 := by
  sorry

end fred_has_ten_balloons_l3453_345384


namespace binomial_expansion_coefficient_l3453_345329

theorem binomial_expansion_coefficient (x : ℝ) (x_ne_zero : x ≠ 0) :
  let expansion := (x^2 - 1/x)^5
  let second_term_coefficient := Finset.sum (Finset.range 6) (fun k => 
    if k = 1 then (-1)^k * (Nat.choose 5 k) * x^(10 - 3*k)
    else 0)
  second_term_coefficient = -5 := by
  sorry

end binomial_expansion_coefficient_l3453_345329


namespace expression_proof_l3453_345326

/-- An expression that, when divided by (3x + 29), equals 2 -/
def E (x : ℝ) : ℝ := 6 * x + 58

theorem expression_proof (x : ℝ) : E x / (3 * x + 29) = 2 := by
  sorry

end expression_proof_l3453_345326


namespace beautiful_point_coordinates_l3453_345309

/-- A point (x,y) is called a "beautiful point" if x + y = x * y -/
def is_beautiful_point (x y : ℝ) : Prop := x + y = x * y

/-- The distance of a point (x,y) from the y-axis is the absolute value of x -/
def distance_from_y_axis (x : ℝ) : ℝ := |x|

theorem beautiful_point_coordinates :
  ∀ x y : ℝ, is_beautiful_point x y → distance_from_y_axis x = 2 →
  ((x = 2 ∧ y = 2) ∨ (x = -2 ∧ y = 2/3)) :=
sorry

end beautiful_point_coordinates_l3453_345309


namespace third_grade_girls_l3453_345393

theorem third_grade_girls (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 123 → boys = 66 → total = boys + girls → girls = 57 := by
  sorry

end third_grade_girls_l3453_345393


namespace two_intersection_points_l3453_345331

-- Define the lines
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0
def line2 (x y : ℝ) : Prop := 5 * x - 2 * y - 10 = 0
def line3 (x : ℝ) : Prop := x = 3
def line4 (y : ℝ) : Prop := y = 1

-- Define an intersection point
def is_intersection_point (x y : ℝ) : Prop :=
  (line1 x y ∧ line2 x y) ∨
  (line1 x y ∧ line3 x) ∨
  (line1 x y ∧ line4 y) ∨
  (line2 x y ∧ line3 x) ∨
  (line2 x y ∧ line4 y) ∨
  (line3 x ∧ line4 y)

-- Theorem: There are exactly two distinct intersection points
theorem two_intersection_points :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    is_intersection_point x₁ y₁ ∧
    is_intersection_point x₂ y₂ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    (∀ (x y : ℝ), is_intersection_point x y → (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
sorry

end two_intersection_points_l3453_345331


namespace min_value_polynomial_min_value_achieved_l3453_345394

theorem min_value_polynomial (x : ℝ) : (x + 1) * (x + 3) * (x + 5) * (x + 7) + 2050 ≥ 2034 := by
  sorry

theorem min_value_achieved : ∃ x : ℝ, (x + 1) * (x + 3) * (x + 5) * (x + 7) + 2050 = 2034 := by
  sorry

end min_value_polynomial_min_value_achieved_l3453_345394


namespace hyperbola_eccentricity_l3453_345327

/-- The eccentricity of a hyperbola with equation x^2 - y^2/m = 1 is 2 if and only if m = 3 -/
theorem hyperbola_eccentricity (m : ℝ) :
  (∀ x y : ℝ, x^2 - y^2/m = 1) →
  (∃ e : ℝ, e = 2 ∧ e^2 = 1 + 1/m) →
  m = 3 :=
by sorry

end hyperbola_eccentricity_l3453_345327


namespace problem_solution_l3453_345324

theorem problem_solution (x y z a b : ℝ) 
  (h1 : (x + y) / 2 = (z + x) / 3)
  (h2 : (x + y) / 2 = (y + z) / 4)
  (h3 : x + y + z = 36 * a)
  (h4 : b = x + y) :
  b = 16 := by
  sorry

end problem_solution_l3453_345324


namespace ship_meetings_count_l3453_345397

/-- The number of ships sailing in each direction -/
def num_ships : ℕ := 5

/-- The total number of meetings between two groups of ships -/
def total_meetings (n : ℕ) : ℕ := n * n

/-- Theorem stating that the total number of meetings is 25 -/
theorem ship_meetings_count : total_meetings num_ships = 25 := by
  sorry

end ship_meetings_count_l3453_345397


namespace stock_market_investment_l3453_345323

theorem stock_market_investment (x : ℝ) (h : x > 0) : 
  x * (1 + 0.8) * (1 - 0.3) > x := by
  sorry

end stock_market_investment_l3453_345323


namespace min_value_theorem_l3453_345349

theorem min_value_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^2 + 1 / (b * (a - b)) ≥ 4 := by
  sorry

end min_value_theorem_l3453_345349


namespace sqrt_calculation_l3453_345374

theorem sqrt_calculation : 
  Real.sqrt 27 / (Real.sqrt 3 / 2) * (2 * Real.sqrt 2) - 6 * Real.sqrt 2 = 6 * Real.sqrt 2 := by
  sorry

end sqrt_calculation_l3453_345374


namespace equivalent_representations_l3453_345335

theorem equivalent_representations : 
  ∀ (a b c d e f : ℚ),
  (a = 9/18) → 
  (b = 1/2) → 
  (c = 27/54) → 
  (d = 1/2) → 
  (e = 1/2) → 
  (f = 1/2) → 
  (a = b ∧ b = c ∧ c = d ∧ d = e ∧ e = f) := by
  sorry

#check equivalent_representations

end equivalent_representations_l3453_345335


namespace square_park_fencing_cost_l3453_345388

/-- Given a square park with a total fencing cost, calculate the cost per side -/
theorem square_park_fencing_cost (total_cost : ℝ) (h_total_cost : total_cost = 172) :
  total_cost / 4 = 43 := by
  sorry

end square_park_fencing_cost_l3453_345388


namespace cos_five_pi_sixth_plus_alpha_l3453_345395

theorem cos_five_pi_sixth_plus_alpha (α : ℝ) (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) :
  Real.cos (5 * π / 6 + α) = -(Real.sqrt 3 / 3) := by
sorry

end cos_five_pi_sixth_plus_alpha_l3453_345395


namespace abs_sum_lt_sum_abs_when_product_negative_l3453_345372

theorem abs_sum_lt_sum_abs_when_product_negative (a b : ℝ) :
  a * b < 0 → |a + b| < |a| + |b| := by sorry

end abs_sum_lt_sum_abs_when_product_negative_l3453_345372


namespace prob_two_cards_sum_17_l3453_345396

-- Define the total number of cards in a standard deck
def total_cards : ℕ := 52

-- Define the number of each specific card (8 or 9) in the deck
def cards_of_each_value : ℕ := 4

-- Define the probability of choosing two specific cards
def prob_two_specific_cards : ℚ := (cards_of_each_value : ℚ) / total_cards * (cards_of_each_value : ℚ) / (total_cards - 1)

-- Define the number of ways to choose two cards that sum to 17 (8+9 or 9+8)
def ways_to_sum_17 : ℕ := 2

theorem prob_two_cards_sum_17 : 
  prob_two_specific_cards * ways_to_sum_17 = 8 / 663 := by sorry

end prob_two_cards_sum_17_l3453_345396


namespace odd_positive_poly_one_real_zero_l3453_345336

/-- A polynomial with positive real coefficients of odd degree -/
structure OddPositivePoly where
  degree : Nat
  coeffs : Fin degree → ℝ
  odd_degree : Odd degree
  positive_coeffs : ∀ i, coeffs i > 0

/-- A permutation of the coefficients of a polynomial -/
def PermutedPoly (p : OddPositivePoly) :=
  { σ : Equiv (Fin p.degree) (Fin p.degree) // True }

/-- The number of real zeros of a polynomial -/
noncomputable def num_real_zeros (p : OddPositivePoly) (perm : PermutedPoly p) : ℕ :=
  sorry

/-- Theorem: For any odd degree polynomial with positive coefficients,
    there exists a permutation of its coefficients such that
    the resulting polynomial has exactly one real zero -/
theorem odd_positive_poly_one_real_zero (p : OddPositivePoly) :
  ∃ perm : PermutedPoly p, num_real_zeros p perm = 1 :=
sorry

end odd_positive_poly_one_real_zero_l3453_345336


namespace f_properties_l3453_345348

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / (1 + Real.exp x) - 1/2

-- Theorem statement
theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x y, x < y → f x < f y)  -- f is increasing
  := by sorry

end f_properties_l3453_345348


namespace odd_function_theorem_l3453_345319

/-- A function f: ℝ → ℝ is odd if f(x) = -f(-x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

/-- The main theorem: if f is odd and satisfies the given functional equation,
    then f is the zero function -/
theorem odd_function_theorem (f : ℝ → ℝ) 
    (h_odd : IsOdd f) 
    (h_eq : ∀ x y, f (x + y) * f (x - y) = f x ^ 2 * f y ^ 2) : 
    ∀ x, f x = 0 := by
  sorry


end odd_function_theorem_l3453_345319


namespace arrangement_count_l3453_345392

/-- Represents the number of different seed types -/
def num_seed_types : ℕ := 5

/-- Represents the number of experimental fields -/
def num_fields : ℕ := 5

/-- Represents the number of seed types that can be placed at the ends -/
def num_end_seeds : ℕ := 3

/-- Represents the number of positions for the A-B pair -/
def num_ab_positions : ℕ := 3

/-- Calculates the number of ways to arrange seeds under the given conditions -/
def calculate_arrangements : ℕ :=
  (num_end_seeds * (num_end_seeds - 1)) * -- Arrangements for the ends
  (num_ab_positions * 2)                  -- Arrangements for A-B pair and remaining seed

/-- Theorem stating that the number of arrangement methods is 24 -/
theorem arrangement_count : calculate_arrangements = 24 := by
  sorry

end arrangement_count_l3453_345392


namespace f_of_g_10_l3453_345358

/-- The function g(x) = 4x + 10 -/
def g (x : ℝ) : ℝ := 4 * x + 10

/-- The function f(x) = 6x - 12 -/
def f (x : ℝ) : ℝ := 6 * x - 12

/-- Theorem: f(g(10)) = 288 -/
theorem f_of_g_10 : f (g 10) = 288 := by sorry

end f_of_g_10_l3453_345358


namespace banana_problem_l3453_345391

/-- Represents the number of bananas eaten on a given day -/
def bananas_eaten (day : ℕ) (first_day : ℕ) : ℕ :=
  first_day + 6 * (day - 1)

/-- The total number of bananas eaten over 5 days -/
def total_bananas (first_day : ℕ) : ℕ :=
  (bananas_eaten 1 first_day) + (bananas_eaten 2 first_day) + 
  (bananas_eaten 3 first_day) + (bananas_eaten 4 first_day) + 
  (bananas_eaten 5 first_day)

theorem banana_problem : 
  ∃ (first_day : ℕ), total_bananas first_day = 100 ∧ first_day = 8 :=
by
  sorry

end banana_problem_l3453_345391


namespace derivative_of_f_l3453_345364

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 / (x + 3)

theorem derivative_of_f (x : ℝ) (h : x ≠ -3) :
  deriv f x = (x^2 + 6*x) / (x + 3)^2 := by sorry

end derivative_of_f_l3453_345364


namespace gcd_2146_1813_l3453_345363

theorem gcd_2146_1813 : Nat.gcd 2146 1813 = 37 := by sorry

end gcd_2146_1813_l3453_345363


namespace f_sum_derivative_equals_two_l3453_345386

noncomputable def f (x : ℝ) : ℝ := ((x + 1)^2 + Real.sin x) / (x^2 + 1)

theorem f_sum_derivative_equals_two :
  let f' := deriv f
  f 2017 + f' 2017 + f (-2017) - f' (-2017) = 2 := by sorry

end f_sum_derivative_equals_two_l3453_345386


namespace custom_op_theorem_l3453_345337

-- Define the custom operation x
def customOp (M N : Set ℕ) : Set ℕ := {x | x ∈ M ∨ x ∈ N ∧ x ∉ M ∩ N}

-- Define sets M and N
def M : Set ℕ := {0, 2, 4, 6, 8, 10}
def N : Set ℕ := {0, 3, 6, 9, 12, 15}

-- Theorem statement
theorem custom_op_theorem : customOp (customOp M N) M = N := by
  sorry

end custom_op_theorem_l3453_345337


namespace money_distribution_l3453_345342

-- Define the variables
variable (A B C : ℕ)

-- State the theorem
theorem money_distribution (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 320) : C = 20 := by
  sorry

end money_distribution_l3453_345342


namespace regular_polygon_properties_l3453_345341

/-- A regular polygon with an exterior angle of 18 degrees has 20 sides and interior angles of 162 degrees. -/
theorem regular_polygon_properties :
  ∀ (n : ℕ) (exterior_angle interior_angle : ℝ),
  exterior_angle = 18 →
  n * exterior_angle = 360 →
  interior_angle = (n - 2 : ℝ) * 180 / n →
  n = 20 ∧ interior_angle = 162 := by
  sorry

end regular_polygon_properties_l3453_345341
