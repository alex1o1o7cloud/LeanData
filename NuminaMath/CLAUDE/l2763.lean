import Mathlib

namespace NUMINAMATH_CALUDE_large_posters_count_l2763_276394

theorem large_posters_count (total : ℕ) (small_fraction : ℚ) (medium_fraction : ℚ) : 
  total = 50 →
  small_fraction = 2 / 5 →
  medium_fraction = 1 / 2 →
  (total : ℚ) * small_fraction + (total : ℚ) * medium_fraction + 5 = total :=
by
  sorry

end NUMINAMATH_CALUDE_large_posters_count_l2763_276394


namespace NUMINAMATH_CALUDE_place_one_after_two_digit_number_l2763_276351

/-- Given a two-digit number with tens digit t and units digit u,
    prove that placing the digit 1 after this number results in 100t + 10u + 1 -/
theorem place_one_after_two_digit_number (t u : ℕ) :
  let original := 10 * t + u
  let new_number := original * 10 + 1
  new_number = 100 * t + 10 * u + 1 := by
sorry

end NUMINAMATH_CALUDE_place_one_after_two_digit_number_l2763_276351


namespace NUMINAMATH_CALUDE_golden_ratio_percentage_l2763_276361

theorem golden_ratio_percentage (a b : ℝ) (h : a > 0) (h' : b > 0) :
  b / a = a / (a + b) → b / a = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_percentage_l2763_276361


namespace NUMINAMATH_CALUDE_fraction_product_is_three_fifths_l2763_276333

theorem fraction_product_is_three_fifths :
  (7 / 4 : ℚ) * (8 / 14 : ℚ) * (20 / 12 : ℚ) * (15 / 25 : ℚ) *
  (21 / 14 : ℚ) * (12 / 18 : ℚ) * (28 / 14 : ℚ) * (30 / 50 : ℚ) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_is_three_fifths_l2763_276333


namespace NUMINAMATH_CALUDE_distance_to_x_axis_l2763_276306

theorem distance_to_x_axis (P : ℝ × ℝ) : P = (3, -2) → |P.2| = 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_x_axis_l2763_276306


namespace NUMINAMATH_CALUDE_max_value_of_fraction_l2763_276317

theorem max_value_of_fraction (a b : ℝ) 
  (h1 : a + b - 2 ≥ 0)
  (h2 : b - a - 1 ≤ 0)
  (h3 : a ≤ 1) :
  ∃ (max : ℝ), max = 7/5 ∧ ∀ x y, 
    x + y - 2 ≥ 0 → y - x - 1 ≤ 0 → x ≤ 1 → 
    (x + 2*y) / (2*x + y) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_fraction_l2763_276317


namespace NUMINAMATH_CALUDE_max_value_of_f_min_value_of_f_in_interval_range_of_a_l2763_276331

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 5) / Real.exp x

theorem max_value_of_f :
  ∃ (x : ℝ), f x = 5 ∧ ∀ (y : ℝ), f y ≤ 5 :=
sorry

theorem min_value_of_f_in_interval :
  ∃ (x : ℝ), x ≤ 0 ∧ f x = -Real.exp 3 ∧ ∀ (y : ℝ), y ≤ 0 → f y ≥ -Real.exp 3 :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ), x^2 + 5*x + 5 - a * Real.exp x ≥ 0) ↔ a ≤ -Real.exp 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_min_value_of_f_in_interval_range_of_a_l2763_276331


namespace NUMINAMATH_CALUDE_complex_sum_equality_l2763_276346

theorem complex_sum_equality : 
  let B : ℂ := 3 + 2*I
  let Q : ℂ := -5
  let R : ℂ := 1 - I
  let T : ℂ := 3 + 5*I
  B - Q + R + T = 2 + 6*I :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_equality_l2763_276346


namespace NUMINAMATH_CALUDE_mangoes_kelly_can_buy_l2763_276387

def mangoes_cost_per_half_pound : ℝ := 0.60
def kelly_budget : ℝ := 12

theorem mangoes_kelly_can_buy :
  let cost_per_pound : ℝ := 2 * mangoes_cost_per_half_pound
  let pounds_kelly_can_buy : ℝ := kelly_budget / cost_per_pound
  pounds_kelly_can_buy = 10 := by sorry

end NUMINAMATH_CALUDE_mangoes_kelly_can_buy_l2763_276387


namespace NUMINAMATH_CALUDE_circle_C_properties_l2763_276363

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 4*y + 4 = 0

-- Define the line that contains the center of circle C
def center_line (x y : ℝ) : Prop :=
  x + 2*y + 1 = 0

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop :=
  8*x - 15*y - 3 = 0 ∨ x = 6

-- Define the line l
def line_l (x y m : ℝ) : Prop :=
  y = x + m

theorem circle_C_properties :
  -- Circle C passes through M(0, -2) and N(3, 1)
  circle_C 0 (-2) ∧ circle_C 3 1 ∧
  -- The center of circle C lies on the line x + 2y + 1 = 0
  ∃ (cx cy : ℝ), center_line cx cy ∧
    ∀ (x y : ℝ), circle_C x y ↔ (x - cx)^2 + (y - cy)^2 = (cx^2 + cy^2 - 4) →
  -- The tangent line to circle C passing through (6, 3) is correct
  tangent_line 6 3 ∧
  -- The line l has the correct equations
  (line_l x y (-1) ∨ line_l x y (-4)) ∧
  -- Circle C₁ with diameter AB (intersection of l and C) passes through the origin
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    ((line_l x₁ y₁ (-1) ∧ line_l x₂ y₂ (-1)) ∨ (line_l x₁ y₁ (-4) ∧ line_l x₂ y₂ (-4))) ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 2 * ((x₁^2 + y₁^2) + (x₂^2 + y₂^2)) :=
sorry

end NUMINAMATH_CALUDE_circle_C_properties_l2763_276363


namespace NUMINAMATH_CALUDE_ticket_popcorn_difference_l2763_276380

/-- Represents the cost of items and the deal in a movie theater. -/
structure MovieTheaterCosts where
  deal : ℝ
  ticket : ℝ
  popcorn : ℝ
  drink : ℝ
  candy : ℝ

/-- The conditions of the movie theater deal problem. -/
def dealConditions (c : MovieTheaterCosts) : Prop :=
  c.deal = 20 ∧
  c.ticket = 8 ∧
  c.drink = c.popcorn + 1 ∧
  c.candy = c.drink / 2 ∧
  c.deal = c.ticket + c.popcorn + c.drink + c.candy - 2

/-- The theorem stating the difference between ticket and popcorn costs. -/
theorem ticket_popcorn_difference (c : MovieTheaterCosts) 
  (h : dealConditions c) : c.ticket - c.popcorn = 3 := by
  sorry


end NUMINAMATH_CALUDE_ticket_popcorn_difference_l2763_276380


namespace NUMINAMATH_CALUDE_tinas_fourth_hour_coins_verify_final_coins_l2763_276320

/-- Represents the number of coins in Tina's jar at different stages -/
structure CoinJar where
  initial : ℕ := 0
  first_hour : ℕ
  second_third_hours : ℕ
  fourth_hour : ℕ
  fifth_hour : ℕ

/-- The coin jar problem setup -/
def tinas_jar : CoinJar :=
  { first_hour := 20
  , second_third_hours := 60
  , fourth_hour := 40  -- This is what we want to prove
  , fifth_hour := 100 }

/-- Theorem stating that the number of coins Tina put in during the fourth hour is 40 -/
theorem tinas_fourth_hour_coins :
  tinas_jar.fourth_hour = 40 :=
by
  -- The actual proof would go here
  sorry

/-- Verify that the final number of coins matches the problem statement -/
theorem verify_final_coins :
  tinas_jar.first_hour + tinas_jar.second_third_hours + tinas_jar.fourth_hour - 20 = tinas_jar.fifth_hour :=
by
  -- The actual proof would go here
  sorry

end NUMINAMATH_CALUDE_tinas_fourth_hour_coins_verify_final_coins_l2763_276320


namespace NUMINAMATH_CALUDE_complex_fraction_difference_l2763_276381

theorem complex_fraction_difference (i : ℂ) (h : i * i = -1) :
  (3 + 2*i) / (2 - 3*i) - (3 - 2*i) / (2 + 3*i) = 2*i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_difference_l2763_276381


namespace NUMINAMATH_CALUDE_wyatt_envelopes_l2763_276359

theorem wyatt_envelopes (blue : ℕ) (yellow : ℕ) : 
  yellow = blue - 4 →
  blue + yellow = 16 →
  blue = 10 := by
sorry

end NUMINAMATH_CALUDE_wyatt_envelopes_l2763_276359


namespace NUMINAMATH_CALUDE_factorization_of_quadratic_l2763_276321

theorem factorization_of_quadratic (a : ℝ) : a^2 - 2*a = a*(a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_quadratic_l2763_276321


namespace NUMINAMATH_CALUDE_diplomats_speaking_french_l2763_276393

theorem diplomats_speaking_french (total : ℕ) (not_russian : ℕ) (neither : ℕ) (both : ℕ) :
  total = 100 →
  not_russian = 32 →
  neither = 20 →
  both = 10 →
  ∃ french : ℕ, french = 22 ∧ french = total - not_russian + both :=
by sorry

end NUMINAMATH_CALUDE_diplomats_speaking_french_l2763_276393


namespace NUMINAMATH_CALUDE_product_greater_than_sum_l2763_276334

theorem product_greater_than_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a - b = a / b) : a * b > a + b := by
  sorry

end NUMINAMATH_CALUDE_product_greater_than_sum_l2763_276334


namespace NUMINAMATH_CALUDE_lanas_initial_pages_l2763_276302

theorem lanas_initial_pages (x : ℕ) : 
  x + (42 / 2) = 29 → x = 8 := by sorry

end NUMINAMATH_CALUDE_lanas_initial_pages_l2763_276302


namespace NUMINAMATH_CALUDE_angle_FDB_is_40_l2763_276325

-- Define the points
variable (A B C D E F : Point)

-- Define the angles
def angle (P Q R : Point) : ℝ := sorry

-- Define isosceles triangle
def isosceles (P Q R : Point) : Prop :=
  angle P Q R = angle P R Q

-- State the theorem
theorem angle_FDB_is_40 :
  isosceles A D E →
  isosceles A B C →
  angle D F C = 150 →
  angle F D B = 40 := by sorry

end NUMINAMATH_CALUDE_angle_FDB_is_40_l2763_276325


namespace NUMINAMATH_CALUDE_trajectory_of_point_P_l2763_276349

/-- The trajectory of a point P on the curve ρcos θ + 2ρsin θ = 3, where 0 ≤ θ ≤ π/4 and ρ > 0,
    is a line segment with endpoints (1,1) and (3,0). -/
theorem trajectory_of_point_P (θ : ℝ) (ρ : ℝ) (h1 : 0 ≤ θ) (h2 : θ ≤ π/4) (h3 : ρ > 0)
  (h4 : ρ * Real.cos θ + 2 * ρ * Real.sin θ = 3) :
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
  ρ * Real.cos θ = 3 - 2 * t ∧
  ρ * Real.sin θ = t :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_point_P_l2763_276349


namespace NUMINAMATH_CALUDE_john_spending_l2763_276384

def supermarket_spending (x : ℝ) (total : ℝ) : Prop :=
  let fruits_veg := x / 100 * total
  let meat := (1 / 3) * total
  let bakery := (1 / 6) * total
  let candy := 6
  fruits_veg + meat + bakery + candy = total ∧
  candy = 0.1 * total ∧
  x = 40 ∧
  fruits_veg = 24 ∧
  total = 60

theorem john_spending :
  ∃ (x : ℝ) (total : ℝ), supermarket_spending x total :=
sorry

end NUMINAMATH_CALUDE_john_spending_l2763_276384


namespace NUMINAMATH_CALUDE_crocodile_earnings_exceed_peter_l2763_276386

theorem crocodile_earnings_exceed_peter (n : ℕ) : (∀ k < n, 2^k ≤ 64*k + 1) ∧ 2^n > 64*n + 1 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_crocodile_earnings_exceed_peter_l2763_276386


namespace NUMINAMATH_CALUDE_range_of_a_l2763_276379

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x - a > 0
def q (x : ℝ) : Prop := x > 1

-- Define what it means for p to be a sufficient condition for q
def sufficient (a : ℝ) : Prop := ∀ x, p x a → q x

-- Define what it means for p to be not a necessary condition for q
def not_necessary (a : ℝ) : Prop := ∃ x, q x ∧ ¬(p x a)

-- Theorem statement
theorem range_of_a (a : ℝ) : 
  (sufficient a ∧ not_necessary a) → a > 1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2763_276379


namespace NUMINAMATH_CALUDE_clothing_business_optimization_l2763_276338

/-- Represents the monthly sales volume as a function of selling price -/
def sales_volume (x : ℝ) : ℝ := -3 * x + 900

/-- Represents the total monthly revenue as a function of selling price -/
def total_revenue (x : ℝ) : ℝ := (x - 80) * (sales_volume x)

/-- The cost price of clothing in yuan -/
def cost_price : ℝ := 100

/-- The government subsidy per piece in yuan -/
def subsidy_per_piece : ℝ := 20

theorem clothing_business_optimization :
  /- Part 1: Government subsidy when selling price is 160 yuan -/
  (sales_volume 160 * subsidy_per_piece = 8400) ∧
  /- Part 2: Optimal selling price and maximum revenue -/
  (∃ (x_max : ℝ), x_max = 190 ∧
    (∀ x, total_revenue x ≤ total_revenue x_max) ∧
    total_revenue x_max = 36300) :=
by sorry

end NUMINAMATH_CALUDE_clothing_business_optimization_l2763_276338


namespace NUMINAMATH_CALUDE_arithmetic_mean_proof_l2763_276309

theorem arithmetic_mean_proof (x a b : ℝ) (hx : x ≠ b ∧ x ≠ -b) :
  (1/2) * ((x + a + b)/(x + b) + (x - a - b)/(x - b)) = 1 - a*b/(x^2 - b^2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_proof_l2763_276309


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2763_276388

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  X^5 + 3 = (X + 1)^2 * q + 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2763_276388


namespace NUMINAMATH_CALUDE_mary_height_to_grow_l2763_276313

/-- The problem of calculating how much Mary needs to grow to ride Kingda Ka -/
theorem mary_height_to_grow (min_height brother_height : ℝ) (h1 : min_height = 140) 
  (h2 : brother_height = 180) : 
  min_height - (2/3 * brother_height) = 20 := by
  sorry

end NUMINAMATH_CALUDE_mary_height_to_grow_l2763_276313


namespace NUMINAMATH_CALUDE_tom_spending_l2763_276374

def apple_count : ℕ := 4
def egg_count : ℕ := 6
def bread_count : ℕ := 3
def cheese_count : ℕ := 2
def chicken_count : ℕ := 1

def apple_price : ℚ := 1
def egg_price : ℚ := 0.5
def bread_price : ℚ := 3
def cheese_price : ℚ := 6
def chicken_price : ℚ := 8

def coupon_threshold : ℚ := 40
def coupon_value : ℚ := 10

def total_cost : ℚ :=
  apple_count * apple_price +
  egg_count * egg_price +
  bread_count * bread_price +
  cheese_count * cheese_price +
  chicken_count * chicken_price

theorem tom_spending :
  (if total_cost ≥ coupon_threshold then total_cost - coupon_value else total_cost) = 36 := by
  sorry

end NUMINAMATH_CALUDE_tom_spending_l2763_276374


namespace NUMINAMATH_CALUDE_equation_solution_l2763_276373

theorem equation_solution :
  ∃ y : ℚ, (2 * y + 3 * y = 600 - (4 * y + 5 * y + 100)) ∧ y = 250 / 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2763_276373


namespace NUMINAMATH_CALUDE_ice_cream_flavors_l2763_276397

/-- The number of ways to distribute indistinguishable objects into distinguishable boxes -/
def distribute (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of new ice cream flavors -/
def new_flavors : ℕ := distribute 6 5

theorem ice_cream_flavors : new_flavors = 210 := by sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_l2763_276397


namespace NUMINAMATH_CALUDE_f_3_eq_9_l2763_276377

/-- A function f that is monotonic on R and satisfies f(f(x) - 2^x) = 3 for all x ∈ R -/
def f : ℝ → ℝ :=
  sorry

/-- f is monotonic on R -/
axiom f_monotonic : Monotone f

/-- f satisfies f(f(x) - 2^x) = 3 for all x ∈ R -/
axiom f_property (x : ℝ) : f (f x - 2^x) = 3

/-- The main theorem: f(3) = 9 -/
theorem f_3_eq_9 : f 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_f_3_eq_9_l2763_276377


namespace NUMINAMATH_CALUDE_expression_evaluation_l2763_276337

theorem expression_evaluation : 
  let x : ℚ := 1/2
  (x^2 * (x - 1) - x * (x^2 + x - 1)) = 0 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2763_276337


namespace NUMINAMATH_CALUDE_max_elements_sum_l2763_276353

/-- A shape formed by adding a pyramid to a rectangular prism -/
structure PrismPyramid where
  prism_faces : Nat
  prism_edges : Nat
  prism_vertices : Nat
  pyramid_new_faces : Nat
  pyramid_new_edges : Nat
  pyramid_new_vertex : Nat

/-- The total number of exterior elements in the combined shape -/
def total_elements (shape : PrismPyramid) : Nat :=
  (shape.prism_faces - 1 + shape.pyramid_new_faces) +
  (shape.prism_edges + shape.pyramid_new_edges) +
  (shape.prism_vertices + shape.pyramid_new_vertex)

/-- Theorem stating the maximum sum of exterior elements -/
theorem max_elements_sum :
  ∀ shape : PrismPyramid,
  shape.prism_faces = 6 →
  shape.prism_edges = 12 →
  shape.prism_vertices = 8 →
  shape.pyramid_new_faces ≤ 4 →
  shape.pyramid_new_edges ≤ 4 →
  shape.pyramid_new_vertex ≤ 1 →
  total_elements shape ≤ 34 :=
sorry

end NUMINAMATH_CALUDE_max_elements_sum_l2763_276353


namespace NUMINAMATH_CALUDE_truck_sand_problem_l2763_276399

/-- The amount of sand remaining on a truck after making several stops --/
def sandRemaining (initialSand : ℕ) (sandLostAtStops : List ℕ) : ℕ :=
  initialSand - sandLostAtStops.sum

/-- Theorem: A truck with 1050 pounds of sand that loses 32, 67, 45, and 54 pounds at four stops will have 852 pounds remaining --/
theorem truck_sand_problem :
  let initialSand : ℕ := 1050
  let sandLostAtStops : List ℕ := [32, 67, 45, 54]
  sandRemaining initialSand sandLostAtStops = 852 := by
  sorry

#eval sandRemaining 1050 [32, 67, 45, 54]

end NUMINAMATH_CALUDE_truck_sand_problem_l2763_276399


namespace NUMINAMATH_CALUDE_line_through_circle_center_l2763_276332

/-- Given a line and a circle, if the line passes through the center of the circle,
    then the value of m in the line equation is 0. -/
theorem line_through_circle_center (m : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*x + 4*y = 0 → 
    ∃ h k : ℝ, (h - 1)^2 + (k + 2)^2 = 0 ∧ 2*h + k + m = 0) → 
  m = 0 := by
sorry

end NUMINAMATH_CALUDE_line_through_circle_center_l2763_276332


namespace NUMINAMATH_CALUDE_activity_participation_l2763_276305

theorem activity_participation (total : ℕ) (books songs movies : ℕ) 
  (books_songs books_movies songs_movies : ℕ) (all_three : ℕ) : 
  total = 200 → 
  books = 80 → 
  songs = 60 → 
  movies = 30 → 
  books_songs = 25 → 
  books_movies = 15 → 
  songs_movies = 20 → 
  all_three = 10 → 
  books + songs + movies - books_songs - books_movies - songs_movies + all_three = 120 :=
by sorry

end NUMINAMATH_CALUDE_activity_participation_l2763_276305


namespace NUMINAMATH_CALUDE_decomposition_675_l2763_276362

theorem decomposition_675 (n : Nat) (h : n = 675) :
  ∃ (num_stacks height : Nat),
    num_stacks > 1 ∧
    height > 1 ∧
    n = 3^3 * 5^2 ∧
    num_stacks = 3 ∧
    height = 3^2 * 5^2 ∧
    height^num_stacks = n := by
  sorry

end NUMINAMATH_CALUDE_decomposition_675_l2763_276362


namespace NUMINAMATH_CALUDE_cylinder_volume_from_rectangle_l2763_276398

/-- The volume of a cylinder formed by rotating a rectangle about its shorter side. -/
theorem cylinder_volume_from_rectangle (h w : ℝ) (h_pos : h > 0) (w_pos : w > 0) :
  let r := w / (2 * Real.pi)
  (Real.pi * r^2 * h) = 1000 / Real.pi → h = 10 ∧ w = 20 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_from_rectangle_l2763_276398


namespace NUMINAMATH_CALUDE_intersecting_lines_sum_of_intercepts_l2763_276310

/-- Two lines intersecting at (3, 3) have the sum of their y-intercepts equal to 4 -/
theorem intersecting_lines_sum_of_intercepts (c d : ℝ) : 
  (3 = (1/3) * 3 + c) ∧ (3 = (1/3) * 3 + d) → c + d = 4 := by
  sorry

#check intersecting_lines_sum_of_intercepts

end NUMINAMATH_CALUDE_intersecting_lines_sum_of_intercepts_l2763_276310


namespace NUMINAMATH_CALUDE_profit_growth_equation_l2763_276342

/-- 
Given an initial profit of 250,000 yuan in May and an expected profit of 360,000 yuan in July,
with an average monthly growth rate of x over 2 months, prove that the equation 25(1+x)^2 = 36 holds true.
-/
theorem profit_growth_equation (x : ℝ) : 
  (250000 : ℝ) * (1 + x)^2 = 360000 → 25 * (1 + x)^2 = 36 := by
  sorry


end NUMINAMATH_CALUDE_profit_growth_equation_l2763_276342


namespace NUMINAMATH_CALUDE_sin_double_angle_plus_pi_sixth_l2763_276390

theorem sin_double_angle_plus_pi_sixth (α : Real) 
  (h : Real.sin (α - π/6) = 1/3) : 
  Real.sin (2*α + π/6) = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_plus_pi_sixth_l2763_276390


namespace NUMINAMATH_CALUDE_brown_mms_second_bag_l2763_276371

theorem brown_mms_second_bag (bags : Nat) (first_bag third_bag fourth_bag fifth_bag average : Nat) : 
  bags = 5 → 
  first_bag = 9 → 
  third_bag = 8 → 
  fourth_bag = 8 → 
  fifth_bag = 3 → 
  average = 8 → 
  ∃ second_bag : Nat, 
    second_bag = 12 ∧ 
    (first_bag + second_bag + third_bag + fourth_bag + fifth_bag) / bags = average := by
  sorry


end NUMINAMATH_CALUDE_brown_mms_second_bag_l2763_276371


namespace NUMINAMATH_CALUDE_division_22_by_8_l2763_276326

theorem division_22_by_8 : (22 : ℚ) / 8 = 2.75 := by sorry

end NUMINAMATH_CALUDE_division_22_by_8_l2763_276326


namespace NUMINAMATH_CALUDE_retail_price_approx_163_59_l2763_276341

/-- Calculates the retail price of a machine before discount -/
def retail_price_before_discount (
  num_machines : ℕ) 
  (wholesale_price : ℚ) 
  (bulk_discount_rate : ℚ) 
  (sales_tax_rate : ℚ) 
  (profit_rate : ℚ) 
  (customer_discount_rate : ℚ) : ℚ :=
  let total_wholesale := num_machines * wholesale_price
  let bulk_discount := bulk_discount_rate * total_wholesale
  let total_cost_after_discount := total_wholesale - bulk_discount
  let profit_per_machine := profit_rate * wholesale_price
  let total_profit := num_machines * profit_per_machine
  let sales_tax := sales_tax_rate * total_profit
  let total_amount_after_tax := total_cost_after_discount + total_profit - sales_tax
  let price_before_discount := total_amount_after_tax / (num_machines * (1 - customer_discount_rate))
  price_before_discount

/-- Theorem stating that the retail price before discount is approximately $163.59 -/
theorem retail_price_approx_163_59 :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ 
  |retail_price_before_discount 15 126 0.06 0.08 0.22 0.12 - 163.59| < ε :=
sorry

end NUMINAMATH_CALUDE_retail_price_approx_163_59_l2763_276341


namespace NUMINAMATH_CALUDE_sum_lower_bound_l2763_276308

theorem sum_lower_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  a + b ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_lower_bound_l2763_276308


namespace NUMINAMATH_CALUDE_power_tower_mod_500_l2763_276354

theorem power_tower_mod_500 : 2^(2^(2^2)) % 500 = 36 := by
  sorry

end NUMINAMATH_CALUDE_power_tower_mod_500_l2763_276354


namespace NUMINAMATH_CALUDE_complex_absolute_value_l2763_276328

theorem complex_absolute_value (z : ℂ) (h : (z + 1) * Complex.I = 3 + 2 * Complex.I) :
  Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l2763_276328


namespace NUMINAMATH_CALUDE_profit_achievement_l2763_276335

/-- The number of pens in a pack -/
def pens_per_pack : ℕ := 4

/-- The cost of a pack of pens in dollars -/
def pack_cost : ℚ := 7

/-- The number of pens sold at the given rate -/
def pens_sold_rate : ℕ := 5

/-- The price for the number of pens sold at the given rate in dollars -/
def price_sold_rate : ℚ := 12

/-- The target profit in dollars -/
def target_profit : ℚ := 50

/-- The minimum number of pens needed to be sold to achieve the target profit -/
def min_pens_to_sell : ℕ := 77

theorem profit_achievement :
  ∃ (n : ℕ), n ≥ min_pens_to_sell ∧
  (n : ℚ) * (price_sold_rate / pens_sold_rate) - 
  (n : ℚ) * (pack_cost / pens_per_pack) ≥ target_profit ∧
  ∀ (m : ℕ), m < min_pens_to_sell →
  (m : ℚ) * (price_sold_rate / pens_sold_rate) - 
  (m : ℚ) * (pack_cost / pens_per_pack) < target_profit :=
by sorry

end NUMINAMATH_CALUDE_profit_achievement_l2763_276335


namespace NUMINAMATH_CALUDE_arc_length_calculation_l2763_276375

theorem arc_length_calculation (r α : Real) (h1 : r = π) (h2 : α = 2 * π / 3) :
  r * α = (2 / 3) * π^2 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_calculation_l2763_276375


namespace NUMINAMATH_CALUDE_car_trip_distance_l2763_276369

theorem car_trip_distance (D : ℝ) : 
  (D / 2 : ℝ) + (D / 2 / 4 : ℝ) + 105 = D → D = 280 :=
by sorry

end NUMINAMATH_CALUDE_car_trip_distance_l2763_276369


namespace NUMINAMATH_CALUDE_car_pedestrian_speed_ratio_l2763_276322

/-- The ratio of a car's speed to a pedestrian's speed on a bridge -/
theorem car_pedestrian_speed_ratio :
  ∀ (L : ℝ) (v_p v_c : ℝ),
  L > 0 → v_p > 0 → v_c > 0 →
  (4/9 * L) / v_p = (5/9 * L) / v_c →
  v_c / v_p = 9 := by
  sorry

end NUMINAMATH_CALUDE_car_pedestrian_speed_ratio_l2763_276322


namespace NUMINAMATH_CALUDE_probability_sum_10_l2763_276304

-- Define a die roll as a natural number between 1 and 6
def DieRoll : Type := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define a function to check if the sum of three die rolls is 10
def sumIs10 (roll1 roll2 roll3 : DieRoll) : Prop :=
  roll1.val + roll2.val + roll3.val = 10

-- Define the total number of possible outcomes
def totalOutcomes : ℕ := 216

-- Define the number of favorable outcomes (sum is 10)
def favorableOutcomes : ℕ := 27

-- Theorem statement
theorem probability_sum_10 :
  (favorableOutcomes : ℚ) / totalOutcomes = 27 / 216 :=
sorry

end NUMINAMATH_CALUDE_probability_sum_10_l2763_276304


namespace NUMINAMATH_CALUDE_b_investment_is_200_l2763_276307

/-- Represents the investment scenario with two investors A and B --/
structure Investment where
  a_amount : ℝ  -- A's investment amount
  b_amount : ℝ  -- B's investment amount
  a_months : ℝ  -- Months A's money was invested
  b_months : ℝ  -- Months B's money was invested
  total_profit : ℝ  -- Total profit at the end of the year
  a_profit : ℝ  -- A's share of the profit

/-- The theorem stating that B's investment is $200 given the conditions --/
theorem b_investment_is_200 (inv : Investment) 
  (h1 : inv.a_amount = 150)
  (h2 : inv.a_months = 12)
  (h3 : inv.b_months = 6)
  (h4 : inv.total_profit = 100)
  (h5 : inv.a_profit = 60)
  (h6 : inv.a_profit / inv.total_profit = 
        (inv.a_amount * inv.a_months) / 
        (inv.a_amount * inv.a_months + inv.b_amount * inv.b_months)) :
  inv.b_amount = 200 := by
  sorry


end NUMINAMATH_CALUDE_b_investment_is_200_l2763_276307


namespace NUMINAMATH_CALUDE_maple_trees_after_planting_l2763_276303

/-- The number of maple trees in the park after planting is equal to the sum of 
    the initial number of trees and the number of newly planted trees. -/
theorem maple_trees_after_planting 
  (initial_trees : ℕ) 
  (planted_trees : ℕ) 
  (h1 : initial_trees = 53) 
  (h2 : planted_trees = 11) : 
  initial_trees + planted_trees = 64 := by
  sorry

#check maple_trees_after_planting

end NUMINAMATH_CALUDE_maple_trees_after_planting_l2763_276303


namespace NUMINAMATH_CALUDE_weight_of_A_l2763_276343

theorem weight_of_A (a b c d : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  (b + c + d + (d + 6)) / 4 = 79 →
  a = 174 :=
by sorry

end NUMINAMATH_CALUDE_weight_of_A_l2763_276343


namespace NUMINAMATH_CALUDE_basketball_team_probabilities_l2763_276368

/-- Represents a series of independent events -/
structure EventSeries where
  n : ℕ  -- number of events
  p : ℝ  -- probability of success for each event
  h1 : 0 ≤ p ∧ p ≤ 1  -- probability is between 0 and 1

/-- The probability of k failures before the first success -/
def prob_k_failures_before_success (es : EventSeries) (k : ℕ) : ℝ :=
  (1 - es.p)^k * es.p

/-- The probability of exactly k successes in n events -/
def prob_exactly_k_successes (es : EventSeries) (k : ℕ) : ℝ :=
  (Nat.choose es.n k : ℝ) * es.p^k * (1 - es.p)^(es.n - k)

/-- The expected number of successes in n events -/
def expected_successes (es : EventSeries) : ℝ :=
  es.n * es.p

theorem basketball_team_probabilities :
  ∀ es : EventSeries,
    es.n = 6 ∧ es.p = 1/3 →
    (prob_k_failures_before_success es 2 = 4/27) ∧
    (prob_exactly_k_successes es 3 = 160/729) ∧
    (expected_successes es = 2) :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_probabilities_l2763_276368


namespace NUMINAMATH_CALUDE_sports_activity_division_l2763_276316

theorem sports_activity_division :
  ∀ (a b c : ℕ),
    a + b + c = 48 →
    a ≠ b ∧ b ≠ c ∧ a ≠ c →
    (∃ (x : ℕ), a = 10 * x + 6) →
    (∃ (y : ℕ), b = 10 * y + 6) →
    (∃ (z : ℕ), c = 10 * z + 6) →
    (a = 6 ∧ b = 16 ∧ c = 26) ∨ (a = 6 ∧ b = 26 ∧ c = 16) ∨
    (a = 16 ∧ b = 6 ∧ c = 26) ∨ (a = 16 ∧ b = 26 ∧ c = 6) ∨
    (a = 26 ∧ b = 6 ∧ c = 16) ∨ (a = 26 ∧ b = 16 ∧ c = 6) :=
by sorry


end NUMINAMATH_CALUDE_sports_activity_division_l2763_276316


namespace NUMINAMATH_CALUDE_no_rearranged_power_of_two_l2763_276385

/-- Checks if all digits of a natural number are non-zero -/
def allDigitsNonZero (n : ℕ) : Prop := sorry

/-- Checks if two natural numbers have the same digits (possibly in different order) -/
def sameDigits (m n : ℕ) : Prop := sorry

/-- There do not exist two distinct powers of 2 with all non-zero digits that are rearrangements of each other -/
theorem no_rearranged_power_of_two : ¬∃ (a b : ℕ), a ≠ b ∧ 
  allDigitsNonZero (2^a) ∧ 
  allDigitsNonZero (2^b) ∧ 
  sameDigits (2^a) (2^b) := by
  sorry

end NUMINAMATH_CALUDE_no_rearranged_power_of_two_l2763_276385


namespace NUMINAMATH_CALUDE_radius_of_circle_M_l2763_276340

/-- Definition of Circle M -/
def CircleM (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x + 6*y = 0

/-- Theorem: The radius of Circle M is 5 -/
theorem radius_of_circle_M : ∃ (h k r : ℝ), r = 5 ∧ 
  ∀ (x y : ℝ), CircleM x y ↔ (x - h)^2 + (y - k)^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_radius_of_circle_M_l2763_276340


namespace NUMINAMATH_CALUDE_characterization_of_matrices_with_power_in_S_l2763_276301

-- Define the set S
def S : Set (Matrix (Fin 2) (Fin 2) ℝ) :=
  {M | ∃ (a r : ℝ), M = !![a, a+r; a+2*r, a+3*r]}

-- Define the property of M^k being in S for some k > 1
def has_power_in_S (M : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  ∃ (k : ℕ), k > 1 ∧ (M ^ k) ∈ S

-- Main theorem
theorem characterization_of_matrices_with_power_in_S :
  ∀ (M : Matrix (Fin 2) (Fin 2) ℝ),
  M ∈ S → (has_power_in_S M ↔ 
    (∃ (c : ℝ), M = c • !![1, 1; 1, 1]) ∨
    (∃ (c : ℝ), M = c • !![-3, -1; 1, 3])) :=
by sorry

end NUMINAMATH_CALUDE_characterization_of_matrices_with_power_in_S_l2763_276301


namespace NUMINAMATH_CALUDE_original_number_before_increase_l2763_276360

theorem original_number_before_increase (x : ℝ) : x * 1.5 = 165 → x = 110 := by
  sorry

end NUMINAMATH_CALUDE_original_number_before_increase_l2763_276360


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2763_276350

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x < 0 → x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x < 0 ∧ x^3 - x^2 + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2763_276350


namespace NUMINAMATH_CALUDE_solution_set_implies_m_value_l2763_276327

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 3|

-- State the theorem
theorem solution_set_implies_m_value (m : ℝ) :
  (∀ x : ℝ, f m x > 2 ↔ 2 < x ∧ x < 4) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_m_value_l2763_276327


namespace NUMINAMATH_CALUDE_train_ride_nap_time_l2763_276356

theorem train_ride_nap_time (total_time reading_time eating_time movie_time : ℕ) 
  (h1 : total_time = 9)
  (h2 : reading_time = 2)
  (h3 : eating_time = 1)
  (h4 : movie_time = 3) :
  total_time - (reading_time + eating_time + movie_time) = 3 := by
  sorry

end NUMINAMATH_CALUDE_train_ride_nap_time_l2763_276356


namespace NUMINAMATH_CALUDE_divisibility_by_3804_l2763_276395

theorem divisibility_by_3804 (n : ℕ+) :
  ∃ k : ℤ, (n.val ^ 3 - n.val : ℤ) * (5 ^ (8 * n.val + 4) + 3 ^ (4 * n.val + 2)) = 3804 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_3804_l2763_276395


namespace NUMINAMATH_CALUDE_calculation_proof_l2763_276339

theorem calculation_proof : 
  (168 / 100 * ((1265^2) / 21)) / (6 - (3^2)) = -42646.26666666667 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2763_276339


namespace NUMINAMATH_CALUDE_sector_central_angle_l2763_276311

/-- Given a sector with radius 1 and perimeter 4, its central angle in radians has an absolute value of 2. -/
theorem sector_central_angle (r : ℝ) (L : ℝ) (α : ℝ) : 
  r = 1 → L = 4 → L = r * α + 2 * r → |α| = 2 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2763_276311


namespace NUMINAMATH_CALUDE_pizza_fraction_l2763_276382

theorem pizza_fraction (initial_parts : ℕ) (cuts_per_part : ℕ) (pieces_eaten : ℕ) : 
  initial_parts = 12 →
  cuts_per_part = 2 →
  pieces_eaten = 3 →
  (pieces_eaten : ℚ) / (initial_parts * cuts_per_part : ℚ) = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_pizza_fraction_l2763_276382


namespace NUMINAMATH_CALUDE_sales_and_profit_theorem_l2763_276396

/-- Represents the monthly sales quantity as a function of selling price -/
def monthly_sales (x : ℝ) : ℝ := -30 * x + 960

/-- Represents the monthly profit as a function of selling price -/
def monthly_profit (x : ℝ) : ℝ := (x - 10) * (monthly_sales x)

theorem sales_and_profit_theorem :
  let cost_price : ℝ := 10
  let price1 : ℝ := 20
  let price2 : ℝ := 30
  let sales1 : ℝ := 360
  let sales2 : ℝ := 60
  let target_profit : ℝ := 3600
  (∀ x, monthly_sales x = -30 * x + 960) ∧
  (monthly_sales price1 = sales1) ∧
  (monthly_sales price2 = sales2) ∧
  (∃ x, monthly_profit x = target_profit) ∧
  (monthly_profit 22 = target_profit) ∧
  (monthly_profit 20 = target_profit) := by
  sorry

#check sales_and_profit_theorem

end NUMINAMATH_CALUDE_sales_and_profit_theorem_l2763_276396


namespace NUMINAMATH_CALUDE_negation_of_inequality_statement_l2763_276392

theorem negation_of_inequality_statement :
  (¬ ∀ x : ℝ, x > 0 → x - 1 ≥ Real.log x) ↔
  (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ - 1 < Real.log x₀) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_inequality_statement_l2763_276392


namespace NUMINAMATH_CALUDE_parabola_hyperbola_equations_l2763_276367

/-- Given a parabola and a hyperbola satisfying certain conditions, 
    prove their equations. -/
theorem parabola_hyperbola_equations :
  ∀ (parabola : ℝ → ℝ → Prop) (hyperbola : ℝ → ℝ → Prop),
  (∀ x y, parabola x y → (x = 0 ∧ y = 0)) →  -- vertex at origin
  (∃ x₀, ∀ y, hyperbola x₀ y → parabola x₀ y) →  -- axis of symmetry passes through focus
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ x y, hyperbola x y ↔ x^2/a^2 - y^2/b^2 = 1) →  -- general form of hyperbola
  hyperbola (3/2) (Real.sqrt 6) →  -- intersection point
  (∀ x y, parabola x y ↔ y^2 = 4*x) ∧  -- equation of parabola
  (∀ x y, hyperbola x y ↔ 4*x^2 - 4*y^2/3 = 1) :=  -- equation of hyperbola
by sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_equations_l2763_276367


namespace NUMINAMATH_CALUDE_square_of_modified_41_l2763_276319

theorem square_of_modified_41 (n : ℕ) :
  let modified_num := (5 * 10^n - 1) * 10^(n+1) + 1
  modified_num^2 = (10^(n+1) - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_modified_41_l2763_276319


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2763_276323

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 5) → x ≥ 5 := by
sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2763_276323


namespace NUMINAMATH_CALUDE_product_of_fractions_l2763_276347

theorem product_of_fractions :
  (3 : ℚ) / 7 * (5 : ℚ) / 13 * (11 : ℚ) / 17 * (19 : ℚ) / 23 = 3135 / 35581 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l2763_276347


namespace NUMINAMATH_CALUDE_triangle_cosA_value_l2763_276378

theorem triangle_cosA_value (A B C : ℝ) (a b c : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  b = Real.sqrt 2 * c →  -- Given condition
  Real.sin A + Real.sqrt 2 * Real.sin C = 2 * Real.sin B →  -- Given condition
  -- Triangle inequality (to ensure it's a valid triangle)
  a + b > c ∧ b + c > a ∧ c + a > b →
  -- Law of sines (to connect side lengths and angles)
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  Real.cos A = Real.sqrt 2 / 4 :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_cosA_value_l2763_276378


namespace NUMINAMATH_CALUDE_smaller_number_proof_l2763_276357

theorem smaller_number_proof (x y m : ℝ) 
  (h1 : x - y = 9)
  (h2 : x + y = 46)
  (h3 : x = m * y) : 
  min x y = 18.5 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l2763_276357


namespace NUMINAMATH_CALUDE_spheres_intersection_similar_triangles_l2763_276345

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a sphere -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Checks if three points are collinear -/
def collinear (p q r : Point3D) : Prop := sorry

/-- Checks if a point lies on an edge of a tetrahedron -/
def on_edge (p : Point3D) (t : Tetrahedron) : Prop := sorry

/-- Checks if a sphere passes through a point -/
def sphere_passes_through (s : Sphere) (p : Point3D) : Prop := sorry

/-- Checks if two triangles are similar -/
def triangles_similar (p1 p2 p3 q1 q2 q3 : Point3D) : Prop := sorry

/-- Main theorem -/
theorem spheres_intersection_similar_triangles 
  (ABCD : Tetrahedron) (G₁ G₂ : Sphere) 
  (K L M P Q R : Point3D) : 
  sphere_passes_through G₁ ABCD.A ∧ 
  sphere_passes_through G₁ ABCD.B ∧ 
  sphere_passes_through G₁ ABCD.C ∧
  sphere_passes_through G₂ ABCD.A ∧ 
  sphere_passes_through G₂ ABCD.B ∧ 
  sphere_passes_through G₂ ABCD.D ∧
  on_edge K ABCD ∧ collinear K ABCD.D ABCD.A ∧
  on_edge L ABCD ∧ collinear L ABCD.D ABCD.B ∧
  on_edge M ABCD ∧ collinear M ABCD.D ABCD.C ∧
  on_edge P ABCD ∧ collinear P ABCD.C ABCD.A ∧
  on_edge Q ABCD ∧ collinear Q ABCD.C ABCD.B ∧
  on_edge R ABCD ∧ collinear R ABCD.C ABCD.D
  →
  triangles_similar K L M P Q R := by
  sorry

end NUMINAMATH_CALUDE_spheres_intersection_similar_triangles_l2763_276345


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2763_276348

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (h1 : a 3 = 4) (h2 : a 6 = 1/2) :
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = q * a n) ∧ q = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2763_276348


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l2763_276315

/-- Two vectors in R² are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- Given vectors a = (-2, 3) and b = (3, m) are perpendicular, prove that m = 2 -/
theorem perpendicular_vectors_m_value :
  let a : ℝ × ℝ := (-2, 3)
  let b : ℝ × ℝ := (3, m)
  perpendicular a b → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l2763_276315


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l2763_276344

/-- The number of ways to place n distinct objects into k distinct containers -/
def placement_count (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 1024 ways to place 5 distinct balls into 4 distinct boxes -/
theorem five_balls_four_boxes : placement_count 5 4 = 1024 := by sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l2763_276344


namespace NUMINAMATH_CALUDE_stratified_sampling_distribution_l2763_276329

theorem stratified_sampling_distribution 
  (total : ℕ) (senior : ℕ) (intermediate : ℕ) (junior : ℕ) (sample_size : ℕ)
  (h_total : total = 150)
  (h_senior : senior = 45)
  (h_intermediate : intermediate = 90)
  (h_junior : junior = 15)
  (h_sum : senior + intermediate + junior = total)
  (h_sample : sample_size = 30) :
  ∃ (sample_senior sample_intermediate sample_junior : ℕ),
    sample_senior + sample_intermediate + sample_junior = sample_size ∧
    sample_senior * total = senior * sample_size ∧
    sample_intermediate * total = intermediate * sample_size ∧
    sample_junior * total = junior * sample_size ∧
    sample_senior = 3 ∧
    sample_intermediate = 18 ∧
    sample_junior = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_distribution_l2763_276329


namespace NUMINAMATH_CALUDE_division_multiplication_equality_l2763_276364

theorem division_multiplication_equality : (144 / 6) * 3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_equality_l2763_276364


namespace NUMINAMATH_CALUDE_complex_number_modulus_l2763_276330

theorem complex_number_modulus (a : ℝ) : a < 0 → Complex.abs (3 + a * Complex.I) = 5 → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l2763_276330


namespace NUMINAMATH_CALUDE_one_in_range_of_f_l2763_276366

/-- The function f(x) = x^2 + bx - 1 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x - 1

/-- Theorem: For all real numbers b, 1 is always in the range of f(x) = x^2 + bx - 1 -/
theorem one_in_range_of_f (b : ℝ) : ∃ x : ℝ, f b x = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_in_range_of_f_l2763_276366


namespace NUMINAMATH_CALUDE_one_seventh_difference_l2763_276389

theorem one_seventh_difference : ∃ (ε : ℚ), 1/7 - 0.14285714285 = ε ∧ ε > 0 ∧ ε < 1/(7*10^10) := by
  sorry

end NUMINAMATH_CALUDE_one_seventh_difference_l2763_276389


namespace NUMINAMATH_CALUDE_train_length_l2763_276358

/-- Calculates the length of a train given its speed, the time it takes to cross a bridge, and the length of the bridge. -/
theorem train_length (train_speed : Real) (bridge_crossing_time : Real) (bridge_length : Real) :
  train_speed = 72 * 1000 / 3600 ∧ 
  bridge_crossing_time = 12.099 ∧ 
  bridge_length = 132 →
  train_speed * bridge_crossing_time - bridge_length = 110 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l2763_276358


namespace NUMINAMATH_CALUDE_purple_balls_count_l2763_276300

theorem purple_balls_count (total : ℕ) (white green yellow red : ℕ) (p : ℚ) :
  total = 100 ∧
  white = 10 ∧
  green = 30 ∧
  yellow = 10 ∧
  red = 47 ∧
  p = 1/2 ∧
  p = (white + green + yellow : ℚ) / total →
  ∃ purple : ℕ, purple = 3 ∧ total = white + green + yellow + red + purple :=
by sorry

end NUMINAMATH_CALUDE_purple_balls_count_l2763_276300


namespace NUMINAMATH_CALUDE_equally_spaced_posts_l2763_276324

/-- Given a sequence of 8 equally spaced posts, if the distance between the first and fifth post
    is 100 meters, then the distance between the first and last post is 175 meters. -/
theorem equally_spaced_posts (posts : Fin 8 → ℝ) 
  (equally_spaced : ∀ i j k : Fin 8, i.val + 1 = j.val → j.val + 1 = k.val → 
    posts k - posts j = posts j - posts i)
  (first_to_fifth : posts 4 - posts 0 = 100) :
  posts 7 - posts 0 = 175 :=
sorry

end NUMINAMATH_CALUDE_equally_spaced_posts_l2763_276324


namespace NUMINAMATH_CALUDE_lines_intersection_l2763_276376

/-- Represents a line in the form ax + by = c -/
structure Line where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Checks if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℚ) : Prop :=
  l.a * x + l.b * y = l.c

/-- The three lines given in the problem -/
def line1 : Line := ⟨-3, 2, 4⟩
def line2 : Line := ⟨1, 3, 3⟩
def line3 : Line := ⟨5, -3, 6⟩

/-- Theorem stating that the given lines intersect at the specified points -/
theorem lines_intersection :
  (line1.contains (10/11) (13/11) ∧
   line2.contains (10/11) (13/11) ∧
   line3.contains 24 38) ∧
  (line1.contains 24 38 ∧
   line2.contains 24 38 ∧
   line3.contains 24 38) := by
  sorry

end NUMINAMATH_CALUDE_lines_intersection_l2763_276376


namespace NUMINAMATH_CALUDE_two_thousand_five_power_l2763_276314

theorem two_thousand_five_power : ∃ a b : ℕ, (2005 : ℕ)^2005 = a^2 + b^2 ∧ ¬∃ c d : ℕ, (2005 : ℕ)^2005 = c^3 + d^3 := by
  sorry

end NUMINAMATH_CALUDE_two_thousand_five_power_l2763_276314


namespace NUMINAMATH_CALUDE_equivalent_discount_l2763_276312

theorem equivalent_discount (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  original_price > 0 →
  0 ≤ discount1 → discount1 < 1 →
  0 ≤ discount2 → discount2 < 1 →
  original_price * (1 - discount1) * (1 - discount2) = original_price * (1 - 0.4) :=
by
  sorry

#check equivalent_discount 50 0.25 0.2

end NUMINAMATH_CALUDE_equivalent_discount_l2763_276312


namespace NUMINAMATH_CALUDE_cube_surface_area_l2763_276352

/-- The surface area of a cube that can be cut into 27 smaller cubes, each with an edge length of 4 cm, is 864 cm². -/
theorem cube_surface_area : 
  ∀ (original_cube_edge : ℝ) (small_cube_edge : ℝ),
  small_cube_edge = 4 →
  (original_cube_edge / small_cube_edge)^3 = 27 →
  6 * original_cube_edge^2 = 864 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_l2763_276352


namespace NUMINAMATH_CALUDE_jake_initial_balloons_count_l2763_276391

/-- The number of balloons Jake brought initially to the park -/
def jake_initial_balloons : ℕ := 3

/-- The number of balloons Allan brought to the park -/
def allan_balloons : ℕ := 6

/-- The number of additional balloons Jake bought at the park -/
def jake_additional_balloons : ℕ := 4

theorem jake_initial_balloons_count :
  jake_initial_balloons = 3 ∧
  allan_balloons = 6 ∧
  jake_additional_balloons = 4 ∧
  jake_initial_balloons + jake_additional_balloons = allan_balloons + 1 :=
by sorry

end NUMINAMATH_CALUDE_jake_initial_balloons_count_l2763_276391


namespace NUMINAMATH_CALUDE_washing_time_proof_l2763_276372

def shirts : ℕ := 18
def pants : ℕ := 12
def sweaters : ℕ := 17
def jeans : ℕ := 13
def max_items_per_cycle : ℕ := 15
def minutes_per_cycle : ℕ := 45

def total_items : ℕ := shirts + pants + sweaters + jeans

def cycles_needed : ℕ := (total_items + max_items_per_cycle - 1) / max_items_per_cycle

def total_minutes : ℕ := cycles_needed * minutes_per_cycle

theorem washing_time_proof : 
  total_minutes / 60 = 3 := by sorry

end NUMINAMATH_CALUDE_washing_time_proof_l2763_276372


namespace NUMINAMATH_CALUDE_baseball_cap_production_l2763_276336

theorem baseball_cap_production (caps_week1 caps_week2 caps_week3 total_4_weeks : ℕ) : 
  caps_week1 = 320 →
  caps_week3 = 300 →
  (caps_week1 + caps_week2 + caps_week3 + (caps_week1 + caps_week2 + caps_week3) / 3) = total_4_weeks →
  total_4_weeks = 1360 →
  caps_week2 = 400 := by
sorry

end NUMINAMATH_CALUDE_baseball_cap_production_l2763_276336


namespace NUMINAMATH_CALUDE_halfway_fraction_l2763_276355

theorem halfway_fraction (a b c d : ℚ) (h1 : a = 3/4) (h2 : b = 5/6) :
  (a + b) / 2 = 19/24 := by sorry

end NUMINAMATH_CALUDE_halfway_fraction_l2763_276355


namespace NUMINAMATH_CALUDE_sum_of_powers_l2763_276370

theorem sum_of_powers : (1 : ℤ)^10 + (-1 : ℤ)^8 + (-1 : ℤ)^7 + (1 : ℤ)^5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l2763_276370


namespace NUMINAMATH_CALUDE_brenda_weighs_220_l2763_276383

def mel_weight : ℕ := 70

def brenda_weight (m : ℕ) : ℕ := 3 * m + 10

theorem brenda_weighs_220 : brenda_weight mel_weight = 220 := by
  sorry

end NUMINAMATH_CALUDE_brenda_weighs_220_l2763_276383


namespace NUMINAMATH_CALUDE_two_color_no_monochromatic_ap_l2763_276365

theorem two_color_no_monochromatic_ap :
  ∃ f : ℕ+ → Bool, ∀ q r : ℕ+, ∃ n1 n2 : ℕ+, f (q * n1 + r) ≠ f (q * n2 + r) :=
by sorry

end NUMINAMATH_CALUDE_two_color_no_monochromatic_ap_l2763_276365


namespace NUMINAMATH_CALUDE_consecutive_integers_product_552_l2763_276318

theorem consecutive_integers_product_552 (x : ℕ) 
  (h1 : x > 0) 
  (h2 : x * (x + 1) = 552) : 
  x + (x + 1) = 47 ∧ (x + 1) - x = 1 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_552_l2763_276318
