import Mathlib

namespace ellipse_eccentricity_l83_8318

/-- Prove that for an ellipse with the given conditions, its eccentricity is √2/2 -/
theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c > 0) :
  let ellipse := fun (x y : ℝ) => x^2 / a^2 + y^2 / b^2 = 1
  let P := (-c, b^2 / a)
  let A := (a, 0)
  let B := (0, b)
  let O := (0, 0)
  ellipse (-c) (b^2 / a) ∧ 
  (B.2 - A.2) / (B.1 - A.1) = (P.2 - O.2) / (P.1 - O.1) →
  c / a = Real.sqrt 2 / 2 :=
by sorry

end ellipse_eccentricity_l83_8318


namespace angle_on_diagonal_line_l83_8358

/-- If the terminal side of angle α lies on the line y = x, then α = kπ + π/4 for some integer k. -/
theorem angle_on_diagonal_line (α : ℝ) :
  (∃ (x y : ℝ), x = y ∧ x = Real.cos α ∧ y = Real.sin α) →
  ∃ (k : ℤ), α = k * π + π / 4 := by sorry

end angle_on_diagonal_line_l83_8358


namespace survey_sampling_suitability_mainland_survey_suitable_l83_8366

/-- Represents a survey with its characteristics -/
structure Survey where
  population_size : ℕ
  requires_comprehensive : Bool
  is_safety_critical : Bool

/-- Defines when a survey is suitable for sampling -/
def suitable_for_sampling (s : Survey) : Prop :=
  s.population_size > 1000 ∧ ¬s.requires_comprehensive ∧ ¬s.is_safety_critical

/-- Theorem stating the condition for a survey to be suitable for sampling -/
theorem survey_sampling_suitability (s : Survey) :
  suitable_for_sampling s ↔
    s.population_size > 1000 ∧ ¬s.requires_comprehensive ∧ ¬s.is_safety_critical := by
  sorry

/-- The mainland population survey (Option C) -/
def mainland_survey : Survey :=
  { population_size := 1000000,  -- Large population
    requires_comprehensive := false,
    is_safety_critical := false }

/-- Theorem stating that the mainland survey is suitable for sampling -/
theorem mainland_survey_suitable :
  suitable_for_sampling mainland_survey := by
  sorry

end survey_sampling_suitability_mainland_survey_suitable_l83_8366


namespace current_speed_current_speed_calculation_l83_8308

/-- The speed of the current in a river given the man's rowing speed in still water,
    the time taken to cover a distance downstream, and the distance covered. -/
theorem current_speed (rowing_speed : ℝ) (time : ℝ) (distance : ℝ) : ℝ :=
  let downstream_speed := distance / (time / 3600)
  downstream_speed - rowing_speed

/-- Proof that the speed of the current is approximately 3.00048 kmph given the specified conditions. -/
theorem current_speed_calculation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.00001 ∧ 
  |current_speed 9 17.998560115190788 0.06 - 3.00048| < ε :=
sorry

end current_speed_current_speed_calculation_l83_8308


namespace log_intersects_x_axis_l83_8380

theorem log_intersects_x_axis : ∃ x : ℝ, x > 0 ∧ Real.log x = 0 := by
  sorry

end log_intersects_x_axis_l83_8380


namespace no_real_sqrt_negative_four_l83_8362

theorem no_real_sqrt_negative_four :
  ¬ ∃ (x : ℝ), x^2 = -4 := by
sorry

end no_real_sqrt_negative_four_l83_8362


namespace complex_equation_solution_l83_8384

theorem complex_equation_solution (z : ℂ) : z * (1 - Complex.I) = 2 → z = 1 + Complex.I := by
  sorry

end complex_equation_solution_l83_8384


namespace min_marbles_for_ten_of_one_color_l83_8398

/-- Represents the number of marbles of each color in the container -/
structure MarbleContainer :=
  (red : ℕ)
  (green : ℕ)
  (yellow : ℕ)
  (blue : ℕ)
  (white : ℕ)
  (black : ℕ)

/-- Defines the specific container from the problem -/
def problemContainer : MarbleContainer :=
  { red := 30
  , green := 25
  , yellow := 23
  , blue := 15
  , white := 10
  , black := 7 }

/-- 
  Theorem: The minimum number of marbles that must be drawn from the container
  without replacement to ensure that at least 10 marbles of a single color are drawn is 53.
-/
theorem min_marbles_for_ten_of_one_color (container : MarbleContainer := problemContainer) :
  (∃ (n : ℕ), n = 53 ∧
    (∀ (m : ℕ), m < n →
      ∃ (r g y b w bl : ℕ),
        r + g + y + b + w + bl = m ∧
        r ≤ container.red ∧
        g ≤ container.green ∧
        y ≤ container.yellow ∧
        b ≤ container.blue ∧
        w ≤ container.white ∧
        bl ≤ container.black ∧
        r < 10 ∧ g < 10 ∧ y < 10 ∧ b < 10 ∧ w < 10 ∧ bl < 10) ∧
    (∀ (r g y b w bl : ℕ),
      r + g + y + b + w + bl = n →
      r ≤ container.red →
      g ≤ container.green →
      y ≤ container.yellow →
      b ≤ container.blue →
      w ≤ container.white →
      bl ≤ container.black →
      (r ≥ 10 ∨ g ≥ 10 ∨ y ≥ 10 ∨ b ≥ 10 ∨ w ≥ 10 ∨ bl ≥ 10))) :=
by sorry


end min_marbles_for_ten_of_one_color_l83_8398


namespace fraction_irreducible_l83_8390

theorem fraction_irreducible (n : ℤ) : Int.gcd (39 * n + 4) (26 * n + 3) = 1 := by
  sorry

end fraction_irreducible_l83_8390


namespace price_per_bracelet_l83_8317

/-- Represents the problem of determining the price per bracelet --/
def bracelet_problem (total_cost : ℕ) (selling_period_weeks : ℕ) (avg_daily_sales : ℕ) : Prop :=
  let total_days : ℕ := selling_period_weeks * 7
  let total_bracelets : ℕ := total_days * avg_daily_sales
  total_bracelets = total_cost ∧ (total_cost : ℚ) / total_bracelets = 1

/-- Proves that the price per bracelet is $1 given the problem conditions --/
theorem price_per_bracelet :
  bracelet_problem 112 2 8 :=
by sorry

end price_per_bracelet_l83_8317


namespace debby_jogging_distance_l83_8375

/-- Represents the number of kilometers Debby jogged on each day -/
structure JoggingDistance where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ

/-- Theorem stating that given the conditions, Debby jogged 5 kilometers on Tuesday -/
theorem debby_jogging_distance (d : JoggingDistance) 
  (h1 : d.monday = 2)
  (h2 : d.wednesday = 9)
  (h3 : d.monday + d.tuesday + d.wednesday = 16) :
  d.tuesday = 5 := by
  sorry

end debby_jogging_distance_l83_8375


namespace shopping_theorem_l83_8348

def shopping_problem (total_amount : ℝ) : Prop :=
  let clothing_percent : ℝ := 0.40
  let food_percent : ℝ := 0.20
  let electronics_percent : ℝ := 0.10
  let cosmetics_percent : ℝ := 0.20
  let household_percent : ℝ := 0.10

  let clothing_discount : ℝ := 0.10
  let food_discount : ℝ := 0.05
  let electronics_discount : ℝ := 0.15
  let cosmetics_discount : ℝ := 0
  let household_discount : ℝ := 0

  let clothing_tax : ℝ := 0.06
  let food_tax : ℝ := 0
  let electronics_tax : ℝ := 0.10
  let cosmetics_tax : ℝ := 0.08
  let household_tax : ℝ := 0.04

  let clothing_amount := total_amount * clothing_percent
  let food_amount := total_amount * food_percent
  let electronics_amount := total_amount * electronics_percent
  let cosmetics_amount := total_amount * cosmetics_percent
  let household_amount := total_amount * household_percent

  let clothing_tax_paid := clothing_amount * (1 - clothing_discount) * clothing_tax
  let food_tax_paid := food_amount * (1 - food_discount) * food_tax
  let electronics_tax_paid := electronics_amount * (1 - electronics_discount) * electronics_tax
  let cosmetics_tax_paid := cosmetics_amount * (1 - cosmetics_discount) * cosmetics_tax
  let household_tax_paid := household_amount * (1 - household_discount) * household_tax

  let total_tax_paid := clothing_tax_paid + food_tax_paid + electronics_tax_paid + cosmetics_tax_paid + household_tax_paid
  let total_tax_percentage := (total_tax_paid / total_amount) * 100

  total_tax_percentage = 5.01

theorem shopping_theorem : ∀ (total_amount : ℝ), total_amount > 0 → shopping_problem total_amount := by
  sorry

end shopping_theorem_l83_8348


namespace bottling_probability_l83_8332

def chocolate_prob : ℚ := 3/4
def vanilla_prob : ℚ := 1/2
def total_days : ℕ := 6
def chocolate_days : ℕ := 4
def vanilla_days : ℕ := 3

theorem bottling_probability : 
  (Nat.choose total_days chocolate_days * chocolate_prob ^ chocolate_days * (1 - chocolate_prob) ^ (total_days - chocolate_days)) *
  (1 - (Nat.choose total_days 0 * vanilla_prob ^ 0 * (1 - vanilla_prob) ^ total_days +
        Nat.choose total_days 1 * vanilla_prob ^ 1 * (1 - vanilla_prob) ^ (total_days - 1) +
        Nat.choose total_days 2 * vanilla_prob ^ 2 * (1 - vanilla_prob) ^ (total_days - 2))) =
  25515/131072 := by
sorry

end bottling_probability_l83_8332


namespace sum_of_four_integers_l83_8349

theorem sum_of_four_integers (k l m n : ℕ+) 
  (h1 : k + l + m + n = k * m) 
  (h2 : k + l + m + n = l * n) : 
  k + l + m + n = 16 ∨ k + l + m + n = 18 ∨ k + l + m + n = 24 ∨ k + l + m + n = 30 :=
sorry

end sum_of_four_integers_l83_8349


namespace circle_radius_order_l83_8387

theorem circle_radius_order :
  let rA : ℝ := Real.sqrt 50
  let aB : ℝ := 16 * Real.pi
  let cC : ℝ := 10 * Real.pi
  let rB : ℝ := Real.sqrt (aB / Real.pi)
  let rC : ℝ := cC / (2 * Real.pi)
  rB < rC ∧ rC < rA := by sorry

end circle_radius_order_l83_8387


namespace fabric_problem_solution_l83_8316

/-- Represents the fabric and flag problem -/
structure FabricProblem where
  initial_fabric : Float
  square_side : Float
  square_count : Nat
  wide_rect_length : Float
  wide_rect_width : Float
  wide_rect_count : Nat
  tall_rect_length : Float
  tall_rect_width : Float
  tall_rect_count : Nat
  triangle_base : Float
  triangle_height : Float
  triangle_count : Nat
  hexagon_side : Float
  hexagon_apothem : Float
  hexagon_count : Nat

/-- Calculates the remaining fabric after making flags -/
def remaining_fabric (p : FabricProblem) : Float :=
  p.initial_fabric -
  (p.square_side * p.square_side * p.square_count.toFloat +
   p.wide_rect_length * p.wide_rect_width * p.wide_rect_count.toFloat +
   p.tall_rect_length * p.tall_rect_width * p.tall_rect_count.toFloat +
   (p.triangle_base * p.triangle_height / 2) * p.triangle_count.toFloat +
   (6 * p.hexagon_side * p.hexagon_apothem / 2) * p.hexagon_count.toFloat)

/-- The theorem stating the remaining fabric for the given problem -/
theorem fabric_problem_solution (p : FabricProblem) :
  p.initial_fabric = 1500 ∧
  p.square_side = 4 ∧
  p.square_count = 22 ∧
  p.wide_rect_length = 5 ∧
  p.wide_rect_width = 3 ∧
  p.wide_rect_count = 28 ∧
  p.tall_rect_length = 3 ∧
  p.tall_rect_width = 5 ∧
  p.tall_rect_count = 14 ∧
  p.triangle_base = 6 ∧
  p.triangle_height = 4 ∧
  p.triangle_count = 18 ∧
  p.hexagon_side = 3 ∧
  p.hexagon_apothem = 2.6 ∧
  p.hexagon_count = 24 →
  remaining_fabric p = -259.6 := by
  sorry

end fabric_problem_solution_l83_8316


namespace henry_actual_earnings_l83_8335

/-- Represents Henry's summer job earnings --/
def HenryEarnings : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ :=
  fun (lawn_rate pile_rate driveway_rate lawns_mowed piles_raked driveways_shoveled : ℕ) =>
    lawn_rate * lawns_mowed + pile_rate * piles_raked + driveway_rate * driveways_shoveled

/-- Theorem stating Henry's actual earnings --/
theorem henry_actual_earnings :
  HenryEarnings 5 10 15 5 3 2 = 85 := by
  sorry

end henry_actual_earnings_l83_8335


namespace colored_paper_problem_l83_8353

/-- The initial number of colored paper pieces Yuna had -/
def initial_yuna : ℕ := 100

/-- The initial number of colored paper pieces Yoojung had -/
def initial_yoojung : ℕ := 210

/-- The number of pieces Yoojung gave to Yuna -/
def transferred : ℕ := 30

/-- The difference in pieces after the transfer -/
def difference : ℕ := 50

theorem colored_paper_problem :
  initial_yuna = 100 ∧
  initial_yoojung = 210 ∧
  transferred = 30 ∧
  difference = 50 ∧
  initial_yoojung - transferred = initial_yuna + transferred + difference :=
by sorry

end colored_paper_problem_l83_8353


namespace equation_one_root_range_l83_8337

-- Define the equation
def equation (k : ℝ) (x : ℝ) : Prop :=
  Real.log (k * x) = 2 * Real.log (x + 1)

-- Define the condition for having only one real root
def has_one_real_root (k : ℝ) : Prop :=
  ∃! x : ℝ, equation k x

-- Define the range of k
def k_range : Set ℝ :=
  Set.Iio 0 ∪ {4}

-- Theorem statement
theorem equation_one_root_range :
  {k : ℝ | has_one_real_root k} = k_range :=
sorry

end equation_one_root_range_l83_8337


namespace zero_in_interval_implies_a_range_l83_8359

/-- The function f(x) = -x^2 + 4x + a has a zero in the interval [-3, 3] -/
def has_zero_in_interval (f : ℝ → ℝ) : Prop :=
  ∃ x ∈ Set.Icc (-3) 3, f x = 0

/-- The main theorem: if f(x) = -x^2 + 4x + a has a zero in [-3, 3], then a ∈ [-3, 21] -/
theorem zero_in_interval_implies_a_range (a : ℝ) :
  has_zero_in_interval (fun x => -x^2 + 4*x + a) → a ∈ Set.Icc (-3) 21 := by
  sorry


end zero_in_interval_implies_a_range_l83_8359


namespace perpendicular_median_triangle_sides_l83_8328

/-- A triangle with sides x, y, and z, where two medians are mutually perpendicular -/
structure PerpendicularMedianTriangle where
  x : ℕ
  y : ℕ
  z : ℕ
  perp_medians : x^2 + y^2 = 5 * z^2

/-- The theorem stating that the triangle with perpendicular medians and integer sides has sides 22, 19, and 13 -/
theorem perpendicular_median_triangle_sides :
  ∀ t : PerpendicularMedianTriangle, t.x = 22 ∧ t.y = 19 ∧ t.z = 13 := by
  sorry

end perpendicular_median_triangle_sides_l83_8328


namespace go_kart_tickets_value_l83_8386

/-- The number of tickets required for a go-kart ride -/
def go_kart_tickets : ℕ := sorry

/-- The number of times Paula rides the go-karts -/
def go_kart_rides : ℕ := 1

/-- The number of times Paula rides the bumper cars -/
def bumper_car_rides : ℕ := 4

/-- The number of tickets required for a bumper car ride -/
def bumper_car_tickets : ℕ := 5

/-- The total number of tickets Paula needs -/
def total_tickets : ℕ := 24

theorem go_kart_tickets_value : 
  go_kart_tickets = 4 :=
by sorry

end go_kart_tickets_value_l83_8386


namespace intersection_implies_a_value_l83_8330

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | 2*x + a ≤ 0}

-- State the theorem
theorem intersection_implies_a_value :
  ∃ a : ℝ, A ∩ B a = {x : ℝ | -2 ≤ x ∧ x ≤ 1} → a = -2 := by
  sorry

end intersection_implies_a_value_l83_8330


namespace largest_constant_inequality_l83_8393

theorem largest_constant_inequality (C : ℝ) : 
  (∀ x y z : ℝ, x^2 + y^2 + z^2 + 1 ≥ C * (x + y + z)) ↔ C ≤ 2 / Real.sqrt 3 :=
by sorry

end largest_constant_inequality_l83_8393


namespace opposite_numbers_l83_8379

theorem opposite_numbers : -5^2 = -((-5)^2) := by sorry

end opposite_numbers_l83_8379


namespace sum_of_cubes_l83_8385

theorem sum_of_cubes (a b c : ℝ) 
  (sum_condition : a + b + c = 3)
  (sum_of_products : a * b + a * c + b * c = 5)
  (product : a * b * c = -6) :
  a^3 + b^3 + c^3 = -36 := by
  sorry

end sum_of_cubes_l83_8385


namespace tangent_line_equation_no_collinear_intersection_l83_8371

-- Define the line l
def line_l (k : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | y = k * x + 2}

-- Define the circle Q
def circle_Q : Set (ℝ × ℝ) :=
  {(x, y) | x^2 + y^2 - 12*x + 32 = 0}

-- Define the point P
def point_P : ℝ × ℝ := (0, 2)

-- Define the center of the circle Q
def center_Q : ℝ × ℝ := (6, 0)

-- Define the tangency condition
def is_tangent (k : ℝ) : Prop :=
  ∃ (x y : ℝ), (x, y) ∈ line_l k ∩ circle_Q ∧
  ∀ (x' y' : ℝ), (x', y') ∈ line_l k ∩ circle_Q → (x', y') = (x, y)

-- Define the intersection condition
def intersects_at_two_points (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ line_l k ∩ circle_Q ∧
  (x₂, y₂) ∈ line_l k ∩ circle_Q ∧ (x₁, y₁) ≠ (x₂, y₂)

-- Define collinearity condition
def are_collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), B - A = t • (C - A)

theorem tangent_line_equation :
  ∀ k : ℝ, is_tangent k →
  (∀ x y : ℝ, (x, y) ∈ line_l k ↔ y = 2 ∨ 3*x + 4*y = 8) :=
sorry

theorem no_collinear_intersection :
  ¬∃ k : ℝ, intersects_at_two_points k ∧
  (∀ A B : ℝ × ℝ, A ∈ circle_Q → B ∈ circle_Q → A ≠ B →
   are_collinear (0, 0) (A.1 + B.1, A.2 + B.2) (6, -2)) :=
sorry

end tangent_line_equation_no_collinear_intersection_l83_8371


namespace stating_clock_confusion_times_l83_8345

/-- Represents the number of degrees the hour hand moves per minute -/
def hourHandSpeed : ℝ := 0.5

/-- Represents the number of degrees the minute hand moves per minute -/
def minuteHandSpeed : ℝ := 6

/-- Represents the total number of degrees in a circle -/
def totalDegrees : ℕ := 360

/-- Represents the number of hours in the given time period -/
def timePeriod : ℕ := 12

/-- Represents the number of times the hands overlap in the given time period -/
def overlapTimes : ℕ := 11

/-- 
  Theorem stating that there are 132 times in a 12-hour period when the clock hands
  can be mistaken for each other, excluding overlaps.
-/
theorem clock_confusion_times : 
  ∃ (confusionTimes : ℕ), 
    confusionTimes = timePeriod * (totalDegrees / (minuteHandSpeed - hourHandSpeed) - 1) - overlapTimes := by
  sorry

end stating_clock_confusion_times_l83_8345


namespace purchasing_power_decrease_l83_8307

def initial_amount : ℝ := 100
def monthly_price_index_increase : ℝ := 0.00465
def months : ℕ := 12

theorem purchasing_power_decrease :
  let final_value := initial_amount * (1 - monthly_price_index_increase) ^ months
  initial_amount - final_value = 4.55 := by
  sorry

end purchasing_power_decrease_l83_8307


namespace basketball_tryouts_l83_8344

/-- The number of boys who tried out for the basketball team -/
def boys_tried_out : ℕ := sorry

/-- The number of girls who tried out for the basketball team -/
def girls_tried_out : ℕ := 9

/-- The number of students who got called back -/
def called_back : ℕ := 2

/-- The number of students who didn't make the cut -/
def didnt_make_cut : ℕ := 21

theorem basketball_tryouts :
  boys_tried_out = 14 ∧
  girls_tried_out + boys_tried_out = called_back + didnt_make_cut :=
by sorry

end basketball_tryouts_l83_8344


namespace games_left_l83_8323

def initial_games : ℕ := 50
def games_given_away : ℕ := 15

theorem games_left : initial_games - games_given_away = 35 := by
  sorry

end games_left_l83_8323


namespace correct_factorization_l83_8382

/-- Given a quadratic expression x^2 - ax + b, if (x+6)(x-1) and (x-2)(x+1) are incorrect
    factorizations due to misreading a and b respectively, then the correct factorization
    is (x+2)(x-3). -/
theorem correct_factorization (a b : ℤ) : 
  (∃ a', (x^2 - a'*x + b = (x+6)*(x-1)) ∧ (a' ≠ a)) →
  (∃ b', (x^2 - a*x + b' = (x-2)*(x+1)) ∧ (b' ≠ b)) →
  (x^2 - a*x + b = (x+2)*(x-3)) :=
sorry

end correct_factorization_l83_8382


namespace speed_in_still_water_l83_8301

theorem speed_in_still_water (upstream_speed downstream_speed : ℝ) :
  upstream_speed = 27 →
  downstream_speed = 35 →
  (upstream_speed + downstream_speed) / 2 = 31 :=
by
  sorry

end speed_in_still_water_l83_8301


namespace ice_cream_bill_calculation_l83_8310

/-- Calculate the final bill amount for an ice cream outing --/
theorem ice_cream_bill_calculation
  (alicia_total brant_total josh_total yvette_total : ℝ)
  (discount_rate tax_rate tip_rate : ℝ)
  (h_alicia : alicia_total = 16.50)
  (h_brant : brant_total = 20.50)
  (h_josh : josh_total = 16.00)
  (h_yvette : yvette_total = 19.50)
  (h_discount : discount_rate = 0.10)
  (h_tax : tax_rate = 0.08)
  (h_tip : tip_rate = 0.20) :
  let subtotal := alicia_total + brant_total + josh_total + yvette_total
  let discounted_subtotal := subtotal * (1 - discount_rate)
  let tax_amount := discounted_subtotal * tax_rate
  let tip_amount := subtotal * tip_rate
  discounted_subtotal + tax_amount + tip_amount = 84.97 := by
  sorry


end ice_cream_bill_calculation_l83_8310


namespace simplify_fraction_l83_8309

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l83_8309


namespace seulgi_kicks_to_win_l83_8319

theorem seulgi_kicks_to_win (hohyeon_first hohyeon_second hyunjeong_first hyunjeong_second seulgi_first : ℕ) 
  (h1 : hohyeon_first = 23)
  (h2 : hohyeon_second = 28)
  (h3 : hyunjeong_first = 32)
  (h4 : hyunjeong_second = 17)
  (h5 : seulgi_first = 27) :
  ∃ seulgi_second : ℕ, 
    seulgi_second ≥ 25 ∧ 
    seulgi_first + seulgi_second > hohyeon_first + hohyeon_second ∧ 
    seulgi_first + seulgi_second > hyunjeong_first + hyunjeong_second :=
by sorry

end seulgi_kicks_to_win_l83_8319


namespace electronic_cat_run_time_l83_8340

/-- Proves that an electronic cat running on a circular track takes 36 seconds to run the last 120 meters -/
theorem electronic_cat_run_time (track_length : ℝ) (speed1 speed2 : ℝ) :
  track_length = 240 →
  speed1 = 5 →
  speed2 = 3 →
  let avg_speed := (speed1 + speed2) / 2
  let total_time := track_length / avg_speed
  let half_time := total_time / 2
  let first_half_distance := speed1 * half_time
  let second_half_distance := track_length - first_half_distance
  let time_at_speed1 := (first_half_distance - track_length / 2) / speed1
  let time_at_speed2 := (track_length / 2 - (first_half_distance - track_length / 2)) / speed2
  (time_at_speed1 + time_at_speed2 : ℝ) = 36 :=
by sorry

end electronic_cat_run_time_l83_8340


namespace min_sum_squares_l83_8312

theorem min_sum_squares (a b c t : ℝ) (h : a + b + c = t) :
  ∃ (min : ℝ), min = t^2 / 3 ∧ 
  (∀ (x y z : ℝ), x + y + z = t → x^2 + y^2 + z^2 ≥ min) ∧
  (∃ (x y z : ℝ), x + y + z = t ∧ x^2 + y^2 + z^2 = min) := by
sorry

end min_sum_squares_l83_8312


namespace juans_number_problem_l83_8351

theorem juans_number_problem (x : ℝ) : 
  (((x + 3) * 3 - 3) / 3 = 10) → x = 8 := by
  sorry

end juans_number_problem_l83_8351


namespace jungkook_apples_l83_8342

theorem jungkook_apples (initial_apples given_apples : ℕ) : 
  initial_apples = 8 → given_apples = 7 → initial_apples + given_apples = 15 := by
  sorry

end jungkook_apples_l83_8342


namespace circle_line_intersection_l83_8314

/-- A circle M with center (a, 2) and radius 2 -/
def circle_M (a x y : ℝ) : Prop := (x - a)^2 + (y - 2)^2 = 4

/-- A line l with equation x - y + 3 = 0 -/
def line_l (x y : ℝ) : Prop := x - y + 3 = 0

/-- The chord intercepted by line l on circle M has a length of 4 -/
def chord_length_4 (a : ℝ) : Prop := ∃ x₁ y₁ x₂ y₂ : ℝ,
  circle_M a x₁ y₁ ∧ circle_M a x₂ y₂ ∧
  line_l x₁ y₁ ∧ line_l x₂ y₂ ∧
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = 16

theorem circle_line_intersection (a : ℝ) :
  circle_M a a 2 ∧ line_l a 2 ∧ chord_length_4 a → a = -1 := by
  sorry

end circle_line_intersection_l83_8314


namespace jennifer_gave_away_six_fruits_l83_8381

/-- Represents the number of fruits Jennifer gave to her sister -/
def fruits_given_away (initial_pears initial_oranges initial_apples fruits_left : ℕ) : ℕ :=
  initial_pears + initial_oranges + initial_apples - fruits_left

/-- Proves that Jennifer gave away 6 fruits -/
theorem jennifer_gave_away_six_fruits :
  ∀ (initial_pears initial_oranges initial_apples fruits_left : ℕ),
    initial_pears = 10 →
    initial_oranges = 20 →
    initial_apples = 2 * initial_pears →
    fruits_left = 44 →
    fruits_given_away initial_pears initial_oranges initial_apples fruits_left = 6 :=
by
  sorry

#check jennifer_gave_away_six_fruits

end jennifer_gave_away_six_fruits_l83_8381


namespace money_sharing_l83_8388

theorem money_sharing (john jose binoy total : ℕ) : 
  john + jose + binoy = total →
  2 * jose = 4 * john →
  3 * binoy = 6 * john →
  john = 1440 →
  total = 8640 := by
sorry

end money_sharing_l83_8388


namespace absolute_value_equation_solution_l83_8365

theorem absolute_value_equation_solution :
  ∃! y : ℝ, abs (y - 4) + 3 * y = 15 :=
by
  -- The unique solution is y = 4.75
  use 4.75
  sorry

end absolute_value_equation_solution_l83_8365


namespace star_property_l83_8391

/-- Operation ⋆ for positive real numbers -/
noncomputable def star (k : ℝ) (x y : ℝ) : ℝ := x^y * k

/-- Theorem stating the properties of the ⋆ operation and the result to be proved -/
theorem star_property (k : ℝ) :
  (k > 0) →
  (∀ x y, x > 0 → y > 0 → (star k (x^y) y = x * star k y y)) →
  (∀ x, x > 0 → star k (star k x 1) x = star k x 1) →
  (star k 1 1 = k) →
  (star k 2 3 = 8 * k) := by
  sorry

end star_property_l83_8391


namespace expression_equals_one_l83_8306

theorem expression_equals_one : 
  (2009 * 2029 + 100) * (1999 * 2039 + 400) / (2019^4 : ℝ) = 1 := by sorry

end expression_equals_one_l83_8306


namespace question_one_l83_8399

theorem question_one (x : ℝ) (h : x^2 - 3*x = 2) :
  1 + 2*x^2 - 6*x = 5 := by
  sorry


end question_one_l83_8399


namespace samuel_fraction_l83_8320

theorem samuel_fraction (total : ℝ) (spent : ℝ) (left : ℝ) :
  total = 240 →
  spent = (1 / 5) * total →
  left = 132 →
  (left + spent) / total = 3 / 4 := by
sorry

end samuel_fraction_l83_8320


namespace highway_time_greater_than_swamp_time_l83_8311

/-- Represents the hunter's journey with different terrains and speeds -/
structure HunterJourney where
  swamp_speed : ℝ
  forest_speed : ℝ
  highway_speed : ℝ
  total_time : ℝ
  total_distance : ℝ

/-- Theorem stating that the time spent on the highway is greater than the time spent in the swamp -/
theorem highway_time_greater_than_swamp_time (journey : HunterJourney) 
  (h1 : journey.swamp_speed = 2)
  (h2 : journey.forest_speed = 4)
  (h3 : journey.highway_speed = 6)
  (h4 : journey.total_time = 4)
  (h5 : journey.total_distance = 17) : 
  ∃ (swamp_time highway_time : ℝ), 
    swamp_time * journey.swamp_speed + 
    (journey.total_time - swamp_time - highway_time) * journey.forest_speed + 
    highway_time * journey.highway_speed = journey.total_distance ∧
    highway_time > swamp_time :=
by sorry

end highway_time_greater_than_swamp_time_l83_8311


namespace algebraic_expression_range_l83_8363

theorem algebraic_expression_range (a : ℝ) : (2 * a - 8) / 3 < 0 → a < 4 := by
  sorry

end algebraic_expression_range_l83_8363


namespace total_cost_is_24_l83_8338

/-- The cost of a burger meal and soda order for two people -/
def total_cost (burger_price : ℚ) : ℚ :=
  let soda_price := burger_price / 3
  let paulo_total := burger_price + soda_price
  let jeremy_total := 2 * paulo_total
  paulo_total + jeremy_total

/-- Theorem: The total cost of Paulo and Jeremy's orders is $24 when a burger meal costs $6 -/
theorem total_cost_is_24 : total_cost 6 = 24 := by
  sorry

end total_cost_is_24_l83_8338


namespace problem_solution_l83_8339

/-- The function f defined on real numbers --/
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := (-2^x + b) / (2^(x+1) + 2)

/-- f is an odd function --/
axiom f_odd (b : ℝ) : ∀ x, f b (-x) = -(f b x)

theorem problem_solution :
  ∃ b : ℝ,
  (∀ x, f b (-x) = -(f b x)) ∧  -- f is odd
  (b = 1) ∧  -- part 1
  (∀ x y, x < y → f b x > f b y) ∧  -- part 2: f is decreasing
  (∀ k, (∀ t, f b (t^2 - 2*t) + f b (2*t^2 - k) < 0) → k < -1/3)  -- part 3
  := by sorry

end problem_solution_l83_8339


namespace function_properties_l83_8341

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- State the theorem
theorem function_properties (a : ℝ) :
  (∀ x : ℝ, f a x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) →
  (a = 2 ∧ 
   ∀ x : ℝ, f 2 (3*x) + f 2 (x+3) ≥ 5/3 ∧
   ∃ x : ℝ, f 2 (3*x) + f 2 (x+3) = 5/3) :=
by sorry

end function_properties_l83_8341


namespace toothpicks_150th_stage_l83_8383

/-- The number of toothpicks in the nth stage of the pattern -/
def toothpicks (n : ℕ) : ℕ := 4 + 4 * (n - 1)

/-- Theorem: The 150th stage of the pattern contains 600 toothpicks -/
theorem toothpicks_150th_stage : toothpicks 150 = 600 := by
  sorry

end toothpicks_150th_stage_l83_8383


namespace janinas_pancakes_l83_8304

/-- Calculates the number of pancakes Janina must sell daily to cover her expenses -/
theorem janinas_pancakes (daily_rent : ℕ) (daily_supplies : ℕ) (price_per_pancake : ℕ) :
  daily_rent = 30 →
  daily_supplies = 12 →
  price_per_pancake = 2 →
  (daily_rent + daily_supplies) / price_per_pancake = 21 := by
  sorry

#check janinas_pancakes

end janinas_pancakes_l83_8304


namespace different_suit_probability_l83_8364

theorem different_suit_probability (total_cards : ℕ) (num_suits : ℕ) 
  (h1 : total_cards = 65) 
  (h2 : num_suits = 5) 
  (h3 : total_cards % num_suits = 0) :
  let cards_per_suit := total_cards / num_suits
  let remaining_cards := total_cards - 1
  let different_suit_cards := total_cards - cards_per_suit
  (different_suit_cards : ℚ) / remaining_cards = 13 / 16 := by
sorry


end different_suit_probability_l83_8364


namespace negation_of_universal_proposition_l83_8333

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 + x - 2 ≥ 0)) ↔ (∃ x : ℝ, x^2 + x - 2 < 0) :=
by sorry

end negation_of_universal_proposition_l83_8333


namespace reciprocal_sum_equality_l83_8396

theorem reciprocal_sum_equality (a b c : ℝ) (n : ℕ) 
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : a + b + c ≠ 0) (h5 : Odd n) 
  (h6 : 1/a + 1/b + 1/c = 1/(a+b+c)) : 
  1/a^n + 1/b^n + 1/c^n = 1/(a^n + b^n + c^n) := by
sorry

end reciprocal_sum_equality_l83_8396


namespace parabola_properties_l83_8367

def parabola (x : ℝ) : ℝ := x^2 - 4*x - 4

theorem parabola_properties :
  (∀ x : ℝ, parabola x ≥ parabola 2) ∧
  (∀ x : ℝ, parabola x = parabola (4 - x)) ∧
  (parabola 2 = -8) :=
sorry

end parabola_properties_l83_8367


namespace four_three_eight_nine_has_two_prime_products_l83_8352

-- Define set C
def C : Set ℕ := {n : ℕ | ∃ k : ℕ, n = 4 * k + 1}

-- Define prime with respect to C
def isPrimeWrtC (k : ℕ) : Prop :=
  k ∈ C ∧ ∀ a b : ℕ, a ∈ C → b ∈ C → k ≠ a * b

-- Define the property of being expressible as product of two primes wrt C in two ways
def hasTwoPrimeProductsInC (n : ℕ) : Prop :=
  ∃ a b c d : ℕ,
    a ≠ c ∧ b ≠ d ∧
    n = a * b ∧ n = c * d ∧
    isPrimeWrtC a ∧ isPrimeWrtC b ∧ isPrimeWrtC c ∧ isPrimeWrtC d

-- Theorem statement
theorem four_three_eight_nine_has_two_prime_products :
  4389 ∈ C ∧ hasTwoPrimeProductsInC 4389 :=
sorry

end four_three_eight_nine_has_two_prime_products_l83_8352


namespace A_symmetric_to_B_about_x_axis_l83_8394

/-- Two points are symmetric about the x-axis if they have the same x-coordinate
    and their y-coordinates are negatives of each other. -/
def symmetric_about_x_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = B.1 ∧ A.2 = -B.2

/-- Point A with coordinates (3, 2) -/
def A : ℝ × ℝ := (3, 2)

/-- Point B with coordinates (3, -2) -/
def B : ℝ × ℝ := (3, -2)

/-- Theorem stating that point A is symmetric to point B about the x-axis -/
theorem A_symmetric_to_B_about_x_axis : symmetric_about_x_axis A B := by
  sorry

end A_symmetric_to_B_about_x_axis_l83_8394


namespace matthew_age_difference_l83_8370

/-- Given three children whose ages sum to 35, with Matthew 2 years older than Rebecca
    and Freddy being 15, prove that Matthew is 4 years younger than Freddy. -/
theorem matthew_age_difference (matthew rebecca freddy : ℕ) : 
  matthew + rebecca + freddy = 35 →
  matthew = rebecca + 2 →
  freddy = 15 →
  freddy - matthew = 4 := by
sorry

end matthew_age_difference_l83_8370


namespace at_least_one_negative_l83_8397

theorem at_least_one_negative (a b : ℝ) (h1 : a ≠ b) (h2 : a ≠ 0) (h3 : b ≠ 0)
  (h4 : a^2 + 1/b = b^2 + 1/a) : a < 0 ∨ b < 0 := by
  sorry

end at_least_one_negative_l83_8397


namespace distance_between_points_l83_8373

theorem distance_between_points (A B C : EuclideanSpace ℝ (Fin 2)) 
  (angle_BAC : Real) (dist_AB : Real) (dist_AC : Real) :
  angle_BAC = 120 * π / 180 →
  dist_AB = 2 →
  dist_AC = 3 →
  ‖B - C‖ = Real.sqrt 19 := by
  sorry

end distance_between_points_l83_8373


namespace roadwork_pitch_barrels_l83_8334

/-- Roadwork problem -/
theorem roadwork_pitch_barrels (total_length day1_paving : ℕ)
  (gravel_per_truckload : ℕ) (gravel_pitch_ratio : ℕ) (truckloads_per_mile : ℕ) :
  total_length = 16 →
  day1_paving = 4 →
  gravel_per_truckload = 2 →
  gravel_pitch_ratio = 5 →
  truckloads_per_mile = 3 →
  (total_length - (day1_paving + (2 * day1_paving - 1))) * truckloads_per_mile * 
    (gravel_per_truckload / gravel_pitch_ratio : ℚ) = 6 :=
by sorry

end roadwork_pitch_barrels_l83_8334


namespace avg_children_with_children_example_l83_8303

/-- The average number of children in families with children, given:
  - total_families: The total number of families
  - avg_children: The average number of children per family (including all families)
  - childless_families: The number of childless families
-/
def avg_children_with_children (total_families : ℕ) (avg_children : ℚ) (childless_families : ℕ) : ℚ :=
  (total_families : ℚ) * avg_children / ((total_families : ℚ) - (childless_families : ℚ))

/-- Theorem stating that given 15 families with an average of 3 children per family,
    and exactly 3 childless families, the average number of children in the families
    with children is 3.75 -/
theorem avg_children_with_children_example :
  avg_children_with_children 15 3 3 = 45 / 12 :=
sorry

end avg_children_with_children_example_l83_8303


namespace scientists_communication_l83_8350

/-- A coloring of edges in a complete graph -/
def Coloring (n : ℕ) := Fin n → Fin n → Fin 3

/-- A triangle in a graph -/
def Triangle (n : ℕ) := { t : Fin n × Fin n × Fin n // t.1 ≠ t.2.1 ∧ t.1 ≠ t.2.2 ∧ t.2.1 ≠ t.2.2 }

/-- A monochromatic triangle under a given coloring -/
def MonochromaticTriangle (n : ℕ) (c : Coloring n) (t : Triangle n) :=
  c t.val.1 t.val.2.1 = c t.val.1 t.val.2.2 ∧
  c t.val.1 t.val.2.1 = c t.val.2.1 t.val.2.2

theorem scientists_communication :
  ∀ c : Coloring 17, ∃ t : Triangle 17, MonochromaticTriangle 17 c t :=
sorry

end scientists_communication_l83_8350


namespace ellipse_focus_y_axis_alpha_range_l83_8395

/-- Represents an ellipse with equation x^2 * sin(α) - y^2 * cos(α) = 1 --/
structure Ellipse (α : Real) where
  equation : ∀ x y, x^2 * Real.sin α - y^2 * Real.cos α = 1

/-- Predicate to check if the focus of an ellipse is on the y-axis --/
def focus_on_y_axis (e : Ellipse α) : Prop :=
  1 / Real.sin α > 0 ∧ 1 / (-Real.cos α) > 0 ∧ 1 / Real.sin α < 1 / (-Real.cos α)

theorem ellipse_focus_y_axis_alpha_range (α : Real) (h1 : 0 ≤ α) (h2 : α < 2 * Real.pi) 
  (e : Ellipse α) (h3 : focus_on_y_axis e) : 
  Real.pi / 2 < α ∧ α < 3 * Real.pi / 4 := by
  sorry

end ellipse_focus_y_axis_alpha_range_l83_8395


namespace negation_of_existential_real_equation_l83_8361

theorem negation_of_existential_real_equation :
  (¬ ∃ x : ℝ, x^3 - 2*x + 1 = 0) ↔ (∀ x : ℝ, x^3 - 2*x + 1 ≠ 0) := by
  sorry

end negation_of_existential_real_equation_l83_8361


namespace hyperbola_eccentricity_ratio_l83_8355

theorem hyperbola_eccentricity_ratio (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (mx^2 - ny^2 = 1 ∧ (m + n) / n = 4) → m / n = 3 := by
  sorry

end hyperbola_eccentricity_ratio_l83_8355


namespace beth_crayon_packs_l83_8369

theorem beth_crayon_packs :
  ∀ (total_crayons : ℕ) (crayons_per_pack : ℕ) (extra_crayons : ℕ),
    total_crayons = 40 →
    crayons_per_pack = 10 →
    extra_crayons = 6 →
    (total_crayons - extra_crayons) / crayons_per_pack = 3 :=
by
  sorry

end beth_crayon_packs_l83_8369


namespace tangent_semiperimeter_median_inequality_l83_8378

variable (a b c : ℝ)
variable (s : ℝ)
variable (ta tb tc : ℝ)
variable (ma mb mc : ℝ)

/-- Triangle inequality --/
axiom triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Semi-perimeter definition --/
axiom semi_perimeter : s = (a + b + c) / 2

/-- Tangent line definitions --/
axiom tangent_a : ta = (2 / (b + c)) * Real.sqrt (b * c * s * (s - a))
axiom tangent_b : tb = (2 / (a + c)) * Real.sqrt (a * c * s * (s - b))
axiom tangent_c : tc = (2 / (a + b)) * Real.sqrt (a * b * s * (s - c))

/-- Median line definitions --/
axiom median_a : ma^2 = (2 * b^2 + 2 * c^2 - a^2) / 4
axiom median_b : mb^2 = (2 * a^2 + 2 * c^2 - b^2) / 4
axiom median_c : mc^2 = (2 * a^2 + 2 * b^2 - c^2) / 4

/-- Theorem: Tangent-Semiperimeter-Median Inequality --/
theorem tangent_semiperimeter_median_inequality :
  ta^2 + tb^2 + tc^2 ≤ s^2 ∧ s^2 ≤ ma^2 + mb^2 + mc^2 :=
sorry

end tangent_semiperimeter_median_inequality_l83_8378


namespace tan_value_from_log_equation_l83_8305

theorem tan_value_from_log_equation (x : Real) 
  (h1 : x ∈ Set.Ioo 0 (π/2))
  (h2 : Real.log (Real.sin (2*x)) - Real.log (Real.sin x) = Real.log (1/2)) :
  Real.tan x = Real.sqrt 15 := by
  sorry

end tan_value_from_log_equation_l83_8305


namespace thirdYearSelected_l83_8368

/-- Represents the number of students in each year -/
structure StudentPopulation where
  firstYear : ℕ
  secondYear : ℕ
  thirdYear : ℕ

/-- Calculates the total number of students -/
def totalStudents (pop : StudentPopulation) : ℕ :=
  pop.firstYear + pop.secondYear + pop.thirdYear

/-- Calculates the number of students selected from a specific year -/
def selectedFromYear (pop : StudentPopulation) (year : ℕ) (sampleSize : ℕ) : ℕ :=
  (year * sampleSize) / totalStudents pop

/-- Theorem: The number of third-year students selected in the stratified sampling -/
theorem thirdYearSelected (pop : StudentPopulation) (sampleSize : ℕ) :
  pop.firstYear = 150 →
  pop.secondYear = 120 →
  pop.thirdYear = 180 →
  sampleSize = 50 →
  selectedFromYear pop pop.thirdYear sampleSize = 20 := by
  sorry


end thirdYearSelected_l83_8368


namespace point_D_coordinates_l83_8372

def vector_AB : ℝ × ℝ := (5, -3)
def point_C : ℝ × ℝ := (-1, 3)

theorem point_D_coordinates :
  ∀ (D : ℝ × ℝ),
  (D.1 - point_C.1, D.2 - point_C.2) = (2 * vector_AB.1, 2 * vector_AB.2) →
  D = (9, -3) := by
  sorry

end point_D_coordinates_l83_8372


namespace distance_to_village_l83_8376

theorem distance_to_village (d : ℝ) : 
  (¬(d ≥ 8) ∧ ¬(d ≤ 7) ∧ ¬(d ≤ 6) ∧ ¬(d ≥ 10)) → 
  (d > 7 ∧ d < 8) :=
sorry

end distance_to_village_l83_8376


namespace tangent_line_at_origin_tangent_line_through_point_l83_8302

-- Define the function f(x) = x³ + 2x
def f (x : ℝ) : ℝ := x^3 + 2*x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 + 2

theorem tangent_line_at_origin :
  ∃ (m : ℝ), ∀ (x : ℝ), (f' 0) * x = m * x ∧ m = 2 :=
sorry

theorem tangent_line_through_point :
  ∃ (m b : ℝ), ∀ (x : ℝ),
    (∃ (x₀ : ℝ), f x₀ = m * x₀ + b ∧ f' x₀ = m) ∧
    (-1 * m + b = -3) ∧
    m = 5 ∧ b = 2 :=
sorry

end tangent_line_at_origin_tangent_line_through_point_l83_8302


namespace count_numbers_theorem_l83_8321

/-- A function that checks if a natural number contains the digit 4 in its decimal representation -/
def contains_four (n : ℕ) : Prop := sorry

/-- The count of numbers from 1 to 1000 that are divisible by 4 and do not contain the digit 4 -/
def count_numbers : ℕ := sorry

/-- Theorem stating that the count of numbers from 1 to 1000 that are divisible by 4 
    and do not contain the digit 4 is equal to 162 -/
theorem count_numbers_theorem : count_numbers = 162 := by sorry

end count_numbers_theorem_l83_8321


namespace parallel_lines_point_l83_8331

def line (m : ℝ) (b : ℝ) := {p : ℝ × ℝ | p.2 = m * p.1 + b}

def parallel (l₁ l₂ : Set (ℝ × ℝ)) : Prop :=
  ∃ m b₁ b₂, l₁ = line m b₁ ∧ l₂ = line m b₂

def angle_of_inclination (l : Set (ℝ × ℝ)) (θ : ℝ) : Prop :=
  ∃ m b, l = line m b ∧ m = Real.tan θ

theorem parallel_lines_point (a : ℝ) : 
  let l₁ : Set (ℝ × ℝ) := line 1 0
  let l₂ : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 = -2 ∧ p.2 = -1) ∨ (p.1 = 3 ∧ p.2 = a)}
  angle_of_inclination l₁ (π/4) → parallel l₁ l₂ → a = 4 := by
  sorry

end parallel_lines_point_l83_8331


namespace pizza_dough_milk_calculation_l83_8300

/-- Given a ratio of milk to flour for pizza dough, calculate the amount of milk needed for a given amount of flour -/
theorem pizza_dough_milk_calculation 
  (milk_per_portion : ℝ) 
  (flour_per_portion : ℝ) 
  (total_flour : ℝ) 
  (h1 : milk_per_portion = 50) 
  (h2 : flour_per_portion = 250) 
  (h3 : total_flour = 750) :
  (total_flour / flour_per_portion) * milk_per_portion = 150 :=
by sorry

end pizza_dough_milk_calculation_l83_8300


namespace chairs_per_round_table_l83_8336

/-- Proves that each round table has 6 chairs in the office canteen -/
theorem chairs_per_round_table : 
  ∀ (x : ℕ),
  (2 * x + 2 * 7 = 26) →
  x = 6 := by
sorry

end chairs_per_round_table_l83_8336


namespace simplify_and_square_l83_8315

theorem simplify_and_square : (8 * (15 / 9) * (-45 / 50))^2 = 144 := by
  sorry

end simplify_and_square_l83_8315


namespace f_properties_l83_8360

noncomputable def f (x : ℝ) : ℝ := x / (1 - |x|)

noncomputable def g (x : ℝ) : ℝ := f x + x

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧ 
  (Set.range f = Set.univ) ∧
  (∃! (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ g a = 0 ∧ g b = 0 ∧ g c = 0) :=
sorry

end f_properties_l83_8360


namespace amp_four_neg_three_l83_8354

-- Define the & operation
def amp (x y : ℤ) : ℤ := x * (y + 2) + x * y

-- Theorem statement
theorem amp_four_neg_three : amp 4 (-3) = -16 := by
  sorry

end amp_four_neg_three_l83_8354


namespace cookie_bags_l83_8329

theorem cookie_bags (total_cookies : ℕ) (cookies_per_bag : ℕ) (h1 : total_cookies = 703) (h2 : cookies_per_bag = 19) :
  total_cookies / cookies_per_bag = 37 := by
  sorry

end cookie_bags_l83_8329


namespace math_course_scheduling_l83_8322

theorem math_course_scheduling (n : ℕ) (k : ℕ) (courses : ℕ) : 
  n = 6 → k = 3 → courses = 3 →
  (Nat.choose (n - k + 1) k) * (Nat.factorial courses) = 24 := by
sorry

end math_course_scheduling_l83_8322


namespace smallest_assembly_size_l83_8346

theorem smallest_assembly_size : ∃ n : ℕ, n > 50 ∧ 
  (∃ x : ℕ, n = 4 * x + (x + 2)) ∧ 
  (∀ m : ℕ, m > 50 → (∃ y : ℕ, m = 4 * y + (y + 2)) → m ≥ n) ∧
  n = 52 :=
by sorry

end smallest_assembly_size_l83_8346


namespace fourth_person_height_l83_8343

/-- Given 4 people with heights in increasing order, prove that the 4th person is 85 inches tall. -/
theorem fourth_person_height (h₁ h₂ h₃ h₄ : ℝ) : 
  (h₁ < h₂) ∧ (h₂ < h₃) ∧ (h₃ < h₄) ∧  -- Heights in increasing order
  (h₂ - h₁ = 2) ∧                      -- Difference between 1st and 2nd
  (h₃ - h₂ = 2) ∧                      -- Difference between 2nd and 3rd
  (h₄ - h₃ = 6) ∧                      -- Difference between 3rd and 4th
  ((h₁ + h₂ + h₃ + h₄) / 4 = 79)       -- Average height
  → h₄ = 85 := by
sorry

end fourth_person_height_l83_8343


namespace rational_absolute_value_equality_l83_8324

theorem rational_absolute_value_equality (a : ℚ) : 
  |(-3 - a)| = 3 + |a| → a ≥ 0 := by
  sorry

end rational_absolute_value_equality_l83_8324


namespace fraction_subtraction_simplification_l83_8313

theorem fraction_subtraction_simplification :
  (7 : ℚ) / 17 - (4 : ℚ) / 51 = (1 : ℚ) / 3 := by
  sorry

end fraction_subtraction_simplification_l83_8313


namespace last_remaining_card_l83_8392

/-- Represents a playing card --/
inductive Card
  | Joker : Bool → Card  -- True for Big Joker, False for Small Joker
  | Regular : Suit → Rank → Card

/-- Represents the suit of a card --/
inductive Suit
  | Spades | Hearts | Diamonds | Clubs

/-- Represents the rank of a card --/
inductive Rank
  | Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- Represents a deck of cards --/
def Deck := List Card

/-- Creates a standard deck of cards in the specified order --/
def createDeck : Deck := sorry

/-- Combines two decks of cards --/
def combinedDecks : Deck := sorry

/-- Simulates the process of discarding and moving cards --/
def discardAndMove (deck : Deck) : Card := sorry

/-- Theorem stating that the last remaining card is the 6 of Diamonds from the second deck --/
theorem last_remaining_card :
  discardAndMove combinedDecks = Card.Regular Suit.Diamonds Rank.Six := by sorry

end last_remaining_card_l83_8392


namespace labeled_cube_probabilities_l83_8389

/-- A cube with 6 faces, where 1 face is labeled with 1, 2 faces are labeled with 2, and 3 faces are labeled with 3 -/
structure LabeledCube where
  total_faces : ℕ
  faces_with_1 : ℕ
  faces_with_2 : ℕ
  faces_with_3 : ℕ
  face_sum : total_faces = faces_with_1 + faces_with_2 + faces_with_3
  face_distribution : faces_with_1 = 1 ∧ faces_with_2 = 2 ∧ faces_with_3 = 3

/-- The probability of an event occurring when rolling the cube -/
def probability (cube : LabeledCube) (favorable_outcomes : ℕ) : ℚ :=
  favorable_outcomes / cube.total_faces

theorem labeled_cube_probabilities (cube : LabeledCube) :
  (probability cube cube.faces_with_2 = 1/3) ∧
  (∀ n, probability cube (cube.faces_with_1) ≤ probability cube n ∧
        probability cube (cube.faces_with_2) ≤ probability cube n →
        n = cube.faces_with_3) ∧
  (probability cube (cube.faces_with_1 + cube.faces_with_2) =
   probability cube cube.faces_with_3) :=
by sorry

end labeled_cube_probabilities_l83_8389


namespace angle_B_is_80_l83_8347

-- Define the quadrilateral and its properties
structure Quadrilateral where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  x : ℝ
  BEC : ℝ

-- Define the conditions
def quadrilateral_conditions (q : Quadrilateral) : Prop :=
  q.A = 60 ∧
  q.B = 2 * q.C ∧
  q.D = 2 * q.C - q.x ∧
  q.x > 0 ∧
  q.A + q.B + q.C + q.D = 360 ∧
  q.BEC = 20

-- Theorem statement
theorem angle_B_is_80 (q : Quadrilateral) 
  (h : quadrilateral_conditions q) : q.B = 80 :=
sorry

end angle_B_is_80_l83_8347


namespace propositions_p_and_q_l83_8357

theorem propositions_p_and_q :
  (¬ ∀ x : ℝ, x^2 ≥ x) ∧ (∃ x : ℝ, x^2 ≥ x) := by
  sorry

end propositions_p_and_q_l83_8357


namespace min_value_quadratic_l83_8356

theorem min_value_quadratic (x y : ℝ) (h : x + y = 4) :
  ∃ (m : ℝ), m = 12 ∧ ∀ (a b : ℝ), a + b = 4 → 3 * a^2 + b^2 ≥ m :=
by sorry

end min_value_quadratic_l83_8356


namespace division_problem_l83_8327

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 265 →
  divisor = 22 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  quotient = 12 := by
sorry

end division_problem_l83_8327


namespace complex_ratio_pure_imaginary_l83_8326

theorem complex_ratio_pure_imaginary (a : ℝ) : 
  let z₁ : ℂ := a + 3*I
  let z₂ : ℂ := 3 + 4*I
  (∃ (b : ℝ), z₁ / z₂ = b*I) → a = -4 :=
by
  sorry

end complex_ratio_pure_imaginary_l83_8326


namespace isosceles_triangle_count_l83_8325

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  a : Point
  b : Point
  c : Point

/-- Represents the set of all colored points in the triangle -/
def ColoredPoints (t : EquilateralTriangle) : Set Point :=
  sorry

/-- Checks if a triangle formed by three points is isosceles -/
def IsIsosceles (p1 p2 p3 : Point) : Prop :=
  sorry

/-- Counts the number of isosceles triangles formed by the colored points -/
def CountIsoscelesTriangles (t : EquilateralTriangle) : ℕ :=
  sorry

/-- Main theorem: There are exactly 18 isosceles triangles with vertices at the colored points -/
theorem isosceles_triangle_count (t : EquilateralTriangle) :
  CountIsoscelesTriangles t = 18 :=
sorry

end isosceles_triangle_count_l83_8325


namespace X_mod_100_l83_8377

/-- The number of sequences satisfying the given conditions -/
def X : ℕ := sorry

/-- Condition: Each aᵢ is either 0 or a power of 2 -/
def is_valid_element (a : ℕ) : Prop :=
  a = 0 ∨ ∃ k : ℕ, a = 2^k

/-- Condition: aᵢ = a₂ᵢ + a₂ᵢ₊₁ for 1 ≤ i ≤ 1023 -/
def satisfies_sum_condition (a : ℕ → ℕ) : Prop :=
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ 1023 → a i = a (2*i) + a (2*i + 1)

/-- All conditions for the sequence -/
def valid_sequence (a : ℕ → ℕ) : Prop :=
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ 2047 → is_valid_element (a i)) ∧
  satisfies_sum_condition a ∧
  a 1 = 1024

theorem X_mod_100 : X % 100 = 15 := by sorry

end X_mod_100_l83_8377


namespace negation_of_proposition_l83_8374

theorem negation_of_proposition (p : Prop) : 
  (¬ (∃ x : ℝ, x > 0 ∧ x^2 - 3*x + 2 > 0)) ↔ 
  (∀ x : ℝ, x > 0 → x^2 - 3*x + 2 ≤ 0) :=
by sorry

end negation_of_proposition_l83_8374
