import Mathlib

namespace average_of_combined_sets_l3594_359456

theorem average_of_combined_sets :
  ∀ (n₁ n₂ : ℕ) (avg₁ avg₂ : ℚ),
    n₁ = 30 →
    n₂ = 20 →
    avg₁ = 20 →
    avg₂ = 30 →
    (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂) = 24 := by
  sorry

end average_of_combined_sets_l3594_359456


namespace investment_profit_distribution_l3594_359437

/-- Represents the investment and profit distribution problem -/
theorem investment_profit_distribution 
  (total_capital : ℕ) 
  (total_profit : ℕ) 
  (a_invest_diff : ℕ) 
  (b_invest_diff : ℕ) 
  (d_invest_diff : ℕ) 
  (a_duration b_duration c_duration d_duration : ℕ) 
  (h1 : total_capital = 100000)
  (h2 : total_profit = 50000)
  (h3 : a_invest_diff = 10000)
  (h4 : b_invest_diff = 5000)
  (h5 : d_invest_diff = 8000)
  (h6 : a_duration = 12)
  (h7 : b_duration = 10)
  (h8 : c_duration = 8)
  (h9 : d_duration = 6) :
  ∃ (c_invest : ℕ),
    let b_invest := c_invest + b_invest_diff
    let a_invest := b_invest + a_invest_diff
    let d_invest := a_invest + d_invest_diff
    c_invest + b_invest + a_invest + d_invest = total_capital ∧
    (b_invest * b_duration : ℚ) / ((c_invest * c_duration + b_invest * b_duration + a_invest * a_duration + d_invest * d_duration) : ℚ) * total_profit = 10925 := by
  sorry

end investment_profit_distribution_l3594_359437


namespace probability_is_half_l3594_359470

/-- An isosceles triangle with 45-degree base angles -/
structure IsoscelesTriangle45 where
  -- We don't need to define the specific geometry, just that it exists
  exists_triangle : True

/-- The triangle is divided into six equal areas -/
def divided_into_six_areas (t : IsoscelesTriangle45) : Prop :=
  ∃ (areas : Finset ℝ), areas.card = 6 ∧ ∀ a ∈ areas, a > 0 ∧ (∀ b ∈ areas, a = b)

/-- Three areas are selected -/
def three_areas_selected (t : IsoscelesTriangle45) (areas : Finset ℝ) : Prop :=
  ∃ (selected : Finset ℝ), selected ⊆ areas ∧ selected.card = 3

/-- The probability of a point falling in the selected areas -/
def probability_in_selected (t : IsoscelesTriangle45) (areas selected : Finset ℝ) : ℚ :=
  (selected.card : ℚ) / (areas.card : ℚ)

/-- The main theorem -/
theorem probability_is_half (t : IsoscelesTriangle45) 
  (h1 : divided_into_six_areas t)
  (h2 : ∃ areas selected, three_areas_selected t areas ∧ selected ⊆ areas) :
  ∃ areas selected, three_areas_selected t areas ∧ 
    probability_in_selected t areas selected = 1/2 := by
  sorry

end probability_is_half_l3594_359470


namespace common_chord_length_l3594_359487

/-- Curve C1 defined by (x-1)^2 + y^2 = 4 -/
def C1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

/-- Curve C2 defined by x^2 + y^2 = 4y -/
def C2 (x y : ℝ) : Prop := x^2 + y^2 = 4*y

/-- The length of the common chord between C1 and C2 is √11 -/
theorem common_chord_length :
  ∃ (a b c d : ℝ), C1 a b ∧ C1 c d ∧ C2 a b ∧ C2 c d ∧
  ((a - c)^2 + (b - d)^2)^(1/2 : ℝ) = Real.sqrt 11 :=
sorry

end common_chord_length_l3594_359487


namespace point_on_axes_l3594_359499

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The coordinate axes, represented as a set of points -/
def CoordinateAxes : Set Point :=
  {p : Point | p.x = 0 ∨ p.y = 0}

/-- Theorem: If xy = 0, then the point is on the coordinate axes -/
theorem point_on_axes (p : Point) (h : p.x * p.y = 0) : p ∈ CoordinateAxes := by
  sorry

end point_on_axes_l3594_359499


namespace intersection_area_theorem_l3594_359434

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  edge_length : ℝ
  vertex : Point3D

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the area of the polygon formed by the intersection of a plane and a cube -/
def intersectionArea (c : Cube) (p : Plane) : ℝ := sorry

/-- Theorem stating the area of the intersection polygon -/
theorem intersection_area_theorem (c : Cube) (p q r : Point3D) : 
  c.edge_length = 30 →
  p.x = 10 ∧ p.y = 0 ∧ p.z = 0 →
  q.x = 30 ∧ q.y = 0 ∧ q.z = 10 →
  r.x = 30 ∧ r.y = 20 ∧ r.z = 30 →
  ∃ (plane : Plane), intersectionArea c plane = 450 := by
  sorry

#check intersection_area_theorem

end intersection_area_theorem_l3594_359434


namespace cost_price_is_65_l3594_359417

/-- Given a cloth sale scenario, calculate the cost price per metre. -/
def cost_price_per_metre (total_metres : ℕ) (total_price : ℕ) (loss_per_metre : ℕ) : ℕ :=
  total_price / total_metres + loss_per_metre

/-- Theorem stating that the cost price per metre is 65 given the problem conditions. -/
theorem cost_price_is_65 :
  cost_price_per_metre 300 18000 5 = 65 := by
  sorry

end cost_price_is_65_l3594_359417


namespace count_divisible_by_11_is_18_l3594_359454

/-- The number obtained by writing the integers 1 to n from left to right -/
def b (n : ℕ) : ℕ := sorry

/-- The count of b_k divisible by 11 for 1 ≤ k ≤ 100 -/
def count_divisible_by_11 : ℕ := sorry

theorem count_divisible_by_11_is_18 : count_divisible_by_11 = 18 := by sorry

end count_divisible_by_11_is_18_l3594_359454


namespace volunteer_selection_l3594_359460

/-- The number of ways to select exactly one person to serve both days
    given 5 volunteers and 2 days of service where 2 people are selected each day. -/
theorem volunteer_selection (n : ℕ) (d : ℕ) (s : ℕ) (p : ℕ) : 
  n = 5 → d = 2 → s = 2 → p = 1 →
  (n.choose p) * ((n - p).choose (s - p)) * ((n - s).choose (s - p)) = 60 :=
by sorry

end volunteer_selection_l3594_359460


namespace prob_one_male_correct_prob_at_least_one_female_correct_l3594_359482

/-- Represents the number of female students in the group -/
def num_females : ℕ := 2

/-- Represents the number of male students in the group -/
def num_males : ℕ := 3

/-- Represents the total number of students in the group -/
def total_students : ℕ := num_females + num_males

/-- Represents the number of students to be selected -/
def num_selected : ℕ := 2

/-- Calculates the probability of selecting exactly one male student -/
def prob_one_male : ℚ := 3 / 5

/-- Calculates the probability of selecting at least one female student -/
def prob_at_least_one_female : ℚ := 7 / 10

/-- Proves that the probability of selecting exactly one male student is 3/5 -/
theorem prob_one_male_correct : 
  prob_one_male = (num_females * num_males : ℚ) / (total_students.choose num_selected : ℚ) := by
  sorry

/-- Proves that the probability of selecting at least one female student is 7/10 -/
theorem prob_at_least_one_female_correct :
  prob_at_least_one_female = 1 - ((num_males.choose num_selected : ℚ) / (total_students.choose num_selected : ℚ)) := by
  sorry

end prob_one_male_correct_prob_at_least_one_female_correct_l3594_359482


namespace mandy_reading_progression_l3594_359420

/-- Calculates the present book length given Mandy's reading progression --/
def present_book_length (starting_age : ℕ) (starting_length : ℕ) 
  (double_age_multiplier : ℕ) (eight_years_later_multiplier : ℕ) 
  (present_multiplier : ℕ) : ℕ :=
  let double_age_length := starting_length * double_age_multiplier
  let eight_years_later_length := double_age_length * eight_years_later_multiplier
  eight_years_later_length * present_multiplier

/-- Theorem stating that the present book length is 480 pages --/
theorem mandy_reading_progression : 
  present_book_length 6 8 5 3 4 = 480 := by
  sorry

#eval present_book_length 6 8 5 3 4

end mandy_reading_progression_l3594_359420


namespace article_large_font_pages_l3594_359416

/-- Represents the number of pages in large font for an article -/
def large_font_pages (total_words : ℕ) (words_per_large_page : ℕ) (words_per_small_page : ℕ) (total_pages : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of large font pages in the article -/
theorem article_large_font_pages :
  large_font_pages 48000 1800 2400 21 = 4 := by
  sorry

end article_large_font_pages_l3594_359416


namespace bubble_sort_probability_l3594_359472

theorem bubble_sort_probability (n : ℕ) (h : n = 36) :
  let arrangements := n.factorial
  let favorable_outcomes := (n - 2).factorial
  (favorable_outcomes : ℚ) / arrangements = 1 / 1260 := by
  sorry

end bubble_sort_probability_l3594_359472


namespace fraction_value_l3594_359453

theorem fraction_value (x y : ℝ) (h1 : 1 < (x - y) / (x + y)) (h2 : (x - y) / (x + y) < 3) (h3 : ∃ (n : ℤ), x / y = ↑n) : x / y = 4 := by
  sorry

end fraction_value_l3594_359453


namespace price_difference_is_1090_l3594_359481

/-- The difference in cents between the TV advertiser price and the in-store price for a microwave --/
def price_difference : ℚ :=
  let in_store_price : ℚ := 149.95
  let tv_payment : ℚ := 27.99
  let shipping_fee : ℚ := 14.95
  let warranty_fee : ℚ := 5.95
  let tv_price : ℚ := 5 * tv_payment + shipping_fee + warranty_fee
  (tv_price - in_store_price) * 100

/-- The price difference is 1090 cents --/
theorem price_difference_is_1090 : 
  price_difference = 1090 := by sorry

end price_difference_is_1090_l3594_359481


namespace only_negative_number_l3594_359410

theorem only_negative_number (a b c d : ℝ) : 
  a = 2023 ∧ b = -2023 ∧ c = 1/2023 ∧ d = 0 →
  (b < 0 ∧ a ≥ 0 ∧ c > 0 ∧ d = 0) := by
  sorry

end only_negative_number_l3594_359410


namespace nell_cards_to_john_l3594_359430

def cards_problem (initial_cards : ℕ) (cards_to_jeff : ℕ) (cards_left : ℕ) : Prop :=
  let total_given_away := initial_cards - cards_left
  let cards_to_john := total_given_away - cards_to_jeff
  cards_to_john = 195

theorem nell_cards_to_john :
  cards_problem 573 168 210 :=
by
  sorry

end nell_cards_to_john_l3594_359430


namespace orange_harvest_sacks_l3594_359491

/-- Proves that harvesting 38 sacks per day for 49 days results in 1862 sacks total. -/
theorem orange_harvest_sacks (daily_harvest : ℕ) (days : ℕ) (total_sacks : ℕ) 
  (h1 : daily_harvest = 38)
  (h2 : days = 49)
  (h3 : total_sacks = 1862) :
  daily_harvest * days = total_sacks :=
by sorry

end orange_harvest_sacks_l3594_359491


namespace hall_tiling_proof_l3594_359439

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℚ
  width : ℚ

/-- Converts inches to feet -/
def inchesToFeet (inches : ℚ) : ℚ := inches / 12

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℚ := d.length * d.width

/-- Calculates the number of smaller rectangles needed to cover a larger rectangle -/
def tilesRequired (hall : Dimensions) (tile : Dimensions) : ℕ :=
  (area hall / area tile).ceil.toNat

theorem hall_tiling_proof :
  let hall : Dimensions := { length := 15, width := 18 }
  let tile : Dimensions := { length := inchesToFeet 3, width := inchesToFeet 9 }
  tilesRequired hall tile = 1440 := by
  sorry

end hall_tiling_proof_l3594_359439


namespace simplify_expression_l3594_359431

theorem simplify_expression (x y : ℝ) (h : x * y ≠ 0) :
  ((x^3 + 2) / x) * ((y^3 + 2) / y) - ((x^3 - 2) / y) * ((y^3 - 2) / x) = 4 * (x^2 / y + y^2 / x) := by
  sorry

end simplify_expression_l3594_359431


namespace min_sum_a_b_is_six_l3594_359450

/-- Given that the roots of x^2 + ax + 2b = 0 and x^2 + 2bx + a = 0 are both real,
    and a, b > 0, the minimum value of a + b is 6. -/
theorem min_sum_a_b_is_six (a b : ℝ) 
    (h1 : a > 0)
    (h2 : b > 0)
    (h3 : a^2 - 8*b ≥ 0)  -- Condition for real roots of x^2 + ax + 2b = 0
    (h4 : 4*b^2 - 4*a ≥ 0)  -- Condition for real roots of x^2 + 2bx + a = 0
    : ∀ a' b' : ℝ, a' > 0 → b' > 0 → a'^2 - 8*b' ≥ 0 → 4*b'^2 - 4*a' ≥ 0 → a + b ≤ a' + b' :=
by sorry

end min_sum_a_b_is_six_l3594_359450


namespace odd_prime_equation_l3594_359479

theorem odd_prime_equation (p a b : ℕ) : 
  Prime p → 
  Odd p → 
  a > 0 → 
  b > 0 → 
  (p + 1)^a - p^b = 1 → 
  a = 1 ∧ b = 1 := by
  sorry

end odd_prime_equation_l3594_359479


namespace puppies_left_l3594_359441

/-- The number of puppies Alyssa had initially -/
def initial_puppies : ℕ := 7

/-- The number of puppies Alyssa gave to her friends -/
def given_puppies : ℕ := 5

/-- Theorem: Alyssa is left with 2 puppies -/
theorem puppies_left : initial_puppies - given_puppies = 2 := by
  sorry

end puppies_left_l3594_359441


namespace cosine_sine_ratio_equals_sqrt_three_l3594_359480

theorem cosine_sine_ratio_equals_sqrt_three : 
  (2 * Real.cos (10 * π / 180) - Real.sin (20 * π / 180)) / Real.cos (20 * π / 180) = Real.sqrt 3 := by
  sorry

end cosine_sine_ratio_equals_sqrt_three_l3594_359480


namespace battle_station_staffing_l3594_359401

/-- Represents the number of ways to staff Captain Zarnin's battle station -/
def staff_battle_station (total_applicants : ℕ) (suitable_resumes : ℕ) 
  (assistant_engineer : ℕ) (weapons_maintenance1 : ℕ) (weapons_maintenance2 : ℕ)
  (field_technician : ℕ) (radio_specialist : ℕ) : ℕ :=
  assistant_engineer * weapons_maintenance1 * weapons_maintenance2 * field_technician * radio_specialist

/-- Theorem stating the number of ways to staff the battle station -/
theorem battle_station_staffing :
  staff_battle_station 30 15 3 4 4 5 5 = 960 := by
  sorry

end battle_station_staffing_l3594_359401


namespace power_seven_mod_eight_l3594_359404

theorem power_seven_mod_eight : 7^51 % 8 = 7 := by
  sorry

end power_seven_mod_eight_l3594_359404


namespace inequality_system_solution_set_l3594_359485

theorem inequality_system_solution_set :
  let S := {x : ℝ | (1 + x > -1) ∧ (4 - 2*x ≥ 0)}
  S = {x : ℝ | -2 < x ∧ x ≤ 2} := by
  sorry

end inequality_system_solution_set_l3594_359485


namespace mary_bike_rental_hours_l3594_359455

/-- Calculates the number of hours a bike was rented given the total payment, fixed fee, and hourly rate. -/
def rent_hours (total_payment fixed_fee hourly_rate : ℚ) : ℚ :=
  (total_payment - fixed_fee) / hourly_rate

/-- Proves that Mary rented the bike for 9 hours given the specified conditions. -/
theorem mary_bike_rental_hours :
  let fixed_fee : ℚ := 17
  let hourly_rate : ℚ := 7
  let total_payment : ℚ := 80
  rent_hours total_payment fixed_fee hourly_rate = 9 := by
  sorry


end mary_bike_rental_hours_l3594_359455


namespace trivia_team_score_l3594_359457

theorem trivia_team_score (total_members : ℕ) (absent_members : ℕ) (total_score : ℕ) :
  total_members = 7 →
  absent_members = 2 →
  total_score = 20 →
  (total_score / (total_members - absent_members) : ℚ) = 4 := by
sorry

end trivia_team_score_l3594_359457


namespace division_problem_l3594_359478

theorem division_problem (L S Q : ℕ) : 
  L - S = 1355 → 
  L = 1608 → 
  L = S * Q + 15 → 
  Q = 6 := by
sorry

end division_problem_l3594_359478


namespace rectangle_dimensions_l3594_359408

/-- Proves that a rectangle with width w and length 3w, whose perimeter is twice its area, has width 4/3 and length 4 -/
theorem rectangle_dimensions (w : ℝ) (h1 : w > 0) : 
  (2 * (w + 3*w) = 2 * (w * 3*w)) → w = 4/3 ∧ 3*w = 4 := by
  sorry

end rectangle_dimensions_l3594_359408


namespace triangle_min_angle_le_60_l3594_359433

theorem triangle_min_angle_le_60 (A B C : ℝ) : 
  A + B + C = 180 → A > 0 → B > 0 → C > 0 → min A (min B C) ≤ 60 := by
  sorry

end triangle_min_angle_le_60_l3594_359433


namespace min_value_x_plus_y_l3594_359490

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y + x * y = 3) :
  ∀ a b : ℝ, a > 0 → b > 0 → a + b + a * b = 3 → x + y ≤ a + b :=
sorry

end min_value_x_plus_y_l3594_359490


namespace fish_estimation_l3594_359407

/-- The number of fish caught and marked on the first day -/
def marked_fish : ℕ := 30

/-- The number of fish caught on the second day -/
def second_catch : ℕ := 40

/-- The number of marked fish caught on the second day -/
def marked_recaught : ℕ := 2

/-- The estimated number of fish in the pond -/
def estimated_fish : ℕ := marked_fish * second_catch / marked_recaught

theorem fish_estimation :
  estimated_fish = 600 :=
sorry

end fish_estimation_l3594_359407


namespace smallest_non_representable_as_cube_sum_l3594_359449

theorem smallest_non_representable_as_cube_sum : ∃ (n : ℕ), n > 0 ∧
  (∀ (m : ℕ), m < n → ∃ (x y : ℤ), m = x^3 + 3*y^3) ∧
  ¬∃ (x y : ℤ), n = x^3 + 3*y^3 ∧ 
  n = 6 := by
  sorry

end smallest_non_representable_as_cube_sum_l3594_359449


namespace third_angle_is_75_l3594_359438

/-- A triangle formed by folding a square piece of paper -/
structure FoldedTriangle where
  /-- Angle formed by splitting a right angle in half -/
  angle_mna : ℝ
  /-- Angle formed by three equal angles adding up to 180° -/
  angle_amn : ℝ
  /-- The third angle of the triangle -/
  angle_anm : ℝ
  /-- Proof that angle_mna is 45° -/
  h_mna : angle_mna = 45
  /-- Proof that angle_amn is 60° -/
  h_amn : angle_amn = 60
  /-- Proof that the sum of all angles is 180° -/
  h_sum : angle_mna + angle_amn + angle_anm = 180

/-- Theorem stating that the third angle is 75° -/
theorem third_angle_is_75 (t : FoldedTriangle) : t.angle_anm = 75 := by
  sorry

end third_angle_is_75_l3594_359438


namespace curve_crosses_at_point_one_eight_l3594_359422

-- Define the curve
def x (t : ℝ) : ℝ := 2 * t^2 + 1
def y (t : ℝ) : ℝ := 2 * t^3 - 6 * t^2 + 8

-- Theorem statement
theorem curve_crosses_at_point_one_eight :
  ∃ (a b : ℝ), a ≠ b ∧ x a = x b ∧ y a = y b ∧ x a = 1 ∧ y a = 8 := by
  sorry

end curve_crosses_at_point_one_eight_l3594_359422


namespace solutions_count_l3594_359409

/-- The number of solutions to the equation √(x+3) = ax + 2 depends on the value of a -/
theorem solutions_count (a : ℝ) :
  (∃! x, Real.sqrt (x + 3) = a * x + 2) ∨
  (¬ ∃ x, Real.sqrt (x + 3) = a * x + 2) ∨
  (∃ x y, x ≠ y ∧ Real.sqrt (x + 3) = a * x + 2 ∧ Real.sqrt (y + 3) = a * y + 2) :=
  by sorry

end solutions_count_l3594_359409


namespace alloy_mixing_solution_exists_l3594_359424

/-- Represents an alloy of copper and tin -/
structure Alloy where
  mass : ℝ
  copper_percentage : ℝ

/-- Proves that a solution exists for the alloy mixing problem if and only if p is within the specified range -/
theorem alloy_mixing_solution_exists (alloy1 alloy2 : Alloy) (target_mass : ℝ) (p : ℝ) :
  alloy1.mass = 3 ∧ 
  alloy1.copper_percentage = 40 ∧
  alloy2.mass = 7 ∧
  alloy2.copper_percentage = 30 ∧
  target_mass = 8 →
  (∃ x : ℝ, 
    0 ≤ x ∧ 
    x ≤ alloy1.mass ∧ 
    0 ≤ target_mass - x ∧ 
    target_mass - x ≤ alloy2.mass ∧
    (alloy1.copper_percentage / 100 * x + alloy2.copper_percentage / 100 * (target_mass - x)) / target_mass = p / 100) ↔
  31.25 ≤ p ∧ p ≤ 33.75 := by
  sorry

#check alloy_mixing_solution_exists

end alloy_mixing_solution_exists_l3594_359424


namespace square_area_from_rectangle_l3594_359419

theorem square_area_from_rectangle (circle_radius : ℝ) (rectangle_length : ℝ) (rectangle_breadth : ℝ) (rectangle_area : ℝ) :
  rectangle_length = (2 / 5) * circle_radius →
  rectangle_breadth = 10 →
  rectangle_area = 160 →
  rectangle_area = rectangle_length * rectangle_breadth →
  (circle_radius ^ 2 : ℝ) = 1600 := by
  sorry

end square_area_from_rectangle_l3594_359419


namespace strawberry_count_l3594_359442

/-- Calculates the total number of strawberries after picking more -/
def total_strawberries (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem: The total number of strawberries is the sum of initial and additional strawberries -/
theorem strawberry_count (initial additional : ℕ) :
  total_strawberries initial additional = initial + additional := by
  sorry

end strawberry_count_l3594_359442


namespace division_problem_l3594_359466

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 15 ∧ quotient = 4 ∧ remainder = 3 ∧ 
  dividend = divisor * quotient + remainder → 
  divisor = 3 := by
sorry

end division_problem_l3594_359466


namespace greatest_plants_per_row_l3594_359436

theorem greatest_plants_per_row (sunflowers corn tomatoes : ℕ) 
  (h_sunflowers : sunflowers = 45)
  (h_corn : corn = 81)
  (h_tomatoes : tomatoes = 63) :
  Nat.gcd sunflowers (Nat.gcd corn tomatoes) = 9 := by
  sorry

end greatest_plants_per_row_l3594_359436


namespace binomial_10_choose_2_l3594_359475

theorem binomial_10_choose_2 : Nat.choose 10 2 = 45 := by
  sorry

end binomial_10_choose_2_l3594_359475


namespace smaller_part_is_eleven_l3594_359414

theorem smaller_part_is_eleven (x y : ℝ) (h1 : x + y = 24) (h2 : 7 * x + 5 * y = 146) : 
  min x y = 11 := by
  sorry

end smaller_part_is_eleven_l3594_359414


namespace average_distance_scientific_notation_l3594_359476

-- Define the average distance between the Earth and the Sun
def average_distance : ℝ := 149600000

-- Define the scientific notation representation
def scientific_notation : ℝ := 1.496 * (10 ^ 8)

-- Theorem to prove the equivalence
theorem average_distance_scientific_notation : average_distance = scientific_notation := by
  sorry

end average_distance_scientific_notation_l3594_359476


namespace cone_height_from_sector_l3594_359489

/-- Given a sector with radius 7 cm and area 21π cm², when used to form the lateral surface of a cone, 
    the height of the cone is 2√10 cm. -/
theorem cone_height_from_sector (r : ℝ) (area : ℝ) (h : ℝ) : 
  r = 7 → 
  area = 21 * Real.pi → 
  area = (1/2) * (2 * Real.pi) * 3 * r → 
  h = Real.sqrt (r^2 - 3^2) → 
  h = 2 * Real.sqrt 10 := by
sorry

end cone_height_from_sector_l3594_359489


namespace mean_problem_l3594_359427

theorem mean_problem (x : ℝ) : 
  (28 + x + 42 + 78 + 104) / 5 = 90 →
  (128 + 255 + 511 + 1023 + x) / 5 = 423 := by
sorry

end mean_problem_l3594_359427


namespace circle_plus_four_two_l3594_359468

/-- Definition of the ⊕ operation for real numbers -/
def circle_plus (a b : ℝ) : ℝ := 2 * a + 5 * b

/-- Theorem stating that 4 ⊕ 2 = 18 -/
theorem circle_plus_four_two : circle_plus 4 2 = 18 := by
  sorry

end circle_plus_four_two_l3594_359468


namespace product_expansion_l3594_359492

theorem product_expansion (y : ℝ) : 4 * (y - 3) * (y + 2) = 4 * y^2 - 4 * y - 24 := by
  sorry

end product_expansion_l3594_359492


namespace complement_A_inter_B_l3594_359496

open Set Real

-- Define the universal set U as ℝ
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {y | ∃ x, y = 3^x + 1}

-- Define set B
def B : Set ℝ := {x | log x < 0}

-- Statement to prove
theorem complement_A_inter_B : 
  (U \ A) ∩ B = {x | 0 < x ∧ x < 1} := by sorry

end complement_A_inter_B_l3594_359496


namespace x_squared_plus_reciprocal_squared_l3594_359440

theorem x_squared_plus_reciprocal_squared (x : ℝ) (h : x + 1/x = 3.5) : 
  x^2 + 1/x^2 = 10.25 := by
sorry

end x_squared_plus_reciprocal_squared_l3594_359440


namespace intersection_of_A_and_B_l3594_359452

def M : Set ℤ := {0, 1, 2}

def A : Set ℤ := {y | ∃ x ∈ M, y = 2 * x}

def B : Set ℤ := {y | ∃ x ∈ M, y = 2 * x - 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 2} := by sorry

end intersection_of_A_and_B_l3594_359452


namespace solution_set_implies_a_range_l3594_359418

theorem solution_set_implies_a_range (a : ℝ) :
  (∀ x, (a - 3) * x > 1 ↔ x < 1 / (a - 3)) →
  a < 3 := by
  sorry

end solution_set_implies_a_range_l3594_359418


namespace fixed_point_of_exponential_function_l3594_359474

/-- Given a > 0 and a ≠ 1, prove that (2, 3) is the fixed point of f(x) = a^(x-2) + 2 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) + 2
  f 2 = 3 ∧ ∀ x : ℝ, f x = x → x = 2 := by
  sorry

end fixed_point_of_exponential_function_l3594_359474


namespace sum_of_digits_inequality_l3594_359423

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_inequality (N : ℕ) :
  sum_of_digits N ≤ 5 * sum_of_digits (5^5 * N) := by sorry

end sum_of_digits_inequality_l3594_359423


namespace food_production_growth_rate_l3594_359428

theorem food_production_growth_rate 
  (initial_production : ℝ) 
  (a b x : ℝ) 
  (h1 : initial_production = 5000)
  (h2 : a > 0)
  (h3 : b > 0)
  (h4 : x > 0)
  (h5 : initial_production * (1 + a) * (1 + b) = initial_production * (1 + x)^2) :
  x ≤ (a + b) / 2 := by
sorry

end food_production_growth_rate_l3594_359428


namespace perpendicular_construction_l3594_359425

/-- A two-sided ruler with parallel edges -/
structure TwoSidedRuler :=
  (width : ℝ)
  (width_pos : width > 0)

/-- A line in a plane -/
structure Line :=
  (a b c : ℝ)
  (not_all_zero : a ≠ 0 ∨ b ≠ 0)

/-- A point in a plane -/
structure Point :=
  (x y : ℝ)

/-- Checks if a point is on a line -/
def Point.on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Checks if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The main theorem -/
theorem perpendicular_construction 
  (l : Line) (M : Point) (h : M.on_line l) :
  ∃ (P : Point), ∃ (n : Line), 
    M.on_line n ∧ P.on_line n ∧ n.perpendicular l :=
sorry

end perpendicular_construction_l3594_359425


namespace marble_probability_l3594_359435

/-- Represents a box of marbles -/
structure MarbleBox where
  total : ℕ
  black : ℕ
  white : ℕ
  sum_check : total = black + white

/-- The problem setup -/
def marble_problem (box1 box2 : MarbleBox) : Prop :=
  box1.total + box2.total = 30 ∧
  box1.black = 3 * box2.black ∧
  (box1.black : ℚ) / box1.total * (box2.black : ℚ) / box2.total = 1/2

theorem marble_probability (box1 box2 : MarbleBox) 
  (h : marble_problem box1 box2) : 
  (box1.white : ℚ) / box1.total * (box2.white : ℚ) / box2.total = 1/3 := by
  sorry

end marble_probability_l3594_359435


namespace kylie_daisies_left_l3594_359464

def daisies_problem (initial_daisies : ℕ) (received_daisies : ℕ) : ℕ :=
  let total_daisies := initial_daisies + received_daisies
  let given_to_mother := total_daisies / 2
  total_daisies - given_to_mother

theorem kylie_daisies_left : daisies_problem 5 9 = 7 := by
  sorry

end kylie_daisies_left_l3594_359464


namespace total_hotdogs_by_wednesday_l3594_359403

def hotdog_sequence (n : ℕ) : ℕ := 10 + 2 * (n - 1)

theorem total_hotdogs_by_wednesday :
  (hotdog_sequence 1) + (hotdog_sequence 2) + (hotdog_sequence 3) = 36 :=
by sorry

end total_hotdogs_by_wednesday_l3594_359403


namespace cone_water_volume_ratio_l3594_359497

theorem cone_water_volume_ratio (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let water_height := (2 / 3) * h
  let water_radius := (2 / 3) * r
  let cone_volume := (1 / 3) * Real.pi * r^2 * h
  let water_volume := (1 / 3) * Real.pi * water_radius^2 * water_height
  water_volume / cone_volume = 8 / 27 := by sorry

end cone_water_volume_ratio_l3594_359497


namespace lanas_winter_clothing_l3594_359443

/-- The number of boxes Lana found -/
def num_boxes : ℕ := 5

/-- The number of scarves in each box -/
def scarves_per_box : ℕ := 7

/-- The number of mittens in each box -/
def mittens_per_box : ℕ := 8

/-- The total number of pieces of winter clothing Lana had -/
def total_clothing : ℕ := num_boxes * scarves_per_box + num_boxes * mittens_per_box

theorem lanas_winter_clothing : total_clothing = 75 := by
  sorry

end lanas_winter_clothing_l3594_359443


namespace roots_properties_l3594_359411

theorem roots_properties (z₁ z₂ : ℂ) (h : x^2 + x + 1 = 0 ↔ x = z₁ ∨ x = z₂) :
  z₁ * z₂ = 1 ∧ z₁^3 = 1 ∧ z₂^3 = 1 := by
  sorry

end roots_properties_l3594_359411


namespace count_polynomials_l3594_359406

-- Define a function to check if an expression is a polynomial
def is_polynomial (expr : String) : Bool :=
  match expr with
  | "3/4x^2" => true
  | "3ab" => true
  | "x+5" => true
  | "y/(5x)" => false
  | "-1" => true
  | "y/3" => true
  | "a^2-b^2" => true
  | "a" => true
  | _ => false

-- Define the list of expressions
def expressions : List String :=
  ["3/4x^2", "3ab", "x+5", "y/(5x)", "-1", "y/3", "a^2-b^2", "a"]

-- Theorem: There are exactly 7 polynomials in the list of expressions
theorem count_polynomials :
  (expressions.filter is_polynomial).length = 7 :=
by sorry

end count_polynomials_l3594_359406


namespace parity_of_cube_plus_multiple_l3594_359467

theorem parity_of_cube_plus_multiple (o n : ℤ) (h_odd : Odd o) :
  Odd (o^3 + n*o) ↔ Even n :=
sorry

end parity_of_cube_plus_multiple_l3594_359467


namespace P_n_roots_P_2018_roots_l3594_359483

-- Define the sequence of polynomials
def P : ℕ → ℝ → ℝ
  | 0, x => 1
  | 1, x => x
  | (n + 2), x => x * P (n + 1) x - P n x

-- Define a function to count distinct real roots
noncomputable def count_distinct_real_roots (f : ℝ → ℝ) : ℕ := sorry

-- Theorem statement
theorem P_n_roots (n : ℕ) : count_distinct_real_roots (P n) = n := by
  sorry

-- Specific case for P_2018
theorem P_2018_roots : count_distinct_real_roots (P 2018) = 2018 := by
  sorry

end P_n_roots_P_2018_roots_l3594_359483


namespace f_is_quadratic_l3594_359484

-- Define what a quadratic equation is
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the specific function we're checking
def f (x : ℝ) : ℝ := x^2 - 2

-- Theorem statement
theorem f_is_quadratic : is_quadratic f := by
  sorry

end f_is_quadratic_l3594_359484


namespace dedekind_cut_property_l3594_359495

-- Define a Dedekind cut
def DedekindCut (M N : Set ℚ) : Prop :=
  (M ∪ N = Set.univ) ∧ 
  (M ∩ N = ∅) ∧ 
  (∀ x ∈ M, ∀ y ∈ N, x < y) ∧
  M.Nonempty ∧ 
  N.Nonempty

-- Theorem stating the impossibility of M having a largest element and N having a smallest element
theorem dedekind_cut_property (M N : Set ℚ) (h : DedekindCut M N) :
  ¬(∃ (m : ℚ), m ∈ M ∧ ∀ x ∈ M, x ≤ m) ∨ ¬(∃ (n : ℚ), n ∈ N ∧ ∀ y ∈ N, n ≤ y) :=
sorry

end dedekind_cut_property_l3594_359495


namespace weight_of_replaced_person_l3594_359451

theorem weight_of_replaced_person (n : ℕ) (avg_increase : ℝ) (new_weight : ℝ) :
  n = 8 →
  avg_increase = 2.5 →
  new_weight = 75 →
  ∃ (old_weight : ℝ), old_weight = 55 ∧ n * avg_increase = new_weight - old_weight :=
by sorry

end weight_of_replaced_person_l3594_359451


namespace z_in_second_quadrant_l3594_359426

-- Define the complex number z
def z : ℂ := (1 + Complex.I) * (2 * Complex.I)

-- Theorem statement
theorem z_in_second_quadrant : Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 := by
  sorry

end z_in_second_quadrant_l3594_359426


namespace geometric_sequence_a6_l3594_359415

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_a6 (a : ℕ → ℝ) :
  geometric_sequence a → a 4 = 2 → a 8 = 32 → a 6 = 8 := by
  sorry

end geometric_sequence_a6_l3594_359415


namespace shelves_needed_l3594_359493

theorem shelves_needed (total_books : ℕ) (books_taken : ℕ) (books_per_shelf : ℕ) 
  (h1 : total_books = 46)
  (h2 : books_taken = 10)
  (h3 : books_per_shelf = 4)
  (h4 : books_per_shelf > 0) :
  (total_books - books_taken) / books_per_shelf = 9 :=
by sorry

end shelves_needed_l3594_359493


namespace children_age_sum_l3594_359461

theorem children_age_sum :
  let num_children : ℕ := 5
  let age_interval : ℕ := 3
  let youngest_age : ℕ := 6
  let ages : List ℕ := List.range num_children |>.map (fun i => youngest_age + i * age_interval)
  ages.sum = 60 := by
  sorry

end children_age_sum_l3594_359461


namespace computer_software_price_sum_l3594_359477

theorem computer_software_price_sum : 
  ∀ (b a : ℝ),
  (b + 0.3 * b = 351) →
  (a + 0.05 * a = 420) →
  2 * b + 2 * a = 1340 :=
by
  sorry

end computer_software_price_sum_l3594_359477


namespace max_volume_angle_l3594_359445

/-- A square ABCD folded along diagonal AC to form a regular pyramid -/
structure FoldedSquare where
  side : ℝ
  fold_angle : ℝ

/-- The angle between line BD and plane ABC in the folded square -/
def angle_bd_abc (s : FoldedSquare) : ℝ := sorry

/-- The volume of the pyramid formed by the folded square -/
def pyramid_volume (s : FoldedSquare) : ℝ := sorry

theorem max_volume_angle (s : FoldedSquare) :
  (∀ t : FoldedSquare, pyramid_volume t ≤ pyramid_volume s) →
  angle_bd_abc s = 45 := by sorry

end max_volume_angle_l3594_359445


namespace intersection_symmetric_implies_p_range_l3594_359444

/-- The line equation: x = ky - 1 -/
def line_equation (k : ℝ) (x y : ℝ) : Prop := x = k * y - 1

/-- The circle equation: x² + y² + kx + my + 2p = 0 -/
def circle_equation (k m p : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + k*x + m*y + 2*p = 0

/-- Two points (x₁, y₁) and (x₂, y₂) are symmetric about y = x -/
def symmetric_about_y_eq_x (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = y₂ ∧ y₁ = x₂

theorem intersection_symmetric_implies_p_range
  (k m p : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    line_equation k x₁ y₁ ∧
    line_equation k x₂ y₂ ∧
    circle_equation k m p x₁ y₁ ∧
    circle_equation k m p x₂ y₂ ∧
    symmetric_about_y_eq_x x₁ y₁ x₂ y₂) →
  p < -3/2 :=
by sorry

end intersection_symmetric_implies_p_range_l3594_359444


namespace geometric_sequence_linear_system_l3594_359469

theorem geometric_sequence_linear_system (a : ℕ → ℝ) (q : ℝ) (h : q ≠ 0) :
  (∀ n : ℕ, a (n + 1) = q * a n) →
  (∃ x y : ℝ, a 1 * x + a 3 * y = 2 ∧ a 2 * x + a 4 * y = 1) ↔ q = 1 / 2 :=
by sorry

end geometric_sequence_linear_system_l3594_359469


namespace area_of_triangle_ABC_l3594_359432

-- Define the points
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (7, 0)
def D : ℝ × ℝ := (0, 4)

-- Define the length of AC
def AC : ℝ := 15

-- Theorem statement
theorem area_of_triangle_ABC :
  let A : ℝ × ℝ := (0, 4) -- We know A is on the y-axis at (0,4) because D is at (0,4)
  (1/2 : ℝ) * ‖(A.1 - B.1, A.2 - B.2)‖ * ‖(C.1 - B.1, C.2 - B.2)‖ = 2 * Real.sqrt 209 := by
sorry


end area_of_triangle_ABC_l3594_359432


namespace transylvanian_truth_telling_l3594_359463

-- Define the types
inductive Being
| Human
| Vampire

-- Define the properties
def declares (b : Being) (x : Prop) : Prop :=
  match b with
  | Being.Human => x
  | Being.Vampire => ¬x

theorem transylvanian_truth_telling (b : Being) (x : Prop) :
  (b = Being.Human → (declares b x → x)) ∧
  (b = Being.Vampire → (declares b x → ¬x)) :=
by sorry

end transylvanian_truth_telling_l3594_359463


namespace lcm_12_18_l3594_359471

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l3594_359471


namespace movie_theater_seating_l3594_359446

/-- The number of ways to seat people in a row of seats with constraints -/
def seating_arrangements (total_seats : ℕ) (people : ℕ) : ℕ :=
  -- Define the function to calculate the number of seating arrangements
  sorry

/-- Theorem stating the number of seating arrangements for the specific problem -/
theorem movie_theater_seating : seating_arrangements 9 3 = 42 := by
  sorry

end movie_theater_seating_l3594_359446


namespace cube_root_equation_solution_l3594_359413

theorem cube_root_equation_solution :
  ∃ x : ℝ, (x - 5)^3 = (1/27)⁻¹ ∧ x = 8 :=
by
  sorry

end cube_root_equation_solution_l3594_359413


namespace min_distance_to_line_l3594_359494

/-- Given that 5x + 12y = 60, the minimum value of √(x² + y²) is 60/13 -/
theorem min_distance_to_line (x y : ℝ) (h : 5 * x + 12 * y = 60) :
  ∃ (min_val : ℝ), min_val = 60 / 13 ∧ 
  ∀ (x' y' : ℝ), 5 * x' + 12 * y' = 60 → Real.sqrt (x' ^ 2 + y' ^ 2) ≥ min_val := by
  sorry


end min_distance_to_line_l3594_359494


namespace quadratic_inequality_theorem_l3594_359459

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Define the solution set
def solution_set (a b c : ℝ) := {x : ℝ | f a b c x ≤ 0}

-- State the theorem
theorem quadratic_inequality_theorem 
  (a b c : ℝ) 
  (h : solution_set a b c = {x : ℝ | x ≤ -1 ∨ x ≥ 3}) :
  (a + b + c > 0) ∧ 
  (4*a - 2*b + c < 0) ∧ 
  ({x : ℝ | c*x^2 - b*x + a < 0} = {x : ℝ | -1/3 < x ∧ x < 1}) :=
by sorry

end quadratic_inequality_theorem_l3594_359459


namespace x_range_l3594_359412

theorem x_range (m : ℝ) (h1 : 0 < m) (h2 : m ≤ 5) :
  (∀ x : ℝ, x^2 + (2*m - 1)*x > 4*x + 2*m - 4) →
  (∀ x : ℝ, x < -6 ∨ x > 4) :=
by sorry

end x_range_l3594_359412


namespace average_after_removal_l3594_359465

theorem average_after_removal (numbers : Finset ℝ) (sum : ℝ) : 
  Finset.card numbers = 12 →
  sum = Finset.sum numbers id →
  sum / 12 = 90 →
  65 ∈ numbers →
  75 ∈ numbers →
  85 ∈ numbers →
  (sum - 65 - 75 - 85) / 9 = 95 :=
by sorry

end average_after_removal_l3594_359465


namespace compute_expression_l3594_359400

theorem compute_expression : 7^2 - 2*(5) + 2^3 = 47 := by
  sorry

end compute_expression_l3594_359400


namespace num_tangent_lines_specific_case_l3594_359421

/-- Two circles are internally tangent if the distance between their centers
    equals the absolute difference of their radii. -/
def internally_tangent (r₁ r₂ d : ℝ) : Prop :=
  d = |r₁ - r₂|

/-- The number of common tangent lines for two internally tangent circles is 1. -/
def num_common_tangents_internal : ℕ := 1

/-- Theorem: For two circles with radii 4 and 5, and distance between centers 3,
    the number of lines simultaneously tangent to both circles is 1. -/
theorem num_tangent_lines_specific_case :
  let r₁ : ℝ := 4
  let r₂ : ℝ := 5
  let d : ℝ := 3
  internally_tangent r₁ r₂ d →
  (num_common_tangents_internal : ℕ) = 1 :=
by
  sorry


end num_tangent_lines_specific_case_l3594_359421


namespace mass_of_man_is_90kg_l3594_359462

/-- The mass of a man who causes a boat to sink by a certain amount. -/
def mass_of_man (boat_length boat_width boat_sink_depth water_density : ℝ) : ℝ :=
  boat_length * boat_width * boat_sink_depth * water_density

/-- Theorem stating that the mass of the man is 90 kg under given conditions. -/
theorem mass_of_man_is_90kg :
  let boat_length : ℝ := 3
  let boat_width : ℝ := 2
  let boat_sink_depth : ℝ := 0.015  -- 1.5 cm in meters
  let water_density : ℝ := 1000     -- kg/m³
  mass_of_man boat_length boat_width boat_sink_depth water_density = 90 := by
sorry

#eval mass_of_man 3 2 0.015 1000  -- Should evaluate to 90

end mass_of_man_is_90kg_l3594_359462


namespace set_inclusion_implies_a_value_l3594_359448

theorem set_inclusion_implies_a_value (a : ℝ) :
  let A : Set ℝ := {x | |x| = 1}
  let B : Set ℝ := {x | a * x = 1}
  A ⊇ B →
  a = -1 ∨ a = 0 ∨ a = 1 :=
by sorry

end set_inclusion_implies_a_value_l3594_359448


namespace expression_evaluation_l3594_359473

theorem expression_evaluation : (3^2 - 3) - 2 * (4^2 - 4) + (5^2 - 5) = 2 := by
  sorry

end expression_evaluation_l3594_359473


namespace inequality_a_l3594_359486

theorem inequality_a (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^4 * y^2 * z + y^4 * x^2 * z + y^4 * z^2 * x + z^4 * y^2 * x + x^4 * z^2 * y + z^4 * x^2 * y ≥
  2 * (x^3 * y^2 * z^2 + x^2 * y^3 * z^2 + x^2 * y^2 * z^3) := by
sorry


end inequality_a_l3594_359486


namespace sqrt_f_squared_2009_l3594_359498

-- Define the function f with the given property
axiom f : ℝ → ℝ
axiom f_property : ∀ a b : ℝ, f (a * f b) = a * b

-- State the theorem to be proved
theorem sqrt_f_squared_2009 : Real.sqrt (f 2009 ^ 2) = 2009 := by
  sorry

end sqrt_f_squared_2009_l3594_359498


namespace child_tickets_sold_l3594_359402

/-- Proves the number of child's tickets sold in a movie theater -/
theorem child_tickets_sold (adult_price child_price total_tickets total_revenue : ℕ) 
  (h1 : adult_price = 7)
  (h2 : child_price = 4)
  (h3 : total_tickets = 900)
  (h4 : total_revenue = 5100) :
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_price * adult_tickets + child_price * child_tickets = total_revenue ∧
    child_tickets = 400 := by
  sorry

end child_tickets_sold_l3594_359402


namespace investment_problem_l3594_359458

/-- Calculates an investor's share of the profit based on their investment, duration, and the total profit --/
def calculate_share (investment : ℕ) (duration : ℕ) (total_investment_time : ℕ) (total_profit : ℕ) : ℚ :=
  (investment * duration : ℚ) / total_investment_time * total_profit

/-- Represents the investment problem with four investors --/
theorem investment_problem (tom_investment : ℕ) (jose_investment : ℕ) (anil_investment : ℕ) (maya_investment : ℕ)
  (tom_duration : ℕ) (jose_duration : ℕ) (anil_duration : ℕ) (maya_duration : ℕ) (total_profit : ℕ) :
  tom_investment = 30000 →
  jose_investment = 45000 →
  anil_investment = 50000 →
  maya_investment = 70000 →
  tom_duration = 12 →
  jose_duration = 10 →
  anil_duration = 7 →
  maya_duration = 1 →
  total_profit = 108000 →
  let total_investment_time := tom_investment * tom_duration + jose_investment * jose_duration +
                               anil_investment * anil_duration + maya_investment * maya_duration
  abs (calculate_share jose_investment jose_duration total_investment_time total_profit - 39512.20) < 0.01 :=
by sorry

end investment_problem_l3594_359458


namespace sum_of_digits_of_multiple_of_five_l3594_359447

/-- Given two natural numbers, returns true if they have the same digits in any order -/
def sameDigits (a b : ℕ) : Prop := sorry

/-- Returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_multiple_of_five (a b : ℕ) :
  sameDigits a b → sumOfDigits (5 * a) = sumOfDigits (5 * b) := by sorry

end sum_of_digits_of_multiple_of_five_l3594_359447


namespace test_score_probability_and_expectation_l3594_359488

-- Define the scoring system
def score_correct : ℕ := 5
def score_incorrect : ℕ := 0

-- Define the total number of questions and correct answers
def total_questions : ℕ := 10
def correct_answers : ℕ := 6

-- Define the probabilities for the remaining questions
def prob_two_eliminated : ℚ := 1/2
def prob_one_eliminated : ℚ := 1/3
def prob_guessed : ℚ := 1/4

-- Define the score distribution
def score_distribution : List (ℕ × ℚ) := [
  (30, 1/8),
  (35, 17/48),
  (40, 17/48),
  (45, 7/48),
  (50, 1/48)
]

-- Theorem statement
theorem test_score_probability_and_expectation :
  (List.lookup 45 score_distribution = some (7/48)) ∧
  (List.foldl (λ acc (score, prob) => acc + score * prob) 0 score_distribution = 455/12) := by
  sorry

end test_score_probability_and_expectation_l3594_359488


namespace even_increasing_function_properties_l3594_359405

/-- A function f: ℝ → ℝ that is even and increasing on (-∞, 0) -/
def EvenIncreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, x < y ∧ y ≤ 0 → f x < f y)

/-- Theorem stating properties of an even increasing function -/
theorem even_increasing_function_properties (f : ℝ → ℝ) 
  (hf : EvenIncreasingFunction f) : 
  (∀ x, f (-x) - f x = 0) ∧ 
  (∀ x y, 0 < x ∧ x < y → f y < f x) :=
by sorry

end even_increasing_function_properties_l3594_359405


namespace g_function_equality_l3594_359429

theorem g_function_equality (x : ℝ) :
  let g : ℝ → ℝ := λ x => -4*x^5 + 4*x^3 - 4*x + 6
  4*x^5 + 3*x^3 + x - 2 + g x = 7*x^3 - 5*x + 4 :=
by
  sorry

end g_function_equality_l3594_359429
